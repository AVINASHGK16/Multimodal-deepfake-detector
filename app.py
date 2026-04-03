import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from preprocess_video import extract_face_pipeline
from preprocess_audio import get_mel_spectrogram
from explainability import generate_gradcam_heatmap 
from model_fusion import build_fusion_model 

def apply_custom_css():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1000px;
        }
        
        .stFileUploader {
            background-color: #1E212B;
            border-radius: 10px;
            padding: 15px;
            border: 1px dashed #00F0FF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_css()

# --- Model Loading with Caching ---
@st.cache_resource
def load_detector_model():
    model = build_fusion_model()
    try:
        # POINT TO THE NEW DUMMY WEIGHTS HERE:
        model.load_weights("dummy_phase2.weights.h5") 
        return model
    except Exception as e:
        st.warning(f"Could not load weights. Error: {e}")
        return model

# --- UI Layout Function ---
def display_results(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_normalized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    if original_img.dtype != np.uint8:
        original_img = np.uint8(original_img * 255)
    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    superimposed_img = cv2.addWeighted(original_img_bgr, 0.6, heatmap_color, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original Face Scan", channels="RGB")
    with col2:
        st.image(superimposed_rgb, caption="Detection Heatmap (XAI)", channels="RGB")

# --- Main Application Layout ---
st.title("🛡️ Multimodal Deepfake & Synthetic Media Detector")
st.markdown("### Protect yourself from synthetic media with AI-driven analysis.")

detector = load_detector_model()

# Create clean navigation tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Media", "🧠 How it Works", "ℹ️ About the Project"])

with tab1:
    # The ONLY file uploader goes here!
    uploaded_file = st.file_uploader("Upload a video for analysis", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Aesthetic loading spinner
        with st.status("Analyzing audiovisual data...", expanded=True) as status:
            st.write("Extracting facial frames for spatial artifacts...")
            video_features = extract_face_pipeline(video_path, max_frames=1) 
            
            st.write("Generating Mel-spectrograms for temporal anomalies...")
            raw_audio_features = get_mel_spectrogram(video_path)
            
            if len(video_features) > 0 and raw_audio_features is not None:
                st.write("Running dual-stream neural network inference...")
                
                # --- AUDIO TENSOR FIX START ---
                # 0. Squeeze the array to ensure it is strictly 2D (128, time_steps)
                audio_2d = np.squeeze(raw_audio_features)
                
                # 1. Pad or truncate the time axis to ensure it is exactly 128
                target_time_steps = 128
                current_time_steps = audio_2d.shape[1] # Now this correctly points to the time axis
                
                if current_time_steps < target_time_steps:
                    # Pad the end with zeros (silence)
                    pad_width = target_time_steps - current_time_steps
                    audio_features = np.pad(audio_2d, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    # Truncate if it's too long
                    audio_features = audio_2d[:, :target_time_steps]
                
                # 2. Add the missing channel dimension (128, 128) -> (128, 128, 1)
                audio_features = np.expand_dims(audio_features, axis=-1)
                # --- AUDIO TENSOR FIX END ---

                # Prepare the final batch dimensions for the model
                v_input = np.expand_dims(video_features[0], axis=0)
                a_input = np.expand_dims(audio_features, axis=0)
                
                # 1. Get the predictions from our new 3-output architecture
                predictions = detector.predict([v_input, a_input])
                v_score = predictions[0][0][0]
                a_score = predictions[1][0][0]
                fused_score = predictions[2][0][0]
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                st.divider()
                
                # 2. Convert raw scores to Fake Probabilities
                v_fake_prob = (1.0 - v_score) * 100
                a_fake_prob = (1.0 - a_score) * 100
                fused_fake_prob = (1.0 - fused_score) * 100
                
                # 3. Calculate Modality Divergence (The Asymmetric Alert System)
                divergence = abs(v_fake_prob - a_fake_prob)
                
                # Display a breakdown of what the AI "saw" vs what it "heard"
                cols = st.columns(3)
                cols[0].metric("👁️ Visual Fake %", f"{v_fake_prob:.1f}%")
                cols[1].metric("👂 Audio Fake %", f"{a_fake_prob:.1f}%")
                cols[2].metric("🧠 Fused Fake %", f"{fused_fake_prob:.1f}%")
                
                st.write("") # Spacer
                
                # 4. The Final Decision Logic
                if divergence > 40.0: 
                    # If the eyes and ears disagree by more than 40%, flag it!
                    st.error(f"🚨 **ASYMMETRIC ATTACK DETECTED:** Severe modality conflict ({divergence:.1f}% divergence). Highly probable Voice Cloning over Real Video, or Vice Versa.")
                elif fused_fake_prob > 50.0:
                    st.warning(f"⚠️ **High Probability of Manipulation:** {fused_fake_prob:.2f}%")
                else:
                    st.success(f"✅ **Content appears Authentic:** {fused_fake_prob:.2f}% fake probability.")

                # 5. Transparent Reasoning (Grad-CAM)
                st.subheader("Transparent Reasoning (XAI)")
                try:
                    # Note: We pass [v_input, a_input] to the heatmap generator
                    real_heatmap = generate_gradcam_heatmap(detector, [v_input, a_input], 'block14_sepconv2_act')
                    display_results(video_features[0], real_heatmap)
                except Exception as e:
                    st.warning("Grad-CAM visualizer fallback. (Requires casting updates for multi-output)")
                    mock_heatmap = np.random.rand(10, 10)
                    display_results(video_features[0], mock_heatmap)
            else:
                status.update(label="Analysis Failed", state="error", expanded=False)
                st.error("Could not detect a clear face or audio track in the uploaded video.")
with tab2:
    st.header("The Architecture")
    st.markdown("""
    This detector uses a **Dual-Stream Architecture** to analyze media holistically:
    * **Visual Stream (Eyes):** An Xception Convolutional Neural Network scans for spatial artifacts like unnatural pixel blending.
    * **Audio Stream (Ears):** An LSTM network tracks temporal inconsistencies in voice and tone.
    * **Explainability:** Grad-CAM maps highlight the exact pixels that influenced the model's decision, ensuring transparent AI.
    """)

with tab3:
    st.header("ℹ️ About the Project")
    st.markdown("""
    **Project Name:** AI-Powered Multimodal Deepfake & Synthetic Media Detector  
    **Version:** 1.0 (Phase 1 Prototype)

    ### 🛠️ Technology Stack
    * **Deep Learning Framework:** TensorFlow & Keras
    * **Computer Vision:** OpenCV (Haar Cascades for facial extraction)
    * **Audio Processing:** Librosa & MoviePy (Mel-spectrogram generation)
    * **Frontend/UI:** Streamlit
    * **Cloud Infrastructure:** Kaggle Dual T4 GPUs (for neural network training)

    ### 🎯 Project Mission
    As Generative AI becomes more accessible, the threat of malicious synthetic media and deepfakes grows exponentially. Traditional detection systems that rely solely on visual cues are easily fooled by modern face-swapping algorithms. 
    
    This project was engineered to explore how **Multimodal Machine Learning**—teaching an AI to simultaneously "see" spatial artifacts and "hear" temporal audio anomalies—can be utilized to restore trust in digital media.

    ### 🚀 Future Roadmap
    Currently operating as a functional prototype, the immediate next steps for this capstone initiative include:
    * **Model Fine-Tuning:** Unfreezing the Xception network to train specifically on deepfake-induced pixel warping.
    * **Temporal Expansion:** Implementing `TimeDistributed` layers to analyze sequences of 10+ frames simultaneously to catch inter-frame glitches.
    * **Dataset Migration:** Transitioning to the FakeAVCeleb dataset to robustly train the audio LSTM on synthesized AI voice clones.
    """)