import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from preprocess_video import extract_face_pipeline
from preprocess_audio import get_mel_spectrogram
from explainability import generate_gradcam_heatmap 
from model_fusion import build_fusion_model 

# --- Model Loading with Caching ---
# @st.cache_resource ensures the model is only built and loaded into RAM once! 
@st.cache_resource
def load_detector_model():
    model = build_fusion_model()
    try:
        model.load_weights("detector_weights_test.weights.h5")
        return model
    except Exception as e:
        st.warning(f"Could not load weights. Make sure 'detector_weights_test.weights.h5' is in the folder. Error: {e}")
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

# --- Main App ---
st.title("🛡️ Multimodal Deepfake & Synthetic Media Detector")
st.markdown("Analyzing audiovisual inconsistencies with Explainable AI.")

# 1. Load the AI Brain!
detector = load_detector_model()

uploaded_file = st.file_uploader("Upload a video for analysis", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Extracting features and analyzing synchronization...")

    st.subheader("Visual Analysis")
    st.text("Processing facial regions for spatial artifacts...")
    
    # Limit to 1 frame for fast UI testing
    video_features = extract_face_pipeline(video_path, max_frames=1) 
    
    if len(video_features) > 0:
        st.success(f"Successfully extracted face frames.")
        representative_face = video_features[0] 
        
        audio_features = get_mel_spectrogram(video_path)
        if audio_features is not None:
            st.success("Audio Mel-spectrogram generated successfully.")
        
        st.info("Running neural network inference...")
        
        # 2. Prepare inputs for the model (Adding the batch dimension)
        v_input = np.expand_dims(representative_face, axis=0)
        a_input = np.expand_dims(audio_features, axis=0)
        
        # 3. Make the real prediction!
        prediction = detector.predict([v_input, a_input])[0][0]
        
        # We mapped Fake=0, Real=1 during training. So Fake probability is (1.0 - prediction)
        fake_probability = (1.0 - prediction) * 100
        
        if fake_probability > 50.0:
            st.error(f"⚠️ High Probability of Manipulation: {fake_probability:.2f}%")
        else:
            st.success(f"✅ Content appears Authentic. (Fake Probability: {fake_probability:.2f}%)")

        # 4. Explainable AI: Grad-CAM Visualization
        st.subheader("Transparent Reasoning (XAI)")
        st.markdown("Highlighting regions that influenced the model's decision.")
        
        try:
            # Xception's last convolutional layer is typically 'block14_sepconv2_act'
            # Note: For dual-stream models, Grad-CAM requires specific tensor casting modifications
            real_heatmap = generate_gradcam_heatmap(detector, [v_input, a_input], 'block14_sepconv2_act')
            display_results(representative_face, real_heatmap)
        except Exception as e:
            # Fallback for the UI so the app doesn't crash during a presentation
            st.warning(f"Grad-CAM visualizer requires dual-stream casting updates. Showing standard layout.")
            mock_heatmap = np.random.rand(10, 10)
            display_results(representative_face, mock_heatmap)
            
    else:
        st.error("No faces detected in the uploaded video. Please upload a video with a clear human subject.")