import os
import numpy as np
import tensorflow as tf
from preprocess_video import extract_face_pipeline
from preprocess_audio import get_mel_spectrogram
from model_fusion import build_fusion_model

def load_mini_batch(dataset_path, sample_size=2):
    """
    Loads a tiny subset of videos to test the training pipeline.
    """
    video_data = []
    audio_data = []
    labels = []
    
    # --- FIXED: Updated to match the Kaggle folder names exactly ---
    categories = {"ffpp_fake": 0, "ffpp_real": 1}
    
    for category, label in categories.items():
        folder_path = os.path.join(dataset_path, category)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"⚠️ Missing folder: {folder_path}. Please check your dataset structure.")
            continue
            
        # Get a few video files
        videos = [f for f in os.listdir(folder_path) if f.endswith('.mp4')][:sample_size]
        
        for video_name in videos:
            video_path = os.path.join(folder_path, video_name)
            print(f"Processing: {video_name} (Label: {category})")
            
            # 1. Extract Visuals (Just grabbing the first valid face frame for testing)
            faces = extract_face_pipeline(video_path, max_frames=1)
            # 2. Extract Audio
            audio = get_mel_spectrogram(video_path)
            
            # Ensure both extractions succeeded before adding to our training batch
            if len(faces) > 0 and audio is not None:
                video_data.append(faces[0]) # Shape: (299, 299, 3)
                audio_data.append(audio)    # Shape: (128, 100, 1)
                labels.append(label)
                
    return np.array(video_data), np.array(audio_data), np.array(labels)

def test_training_loop():
    print("🚀 Initializing Multimodal Training Pipeline...")
    
    # --- FIXED: Updated the path to point inside the extracted folder ---
    dataset_folder = "./deepfake_dataset/face++dataset" 
    
    v_data, a_data, y_labels = load_mini_batch(dataset_folder, sample_size=2)
    
    if len(v_data) == 0:
        print("❌ No data loaded. Check your folder structure.")
        return

    print(f"✅ Data Loaded! Training on {len(y_labels)} samples.")
    
    # Build the Brain
    model = build_fusion_model()
    
    # Train the model for 1 test epoch
    print("🧠 Starting 1 Test Epoch...")
    history = model.fit(
        x=[v_data, a_data], 
        y=y_labels, 
        batch_size=2, 
        epochs=1
    )
    
    # Save the model weights so your Streamlit app can use them!
    model.save_weights("detector_weights_test.weights.h5")
    print("🎉 Test complete! Weights saved as 'detector_weights_test.h5'")

if __name__ == "__main__":
    test_training_loop()