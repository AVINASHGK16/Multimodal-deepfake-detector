import os
import numpy as np
from preprocess_video import extract_face_pipeline

VIDEO_ROOT = "FakeAVCeleb"
SAVE_DIR = "processed_faces"

os.makedirs(SAVE_DIR, exist_ok=True)

video_map = {}

# 🔍 Find all videos
for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.endswith(".mp4"):
            full_path = os.path.join(root, file)
            save_path = os.path.join(SAVE_DIR, file.replace(".mp4", ".npy"))

            if os.path.exists(save_path):
                continue  # already processed

            try:
                faces = extract_face_pipeline(
                    full_path,
                    max_frames=2,
                    frame_skip=5
                )

                if len(faces) > 0:
                    # 🔥 Save averaged face
                    v_input = np.mean(np.array(faces), axis=0)
                    np.save(save_path, v_input)
                else:
                    print("⚠️ No face:", file)

            except Exception as e:
                print("❌ Error:", file, e)