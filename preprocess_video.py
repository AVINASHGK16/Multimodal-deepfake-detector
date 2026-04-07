import cv2
import numpy as np
from mtcnn import MTCNN

# 🔥 Initialize ONCE
detector = MTCNN()

def extract_face_pipeline(video_path, max_frames=2, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    faces = []
    frame_count = 0

    while cap.isOpened() and len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = detector.detect_faces(rgb_frame)

        if len(detections) > 0:
            x, y, w, h = detections[0]['box']

            # 🛡️ Fix negative values
            x, y = max(0, x), max(0, y)

            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (299, 299))
            face = face.astype("float32") / 255.0

            faces.append(face)

    cap.release()
    return faces