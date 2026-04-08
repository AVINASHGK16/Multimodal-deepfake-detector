import cv2
import numpy as np
from mtcnn import MTCNN

# 🔥 Initialize ONCE
detector = MTCNN()

def extract_face_pipeline(video_path, max_frames=2, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    faces = []
    frame_count = 0
    first_frame = None
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while cap.isOpened() and len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame is None:
            first_frame = frame

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
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(haar_faces) > 0:
            x, y, w, h = haar_faces[0]
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face = cv2.resize(face, (299, 299))
            face = face.astype("float32") / 255.0
            faces.append(face)

    cap.release()
    if len(faces) == 0 and first_frame is not None:
        h, w = first_frame.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = first_frame[y0:y0+side, x0:x0+side]
        crop = cv2.resize(crop, (299, 299))
        crop = crop.astype("float32") / 255.0
        faces.append(crop)
    return faces