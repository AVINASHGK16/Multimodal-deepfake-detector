import cv2
import numpy as np
from mtcnn import MTCNN

# 🔥 Initialize ONCE
detector = MTCNN()


def apply_compression_augmentation(face, quality=None):
    """
    Simulate JPEG compression like social media recompression.
    Forces the model to detect manipulation from semantic features
    rather than pristine GAN artifacts that disappear after compression.
    quality=None → random between 30-95 (training)
    quality=95   → near-lossless (inference, no augmentation)
    """
    if quality is None:
        quality = np.random.randint(30, 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    face_uint8 = (face * 255).astype(np.uint8)
    _, encoded  = cv2.imencode('.jpg', face_uint8, encode_param)
    decoded     = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded.astype(np.float32) / 255.0


def augment_face(face):
    """
    Full augmentation pipeline applied during training only.
    Never call this during inference or precomputation.
    """
    # 1. Horizontal flip
    if np.random.rand() > 0.5:
        face = face[:, ::-1, :]

    # 2. Brightness + contrast jitter
    delta = np.random.uniform(-0.1, 0.1)
    face  = np.clip(face + delta, 0.0, 1.0)

    # 3. Random zoom (crop + resize) — simulates different camera distances
    if np.random.rand() > 0.5:
        margin = int(299 * 0.1)   # 10% margin
        x1 = np.random.randint(0, margin)
        y1 = np.random.randint(0, margin)
        x2 = 299 - np.random.randint(0, margin)
        y2 = 299 - np.random.randint(0, margin)
        face = face[y1:y2, x1:x2, :]
        face = cv2.resize(face, (299, 299))

    # 4. Compression augmentation — most important for real-world robustness
    #    Applied 50% of the time with random quality
    if np.random.rand() > 0.5:
        face = apply_compression_augmentation(face, quality=None)

    return face


def extract_face_pipeline(video_path, max_frames=2, frame_skip=5):
    cap        = cv2.VideoCapture(video_path)
    faces      = []
    frame_count = 0

    while cap.isOpened() and len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_frame)

        if len(detections) > 0:
            x, y, w, h = detections[0]['box']

            # Fix negative coordinates from MTCNN
            x, y = max(0, x), max(0, y)

            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (299, 299))
            face = face.astype("float32") / 255.0

            faces.append(face)

    cap.release()
    return faces
