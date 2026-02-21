import os
import cv2
import numpy as np

# Load the cascade once globally
cascade_path = os.path.join(
    cv2.data.haarcascades, 
    "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(cascade_path)

def extract_face_pipeline(video_path, max_frames=20):
    """
    Extracts face regions from a video and returns them as an array.
    max_frames: Limits the number of frames to process to keep it fast.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    processed_faces = []

    while cap.isOpened() and len(processed_faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            # Crop the face
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue

            # Resize to Xception's required input size
            resized_face = cv2.resize(face_roi, (299, 299))
            
            # Convert BGR (OpenCV default) to RGB (Neural Network standard)
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            
            processed_faces.append(rgb_face)
            
            # Break after the first face is found per frame 
            # (Assuming one speaker for now to keep it simple)
            break 

    cap.release()
    
    # Return as a numpy array scaled between 0 and 1 for the neural network
    return np.array(processed_faces) / 255.0

# This block only runs if you execute `python video.py` directly.
# It WON'T run when imported by `app.py`.
if __name__ == "__main__":
    # Test it with a video of a human face!
    test_video = "human_speaking_test.mp4" 
    if os.path.exists(test_video):
        faces = extract_face_pipeline(test_video)
        print(f"Successfully extracted {len(faces)} face frames. Shape: {faces.shape}")
    else:
        print(f"Please provide a valid human test video at {test_video}")