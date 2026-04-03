import cv2
import numpy as np
from mtcnn import MTCNN

# Initialize the detector once at the top of the file so it doesn't 
# slow down your app by rebuilding the network for every single frame.
detector = MTCNN()

def extract_face_pipeline(video_path, max_frames=1):
    """
    Phase 2 Upgrade: Uses MTCNN for highly robust face detection 
    across various angles, poses, and lighting conditions.
    """
    cap = cv2.VideoCapture(video_path)
    extracted_faces = []
    frame_count = 0
    
    while cap.isOpened() and len(extracted_faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # MTCNN mathematically requires RGB images, not OpenCV's default BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Detect faces using the neural network
        results = detector.detect_faces(rgb_frame)
        
        if results:
            # 2. Extract the bounding box of the most prominent face
            # MTCNN returns a dictionary. 'box' gives [x, y, width, height]
            x, y, w, h = results[0]['box']
            
            # Ensure coordinates don't go out of frame bounds (prevents slicing crashes)
            x, y = max(0, x), max(0, y)
            
            # 3. Crop the face from the frame
            face_crop = rgb_frame[y:y+h, x:x+w]
            
            # Safety check to ensure the crop didn't fail
            if face_crop.size == 0:
                continue
                
            # 4. Resize to match the Xception network's required input (299x299)
            face_resized = cv2.resize(face_crop, (299, 299))
            
            # 5. Normalize pixel values to [0, 1] for stable neural network inference
            face_normalized = face_resized / 255.0
            
            extracted_faces.append(face_normalized)
            
    cap.release()
    return extracted_faces