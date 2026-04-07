import librosa
import numpy as np
import os
import uuid
from moviepy.editor import VideoFileClip # Note: In moviepy 1.0.3, it's moviepy.editor. Depending on your version, keep your working import.

def get_mel_spectrogram(video_path, sr=8000, n_mels=128, n_mfcc=20, max_time_steps=128):
    """
    Extracts Mel-spectrograms AND MFCCs, stacking them into a single tensor.
    Gracefully handles silent videos by returning a padded -80dB tensor.
    """
    temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
    try:
        clip = VideoFileClip(video_path)
        
        # --- Handle missing audio gracefully ---
        if clip.audio is None:
            print(f"⚠️ No audio in {video_path}. Generating 'silent' spectrogram for fusion.")
            clip.close()
            # Return an array of -80.0 (silence) in the exact 148-height shape the model expects
            return np.full((n_mels + n_mfcc, max_time_steps, 1), -80.0)
            
        clip.audio.write_audiofile(temp_audio_path, logger=None)
        clip.close()

        # Load audio using Librosa
        y, _ = librosa.load(temp_audio_path, sr=sr)
        
        # 1. Extract standard Mel-spectrogram (128 features)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 2. Extract MFCCs (20 features for vocal tract anomalies)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # 3. Stack them vertically! Shape becomes (148, time_steps)
        combined_features = np.vstack((S_dB, mfccs))
        
        # 4. Pad or truncate to max_time_steps
        if combined_features.shape[1] < max_time_steps:
            pad_width = max_time_steps - combined_features.shape[1]
            combined_features = np.pad(combined_features, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
        else:
            combined_features = combined_features[:, :max_time_steps]
            
        # 5. Add the channel dimension for the CNN: (148, 128) -> (148, 128, 1)
        combined_features = np.expand_dims(combined_features, axis=-1)
        
        return combined_features
        
    except Exception as e:
        print(f"Audio extraction failed for {video_path}: {e}")
        return None
        
    finally:
        try:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        except:
            pass