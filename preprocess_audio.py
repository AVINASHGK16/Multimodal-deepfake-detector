import librosa
import numpy as np
import os
from moviepy import VideoFileClip

def get_mel_spectrogram(video_path, sr=16000, n_mels=128, max_time_steps=100):
    temp_audio_path = "temp_audio.wav"
    try:
        clip = VideoFileClip(video_path)
        
        # --- THE FIX: Handle missing audio gracefully ---
        if clip.audio is None:
            print(f"⚠️ No audio in {video_path}. Generating 'silent' spectrogram for fusion.")
            clip.close()
            # Return an array of -80.0 (silence) in the exact shape the model expects
            return np.full((n_mels, max_time_steps, 1), -80.0)
            
        clip.audio.write_audiofile(temp_audio_path, logger=None)
        clip.close()

        y, _ = librosa.load(temp_audio_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        if S_dB.shape[1] < max_time_steps:
            pad_width = max_time_steps - S_dB.shape[1]
            S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
        else:
            S_dB = S_dB[:, :max_time_steps]
            
        S_dB = np.expand_dims(S_dB, axis=-1)
        return S_dB
        
    except Exception as e:
        print(f"Audio extraction failed for {video_path}: {e}")
        return None
        
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)