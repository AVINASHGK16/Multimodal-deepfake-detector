import librosa
import numpy as np
import os
import uuid
from moviepy.editor import VideoFileClip


def augment_spectrogram(spec):
    """
    Augmentation applied to the mel+MFCC spectrogram during training.
    Improves robustness to real-world audio conditions like
    background noise, microphone differences, and codec compression.

    spec shape: (148, 128, 1)
    """
    spec = spec.copy()

    # 1. Frequency masking — randomly zero out a band of mel/MFCC rows
    #    Simulates missing frequency content from lossy audio codecs
    if np.random.rand() > 0.5:
        f_start = np.random.randint(0, 120)
        f_width = np.random.randint(5, 20)
        spec[f_start:f_start + f_width, :, :] = -80.0

    # 2. Time masking — randomly zero out a time segment
    #    Simulates network dropouts or packet loss in real-world audio
    if np.random.rand() > 0.5:
        t_start = np.random.randint(0, 100)
        t_width = np.random.randint(5, 20)
        spec[:, t_start:t_start + t_width, :] = -80.0

    # 3. Gaussian noise — simulates microphone/background noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 1.5, spec.shape)
        spec  = spec + noise

    return spec


def get_mel_spectrogram(video_path, sr=8000, n_mels=128, n_mfcc=20, max_time_steps=128):
    """
    Extracts Mel-spectrograms AND MFCCs, stacking them into a single tensor.
    Gracefully handles silent videos by returning a padded -80dB tensor.
    """
    temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
    try:
        clip = VideoFileClip(video_path)

        # Handle missing audio gracefully
        if clip.audio is None:
            print(f"⚠️ No audio in {video_path}. Generating silent spectrogram.")
            clip.close()
            return np.full((n_mels + n_mfcc, max_time_steps, 1), -80.0)

        clip.audio.write_audiofile(temp_audio_path, logger=None)
        clip.close()

        y, _ = librosa.load(temp_audio_path, sr=sr)

        # 1. Mel-spectrogram (128 features)
        S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 2. MFCCs (20 features for vocal tract anomalies)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # 3. Stack vertically → (148, time_steps)
        combined_features = np.vstack((S_dB, mfccs))

        # 4. Pad or truncate to max_time_steps
        if combined_features.shape[1] < max_time_steps:
            pad_width = max_time_steps - combined_features.shape[1]
            combined_features = np.pad(
                combined_features,
                pad_width=((0, 0), (0, pad_width)),
                mode='constant',
                constant_values=-80.0
            )
        else:
            combined_features = combined_features[:, :max_time_steps]

        # 5. Add channel dimension → (148, 128, 1)
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
