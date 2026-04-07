import os
import numpy as np
import pandas as pd
from model_fusion import build_fusion_model
from preprocess_audio import get_mel_spectrogram
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=8):
        self.df = df
        self.batch_size = batch_size

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size]

        v_batch = []
        a_batch = []
        y_batch = []

        for _, row in batch_df.iterrows():
            video_path = row['full_path']
            label = row['label']

            # 🔥 Load precomputed face
            filename = os.path.basename(video_path)
            face_path = os.path.join("processed_faces", filename.replace(".mp4", ".npy"))

            if not os.path.exists(face_path):
                continue

            v_input = np.load(face_path)

            # 🎧 Audio
            filename = os.path.basename(video_path)
            audio_path = os.path.join("processed_audio", filename.replace(".mp4", ".npy"))

            if not os.path.exists(audio_path):
                continue

            audio = np.load(audio_path)

            # 🔥 REMOVE EXTRA DIMENSIONS
            audio = np.squeeze(audio)

            # Ensure correct shape
            if audio.ndim == 2:
                audio = np.expand_dims(audio, axis=-1)

            # 🔥 Final safety check
            if audio.shape != (148, 128, 1):
                print("❌ BAD AUDIO SHAPE:", audio.shape)
                continue

            v_batch.append(v_input)
            a_batch.append(audio)
            y_batch.append(label)

        if len(v_batch) == 0:
            return self.__getitem__((idx + 1) % len(self))

        return (
            [np.array(v_batch), np.array(a_batch)],
            [np.array(y_batch), np.array(y_batch), np.array(y_batch)]
        )

if __name__ == "__main__":
    print("🚀 Initializing Local PC Training Pipeline...")
    

    # 1. Load the Metadata
    # Make sure your friend has a CSV with columns 'filename' and 'label' (0=Real, 1=Fake)
    print("📂 Loading dataset...")
    df = pd.read_csv('FakeAVCeleb/FakeAVCeleb/meta_data.csv')
    
    # Create labels
    df['label'] = df['type'].apply(lambda x: 1 if 'Fake' in x else 0)

    print("📂 Indexing all video files from disk...")

    VIDEO_ROOT = "FakeAVCeleb"  # change if needed

    video_map = {}

    for root, dirs, files in os.walk(VIDEO_ROOT):
        for file in files:
            if file.endswith(".mp4"):
                video_map[file] = os.path.join(root, file)

    print(f"✅ Total videos found on disk: {len(video_map)}")

    print("🔗 Matching CSV with actual files...")

    # Your filename is in 'path' column
    df['full_path'] = df['path'].map(video_map)

    # Remove unmatched
    df = df.dropna(subset=['full_path']).reset_index(drop=True)

    print(f"✅ Total valid videos: {len(df)}")

    from sklearn.model_selection import GroupShuffleSplit

    # VERY IMPORTANT: group by video (path column)
    df['video_id'] = df['full_path']

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)

    train_idx, val_idx = next(gss.split(df, groups=df['video_id']))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print("🔍 Overlap check:",
      len(set(train_df['full_path']) & set(val_df['full_path'])))

    # 2. Initialize the Generator

    # 3. Build the Phase 2 Architecture
    print("🧠 Building Auxiliary Multi-Output Architecture...")
    model = build_fusion_model()

    train_gen = DataGenerator(train_df, batch_size=16)
    val_gen = DataGenerator(val_df, batch_size=16)

    print("🔥 Starting training...")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',     # what to track
        patience=3,             # wait 3 epochs before stopping
        restore_best_weights=True,  # go back to best model
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=[early_stop, checkpoint, lr_scheduler]
    )

    print("✅ Training complete!")