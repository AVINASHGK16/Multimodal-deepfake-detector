import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from model_fusion import build_fusion_model, recompile_for_phase2
from preprocess_video import augment_face
from preprocess_audio import augment_spectrogram


# ──────────────────────────────────────────────────────────────
# Identity extraction
# ──────────────────────────────────────────────────────────────
def extract_identity(p):
    match = re.search(r'(id\d+)', p)
    if match:
        return match.group(1)
    else:
        return f'unknown_{p}'


# ──────────────────────────────────────────────────────────────
# DataGenerator
# ──────────────────────────────────────────────────────────────
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=16, augment=False):
        self.df         = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.augment    = augment

        self.real_df = df[df['label'] == 0].reset_index(drop=True)
        self.fake_df = df[df['label'] == 1].reset_index(drop=True)

        if len(self.real_df) == 0 or len(self.fake_df) == 0:
            raise ValueError(
                "DataGenerator requires both real and fake samples. "
                f"Got real={len(self.real_df)}, fake={len(self.fake_df)}"
            )

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.real_df = self.real_df.sample(frac=1).reset_index(drop=True)
        self.fake_df = self.fake_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        half      = self.batch_size // 2
        retries   = 0
        max_retry = 3

        while retries < max_retry:
            real_batch = self.real_df.sample(half, replace=len(self.real_df) < half)
            fake_batch = self.fake_df.sample(half, replace=len(self.fake_df) < half)
            batch_df   = pd.concat([real_batch, fake_batch]).sample(frac=1)

            v_batch, a_batch, y_batch = [], [], []

            for _, row in batch_df.iterrows():
                label    = row['label']
                filename = os.path.basename(row['full_path'])

                face_path  = os.path.join(
                    "processed_faces", filename.replace(".mp4", ".npy")
                )
                audio_path = os.path.join(
                    "processed_audio", filename.replace(".mp4", ".npy")
                )

                if not os.path.exists(face_path) or not os.path.exists(audio_path):
                    continue

                try:
                    v_input = np.load(face_path)
                    audio   = np.load(audio_path)
                except Exception:
                    continue

                audio = np.squeeze(audio)
                if audio.ndim == 2:
                    audio = np.expand_dims(audio, axis=-1)
                if audio.shape != (148, 128, 1):
                    continue

                if self.augment:
                    v_input = augment_face(v_input)
                    audio   = augment_spectrogram(audio)

                v_batch.append(v_input)
                a_batch.append(audio)
                y_batch.append(label)

            if len(v_batch) >= 2:
                break
            retries += 1

        if len(v_batch) == 0:
            return self.__getitem__(0)

        y       = np.array(y_batch, dtype=np.float32)
        weights = np.where(y == 0, 2.0, 1.0)

        return (
            [np.array(v_batch, dtype=np.float32),
             np.array(a_batch, dtype=np.float32)],
            [y, y, y],
            [weights, weights, weights],
        )


# ──────────────────────────────────────────────────────────────
# Load FakeAVCeleb
# ──────────────────────────────────────────────────────────────
def load_fakeavceleb(video_map):
    df = pd.read_csv('FakeAVCeleb/FakeAVCeleb/meta_data.csv')
    df['label']     = df['type'].apply(lambda x: 1 if 'Fake' in x else 0)
    df['full_path'] = df['path'].map(video_map)
    df['source']    = 'fakeavceleb'
    df = df.dropna(subset=['full_path']).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# Load VoxCeleb2 — flat folder structure
# Filenames: id00018_ewCKSXWitUk_143-151.mp4
# ──────────────────────────────────────────────────────────────
def load_voxceleb2(vox_root, max_videos=4067):
    rows = []
    for filename in os.listdir(vox_root):
        if not filename.endswith(".mp4"):
            continue
        full_path = os.path.join(vox_root, filename)
        rows.append({
            'path':      filename,
            'full_path': full_path,
            'type':      'RealVideo-RealAudio',
            'label':     0,
            'source':    'voxceleb2',
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("⚠️  No VoxCeleb2 videos found — check VOX_ROOT path")
        return df

    if len(df) > max_videos:
        df = df.sample(max_videos, random_state=42)

    print(f"✅ VoxCeleb2 real videos loaded: {len(df)}")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
def run_training():
    print("🚀 Initialising training pipeline...")

    # ── Paths ────────────────────────────────────────────────
    # Update VOX_ROOT once you move the folder into your project
    VOX_ROOT   = r"C:\Projects\VoxSample\downloads"
    VIDEO_ROOT = "FakeAVCeleb"

    # ── Index FakeAVCeleb videos ─────────────────────────────
    print("📂 Indexing FakeAVCeleb video files...")
    video_map = {}
    for root, dirs, files in os.walk(VIDEO_ROOT):
        for file in files:
            if file.endswith(".mp4"):
                video_map[file] = os.path.join(root, file)
    print(f"✅ FakeAVCeleb videos found: {len(video_map)}")

    # ── Load both datasets ───────────────────────────────────
    df_fav = load_fakeavceleb(video_map)
    df_vox = load_voxceleb2(VOX_ROOT, max_videos=2500)

    df = pd.concat([df_fav, df_vox], ignore_index=True)

    real_total = len(df[df['label'] == 0])
    fake_total = len(df[df['label'] == 1])
    print(f"✅ Merged dataset: {len(df)} samples (real={real_total}, fake={fake_total})")

    # ── Balance: all real + 4× fake ─────────────────────────
    real_df      = df[df['label'] == 0]
    fake_df      = df[df['label'] == 1]
    max_fake     = len(real_df) * 4
    fake_sampled = fake_df.sample(
        min(len(fake_df), max_fake), random_state=42
    )

    df_balanced = pd.concat([
        real_df,
        fake_sampled,
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    real_count = len(df_balanced[df_balanced['label'] == 0])
    fake_count = len(df_balanced[df_balanced['label'] == 1])
    print(f"✅ Balanced dataset: {len(df_balanced)} samples (real={real_count}, fake={fake_count})")

    # ── Identity-level split ─────────────────────────────────
    df_balanced['identity'] = df_balanced['path'].apply(extract_identity)

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, val_idx = next(
        gss.split(df_balanced, groups=df_balanced['identity'])
    )

    train_df = df_balanced.iloc[train_idx].reset_index(drop=True)
    val_df   = df_balanced.iloc[val_idx].reset_index(drop=True)

    overlap = len(set(train_df['identity']) & set(val_df['identity']))
    print(f"✅ Identity overlap (should be 0): {overlap}")
    print(f"✅ Train: {len(train_df)}  Val: {len(val_df)}")
    print(f"✅ Val real: {len(val_df[val_df['label']==0])}  Val fake: {len(val_df[val_df['label']==1])}")

    # ── Generators ───────────────────────────────────────────
    train_gen = DataGenerator(train_df, batch_size=16, augment=True)
    val_gen   = DataGenerator(val_df,   batch_size=16, augment=False)

    # ── Callbacks ────────────────────────────────────────────
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor='val_fused_pred_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        'training_log.csv', append=False
    )

    # ════════════════════════════════════════════════════════
    # PHASE 1 — freeze Xception, train fusion + audio
    # ════════════════════════════════════════════════════════
    print("\n🔥 Phase 1: training fusion head + audio branch (Xception frozen)...")
    model = build_fusion_model(freeze_xception=True)

    early_stop_p1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        train_gen,
        steps_per_epoch=100,
        validation_data=val_gen,
        validation_steps=25,
        epochs=8,
        callbacks=[early_stop_p1, checkpoint, lr_scheduler, csv_logger],
    )

    # ════════════════════════════════════════════════════════
    # PHASE 2 — unfreeze all, end-to-end fine-tuning
    # ════════════════════════════════════════════════════════
    print("\n🔥 Phase 2: end-to-end fine-tuning (all layers unfrozen)...")

    model = recompile_for_phase2(model)

    early_stop_p2 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        train_gen,
        steps_per_epoch=100,
        validation_data=val_gen,
        validation_steps=25,
        epochs=15,
        callbacks=[early_stop_p2, checkpoint, lr_scheduler, csv_logger],
    )

    print("✅ Training complete! Best model saved to best_model.h5")


if __name__ == "__main__":
    run_training()
