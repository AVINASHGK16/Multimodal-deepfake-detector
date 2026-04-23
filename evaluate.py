import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
)
from sklearn.model_selection import GroupShuffleSplit

from model_fusion import focal_loss


# ──────────────────────────────────────────────────────────────
# Identity extraction (must match training.py exactly)
# ──────────────────────────────────────────────────────────────
def extract_identity(p):
    match = re.search(r'(id\d+)', p)
    if match:
        return match.group(1)
    else:
        return f'unknown_{p}'


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
# Load VoxCeleb2 — flat folder
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
# Rebuild dataset (must match training.py exactly)
# ──────────────────────────────────────────────────────────────
print("📂 Loading dataset...")

VOX_ROOT   = r"C:\Projects\VoxSample\downloads"
VIDEO_ROOT = "FakeAVCeleb"

video_map = {}
for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.endswith(".mp4"):
            video_map[file] = os.path.join(root, file)

df_fav = load_fakeavceleb(video_map)
df_vox = load_voxceleb2(VOX_ROOT, max_videos=2500)

df = pd.concat([df_fav, df_vox], ignore_index=True)

real_df      = df[df['label'] == 0]
fake_df      = df[df['label'] == 1]
max_fake     = len(real_df) * 4
fake_sampled = fake_df.sample(min(len(fake_df), max_fake), random_state=42)

df_balanced = pd.concat([
    real_df,
    fake_sampled,
]).sample(frac=1, random_state=42).reset_index(drop=True)

real_count = len(df_balanced[df_balanced['label'] == 0])
fake_count = len(df_balanced[df_balanced['label'] == 1])
print(f"✅ Balanced dataset: {len(df_balanced)} samples (real={real_count}, fake={fake_count})")

df_balanced['identity'] = df_balanced['path'].apply(extract_identity)

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, val_idx = next(
    gss.split(df_balanced, groups=df_balanced['identity'])
)

train_df = df_balanced.iloc[train_idx].reset_index(drop=True)
val_df   = df_balanced.iloc[val_idx].reset_index(drop=True)

overlap = len(set(train_df['identity']) & set(val_df['identity']))
print(f"✅ Identity overlap (should be 0): {overlap}")
print(f"✅ Val size: {len(val_df)}  (real={len(val_df[val_df['label']==0])}, fake={len(val_df[val_df['label']==1])})")
print(f"✅ Val fake types:\n{val_df['type'].value_counts()}\n")

# ──────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────
print("🧠 Loading best model...")
fl = focal_loss(gamma=2.0, alpha=0.75)

model = tf.keras.models.load_model(
    "best_model.h5",
    custom_objects={
        'tf':      tf,
        'loss_fn': fl,
    },
)

# ──────────────────────────────────────────────────────────────
# Deterministic inference
# ──────────────────────────────────────────────────────────────
print("🚀 Running inference...")
y_true      = []
vis_probs   = []
aud_probs   = []
fused_probs = []
skipped     = 0

for _, row in val_df.iterrows():
    filename   = os.path.basename(row['full_path'])
    face_path  = os.path.join("processed_faces",  filename.replace(".mp4", ".npy"))
    audio_path = os.path.join("processed_audio", filename.replace(".mp4", ".npy"))

    if not os.path.exists(face_path) or not os.path.exists(audio_path):
        skipped += 1
        continue

    v = np.load(face_path)[np.newaxis].astype(np.float32)
    a = np.load(audio_path)
    a = np.squeeze(a)
    if a.ndim == 2:
        a = np.expand_dims(a, axis=-1)
    if a.shape != (148, 128, 1):
        skipped += 1
        continue
    a = a[np.newaxis].astype(np.float32)

    preds = model.predict([v, a], verbose=0)
    vis_probs.append(float(preds[0].flatten()[0]))
    aud_probs.append(float(preds[1].flatten()[0]))
    fused_probs.append(float(preds[2].flatten()[0]))
    y_true.append(int(row['label']))

print(f"✅ Evaluated {len(y_true)} samples, skipped {skipped}")

y_true      = np.array(y_true)
vis_probs   = np.array(vis_probs)
aud_probs   = np.array(aud_probs)
fused_probs = np.array(fused_probs)

# ──────────────────────────────────────────────────────────────
# Threshold sweep
# ──────────────────────────────────────────────────────────────
print("\n🔍 Sweeping threshold on fused output...")
best_thresh, best_f1 = 0.5, 0.0
for thresh in np.arange(0.3, 0.75, 0.05):
    preds_t = (fused_probs > thresh).astype(int)
    f1 = f1_score(y_true, preds_t)
    print(f"  thresh={thresh:.2f}  F1={f1:.4f}")
    if f1 > best_f1:
        best_f1     = f1
        best_thresh = thresh

print(f"\n✅ Best threshold: {best_thresh:.2f}  (F1={best_f1:.4f})")

# ──────────────────────────────────────────────────────────────
# Final metrics
# ──────────────────────────────────────────────────────────────
y_pred = (fused_probs > best_thresh).astype(int)

print("\n📊 Fused prediction — Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

print(f"ROC-AUC (fused):  {roc_auc_score(y_true, fused_probs):.4f}")
print(f"ROC-AUC (visual): {roc_auc_score(y_true, vis_probs):.4f}")
print(f"ROC-AUC (audio):  {roc_auc_score(y_true, aud_probs):.4f}")

# ──────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'], ax=axes[0])
axes[0].set_title(f'Confusion Matrix (thresh={best_thresh:.2f})')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

for probs, label in [
    (fused_probs, 'Fused'),
    (vis_probs,   'Visual'),
    (aud_probs,   'Audio'),
]:
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    axes[1].plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[1].set_title('ROC Curves')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

axes[2].hist(fused_probs[y_true == 0], bins=30, alpha=0.6, label='Real', color='blue')
axes[2].hist(fused_probs[y_true == 1], bins=30, alpha=0.6, label='Fake', color='red')
axes[2].axvline(best_thresh, color='black', linestyle='--', label=f'Thresh={best_thresh:.2f}')
axes[2].set_title('Fused score distribution')
axes[2].set_xlabel('Predicted probability (fake)')
axes[2].legend()

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150)
plt.show()
print("📈 Saved evaluation_results.png")
