"""
Run this ONCE before training to precompute audio features
from VoxCeleb2 videos stored in a flat folder.
Output goes into the same processed_audio/ directory
as FakeAVCeleb audio.
"""
import os
import numpy as np
from preprocess_audio import get_mel_spectrogram

# ── Path to your flat VoxCeleb2 downloads folder ───────────
VOX_ROOT = r"C:\Projects\VoxSample\downloads"
SAVE_DIR = "processed_audio"

os.makedirs(SAVE_DIR, exist_ok=True)

all_files = [f for f in os.listdir(VOX_ROOT) if f.endswith(".mp4")]
total     = len(all_files)
saved     = 0
skipped   = 0

print(f"📂 Found {total} VoxCeleb2 videos to process...")

for filename in all_files:
    full_path = os.path.join(VOX_ROOT, filename)
    save_path = os.path.join(SAVE_DIR, filename.replace(".mp4", ".npy"))

    if os.path.exists(save_path):
        continue   # already processed

    try:
        audio = get_mel_spectrogram(full_path)

        if audio is None:
            skipped += 1
            print(f"❌ Audio failed: {filename}")
            continue

        audio = np.squeeze(audio)

        # Normalise to 128 time steps
        target_time_steps = 128
        if audio.shape[1] < target_time_steps:
            pad   = target_time_steps - audio.shape[1]
            audio = np.pad(audio, ((0, 0), (0, pad)))
        else:
            audio = audio[:, :target_time_steps]

        audio = np.expand_dims(audio, axis=-1)
        np.save(save_path, audio)
        saved += 1

    except Exception as e:
        skipped += 1
        print(f"❌ Error: {filename} — {e}")

    if saved % 100 == 0 and saved > 0:
        print(f"✅ Saved {saved} / {total} audio files...")

print("\n==== DONE ====")
print(f"Total:   {total}")
print(f"Saved:   {saved}")
print(f"Skipped: {skipped}")
