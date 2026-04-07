import os
import numpy as np
from preprocess_audio import get_mel_spectrogram

VIDEO_ROOT = "FakeAVCeleb"
SAVE_DIR = "processed_audio"

os.makedirs(SAVE_DIR, exist_ok=True)

total = 0
saved = 0
skipped = 0

for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.endswith(".mp4"):
            total += 1

            full_path = os.path.join(root, file)
            save_path = os.path.join(SAVE_DIR, file.replace(".mp4", ".npy"))

            if os.path.exists(save_path):
                continue

            try:
                audio = get_mel_spectrogram(full_path)

                if audio is None:
                    skipped += 1
                    print("❌ Audio failed:", file)
                    continue

                audio = np.squeeze(audio)

                # 🔥 Normalize length (IMPORTANT)
                target_time_steps = 128

                if audio.shape[1] < target_time_steps:
                    pad = target_time_steps - audio.shape[1]
                    audio = np.pad(audio, ((0, 0), (0, pad)))
                else:
                    audio = audio[:, :target_time_steps]

                audio = np.expand_dims(audio, axis=-1)

                np.save(save_path, audio)
                saved += 1

                if saved % 100 == 0:
                    print(f"Saved {saved} files...")

            except Exception as e:
                skipped += 1
                print("❌ Error:", file, e)

print("\n==== DONE ====")
print("Total:", total)
print("Saved:", saved)
print("Skipped:", skipped)