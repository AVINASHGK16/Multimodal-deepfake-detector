"""
Run this ONCE before training to precompute face crops
from VoxCeleb2 videos stored in a flat folder.
Includes a per-video timeout to prevent MTCNN from hanging.
"""
import os
import multiprocessing
import numpy as np


def process_video(args):
    """Runs in a separate process so we can kill it on timeout."""
    full_path, save_path = args
    try:
        from preprocess_video import extract_face_pipeline
        import numpy as np

        faces = extract_face_pipeline(full_path, max_frames=2, frame_skip=5)

        if len(faces) > 0:
            v_input = np.mean(np.array(faces), axis=0)
            np.save(save_path, v_input)
            return "saved"
        else:
            return "no_face"
    except Exception as e:
        return f"error: {e}"


if __name__ == "__main__":

    # ── Config ───────────────────────────────────────────────
    VOX_ROOT    = r"C:\Projects\VoxSample\downloads" # Path to downloaded videos from VoxCeleb2
    SAVE_DIR    = "processed_faces"
    TIMEOUT_SEC = 15

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = sorted([f for f in os.listdir(VOX_ROOT) if f.endswith(".mp4")])
    total     = len(all_files)
    saved     = 0
    skipped   = 0

    print(f"📂 Found {total} VoxCeleb2 videos to process...")

    for filename in all_files:
        full_path = os.path.join(VOX_ROOT, filename)
        save_path = os.path.join(SAVE_DIR, filename.replace(".mp4", ".npy"))

        if os.path.exists(save_path):
            continue   # already processed

        ctx  = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=1)

        try:
            result  = pool.apply_async(process_video, args=((full_path, save_path),))
            outcome = result.get(timeout=TIMEOUT_SEC)

            if outcome == "saved":
                saved += 1
            elif outcome == "no_face":
                skipped += 1
                print(f"⚠️  No face: {filename}")
            else:
                skipped += 1
                print(f"❌ {filename} — {outcome}")

        except multiprocessing.TimeoutError:
            skipped += 1
            print(f"⏱️  Timeout ({TIMEOUT_SEC}s): {filename} — skipping")

        except Exception as e:
            skipped += 1
            print(f"❌ Error: {filename} — {e}")

        finally:
            pool.terminate()
            pool.join()

        processed = saved + skipped
        if processed % 100 == 0 and processed > 0:
            print(f"Progress: {processed}/{total} — saved={saved}, skipped={skipped}")

    print("\n==== DONE ====")
    print(f"Total:   {total}")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
