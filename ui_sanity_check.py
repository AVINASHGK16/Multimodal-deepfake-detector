import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from preprocess_audio import get_mel_spectrogram
from preprocess_video import extract_face_pipeline


WEIGHTS_FILE = "best_model.h5"
THRESHOLD_FILE = Path("model_threshold.json")
DEFAULT_THRESHOLD = 0.5


def load_threshold():
    if THRESHOLD_FILE.exists():
        try:
            payload = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
            return float(np.clip(float(payload.get("best_threshold", DEFAULT_THRESHOLD)), 0.0, 1.0))
        except Exception:
            pass
    return DEFAULT_THRESHOLD


def extract_fallback_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, (299, 299)).astype(np.float32) / 255.0
    finally:
        cap.release()
    return None


def preprocess_like_ui(video_path, strict_mode=False):
    faces = extract_face_pipeline(str(video_path), max_frames=2, frame_skip=5)
    if len(faces) > 0:
        visual = np.mean(np.array(faces, dtype=np.float32), axis=0)
    else:
        if strict_mode:
            return None, None
        visual = extract_fallback_frame(video_path)
        if visual is None:
            return None, None

    audio = get_mel_spectrogram(str(video_path))
    if audio is None:
        if strict_mode:
            return None, None
        audio = np.full((148, 128, 1), -80.0, dtype=np.float32)

    audio_2d = np.squeeze(audio)
    if audio_2d.shape[1] < 128:
        audio_2d = np.pad(audio_2d, ((0, 0), (0, 128 - audio_2d.shape[1])), mode="constant")
    else:
        audio_2d = audio_2d[:, :128]
    audio = np.expand_dims(audio_2d, axis=-1).astype(np.float32)

    return visual.astype(np.float32), audio


def collect_videos(folder):
    if not folder.exists():
        return []
    exts = {".mp4", ".mov", ".avi"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="Sanity-check UI inference on known real/fake videos.")
    parser.add_argument("--real-dir", type=Path, required=True, help="Folder containing known real videos.")
    parser.add_argument("--fake-dir", type=Path, required=True, help="Folder containing known fake videos.")
    parser.add_argument("--strict", action="store_true", help="Require successful face+audio extraction.")
    args = parser.parse_args()

    threshold = load_threshold()
    model = tf.keras.models.load_model(WEIGHTS_FILE, compile=False)

    real_videos = collect_videos(args.real_dir)
    fake_videos = collect_videos(args.fake_dir)
    videos = [(p, 0) for p in real_videos] + [(p, 1) for p in fake_videos]
    if not videos:
        raise SystemExit("No videos found in provided folders.")

    tp = tn = fp = fn = skipped = 0
    print(f"Using threshold={threshold:.2f}, strict_mode={args.strict}")
    for video_path, label in videos:
        visual, audio = preprocess_like_ui(video_path, strict_mode=args.strict)
        if visual is None or audio is None:
            skipped += 1
            print(f"SKIP {video_path.name}")
            continue

        preds = model.predict(
            [np.expand_dims(visual, axis=0), np.expand_dims(audio, axis=0)],
            verbose=0,
        )
        fused_fake_prob = float(preds[2].flatten()[0])
        pred_label = 1 if fused_fake_prob >= threshold else 0

        if pred_label == 1 and label == 1:
            tp += 1
        elif pred_label == 0 and label == 0:
            tn += 1
        elif pred_label == 1 and label == 0:
            fp += 1
        else:
            fn += 1

        print(
            f"{video_path.name}: actual={'fake' if label else 'real'} "
            f"pred={'fake' if pred_label else 'real'} "
            f"fused_fake_prob={fused_fake_prob:.3f}"
        )

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    print("\n=== UI Pipeline Sanity Check ===")
    print(f"Total evaluated: {total}, skipped: {skipped}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
