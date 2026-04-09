# Multimodal Deepfake Detection

A deep learning system for detecting deepfake videos using both visual and audio signals. The model combines an Xception-based face analysis branch with a CNN-based audio branch, fused via a cross-modal attention gate that learns which modality to trust on a per-sample basis.

Trained and evaluated on the [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) dataset with identity-level train/val splitting to prevent data leakage.

---

## Results

Evaluated on 404 held-out samples from identities never seen during training.

| Metric | Score |
|---|---|
| Accuracy | 94% |
| ROC-AUC (fused) | 0.968 |
| ROC-AUC (visual branch) | 0.959 |
| ROC-AUC (audio branch) | 0.827 |
| Precision (real) | 0.88 |
| Precision (fake) | 0.96 |
| Recall (real) | 0.87 |
| Recall (fake) | 0.96 |
| Val set size | 404 samples (unseen identities) |

Evaluation uses a deterministic inference loop over the full held-out set with a threshold sweep (0.30–0.70) to find the optimal F1 decision boundary. Best threshold was 0.35.

---

## Architecture

```
Video frames (299×299×3)          Mel+MFCC spectrogram (148×128×1)
        │                                       │
   Xception (pretrained)              Audio CNN (4 blocks)
   last 30 layers trainable           SeparableConv2D + BN
        │                                       │
  v_features [2048]               a_features [256]
        │                                       │
        └──────────┐         ┌──────────────────┘
                   ▼         ▼
           Cross-modal attention gate
           (learns to suppress noisy modality per sample)
                       │
              Fusion head [512→256→128]
              BatchNorm + Dropout
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
       Visual pred  Audio pred  Fused pred  ← main output
```

### Key design decisions

**Cross-modal attention gate** — instead of naively concatenating visual and audio features, a learned gate vector weights each modality based on how informative it is for each sample. This prevents a weak or noisy audio signal from corrupting the fused prediction.

**Two-phase training** — Phase 1 freezes Xception and trains only the audio branch and fusion head. Phase 2 unfreezes all layers with a conservative learning rate (3e-5). This prevents the randomly initialised fusion head from corrupting Xception's pretrained weights during early training.

**Multi-output auxiliary losses** — the model outputs three predictions (visual-only, audio-only, fused) with loss weights of 1.0, 0.3, and 3.0 respectively. This forces each branch to learn independently before fusion, preventing the stronger visual branch from dominating.

---

## Dataset

[FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) — a multimodal deepfake dataset containing four video types:

| Type | Description |
|---|---|
| RealVideo-RealAudio | Genuine videos |
| FakeVideo-FakeAudio | Both face and voice synthesised |
| FakeVideo-RealAudio | Face swapped, original audio |
| RealVideo-FakeAudio | Original face, voice cloned |

The dataset has a ~42:1 fake-to-real imbalance. This is handled via a 3:1 sampling ratio (all real + 3× fake), balanced batch sampling in the DataGenerator (50% real, 50% fake per batch), and 2× sample weights on real samples.

---

## Project structure

```
├── model_fusion.py          # Model architecture (Xception + Audio CNN + attention fusion)
├── training.py              # Two-phase training pipeline + DataGenerator
├── evaluate.py              # Deterministic evaluation with threshold sweep
├── preprocess_video.py      # Face extraction (MTCNN) + augmentation functions
├── preprocess_audio.py      # Mel+MFCC extraction + spectrogram augmentation
├── precompute_faces.py      # Batch face extraction → processed_faces/*.npy
├── precompute_audio.py      # Batch audio extraction → processed_audio/*.npy
├── train_requirements.txt   # Python dependencies
├── processed_faces/         # Precomputed face arrays (not tracked in git)
├── processed_audio/         # Precomputed audio arrays (not tracked in git)
└── best_model.h5            # Saved model weights (not tracked in git)
```

---

## Setup

### Requirements

```bash
pip install -r train_requirements.txt
```

Key dependencies: TensorFlow 2.10, Keras 2.10, librosa, moviepy, mtcnn, opencv, scikit-learn.

### Dataset

1. Download FakeAVCeleb from the [official repository](https://github.com/DASH-Lab/FakeAVCeleb)
2. Place it at `FakeAVCeleb/FakeAVCeleb/` so the metadata CSV is at `FakeAVCeleb/FakeAVCeleb/meta_data.csv`

---

## Usage

### Step 1 — Precompute features

Run these once before training. They extract faces and audio from all videos and save them as `.npy` files so training doesn't process raw video every epoch.

```bash
python precompute_faces.py
python precompute_audio.py
```

This creates `processed_faces/` and `processed_audio/` directories. Expect this to take several hours depending on your hardware.

### Step 2 — Train

```bash
python training.py
```

Training runs in two phases automatically:

- **Phase 1** (~8 epochs): Xception frozen, trains audio branch and fusion head from scratch
- **Phase 2** (up to 15 epochs): all layers unfrozen, end-to-end fine-tuning at lr=3e-5

The best model by `val_fused_pred_accuracy` is saved to `best_model.h5`. Training progress is logged to `training_log.csv`.

### Step 3 — Evaluate

```bash
python evaluate.py
```

Evaluates `best_model.h5` on the held-out identity split. Outputs a classification report, per-branch ROC-AUC scores, threshold sweep, and saves `evaluation_results.png` with confusion matrix, ROC curves, and score distribution plots.

---

## Augmentation

To improve robustness on real-world video (social media compression, varied recording conditions), the following augmentations are applied on-the-fly during training only. Validation always uses clean unaugmented data.

**Visual (`preprocess_video.py`)**
- Horizontal flip
- Brightness and contrast jitter (±10%)
- Random zoom crop — simulates different camera distances
- JPEG compression (quality 30–95, applied 50% of the time) — forces the model to detect manipulation from facial structure rather than pristine GAN artifacts that disappear after social media recompression

**Audio (`preprocess_audio.py`)**
- Frequency masking — zeros out random mel/MFCC frequency bands, simulating lossy codec artifacts
- Time masking — zeros out random time segments, simulating network dropout
- Gaussian noise — simulates microphone background noise

Augmentations are applied to precomputed `.npy` files at batch load time, so precomputation only needs to run once.

---

## .gitignore

Add this to your `.gitignore` to avoid committing large binary files:

```
processed_faces/
processed_audio/
best_model.h5
training_log.csv
*.npy
__pycache__/
venv/
*.pyc
```

---

## Limitations

- Only 499 real videos are available in FakeAVCeleb after preprocessing, limiting the diversity of genuine faces the model has seen
- Real-world performance on heavily compressed or low-quality video may be lower than the reported 94% — the dataset consists of relatively clean studio-quality recordings
- The audio branch (ROC-AUC 0.827) is weaker than the visual branch (0.959) and could be improved by replacing the CNN with a pretrained audio model such as wav2vec2 or VGGish

---

## Acknowledgements

- [FakeAVCeleb dataset](https://github.com/DASH-Lab/FakeAVCeleb) — Khalid et al., 2021
- [Xception](https://arxiv.org/abs/1610.02357) — Chollet, 2017
- [MTCNN](https://arxiv.org/abs/1604.02878) — Zhang et al., 2016
- SpecAugment-style audio augmentation — Park et al., 2019
