# Multimodal Deepfake Detection

A deep learning system for detecting deepfake videos by analysing both visual and audio streams simultaneously. The model combines an Xception-based face analysis branch with a CNN-based audio branch, fused through a cross-modal attention gate that learns which modality to trust on a per-sample basis.

Trained and evaluated on [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) supplemented with real videos from [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html), with strict identity-level train/val splitting to prevent data leakage.

---

## Results

Evaluated on 2,846 samples from identities never seen during training.

| Metric | Score |
|---|---|
| Accuracy | 94% |
| ROC-AUC (fused) | 0.981 |
| ROC-AUC (visual branch) | 0.968 |
| ROC-AUC (audio branch) | 0.827 |
| Precision (real) | 0.93 |
| Precision (fake) | 0.95 |
| Recall (real) | 0.78 |
| Recall (fake) | 0.99 |
| Val set size | 2,846 samples (unseen identities) |

The model is designed to prioritise fake detection — fake recall of 0.99 means nearly all deepfakes are caught, at the cost of some false positives on genuine videos. This is the preferred tradeoff for a deepfake detection system where missing a deepfake is worse than flagging a real video for human review.

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

**Focal loss** — replaces standard binary crossentropy to handle class imbalance. Down-weights easy examples (abundant fake samples already classified correctly) and focuses training on hard examples, particularly the minority real class. Alpha=0.75 gives extra weight to real samples, gamma=2.0 is the standard focusing parameter.

**Multi-output auxiliary losses** — the model outputs three predictions (visual-only, audio-only, fused) with loss weights of 1.0, 0.3, and 3.0 respectively. This forces each branch to learn independently before fusion.

---

## Dataset

### FakeAVCeleb

[FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) contains four video types:

| Type | Description |
|---|---|
| RealVideo-RealAudio | Genuine videos |
| FakeVideo-FakeAudio | Both face and voice synthesised |
| FakeVideo-RealAudio | Face swapped, original audio |
| RealVideo-FakeAudio | Original face, voice cloned |

The dataset has a 42:1 fake-to-real imbalance (~500 real vs ~21,000 fake). Using the dataset as-is produces misleading accuracy — a model predicting everything as fake achieves 97% accuracy trivially.

### VoxCeleb2 supplement

To address the real class scarcity, ~4,000 real videos from [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) were added. FakeAVCeleb's fake videos were generated from VoxCeleb2 identities, making this a natural and domain-consistent supplement. The combined dataset uses all FakeAVCeleb fake videos against ~4,500 real videos at a 4:1 ratio.

### Imbalance handling

Three mechanisms work together to handle the remaining 4:1 imbalance:

- **Balanced batch sampling** — every training batch is 50% real, 50% fake regardless of dataset ratio
- **Sample weights** — real samples receive 2× loss weight within each batch
- **Focal loss** — further emphasises hard examples and minority class

---

## Project structure

```
├── model_fusion.py             # Model architecture + focal loss
├── training.py                 # Two-phase training pipeline + DataGenerator
├── evaluate.py                 # Deterministic evaluation with threshold sweep
├── preprocess_video.py         # Face extraction (MTCNN) + augmentation
├── preprocess_audio.py         # Mel+MFCC extraction + augmentation
├── precompute_faces.py         # Batch face extraction for FakeAVCeleb
├── precompute_audio.py         # Batch audio extraction for FakeAVCeleb
├── precompute_faces_vox.py     # Batch face extraction for VoxCeleb2
├── precompute_audio_vox.py     # Batch audio extraction for VoxCeleb2
├── train_requirements.txt      # Python dependencies
├── processed_faces/            # Precomputed face arrays (not tracked in git)
├── processed_audio/            # Precomputed audio arrays (not tracked in git)
└── best_model.h5               # Saved model weights (not tracked in git)
```

---

## Setup

### Requirements

```bash
pip install -r train_requirements.txt
```

Key dependencies: TensorFlow 2.10, Keras 2.10, librosa, moviepy, mtcnn, opencv, scikit-learn.

### Datasets

**FakeAVCeleb:**
1. Download from the [official repository](https://github.com/DASH-Lab/FakeAVCeleb)
2. Place at `FakeAVCeleb/FakeAVCeleb/` so the metadata CSV is at `FakeAVCeleb/FakeAVCeleb/meta_data.csv`

**VoxCeleb2:**
1. Request access at [https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
2. Download approximately 4,000–5,000 real videos
3. Place all `.mp4` files in a flat folder (e.g. `VoxSample/downloads/`)
4. Update `VOX_ROOT` in `training.py`, `evaluate.py`, `precompute_faces_vox.py`, and `precompute_audio_vox.py` to point to this folder

---

## Usage

### Step 1 — Precompute features

Run these once before training. They extract face crops and audio spectrograms from all videos and save them as `.npy` files so training does not process raw video every epoch.

**FakeAVCeleb:**
```bash
python precompute_faces.py
python precompute_audio.py
```

**VoxCeleb2:**
```bash
python precompute_faces_vox.py
python precompute_audio_vox.py
```

The VoxCeleb2 precompute scripts include a 30-second per-video timeout to handle videos where MTCNN hangs. Both scripts are safe to restart — already-processed files are skipped automatically.

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

Applied on-the-fly during training only. Validation always uses clean unaugmented data.

**Visual (`preprocess_video.py`)**
- Horizontal flip
- Brightness and contrast jitter (±10%)
- Random zoom crop — simulates different camera distances
- JPEG compression augmentation (quality 30–95, 50% probability) — forces the model to detect manipulation from facial structure rather than pristine GAN artifacts that disappear after social media recompression

**Audio (`preprocess_audio.py`)**
- Frequency masking — zeros out random mel/MFCC frequency bands, simulating lossy codec artifacts
- Time masking — zeros out random time segments, simulating network dropout
- Gaussian noise — simulates microphone background noise

---

## Ablation

Performance across different training strategies on the same identity-level split:

| Strategy | Accuracy | Real recall | Fake recall | Fused AUC | Val samples |
|---|---|---|---|---|---|
| Full dataset, focal loss only | 97% | 0.52 | 0.98 | 0.974 | 4,689 |
| 3:1 subset, BCE + weights | 94% | 0.87 | 0.96 | 0.968 | 404 |
| **VoxCeleb2 + 4:1 + focal loss** | **94%** | **0.78** | **0.99** | **0.981** | **2,846** |

The full dataset with focal loss alone produces misleading accuracy — the model correctly predicts 97% of samples but misclassifies half of all real videos. Supplementing with VoxCeleb2 real videos and balanced batch sampling gives the best overall result.

---

## .gitignore

```
processed_faces/
processed_audio/
best_model.h5
training_log.csv
*.npy
__pycache__/
venv/
*.pyc
*.wav
```

---

## Limitations

- Real recall of 0.78 means approximately 1 in 5 genuine videos is incorrectly flagged as fake — a consequence of the dataset's inherent real class scarcity
- Performance on heavily compressed or low-quality video may be lower than reported — the dataset consists of relatively clean studio-quality recordings
- The audio branch (ROC-AUC 0.827) is weaker than the visual branch (0.968) and could be improved by replacing the CNN with a pretrained model such as wav2vec2 or VGGish

---

## Acknowledgements

- [FakeAVCeleb dataset](https://github.com/DASH-Lab/FakeAVCeleb) — Khalid et al., 2021
- [VoxCeleb2 dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) — Chung et al., 2018
- [Xception](https://arxiv.org/abs/1610.02357) — Chollet, 2017
- [MTCNN](https://arxiv.org/abs/1604.02878) — Zhang et al., 2016
- [Focal Loss](https://arxiv.org/abs/1708.02002) — Lin et al., 2017
- SpecAugment-style audio augmentation — Park et al., 2019
