# Running the Multimodal Deepfake Detector (Friends/Tester Guide)

## 1) Prerequisites

- **Python**: 3.10+ recommended
- **ffmpeg**: recommended (helps MoviePy/Librosa decode audio from videos)

## 2) Install dependencies

From the project folder:

```bash
pip install -r requirements.txt
```

If you get an error about `mtcnn` missing (the video preprocessor imports it), install it too:

```bash
pip install mtcnn
```

## 3) Put the trained weights in the right place

This project’s `app.py` loads weights using a **fixed filename**:

```text
dummy_phase2.weights.h5
```

To use the trained weights **without changing any code**:

- Copy your trained weights file into the **project root** (same folder as `app.py`)
- Rename it to:

```text
dummy_phase2.weights.h5
```

After this, your folder should look like:

```text
Multimodal-deepfake-detector/
  app.py
  dummy_phase2.weights.h5    <-- your trained weights (renamed)
  model_fusion.py
  preprocess_video.py
  preprocess_audio.py
  ...
```

## 4) Run the app (Streamlit)

From the project folder:

```bash
python -m streamlit run app.py
```

Streamlit will print a **Local URL** (usually `http://localhost:8501`). Open it in your browser.

## 5) Quick troubleshooting

- **“Could not load weights …”**
  - Confirm the weights file is in the **same folder as `app.py`** and named **exactly** `dummy_phase2.weights.h5`.
- **“No module named mtcnn”**
  - Run: `pip install mtcnn`
- **Audio extraction fails / no audio detected**
  - Install ffmpeg and try a different MP4/MOV with a clear audio track.

