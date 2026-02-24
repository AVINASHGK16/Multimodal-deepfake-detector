# 🛡️ Multimodal Deepfake & Synthetic Media Detector

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

An end-to-end, multimodal deep learning pipeline designed to detect manipulated video content (deepfakes). This project analyzes both spatial visual artifacts and temporal audio inconsistencies simultaneously, providing a holistic authenticity score alongside Explainable AI (XAI) heatmaps.

## 🧠 System Architecture

This detector utilizes a **Dual-Stream Deep Learning Architecture** to process varying modalities before fusing them for a final prediction.

1. **The Visual Stream (Xception CNN):**
   * Processes facial regions extracted via Haar Cascades.
   * Utilizes depthwise separable convolutions to detect micro-artifacts like unnatural pixel blending and warping.
2. **The Audio Stream (CNN + LSTM):**
   * Extracts audio features using Mel-spectrograms.
   * A Convolutional Neural Network handles spatial frequency extraction, while a Long Short-Term Memory (LSTM) network tracks temporal sequences to catch robotic tones and lip-sync anomalies.
3. **The Fusion Layer:**
   * Concatenates the features from both streams into Fully Connected Dense layers.
   * Outputs a final binary classification probability (Real vs. Fake) via a Sigmoid activation function.
4. **Transparent Reasoning (Grad-CAM):**
   * Implements Gradient-weighted Class Activation Mapping.
   * Generates a superimposed heatmap on the UI, highlighting the specific facial regions that influenced the model's manipulation prediction.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (opencv-python-headless)
* **Audio Processing:** Librosa, MoviePy
* **Frontend UI:** Streamlit
* **Data Handling:** NumPy, Scikit-Learn

## 🚀 Installation & Usage

**1. Clone the repository**
```bash
git clone [https://github.com/AVINASHGK16/multimodal-deepfake-detector.git](https://github.com/AVINASHGK16/multimodal-deepfake-detector.git)
cd multimodal-deepfake-detector
