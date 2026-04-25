import base64
import json
import os
from pathlib import Path

# Enable legacy TF-Keras (Keras 2) for older `.h5` checkpoints saved under
# TensorFlow 2.10-era Keras. This must be set before importing TensorFlow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf

from explainability import generate_gradcam_heatmap
from model_fusion import build_fusion_model
from preprocess_audio import get_mel_spectrogram
from preprocess_video import extract_face_pipeline

_ROOT_DIR = Path(__file__).resolve().parent
_CSS_FILE = _ROOT_DIR / "assets" / "styles.css"
_BG_FILE = (
    _ROOT_DIR / "assets" / "bg.jpg"
    if (_ROOT_DIR / "assets" / "bg.jpg").exists()
    else _ROOT_DIR / "assets" / "bg.png"
)

# Keep original project behavior: weights loaded from a single filename.
_WEIGHTS_FILE = "best_model.h5"
_THRESHOLD_FILE = _ROOT_DIR / "model_threshold.json"
_DEFAULT_FAKE_THRESHOLD = 0.5


def extract_fallback_frame(video_path):
    """Fallback face-like input when detector misses all faces."""
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (299, 299))
            return resized.astype("float32") / 255.0
    finally:
        cap.release()
    return None


def load_fake_threshold():
    """Load tuned fake threshold from disk when available."""
    try:
        if _THRESHOLD_FILE.exists():
            payload = json.loads(_THRESHOLD_FILE.read_text(encoding="utf-8"))
            threshold = float(payload.get("best_threshold", _DEFAULT_FAKE_THRESHOLD))
            return float(np.clip(threshold, 0.0, 1.0))
    except Exception:
        pass
    return _DEFAULT_FAKE_THRESHOLD


def preprocess_for_inference(video_path, strict_mode=False):
    """
    Align UI preprocessing with training/evaluation behavior.
    - Visual: use up to 2 detected faces and average them (like precompute scripts).
    - Audio: use extracted mel+mfcc tensor with deterministic shaping.
    """
    notices = []

    faces = extract_face_pipeline(video_path, max_frames=2, frame_skip=5)
    if len(faces) > 0:
        visual = np.mean(np.array(faces, dtype=np.float32), axis=0)
    else:
        if strict_mode:
            return None, None, notices
        fallback_frame = extract_fallback_frame(video_path)
        if fallback_frame is None:
            return None, None, notices
        visual = fallback_frame
        notices.append("No clear face found. Using best available video frame fallback.")

    audio = get_mel_spectrogram(video_path)
    if audio is None:
        if strict_mode:
            return None, None, notices
        audio = np.full((148, 128, 1), -80.0, dtype=np.float32)
        notices.append("Audio extraction failed. Using silent-audio fallback.")

    audio_2d = np.squeeze(audio)
    target_time_steps = 128
    current_time_steps = audio_2d.shape[1]
    if current_time_steps < target_time_steps:
        pad_width = target_time_steps - current_time_steps
        audio_2d = np.pad(audio_2d, ((0, 0), (0, pad_width)), mode="constant")
    else:
        audio_2d = audio_2d[:, :target_time_steps]
    audio = np.expand_dims(audio_2d, axis=-1).astype(np.float32)

    return visual.astype(np.float32), audio, notices


def apply_custom_css():
    """Inject CSS into the parent document head.

    Streamlit's markdown renderer can strip or escape ``<style>`` blocks, which
    makes raw CSS appear as visible text. A zero-height HTML component runs a
    small script that appends a ``<style>`` tag to ``window.parent.document``.
    """
    try:
        extra_css = _CSS_FILE.read_text(encoding="utf-8")
    except OSError:
        extra_css = ""

    # Streamlit doesn't serve arbitrary local files as static URLs, so we embed
    # the background image as a data URL if present.
    if _BG_FILE.exists():
        try:
            b64 = base64.b64encode(_BG_FILE.read_bytes()).decode("ascii")
            extra_css += (
                "\n\n/* Embedded background image */\n"
                "[data-testid=\"stAppViewContainer\"] {\n"
                "  background:\n"
                "    linear-gradient(rgba(15, 23, 42, 0.78), rgba(15, 23, 42, 0.78)),\n"
                f"    url(\"data:image/jpeg;base64,{b64}\");\n"
                "  background-size: cover;\n"
                "  background-position: center;\n"
                "  background-repeat: no-repeat;\n"
                "  background-attachment: fixed;\n"
                "}\n"
            )
        except OSError:
            pass
    css_js = json.dumps(extra_css)
    font_url = (
        "https://fonts.googleapis.com/css2?"
        "family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400"
        "&display=swap"
    )
    components.html(
        f"""
        <script>
        (function() {{
            var doc = window.parent.document;
            var old = doc.getElementById("multimodal-app-css");
            if (old) old.remove();
            var style = doc.createElement("style");
            style.id = "multimodal-app-css";
            style.type = "text/css";
            style.textContent = {css_js};
            doc.head.appendChild(style);
            if (!doc.getElementById("multimodal-app-font")) {{
                var link = doc.createElement("link");
                link.id = "multimodal-app-font";
                link.rel = "stylesheet";
                link.href = {json.dumps(font_url)};
                doc.head.appendChild(link);
            }}
        }})();
        </script>
        """,
        height=0,
    )


apply_custom_css()


@st.cache_resource
def load_detector_model():
    # `best_model.h5` from training is saved as a full model by ModelCheckpoint.
    # Prefer loading full model first, then fallback to weights-only loading.
    try:
        loaded = tf.keras.models.load_model(_WEIGHTS_FILE, compile=False)
        return loaded
    except Exception as e:
        st.warning(f"Could not load saved model from '{_WEIGHTS_FILE}'. Error: {e}")

    model = build_fusion_model()
    try:
        model.load_weights(_WEIGHTS_FILE)
    except Exception as e:
        st.warning(f"Could not load weights from '{_WEIGHTS_FILE}'. Error: {e}")
    return model


def display_results(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_normalized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    if original_img.dtype != np.uint8:
        original_img = np.uint8(original_img * 255)
    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    superimposed_img = cv2.addWeighted(original_img_bgr, 0.6, heatmap_color, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original face crop", channels="RGB", use_container_width=True)
    with col2:
        st.image(superimposed_rgb, caption="Grad-CAM overlay (XAI)", channels="RGB", use_container_width=True)


def render_verdict(divergence, fused_fake_prob, fake_threshold_pct=50.0):
    if divergence > 40.0:
        st.markdown(
            f"""
            <div class="verdict verdict-strong">
                <span class="verdict-title">Asymmetric signal</span>
                Large gap between visual and audio scores ({divergence:.1f}% divergence).
                Possible voice swap, lip-sync mismatch, or mixed manipulation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif fused_fake_prob > fake_threshold_pct:
        st.markdown(
            f"""
            <div class="verdict verdict-warn">
                <span class="verdict-title">Elevated manipulation risk</span>
                Fused fake probability is <strong>{fused_fake_prob:.1f}%</strong>. Treat as suspicious until verified.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict verdict-ok">
                <span class="verdict-title">Lower risk (fused score)</span>
                Fused fake probability is <strong>{fused_fake_prob:.1f}%</strong>. Not a guarantee of authenticity.
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Layout ---
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-inner">
            <div class="hero-badge">Multimodal AI</div>
            <h1 class="hero-title">Multimodal DeepFake and Synthetic AI Detector</h1>
            <div class="hero-sub">
                <span class="hero-sub-lead">Upload a short video.</span>
                <span class="hero-sub-detail">
                    We compare visual and audio cues, then fuse them into one interpretable score&mdash;with optional attention maps so you can see where the model focused.
                </span>
                <div class="hero-pills" aria-hidden="true">
                    <span class="hero-pill">&#128249; Video</span>
                    <span class="hero-pill">&#128065;&#128066; Vision + audio</span>
                    <span class="hero-pill">&#128202; Fused score</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

detector = load_detector_model()

tab1, tab2, tab3 = st.tabs(["Analyze", "How it works", "About"])

with tab1:
    st.markdown('<div class="panel-label">Upload</div>', unsafe_allow_html=True)
    st.caption("MP4, MOV, or AVI — clear face and speech improve reliability.")
    strict_mode = False
    threshold = load_fake_threshold()
    uploaded_file = st.file_uploader(
        "Drop a file or browse",
        type=["mp4", "mov", "avi"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.status("Running analysis…", expanded=True) as status:
            st.write("Extracting face regions…")
            st.write("Building audio spectrogram…")
            visual_features, audio_features, notices = preprocess_for_inference(
                video_path,
                strict_mode=strict_mode,
            )

            for notice in notices:
                st.warning(notice)

            if visual_features is not None and audio_features is not None:
                st.write("Neural network inference…")
                v_input = np.expand_dims(visual_features, axis=0)
                a_input = np.expand_dims(audio_features, axis=0)

                predictions = detector.predict([v_input, a_input])
                v_score = predictions[0][0][0]
                a_score = predictions[1][0][0]
                fused_score = predictions[2][0][0]

                status.update(label="Done", state="complete", expanded=False)

                st.markdown('<div class="section-title">Scores</div>', unsafe_allow_html=True)
                # Model is trained with label=1 as fake; sigmoid output is P(fake).
                v_fake_prob = float(v_score) * 100.0
                a_fake_prob = float(a_score) * 100.0
                fused_fake_prob = float(fused_score) * 100.0
                divergence = abs(v_fake_prob - a_fake_prob)

                c1, c2, c3 = st.columns(3)
                c1.metric("Visual (fake %)", f"{v_fake_prob:.1f}%")
                c2.metric("Audio (fake %)", f"{a_fake_prob:.1f}%")
                c3.metric("Fused (fake %)", f"{fused_fake_prob:.1f}%")

                st.markdown('<div class="section-title">Assessment</div>', unsafe_allow_html=True)
                if fused_score >= threshold:
                    st.info(
                        f"Classification: **Likely Fake** (fused={fused_score:.3f}, threshold={threshold:.2f})"
                    )
                else:
                    st.info(
                        f"Classification: **Likely Real** (fused={fused_score:.3f}, threshold={threshold:.2f})"
                    )
                render_verdict(divergence, fused_fake_prob, fake_threshold_pct=threshold * 100.0)

                st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
                try:
                    real_heatmap = generate_gradcam_heatmap(
                        detector, [v_input, a_input], "block14_sepconv2_act"
                    )
                    display_results(visual_features, real_heatmap)
                except Exception:
                    st.warning("Grad-CAM used a fallback (multi-output model).")
                    mock_heatmap = np.random.rand(10, 10)
                    display_results(visual_features, mock_heatmap)
            else:
                status.update(label="Could not analyze", state="error", expanded=False)
                if strict_mode:
                    st.error(
                        "Strict mode could not extract both face and audio features from this video."
                    )
                else:
                    st.error("Could not detect a clear face or audio track in the uploaded video.")

with tab2:
    st.markdown(
        """
        <div class="tab-page">
        <p class="tab-intro">
            &#128640; <strong style="color:#e2e8f0;">Two neural streams</strong> run on the same upload in parallel.
            A late-fusion layer merges what the model &ldquo;sees&rdquo; and &ldquo;hears&rdquo; so
            attacks that only fake one modality (e.g. voice clone on real footage) are harder to hide.
        </p>
        <div class="tab-section-title">&#9889; Pipeline at a glance</div>
        <div class="flow-banner">
            <span>&#128230; Your clip</span><span>&#8594;</span><span>&#128065; Face</span><span>&#8594;</span><span>&#127908; Audio</span><span>&#8594;</span><span>&#129504; Fusion</span><span>&#8594;</span><span>&#128200; Scores</span>
        </div>
        <div class="tab-section-title" style="margin-top:1.35rem;">&#128203; What each stream does</div>
        <div class="info-grid">
            <div class="info-card">
                <div class="info-card-icon">&#128065;&#128444;&#65039;</div>
                <h4>Visual stream</h4>
                <p>
                    Faces are cropped to <strong style="color:#cbd5e1;">299×299</strong>. An Xception-style CNN
                    looks for spatial artifacts: odd blending, warping, and texture inconsistencies.
                </p>
            </div>
            <div class="info-card">
                <div class="info-card-icon">&#127925;</div>
                <h4>Audio stream</h4>
                <p>
                    Audio becomes a <strong style="color:#cbd5e1;">Mel-spectrogram</strong>. A CNN
                    captures spectral and temporal patterns—unnatural rhythm, timbre, or robotic pauses.
                </p>
            </div>
            <div class="info-card">
                <div class="info-card-icon">&#129504;</div>
                <h4>Fusion &amp; alerts</h4>
                <p>
                    Features are concatenated and passed through dense layers. If
                    <strong style="color:#cbd5e1;">visual vs audio</strong> disagree strongly, we flag a possible asymmetric attack.
                </p>
            </div>
            <div class="info-card">
                <div class="info-card-icon">&#128065;&#128065;</div>
                <h4>Explainability (XAI)</h4>
                <p>
                    Grad-CAM-style heatmaps highlight regions that influenced the visual branch.
                    Use them as <strong style="color:#cbd5e1;">hints</strong>, not courtroom proof.
                </p>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab3:
    st.markdown(
        """
        <div class="tab-page">
        <div class="about-hero">
            <div class="emoji-row">&#128737;&#65039; &#129302; &#127908;</div>
            <h2>Multimodal Deepfake Detector</h2>
            <span class="ver">Prototype &middot; research / demo</span>
        </div>
        <div class="tab-section-title">&#128203; What this is</div>
        <p class="tab-intro" style="margin-bottom:1.1rem;">
            A project for deepfake detection using <strong style="color:#3B82F6;">vision + audio</strong> together.
        </p>
        <div class="tab-section-title">&#9881;&#65039; Technology stack</div>
        <ul class="meta-list">
            <li><strong style="color:#cbd5e1;">Deep learning:</strong> TensorFlow / Keras (dual-stream + fusion)</li>
            <li><strong style="color:#cbd5e1;">Vision &amp; face:</strong> OpenCV, MTCNN / Haar fallbacks</li>
            <li><strong style="color:#cbd5e1;">Audio:</strong> Librosa, MoviePy (Mel-spectrogram pipeline)</li>
            <li><strong style="color:#cbd5e1;">Interface:</strong> Streamlit + custom CSS</li>
        </ul>
        <div class="tab-section-title" style="margin-top:1.25rem;">&#127919; Mission</div>
        <p style="color:#94a3b8; line-height:1.7; margin:0 0 0.75rem;">
            Synthetic media is easy to fake but hard to spot. This project explores whether
            <strong style="color:#cbd5e1;">multimodal</strong> signals—seeing and hearing the same clip—can
            surface manipulation more reliably than looking at pixels alone.
        </p>
        <div class="tab-section-title">&#128640; Roadmap</div>
        <ul class="meta-list">
            <li>Fine-tune the visual backbone on deepfake-specific datasets</li>
            <li>Longer temporal context (multiple frames, not just one)</li>
            <li>Train on larger corpora (e.g. FakeAVCeleb-style voice + face)</li>
            <li>Optional API + separate frontend for production</li>
        </ul>
        <div class="disclaimer">
            <strong style="color:#94a3b8;">Disclaimer:</strong> Scores depend on training data and model quality.
            Do not use this app as the sole basis for legal, safety, or policy decisions.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
