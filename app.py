"""
app.py — MedVision AI Streamlit Web Application
=================================================
The complete user-facing web interface.

Run with:
    streamlit run app.py

Features:
  - Upload a chest X-ray image (JPEG / PNG)
  - Choose between Baseline CNN and ResNet-18
  - View prediction (NORMAL / PNEUMONIA) with confidence bar
  - Inspect Grad-CAM heatmap side-by-side with the original image
  - Download the heatmap
  - Medical disclaimer prominently displayed
  - Model statistics and dataset info in sidebar

Design:
  - Dark medical theme (deep navy + teal + coral)
  - Clean, professional layout
  - Status messages guide the user through each step
"""

import os
import sys
import io
import json
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# ── Path setup ────────────────────────────────────────────────────────────────
# Add the project root to sys.path so imports like "from src.xxx import yyy" work
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.config_loader import get_config
from src.utils.visualization import overlay_gradcam


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedVision AI — Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Medical Dark Theme ────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Dark Background ---- */
.stApp {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2a3d 50%, #0d1b2a 100%);
    color: #e8eaf0;
}

/* ---- Sidebar ---- */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #112040 100%);
    border-right: 1px solid #1e3a5f;
}

/* ---- Header Banner ---- */
.medvision-header {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.medvision-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4fc3f7, #80cbc4, #4fc3f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}

.medvision-subtitle {
    color: #90a4ae;
    font-size: 1rem;
    margin-top: 8px;
    font-weight: 300;
}

/* ---- Disclaimer Banner ---- */
.disclaimer-box {
    background: linear-gradient(135deg, #3d1a00 0%, #5c2700 100%);
    border: 1.5px solid #ff8f00;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 16px 0;
    color: #ffcc80;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ---- Prediction Card ---- */
.pred-card {
    border-radius: 14px;
    padding: 28px 32px;
    margin: 12px 0;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}

.pred-normal {
    background: linear-gradient(135deg, #003d2e 0%, #00573e 100%);
    border: 2px solid #00c896;
}

.pred-pneumonia {
    background: linear-gradient(135deg, #4a1010 0%, #6b1c1c 100%);
    border: 2px solid #ef5350;
}

.pred-uncertain {
    background: linear-gradient(135deg, #3d3000 0%, #5c4800 100%);
    border: 2px solid #ffb300;
}

.pred-label {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

.pred-confidence {
    font-size: 1.1rem;
    margin-top: 8px;
    opacity: 0.9;
}

/* ---- Score Bar ---- */
.score-bar-container {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}

/* ---- Metric Cards ---- */
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 10px;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #4fc3f7;
}

.metric-label {
    font-size: 0.78rem;
    color: #78909c;
    margin-top: 4px;
}

/* ---- Step indicators ---- */
.step-badge {
    display: inline-block;
    background: #0f3460;
    color: #4fc3f7;
    border: 1px solid #1e5f9e;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 10px;
}

/* ---- Image containers ---- */
.img-card {
    background: rgba(15, 52, 96, 0.3);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}

.img-label {
    color: #78909c;
    font-size: 0.82rem;
    margin-top: 8px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Cached: Load Config ────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    return get_config()


# ── Cached: Load Model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_type: str, cfg: dict):
    """
    Load the trained model from checkpoint.
    Wrapped in st.cache_resource so it loads once and stays in memory.
    """
    import torch
    from src.training.evaluate import load_model_from_checkpoint
    from src.training.train import get_device

    device    = get_device(cfg)
    ckpt_dir  = Path(cfg["training"]["checkpoint_dir"])

    if model_type == "resnet18":
        from src.models.resnet import build_resnet
        ckpt_path = ckpt_dir / "best_resnet18.pth"
        model = build_resnet(cfg)
    else:
        from src.models.cnn import build_baseline_cnn
        ckpt_path = ckpt_dir / "best_cnn.pth"
        model = build_baseline_cnn(cfg)

    if not ckpt_path.exists():
        return None, f"Checkpoint not found: {ckpt_path}.\nPlease train the model first."

    try:
        model = load_model_from_checkpoint(model, ckpt_path, device)
        return model, None
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(cfg: dict):
    with st.sidebar:
        st.markdown("## 🔬 MedVision AI")
        st.markdown("---")

        # Model selection
        st.markdown("### 🧠 Model")
        model_type = st.radio(
            "Select model:",
            options=["resnet18", "cnn"],
            format_func=lambda x: "ResNet-18 (Transfer Learning)" if x == "resnet18"
                                   else "Baseline CNN (Scratch)",
            index=0,
            key="model_selector",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Options")
        show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True, key="show_cam")
        alpha = st.slider("Heatmap opacity", 0.2, 0.8, 0.4, 0.05, key="alpha_slider")

        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("""
        **MedVision AI** is a research prototype built as a final-year B.Tech
        AI/ML project.

        - **Dataset**: Chest X-Ray (Pneumonia) — Kaggle
        - **Model**: ResNet-18 + Custom CNN
        - **Explainability**: Grad-CAM
        """)

        st.markdown("---")
        st.markdown(
            '<div style="color: #546e7a; font-size:0.75rem;">'
            'Built with PyTorch · Streamlit<br>'
            '⚠️ Not for clinical use'
            '</div>',
            unsafe_allow_html=True,
        )

    return model_type, show_gradcam, alpha


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()

    # ── Sidebar ──
    model_type, show_gradcam, alpha = render_sidebar(cfg)

    # ── Header ──
    st.markdown("""
    <div class="medvision-header">
        <p class="medvision-title">🫁 MedVision AI</p>
        <p class="medvision-subtitle">
            Explainable Chest X-Ray Classification · Deep Learning Research Prototype
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>MEDICAL DISCLAIMER</strong> — This application is an educational
        research prototype developed as a final-year B.Tech project. It is
        <strong>NOT a certified medical device</strong> and has <strong>NOT been
        clinically validated</strong>. Results must NOT be used to diagnose, treat,
        or make any clinical decisions. Always consult a qualified radiologist or
        physician for medical advice. Predictions may be inaccurate.
    </div>
    """, unsafe_allow_html=True)

    # ── Load Model ──
    with st.spinner(f"Loading {model_type} model …"):
        model, error = load_model(model_type, cfg)

    if error:
        st.error(f"❌ Could not load model:\n\n`{error}`")
        st.info(
            "**To train the model:**\n\n"
            "1. Download the dataset from Kaggle.\n"
            "2. Set `DATASET_ROOT` in your `.env` file.\n"
            "3. Run `python run_pipeline.py` (or the Colab notebook).\n"
            "4. Trained weights will be saved to `artifacts/checkpoints/`."
        )
        st.stop()

    st.success(f"✅ {model_type.upper()} model loaded successfully.")

    # ── Upload Section ──
    st.markdown("---")
    st.markdown('<div class="step-badge">STEP 1 — Upload X-Ray</div>', unsafe_allow_html=True)
    st.markdown("#### Upload a chest X-ray image")

    col_upload, col_tips = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse",
            type=["jpg", "jpeg", "png"],
            key="xray_uploader",
            label_visibility="collapsed",
        )
    with col_tips:
        st.markdown("""
        **Tips for best results:**
        - Use a frontal (PA or AP) chest X-ray
        - JPEG or PNG format
        - Grayscale or RGB image
        - Clear image without heavy watermarks
        """)

    if uploaded_file is None:
        st.markdown("---")
        st.info("👆 Please upload a chest X-ray image to begin analysis.")
        _render_example_info()
        return

    # ── Load Image ──
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── Analyse Button ──
    st.markdown("---")
    st.markdown('<div class="step-badge">STEP 2 — Analyse</div>', unsafe_allow_html=True)

    if not st.button("🔍 Analyse X-Ray", type="primary", use_container_width=True,
                     key="analyse_btn"):
        st.markdown("#### Image Preview")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        return

    # ── Run Inference ──
    with st.spinner("Analysing image …"):
        from src.inference.predict import predict_single_image
        result = predict_single_image(
            image=image,
            model=model,
            cfg=cfg,
            generate_cam=show_gradcam,
        )

    # ── Results Section ──
    st.markdown("---")
    st.markdown('<div class="step-badge">STEP 3 — Results</div>', unsafe_allow_html=True)

    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    is_uncert  = result["is_uncertain"]
    scores     = result["class_scores"]

    # ── Prediction Card ──
    if is_uncert:
        card_cls = "pred-uncertain"
        icon     = "⚠️"
        note     = "Low confidence — result may be unreliable"
    elif pred_class == "NORMAL":
        card_cls = "pred-normal"
        icon     = "✅"
        note     = "No signs of pneumonia detected"
    else:
        card_cls = "pred-pneumonia"
        icon     = "🔴"
        note     = "Signs consistent with pneumonia detected"

    st.markdown(f"""
    <div class="pred-card {card_cls}">
        <p class="pred-label">{icon} {pred_class}</p>
        <p class="pred-confidence">Confidence: {confidence*100:.1f}%</p>
        <p style="color:#b0bec5; font-size:0.85rem; margin-top:8px;">{note}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Class Probability Bars ──
    st.markdown("#### Class Probabilities")
    for cls_name, prob in scores.items():
        color = "#00c896" if cls_name == "NORMAL" else "#ef5350"
        bar_pct = prob * 100
        st.markdown(f"""
        <div class="score-bar-container" style="margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="font-weight:600;">{cls_name}</span>
                <span style="color:{color}; font-weight:700;">{bar_pct:.1f}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:10px;">
                <div style="background:{color}; width:{bar_pct}%; height:10px;
                            border-radius:6px; transition:width 0.6s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Image + Grad-CAM ──
    st.markdown("---")
    st.markdown('<div class="step-badge">STEP 4 — Visual Explanation (Grad-CAM)</div>',
                unsafe_allow_html=True)

    heatmap  = result.get("heatmap")
    orig_img = result.get("original_image")

    if show_gradcam and heatmap is not None and orig_img is not None:
        blended = overlay_gradcam(
            original_image=orig_img,
            heatmap=heatmap,
            alpha=alpha,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(image, caption=None, use_container_width=True)
            st.markdown('<p class="img-label">📷 Original X-Ray</p></div>',
                        unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(blended, caption=None, use_container_width=True)
            st.markdown(
                '<p class="img-label">🌡️ Grad-CAM Heatmap — '
                'Red/Yellow = high model attention</p></div>',
                unsafe_allow_html=True,
            )

        # Download button
        img_pil  = Image.fromarray(blended)
        buf      = io.BytesIO()
        img_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Grad-CAM image",
            data=buf.getvalue(),
            file_name="gradcam_output.png",
            mime="image/png",
            key="download_gradcam",
        )

        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
                    border-radius:10px; padding:14px 18px; margin-top:12px; font-size:0.85rem;
                    color:#90a4ae; line-height:1.6;">
            <strong>🔍 How to interpret Grad-CAM:</strong><br>
            <span style="color:#ef5350;">Red/Yellow</span> regions contributed most to the prediction.<br>
            <span style="color:#4fc3f7;">Blue/Green</span> regions had less influence.<br>
            In PNEUMONIA cases, activation often clusters in the lower lung fields
            where consolidation or opacification typically appears.
        </div>
        """, unsafe_allow_html=True)

    elif show_gradcam:
        st.warning("Grad-CAM could not be generated for this image.")
    else:
        # Show just the original
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    # ── Second Disclaimer ──
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>REMINDER</strong>: This result is generated by an AI model for
        <strong>educational and research purposes only</strong>. It is not a medical
        diagnosis. Do not make clinical decisions based on this output. Consult a
        qualified medical professional.
    </div>
    """, unsafe_allow_html=True)


def _render_example_info():
    """Show helpful information when no image is uploaded."""
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    steps = [
        ("1️⃣", "Upload", "Upload a frontal chest X-ray image in JPEG or PNG format."),
        ("2️⃣", "Analyse", "The AI model classifies the X-ray as NORMAL or PNEUMONIA."),
        ("3️⃣", "Explain", "Grad-CAM highlights the regions that influenced the decision."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], steps):
        with col:
            st.markdown(f"""
            <div style="background:rgba(15,52,96,0.4); border:1px solid #1e3a5f;
                        border-radius:12px; padding:20px; text-align:center; height:140px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-weight:600; margin:8px 0; color:#4fc3f7;">{title}</div>
                <div style="font-size:0.83rem; color:#90a4ae;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
