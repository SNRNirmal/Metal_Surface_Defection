from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from DefectAnalysis.pipeline import ONNXInferencePipeline

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "ONNX_models"
OUTPUTS_DIR = BASE_DIR / "outputs" / "streamlit"

MODEL_CANDIDATES = {
    "FPN + ResNet34": ["fpn_resnet.onnx"]
}

def resolve_models(models_dir: Path) -> dict:
    resolved = {}
    for model_name, candidates in MODEL_CANDIDATES.items():
        for filename in candidates:
            model_path = models_dir / filename
            if model_path.is_file():
                resolved[model_name] = model_path
                break
    return resolved


ONNX_MODELS = resolve_models(MODELS_DIR)

st.set_page_config(
    page_title="Steel Defect Analyser",
    page_icon="🔩",
    layout="wide",
)

if not ONNX_MODELS:
    st.error(f"No ONNX models found in {MODELS_DIR}.")
    st.stop()


@st.cache_resource
def load_pipeline(model_path: str) -> ONNXInferencePipeline:
    return ONNXInferencePipeline(str(model_path))


def severity_badge(severity: str) -> str:
    colors = {
        "Low":      "🟢",
        "Medium":   "🟡",
        "High":     "🟠",
        "Critical": "🔴",
    }
    return f"{colors.get(severity, '')} **{severity}**"

st.sidebar.title("⚙️ Settings")
model_name  = st.sidebar.selectbox("Model", list(ONNX_MODELS.keys()))
threshold   = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
min_pixels  = st.sidebar.slider("Min Defect Pixels",    0,  200, 20,  5)
save_output = st.sidebar.checkbox("Save annotated outputs", value=False)

st.title("🔩 Steel Surface Defect Analyser")
st.markdown("Upload a steel surface image to detect defects and estimate remaining useful life.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    model_path = ONNX_MODELS[model_name]
    if not model_path.is_file():
        st.error(f"ONNX model not found: `{model_path}`. Export it first using `ONNX_converter.py`.")
        st.stop()

    pipe = load_pipeline(model_path)
    pipe.threshold         = threshold
    pipe.min_defect_pixels = min_pixels

    # Decode to BGR numpy
    pil_img    = Image.open(uploaded).convert("RGB")
    img_rgb    = np.array(pil_img)
    img_bgr    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Analysing..."):
        result = pipe.run_from_array(img_bgr)

    report = result["report"]
    rul    = result["rul"]

    # ── Layout ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_rgb, caption="Original Image", use_column_width=True)
    with col2:
        overlay_rgb = cv2.cvtColor(result["overlay_image"], cv2.COLOR_BGR2RGB)
        st.image(overlay_rgb, caption="Defect Overlay", use_column_width=True)
    with col3:
        heatmap_rgb = cv2.cvtColor(result["heatmap_image"], cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, caption="Defect Heatmap", use_column_width=True)

    st.divider()

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Defect Area",  f"{report['defect_percentage']:.2f}%")
    m2.metric("Risk Score",   f"{report['risk_score']:.1f} / 100")
    m3.metric("Severity",     report["severity"])
    m4.metric("Est. RUL",     f"{rul['rul_days']} days")

    st.markdown(f"**Recommendation:** {rul['rul_recommendation']}")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    st.subheader("Per-Class Defect Distribution")
    per_class = report["per_class_area"]
    filtered  = {k: v for k, v in per_class.items() if v > 0}

    if filtered:
        import pandas as pd
        df = pd.DataFrame(
            list(filtered.items()), columns=["Defect Class", "Area (%)"]
        ).sort_values("Area (%)", ascending=False)
        st.bar_chart(df.set_index("Defect Class"))
        st.dataframe(df, use_container_width=True)
    else:
        st.success("No defects detected in this image.")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_output:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(uploaded.name).stem
        cv2.imwrite(str(OUTPUTS_DIR / f"{stem}_overlay.jpg"), result["overlay_image"])
        cv2.imwrite(str(OUTPUTS_DIR / f"{stem}_heatmap.jpg"), result["heatmap_image"])
        st.success(f"Saved to `{OUTPUTS_DIR}/`")