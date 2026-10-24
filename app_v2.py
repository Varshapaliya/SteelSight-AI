import os
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "weights" / "best.pt"
METRICS_PATH = BASE_DIR / "evaluation_results" / "validation_metrics.json"

st.set_page_config(
    page_title="SteelSight AI | Stainless Spark 2026",
    page_icon="🔍",
    layout="wide",
)

# -------------------- Theme: retain the existing dark dashboard --------------------
st.markdown("""
<style>
:root {
  --bg: #101722;
  --panel: #182332;
  --panel-2: #1d2b3d;
  --border: #2b3d52;
  --text: #edf4fb;
  --muted: #a8b8ca;
  --blue: #58a6ff;
  --cyan: #55d6d0;
  --green: #49d7a0;
  --amber: #f2bd62;
  --red: #ff7b83;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { max-width: 1500px; padding-top: 1.25rem; padding-bottom: 2.5rem; }
[data-testid="stSidebar"] { background: #151e2a; border-right: 1px solid var(--border); }
h1, h2, h3, h4, p, label { color: var(--text); }
small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
.hero {
  padding: 25px 28px; border-radius: 18px;
  background: linear-gradient(120deg, #14243a 0%, #193c5c 58%, #14525e 100%);
  border: 1px solid #31516c; margin-bottom: 22px;
}
.eyebrow { color: #82e3df; font-size: .76rem; font-weight: 800; letter-spacing: .13em; }
.hero-title { color: #ffffff; font-size: 2.15rem; font-weight: 800; margin: 5px 0 4px; }
.hero-sub { color: #d2e4f3; margin: 0; font-size: .98rem; }
.pill {
  display: inline-block; padding: 5px 10px; margin-top: 14px;
  border: 1px solid #4b718c; border-radius: 999px;
  color: #d9f8f5; font-size: .75rem; font-weight: 650;
}
.kpi {
  background: linear-gradient(145deg, #1a2737, #182332);
  border: 1px solid var(--border); border-radius: 15px;
  padding: 17px 18px; min-height: 128px;
}
.kpi-label { color: var(--muted); font-size: .82rem; font-weight: 650; }
.kpi-value { color: #f5f9ff; font-size: 1.85rem; font-weight: 800; margin-top: 9px; }
.kpi-note { color: #91a6bc; font-size: .73rem; margin-top: 5px; line-height: 1.4; }
.section-card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 15px; padding: 17px 19px;
}
.section-label { color: #8fe3df; font-size: .76rem; font-weight: 800; letter-spacing: .1em; }
.status-good { color: var(--green); font-weight: 750; }
.status-review { color: var(--amber); font-weight: 750; }
div[data-testid="stMetric"] {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 13px; padding: 12px 14px;
}
div[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stMetricValue"] { color: var(--text); }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 700; min-height: 2.6rem;
}
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def read_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        required = ("precision", "recall", "map50", "map50_95")
        if not all(k in data for k in required):
            return {}
        return data
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def make_kpi(label, value, note):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------- Header --------------------
st.markdown("""
<div class="hero">
  <div class="hero-title">🔍 SteelSight AI</div>
  <p class="hero-sub">AI-powered surface defect detection for stainless steel strip</p>
  <span class="pill">DETECT &nbsp;→&nbsp; CLASSIFY &nbsp;→&nbsp; DECIDE</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Inspection Controls")
    st.caption("Local YOLO model · NEU-DET classes")
    confidence = st.slider(
        "Confidence gate",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.01,
        help="Predictions below this threshold are excluded by the detector. This is not a calibrated production QA threshold.",
    )
    st.caption(f"Detection threshold: {confidence:.0%}")
    st.divider()
    st.markdown("### Decision logic")
    st.markdown("**Above threshold**")
    st.caption("Show model prediction for human review.")
    st.markdown("**No detection**")
    st.caption("No box passed the selected threshold; this does not prove a defect-free surface.")
    st.divider()
    st.caption("Prototype status: image-level inspection")
    st.caption("Not connected to a production line or coil database.")

# -------------------- Model performance --------------------
st.markdown('<div class="section-label">01 / MODEL EVALUATION</div>', unsafe_allow_html=True)
st.subheader("Validation Performance")

metrics = read_metrics()
if metrics:
    cols = st.columns(4, gap="medium")
    values = [
        ("Precision", f"{metrics['precision'] * 100:.2f}%", "Positive prediction quality"),
        ("Recall", f"{metrics['recall'] * 100:.2f}%", "Ground-truth defects recovered"),
        ("mAP @ 50", f"{metrics['map50'] * 100:.2f}%", "Mean AP at IoU 0.50"),
        ("mAP @ 50–95", f"{metrics['map50_95'] * 100:.2f}%", "AP averaged over IoU thresholds"),
    ]
    for col, item in zip(cols, values):
        with col:
            make_kpi(*item)

    st.caption(
        f"Dataset: {metrics.get('dataset', 'NEU-DET')}  ·  "
        f"Split: {metrics.get('split', 'validation')}  ·  "
        f"Images: {metrics.get('images', 'N/A')}  ·  "
        f"Instances: {metrics.get('instances', 'N/A')}"
    )
    st.info(
        "These values were transcribed from the saved evaluation notebook; "
        "they have not been freshly reproduced on this machine. They describe "
        "dataset-level performance, not live plant quality or an individual image."
    )
else:
    st.warning("Validation metrics are unavailable or incomplete. Check evaluation_results/validation_metrics.json.")

st.divider()

# -------------------- Class performance --------------------
st.markdown('<div class="section-label">02 / DEFECT INTELLIGENCE</div>', unsafe_allow_html=True)
st.subheader("Performance Across the Six NEU-DET Classes")
st.caption("The presentation emphasizes precision, recall and class-wise evaluation rather than a single accuracy score.")

class_df = pd.DataFrame({
    "Defect": ["Crazing", "Inclusion", "Patches", "Pitted surface", "Rolled-in scale", "Scratches"],
    "Precision (%)": [59.8, 75.3, 82.5, 87.8, 64.7, 77.0],
    "Recall (%)": [34.4, 86.8, 95.8, 81.2, 65.5, 96.2],
    "mAP@50 (%)": [47.8, 90.1, 97.1, 87.9, 69.8, 97.2],
})

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.markdown("#### Precision vs Recall")
    pr = class_df.set_index("Defect")[["Precision (%)", "Recall (%)"]]
    st.bar_chart(pr, x_label="Defect class", y_label="Score (%)", height=330)
    st.caption("Fixed percentage scale: 0–100. Crazing recall is a key improvement area in the recorded results.")

with right:
    st.markdown("#### mAP@50 by class")
    ap = class_df.set_index("Defect")[["mAP@50 (%)"]]
    st.bar_chart(ap, x_label="Defect class", y_label="mAP@50 (%)", height=330)
    st.caption("Patches and scratches have the strongest recorded AP; crazing is the weakest.")

with st.expander("Open the full class-wise metrics table"):
    st.dataframe(
        class_df.style.format({
            "Precision (%)": "{:.1f}",
            "Recall (%)": "{:.1f}",
            "mAP@50 (%)": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# -------------------- Inspection workflow --------------------
st.markdown('<div class="section-label">03 / IMAGE INSPECTION</div>', unsafe_allow_html=True)
st.subheader("Detect → Classify → Review")
st.caption("Upload an image to inspect model predictions. Low-confidence detections are filtered by the selected threshold; production-grade unknown-defect detection is not implemented in this prototype.")

uploaded_file = st.file_uploader(
    "Choose a steel surface image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, JPEG, PNG and BMP.",
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be opened as a valid image.")
        st.stop()

    with st.expander("Image details", expanded=False):
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Dimensions:** {image.width} × {image.height} px")

    if st.button("Run Defect Detection", type="primary", use_container_width=True):
        if not MODEL_PATH.exists():
            st.error(f"Model weights not found: {MODEL_PATH}")
        else:
            try:
                with st.spinner("Detecting and classifying surface defects..."):
                    model = load_model()
                    start_time = time.perf_counter()
                    result = model.predict(
                        source=image,
                        conf=confidence,
                        imgsz=640,
                        verbose=False,
                    )[0]
                    inference_ms = (time.perf_counter() - start_time) * 1000

                boxes = result.boxes
                rows = []
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        score = float(box.conf[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        rows.append({
                            "Defect": result.names[class_id],
                            "Confidence (%)": round(score * 100, 2),
                            "X1": round(x1, 1),
                            "Y1": round(y1, 1),
                            "X2": round(x2, 1),
                            "Y2": round(y2, 1),
                        })

                image_col, output_col = st.columns(2, gap="large")
                with image_col:
                    st.markdown("#### Original surface")
                    st.image(image, use_container_width=True)
                with output_col:
                    st.markdown("#### Annotated prediction")
                    annotated = result.plot()
                    st.image(annotated[:, :, ::-1], use_container_width=True)

                st.markdown("#### Inspection summary")
                k1, k2, k3 = st.columns(3)
                k1.metric("Detected boxes", len(rows))
                k2.metric(
                    "Defect classes found",
                    len(set(r["Defect"] for r in rows)) if rows else 0,
                )
                k3.metric("Inference time", f"{inference_ms:.1f} ms*")

                if not rows:
                    st.warning(
                        "No detections passed the selected confidence threshold. "
                        "This is not a confirmed PASS result; a human should review the image."
                    )
                    st.markdown('<div class="section-card"><span class="status-review">DECISION: HUMAN REVIEW</span><br><br>No boxes passed the configured detector threshold. Do not interpret this as proof that the surface is defect-free.</div>', unsafe_allow_html=True)
                else:
                    df = pd.DataFrame(rows)
                    avg_conf = df["Confidence (%)"].mean()
                    st.markdown(
                        f'<div class="section-card"><span class="status-review">DECISION: REVIEW FLAGGED DEFECTS</span><br><br>'
                        f'{len(df)} detection(s) found · Average confidence {avg_conf:.1f}%. '
                        f'Model outputs require human confirmation before any quality decision.</div>',
                        unsafe_allow_html=True,
                    )

                    summary = (
                        df["Defect"].value_counts()
                        .rename_axis("Defect")
                        .reset_index(name="Detections")
                    )
                    chart_col, table_col = st.columns([0.8, 1.2], gap="large")
                    with chart_col:
                        st.markdown("#### Defect count by class")
                        st.bar_chart(summary.set_index("Defect"), y="Detections", height=300)
                    with table_col:
                        st.markdown("#### Bounding-box evidence")
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    st.download_button(
                        "Download inspection CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="steelsight_inspection_report.csv",
                        mime="text/csv",
                    )
                st.caption("*Measured wall-clock time for this local inference call; includes overhead and is not a production line-speed benchmark.")
            except Exception as exc:
                st.error("Detection failed. Check the terminal for details.")
                st.exception(exc)
else:
    st.markdown("""
    <div class="section-card">
      <div class="section-label">READY FOR INSPECTION</div>
      <h4>Upload a steel surface image</h4>
      <p style="color:#a8b8ca">The prototype will display the original image, annotated detections, defect classes, confidence scores, bounding-box coordinates and a downloadable report.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown(
    '<p style="color:#91a6bc;font-size:.82rem">SteelSight AI · Stainless Spark 2026 Round 2 · Research prototype. '
    'No live line feed, coil-level database, anomaly-screening model or certified QA decision is connected in this version.</p>',
    unsafe_allow_html=True,
)
