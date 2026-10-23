
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(
    page_title="SteelSight AI",
    page_icon="🔍",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "weights" / "best.pt"

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

st.title("SteelSight AI")
st.caption("AI-powered steel surface defect detection")

with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    st.info("Model: YOLOv8 Nano")

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Detect Defects", type="primary"):
        if not MODEL_PATH.exists():
            st.error(f"Model file not found: {MODEL_PATH}")
        else:
            with st.spinner("Analysing steel surface..."):
                model = load_model()
                result = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    verbose=False,
                )[0]

            with right:
                st.subheader("AI Detection Result")
                annotated = result.plot()
                st.image(
                    annotated[:, :, ::-1],
                    use_container_width=True,
                )

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                st.warning(
                    "No defects detected above this confidence "
                    "threshold. This does not guarantee a "
                    "defect-free surface."
                )
            else:
                rows = []

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

                st.subheader("Detection Summary")

                m1, m2 = st.columns(2)
                m1.metric("Detected defects", len(rows))
                m2.metric(
                    "Average confidence",
                    f"{sum(r['Confidence (%)'] for r in rows) / len(rows):.1f}%",
                )

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "Download CSV report",
                    df.to_csv(index=False),
                    file_name="steelsight_report.csv",
                    mime="text/csv",
                )

st.divider()
st.caption(
    "Educational prototype. Predictions may be incorrect; "
    "not a certified industrial inspection system."
)



