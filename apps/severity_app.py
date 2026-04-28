# ==========================================================
# FILE NAME : severity_app.py
# PURPOSE :
# Test App for AutoClaim Vision Severity Engine
#
# FLOW:
# 1. Upload Image
# 2. Detect Damage using damage best.pt
# 3. Apply severity.py logic
# 4. Show Final Severity Output
# ==========================================================

import os
import sys
import tempfile
import importlib
from PIL import Image
import streamlit as st
from ultralytics import YOLO

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

# import your engine
import train.severity as severity_engine
severity_engine = importlib.reload(severity_engine)


def boxes_to_rows(boxes, names):
    rows = []

    for box in boxes:
        class_id = int(box.cls[0])
        rows.append(
            {
                "class": names[class_id],
                "confidence": float(box.conf[0]),
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
            }
        )

    return rows


# ==========================================================
# PAGE SETTINGS
# ==========================================================
st.set_page_config(
    page_title="AutoClaim Vision - Severity Test",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AutoClaim Vision - Severity Test App")
st.write("Upload damaged car image to test severity engine.")

# ==========================================================
# LOAD MODEL
# ==========================================================
MODEL_PATH = os.path.join(BASE_PATH, "runs", "damage", "weights", "best.pt")
PART_MODEL_PATH = os.path.join(BASE_PATH, "runs", "parts", "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    st.error("Damage model not found.")
    st.stop()

if not os.path.exists(PART_MODEL_PATH):
    st.warning("Part model not found. Severity will fall back to damage-box area only.")

model = YOLO(MODEL_PATH)
part_model = YOLO(PART_MODEL_PATH) if os.path.exists(PART_MODEL_PATH) else None

# ==========================================================
# UPLOAD IMAGE
# ==========================================================
uploaded_file = st.file_uploader(
    "Upload Car Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# MAIN
# ==========================================================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    temp_path = None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width='stretch')

    try:
        # save temp image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # prediction
        results = model.predict(
            source=temp_path,
            conf=0.25,
            imgsz=640
        )

        part_results = []
        if part_model is not None:
            part_results = part_model.predict(
                source=temp_path,
                conf=0.25,
                imgsz=640
            )

        plotted = results[0].plot()

        with col2:
            st.subheader("Detected Damage")
            st.image(plotted, width='stretch', channels="BGR")

        # ======================================================
        # ANALYSIS
        # ======================================================
        st.markdown("---")
        st.subheader("Severity Analysis")

        boxes = results[0].boxes
        part_boxes = part_results[0].boxes if part_results else []
        part_rows = boxes_to_rows(part_boxes, part_model.names) if part_boxes else []

        if len(boxes) == 0:
            st.info("No damage detected.")
            st.stop()

        detections = boxes_to_rows(boxes, model.names)

        report = severity_engine.generate_severity_report(detections, image.width, image.height, part_rows)

        st.markdown("### Severity Summary")
        summary_rows = [
            {"metric": "severity_score", "value": report["severity_score"]},
            {"metric": "severity_level", "value": report["severity_level"]},
            {"metric": "detected_parts", "value": ", ".join(report["detected_parts"]) if report["detected_parts"] else "None"},
            {"metric": "critical_flags", "value": ", ".join(report["critical_flags"]) if report["critical_flags"] else "None"},
        ]
        st.dataframe(summary_rows, width='stretch', hide_index=True)

        st.markdown("### Damage Table")
        st.dataframe(report["damage_table"], width='stretch', hide_index=True)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)