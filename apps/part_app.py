# ==========================================================
# FILE NAME : app_parts.py
# PURPOSE : Streamlit App for Car Parts Detection Model
# ==========================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ----------------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------------
st.set_page_config(
    page_title="Car Parts Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Parts Detection System")
st.write("Upload a car image to detect visible car parts.")

# ----------------------------------------------------------
# LOAD MODEL
# Change path if needed after training completes
# ----------------------------------------------------------
MODEL_PATH = r"../runs/parts/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please check path:")
    st.code(MODEL_PATH)
    st.stop()

model = YOLO(MODEL_PATH)

# ----------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Car Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------------------------------
# DETECTION
# ----------------------------------------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Predict
    results = model.predict(
        source=temp_path,
        conf=0.25,
        imgsz=640
    )

    result = results[0]
    plotted = result.plot()

    with col2:
        st.subheader("Detected Parts")
        st.image(plotted, use_container_width=True, channels="BGR")

    # ------------------------------------------------------
    # SHOW DETECTED PARTS LIST
    # ------------------------------------------------------
    st.subheader("Detected Car Parts")

    boxes = result.boxes

    if len(boxes) == 0:
        st.warning("No car parts detected.")
    else:
        shown = set()

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls_id]

            # avoid duplicate spam
            key = f"{label}"

            if key not in shown:
                shown.add(key)
                st.success(f"{label}  | Confidence: {conf:.2f}")

    # cleanup
    os.remove(temp_path)