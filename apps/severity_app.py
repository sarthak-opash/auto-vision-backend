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

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# import your engine
from train.severity import calculate_severity

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
MODEL_PATH = r"../runs/damage/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Damage model not found.")
    st.stop()

model = YOLO(MODEL_PATH)

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

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

    plotted = results[0].plot()

    with col2:
        st.subheader("Detected Damage")
        st.image(plotted, use_container_width=True)

    # ======================================================
    # ANALYSIS
    # ======================================================
    st.markdown("---")
    st.subheader("🚨 Severity Analysis")

    boxes = results[0].boxes

    if len(boxes) == 0:
        st.success("No damage detected.")

    else:

        final_score = 0
        highest_level = "Minor"

        # fake part for now (future part model)
        default_part = "bumper"

        severity_order = {
            "Minor": 1,
            "Moderate": 2,
            "Severe": 3,
            "Critical": 4
        }

        for i, box in enumerate(boxes):

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            damage_label = model.names[cls]

            result = calculate_severity(
                damage_label=damage_label,
                part_label=default_part,
                box=box,
                image_width=image.width,
                image_height=image.height,
                total_damages=len(boxes)
            )

            final_score += result["score"]

            if severity_order[result["severity"]] > severity_order[highest_level]:
                highest_level = result["severity"]

            st.write(f"### Damage {i+1}")
            st.write(f"Damage Type: {damage_label}")
            st.write(f"Confidence: {conf:.2f}")
            st.write(f"Severity: {result['severity']}")
            st.write(f"Score: {result['score']}")
            st.write(f"Area: {result['damage_percent']}%")
            st.markdown("---")

        # ==================================================
        # FINAL RESULT
        # ==================================================
        st.subheader("📌 Final Vehicle Severity")

        if highest_level == "Minor":
            st.success(f"{highest_level}")

        elif highest_level == "Moderate":
            st.warning(f"{highest_level}")

        elif highest_level == "Severe":
            st.error(f"{highest_level}")

        else:
            st.error(f"{highest_level}")

        st.write(f"Total Combined Score: {final_score}")

    os.remove(temp_path)