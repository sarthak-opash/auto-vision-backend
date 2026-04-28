# ===============================
# File: apps/cost_estimator_app.py
# Put inside apps folder
# Run:
# streamlit run apps/cost_estimator_app.py
# ===============================

import os
import sys
import tempfile
import streamlit as st
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from train.cost_estimator_chatgpt import estimate_cost, format_inr

st.set_page_config(page_title="AutoClaim Vision Cost Estimator", layout="wide")

st.title("🚗 AutoClaim Vision - Cost Estimator")
st.write("Upload damaged car image to estimate repair cost")

file = st.file_uploader("Upload Car Image", type=["jpg","jpeg","png"])

if file:

    image = Image.open(file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    result = estimate_cost(img_path)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.image(result["plot"], caption="Detected Damage", channels="BGR", use_container_width=True)

    st.markdown("---")
    st.subheader("Detected Damages")

    for item in result["detections"]:
        st.write(
            f"✅ {item['name']} | {item['severity']} | "
            f"{format_inr(item['min'])} - {format_inr(item['max'])}"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Minimum Cost", format_inr(result["total_min"]))

    with col2:
        st.metric("Maximum Cost", format_inr(result["total_max"]))

    st.info("Estimated Indian local garage repair prices. Final price depends on city, car brand, OEM parts and labour.")
