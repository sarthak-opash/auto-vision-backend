import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="Model Comparison Tool", page_icon="🏎️", layout="wide")

st.title("🚗 Car Damage Model Comparison")
st.write("Upload one or more `.pt` models and an image to compare results.")

# --- SIDEBAR: MODEL UPLOADER ---
st.sidebar.header("Model Settings")
uploaded_models = st.sidebar.file_uploader(
    "Upload YOLO Models (.pt)", 
    type=["pt"], 
    accept_multiple_files=True
)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# --- MAIN: IMAGE UPLOADER ---
uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "png", "jpeg"])

if uploaded_file and uploaded_models:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Save image to temp path for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_img_path = tmp.name

    # Create a grid based on the number of models
    num_models = len(uploaded_models)
    cols = st.columns(num_models)

    # Process each model
    for idx, model_file in enumerate(uploaded_models):
        with cols[idx]:
            st.subheader(f"Model: {model_file.name}")
            
            # Save the uploaded .pt file to a temporary location so YOLO can load it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
                tmp_model.write(model_file.getvalue())
                tmp_model_path = tmp_model.name

            try:
                # Load and Run Inference
                model = YOLO(tmp_model_path)
                results = model.predict(source=temp_img_path, conf=confidence_threshold)
                
                # Plot and Display
                result_img = results[0].plot()
                st.image(result_img, use_container_width=True, channels="BGR")

                # Summary Statistics
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.info(f"Detected: {len(boxes)} issues")
                    for box in boxes:
                        label = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        st.caption(f"• {label} ({conf:.2f})")
                else:
                    st.success("No damage detected.")

            except Exception as e:
                st.error(f"Error loading {model_file.name}: {e}")
            
            # Cleanup model temp file
            os.remove(tmp_model_path)

    # Cleanup image temp file
    os.remove(temp_img_path)

elif not uploaded_models:
    st.info("Please upload at least one `.pt` model file in the sidebar to begin.")