import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Set page to wide mode to match the parts app
st.set_page_config(page_title="Car Damage Detection", page_icon="🛠️", layout="wide")

st.title("🚗 Car Damage Detection System")
st.write("Upload an image to identify specific areas of vehicle damage.")

# Load the model
MODEL_PATH = "../runs/damage/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at: {MODEL_PATH}")
    st.stop()

model = YOLO(MODEL_PATH)

uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    # Open and convert image to maintain consistency
    image = Image.open(uploaded_file).convert("RGB")

    # Create two columns for side-by-side view
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        # use_container_width prevents the "zoomed-in" look and fits the column
        st.image(image, use_container_width=True)

    # Save to a temporary file for the YOLO model to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run inference
    results = model.predict(source=temp_path, conf=0.25)
    
    # Process results
    result_img = results[0].plot()

    with col2:
        st.subheader("Detected Damage")
        # Display the processed image in the second column
        st.image(result_img, use_container_width=True, channels="BGR")

    # Damage Report Section
    st.markdown("---")
    st.subheader("Damage Analysis Report")

    boxes = results[0].boxes

    if len(boxes) > 0:
        # Using a set to keep the list clean if multiple same-type damages are found
        detected_items = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            detected_items.append(f"**{label.upper()}** | Confidence: `{conf:.2f}`")
        
        # Display as a clean list
        for item in list(set(detected_items)):
            st.write(f"✅ {item}")
    else:
        st.success("No visible damage detected.")

    # Cleanup temporary file
    os.remove(temp_path)