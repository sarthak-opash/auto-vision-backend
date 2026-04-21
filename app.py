import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="Car Damage Detection", layout="wide")

st.title("🚗 Car Damage Detection System")

model = YOLO("runs/detect/train/weights/best.pt")

uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)

    results = model.predict(source=temp_file.name, conf=0.25)

    result_img = results[0].plot()

    st.image(result_img, caption="Detected Damage", use_column_width=True)

    boxes = results[0].boxes

    if len(boxes) > 0:
        st.subheader("Detected Damages:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            st.write(f"✅ {label}  | Confidence: {conf:.2f}")
    else:
        st.success("No Damage Found")