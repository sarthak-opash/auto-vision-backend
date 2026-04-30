"""
Vehicle Model Identification Streamlit App
Identifies car type from uploaded vehicle images using ResNet50 classifier
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from train.model_identity import DamagedCarClassifier, get_transforms, Config

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🚗 Vehicle Model Identification",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🚗 Vehicle Model Identification")
st.markdown("---")

# ============================================================================
# MODEL LOADING (Cached)
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model checkpoint"""
    checkpoint_path = Path(__file__).resolve().parent.parent / 'runs' / 'model_identity' / 'best_model_phase1.pth'
    
    if not checkpoint_path.exists():
        st.error(f"❌ Checkpoint not found at {checkpoint_path}")
        return None, None
    
    try:
        # Load configuration
        config = Config()
        
        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DamagedCarClassifier(num_classes=config.num_classes, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict_car_type(model, image, device):
    """
    Predict car type from image
    
    Args:
        model: Loaded model
        image: PIL Image object
        device: Device to run inference on
    
    Returns:
        Predicted class, confidence score, and top-3 predictions
    """
    try:
        # Preprocess image
        transform = get_transforms('val')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get class name
        config = Config()
        predicted_class = config.class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get top-3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, min(3, len(config.class_names)))
        top3_predictions = [
            (config.class_names[idx.item()], prob.item() * 100)
            for idx, prob in zip(top3_indices[0], top3_prob[0])
        ]
        
        return predicted_class, confidence_score, top3_predictions
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Load model
model, device = load_model()

if model is not None:
    # Step 1: Image Upload
    st.subheader("📸 Step 1: Upload Vehicle Image")
    st.markdown("Upload a clear photo of the vehicle")
    
    uploaded_file = st.file_uploader(
        "Choose an image file:",
        type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Vehicle Image", use_column_width=True)
        
        # Step 2: Run Prediction
        st.subheader("🔍 Step 2: Analyzing Vehicle...")
        
        with st.spinner("Running model inference..."):
            predicted_class, confidence_score, top3_predictions = predict_car_type(model, image, device)
        
        if predicted_class is not None:
            # Step 3: Display Results
            st.subheader("✅ Prediction Results")
            
            # Main prediction with prominent display
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Predicted Vehicle Type",
                    value=predicted_class.upper(),
                )
            with col2:
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence_score:.2f}%"
                )
            
            st.markdown("---")
            
            # Top-3 predictions
            st.subheader("🏆 Top 3 Predictions")
            
            prediction_data = []
            for rank, (car_type, prob) in enumerate(top3_predictions, 1):
                prediction_data.append({
                    'Rank': rank,
                    'Vehicle Type': car_type.capitalize(),
                    'Confidence': f"{prob:.2f}%"
                })
            
            st.dataframe(prediction_data, use_container_width=True)
            
            # Confidence visualization
            st.markdown("---")
            st.subheader("📊 Confidence Distribution")
            
            config = Config()
            # Create a bar chart showing all class predictions
            with torch.no_grad():
                outputs = model(get_transforms('val')(image).unsqueeze(0).to(device))
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            prob_dict = {
                config.class_names[i]: probabilities[0][i].item() * 100
                for i in range(len(config.class_names))
            }
            
            st.bar_chart(prob_dict)
        
        st.markdown("---")
        st.info(
            "💡 **Tip:** For best results, ensure the vehicle is clearly visible and well-lit.",
            icon="ℹ️"
        )
    
    else:
        st.info("👆 Please upload a vehicle image to get started", icon="ℹ️")
        
        # Show example info
        with st.expander("📖 How to use this app"):
            st.markdown("""
            1. **Upload Image**: Choose a clear photo of a vehicle
            2. **Wait for Analysis**: The model will process the image
            3. **View Results**: See the predicted vehicle type and confidence score
            
            **Supported Vehicle Types:**
            - Sedan
            - SUV
            - Truck
            - Coupe
            - Hatchback
            - Van
            - Sports
            """)

else:
    st.error(
        "❌ Failed to load model. Please ensure the checkpoint file exists at:\n"
        "D:\\auto-vison-backend\\runs\\model_identity\\best_model_phase1.pth"
    )
