"""
Streamlit App: Damage Detection Pipeline (Simplified)
Purpose: Fast, simple visualization of damage detection pipeline
Run: streamlit run apps/pipeline_test.py
"""

import sys
import json
import time
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw

# ─── Setup paths ───────────────────────────────────────────────────────
BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))

from inference.detection_pipeline import DamageDetectionPipeline

# ─── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Damage Detection Pipeline",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Simple CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto; }
    .overlap-row { 
        padding: 10px; 
        border-bottom: 1px solid #eee;
        background: #f9f9f9;
    }
    .high { background: #d4edda !important; }
    .med { background: #fff3cd !important; }
    .low { background: #f8d7da !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar: Simple Configuration ──────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.caption("Associations are accepted only when overlap is strictly greater than 50%.")

# Model upload
damage_model_file = st.sidebar.file_uploader(
    "Damage Model (.pt)",
    type=["pt"],
    key="damage_model"
)

parts_model_file = st.sidebar.file_uploader(
    "Parts Model (.pt)",
    type=["pt"],
    key="parts_model"
)

# Thresholds
confidence_threshold = st.sidebar.slider(
    "Confidence", 0.0, 1.0, 0.25, 0.05
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold", 0.0, 1.0, 0.10, 0.05
)
imgsz = st.sidebar.selectbox(
    "Resolution", [320, 480, 640, 800, 1024], index=2
)

# ─── Load Pipeline ─────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_damage_path, _parts_path):
    if not Path(_damage_path).exists():
        return None, "❌ Damage model not found"
    if not Path(_parts_path).exists():
        return None, "❌ Parts model not found"
    try:
        return DamageDetectionPipeline(
            damage_model_path=str(_damage_path),
            part_model_path=str(_parts_path),
            confidence_threshold=0.25,
            iou_threshold=0.10,
            device="0",  # GPU device
            use_parallel_inference=True,  # ✅ PARALLEL INFERENCE
            use_half_precision=True,  # ✅ FP16 for speed
        ), None
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def get_model_paths():
    if damage_model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(damage_model_file.read())
            damage_path = tmp.name
    else:
        damage_path = str(BASE_PATH / "runs/damage_v0.1/weights/best.pt")
    
    if parts_model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(parts_model_file.read())
            parts_path = tmp.name
    else:
        parts_path = str(BASE_PATH / "runs/parts_v2/test_10epoch/weights/best.pt")
    
    return damage_path, parts_path

def draw_boxes_on_image(image, detections, color):
    """Draw bounding boxes on image copy."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for det in detections:
        # Convert normalized coords to pixel coords
        h, w = image.size[1], image.size[0]
        x1 = int(det.bbox.x1 * w)
        y1 = int(det.bbox.y1 * h)
        x2 = int(det.bbox.x2 * w)
        y2 = int(det.bbox.y2 * h)
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Draw label
        label = f"{det.class_name} {det.confidence:.0%}"
        draw.text((x1, y1 - 10), label, fill=color)
    
    return img_copy

# ─── Initialize ────────────────────────────────────────────────────────
st.markdown("# 🚗 Damage Detection Pipeline")
st.markdown("*Simplified view with segmentation visualization*")

damage_path, parts_path = get_model_paths()
pipeline, error = load_pipeline(damage_path, parts_path)

if error:
    st.error(error)
    st.stop()
    
st.sidebar.success("✅ Pipeline ready")

# ─── Upload & Process ──────────────────────────────────────────────────
st.subheader("📸 Upload Image")
uploaded_file = st.file_uploader("Choose car image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is None:
    st.info("👆 Upload an image to get started")
    st.stop()

# Load image
image = Image.open(uploaded_file).convert("RGB")
st.sidebar.success(f"✅ {uploaded_file.name}")

# Process
progress = st.empty()
progress.info("🔄 Processing...")

start = time.time()
pipeline.confidence_threshold = confidence_threshold
pipeline.iou_threshold = iou_threshold
pipeline.min_overlap_percentage = 50.0

try:
    result = pipeline.process(image, imgsz=imgsz)
    elapsed = (time.time() - start) * 1000
    progress.success(f"✅ Done in {elapsed:.0f}ms")
except Exception as e:
    progress.error(f"❌ {str(e)}")
    st.stop()

# ─── MAIN: 3-Column Segmented Visualization ────────────────────────────
st.subheader("👀 Segmentation View")

col1, col2, col3 = st.columns(3)

# Draw damage boxes (red)
damages_img = draw_boxes_on_image(image, result.damage_detections, "red")

# Draw parts boxes (blue)  
parts_img = draw_boxes_on_image(image, result.part_detections, "blue")

with col1:
    st.markdown("### 🔴 Damages (Red)")
    st.image(damages_img, use_container_width=True)
    st.caption(f"{len(result.damage_detections)} detected")

with col2:
    st.markdown("### 🔵 Parts (Blue)")
    st.image(parts_img, use_container_width=True)
    st.caption(f"{len(result.part_detections)} detected")

with col3:
    st.markdown("### 📷 Original")
    st.image(image, use_container_width=True)
    st.caption(f"{image.size[0]}×{image.size[1]}px")

# ─── ALL OVERLAPPING ASSOCIATIONS (MAIN RESULT) ─────────────────────────
st.subheader("🔗 All Overlapping Associations")

if len(result.associations) == 0:
    st.warning("⚠️ No overlapping damage-parts found. Adjust thresholds.")
else:
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Overlaps", len(result.associations))
    with col2:
        avg_iou = np.mean([a.iou for a in result.associations])
        st.metric("Avg IoU", f"{avg_iou:.1%}")
    with col3:
        avg_overlap = np.mean([a.overlap_percentage for a in result.associations])
        st.metric("Avg Overlap", f"{avg_overlap:.0f}%")
    with col4:
        st.metric("Process Time", f"{result.processing_time_ms:.0f}ms")
    
    # Table of all overlaps
    st.markdown("#### 📋 Overlap Table")
    
    table_data = []
    for assoc in result.associations:
        overlap_pct = assoc.overlap_percentage
        if overlap_pct > 75:
            status = "🟢 High"
        elif overlap_pct > 50:
            status = "🟡 Med"
        else:
            status = "🔴 Low"
        
        table_data.append({
            "Damage": assoc.damage_detection.class_name.upper(),
            "Part": assoc.part_detection.class_name,
            "Dmg%": f"{assoc.damage_detection.confidence:.0%}",
            "Part%": f"{assoc.part_detection.confidence:.0%}",
            "IoU": f"{assoc.iou:.2f}",
            "Overlap%": f"{overlap_pct:.0f}%",
            "Status": status
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    # Raw JSON download
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "💾 Download JSON",
            json.dumps(result.to_dict(), indent=2),
            f"results_{int(time.time())}.json",
            "application/json"
        )
    with col2:
        show_raw = st.checkbox("Show Raw JSON")
        if show_raw:
            st.json(result.to_dict())

# ─── Summary
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔴 Damages", len(result.damage_detections))
with col2:
    st.metric("🔵 Parts", len(result.part_detections))
with col3:
    st.metric("🔗 Overlaps", len(result.associations))
