# ==========================================================
# FILE NAME : cost_estimation_app.py
# PURPOSE   : Full pipeline test app for AutoClaim Vision
#
# FLOW:
# 1. Upload Image
# 2. Run damage detection  (damage best.pt)
# 3. Run part detection    (parts best.pt, optional)
# 4. Run severity engine   (severity.py)
# 5. Run cost estimator    (cost_estimation.py)
# 6. Show per-part cost + totals (only damaged parts shown)
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

import train.severity as severity_engine
import train.cost_estimation as cost_engine
severity_engine = importlib.reload(severity_engine)
cost_engine     = importlib.reload(cost_engine)


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def boxes_to_rows(boxes, names):
    rows = []
    for box in boxes:
        class_id = int(box.cls[0])
        rows.append({
            "class":      names[class_id],
            "confidence": float(box.conf[0]),
            "bbox":       [float(v) for v in box.xyxy[0].tolist()],
        })
    return rows


def severity_color(level: str) -> str:
    return {
        "Low":      "🟢",
        "Medium":   "🟡",
        "High":     "🟠",
        "Critical": "🔴",
    }.get(level, "⚪")


# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="AutoClaim Vision - Cost Estimator",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AutoClaim Vision - Cost Estimator")
st.caption("Upload a damaged car image to get repair cost estimate.")

# ─────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────

# MODEL_PATH      = os.path.join(BASE_PATH, "runs", "damage", "weights", "best.pt")
MODEL_PATH      = os.path.join(BASE_PATH, "runs", "damage_seg_v1", "epoch_150", "weights", "best.pt")
PART_MODEL_PATH = os.path.join(BASE_PATH, "runs", "parts",  "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Damage model not found at: " + MODEL_PATH)
    st.stop()

if not os.path.exists(PART_MODEL_PATH):
    st.warning("⚠️ Part model not found. Severity falls back to damage-box area.")

model      = YOLO(MODEL_PATH)
part_model = YOLO(PART_MODEL_PATH) if os.path.exists(PART_MODEL_PATH) else None

# ─────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────

uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "jpeg", "png"])

# ─────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────

if uploaded_file:
    image     = Image.open(uploaded_file).convert("RGB")
    temp_path = None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    try:
        # save temp for YOLO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # ── STEP 1: DAMAGE DETECTION
        results = model.predict(source=temp_path, conf=0.25, imgsz=640)

        # ── STEP 2: PART DETECTION (optional)
        part_results = []
        if part_model:
            part_results = part_model.predict(source=temp_path, conf=0.25, imgsz=640)

        with col2:
            st.subheader("Detected Damage")
            st.image(results[0].plot(), use_container_width=True, channels="BGR")

        st.markdown("---")

        boxes      = results[0].boxes
        part_boxes = part_results[0].boxes if part_results else []
        part_rows  = boxes_to_rows(part_boxes, part_model.names) if part_boxes else []

        if len(boxes) == 0:
            st.info("No damage detected in image.")
            st.stop()

        detections = boxes_to_rows(boxes, model.names)

        # ── STEP 3: SEVERITY
        report = severity_engine.generate_severity_report(
            detections, image.width, image.height, part_rows
        )

        # ── STEP 4: COST ESTIMATION
        cost_report = cost_engine.estimate_cost(report["part_severity"])

        # ══════════════════════════════════════
        # SECTION 1 — SEVERITY SUMMARY
        # ══════════════════════════════════════
        st.subheader("Severity Summary")
        s_col1, s_col2, s_col3 = st.columns(3)

        lvl = report["severity_level"]
        s_col1.metric("Overall Score",  f"{report['severity_score']} / 100")
        s_col2.metric("Severity Level", f"{severity_color(lvl)} {lvl}")
        s_col3.metric("Parts Damaged",  len(report["detected_parts"]))

        # critical flags
        if report["critical_flags"]:
            st.markdown("**⚠️ Critical Flags**")
            for flag in report["critical_flags"]:
                st.warning(flag)

        # ══════════════════════════════════════
        # SECTION 2 — COST BREAKDOWN (per damaged part)
        # WHY only damaged parts shown:
        #   cost_engine.estimate_cost() only receives part_severity,
        #   which already contains ONLY parts with detected damage.
        #   Undamaged parts never enter the pipeline.
        # ══════════════════════════════════════
        st.markdown("---")
        st.subheader("Repair Cost Breakdown")
        st.caption(f"⚠️ {cost_report['note']}")

        if cost_report["line_items"]:
            display_rows = []
            for item in cost_report["line_items"]:
                display_rows.append({
                    "Part":           item["part"],
                    "Severity":       f"{severity_color(item['severity_level'])} {item['severity_level']} ({item['severity_score']})",
                    "Damage Types":   item["damage_types"],
                    "Repair Action":  item["repair_action"],
                    "Est. Cost (₹)":  f"₹ {item['part_cost']:,.0f}",
                })
            st.dataframe(display_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No cost data available.")

        # ══════════════════════════════════════
        # SECTION 3 — TOTALS
        # ══════════════════════════════════════
        st.markdown("---")
        st.subheader("Cost Summary")

        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("Parts Total",  f"₹ {cost_report['parts_total']:,.0f}")
        t_col2.metric("Labor (20%)",  f"₹ {cost_report['labor_total']:,.0f}")
        t_col3.metric("Grand Total",  f"₹ {cost_report['grand_total']:,.0f}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
