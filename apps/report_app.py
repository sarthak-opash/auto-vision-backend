# ==========================================================
# FILE NAME : report_app.py
# PURPOSE   : Full pipeline + PDF report for AutoClaim Vision
#
# FLOW:
# 1. Upload Image
# 2. Damage detection  (damage best.pt)
# 3. Part detection    (parts best.pt, optional)
# 4. Severity engine   (severity.py)
# 5. Cost estimation   (cost_estimation.py)
# 6. Generate PDF      (report.py)
# 7. Download button
# ==========================================================

import os
import sys
import tempfile
import importlib
from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import datetime

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

import train.severity        as severity_engine
import train.cost_estimation as cost_engine
import train.report          as report_engine

severity_engine = importlib.reload(severity_engine)
cost_engine     = importlib.reload(cost_engine)
report_engine   = importlib.reload(report_engine)


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


def severity_icon(level: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}.get(level, "⚪")


# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="AutoClaim Vision - Report",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AutoClaim Vision - Inspection Report")
st.caption("Upload a damaged car image to run the full pipeline and download a PDF report.")

# ─────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────

MODEL_PATH      = os.path.join(BASE_PATH, "runs", "damage", "weights", "best.pt")
PART_MODEL_PATH = os.path.join(BASE_PATH, "runs", "parts",  "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    st.error("Damage model not found: " + MODEL_PATH)
    st.stop()

if not os.path.exists(PART_MODEL_PATH):
    st.warning("Part model not found. Severity falls back to damage-box area.")

model      = YOLO(MODEL_PATH)
part_model = YOLO(PART_MODEL_PATH) if os.path.exists(PART_MODEL_PATH) else None

# ─────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────

uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "jpeg", "png"])

# ─────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────

if uploaded_file:
    image      = Image.open(uploaded_file).convert("RGB")
    temp_path  = None
    annot_path = None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    try:
        # save original temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # ── STEP 1: DETECT
        results = model.predict(source=temp_path, conf=0.25, imgsz=640)

        part_results = []
        if part_model:
            part_results = part_model.predict(source=temp_path, conf=0.25, imgsz=640)

        # save annotated image for PDF
        annotated_img = results[0].plot()   # numpy BGR
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as atmp:
            Image.fromarray(annotated_img[:, :, ::-1]).save(atmp.name)
            annot_path = atmp.name

        with col2:
            st.subheader("Detected Damage")
            st.image(annotated_img, use_container_width=True, channels="BGR")

        st.markdown("---")

        boxes      = results[0].boxes
        part_boxes = part_results[0].boxes if part_results else []
        part_rows  = boxes_to_rows(part_boxes, part_model.names) if part_boxes else []

        if len(boxes) == 0:
            st.info("No damage detected in image.")
            st.stop()

        detections = boxes_to_rows(boxes, model.names)

        # ── STEP 2: SEVERITY
        with st.spinner("Running severity engine..."):
            report = severity_engine.generate_severity_report(
                detections, image.width, image.height, part_rows
            )

        # ── STEP 3: COST
        with st.spinner("Estimating repair cost..."):
            cost_report = cost_engine.estimate_cost(report["part_severity"])

        # ══════════════════════════════════════
        # SEVERITY SUMMARY
        # ══════════════════════════════════════
        st.subheader("Severity Summary")
        lvl = report["severity_level"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Score",  f"{report['severity_score']} / 100")
        c2.metric("Severity Level", f"{severity_icon(lvl)} {lvl}")
        c3.metric("Parts Damaged",  len(report["detected_parts"]))

        if report["critical_flags"]:
            st.markdown("**Critical Flags**")
            for flag in report["critical_flags"]:
                st.warning(flag)

        # ══════════════════════════════════════
        # COST BREAKDOWN
        # ══════════════════════════════════════
        st.markdown("---")
        st.subheader("Repair Cost Breakdown")
        st.caption(f"Note: {cost_report['note']}")

        if cost_report["line_items"]:
            display_rows = []
            for item in cost_report["line_items"]:
                display_rows.append({
                    "Part":           item["part"],
                    "Severity":       f"{severity_icon(item['severity_level'])} {item['severity_level']}",
                    "Repair Action":  item["repair_action"],
                    "Damage Types":   item["damage_types"],
                    "Cost (Rs.)":     f"Rs. {item['part_cost']:,.0f}",
                })
            st.dataframe(display_rows, use_container_width=True, hide_index=True)

        st.markdown("---")
        t1, t2, t3 = st.columns(3)
        t1.metric("Parts Total", f"Rs. {cost_report['parts_total']:,.0f}")
        t2.metric("Labour (20%)", f"Rs. {cost_report['labor_total']:,.0f}")
        t3.metric("Grand Total",  f"Rs. {cost_report['grand_total']:,.0f}")

        # ══════════════════════════════════════
        # GENERATE + DOWNLOAD PDF
        # ══════════════════════════════════════
        st.markdown("---")
        st.subheader("Download Report")

        with st.spinner("Generating PDF..."):
            pdf_bytes = report_engine.generate_report(
                severity_result=report,
                cost_result=cost_report,
                annotated_image_path=annot_path,
            )

        filename = f"autoclaim_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )

    finally:
        for p in [temp_path, annot_path]:
            if p and os.path.exists(p):
                os.remove(p)


# need datetime for filename
import datetime
