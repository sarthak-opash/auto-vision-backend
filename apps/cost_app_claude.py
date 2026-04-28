# =============================================================
# FILE NAME : cost_app.py
# PURPOSE   :
# AutoClaim Vision — Stage 3: Repair Cost Estimator
#
# FLOW:
# 1. Upload car damage photo
# 2. YOLO (best.pt) detects all damage boxes
# 3. severity.py scores each detection → Minor/Moderate/Severe/Critical
# 4. cost_estimate.py maps class + severity → INR cost range
# 5. Show annotated image + per-damage cost table + grand total
#
# RUN:
#   streamlit run apps/cost_app.py
# (run from project root folder)
# =============================================================

import os
import sys
import tempfile

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── path setup so train/ modules are importable ──────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from train.severity import calculate_severity
from train.cost_estimate_claude import build_estimate, format_inr, DISCLAIMER

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="AutoClaim Vision — Cost Estimator",
    page_icon="🚗",
    layout="wide",
)

# =============================================================
# CUSTOM CSS
# =============================================================
st.markdown("""
<style>
    .header-title { font-size:2rem; font-weight:700; color:#1a1a2e; }
    .header-sub   { color:#666; font-size:0.9rem; margin-bottom:1rem; }

    .sev-minor    { background:#e8f5e9; color:#2e7d32; border:1.5px solid #2e7d32;
                    padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.8rem; }
    .sev-moderate { background:#fff3e0; color:#e65100; border:1.5px solid #e65100;
                    padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.8rem; }
    .sev-severe   { background:#fce4ec; color:#b71c1c; border:1.5px solid #b71c1c;
                    padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.8rem; }
    .sev-critical { background:#ede7f6; color:#4a148c; border:1.5px solid #4a148c;
                    padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.8rem; }

    .total-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white; padding: 1.5rem 2rem;
        border-radius: 14px; margin-top: 0.5rem;
    }
    .total-card .label { color:#aaa; font-size:0.85rem; margin:0; }
    .total-card .amount { color:#f5c518; font-size:2rem; font-weight:700; margin:4px 0; }
    .total-card .sub { color:#ccc; font-size:0.8rem; }

    .disclaimer {
        background:#fff8e1; border-left:4px solid #f9a825;
        padding:0.75rem 1rem; border-radius:6px;
        font-size:0.8rem; color:#555; margin-top:1.5rem;
    }

    div[data-testid="metric-container"] {
        background:#f8f9fa; border-radius:10px; padding:0.5rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# HEADER
# =============================================================
st.markdown('<p class="header-title">🚗 AutoClaim Vision — Repair Cost Estimator</p>',
            unsafe_allow_html=True)
st.markdown('<p class="header-sub">Upload a damage photo · AI detects damage · Get Indian market cost estimate</p>',
            unsafe_allow_html=True)
st.divider()

# =============================================================
# LOAD MODEL
# =============================================================
MODEL_PATH = os.path.join(BASE, "runs", "damage", "weights", "best.pt")

# fallback to v1 if v2 not found
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE, "runs", "damage", "weights", "best.pt")

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at: `{MODEL_PATH}`")
    st.info("Make sure you run this app from the project root folder.")
    st.stop()

model = load_model(MODEL_PATH)

# =============================================================
# SIDEBAR — SETTINGS
# =============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    conf_thresh = st.slider(
        "Detection Confidence Threshold",
        min_value=0.10, max_value=0.90,
        value=0.25, step=0.05,
        help="Lower = detects more (may include false positives). Higher = only confident detections."
    )

    st.markdown("---")
    st.markdown("**Model:** `damage_v2 / best.pt`")
    st.markdown("**Classes:** 20 damage types")
    st.markdown("**Prices:** Indian market 2024-25")
    st.markdown("---")
    st.caption("Rates cover economy & mid-segment cars. SUV/luxury: 1.5x-2.5x more.")

# =============================================================
# FILE UPLOAD
# =============================================================
uploaded = st.file_uploader(
    "📤 Upload Car Image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a photo of the damaged vehicle."
)

# =============================================================
# MAIN LOGIC
# =============================================================
if uploaded is None:
    st.info("👆 Upload a car image to get started.")
    st.stop()

# open image
image = Image.open(uploaded).convert("RGB")
img_w, img_h = image.size

# save to temp file for YOLO
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    image.save(tmp.name)
    tmp_path = tmp.name

# run inference
with st.spinner("🔍 Detecting damage..."):
    results = model.predict(source=tmp_path, conf=conf_thresh, imgsz=640, verbose=False)

os.remove(tmp_path)

# ── Display images ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)

annotated = results[0].plot()
with col2:
    st.subheader("🔍 Detected Damage")
    st.image(annotated, use_container_width=True, channels="BGR")

st.divider()

# ── Process detections ────────────────────────────────────────────────────────
boxes = results[0].boxes

if len(boxes) == 0:
    st.success("✅ No damage detected in this image.")
    st.stop()

SEVERITY_ORDER = {"Minor": 1, "Moderate": 2, "Severe": 3, "Critical": 4}

detections_for_cost: list[dict] = []
analysis_rows: list[dict] = []
overall_severity = "Minor"

for box in boxes:
    cls_id     = int(box.cls[0])
    conf       = float(box.conf[0])
    class_name = model.names[cls_id]

    sev_result = calculate_severity(
        damage_label=class_name,
        part_label="bumper",           # default; future: use parts model
        box=box,
        image_width=img_w,
        image_height=img_h,
        total_damages=len(boxes),
    )

    severity = sev_result["severity"]

    if SEVERITY_ORDER[severity] > SEVERITY_ORDER[overall_severity]:
        overall_severity = severity

    detections_for_cost.append({
        "class_name": class_name,
        "severity":   severity,
        "confidence": conf,
    })

    analysis_rows.append({
        "class_name":    class_name,
        "severity":      severity,
        "score":         sev_result["score"],
        "confidence":    conf,
        "area_pct":      sev_result["damage_percent"],
    })

# build cost estimate
estimate = build_estimate(detections_for_cost)

# =============================================================
# SUMMARY METRICS
# =============================================================
st.subheader("📊 Detection Summary")

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Damages Found",   len(boxes))
mc2.metric("Unique Classes",  len(estimate["line_items"]))
mc3.metric("Overall Severity", overall_severity)
mc4.metric("Est. Total (Min)", format_inr(estimate["total_min"]))

st.divider()

# =============================================================
# PER-DAMAGE COST TABLE
# =============================================================
st.subheader("🔧 Per-Damage Cost Breakdown")

SEV_CLASS = {
    "Minor":    "sev-minor",
    "Moderate": "sev-moderate",
    "Severe":   "sev-severe",
    "Critical": "sev-critical",
}

# header row
h1, h2, h3, h4, h5 = st.columns([3, 1.5, 1.5, 2, 2])
h1.markdown("**Damage Class**")
h2.markdown("**Severity**")
h3.markdown("**Conf.**")
h4.markdown("**Min Cost**")
h5.markdown("**Max Cost**")
st.markdown("---")

for item in estimate["line_items"]:
    c1, c2, c3, c4, c5 = st.columns([3, 1.5, 1.5, 2, 2])
    c1.write(item["class_name"])
    sev = item["severity"]
    c2.markdown(f'<span class="{SEV_CLASS[sev]}">{sev}</span>', unsafe_allow_html=True)
    c3.write(f"{item['confidence']:.0%}")
    c4.write(format_inr(item["min_cost"]))
    c5.write(format_inr(item["max_cost"]))

st.divider()

# =============================================================
# COST SUMMARY
# =============================================================
st.subheader("💰 Cost Summary")

left, right = st.columns([1, 1])

with left:
    rows = [
        ("Parts & Paint Subtotal", estimate["subtotal_min"], estimate["subtotal_max"]),
        ("Labour  (15% – 22%)",    estimate["labour_min"],   estimate["labour_max"]),
    ]
    rh1, rh2, rh3 = st.columns([3, 2, 2])
    rh1.markdown("**Item**"); rh2.markdown("**Min**"); rh3.markdown("**Max**")
    for label, mn, mx in rows:
        r1, r2, r3 = st.columns([3, 2, 2])
        r1.write(label)
        r2.write(format_inr(mn))
        r3.write(format_inr(mx))

with right:
    st.markdown(
        f"""<div class="total-card">
            <p class="label">Estimated Total Repair Cost</p>
            <p class="amount">{format_inr(estimate['total_min'])} – {format_inr(estimate['total_max'])}</p>
            <p class="sub">Parts + Paint + Labour &nbsp;|&nbsp; {overall_severity} severity</p>
            <p class="sub" style="margin-top:6px">Economy / mid-segment cars · Indian garage rates</p>
        </div>""",
        unsafe_allow_html=True,
    )

# =============================================================
# BAR CHART
# =============================================================
if len(estimate["line_items"]) > 1:
    st.divider()
    st.subheader("📈 Cost Distribution by Damage")

    import pandas as pd
    chart_data = pd.DataFrame([
        {
            "Damage":   item["class_name"],
            "Min (Rs)": item["min_cost"],
            "Max (Rs)": item["max_cost"],
        }
        for item in estimate["line_items"]
    ]).set_index("Damage")

    st.bar_chart(chart_data)

# =============================================================
# DETAILED SEVERITY ANALYSIS (expandable)
# =============================================================
with st.expander("🔬 View Detailed Severity Analysis per Detection"):
    for i, row in enumerate(analysis_rows):
        st.markdown(f"**Detection {i+1} — {row['class_name']}**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Severity",   row["severity"])
        d2.metric("Score",      row["score"])
        d3.metric("Confidence", f"{row['confidence']:.0%}")
        d4.metric("Box Area",   f"{row['area_pct']}%")
        st.markdown("---")

# =============================================================
# DOWNLOAD REPORT
# =============================================================
st.divider()

report = [
    "AutoClaim Vision — Repair Cost Estimate Report",
    "=" * 55,
    f"Overall Severity : {overall_severity}",
    f"Damages Detected : {len(boxes)}",
    f"Unique Classes   : {len(estimate['line_items'])}",
    "",
    f"{'Damage Class':<35} {'Severity':<12} {'Min':>12} {'Max':>12}",
    "-" * 73,
]
for item in estimate["line_items"]:
    report.append(
        f"{item['class_name']:<35} {item['severity']:<12} "
        f"{format_inr(item['min_cost']):>12} {format_inr(item['max_cost']):>12}"
    )
report += [
    "-" * 73,
    f"{'Parts Subtotal':<35} {'':12} {format_inr(estimate['subtotal_min']):>12} {format_inr(estimate['subtotal_max']):>12}",
    f"{'Labour (15%-22%)':<35} {'':12} {format_inr(estimate['labour_min']):>12} {format_inr(estimate['labour_max']):>12}",
    f"{'TOTAL':<35} {'':12} {format_inr(estimate['total_min']):>12} {format_inr(estimate['total_max']):>12}",
    "",
    "DISCLAIMER",
    "-" * 55,
    DISCLAIMER,
]

st.download_button(
    label="⬇️ Download Report (.txt)",
    data="\n".join(report),
    file_name="autoclaim_cost_estimate.txt",
    mime="text/plain",
)

# =============================================================
# DISCLAIMER
# =============================================================
st.markdown(
    f'<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> {DISCLAIMER}</div>',
    unsafe_allow_html=True,
)
