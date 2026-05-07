"""
main.py
-------
AutoClaim Vision — Unified Streamlit Application
Pipeline: Upload → Damage Detection → Severity → Cost → PDF Report

Run: streamlit run main.py
"""

import os
import sys
import importlib
import tempfile
import datetime
import cv2
import streamlit as st
from pathlib import Path
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_PATH  = Path(__file__).parent
TRAIN_PATH = BASE_PATH / "train"
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoClaim Vision",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}
.stApp {
    background: #07070f;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0a0f !important;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] * { color: #c9c9e0 !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption { color: #6b6b8a !important; font-size: 0.78rem !important; }
section[data-testid="stSidebar"] hr { border-color: #1e1e2e !important; }

/* ── Hero ── */
.ac-hero {
    background: #0a0a0f;
    border-bottom: 1px solid #1e1e2e;
    padding: 2.4rem 0 2rem;
    margin: -1rem -2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.ac-hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(255,75,60,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.ac-hero-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.28em; text-transform: uppercase;
    color: #ff4b3c; margin-bottom: 0.6rem;
}
.ac-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem; font-weight: 800;
    color: #f0f0f8; margin: 0 0 0.5rem;
    letter-spacing: -0.03em; line-height: 1;
}
.ac-hero h1 span { color: #ff4b3c; }
.ac-hero p { color: #6b6b8a; font-size: 0.92rem; font-weight: 300; margin: 0; }

/* ── Upload ── */
div[data-testid="stFileUploader"] {
    background: #0e0e1a;
    border: 1.5px dashed #2a2a40;
    border-radius: 14px;
    padding: 0.4rem 1rem;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover { border-color: #ff4b3c; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: #0e0e1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
div[data-testid="stMetric"] label {
    color: #6b6b8a !important;
    font-size: 0.68rem !important; font-weight: 500 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #f0f0f8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important; font-weight: 700 !important;
}

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child { border-bottom: 1px solid #1e1e2e; gap: 0; }
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 0.05em !important; padding: 0.6rem 1.4rem !important;
    color: #6b6b8a !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff4b3c !important; border-bottom-color: #ff4b3c !important;
    background: transparent !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { border: 1px solid #1e1e2e; border-radius: 12px; overflow: hidden; }

/* ── Expander ── */
details[data-testid="stExpander"] {
    background: #0e0e1a; border: 1px solid #1e1e2e !important;
    border-radius: 10px; margin-bottom: 6px;
}
details[data-testid="stExpander"] summary { font-size: 0.85rem; color: #c9c9e0; padding: 0.6rem 1rem; }

/* ── Severity badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 18px; border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em;
}
.badge-low      { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-medium   { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-high     { background: rgba(239,68,68,0.15);   color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-critical { background: rgba(127,29,29,0.4);    color: #fca5a5; border: 1px solid rgba(239,68,68,0.5); }

/* ── Score bar ── */
.sev-track { background: #1e1e2e; border-radius: 100px; height: 6px; margin: 8px 0 14px; }
.sev-fill  { height: 6px; border-radius: 100px; }

/* ── Cost hero ── */
.cost-hero { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: #f0f0f8; letter-spacing: -0.03em; }
.cost-sub  { font-size: 0.75rem; color: #6b6b8a; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 2px; }

/* ── Section heading ── */
.sec-head {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700; color: #f0f0f8;
    letter-spacing: -0.01em; margin-bottom: 0.8rem;
    border-left: 3px solid #ff4b3c; padding-left: 10px;
}

/* ── Pills ── */
.det-pill {
    display: inline-block; background: #1e1e2e; border: 1px solid #2a2a40;
    border-radius: 100px; padding: 3px 13px; margin: 3px;
    font-size: 0.75rem; color: #c9c9e0; letter-spacing: 0.02em;
}

/* ── Images ── */
div[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #1e1e2e; }

/* ── Buttons ── */
div[data-testid="stDownloadButton"] button,
button[kind="primary"] {
    background: #ff4b3c !important; color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.04em !important;
}
button[kind="primary"]:hover { background: #e03528 !important; }

/* ── Divider ── */
hr { border-color: #1e1e2e !important; }

/* ── Sidebar pipeline ── */
.pipe-step { display: flex; align-items: center; gap: 10px; padding: 7px 0; font-size: 0.82rem; color: #8888aa; }
.pipe-num {
    width: 22px; height: 22px; border-radius: 50%;
    background: #1e1e2e; border: 1px solid #2e2e4e;
    display: inline-flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700;
    color: #ff4b3c; flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
SEV_COLOR = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444", "Critical": "#fca5a5"}
SEV_BADGE = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high", "Critical": "badge-critical"}

def severity_badge(level: str) -> str:
    cls = SEV_BADGE.get(level, "badge-low")
    clr = SEV_COLOR.get(level, "#6b6b8a")
    return (
        f'<span class="badge {cls}">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{clr};display:inline-block"></span>'
        f'{level}</span>'
    )

def boxes_to_rows(boxes, names):
    rows = []
    for box in boxes:
        cid = int(box.cls[0])
        rows.append({
            "class":      names[cid],
            "confidence": float(box.conf[0]),
            "bbox":       [float(v) for v in box.xyxy[0].tolist()],
        })
    return rows


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_damage_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)

def load_engines():
    import train.severity        as sev_eng
    import train.cost_estimation as cost_eng
    import train.report          as rep_eng
    from train.vehicle_catalog import ALL_MAKES, get_models_for_make
    return importlib.reload(sev_eng), importlib.reload(cost_eng), importlib.reload(rep_eng)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
    '<div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;'
    'color:#f0f0f8;letter-spacing:-0.02em">AutoClaim<span style="color:#ff4b3c"> Vision</span></div>'
    
    '<div style=font-family:Syne,sans-serif;font-size:0.9rem;color:#444466;margin-top:2px;margin-bottom:1rem">'
    'AI-Powered Car Damage Assessment</div>'

    '</div>',
    unsafe_allow_html=True
    )
    st.divider()
    st.divider()
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:1rem;color:#444466;'
        'margin-bottom:10px">Pipeline</div>',
        unsafe_allow_html=True
    )
    for i, stage in enumerate(["Damage Detection", "Severity Scoring", "Cost Estimation", "PDF Report"], 1):
        st.markdown(
            f'<div class="pipe-step", style="font-family:Syne,sans-serif;"><span class="pipe-num">{i}</span><span>{stage}</span></div>',
            unsafe_allow_html=True
        )
    damage_conf = 0.25

    st.divider()
    # ── Vehicle Details ────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:1rem;color:#c9c9e0;'
        'margin-bottom:10px">🚗 Vehicle Details</div>',
        unsafe_allow_html=True
    )
    st.caption("Optional · improves cost accuracy")

    from train.vehicle_catalog import ALL_MAKES, get_models_for_make, get_vehicle_info
    selected_make  = st.selectbox("Make",  ALL_MAKES, index=0, key="v_make")
    model_options  = get_models_for_make(selected_make)
    selected_model = st.selectbox("Model", model_options, index=0, key="v_model")
    selected_year  = st.number_input(
        "Year", min_value=2000, max_value=2030, value=2022, step=1, key="v_year"
    )
    # Show segment tag
    if selected_make != "Unknown" and selected_model != "Unknown":
        vinfo = get_vehicle_info(selected_make, selected_model)
        if vinfo:
            seg_color = {"budget": "#10b981","mid": "#f59e0b","premium": "#3b82f6","luxury": "#a855f7"}.get(vinfo.get("segment",""), "#6b6b8a")
            st.markdown(
                f'<div style="margin-top:4px"><span style="background:{seg_color}22;color:{seg_color};'
                f'border:1px solid {seg_color}44;border-radius:100px;padding:2px 10px;font-size:0.72rem;font-weight:600">'
                f'{vinfo.get("segment","").title()} segment</span></div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown(
    '<div style="font-family:Syne,sans-serif;font-size:1rem;color:#444466;margin-top:2px;margin-bottom:1rem">'
    'Developed By :- '
    '</div>'

    '<div style="font-family:Syne,sans-serif;>'
    '<ul list-style="dot">'
    
    '<li style="margin-bottom:6px;">'
    'Manav Katrodiya</li>'
    
    '<li style="margin-bottom:6px;">'
    'Dev Charan</li>'
    
    '<li style="margin-bottom:6px;">'
    'Sarthak Nimbark</li>'
    
    '</ul></div>',
    unsafe_allow_html=True
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ac-hero">
    <div class="ac-hero-label">Insurance Tech &nbsp;·&nbsp; Computer Vision</div>
    <h1>AutoClaim <span>Vision</span></h1>
    <p>Upload a photo · detect damage · estimate cost · download report</p>
</div>
""", unsafe_allow_html=True)


# ── Models ────────────────────────────────────────────────────────────────────
# DAMAGE_MODEL_PATH = BASE_PATH / "runs" / "damage_seg_v1" / "epoch_150" / "weights" / "best.pt"
DAMAGE_MODEL_PATH = BASE_PATH / "runs" / "damage" / "weights" / "best.pt"
PART_MODEL_PATH   = BASE_PATH / "runs" / "parts"  / "weights" / "best.pt"

if not DAMAGE_MODEL_PATH.exists():
    st.error("❌ Damage model not found. Expected at `runs/damage/weights/best.pt`")
    st.stop()

damage_model = load_damage_model(str(DAMAGE_MODEL_PATH))

# Part model is optional — used to remap damage detections to semantic part names
# (same as severity_app.py) so severity scores are consistent across all pages.
part_model = load_damage_model(str(PART_MODEL_PATH)) if PART_MODEL_PATH.exists() else None


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload car image  ( JPG · PNG · WEBP )",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.markdown(
        '<div style="text-align:center;color:#444466;font-size:0.85rem;padding:2.5rem 0">'
        '← Upload an image to begin analysis</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ── Pipeline ──────────────────────────────────────────────────────────────────
tmp_path = annot_path = None

try:
    suffix = Path(uploaded.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    image = Image.open(tmp_path).convert("RGB")

    col_img, col_summary = st.columns([1, 1], gap="large")
    with col_img:
        st.markdown('<div class="sec-head">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, width='stretch')

    with st.spinner("Analysing image…"):
        dmg_results = damage_model.predict(source=tmp_path, conf=damage_conf, imgsz=640)
        ann_bgr     = dmg_results[0].plot()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as atmp:
            Image.fromarray(ann_bgr[:, :, ::-1]).save(atmp.name)
            annot_path = atmp.name

        detections  = boxes_to_rows(dmg_results[0].boxes, damage_model.names)

        # Run part model if available — passes semantic part detections to severity engine
        # so part names and scores match severity_app.py exactly (no divergence)
        part_rows = []
        if part_model is not None:
            part_results = part_model.predict(source=tmp_path, conf=0.25, imgsz=640)
            part_rows    = boxes_to_rows(part_results[0].boxes, part_model.names)

        sev_eng, cost_eng, rep_eng = load_engines()
        sev_report  = sev_eng.generate_severity_report(detections, image.width, image.height, part_rows)

        # Build vehicle_info — pass None if user left both as Unknown
        vehicle_info = None
        if selected_make != "Unknown" and selected_model != "Unknown":
            vehicle_info = {
                "make":  selected_make,
                "model": selected_model,
                "year":  int(selected_year),
            }

        cost_report = cost_eng.estimate_cost(sev_report["part_severity"], vehicle_info=vehicle_info)

    lvl   = sev_report["severity_level"]
    score = sev_report["severity_score"]
    bar_c = SEV_COLOR.get(lvl, "#6b6b8a")

    # ── Summary column ────────────────────────────────────────────────────────
    with col_summary:
        st.markdown('<div class="sec-head">Analysis Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-bottom:4px;font-size:0.68rem;letter-spacing:0.14em;'
            f'color:#444466;text-transform:uppercase">Severity</div>'
            f'{severity_badge(lvl)}'
            f'<div class="sev-track"><div class="sev-fill" style="width:{score}%;background:{bar_c}"></div></div>'
            f'<div style="font-size:0.72rem;color:#6b6b8a;margin-bottom:1rem">Score: {score} / 100</div>',
            unsafe_allow_html=True
        )
        m1, m2 = st.columns(2)
        m1.metric("Damages found",  len(detections))
        m2.metric("Parts affected", len(sev_report["detected_parts"]))
        m3, m4 = st.columns(2)
        m3.metric("Severity Level", f"{lvl}")
        m4.metric("Grand total",    f"₹ {cost_report['grand_total']:,.0f}")
        st.markdown("<br>", unsafe_allow_html=True)
        # ── Vehicle info badge ─────────────────────────────────────────────
        if vehicle_info:
            st.markdown(
                f'<div style="background:#0e0e1a;border:1px solid #1e1e2e;border-radius:10px;'
                f'padding:10px 14px;margin-bottom:12px">'
                f'<div style="font-size:0.65rem;color:#6b6b8a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">Vehicle</div>'
                f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#f0f0f8;font-size:1rem">'
                f'{vehicle_info["make"]} {vehicle_info["model"]}</div>'
                f'<div style="font-size:0.78rem;color:#6b6b8a">{vehicle_info["year"]} &nbsp;·&nbsp; OEM prices applied</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#0e0e1a;border:1px solid #1e1e2e;border-radius:10px;'
                'padding:10px 14px;margin-bottom:12px">'
                '<div style="font-size:0.65rem;color:#6b6b8a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">Vehicle</div>'
                '<div style="font-size:0.78rem;color:#6b6b8a">Not specified · generic prices used</div>'
                '</div>',
                unsafe_allow_html=True
            )
        if cost_report["grand_total"] > 0:
            st.markdown(
                f'<div class="cost-hero">₹ {cost_report["grand_total"]:,.0f}</div>'
                f'<div class="cost-sub">Estimated repair cost incl. 20% labour</div>',
                unsafe_allow_html=True
            )
        else:
            st.success("No damage detected — no repair cost.")

    if sev_report["critical_flags"]:
        for flag in sev_report["critical_flags"]:
            st.error(f"⚠️ {flag}")

    st.markdown('<hr style="margin:2rem 0">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    tab_dmg, tab_sev, tab_cost, tab_dl = st.tabs([
        "  🔍  Damage Detection  ",
        "  📊  Severity Analysis  ",
        "  💰  Cost Estimate  ",
        "  📄  Download Report  ",
    ])

    # ── Tab 1: Damage ─────────────────────────────────────────────────────────
    with tab_dmg:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown('<div class="sec-head">Annotated Image</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB), width='stretch')
        with c2:
            st.markdown(
                f'<div class="sec-head">Detections '
                f'<span style="font-size:0.75rem;font-weight:400;color:#6b6b8a">'
                f'— {len(detections)} found</span></div>',
                unsafe_allow_html=True
            )
            if detections:
                chips = "".join(
                    f'<span class="det-pill">{d["class"].replace("-", " ").title()}</span>'
                    for d in detections
                )
                st.markdown(chips + "<br><br>", unsafe_allow_html=True)
                for det in detections:
                    bb   = det["bbox"]
                    w_px = bb[2] - bb[0]
                    h_px = bb[3] - bb[1]
                    area = (w_px * h_px) / (image.width * image.height) * 100
                    with st.expander(f"{det['class'].replace('-',' ').title()}  —  {det['confidence']:.1%}"):
                        ic1, ic2, ic3 = st.columns(3)
                        ic1.metric("Area covered", f"{area:.1f}%")
                        ic2.metric("Width",        f"{w_px:.0f} px")
                        ic3.metric("Height",       f"{h_px:.0f} px")
            else:
                st.success("✅ No damage detected.")

    # ── Tab 2: Severity ───────────────────────────────────────────────────────
    with tab_sev:
        st.markdown("<br>", unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Severity Level", lvl)
        sc2.metric("Severity Score", f"{score} / 100")
        sc3.metric("Parts Damaged",  len(sev_report["detected_parts"]))
        sc4.metric("Damage Classes", len(set(d["class"] for d in detections)))
        st.markdown("<br>", unsafe_allow_html=True)
        sl, sr = st.columns([1.3, 1], gap="large")
        with sl:
            st.markdown('<div class="sec-head">Part-wise Breakdown</div>', unsafe_allow_html=True)
            if sev_report.get("part_severity"):
                rows = []
                for part_name, info in sev_report["part_severity"].items():
                    rows.append({
                        "Part":         part_name,
                        "Score":        info["severity_score"],
                        "Level":        info["severity_level"],
                        "Damage Count": info["damage_count"],
                        "Damage Types": ", ".join(info["damage_types"]),
                    })
                st.dataframe(rows, width='stretch', hide_index=True)
            else:
                st.info("No part-wise data available.")
        with sr:
            st.markdown('<div class="sec-head">Assessment</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.75rem;color:#6b6b8a;letter-spacing:0.1em;'
                f'text-transform:uppercase;margin-bottom:4px">Detected parts</div>'
                f'<div style="font-size:0.88rem;color:#c9c9e0;line-height:1.8">'
                f'{", ".join(sev_report["detected_parts"]) or "—"}</div>',
                unsafe_allow_html=True
            )
            if sev_report["critical_flags"]:
                st.markdown("<br>", unsafe_allow_html=True)
                for flag in sev_report["critical_flags"]:
                    st.error(f"⚠️ {flag}")

    # ── Tab 3: Cost ───────────────────────────────────────────────────────────
    with tab_cost:
        st.markdown("<br>", unsafe_allow_html=True)
        # Vehicle info header
        if vehicle_info:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;'
                f'background:#0e0e1a;border:1px solid #1e1e2e;border-radius:10px;'
                f'padding:10px 16px;margin-bottom:16px">'
                f'<span style="font-size:1.4rem">🚗</span>'
                f'<div><div style="font-family:Syne,sans-serif;font-weight:700;color:#f0f0f8">'
                f'{vehicle_info["make"]} {vehicle_info["model"]} ({vehicle_info["year"]})</div>'
                f'<div style="font-size:0.75rem;color:#6b6b8a">OEM part prices applied from vehicle catalog</div></div></div>',
                unsafe_allow_html=True
            )
        else:
            st.info("ℹ️ No vehicle specified — generic damage-type prices used. Select Make/Model/Year in the sidebar for accurate OEM pricing.", icon=None)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#6b6b8a;margin-bottom:1rem">'
            f'ℹ️ &nbsp;{cost_report["note"]}</div>',
            unsafe_allow_html=True
        )
        if cost_report["line_items"]:
            rows = []
            for item in cost_report["line_items"]:
                rows.append({
                    "Part":          item["part"],
                    "Severity":      f'{item["severity_level"]} ({item["severity_score"]})',
                    "Damage Types":  item["damage_types"],
                    "Repair Action": item["repair_action"],
                    "Base Price (₹)": f'₹ {item["base_price"]:,.0f}',
                    "Price Source":  item.get("price_source", "generic"),
                    "Est. Cost (₹)": f'₹ {item["part_cost"]:,.0f}',
                })
            st.dataframe(rows, width='stretch', hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Parts Total",  f"₹ {cost_report['parts_total']:,.0f}")
            tc2.metric("Labour (20%)", f"₹ {cost_report['labor_total']:,.0f}")
            tc3.metric("Grand Total",  f"₹ {cost_report['grand_total']:,.0f}")
        else:
            st.info("No cost data — no damage detected.")

    # ── Tab 4: Report ─────────────────────────────────────────────────────────
    with tab_dl:
        st.markdown("<br>", unsafe_allow_html=True)
        rc, _ = st.columns([1, 1])
        with rc:
            st.markdown('<div class="sec-head">Generate PDF Report</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.85rem;color:#6b6b8a;margin-bottom:1.2rem;line-height:1.6">'
                'Professional PDF with annotated image, all detections, '
                'severity breakdown, cost estimate and disclaimer.</div>',
                unsafe_allow_html=True
            )
            if st.button("📄 Generate PDF Report", type="primary", width='stretch'):
                with st.spinner("Building PDF…"):
                    try:
                        # Enrich vehicle_info with segment for PDF
                        pdf_vehicle_info = None
                        if vehicle_info:
                            from train.vehicle_catalog import get_vehicle_info as _gvi
                            vextra = _gvi(vehicle_info["make"], vehicle_info["model"])
                            pdf_vehicle_info = {**vehicle_info, **vextra}

                        pdf_bytes = rep_eng.generate_report(
                            severity_result=sev_report,
                            cost_result=cost_report,
                            annotated_image_path=annot_path,
                            vehicle_info=pdf_vehicle_info,
                        )
                        filename = f"autoclaim_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        st.success("✅ Report ready!")
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            width='stretch',
                        )
                    except ImportError:
                        st.error("Install fpdf2:  `pip install fpdf2`")
                    except Exception as e:
                        st.error(f"Report error: {e}")
                        raise

finally:
    for p in [tmp_path, annot_path]:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except Exception:
                pass