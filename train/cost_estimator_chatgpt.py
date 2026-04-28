# ===========================
# File: train/cost_estimator.py
# Put inside train folder
# ===========================

import os
import sys
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# -----------------------------
# Load Damage Model
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "runs", "damage", "weights", "best.pt")
model = YOLO(MODEL_PATH)

# -----------------------------
# Indian Market Price Table
# -----------------------------
PRICE_TABLE = {
    "Bodypanel-Dent": (2500, 10000),
    "Front-Windscreen-Damage": (4000, 18000),
    "Headlight-Damage": (2500, 20000),
    "Rear-windscreen-Damage": (3000, 12000),
    "RunningBoard-Dent": (2000, 12000),
    "Sidemirror-Damage": (1500, 8000),
    "Signlight-Damage": (500, 4000),
    "Taillight-Damage": (1500, 10000),
    "bonnet-dent": (2500, 12000),
    "boot-dent": (2500, 10000),
    "doorouter-dent": (3000, 15000),
    "fender-dent": (2500, 9000),
    "front-bumper-dent": (3000, 18000),
    "pillar-dent": (5000, 25000),
    "quaterpanel-dent": (4000, 18000),
    "rear-bumper-dent": (3000, 16000),
    "roof-dent": (5000, 22000),
    "scratch": (1000, 8000),
    "crack": (2500, 15000),
    "tire flat": (800, 6000),
}

# -----------------------------
# INR Format
# -----------------------------
def format_inr(num):
    return "₹{:,.0f}".format(num)

# -----------------------------
# Severity by box area
# -----------------------------
def get_severity(box, w, h):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    img_area = w * h
    ratio = area / img_area

    if ratio < 0.02:
        return "Minor", 1.0
    elif ratio < 0.06:
        return "Moderate", 1.3
    elif ratio < 0.12:
        return "Severe", 1.6
    else:
        return "Critical", 2.0

# -----------------------------
# Main Function
# -----------------------------
def estimate_cost(image_path):

    results = model.predict(image_path, conf=0.25, verbose=False)
    r = results[0]

    if len(r.boxes) == 0:
        return {
            "detections": [],
            "total_min": 0,
            "total_max": 0
        }

    h, w = r.orig_shape
    detections = []
    total_min = 0
    total_max = 0

    for box in r.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        severity, multiplier = get_severity((x1,y1,x2,y2), w, h)

        base_min, base_max = PRICE_TABLE.get(name, (2000, 10000))

        est_min = int(base_min * multiplier)
        est_max = int(base_max * multiplier)

        total_min += est_min
        total_max += est_max

        detections.append({
            "name": name,
            "severity": severity,
            "min": est_min,
            "max": est_max
        })

    return {
        "detections": detections,
        "total_min": total_min,
        "total_max": total_max,
        "plot": r.plot()
    }