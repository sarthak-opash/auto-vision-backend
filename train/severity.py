"""Severity scoring utilities for AutoClaim Vision (REFACTORED - STABLE)"""

from typing import List, Dict, Optional


# ---------------- CONFIG ---------------- #

# Part importance scores (raw 1-10 scale, normalized to [0,1] in scoring)
# Matches the 17 part detection model classes (0-16)
PART_SCORE = {
    "Bodypanel-Dent": 5,            # 0
    "Front-Windscreen-Damage": 10,  # 1
    "Headlight-Damage": 9,          # 2
    "Rear-windscreen-Damage": 9,    # 3
    "RunningBoard-Dent": 5,         # 4
    "Sidemirror-Damage": 7,         # 5
    "Signlight-Damage": 6,          # 6
    "Taillight-Damage": 9,          # 7
    "bonnet-dent": 7,               # 8
    "boot-dent": 6,                 # 9
    "doorouter-dent": 6,            # 10
    "fender-dent": 7,               # 11
    "front-bumper-dent": 7,         # 12
    "pillar-dent": 9,               # 13
    "quaterpanel-dent": 7,          # 14
    "rear-bumper-dent": 6,          # 15
    "roof-dent": 8,                 # 16
}

# Damage type severity (already in [0,1] — NOT used as multipliers)
# scratch(0.2) → dent(0.45) → crack(0.6) → damage(0.7) → flat(0.8)
DAMAGE_SCORE = {
    "scratch": 0.2,
    "dent": 0.45,
    "crack": 0.6,
    "damage": 0.7,
    "flat": 0.8,
}




# ---------------- UTILS ---------------- #

def normalize_text(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def get_match_score(text: str, table: Dict, default):
    text = normalize_text(text)
    for key, val in table.items():
        if normalize_text(key) in text:
            return val
    return default


def get_area(bbox, img_w: int, img_h: int) -> float:
    if bbox is None or img_w <= 0 or img_h <= 0:
        return 0.0

    x1, y1, x2, y2 = bbox
    box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    return box_area / (img_w * img_h)


def get_bbox_area(bbox) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def get_intersection_area(a, b) -> float:
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(x2 - x1, 0) * max(y2 - y1, 0)


def get_best_part_match(bbox, part_detections):
    best = None
    best_overlap = 0.0

    for part in part_detections or []:
        overlap = get_intersection_area(bbox, part.get("bbox"))
        if overlap > best_overlap:
            best_overlap = overlap
            best = part

    return best


def get_damage_type(label: str) -> str:
    label = normalize_text(label)
    if "scratch" in label:
        return "scratch"
    if "dent" in label:
        return "dent"
    if "crack" in label:
        return "crack"
    if "flat" in label:
        return "flat"
    return "damage"


# ---------------- CORE SCORING ---------------- #

def compute_damage_score(area: float, damage_type: str, confidence: float, part_label: str) -> float:
    """
    Compute a single damage detection score in [0, 1].

    Uses bounded additive scoring with four normalized components:
      - A: area factor (non-linear, zoom-bias resistant)
      - T: damage type severity (in [0, 1])
      - P: part importance (normalized from PART_SCORE)
      - C: confidence (squared to suppress inflation)

    Weights sum to 1.0 to keep output in [0, 1].
    """

    # ---- AREA (non-linear to resist zoom bias)
    # Higher K = slower growth, exponent > 1 compresses large areas
    K = 0.25
    A = (area / (area + K)) ** 1.2

    # ---- DAMAGE TYPE (already in [0, 1])
    T = DAMAGE_SCORE.get(damage_type, 0.45)

    # ---- PART IMPORTANCE (normalized to [0, 1])
    raw_part = get_match_score(part_label, PART_SCORE, 5)
    P = raw_part / 10.0

    # ---- CONFIDENCE (squared to suppress inflation)
    C = confidence ** 2

    # ---- BOUNDED ADDITIVE SCORE
    # Area-dominant: area drives the score, type differentiates severity
    # Weights: area=0.40, type=0.30, part=0.15, confidence=0.15
    score = 0.40 * A + 0.30 * T + 0.15 * P + 0.15 * C

    return max(0.0, min(score, 1.0))


def aggregate_part(scores: List[float]) -> float:
    """
    Aggregate multiple damage scores for one part.

    Uses: S_part = 1 - Π(1 - S_damage)
    With penalty for excessive small damages to prevent inflation.
    """
    if not scores:
        return 0.0

    # Union-style aggregation: 1 - product(1 - s)
    product = 1.0
    for s in scores:
        product *= (1.0 - s)

    part_score = 1.0 - product

    # Penalty: diminishing returns after 3 detections on same part
    n = len(scores)
    if n > 3:
        penalty = 0.90 ** (n - 3)
        part_score *= penalty

    return max(0.0, min(part_score, 1.0))


def severity_level(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Medium"
    if score < 75:
        return "High"
    return "Critical"


# ---------------- MAIN ---------------- #

def generate_severity_report(
    detections: List[Dict],
    img_w: int,
    img_h: int,
    part_detections: Optional[List[Dict]] = None,
) -> Dict:

    items = []
    part_scores_map = {}       # part_label -> [damage_scores]
    part_damages_map = {}      # part_label -> [damage_types]
    part_areas_map = {}        # part_label -> [areas]
    detected_parts = []
    critical_flags = []

    for det in detections or []:
        label = det.get("class")
        confidence = float(det.get("confidence", 1.0))
        bbox = det.get("bbox")

        matched = get_best_part_match(bbox, part_detections)
        part_label = matched["class"] if matched else label
        part_bbox = matched["bbox"] if matched else None

        detected_parts.append(part_label)

        # area — use overlap ratio if part detected, else image-relative
        if part_bbox:
            part_area = get_bbox_area(part_bbox)
            overlap = get_intersection_area(bbox, part_bbox)
            area = overlap / part_area if part_area > 0 else 0.0
        else:
            area = get_area(bbox, img_w, img_h)

        damage_type = get_damage_type(label)

        score = compute_damage_score(area, damage_type, confidence, part_label)

        part_scores_map.setdefault(part_label, []).append(score)
        part_damages_map.setdefault(part_label, []).append(damage_type)
        part_areas_map.setdefault(part_label, []).append(area)

        items.append({
            "part": part_label,
            "damage_type": damage_type,
            "confidence": round(confidence, 2),
            "area": round(area, 4),
            "damage_score": round(score, 4),
        })

        if damage_type == "flat":
            critical_flags.append("Flat tire detected")

    # ---- AGGREGATE per part ----
    part_agg_scores = {
        p: aggregate_part(s) for p, s in part_scores_map.items()
    }

    # ---- BUILD part_severity (for cost estimation) ----
    part_severity = {}
    for part_label, agg_score in part_agg_scores.items():
        ps = round(agg_score * 100.0, 2)
        damage_types = list(dict.fromkeys(part_damages_map.get(part_label, [])))
        max_area = max(part_areas_map.get(part_label, [0.0]))

        part_severity[part_label] = {
            "severity_score": ps,
            "severity_level": severity_level(ps),
            "damage_count": len(part_scores_map[part_label]),
            "damage_types": damage_types,
            "max_area_ratio": round(max_area, 4),
        }

    # ---- OVERALL severity ----
    if part_agg_scores:
        max_score = max(part_agg_scores.values())
        avg_score = sum(part_agg_scores.values()) / len(part_agg_scores)

        # Blend: 60% max + 40% average — dominant damage drives the score
        raw = 0.6 * max_score + 0.4 * avg_score

        # Multi-part bonus: more damaged parts = more severe
        n_parts = len(part_agg_scores)
        if n_parts > 1:
            part_bonus = min(0.15, 0.06 * (n_parts - 1) ** 0.5)
            raw = min(1.0, raw + part_bonus)

        # Scale to 0-100 with power 1.25 for spread
        severity = raw * 100.0
        severity = (severity ** 1.25) / (100.0 ** 0.25)
        severity = min(severity, 100.0)
    else:
        severity = 0.0

    return {
        "severity_score": round(severity, 2),
        "severity_level": severity_level(severity),
        "detected_parts": list(dict.fromkeys(detected_parts)),
        "damage_table": items,
        "critical_flags": list(set(critical_flags)),
        "part_severity": part_severity,
    }