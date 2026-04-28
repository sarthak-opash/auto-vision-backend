"""Severity scoring utilities for AutoClaim Vision.

This module scores YOLO damage detections directly. The damage model already
predicts labels like ``front-bumper-dent`` or ``scratch``, so the severity
engine can use those labels as the damage source and optionally accept a
separate part label later if a part model is added.
"""


PART_SCORE = {
    "Front-Windscreen-Damage": 10,
    "Rear-windscreen-Damage": 9,
    "Headlight-Damage": 9,
    "Taillight-Damage": 9,
    "tire flat": 10,
    "pillar-dent": 9,
    "roof-dent": 8,
    "fender-dent": 7,
    "quaterpanel-dent": 7,
    "bonnet-dent": 7,
    "boot-dent": 6,
    "front-bumper-dent": 7,
    "rear-bumper-dent": 6,
    "doorouter-dent": 6,
    "RunningBoard-Dent": 5,
    "Sidemirror-Damage": 7,
    "Signlight-Damage": 6,
    "Bodypanel-Dent": 5,
    "scratch": 3,
    "crack": 5,
}

DAMAGE_SCORE = {
    "dent": 1.0,
    "damage": 1.2,
    "scratch": 0.6,
    "crack": 1.1,
    "flat": 1.3,
}

MAX_NORMALIZED_AREA = 0.35
ZOOM_AREA_THRESHOLD = 0.5
ZOOM_PENALTY_FACTOR = 0.5
OVERLAP_IOU_THRESHOLD = 0.3
OVERLAP_PENALTY_FACTOR = 0.65
MULTI_DAMAGE_FACTOR = 0.08
SAFETY_BONUS_FACTOR = 0.12


def normalize_text(value):
    return str(value).strip().lower().replace("_", "-")


def get_match_score(text, table, default_value):
    text_norm = normalize_text(text)

    for key, value in table.items():
        key_norm = normalize_text(key)
        if key_norm == text_norm or key_norm in text_norm or text_norm in key_norm:
            return value

    return default_value


def get_area(bbox, img_w, img_h):
    if bbox is None or img_w <= 0 or img_h <= 0:
        return 0.0

    x1, y1, x2, y2 = bbox
    box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    img_area = img_w * img_h
    if img_area == 0:
        return 0.0
    return box_area / img_area


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def get_bbox_area(bbox):
    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def get_intersection_area(bbox_a, bbox_b):
    if bbox_a is None or bbox_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(x2 - x1, 0) * max(y2 - y1, 0)


def get_iou(bbox_a, bbox_b):
    if bbox_a is None or bbox_b is None:
        return 0.0

    intersection_area = get_intersection_area(bbox_a, bbox_b)
    if intersection_area == 0:
        return 0.0

    area_a = get_bbox_area(bbox_a)
    area_b = get_bbox_area(bbox_b)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def get_best_part_match(damage_bbox, part_detections):
    best_part = None
    best_overlap = 0.0

    for part in part_detections or []:
        part_bbox = part.get("bbox")
        overlap = get_intersection_area(damage_bbox, part_bbox)

        if overlap > best_overlap:
            best_overlap = overlap
            best_part = part

    return best_part, best_overlap


def get_damage_type(label):
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


def _get_part_score(damage_label, part_label):
    if part_label is not None:
        matched_part = get_match_score(part_label, PART_SCORE, None)
        if matched_part is not None:
            return matched_part

    matched_damage = get_match_score(damage_label, PART_SCORE, None)
    if matched_damage is not None:
        return matched_damage

    return 5


def score_to_level(score, normalized=False):
    if normalized:
        if score <= 25:
            return "Minor"
        if score <= 50:
            return "Moderate"
        if score <= 75:
            return "Severe"
        return "Critical"

    if score <= 5:
        return "Minor"
    if score <= 10:
        return "Moderate"
    if score <= 15:
        return "Severe"
    return "Critical"


def normalize_score(raw_score, max_possible=20):
    score = (raw_score / max_possible) * 100
    return min(score, 100)


def damage_portion_from_overlap(damage_bbox, part_bbox):
    if damage_bbox is None or part_bbox is None:
        return 0.0

    part_area = get_bbox_area(part_bbox)
    if part_area == 0:
        return 0.0

    overlap_area = get_intersection_area(damage_bbox, part_bbox)
    return min(overlap_area / part_area, 1.0)


def corrected_area_from_bbox(damage_bbox, img_w, img_h):
    normalized_area = get_area(damage_bbox, img_w, img_h)
    if normalized_area <= 0:
        return 0.0

    clipped_area = clamp(normalized_area, 0.0, MAX_NORMALIZED_AREA)
    corrected_area = clipped_area ** 0.5

    if normalized_area > ZOOM_AREA_THRESHOLD:
        zoom_penalty = ZOOM_PENALTY_FACTOR
        corrected_area *= zoom_penalty

    return clamp(corrected_area, 0.0, 1.0)


def _severity_level_from_score(score):
    if score <= 20:
        return "Low"
    if score <= 50:
        return "Medium"
    if score <= 75:
        return "High"
    return "Critical"


def _is_critical_part(label):
    label_norm = normalize_text(label)
    return any(
        token in label_norm
        for token in [
            "windscreen",
            "glass",
            "tire",
            "headlight",
            "light",
            "pillar",
            "roof",
        ]
    )


def _dedupe_preserve_order(values):
    return list(dict.fromkeys(values))


def _overlap_penalty(current_bbox, previous_boxes):
    max_iou = 0.0

    for previous_bbox in previous_boxes:
        max_iou = max(max_iou, get_iou(current_bbox, previous_bbox))

    if max_iou < OVERLAP_IOU_THRESHOLD:
        return 1.0

    return max(0.4, 1.0 - (max_iou * OVERLAP_PENALTY_FACTOR))


def generate_severity_report(detections, img_w, img_h, part_detections=None):
    items = []
    detected_parts = []
    critical_flags = []
    total_raw = 0.0
    previous_damage_boxes = []

    for det in detections or []:
        damage_label = det.get("class")
        confidence = float(det.get("confidence", 1.0) or 1.0)
        damage_bbox = det.get("bbox")

        # Find matching part from part detections
        matched_part, _ = get_best_part_match(damage_bbox, part_detections)
        part_label = matched_part["class"] if matched_part else damage_label
        part_bbox = matched_part["bbox"] if matched_part else None

        detected_parts.append(part_label)

        # Core scoring: importance × type_factor × damage_portion × confidence
        part_importance = _get_part_score(damage_label, part_label)
        damage_type = get_damage_type(damage_label)
        type_factor = DAMAGE_SCORE[damage_type]
        corrected_area = corrected_area_from_bbox(damage_bbox, img_w, img_h)
        overlap_portion = damage_portion_from_overlap(damage_bbox, part_bbox)
        effective_area = overlap_portion if overlap_portion > 0 else corrected_area
        overlap_factor = _overlap_penalty(damage_bbox, previous_damage_boxes)
        previous_damage_boxes.append(damage_bbox)

        # Calculate raw contribution
        raw_contribution = (
            part_importance
            * type_factor
            * effective_area
            * confidence
            * overlap_factor
            * 10
        )

        total_raw += raw_contribution

        items.append(
            {
                "part": part_label,
                "damage_type": damage_type,
                "confidence": round(confidence, 2),
                "corrected_area": round(effective_area, 4),
                "raw_contribution": raw_contribution,
            }
        )

        # Check for critical flags
        if _is_critical_part(part_label):
            critical_flags.append(f"Safety-critical part affected: {part_label}")

        if damage_type == "flat":
            critical_flags.append("Flat tire detected")

        if damage_type == "crack" and "glass" in normalize_text(part_label):
            critical_flags.append("Glass crack detected")

    # Multi-damage bonus
    if len(detections or []) >= 3:
        total_raw *= 1.0 + MULTI_DAMAGE_FACTOR
        critical_flags.append("Multiple damages detected")

    # Calculate final severity score
    severity_score = round(min(total_raw * 10, 100), 2)

    if any(_is_critical_part(part) for part in detected_parts):
        severity_score = round(min(severity_score * (1.0 + SAFETY_BONUS_FACTOR), 100), 2)
        critical_flags.append("Safety-critical part detected")

    severity_level = _severity_level_from_score(severity_score)

    # Calculate contribution percentages
    if total_raw > 0:
        for item in items:
            share = item["raw_contribution"] / total_raw
            item["severity_contribution"] = round(share * severity_score, 2)
            del item["raw_contribution"]
    else:
        for item in items:
            item["severity_contribution"] = 0.0
            del item["raw_contribution"]

    critical_flags = _dedupe_preserve_order(critical_flags)

    return {
        "severity_score": severity_score,
        "severity_level": severity_level,
        "detected_parts": _dedupe_preserve_order(detected_parts),
        "damage_table": items,
        "critical_flags": critical_flags,
    }


# End of module