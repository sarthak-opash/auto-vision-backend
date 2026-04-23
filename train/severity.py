# ==========================================================
# FILE NAME : utils/severity.py
# PURPOSE :
# Professional Severity Engine for AutoClaim Vision
#
# Uses:
# 1. Damaged Part Importance
# 2. Damage Type Weight
# 3. Damage Size
# 4. Multi Damage Bonus
#
# FUTURE READY:
# Same score can be used for cost estimation
# ==========================================================


# ==========================================================
# PART IMPORTANCE SCORE
# expensive / critical parts = high score
# ==========================================================
PART_SCORE = {
    "front bumper": 3,
    "rear bumper": 3,
    "bumper": 3,

    "hood": 4,
    "bonnet": 4,

    "door": 3,
    "front door": 3,
    "rear door": 3,

    "fender": 2,

    "headlight": 4,
    "tail light": 3,

    "windshield": 5,
    "glass": 4,

    "mirror": 2,

    "wheel": 3,
    "tyre": 3
}


# ==========================================================
# DAMAGE TYPE SCORE
# ==========================================================
DAMAGE_SCORE = {
    "scratch": 1,
    "dent": 2,
    "crack": 3,
    "tear": 3,
    "break": 5,
    "broken": 5,
    "damage": 2
}


# ==========================================================
# HELPER
# ==========================================================
def get_match_score(text, table, default_value):

    text = text.lower()

    for key in table:
        if key in text:
            return table[key]

    return default_value


# ==========================================================
# MAIN ENGINE
# ==========================================================
def calculate_severity(
    damage_label,
    part_label,
    box,
    image_width,
    image_height,
    total_damages
):

    # ------------------------------------------------------
    # 1. PART SCORE
    # ------------------------------------------------------
    part_score = get_match_score(part_label, PART_SCORE, 2)

    # ------------------------------------------------------
    # 2. DAMAGE TYPE SCORE
    # ------------------------------------------------------
    damage_score = get_match_score(damage_label, DAMAGE_SCORE, 1)

    # ------------------------------------------------------
    # 3. DAMAGE SIZE SCORE
    # ------------------------------------------------------
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    area = (x2 - x1) * (y2 - y1)
    img_area = image_width * image_height

    percent = (area / img_area) * 100

    if percent < 2:
        size_score = 1
    elif percent < 6:
        size_score = 2
    else:
        size_score = 4

    # ------------------------------------------------------
    # 4. MULTI DAMAGE BONUS
    # ------------------------------------------------------
    if total_damages == 1:
        bonus = 0
    elif total_damages <= 3:
        bonus = 2
    else:
        bonus = 4

    # ------------------------------------------------------
    # TOTAL SCORE
    # ------------------------------------------------------
    total_score = (
        part_score +
        damage_score +
        size_score +
        bonus
    )

    # ------------------------------------------------------
    # FINAL SEVERITY
    # ------------------------------------------------------
    if total_score <= 5:
        severity = "Minor"

    elif total_score <= 10:
        severity = "Moderate"

    elif total_score <= 15:
        severity = "Severe"

    else:
        severity = "Critical"

    # ------------------------------------------------------
    # FUTURE COST READY
    # ------------------------------------------------------
    return {
        "severity": severity,
        "score": total_score,
        "part_score": part_score,
        "damage_score": damage_score,
        "size_score": size_score,
        "multi_bonus": bonus,
        "damage_percent": round(percent, 2)
    }