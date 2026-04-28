"""
cost_estimate.py  —  train/
----------------------------
Repair cost table for all 20 damage classes in damage.yaml
Indian Rupees (₹), 2024-25 market rates.

Classes (matching damage.yaml exactly):
  0:  Bodypanel-Dent          11: fender-dent
  1:  Front-Windscreen-Damage 12: front-bumper-dent
  2:  Headlight-Damage        13: pillar-dent
  3:  Rear-windscreen-Damage  14: quaterpanel-dent
  4:  RunningBoard-Dent       15: rear-bumper-dent
  5:  Sidemirror-Damage       16: roof-dent
  6:  Signlight-Damage        17: scratch
  7:  Taillight-Damage        18: crack
  8:  bonnet-dent             19: tire flat
  9:  boot-dent
  10: doorouter-dent

Severity tiers: Minor / Moderate / Severe / Critical
Each → (min_cost ₹, max_cost ₹) for PARTS + PAINT only.
Labour added separately: 15% of min, 22% of max.

Sources: GoMechanic, CarDekho, mycarhelpline.com, windshieldstore.in,
         team-bhp forums, OkCar, CarVaidya (verified Apr 2025)
NOTE: SUV / luxury cars cost 1.5x-2.5x these figures.
"""

SEVERITY_LEVELS = ["Minor", "Moderate", "Severe", "Critical"]

COST_TABLE: dict[str, dict[str, tuple[int, int]]] = {

    # Glass
    "Front-Windscreen-Damage": {
        "Minor":    (500,    2_500),
        "Moderate": (3_500,  9_000),
        "Severe":   (9_000,  20_000),
        "Critical": (20_000, 45_000),
    },
    "Rear-windscreen-Damage": {
        "Minor":    (400,    1_500),
        "Moderate": (2_000,  6_000),
        "Severe":   (6_000,  12_000),
        "Critical": (12_000, 25_000),
    },

    # Lights
    "Headlight-Damage": {
        "Minor":    (800,    3_000),
        "Moderate": (3_000,  10_000),
        "Severe":   (10_000, 25_000),
        "Critical": (25_000, 60_000),
    },
    "Taillight-Damage": {
        "Minor":    (500,    2_000),
        "Moderate": (2_000,  7_000),
        "Severe":   (7_000,  18_000),
        "Critical": (18_000, 40_000),
    },
    "Signlight-Damage": {
        "Minor":    (300,    800),
        "Moderate": (800,    3_000),
        "Severe":   (3_000,  8_000),
        "Critical": (8_000,  18_000),
    },

    # Mirror
    "Sidemirror-Damage": {
        "Minor":    (500,    2_000),
        "Moderate": (2_000,  6_000),
        "Severe":   (6_000,  15_000),
        "Critical": (15_000, 35_000),
    },

    # Dents
    "bonnet-dent": {
        "Minor":    (2_000,  6_000),
        "Moderate": (6_000,  18_000),
        "Severe":   (18_000, 40_000),
        "Critical": (40_000, 80_000),
    },
    "boot-dent": {
        "Minor":    (2_000,  6_000),
        "Moderate": (6_000,  16_000),
        "Severe":   (16_000, 35_000),
        "Critical": (35_000, 70_000),
    },
    "doorouter-dent": {
        "Minor":    (2_500,  7_000),
        "Moderate": (7_000,  20_000),
        "Severe":   (20_000, 45_000),
        "Critical": (45_000, 90_000),
    },
    "fender-dent": {
        "Minor":    (2_000,  5_000),
        "Moderate": (5_000,  14_000),
        "Severe":   (14_000, 30_000),
        "Critical": (30_000, 60_000),
    },
    "front-bumper-dent": {
        "Minor":    (1_500,  4_000),
        "Moderate": (4_000,  12_000),
        "Severe":   (12_000, 25_000),
        "Critical": (25_000, 55_000),
    },
    "rear-bumper-dent": {
        "Minor":    (1_500,  4_000),
        "Moderate": (4_000,  11_000),
        "Severe":   (11_000, 22_000),
        "Critical": (22_000, 48_000),
    },
    "pillar-dent": {
        "Minor":    (5_000,  12_000),
        "Moderate": (12_000, 30_000),
        "Severe":   (30_000, 70_000),
        "Critical": (70_000, 1_50_000),
    },
    "quaterpanel-dent": {
        "Minor":    (3_000,  8_000),
        "Moderate": (8_000,  22_000),
        "Severe":   (22_000, 50_000),
        "Critical": (50_000, 1_00_000),
    },
    "roof-dent": {
        "Minor":    (3_000,  8_000),
        "Moderate": (8_000,  25_000),
        "Severe":   (25_000, 60_000),
        "Critical": (60_000, 1_20_000),
    },
    "Bodypanel-Dent": {
        "Minor":    (1_500,  4_500),
        "Moderate": (4_500,  12_000),
        "Severe":   (12_000, 28_000),
        "Critical": (28_000, 55_000),
    },
    "RunningBoard-Dent": {
        "Minor":    (1_000,  3_000),
        "Moderate": (3_000,  8_000),
        "Severe":   (8_000,  18_000),
        "Critical": (18_000, 35_000),
    },

    # New v2 classes
    "scratch": {
        "Minor":    (500,    2_000),
        "Moderate": (2_000,  6_000),
        "Severe":   (6_000,  15_000),
        "Critical": (15_000, 30_000),
    },
    "crack": {
        "Minor":    (800,    3_000),
        "Moderate": (3_000,  10_000),
        "Severe":   (10_000, 25_000),
        "Critical": (25_000, 60_000),
    },
    "tire flat": {
        "Minor":    (500,    1_500),
        "Moderate": (2_000,  6_000),
        "Severe":   (6_000,  14_000),
        "Critical": (14_000, 30_000),
    },
}

FALLBACK: dict[str, tuple[int, int]] = {
    "Minor":    (2_000,  6_000),
    "Moderate": (6_000,  18_000),
    "Severe":   (18_000, 50_000),
    "Critical": (50_000, 1_20_000),
}

DISCLAIMER = (
    "Costs are indicative estimates for economy/mid-segment cars at local garages "
    "in Tier-1 & Tier-2 Indian cities (2024-25 rates). "
    "SUVs and luxury vehicles typically cost 1.5x-2.5x more. "
    "Prices include parts + paint but exclude towing, wheel alignment, or electrical work. "
    "Obtain a written quote from a certified mechanic before filing any insurance claim."
)


def get_cost_range(class_name: str, severity: str) -> tuple[int, int]:
    """Return (min_cost, max_cost) for a class + severity pair."""
    tier = COST_TABLE.get(class_name, FALLBACK)
    return tier.get(severity, tier.get("Severe", (5_000, 20_000)))


def build_estimate(detections: list[dict]) -> dict:
    """
    Build full cost estimate from detection list.

    Each detection dict needs:
        class_name  str    — exact name from damage.yaml
        severity    str    — Minor / Moderate / Severe / Critical
        confidence  float  — model confidence 0-1

    Returns dict with line_items, subtotal_min/max,
    labour_min/max, total_min/max, disclaimer.
    """
    if not detections:
        return _empty()

    seen: set[str] = set()
    line_items: list[dict] = []

    for det in detections:
        cls = det["class_name"]
        sev = det["severity"]
        if cls in seen:
            continue
        seen.add(cls)

        min_c, max_c = get_cost_range(cls, sev)
        line_items.append({
            "class_name": cls,
            "severity":   sev,
            "confidence": det.get("confidence", 0.0),
            "min_cost":   min_c,
            "max_cost":   max_c,
            "in_table":   cls in COST_TABLE,
        })

    subtotal_min = sum(i["min_cost"] for i in line_items)
    subtotal_max = sum(i["max_cost"] for i in line_items)
    labour_min   = int(subtotal_min * 0.15)
    labour_max   = int(subtotal_max * 0.22)

    return {
        "line_items":   line_items,
        "subtotal_min": subtotal_min,
        "subtotal_max": subtotal_max,
        "labour_min":   labour_min,
        "labour_max":   labour_max,
        "total_min":    subtotal_min + labour_min,
        "total_max":    subtotal_max + labour_max,
        "disclaimer":   DISCLAIMER,
    }


def _empty() -> dict:
    return {
        "line_items":   [],
        "subtotal_min": 0, "subtotal_max": 0,
        "labour_min":   0, "labour_max":   0,
        "total_min":    0, "total_max":    0,
        "disclaimer":   "No damage detected.",
    }


def format_inr(amount: int) -> str:
    """Format int as Indian rupee string e.g. 120000 -> Rs.1,20,000"""
    s = str(amount)
    if len(s) <= 3:
        return f"Rs.{s}"
    last3, rest = s[-3:], s[:-3]
    groups: list[str] = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return "Rs." + ",".join(groups) + "," + last3
