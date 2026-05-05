"""
Cost Estimation utilities for AutoClaim Vision.

INPUT  : part_severity dict from severity_engine.generate_severity_report()
OUTPUT : per-part repair cost + totals

Prices are PLACEHOLDERS (INR). Replace with real market data later.
Formula per part:
    repair_cost = base_price × repair_multiplier × damage_type_factor
"""

from typing import Dict, List


# ─────────────────────────────────────────
#  BASE PART PRICES  (INR, placeholder)
#  Represents cost to fully replace the part.
#  Repair actions apply a fraction of this via REPAIR_MULTIPLIER.
#  Source: replace with real Indian market data (OEM/aftermarket split later).
# ─────────────────────────────────────────
PART_BASE_PRICE: Dict[str, float] = {
    "Bodypanel-Dent":           5_000,
    "Front-Windscreen-Damage": 18_000,
    "Headlight-Damage":        12_000,
    "Rear-windscreen-Damage":  14_000,
    "RunningBoard-Dent":        4_000,
    "Sidemirror-Damage":        5_500,
    "Signlight-Damage":         2_500,
    "Taillight-Damage":         8_000,
    "bonnet-dent":             12_000,
    "boot-dent":                9_000,
    "doorouter-dent":          10_000,
    "fender-dent":              8_000,
    "front-bumper-dent":        6_500,
    "pillar-dent":             30_000,   # structural — high base
    "quaterpanel-dent":         9_000,
    "rear-bumper-dent":         5_500,
    "roof-dent":               15_000,
}

DEFAULT_BASE_PRICE: float = 6_000   # fallback for unknown parts


# ─────────────────────────────────────────
#  REPAIR ACTION  (what action to take per severity level)
#  WHY: same part, same price → different action based on how bad damage is.
#       Low damage → polish only (cheap). Critical → full replace (expensive).
# ─────────────────────────────────────────
REPAIR_ACTION: Dict[str, str] = {
    "Low":      "Polish / Touch-up",
    "Medium":   "Panel Repair",
    "High":     "Panel Replacement",
    "Critical": "Full Replacement + Structural Inspection",
}

# Fraction of base price charged per repair action.
# Critical > 1.0 because structural inspection adds labor beyond part cost.
REPAIR_MULTIPLIER: Dict[str, float] = {
    "Low":      0.12,
    "Medium":   0.45,
    "High":     0.85,
    "Critical": 1.25,
}


# ─────────────────────────────────────────
#  DAMAGE TYPE COST FACTOR
#  WHY: a crack costs more to repair than a dent of the same area —
#       cracks risk spreading, require filler/sealing, more prep labor.
#       scratch is cheapest (surface only).
# ─────────────────────────────────────────
DAMAGE_TYPE_COST_FACTOR: Dict[str, float] = {
    "scratch": 0.80,
    "dent":    1.00,
    "crack":   1.30,
    "damage":  1.20,
    "flat":    1.50,   # tyre replacement + alignment check
}

DEFAULT_DAMAGE_FACTOR: float = 1.00

# Labor overhead as fraction of total parts cost.
# WHY separate: labor rate varies by city/garage — easy to tune one constant.
LABOR_OVERHEAD: float = 0.20    # 20% of parts total


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def _normalize(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def _get_base_price(part_label: str) -> float:
    label = _normalize(part_label)
    for key, price in PART_BASE_PRICE.items():
        if _normalize(key) in label:
            return price
    return DEFAULT_BASE_PRICE


def _worst_damage_type(damage_types: List[str]) -> str:
    """
    Return highest-cost damage type from list.
    WHY: a part may have both scratch + crack detected.
         Use worst type to drive cost — conservative estimate.
    """
    order = ["flat", "crack", "damage", "dent", "scratch"]
    for dtype in order:
        if dtype in damage_types:
            return dtype
    return "dent"


# ─────────────────────────────────────────
#  MAIN ESTIMATION FUNCTION
# ─────────────────────────────────────────

def estimate_cost(part_severity: Dict) -> Dict:
    """
    Estimate repair cost from severity report's part_severity dict.

    Args:
        part_severity: dict of { part_label: { severity_score, severity_level,
                                               damage_types, damage_count,
                                               max_area_ratio } }
                       as returned by severity_engine.generate_severity_report()

    Returns:
        {
            "line_items":   list of per-part cost dicts,
            "parts_total":  float,
            "labor_total":  float,
            "grand_total":  float,
            "currency":     "INR",
            "note":         str
        }
    """
    line_items = []
    parts_total = 0.0

    for part_label, info in (part_severity or {}).items():
        severity_lvl  = info.get("severity_level", "Medium")
        damage_types  = info.get("damage_types", ["dent"])
        severity_score = info.get("severity_score", 0.0)

        base_price      = _get_base_price(part_label)
        repair_mult     = REPAIR_MULTIPLIER.get(severity_lvl, 0.45)
        worst_dtype     = _worst_damage_type(damage_types)
        dtype_factor    = DAMAGE_TYPE_COST_FACTOR.get(worst_dtype, DEFAULT_DAMAGE_FACTOR)
        action          = REPAIR_ACTION.get(severity_lvl, "Panel Repair")

        part_cost = base_price * repair_mult * dtype_factor
        part_cost = round(part_cost, 2)
        parts_total += part_cost

        line_items.append({
            "part":            part_label,
            "severity_level":  severity_lvl,
            "severity_score":  severity_score,
            "repair_action":   action,
            "damage_types":    ", ".join(damage_types),
            "base_price":      base_price,
            "part_cost":       part_cost,
        })

    # sort by cost descending — most expensive damage first
    line_items.sort(key=lambda x: x["part_cost"], reverse=True)

    labor_total = round(parts_total * LABOR_OVERHEAD, 2)
    grand_total = round(parts_total + labor_total, 2)

    return {
        "line_items":  line_items,
        "parts_total": round(parts_total, 2),
        "labor_total": labor_total,
        "grand_total": grand_total,
        "currency":    "INR",
        "note":        "Prices are placeholder estimates. Replace PART_BASE_PRICE with real market data.",
    }
