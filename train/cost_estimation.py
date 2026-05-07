"""
Cost Estimation Module — AutoClaim Vision
==========================================
Calculates vehicle damage repair costs using EXACT severity scores
produced by severity.py. No re-normalization, no re-classification.

Pipeline:
    detection → severity.py (generate_severity_report)
              → part_severity[part_name]["severity_score"]  # 0–100
              → cost_estimation.py (estimate_cost)
              → per-part cost + totals

Core formula (per part):
    estimated_cost = base_price × repair_multiplier × severity_score

    Where:
        severity_score   — exact value from severity.py (0.0 – 100.0)
        repair_multiplier — depends on severity band (same thresholds as severity.py)
        base_price       — full replacement cost of the part (INR)

Repair multiplier sizing for score-range formula:
    Low      (0 – 24.99)  → 0.008   Minor surface damage
    Medium   (25 – 49.99) → 0.012   Moderate panel damage
    High     (50 – 74.99) → 0.018   Severe structural damage
    Critical (75 – 100)   → 0.025   Full replacement needed

Example:  bonnet-dent, score=45, base=₹12,000
    cost = 12000 × 0.012 × 45 = ₹6,480
"""

import logging
from typing import Dict, List, Optional
from train.severity import severity_level, normalize_text

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("cost_estimation")
# ─────────────────────────────────────────────────────────────────────────────

# ── Part base price (INR) — 2024-25 Indian OEM/authorised workshop market ──────
# Full repair/replacement cost for each part at 100% damage.
# Source: mid-segment sedan (Swift, i20, City equivalent).
PART_BASE_PRICE: Dict[str, float] = {
    "Bodypanel-Dent":           8_000,
    "Front-Windscreen-Damage": 20_000,
    "Headlight-Damage":        14_000,
    "Rear-windscreen-Damage":  16_000,
    "RunningBoard-Dent":        5_000,
    "Sidemirror-Damage":        7_500,
    "Signlight-Damage":         3_500,
    "Taillight-Damage":        10_000,
    "bonnet-dent":             18_000,
    "boot-dent":               14_000,
    "doorouter-dent":          16_000,
    "fender-dent":             12_000,
    "front-bumper-dent":       10_000,
    "pillar-dent":             45_000,   # structural — cabin integrity
    "quaterpanel-dent":        13_000,
    "rear-bumper-dent":         8_500,
    "roof-dent":               22_000,
}

DEFAULT_PART_PRICE: float = 10_000   # fallback when part not in table

# ── Damage type factor — what fraction of the part base price this damage
#    type typically costs to repair (0.0 – 1.0 scale).
#    scratch < dent < crack < damage (severity-ordered).
DAMAGE_TYPE_FACTOR: Dict[str, float] = {
    "scratch": 0.25,   # surface polish / touch-up  → 25% of part cost
    "dent":    0.60,   # panel repair / PDR / filler → 60% of part cost
    "crack":   0.85,   # structural filler + repaint → 85% of part cost
    "damage":  1.00,   # full replacement / major work → 100% of part cost
    "flat":    0.50,   # tyre + alignment              → 50% of part cost
}

DEFAULT_BASE_PRICE: float = 10_000   # final fallback


REPAIR_MULTIPLIER: Dict[str, float] = {
    "Low":      0.008,   # Minor     — surface polish / touch-up
    "Medium":   0.012,   # Moderate  — panel repair / filling
    "High":     0.018,   # Severe    — panel replacement
    "Critical": 0.025,   # Critical  — full replacement + structural inspection
}

REPAIR_ACTION: Dict[str, str] = {
    "Low":      "Polish / Touch-up",
    "Medium":   "Panel Repair",
    "High":     "Panel Replacement",
    "Critical": "Full Replacement + Structural Inspection",
}

# Labor overhead applied to total parts cost (painting, masking, finishing).
LABOR_OVERHEAD: float = 0.20   # 20%


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_base_price(part_name: str, damage_type: str) -> float:
    """
    Compute base price using BOTH part name AND damage type.

    base_price = PART_BASE_PRICE[part] × DAMAGE_TYPE_FACTOR[damage_type]

    This prevents hallucination:
      - Part anchor: reflects real market cost for that specific part
      - Damage factor: reflects how much of the part cost this type requires

    Example:
      fender-dent + scratch → 12000 × 0.25 = ₹3,000
      fender-dent + crack   → 12000 × 0.85 = ₹10,200
      pillar-dent + damage  → 45000 × 1.00 = ₹45,000
    """
    # Part base — substring match using normalize_text from severity.py
    label = normalize_text(part_name)
    part_price = DEFAULT_PART_PRICE
    for key, price in PART_BASE_PRICE.items():
        if normalize_text(key) in label or label in normalize_text(key):
            part_price = price
            break
    else:
        logger.warning("No part price for '%s'. Using ₹%s.", part_name, DEFAULT_PART_PRICE)

    # Damage type factor
    dtype_key = normalize_text(damage_type)
    dtype_factor = DAMAGE_TYPE_FACTOR.get(dtype_key, 0.60)   # default to dent-level
    if dtype_key not in DAMAGE_TYPE_FACTOR:
        logger.warning("Unknown damage_type '%s'. Using factor %.2f.", damage_type, dtype_factor)

    combined = round(part_price * dtype_factor, 2)
    logger.debug("Base price: %s × %s(%.2f) = ₹%.0f", part_name, damage_type, dtype_factor, combined)
    return combined


def _validate_part_entry(part_name: str, info: Dict) -> Optional[str]:
    """
    Validate a part entry from part_severity.

    Returns an error string if invalid, None if valid.
    """
    if not part_name or not isinstance(part_name, str):
        return "Part name is missing or not a string."

    severity_score = info.get("severity_score")
    if severity_score is None:
        return f"Part '{part_name}': severity_score is missing."
    if not isinstance(severity_score, (int, float)):
        return f"Part '{part_name}': severity_score is not numeric (got {type(severity_score).__name__})."
    if not (0.0 <= float(severity_score) <= 100.0):
        return f"Part '{part_name}': severity_score {severity_score} out of expected range [0, 100]."

    return None   # valid


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ESTIMATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_cost(part_severity: Dict) -> Dict:
    """
    Calculate repair cost for each damaged part using exact severity scores
    from severity.py — no re-computation, no re-normalization.

    Args:
        part_severity (Dict):
            Directly from generate_severity_report()["part_severity"].
            Structure:
                {
                    "bonnet-dent": {
                        "severity_score": 45.2,      ← exact aggregated score [0-100]
                        "severity_level": "Medium",
                        "damage_types":  ["dent"],
                        "damage_count":  2,
                        "max_area_ratio": 0.031,
                    },
                    ...
                }

    Returns:
        {
            "line_items":   List[Dict]  — one row per part, sorted by cost desc
            "parts_total":  float       — sum of all repair costs (INR)
            "labor_total":  float       — 20% labor overhead (INR)
            "grand_total":  float       — parts + labor (INR)
            "currency":     "INR"
            "skipped_parts": List[str] — parts skipped due to validation errors
            "note":         str
        }
    """
    if not part_severity:
        logger.warning("estimate_cost() called with empty part_severity.")
        return {
            "line_items":    [],
            "parts_total":   0.0,
            "labor_total":   0.0,
            "grand_total":   0.0,
            "currency":      "INR",
            "skipped_parts": [],
            "note":          "No damaged parts provided.",
        }

    line_items    = []
    parts_total   = 0.0
    skipped_parts = []

    for part_name, info in part_severity.items():

        # ── Validate entry ────────────────────────────────────────────────
        error = _validate_part_entry(part_name, info)
        if error:
            logger.error("Skipping part — %s", error)
            skipped_parts.append(part_name)
            continue

        # ── Extract EXACT severity_score from severity.py output ──────────
        #    This is: round(aggregate_part(scores) * 100.0, 2)
        #    DO NOT re-compute or normalize this value.
        severity_score: float = float(info["severity_score"])

        # ── Derive severity_level using the SAME imported function ─────────
        #    Guarantees: same thresholds as severity.py
        #    (<25 → Low, <50 → Medium, <75 → High, ≥75 → Critical)
        sev_level: str = severity_level(severity_score)

        # ── Extract part + damage_type directly from the info dict ───────────
        # severity.py now keys part_severity by "{part}__{damage_type}"
        # and stores both fields inside the value dict.
        part_label      = info.get("part", part_name)          # e.g. "bonnet-dent"
        damage_type     = info.get("damage_type", "dent")      # e.g. "dent"
        damage_types    = info.get("damage_types", [damage_type])
        damage_count    = info.get("damage_count", 1)
        max_area_ratio  = info.get("max_area_ratio", 0.0)

        # ── Look up pricing: part name + damage type (no hallucination) ──────
        base_price   = _get_base_price(part_label, damage_type)
        repair_mult  = REPAIR_MULTIPLIER[sev_level]
        action       = REPAIR_ACTION[sev_level]

        # ── Core formula ──────────────────────────────────────────────────
        #    estimated_cost = base_price × repair_multiplier × severity_score
        #
        #    severity_score used AS-IS (0–100) — no division, no clamping.
        #    repair_multiplier is sized so the product yields realistic INR values.
        estimated_cost = base_price * repair_mult * severity_score
        estimated_cost = round(estimated_cost, 2)
        parts_total   += estimated_cost

        logger.debug(
            "%-30s | score=%5.1f | level=%-8s | base=₹%6.0f | mult=%5.3f | cost=₹%8.2f",
            part_name, severity_score, sev_level, base_price, repair_mult, estimated_cost,
        )

        line_items.append({
            # ── Identification ────────────────────────────────────────────
            "part":             part_label,            # e.g. "bonnet-dent"
            "damage_type":      damage_type,           # e.g. "dent" — drives base price
            "damage_types":     damage_types,          # list, backward compat
            "damage_count":     damage_count,

            # ── Exact severity from severity.py ───────────────────────────
            "severity_score":   round(severity_score, 2),
            "severity_level":   sev_level,

            # ── Pricing breakdown ─────────────────────────────────────────
            "base_price":        base_price,           # from DAMAGE_TYPE_BASE_PRICE
            "repair_multiplier": repair_mult,
            "estimated_cost":    estimated_cost,
            "part_cost":         estimated_cost,       # alias for backward compat
            "repair_action":     action,

            # ── Metadata ──────────────────────────────────────────────────
            "max_area_ratio":   round(max_area_ratio, 4),
        })

    # Sort: highest estimated cost first
    line_items.sort(key=lambda x: x["estimated_cost"], reverse=True)

    labor_total = round(parts_total * LABOR_OVERHEAD, 2)
    grand_total = round(parts_total + labor_total, 2)

    logger.info(
        "Cost estimation complete: %d parts | parts=₹%.2f | labor=₹%.2f | total=₹%.2f | skipped=%d",
        len(line_items), parts_total, labor_total, grand_total, len(skipped_parts),
    )

    return {
        "line_items":    line_items,
        "parts_total":   round(parts_total, 2),
        "labor_total":   labor_total,
        "grand_total":   grand_total,
        "currency":      "INR",
        "skipped_parts": skipped_parts,
        "note":          (
            "Costs calculated from exact severity scores produced by severity.py. "
            "Formula: base_price × repair_multiplier × severity_score. "
            "Replace PART_BASE_PRICE with OEM/regional market data for production use."
        ),
    }
