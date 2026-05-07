"""
vehicle_catalog.py — AutoClaim Vision
======================================
Scalable OEM-approximate part prices (INR) for Indian vehicles.

ARCHITECTURE
------------
1. SEVERITY_PART_MAP  — maps every severity.py class label → catalog part key
2. SEGMENT_BASE       — price templates per vehicle segment (budget/mid/premium/luxury)
3. VEHICLE_CATALOG    — per-vehicle overrides on top of the segment base
4. lookup_vehicle_price() — called by cost_estimation.py

Adding a new car = 3 lines:
    "jazz": {"segment": "budget", "years": (2018, 2023), "overrides": {"headlight": 9_000}}
"""

from __future__ import annotations
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  SEVERITY PART MAP
#  Maps EVERY label emitted by severity.py → catalog part key.
#  This is the single source of truth — update here if severity.py changes.
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_PART_MAP: dict[str, str] = {
    # YOLO class names (exact, from damage_v2.yaml + PART_SCORE in severity.py)
    "bodypanel-dent":           "bodypanel",
    "front-windscreen-damage":  "front-windscreen",
    "headlight-damage":         "headlight",
    "rear-windscreen-damage":   "rear-windscreen",
    "runningboard-dent":        "runningboard",
    "sidemirror-damage":        "sidemirror",
    "signlight-damage":         "signlight",
    "taillight-damage":         "taillight",
    "bonnet-dent":              "bonnet",
    "boot-dent":                "boot",
    "doorouter-dent":           "doorouter",
    "fender-dent":              "fender",
    "front-bumper-dent":        "front-bumper",
    "pillar-dent":              "pillar",
    "quaterpanel-dent":         "quaterpanel",
    "rear-bumper-dent":         "rear-bumper",
    "roof-dent":                "roof",
    # Generic damage classes (map to nearest structural part)
    "scratch":                  "bodypanel",
    "crack":                    "bodypanel",
    "tire-flat":                "tire",
    "tire flat":                "tire",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENT BASE PRICES  (INR)
#  All 18 catalog part keys present in every segment.
# ─────────────────────────────────────────────────────────────────────────────
_PARTS = [
    "bonnet","boot","doorouter","fender","front-bumper","rear-bumper",
    "roof","quaterpanel","pillar","runningboard","bodypanel",
    "headlight","taillight","front-windscreen","rear-windscreen",
    "sidemirror","signlight","tire",
]

SEGMENT_BASE: dict[str, dict[str, float]] = {
    "budget": dict(zip(_PARTS, [
        10_000, 8_500, 14_000, 10_000,  6_500,  5_500,
        16_000,10_500, 18_000,  4_000,  7_500,
         7_000,  5_000, 10_000,  7_500,
         4_000,  2_000,  3_500,
    ])),
    "mid": dict(zip(_PARTS, [
        16_000,13_000, 21_000, 14_000, 10_500,  9_000,
        23_000,14_500, 26_000,  7_000, 11_500,
        15_000,  9_500, 17_000, 12_500,
         8_000,  3_500,  6_000,
    ])),
    "premium": dict(zip(_PARTS, [
        24_000,20_000, 32_000, 21_000, 17_000, 14_500,
        35_000,22_000, 42_000, 11_000, 17_000,
        26_000, 16_000, 25_000, 19_000,
        12_000,  5_500,  9_000,
    ])),
    "luxury": dict(zip(_PARTS, [
        40_000,34_000, 52_000, 35_000, 28_000, 24_000,
        58_000,37_000, 70_000, 18_000, 28_000,
        45_000, 28_000, 42_000, 32_000,
        20_000,  9_000, 14_000,
    ])),
}

# ─────────────────────────────────────────────────────────────────────────────
#  VEHICLE CATALOG
#  Each entry: segment key + year range + optional per-part overrides.
# ─────────────────────────────────────────────────────────────────────────────
_RAW: dict = {

    # ── MARUTI SUZUKI ─────────────────────────────────────────────────────
    "maruti": {
        "alto-k10":    {"segment": "budget", "years": (2022, 2025), "overrides": {"bonnet": 8_000, "doorouter": 11_000}},
        "wagonr":      {"segment": "budget", "years": (2019, 2025), "overrides": {"bonnet": 9_500}},
        "swift":       {"segment": "budget", "years": (2018, 2025), "overrides": {"bonnet": 12_000, "doorouter": 16_000, "headlight": 8_500}},
        "dzire":       {"segment": "budget", "years": (2017, 2025), "overrides": {"bonnet": 12_500, "doorouter": 15_500}},
        "baleno":      {"segment": "mid",    "years": (2019, 2025), "overrides": {"headlight": 11_000, "front-windscreen": 14_000}},
        "brezza":      {"segment": "mid",    "years": (2020, 2025), "overrides": {"bonnet": 15_000, "headlight": 15_000}},
        "ertiga":      {"segment": "mid",    "years": (2018, 2025), "overrides": {"doorouter": 22_000, "roof": 25_000}},
        "xl6":         {"segment": "mid",    "years": (2019, 2025), "overrides": {"doorouter": 23_000}},
        "grand-vitara":{"segment": "premium","years": (2022, 2025), "overrides": {"headlight": 24_000, "front-bumper": 16_000}},
        "ciaz":        {"segment": "mid",    "years": (2017, 2023), "overrides": {}},
        "s-cross":     {"segment": "mid",    "years": (2017, 2022), "overrides": {}},
    },

    # ── HYUNDAI ───────────────────────────────────────────────────────────
    "hyundai": {
        "i10-nios":  {"segment": "budget", "years": (2019, 2025), "overrides": {"bonnet": 9_000}},
        "i20":       {"segment": "mid",    "years": (2020, 2025), "overrides": {"headlight": 14_000, "taillight": 9_500}},
        "venue":     {"segment": "mid",    "years": (2019, 2025), "overrides": {"headlight": 16_000}},
        "creta":     {"segment": "premium","years": (2020, 2025), "overrides": {"headlight": 18_000, "front-windscreen": 20_000}},
        "alcazar":   {"segment": "premium","years": (2021, 2025), "overrides": {"doorouter": 34_000, "roof": 36_000}},
        "verna":     {"segment": "mid",    "years": (2023, 2025), "overrides": {"headlight": 19_000}},
        "tucson":    {"segment": "premium","years": (2022, 2025), "overrides": {"bonnet": 26_000, "headlight": 28_000}},
        "exter":     {"segment": "budget", "years": (2023, 2025), "overrides": {}},
        "aura":      {"segment": "budget", "years": (2020, 2025), "overrides": {}},
    },

    # ── TATA ──────────────────────────────────────────────────────────────
    "tata": {
        "altroz":  {"segment": "budget", "years": (2020, 2025), "overrides": {"headlight": 12_000}},
        "punch":   {"segment": "budget", "years": (2021, 2025), "overrides": {"bonnet": 14_000}},
        "tiago":   {"segment": "budget", "years": (2016, 2025), "overrides": {"bonnet": 9_000}},
        "nexon":   {"segment": "mid",    "years": (2020, 2025), "overrides": {"headlight": 17_000}},
        "harrier":  {"segment": "premium","years": (2019, 2025), "overrides": {"bonnet": 26_000, "headlight": 30_000}},
        "safari":   {"segment": "premium","years": (2021, 2025), "overrides": {"doorouter": 35_000, "roof": 38_000}},
        "tigor":    {"segment": "budget", "years": (2017, 2025), "overrides": {}},
        "curvv":    {"segment": "mid",    "years": (2024, 2025), "overrides": {"headlight": 20_000}},
    },

    # ── HONDA ─────────────────────────────────────────────────────────────
    "honda": {
        "amaze":   {"segment": "budget", "years": (2018, 2025), "overrides": {"headlight": 14_000}},
        "jazz":    {"segment": "budget", "years": (2018, 2023), "overrides": {}},
        "city":    {"segment": "mid",    "years": (2020, 2025), "overrides": {"headlight": 19_000, "front-windscreen": 19_000}},
        "elevate": {"segment": "mid",    "years": (2023, 2025), "overrides": {"headlight": 21_000}},
        "wr-v":    {"segment": "budget", "years": (2017, 2023), "overrides": {}},
    },

    # ── TOYOTA ────────────────────────────────────────────────────────────
    "toyota": {
        "glanza":           {"segment": "budget", "years": (2019, 2025), "overrides": {}},
        "urban-cruiser-hyryder": {"segment": "mid", "years": (2022, 2025), "overrides": {"headlight": 20_000}},
        "innova-crysta":    {"segment": "premium","years": (2016, 2025), "overrides": {"doorouter": 30_000, "headlight": 22_000}},
        "innova-hycross":   {"segment": "premium","years": (2022, 2025), "overrides": {"headlight": 28_000, "doorouter": 34_000}},
        "fortuner":         {"segment": "luxury", "years": (2016, 2025), "overrides": {"bonnet": 30_000, "headlight": 35_000}},
        "camry":            {"segment": "luxury", "years": (2018, 2025), "overrides": {}},
        "hilux":            {"segment": "premium","years": (2022, 2025), "overrides": {}},
    },

    # ── KIA ───────────────────────────────────────────────────────────────
    "kia": {
        "sonet":   {"segment": "mid",    "years": (2020, 2025), "overrides": {"headlight": 17_000}},
        "seltos":  {"segment": "premium","years": (2019, 2025), "overrides": {"headlight": 21_000}},
        "carens":  {"segment": "premium","years": (2022, 2025), "overrides": {"doorouter": 33_000}},
        "ev6":     {"segment": "luxury", "years": (2022, 2025), "overrides": {"headlight": 50_000}},
    },

    # ── MAHINDRA ──────────────────────────────────────────────────────────
    "mahindra": {
        "bolero-neo": {"segment": "mid",    "years": (2021, 2025), "overrides": {"roof": 28_000}},
        "xuv300":     {"segment": "mid",    "years": (2019, 2025), "overrides": {"headlight": 18_000}},
        "thar":       {"segment": "premium","years": (2020, 2025), "overrides": {"bonnet": 28_000, "roof": 38_000}},
        "scorpio-n":  {"segment": "premium","years": (2022, 2025), "overrides": {"headlight": 25_000}},
        "xuv700":     {"segment": "premium","years": (2021, 2025), "overrides": {"headlight": 32_000, "bonnet": 28_000}},
        "xuv400":     {"segment": "mid",    "years": (2023, 2025), "overrides": {}},
        "be6":        {"segment": "luxury", "years": (2025, 2026), "overrides": {}},
    },

    # ── MG ────────────────────────────────────────────────────────────────
    "mg": {
        "hector":   {"segment": "premium","years": (2019, 2025), "overrides": {"headlight": 22_000}},
        "astor":    {"segment": "mid",    "years": (2021, 2025), "overrides": {}},
        "zs-ev":    {"segment": "premium","years": (2020, 2025), "overrides": {"headlight": 26_000}},
        "gloster":  {"segment": "luxury", "years": (2020, 2025), "overrides": {"bonnet": 38_000}},
        "comet-ev": {"segment": "budget", "years": (2023, 2025), "overrides": {}},
    },

    # ── SKODA ─────────────────────────────────────────────────────────────
    "skoda": {
        "kushaq":  {"segment": "mid",    "years": (2021, 2025), "overrides": {"headlight": 19_000}},
        "slavia":  {"segment": "mid",    "years": (2022, 2025), "overrides": {}},
        "octavia": {"segment": "luxury", "years": (2021, 2025), "overrides": {}},
        "superb":  {"segment": "luxury", "years": (2016, 2025), "overrides": {"bonnet": 44_000}},
    },

    # ── VOLKSWAGEN ────────────────────────────────────────────────────────
    "volkswagen": {
        "taigun":  {"segment": "mid",    "years": (2021, 2025), "overrides": {}},
        "virtus":  {"segment": "mid",    "years": (2022, 2025), "overrides": {}},
        "tiguan":  {"segment": "luxury", "years": (2021, 2025), "overrides": {}},
    },

    # ── RENAULT ───────────────────────────────────────────────────────────
    "renault": {
        "kwid":   {"segment": "budget", "years": (2015, 2025), "overrides": {"bonnet": 7_500}},
        "kiger":  {"segment": "budget", "years": (2021, 2025), "overrides": {}},
        "triber": {"segment": "budget", "years": (2019, 2025), "overrides": {}},
    },

    # ── NISSAN ────────────────────────────────────────────────────────────
    "nissan": {
        "magnite": {"segment": "budget", "years": (2020, 2025), "overrides": {}},
        "kicks":   {"segment": "mid",    "years": (2019, 2023), "overrides": {}},
    },

    # ── JEEP ──────────────────────────────────────────────────────────────
    "jeep": {
        "compass":  {"segment": "luxury", "years": (2017, 2025), "overrides": {"bonnet": 38_000, "headlight": 40_000}},
        "meridian": {"segment": "luxury", "years": (2022, 2025), "overrides": {"doorouter": 55_000}},
        "wrangler": {"segment": "luxury", "years": (2021, 2025), "overrides": {"bonnet": 45_000}},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD RESOLVED CATALOG  (segment base + overrides merged)
# ─────────────────────────────────────────────────────────────────────────────
def _build_catalog(raw: dict) -> dict:
    catalog = {}
    for make, models in raw.items():
        catalog[make] = {}
        for model, cfg in models.items():
            seg   = cfg["segment"]
            base  = dict(SEGMENT_BASE[seg])          # copy segment template
            base.update(cfg.get("overrides", {}))    # apply vehicle overrides
            catalog[make][model] = {
                "years":   cfg["years"],
                "segment": seg,
                "parts":   base,
            }
    return catalog

VEHICLE_CATALOG = _build_catalog(_RAW)


# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MAKE_MODEL_MAP: dict[str, list[str]] = {
    make.title(): [m.replace("-", " ").title() for m in models]
    for make, models in VEHICLE_CATALOG.items()
}

ALL_MAKES: list[str] = ["Unknown"] + sorted(MAKE_MODEL_MAP.keys())


def get_models_for_make(make: str) -> list[str]:
    return ["Unknown"] + MAKE_MODEL_MAP.get(make, [])


# ─────────────────────────────────────────────────────────────────────────────
#  LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", "-").replace(" ", "-")


def lookup_vehicle_price(
    make: str,
    model: str,
    year: int,
    part_label: str,
) -> Optional[float]:
    """
    Return OEM-approximate part price (INR) or None if not found.

    Lookup order:
      1. Normalize make/model/part_label
      2. Resolve severity part_label → catalog key via SEVERITY_PART_MAP
      3. Exact catalog key lookup in vehicle's parts dict
      4. Substring fallback within parts dict
    """
    make_key  = _norm(make)
    model_key = _norm(model)
    part_norm = _norm(part_label)

    # Resolve exact severity label → catalog key
    cat_key = SEVERITY_PART_MAP.get(part_norm)
    # Substring fallback for SEVERITY_PART_MAP
    if cat_key is None:
        for sev_label, ck in SEVERITY_PART_MAP.items():
            if sev_label in part_norm or part_norm in sev_label:
                cat_key = ck
                break

    make_data = VEHICLE_CATALOG.get(make_key)
    if not make_data:
        return None

    model_data = make_data.get(model_key)
    if not model_data:
        return None

    parts = model_data["parts"]

    # Direct key hit
    if cat_key and cat_key in parts:
        return float(parts[cat_key])

    # Substring fallback in parts dict
    for pk, price in parts.items():
        if pk in part_norm or part_norm in pk:
            return float(price)

    return None


def get_vehicle_info(make: str, model: str) -> dict:
    """Return segment and year range for a vehicle (for display)."""
    make_key  = _norm(make)
    model_key = _norm(model)
    data = VEHICLE_CATALOG.get(make_key, {}).get(model_key)
    if not data:
        return {}
    return {"segment": data["segment"], "years": data["years"]}
