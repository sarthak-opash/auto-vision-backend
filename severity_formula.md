# Severity Score Calculation Formula

## Overview

The scoring pipeline has **3 stages**: per-detection → per-part → overall.

---

## Stage 1: Per-Detection Score (`compute_damage_score`)

Each detected damage gets a score in **[0, 1]** using four components:

```
S_damage = 0.40 × A + 0.30 × T + 0.15 × P + 0.15 × C
```

| Component | Formula | Range | What it measures |
|---|---|---|---|
| **A** (Area) | `(area / (area + 0.25))^1.2` | [0, ~0.8] | How much of the part is damaged |
| **T** (Type) | Lookup from `DAMAGE_SCORE` table | [0.2, 0.8] | Severity of damage type |
| **P** (Part) | `PART_SCORE[part] / 10.0` | [0.5, 1.0] | Importance of the damaged part |
| **C** (Conf) | `confidence²` | [0, 1] | Model confidence (squared to suppress) |

### DAMAGE_SCORE Table (T values)
| Damage Type | Score |
|---|---|
| scratch | 0.2 |
| dent | 0.45 |
| crack | 0.6 |
| damage | 0.7 |
| flat | 0.8 |

### Area Calculation
- **With part detection**: `area = overlap(damage_bbox, part_bbox) / area(part_bbox)` → ratio of part covered
- **Without part detection**: `area = area(damage_bbox) / (img_width × img_height)` → ratio of full image

### Why `(area / (area + 0.25))^1.2`?
This is a **saturation curve** that prevents zoom bias:

```
area = 0.02  →  A = 0.05   (tiny scratch → very low)
area = 0.05  →  A = 0.11   (small dent → low)
area = 0.10  →  A = 0.21   (medium damage → moderate)
area = 0.30  →  A = 0.47   (large damage → higher)
area = 0.80  →  A = 0.73   (massive → caps out)
```

Even if a zoomed photo makes the bbox fill the whole image, A caps at ~0.8, not 1.0.

---

## Stage 2: Per-Part Aggregation (`aggregate_part`)

If one part has multiple damages, they are combined using **union probability**:

```
S_part = 1 - Π(1 - S_damage_i)
```

### Example
- Part has 2 damages: scores = [0.3, 0.4]
- `S_part = 1 - (1-0.3) × (1-0.4) = 1 - 0.7 × 0.6 = 1 - 0.42 = 0.58`

### Noise Penalty
If more than 3 detections on the same part (likely duplicates/noise):
```
if n > 3:  S_part *= 0.90^(n-3)
```

---

## Stage 3: Overall Severity Score (0–100)

### Step 3a: Blend max and average part scores
```
raw = 0.6 × max(part_scores) + 0.4 × mean(part_scores)
```
> Max-biased so one heavily damaged part dominates (a totaled hood shouldn't be averaged down by an intact bumper).

### Step 3b: Multi-part bonus
```
if n_parts > 1:
    bonus = min(0.15, 0.06 × √(n_parts - 1))
    raw = min(1.0, raw + bonus)
```
> More damaged parts = worse situation. Bonus caps at +0.15.

### Step 3c: Non-linear scaling to 0–100
```
severity = raw × 100
severity = severity^1.25 / 100^0.25
```
> Power of 1.25 stretches the mid-range for better differentiation. Low scores stay low, high scores get pushed higher.

---

## Severity Levels

| Score Range | Level |
|---|---|
| 0 – 24 | **Low** |
| 25 – 49 | **Medium** |
| 50 – 74 | **High** |
| 75 – 100 | **Critical** |

---

## End-to-End Example

### Input: Single scratch on fender, confidence=0.6, area=0.02
```
A = (0.02 / 0.27)^1.2 = 0.074^1.2 = 0.047
T = 0.2     (scratch)
P = 7/10 = 0.7   (fender-dent score)
C = 0.6² = 0.36

S_damage = 0.40×0.047 + 0.30×0.2 + 0.15×0.7 + 0.15×0.36
         = 0.019 + 0.06 + 0.105 + 0.054
         = 0.238

S_part = 0.238  (only one damage)

raw = 0.6×0.238 + 0.4×0.238 = 0.238  (only one part)

severity = 0.238 × 100 = 23.8
severity = 23.8^1.25 / 100^0.25 = 58.8 / 3.162 = 18.6

→ Score: 18.6 (Low)
```

### Input: Windshield damage + bonnet dent + headlight damage
```
Windshield: S = 0.65 → Part: 0.65
Bonnet:     S = 0.50 → Part: 0.50
Headlight:  S = 0.48 → Part: 0.48

raw = 0.6×0.65 + 0.4×mean(0.65,0.50,0.48)
    = 0.39 + 0.4×0.543 = 0.39 + 0.217 = 0.607

bonus = min(0.15, 0.06×√2) = min(0.15, 0.085) = 0.085
raw = 0.607 + 0.085 = 0.692

severity = 69.2
severity = 69.2^1.25 / 100^0.25 = 198.4 / 3.162 = 62.7

→ Score: 62.7 (High)
```
