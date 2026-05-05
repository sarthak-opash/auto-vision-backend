"""
report.py
---------
Stage 5 — Generate PDF inspection report for AutoClaim Vision.

Consumes output from:
  - severity_engine.generate_severity_report()
  - cost_engine.estimate_cost()

Requires: pip install fpdf2
"""

from __future__ import annotations
import datetime
from pathlib import Path

try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ─────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────
C_DARK   = (28,  28,  28)
C_WHITE  = (255, 255, 255)
C_LIGHT  = (245, 247, 250)
C_ACCENT = (15,  52,  96)

SEVERITY_COLOURS = {
    "Low":      (25,  135,  84),
    "Medium":   (255, 193,   7),
    "High":     (253, 126,  20),
    "Critical": (220,  53,  69),
}


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def _clean(txt: str) -> str:
    """Strip/replace anything outside latin-1 (Helvetica safe)."""
    txt = str(txt)
    subs = {
        "₹": "Rs.", "→": "->", "←": "<-", "•": "-", "–": "-", "—": "-",
        "×": "x",  "≥": ">=", "≤": "<=", "≈": "~",
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "✅": "OK", "✓": "OK", "✗": "No",
        "🟢": "", "🟡": "", "🟠": "", "🔴": "", "⚪": "",
    }
    for ch, sub in subs.items():
        txt = txt.replace(ch, sub)
    return txt.encode("latin-1", errors="replace").decode("latin-1")


def _fmt_inr(amount: float) -> str:
    return f"Rs. {amount:,.0f}"


class _PDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 22, 15)
        self.add_page()

    def header(self):
        self.set_fill_color(*C_ACCENT)
        self.rect(0, 0, 210, 16, "F")
        self.set_y(3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_WHITE)
        self.cell(0, 10, "AutoClaim Vision  |  AI-Powered Damage Assessment Report", align="C")
        self.set_text_color(*C_DARK)
        self.set_y(20)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(160, 160, 160)
        self.cell(
            0, 8,
            _clean(
                f"Page {self.page_no()}/{{nb}}  |  "
                f"Generated {datetime.date.today().strftime('%d %b %Y')}  |  "
                "Indicative only — not a certified insurance assessment"
            ),
            align="C",
        )
        self.set_text_color(*C_DARK)


def _section(pdf: _PDF, title: str):
    pdf.set_fill_color(*C_ACCENT)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, _clean(f"  {title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(*C_DARK)
    pdf.ln(2)


def _kv(pdf: _PDF, key: str, value: str, shade: bool = False):
    pdf.set_fill_color(*(C_LIGHT if shade else C_WHITE))
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(65, 6, _clean(f"  {key}"), fill=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _clean(str(value)), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)


def _thead(pdf: _PDF, cols: list[tuple[str, int]]):
    pdf.set_fill_color(*C_DARK)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font("Helvetica", "B", 8)
    for lbl, w in cols:
        pdf.cell(w, 6, _clean(f" {lbl}"), fill=True, border="B")
    pdf.ln()
    pdf.set_text_color(*C_DARK)


def _trow(pdf: _PDF, vals: list[tuple[str, int]], shade: bool = False):
    pdf.set_fill_color(*(C_LIGHT if shade else C_WHITE))
    pdf.set_font("Helvetica", "", 8)
    for txt, w in vals:
        pdf.cell(w, 5, _clean(f" {txt}"), fill=True, border="B")
    pdf.ln()


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def generate_report(
    severity_result: dict,
    cost_result:     dict,
    annotated_image_path: str | None = None,
    output_path: str | None = None,
) -> bytes:
    """
    Build PDF report from severity + cost outputs.

    Args:
        severity_result:      generate_severity_report() output
        cost_result:          estimate_cost() output
        annotated_image_path: path to YOLO-annotated image (optional)
        output_path:          if given, also save PDF to disk

    Returns:
        bytes — raw PDF
    """
    if not FPDF_AVAILABLE:
        raise ImportError("Run:  pip install fpdf2")

    pdf = _PDF()
    pdf.alias_nb_pages()
    now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

    sev_level = severity_result.get("severity_level", "N/A")
    sev_score = severity_result.get("severity_score", 0.0)

    # ── TITLE BLOCK ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 17)
    pdf.cell(0, 10, "Vehicle Damage Inspection Report",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"Generated: {now}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*C_DARK)
    pdf.ln(3)

    # severity badge
    sev_col = SEVERITY_COLOURS.get(sev_level, C_DARK)
    pdf.set_fill_color(*sev_col)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(90, 9, _clean(f"  Overall Severity: {sev_level}  ({sev_score}/100)"), fill=True)
    pdf.ln(6)
    pdf.set_text_color(*C_DARK)
    pdf.ln(3)

    sec = 1

    # ── ANNOTATED IMAGE ───────────────────────────────────────────────────────
    if annotated_image_path and Path(annotated_image_path).exists():
        _section(pdf, f"{sec}. Annotated Damage Image")
        sec += 1
        img_w = 165
        pdf.image(annotated_image_path, x=(210 - img_w) / 2, w=img_w)
        pdf.ln(3)

    # ── DAMAGE DETECTIONS ─────────────────────────────────────────────────────
    _section(pdf, f"{sec}. Damage Detections")
    sec += 1

    damage_table = severity_result.get("damage_table", [])
    if damage_table:
        cols = [
            ("Part",          70),
            ("Damage Type",   35),
            ("Confidence",    28),
            ("Area Ratio",    25),
            ("Score",         22),
        ]
        _thead(pdf, cols)
        for i, det in enumerate(damage_table):
            _trow(pdf, [
                (det.get("part", ""),                       70),
                (det.get("damage_type", ""),                35),
                (f"{det.get('confidence', 0):.0%}",         28),
                (f"{det.get('area', 0):.3f}",               25),
                (f"{det.get('damage_score', 0):.3f}",       22),
            ], shade=(i % 2 == 0))
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, f"  {len(damage_table)} detection(s) total.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*C_DARK)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "  No damage detected.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── SEVERITY ANALYSIS ─────────────────────────────────────────────────────
    # Estimate height needed: section header(10) + 3 kv rows(18) + padding(6)
    # + per-part table header(6) + rows(5 per part) + buffer(10)
    part_count = len(severity_result.get("part_severity", {}))
    estimated_h = 10 + 18 + 6 + 6 + (part_count * 5) + 10
    remaining = pdf.h - pdf.get_y() - pdf.b_margin
    if remaining < estimated_h:
        pdf.add_page()
    _section(pdf, f"{sec}. Severity Analysis")
    sec += 1

    _kv(pdf, "Overall Level",    sev_level,              shade=True)
    _kv(pdf, "Overall Score",    f"{sev_score} / 100",   shade=False)
    _kv(pdf, "Parts Affected",
        ", ".join(severity_result.get("detected_parts", [])) or "None",
        shade=True)
    pdf.ln(3)

    # per-part severity table
    part_severity = severity_result.get("part_severity", {})
    if part_severity:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "  Per-Part Breakdown:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        cols = [
            ("Part",             50),
            ("Level",            22),
            ("Score",            20),
            ("Damage Types",     35),
            ("Structural",       21),
            ("Safety Critical",  32),
        ]
        _thead(pdf, cols)
        for i, (part_label, info) in enumerate(part_severity.items()):
            _trow(pdf, [
                (part_label,                                     50),
                (info.get("severity_level", ""),                 22),
                (f"{info.get('severity_score', 0):.1f}",         20),
                (", ".join(info.get("damage_types", [])),        35),
                ("Yes" if info.get("is_structural")  else "No",  21),
                ("Yes" if info.get("is_safety_critical") else "No", 32),
            ], shade=(i % 2 == 0))
    pdf.ln(3)

    # ── CRITICAL FLAGS ────────────────────────────────────────────────────────
    critical_flags = severity_result.get("critical_flags", [])
    if critical_flags:
        _section(pdf, f"{sec}. Critical Flags")
        sec += 1
        pdf.set_fill_color(255, 235, 235)
        pdf.set_font("Helvetica", "", 9)
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        for flag in critical_flags:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_w, 5, _clean(f"  !! {flag}"),
                           fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
        pdf.ln(2)

    # ── COST ESTIMATE ─────────────────────────────────────────────────────────
    line_items = cost_result.get("line_items", [])
    estimated_h = 10 + 6 + (len(line_items) * 5) + 25  # header + table + totals + buffer
    remaining = pdf.h - pdf.get_y() - pdf.b_margin

    if remaining < estimated_h:
        pdf.add_page()
    _section(pdf, f"{sec}. Repair Cost Estimate")
    sec += 1

    line_items = cost_result.get("line_items", [])
    if line_items:
        cols = [
            ("Part",            68),
            ("Repair Action",   60),
            ("Severity",        20),
            ("Est. Cost",       32),
        ]
        _thead(pdf, cols)
        for i, item in enumerate(line_items):
            _trow(pdf, [
                (item.get("part", ""),                   68),
                (item.get("repair_action", ""),          60),
                (item.get("severity_level", ""),         20),
                (_fmt_inr(item.get("part_cost", 0)),     32),
            ], shade=(i % 2 == 0))

        pdf.ln(3)

        # totals block
        totals = [
            ("Parts Subtotal",          cost_result.get("parts_total", 0), False),
            ("Labour (20%)",            cost_result.get("labor_total",  0), True),
            ("GRAND TOTAL",             cost_result.get("grand_total",  0), False),
        ]
        for label, amount, shade in totals:
            pdf.set_fill_color(*(C_LIGHT if shade else (235, 240, 248)))
            bold = "B" if "GRAND" in label else ""
            pdf.set_font("Helvetica", bold, 9)
            pdf.cell(120, 7, _clean(f"  {label}"), fill=True)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, _clean(f"  {_fmt_inr(amount)}"),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 4, _clean(cost_result.get("note", "")),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*C_DARK)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "  No cost data — no damage detected.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    _section(pdf, f"{sec}. Disclaimer")
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(90, 90, 90)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        usable_w, 4,
        _clean(
            "This report is generated by an AI system for informational purposes only. "
            "It does not constitute a professional insurance assessment or certified repair quote. "
            "Consult a certified mechanic or IRDAI-authorised insurance surveyor for official claims. "
            "Part prices are placeholder estimates and must be validated against current market rates."
        ),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_text_color(*C_DARK)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    pdf_bytes = bytes(pdf.output())
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(pdf_bytes)
    return pdf_bytes
