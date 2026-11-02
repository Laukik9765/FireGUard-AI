"""
core/report_generator.py
─────────────────────────
FireGuard AI — PDF Report Generator

Produces a professionally formatted, human-readable PDF analysis
report from a JSON analysis result dict (returned by ImageProcessor
or VideoFileProcessor).

Dependencies: fpdf2 (pip install fpdf2)
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF, XPos, YPos


# ── Colour palette (RGB tuples) ────────────────────────────────────────────────
_C_BG        = (10,  15,  30)   # dark navy background
_C_PANEL     = (18,  26,  48)   # panel
_C_RED       = (239, 35,  60)   # accent red / fire
_C_AMBER     = (255, 183, 3)    # amber / warning
_C_GREEN     = (0,   230, 118)  # green / safe
_C_CYAN      = (0,   200, 255)  # cyan / info
_C_WHITE     = (255, 255, 255)
_C_GREY      = (140, 155, 185)
_C_DARK_LINE = (40,  55,  85)


def _hazard_color(level: str) -> tuple[int, int, int]:
    lvl = (level or "").upper()
    if lvl in ("CRITICAL", "EXTREME", "EMERGENCY"):
        return _C_RED
    if lvl in ("HIGH", "SEVERE"):
        return (255, 80, 40)
    if lvl in ("MODERATE", "MODERATE"):
        return _C_AMBER
    if lvl == "LOW":
        return _C_CYAN
    return _C_GREEN


def _smoke_color(hazard: str) -> tuple[int, int, int]:
    h = (hazard or "").lower()
    if h == "heavy":    return _C_RED
    if h == "moderate": return _C_AMBER
    if h == "low":      return _C_CYAN
    return _C_GREEN


class FireGuardPDF(FPDF):
    """Custom FPDF subclass with FireGuard styling."""

    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._title = title
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    # ── Header / Footer ────────────────────────────────────────────────────────

    def header(self):
        # Solid dark header bar
        self.set_fill_color(*_C_BG)
        self.rect(0, 0, 210, 22, style="F")

        # Brand wordmark
        self.set_y(6)
        self.set_x(18)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*_C_RED)
        self.cell(0, 8, "FIREGUARD AI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Sub-title on same line (manual position)
        self.set_y(6)
        self.set_x(72)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_C_GREY)
        self.cell(0, 8, "Fire & Smoke Situational Awareness - Analysis Report")

        # Right-hand timestamp
        now = datetime.now().strftime("%d %b %Y  %H:%M:%S")
        self.set_y(6)
        self.set_x(148)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_C_GREY)
        self.cell(44, 8, now, align="R")

        # Separator line
        self.set_draw_color(*_C_RED)
        self.set_line_width(0.5)
        self.line(0, 22, 210, 22)
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*_C_DARK_LINE)
        self.set_line_width(0.3)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_C_GREY)
        self.cell(0, 6, f"FireGuard AI - Confidential Incident Report - Page {self.page_no()}", align="C")

    # ── Helper primitives ──────────────────────────────────────────────────────

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_C_RED)
        self.cell(0, 7, text.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*_C_DARK_LINE)
        self.set_line_width(0.3)
        self.line(self.get_x(), self.get_y(), 192, self.get_y())
        self.ln(3)

    def kv_row(self, label: str, value: str,
               val_color: tuple[int, int, int] = _C_WHITE,
               bold_val: bool = False):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_C_GREY)
        self.cell(54, 6, label, new_x=XPos.RIGHT, new_y=YPos.LAST)
        self.set_font("Helvetica", "B" if bold_val else "", 9)
        self.set_text_color(*val_color)
        self.cell(0, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def stat_pill(self, x: float, y: float, label: str, value: str,
                  color: tuple[int, int, int], w: float = 38, h: float = 16):
        """Draw a rounded-rect stat pill at (x, y)."""
        self.set_fill_color(*_C_PANEL)
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        self.rect(x, y, w, h, style="FD", round_corners=True, corner_radius=3)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_C_GREY)
        self.set_xy(x + 2, y + 2)
        self.cell(w - 4, 5, label)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.set_xy(x + 2, y + 7)
        self.cell(w - 4, 6, value)


def generate_pdf_report(report: dict[str, Any]) -> bytes:
    """
    Convert a FireGuard analysis report dict into a PDF byte-string.

    Parameters
    ----------
    report : dict
        The JSON analysis report produced by ImageProcessor or
        VideoFileProcessor.

    Returns
    -------
    bytes
        Raw PDF bytes, ready to serve or write to disk.
    """
    source        = report.get("source", "Unknown File")
    duration      = report.get("duration_formatted", "N/A")
    proc_time     = report.get("processing_time", 0)
    hazard_level  = report.get("hazard_level", "NORMAL")
    fire_detected = bool(report.get("fire", {}).get("detected", False))
    fire_count    = report.get("fire", {}).get("count", 0)
    smoke         = report.get("smoke", {})
    people        = report.get("people", {})
    livestock     = report.get("livestock", {})
    timeline      = report.get("timeline", [])
    boxes         = report.get("boxes", [])

    pdf = FireGuardPDF(title=f"FireGuard AI Report - {source}")
    pdf.add_page()

    # ── Title block ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_C_WHITE)
    pdf.cell(0, 10, "Incident Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_C_GREY)
    pdf.cell(0, 6, f"Source: {source}  |  Duration: {duration}  |  Processed in {proc_time} s",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Overall Hazard status pill (wide) ────────────────────────────────────
    hcol = _hazard_color(hazard_level)
    pdf.set_fill_color(*_C_PANEL)
    pdf.set_draw_color(*hcol)
    pdf.set_line_width(0.8)
    pdf.rect(18, pdf.get_y(), 174, 20, style="FD", round_corners=True, corner_radius=4)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_C_GREY)
    pdf.set_xy(24, pdf.get_y() + 3)
    pdf.cell(50, 5, "OVERALL HAZARD LEVEL")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*hcol)
    pdf.set_xy(24, pdf.get_y() + 4)
    pdf.cell(0, 8, hazard_level)
    pdf.ln(26)

    # ── 4 stat pills row ────────────────────────────────────────────────────
    py = pdf.get_y()
    fire_col   = _C_RED   if fire_detected   else _C_GREEN
    fire_val   = f"{'Detected' if fire_detected else 'None'}"
    smoke_col  = _smoke_color(smoke.get("hazard", "None"))
    people_col = _C_CYAN  if (people.get("count", 0) > 0) else _C_GREY

    pdf.stat_pill(18,  py, "FIRE STATUS",    fire_val,                          fire_col)
    pdf.stat_pill(60,  py, "SMOKE LEVEL",    smoke.get("hazard", "None"),       smoke_col)
    pdf.stat_pill(102, py, "PEOPLE DETECTED", str(people.get("count", 0)),      people_col)
    pdf.stat_pill(144, py, "SMOKE SCORE",    f"{smoke.get('smoke_score', 0.0):.2f}", smoke_col)
    pdf.ln(24)

    # ── Fire Detection ───────────────────────────────────────────────────────
    pdf.section_title("1. Fire Detection")
    pdf.kv_row("Status",      "DETECTED" if fire_detected else "Not Detected",
               _C_RED if fire_detected else _C_GREEN, bold_val=True)
    pdf.kv_row("Flame Count", str(fire_count))
    pdf.ln(3)

    # ── Smoke Analysis ───────────────────────────────────────────────────────
    pdf.section_title("2. Smoke Analysis")
    pdf.kv_row("Hazard Level",  smoke.get("hazard",       "None"),   _smoke_color(smoke.get("hazard", "None")), bold_val=True)
    pdf.kv_row("Smoke Score",   f"{smoke.get('smoke_score', 0.0):.3f}")
    pdf.kv_row("Density",       f"{smoke.get('density', 0.0):.1f} %")
    pdf.kv_row("Coverage",      f"{smoke.get('coverage', 0.0):.1f} %")
    pdf.kv_row("Trend",         smoke.get("trend", "Stable"))
    pdf.ln(3)

    # ── Occupancy ────────────────────────────────────────────────────────────
    pdf.section_title("3. Occupancy & Life-Safety")
    pdf.kv_row("People Detected",   str(people.get("count", 0)),    _C_CYAN if people.get("count", 0) > 0 else _C_GREY)
    pdf.kv_row("Livestock / Animals", str(livestock.get("count", 0)))
    pdf.ln(3)

    # ── Detection Boxes ──────────────────────────────────────────────────────
    if boxes:
        pdf.section_title("4. Detection Bounding Boxes")
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_C_GREY)
        # table header
        pdf.set_fill_color(*_C_PANEL)
        pdf.rect(18, pdf.get_y(), 174, 7, style="F")
        pdf.cell(30, 7, "Type", border=0)
        pdf.cell(20, 7, "Conf", border=0)
        pdf.cell(20, 7, "Left", border=0)
        pdf.cell(20, 7, "Top", border=0)
        pdf.cell(20, 7, "Width", border=0)
        pdf.cell(20, 7, "Height", border=0)
        pdf.ln(7)

        pdf.set_font("Helvetica", "", 8)
        for i, b in enumerate(boxes[:30]):  # cap at 30 rows
            btype = b.get("query", "").upper()
            conf  = b.get("confidence", 0)
            left  = b.get("left",   "-")
            top   = b.get("top",    "-")
            bw    = b.get("width",  "-")
            bh    = b.get("height", "-")
            if i % 2 == 0:
                pdf.set_fill_color(*_C_PANEL)
                pdf.rect(18, pdf.get_y(), 174, 6, style="F")
            tcol = _C_RED if "fire" in btype.lower() else (_C_CYAN if "smoke" in btype.lower() else _C_GREEN)
            pdf.set_text_color(*tcol)
            pdf.cell(30, 6, btype)
            pdf.set_text_color(*_C_WHITE)
            pdf.cell(20, 6, f"{conf:.2f}")
            pdf.set_text_color(*_C_GREY)
            pdf.cell(20, 6, str(left))
            pdf.cell(20, 6, str(top))
            pdf.cell(20, 6, str(bw))
            pdf.cell(20, 6, str(bh))
            pdf.ln(6)
        if len(boxes) > 30:
            pdf.set_text_color(*_C_GREY)
            pdf.set_font("Helvetica", "I", 7)
            pdf.cell(0, 5, f"... and {len(boxes) - 30} more detections omitted for brevity.")
            pdf.ln(5)
        pdf.ln(3)

    # ── Frame Timeline ───────────────────────────────────────────────────────
    if timeline:
        pdf.section_title("5. Frame-by-Frame Timeline")
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_C_GREY)
        pdf.set_fill_color(*_C_PANEL)
        pdf.rect(18, pdf.get_y(), 174, 7, style="F")
        pdf.cell(20, 7, "Time")
        pdf.cell(28, 7, "Hazard")
        pdf.cell(28, 7, "Fire State")
        pdf.cell(28, 7, "Smoke State")
        pdf.cell(22, 7, "People")
        pdf.cell(24, 7, "Smoke Density")
        pdf.ln(7)

        pdf.set_font("Helvetica", "", 7.5)
        for i, pt in enumerate(timeline[:40]):
            hazard    = pt.get("hazard", "-")
            fire_st   = pt.get("fire", "-")
            smoke_st  = pt.get("smoke", "-")
            ppl       = pt.get("people", 0)
            density   = pt.get("smoke_density", 0)
            ts        = pt.get("time", "-")
            if i % 2 == 0:
                pdf.set_fill_color(*_C_PANEL)
                pdf.rect(18, pdf.get_y(), 174, 6, style="F")
            hcol_pt = _hazard_color(hazard)
            pdf.set_text_color(*_C_GREY); pdf.cell(20, 6, ts)
            pdf.set_text_color(*hcol_pt); pdf.cell(28, 6, hazard)
            pdf.set_text_color(*(_C_RED if "fire" in fire_st.lower() or "confirm" in fire_st.lower() or "emergency" in fire_st.lower() else _C_GREY))
            pdf.cell(28, 6, fire_st)
            pdf.set_text_color(*(_smoke_color(smoke_st)))
            pdf.cell(28, 6, smoke_st)
            pdf.set_text_color(*_C_GREY); pdf.cell(22, 6, str(ppl))
            pdf.set_text_color(*(_C_RED if density > 60 else _C_AMBER if density > 20 else _C_GREY))
            pdf.cell(24, 6, f"{density:.1f}%")
            pdf.ln(6)
        if len(timeline) > 40:
            pdf.set_text_color(*_C_GREY)
            pdf.set_font("Helvetica", "I", 7)
            pdf.cell(0, 5, f"... {len(timeline) - 40} additional timeline frames not shown.")
            pdf.ln(5)
        pdf.ln(3)

    # ── Disclaimer ──────────────────────────────────────────────────────────
    pdf.section_title("Disclaimer & Methodology")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_C_GREY)
    pdf.multi_cell(
        0, 5,
        "This report was generated automatically by FireGuard AI using an ensemble of YOLO neural "
        "fire/smoke detection and OpenCV-based multi-gate validation (HSV colour gating, temporal "
        "variance, spatial variance, and edge-density suppression).  All detections should be "
        "verified by a qualified fire-safety professional before acting on them.  FireGuard AI is an "
        "assistance tool and does not replace mandatory safety inspections, fire-alarm systems, or "
        "emergency-service protocols.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    return bytes(pdf.output())

