"""
Full Dossier PDF Renderer — Sprint 22
========================================
Renders dossier_v3.json into full_dossier.pdf.
PDF is NOT source of truth — it's a presentation layer from structured JSON.

Uses reportlab for PDF generation (available in RAG worker environment).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("reportlab not installed — PDF rendering disabled")


# ── Styles ──────────────────────────────────────────────────────────────────

def _get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "DossierTitle", parent=styles["Heading1"], fontSize=18,
        spaceAfter=12, textColor=colors.HexColor("#1a237e"),
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"], fontSize=14,
        spaceAfter=8, spaceBefore=12, textColor=colors.HexColor("#283593"),
    ))
    styles.add(ParagraphStyle(
        "SubHead", parent=styles["Heading3"], fontSize=11,
        spaceAfter=4, spaceBefore=8,
    ))
    styles.add(ParagraphStyle(
        "BodySmall", parent=styles["Normal"], fontSize=9,
        spaceAfter=4, leading=12,
    ))
    styles.add(ParagraphStyle(
        "Evidence", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#616161"), leftIndent=12,
    ))
    return styles


def _ev_val(ev: Any) -> str:
    """Extract display value from EvidencedValue or plain value."""
    if ev is None:
        return "—"
    if isinstance(ev, dict):
        v = ev.get("value")
        return str(v) if v is not None else "—"
    if isinstance(ev, list):
        parts = []
        for item in ev:
            parts.append(_ev_val(item))
        return ", ".join(p for p in parts if p != "—")
    return str(ev)


# ── Renderer ────────────────────────────────────────────────────────────────

def render_full_dossier_pdf(
    dossier: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Render dossier JSON into a structured PDF.

    Args:
        dossier: Dossier v3 JSON dict.
        output_path: File path for output PDF.

    Returns:
        Path to generated PDF.
    """
    if not HAS_REPORTLAB:
        raise ImportError("reportlab is required for PDF rendering")

    styles = _get_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = []
    passport = dossier.get("passport", {})

    # Title
    inn = passport.get("inn", "Unknown INN")
    story.append(Paragraph(f"Dossier Report: {_esc(inn)}", styles["DossierTitle"]))
    story.append(Paragraph(
        f"Generated: {dossier.get('generated_at', 'N/A')} | "
        f"Schema: {dossier.get('schema_version', '3.0')}",
        styles["BodySmall"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 8))

    # ── Passport ────────────────────────────────────────────────────
    story.append(Paragraph("Passport", styles["SectionHead"]))
    passport_rows = [
        ["INN", _esc(inn)],
        ["Trade Names", _esc(_ev_val(passport.get("trade_names")))],
        ["Drug Class", _esc(_ev_val(passport.get("drug_class")))],
        ["MoA", _esc(_ev_val(passport.get("mechanism_of_action")))],
        ["Chemical Formula", _esc(_ev_val(passport.get("chemical_formula")))],
        ["SMILES", _esc(_ev_val(passport.get("smiles")))[:80]],
        ["Molecular Weight", _esc(_ev_val(passport.get("molecular_weight")))],
        ["Route", _esc(_ev_val(passport.get("route_of_administration")))],
        ["Dosage Forms", _esc(_ev_val(passport.get("dosage_forms")))],
        ["MAH Holders", _esc(_ev_val(passport.get("mah_holders")))],
        ["FDA Approval", _esc(_ev_val(passport.get("fda_approval_date")))],
    ]
    story.append(_make_kv_table(passport_rows))
    story.append(Spacer(1, 8))

    # ── Registrations ───────────────────────────────────────────────
    registrations = dossier.get("registrations", [])
    if registrations:
        story.append(Paragraph("Registrations", styles["SectionHead"]))
        for reg in registrations:
            region = reg.get("region", "?")
            story.append(Paragraph(f"Region: {_esc(region)}", styles["SubHead"]))
            rows = [
                ["Status", _esc(_ev_val(reg.get("status")))],
                ["MAH", _esc(_ev_val(reg.get("mah")))],
                ["Identifiers", _esc(_ev_val(reg.get("identifiers")))],
                ["Forms/Strengths", _esc(_ev_val(reg.get("forms_strengths")))],
            ]
            story.append(_make_kv_table(rows))
            story.append(Spacer(1, 4))

    # ── Clinical Studies ────────────────────────────────────────────
    studies = dossier.get("clinical_studies", [])
    if studies:
        story.append(PageBreak())
        story.append(Paragraph("Clinical Studies", styles["SectionHead"]))
        for i, study in enumerate(studies[:20], 1):
            title = _ev_val(study.get("title"))
            story.append(Paragraph(f"{i}. {_esc(title[:120])}", styles["SubHead"]))
            rows = [
                ["Phase", _esc(_ev_val(study.get("phase")))],
                ["N Enrolled", _esc(_ev_val(study.get("n_enrolled")))],
                ["Status", _esc(_ev_val(study.get("status")))],
                ["Comparator", _esc(_ev_val(study.get("comparator")))],
                ["Conclusion", _esc(_ev_val(study.get("conclusion")))[:200]],
            ]
            story.append(_make_kv_table(rows))
            story.append(Spacer(1, 4))

    # ── Patent Families ─────────────────────────────────────────────
    families = dossier.get("patent_families", [])
    if families:
        story.append(PageBreak())
        story.append(Paragraph("Patent Families", styles["SectionHead"]))
        for fam in families[:20]:
            pub = _ev_val(fam.get("representative_pub"))
            story.append(Paragraph(f"Family: {_esc(pub)}", styles["SubHead"]))
            rows = [
                ["Priority Date", _esc(_ev_val(fam.get("priority_date")))],
                ["Assignees", _esc(_ev_val(fam.get("assignees")))],
                ["Blocks", _esc(_ev_val(fam.get("what_blocks")))],
                ["Technical Focus", _esc(_ev_val(fam.get("technical_focus")))],
                ["Legal Status", _esc(_ev_val(fam.get("legal_status_snapshot")))],
                ["Country Coverage", _esc(_ev_val(fam.get("country_coverage")))],
            ]
            story.append(_make_kv_table(rows))
            story.append(Spacer(1, 4))

    # ── Synthesis Steps ─────────────────────────────────────────────
    steps = dossier.get("synthesis_steps", [])
    if steps:
        story.append(Paragraph("Synthesis Steps", styles["SectionHead"]))
        for step in steps[:10]:
            n = step.get("step_number", "?")
            desc = _ev_val(step.get("description"))
            story.append(Paragraph(f"Step {n}: {_esc(desc[:200])}", styles["BodySmall"]))

    # ── Unknowns ────────────────────────────────────────────────────
    unknowns = dossier.get("unknowns", [])
    if unknowns:
        story.append(PageBreak())
        story.append(Paragraph("Unknowns / Gaps", styles["SectionHead"]))
        for u in unknowns[:30]:
            fp = u.get("field_path", "?")
            msg = u.get("message", "")
            rc = u.get("reason_code", "")
            story.append(Paragraph(
                f"<b>{_esc(fp)}</b> [{_esc(rc)}]: {_esc(msg)}",
                styles["BodySmall"],
            ))

    # ── Quality ─────────────────────────────────────────────────────
    quality_v2 = dossier.get("dossier_quality_v2")
    if quality_v2:
        story.append(Paragraph("Data Quality", styles["SectionHead"]))
        coverage = quality_v2.get("coverage", {})
        for k, v in coverage.items():
            story.append(Paragraph(f"{_esc(k)}: {v:.0%}", styles["BodySmall"]))
        readiness = quality_v2.get("decision_readiness", {})
        for k, v in readiness.items():
            story.append(Paragraph(f"{_esc(k)}: {_esc(str(v))}", styles["BodySmall"]))

    # Build PDF
    doc.build(story)
    return output_path


# ── Utilities ───────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """Escape for reportlab Paragraph XML."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _make_kv_table(rows: List[List[str]]) -> Table:
    """Create a simple key-value table."""
    style = TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#455a64")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ])
    # Wrap long values in Paragraph
    formatted = []
    for key, val in rows:
        val_display = val if len(val) < 80 else val[:80] + "..."
        formatted.append([key, val_display])
    t = Table(formatted, colWidths=[4 * cm, 12 * cm])
    t.setStyle(style)
    return t


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render full dossier PDF")
    parser.add_argument("--dossier", required=True, help="Path to dossier_v3.json")
    parser.add_argument("--output", required=True, help="Output PDF path")
    args = parser.parse_args()

    with open(args.dossier, "r", encoding="utf-8") as f:
        dossier = json.load(f)

    path = render_full_dossier_pdf(dossier, args.output)
    print(f"PDF written to {path}")
