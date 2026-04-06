"""
Exec Questions Report PDF Renderer — Sprint 22
==================================================
Renders exec Q&A results (answer_frames + final answers) into
exec_questions_report.pdf.

Source of truth: answer_frame.json, NOT this PDF.
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


def _get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"], fontSize=18,
        spaceAfter=12, textColor=colors.HexColor("#1a237e"),
    ))
    styles.add(ParagraphStyle(
        "QuestionHead", parent=styles["Heading2"], fontSize=13,
        spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#283593"),
    ))
    styles.add(ParagraphStyle(
        "SectionLabel", parent=styles["Heading3"], fontSize=10,
        spaceAfter=4, spaceBefore=6, textColor=colors.HexColor("#455a64"),
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9,
        spaceAfter=3, leading=12,
    ))
    styles.add(ParagraphStyle(
        "Confidence", parent=styles["Normal"], fontSize=10,
        spaceAfter=4, textColor=colors.HexColor("#2e7d32"),
    ))
    styles.add(ParagraphStyle(
        "Warning", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#e65100"), spaceAfter=3,
    ))
    return styles


def _esc(text: str) -> str:
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def render_exec_questions_report(
    exec_results: List[Dict[str, Any]],
    output_path: str,
    inn: str = "Unknown",
) -> str:
    """
    Render multiple exec Q&A results into a single PDF.

    Args:
        exec_results: List of dicts, each from exec_answer_runner.run_exec_pipeline().
        output_path: Output PDF path.
        inn: Drug INN for title.

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

    # Title page
    story.append(Paragraph(
        f"Executive Q&amp;A Report: {_esc(inn)}", styles["ReportTitle"]
    ))
    story.append(Paragraph(
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} | "
        f"Questions: {len(exec_results)}",
        styles["Body"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 12))

    # Table of contents
    story.append(Paragraph("Questions Covered:", styles["SectionLabel"]))
    for i, result in enumerate(exec_results, 1):
        af = result.get("answer_frame", {})
        q_text = af.get("question_text", result.get("question_id", "?"))
        conf = result.get("confidence", {}).get("overall", "?")
        story.append(Paragraph(
            f"{i}. {_esc(q_text)} — confidence: {_esc(conf)}",
            styles["Body"],
        ))
    story.append(PageBreak())

    # Each question
    for i, result in enumerate(exec_results, 1):
        af = result.get("answer_frame", {})
        q_text = af.get("question_text", result.get("question_id", "?"))

        story.append(Paragraph(f"Q{i}: {_esc(q_text)}", styles["QuestionHead"]))

        # Confidence
        conf = result.get("confidence", {})
        conf_label = conf.get("overall", "unknown")
        story.append(Paragraph(
            f"Confidence: <b>{_esc(conf_label.upper())}</b> "
            f"({conf.get('strong_claims', 0)} strong, "
            f"{conf.get('moderate_claims', 0)} moderate, "
            f"{conf.get('weak_claims', 0)} weak)",
            styles["Confidence"],
        ))

        # Claims as conclusion
        claims = result.get("claims", [])
        story.append(Paragraph("Key Findings:", styles["SectionLabel"]))
        for claim in claims:
            support = claim.get("support_level", "?")
            text = claim.get("text", "")
            marker = "✓" if support in ("strong", "moderate") else "?"
            story.append(Paragraph(
                f"{marker} [{_esc(support)}] {_esc(text[:200])}",
                styles["Body"],
            ))

        # Unknowns
        unknowns = result.get("unknowns", [])
        if unknowns:
            story.append(Paragraph("Uncertainties:", styles["SectionLabel"]))
            for u in unknowns[:5]:
                fp = u.get("field_path", "?")
                msg = u.get("message", u.get("reason_code", ""))
                story.append(Paragraph(
                    f"• {_esc(fp)}: {_esc(msg)}", styles["Warning"]
                ))

        # Scope warnings
        scope = result.get("scope", {})
        warnings = scope.get("warnings", [])
        if warnings:
            story.append(Paragraph("Scope Warnings:", styles["SectionLabel"]))
            for w in warnings:
                story.append(Paragraph(f"⚠ {_esc(w)}", styles["Warning"]))

        # Next actions
        af_data = result.get("answer_frame", {})
        next_actions = af_data.get("recommended_next_actions", [])
        if next_actions:
            story.append(Paragraph("Recommended Next Steps:", styles["SectionLabel"]))
            for act in next_actions:
                story.append(Paragraph(f"→ {_esc(act)}", styles["Body"]))

        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e0e0e0")))
        story.append(Spacer(1, 8))

    doc.build(story)
    return output_path


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render exec questions report PDF")
    parser.add_argument("--results", required=True, help="Path to JSON file with exec results array")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--inn", default="Unknown", help="Drug INN for title")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        results = [results]

    path = render_exec_questions_report(results, args.output, inn=args.inn)
    print(f"PDF written to {path}")
