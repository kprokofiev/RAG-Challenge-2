"""
Decision memo PDF renderer for exec decision reports.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    HAS_REPORTLAB = True
except ImportError:  # pragma: no cover
    HAS_REPORTLAB = False

try:
    from src.pdf_fonts import register_cyrillic_fonts
except ImportError:  # pragma: no cover
    from pdf_fonts import register_cyrillic_fonts  # type: ignore


def _esc(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _styles(require_cyrillic: bool = False):
    font = register_cyrillic_fonts(require_cyrillic=require_cyrillic)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("MemoTitle", parent=styles["Heading1"], fontName=font, fontSize=20, textColor=colors.HexColor("#17324d"), spaceAfter=10))
    styles.add(ParagraphStyle("MemoHead", parent=styles["Heading2"], fontName=font, fontSize=13, textColor=colors.HexColor("#264b73"), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle("MemoBody", parent=styles["Normal"], fontName=font, fontSize=9, leading=12, spaceAfter=4))
    styles.add(ParagraphStyle("MemoMeta", parent=styles["Normal"], fontName=font, fontSize=8, leading=10, textColor=colors.HexColor("#5f6b7a"), spaceAfter=3))
    styles.add(ParagraphStyle("MemoWarn", parent=styles["Normal"], fontName=font, fontSize=9, leading=12, textColor=colors.HexColor("#8d4a00"), spaceAfter=4))
    return styles


def _ensure_dict(report: Any) -> Dict[str, Any]:
    if hasattr(report, "model_dump"):
        return report.model_dump()
    return dict(report)


def _render_lines(story: List[Any], title: str, lines: List[str], styles, style_name: str = "MemoBody") -> None:
    if not lines:
        return
    story.append(Paragraph(_esc(title), styles["MemoHead"]))
    for line in lines:
        story.append(Paragraph(f"- {_esc(line)}", styles[style_name]))


def render_exec_decision_report(report: Any, output_path: str, mode: str = "customer") -> str:
    if not HAS_REPORTLAB:
        raise ImportError("reportlab is required for PDF rendering")

    mode = (mode or "customer").strip().lower()
    is_internal = mode == "internal"
    payload = _ensure_dict(report)
    styles = _styles(require_cyrillic=not is_internal)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    story: List[Any] = []

    topline = payload.get("topline", {})
    major_keys = [
        "asset_attractiveness",
        "rf_entry",
        "eaeu_entry",
        "generic_opportunity",
        "licensing_opportunity",
        "portfolio_opportunity",
    ]
    blockers = payload.get("decision_blockers", [])
    next_actions = payload.get("recommended_next_actions", [])
    sufficiency = payload.get("evidence_sufficiency", {})

    story.append(Paragraph(f"Executive Decision Memo: {_esc(payload.get('inn') or 'Unknown asset')}", styles["MemoTitle"]))
    if is_internal:
        story.append(Paragraph(f"Case ID: {_esc(payload.get('case_id') or '')} | Generated: {_esc(payload.get('generated_at') or '')}", styles["MemoMeta"]))
        story.append(Paragraph(f"Mode: {_esc(mode)} | Overall sufficiency: {_esc(sufficiency.get('overall_verdict', 'PARTIAL'))} | Confidence: {_esc(sufficiency.get('topline_confidence', 'LOW'))}", styles["MemoMeta"]))
    else:
        story.append(Paragraph(f"Generated: {_esc(payload.get('generated_at') or '')}", styles["MemoMeta"]))
        story.append(Paragraph(f"Evidence sufficiency: {_esc(sufficiency.get('overall_verdict', 'PARTIAL'))} | Confidence: {_esc(sufficiency.get('topline_confidence', 'LOW'))}", styles["MemoMeta"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d8dee8")))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Topline verdicts", styles["MemoHead"]))
    for key in major_keys:
        item = topline.get(key, {})
        if not item:
            continue
        story.append(
            Paragraph(
                f"<b>{_esc(key.replace('_', ' ').title())}</b>: {_esc(item.get('verdict', ''))} | "
                f"confidence={_esc(item.get('confidence', ''))} | sufficiency={_esc(item.get('sufficiency', ''))}<br/>"
                f"{_esc(item.get('short_answer', ''))}",
                styles["MemoBody"],
            )
        )

    _render_lines(story, "Top blockers", [item.get("title", "") for item in blockers[:5] if item.get("title")], styles, "MemoWarn")
    _render_lines(story, "Recommended next step", [item.get("action", "") for item in next_actions[:5] if item.get("action")], styles)

    story.append(PageBreak())

    for block in payload.get("decision_blocks", []):
        story.append(Paragraph(_esc(block.get("title", block.get("block_id", ""))), styles["MemoHead"]))
        story.append(
            Paragraph(
                f"Verdict: <b>{_esc(block.get('verdict', ''))}</b> | confidence={_esc(block.get('confidence', ''))} | "
                f"sufficiency={_esc(block.get('sufficiency', ''))}",
                styles["MemoBody"],
            )
        )
        story.append(Paragraph(_esc(block.get("short_answer", "")), styles["MemoBody"]))
        story.append(Paragraph(_esc(block.get("full_answer", "")), styles["MemoBody"]))
        _render_lines(story, "Why this verdict", [item.get("claim", "") for item in block.get("why_this_verdict", [])], styles)
        _render_lines(story, "Blockers", [item.get("title", "") for item in block.get("decision_blockers", [])], styles, "MemoWarn")
        _render_lines(story, "Caveats", block.get("caveats", []), styles, "MemoWarn")
        _render_lines(story, "Next actions", [item.get("action", "") for item in block.get("next_actions", [])], styles)
        _render_lines(story, "Evidence refs", block.get("top_evidence_refs", []), styles, "MemoMeta")
        _render_lines(story, "Unknowns", block.get("unknowns", []), styles, "MemoWarn")

        if is_internal:
            verification = block.get("verification", {})
            story.append(Paragraph(f"Verifier: {_esc(verification.get('overall_status', 'PASS'))}", styles["MemoMeta"]))
            for issue in verification.get("issues", [])[:6]:
                story.append(Paragraph(f"- {_esc(issue.get('severity', ''))}: {_esc(issue.get('message', ''))}", styles["MemoMeta"]))
            model_trace = block.get("model_trace", {})
            if model_trace:
                planner_trace = model_trace.get("planner_trace", {})
                answer_trace = model_trace.get("answer_trace", {})
                if planner_trace:
                    story.append(
                        Paragraph(
                            f"Planner model: {_esc(planner_trace.get('model_selected', ''))} | thinking={_esc(planner_trace.get('thinking_mode', ''))}",
                            styles["MemoMeta"],
                        )
                    )
                if answer_trace:
                    story.append(
                        Paragraph(
                            f"Answerer model: {_esc(answer_trace.get('model_selected', ''))} | thinking={_esc(answer_trace.get('thinking_mode', ''))}",
                            styles["MemoMeta"],
                        )
                    )
                if not planner_trace and not answer_trace:
                    story.append(Paragraph(f"Model: {_esc(model_trace.get('model_selected', ''))} | thinking={_esc(model_trace.get('thinking_mode', ''))}", styles["MemoMeta"]))
                budget_trace = answer_trace.get("budget_trace", {}) if answer_trace else model_trace.get("budget_trace", {})
                if budget_trace:
                    story.append(
                        Paragraph(
                            f"Budget date={_esc(budget_trace.get('budget_date_utc', ''))} | "
                            f"requested={_esc(budget_trace.get('model_requested', ''))} | selected={_esc(budget_trace.get('model_selected', ''))}",
                            styles["MemoMeta"],
                        )
                    )
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d8dee8")))
        story.append(Spacer(1, 8))

    appendix = payload.get("appendix", {})
    if appendix and is_internal:
        story.append(PageBreak())
        story.append(Paragraph("Appendix", styles["MemoHead"]))
        source_snapshot = appendix.get("source_snapshot", {})
        budget_snapshot = appendix.get("budget_snapshot", {})
        question_traces = appendix.get("question_traces", [])
        for label, value in (
            ("Source snapshot", source_snapshot),
            ("Question traces", question_traces),
        ):
            story.append(Paragraph(_esc(label), styles["MemoHead"]))
            story.append(Paragraph(_esc(value), styles["MemoMeta"]))
        story.append(Paragraph("Budget snapshot", styles["MemoHead"]))
        story.append(Paragraph(_esc(budget_snapshot), styles["MemoMeta"]))

    doc.build(story)
    return output_path
