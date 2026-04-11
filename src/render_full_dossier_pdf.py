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
from collections import Counter
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("reportlab not installed — PDF rendering disabled")

from pdf_fonts import FONT_BOLD, FONT_NORMAL, register_cyrillic_fonts


_REGION_ORDER = ["RU", "EU", "US", "EAEU"]
_REGION_LABELS = {
    "RU": "РФ",
    "EU": "EU",
    "US": "US",
    "EAEU": "EAEU",
}
_REGION_DOC_KINDS = {
    "grls": "RU",
    "grls_card": "RU",
    "ru_instruction": "RU",
    "ru_quality_letter": "RU",
    "smpc": "EU",
    "epar": "EU",
    "pil": "EU",
    "assessment_report": "EU",
    "epi": "EU",
    "eaeu_document": "EAEU",
    "eaeu_registration": "EAEU",
    "label": "US",
    "approval_letter": "US",
    "us_fda": "US",
    "rems_document": "US",
    "complete_response_letter": "US",
    "anda_package": "US",
}


def _get_styles():
    font = register_cyrillic_fonts()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "DossierTitle",
            parent=styles["Heading1"],
            fontSize=21,
            spaceAfter=10,
            textColor=colors.HexColor("#13294b"),
            fontName=font,
        )
    )
    styles.add(
        ParagraphStyle(
            "QuestionHead",
            parent=styles["Heading1"],
            fontSize=17,
            spaceAfter=10,
            textColor=colors.HexColor("#13294b"),
            fontName=font,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHead",
            parent=styles["Heading2"],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=10,
            textColor=colors.HexColor("#1f4b7a"),
            fontName=font,
        )
    )
    styles.add(
        ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            spaceAfter=4,
            fontName=font,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodySmall",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            spaceAfter=3,
            textColor=colors.HexColor("#455a64"),
            fontName=font,
        )
    )
    styles.add(
        ParagraphStyle(
            "DossierBullet",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            leftIndent=10,
            firstLineIndent=0,
            spaceAfter=3,
            fontName=font,
        )
    )
    styles.add(
        ParagraphStyle(
            "DossierSource",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            leftIndent=10,
            spaceAfter=2,
            textColor=colors.HexColor("#5f6368"),
            fontName=font,
        )
    )
    return styles


def _esc(text: Any) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _ev_val(ev: Any) -> str:
    if ev is None:
        return "—"
    if isinstance(ev, dict):
        value = ev.get("value")
        return str(value) if value not in (None, "", []) else "—"
    if isinstance(ev, list):
        parts = [_ev_val(item) for item in ev]
        return ", ".join(part for part in parts if part != "—") or "—"
    if ev == "":
        return "—"
    return str(ev)


def _ev_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    values: List[str] = []
    for item in items:
        value = _ev_val(item)
        if value != "—":
            values.append(value)
    return values


def _unique(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalize_region(region: Any) -> str:
    raw = str(region or "").strip().upper()
    if raw in {"US", "USA", "UNITED STATES"}:
        return "US"
    if raw in {"EU", "EMA"}:
        return "EU"
    if raw in {"RU", "RUSSIA", "RUSSIAN FEDERATION"}:
        return "RU"
    if raw == "EAEU":
        return "EAEU"
    return raw


def _region_label(region: str) -> str:
    return _REGION_LABELS.get(region, region)


def _display_source_title(title: Any, source_url: Any = None, doc_kind: Any = None) -> str:
    raw_title = str(title or "").strip()
    if raw_title:
        return raw_title
    raw_url = str(source_url or "").strip()
    if raw_url:
        parsed = urlparse(raw_url)
        path = parsed.path.strip("/") or parsed.netloc or raw_url
        return f"{parsed.netloc}{('/' + path) if path and path != parsed.netloc else ''}".strip("/")
    raw_kind = str(doc_kind or "").strip()
    return raw_kind or "Untitled source"


def _make_evidence_maps(dossier: Dict[str, Any]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_doc_id: Dict[str, List[Dict[str, Any]]] = {}
    for ev in dossier.get("evidence_registry", []) or []:
        if not isinstance(ev, dict):
            continue
        evidence_id = str(ev.get("evidence_id") or "").strip()
        doc_id = str(ev.get("doc_id") or "").strip()
        if evidence_id:
            by_id[evidence_id] = ev
        if doc_id:
            by_doc_id.setdefault(doc_id, []).append(ev)
    return by_id, by_doc_id


def _titles_from_primary_docs(primary_docs: Any) -> List[str]:
    if not isinstance(primary_docs, list):
        return []
    titles = []
    for doc in primary_docs:
        if not isinstance(doc, dict):
            continue
        titles.append(
            _display_source_title(
                doc.get("title"),
                source_url=doc.get("source_url"),
                doc_kind=doc.get("doc_kind"),
            )
        )
    return _unique(titles)


def _titles_from_evidence_refs(evidence_by_id: Dict[str, Dict[str, Any]], evidence_refs: Any) -> List[str]:
    if not isinstance(evidence_refs, list):
        return []
    titles = []
    for ref in evidence_refs:
        evidence = evidence_by_id.get(str(ref))
        if not evidence:
            continue
        titles.append(
            _display_source_title(
                evidence.get("title"),
                source_url=evidence.get("source_url"),
                doc_kind=evidence.get("doc_kind"),
            )
        )
    return _unique(titles)


def _titles_from_doc_ids(evidence_by_doc_id: Dict[str, List[Dict[str, Any]]], doc_ids: Any) -> List[str]:
    if not isinstance(doc_ids, list):
        return []
    titles = []
    for doc_id in doc_ids:
        items = evidence_by_doc_id.get(str(doc_id), [])
        for evidence in items:
            titles.append(
                _display_source_title(
                    evidence.get("title"),
                    source_url=evidence.get("source_url"),
                    doc_kind=evidence.get("doc_kind"),
                )
            )
    return _unique(titles)


def _extend_unique(target: List[str], values: List[str]) -> None:
    seen = set(target)
    for value in values:
        if value and value not in seen:
            target.append(value)
            seen.add(value)


def _collect_front_page_sources(
    dossier: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {region: [] for region in _REGION_ORDER}

    for reg in dossier.get("registrations", []) or []:
        if not isinstance(reg, dict):
            continue
        region = _normalize_region(reg.get("region"))
        if region not in buckets:
            continue
        _extend_unique(buckets[region], _titles_from_primary_docs(reg.get("primary_docs")))
        _extend_unique(buckets[region], _titles_from_evidence_refs(evidence_by_id, reg.get("evidence_refs")))

    for ctx in dossier.get("product_contexts", []) or []:
        if not isinstance(ctx, dict):
            continue
        region = _normalize_region(ctx.get("region"))
        if region not in buckets:
            continue
        _extend_unique(buckets[region], _titles_from_primary_docs(ctx.get("primary_docs")))
        _extend_unique(buckets[region], _titles_from_evidence_refs(evidence_by_id, ctx.get("evidence_refs")))

    for evidence in evidence_by_id.values():
        region = _REGION_DOC_KINDS.get(str(evidence.get("doc_kind") or "").strip().lower())
        if not region or region not in buckets:
            continue
        _extend_unique(
            buckets[region],
            [
                _display_source_title(
                    evidence.get("title"),
                    source_url=evidence.get("source_url"),
                    doc_kind=evidence.get("doc_kind"),
                )
            ],
        )

    return {region: titles for region, titles in buckets.items() if titles}


def _collect_registration_sources(
    dossier: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    titles: List[str] = []
    for reg in dossier.get("registrations", []) or []:
        if not isinstance(reg, dict):
            continue
        _extend_unique(titles, _titles_from_primary_docs(reg.get("primary_docs")))
        _extend_unique(titles, _titles_from_evidence_refs(evidence_by_id, reg.get("evidence_refs")))
    return titles


def _collect_clinical_sources(
    dossier: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    titles: List[str] = []
    for study in dossier.get("clinical_studies", []) or []:
        if not isinstance(study, dict):
            continue
        _extend_unique(titles, _titles_from_evidence_refs(evidence_by_id, study.get("evidence_refs")))
    return titles


def _collect_patent_sources(
    dossier: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
    evidence_by_doc_id: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    titles: List[str] = []
    for family in dossier.get("patent_families", []) or []:
        if not isinstance(family, dict):
            continue
        _extend_unique(titles, _titles_from_evidence_refs(evidence_by_id, family.get("evidence_refs")))
        _extend_unique(titles, _titles_from_doc_ids(evidence_by_doc_id, family.get("key_docs")))
    return titles


def _collect_synthesis_sources(
    dossier: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
    evidence_by_doc_id: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    titles: List[str] = []
    for step in dossier.get("synthesis_steps", []) or []:
        if not isinstance(step, dict):
            continue
        _extend_unique(titles, _titles_from_evidence_refs(evidence_by_id, step.get("evidence_refs")))
        _extend_unique(titles, _titles_from_doc_ids(evidence_by_doc_id, step.get("source_patent_refs")))
    return titles


def _append_bullets(story: List[Any], lines: List[str], styles, style_name: str = "DossierBullet", max_items: Optional[int] = None):
    subset = lines[:max_items] if max_items else lines
    for line in subset:
        story.append(Paragraph(f"- {_esc(line)}", styles[style_name]))


def _append_sources(story: List[Any], styles, titles: List[str], max_items: int = 14):
    if not titles:
        return
    story.append(Spacer(1, 6))
    story.append(Paragraph("Источники", styles["SectionHead"]))
    _append_bullets(story, _unique(titles), styles, style_name="DossierSource", max_items=max_items)


def _format_reg_line(reg: Dict[str, Any]) -> str:
    region = _region_label(_normalize_region(reg.get("region")) or "—")
    parts = []
    status = _ev_val(reg.get("status"))
    mah = _ev_val(reg.get("mah"))
    identifiers = ", ".join(_ev_list(reg.get("identifiers"))[:3])
    forms = ", ".join(_ev_list(reg.get("forms_strengths"))[:3])
    if status != "—":
        parts.append(f"статус: {status}")
    if mah != "—":
        parts.append(f"держатель: {mah}")
    if identifiers:
        parts.append(f"ID: {identifiers}")
    if forms:
        parts.append(f"формы: {forms}")
    if not parts:
        parts.append("структурированные детали не выделены")
    return f"[{region}] " + "; ".join(parts)


def _is_truthy_evidence_value(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("value"))
    return bool(value)


def _clinical_summary_lines(studies: List[Dict[str, Any]]) -> List[str]:
    if not studies:
        return ["В структурированном досье нет клинических карточек."]
    status_counter = Counter()
    phase_values: List[str] = []
    ru_presence = 0
    conclusions = 0
    for study in studies:
        status = _ev_val(study.get("status"))
        if status != "—":
            status_counter[status] += 1
        phase = _ev_val(study.get("phase"))
        if phase != "—":
            phase_values.append(phase)
        if _is_truthy_evidence_value(study.get("has_ru_presence")):
            ru_presence += 1
        if _ev_val(study.get("conclusion")) != "—":
            conclusions += 1

    lines = [f"В досье собрано {len(studies)} клинических карточек."]
    if status_counter:
        top_statuses = ", ".join(f"{status}: {count}" for status, count in status_counter.most_common(4))
        lines.append(f"По статусам: {top_statuses}.")
    phases = _unique(phase_values)
    if phases:
        lines.append(f"Зафиксированные фазы: {', '.join(phases[:5])}.")
    if conclusions:
        lines.append(f"У {conclusions} исследований есть выделенный вывод или conclusion.")
    if ru_presence:
        lines.append(f"Признак присутствия в РФ выделен у {ru_presence} исследований.")
    return lines


def _format_study_line(study: Dict[str, Any]) -> str:
    study_id = _ev_val(study.get("study_id"))
    title = _ev_val(study.get("title"))
    status = _ev_val(study.get("status"))
    phase = _ev_val(study.get("phase"))
    countries = ", ".join(_ev_list(study.get("countries"))[:4])
    conclusion = _ev_val(study.get("conclusion"))

    head = study_id if study_id != "—" else (title if title != "—" else "Исследование без ID")
    parts = []
    if title != "—" and title != head:
        parts.append(title)
    if status != "—":
        parts.append(f"статус: {status}")
    if phase != "—":
        parts.append(f"фаза: {phase}")
    if countries:
        parts.append(f"география: {countries}")
    if conclusion != "—":
        parts.append(f"вывод: {conclusion[:180]}")
    return head + ("; " + "; ".join(parts) if parts else "")


def _patent_summary_lines(families: List[Dict[str, Any]]) -> List[str]:
    if not families:
        return ["Патентные семьи в структурированном досье не выделены."]
    block_counter = Counter()
    legal_counter = Counter()
    coverage: List[str] = []
    expiry_points: List[str] = []
    for family in families:
        what_blocks = _ev_val(family.get("what_blocks"))
        legal = _ev_val(family.get("legal_status_snapshot"))
        if what_blocks != "—":
            block_counter[what_blocks] += 1
        if legal != "—":
            legal_counter[legal] += 1
        coverage.extend(_ev_list(family.get("country_coverage")))
        expiry_points.extend(_ev_list(family.get("expiry_by_country")))

    lines = [f"В досье собрано {len(families)} патентных семей."]
    if block_counter:
        lines.append(
            "Что покрывают семьи: " + ", ".join(
                f"{name}: {count}" for name, count in block_counter.most_common(5)
            ) + "."
        )
    if legal_counter:
        lines.append(
            "По legal status: " + ", ".join(
                f"{name}: {count}" for name, count in legal_counter.most_common(5)
            ) + "."
        )
    coverage_values = _unique(coverage)
    if coverage_values:
        lines.append(f"Географическое покрытие в корпусе: {', '.join(coverage_values[:10])}.")
    if expiry_points:
        lines.append(f"Выделенные сроки / expiry: {', '.join(_unique(expiry_points)[:6])}.")
    return lines


def _format_patent_line(family: Dict[str, Any]) -> str:
    rep_pub = _ev_val(family.get("representative_pub"))
    parts = []
    priority = _ev_val(family.get("priority_date"))
    blocks = _ev_val(family.get("what_blocks"))
    technical = _ev_val(family.get("technical_focus"))
    legal = _ev_val(family.get("legal_status_snapshot"))
    coverage = ", ".join(_ev_list(family.get("country_coverage"))[:6])
    expiry = ", ".join(_ev_list(family.get("expiry_by_country"))[:4])
    if blocks != "—":
        parts.append(f"на что: {blocks}")
    if technical != "—":
        parts.append(f"фокус: {technical}")
    if coverage:
        parts.append(f"покрытие: {coverage}")
    if legal != "—":
        parts.append(f"статус: {legal}")
    if priority != "—":
        parts.append(f"priority: {priority}")
    if expiry:
        parts.append(f"до: {expiry}")
    summary = _ev_val(family.get("summary"))
    if summary != "—":
        parts.append(f"суть: {summary[:140]}")
    return rep_pub + ("; " + "; ".join(parts) if parts else "")


def _synthesis_summary_lines(steps: List[Dict[str, Any]]) -> List[str]:
    if not steps:
        return ["Синтез-путь в структурированном досье не выделен."]
    kind_counter = Counter()
    for step in steps:
        kind = str(step.get("kind") or "unknown").strip()
        kind_counter[kind] += 1
    lines = [f"В досье выделено {len(steps)} шагов синтеза / производственного процесса."]
    if kind_counter:
        lines.append(
            "По типам шагов: " + ", ".join(
                f"{name}: {count}" for name, count in kind_counter.most_common(5)
            ) + "."
        )
    return lines


def _format_step_line(step: Dict[str, Any]) -> str:
    step_number = step.get("step_number", "?")
    kind = str(step.get("kind") or "unknown").strip()
    description = _ev_val(step.get("description"))
    reagents = ", ".join(_ev_list(step.get("reagents"))[:4])
    intermediates = ", ".join(_ev_list(step.get("intermediates"))[:4])
    parts = [description[:220] if description != "—" else "описание не выделено"]
    if reagents:
        parts.append(f"реагенты: {reagents}")
    if intermediates:
        parts.append(f"интермедиаты: {intermediates}")
    return f"Шаг {step_number} ({kind}): " + "; ".join(parts)


def _render_cover_page(story: List[Any], styles, dossier: Dict[str, Any], source_buckets: Dict[str, List[str]]):
    passport = dossier.get("passport", {}) or {}
    inn = passport.get("inn", "Unknown INN")
    story.append(Paragraph(_esc(inn), styles["DossierTitle"]))
    story.append(
        Paragraph(
            f"Generated: {_esc(dossier.get('generated_at', 'N/A'))} | Report ID: {_esc(dossier.get('report_id', 'N/A'))}",
            styles["BodySmall"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d9dee3")))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Прикрепленные источники", styles["SectionHead"]))

    if not source_buckets:
        story.append(Paragraph("Регуляторные источники по географиям не выделены.", styles["Body"]))
        return

    for region in _REGION_ORDER:
        titles = source_buckets.get(region)
        if not titles:
            continue
        story.append(Paragraph(f"{_region_label(region)}:", styles["Body"]))
        _append_bullets(story, titles, styles, style_name="DossierSource", max_items=16)
        story.append(Spacer(1, 4))


def _render_registrations_page(story: List[Any], styles, dossier: Dict[str, Any], source_titles: List[str]):
    registrations = dossier.get("registrations", []) or []
    story.append(PageBreak())
    story.append(Paragraph("Где зарегистрирован препарат", styles["QuestionHead"]))
    story.append(Paragraph("Текущие регистрации", styles["SectionHead"]))

    if registrations:
        regions = _unique(
            [_region_label(_normalize_region(reg.get("region"))) for reg in registrations if reg.get("region")]
        )
        if regions:
            story.append(Paragraph(f"Подтвержденные регионы: {', '.join(regions)}.", styles["Body"]))
        _append_bullets(
            story,
            [_format_reg_line(reg) for reg in registrations],
            styles,
            max_items=12,
        )
    else:
        story.append(Paragraph("Подтвержденные регистрационные записи в structured dossier не заполнены.", styles["Body"]))

    _append_sources(story, styles, source_titles)


def _render_clinical_page(story: List[Any], styles, dossier: Dict[str, Any], source_titles: List[str]):
    studies = [item for item in (dossier.get("clinical_studies", []) or []) if isinstance(item, dict)]
    story.append(PageBreak())
    story.append(Paragraph("Текущие клинические исследования", styles["QuestionHead"]))
    story.append(Paragraph("Выводы", styles["SectionHead"]))
    _append_bullets(story, _clinical_summary_lines(studies), styles, max_items=6)
    story.append(Paragraph("Карточки исследований", styles["SectionHead"]))
    if studies:
        _append_bullets(story, [_format_study_line(study) for study in studies], styles, max_items=10)
    else:
        story.append(Paragraph("Структурированных study cards в корпусе нет.", styles["Body"]))
    _append_sources(story, styles, source_titles)


def _render_patents_page(story: List[Any], styles, dossier: Dict[str, Any], source_titles: List[str]):
    families = [item for item in (dossier.get("patent_families", []) or []) if isinstance(item, dict)]
    story.append(PageBreak())
    story.append(Paragraph("Патенты", styles["QuestionHead"]))
    story.append(Paragraph("Сводка по патентному покрытию", styles["SectionHead"]))
    _append_bullets(story, _patent_summary_lines(families), styles, max_items=6)
    story.append(Paragraph("Семьи", styles["SectionHead"]))
    if families:
        _append_bullets(story, [_format_patent_line(family) for family in families], styles, max_items=12)
    else:
        story.append(Paragraph("Патентные семьи в structured dossier отсутствуют.", styles["Body"]))
    _append_sources(story, styles, source_titles)


def _render_synthesis_page(story: List[Any], styles, dossier: Dict[str, Any], source_titles: List[str]):
    steps = [item for item in (dossier.get("synthesis_steps", []) or []) if isinstance(item, dict)]
    story.append(PageBreak())
    story.append(Paragraph("Синтез-пути", styles["QuestionHead"]))
    story.append(Paragraph("Сводка", styles["SectionHead"]))
    _append_bullets(story, _synthesis_summary_lines(steps), styles, max_items=4)
    story.append(Paragraph("Шаги", styles["SectionHead"]))
    if steps:
        _append_bullets(story, [_format_step_line(step) for step in steps], styles, max_items=10)
    else:
        story.append(Paragraph("Синтезные шаги в structured dossier не выделены.", styles["Body"]))
    _append_sources(story, styles, source_titles)


def _render_unknowns_page(story: List[Any], styles, dossier: Dict[str, Any]):
    unknowns = [item for item in (dossier.get("unknowns", []) or []) if isinstance(item, dict)]
    if not unknowns:
        return
    story.append(PageBreak())
    story.append(Paragraph("Unknowns", styles["QuestionHead"]))
    story.append(Paragraph("Пробелы и то, что осталось незакрытым", styles["SectionHead"]))
    lines = []
    for item in unknowns[:40]:
        field_path = item.get("field_path", "?")
        reason_code = item.get("reason_code", "")
        message = item.get("message", "")
        if reason_code:
            lines.append(f"{field_path} [{reason_code}] — {message}")
        else:
            lines.append(f"{field_path} — {message}")
    _append_bullets(story, lines, styles, max_items=40)


def render_full_dossier_pdf(dossier: Dict[str, Any], output_path: str) -> str:
    """
    Render dossier JSON into a question-style PDF.

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
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    story: List[Any] = []

    evidence_by_id, evidence_by_doc_id = _make_evidence_maps(dossier)
    source_buckets = _collect_front_page_sources(dossier, evidence_by_id)
    registration_sources = _collect_registration_sources(dossier, evidence_by_id)
    clinical_sources = _collect_clinical_sources(dossier, evidence_by_id)
    patent_sources = _collect_patent_sources(dossier, evidence_by_id, evidence_by_doc_id)
    synthesis_sources = _collect_synthesis_sources(dossier, evidence_by_id, evidence_by_doc_id)

    _render_cover_page(story, styles, dossier, source_buckets)
    _render_registrations_page(story, styles, dossier, registration_sources)
    _render_clinical_page(story, styles, dossier, clinical_sources)
    _render_patents_page(story, styles, dossier, patent_sources)
    _render_synthesis_page(story, styles, dossier, synthesis_sources)
    _render_unknowns_page(story, styles, dossier)

    doc.build(story)
    return output_path


def _make_kv_table(rows: List[List[str]]) -> Table:
    style = TableStyle(
        [
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#455a64")),
            ("FONTNAME", (0, 0), (0, -1), FONT_BOLD),
            ("FONTNAME", (1, 0), (1, -1), FONT_NORMAL),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]
    )
    table = Table(rows, colWidths=[4 * cm, 12 * cm])
    table.setStyle(style)
    return table


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
