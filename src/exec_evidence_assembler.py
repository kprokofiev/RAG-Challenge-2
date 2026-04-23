"""
Deterministic evidence packet assembly for question-first exec reasoning.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import re
from typing import Any, Dict, Iterable, List, Optional

try:
    from src.exec_prompt_builder import ExecQuestionPlan
    from src.exec_retrieval_escalation import normalize_retrieval_item
except ImportError:  # pragma: no cover
    from exec_prompt_builder import ExecQuestionPlan  # type: ignore
    from exec_retrieval_escalation import normalize_retrieval_item  # type: ignore


_DOC_KIND_LABELS = {
    "grls": "GRLS listing",
    "grls_card": "GRLS card",
    "ru_instruction": "RU instruction",
    "ru_registration_export": "RU registration export",
    "ru_esklp_snapshot": "ESKLP snapshot",
    "ru_procurement_snapshot": "RU procurement snapshot",
    "ru_official_act": "RU official act",
    "ru_commercial_summary": "RU commercial summary",
    "ru_formulary_summary": "RU formulary summary",
    "ru_policy_act": "RU policy act",
    "eaeu_registration": "EAEU registration record",
    "eaeu_document": "EAEU document",
    "ctgov": "ClinicalTrials.gov",
    "ctgov_results": "ClinicalTrials.gov results",
    "smpc": "SmPC",
    "label": "Label",
    "us_fda": "US FDA source",
    "approval_letter": "FDA approval letter",
    "eu_regulatory_summary": "EU regulatory summary",
    "epar": "EPAR",
    "assessment_report": "Assessment report",
    "patent_family_summary": "Patent family summary",
    "patent_legal_events": "Patent legal events",
    "patent_expiry_us": "US patent expiry",
    "formulary": "Formulary source",
    "pricing": "Pricing source",
    "payer_policy": "Payer policy",
}

_DOC_KIND_ALIASES = {
    "fda_drugs_at_fda": "us_fda",
    "drugs_at_fda": "us_fda",
    "openfda": "us_fda",
    "ema_epar": "eu_regulatory_summary",
    "epar": "eu_regulatory_summary",
    "assessment_report": "eu_regulatory_summary",
    "ema_smpc": "smpc",
    "eaeu_register": "eaeu_document",
    "eaeu_portal": "eaeu_document",
    "eaeu_registration": "eaeu_document",
    "orange_book": "patent_expiry_us",
    "epo_register": "patent_legal_events",
    "epo_legal": "patent_legal_events",
    "spc": "patent_legal_events",
}

_DOC_KIND_EXPANSIONS = {
    "us_fda": ["us_fda", "label", "approval_letter"],
    "eu_regulatory_summary": ["eu_regulatory_summary", "epar", "assessment_report", "smpc"],
    "eaeu_document": ["eaeu_document", "eaeu_registration"],
    "grls": ["grls", "grls_card", "ru_instruction"],
    "patent_legal_events": ["patent_legal_events"],
    "patent_expiry_us": ["patent_expiry_us"],
}


def normalize_exec_doc_kind(value: Any) -> str:
    doc_kind = str(value or "").strip().lower()
    return _DOC_KIND_ALIASES.get(doc_kind, doc_kind)


def normalize_exec_doc_kind_list(values: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values or []:
        doc_kind = normalize_exec_doc_kind(value)
        if not doc_kind or doc_kind in seen:
            continue
        seen.add(doc_kind)
        normalized.append(doc_kind)
    return normalized


def expand_exec_doc_kinds(values: Iterable[Any]) -> List[str]:
    expanded: List[str] = []
    seen = set()
    for doc_kind in normalize_exec_doc_kind_list(values):
        candidates = _DOC_KIND_EXPANSIONS.get(doc_kind, [doc_kind])
        for candidate in candidates:
            text = str(candidate or "").strip().lower()
            if not text or text in seen:
                continue
            seen.add(text)
            expanded.append(text)
    return expanded


def reconcile_exec_doc_kinds(
    base_allowed_doc_kinds: Iterable[Any],
    planned_doc_kinds: Iterable[Any],
) -> List[str]:
    base_allowed = normalize_exec_doc_kind_list(base_allowed_doc_kinds)
    planned = normalize_exec_doc_kind_list(planned_doc_kinds)
    if base_allowed and planned:
        intersection = [doc_kind for doc_kind in planned if doc_kind in set(base_allowed)]
        return intersection or list(base_allowed)
    return list(base_allowed or planned)


def _source_label(item: Dict[str, Any]) -> str:
    doc_kind = normalize_exec_doc_kind(item.get("doc_kind"))
    return _DOC_KIND_LABELS.get(doc_kind, doc_kind or "evidence")


def _iter_evidence_refs(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        refs = value.get("evidence_refs", [])
        if isinstance(refs, list):
            for ref in refs:
                if str(ref or "").strip():
                    yield str(ref)
        for nested in value.values():
            yield from _iter_evidence_refs(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_evidence_refs(item)


def _region_from_record(record: Dict[str, Any]) -> str:
    return str(
        record.get("region")
        or record.get("jurisdiction")
        or record.get("country")
        or "GLOBAL"
    ).strip().upper() or "GLOBAL"


def _compact_value(value: Any) -> Any:
    if isinstance(value, dict):
        keep = {}
        for key in (
            "region",
            "verdict",
            "context_id",
            "status",
            "summary",
            "title",
            "study_id",
            "phase",
            "conclusion",
            "efficacy_keypoints",
            "n_enrolled",
            "category",
            "mah",
            "identifiers",
            "forms_strengths",
            "legal_status_snapshot",
            "expiry_by_country",
            "description",
            "evidence_refs",
        ):
            if key in value:
                keep[key] = value[key]
        return keep or value
    return value


def _value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "value" in value:
            return _value_text(value.get("value"))
        return " ".join(_value_text(item) for item in value.values())
    if isinstance(value, list):
        return "; ".join(_value_text(item) for item in value if _value_text(item))
    return str(value).strip()


def _compact_refs(value: Any, limit: int = 8) -> List[str]:
    return list(dict.fromkeys(_iter_evidence_refs(value)))[:limit]


def _is_phase3(value: Any) -> bool:
    text = _value_text(value).lower().replace("_", " ")
    return "phase 3" in text or "phase iii" in text


def _has_negative_results_phrase(value: Any) -> bool:
    text = _value_text(value).lower()
    return any(
        phrase in text
        for phrase in (
            "no results-based conclusion",
            "no outcome results",
            "does not provide outcome",
            "not provide outcome",
            "no numeric outcome",
        )
    )


_EXPIRY_RE = re.compile(r"\b([A-Z]{2,5})\s*:\s*(\d{4}-\d{2}-\d{2})\b")
_VALID_TO_RE = re.compile(
    r"Valid\s*To\s*:\s*([^\n\r]+?)(?=\n\s*(?:MAH|Holder|Dosage|Manufacturing|Registration|$))",
    re.IGNORECASE | re.DOTALL,
)


def _canonical_ip_region(raw_region: str) -> str:
    region = str(raw_region or "").strip().upper()
    if region in {"EP", "EU"}:
        return "EU"
    if region in {"RU", "EA", "EAEU"}:
        return "EAEU" if region in {"EA", "EAEU"} else "RU"
    return region


def _remaining_months(expiry_date: str) -> Optional[int]:
    try:
        year, month, day = [int(part) for part in expiry_date.split("-")]
        target = date(year, month, day)
    except Exception:
        return None
    today = date.today()
    return (target.year - today.year) * 12 + (target.month - today.month) - (1 if target.day < today.day else 0)


def _extract_valid_to(snippet: str) -> Optional[str]:
    match = _VALID_TO_RE.search(str(snippet or ""))
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip(" :;-")
    if not value or value.lower() in {"n/a", "na", "none", "null"}:
        return None
    if value.lower().startswith(("mah", "holder", "dosage", "manufacturing", "registration")):
        return None
    return value


class ExecEvidenceAssembler:
    def __init__(self, retriever: Any = None):
        self.retriever = retriever

    def _resolved_allowed_doc_kinds(
        self,
        base_packet: Dict[str, Any],
        plan: ExecQuestionPlan,
    ) -> List[str]:
        return reconcile_exec_doc_kinds(
            base_packet.get("allowed_doc_kinds", []),
            plan.retrieval_plan.doc_kinds or [],
        )

    def _selected_sections(
        self,
        base_packet: Dict[str, Any],
        plan: ExecQuestionPlan,
    ) -> Dict[str, Any]:
        selected: Dict[str, Any] = {}
        ordered_sections = []
        seen_sections = set()
        for section in list(base_packet.get("required_sections", []) or []) + list(plan.needed_dossier_sections or []):
            section_name = str(section or "").strip()
            if not section_name or section_name in seen_sections:
                continue
            seen_sections.add(section_name)
            ordered_sections.append(section_name)
        for section in ordered_sections:
            if section not in base_packet:
                continue
            value = base_packet.get(section)
            if isinstance(value, list):
                sample_limit = 12 if section in {"clinical_studies", "patent_families", "commercial_signals"} else 8
                selected[section] = [_compact_value(item) for item in value[:sample_limit]]
            elif isinstance(value, dict):
                if section == "dossier_quality_v2":
                    selected[section] = {
                        "coverage": value.get("coverage", {}),
                        "decision_readiness": value.get("decision_readiness", {}),
                        "critical_unknowns": value.get("critical_unknowns", [])[:6],
                        "notes": value.get("notes", [])[:6],
                    }
                elif section == "coverage_ledger":
                    selected[section] = {
                        "totals": value.get("totals", {}),
                        "section_coverage": {
                            key: {
                                "decision_readiness": (section_value or {}).get("decision_readiness"),
                                "indexed_docs": (section_value or {}).get("indexed_docs"),
                                "attached_docs": (section_value or {}).get("attached_docs"),
                            }
                            for key, section_value in list((value.get("section_coverage", {}) or {}).items())[:8]
                        },
                    }
                else:
                    selected[section] = value
            else:
                selected[section] = value
        return selected

    def _contract_limits(self, plan: ExecQuestionPlan) -> Dict[str, Any]:
        retrieval = plan.retrieval_plan
        return {
            "max_docs": max(1, int(retrieval.max_docs or 12)),
            "max_chunks": max(1, int(retrieval.max_chunks or 30)),
            "max_per_doc_kind": {
                normalize_exec_doc_kind(item.doc_kind): max(1, int(item.max_chunks))
                for item in (retrieval.doc_kind_limits or [])
                if normalize_exec_doc_kind(item.doc_kind)
            },
        }

    def _select_existing_evidence(
        self,
        base_packet: Dict[str, Any],
        plan: ExecQuestionPlan,
    ) -> List[Dict[str, Any]]:
        limits = self._contract_limits(plan)
        allowed_doc_kinds = set(self._resolved_allowed_doc_kinds(base_packet, plan))
        per_kind_limit = limits["max_per_doc_kind"]
        counts = Counter()
        selected: List[Dict[str, Any]] = []
        for item in base_packet.get("evidence_registry", []) or []:
            doc_kind = normalize_exec_doc_kind(item.get("doc_kind"))
            if allowed_doc_kinds and doc_kind not in allowed_doc_kinds:
                continue
            if per_kind_limit.get(doc_kind) and counts[doc_kind] >= per_kind_limit[doc_kind]:
                continue
            selected.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "doc_id": item.get("doc_id"),
                    "doc_kind": doc_kind,
                    "source_label": _source_label(item),
                    "source_url": item.get("source_url"),
                    "page": item.get("page"),
                    "snippet": item.get("snippet", ""),
                }
            )
            counts[doc_kind] += 1
            if len(selected) >= limits["max_chunks"]:
                break
        return selected[: limits["max_chunks"]]

    def _retrieve_additional_evidence(
        self,
        base_packet: Dict[str, Any],
        plan: ExecQuestionPlan,
        case_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not self.retriever:
            return []
        limits = self._contract_limits(plan)
        selected: List[Dict[str, Any]] = []
        canonical_doc_kinds = self._resolved_allowed_doc_kinds(base_packet, plan)
        allowed_doc_kinds = expand_exec_doc_kinds(canonical_doc_kinds)
        per_kind_limit = limits["max_per_doc_kind"]
        counts = Counter()
        for query in (plan.retrieval_plan.queries or [])[: limits["max_docs"]]:
            try:
                raw_items = self.retriever.retrieve_by_case(
                    query=query,
                    case_id=case_id,
                    doc_kind=allowed_doc_kinds or None,
                    top_n=limits["max_chunks"],
                )
            except TypeError:
                raw_items = self.retriever.retrieve_by_case(
                    query=query,
                    case_id=case_id,
                    doc_kind=allowed_doc_kinds or None,
                )
            except Exception:
                continue
            for item in raw_items or []:
                normalized = normalize_retrieval_item(item)
                doc_kind = normalize_exec_doc_kind(normalized.get("doc_kind"))
                if canonical_doc_kinds and doc_kind not in canonical_doc_kinds:
                    continue
                if per_kind_limit.get(doc_kind) and counts[doc_kind] >= per_kind_limit[doc_kind]:
                    continue
                selected.append(
                    {
                        "evidence_id": "",
                        "doc_id": normalized.get("doc_id"),
                        "doc_kind": doc_kind,
                        "source_label": _source_label(normalized),
                        "source_url": normalized.get("source_url"),
                        "page": normalized.get("page"),
                        "snippet": normalized.get("snippet", ""),
                    }
                )
                counts[doc_kind] += 1
                if len(selected) >= limits["max_chunks"]:
                    return selected
        return selected

    def _group_evidence(
        self,
        selected_sections: Dict[str, Any],
        selected_evidence: List[Dict[str, Any]],
        plan: ExecQuestionPlan,
    ) -> Dict[str, Any]:
        by_geo: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"facts": [], "evidence_refs": [], "doc_kinds": []}
        )
        for section_name, value in selected_sections.items():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                region = _region_from_record(item)
                by_geo[region]["facts"].append({"section": section_name, "record": item})
                by_geo[region]["evidence_refs"].extend(list(_iter_evidence_refs(item)))
        by_doc_kind: Dict[str, List[str]] = defaultdict(list)
        for item in selected_evidence:
            doc_kind = normalize_exec_doc_kind(item.get("doc_kind")) or "unknown"
            by_doc_kind[doc_kind].append(str(item.get("evidence_id") or item.get("doc_id") or ""))
            for region, payload in by_geo.items():
                if item.get("evidence_id") and item["evidence_id"] in payload["evidence_refs"]:
                    payload["doc_kinds"].append(doc_kind)

        by_needed_fact: Dict[str, List[str]] = {}
        searchable = [
            " ".join(
                [
                    str(item.get("doc_kind") or ""),
                    str(item.get("source_label") or ""),
                    str(item.get("snippet") or ""),
                ]
            ).lower()
            for item in selected_evidence
        ]
        evidence_ids = [
            str(item.get("evidence_id") or item.get("doc_id") or "")
            for item in selected_evidence
        ]
        for fact in plan.needed_facts:
            fact_tokens = [token for token in str(fact).lower().replace("_", " ").split() if token]
            matched_ids = []
            for idx, haystack in enumerate(searchable):
                if fact_tokens and all(token in haystack for token in fact_tokens[:2]):
                    matched_ids.append(evidence_ids[idx])
            by_needed_fact[fact] = matched_ids[:8]
        return {
            "by_geo": dict(by_geo),
            "by_doc_kind": {key: value[:12] for key, value in by_doc_kind.items()},
            "by_needed_fact": by_needed_fact,
        }

    def _find_contradictions(self, selected_sections: Dict[str, Any]) -> List[Dict[str, Any]]:
        contradictions: List[Dict[str, Any]] = []
        registrations = selected_sections.get("registrations", [])
        if isinstance(registrations, list):
            verdicts_by_region: Dict[str, set] = defaultdict(set)
            for item in registrations:
                if not isinstance(item, dict):
                    continue
                verdicts_by_region[_region_from_record(item)].add(str(item.get("verdict") or ""))
            for region, verdicts in verdicts_by_region.items():
                cleaned = {value for value in verdicts if value}
                if len(cleaned) > 1:
                    contradictions.append(
                        {
                            "summary": f"Conflicting registration verdicts for {region}: {sorted(cleaned)}",
                            "evidence_refs": [],
                        }
                    )
        return contradictions

    def _missing_evidence_classes(
        self,
        plan: ExecQuestionPlan,
        grouped: Dict[str, Any],
    ) -> List[str]:
        missing: List[str] = []
        by_needed_fact = grouped.get("by_needed_fact", {}) or {}
        required = list(plan.gates.positive_verdict_requires) + list(plan.needed_facts)
        for fact in required:
            if fact and not by_needed_fact.get(fact):
                missing.append(fact)
        return list(dict.fromkeys(missing))

    def _build_contract_linkage(
        self,
        selected_sections: Dict[str, Any],
        selected_evidence: List[Dict[str, Any]],
        base_packet: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        evidence_by_ref: Dict[str, Dict[str, Any]] = {}
        for item in list(selected_evidence) + list((base_packet or {}).get("evidence_registry", []) or []):
            for ref_key in ("evidence_id", "doc_id"):
                ref = str(item.get(ref_key) or "").strip()
                if ref and ref not in evidence_by_ref:
                    evidence_by_ref[ref] = item

        def _doc_kind(ref: str) -> str:
            return normalize_exec_doc_kind((evidence_by_ref.get(ref) or {}).get("doc_kind"))

        clinical_linked: List[Dict[str, Any]] = []
        phase3_with_results_refs = 0
        for study in selected_sections.get("clinical_studies", []) or []:
            if not isinstance(study, dict) or not _is_phase3(study.get("phase")):
                continue
            refs = _compact_refs(study)
            result_refs = [
                ref for ref in refs
                if _doc_kind(ref) in {"ctgov_results", "ctgov_documents", "publication", "scientific_pmc", "scientific_pdf"}
            ]
            if result_refs:
                phase3_with_results_refs += 1
            conclusion_text = _value_text(study.get("conclusion"))
            clinical_linked.append(
                {
                    "study_id": _value_text(study.get("study_id")),
                    "phase": _value_text(study.get("phase")),
                    "status": _value_text(study.get("status")),
                    "has_ctgov_results_evidence": bool(result_refs),
                    "has_result_conclusion": bool(conclusion_text) and not _has_negative_results_phrase(conclusion_text),
                    "result_evidence_refs": result_refs[:5],
                }
            )

        expiry_by_region: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for family in selected_sections.get("patent_families", []) or []:
            if not isinstance(family, dict):
                continue
            family_id = family.get("family_id") or _value_text(family.get("representative_pub")) or "unknown"
            for item in family.get("expiry_by_country", []) or []:
                value = _value_text(item)
                match = _EXPIRY_RE.search(value)
                if not match:
                    continue
                raw_region, expiry_date = match.groups()
                region = _canonical_ip_region(raw_region)
                refs = _compact_refs(item) or _compact_refs(family)
                expiry_by_region[region].append(
                    {
                        "family_id": family_id,
                        "raw_region": raw_region,
                        "expiry_date": expiry_date,
                        "remaining_time_months": _remaining_months(expiry_date),
                        "evidence_refs": refs[:5],
                    }
                )

        eaeu_regs: List[Dict[str, Any]] = []
        eaeu_evidence_snippets = [
            item for item in selected_evidence
            if normalize_exec_doc_kind(item.get("doc_kind")) == "eaeu_document"
        ]
        for reg in selected_sections.get("registrations", []) or []:
            if not isinstance(reg, dict) or str(reg.get("region") or "").strip().upper() != "EAEU":
                continue
            refs = _compact_refs(reg)
            snippets = [
                str((evidence_by_ref.get(ref) or {}).get("snippet") or "")
                for ref in refs
            ]
            snippets.extend(str(item.get("snippet") or "") for item in eaeu_evidence_snippets[:3])
            valid_to_values = [value for value in (_extract_valid_to(snippet) for snippet in snippets) if value]
            identifiers = [
                _value_text(item)
                for item in reg.get("identifiers", []) or []
                if _value_text(item)
            ]
            forms_strengths = [
                _value_text(item)
                for item in reg.get("forms_strengths", []) or []
                if _value_text(item)
            ]
            eaeu_regs.append(
                {
                    "status": _value_text(reg.get("status")),
                    "mah": _value_text(reg.get("mah")),
                    "identifiers": identifiers[:5],
                    "forms_strengths": forms_strengths[:6],
                    "valid_to": valid_to_values[0] if valid_to_values else None,
                    "valid_to_status": "evidenced" if valid_to_values else "missing_in_source",
                    "identifier_mah_linked": bool(identifiers and _value_text(reg.get("mah")) and refs),
                    "strength_traceability": "registration_forms_strengths_present" if forms_strengths else "missing",
                    "evidence_refs": refs[:8],
                }
            )

        ip_regions_required = {"US", "EU", "RU", "EAEU"}
        expiry_region_map = {region: items for region, items in sorted(expiry_by_region.items())}
        return {
            "phase3_results": {
                "phase3_study_count": len(clinical_linked),
                "phase3_with_ctgov_results_evidence": phase3_with_results_refs,
                "studies": clinical_linked[:12],
            },
            "ip_window": {
                "expiry_by_region": expiry_region_map,
                "missing_required_regions": sorted(ip_regions_required - set(expiry_region_map)),
            },
            "eaeu_registration": {
                "registrations": eaeu_regs[:6],
                "has_identifier_mah_linkage": any(item["identifier_mah_linked"] for item in eaeu_regs),
                "has_valid_to": any(item["valid_to"] for item in eaeu_regs),
            },
        }

    def assemble(
        self,
        base_packet: Dict[str, Any],
        plan: ExecQuestionPlan,
        case_id: Optional[str] = None,
        allow_retrieval: bool = True,
    ) -> Dict[str, Any]:
        selected_sections = self._selected_sections(base_packet, plan)
        selected_evidence = self._select_existing_evidence(base_packet, plan)
        retrieved_evidence = self._retrieve_additional_evidence(base_packet, plan, case_id) if allow_retrieval else []
        all_evidence = list(selected_evidence)
        seen = {
            (
                str(item.get("evidence_id") or ""),
                str(item.get("doc_id") or ""),
                str(item.get("doc_kind") or ""),
                str(item.get("snippet") or "")[:160],
            )
            for item in all_evidence
        }
        for item in retrieved_evidence:
            key = (
                str(item.get("evidence_id") or ""),
                str(item.get("doc_id") or ""),
                str(item.get("doc_kind") or ""),
                str(item.get("snippet") or "")[:160],
            )
            if key in seen:
                continue
            seen.add(key)
            all_evidence.append(item)
        grouped = self._group_evidence(selected_sections, all_evidence, plan)
        contradictions = self._find_contradictions(selected_sections)
        missing = self._missing_evidence_classes(plan, grouped)
        contract_linkage = self._build_contract_linkage(selected_sections, all_evidence, base_packet)
        return {
            "question_id": plan.question_id,
            "answer_type": plan.answer_type,
            "business_lens": plan.business_lens,
            "contract_summary": {
                "needed_facts": list(plan.needed_facts),
                "needed_dossier_sections": list(plan.needed_dossier_sections),
                "doc_kinds": list(plan.retrieval_plan.doc_kinds),
                "queries": list(plan.retrieval_plan.queries),
            },
            "contract_linkage": contract_linkage,
            "selected_sections": selected_sections,
            "selected_evidence": all_evidence,
            "selected_evidence_ids": [
                str(item.get("evidence_id") or item.get("doc_id") or "")
                for item in all_evidence
                if str(item.get("evidence_id") or item.get("doc_id") or "")
            ],
            "grouped_evidence": grouped,
            "contradictions": contradictions,
            "missing_evidence_classes": missing,
            "critical_unknowns": list((base_packet.get("critical_unknowns") or [])[:8]),
            "provenance_summary": [
                {
                    "source_label": item.get("source_label"),
                    "doc_kind": item.get("doc_kind"),
                    "doc_id": item.get("doc_id"),
                    "page": item.get("page"),
                }
                for item in all_evidence[:12]
            ],
            "evidence_packet_summary": {
                "selected_evidence_count": len(all_evidence),
                "retrieved_extra_count": len(retrieved_evidence),
                "doc_kind_counts": dict(
                    Counter(str(item.get("doc_kind") or "unknown") for item in all_evidence)
                ),
                "geo_count": len(grouped.get("by_geo", {})),
                "contract_linkage_summary": {
                    "phase3_with_ctgov_results_evidence": contract_linkage.get("phase3_results", {}).get("phase3_with_ctgov_results_evidence", 0),
                    "ip_regions_with_expiry": sorted((contract_linkage.get("ip_window", {}).get("expiry_by_region") or {}).keys()),
                    "eaeu_has_valid_to": contract_linkage.get("eaeu_registration", {}).get("has_valid_to", False),
                    "eaeu_has_identifier_mah_linkage": contract_linkage.get("eaeu_registration", {}).get("has_identifier_mah_linkage", False),
                },
            },
        }
