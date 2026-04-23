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
    "ru_patent_fips": "RU / FIPS patent registry",
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
    "patent_legal_events": ["patent_legal_events", "ru_patent_fips"],
    "patent_expiry_us": ["patent_expiry_us"],
}

_EAEU_MEMBER_STATES = {"RU", "BY", "AM", "KZ", "KG"}
_VALIDITY_INDEFINITE_RE = re.compile(
    r"\b(indefinite(?:ly)?|без\s+срока(?:\s+действия)?|бессроч(?:но|ный|ная|ные))\b",
    re.IGNORECASE,
)
_VALID_TO_LINE_RE = re.compile(
    r"(?:Valid\s*(?:To|Until)|Действительно\s*до|Срок\s*действия(?:\s*до)?)\s*:\s*([^\n\r]*)",
    re.IGNORECASE,
)
_RU_FIPS_DOC_ID_RE = re.compile(
    r'"doc_id"\s*:\s*"((?:RU|EA)[\dA-Z ]+?)_[\d ]+?"',
)
_RU_FIPS_EXPIRY_DATE_RE = re.compile(
    r'"expiry_date"\s*:\s*"([\d\-\s]+?)"',
)
_RU_FIPS_JURISDICTION_RE = re.compile(
    r'"jurisdiction"\s*:\s*"([A-Z]{2})"',
)


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
            "family_id",
            "representative_pub",
            "priority_date",
            "what_blocks",
            "technical_focus",
            "process_relevance",
            "mah",
            "identifiers",
            "forms_strengths",
            "valid_to",
            "validity_type",
            "validity_evidence_refs",
            "country_coverage",
            "legal_status_snapshot",
            "expiry_by_country",
            "description",
            "evidence_grade",
            "source_patent_refs",
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


def _extract_validity_state(snippet: str) -> Optional[Dict[str, Optional[str]]]:
    text = str(snippet or "")
    if not text:
        return None
    line_match = _VALID_TO_LINE_RE.search(text)
    if line_match:
        raw_value = re.sub(r"\s+", " ", line_match.group(1) or "").strip(" :;-")
        lowered = raw_value.lower()
        if not raw_value or lowered in {"n/a", "na", "none", "null", "-", "—"}:
            return {"valid_to": None, "validity_type": "missing_in_source"}
        if lowered.startswith(("mah", "holder", "dosage", "manufacturing", "registration")):
            return {"valid_to": None, "validity_type": "missing_in_source"}
        if _VALIDITY_INDEFINITE_RE.search(raw_value):
            return {"valid_to": None, "validity_type": "indefinite"}
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", raw_value)
        return {
            "valid_to": date_match.group(0) if date_match else raw_value,
            "validity_type": "date_present",
        }
    if _VALIDITY_INDEFINITE_RE.search(text):
        return {"valid_to": None, "validity_type": "indefinite"}
    return None


def _validity_rank(value: str) -> int:
    order = {
        "date_present": 3,
        "indefinite": 2,
        "not_applicable": 1,
        "missing_in_source": 0,
    }
    return order.get(str(value or "").strip().lower(), -1)


def _infer_region_window_status(
    *,
    legal_statuses: List[str],
    expiry_dates: List[str],
    has_source_only: bool,
) -> str:
    normalized = [status.strip().lower() for status in legal_statuses if str(status or "").strip()]
    for status in normalized:
        if any(marker in status for marker in ("expired", "lapsed", "revoked", "withdrawn", "ceased")):
            return "open"
        if any(marker in status for marker in ("granted", "pending", "active", "in force")):
            return "potentially_blocked"
    remaining_months = [months for months in (_remaining_months(item) for item in expiry_dates) if months is not None]
    if remaining_months:
        return "open" if min(remaining_months) <= 0 else "potentially_blocked"
    if has_source_only:
        return "unresolved_with_source_evidence"
    return "missing"


def _extract_ru_fips_source_entries(snippet: str, evidence_ref: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    text = str(snippet or "")
    for match in _RU_FIPS_DOC_ID_RE.finditer(text):
        raw_pub = match.group(1).replace(" ", "")
        window = text[match.end():match.end() + 3000]
        expiry_match = _RU_FIPS_EXPIRY_DATE_RE.search(window)
        if not expiry_match:
            continue
        expiry_value = expiry_match.group(1).replace(" ", "").replace("\n", "")
        if not re.match(r"\d{4}-\d{2}-\d{2}$", expiry_value):
            continue
        region_match = _RU_FIPS_JURISDICTION_RE.search(window)
        region = _canonical_ip_region(region_match.group(1) if region_match else "RU")
        entries.append(
            {
                "region": region,
                "representative_pub": raw_pub,
                "expiry_date": expiry_value,
                "remaining_time_months": _remaining_months(expiry_value),
                "legal_status": "",
                "evidence_refs": [evidence_ref],
            }
        )
    return entries


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
        patent_snapshot_by_region: Dict[str, Dict[str, Any]] = {
            region: {
                "family_entries": [],
                "source_only_entries": [],
                "legal_statuses": [],
                "expiry_dates": [],
                "evidence_refs": [],
            }
            for region in ("US", "EU", "RU", "EAEU")
        }
        for family in selected_sections.get("patent_families", []) or []:
            if not isinstance(family, dict):
                continue
            family_id = family.get("family_id") or _value_text(family.get("representative_pub")) or "unknown"
            representative_pub = _value_text(family.get("representative_pub")) or family_id
            legal_status = _value_text(family.get("legal_status_snapshot"))
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
                        "representative_pub": representative_pub,
                        "raw_region": raw_region,
                        "expiry_date": expiry_date,
                        "legal_status": legal_status,
                        "remaining_time_months": _remaining_months(expiry_date),
                        "evidence_refs": refs[:5],
                    }
                )
                region_snapshot = patent_snapshot_by_region.setdefault(
                    region,
                    {"family_entries": [], "source_only_entries": [], "legal_statuses": [], "expiry_dates": [], "evidence_refs": []},
                )
                region_snapshot["family_entries"].append(
                    {
                        "family_id": family_id,
                        "representative_pub": representative_pub,
                        "expiry_date": expiry_date,
                        "legal_status": legal_status,
                        "remaining_time_months": _remaining_months(expiry_date),
                        "evidence_refs": refs[:5],
                    }
                )
                if legal_status:
                    region_snapshot["legal_statuses"].append(legal_status)
                region_snapshot["expiry_dates"].append(expiry_date)
                region_snapshot["evidence_refs"].extend(refs[:5])

        for item in selected_evidence:
            doc_kind = normalize_exec_doc_kind(item.get("doc_kind"))
            evidence_ref = str(item.get("evidence_id") or item.get("doc_id") or "").strip()
            if not evidence_ref:
                continue
            snippet = str(item.get("snippet") or "")
            source_entries: List[Dict[str, Any]] = []
            if doc_kind == "ru_patent_fips":
                source_entries = _extract_ru_fips_source_entries(snippet, evidence_ref)
            else:
                for match in _EXPIRY_RE.finditer(snippet):
                    raw_region, expiry_date = match.groups()
                    source_entries.append(
                        {
                            "representative_pub": "",
                            "expiry_date": expiry_date,
                            "remaining_time_months": _remaining_months(expiry_date),
                            "legal_status": "",
                            "evidence_refs": [evidence_ref],
                            "region": _canonical_ip_region(raw_region),
                        }
                    )
            for entry in source_entries:
                region = _canonical_ip_region(entry.get("region") or "RU")
                region_snapshot = patent_snapshot_by_region.setdefault(
                    region,
                    {"family_entries": [], "source_only_entries": [], "legal_statuses": [], "expiry_dates": [], "evidence_refs": []},
                )
                source_payload = {
                    "representative_pub": entry.get("representative_pub") or "",
                    "expiry_date": entry.get("expiry_date") or "",
                    "remaining_time_months": entry.get("remaining_time_months"),
                    "legal_status": entry.get("legal_status") or "",
                    "evidence_refs": list(entry.get("evidence_refs") or [])[:5],
                }
                dedupe_key = (
                    region,
                    source_payload["representative_pub"],
                    source_payload["expiry_date"],
                    tuple(source_payload["evidence_refs"]),
                )
                existing_keys = {
                    (
                        region,
                        str(existing.get("representative_pub") or ""),
                        str(existing.get("expiry_date") or ""),
                        tuple(existing.get("evidence_refs") or []),
                    )
                    for existing in region_snapshot["source_only_entries"]
                }
                if dedupe_key in existing_keys:
                    continue
                region_snapshot["source_only_entries"].append(source_payload)
                if source_payload["legal_status"]:
                    region_snapshot["legal_statuses"].append(source_payload["legal_status"])
                if source_payload["expiry_date"]:
                    region_snapshot["expiry_dates"].append(source_payload["expiry_date"])
                region_snapshot["evidence_refs"].extend(source_payload["evidence_refs"])

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
            validity_refs = list(dict.fromkeys(
                list(reg.get("validity_evidence_refs") or [])
                + _compact_refs(reg.get("valid_to"))
            ))
            validity_type = str(reg.get("validity_type") or "").strip().lower() or "missing_in_source"
            valid_to_value = _value_text(reg.get("valid_to")) or None
            best_validity_rank = _validity_rank(validity_type)
            for ref in refs:
                snippet_item = evidence_by_ref.get(ref) or {}
                inferred = _extract_validity_state(snippet_item.get("snippet") or "")
                if not inferred:
                    continue
                inferred_type = str(inferred.get("validity_type") or "").strip().lower()
                if _validity_rank(inferred_type) > best_validity_rank:
                    validity_type = inferred_type
                    valid_to_value = inferred.get("valid_to") or valid_to_value
                    best_validity_rank = _validity_rank(inferred_type)
                if str(snippet_item.get("evidence_id") or "").strip():
                    validity_refs.append(str(snippet_item.get("evidence_id")).strip())
            for item in eaeu_evidence_snippets[:3]:
                inferred = _extract_validity_state(item.get("snippet") or "")
                if not inferred:
                    continue
                inferred_type = str(inferred.get("validity_type") or "").strip().lower()
                if _validity_rank(inferred_type) > best_validity_rank:
                    validity_type = inferred_type
                    valid_to_value = inferred.get("valid_to") or valid_to_value
                    best_validity_rank = _validity_rank(inferred_type)
                if str(item.get("evidence_id") or "").strip():
                    validity_refs.append(str(item.get("evidence_id")).strip())
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
                    "valid_to": valid_to_value,
                    "validity_type": validity_type,
                    "validity_evidence_refs": list(dict.fromkeys(validity_refs))[:8],
                    "identifier_mah_linked": bool(identifiers and _value_text(reg.get("mah")) and refs),
                    "strength_traceability": "registration_forms_strengths_present" if forms_strengths else "missing",
                    "evidence_refs": refs[:8],
                }
            )

        ip_regions_required = {"US", "EU", "RU", "EAEU"}
        expiry_region_map = {region: items for region, items in sorted(expiry_by_region.items())}
        patent_snapshot = {}
        for region in sorted(ip_regions_required):
            payload = patent_snapshot_by_region.get(region, {}) or {}
            legal_statuses = list(dict.fromkeys(payload.get("legal_statuses", []) or []))
            expiry_dates = list(dict.fromkeys(payload.get("expiry_dates", []) or []))
            evidence_refs = list(dict.fromkeys(payload.get("evidence_refs", []) or []))
            family_entries = payload.get("family_entries", []) or []
            source_only_entries = payload.get("source_only_entries", []) or []
            patent_snapshot[region] = {
                "window_status": _infer_region_window_status(
                    legal_statuses=legal_statuses,
                    expiry_dates=expiry_dates,
                    has_source_only=bool(source_only_entries),
                ),
                "family_entries": family_entries[:8],
                "source_only_entries": source_only_entries[:8],
                "legal_statuses": legal_statuses[:6],
                "expiry_dates": expiry_dates[:8],
                "evidence_refs": evidence_refs[:10],
            }
        resolved_ip_regions = {
            region for region, payload in patent_snapshot.items()
            if payload.get("window_status") != "missing"
        }
        return {
            "phase3_results": {
                "phase3_study_count": len(clinical_linked),
                "phase3_with_ctgov_results_evidence": phase3_with_results_refs,
                "studies": clinical_linked[:12],
            },
            "ip_window": {
                "expiry_by_region": expiry_region_map,
                "family_linked_regions": sorted(expiry_region_map),
                "resolved_required_regions": sorted(resolved_ip_regions),
                "missing_required_regions": sorted(ip_regions_required - resolved_ip_regions),
            },
            "patent_legal_status_snapshot": {
                "required_regions": sorted(ip_regions_required),
                "regions": patent_snapshot,
                "resolved_regions": sorted(resolved_ip_regions),
                "unresolved_regions": sorted(ip_regions_required - resolved_ip_regions),
            },
            "eaeu_registration": {
                "registrations": eaeu_regs[:6],
                "has_identifier_mah_linkage": any(item["identifier_mah_linked"] for item in eaeu_regs),
                "has_valid_to": any(item["validity_type"] == "date_present" for item in eaeu_regs),
                "has_validity_state": any(item["validity_type"] != "missing_in_source" for item in eaeu_regs),
                "validity_types": sorted({item["validity_type"] for item in eaeu_regs if item.get("validity_type")}),
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
                    "ip_regions_resolved": list((contract_linkage.get("patent_legal_status_snapshot", {}) or {}).get("resolved_regions", [])),
                    "eaeu_has_valid_to": contract_linkage.get("eaeu_registration", {}).get("has_valid_to", False),
                    "eaeu_has_validity_state": contract_linkage.get("eaeu_registration", {}).get("has_validity_state", False),
                    "eaeu_validity_types": list((contract_linkage.get("eaeu_registration", {}) or {}).get("validity_types", [])),
                    "eaeu_has_identifier_mah_linkage": contract_linkage.get("eaeu_registration", {}).get("has_identifier_mah_linkage", False),
                },
            },
        }
