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
_OFFICIAL_PATENT_REGISTER_NO_HIT_RE = re.compile(
    r"OFFICIAL_PATENT_REGISTER_NO_HIT\s*\|\s*region=([A-Z]+)\s*\|\s*search_term=([^|]+)\|\s*patents=0(?:\s*\|\s*as_of=([\d-]+))?",
    re.IGNORECASE,
)
_POSITIVE_STATUS_MARKERS = {
    "active",
    "approved",
    "authorised",
    "authorized",
    "confirmed",
    "in force",
    "registered",
    "valid",
}
_STRENGTH_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|%)\b", re.IGNORECASE)


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


def _is_priority_contract_evidence(item: Dict[str, Any], doc_kind: str) -> bool:
    if doc_kind != "ru_patent_fips":
        return False
    snippet = str(item.get("snippet") or "")
    if _OFFICIAL_PATENT_REGISTER_NO_HIT_RE.search(snippet):
        return True
    return bool(_RU_FIPS_DOC_ID_RE.search(snippet) and _RU_FIPS_EXPIRY_DATE_RE.search(snippet))


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


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _value_text(value or "")).strip()


def _status_positive(value: Any) -> bool:
    text = _normalize_text(value).lower()
    return any(marker in text for marker in _POSITIVE_STATUS_MARKERS)


def _dedupe_text(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values or []:
        text = _normalize_text(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


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
        if any(marker in status for marker in ("no_listed_pharma_patents", "no patents found", "no_listed_patents")):
            return "open"
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
    official_no_hit = _OFFICIAL_PATENT_REGISTER_NO_HIT_RE.search(text)
    if official_no_hit:
        region_raw, search_term, as_of = official_no_hit.groups()
        entries.append(
            {
                "region": _canonical_ip_region(region_raw or "EAEU"),
                "representative_pub": "",
                "expiry_date": "",
                "remaining_time_months": None,
                "legal_status": "no_listed_pharma_patents",
                "search_term": str(search_term or "").strip(),
                "status_date": str(as_of or "").strip(),
                "source_kind": "official_no_hit",
                "evidence_refs": [evidence_ref],
            }
        )
        return entries
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
                "source_kind": "fips_expiry_record",
                "evidence_refs": [evidence_ref],
            }
        )
    return entries


def _extract_forms_strengths(
    registration: Dict[str, Any],
    product_contexts: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    forms_strengths: List[str] = []
    dosage_forms: List[str] = []
    strengths: List[str] = []

    for item in registration.get("forms_strengths", []) or []:
        raw = _normalize_text(item)
        if not raw:
            continue
        forms_strengths.append(raw)
        for match in _STRENGTH_TOKEN_RE.finditer(raw):
            strengths.append(match.group(0))
        form_candidate = _STRENGTH_TOKEN_RE.sub("", raw.split("|", 1)[0])
        form_candidate = re.sub(r"[;,:()]+", " ", form_candidate)
        form_candidate = re.sub(r"\s+", " ", form_candidate).strip(" |-")
        if form_candidate and not form_candidate.isdigit():
            dosage_forms.append(form_candidate)

    for context in product_contexts:
        if not isinstance(context, dict):
            continue
        dosage_forms.extend(context.get("dosage_forms", []) or [])
        strengths.extend(context.get("strengths", []) or [])

    return {
        "forms_strengths": _dedupe_text(forms_strengths),
        "dosage_forms": _dedupe_text(dosage_forms),
        "strengths": _dedupe_text(strengths),
    }


def _infer_registration_source_class(
    region: str,
    evidence_refs: List[str],
    evidence_by_ref: Dict[str, Dict[str, Any]],
) -> str:
    doc_kinds = {
        normalize_exec_doc_kind((evidence_by_ref.get(ref) or {}).get("doc_kind"))
        for ref in evidence_refs
        if str(ref or "").strip()
    }
    if region == "EAEU" and doc_kinds & {"eaeu_document", "eaeu_registration"}:
        return "EAEU-native"
    if region == "RU" and doc_kinds & {"grls", "grls_card", "ru_registration_export"}:
        return "GRLS"
    if region == "EU" and doc_kinds & {"smpc", "epar", "assessment_report", "eu_regulatory_summary"}:
        return "EU-native"
    if region == "US" and doc_kinds & {"label", "us_fda", "approval_letter"}:
        return "US-native"
    return "mixed"


def _identity_confidence(
    *,
    region: str,
    status_text: str,
    identifiers: List[str],
    evidence_refs: List[str],
    source_class: str,
    validity_type: str = "",
) -> str:
    source_native = source_class in {"EAEU-native", "GRLS", "EU-native", "US-native"}
    if _status_positive(status_text) and identifiers and evidence_refs and source_native:
        if region != "EAEU" or validity_type in {"date_present", "indefinite"}:
            return "HIGH"
        return "MEDIUM"
    if _status_positive(status_text) and evidence_refs:
        return "MEDIUM"
    return "LOW"


def _signal_searchable_text(
    signal: Dict[str, Any],
    evidence_by_ref: Dict[str, Dict[str, Any]],
) -> str:
    parts = [
        _normalize_text(signal.get("summary")),
        _normalize_text(signal.get("category")),
        _normalize_text(signal.get("source_name")),
        _normalize_text(signal.get("region")),
    ]
    for ref in _compact_refs(signal):
        item = evidence_by_ref.get(ref) or {}
        parts.extend(
            [
                _normalize_text(item.get("title")),
                _normalize_text(item.get("snippet")),
                _normalize_text(item.get("source_label")),
            ]
        )
    return " ".join(part for part in parts if part).lower()


def _best_identity_match(
    searchable_text: str,
    identifiers: List[str],
    mah: str,
    dosage_forms: List[str],
    strengths: List[str],
) -> str:
    if not searchable_text:
        return "none"
    for identifier in identifiers:
        token = _normalize_text(identifier).lower()
        if len(token) >= 4 and token in searchable_text:
            return "same_identifier"
    mah_token = _normalize_text(mah).lower()
    if mah_token and mah_token in searchable_text:
        return "mah_or_product_context"
    for token in _dedupe_text(list(dosage_forms) + list(strengths)):
        lowered = token.lower()
        if len(lowered) >= 3 and lowered in searchable_text:
            return "mah_or_product_context"
    return "inn_level_only"


def _identity_match_rank(value: str) -> int:
    return {
        "none": 0,
        "inn_level_only": 1,
        "mah_or_product_context": 2,
        "same_identifier": 3,
    }.get(str(value or "").strip().lower(), 0)


def _commercial_linkage_confidence(match_level: str, signal_count: int) -> str:
    if match_level == "same_identifier":
        return "HIGH"
    if match_level == "mah_or_product_context":
        return "MEDIUM"
    if match_level == "inn_level_only" and signal_count > 0:
        return "LOW"
    return "LOW"


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
        priority_allowed_doc_kinds = set(normalize_exec_doc_kind_list(base_packet.get("allowed_doc_kinds", []))) or allowed_doc_kinds
        per_kind_limit = limits["max_per_doc_kind"]
        counts = Counter()
        selected: List[Dict[str, Any]] = []
        seen_keys = set()

        def _append(item: Dict[str, Any], doc_kind: str, *, enforce_per_kind: bool = True) -> bool:
            key = (
                str(item.get("evidence_id") or ""),
                str(item.get("doc_id") or ""),
                str(item.get("snippet") or "")[:160],
            )
            if key in seen_keys:
                return False
            if enforce_per_kind and per_kind_limit.get(doc_kind) and counts[doc_kind] >= per_kind_limit[doc_kind]:
                return False
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
            seen_keys.add(key)
            counts[doc_kind] += 1
            return True

        registry = list(base_packet.get("evidence_registry", []) or [])
        for item in registry:
            doc_kind = normalize_exec_doc_kind(item.get("doc_kind"))
            if priority_allowed_doc_kinds and doc_kind not in priority_allowed_doc_kinds:
                continue
            if not _is_priority_contract_evidence(item, doc_kind):
                continue
            _append(item, doc_kind, enforce_per_kind=False)
            if len(selected) >= limits["max_chunks"]:
                return selected[: limits["max_chunks"]]

        for item in registry:
            doc_kind = normalize_exec_doc_kind(item.get("doc_kind"))
            if allowed_doc_kinds and doc_kind not in allowed_doc_kinds:
                continue
            _append(item, doc_kind)
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

        product_contexts_by_region: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for context in selected_sections.get("product_contexts", []) or []:
            if not isinstance(context, dict):
                continue
            product_contexts_by_region[_region_from_record(context)].append(context)

        registration_identity_map: List[Dict[str, Any]] = []
        registrations_by_region: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for reg in selected_sections.get("registrations", []) or []:
            if not isinstance(reg, dict):
                continue
            region = _region_from_record(reg)
            refs = _compact_refs(reg)
            identifiers = _dedupe_text(
                _value_text(item)
                for item in reg.get("identifiers", []) or []
                if _value_text(item)
            )
            mah = _normalize_text(reg.get("mah"))
            forms_payload = _extract_forms_strengths(reg, product_contexts_by_region.get(region, []))
            validity_type = str(reg.get("validity_type") or "").strip().lower() or "missing_in_source"
            valid_to_value = _normalize_text(reg.get("valid_to")) or None
            status_text = _normalize_text(reg.get("status") or reg.get("verdict"))
            source_class = _infer_registration_source_class(region, refs, evidence_by_ref)
            identity_entry = {
                "context": region,
                "status": status_text,
                "status_positive": _status_positive(status_text),
                "mah": mah,
                "identifiers": identifiers[:6],
                "forms_strengths": forms_payload["forms_strengths"][:6],
                "dosage_forms": forms_payload["dosage_forms"][:6],
                "strengths": forms_payload["strengths"][:6],
                "valid_to": valid_to_value,
                "validity_type": validity_type,
                "source_class": source_class,
                "identity_confidence": _identity_confidence(
                    region=region,
                    status_text=status_text,
                    identifiers=identifiers,
                    evidence_refs=refs,
                    source_class=source_class,
                    validity_type=validity_type,
                ),
                "evidence_refs": refs[:8],
            }
            registration_identity_map.append(identity_entry)
            registrations_by_region[region].append(identity_entry)

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
                    "search_term": entry.get("search_term") or "",
                    "status_date": entry.get("status_date") or "",
                    "source_kind": entry.get("source_kind") or "",
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
            official_no_hit_supported = any(
                str(item.get("source_kind") or "").strip() == "official_no_hit"
                or "no_listed_pharma_patents" in str(item.get("legal_status") or "").strip().lower()
                for item in source_only_entries
            )
            window_status = _infer_region_window_status(
                legal_statuses=legal_statuses,
                expiry_dates=expiry_dates,
                has_source_only=bool(source_only_entries),
            )
            if official_no_hit_supported:
                conclusion = "NO_LISTED_BLOCKING_PATENT_EVIDENCE"
                status_basis = "official_no_hit"
            elif window_status == "open":
                conclusion = "OPEN_WINDOW_EVIDENCE"
                status_basis = "expiry_or_status_evidence"
            elif window_status == "potentially_blocked":
                conclusion = "BLOCKING_OR_PENDING_EVIDENCE_PRESENT"
                status_basis = "active_or_future_expiry_evidence"
            elif window_status == "unresolved_with_source_evidence":
                conclusion = "SOURCE_EVIDENCE_PRESENT_BUT_WINDOW_UNRESOLVED"
                status_basis = "source_only_partial"
            else:
                conclusion = "UNRESOLVED"
                status_basis = "missing"
            sources_checked = []
            for entry in source_only_entries[:8]:
                source_kind = str(entry.get("source_kind") or "").strip()
                if source_kind == "official_no_hit":
                    source_name = "EAPO pharma register" if region == "EAEU" else "FIPS / official patent register"
                    result_text = "0 listed patents / no listed blocking pharma patents found"
                elif source_kind == "fips_expiry_record":
                    source_name = "FIPS / patent record"
                    expiry_text = str(entry.get("expiry_date") or "").strip()
                    result_text = f"expiry evidence {expiry_text}" if expiry_text else "source patent record"
                else:
                    source_name = "official source snippet"
                    result_text = str(entry.get("legal_status") or entry.get("expiry_date") or "source evidence").strip()
                sources_checked.append(
                    {
                        "source": source_name,
                        "query": str(entry.get("search_term") or "").strip() or None,
                        "result": result_text,
                        "status_date": str(entry.get("status_date") or "").strip() or None,
                        "evidence_refs": list(entry.get("evidence_refs") or [])[:5],
                    }
                )
            patent_snapshot[region] = {
                "window_status": window_status,
                "conclusion": conclusion,
                "status_basis": status_basis,
                "official_no_hit_supported": official_no_hit_supported,
                "family_entries": family_entries[:8],
                "source_only_entries": source_only_entries[:8],
                "legal_statuses": legal_statuses[:6],
                "expiry_dates": expiry_dates[:8],
                "sources_checked": sources_checked,
                "evidence_refs": evidence_refs[:10],
            }
        resolved_ip_regions = {
            region for region, payload in patent_snapshot.items()
            if payload.get("window_status") != "missing"
        }
        registration_context_relationships: List[Dict[str, Any]] = []
        ru_entries = registrations_by_region.get("RU", [])
        eaeu_entries = registrations_by_region.get("EAEU", [])
        if ru_entries and eaeu_entries:
            ru_identifiers = {identifier for entry in ru_entries for identifier in entry.get("identifiers", []) or []}
            eaeu_identifiers = {identifier for entry in eaeu_entries for identifier in entry.get("identifiers", []) or []}
            ru_mahs = {entry.get("mah") for entry in ru_entries if entry.get("mah")}
            eaeu_mahs = {entry.get("mah") for entry in eaeu_entries if entry.get("mah")}
            if (ru_identifiers and eaeu_identifiers and ru_identifiers.isdisjoint(eaeu_identifiers)) or (
                ru_mahs and eaeu_mahs and ru_mahs.isdisjoint(eaeu_mahs)
            ):
                registration_context_relationships.append(
                    {
                        "relationship": "separate_product_contexts",
                        "regions": ["RU", "EAEU"],
                        "basis": "Different RU and EAEU identifiers/MAH are treated as separate product contexts unless same-id linkage is evidenced.",
                        "evidence_refs": list(
                            dict.fromkeys(
                                [
                                    ref
                                    for entry in (ru_entries[:2] + eaeu_entries[:2])
                                    for ref in entry.get("evidence_refs", []) or []
                                ]
                            )
                        )[:10],
                    }
                )

        commercial_signals = [item for item in selected_sections.get("commercial_signals", []) or [] if isinstance(item, dict)]
        market_entry_linkage: Dict[str, Dict[str, Any]] = {}
        for target_region in ("RU", "EAEU"):
            identity_entries = registrations_by_region.get(target_region, [])
            if target_region == "RU":
                relevant_signals = [item for item in commercial_signals if _region_from_record(item) == "RU"]
                proxy_signals: List[Dict[str, Any]] = []
            else:
                relevant_signals = [
                    item for item in commercial_signals
                    if _region_from_record(item) in ({"EAEU"} | (_EAEU_MEMBER_STATES - {"RU"}))
                ]
                proxy_signals = [item for item in commercial_signals if _region_from_record(item) == "RU"]
            identifiers = _dedupe_text(
                identifier
                for entry in identity_entries
                for identifier in entry.get("identifiers", []) or []
            )
            mahs = _dedupe_text(entry.get("mah") for entry in identity_entries if entry.get("mah"))
            dosage_forms = _dedupe_text(
                form
                for entry in identity_entries
                for form in entry.get("dosage_forms", []) or []
            )
            strengths = _dedupe_text(
                strength
                for entry in identity_entries
                for strength in entry.get("strengths", []) or []
            )
            best_match = "none"
            linkage_refs: List[str] = []
            signal_regions = set()
            for signal in relevant_signals:
                searchable_text = _signal_searchable_text(signal, evidence_by_ref)
                match_level = _best_identity_match(
                    searchable_text,
                    identifiers,
                    "; ".join(mahs),
                    dosage_forms,
                    strengths,
                )
                if _identity_match_rank(match_level) > _identity_match_rank(best_match):
                    best_match = match_level
                signal_regions.add(_region_from_record(signal))
                if _identity_match_rank(match_level) >= 2:
                    linkage_refs.extend(_compact_refs(signal))
            all_signal_refs = [
                ref
                for item in relevant_signals
                for ref in _compact_refs(item)
            ]
            market_entry_linkage[target_region] = {
                "registration_anchor_present": any(entry.get("status_positive") for entry in identity_entries),
                "identity_confidence": max(
                    (str(entry.get("identity_confidence") or "LOW") for entry in identity_entries),
                    default="LOW",
                    key=lambda value: {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(value, 0),
                ),
                "registration_identifiers": identifiers[:6],
                "registration_mahs": mahs[:4],
                "dosage_forms": dosage_forms[:6],
                "strengths": strengths[:6],
                "commercial_signal_count": len(relevant_signals),
                "ru_proxy_signal_count": len(proxy_signals),
                "signal_regions": sorted(signal_regions),
                "identity_match": best_match if relevant_signals else "none",
                "linkage_confidence": _commercial_linkage_confidence(best_match, len(relevant_signals)),
                "validity_confirmed": any(
                    str(entry.get("validity_type") or "").strip().lower() in {"date_present", "indefinite"}
                    for entry in identity_entries
                ),
                "same_identifier_confirmed": best_match == "same_identifier",
                "product_context_match_confirmed": _identity_match_rank(best_match) >= 2,
                "evidence_refs": list(dict.fromkeys(linkage_refs or all_signal_refs))[:10],
            }

        ru_eaeu_sources = []
        ru_eaeu_evidence_refs: List[str] = []
        ru_eaeu_as_of_dates = []
        for region in ("RU", "EAEU"):
            region_payload = patent_snapshot.get(region, {}) or {}
            for source in region_payload.get("sources_checked", []) or []:
                ru_eaeu_sources.append(
                    {
                        "source": source.get("source"),
                        "jurisdiction": region,
                        "query": source.get("query"),
                        "result": source.get("result"),
                        "as_of_date": source.get("status_date"),
                        "evidence_refs": list(source.get("evidence_refs") or [])[:5],
                    }
                )
                ru_eaeu_evidence_refs.extend(list(source.get("evidence_refs") or [])[:5])
                if source.get("status_date"):
                    ru_eaeu_as_of_dates.append(str(source.get("status_date")))
        ru_eaeu_region_conclusions = {
            region: str((patent_snapshot.get(region) or {}).get("conclusion") or "UNRESOLVED")
            for region in ("RU", "EAEU")
        }
        if all(
            value in {"NO_LISTED_BLOCKING_PATENT_EVIDENCE", "OPEN_WINDOW_EVIDENCE"}
            for value in ru_eaeu_region_conclusions.values()
        ):
            ru_eaeu_conclusion = "NO_LISTED_BLOCKING_PATENT_EVIDENCE"
        elif any(value == "BLOCKING_OR_PENDING_EVIDENCE_PRESENT" for value in ru_eaeu_region_conclusions.values()):
            ru_eaeu_conclusion = "BLOCKING_OR_PENDING_EVIDENCE_PRESENT"
        elif any(value in {"NO_LISTED_BLOCKING_PATENT_EVIDENCE", "OPEN_WINDOW_EVIDENCE"} for value in ru_eaeu_region_conclusions.values()):
            ru_eaeu_conclusion = "PARTIAL_OPEN_WINDOW_EVIDENCE"
        else:
            ru_eaeu_conclusion = "UNRESOLVED"
        ru_eaeu_confidence = (
            "MEDIUM_HIGH"
            if ru_eaeu_conclusion == "NO_LISTED_BLOCKING_PATENT_EVIDENCE" and ru_eaeu_sources
            else "MEDIUM"
            if ru_eaeu_conclusion == "PARTIAL_OPEN_WINDOW_EVIDENCE"
            else "LOW"
        )

        regional_generic_opportunity = {}
        regional_licensing_opportunity = {}
        for region in ("RU", "EAEU", "EU", "US"):
            registration_supported = any(entry.get("status_positive") for entry in registrations_by_region.get(region, []))
            patent_payload = patent_snapshot.get(region, {}) or {}
            patent_conclusion = str(patent_payload.get("conclusion") or "UNRESOLVED")
            generic_refs = list(
                dict.fromkeys(
                    [
                        ref
                        for entry in registrations_by_region.get(region, [])
                        for ref in entry.get("evidence_refs", []) or []
                    ]
                    + list(patent_payload.get("evidence_refs", []) or [])
                )
            )[:10]
            if region in {"RU", "EAEU"} and registration_supported and patent_conclusion in {
                "NO_LISTED_BLOCKING_PATENT_EVIDENCE",
                "OPEN_WINDOW_EVIDENCE",
            }:
                generic_verdict = "POTENTIAL_GO"
                generic_reason = "Registration anchor exists and checked RU/EAEU patent sources do not show listed blocking window evidence."
            elif patent_conclusion == "BLOCKING_OR_PENDING_EVIDENCE_PRESENT":
                generic_verdict = "HOLD_OR_NO_GO"
                generic_reason = "Patent expiry/legal-status evidence still indicates active or pending blocker risk."
            else:
                generic_verdict = "NOT_EVIDENCED"
                generic_reason = "Generic opportunity remains region-dependent and not yet decision-grade for this jurisdiction."
            regional_generic_opportunity[region] = {
                "verdict": generic_verdict,
                "reason": generic_reason,
                "evidence_refs": generic_refs,
            }

            commercial_support = market_entry_linkage.get(region, {}) or {}
            if registration_supported and int(commercial_support.get("commercial_signal_count") or 0) > 0:
                licensing_verdict = "LOW"
                licensing_reason = "Registration/access context already exists, so the packet does not show a strong licensing gap in this jurisdiction."
            elif registration_supported:
                licensing_verdict = "LOW"
                licensing_reason = "Registration context exists, but the current packet does not show a clear incremental licensing unlock."
            elif clinical_linked:
                licensing_verdict = "MEDIUM"
                licensing_reason = "Clinical maturity exists, but jurisdiction-specific registration/access gap would still need active business development work."
            else:
                licensing_verdict = "NOT_EVIDENCED"
                licensing_reason = "Licensing opportunity is not evidenced in the current packet for this jurisdiction."
            regional_licensing_opportunity[region] = {
                "verdict": licensing_verdict,
                "reason": licensing_reason,
                "evidence_refs": generic_refs[:8],
            }

        synthesis_steps = selected_sections.get("synthesis_steps", []) or []
        synthesis_screening = {
            "step_count": len(synthesis_steps),
            "route_found": bool(synthesis_steps),
            "corroboration_level": (
                "partial"
                if (base_packet or {}).get("partial_route_corroboration")
                else "supported"
                if len(synthesis_steps) >= 2
                else "limited"
                if synthesis_steps
                else "missing"
            ),
            "decision_use": "technical_screening_only",
            "business_blocker": False,
            "summary": (
                "Synthesis evidence is suitable for initial technical screening, not for a manufacturing / CMC decision."
                if synthesis_steps
                else "No synthesis route evidence was selected for this packet."
            ),
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
            "ru_eaeu_ip_window_snapshot": {
                "inn": _normalize_text((base_packet or {}).get("inn") or ((base_packet or {}).get("passport") or {}).get("inn")),
                "jurisdictions": ["RU", "EAEU"],
                "as_of_date": sorted(ru_eaeu_as_of_dates)[-1] if ru_eaeu_as_of_dates else None,
                "sources_checked": ru_eaeu_sources[:10],
                "conclusion": ru_eaeu_conclusion,
                "confidence": ru_eaeu_confidence,
                "region_conclusions": ru_eaeu_region_conclusions,
                "residual_risk": ru_eaeu_conclusion in {"NO_LISTED_BLOCKING_PATENT_EVIDENCE", "PARTIAL_OPEN_WINDOW_EVIDENCE"},
                "limitations": [
                    "Official no-hit and expiry snapshots are not a full freedom-to-operate opinion.",
                    "Unlisted formulation, process, or non-pharma patents may still exist.",
                ],
                "evidence_refs": list(dict.fromkeys(ru_eaeu_evidence_refs))[:10],
            },
            "registration_identity_map": registration_identity_map[:12],
            "registration_context_relationships": registration_context_relationships,
            "market_entry_linkage": market_entry_linkage,
            "eaeu_registration": {
                "registrations": eaeu_regs[:6],
                "has_identifier_mah_linkage": any(item["identifier_mah_linked"] for item in eaeu_regs),
                "has_valid_to": any(item["validity_type"] == "date_present" for item in eaeu_regs),
                "has_validity_state": any(item["validity_type"] != "missing_in_source" for item in eaeu_regs),
                "validity_types": sorted({item["validity_type"] for item in eaeu_regs if item.get("validity_type")}),
                "has_source_native_identity_anchor": any(
                    entry.get("context") == "EAEU"
                    and entry.get("source_class") == "EAEU-native"
                    and entry.get("identity_confidence") in {"HIGH", "MEDIUM"}
                    for entry in registration_identity_map
                ),
            },
            "generic_opportunity_by_region": regional_generic_opportunity,
            "licensing_opportunity_by_region": regional_licensing_opportunity,
            "synthesis_screening": synthesis_screening,
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
                    "ru_eaeu_ip_conclusion": (contract_linkage.get("ru_eaeu_ip_window_snapshot", {}) or {}).get("conclusion"),
                    "ru_identity_match": ((contract_linkage.get("market_entry_linkage", {}) or {}).get("RU", {}) or {}).get("identity_match"),
                    "eaeu_identity_match": ((contract_linkage.get("market_entry_linkage", {}) or {}).get("EAEU", {}) or {}).get("identity_match"),
                    "eaeu_has_valid_to": contract_linkage.get("eaeu_registration", {}).get("has_valid_to", False),
                    "eaeu_has_validity_state": contract_linkage.get("eaeu_registration", {}).get("has_validity_state", False),
                    "eaeu_validity_types": list((contract_linkage.get("eaeu_registration", {}) or {}).get("validity_types", [])),
                    "eaeu_has_identifier_mah_linkage": contract_linkage.get("eaeu_registration", {}).get("has_identifier_mah_linkage", False),
                    "generic_regions_with_potential": sorted(
                        [
                            region
                            for region, payload in (contract_linkage.get("generic_opportunity_by_region", {}) or {}).items()
                            if str((payload or {}).get("verdict") or "") == "POTENTIAL_GO"
                        ]
                    ),
                },
            },
        }
