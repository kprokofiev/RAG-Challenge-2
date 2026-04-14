"""
Deterministic evidence packet assembly for question-first exec reasoning.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
    "patent_family_summary": "Patent family summary",
    "patent_legal_events": "Patent legal events",
    "patent_expiry_us": "US patent expiry",
    "formulary": "Formulary source",
    "pricing": "Pricing source",
    "payer_policy": "Payer policy",
}


def _source_label(item: Dict[str, Any]) -> str:
    doc_kind = str(item.get("doc_kind") or "").strip().lower()
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
            "phase",
            "category",
            "mah",
            "identifiers",
            "legal_status_snapshot",
            "expiry_by_country",
            "description",
            "evidence_refs",
        ):
            if key in value:
                keep[key] = value[key]
        return keep or value
    return value


class ExecEvidenceAssembler:
    def __init__(self, retriever: Any = None):
        self.retriever = retriever

    def _selected_sections(
        self,
        base_packet: Dict[str, Any],
        plan: ExecQuestionPlan,
    ) -> Dict[str, Any]:
        selected: Dict[str, Any] = {}
        for section in plan.needed_dossier_sections:
            if section not in base_packet:
                continue
            value = base_packet.get(section)
            if isinstance(value, list):
                selected[section] = [_compact_value(item) for item in value[:6]]
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
                str(item.doc_kind): max(1, int(item.max_chunks))
                for item in (retrieval.doc_kind_limits or [])
                if str(item.doc_kind or "").strip()
            },
        }

    def _select_existing_evidence(
        self,
        base_packet: Dict[str, Any],
        plan: ExecQuestionPlan,
    ) -> List[Dict[str, Any]]:
        limits = self._contract_limits(plan)
        allowed_doc_kinds = {
            str(kind).strip()
            for kind in (plan.retrieval_plan.doc_kinds or base_packet.get("allowed_doc_kinds", []))
            if str(kind).strip()
        }
        per_kind_limit = limits["max_per_doc_kind"]
        counts = Counter()
        selected: List[Dict[str, Any]] = []
        for item in base_packet.get("evidence_registry", []) or []:
            doc_kind = str(item.get("doc_kind") or "").strip()
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
        plan: ExecQuestionPlan,
        case_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not self.retriever:
            return []
        limits = self._contract_limits(plan)
        selected: List[Dict[str, Any]] = []
        allowed_doc_kinds = list(plan.retrieval_plan.doc_kinds or [])
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
                doc_kind = str(normalized.get("doc_kind") or "").strip()
                if allowed_doc_kinds and doc_kind not in allowed_doc_kinds:
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
            doc_kind = str(item.get("doc_kind") or "").strip() or "unknown"
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

    def assemble(
        self,
        base_packet: Dict[str, Any],
        plan: ExecQuestionPlan,
        case_id: Optional[str] = None,
        allow_retrieval: bool = True,
    ) -> Dict[str, Any]:
        selected_sections = self._selected_sections(base_packet, plan)
        selected_evidence = self._select_existing_evidence(base_packet, plan)
        retrieved_evidence = self._retrieve_additional_evidence(plan, case_id) if allow_retrieval else []
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
            },
        }
