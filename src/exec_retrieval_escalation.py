"""
Bounded targeted retrieval escalation for exec decision blocks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


_COMMERCIAL_SOURCE_PRIORITY = {
    "ru_registration_export": 100,
    "ru_esklp_snapshot": 95,
    "ru_official_act": 90,
    "ru_procurement_snapshot": 85,
    "ru_commercial_summary": 55,
    "ru_formulary_summary": 52,
    "ru_policy_act": 50,
    "ru_procurement_summary": 48,
    "formulary": 40,
    "pricing": 40,
    "payer_policy": 38,
}


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def escalation_enabled() -> bool:
    return (os.getenv("DDKIT_EXEC_ESCALATION_ENABLED", "true") or "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _doc_kind_priority(doc_kind: Optional[str]) -> int:
    return _COMMERCIAL_SOURCE_PRIORITY.get(str(doc_kind or "").strip().lower(), 30)


def normalize_retrieval_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        doc_id = item.get("doc_id") or item.get("id") or ""
        score = item.get("score")
        if score is None:
            score = item.get("distance")
        return {
            "doc_id": str(doc_id or ""),
            "doc_kind": str(item.get("doc_kind") or ""),
            "page": item.get("page"),
            "snippet": str(item.get("snippet") or item.get("text") or ""),
            "score": float(score or 0.0),
            "source_url": item.get("source_url"),
        }
    metadata = getattr(item, "metadata", {}) or {}
    return {
        "doc_id": str(getattr(item, "doc_id", "") or metadata.get("doc_id") or ""),
        "doc_kind": str(getattr(item, "doc_kind", "") or metadata.get("doc_kind") or ""),
        "page": getattr(item, "page", None),
        "snippet": str(getattr(item, "text", "") or getattr(item, "snippet", "")),
        "score": float(getattr(item, "score", 0.0) or getattr(item, "distance", 0.0) or 0.0),
        "source_url": metadata.get("source_url"),
    }


@dataclass
class EscalationResult:
    performed: bool = False
    retrieved_items: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_doc_ids: List[str] = field(default_factory=list)
    missing_evidence_classes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)


class ExecRetrievalEscalator:
    def __init__(self, retriever: Any = None):
        self.retriever = retriever

    def _limits(self) -> Dict[str, int]:
        return {
            "rounds": _int_env("DDKIT_EXEC_MAX_ESCALATION_ROUNDS", 1),
            "calls": _int_env("DDKIT_EXEC_MAX_EXTRA_RETRIEVE_CALLS_PER_BLOCK", 6),
            "items": _int_env("DDKIT_EXEC_MAX_EXTRA_EVIDENCE_ITEMS_PER_BLOCK", 20),
            "chunks": _int_env("DDKIT_EXEC_MAX_EXTRA_CONTEXT_CHUNKS_PER_BLOCK", 18),
        }

    def _build_queries(
        self,
        block_spec,
        packet: Dict[str, Any],
        missing_evidence_classes: Iterable[str],
    ) -> List[str]:
        inn = packet.get("inn") or packet.get("passport", {}).get("inn") or "asset"
        block_title = block_spec.title
        queries: List[str] = []
        for evidence_class in missing_evidence_classes:
            phrase = str(evidence_class or "").replace("_", " ").strip()
            if not phrase:
                continue
            queries.append(f"{inn} {block_title} {phrase}".strip())
        if not queries:
            queries.append(f"{inn} {block_title}".strip())
        return queries

    def _sort_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (
                _doc_kind_priority(item.get("doc_kind")),
                item.get("score", 0.0),
                item.get("doc_id", ""),
            ),
            reverse=True,
        )

    def escalate(
        self,
        block_spec,
        packet: Dict[str, Any],
        missing_evidence_classes: List[str],
        case_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> EscalationResult:
        limits = self._limits()
        result = EscalationResult(missing_evidence_classes=list(missing_evidence_classes))
        result.trace = {
            "enabled": escalation_enabled(),
            "limits": limits,
            "block_id": block_spec.block_id,
            "allowlist_doc_kinds": list(block_spec.allowed_doc_kinds),
            "queries": [],
        }
        if not escalation_enabled():
            result.reasons.append("escalation_disabled")
            return result
        if not self.retriever:
            result.reasons.append("retriever_unavailable")
            return result
        if not missing_evidence_classes:
            result.reasons.append("no_missing_evidence_classes")
            return result

        queries = self._build_queries(block_spec, packet, missing_evidence_classes)
        result.trace["queries"] = queries[: limits["calls"]]
        allowed_doc_kinds = list(block_spec.allowed_doc_kinds)
        collected: List[Dict[str, Any]] = []

        for query in queries[: limits["calls"]]:
            try:
                raw_items = self.retriever.retrieve_by_case(
                    query=query,
                    case_id=case_id,
                    tenant_id=tenant_id,
                    doc_kind=allowed_doc_kinds or None,
                    top_n=limits["chunks"],
                )
            except TypeError:
                raw_items = self.retriever.retrieve_by_case(
                    query=query,
                    case_id=case_id,
                    tenant_id=tenant_id,
                    doc_kind=allowed_doc_kinds or None,
                )
            except Exception as exc:
                result.reasons.append(f"retrieval_error:{exc}")
                continue
            for item in raw_items or []:
                normalized = normalize_retrieval_item(item)
                if allowed_doc_kinds and normalized["doc_kind"] not in allowed_doc_kinds:
                    continue
                collected.append(normalized)

        deduped: List[Dict[str, Any]] = []
        seen_keys = set()
        for item in self._sort_items(collected):
            dedupe_key = (
                item.get("doc_id", ""),
                item.get("page"),
                item.get("doc_kind", ""),
                item.get("snippet", "")[:160],
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped.append(item)
            if len(deduped) >= limits["items"]:
                break

        result.performed = bool(deduped)
        result.retrieved_items = deduped
        result.retrieved_doc_ids = list(dict.fromkeys(item["doc_id"] for item in deduped if item.get("doc_id")))
        if result.performed:
            result.reasons.append("targeted_retrieval_performed")
        else:
            result.reasons.append("no_additional_evidence_found")
        result.trace["retrieved_doc_ids"] = result.retrieved_doc_ids
        result.trace["retrieved_items"] = len(result.retrieved_items)
        return result
