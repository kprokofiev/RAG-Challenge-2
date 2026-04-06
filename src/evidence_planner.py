"""
Evidence Planner — Sprint 22 WS2-D5
======================================
Builds EvidencePack from 3 layers:
  1. Structured dossier fields (direct extraction)
  2. Evidence-linked snippets (from evidence_registry)
  3. Supplemental chunks from corpus (via HybridRetriever)

Reuses existing retrieval.py HybridRetriever and case_view_v2_generator patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvidencePack(BaseModel):
    """Collected evidence for answering a question."""
    structured_facts: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_gaps: List[str] = Field(default_factory=list)


def _resolve_field(dossier: Dict[str, Any], field_path: str) -> Any:
    """Navigate dossier using dot-notation with [*] array support."""
    parts = field_path.split(".")
    current = dossier

    for i, part in enumerate(parts):
        if current is None:
            return None
        if part.endswith("[*]"):
            key = part[:-3]
            arr = current.get(key, [])
            if not isinstance(arr, list) or not arr:
                return None
            remaining = ".".join(parts[i + 1:])
            if remaining:
                return [_resolve_field(item, remaining) for item in arr if isinstance(item, dict)]
            return arr
        if isinstance(current, list):
            for item in current:
                if isinstance(item, dict) and item.get("region") == part:
                    current = item
                    break
            else:
                return None
            continue
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None

    return current


def _extract_evidence_refs(value: Any) -> List[str]:
    """Extract evidence_refs from EvidencedValue or nested structures."""
    refs = []
    if isinstance(value, dict):
        refs.extend(value.get("evidence_refs", []))
        if "evidence_id" in value and value["evidence_id"]:
            refs.append(value["evidence_id"])
    elif isinstance(value, list):
        for item in value:
            refs.extend(_extract_evidence_refs(item))
    return refs


class EvidencePlanner:
    """
    Plans and collects evidence for an exec question answer.

    Three layers:
    1. Structured facts from dossier fields
    2. Evidence snippets from evidence_registry
    3. Supplemental retrieval chunks (optional, via HybridRetriever)
    """

    def __init__(self, retriever=None):
        """
        Args:
            retriever: Optional HybridRetriever instance for supplemental chunk retrieval.
        """
        self.retriever = retriever

    def plan(
        self,
        routed_question,   # RoutedQuestion
        resolved_scope,    # ResolvedScope
        coverage_decision,  # CoverageDecision
        dossier: Dict[str, Any],
    ) -> EvidencePack:
        """
        Build evidence pack for a question.

        Layer 1: Extract structured facts from must_have_fields.
        Layer 2: Resolve evidence_refs from extracted values.
        Layer 3: (Optional) Supplemental retrieval if gaps exist.
        """
        pack = EvidencePack()

        # Layer 1: Structured facts from dossier fields
        for field_path in routed_question.must_have_fields:
            value = _resolve_field(dossier, field_path)
            if value is not None:
                pack.structured_facts.append({
                    "field_path": field_path,
                    "value": self._unwrap_value(value),
                    "raw": value,
                })
                # Collect evidence refs
                refs = _extract_evidence_refs(value)
                pack.evidence_refs.extend(refs)
            else:
                pack.unresolved_gaps.append(field_path)

        # Also extract from required_sections that aren't field-level
        for section in routed_question.required_sections:
            section_data = dossier.get(section)
            if section_data is not None and section not in ("unknowns", "dossier_quality_v2", "coverage_ledger"):
                if isinstance(section_data, list):
                    for item in section_data:
                        if isinstance(item, dict):
                            refs = item.get("evidence_refs", [])
                            pack.evidence_refs.extend(refs)

        # Layer 2: Resolve evidence snippets from evidence_registry
        evidence_registry = dossier.get("evidence_registry", [])
        ref_set = set(pack.evidence_refs)
        evidence_by_id = {e.get("evidence_id", ""): e for e in evidence_registry}

        for ref_id in ref_set:
            ev = evidence_by_id.get(ref_id)
            if ev:
                doc_id = ev.get("doc_id", "")
                if doc_id and doc_id not in pack.source_ids:
                    pack.source_ids.append(doc_id)

        # Deduplicate evidence refs
        pack.evidence_refs = list(ref_set)

        # Layer 3: Supplemental chunks (only if retriever available and gaps exist)
        if self.retriever and pack.unresolved_gaps and not coverage_decision.should_trigger_ws1:
            try:
                query = self._build_retrieval_query(routed_question, pack.unresolved_gaps)
                candidates = self.retriever.retrieve_by_case(
                    query=query,
                    dense_k=20,
                    sparse_k=20,
                    final_candidates_k=10,
                )
                for c in candidates:
                    pack.chunks.append({
                        "doc_id": getattr(c, "doc_id", ""),
                        "snippet": getattr(c, "text", str(c))[:500],
                        "score": getattr(c, "score", 0.0),
                        "doc_kind": getattr(c, "metadata", {}).get("doc_kind", ""),
                    })
            except Exception as e:
                logger.warning("Supplemental retrieval failed: %s", e)

        return pack

    @staticmethod
    def _unwrap_value(value: Any) -> Any:
        """Unwrap EvidencedValue to plain value."""
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        if isinstance(value, list):
            return [
                item.get("value", item) if isinstance(item, dict) and "value" in item else item
                for item in value
            ]
        return value

    @staticmethod
    def _build_retrieval_query(routed_question, gaps: List[str]) -> str:
        """Build a retrieval query string from question + gaps."""
        parts = [routed_question.question_text]
        for g in gaps[:5]:
            parts.append(g.replace("[*]", "").replace(".", " "))
        return " ".join(parts)
