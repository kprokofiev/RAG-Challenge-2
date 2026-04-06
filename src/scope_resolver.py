"""
Scope Resolver — Sprint 22 WS2-D3
====================================
Rules-first scope resolver. Deterministically selects which product_contexts
from the dossier are in scope for a given question.

Prevents mixing INN-level truth, brand-level truth, and regional product
contexts in a single answer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResolvedScope(BaseModel):
    """Output of scope resolution — which contexts are in play."""
    entity_mode: str  # inn | product_context | brand
    selected_context_ids: List[str] = Field(default_factory=list)
    excluded_context_ids: List[str] = Field(default_factory=list)
    jurisdiction_map: Dict[str, str] = Field(default_factory=dict)
    scope_warnings: List[str] = Field(default_factory=list)
    reason: str = ""


def _extract_contexts(dossier: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract product_contexts from dossier."""
    return dossier.get("product_contexts", [])


def _extract_registrations(dossier: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract registrations from dossier."""
    return dossier.get("registrations", [])


class ScopeResolver:
    """
    Rules-first scope resolver for WS2 exec Q&A.

    Rules:
    1. If question requires 4-geo → select all contexts matching required jurisdictions.
    2. If question is single-region → only that region's context(s).
    3. If question is brand-specific → filter by brand.
    4. If question is INN-level → entity_mode='inn', all contexts, but warn if mixed.
    """

    def resolve(
        self,
        routed_question,  # RoutedQuestion from question_router
        dossier: Dict[str, Any],
    ) -> ResolvedScope:
        """
        Resolve scope for a routed question against a dossier.
        """
        contexts = _extract_contexts(dossier)
        registrations = _extract_registrations(dossier)
        required_jurisdictions = set(routed_question.required_jurisdictions)

        # If no product_contexts, fall back to INN-level
        if not contexts:
            return self._inn_level_scope(registrations, required_jurisdictions)

        # Determine entity mode based on question type
        q_type = routed_question.question_type

        # Registration / regulatory questions → multi-context, jurisdiction-filtered
        if q_type in ("registration_status", "ip_fto_risk", "commercial_assessment"):
            return self._multi_jurisdiction_scope(
                contexts, registrations, required_jurisdictions, q_type
            )

        # Clinical / chemistry / synthesis → INN-level (all contexts)
        if q_type in ("clinical_evidence", "chemistry_identity", "synthesis_manufacturing", "data_quality"):
            return self._inn_level_scope_with_contexts(contexts, required_jurisdictions)

        # Default: INN-level
        return self._inn_level_scope_with_contexts(contexts, required_jurisdictions)

    def _multi_jurisdiction_scope(
        self,
        contexts: List[Dict],
        registrations: List[Dict],
        required_jurisdictions: set,
        q_type: str,
    ) -> ResolvedScope:
        """Select contexts by jurisdiction match."""
        selected = []
        excluded = []
        jurisdiction_map = {}
        warnings = []

        # Map contexts by region
        for ctx in contexts:
            ctx_id = ctx.get("context_id", "")
            region = ctx.get("region", "")

            if not required_jurisdictions or region in required_jurisdictions:
                selected.append(ctx_id)
                jurisdiction_map[region] = ctx_id
            else:
                excluded.append(ctx_id)

        # Also check registrations for covered jurisdictions (may not have contexts)
        covered_regions = set(jurisdiction_map.keys())
        for reg in registrations:
            region = reg.get("region", "")
            if region in required_jurisdictions and region not in covered_regions:
                # Registration exists but no product_context — add warning
                warnings.append(
                    f"Registration for {region} exists but no product_context found"
                )

        # Check for missing required jurisdictions
        missing = required_jurisdictions - covered_regions
        if missing:
            for r in registrations:
                region = r.get("region", "")
                if region in missing:
                    missing.discard(region)
            if missing:
                warnings.append(
                    f"No data for required jurisdictions: {', '.join(sorted(missing))}"
                )

        entity_mode = "product_context" if len(selected) > 1 else "product_context"

        # Warn if mixed context strengths
        strengths = set()
        for ctx in contexts:
            if ctx.get("context_id") in selected:
                s = ctx.get("context_strength", "unknown")
                strengths.add(s)
        if len(strengths) > 1:
            warnings.append(
                f"Mixed context strengths in scope: {', '.join(sorted(strengths))}"
            )

        return ResolvedScope(
            entity_mode=entity_mode,
            selected_context_ids=selected,
            excluded_context_ids=excluded,
            jurisdiction_map=jurisdiction_map,
            scope_warnings=warnings,
            reason=f"Multi-jurisdiction scope for {q_type}, {len(selected)} contexts selected",
        )

    def _inn_level_scope(
        self,
        registrations: List[Dict],
        required_jurisdictions: set,
    ) -> ResolvedScope:
        """INN-level scope when no product_contexts exist."""
        jurisdiction_map = {}
        warnings = []

        for reg in registrations:
            region = reg.get("region", "")
            jurisdiction_map[region] = f"registration_{region}"

        missing = required_jurisdictions - set(jurisdiction_map.keys())
        if missing:
            warnings.append(
                f"No registration data for: {', '.join(sorted(missing))}"
            )

        return ResolvedScope(
            entity_mode="inn",
            selected_context_ids=[],
            excluded_context_ids=[],
            jurisdiction_map=jurisdiction_map,
            scope_warnings=warnings,
            reason="INN-level scope (no product_contexts in dossier)",
        )

    def _inn_level_scope_with_contexts(
        self,
        contexts: List[Dict],
        required_jurisdictions: set,
    ) -> ResolvedScope:
        """INN-level scope but include all contexts for evidence."""
        all_ids = [ctx.get("context_id", "") for ctx in contexts]
        jurisdiction_map = {}
        for ctx in contexts:
            r = ctx.get("region", "")
            if r:
                jurisdiction_map[r] = ctx.get("context_id", "")

        warnings = []
        if len(contexts) > 3:
            warnings.append(
                f"INN-level query spans {len(contexts)} product contexts — "
                f"answer may aggregate across different formulations"
            )

        return ResolvedScope(
            entity_mode="inn",
            selected_context_ids=all_ids,
            excluded_context_ids=[],
            jurisdiction_map=jurisdiction_map,
            scope_warnings=warnings,
            reason=f"INN-level scope with {len(all_ids)} contexts",
        )
