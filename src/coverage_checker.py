"""
Coverage Checker — Sprint 22 WS2-D4
======================================
Deterministic checker that verifies whether the dossier has enough data
to answer a given question honestly. No LLM involved.

Checks: required sections, required fields, critical unknowns, source gaps.
Decides: can_answer / answer_mode / should_trigger_ws1.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CoverageDecision(BaseModel):
    """Output of coverage check — can we answer, and how."""
    can_answer: bool
    answer_mode: str  # direct | direct_with_warnings | ws1_backfill_required | insufficient
    missing_fields: List[str] = Field(default_factory=list)
    critical_unknowns: List[Dict[str, Any]] = Field(default_factory=list)
    should_trigger_ws1: bool = False
    warnings: List[str] = Field(default_factory=list)


def _resolve_field(dossier: Dict[str, Any], field_path: str) -> Any:
    """
    Navigate dossier dict using dot-notation.
    Handles array patterns like registrations[*].status and clinical_studies[*].phase.
    """
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
                values = []
                for item in arr:
                    if isinstance(item, dict):
                        v = _resolve_field(item, remaining)
                        if v is not None:
                            values.append(v)
                return values if values else None
            return arr

        if isinstance(current, list):
            # Try to navigate list items (registrations list)
            for item in current:
                if isinstance(item, dict):
                    if item.get("region") == part:
                        current = item
                        break
            else:
                return None
            continue

        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None

    # Unwrap EvidencedValue
    if isinstance(current, dict) and "value" in current:
        return current.get("value")
    return current


def _has_value(v: Any) -> bool:
    """Check if a resolved value is non-empty."""
    if v is None:
        return False
    if isinstance(v, list):
        return len(v) > 0
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, dict):
        return bool(v)
    return True


class CoverageChecker:
    """
    Deterministic coverage checker for WS2 exec Q&A pipeline.

    Given a RoutedQuestion + ResolvedScope + dossier, determines whether
    the system can answer honestly or needs WS1 backfill first.
    """

    def check(
        self,
        routed_question,  # RoutedQuestion
        resolved_scope,   # ResolvedScope
        dossier: Dict[str, Any],
        coverage_ledger: Optional[Dict[str, Any]] = None,
    ) -> CoverageDecision:
        """
        Check coverage for a routed question.
        """
        missing_fields = []
        warnings = list(resolved_scope.scope_warnings)  # inherit scope warnings
        critical_unknowns = []

        # 1. Check required sections exist in dossier
        for section in routed_question.required_sections:
            section_data = dossier.get(section)
            if section_data is None:
                warnings.append(f"Required section '{section}' not present in dossier")
            elif isinstance(section_data, list) and len(section_data) == 0:
                warnings.append(f"Required section '{section}' is empty")

        # 2. Check must_have_fields
        for field_path in routed_question.must_have_fields:
            value = _resolve_field(dossier, field_path)
            if not _has_value(value):
                missing_fields.append(field_path)

        # 3. Check critical unknowns
        unknowns = dossier.get("unknowns", [])
        critical_reason_codes = {
            "NO_DOCUMENT_IN_CORPUS",
            "LEGAL_STATUS_NOT_AVAILABLE",
            "EXTRACTION_FAILED",
            "EXTERNAL_SERVICE_UNAVAILABLE",
        }
        for u in unknowns:
            if u.get("reason_code") in critical_reason_codes:
                # Check if this unknown is relevant to required fields
                uf = u.get("field_path", "")
                for mf in routed_question.must_have_fields:
                    # Match: unknowns about registrations.* match registrations[*].*
                    if self._field_matches(uf, mf):
                        critical_unknowns.append(u)
                        break

        # 4. Check coverage ledger for source gaps
        should_ws1 = False
        if coverage_ledger:
            section_coverage = coverage_ledger.get("section_coverage", {})
            for section in routed_question.required_sections:
                sc = section_coverage.get(section, {})
                readiness = sc.get("readiness", "insufficient")
                if readiness == "insufficient":
                    warnings.append(
                        f"Section '{section}' readiness is 'insufficient' per coverage ledger"
                    )

        # 5. Determine answer_mode
        total_required = len(routed_question.must_have_fields)
        total_missing = len(missing_fields)
        confidence_req = routed_question.required_confidence
        fallback_policy = routed_question.fallback_policy

        if total_missing == 0 and not critical_unknowns:
            answer_mode = "direct"
            can_answer = True
        elif total_missing == 0 and critical_unknowns:
            answer_mode = "direct_with_warnings"
            can_answer = True
        elif total_required > 0 and total_missing / total_required <= 0.3:
            # Less than 30% missing — can answer with warnings
            answer_mode = "direct_with_warnings"
            can_answer = True
            if confidence_req == "high" and critical_unknowns:
                should_ws1 = True
                answer_mode = "ws1_backfill_required"
        elif fallback_policy == "allow_answer_with_unknowns":
            answer_mode = "direct_with_warnings"
            can_answer = True
            if total_missing / max(total_required, 1) > 0.5:
                should_ws1 = True
        elif fallback_policy in ("require_warning_if_incomplete", "require_warning_if_legal_incomplete"):
            if total_missing / max(total_required, 1) > 0.5:
                answer_mode = "ws1_backfill_required"
                should_ws1 = True
                can_answer = False
            else:
                answer_mode = "direct_with_warnings"
                can_answer = True
        else:
            answer_mode = "insufficient"
            can_answer = False
            should_ws1 = True

        return CoverageDecision(
            can_answer=can_answer,
            answer_mode=answer_mode,
            missing_fields=missing_fields,
            critical_unknowns=critical_unknowns,
            should_trigger_ws1=should_ws1,
            warnings=warnings,
        )

    @staticmethod
    def _field_matches(unknown_path: str, required_path: str) -> bool:
        """Check if an unknown field_path matches a required must_have_field pattern."""
        # Normalize array wildcards for comparison
        # unknown: "registrations.EU.status" should match "registrations[*].status"
        u_parts = unknown_path.split(".")
        r_parts = required_path.split(".")

        if len(u_parts) != len(r_parts):
            # Try dropping the region qualifier in unknown path
            # registrations.EU.status (3 parts) vs registrations[*].status (2 semantic parts)
            # After removing [*]: registrations.status
            r_clean = required_path.replace("[*].", ".")
            u_clean = re.sub(r"\.(US|EU|RU|EAEU)\.", ".", unknown_path)
            return u_clean == r_clean or unknown_path.startswith(r_clean.split("[*]")[0])

        for u, r in zip(u_parts, r_parts):
            if r.endswith("[*]"):
                r_base = r[:-3]
                if u != r_base and not u.startswith(r_base):
                    return False
            elif u != r:
                return False
        return True
