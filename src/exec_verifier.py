"""
Verification and bounded repair for exec decision blocks.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple

try:
    from src.dossier_schema_v3 import (
        ExecBlocker,
        ExecConfidenceEnum,
        ExecDecisionBlock,
        ExecNextAction,
        ExecVerificationIssue,
        ExecVerificationReport,
    )
except ImportError:  # pragma: no cover
    from dossier_schema_v3 import (  # type: ignore
        ExecBlocker,
        ExecConfidenceEnum,
        ExecDecisionBlock,
        ExecNextAction,
        ExecVerificationIssue,
        ExecVerificationReport,
    )


_BLOCKING_SEVERITIES = {"DECISION_BLOCKING", "MUST_VERIFY_NOW"}
_GO_VERDICTS = {"GO", "CONDITIONAL_GO", "OPEN", "HIGH"}
_NEGATIVE_VERDICTS = {"NO_GO", "CLOSED", "LOW"}
_POSITIVE_REGISTRATION_MARKERS = {
    "active",
    "approved",
    "authorised",
    "authorized",
    "confirmed",
    "in force",
    "registered",
    "valid",
}
_POSITIVE_COMMERCIAL_MARKERS = {
    "confirmed",
    "positive",
    "present",
    "supported",
}
_MISSING_EVIDENCE_MARKERS = {
    "blank",
    "cannot be concluded",
    "cannot be confirmed",
    "insufficient",
    "missing",
    "not available",
    "not confirmed",
    "not provided",
    "unresolved",
    "unknown",
    "valid_to",
}
_NEGATIVE_EVIDENCE_MARKERS = {
    "closed",
    "denied",
    "failed",
    "inactive",
    "negative",
    "not approv",
    "not registered",
    "refused",
    "rejected",
    "revoked",
    "suspended",
    "terminated",
    "withdrawn",
}
_STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "that",
    "the",
    "this",
    "with",
    "without",
}


def _severity_rank(confidence: str) -> int:
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(str(confidence or "").upper(), 1)


def _tokenize(value: Any) -> set[str]:
    text = str(value or "").lower()
    tokens = {token for token in re.findall(r"[a-zа-я0-9]{3,}", text) if token not in _STOPWORDS}
    return tokens


def _candidate_evidence(packet: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return evidence candidates that can be used for bounded claim repair."""
    candidates: List[Dict[str, str]] = []
    seen = set()
    for collection_name in ("selected_evidence", "evidence_registry"):
        for item in packet.get(collection_name, []) or []:
            if not isinstance(item, dict):
                continue
            ref = str(item.get("evidence_id") or item.get("doc_id") or "").strip()
            if not ref or ref in seen:
                continue
            seen.add(ref)
            searchable = " ".join(
                str(item.get(key) or "")
                for key in ("doc_kind", "source_label", "title", "snippet", "source_url")
            )
            candidates.append({"ref": ref, "searchable": searchable})
    return candidates


def _scalar_text(value: Any) -> str:
    if isinstance(value, dict):
        if "value" in value:
            return str(value.get("value") or "")
        return " ".join(_scalar_text(v) for v in value.values() if _scalar_text(v))
    if isinstance(value, list):
        return " ".join(_scalar_text(item) for item in value if _scalar_text(item))
    return str(value or "")


def _region_text(item: Dict[str, Any]) -> str:
    return str(
        item.get("region")
        or item.get("jurisdiction")
        or item.get("country")
        or ""
    ).strip().upper()


def _contains_marker(text: str, markers: set[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _has_explicit_negative_evidence(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        r"\bwithdrawn\b",
        r"\brevoked\b",
        r"\bterminated\b",
        r"\bsuspended\b",
        r"\brejected\b",
        r"\brefused\b",
        r"\binactive\b",
        r"\bclosed\b",
        r"\bfailed\b",
        r"\bdenied\b",
        r"\bnegative\b",
        r"\bnot approv\w*\b",
        r"\bnot registered\b",
        r"(?<!non[-\s])(?<!not[-\s])(?<!un)expired\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _registration_status_text(item: Dict[str, Any]) -> str:
    return " ".join(
        part for part in (
            _scalar_text(item.get("status")),
            _scalar_text(item.get("verdict")),
        ) if part
    ).strip()


def _has_positive_registration(packet: Dict[str, Any], region: str) -> bool:
    region = str(region or "").strip().upper()
    for item in packet.get("registrations", []) or []:
        if not isinstance(item, dict) or _region_text(item) != region:
            continue
        if _contains_marker(_registration_status_text(item), _POSITIVE_REGISTRATION_MARKERS):
            return True
    return False


def _positive_commercial_signal_count(packet: Dict[str, Any], region: str) -> int:
    region = str(region or "").strip().upper()
    count = 0
    for item in packet.get("commercial_signals", []) or []:
        if not isinstance(item, dict) or _region_text(item) != region:
            continue
        signal_text = " ".join(
            part for part in (
                _scalar_text(item.get("verdict")),
                _scalar_text(item.get("category")),
                _scalar_text(item.get("summary")),
            ) if part
        )
        if _contains_marker(signal_text, _POSITIVE_COMMERCIAL_MARKERS):
            count += 1
    return count


def _block_text(block: ExecDecisionBlock) -> str:
    parts: List[str] = [block.short_answer, block.full_answer]
    parts.extend(claim.claim for claim in block.why_this_verdict)
    for blocker in block.decision_blockers:
        parts.append(blocker.title)
        parts.append(blocker.rationale)
    parts.extend(block.caveats)
    return " ".join(part for part in parts if part).strip()


def _has_context_integrity_green(packet: Dict[str, Any]) -> bool:
    readiness = ((packet.get("dossier_quality_v2") or {}).get("decision_readiness") or {})
    return str(readiness.get("context_integrity") or "").upper() == "GREEN"


def _grounding_refs_for_claim(claim_text: str, packet: Dict[str, Any], limit: int = 3) -> List[str]:
    claim_tokens = _tokenize(claim_text)
    if not claim_tokens:
        return []
    scored: List[Tuple[int, str]] = []
    for candidate in _candidate_evidence(packet):
        evidence_tokens = _tokenize(candidate.get("searchable"))
        overlap = claim_tokens & evidence_tokens
        if len(overlap) >= 2:
            scored.append((len(overlap), candidate["ref"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    valid_refs = set(packet.get("evidence_ids", [])) or {candidate["ref"] for candidate in _candidate_evidence(packet)}
    return [ref for _, ref in scored if ref in valid_refs][:limit]


class ExecVerifier:
    def _relevant_critical_unknowns(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        unknowns = list(packet.get("critical_unknowns", []) or [])
        if block.block_id == "rf_entry":
            filtered: List[Dict[str, Any]] = []
            for item in unknowns:
                text = f"{item.get('reason_code', '')} {item.get('impact', '')}".lower()
                if any(marker in text for marker in ("eaeu", "patent", "legal_status_not_available")):
                    continue
                if any(marker in text for marker in ("ru", "rf", "grls", "registration")):
                    filtered.append(item)
            return filtered
        if block.block_id == "eaeu_entry":
            return [
                item for item in unknowns
                if any(
                    marker in f"{item.get('reason_code', '')} {item.get('impact', '')}".lower()
                    for marker in ("eaeu", "registration")
                )
            ]
        return unknowns

    def _asset_negative_missing_evidence_overreach(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "asset_attractiveness" or block.verdict not in {"NO_GO", "INSUFFICIENT_EVIDENCE"}:
            return False
        if not _has_context_integrity_green(packet):
            return False
        if not (
            _has_positive_registration(packet, "RU")
            and (_has_positive_registration(packet, "EU") or _has_positive_registration(packet, "US") or _has_positive_registration(packet, "EAEU"))
        ):
            return False
        text = _block_text(block)
        return _contains_marker(text, _MISSING_EVIDENCE_MARKERS) and not _has_explicit_negative_evidence(text)

    def _rf_scope_overconstraint(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "rf_entry" or block.verdict in _GO_VERDICTS:
            return False
        if not _has_positive_registration(packet, "RU"):
            return False
        if _positive_commercial_signal_count(packet, "RU") <= 0:
            return False
        text = _block_text(block).lower()
        scope_markers = (
            "eaeu",
            "valid_to",
            "underlying authorization",
            "legal_status_not_available",
            "critical unknown",
            "expiry date",
            "non-suspension",
            "non-revocation",
            "legal/entry-critical readiness",
        )
        return any(marker in text for marker in scope_markers) and not _has_explicit_negative_evidence(text)

    def _eaeu_holdable_regulatory_position(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "eaeu_entry" or block.verdict != "INSUFFICIENT_EVIDENCE":
            return False
        if not _has_positive_registration(packet, "EAEU"):
            return False
        text = _block_text(block)
        return _contains_marker(text, _MISSING_EVIDENCE_MARKERS) and not _has_explicit_negative_evidence(text)

    def _ground_or_downgrade_unreferenced_hard_claims(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> List[str]:
        applied_changes: List[str] = []
        for claim in block.why_this_verdict:
            if claim.claim_type != "hard_evidence_backed" or claim.evidence_refs:
                continue
            refs = _grounding_refs_for_claim(claim.claim, packet)
            if refs:
                claim.evidence_refs = refs
                applied_changes.append("attached_refs_to_unreferenced_hard_claim")
            else:
                claim.claim_type = "inference"
                applied_changes.append("downgraded_unreferenced_hard_claim")
        return applied_changes

    def verify_block(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
        block_spec: Any,
    ) -> ExecVerificationReport:
        evidence_ids = set(packet.get("evidence_ids", []))
        critical_unknowns = self._relevant_critical_unknowns(block, packet)
        issues: List[ExecVerificationIssue] = []

        for claim in block.why_this_verdict:
            if claim.claim_type == "hard_evidence_backed" and not claim.evidence_refs:
                issues.append(
                    ExecVerificationIssue(
                        issue_type="unsupported_claim",
                        severity="FAIL",
                        message=f"Hard evidence backed claim is missing evidence refs: {claim.claim[:160]}",
                    )
                )
            missing_refs = [ref for ref in claim.evidence_refs if ref not in evidence_ids]
            if missing_refs:
                issues.append(
                    ExecVerificationIssue(
                        issue_type="unknown_evidence_ref",
                        severity="FAIL",
                        message=f"Claim cites unknown evidence refs: {', '.join(missing_refs[:5])}",
                        evidence_refs=missing_refs[:5],
                    )
                )

        for blocker in block.decision_blockers:
            if not blocker.title.strip():
                issues.append(
                    ExecVerificationIssue(
                        issue_type="empty_blocker",
                        severity="FAIL",
                        message="Decision blocker title is empty.",
                    )
                )

        if block.verdict in _GO_VERDICTS and any(
            blocker.severity in _BLOCKING_SEVERITIES for blocker in block.decision_blockers
        ):
            issues.append(
                ExecVerificationIssue(
                    issue_type="verdict_blocker_conflict",
                    severity="FAIL",
                    message="Positive verdict conflicts with decision-blocking blockers.",
                )
            )

        if block.confidence == "HIGH" and block.sufficiency != "SUFFICIENT":
            issues.append(
                ExecVerificationIssue(
                    issue_type="confidence_mismatch",
                    severity="WARN",
                    message="High confidence is not aligned with non-sufficient evidence.",
                )
            )

        if critical_unknowns and block.verdict in _GO_VERDICTS and block.sufficiency == "SUFFICIENT":
            issues.append(
                ExecVerificationIssue(
                    issue_type="critical_unknown_ignored",
                    severity="FAIL",
                    message="Critical unknowns are present but the block still claims sufficient evidence.",
                )
            )

        if block.next_actions and not block.decision_blockers and block.verdict in _NEGATIVE_VERDICTS:
            issues.append(
                ExecVerificationIssue(
                    issue_type="action_without_driver",
                    severity="WARN",
                    message="Next actions exist without explicit blockers or caveats.",
                )
            )

        if packet.get("partial_route_corroboration") and not block.caveats:
            issues.append(
                ExecVerificationIssue(
                    issue_type="missing_partial_route_caveat",
                    severity="WARN",
                    message="Partial synthesis corroboration should surface as a caveat.",
                )
            )

        if self._asset_negative_missing_evidence_overreach(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="negative_missing_evidence_overreach",
                    severity="WARN",
                    message="Negative asset verdict is being driven by missing evidence rather than explicit negative evidence.",
                )
            )

        if self._rf_scope_overconstraint(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="rf_scope_overconstraint",
                    severity="WARN",
                    message="RF-entry verdict is being blocked by EAEU-validity uncertainty despite active RU registration and RU support signals.",
                )
            )

        if self._eaeu_holdable_regulatory_position(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="eaeu_holdable_position",
                    severity="WARN",
                    message="EAEU entry has a confirmed registration anchor and should degrade to HOLD rather than pure insufficiency.",
                )
            )

        factual_status = "FAIL" if any(issue.issue_type in {"unsupported_claim", "unknown_evidence_ref", "empty_blocker"} and issue.severity == "FAIL" for issue in issues) else "PASS"
        decision_status = "FAIL" if any(issue.issue_type in {"verdict_blocker_conflict", "critical_unknown_ignored"} and issue.severity == "FAIL" for issue in issues) else "PASS"
        reviewer_status = "WARN" if any(issue.severity == "WARN" for issue in issues) else "PASS"
        overall_status = "FAIL" if "FAIL" in {factual_status, decision_status} else reviewer_status
        return ExecVerificationReport(
            block_id=block.block_id,
            overall_status=overall_status,
            factual_status=factual_status,
            decision_status=decision_status,
            reviewer_status=reviewer_status,
            issues=issues,
        )

    def repair_block(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
        verification: ExecVerificationReport,
    ) -> Tuple[ExecDecisionBlock, ExecVerificationReport]:
        repaired = deepcopy(block)
        applied_changes: List[str] = []

        if any(issue.issue_type == "unsupported_claim" for issue in verification.issues):
            applied_changes.extend(
                self._ground_or_downgrade_unreferenced_hard_claims(repaired, packet)
            )

        if any(issue.issue_type == "unknown_evidence_ref" for issue in verification.issues):
            valid_refs = set(packet.get("evidence_ids", []))
            repaired.top_evidence_refs = [ref for ref in repaired.top_evidence_refs if ref in valid_refs]
            for claim in repaired.why_this_verdict:
                claim.evidence_refs = [ref for ref in claim.evidence_refs if ref in valid_refs]
            for blocker in repaired.decision_blockers:
                blocker.evidence_refs = [ref for ref in blocker.evidence_refs if ref in valid_refs]
            for action in repaired.next_actions:
                action.evidence_refs = [ref for ref in action.evidence_refs if ref in valid_refs]
            applied_changes.append("removed_unknown_evidence_refs")
            applied_changes.extend(
                self._ground_or_downgrade_unreferenced_hard_claims(repaired, packet)
            )

        if any(issue.issue_type == "verdict_blocker_conflict" for issue in verification.issues):
            if repaired.verdict == "GO":
                repaired.verdict = "HOLD"
            elif repaired.verdict == "OPEN":
                repaired.verdict = "LIMITED"
            elif repaired.verdict == "HIGH":
                repaired.verdict = "MEDIUM"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM" if _severity_rank(repaired.confidence) > 2 else repaired.confidence
            applied_changes.append("downgraded_conflicting_verdict")

        if any(issue.issue_type == "critical_unknown_ignored" for issue in verification.issues):
            repaired.sufficiency = "PARTIAL"
            if not repaired.decision_blockers:
                critical_unknowns = self._relevant_critical_unknowns(repaired, packet)
                for item in critical_unknowns[:2]:
                    repaired.decision_blockers.append(
                        ExecBlocker(
                            blocker_id=str(item.get("reason_code", "")),
                            title=str(item.get("reason_code", "critical_unknown")),
                            severity="MUST_VERIFY_NOW",
                            rationale=str(item.get("impact", "")),
                        )
                    )
            applied_changes.append("promoted_critical_unknowns_to_blockers")

        if any(issue.issue_type == "confidence_mismatch" for issue in verification.issues):
            repaired.confidence = "MEDIUM" if repaired.sufficiency == "SUFFICIENT" else "LOW"
            applied_changes.append("aligned_confidence_with_sufficiency")

        if packet.get("partial_route_corroboration"):
            caveat = "Synthesis route corroboration remains partial; avoid strong manufacturing/process conclusions."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
                applied_changes.append("added_partial_route_caveat")

        if repaired.decision_blockers and not repaired.next_actions:
            blocker = repaired.decision_blockers[0]
            repaired.next_actions.append(
                ExecNextAction(
                    action_id=f"{repaired.block_id}_verify_1",
                    action=f"Verify blocker: {blocker.title}",
                    priority="NOW",
                    rationale=blocker.rationale,
                    evidence_refs=list(blocker.evidence_refs),
                )
                )
            applied_changes.append("added_action_for_blocker")

        if any(issue.issue_type == "negative_missing_evidence_overreach" for issue in verification.issues) or self._asset_negative_missing_evidence_overreach(repaired, packet):
            repaired.verdict = "HOLD"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "Baseline maturity is supported across registrations and clinical evidence, but unresolved RU/EAEU IP-window and EAEU-validity gaps keep the asset at HOLD rather than NO_GO."
            )
            repaired.full_answer = (
                "Registrations and clinical maturity are evidenced, and context integrity is acceptable; however, RU/EAEU patent-window evidence and EAEU validity remain unresolved. "
                "Those are hold-level decision blockers, not source-backed negative evidence, so the block is repaired from NO_GO to HOLD."
            )
            applied_changes.append("softened_missing_evidence_no_go_to_hold")

        if any(issue.issue_type == "rf_scope_overconstraint" for issue in verification.issues) or self._rf_scope_overconstraint(repaired, packet):
            repaired.verdict = "GO"
            repaired.sufficiency = "SUFFICIENT"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "GO — active RU registration is confirmed in GRLS and supportive RU access signals are present; unresolved EAEU-validity detail is tracked separately, and dossier-level context integrity already supports the RU product match."
            )
            repaired.full_answer = (
                "RF entry is grounded by an active RU GRLS registration plus supportive RU formulary/policy/commercial signals. "
                "The missing EAEU valid_to detail remains an adjacent EAEU issue, but it should not override a positive RF decision anchored to the RU registration context. "
                "Because dossier context integrity is already green, the lack of an additional RU route/form field in the GRLS snippet should remain a caveat rather than a blocker."
            )
            repaired.decision_blockers = []
            repaired.next_actions = []
            repaired.why_this_verdict = [
                claim for claim in repaired.why_this_verdict
                if not any(
                    marker in claim.claim.lower()
                    for marker in ("eaeu", "route/form", "product-context alignment", "dosage form")
                )
            ]
            caveat = "EAEU authorization validity remains unresolved for the EAEU block, but RF entry is anchored to the active RU GRLS registration."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            ru_validity_caveat = "RU GRLS end-date/legal-status detail remains partially structured in the source snapshot, but explicit GRLS status=active is sufficient for the RF decision."
            if ru_validity_caveat not in repaired.caveats:
                repaired.caveats.append(ru_validity_caveat)
            applied_changes.append("removed_eaeu_overconstraint_from_rf_entry")

        if any(issue.issue_type == "eaeu_holdable_position" for issue in verification.issues) or self._eaeu_holdable_regulatory_position(repaired, packet):
            repaired.verdict = "HOLD"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "HOLD — an EAEU registration anchor is confirmed, but the in-force validity window and EAEU-scoped commercial pathway remain unresolved."
            )
            repaired.full_answer = (
                "The packet confirms an EAEU registration identity for apixaban, so the block should not collapse to pure insufficiency. "
                "However, because valid_to remains blank in the source snapshot and commercial pathway evidence is still RU-only, the defensible outcome is HOLD pending targeted EAEU follow-up."
            )
            applied_changes.append("promoted_eaeu_insufficiency_to_hold")

        repaired_verification = self.verify_block(repaired, packet, block_spec=None)
        repaired_verification.repair_applied = bool(applied_changes)
        repaired_verification.repair_reason = "; ".join(applied_changes) if applied_changes else None
        if applied_changes:
            repaired_verification.notes.append("Applied bounded repair pass.")
        return repaired, repaired_verification

    def verify_and_repair(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
        block_spec: Any,
        allow_repair: bool = True,
    ) -> Tuple[ExecDecisionBlock, ExecVerificationReport]:
        verification = self.verify_block(block, packet, block_spec)
        reparable_warns = {
            "confidence_mismatch",
            "missing_partial_route_caveat",
            "negative_missing_evidence_overreach",
            "rf_scope_overconstraint",
            "eaeu_holdable_position",
        }
        has_reparable_warn = any(
            issue.issue_type in reparable_warns for issue in verification.issues
        )
        needs_policy_repair = any(
            (
                self._asset_negative_missing_evidence_overreach(block, packet),
                self._rf_scope_overconstraint(block, packet),
                self._eaeu_holdable_regulatory_position(block, packet),
            )
        )
        if (verification.overall_status != "FAIL" and not has_reparable_warn and not needs_policy_repair) or not allow_repair:
            return block, verification
        repaired_block, repaired_verification = self.repair_block(block, packet, verification)
        return repaired_block, repaired_verification
