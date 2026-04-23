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
        critical_unknowns = packet.get("critical_unknowns", [])
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
                critical_unknowns = packet.get("critical_unknowns", [])
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
        if verification.overall_status != "FAIL" or not allow_repair:
            return block, verification
        repaired_block, repaired_verification = self.repair_block(block, packet, verification)
        return repaired_block, repaired_verification
