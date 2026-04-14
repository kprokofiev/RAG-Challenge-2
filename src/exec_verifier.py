"""
Verification and bounded repair for exec decision blocks.
"""

from __future__ import annotations

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


def _severity_rank(confidence: str) -> int:
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(str(confidence or "").upper(), 1)


class ExecVerifier:
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
                        message="Hard evidence backed claim is missing evidence refs.",
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
            for claim in repaired.why_this_verdict:
                if claim.claim_type == "hard_evidence_backed" and not claim.evidence_refs:
                    claim.claim_type = "inference"
                    applied_changes.append("downgraded_unreferenced_hard_claim")

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
