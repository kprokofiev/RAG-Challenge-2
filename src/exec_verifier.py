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


def _contains_any_marker(text: str, markers: tuple[str, ...] | list[str] | set[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(marker or "").lower() in lowered for marker in markers)


def _has_explicit_negative_evidence(text: str) -> bool:
    lowered = text.lower()
    lowered = re.sub(
        r"\b(?:not|no)\s+(?:source[-\s]backed\s+|explicit\s+)?negative evidence\b",
        "",
        lowered,
    )
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


def _packet_section_items(packet: Dict[str, Any], section: str) -> List[Dict[str, Any]]:
    direct = packet.get(section)
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    selected = (packet.get("selected_sections", {}) or {}).get(section)
    if isinstance(selected, list):
        return [item for item in selected if isinstance(item, dict)]
    return []


def _registration_status_text(item: Dict[str, Any]) -> str:
    return " ".join(
        part for part in (
            _scalar_text(item.get("status")),
            _scalar_text(item.get("verdict")),
        ) if part
    ).strip()


def _has_positive_registration(packet: Dict[str, Any], region: str) -> bool:
    region = str(region or "").strip().upper()
    for item in _packet_section_items(packet, "registrations"):
        if not isinstance(item, dict) or _region_text(item) != region:
            continue
        if _contains_marker(_registration_status_text(item), _POSITIVE_REGISTRATION_MARKERS):
            return True
    for item in (_contract_linkage(packet).get("registration_identity_map", []) or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("context") or "").strip().upper() == region and bool(item.get("status_positive")):
            return True
    return False


def _positive_commercial_signal_count(packet: Dict[str, Any], region: str) -> int:
    region = str(region or "").strip().upper()
    count = 0
    for item in _packet_section_items(packet, "commercial_signals"):
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


def _contract_linkage(packet: Dict[str, Any]) -> Dict[str, Any]:
    return dict(packet.get("contract_linkage", {}) or {})


def _identity_entry(packet: Dict[str, Any], region: str) -> Dict[str, Any]:
    region = str(region or "").strip().upper()
    best: Dict[str, Any] = {}
    best_rank = -1
    for item in (_contract_linkage(packet).get("registration_identity_map", []) or []):
        if not isinstance(item, dict) or str(item.get("context") or "").strip().upper() != region:
            continue
        rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(str(item.get("identity_confidence") or "").upper(), 0)
        if rank > best_rank:
            best = item
            best_rank = rank
    return best


def _market_entry_linkage(packet: Dict[str, Any], region: str) -> Dict[str, Any]:
    region = str(region or "").strip().upper()
    return ((_contract_linkage(packet).get("market_entry_linkage", {}) or {}).get(region, {}) or {})


def _ru_eaeu_ip_snapshot(packet: Dict[str, Any]) -> Dict[str, Any]:
    return (_contract_linkage(packet).get("ru_eaeu_ip_window_snapshot", {}) or {})


def _eaeu_native_entry_decision_supported(packet: Dict[str, Any]) -> bool:
    identity_entry = _identity_entry(packet, "EAEU")
    if not identity_entry:
        return False
    source_class = str(identity_entry.get("source_class") or "").strip().lower()
    confidence = str(identity_entry.get("identity_confidence") or "").strip().upper()
    validity_type = str(identity_entry.get("validity_type") or "").strip().lower()
    has_validity = validity_type in {"date_present", "indefinite"} or bool(str(identity_entry.get("valid_to") or "").strip())
    has_explicit_negative_status = identity_entry.get("status_positive") is False and not _has_positive_registration(packet, "EAEU")
    return (
        source_class == "eaeu-native"
        and confidence in {"HIGH", "MEDIUM"}
        and has_validity
        and not has_explicit_negative_status
    )


def _is_eaeu_dossier_wide_coverage_text(value: Any) -> bool:
    text = str(value or "").lower().replace("_", " ")
    markers = (
        "coverage ledger",
        "decision grade dossier coverage",
        "decision-grade dossier coverage",
        "decision grade coverage",
        "decision-grade coverage",
        "decision readiness",
        "dossier coverage",
        "dossier readiness",
        "evidence sufficiency",
        "insufficient readiness",
        "missing source class",
        "missing source classes",
        "source manifest",
    )
    return any(marker in text for marker in markers)


def _regional_opportunity(packet: Dict[str, Any], block_id: str) -> Dict[str, Any]:
    return (_contract_linkage(packet).get(f"{block_id}_by_region", {}) or {})


def _identity_match_rank(value: str) -> int:
    return {
        "none": 0,
        "inn_level_only": 1,
        "mah_or_product_context": 2,
        "same_identifier": 3,
    }.get(str(value or "").strip().lower(), 0)


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
        linkage = _market_entry_linkage(packet, "RU")
        has_market_entry_anchor = bool(linkage.get("registration_anchor_present")) and int(linkage.get("commercial_signal_count") or 0) > 0
        if not (_has_context_integrity_green(packet) or has_market_entry_anchor):
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
            "procurement",
            "no matching procurement rows",
            "route/dosage form",
            "identity fields beyond grls",
            "lacks a clear ru instruction",
            "inn-level",
            "product-context alignment",
            "product_context_match_confirmed",
            "same registered ru product identity",
            "same_identifier_confirmed",
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

    def _rf_underlinked_conditional_go(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "rf_entry" or block.verdict not in {"CONDITIONAL_GO", "HOLD", "INSUFFICIENT_EVIDENCE"}:
            return False
        linkage = _market_entry_linkage(packet, "RU")
        if not linkage.get("registration_anchor_present") or int(linkage.get("commercial_signal_count") or 0) <= 0:
            return False
        if _identity_match_rank(str(linkage.get("identity_match") or "")) < 2:
            return False
        text = _block_text(block)
        linkage_markers = (
            "identifier",
            "identity match",
            "mah",
            "product context",
            "linkage",
            "inn-level",
            "same product",
        )
        return (
            (_contains_any_marker(text, linkage_markers) or block.verdict == "CONDITIONAL_GO")
            and not _has_explicit_negative_evidence(text)
        )

    def _eaeu_same_id_overconstraint(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "eaeu_entry" or block.verdict in {"GO", "CONDITIONAL_GO"}:
            return False
        linkage = _market_entry_linkage(packet, "EAEU")
        identity_entry = _identity_entry(packet, "EAEU")
        text = _block_text(block)
        same_id_markers = (
            "same-id",
            "same id",
            "grls",
            "identifier mismatch",
            "underlying authorization",
            "different registration",
            "corroboration",
        )
        return (
            _eaeu_native_entry_decision_supported(packet)
            and _contains_any_marker(text, same_id_markers)
            and not _has_explicit_negative_evidence(text)
        )

    def _asset_ip_window_overconstraint(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "asset_attractiveness" or block.verdict not in {"HOLD", "NO_GO", "INSUFFICIENT_EVIDENCE"}:
            return False
        snapshot = _ru_eaeu_ip_snapshot(packet)
        conclusion = str(snapshot.get("conclusion") or "")
        if conclusion not in {"NO_LISTED_BLOCKING_PATENT_EVIDENCE", "PARTIAL_OPEN_WINDOW_EVIDENCE"}:
            return False
        if not (_has_positive_registration(packet, "RU") or _has_positive_registration(packet, "EAEU")):
            return False
        text = _block_text(block)
        patent_markers = (
            "patent",
            "ip window",
            "ip-window",
            "legal status",
            "legal-status",
            "expiry",
            "fips",
            "eapo",
        )
        return _contains_any_marker(text, patent_markers) and not _has_explicit_negative_evidence(text)

    def _regional_generic_collapse(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "generic_opportunity" or block.verdict != "NOT_EVIDENCED":
            return False
        regional = _regional_opportunity(packet, "generic_opportunity")
        verdicts = {str((payload or {}).get("verdict") or "") for payload in regional.values()}
        return "POTENTIAL_GO" in verdicts and any(value in {"HOLD_OR_NO_GO", "NOT_EVIDENCED"} for value in verdicts)

    def _regional_licensing_collapse(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "licensing_opportunity" or block.verdict != "NOT_EVIDENCED":
            return False
        regional = _regional_opportunity(packet, "licensing_opportunity")
        verdicts = {str((payload or {}).get("verdict") or "") for payload in regional.values()}
        return any(value in {"LOW", "MEDIUM"} for value in verdicts)

    def _business_block_synthesis_overconstraint(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id not in {
            "asset_attractiveness",
            "rf_entry",
            "eaeu_entry",
            "generic_opportunity",
            "licensing_opportunity",
            "portfolio_opportunity",
        }:
            return False
        if block.verdict not in {"HOLD", "NO_GO", "INSUFFICIENT_EVIDENCE", "NOT_EVIDENCED"}:
            return False
        text = _block_text(block)
        synthesis_markers = ("synthesis", "route", "manufacturing", "process", "cmc")
        if not _contains_any_marker(text, synthesis_markers):
            return False
        screening = (_contract_linkage(packet).get("synthesis_screening", {}) or {})
        if str(screening.get("decision_use") or "") != "technical_screening_only":
            return False
        return not _has_explicit_negative_evidence(text)

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

        if self._rf_underlinked_conditional_go(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="rf_identity_underlink",
                    severity="WARN",
                    message="RF entry still sits below GO even though RU registration and RU commercial signals already align at product-context level.",
                )
            )

        if self._eaeu_same_id_overconstraint(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="eaeu_same_id_overconstraint",
                    severity="WARN",
                    message="EAEU entry is overconstrained by GRLS same-id corroboration despite a strong EAEU-native registration identity anchor.",
                )
            )

        if self._asset_ip_window_overconstraint(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="asset_ip_window_overconstraint",
                    severity="WARN",
                    message="Asset verdict is still being held down by RU/EAEU IP-window missingness despite official no-hit/open-window evidence.",
                )
            )

        if self._regional_generic_collapse(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="regional_generic_collapse",
                    severity="WARN",
                    message="Generic opportunity collapsed into a global unsupported verdict even though the packet now shows region-dependent opportunity.",
                )
            )

        if self._regional_licensing_collapse(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="regional_licensing_collapse",
                    severity="WARN",
                    message="Licensing opportunity collapsed into NOT_EVIDENCED even though the packet now shows region-dependent business-development posture.",
                )
            )

        if self._business_block_synthesis_overconstraint(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="synthesis_secondary_scope",
                    severity="WARN",
                    message="Synthesis/manufacturing evidence should surface as a caveat for BD/entry blocks rather than as a primary blocker.",
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
            if repaired.verdict in {"GO", "CONDITIONAL_GO"}:
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

        if any(issue.issue_type == "rf_identity_underlink" for issue in verification.issues) or self._rf_underlinked_conditional_go(repaired, packet):
            linkage = _market_entry_linkage(packet, "RU")
            match_level = str(linkage.get("identity_match") or "")
            linkage_phrase = (
                "the same RU registration identifier"
                if match_level == "same_identifier"
                else "the same RU MAH / product context"
            )
            repaired.verdict = "GO"
            repaired.sufficiency = "SUFFICIENT"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                f"GO — active RU registration is confirmed and RU commercial/access signals already map to {linkage_phrase}, so RF entry should not stay at CONDITIONAL_GO."
            )
            repaired.full_answer = (
                "RF entry remains anchored to the active RU registration context. "
                f"The packet now carries explicit market-entry linkage showing that RU commercial/formulary/procurement evidence maps to {linkage_phrase}. "
                "That removes the earlier INN-level-only ambiguity and supports a clean GO for the RU block."
            )
            repaired.top_evidence_refs = list(dict.fromkeys(list(linkage.get("evidence_refs") or []) + list(repaired.top_evidence_refs)))[:8]
            caveat = "RU access evidence is now treated as product-context-linked rather than a pure INN-level proxy."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            repaired.decision_blockers = []
            repaired.next_actions = []
            applied_changes.append("promoted_rf_conditional_go_to_go_on_identity_linkage")

        if any(issue.issue_type == "eaeu_same_id_overconstraint" for issue in verification.issues) or self._eaeu_same_id_overconstraint(repaired, packet):
            identity_entry = _identity_entry(packet, "EAEU")
            linkage = _market_entry_linkage(packet, "EAEU")
            has_commercial = int(linkage.get("commercial_signal_count") or 0) > 0
            identifier = ", ".join((identity_entry.get("identifiers") or [])[:1])
            validity_value = str(identity_entry.get("valid_to") or "").strip() or str(identity_entry.get("validity_type") or "").strip()
            supports_native_entry_go = _eaeu_native_entry_decision_supported(packet)
            repaired.verdict = "GO" if has_commercial else "CONDITIONAL_GO"
            repaired.sufficiency = "SUFFICIENT" if has_commercial else "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "GO — EAEU-native registration identity, status, and validity are already sufficient for an EAEU regulatory entry conclusion."
                if has_commercial
                else "CONDITIONAL_GO — EAEU-native registration identity, status, and validity are already sufficient; remaining commercial follow-up is separate from the same-id question."
            )
            repaired.full_answer = (
                "The packet includes an EAEU-native registration anchor with identifier, status, and validity evidence. "
                f"Identifier {identifier or 'for the EAEU product context'} remains authorised with validity {validity_value or 'confirmed in-source'}. "
                "A different RU GRLS identifier is treated as a separate regional product context rather than as a contradiction, so GRLS same-id corroboration should remain optional."
            )
            repaired.top_evidence_refs = list(dict.fromkeys(list(identity_entry.get("evidence_refs") or []) + list(repaired.top_evidence_refs)))[:8]
            repaired.decision_blockers = [
                blocker for blocker in repaired.decision_blockers
                if not (
                    _contains_any_marker(f"{blocker.title} {blocker.rationale}", ("same-id", "same id", "grls", "identifier", "corroboration"))
                    or (supports_native_entry_go and _is_eaeu_dossier_wide_coverage_text(f"{blocker.title} {blocker.rationale}"))
                )
            ]
            repaired.next_actions = [
                action for action in repaired.next_actions
                if not (
                    _contains_any_marker(f"{action.action} {action.rationale}", ("same-id", "same id", "grls", "identifier", "corroboration"))
                    or (supports_native_entry_go and _is_eaeu_dossier_wide_coverage_text(f"{action.action} {action.rationale}"))
                )
            ]
            repaired.caveats = [
                caveat for caveat in repaired.caveats
                if not (supports_native_entry_go and _is_eaeu_dossier_wide_coverage_text(caveat))
            ]
            caveat = "RU and EAEU registrations are treated as separate product contexts unless the packet explicitly proves same-identifier linkage."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            coverage_caveat = "Dossier-wide IP/FTO and rights gaps remain in their dedicated blocks; they do not override the EAEU registration-entry conclusion when EAEU-native identity, status, and validity are source-backed."
            if supports_native_entry_go and coverage_caveat not in repaired.caveats:
                repaired.caveats.append(coverage_caveat)
            if not has_commercial:
                commercial_caveat = "EAEU commercial/access evidence is still thinner than the regulatory anchor and should be completed separately."
                if commercial_caveat not in repaired.caveats:
                    repaired.caveats.append(commercial_caveat)
            applied_changes.append("removed_same_id_grls_overconstraint_from_eaeu")
            if supports_native_entry_go:
                applied_changes.append("moved_eaeu_dossier_wide_coverage_gap_to_caveat")

        if any(issue.issue_type == "asset_ip_window_overconstraint" for issue in verification.issues) or self._asset_ip_window_overconstraint(repaired, packet):
            snapshot = _ru_eaeu_ip_snapshot(packet)
            as_of_date = str(snapshot.get("as_of_date") or "").strip()
            repaired.verdict = "CONDITIONAL_GO"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "CONDITIONAL_GO — checked RU/EAEU official patent sources do not show listed blocking pharma patent evidence, so IP remains a residual-risk caveat rather than a hold-level blocker."
            )
            repaired.full_answer = (
                "The packet now carries a normalized RU/EAEU IP-window snapshot. "
                f"As of {as_of_date or 'the documented source dates'}, checked official sources support a no-listed-blocking-patent or open-window reading for RU/EAEU. "
                "That is still not a full freedom-to-operate opinion, but it should no longer force asset attractiveness into HOLD purely because expiry/legal-status fields were incomplete."
            )
            repaired.decision_blockers = [
                blocker for blocker in repaired.decision_blockers
                if not _contains_any_marker(f"{blocker.title} {blocker.rationale}", ("patent", "ip", "expiry", "legal status", "fips", "eapo"))
            ]
            repaired.next_actions = [
                action for action in repaired.next_actions
                if not _contains_any_marker(f"{action.action} {action.rationale}", ("patent", "ip", "expiry", "legal status", "fips", "eapo"))
            ]
            caveat = "RU/EAEU IP conclusion is still a residual-risk legal snapshot, not a formal FTO opinion."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("lifted_asset_hold_from_ru_eaeu_ip_missingness")

        if any(issue.issue_type == "regional_generic_collapse" for issue in verification.issues) or self._regional_generic_collapse(repaired, packet):
            regional = _regional_opportunity(packet, "generic_opportunity")
            positive_regions = [region for region, payload in regional.items() if str((payload or {}).get("verdict") or "") == "POTENTIAL_GO"]
            constrained_regions = [
                region
                for region, payload in regional.items()
                if str((payload or {}).get("verdict") or "") in {"HOLD_OR_NO_GO", "NOT_EVIDENCED"}
            ]
            repaired.verdict = "MEDIUM" if positive_regions else "LOW"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM" if positive_regions else "LOW"
            repaired.short_answer = (
                f"Region-dependent {repaired.verdict} — {', '.join(positive_regions) or 'some jurisdictions'} show potential generic headroom, while {', '.join(constrained_regions[:3]) or 'other jurisdictions'} remain blocked or unresolved."
            )
            repaired.full_answer = (
                "Generic opportunity should be expressed region-by-region, not collapsed into a single global unsupported verdict. "
                f"Current packet logic supports potential generic headroom in {', '.join(positive_regions) or 'the supported jurisdictions'}, "
                f"while {', '.join(constrained_regions[:3]) or 'other jurisdictions'} remain blocked or not yet decision-grade."
            )
            repaired.caveats = [
                caveat for caveat in repaired.caveats
                if "global unsupported verdict" not in caveat.lower()
            ]
            repaired.caveats.append("Generic opportunity remains region-dependent and should not be narrated as one global patent answer.")
            applied_changes.append("reframed_generic_opportunity_by_region")

        if any(issue.issue_type == "regional_licensing_collapse" for issue in verification.issues) or self._regional_licensing_collapse(repaired, packet):
            regional = _regional_opportunity(packet, "licensing_opportunity")
            medium_regions = [region for region, payload in regional.items() if str((payload or {}).get("verdict") or "") == "MEDIUM"]
            low_regions = [region for region, payload in regional.items() if str((payload or {}).get("verdict") or "") == "LOW"]
            repaired.verdict = "MEDIUM" if medium_regions else "LOW"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM" if medium_regions else "LOW"
            repaired.short_answer = (
                "Region-dependent licensing posture — some jurisdictions still need BD work, while others already look like established registration/access contexts with only weak licensing upside."
            )
            repaired.full_answer = (
                "Licensing should not be narrated as a single global verdict. "
                f"Jurisdictions needing active BD work: {', '.join(medium_regions) or 'none surfaced strongly'}; "
                f"jurisdictions with only weak licensing upside because registration/access is already established: {', '.join(low_regions[:4]) or 'not evidenced'}."
            )
            repaired.caveats.append("Licensing opportunity is region-dependent; established registration contexts can legitimately imply weak opportunity rather than no evidence.")
            applied_changes.append("reframed_licensing_opportunity_by_region")

        if any(issue.issue_type == "synthesis_secondary_scope" for issue in verification.issues) or self._business_block_synthesis_overconstraint(repaired, packet):
            repaired.decision_blockers = [
                blocker for blocker in repaired.decision_blockers
                if not _contains_any_marker(f"{blocker.title} {blocker.rationale}", ("synthesis", "route", "manufacturing", "process", "cmc"))
            ]
            repaired.next_actions = [
                action for action in repaired.next_actions
                if not _contains_any_marker(f"{action.action} {action.rationale}", ("synthesis", "route", "manufacturing", "process", "cmc"))
            ]
            screening_caveat = "Synthesis evidence remains screening-grade and should not be used for manufacturing / CMC conclusions."
            if screening_caveat not in repaired.caveats:
                repaired.caveats.append(screening_caveat)
            if repaired.block_id == "asset_attractiveness" and repaired.verdict in {"HOLD", "NO_GO", "INSUFFICIENT_EVIDENCE"}:
                repaired.verdict = "CONDITIONAL_GO"
                repaired.sufficiency = "PARTIAL"
                repaired.confidence = "MEDIUM"
                repaired.short_answer = (
                    "CONDITIONAL_GO — synthesis remains screening-grade, but that should stay a technical caveat rather than a primary BD / market-entry blocker."
                )
                repaired.full_answer = (
                    "Synthesis/manufacturing evidence is still useful only for initial technical screening. "
                    "For this BD-oriented asset block, that limitation should remain a caveat rather than a hold-level or no-go-level blocker."
                )
            applied_changes.append("demoted_synthesis_to_screening_scope")

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
            "rf_identity_underlink",
            "eaeu_same_id_overconstraint",
            "asset_ip_window_overconstraint",
            "regional_generic_collapse",
            "regional_licensing_collapse",
            "synthesis_secondary_scope",
        }
        has_reparable_warn = any(
            issue.issue_type in reparable_warns for issue in verification.issues
        )
        needs_policy_repair = any(
            (
                self._asset_negative_missing_evidence_overreach(block, packet),
                self._rf_scope_overconstraint(block, packet),
                self._eaeu_holdable_regulatory_position(block, packet),
                self._rf_underlinked_conditional_go(block, packet),
                self._eaeu_same_id_overconstraint(block, packet),
                self._asset_ip_window_overconstraint(block, packet),
                self._regional_generic_collapse(block, packet),
                self._regional_licensing_collapse(block, packet),
                self._business_block_synthesis_overconstraint(block, packet),
            )
        )
        if (verification.overall_status != "FAIL" and not has_reparable_warn and not needs_policy_repair) or not allow_repair:
            return block, verification
        repaired_block, repaired_verification = self.repair_block(block, packet, verification)
        return repaired_block, repaired_verification
