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
        ExecWhyClaim,
        ExecVerificationIssue,
        ExecVerificationReport,
    )
except ImportError:  # pragma: no cover
    from dossier_schema_v3 import (  # type: ignore
        ExecBlocker,
        ExecConfidenceEnum,
        ExecDecisionBlock,
        ExecNextAction,
        ExecWhyClaim,
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
    lowered = re.sub(
        r"\b(?:not\s+)?(?:fully\s+)?closed evidence state\b",
        "",
        lowered,
    )
    lowered = re.sub(
        r"\bnot\s+(?:expired|withdrawn|revoked|suspended|inactive|refused)\b",
        "",
        lowered,
    )
    lowered = re.sub(
        r"\brather than an? (?:expired|withdrawn|revoked|suspended|inactive)(?: or (?:expired|withdrawn|revoked|suspended|inactive))* state\b",
        "",
        lowered,
    )
    lowered = re.sub(r"\bnon-negative\b", "", lowered)
    lowered = re.sub(r"\bnot\s+fully\s+closed\b", "", lowered)
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


def _block_primary_decision_text(block: ExecDecisionBlock) -> str:
    parts: List[str] = [block.short_answer, block.full_answer]
    parts.extend(claim.claim for claim in block.why_this_verdict)
    for blocker in block.decision_blockers:
        parts.append(blocker.title)
        parts.append(blocker.rationale)
    for action in block.next_actions:
        parts.append(action.action)
        parts.append(action.rationale)
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


def _market_reimbursement_snapshot(packet: Dict[str, Any]) -> Dict[str, Any]:
    return (_contract_linkage(packet).get("market_reimbursement_snapshot", {}) or {})


def _ru_eaeu_ip_snapshot(packet: Dict[str, Any]) -> Dict[str, Any]:
    return (_contract_linkage(packet).get("ru_eaeu_ip_window_snapshot", {}) or {})


def _contract_linkage_summary(packet: Dict[str, Any]) -> Dict[str, Any]:
    return (((packet.get("evidence_packet_summary") or {}).get("contract_linkage_summary") or {}) or {})


def _operations_screening_ready(packet: Dict[str, Any]) -> bool:
    linkage = _contract_linkage(packet)
    operations = linkage.get("operations_readiness_snapshot", {}) or {}
    summary = _contract_linkage_summary(packet)
    return bool(operations.get("screening_ready")) or bool(summary.get("operations_screening_ready"))


def _source_manifest_count(packet: Dict[str, Any]) -> int:
    linkage = _contract_linkage(packet)
    source_manifest = linkage.get("source_evidence_manifest", {}) or {}
    summary = _contract_linkage_summary(packet)
    return (
        int(source_manifest.get("checked_source_count") or 0)
        + int(source_manifest.get("limited_source_count") or 0)
        + int(summary.get("source_manifest_checked_count") or 0)
    )


def _explicit_no_registration_refs(packet: Dict[str, Any], region: str) -> List[str]:
    region = str(region or "").strip().upper()
    region_markers = {region.lower(), f"jurisdiction={region.lower()}", f"region={region.lower()}"}
    no_record_markers = (
        "no public registration record",
        "no public registration record verified",
        "no public registration",
        "no registration record",
        "official absence",
        "source-native no-record",
        "no-record",
    )
    refs: List[str] = []
    for item in _packet_section_items(packet, "registrations"):
        if not isinstance(item, dict) or _region_text(item) != region:
            continue
        text = " ".join(
            str(part or "")
            for part in (
                _registration_status_text(item),
                _scalar_text(item.get("summary")),
                _scalar_text(item.get("limitations")),
            )
        ).lower()
        if any(marker in text for marker in no_record_markers):
            refs.extend(str(ref) for ref in item.get("evidence_refs", []) or [] if ref)
    for candidate in _candidate_evidence(packet):
        text = str(candidate.get("searchable") or "").lower()
        if not any(marker in text for marker in region_markers):
            continue
        if "registration" not in text and "product_identity_bridge" not in text:
            continue
        if any(marker in text for marker in no_record_markers):
            refs.append(candidate["ref"])
    return list(dict.fromkeys(refs))


def _has_explicit_no_registration_record(packet: Dict[str, Any], region: str) -> bool:
    region = str(region or "").strip().upper()
    for item in _packet_section_items(packet, "registrations"):
        if not isinstance(item, dict) or _region_text(item) != region:
            continue
        text = " ".join(
            str(part or "")
            for part in (
                _registration_status_text(item),
                _scalar_text(item.get("summary")),
                _scalar_text(item.get("limitations")),
            )
        ).lower()
        if any(
            marker in text
            for marker in (
                "no public registration record",
                "no public registration record verified",
                "no public registration",
                "no registration record",
                "official absence",
                "source-native no-record",
                "no-record",
            )
        ):
            return True
    return bool(_explicit_no_registration_refs(packet, region))


def _strip_no_record_phrases(text: str) -> str:
    stripped = str(text or "").lower()
    no_record_phrases = (
        "no public registration record verified",
        "no public registration record",
        "no public registration",
        "no registration record found",
        "no registration record",
        "no public record",
        "no-record",
        "not found in public registry",
    )
    for phrase in no_record_phrases:
        stripped = stripped.replace(phrase, " ")
    return stripped


def _has_source_backed_registration_blocker(packet: Dict[str, Any], region: str) -> bool:
    """True only for source-backed blockers, not mere absence of a public record."""
    region = str(region or "").strip().upper()
    blocker_markers = (
        "application withdrawn",
        "authorisation refused",
        "authorization refused",
        "clinical hold",
        "failed registration",
        "import ban",
        "marketing authorisation refused",
        "marketing authorization refused",
        "prohibited",
        "registration failed",
        "rejected",
        "refused",
        "revoked",
        "suspended",
        "terminated",
        "withdrawn",
    )
    texts: List[str] = []
    for item in _packet_section_items(packet, "registrations"):
        if not isinstance(item, dict) or _region_text(item) != region:
            continue
        texts.append(
            " ".join(
                str(part or "")
                for part in (
                    _registration_status_text(item),
                    _scalar_text(item.get("summary")),
                    _scalar_text(item.get("limitations")),
                )
            )
        )
    for candidate in _candidate_evidence(packet):
        text = str(candidate.get("searchable") or "")
        lowered = text.lower()
        if region.lower() not in lowered and f"jurisdiction={region.lower()}" not in lowered and f"region={region.lower()}" not in lowered:
            continue
        texts.append(text)
    cleaned = " ".join(_strip_no_record_phrases(text) for text in texts)
    return any(marker in cleaned for marker in blocker_markers)


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
            "product context",
            "product-context alignment",
            "product_context_match_confirmed",
            "same registered ru product identity",
            "same_identifier_confirmed",
        )
        return any(marker in text for marker in scope_markers) and not _has_explicit_negative_evidence(text)

    def _rf_no_record_understated(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "rf_entry" or block.verdict not in {"NO_GO", "CONDITIONAL_GO", "INSUFFICIENT_EVIDENCE"}:
            return False
        if _has_positive_registration(packet, "RU"):
            return False
        if not _has_explicit_no_registration_record(packet, "RU"):
            return False
        if _has_source_backed_registration_blocker(packet, "RU"):
            return False
        if block.verdict == "NO_GO":
            return True
        text = _block_text(block)
        return _contains_any_marker(
            text,
            (
                "no verified ru registration",
                "no ru registration",
                "no public registration",
                "grls",
                "registration record",
                "cannot be approved",
                "missing registry",
            ),
        )

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

    def _eaeu_no_record_understated(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "eaeu_entry" or block.verdict not in {"NO_GO", "CONDITIONAL_GO", "INSUFFICIENT_EVIDENCE"}:
            return False
        if _has_positive_registration(packet, "EAEU") or _eaeu_native_entry_decision_supported(packet):
            return False
        if not _has_explicit_no_registration_record(packet, "EAEU"):
            return False
        if _has_source_backed_registration_blocker(packet, "EAEU"):
            return False
        if block.verdict == "NO_GO":
            return True
        text = _block_text(block)
        no_record_markers = (
            "no confirmed",
            "no public registration",
            "does not confirm",
            "not confirm",
            "registration anchor",
            "not registered",
            "no-record",
        )
        return _contains_any_marker(text, no_record_markers)

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

    def _eaeu_underlinked_conditional_go(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "eaeu_entry" or block.verdict not in {"CONDITIONAL_GO", "HOLD", "INSUFFICIENT_EVIDENCE"}:
            return False
        if not _eaeu_native_entry_decision_supported(packet):
            return False
        linkage = _market_entry_linkage(packet, "EAEU")
        if int(linkage.get("commercial_signal_count") or 0) <= 0:
            return False
        if _identity_match_rank(str(linkage.get("identity_match") or "")) < 2:
            return False
        text = _block_text(block)
        linkage_markers = (
            "commercial",
            "access",
            "identity",
            "linkage",
            "product context",
            "same identifier",
            "inn-level",
            "source-native",
        )
        return (
            (block.verdict == "CONDITIONAL_GO" or _contains_any_marker(text, linkage_markers))
            and not _has_explicit_negative_evidence(text)
        )

    def _eaeu_validity_understated_hold(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "eaeu_entry" or block.verdict not in {"HOLD", "CONDITIONAL_GO", "INSUFFICIENT_EVIDENCE"}:
            return False
        summary = (((packet.get("evidence_packet_summary") or {}).get("contract_linkage_summary") or {}) or {})
        has_validity = bool(summary.get("eaeu_has_valid_to")) or bool(summary.get("eaeu_has_validity_state"))
        if not has_validity:
            identity_entry = _identity_entry(packet, "EAEU")
            has_validity = bool(identity_entry.get("valid_to")) or str(identity_entry.get("validity_type") or "").strip().lower() in {
                "date_present",
                "indefinite",
            }
        if not has_validity:
            return False
        identity_match = str(summary.get("eaeu_identity_match") or (_market_entry_linkage(packet, "EAEU").get("identity_match") or ""))
        if _identity_match_rank(identity_match) < 2:
            return False
        linkage = _market_entry_linkage(packet, "EAEU")
        has_commercial_link = bool(summary.get("eaeu_access_registration_id_overlap")) or int(linkage.get("commercial_signal_count") or 0) > 0
        if not has_commercial_link:
            return False
        text = _block_text(block)
        validity_markers = ("validity", "valid_to", "valid to", "validity dates", "validity term", "срок", "действ")
        hard_entry_negative = ("withdrawn", "suspended", "revoked", "not registered", "inactive", "refused")
        return _contains_any_marker(text, validity_markers) and not _contains_any_marker(text, hard_entry_negative)

    def _eaeu_conditional_go_underpromoted(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "eaeu_entry" or block.verdict not in {"CONDITIONAL_GO", "HOLD"}:
            return False
        if any(blocker.severity in _BLOCKING_SEVERITIES for blocker in block.decision_blockers):
            return False
        summary = (((packet.get("evidence_packet_summary") or {}).get("contract_linkage_summary") or {}) or {})
        identity_match = str(summary.get("eaeu_identity_match") or (_market_entry_linkage(packet, "EAEU").get("identity_match") or ""))
        has_identity = _identity_match_rank(identity_match) >= 2
        has_validity = bool(summary.get("eaeu_has_valid_to")) or bool(summary.get("eaeu_has_validity_state"))
        has_access_link = (
            bool(summary.get("eaeu_access_registration_id_overlap"))
            or int(summary.get("eaeu_structured_bridge_signal_count") or 0) > 0
            or int((_market_entry_linkage(packet, "EAEU") or {}).get("commercial_signal_count") or 0) > 0
        )
        has_payer_scope = str(summary.get("market_reimbursement_verdict_hint") or "").upper() in {"LIMITED", "OPEN"}
        text = _block_text(block)
        conditional_markers = (
            "primary commercial",
            "commercial source",
            "commercial artifact",
            "payer",
            "policy",
            "pricing",
            "reimbursement",
            "access evidence",
            "market-access",
            "strength",
            "patent",
            "fto",
            "freedom-to-operate",
            "execution risk",
            "dedicated",
        )
        return (
            has_identity
            and has_validity
            and has_access_link
            and has_payer_scope
            and _contains_any_marker(text, conditional_markers)
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

    def _asset_dedicated_ip_fto_overconstraint(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "asset_attractiveness" or block.verdict not in {"HOLD", "NO_GO", "INSUFFICIENT_EVIDENCE"}:
            return False
        text = _block_text(block)
        if _has_explicit_negative_evidence(text):
            non_ip_negative_markers = (
                "clinical failed",
                "failed clinical",
                "failed study",
                "registration withdrawn",
                "marketing authorization withdrawn",
                "marketing authorisation withdrawn",
                "registration suspended",
                "not approved",
                "not registered",
            )
            if _contains_any_marker(text, non_ip_negative_markers):
                return False
        ip_markers = (
            "exclusivity",
            "freedom-to-operate",
            "fto",
            "ip window",
            "ip-window",
            "legal window",
            "legal status",
            "patent",
        )
        if not _contains_any_marker(text, ip_markers):
            return False
        linkage = _contract_linkage(packet)
        fto = linkage.get("fto_screening_snapshot", {}) or {}
        family_events = linkage.get("family_legal_events_snapshot", {}) or {}
        has_dedicated_ip_screening = (
            bool(fto)
            and str(fto.get("conclusion") or "") in {
                "POTENTIAL_BLOCKERS_REQUIRE_REVIEW",
                "INSUFFICIENT_FOR_FTO",
            }
        ) or bool(family_events)
        if not has_dedicated_ip_screening:
            return False
        phase3 = linkage.get("phase3_results", {}) or {}
        has_clinical_or_market_anchor = (
            int(phase3.get("phase3_study_count") or 0) > 0
            or int(phase3.get("phase3_with_ctgov_results_evidence") or 0) > 0
            or any(_positive_commercial_signal_count(packet, region) > 0 for region in ("RU", "EU", "US", "EAEU"))
        )
        has_registration_anchor = any(_has_positive_registration(packet, region) for region in ("RU", "EU", "US", "EAEU"))
        return has_registration_anchor and has_clinical_or_market_anchor

    def _asset_screening_coverage_overconstraint(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "asset_attractiveness" or block.verdict not in {"GO", "CONDITIONAL_GO", "HOLD", "INSUFFICIENT_EVIDENCE"}:
            return False
        blocker_text = " ".join(f"{blocker.title} {blocker.rationale}" for blocker in block.decision_blockers)
        coverage_markers = (
            "coverage ledger",
            "decision readiness",
            "decision_readiness",
            "decision-complete",
            "run_manifest",
            "expected fields",
            "dossier incompleteness",
            "synthesis is yellow",
            "complete coverage",
        )
        if not _contains_any_marker(blocker_text, coverage_markers):
            return False
        linkage = _contract_linkage(packet)
        summary = (((packet.get("evidence_packet_summary") or {}).get("contract_linkage_summary") or {}) or {})
        has_entry_anchor = any(_has_positive_registration(packet, region) for region in ("RU", "EAEU", "US", "EU")) or bool(linkage.get("registration_identity_map"))
        has_market_anchor = (
            int(summary.get("ru_source_native_access_signal_count") or 0) > 0
            or str(summary.get("market_reimbursement_verdict_hint") or "").upper() in {"LIMITED", "OPEN"}
            or any(_positive_commercial_signal_count(packet, region) > 0 for region in ("RU", "EAEU", "US", "EU"))
        )
        return has_entry_anchor and has_market_anchor and not _has_explicit_negative_evidence(_block_text(block))

    def _asset_conditional_go_underpromoted(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "asset_attractiveness" or block.verdict != "CONDITIONAL_GO":
            return False
        if block.sufficiency != "SUFFICIENT" or block.decision_blockers:
            return False
        linkage = _contract_linkage(packet)
        summary = (((packet.get("evidence_packet_summary") or {}).get("contract_linkage_summary") or {}) or {})
        has_entry_anchor = any(_has_positive_registration(packet, region) for region in ("RU", "EAEU", "US", "EU")) or bool(linkage.get("registration_identity_map"))
        has_market_or_clinical_anchor = (
            int((linkage.get("phase3_results", {}) or {}).get("phase3_study_count") or 0) > 0
            or int((linkage.get("phase3_results", {}) or {}).get("phase3_with_ctgov_results_evidence") or 0) > 0
            or int(summary.get("ru_source_native_access_signal_count") or 0) > 0
            or str(summary.get("market_reimbursement_verdict_hint") or "").upper() in {"LIMITED", "OPEN"}
            or any(_positive_commercial_signal_count(packet, region) > 0 for region in ("RU", "EAEU", "US", "EU"))
        )
        return has_entry_anchor and has_market_or_clinical_anchor and not _has_explicit_negative_evidence(_block_text(block))

    def _ip_window_closed_without_decision_grade_legal_status(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "ip_legal_window" or block.verdict != "CLOSED":
            return False
        linkage = _contract_linkage(packet)
        fto = linkage.get("fto_screening_snapshot", {}) or {}
        family_events = linkage.get("family_legal_events_snapshot", {}) or {}
        if bool(fto.get("full_fto_verdict_allowed")) and bool(family_events.get("decision_grade")):
            return False
        text = _block_text(block)
        uncertainty_markers = (
            "conflict",
            "contradict",
            "incomplete",
            "mixed",
            "no reconciled",
            "not reconciled",
            "not fully",
            "not source-native",
            "screening",
            "unresolved",
        )
        return _contains_any_marker(text, uncertainty_markers)

    def _ip_window_underresolved_with_source_native_blockers(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "ip_legal_window" or block.verdict != "UNRESOLVED":
            return False
        linkage = _contract_linkage(packet)
        fto = linkage.get("fto_screening_snapshot", {}) or {}
        family_events = linkage.get("family_legal_events_snapshot", {}) or {}
        potential_regions = list(fto.get("potential_blocker_regions") or [])
        country_status = fto.get("country_effect_status_by_region", {}) or {}
        has_source_native_status = any(
            str((payload or {}).get("window_status") or "") in {
                "potentially_blocked",
                "open",
                "unresolved_with_source_evidence",
            }
            for payload in country_status.values()
            if isinstance(payload, dict)
        )
        return bool(potential_regions or has_source_native_status or family_events.get("evidence_refs"))

    def _market_reimbursement_underresolved(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "market_reimbursement_window" or block.verdict != "UNRESOLVED":
            return False
        snapshot = _market_reimbursement_snapshot(packet)
        if str(snapshot.get("verdict_hint") or "").strip().upper() != "LIMITED":
            return False
        regions = snapshot.get("regions", {}) or {}
        ru_payload = regions.get("RU", {}) or {}
        eaeu_payload = regions.get("EAEU", {}) or {}
        has_ru_source_native = int(ru_payload.get("listed_active_count") or 0) > 0
        eaeu_scoped = bool(eaeu_payload.get("member_state_scope"))
        return has_ru_source_native or eaeu_scoped

    def _market_reimbursement_overopen(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "market_reimbursement_window" or block.verdict != "OPEN":
            return False
        snapshot = _market_reimbursement_snapshot(packet)
        if str(snapshot.get("verdict_hint") or "").strip().upper() != "LIMITED":
            return False
        text = _block_text(block)
        payer_tier_markers = ("payer tier", "restriction", "formulary breadth", "coverage breadth", "payer-coverage", "direct payer")
        return _contains_any_marker(text, payer_tier_markers) or not bool(snapshot.get("payer_tier_clearance"))

    def _evidence_sufficiency_screening_ready_understated(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "evidence_sufficiency_note" or block.verdict != "INSUFFICIENT":
            return False
        if self._relevant_critical_unknowns(block, packet) and not _operations_screening_ready(packet):
            return False
        linkage = _contract_linkage(packet)
        source_count = _source_manifest_count(packet)
        family_events = linkage.get("family_legal_events_snapshot", {}) or {}
        fto = linkage.get("fto_screening_snapshot", {}) or {}
        reimbursement = _market_reimbursement_snapshot(packet)
        summary = _contract_linkage_summary(packet)
        has_ip_screening = (
            str(fto.get("screening_level") or "") == "FTO_SCREENING_ONLY"
            or bool(fto.get("evidence_refs"))
            or bool(family_events.get("evidence_refs"))
            or str(family_events.get("coverage_status") or "").upper() in {"PARTIAL", "LIMITED"}
            or str(summary.get("fto_screening_conclusion") or "").upper() in {"POTENTIAL_BLOCKERS_REQUIRE_REVIEW", "INSUFFICIENT_FOR_FTO"}
            or str(summary.get("family_legal_events_coverage_status") or "").upper() in {"PARTIAL", "LIMITED"}
        )
        has_payer_screening = (
            str(reimbursement.get("verdict_hint") or "").upper() == "LIMITED"
            and int(reimbursement.get("check_count") or 0) > 0
        ) or (
            str(summary.get("market_reimbursement_verdict_hint") or "").upper() == "LIMITED"
            and _operations_screening_ready(packet)
        )
        has_entry_state = (
            any(_has_positive_registration(packet, region) for region in ("RU", "EAEU", "US", "EU"))
            or bool(linkage.get("registration_identity_map"))
            or any(_has_explicit_no_registration_record(packet, region) for region in ("RU", "EAEU"))
        )
        text = _block_text(block)
        has_operations_ready_gap = _contains_any_marker(
            text,
            (
                "operations-ready",
                "decision-grade",
                "full fto",
                "freedom-to-operate",
                "payer tier",
                "restriction",
                "file-wrapper",
                "spc",
                "terminal disclaimer",
                "insufficient",
            ),
        )
        return (
            (source_count >= 4 or _operations_screening_ready(packet))
            and has_entry_state
            and has_ip_screening
            and has_payer_screening
            and has_operations_ready_gap
        )

    def _decision_blockers_screening_sufficiency_understated(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "decision_blockers" or block.sufficiency != "INSUFFICIENT":
            return False
        linkage = _contract_linkage(packet)
        source_manifest = linkage.get("source_evidence_manifest", {}) or {}
        source_count = int(source_manifest.get("checked_source_count") or 0) + int(source_manifest.get("limited_source_count") or 0)
        fto = linkage.get("fto_screening_snapshot", {}) or {}
        family_events = linkage.get("family_legal_events_snapshot", {}) or {}
        has_ip_screening = (
            str(fto.get("screening_level") or "") == "FTO_SCREENING_ONLY"
            or bool(fto.get("evidence_refs"))
            or bool(family_events.get("evidence_refs"))
            or str(family_events.get("coverage_status") or "").upper() in {"PARTIAL", "LIMITED"}
        )
        return source_count >= 4 and has_ip_screening and not _has_explicit_negative_evidence(_block_text(block))

    @staticmethod
    def _conditional_asset_blockers_are_reflected(block: ExecDecisionBlock) -> bool:
        if block.block_id != "asset_attractiveness":
            return False
        if block.verdict != "CONDITIONAL_GO":
            return False
        if block.sufficiency == "SUFFICIENT" or block.confidence == "HIGH":
            return False
        if not block.decision_blockers:
            return False
        text = _block_text(block).lower()
        if not any(marker in text for marker in ("conditional", "partial", "limited", "caveat", "blocker", "not operations-ready")):
            return False
        return bool(block.next_actions or block.caveats or "retrieve" in text or "verify" in text)

    def _generic_screening_sufficiency_understated(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "generic_opportunity" or block.verdict != "NOT_EVIDENCED" or block.sufficiency != "INSUFFICIENT":
            return False
        linkage = _contract_linkage(packet)
        source_manifest = linkage.get("source_evidence_manifest", {}) or {}
        source_count = int(source_manifest.get("checked_source_count") or 0) + int(source_manifest.get("limited_source_count") or 0)
        family_events = linkage.get("family_legal_events_snapshot", {}) or {}
        text = _block_text(block)
        has_source_screening = source_count >= 4 or bool(family_events.get("evidence_refs")) or str(family_events.get("coverage_status") or "").upper() in {"PARTIAL", "LIMITED"}
        has_legal_gap_reasoning = _contains_any_marker(text, ("patent", "legal-status", "expiry", "spc", "pte", "term extension", "positive gate"))
        return has_source_screening and has_legal_gap_reasoning

    def _portfolio_screening_underpromoted(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "portfolio_opportunity" or block.verdict != "LOW":
            return False
        if any(blocker.severity in _BLOCKING_SEVERITIES for blocker in block.decision_blockers):
            return False
        if _has_explicit_no_registration_record(packet, "EAEU") and not _has_positive_registration(packet, "EAEU"):
            return False
        linkage = _contract_linkage(packet)
        summary = (((packet.get("evidence_packet_summary") or {}).get("contract_linkage_summary") or {}) or {})
        phase3 = linkage.get("phase3_results", {}) or {}
        has_entry_anchor = any(_has_positive_registration(packet, region) for region in ("RU", "EAEU", "US", "EU")) or bool(linkage.get("registration_identity_map"))
        has_clinical_or_market_anchor = (
            int(phase3.get("phase3_study_count") or 0) > 0
            or int(phase3.get("phase3_with_ctgov_results_evidence") or 0) > 0
            or int(summary.get("ru_source_native_access_signal_count") or 0) > 0
            or str(summary.get("market_reimbursement_verdict_hint") or "").upper() in {"LIMITED", "OPEN"}
            or any(_positive_commercial_signal_count(packet, region) > 0 for region in ("RU", "EAEU", "US", "EU"))
        )
        return has_entry_anchor and has_clinical_or_market_anchor and not _has_explicit_negative_evidence(_block_text(block))

    def _portfolio_not_evidenced_understated(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "portfolio_opportunity" or block.verdict != "NOT_EVIDENCED":
            return False
        linkage = _contract_linkage(packet)
        summary = _contract_linkage_summary(packet)
        phase3 = linkage.get("phase3_results", {}) or {}
        has_entry_state = (
            any(_has_positive_registration(packet, region) for region in ("US", "EU"))
            or any(_has_explicit_no_registration_record(packet, region) for region in ("RU", "EAEU"))
            or bool(linkage.get("registration_identity_map"))
        )
        has_clinical_anchor = (
            int(phase3.get("phase3_study_count") or 0) > 0
            or int(phase3.get("phase3_with_ctgov_results_evidence") or 0) > 0
            or int(summary.get("phase3_with_ctgov_results_evidence") or 0) > 0
        )
        has_screening_sources = _source_manifest_count(packet) >= 4 or _operations_screening_ready(packet)
        has_ip_or_payer_context = (
            str(summary.get("market_reimbursement_verdict_hint") or "").upper() in {"LIMITED", "OPEN"}
            or str(summary.get("family_legal_events_coverage_status") or "").upper() in {"PARTIAL", "LIMITED"}
            or str(summary.get("fto_screening_conclusion") or "").upper() in {"POTENTIAL_BLOCKERS_REQUIRE_REVIEW", "INSUFFICIENT_FOR_FTO"}
        )
        text = _block_text(block).lower()
        collapse_markers = ("not evidenced", "missing explicit", "eaeu registration", "commercial signal", "not finalized")
        return (
            has_entry_state
            and has_clinical_anchor
            and has_screening_sources
            and has_ip_or_payer_context
            and any(marker in text for marker in collapse_markers)
        )

    def _key_risks_not_evidenced_understated(
        self,
        block: ExecDecisionBlock,
        packet: Dict[str, Any],
    ) -> bool:
        if block.block_id != "key_risks" or block.verdict != "NOT_EVIDENCED":
            return False
        linkage = _contract_linkage(packet)
        summary = _contract_linkage_summary(packet)
        fto = linkage.get("fto_screening_snapshot", {}) or {}
        family_events = linkage.get("family_legal_events_snapshot", {}) or {}
        has_ip_risk = (
            str(fto.get("conclusion") or "").upper() == "POTENTIAL_BLOCKERS_REQUIRE_REVIEW"
            or bool(fto.get("potential_blocker_regions"))
            or str(summary.get("fto_screening_conclusion") or "").upper() == "POTENTIAL_BLOCKERS_REQUIRE_REVIEW"
            or str(summary.get("ru_eaeu_ip_conclusion") or "").upper() == "BLOCKING_OR_PENDING_EVIDENCE_PRESENT"
            or str(family_events.get("coverage_status") or "").upper() in {"PARTIAL", "LIMITED"}
            or str(summary.get("family_legal_events_coverage_status") or "").upper() in {"PARTIAL", "LIMITED"}
        )
        risk_text = _block_text(block).lower()
        return has_ip_risk and _contains_any_marker(risk_text, ("risk", "blocker", "patent", "legal", "unresolved", "active", "pending"))

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
        text = _block_primary_decision_text(block)
        synthesis_markers = (
            "synthesis",
            "manufacturing",
            "cmc",
            "process chemistry",
            "chemical process",
            "manufacturing process",
            "synthesis route",
        )
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
        evidence_ids = set(packet.get("evidence_ids", [])) or set(packet.get("selected_evidence_ids", [])) or {
            candidate["ref"] for candidate in _candidate_evidence(packet)
        }
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

        if block.block_id != "key_risks" and block.verdict in _GO_VERDICTS and any(
            blocker.severity in _BLOCKING_SEVERITIES for blocker in block.decision_blockers
        ) and not self._conditional_asset_blockers_are_reflected(block):
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

        if self._rf_no_record_understated(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="rf_no_record_understated",
                    severity="WARN",
                    message="RF entry treats explicit RU no-registration/no-public-record as a terminal entry closure; for business screening it should be a no-current-market-status HOLD with original/new-registration pathway checks.",
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

        if self._eaeu_no_record_understated(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="eaeu_no_record_understated",
                    severity="WARN",
                    message="EAEU entry treats explicit no-registration/no-public-record as terminal closure; for business screening it should be a no-current-market-status HOLD with original/new-registration pathway checks.",
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

        if self._eaeu_underlinked_conditional_go(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="eaeu_identity_underlink",
                    severity="WARN",
                    message="EAEU entry still sits below GO even though EAEU-native registration and source-native linkage align at product-context level.",
                )
            )

        if self._eaeu_validity_understated_hold(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="eaeu_validity_understated_hold",
                    severity="WARN",
                    message="EAEU entry is held down by a missing-validity claim even though packet linkage already reports source-native validity state.",
                )
            )

        if self._eaeu_conditional_go_underpromoted(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="eaeu_conditional_go_underpromoted",
                    severity="WARN",
                    message="EAEU entry remains conditional despite source-linked identity, validity, and access evidence; residual FTO/commercial-depth gaps belong in caveats or dedicated blocks.",
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

        if self._asset_dedicated_ip_fto_overconstraint(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="asset_dedicated_ip_fto_overconstraint",
                    severity="WARN",
                    message="Asset attractiveness is being held down by IP/FTO screening gaps that belong in the dedicated legal-window block.",
                )
            )

        if self._asset_screening_coverage_overconstraint(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="asset_screening_coverage_overconstraint",
                    severity="WARN",
                    message="Asset attractiveness is blocked by operations-ready coverage-ledger gaps even though the packet supports screening-ready asset attractiveness.",
                )
            )

        if self._asset_conditional_go_underpromoted(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="asset_conditional_go_underpromoted",
                    severity="WARN",
                    message="Asset attractiveness remains conditional despite sufficient asset evidence and no decision blockers; legal/payer operations gaps belong in dedicated blocks.",
                )
            )

        if self._ip_window_closed_without_decision_grade_legal_status(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="ip_window_closed_without_decision_grade_legal_status",
                    severity="WARN",
                    message="IP legal window is being closed despite incomplete or conflicted source-native family/legal-status coverage.",
                )
            )

        if self._ip_window_underresolved_with_source_native_blockers(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="ip_window_underresolved_with_source_native_blockers",
                    severity="WARN",
                    message="IP legal window remains unresolved despite source-native patent/exclusivity/legal-status evidence supporting at least a limited screening posture.",
                )
            )

        if self._market_reimbursement_underresolved(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="market_reimbursement_underresolved",
                    severity="WARN",
                    message="Market reimbursement window remains unresolved despite RU source-native price/access evidence or EAEU member-state-scope evidence.",
                )
            )

        if self._market_reimbursement_overopen(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="market_reimbursement_overopen",
                    severity="WARN",
                    message="Market reimbursement is marked OPEN even though source linkage only supports a LIMITED payer/access screening posture.",
                )
            )

        if self._evidence_sufficiency_screening_ready_understated(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="evidence_sufficiency_screening_ready_understated",
                    severity="WARN",
                    message="Evidence sufficiency is marked insufficient even though the packet supports screening-ready partial use with explicit IP/FTO and payer limitations.",
                )
            )

        if self._generic_screening_sufficiency_understated(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="generic_screening_sufficiency_understated",
                    severity="WARN",
                    message="Generic opportunity is not positive, but source-native legal screening evidence makes the block partial rather than pure insufficiency.",
                )
            )

        if self._decision_blockers_screening_sufficiency_understated(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="decision_blockers_screening_sufficiency_understated",
                    severity="WARN",
                    message="Decision blockers are screening-classified from source-native IP evidence; the block should be partial rather than pure insufficiency.",
                )
            )

        if self._portfolio_screening_underpromoted(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="portfolio_screening_underpromoted",
                    severity="WARN",
                    message="Portfolio opportunity is marked LOW despite registration and clinical/market screening anchors; legal-status caveats belong in risk/follow-up blocks.",
                )
            )

        if self._portfolio_not_evidenced_understated(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="portfolio_not_evidenced_understated",
                    severity="WARN",
                    message="Portfolio opportunity collapsed to NOT_EVIDENCED even though screening-level jurisdiction, clinical, and legal-status anchors exist.",
                )
            )

        if self._key_risks_not_evidenced_understated(block, packet):
            issues.append(
                ExecVerificationIssue(
                    issue_type="key_risks_not_evidenced_understated",
                    severity="WARN",
                    message="Key risks are marked NOT_EVIDENCED despite source-native IP/legal-status risk evidence.",
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

        if any(issue.issue_type == "rf_no_record_understated" for issue in verification.issues) or self._rf_no_record_understated(repaired, packet):
            refs = _explicit_no_registration_refs(packet, "RU")
            repaired.verdict = "HOLD"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "HOLD — no public RU registration is verified, so current marketed entry is absent, but this is an original/new-registration screening opportunity rather than a terminal NO_GO."
            )
            repaired.full_answer = (
                "The packet supports `current_registration_status=NOT_REGISTERED_PUBLIC_RECORD` for RU. "
                "That closes only the already-marketed/current-access path: it does not prove that RU entry is impossible. "
                "For primary business screening, the defensible branch is `generic_entry_path=NOT_APPLICABLE_NO_REFERENCE_REGISTRATION` and "
                "`original_registration_path=POSSIBLE_BUT_UNPROVEN`, pending source-backed checks of local clinical activity, foreign approval maturity, regulatory route, IP/FTO, and payer feasibility."
            )
            repaired.top_evidence_refs = list(dict.fromkeys(refs + list(repaired.top_evidence_refs)))[:8]
            repaired.decision_blockers = [
                ExecBlocker(
                    blocker_id="ru_current_registration_absent",
                    title="No current RU public registration",
                    severity="IMPORTANT",
                    rationale="No source-native RU public registration record is verified, so current marketed access is absent; this is not, by itself, a source-backed prohibition on original/new-drug registration.",
                    evidence_refs=refs[:4],
                )
            ]
            repaired.next_actions = [
                ExecNextAction(
                    action_id="check_ru_original_registration_pathway",
                    action="Evaluate RU original/new-drug registration pathway, including foreign approval package, local bridging or full-dossier requirements, orphan/special-access options, and sponsor/partner posture.",
                    priority="NOW",
                    rationale="Absence of current RU registration shifts the decision from marketed-entry approval to pathway feasibility screening.",
                    evidence_refs=refs[:4],
                ),
                ExecNextAction(
                    action_id="check_ru_local_development_activity",
                    action="Check RU clinical-trial/development activity: sponsor, collaborators, phase, status, sites, centers, and whether activity signals launch preparation.",
                    priority="NOW",
                    rationale="Local development activity determines whether the no-record state is white space, an early launch signal, or a low-priority region.",
                    evidence_refs=[],
                ),
            ]
            repaired.why_this_verdict.append(
                ExecWhyClaim(
                    claim="Explicit no-public-registration evidence supports no current RU marketed-access anchor, but no source-backed prohibition or failed registration is present in the packet.",
                    claim_type="hard_evidence_backed" if refs else "inference",
                    evidence_refs=refs[:4],
                )
            )
            caveat = "No public RU registration means no current RU marketed-entry anchor; it must not be interpreted as proof that original/new-drug registration is impossible."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("reframed_rf_no_record_as_original_registration_opportunity")

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

        if any(issue.issue_type == "eaeu_no_record_understated" for issue in verification.issues) or self._eaeu_no_record_understated(repaired, packet):
            refs = _explicit_no_registration_refs(packet, "EAEU")
            repaired.verdict = "HOLD"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "HOLD — no public EAEU registration is verified, so current EAEU marketed entry is absent, but this is a white-space/original-registration screening branch rather than terminal NO_GO."
            )
            repaired.full_answer = (
                "The packet supports `current_registration_status=NOT_REGISTERED_PUBLIC_RECORD` for EAEU. "
                "That means no confirmed current EAEU market authorization/product-context anchor is present, but it does not prove that entry is impossible. "
                "The proper screening split is `generic_entry_path=NOT_APPLICABLE_NO_REFERENCE_REGISTRATION` and "
                "`original_registration_path=POSSIBLE_BUT_UNPROVEN`, pending checks of EAEU/member-state regulatory route, local development activity, foreign approvals, IP/FTO, and payer feasibility."
            )
            repaired.top_evidence_refs = list(dict.fromkeys(refs + list(repaired.top_evidence_refs)))[:8]
            repaired.decision_blockers = [
                ExecBlocker(
                    blocker_id="eaeu_current_registration_absent",
                    title="No current EAEU public registration",
                    severity="IMPORTANT",
                    rationale="No source-native EAEU public registration record is verified, so current marketed access is absent; this is not, by itself, a source-backed prohibition on original/new-drug registration.",
                    evidence_refs=refs[:4],
                )
            ]
            repaired.next_actions = [
                ExecNextAction(
                    action_id="check_eaeu_original_registration_pathway",
                    action="Evaluate EAEU original/new-drug registration pathway and member-state route requirements, using foreign approvals, clinical package maturity, and potential partner/sponsor evidence.",
                    priority="NOW",
                    rationale="Absence of current EAEU registration shifts the decision from marketed-entry approval to pathway feasibility screening.",
                    evidence_refs=refs[:4],
                ),
                ExecNextAction(
                    action_id="check_eaeu_local_development_activity",
                    action="Check EAEU/RU/BY/AM/KZ/KG clinical-trial/development activity: sponsor, collaborators, phase, status, sites, centers, and launch-preparation signals.",
                    priority="NOW",
                    rationale="Local development activity determines whether the no-record state is white space, an early launch signal, or a low-priority region.",
                    evidence_refs=[],
                ),
            ]
            repaired.why_this_verdict.append(
                ExecWhyClaim(
                    claim="Explicit no-public-registration evidence supports no current EAEU marketed-access anchor, but no source-backed prohibition or failed registration is present in the packet.",
                    claim_type="hard_evidence_backed" if refs else "inference",
                    evidence_refs=refs[:4],
                )
            )
            caveat = "No public EAEU registration means no current EAEU marketed-entry anchor; it must not be interpreted as proof that original/new-drug registration is impossible."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("reframed_eaeu_no_record_as_original_registration_opportunity")

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

        if (
            any(issue.issue_type in {"eaeu_identity_underlink", "eaeu_validity_understated_hold", "eaeu_conditional_go_underpromoted"} for issue in verification.issues)
            or self._eaeu_underlinked_conditional_go(repaired, packet)
            or self._eaeu_validity_understated_hold(repaired, packet)
            or self._eaeu_conditional_go_underpromoted(repaired, packet)
        ):
            identity_entry = _identity_entry(packet, "EAEU")
            linkage = _market_entry_linkage(packet, "EAEU")
            summary = (((packet.get("evidence_packet_summary") or {}).get("contract_linkage_summary") or {}) or {})
            match_level = str(linkage.get("identity_match") or summary.get("eaeu_identity_match") or "")
            linkage_phrase = (
                "the same EAEU registration identifier"
                if match_level == "same_identifier"
                else "the same EAEU MAH / product context"
            )
            identifier = ", ".join((identity_entry.get("identifiers") or [])[:1])
            validity_value = (
                str(identity_entry.get("valid_to") or "").strip()
                or str(identity_entry.get("validity_type") or "").strip()
                or ("source linkage confirms validity state" if summary.get("eaeu_has_valid_to") or summary.get("eaeu_has_validity_state") else "")
            )
            repaired.verdict = "GO"
            repaired.sufficiency = "SUFFICIENT"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                f"GO — EAEU-native registration is confirmed and source-native access/commercial linkage maps to {linkage_phrase}."
            )
            repaired.full_answer = (
                "The packet carries an EAEU-native registration anchor with identifier, status, and validity evidence. "
                f"Identifier {identifier or 'for the EAEU product context'} remains authorised with validity {validity_value or 'confirmed in-source'}, "
                f"and the market-entry linkage now maps source-native access/commercial evidence to {linkage_phrase}. "
                "Dossier-wide IP/FTO and rights gaps remain in their dedicated blocks, but they should not keep the regulatory entry block at CONDITIONAL_GO."
            )
            repaired.top_evidence_refs = list(
                dict.fromkeys(
                    list(linkage.get("evidence_refs") or [])
                    + list(identity_entry.get("evidence_refs") or [])
                    + list(repaired.top_evidence_refs)
                )
            )[:8]
            repaired.decision_blockers = [
                blocker for blocker in repaired.decision_blockers
                if not _contains_any_marker(
                    f"{blocker.title} {blocker.rationale}",
                    (
                        "inn-level",
                        "identity",
                        "linkage",
                        "commercial",
                        "access",
                        "product context",
                        "validity",
                        "validity dates",
                        "validity term",
                        "primary commercial",
                        "strength",
                        "patent",
                        "fto",
                        "freedom-to-operate",
                    ),
                )
            ]
            repaired.next_actions = [
                action for action in repaired.next_actions
                if not _contains_any_marker(
                    f"{action.action} {action.rationale}",
                    (
                        "inn-level",
                        "identity",
                        "linkage",
                        "commercial",
                        "access",
                        "product context",
                        "validity",
                        "validity dates",
                        "validity term",
                        "primary commercial",
                        "strength",
                        "patent",
                        "fto",
                        "freedom-to-operate",
                    ),
                )
            ]
            caveat = "Dossier-wide IP/FTO and rights gaps remain in their dedicated blocks; they do not override the EAEU registration-entry conclusion when EAEU-native identity, status, validity, and market-entry linkage are source-backed."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            if any(issue.issue_type == "eaeu_validity_understated_hold" for issue in verification.issues):
                applied_changes.append("fixed_eaeu_validity_understated_hold")
            if any(issue.issue_type == "eaeu_conditional_go_underpromoted" for issue in verification.issues):
                applied_changes.append("promoted_eaeu_conditional_go_from_summary_linkage")
            applied_changes.append("promoted_eaeu_conditional_go_to_go_on_identity_linkage")

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

        if (
            any(issue.issue_type == "asset_dedicated_ip_fto_overconstraint" for issue in verification.issues)
            or self._asset_dedicated_ip_fto_overconstraint(repaired, packet)
        ):
            repaired.verdict = "CONDITIONAL_GO"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "CONDITIONAL_GO — asset fundamentals remain supportable, while IP/FTO remains a dedicated legal-window caveat rather than a primary asset-attractiveness HOLD."
            )
            repaired.full_answer = (
                "The packet carries source-backed regulatory, clinical, or market anchors for the asset, but the FTO/IP layer is still screening-grade. "
                "That uncertainty should continue to drive the dedicated IP/legal-window and next-step blocks instead of collapsing asset attractiveness into HOLD."
            )
            repaired.decision_blockers = [
                blocker for blocker in repaired.decision_blockers
                if not _contains_any_marker(f"{blocker.title} {blocker.rationale}", ("patent", "ip", "fto", "exclusivity", "legal status"))
            ]
            repaired.next_actions = [
                action for action in repaired.next_actions
                if not _contains_any_marker(f"{action.action} {action.rationale}", ("patent", "ip", "fto", "exclusivity", "legal status"))
            ]
            caveat = "IP/FTO remains screening-grade and is handled in the dedicated legal-window block; it should not be treated as the primary asset-attractiveness blocker."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("moved_asset_ip_fto_hold_to_dedicated_legal_window_caveat")

        if (
            any(issue.issue_type == "asset_screening_coverage_overconstraint" for issue in verification.issues)
            or self._asset_screening_coverage_overconstraint(repaired, packet)
        ):
            repaired.decision_blockers = [
                blocker for blocker in repaired.decision_blockers
                if not _contains_any_marker(
                    f"{blocker.title} {blocker.rationale}",
                    (
                        "coverage ledger",
                        "decision readiness",
                        "decision_readiness",
                        "decision-complete",
                        "run_manifest",
                        "expected fields",
                        "dossier incompleteness",
                        "synthesis is yellow",
                        "complete coverage",
                        "commercial evidence is not balanced",
                    ),
                )
            ]
            repaired.next_actions = [
                action for action in repaired.next_actions
                if not _contains_any_marker(
                    f"{action.action} {action.rationale}",
                    (
                        "coverage ledger",
                        "decision readiness",
                        "decision_readiness",
                        "decision-complete",
                        "run_manifest",
                        "expected fields",
                        "dossier incompleteness",
                        "complete coverage",
                        "source-native commercial signal summaries for us and eu",
                    ),
                )
            ]
            if not any(blocker.severity in _BLOCKING_SEVERITIES for blocker in repaired.decision_blockers):
                repaired.verdict = "GO"
                repaired.sufficiency = "SUFFICIENT"
                repaired.confidence = "MEDIUM"
                repaired.short_answer = (
                    "GO — the asset is screening-ready attractive on registration, clinical, and market anchors; operations-ready coverage gaps remain caveats and follow-up items."
                )
            caveat = "Coverage-ledger and operations-ready completeness gaps remain in the sufficiency/legal follow-up blocks; they should not be treated as primary asset-attractiveness blockers for screening."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("moved_asset_coverage_gap_to_screening_caveat")

        if (
            any(issue.issue_type == "asset_conditional_go_underpromoted" for issue in verification.issues)
            or self._asset_conditional_go_underpromoted(repaired, packet)
        ):
            repaired.verdict = "GO"
            repaired.sufficiency = "SUFFICIENT"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "GO — the asset is commercially attractive for screening: registration and market/clinical anchors are present, while IP/FTO and payer operations gaps remain in their dedicated blocks."
            )
            if not repaired.full_answer:
                repaired.full_answer = repaired.short_answer
            caveat = "Operations-ready legal/FTO and payer clearance remains separate from the asset-attractiveness GO."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("promoted_asset_conditional_go_to_go_without_decision_blockers")

        if (
            any(issue.issue_type == "ip_window_closed_without_decision_grade_legal_status" for issue in verification.issues)
            or self._ip_window_closed_without_decision_grade_legal_status(repaired, packet)
        ):
            linkage = _contract_linkage(packet)
            fto = linkage.get("fto_screening_snapshot", {}) or {}
            potential_regions = list(fto.get("potential_blocker_regions") or [])
            repaired.verdict = "LIMITED" if potential_regions else "UNRESOLVED"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "LIMITED — future-dated patent blocker signals are present, but source-native family/legal-event coverage is not decision-grade enough to call the window fully CLOSED."
                if potential_regions
                else "UNRESOLVED — the packet does not provide decision-grade source-native family/legal-event coverage for the IP window."
            )
            repaired.full_answer = (
                "The packet can support a screening-level blocker posture, but it also carries unresolved or conflicted legal-status coverage. "
                "A CLOSED IP-window verdict requires reconciled source-native family status, legal events, and term-extension/SPC/PTE handling. "
                "Until those are decision-grade, the block should remain LIMITED/UNRESOLVED rather than a full closed-window conclusion."
            )
            caveat = "IP-window status is screening-grade until family-by-family legal events and term-extension/SPC/PTE checks are reconciled source-natively."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("downgraded_overclosed_ip_window_to_screening_status")

        if (
            any(issue.issue_type == "ip_window_underresolved_with_source_native_blockers" for issue in verification.issues)
            or self._ip_window_underresolved_with_source_native_blockers(repaired, packet)
        ):
            linkage = _contract_linkage(packet)
            fto = linkage.get("fto_screening_snapshot", {}) or {}
            family_events = linkage.get("family_legal_events_snapshot", {}) or {}
            potential_regions = list(fto.get("potential_blocker_regions") or [])
            refs = list(
                dict.fromkeys(
                    list(fto.get("evidence_refs") or [])
                    + list(family_events.get("evidence_refs") or [])
                )
            )
            repaired.verdict = "LIMITED"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            region_text = ", ".join(potential_regions) if potential_regions else "at least one required jurisdiction"
            repaired.short_answer = (
                f"LIMITED — source-native patent/legal-status evidence supports a screening-level blocker posture in {region_text}, "
                "but family-by-family FTO coverage is not decision-grade enough for OPEN or CLOSED."
            )
            repaired.full_answer = (
                "The packet is no longer a pure unknown: it carries source-native patent expiry, legal-event, no-hit/conflict, or clearance-check evidence. "
                "That supports a LIMITED screening verdict rather than UNRESOLVED. The remaining gap is decision-grade reconciliation: country-level legal status, "
                "claim mapping, and term-extension/SPC/PTE/file-wrapper effects still need attorney-grade review before the window can be called OPEN or CLOSED."
            )
            if refs:
                repaired.why_this_verdict.append(
                    ExecWhyClaim(
                        claim="Source-native patent/legal-event evidence supports a limited IP-window screening posture, while full FTO remains incomplete.",
                        claim_type="hard_evidence_backed",
                        evidence_refs=refs[:6],
                    )
                )
            caveat = "IP-window status is LIMITED at screening level; it is not a full freedom-to-operate opinion or a clean OPEN/CLOSED legal conclusion."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("lifted_underresolved_ip_window_to_limited_screening_status")

        if (
            any(issue.issue_type in {"market_reimbursement_underresolved", "market_reimbursement_overopen"} for issue in verification.issues)
            or self._market_reimbursement_underresolved(repaired, packet)
            or self._market_reimbursement_overopen(repaired, packet)
        ):
            snapshot = _market_reimbursement_snapshot(packet)
            regions = snapshot.get("regions", {}) or {}
            ru_payload = regions.get("RU", {}) or {}
            eaeu_payload = regions.get("EAEU", {}) or {}
            dates = list(ru_payload.get("current_effective_dates") or [])
            effective_text = f" effective {dates[-1]}" if dates else ""
            repaired.verdict = "LIMITED"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "LIMITED — RU source-native price/access evidence is present"
                f"{effective_text}, while EAEU reimbursement remains member-state scoped rather than supported by a single union payer list."
            )
            repaired.full_answer = (
                "The packet now carries structured reimbursement checks. RU Minzdrav price-limit/JNVLP evidence supports a current regulated access/pricing signal, "
                "but that is not the same as complete payer-coverage clearance or restriction analysis. "
                "For EAEU, the defensible reading is member-state scope: non-RU member-state payer/formulary checks remain follow-up items, not a reason to leave the RU window unresolved."
            )
            refs = list(dict.fromkeys(list(snapshot.get("evidence_refs") or [])))
            if refs:
                repaired.why_this_verdict.append(
                    ExecWhyClaim(
                        claim="Structured reimbursement checks support RU source-native price/access evidence and EAEU member-state scope.",
                        claim_type="hard_evidence_backed",
                        evidence_refs=refs[:6],
                    )
                )
            repaired.decision_blockers = [
                blocker for blocker in repaired.decision_blockers
                if not _contains_any_marker(
                    f"{blocker.title} {blocker.rationale}",
                    ("no current source-native ru", "no source-native ru", "single eaeu", "union reimbursement"),
                )
            ]
            caveat = (
                "RU price-limit/JNVLP evidence is a regulated access/pricing signal; payer restrictions and non-RU EAEU member-state coverage still require targeted checks."
            )
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            if bool(eaeu_payload.get("member_state_scope")):
                repaired.next_actions.append(
                    ExecNextAction(
                        action_id="market_reimbursement_member_state_followup",
                        action="Collect member-state payer/formulary sources for BY/KZ/AM/KG if the EAEU commercial window is needed beyond RU.",
                        priority="NEXT",
                        rationale="The current packet closes the false union-level requirement but does not assess non-RU national reimbursement rules.",
                        evidence_refs=list(eaeu_payload.get("evidence_refs") or [])[:4],
                    )
                )
            applied_changes.append("aligned_market_reimbursement_to_limited_screening_status")

        if (
            any(issue.issue_type == "evidence_sufficiency_screening_ready_understated" for issue in verification.issues)
            or self._evidence_sufficiency_screening_ready_understated(repaired, packet)
        ):
            linkage = _contract_linkage(packet)
            source_manifest = linkage.get("source_evidence_manifest", {}) or {}
            reimbursement = _market_reimbursement_snapshot(packet)
            refs = list(
                dict.fromkeys(
                    list((linkage.get("fto_screening_snapshot", {}) or {}).get("evidence_refs") or [])
                    + list((linkage.get("family_legal_events_snapshot", {}) or {}).get("evidence_refs") or [])
                    + list(reimbursement.get("evidence_refs") or [])
                    + list(source_manifest.get("legal_event_evidence_refs") or [])
                    + list(source_manifest.get("rights_evidence_refs") or [])
                )
            )
            repaired.verdict = "PARTIAL"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "PARTIAL — the packet is screening-ready with explicit IP/FTO and payer-breadth limitations, but it is not operations-ready legal or reimbursement clearance."
            )
            repaired.full_answer = (
                "The packet should not stay at pure insufficiency once it has source-enriched screening coverage: source manifests, IP/legal-event checks, RU/EAEU reconciliation, "
                "and reimbursement/payer checks support preliminary decision support. The remaining gaps still matter, but they are operations-ready gaps: full US file-wrapper/terminal-disclaimer "
                "clearance, EU country SPC/lapse/revocation closure, RU/EAEU patent reconciliation, and payer tier/restriction detail."
            )
            if refs:
                repaired.why_this_verdict.append(
                    ExecWhyClaim(
                        claim="Source-enriched IP/FTO and reimbursement checks support screening-ready partial sufficiency, while full operations-ready clearance remains incomplete.",
                        claim_type="hard_evidence_backed",
                        evidence_refs=refs[:6],
                    )
                )
                repaired.top_evidence_refs = list(dict.fromkeys(list(repaired.top_evidence_refs) + refs))[:8]
            repaired.caveats = [
                caveat for caveat in repaired.caveats
                if not _contains_any_marker(
                    caveat,
                    (
                        "partial verdict would only be appropriate",
                        "that condition is not met",
                        "not enough to elevate the package to partial",
                    ),
                )
            ]
            caveat = "Evidence sufficiency is screening-ready only; it is not a legal FTO opinion, payer-tier clearance, or operations-ready launch package."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            repaired.next_actions.append(
                ExecNextAction(
                    action_id="complete_operations_ready_legal_payer_clearance",
                    action="Complete patent-family legal-status reconciliation, file-wrapper/SPC checks, and payer tier/restriction review before treating the dossier as operations-ready.",
                    priority="NEXT",
                    rationale="Screening-grade evidence supports a partial decision package, but commercial launch or paid client legal conclusions need source-native closure of the remaining gaps.",
                    evidence_refs=refs[:6],
                )
            )
            applied_changes.append("lifted_sufficiency_to_screening_ready_partial")

        if (
            any(issue.issue_type == "decision_blockers_screening_sufficiency_understated" for issue in verification.issues)
            or self._decision_blockers_screening_sufficiency_understated(repaired, packet)
        ):
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM" if repaired.confidence == "LOW" else repaired.confidence
            caveat = "Decision blockers are classified at screening level; source-native IP evidence exists, but operations-ready legal/FTO reconciliation remains incomplete."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("lifted_decision_blockers_to_screening_partial")

        if (
            any(issue.issue_type == "generic_screening_sufficiency_understated" for issue in verification.issues)
            or self._generic_screening_sufficiency_understated(repaired, packet)
        ):
            repaired.verdict = "LOW"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "LOW" if repaired.confidence == "LOW" else "MEDIUM"
            repaired.short_answer = (
                "LOW — source-native legal screening evidence exists, but active/pending patent coverage and missing positive-gate closure keep generic opportunity weak rather than absent."
            )
            caveat = "Generic opportunity is not positively evidenced, but source-native legal screening evidence exists; the gap is positive-gate closure, not absence of evidence."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("lifted_generic_not_evidenced_to_screening_partial")

        if (
            any(issue.issue_type == "portfolio_screening_underpromoted" for issue in verification.issues)
            or self._portfolio_screening_underpromoted(repaired, packet)
        ):
            repaired.verdict = "MEDIUM"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "MEDIUM — portfolio value is supported at screening level by registration and clinical/market anchors, while IP/legal-status reconciliation remains a caveat rather than a reason to collapse the opportunity to LOW."
            )
            caveat = "Portfolio opportunity remains screening-grade until jurisdiction-level legal-status and payer evidence are reconciled."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("lifted_portfolio_low_to_screening_medium")

        if (
            any(issue.issue_type == "portfolio_not_evidenced_understated" for issue in verification.issues)
            or self._portfolio_not_evidenced_understated(repaired, packet)
        ):
            linkage = _contract_linkage(packet)
            source_manifest = linkage.get("source_evidence_manifest", {}) or {}
            refs = list(
                dict.fromkeys(
                    list((linkage.get("family_legal_events_snapshot", {}) or {}).get("evidence_refs") or [])
                    + list((linkage.get("fto_screening_snapshot", {}) or {}).get("evidence_refs") or [])
                    + list(source_manifest.get("legal_event_evidence_refs") or [])
                    + list(_explicit_no_registration_refs(packet, "EAEU"))
                    + list(repaired.top_evidence_refs)
                )
            )
            repaired.verdict = "LOW"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM" if _operations_screening_ready(packet) else "LOW"
            repaired.short_answer = (
                "LOW — the packet has screening-level US/EU, clinical, and IP/legal-status anchors, but RU/EAEU registration and commercial-access gaps keep the portfolio opportunity weak rather than absent."
            )
            repaired.full_answer = (
                "Portfolio opportunity should not collapse to NOT_EVIDENCED once the packet contains source-native registration/status, clinical, and patent/legal-event screening evidence. "
                "The current posture is still weak because RU/EAEU entry is not available, commercial signals are thin, and legal/FTO coverage is not operations-ready. "
                "Those facts support LOW/PARTIAL rather than a claim that the opportunity is unevidenced."
            )
            if refs:
                repaired.top_evidence_refs = refs[:8]
                repaired.why_this_verdict.append(
                    ExecWhyClaim(
                        claim="Source-enriched jurisdiction, clinical, and IP/legal-status evidence supports a low screening-level portfolio opportunity rather than a pure no-evidence posture.",
                        claim_type="hard_evidence_backed",
                        evidence_refs=refs[:6],
                    )
                )
            caveat = "Portfolio opportunity is screening-grade and low because RU/EAEU registration/commercial gaps and legal/FTO limitations remain material."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("lifted_portfolio_not_evidenced_to_low_screening")

        if (
            any(issue.issue_type == "key_risks_not_evidenced_understated" for issue in verification.issues)
            or self._key_risks_not_evidenced_understated(repaired, packet)
        ):
            linkage = _contract_linkage(packet)
            refs = list(
                dict.fromkeys(
                    list((linkage.get("fto_screening_snapshot", {}) or {}).get("evidence_refs") or [])
                    + list((linkage.get("family_legal_events_snapshot", {}) or {}).get("evidence_refs") or [])
                    + list((linkage.get("source_evidence_manifest", {}) or {}).get("legal_event_evidence_refs") or [])
                    + list(repaired.top_evidence_refs)
                )
            )
            repaired.verdict = "HIGH"
            repaired.sufficiency = "PARTIAL"
            repaired.confidence = "MEDIUM"
            repaired.short_answer = (
                "HIGH — source-native IP/legal-status screening evidence shows active or pending blocker exposure and unresolved national-status gaps; this is a risk-positive record, not NOT_EVIDENCED."
            )
            repaired.full_answer = (
                "The packet contains enough patent/legal-status evidence to identify material execution risk, while still lacking attorney-grade family-by-family clearance. "
                "Therefore the risk block should be HIGH/PARTIAL: risks are evidenced for screening, but final enforceability/FTO and country-level reconciliation remain follow-up work."
            )
            if refs:
                repaired.top_evidence_refs = list(dict.fromkeys(refs + list(repaired.top_evidence_refs)))[:8]
                repaired.why_this_verdict.append(
                    ExecWhyClaim(
                        claim="Patent/legal-status screening evidence supports a high risk posture while full legal/FTO reconciliation remains incomplete.",
                        claim_type="hard_evidence_backed",
                        evidence_refs=refs[:6],
                    )
                )
            caveat = "Risk severity is screening-grade; it is not a formal legal enforceability or FTO opinion."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            applied_changes.append("lifted_key_risks_not_evidenced_to_high_screening")

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

        if self._rf_underlinked_conditional_go(repaired, packet):
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
                f"The packet carries explicit market-entry linkage showing that RU commercial/formulary/procurement evidence maps to {linkage_phrase}. "
                "Residual IP/FTO and payer-breadth gaps remain in their dedicated blocks rather than blocking RU registration entry."
            )
            repaired.top_evidence_refs = list(dict.fromkeys(list(linkage.get("evidence_refs") or []) + list(repaired.top_evidence_refs)))[:8]
            caveat = "RU access evidence is treated as product-context-linked rather than a pure INN-level proxy."
            if caveat not in repaired.caveats:
                repaired.caveats.append(caveat)
            repaired.decision_blockers = []
            repaired.next_actions = []
            if "promoted_rf_conditional_go_to_go_on_identity_linkage" not in applied_changes:
                applied_changes.append("promoted_rf_conditional_go_to_go_on_identity_linkage")

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
            "rf_no_record_understated",
            "eaeu_holdable_position",
            "eaeu_no_record_understated",
            "rf_identity_underlink",
            "eaeu_same_id_overconstraint",
            "eaeu_identity_underlink",
            "eaeu_validity_understated_hold",
            "eaeu_conditional_go_underpromoted",
            "asset_ip_window_overconstraint",
            "asset_dedicated_ip_fto_overconstraint",
            "asset_screening_coverage_overconstraint",
            "asset_conditional_go_underpromoted",
            "ip_window_closed_without_decision_grade_legal_status",
            "ip_window_underresolved_with_source_native_blockers",
            "market_reimbursement_underresolved",
            "market_reimbursement_overopen",
            "evidence_sufficiency_screening_ready_understated",
            "generic_screening_sufficiency_understated",
            "decision_blockers_screening_sufficiency_understated",
            "portfolio_screening_underpromoted",
            "portfolio_not_evidenced_understated",
            "key_risks_not_evidenced_understated",
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
                self._rf_no_record_understated(block, packet),
                self._eaeu_holdable_regulatory_position(block, packet),
                self._eaeu_no_record_understated(block, packet),
                self._rf_underlinked_conditional_go(block, packet),
                self._eaeu_same_id_overconstraint(block, packet),
                self._eaeu_underlinked_conditional_go(block, packet),
                self._eaeu_validity_understated_hold(block, packet),
                self._eaeu_conditional_go_underpromoted(block, packet),
                self._asset_ip_window_overconstraint(block, packet),
                self._asset_dedicated_ip_fto_overconstraint(block, packet),
                self._asset_screening_coverage_overconstraint(block, packet),
                self._asset_conditional_go_underpromoted(block, packet),
                self._ip_window_closed_without_decision_grade_legal_status(block, packet),
                self._market_reimbursement_underresolved(block, packet),
                self._market_reimbursement_overopen(block, packet),
                self._evidence_sufficiency_screening_ready_understated(block, packet),
                self._generic_screening_sufficiency_understated(block, packet),
                self._decision_blockers_screening_sufficiency_understated(block, packet),
                self._portfolio_screening_underpromoted(block, packet),
                self._portfolio_not_evidenced_understated(block, packet),
                self._key_risks_not_evidenced_understated(block, packet),
                self._regional_generic_collapse(block, packet),
                self._regional_licensing_collapse(block, packet),
                self._business_block_synthesis_overconstraint(block, packet),
            )
        )
        if (verification.overall_status != "FAIL" and not has_reparable_warn and not needs_policy_repair) or not allow_repair:
            return block, verification
        repaired_block, repaired_verification = self.repair_block(block, packet, verification)
        return repaired_block, repaired_verification
