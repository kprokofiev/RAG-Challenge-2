"""
Claim Builder — Sprint 22 WS2-D6
===================================
Hybrid claim builder:
  - Deterministic skeleton from structured facts + evidence
  - Optional LLM refinement for wording / conflict summarization

Produces AnswerFrame — the structured truth object that exec_writer
turns into human-readable answers.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from src.registration_truth import (
    VERDICT_CONFIRMED,
    VERDICT_PARTIAL,
    infer_registration_verdict,
    registration_status_role,
)

logger = logging.getLogger(__name__)

_ONGOING_STATUSES = {
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
}
_PATENT_EXPIRY_REGION_PREFIXES = {
    "US": ("US:", "USA:", "UNITED STATES:"),
    "EU": ("EP:", "EU:", "EPO:", "EUROPE:"),
}


# ── Models ──────────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """Single factual claim with evidence backing."""
    claim_id: str
    text: str
    jurisdiction: Optional[str] = None
    semantic_role: Optional[str] = None
    support_level: str  # strong | moderate | weak | unsupported
    support_fields: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)


class AnswerFrame(BaseModel):
    """Structured truth object for an executive answer."""
    question_id: str
    question_text: str
    scope: Dict[str, Any] = Field(default_factory=dict)
    claims: List[Claim] = Field(default_factory=list)
    unknowns: List[Dict[str, Any]] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Dict[str, Any] = Field(default_factory=dict)
    business_implication: List[str] = Field(default_factory=list)
    recommended_next_actions: List[str] = Field(default_factory=list)


# ── Deterministic claim builders per question type ──────────────────────────

def _build_registration_claims(
    dossier: Dict[str, Any],
    evidence_pack,  # EvidencePack
    resolved_scope,  # ResolvedScope
) -> List[Claim]:
    """Build claims for registration_status questions."""
    claims = []
    registrations = dossier.get("registrations", [])

    for reg in registrations:
        region = reg.get("region", "unknown")
        status_ev = reg.get("status")
        status_val = _ev_value(status_ev)
        status_refs = _ev_refs(status_ev)

        mah_ev = reg.get("mah")
        mah_val = _ev_value(mah_ev)

        identifiers = reg.get("identifiers", [])
        id_strs = [_ev_value(i) for i in identifiers if _ev_value(i)]
        verdict = str(
            reg.get("verdict")
            or infer_registration_verdict(
                status=status_ev,
                mah=mah_ev,
                identifiers=identifiers,
                forms_strengths=reg.get("forms_strengths"),
            )
        ).strip().lower()

        if verdict == VERDICT_CONFIRMED and status_val:
            text = f"{region}: {status_val}"
            if mah_val:
                text += f" (MAH: {mah_val})"
            if id_strs:
                text += f" [{', '.join(id_strs[:3])}]"

            semantic_role = _classify_registration_status(status_val)
            support = "strong" if status_refs else "weak"
            claims.append(Claim(
                claim_id=_claim_id(f"reg_{region}"),
                text=text,
                jurisdiction=region,
                semantic_role=semantic_role,
                support_level=support,
                support_fields=[f"registrations.{region}.status"],
                evidence_refs=status_refs,
            ))
        elif verdict == VERDICT_PARTIAL:
            parts = [f"{region}: Partial registration evidence"]
            if mah_val:
                parts.append(f"MAH: {mah_val}")
            if id_strs:
                parts.append(f"IDs: {', '.join(id_strs[:3])}")
            form_values = [_ev_value(item) for item in (reg.get("forms_strengths") or []) if _ev_value(item)]
            if form_values:
                parts.append(f"forms: {', '.join(form_values[:3])}")
            evidence_refs = _collect_evidence_refs(
                status_ev,
                mah_ev,
                identifiers,
                reg.get("forms_strengths"),
                reg.get("evidence_refs"),
            )
            claims.append(Claim(
                claim_id=_claim_id(f"reg_{region}_partial"),
                text=" | ".join(parts),
                jurisdiction=region,
                semantic_role="registration_partial",
                support_level="strong" if evidence_refs else "moderate",
                support_fields=[f"registrations.{region}.verdict", f"registrations.{region}.identifiers"],
                evidence_refs=evidence_refs,
            ))
        elif status_val and registration_status_role(status_val) == "negative":
            claims.append(Claim(
                claim_id=_claim_id(f"reg_{region}_negative"),
                text=f"{region}: {status_val}",
                jurisdiction=region,
                semantic_role="registration_negative",
                support_level="strong" if status_refs else "moderate",
                support_fields=[f"registrations.{region}.status"],
                evidence_refs=status_refs,
            ))
        else:
            claims.append(Claim(
                claim_id=_claim_id(f"reg_{region}_missing"),
                text=f"{region}: Registration status unknown",
                jurisdiction=region,
                semantic_role="registration_unknown",
                support_level="unsupported",
                support_fields=[f"registrations.{region}.status"],
                evidence_refs=[],
            ))

    return claims


def _build_ip_claims(
    dossier: Dict[str, Any],
    evidence_pack,
    resolved_scope,
) -> List[Claim]:
    """Build claims for IP / patent questions."""
    claims = []
    families = dossier.get("patent_families", [])

    for fam in families:
        fam_id = fam.get("family_id", "unknown")
        rep_pub = _ev_value(fam.get("representative_pub"))
        what_blocks = _ev_value(fam.get("what_blocks"))
        legal_status = _ev_value(fam.get("legal_status_snapshot"))
        tech_focus = _ev_value(fam.get("technical_focus"))

        expiry_list = fam.get("expiry_by_country", [])
        expiry_strs = [_ev_value(e) for e in expiry_list if _ev_value(e)]

        refs = fam.get("evidence_refs", [])

        text_parts = []
        if rep_pub:
            text_parts.append(rep_pub)
        if what_blocks:
            text_parts.append(f"blocks: {what_blocks}")
        if tech_focus:
            text_parts.append(f"focus: {tech_focus}")
        if legal_status:
            text_parts.append(f"status: {legal_status}")
        if expiry_strs:
            text_parts.append(f"expiry: {'; '.join(expiry_strs[:3])}")

        if text_parts:
            claims.append(Claim(
                claim_id=_claim_id(f"pat_{fam_id}"),
                text=" | ".join(text_parts),
                support_level="strong" if refs else "moderate",
                support_fields=[f"patent_families.{fam_id}"],
                evidence_refs=refs,
            ))

    if not families:
        claims.append(Claim(
            claim_id=_claim_id("pat_none"),
            text="No patent families found in dossier",
            support_level="unsupported",
            support_fields=["patent_families"],
            evidence_refs=[],
        ))

    return claims


def _build_clinical_claims(
    dossier: Dict[str, Any],
    evidence_pack,
    resolved_scope,
) -> List[Claim]:
    """Build claims for clinical evidence questions."""
    claims = []
    studies = dossier.get("clinical_studies", [])

    for study in studies:
        title = _ev_value(study.get("title"))
        phase = _ev_value(study.get("phase"))
        n_enrolled = _ev_value(study.get("n_enrolled"))
        conclusion = _ev_value(study.get("conclusion"))
        status = _ev_value(study.get("status"))
        refs = study.get("evidence_refs", [])

        text_parts = []
        if title:
            text_parts.append(title[:100])
        if phase:
            text_parts.append(f"Phase {phase}")
        if n_enrolled:
            text_parts.append(f"N={n_enrolled}")
        if status:
            text_parts.append(f"Status: {status}")
        if conclusion:
            text_parts.append(f"Conclusion: {conclusion[:150]}")

        if text_parts:
            claims.append(Claim(
                claim_id=_claim_id(f"clin_{title or 'study'}"),
                text=" | ".join(text_parts),
                support_level="strong" if refs else "moderate",
                support_fields=["clinical_studies"],
                evidence_refs=refs,
            ))

    return claims


def _build_ongoing_trials_claims(
    dossier: Dict[str, Any],
    evidence_pack,
    resolved_scope,
) -> List[Claim]:
    """Build claims strictly for ongoing trials, not the whole historical inventory."""
    claims: List[Claim] = []
    studies = dossier.get("clinical_studies", [])
    status_counts: Counter[str] = Counter()
    fallback_refs: List[str] = []

    for study in studies:
        title = _ev_value(study.get("title"))
        phase = _ev_value(study.get("phase"))
        n_enrolled = _ev_value(study.get("n_enrolled"))
        status = (_ev_value(study.get("status")) or "").strip()
        status_upper = status.upper()
        conclusion = _ev_value(study.get("conclusion"))
        is_ongoing = _ev_bool(study.get("is_ongoing"))
        refs = _collect_evidence_refs(
            study.get("title"),
            study.get("phase"),
            study.get("status"),
            study.get("is_ongoing"),
            study.get("n_enrolled"),
            study.get("conclusion"),
            study.get("evidence_refs"),
        )

        if is_ongoing is True or status_upper in _ONGOING_STATUSES:
            text_parts = []
            if title:
                text_parts.append(title[:100])
            if phase:
                text_parts.append(f"Phase {phase}")
            if n_enrolled:
                text_parts.append(f"N={n_enrolled}")
            if status:
                text_parts.append(f"Status: {status}")
            if conclusion:
                text_parts.append(f"Studying: {conclusion[:150]}")

            claims.append(Claim(
                claim_id=_claim_id(f"ongoing_{title or status or 'study'}"),
                text=" | ".join(text_parts) if text_parts else "Ongoing trial identified",
                semantic_role="ongoing_trial",
                support_level="strong" if refs else "moderate",
                support_fields=["clinical_studies.is_ongoing", "clinical_studies.status"],
                evidence_refs=refs,
            ))
        else:
            status_counts[status_upper or "UNKNOWN"] += 1
            fallback_refs.extend(refs[:2])

    if claims:
        return claims

    refs = list(dict.fromkeys(fallback_refs))
    status_summary = ", ".join(
        f"{status.title()} ({count})" for status, count in status_counts.most_common(3)
    ) or "no status breakdown available"
    claims.append(Claim(
        claim_id=_claim_id("ongoing_none_identified"),
        text=(
            "No ongoing clinical trials were identified in the current dossier; "
            f"indexed registry studies are mainly {status_summary}."
        ),
        semantic_role="ongoing_none_identified",
        support_level="strong" if refs else "moderate",
        support_fields=["clinical_studies.status", "clinical_studies.is_ongoing"],
        evidence_refs=refs,
    ))
    return claims


def _build_chemistry_claims(
    dossier: Dict[str, Any],
    evidence_pack,
    resolved_scope,
) -> List[Claim]:
    """Build claims for chemistry identity questions."""
    claims = []
    passport = dossier.get("passport", {})

    fields = [
        ("SMILES", "smiles"),
        ("Chemical Formula", "chemical_formula"),
        ("Molecular Weight", "molecular_weight"),
        ("InChI Key", "inchi_key"),
    ]

    for label, key in fields:
        ev = passport.get(key)
        val = _ev_value(ev)
        refs = _ev_refs(ev)
        if val:
            claims.append(Claim(
                claim_id=_claim_id(f"chem_{key}"),
                text=f"{label}: {val}",
                support_level="strong" if refs else "weak",
                support_fields=[f"passport.{key}"],
                evidence_refs=refs,
            ))
        else:
            claims.append(Claim(
                claim_id=_claim_id(f"chem_{key}_missing"),
                text=f"{label}: Not available",
                support_level="unsupported",
                support_fields=[f"passport.{key}"],
                evidence_refs=[],
            ))

    return claims


def _build_patent_expiry_claims(
    dossier: Dict[str, Any],
    evidence_pack,
    resolved_scope,
) -> List[Claim]:
    """Build expiry claims only for jurisdictions explicitly asked by the question."""
    claims: List[Claim] = []
    families = dossier.get("patent_families", [])
    target_regions = {"US", "EU"}

    for fam in families:
        fam_id = fam.get("family_id", "unknown")
        rep_pub = _ev_value(fam.get("representative_pub")) or fam_id
        legal_status = _ev_value(fam.get("legal_status_snapshot"))
        expiry_hits = _extract_patent_expiry_hits(fam.get("expiry_by_country", []), target_regions)
        if not expiry_hits:
            continue

        expiry_texts = [hit["value"] for hit in expiry_hits if hit.get("value")]
        refs = _collect_evidence_refs(
            fam.get("representative_pub"),
            fam.get("legal_status_snapshot"),
            fam.get("evidence_refs"),
            [hit.get("refs", []) for hit in expiry_hits],
        )

        text_parts = [rep_pub]
        if legal_status:
            text_parts.append(f"status: {legal_status}")
        if expiry_texts:
            text_parts.append(f"expiry: {'; '.join(expiry_texts[:3])}")

        claims.append(Claim(
            claim_id=_claim_id(f"pat_expiry_{fam_id}"),
            text=" | ".join(text_parts),
            semantic_role="patent_expiry",
            support_level="strong" if refs else "moderate",
            support_fields=["patent_families.expiry_by_country"],
            evidence_refs=refs,
        ))

    if not claims:
        claims.append(Claim(
            claim_id=_claim_id("pat_expiry_missing"),
            text="No explicit US/EU patent expiry entries were identified in the dossier.",
            semantic_role="patent_expiry_missing",
            support_level="unsupported",
            support_fields=["patent_families.expiry_by_country"],
            evidence_refs=[],
        ))

    return claims


def _build_synthesis_overview_claims(
    dossier: Dict[str, Any],
    evidence_pack,
    resolved_scope,
) -> List[Claim]:
    """
    Separate true API-route evidence from formulation/manufacturing steps.

    This prevents drug-product preparation snippets from being presented as a
    verified API synthesis route.
    """
    claims: List[Claim] = []
    steps = dossier.get("synthesis_steps", [])
    families = dossier.get("patent_families", [])

    api_steps = []
    non_api_steps = []
    for step in steps:
        if _normalize_step_kind(step) == "api_synthesis":
            api_steps.append(step)
        else:
            non_api_steps.append(step)

    process_families = []
    for fam in families:
        tech_focus = (_ev_value(fam.get("technical_focus")) or "").strip().lower()
        process_rel = (_ev_value(fam.get("process_relevance")) or "").strip().lower()
        if tech_focus in {"process_manufacturing", "intermediate_synthesis"} or process_rel in {"moderate", "strong"}:
            process_families.append(fam)

    for idx, step in enumerate(api_steps[:3], 1):
        desc = _ev_value(step.get("description"))
        refs = _collect_evidence_refs(
            step.get("description"),
            step.get("reagents"),
            step.get("intermediates"),
            step.get("evidence_refs"),
        )
        if desc:
            claims.append(Claim(
                claim_id=_claim_id(f"api_synth_step_{idx}_{desc[:40]}"),
                text=f"API synthesis step {idx}: {desc[:220]}",
                semantic_role="api_synthesis_step",
                support_level="strong" if refs else "moderate",
                support_fields=["synthesis_steps.description"],
                evidence_refs=refs,
            ))

    for fam in process_families[:2]:
        rep_pub = _ev_value(fam.get("representative_pub")) or fam.get("family_id", "unknown")
        tech_focus = _ev_value(fam.get("technical_focus"))
        process_rel = _ev_value(fam.get("process_relevance"))
        refs = _collect_evidence_refs(
            fam.get("representative_pub"),
            fam.get("technical_focus"),
            fam.get("process_relevance"),
            fam.get("evidence_refs"),
        )
        claims.append(Claim(
            claim_id=_claim_id(f"process_family_{rep_pub}"),
            text=(
                f"Process-relevant patent family identified: {rep_pub}"
                + (f" | focus: {tech_focus}" if tech_focus else "")
                + (f" | process relevance: {process_rel}" if process_rel else "")
            ),
            semantic_role="process_relevant_patent",
            support_level="strong" if refs else "moderate",
            support_fields=["patent_families.process_relevance", "patent_families.technical_focus"],
            evidence_refs=refs,
        ))

    if not api_steps and process_families:
        refs = _collect_evidence_refs(
            [fam.get("representative_pub") for fam in process_families[:3]],
            [fam.get("technical_focus") for fam in process_families[:3]],
            [fam.get("process_relevance") for fam in process_families[:3]],
        )
        claims.append(Claim(
            claim_id=_claim_id("api_route_not_verified"),
            text=(
                "Process-relevant patents are present, but a directly verified API synthesis route "
                "was not assembled from the current corpus."
            ),
            semantic_role="api_synthesis_not_verified",
            support_level="strong" if refs else "moderate",
            support_fields=["synthesis_steps.kind", "patent_families.process_relevance"],
            evidence_refs=refs,
        ))

    if non_api_steps:
        refs = _collect_evidence_refs(
            [step.get("description") for step in non_api_steps[:3]],
            [step.get("evidence_refs") for step in non_api_steps[:3]],
        )
        claims.append(Claim(
            claim_id=_claim_id("non_api_synthesis_evidence"),
            text=(
                "Extracted stepwise evidence currently describes formulation/manufacturing operations, "
                "not a directly verified API synthesis route."
            ),
            semantic_role="non_api_process_only",
            support_level="strong" if refs else "moderate",
            support_fields=["synthesis_steps.kind", "synthesis_steps.description"],
            evidence_refs=refs,
        ))

    if not claims:
        claims.append(Claim(
            claim_id=_claim_id("synthesis_no_evidence"),
            text="No process or synthesis evidence was identified in the dossier.",
            semantic_role="synthesis_unknown",
            support_level="unsupported",
            support_fields=["synthesis_steps", "patent_families.process_relevance"],
            evidence_refs=[],
        ))

    return claims


def _build_generic_claims(
    dossier: Dict[str, Any],
    evidence_pack,
    resolved_scope,
) -> List[Claim]:
    """Build generic claims from structured facts in evidence pack."""
    claims = []
    for fact in evidence_pack.structured_facts:
        val = fact.get("value")
        if val is not None:
            claims.append(Claim(
                claim_id=_claim_id(f"fact_{fact['field_path']}"),
                text=f"{fact['field_path']}: {_format_value(val)}",
                support_level="moderate",
                support_fields=[fact["field_path"]],
                evidence_refs=_extract_refs_from_raw(fact.get("raw")),
            ))
    return claims


def _build_data_quality_claims(
    dossier: Dict[str, Any],
    evidence_pack,
    resolved_scope,
) -> List[Claim]:
    """Build completeness / readiness claims from dossier-quality summaries."""
    claims = []
    quality = dossier.get("dossier_quality_v2") or {}
    coverage = quality.get("coverage") or {}
    readiness = quality.get("decision_readiness") or {}
    critical_unknowns = quality.get("critical_unknowns") or []
    notes = quality.get("notes") or []
    all_unknowns = dossier.get("unknowns") or []
    coverage_ledger = dossier.get("coverage_ledger") or {}

    if readiness:
        readiness_parts = [f"{k}={v}" for k, v in readiness.items()]
        claims.append(Claim(
            claim_id=_claim_id("quality_readiness"),
            text=f"Decision readiness gates: {'; '.join(readiness_parts)}",
            semantic_role="quality_readiness",
            support_level="moderate",
            support_fields=["dossier_quality_v2.decision_readiness"],
            evidence_refs=[],
        ))

    if coverage:
        coverage_parts = [f"{k}={round(float(v) * 100):d}%" for k, v in coverage.items()]
        claims.append(Claim(
            claim_id=_claim_id("quality_coverage"),
            text=f"Section coverage: {'; '.join(coverage_parts)}",
            semantic_role="quality_coverage",
            support_level="moderate",
            support_fields=["dossier_quality_v2.coverage"],
            evidence_refs=[],
        ))

    if critical_unknowns:
        impacts = []
        total_count = 0
        for item in critical_unknowns:
            total_count += int(item.get("count", 0) or 0)
            impact = str(item.get("impact") or "").strip()
            if impact:
                impacts.append(impact)
        impact_text = "; ".join(impacts[:3]) if impacts else "critical coverage issues remain"
        claims.append(Claim(
            claim_id=_claim_id("quality_critical_unknowns"),
            text=f"Critical unknowns: {total_count or len(critical_unknowns)} item(s) flagged; impact: {impact_text}",
            semantic_role="quality_critical_unknowns",
            support_level="moderate",
            support_fields=["dossier_quality_v2.critical_unknowns"],
            evidence_refs=[],
        ))

    claims.append(Claim(
        claim_id=_claim_id("quality_unknown_count"),
        text=f"Outstanding dossier unknowns: {len(all_unknowns)}",
        semantic_role="quality_unknown_count",
        support_level="moderate",
        support_fields=["unknowns"],
        evidence_refs=[],
    ))

    ledger_summary = coverage_ledger.get("summary")
    if isinstance(ledger_summary, dict) and ledger_summary:
        summary_parts = []
        for key in ("total_declared_sources", "total_attached_sources", "total_indexed_sources", "passport_pct"):
            if key in ledger_summary:
                summary_parts.append(f"{key}={ledger_summary[key]}")
        if summary_parts:
            claims.append(Claim(
                claim_id=_claim_id("quality_coverage_ledger"),
                text=f"Coverage ledger summary: {'; '.join(summary_parts)}",
                semantic_role="quality_coverage_ledger",
                support_level="moderate",
                support_fields=["coverage_ledger.summary"],
                evidence_refs=[],
            ))
    elif notes:
        claims.append(Claim(
            claim_id=_claim_id("quality_notes"),
            text=f"Quality notes: {'; '.join(str(n) for n in notes[:2])}",
            semantic_role="quality_notes",
            support_level="weak",
            support_fields=["dossier_quality_v2.notes"],
            evidence_refs=[],
        ))

    return claims


# ── Claim builder dispatch ──────────────────────────────────────────────────

_CLAIM_BUILDERS = {
    "registration_status": _build_registration_claims,
    "ip_fto_risk": _build_ip_claims,
    "clinical_evidence": _build_clinical_claims,
    "chemistry_identity": _build_chemistry_claims,
    "synthesis_manufacturing": _build_generic_claims,
    "commercial_assessment": _build_generic_claims,
    "data_quality": _build_data_quality_claims,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _ev_value(ev: Any) -> Optional[str]:
    """Extract string value from EvidencedValue."""
    if ev is None:
        return None
    if isinstance(ev, dict):
        v = ev.get("value")
        return str(v) if v is not None else None
    return str(ev)


def _ev_refs(ev: Any) -> List[str]:
    """Extract evidence_refs from EvidencedValue."""
    if isinstance(ev, dict):
        return ev.get("evidence_refs", [])
    return []


def _ev_bool(ev: Any) -> Optional[bool]:
    """Extract a boolean value from EvidencedValue-ish payloads."""
    if ev is None:
        return None
    value = ev.get("value") if isinstance(ev, dict) else ev
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _collect_evidence_refs(*items: Any) -> List[str]:
    """Collect and deduplicate evidence refs from nested dossier structures."""
    refs: List[str] = []

    def _visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            refs.extend(item.get("evidence_refs", []))
            evidence_id = item.get("evidence_id")
            if evidence_id:
                refs.append(evidence_id)
            return
        if isinstance(item, list):
            for sub in item:
                _visit(sub)
            return
        if isinstance(item, str) and item.startswith(("ev_", "ctgov_api_")):
            refs.append(item)

    for item in items:
        _visit(item)
    return list(dict.fromkeys(refs))


def _classify_patent_expiry_region(expiry_value: Optional[str]) -> Optional[str]:
    text = (expiry_value or "").strip().upper()
    for region, prefixes in _PATENT_EXPIRY_REGION_PREFIXES.items():
        if any(text.startswith(prefix) for prefix in prefixes):
            return region
    return None


def _extract_patent_expiry_hits(expiry_list: Any, target_regions: Optional[set] = None) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for item in expiry_list or []:
        value = _ev_value(item)
        region = _classify_patent_expiry_region(value)
        if not region:
            continue
        if target_regions and region not in target_regions:
            continue
        hits.append({
            "region": region,
            "value": value,
            "refs": _collect_evidence_refs(item),
        })
    return hits


def _normalize_step_kind(step: Dict[str, Any]) -> str:
    """Use stored step kind when present, else infer a safer non-API classification from text."""
    kind = (step.get("kind") or "unknown").strip().lower()
    if kind and kind != "unknown":
        return kind

    text = (_ev_value(step.get("description")) or "").strip().lower()
    formulation_markers = (
        "sterile water",
        "propylene glycol",
        "batch volume",
        "sterile injection",
        "steriliz",
        "formulation",
        "solution",
        "suspension",
        "tablet",
    )
    api_markers = (
        "intermediate",
        "cycliz",
        "coupl",
        "react",
        "condens",
        "protected",
        "deprotected",
    )
    if any(marker in text for marker in formulation_markers):
        return "formulation_process"
    if any(marker in text for marker in api_markers):
        return "api_synthesis"
    return "unknown"


def _claim_id(seed: str) -> str:
    """Generate deterministic claim ID."""
    h = hashlib.md5(seed.encode()).hexdigest()[:8]
    return f"clm_{h}"


def _format_value(val: Any) -> str:
    """Format a value for display in a claim."""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val[:5])
    return str(val)[:300]


def _extract_refs_from_raw(raw: Any) -> List[str]:
    """Extract all evidence_refs from raw value."""
    if isinstance(raw, dict):
        return raw.get("evidence_refs", [])
    if isinstance(raw, list):
        refs = []
        for item in raw:
            if isinstance(item, dict):
                refs.extend(item.get("evidence_refs", []))
        return refs
    return []


def _classify_registration_status(status_val: str) -> str:
    """
    Separate positive registration outcomes from verified absence / negative outcomes.

    This keeps business implications honest: a verified negative lookup such as
    "No public registration record verified" must not be counted as
    "registered in jurisdiction X".
    """
    role = registration_status_role(status_val)
    if role == "negative":
        return "registration_negative"
    if role == "positive":
        return "registration_positive"
    if role == "unknown":
        return "registration_unknown"

    s = (status_val or "").strip().lower()
    if not s:
        return "registration_unknown"

    negative_markers = (
        "no public registration record verified",
        "no public record verified",
        "not registered",
        "not approved",
        "не зарегистр",
        "нет публичной записи",
        "not found after lookup",
        "no registration record found",
        "no eu registration record verified",
    )
    positive_markers = (
        "approved",
        "registered",
        "marketing authorisation granted",
        "marketing authorization granted",
        "authorised",
        "authorized",
        "valid",
        "active",
    )

    if any(marker in s for marker in negative_markers):
        return "registration_negative"
    if any(marker in s for marker in positive_markers):
        return "registration_positive"
    return "registration_unknown"


# ── Main ClaimBuilder ──────────────────────────────────────────────────────

class ClaimBuilder:
    """
    Hybrid claim builder:
    - Deterministic skeleton from dossier fields
    - Optional LLM refinement (not used in V1 Sprint 22 for claim generation)
    """

    def __init__(self, api_processor=None):
        """
        Args:
            api_processor: Optional APIProcessor for LLM refinement.
                           Not used in V1 for claim building itself.
        """
        self.api = api_processor

    def build(
        self,
        routed_question,   # RoutedQuestion
        resolved_scope,    # ResolvedScope
        coverage_decision,  # CoverageDecision
        evidence_pack,      # EvidencePack
        dossier: Dict[str, Any],
    ) -> AnswerFrame:
        """
        Build AnswerFrame from evidence.

        Steps:
        1. Select claim builder based on question_type.
        2. Build claims deterministically.
        3. Collect unknowns relevant to this question.
        4. Detect conflicts.
        5. Compute confidence.
        6. Derive business implications and next actions.
        """
        builder_fn = self._select_builder(routed_question)

        # Build claims
        claims = builder_fn(dossier, evidence_pack, resolved_scope)

        # Collect relevant unknowns
        unknowns = self._collect_relevant_unknowns(dossier, routed_question)
        unknowns.extend(
            self._derive_question_unknowns(
                routed_question=routed_question,
                dossier=dossier,
                claims=claims,
                resolved_scope=resolved_scope,
            )
        )
        unknowns = self._dedupe_unknowns(unknowns)

        # Detect conflicts
        conflicts = self._detect_conflicts(claims)

        # Compute confidence
        confidence = self._compute_confidence(
            claims, unknowns, coverage_decision, routed_question
        )

        # Business implications
        implications = self._derive_implications(claims, unknowns, routed_question)

        # Next actions
        next_actions = self._derive_next_actions(
            coverage_decision, unknowns, routed_question
        )

        return AnswerFrame(
            question_id=routed_question.question_id,
            question_text=routed_question.question_text,
            scope={
                "entity_mode": resolved_scope.entity_mode,
                "selected_contexts": resolved_scope.selected_context_ids,
                "jurisdictions": resolved_scope.jurisdiction_map,
                "warnings": resolved_scope.scope_warnings,
            },
            claims=claims,
            unknowns=unknowns,
            conflicts=conflicts,
            confidence=confidence,
            business_implication=implications,
            recommended_next_actions=next_actions,
        )

    @staticmethod
    def _select_builder(routed_question):
        """Question-specific overrides where generic question_type logic is too broad."""
        qid = routed_question.question_id
        if qid == "q_ongoing_trials":
            return _build_ongoing_trials_claims
        if qid == "q_patent_expiry":
            return _build_patent_expiry_claims
        if qid == "q_synthesis_overview":
            return _build_synthesis_overview_claims
        return _CLAIM_BUILDERS.get(routed_question.question_type, _build_generic_claims)

    def _collect_relevant_unknowns(
        self, dossier: Dict[str, Any], routed_question
    ) -> List[Dict[str, Any]]:
        """Filter dossier unknowns relevant to the question's must_have_fields."""
        all_unknowns = dossier.get("unknowns", [])
        if routed_question.question_type == "data_quality":
            return list(all_unknowns)
        relevant = []
        must_sections = set()
        for f in routed_question.must_have_fields:
            must_sections.add(f.split(".")[0].replace("[*]", ""))

        for u in all_unknowns:
            fp = u.get("field_path", "")
            section = fp.split(".")[0]
            if section in must_sections:
                relevant.append(u)

        return relevant

    def _derive_question_unknowns(
        self,
        routed_question,
        dossier: Dict[str, Any],
        claims: List[Claim],
        resolved_scope,
    ) -> List[Dict[str, Any]]:
        """Add question-scoped unknowns that are not explicit dossier fields."""
        qid = routed_question.question_id
        derived: List[Dict[str, Any]] = []

        if qid == "q_patent_expiry":
            covered_regions = set()
            for fam in dossier.get("patent_families", []):
                for hit in _extract_patent_expiry_hits(fam.get("expiry_by_country", []), {"US", "EU"}):
                    covered_regions.add(hit["region"])
            for region in sorted({"US", "EU"} - covered_regions):
                derived.append({
                    "field_path": f"patent_families.expiry_by_country.{region}",
                    "reason_code": "NO_EVIDENCE_IN_CORPUS",
                    "message": f"No explicit {region} patent expiry evidence was identified in the current dossier.",
                    "suggested_next_action": f"Attach authoritative {region} patent expiry evidence before treating {region} expiry as verified.",
                })

        if qid == "q_synthesis_overview":
            api_steps = [
                step for step in (dossier.get("synthesis_steps") or [])
                if _normalize_step_kind(step) == "api_synthesis"
            ]
            if not api_steps:
                derived.append({
                    "field_path": "synthesis_steps.api_route",
                    "reason_code": "NO_EVIDENCE_IN_CORPUS",
                    "message": (
                        "Current process evidence is formulation/manufacturing-oriented; "
                        "a verified API synthesis route is not directly evidenced in the corpus."
                    ),
                    "suggested_next_action": "Attach process-chemistry/API synthesis patents or CMC sources before treating synthesis as closed.",
                })

        if routed_question.question_type == "data_quality":
            for idx, warning in enumerate(resolved_scope.scope_warnings or [], 1):
                derived.append({
                    "field_path": f"scope.warning_{idx}",
                    "reason_code": "SCOPE_WARNING",
                    "message": warning,
                    "suggested_next_action": "Tighten product-context gating before using this answer for external executive delivery.",
                })

        return derived

    @staticmethod
    def _dedupe_unknowns(unknowns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in unknowns:
            key = (
                item.get("field_path"),
                item.get("reason_code"),
                item.get("message"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _detect_conflicts(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Detect conflicting claims (e.g., different status for same jurisdiction)."""
        conflicts = []
        by_jurisdiction = {}
        for c in claims:
            if c.jurisdiction:
                if c.jurisdiction not in by_jurisdiction:
                    by_jurisdiction[c.jurisdiction] = []
                by_jurisdiction[c.jurisdiction].append(c)

        for jur, jur_claims in by_jurisdiction.items():
            if len(jur_claims) > 1:
                # Check for actual conflicts (different values for same field)
                texts = set(c.text for c in jur_claims)
                if len(texts) > 1:
                    conflicts.append({
                        "jurisdiction": jur,
                        "conflicting_claims": [c.claim_id for c in jur_claims],
                        "description": f"Multiple claims for {jur} with different values",
                    })

        return conflicts

    def _compute_confidence(
        self, claims: List[Claim], unknowns: List[Dict],
        coverage_decision, routed_question,
    ) -> Dict[str, Any]:
        """Compute overall and per-dimension confidence."""
        total = len(claims)
        strong = sum(1 for c in claims if c.support_level == "strong")
        moderate = sum(1 for c in claims if c.support_level == "moderate")
        weak = sum(1 for c in claims if c.support_level == "weak")
        unsupported = sum(1 for c in claims if c.support_level == "unsupported")

        # Overall confidence
        if total == 0:
            overall = "none"
        elif unsupported / max(total, 1) > 0.5:
            overall = "low"
        elif strong / max(total, 1) >= 0.5 and not unknowns:
            overall = "high"
        elif (strong + moderate) / max(total, 1) >= 0.5:
            overall = "medium_high" if not unknowns else "medium"
        else:
            overall = "low"

        # Downgrade if coverage says WS1 needed
        if coverage_decision.should_trigger_ws1:
            if overall in ("high", "medium_high"):
                overall = "medium"

        return {
            "overall": overall,
            "strong_claims": strong,
            "moderate_claims": moderate,
            "weak_claims": weak,
            "unsupported_claims": unsupported,
            "total_claims": total,
            "relevant_unknowns": len(unknowns),
            "answer_mode": coverage_decision.answer_mode,
        }

    def _derive_implications(
        self, claims: List[Claim], unknowns: List[Dict], routed_question,
    ) -> List[str]:
        """Derive business implications from claims."""
        implications = []
        q_type = routed_question.question_type

        if q_type == "registration_status":
            positive = [c for c in claims if c.semantic_role == "registration_positive"]
            partial = [c for c in claims if c.semantic_role == "registration_partial"]
            negative = [c for c in claims if c.semantic_role == "registration_negative"]
            missing = [c for c in claims if c.semantic_role == "registration_unknown" or c.support_level == "unsupported"]

            if positive:
                regions = sorted({c.jurisdiction for c in positive if c.jurisdiction})
                implications.append(
                    f"Product is registered in {len(regions)} jurisdiction(s): {', '.join(regions)}"
                )

            if partial:
                regions = sorted({c.jurisdiction for c in partial if c.jurisdiction})
                if regions:
                    implications.append(
                        f"Partial registration grounding remains for: {', '.join(regions)}"
                    )

            if negative:
                regions = sorted({c.jurisdiction for c in negative if c.jurisdiction})
                if regions:
                    implications.append(
                        f"No public registration record verified for: {', '.join(regions)}"
                    )

            if missing:
                regions = sorted({c.jurisdiction for c in missing if c.jurisdiction})
                if regions:
                    implications.append(f"Registration status unknown for: {', '.join(regions)}")

        elif q_type == "ip_fto_risk":
            blocking = [c for c in claims if "blocks" in c.text.lower()]
            if blocking:
                implications.append(f"{len(blocking)} patent families with blocking coverage identified")
            if unknowns:
                implications.append("Patent landscape incomplete — FTO assessment may need additional data")

        elif q_type == "clinical_evidence":
            ongoing = [c for c in claims if c.semantic_role == "ongoing_trial"]
            if ongoing:
                implications.append(f"{len(ongoing)} ongoing clinical trial(s) remain active in the current corpus.")
            elif routed_question.question_id == "q_ongoing_trials":
                implications.append("No ongoing clinical trials were identified in the current corpus snapshot.")

        elif q_type == "synthesis_manufacturing":
            if any(c.semantic_role == "non_api_process_only" for c in claims):
                implications.append("Current process evidence reflects formulation/manufacturing steps more than verified API route chemistry.")

        elif q_type == "data_quality":
            readiness = next((c for c in claims if c.semantic_role == "quality_readiness"), None)
            coverage = next((c for c in claims if c.semantic_role == "quality_coverage"), None)
            critical = next((c for c in claims if c.semantic_role == "quality_critical_unknowns"), None)

            if readiness:
                implications.append(readiness.text)
                if "registrations=YELLOW" in readiness.text or "registrations=RED" in readiness.text:
                    implications.append("Registration layer still limits decision readiness.")
            if coverage:
                implications.append(coverage.text)
            if critical:
                implications.append("Critical unknowns remain and should be resolved before treating the dossier as decision-ready.")
            elif unknowns:
                implications.append(
                    f"The dossier is answerable but still carries {len(unknowns)} unresolved unknown(s)."
                )

        return implications

    def _derive_next_actions(
        self, coverage_decision, unknowns: List[Dict], routed_question,
    ) -> List[str]:
        """Derive recommended next actions."""
        actions = []

        if coverage_decision.should_trigger_ws1:
            actions.append("Run WS1 gap resolver to fill missing source coverage")

        if coverage_decision.missing_fields:
            actions.append(
                f"Investigate {len(coverage_decision.missing_fields)} missing fields: "
                f"{', '.join(coverage_decision.missing_fields[:3])}"
            )

        for u in unknowns[:3]:
            action = u.get("suggested_next_action")
            if action:
                actions.append(action)

        return actions
