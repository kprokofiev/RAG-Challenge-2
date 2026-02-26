"""
Dossier Report Schema v3.0
==========================
Sprint 4 — structured dosier: passport + registrations + clinical_studies +
patent_families + synthesis_steps + unknowns + evidence_registry.

Rule: every field has evidence (doc_id + page + snippet) OR goes to unknowns
with a typed reason_code.  No hallucination / "likely" values without proof.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, validator

# ── Reason codes for unknowns ────────────────────────────────────────────────
# Used in DossierUnknown.reason_code.  Keep list exhaustive & typed so
# downstream consumers can build UI / filters on these codes.

DossierReasonCode = Literal[
    # Corpus gaps
    "NO_DOCUMENT_IN_CORPUS",       # no doc of required kind was indexed
    "PATENT_DISCOVERY_GAP",        # research phase did not discover patent URLs
    "EPAR_SMPC_DISCOVERY_GAP",     # EMA EPAR/SmPC not found by research
    "RU_INSTRUCTION_NOT_AVAILABLE",# GRLS instruction_url is nil / not fetched
    "EAEU_NOT_IMPLEMENTED",        # EAEU registry client is a stub
    # Legal / data quality
    "LEGAL_STATUS_NOT_AVAILABLE",  # expiry/SPC data not in corpus
    "NO_EVIDENCE_IN_CORPUS",       # docs present but no relevant passage found
    "EXTRACTION_FAILED",           # LLM could not extract structured value
    "AMBIGUOUS_VALUE",             # multiple conflicting values, no clear authority
    # Out-of-scope / not applicable
    "NOT_APPLICABLE",              # field not relevant for this drug/jurisdiction
    "EXTERNAL_SERVICE_UNAVAILABLE",# required API/service down
]


# ── Evidence ─────────────────────────────────────────────────────────────────

class DossierEvidence(BaseModel):
    """Single evidence citation — doc_id + page + snippet (mandatory)."""
    evidence_id: str = Field(description="Unique evidence ID, e.g. ev_{doc_id}_{page}_{hash}")
    doc_id: str = Field(description="Document identifier (UUID from documents table)")
    title: Optional[str] = Field(None, description="Document title for display")
    source_url: Optional[str] = Field(None, description="Original source URL")
    page: Optional[int] = Field(None, description="Page number; None for structured-data sources")
    snippet: str = Field(description="Verbatim text snippet (200-400 chars) from the source")


# ── Unknown (typed gap) ───────────────────────────────────────────────────────

class DossierUnknown(BaseModel):
    """Typed gap record — replaces every field that cannot be filled with evidence."""
    field_path: str = Field(
        description="Dot-notation path of the missing field, e.g. 'passport.fda_approval_date'"
    )
    reason_code: str = Field(description="One of DossierReasonCode literals")
    message: str = Field(description="Human-readable explanation of why the field is missing")
    suggested_next_action: Optional[str] = Field(
        None,
        description="Concrete action to resolve this gap, e.g. 'Attach EMA EPAR PDF to corpus'"
    )


# ── Evidence-locked value helper ─────────────────────────────────────────────

class EvidencedValue(BaseModel):
    """A single scalar value with mandatory evidence_ids linkage."""
    value: Union[str, int, float, List[str], None] = Field(
        description="The extracted value (str/num/list)"
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="List of evidence_ids from the report evidence_registry"
    )


# ── Passport ─────────────────────────────────────────────────────────────────

class DossierPassport(BaseModel):
    """
    Core drug identity & regulatory snapshot.
    All fields are EvidencedValue — none are free-text without citation.
    Missing data → DossierUnknown list at report level, NOT null here.
    """
    inn: str = Field(description="International Nonproprietary Name (query key)")
    trade_names: List[EvidencedValue] = Field(
        default_factory=list,
        description="Known trade names with evidence"
    )
    fda_approval_date: Optional[EvidencedValue] = Field(
        None, description="US FDA approval date + indication"
    )
    fda_indication: Optional[EvidencedValue] = Field(
        None, description="FDA approved indication (brief)"
    )
    registered_where: List[EvidencedValue] = Field(
        default_factory=list,
        description="Jurisdictions with valid marketing authorization (chips)"
    )
    chemical_formula: Optional[EvidencedValue] = Field(None, description="Molecular formula")
    drug_class: Optional[EvidencedValue] = Field(None, description="Pharmacological class / ATC")
    mechanism_of_action: Optional[EvidencedValue] = Field(None, description="MoA summary")
    mah_holders: List[EvidencedValue] = Field(
        default_factory=list,
        description="Marketing Authorization Holders (all jurisdictions)"
    )
    route_of_administration: Optional[EvidencedValue] = Field(None)
    dosage_forms: List[EvidencedValue] = Field(
        default_factory=list, description="Available pharmaceutical forms"
    )
    key_dosages: List[EvidencedValue] = Field(
        default_factory=list,
        description="Key approved doses / regimens"
    )


# ── Registration ─────────────────────────────────────────────────────────────

class DossierRegistration(BaseModel):
    """One marketing authorization record (one country/region)."""
    region: str = Field(description="Region/country code, e.g. 'RU', 'EU', 'US', 'EAEU'")
    status: Optional[EvidencedValue] = Field(None, description="Registered / cancelled / expired")
    forms_strengths: List[EvidencedValue] = Field(
        default_factory=list,
        description="Registered forms and strengths"
    )
    mah: Optional[EvidencedValue] = Field(None, description="MAH name")
    identifiers: List[EvidencedValue] = Field(
        default_factory=list,
        description="Registration numbers (GRLS reg#, NDA#, EMA/H/C/#, etc.)"
    )
    primary_docs: List[str] = Field(
        default_factory=list,
        description="doc_ids of primary regulatory documents for this registration"
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="evidence_ids backing this registration record"
    )


# ── Clinical Study Card ───────────────────────────────────────────────────────

class DossierClinicalStudy(BaseModel):
    """
    Structured clinical study card per Sprint 4 TZ template.
    Fields are EvidencedValue so every data point traces to a source.
    """
    title: Optional[EvidencedValue] = Field(None, description="Study title")
    study_id: Optional[EvidencedValue] = Field(
        None, description="Registry ID (NCT#, EudraCT#, CTRI#, etc.)"
    )
    phase: Optional[EvidencedValue] = Field(None, description="Phase (I/II/III/IV)")
    study_type: Optional[EvidencedValue] = Field(
        None, description="Study type: randomized / non-randomized / observational"
    )
    n_enrolled: Optional[EvidencedValue] = Field(None, description="Enrollment count")
    countries: List[EvidencedValue] = Field(
        default_factory=list, description="Countries where study was conducted"
    )
    comparator: Optional[EvidencedValue] = Field(
        None, description="Comparator arm (placebo / active / none)"
    )
    regimen_dosing: Optional[EvidencedValue] = Field(None, description="Dosing regimen")
    efficacy_keypoints: List[EvidencedValue] = Field(
        default_factory=list,
        description="Key efficacy findings (primary endpoint result, p-value, CI)"
    )
    conclusion: Optional[EvidencedValue] = Field(None, description="Author/registry conclusion")
    status: Optional[EvidencedValue] = Field(
        None, description="Completed / ongoing / terminated"
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="All evidence_ids for this study card"
    )


# ── Patent Family ─────────────────────────────────────────────────────────────

class DossierPatentFamily(BaseModel):
    """
    One patent family (INPADOC family or synthetic family from Rospatent CSV).
    expiry_by_country is only populated when reliable data exists — otherwise
    the field is absent and a DossierUnknown with reason_code=LEGAL_STATUS_NOT_AVAILABLE
    appears in the report unknowns list.
    """
    family_id: str = Field(description="INPADOC family ID or synthetic hash (e.g. INPADOC_EP1234567)")
    representative_pub: Optional[EvidencedValue] = Field(
        None, description="Representative publication number (EP/WO/US)"
    )
    priority_date: Optional[EvidencedValue] = Field(
        None, description="Earliest priority date (YYYY-MM-DD)"
    )
    assignees: List[EvidencedValue] = Field(
        default_factory=list, description="Patent assignees / applicants"
    )
    what_blocks: Optional[EvidencedValue] = Field(
        None,
        description="Classification of what this patent blocks: compound / formulation / method_of_use / synthesis / other"
    )
    summary: Optional[EvidencedValue] = Field(
        None, description="One-sentence summary from abstract or title"
    )
    country_coverage: List[EvidencedValue] = Field(
        default_factory=list,
        description="Countries where patent is in force (EP/US/CN/JP/RU/etc.)"
    )
    expiry_by_country: List[EvidencedValue] = Field(
        default_factory=list,
        description=(
            "Expiry dates per country — only populated when reliable data present. "
            "Missing → DossierUnknown(reason_code=LEGAL_STATUS_NOT_AVAILABLE)"
        )
    )
    key_docs: List[str] = Field(
        default_factory=list,
        description="doc_ids of patent PDFs or ops records in corpus"
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="evidence_ids backing this family record"
    )


# ── Synthesis Step ────────────────────────────────────────────────────────────

class DossierSynthesisStep(BaseModel):
    """
    One step in a synthesis / manufacturing process extracted from patent text.
    Only populated when patent text is in corpus — otherwise DossierUnknown.
    """
    step_number: int = Field(description="Step index (1-based)")
    description: EvidencedValue = Field(description="Description of the step with evidence")
    reagents: List[EvidencedValue] = Field(
        default_factory=list,
        description="Reagents / starting materials"
    )
    intermediates: List[EvidencedValue] = Field(
        default_factory=list,
        description="Intermediates produced"
    )
    source_patent_refs: List[str] = Field(
        default_factory=list,
        description="Patent doc_ids from which this step was extracted"
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="evidence_ids for this step"
    )


# ── Top-level Dossier Report ──────────────────────────────────────────────────

class DossierReport(BaseModel):
    """
    Dossier Report v3.0 — top-level container.

    Schema discipline:
      - All data fields are EvidencedValue or have explicit evidence_refs.
      - All missing data fields are represented in unknowns[] with typed reason_code.
      - No free-text claims without evidence_id linkage.
    """
    schema_version: str = Field("3.0", description="Dossier schema version")
    report_id: str = Field(description="Unique report identifier")
    case_id: Optional[str] = Field(None, description="Case ID from DDKit pipeline")
    run_id: Optional[str] = Field(None, description="Pipeline run ID")
    generated_at: str = Field(description="ISO-8601 generation timestamp")

    # Core sections
    passport: DossierPassport
    registrations: List[DossierRegistration] = Field(default_factory=list)
    clinical_studies: List[DossierClinicalStudy] = Field(default_factory=list)
    patent_families: List[DossierPatentFamily] = Field(default_factory=list)
    synthesis_steps: List[DossierSynthesisStep] = Field(default_factory=list)

    # Evidence & gaps
    unknowns: List[DossierUnknown] = Field(
        default_factory=list,
        description="All typed gaps — every missing field with reason_code"
    )
    evidence_registry: List[DossierEvidence] = Field(
        default_factory=list,
        description="Global deduplicated evidence index"
    )

    # Pipeline metadata (backward-compat: also include legacy sections for UI)
    sections: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Legacy sections[] from v2.x DD report (if generated in parallel)"
    )
    completeness: Optional[Dict[str, Any]] = Field(
        None,
        description="Completeness block from DDReportGenerator (expected/included/missing/ratio)"
    )

    # Quality metrics
    dossier_quality: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Coverage per block: passport%, registrations%, clinical%, patents%, synthesis%. "
            "Evidence completeness: fraction of fields with ≥1 evidence_ref."
        )
    )

    class Config:
        # Allows extra fields from legacy reports — forward-compat
        extra = "allow"


# ── Quality scorer ────────────────────────────────────────────────────────────

def compute_dossier_quality(report: DossierReport) -> Dict[str, Any]:
    """
    Compute per-block coverage and overall evidence completeness.

    Returns dict ready to assign to report.dossier_quality.
    """
    def _ev_filled(ev: Optional[EvidencedValue]) -> bool:
        return ev is not None and ev.value is not None and bool(ev.evidence_refs)

    def _list_filled(lst: list) -> int:
        return sum(1 for x in lst if (_ev_filled(x) if isinstance(x, EvidencedValue) else bool(x)))

    # Passport coverage
    pp = report.passport
    passport_fields = [
        pp.fda_approval_date, pp.fda_indication, pp.chemical_formula,
        pp.drug_class, pp.mechanism_of_action, pp.route_of_administration,
    ]
    passport_list_fields = [pp.trade_names, pp.registered_where, pp.mah_holders,
                            pp.dosage_forms, pp.key_dosages]
    pp_filled = sum(1 for f in passport_fields if _ev_filled(f))
    pp_filled += sum(1 for lst in passport_list_fields if len(lst) > 0)
    pp_total = len(passport_fields) + len(passport_list_fields)
    passport_pct = round(pp_filled / pp_total * 100, 1) if pp_total else 0.0

    # Registrations coverage
    reg_pct = round(
        sum(1 for r in report.registrations if r.evidence_refs) / max(len(report.registrations), 1) * 100,
        1
    ) if report.registrations else 0.0

    # Clinical coverage
    clinical_pct = round(
        sum(1 for cs in report.clinical_studies if cs.evidence_refs) / max(len(report.clinical_studies), 1) * 100,
        1
    ) if report.clinical_studies else 0.0

    # Patents coverage
    patent_pct = round(
        sum(1 for pf in report.patent_families if pf.evidence_refs) / max(len(report.patent_families), 1) * 100,
        1
    ) if report.patent_families else 0.0

    # Synthesis coverage
    synth_pct = round(
        sum(1 for ss in report.synthesis_steps if ss.evidence_refs) / max(len(report.synthesis_steps), 1) * 100,
        1
    ) if report.synthesis_steps else 0.0

    # Unknown reason distribution
    reason_dist: Dict[str, int] = {}
    for u in report.unknowns:
        reason_dist[u.reason_code] = reason_dist.get(u.reason_code, 0) + 1

    # Overall evidence completeness: fraction of evidence-locked values that have ≥1 ref
    total_ev_values = 0
    filled_ev_values = 0
    for field in [pp.fda_approval_date, pp.fda_indication, pp.chemical_formula,
                  pp.drug_class, pp.mechanism_of_action, pp.route_of_administration]:
        total_ev_values += 1
        if _ev_filled(field):
            filled_ev_values += 1
    for lst in [pp.trade_names, pp.registered_where, pp.mah_holders, pp.dosage_forms, pp.key_dosages]:
        for ev in lst:
            total_ev_values += 1
            if _ev_filled(ev):
                filled_ev_values += 1

    evidence_completeness_pct = round(
        filled_ev_values / max(total_ev_values, 1) * 100, 1
    )

    return {
        "passport_pct": passport_pct,
        "registrations_pct": reg_pct,
        "clinical_pct": clinical_pct,
        "patents_pct": patent_pct,
        "synthesis_pct": synth_pct,
        "evidence_completeness_pct": evidence_completeness_pct,
        "unknowns_count": len(report.unknowns),
        "unknown_reason_distribution": reason_dist,
        "evidence_registry_size": len(report.evidence_registry),
    }


# ── JSON Schema export ────────────────────────────────────────────────────────

def get_json_schema() -> Dict[str, Any]:
    """Return the JSON Schema for DossierReport v3.0 (for validation tooling)."""
    return DossierReport.model_json_schema()
