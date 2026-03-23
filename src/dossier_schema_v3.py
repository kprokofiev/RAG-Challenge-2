"""
Dossier Report Schema v3.0 (Sprint 7.5 additions)
===================================================
Sprint 4 — structured dossier: passport + registrations + clinical_studies +
patent_families + synthesis_steps + unknowns + evidence_registry.

Sprint 7.5 additions:
  - product_contexts[] — multi-product context separation
  - primary_docs in registrations (enriched PrimaryDoc model)
  - synthesis_steps[].kind typification
  - dossier_quality_v2 (coverage + decision_readiness)
  - run_manifest for reproducibility
  - passport_scope / passport_notice for multi-context

Rule: every field has evidence (doc_id + page + snippet) OR goes to unknowns
with a typed reason_code.  No hallucination / "likely" values without proof.
"""

from __future__ import annotations

import hashlib
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
    doc_kind: Optional[str] = Field(None, description="Source document type (ctgov, smpc, label, pubchem, ...)")
    # JSON evidence locator fields (for JSON-as-document sources like openFDA, PubChem)
    mime_type: Optional[str] = Field(None, description="MIME type of original artifact, e.g. application/json")
    content_hash: Optional[str] = Field(None, description="SHA-256 hash of original artifact bytes")
    locator: Optional[str] = Field(None, description="JSON Pointer (RFC 6901) or JSONPath to the exact value, e.g. /results/0/indications_and_usage/0")


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


# ── Sprint 7.5: Primary doc reference ────────────────────────────────────────

class PrimaryDoc(BaseModel):
    """Reference to a primary Tier-1 regulatory document."""
    doc_id: str = Field(description="Document UUID from corpus")
    doc_kind: str = Field(description="Document type (smpc, label, grls_card, epar, ...)")
    source_url: Optional[str] = Field(None, description="Original URL")
    mime_type: Optional[str] = Field(None, description="MIME type (application/pdf, text/html, ...)")
    title: Optional[str] = Field(None, description="Document title")
    page_hint: Optional[str] = Field(None, description="Page/locator hint if known")


# ── Sprint 7.5: Product context ─────────────────────────────────────────────

class ProductContext(BaseModel):
    """
    One product context — a unique combination of region/route/form/strength/MAH.
    Prevents semantic mixing of e.g. US OTC ibuprofen 200mg tablet vs EU IV 400mg/4ml.
    context_id is deterministic (hash of normalized fields).

    Sprint 13 WS2: Added context_strength and context_origin for trust classification.
    """
    context_id: str = Field(description="Deterministic ID: hash(region+route+form+strength+mah)")
    label: str = Field(description="Human-readable label, e.g. 'EU - IV - 400mg/4ml - Kabi'")
    region: Optional[str] = Field(None, description="Region code (US/EU/RU/EAEU)")
    route: Optional[str] = Field(None, description="Route of administration (oral/IV/topical/...)")
    dosage_forms: List[str] = Field(default_factory=list, description="Dosage forms")
    strengths: List[str] = Field(default_factory=list, description="Strengths")
    mah: Optional[str] = Field(None, description="MAH or identifier")
    identifiers: List[str] = Field(default_factory=list, description="Registration numbers")
    primary_docs: List[PrimaryDoc] = Field(default_factory=list, description="Tier-1 docs for this context")
    evidence_refs: List[str] = Field(default_factory=list, description="Evidence IDs")
    # Sprint 13 WS2: Trust classification
    context_strength: Optional[str] = Field(
        None,
        description=(
            "Sprint 13: Trust level classification. One of: "
            "registration_confirmed | evidence_supported | weak_signal"
        )
    )
    context_origin: Optional[str] = Field(
        None,
        description="Sprint 13: How this context was derived (e.g., 'registration: US ANDA', 'evidence: label snippet')"
    )


# ── Sprint 7.5: Run manifest ────────────────────────────────────────────────

class RunManifest(BaseModel):
    """Minimal manifest for run reproducibility and QA audit trail."""
    run_id: str = Field(description="Pipeline run UUID")
    report_id: str = Field(description="Report ID")
    case_id: Optional[str] = Field(None)
    pipeline_version: Optional[str] = Field(None, description="Git SHA or tag")
    config_digest: Optional[str] = Field(None, description="Hash of env/config")
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Stage timings")
    docs_attached: int = Field(0)
    docs_indexed: int = Field(0)
    docs_failed: int = Field(0)
    critical_failures: List[str] = Field(default_factory=list)
    source_verdicts: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-source verdict codes from gateway preflight (e.g. grls=INFRA_UNAVAILABLE)",
    )
    operator_actions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Actionable steps for the operator when issues are detected",
    )


# ── Sprint 7.5: Quality v2 ──────────────────────────────────────────────────

class DossierQualityV2(BaseModel):
    """
    Separated metrics: coverage (data completeness) + decision_readiness (actionability).
    Prevents "green 100%" when critical unknowns exist.
    """
    coverage: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-block coverage: passport, registrations, clinical, patents, synthesis (0.0-1.0)"
    )
    decision_readiness: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-gate readiness: GREEN/YELLOW/RED for registrations, patents_legal, context_integrity"
    )
    critical_unknowns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of critical unknowns with reason_code, count, impact"
    )
    notes: List[str] = Field(default_factory=list, description="Human-readable notes")


# ── Passport ─────────────────────────────────────────────────────────────────

class DossierPassport(BaseModel):
    """
    Core drug identity & regulatory snapshot.
    All fields are EvidencedValue — none are free-text without citation.
    Missing data → DossierUnknown list at report level, NOT null here.

    S6 additions: smiles, inchi_key, molecular_weight (PubChem chemistry block).
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
    chemical_formula: Optional[EvidencedValue] = Field(None, description="Molecular formula (e.g. C10H14N2)")
    smiles: Optional[EvidencedValue] = Field(
        None,
        description="Canonical SMILES string (PubChem source). S6: PubChem chemistry block."
    )
    inchi_key: Optional[EvidencedValue] = Field(
        None,
        description="InChIKey identifier (PubChem source). S6: PubChem chemistry block."
    )
    molecular_weight: Optional[EvidencedValue] = Field(
        None,
        description="Molecular weight in g/mol (PubChem source). S6: PubChem chemistry block."
    )
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
    # Sprint 7.5: multi-context awareness
    passport_scope: Optional[str] = Field(
        None,
        description="'single_context' or 'multi_context_ambiguous' — set when >1 product context detected"
    )
    passport_notice: Optional[str] = Field(
        None,
        description="Notice when passport contains product-specific fields from mixed contexts"
    )


# ── Registration ─────────────────────────────────────────────────────────────

class DossierRegistration(BaseModel):
    """One marketing authorization record (one country/region)."""
    region: str = Field(description="Region/country code, e.g. 'RU', 'EU', 'US', 'EAEU'")
    context_id: Optional[str] = Field(
        None, description="Sprint 7.5: link to ProductContext.context_id"
    )
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
    primary_docs: List[PrimaryDoc] = Field(
        default_factory=list,
        description="Sprint 7.5: enriched Tier-1 doc references for this registration"
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
    # Sprint 17 WS6: Richer technical focus classification
    technical_focus: Optional[EvidencedValue] = Field(
        None,
        description=(
            "Sprint 17 WS6: detailed technical focus of this patent family. One of: "
            "composition | formulation | process_manufacturing | method_of_use | "
            "combination | salt_polymorph | dosage_form_delivery | intermediate_synthesis | other"
        )
    )
    # Sprint 17 WS7: Process/synthesis relevance classification
    process_relevance: Optional[EvidencedValue] = Field(
        None,
        description=(
            "Sprint 17 WS7: process/synthesis relevance level. One of: "
            "none | weak | moderate | strong. "
            "Based on presence of synthesis examples, manufacturing steps, intermediates in patent text."
        )
    )
    # Sprint 17 WS8: Legal status snapshot at family level
    legal_status_snapshot: Optional[EvidencedValue] = Field(
        None,
        description=(
            "Sprint 17 WS8: legal status of this patent family. One of: "
            "granted | pending | expired | revoked | lapsed | unknown. "
            "Derived from EPO OPS legal status or patent text."
        )
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
    Sprint 7.5: added `kind` field for api_synthesis vs formulation_process classification.
    """
    step_number: int = Field(description="Step index (1-based)")
    kind: str = Field(
        "unknown",
        description=(
            "Sprint 7.5: step type classification. One of: "
            "api_synthesis | formulation_process | manufacturing_process | unknown"
        )
    )
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

    # Sprint 7.5: product contexts for multi-product INN disambiguation
    product_contexts: List[ProductContext] = Field(
        default_factory=list,
        description="Sprint 7.5: product contexts derived from registrations (region+route+form+strength+mah)"
    )

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
            "Legacy quality: passport%, registrations%, clinical%, patents%, synthesis%. "
            "Evidence completeness: fraction of fields with ≥1 evidence_ref."
        )
    )
    # Sprint 7.5: quality v2 (coverage + decision readiness)
    dossier_quality_v2: Optional[DossierQualityV2] = Field(
        None,
        description="Sprint 7.5: separated coverage vs decision_readiness metrics"
    )
    # Sprint 9: coverage ledger (source-universe-aware coverage)
    coverage_ledger: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Sprint 9: source-universe-aware coverage ledger. "
            "Tracks declared/reachable/attached/indexed/extracted/evidenced per source and section. "
            "Distinct from quality_v2 (artifact quality) — this measures source universe completeness."
        )
    )
    # Sprint 7.5: run reproducibility manifest
    run_manifest: Optional[RunManifest] = Field(
        None,
        description="Sprint 7.5: audit-grade run metadata for reproducibility"
    )

    class Config:
        # Allows extra fields from legacy reports — forward-compat
        extra = "allow"


# ── Quality scorer ────────────────────────────────────────────────────────────

def compute_dossier_quality(report: DossierReport) -> Dict[str, Any]:
    """
    Compute per-block coverage and overall evidence completeness.

    S6 additions:
      - registrations_coverage: number of regions with non-empty registration (no pct, raw count)
      - evidence_coverage_pct: fraction of filled fields that have ≥1 evidence_ref
      - chemistry_filled: bool — at least one of smiles/inchi_key/chemical_formula filled
      - Passport now includes smiles, inchi_key, molecular_weight in coverage count (S6 fields)

    Returns dict ready to assign to report.dossier_quality.
    """
    def _ev_filled(ev: Optional[EvidencedValue]) -> bool:
        return ev is not None and ev.value is not None and bool(ev.evidence_refs)

    def _ev_has_value(ev: Optional[EvidencedValue]) -> bool:
        """Filled even without evidence refs — used for partial-credit counting."""
        return ev is not None and ev.value is not None

    def _list_filled(lst: list) -> int:
        return sum(1 for x in lst if (_ev_filled(x) if isinstance(x, EvidencedValue) else bool(x)))

    # ── Passport coverage (S7: region-aware denominator) ─────────────────────
    pp = report.passport

    # Determine registered regions from the registrations block
    registered_regions: set = set()
    for reg in report.registrations:
        if reg.status and reg.status.value:
            registered_regions.add((reg.region or "").upper().strip())
    has_us = bool(registered_regions & {"US", "USA", "UNITED STATES"})

    # US-specific fields: excluded from denominator when drug has no US registration
    _US_SPECIFIC_FIELDS = {"fda_approval_date", "fda_indication"}

    all_scalar_entries = [
        ("fda_approval_date", pp.fda_approval_date),
        ("fda_indication", pp.fda_indication),
        ("chemical_formula", pp.chemical_formula),
        ("drug_class", pp.drug_class),
        ("mechanism_of_action", pp.mechanism_of_action),
        ("route_of_administration", pp.route_of_administration),
        ("smiles", pp.smiles),
        ("inchi_key", pp.inchi_key),
        ("molecular_weight", pp.molecular_weight),
    ]

    not_applicable_fields: list = []
    passport_scalar_fields = []
    for field_name, field_obj in all_scalar_entries:
        if field_name in _US_SPECIFIC_FIELDS and not has_us:
            not_applicable_fields.append(f"passport.{field_name}")
            continue
        passport_scalar_fields.append(field_obj)

    passport_list_fields = [
        pp.trade_names, pp.registered_where, pp.mah_holders,
        pp.dosage_forms, pp.key_dosages,
    ]
    pp_filled = sum(1 for f in passport_scalar_fields if _ev_filled(f))
    pp_filled += sum(1 for lst in passport_list_fields if len(lst) > 0)
    pp_total = len(passport_scalar_fields) + len(passport_list_fields)
    passport_pct = round(pp_filled / pp_total * 100, 1) if pp_total else 0.0

    # ── Registrations coverage (S7: per-field) ──────────────────────────────
    # Per-registration completeness: status (required), mah (required),
    # identifiers (required, list).
    # Score = filled_mandatory_with_evidence / 3 per registration, averaged.
    _REG_MANDATORY_COUNT = 3  # status, mah, identifiers

    reg_field_scores: list = []
    for reg in report.registrations:
        filled = 0
        if _ev_filled(reg.status):
            filled += 1
        if _ev_filled(reg.mah):
            filled += 1
        if reg.identifiers and any(_ev_filled(ident) for ident in reg.identifiers):
            filled += 1
        reg_field_scores.append(round(filled / _REG_MANDATORY_COUNT * 100, 1))

    reg_pct = round(sum(reg_field_scores) / len(reg_field_scores), 1) if reg_field_scores else 0.0
    registrations_coverage = len(report.registrations)
    regions_with_data = list({r.region for r in report.registrations})
    has_empty_registrations = any(
        not r.evidence_refs and not r.identifiers
        for r in report.registrations
    )

    # WS5-P0: Clinical coverage — measure per-study field completeness, not just evidence presence.
    # A study card with evidence but null status/countries/conclusion should NOT be 100%.
    # 6 key meta-fields per study: study_id, phase, status, n_enrolled, countries, conclusion
    _CLINICAL_META_FIELDS = 6
    if report.clinical_studies:
        study_scores = []
        for cs in report.clinical_studies:
            filled = 0
            if cs.study_id and cs.study_id.value:
                filled += 1
            if cs.phase and cs.phase.value:
                filled += 1
            if cs.status and cs.status.value:
                filled += 1
            if cs.n_enrolled and cs.n_enrolled.value:
                filled += 1
            if cs.countries:
                filled += 1
            if cs.conclusion and cs.conclusion.value:
                filled += 1
            study_scores.append(filled / _CLINICAL_META_FIELDS)
        clinical_pct = round(sum(study_scores) / len(study_scores) * 100, 1)
    else:
        clinical_pct = 0.0

    # Sprint 17 WS8: Patents coverage — measure per-family field completeness including new WS6/WS7 fields.
    # Key fields: representative_pub, priority_date, assignees, what_blocks,
    #             technical_focus (WS6), process_relevance (WS7), legal_status_snapshot (WS8), expiry_by_country
    _PATENT_KEY_FIELDS = 8
    if report.patent_families:
        fam_scores = []
        for pf in report.patent_families:
            filled = 0
            if pf.representative_pub and pf.representative_pub.value:
                filled += 1
            if pf.priority_date and pf.priority_date.value:
                filled += 1
            if pf.assignees:
                filled += 1
            if pf.what_blocks and pf.what_blocks.value:
                filled += 1
            if pf.technical_focus and pf.technical_focus.value:
                filled += 1
            if pf.process_relevance and pf.process_relevance.value:
                filled += 1
            if pf.legal_status_snapshot and pf.legal_status_snapshot.value:
                filled += 1
            if pf.expiry_by_country:
                filled += 1
            fam_scores.append(filled / _PATENT_KEY_FIELDS)
        patent_pct = round(sum(fam_scores) / len(fam_scores) * 100, 1)
    else:
        patent_pct = 0.0

    # Synthesis coverage
    synth_pct = round(
        sum(1 for ss in report.synthesis_steps if ss.evidence_refs) / max(len(report.synthesis_steps), 1) * 100,
        1
    ) if report.synthesis_steps else 0.0

    # ── Chemistry block (S6-T4) ───────────────────────────────────────────────
    chemistry_filled = any([
        _ev_filled(pp.chemical_formula),
        _ev_filled(pp.smiles),
        _ev_filled(pp.inchi_key),
    ])

    # ── Unknown reason distribution ───────────────────────────────────────────
    reason_dist: Dict[str, int] = {}
    for u in report.unknowns:
        reason_dist[u.reason_code] = reason_dist.get(u.reason_code, 0) + 1

    # ── Overall evidence completeness (S6-T5) ─────────────────────────────────
    # evidence_coverage_pct = filled fields WITH evidence_refs / all filled fields
    # (measures: "of what we extracted, how much is properly sourced?")
    filled_with_evidence = 0
    filled_total = 0
    all_scalar_fields = passport_scalar_fields + [
        pp.fda_indication,  # also count fda_indication separately
    ]
    for field in passport_scalar_fields:
        if _ev_has_value(field):
            filled_total += 1
            if _ev_filled(field):
                filled_with_evidence += 1
    for lst in passport_list_fields:
        for ev in lst:
            if _ev_has_value(ev):
                filled_total += 1
                if _ev_filled(ev):
                    filled_with_evidence += 1

    evidence_coverage_pct = round(
        filled_with_evidence / max(filled_total, 1) * 100, 1
    )

    # Legacy metric: evidence_completeness_pct (all EvidencedValue slots vs. filled)
    total_ev_values = 0
    filled_ev_values_legacy = 0
    for field in [pp.fda_approval_date, pp.fda_indication, pp.chemical_formula,
                  pp.drug_class, pp.mechanism_of_action, pp.route_of_administration,
                  pp.smiles, pp.inchi_key, pp.molecular_weight]:
        total_ev_values += 1
        if _ev_filled(field):
            filled_ev_values_legacy += 1
    for lst in [pp.trade_names, pp.registered_where, pp.mah_holders, pp.dosage_forms, pp.key_dosages]:
        for ev in lst:
            total_ev_values += 1
            if _ev_filled(ev):
                filled_ev_values_legacy += 1

    evidence_completeness_pct = round(
        filled_ev_values_legacy / max(total_ev_values, 1) * 100, 1
    )

    # Sprint 12: All clinical studies now have study_id (WS1 gate).
    # Count studies with study_id for backward compat transparency.
    studies_with_id = sum(
        1 for cs in report.clinical_studies
        if cs.study_id and cs.study_id.value
    )

    # Sprint 13 WS3: Count studies with core fields filled (study_id + at least 3 of
    # phase, status, n_enrolled, countries, conclusion) — measures usable entities
    _CORE_THRESHOLD = 3  # need at least 3 of 5 non-id fields
    studies_with_core = 0
    for cs in report.clinical_studies:
        if not (cs.study_id and cs.study_id.value):
            continue
        core_filled = 0
        if cs.phase and cs.phase.value:
            core_filled += 1
        if cs.status and cs.status.value:
            core_filled += 1
        if cs.n_enrolled and cs.n_enrolled.value:
            core_filled += 1
        if cs.countries:
            core_filled += 1
        if cs.conclusion and cs.conclusion.value:
            core_filled += 1
        if core_filled >= _CORE_THRESHOLD:
            studies_with_core += 1

    # Sprint 13 WS3: Context strength breakdown for v1 quality
    ctx_strength_counts = {}
    for c in report.product_contexts:
        strength = getattr(c, "context_strength", None) or "unknown"
        ctx_strength_counts[strength] = ctx_strength_counts.get(strength, 0) + 1

    return {
        "passport_pct": passport_pct,
        "passport_total_fields": pp_total,
        "passport_not_applicable": not_applicable_fields,
        "registrations_pct": reg_pct,
        "registrations_coverage": registrations_coverage,
        "regions_with_data": regions_with_data,
        "has_empty_registrations": has_empty_registrations,
        "clinical_pct": clinical_pct,
        "clinical_studies_total": len(report.clinical_studies),
        "clinical_studies_with_id": studies_with_id,
        "clinical_studies_with_core_fields": studies_with_core,
        "patents_pct": patent_pct,
        "patent_families_total": len(report.patent_families),
        "synthesis_pct": synth_pct,
        "chemistry_filled": chemistry_filled,
        "evidence_coverage_pct": evidence_coverage_pct,
        "evidence_completeness_pct": evidence_completeness_pct,
        "unknowns_count": len(report.unknowns),
        "unknown_reason_distribution": reason_dist,
        "evidence_registry_size": len(report.evidence_registry),
        "context_strength_breakdown": ctx_strength_counts,
    }


# ── Sprint 7.5 + Sprint 12: Product context builder ──────────────────────────

def _context_id(region: str, route: str, form: str, strength: str, mah: str) -> str:
    """Deterministic context_id from normalized fields."""
    key = "|".join(s.strip().lower() for s in [region, route, form, strength, mah])
    return "ctx_" + hashlib.md5(key.encode()).hexdigest()[:12]


# Sprint 12 WS2: Route family normalization for context candidates
_ROUTE_FAMILIES = {
    "oral": {"oral", "po", "per os", "by mouth", "sublingual", "buccal"},
    "injectable": {"iv", "intravenous", "im", "intramuscular", "sc", "subcutaneous",
                   "injection", "infusion", "parenteral"},
    "topical": {"topical", "cutaneous", "dermal", "transdermal", "patch"},
    "inhalation": {"inhalation", "inhaled", "nasal", "intranasal", "pulmonary"},
    "ophthalmic": {"ophthalmic", "eye", "ocular"},
    "rectal": {"rectal", "suppository"},
}

# Sprint 12 WS2: Dosage form family normalization
_FORM_FAMILIES = {
    "tablet": {"tablet", "tab", "film-coated tablet", "enteric-coated tablet",
               "chewable tablet", "dispersible tablet", "effervescent tablet",
               "modified-release tablet", "extended-release tablet"},
    "capsule": {"capsule", "cap", "softgel", "soft capsule", "hard capsule",
                "liquid filled capsule", "modified-release capsule"},
    "solution": {"solution", "oral solution", "syrup", "elixir", "drops",
                 "oral suspension", "suspension"},
    "injection": {"injection", "solution for injection", "powder for injection",
                  "infusion", "concentrate for infusion"},
    "cream_ointment": {"cream", "ointment", "gel", "paste", "emulsion"},
    "suppository": {"suppository", "rectal"},
    "patch": {"patch", "transdermal patch", "transdermal system"},
    "inhaler": {"inhaler", "nebuliser", "nebulizer", "metered dose"},
}


import re as _re_mod

# Short keywords (<=3 chars) need word-boundary matching to avoid false
# positives like "po" inside "compound", "im" inside "important", etc.
_SHORT_KW_THRESHOLD = 3
_SHORT_KW_REGEX_CACHE: dict = {}


def _kw_matches(kw: str, text_lower: str) -> bool:
    """Match keyword in text: word-boundary regex for short keywords, substring for long."""
    if len(kw) <= _SHORT_KW_THRESHOLD:
        pat = _SHORT_KW_REGEX_CACHE.get(kw)
        if pat is None:
            pat = _re_mod.compile(r"\b" + _re_mod.escape(kw) + r"\b")
            _SHORT_KW_REGEX_CACHE[kw] = pat
        return pat.search(text_lower) is not None
    if kw not in text_lower:
        return False
    # Sprint 17: Reject matches inside parenthetical enumerations like
    # "(syringe, patch, etc.)" — these are boilerplate device disclaimers,
    # not actual dosage form / route signals.
    _PAREN_ENUM_RE = _re_mod.compile(r"\([^)]*,\s*[^)]*\)")
    for m in _PAREN_ENUM_RE.finditer(text_lower):
        if m.start() <= text_lower.index(kw) < m.end():
            return False
    return True


def _normalize_route_family(text: str) -> Optional[str]:
    """Sprint 12 WS2: Map a route description to a normalized family."""
    text_lower = text.lower().strip()
    for family, keywords in _ROUTE_FAMILIES.items():
        for kw in keywords:
            if _kw_matches(kw, text_lower):
                return family
    return None


def _normalize_form_family(text: str) -> Optional[str]:
    """Sprint 12 WS2: Map a dosage form description to a normalized family."""
    text_lower = text.lower().strip()
    for family, keywords in _FORM_FAMILIES.items():
        for kw in keywords:
            if _kw_matches(kw, text_lower):
                return family
    return None


def _collect_evidence_context_signals(
    evidence_registry: List[DossierEvidence],
    registrations: List[DossierRegistration],
) -> List[Dict[str, Any]]:
    """Sprint 12 WS2: Collect context candidate signals from evidence beyond registrations.

    Scans evidence snippets for form/route signals that may not be in registrations[].
    Returns list of dicts with: source_type, region, route_family, form_family, evidence_refs.
    This is used to detect additional product contexts (e.g., injectable vs oral) that
    registrations alone may not capture.
    """
    import re as _re

    # Regions already covered by registrations
    reg_regions = {(r.region or "").upper().strip() for r in registrations}

    # Doc kinds that hint at regions
    _REGION_DOC_KINDS = {
        "smpc": "EU", "epar": "EU", "pil": "EU", "assessment_report": "EU",
        "label": "US", "us_fda": "US",
        "grls_card": "RU", "grls": "RU", "ru_instruction": "RU",
    }

    signals: List[Dict[str, Any]] = []
    seen_combos: set = set()

    for ev in evidence_registry:
        snippet_lower = (ev.snippet or "").lower()

        # Detect region from doc_kind
        region = _REGION_DOC_KINDS.get(ev.doc_kind, "")

        # Detect route family from snippet
        route_family = _normalize_route_family(snippet_lower)

        # Detect form family from snippet
        form_family = _normalize_form_family(snippet_lower)

        if not (route_family or form_family):
            continue

        combo = (region, route_family or "", form_family or "")
        if combo in seen_combos:
            continue
        seen_combos.add(combo)

        signals.append({
            "source_type": ev.doc_kind or "unknown",
            "region": region,
            "route_family": route_family,
            "form_family": form_family,
            "evidence_refs": [ev.evidence_id],
        })

    return signals


def build_product_contexts(
    registrations: List[DossierRegistration],
    evidence_registry: List[DossierEvidence],
) -> List[ProductContext]:
    """
    Sprint 7.5 TZ-1 + Sprint 12 WS2 + Sprint 13 WS2:
    Build product_contexts from registrations + evidence signals.

    Phase 1: Registration-driven contexts → context_strength=registration_confirmed
    Phase 2: Evidence-based contexts → context_strength=evidence_supported or weak_signal
    Sprint 13: Adds context_strength / context_origin, fixes route/form consistency,
    deduplicates evidence contexts that map to same form/route family.
    """
    _TIER1_DOC_KINDS = {"smpc", "label", "epar", "grls_card", "grls", "ru_instruction",
                        "us_fda", "approval_letter", "pil", "assessment_report"}

    # Sprint 13 WS2: Form families that imply specific routes (for consistency)
    _FORM_TO_ROUTE = {
        "cream_ointment": "topical",
        "patch": "topical",
        "suppository": "rectal",
        "injection": "injectable",
        "inhaler": "inhalation",
    }

    ctx_map: Dict[str, ProductContext] = {}

    # ── Phase 1: Registration-driven contexts ────────────────────────────────
    for reg in registrations:
        region = (reg.region or "").strip().upper()
        forms = []
        for fs in reg.forms_strengths:
            if fs.value:
                val = fs.value if isinstance(fs.value, str) else str(fs.value)
                forms.append(val)
        mah_val = ""
        if reg.mah and reg.mah.value:
            mah_val = str(reg.mah.value)

        form_str = "; ".join(sorted(forms)) if forms else ""
        ctx = _context_id(region, "", form_str, "", mah_val)

        if ctx not in ctx_map:
            label_parts = [region]
            if forms:
                label_parts.append(form_str[:60])
            if mah_val:
                label_parts.append(mah_val[:40])
            label = " - ".join(label_parts)

            primary_docs: List[PrimaryDoc] = []
            seen_doc_ids: set = set()
            for ev_id in reg.evidence_refs:
                for ev in evidence_registry:
                    if ev.evidence_id == ev_id and ev.doc_id not in seen_doc_ids:
                        if ev.doc_kind and ev.doc_kind in _TIER1_DOC_KINDS:
                            primary_docs.append(PrimaryDoc(
                                doc_id=ev.doc_id,
                                doc_kind=ev.doc_kind,
                                source_url=ev.source_url,
                                mime_type=None,
                                title=ev.title,
                            ))
                            seen_doc_ids.add(ev.doc_id)

            # Sprint 13 WS2: Determine registration origin description
            reg_ids = [i.value for i in reg.identifiers if i.value] if reg.identifiers else []
            origin = f"registration: {region}"
            if reg_ids:
                origin += f" {reg_ids[0]}"

            ctx_map[ctx] = ProductContext(
                context_id=ctx,
                label=label,
                region=region,
                route=None,
                dosage_forms=forms,
                strengths=[],
                mah=mah_val or None,
                identifiers=reg_ids,
                primary_docs=primary_docs,
                evidence_refs=list(reg.evidence_refs),
                context_strength="registration_confirmed",
                context_origin=origin,
            )
        else:
            existing = ctx_map[ctx]
            for i in reg.identifiers:
                if i.value and i.value not in existing.identifiers:
                    existing.identifiers.append(i.value)
            for ref in reg.evidence_refs:
                if ref not in existing.evidence_refs:
                    existing.evidence_refs.append(ref)

        reg.context_id = ctx

        if not reg.primary_docs:
            seen: set = set()
            for ev_id in reg.evidence_refs:
                for ev in evidence_registry:
                    if ev.evidence_id == ev_id and ev.doc_id not in seen:
                        if ev.doc_kind and ev.doc_kind in _TIER1_DOC_KINDS:
                            reg.primary_docs.append(PrimaryDoc(
                                doc_id=ev.doc_id,
                                doc_kind=ev.doc_kind,
                                source_url=ev.source_url,
                                title=ev.title,
                            ))
                            seen.add(ev.doc_id)

    # ── Phase 2: Evidence-based context enrichment ───────────────────────────
    evidence_signals = _collect_evidence_context_signals(evidence_registry, registrations)

    existing_form_families: set = set()
    existing_route_families: set = set()
    for ctx_obj in ctx_map.values():
        for form in ctx_obj.dosage_forms:
            fam = _normalize_form_family(form)
            if fam:
                existing_form_families.add(fam)
        if ctx_obj.route:
            rfam = _normalize_route_family(ctx_obj.route)
            if rfam:
                existing_route_families.add(rfam)

    # Sprint 14 P0.4: Pre-aggregate evidence signals by route/form family
    # for corroboration gating. A weak_signal context requires >=2 independent
    # evidence items (by doc_id) mentioning the same route/form to be created.
    # Count from raw evidence_registry (not deduped signals) to capture true
    # corroboration across multiple documents.
    _WEAK_SIGNAL_DOC_KINDS = {"pubchem", "drug_monograph", "publication",
                               "scientific_pmc", "scientific_pdf", "preprint"}
    evidence_corroboration: Dict[str, set] = {}  # key = route_fam|form_fam → set of doc_ids
    for ev in evidence_registry:
        snippet_lower = (ev.snippet or "").lower()
        rf = _normalize_route_family(snippet_lower)
        ff = _normalize_form_family(snippet_lower)
        if rf or ff:
            key = f"{rf or ''}|{ff or ''}"
            if key not in evidence_corroboration:
                evidence_corroboration[key] = set()
            evidence_corroboration[key].add(ev.doc_id or ev.evidence_id)

    suppressed_contexts: List[Dict[str, str]] = []

    for signal in evidence_signals:
        route_fam = signal.get("route_family")
        form_fam = signal.get("form_family")
        region = signal.get("region", "")

        # Sprint 13 WS2: Fix route/form consistency — if form implies a specific
        # route (e.g., cream → topical, suppository → rectal), use that instead
        # of whatever route was detected from the snippet (which could be wrong)
        if form_fam and form_fam in _FORM_TO_ROUTE:
            implied_route = _FORM_TO_ROUTE[form_fam]
            if route_fam and route_fam != implied_route:
                route_fam = implied_route  # form is more reliable than snippet route
            elif not route_fam:
                route_fam = implied_route

        is_new_route = route_fam and route_fam not in existing_route_families
        is_new_form = form_fam and form_fam not in existing_form_families

        if not (is_new_route or is_new_form):
            continue

        form_desc = form_fam or ""
        route_desc = route_fam or ""
        ctx = _context_id(region or "EVIDENCE", route_desc, form_desc, "", "")

        if ctx in ctx_map:
            continue

        # Sprint 13 WS2: Determine strength based on evidence source type
        source_type = signal.get("source_type", "unknown")
        if source_type in _TIER1_DOC_KINDS:
            strength = "evidence_supported"
        else:
            strength = "weak_signal"

        # Sprint 17: Route-contradiction gate — if registrations established
        # specific routes (e.g., injectable/subcutaneous) and the evidence
        # signal introduces a contradictory route (e.g., topical from "patch"
        # keyword in boilerplate), downgrade to weak_signal even for Tier-1.
        if strength == "evidence_supported" and is_new_route and existing_route_families:
            strength = "weak_signal"

        # Sprint 14 P0.4: Corroboration gate for weak_signal contexts.
        # Single PubChem/publication snippets should NOT create standalone
        # product contexts — they can mislead operators into thinking the
        # product has regulatory support for that route/form.
        if strength == "weak_signal":
            corr_key = f"{route_fam or ''}|{form_fam or ''}"
            corr_docs = evidence_corroboration.get(corr_key, set())
            if len(corr_docs) < 2:
                suppressed_contexts.append({
                    "route_family": route_fam or "",
                    "form_family": form_fam or "",
                    "source_type": source_type,
                    "reason": "single_source_weak_signal",
                })
                continue  # Do not create context

        # Sprint 13 WS2: Build informative label with strength indicator
        label_parts = []
        if region:
            label_parts.append(region)
        if form_fam and route_fam:
            label_parts.append(f"{form_fam} ({route_fam})")
        elif form_fam:
            label_parts.append(form_fam)
        elif route_fam:
            label_parts.append(f"({route_fam})")
        label_parts.append(f"[{strength}]")
        label = " - ".join(label_parts)

        origin = f"evidence: {source_type} snippet"

        ctx_map[ctx] = ProductContext(
            context_id=ctx,
            label=label,
            region=region or None,
            route=route_desc or None,
            dosage_forms=[form_fam] if form_fam else [],
            strengths=[],
            mah=None,
            identifiers=[],
            primary_docs=[],
            evidence_refs=signal.get("evidence_refs", []),
            context_strength=strength,
            context_origin=origin,
        )

        if route_fam:
            existing_route_families.add(route_fam)
        if form_fam:
            existing_form_families.add(form_fam)

    return list(ctx_map.values()), suppressed_contexts


# ── Sprint 7.5: Synthesis kind classifier ────────────────────────────────────

_FORMULATION_KEYWORDS = {
    "granulation", "granulate", "sieving", "tableting", "tablet press", "coating",
    "film-coated", "cores", "encapsulation", "capsule fill", "blending", "mixing",
    "compression", "drying", "spray-dry", "lyophilization", "packaging",
    "excipient", "binder", "disintegrant", "lubricant", "filler",
}
_MANUFACTURING_KEYWORDS = {
    "manufacturing", "scale-up", "batch", "gmp", "quality control", "in-process",
    "sterilization", "sterile", "aseptic", "clean room", "validation",
}
_API_SYNTHESIS_KEYWORDS = {
    "synthesis", "reaction", "reflux", "distill", "crystalliz", "recrystalliz",
    "precipitat", "filtrat", "chromatograph", "purif", "coupling", "hydrogenat",
    "acylat", "alkylat", "oxidat", "reduct", "saponif", "esterif",
    "salt formation", "free base", "yield", "mol", "mmol", "equiv",
}


def classify_synthesis_kind(description_text: str) -> str:
    """
    Sprint 7.5 TZ-4: Classify a synthesis step description.
    Returns: api_synthesis | formulation_process | manufacturing_process | unknown
    """
    text = (description_text or "").lower()
    api_score = sum(1 for kw in _API_SYNTHESIS_KEYWORDS if kw in text)
    form_score = sum(1 for kw in _FORMULATION_KEYWORDS if kw in text)
    mfg_score = sum(1 for kw in _MANUFACTURING_KEYWORDS if kw in text)

    if api_score >= 2 or (api_score >= 1 and form_score == 0 and mfg_score == 0):
        return "api_synthesis"
    if form_score >= 2 or (form_score >= 1 and api_score == 0):
        return "formulation_process"
    if mfg_score >= 2 or (mfg_score >= 1 and api_score == 0 and form_score == 0):
        return "manufacturing_process"
    if api_score == 1:
        return "api_synthesis"
    if form_score == 1:
        return "formulation_process"
    return "unknown"


# ── Sprint 7.5: Quality v2 builder ──────────────────────────────────────────

def compute_dossier_quality_v2(
    report: DossierReport,
    source_verdicts: Optional[Dict[str, str]] = None,
) -> DossierQualityV2:
    """
    Sprint 7.5 TZ-5: Compute quality_v2 with coverage + decision_readiness.
    source_verdicts: optional dict from gateway dossier.json (e.g. {"grls": "OK", "openfda": "OK"}).
    """
    q = report.dossier_quality or {}

    # Coverage (normalized 0.0 - 1.0)
    coverage = {
        "passport": round(q.get("passport_pct", 0) / 100, 2),
        "registrations": round(q.get("registrations_pct", 0) / 100, 2),
        "clinical": round(q.get("clinical_pct", 0) / 100, 2),
        "patents": round(q.get("patents_pct", 0) / 100, 2),
        "synthesis": round(q.get("synthesis_pct", 0) / 100, 2),
    }

    # Decision readiness gates
    reason_dist = q.get("unknown_reason_distribution", {})
    legal_unknowns = reason_dist.get("LEGAL_STATUS_NOT_AVAILABLE", 0)
    no_doc_unknowns = reason_dist.get("NO_DOCUMENT_IN_CORPUS", 0)

    # Patents legal readiness
    total_families = len(report.patent_families)
    families_with_expiry = sum(1 for f in report.patent_families if f.expiry_by_country)
    patents_legal_pct = (families_with_expiry / total_families) if total_families > 0 else 0.0

    if patents_legal_pct >= 0.7 and legal_unknowns == 0:
        patents_legal = "GREEN"
    elif patents_legal_pct >= 0.3 or legal_unknowns <= 5:
        patents_legal = "YELLOW"
    else:
        patents_legal = "RED"

    # Context integrity (Sprint 13 WS2: use context_strength field)
    ctx_count = len(report.product_contexts)
    # Count by context_strength classification
    reg_confirmed_ctx = sum(
        1 for c in report.product_contexts
        if getattr(c, "context_strength", None) == "registration_confirmed"
    )
    evidence_supported_ctx = sum(
        1 for c in report.product_contexts
        if getattr(c, "context_strength", None) == "evidence_supported"
    )
    weak_signal_ctx = sum(
        1 for c in report.product_contexts
        if getattr(c, "context_strength", None) == "weak_signal"
    )
    evidence_only_ctx = ctx_count - reg_confirmed_ctx
    if ctx_count <= 1:
        context_integrity = "GREEN"
    elif all(r.context_id for r in report.registrations):
        context_integrity = "GREEN"
    elif evidence_only_ctx > 0 and reg_confirmed_ctx <= 1:
        # Has evidence signals for other forms but only 1 registration → still GREEN
        # (evidence-based contexts are informational, not conflicting)
        context_integrity = "GREEN"
    else:
        context_integrity = "YELLOW" if ctx_count <= 3 else "RED"

    # WS2-P0: Registrations readiness — must check status, mah, identifiers,
    # AND primary_docs.  primary_docs alone is NOT sufficient for GREEN.
    # A registration without status/mah is essentially unverified.
    def _reg_field_ok(ev_val) -> bool:
        return ev_val is not None and ev_val.value is not None and bool(ev_val.evidence_refs)

    regs_complete = 0  # has status + mah + identifiers + primary_docs
    regs_partial = 0   # has some but not all
    for r in report.registrations:
        has_status = _reg_field_ok(r.status)
        has_mah = _reg_field_ok(r.mah)
        has_ids = bool(r.identifiers) and any(
            i.value is not None and bool(i.evidence_refs) for i in r.identifiers
        )
        has_docs = bool(r.primary_docs)
        filled = sum([has_status, has_mah, has_ids, has_docs])
        if filled >= 3:  # at least status + mah + one of (ids, docs)
            regs_complete += 1
        elif filled >= 1:
            regs_partial += 1

    total_regs = len(report.registrations)
    if total_regs == 0:
        registrations_gate = "YELLOW"  # no registrations at all
    elif regs_complete >= total_regs:
        registrations_gate = "GREEN"
    elif regs_complete > 0 or regs_partial > 0:
        registrations_gate = "YELLOW"
    else:
        registrations_gate = "RED"

    # Sprint 17: Region-aware registrations gate.
    # If RU regulatory sources were expected (GRLS was reachable or RU evidence exists)
    # but no RU registration is present, downgrade from GREEN.
    actual_regions = {(r.region or "").upper().strip() for r in report.registrations}
    _sv = source_verdicts or {}
    grls_verdict = _sv.get("grls", "")
    # RU was expected if: (a) GRLS was reachable (OK or SOURCE_EMPTY), or
    # (b) grls_card/ru_instruction evidence already exists in corpus
    _ru_evidence_kinds = {"grls_card", "grls", "ru_instruction"}
    has_ru_evidence = any(
        ev.doc_kind and ev.doc_kind.lower() in _ru_evidence_kinds
        for ev in report.evidence_registry
    )
    ru_expected = grls_verdict in ("OK",) or has_ru_evidence
    missing_expected_regions: List[str] = []
    if ru_expected and "RU" not in actual_regions:
        missing_expected_regions.append("RU")
    if missing_expected_regions and registrations_gate == "GREEN":
        registrations_gate = "YELLOW"

    # Sprint 13 WS3: Clinical readiness — semantic-aware gate
    # Uses both field coverage AND core-field completeness
    clinical_cov = coverage.get("clinical", 0)
    n_studies = len(report.clinical_studies)

    # Sprint 13 WS3: Count studies with core fields (study_id + 3+ of phase/status/n_enrolled/countries/conclusion)
    _CORE_THRESHOLD_V2 = 3
    studies_with_core = 0
    for cs in report.clinical_studies:
        if not (cs.study_id and cs.study_id.value):
            continue
        core_filled = 0
        if cs.phase and cs.phase.value:
            core_filled += 1
        if cs.status and cs.status.value:
            core_filled += 1
        if cs.n_enrolled and cs.n_enrolled.value:
            core_filled += 1
        if cs.countries:
            core_filled += 1
        if cs.conclusion and cs.conclusion.value:
            core_filled += 1
        if core_filled >= _CORE_THRESHOLD_V2:
            studies_with_core += 1

    if n_studies == 0:
        clinical_gate = "RED"
    elif clinical_cov >= 0.6:
        clinical_gate = "GREEN"
    elif clinical_cov >= 0.3:
        clinical_gate = "YELLOW"
    else:
        clinical_gate = "RED"

    # WS5-P0: Patents discovery vs legal — separate gates
    patents_cov = coverage.get("patents", 0)
    if total_families == 0:
        patents_discovery_gate = "RED"
    elif patents_cov >= 0.6:
        patents_discovery_gate = "GREEN"
    else:
        patents_discovery_gate = "YELLOW"

    decision_readiness = {
        "registrations": registrations_gate,
        "clinical": clinical_gate,
        "patents_discovery": patents_discovery_gate,
        "patents_legal": patents_legal,
        "context_integrity": context_integrity,
    }

    # Critical unknowns
    critical_unknowns: List[Dict[str, Any]] = []
    if legal_unknowns > 0:
        critical_unknowns.append({
            "reason_code": "LEGAL_STATUS_NOT_AVAILABLE",
            "count": legal_unknowns,
            "impact": f"patents_legal={patents_legal}",
        })
    if no_doc_unknowns > 0:
        critical_unknowns.append({
            "reason_code": "NO_DOCUMENT_IN_CORPUS",
            "count": no_doc_unknowns,
            "impact": "registrations may be incomplete",
        })
    # WS5-P0: Flag clinical meta-field gaps as critical unknown
    no_ev_unknowns = reason_dist.get("NO_EVIDENCE_IN_CORPUS", 0)
    if n_studies > 0 and clinical_cov < 0.5:
        critical_unknowns.append({
            "reason_code": "CLINICAL_META_FIELDS_INCOMPLETE",
            "count": n_studies,
            "impact": f"clinical={clinical_gate}, clinical_coverage={round(clinical_cov*100,1)}%",
        })
    # Sprint 17: Infra-level critical unknowns from source_verdicts
    if grls_verdict in ("INFRA_UNAVAILABLE", "NOT_CONFIGURED", "SOURCE_TIMEOUT"):
        critical_unknowns.append({
            "reason_code": grls_verdict,
            "count": 1,
            "impact": "RU regulatory data absent — not a data gap, infrastructure/tunnel failure",
        })
    if missing_expected_regions:
        critical_unknowns.append({
            "reason_code": "MISSING_EXPECTED_REGION",
            "count": len(missing_expected_regions),
            "impact": f"Expected regions {missing_expected_regions} have no registration data",
        })

    notes: List[str] = []
    if ctx_count > 1:
        # Sprint 13 WS2: context strength breakdown
        strength_parts = []
        if reg_confirmed_ctx:
            strength_parts.append(f"{reg_confirmed_ctx} registration_confirmed")
        if evidence_supported_ctx:
            strength_parts.append(f"{evidence_supported_ctx} evidence_supported")
        if weak_signal_ctx:
            strength_parts.append(f"{weak_signal_ctx} weak_signal")
        if strength_parts:
            notes.append(f"Product contexts: {', '.join(strength_parts)}")
        else:
            notes.append(f"Multiple product contexts detected: {ctx_count}")
    if patents_legal_pct < 1.0 and total_families > 0:
        notes.append(f"patents_legal_pct={round(patents_legal_pct*100,1)}% ({families_with_expiry}/{total_families} families with expiry)")
    if total_families == 0:
        # Sprint 12 WS3: Explicit note when patent_families is empty
        notes.append(
            "patent_families=[] — no minimally valid patent families in corpus. "
            "For off-patent drugs this is expected; for on-patent drugs check EPO OPS/patent sources."
        )
    if n_studies > 0 and clinical_cov < 0.5:
        notes.append(f"clinical_field_completeness={round(clinical_cov*100,1)}% — many meta-fields (status/countries/conclusion) are empty")
    # Sprint 13 WS3: Clinical integrity note — shows usable entities vs total
    if n_studies > 0:
        notes.append(
            f"clinical_studies: {n_studies} total, all have study_id (per-NCT assembly), "
            f"{studies_with_core}/{n_studies} with core fields (phase+status+enrollment+countries+conclusion >= 3/5)"
        )
    # Sprint 17: Region-aware notes
    if missing_expected_regions:
        notes.append(
            f"registrations: regions_with_data={sorted(actual_regions)}, "
            f"expected_but_missing={missing_expected_regions}"
        )
    if grls_verdict and grls_verdict not in ("OK", "SOURCE_EMPTY", ""):
        notes.append(
            f"RU regulatory path unavailable: grls_verdict={grls_verdict}. "
            "Check: (1) VPS SSH tunnel alive? (2) local grls-service container stopped? (3) port 9095 forwarded?"
        )

    return DossierQualityV2(
        coverage=coverage,
        decision_readiness=decision_readiness,
        critical_unknowns=critical_unknowns,
        notes=notes,
    )


# ── JSON Schema export ────────────────────────────────────────────────────────

def get_json_schema() -> Dict[str, Any]:
    """Return the JSON Schema for DossierReport v3.0 (for validation tooling)."""
    return DossierReport.model_json_schema()
