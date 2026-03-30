"""
Dossier Report Generator v3.0
==============================
Sprint 4 — produces DossierReport v3.0 from the indexed document corpus.

Architecture: evidence-first, 4-stage pipeline per block:
  A. Gather candidates (authority-first, via HybridRetriever)
  B. LLM extraction → strict schema block (EvidencedValue items)
  C. Validate — every field has evidence_ref or → DossierUnknown
  D. Assemble DossierReport

Rule: no field is populated without evidence_refs OR explicit reason_code in unknowns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.api_requests import APIProcessor
from src.dossier_schema_v3 import (
    DossierEvidence,
    DossierUnknown,
    EvidencedValue,
    DossierPassport,
    DossierRegistration,
    DossierClinicalStudy,
    DossierPatentFamily,
    DossierSynthesisStep,
    DossierReport,
    RunManifest,
    compute_dossier_quality,
    build_product_contexts,
    classify_synthesis_kind,
    compute_dossier_quality_v2,
)
from src.evidence_builder import EvidenceCandidatesBuilder
from src.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class RateLimitExhausted(Exception):
    """Raised when OpenAI 429 retries are exhausted at request level.

    Signals that the job should be parked (deferred), NOT restarted from scratch.
    """
    pass


# ── Authority-tiering policy (S6-T2) ───────────────��─────────────────────────
# Maps passport/registration field → allowed Tier-1 doc_kinds.
# Fields marked Tier-2 are populated only from listed doc_kinds with confidence=medium.
# Registration status/numbers are FORBIDDEN from Tier-2 sources.
#
# Tier-1: authoritative regulatory filings (label, EPAR, SmPC, GRLS, DailyMed, Drugs@FDA)
# Tier-2: secondary (moa_overview, drug_monograph, press_release — only for MoA/class)

FIELD_ALLOWED_SOURCES: Dict[str, List[str]] = {
    # Registration facts — strict Tier-1 only
    "registered_where":        ["epar", "smpc", "label", "us_fda", "grls_card", "grls", "eaeu_document", "eaeu_registration"],
    "fda_approval_date":       ["label", "us_fda", "approval_letter"],
    "mah_holders":             ["smpc", "epar", "grls_card", "grls", "ru_instruction", "label", "eaeu_registration"],
    # Identity / forms — Tier-1 preferred, Tier-2 fallback for moa/class
    "trade_names":             ["label", "smpc", "grls_card", "grls", "ru_instruction", "epar"],
    "dosage_forms":            ["label", "smpc", "grls_card", "ru_instruction"],
    "key_dosages":             ["label", "smpc", "ru_instruction"],
    # MoA / class — Tier-2 allowed with medium confidence
    "drug_class":              ["label", "smpc", "drug_monograph", "moa_overview"],
    "mechanism_of_action":     ["label", "smpc", "drug_monograph", "moa_overview"],
    # Chemistry — PubChem only
    "chemical_formula":        ["pubchem", "drug_monograph"],
    "smiles":                  ["pubchem"],
    "inchi_key":               ["pubchem"],
}

# Tier-2 doc_kinds: populates with confidence=medium, forbidden for regulatory identifiers
_TIER2_DOC_KINDS = {"drug_monograph", "moa_overview", "review_article"}

# Fields where Tier-2 is FORBIDDEN (reg statuses, reg numbers, approval dates)
_TIER1_ONLY_FIELDS = {"registered_where", "fda_approval_date", "mah_holders"}


# ── LLM schemas for structured extraction ────────────────────────────────────

class _EvidencedValueLLM(BaseModel):
    value: Optional[str] = Field(None, description="Extracted value as string")
    evidence_id: Optional[str] = Field(
        None,
        description=(
            "EXACTLY ONE evidence alias from the Available Evidence Candidates list "
            "(format: E1, E2, ..., E25). "
            "Set to null ONLY if no candidate contains information about this field."
        )
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description=(
            "REQUIRED when value is non-null: list of evidence aliases from the Available Evidence Candidates list. "
            "At least one alias must be provided (e.g. [\"E1\", \"E4\"]). "
            "Empty list is NOT acceptable when value is set. Use exact alias format: E1, E2, ... not '1' or 'evidence_1'."
        )
    )


class _PassportExtractLLM(BaseModel):
    inn: Optional[str] = None
    trade_names: List[_EvidencedValueLLM] = Field(default_factory=list)
    fda_approval_date: Optional[_EvidencedValueLLM] = None
    fda_indication: Optional[_EvidencedValueLLM] = None
    registered_where: List[_EvidencedValueLLM] = Field(default_factory=list)
    chemical_formula: Optional[_EvidencedValueLLM] = None
    # S6-T4: PubChem chemistry block
    smiles: Optional[_EvidencedValueLLM] = Field(None, description="Canonical SMILES from PubChem")
    inchi_key: Optional[_EvidencedValueLLM] = Field(None, description="InChIKey from PubChem")
    molecular_weight: Optional[_EvidencedValueLLM] = Field(None, description="Molecular weight g/mol from PubChem")
    drug_class: Optional[_EvidencedValueLLM] = None
    mechanism_of_action: Optional[_EvidencedValueLLM] = None
    mah_holders: List[_EvidencedValueLLM] = Field(default_factory=list)
    route_of_administration: Optional[_EvidencedValueLLM] = None
    dosage_forms: List[_EvidencedValueLLM] = Field(default_factory=list)
    key_dosages: List[_EvidencedValueLLM] = Field(default_factory=list)


class _RegistrationLLM(BaseModel):
    region: str = ""
    status: Optional[_EvidencedValueLLM] = None
    forms_strengths: List[_EvidencedValueLLM] = Field(default_factory=list)
    mah: Optional[_EvidencedValueLLM] = None
    identifiers: List[_EvidencedValueLLM] = Field(default_factory=list)


class _RegistrationsExtractLLM(BaseModel):
    registrations: List[_RegistrationLLM] = Field(default_factory=list)


class _ClinicalStudyLLM(BaseModel):
    title: Optional[_EvidencedValueLLM] = None
    study_id: Optional[_EvidencedValueLLM] = None
    phase: Optional[_EvidencedValueLLM] = None
    study_type: Optional[_EvidencedValueLLM] = None
    n_enrolled: Optional[_EvidencedValueLLM] = None
    countries: List[_EvidencedValueLLM] = Field(default_factory=list)
    comparator: Optional[_EvidencedValueLLM] = None
    regimen_dosing: Optional[_EvidencedValueLLM] = None
    efficacy_keypoints: List[_EvidencedValueLLM] = Field(default_factory=list)
    conclusion: Optional[_EvidencedValueLLM] = None
    status: Optional[_EvidencedValueLLM] = None


class _ClinicalStudiesExtractLLM(BaseModel):
    studies: List[_ClinicalStudyLLM] = Field(default_factory=list)


class _PatentFamilyLLM(BaseModel):
    family_id: str = ""
    representative_pub: Optional[_EvidencedValueLLM] = None
    priority_date: Optional[_EvidencedValueLLM] = None
    assignees: List[_EvidencedValueLLM] = Field(default_factory=list)
    what_blocks: Optional[_EvidencedValueLLM] = None
    # Sprint 17 WS6: richer technical focus
    technical_focus: Optional[_EvidencedValueLLM] = None
    # Sprint 17 WS7: process/synthesis relevance
    process_relevance: Optional[_EvidencedValueLLM] = None
    # Sprint 17 WS8: legal status snapshot
    legal_status_snapshot: Optional[_EvidencedValueLLM] = None
    summary: Optional[_EvidencedValueLLM] = None
    country_coverage: List[_EvidencedValueLLM] = Field(default_factory=list)
    expiry_by_country: List[_EvidencedValueLLM] = Field(default_factory=list)


class _PatentFamiliesExtractLLM(BaseModel):
    families: List[_PatentFamilyLLM] = Field(default_factory=list)


class _SynthesisStepLLM(BaseModel):
    step_number: int = 0
    description: Optional[_EvidencedValueLLM] = None
    reagents: List[_EvidencedValueLLM] = Field(default_factory=list)
    intermediates: List[_EvidencedValueLLM] = Field(default_factory=list)


class _SynthesisExtractLLM(BaseModel):
    steps: List[_SynthesisStepLLM] = Field(default_factory=list)


# ── Prompts ───────────────────────────────────────────────────────────────────

_PASSPORT_INSTRUCTION = """
You are a pharmaceutical dossier extraction system.
Extract drug passport fields from the provided context.

INSTRUCTIONS:
1. For EACH field, provide the extracted value AND the evidence alias from the
   "Available Evidence Candidates" list (format: E1, E2, ..., E25).
2. You MAY supply multiple aliases in the "evidence_ids" list field when several candidates
   support the same value (e.g. ["E1", "E4"]).
3. If a field is present in the context, you MUST fill both "value" and at least one of
   "evidence_id" / "evidence_ids". Do NOT leave evidence_id null if you filled value.
4. If a field cannot be found in the context → set value to null (do NOT guess or hallucinate).
5. Tier-1 sources take priority: FDA label (doc_kind=label/us_fda) > EMA EPAR/SmPC > GRLS > other.

FIELD-SPECIFIC EXTRACTION GUIDANCE:
- fda_approval_date: Look for "FDA Approval Date (Drugs@FDA):", "approval date", "first approved",
  "original approval date", or earliest submission with status "AP" in the FDA label document.
  Format as YYYY-MM-DD. This field appears in rendered FDA label documents.
- mechanism_of_action: Look for "mechanism of action", "pharmacodynamics", "MOA",
  "mode of action" in FDA label or SmPC sections. Common in Section 12 of US labels
  or Section 5.1 of SmPCs.

EXAMPLE of correct output for one field:
  "fda_approval_date": {"value": "2021-06-04", "evidence_id": "E3"}
  "trade_names": [{"value": "Ozempic", "evidence_id": "E1"}]

CRITICAL: Use ONLY the aliases shown in brackets (E1, E2, ...). Do NOT invent aliases.
""".strip()

_REGISTRATIONS_INSTRUCTION = """
You are a pharmaceutical dossier extraction system.
Extract marketing authorization records from the provided context.

INSTRUCTIONS:
1. For each registration (one per country/region), provide: region (RU/EU/US/EAEU), status, MAH, identifiers (reg numbers), forms/strengths.
2. Each value field MUST include an evidence alias (E1, E2, ...) from the Available Evidence Candidates list.
3. Only populate regions for which real registration data exists in context.
4. Do NOT create empty rows — if region data is absent, omit that registration entirely.
5. If no registration data found for any region, return empty list.
CRITICAL: Use ONLY the aliases shown in brackets (E1, E2, ...). Never invent registration numbers.
""".strip()

_CLINICAL_INSTRUCTION = """
You are a pharmaceutical dossier extraction system.
Extract structured clinical study cards from the provided context.

IMPORTANT: Extract ALL studies present in the context, regardless of phase (Phase 1, 2, 3, 4, or NA).
Do NOT filter to only Phase 2/3 studies. Include observational studies, Phase 1 PK studies, etc.

CRITICAL — STUDY IDENTIFICATION:
- The study_id field (NCT number or other registry ID) is MANDATORY for each study card.
- If a study has an NCT ID (e.g., NCT12345678), you MUST extract it into study_id.
- If no registry ID can be found for a study, set study_id to null — but be aware that
  study cards without a study_id will be FILTERED OUT of the final dossier.
- Do NOT create study cards from general clinical summaries, label text, or review articles
  that describe overall drug efficacy without identifying specific individual studies.
- Each study card must correspond to ONE specific clinical study/trial, not a summary of multiple studies.

FOR EACH STUDY, extract:
1. title: official study title or brief title
2. study_id: NCT number (e.g., NCT12345678) or other registry ID — REQUIRED, extract from URL or text
3. phase: Phase 1, Phase 2, Phase 3, Phase 4, etc.
4. study_type: interventional, observational, etc.
5. enrollment: number of participants (integer)
6. countries: list of unique country names from locations data (e.g., ["United States", "Germany", "Japan"])
7. comparator: comparator arm — EXTRACT THIS CAREFULLY:
   - If placebo-controlled: "placebo"
   - If active comparator: name the specific drug (e.g., "rituximab", "standard chemotherapy")
   - If single-arm / no comparator: "none (single-arm)"
   - Look for arm labels, armGroups, interventions with type=ACTIVE_COMPARATOR or PLACEBO_COMPARATOR
   - Do NOT leave null if arm data is present in context
8. regimen_dosing: dosing details — EXTRACT THIS CAREFULLY:
   - Include dose (mg), frequency (daily/weekly/BID/etc.), route (oral/IV/SC), and cycle length if stated
   - Example: "30 mg orally twice weekly" or "1000 mg/m² IV on Day 1 of 21-day cycles"
   - Look in arm descriptions, intervention descriptions, eligibility criteria
   - Do NOT leave null if dosing text is present
9. primary_endpoint: primary outcome measure description
10. primary_result: primary endpoint result value with units
11. p_value: p-value for primary endpoint (as string, e.g., "0.001", "<0.0001")
12. confidence_interval: confidence interval (e.g., "95% CI: 0.65-0.82")
13. status: extract from OverallStatus field (e.g., "Completed", "Recruiting", "Terminated", "Active, not recruiting")
14. conclusion: EXTRACT THIS CAREFULLY — this is the most important field for screening:
    - If an explicit author/sponsor conclusion is stated in results or publications, use it verbatim (1-3 sentences max)
    - If primary endpoint numeric results exist (ORR, PFS, OS, p-value), synthesize a 1-2 sentence conclusion
      (e.g., "The study met its primary endpoint with ORR of 68% (p<0.001), demonstrating efficacy in...")
    - If only abstract or title-level information: synthesize from the study objective and phase
      (e.g., "Phase 2 study evaluating X in Y population; results not yet reported")
    - Set null ONLY if truly no clinical content is present beyond the NCT ID itself

CRITICAL RULES FOR EVIDENCE ALIASES:
- Every non-null field value MUST include at least one evidence alias from the candidates list (E1, E2, E3, ...).
- Use the alias exactly as shown: E1, E2, E3 — NOT "1", NOT "evidence_1", NOT full evidence IDs.
- If you cannot find a supporting snippet for a field, set value=null rather than omitting the alias.
- A study card where ALL fields have empty evidence_ids will be rejected entirely — do not submit such cards.
If a field is not stated in context, set it to null. Do NOT hallucinate N, p-values, or conclusions.
""".strip()

_PATENTS_INSTRUCTION = """
You are a pharmaceutical patent dossier extraction system.
Extract patent family records from the provided context.

FOR EACH PATENT FAMILY, extract:
1. family_id: INPADOC family ID if available, otherwise use representative publication number
2. representative_pub: the most informative publication number (EP/WO preferred over national)
3. priority_date: earliest priority date (YYYY-MM-DD format)
4. assignees: all patent assignees/applicants/owners
5. what_blocks: classify what the patent protects. Use EXACTLY one of:
   - "compound" = covers the active substance/molecule itself
   - "formulation" = covers a specific pharmaceutical formulation/dosage form
   - "method_of_use" = covers a specific therapeutic use/indication/method of treatment
   - "synthesis" = covers a manufacturing/synthesis process
   - "other" = if none of the above clearly applies
   Base classification on the patent title, abstract, and claims. If insufficient text, use "other".
6. technical_focus: detailed technical focus classification. Use EXACTLY one of:
   - "composition" = active pharmaceutical ingredient, new chemical entity, salt form
   - "formulation" = specific pharmaceutical formulation, excipients, dosage form design
   - "process_manufacturing" = manufacturing/synthesis process, production method, purification
   - "method_of_use" = therapeutic indication, treatment method, dosing regimen, patient population
   - "combination" = combination therapy, fixed-dose combination, co-administration
   - "salt_polymorph" = specific salt, polymorph, crystal form, co-crystal, hydrate, solvate
   - "dosage_form_delivery" = drug delivery system, sustained release, device, injection system
   - "intermediate_synthesis" = synthetic intermediate, key building block, reagent
   - "other" = if none of the above clearly applies
   Base on title, abstract, claims text. If unclear, use "other".
7. process_relevance: classify how relevant this patent is for synthesis/process understanding.
   Use EXACTLY one of:
   - "strong" = patent contains detailed synthesis routes, manufacturing examples, specific conditions
   - "moderate" = patent mentions process aspects but not as primary focus (e.g. formulation with brief process)
   - "weak" = patent references manufacturing generally but no useful process detail
   - "none" = patent has no synthesis/process content (pure composition, use, or device patent)
   Base on whether Examples/Preparations/Synthesis sections exist in the text.
8. legal_status_snapshot: legal status of the representative publication. Use EXACTLY one of:
   - "granted" = patent has been granted/issued
   - "pending" = application filed, not yet granted
   - "expired" = patent term has ended
   - "revoked" = patent was revoked after grant
   - "lapsed" = patent lapsed due to non-payment of fees
   - "unknown" = legal status not determinable from evidence
   Extract from legal status fields, kind codes (B1/B2=granted, A1/A2=pending), or explicit text.
9. summary: one-sentence description of what the patent covers (from title/abstract)
10. country_coverage: list of countries where patent publications exist (derive from publication
    numbers: EP->EP, WO->WO, US->US, JP->JP, CN->CN, RU->RU, etc.)
11. expiry_by_country: expiry dates per country. ONLY fill if expiry/legal status is EXPLICITLY
    stated in the context. Do NOT calculate expiry from priority+20 years. If no expiry data, leave empty.

CRITICAL RULES:
- Use ONLY evidence aliases (E1, E2, ...) from the candidates list
- Never hallucinate expiry dates or legal status
- Every field with a value MUST have at least one evidence alias
- If data is missing for a family, still include the family with available fields
- Group publications into families by INPADOC family_id when available
- For representative_pub: ALWAYS extract the publication number (e.g. EP1234567, WO2005/123456, US7654321)
  from the evidence text. Do NOT leave it null if any publication number is mentioned.
- For assignees: extract ALL assignee/applicant/owner names found in the evidence for this family.
  Do NOT return an empty list if assignee names appear in the evidence text.
- For summary: derive from the patent title or abstract text in the evidence. Set null ONLY if
  no title/abstract text exists in any evidence candidate for this family.
- For technical_focus and process_relevance: classify based on title, abstract, and claims text.
  If patent text has synthesis Examples with specific reagents/conditions, process_relevance = "strong".
  If only abstract is available, classify conservatively.
""".strip()

_SYNTHESIS_INSTRUCTION = """
You are a pharmaceutical chemistry extraction system.
Extract synthesis/manufacturing steps ONLY from patent text or official monograph sections
(Examples, Preparations, Manufacturing Process).

FOR EACH STEP, extract:
1. step_number: sequential step index (1, 2, 3, ...)
2. description: what happens in this step (reaction, purification, etc.)
3. reagents: starting materials and reagents used
4. intermediates: products/intermediates formed

CRITICAL RULES:
- Each description MUST reference an evidence alias (E1, E2, ...) from candidates
- Do NOT invent synthesis steps not in context
- Do NOT add reagents or intermediates not explicitly mentioned
- Focus on sections titled "Examples", "Preparations", "Synthesis", "Manufacturing Process"
- If the patent only describes composition/use without synthesis, return empty steps
""".strip()


# ── Evidence registry helper ──────────────────────────────────────────────────

def _ev_id(doc_id: str, page: Optional[int], text: str) -> str:
    """Generate a deterministic evidence_id.

    Uses full normalized text for hash (not text[:100]) to avoid collisions
    on chunks sharing identical preambles (disclaimers, table headers).
    12-char hex digest = 48-bit address space — collision-safe for large corpora.
    """
    import re as _re
    normalized = _re.sub(r"\s+", " ", text).strip()
    h = hashlib.sha256(f"{doc_id}:{page}:{normalized}".encode()).hexdigest()[:12]
    return f"ev_{doc_id[:8]}_{page or 0}_{h}"


# doc_kinds whose original artifact is JSON (not PDF)
_JSON_DOC_KINDS = {"pubchem", "label", "us_fda", "ctgov", "ctgov_results", "ctgov_protocol"}


def _build_evidence(doc_id: str, page: Optional[int], snippet: str,
                    title: Optional[str], source_url: Optional[str],
                    doc_kind: Optional[str] = None,
                    content_hash: Optional[str] = None,
                    locator: Optional[str] = None) -> DossierEvidence:
    ev_id = _ev_id(doc_id, page, snippet)
    mime_type = "application/json" if doc_kind and doc_kind.lower() in _JSON_DOC_KINDS else None
    return DossierEvidence(
        evidence_id=ev_id,
        doc_id=doc_id,
        title=title,
        source_url=source_url,
        page=page,
        snippet=snippet[:400],
        doc_kind=doc_kind or None,
        mime_type=mime_type,
        content_hash=content_hash,
        locator=locator,
    )


def _resolve_evidence_refs_via_alias(
    raw_aliases: List[str],
    alias_map: Dict[str, str],
) -> List[str]:
    """
    Resolve LLM-returned aliases (E1, E7, ...) → real evidence_ids.

    Strict: unknown aliases are logged and dropped — NO prefix/fuzzy matching.
    This replaces the old prefix-match strategy that could silently bind the
    wrong evidence chunk when multiple candidates shared a doc_id prefix.
    """
    resolved: List[str] = []
    seen: set = set()
    for alias in raw_aliases:
        if not alias:
            continue
        alias_upper = alias.strip().upper()
        if alias_upper in alias_map:
            ev_id = alias_map[alias_upper]
            if ev_id not in seen:
                resolved.append(ev_id)
                seen.add(ev_id)
        else:
            logger.warning(
                "unknown_evidence_alias alias=%s known_range=E1..E%d",
                alias, len(alias_map),
            )
    return resolved


def _ev_to_evidenced_value(llm_val: Optional[_EvidencedValueLLM],
                            alias_map: Dict[str, str]) -> Optional[EvidencedValue]:
    """
    Convert LLM-extracted value + evidence alias(es) → EvidencedValue.

    Uses alias_map (E1→real_ev_id) instead of candidates_map prefix matching.
    A value with no linked evidence is retained with empty refs — the caller's
    sanitizer decides whether to strip it and emit a DossierUnknown.
    """
    if llm_val is None or llm_val.value is None:
        return None
    # Collect raw aliases from both fields
    raw_aliases: List[str] = list(llm_val.evidence_ids or [])
    if llm_val.evidence_id:
        raw_aliases.insert(0, llm_val.evidence_id)
    refs = _resolve_evidence_refs_via_alias(raw_aliases, alias_map)
    return EvidencedValue(value=llm_val.value, evidence_refs=refs)


def _ev_list(llm_list: List[_EvidencedValueLLM],
             alias_map: Dict[str, str]) -> List[EvidencedValue]:
    result = []
    for item in llm_list:
        ev = _ev_to_evidenced_value(item, alias_map)
        if ev is not None:
            result.append(ev)
    return result


# ── Sprint 12: Clinical study ID validation ──────────────────────────────────

_NCT_PATTERN = re.compile(r"NCT\d{7,8}", re.IGNORECASE)

# Doc kinds that can only enrich existing studies, NOT create standalone cards
_ENRICHMENT_ONLY_DOC_KINDS = {"scientific_pmc", "scientific_pdf", "publication", "preprint"}

# Doc kinds that are authoritative CTGov sources (can create study cards)
_CTGOV_DOC_KINDS = {"ctgov_protocol", "ctgov_results", "ctgov", "ctgov_documents", "trial_registry"}


def _is_valid_clinical_card(study: "DossierClinicalStudy") -> bool:
    """Sprint 12 WS1: A study card is valid for final output only if it has a study_id.

    Cards without study_id are noise — typically extracted from label summaries,
    review text, or mixed clinical context that doesn't identify a specific trial.
    """
    if study.study_id is None:
        return False
    if study.study_id.value is None:
        return False
    val = str(study.study_id.value).strip()
    if not val:
        return False
    return True


def _extract_nct_ids_from_corpus(documents_dir: Path) -> Set[str]:
    """Sprint 12 WS1: Pre-scan corpus for NCT IDs to build a study candidate registry.

    Scans all chunk JSON files in the documents directory for NCT\\d{7,8} patterns.
    Returns a set of unique NCT IDs found in the corpus.
    """
    nct_ids: Set[str] = set()
    for doc_path in documents_dir.glob("*.json"):
        try:
            with open(doc_path, encoding="utf-8") as f:
                doc = json.load(f)
            # Check metainfo for source_url with NCT
            meta = doc.get("metainfo", {})
            source_url = meta.get("source_url", "")
            if source_url:
                for m in _NCT_PATTERN.finditer(source_url):
                    nct_ids.add(m.group(0).upper())
            # Check title
            title = meta.get("title", "")
            if title:
                for m in _NCT_PATTERN.finditer(title):
                    nct_ids.add(m.group(0).upper())
            # Scan text content (first 5000 chars of each page to avoid too much scanning)
            pages = doc.get("content", {}).get("pages", [])
            for page in pages:
                text = page.get("text", "")[:5000]
                for m in _NCT_PATTERN.finditer(text):
                    nct_ids.add(m.group(0).upper())
        except Exception:
            continue
    return nct_ids


# ── Sprint 12 WS3: Patent family validation ──────────────────────────────────

# Patterns that indicate synthetic/empty patent family shells
_SYNTHETIC_PATENT_PATTERNS = [
    "no families found",
    "no patent families",
    "no patents found",
    "not found",
    "no relevant patents",
    "no active patents",
    "off-patent",
    "expired patent",
    "patent expired",
]


def _is_synthetic_patent_family(family: "DossierPatentFamily") -> bool:
    """Sprint 12 WS3: Detect synthetic empty shell patent families.

    These are families created by the LLM with a family_id like
    "ibuprofen (EPO OPS: no families found)" that contain no real data.
    """
    fid = (family.family_id or "").lower()
    for pattern in _SYNTHETIC_PATENT_PATTERNS:
        if pattern in fid:
            return True
    return False


def _is_minimally_valid_patent_family(family: "DossierPatentFamily") -> bool:
    """Sprint 12 WS3: Check if a patent family has minimum viable content.

    A family must have at least:
    - representative_pub OR priority_date (basic identification)
    AND at least one of:
    - assignees (who owns it)
    - summary (what it covers)
    - what_blocks (what it protects)

    Without this minimum, the family is just an empty shell and should not
    appear in the final dossier as if it's usable IP information.
    """
    # Check synthetic patterns first
    if _is_synthetic_patent_family(family):
        return False

    # Must have basic identification
    has_pub = family.representative_pub and family.representative_pub.value
    has_priority = family.priority_date and family.priority_date.value
    if not (has_pub or has_priority):
        return False

    # Must have at least one content field
    has_assignees = bool(family.assignees)
    has_summary = family.summary and family.summary.value
    has_blocks = family.what_blocks and family.what_blocks.value
    if not (has_assignees or has_summary or has_blocks):
        return False

    return True


# ── DossierReportGenerator ────────────────────────────────────────────────────

class DossierReportGenerator:
    """
    Generates a DossierReport v3.0 from indexed document corpus.

    Usage:
        gen = DossierReportGenerator(vector_db_dir, documents_dir, inn="semaglutide", ...)
        report = gen.generate(case_id="...", run_id="...", deadline=time.time()+3600)
    """

    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        inn: str,
        tenant_id: Optional[str] = None,
        case_id: Optional[str] = None,
    ):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.inn = inn.strip() if inn else ""
        self.inn_lower = self.inn.lower()
        self.tenant_id = tenant_id
        self.case_id = case_id
        self.retriever = HybridRetriever(vector_db_dir, documents_dir)
        self.api = APIProcessor(provider=os.getenv("DDKIT_LLM_PROVIDER", "openai"))
        self.answering_model = os.getenv("DDKIT_ANSWER_MODEL", None)
        self.evidence_builder = EvidenceCandidatesBuilder()
        # Evidence registry: evidence_id → DossierEvidence
        self._evidence_registry: Dict[str, DossierEvidence] = {}

    def _retrieve(self, question: str, doc_kinds: List[str], top_k: int = 40) -> List[Dict[str, Any]]:
        """Retrieve top-K evidence candidates for a question with given doc_kind filter.

        Retries up to 2 times with backoff on rate-limit (429) errors.
        Raises RateLimitExhausted after exhausting retries so the job can be deferred.
        """
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                results = self.retriever.retrieve_by_case(
                    query=f"{question} {self.inn}",
                    doc_kind=doc_kinds if doc_kinds else None,
                    top_n=top_k,
                    tenant_id=self.tenant_id,
                    case_id=self.case_id,
                )
                return results or []
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = "429" in exc_str or "rate_limit" in exc_str.lower()
                if is_rate_limit and attempt < max_retries:
                    wait = 2 ** attempt * 10  # 10s, 20s
                    logger.warning(
                        "retrieve 429 rate-limit for question=%r, retry %d/%d in %ds",
                        question[:60], attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                if is_rate_limit:
                    logger.error(
                        "retrieve 429 rate-limit EXHAUSTED for question=%r after %d retries",
                        question[:60], max_retries,
                    )
                    raise RateLimitExhausted(
                        f"OpenAI 429 after {max_retries} retries on retrieve: {question[:60]}"
                    ) from exc
                logger.warning("retrieve failed for question=%r: %s", question[:60], exc)
                return []

    def _candidates_map(self, retrieved: List[Dict[str, Any]]) -> Dict[str, DossierEvidence]:
        """Build evidence_id → DossierEvidence from retrieved chunks, register globally."""
        candidates: Dict[str, DossierEvidence] = {}
        for item in retrieved:
            doc_id = item.get("doc_id", "")
            page = item.get("page")
            text = item.get("text", "")
            title = item.get("doc_title") or item.get("title", "")
            source_url = item.get("source_url", "")
            doc_kind = item.get("doc_kind")
            if not text:
                continue
            ev = _build_evidence(doc_id, page, text, title, source_url, doc_kind=doc_kind)
            candidates[ev.evidence_id] = ev
            self._evidence_registry[ev.evidence_id] = ev
        return candidates

    def _build_alias_map(
        self,
        candidates_map: Dict[str, DossierEvidence],
        token_budget: int = 0,
    ) -> Tuple[Dict[str, str], str]:
        """
        Build alias mapping E1..EN → real evidence_id and formatted prompt text.

        Returns (alias_map, formatted_candidates_str).
        Using short aliases instead of raw ev_ids prevents LLM from truncating
        or mutating long IDs (the old prefix-match risk).

        When *token_budget* > 0, stops adding candidates once the cumulative
        token count exceeds the budget.  Candidates are already ranked by
        retrieval score (highest first), so the budget naturally keeps the
        most relevant evidence and trims the tail.
        """
        import tiktoken as _tiktoken

        alias_map: Dict[str, str] = {}
        lines: List[str] = []
        running_tokens = 0
        effective_budget = token_budget or int(os.getenv("DDKIT_EVIDENCE_TOKEN_BUDGET", "0"))
        _enc = _tiktoken.get_encoding("o200k_base") if effective_budget > 0 else None

        for i, (ev_id, ev) in enumerate(candidates_map.items(), 1):
            alias = f"E{i}"
            pg = f"p.{ev.page}" if ev.page else "p.?"
            src = ev.title or ev.source_url or ev.doc_id or "?"
            line = f"[{alias}] {src[:60]} ({pg}) | {ev.snippet[:200]}"

            if effective_budget > 0:
                line_tokens = len(_enc.encode(line))
                if running_tokens + line_tokens > effective_budget and lines:
                    logger.info(
                        "evidence_token_budget_reached after E%d, budget=%d, used=%d",
                        i - 1, effective_budget, running_tokens,
                    )
                    break
                running_tokens += line_tokens

            alias_map[alias] = ev_id
            lines.append(line)
        return alias_map, "\n".join(lines)

    def _call_llm(self, instruction: str, context: str, question: str,
                   candidates_str: str, schema_class) -> Optional[Any]:
        """Call LLM with structured output schema; returns parsed Pydantic object or None.

        Retries up to 3 times with exponential backoff on rate-limit (429) errors.
        Raises RateLimitExhausted after exhausting retries so the job can be deferred.
        """
        schema_str = str(schema_class.model_json_schema())
        system_prompt = (
            f"{instruction}\n\n"
            f"Your answer MUST be valid JSON matching this schema:\n```\n{schema_str}\n```\n\n"
            "CRITICAL: Only use evidence aliases (E1, E2, ...) that appear in the Available Evidence Candidates section below."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Available Evidence Candidates:\n{candidates_str}\n\n"
            "Generate structured JSON answer."
        )
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                result_dict = self.api.send_message(
                    model=self.answering_model,
                    system_content=system_prompt,
                    human_content=user_prompt,
                    is_structured=True,
                    response_format=schema_class,
                )
                # send_message returns .dict() — re-parse into Pydantic object
                if isinstance(result_dict, dict):
                    return schema_class.model_validate(result_dict)
                return result_dict
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = "429" in exc_str or "rate_limit" in exc_str.lower()
                if is_rate_limit and attempt < max_retries:
                    wait = 2 ** attempt * 10  # 10s, 20s, 40s
                    logger.warning(
                        "LLM 429 rate-limit for question=%r, retry %d/%d in %ds",
                        question[:60], attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                if is_rate_limit:
                    logger.error(
                        "LLM 429 rate-limit EXHAUSTED for question=%r after %d retries",
                        question[:60], max_retries,
                    )
                    raise RateLimitExhausted(
                        f"OpenAI 429 after {max_retries} retries on LLM call: {question[:60]}"
                    ) from exc
                logger.warning("LLM call failed for question=%r: %s", question[:60], exc)
                return None

    def _context_str(self, retrieved: List[Dict[str, Any]]) -> str:
        parts = []
        for item in retrieved:
            page = item.get("page", "?")
            text = item.get("text", "")
            parts.append(f'Page {page}:\n"""\n{text[:800]}\n"""')
        return "\n\n---\n\n".join(parts[:25])  # limit context

    def _add_unknown(self, unknowns: List[DossierUnknown], field_path: str,
                     reason_code: str, message: str, next_action: Optional[str] = None) -> None:
        unknowns.append(DossierUnknown(
            field_path=field_path,
            reason_code=reason_code,
            message=message,
            suggested_next_action=next_action,
        ))

    @staticmethod
    def _sanitize_evidenced_value(
        ev: Optional[EvidencedValue],
        field_path: str,
        unknowns: List[DossierUnknown],
    ) -> Optional[EvidencedValue]:
        """Strip values without evidence refs — enforce 'no facts without proof' contract.

        If an EvidencedValue has value != None but evidence_refs == [],
        strip the value and emit a DossierUnknown. This prevents the dossier
        from containing claims that look like facts but have no evidence trail.
        """
        if ev is None:
            return None
        if ev.value is not None and not ev.evidence_refs:
            logger.info(
                "sanitize_strip field=%s value=%r — no evidence refs resolved",
                field_path, str(ev.value)[:60],
            )
            unknowns.append(DossierUnknown(
                field_path=field_path,
                reason_code="NO_EVIDENCE_IN_CORPUS",
                message=(
                    f"Value '{str(ev.value)[:80]}' was extracted but no evidence aliases "
                    "resolved to valid candidates. Value stripped to enforce evidence contract."
                ),
                suggested_next_action="Re-index source documents or verify evidence aliases.",
            ))
            return None
        return ev

    def _has_patent_corpus(self) -> bool:
        """Check if any patent doc_kinds are present in the downloaded corpus."""
        patent_kinds = {"patent_family", "ops", "patent_pdf", "ru_patent_pdf", "ru_patent_fips", "patent", "patent_family_summary", "patent_legal_events", "patent_expiry_us"}
        for doc_path in self.documents_dir.glob("*.json"):
            try:
                with open(doc_path, encoding="utf-8") as f:
                    doc = json.load(f)
                kind = doc.get("metainfo", {}).get("doc_kind", "")
                if kind in patent_kinds:
                    return True
            except Exception:
                continue
        return False

    # ── Deterministic extractors (Sprint 15) ─────────────────────────────────

    # Patterns for FDA approval date, ordered by specificity (most specific first).
    # Group convention: patterns yield (year, month, day), (year, month_word, day),
    #   (year,) or (month_word, day, year).  The extractor normalises all forms.
    _FDA_DATE_PATTERNS = [
        # Exact marker injected by processor.ts fetchDrugsFdaApprovalDate()
        re.compile(
            r"FDA Approval Date\s*(?:\(Drugs@FDA\))?\s*:\s*(\d{4})[-/]?(\d{2})[-/]?(\d{2})",
            re.IGNORECASE,
        ),
        # "Original Approval Date: 2000-12-18" or "Approval Date: 20001218"
        re.compile(
            r"(?:original\s+)?approval\s+date\s*:\s*(\d{4})[-/]?(\d{2})[-/]?(\d{2})",
            re.IGNORECASE,
        ),
        # "Initial U.S. Approval: 2023" (year-only, common in FDA labels)
        re.compile(
            r"initial\s+u\.?s\.?\s+approval\s*:\s*(\d{4})\b",
            re.IGNORECASE,
        ),
        # "first approved: 2000" (year-only fallback)
        re.compile(
            r"first\s+approved\s*:\s*(\d{4})\b",
            re.IGNORECASE,
        ),
    ]

    _MONTH_MAP = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }

    # Word-month patterns — tried AFTER numeric patterns.
    # "Approved May 19, 2023" / "approved on January 15, 2023"
    _FDA_DATE_WORD_PATTERNS = [
        re.compile(
            r"approv(?:ed|al)\s+(?:on\s+)?(?:in\s+)?"
            r"(january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
            r"\s+(\d{1,2}),?\s+(\d{4})",
            re.IGNORECASE,
        ),
        # "Approved 19 May 2023"
        re.compile(
            r"approv(?:ed|al)\s+(?:on\s+)?(\d{1,2})\s+"
            r"(january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
            r",?\s+(\d{4})",
            re.IGNORECASE,
        ),
        # "Marketing Authorization Date: May 19, 2023"
        re.compile(
            r"(?:marketing\s+)?authorization\s+date\s*:\s*"
            r"(january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
            r"\s+(\d{1,2}),?\s+(\d{4})",
            re.IGNORECASE,
        ),
    ]

    def _extract_fda_approval_date_deterministic(
        self,
    ) -> Optional[EvidencedValue]:
        """Scan ALL label/us_fda chunks for FDA approval date via regex.

        Bypasses top-K retrieval — reads chunk JSONs directly from documents_dir.
        Returns EvidencedValue with evidence ref, or None.
        """
        target_doc_kinds = {"label", "us_fda"}
        best_date: Optional[str] = None
        best_evidence: Optional[DossierEvidence] = None

        for doc_path in self.documents_dir.glob("*.json"):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                continue

            metainfo = doc.get("metainfo", {})
            doc_kind = (metainfo.get("doc_kind") or "").lower()
            if doc_kind not in target_doc_kinds:
                continue
            if self.tenant_id and metainfo.get("tenant_id") != self.tenant_id:
                continue
            if self.case_id and metainfo.get("case_id") != self.case_id:
                continue

            doc_id = metainfo.get("doc_id", doc_path.stem)
            doc_title = metainfo.get("title") or doc_id

            content = doc.get("content", {})
            raw_chunks = content.get("chunks") or content.get("pages") or []

            for chunk in raw_chunks:
                text = chunk.get("text", "")
                if not text:
                    continue

                # --- Numeric patterns first ---
                for pattern in self._FDA_DATE_PATTERNS:
                    m = pattern.search(text)
                    if not m:
                        continue

                    groups = m.groups()
                    if len(groups) == 3:
                        year, month, day = groups
                        date_str = f"{year}-{month}-{day}"
                    elif len(groups) == 1:
                        # Year-only fallback
                        date_str = f"{groups[0]}-01-01"
                    else:
                        continue

                    # Validate year range
                    try:
                        yr = int(date_str[:4])
                        if yr < 1900 or yr > 2030:
                            continue
                    except ValueError:
                        continue

                    page = chunk.get("page")
                    ev = _build_evidence(
                        doc_id, page, text, doc_title,
                        source_url=metainfo.get("source_url"),
                        doc_kind=doc_kind,
                    )
                    self._evidence_registry[ev.evidence_id] = ev

                    # Prefer the most specific match (3-group > 1-group)
                    if best_date is None or len(groups) > 1:
                        best_date = date_str
                        best_evidence = ev

                    # Found a full date — no need to keep scanning this chunk
                    if len(groups) == 3:
                        break

                # --- Word-month patterns (e.g. "Approved May 19, 2023") ---
                if best_date and len(best_date) > 5 and best_date[5:] != "01-01":
                    # Already have a full date, skip word patterns
                    continue

                for wp in self._FDA_DATE_WORD_PATTERNS:
                    wm = wp.search(text)
                    if not wm:
                        continue
                    wgroups = wm.groups()
                    # Determine order: (month_word, day, year) or (day, month_word, year)
                    if wgroups[0].isdigit():
                        # (day, month_word, year)
                        day_s, month_word, year_s = wgroups
                    else:
                        # (month_word, day, year)
                        month_word, day_s, year_s = wgroups
                    month_num = self._MONTH_MAP.get(month_word.lower())
                    if not month_num:
                        continue
                    try:
                        yr = int(year_s)
                        dy = int(day_s)
                        if yr < 1900 or yr > 2030 or dy < 1 or dy > 31:
                            continue
                    except ValueError:
                        continue
                    date_str = f"{year_s}-{month_num}-{int(day_s):02d}"

                    page = chunk.get("page")
                    ev = _build_evidence(
                        doc_id, page, text, doc_title,
                        source_url=metainfo.get("source_url"),
                        doc_kind=doc_kind,
                    )
                    self._evidence_registry[ev.evidence_id] = ev
                    best_date = date_str
                    best_evidence = ev
                    break

        if best_date and best_evidence:
            logger.info(
                "fda_approval_date_deterministic found=%s doc=%s",
                best_date, best_evidence.doc_id,
            )
            return EvidencedValue(
                value=best_date,
                evidence_refs=[best_evidence.evidence_id],
            )

        logger.info("fda_approval_date_deterministic: no match in %d label/us_fda docs",
                     sum(1 for _ in self.documents_dir.glob("*.json")))
        return None

    # Patterns for RU registration status from grls_card chunks.
    # Matches the injected marker "RU Reg Status: <text> | RU Reg Expiry: DD.MM.YYYY"
    # or native GRLS text "Действует до DD.MM.YYYY" / "Дата окончания действия DD.MM.YYYY".
    _RU_STATUS_PATTERNS = [
        # Injected marker: "RU Reg Status: Действует | RU Reg Expiry: 03.12.2030"
        re.compile(
            r"RU Reg Status:\s*([^\|]+?)(?:\s*\||\s*$)",
            re.IGNORECASE,
        ),
        # Injected expiry marker: "RU Reg Expiry: DD.MM.YYYY"
        re.compile(
            r"RU Reg Expiry:\s*(\d{2}\.\d{2}\.\d{4})",
            re.IGNORECASE,
        ),
        # Native GRLS text: "Дата окончания действия 03.12.2030"
        re.compile(
            r"Дата окончания действия\s+(\d{2}\.\d{2}\.\d{4})",
        ),
        # Native GRLS text: "Действует до 03.12.2030"
        re.compile(
            r"Действует до\s+(\d{2}\.\d{2}\.\d{4})",
        ),
        # Table cell format seen in semaglutide: "Дата регистрации 03.12.2025 Дата окончания действия 03.12.2030"
        re.compile(
            r"Дата окончания действия\s+(\d{2}\.\d{2}\.\d{4})",
        ),
    ]

    def _extract_ru_reg_status_deterministic(self) -> Optional[tuple]:
        """Scan grls_card chunks for RU registration status and expiry date.

        Returns (status_str, expiry_str, evidence) or None.
        Checks for:
          1. Injected marker block added by grls-service /proxy/card
          2. Native GRLS text patterns (Дата окончания действия, Действует до)
        """
        target_doc_kinds = {"grls_card", "grls", "ru_instruction"}
        best_status: Optional[str] = None
        best_expiry: Optional[str] = None
        best_evidence: Optional[DossierEvidence] = None

        json_files = list(self.documents_dir.glob("*.json"))
        for doc_path in json_files:
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                continue

            metainfo = doc.get("metainfo", {})
            doc_kind = (metainfo.get("doc_kind") or "").lower()
            if doc_kind not in target_doc_kinds:
                continue
            if self.tenant_id and metainfo.get("tenant_id") != self.tenant_id:
                continue
            if self.case_id and metainfo.get("case_id") != self.case_id:
                continue

            doc_id = metainfo.get("doc_id", doc_path.stem)
            doc_title = metainfo.get("title") or doc_id

            content = doc.get("content", {})
            raw_chunks = content.get("chunks") or content.get("pages") or []

            for chunk in raw_chunks:
                text = chunk.get("text", "")
                if not text:
                    continue

                page = chunk.get("page")
                ev = None

                # Check injected marker first (highest confidence)
                m_status = self._RU_STATUS_PATTERNS[0].search(text)
                m_expiry = self._RU_STATUS_PATTERNS[1].search(text)
                if m_status or m_expiry:
                    if ev is None:
                        ev = _build_evidence(doc_id, page, text, doc_title,
                                             source_url=metainfo.get("source_url"),
                                             doc_kind=doc_kind)
                        self._evidence_registry[ev.evidence_id] = ev
                    if m_status and not best_status:
                        best_status = m_status.group(1).strip()
                        best_evidence = ev
                    if m_expiry and not best_expiry:
                        best_expiry = m_expiry.group(1).strip()
                    continue

                # Check native GRLS text patterns
                for pat in self._RU_STATUS_PATTERNS[2:]:
                    m = pat.search(text)
                    if m:
                        if ev is None:
                            ev = _build_evidence(doc_id, page, text, doc_title,
                                                 source_url=metainfo.get("source_url"),
                                                 doc_kind=doc_kind)
                            self._evidence_registry[ev.evidence_id] = ev
                        val = m.group(1).strip()
                        # Pattern 2/3 capture expiry date; pattern 4 also expiry
                        if not best_expiry and re.match(r"\d{2}\.\d{2}\.\d{4}", val):
                            best_expiry = val
                            best_evidence = ev
                        elif not best_status:
                            best_status = val
                            best_evidence = ev
                        break

            if best_status and best_expiry:
                break  # found both in this doc — stop scanning

        if best_status or best_expiry:
            parts = []
            if best_status:
                parts.append(best_status)
            if best_expiry:
                parts.append(f"до {best_expiry}")
            combined = " ".join(parts) if parts else best_status or best_expiry
            logger.info(
                "ru_reg_status_deterministic found: status=%r expiry=%r doc=%s combined=%r",
                best_status, best_expiry,
                best_evidence.doc_id if best_evidence else "?",
                combined,
            )
            return combined, best_evidence
        return None

    # WSx.4: Regex to detect EAEU mutual-recognition marker in RU reg numbers.
    # Format: ЛП-№(013035)-(РГ-RU) or ЛП-013035-РГ-RU or variants.
    # РГ = "Регистрационное Государство" / mutual-recognition route.
    _RU_EAEU_MARKER_RE = re.compile(
        r"ЛП-[^\s]*[-\(]РГ-RU[\)\s]?",
        re.IGNORECASE,
    )

    def _detect_ru_eaeu_marker(self) -> Optional[DossierEvidence]:
        """WSx.4: Scan grls_card / grls chunks for РГ-RU mutual-recognition reg number.

        If found, returns the evidence for that chunk so the caller can emit
        an EAEU_RELATED_RU_MARKER informational unknown.

        Returns DossierEvidence if marker found, else None.
        """
        target_doc_kinds = {"grls_card", "grls", "ru_instruction"}
        json_files = list(self.documents_dir.glob("*.json"))
        for doc_path in json_files:
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                continue
            metainfo = doc.get("metainfo", {})
            doc_kind = (metainfo.get("doc_kind") or "").lower()
            if doc_kind not in target_doc_kinds:
                continue
            if self.tenant_id and metainfo.get("tenant_id") != self.tenant_id:
                continue
            if self.case_id and metainfo.get("case_id") != self.case_id:
                continue

            doc_id = metainfo.get("doc_id", doc_path.stem)
            doc_title = metainfo.get("title") or doc_id
            content = doc.get("content", {})
            raw_chunks = content.get("chunks") or content.get("pages") or []
            for chunk in raw_chunks:
                text = chunk.get("text", "")
                if not text:
                    continue
                m = self._RU_EAEU_MARKER_RE.search(text)
                if m:
                    ev = _build_evidence(
                        doc_id, chunk.get("page"), text, doc_title,
                        source_url=metainfo.get("source_url"),
                        doc_kind=doc_kind,
                    )
                    self._evidence_registry[ev.evidence_id] = ev
                    logger.info(
                        "ru_eaeu_marker detected: reg_no_fragment=%r doc=%s",
                        m.group(0), doc_id,
                    )
                    return ev
        return None

    # Sprint 18: PubChem chemistry deterministic extraction patterns.
    # Matches OCR-rendered PubChem HTML (with possible spaces inserted by OCR).
    # Formula: capture until end-of-line or next label (e.g. "\nMolecular Weight:").
    # The alternation stops at newline or at a capital-word boundary that looks like
    # another label, keeping the formula isolated (e.g. "C18H19F2N5O4").
    _PUBCHEM_FORMULA_RE = re.compile(
        r"Molecular\s+Formula:\s*([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*?)(?=\s*\n|\s+[A-Z][a-z]|\Z)",
        re.IGNORECASE,
    )
    _PUBCHEM_MW_RE = re.compile(r"Molecular\s+Weight:\s*([\d\.,]+)", re.IGNORECASE)
    _PUBCHEM_SMILES_RE = re.compile(r"Canonical\s+SMILES:\s*(\S+)", re.IGNORECASE)
    # InChIKey has format XXXXXXXXXXXXXX-XXXXXXXXXX-X (14-10-1 uppercase letters+digits).
    # OCR may insert a space anywhere within a segment. We allow [\s]? inside each segment.
    _PUBCHEM_INCHIKEY_RE = re.compile(
        r"InChIKey:\s*([A-Z0-9]{2,14}\s?[A-Z0-9]{0,12}[-–][A-Z0-9]{2,10}\s?[A-Z0-9]{0,8}[-–][A-Z0-9])",
        re.IGNORECASE,
    )

    def _extract_pubchem_chemistry_deterministic(self) -> Optional[Dict[str, Any]]:
        """Scan pubchem doc chunks directly for chemistry fields.

        Bypasses LLM retrieval — the single pubchem chunk is often outcompeted
        by label/GRLS chunks in top-K retrieval. Returns dict with keys:
        chemical_formula, molecular_weight, smiles, inchi_key (each EvidencedValue),
        or None if no pubchem doc found.
        """
        json_files = list(self.documents_dir.glob("*.json"))
        for doc_path in json_files:
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                continue

            metainfo = doc.get("metainfo", {})
            doc_kind = (metainfo.get("doc_kind") or "").lower()
            if doc_kind != "pubchem":
                continue
            if self.tenant_id and metainfo.get("tenant_id") != self.tenant_id:
                continue
            if self.case_id and metainfo.get("case_id") != self.case_id:
                continue

            doc_id = metainfo.get("doc_id", doc_path.stem)
            doc_title = metainfo.get("title") or doc_id
            source_url = metainfo.get("source_url", "")

            content = doc.get("content", {})
            raw_chunks = content.get("chunks") or content.get("pages") or []

            for chunk in raw_chunks:
                text = chunk.get("text", "")
                if not text:
                    continue

                page = chunk.get("page")
                results: Dict[str, Any] = {}

                m_formula = self._PUBCHEM_FORMULA_RE.search(text)
                m_mw = self._PUBCHEM_MW_RE.search(text)
                m_smiles = self._PUBCHEM_SMILES_RE.search(text)
                m_inchikey = self._PUBCHEM_INCHIKEY_RE.search(text)

                if not any([m_formula, m_mw, m_smiles, m_inchikey]):
                    continue

                ev = _build_evidence(doc_id, page, text, doc_title,
                                     source_url=source_url, doc_kind=doc_kind)
                self._evidence_registry[ev.evidence_id] = ev

                if m_formula:
                    # Collapse OCR spaces inside formula: "C18H19 F2N5O4" -> "C18H19F2N5O4"
                    raw = m_formula.group(1)
                    formula = re.sub(r"\s+", "", raw)
                    results["chemical_formula"] = EvidencedValue(value=formula, confidence=0.98,
                                                                  evidence_refs=[ev.evidence_id])
                if m_mw:
                    results["molecular_weight"] = EvidencedValue(value=m_mw.group(1).strip(),
                                                                  confidence=0.98,
                                                                  evidence_refs=[ev.evidence_id])
                if m_smiles:
                    results["smiles"] = EvidencedValue(value=m_smiles.group(1).strip(),
                                                       confidence=0.95,
                                                       evidence_refs=[ev.evidence_id])
                if m_inchikey:
                    # Remove OCR-inserted spaces from InChIKey
                    raw_ik = re.sub(r"\s+", "", m_inchikey.group(1))
                    # Normalise dashes (OCR may use en-dash)
                    raw_ik = re.sub(r"[–—]", "-", raw_ik)
                    results["inchi_key"] = EvidencedValue(value=raw_ik, confidence=0.97,
                                                          evidence_refs=[ev.evidence_id])

                if results:
                    logger.info(
                        "pubchem_chemistry_deterministic found fields=%s doc=%s",
                        list(results.keys()), doc_id,
                    )
                    return results

        return None

    # Sprint 17: Regex patterns for deterministic patent expiry extraction.
    # patent_legal_events chunks: "Patent N: EP1234567B1 ... Country: EP ... Estimated expiry: 2045-11-28 (method: ...)"
    _PATENT_LEGAL_EVENT_EXPIRY_RE = re.compile(
        r"Patent\s+\d+:\s*([A-Z]{2}\d{5,}[A-Z]?\d?)\s*"   # patent number
        r".*?Country:\s*([A-Z]{2})"                          # country code
        r".*?Estimated expiry:\s*(\d{4}-\d{2}-\d{2})",      # expiry date
        re.DOTALL,
    )
    # patent_expiry_us chunks (Orange Book table): "US12345678  2038-10-10  U-2628 ..."
    _PATENT_EXPIRY_US_RE = re.compile(
        r"(US\d{7,})\s+(\d{4}-\d{2}-\d{2})",
    )
    # ru_patent_fips chunks: embedded JSON with "doc_id":"RU2803237C2_20230911"
    # ... "jurisdiction":"RU" ... "expiry_date":"2039-10-25"
    # OCR may insert spaces within dates/numbers, and newlines within JSON.
    _RU_FIPS_DOC_ID_RE = re.compile(
        r'"doc_id"\s*:\s*"((?:RU|EA)[\dA-Z ]+?)_[\d ]+?"',
    )
    _RU_FIPS_EXPIRY_DATE_RE = re.compile(
        r'"expiry_date"\s*:\s*"([\d\-\s]+?)"',
    )
    _RU_FIPS_JURISDICTION_RE = re.compile(
        r'"jurisdiction"\s*:\s*"([A-Z]{2})"',
    )

    def _extract_patent_expiry_deterministic(self) -> Dict[str, Dict[str, str]]:
        """Scan patent_legal_events, patent_expiry_us, and ru_patent_fips chunks for expiry dates.

        Returns dict: patent_number_normalised -> {"country": "XX", "expiry": "YYYY-MM-DD",
                                                    "evidence": DossierEvidence}
        Used to patch LLM-extracted patent families that are missing expiry_by_country.
        """
        target_doc_kinds = {"patent_legal_events", "patent_expiry_us", "ru_patent_fips"}
        expiry_map: Dict[str, Dict[str, Any]] = {}

        json_files = list(self.documents_dir.glob("*.json"))
        for doc_path in json_files:
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                continue

            metainfo = doc.get("metainfo", {})
            doc_kind = (metainfo.get("doc_kind") or "").lower()
            if doc_kind not in target_doc_kinds:
                continue
            if self.tenant_id and metainfo.get("tenant_id") != self.tenant_id:
                continue
            if self.case_id and metainfo.get("case_id") != self.case_id:
                continue

            doc_id = metainfo.get("doc_id", doc_path.stem)
            doc_title = metainfo.get("title") or doc_id

            content = doc.get("content", {})
            raw_chunks = content.get("chunks") or content.get("pages") or []

            for chunk in raw_chunks:
                text = chunk.get("text", "")
                if not text:
                    continue
                page = chunk.get("page")

                if doc_kind == "patent_legal_events":
                    # Each chunk may have multiple patent blocks
                    for m in self._PATENT_LEGAL_EVENT_EXPIRY_RE.finditer(text):
                        pat_num = m.group(1).strip()
                        country = m.group(2).strip()
                        expiry = m.group(3).strip()
                        norm = self._normalise_patent_number(pat_num)
                        if norm not in expiry_map:
                            ev = _build_evidence(doc_id, page, text, doc_title,
                                                 source_url=metainfo.get("source_url"),
                                                 doc_kind=doc_kind)
                            self._evidence_registry[ev.evidence_id] = ev
                            expiry_map[norm] = {"country": country, "expiry": expiry, "evidence": ev}

                elif doc_kind == "patent_expiry_us":
                    for m in self._PATENT_EXPIRY_US_RE.finditer(text):
                        pat_num = m.group(1).strip()
                        expiry = m.group(2).strip()
                        norm = self._normalise_patent_number(pat_num)
                        if norm not in expiry_map:
                            ev = _build_evidence(doc_id, page, text, doc_title,
                                                 source_url=metainfo.get("source_url"),
                                                 doc_kind=doc_kind)
                            self._evidence_registry[ev.evidence_id] = ev
                            expiry_map[norm] = {"country": "US", "expiry": expiry, "evidence": ev}

                elif doc_kind == "ru_patent_fips":
                    # FIPS chunks contain embedded JSON items with doc_id, jurisdiction,
                    # expiry_date fields. OCR may insert spaces within values.
                    self._extract_ru_fips_expiry(
                        text, page, doc_id, doc_title, metainfo, expiry_map
                    )

        logger.info(
            "patent_expiry_deterministic inn=%s extracted=%d patents with expiry dates",
            self.inn, len(expiry_map),
        )
        return expiry_map

    def _extract_ru_fips_expiry(
        self, text: str, page, doc_id: str, doc_title: str,
        metainfo: dict, expiry_map: Dict[str, Dict[str, Any]],
    ) -> None:
        """Parse RU FIPS registry chunks for patent expiry dates.

        FIPS chunks embed serialized JSON with items like:
            "doc_id":"RU2803237C2_20230911" ... "jurisdiction":"RU" ... "expiry_date":"2039-10-25"
        OCR may insert spaces within numbers/dates.
        """
        # Find all doc_id positions, then search forwards for expiry_date + jurisdiction.
        for id_m in self._RU_FIPS_DOC_ID_RE.finditer(text):
            raw_patent = id_m.group(1).replace(" ", "")  # e.g. RU2803237C2
            # Search within next 3000 chars for expiry_date
            window = text[id_m.end():id_m.end() + 3000]
            exp_m = self._RU_FIPS_EXPIRY_DATE_RE.search(window)
            if not exp_m:
                continue
            expiry_raw = exp_m.group(1).replace(" ", "").replace("\n", "")
            # Validate date format YYYY-MM-DD
            if not re.match(r"\d{4}-\d{2}-\d{2}$", expiry_raw):
                continue

            # Jurisdiction (defaults to RU for FIPS)
            jur_m = self._RU_FIPS_JURISDICTION_RE.search(window)
            country = jur_m.group(1) if jur_m else "RU"

            norm = self._normalise_patent_number(raw_patent)
            if norm not in expiry_map:
                ev = _build_evidence(doc_id, page, text, doc_title,
                                     source_url=metainfo.get("source_url"),
                                     doc_kind="ru_patent_fips")
                self._evidence_registry[ev.evidence_id] = ev
                expiry_map[norm] = {"country": country, "expiry": expiry_raw, "evidence": ev}

    @staticmethod
    def _normalise_patent_number(raw: str) -> str:
        """Strip country prefix and trailing kind codes to get base number for matching.
        EP1234567B1 → 1234567, US12345678 → 12345678, WO2020123456A1 → 2020123456
        """
        s = raw.strip().upper()
        # Remove country prefix (2 letters)
        if len(s) > 2 and s[:2].isalpha():
            s = s[2:]
        # Remove trailing kind code (A1, B1, B2, C1 etc.)
        s = re.sub(r"[A-Z]\d?$", "", s)
        return s

    # ── Block generators ─────────────────────────────────────────────────────

    def _generate_passport(self, unknowns: List[DossierUnknown]) -> DossierPassport:
        """Stage A+B+C+D for passport block. S6: includes PubChem chemistry fields."""
        # S6-T2: Authority-tiering — Tier-1 sources first, pubchem for chemistry
        # WSx.3: Added eaeu_registration/eaeu_document so EAEU docs contribute to
        # registered_where even when they are not retrieved in _generate_registrations.
        passport_doc_kinds = [
            "label", "us_fda", "epar", "smpc", "grls_card", "grls",
            "ru_instruction", "pubchem", "drug_monograph",
            "eaeu_registration", "eaeu_document",
        ]
        question = (
            f"Extract drug passport fields for {self.inn}: "
            "INN, trade names (brand names), FDA approval date, FDA indication, "
            "registered jurisdictions (where is it approved: US/EU/RU/EAEU/UK), "
            "chemical formula, canonical SMILES string, InChIKey, molecular weight in g/mol, "
            "drug class (ATC/pharmacological class), mechanism of action, "
            "MAH/marketing authorization holders, route of administration, dosage forms, key dosing regimens."
        )
        retrieved = self._retrieve(question, passport_doc_kinds, top_k=40)
        if not retrieved:
            self._add_unknown(
                unknowns, "passport.*", "NO_DOCUMENT_IN_CORPUS",
                f"No passport-relevant documents (label/EPAR/GRLS) indexed for {self.inn}.",
                "Run sources:attach to fetch FDA label, EMA EPAR, and GRLS card."
            )
            return DossierPassport(inn=self.inn)

        candidates_map = self._candidates_map(retrieved)
        context = self._context_str(retrieved)
        alias_map, candidates_str = self._build_alias_map(candidates_map)

        result = self._call_llm(
            _PASSPORT_INSTRUCTION, context, question, candidates_str, _PassportExtractLLM
        )

        if result is None:
            self._add_unknown(
                unknowns, "passport.*", "EXTRACTION_FAILED",
                "LLM failed to extract passport fields.",
            )
            return DossierPassport(inn=self.inn)

        am = alias_map
        _san = self._sanitize_evidenced_value  # shorthand
        passport = DossierPassport(
            inn=result.inn or self.inn,
            trade_names=_ev_list(result.trade_names, am),
            fda_approval_date=_san(_ev_to_evidenced_value(result.fda_approval_date, am), "passport.fda_approval_date", unknowns),
            fda_indication=_san(_ev_to_evidenced_value(result.fda_indication, am), "passport.fda_indication", unknowns),
            registered_where=_ev_list(result.registered_where, am),
            chemical_formula=_san(_ev_to_evidenced_value(result.chemical_formula, am), "passport.chemical_formula", unknowns),
            # S6-T4: PubChem chemistry block
            smiles=_san(_ev_to_evidenced_value(result.smiles, am), "passport.smiles", unknowns),
            inchi_key=_san(_ev_to_evidenced_value(result.inchi_key, am), "passport.inchi_key", unknowns),
            molecular_weight=_san(_ev_to_evidenced_value(result.molecular_weight, am), "passport.molecular_weight", unknowns),
            drug_class=_san(_ev_to_evidenced_value(result.drug_class, am), "passport.drug_class", unknowns),
            mechanism_of_action=_san(_ev_to_evidenced_value(result.mechanism_of_action, am), "passport.mechanism_of_action", unknowns),
            mah_holders=_ev_list(result.mah_holders, am),
            route_of_administration=_san(_ev_to_evidenced_value(result.route_of_administration, am), "passport.route_of_administration", unknowns),
            dosage_forms=_ev_list(result.dosage_forms, am),
            key_dosages=_ev_list(result.key_dosages, am),
        )

        # Sprint 15 P0.1: Deterministic fallback for fda_approval_date
        if not passport.fda_approval_date or not passport.fda_approval_date.value:
            det_date = self._extract_fda_approval_date_deterministic()
            if det_date:
                passport.fda_approval_date = det_date
                logger.info("passport.fda_approval_date patched by deterministic extractor: %s", det_date.value)

        # Sprint 18: Deterministic PubChem chemistry patch.
        # LLM retrieval (top_k=40) often prioritises label/GRLS/CTGov chunks over the
        # single pubchem chunk, leaving formula/SMILES/InChIKey/MW empty.
        # This extractor reads pubchem chunks directly and patches any missing fields.
        needs_chemistry = not any([
            passport.chemical_formula and passport.chemical_formula.value,
            passport.smiles and passport.smiles.value,
            passport.inchi_key and passport.inchi_key.value,
            passport.molecular_weight and passport.molecular_weight.value,
        ])
        if needs_chemistry:
            chem = self._extract_pubchem_chemistry_deterministic()
            if chem:
                patched = []
                if not (passport.chemical_formula and passport.chemical_formula.value) and "chemical_formula" in chem:
                    passport.chemical_formula = chem["chemical_formula"]
                    patched.append("chemical_formula")
                if not (passport.smiles and passport.smiles.value) and "smiles" in chem:
                    passport.smiles = chem["smiles"]
                    patched.append("smiles")
                if not (passport.inchi_key and passport.inchi_key.value) and "inchi_key" in chem:
                    passport.inchi_key = chem["inchi_key"]
                    patched.append("inchi_key")
                if not (passport.molecular_weight and passport.molecular_weight.value) and "molecular_weight" in chem:
                    passport.molecular_weight = chem["molecular_weight"]
                    patched.append("molecular_weight")
                if patched:
                    logger.info("passport chemistry patched by pubchem_deterministic: fields=%s", patched)

        # S7: Validate — ALL mandatory fields without evidence → unknowns
        # Expanded from 4 to cover all 9 scalar fields + trade_names.
        mandatory_fields = [
            ("passport.fda_approval_date", passport.fda_approval_date,
             "Ensure FDA label or Drugs@FDA is indexed in corpus."),
            ("passport.fda_indication", passport.fda_indication,
             "Ensure FDA label is indexed in corpus."),
            ("passport.drug_class", passport.drug_class,
             "Ensure FDA label or EMA EPAR/SmPC is indexed."),
            ("passport.mechanism_of_action", passport.mechanism_of_action,
             "Ensure FDA label, EMA SmPC, or drug_monograph is indexed."),
            ("passport.chemical_formula", passport.chemical_formula,
             "Ensure PubChem or drug_monograph is indexed."),
            ("passport.route_of_administration", passport.route_of_administration,
             "Ensure FDA label or SmPC is indexed."),
            ("passport.smiles", passport.smiles,
             "Ensure PubChem compound data (doc_kind=pubchem) is attached."),
            ("passport.inchi_key", passport.inchi_key,
             "Ensure PubChem compound data (doc_kind=pubchem) is attached."),
            ("passport.molecular_weight", passport.molecular_weight,
             "Ensure PubChem compound data (doc_kind=pubchem) is attached."),
        ]
        for field_path, ev_val, next_action in mandatory_fields:
            if ev_val is None or not ev_val.evidence_refs:
                self._add_unknown(
                    unknowns, field_path, "NO_EVIDENCE_IN_CORPUS",
                    f"Field {field_path} could not be extracted with evidence from available documents.",
                    next_action,
                )

        if not passport.trade_names:
            self._add_unknown(
                unknowns, "passport.trade_names", "NO_EVIDENCE_IN_CORPUS",
                f"No trade names found in corpus for {self.inn}.",
                "Ensure FDA label or EMA SmPC is indexed (trade_names = brand names).",
            )

        # S6-T4: PubChem chemistry block — detect biologics vs small molecules
        _drug_class_lower = (passport.drug_class.value if passport.drug_class and passport.drug_class.value else "").lower()
        _biologic_kws = {"antibody", "bispecific", "monoclonal", "biologic", "biosimilar",
                         "fusion protein", "peptide", "conjugate", "adc", "recombinant",
                         "immunoglobulin", "nanobody", "fab fragment"}
        _is_biologic = any(kw in _drug_class_lower for kw in _biologic_kws)

        if not any([passport.smiles, passport.inchi_key, passport.chemical_formula]):
            if _is_biologic:
                self._add_unknown(
                    unknowns, "passport.chemistry", "NOT_APPLICABLE_BIOLOGIC",
                    f"{self.inn} is a biologic ({passport.drug_class.value if passport.drug_class else 'unknown'}). "
                    "SMILES/InChIKey/molecular formula are not applicable for large-molecule biologics.",
                    "No action needed — chemistry fields excluded from passport score for biologics."
                )
            else:
                self._add_unknown(
                    unknowns, "passport.chemistry", "NO_DOCUMENT_IN_CORPUS",
                    f"No chemistry data (SMILES/InChIKey/formula) for {self.inn}. "
                    "PubChem document not in corpus.",
                    "Ensure PubChem compound data (doc_kind=pubchem) is attached to corpus."
                )

        return passport

    def _generate_registrations(self, unknowns: List[DossierUnknown]) -> List[DossierRegistration]:
        """Generate registration records for RU, EU, US."""
        regions = [
            ("RU", ["grls_card", "grls", "ru_instruction"],
             "What are the GRLS registration numbers, trade names, MAH, drug forms, and status for this drug in Russia?"),
            ("EU", ["epar", "smpc", "assessment_report", "pil"],
             "What are the EMA marketing authorization numbers, MAH, authorized forms, and status for this drug in the EU?"),
            ("US", ["label", "us_fda", "approval_letter", "anda_package"],
             "What are the FDA NDA/BLA numbers, applicant names, drug forms, and approval dates for this drug in the US?"),
            ("EAEU", ["eaeu_registration", "eaeu_document"],
             "What are the EAEU registration numbers, trade names, MAH, drug forms, validity dates, and status for this drug in EAEU member states?"),
        ]

        registrations: List[DossierRegistration] = []

        for region, doc_kinds, question in regions:
            retrieved = self._retrieve(question, doc_kinds, top_k=30)
            if not retrieved:
                self._add_unknown(
                    unknowns, f"registrations[{region}].*", "NO_DOCUMENT_IN_CORPUS",
                    f"No {region} registration documents indexed for {self.inn}.",
                    f"Attach {region} regulatory documents to corpus."
                )
                continue

            candidates_map = self._candidates_map(retrieved)
            context = self._context_str(retrieved)
            alias_map, candidates_str = self._build_alias_map(candidates_map)

            result = self._call_llm(
                _REGISTRATIONS_INSTRUCTION, context,
                f"{question} (region: {region})", candidates_str, _RegistrationsExtractLLM
            )

            if result is None or not result.registrations:
                self._add_unknown(
                    unknowns, f"registrations[{region}].*", "NO_EVIDENCE_IN_CORPUS",
                    f"No {region} registration data extractable from indexed documents.",
                )
                continue

            for reg_llm in result.registrations:
                am = alias_map
                ev_refs: List[str] = []
                status = _ev_to_evidenced_value(reg_llm.status, am)
                if status and status.evidence_refs:
                    ev_refs.extend(status.evidence_refs)
                mah = _ev_to_evidenced_value(reg_llm.mah, am)
                if mah and mah.evidence_refs:
                    ev_refs.extend(mah.evidence_refs)
                identifiers = _ev_list(reg_llm.identifiers, am)
                for ev in identifiers:
                    ev_refs.extend(ev.evidence_refs)

                # S5-P0-D: Drop empty rows — a registration with no status,
                # no MAH, and no identifiers contains nothing useful and would
                # appear as an empty row in the dossier table.
                has_content = (
                    (status and status.value not in (None, "", []))
                    or (mah and mah.value not in (None, "", []))
                    or len(identifiers) > 0
                )
                if not has_content:
                    logger.debug(
                        "s5_reg_drop_empty region=%s inn=%s",
                        reg_llm.region or region, self.inn,
                    )
                    continue

                registration = DossierRegistration(
                    region=reg_llm.region or region,
                    status=status,
                    forms_strengths=_ev_list(reg_llm.forms_strengths, am),
                    mah=mah,
                    identifiers=identifiers,
                    evidence_refs=list(set(ev_refs)),
                )
                registrations.append(registration)

        # Sprint 16.2: Deterministic fallback for RU registration status.
        # If the RU DossierRegistration has no status (LLM failed to extract it from
        # grls_card chunks — e.g. because the accordion was collapsed), try the
        # regex-based extractor that scans grls_card chunks directly.
        det_ru = self._extract_ru_reg_status_deterministic()
        if det_ru is not None:
            combined_status, det_ev = det_ru
            ev_id = det_ev.evidence_id if det_ev else None
            ru_regs = [r for r in registrations if r.region.upper() == "RU"]
            if ru_regs:
                for ru_reg in ru_regs:
                    if not ru_reg.status or not ru_reg.status.value:
                        ru_reg.status = EvidencedValue(
                            value=combined_status,
                            evidence_refs=[ev_id] if ev_id else [],
                        )
                        logger.info(
                            "ru_reg_status patched by deterministic extractor: %r ev=%s",
                            combined_status, ev_id,
                        )
            else:
                # No RU registration from LLM at all — create a minimal stub
                # so the deterministic status isn't silently lost.
                pass  # Don't create a stub without reg_number — would be misleading

        # WSx.3+4: EAEU truth contract.
        # If EAEU region was populated by the LLM from eaeu_registration docs, skip unknowns.
        # Otherwise emit a precise state code — not a generic "unavailable" message.
        has_eaeu_reg = any(r.region.upper() == "EAEU" for r in registrations)
        if not has_eaeu_reg:
            # Check whether we have an EAEU doc in corpus (seeded but LLM found nothing).
            # Scan documents_dir for any indexed doc with eaeu_registration/eaeu_document kind.
            eaeu_doc_kinds = {"eaeu_registration", "eaeu_document"}
            has_eaeu_doc = False
            if hasattr(self, "documents_dir") and self.documents_dir is not None:
                for doc_path in self.documents_dir.glob("*.json"):
                    try:
                        import json as _json
                        with open(doc_path) as _f:
                            _meta = _json.load(_f)
                        _kind = (_meta.get("metainfo", {}) or {}).get("doc_kind", "").lower()
                        if _kind in eaeu_doc_kinds:
                            has_eaeu_doc = True
                            break
                    except Exception:
                        pass

            if has_eaeu_doc:
                # EAEU doc was indexed but LLM extracted nothing from it
                self._add_unknown(
                    unknowns, "registrations[EAEU].*", "EAEU_NO_EVIDENCE_IN_CORPUS",
                    f"EAEU registration document was indexed for {self.inn} but "
                    "no registration records could be extracted from it.",
                    "Review eaeu_registration document content; INN may not match.",
                )
            else:
                # No EAEU doc in corpus — SPD returned 0 results (drug not registered in EAEU).
                # NOTE: If SPD portal was down during seeding, use EAEU_PORTAL_UNAVAILABLE instead.
                # This code fires only when SPD lookup completed successfully with 0 results.
                self._add_unknown(
                    unknowns, "registrations[EAEU].*", "EAEU_NOT_FOUND_AFTER_LOOKUP",
                    f"No EAEU registration records found for {self.inn}. "
                    "SPD lookup completed but returned 0 results — "
                    "the drug is likely not directly registered in the EAEU union register.",
                    "Cross-check EAEU SPD manually; if portal was down during seeding, re-seed.",
                )

        # WSx.4: Detect РГ-RU mutual-recognition marker in GRLS reg number.
        # This is NOT a direct EAEU registration — it is an EAEU-format framing on the
        # RU registration. Emit as informational unknown, not as EAEU confirmation.
        eaeu_marker_ev = self._detect_ru_eaeu_marker()
        if eaeu_marker_ev and not has_eaeu_reg:
            self._add_unknown(
                unknowns, "registrations[EAEU].mutual_recognition", "EAEU_RELATED_RU_MARKER",
                f"RU registration for {self.inn} contains an EAEU mutual-recognition marker "
                "(ЛП-...-(РГ-RU)). This indicates the RU registration was processed via "
                "the EAEU mutual-recognition route, but does NOT confirm a standalone "
                "direct EAEU union-register entry.",
                "Check GRLS reg number suffix; verify EAEU SPD for direct registration separately.",
            )

        # S5-P0-D + S6: Dedup by region — keep only the richest entry per region.
        # LLM may produce duplicate rows for the same region (e.g. 2 × "RU").
        # Sort descending by (identifiers count, evidence count) so the most
        # evidence-rich record wins; ties broken by original order.
        seen_regions: dict = {}
        deduped: List[DossierRegistration] = []
        for reg in sorted(
            registrations,
            key=lambda r: (len(r.identifiers), len(r.evidence_refs)),
            reverse=True,
        ):
            r_key = reg.region.upper().strip()
            if r_key not in seen_regions:
                seen_regions[r_key] = True
                deduped.append(reg)
            else:
                logger.debug(
                    "s5_reg_dedup_drop region=%s inn=%s (kept higher-evidence entry)",
                    r_key, self.inn,
                )

        return deduped

    # ── Sprint 13 WS1: Per-study helpers ──────────────────────────────────────

    def _retrieve_clinical_evidence_for_study(
        self, nct_id: str, top_k: int = 30,
    ) -> List[Dict[str, Any]]:
        """Sprint 13 WS1: Retrieve evidence scoped to a single NCT study.

        Uses the NCT ID as the primary query anchor so that FAISS retrieval
        returns chunks most relevant to THIS specific study, not the drug broadly.
        Primary doc_kinds: ctgov_protocol/ctgov_results/ctgov/ctgov_documents.
        Secondary (enrichment): publication/scientific_pmc/scientific_pdf.
        """
        primary_kinds = list(_CTGOV_DOC_KINDS)
        enrichment_kinds = list(_ENRICHMENT_ONLY_DOC_KINDS)
        all_kinds = primary_kinds + enrichment_kinds

        question = (
            f"Clinical study {nct_id} for {self.inn}: "
            f"extract study title, phase, type, enrollment, countries, "
            f"comparator, dosing, primary endpoint, results, status, conclusion."
        )
        return self._retrieve(question, all_kinds, top_k=top_k)

    def _assemble_single_clinical_study(
        self,
        nct_id: str,
        retrieved: List[Dict[str, Any]],
        unknowns: List[DossierUnknown],
    ) -> Optional[DossierClinicalStudy]:
        """Sprint 13 WS1: Assemble one DossierClinicalStudy from study-scoped evidence.

        Filters retrieved chunks to prefer those mentioning the specific NCT ID,
        then calls LLM to extract exactly ONE study card.
        Returns None if extraction fails or card lacks evidence.
        """
        # Partition: chunks mentioning this NCT vs general clinical
        nct_upper = nct_id.upper()
        scoped = [r for r in retrieved if nct_upper in (r.get("text", "") + r.get("doc_title", "")).upper()]
        other = [r for r in retrieved if r not in scoped]

        # Build evidence from scoped first, then pad with general if needed
        combined = scoped + other
        if not combined:
            return None

        candidates_map = self._candidates_map(combined[:30])
        context = self._context_str(combined[:20])
        alias_map, candidates_str = self._build_alias_map(candidates_map)

        # Single-study extraction prompt
        instruction = (
            f"You are a pharmaceutical dossier extraction system.\n"
            f"Extract EXACTLY ONE clinical study card for study {nct_id} ({self.inn}).\n\n"
            f"IMPORTANT:\n"
            f"- study_id MUST be set to \"{nct_id}\".\n"
            f"- Extract fields ONLY from evidence that relates to this specific study.\n"
            f"- Do NOT mix data from other studies into this card.\n"
            f"- If a field cannot be found for this study specifically, set it to null.\n\n"
            f"Extract these fields:\n"
            f"- title: official or brief study title\n"
            f"- study_id: must be \"{nct_id}\"\n"
            f"- phase: Phase 1/2/3/4\n"
            f"- study_type: interventional/observational\n"
            f"- n_enrolled: number of participants\n"
            f"- countries: list of countries from locations\n"
            f"- comparator: REQUIRED if present — placebo, specific drug name, or 'none (single-arm)'. "
            f"Look for arm labels and intervention types (ACTIVE_COMPARATOR, PLACEBO_COMPARATOR).\n"
            f"- regimen_dosing: REQUIRED if present — dose (mg), frequency, route. "
            f"Look in arm/intervention descriptions.\n"
            f"- efficacy_keypoints: primary endpoint results, ORR, PFS, OS, p-value, CI\n"
            f"- conclusion: REQUIRED if any results exist — verbatim conclusion or synthesized 1-2 sentence "
            f"summary from numeric results. Set null only if truly no clinical content beyond the NCT ID.\n"
            f"- status: Completed/Recruiting/Terminated/etc.\n\n"
            f"CRITICAL: Use ONLY evidence aliases (E1, E2, ...) from the candidates list.\n"
            f"Every non-null field MUST include at least one evidence alias."
        )
        question = f"Extract clinical study card for {nct_id} ({self.inn})"

        result = self._call_llm(
            instruction, context, question, candidates_str, _ClinicalStudiesExtractLLM
        )

        if result is None or not result.studies:
            return None

        # Take only the first study (we asked for exactly one)
        study_llm = result.studies[0]
        am = alias_map
        ev_refs: List[str] = []

        def _collect(ev_val: Optional[EvidencedValue]) -> Optional[EvidencedValue]:
            if ev_val and ev_val.evidence_refs:
                ev_refs.extend(ev_val.evidence_refs)
            return ev_val

        title = _collect(_ev_to_evidenced_value(study_llm.title, am))
        study_id_ev = _collect(_ev_to_evidenced_value(study_llm.study_id, am))
        phase = _collect(_ev_to_evidenced_value(study_llm.phase, am))
        study_type = _collect(_ev_to_evidenced_value(study_llm.study_type, am))
        n_enrolled = _collect(_ev_to_evidenced_value(study_llm.n_enrolled, am))
        comparator = _collect(_ev_to_evidenced_value(study_llm.comparator, am))
        regimen = _collect(_ev_to_evidenced_value(study_llm.regimen_dosing, am))
        conclusion = _collect(_ev_to_evidenced_value(study_llm.conclusion, am))
        status = _collect(_ev_to_evidenced_value(study_llm.status, am))
        countries = _ev_list(study_llm.countries, am)
        efficacy = _ev_list(study_llm.efficacy_keypoints, am)
        for ev in countries + efficacy:
            ev_refs.extend(ev.evidence_refs)

        if not ev_refs:
            return None

        # Force study_id to the known NCT ID (deterministic, not LLM-dependent)
        if study_id_ev is None or not study_id_ev.value:
            # Use the known NCT ID with whatever evidence refs we have
            study_id_ev = EvidencedValue(value=nct_id, evidence_refs=ev_refs[:1])

        study = DossierClinicalStudy(
            title=title, study_id=study_id_ev, phase=phase, study_type=study_type,
            n_enrolled=n_enrolled, countries=countries, comparator=comparator,
            regimen_dosing=regimen, efficacy_keypoints=efficacy,
            conclusion=conclusion, status=status,
            evidence_refs=list(set(ev_refs)),
        )
        return study

    @staticmethod
    def _fetch_ctgov_study(nct_id: str) -> Optional[Dict[str, Any]]:
        """Fetch study metadata from ClinicalTrials.gov v2 API.

        Returns parsed JSON or None on failure. Timeout: 10s.
        """
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?fields=protocolSection"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug("ctgov_api_fetch nct=%s error=%s", nct_id, exc)
            return None

    def _enrich_clinical_from_ctgov_api(
        self, studies: List[DossierClinicalStudy]
    ) -> None:
        """Deterministic post-LLM enrichment: fill null phase/status/enrollment/countries
        from ClinicalTrials.gov v2 API.

        Only patches fields that are null — never overwrites LLM-extracted values.
        Evidence ref: synthetic "ctgov_api:{NCT}" entry.
        """
        for study in studies:
            nct_id = study.study_id.value if study.study_id else None
            if not nct_id or not nct_id.startswith("NCT"):
                continue

            needs_phase = not (study.phase and study.phase.value)
            needs_status = not (study.status and study.status.value)
            needs_enrolled = not (study.n_enrolled and study.n_enrolled.value)
            needs_countries = not study.countries
            needs_title = not (study.title and study.title.value)
            needs_study_type = not (study.study_type and study.study_type.value)
            # Flags are always (re)computed from API regardless of prior value
            needs_flags = True

            if not any([needs_phase, needs_status, needs_enrolled, needs_countries,
                        needs_title, needs_study_type, needs_flags]):
                continue

            data = self._fetch_ctgov_study(nct_id)
            if not data:
                continue

            protocol = data.get("protocolSection", {})
            design = protocol.get("designModule", {})
            status_mod = protocol.get("statusModule", {})
            ident = protocol.get("identificationModule", {})
            contacts = protocol.get("contactsLocationsModule", {})
            arms_mod = protocol.get("armsInterventionsModule", {})

            ev_id = f"ctgov_api_{nct_id}"
            if ev_id not in self._evidence_registry:
                ev = DossierEvidence(
                    evidence_id=ev_id,
                    doc_id=f"ctgov_api:{nct_id}",
                    page=None,
                    snippet=f"ClinicalTrials.gov API v2 metadata for {nct_id}",
                    title=f"CTGov API: {nct_id}",
                    source_url=f"https://clinicaltrials.gov/study/{nct_id}",
                    doc_kind="ctgov_api",
                )
                self._evidence_registry[ev_id] = ev

            refs = [ev_id]

            if needs_phase:
                phases = design.get("phases") or []
                if phases:
                    phase_str = phases[-1].replace("PHASE", "Phase ").replace("_", " ").strip()
                    study.phase = EvidencedValue(value=phase_str, evidence_refs=refs)

            if needs_status:
                overall_status = status_mod.get("overallStatus", "")
                if overall_status:
                    study.status = EvidencedValue(value=overall_status, evidence_refs=refs)

            if needs_enrolled:
                enroll_info = design.get("enrollmentInfo", {})
                enroll_count = enroll_info.get("count")
                if enroll_count:
                    study.n_enrolled = EvidencedValue(value=str(enroll_count), evidence_refs=refs)

            if needs_countries:
                locations = contacts.get("locations", [])
                country_set = {loc.get("country", "") for loc in locations if loc.get("country")}
                if country_set:
                    study.countries = [
                        EvidencedValue(value=c, evidence_refs=refs) for c in sorted(country_set)
                    ]

            if needs_title:
                title_str = ident.get("officialTitle") or ident.get("briefTitle", "")
                if title_str:
                    study.title = EvidencedValue(value=title_str, evidence_refs=refs)

            # WS3.2: study_type from CTGov designModule.studyType
            if needs_study_type:
                raw_type = design.get("studyType", "")
                if raw_type:
                    # Normalize: INTERVENTIONAL → Interventional, OBSERVATIONAL → Observational
                    type_str = raw_type.capitalize()
                    # Add allocation detail if available
                    alloc = design.get("designInfo", {}).get("allocation", "")
                    if alloc and alloc not in ("NA", "N_A"):
                        type_str = f"{type_str} ({alloc.replace('_', ' ').title()})"
                    study.study_type = EvidencedValue(value=type_str, evidence_refs=refs)

            # WS3.7: Deterministic screening signal flags from CTGov API
            # is_ongoing: actively recruiting or not yet completed
            overall_status = status_mod.get("overallStatus", "")
            _ongoing_statuses = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION",
                                  "NOT_YET_RECRUITING", "AVAILABLE"}
            _completed_statuses = {"COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"}
            if overall_status:
                study.is_ongoing = overall_status.upper() in _ongoing_statuses

            # is_post_reg: Phase 4 or EXPANDED_ACCESS or observational post-approval
            phases = design.get("phases") or []
            study_type_raw = design.get("studyType", "")
            primary_purpose = design.get("designInfo", {}).get("primaryPurpose", "")
            study.is_post_reg = (
                "PHASE4" in [p.upper().replace(" ", "").replace("_", "") for p in phases]
                or study_type_raw.upper() == "EXPANDED_ACCESS"
                or primary_purpose.upper() in ("POST_MARKETING", "PREVENTION")
            )

            # is_combination_therapy: any arm group label / intervention name mentions ≥2 drugs
            arm_groups = arms_mod.get("armGroups", [])
            interventions = arms_mod.get("interventions", [])
            combo_keywords = [" and ", " combined with ", " plus ", " + ", " with ", " in combination"]
            inn_lower = self.inn.lower()
            combo_signals = []
            for arm in arm_groups:
                label = (arm.get("label", "") + " " + arm.get("description", "")).lower()
                combo_signals.append(any(kw in label for kw in combo_keywords) and inn_lower in label)
            for itv in interventions:
                names = " ".join(itv.get("otherNames", [])).lower()
                itv_name = itv.get("name", "").lower()
                combo_signals.append(
                    any(kw in (itv_name + " " + names) for kw in combo_keywords)
                )
            study.is_combination_therapy = any(combo_signals) if combo_signals else None

            # has_ru_presence: Russia listed in locations
            locations = contacts.get("locations", [])
            ru_countries = {"Russia", "Russian Federation"}
            study.has_ru_presence = any(
                loc.get("country", "") in ru_countries for loc in locations
            ) if locations else None

            logger.info(
                "ctgov_api_enrichment nct=%s patched: phase=%s status=%s enrolled=%s "
                "countries=%d title=%s study_type=%s ongoing=%s post_reg=%s combo=%s ru=%s",
                nct_id,
                "yes" if needs_phase and study.phase else "no",
                "yes" if needs_status and study.status else "no",
                "yes" if needs_enrolled and study.n_enrolled else "no",
                len(study.countries) if study.countries else 0,
                "yes" if needs_title and study.title else "no",
                "yes" if needs_study_type and study.study_type else "no",
                study.is_ongoing, study.is_post_reg, study.is_combination_therapy, study.has_ru_presence,
            )

    def _generate_clinical_studies(self, unknowns: List[DossierUnknown]) -> List[DossierClinicalStudy]:
        """Generate structured clinical study cards.

        Sprint 13 WS1: Per-study assembly pipeline —
        1. Pre-scan corpus for NCT IDs (deterministic candidate list)
        2. For each NCT ID: scoped retrieval → scoped LLM extraction → one card
        3. Dedup by NCT ID
        4. Only cards with study_id pass (enforced by construction)
        5. CTGov API enrichment: fill null phase/status/enrollment from CTGov v2

        This replaces the Sprint 12 approach of one global LLM call + post-hoc filtering.
        Each study gets its own evidence bucket, preventing cross-study contamination.
        """
        # Step 1: Pre-scan corpus for NCT IDs
        corpus_nct_ids = _extract_nct_ids_from_corpus(self.documents_dir)
        logger.info(
            "clinical_nct_prescan found %d NCT IDs in corpus: %s",
            len(corpus_nct_ids), sorted(corpus_nct_ids)[:20],
        )

        if not corpus_nct_ids:
            # Fallback: try a broad retrieval to find any clinical study mentions
            clinical_doc_kinds = list(_CTGOV_DOC_KINDS) + list(_ENRICHMENT_ONLY_DOC_KINDS)
            retrieved = self._retrieve(
                f"Clinical studies for {self.inn}: NCT IDs, trial registry identifiers",
                clinical_doc_kinds, top_k=30,
            )
            # Scan retrieved text for NCT IDs
            for item in retrieved:
                text = item.get("text", "")
                for m in _NCT_PATTERN.finditer(text):
                    corpus_nct_ids.add(m.group(0).upper())
            if corpus_nct_ids:
                logger.info(
                    "clinical_nct_fallback_scan found %d NCT IDs from retrieval: %s",
                    len(corpus_nct_ids), sorted(corpus_nct_ids)[:10],
                )

        if not corpus_nct_ids:
            self._add_unknown(
                unknowns, "clinical_studies[*]", "NO_DOCUMENT_IN_CORPUS",
                f"No clinical studies with NCT IDs found in corpus for {self.inn}.",
                "Ensure CTGov protocol/results pages are attached to corpus."
            )
            return []

        # Step 2: Per-study retrieval and assembly
        sorted_nct_ids = sorted(corpus_nct_ids)  # deterministic order
        studies: List[DossierClinicalStudy] = []
        seen_nct: set = set()
        failed_nct: List[str] = []

        for nct_id in sorted_nct_ids:
            nct_upper = nct_id.upper()
            if nct_upper in seen_nct:
                continue

            retrieved = self._retrieve_clinical_evidence_for_study(nct_id, top_k=30)
            if not retrieved:
                failed_nct.append(nct_id)
                logger.info("clinical_per_study_no_evidence nct=%s inn=%s", nct_id, self.inn)
                continue

            study = self._assemble_single_clinical_study(nct_id, retrieved, unknowns)
            if study is None:
                failed_nct.append(nct_id)
                logger.info("clinical_per_study_assembly_failed nct=%s inn=%s", nct_id, self.inn)
                continue

            # Validate: must have study_id (enforced by construction, but verify)
            if not _is_valid_clinical_card(study):
                failed_nct.append(nct_id)
                continue

            seen_nct.add(nct_upper)
            studies.append(study)
            logger.info(
                "clinical_per_study_assembled nct=%s inn=%s phase=%s status=%s",
                nct_id, self.inn,
                study.phase.value if study.phase else "?",
                study.status.value if study.status else "?",
            )

        # Step 3: Log summary
        logger.info(
            "clinical_per_study_summary inn=%s candidates=%d assembled=%d failed=%d",
            self.inn, len(sorted_nct_ids), len(studies), len(failed_nct),
        )

        if failed_nct:
            self._add_unknown(
                unknowns, "clinical_studies[*].per_study_failed",
                "NO_EVIDENCE_IN_CORPUS",
                f"{len(failed_nct)} NCT ID(s) found in corpus but could not assemble study cards: "
                f"{', '.join(failed_nct[:5])}{'...' if len(failed_nct) > 5 else ''}. "
                "Likely insufficient scoped evidence in retrieved chunks.",
                "Ensure per-study CTGov protocol/results pages are attached.",
            )

        if not studies:
            self._add_unknown(
                unknowns, "clinical_studies[*]", "NO_EVIDENCE_IN_CORPUS",
                f"Found {len(corpus_nct_ids)} NCT IDs but could not assemble any study cards.",
            )

        # Step 4: CTGov API enrichment — fill null phase/status/enrollment/countries
        if studies:
            self._enrich_clinical_from_ctgov_api(studies)

        return studies

    def _generate_patent_families(self, unknowns: List[DossierUnknown]) -> List[DossierPatentFamily]:
        """Generate patent family records."""
        if not self._has_patent_corpus():
            self._add_unknown(
                unknowns, "patent_families[*]", "PATENT_DISCOVERY_GAP",
                f"No patent documents (patent_family/patent_pdf/ops/ru_patent_fips) indexed for {self.inn}. "
                "gpt-researcher did not discover patent URLs during research phase.",
                "Strengthen Q19/Q21/Q22 templates in router-map.json (done Sprint 4). "
                "Re-run research:run to discover patent URLs. "
                "Alternatively use EPO OPS direct API attach (follow-up item)."
            )
            return []

        patent_doc_kinds = ["patent_family", "ops", "patent_pdf", "ru_patent_pdf", "ru_patent_fips", "patent", "patent_family_summary", "patent_legal_events", "patent_expiry_us", "patent_discovery_us"]
        # Sprint 17 WS6/WS7/WS8: enriched patent extraction query — technical focus + process relevance + legal status
        question = (
            f"For {self.inn}: identify all patent families. "
            "For each: representative publication number (EP/WO/US), priority date, assignee, "
            "what it protects (compound/formulation/method_of_use/synthesis), "
            "detailed technical focus (composition/formulation/process_manufacturing/method_of_use/"
            "combination/salt_polymorph/dosage_form_delivery/intermediate_synthesis), "
            "process/synthesis relevance (strong/moderate/weak/none), "
            "legal status (granted/pending/expired/revoked/lapsed/unknown), "
            "one-sentence summary, country coverage, "
            "expiry dates per country (only if explicitly stated)."
        )
        retrieved = self._retrieve(question, patent_doc_kinds, top_k=60)
        if not retrieved:
            self._add_unknown(
                unknowns, "patent_families[*]", "NO_EVIDENCE_IN_CORPUS",
                f"Patent documents are in corpus but no relevant passages found for {self.inn}.",
            )
            return []

        candidates_map = self._candidates_map(retrieved)
        context = self._context_str(retrieved)
        alias_map, candidates_str = self._build_alias_map(candidates_map)

        result = self._call_llm(
            _PATENTS_INSTRUCTION, context, question, candidates_str, _PatentFamiliesExtractLLM
        )

        if result is None or not result.families:
            self._add_unknown(
                unknowns, "patent_families[*]", "NO_EVIDENCE_IN_CORPUS",
                f"Could not extract patent family records for {self.inn} from patent documents.",
            )
            return []

        families: List[DossierPatentFamily] = []
        filtered_synthetic = 0
        filtered_minimal = 0
        am = alias_map
        for fam_llm in result.families:
            ev_refs: List[str] = []

            rep = _ev_to_evidenced_value(fam_llm.representative_pub, am)
            if rep and rep.evidence_refs:
                ev_refs.extend(rep.evidence_refs)
            priority = _ev_to_evidenced_value(fam_llm.priority_date, am)
            if priority and priority.evidence_refs:
                ev_refs.extend(priority.evidence_refs)
            what_blocks = _ev_to_evidenced_value(fam_llm.what_blocks, am)
            if what_blocks and what_blocks.evidence_refs:
                ev_refs.extend(what_blocks.evidence_refs)
            # Sprint 17 WS6: technical focus extraction
            technical_focus = _ev_to_evidenced_value(fam_llm.technical_focus, am)
            if technical_focus and technical_focus.evidence_refs:
                ev_refs.extend(technical_focus.evidence_refs)
            # Sprint 17 WS7: process/synthesis relevance
            process_relevance = _ev_to_evidenced_value(fam_llm.process_relevance, am)
            if process_relevance and process_relevance.evidence_refs:
                ev_refs.extend(process_relevance.evidence_refs)
            # Sprint 17 WS8: legal status snapshot
            legal_status_snapshot = _ev_to_evidenced_value(fam_llm.legal_status_snapshot, am)
            if legal_status_snapshot and legal_status_snapshot.evidence_refs:
                ev_refs.extend(legal_status_snapshot.evidence_refs)
            summary = _ev_to_evidenced_value(fam_llm.summary, am)
            if summary and summary.evidence_refs:
                ev_refs.extend(summary.evidence_refs)
            assignees = _ev_list(fam_llm.assignees, am)
            coverage = _ev_list(fam_llm.country_coverage, am)
            expiry = _ev_list(fam_llm.expiry_by_country, am)
            for ev in assignees + coverage + expiry:
                ev_refs.extend(ev.evidence_refs)

            # family_id: use representative pub number or generate
            family_id = fam_llm.family_id
            if not family_id:
                family_id = rep.value if (rep and rep.value) else f"FAMILY_{len(families)+1}"

            # Key docs: doc_ids of used evidence
            key_docs = list({self._evidence_registry[e].doc_id for e in ev_refs if e in self._evidence_registry})

            if not ev_refs:
                self._add_unknown(
                    unknowns, f"patent_families[{family_id}]", "NO_EVIDENCE_IN_CORPUS",
                    "Patent family extracted but could not link evidence_ids.",
                )
                continue

            fam = DossierPatentFamily(
                family_id=family_id,
                representative_pub=rep,
                priority_date=priority,
                assignees=assignees,
                what_blocks=what_blocks,
                technical_focus=technical_focus,       # Sprint 17 WS6
                process_relevance=process_relevance,   # Sprint 17 WS7
                legal_status_snapshot=legal_status_snapshot,  # Sprint 17 WS8
                summary=summary,
                country_coverage=coverage,
                expiry_by_country=expiry,
                key_docs=key_docs,
                evidence_refs=list(set(ev_refs)),
            )

            # Sprint 12 WS3: Reject synthetic empty families
            if _is_synthetic_patent_family(fam):
                filtered_synthetic += 1
                logger.info(
                    "patent_family_filtered_synthetic inn=%s family_id=%s — synthetic shell detected",
                    self.inn, family_id[:60],
                )
                continue

            # Sprint 12 WS3: Reject minimally-invalid families
            if not _is_minimally_valid_patent_family(fam):
                filtered_minimal += 1
                logger.info(
                    "patent_family_filtered_minimal inn=%s family_id=%s — lacks minimum viable content "
                    "(needs representative_pub/priority_date + assignees/summary/what_blocks)",
                    self.inn, family_id[:60],
                )
                continue

            # Expiry without data → warn in unknowns
            if not expiry:
                self._add_unknown(
                    unknowns, f"patent_families[{family_id}].expiry_by_country",
                    "LEGAL_STATUS_NOT_AVAILABLE",
                    f"Patent expiry dates not found in corpus for family {family_id}. "
                    "Legal status data requires EPO Register or OrangeBook integration.",
                    "Integrate EPO Register API (eporegister.go) or OrangeBook service."
                )

            families.append(fam)

        # Sprint 12 WS3: Log filtering summary
        if filtered_synthetic + filtered_minimal > 0:
            logger.info(
                "patent_family_filter inn=%s total_extracted=%d passed=%d "
                "filtered_synthetic=%d filtered_minimal=%d",
                self.inn, len(result.families), len(families),
                filtered_synthetic, filtered_minimal,
            )
            if filtered_synthetic > 0:
                self._add_unknown(
                    unknowns, "patent_families[*].discovery",
                    "PATENT_DISCOVERY_GAP",
                    f"{filtered_synthetic} patent family shell(s) were synthetic placeholders "
                    f"(e.g., '{self.inn} (EPO OPS: no families found)'). "
                    "These indicate empty discovery, not real patent families. "
                    "Filtered out per Sprint 12 semantic integrity rules.",
                    "For off-patent drugs, empty patent_families is the honest state. "
                    "For on-patent drugs, verify EPO OPS search or add manual patent sources."
                )
            if filtered_minimal > 0:
                self._add_unknown(
                    unknowns, "patent_families[*].minimal_validity",
                    "NO_EVIDENCE_IN_CORPUS",
                    f"{filtered_minimal} patent family(ies) lacked minimum viable content "
                    "(no representative_pub/priority_date or no assignees/summary/what_blocks). "
                    "Filtered out to prevent empty shells from appearing as usable IP data.",
                )

        # ── Sprint 17: Deterministic patent expiry patching ─────────────────
        # Scan patent_legal_events + patent_expiry_us chunks for explicit expiry
        # dates, then patch families that LLM left without expiry_by_country.
        det_expiry = self._extract_patent_expiry_deterministic()
        if det_expiry and families:
            patched_count = 0
            for fam in families:
                if fam.expiry_by_country:
                    continue  # LLM already provided expiry — skip

                # Try to match representative_pub against expiry map
                rep_val = fam.representative_pub.value if fam.representative_pub else None
                matched_entries: List[tuple] = []  # (country, expiry, evidence)

                if rep_val and isinstance(rep_val, str):
                    norm_rep = self._normalise_patent_number(rep_val)
                    for pat_norm, info in det_expiry.items():
                        if pat_norm == norm_rep or norm_rep in pat_norm or pat_norm in norm_rep:
                            matched_entries.append((info["country"], info["expiry"], info["evidence"]))

                # Also try matching family_id (might contain patent number)
                if not matched_entries:
                    fid_norm = self._normalise_patent_number(fam.family_id)
                    if fid_norm and len(fid_norm) >= 5:
                        for pat_norm, info in det_expiry.items():
                            if pat_norm == fid_norm or fid_norm in pat_norm or pat_norm in fid_norm:
                                matched_entries.append((info["country"], info["expiry"], info["evidence"]))

                # Also: if country_coverage lists countries that appear in expiry_map
                if not matched_entries and fam.country_coverage:
                    coverage_countries = set()
                    for cv in fam.country_coverage:
                        if cv.value and isinstance(cv.value, str):
                            coverage_countries.add(cv.value.upper().strip())
                    for pat_norm, info in det_expiry.items():
                        if info["country"] in coverage_countries:
                            matched_entries.append((info["country"], info["expiry"], info["evidence"]))

                if matched_entries:
                    # Deduplicate by country
                    seen_countries: set = set()
                    for country, expiry_date, ev in matched_entries:
                        if country in seen_countries:
                            continue
                        seen_countries.add(country)
                        fam.expiry_by_country.append(
                            EvidencedValue(
                                value=f"{country}: {expiry_date}",
                                evidence_refs=[ev.evidence_id],
                            )
                        )
                        fam.evidence_refs.append(ev.evidence_id)
                    patched_count += 1

                    # Remove the LEGAL_STATUS_NOT_AVAILABLE unknown we added earlier
                    unknowns[:] = [
                        u for u in unknowns
                        if not (u.field_path == f"patent_families[{fam.family_id}].expiry_by_country"
                                and u.reason_code == "LEGAL_STATUS_NOT_AVAILABLE")
                    ]

            if patched_count:
                logger.info(
                    "patent_expiry_deterministic_patch inn=%s patched=%d/%d families with expiry data",
                    self.inn, patched_count, len(families),
                )

        return families

    def _generate_synthesis_steps(self, unknowns: List[DossierUnknown]) -> List[DossierSynthesisStep]:
        """Generate synthesis steps from patent text, with EPAR/assessment_report fallback."""
        # S7-E1: Primary sources are patents; if no patent corpus, try EPAR as fallback
        _PATENT_DOC_KINDS = ["patent_pdf", "ru_patent_pdf", "patent", "drug_monograph"]
        _EPAR_FALLBACK_KINDS = ["epar", "assessment_report"]

        use_epar_fallback = False
        if not self._has_patent_corpus():
            use_epar_fallback = True
            logger.info("synthesis_epar_fallback inn=%s — no patent corpus, trying EPAR", self.inn)

        patent_doc_kinds = _EPAR_FALLBACK_KINDS if use_epar_fallback else _PATENT_DOC_KINDS
        question = (
            f"For {self.inn}: extract synthesis/manufacturing steps from patent or monograph text. "
            "For each step: step number, description, starting materials/reagents, intermediates, yield."
        )
        retrieved = self._retrieve(question, patent_doc_kinds, top_k=60)
        if not retrieved:
            reason = "PATENT_DISCOVERY_GAP" if use_epar_fallback else "NO_EVIDENCE_IN_CORPUS"
            msg = (
                f"No synthesis-relevant passages found for {self.inn} in "
                f"{'EPAR/assessment_report (fallback)' if use_epar_fallback else 'patent corpus'}."
            )
            self._add_unknown(
                unknowns, "synthesis_steps[*]", reason, msg,
                "Attach patent PDFs with full text (claims + examples) to corpus."
            )
            return []

        candidates_map = self._candidates_map(retrieved)
        context = self._context_str(retrieved)
        alias_map, candidates_str = self._build_alias_map(candidates_map)

        result = self._call_llm(
            _SYNTHESIS_INSTRUCTION, context, question, candidates_str, _SynthesisExtractLLM
        )

        if result is None or not result.steps:
            self._add_unknown(
                unknowns, "synthesis_steps[*]", "NO_EVIDENCE_IN_CORPUS",
                f"Synthesis steps not found in patent documents for {self.inn}. "
                "Patent may describe composition/use only (no Example synthesis section).",
            )
            return []

        steps: List[DossierSynthesisStep] = []
        am = alias_map
        for step_llm in result.steps:
            if step_llm.description is None:
                continue
            desc = _ev_to_evidenced_value(step_llm.description, am)
            if desc is None or not desc.evidence_refs:
                self._add_unknown(
                    unknowns, f"synthesis_steps[{step_llm.step_number}].description",
                    "NO_EVIDENCE_IN_CORPUS",
                    "Synthesis step description lacked linked evidence_id.",
                )
                continue

            reagents = _ev_list(step_llm.reagents, am)
            intermediates = _ev_list(step_llm.intermediates, am)
            ev_refs = list(desc.evidence_refs)
            for ev in reagents + intermediates:
                ev_refs.extend(ev.evidence_refs)

            key_docs = list({self._evidence_registry[e].doc_id for e in ev_refs if e in self._evidence_registry})

            # Sprint 7.5 TZ-4: classify synthesis kind
            kind = classify_synthesis_kind(str(desc.value or ""))

            step = DossierSynthesisStep(
                step_number=step_llm.step_number or (len(steps) + 1),
                kind=kind,
                description=desc,
                reagents=reagents,
                intermediates=intermediates,
                source_patent_refs=key_docs,
                evidence_refs=list(set(ev_refs)),
            )
            steps.append(step)

        return steps

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(
        self,
        case_id: Optional[str] = None,
        run_id: Optional[str] = None,
        deadline: Optional[float] = None,
        legacy_sections: Optional[List[Dict[str, Any]]] = None,
        completeness: Optional[Dict[str, Any]] = None,
        source_verdicts: Optional[Dict[str, str]] = None,
    ) -> DossierReport:
        """
        Generate a complete DossierReport v3.0.

        Args:
            case_id: DDKit case identifier
            run_id: Pipeline run identifier (for Redis barrier correlation)
            deadline: Unix timestamp deadline; raises TimeoutError if exceeded
            legacy_sections: Optional v2.x sections[] to embed for backward compat
            completeness: Optional completeness block from DDReportGenerator

        Returns:
            DossierReport (Pydantic object, call .model_dump() for JSON)
        """
        start_ts = time.time()
        case_id = case_id or self.case_id or ""
        logger.info("DossierReportGenerator: starting for INN=%r case_id=%s", self.inn, case_id)

        unknowns: List[DossierUnknown] = []

        def _check_deadline(stage: str) -> None:
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"dossier_report_timeout at stage={stage}")

        # ── A: Passport ──────────────────────────────────────────────────────
        _check_deadline("passport")
        passport = self._generate_passport(unknowns)
        logger.info("passport done (%.1fs)", time.time() - start_ts)

        # ── B: Registrations ─────────────────────────────────────────────────
        _check_deadline("registrations")
        registrations = self._generate_registrations(unknowns)
        logger.info("registrations done (%.1fs)", time.time() - start_ts)

        # ── C: Clinical studies ───────────────────────────────────────────────
        _check_deadline("clinical_studies")
        clinical_studies = self._generate_clinical_studies(unknowns)
        logger.info("clinical_studies done (%.1fs)", time.time() - start_ts)

        # ── D: Patent families ────────────────────────────────────────────────
        _check_deadline("patent_families")
        patent_families = self._generate_patent_families(unknowns)
        logger.info("patent_families done (%.1fs)", time.time() - start_ts)

        # ── E: Synthesis steps ────────────────────────────────────────────────
        _check_deadline("synthesis_steps")
        synthesis_steps = self._generate_synthesis_steps(unknowns)
        logger.info("synthesis_steps done (%.1fs)", time.time() - start_ts)

        # ── F: Assemble report ────────────────────────────────────────────────
        import uuid as _uuid

        report_id = f"dossier_{case_id}_{int(time.time())}"
        evidence_list = list(self._evidence_registry.values())

        # Sprint 7.5 TZ-6b: ensure run_id is always set
        if not run_id:
            run_id = str(_uuid.uuid4())
            logger.info("run_id was None, generated: %s", run_id)

        # Sprint 7.5 TZ-1: build product_contexts + set passport scope
        # Sprint 14 P0.4: build_product_contexts now returns (contexts, suppressed_weak_signals)
        product_contexts, suppressed_weak_signals = build_product_contexts(registrations, evidence_list)
        if len(product_contexts) > 1:
            passport.passport_scope = "multi_context_ambiguous"
            passport.passport_notice = (
                f"This passport contains molecule-level fields only. "
                f"{len(product_contexts)} product contexts detected — "
                "product-specific data (route, dosage, indications) may vary by context."
            )
        else:
            passport.passport_scope = "single_context"

        report = DossierReport(
            schema_version="3.0",
            report_id=report_id,
            case_id=case_id,
            run_id=run_id,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            product_contexts=product_contexts,
            passport=passport,
            registrations=registrations,
            clinical_studies=clinical_studies,
            patent_families=patent_families,
            synthesis_steps=synthesis_steps,
            unknowns=unknowns,
            evidence_registry=evidence_list,
            sections=legacy_sections,
            completeness=completeness,
        )

        # Compute legacy quality scores
        report.dossier_quality = compute_dossier_quality(report)
        # Sprint 14 P0.4: Surface suppressed weak_signal contexts in quality
        if suppressed_weak_signals:
            report.dossier_quality["context_suppressed_weak_signals"] = suppressed_weak_signals

        # Sprint 7.5 TZ-5: compute quality_v2 (Sprint 17: pass source_verdicts for region awareness)
        report.dossier_quality_v2 = compute_dossier_quality_v2(report, source_verdicts=source_verdicts)

        # Sprint 17: Build operator actions from quality gates + source verdicts
        _sv = source_verdicts or {}
        operator_actions: list[dict[str, str]] = []
        _grls_v = _sv.get("grls", "")
        if _grls_v in ("INFRA_UNAVAILABLE", "NOT_CONFIGURED", "SOURCE_TIMEOUT"):
            operator_actions.append({
                "code": "GRLS_TUNNEL_DOWN",
                "severity": "critical",
                "message": f"GRLS regulatory data missing — grls_verdict={_grls_v}",
                "fix_hint": "Run: scripts/start_regulatory_stack.sh, then re-trigger dossier:build",
            })
        qv2 = report.dossier_quality_v2
        if qv2 and qv2.decision_readiness.get("registrations") == "RED":
            operator_actions.append({
                "code": "REGISTRATIONS_RED",
                "severity": "critical",
                "message": "No valid registration data in any region",
                "fix_hint": "Check source seeding: dossier:build response should show seeded grls_card + label",
            })
        if qv2 and qv2.decision_readiness.get("registrations") == "YELLOW":
            # Check if it's due to missing expected region
            missing_region_unk = [
                u for u in (qv2.critical_unknowns or [])
                if u.get("reason_code") == "MISSING_EXPECTED_REGION"
            ]
            if missing_region_unk:
                operator_actions.append({
                    "code": "MISSING_EXPECTED_REGION",
                    "severity": "warning",
                    "message": "Expected RU registration but none found — registrations downgraded to YELLOW",
                    "fix_hint": "Verify VPS tunnel is up, re-run dossier:build to seed GRLS sources",
                })

        # Sprint 7.5 TZ-6b: run_manifest
        elapsed = time.time() - start_ts

        # Sprint 18 WS1.4: Honest stage status — "empty" when stage produced 0 results
        _stage_status = lambda count: "ok" if count > 0 else "empty"  # noqa: E731
        _critical_failures = [a["code"] for a in operator_actions if a.get("severity") == "critical"]

        # Sprint 18 WS1.4: Run-level verdict — single field for screening trust
        _dr = report.dossier_quality_v2.decision_readiness if report.dossier_quality_v2 else {}
        _has_red = any(v == "RED" for v in _dr.values())
        if _critical_failures:
            _run_verdict = "BLOCKED"
        elif _has_red:
            _run_verdict = "DEGRADED"
        else:
            _run_verdict = "PASS"

        report.run_manifest = RunManifest(
            run_id=run_id,
            report_id=report_id,
            case_id=case_id,
            pipeline_version=os.getenv("PIPELINE_VERSION", None),
            config_digest=None,
            stages=[
                {"name": "passport", "status": "ok"},
                {"name": "registrations", "status": _stage_status(len(registrations)), "count": len(registrations)},
                {"name": "clinical_studies", "status": _stage_status(len(clinical_studies)), "count": len(clinical_studies)},
                {"name": "patent_families", "status": _stage_status(len(patent_families)), "count": len(patent_families)},
                {"name": "synthesis_steps", "status": _stage_status(len(synthesis_steps)), "count": len(synthesis_steps)},
                {"name": "total", "elapsed_s": round(elapsed, 1), "run_verdict": _run_verdict},
            ],
            docs_attached=0,
            docs_indexed=0,
            docs_failed=0,
            source_verdicts=_sv,
            operator_actions=operator_actions,
            critical_failures=_critical_failures,
        )

        logger.info(
            "DossierReport v3.0 assembled: contexts=%d registrations=%d "
            "clinical=%d patents=%d synthesis=%d unknowns=%d evidence=%d elapsed=%.1fs "
            "run_id=%s quality_v2_gates=%s",
            len(product_contexts),
            len(registrations), len(clinical_studies), len(patent_families),
            len(synthesis_steps), len(unknowns), len(evidence_list), elapsed,
            run_id,
            report.dossier_quality_v2.decision_readiness if report.dossier_quality_v2 else "N/A",
        )

        return report
