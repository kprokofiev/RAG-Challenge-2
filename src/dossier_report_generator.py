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

# ── Authority-tiering policy (S6-T2) ─────────────────────────────────────────
# Maps passport/registration field → allowed Tier-1 doc_kinds.
# Fields marked Tier-2 are populated only from listed doc_kinds with confidence=medium.
# Registration status/numbers are FORBIDDEN from Tier-2 sources.
#
# Tier-1: authoritative regulatory filings (label, EPAR, SmPC, GRLS, DailyMed, Drugs@FDA)
# Tier-2: secondary (moa_overview, drug_monograph, press_release — only for MoA/class)

FIELD_ALLOWED_SOURCES: Dict[str, List[str]] = {
    # Registration facts — strict Tier-1 only
    "registered_where":        ["epar", "smpc", "label", "us_fda", "grls_card", "grls", "eaeu_document"],
    "fda_approval_date":       ["label", "us_fda", "approval_letter"],
    "mah_holders":             ["smpc", "epar", "grls_card", "grls", "ru_instruction", "label"],
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
7. comparator: comparator arm description (placebo, active control, etc.)
8. dosing_regimen: dosing details from the intervention description
9. primary_endpoint: primary outcome measure description
10. primary_result: primary endpoint result value with units
11. p_value: p-value for primary endpoint (as string, e.g., "0.001", "<0.0001")
12. confidence_interval: confidence interval (e.g., "95% CI: 0.65-0.82")
13. status: extract from OverallStatus field (e.g., "Completed", "Recruiting", "Terminated", "Active, not recruiting")
14. conclusion: if an explicit conclusion is stated, use it verbatim. If NOT, synthesize 1-2 sentences
    from primary outcome numeric results (effect size, p-value, CI). If no results data exists, set null.

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
6. summary: one-sentence description of what the patent covers (from title/abstract)
7. country_coverage: list of countries where patent publications exist (derive from publication
   numbers: EP->EP, WO->WO, US->US, JP->JP, CN->CN, RU->RU, etc.)
8. expiry_by_country: expiry dates per country. ONLY fill if expiry/legal status is EXPLICITLY
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
        """Retrieve top-K evidence candidates for a question with given doc_kind filter."""
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
        """Call LLM with structured output schema; returns parsed Pydantic object or None."""
        try:
            import inspect
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
        patent_kinds = {"patent_family", "ops", "patent_pdf", "ru_patent_fips", "patent", "patent_family_summary"}
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
        # "first approved: 2000" (year-only fallback)
        re.compile(
            r"first\s+approved\s*:\s*(\d{4})\b",
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

    # ── Block generators ─────────────────────────────────────────────────────

    def _generate_passport(self, unknowns: List[DossierUnknown]) -> DossierPassport:
        """Stage A+B+C+D for passport block. S6: includes PubChem chemistry fields."""
        # S6-T2: Authority-tiering — Tier-1 sources first, pubchem for chemistry
        passport_doc_kinds = [
            "label", "us_fda", "epar", "smpc", "grls_card", "grls",
            "ru_instruction", "pubchem", "drug_monograph",
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

        # S6-T4: PubChem chemistry block — if no pubchem doc indexed, add typed unknown
        if not any([passport.smiles, passport.inchi_key, passport.chemical_formula]):
            self._add_unknown(
                unknowns, "passport.chemistry", "NO_DOCUMENT_IN_CORPUS",
                f"No chemistry data (SMILES/InChIKey/formula) for {self.inn}. "
                "PubChem document not in corpus.",
                "Ensure PubChem compound data (doc_kind=pubchem) is attached to corpus."
            )

        return passport

    def _generate_registrations(self, unknowns: List[DossierUnknown]) -> List[DossierRegistration]:
        """Generate registration records for RU, EU, US."""
        # S6-T6: EAEU is NOT in the live search list because the EAEU portal client
        # is a URL-builder stub only (pharm_search/packages/clients/ru/eaeu/eaeu.go).
        # It gets a forced typed-unknown below instead of being searched.
        regions = [
            ("RU", ["grls_card", "grls", "ru_instruction"],
             "What are the GRLS registration numbers, trade names, MAH, drug forms, and status for this drug in Russia?"),
            ("EU", ["epar", "smpc", "assessment_report", "pil"],
             "What are the EMA marketing authorization numbers, MAH, authorized forms, and status for this drug in the EU?"),
            ("US", ["label", "us_fda", "approval_letter", "anda_package"],
             "What are the FDA NDA/BLA numbers, applicant names, drug forms, and approval dates for this drug in the US?"),
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

        # S6-T6: EAEU typed-unknown — always add because client is a stub.
        # TZ §5.2: if EAEU not implemented → write typed unknown, do NOT create empty row.
        self._add_unknown(
            unknowns, "registrations[EAEU].*", "EAEU_NOT_IMPLEMENTED",
            f"EAEU drug registry data unavailable for {self.inn}. "
            "The EAEU portal client (pharm_search/packages/clients/ru/eaeu/eaeu.go) "
            "is a URL-builder stub that does not query portal.eaeunion.org.",
            "Implement HTTP client for portal.eaeunion.org public REST API "
            "(endpoint: /sites/portal/ru-ru/Pages/registry/register-of-drugs.aspx). "
            "Replace BuildEAEULink stub with real drug registry search.",
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
            f"Extract: title, study_id, phase, study_type, n_enrolled, countries, "
            f"comparator, regimen_dosing, efficacy_keypoints, conclusion, status.\n\n"
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

    def _generate_clinical_studies(self, unknowns: List[DossierUnknown]) -> List[DossierClinicalStudy]:
        """Generate structured clinical study cards.

        Sprint 13 WS1: Per-study assembly pipeline —
        1. Pre-scan corpus for NCT IDs (deterministic candidate list)
        2. For each NCT ID: scoped retrieval → scoped LLM extraction → one card
        3. Dedup by NCT ID
        4. Only cards with study_id pass (enforced by construction)

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

        patent_doc_kinds = ["patent_family", "ops", "patent_pdf", "ru_patent_fips", "patent", "patent_family_summary"]
        question = (
            f"For {self.inn}: identify all patent families. "
            "For each: representative publication number (EP/WO/US), priority date, assignee, "
            "what it protects (compound/formulation/method_of_use/synthesis), one-sentence summary, "
            "country coverage, expiry dates per country (only if explicitly stated)."
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

        return families

    def _generate_synthesis_steps(self, unknowns: List[DossierUnknown]) -> List[DossierSynthesisStep]:
        """Generate synthesis steps from patent text, with EPAR/assessment_report fallback."""
        # S7-E1: Primary sources are patents; if no patent corpus, try EPAR as fallback
        _PATENT_DOC_KINDS = ["patent_pdf", "patent", "drug_monograph"]
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

        # Sprint 7.5 TZ-5: compute quality_v2
        report.dossier_quality_v2 = compute_dossier_quality_v2(report)

        # Sprint 7.5 TZ-6b: run_manifest
        elapsed = time.time() - start_ts
        report.run_manifest = RunManifest(
            run_id=run_id,
            report_id=report_id,
            case_id=case_id,
            pipeline_version=os.getenv("PIPELINE_VERSION", None),
            config_digest=None,
            stages=[
                {"name": "passport", "status": "ok"},
                {"name": "registrations", "status": "ok", "count": len(registrations)},
                {"name": "clinical_studies", "status": "ok", "count": len(clinical_studies)},
                {"name": "patent_families", "status": "ok", "count": len(patent_families)},
                {"name": "synthesis_steps", "status": "ok", "count": len(synthesis_steps)},
                {"name": "total", "elapsed_s": round(elapsed, 1)},
            ],
            docs_attached=0,
            docs_indexed=0,
            docs_failed=0,
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
