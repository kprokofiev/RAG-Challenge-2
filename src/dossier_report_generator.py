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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    compute_dossier_quality,
)
from src.evidence_builder import EvidenceCandidatesBuilder
from src.retrieval import HybridRetriever

logger = logging.getLogger(__name__)

# ── LLM schemas for structured extraction ────────────────────────────────────

class _EvidencedValueLLM(BaseModel):
    value: Optional[str] = Field(None, description="Extracted value as string")
    evidence_id: Optional[str] = Field(None, description="evidence_id from candidates list")


class _PassportExtractLLM(BaseModel):
    inn: Optional[str] = None
    trade_names: List[_EvidencedValueLLM] = Field(default_factory=list)
    fda_approval_date: Optional[_EvidencedValueLLM] = None
    fda_indication: Optional[_EvidencedValueLLM] = None
    registered_where: List[_EvidencedValueLLM] = Field(default_factory=list)
    chemical_formula: Optional[_EvidencedValueLLM] = None
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
For each field, provide the value AND the evidence_id from the candidates list.
CRITICAL: Only use evidence_ids that appear in the candidates list.
If a field cannot be found in the provided context, set it to null — do NOT guess or hallucinate.
""".strip()

_REGISTRATIONS_INSTRUCTION = """
You are a pharmaceutical dossier extraction system.
Extract marketing authorization records from the provided context.
For each registration (one per country/region), list: region (RU/EU/US/EAEU), status, MAH, identifiers (reg numbers), forms/strengths.
CRITICAL: Only use evidence_ids from the candidates list. If no registration data found, return empty list.
""".strip()

_CLINICAL_INSTRUCTION = """
You are a pharmaceutical dossier extraction system.
Extract structured clinical study cards from the provided context.
For each study: title, registry ID (NCT#), phase, study type, enrollment N, countries, comparator, dosing regimen, key efficacy findings (primary endpoint result, p-value, CI), conclusion, status.
CRITICAL: Each value must reference an evidence_id from the candidates list.
If a field is not stated in context, set it to null. Do NOT hallucinate N, p-values, or conclusions.
""".strip()

_PATENTS_INSTRUCTION = """
You are a pharmaceutical patent dossier extraction system.
Extract patent family records from the provided context.
For each family: family_id (use patent number if INPADOC unknown), representative publication number, priority date, assignees, what_blocks (compound/formulation/method_of_use/synthesis/other), one-sentence summary, country coverage, expiry dates per country (ONLY if explicitly stated in context).
CRITICAL: Only use evidence_ids from candidates list. Never hallucinate expiry dates. If expiry not stated, leave expiry_by_country empty.
""".strip()

_SYNTHESIS_INSTRUCTION = """
You are a pharmaceutical chemistry extraction system.
Extract synthesis/manufacturing steps from the provided patent or monograph context.
For each step: step number, description, reagents/starting materials, intermediates produced.
CRITICAL: Each description must reference an evidence_id from candidates. Do NOT invent synthesis steps not in context.
""".strip()


# ── Evidence registry helper ──────────────────────────────────────────────────

def _ev_id(doc_id: str, page: Optional[int], text: str) -> str:
    """Generate a deterministic evidence_id."""
    h = hashlib.sha256(f"{doc_id}:{page}:{text[:100]}".encode()).hexdigest()[:8]
    return f"ev_{doc_id[:8]}_{page or 0}_{h}"


def _build_evidence(doc_id: str, page: Optional[int], snippet: str,
                    title: Optional[str], source_url: Optional[str]) -> DossierEvidence:
    ev_id = _ev_id(doc_id, page, snippet)
    return DossierEvidence(
        evidence_id=ev_id,
        doc_id=doc_id,
        title=title,
        source_url=source_url,
        page=page,
        snippet=snippet[:400],
    )


def _ev_to_evidenced_value(llm_val: Optional[_EvidencedValueLLM],
                            candidates_map: Dict[str, DossierEvidence]) -> Optional[EvidencedValue]:
    """Convert LLM-extracted value + evidence_id → EvidencedValue, validating evidence_id exists."""
    if llm_val is None or llm_val.value is None:
        return None
    refs = []
    if llm_val.evidence_id and llm_val.evidence_id in candidates_map:
        refs = [llm_val.evidence_id]
    return EvidencedValue(value=llm_val.value, evidence_refs=refs)


def _ev_list(llm_list: List[_EvidencedValueLLM],
             candidates_map: Dict[str, DossierEvidence]) -> List[EvidencedValue]:
    result = []
    for item in llm_list:
        ev = _ev_to_evidenced_value(item, candidates_map)
        if ev is not None:
            result.append(ev)
    return result


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
            if not text:
                continue
            ev = _build_evidence(doc_id, page, text, title, source_url)
            candidates[ev.evidence_id] = ev
            self._evidence_registry[ev.evidence_id] = ev
        return candidates

    def _format_candidates(self, candidates_map: Dict[str, DossierEvidence]) -> str:
        lines = []
        for ev_id, ev in candidates_map.items():
            pg = f"Page {ev.page}" if ev.page else "p.?"
            lines.append(f"- {ev_id}: {ev.title or ev.doc_id} ({pg}): {ev.snippet[:200]}")
        return "\n".join(lines)

    def _call_llm(self, instruction: str, context: str, question: str,
                   candidates_str: str, schema_class) -> Optional[Any]:
        """Call LLM with structured output schema; returns parsed Pydantic object or None."""
        try:
            import inspect
            schema_str = str(schema_class.model_json_schema())
            system_prompt = (
                f"{instruction}\n\n"
                f"Your answer MUST be valid JSON matching this schema:\n```\n{schema_str}\n```\n\n"
                "CRITICAL: Only use evidence_ids that appear in the Available Evidence Candidates section below."
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

    def _has_patent_corpus(self) -> bool:
        """Check if any patent doc_kinds are present in the downloaded corpus."""
        patent_kinds = {"patent_family", "ops", "patent_pdf", "ru_patent_fips", "patent"}
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

    # ── Block generators ─────────────────────────────────────────────────────

    def _generate_passport(self, unknowns: List[DossierUnknown]) -> DossierPassport:
        """Stage A+B+C+D for passport block."""
        passport_doc_kinds = ["label", "us_fda", "epar", "smpc", "grls_card", "grls", "drug_monograph"]
        question = (
            f"Extract drug passport fields for {self.inn}: INN, trade names, FDA approval date, "
            "FDA indication, registered jurisdictions, chemical formula, drug class, mechanism of action, "
            "MAH holders, route of administration, dosage forms, key dosing regimens."
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
        candidates_str = self._format_candidates(candidates_map)

        result = self._call_llm(
            _PASSPORT_INSTRUCTION, context, question, candidates_str, _PassportExtractLLM
        )

        if result is None:
            self._add_unknown(
                unknowns, "passport.*", "EXTRACTION_FAILED",
                "LLM failed to extract passport fields.",
            )
            return DossierPassport(inn=self.inn)

        cm = candidates_map
        passport = DossierPassport(
            inn=result.inn or self.inn,
            trade_names=_ev_list(result.trade_names, cm),
            fda_approval_date=_ev_to_evidenced_value(result.fda_approval_date, cm),
            fda_indication=_ev_to_evidenced_value(result.fda_indication, cm),
            registered_where=_ev_list(result.registered_where, cm),
            chemical_formula=_ev_to_evidenced_value(result.chemical_formula, cm),
            drug_class=_ev_to_evidenced_value(result.drug_class, cm),
            mechanism_of_action=_ev_to_evidenced_value(result.mechanism_of_action, cm),
            mah_holders=_ev_list(result.mah_holders, cm),
            route_of_administration=_ev_to_evidenced_value(result.route_of_administration, cm),
            dosage_forms=_ev_list(result.dosage_forms, cm),
            key_dosages=_ev_list(result.key_dosages, cm),
        )

        # Validate: fields without evidence → unknowns
        mandatory_fields = [
            ("passport.fda_approval_date", passport.fda_approval_date),
            ("passport.fda_indication", passport.fda_indication),
            ("passport.drug_class", passport.drug_class),
            ("passport.mechanism_of_action", passport.mechanism_of_action),
        ]
        for field_path, ev_val in mandatory_fields:
            if ev_val is None or not ev_val.evidence_refs:
                self._add_unknown(
                    unknowns, field_path, "NO_EVIDENCE_IN_CORPUS",
                    f"Field {field_path} could not be extracted with evidence from available documents.",
                    "Ensure FDA label or EPAR is in the corpus and properly indexed."
                )

        if not passport.trade_names:
            self._add_unknown(
                unknowns, "passport.trade_names", "NO_EVIDENCE_IN_CORPUS",
                f"No trade names found in corpus for {self.inn}.",
            )

        return passport

    def _generate_registrations(self, unknowns: List[DossierUnknown]) -> List[DossierRegistration]:
        """Generate registration records for RU, EU, US."""
        regions = [
            ("RU", ["grls_card", "grls", "ru_instruction", "eaeu_document"],
             "What are the GRLS registration numbers, trade names, MAH, drug forms, and status?"),
            ("EU", ["epar", "smpc", "assessment_report"],
             "What are the EMA marketing authorization numbers, MAH, authorized forms, and status?"),
            ("US", ["label", "us_fda", "approval_letter", "anda_package"],
             "What are the FDA NDA/BLA numbers, applicant names, drug forms, and approval dates?"),
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
            candidates_str = self._format_candidates(candidates_map)

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
                cm = candidates_map
                ev_refs: List[str] = []
                status = _ev_to_evidenced_value(reg_llm.status, cm)
                if status and status.evidence_refs:
                    ev_refs.extend(status.evidence_refs)
                mah = _ev_to_evidenced_value(reg_llm.mah, cm)
                if mah and mah.evidence_refs:
                    ev_refs.extend(mah.evidence_refs)
                identifiers = _ev_list(reg_llm.identifiers, cm)
                for ev in identifiers:
                    ev_refs.extend(ev.evidence_refs)

                registration = DossierRegistration(
                    region=reg_llm.region or region,
                    status=status,
                    forms_strengths=_ev_list(reg_llm.forms_strengths, cm),
                    mah=mah,
                    identifiers=identifiers,
                    evidence_refs=list(set(ev_refs)),
                )
                registrations.append(registration)

        return registrations

    def _generate_clinical_studies(self, unknowns: List[DossierUnknown]) -> List[DossierClinicalStudy]:
        """Generate structured clinical study cards."""
        clinical_doc_kinds = [
            "ctgov_protocol", "ctgov_results", "ctgov", "trial_registry",
            "scientific_pmc", "scientific_pdf", "publication",
        ]
        question = (
            f"For {self.inn}: list all phase 2/3 clinical studies. "
            "For each: NCT ID, trial name, phase, study type (RCT/non-RCT/obs), "
            "enrollment N, countries, comparator, dosing regimen, primary endpoint result, conclusion, status."
        )
        retrieved = self._retrieve(question, clinical_doc_kinds, top_k=55)
        if not retrieved:
            self._add_unknown(
                unknowns, "clinical_studies[*]", "NO_DOCUMENT_IN_CORPUS",
                f"No clinical documents (CTGov/publications) indexed for {self.inn}.",
                "Ensure CTGov protocol/results pages are in the corpus."
            )
            return []

        candidates_map = self._candidates_map(retrieved)
        context = self._context_str(retrieved)
        candidates_str = self._format_candidates(candidates_map)

        result = self._call_llm(
            _CLINICAL_INSTRUCTION, context, question, candidates_str, _ClinicalStudiesExtractLLM
        )

        if result is None or not result.studies:
            self._add_unknown(
                unknowns, "clinical_studies[*]", "NO_EVIDENCE_IN_CORPUS",
                f"Could not extract clinical study cards for {self.inn} from available documents.",
            )
            return []

        studies: List[DossierClinicalStudy] = []
        cm = candidates_map
        for i, study_llm in enumerate(result.studies):
            ev_refs: List[str] = []

            def _collect(ev_val: Optional[EvidencedValue]) -> Optional[EvidencedValue]:
                if ev_val and ev_val.evidence_refs:
                    ev_refs.extend(ev_val.evidence_refs)
                return ev_val

            title = _collect(_ev_to_evidenced_value(study_llm.title, cm))
            study_id = _collect(_ev_to_evidenced_value(study_llm.study_id, cm))
            phase = _collect(_ev_to_evidenced_value(study_llm.phase, cm))
            study_type = _collect(_ev_to_evidenced_value(study_llm.study_type, cm))
            n_enrolled = _collect(_ev_to_evidenced_value(study_llm.n_enrolled, cm))
            comparator = _collect(_ev_to_evidenced_value(study_llm.comparator, cm))
            regimen = _collect(_ev_to_evidenced_value(study_llm.regimen_dosing, cm))
            conclusion = _collect(_ev_to_evidenced_value(study_llm.conclusion, cm))
            status = _collect(_ev_to_evidenced_value(study_llm.status, cm))
            countries = _ev_list(study_llm.countries, cm)
            efficacy = _ev_list(study_llm.efficacy_keypoints, cm)
            for ev in countries + efficacy:
                ev_refs.extend(ev.evidence_refs)

            if not ev_refs:
                # Study card has no evidence at all → skip + log unknown
                self._add_unknown(
                    unknowns, f"clinical_studies[{i}].*", "NO_EVIDENCE_IN_CORPUS",
                    "LLM extracted study card but could not link any evidence_ids.",
                )
                continue

            study = DossierClinicalStudy(
                title=title, study_id=study_id, phase=phase, study_type=study_type,
                n_enrolled=n_enrolled, countries=countries, comparator=comparator,
                regimen_dosing=regimen, efficacy_keypoints=efficacy,
                conclusion=conclusion, status=status,
                evidence_refs=list(set(ev_refs)),
            )
            studies.append(study)

        if not studies:
            self._add_unknown(
                unknowns, "clinical_studies[*]", "NO_EVIDENCE_IN_CORPUS",
                "All extracted study cards lacked linked evidence; see individual unknowns above.",
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

        patent_doc_kinds = ["patent_family", "ops", "patent_pdf", "ru_patent_fips", "patent"]
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
        candidates_str = self._format_candidates(candidates_map)

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
        cm = candidates_map
        for fam_llm in result.families:
            ev_refs: List[str] = []

            rep = _ev_to_evidenced_value(fam_llm.representative_pub, cm)
            if rep and rep.evidence_refs:
                ev_refs.extend(rep.evidence_refs)
            priority = _ev_to_evidenced_value(fam_llm.priority_date, cm)
            if priority and priority.evidence_refs:
                ev_refs.extend(priority.evidence_refs)
            what_blocks = _ev_to_evidenced_value(fam_llm.what_blocks, cm)
            if what_blocks and what_blocks.evidence_refs:
                ev_refs.extend(what_blocks.evidence_refs)
            summary = _ev_to_evidenced_value(fam_llm.summary, cm)
            if summary and summary.evidence_refs:
                ev_refs.extend(summary.evidence_refs)
            assignees = _ev_list(fam_llm.assignees, cm)
            coverage = _ev_list(fam_llm.country_coverage, cm)
            expiry = _ev_list(fam_llm.expiry_by_country, cm)
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

            # Expiry without data → warn in unknowns
            if not expiry:
                self._add_unknown(
                    unknowns, f"patent_families[{family_id}].expiry_by_country",
                    "LEGAL_STATUS_NOT_AVAILABLE",
                    f"Patent expiry dates not found in corpus for family {family_id}. "
                    "Legal status data requires EPO Register or OrangeBook integration.",
                    "Integrate EPO Register API (eporegister.go) or OrangeBook service."
                )

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
            families.append(fam)

        return families

    def _generate_synthesis_steps(self, unknowns: List[DossierUnknown]) -> List[DossierSynthesisStep]:
        """Generate synthesis steps from patent text."""
        if not self._has_patent_corpus():
            self._add_unknown(
                unknowns, "synthesis_steps[*]", "PATENT_DISCOVERY_GAP",
                f"No patent documents in corpus for {self.inn}. Cannot extract synthesis steps.",
                "Attach patent PDFs to corpus. See patent_families unknowns for discovery gap details."
            )
            return []

        patent_doc_kinds = ["patent_pdf", "patent", "drug_monograph"]
        question = (
            f"For {self.inn}: extract synthesis/manufacturing steps from patent or monograph text. "
            "For each step: step number, description, starting materials/reagents, intermediates, yield."
        )
        retrieved = self._retrieve(question, patent_doc_kinds, top_k=60)
        if not retrieved:
            self._add_unknown(
                unknowns, "synthesis_steps[*]", "NO_EVIDENCE_IN_CORPUS",
                f"Patent corpus present but no synthesis-relevant passages found for {self.inn}. "
                "Patent may be title-only (RU CSV) without claims/synthesis text.",
                "Ensure patent PDF (full text) is attached, not only metadata."
            )
            return []

        candidates_map = self._candidates_map(retrieved)
        context = self._context_str(retrieved)
        candidates_str = self._format_candidates(candidates_map)

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
        cm = candidates_map
        for step_llm in result.steps:
            if step_llm.description is None:
                continue
            desc = _ev_to_evidenced_value(step_llm.description, cm)
            if desc is None or not desc.evidence_refs:
                self._add_unknown(
                    unknowns, f"synthesis_steps[{step_llm.step_number}].description",
                    "NO_EVIDENCE_IN_CORPUS",
                    "Synthesis step description lacked linked evidence_id.",
                )
                continue

            reagents = _ev_list(step_llm.reagents, cm)
            intermediates = _ev_list(step_llm.intermediates, cm)
            ev_refs = list(desc.evidence_refs)
            for ev in reagents + intermediates:
                ev_refs.extend(ev.evidence_refs)

            key_docs = list({self._evidence_registry[e].doc_id for e in ev_refs if e in self._evidence_registry})

            step = DossierSynthesisStep(
                step_number=step_llm.step_number or (len(steps) + 1),
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
        report_id = f"dossier_{case_id}_{int(time.time())}"
        evidence_list = list(self._evidence_registry.values())

        report = DossierReport(
            schema_version="3.0",
            report_id=report_id,
            case_id=case_id,
            run_id=run_id,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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

        # Compute quality scores
        report.dossier_quality = compute_dossier_quality(report)

        elapsed = time.time() - start_ts
        logger.info(
            "DossierReport v3.0 assembled: passport_fields=%d registrations=%d "
            "clinical=%d patents=%d synthesis=%d unknowns=%d evidence=%d elapsed=%.1fs",
            len(evidence_list),
            len(registrations), len(clinical_studies), len(patent_families),
            len(synthesis_steps), len(unknowns), len(evidence_list), elapsed,
        )

        return report
