from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# NOTE:
# These schemas are used with OpenAI's structured outputs (`beta.chat.completions.parse`).
# The JSON Schema subset supported by OpenAI is stricter than full JSON Schema/Pydantic.
#
# In particular, unconstrained `Any`/`List[Any]` often produces schema fragments like
# `{"type":"array","items":{}}` which OpenAI rejects ("items must have a type").
#
# For our case-view UI we mostly need strings and lists of strings for evidence-locked fields.
# If we later need richer payloads, introduce dedicated typed models instead of widening `value`.
JSONValue = Union[str, List[str], None]


class EvidenceLockedValue(BaseModel):
    """
    A value extracted from documents, linked to evidence candidates via evidence_ids.
    """

    value: JSONValue = Field(default=None)
    evidence_ids: List[str] = Field(default_factory=list)
    note: Optional[str] = Field(default=None, description="Optional short note about interpretation")


class RegulatoryMarketExtraction(BaseModel):
    """
    Extracted regulatory facts for a market (US/EU).
    """

    trade_names: Optional[EvidenceLockedValue] = None
    holders: Optional[EvidenceLockedValue] = None
    dosage_forms_and_strengths: Optional[EvidenceLockedValue] = None
    status: Optional[EvidenceLockedValue] = None
    countries_covered: Optional[EvidenceLockedValue] = None
    notes: Optional[str] = None


class RuRegulatoryEntryExtraction(BaseModel):
    """
    Extracted RU regulatory entry (GRLS/EAES).
    """

    trade_name: Optional[EvidenceLockedValue] = None
    holder: Optional[EvidenceLockedValue] = None
    reg_number: Optional[EvidenceLockedValue] = None
    reg_date: Optional[EvidenceLockedValue] = None
    dosage_forms: Optional[EvidenceLockedValue] = None
    status: Optional[EvidenceLockedValue] = None
    notes: Optional[str] = None


class RuRegulatoryExtraction(BaseModel):
    """
    Extracted RU regulatory facts (list of registration entries).
    """

    entries: List[RuRegulatoryEntryExtraction] = Field(default_factory=list)
    notes: Optional[str] = None


class PassportExtraction(BaseModel):
    """
    Missing passport fields extracted from authoritative documents.
    """

    fda_approval: Optional[EvidenceLockedValue] = None
    chemical_formula: Optional[EvidenceLockedValue] = None
    drug_class: Optional[EvidenceLockedValue] = None
    notes: Optional[str] = None


class InstructionHighlightsExtraction(BaseModel):
    """
    Highlights extracted from label / instruction / SmPC/PIL.
    """

    indications: List[EvidenceLockedValue] = Field(default_factory=list)
    dosing: List[EvidenceLockedValue] = Field(default_factory=list)
    restrictions: List[EvidenceLockedValue] = Field(default_factory=list)
    notes: Optional[str] = None


class TrialCardExtraction(BaseModel):
    """
    Clinical trial card extracted from registries / publications.
    """

    trial_id: str = Field(default="", description="NCT / CTIS / registry identifier")
    title: str = Field(default="")
    phase: str = Field(default="")
    study_type: str = Field(default="", description="randomized / non-randomized / observational / etc.")
    countries: List[str] = Field(default_factory=list)
    enrollment: Optional[str] = Field(default=None, description="Number of patients (can be string if ranges)")
    comparator: Optional[str] = Field(default=None)
    regimen: Optional[str] = Field(default=None, description="Therapy regimen incl. dosing")
    status: Optional[str] = Field(default=None, description="Completed / Recruiting / Active / etc.")
    efficacy_key_points: List[str] = Field(default_factory=list)
    conclusion: Optional[str] = None
    where_conducted: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)


class TrialsExtraction(BaseModel):
    trials: List[TrialCardExtraction] = Field(default_factory=list)
    notes: Optional[str] = None


class PublicationItemExtraction(BaseModel):
    category: Literal["comparative", "abstracts", "real_world", "combination"] = Field(
        description="Target bucket for UI"
    )
    title: str = Field(default="")
    summary: str = Field(default="")
    evidence_ids: List[str] = Field(default_factory=list)


class PublicationsExtraction(BaseModel):
    items: List[PublicationItemExtraction] = Field(default_factory=list)
    notes: Optional[str] = None


class PatentCoverageByCountryItem(BaseModel):
    country: str = Field(default="", description="Jurisdiction/country code (e.g., US, EP)")
    expires_at: Optional[str] = Field(default=None, description="Expiry/term date if explicitly stated")
    evidence_ids: List[str] = Field(default_factory=list)


class PatentJurisdictionStatusItem(BaseModel):
    jurisdiction: str = Field(default="", description="Jurisdiction/country code (e.g., US, EP)")
    status: str = Field(default="", description="Granted / Pending / Expired / etc.")
    event_date: Optional[str] = Field(default=None, description="Event/record date if present")
    publication_number: Optional[str] = Field(default=None, description="Publication number if present")
    evidence_ids: List[str] = Field(default_factory=list)


class PatentFamilyInsightExtraction(BaseModel):
    """
    Insight for a single patent family or representative patent document.
    """

    coverage_type: Optional[EvidenceLockedValue] = Field(
        default=None,
        description="What the family covers. Expected values: composition, treatment, synthesis (can be list).",
    )
    summary: Optional[EvidenceLockedValue] = Field(default=None, description="2-4 sentence summary of what is claimed")
    key_points: List[EvidenceLockedValue] = Field(default_factory=list, description="Optional key points/claims")
    key_claims: List[EvidenceLockedValue] = Field(
        default_factory=list,
        description="Optional extracted key (preferably independent) claims. Include claim number if possible.",
    )
    coverage_by_country: List[PatentCoverageByCountryItem] = Field(
        default_factory=list,
        description="Optional extracted coverage by country with expires_at if explicitly stated. "
                    "Each item: {country, expires_at, evidence_ids}.",
    )
    jurisdiction_statuses: List[PatentJurisdictionStatusItem] = Field(
        default_factory=list,
        description="Optional legal/jurisdiction status highlights. "
                    "Each item: {jurisdiction, status, event_date, publication_number, evidence_ids}.",
    )
    notes: Optional[str] = None


class SynthesisStepExtraction(BaseModel):
    text: str = Field(default="")
    evidence_ids: List[str] = Field(default_factory=list)


class SynthesisExtraction(BaseModel):
    steps: List[SynthesisStepExtraction] = Field(default_factory=list)
    treatment_method_from_patents: Optional[EvidenceLockedValue] = None
    notes: Optional[str] = None
