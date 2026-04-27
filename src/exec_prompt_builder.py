"""
Exec prompt builder and config loader for the decision engine.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

import yaml
from pydantic import BaseModel, ConfigDict, Field


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DECISION_LIBRARY_PATH = _CONFIG_DIR / "exec_decision_library.yaml"
_MODEL_PROFILES_PATH = _CONFIG_DIR / "exec_model_profiles.yaml"
_QUESTION_LIBRARY_PATH = _CONFIG_DIR / "exec_question_library.yaml"


class BlockSpec(BaseModel):
    block_id: str
    title: str
    block_class: Literal["critical", "secondary"]
    verdict_family: str
    verdicts: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)
    field_prefixes: List[str] = Field(default_factory=list)
    allowed_doc_kinds: List[str] = Field(default_factory=list)
    escalation_classes: List[str] = Field(default_factory=list)
    appendix_question_ids: List[str] = Field(default_factory=list)
    topline_key: Optional[str] = None


class BlockClassPolicy(BaseModel):
    preferred_model: str
    fallback_model: str
    allow_nano_final: bool = True


class ModelProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_ladder: List[str] = Field(default_factory=list)
    default_requested_model: str = "gpt-5.4-mini"
    planner_model: str = "gpt-5.4-mini"
    answerer_model: str = "gpt-5.4-mini"
    summary_model: str = "gpt-5.4-mini"
    verifier_model: str = "gpt-5.4-mini"
    thinking_defaults: Dict[str, str] = Field(default_factory=dict)
    block_classes: Dict[str, BlockClassPolicy] = Field(default_factory=dict)
    reserve_thresholds: Dict[str, int] = Field(default_factory=dict)
    manual_overrides: Dict[str, Any] = Field(default_factory=dict)


class ExecReasonerClaim(BaseModel):
    claim: str
    claim_type: Literal["hard_evidence_backed", "inference", "tentative"] = "inference"
    evidence_refs: List[str] = Field(default_factory=list)


class ExecReasonerContradiction(BaseModel):
    summary: str
    evidence_refs: List[str] = Field(default_factory=list)


class ExecReasonerBlocker(BaseModel):
    title: str
    severity: Literal["NON_BLOCKING", "IMPORTANT", "DECISION_BLOCKING", "MUST_VERIFY_NOW"] = "IMPORTANT"
    rationale: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)


class ExecReasonerAction(BaseModel):
    action: str
    priority: str = "NEXT"
    rationale: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)


class ExecReasonerOutput(BaseModel):
    verdict: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    sufficiency: Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"] = "INSUFFICIENT"
    short_answer: str
    full_answer: str
    why_this_verdict: List[ExecReasonerClaim] = Field(default_factory=list)
    contradictions: List[ExecReasonerContradiction] = Field(default_factory=list)
    decision_blockers: List[ExecReasonerBlocker] = Field(default_factory=list)
    next_actions: List[ExecReasonerAction] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    top_evidence_refs: List[str] = Field(default_factory=list)
    missing_evidence_classes: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)


class ExecDocKindLimit(BaseModel):
    doc_kind: str
    max_chunks: int = 4


class ExecRetrievalPlan(BaseModel):
    doc_kinds: List[str] = Field(default_factory=list)
    queries: List[str] = Field(default_factory=list)
    max_docs: int = 12
    max_chunks: int = 30
    chunk_policy: str = "prefer source-native confirmation chunks"
    doc_kind_limits: List[ExecDocKindLimit] = Field(default_factory=list)


class ExecAnswerContract(BaseModel):
    verdict: List[str] = Field(default_factory=list)
    must_include: List[str] = Field(default_factory=list)


class ExecPolicyGates(BaseModel):
    positive_verdict_requires: List[str] = Field(default_factory=list)
    hold_requires: List[str] = Field(default_factory=list)
    no_go_triggers: List[str] = Field(default_factory=list)


class ExecQuestionPlan(BaseModel):
    question_id: str
    answer_type: str
    business_lens: str = ""
    needed_facts: List[str] = Field(default_factory=list)
    needed_dossier_sections: List[str] = Field(default_factory=list)
    retrieval_plan: ExecRetrievalPlan = Field(default_factory=ExecRetrievalPlan)
    answer_schema: ExecAnswerContract = Field(default_factory=ExecAnswerContract)
    gates: ExecPolicyGates = Field(default_factory=ExecPolicyGates)
    policy_notes: List[str] = Field(default_factory=list)


class PromptPackage(BaseModel):
    block_spec: BlockSpec
    requested_model: str
    thinking_mode: str
    max_output_tokens: Optional[int] = None
    system_content: str
    human_content: str
    response_model: Type[BaseModel]
    phase: str = "final"


def _optional_env_int(name: str) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _bounded_output_tokens(name: str, default: int) -> int:
    value = _optional_env_int(name)
    if value is None:
        return default
    return max(256, value)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def load_exec_decision_library(path: Optional[Path] = None) -> Dict[str, BlockSpec]:
    raw = _read_yaml(path or _DECISION_LIBRARY_PATH)
    blocks = raw.get("blocks", []) or []
    return {item["block_id"]: BlockSpec.model_validate(item) for item in blocks if item.get("block_id")}


@lru_cache(maxsize=1)
def load_exec_model_profiles(path: Optional[Path] = None) -> Dict[str, ModelProfile]:
    raw = _read_yaml(path or _MODEL_PROFILES_PATH)
    profiles = raw.get("profiles", {}) or {}
    return {name: ModelProfile.model_validate(payload) for name, payload in profiles.items()}


@lru_cache(maxsize=1)
def load_exec_question_library(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    raw = _read_yaml(path or _QUESTION_LIBRARY_PATH)
    questions = raw.get("questions", []) or []
    return {item["id"]: item for item in questions if item.get("id")}


def get_model_profile(profile_name: Optional[str] = None) -> ModelProfile:
    profiles = load_exec_model_profiles()
    resolved_name = (profile_name or os.getenv("DDKIT_EXEC_MODEL_PROFILE") or "smart").strip()
    return profiles.get(resolved_name) or next(iter(profiles.values()))


def block_thinking_mode(block_spec: BlockSpec, profile: Optional[ModelProfile] = None, phase: str = "final") -> str:
    active_profile = profile or get_model_profile()
    defaults = active_profile.thinking_defaults
    if phase == "summary":
        return os.getenv("DDKIT_EXEC_THINKING_SUMMARY", defaults.get("summary", "off"))
    if phase == "verifier":
        return defaults.get("verifier", "medium")
    if block_spec.block_class == "critical":
        return os.getenv("DDKIT_EXEC_THINKING_CRITICAL", defaults.get("critical", "high"))
    return os.getenv("DDKIT_EXEC_THINKING_DEFAULT", defaults.get("secondary", "medium"))


def planner_thinking_mode(profile: Optional[ModelProfile] = None) -> str:
    active_profile = profile or get_model_profile()
    return os.getenv("DDKIT_EXEC_THINKING_PLANNER", active_profile.thinking_defaults.get("planner", "high"))


def answerer_thinking_mode(profile: Optional[ModelProfile] = None) -> str:
    active_profile = profile or get_model_profile()
    return os.getenv("DDKIT_EXEC_THINKING_ANSWERER", active_profile.thinking_defaults.get("answerer", "high"))


def block_requested_model(block_spec: BlockSpec, profile: Optional[ModelProfile] = None, phase: str = "final") -> str:
    active_profile = profile or get_model_profile()
    if phase == "summary":
        return active_profile.summary_model
    if phase == "verifier":
        return active_profile.verifier_model
    policy = active_profile.block_classes.get(block_spec.block_class)
    if policy:
        return policy.preferred_model
    return active_profile.default_requested_model


def planner_requested_model(profile: Optional[ModelProfile] = None) -> str:
    active_profile = profile or get_model_profile()
    return (os.getenv("DDKIT_EXEC_PLANNER_MODEL") or active_profile.planner_model or active_profile.default_requested_model).strip()


def answerer_requested_model(block_spec: BlockSpec, profile: Optional[ModelProfile] = None) -> str:
    active_profile = profile or get_model_profile()
    explicit = (os.getenv("DDKIT_EXEC_ANSWERER_MODEL") or active_profile.answerer_model or "").strip()
    if explicit:
        return explicit
    return block_requested_model(block_spec, active_profile, phase="final")


def _prompt_contract(block_spec: BlockSpec) -> str:
    verdicts = ", ".join(block_spec.verdicts)
    escalation = ", ".join(block_spec.escalation_classes) or "none"
    allowed_doc_kinds = ", ".join(block_spec.allowed_doc_kinds) or "any dossier-backed evidence"
    return (
        f"You are writing the '{block_spec.title}' block of an executive decision memo.\n"
        f"Verdict family: {block_spec.verdict_family}. Allowed verdicts: {verdicts}.\n"
        "Rules:\n"
        "1. Do not invent facts. Use only dossier packet evidence and cited evidence aliases.\n"
        "2. If evidence is weak or contradictory, use lower confidence and lower sufficiency.\n"
        "3. Hard evidence backed claims must cite real evidence_refs from the packet.\n"
        "4. Unknowns and blockers must not be ignored. Promote them into caveats when material.\n"
        "5. Do not convert source unavailable into not found.\n"
        "6. If evidence is insufficient for a hard verdict, return the explicit insufficient verdict for this block family.\n"
        f"Allowed doc kinds for extra escalation context: {allowed_doc_kinds}.\n"
        f"Potential escalation classes: {escalation}.\n"
        "Return only the structured schema."
    )


def _planner_prompt_contract(block_spec: BlockSpec, question_trace: Dict[str, Any]) -> str:
    verdicts = ", ".join(block_spec.verdicts)
    contract = (
        f"You are planning evidence requirements for the '{block_spec.title}' executive question.\n"
        f"Verdict family: {block_spec.verdict_family}. Allowed verdicts later: {verdicts}.\n"
        f"Business lens: {question_trace.get('business_lens', '') or 'exec'}.\n"
        "Return a strict data contract, not prose.\n"
        "Rules:\n"
        "1. Use compact dossier state only to identify what is already known vs unknown.\n"
        "2. Ask for the minimum evidence needed to answer the question safely.\n"
        "3. Positive verdict requirements must be explicit in gates. Do not allow optimistic contracts.\n"
        "4. Retrieval plan must prefer source-native, jurisdiction-specific evidence.\n"
        "5. Retrieval plan doc_kinds must stay inside the deterministic block allowlist supplied in the prompt.\n"
        "6. Needed dossier sections must stay inside the deterministic block section list supplied in the prompt.\n"
        "7. Reuse dossier sections only as compressed memory, not as the primary evidence base.\n"
        "8. Return only the structured schema."
    )
    notes = _block_specific_policy_notes(block_spec)
    if notes:
        contract += "\nAdditional block policies:\n" + "\n".join(
            f"{idx + 1}. {note}" for idx, note in enumerate(notes)
        )
    return contract


def _block_specific_policy_notes(block_spec: BlockSpec) -> List[str]:
    notes: List[str] = []
    if block_spec.block_id == "asset_attractiveness":
        notes.append("Treat official RU/EAEU no-hit patent snapshots as residual-risk evidence, not as automatic negative evidence.")
    if block_spec.block_id == "rf_entry":
        notes.append("RF entry must stay anchored to RU registration identity plus RU-linked commercial/access evidence; unresolved EAEU details are adjacent, not automatic RF blockers.")
        notes.append("Do not require a separate RU policy act proving absence of restrictions when active GRLS identity and RU-linked access evidence are present and no source-backed RU registration/access block is surfaced; carry that as a caveat or follow-up, not as the reason to downgrade GO.")
        notes.append("If the packet explicitly carries a RU no-public-registration/no-record state, treat RF entry as NO_GO for the current snapshot rather than INSUFFICIENT_EVIDENCE.")
    if block_spec.block_id == "eaeu_entry":
        notes.append("Different RU and EAEU registration identifiers may represent separate product contexts; GRLS same-id corroboration is optional when EAEU-native identity, status, and validity are already confirmed.")
        notes.append("If a source-native product identity bridge or registry artifact explicitly says no public EAEU registration record is verified, treat entry as NO_GO for the current snapshot rather than unresolved HOLD.")
    if block_spec.block_id in {"generic_opportunity", "licensing_opportunity"}:
        notes.append("Reason region-by-region; do not collapse RU/EAEU opportunity with EU/US unresolved or blocked positions into one global unsupported verdict.")
    if block_spec.block_id == "market_reimbursement_window":
        notes.append("Do not require a single EAEU-union reimbursement list when source evidence establishes reimbursement/payer coverage is member-state scoped; decide RU on RU source-native evidence and carry non-RU EAEU member-state gaps as caveats/actions.")
    if block_spec.block_id == "evidence_sufficiency_note":
        notes.append("Distinguish screening-ready partial evidence from operations-ready sufficiency: IP/FTO and payer gaps can keep the package PARTIAL without collapsing it to pure INSUFFICIENT when source-enriched screening evidence is present.")
    if block_spec.block_id == "portfolio_opportunity":
        notes.append("Documented no-record states satisfy jurisdictional coverage for screening; they should lower the opportunity, not collapse US/EU/clinical/IP-supported portfolios to NOT_EVIDENCED.")
    if block_spec.block_id in {"asset_attractiveness", "rf_entry", "eaeu_entry", "generic_opportunity", "licensing_opportunity", "portfolio_opportunity"}:
        notes.append("Treat synthesis/manufacturing evidence as technical screening unless the question is explicitly CMC/manufacturing.")
    return notes


def _answer_prompt_contract(block_spec: BlockSpec, plan: ExecQuestionPlan) -> str:
    verdicts = ", ".join(block_spec.verdicts)
    positive_gates = ", ".join(plan.gates.positive_verdict_requires) or "none"
    contract = (
        f"You are answering the '{block_spec.title}' executive question.\n"
        f"Verdict family: {block_spec.verdict_family}. Allowed verdicts: {verdicts}.\n"
        f"Question contract answer type: {plan.answer_type}.\n"
        f"Positive verdict gates: {positive_gates}.\n"
        "Rules:\n"
        "1. Base reasoning on the evidence packet first; dossier snapshot is only supporting memory.\n"
        "2. Do not invent evidence. Use source-backed claims where possible.\n"
        "3. If a positive verdict gate is not satisfied, do not return a positive verdict.\n"
        "4. Unknowns, contradictions, and missing evidence classes must surface in blockers/caveats/actions.\n"
        "5. Translate evidence into decision language, not signal counting prose.\n"
        "6. Return only the structured schema."
    )
    notes = _block_specific_policy_notes(block_spec)
    if notes:
        contract += "\nAdditional block policies:\n" + "\n".join(
            f"{idx + 1}. {note}" for idx, note in enumerate(notes)
        )
    return contract


def _truncate_payload(value: Any, max_chars: int = 12000, compact: bool = False) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
    )
    if len(payload) <= max_chars:
        return payload
    return payload[: max_chars - 32] + "\n...TRUNCATED FOR PROMPT BOUNDING..."


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(256, int(raw))
    except ValueError:
        return default


def _answer_prompt_limits(block_spec: BlockSpec, phase: str) -> Dict[str, int]:
    contract_default = 3200
    snapshot_default = 5200 if block_spec.block_class == "critical" else 3200
    evidence_default = 14000 if block_spec.block_class == "critical" else 10000
    if phase == "final_answerer":
        snapshot_default += 1200
        evidence_default += 2000
    return {
        "contract": _int_env("DDKIT_EXEC_PROMPT_CONTRACT_MAX_CHARS", contract_default),
        "snapshot": _int_env("DDKIT_EXEC_PROMPT_DOSSIER_MAX_CHARS", snapshot_default),
        "evidence": _int_env("DDKIT_EXEC_PROMPT_EVIDENCE_MAX_CHARS", evidence_default),
    }


def build_block_prompt(
    block_spec: BlockSpec,
    packet: Dict[str, Any],
    profile: Optional[ModelProfile] = None,
    phase: str = "final",
) -> PromptPackage:
    active_profile = profile or get_model_profile()
    thinking_mode = block_thinking_mode(block_spec, active_profile, phase=phase)
    requested_model = block_requested_model(block_spec, active_profile, phase=phase)
    system_content = _prompt_contract(block_spec)
    human_content = (
        f"Block ID: {block_spec.block_id}\n"
        f"Phase: {phase}\n"
        "Produce a decision-grade structured answer from this bounded packet.\n"
        "Packet:\n"
        f"{_truncate_payload(packet)}"
    )
    return PromptPackage(
        block_spec=block_spec,
        requested_model=requested_model,
        thinking_mode=thinking_mode,
        max_output_tokens=_bounded_output_tokens("DDKIT_EXEC_BLOCK_MAX_OUTPUT_TOKENS", 3600),
        system_content=system_content,
        human_content=human_content,
        response_model=ExecReasonerOutput,
        phase=phase,
    )


def build_planner_prompt(
    block_spec: BlockSpec,
    dossier_snapshot: Dict[str, Any],
    corpus_inventory: Dict[str, Any],
    profile: Optional[ModelProfile] = None,
) -> PromptPackage:
    active_profile = profile or get_model_profile()
    question_trace = resolve_primary_question_trace(block_spec)
    system_content = _planner_prompt_contract(block_spec, question_trace)
    human_content = (
        f"Question ID: {block_spec.block_id}\n"
        f"Question title: {block_spec.title}\n"
        "Produce an execution contract for retrieval and answering.\n"
        "Deterministic block contract:\n"
        f"{_truncate_payload({'regions': block_spec.regions, 'required_sections': block_spec.sections, 'allowed_doc_kinds': block_spec.allowed_doc_kinds, 'verdict_family': block_spec.verdict_family}, max_chars=1200, compact=True)}\n"
        "Question trace:\n"
        f"{_truncate_payload(question_trace, max_chars=900, compact=True)}\n"
        "Compact dossier snapshot:\n"
        f"{_truncate_payload(dossier_snapshot, max_chars=2600, compact=True)}\n"
        "Corpus inventory:\n"
        f"{_truncate_payload(corpus_inventory, max_chars=1400, compact=True)}"
    )
    return PromptPackage(
        block_spec=block_spec,
        requested_model=planner_requested_model(active_profile),
        thinking_mode=planner_thinking_mode(active_profile),
        max_output_tokens=_bounded_output_tokens("DDKIT_EXEC_PLANNER_MAX_OUTPUT_TOKENS", 2400),
        system_content=system_content,
        human_content=human_content,
        response_model=ExecQuestionPlan,
        phase="planner",
    )


def build_answer_prompt(
    block_spec: BlockSpec,
    plan: ExecQuestionPlan,
    dossier_snapshot: Dict[str, Any],
    evidence_packet: Dict[str, Any],
    profile: Optional[ModelProfile] = None,
    phase: str = "answerer",
) -> PromptPackage:
    active_profile = profile or get_model_profile()
    system_content = _answer_prompt_contract(block_spec, plan)
    limits = _answer_prompt_limits(block_spec, phase)
    human_content = (
        f"Question ID: {block_spec.block_id}\n"
        f"Phase: {phase}\n"
        "Answer using the contract and the assembled evidence packet.\n"
        "Question contract:\n"
        f"{_truncate_payload(plan.model_dump(), max_chars=limits['contract'], compact=True)}\n"
        "Compact dossier snapshot:\n"
        f"{_truncate_payload(dossier_snapshot, max_chars=limits['snapshot'], compact=True)}\n"
        "Evidence packet:\n"
        f"{_truncate_payload(evidence_packet, max_chars=limits['evidence'], compact=True)}"
    )
    return PromptPackage(
        block_spec=block_spec,
        requested_model=answerer_requested_model(block_spec, active_profile),
        thinking_mode=answerer_thinking_mode(active_profile),
        max_output_tokens=_bounded_output_tokens("DDKIT_EXEC_ANSWERER_MAX_OUTPUT_TOKENS", 5200),
        system_content=system_content,
        human_content=human_content,
        response_model=ExecReasonerOutput,
        phase=phase,
    )


def resolve_primary_question_trace(block_spec: BlockSpec) -> Dict[str, Any]:
    library = load_exec_question_library()
    block_regions = {str(region or "").strip().upper() for region in block_spec.regions if str(region or "").strip()}
    block_sections = {str(section or "").strip() for section in block_spec.sections if str(section or "").strip()}
    best_match: Optional[Dict[str, Any]] = None
    best_score = -1
    for question_id in block_spec.appendix_question_ids:
        question = library.get(question_id)
        if not question:
            continue
        question_regions = {
            str(region or "").strip().upper()
            for region in question.get("required_jurisdictions", []) or []
            if str(region or "").strip()
        }
        if question_regions and block_regions and not question_regions.issubset(block_regions):
            continue
        question_sections = {
            str(section or "").strip()
            for section in question.get("required_sections", []) or []
            if str(section or "").strip()
        }
        overlap = len(question_sections.intersection(block_sections))
        if question_sections and overlap == 0:
            continue
        score = overlap + (4 if question_regions else 1)
        if score <= best_score:
            continue
        best_score = score
        best_match = {
            "question_id": question_id,
            "title": question.get("title", block_spec.title),
            "question_type": question.get("question_type", block_spec.verdict_family),
            "business_lens": question.get("business_lens", ""),
            "required_jurisdictions": question.get("required_jurisdictions", []),
            "required_sections": question.get("required_sections", []),
            "must_have_fields": question.get("must_have_fields", []),
            "fallback_policy": question.get("fallback_policy", ""),
        }
    if best_match:
        return best_match
    return {
        "question_id": block_spec.block_id,
        "title": block_spec.title,
        "question_type": block_spec.verdict_family,
        "business_lens": "",
        "required_jurisdictions": list(block_spec.regions),
        "required_sections": list(block_spec.sections),
        "must_have_fields": [],
        "fallback_policy": "",
    }


def build_appendix_question_traces(block_spec: BlockSpec) -> List[Dict[str, Any]]:
    library = load_exec_question_library()
    traces: List[Dict[str, Any]] = []
    for question_id in block_spec.appendix_question_ids:
        question = library.get(question_id, {})
        traces.append(
            {
                "question_id": question_id,
                "title": question.get("title", ""),
                "question_type": question.get("question_type", ""),
                "business_lens": question.get("business_lens", ""),
            }
        )
    return traces
