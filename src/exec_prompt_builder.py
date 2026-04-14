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


class ExecRetrievalPlan(BaseModel):
    doc_kinds: List[str] = Field(default_factory=list)
    queries: List[str] = Field(default_factory=list)
    max_docs: int = 12
    max_chunks: int = 30
    chunk_policy: str = "prefer source-native confirmation chunks"
    max_per_doc_kind: Dict[str, int] = Field(default_factory=dict)


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
    max_output_tokens: int = 2400
    system_content: str
    human_content: str
    response_model: Type[BaseModel]
    phase: str = "final"


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
    return (
        f"You are planning evidence requirements for the '{block_spec.title}' executive question.\n"
        f"Verdict family: {block_spec.verdict_family}. Allowed verdicts later: {verdicts}.\n"
        f"Business lens: {question_trace.get('business_lens', '') or 'exec'}.\n"
        "Return a strict data contract, not prose.\n"
        "Rules:\n"
        "1. Use compact dossier state only to identify what is already known vs unknown.\n"
        "2. Ask for the minimum evidence needed to answer the question safely.\n"
        "3. Positive verdict requirements must be explicit in gates. Do not allow optimistic contracts.\n"
        "4. Retrieval plan must prefer source-native, jurisdiction-specific evidence.\n"
        "5. Reuse dossier sections only as compressed memory, not as the primary evidence base.\n"
        "6. Return only the structured schema."
    )


def _answer_prompt_contract(block_spec: BlockSpec, plan: ExecQuestionPlan) -> str:
    verdicts = ", ".join(block_spec.verdicts)
    positive_gates = ", ".join(plan.gates.positive_verdict_requires) or "none"
    return (
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


def _truncate_payload(value: Any, max_chars: int = 12000) -> str:
    payload = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    if len(payload) <= max_chars:
        return payload
    return payload[: max_chars - 32] + "\n...TRUNCATED FOR PROMPT BOUNDING..."


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
        "Question trace:\n"
        f"{_truncate_payload(question_trace, max_chars=3000)}\n"
        "Compact dossier snapshot:\n"
        f"{_truncate_payload(dossier_snapshot, max_chars=6000)}\n"
        "Corpus inventory:\n"
        f"{_truncate_payload(corpus_inventory, max_chars=4000)}"
    )
    return PromptPackage(
        block_spec=block_spec,
        requested_model=planner_requested_model(active_profile),
        thinking_mode=planner_thinking_mode(active_profile),
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
    human_content = (
        f"Question ID: {block_spec.block_id}\n"
        f"Phase: {phase}\n"
        "Answer using the contract and the assembled evidence packet.\n"
        "Question contract:\n"
        f"{_truncate_payload(plan.model_dump(), max_chars=5000)}\n"
        "Compact dossier snapshot:\n"
        f"{_truncate_payload(dossier_snapshot, max_chars=4000)}\n"
        "Evidence packet:\n"
        f"{_truncate_payload(evidence_packet, max_chars=12000)}"
    )
    return PromptPackage(
        block_spec=block_spec,
        requested_model=answerer_requested_model(block_spec, active_profile),
        thinking_mode=answerer_thinking_mode(active_profile),
        system_content=system_content,
        human_content=human_content,
        response_model=ExecReasonerOutput,
        phase=phase,
    )


def resolve_primary_question_trace(block_spec: BlockSpec) -> Dict[str, Any]:
    library = load_exec_question_library()
    for question_id in block_spec.appendix_question_ids:
        question = library.get(question_id)
        if question:
            return {
                "question_id": question_id,
                "title": question.get("title", block_spec.title),
                "question_type": question.get("question_type", block_spec.verdict_family),
                "business_lens": question.get("business_lens", ""),
                "required_jurisdictions": question.get("required_jurisdictions", []),
                "required_sections": question.get("required_sections", []),
                "must_have_fields": question.get("must_have_fields", []),
                "fallback_policy": question.get("fallback_policy", ""),
            }
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
