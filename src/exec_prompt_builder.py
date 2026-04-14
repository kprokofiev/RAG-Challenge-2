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
