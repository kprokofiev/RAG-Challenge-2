"""
Exec decision engine v1.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from src.dossier_schema_v3 import (
        DossierReport,
        ExecAppendix,
        ExecBlockTrace,
        ExecBlocker,
        ExecDecisionBlock,
        ExecDecisionReportV1,
        ExecEvidenceSufficiency,
        ExecModelStageTrace,
        ExecNextAction,
        ExecRunManifest,
        ExecToplineSummary,
        ExecVerificationIssue,
        ExecVerificationReport,
        ModelBudgetTrace,
    )
    from src.exec_evidence_assembler import ExecEvidenceAssembler
    from src.exec_llm_env import require_exec_openai_api_key
    from src.exec_prompt_builder import (
        ExecQuestionPlan,
        ExecReasonerOutput,
        build_answer_prompt,
        build_appendix_question_traces,
        build_planner_prompt,
        get_model_profile,
        load_exec_decision_library,
        resolve_primary_question_trace,
    )
    from src.exec_verifier import ExecVerifier
except ImportError:  # pragma: no cover
    from dossier_schema_v3 import (  # type: ignore
        DossierReport,
        ExecAppendix,
        ExecBlockTrace,
        ExecBlocker,
        ExecDecisionBlock,
        ExecDecisionReportV1,
        ExecEvidenceSufficiency,
        ExecModelStageTrace,
        ExecNextAction,
        ExecRunManifest,
        ExecToplineSummary,
        ExecVerificationIssue,
        ExecVerificationReport,
        ModelBudgetTrace,
    )
    from exec_evidence_assembler import ExecEvidenceAssembler  # type: ignore
    from exec_llm_env import require_exec_openai_api_key  # type: ignore
    from exec_prompt_builder import (  # type: ignore
        ExecQuestionPlan,
        ExecReasonerOutput,
        build_answer_prompt,
        build_appendix_question_traces,
        build_planner_prompt,
        get_model_profile,
        load_exec_decision_library,
        resolve_primary_question_trace,
    )
    from exec_verifier import ExecVerifier  # type: ignore


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _hash_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_region(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"RUSSIA", "RU"}:
        return "RU"
    if text in {"UNITED STATES", "USA", "US"}:
        return "US"
    if text in {"EMA", "EU"}:
        return "EU"
    return text


def _scalar_text(value: Any, limit: int = 80) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("value", "label", "title", "summary", "description", "message", "status"):
            nested = value.get(key)
            if nested is not None:
                text = _scalar_text(nested, limit=limit)
                if text:
                    return text
        return ""
    if isinstance(value, list):
        parts = [_scalar_text(item, limit=limit) for item in value[:3]]
        text = ", ".join(part for part in parts if part)
    else:
        text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _compact_unknowns(items: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, str]]:
    compacted: List[Dict[str, str]] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "field_path": _scalar_text(item.get("field_path"), limit=80),
                "reason_code": _scalar_text(item.get("reason_code"), limit=50),
                "message": _scalar_text(item.get("message"), limit=120),
            }
        )
    return compacted


def _sample_registrations(items: List[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    samples: List[Dict[str, str]] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        samples.append(
            {
                "region": _normalize_region(item.get("region")),
                "verdict": _scalar_text(item.get("verdict") or item.get("registration_verdict"), limit=32),
                "status": _scalar_text(item.get("status"), limit=80),
                "identifier": _scalar_text(item.get("identifiers"), limit=80),
            }
        )
    return {
        "count": len(items),
        "regions": sorted({_normalize_region(item.get("region")) for item in items if isinstance(item, dict)}),
        "confirmed_count": sum(1 for item in items if isinstance(item, dict) and _infer_registration_positive(item)),
        "samples": samples,
    }


def _sample_commercial_signals(items: List[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    samples: List[Dict[str, str]] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        samples.append(
            {
                "region": _normalize_region(item.get("region") or item.get("jurisdiction")),
                "category": _scalar_text(item.get("category"), limit=40),
                "verdict": _scalar_text(item.get("verdict"), limit=32),
                "summary": _scalar_text(item.get("summary"), limit=90),
            }
        )
    return {
        "count": len(items),
        "regions": sorted(
            {
                _normalize_region(item.get("region") or item.get("jurisdiction"))
                for item in items
                if isinstance(item, dict)
            }
        ),
        "samples": samples,
    }


def _sample_product_contexts(items: List[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    samples: List[Dict[str, str]] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        samples.append(
            {
                "region": _normalize_region(item.get("region")),
                "label": _scalar_text(item.get("label"), limit=60),
                "dosage_forms": _scalar_text(item.get("dosage_forms"), limit=60),
                "strengths": _scalar_text(item.get("strengths"), limit=60),
            }
        )
    return {
        "count": len(items),
        "regions": sorted({_normalize_region(item.get("region")) for item in items if isinstance(item, dict)}),
        "samples": samples,
    }


def _iter_evidence_refs(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        refs = value.get("evidence_refs", [])
        if isinstance(refs, list):
            for ref in refs:
                if str(ref or "").strip():
                    yield str(ref)
        for nested in value.values():
            yield from _iter_evidence_refs(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_evidence_refs(item)


def _infer_registration_positive(item: Dict[str, Any]) -> bool:
    verdict = str(item.get("verdict") or item.get("registration_verdict") or "").upper()
    if verdict in {"CONFIRMED", "PARTIAL", "REGISTERED"}:
        return True
    status = str(item.get("status", "")).lower()
    return any(token in status for token in ("approved", "registered", "active", "granted", "listed"))


def _looks_missing(section_value: Any) -> bool:
    if section_value is None:
        return True
    if isinstance(section_value, list):
        return len(section_value) == 0
    if isinstance(section_value, dict):
        return len(section_value) == 0
    return False


@dataclass
class SufficiencyGateResult:
    final_without_escalation: bool
    needs_escalation: bool
    reasons: List[str]


class ExecDecisionEngine:
    def __init__(self, retriever: Any = None):
        self.block_specs = load_exec_decision_library()
        self.model_profile = get_model_profile()
        self.assembler = ExecEvidenceAssembler(retriever=retriever)
        self.verifier = ExecVerifier()

    def _budget_snapshot(self) -> Dict[str, Any]:
        try:
            try:
                from src.openai_model_router import get_budget_snapshot
            except ImportError:  # pragma: no cover
                from openai_model_router import get_budget_snapshot  # type: ignore
            return get_budget_snapshot()
        except Exception:
            return {}

    def _partial_route_corroboration(self, dossier: Dict[str, Any]) -> bool:
        for unknown in dossier.get("unknowns", []) or []:
            if str(unknown.get("reason_code") or "").strip() == "PARTIAL_ROUTE_CORROBORATION":
                return True
        return False

    def _collect_section(self, dossier: Dict[str, Any], block_spec: Any, section: str) -> Any:
        data = dossier.get(section)
        if section == "registrations" and isinstance(data, list) and block_spec.regions:
            allowed = {_normalize_region(region) for region in block_spec.regions}
            return [item for item in data if _normalize_region(item.get("region")) in allowed]
        if section == "product_contexts" and isinstance(data, list) and block_spec.regions:
            allowed = {_normalize_region(region) for region in block_spec.regions}
            return [item for item in data if _normalize_region(item.get("region")) in allowed]
        if section == "commercial_signals" and isinstance(data, list) and block_spec.regions:
            allowed = {_normalize_region(region) for region in block_spec.regions}
            filtered = []
            for item in data:
                region = _normalize_region(item.get("region") or item.get("jurisdiction"))
                if not region or region in allowed:
                    filtered.append(item)
            return filtered
        if section == "unknowns" and isinstance(data, list):
            prefixes = tuple(block_spec.field_prefixes)
            if not prefixes:
                return data
            matched = []
            for item in data:
                field_path = str(item.get("field_path") or "")
                if field_path.startswith(prefixes):
                    matched.append(item)
            return matched
        return data

    def _build_packet(
        self,
        dossier: Dict[str, Any],
        case_id: Optional[str],
        block_spec: Any,
        prior_trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        packet: Dict[str, Any] = {
            "case_id": case_id or dossier.get("run_manifest", {}).get("case_id"),
            "inn": dossier.get("passport", {}).get("inn"),
            "block_id": block_spec.block_id,
            "title": block_spec.title,
            "block_class": block_spec.block_class,
            "verdict_family": block_spec.verdict_family,
            "allowed_doc_kinds": list(block_spec.allowed_doc_kinds),
            "budget_snapshot": self._budget_snapshot(),
            "prior_exec_trace": prior_trace or {},
            "partial_route_corroboration": self._partial_route_corroboration(dossier),
        }
        for section in block_spec.sections:
            packet[section] = self._collect_section(dossier, block_spec, section)

        selected_refs = set()
        for section in block_spec.sections:
            selected_refs.update(_iter_evidence_refs(packet.get(section)))
        evidence_registry = dossier.get("evidence_registry", []) or []
        filtered_evidence = []
        for item in evidence_registry:
            evidence_id = str(item.get("evidence_id") or "")
            doc_kind = str(item.get("doc_kind") or "")
            if evidence_id in selected_refs or doc_kind in block_spec.allowed_doc_kinds:
                filtered_evidence.append(item)
                if evidence_id:
                    selected_refs.add(evidence_id)
        packet["evidence_registry"] = filtered_evidence[:80]
        packet["evidence_ids"] = list(selected_refs)[:120]
        packet["critical_unknowns"] = (
            packet.get("dossier_quality_v2", {}) or {}
        ).get("critical_unknowns", [])
        return packet

    def _build_dossier_snapshot(self, dossier: Dict[str, Any], block_spec: Any, packet: Dict[str, Any]) -> Dict[str, Any]:
        registrations = packet.get("registrations", []) or []
        commercial = packet.get("commercial_signals", []) or []
        clinical = packet.get("clinical_studies", []) or []
        patents = packet.get("patent_families", []) or []
        synthesis = packet.get("synthesis_steps", []) or []
        product_contexts = packet.get("product_contexts", []) or []
        question_trace = resolve_primary_question_trace(block_spec)
        return {
            "case_id": packet.get("case_id"),
            "inn": packet.get("inn"),
            "block_id": block_spec.block_id,
            "title": block_spec.title,
            "verdict_family": block_spec.verdict_family,
            "question_trace": question_trace,
            "regions": list(block_spec.regions),
            "known_facts": {
                "registrations": _sample_registrations(registrations),
                "commercial_signals": _sample_commercial_signals(commercial),
                "product_contexts": _sample_product_contexts(product_contexts),
                "clinical_summary": {
                    "count": len(clinical),
                    "sample_titles": [
                        _scalar_text(item.get("title"), limit=80)
                        for item in clinical[:5]
                        if isinstance(item, dict)
                        and _scalar_text(item.get("title"), limit=80)
                    ],
                },
                "patent_summary": {
                    "count": len(patents),
                    "sample_statuses": [
                        _scalar_text(item.get("legal_status_snapshot"), limit=80)
                        for item in patents[:5]
                        if isinstance(item, dict)
                        and _scalar_text(item.get("legal_status_snapshot"), limit=80)
                    ],
                },
                "synthesis_summary": {
                    "count": len(synthesis),
                    "kinds": [
                        _scalar_text(item.get("kind"), limit=40)
                        for item in synthesis[:6]
                        if isinstance(item, dict)
                        and _scalar_text(item.get("kind"), limit=40)
                    ],
                },
            },
            "known_unknowns": _compact_unknowns(packet.get("unknowns") or []),
            "critical_unknowns": _compact_unknowns(packet.get("critical_unknowns") or []),
            "coverage": ((packet.get("dossier_quality_v2") or {}).get("coverage") or {}),
            "decision_readiness": ((packet.get("dossier_quality_v2") or {}).get("decision_readiness") or {}),
            "notes": [
                _scalar_text(item, limit=120)
                for item in ((packet.get("dossier_quality_v2") or {}).get("notes") or [])[:4]
                if _scalar_text(item, limit=120)
            ],
            "coverage_ledger_totals": ((packet.get("coverage_ledger") or {}).get("totals") or {}),
            "partial_route_corroboration": packet.get("partial_route_corroboration"),
        }

    def _build_corpus_inventory(self, dossier: Dict[str, Any], block_spec: Any, packet: Dict[str, Any]) -> Dict[str, Any]:
        evidence_registry = packet.get("evidence_registry", []) or []
        doc_kind_counts: Dict[str, int] = {}
        for item in evidence_registry:
            doc_kind = str(item.get("doc_kind") or "unknown")
            doc_kind_counts[doc_kind] = doc_kind_counts.get(doc_kind, 0) + 1
        regions = set()
        for section_name in ("registrations", "commercial_signals", "product_contexts"):
            for item in packet.get(section_name, []) or []:
                if isinstance(item, dict):
                    regions.add(_normalize_region(item.get("region") or item.get("jurisdiction") or item.get("country")))
        return {
            "available_sections": {
                section: (
                    len(packet.get(section, []) or [])
                    if isinstance(packet.get(section), list)
                    else bool(packet.get(section))
                )
                for section in block_spec.sections
            },
            "evidence_registry_count": len(evidence_registry),
            "available_doc_kinds": sorted(doc_kind_counts.keys()),
            "doc_kind_counts": dict(sorted(doc_kind_counts.items(), key=lambda item: (-item[1], item[0]))[:12]),
            "regions_with_data": sorted(region for region in regions if region),
            "retrieval_budget": {
                "max_docs": int(os.getenv("DDKIT_EXEC_PLAN_MAX_DOCS", "12")),
                "max_chunks": int(os.getenv("DDKIT_EXEC_PLAN_MAX_CHUNKS", "30")),
                "max_per_source_kind": int(os.getenv("DDKIT_EXEC_PLAN_MAX_PER_DOC_KIND", "6")),
            },
        }

    def _missing_evidence_classes(self, packet: Dict[str, Any], block_spec: Any) -> List[str]:
        missing: List[str] = []
        for section in ("registrations", "commercial_signals", "clinical_studies", "patent_families", "synthesis_steps"):
            if section in block_spec.sections and _looks_missing(packet.get(section)):
                for evidence_class in block_spec.escalation_classes:
                    if evidence_class not in missing:
                        missing.append(evidence_class)
                        break
        if packet.get("critical_unknowns"):
            for evidence_class in block_spec.escalation_classes:
                if evidence_class not in missing:
                    missing.append(evidence_class)
        return missing

    def _heuristic_reasoner(self, block_spec: Any, packet: Dict[str, Any], phase: str) -> ExecReasonerOutput:
        registrations = packet.get("registrations", []) or []
        commercial = packet.get("commercial_signals", []) or []
        clinical = packet.get("clinical_studies", []) or []
        patents = packet.get("patent_families", []) or []
        synthesis = packet.get("synthesis_steps", []) or []
        critical_unknowns = packet.get("critical_unknowns", []) or []
        confirmed_regions = sum(1 for item in registrations if isinstance(item, dict) and _infer_registration_positive(item))
        missing_classes = self._missing_evidence_classes(packet, block_spec)
        blockers: List[Dict[str, Any]] = []
        if critical_unknowns:
            for item in critical_unknowns[:2]:
                blockers.append(
                    {
                        "title": str(item.get("reason_code") or "critical_unknown"),
                        "severity": "MUST_VERIFY_NOW",
                        "rationale": str(item.get("impact") or ""),
                        "evidence_refs": [],
                    }
                )
        if block_spec.verdict_family == "go_no_go":
            if confirmed_regions or commercial:
                verdict = "CONDITIONAL_GO" if blockers else "GO"
                sufficiency = "PARTIAL" if blockers or missing_classes else "SUFFICIENT"
                confidence = "MEDIUM" if blockers or missing_classes else "HIGH"
            else:
                verdict = "INSUFFICIENT_EVIDENCE"
                sufficiency = "INSUFFICIENT"
                confidence = "LOW"
        elif block_spec.verdict_family == "opportunity":
            score = len(patents) + len(clinical) + len(commercial) + confirmed_regions
            if score >= 5 and not blockers:
                verdict, confidence, sufficiency = "HIGH", "HIGH", "SUFFICIENT"
            elif score >= 2:
                verdict, confidence, sufficiency = "MEDIUM", "MEDIUM", "PARTIAL"
            elif score >= 1:
                verdict, confidence, sufficiency = "LOW", "LOW", "PARTIAL"
            else:
                verdict, confidence, sufficiency = "NOT_EVIDENCED", "LOW", "INSUFFICIENT"
        elif block_spec.verdict_family == "window":
            if patents and not blockers:
                if missing_classes:
                    verdict, confidence, sufficiency = "OPEN", "MEDIUM", "PARTIAL"
                else:
                    verdict, confidence, sufficiency = "OPEN", "HIGH", "SUFFICIENT"
            elif patents:
                verdict, confidence, sufficiency = "LIMITED", "LOW", "PARTIAL"
            else:
                verdict, confidence, sufficiency = "UNRESOLVED", "LOW", "INSUFFICIENT"
        elif block_spec.verdict_family == "sufficiency":
            verdict = "INSUFFICIENT" if missing_classes else "SUFFICIENT"
            confidence = "LOW" if missing_classes else "MEDIUM"
            sufficiency = verdict
        else:
            verdict = "HIGH" if blockers else "LOW"
            confidence = "MEDIUM" if blockers else "LOW"
            sufficiency = "PARTIAL" if blockers else "SUFFICIENT"

        why_claims = [
            {
                "claim": f"Packet contains {confirmed_regions} positive registration signals, {len(commercial)} commercial signals, {len(patents)} patent families, {len(clinical)} clinical studies, and {len(synthesis)} synthesis steps.",
                "claim_type": "inference",
                "evidence_refs": packet.get("evidence_ids", [])[:5],
            }
        ]
        next_actions = []
        if missing_classes:
            next_actions.append(
                {
                    "action": f"Collect missing evidence classes for {block_spec.title.lower()}: {', '.join(missing_classes[:3])}.",
                    "priority": "NOW" if critical_unknowns else "NEXT",
                    "rationale": "Sufficiency gate flagged unresolved evidence gaps.",
                    "evidence_refs": [],
                }
            )
        full_answer = (
            f"{block_spec.title} is assessed as {verdict}. "
            f"The packet shows {confirmed_regions} positive registration signals and {len(commercial)} commercial signals. "
            f"Critical unknowns: {len(critical_unknowns)}."
        )
        caveats = []
        if packet.get("partial_route_corroboration"):
            caveats.append("Synthesis route corroboration remains partial.")
        return ExecReasonerOutput(
            verdict=verdict,
            confidence=confidence,
            sufficiency=sufficiency,
            short_answer=full_answer[:220],
            full_answer=full_answer,
            why_this_verdict=why_claims,
            decision_blockers=blockers,
            next_actions=next_actions,
            caveats=caveats,
            top_evidence_refs=packet.get("evidence_ids", [])[:8],
            missing_evidence_classes=missing_classes,
            key_risks=[item["title"] for item in blockers],
        )

    def _call_reasoning_prompt(
        self,
        prompt: Any,
        block_spec: Any,
        phase: str,
    ) -> Tuple[Any, Dict[str, Any], str]:
        require_exec_openai_api_key()
        try:
            try:
                from src.api_requests import call_exec_reasoning_model
            except ImportError:  # pragma: no cover
                from api_requests import call_exec_reasoning_model  # type: ignore
            result = call_exec_reasoning_model(
                system_content=prompt.system_content,
                human_content=prompt.human_content,
                response_format=prompt.response_model,
                requested_model=prompt.requested_model,
                thinking_mode=prompt.thinking_mode,
                max_output_tokens=prompt.max_output_tokens,
                metadata={"block_id": block_spec.block_id, "phase": phase},
                block_class=block_spec.block_class,
            )
            return result.parsed_output, result.budget_trace, result.reasoning_summary or ""
        except Exception as exc:
            raise RuntimeError(
                f"Exec reasoning failed for block '{block_spec.block_id}' during phase '{phase}': {exc}"
            ) from exc

    def _invoke_planner(
        self,
        block_spec: Any,
        dossier_snapshot: Dict[str, Any],
        corpus_inventory: Dict[str, Any],
    ) -> Tuple[ExecQuestionPlan, Dict[str, Any], str]:
        prompt = build_planner_prompt(
            block_spec=block_spec,
            dossier_snapshot=dossier_snapshot,
            corpus_inventory=corpus_inventory,
            profile=self.model_profile,
        )
        parsed, budget_trace, reasoning_summary = self._call_reasoning_prompt(prompt, block_spec, phase="planner")
        if not isinstance(parsed, ExecQuestionPlan):
            parsed = ExecQuestionPlan.model_validate(parsed)
        return parsed, budget_trace, reasoning_summary

    def _invoke_answerer(
        self,
        block_spec: Any,
        plan: ExecQuestionPlan,
        dossier_snapshot: Dict[str, Any],
        evidence_packet: Dict[str, Any],
        phase: str = "answerer",
    ) -> Tuple[ExecReasonerOutput, Dict[str, Any], str]:
        prompt = build_answer_prompt(
            block_spec=block_spec,
            plan=plan,
            dossier_snapshot=dossier_snapshot,
            evidence_packet=evidence_packet,
            profile=self.model_profile,
            phase=phase,
        )
        parsed, budget_trace, reasoning_summary = self._call_reasoning_prompt(prompt, block_spec, phase=phase)
        if not isinstance(parsed, ExecReasonerOutput):
            parsed = ExecReasonerOutput.model_validate(parsed)
        return parsed, budget_trace, reasoning_summary

    def _sufficiency_gate(self, output: ExecReasonerOutput, packet: Dict[str, Any]) -> SufficiencyGateResult:
        reasons: List[str] = []
        decision_blocking = any(item.severity in {"DECISION_BLOCKING", "MUST_VERIFY_NOW"} for item in output.decision_blockers)
        contradictions = bool(output.contradictions)
        unresolved_unknowns = bool(packet.get("critical_unknowns"))
        if output.sufficiency != "SUFFICIENT":
            reasons.append("non_sufficient_output")
        if decision_blocking:
            reasons.append("decision_blocking_blocker")
        if contradictions:
            reasons.append("unresolved_contradiction")
        if unresolved_unknowns:
            reasons.append("critical_unknowns_present")
        final_without_escalation = not reasons
        return SufficiencyGateResult(
            final_without_escalation=final_without_escalation,
            needs_escalation=not final_without_escalation and bool(output.missing_evidence_classes or self._missing_evidence_classes(packet, self.block_specs[packet["block_id"]])),
            reasons=reasons,
        )

    def _hydrate_answer_output(
        self,
        output: ExecReasonerOutput,
        evidence_packet: Dict[str, Any],
    ) -> ExecReasonerOutput:
        if not output.top_evidence_refs:
            output.top_evidence_refs = list(evidence_packet.get("selected_evidence_ids", [])[:8])
        if not output.missing_evidence_classes:
            output.missing_evidence_classes = list(evidence_packet.get("missing_evidence_classes", [])[:8])
        if not output.caveats and evidence_packet.get("contradictions"):
            output.caveats = [item.get("summary", "") for item in evidence_packet.get("contradictions", [])[:3] if item.get("summary")]
        return output

    def _to_block(
        self,
        block_spec: Any,
        packet: Dict[str, Any],
        plan: ExecQuestionPlan,
        evidence_packet: Dict[str, Any],
        output: ExecReasonerOutput,
        planner_trace: ExecModelStageTrace,
        answer_trace: ExecModelStageTrace,
        phase: str,
        escalated: bool,
    ) -> ExecDecisionBlock:
        unknowns = []
        for item in packet.get("unknowns", [])[:8]:
            field_path = item.get("field_path", "")
            message = item.get("message", "")
            unknowns.append(f"{field_path}: {message}".strip(": "))
        return ExecDecisionBlock(
            block_id=block_spec.block_id,
            title=block_spec.title,
            verdict=output.verdict,
            confidence=output.confidence,
            sufficiency=output.sufficiency,
            short_answer=output.short_answer,
            full_answer=output.full_answer,
            why_this_verdict=[item.model_dump() if hasattr(item, "model_dump") else item for item in output.why_this_verdict],
            contradictions=[item.model_dump() if hasattr(item, "model_dump") else item for item in output.contradictions],
            unknowns=unknowns,
            decision_blockers=[
                ExecBlocker(
                    blocker_id=f"{block_spec.block_id}_blocker_{idx + 1}",
                    title=item.title,
                    severity=item.severity,
                    rationale=item.rationale,
                    evidence_refs=item.evidence_refs,
                )
                for idx, item in enumerate(output.decision_blockers)
            ],
            next_actions=[
                ExecNextAction(
                    action_id=f"{block_spec.block_id}_action_{idx + 1}",
                    action=item.action,
                    priority=item.priority,
                    rationale=item.rationale,
                    evidence_refs=item.evidence_refs,
                )
                for idx, item in enumerate(output.next_actions)
            ],
            caveats=list(output.caveats),
            top_evidence_refs=list(output.top_evidence_refs),
            missing_evidence_classes=list(output.missing_evidence_classes),
            model_trace=ExecBlockTrace(
                stage=phase,
                input_packet_hash=_hash_payload(packet),
                selected_evidence_ids=list(evidence_packet.get("selected_evidence_ids", [])[:20]),
                missing_evidence_classes=list(output.missing_evidence_classes),
                escalation_performed=escalated,
                model_selected=answer_trace.model_selected,
                thinking_mode=answer_trace.thinking_mode,
                reasoning_summary=answer_trace.reasoning_summary,
                budget_trace=answer_trace.budget_trace,
                planner_trace=planner_trace,
                answer_trace=answer_trace,
                contract_summary={
                    "question_id": plan.question_id,
                    "answer_type": plan.answer_type,
                    "needed_facts": list(plan.needed_facts),
                    "doc_kinds": list(plan.retrieval_plan.doc_kinds),
                    "positive_verdict_requires": list(plan.gates.positive_verdict_requires),
                },
                evidence_packet_summary=dict(evidence_packet.get("evidence_packet_summary", {})),
            ),
        )

    def _aggregate_verification(self, blocks: List[ExecDecisionBlock]) -> ExecVerificationReport:
        issues: List[ExecVerificationIssue] = []
        statuses = []
        for block in blocks:
            if not block.verification:
                continue
            statuses.append(block.verification.overall_status)
            for issue in block.verification.issues[:5]:
                issues.append(issue)
        overall = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
        return ExecVerificationReport(
            overall_status=overall,
            factual_status=overall,
            decision_status=overall,
            reviewer_status=overall,
            issues=issues[:20],
        )

    def generate(self, dossier: Dict[str, Any], case_id: Optional[str] = None) -> ExecDecisionReportV1:
        validated = DossierReport.model_validate(dossier)
        payload = validated.model_dump()
        budget_snapshot = self._budget_snapshot()
        blocks: List[ExecDecisionBlock] = []
        appendix_question_traces: List[Dict[str, Any]] = []
        escalation_triggers: List[Dict[str, Any]] = []
        retrieved_extra_doc_ids: List[str] = []

        for block_spec in self.block_specs.values():
            packet = self._build_packet(payload, case_id, block_spec)
            dossier_snapshot = self._build_dossier_snapshot(payload, block_spec, packet)
            corpus_inventory = self._build_corpus_inventory(payload, block_spec, packet)
            plan, planner_budget_trace, planner_reasoning_summary = self._invoke_planner(
                block_spec,
                dossier_snapshot,
                corpus_inventory,
            )

            evidence_packet = self.assembler.assemble(
                base_packet=packet,
                plan=plan,
                case_id=case_id,
                allow_retrieval=False,
            )
            answer_output, answer_budget_trace, answer_reasoning_summary = self._invoke_answerer(
                block_spec,
                plan,
                dossier_snapshot,
                evidence_packet,
                phase="answerer",
            )
            answer_output = self._hydrate_answer_output(answer_output, evidence_packet)
            gate_packet = dict(packet)
            gate_packet["critical_unknowns"] = list(evidence_packet.get("critical_unknowns", packet.get("critical_unknowns", [])))
            gate = self._sufficiency_gate(answer_output, gate_packet)
            final_output = answer_output
            final_answer_budget_trace = answer_budget_trace
            final_answer_reasoning_summary = answer_reasoning_summary
            final_evidence_packet = evidence_packet
            escalated = False

            if gate.needs_escalation:
                escalated_packet = self.assembler.assemble(
                    base_packet=packet,
                    plan=plan,
                    case_id=case_id,
                    allow_retrieval=True,
                )
                escalation_triggers.append(
                    {
                        "block_id": block_spec.block_id,
                        "reasons": gate.reasons,
                        "contract": {
                            "needed_facts": list(plan.needed_facts),
                            "doc_kinds": list(plan.retrieval_plan.doc_kinds),
                            "queries": list(plan.retrieval_plan.queries),
                        },
                        "trace": {
                            "retrieved_extra_count": (escalated_packet.get("evidence_packet_summary", {}) or {}).get("retrieved_extra_count", 0),
                            "missing_evidence_classes": list(escalated_packet.get("missing_evidence_classes", [])),
                        },
                    }
                )
                if (escalated_packet.get("evidence_packet_summary", {}) or {}).get("retrieved_extra_count", 0):
                    escalated = True
                    final_evidence_packet = escalated_packet
                    retrieved_extra_doc_ids.extend(
                        [
                            str(item.get("doc_id") or "")
                            for item in escalated_packet.get("selected_evidence", [])
                            if item.get("doc_id")
                        ]
                    )
                    final_output, final_answer_budget_trace, final_answer_reasoning_summary = self._invoke_answerer(
                        block_spec,
                        plan,
                        dossier_snapshot,
                        final_evidence_packet,
                        phase="final_answerer",
                    )
                    final_output = self._hydrate_answer_output(final_output, final_evidence_packet)

            planner_trace = ExecModelStageTrace(
                model_selected=str(planner_budget_trace.get("model_selected") or ""),
                thinking_mode=str(planner_budget_trace.get("thinking_mode_requested") or ""),
                reasoning_summary=planner_reasoning_summary or None,
                budget_trace=ModelBudgetTrace.model_validate(planner_budget_trace or {}),
            )
            answer_trace = ExecModelStageTrace(
                model_selected=str(final_answer_budget_trace.get("model_selected") or ""),
                thinking_mode=str(final_answer_budget_trace.get("thinking_mode_requested") or ""),
                reasoning_summary=final_answer_reasoning_summary or None,
                budget_trace=ModelBudgetTrace.model_validate(final_answer_budget_trace or {}),
            )

            block = self._to_block(
                block_spec,
                packet,
                plan,
                final_evidence_packet,
                final_output,
                planner_trace,
                answer_trace,
                phase="final" if escalated else "single_pass",
                escalated=escalated,
            )
            allow_repair = _env_bool("DDKIT_EXEC_REPAIR_ENABLED", True)
            block, verification = self.verifier.verify_and_repair(block, packet, block_spec, allow_repair=allow_repair)
            block.verification = verification
            if block.model_trace:
                block.model_trace.verifier_verdict = verification.overall_status
            blocks.append(block)
            appendix_question_traces.extend(build_appendix_question_traces(block_spec))

        topline: Dict[str, ExecToplineSummary] = {}
        key_risks: List[str] = []
        all_blockers: List[ExecBlocker] = []
        next_actions: List[ExecNextAction] = []
        sufficiency_by_block = {}
        for block in blocks:
            sufficiency_by_block[block.block_id] = block.sufficiency
            if block.block_id in {"asset_attractiveness", "rf_entry", "eaeu_entry", "generic_opportunity", "licensing_opportunity", "portfolio_opportunity"}:
                topline[block.block_id] = ExecToplineSummary(
                    verdict=block.verdict,
                    confidence=block.confidence,
                    sufficiency=block.sufficiency,
                    short_answer=block.short_answer,
                )
            for caveat in block.caveats:
                if caveat not in key_risks:
                    key_risks.append(caveat)
            for blocker in block.decision_blockers:
                if blocker.title not in {item.title for item in all_blockers}:
                    all_blockers.append(blocker)
            for action in block.next_actions:
                if action.action not in {item.action for item in next_actions}:
                    next_actions.append(action)

        overall_sufficiency = "INSUFFICIENT" if any(value == "INSUFFICIENT" for value in sufficiency_by_block.values()) else "PARTIAL" if any(value == "PARTIAL" for value in sufficiency_by_block.values()) else "SUFFICIENT"
        overall_confidence = "LOW" if overall_sufficiency != "SUFFICIENT" else "MEDIUM"
        verification = self._aggregate_verification(blocks)

        run_manifest = ExecRunManifest(
            run_id=payload.get("run_manifest", {}).get("run_id") or f"exec-{int(time.time())}",
            case_id=case_id or payload.get("run_manifest", {}).get("case_id"),
            dossier_hash=_hash_payload(payload),
            exec_config={
                "engine_enabled": _env_bool("DDKIT_EXEC_ENGINE_ENABLED", True),
                "engine_version": os.getenv("DDKIT_EXEC_ENGINE_VERSION", "v1"),
                "output_mode": os.getenv("DDKIT_EXEC_OUTPUT_MODE", "both"),
                "question_first_pipeline": True,
                "planner_reasoning_mode": self.model_profile.thinking_defaults.get("planner", "high"),
                "answerer_reasoning_mode": self.model_profile.thinking_defaults.get("answerer", "high"),
            },
            model_profile=self.model_profile.model_dump(),
            budget_snapshot=budget_snapshot,
            decision_blocks=[block.block_id for block in blocks],
            escalation_triggers=escalation_triggers,
            retrieved_extra_doc_ids=list(dict.fromkeys(retrieved_extra_doc_ids)),
            verifier_results=[block.verification.model_dump() for block in blocks if block.verification],
            repair_pass_results=[
                {
                    "block_id": block.block_id,
                    "repair_applied": bool(block.verification and block.verification.repair_applied),
                    "repair_reason": block.verification.repair_reason if block.verification else None,
                }
                for block in blocks
            ],
            final_verdict_summary={key: value.model_dump() for key, value in topline.items()},
            block_traces=[block.model_trace.model_dump() for block in blocks if block.model_trace],
        )

        source_snapshot = {
            "evidence_count": len(payload.get("evidence_registry", []) or []),
            "registration_count": len(payload.get("registrations", []) or []),
            "clinical_count": len(payload.get("clinical_studies", []) or []),
            "patent_family_count": len(payload.get("patent_families", []) or []),
            "commercial_signal_count": len(payload.get("commercial_signals", []) or []),
        }
        appendix = ExecAppendix(
            source_snapshot=source_snapshot,
            budget_snapshot=budget_snapshot,
            question_traces=list({json.dumps(item, sort_keys=True): item for item in appendix_question_traces}.values()),
        )
        return ExecDecisionReportV1(
            report_version="v1",
            case_id=case_id or payload.get("run_manifest", {}).get("case_id"),
            inn=payload.get("passport", {}).get("inn"),
            generated_at=_now_iso(),
            engine_manifest=run_manifest,
            topline=topline,
            decision_blocks=blocks,
            key_risks=key_risks[:20],
            decision_blockers=all_blockers[:12],
            recommended_next_actions=next_actions[:12],
            evidence_sufficiency=ExecEvidenceSufficiency(
                overall_verdict=overall_sufficiency,
                topline_confidence=overall_confidence,
                notes=(payload.get("dossier_quality_v2", {}) or {}).get("notes", [])[:12],
                by_block=sufficiency_by_block,
            ),
            verification=verification,
            appendix=appendix,
        )
