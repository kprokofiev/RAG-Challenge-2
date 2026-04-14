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
        ExecNextAction,
        ExecRunManifest,
        ExecToplineSummary,
        ExecVerificationIssue,
        ExecVerificationReport,
        ModelBudgetTrace,
    )
    from src.exec_prompt_builder import (
        ExecReasonerOutput,
        build_appendix_question_traces,
        build_block_prompt,
        get_model_profile,
        load_exec_decision_library,
    )
    from src.exec_retrieval_escalation import ExecRetrievalEscalator
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
        ExecNextAction,
        ExecRunManifest,
        ExecToplineSummary,
        ExecVerificationIssue,
        ExecVerificationReport,
        ModelBudgetTrace,
    )
    from exec_prompt_builder import (  # type: ignore
        ExecReasonerOutput,
        build_appendix_question_traces,
        build_block_prompt,
        get_model_profile,
        load_exec_decision_library,
    )
    from exec_retrieval_escalation import ExecRetrievalEscalator  # type: ignore
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
        self.escalator = ExecRetrievalEscalator(retriever=retriever)
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

    def _invoke_reasoner(self, block_spec: Any, packet: Dict[str, Any], phase: str) -> Tuple[ExecReasonerOutput, Dict[str, Any], str]:
        prompt = build_block_prompt(block_spec, packet, self.model_profile, phase=phase)
        if not os.getenv("OPENAI_API_KEY"):
            return self._heuristic_reasoner(block_spec, packet, phase), {"mode": "heuristic"}, ""
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
            parsed = result.parsed_output
            if not isinstance(parsed, ExecReasonerOutput):
                parsed = ExecReasonerOutput.model_validate(parsed)
            return parsed, result.budget_trace, result.reasoning_summary or ""
        except Exception:
            return self._heuristic_reasoner(block_spec, packet, phase), {"mode": "heuristic_fallback"}, ""

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

    def _to_block(
        self,
        block_spec: Any,
        packet: Dict[str, Any],
        output: ExecReasonerOutput,
        budget_trace: Dict[str, Any],
        reasoning_summary: str,
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
                selected_evidence_ids=list(packet.get("evidence_ids", [])[:20]),
                missing_evidence_classes=list(output.missing_evidence_classes),
                escalation_performed=escalated,
                model_selected=str(budget_trace.get("model_selected") or budget_trace.get("mode") or ""),
                thinking_mode=str(budget_trace.get("thinking_mode_requested") or ""),
                reasoning_summary=reasoning_summary or None,
                budget_trace=ModelBudgetTrace.model_validate(budget_trace or {}),
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
            first_output, first_budget_trace, first_reasoning_summary = self._invoke_reasoner(block_spec, packet, phase="first_pass")
            gate = self._sufficiency_gate(first_output, packet)
            final_output = first_output
            final_budget_trace = first_budget_trace
            final_reasoning_summary = first_reasoning_summary
            escalated = False

            if gate.needs_escalation:
                escalation = self.escalator.escalate(
                    block_spec,
                    packet,
                    first_output.missing_evidence_classes or self._missing_evidence_classes(packet, block_spec),
                    case_id=case_id,
                )
                escalation_triggers.append(
                    {
                        "block_id": block_spec.block_id,
                        "reasons": gate.reasons,
                        "trace": escalation.trace,
                    }
                )
                if escalation.performed:
                    escalated = True
                    normalized_extra = []
                    synthetic_refs = []
                    for idx, item in enumerate(escalation.retrieved_items, start=1):
                        evidence_id = f"extra_{block_spec.block_id}_{idx}"
                        normalized = dict(item)
                        normalized["evidence_id"] = evidence_id
                        normalized_extra.append(normalized)
                        synthetic_refs.append(evidence_id)
                    packet["extra_retrieval"] = normalized_extra
                    packet["evidence_ids"] = list(dict.fromkeys(list(packet.get("evidence_ids", [])) + synthetic_refs))
                    packet["evidence_registry"] = list(packet.get("evidence_registry", [])) + [
                        {
                            "evidence_id": item["evidence_id"],
                            "doc_id": item.get("doc_id"),
                            "page": item.get("page"),
                            "snippet": item.get("snippet", ""),
                            "doc_kind": item.get("doc_kind"),
                            "source_url": item.get("source_url"),
                        }
                        for item in normalized_extra
                    ]
                    retrieved_extra_doc_ids.extend(escalation.retrieved_doc_ids)
                    final_output, final_budget_trace, final_reasoning_summary = self._invoke_reasoner(block_spec, packet, phase="final")

            block = self._to_block(
                block_spec,
                packet,
                final_output,
                final_budget_trace,
                final_reasoning_summary,
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
