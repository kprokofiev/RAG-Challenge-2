import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dossier_schema_v3 import ExecDecisionBlock, ExecWhyClaim
from src.exec_answer_runner import _build_exec_retriever
from src.exec_decision_engine import ExecDecisionEngine
from src.exec_evidence_assembler import ExecEvidenceAssembler
from src.exec_llm_env import require_exec_openai_api_key
from src.exec_prompt_builder import (
    ExecDocKindLimit,
    ExecQuestionPlan,
    ExecReasonerOutput,
    ExecRetrievalPlan,
    build_answer_prompt,
    build_planner_prompt,
    resolve_primary_question_trace,
)
from src.exec_retrieval_escalation import ExecRetrievalEscalator
from src.exec_verifier import ExecVerifier
from src.render_exec_decision_report import HAS_REPORTLAB, render_exec_decision_report


def _sample_dossier():
    return {
        "schema_version": "3.0",
        "report_id": "rep-1",
        "case_id": "case-1",
        "run_id": "run-1",
        "generated_at": "2026-04-14T00:00:00Z",
        "passport": {"inn": "Apixaban"},
        "product_contexts": [
            {
                "context_id": "ctx-ru",
                "label": "RU oral",
                "region": "RU",
                "dosage_forms": ["tablet"],
                "strengths": ["5 mg"],
                "evidence_refs": ["ev-reg-ru"],
            }
        ],
        "registrations": [
            {
                "region": "RU",
                "verdict": "confirmed",
                "status": {"value": "registered", "evidence_refs": ["ev-reg-ru"]},
                "identifiers": [{"value": "LP-001", "evidence_refs": ["ev-reg-ru"]}],
                "evidence_refs": ["ev-reg-ru"],
            },
            {
                "region": "EU",
                "verdict": "partial",
                "status": {"value": "approved", "evidence_refs": ["ev-reg-eu"]},
                "identifiers": [{"value": "EU-123", "evidence_refs": ["ev-reg-eu"]}],
                "evidence_refs": ["ev-reg-eu"],
            },
        ],
        "clinical_studies": [
            {
                "title": {"value": "ARISTOTLE", "evidence_refs": ["ev-clin-1"]},
                "phase": {"value": "Phase 3", "evidence_refs": ["ev-clin-1"]},
                "status": {"value": "Completed", "evidence_refs": ["ev-clin-1"]},
                "evidence_refs": ["ev-clin-1"],
            }
        ],
        "patent_families": [
            {
                "family_id": "fam-1",
                "representative_pub": {"value": "US123", "evidence_refs": ["ev-pat-1"]},
                "legal_status_snapshot": {"value": "expired", "evidence_refs": ["ev-pat-1"]},
                "expiry_by_country": [{"value": "US:2025-01-01", "evidence_refs": ["ev-pat-1"]}],
                "evidence_refs": ["ev-pat-1"],
            }
        ],
        "synthesis_steps": [
            {
                "step_number": 1,
                "kind": "api_synthesis",
                "description": {"value": "API intermediate coupling", "evidence_refs": ["ev-syn-1"]},
                "reagents": [{"value": "Intermediate A", "evidence_refs": ["ev-syn-1"]}],
                "evidence_refs": ["ev-syn-1"],
            }
        ],
        "commercial_signals": [
            {
                "signal_id": "sig-1",
                "region": "RU",
                "category": "procurement",
                "verdict": "confirmed",
                "summary": {"value": "Observed procurement signal", "evidence_refs": ["ev-com-1"]},
                "source_name": "ru_registration_export",
                "source_tier": "primary",
                "source_priority": 100,
                "evidence_refs": ["ev-com-1"],
            }
        ],
        "unknowns": [
            {
                "field_path": "synthesis_steps[0].description",
                "reason_code": "PARTIAL_ROUTE_CORROBORATION",
                "message": "Route only partially corroborated.",
            }
        ],
        "evidence_registry": [
            {"evidence_id": "ev-reg-ru", "doc_id": "doc-ru", "page": 1, "snippet": "RU registered", "doc_kind": "ru_registration_export"},
            {"evidence_id": "ev-reg-eu", "doc_id": "doc-eu", "page": 2, "snippet": "EU approved", "doc_kind": "smpc"},
            {"evidence_id": "ev-clin-1", "doc_id": "doc-clin", "page": 3, "snippet": "Phase 3 trial", "doc_kind": "ctgov"},
            {"evidence_id": "ev-pat-1", "doc_id": "doc-pat", "page": 4, "snippet": "Patent expired", "doc_kind": "patent_legal_events"},
            {"evidence_id": "ev-syn-1", "doc_id": "doc-syn", "page": 5, "snippet": "Synthesis step", "doc_kind": "patent_family_summary"},
            {"evidence_id": "ev-com-1", "doc_id": "doc-com", "page": 6, "snippet": "Procurement snapshot", "doc_kind": "ru_procurement_snapshot"},
        ],
        "dossier_quality_v2": {
            "coverage": {"registrations": 0.8, "patents": 0.7},
            "decision_readiness": {"registrations": "GREEN", "patents_legal": "YELLOW"},
            "critical_unknowns": [
                {
                    "reason_code": "PARTIAL_ROUTE_CORROBORATION",
                    "count": 1,
                    "impact": "Route corroboration is partial",
                }
            ],
            "notes": ["Route support is partial."],
        },
        "coverage_ledger": {"registrations": {"decision_readiness": "ready"}},
        "run_manifest": {"run_id": "run-1", "report_id": "rep-1", "case_id": "case-1"},
    }


def _stub_reasoner_output():
    return ExecReasonerOutput(
        verdict="HOLD",
        confidence="LOW",
        sufficiency="PARTIAL",
        short_answer="Need LLM-backed review before decision.",
        full_answer="Need LLM-backed review before decision.",
        caveats=["Synthesis route corroboration remains partial."],
        missing_evidence_classes=[],
    )


def _stub_question_plan():
    return ExecQuestionPlan(
        question_id="rf_entry",
        answer_type="geo_decision",
        business_lens="regulatory",
        needed_facts=["direct_ru_registration_confirmation"],
        needed_dossier_sections=["registrations", "unknowns", "dossier_quality_v2"],
        retrieval_plan={
            "doc_kinds": ["ru_registration_export", "grls_card"],
            "queries": ["apixaban RU registration"],
            "max_docs": 8,
            "max_chunks": 12,
            "chunk_policy": "prefer source-native confirmation chunks",
        },
        answer_schema={
            "verdict": ["GO", "HOLD", "NO_GO", "INSUFFICIENT_EVIDENCE"],
            "must_include": ["what_is_confirmed", "critical_gap", "next_action"],
        },
        gates={"positive_verdict_requires": ["direct_ru_registration_confirmation"]},
    )


class ExecDecisionEngineTests(unittest.TestCase):
    def test_packet_builder_filters_sections_and_regions(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["rf_entry"]
        packet = engine._build_packet(_sample_dossier(), "case-1", block_spec)
        self.assertIn("registrations", packet)
        self.assertNotIn("clinical_studies", packet)
        self.assertEqual([item["region"] for item in packet["registrations"]], ["RU"])
        self.assertTrue(packet["critical_unknowns"])

    def test_sufficiency_gate_flags_unknowns(self):
        engine = ExecDecisionEngine()
        output = ExecReasonerOutput(
            verdict="GO",
            confidence="HIGH",
            sufficiency="SUFFICIENT",
            short_answer="go",
            full_answer="go",
            missing_evidence_classes=["ru_registration_confirmation"],
        )
        packet = {"block_id": "rf_entry", "critical_unknowns": [{"reason_code": "X"}]}
        gate = engine._sufficiency_gate(output, packet)
        self.assertFalse(gate.final_without_escalation)
        self.assertTrue(gate.needs_escalation)

    def test_generate_report_surfaces_partial_route_caveat(self):
        engine = ExecDecisionEngine()
        with patch.object(
            engine,
            "_invoke_planner",
            return_value=(_stub_question_plan(), {"model_selected": "gpt-5.4-mini", "thinking_mode_requested": "high"}, "planner summary"),
        ), patch.object(
            engine,
            "_invoke_answerer",
            return_value=(_stub_reasoner_output(), {"model_selected": "gpt-5.4-mini", "thinking_mode_requested": "high"}, "answerer summary"),
        ):
            report = engine.generate(_sample_dossier(), case_id="case-1")
        self.assertEqual(report.report_version, "v1")
        self.assertTrue(any("partial" in caveat.lower() for block in report.decision_blocks for caveat in block.caveats))
        self.assertTrue(
            all(
                block.model_trace.answer_trace.thinking_mode == "high"
                for block in report.decision_blocks
                if block.model_trace and block.model_trace.answer_trace
            )
        )

    @unittest.skipUnless(HAS_REPORTLAB, "reportlab is required")
    def test_render_internal_and_customer_pdf(self):
        engine = ExecDecisionEngine()
        with patch.object(
            engine,
            "_invoke_planner",
            return_value=(_stub_question_plan(), {"model_selected": "gpt-5.4-mini", "thinking_mode_requested": "high"}, "planner summary"),
        ), patch.object(
            engine,
            "_invoke_answerer",
            return_value=(_stub_reasoner_output(), {"model_selected": "gpt-5.4-mini", "thinking_mode_requested": "high"}, "answerer summary"),
        ):
            report = engine.generate(_sample_dossier(), case_id="case-1")
        with tempfile.TemporaryDirectory() as td:
            customer = Path(td) / "customer.pdf"
            internal = Path(td) / "internal.pdf"
            render_exec_decision_report(report, str(customer), mode="customer")
            render_exec_decision_report(report, str(internal), mode="internal")
            self.assertTrue(customer.exists())
            self.assertTrue(internal.exists())
            self.assertGreater(customer.stat().st_size, 0)
            self.assertGreater(internal.stat().st_size, 0)

    def test_invoke_reasoner_requires_exec_llm_key(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["rf_entry"]
        snapshot = engine._build_dossier_snapshot(_sample_dossier(), block_spec, engine._build_packet(_sample_dossier(), "case-1", block_spec))
        inventory = engine._build_corpus_inventory(_sample_dossier(), block_spec, engine._build_packet(_sample_dossier(), "case-1", block_spec))
        missing_env_path = str(Path(tempfile.gettempdir()) / "missing_exec_llm.env")
        with patch.dict(os.environ, {"DDKIT_EXEC_OPENAI_ENV_FILE": missing_env_path}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                engine._invoke_planner(block_spec, snapshot, inventory)
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_planner_and_answerer_prompts_default_to_high_reasoning(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["rf_entry"]
        packet = engine._build_packet(_sample_dossier(), "case-1", block_spec)
        snapshot = engine._build_dossier_snapshot(_sample_dossier(), block_spec, packet)
        inventory = engine._build_corpus_inventory(_sample_dossier(), block_spec, packet)
        planner_prompt = build_planner_prompt(block_spec, snapshot, inventory, engine.model_profile)
        answer_prompt = build_answer_prompt(block_spec, _stub_question_plan(), snapshot, {"selected_evidence": []}, engine.model_profile)
        self.assertEqual(planner_prompt.thinking_mode, "high")
        self.assertEqual(answer_prompt.thinking_mode, "high")
        self.assertIsNone(planner_prompt.max_output_tokens)
        self.assertIsNone(answer_prompt.max_output_tokens)

    def test_dossier_snapshot_is_compact_for_planner_stage(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["rf_entry"]
        packet = engine._build_packet(_sample_dossier(), "case-1", block_spec)
        snapshot = engine._build_dossier_snapshot(_sample_dossier(), block_spec, packet)
        self.assertEqual(snapshot["known_facts"]["registrations"]["count"], 1)
        self.assertIn("samples", snapshot["known_facts"]["registrations"])
        self.assertNotIsInstance(snapshot["known_facts"]["registrations"], list)
        self.assertIn("reason_code", snapshot["critical_unknowns"][0])

    def test_rf_entry_trace_prefers_ru_specific_contract(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["rf_entry"]
        trace = resolve_primary_question_trace(block_spec)
        self.assertEqual(trace["question_id"], "rf_entry")
        self.assertEqual(trace["required_jurisdictions"], ["RU"])

    def test_normalize_plan_restores_block_sections_and_doc_kinds(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["rf_entry"]
        plan = ExecQuestionPlan(
            question_id="wrong_question",
            answer_type="",
            needed_dossier_sections=["registrations"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["fda_drugs_at_fda", "ema_epar", "eaeu_register"],
                queries=["apixaban entry readiness"],
                doc_kind_limits=[
                    ExecDocKindLimit(doc_kind="fda_drugs_at_fda", max_chunks=2),
                    ExecDocKindLimit(doc_kind="ema_epar", max_chunks=2),
                ],
            ),
        )
        normalized = engine._normalize_plan(block_spec, plan)
        self.assertEqual(normalized.question_id, "rf_entry")
        self.assertEqual(normalized.answer_type, block_spec.verdict_family)
        self.assertEqual(normalized.needed_dossier_sections, list(block_spec.sections))
        self.assertEqual(normalized.retrieval_plan.doc_kinds, list(block_spec.allowed_doc_kinds))
        self.assertEqual(normalized.retrieval_plan.doc_kind_limits, [])

    def test_asset_packet_keeps_required_sections_when_planner_omits_them(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["asset_attractiveness"]
        packet = engine._build_packet(_sample_dossier(), "case-1", block_spec)
        plan = ExecQuestionPlan(
            question_id="asset_attractiveness",
            answer_type="opportunity",
            needed_dossier_sections=["registrations"],
            retrieval_plan=ExecRetrievalPlan(doc_kinds=["label"], queries=["apixaban asset"]),
        )
        normalized = engine._normalize_plan(block_spec, plan)
        evidence_packet = engine.assembler.assemble(packet, normalized, case_id="case-1", allow_retrieval=False)
        selected_sections = evidence_packet["selected_sections"]
        self.assertIn("clinical_studies", selected_sections)
        self.assertIn("patent_families", selected_sections)
        self.assertIn("unknowns", selected_sections)
        self.assertEqual(len(selected_sections["clinical_studies"]), 1)
        self.assertEqual(len(selected_sections["patent_families"]), 1)

    def test_rf_entry_alias_doc_kinds_select_existing_ru_evidence(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["rf_entry"]
        packet = engine._build_packet(_sample_dossier(), "case-1", block_spec)
        plan = ExecQuestionPlan(
            question_id="rf_entry",
            answer_type="go_no_go",
            needed_dossier_sections=["registrations"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["fda_drugs_at_fda", "ema_epar", "eaeu_register"],
                queries=["apixaban russian federation registration"],
            ),
        )
        normalized = engine._normalize_plan(block_spec, plan)
        evidence_packet = engine.assembler.assemble(packet, normalized, case_id="case-1", allow_retrieval=False)
        self.assertTrue(evidence_packet["selected_evidence"])
        self.assertTrue(
            any(item["doc_kind"] in {"grls", "grls_card", "ru_instruction", "ru_registration_export"} for item in evidence_packet["selected_evidence"])
        )


class ExecVerifierTests(unittest.TestCase):
    def test_verifier_catches_unsupported_claim_and_repairs(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="rf_entry",
            title="RF entry",
            verdict="GO",
            confidence="HIGH",
            sufficiency="SUFFICIENT",
            short_answer="Go",
            full_answer="Go",
            why_this_verdict=[ExecWhyClaim(claim="Hard claim without refs", claim_type="hard_evidence_backed", evidence_refs=[])],
            decision_blockers=[],
            next_actions=[],
        )
        packet = {"evidence_ids": ["ev-1"], "critical_unknowns": []}
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.why_this_verdict[0].claim_type, "inference")


class ExecRetrievalEscalationTests(unittest.TestCase):
    def test_commercial_priority_prefers_source_native(self):
        escalator = ExecRetrievalEscalator(retriever=None)
        items = [
            {"doc_id": "doc-summary", "doc_kind": "ru_commercial_summary", "score": 0.95},
            {"doc_id": "doc-native", "doc_kind": "ru_registration_export", "score": 0.60},
        ]
        ordered = escalator._sort_items(items)
        self.assertEqual(ordered[0]["doc_id"], "doc-native")

    def test_evidence_assembler_builds_contract_driven_packet(self):
        engine = ExecDecisionEngine()
        assembler = ExecEvidenceAssembler(retriever=None)
        block_spec = engine.block_specs["rf_entry"]
        packet = engine._build_packet(_sample_dossier(), "case-1", block_spec)
        evidence_packet = assembler.assemble(packet, _stub_question_plan(), case_id="case-1", allow_retrieval=False)
        self.assertIn("selected_evidence", evidence_packet)
        self.assertTrue(any(item["doc_kind"] == "ru_registration_export" for item in evidence_packet["selected_evidence"]))
        self.assertIn("direct_ru_registration_confirmation", evidence_packet["missing_evidence_classes"])

    def test_retrieval_assembler_expands_queries_but_keeps_canonical_filter(self):
        class FakeRetriever:
            def __init__(self):
                self.calls = []

            def retrieve_by_case(self, query, case_id=None, doc_kind=None, top_n=None):
                self.calls.append(
                    {
                        "query": query,
                        "case_id": case_id,
                        "doc_kind": list(doc_kind or []),
                        "top_n": top_n,
                    }
                )
                return [
                    {"doc_id": "doc-approval", "doc_kind": "approval_letter", "snippet": "FDA approval", "score": 0.95},
                    {"doc_id": "doc-grls", "doc_kind": "grls_card", "snippet": "GRLS card", "score": 0.90},
                    {"doc_id": "doc-eaeu", "doc_kind": "eaeu_register", "snippet": "EAEU register", "score": 0.85},
                ]

        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["rf_entry"]
        packet = engine._build_packet(_sample_dossier(), "case-1", block_spec)
        plan = ExecQuestionPlan(
            question_id="rf_entry",
            answer_type="go_no_go",
            needed_dossier_sections=["registrations"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["fda_drugs_at_fda", "ema_epar", "eaeu_register"],
                queries=["apixaban russian federation registration"],
            ),
        )
        normalized = engine._normalize_plan(block_spec, plan)
        retriever = FakeRetriever()
        assembler = ExecEvidenceAssembler(retriever=retriever)
        evidence_packet = assembler.assemble(packet, normalized, case_id="case-1", allow_retrieval=True)
        self.assertTrue(retriever.calls)
        self.assertIn("grls_card", retriever.calls[0]["doc_kind"])
        self.assertNotIn("approval_letter", retriever.calls[0]["doc_kind"])
        self.assertEqual(evidence_packet["evidence_packet_summary"]["retrieved_extra_count"], 1)
        self.assertTrue(any(item["doc_id"] == "doc-grls" for item in evidence_packet["selected_evidence"]))
        self.assertFalse(any(item["doc_id"] == "doc-approval" for item in evidence_packet["selected_evidence"]))


class ExecLlmEnvTests(unittest.TestCase):
    def test_require_exec_openai_api_key_loads_explicit_env_file(self):
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"DDKIT_EXEC_OPENAI_ENV_FILE": str(env_path)}, clear=True):
                key = require_exec_openai_api_key()
        self.assertEqual(key, "test-key")


class ExecAnswerRunnerTests(unittest.TestCase):
    def test_build_exec_retriever_returns_injected_instance(self):
        marker = object()
        self.assertIs(_build_exec_retriever(retriever=marker), marker)


if __name__ == "__main__":
    unittest.main()
