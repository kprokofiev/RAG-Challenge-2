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

    def test_packet_builder_retains_priority_contract_evidence_across_block_allowlists(self):
        dossier = _sample_dossier()
        dossier["registrations"] = [
            {
                "region": "RU",
                "verdict": "confirmed",
                "status": {"value": "registered", "evidence_refs": ["ev-reg-ru"]},
                "identifiers": [{"value": "LP-007", "evidence_refs": ["ev-reg-ru"]}],
                "evidence_refs": ["ev-reg-ru"],
            }
        ]
        dossier["evidence_registry"].extend(
            [
                {
                    "evidence_id": "ev-priority-bridge",
                    "doc_kind": "product_identity_bridge",
                    "source_label": "eaeu_product_identity_bridge",
                    "snippet": (
                        "PRODUCT_IDENTITY_BRIDGE | source=eaeu_product_identity_bridge | jurisdiction=RU "
                        "| registration_id=LP-007 | trade_name=Apixaban | inn=apixaban | mah=Example MAH "
                        "| form=tablet | strength=5 mg | status=Authorised | linked_signal=LP-007 "
                        "| match_level=exact"
                    ),
                },
                {
                    "evidence_id": "ev-priority-reimbursement",
                    "doc_kind": "pricing",
                    "source_label": "ru_minzdrav_public_price_limits",
                    "snippet": (
                        "REIMBURSEMENT_CHECK | source=ru_minzdrav_public_price_limits | jurisdiction=RU "
                        "| check_class=jnvlp_price_limit_row | status=listed_active | registration_id=LP-007"
                    ),
                },
            ]
        )

        engine = ExecDecisionEngine()
        packet = engine._build_packet(dossier, "case-1", engine.block_specs["rf_entry"])
        selected_ids = {item.get("evidence_id") for item in packet["evidence_registry"]}

        self.assertIn("ev-priority-bridge", selected_ids)
        self.assertIn("ev-priority-reimbursement", selected_ids)

        evidence_packet = engine.assembler.assemble(packet, _stub_question_plan(), case_id="case-1", allow_retrieval=False)
        summary = evidence_packet["evidence_packet_summary"]["contract_linkage_summary"]
        self.assertEqual(summary["ru_identity_match"], "same_identifier")
        self.assertTrue(summary["ru_access_registration_id_overlap"])
        self.assertEqual(summary["market_reimbursement_verdict_hint"], "LIMITED")

    def test_operations_readiness_snapshot_separates_screening_from_legal_opinion(self):
        dossier = _sample_dossier()
        dossier["registrations"].append(
            {
                "region": "EAEU",
                "verdict": "confirmed",
                "status": {"value": "authorised", "evidence_refs": ["ev-eaeu-reg"]},
                "identifiers": [{"value": "LP-777", "evidence_refs": ["ev-eaeu-reg"]}],
                "mah": {"value": "Example MAH", "evidence_refs": ["ev-eaeu-reg"]},
                "evidence_refs": ["ev-eaeu-reg"],
            }
        )
        dossier["evidence_registry"].extend(
            [
                {
                    "evidence_id": "ev-eaeu-reg",
                    "doc_kind": "eaeu_document",
                    "snippet": "EAEU reg_no: LP-777 status authorised valid_to=2029-11-19",
                },
                {
                    "evidence_id": "ev-ob",
                    "doc_kind": "patent_expiry_us",
                    "snippet": "LEGAL_EVENT | source=fda_orange_book_data_files | jurisdiction=US | patent=US9326945 | event_type=expiry | event_date=2031-08-24 | status=Orange Book listed",
                },
                {
                    "evidence_id": "ev-fw",
                    "doc_kind": "patent_file_wrapper",
                    "snippet": "CLEARANCE_CHECK | source=uspto_file_wrapper_open_data | jurisdiction=US | patent=US9326945 | check_class=terminal_disclaimer_file_wrapper | status=not_source_verified",
                },
                {
                    "evidence_id": "ev-assign",
                    "doc_kind": "uspto_assignment",
                    "snippet": "RIGHTS_RECORD | source=uspto_assignment | jurisdiction=US | patent=US9326945 | record_type=assignment | status=public USPTO assignment record",
                },
                {
                    "evidence_id": "ev-epo",
                    "doc_kind": "patent_legal_events",
                    "snippet": "LEGAL_EVENT | source=epo_register | jurisdiction=EU | patent=EP4353312 | event_type=grant | event_date=2026-01-01 | status=granted",
                },
                {
                    "evidence_id": "ev-spc",
                    "doc_kind": "patent_national_legal_status",
                    "snippet": "CLEARANCE_CHECK | source=eu_national_spc_registers | jurisdiction=EU | country=DE | patent=EP4353312 | check_class=SPC | status=not_source_verified",
                },
                {
                    "evidence_id": "ev-ru",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "LEGAL_EVENT | source=rospatent_searchplatform | jurisdiction=RU | patent=RU1234567 | event_type=ru_legal_status | event_date=2026-01-01 | status=active",
                },
                {
                    "evidence_id": "ev-eapo",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=EAEU | search_term=apixaban | patents=0 | as_of=2026-04-26",
                },
                {
                    "evidence_id": "ev-price",
                    "doc_kind": "pricing",
                    "snippet": "REIMBURSEMENT_CHECK | source=ru_minzdrav_public_price_limits | jurisdiction=RU | check_class=jnvlp_price_limit_row | status=listed_active | registration_id=LP-777",
                },
                {
                    "evidence_id": "ev-policy",
                    "doc_kind": "payer_policy",
                    "snippet": "REIMBURSEMENT_CHECK | source=ru_federal_program_sources | jurisdiction=RU | check_class=federal_program_or_pathway | status=payer_pathway_source_checked",
                },
                {
                    "evidence_id": "ev-eec",
                    "doc_kind": "payer_policy",
                    "snippet": "REIMBURSEMENT_CHECK | source=eec_market_access_scope | jurisdiction=EAEU | check_class=eaeu_union_reimbursement_scope | status=member_state_scope",
                },
                {
                    "evidence_id": "ev-bridge",
                    "doc_kind": "product_identity_bridge",
                    "snippet": "PRODUCT_IDENTITY_BRIDGE | source=eaeu_product_identity_bridge | jurisdiction=EAEU | registration_id=LP-777 | inn=apixaban | mah=Example MAH | linked_signal=LP-777 | match_level=exact",
                },
            ]
        )

        engine = ExecDecisionEngine()
        packet = engine._build_packet(dossier, "case-1", engine.block_specs["asset_attractiveness"])
        evidence_packet = engine.assembler.assemble(packet, _stub_question_plan(), case_id="case-1", allow_retrieval=False)
        snapshot = evidence_packet["contract_linkage"]["operations_readiness_snapshot"]

        self.assertTrue(snapshot["screening_ready"])
        self.assertFalse(snapshot["operations_evidence_ready"])
        self.assertFalse(snapshot["legal_opinion_ready"])
        self.assertFalse(snapshot["payer_tier_clearance"])
        self.assertIn("ru_payer_tier_restrictions", snapshot["missing_operations_checks"])
        self.assertIn("us_pte_file_wrapper", snapshot["partial_operations_checks"])

    def test_eaeu_packet_includes_member_state_commercial_signals(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["eaeu_entry"]
        dossier = _sample_dossier()
        dossier["commercial_signals"].append(
            {
                "signal_id": "sig-2",
                "region": "KZ",
                "category": "pricing",
                "verdict": "partial",
                "summary": {"value": "Kazakhstan market signal", "evidence_refs": ["ev-com-2"]},
                "source_name": "pricing",
                "source_tier": "secondary",
                "source_priority": 20,
                "evidence_refs": ["ev-com-2"],
            }
        )
        dossier["evidence_registry"].append(
            {
                "evidence_id": "ev-com-2",
                "doc_id": "doc-com-2",
                "page": 7,
                "snippet": "Kazakhstan market signal",
                "doc_kind": "pricing",
            }
        )

        packet = engine._build_packet(dossier, "case-1", block_spec)

        self.assertEqual(
            {item["region"] for item in packet["commercial_signals"]},
            {"RU", "KZ"},
        )

    def test_packet_builder_preserves_section_linked_evidence_before_truncation(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["asset_attractiveness"]
        dossier = _sample_dossier()
        dossier["clinical_studies"][0]["evidence_refs"] = ["ev-linked-late"]
        dossier["clinical_studies"][0]["phase"]["evidence_refs"] = ["ev-linked-late"]
        dossier["evidence_registry"] = [
            {
                "evidence_id": f"ev-filler-{idx}",
                "doc_id": f"doc-filler-{idx}",
                "page": 1,
                "snippet": "FDA filler",
                "doc_kind": "us_fda",
            }
            for idx in range(100)
        ] + [
            {
                "evidence_id": "ev-linked-late",
                "doc_id": "doc-linked",
                "page": 1,
                "snippet": "Late Phase 3 ctgov results",
                "doc_kind": "ctgov_results",
            }
        ]

        packet = engine._build_packet(dossier, "case-1", block_spec)

        self.assertIn(
            "ev-linked-late",
            {item.get("evidence_id") for item in packet["evidence_registry"]},
        )

    def test_packet_builder_preserves_priority_patent_evidence_before_truncation(self):
        engine = ExecDecisionEngine()
        block_spec = engine.block_specs["asset_attractiveness"]
        dossier = _sample_dossier()
        dossier["registrations"] = [
            {
                "region": "RU",
                "verdict": "confirmed",
                "status": {"value": "registered", "evidence_refs": [f"ev-linked-{idx}" for idx in range(100)]},
                "evidence_refs": [f"ev-linked-{idx}" for idx in range(100)],
            }
        ]
        dossier["evidence_registry"] = [
            {
                "evidence_id": f"ev-linked-{idx}",
                "doc_id": f"doc-linked-{idx}",
                "page": 1,
                "snippet": "Linked registration filler",
                "doc_kind": "grls",
            }
            for idx in range(100)
        ] + [
            {
                "evidence_id": "ev-eapo-nohit",
                "doc_id": "doc-eapo-nohit",
                "page": 1,
                "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=EAEU | search_term=апиксабан | patents=0 | as_of=2025-10-30",
                "doc_kind": "ru_patent_fips",
            }
        ]

        packet = engine._build_packet(dossier, "case-1", block_spec)

        self.assertIn(
            "ev-eapo-nohit",
            {item.get("evidence_id") for item in packet["evidence_registry"]},
        )

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
        self.assertEqual(planner_prompt.max_output_tokens, 2400)
        self.assertEqual(answer_prompt.max_output_tokens, 5200)

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

    def test_verifier_attaches_matching_refs_before_downgrading(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="MEDIUM",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Partial",
            full_answer="Partial",
            why_this_verdict=[
                ExecWhyClaim(
                    claim="Phase 3 ARISTOTLE results are present in the packet",
                    claim_type="hard_evidence_backed",
                    evidence_refs=[],
                )
            ],
            decision_blockers=[],
            next_actions=[],
        )
        packet = {
            "evidence_ids": ["ev-clin-results"],
            "selected_evidence": [
                {
                    "evidence_id": "ev-clin-results",
                    "doc_kind": "ctgov_results",
                    "snippet": "ARISTOTLE Phase 3 results met the primary endpoint.",
                }
            ],
            "critical_unknowns": [],
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.why_this_verdict[0].claim_type, "hard_evidence_backed")
        self.assertEqual(repaired.why_this_verdict[0].evidence_refs, ["ev-clin-results"])

    def test_verifier_repairs_confidence_mismatch_warns(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="generic_opportunity",
            title="Generic opportunity",
            verdict="NOT_EVIDENCED",
            confidence="HIGH",
            sufficiency="PARTIAL",
            short_answer="Insufficient",
            full_answer="Insufficient",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        repaired, verification = verifier.verify_and_repair(
            block,
            {"evidence_ids": [], "critical_unknowns": []},
            block_spec=None,
            allow_repair=True,
        )
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.confidence, "LOW")

    def test_verifier_softens_asset_no_go_when_only_missing_evidence_drives_block(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="NO_GO",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="NO_GO because RU/EAEU IP window is missing and validity is not confirmed.",
            full_answer="The packet shows registrations and clinical maturity, but RU/EAEU IP window evidence is unresolved and valid_to is blank, so the model returned NO_GO.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["dossier_quality_v2"]["decision_readiness"]["context_integrity"] = "GREEN"
        packet["registrations"].append(
            {
                "region": "US",
                "verdict": "confirmed",
                "status": {"value": "approved", "evidence_refs": ["ev-reg-us"]},
                "evidence_refs": ["ev-reg-us"],
            }
        )
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "HOLD")
        self.assertEqual(repaired.sufficiency, "PARTIAL")

    def test_verifier_repairs_rf_entry_when_only_eaeu_validity_blocks_ru_go(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="rf_entry",
            title="RF entry",
            verdict="INSUFFICIENT_EVIDENCE",
            confidence="LOW",
            sufficiency="INSUFFICIENT",
            short_answer="RU entry cannot be concluded because EAEU valid_to is missing.",
            full_answer="RU GRLS is active, but the answer is blocked only because EAEU validity is not confirmed.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["dossier_quality_v2"]["decision_readiness"]["context_integrity"] = "GREEN"
        packet["registrations"][0]["status"] = {"value": "active", "evidence_refs": ["ev-reg-ru"]}
        packet["commercial_signals"][0]["verdict"] = "confirmed"
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertTrue(any("EAEU" in caveat for caveat in repaired.caveats))

    def test_verifier_repairs_rf_entry_when_only_legal_status_unknown_blocks_ru_go(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="rf_entry",
            title="RF entry",
            verdict="INSUFFICIENT_EVIDENCE",
            confidence="MEDIUM",
            sufficiency="INSUFFICIENT",
            short_answer="RU GRLS is active, but LEGAL_STATUS_NOT_AVAILABLE still blocks RF entry.",
            full_answer=(
                "The RU packet confirms ACTIVE registration and RU commercial support, "
                "but the answer still points to LEGAL_STATUS_NOT_AVAILABLE and asks for "
                "non-suspension/non-revocation confirmation."
            ),
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["dossier_quality_v2"]["decision_readiness"]["context_integrity"] = "GREEN"
        packet["registrations"][0]["status"] = {"value": "active", "evidence_refs": ["ev-reg-ru"]}
        packet["commercial_signals"][0]["verdict"] = "confirmed"
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")

    def test_verifier_repairs_rf_entry_when_procurement_or_route_corroboration_overconstrains(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="rf_entry",
            title="RF entry",
            verdict="NO_GO",
            confidence="MEDIUM",
            sufficiency="INSUFFICIENT",
            short_answer="RU GRLS is active, but procurement shows no matching rows and route/dosage form is not corroborated beyond GRLS.",
            full_answer=(
                "RU commercial signals are otherwise confirmed, but the answer still blocks on no matching procurement rows "
                "and says it lacks a clear RU instruction / identity fields beyond GRLS."
            ),
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["dossier_quality_v2"]["decision_readiness"]["context_integrity"] = "GREEN"
        packet["registrations"][0]["status"] = {"value": "active", "evidence_refs": ["ev-reg-ru"]}
        packet["commercial_signals"][0]["verdict"] = "confirmed"
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")

    def test_verifier_repairs_rf_hold_when_inn_level_commercial_linkage_is_only_caveat(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="rf_entry",
            title="RF entry",
            verdict="HOLD",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer=(
                "RU registration is active and RU access signals are present, but product-context alignment "
                "is not fully proven."
            ),
            full_answer=(
                "The packet says same_identifier_confirmed is false and product_context_match_confirmed is false, "
                "so the answer blocks RF entry on INN-level commercial linkage."
            ),
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "rf_entry_blocker_1",
                    "title": "Product-context alignment is not fully proven",
                    "severity": "DECISION_BLOCKING",
                    "rationale": "Commercial/access signals exist, but linkage is only INN-level.",
                    "evidence_refs": ["ev-com-1"],
                }
            ],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["dossier_quality_v2"]["decision_readiness"]["context_integrity"] = "YELLOW"
        packet["registrations"][0]["status"] = {"value": "active", "evidence_refs": ["ev-reg-ru"]}
        packet["commercial_signals"][0]["verdict"] = "confirmed"
        packet["contract_linkage"] = {
            "market_entry_linkage": {
                "RU": {
                    "registration_anchor_present": True,
                    "commercial_signal_count": 3,
                    "identity_match": "inn_level_only",
                    "same_identifier_confirmed": False,
                    "product_context_match_confirmed": False,
                    "evidence_refs": ["ev-com-1"],
                }
            }
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertFalse(repaired.decision_blockers)
        self.assertTrue(any("GRLS" in caveat for caveat in repaired.caveats))

    def test_verifier_repairs_rf_hold_when_product_context_wording_is_only_caveat(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="rf_entry",
            title="RF entry",
            verdict="HOLD",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="RU registration is active, but the packet does not conclusively map the GRLS record to the exact dossier product context.",
            full_answer="Hold only because exact product context linkage for commercial signals is not perfect.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "rf_entry_blocker_1",
                    "title": "Product context linkage is not exact",
                    "severity": "DECISION_BLOCKING",
                    "rationale": "Commercial/access signals remain only INN-level rather than exact product context.",
                    "evidence_refs": ["ev-com-1"],
                }
            ],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["dossier_quality_v2"]["decision_readiness"]["context_integrity"] = "YELLOW"
        packet["registrations"][0]["status"] = {"value": "active", "evidence_refs": ["ev-reg-ru"]}
        packet["commercial_signals"][0]["verdict"] = "confirmed"
        packet["contract_linkage"] = {
            "market_entry_linkage": {
                "RU": {
                    "registration_anchor_present": True,
                    "commercial_signal_count": 3,
                    "identity_match": "inn_level_only",
                    "evidence_refs": ["ev-com-1"],
                }
            }
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertFalse(repaired.decision_blockers)

    def test_verifier_promotes_eaeu_insufficiency_to_hold_when_reg_anchor_exists(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="eaeu_entry",
            title="EAEU entry",
            verdict="INSUFFICIENT_EVIDENCE",
            confidence="LOW",
            sufficiency="INSUFFICIENT",
            short_answer="Insufficient because valid_to is blank and commercial pathway is missing.",
            full_answer="An EAEU registration exists, but valid_to is blank and commercial evidence is RU-only.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["registrations"].append(
            {
                "region": "EAEU",
                "verdict": "confirmed",
                "status": {"value": "Authorised", "evidence_refs": ["ev-eaeu"]},
                "identifiers": [{"value": "LP-EAEU-1", "evidence_refs": ["ev-eaeu"]}],
                "evidence_refs": ["ev-eaeu"],
            }
        )
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "HOLD")
        self.assertEqual(repaired.sufficiency, "PARTIAL")

    def test_verifier_promotes_rf_conditional_go_to_go_when_identity_linkage_is_explicit(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="rf_entry",
            title="RF entry",
            verdict="CONDITIONAL_GO",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="RU registration exists, but commercial evidence still looks INN-level and not clearly linked to the identifier/MAH.",
            full_answer="The packet confirms RU registration, but the answer still asks for explicit identifier linkage before a clean GO.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["contract_linkage"] = {
            "market_entry_linkage": {
                "RU": {
                    "registration_anchor_present": True,
                    "commercial_signal_count": 1,
                    "identity_match": "same_identifier",
                    "evidence_refs": ["ev-com-1"],
                }
            }
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")

    def test_verifier_promotes_rf_conditional_go_despite_closed_evidence_state_phrase(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="rf_entry",
            title="RF entry",
            verdict="CONDITIONAL_GO",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Conditional GO because the record is not a fully closed evidence state.",
            full_answer=(
                "RU registration and access linkage are confirmed, but the record is not a fully closed evidence state "
                "because an explicit RU policy-act source was not retrieved."
            ),
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "rf_entry_blocker_1",
                    "title": "Explicit RU policy-act confirmation not retrieved",
                    "severity": "IMPORTANT",
                    "rationale": "This is a verification gap, not a demonstrated source-backed block.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[],
            caveats=["The record is not a fully closed evidence state."],
        )
        packet = _sample_dossier()
        packet["contract_linkage"] = {
            "market_entry_linkage": {
                "RU": {
                    "registration_anchor_present": True,
                    "commercial_signal_count": 39,
                    "identity_match": "same_identifier",
                    "identity_match_scope": "source_native_registered_product_context",
                    "source_native_registration_id_overlap": True,
                    "source_native_reimbursement_signal_count": 32,
                    "evidence_refs": ["ev-com-1"],
                }
            }
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertIn("promoted_rf_conditional_go_to_go_on_identity_linkage", verification.repair_reason)

    def test_verifier_removes_eaeu_same_id_overconstraint_when_native_identity_is_strong(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="eaeu_entry",
            title="EAEU entry",
            verdict="HOLD",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="EAEU entry is held because GRLS same-id corroboration is missing.",
            full_answer="The packet shows an EAEU registration, but the answer still blocks on GRLS same-id corroboration and different registration numbers.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["contract_linkage"] = {
            "registration_identity_map": [
                {
                    "context": "EAEU",
                    "source_class": "EAEU-native",
                    "identity_confidence": "HIGH",
                    "validity_type": "date_present",
                    "valid_to": "2029-11-19",
                    "identifiers": ["LP-EAEU-1"],
                    "evidence_refs": ["ev-eaeu"],
                }
            ],
            "market_entry_linkage": {
                "EAEU": {"commercial_signal_count": 0}
            },
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "CONDITIONAL_GO")
        self.assertEqual(repaired.sufficiency, "PARTIAL")

    def test_verifier_moves_generic_eaeu_coverage_blocker_to_caveat_when_native_entry_is_strong(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="eaeu_entry",
            title="EAEU entry",
            verdict="GO",
            confidence="MEDIUM",
            sufficiency="SUFFICIENT",
            short_answer="EAEU entry is positive, but the answer still mentions GRLS same-id corroboration.",
            full_answer="The EAEU-native registration is active, but the narrative still asks for GRLS same-id corroboration.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "eaeu_entry_blocker_1",
                    "title": "Decision-grade dossier coverage is incomplete",
                    "severity": "DECISION_BLOCKING",
                    "rationale": (
                        "The coverage ledger still flags decision_readiness as insufficient and "
                        "the source manifest shows missing source classes."
                    ),
                    "evidence_refs": [],
                }
            ],
            next_actions=[
                {
                    "action_id": "eaeu_entry_action_1",
                    "action": "Attach missing source classes from the source manifest.",
                    "priority": "HIGH",
                    "rationale": "Needed to move the coverage ledger to decision-grade readiness.",
                    "evidence_refs": [],
                }
            ],
            caveats=[
                "The packet is source-native for the EAEU registration decision, but dossier readiness is still below the threshold for a positive verdict."
            ],
        )
        packet = {
            "evidence_ids": ["ev-eaeu"],
            "critical_unknowns": [],
            "contract_linkage": {
                "registration_identity_map": [
                    {
                        "context": "EAEU",
                        "source_class": "EAEU-native",
                        "identity_confidence": "HIGH",
                        "status_positive": True,
                        "validity_type": "date_present",
                        "valid_to": "2029-11-19",
                        "identifiers": ["LP-EAEU-1"],
                        "evidence_refs": ["ev-eaeu"],
                    }
                ],
                "market_entry_linkage": {
                    "EAEU": {"commercial_signal_count": 1}
                },
            },
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertFalse(repaired.decision_blockers)
        self.assertFalse(repaired.next_actions)
        self.assertIn("dedicated blocks", " ".join(repaired.caveats))

    def test_verifier_lifts_asset_when_ru_eaeu_ip_snapshot_supports_no_hit(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="HOLD",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Asset is still on HOLD because RU/EAEU patent expiry and legal status remain missing.",
            full_answer="Registrations are present, but the answer still treats RU/EAEU patent window missingness as a hold-level blocker.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["contract_linkage"] = {
            "ru_eaeu_ip_window_snapshot": {
                "conclusion": "NO_LISTED_BLOCKING_PATENT_EVIDENCE",
                "as_of_date": "2025-10-30",
            }
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "CONDITIONAL_GO")
        self.assertEqual(repaired.sufficiency, "PARTIAL")

    def test_verifier_lifts_asset_with_selected_sections_packet_and_negated_negative_phrase(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="HOLD",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Asset remains HOLD because RU/EAEU IP-window evidence is unresolved.",
            full_answer="The gap is not source-backed negative evidence; it is an incomplete IP-window snapshot.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = {
            "selected_sections": {
                "registrations": [
                    {
                        "region": "EAEU",
                        "status": {"value": "Authorised", "evidence_refs": ["ev-eaeu"]},
                        "evidence_refs": ["ev-eaeu"],
                    }
                ]
            },
            "contract_linkage": {
                "registration_identity_map": [
                    {
                        "context": "EAEU",
                        "status_positive": True,
                        "evidence_refs": ["ev-eaeu"],
                    }
                ],
                "ru_eaeu_ip_window_snapshot": {
                    "conclusion": "PARTIAL_OPEN_WINDOW_EVIDENCE",
                    "as_of_date": "2025-10-30",
                },
            },
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "CONDITIONAL_GO")
        self.assertIn("residual-risk", " ".join(repaired.caveats).lower())

    def test_verifier_moves_asset_coverage_gap_to_caveat_for_screening_ready_packet(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="CONDITIONAL_GO",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Conditional because coverage ledger is not decision-complete.",
            full_answer="Core asset signals are favorable, but run_manifest and decision_readiness are incomplete.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "coverage_gap",
                    "title": "Material dossier incompleteness prevents a positive BD verdict",
                    "severity": "DECISION_BLOCKING",
                    "rationale": "The coverage ledger is not decision-complete and decision_readiness is insufficient.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "registrations": [{"region": "RU", "status": {"value": "active"}}],
            "contract_linkage": {"registration_identity_map": [{"context": "RU"}]},
            "evidence_packet_summary": {
                "contract_linkage_summary": {
                    "ru_source_native_access_signal_count": 3,
                    "market_reimbursement_verdict_hint": "LIMITED",
                }
            },
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertFalse(repaired.decision_blockers)
        self.assertIn("moved_asset_coverage_gap_to_screening_caveat", verification.repair_reason)

    def test_verifier_downgrades_closed_ip_window_when_legal_status_is_not_decision_grade(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="ip_legal_window",
            title="IP legal window",
            verdict="CLOSED",
            confidence="HIGH",
            sufficiency="SUFFICIENT",
            short_answer="CLOSED even though RU/EAEU legal status is incomplete and no reconciled SPC/PTE evidence is present.",
            full_answer="Future expiries exist, but family legal events are mixed, incomplete, and not reconciled source-natively.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = {
            "evidence_ids": [],
            "critical_unknowns": [],
            "contract_linkage": {
                "family_legal_events_snapshot": {"decision_grade": False},
                "fto_screening_snapshot": {
                    "full_fto_verdict_allowed": False,
                    "potential_blocker_regions": ["US", "EU", "RU", "EAEU"],
                    "decision_grade_blockers": ["FAMILY_LEGAL_EVENTS_INCOMPLETE"],
                },
            },
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "LIMITED")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertIn("screening-grade", " ".join(repaired.caveats).lower())

    def test_verifier_reframes_generic_opportunity_by_region(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="generic_opportunity",
            title="Generic opportunity",
            verdict="NOT_EVIDENCED",
            confidence="LOW",
            sufficiency="INSUFFICIENT",
            short_answer="Generic opportunity is not evidenced globally.",
            full_answer="EU and US remain unresolved, so the answer collapsed the full generic question into NOT_EVIDENCED.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = {
            "evidence_ids": [],
            "critical_unknowns": [],
            "contract_linkage": {
                "generic_opportunity_by_region": {
                    "RU": {"verdict": "POTENTIAL_GO"},
                    "EAEU": {"verdict": "POTENTIAL_GO"},
                    "EU": {"verdict": "HOLD_OR_NO_GO"},
                    "US": {"verdict": "NOT_EVIDENCED"},
                }
            },
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "MEDIUM")
        self.assertEqual(repaired.sufficiency, "PARTIAL")

    def test_verifier_demotes_synthesis_to_screening_scope_for_asset(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="HOLD",
            confidence="LOW",
            sufficiency="INSUFFICIENT",
            short_answer="Asset is on HOLD because synthesis route corroboration is partial.",
            full_answer="The answer makes synthesis/manufacturing route a primary blocker for the BD asset verdict.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = {
            "evidence_ids": [],
            "critical_unknowns": [],
            "contract_linkage": {
                "synthesis_screening": {
                    "decision_use": "technical_screening_only",
                }
            },
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "CONDITIONAL_GO")
        self.assertIn("screening-grade", " ".join(repaired.caveats).lower())

    def test_verifier_lifts_generic_not_evidenced_to_low_when_source_screening_exists(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="generic_opportunity",
            title="Generic opportunity",
            verdict="NOT_EVIDENCED",
            confidence="MEDIUM",
            sufficiency="INSUFFICIENT",
            short_answer="The packet does not support a generic launch opportunity.",
            full_answer="US and EU patent positions remain active/pending, with some EP entries withdrawn, so positive gate closure is missing.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "contract_linkage": {
                "source_evidence_manifest": {"checked_source_count": 4, "limited_source_count": 4},
                "family_legal_events_snapshot": {
                    "coverage_status": "PARTIAL",
                    "evidence_refs": ["ev-family"],
                },
            }
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "LOW")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertIn("lifted_generic_not_evidenced_to_screening_partial", verification.repair_reason)

    def test_verifier_lifts_portfolio_low_to_medium_when_screening_anchors_exist(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="portfolio_opportunity",
            title="Portfolio opportunity",
            verdict="LOW",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Commercially strong, but not a clean portfolio opportunity due to legal-status caveats.",
            full_answer="Approvals and phase 3 maturity exist, while jurisdiction-level legal/status reconciliation remains incomplete.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "portfolio_gap",
                    "title": "Unreconciled legal-status trail",
                    "severity": "IMPORTANT",
                    "rationale": "Legal-status reconciliation remains a follow-up.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[],
            caveats=[],
        )
        packet = _sample_dossier()
        packet["contract_linkage"] = {
            "phase3_results": {
                "phase3_study_count": 2,
                "phase3_with_ctgov_results_evidence": 1,
            },
        }
        packet["evidence_packet_summary"] = {
            "contract_linkage_summary": {
                "ru_source_native_access_signal_count": 2,
                "market_reimbursement_verdict_hint": "LIMITED",
            }
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "MEDIUM")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertIn("lifted_portfolio_low_to_screening_medium", verification.repair_reason)

    def test_verifier_promotes_sufficient_asset_conditional_go_without_blockers(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="CONDITIONAL_GO",
            confidence="MEDIUM",
            sufficiency="SUFFICIENT",
            short_answer="Commercially attractive, but proceed conditionally because final legal/payer diligence is separate.",
            full_answer="The asset has registration and market anchors; IP/FTO and payer diligence remain in dedicated blocks.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["contract_linkage"] = {
            "phase3_results": {
                "phase3_study_count": 2,
                "phase3_with_ctgov_results_evidence": 1,
            },
            "market_reimbursement_snapshot": {
                "verdict_hint": "LIMITED",
            },
        }
        packet["evidence_packet_summary"] = {
            "contract_linkage_summary": {
                "ru_source_native_access_signal_count": 2,
                "market_reimbursement_verdict_hint": "LIMITED",
            }
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertIn("promoted_asset_conditional_go_to_go_without_decision_blockers", verification.repair_reason)

    def test_verifier_moves_asset_ip_fto_hold_to_dedicated_legal_window_caveat(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="HOLD",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Apixaban is commercially present and clinically mature, but I would hold because the patent/exclusivity picture is not cleanly resolved.",
            full_answer="The IP/FTO legal window is screening-grade and not decision-complete; EU patent legal events include mixed pending/withdrawn signals.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "asset_attractiveness_blocker_1",
                    "title": "Patent/exclusivity is not resolved",
                    "severity": "DECISION_BLOCKING",
                    "rationale": "FTO is not decision-grade.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[],
        )
        packet = _sample_dossier()
        packet["contract_linkage"] = {
            "phase3_results": {
                "phase3_study_count": 2,
                "phase3_with_ctgov_results_evidence": 1,
            },
            "fto_screening_snapshot": {
                "conclusion": "POTENTIAL_BLOCKERS_REQUIRE_REVIEW",
                "full_fto_verdict_allowed": False,
                "potential_blocker_regions": ["US", "EU", "RU", "EAEU"],
            },
        }
        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)
        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "CONDITIONAL_GO")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertFalse(repaired.decision_blockers)
        self.assertIn("dedicated legal-window block", " ".join(repaired.caveats))

    def test_verifier_lifts_underresolved_ip_window_to_limited_when_source_native_blockers_exist(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="ip_legal_window",
            title="IP legal window",
            verdict="UNRESOLVED",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Patent activity exists, but the window is unresolved.",
            full_answer="US and EU patent activity is visible, while RU/EAEU reconciliation is incomplete.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = {
            "evidence_ids": ["ev-ip"],
            "contract_linkage": {
                "fto_screening_snapshot": {
                    "conclusion": "POTENTIAL_BLOCKERS_REQUIRE_REVIEW",
                    "full_fto_verdict_allowed": False,
                    "potential_blocker_regions": ["US", "EU"],
                    "evidence_refs": ["ev-ip"],
                    "country_effect_status_by_region": {
                        "US": {"window_status": "potentially_blocked"},
                        "EU": {"window_status": "potentially_blocked"},
                    },
                },
                "family_legal_events_snapshot": {
                    "decision_grade": False,
                    "evidence_refs": ["ev-ip"],
                },
            },
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "LIMITED")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertIn("screening", " ".join(repaired.caveats).lower())

    def test_verifier_downgrades_conditional_go_with_blockers_to_hold(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="eaeu_entry",
            title="EAEU entry",
            verdict="CONDITIONAL_GO",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Conditional go despite unresolved blockers.",
            full_answer="Positive verdict conflicts with explicit blockers.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "blk-1",
                    "title": "Missing payer evidence",
                    "severity": "MUST_VERIFY_NOW",
                    "rationale": "Still unresolved.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[],
        )
        repaired, verification = verifier.verify_and_repair(block, _sample_dossier(), block_spec=None, allow_repair=True)
        self.assertEqual(repaired.verdict, "HOLD")
        self.assertEqual(verification.overall_status, "PASS")

    def test_verifier_allows_conditional_asset_when_blockers_are_reflected(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="asset_attractiveness",
            title="Asset attractiveness",
            verdict="CONDITIONAL_GO",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="CONDITIONAL_GO with limited regional readiness.",
            full_answer="The asset remains attractive in the US/EU, but RF/EAEU entry blockers and operations-readiness caveats remain explicit.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "blk-1",
                    "title": "RF/EAEU registration blockers",
                    "severity": "MUST_VERIFY_NOW",
                    "rationale": "No RU or EAEU registration is source-verified for the scoped product context.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[
                {
                    "action_id": "act-1",
                    "action": "Verify missing regional registrations.",
                    "priority": "HIGH",
                    "rationale": "Retrieve or confirm source-native RU/EAEU no-registration evidence before any multi-region entry decision.",
                    "evidence_refs": [],
                }
            ],
            caveats=["Partial screening verdict, not operations-ready."],
        )

        repaired, verification = verifier.verify_and_repair(block, _sample_dossier(), block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "CONDITIONAL_GO")
        self.assertIsNone(verification.repair_reason)


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

    def test_evidence_assembler_emits_contract_linkage(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "allowed_doc_kinds": ["ctgov_results", "patent_expiry_us", "eaeu_document", "ru_patent_fips"],
            "required_sections": ["clinical_studies", "patent_families", "registrations"],
            "clinical_studies": [
                {
                    "study_id": {"value": "NCT-1", "evidence_refs": ["ev-clin"]},
                    "phase": {"value": "Phase 3", "evidence_refs": ["ev-clin"]},
                    "status": {"value": "Completed", "evidence_refs": ["ev-clin"]},
                    "conclusion": {"value": "Primary endpoint results were reported.", "evidence_refs": ["ev-clin"]},
                    "evidence_refs": ["ev-clin"],
                }
            ],
            "patent_families": [
                {
                    "family_id": "fam-1",
                    "representative_pub": {"value": "US123", "evidence_refs": ["ev-pat-us"]},
                    "expiry_by_country": [
                        {"value": "US: 2046-02-12", "evidence_refs": ["ev-pat-us"]},
                        {"value": "EP: 2046-03-04", "evidence_refs": ["ev-pat-eu"]},
                    ],
                    "evidence_refs": ["ev-pat-us", "ev-pat-eu"],
                }
            ],
            "registrations": [
                {
                    "region": "EAEU",
                    "status": {"value": "Authorised", "evidence_refs": ["ev-eaeu"]},
                    "mah": {"value": "MAH Ltd", "evidence_refs": ["ev-eaeu"]},
                    "identifiers": [{"value": "LP-001", "evidence_refs": ["ev-eaeu"]}],
                    "forms_strengths": [{"value": "tablet | 5 mg", "evidence_refs": ["ev-eaeu"]}],
                    "validity_type": "missing_in_source",
                    "validity_evidence_refs": ["ev-eaeu"],
                    "evidence_refs": ["ev-eaeu"],
                }
            ],
            "evidence_registry": [
                {"evidence_id": "ev-clin", "doc_id": "doc-clin", "doc_kind": "ctgov_results", "snippet": "Phase 3 primary endpoint results"},
                {"evidence_id": "ev-pat-us", "doc_id": "doc-pat-us", "doc_kind": "patent_expiry_us", "snippet": "US: 2046-02-12"},
                {"evidence_id": "ev-pat-eu", "doc_id": "doc-pat-eu", "doc_kind": "patent_expiry_us", "snippet": "EP: 2046-03-04"},
                {"evidence_id": "ev-eaeu", "doc_id": "doc-eaeu", "doc_kind": "eaeu_document", "snippet": "Status: Authorised\nValid To:\nMAH (Holder): MAH Ltd"},
                {
                    "evidence_id": "ev-pat-ru",
                    "doc_id": "doc-pat-ru",
                    "doc_kind": "ru_patent_fips",
                    "snippet": '{"doc_id":"RU2642983C2_20180129","jurisdiction":"RU","expiry_date":"2039-10-25"}',
                },
            ],
        }
        plan = ExecQuestionPlan(
            question_id="asset_attractiveness",
            answer_type="go_no_go",
            needed_dossier_sections=["clinical_studies", "patent_families", "registrations"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["ctgov_results", "patent_expiry_us", "eaeu_document", "ru_patent_fips"],
                queries=["apixaban evidence"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        self.assertEqual(linkage["phase3_results"]["phase3_with_ctgov_results_evidence"], 1)
        self.assertIn("US", linkage["ip_window"]["expiry_by_region"])
        self.assertIn("EU", linkage["ip_window"]["expiry_by_region"])
        self.assertIn("RU", linkage["patent_legal_status_snapshot"]["resolved_regions"])
        self.assertEqual(
            linkage["patent_legal_status_snapshot"]["regions"]["RU"]["window_status"],
            "potentially_blocked",
        )
        self.assertFalse(linkage["eaeu_registration"]["has_valid_to"])
        self.assertFalse(linkage["eaeu_registration"]["has_validity_state"])
        self.assertEqual(linkage["eaeu_registration"]["validity_types"], ["missing_in_source"])
        self.assertTrue(linkage["eaeu_registration"]["has_identifier_mah_linkage"])

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

    def test_contract_linkage_accepts_official_eapo_no_hit_for_eaeu(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "asset_attractiveness",
            "allowed_doc_kinds": ["ru_patent_fips"],
            "patent_families": [],
            "evidence_registry": [
                {
                    "evidence_id": "ev-eapo-nohit",
                    "doc_id": "doc-eapo-nohit",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=EAEU | search_term=апиксабан | patents=0 | as_of=2025-10-30",
                }
            ],
        }
        plan = ExecQuestionPlan(
            question_id="asset_attractiveness",
            answer_type="go_no_go",
            needed_dossier_sections=["patent_families"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["ru_patent_fips"],
                queries=["apixaban eaeu patent status"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        self.assertIn("EAEU", linkage["patent_legal_status_snapshot"]["resolved_regions"])
        self.assertEqual(
            linkage["patent_legal_status_snapshot"]["regions"]["EAEU"]["window_status"],
            "open",
        )
        self.assertEqual(
            linkage["patent_legal_status_snapshot"]["regions"]["EAEU"]["conclusion"],
            "NO_LISTED_BLOCKING_PATENT_EVIDENCE",
        )

    def test_contract_linkage_prioritizes_official_eapo_no_hit_past_chunk_limit(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        filler = [
            {
                "evidence_id": f"ev-filler-{idx}",
                "doc_id": f"doc-filler-{idx}",
                "doc_kind": "ru_patent_fips",
                "snippet": f"Filler patent registry snippet {idx} without decision-grade no-hit marker.",
            }
            for idx in range(40)
        ]
        base_packet = {
            "block_id": "asset_attractiveness",
            "allowed_doc_kinds": ["ru_patent_fips"],
            "patent_families": [],
            "evidence_registry": filler
            + [
                {
                    "evidence_id": "ev-eapo-nohit",
                    "doc_id": "doc-eapo-nohit",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=EAEU | search_term=апиксабан | patents=0 | as_of=2025-10-30",
                }
            ],
        }
        plan = ExecQuestionPlan(
            question_id="asset_attractiveness",
            answer_type="go_no_go",
            needed_dossier_sections=["patent_families"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["ru_patent_fips"],
                queries=["apixaban eaeu patent status"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        self.assertLessEqual(len(evidence_packet["selected_evidence"]), 30)
        self.assertTrue(any(item["doc_id"] == "doc-eapo-nohit" for item in evidence_packet["selected_evidence"]))
        self.assertEqual(
            linkage["patent_legal_status_snapshot"]["regions"]["EAEU"]["conclusion"],
            "NO_LISTED_BLOCKING_PATENT_EVIDENCE",
        )

    def test_contract_linkage_keeps_priority_patent_snapshot_when_planner_omits_doc_kind(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "asset_attractiveness",
            "allowed_doc_kinds": ["grls", "ru_patent_fips"],
            "patent_families": [],
            "evidence_registry": [
                {
                    "evidence_id": "ev-ru-reg",
                    "doc_id": "doc-ru-reg",
                    "doc_kind": "grls",
                    "snippet": "GRLS active apixaban registration",
                },
                {
                    "evidence_id": "ev-eapo-nohit",
                    "doc_id": "doc-eapo-nohit",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=EAEU | search_term=апиксабан | patents=0 | as_of=2025-10-30",
                },
            ],
        }
        plan = ExecQuestionPlan(
            question_id="asset_attractiveness",
            answer_type="go_no_go",
            needed_dossier_sections=["patent_families"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["grls"],
                queries=["apixaban registration"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        self.assertTrue(any(item["doc_id"] == "doc-eapo-nohit" for item in evidence_packet["selected_evidence"]))
        self.assertEqual(
            linkage["patent_legal_status_snapshot"]["regions"]["EAEU"]["conclusion"],
            "NO_LISTED_BLOCKING_PATENT_EVIDENCE",
        )

    def test_evidence_assembler_builds_identity_maps_and_regional_snapshots(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "asset_attractiveness",
            "allowed_doc_kinds": ["ru_registration_export", "eaeu_document", "ru_procurement_summary", "ru_patent_fips"],
            "required_sections": ["registrations", "commercial_signals", "product_contexts", "patent_families"],
            "registrations": [
                {
                    "region": "RU",
                    "status": {"value": "active", "evidence_refs": ["ev-ru-reg"]},
                    "mah": {"value": "Canon", "evidence_refs": ["ev-ru-reg"]},
                    "identifiers": [{"value": "LP-001", "evidence_refs": ["ev-ru-reg"]}],
                    "forms_strengths": [{"value": "tablet | 5 mg", "evidence_refs": ["ev-ru-reg"]}],
                    "evidence_refs": ["ev-ru-reg"],
                },
                {
                    "region": "EAEU",
                    "status": {"value": "Authorised", "evidence_refs": ["ev-eaeu-reg"]},
                    "mah": {"value": "Lekpharm", "evidence_refs": ["ev-eaeu-reg"]},
                    "identifiers": [{"value": "LP-EAEU-1", "evidence_refs": ["ev-eaeu-reg"]}],
                    "forms_strengths": [{"value": "tablet | 5 mg", "evidence_refs": ["ev-eaeu-reg"]}],
                    "validity_type": "date_present",
                    "valid_to": {"value": "2029-11-19", "evidence_refs": ["ev-eaeu-reg"]},
                    "evidence_refs": ["ev-eaeu-reg"],
                },
            ],
            "commercial_signals": [
                {
                    "region": "RU",
                    "category": "procurement",
                    "summary": {"value": "Procurement for LP-001 Canon tablet 5 mg", "evidence_refs": ["ev-ru-com"]},
                    "evidence_refs": ["ev-ru-com"],
                }
            ],
            "product_contexts": [
                {"region": "RU", "dosage_forms": ["tablet"], "strengths": ["5 mg"], "evidence_refs": ["ev-ru-reg"]},
                {"region": "EAEU", "dosage_forms": ["tablet"], "strengths": ["5 mg"], "evidence_refs": ["ev-eaeu-reg"]},
            ],
            "patent_families": [],
            "evidence_registry": [
                {"evidence_id": "ev-ru-reg", "doc_id": "doc-ru-reg", "doc_kind": "ru_registration_export", "snippet": "LP-001 Canon tablet 5 mg active"},
                {"evidence_id": "ev-eaeu-reg", "doc_id": "doc-eaeu-reg", "doc_kind": "eaeu_document", "snippet": "LP-EAEU-1 Authorised Valid To: 2029-11-19 MAH (Holder): Lekpharm"},
                {"evidence_id": "ev-ru-com", "doc_id": "doc-ru-com", "doc_kind": "ru_procurement_summary", "snippet": "Procurement for LP-001 Canon tablet 5 mg"},
                {"evidence_id": "ev-ru-nohit", "doc_id": "doc-ru-nohit", "doc_kind": "ru_patent_fips", "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=RU | search_term=апиксабан | patents=0 | as_of=2025-10-30"},
                {"evidence_id": "ev-eaeu-nohit", "doc_id": "doc-eaeu-nohit", "doc_kind": "ru_patent_fips", "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=EAEU | search_term=апиксабан | patents=0 | as_of=2025-10-30"},
            ],
        }
        plan = ExecQuestionPlan(
            question_id="asset_attractiveness",
            answer_type="go_no_go",
            needed_dossier_sections=["registrations", "commercial_signals", "product_contexts", "patent_families"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["ru_registration_export", "eaeu_document", "ru_procurement_summary", "ru_patent_fips"],
                queries=["apixaban ru/eaeu entry"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        self.assertTrue(any(item["context"] == "RU" and item["source_class"] == "GRLS" for item in linkage["registration_identity_map"]))
        self.assertEqual(linkage["market_entry_linkage"]["RU"]["identity_match"], "same_identifier")
        self.assertEqual(
            linkage["ru_eaeu_ip_window_snapshot"]["conclusion"],
            "NO_LISTED_BLOCKING_PATENT_EVIDENCE",
        )
        self.assertEqual(linkage["generic_opportunity_by_region"]["RU"]["verdict"], "POTENTIAL_GO")
        self.assertEqual(linkage["generic_opportunity_by_region"]["EAEU"]["verdict"], "POTENTIAL_GO")
        self.assertEqual(linkage["registration_context_relationships"][0]["relationship"], "separate_product_contexts")

    def test_evidence_assembler_builds_rights_legal_events_and_fto_snapshots(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "ip_legal_window",
            "inn": "apixaban",
            "allowed_doc_kinds": [
                "patent_family_summary",
                "patent_legal_events",
                "patent_term_extension",
                "patent_file_wrapper",
                "patent_national_legal_status",
                "uspto_assignment",
                "sec_filing",
                "ru_patent_fips",
            ],
            "required_sections": ["patent_families"],
            "patent_families": [
                {
                    "family_id": "fam-us-eu-1",
                    "representative_pub": {"value": "US12345678", "evidence_refs": ["ev-family"]},
                    "what_blocks": {"value": "compound claims for apixaban", "evidence_refs": ["ev-family"]},
                    "technical_focus": {"value": "composition", "evidence_refs": ["ev-family"]},
                    "expiry_by_country": [
                        {"value": "US: 2027-01-01", "evidence_refs": ["ev-us-pte"]},
                        {"value": "EP: 2028-02-02", "evidence_refs": ["ev-eu-spc"]},
                    ],
                    "evidence_refs": ["ev-family"],
                }
            ],
            "evidence_registry": [
                {
                    "evidence_id": "ev-family",
                    "doc_id": "doc-family",
                    "doc_kind": "patent_family_summary",
                    "snippet": "US12345678 family contains compound claims for apixaban.",
                },
                {
                    "evidence_id": "ev-us-pte",
                    "doc_id": "doc-pte",
                    "doc_kind": "patent_term_extension",
                    "snippet": "US12345678 patent term extension PTE granted 2027-01-01.",
                },
                {
                    "evidence_id": "ev-us-td",
                    "doc_id": "doc-wrapper",
                    "doc_kind": "patent_file_wrapper",
                    "snippet": "US12345678 terminal disclaimer recorded 2024-01-15.",
                },
                {
                    "evidence_id": "ev-eu-spc",
                    "doc_id": "doc-spc",
                    "doc_kind": "patent_legal_events",
                    "snippet": "EP1234567 SPC granted 2028-02-02; opposition filed 2024-03-01.",
                },
                {
                    "evidence_id": "ev-us-assignment",
                    "doc_id": "doc-assignment",
                    "doc_kind": "uspto_assignment",
                    "snippet": "US12345678 assignment recorded 2024-02-02 Assignor A Assignee B.",
                },
                {
                    "evidence_id": "ev-sec-license",
                    "doc_id": "doc-sec",
                    "doc_kind": "sec_filing",
                    "snippet": "Exclusive license agreement territory US: licensee may sublicense, but may not assign without prior written consent.",
                },
                {
                    "evidence_id": "ev-eapo-nohit",
                    "doc_id": "doc-eapo-nohit",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=EAEU | search_term=апиксабан | patents=0 | as_of=2025-10-30",
                },
            ],
        }
        plan = ExecQuestionPlan(
            question_id="ip_legal_window",
            answer_type="window",
            needed_dossier_sections=["patent_families"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=[
                    "patent_family_summary",
                    "patent_legal_events",
                    "patent_term_extension",
                    "patent_file_wrapper",
                    "patent_national_legal_status",
                    "uspto_assignment",
                    "sec_filing",
                    "ru_patent_fips",
                ],
                queries=["apixaban patent legal events and rights"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        rights = linkage["rights_transferability_snapshot"]
        self.assertEqual(rights["conclusion"], "TRANSFERABILITY_TERMS_EVIDENCED")
        self.assertIn("assignment", rights["observed_record_types"])
        self.assertIn("license", rights["observed_record_types"])
        self.assertTrue(any(record["transferability"] == "prohibited" for record in rights["records"]))

        family_events = linkage["family_legal_events_snapshot"]
        self.assertIn("PTE", family_events["event_types_by_region"]["US"])
        self.assertIn("terminal_disclaimer", family_events["event_types_by_region"]["US"])
        self.assertIn("SPC", family_events["event_types_by_region"]["EU"])
        self.assertFalse(family_events["decision_grade"])

        fto = linkage["fto_screening_snapshot"]
        self.assertEqual(fto["screening_level"], "FTO_SCREENING_ONLY")
        self.assertTrue(fto["claim_scope_evidence_present"])
        self.assertFalse(fto["full_fto_verdict_allowed"])
        self.assertIn("FAMILY_LEGAL_EVENTS_INCOMPLETE", fto["decision_grade_blockers"])

        manifest = linkage["source_evidence_manifest"]
        self.assertGreaterEqual(manifest["checked_source_count"], 5)
        self.assertEqual(
            evidence_packet["evidence_packet_summary"]["contract_linkage_summary"]["rights_conclusion"],
            "TRANSFERABILITY_TERMS_EVIDENCED",
        )

    def test_evidence_assembler_parses_collector_legal_event_lines_and_retains_priority_manifest(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "ip_legal_window",
            "inn": "apixaban",
            "allowed_doc_kinds": ["patent_legal_events", "patent_expiry_us", "ru_patent_fips"],
            "required_sections": ["patent_families"],
            "patent_families": [
                {
                    "family_id": "fam-claim",
                    "representative_pub": {"value": "US11896586", "evidence_refs": ["ev-family"]},
                    "what_blocks": {"value": "compound claims for apixaban", "evidence_refs": ["ev-family"]},
                    "evidence_refs": ["ev-family"],
                }
            ],
            "evidence_registry": [
                {
                    "evidence_id": "ev-family",
                    "doc_id": "doc-family",
                    "doc_kind": "patent_family_summary",
                    "snippet": "US11896586 compound claims for apixaban.",
                },
                {
                    "evidence_id": "ev-ob",
                    "doc_id": "doc-ob",
                    "doc_kind": "patent_expiry_us",
                    "snippet": "\n".join(
                        [
                            "LEGAL_EVENT | source=fda_orange_book | jurisdiction=US | patent=US11896586 | event_type=expiry | event_date=2040-11-22 | status=Orange Book listed",
                            "LEGAL_EVENT | source=fda_orange_book_data_files | jurisdiction=US | event_type=regulatory_exclusivity | event_date=2028-10-17 | status=active_or_future Orange Book exclusivity | application=N202155 | product_no=001 | exclusivity_code=PED",
                        ]
                    ),
                },
                {
                    "evidence_id": "ev-epo",
                    "doc_id": "doc-epo",
                    "doc_kind": "patent_legal_events",
                    "snippet": "\n".join(
                        [
                            "LEGAL_EVENT | source=epo_register | jurisdiction=EU | patent=EP4412586 | event_type=EPIDOSNIGR3 | event_date= | status=EPIDOSNIGR3 | raw={\"date\":\"20260402\",\"text\":\"New entry: Payment of fee for grant\"}",
                            "LEGAL_EVENT | source=epo_register | jurisdiction=EU | patent=EP4353312 | event_type=expiry | event_date={'date': '2044-04-17', 'method': 'filing_plus_20y_no_pta'} | status=effective expiry reported",
                        ]
                    ),
                },
                {
                    "evidence_id": "ev-ru",
                    "doc_id": "doc-ru",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "LEGAL_EVENT | source=rospatent_searchplatform | jurisdiction=RU | patent=RU2819897 | event_type=ru_legal_status | event_date= | status=unknown | {\"doc_id\":\"RU2819897C1_20240528\",\"jurisdiction\":\"RU\",\"expiry_date\":\"2043-04-28\",\"legal_status\":\"unknown\",\"source\":\"rospatent_searchplatform\"}",
                },
                {
                    "evidence_id": "ev-ea",
                    "doc_id": "doc-ea",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "LEGAL_EVENT | source=rospatent_searchplatform | jurisdiction=EA | patent=EA0000037815 | event_type=ru_legal_status | event_date= | status=active | {\"doc_id\":\"EA0000037815B1_20210525\",\"jurisdiction\":\"EA\",\"expiry_date\":\"2037-06-19\",\"legal_status\":\"active\",\"source\":\"rospatent_searchplatform\"}",
                },
                {
                    "evidence_id": "ev-eapo-nohit",
                    "doc_id": "doc-eapo-nohit",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "OFFICIAL_PATENT_REGISTER_NO_HIT | region=EAEU | search_term=апиксабан | patents=0 | as_of=2025-10-30",
                },
            ],
        }
        plan = ExecQuestionPlan(
            question_id="ip_legal_window",
            answer_type="window",
            needed_dossier_sections=["patent_families"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["patent_legal_events", "patent_expiry_us", "ru_patent_fips"],
                queries=["apixaban collector legal events"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        family_events = linkage["family_legal_events_snapshot"]
        self.assertIn("expiry", family_events["event_types_by_region"]["US"])
        self.assertIn("regulatory_exclusivity", family_events["event_types_by_region"]["US"])
        self.assertIn("maintenance_fee", family_events["event_types_by_region"]["EU"])
        self.assertIn("ru_legal_status", family_events["event_types_by_region"]["RU"])
        self.assertIn("eaeu_pharma_register", family_events["event_types_by_region"]["EAEU"])

        patent_snapshot = linkage["patent_legal_status_snapshot"]["regions"]
        self.assertEqual(patent_snapshot["RU"]["conclusion"], "BLOCKING_OR_PENDING_EVIDENCE_PRESENT")
        self.assertEqual(patent_snapshot["EAEU"]["conclusion"], "BLOCKING_OR_PENDING_EVIDENCE_PRESENT")
        self.assertTrue(patent_snapshot["EAEU"]["official_no_hit_supported"])

        retention = linkage["priority_evidence_retention"]
        self.assertEqual(retention["status"], "ok")
        self.assertIn("ev-eapo-nohit", retention["retained_refs"])
        self.assertEqual(
            evidence_packet["evidence_packet_summary"]["contract_linkage_summary"]["priority_evidence_missing_count"],
            0,
        )

    def test_evidence_assembler_retains_clearance_checks_without_promoting_to_events(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "ip_legal_window",
            "inn": "apixaban",
            "allowed_doc_kinds": [
                "patent_term_extension",
                "patent_file_wrapper",
                "patent_national_legal_status",
                "patent_legal_events",
                "ru_patent_fips",
                "uspto_assignment",
            ],
            "required_sections": ["patent_families"],
            "patent_families": [
                {
                    "family_id": "fam-claim",
                    "representative_pub": {"value": "US11896586", "evidence_refs": ["ev-family"]},
                    "what_blocks": {"value": "compound claims for apixaban", "evidence_refs": ["ev-family"]},
                    "evidence_refs": ["ev-family"],
                }
            ],
            "evidence_registry": [
                {
                    "evidence_id": "ev-pte-check",
                    "doc_id": "doc-pte-check",
                    "doc_kind": "patent_term_extension",
                    "snippet": "CLEARANCE_CHECK | source=uspto_pte | jurisdiction=US | patent=US11896586 | check_class=PTE | status=no_public_listing_found | sources_checked=2",
                },
                {
                    "evidence_id": "ev-wrapper-limited",
                    "doc_id": "doc-wrapper-limited",
                    "doc_kind": "patent_file_wrapper",
                    "snippet": "CLEARANCE_CHECK | source=uspto_patent_center_file_wrapper | jurisdiction=US | patent=US11896586 | check_class=terminal_disclaimer_file_wrapper | status=source_not_collected | limitation=manual Patent Center review required",
                },
                {
                    "evidence_id": "ev-wrapper-extra",
                    "doc_id": "doc-wrapper-extra",
                    "doc_kind": "patent_file_wrapper",
                    "snippet": "\n".join(
                        [
                            "CLEARANCE_CHECK | source=uspto_maintenance_fees | jurisdiction=US | patent=US11896586 | check_class=maintenance_fee_status | status=not_source_verified",
                            "CLEARANCE_CHECK | source=uspto_ptab | jurisdiction=US | patent=US11896586 | check_class=ptab_reexam_reissue_review | status=not_source_verified",
                        ]
                    ),
                },
                {
                    "evidence_id": "ev-epo-check",
                    "doc_id": "doc-epo-check",
                    "doc_kind": "patent_legal_events",
                        "snippet": "CLEARANCE_CHECK | source=epo_register | jurisdiction=EU | patent=EP4353312 | check_class=SPC | status=no_source_event_found",
                },
                {
                    "evidence_id": "ev-eu-national",
                    "doc_id": "doc-eu-national",
                    "doc_kind": "patent_national_legal_status",
                    "snippet": "CLEARANCE_CHECK | source=eu_national_spc_registers | jurisdiction=EU | country=DE | patent=EP4353312 | check_class=SPC | status=not_source_verified",
                },
                {
                    "evidence_id": "ev-ru-conflict",
                    "doc_id": "doc-ru-conflict",
                    "doc_kind": "ru_patent_fips",
                    "snippet": "CLEARANCE_CHECK | source=ru_eaeu_conflict_classifier | jurisdiction=EAEU | check_class=eapo_term_no_hit_vs_fips_future_expiry | status=conflict_requires_review | conclusion=TERM_NO_HIT_NOT_FTO_CLEARANCE",
                },
                {
                    "evidence_id": "ev-rights-check",
                    "doc_id": "doc-rights-check",
                    "doc_kind": "uspto_assignment",
                    "snippet": "RIGHTS_CLEARANCE_CHECK | source=uspto_assignment | jurisdiction=US | patent=US11896586 | check_class=assignment_transferability | status=no_source_native_assignment_terms_found",
                },
            ],
        }
        plan = ExecQuestionPlan(
            question_id="ip_legal_window",
            answer_type="window",
            needed_dossier_sections=["patent_families"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=[
                    "patent_term_extension",
                    "patent_file_wrapper",
                    "patent_national_legal_status",
                    "patent_legal_events",
                    "ru_patent_fips",
                    "uspto_assignment",
                ],
                queries=["apixaban clearance checks"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        family_events = linkage["family_legal_events_snapshot"]
        self.assertNotIn("PTE", family_events.get("event_types_by_region", {}).get("US", []))
        self.assertIn("PTE", family_events["coverage_by_region"]["US"]["checked_missing_event_classes"])
        self.assertIn("terminal_disclaimer_file_wrapper", family_events["coverage_by_region"]["US"]["limited_event_checks"])
        self.assertIn("SPC", family_events["coverage_by_region"]["EU"]["checked_missing_event_classes"])

        rights = linkage["rights_transferability_snapshot"]
        self.assertEqual(rights["conclusion"], "SOURCE_CHECKS_ONLY_TERMS_MISSING")
        self.assertEqual(rights["record_count"], 0)
        self.assertEqual(rights["clearance_check_count"], 1)

        manifest = linkage["source_evidence_manifest"]
        statuses = {source["source_id"]: source["status"] for source in manifest["sources"]}
        self.assertEqual(statuses["uspto_pte_file_wrapper"], "limited")
        self.assertEqual(statuses["eu_national_registers"], "limited")
        self.assertEqual(statuses["rospatent_searchplatform"], "checked_with_conflict")
        self.assertGreaterEqual(manifest["limited_source_count"], 3)

        retention = linkage["priority_evidence_retention"]
        self.assertEqual(retention["status"], "ok")
        self.assertIn("ev-pte-check", retention["retained_refs"])
        self.assertIn("ev-rights-check", retention["retained_refs"])

    def test_evidence_assembler_retains_reimbursement_checks_for_market_window(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "market_reimbursement_window",
            "inn": "apixaban",
            "allowed_doc_kinds": ["pricing", "payer_policy"],
            "required_sections": ["registrations", "commercial_signals", "product_contexts"],
            "evidence_registry": [
                {
                    "evidence_id": "ev-ru-price",
                    "doc_id": "doc-ru-price",
                    "doc_kind": "pricing",
                    "snippet": (
                        "REIMBURSEMENT_CHECK | source=ru_minzdrav_public_price_limits | jurisdiction=RU | "
                        "check_class=jnvlp_price_limit_row | status=listed_active | inn=Апиксабан | "
                        "registry_entry=1000170708 | registration_id=ЛП-№(012345)-(РГ-RU) | effective_date=2026-02-17"
                    ),
                },
                {
                    "evidence_id": "ev-eaeu-scope",
                    "doc_id": "doc-eaeu-scope",
                    "doc_kind": "payer_policy",
                    "snippet": (
                        "REIMBURSEMENT_CHECK | source=eec_market_access_scope | jurisdiction=EAEU | "
                        "check_class=eaeu_union_reimbursement_scope | status=member_state_scope | "
                        "conclusion=NO_SINGLE_EAEU_UNION_REIMBURSEMENT_LIST_IDENTIFIED"
                    ),
                },
                {
                    "evidence_id": "ev-ru-policy",
                    "doc_id": "doc-ru-policy",
                    "doc_kind": "payer_policy",
                    "snippet": (
                        "REIMBURSEMENT_CHECK | source=ru_federal_program_sources | jurisdiction=RU | "
                        "check_class=federal_state_guarantees_reimbursement_pathway | "
                        "status=payer_pathway_source_checked | payer_signal=payer_pathway_scope"
                    ),
                },
            ],
        }
        plan = ExecQuestionPlan(
            question_id="market_reimbursement_window",
            answer_type="window",
            needed_dossier_sections=["registrations", "commercial_signals", "product_contexts"],
            retrieval_plan=ExecRetrievalPlan(doc_kinds=["pricing", "payer_policy"], queries=["apixaban reimbursement"]),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]

        self.assertEqual(linkage["market_reimbursement_snapshot"]["verdict_hint"], "LIMITED")
        self.assertEqual(linkage["market_reimbursement_snapshot"]["regions"]["RU"]["listed_active_count"], 1)
        self.assertEqual(linkage["market_reimbursement_snapshot"]["regions"]["RU"]["pathway_source_count"], 1)
        self.assertEqual(
            linkage["market_reimbursement_snapshot"]["regions"]["RU"]["conclusion"],
            "SOURCE_NATIVE_PRICE_ACCESS_AND_POLICY_BREADTH_SIGNALS_PRESENT",
        )
        self.assertTrue(linkage["market_reimbursement_snapshot"]["regions"]["EAEU"]["member_state_scope"])
        self.assertIn("ev-ru-price", linkage["priority_evidence_retention"]["retained_refs"])

    def test_verifier_lifts_market_reimbursement_from_unresolved_to_limited(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="market_reimbursement_window",
            title="Market / reimbursement window",
            verdict="UNRESOLVED",
            confidence="LOW",
            sufficiency="INSUFFICIENT",
            short_answer="No source-native payer evidence.",
            full_answer="The window is unresolved because no EAEU union reimbursement list is present.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
        )
        packet = {
            "evidence_registry": [
                {
                    "evidence_id": "ev-ru-price",
                    "doc_id": "doc-ru-price",
                    "doc_kind": "pricing",
                    "snippet": "REIMBURSEMENT_CHECK | source=ru_minzdrav_public_price_limits | jurisdiction=RU | status=listed_active",
                },
                {
                    "evidence_id": "ev-eaeu-scope",
                    "doc_id": "doc-eaeu-scope",
                    "doc_kind": "payer_policy",
                    "snippet": "REIMBURSEMENT_CHECK | source=eec_market_access_scope | jurisdiction=EAEU | status=member_state_scope",
                },
            ],
            "contract_linkage": {
                "market_reimbursement_snapshot": {
                    "verdict_hint": "LIMITED",
                    "evidence_refs": ["ev-ru-price", "ev-eaeu-scope"],
                    "regions": {
                        "RU": {
                            "listed_active_count": 1,
                            "current_effective_dates": ["2026-02-17"],
                            "evidence_refs": ["ev-ru-price"],
                        },
                        "EAEU": {
                            "member_state_scope": True,
                            "evidence_refs": ["ev-eaeu-scope"],
                        },
                    },
                }
            }
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "LIMITED")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertIn("RU source-native", repaired.short_answer)

    def test_verifier_downgrades_market_reimbursement_open_to_limited_without_payer_tier(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="market_reimbursement_window",
            title="Market / reimbursement window",
            verdict="OPEN",
            confidence="MEDIUM",
            sufficiency="SUFFICIENT",
            short_answer="RU access is open but direct payer tier and restrictions are not fully closed.",
            full_answer="The evidence supports JNVLP access, but payer tier and coverage breadth remain caveats.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "evidence_ids": ["ev-ru-price"],
            "contract_linkage": {
                "market_reimbursement_snapshot": {
                    "verdict_hint": "LIMITED",
                    "check_count": 4,
                    "evidence_refs": ["ev-ru-price"],
                    "regions": {
                        "RU": {
                            "listed_active_count": 1,
                            "pathway_source_count": 1,
                            "current_effective_dates": ["2026-02-27"],
                            "evidence_refs": ["ev-ru-price"],
                        },
                        "EAEU": {"member_state_scope": True},
                    },
                }
            }
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "LIMITED")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertIn("aligned_market_reimbursement_to_limited_screening_status", verification.repair_reason)

    def test_verifier_lifts_generic_not_evidenced_screening_partial_when_legal_sources_exist(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="generic_opportunity",
            title="Generic opportunity",
            verdict="NOT_EVIDENCED",
            confidence="LOW",
            sufficiency="INSUFFICIENT",
            short_answer="US/EU patent and legal-status evidence exists, but positive gates are not met.",
            full_answer="The packet has expiry/legal-status screening data, but PTE/SPC and legal-event reconciliation are incomplete.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "contract_linkage": {
                "source_evidence_manifest": {"checked_source_count": 4, "limited_source_count": 2},
                "family_legal_events_snapshot": {"coverage_status": "PARTIAL", "evidence_refs": ["ev-family"]},
            }
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "LOW")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertIn("lifted_generic_not_evidenced_to_screening_partial", verification.repair_reason)

    def test_verifier_lifts_sufficiency_to_screening_ready_partial(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="evidence_sufficiency_note",
            title="Evidence sufficiency note",
            verdict="INSUFFICIENT",
            confidence="LOW",
            sufficiency="INSUFFICIENT",
            short_answer="INSUFFICIENT because full FTO, file-wrapper, SPC, payer tier, and restriction evidence are not decision-grade.",
            full_answer=(
                "The packet is not operations-ready: US terminal disclaimer/file-wrapper and EU SPC country-level checks are incomplete, "
                "and payer tier/restriction evidence is still missing."
            ),
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "evidence_ids": ["ev-fto", "ev-family", "ev-payer"],
            "critical_unknowns": [],
            "registrations": [{"region": "RU", "status": {"value": "active"}}],
            "contract_linkage": {
                "registration_identity_map": [{"context": "RU", "identity_confidence": "HIGH"}],
                "source_evidence_manifest": {
                    "checked_source_count": 5,
                    "limited_source_count": 3,
                    "legal_event_evidence_refs": ["ev-family"],
                    "rights_evidence_refs": ["ev-fto"],
                },
                "fto_screening_snapshot": {
                    "screening_level": "FTO_SCREENING_ONLY",
                    "full_fto_verdict_allowed": False,
                    "evidence_refs": ["ev-fto"],
                },
                "family_legal_events_snapshot": {
                    "decision_grade": False,
                    "coverage_status": "PARTIAL",
                    "evidence_refs": ["ev-family"],
                },
                "market_reimbursement_snapshot": {
                    "verdict_hint": "LIMITED",
                    "check_count": 4,
                    "evidence_refs": ["ev-payer"],
                },
            },
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "PARTIAL")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertIn("screening-ready", repaired.short_answer)
        self.assertIn("operations-ready", " ".join(repaired.caveats).lower())
        self.assertIn("lifted_sufficiency_to_screening_ready_partial", verification.repair_reason)

    def test_market_entry_linkage_uses_product_context_form_and_strength_bridge(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "rf_entry",
            "allowed_doc_kinds": ["ru_registration_export", "ru_commercial_summary"],
            "required_sections": ["registrations", "commercial_signals", "product_contexts"],
            "registrations": [
                {
                    "region": "RU",
                    "status": {"value": "active", "evidence_refs": ["ev-reg"]},
                    "mah": {"value": "Holder A", "evidence_refs": ["ev-reg"]},
                    "identifiers": [{"value": "LP-777", "evidence_refs": ["ev-reg"]}],
                    "forms_strengths": [{"value": "film-coated tablet | 5 mg", "evidence_refs": ["ev-reg"]}],
                    "evidence_refs": ["ev-reg"],
                }
            ],
            "commercial_signals": [
                {
                    "region": "RU",
                    "category": "access",
                    "summary": {"value": "Regional access signal for film-coated tablets 5 mg in the same product context.", "evidence_refs": ["ev-com"]},
                    "evidence_refs": ["ev-com"],
                }
            ],
            "product_contexts": [
                {
                    "region": "RU",
                    "label": "Apixaban film-coated tablet 5 mg",
                    "dosage_forms": ["film-coated tablet"],
                    "strengths": ["5 mg"],
                    "evidence_refs": ["ev-reg"],
                }
            ],
            "evidence_registry": [
                {"evidence_id": "ev-reg", "doc_id": "doc-reg", "doc_kind": "ru_registration_export", "snippet": "LP-777 active Holder A film-coated tablet 5 mg"},
                {"evidence_id": "ev-com", "doc_id": "doc-com", "doc_kind": "ru_commercial_summary", "snippet": "Regional access signal for film-coated tablets 5 mg."},
            ],
        }
        plan = ExecQuestionPlan(
            question_id="rf_entry",
            answer_type="go_no_go",
            needed_dossier_sections=["registrations", "commercial_signals", "product_contexts"],
            retrieval_plan=ExecRetrievalPlan(
                doc_kinds=["ru_registration_export", "ru_commercial_summary"],
                queries=["apixaban commercial linkage"],
            ),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        ru_linkage = evidence_packet["contract_linkage"]["market_entry_linkage"]["RU"]

        self.assertEqual(ru_linkage["identity_match"], "mah_or_product_context")
        self.assertTrue(ru_linkage["product_context_match_confirmed"])

    def test_contract_linkage_uses_structured_registry_bridges_for_trace_sync(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        registration_id = "ЛП-№(007734)-(РГ-RU)"
        base_packet = {
            "block_id": "eaeu_entry",
            "inn": "apixaban",
            "allowed_doc_kinds": ["eaeu_document"],
            "required_sections": ["registrations", "product_contexts"],
            "registrations": [
                {
                    "region": "RU",
                    "status": {"value": "registered", "evidence_refs": ["ev-ru-reg"]},
                    "mah": {"value": "СООО \"Лекфарм\"", "evidence_refs": ["ev-ru-reg"]},
                    "identifiers": [{"value": registration_id, "evidence_refs": ["ev-ru-reg"]}],
                    "forms_strengths": [{"value": "tablet", "evidence_refs": ["ev-ru-reg"]}],
                    "evidence_refs": ["ev-ru-reg"],
                },
                {
                    "region": "EAEU",
                    "status": {"value": "Authorised", "evidence_refs": ["ev-eaeu-reg"]},
                    "mah": {"value": "СООО \"Лекфарм\"", "evidence_refs": ["ev-eaeu-reg"]},
                    "identifiers": [{"value": registration_id, "evidence_refs": ["ev-eaeu-reg"]}],
                    "forms_strengths": [{"value": "tablet", "evidence_refs": ["ev-eaeu-reg"]}],
                    "valid_to": {"value": "2029-11-19", "evidence_refs": ["ev-eaeu-reg"]},
                    "validity_type": "date_present",
                    "evidence_refs": ["ev-eaeu-reg"],
                },
            ],
            "product_contexts": [
                {
                    "region": "EAEU",
                    "label": "Apixaban tablet СООО \"Лекфарм\"",
                    "dosage_forms": ["tablet"],
                    "strengths": [],
                    "evidence_refs": ["ev-eaeu-reg"],
                }
            ],
            "evidence_registry": [
                {
                    "evidence_id": "ev-ru-reg",
                    "doc_id": "doc-ru-reg",
                    "doc_kind": "ru_registration_export",
                    "snippet": f"{registration_id} registered СООО \"Лекфарм\" tablet",
                },
                {
                    "evidence_id": "ev-eaeu-reg",
                    "doc_id": "doc-eaeu-reg",
                    "doc_kind": "eaeu_document",
                    "snippet": f"{registration_id} Authorised СООО \"Лекфарм\" Valid To: 2029-11-19 tablet",
                },
                {
                    "evidence_id": "ev-bridge",
                    "doc_id": "doc-bridge",
                    "doc_kind": "product_identity_bridge",
                    "snippet": (
                        f"PRODUCT_IDENTITY_BRIDGE | source=eaeu_product_identity_bridge | jurisdiction=EAEU | "
                        f"registration_id={registration_id} | trade_name=EAEU - tablet - СООО \"Лекфарм\" | "
                        "inn=apixaban | mah=СООО \"Лекфарм\" | form=tablet | status=Authorised | "
                        f"valid_to=2029-11-19 | linked_signal={registration_id} | match_level=exact\n"
                        f"COMMERCIAL_SIGNAL_LINKAGE | source=eaeu_product_identity_bridge | jurisdiction=EAEU | "
                        f"registration_id={registration_id} | linked_signal={registration_id} | "
                        "match_level=exact | signal_region=EAEU"
                    ),
                },
                {
                    "evidence_id": "ev-ru-price",
                    "doc_id": "doc-ru-price",
                    "doc_kind": "pricing",
                    "snippet": (
                        "REIMBURSEMENT_CHECK | source=ru_minzdrav_public_price_limits | jurisdiction=RU | "
                        f"status=listed_active | registration_id={registration_id} | inn=Апиксабан | "
                        "effective_date=2026-02-17"
                    ),
                },
                {
                    "evidence_id": "ev-us-legal",
                    "doc_id": "doc-us-legal",
                    "doc_kind": "patent_legal_events",
                    "snippet": "LEGAL_EVENT | jurisdiction=US | event_type=PTE | patent_no=US1234567 | event_date=2026-01-01",
                },
            ],
        }
        plan = ExecQuestionPlan(
            question_id="eaeu_entry",
            answer_type="go_no_go",
            needed_dossier_sections=["registrations", "product_contexts"],
            retrieval_plan=ExecRetrievalPlan(doc_kinds=["eaeu_document"], queries=["apixaban eaeu registration"]),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        linkage = evidence_packet["contract_linkage"]
        summary = evidence_packet["evidence_packet_summary"]["contract_linkage_summary"]

        self.assertEqual(linkage["market_entry_linkage"]["RU"]["identity_match"], "same_identifier")
        self.assertEqual(linkage["market_entry_linkage"]["EAEU"]["identity_match"], "same_identifier")
        self.assertEqual(linkage["market_entry_linkage"]["EAEU"]["commercial_signal_count"], 1)
        self.assertTrue(linkage["market_entry_linkage"]["RU"]["access_registration_id_overlap"])
        self.assertTrue(linkage["market_entry_linkage"]["EAEU"]["access_registration_id_overlap"])
        self.assertIn("structured_product_identity_bridge", linkage["market_entry_linkage"]["EAEU"]["identity_match_basis"])
        self.assertEqual(linkage["market_reimbursement_snapshot"]["verdict_hint"], "LIMITED")
        self.assertEqual(summary["ru_identity_match"], "same_identifier")
        self.assertEqual(summary["eaeu_identity_match"], "same_identifier")
        self.assertTrue(summary["ru_access_registration_id_overlap"])
        self.assertTrue(summary["eaeu_access_registration_id_overlap"])
        self.assertEqual(summary["market_reimbursement_verdict_hint"], "LIMITED")
        self.assertFalse(summary["family_legal_events_decision_grade"])
        self.assertEqual(summary["family_legal_events_coverage_status"], "PARTIAL")

    def test_verifier_lifts_eaeu_conditional_go_when_source_native_linkage_is_synced(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="eaeu_entry",
            title="EAEU entry",
            verdict="CONDITIONAL_GO",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="EAEU registration is present, but commercial identity linkage has caveats.",
            full_answer="EAEU entry is conditional because access signals are only linked at INN level.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "contract_linkage": {
                "registration_identity_map": [
                    {
                        "context": "EAEU",
                        "source_class": "EAEU-native",
                        "identity_confidence": "HIGH",
                        "status_positive": True,
                        "identifiers": ["ЛП-№(007734)-(РГ-RU)"],
                        "valid_to": "2029-11-19",
                        "validity_type": "date_present",
                        "evidence_refs": ["ev-eaeu-reg"],
                    }
                ],
                "market_entry_linkage": {
                    "EAEU": {
                        "registration_anchor_present": True,
                        "commercial_signal_count": 1,
                        "identity_match": "same_identifier",
                        "evidence_refs": ["ev-bridge"],
                    }
                },
            },
            "evidence_registry": [
                {
                    "evidence_id": "ev-eaeu-reg",
                    "doc_id": "doc-eaeu-reg",
                    "doc_kind": "eaeu_document",
                    "snippet": "ЛП-№(007734)-(РГ-RU) Authorised Valid To: 2029-11-19",
                },
                {
                    "evidence_id": "ev-bridge",
                    "doc_id": "doc-bridge",
                    "doc_kind": "product_identity_bridge",
                    "snippet": "COMMERCIAL_SIGNAL_LINKAGE | jurisdiction=EAEU | registration_id=ЛП-№(007734)-(РГ-RU) | match_level=exact",
                },
            ],
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertIn("promoted_eaeu_conditional_go_to_go_on_identity_linkage", verification.repair_reason)

    def test_verifier_lifts_eaeu_hold_when_validity_is_present_in_linkage_summary(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="eaeu_entry",
            title="EAEU entry",
            verdict="HOLD",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="HOLD because validity dates are not explicitly surfaced.",
            full_answer="The EAEU registration is authorised, but validity dates are missing from the answer text.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "eaeu_validity_missing",
                    "title": "Missing explicit validity dates",
                    "severity": "DECISION_BLOCKING",
                    "rationale": "Validity dates are missing for the matched EAEU registration.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "evidence_ids": ["ev-eaeu-reg", "ev-bridge"],
            "contract_linkage": {
                "market_entry_linkage": {
                    "EAEU": {
                        "commercial_signal_count": 1,
                        "identity_match": "same_identifier",
                        "evidence_refs": ["ev-bridge"],
                    }
                }
            },
            "evidence_packet_summary": {
                "contract_linkage_summary": {
                    "eaeu_identity_match": "same_identifier",
                    "eaeu_access_registration_id_overlap": True,
                    "eaeu_has_valid_to": True,
                    "eaeu_has_validity_state": True,
                    "eaeu_validity_types": ["date_present"],
                }
            },
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertFalse(repaired.decision_blockers)
        self.assertIn("eaeu_validity_understated_hold", verification.repair_reason)

    def test_verifier_lifts_eaeu_conditional_when_summary_linkage_is_complete(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="eaeu_entry",
            title="EAEU entry",
            verdict="CONDITIONAL_GO",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="Conditional because primary commercial source and FTO are still caveats.",
            full_answer="The EAEU record is authorised, but primary commercial source depth and patent/FTO execution risk keep it conditional.",
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "commercial_depth",
                    "title": "Primary commercial source not attached",
                    "severity": "IMPORTANT",
                    "rationale": "Primary commercial source is lighter than ideal.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "evidence_ids": ["ev-eaeu-reg", "ev-bridge"],
            "contract_linkage": {
                "market_entry_linkage": {
                    "EAEU": {
                        "commercial_signal_count": 1,
                        "identity_match": "same_identifier",
                        "evidence_refs": ["ev-bridge"],
                    }
                }
            },
            "evidence_packet_summary": {
                "contract_linkage_summary": {
                    "eaeu_identity_match": "same_identifier",
                    "eaeu_access_registration_id_overlap": True,
                    "eaeu_has_valid_to": True,
                    "eaeu_has_validity_state": True,
                    "market_reimbursement_verdict_hint": "LIMITED",
                }
            },
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertFalse(repaired.decision_blockers)
        self.assertIn("promoted_eaeu_conditional_go_from_summary_linkage", verification.repair_reason)

    def test_verifier_lifts_eaeu_hold_when_only_payer_policy_depth_is_missing(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="eaeu_entry",
            title="EAEU entry",
            verdict="HOLD",
            confidence="MEDIUM",
            sufficiency="PARTIAL",
            short_answer="EAEU authorization is valid, but direct payer/policy/access evidence is missing.",
            full_answer=(
                "The status is authorised, which supports current validity rather than an expired or withdrawn state. "
                "Commercial signals are non-negative overall, but a source-native payer/policy act is not fully closed."
            ),
            why_this_verdict=[],
            decision_blockers=[
                {
                    "blocker_id": "payer_depth",
                    "title": "Direct EAEU/RU payer-policy/access evidence is missing",
                    "severity": "IMPORTANT",
                    "rationale": "Payer depth is lighter than ideal.",
                    "evidence_refs": [],
                }
            ],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "evidence_ids": ["ev-eaeu-reg", "ev-bridge"],
            "contract_linkage": {
                "market_entry_linkage": {
                    "EAEU": {
                        "commercial_signal_count": 1,
                        "identity_match": "same_identifier",
                        "evidence_refs": ["ev-bridge"],
                    }
                }
            },
            "evidence_packet_summary": {
                "contract_linkage_summary": {
                    "eaeu_identity_match": "same_identifier",
                    "eaeu_structured_bridge_signal_count": 1,
                    "eaeu_has_valid_to": True,
                    "eaeu_has_validity_state": True,
                    "market_reimbursement_verdict_hint": "LIMITED",
                }
            },
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.verdict, "GO")
        self.assertEqual(repaired.sufficiency, "SUFFICIENT")
        self.assertIn("promoted_eaeu_conditional_go_from_summary_linkage", verification.repair_reason)

    def test_verifier_lifts_decision_blockers_sufficiency_to_screening_partial(self):
        verifier = ExecVerifier()
        block = ExecDecisionBlock(
            block_id="decision_blockers",
            title="Decision blockers",
            verdict="NOT_EVIDENCED",
            confidence="LOW",
            sufficiency="INSUFFICIENT",
            short_answer="No confirmed blocker classification because legal/FTO coverage is incomplete.",
            full_answer="Source-native IP records exist but operations-ready family reconciliation is incomplete.",
            why_this_verdict=[],
            decision_blockers=[],
            next_actions=[],
            caveats=[],
        )
        packet = {
            "contract_linkage": {
                "source_evidence_manifest": {"checked_source_count": 4, "limited_source_count": 2},
                "fto_screening_snapshot": {
                    "screening_level": "FTO_SCREENING_ONLY",
                    "evidence_refs": ["ev-fto"],
                },
                "family_legal_events_snapshot": {
                    "coverage_status": "PARTIAL",
                    "evidence_refs": ["ev-family"],
                },
            }
        }

        repaired, verification = verifier.verify_and_repair(block, packet, block_spec=None, allow_repair=True)

        self.assertEqual(verification.overall_status, "PASS")
        self.assertEqual(repaired.sufficiency, "PARTIAL")
        self.assertEqual(repaired.confidence, "MEDIUM")
        self.assertIn("lifted_decision_blockers_to_screening_partial", verification.repair_reason)

    def test_ru_market_linkage_can_match_source_native_portfolio_registration(self):
        assembler = ExecEvidenceAssembler(retriever=None)
        base_packet = {
            "block_id": "rf_entry",
            "inn": "apixaban",
            "allowed_doc_kinds": ["ru_registration_export"],
            "required_sections": ["registrations", "product_contexts"],
            "registrations": [
                {
                    "region": "RU",
                    "status": {"value": "active", "evidence_refs": ["ev-ru-reg-selected"]},
                    "mah": {"value": "Selected Holder", "evidence_refs": ["ev-ru-reg-selected"]},
                    "identifiers": [{"value": "LP-003276", "evidence_refs": ["ev-ru-reg-selected"]}],
                    "evidence_refs": ["ev-ru-reg-selected"],
                }
            ],
            "product_contexts": [
                {
                    "region": "RU",
                    "label": "Selected RU context",
                    "evidence_refs": ["ev-ru-reg-selected"],
                }
            ],
            "evidence_registry": [
                {
                    "evidence_id": "ev-ru-reg-selected",
                    "doc_id": "doc-ru-reg-selected",
                    "doc_kind": "ru_registration_export",
                    "snippet": "LP-003276 active Selected Holder",
                },
                {
                    "evidence_id": "ev-grls-alt",
                    "doc_id": "doc-grls-alt",
                    "doc_kind": "grls",
                    "snippet": "GRLS reg_no: LP-007734",
                },
                {
                    "evidence_id": "ev-ru-price-alt",
                    "doc_id": "doc-ru-price-alt",
                    "doc_kind": "pricing",
                    "snippet": (
                        "REIMBURSEMENT_CHECK | source=ru_minzdrav_public_price_limits | jurisdiction=RU | "
                        "status=listed_active | registration_id=LP-007734 | inn=Apixaban | effective_date=2026-02-17"
                    ),
                },
            ],
        }
        plan = ExecQuestionPlan(
            question_id="rf_entry",
            answer_type="go_no_go",
            needed_dossier_sections=["registrations", "product_contexts"],
            retrieval_plan=ExecRetrievalPlan(doc_kinds=["ru_registration_export"], queries=["apixaban ru entry"]),
        )

        evidence_packet = assembler.assemble(base_packet, plan, case_id="case-1", allow_retrieval=False)
        ru_linkage = evidence_packet["contract_linkage"]["market_entry_linkage"]["RU"]
        summary = evidence_packet["evidence_packet_summary"]["contract_linkage_summary"]

        self.assertEqual(ru_linkage["registration_identifiers"], ["LP-003276"])
        self.assertIn("LP-007734", ru_linkage["source_native_registration_identifiers"])
        self.assertEqual(ru_linkage["identity_match"], "same_identifier")
        self.assertEqual(ru_linkage["identity_match_scope"], "source_native_registered_product_context")
        self.assertFalse(ru_linkage["selected_registration_id_overlap"])
        self.assertTrue(ru_linkage["source_native_registration_id_overlap"])
        self.assertEqual(ru_linkage["matched_registration_identifiers"], ["LP-007734"])
        self.assertIn("source_native_registration_registry_overlap", ru_linkage["identity_match_basis"])
        self.assertEqual(summary["ru_identity_match"], "same_identifier")
        self.assertEqual(summary["ru_identity_match_scope"], "source_native_registered_product_context")
        self.assertFalse(summary["ru_selected_registration_id_overlap"])
        self.assertTrue(summary["ru_source_native_registration_id_overlap"])


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
