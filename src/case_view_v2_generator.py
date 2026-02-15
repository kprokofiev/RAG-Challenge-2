import json
import logging
import os
import time
import inspect
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.api_requests import APIProcessor
from src.case_view_schemas import (
    EvidenceLockedValue,
    InstructionHighlightsExtraction,
    PassportExtraction,
    PatentFamilyInsightExtraction,
    PublicationsExtraction,
    RegulatoryMarketExtraction,
    RuRegulatoryExtraction,
    SynthesisExtraction,
    TrialsExtraction,
)
from src.evidence_builder import EvidenceCandidatesBuilder
from src.prompts import build_system_prompt
from src.retrieval import HybridRetriever


logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_path(data: Any, path: List[Any]) -> Any:
    cur = data
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return None
            cur = cur[key]
            continue
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _pick_value(snapshot: Optional[dict], paths: List[List[Any]]) -> Tuple[Optional[Any], Optional[str]]:
    if not snapshot:
        return None, None
    for path in paths:
        val = _get_path(snapshot, path)
        if val is None or val == "" or val == []:
            continue
        json_path = "$." + ".".join(str(p) for p in path)
        return val, json_path
    return None, None


class CaseViewV2Generator:
    def __init__(self, documents_dir: Path, vector_db_dir: Path,
                 tenant_id: Optional[str] = None, case_id: Optional[str] = None,
                 retriever: Optional[HybridRetriever] = None,
                 api: Optional[APIProcessor] = None):
        self.documents_dir = documents_dir
        self.vector_db_dir = vector_db_dir
        self.tenant_id = tenant_id
        self.case_id = case_id
        self.retriever = retriever or HybridRetriever(vector_db_dir, documents_dir)
        self.api = api or APIProcessor(provider=os.getenv("DDKIT_LLM_PROVIDER", "openai"))
        self.answering_model = os.getenv("DDKIT_ANSWER_MODEL", None)
        self.evidence_builder = EvidenceCandidatesBuilder()

    def generate_case_view(self,
                           snapshot: Optional[dict] = None,
                           query: Optional[str] = None,
                           inn: Optional[str] = None,
                           use_web: bool = True,
                           use_snapshot: bool = True,
                           deadline: Optional[float] = None) -> Dict[str, Any]:
        start_ts = time.time()
        documents = self._load_documents_meta()
        doc_map = {d["doc_id"]: d for d in documents if d.get("doc_id")}

        inn_normalized = (inn or query or "").strip().lower()
        unknowns: List[Dict[str, Any]] = []

        passport = self.build_passport(snapshot, use_snapshot, inn_normalized, unknowns)
        regulatory = self.build_regulatory(snapshot, use_snapshot, unknowns)
        regulatory = self._enrich_regulatory_from_docs(regulatory, doc_map, unknowns, inn_normalized, use_web, deadline)
        passport = self._enrich_passport_from_regulatory(passport, regulatory, unknowns)
        # Fill missing passport fields from documents (US/EU) when possible.
        passport = self._enrich_passport_from_docs(passport, doc_map, unknowns, inn_normalized, use_web, deadline)
        clinical = self.build_clinical(snapshot, use_snapshot, unknowns)
        clinical = self._enrich_clinical_from_docs(clinical, doc_map, unknowns, inn_normalized, use_web, deadline)
        patents = self.build_patents(snapshot, use_snapshot, unknowns)
        patents = self._enrich_patents_from_docs(patents, doc_map, unknowns, inn_normalized, use_web, deadline)
        synthesis = self.build_synthesis(patents, unknowns)
        synthesis = self._enrich_synthesis_from_docs(synthesis, doc_map, unknowns, inn_normalized, use_web, deadline)
        sources = self.build_sources(snapshot if use_snapshot else None, documents, clinical=clinical, patents=patents)
        brief = self.build_brief(passport, regulatory, clinical, patents, synthesis, unknowns)

        unknowns = self._prune_unknowns(
            unknowns,
            passport=passport,
            regulatory=regulatory,
            clinical=clinical,
            patents=patents,
            synthesis=synthesis,
        )

        # Global deduplication of citations across all facts to reduce UI noise.
        self._global_dedupe_citations(
            passport=passport,
            brief=brief,
            regulatory=regulatory,
            clinical=clinical,
            patents=patents,
            synthesis=synthesis,
        )

        # UI helper: mark each block as ok/partial/empty so the frontend can show
        # "this section is incomplete" without guessing.
        self._apply_data_quality(
            passport=passport,
            brief=brief,
            regulatory=regulatory,
            clinical=clinical,
            patents=patents,
            synthesis=synthesis,
            sources=sources,
            unknowns=unknowns,
        )

        case_view = {
            "schema_version": "2.0",
            "case_id": self.case_id,
            "query": query or "",
            "inn_normalized": inn_normalized,
            "generated_at": _now_iso(),
            "passport": passport,
            "sections": {
                "brief": brief,
                "regulatory": regulatory,
                "clinical": clinical,
                "patents": patents,
                "synthesis": synthesis,
                "sources": sources,
                "unknowns": {"gaps": unknowns, "data_quality": ("ok" if not unknowns else "partial")},
            }
        }

        source_stats = self._build_source_stats(case_view, documents, unknowns)
        quality = self._evaluate_quality(case_view)
        source_stats["ready_for_ui"] = quality["ready_for_ui"]
        source_stats["quality_gates"] = quality["checks"]
        case_view["source_stats"] = source_stats

        logger.info(
            "Case view v2 generated in %.2fs (facts=%d, gaps=%d)",
            time.time() - start_ts,
            source_stats.get("facts_total", 0),
            source_stats.get("gaps_total", 0),
        )
        return case_view

    def _prune_unknowns(
        self,
        unknowns: List[Dict[str, Any]],
        passport: Dict[str, Any],
        regulatory: Dict[str, Any],
        clinical: Dict[str, Any],
        patents: Optional[Dict[str, Any]] = None,
        synthesis: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Remove gaps that were later filled during doc-based enrichment.

        We keep pruning rules simple and UI-oriented: if a section/field has evidence-backed data,
        we remove the corresponding gap entries.
        """
        if not unknowns:
            return []

        def has_fact(key: str) -> bool:
            return self._fact_has_citations(passport.get(key))

        def has_reg_market(market: str) -> bool:
            return bool(regulatory.get(market))

        def has_any_trials(bucket: str) -> bool:
            phases = clinical.get(bucket, {})
            if not isinstance(phases, dict):
                return False
            return any(bool(v) for v in phases.values())

        def has_pubmed() -> bool:
            pb = clinical.get("pubmed", {})
            if not isinstance(pb, dict):
                return False
            return any(bool(v) for v in pb.values())

        def has_patents() -> bool:
            if not patents:
                return False
            if patents.get("blocking_families"):
                return True
            views = patents.get("views") or {}
            if isinstance(views, dict) and views.get("blocking"):
                return True
            return bool(patents.get("families"))

        def has_synthesis_steps() -> bool:
            if not synthesis:
                return False
            return bool((synthesis.get("synthesis_route") or {}).get("steps"))

        seen_fields: set[str] = set()
        pruned: List[Dict[str, Any]] = []
        for gap in unknowns:
            field = str(gap.get("field") or "")
            if not field:
                continue
            # Deduplicate by field: keep first remaining entry only.
            if field in seen_fields:
                continue
            resolved = False

            if field.startswith("passport."):
                key = field.split(".", 1)[1]
                resolved = has_fact(key)
            elif field == "regulatory.us":
                resolved = has_reg_market("us")
            elif field == "regulatory.eu":
                resolved = has_reg_market("eu")
            elif field.startswith("regulatory.ru"):
                resolved = bool(regulatory.get("ru", {}).get("entries"))
            elif field == "regulatory.instructions_highlights":
                resolved = bool(regulatory.get("instructions_highlights"))
            elif field.startswith("clinical.global"):
                resolved = has_any_trials("global")
            elif field.startswith("clinical.ongoing"):
                resolved = has_any_trials("ongoing")
            elif field.startswith("clinical.ru"):
                resolved = has_any_trials("ru")
            elif field == "clinical":
                resolved = has_any_trials("global") or has_any_trials("ru") or has_any_trials("ongoing") or has_pubmed()
            elif field.startswith("clinical.pubmed"):
                resolved = has_pubmed()
            elif field.startswith("patents"):
                resolved = has_patents()
            elif field.startswith("synthesis.synthesis_route"):
                resolved = has_synthesis_steps()

            if resolved:
                continue
            seen_fields.add(field)
            pruned.append(gap)
        return pruned

    def build_passport(self, snapshot: Optional[dict], use_snapshot: bool,
                       inn_normalized: str, unknowns: List[Dict[str, Any]]) -> Dict[str, Any]:
        passport: Dict[str, Any] = {}
        snapshot_citations = []
        if use_snapshot and snapshot:
            snapshot_citations = [self._snapshot_citation("$.")]

        inn_value, inn_path = _pick_value(snapshot, [
            ["inn"], ["INN"], ["meta", "inn"], ["summary", "inn"], ["dossier_slice", "inn"]
        ])
        if inn_value:
            passport["inn"] = self._fact("МНН", inn_value, [self._snapshot_citation(inn_path)])
        elif inn_normalized:
            self._add_unknown(unknowns, "passport.inn", "no snapshot for INN", ["frontend_snapshot"])

        trade_names = self._filter_demo_market_map(self._collect_trade_names(snapshot))
        if trade_names:
            self._add_unknown(unknowns, "passport.trade_names", "snapshot-only (needs docs)", ["grls_card", "label", "epar"])
        else:
            self._add_unknown(unknowns, "passport.trade_names", "not found", ["grls_card", "label", "epar"])

        reg_map = self._detect_registration_markets(snapshot)
        if reg_map:
            passport["registered_in"] = self._fact("Где зарегистрирован", reg_map, snapshot_citations)
        else:
            self._add_unknown(unknowns, "passport.registered_in", "not found", ["regulatory_registry"])

        holders = self._filter_demo_market_map(self._collect_holders(snapshot))
        if holders:
            self._add_unknown(unknowns, "passport.registration_holders", "snapshot-only (needs docs)", ["regulatory_registry"])
        else:
            self._add_unknown(unknowns, "passport.registration_holders", "not found", ["regulatory_registry"])

        forms = self._collect_forms(snapshot)
        if forms:
            passport["dosage_forms"] = self._fact("Форма / путь введения", forms, snapshot_citations)

        strengths = self._collect_strengths(snapshot)
        if strengths:
            passport["dosages"] = self._fact("Дозировки (ключевые)", strengths, snapshot_citations)

        updated_at, updated_path = _pick_value(snapshot, [["generatedAt"], ["generated_at"], ["meta", "generated_at"]])
        if updated_at:
            passport["updated_at"] = self._fact("Обновлено", updated_at, [self._snapshot_citation(updated_path)])

        self._add_unknown(unknowns, "passport.fda_approval", "not found", ["approval_letter", "label"])

        formula_value, formula_path = _pick_value(snapshot, [
            ["chemical_formula"], ["chemicalFormula"], ["meta", "chemical_formula"]
        ])
        if formula_value:
            passport["chemical_formula"] = self._fact("Химическая формула", formula_value, [self._snapshot_citation(formula_path)])
        else:
            self._add_unknown(unknowns, "passport.chemical_formula", "not found", ["label", "scientific_pdf"])

        class_value, class_path = _pick_value(snapshot, [
            ["drug_class"], ["drugClass"], ["moa"], ["class"],
            ["summary", "class"], ["summary", "moa"]
        ])
        if class_value and not self._is_demo_text(str(class_value)):
            passport["drug_class"] = self._fact("Относится к классу (класс/MoA)", class_value, [self._snapshot_citation(class_path)])
        else:
            self._add_unknown(unknowns, "passport.drug_class", "not found", ["review_article", "label"])

        return passport

    def build_brief(self, passport: Dict[str, Any], regulatory: Dict[str, Any],
                    clinical: Dict[str, Any], patents: Dict[str, Any],
                    synthesis: Dict[str, Any], unknowns: List[Dict[str, Any]]) -> Dict[str, Any]:
        key_facts: List[Dict[str, Any]] = []

        for key in ("registered_in", "fda_approval", "trade_names"):
            fact = passport.get(key)
            if fact and self._fact_has_citations(fact):
                key_facts.append(fact)

        patent_wall_fact = self._build_patent_wall_fact(patents)
        if patent_wall_fact:
            key_facts.append(patent_wall_fact)

        # 1–2 key blocking families (what covers + summary), if available.
        for fam in (patents.get("blocking_families") or [])[:2]:
            if not isinstance(fam, dict):
                continue
            fam_cits = fam.get("citations") or []
            if not fam_cits:
                continue
            summary = fam.get("summary")
            if not summary:
                continue
            payload = {
                "family_id": fam.get("family_id"),
                "representative_doc": fam.get("representative_doc"),
                "what_covers": fam.get("coverage_type"),
                "summary": summary,
            }
            key_facts.append(self._fact("Ключевое патентное семейство", payload, fam_cits[:3]))

        synth_fact = self._build_synthesis_fact(synthesis)
        if synth_fact:
            key_facts.append(synth_fact)

        # One key clinical trial card (complete) as a brief fact.
        best_trial = None
        for bucket in ("global", "ru", "ongoing"):
            phases = clinical.get(bucket, {})
            if not isinstance(phases, dict):
                continue
            for trials in phases.values():
                for tr in _as_list(trials):
                    if not isinstance(tr, dict):
                        continue
                    if (
                        tr.get("study_type")
                        and tr.get("countries")
                        and tr.get("enrollment")
                        and tr.get("comparator")
                        and tr.get("regimen")
                        and tr.get("citations")
                    ):
                        best_trial = tr
                        break
                if best_trial:
                    break
            if best_trial:
                break
        if best_trial:
            value = {
                "trial_id": best_trial.get("trial_id"),
                "title": best_trial.get("title"),
                "phase": best_trial.get("phase"),
                "study_type": best_trial.get("study_type"),
            }
            key_facts.append(self._fact("Ключевое исследование", value, (best_trial.get("citations") or [])[:3]))

        # Generate narrative summary using LLM instead of simple concatenation
        summary_text = self._generate_narrative_summary(passport, regulatory, clinical, patents, synthesis)

        if not summary_text or len(summary_text.strip()) < 50:
            # Fallback to simple lines if LLM fails
            summary_lines = self._build_summary_lines(passport, regulatory, clinical, patents, synthesis)
            summary_text = "\n".join(summary_lines[:6]) if summary_lines else ""

        if not summary_text:
            self._add_unknown(unknowns, "brief.summary_text", "not enough evidence", ["structured_sources"])

        return {
            "summary_text": summary_text,
            "key_facts": key_facts[:15],
        }

    def build_regulatory(self, snapshot: Optional[dict], use_snapshot: bool,
                         unknowns: List[Dict[str, Any]]) -> Dict[str, Any]:
        regulatory: Dict[str, Any] = {"us": {}, "eu": {}, "ru": {}, "instructions_highlights": []}
        if not use_snapshot or not snapshot:
            self._add_unknown(unknowns, "regulatory.ru", "snapshot missing", ["frontend_snapshot"])
            return regulatory

        # Use snapshot as a baseline for US/EU if the frontend snapshot contains anything for those markets.
        # RU remains "truth" via ruSections.
        us_from_snapshot = self._market_regulatory_from_snapshot(snapshot, "us")
        if us_from_snapshot:
            regulatory["us"] = us_from_snapshot
        eu_from_snapshot = self._market_regulatory_from_snapshot(snapshot, "eu")
        if eu_from_snapshot:
            regulatory["eu"] = eu_from_snapshot

        ru_entries = self._collect_ru_reg_entries(snapshot)
        if ru_entries:
            regulatory["ru"]["entries"] = ru_entries
        else:
            self._add_unknown(unknowns, "regulatory.ru.entries", "not found", ["grls_card"])

        return regulatory

    def _market_regulatory_from_snapshot(self, snapshot: dict, market: str) -> Dict[str, Any]:
        """
        Best-effort extraction of US/EU regulatory fields from the frontend snapshot.

        This is a fallback/baseline: if we later attach primary documents (label/EPAR/SmPC/etc),
        doc-based enrichment will replace or fill missing fields with document-backed citations.
        """
        market = (market or "").strip().lower()
        if market not in {"us", "eu"}:
            return {}

        out: Dict[str, Any] = {}

        # Trade names
        tn_value, tn_path = _pick_value(snapshot, [
            ["regulatory", market, "trade_names"],
            ["regulatory", market, "tradeNames"],
            ["regulatory", market, "brands"],
            ["regulatory", market, "brand_names"],
        ])
        if tn_value:
            tn_values = self._filter_demo_strings(tn_value)
            if tn_values:
                out["trade_names"] = self._fact("Торговые названия", tn_values, [self._snapshot_citation(tn_path)])

        # Holders (MAH/applicant/etc)
        h_value, h_path = _pick_value(snapshot, [
            ["regulatory", market, "holders"],
            ["regulatory", market, "holder"],
            ["regulatory", market, "mah"],
            ["regulatory", market, "registration_holders"],
        ])
        if h_value:
            h_values = self._filter_demo_strings(h_value)
            if h_values:
                out["holders"] = self._fact("Держатель регистрации", h_values, [self._snapshot_citation(h_path)])

        # Dosage forms + strengths (free-form)
        dfs_value, dfs_path = _pick_value(snapshot, [
            ["regulatory", market, "dosage_forms_and_strengths"],
            ["regulatory", market, "dosageFormsAndStrengths"],
            ["regulatory", market, "dosage_forms"],
            ["regulatory", market, "forms"],
            ["regulatory", market, "strengths"],
        ])
        if dfs_value:
            out["dosage_forms_and_strengths"] = self._fact("Дозировки/формы", dfs_value, [self._snapshot_citation(dfs_path)])

        # Status: prefer approval-like field if present (contains date/summary), else fallback to status.
        s_value, s_path = _pick_value(snapshot, [
            ["regulatory", market, "approval"],
            ["regulatory", market, "fda_approval"],
            ["regulatory", market, "status"],
        ])
        if s_value and not self._is_demo_text(str(s_value)):
            out["status"] = self._fact("Статус", s_value, [self._snapshot_citation(s_path)])

        # EU coverage (if present)
        if market == "eu":
            c_value, c_path = _pick_value(snapshot, [
                ["regulatory", "eu", "countries_covered"],
                ["regulatory", "eu", "countriesCovered"],
                ["regulatory", "eu", "countries"],
                ["regulatory", "eu", "eea_countries"],
            ])
            if c_value:
                out["countries_covered"] = self._fact("Страны покрытия", c_value, [self._snapshot_citation(c_path)])

        return out

    def build_clinical(self, snapshot: Optional[dict], use_snapshot: bool,
                       unknowns: List[Dict[str, Any]]) -> Dict[str, Any]:
        clinical = {
            "global": self._empty_phase_map(),
            "ru": self._empty_phase_map(),
            "ongoing": self._empty_phase_map(),
            "pubmed": {
                "comparative": [],
                "abstracts": [],
                "real_world": [],
                "combination": []
            }
        }

        if not use_snapshot or not snapshot:
            self._add_unknown(unknowns, "clinical", "snapshot missing", ["frontend_snapshot"])
            return clinical

        ru_trials = self._collect_ru_clinical_trials(snapshot)
        for trial in ru_trials:
            phase_key = self._normalize_phase(trial.get("phase"))
            clinical["ru"].setdefault(phase_key, []).append(trial)

        if not ru_trials:
            self._add_unknown(unknowns, "clinical.ru", "not found", ["ru_clinical_permission"])

        return clinical

    # -------------------------
    # Stage 1–3: Enrichers (docs)
    # -------------------------

    def _enrich_passport_from_docs(
        self,
        passport: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
        unknowns: List[Dict[str, Any]],
        inn_normalized: str,
        use_web: bool,
        deadline: Optional[float],
    ) -> Dict[str, Any]:
        """
        Stage 1: citations infrastructure + minimal passport enrichment from documents.
        We only fill fields that are missing OR have no citations.
        """
        if not use_web:
            return passport

        # We keep RU snapshot as "truth". If snapshot already provided the value, we do not override it.
        # We only attempt to fill missing values.
        needs = []
        if not self._fact_has_non_snapshot_citations(passport.get("fda_approval")):
            needs.append("fda_approval")
        if not self._fact_has_citations(passport.get("chemical_formula")):
            needs.append("chemical_formula")
        if not self._fact_has_citations(passport.get("drug_class")):
            needs.append("drug_class")

        if not needs:
            return passport

        # One LLM call to extract all missing passport fields from the most relevant docs.
        task = (
            "Extract the missing passport fields for the drug case view.\n"
            f"INN: {inn_normalized}\n"
            "Return ONLY fields that you can support with the provided evidence candidates.\n"
            "Fields needed:\n"
            f"- {', '.join(needs)}\n"
            "For fda_approval: include approval date (if present) + short what was approved (indication).\n"
            "For chemical_formula: return the chemical formula string.\n"
            "For drug_class: return class + MoA (short)."
        )

        # Prefer primary regulatory docs and authoritative pages
        doc_kinds = [
            "label",
            "approval_letter",
            "us_fda",
            "smpc",
            "epar",
            "assessment_report",
            "review_article",
            "moa_overview",
            "scientific_pdf",
            "scientific_article",
            "publication",
            "journal_page",
            "drug_monograph",
            "company_pdf",
            "grls_card",
            "grls",
            "ru_instruction",
            "eaeu_document",
        ]

        extracted, cand_map = self._extract_structured_from_docs(
            task=task,
            response_format=PassportExtraction,
            doc_map=doc_map,
            doc_kinds=doc_kinds,
            top_n=18,
            deadline=deadline,
        )

        if not extracted or not cand_map:
            for field in needs:
                self._add_unknown(unknowns, f"passport.{field}", "not found in documents", doc_kinds)
            return passport

        if "fda_approval" in needs and extracted.get("fda_approval"):
            fact = self._value_to_fact("Одобрение FDA", extracted["fda_approval"], cand_map, doc_map)
            if fact:
                passport["fda_approval"] = fact
        if "chemical_formula" in needs and extracted.get("chemical_formula"):
            fact = self._value_to_fact("Химическая формула", extracted["chemical_formula"], cand_map, doc_map)
            if fact:
                passport["chemical_formula"] = fact
        if "drug_class" in needs and extracted.get("drug_class"):
            fact = self._value_to_fact("Относится к классу (класс/MoA)", extracted["drug_class"], cand_map, doc_map)
            if fact:
                passport["drug_class"] = fact

        # If still missing, add unknowns.
        for field in needs:
            if not self._fact_has_citations(passport.get(field)):
                self._add_unknown(unknowns, f"passport.{field}", "not found", doc_kinds)

        return passport

    def _enrich_regulatory_from_docs(
        self,
        regulatory: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
        unknowns: List[Dict[str, Any]],
        inn_normalized: str,
        use_web: bool,
        deadline: Optional[float],
    ) -> Dict[str, Any]:
        """
        Stage 2: fill US/EU regulatory sections and instruction highlights from documents.
        """
        if not use_web:
            return regulatory

        def has_any_market_fact(market: str) -> bool:
            data = regulatory.get(market) or {}
            if not isinstance(data, dict):
                return False
            for v in data.values():
                if isinstance(v, dict) and self._fact_has_citations(v):
                    return True
            return False

        def merge_market(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            """
            Prefer document-backed facts (incoming) over snapshot-backed facts (existing),
            but keep snapshot values when the doc extractor didn't return a field.
            """
            if not isinstance(existing, dict):
                existing = {}
            if not isinstance(incoming, dict):
                return existing
            out = dict(existing)
            for k, v in incoming.items():
                if isinstance(v, dict) and self._fact_has_citations(v):
                    out[k] = v
                elif k not in out:
                    out[k] = v
            return out

        # US
        us_task = (
            "Extract US regulatory facts for the drug case view.\n"
            f"INN: {inn_normalized}\n"
            "Return these fields only if supported by evidence:\n"
            "- trade_names (brand/trade names)\n"
            "- holders (marketing authorization holder / applicant / sponsor as stated)\n"
            "- dosage_forms_and_strengths (dosage forms + strengths)\n"
            "- status (approval status incl. approval date and indication summary if present)\n"
        )
        us_doc_kinds = [
            "label",
            "approval_letter",
            "us_fda",
            "complete_response_letter",
            "anda_package",
        ]
        us_extracted, us_cands = self._extract_structured_from_docs(
            task=us_task,
            response_format=RegulatoryMarketExtraction,
            doc_map=doc_map,
            doc_kinds=us_doc_kinds,
            top_n=18,
            deadline=deadline,
        )
        if us_extracted and us_cands:
            incoming = self._market_regulatory_from_extraction("us", us_extracted, us_cands, doc_map)
            regulatory["us"] = merge_market(regulatory.get("us") or {}, incoming)
        else:
            # If snapshot already has something for US, don't mark as unknown.
            if not has_any_market_fact("us"):
                self._add_unknown(unknowns, "regulatory.us", "not found", us_doc_kinds)

        # EU
        eu_task = (
            "Extract EU regulatory facts for the drug case view.\n"
            f"INN: {inn_normalized}\n"
            "Return these fields only if supported by evidence:\n"
            "- status (authorization/approval status)\n"
            "- countries_covered (EEA/EU coverage if explicitly stated)\n"
            "- trade_names (brand/trade names in EU)\n"
            "- holders (MAH)\n"
            "- dosage_forms_and_strengths\n"
        )
        eu_doc_kinds = ["epar", "smpc", "pil", "assessment_report"]
        eu_extracted, eu_cands = self._extract_structured_from_docs(
            task=eu_task,
            response_format=RegulatoryMarketExtraction,
            doc_map=doc_map,
            doc_kinds=eu_doc_kinds,
            top_n=18,
            deadline=deadline,
        )
        if eu_extracted and eu_cands:
            incoming = self._market_regulatory_from_extraction("eu", eu_extracted, eu_cands, doc_map)
            regulatory["eu"] = merge_market(regulatory.get("eu") or {}, incoming)
        else:
            if not has_any_market_fact("eu"):
                self._add_unknown(unknowns, "regulatory.eu", "not found", eu_doc_kinds)

        # RU (GRLS / EAEU)
        ru_task = (
            "Extract RU regulatory facts for the drug case view (GRLS / EAEU registry).\n"
            f"INN: {inn_normalized}\n"
            "Return a list of registration entries. For EACH entry provide:\n"
            "- trade_name (brand/trade name)\n"
            "- holder (marketing authorization holder)\n"
            "- reg_number (registration number)\n"
            "- reg_date (registration date if present)\n"
            "- dosage_forms (dosage forms/strengths as stated)\n"
            "- status (active/suspended/withdrawn/expired if stated)\n"
            "Use only evidence_ids from candidates. Leave empty if not supported."
        )
        ru_doc_kinds = ["grls_card", "grls", "ru_instruction", "ru_quality_letter", "eaeu_document"]
        ru_extracted, ru_cands = self._extract_structured_from_docs(
            task=ru_task,
            response_format=RuRegulatoryExtraction,
            doc_map=doc_map,
            doc_kinds=ru_doc_kinds,
            top_n=20,
            deadline=deadline,
        )
        if ru_extracted and ru_cands:
            entries_out: List[Dict[str, Any]] = []
            for entry in ru_extracted.get("entries") or []:
                if not isinstance(entry, dict):
                    continue
                out_entry: Dict[str, Any] = {}
                tn = self._value_to_fact("Торговое название", entry.get("trade_name"), ru_cands, doc_map)
                holder = self._value_to_fact("Держатель регистрации", entry.get("holder"), ru_cands, doc_map)
                reg_no = self._value_to_fact("Рег. номер", entry.get("reg_number"), ru_cands, doc_map)
                reg_date = self._value_to_fact("Дата регистрации", entry.get("reg_date"), ru_cands, doc_map)
                forms = self._value_to_fact("Формы/дозировки", entry.get("dosage_forms"), ru_cands, doc_map)
                status = self._value_to_fact("Статус", entry.get("status"), ru_cands, doc_map)
                if tn:
                    out_entry["trade_name"] = tn
                if holder:
                    out_entry["holder"] = holder
                if reg_no:
                    out_entry["reg_no"] = reg_no
                    out_entry["reg_number"] = reg_no
                if reg_date:
                    out_entry["reg_date"] = reg_date
                if forms:
                    out_entry["forms"] = forms
                    out_entry["dosage_forms"] = forms
                if status:
                    out_entry["status"] = status
                if out_entry:
                    entries_out.append(out_entry)

            if entries_out:
                def fact_value(fact: Any) -> Optional[str]:
                    if isinstance(fact, dict):
                        val = fact.get("value")
                        if isinstance(val, str):
                            return val.strip()
                    return None

                def entry_key(entry: Dict[str, Any]) -> str:
                    for key in ("reg_no", "reg_number"):
                        v = fact_value(entry.get(key))
                        if v:
                            return v.lower()
                    v = fact_value(entry.get("trade_name"))
                    if v:
                        return v.lower()
                    return ""

                def merge_entry(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
                    out = dict(existing)
                    for field in ("trade_name", "holder", "reg_no", "reg_number", "reg_date", "forms", "dosage_forms", "status"):
                        inc = incoming.get(field)
                        if inc is None:
                            continue
                        if self._fact_has_non_snapshot_citations(inc) or not self._fact_has_citations(out.get(field)):
                            out[field] = inc
                    return out

                existing_entries = _as_list((regulatory.get("ru") or {}).get("entries"))
                merged_entries: List[Dict[str, Any]] = []
                seen = {}
                for entry in existing_entries:
                    if not isinstance(entry, dict):
                        continue
                    key = entry_key(entry)
                    if key:
                        seen[key] = entry
                    merged_entries.append(entry)

                for entry in entries_out:
                    key = entry_key(entry)
                    if key and key in seen:
                        for idx, existing in enumerate(merged_entries):
                            if entry_key(existing) == key:
                                merged_entries[idx] = merge_entry(existing, entry)
                                break
                    else:
                        merged_entries.append(entry)
                        if key:
                            seen[key] = entry

                regulatory.setdefault("ru", {})
                regulatory["ru"]["entries"] = merged_entries
            else:
                if not (regulatory.get("ru") or {}).get("entries"):
                    self._add_unknown(unknowns, "regulatory.ru.entries", "not found", ru_doc_kinds)
        else:
            if not (regulatory.get("ru") or {}).get("entries"):
                self._add_unknown(unknowns, "regulatory.ru.entries", "not found", ru_doc_kinds)

        # Instructions highlights (RU instruction / SmPC / label)
        instr_task = (
            "Extract instruction/label highlights for the drug case view.\n"
            f"INN: {inn_normalized}\n"
            "Return short bullet-like items, each supported by evidence.\n"
            "Target volume: 8-12 items total (balanced across categories).\n"
            "We need:\n"
            "- 3-5 indications\n"
            "- 3-5 dosing / regimen items\n"
            "- 2-4 important restrictions / warnings\n"
            "Do NOT include a separate 'safety monitoring' block."
        )
        instr_doc_kinds = ["ru_instruction", "grls_card", "grls", "label", "smpc", "pil"]
        instr_extracted, instr_cands = self._extract_structured_from_docs(
            task=instr_task,
            response_format=InstructionHighlightsExtraction,
            doc_map=doc_map,
            doc_kinds=instr_doc_kinds,
            top_n=30,
            deadline=deadline,
        )
        highlights: List[Dict[str, Any]] = []
        if instr_extracted and instr_cands:
            for it in (instr_extracted.get("indications") or [])[:5]:
                f = self._value_to_fact("Показания", it, instr_cands, doc_map)
                if f:
                    highlights.append(f)
            for it in (instr_extracted.get("dosing") or [])[:5]:
                f = self._value_to_fact("Дозирование / режим применения", it, instr_cands, doc_map)
                if f:
                    highlights.append(f)
            for it in (instr_extracted.get("restrictions") or [])[:4]:
                f = self._value_to_fact("Ограничения / предупреждения", it, instr_cands, doc_map)
                if f:
                    highlights.append(f)

        if highlights:
            regulatory["instructions_highlights"] = highlights[:12]
            # Warn if we got fewer than 8 items (target is 8-12 balanced across categories)
            if len(highlights) < 8:
                self._add_unknown(
                    unknowns,
                    "regulatory.instructions_highlights",
                    f"only {len(highlights)} items extracted (target 8-12)",
                    instr_doc_kinds,
                )
        else:
            # Keep existing gap entry (or add a more specific one if missing)
            if not regulatory.get("instructions_highlights"):
                self._add_unknown(unknowns, "regulatory.instructions_highlights", "not found", instr_doc_kinds)

        return regulatory

    def _enrich_passport_from_regulatory(
        self,
        passport: Dict[str, Any],
        regulatory: Dict[str, Any],
        unknowns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Passport fields should not be demo placeholders. When we have document-backed regulatory facts,
        we can re-use them to fill passport trade names / holders / FDA approval.
        """

        def fact_value(f: Any) -> Any:
            if isinstance(f, dict):
                return f.get("value")
            return None

        def fact_citations(f: Any) -> List[Dict[str, Any]]:
            if isinstance(f, dict) and isinstance(f.get("citations"), list):
                return [c for c in f.get("citations") or [] if isinstance(c, dict)]
            return []

        def non_snapshot_citations(cits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [c for c in cits if c.get("structured_source") != "frontend_snapshot"]

        # Trade names by market
        tn_map: Dict[str, List[str]] = {"ru": [], "eu": [], "us": []}
        tn_cits: List[Dict[str, Any]] = []

        # RU (from snapshot entries)
        ru = regulatory.get("ru") or {}
        if isinstance(ru, dict):
            for entry in ru.get("entries") or []:
                if not isinstance(entry, dict):
                    continue
                t_fact = entry.get("trade_name")
                vals = self._filter_demo_strings(fact_value(t_fact))
                ru_cits = fact_citations(t_fact)
                if vals and ru_cits:
                    for v in vals:
                        if v not in tn_map["ru"]:
                            tn_map["ru"].append(v)
                    tn_cits.extend(ru_cits)

        # US/EU (from doc extraction)
        for market in ("us", "eu"):
            block = regulatory.get(market) or {}
            if not isinstance(block, dict):
                continue
            t_fact = block.get("trade_names")
            vals = self._filter_demo_strings(fact_value(t_fact))
            market_cits = non_snapshot_citations(fact_citations(t_fact))
            if vals and market_cits:
                for v in vals:
                    if v not in tn_map[market]:
                        tn_map[market].append(v)
                tn_cits.extend(market_cits)

        if any(tn_map.values()) and tn_cits:
            passport["trade_names"] = self._fact("ТН", tn_map, self._dedupe_citations(tn_cits)[:5])

        # Registration holders by market
        h_map: Dict[str, List[str]] = {"ru": [], "eu": [], "us": []}
        h_cits: List[Dict[str, Any]] = []

        if isinstance(ru, dict):
            for entry in ru.get("entries") or []:
                if not isinstance(entry, dict):
                    continue
                h_fact = entry.get("holder")
                vals = self._filter_demo_strings(fact_value(h_fact))
                ru_cits = fact_citations(h_fact)
                if vals and ru_cits:
                    for v in vals:
                        if v not in h_map["ru"]:
                            h_map["ru"].append(v)
                    h_cits.extend(ru_cits)

        for market in ("us", "eu"):
            block = regulatory.get(market) or {}
            if not isinstance(block, dict):
                continue
            h_fact = block.get("holders")
            vals = self._filter_demo_strings(fact_value(h_fact))
            market_cits = non_snapshot_citations(fact_citations(h_fact))
            if vals and market_cits:
                for v in vals:
                    if v not in h_map[market]:
                        h_map[market].append(v)
                h_cits.extend(market_cits)

        if any(h_map.values()) and h_cits:
            passport["registration_holders"] = self._fact("Владельцы регистрации", h_map, self._dedupe_citations(h_cits)[:5])

        # Registered in: derive from available regulatory facts if missing.
        if not self._fact_has_citations(passport.get("registered_in")):
            reg_map: Dict[str, str] = {}
            reg_cits: List[Dict[str, Any]] = []

            if isinstance(ru, dict) and (ru.get("entries") or []):
                reg_map["ru"] = "active"
                for entry in ru.get("entries") or []:
                    if not isinstance(entry, dict):
                        continue
                    for key in ("trade_name", "holder", "reg_no", "reg_number", "forms", "dosage_forms", "status"):
                        reg_cits.extend(fact_citations(entry.get(key)))

            for market in ("us", "eu"):
                block = regulatory.get(market) or {}
                if not isinstance(block, dict):
                    continue
                if any(self._fact_has_citations(v) for v in block.values() if isinstance(v, dict)):
                    reg_map[market] = "active"
                    for v in block.values():
                        if isinstance(v, dict):
                            reg_cits.extend(fact_citations(v))

            if reg_map and reg_cits:
                passport["registered_in"] = self._fact("Где зарегистрирован", reg_map, self._dedupe_citations(reg_cits)[:5])

        # FDA approval: reuse US status if it contains approval date/summary.
        if not self._fact_has_citations(passport.get("fda_approval")):
            us = regulatory.get("us") or {}
            if isinstance(us, dict):
                st = us.get("status")
                st_cits = non_snapshot_citations(fact_citations(st))
                if st_cits:
                    passport["fda_approval"] = self._fact("Одобрение FDA", fact_value(st), st_cits[:3])
                else:
                    self._add_unknown(unknowns, "passport.fda_approval", "not found", ["approval_letter", "label", "us_fda"])

        return passport

    def _enrich_clinical_from_docs(
        self,
        clinical: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
        unknowns: List[Dict[str, Any]],
        inn_normalized: str,
        use_web: bool,
        deadline: Optional[float],
    ) -> Dict[str, Any]:
        """
        Stage 3: extract global and ongoing clinical trials and a minimal publications (pubmed-style) block.
        """
        if not use_web:
            return clinical

        # 3.1 Global + ongoing trials from registries (CT.gov, CTIS, etc.)
        trials_task = (
            "Extract clinical trials for the drug case view.\n"
            f"INN: {inn_normalized}\n"
            "Return a list of trials mentioned in the evidence.\n"
            "For EACH trial provide:\n"
            "- trial_id (NCT/CTIS/registry identifier)\n"
            "- title\n"
            "- phase\n"
            "- study_type (randomized / non-randomized / observational / etc.)\n"
            "- countries (list)\n"
            "- enrollment (patients number)\n"
            "- comparator\n"
            "- regimen (therapy + dosing)\n"
            "- status (completed/recruiting/etc.)\n"
            "- efficacy_key_points (if available)\n"
            "- conclusion (1-2 sentences): always provide a short conclusion based on evidence.\n"
            "  If the trial is ongoing/recruiting and no results are available, say so explicitly.\n"
            "- where_conducted (site/country summary if present)\n"
            "Use only evidence_ids from candidates. For factual fields, if not supported, leave empty/null.\n"
            "For conclusion: you may summarize from supported fields (status/design/endpoints) without inventing results."
        )
        clinical_doc_kinds = [
            "ctgov*",
            "ctis*",
            "clinical_trials*",
            "trial_registry",
            "ru_clinical_permission",
        ]
        trials_extracted, trials_cands = self._extract_structured_from_docs(
            task=trials_task,
            response_format=TrialsExtraction,
            doc_map=doc_map,
            doc_kinds=clinical_doc_kinds,
            top_n=24,
            deadline=deadline,
        )

        if trials_extracted and trials_cands:
            added_any = self._merge_extracted_trials_into_clinical(
                clinical=clinical,
                extracted=trials_extracted,
                cand_map=trials_cands,
                doc_map=doc_map,
                unknowns=unknowns,
            )
            if not added_any:
                self._add_unknown(unknowns, "clinical.global", "no usable trials extracted", clinical_doc_kinds)
        else:
            self._add_unknown(unknowns, "clinical.global", "not found", clinical_doc_kinds)

        # 3.2 Minimal pubmed-style publications block (from already-attached publications in the case)
        pubs_task = (
            "Extract publications/abstracts about the drug and classify them for the clinical UI block.\n"
            f"INN: {inn_normalized}\n"
            "Return items with category one of: comparative, abstracts, real_world, combination.\n"
            "Each item should have a short title and 1-2 sentence summary."
        )
        pubs_doc_kinds = [
            "publication",
            "scientific_pmc",
            "scientific_article",
            "conference_abstract",
            "poster_pdf",
            "case_report",
        ]
        pubs_extracted, pubs_cands = self._extract_structured_from_docs(
            task=pubs_task,
            response_format=PublicationsExtraction,
            doc_map=doc_map,
            doc_kinds=pubs_doc_kinds,
            top_n=18,
            deadline=deadline,
        )
        if pubs_extracted and pubs_cands:
            for item in pubs_extracted.get("items") or []:
                citations = self._citations_from_evidence_ids(item.get("evidence_ids") or [], pubs_cands, doc_map)
                if not citations:
                    continue
                payload = {
                    "title": item.get("title") or "",
                    "summary": item.get("summary") or "",
                    "citations": citations[:3],
                }
                category = item.get("category")
                if category in clinical.get("pubmed", {}):
                    clinical["pubmed"][category].append(payload)
        else:
            self._add_unknown(unknowns, "clinical.pubmed", "not found", pubs_doc_kinds)

        return clinical

    def _enrich_patents_from_docs(
        self,
        patents: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
        unknowns: List[Dict[str, Any]],
        inn_normalized: str,
        use_web: bool,
        deadline: Optional[float],
    ) -> Dict[str, Any]:
        """
        Stage 4: Patent family enrichment using patent PDFs.

        Inputs:
        - patents section already has RU snapshot families (carcass).

        Outputs:
        - fills coverage_type (composition/treatment/synthesis)
        - fills summary (short, human-readable)
        - refines blocking_families ranking
        - fills patents.views (composition/treatment/synthesis family_id lists)
        """
        if not use_web:
            return patents

        families = patents.get("families") or []
        if not isinstance(families, list) or not families:
            self._add_unknown(unknowns, "patents.blocking_families", "not found", ["ru_patent_fips", "patent_pdf"])
            return patents

        # Normalize legacy snapshot shape: coverage_by_jurisdiction -> coverage_by_country.
        for fam in families:
            if not isinstance(fam, dict):
                continue
            if isinstance(fam.get("coverage_by_country"), list):
                continue
            legacy = fam.get("coverage_by_jurisdiction")
            if not isinstance(legacy, list):
                continue
            out_cov: List[Dict[str, Any]] = []
            for row in legacy:
                if not isinstance(row, dict):
                    continue
                c = row.get("country") or row.get("jurisdiction")
                if not c:
                    continue
                entry: Dict[str, Any] = {"country": str(c).strip().upper(), "expires_at": row.get("expires_at")}
                if row.get("status"):
                    entry["status"] = row.get("status")
                out_cov.append(entry)
            fam["coverage_by_country"] = out_cov
            # Remove the legacy field.
            fam.pop("coverage_by_jurisdiction", None)

        # Choose a limited set of families to enrich (LLM calls).
        try:
            max_enrich = int(os.getenv("DDKIT_CASE_VIEW_PATENT_FAMILIES_MAX", "7"))
        except ValueError:
            max_enrich = 7
        max_enrich = max(1, min(max_enrich, 12))

        ranked = self._rank_patent_families(families)
        to_enrich = ranked[:max_enrich]

        patent_doc_kinds = ["patent_pdf", "patent", "ru_patent_fips", "patent_family_summary", "ip_landscape"]

        for fam in to_enrich:
            self._check_deadline(deadline, "patents_before_family_llm")
            fam_id = fam.get("family_id") or ""
            rep_doc = fam.get("representative_doc") or ""
            query = (
                "Patent family analysis for a pharmaceutical due diligence UI.\n"
                f"INN: {inn_normalized}\n"
                f"Family ID: {fam_id}\n"
                f"Representative document/publication number: {rep_doc}\n\n"
                "Tasks:\n"
                "1) Classify what this family covers. Choose one or more of: composition, treatment, synthesis.\n"
                "2) Provide a concise 2-4 sentence summary of what is claimed/covered.\n"
                "3) Add 2-5 short key points if supported by evidence (optional).\n\n"
                "4) Extract 1-3 key (preferably independent) claims if present. "
                "Include claim numbers if possible. Put them into key_claims.\n"
                "5) If the evidence contains a legal status / jurisdiction status table or explicit statements "
                "(e.g., granted / pending / expired / lapsed) for specific jurisdictions, extract highlights into "
                "jurisdiction_statuses items (jurisdiction, status, event_date if present).\n\n"
                "6) If expiry/term dates by jurisdiction are explicitly stated in evidence, extract them into coverage_by_country "
                "items (country + expires_at).\n\n"
                "Rules:\n"
                "- Use ONLY evidence_ids from candidates.\n"
                "- If not supported, leave empty.\n"
            )

            extracted, cand_map = self._extract_structured_from_docs(
                task=query,
                response_format=PatentFamilyInsightExtraction,
                doc_map=doc_map,
                doc_kinds=patent_doc_kinds,
                top_n=22,
                deadline=deadline,
            )
            if not extracted or not cand_map:
                continue

            # Apply coverage_type + summary + citations to the family card.
            new_citations: List[Dict[str, Any]] = []

            cov = extracted.get("coverage_type")
            cov_types = self._normalize_coverage_types((cov or {}).get("value") if isinstance(cov, dict) else None)
            if not cov_types and isinstance(cov, dict) and isinstance(cov.get("value"), (str, list)):
                cov_types = self._normalize_coverage_types(cov.get("value"))
            if cov_types:
                fam["coverage_type"] = cov_types
                new_citations.extend(self._citations_from_evidence_ids(cov.get("evidence_ids") or [], cand_map, doc_map) if isinstance(cov, dict) else [])

            summ = extracted.get("summary")
            if isinstance(summ, dict):
                val = summ.get("value")
                if isinstance(val, str) and val.strip():
                    fam["summary"] = val.strip()
                    new_citations.extend(self._citations_from_evidence_ids(summ.get("evidence_ids") or [], cand_map, doc_map))

            key_points = []
            for kp in extracted.get("key_points") or []:
                if not isinstance(kp, dict):
                    continue
                v = kp.get("value")
                if isinstance(v, str) and v.strip():
                    key_points.append(v.strip())
                    new_citations.extend(self._citations_from_evidence_ids(kp.get("evidence_ids") or [], cand_map, doc_map))
            if key_points:
                fam["key_points"] = key_points[:8]

            # Key claims (preferably independent) as evidence-backed facts (for UI tooltips).
            claims_facts: List[Dict[str, Any]] = []
            for kc in extracted.get("key_claims") or []:
                fact = self._value_to_fact("Ключевой claim", kc, cand_map, doc_map)
                if fact:
                    claims_facts.append(fact)
            if claims_facts:
                fam["claims_highlights"] = claims_facts[:8]

            # Jurisdiction status highlights (if present in evidence).
            jur_statuses_out: List[Dict[str, Any]] = []
            for row in extracted.get("jurisdiction_statuses") or []:
                if not isinstance(row, dict):
                    continue
                jur = str(row.get("jurisdiction") or row.get("country") or "").strip().upper()
                status = str(row.get("status") or "").strip()
                if not jur or not status:
                    continue
                eids = row.get("evidence_ids") or []
                citations = self._citations_from_evidence_ids(eids, cand_map, doc_map)
                if not citations:
                    continue
                out_row: Dict[str, Any] = {"jurisdiction": jur, "status": status, "citations": citations[:3]}
                if row.get("event_date"):
                    out_row["event_date"] = row.get("event_date")
                if row.get("publication_number"):
                    out_row["publication_number"] = row.get("publication_number")
                jur_statuses_out.append(out_row)
            if jur_statuses_out:
                fam["jurisdiction_statuses"] = jur_statuses_out[:25]

            # Optional coverage_by_country refinement from patent text.
            cov_updates = extracted.get("coverage_by_country") or []
            if isinstance(cov_updates, list) and cov_updates:
                existing_cov = fam.get("coverage_by_country") or []
                by_country = {}
                for row in existing_cov:
                    if isinstance(row, dict) and (row.get("country") or row.get("jurisdiction")):
                        key = str(row.get("country") or row.get("jurisdiction")).upper()
                        by_country[key] = row
                for row in cov_updates:
                    if not isinstance(row, dict):
                        continue
                    c = str(row.get("country") or row.get("jurisdiction") or "").strip().upper()
                    if not c:
                        continue
                    expires = row.get("expires_at")
                    status = row.get("status")
                    eids = row.get("evidence_ids") or []
                    # Only fill missing expires_at; snapshot remains "truth" when present.
                    if c in by_country:
                        if not by_country[c].get("expires_at") and expires:
                            by_country[c]["expires_at"] = expires
                        if status and not by_country[c].get("status"):
                            by_country[c]["status"] = status
                    else:
                        entry: Dict[str, Any] = {"country": c, "expires_at": expires}
                        if status:
                            entry["status"] = status
                        existing_cov.append(entry)
                        by_country[c] = existing_cov[-1]
                    if eids:
                        new_citations.extend(self._citations_from_evidence_ids(eids, cand_map, doc_map))
                fam["coverage_by_country"] = existing_cov

            if new_citations:
                new_citations = self._dedupe_citations(new_citations)
                # Keep snapshot citations as fallback at the end.
                existing = fam.get("citations") or []
                fam["citations"] = (new_citations[:5] + existing)[:8]

        # Rebuild UI views based on coverage_type classification.
        comp_ids: List[str] = []
        treat_ids: List[str] = []
        synth_ids: List[str] = []
        seen_comp: set[str] = set()
        seen_treat: set[str] = set()
        seen_synth: set[str] = set()
        for fam in families:
            if not isinstance(fam, dict):
                continue
            fid = str(fam.get("family_id") or "").strip()
            if not fid:
                continue
            types = self._normalize_coverage_types(fam.get("coverage_type"))
            if not types:
                if fid not in seen_comp:
                    comp_ids.append(fid)
                    seen_comp.add(fid)
                continue
            if "composition" in types and fid not in seen_comp:
                comp_ids.append(fid)
                seen_comp.add(fid)
            if "treatment" in types and fid not in seen_treat:
                treat_ids.append(fid)
                seen_treat.add(fid)
            if "synthesis" in types and fid not in seen_synth:
                synth_ids.append(fid)
                seen_synth.add(fid)

        patents["views"] = {"blocking": [], "composition": comp_ids, "treatment": treat_ids, "synthesis": synth_ids}

        # Blocking families: pick top 3-7 by score, prefer families with coverage_by_country.
        ranked2 = self._rank_patent_families(families, prefer_types=True)
        blocking: List[Dict[str, Any]] = []
        for fam in ranked2:
            if fam.get("coverage_by_country"):
                blocking.append(fam)
            if len(blocking) >= 7:
                break
        patents["blocking_families"] = blocking
        patents["views"]["blocking"] = [
            str(f.get("family_id") or "").strip() for f in blocking if isinstance(f, dict) and f.get("family_id")
        ]

        if not patents.get("blocking_families"):
            self._add_unknown(unknowns, "patents.blocking_families", "not found", patent_doc_kinds)

        return patents

    def _enrich_synthesis_from_docs(
        self,
        synthesis: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
        unknowns: List[Dict[str, Any]],
        inn_normalized: str,
        use_web: bool,
        deadline: Optional[float],
    ) -> Dict[str, Any]:
        """
        Stage 5: Extract synthesis route steps (and optional treatment method) from patent PDFs.
        """
        if not use_web:
            return synthesis

        task = (
            "Extract synthesis route information from patent documents for a pharmaceutical case-view UI.\n"
            f"INN: {inn_normalized}\n\n"
            "Return 3–12 high-level synthesis steps if described in the patents.\n"
            "Each step MUST be supported by evidence_ids.\n"
            "Also, if patents describe a method of treatment/use, provide a short summary in treatment_method_from_patents.\n"
            "Do NOT guess if not supported by evidence.\n"
        )
        patent_doc_kinds = ["patent_pdf", "patent", "ru_patent_fips"]
        extracted, cand_map = self._extract_structured_from_docs(
            task=task,
            response_format=SynthesisExtraction,
            doc_map=doc_map,
            doc_kinds=patent_doc_kinds,
            top_n=26,
            deadline=deadline,
        )
        if not extracted or not cand_map:
            self._add_unknown(unknowns, "synthesis.synthesis_route", "not found", patent_doc_kinds)
            return synthesis

        steps_out: List[Dict[str, Any]] = []
        for step in extracted.get("steps") or []:
            if not isinstance(step, dict):
                continue
            text = (step.get("text") or "").strip()
            if not text:
                continue
            citations = self._citations_from_evidence_ids(step.get("evidence_ids") or [], cand_map, doc_map)
            if not citations:
                continue
            steps_out.append({"text": text, "citations": citations[:3]})
        if steps_out:
            synthesis.setdefault("synthesis_route", {})
            synthesis["synthesis_route"]["steps"] = steps_out[:20]
        else:
            self._add_unknown(unknowns, "synthesis.synthesis_route", "not found", patent_doc_kinds)

        tm = extracted.get("treatment_method_from_patents")
        tm_fact = self._value_to_fact("Способ лечения (из патентов)", tm, cand_map, doc_map)
        if tm_fact:
            synthesis["treatment_method_from_patents"] = tm_fact
        return synthesis

    def build_patents(self, snapshot: Optional[dict], use_snapshot: bool,
                      unknowns: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Patents are presented in the UI as:
        # - top "blocking" families (cards)
        # - three subtabs (composition / treatment / synthesis)
        #
        # We avoid duplicating the same family object in 3 lists by keeping a single `families[]`
        # and referencing them by id in `views`.
        patents = {
            "blocking_families": [],
            "families": [],
            "views": {"blocking": [], "composition": [], "treatment": [], "synthesis": []},
        }
        if not use_snapshot or not snapshot:
            self._add_unknown(unknowns, "patents", "snapshot missing", ["frontend_snapshot"])
            return patents

        # Collect from BOTH global (EPO/PatentsView/Lens) and RU (Rospatent) sources
        global_families = self._collect_global_patent_families(snapshot)
        ru_families = self._collect_ru_patent_families(snapshot)

        # Merge and dedupe by family_id (global takes priority)
        families_by_id: Dict[str, Dict[str, Any]] = {}
        for fam in global_families:
            fid = fam.get("family_id") or fam.get("doc_id")  # Fallback to doc_id
            if fid:
                families_by_id[fid] = fam
        for fam in ru_families:
            fid = fam.get("family_id") or fam.get("doc_id")  # Fallback to doc_id
            if fid and fid not in families_by_id:
                families_by_id[fid] = fam

        families = list(families_by_id.values())

        if families:
            patents["blocking_families"] = families[:7]
            patents["families"] = families
        else:
            self._add_unknown(unknowns, "patents.blocking_families", "not found", ["epo_ops", "patentsview", "lens"])

        return patents

    def build_synthesis(self, patents: Dict[str, Any], unknowns: List[Dict[str, Any]]) -> Dict[str, Any]:
        synthesis = {
            "synthesis_route": {"steps": []},
            "treatment_method_from_patents": None,
        }
        return synthesis

    def build_sources(
        self,
        snapshot: Optional[dict],
        documents: List[Dict[str, Any]],
        clinical: Optional[Dict[str, Any]] = None,
        patents: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        structured_sources = []
        if snapshot:
            structured_sources.append({"source": "frontend_snapshot"})

        by_trial_id: Dict[str, List[str]] = {}
        if clinical:
            for bucket in ("global", "ru", "ongoing"):
                phases = clinical.get(bucket, {})
                if not isinstance(phases, dict):
                    continue
                for trials in phases.values():
                    for trial in _as_list(trials):
                        if not isinstance(trial, dict):
                            continue
                        tid = str(trial.get("trial_id") or "").strip()
                        if not tid:
                            continue
                        doc_ids = set()
                        for cit in _as_list(trial.get("citations")):
                            if isinstance(cit, dict) and cit.get("doc_id"):
                                doc_ids.add(str(cit["doc_id"]))
                        if not doc_ids:
                            continue
                        by_trial_id.setdefault(tid, [])
                        existing = set(by_trial_id[tid])
                        for d in sorted(doc_ids):
                            if d not in existing:
                                by_trial_id[tid].append(d)
                                existing.add(d)

        by_family_id: Dict[str, List[str]] = {}
        if patents:
            for fam in (_as_list(patents.get("blocking_families")) + _as_list(patents.get("families"))):
                if not isinstance(fam, dict):
                    continue
                fid = str(fam.get("family_id") or "").strip()
                if not fid:
                    continue
                doc_ids = set()
                for cit in _as_list(fam.get("citations")):
                    if isinstance(cit, dict) and cit.get("doc_id"):
                        doc_ids.add(str(cit["doc_id"]))
                if not doc_ids:
                    continue
                by_family_id.setdefault(fid, [])
                existing = set(by_family_id[fid])
                for d in sorted(doc_ids):
                    if d not in existing:
                        by_family_id[fid].append(d)
                        existing.add(d)

        # Filters (counts) for UI convenience.
        by_kind: Dict[str, int] = {}
        by_region: Dict[str, int] = {}
        by_year: Dict[str, int] = {}
        for d in documents:
            kind = str(d.get("doc_kind") or "").strip().lower()
            if kind:
                by_kind[kind] = by_kind.get(kind, 0) + 1
            region = str(d.get("region") or "").strip().lower()
            if region:
                by_region[region] = by_region.get(region, 0) + 1
            date = str(d.get("date") or "").strip()
            if len(date) >= 4 and date[:4].isdigit():
                y = date[:4]
                by_year[y] = by_year.get(y, 0) + 1

        return {
            "documents": documents,
            "structured_sources": structured_sources,
            "groups": {
                "by_trial_id": by_trial_id,
                "by_family_id": by_family_id,
            },
            "filters": {
                "by_kind": by_kind,
                "by_region": by_region,
                "by_year": by_year,
            },
        }

    # -------------------------
    # Evidence + LLM helpers (Stage 1)
    # -------------------------

    @staticmethod
    def _check_deadline(deadline: Optional[float], stage: str) -> None:
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"case_view_timeout at {stage}")

    def _build_schema_prompt(self, instruction: str, schema_model: Any, example: str = "") -> str:
        schema_src = re.sub(r"^ {4}", "", inspect.getsource(schema_model), flags=re.MULTILINE)
        return build_system_prompt(instruction, example, schema_src)

    def _format_retrieved_context(self, retrieved: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in retrieved:
            doc_id = item.get("doc_id")
            page_number = item.get("page")
            text = (item.get("text") or "").strip()
            if not text:
                continue
            header = f"Doc {doc_id} | Page {page_number}"
            parts.append(f"{header}:\n\"\"\"\n{text}\n\"\"\"")
        return "\n\n---\n\n".join(parts[:12])

    def _retrieve_candidates(
        self,
        query: str,
        doc_map: Dict[str, Dict[str, Any]],
        doc_kinds: Optional[List[str]] = None,
        top_n: int = 12,
    ) -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, Any]]:
        # Retrieve across the case vector store.
        retrieved = self.retriever.retrieve_by_case(
            query=query,
            top_n=top_n,
            tenant_id=self.tenant_id,
            case_id=self.case_id,
            doc_kind=doc_kinds,
        )
        retrieved_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for item in retrieved:
            doc_id = item.get("doc_id") or "unknown"
            retrieved_by_doc.setdefault(doc_id, []).append(item)

        doc_titles = {d_id: (doc_map.get(d_id) or {}).get("title") for d_id in retrieved_by_doc.keys()}
        candidates = self.evidence_builder.build_candidates_from_multiple_docs(
            retrieved_by_doc,
            doc_titles=doc_titles,
        )
        cand_map = {c.evidence_id: c for c in candidates}
        return retrieved, candidates, cand_map

    def _citations_from_evidence_ids(
        self,
        evidence_ids: List[str],
        cand_map: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple] = set()
        for eid in evidence_ids or []:
            cand = cand_map.get(eid)
            if not cand:
                continue
            doc_id = getattr(cand, "doc_id", None)
            page = getattr(cand, "page", None)
            snippet = getattr(cand, "snippet", None)
            if not doc_id or not snippet:
                continue
            page_num = page if isinstance(page, int) else None
            meta = doc_map.get(doc_id) or {}
            source_url = meta.get("source_url")
            # Deduplicate inside a single fact by exact evidence tuple.
            key = (doc_id, page_num, snippet, source_url)
            if key in seen:
                continue
            seen.add(key)
            citation = {
                "doc_id": doc_id,
                "page": page_num,
                "snippet": snippet,
            }
            if source_url:
                citation["source_url"] = source_url
            out.append(citation)
        return out

    @staticmethod
    def _dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate citations while preserving order.

        This is helpful when we aggregate citations from multiple extracted fields
        and end up with the same (doc_id,page,snippet) repeated.
        """
        out: List[Dict[str, Any]] = []
        seen: set[tuple] = set()
        for c in citations or []:
            if not isinstance(c, dict):
                continue
            key = (
                c.get("structured_source"),
                c.get("json_path"),
                c.get("doc_id"),
                c.get("page"),
                c.get("snippet"),
                c.get("source_url"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _global_dedupe_citations(
        self,
        passport: Dict[str, Any],
        brief: Dict[str, Any],
        regulatory: Dict[str, Any],
        clinical: Dict[str, Any],
        patents: Dict[str, Any],
        synthesis: Dict[str, Any],
    ) -> None:
        """
        Global deduplication of citations across all facts in the case view.

        This reduces UI clutter when the same evidence snippet appears in multiple facts.
        We keep only unique (doc_id, page, snippet, source_url) tuples globally to avoid
        overwhelming the right-panel evidence viewer while preserving distinct snippets.
        """
        global_seen: set[tuple] = set()

        def dedupe_in_place(node: Any) -> None:
            """Recursively find facts and dedupe their citations in place."""
            if isinstance(node, dict):
                # If this looks like a fact with citations, dedupe citations.
                if "citations" in node and isinstance(node.get("citations"), list):
                    original_cits = [c for c in node.get("citations") or [] if isinstance(c, dict)]
                    new_cits: List[Dict[str, Any]] = []
                    for c in original_cits:
                        if not isinstance(c, dict):
                            continue
                        # Skip dedup for snapshot citations — they are expected to repeat
                        # across multiple facts and should not be removed.
                        if c.get("structured_source") == "frontend_snapshot":
                            new_cits.append(c)
                            continue
                        # Key by exact evidence tuple to avoid dropping distinct snippets.
                        key = (
                            c.get("doc_id"),
                            c.get("page"),
                            c.get("snippet"),
                            c.get("source_url"),
                        )
                        if key in global_seen:
                            continue
                        global_seen.add(key)
                        new_cits.append(c)
                    # Ensure we keep at least one citation per fact if it had any.
                    if not new_cits and original_cits:
                        new_cits = [original_cits[0]]
                    node["citations"] = new_cits
                # Recurse into values.
                for v in node.values():
                    dedupe_in_place(v)
            elif isinstance(node, list):
                for item in node:
                    dedupe_in_place(item)

        # Process in priority order: passport facts are most important,
        # then brief, regulatory, clinical, patents, synthesis.
        dedupe_in_place(passport)
        dedupe_in_place(brief)
        dedupe_in_place(regulatory)
        dedupe_in_place(clinical)
        dedupe_in_place(patents)
        dedupe_in_place(synthesis)

    def _extract_structured_from_docs(
        self,
        task: str,
        response_format: Any,
        doc_map: Dict[str, Dict[str, Any]],
        doc_kinds: List[str],
        top_n: int,
        deadline: Optional[float],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Generic evidence-locked extraction helper.

        Returns: (parsed_payload_dict, cand_map[evidence_id->EvidenceCandidate])
        """
        self._check_deadline(deadline, "before_retrieval")
        retrieved, candidates, cand_map = self._retrieve_candidates(
            query=task,
            doc_map=doc_map,
            doc_kinds=doc_kinds,
            top_n=top_n,
        )
        if not candidates:
            return None, None

        candidates_prompt = self.evidence_builder.candidates_to_prompt_format(candidates)
        system_prompt = self._build_schema_prompt(
            instruction=(
                "You are extracting structured data for a pharmaceutical case-view UI.\n"
                "Use ONLY the provided evidence candidates.\n"
                "CRITICAL: You MUST ONLY reference evidence via evidence_ids from the candidates list.\n"
                "Do NOT invent evidence IDs. Do NOT guess.\n"
                "If evidence is missing, leave the field empty/null.\n"
                "Output must be valid JSON and match the schema exactly."
            ),
            schema_model=response_format,
        )
        user_prompt = (
            f"Task:\n{task}\n\n"
            f"Context:\n{self._format_retrieved_context(retrieved)}\n\n"
            f"Available evidence candidates:\n{candidates_prompt}\n"
        )
        self._check_deadline(deadline, "before_llm")
        try:
            answer = self.api.send_message(
                model=self.answering_model,
                system_content=system_prompt,
                human_content=user_prompt,
                is_structured=True,
                response_format=response_format,
            )
        except Exception as exc:
            logger.warning("LLM extraction failed: %s", exc)
            return None, None
        self._check_deadline(deadline, "after_llm")

        # Basic evidence id sanity: keep only ids that exist in candidates.
        valid_ids = set(cand_map.keys())
        def _filter_value(v: Any) -> Any:
            if isinstance(v, dict) and "evidence_ids" in v:
                ids = [eid for eid in (v.get("evidence_ids") or []) if eid in valid_ids]
                v["evidence_ids"] = ids
            return v
        if isinstance(answer, dict):
            for k, v in list(answer.items()):
                if isinstance(v, list):
                    answer[k] = [_filter_value(it) for it in v]
                else:
                    answer[k] = _filter_value(v)
        return answer if isinstance(answer, dict) else None, cand_map

    def _value_to_fact(
        self,
        label: str,
        value_obj: Any,
        cand_map: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if value_obj is None:
            return None
        if isinstance(value_obj, EvidenceLockedValue):
            value = value_obj.value
            eids = value_obj.evidence_ids
        elif isinstance(value_obj, dict):
            value = value_obj.get("value")
            eids = value_obj.get("evidence_ids") or []
        else:
            return None
        if value is None or value == "" or value == [] or value == {}:
            return None
        citations = self._citations_from_evidence_ids(eids, cand_map, doc_map)
        if not citations:
            return None
        citations = self._dedupe_citations(citations)
        return self._fact(label, value, citations[:3])

    def _market_regulatory_from_extraction(
        self,
        market: str,
        extracted: Dict[str, Any],
        cand_map: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        trade_names = self._value_to_fact("Торговые названия", extracted.get("trade_names"), cand_map, doc_map)
        holders = self._value_to_fact("Держатель регистрации", extracted.get("holders"), cand_map, doc_map)
        dfs = self._value_to_fact("Дозировки/формы", extracted.get("dosage_forms_and_strengths"), cand_map, doc_map)
        status = self._value_to_fact("Статус", extracted.get("status"), cand_map, doc_map)
        countries = self._value_to_fact("Страны покрытия", extracted.get("countries_covered"), cand_map, doc_map)

        if trade_names:
            out["trade_names"] = trade_names
        if holders:
            out["holders"] = holders
        if dfs:
            out["dosage_forms_and_strengths"] = dfs
        if status:
            out["status"] = status
        if countries:
            out["countries_covered"] = countries
        out["market"] = market
        return out

    def _merge_extracted_trials_into_clinical(
        self,
        clinical: Dict[str, Any],
        extracted: Dict[str, Any],
        cand_map: Dict[str, Any],
        doc_map: Dict[str, Dict[str, Any]],
        unknowns: List[Dict[str, Any]],
    ) -> bool:
        trials = extracted.get("trials") or []
        if not isinstance(trials, list) or not trials:
            return False

        def is_ongoing(status: Optional[str]) -> bool:
            if not status:
                return False
            s = status.strip().lower()
            for token in ["recruit", "enrolling", "active", "not yet", "ongoing", "suspend"]:
                if token in s:
                    return True
            return False

        seen_ids: set[str] = set()
        added = False
        for it in trials:
            if not isinstance(it, dict):
                continue
            trial_id = (it.get("trial_id") or "").strip()
            if not trial_id or trial_id.lower() in seen_ids:
                continue
            seen_ids.add(trial_id.lower())

            citations = self._citations_from_evidence_ids(it.get("evidence_ids") or [], cand_map, doc_map)
            if not citations:
                continue

            trial = {
                "trial_id": trial_id,
                "title": it.get("title"),
                "phase": it.get("phase"),
                "study_type": it.get("study_type"),
                "countries": it.get("countries") or [],
                "enrollment": it.get("enrollment"),
                "comparator": it.get("comparator"),
                "regimen": it.get("regimen"),
                "status": it.get("status"),
                "efficacy_key_points": it.get("efficacy_key_points") or [],
                "conclusion": it.get("conclusion"),
                "where_conducted": it.get("where_conducted"),
                "citations": citations[:3],
            }

            # Ensure conclusion is always present for UI cards (especially for ongoing trials).
            if not trial.get("conclusion"):
                status = str(trial.get("status") or "").strip()
                status_l = status.lower()
                is_recruiting = any(tok in status_l for tok in ["recruit", "enroll", "active", "not yet", "ongoing", "suspend"])
                if is_recruiting:
                    trial["conclusion"] = f"Trial is {status}; results are not yet available."
                elif trial.get("efficacy_key_points"):
                    trial["conclusion"] = "; ".join([str(x) for x in (trial.get("efficacy_key_points") or [])[:2] if str(x).strip()])
                elif status:
                    trial["conclusion"] = f"Trial status: {status}. No efficacy conclusion extracted from available evidence."
                else:
                    trial["conclusion"] = "No conclusion extracted from available evidence."

            if not trial.get("where_conducted"):
                countries = [str(c).strip() for c in (trial.get("countries") or []) if str(c).strip()]
                if countries:
                    trial["where_conducted"] = ", ".join(countries[:6])

            phase_key = self._normalize_phase(trial.get("phase"))
            bucket = "global"
            countries = [str(c).lower() for c in (trial.get("countries") or [])]
            if countries and all(c in {"ru", "russia", "russian federation"} for c in countries):
                bucket = "ru"
            if is_ongoing(trial.get("status")):
                bucket = "ongoing"

            clinical.setdefault(bucket, self._empty_phase_map())
            clinical[bucket].setdefault(phase_key, []).append(trial)
            added = True

        if not added:
            self._add_unknown(unknowns, "clinical.global", "extracted trials missing citations", ["ctgov", "ctis"])
        return added

    def _load_documents_meta(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for doc_path in self.documents_dir.glob("*.json"):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                meta = doc.get("metainfo", {})
                doc_id = meta.get("doc_id", doc_path.stem)
                documents.append({
                    "doc_id": doc_id,
                    "title": meta.get("title") or meta.get("company_name") or doc_id,
                    "doc_kind": meta.get("doc_kind"),
                    "source_url": meta.get("source_url"),
                    "region": meta.get("region"),
                    "date": meta.get("published_at") or meta.get("date")
                })
            except Exception:
                continue
        return documents

    def _snapshot_citation(self, json_path: Optional[str]) -> Dict[str, Any]:
        citation = {
            "structured_source": "frontend_snapshot",
        }
        if json_path:
            citation["json_path"] = json_path
        return citation

    def _fact(self, label: str, value: Any, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Copy citations to avoid shared references being mutated by _global_dedupe_citations
        return {
            "label": label,
            "value": value,
            "citations": list(citations) if citations else []
        }

    def _fact_has_citations(self, fact: Dict[str, Any]) -> bool:
        return bool(fact and isinstance(fact.get("citations"), list) and fact["citations"])

    def _fact_has_non_snapshot_citations(self, fact: Dict[str, Any]) -> bool:
        if not fact or not isinstance(fact.get("citations"), list):
            return False
        return any(
            isinstance(c, dict) and c.get("structured_source") != "frontend_snapshot"
            for c in fact.get("citations") or []
        )

    def _add_unknown(self, unknowns: List[Dict[str, Any]], field: str,
                     reason: str, needed_sources: Optional[List[str]] = None) -> None:
        unknowns.append({
            "field": field,
            "reason": reason,
            "needed_sources": needed_sources or []
        })

    @staticmethod
    def _is_demo_text(value: Any) -> bool:
        """
        Heuristic: frontend snapshots/e2e fixtures sometimes contain "demo" placeholders.
        We treat such values as missing to avoid leaking demo data into the UI.
        """
        if value is None:
            return False
        if isinstance(value, str):
            s = value.strip().lower()
            if not s:
                return False
            return any(tok in s for tok in ["demo", "демо", "пример", "stub", "test", "snapshot"])
        return False

    def _filter_demo_strings(self, items: Any) -> List[str]:
        out: List[str] = []
        for it in _as_list(items):
            if it is None:
                continue
            s = str(it).strip()
            if not s or self._is_demo_text(s):
                continue
            if s not in out:
                out.append(s)
        return out

    def _filter_demo_market_map(self, value: Any) -> Dict[str, List[str]]:
        if not isinstance(value, dict):
            return {}
        out: Dict[str, List[str]] = {}
        for k, v in value.items():
            out[str(k)] = self._filter_demo_strings(v)
        # prune empties
        if not any(out.values()):
            return {}
        return out

    def _collect_trade_names(self, snapshot: Optional[dict]) -> Dict[str, List[str]]:
        trade_names: Dict[str, List[str]] = {"ru": [], "eu": [], "us": []}
        if not snapshot:
            return {}
        ru_items = _get_path(snapshot, ["ruSections", "regulatory", "items"]) or []
        for item in _as_list(ru_items):
            for key in ("trade_name", "tradeName", "drug_name", "drugName"):
                val = item.get(key)
                if val and val not in trade_names["ru"]:
                    trade_names["ru"].append(val)
        for key in ("ru", "eu", "us"):
            market = _get_path(snapshot, ["regulatory", key]) or {}
            for field in ("trade_names", "tradeNames", "brands", "brand_names"):
                for name in _as_list(market.get(field)):
                    if name and name not in trade_names[key]:
                        trade_names[key].append(name)
        if any(trade_names.values()):
            return trade_names
        return {}

    def _collect_holders(self, snapshot: Optional[dict]) -> Dict[str, List[str]]:
        holders: Dict[str, List[str]] = {"ru": [], "eu": [], "us": []}
        if not snapshot:
            return {}
        ru_items = _get_path(snapshot, ["ruSections", "regulatory", "items"]) or []
        for item in _as_list(ru_items):
            holder = item.get("holder")
            if holder and holder not in holders["ru"]:
                holders["ru"].append(holder)
        for key in ("eu", "us"):
            market = _get_path(snapshot, ["regulatory", key]) or {}
            for field in ("holders", "mah", "mah_list"):
                for holder in _as_list(market.get(field)):
                    if holder and holder not in holders[key]:
                        holders[key].append(holder)
        if any(holders.values()):
            return holders
        return {}

    def _collect_forms(self, snapshot: Optional[dict]) -> Dict[str, List[str]]:
        forms: Dict[str, List[str]] = {"ru": [], "eu": [], "us": []}
        if not snapshot:
            return {}
        ru_items = _get_path(snapshot, ["ruSections", "regulatory", "items"]) or []
        for item in _as_list(ru_items):
            for form in _as_list(item.get("forms")):
                if form and form not in forms["ru"]:
                    forms["ru"].append(form)
            for auth in _as_list(item.get("authorized_presentations")):
                form = auth.get("form")
                if form and form not in forms["ru"]:
                    forms["ru"].append(form)
        if any(forms.values()):
            return forms
        return {}

    def _collect_strengths(self, snapshot: Optional[dict]) -> Dict[str, List[str]]:
        strengths: Dict[str, List[str]] = {"ru": [], "eu": [], "us": []}
        if not snapshot:
            return {}
        ru_items = _get_path(snapshot, ["ruSections", "regulatory", "items"]) or []
        for item in _as_list(ru_items):
            for strength in _as_list(item.get("strengths")):
                if strength and strength not in strengths["ru"]:
                    strengths["ru"].append(strength)
            for auth in _as_list(item.get("authorized_presentations")):
                strength = auth.get("strength")
                if strength and strength not in strengths["ru"]:
                    strengths["ru"].append(strength)
        if any(strengths.values()):
            return strengths
        return {}

    def _detect_registration_markets(self, snapshot: Optional[dict]) -> Dict[str, Any]:
        if not snapshot:
            return {}

        out: Dict[str, Any] = {
            "ru": {"status": "unknown", "countries": ["RU"]},
            "eu": {"status": "unknown", "countries": []},
            "us": {"status": "unknown", "countries": ["US"]},
        }

        if _get_path(snapshot, ["ruSections", "regulatory", "items"]):
            out["ru"]["status"] = "registered"

        eu_block = _get_path(snapshot, ["regulatory", "eu"]) or {}
        if eu_block:
            out["eu"]["status"] = "registered"
            eu_countries, _ = _pick_value(snapshot, [
                ["regulatory", "eu", "countries"],
                ["regulatory", "eu", "covered_countries"],
                ["summary", "eu", "countries"],
            ])
            if isinstance(eu_countries, list):
                out["eu"]["countries"] = [str(x) for x in eu_countries if str(x).strip()]

        us_block = _get_path(snapshot, ["regulatory", "us"]) or {}
        if us_block:
            out["us"]["status"] = "registered"

        # If everything is unknown and no hint in snapshot, return empty.
        if all(out[m]["status"] == "unknown" for m in ("ru", "eu", "us")):
            return {}
        return out

    def _collect_ru_reg_entries(self, snapshot: Optional[dict]) -> List[Dict[str, Any]]:
        ru_entries: List[Dict[str, Any]] = []
        if not snapshot:
            return ru_entries
        ru_reg = _get_path(snapshot, ["ruSections", "regulatory"]) or {}
        items = ru_reg.get("items") or []
        for idx, item in enumerate(_as_list(items)):
            base_path = f"$.ruSections.regulatory.items[{idx}]"
            citations = [self._snapshot_citation(base_path)]
            entry = {
                "reg_no": self._fact("Рег. номер", item.get("reg_no"), citations),
                "reg_number": self._fact("Рег. номер", item.get("reg_no"), citations),
                "status": self._fact("Статус", item.get("status"), citations),
                "trade_name": self._fact("Торговое название", item.get("trade_name") or item.get("tradeName"), citations),
                "holder": self._fact("Держатель", item.get("holder"), citations),
                "forms": self._fact("Формы", item.get("forms"), citations),
                "dosage_forms": self._fact("Формы", item.get("forms"), citations),
                "strengths": self._fact("Дозировки", item.get("strengths"), citations),
                "routes": self._fact("Пути введения", item.get("routes"), citations),
                "authorized_presentations": self._fact("Презентации", item.get("authorized_presentations"), citations),
                "links": item.get("links", {}),
            }
            ru_entries.append(entry)
        return ru_entries

    def _collect_ru_clinical_trials(self, snapshot: Optional[dict]) -> List[Dict[str, Any]]:
        trials: List[Dict[str, Any]] = []
        ru_clinical = _get_path(snapshot, ["ruSections", "clinical"]) or {}
        studies = ru_clinical.get("studies") or []
        for idx, item in enumerate(_as_list(studies)):
            base_path = f"$.ruSections.clinical.studies[{idx}]"
            citations = [self._snapshot_citation(base_path)]
            trials.append({
                "trial_id": item.get("id"),
                "title": item.get("title"),
                "phase": item.get("phase"),
                "study_type": item.get("study_type"),
                "countries": item.get("countries") or [],
                "enrollment": item.get("enrollment"),
                "comparator": item.get("comparator"),
                "regimen": item.get("regimen"),
                "status": item.get("status"),
                "efficacy_key_points": item.get("efficacy_key_points") or [],
                "conclusion": item.get("conclusion"),
                "where_conducted": item.get("where_conducted"),
                "citations": citations
            })
        return trials

    def _collect_ru_patent_families(self, snapshot: Optional[dict]) -> List[Dict[str, Any]]:
        families_by_id: Dict[str, Dict[str, Any]] = {}
        ru_patents = _get_path(snapshot, ["ruSections", "patents"]) or {}
        items = ru_patents.get("patent_families") or []
        for idx, item in enumerate(_as_list(items)):
            base_path = f"$.ruSections.patents.patent_families[{idx}]"
            citations = [self._snapshot_citation(base_path)]
            members = _as_list(item.get("members"))
            jurisdictions = []
            coverage_by_country = []
            rep_doc = None

            # Extract assignees from snapshot (if available)
            assignees_raw = _as_list(item.get("assignees") or item.get("Assignees") or item.get("applicants"))
            assignees = []
            for a in assignees_raw:
                if isinstance(a, str) and a.strip():
                    assignees.append(a.strip())
                elif isinstance(a, dict):
                    name = a.get("name") or a.get("Name") or a.get("assignee") or a.get("Assignee")
                    if name and isinstance(name, str):
                        assignees.append(name.strip())
            if not assignees and item.get("owner"):
                owner = item.get("owner")
                if isinstance(owner, str):
                    assignees.append(owner.strip())
            assignees = list(dict.fromkeys(assignees)) or ["Unknown"]

            for member in members:
                country = member.get("jurisdiction") or member.get("country")
                if country and country not in jurisdictions:
                    jurisdictions.append(country)
                expiry = member.get("expiryDateBase")
                if country:
                    coverage_by_country.append({"country": country, "expires_at": expiry})
                if rep_doc is None:
                    rep_doc = member.get("publicationNumber")
            fam_id = str(item.get("familyId") or "").strip()
            if not fam_id:
                continue
            existing = families_by_id.get(fam_id)
            if not existing:
                families_by_id[fam_id] = {
                    "family_id": fam_id,
                    "representative_doc": rep_doc,
                    "summary": item.get("summary", {}).get("mainStatus") if isinstance(item.get("summary"), dict) else None,
                    "coverage_type": ["unknown"],
                    "countries": jurisdictions,
                    "coverage_by_country": coverage_by_country,
                    "assignees": assignees,
                    "citations": citations
                }
                continue

            # Merge duplicates by family_id.
            if not existing.get("representative_doc") and rep_doc:
                existing["representative_doc"] = rep_doc
            if not existing.get("summary") and isinstance(item.get("summary"), dict):
                existing["summary"] = item.get("summary", {}).get("mainStatus")

            for country in jurisdictions:
                if country not in (existing.get("countries") or []):
                    existing.setdefault("countries", []).append(country)

            by_country = {str(c.get("country") or c.get("jurisdiction")): c for c in _as_list(existing.get("coverage_by_country")) if isinstance(c, dict)}
            for row in coverage_by_country:
                country = str(row.get("country") or row.get("jurisdiction") or "")
                if not country:
                    continue
                if country in by_country:
                    if not by_country[country].get("expires_at") and row.get("expires_at"):
                        by_country[country]["expires_at"] = row.get("expires_at")
                else:
                    existing.setdefault("coverage_by_country", []).append(row)
                    by_country[country] = row

            # Merge assignees
            existing_assignees = existing.get("assignees") or []
            for a in assignees:
                if a not in existing_assignees:
                    existing_assignees.append(a)
            existing["assignees"] = existing_assignees

            existing["citations"] = self._dedupe_citations((existing.get("citations") or []) + citations)

        return list(families_by_id.values())

    @staticmethod
    def _calculate_expires_at(priority_date_str: Optional[str], years_to_add: int = 20) -> Optional[str]:
        """
        Calculate patent expiry date from priority date + years_to_add (default 20 years).
        Accepts formats: YYYY, YYYY-MM, YYYY-MM-DD, or ISO datetime.
        Returns ISO date string (YYYY-MM-DD) or None if parsing fails.
        """
        if not priority_date_str:
            return None

        date_str = str(priority_date_str).strip()
        if not date_str:
            return None

        try:
            # Try parsing various formats
            if len(date_str) == 4 and date_str.isdigit():
                # YYYY
                priority_date = datetime(int(date_str), 1, 1)
            elif len(date_str) == 7 and date_str[4] == '-':
                # YYYY-MM
                parts = date_str.split('-')
                priority_date = datetime(int(parts[0]), int(parts[1]), 1)
            elif len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                # YYYY-MM-DD
                parts = date_str.split('-')
                priority_date = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
            else:
                # Try ISO datetime parsing
                priority_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

            # Add years_to_add years
            expiry_date = priority_date + timedelta(days=365 * years_to_add)
            return expiry_date.strftime("%Y-%m-%d")
        except (ValueError, AttributeError, IndexError):
            logger.debug(f"Failed to parse priority date for expiry calculation: {date_str}")
            return None

    def _collect_global_patent_families(self, snapshot: Optional[dict]) -> List[Dict[str, Any]]:
        """
        Collect patent families from global sources (EPO, PatentsView, Lens).
        These are stored in snapshot["patents"]["families"].
        """
        families_by_id: Dict[str, Dict[str, Any]] = {}
        if not snapshot:
            return []

        # Global patents from EPO/PatentsView/Lens are in snapshot["patents"]["families"]
        patents_block = _get_path(snapshot, ["patents"]) or {}
        items = patents_block.get("families") or []

        for idx, item in enumerate(_as_list(items)):
            if not isinstance(item, dict):
                continue

            base_path = f"$.patents.families[{idx}]"
            citations = [self._snapshot_citation(base_path)]

            # Extract family_id - try multiple field names, fallback to DocID
            fam_id = str(
                item.get("family_id") or
                item.get("familyId") or
                item.get("FamilyID") or
                item.get("DocID") or  # Fallback to DocID if no family_id
                item.get("doc_id") or
                ""
            ).strip()
            if not fam_id:
                continue

            # Skip demo/mock data
            if self._is_demo_text(fam_id):
                continue

            # Extract representative doc
            rep_doc = (
                item.get("representative") or
                item.get("representative_doc") or
                item.get("publicationNumber") or
                None
            )

            # Extract assignees/holders (правообладатели)
            assignees_raw = _as_list(item.get("assignees") or item.get("Assignees") or item.get("applicants"))
            assignees = []
            for a in assignees_raw:
                if isinstance(a, str) and a.strip():
                    assignees.append(a.strip())
                elif isinstance(a, dict):
                    name = a.get("name") or a.get("Name") or a.get("assignee") or a.get("Assignee")
                    if name and isinstance(name, str):
                        assignees.append(name.strip())
            # Fallback: если assignees пусто, берём из владельца
            if not assignees and item.get("owner"):
                owner = item.get("owner")
                if isinstance(owner, str):
                    assignees.append(owner.strip())
            # Уникализируем
            assignees = list(dict.fromkeys(assignees)) or ["Unknown"]

            # Extract priority date for expires_at calculation
            earliest_priority = (
                item.get("earliest_priority") or
                item.get("earliestPriority") or
                item.get("priority_date") or
                item.get("priorityDate") or
                item.get("filingDate") or
                item.get("filing_date")
            )

            # Extract jurisdictions from members or directly
            members = _as_list(item.get("members"))
            jurisdictions = []
            coverage_by_country = []

            # If members is a list of doc IDs (strings), extract jurisdiction from prefix
            for member in members:
                if isinstance(member, str):
                    # Doc IDs like "EP1234567" or "US1234567"
                    country = member[:2].upper() if len(member) >= 2 else None
                    if country and country not in jurisdictions:
                        jurisdictions.append(country)
                elif isinstance(member, dict):
                    country = member.get("jurisdiction") or member.get("country")
                    if country and country not in jurisdictions:
                        jurisdictions.append(country)
                    # Расчет expires_at: если указана дата в member, используем её, иначе рассчитываем
                    expiry = member.get("expiryDateBase") or member.get("expires_at")
                    if not expiry and earliest_priority:
                        expiry = self._calculate_expires_at(earliest_priority, years_to_add=20)
                    if country:
                        coverage_by_country.append({"country": country, "expires_at": expiry})

            # Extract expiry info from estimated_expiry field
            estimated_expiry = item.get("estimated_expiry") or {}
            if isinstance(estimated_expiry, dict):
                for country_key in ["US", "EP", "WO", "CN", "JP"]:
                    exp_date = estimated_expiry.get(country_key)
                    # Если даты нет, рассчитываем
                    if not exp_date and earliest_priority:
                        exp_date = self._calculate_expires_at(earliest_priority, years_to_add=20)
                    if exp_date and country_key not in [c.get("country") for c in coverage_by_country]:
                        coverage_by_country.append({"country": country_key, "expires_at": exp_date})
                        if country_key not in jurisdictions:
                            jurisdictions.append(country_key)

            # Extract status
            status = item.get("status") or item.get("summary")
            if isinstance(status, dict):
                status = status.get("mainStatus") or status.get("status")

            existing = families_by_id.get(fam_id)
            if not existing:
                families_by_id[fam_id] = {
                    "family_id": fam_id,
                    "representative_doc": rep_doc,
                    "summary": status,
                    "coverage_type": ["unknown"],  # Will be enriched later by LLM
                    "countries": jurisdictions,
                    "coverage_by_country": coverage_by_country,
                    "assignees": assignees,
                    "cpc": _as_list(item.get("cpc_top") or item.get("cpc")),
                    "earliest_priority": earliest_priority,
                    "citations": citations,
                }
            else:
                # Merge
                if not existing.get("representative_doc") and rep_doc:
                    existing["representative_doc"] = rep_doc
                if not existing.get("summary") and status:
                    existing["summary"] = status
                if not existing.get("assignees") or existing.get("assignees") == ["Unknown"]:
                    existing["assignees"] = assignees
                for country in jurisdictions:
                    if country not in (existing.get("countries") or []):
                        existing.setdefault("countries", []).append(country)

                # Merge coverage_by_country
                by_country = {str(c.get("country")): c for c in _as_list(existing.get("coverage_by_country")) if isinstance(c, dict)}
                for row in coverage_by_country:
                    country_key = str(row.get("country") or "")
                    if not country_key:
                        continue
                    if country_key in by_country:
                        if not by_country[country_key].get("expires_at") and row.get("expires_at"):
                            by_country[country_key]["expires_at"] = row.get("expires_at")
                    else:
                        existing.setdefault("coverage_by_country", []).append(row)
                        by_country[country_key] = row

                existing["citations"] = self._dedupe_citations((existing.get("citations") or []) + citations)

        return list(families_by_id.values())

    def _normalize_coverage_types(self, value: Any) -> List[str]:
        """
        Normalize patent coverage type(s) into a stable list of:
        - composition
        - treatment
        - synthesis
        """
        if value is None:
            return []
        raw_items: List[str] = []
        if isinstance(value, str):
            raw_items = [p.strip() for p in re.split(r"[;,/\\|]+", value) if p.strip()]
        elif isinstance(value, (list, tuple, set)):
            for it in value:
                if it is None:
                    continue
                if isinstance(it, str):
                    raw_items.extend([p.strip() for p in re.split(r"[;,/\\|]+", it) if p.strip()])
        else:
            return []

        out: List[str] = []
        def add(t: str) -> None:
            if t not in out:
                out.append(t)

        for item in raw_items:
            s = item.strip().lower()
            if not s:
                continue
            # Russian + English mapping
            if any(tok in s for tok in ["composition", "compound", "substance", "состав", "веществ", "соединен", "композици", "формуляц"]):
                add("composition")
                continue
            if any(tok in s for tok in ["treatment", "use", "method of", "therapeut", "indication", "способ леч", "применен", "терапевт"]):
                add("treatment")
                continue
            if any(tok in s for tok in ["synthesis", "process", "manufactur", "синтез", "получен", "процесс", "способ получ"]):
                add("synthesis")
                continue
            if s in {"composition", "treatment", "synthesis"}:
                add(s)

        # Stable order
        ordered = [t for t in ["composition", "treatment", "synthesis"] if t in out]
        return ordered

    @staticmethod
    def _family_max_expiry_year(fam: Dict[str, Any]) -> Optional[int]:
        max_year: Optional[int] = None
        cov_rows = fam.get("coverage_by_country") or fam.get("coverage_by_jurisdiction") or []
        for cov in cov_rows:
            expires = cov.get("expires_at")
            if isinstance(expires, str) and len(expires) >= 4 and expires[:4].isdigit():
                year = int(expires[:4])
                if max_year is None or year > max_year:
                    max_year = year
        return max_year

    def _score_patent_family(self, fam: Dict[str, Any], prefer_types: bool = False) -> float:
        score = 0.0
        types = self._normalize_coverage_types(fam.get("coverage_type"))
        if prefer_types and types:
            if "composition" in types:
                score += 3.0
            if "treatment" in types:
                score += 2.0
            if "synthesis" in types:
                score += 1.0

        jurisdictions = set()
        for c in _as_list(fam.get("countries")):
            if c:
                jurisdictions.add(str(c).upper())
        cov_rows = fam.get("coverage_by_country") or fam.get("coverage_by_jurisdiction") or []
        for cov in cov_rows:
            c = cov.get("country") or cov.get("jurisdiction")
            if c:
                jurisdictions.add(str(c).upper())
        score += min(len(jurisdictions), 60) * 0.05

        # Key jurisdictions boost
        if "US" in jurisdictions:
            score += 1.5
        if "EP" in jurisdictions or "EU" in jurisdictions:
            score += 1.2
        if "RU" in jurisdictions:
            score += 0.8

        max_year = self._family_max_expiry_year(fam)
        if max_year is not None:
            now_year = int(time.strftime("%Y", time.gmtime()))
            score += max(0, max_year - now_year) * 0.08
            score += (max_year - 2000) * 0.005  # minor absolute boost for longer-lived families
        else:
            score -= 0.3

        # Prefer families that have a meaningful summary
        if isinstance(fam.get("summary"), str) and fam["summary"].strip():
            score += 0.4
        if fam.get("coverage_by_country") or fam.get("coverage_by_jurisdiction"):
            score += 0.2
        return score

    def _rank_patent_families(self, families: List[Dict[str, Any]], prefer_types: bool = False) -> List[Dict[str, Any]]:
        scored = []
        for fam in families:
            if not isinstance(fam, dict):
                continue
            scored.append((self._score_patent_family(fam, prefer_types=prefer_types), fam))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [fam for _, fam in scored]

    def _normalize_phase(self, phase: Optional[str]) -> str:
        if not phase:
            return "post_marketing"
        phase_val = str(phase).lower()
        if "iii" in phase_val or "3" in phase_val:
            return "phase3"
        if "ii" in phase_val or "2" in phase_val:
            return "phase2"
        if "i" in phase_val or "1" in phase_val:
            return "phase1"
        if "post" in phase_val or "iv" in phase_val or "4" in phase_val:
            return "post_marketing"
        return "post_marketing"

    def _empty_phase_map(self) -> Dict[str, List[Any]]:
        return {
            "phase3": [],
            "phase2": [],
            "phase1": [],
            "post_marketing": []
        }

    def _build_patent_wall_fact(self, patents: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        families = patents.get("blocking_families") or []
        if not families:
            views = patents.get("views") or {}
            ids = views.get("blocking") or []
            fam_by_id = {
                str(f.get("family_id")): f
                for f in _as_list(patents.get("families"))
                if isinstance(f, dict) and f.get("family_id")
            }
            families = [fam_by_id.get(str(fid)) for fid in ids if fam_by_id.get(str(fid))]
        max_year = None
        citations: List[Dict[str, Any]] = []
        for family in families:
            cov_rows = family.get("coverage_by_country") or family.get("coverage_by_jurisdiction") or []
            for cov in cov_rows:
                expires = cov.get("expires_at")
                if isinstance(expires, str) and len(expires) >= 4:
                    try:
                        year = int(expires[:4])
                        if max_year is None or year > max_year:
                            max_year = year
                    except ValueError:
                        continue
            citations.extend(family.get("citations") or [])
        if max_year is None:
            return None
        return self._fact("Патентная стена: до какого года", str(max_year), citations[:3])

    def _build_synthesis_fact(self, synthesis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        steps = synthesis.get("synthesis_route", {}).get("steps") or []
        if not steps:
            return None
        citations: List[Dict[str, Any]] = []
        for step in steps:
            if isinstance(step, dict):
                citations.extend(step.get("citations") or [])
            if len(citations) >= 3:
                break
        if not citations:
            return None
        return self._fact("Синтез", "Есть данные о синтезе", citations[:3])

    def _generate_narrative_summary(self, passport: Dict[str, Any], regulatory: Dict[str, Any],
                                     clinical: Dict[str, Any], patents: Dict[str, Any],
                                     synthesis: Dict[str, Any]) -> str:
        """
        Generate a narrative summary (3-5 sentences) using LLM.
        Input: Passport + Clinical conclusions + Patent status.
        Output: Coherent Russian text describing the drug.
        """
        try:
            # Collect key facts for summary
            inn = passport.get("inn", {}).get("value", "N/A")
            trade_name = passport.get("trade_names", {}).get("value", {})
            drug_class = passport.get("drug_class", {}).get("value", "N/A")
            fda_approval = passport.get("fda_approval", {}).get("value", "N/A")
            registered_in = passport.get("registered_in", {}).get("value", {})

            # Patent expiry
            patent_wall = self._build_patent_wall_fact(patents)
            patent_expiry_year = patent_wall.get("value") if patent_wall else "N/A"

            # Clinical efficacy (simplified)
            clinical_summary = "Клинические данные недоступны"
            for bucket in ("global", "ru", "ongoing"):
                phases = clinical.get(bucket, {})
                if isinstance(phases, dict):
                    for trials in phases.values():
                        for tr in _as_list(trials):
                            if isinstance(tr, dict) and tr.get("efficacy_key_points"):
                                efficacy = tr.get("efficacy_key_points")
                                if efficacy:
                                    clinical_summary = f"Эффективность показана: {efficacy[0] if isinstance(efficacy, list) else efficacy}"
                                    break
                        if clinical_summary != "Клинические данные недоступны":
                            break
                if clinical_summary != "Клинические данные недоступны":
                    break

            # Build prompt for LLM
            prompt = f"""Напиши связный текст-саммари (3-5 предложений) о препарате на русском языке.

Факты:
- МНН: {inn}
- Торговые названия: {trade_name}
- Класс: {drug_class}
- Одобрение FDA: {fda_approval}
- Регистрация: {registered_in}
- Патентная защита до: {patent_expiry_year}
- Клиника: {clinical_summary}

Требования:
1. Текст должен быть связным, без перечислений
2. Начни с "Препарат [Название] (МНН: [МНН]) — это..."
3. Упомяни ключевые факты: класс, одобрение, патенты, эффективность
4. 3-5 предложений, стиль деловой/научный
5. Только факты, никаких домыслов"""

            try:
                response = self.api.send_message(
                    system_content="Ты - медицинский писатель, генерирующий краткие саммари о препаратах.",
                    human_content=prompt,
                    model=self.answering_model or "gpt-4o",
                    temperature=0.3,
                    is_structured=False,
                )
            except Exception as api_error:
                # Handle 401 Unauthorized and other API errors gracefully
                if "401" in str(api_error) or "Unauthorized" in str(api_error):
                    logger.error(f"OpenAI API authentication failed (401 Unauthorized). Check OPENAI_API_KEY: {api_error}")
                else:
                    logger.error(f"OpenAI API error during summary generation: {api_error}")
                return ""

            if response and isinstance(response, str) and len(response.strip()) > 50:
                return response.strip()
            else:
                logger.warning("LLM returned empty or short summary, falling back to simple lines")
                return ""
        except Exception as e:
            logger.warning(f"Failed to generate narrative summary with LLM: {e}")
            return ""

    def _build_summary_lines(self, passport: Dict[str, Any], regulatory: Dict[str, Any],
                              clinical: Dict[str, Any], patents: Dict[str, Any],
                              synthesis: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        inn_fact = passport.get("inn")
        if inn_fact:
            lines.append(f"МНН: {inn_fact.get('value')}")
        reg_fact = passport.get("registered_in")
        if reg_fact:
            lines.append(f"Регистрация: {reg_fact.get('value')}")
        patent_wall = self._build_patent_wall_fact(patents)
        if patent_wall:
            lines.append(f"Патентная защита до: {patent_wall.get('value')}")
        if synthesis.get("synthesis_route", {}).get("steps"):
            lines.append("Синтез: обнаружены шаги из патентов")
        return lines

    def _build_source_stats(self, case_view: Dict[str, Any], documents: List[Dict[str, Any]],
                            unknowns: List[Dict[str, Any]]) -> Dict[str, Any]:
        facts = self._collect_facts(case_view)
        citations_total = sum(len(fact.get("citations") or []) for fact in facts)
        structured_sources = case_view.get("sections", {}).get("sources", {}).get("structured_sources") or []
        return {
            "documents_total": len(documents),
            "sources_total": len(documents) + len(structured_sources),
            "facts_total": len(facts),
            "citations_total": citations_total,
            "gaps_total": len(unknowns)
        }

    def _apply_data_quality(
        self,
        passport: Dict[str, Any],
        brief: Dict[str, Any],
        regulatory: Dict[str, Any],
        clinical: Dict[str, Any],
        patents: Dict[str, Any],
        synthesis: Dict[str, Any],
        sources: Dict[str, Any],
        unknowns: List[Dict[str, Any]],
    ) -> None:
        def dq_from_counts(present: int, ok_threshold: int) -> str:
            if present <= 0:
                return "empty"
            if present < ok_threshold:
                return "partial"
            return "ok"

        # Passport
        required_fields = [
            "inn", "trade_names", "fda_approval", "registered_in",
            "chemical_formula", "drug_class", "registration_holders"
        ]
        present = sum(1 for f in required_fields if self._fact_has_citations(passport.get(f)))
        passport["data_quality"] = "ok" if present >= 5 else ("partial" if present > 0 else "empty")

        # Brief
        summary_ok = bool((brief.get("summary_text") or "").strip())
        key_facts_n = len(brief.get("key_facts") or [])
        brief["data_quality"] = "ok" if (summary_ok and key_facts_n >= 8) else ("partial" if (summary_ok or key_facts_n > 0) else "empty")

        # Regulatory
        def market_fact_count(market: str) -> int:
            block = regulatory.get(market) or {}
            if not isinstance(block, dict):
                return 0
            n = 0
            for v in block.values():
                if isinstance(v, dict) and self._fact_has_citations(v):
                    n += 1
            return n

        us_n = market_fact_count("us")
        eu_n = market_fact_count("eu")
        ru_entries = (regulatory.get("ru") or {}).get("entries") if isinstance(regulatory.get("ru"), dict) else None
        ru_n = len(ru_entries or []) if isinstance(ru_entries, list) else 0
        instr_n = len(regulatory.get("instructions_highlights") or [])

        if isinstance(regulatory.get("us"), dict):
            regulatory["us"]["data_quality"] = dq_from_counts(us_n, ok_threshold=3)
        if isinstance(regulatory.get("eu"), dict):
            regulatory["eu"]["data_quality"] = dq_from_counts(eu_n, ok_threshold=3)
        if isinstance(regulatory.get("ru"), dict):
            regulatory["ru"]["data_quality"] = dq_from_counts(ru_n, ok_threshold=1)

        regulatory["instructions_highlights_data_quality"] = dq_from_counts(instr_n, ok_threshold=8)
        regulatory["data_quality"] = dq_from_counts(us_n + eu_n + ru_n + instr_n, ok_threshold=4)

        # Clinical (avoid inserting keys into phase maps!)
        def bucket_trials(bucket: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            phases = clinical.get(bucket, {})
            if not isinstance(phases, dict):
                return out
            for trials in phases.values():
                for t in _as_list(trials):
                    if isinstance(t, dict):
                        out.append(t)
            return out

        def has_complete_trial_in(bucket: str) -> bool:
            for t in bucket_trials(bucket):
                if (
                    t.get("study_type")
                    and t.get("countries")
                    and t.get("enrollment")
                    and t.get("comparator")
                    and t.get("regimen")
                ):
                    return True
            return False

        dq_clinical: Dict[str, str] = {}
        for bucket in ("global", "ru", "ongoing"):
            trials_n = len(bucket_trials(bucket))
            if trials_n <= 0:
                dq_clinical[bucket] = "empty"
            else:
                dq_clinical[bucket] = "ok" if has_complete_trial_in(bucket) else "partial"

        pubmed = clinical.get("pubmed", {})
        pub_n = 0
        if isinstance(pubmed, dict):
            for v in pubmed.values():
                if isinstance(v, list):
                    pub_n += len(v)
        dq_clinical["pubmed"] = dq_from_counts(pub_n, ok_threshold=4)
        clinical["data_quality"] = dq_clinical

        # Patents
        blocking = patents.get("blocking_families") or []
        if not blocking:
            views = patents.get("views") or {}
            ids = views.get("blocking") or []
            fam_by_id = {
                str(f.get("family_id")): f
                for f in _as_list(patents.get("families"))
                if isinstance(f, dict) and f.get("family_id")
            }
            blocking = [fam_by_id.get(str(fid)) for fid in ids if fam_by_id.get(str(fid))]
        blocking_ok = any(
            isinstance(f, dict) and (f.get("coverage_by_country") or f.get("coverage_by_jurisdiction"))
            for f in blocking
        )
        patents["data_quality"] = "ok" if blocking_ok else ("partial" if blocking else "empty")

        # Synthesis
        steps_n = len(((synthesis.get("synthesis_route") or {}).get("steps") or []))
        synthesis["data_quality"] = dq_from_counts(steps_n, ok_threshold=1)

        # Sources
        docs_n = len(sources.get("documents") or [])
        sources["data_quality"] = dq_from_counts(docs_n, ok_threshold=1)

    def _collect_facts(self, node: Any) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        if isinstance(node, dict):
            if "label" in node and "value" in node and "citations" in node:
                facts.append(node)
            for value in node.values():
                facts.extend(self._collect_facts(value))
        elif isinstance(node, list):
            for item in node:
                facts.extend(self._collect_facts(item))
        return facts

    def _evaluate_quality(self, case_view: Dict[str, Any]) -> Dict[str, Any]:
        passport = case_view.get("passport", {})
        required_fields = [
            "inn", "trade_names", "fda_approval", "registered_in",
            "chemical_formula", "drug_class", "registration_holders"
        ]
        present = 0
        for field in required_fields:
            fact = passport.get(field)
            if self._fact_has_citations(fact):
                present += 1
        passport_gate = {
            "required": 5,
            "present": present,
            "passed": present >= 5
        }

        patents = case_view.get("sections", {}).get("patents", {})
        blocking = patents.get("blocking_families") or []
        if not blocking:
            views = patents.get("views") or {}
            ids = views.get("blocking") or []
            fam_by_id = {
                str(f.get("family_id")): f
                for f in _as_list(patents.get("families"))
                if isinstance(f, dict) and f.get("family_id")
            }
            blocking = [fam_by_id.get(str(fid)) for fid in ids if fam_by_id.get(str(fid))]
        blocking_ok = any(
            isinstance(fam, dict) and (fam.get("coverage_by_country") or fam.get("coverage_by_jurisdiction"))
            for fam in blocking
        )
        patents_gate = {
            "required": 1,
            "present": 1 if blocking_ok else 0,
            "passed": blocking_ok
        }

        clinical = case_view.get("sections", {}).get("clinical", {})
        trial_ok = self._has_complete_trial(clinical)
        clinical_gate = {
            "required": 1,
            "present": 1 if trial_ok else 0,
            "passed": trial_ok
        }

        facts = self._collect_facts(case_view)
        facts_total = len(facts)
        facts_with_citations = sum(1 for fact in facts if self._fact_has_citations(fact))
        facts_missing_citations = facts_total - facts_with_citations
        citations_gate = {
            "required": facts_total,
            "present": facts_with_citations,
            "missing": facts_missing_citations,
            "passed": facts_missing_citations == 0
        }

        ready = all(g["passed"] for g in [passport_gate, patents_gate, clinical_gate, citations_gate])
        return {
            "ready_for_ui": ready,
            "checks": {
                "passport_min_fields": passport_gate,
                "patents_blocking_families": patents_gate,
                "clinical_trial_complete": clinical_gate,
                "facts_have_citations": citations_gate
            }
        }

    def _has_complete_trial(self, clinical: Dict[str, Any]) -> bool:
        for bucket in ("global", "ru", "ongoing"):
            phases = clinical.get(bucket, {})
            for trials in phases.values():
                for trial in _as_list(trials):
                    if (
                        trial.get("study_type")
                        and trial.get("countries")
                        and trial.get("enrollment")
                        and trial.get("comparator")
                        and trial.get("regimen")
                    ):
                        return True
        return False
