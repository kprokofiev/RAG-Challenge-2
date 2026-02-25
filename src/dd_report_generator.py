import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.api_requests import APIProcessor
from src.evidence_builder import EvidenceCandidatesBuilder
from src.prompts import DDSectionAnswerPrompt, DDSectionAnswerSchema, build_system_prompt
from src.retrieval import HybridRetriever
from src.validation_gates import ValidationGates

# ── Adaptive top-K table (Sprint 3, §3.1) ────────────────────────────────────────
# Maps section_id prefix (or exact id) → evidence candidate K to use.
# Regulatory/instruction: 20-30 (dense, structured docs).
# Clinical results: 40-60 (numbers scattered across tables).
# Patents: 40-80 (long text + many relevant sections).
# Publications/RWE: 40-50 (varying depth).
_ADAPTIVE_K_TABLE: Dict[str, int] = {
    # Regulatory
    "regulatory_status_eu": 25,
    "regulatory_status_us": 25,
    "regulatory_status_ru": 20,
    "safety_profile": 30,
    # Clinical
    "clinical_trials_overview": 40,
    "clinical_results": 55,
    "clinical_protocols": 40,
    "clinical_efficacy_publications": 45,
    "clinical_congress_abstracts": 35,
    "clinical_preprints": 30,
    # RWE / trial registry
    "rwe_evidence": 50,
    "trial_registry": 40,
    # Patents
    "patents": 60,
    "synthesis": 50,
    # Manufacturers / other
    "manufacturers": 25,
    "manufacturing_quality": 25,
}

# Prefixes for pattern matching (section_id.startswith(prefix))
_ADAPTIVE_K_PREFIXES: List[tuple] = [
    ("regulatory_", 25),
    ("clinical_result", 55),
    ("clinical_trial", 40),
    ("clinical_", 40),
    ("rwe_", 50),
    ("patent", 60),
    ("safety", 30),
    ("manufactur", 25),
]


class DDReportGenerator:
    def __init__(
        self,
        documents_dir: Path,
        vector_db_dir: Path,
        tenant_id: Optional[str] = None,
        case_id: Optional[str] = None,
        ddkit_db: Optional[Any] = None,
        inn: Optional[str] = None,
    ):
        self.documents_dir = documents_dir
        self.vector_db_dir = vector_db_dir
        self.tenant_id = tenant_id
        self.case_id = case_id
        self.ddkit_db = ddkit_db
        # Normalize INN: lowercase for matching, keep original for display/query augment
        self.inn: Optional[str] = inn.strip() if inn and inn.strip() else None
        self.inn_lower: Optional[str] = self.inn.lower() if self.inn else None
        self.retriever = HybridRetriever(vector_db_dir, documents_dir)
        self.api = APIProcessor(provider=os.getenv("DDKIT_LLM_PROVIDER", "openai"))
        self.answering_model = os.getenv("DDKIT_ANSWER_MODEL", None)
        self.evidence_builder = EvidenceCandidatesBuilder()
        self.validator = ValidationGates()
        # Top-K evidence candidates per question (#9). Configurable via env; default 25.
        # Larger K reduces false-unknowns by making 11-25th ranked chunks available to LLM.
        try:
            self._top_k = max(1, int(os.getenv("DDKIT_EVIDENCE_TOP_K", "25")))
        except (ValueError, TypeError):
            self._top_k = 25
        # sections_plan version — injected into report JSON for traceability
        self._sections_plan_version: Optional[str] = None

    # ── Adaptive K (Sprint 3, §3.1) ──────────────────────────────────────────────

    def _adaptive_top_k(self, section_id: str, doc_kind: Any = None) -> int:
        """
        Return evidence candidate K appropriate for this section type.

        Priority:
        1. Env override DDKIT_EVIDENCE_TOP_K (global cap, if set explicitly).
        2. doc_kind-based heuristic (patents are deep → high K).
        3. Exact section_id table lookup.
        4. Prefix match.
        5. Global default self._top_k.
        """
        # doc_kind heuristic overrides table when the caller asks for a very specific type
        if doc_kind:
            dk_str = str(doc_kind) if isinstance(doc_kind, str) else ",".join(str(d) for d in (doc_kind or []))
            if any(p in dk_str for p in ("patent", "ops", "patent_family")):
                k = 60
                return max(k, self._top_k)
            if any(p in dk_str for p in ("ctgov_results", "rwe_study", "rwe_safety", "scientific")):
                k = 50
                return max(k, self._top_k)
        # Exact match
        if section_id in _ADAPTIVE_K_TABLE:
            return max(_ADAPTIVE_K_TABLE[section_id], self._top_k)
        # Prefix match
        for prefix, k in _ADAPTIVE_K_PREFIXES:
            if section_id.startswith(prefix):
                return max(k, self._top_k)
        return self._top_k

    # ── Completeness tracking ────────────────────────────────────────────────

    def _build_completeness(
        self,
        included_docs: List[Dict[str, Any]],
        is_partial: bool,
        partial_reasons: List[str],
    ) -> Dict[str, Any]:
        """
        Build the `completeness` block that is added to every DD report.

        expected = all indexed docs in the DB for this case (if DB configured).
        included = docs that the generator actually downloaded and used.
        missing  = expected(indexed) minus included.
        """
        # ---- included counts (from downloaded chunk files) ------------------
        included_by_kind: Dict[str, int] = defaultdict(int)
        for d in included_docs:
            k = d.get("kind") or "unknown"
            included_by_kind[k] += 1
        included_total = len(included_docs)

        # ---- expected counts (from DB) --------------------------------------
        expected_by_kind: Dict[str, int] = {}
        expected_total = 0
        missing_by_kind: Dict[str, int] = {}
        missing_total = 0

        if self.ddkit_db is not None and getattr(self.ddkit_db, "is_configured", lambda: False)():
            try:
                db_docs = self.ddkit_db.list_case_documents(
                    tenant_id=self.tenant_id or "",
                    case_id=self.case_id or "",
                )
                for d in db_docs:
                    if str(d.get("status", "")).lower() == "indexed":
                        k = d.get("doc_kind") or "unknown"
                        expected_by_kind[k] = expected_by_kind.get(k, 0) + 1
                        expected_total += 1

                included_kind_set: Dict[str, int] = dict(included_by_kind)
                for k, exp_n in expected_by_kind.items():
                    inc_n = included_kind_set.get(k, 0)
                    diff = exp_n - inc_n
                    if diff > 0:
                        missing_by_kind[k] = diff
                        missing_total += diff
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "completeness DB query failed (non-fatal): %s", exc
                )

        raw_ratio = (included_total / expected_total) if expected_total > 0 else 1.0
        # Cap ratio at 1.0 — over-inclusion (duplicates/extra docs) must not inflate the metric (#11).
        completeness_ratio = min(raw_ratio, 1.0)
        over_included = max(0, included_total - expected_total) if expected_total > 0 else 0
        logging.getLogger(__name__).info(
            "completeness: included=%d expected=%d missing=%d ratio=%.2f over_included=%d is_partial=%s",
            included_total, expected_total, missing_total, completeness_ratio, over_included, is_partial,
        )

        return {
            "is_partial": is_partial,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "expected": {
                "total": expected_total,
                "by_kind": expected_by_kind,
            },
            "included": {
                "total": included_total,
                "by_kind": dict(included_by_kind),
            },
            "missing": {
                "total": missing_total,
                "by_kind": missing_by_kind,
            },
            "completeness_ratio": round(completeness_ratio, 4),
            "over_included": over_included,
            "reasons": partial_reasons,
        }

    # ── Main report generation ───────────────────────────────────────────────

    def generate_report(
        self,
        sections_plan: Dict[str, Any],
        deadline: Optional[float] = None,
        is_partial: bool = False,
        partial_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        logger = logging.getLogger(__name__)
        start_ts = time.time()

        # Extract plan metadata (version etc.) from _meta key if present
        plan_meta = sections_plan.get("_meta", {}) if isinstance(sections_plan, dict) else {}
        sections_plan_version = (
            plan_meta.get("sections_plan_version")
            or self._sections_plan_version
            or os.getenv("DDKIT_SECTIONS_PLAN_VERSION", "unknown")
        )

        documents = self._load_documents_meta()
        doc_titles = {d["doc_id"]: d.get("title") for d in documents if d.get("doc_id")}

        evidence_index_map: Dict[str, Dict[str, Any]] = {}
        sections_output: List[Dict[str, Any]] = []

        logger.info(
            "DD report generation started (sections=%d, documents=%d, plan_version=%s)",
            len(sections_plan),
            len(documents),
            sections_plan_version,
        )
        if deadline is not None:
            logger.info("Report generation timeout in %.2fs", max(0.0, deadline - start_ts))

        # Skip the _meta pseudo-section key
        for sec_key, section_cfg in sections_plan.items():
            if sec_key == "_meta" or not isinstance(section_cfg, dict):
                continue
            self._check_deadline(deadline, "before_section")
            section_id = section_cfg.get("section_id") or section_cfg.get("id") or sec_key
            title = section_cfg.get("title", section_id)
            questions = section_cfg.get("questions", [])
            authority_scope = section_cfg.get("authority_scope")
            section_start = time.time()
            logger.info(
                "Generating section '%s' (%s), questions=%d, authority_scope=%s",
                title,
                section_id,
                len(questions),
                authority_scope or "-",
            )

            section_output = {
                "section_id": section_id,
                "title": title,
                "claims": [],
                "numbers": [],
                "risks": [],
                "unknowns": [],
                "evidence": []
            }
            section_evidence_ids: set[str] = set()

            for question in questions:
                self._check_deadline(deadline, f"before_question:{section_id}")
                q_text = question.get("question", "")
                if not q_text:
                    continue
                doc_kind = self._parse_doc_kind(question.get("doc_kind_preference"))
                focus_terms = self._parse_focus_terms(
                    question.get("focus_terms") or question.get("drug_terms")
                )
                query_text = q_text
                if focus_terms:
                    query_text = f"{q_text} {' '.join(focus_terms)}"
                # Always augment query with INN so vector search is drug-anchored,
                # even for sections whose focus_terms are schema field names.
                if self.inn and self.inn_lower not in query_text.lower():
                    query_text = f"{query_text} {self.inn}"
                logger.info(
                    "Question: %s | doc_kind=%s | focus_terms=%s | inn=%s",
                    q_text[:160],
                    doc_kind,
                    ",".join(focus_terms) if focus_terms else "-",
                    self.inn or "-",
                )
                t0 = time.time()
                # Adaptive top-K (#9, Sprint 3 §3.1): section type determines evidence depth.
                adaptive_k = self._adaptive_top_k(section_id, doc_kind)
                # Retrieve more candidates when focus_terms are present (broader recall needed).
                retrieve_limit = max(adaptive_k, 30) if focus_terms else adaptive_k
                retrieved = self.retriever.retrieve_by_case(
                    query=query_text,
                    final_candidates_k=retrieve_limit,
                    dense_k=min(retrieve_limit * 2, 100),
                    sparse_k=min(retrieve_limit * 2, 100),
                    rerank_sample_k=min(retrieve_limit * 3, 150),
                    tenant_id=self.tenant_id,
                    case_id=self.case_id,
                    doc_kind=doc_kind,
                    authority_scope=authority_scope,
                )
                if focus_terms and not self._is_schema_field_terms(focus_terms):
                    before_count = len(retrieved)
                    retrieved = [item for item in retrieved if self._candidate_matches_terms(item, focus_terms)]
                    retrieved = retrieved[:adaptive_k]
                    logger.info(
                        "Filtered retrieved passages by focus_terms (%s): %d -> %d (adaptive_k=%d)",
                        ",".join(focus_terms),
                        before_count,
                        len(retrieved),
                        adaptive_k,
                    )
                elif focus_terms and self._is_schema_field_terms(focus_terms):
                    logger.info(
                        "focus_terms (%s) detected as schema field names — skipping candidate filter, "
                        "keeping top-%d from retrieval",
                        ",".join(focus_terms),
                        adaptive_k,
                    )
                    retrieved = retrieved[:adaptive_k]
                else:
                    retrieved = retrieved[:adaptive_k]
                # INN-based relevance scoring: boost passages that mention INN/synonyms,
                # but never hard-exclude — regulatory doc_kind items always pass (#8).
                # This prevents false-unknowns when a publication uses a brand name or salt form.
                if self.inn_lower and retrieved:
                    # Regulatory doc kinds that should always be kept regardless of INN mention.
                    _regulatory_kinds = {"label", "smpc", "pil", "epar", "grls", "grls_card",
                                         "ru_instruction", "us_fda", "assessment_report"}
                    inn_terms = self._build_inn_terms(focus_terms)
                    inn_matched = []
                    inn_unmatched = []
                    for item in retrieved:
                        dk = str(item.get("doc_kind") or item.get("kind") or "").lower()
                        haystack = (item.get("text", "") + " " + item.get("doc_title", "")).lower()
                        if dk in _regulatory_kinds or any(t in haystack for t in inn_terms):
                            inn_matched.append(item)
                        else:
                            inn_unmatched.append(item)
                    if inn_matched:
                        logger.info(
                            "INN filter (%s): %d matched, %d unmatched (unmatched demoted, not removed)",
                            self.inn, len(inn_matched), len(inn_unmatched),
                        )
                        # Prefer matched but append unmatched at the end so LLM still has them
                        retrieved = inn_matched + inn_unmatched
                    else:
                        logger.info(
                            "INN filter (%s): no passages matched — keeping all %d (soft fallback)",
                            self.inn, len(retrieved),
                        )
                self._check_deadline(deadline, f"after_retrieval:{section_id}")
                logger.info(
                    "Retrieved %d passages for question '%s' in %.2fs",
                    len(retrieved),
                    q_text[:80],
                    time.time() - t0
                )
                retrieved_by_doc: Dict[str, List[Dict[str, Any]]] = {}
                for item in retrieved:
                    doc_id = item.get("doc_id", "unknown")
                    retrieved_by_doc.setdefault(doc_id, []).append(item)

                candidates = self.evidence_builder.build_candidates_from_multiple_docs(
                    retrieved_by_doc, doc_titles=doc_titles
                )
                logger.info(
                    "Evidence candidates: %d (doc_kind=%s)",
                    len(candidates),
                    doc_kind or "any"
                )
                candidates_prompt = self.evidence_builder.candidates_to_prompt_format(candidates)

                system_prompt = build_system_prompt(
                    DDSectionAnswerPrompt.instruction,
                    DDSectionAnswerPrompt.example,
                    DDSectionAnswerPrompt.pydantic_schema
                )
                user_prompt = DDSectionAnswerPrompt.user_prompt.format(
                    context=self._format_context(retrieved),
                    question=q_text,
                    answer_type=question.get("answer_type", "facts"),
                    scope=question.get("scope", "UNKNOWN"),
                    evidence_candidates=candidates_prompt
                )

                t1 = time.time()
                answer = self.api.send_message(
                    model=self.answering_model,
                    system_content=system_prompt,
                    human_content=user_prompt,
                    is_structured=True,
                    response_format=DDSectionAnswerSchema
                )
                self._check_deadline(deadline, f"after_llm:{section_id}")
                logger.info(
                    "LLM answered question in %.2fs (claims=%d, numbers=%d, risks=%d)",
                    time.time() - t1,
                    len(answer.get("claims", [])),
                    len(answer.get("numbers", [])),
                    len(answer.get("risks", []))
                )
                answer["section_id"] = section_id
                answer.setdefault("evidence", [])
                validation = self.validator.validate_section_output(answer, candidates)
                if validation.fixed_output:
                    answer = validation.fixed_output
                    logger.info(
                        "repair loop fixed section %s (remapped+moved to unknowns)",
                        section_id,
                    )
                # Pass candidates so move_orphaned_to_unknowns validates ID existence, not just presence
                answer = self.validator.move_orphaned_to_unknowns(answer, evidence_candidates=candidates)
                logger.info(
                    "Validation result for %s: valid=%s errors=%d unknowns_now=%d needs_expand_k=%s",
                    section_id,
                    validation.is_valid,
                    len(validation.errors),
                    len(answer.get("unknowns", [])),
                    getattr(validation, "_needs_expand_k", False),
                )

                section_output["claims"].extend(answer.get("claims", []))
                section_output["numbers"].extend(answer.get("numbers", []))
                section_output["risks"].extend(answer.get("risks", []))
                section_output["unknowns"].extend(answer.get("unknowns", []))

                used_ids = self._collect_evidence_ids(answer)
                for candidate in candidates:
                    if candidate.evidence_id in used_ids:
                        evidence_index_map[candidate.evidence_id] = {
                            "id": candidate.evidence_id,
                            "doc_id": candidate.doc_id,
                            "doc_title": candidate.doc_title,
                            "page": candidate.page,
                            "snippet": candidate.snippet
                        }
                        section_evidence_ids.add(candidate.evidence_id)

            section_output["evidence"] = [evidence_index_map[eid] for eid in section_evidence_ids if eid in evidence_index_map]
            sections_output.append(section_output)
            logger.info("Section '%s' completed in %.2fs", section_id, time.time() - section_start)

        logger.info("DD report generation finished in %.2fs", time.time() - start_ts)
        completeness = self._build_completeness(
            included_docs=documents,
            is_partial=bool(is_partial),
            partial_reasons=list(partial_reasons or []),
        )
        return {
            "report_id": f"dd_report_{self.case_id}_{int(time.time())}",
            "case_id": self.case_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sections_plan_version": sections_plan_version,
            "sections": sections_output,
            "evidence_index": list(evidence_index_map.values()),
            "documents": documents,
            "completeness": completeness,
        }

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
                    "kind": meta.get("doc_kind"),
                    "source_url": meta.get("source_url")
                })
            except Exception:
                continue
        return documents

    def _format_context(self, retrieved: List[Dict[str, Any]]) -> str:
        parts = []
        for item in retrieved:
            page_number = item.get("page")
            text = item.get("text", "")
            parts.append(f'Text retrieved from page {page_number}:\n"""\n{text}\n"""')
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _check_deadline(deadline: Optional[float], stage: str) -> None:
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"report_timeout at {stage}")

    @staticmethod
    def _parse_doc_kind(raw_value: Any) -> Optional[Union[str, List[str]]]:
        if raw_value is None:
            return None
        if isinstance(raw_value, (list, tuple, set)):
            parts = [str(item).strip() for item in raw_value if str(item).strip()]
            return parts or None
        if isinstance(raw_value, str):
            if "," in raw_value:
                parts = [part.strip() for part in raw_value.split(",") if part.strip()]
                return parts or None
            return raw_value.strip() or None
        return None

    @staticmethod
    def _parse_focus_terms(raw_value: Any) -> List[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            items = [raw_value]
        elif isinstance(raw_value, (list, tuple, set)):
            items = list(raw_value)
        else:
            return []
        terms: List[str] = []
        for item in items:
            term = str(item).strip().lower()
            if term and term not in terms:
                terms.append(term)
        return terms

    @staticmethod
    def _is_schema_field_terms(terms: List[str]) -> bool:
        """Return True if terms look like router-map extraction_targets (snake_case field names).

        Extraction targets like 'one_sentence_moa', 'trial_name', 'primary_endpoint_result'
        are data-schema field names, not pharmacological search terms.  They must not be used
        for candidate filtering because they never appear literally in document text and would
        eliminate all retrieved passages.

        Heuristic: if the majority of terms contain underscores and no spaces, treat as schema fields.
        """
        if not terms:
            return False
        schema_like = sum(1 for t in terms if "_" in t and " " not in t)
        return schema_like >= max(1, len(terms) // 2)

    @staticmethod
    def _candidate_matches_terms(item: Dict[str, Any], terms: List[str]) -> bool:
        if not terms:
            return True
        haystack = " ".join([
            item.get("text", ""),
            item.get("doc_title", "")
        ]).lower()
        return any(term in haystack for term in terms)

    def _build_inn_terms(self, focus_terms: List[str]) -> List[str]:
        """
        Build a list of lowercase terms to use for INN-matching in the soft filter (#8).

        Includes:
        - INN itself
        - Any non-schema focus_terms (brand names, salt forms, abbreviations passed by caller)
        - Common salt/ester suffixes derived from INN (e.g. "metformin" -> "metformin hydrochloride")
        """
        terms: List[str] = []
        if self.inn_lower:
            terms.append(self.inn_lower)
            # Add common salt/ester variants
            for suffix in (" hydrochloride", " hcl", " sodium", " potassium",
                           " acetate", " succinate", " mesylate", " tartrate",
                           " phosphate", " sulfate", " maleate", " fumarate"):
                terms.append(self.inn_lower + suffix)
        # Add non-schema focus terms as synonyms (brand names, codes, etc.)
        if focus_terms and not self._is_schema_field_terms(focus_terms):
            for t in focus_terms:
                if t and t not in terms:
                    terms.append(t.lower())
        return terms

    @staticmethod
    def _collect_evidence_ids(answer: Dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        for section_key in ("claims", "numbers", "risks"):
            for item in answer.get(section_key, []):
                for evidence_id in item.get("evidence_ids", []):
                    ids.add(evidence_id)
        return ids
