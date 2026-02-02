import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.api_requests import APIProcessor
from src.evidence_builder import EvidenceCandidatesBuilder
from src.prompts import DDSectionAnswerPrompt, DDSectionAnswerSchema, build_system_prompt
from src.retrieval import HybridRetriever
from src.validation_gates import ValidationGates


class DDReportGenerator:
    def __init__(self, documents_dir: Path, vector_db_dir: Path,
                 tenant_id: Optional[str] = None, case_id: Optional[str] = None):
        self.documents_dir = documents_dir
        self.vector_db_dir = vector_db_dir
        self.tenant_id = tenant_id
        self.case_id = case_id
        self.retriever = HybridRetriever(vector_db_dir, documents_dir)
        self.api = APIProcessor(provider=os.getenv("DDKIT_LLM_PROVIDER", "openai"))
        self.answering_model = os.getenv("DDKIT_ANSWER_MODEL", None)
        self.evidence_builder = EvidenceCandidatesBuilder()
        self.validator = ValidationGates()

    def generate_report(self, sections_plan: Dict[str, Any], deadline: Optional[float] = None) -> Dict[str, Any]:
        logger = logging.getLogger(__name__)
        start_ts = time.time()
        documents = self._load_documents_meta()
        doc_titles = {d["doc_id"]: d.get("title") for d in documents if d.get("doc_id")}

        evidence_index_map: Dict[str, Dict[str, Any]] = {}
        sections_output: List[Dict[str, Any]] = []

        logger.info(
            "DD report generation started (sections=%d, documents=%d)",
            len(sections_plan),
            len(documents)
        )
        if deadline is not None:
            logger.info("Report generation timeout in %.2fs", max(0.0, deadline - start_ts))

        for _, section_cfg in sections_plan.items():
            self._check_deadline(deadline, "before_section")
            section_id = section_cfg.get("section_id") or section_cfg.get("id")
            title = section_cfg.get("title", section_id)
            questions = section_cfg.get("questions", [])
            section_start = time.time()
            logger.info(
                "Generating section '%s' (%s), questions=%d",
                title,
                section_id,
                len(questions)
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
                logger.info(
                    "Question: %s | doc_kind=%s | focus_terms=%s",
                    q_text[:160],
                    doc_kind,
                    ",".join(focus_terms) if focus_terms else "-"
                )
                t0 = time.time()
                retrieve_limit = 30 if focus_terms else 10
                retrieved = self.retriever.retrieve_by_case(
                    query=query_text,
                    top_n=retrieve_limit,
                    tenant_id=self.tenant_id,
                    case_id=self.case_id,
                    doc_kind=doc_kind
                )
                if focus_terms:
                    before_count = len(retrieved)
                    retrieved = [item for item in retrieved if self._candidate_matches_terms(item, focus_terms)]
                    retrieved = retrieved[:10]
                    logger.info(
                        "Filtered retrieved passages by focus_terms (%s): %d -> %d",
                        ",".join(focus_terms),
                        before_count,
                        len(retrieved)
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
                answer = self.validator.move_orphaned_to_unknowns(answer)

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
        return {
            "report_id": f"dd_report_{self.case_id}_{int(time.time())}",
            "case_id": self.case_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sections": sections_output,
            "evidence_index": list(evidence_index_map.values()),
            "documents": documents
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
    def _candidate_matches_terms(item: Dict[str, Any], terms: List[str]) -> bool:
        if not terms:
            return True
        haystack = " ".join([
            item.get("text", ""),
            item.get("doc_title", "")
        ]).lower()
        return any(term in haystack for term in terms)

    @staticmethod
    def _collect_evidence_ids(answer: Dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        for section_key in ("claims", "numbers", "risks"):
            for item in answer.get(section_key, []):
                for evidence_id in item.get("evidence_ids", []):
                    ids.add(evidence_id)
        return ids
