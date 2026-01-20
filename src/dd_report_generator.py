import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

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

    def generate_report(self, sections_plan: Dict[str, Any]) -> Dict[str, Any]:
        documents = self._load_documents_meta()
        doc_titles = {d["doc_id"]: d.get("title") for d in documents if d.get("doc_id")}

        evidence_index_map: Dict[str, Dict[str, Any]] = {}
        sections_output: List[Dict[str, Any]] = []

        for _, section_cfg in sections_plan.items():
            section_id = section_cfg.get("section_id") or section_cfg.get("id")
            title = section_cfg.get("title", section_id)
            questions = section_cfg.get("questions", [])

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
                q_text = question.get("question", "")
                if not q_text:
                    continue
                doc_kind = question.get("doc_kind_preference")
                retrieved = self.retriever.retrieve_by_case(
                    query=q_text,
                    top_n=10,
                    tenant_id=self.tenant_id,
                    case_id=self.case_id,
                    doc_kind=doc_kind
                )
                retrieved_by_doc: Dict[str, List[Dict[str, Any]]] = {}
                for item in retrieved:
                    doc_id = item.get("doc_id", "unknown")
                    retrieved_by_doc.setdefault(doc_id, []).append(item)

                candidates = self.evidence_builder.build_candidates_from_multiple_docs(
                    retrieved_by_doc, doc_titles=doc_titles
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

                answer = self.api.send_message(
                    model=self.answering_model,
                    system_content=system_prompt,
                    human_content=user_prompt,
                    is_structured=True,
                    response_format=DDSectionAnswerSchema
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
    def _collect_evidence_ids(answer: Dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        for section_key in ("claims", "numbers", "risks"):
            for item in answer.get(section_key, []):
                for evidence_id in item.get("evidence_ids", []):
                    ids.add(evidence_id)
        return ids
