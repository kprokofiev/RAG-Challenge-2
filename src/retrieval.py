import json
import logging
from typing import Any, List, Tuple, Dict, Optional, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)

# ── BM25 in-process cache (P1): avoids rebuilding index on every question ──────────
# Key: (case_id, tenant_id, doc_kind_key) → (corpus_chunks_list, BM25Okapi_instance)
# Invalidated when the number of chunks changes (new documents indexed mid-run is rare).
_BM25_CACHE: Dict[tuple, Any] = {}


# ── Authority tiers: maps section scope → (tier-1 doc_kinds, tier-2 doc_kinds) ──────
# Retriever first fills quota from tier-1; supplements with tier-2 when tier-1 < threshold.
AUTHORITY_TIERS: Dict[str, Dict[str, List[str]]] = {
    # Regulatory registers / instructions
    "EU_REG": {
        "tier_1": ["epar", "smpc", "assessment_report", "pil"],
        "tier_2": ["label", "us_fda", "grls_card", "grls", "ru_instruction"],
    },
    "US_REG": {
        "tier_1": ["label", "us_fda", "approval_letter"],
        "tier_2": ["epar", "smpc", "grls_card"],
    },
    "RU_REG": {
        "tier_1": ["grls_card", "grls", "ru_instruction"],
        "tier_2": ["smpc", "epar", "label"],
    },
    # Clinical
    "CLINICAL": {
        "tier_1": ["ctgov_results", "ctgov_protocol", "ctgov", "ctgov_lay_summary", "clinical_trials"],
        "tier_2": ["scientific_pmc", "scientific_pdf", "publication", "congress_abstract"],
    },
    "CLINICAL_US": {
        "tier_1": ["ctgov_protocol", "ctgov", "ctgov_results"],
        "tier_2": ["label", "us_fda", "publication"],
    },
    "PUBLICATIONS": {
        "tier_1": ["scientific_pmc", "scientific_pdf", "publication"],
        "tier_2": ["congress_abstract", "preprint", "rwe_study"],
    },
    "CONGRESS": {
        "tier_1": ["congress_abstract", "poster"],
        "tier_2": ["scientific_pdf", "scientific_pmc", "publication"],
    },
    "PREPRINTS": {
        "tier_1": ["preprint"],
        "tier_2": ["scientific_pmc", "publication"],
    },
    "RWE": {
        "tier_1": ["rwe_study", "rwe_safety"],
        "tier_2": ["scientific_pmc", "publication", "congress_abstract"],
    },
    "TRIAL_REGISTRY": {
        "tier_1": ["trial_registry", "ctgov_protocol", "ctgov"],
        "tier_2": ["publication", "congress_abstract"],
    },
    # Patents / chemistry
    "PATENTS": {
        "tier_1": ["patent_family", "ops", "patent_pdf"],
        "tier_2": ["patent", "patent_text", "drug_monograph"],
    },
    "CHEMISTRY": {
        "tier_1": ["patent_pdf", "drug_monograph"],
        "tier_2": ["patent_family", "scientific_pdf"],
    },
    # Safety
    "SAFETY": {
        "tier_1": ["label", "smpc", "epar", "us_fda", "pil"],
        "tier_2": ["rwe_safety", "scientific_pdf", "congress_abstract"],
    },
    # Manufacturing
    "MANUFACTURING": {
        "tier_1": ["manufacturers", "epar"],
        "tier_2": ["smpc", "label", "grls_card"],
    },
}

# Minimum number of tier-1 results to consider tier-1 "sufficient" (below → supplement w/ tier-2)
_TIER1_SUFFICIENT_THRESHOLD = 3


def _get_llm_timeout_seconds() -> float:
    raw = os.getenv("DDKIT_LLM_TIMEOUT_SECONDS", "120")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 120.0

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break
                    
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")
            
        # Load corresponding BM25 index
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # Get the document content and BM25 index
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # Get BM25 scores for the query
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results

    def retrieve_by_case(
        self,
        query: str,
        top_n: int = 20,
        tenant_id: Optional[str] = None,
        case_id: Optional[str] = None,
        doc_kind: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict]:
        """
        Retrieve top-N BM25 chunks across all chunk JSON files for the given case.

        Builds an in-memory BM25 index from all chunks in documents_dir that match
        the tenant/case/doc_kind filters.  Returns scored dicts compatible with
        VectorRetriever.retrieve_by_case() output so results can be merged and reranked.
        """
        # ── P1 cache key: (case_id, tenant_id, normalised doc_kind string) ──────────────
        _dk_key = ",".join(sorted(doc_kind)) if isinstance(doc_kind, list) else (doc_kind or "")
        _cache_key = (case_id or "", tenant_id or "", _dk_key)
        _cached = _BM25_CACHE.get(_cache_key)

        if _cached is not None:
            all_chunks, bm25_index = _cached
            _log.debug("BM25 cache hit: case=%s doc_kind=%s (%d chunks)", case_id, _dk_key, len(all_chunks))
        else:
            all_chunks = []

            for doc_path in self.documents_dir.glob("*.json"):
                try:
                    with open(doc_path, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                except Exception as exc:
                    _log.debug("BM25: skip %s — %s", doc_path.name, exc)
                    continue

                metainfo = doc.get("metainfo", {})
                if tenant_id and metainfo.get("tenant_id") != tenant_id:
                    continue
                if case_id and metainfo.get("case_id") != case_id:
                    continue
                if doc_kind:
                    if not VectorRetriever._doc_kind_matches(metainfo.get("doc_kind", ""), doc_kind):
                        continue

                doc_id = metainfo.get("doc_id", doc_path.stem)
                doc_title = metainfo.get("title") or metainfo.get("company_name") or doc_id

                for chunk in doc.get("content", {}).get("chunks", []):
                    all_chunks.append({
                        "_text": chunk.get("text", ""),
                        "_chunk": chunk,
                        "_doc_id": doc_id,
                        "_doc_title": doc_title,
                        "_metainfo": metainfo,
                    })

            if not all_chunks:
                _log.debug("BM25 retrieve_by_case: no chunks found (case=%s, doc_kind=%s)", case_id, doc_kind)
                return []

            # Tokenise and build BM25 index on the fly
            tokenized_corpus = [c["_text"].lower().split() for c in all_chunks]
            try:
                bm25_index = BM25Okapi(tokenized_corpus)
            except Exception as exc:
                _log.warning("BM25 index build failed: %s", exc)
                return []

            # Store in process-level cache
            _BM25_CACHE[_cache_key] = (all_chunks, bm25_index)
            _log.debug("BM25 cache stored: case=%s doc_kind=%s (%d chunks)", case_id, _dk_key, len(all_chunks))

        tokenized_query = query.lower().split()
        scores = bm25_index.get_scores(tokenized_query)

        actual_top_n = min(top_n, len(all_chunks))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]

        results: List[Dict] = []
        for idx in top_indices:
            score = round(float(scores[idx]), 4)
            if score <= 0.0:
                continue  # skip zero-score entries
            item = all_chunks[idx]
            chunk = item["_chunk"]
            page_num = chunk.get("page", chunk.get("page_from", 0))
            results.append({
                "distance": score,
                "page": page_num,
                "text": chunk.get("text", ""),
                "type": chunk.get("type", "content"),
                "doc_id": item["_doc_id"],
                "doc_title": item["_doc_title"],
                "_retrieval_source": "bm25",
            })

        _log.info(
            "BM25 retrieve_by_case: case=%s doc_kind=%s → %d chunks indexed, %d results (top score=%.3f)",
            case_id, doc_kind, len(all_chunks), len(results),
            results[0]["distance"] if results else 0.0,
        )
        return results


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=_get_llm_timeout_seconds(),
            max_retries=2
            )
        return llm
    
    @staticmethod
    def set_up_llm():
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=_get_llm_timeout_seconds(),
            max_retries=2
            )
        return llm

    def _load_dbs(self):
        all_dbs = []
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}
        
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            # Validate that the document meets the expected schema
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def _doc_kind_matches(value: str, pattern: Union[str, List[str], Tuple[str, ...], set]) -> bool:
        if not value or not pattern:
            return False
        if isinstance(pattern, (list, tuple, set)):
            return any(VectorRetriever._doc_kind_matches(value, item) for item in pattern if item)
        if not isinstance(pattern, str):
            return False
        if "*" in pattern:
            prefix = pattern.rstrip("*")
            return value.startswith(prefix)
        return value == pattern

    @staticmethod
    def _match_filters(metainfo: dict, tenant_id: str = None, case_id: str = None, doc_kind: Union[str, List[str], Tuple[str, ...], set] = None) -> bool:
        if tenant_id and metainfo.get("tenant_id") != tenant_id:
            return False
        if case_id and metainfo.get("case_id") != case_id:
            return False
        if doc_kind and metainfo.get("doc_kind") != doc_kind:
            if not VectorRetriever._doc_kind_matches(metainfo.get("doc_kind"), doc_kind):
                return False
        return True

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False, tenant_id: str = None, case_id: str = None, doc_kind: str = None) -> List[Tuple[str, float]]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        actual_top_n = min(top_n, len(chunks))
        
        embedding = self.llm.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # L2-normalize query vector to match normalized index vectors (#6).
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)

        retrieval_results = []
        seen_pages = set()
        
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
            
        return retrieval_results

    def retrieve_by_case(
        self,
        query: str,
        top_n: int = 6,
        return_parent_pages: bool = False,
        tenant_id: str = None,
        case_id: str = None,
        doc_kind: str = None
    ) -> List[Dict]:
        """
        Retrieve chunks across all documents for the given tenant/case.
        """
        embedding = self.llm.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # L2-normalize query vector to match normalized index vectors (#6).
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm

        candidates: List[Dict] = []
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo", {})
            if not self._match_filters(metainfo, tenant_id=tenant_id, case_id=case_id, doc_kind=doc_kind):
                continue
            vector_db = report.get("vector_db")
            if vector_db is None:
                continue
            chunks = document.get("content", {}).get("chunks", [])
            pages = document.get("content", {}).get("pages", [])
            if not chunks:
                continue
            actual_top_n = min(top_n, len(chunks))
            distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
            for distance, index in zip(distances[0], indices[0]):
                distance = round(float(distance), 4)
                chunk = chunks[index]
                page_num = chunk.get("page", chunk.get("page_from", 0))
                parent_page = next((page for page in pages if page.get("page") == page_num), None)
                text = parent_page["text"] if (return_parent_pages and parent_page) else chunk.get("text", "")
                candidates.append({
                    "distance": distance,
                    "page": page_num,
                    "text": text,
                    "type": chunk.get("type", "content"),
                    "doc_id": metainfo.get("doc_id", report.get("name")),
                    "doc_title": metainfo.get("title", metainfo.get("company_name"))
                })
        candidates.sort(key=lambda r: r["distance"], reverse=True)
        return candidates[:top_n]

    def retrieve_all(self, company_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


def _merge_dense_sparse(
    dense: List[Dict],
    sparse: List[Dict],
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict]:
    """
    Merge dense (FAISS cosine) and sparse (BM25) results into a single ranked list.

    Scores are normalized within each list to [0,1] then combined with weights.
    Duplicates (same doc_id + page) are merged, keeping the higher combined score.
    """
    def _normalize_scores(items: List[Dict], key: str = "distance") -> List[Dict]:
        vals = [x[key] for x in items]
        mn, mx = min(vals, default=0.0), max(vals, default=1.0)
        span = mx - mn if mx != mn else 1.0
        for x in items:
            x["_norm_score"] = (x[key] - mn) / span
        return items

    dense = _normalize_scores([d.copy() for d in dense])
    sparse = _normalize_scores([s.copy() for s in sparse])

    merged: Dict[str, Dict] = {}
    for item in dense:
        key = f"{item.get('doc_id','?')}::{item.get('page', 0)}"
        item["_combined"] = dense_weight * item["_norm_score"]
        item.setdefault("_retrieval_source", "dense")
        merged[key] = item
    for item in sparse:
        key = f"{item.get('doc_id','?')}::{item.get('page', 0)}"
        contrib = sparse_weight * item["_norm_score"]
        if key in merged:
            merged[key]["_combined"] += contrib
            merged[key]["_retrieval_source"] = "dense+sparse"
        else:
            item["_combined"] = contrib
            item.setdefault("_retrieval_source", "sparse")
            merged[key] = item

    result = sorted(merged.values(), key=lambda x: x["_combined"], reverse=True)
    # Copy combined score into "distance" field so downstream code works unchanged
    for item in result:
        item["distance"] = round(item["_combined"], 4)
    return result


class HybridRetriever:
    """
    Hybrid dense+sparse retriever with LLM reranking and authority-tiering.

    Retrieval parameter contract (Sprint 3):
      dense_k          — chunks fetched from FAISS per doc_kind filter (default 50)
      sparse_k         — chunks fetched from BM25 (default 50)
      rerank_sample_k  — total candidates sent to LLM reranker (default 80)
      final_candidates_k — returned after reranking (default 40); consumed by report generator

    The `top_n` param is kept as an alias for final_candidates_k for backward compat.
    """

    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.bm25_retriever = BM25Retriever(
            bm25_db_dir=vector_db_dir,   # not used for case retrieval; dir kept for compat
            documents_dir=documents_dir,
        )
        self.reranker = LLMReranker()

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False,
        tenant_id: str = None,
        case_id: str = None,
        doc_kind: str = None
    ) -> List[Dict]:
        """Backward-compatible single-company retrieval (dense only + rerank)."""
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages,
            tenant_id=tenant_id,
            case_id=case_id,
            doc_kind=doc_kind,
        )
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight,
        )
        return reranked_results[:top_n]

    def retrieve_by_case(
        self,
        query: str,
        # ── Sprint-3 explicit param contract ─────────────────────────────────
        dense_k: int = 50,
        sparse_k: int = 50,
        rerank_sample_k: int = 80,
        final_candidates_k: int = 40,
        # ── Kept for backward compat (maps to final_candidates_k) ─────────────
        top_n: Optional[int] = None,
        llm_reranking_sample_size: Optional[int] = None,   # legacy alias → rerank_sample_k
        # ── Shared options ─────────────────────────────────────────────────────
        documents_batch_size: int = 2,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False,
        tenant_id: Optional[str] = None,
        case_id: Optional[str] = None,
        doc_kind: Optional[Union[str, List[str]]] = None,
        authority_scope: Optional[str] = None,
    ) -> List[Dict]:
        """
        Hybrid dense+sparse retrieval with LLM reranking and authority-tiering.

        Steps:
          1. Authority-tiering: if authority_scope given and doc_kind is None,
             first search tier-1 doc_kinds; supplement with tier-2 if insufficient.
          2. Dense retrieval (FAISS cosine, top dense_k).
          3. Sparse retrieval (BM25, top sparse_k).
          4. Merge + normalize scores (dense_weight=0.6, sparse_weight=0.4).
          5. LLM reranking on top rerank_sample_k merged candidates.
          6. Return top final_candidates_k.
        """
        # Backward compat aliases
        if top_n is not None:
            final_candidates_k = top_n
        if llm_reranking_sample_size is not None:
            rerank_sample_k = llm_reranking_sample_size

        effective_doc_kind = doc_kind

        # ── 1. Authority-tiering ────────────────────────────────────────────────
        tier_info = AUTHORITY_TIERS.get(authority_scope or "") if not doc_kind else None
        tier_1_kinds = tier_info["tier_1"] if tier_info else None
        tier_2_kinds = tier_info["tier_2"] if tier_info else None

        if tier_1_kinds:
            effective_doc_kind = tier_1_kinds

        # ── 2. Dense retrieval ──────────────────────────────────────────────────
        dense_results = self.vector_retriever.retrieve_by_case(
            query=query,
            top_n=dense_k,
            return_parent_pages=return_parent_pages,
            tenant_id=tenant_id,
            case_id=case_id,
            doc_kind=effective_doc_kind,
        )

        # ── 3. Sparse (BM25) retrieval ──────────────────────────────────────────
        sparse_results = self.bm25_retriever.retrieve_by_case(
            query=query,
            top_n=sparse_k,
            tenant_id=tenant_id,
            case_id=case_id,
            doc_kind=effective_doc_kind,
        )

        # Authority supplement: if tier-1 results < threshold, add tier-2
        if tier_2_kinds and (len(dense_results) + len(sparse_results)) < _TIER1_SUFFICIENT_THRESHOLD * 2:
            _log.info(
                "authority-tiering: tier-1 (%s) returned %d dense + %d sparse — supplementing with tier-2 (%s)",
                tier_1_kinds, len(dense_results), len(sparse_results), tier_2_kinds,
            )
            dense_results += self.vector_retriever.retrieve_by_case(
                query=query,
                top_n=dense_k // 2,
                return_parent_pages=return_parent_pages,
                tenant_id=tenant_id,
                case_id=case_id,
                doc_kind=tier_2_kinds,
            )
            sparse_results += self.bm25_retriever.retrieve_by_case(
                query=query,
                top_n=sparse_k // 2,
                tenant_id=tenant_id,
                case_id=case_id,
                doc_kind=tier_2_kinds,
            )

        _log.info(
            "HybridRetriever.retrieve_by_case: dense=%d sparse=%d (case=%s doc_kind=%s scope=%s)",
            len(dense_results), len(sparse_results),
            case_id, effective_doc_kind, authority_scope,
        )

        # ── 4. Merge dense + sparse ─────────────────────────────────────────────
        merged = _merge_dense_sparse(dense_results, sparse_results)

        # Limit to rerank_sample_k before sending to LLM reranker (cost guardrail)
        candidates_for_rerank = merged[:rerank_sample_k]

        if not candidates_for_rerank:
            _log.warning("HybridRetriever: no candidates to rerank (case=%s)", case_id)
            return []

        # ── 5. LLM reranking ────────────────────────────────────────────────────
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=candidates_for_rerank,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight,
        )

        # ── 6. Return top final_candidates_k ────────────────────────────────────
        return reranked_results[:final_candidates_k]
