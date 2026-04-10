import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor
from src.settings import settings
from src.tokenizer import tokenize as _pharma_tokenize
from src.openai_model_router import (
    choose_routed_model,
    extract_usage_metrics,
    is_quota_exhausted_error,
    mark_tier_exhausted,
    next_tier_index,
    record_usage,
)

_log = logging.getLogger(__name__)


class JinaReranker:
    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()

class LLMReranker:
    # Default model used when DDKIT_RERANK_MODEL env is not set (Sprint 3 §6).
    _DEFAULT_RERANK_MODEL = "gpt-5.4"

    def __init__(self):
        self.llm = self.set_up_llm()
        # Model configurable via DDKIT_RERANK_MODEL env (Sprint 3 §6)
        self.rerank_model = os.getenv("DDKIT_RERANK_MODEL", self._DEFAULT_RERANK_MODEL)
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks

    def set_up_llm(self):
        load_dotenv()
        timeout_raw = os.getenv("DDKIT_LLM_TIMEOUT_SECONDS", "120")
        try:
            timeout_seconds = float(timeout_raw)
        except (TypeError, ValueError):
            timeout_seconds = 120.0
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=timeout_seconds, max_retries=2)
        return llm

    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        minimum_tier_index = 0
        while True:
            routed = choose_routed_model(self.rerank_model, minimum_tier_index=minimum_tier_index)
            try:
                completion = self.llm.beta.chat.completions.parse(
                    model=routed.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self.system_prompt_rerank_single_block},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=self.schema_for_single_block
                )

                usage = extract_usage_metrics(completion)
                record_usage(routed, usage)
                response = completion.choices[0].message.parsed
                return response.model_dump()
            except Exception as exc:
                if is_quota_exhausted_error(exc) and routed.tier in {"elite", "mini"}:
                    mark_tier_exhausted(routed, str(exc))
                    minimum_tier_index = next_tier_index(routed.tier)
                    continue
                raise

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        minimum_tier_index = 0
        while True:
            routed = choose_routed_model(self.rerank_model, minimum_tier_index=minimum_tier_index)
            try:
                completion = self.llm.beta.chat.completions.parse(
                    model=routed.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=self.schema_for_multiple_blocks
                )

                usage = extract_usage_metrics(completion)
                record_usage(routed, usage)
                response = completion.choices[0].message.parsed
                return response.model_dump()
            except Exception as exc:
                if is_quota_exhausted_error(exc) and routed.tier in {"elite", "mini"}:
                    mark_tier_exhausted(routed, str(exc))
                    minimum_tier_index = next_tier_index(routed.tier)
                    continue
                raise

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 8, llm_weight: float = 0.7):
        """
        Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.
        """
        # Create batches of documents
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                # Get ranking for single document
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # Calculate combined score - note that distance is inverted since lower is better
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # Process all documents in parallel using single-block method
            with ThreadPoolExecutor(max_workers=settings.ddkit_rerank_max_concurrency) as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                    
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results

            # Process batches in parallel using threads
            with ThreadPoolExecutor(max_workers=settings.ddkit_rerank_max_concurrency) as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results


class CrossEncoderReranker:
    """S7-B2: Cross-encoder reranker using sentence-transformers.

    Feature-flagged via DDKIT_RERANKER=cross_encoder (default: cross_encoder).
    Falls back to a cheap deterministic reranker if sentence-transformers is
    not installed or the model cannot be loaded.

    10-20x faster than LLM reranking, deterministic, zero marginal API cost.
    """

    _DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self):
        model_name = os.getenv("DDKIT_CROSS_ENCODER_MODEL", self._DEFAULT_MODEL)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
            _log.info("cross_encoder_loaded model=%s", model_name)
        except Exception as exc:
            _log.warning(
                "cross_encoder_unavailable model=%s reason=%s",
                model_name,
                exc,
            )
            self._model = None

    def rerank_documents(
        self,
        query: str,
        documents: list,
        documents_batch_size: int = 4,  # unused, kept for interface compat
        llm_weight: float = 0.7,
    ):
        """Rerank documents using cross-encoder scores + vector distance."""
        if self._model is None:
            _log.warning("cross_encoder_fallback_to_heuristic — model not loaded")
            return HeuristicReranker().rerank_documents(query, documents, documents_batch_size, llm_weight)

        vector_weight = 1 - llm_weight
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self._model.predict(pairs)

        # Normalize scores to [0, 1] range (cross-encoder outputs logits)
        import numpy as np
        scores_arr = np.array(scores, dtype=np.float64)
        min_s, max_s = scores_arr.min(), scores_arr.max()
        if max_s > min_s:
            norm_scores = (scores_arr - min_s) / (max_s - min_s)
        else:
            norm_scores = np.ones_like(scores_arr) * 0.5

        all_results = []
        for doc, ce_score in zip(documents, norm_scores):
            doc_with_score = doc.copy()
            doc_with_score["relevance_score"] = round(float(ce_score), 4)
            doc_with_score["combined_score"] = round(
                llm_weight * float(ce_score) + vector_weight * doc.get("distance", 0),
                4,
            )
            all_results.append(doc_with_score)

        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results


class HeuristicReranker:
    """Cheap lexical reranker used when API-heavy rerankers are undesirable.

    This is intentionally simple and deterministic. It keeps retrieval on a
    low-cost path when cross-encoder assets are unavailable and avoids falling
    back to token-expensive LLM reranking on the hot path.
    """

    _RESULT_MARKERS = (
        "primary endpoint",
        "secondary endpoint",
        "overall survival",
        "progression-free survival",
        "objective response",
        "confidence interval",
        "p-value",
        "response rate",
        "results",
        "adverse event",
    )

    def rerank_documents(
        self,
        query: str,
        documents: list,
        documents_batch_size: int = 8,  # kept for interface compatibility
        llm_weight: float = 0.7,
    ):
        vector_weight = 1 - llm_weight
        query_tokens = [tok for tok in _pharma_tokenize(query) if len(tok) >= 3]
        query_token_set = set(query_tokens)
        nct_tokens = {tok for tok in query_token_set if tok.startswith("nct")}

        all_results = []
        for doc in documents:
            text = doc.get("text", "") or ""
            text_lower = text.lower()
            doc_tokens = set(_pharma_tokenize(text[:5000]))
            overlap = len(query_token_set & doc_tokens)
            overlap_score = overlap / max(1, min(len(query_token_set), 12))
            nct_boost = 0.25 if any(tok in doc_tokens for tok in nct_tokens) else 0.0
            numeric_boost = 0.1 if any(ch.isdigit() for ch in text[:600]) else 0.0
            result_boost = 0.15 if any(marker in text_lower for marker in self._RESULT_MARKERS) else 0.0
            heuristic_score = min(1.0, overlap_score + nct_boost + numeric_boost + result_boost)

            doc_with_score = doc.copy()
            doc_with_score["relevance_score"] = round(float(heuristic_score), 4)
            doc_with_score["combined_score"] = round(
                llm_weight * float(heuristic_score) + vector_weight * doc.get("distance", 0),
                4,
            )
            all_results.append(doc_with_score)

        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results


def get_reranker():
    """Factory: return reranker based on DDKIT_RERANKER env var.

    Values: "cross_encoder" | "heuristic" | "llm" (default: "cross_encoder").
    """
    reranker_type = os.getenv("DDKIT_RERANKER", "cross_encoder").lower().strip()
    if reranker_type == "llm":
        return LLMReranker()
    if reranker_type == "heuristic":
        return HeuristicReranker()
    if reranker_type == "cross_encoder":
        ce = CrossEncoderReranker()
        if ce._model is not None:
            return ce
        _log.warning("cross_encoder requested but unavailable, falling back to heuristic")
        return HeuristicReranker()
    _log.warning("unknown_reranker_type=%s — falling back to heuristic", reranker_type)
    return HeuristicReranker()
