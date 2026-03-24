import os
import json
import pickle
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt

from src.settings import settings
from src.embedding_utils import (
    write_manifest, read_manifest, validate_manifest_compat, current_manifest,
    get_embedding_cache, EmbeddingCache,
)
from src.tokenizer import tokenize as _pharma_tokenize

import logging as _logging
_ingest_log = _logging.getLogger(__name__)


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [_pharma_tokenize(chunk) for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.
        
        Args:
            all_reports_dir (Path): Directory containing the JSON report files
            output_dir (Path): Directory where to save the BM25 indices
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self._metrics_lock = threading.Lock()
        self._reset_embedding_metrics()
        self.last_report_metrics: List[dict] = []
        self._emb_cache: EmbeddingCache | None = get_embedding_cache()

    def _reset_embedding_metrics(self):
        self._batch_timings: List[float] = []
        self._batch_attempts: List[int] = []
        self.last_metrics: dict = {}

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = None) -> List[float]:
        model = model or settings.embeddings_model
        start_total = time.perf_counter()
        self._reset_embedding_metrics()
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        if isinstance(text, list):
            if not text:
                return []
            sanitized = [t if (isinstance(t, str) and t.strip()) else " " for t in text]

            # ── Embedding cache: resolve hits, embed only misses ──────────
            results: List[List[float]] = [None] * len(sanitized)  # type: ignore
            miss_indices: List[int]

            if self._emb_cache:
                cached, miss_indices = self._emb_cache.get_batch(sanitized, model)
                for i, vec in enumerate(cached):
                    if vec is not None:
                        results[i] = vec
            else:
                miss_indices = list(range(len(sanitized)))

            if miss_indices:
                miss_texts = [sanitized[i] for i in miss_indices]
                batch_size = max(1, settings.embeddings_batch_size)
                batches = [(j, miss_texts[j:j + batch_size]) for j in range(0, len(miss_texts), batch_size)]
                miss_results: List[List[float]] = [None] * len(miss_texts)  # type: ignore

                with ThreadPoolExecutor(max_workers=max(1, settings.embeddings_max_concurrency)) as executor:
                    future_map = {
                        executor.submit(self._embed_batch_with_retry, batch, model): start_idx
                        for start_idx, batch in batches
                    }
                    for future in as_completed(future_map):
                        start_idx = future_map[future]
                        batch_embeddings = future.result()
                        for offset, emb in enumerate(batch_embeddings):
                            miss_results[start_idx + offset] = emb

                # Write fresh embeddings to cache + fill results
                cache_texts = []
                cache_vecs = []
                for mi, orig_idx in enumerate(miss_indices):
                    results[orig_idx] = miss_results[mi]
                    cache_texts.append(sanitized[orig_idx])
                    cache_vecs.append(miss_results[mi])
                if self._emb_cache and cache_vecs:
                    self._emb_cache.put_batch(cache_texts, model, cache_vecs)

            timings = list(self._batch_timings)
            attempts = list(self._batch_attempts)
            avg_latency_ms = (sum(timings) / len(timings)) * 1000 if timings else 0
            p95_latency_ms = float(np.percentile(timings, 95) * 1000) if timings else 0
            max_latency_ms = max(timings) * 1000 if timings else 0
            total_time_ms = (time.perf_counter() - start_total) * 1000
            dimensions = len(results[0]) if results and results[0] is not None else 0
            n_cached = len(sanitized) - len(miss_indices)
            batch_size_used = max(1, settings.embeddings_batch_size)
            n_api_batches = len(timings)  # actual API batches sent
            self.last_metrics.update({
                "embeddings_model": model,
                "embeddings_batch_size": batch_size_used,
                "embeddings_max_concurrency": settings.embeddings_max_concurrency,
                "embeddings_requests": n_api_batches,
                "embeddings_attempts": sum(attempts) if attempts else n_api_batches,
                "embeddings_avg_latency_ms": avg_latency_ms,
                "embeddings_p95_latency_ms": p95_latency_ms,
                "embeddings_max_latency_ms": max_latency_ms,
                "embeddings_total_time_ms": total_time_ms,
                "embedding_dimensions": dimensions,
                "embedding_cache_hits": n_cached,
                "embedding_cache_misses": len(miss_indices),
            })
            return results  # type: ignore

        start_single = time.perf_counter()
        response = self.llm.embeddings.create(input=text, model=model)
        latency_ms = (time.perf_counter() - start_single) * 1000
        dimension = len(response.data[0].embedding) if response.data else 0
        self.last_metrics.update({
            "embeddings_model": model,
            "embeddings_batch_size": 1,
            "embeddings_max_concurrency": 1,
            "embeddings_requests": 1,
            "embeddings_attempts": 1,
            "embeddings_avg_latency_ms": latency_ms,
            "embeddings_p95_latency_ms": latency_ms,
            "embeddings_max_latency_ms": latency_ms,
            "embeddings_total_time_ms": (time.perf_counter() - start_total) * 1000,
            "embedding_dimensions": dimension
        })
        return [embedding.embedding for embedding in response.data]

    def _embed_batch_with_retry(self, batch: List[str], model: str) -> List[List[float]]:
        import logging as _logging
        _log = _logging.getLogger(__name__)
        start = time.perf_counter()
        for attempt in range(1, settings.embeddings_retry_max + 1):
            try:
                response = self.llm.embeddings.create(input=batch, model=model)
                elapsed = time.perf_counter() - start
                with self._metrics_lock:
                    self._batch_timings.append(elapsed)
                    self._batch_attempts.append(attempt)
                return [embedding.embedding for embedding in response.data]
            except Exception as e:
                delay = settings.embeddings_backoff_seconds * attempt
                if attempt >= settings.embeddings_retry_max:
                    elapsed = time.perf_counter() - start
                    with self._metrics_lock:
                        self._batch_timings.append(elapsed)
                        self._batch_attempts.append(attempt)
                    _log.error(
                        "embedding_failed_terminal attempt=%d/%d last_error=%s",
                        attempt, settings.embeddings_retry_max, e,
                    )
                    raise
                next_retry_at = int(time.time()) + delay
                _log.warning(
                    "embedding_retry attempt=%d/%d delay=%ds next_retry_at=%d last_error=%s",
                    attempt, settings.embeddings_retry_max, delay, next_retry_at, e,
                )
                time.sleep(delay)

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        # L2-normalize so that IndexFlatIP computes true cosine similarity.
        # Without normalization, inner-product scores are magnitude-dependent and
        # do not rank by angular similarity — a silent correctness bug (#6).
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero for zero vectors
        embeddings_array = embeddings_array / norms
        dimension = len(embeddings[0])
        start = time.perf_counter()
        index = faiss.IndexFlatIP(dimension)  # IndexFlatIP + L2-normalized = cosine similarity
        index.add(embeddings_array)
        build_ms = (time.perf_counter() - start) * 1000
        return index, build_ms
    
    def _process_report(self, report: dict):
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index, build_ms = self._create_vector_db(embeddings)
        self.last_metrics["faiss_build_ms"] = build_ms
        self.last_metrics["vectors_count"] = len(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_report_metrics = []
        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            self._reset_embedding_metrics()
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            write_start = time.perf_counter()
            faiss.write_index(index, str(faiss_file_path))
            write_ms = (time.perf_counter() - write_start) * 1000
            metrics = dict(self.last_metrics)
            metrics["faiss_write_ms"] = write_ms
            metrics["doc_name"] = report_path.name
            self.last_report_metrics.append(metrics)

        # Write embedding manifest for index versioning
        write_manifest(output_dir)
        print(f"Processed {len(all_report_paths)} reports")

    def process_single_report(self, report_path: Path, output_dir: Path) -> dict:
        """
        Process a single chunked report JSON and write its FAISS index to output_dir.

        Useful for on-the-fly ingestion of extra documents (e.g., PubMed abstracts) without
        re-embedding the full case corpus.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        self._reset_embedding_metrics()
        index = self._process_report(report_data)
        sha1_name = report_data.get("metainfo", {}).get("sha1_name") or report_path.stem
        faiss_file_path = output_dir / f"{sha1_name}.faiss"
        write_start = time.perf_counter()
        faiss.write_index(index, str(faiss_file_path))
        write_ms = (time.perf_counter() - write_start) * 1000
        metrics = dict(self.last_metrics)
        metrics["faiss_write_ms"] = write_ms
        metrics["doc_name"] = report_path.name
        self.last_report_metrics = [metrics]
        # Write/update embedding manifest for index versioning
        write_manifest(output_dir)
        return metrics

    @staticmethod
    def build_case_index(vector_dir: Path) -> dict:
        """
        Merge all per-document FAISS indices in *vector_dir* into a single
        case-level index stored as ``case_index.faiss`` in the same directory (#7).

        The merged index enables fast, single-pass retrieval across the entire
        case corpus instead of per-document scans.  All individual indices must
        have the same dimension (they always do because the embedding model is
        fixed).

        Returns a metrics dict with keys:
            doc_count         – number of doc-level indices merged
            total_vectors     – total number of vectors in the merged index
            dimension         – embedding dimension
            build_ms          – time to merge (ms)
            write_ms          – time to write to disk (ms)
            output_path       – absolute path to the written file (str)
        """
        metrics: dict = {
            "doc_count": 0,
            "total_vectors": 0,
            "dimension": 0,
            "build_ms": 0.0,
            "write_ms": 0.0,
            "output_path": "",
        }
        faiss_files = [f for f in vector_dir.glob("*.faiss") if f.name != "case_index.faiss"]
        if not faiss_files:
            return metrics

        # ── Index version validation ──────────────────────────────────────
        # Check manifest to detect stale indices from a different model/dim.
        stored_manifest = read_manifest(vector_dir)
        cur = current_manifest()
        expected_dim = cur["embedding_dim"]
        skipped_dim_mismatch = 0

        if stored_manifest and not validate_manifest_compat(stored_manifest, label="build_case_index"):
            _ingest_log.warning(
                "case_index_manifest_mismatch dir=%s stored=%s current=%s — "
                "stale indices from old model will be skipped during merge",
                vector_dir, stored_manifest, cur,
            )

        build_start = time.perf_counter()
        merged_index = None
        for faiss_file in faiss_files:
            try:
                idx = faiss.read_index(str(faiss_file))
            except Exception:
                continue  # skip corrupt / incompatible shards
            # Warn about dimension mismatch (model migration tracking)
            # but do NOT skip — let first shard set the target dimension.
            if expected_dim and idx.d != expected_dim:
                _ingest_log.info(
                    "case_index_dim_note file=%s dim=%d settings_expected=%d "
                    "(will merge by first-shard dimension, not settings)",
                    faiss_file.name, idx.d, expected_dim,
                )
            if merged_index is None:
                dim = idx.d
                merged_index = faiss.IndexFlatIP(dim)
                metrics["dimension"] = dim
            if idx.d != merged_index.d:
                skipped_dim_mismatch += 1
                continue  # skip shards with mismatched dimension
            # Extract all vectors and add to merged index
            n = idx.ntotal
            if n == 0:
                continue
            vecs = np.zeros((n, idx.d), dtype=np.float32)
            idx.reconstruct_n(0, n, vecs)
            merged_index.add(vecs)
            metrics["doc_count"] += 1
            metrics["total_vectors"] += n

        metrics["build_ms"] = (time.perf_counter() - build_start) * 1000
        metrics["skipped_dim_mismatch"] = skipped_dim_mismatch

        if skipped_dim_mismatch:
            _ingest_log.warning(
                "case_index_dim_mismatch_summary skipped=%d/%d shards "
                "(likely from old embedding model, need re-indexing)",
                skipped_dim_mismatch, len(faiss_files),
            )

        if merged_index is None or merged_index.ntotal == 0:
            return metrics

        output_path = vector_dir / "case_index.faiss"
        write_start = time.perf_counter()
        faiss.write_index(merged_index, str(output_path))
        metrics["write_ms"] = (time.perf_counter() - write_start) * 1000
        metrics["output_path"] = str(output_path)
        # Write manifest alongside case index
        write_manifest(vector_dir)
        metrics["embedding_model"] = cur["embedding_model"]
        metrics["embedding_dim"] = cur["embedding_dim"]
        return metrics
