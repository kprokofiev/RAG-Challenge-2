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


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
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
            batch_size = max(1, settings.embeddings_batch_size)
            batches = [(i, sanitized[i:i + batch_size]) for i in range(0, len(sanitized), batch_size)]
            results: List[List[float]] = [None] * len(sanitized)  # type: ignore

            with ThreadPoolExecutor(max_workers=max(1, settings.embeddings_max_concurrency)) as executor:
                future_map = {
                    executor.submit(self._embed_batch_with_retry, batch, model): start_idx
                    for start_idx, batch in batches
                }
                for future in as_completed(future_map):
                    start_idx = future_map[future]
                    batch_embeddings = future.result()
                    for offset, emb in enumerate(batch_embeddings):
                        results[start_idx + offset] = emb

            timings = list(self._batch_timings)
            attempts = list(self._batch_attempts)
            avg_latency_ms = (sum(timings) / len(timings)) * 1000 if timings else 0
            p95_latency_ms = float(np.percentile(timings, 95) * 1000) if timings else 0
            max_latency_ms = max(timings) * 1000 if timings else 0
            total_time_ms = (time.perf_counter() - start_total) * 1000
            dimensions = len(results[0]) if results and results[0] is not None else 0
            self.last_metrics.update({
                "embeddings_model": model,
                "embeddings_batch_size": batch_size,
                "embeddings_max_concurrency": settings.embeddings_max_concurrency,
                "embeddings_requests": len(batches),
                "embeddings_attempts": sum(attempts) if attempts else len(batches),
                "embeddings_avg_latency_ms": avg_latency_ms,
                "embeddings_p95_latency_ms": p95_latency_ms,
                "embeddings_max_latency_ms": max_latency_ms,
                "embeddings_total_time_ms": total_time_ms,
                "embedding_dimensions": dimensions
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
                if attempt >= settings.embeddings_retry_max:
                    elapsed = time.perf_counter() - start
                    with self._metrics_lock:
                        self._batch_timings.append(elapsed)
                        self._batch_attempts.append(attempt)
                    raise
                time.sleep(settings.embeddings_backoff_seconds * attempt)

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

        build_start = time.perf_counter()
        merged_index = None
        for faiss_file in faiss_files:
            try:
                idx = faiss.read_index(str(faiss_file))
            except Exception:
                continue  # skip corrupt / incompatible shards
            if merged_index is None:
                dim = idx.d
                merged_index = faiss.IndexFlatIP(dim)
                metrics["dimension"] = dim
            if idx.d != merged_index.d:
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

        if merged_index is None or merged_index.ntotal == 0:
            return metrics

        output_path = vector_dir / "case_index.faiss"
        write_start = time.perf_counter()
        faiss.write_index(merged_index, str(output_path))
        metrics["write_ms"] = (time.perf_counter() - write_start) * 1000
        metrics["output_path"] = str(output_path)
        return metrics
