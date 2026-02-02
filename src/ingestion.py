import os
import json
import pickle
import time
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

            return results  # type: ignore

        response = self.llm.embeddings.create(input=text, model=model)
        return [embedding.embedding for embedding in response.data]

    def _embed_batch_with_retry(self, batch: List[str], model: str) -> List[List[float]]:
        for attempt in range(1, settings.embeddings_retry_max + 1):
            try:
                response = self.llm.embeddings.create(input=batch, model=model)
                return [embedding.embedding for embedding in response.data]
            except Exception as e:
                if attempt >= settings.embeddings_retry_max:
                    raise
                time.sleep(settings.embeddings_backoff_seconds * attempt)

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")