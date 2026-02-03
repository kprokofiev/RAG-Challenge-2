import json
import logging
import tempfile
import time
import tarfile
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import requests
from PyPDF2 import PdfReader

from src.pipeline import Pipeline
from src.dd_report_generator import DDReportGenerator
from src.storage_client import StorageClient
from src.settings import settings
from src.ddkit_db import DDKitDB
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor

logger = logging.getLogger(__name__)


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _doc_parse_worker(payload: Dict[str, Any], result_queue: Queue) -> None:
    """Run doc parsing pipeline in a subprocess with isolation."""
    try:
        processor = DocParseIndexProcessor()
        temp_path = Path(payload["temp_path"])
        pdf_path = Path(payload["pdf_path"])
        pipeline = Pipeline(temp_path)
        parser_mode = payload.get("parser_mode", "docling")
        docling_do_ocr = payload.get("docling_do_ocr")
        docling_do_tables = payload.get("docling_do_tables")
        if parser_mode == "fast_text":
            success, parsed_key, metrics = processor._process_single_document_fast(
                pipeline=pipeline,
                pdf_path=pdf_path,
                temp_path=temp_path,
                tenant_id=payload["tenant_id"],
                case_id=payload["case_id"],
                doc_id=payload["doc_id"],
                doc_kind=payload["doc_kind"],
                title=payload.get("title", ""),
                source_url=payload.get("source_url", ""),
                s3_parsed_json_key=payload.get("s3_parsed_json_key"),
            )
        else:
            success, parsed_key, metrics = processor._process_single_document(
                pipeline=pipeline,
                pdf_path=pdf_path,
                temp_path=temp_path,
                tenant_id=payload["tenant_id"],
                case_id=payload["case_id"],
                doc_id=payload["doc_id"],
                doc_kind=payload["doc_kind"],
                title=payload.get("title", ""),
                source_url=payload.get("source_url", ""),
                s3_parsed_json_key=payload.get("s3_parsed_json_key"),
                docling_do_ocr=docling_do_ocr,
                docling_do_tables=docling_do_tables,
            )
        result_queue.put({"success": success, "parsed_key": parsed_key, "metrics": metrics})
    except Exception as exc:
        result_queue.put({"success": False, "error": str(exc)})


class DocParseIndexProcessor:
    """Processor for doc_parse_index jobs."""

    def __init__(self):
        self.storage_client = StorageClient()
        self.ddkit_db = DDKitDB()

    def _detect_text_layer(self, pdf_path: Path, max_pages: int = 3, min_chars: int = 200) -> Dict[str, Any]:
        """Detect whether PDF has a usable text layer."""
        result = {
            "has_text_layer": False,
            "pages_total": 0,
            "pages_checked": 0,
            "text_chars": 0
        }
        try:
            reader = PdfReader(str(pdf_path))
            pages_total = len(reader.pages)
            pages_checked = min(max_pages, pages_total)
            text_chars = 0
            for idx in range(pages_checked):
                page_text = reader.pages[idx].extract_text() or ""
                text_chars += len(page_text.strip())
            result.update({
                "has_text_layer": text_chars >= min_chars,
                "pages_total": pages_total,
                "pages_checked": pages_checked,
                "text_chars": text_chars
            })
        except Exception:
            return result
        return result

    def process_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Process a doc_parse_index job.

        Args:
            job_data: Job data containing tenant_id, case_id, doc_id, doc_kind, s3_rendered_pdf_key

        Returns:
            bool: True if processing successful
        """
        tenant_id = job_data["tenant_id"]
        case_id = job_data["case_id"]
        doc_id = job_data["doc_id"]
        job_id = job_data.get("job_id")
        doc_kind = job_data["doc_kind"]
        title = job_data.get("title", "")
        source_url = job_data.get("source_url", "")
        s3_parsed_json_key = job_data.get("s3_parsed_json_key")
        s3_pdf_key = job_data["s3_rendered_pdf_key"]

        logger.info(f"Processing doc_parse_index job: tenant={tenant_id}, case={case_id}, doc={doc_id}")
        metrics: Dict[str, Any] = {
            "job_type": "doc_parse_index",
            "job_id": job_id,
            "tenant_id": tenant_id,
            "case_id": case_id,
            "doc_id": doc_id,
            "doc_kind": doc_kind,
            "source_url": source_url,
            "stages": {}
        }

        try:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_running(job_id)
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Download PDF from S3
                local_pdf_path = temp_path / f"{doc_id}.pdf"
                download_start = time.perf_counter()
                if not self.storage_client.download_to_path(s3_pdf_key, local_pdf_path):
                    logger.error(f"Failed to download PDF for doc {doc_id}")
                    metrics["stages"]["download_pdf_ms"] = _ms_since(download_start)
                    job_data["metrics"] = metrics
                    return False
                metrics["stages"]["download_pdf_ms"] = _ms_since(download_start)
                try:
                    metrics["pdf_bytes"] = local_pdf_path.stat().st_size
                except Exception:
                    metrics["pdf_bytes"] = None

                validate_start = time.perf_counter()
                invalid_reason = self._validate_pdf_file(local_pdf_path)
                metrics["stages"]["validate_pdf_ms"] = _ms_since(validate_start)
                if invalid_reason:
                    logger.error(f"Invalid PDF for doc {doc_id}: {invalid_reason}")
                    if job_id and self.ddkit_db.is_configured():
                        self.ddkit_db.mark_job_failed(job_id, invalid_reason)
                    job_data["status"] = "failed"
                    job_data["error_message"] = invalid_reason
                    metrics["error"] = invalid_reason
                    job_data["metrics"] = metrics
                    logger.info("doc_parse_index_metrics=%s", json.dumps(metrics, ensure_ascii=False))
                    return False

                text_layer = self._detect_text_layer(local_pdf_path)
                metrics.update({
                    "pages": text_layer.get("pages_total"),
                    "text_layer_chars": text_layer.get("text_chars"),
                    "text_layer_pages_checked": text_layer.get("pages_checked"),
                })
                parser_mode = "fast_text" if text_layer.get("has_text_layer") else "docling"
                docling_do_ocr = not text_layer.get("has_text_layer")
                docling_do_tables = bool(settings.docling_do_tables) if parser_mode == "docling" else False
                metrics["parser_path"] = parser_mode
                metrics["parser_used"] = parser_mode
                metrics["docling_do_ocr"] = docling_do_ocr
                metrics["docling_do_tables"] = docling_do_tables

                # Process single PDF through the pipeline with timeout isolation
                success, parsed_key, error_message, stage_metrics = self._process_single_document_with_timeout(
                    temp_path=temp_path,
                    pdf_path=local_pdf_path,
                    tenant_id=tenant_id,
                    case_id=case_id,
                    doc_id=doc_id,
                    doc_kind=doc_kind,
                    title=title,
                    source_url=source_url,
                    s3_parsed_json_key=s3_parsed_json_key,
                    parser_mode=parser_mode,
                    docling_do_ocr=docling_do_ocr,
                    docling_do_tables=docling_do_tables
                )
                if stage_metrics:
                    metrics.update(stage_metrics)
                    chunk_metrics = stage_metrics.get("chunks") if isinstance(stage_metrics, dict) else None
                    if isinstance(chunk_metrics, dict):
                        metrics["chunks_count"] = chunk_metrics.get("chunks_after_dedup")
                        metrics["avg_tokens_per_chunk"] = chunk_metrics.get("tokens_avg")
                        metrics["dedup_ratio"] = chunk_metrics.get("dedup_ratio")
                        metrics["unique_chunks_after_dedup"] = chunk_metrics.get("chunks_after_dedup")
                    embed_metrics = stage_metrics.get("embeddings") if isinstance(stage_metrics, dict) else None
                    if isinstance(embed_metrics, dict):
                        metrics["embeddings_requests"] = embed_metrics.get("embeddings_requests")
                        metrics["embeddings_batches"] = embed_metrics.get("embeddings_requests")
                        metrics["embeddings_time_ms"] = embed_metrics.get("embeddings_total_time_ms")
                        metrics["embedding_dimensions"] = embed_metrics.get("embedding_dimensions")
                        metrics["embeddings_model"] = embed_metrics.get("embeddings_model")
                        metrics["embeddings_batch_size"] = embed_metrics.get("embeddings_batch_size")
                        metrics["embeddings_max_concurrency"] = embed_metrics.get("embeddings_max_concurrency")
                        metrics["embeddings_avg_latency_ms"] = embed_metrics.get("embeddings_avg_latency_ms")
                        metrics["embeddings_p95_latency_ms"] = embed_metrics.get("embeddings_p95_latency_ms")
                        metrics["faiss_build_ms"] = embed_metrics.get("faiss_build_ms")
                        metrics["faiss_write_ms"] = embed_metrics.get("faiss_write_ms")
                        metrics["vectors_count"] = embed_metrics.get("vectors_count")
                        metrics.setdefault("stages", {})
                        metrics["stages"]["embeddings_ms"] = embed_metrics.get("embeddings_total_time_ms")
                        metrics["stages"]["faiss_build_ms"] = embed_metrics.get("faiss_build_ms")
                        metrics["stages"]["faiss_write_ms"] = embed_metrics.get("faiss_write_ms")
                    upload_metrics = stage_metrics.get("upload") if isinstance(stage_metrics, dict) else None
                    if isinstance(upload_metrics, dict):
                        metrics["upload_objects_count"] = upload_metrics.get("upload_objects_count")
                        metrics["upload_bytes_total"] = upload_metrics.get("upload_bytes_total")

                if success:
                    if job_id and self.ddkit_db.is_configured():
                        self.ddkit_db.mark_job_succeeded(job_id)
                    if parsed_key:
                        job_data["artifacts"] = {"s3_parsed_json_key": parsed_key}
                    job_data["status"] = "succeeded"
                    job_data["metrics"] = metrics
                    logger.info("doc_parse_index_metrics=%s", json.dumps(metrics, ensure_ascii=False))
                    logger.info(f"Successfully processed document {doc_id}")
                    return True
                else:
                    if job_id and self.ddkit_db.is_configured():
                        self.ddkit_db.mark_job_failed(job_id, error_message or "processing_failed")
                    job_data["status"] = "failed"
                    if error_message:
                        job_data["error_message"] = error_message
                    metrics["error"] = error_message or "processing_failed"
                    job_data["metrics"] = metrics
                    logger.info("doc_parse_index_metrics=%s", json.dumps(metrics, ensure_ascii=False))
                    logger.error(f"Failed to process document {doc_id}")
                    return False

        except Exception as e:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_failed(job_id, str(e))
            logger.error(f"Error processing doc_parse_index job for {doc_id}: {e}")
            metrics["error"] = str(e)
            job_data["metrics"] = metrics
            logger.info("doc_parse_index_metrics=%s", json.dumps(metrics, ensure_ascii=False))
            return False

    def _process_single_document(self, pipeline: Pipeline, pdf_path: Path, temp_path: Path,
                                tenant_id: str, case_id: str, doc_id: str, doc_kind: str,
                                title: str, source_url: str,
                                s3_parsed_json_key: Optional[str],
                                docling_do_ocr: Optional[bool] = None,
                                docling_do_tables: Optional[bool] = None) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """Process a single document through the ingestion pipeline (Docling path)."""
        metrics: Dict[str, Any] = {"stages": {}}
        try:
            # Step 1: Parse PDF
            # Use pipeline's expected debug_data/01_parsed_reports directory
            parsed_reports_dir = pipeline.paths.parsed_reports_path
            parsed_reports_dir.mkdir(parents=True, exist_ok=True)

            # Use PDFParser to parse the single PDF
            from src.pdf_parsing import PDFParser

            # Create metadata CSV for this document
            metadata_csv = temp_path / "metadata.csv"
            with open(metadata_csv, 'w', encoding='utf-8') as f:
                f.write("filename,doc_id,doc_kind,tenant_id,case_id,title,source_url\n")
                f.write(f"{pdf_path.name},{doc_id},{doc_kind},{tenant_id},{case_id},{title},{source_url}\n")

            parse_start = time.perf_counter()
            parser = PDFParser(
                output_dir=parsed_reports_dir,
                csv_metadata_path=metadata_csv,
                docling_do_ocr=docling_do_ocr,
                docling_do_tables=docling_do_tables
            )
            parser.parse_and_export(input_doc_paths=[pdf_path])
            metrics["stages"]["parse_ms"] = _ms_since(parse_start)

            # Step 2: Merge reports
            merge_start = time.perf_counter()
            pipeline.merge_reports()
            metrics["stages"]["merge_ms"] = _ms_since(merge_start)

            # Step 3: Chunk reports
            chunk_start = time.perf_counter()
            chunk_stats = pipeline.chunk_reports(include_serialized_tables=False)
            metrics["stages"]["chunk_ms"] = _ms_since(chunk_start)
            if chunk_stats:
                metrics["chunks"] = chunk_stats[0]

            # Step 4: Create vector databases
            embed_start = time.perf_counter()
            vector_stats = pipeline.create_vector_dbs()
            metrics["stages"]["embeddings_faiss_ms"] = _ms_since(embed_start)
            if vector_stats:
                metrics["embeddings"] = vector_stats[0]

            # Step 5: Upload results to S3
            upload_start = time.perf_counter()
            parsed_key, upload_metrics = self._upload_ingestion_results(
                temp_path, tenant_id, case_id, doc_id, s3_parsed_json_key
            )
            metrics["stages"]["upload_ms"] = _ms_since(upload_start)
            if upload_metrics:
                metrics["upload"] = upload_metrics
            return (parsed_key is not None), parsed_key, metrics

        except Exception as e:
            logger.error(f"Error in document processing pipeline for {doc_id}: {e}")
            return self._process_single_document_fallback(
                pipeline=pipeline,
                pdf_path=pdf_path,
                temp_path=temp_path,
                tenant_id=tenant_id,
                case_id=case_id,
                doc_id=doc_id,
                doc_kind=doc_kind,
                title=title,
                source_url=source_url,
                s3_parsed_json_key=s3_parsed_json_key
            )

    def _process_single_document_fast(self, pipeline: Pipeline, pdf_path: Path, temp_path: Path,
                                      tenant_id: str, case_id: str, doc_id: str, doc_kind: str,
                                      title: str, source_url: str,
                                      s3_parsed_json_key: Optional[str]) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """Fast parsing path using PyPDF2 text extraction."""
        metrics: Dict[str, Any] = {"stages": {}}
        try:
            parse_start = time.perf_counter()
            reader = PdfReader(str(pdf_path))
            pages = []
            for idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page": idx, "text": text})
            metrics["pages"] = len(pages)
            metrics["stages"]["parse_ms"] = _ms_since(parse_start)

            if not pages:
                raise RuntimeError("No text extracted from PDF")

            metainfo = {
                "sha1_name": doc_id,
                "doc_id": doc_id,
                "doc_kind": doc_kind,
                "tenant_id": tenant_id,
                "case_id": case_id,
                "title": title,
                "source_url": source_url
            }

            merge_start = time.perf_counter()
            parsed_dir = pipeline.paths.parsed_reports_path
            parsed_dir.mkdir(parents=True, exist_ok=True)
            parsed_path = parsed_dir / f"{doc_id}.json"
            with open(parsed_path, "w", encoding="utf-8") as f:
                json.dump({"metainfo": metainfo, "content": {"pages": pages}}, f, ensure_ascii=False, indent=2)

            merged_dir = pipeline.paths.merged_reports_path
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_path = merged_dir / f"{doc_id}.json"
            with open(merged_path, "w", encoding="utf-8") as f:
                json.dump({"metainfo": metainfo, "content": {"pages": pages, "chunks": None}}, f, ensure_ascii=False, indent=2)
            metrics["stages"]["merge_ms"] = _ms_since(merge_start)

            chunk_start = time.perf_counter()
            documents_dir = pipeline.paths.documents_dir
            splitter = TextSplitter()
            chunk_stats = splitter.split_all_reports(merged_dir, documents_dir)
            metrics["stages"]["chunk_ms"] = _ms_since(chunk_start)
            if chunk_stats:
                metrics["chunks"] = chunk_stats[0]

            embed_start = time.perf_counter()
            vector_dir = pipeline.paths.vector_db_dir
            vector_ingestor = VectorDBIngestor()
            vector_ingestor.process_reports(documents_dir, vector_dir)
            metrics["stages"]["embeddings_faiss_ms"] = _ms_since(embed_start)
            if vector_ingestor.last_report_metrics:
                metrics["embeddings"] = vector_ingestor.last_report_metrics[0]

            upload_start = time.perf_counter()
            parsed_key, upload_metrics = self._upload_ingestion_results(
                temp_path, tenant_id, case_id, doc_id, s3_parsed_json_key
            )
            metrics["stages"]["upload_ms"] = _ms_since(upload_start)
            if upload_metrics:
                metrics["upload"] = upload_metrics

            return (parsed_key is not None), parsed_key, metrics

        except Exception as fast_err:
            logger.error(f"Fast parsing failed for {doc_id}: {fast_err}")
            metrics["error"] = str(fast_err)
            return False, None, metrics

    def _process_single_document_with_timeout(
        self,
        temp_path: Path,
        pdf_path: Path,
        tenant_id: str,
        case_id: str,
        doc_id: str,
        doc_kind: str,
        title: str,
        source_url: str,
        s3_parsed_json_key: Optional[str],
        parser_mode: str = "docling",
        docling_do_ocr: Optional[bool] = None,
        docling_do_tables: Optional[bool] = None
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Run the pipeline in a subprocess and enforce a timeout."""
        timeout_seconds = max(1, settings.job_timeout_seconds)
        result_queue: Queue = Queue()
        payload = {
            "temp_path": str(temp_path),
            "pdf_path": str(pdf_path),
            "tenant_id": tenant_id,
            "case_id": case_id,
            "doc_id": doc_id,
            "doc_kind": doc_kind,
            "title": title,
            "source_url": source_url,
            "s3_parsed_json_key": s3_parsed_json_key,
            "parser_mode": parser_mode,
            "docling_do_ocr": docling_do_ocr,
            "docling_do_tables": docling_do_tables
        }
        proc = Process(target=_doc_parse_worker, args=(payload, result_queue))
        proc_start = time.perf_counter()
        proc.start()
        proc.join(timeout=timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return False, None, f"processing_timeout_{timeout_seconds}s", {"stages": {"subprocess_ms": _ms_since(proc_start)}}

        if result_queue.empty():
            return False, None, "processing_failed_no_result", {"stages": {"subprocess_ms": _ms_since(proc_start)}}

        result = result_queue.get()
        if result.get("success"):
            metrics = result.get("metrics") or {}
            metrics.setdefault("stages", {})
            metrics["stages"]["subprocess_ms"] = _ms_since(proc_start)
            return True, result.get("parsed_key"), None, metrics
        metrics = result.get("metrics") or {"stages": {}}
        metrics["stages"]["subprocess_ms"] = _ms_since(proc_start)
        return False, None, result.get("error", "processing_failed"), metrics

    def _validate_pdf_file(self, pdf_path: Path) -> Optional[str]:
        if not pdf_path.exists():
            return "pdf_missing"
        size = pdf_path.stat().st_size
        if size < settings.min_pdf_bytes:
            return f"pdf_too_small_{size}_bytes"
        try:
            with open(pdf_path, "rb") as handle:
                sig = handle.read(4)
            if sig != b"%PDF":
                return "pdf_signature_missing"
        except Exception as exc:
            return f"pdf_validation_error_{exc}"
        return None

    def _upload_ingestion_results(self, temp_path: Path, tenant_id: str, case_id: str, doc_id: str,
                                  s3_parsed_json_key: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Upload ingestion results to S3."""
        upload_metrics: Dict[str, Any] = {
            "upload_objects_count": 0,
            "upload_bytes_total": 0
        }
        try:
            # Upload parsed JSON (docling)
            parsed_path = temp_path / "debug_data" / "01_parsed_reports" / f"{doc_id}.json"
            parsed_key = s3_parsed_json_key or f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/parsed/docling.json"
            if parsed_path.exists():
                if not self.storage_client.upload_file(parsed_key, parsed_path):
                    logger.error(f"Failed to upload parsed JSON for doc {doc_id}")
                    return None, upload_metrics
                upload_metrics["upload_objects_count"] += 1
                upload_metrics["upload_bytes_total"] += parsed_path.stat().st_size
                if self.ddkit_db.is_configured():
                    pages_count = None
                    try:
                        with open(parsed_path, "r", encoding="utf-8") as f:
                            parsed_json = json.load(f)
                        pages_count = len(parsed_json.get("content", {}).get("pages", []))
                    except Exception:
                        pages_count = None
                    self.ddkit_db.update_document_parsed(doc_id, parsed_key, pages_count)

            # Upload chunked reports
            chunked_dir = temp_path / "databases" / "chunked_reports"
            if chunked_dir.exists():
                chunk_files = list(chunked_dir.glob("*.json"))
                if chunk_files:
                    bundle_path = chunked_dir / "chunks_bundle.tar.gz"
                    with tarfile.open(bundle_path, "w:gz") as tar:
                        for chunk_file in chunk_files:
                            tar.add(chunk_file, arcname=chunk_file.name)
                    bundle_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/chunks/chunks_bundle.tar.gz"
                    if not self.storage_client.upload_file(bundle_key, bundle_path):
                        logger.error(f"Failed to upload chunk bundle for {doc_id}")
                        return None, upload_metrics
                    upload_metrics["upload_objects_count"] += 1
                    upload_metrics["upload_bytes_total"] += bundle_path.stat().st_size

                    manifest = {
                        "doc_id": doc_id,
                        "chunks_count": len(chunk_files),
                        "format": "tar.gz",
                        "schema_version": 1
                    }
                    manifest_path = chunked_dir / "chunks_manifest.json"
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, ensure_ascii=False, indent=2)
                    manifest_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/chunks/manifest.json"
                    if not self.storage_client.upload_file(manifest_key, manifest_path):
                        logger.error(f"Failed to upload chunk manifest for {doc_id}")
                        return None, upload_metrics
                    upload_metrics["upload_objects_count"] += 1
                    upload_metrics["upload_bytes_total"] += manifest_path.stat().st_size

            # Upload vector databases
            vector_dir = temp_path / "databases" / "vector_dbs"
            if vector_dir.exists():
                for vector_file in vector_dir.glob("*.faiss"):
                    s3_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/vectors/{vector_file.name}"
                    if not self.storage_client.upload_file(s3_key, vector_file):
                        logger.error(f"Failed to upload vector file {vector_file.name}")
                        return None, upload_metrics
                    upload_metrics["upload_objects_count"] += 1
                    upload_metrics["upload_bytes_total"] += vector_file.stat().st_size

            logger.info(f"Successfully uploaded ingestion results for doc {doc_id}")
            return parsed_key, upload_metrics

        except Exception as e:
            logger.error(f"Error uploading ingestion results for {doc_id}: {e}")
            return None, upload_metrics

    def _process_single_document_fallback(self, pipeline: Pipeline, pdf_path: Path, temp_path: Path,
                                          tenant_id: str, case_id: str, doc_id: str, doc_kind: str,
                                          title: str, source_url: str,
                                          s3_parsed_json_key: Optional[str]) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """Fallback ingestion using PyPDF2 text extraction when Docling fails."""
        success, parsed_key, metrics = self._process_single_document_fast(
            pipeline=pipeline,
            pdf_path=pdf_path,
            temp_path=temp_path,
            tenant_id=tenant_id,
            case_id=case_id,
            doc_id=doc_id,
            doc_kind=doc_kind,
            title=title,
            source_url=source_url,
            s3_parsed_json_key=s3_parsed_json_key
        )
        metrics.setdefault("parser_path", "pypdf2_fallback")
        return success, parsed_key, metrics


class ReportGenerateProcessor:
    """Processor for report_generate jobs."""

    def __init__(self):
        self.storage_client = StorageClient()
        self.ddkit_db = DDKitDB()

    def process_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Process a report_generate job.

        Args:
            job_data: Job data containing tenant_id, case_id, sections_plan_key

        Returns:
            bool: True if processing successful
        """
        tenant_id = job_data["tenant_id"]
        case_id = job_data["case_id"]
        sections_plan_key = job_data["sections_plan_key"]
        report_id = job_data.get("report_id") or f"report_{case_id}"
        job_id = job_data.get("job_id")

        logger.info(f"Processing report_generate job: tenant={tenant_id}, case={case_id}")

        try:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_running(job_id)
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Download sections plan from S3
                local_plan_path = temp_path / "sections_plan.json"
                if not self.storage_client.download_to_path(sections_plan_key, local_plan_path):
                    logger.error(f"Failed to download sections plan for case {case_id}")
                    return False

                # Load sections plan
                with open(local_plan_path, 'r', encoding='utf-8') as f:
                    sections_plan = json.load(f)

                # Download case artifacts (chunks + vectors) from storage
                if not self._download_case_artifacts(temp_path, tenant_id, case_id):
                    logger.error(f"Failed to download case artifacts for case {case_id}")
                    return False

                documents_dir = temp_path / "databases" / "chunked_reports"
                vector_dir = temp_path / "databases" / "vector_dbs"
                chunk_files = list(documents_dir.glob("*.json"))
                vector_files = list(vector_dir.glob("*.faiss"))
                logger.info(
                    "Report artifacts ready: chunks=%d vectors=%d",
                    len(chunk_files),
                    len(vector_files)
                )

                # Generate DD report with timeout
                output_path = temp_path / "dd_report.json"
                generator = DDReportGenerator(
                    documents_dir=documents_dir,
                    vector_db_dir=vector_dir,
                    tenant_id=tenant_id,
                    case_id=case_id
                )
                start_ts = time.time()
                deadline = time.time() + settings.job_timeout_seconds
                logger.info("Report generation timeout set to %ds", settings.job_timeout_seconds)
                try:
                    report_payload = generator.generate_report(sections_plan, deadline=deadline)
                except TimeoutError as err:
                    logger.error(f"Report generation timed out: {err}")
                    if self.ddkit_db.is_configured():
                        self.ddkit_db.update_report_failed(
                            report_id=report_id,
                            tenant_id=tenant_id,
                            case_id=case_id,
                            error_message=str(err)
                        )
                    if job_id and self.ddkit_db.is_configured():
                        self.ddkit_db.mark_job_failed(job_id, str(err))
                    return False
                logger.info("Report generation finished in %.2fs", time.time() - start_ts)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report_payload, f, ensure_ascii=False, indent=2)

                # Upload report to S3
                s3_key = f"tenants/{tenant_id}/cases/{case_id}/reports/{report_id}/report.json"
                if not self.storage_client.upload_file(s3_key, output_path):
                    logger.error(f"Failed to upload report for case {case_id}")
                    return False

                if self.ddkit_db.is_configured():
                    self.ddkit_db.update_report_published(
                        report_id=report_id,
                        tenant_id=tenant_id,
                        case_id=case_id,
                        title=f"DD Report — {case_id}",
                        s3_report_json_key=s3_key,
                        sections_plan_key=sections_plan_key
                    )
                if job_id and self.ddkit_db.is_configured():
                    self.ddkit_db.mark_job_succeeded(job_id)

                job_data["artifacts"] = {"s3_report_json_key": s3_key}
                job_data["status"] = "succeeded"
                logger.info(f"Successfully generated and uploaded report for case {case_id}")
                return True

        except Exception as e:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_failed(job_id, str(e))
            job_data["status"] = "failed"
            logger.error(f"Error processing report_generate job for case {case_id}: {e}")
            return False

    def _download_case_artifacts(self, temp_path: Path, tenant_id: str, case_id: str) -> bool:
        base_prefix = f"tenants/{tenant_id}/cases/{case_id}/documents/"
        keys = self.storage_client.list_objects(base_prefix)
        if not keys:
            logger.error(f"No artifacts found under {base_prefix}")
            return False
        logger.info("Found %d artifacts under %s", len(keys), base_prefix)

        chunks_dir = temp_path / "databases" / "chunked_reports"
        vectors_dir = temp_path / "databases" / "vector_dbs"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        vectors_dir.mkdir(parents=True, exist_ok=True)

        chunk_count = 0
        vector_count = 0
        for key in keys:
            if "/chunks/" in key and key.endswith("chunks_bundle.tar.gz"):
                bundle_path = chunks_dir / Path(key).name
                if not self.storage_client.download_to_path(key, bundle_path):
                    logger.error(f"Failed to download chunk bundle {key}")
                    return False
                try:
                    with tarfile.open(bundle_path, "r:gz") as tar:
                        members = [m for m in tar.getmembers() if m.name.endswith(".json")]
                        tar.extractall(path=chunks_dir)
                    chunk_count += len(members)
                except Exception as exc:
                    logger.error(f"Failed to extract chunk bundle {key}: {exc}")
                    return False
                continue
            if "/chunks/" in key and key.endswith(".json"):
                if key.endswith("manifest.json"):
                    continue
                local_path = chunks_dir / Path(key).name
                if not self.storage_client.download_to_path(key, local_path):
                    logger.error(f"Failed to download chunk {key}")
                    return False
                chunk_count += 1
            if "/vectors/" in key and key.endswith(".faiss"):
                local_path = vectors_dir / Path(key).name
                if not self.storage_client.download_to_path(key, local_path):
                    logger.error(f"Failed to download vector {key}")
                    return False
                vector_count += 1
        logger.info("Downloaded artifacts: chunks=%d vectors=%d", chunk_count, vector_count)
        return True


class JobCallback:
    """Handle job completion callbacks."""

    @staticmethod
    def send_callback(callback_url: str, job_data: Dict[str, Any], success: bool, error_message: Optional[str] = None) -> Optional[float]:
        """Send callback to external service about job completion."""
        if not callback_url:
            return None

        try:
            start = time.perf_counter()
            status = "succeeded" if success else "failed"
            payload = {
                "job_id": job_data.get("job_id"),
                "job_type": job_data.get("job_type"),
                "status": status,
                "tenant_id": job_data.get("tenant_id"),
                "case_id": job_data.get("case_id"),
                "doc_id": job_data.get("doc_id"),
                "report_id": job_data.get("report_id"),
                "artifacts": job_data.get("artifacts", {}),
                "success": success,
                "timestamp": int(time.time()),
                "error_message": error_message
            }
            if job_data.get("metrics"):
                payload["metrics"] = job_data.get("metrics")

            headers = {}
            if settings.job_callback_token:
                headers["X-DDKIT-TOKEN"] = settings.job_callback_token
            response = requests.post(callback_url, json=payload, timeout=30, headers=headers)
            response.raise_for_status()

            duration_ms = _ms_since(start)
            logger.info(
                "Sent callback for job %s to %s in %.2fms",
                job_data.get("job_type"),
                callback_url,
                duration_ms
            )
            return duration_ms

        except Exception as e:
            logger.error(f"Failed to send callback: {e}")
            return None


