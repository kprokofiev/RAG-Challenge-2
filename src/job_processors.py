import json
import logging
import os
import tempfile
import time
import tarfile
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import requests
from urllib.parse import quote
from PyPDF2 import PdfReader
import redis as redis_lib

from src.pipeline import Pipeline
from src.dd_report_generator import DDReportGenerator
from src.case_view_v2_generator import CaseViewV2Generator
from src.pubmed_pipeline import PubMedIngestor
from src.storage_client import StorageClient
from src.settings import settings
from src.ddkit_db import DDKitDB
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor

logger = logging.getLogger(__name__)

# ── Report-readiness barrier constants ──────────────────────────────────────
# TTL for all run-flag Redis keys (48 h); they auto-expire.
_RUN_FLAG_TTL_S = 48 * 3600
# Maximum time the report worker will wait for corpus readiness before
# generating a partial report.
_PREFLIGHT_MAX_WAIT_S = int(os.getenv("DDKIT_REPORT_PREFLIGHT_MAX_WAIT_S", "1800"))  # 30 min
# Backoff schedule for requeue (seconds).
_PREFLIGHT_BACKOFF = [30, 60, 120, 180, 300]


def _redis_client() -> Optional[redis_lib.Redis]:
    """Return a Redis client using the same URL as the queue handler, or None."""
    url = settings.redis_url
    if not url:
        return None
    try:
        client = redis_lib.from_url(url, socket_connect_timeout=3)
        client.ping()
        return client
    except Exception as exc:
        logger.warning("Redis unavailable for run-flags: %s", exc)
        return None


def _run_flag_key(tenant_id: str, case_id: str, flag: str, run_id: Optional[str] = None) -> str:
    """Build the Redis key for a pipeline barrier flag.

    When run_id is provided the key is run-scoped (#12):
        ddkit:run:{tenant}:{case}:{run_id}:{flag}
    Otherwise the legacy key is used (backward-compatible for flags set without run_id):
        ddkit:run:{tenant}:{case}:{flag}
    """
    if run_id:
        return f"ddkit:run:{tenant_id}:{case_id}:{run_id}:{flag}"
    return f"ddkit:run:{tenant_id}:{case_id}:{flag}"


def set_run_flag(tenant_id: str, case_id: str, flag: str, run_id: Optional[str] = None) -> None:
    """Atomically set a pipeline-barrier flag in Redis.

    run_id isolates flags per pipeline run so re-runs of the same case don't
    inherit stale flags from a previous run (#12).
    """
    rdb = _redis_client()
    if rdb is None:
        return
    key = _run_flag_key(tenant_id, case_id, flag, run_id)
    try:
        rdb.set(key, str(int(time.time())), ex=_RUN_FLAG_TTL_S)
        logger.info("run_flag_set tenant=%s case=%s run=%s flag=%s", tenant_id, case_id, run_id, flag)
    except Exception as exc:
        logger.warning("Failed to set run flag %s: %s", key, exc)


def get_run_flag(tenant_id: str, case_id: str, flag: str, run_id: Optional[str] = None) -> bool:
    """Return True if the pipeline-barrier flag is set.

    Checks run-scoped key first (when run_id provided), then falls back to
    legacy key to preserve backward compatibility with flags set by older code.
    """
    rdb = _redis_client()
    if rdb is None:
        return True  # fail-open: no Redis → don't block
    try:
        if run_id:
            run_key = _run_flag_key(tenant_id, case_id, flag, run_id)
            if rdb.exists(run_key):
                return True
        # Fallback: legacy key (no run_id) — covers flags set by Go code
        # without run_id or by older worker versions.
        legacy_key = _run_flag_key(tenant_id, case_id, flag)
        return bool(rdb.exists(legacy_key))
    except Exception:
        return True  # fail-open


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

    def _set_index_done_if_ready(self, tenant_id: str, case_id: str, run_id: Optional[str] = None) -> None:
        """
        After each terminal doc_parse_index outcome (success OR failure), check whether
        the entire corpus for this case is "settled" (no docs stuck in non-terminal states).
        If so, set the ddkit:run:{tenant}:{case}:{run_id}:index_done Redis flag (#12).

        Terminal statuses (corpus settled):
          - indexed  : successfully parsed + embedded + uploaded
          - parsed   : upload done but callback to api-gateway pending (transient; treated as terminal here)
          - failed   : processing failed; counted as missing in completeness
          - skipped  : explicitly skipped (e.g. duplicate, unsupported type)
          - unsupported : doc kind not supported by parser

        We also require at least one indexed or parsed document so the flag is
        not set on a completely empty or all-failed corpus.
        """
        if not self.ddkit_db.is_configured():
            return
        try:
            docs = self.ddkit_db.list_case_documents(tenant_id=tenant_id, case_id=case_id)
            if not docs:
                return
            # Terminal states: corpus is settled when no doc is still in-flight (#2).
            # Must include all blocked/error statuses so paywalled docs don't block the pipeline.
            terminal = {"indexed", "failed", "parsed", "skipped", "unsupported",
                        "blocked_paywall", "captcha", "forbidden_403", "rate_limited_429",
                        "requires_login", "robots_denied", "timeout",
                        "parsed_empty",  # Sprint-2: docs that parsed but produced no text
                        }
            in_progress = [d for d in docs if str(d.get("status", "")).lower() not in terminal]
            indexed_or_parsed = [
                d for d in docs
                if str(d.get("status", "")).lower() in {"indexed", "parsed"}
            ]
            failed = [d for d in docs if str(d.get("status", "")).lower() in {
                "failed", "skipped", "unsupported", "parsed_empty",
            }]
            if in_progress:
                # Check whether the parse queue is drained — if so, any remaining
                # 'rendered' (or other pre-queue) docs are effectively stuck and
                # should not block index_done indefinitely.
                rendered_only = all(
                    str(d.get("status", "")).lower() in {"rendered", "created"} for d in in_progress
                )
                if rendered_only:
                    rdb = _redis_client()
                    queue_len = 0
                    fetch_queue_len = 0
                    if rdb is not None:
                        try:
                            queue_len = rdb.llen("ddkit:doc_parse_index")
                            fetch_queue_len = rdb.llen("ddkit:doc_fetch_render")
                        except Exception:
                            pass
                    # Only treat as terminal if BOTH queues are drained.
                    # If doc_fetch_render still has items, docs are en-route to rendered state.
                    if queue_len == 0 and fetch_queue_len == 0:
                        logger.warning(
                            "index_done_check: %d docs stuck in pre-queue state but both queues empty — "
                            "treating as terminal to unblock corpus case=%s statuses=%s",
                            len(in_progress), case_id,
                            {str(d.get("status", "")).lower() for d in in_progress},
                        )
                        # Fall through to set index_done below
                    else:
                        logger.info(
                            "index_done_check: %d docs still in progress case=%s parse_queue=%d fetch_queue=%d",
                            len(in_progress), case_id, queue_len, fetch_queue_len,
                        )
                        return
                else:
                    statuses_blocking = {}
                    for d in in_progress:
                        s = str(d.get("status", "unknown")).lower()
                        statuses_blocking[s] = statuses_blocking.get(s, 0) + 1
                    logger.info(
                        "index_done_check: %d docs still in progress case=%s blocking_statuses=%s",
                        len(in_progress), case_id, statuses_blocking,
                    )
                    return
            if not indexed_or_parsed:
                logger.info(
                    "index_done_check: 0 indexed/parsed docs (all failed?) case=%s failed=%d — not setting flag",
                    case_id, len(failed),
                )
                return
            set_run_flag(tenant_id, case_id, "index_done", run_id)
            logger.info(
                "index_done: corpus settled case=%s run=%s indexed_or_parsed=%d failed=%d total=%d",
                case_id, run_id, len(indexed_or_parsed), len(failed), len(docs),
            )
            # Trigger async case-level index build (#7). Run in a background thread
            # so we don't block the current job callback path.
            import threading
            def _build_case_index_bg():
                try:
                    from src.ingestion import VectorDBIngestor
                    from src.storage_client import StorageClient as _SC
                    sc = _SC()
                    # Download all vector shards for this case to a temp dir
                    import tempfile
                    with tempfile.TemporaryDirectory() as _tmp:
                        _tmp_path = Path(_tmp)
                        prefix = f"tenants/{tenant_id}/cases/{case_id}/documents/"
                        keys = sc.list_objects(prefix)
                        downloaded = 0
                        for key in (keys or []):
                            if "/vectors/" in key and key.endswith(".faiss"):
                                local = _tmp_path / Path(key).name
                                if sc.download_to_path(key, local):
                                    downloaded += 1
                        if downloaded == 0:
                            return
                        m = VectorDBIngestor.build_case_index(_tmp_path)
                        if m.get("output_path"):
                            case_idx_key = f"tenants/{tenant_id}/cases/{case_id}/vectors/case_index.faiss"
                            sc.upload_file(case_idx_key, Path(m["output_path"]))
                            logger.info(
                                "case_index_built case=%s vectors=%d docs=%d build_ms=%.1f",
                                case_id, m.get("total_vectors", 0), m.get("doc_count", 0), m.get("build_ms", 0),
                            )
                except Exception as _exc:
                    logger.warning("case_index_build_failed case=%s: %s", case_id, _exc)
            threading.Thread(target=_build_case_index_bg, daemon=True).start()
        except Exception as exc:
            logger.warning("_set_index_done_if_ready failed (non-fatal): %s", exc)

    def _detect_text_layer(self, pdf_path: Path, max_pages: int = 3, min_chars: int = 200) -> Dict[str, Any]:
        """Detect whether PDF has a usable text layer.

        Sprint-2 improvements:
        - Sample pages more representatively (first, middle, last up to max_pages)
        - Track Cyrillic character ratio to detect RU-language scans with partial text layer
        - Detect likely tabular/column layout via whitespace ratio
        """
        result = {
            "has_text_layer": False,
            "pages_total": 0,
            "pages_checked": 0,
            "text_chars": 0,
            "cyrillic_ratio": 0.0,
            "whitespace_ratio": 0.0,
        }
        try:
            reader = PdfReader(str(pdf_path))
            pages_total = len(reader.pages)
            if pages_total == 0:
                return result

            # Sample: first page, last page, and middle pages up to max_pages
            indices = set()
            indices.add(0)
            if pages_total > 1:
                indices.add(pages_total - 1)
            if pages_total > 2:
                indices.add(pages_total // 2)
            # Fill remaining slots from beginning
            for i in range(pages_total):
                if len(indices) >= max_pages:
                    break
                indices.add(i)
            sample_indices = sorted(indices)[:max_pages]

            text_chars = 0
            cyrillic_chars = 0
            whitespace_chars = 0
            total_extracted = 0
            for idx in sample_indices:
                page_text = reader.pages[idx].extract_text() or ""
                stripped = page_text.strip()
                text_chars += len(stripped)
                total_extracted += len(page_text)
                for ch in page_text:
                    if '\u0400' <= ch <= '\u04FF':
                        cyrillic_chars += 1
                    if ch in ' \t':
                        whitespace_chars += 1

            cyrillic_ratio = cyrillic_chars / total_extracted if total_extracted > 0 else 0.0
            whitespace_ratio = whitespace_chars / total_extracted if total_extracted > 0 else 0.0
            result.update({
                "has_text_layer": text_chars >= min_chars,
                "pages_total": pages_total,
                "pages_checked": len(sample_indices),
                "text_chars": text_chars,
                "cyrillic_ratio": round(cyrillic_ratio, 3),
                "whitespace_ratio": round(whitespace_ratio, 3),
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
        # Correlation fields for distributed tracing (#13)
        trace_id = job_data.get("trace_id")
        run_id = job_data.get("run_id")
        attempt = job_data.get("attempt", 0)

        logger.info(
            "doc_parse_index_start tenant=%s case=%s doc=%s job=%s trace=%s run=%s attempt=%s",
            tenant_id, case_id, doc_id, job_id, trace_id, run_id, attempt,
        )
        metrics: Dict[str, Any] = {
            "job_type": "doc_parse_index",
            "job_id": job_id,
            "tenant_id": tenant_id,
            "case_id": case_id,
            "doc_id": doc_id,
            "trace_id": trace_id,
            "run_id": run_id,
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
                    "cyrillic_ratio": text_layer.get("cyrillic_ratio"),
                    "whitespace_ratio": text_layer.get("whitespace_ratio"),
                })

                # Sprint-2: doc_kind-aware parser selection.
                # Some doc_kinds always need Docling+OCR (scanned RU instructions, patents)
                # regardless of whether PyPDF2 finds a text layer.
                _ocr_doc_kinds = {
                    k.strip().lower()
                    for k in settings.docling_ocr_doc_kinds.split(",")
                    if k.strip()
                }
                _tables_doc_kinds = {
                    k.strip().lower()
                    for k in settings.docling_tables_doc_kinds.split(",")
                    if k.strip()
                }
                _dk = (doc_kind or "").lower()

                force_docling_ocr = _dk in _ocr_doc_kinds
                force_tables = _dk in _tables_doc_kinds

                if force_docling_ocr and not text_layer.get("has_text_layer"):
                    # Scan doc: Docling + OCR
                    parser_mode = "docling"
                    docling_do_ocr = True
                elif force_docling_ocr and text_layer.get("has_text_layer"):
                    # OCR-mandatory kind but has text — still use Docling for layout/tables
                    parser_mode = "docling"
                    docling_do_ocr = False  # text layer present; skip OCR for speed
                elif not text_layer.get("has_text_layer"):
                    # Generic scan: Docling + OCR
                    parser_mode = "docling"
                    docling_do_ocr = True
                else:
                    # Has text layer, not a forced-OCR kind: fast path
                    parser_mode = "fast_text"
                    docling_do_ocr = False

                # Tables: always Docling path + tables for table-required doc_kinds
                if force_tables and parser_mode == "fast_text":
                    # Upgrade to Docling (no OCR needed if text layer exists)
                    parser_mode = "docling"
                    docling_do_ocr = False

                docling_do_tables = force_tables or (bool(settings.docling_do_tables) and parser_mode == "docling")

                metrics["parser_path"] = parser_mode
                metrics["parser_used"] = parser_mode
                metrics["docling_do_ocr"] = docling_do_ocr
                metrics["docling_do_tables"] = docling_do_tables
                metrics["force_docling_ocr"] = force_docling_ocr
                metrics["force_tables"] = force_tables

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
                    # Sprint-2: compute quality metrics from parsed result
                    quality = self._compute_parse_quality(metrics)
                    metrics["parse_quality"] = quality

                    if quality.get("is_empty"):
                        # Document parsed but produced no usable text — treat as parsed_empty
                        logger.warning(
                            "doc_parse_index_empty doc=%s reason=%s chars=%d",
                            doc_id, quality.get("empty_reason"), quality.get("text_chars_total", 0),
                        )
                        if job_id and self.ddkit_db.is_configured():
                            self.ddkit_db.mark_job_failed(job_id, "parsed_empty:" + (quality.get("empty_reason") or ""))
                        job_data["status"] = "parsed_empty"
                        job_data["error_message"] = "parsed_empty:" + (quality.get("empty_reason") or "")
                        metrics["error"] = job_data["error_message"]
                        job_data["metrics"] = metrics
                        logger.info("doc_parse_index_metrics=%s", json.dumps(metrics, ensure_ascii=False))
                        self._set_index_done_if_ready(tenant_id, case_id, run_id)
                        return False

                    if job_id and self.ddkit_db.is_configured():
                        self.ddkit_db.mark_job_succeeded(job_id)
                    if parsed_key:
                        job_data["artifacts"] = {"s3_parsed_json_key": parsed_key}
                    job_data["status"] = "succeeded"
                    job_data["metrics"] = metrics
                    logger.info("doc_parse_index_metrics=%s", json.dumps(metrics, ensure_ascii=False))
                    logger.info(
                        "doc_parse_index_ok doc=%s parser=%s ocr=%s tables=%s "
                        "chars=%d pages=%d tables_count=%d garbage_ratio=%.3f",
                        doc_id, parser_mode, docling_do_ocr, docling_do_tables,
                        quality.get("text_chars_total", 0),
                        quality.get("pages_total", 0),
                        quality.get("tables_count", 0),
                        quality.get("garbage_ratio", 0.0),
                    )
                    # After each terminal outcome (success or failure), check if the full corpus
                    # is settled and set index_done if so. This ensures the flag is set even
                    # when some docs fail — partial corpus → partial report is still valid.
                    self._set_index_done_if_ready(tenant_id, case_id, run_id)
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
                    # Check if corpus is settled despite this doc failing.
                    self._set_index_done_if_ready(tenant_id, case_id, run_id)
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

    def _compute_parse_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Sprint-2: compute sanity metrics from job metrics after successful parsing.

        Returns a dict with:
          text_chars_total      — total extracted characters (chunks × avg tokens * ~4)
          pages_total           — page count from PDF detection
          pages_with_text_ratio — fraction of pages that contributed text
          tables_count          — number of tables extracted (if available)
          ocr_used              — whether OCR was enabled for this doc
          garbage_ratio         — fraction of suspicious chars (replacement char / non-printable)
          is_empty              — True if doc effectively produced no usable text
          empty_reason          — explanation if is_empty
        """
        quality: Dict[str, Any] = {
            "text_chars_total": 0,
            "pages_total": metrics.get("pages") or 0,
            "pages_with_text_ratio": 0.0,
            "tables_count": 0,
            "ocr_used": bool(metrics.get("docling_do_ocr")),
            "garbage_ratio": 0.0,
            "is_empty": False,
            "empty_reason": None,
        }

        # Estimate total chars from chunk metrics (chunks_count × avg_tokens × ~4 chars/token)
        chunks_count = metrics.get("chunks_count") or 0
        avg_tokens = metrics.get("avg_tokens_per_chunk") or 0
        if chunks_count and avg_tokens:
            quality["text_chars_total"] = int(chunks_count * avg_tokens * 4)
        elif metrics.get("text_layer_chars"):
            quality["text_chars_total"] = metrics["text_layer_chars"]

        # tables_count from Docling stage if present in stage_metrics
        embed_m = metrics.get("embeddings") if isinstance(metrics.get("embeddings"), dict) else {}
        quality["tables_count"] = embed_m.get("tables_count", 0)

        # pages_with_text_ratio (approximation: if we got chunks, assume all sampled pages had text)
        pages_total = quality["pages_total"] or 1
        if quality["text_chars_total"] > 0:
            quality["pages_with_text_ratio"] = min(1.0, chunks_count / max(pages_total, 1))

        # Garbage ratio: from text_layer detection if available
        # (exact ratio requires reading parsed JSON — use heuristic from cyrillic/whitespace metrics)
        # If docling/ocr produced very low char count vs page count → suspicious
        chars_per_page = quality["text_chars_total"] / max(pages_total, 1)
        if chars_per_page < 50 and quality["text_chars_total"] < settings.parse_quality_min_chars:
            quality["garbage_ratio"] = 1.0
        else:
            quality["garbage_ratio"] = 0.0

        # is_empty check
        if quality["text_chars_total"] < settings.parse_quality_min_chars:
            quality["is_empty"] = True
            quality["empty_reason"] = (
                f"text_chars_total={quality['text_chars_total']} < threshold={settings.parse_quality_min_chars}"
            )
        elif chunks_count == 0:
            quality["is_empty"] = True
            quality["empty_reason"] = "chunks_count=0"

        return quality

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

    # ── Preflight readiness gate ─────────────────────────────────────────────

    def _corpus_ready(self, tenant_id: str, case_id: str, run_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check whether the document corpus for this case is ready for report generation.

        Returns (is_ready, reason_if_not_ready).

        Rules (all must pass):
        1. attach_deep_done flag is set   — Wave-2 research sources were attached.
           (Skipped if research_run_started flag is NOT set — fast-path cases that
           never trigger research:run don't need to wait for Wave-2.)
        2. index_done flag is set          — all docs settled in terminal states.
        3. At least 1 indexed document.

        run_id is used to scope Redis flag checks to this specific run (#12).
        Falls back to legacy (unscoped) key for backward compatibility.
        If Redis is unavailable we fail-open (return True) so the system degrades
        gracefully rather than blocking forever.
        """
        # Only wait for attach_deep_done if research:run was actually started.
        research_started = get_run_flag(tenant_id, case_id, "research_run_started", run_id)
        if research_started:
            deep_done = get_run_flag(tenant_id, case_id, "attach_deep_done", run_id)
            if not deep_done:
                return False, "attach_deep_not_done"

        index_done = get_run_flag(tenant_id, case_id, "index_done", run_id)
        if not index_done:
            # Fallback: query DB directly — Redis flag might have been missed.
            if self.ddkit_db.is_configured():
                try:
                    docs = self.ddkit_db.list_case_documents(tenant_id=tenant_id, case_id=case_id)
                    # All states that count as "terminal" for corpus-ready check (#2).
                    # Includes blocked/paywall statuses so they don't hold up the pipeline forever.
                    terminal = {"indexed", "failed", "parsed", "skipped", "unsupported",
                                "blocked_paywall", "captcha", "forbidden_403", "rate_limited_429",
                                "requires_login", "robots_denied", "timeout",
                                "parsed_empty",  # Sprint-2
                                }
                    in_progress = [d for d in docs if str(d.get("status", "")).lower() not in terminal]
                    indexed = [d for d in docs if str(d.get("status", "")).lower() == "indexed"]
                    if in_progress:
                        return False, f"docs_still_indexing:{len(in_progress)}"
                    if not indexed:
                        return False, "no_indexed_docs"
                    # DB says ready → set the flag retroactively
                    set_run_flag(tenant_id, case_id, "index_done", run_id)
                except Exception as exc:
                    logger.warning("DB readiness fallback check failed: %s", exc)
                    return True, ""  # fail-open
            else:
                return True, ""  # no DB → fail-open

        return True, ""

    def preflight_case_ready(
        self,
        job_data: Dict[str, Any],
        attempt: int,
    ) -> Tuple[bool, bool]:
        """
        Run the pre-flight readiness check and decide whether to proceed or requeue.

        Returns (proceed, requeued):
          - proceed=True  → start report generation now
          - proceed=False, requeued=True  → job re-enqueued with backoff; caller should return True
          - proceed=False, requeued=False → max wait exceeded; generate partial report
        """
        tenant_id = job_data["tenant_id"]
        case_id = job_data["case_id"]
        run_id = job_data.get("run_id")
        enqueued_at = job_data.get("enqueued_at", time.time())
        elapsed = time.time() - float(enqueued_at)

        ready, reason = self._corpus_ready(tenant_id, case_id, run_id)
        if ready:
            logger.info(
                "preflight_pass tenant=%s case=%s elapsed=%.1fs",
                tenant_id, case_id, elapsed,
            )
            return True, False

        if elapsed >= _PREFLIGHT_MAX_WAIT_S:
            logger.warning(
                "preflight_timeout tenant=%s case=%s elapsed=%.1fs reason=%s — generating partial report",
                tenant_id, case_id, elapsed, reason,
            )
            job_data["is_partial"] = True
            job_data["partial_reasons"] = [f"timeout_preflight:{reason}"]
            return True, False  # proceed with partial

        # Requeue with backoff
        backoff_idx = min(attempt, len(_PREFLIGHT_BACKOFF) - 1)
        delay = _PREFLIGHT_BACKOFF[backoff_idx]
        job_data["attempt"] = attempt + 1
        if "enqueued_at" not in job_data:
            job_data["enqueued_at"] = time.time()
        logger.info(
            "preflight_requeue tenant=%s case=%s attempt=%d delay=%ds elapsed=%.1fs reason=%s",
            tenant_id, case_id, attempt, delay, elapsed, reason,
        )
        rdb = _redis_client()
        if rdb is not None:
            try:
                time.sleep(delay)
                queue = settings.queue_report_generate
                rdb.lpush(queue, json.dumps(job_data))
            except Exception as exc:
                logger.error("Failed to requeue report_generate job: %s", exc)
                # Fall through → generate partial rather than lose the job
                job_data["is_partial"] = True
                job_data["partial_reasons"] = [f"requeue_failed:{exc}"]
                return True, False
        return False, True

    # ── Main process_job ─────────────────────────────────────────────────────

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
        attempt = int(job_data.get("attempt", 0))
        # Correlation fields for distributed tracing (#13)
        trace_id = job_data.get("trace_id")
        run_id = job_data.get("run_id")

        logger.info(
            "report_generate_start tenant=%s case=%s report=%s attempt=%d trace=%s run=%s",
            tenant_id, case_id, report_id, attempt, trace_id, run_id,
        )

        try:
            # ── Preflight readiness gate ──────────────────────────────────────
            # Ensure Wave-2 sources are attached and all docs are indexed before
            # downloading artifacts. If not ready, requeue with backoff.
            # On timeout (>_PREFLIGHT_MAX_WAIT_S) fall through and generate a
            # partial report.
            # skip_preflight=True is set by report:regenerate endpoint (corpus settled).
            if not job_data.get("is_partial") and not job_data.get("skip_preflight"):
                proceed, requeued = self.preflight_case_ready(job_data, attempt)
                if requeued:
                    return False  # job re-enqueued with backoff; suppress success callback
                # proceed=True here (either ready or partial fallback)

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
                # Resolve INN so DDReportGenerator can anchor retrieval to the drug.
                # Priority: job_data["inn"] (set by api-gateway) → case_views DB lookup.
                case_inn: Optional[str] = job_data.get("inn") or None
                if case_inn:
                    case_inn = str(case_inn).strip() or None
                if not case_inn and self.ddkit_db.is_configured():
                    try:
                        case_inn = self.ddkit_db.get_case_inn(tenant_id, case_id)
                    except Exception as _inn_err:
                        logger.warning("Could not retrieve INN for case %s: %s", case_id, _inn_err)
                logger.info("Report generator INN for case %s: %s", case_id, case_inn or "(unknown)")
                generator = DDReportGenerator(
                    documents_dir=documents_dir,
                    vector_db_dir=vector_dir,
                    tenant_id=tenant_id,
                    case_id=case_id,
                    ddkit_db=self.ddkit_db,
                    inn=case_inn,
                )
                start_ts = time.time()
                deadline = time.time() + settings.job_timeout_seconds
                logger.info("Report generation timeout set to %ds", settings.job_timeout_seconds)
                is_partial = bool(job_data.get("is_partial", False))
                partial_reasons = list(job_data.get("partial_reasons") or [])
                try:
                    report_payload = generator.generate_report(
                        sections_plan,
                        deadline=deadline,
                        is_partial=is_partial,
                        partial_reasons=partial_reasons,
                    )
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
                completeness = report_payload.get("completeness", {})
                included_total = completeness.get("included", {}).get("total", "?")
                expected_total = completeness.get("expected", {}).get("total", "?")
                missing_total = completeness.get("missing", {}).get("total", "?")
                is_partial = completeness.get("is_partial", False)
                ratio = completeness.get("ratio", "?")
                partial_reasons = completeness.get("partial_reasons") or []
                logger.info(
                    "report_generate_done tenant=%s case=%s report=%s "
                    "is_partial=%s included=%s expected=%s missing=%s",
                    tenant_id, case_id, report_id,
                    is_partial, included_total, expected_total, missing_total,
                )
                # One-line machine-readable summary for quick debugging (P1.2)
                logger.info(
                    "full_run_summary case=%s expected=%s included=%s missing=%s ratio=%s "
                    "is_partial=%s partial_reasons=%s",
                    case_id, expected_total, included_total, missing_total, ratio,
                    is_partial, partial_reasons,
                )
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
            # Snapshot-only cases are valid: we can still generate a (partial) case view,
            # and later enrichment (auto-attach / PubMed) can add documents.
            logger.info("No document artifacts found under %s (continuing with empty corpus)", base_prefix)
            chunks_dir = temp_path / "databases" / "chunked_reports"
            vectors_dir = temp_path / "databases" / "vector_dbs"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            vectors_dir.mkdir(parents=True, exist_ok=True)
            return True
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


class CaseViewGenerateProcessor:
    """Processor for case_view_generate jobs."""

    def __init__(self):
        self.storage_client = StorageClient()
        self.ddkit_db = DDKitDB()

    def _gateway_base_url(self) -> str:
        """
        Base URL for the Go API gateway used for auto-attach of missing sources.

        In docker compose, the worker connects via host.docker.internal:8085 (published port).
        """
        base = (
            os.getenv("DDKIT_GATEWAY_BASE_URL")
            or os.getenv("DDKIT_API_BASE_URL")
            or os.getenv("DDKIT_BASE_URL")
            or "http://host.docker.internal:8085"
        )
        return base.rstrip("/")

    def _attach_sources(self, case_id: str, sources: list[dict[str, Any]], force: bool = False) -> list[str]:
        """
        Calls POST /v1/cases/{caseId}/sources:attach and returns list of created/queued doc_ids.
        """
        if not sources:
            return []
        url = f"{self._gateway_base_url()}/v1/cases/{case_id}/sources:attach"
        payload = {"sources": sources, "force": bool(force)}
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json() or {}
        doc_ids: list[str] = []
        for it in data.get("results") or []:
            if isinstance(it, dict) and it.get("document_id"):
                doc_ids.append(str(it["document_id"]))
        return doc_ids

    def _wait_for_documents_indexed(self, doc_ids: list[str], deadline: Optional[float]) -> None:
        """
        Wait until doc_fetch_render -> doc_parse_index completes and the document becomes usable for retrieval.

        We consider a document "ready" when:
        - status in {"indexed", "parsed"} AND s3_parsed_json_key is present
        """
        if not doc_ids or not self.ddkit_db.is_configured():
            return

        remaining = set(str(d) for d in doc_ids if str(d).strip())
        backoff_s = 2.0
        while remaining:
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"Timeout waiting for documents to index: {sorted(remaining)[:5]}")

            done: set[str] = set()
            for doc_id in list(remaining):
                row = self.ddkit_db.get_document(doc_id)
                if not row:
                    continue
                status = str(row.get("status") or "").lower()
                parsed_key = row.get("s3_parsed_json_key")
                if status == "failed":
                    msg = row.get("error_message") or "unknown"
                    logger.warning("Document failed during fetch/index: doc=%s err=%s", doc_id, msg)
                    done.add(doc_id)  # don't block case view generation forever
                    continue
                if status in {"indexed", "parsed"} and parsed_key:
                    done.add(doc_id)

            remaining -= done
            if remaining:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.2, 8.0)

    def _ensure_min_case_corpus(
        self,
        tenant_id: str,
        case_id: str,
        inn: str,
        deadline: Optional[float],
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Minimal auto-attach to make CaseView v2 usable even when the case starts with only
        a snapshot and no attached documents.

        What we try to ensure:
        - US: openFDA label + openFDA drugsfda (approval metadata)
        - Clinical: 1-2 CTGov registry documents
        - EU: best-effort (SmPC/PIL links if available)
        - RU: instruction PDF if a link exists in the frontend snapshot
        """
        inn = (inn or "").strip()
        if not inn or not self.ddkit_db.is_configured():
            return

        docs = self.ddkit_db.list_case_documents(tenant_id=tenant_id, case_id=case_id)
        kinds = {str(d.get("doc_kind") or "").lower() for d in docs}

        sources: list[dict[str, Any]] = []
        snapshot = snapshot if isinstance(snapshot, dict) else None

        def _collect_ru_instruction_links() -> list[str]:
            if not snapshot:
                return []
            urls: list[str] = []
            seeds = []
            for key in ["ru_instruction_url", "ru_instruction", "instruction_url", "instructionUrl"]:
                val = snapshot.get(key)
                if isinstance(val, str):
                    seeds.append(val)
                elif isinstance(val, list):
                    seeds.extend([v for v in val if isinstance(v, str)])
            ru_sections = snapshot.get("ruSections") or {}
            reg = ru_sections.get("regulatory") or {}
            items = reg.get("items") or []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                links = item.get("links") or {}
                if not isinstance(links, dict):
                    continue
                for key, value in links.items():
                    values = value if isinstance(value, list) else [value]
                    for v in values:
                        if not isinstance(v, str):
                            continue
                        key_l = str(key).lower()
                        url_l = v.lower()
                        if any(tok in key_l for tok in ["instruction", "instr", "leaflet", "pil"]):
                            seeds.append(v)
                        elif any(tok in url_l for tok in ["instruction", "instr", "leaflet"]):
                            seeds.append(v)
            for v in seeds:
                v = v.strip()
                if v and v.startswith("http") and v not in urls:
                    urls.append(v)
            return urls

        # US regulatory documents
        if "label" not in kinds:
            sources.append(
                {
                    "url": f"https://api.fda.gov/drug/label.json?searchTerm={quote(inn)}",
                    "doc_kind": "label",
                    "title": f"FDA label (openFDA): {inn}",
                    "region": "us",
                }
            )
        if "us_fda" not in kinds:
            # Drugs@FDA dataset usually contains approval/application metadata that label JSON may not have.
            sources.append(
                {
                    "url": f"https://api.fda.gov/drug/drugsfda.json?search=openfda.generic_name:{quote(inn)}&limit=5",
                    "doc_kind": "us_fda",
                    "title": f"FDA Drugs@FDA (openFDA): {inn}",
                    "region": "us",
                }
            )
            # API gateway aggregation can include approval and label links in one payload.
            sources.append(
                {
                    "url": f"{self._gateway_base_url()}/api/v1/regulation/us?inn={quote(inn)}",
                    "doc_kind": "us_fda",
                    "title": f"US regulatory summary: {inn}",
                    "region": "us",
                }
            )

        # EU best-effort: if the API gateway can provide direct SmPC/PIL URLs, attach them.
        if not ({"smpc", "pil", "assessment_report", "epar"} & kinds):
            try:
                eu_url = f"{self._gateway_base_url()}/api/v1/regulation/eu?inn={quote(inn)}"
                eu_resp = requests.get(eu_url, timeout=60)
                eu_resp.raise_for_status()
                eu_data = (
                    eu_resp.json()
                    if eu_resp.headers.get("content-type", "").startswith("application/json")
                    else json.loads(eu_resp.text)
                )
                eu_added = False
                eu_auth = eu_data.get("eu_authorization") or {}
                if isinstance(eu_auth, dict):
                    if eu_auth.get("smpc_url"):
                        sources.append({"url": eu_auth["smpc_url"], "doc_kind": "smpc", "title": f"EU SmPC: {inn}", "region": "eu"})
                        eu_added = True
                    if eu_auth.get("pil_url"):
                        sources.append({"url": eu_auth["pil_url"], "doc_kind": "pil", "title": f"EU PIL: {inn}", "region": "eu"})
                        eu_added = True
                    if eu_auth.get("epar_url"):
                        # Some procedures return only an overview link; still useful as a UI evidence target.
                        sources.append({"url": eu_auth["epar_url"], "doc_kind": "epar", "title": f"EU EPAR/Overview: {inn}", "region": "eu"})
                        eu_added = True
                ema = eu_data.get("ema_centralized") or {}
                if isinstance(ema, dict):
                    for link in ema.get("smcp_links") or []:
                        if link:
                            sources.append({"url": link, "doc_kind": "smpc", "title": f"EU SmPC (EMA): {inn}", "region": "eu"})
                            eu_added = True
                    for link in ema.get("pil_links") or []:
                        if link:
                            sources.append({"url": link, "doc_kind": "pil", "title": f"EU PIL (EMA): {inn}", "region": "eu"})
                            eu_added = True
                    if ema.get("epar_url"):
                        sources.append({"url": ema["epar_url"], "doc_kind": "epar", "title": f"EU EPAR (EMA): {inn}", "region": "eu"})
                        eu_added = True
                if not eu_added:
                    sources.append(
                        {
                            "url": eu_url,
                            "doc_kind": "epar",
                            "title": f"EU regulatory summary: {inn}",
                            "region": "eu",
                        }
                    )
            except Exception as exc:
                logger.info("EU regulatory preflight skipped: %s", exc)

        # RU instruction (if snapshot contains a link)
        if "ru_instruction" not in kinds:
            ru_links = _collect_ru_instruction_links()
            if ru_links:
                for link in ru_links[:2]:
                    sources.append(
                        {
                            "url": link,
                            "doc_kind": "ru_instruction",
                            "title": f"RU instruction: {inn}",
                            "region": "ru",
                        }
                    )

        # RU regulatory (GRLS) JSON + card/instruction links if available
        if not ({"grls", "grls_card"} & kinds):
            try:
                ru_reg_url = f"{self._gateway_base_url()}/v1/ru/drugs/{quote(inn)}/regulatory"
                ru_resp = requests.get(ru_reg_url, timeout=60)
                ru_resp.raise_for_status()
                ru_data = (
                    ru_resp.json()
                    if ru_resp.headers.get("content-type", "").startswith("application/json")
                    else json.loads(ru_resp.text)
                )
                sources.append(
                    {
                        "url": ru_reg_url,
                        "doc_kind": "grls",
                        "title": f"RU regulatory (GRLS): {inn}",
                        "region": "ru",
                    }
                )
                reg_block = (ru_data.get("regulatory") or {}) if isinstance(ru_data, dict) else {}
                registrations = reg_block.get("registrations") or []
                for reg in registrations if isinstance(registrations, list) else []:
                    if not isinstance(reg, dict):
                        continue
                    card = reg.get("card_url")
                    if isinstance(card, str) and card.startswith("http"):
                        sources.append({"url": card, "doc_kind": "grls_card", "title": f"GRLS card: {inn}", "region": "ru"})
                    instr = reg.get("instruction_url")
                    if isinstance(instr, str) and instr.startswith("http"):
                        sources.append({"url": instr, "doc_kind": "ru_instruction", "title": f"RU instruction: {inn}", "region": "ru"})
                links = reg_block.get("links") or {}
                if isinstance(links, dict):
                    deeplink = links.get("grls_deeplink")
                    if isinstance(deeplink, str) and deeplink.startswith("http"):
                        sources.append({"url": deeplink, "doc_kind": "grls_card", "title": f"GRLS card: {inn}", "region": "ru"})
            except Exception as exc:
                logger.info("RU regulatory preflight skipped: %s", exc)

        # RU clinical registry (JSON)
        if "ru_clinical_permission" not in kinds:
            try:
                ru_clin_url = f"{self._gateway_base_url()}/v1/ru/drugs/{quote(inn)}/clinical"
                ru_clin_resp = requests.get(ru_clin_url, timeout=60)
                ru_clin_resp.raise_for_status()
                sources.append(
                    {
                        "url": ru_clin_url,
                        "doc_kind": "ru_clinical_permission",
                        "title": f"RU clinical registry: {inn}",
                        "region": "ru",
                    }
                )
            except Exception as exc:
                logger.info("RU clinical preflight skipped: %s", exc)

        # RU patents (FIPS/registry) JSON
        if "ru_patent_fips" not in kinds:
            try:
                ru_pat_url = f"{self._gateway_base_url()}/v1/ru/drugs/{quote(inn)}/patents"
                ru_pat_resp = requests.get(ru_pat_url, timeout=60)
                ru_pat_resp.raise_for_status()
                sources.append(
                    {
                        "url": ru_pat_url,
                        "doc_kind": "ru_patent_fips",
                        "title": f"RU patents registry: {inn}",
                        "region": "ru",
                    }
                )
            except Exception as exc:
                logger.info("RU patents preflight skipped: %s", exc)

        # Chemistry profile (PubChem) for formula/class when missing in labels
        if "drug_monograph" not in kinds:
            enabled_raw = os.getenv("DDKIT_PUBCHEM_ENABLED", "1").strip().lower()
            pubchem_enabled = enabled_raw not in {"0", "false", "no", "off"}
            if pubchem_enabled:
                pubchem_url = (
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                    f"{quote(inn)}/property/MolecularFormula,CanonicalSMILES,InChIKey/JSON"
                )
                sources.append(
                    {
                        "url": pubchem_url,
                        "doc_kind": "drug_monograph",
                        "title": f"PubChem compound profile: {inn}",
                        "region": "global",
                    }
                )

        # Clinical: attach 1-2 CTGov studies (registry page)
        if not any(k.startswith("ctgov") for k in kinds):
            try:
                clin_url = f"{self._gateway_base_url()}/v1/aggregate/clinical?inn={quote(inn)}&regions=us&limit=2&offset=0"
                r = requests.get(clin_url, timeout=60)
                r.raise_for_status()
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else json.loads(r.text)
                items = body.get("items") or []
                if isinstance(items, list):
                    for item in items[:2]:
                        if not isinstance(item, dict):
                            continue
                        trial_id = str(item.get("trial_id") or "").strip()
                        title = str(item.get("title") or "").strip()
                        links = item.get("links") or {}
                        if not isinstance(links, dict):
                            continue
                        registry = links.get("registry")
                        if not registry:
                            continue
                        label = title or trial_id or "CTGov study"
                        sources.append(
                            {
                                "url": registry,
                                "doc_kind": "ctgov",
                                "title": f"CTGov: {label}",
                                "region": "us",
                            }
                        )
            except Exception as exc:
                logger.info("CTGov preflight skipped: %s", exc)

        if not sources:
            return

        doc_ids = self._attach_sources(case_id=case_id, sources=sources, force=False)
        if doc_ids:
            logger.info("Auto-attached %d sources for case=%s", len(doc_ids), case_id)
            self._wait_for_documents_indexed(doc_ids=doc_ids, deadline=deadline)

    def process_job(self, job_data: Dict[str, Any]) -> bool:
        tenant_id = job_data["tenant_id"]
        case_id = job_data["case_id"]
        job_id = job_data.get("job_id")
        query = job_data.get("query") or job_data.get("inn") or ""
        inn = job_data.get("inn") or ""
        use_web = bool(job_data.get("use_web", True))
        use_snapshot = bool(job_data.get("use_snapshot", True))
        snapshot = job_data.get("snapshot")

        logger.info(f"Processing case_view_generate job: tenant={tenant_id}, case={case_id}")

        try:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_running(job_id)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                deadline = time.time() + settings.job_timeout_seconds
                logger.info("Case view generation timeout set to %ds", settings.job_timeout_seconds)

                if not self._download_case_artifacts(temp_path, tenant_id, case_id):
                    logger.error(f"Failed to download case artifacts for case {case_id}")
                    return False

                documents_dir = temp_path / "databases" / "chunked_reports"
                vector_dir = temp_path / "databases" / "vector_dbs"

                # Preflight: auto-attach a minimal set of authoritative sources (label/CTGov/etc)
                # so regulatory + clinical sections are not empty in snapshot-only cases.
                if use_web and (inn or query):
                    try:
                        self._ensure_min_case_corpus(
                            tenant_id=tenant_id,
                            case_id=case_id,
                            inn=(inn or query),
                            deadline=deadline,
                            snapshot=snapshot if use_snapshot else None,
                        )
                        # Re-download to include newly parsed/indexed docs (chunks + vectors).
                        self._download_case_artifacts(temp_path, tenant_id, case_id)
                    except Exception as exc:
                        logger.warning("Preflight auto-attach failed (continuing): %s", exc)

                # PubMed: search -> rerank -> attach -> parse (text ingest) to enrich the case corpus.
                if use_web and (inn or query):
                    enabled_raw = os.getenv("DDKIT_PUBMED_ENABLED", "1").strip().lower()
                    pubmed_enabled = enabled_raw not in {"0", "false", "no", "off"}
                    if pubmed_enabled:
                        try:
                            retmax = int(os.getenv("DDKIT_PUBMED_RETMAX", "80"))
                        except ValueError:
                            retmax = 80
                        try:
                            top_n = int(os.getenv("DDKIT_PUBMED_ATTACH_TOP_N", "12"))
                        except ValueError:
                            top_n = 12
                        top_n = max(0, min(top_n, 30))
                        if top_n > 0:
                            try:
                                ingestor = PubMedIngestor(storage=self.storage_client, db=self.ddkit_db)
                                pubmed_res, _ = ingestor.ingest_for_inn(
                                    tenant_id=tenant_id,
                                    case_id=case_id,
                                    inn=(inn or query),
                                    documents_dir=documents_dir,
                                    vector_dir=vector_dir,
                                    retmax=retmax,
                                    attach_top_n=top_n,
                                    deadline=deadline,
                                )
                                logger.info(
                                    "PubMed ingestion done: attached=%d skipped=%d pmids=%d",
                                    pubmed_res.attached_docs,
                                    pubmed_res.skipped_existing,
                                    pubmed_res.fetched_pmids,
                                )
                            except Exception as exc:
                                logger.warning("PubMed ingestion failed (skipping): %s", exc)

                output_path = temp_path / "case_view_v2.json"
                generator = CaseViewV2Generator(
                    documents_dir=documents_dir,
                    vector_db_dir=vector_dir,
                    tenant_id=tenant_id,
                    case_id=case_id
                )
                try:
                    payload = generator.generate_case_view(
                        snapshot=snapshot if use_snapshot else None,
                        query=query,
                        inn=inn,
                        use_web=use_web,
                        use_snapshot=use_snapshot,
                        deadline=deadline
                    )
                except TimeoutError as err:
                    logger.error(f"Case view generation timed out: {err}")
                    if job_id and self.ddkit_db.is_configured():
                        self.ddkit_db.mark_job_failed(job_id, str(err))
                    return False

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                s3_key = f"tenants/{tenant_id}/cases/{case_id}/case-view/case_view_v2.json"
                if not self.storage_client.upload_file(s3_key, output_path):
                    logger.error(f"Failed to upload case view for case {case_id}")
                    return False

                if self.ddkit_db.is_configured():
                    self.ddkit_db.upsert_case_view(
                        case_id=case_id,
                        tenant_id=tenant_id,
                        inn=inn,
                        payload=payload,
                        schema_version=payload.get("schema_version", "2.0"),
                        source_stats=payload.get("source_stats")
                    )

                if job_id and self.ddkit_db.is_configured():
                    self.ddkit_db.mark_job_succeeded(job_id)

                job_data["artifacts"] = {"s3_case_view_key": s3_key}
                job_data["status"] = "succeeded"
                logger.info(f"Successfully generated case view for case {case_id}")
                return True

        except Exception as e:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_failed(job_id, str(e))
            job_data["status"] = "failed"
            logger.error(f"Error processing case_view_generate job for case {case_id}: {e}")
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
