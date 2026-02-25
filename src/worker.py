import json
import logging
import os
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, Any, Optional, Set, List

from src.queue_handler import JobQueueHandler
from src.job_processors import DocParseIndexProcessor, ReportGenerateProcessor, CaseViewGenerateProcessor, JobCallback
from src.settings import settings
from src.storage_client import StorageClient

logger = logging.getLogger(__name__)


class DDKitWorker:
    """Main DD Kit RAG worker class."""

    def __init__(self):
        self.queue_handler = JobQueueHandler()
        self.doc_processor = DocParseIndexProcessor()
        self.report_processor = ReportGenerateProcessor()
        self.case_view_processor = CaseViewGenerateProcessor()
        self.running = False
        self.job_attempts: Dict[str, int] = {}
        self.job_attempts_lock = Lock()
        self.mode = settings.worker_mode
        self.concurrency = max(1, settings.worker_concurrency)
        self._watchdog_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the worker loop."""
        logger.info("Starting DD Kit RAG worker")
        if os.getenv("DDKIT_DB_DSN"):
            logger.info("DDKit DB callbacks enabled")
        else:
            logger.warning("DDKit DB callbacks disabled (DDKIT_DB_DSN missing)")
        self.running = True

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start watchdog in a background daemon thread (#1)
        self._start_watchdog()

        try:
            self._worker_loop()
        except Exception as e:
            logger.error(f"Worker loop failed: {e}")
            sys.exit(1)
        finally:
            logger.info("Worker stopped")

    def _start_watchdog(self) -> None:
        """Start the background watchdog thread that reclaims stale jobs (#1)."""
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="ddkit-watchdog",
            daemon=True,  # exits when main thread exits
        )
        self._watchdog_thread.start()
        logger.info(
            "queue_watchdog_started interval_s=%d visibility_s=%d max_reclaims=%d",
            settings.watchdog_interval_s,
            settings.queue_visibility_timeout_s,
            settings.watchdog_max_reclaims,
        )

    def _watchdog_loop(self) -> None:
        """
        Background loop: periodically scan processing lists and reclaim stale jobs.

        Runs every DDKIT_WATCHDOG_INTERVAL_S seconds. Uses a distributed Redis
        lock (SET NX) inside watchdog_tick() so only one worker instance
        performs the reclaim at a time.
        """
        # Queues to watch depend on worker mode
        all_queues = [
            settings.queue_doc_parse_index,
            settings.queue_report_generate,
            settings.queue_case_view_generate,
        ]
        mode_queues: List[str] = {
            "doc_parse_index": [settings.queue_doc_parse_index],
            "report_generate": [settings.queue_report_generate],
            "case_view_generate": [settings.queue_case_view_generate],
        }.get(self.mode, all_queues)

        while self.running:
            time.sleep(settings.watchdog_interval_s)
            if not self.running:
                break
            try:
                reclaimed = self.queue_handler.watchdog_tick(mode_queues)
                if reclaimed:
                    logger.info("queue_watchdog_tick_done reclaimed=%d", reclaimed)
            except Exception as exc:
                logger.error("queue_watchdog_tick_error: %s", exc)

    def stop(self):
        """Stop the worker."""
        logger.info("Stopping worker...")
        self.running = False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def _worker_loop(self):
        """Main worker processing loop."""
        logger.info(
            "Worker loop started (mode=%s, concurrency=%d)",
            self.mode,
            self.concurrency,
        )

        futures: Set = set()
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            while self.running:
                try:
                    # Clean up finished jobs
                    done = {f for f in futures if f.done()}
                    for f in done:
                        futures.remove(f)
                        if exc := f.exception():
                            logger.error(f"Job execution failed: {exc}")

                    if len(futures) >= self.concurrency:
                        time.sleep(0.5)
                        continue

                    # Try to get a job from the selected queue(s)
                    job_data = self._get_next_job()

                    if job_data:
                        futures.add(executor.submit(self._process_job, job_data))
                    else:
                        # No jobs available, sleep briefly
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Error in worker loop: {e}")
                    time.sleep(5)  # Sleep on error

    def _get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get the next job from available queues."""
        timeout = settings.worker_poll_timeout
        if self.mode == "doc_parse_index":
            return self.queue_handler.dequeue_job("doc_parse_index", timeout=timeout)
        if self.mode == "report_generate":
            return self.queue_handler.dequeue_job("report_generate", timeout=timeout)
        if self.mode == "case_view_generate":
            return self.queue_handler.dequeue_job("case_view_generate", timeout=timeout)

        # default: try doc_parse_index first, then report_generate, then case_view_generate
        job_data = self.queue_handler.dequeue_job("doc_parse_index", timeout=timeout)
        if job_data:
            return job_data
        job_data = self.queue_handler.dequeue_job("report_generate", timeout=timeout)
        if job_data:
            return job_data
        return self.queue_handler.dequeue_job("case_view_generate", timeout=timeout)

    def _process_job(self, job_data: Dict[str, Any]):
        """Process a single job with retry logic."""
        job_type = job_data.get("job_type")
        job_id = self._get_job_id(job_data)

        if not job_id:
            logger.error("Job missing required fields for identification")
            return

        # Initialize attempt counter
        with self.job_attempts_lock:
            if job_id not in self.job_attempts:
                self.job_attempts[job_id] = 0
            self.job_attempts[job_id] += 1
            attempt = self.job_attempts[job_id]

        logger.info(
            "job_start job=%s attempt=%d/%d trace=%s run=%s",
            job_id, attempt, settings.max_job_attempts,
            job_data.get("trace_id"), job_data.get("run_id"),
        )

        try:
            success = self._execute_job_processor(job_type, job_data)

            if success:
                logger.info(
                    "job_succeeded job=%s attempt=%d/%d",
                    job_id, attempt, settings.max_job_attempts,
                )
                # ACK: remove from processing list (#1)
                self.queue_handler.ack_job(job_data)
                self._send_callback(job_data, True)
                # Clean up attempt counter on success
                with self.job_attempts_lock:
                    if job_id in self.job_attempts:
                        del self.job_attempts[job_id]
            else:
                logger.error(
                    "job_failed job=%s attempt=%d/%d",
                    job_id, attempt, settings.max_job_attempts,
                )
                # ACK the current dequeue; _handle_job_failure re-enqueues with backoff if retries remain.
                self.queue_handler.ack_job(job_data)
                self._handle_job_failure(job_data, job_id, attempt)

        except Exception as e:
            logger.error("job_exception job=%s attempt=%d: %s", job_id, attempt, e)
            self.queue_handler.ack_job(job_data)
            self._handle_job_failure(job_data, job_id, attempt, str(e))

    def _execute_job_processor(self, job_type: str, job_data: Dict[str, Any]) -> bool:
        """Execute the appropriate job processor."""
        if job_type == "doc_parse_index":
            return self.doc_processor.process_job(job_data)
        elif job_type == "report_generate":
            return self.report_processor.process_job(job_data)
        elif job_type == "case_view_generate":
            return self.case_view_processor.process_job(job_data)
        else:
            logger.error(f"Unknown job type: {job_type}")
            return False

    # Exponential backoff delays in seconds for retries (#3).
    _RETRY_BACKOFF = [30, 60, 120, 180, 300]

    def _handle_job_failure(self, job_data: Dict[str, Any], job_id: str, attempt: int, error_msg: Optional[str] = None):
        """Handle job failure: re-queue with backoff or send to DLQ after max attempts (#3)."""
        if attempt >= settings.max_job_attempts:
            logger.error(
                "job_failed_terminal job=%s attempt=%d/%d error=%s — sending to DLQ",
                job_id, attempt, settings.max_job_attempts, error_msg,
            )
            self._send_callback(job_data, False, error_msg)
            self._enqueue_dlq(job_data, error_msg)
            # Clean up attempt counter
            with self.job_attempts_lock:
                if job_id in self.job_attempts:
                    del self.job_attempts[job_id]
        else:
            # Exponential backoff delay before re-queueing (#3)
            backoff_idx = min(attempt - 1, len(self._RETRY_BACKOFF) - 1)
            delay_s = self._RETRY_BACKOFF[backoff_idx]
            logger.warning(
                "job_retrying job=%s attempt=%d/%d delay=%ds error=%s",
                job_id, attempt, settings.max_job_attempts, delay_s, error_msg,
            )
            time.sleep(delay_s)
            # Update attempt count in job payload so workers can propagate it.
            # Strip runtime-only keys (metrics may contain bytes from docling) before requeue.
            job_data["attempt"] = attempt + 1
            job_data.pop("metrics", None)
            self._requeue_job(job_data)

    def _requeue_job(self, job_data: Dict[str, Any]) -> None:
        """Re-push a failed job back to its source queue for retry (#3)."""
        try:
            queued = self.queue_handler.enqueue_job(job_data)
            if queued:
                logger.info(
                    "job_requeued job_type=%s job_id=%s attempt=%d",
                    job_data.get("job_type"), job_data.get("job_id"), job_data.get("attempt"),
                )
            else:
                logger.error(
                    "job_requeue_failed job_type=%s job_id=%s — falling back to DLQ",
                    job_data.get("job_type"), job_data.get("job_id"),
                )
                self._enqueue_dlq(job_data, "requeue_failed")
        except Exception as exc:
            logger.error("job_requeue_exception job_id=%s: %s", job_data.get("job_id"), exc)

    def _enqueue_dlq(self, job_data: Dict[str, Any], error_msg: Optional[str]) -> None:
        """
        Push a job to the Dead Letter Queue list in Redis (#3).

        DLQ key format: ddkit:dlq:{job_type}
        Payload includes original job + error context.
        """
        job_type = job_data.get("job_type", "unknown")
        dlq_key = f"ddkit:dlq:{job_type}"
        dlq_entry = {
            **job_data,
            "dlq_reason": error_msg or "unknown",
            "dlq_at": int(time.time()),
        }
        try:
            self.queue_handler.redis_client.lpush(dlq_key, json.dumps(dlq_entry))
            logger.info(
                "job_dlq_enqueued job_type=%s job_id=%s queue=%s",
                job_type, job_data.get("job_id"), dlq_key,
            )
        except Exception as exc:
            logger.error("job_dlq_failed job_id=%s: %s", job_data.get("job_id"), exc)

    def _get_job_id(self, job_data: Dict[str, Any]) -> Optional[str]:
        """Generate a unique job ID for tracking attempts."""
        job_type = job_data.get("job_type")
        if job_type == "doc_parse_index":
            return f"{job_type}:{job_data.get('tenant_id')}:{job_data.get('case_id')}:{job_data.get('doc_id')}"
        elif job_type == "report_generate":
            return f"{job_type}:{job_data.get('tenant_id')}:{job_data.get('case_id')}"
        elif job_type == "case_view_generate":
            return f"{job_type}:{job_data.get('tenant_id')}:{job_data.get('case_id')}"
        return None

    def _send_callback(self, job_data: Dict[str, Any], success: bool, error_message: Optional[str] = None):
        """Send job completion callback if configured."""
        if settings.job_callback_url:
            duration_ms = JobCallback.send_callback(settings.job_callback_url, job_data, success, error_message)
            if duration_ms is not None and job_data.get("metrics"):
                metrics = job_data["metrics"]
                metrics.setdefault("stages", {})
                metrics["stages"]["callback_ms"] = duration_ms
                if job_data.get("job_type") == "doc_parse_index":
                    logger.info("doc_parse_index_metrics=%s", json.dumps(metrics, ensure_ascii=False))

    def check_health(self) -> Dict[str, Any]:
        """Perform health check."""
        health_status = {
            "healthy": True,
            "checks": {}
        }

        # Check Redis connection
        try:
            redis_healthy = self.queue_handler.is_healthy()
            health_status["checks"]["redis"] = {"healthy": redis_healthy}
            if not redis_healthy:
                health_status["healthy"] = False
        except Exception as e:
            health_status["checks"]["redis"] = {"healthy": False, "error": str(e)}
            health_status["healthy"] = False

        # Check storage connection
        try:
            storage_client = StorageClient()
            # Try to list bucket (basic connectivity check)
            storage_healthy = True  # Assume healthy if no exception
            health_status["checks"]["storage"] = {"healthy": storage_healthy}
        except Exception as e:
            health_status["checks"]["storage"] = {"healthy": False, "error": str(e)}
            health_status["healthy"] = False

        # Check environment
        try:
            # Validate that required settings are present
            env_healthy = all([
                settings.openai_api_key,
                settings.redis_url,
                settings.storage_endpoint_url,
                settings.storage_access_key,
                settings.storage_secret_key
            ])
            health_status["checks"]["environment"] = {"healthy": env_healthy}
            if not env_healthy:
                health_status["healthy"] = False
        except Exception as e:
            health_status["checks"]["environment"] = {"healthy": False, "error": str(e)}
            health_status["healthy"] = False

        return health_status


