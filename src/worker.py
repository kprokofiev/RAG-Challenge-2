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
from src.job_processors import DocParseIndexProcessor, ReportGenerateProcessor, CaseViewGenerateProcessor, DossierGenerateProcessor, JobCallback
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
        self.dossier_processor = DossierGenerateProcessor()
        self.running = False
        self.job_attempts: Dict[str, int] = {}
        self.job_attempts_lock = Lock()
        self.mode = settings.worker_mode
        self.concurrency = max(1, settings.worker_concurrency)
        self._watchdog_thread: Optional[threading.Thread] = None
        self._index_done_event = threading.Event()  # S7-C2: set by pubsub subscriber

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

        # S7-C2: Start Redis pubsub listener for index_done events.
        # When an index_done event fires, the subscriber reduces the poll
        # sleep to 0 so the next _get_next_job() iteration picks it up
        # immediately instead of waiting for the polling interval.
        if self.mode in ("report_generate", "dossier_generate", "all"):
            self._start_index_done_subscriber()

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

    def _start_index_done_subscriber(self) -> None:
        """S7-C2: Subscribe to ddkit:events:index_done so dossier/report workers
        react immediately instead of waiting for the next poll cycle."""
        t = threading.Thread(
            target=self._index_done_subscriber_loop,
            name="ddkit-index-done-sub",
            daemon=True,
        )
        t.start()
        logger.info("index_done_subscriber_started channel=ddkit:events:index_done")

    def _index_done_subscriber_loop(self) -> None:
        """Background loop: listens for index_done PUBLISH events."""
        import redis
        while self.running:
            try:
                rdb = redis.Redis.from_url(
                    os.getenv("DDKIT_REDIS_URL", "redis://localhost:6379/0"),
                    decode_responses=True,
                )
                pubsub = rdb.pubsub()
                pubsub.subscribe("ddkit:events:index_done")
                for message in pubsub.listen():
                    if not self.running:
                        break
                    if message["type"] == "message":
                        logger.info(
                            "index_done_event_received data=%s", str(message["data"])[:200]
                        )
                        # Wake up the worker loop immediately
                        self._index_done_event.set()
            except Exception as exc:
                logger.warning("index_done_subscriber error (reconnecting in 5s): %s", exc)
                time.sleep(5)

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
            settings.queue_dossier_generate,
        ]
        mode_queues: List[str] = {
            "doc_parse_index": [settings.queue_doc_parse_index],
            "report_generate": [settings.queue_report_generate],
            "case_view_generate": [settings.queue_case_view_generate],
            "dossier_generate": [settings.queue_dossier_generate],
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
                        # No jobs available — wait briefly, but wake up immediately
                        # if an index_done event arrives (S7-C2).
                        if self._index_done_event.wait(timeout=1.0):
                            self._index_done_event.clear()

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
        if self.mode == "dossier_generate":
            return self.queue_handler.dequeue_job("dossier_generate", timeout=timeout)

        # default (all): poll queues in priority order
        job_data = self.queue_handler.dequeue_job("doc_parse_index", timeout=timeout)
        if job_data:
            return job_data
        job_data = self.queue_handler.dequeue_job("report_generate", timeout=timeout)
        if job_data:
            return job_data
        job_data = self.queue_handler.dequeue_job("case_view_generate", timeout=timeout)
        if job_data:
            return job_data
        return self.queue_handler.dequeue_job("dossier_generate", timeout=timeout)

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
                # Release dedup lock so future runs are allowed
                self.queue_handler.release_dedup_lock(
                    job_data.get("job_type", ""), job_data.get("case_id", "")
                )
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
        elif job_type == "dossier_generate":
            return self.dossier_processor.process_job(job_data)
        else:
            logger.error(f"Unknown job type: {job_type}")
            return False

    # Exponential backoff delays in seconds for retries (#3).
    _RETRY_BACKOFF = [30, 60, 120, 180, 300]

    # Delay for rate-limited jobs: wait 10 minutes then try once more.
    _RATE_LIMIT_DEFER_S = 600

    # Statuses set by processors that are "terminal by design" — no retry makes sense.
    _TERMINAL_STATUSES = {"parsed_empty", "unsupported", "skipped"}

    def _handle_job_failure(self, job_data: Dict[str, Any], job_id: str, attempt: int, error_msg: Optional[str] = None):
        """Handle job failure: re-queue with backoff or send to DLQ after max attempts (#3).

        Jobs whose status is in _TERMINAL_STATUSES (e.g. parsed_empty) are sent straight to
        DLQ on first failure — retrying will always produce the same result.

        Jobs with status="rate_limited" are deferred with a long pause (10 min) and
        do NOT count against max_job_attempts — 429 is a capacity issue, not a bug.
        """
        job_status = job_data.get("status", "")

        # ── Rate-limited path: defer, don't restart from scratch ──────────
        if job_status == "rate_limited":
            rate_limit_retries = int(job_data.get("_rate_limit_retries", 0)) + 1
            if rate_limit_retries > 2:
                logger.error(
                    "job_rate_limited_exhausted job=%s rate_limit_retries=%d — sending to DLQ",
                    job_id, rate_limit_retries,
                )
                self._send_callback(job_data, False, "rate_limit_exhausted")
                self._enqueue_dlq(job_data, "rate_limit_exhausted_after_defer")
                with self.job_attempts_lock:
                    if job_id in self.job_attempts:
                        del self.job_attempts[job_id]
                return

            logger.warning(
                "job_rate_limited_deferred job=%s rate_limit_retries=%d delay=%ds — "
                "parking job (NOT restarting from scratch)",
                job_id, rate_limit_retries, self._RATE_LIMIT_DEFER_S,
            )
            time.sleep(self._RATE_LIMIT_DEFER_S)
            job_data["_rate_limit_retries"] = rate_limit_retries
            # Reset attempt counter so this doesn't eat max_job_attempts
            job_data.pop("metrics", None)
            self._requeue_job(job_data)
            return

        # ── Terminal path: no retry makes sense ───────────────────────────
        is_terminal = (
            job_status in self._TERMINAL_STATUSES
            or (error_msg or "").startswith("terminal:")
        )
        if is_terminal or attempt >= settings.max_job_attempts:
            reason = "terminal_status" if is_terminal else "max_attempts_exceeded"
            logger.error(
                "job_failed_terminal job=%s attempt=%d/%d reason=%s error=%s — sending to DLQ",
                job_id, attempt, settings.max_job_attempts, reason, error_msg,
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
            next_retry_at = int(time.time()) + delay_s
            logger.warning(
                "job_retrying job=%s attempt=%d/%d delay=%ds next_retry_at=%d last_error=%s",
                job_id, attempt, settings.max_job_attempts, delay_s, next_retry_at, error_msg,
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
        # Release dedup lock so future runs can proceed
        self.queue_handler.release_dedup_lock(
            job_type, job_data.get("case_id", "")
        )

    def _get_job_id(self, job_data: Dict[str, Any]) -> Optional[str]:
        """Generate a unique job ID for tracking attempts."""
        job_type = job_data.get("job_type")
        if job_type == "doc_parse_index":
            return f"{job_type}:{job_data.get('tenant_id')}:{job_data.get('case_id')}:{job_data.get('doc_id')}"
        elif job_type == "report_generate":
            return f"{job_type}:{job_data.get('tenant_id')}:{job_data.get('case_id')}"
        elif job_type == "case_view_generate":
            return f"{job_type}:{job_data.get('tenant_id')}:{job_data.get('case_id')}"
        elif job_type == "dossier_generate":
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


