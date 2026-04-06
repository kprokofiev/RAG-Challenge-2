import json
import logging
import time
from typing import Optional, Dict, Any, List
import redis
from redis.exceptions import RedisError

from src.settings import settings

logger = logging.getLogger(__name__)


class JobQueueHandler:
    """Redis-based job queue handler for DD Kit RAG worker."""

    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url)
        self.doc_parse_index_queue = settings.queue_doc_parse_index
        self.report_generate_queue = settings.queue_report_generate
        self.case_view_generate_queue = settings.queue_case_view_generate
        self.dossier_generate_queue = settings.queue_dossier_generate

    def _get_queue_name(self, job_type: str) -> str:
        """Get the appropriate queue name for job type."""
        if job_type == "doc_parse_index":
            return self.doc_parse_index_queue
        elif job_type == "report_generate":
            return self.report_generate_queue
        elif job_type == "case_view_generate":
            return self.case_view_generate_queue
        elif job_type == "dossier_generate":
            return self.dossier_generate_queue
        elif job_type == "exec_answer":
            # Sprint 22 WS2: exec_answer reuses dossier_generate queue
            # (P1 will get its own queue if async processing is needed)
            return self.dossier_generate_queue
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    # TTL for dossier dedup lock (30 min). If the job doesn't finish by then,
    # the lock auto-expires so a new run can proceed.
    _DOSSIER_DEDUP_TTL_S = 1800

    def enqueue_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Enqueue a job to the appropriate queue.

        For dossier_generate jobs, acquires a Redis dedup lock first.
        If the same case_id already has a running/queued dossier job,
        the new job is silently skipped (returns True to avoid DLQ).

        Args:
            job_data: Job data dictionary with 'job_type' key

        Returns:
            bool: True if successfully enqueued (or silently deduped)
        """
        try:
            job_type = job_data["job_type"]
            queue_name = self._get_queue_name(job_type)

            # ── Dedup lock for dossier_generate ───────────────────────────
            if job_type == "dossier_generate":
                case_id = job_data.get("case_id", "unknown")
                lock_key = f"ddkit:job_lock:dossier_generate:{case_id}"
                acquired = self.redis_client.set(
                    lock_key, "running", nx=True, ex=self._DOSSIER_DEDUP_TTL_S
                )
                if not acquired:
                    current = (self.redis_client.get(lock_key) or b"").decode()
                    logger.warning(
                        "job_dedup_skipped case=%s lock_state=%s — another dossier_generate is active",
                        case_id, current,
                    )
                    return True  # Silently skip — not an error

            job_json = json.dumps(job_data, default=str)
            self.redis_client.lpush(queue_name, job_json)
            logger.info(f"Enqueued job {job_data.get('job_type')} to queue {queue_name}")
            return True

        except (RedisError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to enqueue job: {e}")
            return False

    def release_dedup_lock(self, job_type: str, case_id: str) -> None:
        """Release the dedup lock for a completed/DLQ'd dossier job."""
        if job_type != "dossier_generate":
            return
        lock_key = f"ddkit:job_lock:dossier_generate:{case_id}"
        try:
            self.redis_client.delete(lock_key)
            logger.debug("dedup_lock_released case=%s", case_id)
        except RedisError as e:
            logger.warning("dedup_lock_release_failed case=%s: %s", case_id, e)

    def dequeue_job(self, job_type: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Dequeue a job using BRPOPLPUSH for at-least-once delivery (#1).

        The job is moved atomically from the source queue to a processing list.
        The caller is responsible for calling ack_job() on success, or the job
        will be reclaimed by the watchdog after DDKIT_QUEUE_VISIBILITY_TIMEOUT_S.

        Args:
            job_type: Type of job to dequeue
            timeout: Timeout in seconds to wait for job

        Returns:
            Job data dict (with _raw_payload and _processing_list set) or None
        """
        try:
            queue_name = self._get_queue_name(job_type)
            processing_list = queue_name + ":processing"

            # BRPOPLPUSH: atomically move from queue to processing list
            job_json = self.redis_client.brpoplpush(queue_name, processing_list, timeout)

            if job_json is None:
                return None

            job_data = json.loads(job_json)
            # Stamp acquisition time inside the processing-list payload so the
            # watchdog can reclaim jobs that were abandoned after a worker crash.
            job_data["_acquired_at"] = int(time.time())
            stamped_json = json.dumps(job_data, default=str)
            try:
                pipe = self.redis_client.pipeline()
                pipe.lrem(processing_list, 1, job_json)
                pipe.lpush(processing_list, stamped_json)
                pipe.execute()
            except RedisError as exc:
                logger.warning(
                    "Failed to stamp processing payload with _acquired_at for %s: %s",
                    queue_name, exc,
                )
                stamped_json = job_json

            job_data["_raw_payload"] = stamped_json
            job_data["_processing_list"] = processing_list

            logger.info(
                "dequeue_job job_type=%s queue=%s trace=%s",
                job_data.get("job_type"), queue_name, job_data.get("trace_id"),
            )
            return job_data

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to dequeue job: {e}")
            return None

    def ack_job(self, job_data: Dict[str, Any]) -> None:
        """
        Acknowledge successful job processing by removing it from the processing list (#1).

        Args:
            job_data: Job data dict returned by dequeue_job (must have _raw_payload/_processing_list)
        """
        raw_payload = job_data.get("_raw_payload")
        processing_list = job_data.get("_processing_list")
        if not raw_payload or not processing_list:
            return
        try:
            self.redis_client.lrem(processing_list, 1, raw_payload)
            logger.debug("ack_job processing_list=%s", processing_list)
        except RedisError as e:
            logger.warning("ack_job_failed: %s", e)

    def get_queue_length(self, job_type: str) -> int:
        """Get the length of a job queue."""
        try:
            queue_name = self._get_queue_name(job_type)
            return self.redis_client.llen(queue_name)
        except RedisError as e:
            logger.error(f"Failed to get queue length for {job_type}: {e}")
            return 0

    def watchdog_tick(self, queue_names: List[str]) -> int:
        """
        Reclaim stale jobs from processing lists back to their source queues.

        Algorithm (same as Node doc-worker watchdog):
          For each queue:
            1. Try to acquire a distributed lock (SET NX EX) so only one
               worker instance runs reclaim at a time.
            2. Scan the :processing list for items whose _acquired_at is
               older than DDKIT_QUEUE_VISIBILITY_TIMEOUT_S.
            3. For each stale item:
               - If reclaimed_count >= DDKIT_WATCHDOG_MAX_RECLAIMS → DLQ.
               - Otherwise LREM processing_list 1 item, LPUSH source_queue item
                 (with reclaimed_count incremented).
            4. Release the lock.

        Returns the total number of jobs reclaimed across all queues.
        """
        visibility_s = settings.queue_visibility_timeout_s
        max_reclaims = settings.watchdog_max_reclaims
        now = int(time.time())
        total_reclaimed = 0

        for queue_name in queue_names:
            processing_list = queue_name + ":processing"
            lock_key = f"ddkit:watchdog:{processing_list}"
            lock_ttl = settings.watchdog_interval_s * 2

            # Distributed lock: only one worker reclaims per interval
            try:
                acquired = self.redis_client.set(lock_key, "1", nx=True, ex=lock_ttl)
            except RedisError as e:
                logger.warning("watchdog_lock_failed queue=%s: %s", queue_name, e)
                continue
            if not acquired:
                logger.debug("watchdog_lock_held queue=%s — skipping", queue_name)
                continue

            try:
                items: List[bytes] = self.redis_client.lrange(processing_list, 0, -1)
            except RedisError as e:
                logger.warning("watchdog_lrange_failed queue=%s: %s", queue_name, e)
                self.redis_client.delete(lock_key)
                continue

            reclaimed = 0
            for item in items:
                try:
                    item_str = item.decode() if isinstance(item, bytes) else item
                    job_data = json.loads(item_str)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue  # malformed — leave for human inspection

                acquired_at = job_data.get("_acquired_at")
                if acquired_at is None:
                    continue
                age_s = now - int(acquired_at)
                if age_s <= visibility_s:
                    continue

                # Item is stale — decide: reclaim or DLQ
                reclaim_count = int(job_data.get("_reclaimed_count", 0)) + 1
                job_id = job_data.get("job_id") or job_data.get("doc_id", "?")
                job_type = job_data.get("job_type", "unknown")

                try:
                    removed = self.redis_client.lrem(processing_list, 1, item_str)
                except RedisError as e:
                    logger.warning("watchdog_lrem_failed job=%s: %s", job_id, e)
                    continue

                if removed == 0:
                    # Already acked by the worker — nothing to do
                    continue

                if reclaim_count > max_reclaims:
                    # Too many reclaims → DLQ
                    dlq_key = f"ddkit:dlq:{job_type}"
                    dlq_entry = {
                        **job_data,
                        "dlq_reason": "watchdog_max_reclaims_exceeded",
                        "dlq_at": now,
                        "_reclaimed_count": reclaim_count,
                    }
                    try:
                        self.redis_client.lpush(dlq_key, json.dumps(dlq_entry))
                    except RedisError as e:
                        logger.error("watchdog_dlq_push_failed job=%s: %s", job_id, e)
                    logger.error(
                        "queue_watchdog_dlq job=%s job_type=%s queue=%s reclaim_count=%d age_s=%d",
                        job_id, job_type, queue_name, reclaim_count, age_s,
                    )
                else:
                    # Re-enqueue with incremented reclaim counter
                    job_data["_reclaimed_count"] = reclaim_count
                    job_data.pop("_raw_payload", None)
                    job_data.pop("_processing_list", None)
                    try:
                        self.redis_client.lpush(queue_name, json.dumps(job_data))
                    except RedisError as e:
                        logger.error("watchdog_requeue_failed job=%s: %s", job_id, e)
                        continue
                    logger.warning(
                        "queue_watchdog_reclaimed job=%s job_type=%s queue=%s "
                        "age_s=%d reclaim_count=%d",
                        job_id, job_type, queue_name, age_s, reclaim_count,
                    )
                    reclaimed += 1

            if reclaimed > 0:
                logger.info(
                    "queue_watchdog_done queue=%s reclaimed=%d", queue_name, reclaimed
                )
            total_reclaimed += reclaimed

            try:
                self.redis_client.delete(lock_key)
            except RedisError:
                pass  # TTL will expire naturally

        return total_reclaimed

    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            self.redis_client.ping()
            return True
        except RedisError:
            return False


