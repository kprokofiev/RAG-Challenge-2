import json
import logging
import time
from typing import Optional, Dict, Any
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

    def _get_queue_name(self, job_type: str) -> str:
        """Get the appropriate queue name for job type."""
        if job_type == "doc_parse_index":
            return self.doc_parse_index_queue
        elif job_type == "report_generate":
            return self.report_generate_queue
        elif job_type == "case_view_generate":
            return self.case_view_generate_queue
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    def enqueue_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Enqueue a job to the appropriate queue.

        Args:
            job_data: Job data dictionary with 'job_type' key

        Returns:
            bool: True if successfully enqueued
        """
        try:
            queue_name = self._get_queue_name(job_data["job_type"])
            job_json = json.dumps(job_data)

            self.redis_client.lpush(queue_name, job_json)
            logger.info(f"Enqueued job {job_data.get('job_type')} to queue {queue_name}")
            return True

        except (RedisError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to enqueue job: {e}")
            return False

    def dequeue_job(self, job_type: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Dequeue a job from the specified queue with timeout.

        Args:
            job_type: Type of job to dequeue
            timeout: Timeout in seconds to wait for job

        Returns:
            Job data dict or None if timeout/no job
        """
        try:
            queue_name = self._get_queue_name(job_type)

            # BRPOP returns (queue_name, job_data) or None on timeout
            result = self.redis_client.brpop([queue_name], timeout)

            if result is None:
                return None

            _, job_json = result
            job_data = json.loads(job_json)

            logger.info(f"Dequeued job {job_data.get('job_type')} from queue {queue_name}")
            return job_data

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to dequeue job: {e}")
            return None

    def get_queue_length(self, job_type: str) -> int:
        """Get the length of a job queue."""
        try:
            queue_name = self._get_queue_name(job_type)
            return self.redis_client.llen(queue_name)
        except RedisError as e:
            logger.error(f"Failed to get queue length for {job_type}: {e}")
            return 0

    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            self.redis_client.ping()
            return True
        except RedisError:
            return False


