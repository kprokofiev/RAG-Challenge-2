import logging
import os
from typing import Optional

import psycopg

logger = logging.getLogger(__name__)


class DDKitDB:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("DDKIT_DB_DSN")

    def is_configured(self) -> bool:
        return bool(self.dsn)

    def mark_job_running(self, job_id: str) -> None:
        self._exec(
            "UPDATE document_jobs SET status='running', started_at=NOW(), attempt=attempt+1, updated_at=NOW() WHERE id=%s",
            (job_id,)
        )

    def mark_job_succeeded(self, job_id: str) -> None:
        self._exec(
            "UPDATE document_jobs SET status='succeeded', finished_at=NOW(), updated_at=NOW() WHERE id=%s",
            (job_id,)
        )

    def mark_job_failed(self, job_id: str, error_message: str) -> None:
        self._exec(
            "UPDATE document_jobs SET status='failed', error_message=%s, finished_at=NOW(), updated_at=NOW() WHERE id=%s",
            (error_message, job_id)
        )

    def update_document_parsed(self, doc_id: str, s3_parsed_json_key: str, pages_count: Optional[int] = None) -> None:
        self._exec(
            "UPDATE documents SET status='parsed', s3_parsed_json_key=%s, pages_count=%s, updated_at=NOW() WHERE id=%s",
            (s3_parsed_json_key, pages_count, doc_id)
        )

    def update_report_published(self, report_id: str, tenant_id: str, case_id: str, title: str,
                                 s3_report_json_key: str, sections_plan_key: Optional[str] = None) -> None:
        self._exec(
            """
            INSERT INTO reports (id, tenant_id, case_id, title, status, sections_plan_key, s3_report_json_key, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'published', %s, %s, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                status='published',
                s3_report_json_key=EXCLUDED.s3_report_json_key,
                updated_at=NOW()
            """,
            (report_id, tenant_id, case_id, title, sections_plan_key, s3_report_json_key)
        )

    def update_report_failed(self, report_id: str, tenant_id: str, case_id: str, error_message: str) -> None:
        self._exec(
            """
            INSERT INTO reports (id, tenant_id, case_id, title, status, error_message, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'failed', %s, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                status='failed',
                error_message=EXCLUDED.error_message,
                updated_at=NOW()
            """,
            (report_id, tenant_id, case_id, f"DD Report — {case_id}", error_message)
        )

    def _exec(self, query: str, params: tuple) -> None:
        if not self.dsn:
            return
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
        except Exception as exc:
            logger.warning(f"DDKit DB update failed: {exc}")
