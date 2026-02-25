import json
import logging
import os
import uuid
from typing import Optional, Tuple, Any, List, Dict

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

    def upsert_document_by_source_url(
        self,
        tenant_id: str,
        case_id: str,
        doc_kind: str,
        title: str,
        source_type: str,
        source_url: Optional[str],
        status: str = "created",
        language: Optional[str] = None,
    ) -> Tuple[Optional[str], bool]:
        """
        Insert (or update) a document identified by (tenant_id, case_id, source_url).

        Returns (doc_id, is_duplicate).
        """
        if not self.dsn:
            return None, False
        try:
            # We rely on the unique index documents_unique_source_url
            #   ON documents (tenant_id, case_id, source_url) WHERE source_url IS NOT NULL;
            #
            # Postgres supports inference with the WHERE clause to target partial indexes.
            query = """
            INSERT INTO documents (id, tenant_id, case_id, doc_kind, title, source_type, source_url, status, language, created_at, updated_at)
            VALUES (%s::uuid, %s, %s::uuid, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (tenant_id, case_id, source_url) WHERE source_url IS NOT NULL DO UPDATE SET
                doc_kind = EXCLUDED.doc_kind,
                title = EXCLUDED.title,
                source_type = EXCLUDED.source_type,
                status = EXCLUDED.status,
                language = COALESCE(EXCLUDED.language, documents.language),
                updated_at = NOW()
            RETURNING id::text, (xmax <> 0) as is_duplicate
            """
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            str(uuid.uuid4()),
                            tenant_id,
                            case_id,
                            doc_kind,
                            title,
                            source_type,
                            source_url,
                            status,
                            language,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            if not row:
                return None, False
            doc_id, is_dup = row[0], bool(row[1])
            return str(doc_id), bool(is_dup)
        except Exception as exc:
            logger.warning("DDKit DB upsert_document_by_source_url failed: %s", exc)
            return None, False

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

    def upsert_case_view(self, case_id: str, tenant_id: str, inn: str,
                         payload: dict, schema_version: str,
                         source_stats: Optional[dict] = None) -> None:
        payload_json = json.dumps(payload, ensure_ascii=False)
        source_stats_json = json.dumps(source_stats or {}, ensure_ascii=False)
        self._exec(
            """
            INSERT INTO case_views (case_id, tenant_id, inn, payload, schema_version, source_stats, created_at, updated_at)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, NOW(), NOW())
            ON CONFLICT (case_id) DO UPDATE SET
                inn = EXCLUDED.inn,
                payload = EXCLUDED.payload,
                schema_version = EXCLUDED.schema_version,
                source_stats = EXCLUDED.source_stats,
                updated_at = NOW()
            """,
            (case_id, tenant_id, inn, payload_json, schema_version, source_stats_json)
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

    def _query_one(self, query: str, params: tuple) -> Optional[tuple[Any, ...]]:
        if not self.dsn:
            return None
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    row = cur.fetchone()
                conn.commit()
            return row
        except Exception as exc:
            logger.warning(f"DDKit DB query failed: {exc}")
            return None

    def _query_all(self, query: str, params: tuple) -> List[tuple[Any, ...]]:
        if not self.dsn:
            return []
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall() or []
                conn.commit()
            return list(rows)
        except Exception as exc:
            logger.warning(f"DDKit DB query failed: {exc}")
            return []

    def list_case_documents(self, tenant_id: str, case_id: str) -> List[Dict[str, Any]]:
        """
        Return lightweight document rows for a case. Used by generators/e2e to decide what to auto-attach
        and to wait for doc_parse_index completion.
        """
        rows = self._query_all(
            """
            SELECT
                id::text,
                doc_kind,
                status,
                title,
                source_url,
                s3_rendered_pdf_key,
                s3_parsed_json_key,
                updated_at
            FROM documents
            WHERE tenant_id=%s AND case_id=%s::uuid
            ORDER BY created_at DESC
            """,
            (tenant_id, case_id),
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "doc_id": r[0],
                    "doc_kind": r[1],
                    "status": r[2],
                    "title": r[3],
                    "source_url": r[4],
                    "s3_rendered_pdf_key": r[5],
                    "s3_parsed_json_key": r[6],
                    "updated_at": r[7].isoformat() if hasattr(r[7], "isoformat") else r[7],
                }
            )
        return out

    def get_case_inn(self, tenant_id: str, case_id: str) -> Optional[str]:
        """Return the INN (drug name) stored in case_views for this case, or None."""
        row = self._query_one(
            "SELECT inn FROM case_views WHERE tenant_id=%s AND case_id=%s::uuid",
            (tenant_id, case_id),
        )
        if row and row[0]:
            return str(row[0]).strip() or None
        return None

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self._query_one(
            """
            SELECT
                id::text,
                tenant_id,
                case_id::text,
                doc_kind,
                status,
                title,
                source_url,
                s3_rendered_pdf_key,
                s3_parsed_json_key,
                error_message,
                updated_at
            FROM documents
            WHERE id=%s::uuid
            """,
            (doc_id,),
        )
        if not row:
            return None
        return {
            "doc_id": row[0],
            "tenant_id": row[1],
            "case_id": row[2],
            "doc_kind": row[3],
            "status": row[4],
            "title": row[5],
            "source_url": row[6],
            "s3_rendered_pdf_key": row[7],
            "s3_parsed_json_key": row[8],
            "error_message": row[9],
            "updated_at": row[10].isoformat() if hasattr(row[10], "isoformat") else row[10],
        }
