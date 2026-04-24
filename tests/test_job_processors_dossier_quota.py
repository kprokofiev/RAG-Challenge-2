import importlib
import os
import unittest
from unittest import mock


_TEST_ENV = {
    "OPENAI_API_KEY": "test-key",
    "STORAGE_ENDPOINT_URL": "http://localhost:9000",
    "STORAGE_ACCESS_KEY": "test-access",
    "STORAGE_SECRET_KEY": "test-secret",
    "REDIS_URL": "redis://localhost:6379/0",
}

with mock.patch.dict(os.environ, _TEST_ENV, clear=False):
    job_processors = importlib.import_module("src.job_processors")


class _FakeDB:
    def __init__(self):
        self.failed = []

    def is_configured(self):
        return True

    def mark_job_failed(self, job_id, error_message):
        self.failed.append((job_id, error_message))


class DossierQuotaAutostopTests(unittest.TestCase):
    def _build_processor(self):
        processor = job_processors.DossierGenerateProcessor.__new__(job_processors.DossierGenerateProcessor)
        processor.storage_client = object()
        processor.ddkit_db = _FakeDB()
        return processor

    def test_process_job_blocks_before_preflight_when_quota_autostop_active(self):
        processor = self._build_processor()
        job_data = {
            "tenant_id": "tenant-1",
            "case_id": "case-1",
            "report_id": "report-1",
            "job_id": "job-1",
            "job_type": "dossier_generate",
            "attempt": 0,
        }
        quota_state = {
            "enabled": True,
            "blocked": True,
            "budget_date_utc": "2026-04-24",
            "reset_at_utc": "2026-04-25T00:00:00Z",
            "redis_key": "openai:model-router:v2:2026-04-24",
            "exhausted_tiers": ["mini"],
            "last_errors": {"mini": "429 insufficient_quota"},
            "reason": "exec quota autostop active; exhausted_tiers=mini; reset_at_utc=2026-04-25T00:00:00Z",
        }

        with mock.patch.object(processor, "_quota_autostop_status", return_value=quota_state), mock.patch.object(
            processor,
            "_corpus_ready_for_dossier",
            side_effect=AssertionError("preflight should not run when autostop is blocked"),
        ):
            result = processor.process_job(job_data)

        self.assertFalse(result)
        self.assertEqual(job_data["status"], "quota_autostop_blocked")
        self.assertEqual(job_data["quota_autostop"], quota_state)
        self.assertEqual(
            processor.ddkit_db.failed,
            [("job-1", "exec quota autostop active; exhausted_tiers=mini; reset_at_utc=2026-04-25T00:00:00Z")],
        )


if __name__ == "__main__":
    unittest.main()
