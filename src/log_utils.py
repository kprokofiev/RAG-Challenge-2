"""Structured logging utilities for DDKit RAG workers (Sprint 8).

Provides a thin wrapper around Python's standard logging that injects
correlation fields (tenant_id, case_id, run_id, job_id, trace_id, stage)
into every log record as JSON-formatted extra fields.

Usage:
    from src.log_utils import get_bound_logger

    log = get_bound_logger(case_id="abc", run_id="xyz", stage="parse_index")
    log.info("Processing document", doc_id="d001", doc_kind="smpc")
    log.error("Failed to embed", error="timeout", attempt=2)

The output format (when LOG_FORMAT=json or LOG_FORMAT=structured) is:
    {"level":"INFO","ts":"2026-03-06T12:00:00Z","stage":"parse_index",
     "case_id":"abc","run_id":"xyz","msg":"Processing document","doc_id":"d001"}

When LOG_FORMAT is anything else (default), falls back to human-readable
key=value format suitable for docker logs / stdout tailing.
"""

import json
import logging
import os
import time
from typing import Any, Optional


# Detect desired format from env (json | kv, default kv)
_LOG_FORMAT = os.getenv("DDKIT_LOG_FORMAT", "kv").lower()


class BoundLogger:
    """A logger pre-bound with correlation context fields.

    Immutable: every call to ``bind`` returns a new BoundLogger.
    """

    def __init__(self, name: str, fields: dict):
        self._logger = logging.getLogger(name)
        self._fields = fields

    def bind(self, **kwargs) -> "BoundLogger":
        """Return a new BoundLogger with additional context fields merged."""
        merged = {**self._fields, **kwargs}
        return BoundLogger(self._logger.name, merged)

    def _emit(self, level: int, msg: str, **kwargs):
        ctx = {**self._fields, **kwargs}
        if _LOG_FORMAT == "json":
            record_dict = {
                "level": logging.getLevelName(level),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "msg": msg,
                **ctx,
            }
            self._logger.log(level, json.dumps(record_dict, default=str))
        else:
            # key=value format
            parts = [msg]
            for k, v in ctx.items():
                parts.append(f"{k}={v!r}")
            self._logger.log(level, " ".join(parts))

    def debug(self, msg: str, **kwargs):
        self._emit(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._emit(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._emit(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._emit(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._emit(logging.CRITICAL, msg, **kwargs)


def get_bound_logger(
    name: str = "ddkit",
    *,
    tenant_id: Optional[str] = None,
    case_id: Optional[str] = None,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    stage: Optional[str] = None,
    **extra: Any,
) -> BoundLogger:
    """Create a BoundLogger pre-populated with correlation context.

    All None fields are omitted from the log output to keep records concise.
    """
    fields = {}
    if tenant_id:
        fields["tenant_id"] = tenant_id
    if case_id:
        fields["case_id"] = case_id
    if run_id:
        fields["run_id"] = run_id
    if job_id:
        fields["job_id"] = job_id
    if trace_id:
        fields["trace_id"] = trace_id
    if stage:
        fields["stage"] = stage
    fields.update(extra)
    return BoundLogger(name, fields)
