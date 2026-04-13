"""
Clinical status normalization helpers.

Keeps CTGov/CTIS/LLM-derived study statuses stable across dossier JSON,
exec answers, and PDF rendering.
"""

from __future__ import annotations

import re
from typing import Optional


_STATUS_ALIASES = {
    "recruiting": "RECRUITING",
    "active_not_recruiting": "ACTIVE_NOT_RECRUITING",
    "not_yet_recruiting": "NOT_YET_RECRUITING",
    "enrolling_by_invitation": "ENROLLING_BY_INVITATION",
    "completed": "COMPLETED",
    "terminated": "TERMINATED",
    "withdrawn": "WITHDRAWN",
    "suspended": "SUSPENDED",
    "available": "AVAILABLE",
    "approved_for_marketing": "APPROVED_FOR_MARKETING",
    "unknown": "UNKNOWN",
    "unknown_status": "UNKNOWN",
}

_STATUS_LABELS = {
    "RECRUITING": "Recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
    "COMPLETED": "Completed",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "SUSPENDED": "Suspended",
    "AVAILABLE": "Available",
    "APPROVED_FOR_MARKETING": "Approved for marketing",
    "UNKNOWN": "Unknown",
}


def _status_slug(raw_status: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", raw_status.lower()).strip("_")
    slug = slug.replace("__", "_")
    return slug


def normalize_clinical_status(raw_status: Optional[str]) -> Optional[str]:
    """Return canonical uppercase status token for storage/comparison."""
    if raw_status is None:
        return None
    cleaned = str(raw_status).strip()
    if not cleaned:
        return None
    slug = _status_slug(cleaned)
    if not slug:
        return cleaned
    return _STATUS_ALIASES.get(slug, slug.upper())


def format_clinical_status(raw_status: Optional[str]) -> str:
    """Return human-friendly status label for UI/PDF/exec output."""
    normalized = normalize_clinical_status(raw_status)
    if not normalized:
        return ""
    return _STATUS_LABELS.get(normalized, normalized.replace("_", " ").title())
