from __future__ import annotations

from typing import Any, Iterable


VERDICT_CONFIRMED = "confirmed"
VERDICT_PARTIAL = "partial"
VERDICT_UNKNOWN = "unknown"


def coerce_evidenced_value(value: Any) -> str:
    """Return a plain string from either EvidencedValue-like dict/object or raw text."""
    if value is None:
        return ""
    if isinstance(value, dict):
        inner = value.get("value")
        return "" if inner is None else str(inner).strip()
    inner = getattr(value, "value", None)
    if inner is not None:
        return str(inner).strip()
    return str(value).strip()


def has_value(value: Any) -> bool:
    return bool(coerce_evidenced_value(value))


def count_present(items: Iterable[Any] | None) -> int:
    if not items:
        return 0
    return sum(1 for item in items if has_value(item))


def registration_status_role(status_value: str | None) -> str:
    """
    Normalize a status string into positive / negative / unknown semantics.

    This is shared by generator, exec, and PDF layers so we do not drift into
    conflicting interpretations of the same registration row.
    """
    s = (status_value or "").strip().lower()
    if not s:
        return "unknown"

    negative_markers = (
        "no public registration record verified",
        "no public eu registration record verified",
        "no public record verified",
        "no registration record found",
        "not found after lookup",
        "registration status unknown",
        "not registered",
        "not approved",
        "cancelled",
        "canceled",
        "revoked",
        "withdrawn",
        "suspended",
        "expired",
        "не зарегистр",
        "нет публичной записи",
        "не действует",
        "аннулир",
        "отозван",
        "приостанов",
    )
    positive_markers = (
        "approved",
        "approval letter referenced",
        "fda approval",
        "registered",
        "registration active",
        "marketing authorisation granted",
        "marketing authorization granted",
        "authorised",
        "authorized",
        "valid",
        "active",
        "действует",
        "действующ",
        "зарегистрир",
        "разрешен",
        "разрешён",
        "одобрен",
    )

    if any(marker in s for marker in negative_markers):
        return "negative"
    if any(marker in s for marker in positive_markers):
        return "positive"
    return "unknown"


def infer_registration_verdict(
    *,
    status: Any = None,
    mah: Any = None,
    identifiers: Iterable[Any] | None = None,
    forms_strengths: Iterable[Any] | None = None,
) -> str:
    """
    Collapse a registration row into one normalized verdict:
    confirmed | partial | unknown
    """
    status_value = coerce_evidenced_value(status)
    role = registration_status_role(status_value)
    if role == "positive":
        return VERDICT_CONFIRMED
    if role == "negative":
        return VERDICT_UNKNOWN

    grounded_fields = 0
    if has_value(mah):
        grounded_fields += 1
    if count_present(identifiers) > 0:
        grounded_fields += 1
    if count_present(forms_strengths) > 0:
        grounded_fields += 1

    if grounded_fields >= 2:
        return VERDICT_PARTIAL
    return VERDICT_UNKNOWN
