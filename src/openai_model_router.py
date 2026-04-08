from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover
    redis_lib = None


_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutedModel:
    model: str
    tier: str
    requested_model: Optional[str]
    day_key: str
    redis_key: str


_TIER_SEQUENCE = (
    ("elite", "gpt-5.4", 250_000),
    ("mini", "gpt-5.4-mini", 2_500_000),
    ("nano", "gpt-5.4-nano", None),
)

_TIER_INDEX = {name: idx for idx, (name, _, _) in enumerate(_TIER_SEQUENCE)}
_LEGACY_ROUTED_MODELS = {
    "",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.2",
    "gpt-5.2-2025-12-11",
}
_TIER_ALIAS_TO_INDEX = {
    "gpt-5.4": 0,
    "gpt-5.4-mini": 1,
    "gpt-5.4-nano": 2,
}


def _router_enabled() -> bool:
    raw = (os.getenv("OPENAI_MODEL_ROUTER_ENABLED", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _redis_url() -> Optional[str]:
    for key in ("OPENAI_MODEL_ROUTER_REDIS_URL", "DDKIT_REDIS_URL", "REDIS_URL"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _redis_client():
    if redis_lib is None:
        return None
    url = _redis_url()
    if not url:
        return None
    try:
        return redis_lib.Redis.from_url(url, decode_responses=True)
    except Exception as exc:  # pragma: no cover
        _log.warning("openai_model_router_redis_init_failed: %s", exc)
        return None


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_day_key(now: Optional[dt.datetime] = None) -> str:
    current = now or _utc_now()
    return current.date().isoformat()


def _seconds_until_next_utc_midnight(now: Optional[dt.datetime] = None) -> int:
    current = now or _utc_now()
    next_midnight = (current + dt.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(60, int((next_midnight - current).total_seconds()))


def _redis_state_key(day_key: str) -> str:
    return f"openai:model-router:v1:{day_key}"


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _elite_budget() -> int:
    return _int_env("OPENAI_MODEL_ROUTER_ELITE_DAILY_TOKENS", 250_000)


def _mini_budget() -> int:
    return _int_env("OPENAI_MODEL_ROUTER_MINI_DAILY_TOKENS", 2_500_000)


def _default_state() -> dict[str, int]:
    return {
        "elite_used_tokens": 0,
        "elite_cached_tokens": 0,
        "elite_reasoning_tokens": 0,
        "elite_calls": 0,
        "elite_exhausted": 0,
        "mini_used_tokens": 0,
        "mini_cached_tokens": 0,
        "mini_reasoning_tokens": 0,
        "mini_calls": 0,
        "mini_exhausted": 0,
        "nano_used_tokens": 0,
        "nano_cached_tokens": 0,
        "nano_reasoning_tokens": 0,
        "nano_calls": 0,
    }


def _read_state(day_key: str) -> dict[str, int]:
    state = _default_state()
    client = _redis_client()
    if client is None:
        return state
    try:
        key = _redis_state_key(day_key)
        raw = client.hgetall(key)
        if not raw:
            return state
        for field in state:
            try:
                state[field] = int(raw.get(field, state[field]))
            except (TypeError, ValueError):
                continue
        return state
    except Exception as exc:  # pragma: no cover
        _log.warning("openai_model_router_read_state_failed: %s", exc)
        return state


def _tier_start_index(requested_model: Optional[str]) -> Optional[int]:
    normalized = (requested_model or "").strip().lower()
    if normalized in _TIER_ALIAS_TO_INDEX:
        return _TIER_ALIAS_TO_INDEX[normalized]
    if not normalized or normalized in _LEGACY_ROUTED_MODELS:
        return 0
    if normalized.startswith("gpt-"):
        return 0
    return None


def choose_routed_model(
    requested_model: Optional[str] = None,
    minimum_tier_index: int = 0,
) -> RoutedModel:
    normalized = (requested_model or "").strip()
    day_key = _utc_day_key()
    redis_key = _redis_state_key(day_key)

    if not _router_enabled():
        return RoutedModel(
            model=normalized or "gpt-5.4",
            tier="disabled",
            requested_model=requested_model,
            day_key=day_key,
            redis_key=redis_key,
        )

    start_index = _tier_start_index(requested_model)
    if start_index is None:
        return RoutedModel(
            model=normalized,
            tier="explicit",
            requested_model=requested_model,
            day_key=day_key,
            redis_key=redis_key,
        )

    state = _read_state(day_key)
    start_index = max(start_index, minimum_tier_index)
    budgets = {"elite": _elite_budget(), "mini": _mini_budget()}

    for idx in range(start_index, len(_TIER_SEQUENCE)):
        tier_name, model_name, budget = _TIER_SEQUENCE[idx]
        if budget is None:
            return RoutedModel(
                model=model_name,
                tier=tier_name,
                requested_model=requested_model,
                day_key=day_key,
                redis_key=redis_key,
            )
        used_key = f"{tier_name}_used_tokens"
        exhausted_key = f"{tier_name}_exhausted"
        if state.get(exhausted_key, 0) == 1:
            continue
        if state.get(used_key, 0) >= budgets[tier_name]:
            continue
        return RoutedModel(
            model=model_name,
            tier=tier_name,
            requested_model=requested_model,
            day_key=day_key,
            redis_key=redis_key,
        )

    return RoutedModel(
        model="gpt-5.4-nano",
        tier="nano",
        requested_model=requested_model,
        day_key=day_key,
        redis_key=redis_key,
    )


def extract_usage_metrics(response_or_completion: Any) -> dict[str, Optional[int]]:
    usage = getattr(response_or_completion, "usage", None)
    if usage is None and isinstance(response_or_completion, dict):
        usage = response_or_completion.get("usage")
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens")
        if input_tokens is None:
            input_tokens = usage.get("input_tokens")

        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens")

        total_tokens = usage.get("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        input_details = usage.get("prompt_tokens_details")
        if input_details is None:
            input_details = usage.get("input_tokens_details")

        output_details = usage.get("completion_tokens_details")
        if output_details is None:
            output_details = usage.get("output_tokens_details")

        cached_tokens = (input_details or {}).get("cached_tokens", 0) if isinstance(input_details, dict) else 0
        reasoning_tokens = (
            (output_details or {}).get("reasoning_tokens", 0) if isinstance(output_details, dict) else 0
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens or 0,
            "reasoning_tokens": reasoning_tokens or 0,
        }

    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", None)

    output_tokens = getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", None)

    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    input_details = getattr(usage, "prompt_tokens_details", None)
    if input_details is None:
        input_details = getattr(usage, "input_tokens_details", None)

    output_details = getattr(usage, "completion_tokens_details", None)
    if output_details is None:
        output_details = getattr(usage, "output_tokens_details", None)

    cached_tokens = getattr(input_details, "cached_tokens", 0) if input_details is not None else 0
    reasoning_tokens = (
        getattr(output_details, "reasoning_tokens", 0) if output_details is not None else 0
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens or 0,
        "reasoning_tokens": reasoning_tokens or 0,
    }


def record_usage(routed: RoutedModel, usage: dict[str, Optional[int]]) -> None:
    if routed.tier not in {"elite", "mini", "nano"}:
        return
    client = _redis_client()
    if client is None:
        return
    ttl = _seconds_until_next_utc_midnight()
    try:
        key = routed.redis_key
        pipe = client.pipeline()
        total_tokens = int(usage.get("total_tokens") or 0)
        cached_tokens = int(usage.get("cached_tokens") or 0)
        reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
        pipe.hincrby(key, f"{routed.tier}_used_tokens", total_tokens)
        pipe.hincrby(key, f"{routed.tier}_cached_tokens", cached_tokens)
        pipe.hincrby(key, f"{routed.tier}_reasoning_tokens", reasoning_tokens)
        pipe.hincrby(key, f"{routed.tier}_calls", 1)
        pipe.expire(key, ttl)
        pipe.execute()
    except Exception as exc:  # pragma: no cover
        _log.warning("openai_model_router_record_usage_failed: %s", exc)


def mark_tier_exhausted(routed: RoutedModel, reason: str) -> None:
    if routed.tier not in {"elite", "mini"}:
        return
    client = _redis_client()
    if client is None:
        return
    ttl = _seconds_until_next_utc_midnight()
    try:
        key = routed.redis_key
        client.hset(
            key,
            mapping={
                f"{routed.tier}_exhausted": 1,
                f"{routed.tier}_last_error": reason[:500],
            },
        )
        client.expire(key, ttl)
    except Exception as exc:  # pragma: no cover
        _log.warning("openai_model_router_mark_exhausted_failed: %s", exc)


def is_quota_exhausted_error(exc: Exception) -> bool:
    text = str(exc).lower()
    signals = (
        "insufficient_quota",
        "exceeded your current quota",
        "quota",
        "billing",
        "daily limit",
        "usage limit",
        "free tier",
        "token limit reached",
    )
    return any(signal in text for signal in signals)


def next_tier_index(current_tier: str) -> int:
    return _TIER_INDEX.get(current_tier, 0) + 1
