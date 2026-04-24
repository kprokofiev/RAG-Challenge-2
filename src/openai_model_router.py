from __future__ import annotations

import datetime as dt
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import ParseResult, urlparse, urlunparse

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover
    redis_lib = None


_log = logging.getLogger(__name__)
_redis_client_cache: Dict[str, Any] = {}


@dataclass(frozen=True)
class RoutedModel:
    model: str
    tier: str
    requested_model: Optional[str]
    day_key: str
    redis_key: str
    reservation_id: Optional[str] = None
    reserved_tokens: int = 0
    fallback_reason: Optional[str] = None
    budget_snapshot_before: Dict[str, Any] = field(default_factory=dict)
    reset_at_utc: Optional[str] = None
    thinking_mode_requested: Optional[str] = None
    block_class: Optional[str] = None


_TIER_SEQUENCE = (
    ("elite", "gpt-5.4", 250_000),
    ("mini", "gpt-5.4-mini", 2_500_000),
    ("nano", "gpt-5.4-nano", None),
)
_TIER_INDEX = {name: idx for idx, (name, _, _) in enumerate(_TIER_SEQUENCE)}
_TIER_ALIAS_TO_INDEX = {"gpt-5.4": 0, "gpt-5.4-mini": 1, "gpt-5.4-nano": 2}
_LEGACY_ROUTED_MODELS = {
    "",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.2",
    "gpt-5.2-2025-12-11",
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


def _rebuild_netloc(parsed: ParseResult, host: str, port: Optional[int]) -> str:
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port:
        return f"{userinfo}{host}:{port}"
    return f"{userinfo}{host}"


def _host_fallback_redis_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    hostname = (parsed.hostname or "").strip().lower()
    if hostname not in {"redis", "host.docker.internal"}:
        return None
    port = parsed.port or 6379
    rebuilt = parsed._replace(netloc=_rebuild_netloc(parsed, "localhost", port))
    return urlunparse(rebuilt)


def _redis_url_candidates() -> list[str]:
    raw = _redis_url()
    if not raw:
        return []
    candidates: list[str] = []
    for candidate in (
        raw,
        (os.getenv("OPENAI_MODEL_ROUTER_HOST_REDIS_URL") or "").strip(),
        _host_fallback_redis_url(raw),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _redis_client():
    if redis_lib is None:
        return None
    urls = _redis_url_candidates()
    if not urls:
        return None
    last_exc: Optional[Exception] = None
    for url in urls:
        cached = _redis_client_cache.get(url)
        if cached is not None:
            return cached
        try:
            client = redis_lib.Redis.from_url(url, decode_responses=True)
            client.ping()
            _redis_client_cache[url] = client
            return client
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            continue
    if last_exc is not None:
        _log.warning("openai_model_router_redis_init_failed: %s", last_exc)
    return None


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_day_key(now: Optional[dt.datetime] = None) -> str:
    current = now or _utc_now()
    return current.date().isoformat()


def _reset_at_utc(now: Optional[dt.datetime] = None) -> str:
    current = now or _utc_now()
    next_midnight = (current + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return next_midnight.isoformat().replace("+00:00", "Z")


def _seconds_until_next_utc_midnight(now: Optional[dt.datetime] = None) -> int:
    current = now or _utc_now()
    next_midnight = (current + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(60, int((next_midnight - current).total_seconds()))


def _redis_state_key(day_key: str) -> str:
    return f"openai:model-router:v2:{day_key}"


def _reservation_key(redis_key: str, reservation_id: str) -> str:
    return f"{redis_key}:reservation:{reservation_id}"


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _elite_budget() -> int:
    return _int_env("OPENAI_MODEL_ROUTER_ELITE_DAILY_TOKENS", 250_000)


def _mini_budget() -> int:
    return _int_env("OPENAI_MODEL_ROUTER_MINI_DAILY_TOKENS", 2_500_000)


def _reserve_min_for_tier(tier: str) -> int:
    if tier == "elite":
        return _int_env("DDKIT_EXEC_ELITE_RESERVE_MIN", 40_000)
    if tier == "mini":
        return _int_env("DDKIT_EXEC_MINI_RESERVE_MIN", 300_000)
    return 0


def _default_state() -> Dict[str, int]:
    return {
        "elite_used_tokens": 0,
        "elite_reserved_tokens": 0,
        "elite_cached_tokens": 0,
        "elite_reasoning_tokens": 0,
        "elite_calls": 0,
        "elite_exhausted": 0,
        "mini_used_tokens": 0,
        "mini_reserved_tokens": 0,
        "mini_cached_tokens": 0,
        "mini_reasoning_tokens": 0,
        "mini_calls": 0,
        "mini_exhausted": 0,
        "nano_used_tokens": 0,
        "nano_reserved_tokens": 0,
        "nano_cached_tokens": 0,
        "nano_reasoning_tokens": 0,
        "nano_calls": 0,
    }


def _read_raw_state(day_key: str) -> Dict[str, Any]:
    client = _redis_client()
    if client is None:
        return {}
    key = _redis_state_key(day_key)
    try:
        raw = client.hgetall(key)
    except Exception as exc:  # pragma: no cover
        _log.warning("openai_model_router_read_state_failed: %s", exc)
        return {}
    return raw or {}


def _read_state(day_key: str) -> Dict[str, int]:
    state = _default_state()
    raw = _read_raw_state(day_key)
    for field in state:
        try:
            state[field] = int(raw.get(field, state[field]))
        except (TypeError, ValueError):
            continue
    return state


def get_budget_snapshot(day_key: Optional[str] = None) -> Dict[str, Any]:
    budget_day = day_key or _utc_day_key()
    state = _read_state(budget_day)
    budgets = {"elite": _elite_budget(), "mini": _mini_budget(), "nano": None}
    snapshot: Dict[str, Any] = {
        "budget_date_utc": budget_day,
        "reset_at_utc": _reset_at_utc(),
        "budgets": budgets,
    }
    for tier in ("elite", "mini", "nano"):
        used = state.get(f"{tier}_used_tokens", 0)
        reserved = state.get(f"{tier}_reserved_tokens", 0)
        budget = budgets[tier]
        snapshot[tier] = {
            "used_tokens": used,
            "reserved_tokens": reserved,
            "cached_tokens": state.get(f"{tier}_cached_tokens", 0),
            "reasoning_tokens": state.get(f"{tier}_reasoning_tokens", 0),
            "calls": state.get(f"{tier}_calls", 0),
            "exhausted": bool(state.get(f"{tier}_exhausted", 0)) if tier in {"elite", "mini"} else False,
            "remaining_tokens": None if budget is None else max(0, budget - used - reserved),
            "reserve_min_remaining": _reserve_min_for_tier(tier),
        }
    return snapshot


def get_quota_autostop_status(day_key: Optional[str] = None) -> Dict[str, Any]:
    budget_day = day_key or _utc_day_key()
    raw = _read_raw_state(budget_day)
    snapshot = get_budget_snapshot(budget_day)
    exhausted_tiers = [
        tier for tier in ("elite", "mini") if snapshot.get(tier, {}).get("exhausted")
    ]
    enabled = _bool_env("DDKIT_EXEC_QUOTA_AUTOSTOP", False)
    blocked = enabled and bool(exhausted_tiers)
    last_errors = {
        tier: raw.get(f"{tier}_last_error")
        for tier in ("elite", "mini")
        if raw.get(f"{tier}_last_error")
    }
    reason = None
    if blocked:
        reason = (
            "exec quota autostop active; "
            f"exhausted_tiers={','.join(exhausted_tiers)}; "
            f"reset_at_utc={snapshot.get('reset_at_utc')}"
        )
    return {
        "enabled": enabled,
        "blocked": blocked,
        "budget_date_utc": budget_day,
        "reset_at_utc": snapshot.get("reset_at_utc"),
        "redis_key": _redis_state_key(budget_day),
        "exhausted_tiers": exhausted_tiers,
        "last_errors": last_errors,
        "reason": reason,
    }


def _tier_start_index(requested_model: Optional[str]) -> Optional[int]:
    normalized = (requested_model or "").strip().lower()
    if normalized in _TIER_ALIAS_TO_INDEX:
        return _TIER_ALIAS_TO_INDEX[normalized]
    if not normalized or normalized in _LEGACY_ROUTED_MODELS:
        return 0
    if normalized.startswith("gpt-"):
        return 0
    return None


def _can_use_tier(state: Dict[str, int], tier_name: str, budget: Optional[int], estimated_tokens: int) -> bool:
    if budget is None:
        return True
    if tier_name in {"elite", "mini"} and state.get(f"{tier_name}_exhausted", 0) == 1:
        return False
    projected = state.get(f"{tier_name}_used_tokens", 0) + state.get(f"{tier_name}_reserved_tokens", 0) + max(0, estimated_tokens)
    reserve_floor = _reserve_min_for_tier(tier_name)
    return projected <= max(0, budget - reserve_floor)


def _select_tier(
    requested_model: Optional[str],
    minimum_tier_index: int,
    estimated_tokens: int,
    allow_nano_final: bool,
) -> RoutedModel:
    normalized = (requested_model or "").strip()
    day_key = _utc_day_key()
    redis_key = _redis_state_key(day_key)
    snapshot = get_budget_snapshot(day_key)
    reset_at_utc = snapshot.get("reset_at_utc")

    if not _router_enabled():
        return RoutedModel(
            model=normalized or "gpt-5.4",
            tier="disabled",
            requested_model=requested_model,
            day_key=day_key,
            redis_key=redis_key,
            budget_snapshot_before=snapshot,
            reset_at_utc=reset_at_utc,
        )

    start_index = _tier_start_index(requested_model)
    if start_index is None:
        return RoutedModel(
            model=normalized,
            tier="explicit",
            requested_model=requested_model,
            day_key=day_key,
            redis_key=redis_key,
            budget_snapshot_before=snapshot,
            reset_at_utc=reset_at_utc,
        )

    state = _read_state(day_key)
    start_index = max(start_index, minimum_tier_index)
    fallback_reason = None
    for idx in range(start_index, len(_TIER_SEQUENCE)):
        tier_name, model_name, budget = _TIER_SEQUENCE[idx]
        if tier_name == "elite" and state.get("elite_exhausted", 0) == 1:
            fallback_reason = "elite_exhausted"
            continue
        if not _can_use_tier(state, tier_name, budget, estimated_tokens):
            fallback_reason = f"{tier_name}_reserve_shortfall"
            continue
        if tier_name == "nano" and not allow_nano_final:
            fallback_reason = "forced_nano_degrade"
        return RoutedModel(
            model=model_name,
            tier=tier_name,
            requested_model=requested_model,
            day_key=day_key,
            redis_key=redis_key,
            fallback_reason=fallback_reason,
            budget_snapshot_before=snapshot,
            reset_at_utc=reset_at_utc,
        )

    return RoutedModel(
        model="gpt-5.4-nano",
        tier="nano",
        requested_model=requested_model,
        day_key=day_key,
        redis_key=redis_key,
        fallback_reason=fallback_reason or "all_tiers_degraded",
        budget_snapshot_before=snapshot,
        reset_at_utc=reset_at_utc,
    )


def choose_routed_model(requested_model: Optional[str] = None, minimum_tier_index: int = 0) -> RoutedModel:
    return _select_tier(requested_model, minimum_tier_index, estimated_tokens=0, allow_nano_final=True)


def reserve_routed_model(
    requested_model: Optional[str] = None,
    estimated_total_tokens: int = 0,
    minimum_tier_index: int = 0,
    block_class: Optional[str] = None,
    thinking_mode: Optional[str] = None,
    allow_nano_final: bool = True,
) -> RoutedModel:
    routed = _select_tier(requested_model, minimum_tier_index, estimated_total_tokens, allow_nano_final)
    if routed.tier not in {"elite", "mini", "nano"} or estimated_total_tokens <= 0:
        return RoutedModel(
            **{**routed.__dict__, "block_class": block_class, "thinking_mode_requested": thinking_mode}
        )
    client = _redis_client()
    if client is None:
        return RoutedModel(
            **{**routed.__dict__, "block_class": block_class, "thinking_mode_requested": thinking_mode}
        )
    reservation_id = uuid.uuid4().hex
    ttl = _seconds_until_next_utc_midnight()
    try:
        state_key = routed.redis_key
        client.hincrby(state_key, f"{routed.tier}_reserved_tokens", estimated_total_tokens)
        client.expire(state_key, ttl)
        client.hset(
            _reservation_key(state_key, reservation_id),
            mapping={"tier": routed.tier, "tokens": estimated_total_tokens},
        )
        client.expire(_reservation_key(state_key, reservation_id), ttl)
    except Exception as exc:  # pragma: no cover
        _log.warning("openai_model_router_reservation_failed: %s", exc)
        reservation_id = None
        estimated_total_tokens = 0
    return RoutedModel(
        **{
            **routed.__dict__,
            "reservation_id": reservation_id,
            "reserved_tokens": estimated_total_tokens,
            "block_class": block_class,
            "thinking_mode_requested": thinking_mode,
        }
    )


def _release_reservation_internal(routed: RoutedModel) -> None:
    if not routed.reservation_id or routed.tier not in {"elite", "mini", "nano"}:
        return
    client = _redis_client()
    if client is None:
        return
    try:
        res_key = _reservation_key(routed.redis_key, routed.reservation_id)
        raw = client.hgetall(res_key)
        tokens = int(raw.get("tokens", routed.reserved_tokens or 0))
        tier = raw.get("tier", routed.tier)
        if tokens > 0:
            client.hincrby(routed.redis_key, f"{tier}_reserved_tokens", -tokens)
        client.delete(res_key)
    except Exception as exc:  # pragma: no cover
        _log.warning("openai_model_router_release_reservation_failed: %s", exc)


def release_routed_reservation(routed: RoutedModel) -> None:
    _release_reservation_internal(routed)


def extract_usage_metrics(response_or_completion: Any) -> Dict[str, Optional[int]]:
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
        reasoning_tokens = (output_details or {}).get("reasoning_tokens", 0) if isinstance(output_details, dict) else 0
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
    reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) if output_details is not None else 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens or 0,
        "reasoning_tokens": reasoning_tokens or 0,
    }


def commit_routed_usage(routed: RoutedModel, usage: Dict[str, Optional[int]]) -> Dict[str, Any]:
    if routed.tier not in {"elite", "mini", "nano"}:
        return routed.budget_snapshot_before
    client = _redis_client()
    ttl = _seconds_until_next_utc_midnight()
    total_tokens = int(usage.get("total_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
    if client is None:
        return get_budget_snapshot(routed.day_key)
    try:
        if routed.reservation_id:
            _release_reservation_internal(routed)
        pipe = client.pipeline()
        pipe.hincrby(routed.redis_key, f"{routed.tier}_used_tokens", total_tokens)
        pipe.hincrby(routed.redis_key, f"{routed.tier}_cached_tokens", cached_tokens)
        pipe.hincrby(routed.redis_key, f"{routed.tier}_reasoning_tokens", reasoning_tokens)
        pipe.hincrby(routed.redis_key, f"{routed.tier}_calls", 1)
        pipe.expire(routed.redis_key, ttl)
        pipe.execute()
    except Exception as exc:  # pragma: no cover
        _log.warning("openai_model_router_commit_usage_failed: %s", exc)
    return get_budget_snapshot(routed.day_key)


def record_usage(routed: RoutedModel, usage: Dict[str, Optional[int]]) -> None:
    commit_routed_usage(routed, usage)


def build_budget_trace(
    routed: RoutedModel,
    usage_actual: Optional[Dict[str, Any]] = None,
    budget_snapshot_after: Optional[Dict[str, Any]] = None,
    reasoning_effort_actual: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "budget_date_utc": routed.day_key,
        "budget_snapshot_before": routed.budget_snapshot_before,
        "reservation_estimate": {"tokens": routed.reserved_tokens, "reservation_id": routed.reservation_id},
        "model_requested": routed.requested_model,
        "model_selected": routed.model,
        "fallback_reason": routed.fallback_reason,
        "thinking_mode_requested": routed.thinking_mode_requested,
        "reasoning_effort_actual": reasoning_effort_actual,
        "usage_actual": usage_actual or {},
        "budget_snapshot_after": budget_snapshot_after or {},
        "reset_at_utc": routed.reset_at_utc,
    }


def mark_tier_exhausted(routed: RoutedModel, reason: str) -> None:
    if routed.tier not in {"elite", "mini"}:
        return
    client = _redis_client()
    if client is None:
        return
    ttl = _seconds_until_next_utc_midnight()
    try:
        client.hset(
            routed.redis_key,
            mapping={f"{routed.tier}_exhausted": 1, f"{routed.tier}_last_error": reason[:500]},
        )
        client.expire(routed.redis_key, ttl)
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
