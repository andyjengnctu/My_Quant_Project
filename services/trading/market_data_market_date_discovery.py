"""Persistent local planner/state for sparse Trading market-date discovery.

This state is operational only.  It does not participate in Provider Snapshot,
Research, registry, or Trading scientific identity.  The scheduler reads it
before resolving a FinMind token, so non-due wake-ups remain zero-provider-call.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.market_data_auto_update_policy import MarketDataAutoUpdatePolicy
from core.market_data_trading_storage_contract import (
    resolve_trading_market_data_v2_market_date_discovery_state_path,
)

MARKET_DATE_DISCOVERY_SCHEMA_VERSION = 1
DISCOVERY_RESULT_NEW_DATE = "NEW_DATE"
DISCOVERY_RESULT_NO_NEW_DATE = "NO_NEW_DATE"
DISCOVERY_RESULT_ERROR = "ERROR"
DISCOVERY_RESULT_WAIT_QUOTA = "WAIT_QUOTA"


def _with_tz(value: datetime, reference: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=reference.tzinfo)
    return value


def _next_weekday_first_check(now: datetime, policy: MarketDataAutoUpdatePolicy, *, include_today: bool) -> datetime:
    candidate = now.date()
    if not include_today:
        candidate += timedelta(days=1)
    while True:
        if candidate.weekday() < 5:
            result = datetime.combine(candidate, policy.market_date_discovery_first_check_time, tzinfo=now.tzinfo)
            if result > now:
                return result
        candidate += timedelta(days=1)


def default_next_market_date_probe_at(
    *,
    now: datetime,
    current_market_date: str,
    policy: MarketDataAutoUpdatePolicy,
) -> datetime:
    """Resolve the first locally useful provider-probe time.

    If today's canonical snapshot is already current, probe the next weekday.
    Otherwise allow today's first-check window when it has not passed yet; if it
    has passed, the caller should probe immediately instead of manufacturing a
    future delay.
    """

    today = now.date().isoformat()
    if str(current_market_date) >= today:
        return _next_weekday_first_check(now, policy, include_today=False)
    if now.weekday() >= 5:
        return _next_weekday_first_check(now, policy, include_today=False)
    first = datetime.combine(now.date(), policy.market_date_discovery_first_check_time, tzinfo=now.tzinfo)
    return first if first > now else now


def load_market_date_discovery_state(project_root, *, required: bool = False) -> dict[str, Any] | None:
    path = resolve_trading_market_data_v2_market_date_discovery_state_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Trading market-date discovery state 尚未建立")
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != MARKET_DATE_DISCOVERY_SCHEMA_VERSION:
        raise ValueError("Trading market-date discovery state schema 不相容")
    core = {key: value for key, value in payload.items() if key != "state_fingerprint"}
    if str(payload.get("state_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading market-date discovery state fingerprint 不一致")
    return payload


def publish_market_date_discovery_state(project_root, payload: dict[str, Any]) -> dict[str, Any]:
    state = {"schema_version": MARKET_DATE_DISCOVERY_SCHEMA_VERSION, **dict(payload)}
    state["state_fingerprint"] = canonical_json_sha256(state)
    atomic_write_json(resolve_trading_market_data_v2_market_date_discovery_state_path(project_root), state)
    return state


def plan_market_date_discovery(
    project_root,
    *,
    current_market_date: str,
    now: datetime,
    policy: MarketDataAutoUpdatePolicy,
) -> dict[str, Any]:
    state = load_market_date_discovery_state(project_root, required=False)
    if state is None or str(state.get("current_market_date") or "") != str(current_market_date):
        next_at = default_next_market_date_probe_at(
            now=now,
            current_market_date=str(current_market_date),
            policy=policy,
        )
        state = publish_market_date_discovery_state(
            project_root,
            {
                "current_market_date": str(current_market_date),
                "last_probe_at": None,
                "last_probe_market_date": None,
                "last_probe_result": None,
                "retry_count": 0,
                "next_probe_at": next_at.isoformat(),
                "last_error": None,
                "updated_at": now.isoformat(),
            },
        )
    next_raw = str(state.get("next_probe_at") or "")
    next_at = _with_tz(datetime.fromisoformat(next_raw), now) if next_raw else now
    return {
        "due": bool(now >= next_at),
        "next_probe_at": next_at.isoformat(),
        "state": state,
    }


def record_market_date_probe_result(
    project_root,
    *,
    current_market_date: str,
    observed_market_date: str | None,
    now: datetime,
    policy: MarketDataAutoUpdatePolicy,
    result: str,
    error: str | None = None,
) -> dict[str, Any]:
    prior = load_market_date_discovery_state(project_root, required=False) or {}
    same_target = str(prior.get("current_market_date") or "") == str(current_market_date)
    prior_retry = int(prior.get("retry_count") or 0) if same_target else 0

    if result == DISCOVERY_RESULT_NEW_DATE:
        retry_count = 0
        next_at = _next_weekday_first_check(now, policy, include_today=False)
    elif result == DISCOVERY_RESULT_NO_NEW_DATE:
        retry_count = prior_retry + 1
        if retry_count <= policy.market_date_discovery_max_retries:
            delay = policy.market_date_discovery_retry_delay_minutes(retry_count)
            next_at = now + timedelta(minutes=delay)
        else:
            retry_count = 0
            next_at = _next_weekday_first_check(now, policy, include_today=False)
    elif result == DISCOVERY_RESULT_WAIT_QUOTA:
        retry_count = prior_retry
        next_at = now + timedelta(minutes=policy.quota_defer_minutes)
    elif result == DISCOVERY_RESULT_ERROR:
        retry_count = prior_retry
        next_at = now + timedelta(minutes=policy.error_defer_minutes)
    else:
        raise ValueError(f"未知 market-date discovery result: {result}")

    return publish_market_date_discovery_state(
        project_root,
        {
            "current_market_date": str(current_market_date),
            "last_probe_at": now.isoformat(),
            "last_probe_market_date": None if observed_market_date is None else str(observed_market_date),
            "last_probe_result": str(result),
            "retry_count": int(retry_count),
            "next_probe_at": next_at.isoformat(),
            "last_error": None if error is None else str(error),
            "updated_at": now.isoformat(),
        },
    )


__all__ = [
    "MARKET_DATE_DISCOVERY_SCHEMA_VERSION",
    "DISCOVERY_RESULT_NEW_DATE",
    "DISCOVERY_RESULT_NO_NEW_DATE",
    "DISCOVERY_RESULT_ERROR",
    "DISCOVERY_RESULT_WAIT_QUOTA",
    "default_next_market_date_probe_at",
    "load_market_date_discovery_state",
    "publish_market_date_discovery_state",
    "plan_market_date_discovery",
    "record_market_date_probe_result",
]
