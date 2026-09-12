"""Local-first read model for Workbench Market Data Operations.

This service composes existing canonical Trading market-data owners.  It does
not call FinMind and does not persist state; the Workbench Data Ops panel may
therefore refresh this read model without consuming provider quota.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from core.market_data_auto_update_policy import get_market_data_auto_update_policy
from core.market_data_dataset_readiness import (
    VALIDATION_STATUS_NO_ROW_VALID,
    is_market_data_dataset_ready,
)
from core.market_data_due_planner import normalize_market_data_planner_now, plan_market_data_due_datasets
from core.market_data_dataset_registry import get_market_dataset_display_name_zh
from core.market_data_freshness_contract import (
    CADENCE_CURRENT_VINTAGE,
    CADENCE_EVENT_DRIVEN,
    EXPECTED_DATE_LATEST_AVAILABLE,
    EXPECTED_DATE_NONE,
    EXPECTED_DATE_PERIOD_DUE,
    EXPECTED_DATE_TRADING_TARGET,
    FRESHNESS_STATUS_NOT_APPLICABLE,
    FRESHNESS_STATUS_READY,
)
from services.trading.data_readiness import build_trading_data_readiness
from services.trading.market_data_dataset_state import build_market_data_dataset_state_read_model
from services.trading.market_data_consumer import load_trading_v2_consumer_state
from services.trading.market_data_v2_state import (
    build_trading_market_data_v2_read_model,
    resolve_trading_market_data_update_target_date,
)
from services.trading.market_data_market_date_discovery import (
    default_next_market_date_probe_at,
    load_market_date_discovery_state,
)
from services.trading.market_data_scheduler import get_market_data_scheduler_status


def _parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed


def _earliest_iso(values) -> str | None:
    candidates = []
    for value in values:
        parsed = _parse_datetime(value)
        if parsed is not None:
            candidates.append(parsed)
    if not candidates:
        return None
    return min(candidates).isoformat()


def _dataset_display_semantics(row: dict[str, Any]) -> dict[str, str]:
    """Resolve explicit UI semantics from the canonical freshness contract/state."""

    latest = str(row.get("latest_data_date") or "").strip()
    expected = str(row.get("latest_expected_date") or "").strip()
    next_check = str(row.get("next_check_at") or "").strip()
    status = str(row.get("projected_status") or row.get("status") or "UNKNOWN").strip().upper()
    cadence = str(row.get("cadence") or "").strip()
    expected_mode = str(row.get("expected_date_mode") or "").strip()
    schema_status = str(row.get("schema_status") or "").strip().upper()
    coverage_status = str(row.get("coverage_status") or "").strip().upper()
    no_row_valid = VALIDATION_STATUS_NO_ROW_VALID in {schema_status, coverage_status}

    if latest:
        latest_display = latest
    elif no_row_valid:
        latest_display = "NO ROW · valid"
    elif cadence == CADENCE_CURRENT_VINTAGE:
        latest_display = "CURRENT · no date"
    else:
        latest_display = "UNKNOWN"

    if expected:
        expected_display = expected
    elif expected_mode == EXPECTED_DATE_PERIOD_DUE:
        expected_display = "PERIODIC · due window"
    elif expected_mode == EXPECTED_DATE_LATEST_AVAILABLE:
        expected_display = "LATEST AVAILABLE"
    elif expected_mode == EXPECTED_DATE_NONE and cadence == CADENCE_EVENT_DRIVEN:
        expected_display = "EVENT · when present"
    elif expected_mode == EXPECTED_DATE_NONE:
        expected_display = "N/A · current vintage"
    elif expected_mode == EXPECTED_DATE_TRADING_TARGET:
        expected_display = "TARGET · unresolved"
    else:
        expected_display = "UNKNOWN"

    if next_check:
        next_check_display = next_check
    elif status == FRESHNESS_STATUS_READY:
        next_check_display = "READY · await new target"
    elif status == FRESHNESS_STATUS_NOT_APPLICABLE:
        next_check_display = "N/A"
    else:
        next_check_display = "UNSCHEDULED"

    return {
        "latest_display": latest_display,
        "expected_display": expected_display,
        "next_check_display": next_check_display,
    }


def build_market_data_ops_read_model(
    project_root: str | Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a provider-free Workbench Data Ops snapshot."""

    root = Path(project_root).resolve()
    local_now = normalize_market_data_planner_now(now or datetime.now().astimezone())
    consumer_state = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
    target_date = None if consumer_state is None else str(consumer_state.get("market_date") or "") or None
    # AI: Execution may remain at the common READY horizon while the updater
    # already targets a newer date. Due/freshness must use the updater's owner.
    update_target_date = resolve_trading_market_data_update_target_date(root)
    trading_readiness = build_trading_data_readiness(root, verify_consumer_view=False)
    dataset_state = build_market_data_dataset_state_read_model(root)
    v2 = build_trading_market_data_v2_read_model(root)
    auto_policy = get_market_data_auto_update_policy()
    scheduler = get_market_data_scheduler_status(root)

    dynamic_rows = {str(row.get("dataset")): dict(row) for row in dataset_state.get("datasets") or []}
    decisions = {}
    if update_target_date:
        state_payload = None
        if dataset_state.get("state_ready"):
            from services.trading.market_data_dataset_state import load_market_data_dataset_state

            state_payload = load_market_data_dataset_state(root, required=False)
        plan = plan_market_data_due_datasets(target_date=update_target_date, now=local_now, state=state_payload)
        decisions = {item.dataset: item for item in plan.decisions}
    else:
        plan = None

    rows: list[dict[str, Any]] = []
    for dataset in sorted(dynamic_rows):
        row = dict(dynamic_rows[dataset])
        decision = decisions.get(dataset)
        projected_status = row.get("status")
        expected_publish_at = row.get("expected_publish_at")
        next_check_at = row.get("next_check_at")
        due = False
        due_reason = None
        if decision is not None:
            projected_status = decision.status
            expected_publish_at = decision.expected_publish_at
            next_check_at = decision.next_check_at
            due = bool(decision.due)
            due_reason = decision.reason
        rows.append(
            {
                **row,
                "display_name_zh": get_market_dataset_display_name_zh(dataset),
                "projected_status": projected_status,
                "expected_publish_at": expected_publish_at,
                "next_check_at": next_check_at,
                "due": due,
                "due_reason": due_reason,
                **_dataset_display_semantics(
                    {
                        **row,
                        "projected_status": projected_status,
                        "expected_publish_at": expected_publish_at,
                        "next_check_at": next_check_at,
                    }
                ),
            }
        )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("projected_status") or row.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    ready_count = sum(
        1
        for row in rows
        if update_target_date is not None
        and is_market_data_dataset_ready(row, target_date=str(update_target_date))
    )
    due_count = sum(bool(row.get("due")) for row in rows)
    dataset_next_check_at = _earliest_iso(row.get("next_check_at") for row in rows)
    discovery_state = load_market_date_discovery_state(root, required=False)
    discovery_next_check_at = None
    discovery_last_result = None
    discovery_last_probe_at = None
    if update_target_date:
        if discovery_state is not None and str(discovery_state.get("current_market_date") or "") == update_target_date:
            discovery_next_check_at = discovery_state.get("next_probe_at")
            discovery_last_result = discovery_state.get("last_probe_result")
            discovery_last_probe_at = discovery_state.get("last_probe_at")
        else:
            discovery_next_check_at = default_next_market_date_probe_at(
                now=local_now,
                current_market_date=update_target_date,
                policy=auto_policy,
            ).isoformat()
    next_check_at = _earliest_iso((dataset_next_check_at, discovery_next_check_at))
    recent_activity = [
        {
            "activity_type": "dataset",
            "dataset": row.get("dataset"),
            "display_name_zh": row.get("display_name_zh"),
            "at": row.get("last_attempt_at") or row.get("last_success_at"),
            "result": row.get("last_attempt_result") or row.get("projected_status"),
            "detail": row.get("last_error") or "-",
        }
        for row in rows
        if row.get("last_attempt_at") or row.get("last_success_at")
    ]

    quota_used = v2.get("quota_user_count")
    quota_limit = v2.get("quota_limit")
    quota_observed_at = _parse_datetime(v2.get("quota_observed_at"))
    quota_observation_status = "NONE"
    if quota_observed_at is not None:
        if quota_observed_at.tzinfo is not None:
            quota_observed_at = quota_observed_at.astimezone(local_now.tzinfo)
        quota_observation_status = "CURRENT" if quota_observed_at.date() == local_now.date() else "STALE"
    quota_percent = None
    if quota_used is not None and quota_limit not in (None, 0):
        try:
            quota_percent = max(0.0, min(100.0, 100.0 * float(quota_used) / float(quota_limit)))
        except (TypeError, ValueError, ZeroDivisionError):
            quota_percent = None

    if v2.get("quota_observed_at") and quota_limit is not None:
        percent_text = "-" if quota_percent is None else f"{float(quota_percent):.1f}%"
        recent_activity.append(
            {
                "activity_type": "quota",
                "dataset": "Provider Quota",
                "display_name_zh": "FinMind 配額用量",
                "at": v2.get("quota_observed_at"),
                "result": quota_observation_status,
                "detail": (
                    f"quota={quota_used}/{quota_limit} ({percent_text}) | "
                    f"last auto data/usage={int(v2.get('latest_auto_data_requests') or 0)}/"
                    f"{int(v2.get('latest_auto_usage_requests') or 0)}"
                ),
            }
        )
    quota_activity_present = any(item.get("activity_type") == "quota" for item in recent_activity)
    recent_activity = sorted(
        recent_activity,
        key=lambda item: (
            str(item.get("at") or ""),
            1 if item.get("activity_type") == "quota" else 0,
        ),
        reverse=True,
    )
    if quota_activity_present and not any(item.get("activity_type") == "quota" for item in recent_activity[:20]):
        latest_quota_activity = next(item for item in recent_activity if item.get("activity_type") == "quota")
        recent_activity = recent_activity[:19] + [latest_quota_activity]
    else:
        recent_activity = recent_activity[:20]

    scheduler_status = str(scheduler.get("status") or "")
    auto_sync_active = bool(auto_policy.enabled and scheduler_status == "INSTALLED_ENABLED")

    return {
        "generated_at": local_now.isoformat(),
        "provider_calls": 0,
        "trading_target_date": target_date,
        "update_target_date": update_target_date,
        "trading_consumer_state_exists": consumer_state is not None,
        "trading_market_data_source": None if consumer_state is None else consumer_state.get("source"),
        "trading_strategy_id": trading_readiness.get("strategy_id"),
        "trading_ready": bool(trading_readiness.get("ready")),
        "trading_readiness_status": trading_readiness.get("status"),
        "trading_dependency_fingerprint": trading_readiness.get("dependency_fingerprint"),
        "trading_consumer_state_required": bool(trading_readiness.get("consumer_state_required")),
        "trading_consumer_state_ready": bool(trading_readiness.get("consumer_state_ready")),
        "trading_required_v2_count": int(trading_readiness.get("required_v2_dataset_count") or 0),
        "trading_ready_v2_count": int(trading_readiness.get("ready_v2_dataset_count") or 0),
        "trading_blocking_v2_count": int(trading_readiness.get("blocking_v2_dataset_count") or 0),
        "trading_required_v2_datasets": list(trading_readiness.get("required_v2_datasets") or []),
        "trading_blocking_dependencies": list(trading_readiness.get("blocking_dependencies") or []),
        "v2_status": v2.get("status"),
        "v2_latest_sync_target_date": v2.get("latest_sync_target_date"),
        "provider_ready": bool(v2.get("provider_ready")),
        "provider_as_of_date": v2.get("provider_as_of_date"),
        "dataset_state_ready": bool(dataset_state.get("state_ready")),
        "dataset_state_updated_at": dataset_state.get("updated_at"),
        "dataset_count": len(rows),
        "ready_count": ready_count,
        "pending_count": max(0, len(rows) - ready_count),
        "due_count": due_count,
        "status_counts": status_counts,
        "next_check_at": next_check_at,
        "dataset_next_check_at": dataset_next_check_at,
        "market_date_discovery_next_check_at": discovery_next_check_at,
        "market_date_discovery_last_probe_at": discovery_last_probe_at,
        "market_date_discovery_last_result": discovery_last_result,
        "auto_worker_enabled": bool(auto_policy.enabled),
        "auto_sync_active": auto_sync_active,
        "scheduler_wake_minutes": int(auto_policy.scheduler_wake_minutes),
        "scheduler_registration_status": scheduler.get("status"),
        "scheduler_supported": bool(scheduler.get("supported")),
        "scheduler_installed": bool(scheduler.get("installed")),
        "scheduler_enabled": bool(scheduler.get("enabled")),
        "scheduler_state": scheduler.get("state"),
        "scheduler_task_name": scheduler.get("task_name"),
        "scheduler_next_run_at": scheduler.get("next_run_at"),
        "scheduler_last_run_at": scheduler.get("last_run_at"),
        "scheduler_last_task_result": scheduler.get("last_task_result"),
        "scheduler_missed_runs": scheduler.get("missed_runs"),
        "scheduler_drift_reasons": scheduler.get("drift_reasons") or (),
        "scheduler_error": scheduler.get("error"),
        "scheduler_app_path": scheduler.get("app_path") or "apps/market_data_auto_update.py",
        "quota_user_count": quota_used,
        "quota_limit": quota_limit,
        "quota_percent": quota_percent,
        "quota_remaining": v2.get("quota_remaining"),
        "quota_usable_remaining": v2.get("quota_usable_remaining"),
        "quota_observed_at": v2.get("quota_observed_at"),
        "quota_observation_status": quota_observation_status,
        "latest_auto_request_count": v2.get("latest_auto_request_count"),
        "latest_auto_data_requests": v2.get("latest_auto_data_requests"),
        "latest_auto_usage_requests": v2.get("latest_auto_usage_requests"),
        "latest_request_count": v2.get("latest_request_count"),
        "latest_row_count": v2.get("latest_row_count"),
        "overall_v2_ready": bool(update_target_date and len(rows) and ready_count == len(rows)),
        "datasets": rows,
        "recent_activity": recent_activity,
    }


__all__ = ["build_market_data_ops_read_model"]
