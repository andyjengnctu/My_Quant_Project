"""One-shot automatic updater for Trading Market Data V2 due datasets.

The worker is intentionally scheduler-friendly: it performs a local due-plan first
and does not resolve a FinMind token/client when no dataset is due.  Windows Task
Scheduler may therefore wake it frequently without consuming provider quota.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

from core.market_data_auto_update_policy import get_market_data_auto_update_policy
from core.market_data_dataset_readiness import (
    MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION,
    VALID_DATASET_VALIDATION_STATUSES,
    has_current_market_data_dataset_validation,
    is_market_data_dataset_ready,
)
from core.market_data_dataset_registry import get_market_dataset_specs
from core.trading_data_dependencies import get_trading_data_dependency_spec
from core.trading_policy import get_trading_strategy_profile
from core.market_data_freshness_contract import (
    FRESHNESS_STATUS_READY,
    FRESHNESS_STATUS_WAIT_PUBLISH,
)
from core.market_data_trading_storage_contract import resolve_trading_market_data_v2_auto_update_lock_path
from services.downloader.finmind_http import FinMindHttpClient, FinMindHttpError
from services.downloader.finmind_shared_client import SharedFinMindRequestClient
from services.downloader.trading_price_refresh import probe_latest_adjusted_price_market_date
from services.downloader.market_data_ledger import WORKLOAD_BLOCKED, WORKLOAD_WAIT_QUOTA
from services.downloader.market_data_trading_sync import sync_market_data_v2_due_datasets
from services.trading.market_data_dataset_state import (
    build_market_data_due_plan,
    load_market_data_dataset_state,
    refresh_market_data_due_state,
    schedule_market_data_auto_update_outcomes,
)
from services.trading.market_data_v2_state import (
    find_latest_ready_provider_snapshot,
    load_trading_market_data_v2_state,
    publish_trading_market_data_v2_auto_rollup,
)
from services.trading.market_data_market_date_discovery import (
    DISCOVERY_RESULT_ERROR,
    DISCOVERY_RESULT_NEW_DATE,
    DISCOVERY_RESULT_NO_NEW_DATE,
    DISCOVERY_RESULT_WAIT_QUOTA,
    load_market_date_discovery_state,
    plan_market_date_discovery,
    record_market_date_probe_result,
)

AUTO_UPDATE_STATUS_NO_TARGET = "NO_TARGET"
AUTO_UPDATE_STATUS_NO_DUE = "NO_DUE"
AUTO_UPDATE_STATUS_DISABLED = "DISABLED"
AUTO_UPDATE_STATUS_BUSY = "BUSY"
AUTO_UPDATE_STATUS_UPDATED = "UPDATED"
AUTO_UPDATE_STATUS_DEFERRED = "DEFERRED"
AUTO_UPDATE_STATUS_BLOCKED = "BLOCKED"
AUTO_UPDATE_STATUS_TARGET_ADVANCED = "TARGET_ADVANCED"


def _local_now(now_fn: Callable[[], datetime] | None) -> datetime:
    value = now_fn() if now_fn is not None else datetime.now().astimezone()
    if value.tzinfo is None:
        return value.astimezone()
    return value


def _resolve_target_date(project_root: Path, explicit: str | None) -> str | None:
    if explicit:
        return str(explicit)
    provider = find_latest_ready_provider_snapshot(project_root)
    if provider is None:
        return None
    _provider_path, provider_payload = provider
    candidates = [str(provider_payload.get("as_of_date") or "").strip()]
    archive_state = load_trading_market_data_v2_state(project_root, required=False)
    if archive_state is not None:
        candidates.extend(
            str(archive_state.get(key) or "").strip()
            for key in ("latest_sync_target_date", "last_attempt_target_date")
        )
    discovery_state = load_market_date_discovery_state(project_root, required=False)
    if discovery_state is not None:
        candidates.append(str(discovery_state.get("current_market_date") or "").strip())
    resolved = [value for value in candidates if value]
    return max(resolved) if resolved else None


def _next_check_at(plan) -> str | None:
    values = [str(item.next_check_at) for item in plan.decisions if item.next_check_at]
    return min(values) if values else None


def _prepare_provider_client(*, root: Path, token: str | None, client):
    if isinstance(client, SharedFinMindRequestClient):
        return token, client
    if client is not None:
        return token, SharedFinMindRequestClient(client)
    if token is None:
        from services.downloader import runtime as downloader_runtime

        token = downloader_runtime.resolve_finmind_api_token(project_root=root)
    return token, SharedFinMindRequestClient(FinMindHttpClient(token=str(token or "")))


def _client_counts(client) -> tuple[int, int]:
    return (
        int(getattr(client, "data_request_count", 0)) if client is not None else 0,
        int(getattr(client, "usage_request_count", 0)) if client is not None else 0,
    )


def _dataset_readiness_summary(state, *, target_date: str) -> dict[str, object]:
    rows = dict((state or {}).get("datasets") or {})
    total = len(rows)
    archive_ready = sum(
        is_market_data_dataset_ready(dict(row or {}), target_date=target_date)
        for row in rows.values()
    )
    schema_ready = 0
    coverage_ready = 0
    validation_current = 0
    for raw in rows.values():
        row = dict(raw or {})
        try:
            current = int(row.get("validation_contract_version", -1)) == MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION
        except (TypeError, ValueError):
            current = False
        if current:
            validation_current += 1
            if str(row.get("schema_status") or "") in VALID_DATASET_VALIDATION_STATUSES:
                schema_ready += 1
            if str(row.get("coverage_status") or "") in VALID_DATASET_VALIDATION_STATUSES:
                coverage_ready += 1

    strategy_id = get_trading_strategy_profile().strategy_id
    dependency = get_trading_data_dependency_spec(strategy_id)
    required = tuple(dependency.required_v2_datasets)
    required_ready = sum(
        is_market_data_dataset_ready(dict(rows.get(dataset) or {}), target_date=target_date)
        for dataset in required
    )
    blocking_required = tuple(
        dataset
        for dataset in required
        if not is_market_data_dataset_ready(dict(rows.get(dataset) or {}), target_date=target_date)
    )
    archive_incomplete = tuple(
        {
            "dataset": dataset,
            "status": str(dict(rows.get(dataset) or {}).get("status") or "UNKNOWN"),
            "latest_data_date": dict(rows.get(dataset) or {}).get("latest_data_date"),
            "schema_status": str(dict(rows.get(dataset) or {}).get("schema_status") or "UNKNOWN"),
            "coverage_status": str(dict(rows.get(dataset) or {}).get("coverage_status") or "UNKNOWN"),
            "last_error": dict(rows.get(dataset) or {}).get("last_error"),
        }
        for dataset in sorted(rows)
        if not is_market_data_dataset_ready(dict(rows.get(dataset) or {}), target_date=target_date)
    )
    return {
        "archive_ready_dataset_count": int(archive_ready),
        "archive_dataset_count": int(total),
        "current_validation_dataset_count": int(validation_current),
        "schema_ready_dataset_count": int(schema_ready),
        "coverage_ready_dataset_count": int(coverage_ready),
        "trading_strategy_id": strategy_id,
        "trading_required_ready_dataset_count": int(required_ready),
        "trading_required_dataset_count": len(required),
        "trading_blocking_datasets": blocking_required,
        "archive_incomplete_datasets": archive_incomplete,
    }


def _discover_new_market_date_if_due(
    *,
    root: Path,
    current_target: str,
    now: datetime,
    policy,
    token: str | None,
    client,
    force: bool = False,
):
    discovery = plan_market_date_discovery(
        root,
        current_market_date=current_target,
        now=now,
        policy=policy,
    )
    if not discovery["due"] and not force:
        return current_target, token, client, False, discovery["next_probe_at"], None

    token, client = _prepare_provider_client(root=root, token=token, client=client)
    try:
        probe = probe_latest_adjusted_price_market_date(
            client=client,
            now=now,
            current_market_date=current_target,
        )
    except FinMindHttpError as exc:
        result = DISCOVERY_RESULT_WAIT_QUOTA if exc.quota_exhausted else DISCOVERY_RESULT_ERROR
        state = record_market_date_probe_result(
            root,
            current_market_date=current_target,
            observed_market_date=None,
            now=now,
            policy=policy,
            result=result,
            error=f"{type(exc).__name__}: {exc}",
        )
        return current_target, token, client, False, state["next_probe_at"], f"{type(exc).__name__}: {exc}"
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        state = record_market_date_probe_result(
            root,
            current_market_date=current_target,
            observed_market_date=None,
            now=now,
            policy=policy,
            result=DISCOVERY_RESULT_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )
        return current_target, token, client, False, state["next_probe_at"], f"{type(exc).__name__}: {exc}"

    observed = str(probe.market_date)
    if observed <= str(current_target):
        state = record_market_date_probe_result(
            root,
            current_market_date=current_target,
            observed_market_date=observed,
            now=now,
            policy=policy,
            result=DISCOVERY_RESULT_NO_NEW_DATE,
        )
        return current_target, token, client, False, state["next_probe_at"], None

    state = record_market_date_probe_result(
        root,
        current_market_date=observed,
        observed_market_date=observed,
        now=now,
        policy=policy,
        result=DISCOVERY_RESULT_NEW_DATE,
    )
    return observed, token, client, True, state["next_probe_at"], None


@contextmanager
def _auto_update_lock(project_root: Path, *, now: datetime, lease_minutes: int):
    path = resolve_trading_market_data_v2_auto_update_lock_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = f"auto-{uuid4().hex}"
    payload = {"owner": owner, "acquired_at": now.isoformat(), "lease_until": (now + timedelta(minutes=lease_minutes)).isoformat()}
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                lease_until = datetime.fromisoformat(str(existing.get("lease_until") or ""))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                lease_until = now - timedelta(seconds=1)
            if lease_until.tzinfo is None:
                lease_until = lease_until.replace(tzinfo=now.tzinfo)
            if lease_until > now:
                yield False
                return
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            break
    try:
        yield True
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            current = {}
        if str(current.get("owner") or "") == owner:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def run_trading_market_data_auto_update(
    *,
    project_root: str | Path,
    target_date: str | None = None,
    token: str | None = None,
    client: FinMindHttpClient | None = None,
    sink=None,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    force_market_date_discovery: bool = False,
    force_refresh_current_target: bool = False,
    progress_fn: Callable[[dict[str, object]], None] | None = None,
    quota_wait_fn: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run exactly one scheduler-safe auto-update iteration."""

    root = Path(project_root).resolve()
    policy = get_market_data_auto_update_policy()
    now = _local_now(now_fn)
    if not policy.enabled:
        return {"status": AUTO_UPDATE_STATUS_DISABLED, "provider_requests_required": False, "data_requests": 0, "usage_requests": 0}

    resolved_target = _resolve_target_date(root, target_date)
    if not resolved_target:
        return {"status": AUTO_UPDATE_STATUS_NO_TARGET, "provider_requests_required": False, "data_requests": 0, "usage_requests": 0, "next_check_at": None}

    with _auto_update_lock(root, now=now, lease_minutes=policy.worker_lock_minutes) as acquired:
        if not acquired:
            return {"status": AUTO_UPDATE_STATUS_BUSY, "target_date": resolved_target, "provider_requests_required": False, "data_requests": 0, "usage_requests": 0}

        discovery_next_check = None
        target_advanced = False
        discovery_error = None
        data_before, usage_before = _client_counts(client)
        if target_date is None:
            (
                resolved_target,
                token,
                client,
                target_advanced,
                discovery_next_check,
                discovery_error,
            ) = _discover_new_market_date_if_due(
                root=root,
                current_target=resolved_target,
                now=now,
                policy=policy,
                token=token,
                client=client,
                force=force_market_date_discovery,
            )
            if discovery_error is not None:
                data_after, usage_after = _client_counts(client)
                return {
                    "status": AUTO_UPDATE_STATUS_DEFERRED,
                    "target_date": resolved_target,
                    "provider_requests_required": True,
                    "due_dataset_count": 0,
                    "due_datasets": (),
                    "data_requests": data_after - data_before,
                    "usage_requests": usage_after - usage_before,
                    "market_date_discovery_error": discovery_error,
                    "market_date_discovery_next_check_at": discovery_next_check,
                    "next_check_at": discovery_next_check,
                }

        plan, _state = refresh_market_data_due_state(root, target_date=resolved_target, now=now)
        due = (
            tuple(sorted(spec.dataset for spec in get_market_dataset_specs(included_only=True)))
            if force_refresh_current_target
            else tuple(plan.due_datasets)
        )
        if not due:
            data_after, usage_after = _client_counts(client)
            local_next = _next_check_at(plan)
            candidates = [value for value in (local_next, discovery_next_check) if value]
            next_check = min(candidates) if candidates else None
            state_summary = _dataset_readiness_summary(_state, target_date=resolved_target)
            return {
                "status": AUTO_UPDATE_STATUS_TARGET_ADVANCED if target_advanced else AUTO_UPDATE_STATUS_NO_DUE,
                "target_date": resolved_target,
                "provider_requests_required": bool(data_after > data_before or usage_after > usage_before),
                "due_dataset_count": 0,
                "due_datasets": (),
                "data_requests": data_after - data_before,
                "usage_requests": usage_after - usage_before,
                "market_date_discovery_next_check_at": discovery_next_check,
                "v2_target_advanced": target_advanced,
                "next_check_at": next_check,
                "request_count": 0,
                "exact_date_request_count": 0,
                "fixed_data_id_exact_date_request_count": 0,
                "unique_exact_date_count": 0,
                "full_market_exact_date_request_count": 0,
                "range_request_count": 0,
                "undated_request_count": 0,
                "request_date_start": None,
                "request_date_end": None,
                "force_refresh": False,
                "verification": {},
                **state_summary,
            }

        if client is None:
            if token is None:
                from services.downloader import runtime as downloader_runtime

                token = downloader_runtime.resolve_finmind_api_token(project_root=root)
            client = FinMindHttpClient(token=str(token or ""))

        batch_data_before, batch_usage_before = _client_counts(client)
        try:
            batch = sync_market_data_v2_due_datasets(
                project_root=root,
                target_date=resolved_target,
                token=str(token or ""),
                due_datasets=due,
                client=client,
                sink=sink,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
                progress_fn=progress_fn,
                quota_wait_fn=quota_wait_fn,
                force_refresh_current_target=bool(force_refresh_current_target),
                refresh_token=(f"manual-refresh:{now.isoformat()}:{uuid4().hex[:8]}" if force_refresh_current_target else None),
            )
        except (FinMindHttpError, OSError, ValueError, RuntimeError, ImportError) as exc:
            schedule_market_data_auto_update_outcomes(
                root,
                target_date=resolved_target,
                now=now,
                publication_retry_minutes=policy.publication_retry_minutes,
                max_publication_retries=policy.max_publication_retries,
                quota_defer_minutes=policy.quota_defer_minutes,
                error_defer_minutes=policy.error_defer_minutes,
                error_datasets=due,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            post = build_market_data_due_plan(root, target_date=resolved_target, now=now)
            publish_trading_market_data_v2_auto_rollup(root, target_date=resolved_target, updated_at=now, batch_result=None)
            return {
                "status": AUTO_UPDATE_STATUS_DEFERRED,
                "target_date": resolved_target,
                "provider_requests_required": True,
                "due_dataset_count": len(due),
                "due_datasets": due,
                "data_requests": _client_counts(client)[0] - data_before,
                "usage_requests": _client_counts(client)[1] - usage_before,
                "error": f"{type(exc).__name__}: {exc}",
                "next_check_at": _next_check_at(post),
            }

        completed = set(batch.get("completed_datasets") or ())
        incomplete = set(batch.get("incomplete_datasets") or ())
        current_state = load_market_data_dataset_state(root, required=True)
        rows = dict(current_state["datasets"])
        wait_publish = {
            dataset
            for dataset in completed
            if str(dict(rows.get(dataset) or {}).get("status") or "") == FRESHNESS_STATUS_WAIT_PUBLISH
        }
        if wait_publish:
            schedule_market_data_auto_update_outcomes(
                root,
                target_date=resolved_target,
                now=now,
                publication_retry_minutes=policy.publication_retry_minutes,
                max_publication_retries=policy.max_publication_retries,
                quota_defer_minutes=policy.quota_defer_minutes,
                error_defer_minutes=policy.error_defer_minutes,
                wait_publish_datasets=wait_publish,
            )

        workload_status = str(batch.get("status") or "")
        if incomplete:
            if workload_status == WORKLOAD_WAIT_QUOTA:
                schedule_market_data_auto_update_outcomes(
                    root,
                    target_date=resolved_target,
                    now=now,
                    publication_retry_minutes=policy.publication_retry_minutes,
                    max_publication_retries=policy.max_publication_retries,
                    quota_defer_minutes=policy.quota_defer_minutes,
                    error_defer_minutes=policy.error_defer_minutes,
                    wait_quota_datasets=incomplete,
                )
            elif workload_status in {WORKLOAD_BLOCKED, "NOT_BOOTSTRAPPED"} or int(batch.get("blocked") or 0) > 0:
                schedule_market_data_auto_update_outcomes(
                    root,
                    target_date=resolved_target,
                    now=now,
                    publication_retry_minutes=policy.publication_retry_minutes,
                    max_publication_retries=policy.max_publication_retries,
                    quota_defer_minutes=policy.quota_defer_minutes,
                    error_defer_minutes=policy.error_defer_minutes,
                    blocked_datasets=incomplete,
                    error_message="automatic due batch blocked",
                )
            else:
                schedule_market_data_auto_update_outcomes(
                    root,
                    target_date=resolved_target,
                    now=now,
                    publication_retry_minutes=policy.publication_retry_minutes,
                    max_publication_retries=policy.max_publication_retries,
                    quota_defer_minutes=policy.quota_defer_minutes,
                    error_defer_minutes=policy.error_defer_minutes,
                    error_datasets=incomplete,
                    error_message=f"automatic due batch incomplete: {workload_status}",
                )

        final_state = load_market_data_dataset_state(root, required=True)
        final_rows = dict(final_state["datasets"])
        state_summary = _dataset_readiness_summary(final_state, target_date=resolved_target)
        ready_count = int(state_summary["archive_ready_dataset_count"])
        rollup = publish_trading_market_data_v2_auto_rollup(root, target_date=resolved_target, updated_at=now, batch_result=batch)
        post_plan = build_market_data_due_plan(root, target_date=resolved_target, now=now)
        overall = AUTO_UPDATE_STATUS_UPDATED if ready_count == len(final_rows) else AUTO_UPDATE_STATUS_DEFERRED
        if any(str(dict(final_rows.get(name) or {}).get("status") or "") == "BLOCKED" for name in due):
            overall = AUTO_UPDATE_STATUS_BLOCKED
        return {
            "status": overall,
            "target_date": resolved_target,
            "provider_requests_required": True,
            "due_dataset_count": len(due),
            "due_datasets": due,
            "completed_datasets": tuple(sorted(completed)),
            "incomplete_datasets": tuple(sorted(incomplete)),
            "ready_dataset_count": int(ready_count),
            "dataset_count": len(final_rows),
            "request_count": int(batch.get("request_count") or 0),
            "done": int(batch.get("done") or 0),
            "data_requests": _client_counts(client)[0] - data_before,
            "usage_requests": _client_counts(client)[1] - usage_before,
            "batch_data_requests": _client_counts(client)[0] - batch_data_before,
            "batch_usage_requests": _client_counts(client)[1] - batch_usage_before,
            "market_date_discovery_next_check_at": discovery_next_check,
            "v2_target_advanced": target_advanced,
            "next_check_at": min(
                [value for value in (_next_check_at(post_plan), discovery_next_check) if value],
                default=None,
            ),
            "archive_status": None if rollup is None else rollup.get("status"),
            "batch_fingerprint": batch.get("batch_fingerprint"),
            "force_refresh": bool(force_refresh_current_target),
            "refresh_token": batch.get("refresh_token"),
            "verification": dict(batch.get("verification") or {}),
            "request_date_start": batch.get("request_date_start"),
            "request_date_end": batch.get("request_date_end"),
            "unique_exact_date_count": int(batch.get("unique_exact_date_count") or 0),
            "exact_date_request_count": int(batch.get("exact_date_request_count") or 0),
            "fixed_data_id_exact_date_request_count": int(batch.get("fixed_data_id_exact_date_request_count") or 0),
            "full_market_exact_date_request_count": int(batch.get("full_market_exact_date_request_count") or 0),
            "range_request_count": int(batch.get("range_request_count") or 0),
            "undated_request_count": int(batch.get("undated_request_count") or 0),
            **state_summary,
        }


__all__ = [
    "AUTO_UPDATE_STATUS_NO_TARGET",
    "AUTO_UPDATE_STATUS_NO_DUE",
    "AUTO_UPDATE_STATUS_DISABLED",
    "AUTO_UPDATE_STATUS_BUSY",
    "AUTO_UPDATE_STATUS_UPDATED",
    "AUTO_UPDATE_STATUS_DEFERRED",
    "AUTO_UPDATE_STATUS_BLOCKED",
    "AUTO_UPDATE_STATUS_TARGET_ADVANCED",
    "run_trading_market_data_auto_update",
]
