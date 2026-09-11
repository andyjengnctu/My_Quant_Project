"""Non-blocking Trading Market Data V2 archive synchronization service."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, atomic_write_text, canonical_json_sha256
from core.market_data_dataset_readiness import has_current_market_data_dataset_validation
from core.market_data_dataset_registry import get_market_dataset_specs
from core.market_data_freshness_contract import EXPECTED_DATE_LATEST_AVAILABLE, get_market_data_freshness_contracts
from core.market_data_execution_policy import get_market_data_execution_policy
from core.market_data_trading_storage_contract import resolve_trading_market_data_v2_ledger_path
from core.market_data_trading_sync import build_trading_sync_request_manifest
from core.market_data_trading_sync_policy import get_market_data_trading_sync_policy
from services.downloader.finmind_http import FinMindHttpClient
from services.downloader.market_data_executor import MarketDataBootstrapExecutor
from services.downloader.market_data_ledger import MarketDataJobLedger, WORKLOAD_DONE
from services.trading.market_data_v2_state import (
    TRADING_V2_ARCHIVE_STATUS_NOT_BOOTSTRAPPED,
    TRADING_V2_ARCHIVE_STATUS_STALE,
    TRADING_V2_ARCHIVE_STATUS_SYNCED,
    find_latest_ready_provider_snapshot,
    find_ready_provider_snapshot_by_fingerprint,
    load_trading_market_data_v2_state,
    publish_trading_market_data_v2_state,
)
from services.downloader.market_data_trading_storage import MarketDataTradingStorageSink
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.market_data.provider_snapshot_view import ProviderSnapshotView

TRADING_SYNC_REPORT_PREFIX = "market_data_v2_trading_sync_"


def _merge_latest_available_provider_snapshot_evidence(
    *,
    project_root: Path,
    provider_payload: dict[str, object],
    selected_datasets: tuple[str, ...],
    dynamic_rows: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Seed missing latest-available lane dates from immutable provider evidence."""

    specs = {spec.dataset: spec for spec in get_market_dataset_specs(included_only=True)}
    contracts = {item.dataset: item for item in get_market_data_freshness_contracts()}
    wanted: list[str] = []
    for dataset in selected_datasets:
        contract = contracts.get(dataset)
        if contract is None or contract.expected_date_mode != EXPECTED_DATE_LATEST_AVAILABLE:
            continue
        spec = specs[dataset]
        row = dict(dynamic_rows.get(dataset) or {})
        lane_map = dict(row.get("latest_data_date_by_data_id") or {})
        expected_lanes = tuple(spec.trading_fixed_data_ids or spec.fixed_data_ids)
        lane_missing = any(not str(lane_map.get(str(lane)) or "").strip() for lane in expected_lanes)
        dataset_missing = not str(row.get("latest_data_date") or "").strip()
        if dataset_missing or lane_missing:
            wanted.append(dataset)
    if not wanted:
        return {}

    snapshot_fingerprint = str(provider_payload.get("snapshot_fingerprint") or "").strip()
    archive = load_ready_provider_snapshot_archive(
        project_root,
        snapshot_fingerprint=snapshot_fingerprint,
    )
    view = ProviderSnapshotView(project_root=project_root, archive=archive)
    evidence: dict[str, dict[str, object]] = {}
    for dataset in wanted:
        evidence[dataset] = view.latest_data_dates(dataset)
    return evidence


def _merge_latest_date_evidence(
    operational: dict[str, object],
    provider: dict[str, object] | None,
) -> dict[str, object]:
    provider_row = dict(provider or {})
    operational_row = dict(operational or {})
    provider_lanes = dict(provider_row.get("latest_data_date_by_data_id") or {})
    operational_lanes = dict(operational_row.get("latest_data_date_by_data_id") or {})
    merged_lanes = {**provider_lanes, **operational_lanes}
    dates = [
        str(value).strip()
        for value in (
            provider_row.get("latest_data_date"),
            operational_row.get("latest_data_date"),
            *merged_lanes.values(),
        )
        if str(value or "").strip()
    ]
    return {
        "latest_data_date": max(dates) if dates else None,
        "latest_data_date_by_data_id": merged_lanes,
    }


def _request_geometry_summary(requests) -> dict[str, object]:
    dated = [request for request in requests if request.start_date and request.end_date]
    exact = [request for request in dated if request.start_date == request.end_date]
    full_market_exact = [request for request in exact if request.data_id is None]
    range_requests = [request for request in dated if request.start_date != request.end_date]
    undated = [request for request in requests if not request.start_date and not request.end_date]
    start_dates = [str(request.start_date) for request in dated]
    end_dates = [str(request.end_date) for request in dated]
    exact_dates = sorted({str(request.start_date) for request in exact})
    fixed_id_exact = [request for request in exact if request.data_id is not None]
    return {
        "request_date_start": min(start_dates) if start_dates else None,
        "request_date_end": max(end_dates) if end_dates else None,
        "unique_exact_date_count": len(exact_dates),
        "exact_date_request_count": len(exact),
        "full_market_exact_date_request_count": len(full_market_exact),
        "fixed_data_id_exact_date_request_count": len(fixed_id_exact),
        "range_request_count": len(range_requests),
        "undated_request_count": len(undated),
    }


def _markdown(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Market Data V2 Trading Archive Sync",
            "",
            f"- status: `{payload.get('status')}`",
            f"- target_date: `{payload.get('target_date')}`",
            f"- base_as_of_date: `{payload.get('base_as_of_date')}`",
            f"- request_count: **{payload.get('request_count')}**",
            f"- done: **{payload.get('done')}**",
            f"- rows: **{payload.get('row_count')}**",
            f"- data HTTP attempts: **{payload.get('http_attempts')}**",
            f"- batch_fingerprint: `{payload.get('batch_fingerprint')}`",
            f"- error: `{payload.get('error') or ''}`",
            "",
        ]
    )


def _resolve_base_provider(project_root, state):
    if state is not None and state.get("base_provider_snapshot_fingerprint"):
        pinned = find_ready_provider_snapshot_by_fingerprint(
            project_root, str(state["base_provider_snapshot_fingerprint"])
        )
        if pinned is None:
            raise RuntimeError("Trading V2 已 pin 的 Provider Snapshot 不存在或不再通過 current registry 驗證")
        return pinned
    latest = find_latest_ready_provider_snapshot(project_root)
    return latest


def sync_market_data_v2_trading_archive(
    *,
    project_root,
    target_date: str,
    token: str,
    output_dir,
    client: FinMindHttpClient | None = None,
    sink=None,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, object]:
    """Sync all included V2 archive datasets without changing execution-critical CSV truth."""

    root = Path(project_root).resolve()
    policy = get_market_data_trading_sync_policy()
    if not policy.enabled:
        return {
            "status": "DISABLED",
            "target_date": str(target_date),
            "execution_blocking": False,
            "request_count": 0,
            "done": 0,
            "row_count": 0,
        }

    previous = load_trading_market_data_v2_state(root, required=False)
    provider = _resolve_base_provider(root, previous)
    if provider is None:
        return {
            "status": TRADING_V2_ARCHIVE_STATUS_NOT_BOOTSTRAPPED,
            "target_date": str(target_date),
            "execution_blocking": False,
            "request_count": 0,
            "done": 0,
            "row_count": 0,
            "error": None,
        }
    provider_path, provider_payload = provider
    previous_sync_date = None if previous is None else previous.get("latest_sync_target_date")
    manifest = build_trading_sync_request_manifest(
        specs=get_market_dataset_specs(included_only=True),
        provider_snapshot=provider_payload,
        target_date=str(target_date),
        previous_sync_date=str(previous_sync_date) if previous_sync_date else None,
        policy=policy,
    )
    ledger_path = resolve_trading_market_data_v2_ledger_path(root, manifest.manifest_fingerprint)
    ledger = MarketDataJobLedger(ledger_path, workload_namespace="market_data_v2_trading_sync")
    storage = sink or MarketDataTradingStorageSink(project_root=root, manifest=manifest)
    readiness = getattr(storage, "validate_activation_readiness", None)
    if callable(readiness):
        readiness()
    http = client or FinMindHttpClient(token=token)
    data_request_count_before = int(getattr(http, "data_request_count", 0))
    usage_request_count_before = int(getattr(http, "usage_request_count", 0))
    executor = MarketDataBootstrapExecutor(
        ledger=ledger,
        client=http,
        policy=get_market_data_execution_policy(),
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        blocking_waits=False,
    )
    summary = executor.run(manifest=manifest, sink=storage)
    quota_snapshot = executor.quota_progress_snapshot()
    workload_id = ledger.workload_id_for_manifest(manifest)
    committed = ledger.list_committed_artifacts(workload_id) if summary.done else ()
    row_count = sum(int(item.row_count) for item in committed)
    finished_at = now_fn() if now_fn is not None else datetime.now().astimezone()
    dataset_state = None

    if summary.workload_status == WORKLOAD_DONE and summary.done == manifest.total_requests:
        from services.trading.market_data_dataset_state import record_market_data_sync_success

        observation_reader = getattr(storage, "dataset_observations", None)
        observations = observation_reader() if callable(observation_reader) else {}
        dataset_state = record_market_data_sync_success(
            root,
            target_date=manifest.as_of_date,
            finished_at=finished_at,
            observations=observations,
            attempted_datasets={request.dataset for request in manifest.requests},
        )
        batch_artifact_fingerprint = canonical_json_sha256(
            [
                {
                    "request_id": item.request_id,
                    "row_count": int(item.row_count),
                    "content_sha256": item.content_sha256,
                }
                for item in committed
            ]
        )
        state = publish_trading_market_data_v2_state(
            root,
            {
                "status": TRADING_V2_ARCHIVE_STATUS_SYNCED,
                "base_provider_snapshot_fingerprint": manifest.base_provider_snapshot_fingerprint,
                "base_provider_manifest_fingerprint": manifest.base_provider_manifest_fingerprint,
                "base_provider_snapshot_path": project_relative_display_path(provider_path, project_root=root),
                "base_as_of_date": manifest.base_as_of_date,
                "latest_sync_target_date": manifest.as_of_date,
                "latest_batch_fingerprint": manifest.manifest_fingerprint,
                "latest_batch_artifact_fingerprint": batch_artifact_fingerprint,
                "latest_request_count": manifest.total_requests,
                "latest_row_count": row_count,
                "latest_quota_user_count": quota_snapshot.get("quota_user_count"),
                "latest_quota_limit": quota_snapshot.get("quota_limit"),
                "latest_quota_remaining": quota_snapshot.get("quota_remaining"),
                "latest_quota_usable_remaining": quota_snapshot.get("quota_usable_remaining"),
                "latest_quota_observed_at": finished_at.isoformat() if quota_snapshot.get("quota_limit") is not None else None,
                "last_error": None,
                "updated_at": finished_at.isoformat(),
            },
        )
        status = TRADING_V2_ARCHIVE_STATUS_SYNCED
        error = None
    else:
        error = (
            f"Trading V2 sync 未完成: status={summary.workload_status}, "
            f"done={summary.done}/{summary.total}, blocked={summary.blocked}"
        )
        base_state = dict(previous or {})
        state = publish_trading_market_data_v2_state(
            root,
            {
                **{key: value for key, value in base_state.items() if key not in {"schema_version", "state_fingerprint"}},
                "status": TRADING_V2_ARCHIVE_STATUS_STALE,
                "base_provider_snapshot_fingerprint": manifest.base_provider_snapshot_fingerprint,
                "base_provider_manifest_fingerprint": manifest.base_provider_manifest_fingerprint,
                "base_provider_snapshot_path": project_relative_display_path(provider_path, project_root=root),
                "base_as_of_date": manifest.base_as_of_date,
                "latest_quota_user_count": quota_snapshot.get("quota_user_count"),
                "latest_quota_limit": quota_snapshot.get("quota_limit"),
                "latest_quota_remaining": quota_snapshot.get("quota_remaining"),
                "latest_quota_usable_remaining": quota_snapshot.get("quota_usable_remaining"),
                "latest_quota_observed_at": finished_at.isoformat() if quota_snapshot.get("quota_limit") is not None else base_state.get("latest_quota_observed_at"),
                "last_error": error,
                "updated_at": finished_at.isoformat(),
            },
        )
        status = TRADING_V2_ARCHIVE_STATUS_STALE

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "execution_blocking": bool(policy.execution_fail_closed),
        "target_date": manifest.as_of_date,
        "base_as_of_date": manifest.base_as_of_date,
        "base_provider_snapshot_fingerprint": manifest.base_provider_snapshot_fingerprint,
        "batch_fingerprint": manifest.manifest_fingerprint,
        "request_count": manifest.total_requests,
        "done": summary.done,
        "row_count": row_count,
        "http_attempts": summary.http_attempts,
        "process_data_requests": int(getattr(http, "data_request_count", 0)) - data_request_count_before,
        "process_usage_requests": int(getattr(http, "usage_request_count", 0)) - usage_request_count_before,
        "quota_user_count": quota_snapshot.get("quota_user_count"),
        "quota_limit": quota_snapshot.get("quota_limit"),
        "quota_remaining": quota_snapshot.get("quota_remaining"),
        "quota_usable_remaining": quota_snapshot.get("quota_usable_remaining"),
        "error": error,
        "state_fingerprint": state.get("state_fingerprint"),
        "dataset_state_fingerprint": dataset_state.get("state_fingerprint") if dataset_state else None,
        "ledger_path": project_relative_display_path(ledger_path, project_root=root),
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = finished_at.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{TRADING_SYNC_REPORT_PREFIX}{stamp}.json"
    md_path = out_dir / f"{TRADING_SYNC_REPORT_PREFIX}{stamp}.md"
    atomic_write_json(json_path, payload)
    atomic_write_text(md_path, _markdown(payload))
    return {
        **payload,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


def sync_market_data_v2_due_datasets(
    *,
    project_root,
    target_date: str,
    token: str,
    due_datasets: Iterable[str],
    client: FinMindHttpClient | None = None,
    sink=None,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    progress_fn: Callable[[dict[str, object]], None] | None = None,
    quota_wait_fn: Callable[[dict[str, object]], None] | None = None,
    force_refresh_current_target: bool = False,
    refresh_token: str | None = None,
) -> dict[str, object]:
    """Execute one non-blocking partial V2 batch for locally planned due datasets.

    This function never promotes the global archive state by itself.  It records
    dataset success only when every logical request for that dataset committed;
    the caller owns retry/defer state and overall rollup.
    """

    root = Path(project_root).resolve()
    selected = tuple(sorted({str(item) for item in due_datasets}))
    if not selected:
        return {
            "status": "NO_DUE",
            "target_date": str(target_date),
            "request_count": 0,
            "done": 0,
            "row_count": 0,
            "completed_datasets": (),
            "incomplete_datasets": (),
            "process_data_requests": 0,
            "process_usage_requests": 0,
        }

    policy = get_market_data_trading_sync_policy()
    previous = load_trading_market_data_v2_state(root, required=False)
    provider = _resolve_base_provider(root, previous)
    if provider is None:
        return {
            "status": TRADING_V2_ARCHIVE_STATUS_NOT_BOOTSTRAPPED,
            "target_date": str(target_date),
            "request_count": 0,
            "done": 0,
            "row_count": 0,
            "completed_datasets": (),
            "incomplete_datasets": selected,
            "process_data_requests": 0,
            "process_usage_requests": 0,
        }
    _provider_path, provider_payload = provider

    from services.trading.market_data_dataset_state import load_market_data_dataset_state, record_market_data_sync_success

    dynamic = load_market_data_dataset_state(root, required=False)
    rows = dict((dynamic or {}).get("datasets") or {})
    recovery_floor = date.fromisoformat(str(target_date)) - timedelta(days=int(policy.recent_repair_calendar_days))
    base_as_of = date.fromisoformat(str(provider_payload.get("as_of_date")))
    recovery_anchor = max(base_as_of, recovery_floor).isoformat()
    provider_latest = (
        _merge_latest_available_provider_snapshot_evidence(
            project_root=root,
            provider_payload=provider_payload,
            selected_datasets=selected,
            dynamic_rows=rows,
        )
        if force_refresh_current_target
        else {}
    )
    previous_ready: dict[str, str | None] = {}
    previous_latest: dict[str, dict[str, object]] = {}
    for dataset in selected:
        row = dict(rows.get(dataset) or {})
        operational_latest = {
            "latest_data_date": str(row.get("latest_data_date") or "").strip() or None,
            "latest_data_date_by_data_id": dict(row.get("latest_data_date_by_data_id") or {}),
        }
        previous_latest[dataset] = _merge_latest_date_evidence(
            operational_latest,
            provider_latest.get(dataset),
        )
        if has_current_market_data_dataset_validation(row):
            previous_ready[dataset] = str(row.get("last_ready_target_date") or "").strip() or None
        else:
            # Legacy/incomplete validation evidence may be used only as a bounded
            # resume hint.  Re-query at least the configured recent window so a
            # pre-contract target date cannot silently authorize current data.
            previous_ready[dataset] = recovery_anchor
    manifest = build_trading_sync_request_manifest(
        specs=get_market_dataset_specs(included_only=True),
        provider_snapshot=provider_payload,
        target_date=str(target_date),
        previous_sync_date=None,
        policy=policy,
        selected_datasets=selected,
        previous_ready_dates_by_dataset=previous_ready,
        previous_latest_data_dates_by_dataset=previous_latest,
        force_refresh_current_target=bool(force_refresh_current_target),
        refresh_token=refresh_token,
    )
    ledger_path = resolve_trading_market_data_v2_ledger_path(root, manifest.manifest_fingerprint)
    ledger = MarketDataJobLedger(ledger_path, workload_namespace="market_data_v2_trading_auto_due")
    storage = sink or MarketDataTradingStorageSink(project_root=root, manifest=manifest)
    readiness = getattr(storage, "validate_activation_readiness", None)
    if callable(readiness):
        readiness()
    http = client or FinMindHttpClient(token=token)
    data_request_count_before = int(getattr(http, "data_request_count", 0))
    usage_request_count_before = int(getattr(http, "usage_request_count", 0))
    if progress_fn is not None:
        progress_fn(
            {
                "kind": "PLAN",
                "target_date": manifest.as_of_date,
                "dataset_count": len(selected),
                "total": manifest.total_requests,
                "force_refresh": bool(force_refresh_current_target),
                **_request_geometry_summary(manifest.requests),
            }
        )
    executor = MarketDataBootstrapExecutor(
        ledger=ledger,
        client=http,
        policy=get_market_data_execution_policy(),
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        blocking_waits=False,
        quota_wait_observer=quota_wait_fn,
        progress_observer=progress_fn,
        force_uncached_data_requests=bool(force_refresh_current_target),
    )
    summary = executor.run(manifest=manifest, sink=storage)
    quota_snapshot = executor.quota_progress_snapshot()
    workload_id = ledger.workload_id_for_manifest(manifest)
    committed = ledger.list_committed_artifacts(workload_id)
    committed_ids = {item.request_id for item in committed}
    request_ids_by_dataset: dict[str, set[str]] = {}
    for request in manifest.requests:
        request_ids_by_dataset.setdefault(request.dataset, set()).add(request.request_id)
    completed = tuple(
        sorted(
            dataset
            for dataset, request_ids in request_ids_by_dataset.items()
            if request_ids and request_ids <= committed_ids
        )
    )
    incomplete = tuple(sorted(set(selected) - set(completed)))
    finished_at = now_fn() if now_fn is not None else datetime.now().astimezone()
    dataset_state = None
    if completed:
        observation_reader = getattr(storage, "dataset_observations", None)
        observations = observation_reader() if callable(observation_reader) else {}
        dataset_state = record_market_data_sync_success(
            root,
            target_date=manifest.as_of_date,
            finished_at=finished_at,
            observations=observations,
            attempted_datasets=completed,
        )
    verification_reader = getattr(storage, "verification_summary", None)
    verification = verification_reader() if callable(verification_reader) else {}
    return {
        "status": str(summary.workload_status),
        "target_date": manifest.as_of_date,
        "batch_fingerprint": manifest.manifest_fingerprint,
        "refresh_token": manifest.refresh_token,
        "force_refresh": bool(force_refresh_current_target),
        "request_count": manifest.total_requests,
        "done": summary.done,
        "blocked": summary.blocked,
        "retryable": summary.retryable,
        "row_count": sum(int(item.row_count) for item in committed),
        "completed_datasets": completed,
        "incomplete_datasets": incomplete,
        "process_data_requests": int(getattr(http, "data_request_count", 0)) - data_request_count_before,
        "process_usage_requests": int(getattr(http, "usage_request_count", 0)) - usage_request_count_before,
        "quota_user_count": quota_snapshot.get("quota_user_count"),
        "quota_limit": quota_snapshot.get("quota_limit"),
        "quota_remaining": quota_snapshot.get("quota_remaining"),
        "quota_usable_remaining": quota_snapshot.get("quota_usable_remaining"),
        "dataset_state_fingerprint": dataset_state.get("state_fingerprint") if dataset_state else None,
        "ledger_path": project_relative_display_path(ledger_path, project_root=root),
        "verification": verification,
        **_request_geometry_summary(manifest.requests),
    }


__all__ = ["TRADING_SYNC_REPORT_PREFIX", "sync_market_data_v2_trading_archive", "sync_market_data_v2_due_datasets"]
