"""Non-blocking Trading Market Data V2 archive synchronization service."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, atomic_write_text, canonical_json_sha256
from core.market_data_dataset_registry import get_market_dataset_specs
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

TRADING_SYNC_REPORT_PREFIX = "market_data_v2_trading_sync_"


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
    executor = MarketDataBootstrapExecutor(
        ledger=ledger,
        client=http,
        policy=get_market_data_execution_policy(),
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        blocking_waits=False,
    )
    summary = executor.run(manifest=manifest, sink=storage)
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
        "process_data_requests": int(http.data_request_count),
        "process_usage_requests": int(http.usage_request_count),
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


__all__ = ["TRADING_SYNC_REPORT_PREFIX", "sync_market_data_v2_trading_archive"]
