"""Trading-owned Market Data V2 archive state and neutral provider-snapshot resolution."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.console_report import project_relative_display_path
from typing import Any

from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.market_data_bootstrap_requests import build_registry_fingerprint
from core.market_data_dataset_registry import get_market_dataset_specs
from core.market_data_freshness_contract import build_market_data_freshness_contract_summary
from core.market_data_provider_snapshot import provider_snapshot_identity_from_payload
from core.market_data_storage_contract import MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT, MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME
from core.market_data_trading_storage_contract import (
    TRADING_MARKET_DATA_V2_SCHEMA_VERSION,
    resolve_trading_market_data_v2_state_path,
)

TRADING_V2_ARCHIVE_STATUS_SYNCED = "SYNCED"
TRADING_V2_ARCHIVE_STATUS_NOT_BOOTSTRAPPED = "NOT_BOOTSTRAPPED"
TRADING_V2_ARCHIVE_STATUS_STALE = "STALE"


def _validate_provider_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or str(payload.get("status") or "") != "READY":
        raise ValueError("Market Data V2 provider snapshot 尚未 READY")
    identity = provider_snapshot_identity_from_payload(payload)
    if str(payload.get("snapshot_fingerprint") or "") != canonical_json_sha256(identity):
        raise ValueError("Market Data V2 provider snapshot fingerprint 不一致")
    current_registry = build_registry_fingerprint(get_market_dataset_specs(included_only=True))
    if str(payload.get("registry_fingerprint") or "") != current_registry:
        raise ValueError("Market Data V2 provider snapshot registry 與 current registry 已 drift")
    return payload


def find_latest_ready_provider_snapshot(project_root) -> tuple[Path, dict[str, Any]] | None:
    root = Path(project_root).resolve()
    base = root / MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT
    candidates: list[tuple[str, str, Path, dict[str, Any]]] = []
    if not base.is_dir():
        return None
    for path in base.glob(f"*/{MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME}"):
        try:
            payload = _validate_provider_snapshot(load_json_strict(path))
        except (OSError, ValueError, TypeError):
            continue
        candidates.append(
            (
                str(payload.get("as_of_date") or ""),
                str(payload.get("finalized_at") or ""),
                path,
                payload,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    _as_of, _finalized, path, payload = candidates[-1]
    return path, payload




def find_ready_provider_snapshot_by_fingerprint(project_root, snapshot_fingerprint: str) -> tuple[Path, dict[str, Any]] | None:
    wanted = str(snapshot_fingerprint or "").strip()
    if len(wanted) != 64:
        raise ValueError("provider snapshot fingerprint 不合法")
    root = Path(project_root).resolve()
    base = root / MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT
    if not base.is_dir():
        return None
    for path in base.glob(f"*/{MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME}"):
        try:
            payload = _validate_provider_snapshot(load_json_strict(path))
        except (OSError, ValueError, TypeError):
            continue
        if str(payload.get("snapshot_fingerprint") or "") == wanted:
            return path, payload
    return None


def load_trading_market_data_v2_state(project_root, *, required: bool = False) -> dict[str, Any] | None:
    path = resolve_trading_market_data_v2_state_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Trading Market Data V2 archive state 尚未建立")
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("Trading Market Data V2 archive state 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_MARKET_DATA_V2_SCHEMA_VERSION:
        raise ValueError("Trading Market Data V2 archive state schema 不相容")
    core = {key: value for key, value in payload.items() if key != "state_fingerprint"}
    if str(payload.get("state_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading Market Data V2 archive state fingerprint 不一致")
    return payload


def publish_trading_market_data_v2_state(project_root, payload: dict[str, Any]) -> dict[str, Any]:
    state = {
        "schema_version": TRADING_MARKET_DATA_V2_SCHEMA_VERSION,
        **dict(payload),
    }
    state["state_fingerprint"] = canonical_json_sha256(state)
    atomic_write_json(resolve_trading_market_data_v2_state_path(project_root), state)
    return state


def publish_trading_market_data_v2_failure(
    project_root,
    *,
    target_date: str,
    error: str,
) -> dict[str, Any] | None:
    """Persist a non-blocking sidecar failure without advancing the last synced date."""

    previous = load_trading_market_data_v2_state(project_root, required=False)
    if previous is not None and previous.get("base_provider_snapshot_fingerprint"):
        provider = find_ready_provider_snapshot_by_fingerprint(
            project_root, str(previous["base_provider_snapshot_fingerprint"])
        )
    else:
        provider = find_latest_ready_provider_snapshot(project_root)
    if provider is None:
        return None
    _provider_path, provider_payload = provider
    payload = {
        **{
            key: value
            for key, value in dict(previous or {}).items()
            if key not in {"schema_version", "state_fingerprint", "status", "last_error", "updated_at"}
        },
        "status": TRADING_V2_ARCHIVE_STATUS_STALE,
        "base_provider_snapshot_fingerprint": provider_payload.get("snapshot_fingerprint"),
        "base_provider_manifest_fingerprint": provider_payload.get("manifest_fingerprint"),
        "base_as_of_date": provider_payload.get("as_of_date"),
        "last_attempt_target_date": str(target_date),
        "last_error": str(error),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    return publish_trading_market_data_v2_state(project_root, payload)


def publish_trading_market_data_v2_auto_rollup(
    project_root,
    *,
    target_date: str,
    updated_at: datetime,
    batch_result: dict[str, object] | None = None,
) -> dict[str, Any] | None:
    """Roll dataset-level auto-update truth into the aggregate Trading V2 state."""

    previous = load_trading_market_data_v2_state(project_root, required=False)
    if previous is not None and previous.get("base_provider_snapshot_fingerprint"):
        provider = find_ready_provider_snapshot_by_fingerprint(
            project_root, str(previous["base_provider_snapshot_fingerprint"])
        )
        if provider is None:
            raise RuntimeError("Trading V2 auto rollup 已 pin 的 Provider Snapshot 不存在或不合法")
    else:
        provider = find_latest_ready_provider_snapshot(project_root)
    if provider is None:
        return None
    provider_path, provider_payload = provider
    from services.trading.market_data_dataset_state import load_market_data_dataset_state

    dataset_state = load_market_data_dataset_state(project_root, required=False)
    rows = dict((dataset_state or {}).get("datasets") or {})
    ready = [
        dataset
        for dataset, raw in rows.items()
        if str(dict(raw or {}).get("last_ready_target_date") or "") >= str(target_date)
    ]
    total = len(rows)
    all_ready = bool(total and len(ready) == total)
    base = {
        key: value
        for key, value in dict(previous or {}).items()
        if key not in {"schema_version", "state_fingerprint", "status", "last_error", "updated_at"}
    }
    result = dict(batch_result or {})
    payload = {
        **base,
        "status": TRADING_V2_ARCHIVE_STATUS_SYNCED if all_ready else TRADING_V2_ARCHIVE_STATUS_STALE,
        "base_provider_snapshot_fingerprint": provider_payload.get("snapshot_fingerprint"),
        "base_provider_manifest_fingerprint": provider_payload.get("manifest_fingerprint"),
        "base_provider_snapshot_path": project_relative_display_path(provider_path, project_root=Path(project_root).resolve()),
        "base_as_of_date": provider_payload.get("as_of_date"),
        "last_attempt_target_date": str(target_date),
        "latest_sync_target_date": str(target_date) if all_ready else base.get("latest_sync_target_date"),
        "latest_auto_batch_fingerprint": result.get("batch_fingerprint"),
        "latest_auto_request_count": int(result.get("request_count") or 0),
        "latest_auto_data_requests": int(result.get("process_data_requests") or 0),
        "latest_auto_usage_requests": int(result.get("process_usage_requests") or 0),
        "ready_dataset_count": len(ready),
        "pending_dataset_count": max(0, total - len(ready)),
        "last_error": None if all_ready else "dataset-level auto update 尚有未 READY dataset",
        "updated_at": updated_at.isoformat(),
    }
    return publish_trading_market_data_v2_state(project_root, payload)


def build_trading_market_data_v2_read_model(project_root) -> dict[str, Any]:
    provider = find_latest_ready_provider_snapshot(project_root)
    state = load_trading_market_data_v2_state(project_root, required=False)
    freshness_contract = build_market_data_freshness_contract_summary()
    from services.trading.market_data_dataset_state import build_market_data_dataset_state_read_model

    dataset_state_model = build_market_data_dataset_state_read_model(project_root)
    dataset_status_counts: dict[str, int] = {}
    for row in dataset_state_model.get("datasets") or []:
        status = str(row.get("status") or "")
        dataset_status_counts[status] = dataset_status_counts.get(status, 0) + 1
    if provider is None:
        return {
            "status": TRADING_V2_ARCHIVE_STATUS_NOT_BOOTSTRAPPED,
            "provider_ready": False,
            "latest_sync_target_date": None,
            "dataset_contract_count": freshness_contract["contract_count"],
            "verified_schedule_count": freshness_contract["verified_schedule_count"],
            "fallback_schedule_count": freshness_contract["fallback_schedule_count"],
            "dataset_state_ready": dataset_state_model.get("state_ready"),
            "dataset_state_updated_at": dataset_state_model.get("updated_at"),
            "dataset_status_counts": dataset_status_counts,
            "error": None,
        }
    _provider_path, provider_payload = provider
    if state is None:
        return {
            "status": TRADING_V2_ARCHIVE_STATUS_STALE,
            "provider_ready": True,
            "provider_as_of_date": provider_payload.get("as_of_date"),
            "provider_snapshot_fingerprint": provider_payload.get("snapshot_fingerprint"),
            "latest_sync_target_date": None,
            "dataset_contract_count": freshness_contract["contract_count"],
            "verified_schedule_count": freshness_contract["verified_schedule_count"],
            "fallback_schedule_count": freshness_contract["fallback_schedule_count"],
            "dataset_state_ready": dataset_state_model.get("state_ready"),
            "dataset_state_updated_at": dataset_state_model.get("updated_at"),
            "dataset_status_counts": dataset_status_counts,
            "error": "Provider Snapshot READY，但 Trading V2 archive 尚未建立 sync state",
        }
    return {
        "status": state.get("status"),
        "provider_ready": True,
        "provider_as_of_date": state.get("base_as_of_date"),
        "provider_snapshot_fingerprint": state.get("base_provider_snapshot_fingerprint"),
        "latest_sync_target_date": state.get("latest_sync_target_date"),
        "latest_batch_fingerprint": state.get("latest_batch_fingerprint"),
        "latest_request_count": state.get("latest_request_count"),
        "latest_row_count": state.get("latest_row_count"),
        "dataset_contract_count": freshness_contract["contract_count"],
        "verified_schedule_count": freshness_contract["verified_schedule_count"],
        "fallback_schedule_count": freshness_contract["fallback_schedule_count"],
        "dataset_state_ready": dataset_state_model.get("state_ready"),
        "dataset_state_updated_at": dataset_state_model.get("updated_at"),
        "dataset_status_counts": dataset_status_counts,
        "error": state.get("last_error"),
    }


__all__ = [
    "TRADING_V2_ARCHIVE_STATUS_SYNCED",
    "TRADING_V2_ARCHIVE_STATUS_NOT_BOOTSTRAPPED",
    "TRADING_V2_ARCHIVE_STATUS_STALE",
    "find_latest_ready_provider_snapshot",
    "find_ready_provider_snapshot_by_fingerprint",
    "load_trading_market_data_v2_state",
    "publish_trading_market_data_v2_state",
    "publish_trading_market_data_v2_failure",
    "publish_trading_market_data_v2_auto_rollup",
    "build_trading_market_data_v2_read_model",
]
