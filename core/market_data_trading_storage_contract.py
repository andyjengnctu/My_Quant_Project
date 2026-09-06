"""Filesystem/identity contract for the isolated Trading Market Data V2 archive overlay."""
from __future__ import annotations

from pathlib import Path
import re

from core.file_integrity import canonical_json_sha256
from core.market_data_bootstrap_requests import BootstrapHttpRequest
from core.market_data_trading_sync import TradingSyncRequestManifest

TRADING_MARKET_DATA_V2_RELATIVE_ROOT = Path("data") / "trading" / "market_data_v2"
TRADING_MARKET_DATA_V2_STATE_RELATIVE_PATH = Path("state") / "trading" / "market_data_v2" / "archive_state.json"
TRADING_MARKET_DATA_V2_LEDGER_RELATIVE_ROOT = Path("state") / "trading" / "market_data_v2" / "ledgers"
TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME = "batch_manifest.json"
TRADING_MARKET_DATA_V2_SCHEMA_VERSION = 1
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_DATASET_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _hex64(value: str, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _HEX64_RE.fullmatch(text):
        raise ValueError(f"{field} 必須是 SHA256 hex")
    return text


def _dataset(value: str) -> str:
    text = str(value or "").strip()
    if not _DATASET_RE.fullmatch(text):
        raise ValueError(f"不合法 dataset path identity: {value!r}")
    return text


def resolve_trading_market_data_v2_root(project_root) -> Path:
    return Path(project_root).resolve() / TRADING_MARKET_DATA_V2_RELATIVE_ROOT


def resolve_trading_market_data_v2_state_path(project_root) -> Path:
    return Path(project_root).resolve() / TRADING_MARKET_DATA_V2_STATE_RELATIVE_PATH


def resolve_trading_market_data_v2_batch_dir(project_root, batch_fingerprint: str) -> Path:
    return resolve_trading_market_data_v2_root(project_root) / "batches" / _hex64(batch_fingerprint, field="batch_fingerprint")


def resolve_trading_market_data_v2_dataset_dir(project_root, batch_fingerprint: str, dataset: str) -> Path:
    return resolve_trading_market_data_v2_batch_dir(project_root, batch_fingerprint) / "datasets" / _dataset(dataset)


def resolve_trading_market_data_v2_request_path(project_root, batch_fingerprint: str, request: BootstrapHttpRequest) -> Path:
    request_id = _hex64(request.request_id, field="request_id")
    return resolve_trading_market_data_v2_dataset_dir(project_root, batch_fingerprint, request.dataset) / f"{request_id}.parquet"


def resolve_trading_market_data_v2_ledger_path(project_root, batch_fingerprint: str) -> Path:
    return Path(project_root).resolve() / TRADING_MARKET_DATA_V2_LEDGER_RELATIVE_ROOT / f"{_hex64(batch_fingerprint, field='batch_fingerprint')}.sqlite3"


def build_trading_sync_batch_manifest_payload(manifest: TradingSyncRequestManifest) -> dict[str, object]:
    identity = {
        "schema_version": TRADING_MARKET_DATA_V2_SCHEMA_VERSION,
        "role": "trading_market_data_v2_archive_sync_batch",
        "status": "READY",
        "target_date": manifest.as_of_date,
        "base_as_of_date": manifest.base_as_of_date,
        "previous_sync_date": manifest.previous_sync_date,
        "base_provider_snapshot_fingerprint": manifest.base_provider_snapshot_fingerprint,
        "base_provider_manifest_fingerprint": manifest.base_provider_manifest_fingerprint,
        "registry_fingerprint": manifest.registry_fingerprint,
        "batch_fingerprint": manifest.manifest_fingerprint,
        "request_count": manifest.total_requests,
        "request_ids": [request.request_id for request in manifest.requests],
    }
    return {**identity, "identity_fingerprint": canonical_json_sha256(identity)}


def build_trading_request_metadata(
    *,
    manifest: TradingSyncRequestManifest,
    request: BootstrapHttpRequest,
    row_count: int,
    schema_payload: dict[str, object],
) -> dict[str, object]:
    return {
        "storage_layout_version": TRADING_MARKET_DATA_V2_SCHEMA_VERSION,
        "role": "trading_market_data_v2_archive_sync",
        "batch_fingerprint": manifest.manifest_fingerprint,
        "registry_fingerprint": manifest.registry_fingerprint,
        "base_provider_snapshot_fingerprint": manifest.base_provider_snapshot_fingerprint,
        "base_provider_manifest_fingerprint": manifest.base_provider_manifest_fingerprint,
        "request_id": request.request_id,
        "dataset": request.dataset,
        "bootstrap_mode": request.bootstrap_mode,
        "data_id": request.data_id,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "row_count": int(row_count),
        "column_fingerprint": schema_payload["column_fingerprint"],
        "schema_fingerprint": schema_payload["schema_fingerprint"],
    }


__all__ = [
    "TRADING_MARKET_DATA_V2_RELATIVE_ROOT",
    "TRADING_MARKET_DATA_V2_STATE_RELATIVE_PATH",
    "TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME",
    "TRADING_MARKET_DATA_V2_SCHEMA_VERSION",
    "resolve_trading_market_data_v2_root",
    "resolve_trading_market_data_v2_state_path",
    "resolve_trading_market_data_v2_batch_dir",
    "resolve_trading_market_data_v2_dataset_dir",
    "resolve_trading_market_data_v2_request_path",
    "resolve_trading_market_data_v2_ledger_path",
    "build_trading_sync_batch_manifest_payload",
    "build_trading_request_metadata",
]
