"""Stable filesystem and metadata contract for Market Data V2 bootstrap storage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from core.file_integrity import canonical_json_sha256
from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest
from core.market_data_storage_policy import MarketDataStoragePolicy

MARKET_DATA_STORAGE_LAYOUT_VERSION = 1
MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT = Path("data") / "market_data_v2" / "bootstrap"
MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME = "bootstrap_manifest.json"
MARKET_DATA_BOOTSTRAP_LEDGER_FILENAME = "bootstrap_ledger.sqlite3"
MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME = "provider_snapshot_manifest.json"
MARKET_DATA_DATASET_SCHEMA_FILENAME = "dataset_schema.json"
MARKET_DATA_PARQUET_METADATA_KEY = "my_quant_market_data_v2"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_DATASET_RE = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class MarketDataCommitReceipt:
    committed: bool
    row_count: int
    content_sha256: str | None = None


class MarketDataCommitError(RuntimeError):
    pass


def _require_hex64(value: str, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _HEX64_RE.fullmatch(text):
        raise ValueError(f"{field} 必須是 64-byte hex SHA256 identity")
    return text


def _require_dataset_name(value: str) -> str:
    text = str(value or "").strip()
    if not _DATASET_RE.fullmatch(text):
        raise ValueError(f"不合法的 Market Data dataset path identity: {value!r}")
    return text


def resolve_market_data_bootstrap_archive_dir(project_root, manifest_fingerprint: str) -> Path:
    fingerprint = _require_hex64(manifest_fingerprint, field="manifest_fingerprint")
    return Path(project_root).resolve() / MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT / fingerprint


def resolve_market_data_bootstrap_ledger_path(project_root, manifest_fingerprint: str) -> Path:
    return resolve_market_data_bootstrap_archive_dir(project_root, manifest_fingerprint) / MARKET_DATA_BOOTSTRAP_LEDGER_FILENAME


def resolve_market_data_provider_snapshot_path(project_root, manifest_fingerprint: str) -> Path:
    return resolve_market_data_bootstrap_archive_dir(project_root, manifest_fingerprint) / MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME


def resolve_market_data_dataset_dir(project_root, manifest_fingerprint: str, dataset: str) -> Path:
    return resolve_market_data_bootstrap_archive_dir(project_root, manifest_fingerprint) / "datasets" / _require_dataset_name(dataset)


def resolve_market_data_request_parquet_path(
    project_root,
    manifest_fingerprint: str,
    request: BootstrapHttpRequest,
) -> Path:
    request_id = _require_hex64(request.request_id, field="request_id")
    return resolve_market_data_dataset_dir(project_root, manifest_fingerprint, request.dataset) / f"{request_id}.parquet"


def build_frame_schema_payload(frame) -> dict[str, object]:
    columns = [str(column) for column in frame.columns]
    dtypes = [str(dtype) for dtype in frame.dtypes]
    return {
        "columns": columns,
        "dtypes": dtypes,
        "schema_fingerprint": canonical_json_sha256({"columns": columns, "dtypes": dtypes}),
        "column_fingerprint": canonical_json_sha256(columns),
    }


def build_bootstrap_storage_manifest_payload(
    manifest: BootstrapRequestManifest,
    policy: MarketDataStoragePolicy,
) -> dict[str, object]:
    return {
        "storage_layout_version": MARKET_DATA_STORAGE_LAYOUT_VERSION,
        "format": policy.format,
        "compression": policy.compression,
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "registry_fingerprint": manifest.registry_fingerprint,
        "as_of_date": manifest.as_of_date,
        "full_range_start": manifest.full_range_start,
        "historical_instrument_count": manifest.historical_instrument_count,
        "total_requests": manifest.total_requests,
    }


def build_request_parquet_metadata(
    *,
    manifest: BootstrapRequestManifest,
    request: BootstrapHttpRequest,
    row_count: int,
    schema_payload: dict[str, object],
) -> dict[str, object]:
    return {
        "storage_layout_version": MARKET_DATA_STORAGE_LAYOUT_VERSION,
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "registry_fingerprint": manifest.registry_fingerprint,
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
    "MARKET_DATA_STORAGE_LAYOUT_VERSION",
    "MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT",
    "MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME",
    "MARKET_DATA_BOOTSTRAP_LEDGER_FILENAME",
    "MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME",
    "MARKET_DATA_DATASET_SCHEMA_FILENAME",
    "MARKET_DATA_PARQUET_METADATA_KEY",
    "MarketDataCommitReceipt",
    "MarketDataCommitError",
    "resolve_market_data_bootstrap_archive_dir",
    "resolve_market_data_bootstrap_ledger_path",
    "resolve_market_data_provider_snapshot_path",
    "resolve_market_data_dataset_dir",
    "resolve_market_data_request_parquet_path",
    "build_frame_schema_payload",
    "build_bootstrap_storage_manifest_payload",
    "build_request_parquet_metadata",
]
