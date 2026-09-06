"""Atomic Parquet storage for the isolated Trading Market Data V2 overlay batches."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Callable

import pandas as pd

from core.file_integrity import atomic_write_json, compute_file_sha256, load_json_strict
from core.market_data_storage_contract import (
    MARKET_DATA_DATASET_SCHEMA_FILENAME,
    MarketDataCommitError,
    MarketDataCommitReceipt,
    build_frame_schema_payload,
    resolve_market_data_dataset_dir,
)
from core.market_data_storage_policy import MarketDataStoragePolicy, get_market_data_storage_policy
from core.market_data_trading_storage_contract import (
    TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME,
    build_trading_request_metadata,
    build_trading_sync_batch_manifest_payload,
    resolve_trading_market_data_v2_batch_dir,
    resolve_trading_market_data_v2_dataset_dir,
    resolve_trading_market_data_v2_request_path,
    resolve_trading_market_data_v2_root,
)
from core.market_data_trading_sync import TradingSyncRequestManifest
from services.downloader.market_data_storage import (
    ParquetCodec,
    PyArrowParquetCodec,
    commit_market_data_parquet_frame,
    validate_market_data_raw_frame,
)


class MarketDataTradingStorageSink:
    def __init__(
        self,
        *,
        project_root,
        manifest: TradingSyncRequestManifest,
        policy: MarketDataStoragePolicy | None = None,
        codec: ParquetCodec | None = None,
        disk_usage_fn: Callable | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.manifest = manifest
        self.policy = policy or get_market_data_storage_policy()
        self.codec = codec or PyArrowParquetCodec()
        self.disk_usage_fn = disk_usage_fn or shutil.disk_usage
        self.batch_dir = resolve_trading_market_data_v2_batch_dir(self.project_root, manifest.manifest_fingerprint)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self._requests = {request.request_id: request for request in manifest.requests}
        if len(self._requests) != manifest.total_requests:
            raise ValueError("Trading V2 storage manifest request identity 重複")
        self._ensure_batch_manifest()

    def _ensure_batch_manifest(self) -> None:
        path = self.batch_dir / TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME
        expected = build_trading_sync_batch_manifest_payload(self.manifest)
        if path.is_file():
            try:
                actual = load_json_strict(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise MarketDataCommitError(
                    f"既有 Trading V2 batch manifest 無法驗證: {type(exc).__name__}: {exc}"
                ) from exc
            if actual != expected:
                raise MarketDataCommitError("既有 Trading V2 batch manifest identity 不一致")
            return
        atomic_write_json(path, expected)

    def _validate_request(self, request) -> None:
        canonical = self._requests.get(request.request_id)
        if canonical is None or canonical != request:
            raise MarketDataCommitError("Trading V2 storage 收到不屬於目前 sync manifest 的 request")

    def _schema_owner_path(self, dataset: str) -> Path:
        root = resolve_trading_market_data_v2_root(self.project_root) / "schemas"
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{dataset}.json"

    def _base_schema_path(self, dataset: str) -> Path:
        return resolve_market_data_dataset_dir(
            self.project_root,
            self.manifest.base_provider_manifest_fingerprint,
            dataset,
        ) / MARKET_DATA_DATASET_SCHEMA_FILENAME

    def _ensure_schema(self, request, frame: pd.DataFrame) -> dict[str, object]:
        schema = build_frame_schema_payload(frame)
        if len(frame.columns) == 0:
            return schema
        expected_columns = list(schema["columns"])
        local_path = self._schema_owner_path(request.dataset)
        base_path = self._base_schema_path(request.dataset)
        source_path = local_path if local_path.is_file() else base_path
        if source_path.is_file():
            try:
                existing = load_json_strict(source_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise MarketDataCommitError(
                    f"{request.dataset} schema owner 無法驗證: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(existing, dict) or list(existing.get("columns") or []) != expected_columns:
                raise MarketDataCommitError(
                    f"{request.dataset} provider schema drift: existing={existing.get('columns') if isinstance(existing, dict) else None}, current={expected_columns}"
                )
            if not local_path.is_file():
                atomic_write_json(local_path, dict(existing))
            return schema
        atomic_write_json(
            local_path,
            {
                "dataset": request.dataset,
                "columns": expected_columns,
                "column_fingerprint": schema["column_fingerprint"],
            },
        )
        return schema

    def validate_activation_readiness(self) -> None:
        validate_runtime = getattr(self.codec, "validate_runtime", None)
        if callable(validate_runtime):
            validate_runtime()
        root = resolve_trading_market_data_v2_root(self.project_root)
        root.mkdir(parents=True, exist_ok=True)
        free = int(getattr(self.disk_usage_fn(root), "free"))
        required = int(self.policy.minimum_free_bytes + self.policy.minimum_staging_headroom_bytes)
        if free < required:
            raise MarketDataCommitError(
                f"Trading Market Data V2 startup free-space gate: free={free} bytes < required={required} bytes"
            )

    def _expected_identity(self, request) -> dict[str, object]:
        return {
            "batch_fingerprint": self.manifest.manifest_fingerprint,
            "registry_fingerprint": self.manifest.registry_fingerprint,
            "base_provider_snapshot_fingerprint": self.manifest.base_provider_snapshot_fingerprint,
            "base_provider_manifest_fingerprint": self.manifest.base_provider_manifest_fingerprint,
            "request_id": request.request_id,
            "dataset": request.dataset,
            "bootstrap_mode": request.bootstrap_mode,
            "data_id": request.data_id,
            "start_date": request.start_date,
            "end_date": request.end_date,
        }

    def recover_committed(self, request) -> MarketDataCommitReceipt | None:
        self._validate_request(request)
        path = resolve_trading_market_data_v2_request_path(
            self.project_root, self.manifest.manifest_fingerprint, request
        )
        if not path.is_file():
            return None
        inspection = self.codec.inspect(path)
        metadata = inspection.metadata
        expected = self._expected_identity(request)
        actual = {key: metadata.get(key) for key in expected}
        if actual != expected:
            raise MarketDataCommitError(
                f"既有 Trading V2 Parquet request identity 不一致: actual={actual}, expected={expected}"
            )
        if int(metadata.get("row_count", -1)) != inspection.row_count:
            raise MarketDataCommitError("既有 Trading V2 Parquet row_count metadata 與實體不一致")
        return MarketDataCommitReceipt(
            committed=True,
            row_count=inspection.row_count,
            content_sha256=compute_file_sha256(path),
        )

    def __call__(self, request, frame: pd.DataFrame) -> MarketDataCommitReceipt:
        self._validate_request(request)
        validate_market_data_raw_frame(frame)
        existing = self.recover_committed(request)
        if existing is not None:
            if existing.row_count != len(frame):
                raise MarketDataCommitError(
                    f"既有 Trading V2 artifact rows={existing.row_count} 與重新取得 rows={len(frame)} 不一致"
                )
            return existing
        schema = self._ensure_schema(request, frame)
        metadata = build_trading_request_metadata(
            manifest=self.manifest,
            request=request,
            row_count=len(frame),
            schema_payload=schema,
        )
        path = resolve_trading_market_data_v2_request_path(
            self.project_root, self.manifest.manifest_fingerprint, request
        )
        return commit_market_data_parquet_frame(
            frame=frame,
            final_path=path,
            expected_metadata=metadata,
            policy=self.policy,
            codec=self.codec,
            disk_usage_fn=self.disk_usage_fn,
        )


__all__ = ["MarketDataTradingStorageSink"]
