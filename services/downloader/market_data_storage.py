"""Atomic Parquet/ZSTD storage sink for Market Data V2 bootstrap requests."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Protocol

import pandas as pd

from core.file_integrity import (
    atomic_replace_with_retry,
    atomic_write_json,
    compute_file_sha256,
    load_json_strict,
)
from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest
from core.market_data_storage_contract import (
    MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME,
    MARKET_DATA_DATASET_SCHEMA_FILENAME,
    MARKET_DATA_PARQUET_METADATA_KEY,
    MarketDataCommitError,
    MarketDataCommitReceipt,
    build_bootstrap_storage_manifest_payload,
    build_frame_schema_payload,
    build_request_parquet_metadata,
    resolve_market_data_bootstrap_archive_dir,
    resolve_market_data_dataset_dir,
    resolve_market_data_request_parquet_path,
)
from core.market_data_storage_policy import MarketDataStoragePolicy, get_market_data_storage_policy


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParquetArtifactInspection:
    row_count: int
    columns: tuple[str, ...]
    metadata: dict[str, object]


class ParquetCodec(Protocol):
    def write(self, frame: pd.DataFrame, path: Path, *, metadata: dict[str, object], compression: str) -> None: ...
    def inspect(self, path: Path) -> ParquetArtifactInspection: ...


class PyArrowParquetCodec:
    @staticmethod
    def _modules():
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise MarketDataCommitError(
                "Market Data V2 Parquet storage 需要 pyarrow；請依 requirements lock 安裝相依套件"
            ) from exc
        return pa, pq

    def write(self, frame: pd.DataFrame, path: Path, *, metadata: dict[str, object], compression: str) -> None:
        pa, pq = self._modules()
        try:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            schema_metadata = dict(table.schema.metadata or {})
            schema_metadata[MARKET_DATA_PARQUET_METADATA_KEY.encode("utf-8")] = json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            ).encode("utf-8")
            table = table.replace_schema_metadata(schema_metadata)
            pq.write_table(table, path, compression=compression)
        except Exception as exc:
            raise MarketDataCommitError(f"Parquet write 失敗: {type(exc).__name__}: {exc}") from exc

    def validate_runtime(self) -> None:
        self._modules()

    def inspect(self, path: Path) -> ParquetArtifactInspection:
        _pa, pq = self._modules()
        try:
            parquet_file = pq.ParquetFile(path)
            schema = parquet_file.schema_arrow
            raw_metadata = dict(schema.metadata or {})
            raw_payload = raw_metadata.get(MARKET_DATA_PARQUET_METADATA_KEY.encode("utf-8"))
            if raw_payload is None:
                raise ValueError("缺少 Market Data V2 parquet identity metadata")
            payload = json.loads(raw_payload.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Market Data V2 parquet metadata root 必須為 object")
            return ParquetArtifactInspection(
                row_count=int(parquet_file.metadata.num_rows),
                columns=tuple(str(name) for name in schema.names),
                metadata=payload,
            )
        except MarketDataCommitError:
            raise
        except Exception as exc:
            raise MarketDataCommitError(f"Parquet verify 失敗: {type(exc).__name__}: {exc}") from exc




def validate_market_data_raw_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise MarketDataCommitError("Storage sink 只接受 pandas.DataFrame")
    labels = list(frame.columns)
    if any(not isinstance(column, str) for column in labels):
        raise MarketDataCommitError("FinMind raw schema column name 必須全部為字串")
    if len(set(labels)) != len(labels):
        raise MarketDataCommitError("FinMind raw schema 含 duplicate column name，禁止靜默改名")


def validate_market_data_parquet_inspection(
    inspection: ParquetArtifactInspection,
    *,
    expected_metadata: dict[str, object],
    expected_columns: tuple[str, ...],
    expected_rows: int,
) -> None:
    if inspection.row_count != int(expected_rows):
        raise MarketDataCommitError(
            f"Parquet row count 驗證失敗: actual={inspection.row_count}, expected={expected_rows}"
        )
    if inspection.columns != expected_columns:
        raise MarketDataCommitError(
            f"Parquet columns 驗證失敗: actual={inspection.columns}, expected={expected_columns}"
        )
    if inspection.metadata != expected_metadata:
        raise MarketDataCommitError("Parquet Market Data identity metadata 驗證失敗")


def commit_market_data_parquet_frame(
    *,
    frame: pd.DataFrame,
    final_path: Path,
    expected_metadata: dict[str, object],
    policy: MarketDataStoragePolicy,
    codec: ParquetCodec,
    disk_usage_fn,
) -> MarketDataCommitReceipt:
    validate_market_data_raw_frame(frame)
    dataset_dir = final_path.parent
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame_bytes = int(frame.memory_usage(index=False, deep=True).sum()) if len(frame.columns) else 0
    staging = max(
        policy.minimum_staging_headroom_bytes,
        int(frame_bytes * policy.staging_headroom_multiplier),
    )
    required = int(policy.minimum_free_bytes + staging)
    free = int(getattr(disk_usage_fn(dataset_dir), "free"))
    if free < required:
        raise MarketDataCommitError(
            f"Market Data storage free-space gate: free={free} bytes < required={required} bytes"
        )
    expected_columns = tuple(str(column) for column in frame.columns)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{expected_metadata.get('request_id') or 'market_data'}.",
        suffix=".parquet.tmp",
        dir=str(dataset_dir),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        codec.write(frame, temp_path, metadata=expected_metadata, compression=policy.compression)
        staged = codec.inspect(temp_path)
        validate_market_data_parquet_inspection(
            staged,
            expected_metadata=expected_metadata,
            expected_columns=expected_columns,
            expected_rows=len(frame),
        )
        compute_file_sha256(temp_path)
        atomic_replace_with_retry(temp_path, final_path)
        published = codec.inspect(final_path)
        validate_market_data_parquet_inspection(
            published,
            expected_metadata=expected_metadata,
            expected_columns=expected_columns,
            expected_rows=len(frame),
        )
        return MarketDataCommitReceipt(
            committed=True,
            row_count=len(frame),
            content_sha256=compute_file_sha256(final_path),
        )
    except MarketDataCommitError:
        raise
    except Exception as exc:
        raise MarketDataCommitError(
            f"Market Data parquet commit 失敗: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as exc:
                _LOG.warning("Market Data staging temp cleanup failed for %s: %s", temp_path, exc)

class MarketDataBootstrapStorageSink:
    def __init__(
        self,
        *,
        project_root,
        manifest: BootstrapRequestManifest,
        policy: MarketDataStoragePolicy | None = None,
        codec: ParquetCodec | None = None,
        disk_usage_fn: Callable[[str | os.PathLike[str]], object] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.manifest = manifest
        self.policy = policy or get_market_data_storage_policy()
        self.codec = codec or PyArrowParquetCodec()
        self.disk_usage_fn = disk_usage_fn or shutil.disk_usage
        self.archive_dir = resolve_market_data_bootstrap_archive_dir(
            self.project_root, manifest.manifest_fingerprint
        )
        self._requests = {request.request_id: request for request in manifest.requests}
        if len(self._requests) != manifest.total_requests:
            raise ValueError("Market Data storage manifest request identity 重複")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_bootstrap_manifest()

    def _ensure_bootstrap_manifest(self) -> None:
        path = self.archive_dir / MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME
        expected = build_bootstrap_storage_manifest_payload(self.manifest, self.policy)
        if path.exists():
            try:
                actual = load_json_strict(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise MarketDataCommitError(
                    f"既有 Market Data bootstrap manifest 無法驗證: {type(exc).__name__}: {exc}"
                ) from exc
            if actual != expected:
                raise MarketDataCommitError("既有 Market Data bootstrap manifest identity 不一致")
            return
        atomic_write_json(path, expected)

    def _validate_request(self, request: BootstrapHttpRequest) -> None:
        canonical = self._requests.get(request.request_id)
        if canonical is None or canonical != request:
            raise MarketDataCommitError("Storage sink 收到不屬於目前 bootstrap manifest 的 request")

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        validate_market_data_raw_frame(frame)

    def _ensure_dataset_schema(self, request: BootstrapHttpRequest, frame: pd.DataFrame) -> dict[str, object]:
        schema_payload = build_frame_schema_payload(frame)
        # A zero-column empty response contains no provider schema evidence.  It
        # can be archived, but must not create/replace the dataset schema owner.
        if len(frame.columns) == 0:
            return schema_payload
        dataset_dir = resolve_market_data_dataset_dir(
            self.project_root, self.manifest.manifest_fingerprint, request.dataset
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)
        schema_path = dataset_dir / MARKET_DATA_DATASET_SCHEMA_FILENAME
        expected_columns = list(schema_payload["columns"])
        if schema_path.exists():
            try:
                existing = load_json_strict(schema_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise MarketDataCommitError(
                    f"{request.dataset} dataset schema manifest 無法驗證: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(existing, dict) or list(existing.get("columns") or []) != expected_columns:
                raise MarketDataCommitError(
                    f"{request.dataset} provider schema drift: existing={existing.get('columns') if isinstance(existing, dict) else None}, current={expected_columns}"
                )
            return schema_payload
        atomic_write_json(
            schema_path,
            {
                "dataset": request.dataset,
                "columns": expected_columns,
                "column_fingerprint": schema_payload["column_fingerprint"],
            },
        )
        return schema_payload

    def _required_free_bytes(self, frame: pd.DataFrame) -> int:
        frame_bytes = int(frame.memory_usage(index=False, deep=True).sum()) if len(frame.columns) else 0
        staging = max(
            self.policy.minimum_staging_headroom_bytes,
            int(frame_bytes * self.policy.staging_headroom_multiplier),
        )
        return int(self.policy.minimum_free_bytes + staging)

    def _assert_disk_headroom(self, dataset_dir: Path, frame: pd.DataFrame) -> None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        usage = self.disk_usage_fn(dataset_dir)
        free = int(getattr(usage, "free"))
        required = self._required_free_bytes(frame)
        if free < required:
            raise MarketDataCommitError(
                f"Market Data storage free-space gate: free={free} bytes < required={required} bytes"
            )

    def validate_activation_readiness(self) -> None:
        """Fail before provider data HTTP when runtime/storage prerequisites are absent."""

        validate_runtime = getattr(self.codec, "validate_runtime", None)
        if callable(validate_runtime):
            validate_runtime()
        usage = self.disk_usage_fn(self.archive_dir)
        free = int(getattr(usage, "free"))
        required = int(self.policy.minimum_free_bytes + self.policy.minimum_staging_headroom_bytes)
        if free < required:
            raise MarketDataCommitError(
                f"Market Data bootstrap startup free-space gate: free={free} bytes < required={required} bytes"
            )

    def _expected_metadata(
        self,
        request: BootstrapHttpRequest,
        *,
        row_count: int,
        schema_payload: dict[str, object],
    ) -> dict[str, object]:
        return build_request_parquet_metadata(
            manifest=self.manifest,
            request=request,
            row_count=row_count,
            schema_payload=schema_payload,
        )

    @staticmethod
    def _validate_inspection(
        inspection: ParquetArtifactInspection,
        *,
        expected_metadata: dict[str, object],
        expected_columns: tuple[str, ...],
        expected_rows: int,
    ) -> None:
        validate_market_data_parquet_inspection(
            inspection,
            expected_metadata=expected_metadata,
            expected_columns=expected_columns,
            expected_rows=expected_rows,
        )

    def recover_committed(self, request: BootstrapHttpRequest) -> MarketDataCommitReceipt | None:
        self._validate_request(request)
        final_path = resolve_market_data_request_parquet_path(
            self.project_root, self.manifest.manifest_fingerprint, request
        )
        if not final_path.is_file():
            return None
        inspection = self.codec.inspect(final_path)
        metadata = inspection.metadata
        expected_identity = {
            "manifest_fingerprint": self.manifest.manifest_fingerprint,
            "registry_fingerprint": self.manifest.registry_fingerprint,
            "request_id": request.request_id,
            "dataset": request.dataset,
            "bootstrap_mode": request.bootstrap_mode,
            "data_id": request.data_id,
            "start_date": request.start_date,
            "end_date": request.end_date,
        }
        actual_identity = {key: metadata.get(key) for key in expected_identity}
        if actual_identity != expected_identity:
            raise MarketDataCommitError(
                f"既有 Parquet request identity 不一致: actual={actual_identity}, expected={expected_identity}"
            )
        if int(metadata.get("row_count", -1)) != inspection.row_count:
            raise MarketDataCommitError("既有 Parquet row_count metadata 與實體檔案不一致")
        if tuple(inspection.columns) and metadata.get("column_fingerprint"):
            column_payload = build_frame_schema_payload(
                pd.DataFrame(columns=list(inspection.columns))
            )
            if metadata.get("column_fingerprint") != column_payload["column_fingerprint"]:
                raise MarketDataCommitError("既有 Parquet column fingerprint 不一致")
        return MarketDataCommitReceipt(
            committed=True,
            row_count=inspection.row_count,
            content_sha256=compute_file_sha256(final_path),
        )

    def __call__(self, request: BootstrapHttpRequest, frame: pd.DataFrame) -> MarketDataCommitReceipt:
        self._validate_request(request)
        self._validate_frame(frame)
        existing = self.recover_committed(request)
        if existing is not None:
            if existing.row_count != len(frame):
                raise MarketDataCommitError(
                    f"既有 committed Parquet rows={existing.row_count} 與重新取得 rows={len(frame)} 不一致"
                )
            return existing

        schema_payload = self._ensure_dataset_schema(request, frame)
        expected_metadata = self._expected_metadata(
            request, row_count=len(frame), schema_payload=schema_payload
        )
        final_path = resolve_market_data_request_parquet_path(
            self.project_root, self.manifest.manifest_fingerprint, request
        )
        return commit_market_data_parquet_frame(
            frame=frame,
            final_path=final_path,
            expected_metadata=expected_metadata,
            policy=self.policy,
            codec=self.codec,
            disk_usage_fn=self.disk_usage_fn,
        )



__all__ = [
    "ParquetArtifactInspection",
    "ParquetCodec",
    "PyArrowParquetCodec",
    "validate_market_data_raw_frame",
    "validate_market_data_parquet_inspection",
    "commit_market_data_parquet_frame",
    "MarketDataBootstrapStorageSink",
]
