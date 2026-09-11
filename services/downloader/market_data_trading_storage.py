"""Atomic Parquet storage for the isolated Trading Market Data V2 overlay batches."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Callable

import pandas as pd

from core.file_integrity import atomic_write_json, compute_file_sha256, load_json_strict
from core.market_data_dataset_registry import get_market_dataset_spec
from core.market_data_freshness_contract import EXPECTED_DATE_TRADING_TARGET, get_market_data_freshness_contract
from core.market_data_instrument_universe import build_current_stock_etf_universe
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
        self._dataset_observations: dict[str, dict[str, object]] = {}
        self._target_instrument_ids: dict[str, set[str]] = {}
        self._schema_columns: dict[str, tuple[str, ...]] = {}
        self._stock_info_frame: pd.DataFrame | None = None
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


    @staticmethod
    def _frame_date_bounds(frame: pd.DataFrame) -> tuple[str | None, str | None]:
        if frame.empty or "date" not in frame.columns:
            return None, None
        values = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if values.empty:
            return None, None
        return values.min().date().isoformat(), values.max().date().isoformat()

    def _record_observation(
        self,
        *,
        request,
        row_count: int,
        observed_min_date: str | None,
        observed_max_date: str | None,
    ) -> None:
        dataset = str(request.dataset)
        row = self._dataset_observations.setdefault(
            dataset,
            {
                "request_count": 0,
                "nonempty_request_count": 0,
                "target_covering_request_count": 0,
                "target_fresh_request_count": 0,
                "row_count": 0,
                "observed_min_date": None,
                "observed_max_date": None,
                "observed_max_date_by_data_id": {},
            },
        )
        row["request_count"] = int(row["request_count"]) + 1
        row["row_count"] = int(row["row_count"]) + int(row_count)
        if int(row_count) > 0:
            row["nonempty_request_count"] = int(row["nonempty_request_count"]) + 1

        target = str(self.manifest.as_of_date)
        covers_target = bool(
            request.start_date is not None
            and request.end_date is not None
            and str(request.start_date) <= target <= str(request.end_date)
        )
        if covers_target:
            row["target_covering_request_count"] = int(row["target_covering_request_count"]) + 1
            if observed_max_date and str(observed_max_date) >= target:
                row["target_fresh_request_count"] = int(row["target_fresh_request_count"]) + 1

        if observed_min_date:
            current = str(row.get("observed_min_date") or "").strip()
            row["observed_min_date"] = observed_min_date if not current else min(current, observed_min_date)
        if observed_max_date:
            current = str(row.get("observed_max_date") or "").strip()
            row["observed_max_date"] = observed_max_date if not current else max(current, observed_max_date)
            lane_key = str(request.data_id) if request.data_id is not None else "__ALL__"
            lane_map = dict(row.get("observed_max_date_by_data_id") or {})
            prior_lane = str(lane_map.get(lane_key) or "").strip()
            lane_map[lane_key] = observed_max_date if not prior_lane else max(prior_lane, observed_max_date)
            row["observed_max_date_by_data_id"] = lane_map

    def _capture_verification_frame(self, request, frame: pd.DataFrame) -> None:
        dataset = str(request.dataset)
        columns = tuple(str(value) for value in frame.columns)
        if columns:
            existing = self._schema_columns.get(dataset)
            if existing is not None and existing != columns:
                raise MarketDataCommitError(
                    f"{dataset} 同一 batch 內 schema columns 不一致: first={existing}, current={columns}"
                )
            self._schema_columns[dataset] = columns
        if dataset == "TaiwanStockInfo" and not frame.empty:
            self._stock_info_frame = frame.copy(deep=True)
        if frame.empty or "date" not in frame.columns or "stock_id" not in frame.columns:
            return
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        mask = parsed.dt.strftime("%Y-%m-%d") == str(self.manifest.as_of_date)
        if not mask.any():
            return
        ids = {str(value or "").strip() for value in frame.loc[mask, "stock_id"].tolist()}
        ids.discard("")
        self._target_instrument_ids.setdefault(dataset, set()).update(ids)

    def dataset_observations(self) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for dataset, values in self._dataset_observations.items():
            row = dict(values)
            columns = self._schema_columns.get(dataset, ())
            instruments = self._target_instrument_ids.get(dataset, set())
            row["schema_column_count"] = len(columns)
            row["schema_columns"] = list(columns)
            row["target_instrument_count"] = len(instruments)
            row["target_instrument_sample"] = sorted(instruments)[:20]
            payload[dataset] = row
        return payload

    def verification_summary(self) -> dict[str, object]:
        observations = self.dataset_observations()
        schema_verified = sum(bool(row.get("schema_column_count")) for row in observations.values())
        reference_ids: set[str] = set()
        reference_error = None
        if self._stock_info_frame is not None:
            try:
                reference_ids = set(
                    build_current_stock_etf_universe(
                        self._stock_info_frame,
                        pd.DataFrame(columns=["date", "stock_id"]),
                        as_of_date=self.manifest.as_of_date,
                    )
                )
            except (TypeError, ValueError) as exc:
                reference_error = f"{type(exc).__name__}: {exc}"

        # StockInfo is an independent broad stock/ETF identity reference, but it
        # is not an authoritative expected-membership set for dataset-specific
        # feeds such as PER, margin, short-sale, institutional flows or raw
        # Price (which can include warrants and other security types).  Do not
        # render false MATCH/DIFF completeness verdicts from incomparable sets.
        target_counts = [
            {
                "dataset": dataset,
                "observed_target_instrument_count": len(ids),
                "comparison_status": "UNVERIFIED",
                "reason": "no_authoritative_dataset_specific_expected_universe",
            }
            for dataset, ids in sorted(self._target_instrument_ids.items())
            if ids
        ]
        return {
            "schema_verified_dataset_count": int(schema_verified),
            "observed_dataset_count": len(observations),
            "current_stockinfo_reference_status": "AVAILABLE" if reference_ids else "UNAVAILABLE",
            "current_stockinfo_reference_count": len(reference_ids),
            "current_stockinfo_reference_error": reference_error,
            "instrument_completeness_status": "UNVERIFIED",
            "instrument_completeness_is_blocking": False,
            "instrument_completeness_reason": "no_authoritative_dataset_specific_expected_universe",
            "instrument_target_observation_rows": target_counts,
        }

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

    def _validate_request_payload_scope(self, request, frame: pd.DataFrame) -> None:
        """Fail closed when a dated provider response escapes its request window."""

        if request.start_date is None and request.end_date is None:
            return
        if request.start_date is None or request.end_date is None:
            raise MarketDataCommitError(
                f"{request.dataset} dated request 必須同時具有 start_date/end_date"
            )
        if frame.empty:
            return
        if "date" not in frame.columns:
            raise MarketDataCommitError(
                f"{request.dataset} dated request payload 缺少 date 欄位，無法驗證 request scope"
            )
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        if parsed.isna().any():
            raise MarketDataCommitError(f"{request.dataset} dated request payload 含不合法 date")
        observed = parsed.dt.strftime("%Y-%m-%d")
        outside = (observed < str(request.start_date)) | (observed > str(request.end_date))
        if outside.any():
            sample = sorted(set(observed.loc[outside].head(10).tolist()))
            raise MarketDataCommitError(
                f"{request.dataset} provider payload 超出 request date scope: "
                f"requested={request.start_date}~{request.end_date}, observed={sample}"
            )
        if request.start_date == request.end_date:
            unique_dates = set(observed.tolist())
            if unique_dates != {str(request.start_date)}:
                raise MarketDataCommitError(
                    f"{request.dataset} exact-date payload scope 不一致: "
                    f"requested={request.start_date}, observed={sorted(unique_dates)}"
                )

    def _expected_identity(self, request) -> dict[str, object]:
        identity: dict[str, object] = {
            "batch_fingerprint": self.manifest.manifest_fingerprint,
            "validation_contract_version": int(self.manifest.validation_contract_version),
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
        if self.manifest.refresh_token is not None:
            identity["refresh_token"] = self.manifest.refresh_token
        return identity

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
        if inspection.columns:
            self._schema_columns[str(request.dataset)] = tuple(inspection.columns)
        self._record_observation(
            request=request,
            row_count=inspection.row_count,
            observed_min_date=str(metadata.get("observed_min_date") or "").strip() or None,
            observed_max_date=str(metadata.get("observed_max_date") or "").strip() or None,
        )
        return MarketDataCommitReceipt(
            committed=True,
            row_count=inspection.row_count,
            content_sha256=compute_file_sha256(path),
        )

    def __call__(self, request, frame: pd.DataFrame) -> MarketDataCommitReceipt:
        self._validate_request(request)
        validate_market_data_raw_frame(frame)
        self._validate_request_payload_scope(request, frame)
        existing = self.recover_committed(request)
        if existing is not None:
            if existing.row_count != len(frame):
                raise MarketDataCommitError(
                    f"既有 Trading V2 artifact rows={existing.row_count} 與重新取得 rows={len(frame)} 不一致"
                )
            return existing
        schema = self._ensure_schema(request, frame)
        observed_min_date, observed_max_date = self._frame_date_bounds(frame)
        metadata = build_trading_request_metadata(
            manifest=self.manifest,
            request=request,
            row_count=len(frame),
            schema_payload=schema,
            observed_min_date=observed_min_date,
            observed_max_date=observed_max_date,
        )
        path = resolve_trading_market_data_v2_request_path(
            self.project_root, self.manifest.manifest_fingerprint, request
        )
        receipt = commit_market_data_parquet_frame(
            frame=frame,
            final_path=path,
            expected_metadata=metadata,
            policy=self.policy,
            codec=self.codec,
            disk_usage_fn=self.disk_usage_fn,
        )
        self._capture_verification_frame(request, frame)
        self._record_observation(
            request=request,
            row_count=len(frame),
            observed_min_date=observed_min_date,
            observed_max_date=observed_max_date,
        )
        return receipt


__all__ = ["MarketDataTradingStorageSink"]
