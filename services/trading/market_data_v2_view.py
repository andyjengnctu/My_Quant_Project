"""Trading Market Data V2 historical/latest read seam.

This module composes the immutable neutral Provider Snapshot with only fully
verified Trading overlay batches.  It is a read service: it performs no provider
calls, writes no market-data artifact, does not choose model scientific scope,
and does not define today's execution pool.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from core.file_integrity import canonical_json_sha256, compute_file_sha256, load_json_strict
from core.market_data_dataset_registry import get_market_dataset_spec, resolve_market_dataset_row_identity
from core.market_data_instrument_universe import (
    build_historical_market_state_guard,
    build_historical_stock_etf_universe,
)
from core.market_data_pit_universe import build_daily_pit_market_universe
from core.market_data_pool_contract import project_daily_model_context_pool
from core.market_data_trading_storage_contract import (
    TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME,
    resolve_trading_market_data_v2_ledger_path,
    resolve_trading_market_data_v2_request_path,
    resolve_trading_market_data_v2_root,
    validate_trading_sync_batch_manifest_payload_for_read,
)
from core.market_data_trading_view import (
    TradingV2TrainingHorizon,
    merge_market_data_v2_fragments,
    resolve_trading_v2_training_horizon,
)
from services.downloader.market_data_ledger import MarketDataJobLedger, WORKLOAD_DONE
from services.market_data.daily_pit_universe import read_market_data_v2_daily_pit_members
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.market_data.provider_snapshot_view import ProviderSnapshotView, read_parquet_frame
from services.trading.market_data_dataset_state import load_market_data_dataset_state
from services.trading.market_data_v2_state import load_trading_market_data_v2_state


@dataclass(frozen=True)
class VerifiedTradingOverlayBatch:
    batch_fingerprint: str
    target_date: str
    completed_at: str
    artifacts: tuple


class TradingMarketDataV2View:
    def __init__(
        self,
        *,
        project_root,
        archive,
        frame_reader=None,
        hash_fn=None,
        overlay_target_date_cutoff: str | None = None,
        overlay_batch_fingerprints: tuple[str, ...] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.archive = archive
        self._overlay_target_date_cutoff = (
            None if overlay_target_date_cutoff is None else pd.to_datetime(overlay_target_date_cutoff, errors="raise").date().isoformat()
        )
        self._overlay_batch_fingerprints = (
            None
            if overlay_batch_fingerprints is None
            else tuple(dict.fromkeys(str(value).strip() for value in overlay_batch_fingerprints if str(value).strip()))
        )
        self._frame_reader = frame_reader or read_parquet_frame
        self._hash_fn = hash_fn or compute_file_sha256
        self._provider_view = ProviderSnapshotView(
            project_root=self.project_root,
            archive=archive,
            frame_reader=self._frame_reader,
            hash_fn=self._hash_fn,
        )
        self._overlay_batches: tuple[VerifiedTradingOverlayBatch, ...] | None = None

    @classmethod
    def open(cls, project_root, *, frame_reader=None, hash_fn=None):
        root = Path(project_root).resolve()
        state = load_trading_market_data_v2_state(root, required=False)
        pinned = None if state is None else str(state.get("base_provider_snapshot_fingerprint") or "").strip()
        archive = load_ready_provider_snapshot_archive(
            root,
            snapshot_fingerprint=pinned or None,
        )
        return cls(
            project_root=root,
            archive=archive,
            frame_reader=frame_reader,
            hash_fn=hash_fn,
        )

    @classmethod
    def open_as_of_target(
        cls,
        project_root,
        *,
        target_date: str,
        frame_reader=None,
        hash_fn=None,
    ):
        """Open the immutable operational view that was eligible through ``target_date``.

        Later verified batches are intentionally excluded.  This preserves a
        finalized Scanner/Params consumer when a newer target is only partially
        synchronized, including current-vintage adjusted-price restatements.
        """

        root = Path(project_root).resolve()
        state = load_trading_market_data_v2_state(root, required=False)
        pinned = None if state is None else str(state.get("base_provider_snapshot_fingerprint") or "").strip()
        archive = load_ready_provider_snapshot_archive(
            root,
            snapshot_fingerprint=pinned or None,
        )
        return cls(
            project_root=root,
            archive=archive,
            frame_reader=frame_reader,
            hash_fn=hash_fn,
            overlay_target_date_cutoff=str(target_date),
        )


    @classmethod
    def open_pinned(
        cls,
        project_root,
        *,
        target_date: str,
        overlay_batch_fingerprints,
        provider_snapshot_fingerprint: str | None = None,
        frame_reader=None,
        hash_fn=None,
    ):
        """Open an immutable consumer view pinned to an exact verified overlay set.

        This is stronger than ``open_as_of_target``: later DONE batches carrying
        the same target date are excluded as well, so a finalized Scanner/Params
        lineage cannot drift after its consumer state has been published.
        """

        root = Path(project_root).resolve()
        state = load_trading_market_data_v2_state(root, required=False)
        current_pinned = None if state is None else str(state.get("base_provider_snapshot_fingerprint") or "").strip()
        pinned = str(provider_snapshot_fingerprint or "").strip() or current_pinned
        archive = load_ready_provider_snapshot_archive(
            root,
            snapshot_fingerprint=pinned or None,
        )
        return cls(
            project_root=root,
            archive=archive,
            frame_reader=frame_reader,
            hash_fn=hash_fn,
            overlay_target_date_cutoff=str(target_date),
            overlay_batch_fingerprints=tuple(overlay_batch_fingerprints or ()),
        )

    def training_horizon(
        self,
        *,
        required_datasets,
        maximum_training_date: str | None = None,
    ) -> TradingV2TrainingHorizon:
        state = load_market_data_dataset_state(self.project_root, required=False)
        rows = dict((state or {}).get("datasets") or {})
        return resolve_trading_v2_training_horizon(
            provider_as_of_date=self.archive.as_of_date,
            required_datasets=required_datasets,
            dataset_state=rows,
            maximum_training_date=maximum_training_date,
        )

    def _load_overlay_batches(self) -> tuple[VerifiedTradingOverlayBatch, ...]:
        if self._overlay_batches is not None:
            return self._overlay_batches
        base = resolve_trading_market_data_v2_root(self.project_root) / "batches"
        batches: list[VerifiedTradingOverlayBatch] = []
        if not base.is_dir():
            self._overlay_batches = ()
            return self._overlay_batches

        if self._overlay_batch_fingerprints is None:
            manifest_paths = sorted(base.glob(f"*/{TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME}"))
        else:
            manifest_paths = []
            for batch_fp in self._overlay_batch_fingerprints:
                manifest_path = base / batch_fp / TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"Trading V2 pinned batch manifest 不存在: {batch_fp}")
                manifest_paths.append(manifest_path)

        pinned_set = None if self._overlay_batch_fingerprints is None else set(self._overlay_batch_fingerprints)
        for manifest_path in manifest_paths:
            payload = validate_trading_sync_batch_manifest_payload_for_read(load_json_strict(manifest_path))
            batch_fp = str(payload["batch_fingerprint"])
            if manifest_path.parent.name != batch_fp:
                raise ValueError("Trading V2 batch directory 與 manifest fingerprint 不一致")
            if str(payload.get("base_provider_snapshot_fingerprint") or "") != self.archive.snapshot_fingerprint:
                # A retained batch pinned to another immutable Provider Snapshot
                # belongs to another lineage and is not part of this view.
                continue
            if (
                self._overlay_target_date_cutoff is not None
                and str(payload.get("target_date") or "") > self._overlay_target_date_cutoff
            ):
                continue
            if (
                pinned_set is not None
                and batch_fp not in pinned_set
            ):
                continue
            ledger_path = resolve_trading_market_data_v2_ledger_path(self.project_root, batch_fp)
            if not ledger_path.is_file():
                raise FileNotFoundError(f"Trading V2 batch ledger 不存在: {batch_fp}")
            ledger = MarketDataJobLedger(ledger_path, read_only=True)
            workload_id = ledger.find_unique_workload_id_by_manifest_fingerprint(batch_fp)
            if workload_id is None:
                raise ValueError(f"Trading V2 batch ledger 找不到 manifest workload: {batch_fp}")
            summary = ledger.get_summary(workload_id)
            # Batch manifests are created before execution; unfinished batches are
            # legitimate resumable state and must never leak into the read view.
            if summary.workload_status != WORKLOAD_DONE:
                continue
            expected = int(payload.get("request_count") or 0)
            if summary.total != expected or summary.done != expected or summary.blocked:
                raise ValueError(
                    "Trading V2 DONE batch ledger completeness drift: "
                    f"batch={batch_fp}, done={summary.done}, total={summary.total}, expected={expected}"
                )
            artifacts = ledger.list_committed_artifacts(workload_id)
            request_ids = tuple(str(value) for value in payload.get("request_ids") or ())
            if tuple(item.request_id for item in artifacts) != request_ids:
                raise ValueError(f"Trading V2 batch committed request identity drift: {batch_fp}")
            completed_at = max((str(item.completed_at or "") for item in artifacts), default="")
            batches.append(
                VerifiedTradingOverlayBatch(
                    batch_fingerprint=batch_fp,
                    target_date=str(payload.get("target_date") or ""),
                    completed_at=completed_at,
                    artifacts=artifacts,
                )
            )
        batches.sort(key=lambda item: (item.target_date, item.completed_at, item.batch_fingerprint))
        self._overlay_batches = tuple(batches)
        return self._overlay_batches

    @staticmethod
    def _filter_frame(
        frame: pd.DataFrame,
        *,
        start_date: str | None,
        end_date: str | None,
        data_id: str | None,
    ) -> pd.DataFrame:
        work = frame
        if data_id is not None and "stock_id" in work.columns:
            work = work.loc[work["stock_id"].astype(str).str.strip() == str(data_id).strip()]
        if (start_date is not None or end_date is not None) and "date" in work.columns:
            dates = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            mask = dates.notna()
            if start_date is not None:
                mask &= dates >= str(start_date)
            if end_date is not None:
                mask &= dates <= str(end_date)
            work = work.loc[mask]
        return work.reset_index(drop=True)

    def _iter_verified_overlay_frames(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...] | None,
        data_id: str | None,
        start_date: str | None,
        end_date: str | None,
    ) -> Iterator[pd.DataFrame]:
        for batch in self._load_overlay_batches():
            for artifact in batch.artifacts:
                if str(artifact.dataset) != dataset:
                    continue
                if data_id is not None and artifact.data_id is not None and str(artifact.data_id) != str(data_id):
                    continue
                request = artifact.to_request()
                path = resolve_trading_market_data_v2_request_path(
                    self.project_root,
                    batch.batch_fingerprint,
                    request,
                )
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Trading V2 committed overlay artifact 不存在: batch={batch.batch_fingerprint}, request={artifact.request_id}"
                    )
                if str(self._hash_fn(path) or "").strip().lower() != str(artifact.content_sha256):
                    raise ValueError(f"Trading V2 overlay artifact SHA256 drift: {artifact.request_id}")
                frame = self._frame_reader(path, columns)
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError("Trading V2 frame reader 必須回傳 pandas.DataFrame")
                yield self._filter_frame(
                    frame,
                    start_date=start_date,
                    end_date=end_date,
                    data_id=data_id,
                )

    def _iter_verified_overlay_frames_many_data_ids(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...] | None,
        data_ids,
        start_date: str | None,
        end_date: str | None,
    ) -> Iterator[pd.DataFrame]:
        wanted = {str(value).strip() for value in data_ids if str(value).strip()}
        if not wanted:
            return
        for batch in self._load_overlay_batches():
            for artifact in batch.artifacts:
                if str(artifact.dataset) != dataset:
                    continue
                artifact_id = str(artifact.data_id or "").strip()
                if artifact_id and artifact_id not in wanted:
                    continue
                request = artifact.to_request()
                path = resolve_trading_market_data_v2_request_path(
                    self.project_root,
                    batch.batch_fingerprint,
                    request,
                )
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Trading V2 committed overlay artifact 不存在: batch={batch.batch_fingerprint}, request={artifact.request_id}"
                    )
                if str(self._hash_fn(path) or "").strip().lower() != str(artifact.content_sha256):
                    raise ValueError(f"Trading V2 overlay artifact SHA256 drift: {artifact.request_id}")
                frame = self._frame_reader(path, columns)
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError("Trading V2 frame reader 必須回傳 pandas.DataFrame")
                if not artifact_id and "stock_id" in frame.columns:
                    frame = frame.loc[frame["stock_id"].astype(str).str.strip().isin(wanted)]
                yield self._filter_frame(
                    frame,
                    start_date=start_date,
                    end_date=end_date,
                    data_id=None,
                )

    def read_dataset_frame_many_data_ids(
        self,
        dataset: str,
        *,
        data_ids,
        columns: tuple[str, ...] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Read one latest dataset view for many stock IDs in one I/O pass."""

        name = str(dataset or "").strip()
        wanted = tuple(dict.fromkeys(str(value).strip() for value in data_ids if str(value).strip()))
        if not wanted:
            return pd.DataFrame(columns=list(columns or ()))
        spec = get_market_dataset_spec(name)
        if not spec.included:
            raise ValueError(f"Trading V2 view 不允許讀未納入 archive 的 dataset: {name}")
        keys = resolve_market_dataset_row_identity(spec)
        if not keys:
            raise RuntimeError(f"{name} 尚未宣告 row identity，禁止建立可覆寫 latest view")
        requested = None if columns is None else tuple(dict.fromkeys(str(value) for value in columns))
        read_columns = None if requested is None else tuple(dict.fromkeys((*requested, *keys)))
        fragments: list[pd.DataFrame] = []
        for frame in self._provider_view.iter_verified_frames_many_data_ids(
            name, columns=read_columns, data_ids=wanted
        ):
            fragments.append(
                self._filter_frame(
                    frame,
                    start_date=start_date,
                    end_date=end_date,
                    data_id=None,
                )
            )
        fragments.extend(
            self._iter_verified_overlay_frames_many_data_ids(
                name,
                columns=read_columns,
                data_ids=wanted,
                start_date=start_date,
                end_date=end_date,
            )
        )
        merged = merge_market_data_v2_fragments(fragments, primary_key=keys)
        if "stock_id" in merged.columns:
            merged = merged.loc[merged["stock_id"].astype(str).str.strip().isin(set(wanted))].reset_index(drop=True)
        if requested is not None:
            missing = [column for column in requested if column not in merged.columns]
            if missing and not merged.empty:
                raise ValueError(f"Trading V2 latest view 缺 requested columns: {missing}")
            return merged.reindex(columns=list(requested))
        return merged


    def read_dataset_frame(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...] | None = None,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Read one verified latest operational dataset view.

        Provider fragments are oldest.  Complete Trading overlay fragments are
        ordered deterministically after them, so the latest verified row for the
        registry primary key wins.  Incomplete/resumable batches are invisible.
        """

        name = str(dataset or "").strip()
        spec = get_market_dataset_spec(name)
        if not spec.included:
            raise ValueError(f"Trading V2 view 不允許讀未納入 archive 的 dataset: {name}")
        keys = resolve_market_dataset_row_identity(spec)
        if not keys:
            raise RuntimeError(f"{name} 尚未宣告 row identity，禁止建立可覆寫 latest view")

        requested = None if columns is None else tuple(dict.fromkeys(str(value) for value in columns))
        read_columns = None if requested is None else tuple(dict.fromkeys((*requested, *keys)))
        fragments: list[pd.DataFrame] = []
        for frame in self._provider_view.iter_verified_frames(name, columns=read_columns, data_id=data_id):
            fragments.append(
                self._filter_frame(
                    frame,
                    start_date=start_date,
                    end_date=end_date,
                    data_id=data_id,
                )
            )
        fragments.extend(
            self._iter_verified_overlay_frames(
                name,
                columns=read_columns,
                data_id=data_id,
                start_date=start_date,
                end_date=end_date,
            )
        )
        merged = merge_market_data_v2_fragments(fragments, primary_key=keys)
        if requested is not None:
            missing = [column for column in requested if column not in merged.columns]
            if missing and not merged.empty:
                raise ValueError(f"Trading V2 latest view 缺 requested columns: {missing}")
            return merged.reindex(columns=list(requested))
        return merged

    def daily_pit_market_members(self, date_value: str) -> tuple[str, ...]:
        """Return date-local market membership without consulting execution pool state."""

        date_text = pd.to_datetime(date_value, errors="raise").date().isoformat()
        if date_text <= self.archive.as_of_date:
            return read_market_data_v2_daily_pit_members(
                self.project_root,
                provider_snapshot_fingerprint=self.archive.snapshot_fingerprint,
                date_value=date_text,
            )

        pit_sources = (
            "TaiwanStockPrice",
            "TaiwanStockTradingDate",
            "TaiwanStockInfo",
            "TaiwanStockDelisting",
        )
        horizon = self.training_horizon(required_datasets=pit_sources)
        if date_text > horizon.training_through_date:
            raise RuntimeError(
                "Trading V2 PIT membership 日期超過共同 READY horizon: "
                f"date={date_text}, ready_through={horizon.training_through_date}"
            )
        calendar = self.read_dataset_frame(
            "TaiwanStockTradingDate",
            columns=("date",),
            start_date=date_text,
            end_date=date_text,
        )
        calendar_dates = tuple(
            sorted(set(pd.to_datetime(calendar["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d")))
        )
        if date_text not in calendar_dates:
            raise ValueError(f"Trading V2 date 不在 verified TaiwanStockTradingDate: {date_text}")

        stock_info = self.read_dataset_frame(
            "TaiwanStockInfo",
            columns=("date", "stock_id", "type", "industry_category"),
        )
        delisting = self.read_dataset_frame(
            "TaiwanStockDelisting",
            columns=("date", "stock_id"),
        )
        historical_instruments = build_historical_stock_etf_universe(stock_info, delisting)
        transition_guard = build_historical_market_state_guard(
            stock_info,
            historical_instruments=historical_instruments,
        )
        raw_price = self.read_dataset_frame(
            "TaiwanStockPrice",
            columns=("date", "stock_id"),
            start_date=date_text,
            end_date=date_text,
        )
        daily = build_daily_pit_market_universe(
            raw_price,
            historical_instruments=historical_instruments,
            transition_excluded_through=transition_guard,
            trading_dates=(date_text,),
            provider_as_of_date=date_text,
        )
        return tuple(daily["stock_id"].astype(str).tolist())

    def model_context_members(self, date_value: str, *, requested_members) -> tuple[str, ...]:
        return project_daily_model_context_pool(
            self.daily_pit_market_members(date_value),
            requested_members=requested_members,
        )

    def view_identity(
        self,
        *,
        required_datasets,
        maximum_training_date: str | None = None,
    ) -> dict[str, object]:
        horizon = self.training_horizon(
            required_datasets=required_datasets,
            maximum_training_date=maximum_training_date,
        )
        payload = {
            "role": "trading_market_data_v2_historical_latest_view",
            "provider_snapshot_fingerprint": self.archive.snapshot_fingerprint,
            "provider_as_of_date": self.archive.as_of_date,
            "training_horizon": horizon.as_dict(),
            "overlay_batch_fingerprints": [item.batch_fingerprint for item in self._load_overlay_batches()],
            "historical_membership_source": "neutral_daily_pit_market_universe",
            "current_execution_pool_used_for_historical_membership": False,
        }
        return {**payload, "view_fingerprint": canonical_json_sha256(payload)}


__all__ = ["VerifiedTradingOverlayBatch", "TradingMarketDataV2View"]
