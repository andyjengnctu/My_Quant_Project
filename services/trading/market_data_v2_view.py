"""Trading Market Data V2 historical/latest read seam.

This module composes the immutable neutral Provider Snapshot with verified
Trading overlay evidence. A due batch may be only partially finished globally;
only datasets whose entire request set is committed are readable. It is a read
service: it performs no provider
calls, writes no market-data artifact, does not choose model scientific scope,
and does not define today's execution pool.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Iterator

import pandas as pd

from config.market_data import MARKET_DATA_V2_TRADING_OVERLAY_READ_CACHE_ENTRIES

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
from core.market_data_trading_sync import TRADING_SYNC_VALIDATION_CONTRACT_VERSION
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
from services.trading.market_data_v2_state_store import load_trading_market_data_v2_state


@dataclass(frozen=True)
class VerifiedTradingOverlayBatch:
    batch_fingerprint: str
    target_date: str
    completed_at: str
    artifacts: tuple


_TRADING_OVERLAY_BATCH_CACHE_LOCK = threading.RLock()
_TRADING_OVERLAY_BATCH_CACHE: OrderedDict[
    tuple[str, str],
    tuple[tuple[int, ...], VerifiedTradingOverlayBatch | None],
] = OrderedDict()


def _overlay_batch_stat_signature(manifest_path: Path, ledger_path: Path) -> tuple[int, ...]:
    manifest_stat = manifest_path.stat()
    ledger_stat = ledger_path.stat()
    signature = [
        int(manifest_stat.st_mtime_ns),
        int(manifest_stat.st_size),
        int(ledger_stat.st_mtime_ns),
        int(ledger_stat.st_size),
    ]
    # SQLite can retain recent writes in WAL before checkpointing the main DB.
    # Completed batches are immutable, but include WAL evidence in the cache
    # signature so any unexpected post-publication mutation invalidates reuse.
    wal_path = Path(f"{ledger_path}-wal")
    if wal_path.is_file():
        wal_stat = wal_path.stat()
        signature.extend((int(wal_stat.st_mtime_ns), int(wal_stat.st_size)))
    else:
        signature.extend((0, 0))
    return tuple(signature)


def _load_verified_overlay_batch(
    *,
    project_root: Path,
    manifest_path: Path,
    payload: dict,
) -> VerifiedTradingOverlayBatch | None:
    batch_fp = str(payload["batch_fingerprint"])
    ledger_path = resolve_trading_market_data_v2_ledger_path(project_root, batch_fp)
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Trading V2 batch ledger 不存在: {batch_fp}")

    cache_limit = max(0, int(MARKET_DATA_V2_TRADING_OVERLAY_READ_CACHE_ENTRIES))
    cache_key = (str(project_root), batch_fp)

    def _materialize() -> VerifiedTradingOverlayBatch | None:
        ledger = MarketDataJobLedger(ledger_path, read_only=True)
        workload_id = ledger.find_unique_workload_id_by_manifest_fingerprint(batch_fp)
        if workload_id is None:
            raise ValueError(f"Trading V2 batch ledger 找不到 manifest workload: {batch_fp}")
        summary = ledger.get_summary(workload_id)
        expected = int(payload.get("request_count") or 0)
        request_ids = tuple(str(value) for value in payload.get("request_ids") or ())
        if summary.total != expected or len(request_ids) != expected:
            raise ValueError(
                "Trading V2 batch ledger request-count drift: "
                f"batch={batch_fp}, total={summary.total}, manifest={expected}, ids={len(request_ids)}"
            )
        if summary.workload_status == WORKLOAD_DONE:
            if summary.done != expected or summary.blocked:
                raise ValueError(
                    "Trading V2 DONE batch ledger completeness drift: "
                    f"batch={batch_fp}, done={summary.done}, total={summary.total}, expected={expected}"
                )
            artifacts = ledger.list_committed_artifacts(workload_id)
            if tuple(item.request_id for item in artifacts) != request_ids:
                raise ValueError(f"Trading V2 batch committed request identity drift: {batch_fp}")
        else:
            # Dataset-level partial publication is a current-contract behavior.
            # Retained legacy batches keep their historical complete-workload
            # read rule so compatibility never silently broadens old evidence.
            if int(payload.get("validation_contract_version") or 0) != int(TRADING_SYNC_VALIDATION_CONTRACT_VERSION):
                return None
            jobs = ledger.list_jobs(workload_id)
            if tuple(item.request_id for item in jobs) != request_ids:
                raise ValueError(f"Trading V2 batch ledger request identity drift: {batch_fp}")
            if any(item.to_request().request_id != item.request_id for item in jobs):
                raise ValueError(f"Trading V2 batch ledger request payload drift: {batch_fp}")
            # Auto/Due sync publishes READY evidence per dataset as soon as every
            # logical request for that dataset is committed. Requiring unrelated
            # datasets in the same batch to finish made dataset state say READY
            # while Trading consumers still saw the previous batch. Mirror the
            # canonical dataset-level completion boundary here: a partially
            # completed dataset remains invisible, but a fully committed dataset
            # is valid overlay evidence even while other datasets wait/retry.
            artifacts = ledger.list_dataset_complete_artifacts(workload_id)
            if not artifacts:
                return None
            manifest_request_ids = set(request_ids)
            artifact_request_ids = {item.request_id for item in artifacts}
            if not artifact_request_ids <= manifest_request_ids:
                raise ValueError(f"Trading V2 dataset-complete artifact identity drift: {batch_fp}")
            if any(item.to_request().request_id != item.request_id for item in artifacts):
                raise ValueError(f"Trading V2 dataset-complete artifact payload drift: {batch_fp}")
        completed_at = max((str(item.completed_at or "") for item in artifacts), default="")
        return VerifiedTradingOverlayBatch(
            batch_fingerprint=batch_fp,
            target_date=str(payload.get("target_date") or ""),
            completed_at=completed_at,
            artifacts=artifacts,
        )

    if cache_limit <= 0:
        return _materialize()

    # Several initial Trading Center read models open independent views in
    # parallel. Serialize ledger verification; the cache signature includes the
    # live SQLite/WAL evidence so newly completed dataset subsets invalidate reuse.
    with _TRADING_OVERLAY_BATCH_CACHE_LOCK:
        signature = _overlay_batch_stat_signature(manifest_path, ledger_path)
        cached = _TRADING_OVERLAY_BATCH_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            _TRADING_OVERLAY_BATCH_CACHE.move_to_end(cache_key)
            return cached[1]
        batch = _materialize()
        # Cache both visible partial batches and the current "no complete dataset"
        # result.  The signature includes SQLite/WAL evidence, so the cache is
        # invalidated immediately when any job completion mutates the ledger.
        # This avoids repeatedly opening/querying every retained partial ledger
        # while several Workbench read models initialize in parallel.
        signature = _overlay_batch_stat_signature(manifest_path, ledger_path)
        _TRADING_OVERLAY_BATCH_CACHE[cache_key] = (signature, batch)
        _TRADING_OVERLAY_BATCH_CACHE.move_to_end(cache_key)
        while len(_TRADING_OVERLAY_BATCH_CACHE) > cache_limit:
            _TRADING_OVERLAY_BATCH_CACHE.popitem(last=False)
        return batch


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
        overlay_dataset_scope: tuple[str, ...] | None = None,
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
        self._overlay_dataset_scope = (
            None
            if overlay_dataset_scope is None
            else frozenset(str(value).strip() for value in overlay_dataset_scope if str(value).strip())
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
        overlay_dataset_scope=None,
        frame_reader=None,
        hash_fn=None,
    ):
        """Open an immutable consumer view pinned to an exact verified overlay set.

        This is stronger than ``open_as_of_target``: later batches carrying the
        same target date are excluded as well. ``overlay_dataset_scope`` freezes
        the consumer to its declared dependency datasets, so unrelated datasets
        that finish later inside an already pinned partial batch cannot drift the
        finalized Scanner/Params source view.
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
            overlay_dataset_scope=None if overlay_dataset_scope is None else tuple(overlay_dataset_scope),
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
            batch = _load_verified_overlay_batch(
                project_root=self.project_root,
                manifest_path=manifest_path,
                payload=payload,
            )
            if batch is not None:
                batches.append(batch)
        # Global batch identity order must not move merely because an unrelated
        # dataset in a partially completed batch finishes later. Dataset merge
        # precedence is resolved separately from that dataset's own completion.
        batches.sort(key=lambda item: (item.target_date, item.batch_fingerprint))
        self._overlay_batches = tuple(batches)
        return self._overlay_batches

    def _overlay_batches_for_dataset(self, dataset: str) -> tuple[VerifiedTradingOverlayBatch, ...]:
        name = str(dataset or "").strip()
        if self._overlay_dataset_scope is not None and name not in self._overlay_dataset_scope:
            return ()
        relevant = [
            batch
            for batch in self._load_overlay_batches()
            if any(str(artifact.dataset) == name for artifact in batch.artifacts)
        ]

        def _dataset_completion(batch: VerifiedTradingOverlayBatch) -> str:
            return max(
                (
                    str(artifact.completed_at or "")
                    for artifact in batch.artifacts
                    if str(artifact.dataset) == name
                ),
                default="",
            )

        relevant.sort(
            key=lambda batch: (
                batch.target_date,
                _dataset_completion(batch),
                batch.batch_fingerprint,
            )
        )
        return tuple(relevant)

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
        for batch in self._overlay_batches_for_dataset(dataset):
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
        for batch in self._overlay_batches_for_dataset(dataset):
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

        Provider fragments are oldest. Dataset-complete Trading overlay fragments
        are ordered deterministically after them, so the latest verified row for
        the registry primary key wins. Incomplete datasets remain invisible even
        when sibling datasets in the same resumable batch have completed.
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

    def overlay_batch_fingerprints_for_datasets(self, datasets) -> tuple[str, ...]:
        """Return verified overlay batches that can affect the named datasets.

        Consumer promotion uses this narrow identity to distinguish an irrelevant
        late batch from a same-target revision of an execution dependency.  The
        full consumer view still pins the complete verified overlay set when it is
        republished.
        """

        names = {str(dataset or "").strip() for dataset in datasets}
        names.discard("")
        if self._overlay_dataset_scope is not None:
            names.intersection_update(self._overlay_dataset_scope)
        if not names:
            return ()
        return tuple(
            batch.batch_fingerprint
            for batch in self._load_overlay_batches()
            if any(str(artifact.dataset) in names for artifact in batch.artifacts)
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
