"""Generic immutable read view over one READY Market Data V2 Provider Snapshot."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from core.console_report import project_relative_display_path
from core.file_integrity import compute_file_sha256
from core.market_data_storage_contract import resolve_market_data_request_parquet_path
from services.market_data.provider_snapshot_repository import ReadyProviderSnapshotArchive

FrameReader = Callable[[Path, tuple[str, ...] | None], pd.DataFrame]
HashFn = Callable[[Path], str]


def read_parquet_frame(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Read a Market Data V2 parquet fragment without mutating archive state."""

    if not columns:
        return pd.read_parquet(path)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Market Data V2 Parquet read 需要 pyarrow") from exc
    parquet = pq.ParquetFile(path)
    names = tuple(str(name) for name in parquet.schema_arrow.names)
    missing = [column for column in columns if column not in names]
    if missing:
        if int(parquet.metadata.num_rows) == 0 and not names:
            return pd.DataFrame(columns=list(columns))
        raise ValueError(f"Market Data V2 parquet 缺少欄位: {missing}")
    return pd.read_parquet(path, columns=list(columns))


class ProviderSnapshotView:
    def __init__(
        self,
        *,
        project_root,
        archive: ReadyProviderSnapshotArchive,
        frame_reader: FrameReader | None = None,
        hash_fn: HashFn | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.archive = archive
        self._frame_reader = frame_reader or read_parquet_frame
        self._hash_fn = hash_fn or compute_file_sha256
        grouped: dict[str, list] = defaultdict(list)
        for item in archive.artifacts:
            grouped[str(item.dataset)].append(item)
        self._artifacts_by_dataset = {key: tuple(value) for key, value in grouped.items()}

    def dataset_artifacts(self, dataset: str):
        return self._artifacts_by_dataset.get(str(dataset or "").strip(), ())

    def iter_verified_frames(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...] | None = None,
        data_id: str | None = None,
    ) -> Iterator[pd.DataFrame]:
        name = str(dataset or "").strip()
        wanted_data_id = None if data_id is None else str(data_id).strip()
        for artifact in self.dataset_artifacts(name):
            if wanted_data_id is not None and str(artifact.data_id or "").strip() != wanted_data_id:
                continue
            request = artifact.to_request()
            path = resolve_market_data_request_parquet_path(
                self.project_root,
                self.archive.manifest_fingerprint,
                request,
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Provider Snapshot artifact 不存在: {project_relative_display_path(path, project_root=self.project_root)}"
                )
            actual_hash = str(self._hash_fn(path) or "").strip().lower()
            if actual_hash != artifact.content_sha256:
                raise ValueError(f"Provider Snapshot artifact SHA256 drift: {request.request_id}")
            frame = self._frame_reader(path, columns)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Market Data V2 provider frame reader 必須回傳 pandas.DataFrame")
            yield frame


    def latest_data_dates(self, dataset: str) -> dict[str, object]:
        """Return immutable provider-snapshot latest-date evidence by request lane.

        This is intentionally derived from the committed Provider Snapshot
        artifacts rather than from the snapshot as-of date.  A latest-available
        dataset may legitimately lag the snapshot target date (for example, a
        delayed macro series), so ``as_of_date`` is not a valid substitute for
        the latest observed row.
        """

        name = str(dataset or "").strip()
        artifacts = self.dataset_artifacts(name)
        lane_latest: dict[str, str] = {}
        dataset_latest: str | None = None
        for artifact in artifacts:
            request = artifact.to_request()
            path = resolve_market_data_request_parquet_path(
                self.project_root,
                self.archive.manifest_fingerprint,
                request,
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Provider Snapshot artifact 不存在: {project_relative_display_path(path, project_root=self.project_root)}"
                )
            actual_hash = str(self._hash_fn(path) or "").strip().lower()
            if actual_hash != artifact.content_sha256:
                raise ValueError(f"Provider Snapshot artifact SHA256 drift: {request.request_id}")
            frame = self._frame_reader(path, ("date",))
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Market Data V2 provider frame reader 必須回傳 pandas.DataFrame")
            if frame.empty:
                continue
            if "date" not in frame.columns:
                raise ValueError(f"{name} Provider Snapshot latest-date evidence 缺少 date 欄位")
            parsed = pd.to_datetime(frame["date"], errors="coerce")
            if parsed.isna().any():
                raise ValueError(f"{name} Provider Snapshot latest-date evidence 含不合法 date")
            latest = parsed.max().strftime("%Y-%m-%d")
            if latest > self.archive.as_of_date:
                raise ValueError(
                    f"{name} Provider Snapshot row date 超出 snapshot as_of_date: {latest} > {self.archive.as_of_date}"
                )
            lane_key = str(request.data_id) if request.data_id is not None else "__ALL__"
            prior_lane = lane_latest.get(lane_key)
            lane_latest[lane_key] = latest if prior_lane is None else max(prior_lane, latest)
            dataset_latest = latest if dataset_latest is None else max(dataset_latest, latest)
        return {
            "latest_data_date": dataset_latest,
            "latest_data_date_by_data_id": lane_latest,
        }

    def historical_instruments(self, *, source_dataset: str) -> tuple[str, ...]:
        ids = sorted(
            {
                str(item.data_id or "").strip()
                for item in self.dataset_artifacts(source_dataset)
                if str(item.data_id or "").strip()
            }
        )
        expected = int(self.archive.payload.get("historical_instrument_count") or 0)
        if len(ids) != expected:
            raise ValueError(
                "Market Data V2 historical instrument pool 與 Provider Snapshot 不一致: "
                f"actual={len(ids)}, expected={expected}"
            )
        return tuple(ids)


__all__ = ["FrameReader", "HashFn", "read_parquet_frame", "ProviderSnapshotView"]
