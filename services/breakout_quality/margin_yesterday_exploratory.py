"""Isolated current-vintage margin-balance observation capability for MR-13BY.

Research-exploration ONLY: the immutable archive is not a historical as-of vintage.
Never route these features to the official AO model, PIT producer or Trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from config.research import RESEARCH_MARKET_DATA_CUTOFF
from core.file_integrity import canonical_json_sha256
from core.breakout_quality_registry import get_exploratory_model_research_study
from core.market_data_research_pit_contract import build_research_v2_pit_review_contracts
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.market_data.provider_snapshot_view import ProviderSnapshotView

DATASET = "TaiwanStockMarginPurchaseShortSale"
CALENDAR_DATASET = "TaiwanStockTradingDate"
# One scientific owner: do not duplicate these identities in the feature loader.
_STUDY = get_exploratory_model_research_study("MR-13BY")
SCIENTIFIC_ID = _STUDY.model_research_id
EXPERIMENT_ARCHITECTURE = _STUDY.architecture_identity
FEATURE_CONTRACT = _STUDY.feature_contract
FEATURE_COLUMNS = (
    "Margin_Confirmed_Log_Deviation", "Short_Confirmed_Log_Deviation",
    "Margin_Confirmed_Available", "Short_Confirmed_Available",
)
SOURCE_COLUMNS = ("date", "stock_id", "MarginPurchaseYesterdayBalance", "ShortSaleYesterdayBalance")


def _verified_calendar(frames) -> pd.DatetimeIndex:
    parts = list(frames)
    if not parts or any("date" not in part.columns for part in parts):
        raise ValueError("缺少SHA驗證的TradingDate交易日曆")
    series = pd.concat([part["date"] for part in parts], ignore_index=True)
    dates = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("交易日曆包含不合法日期")
    idx = pd.DatetimeIndex(dates.dt.normalize()).sort_values()
    if idx.has_duplicates:
        raise ValueError("交易日曆包含重複日期，不能自行去重")
    return idx


def align_confirmed_balances(
    control_dates: pd.DatetimeIndex,
    trading_days: pd.DatetimeIndex,
    observations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """For prediction day D use YesterdayBalance in report on the EXACT preceding session.

    The report on D-1 describes confirmed D-2 balance.  No lookup on D,
    forward-fill, or interpolation is ever permitted.  Missing/zero are distinct.
    """
    dates = pd.DatetimeIndex(control_dates).normalize()
    calendar = pd.DatetimeIndex(trading_days).normalize()
    if dates.hasnans or dates.has_duplicates or calendar.empty or calendar.hasnans or calendar.has_duplicates:
        raise ValueError("Adjusted日期或交易日曆為空／重複／含NaT")
    if not calendar.is_monotonic_increasing:
        raise ValueError("交易日曆必須遞增")
    if not dates.is_monotonic_increasing:
        raise ValueError("Adjusted日期必須遞增")
    if (dates > calendar[-1]).any():
        raise ValueError("Adjusted日期超出已驗證TradingDate日曆尾端")
    inside = (dates >= calendar[0]) & (dates <= calendar[-1])
    if len(dates[inside].difference(calendar)):
        raise ValueError("Adjusted日期未落於已驗證TradingDate日曆")
    prior = calendar.searchsorted(dates, side="left") - 1
    has_prior = prior >= 0
    previous_dates = pd.DatetimeIndex([calendar[i] if i >= 0 else pd.NaT for i in prior])
    names = SOURCE_COLUMNS[2:]
    if not set(("date", *names)).issubset(observations.columns):
        raise ValueError("YesterdayBalance來源缺少必要欄位")
    data = observations.loc[:, ["date", *names]].copy()
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="coerce").dt.normalize()
    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError("YesterdayBalance來源日期不合法／重複；不可自行擇一")
    data = data.set_index("date")
    for column in names:
        raw = data[column]
        parsed = pd.to_numeric(raw, errors="coerce")
        if (parsed.isna() & raw.notna()).any():
            raise ValueError(f"YesterdayBalance來源非數值: {column}")
        valid = parsed.notna()
        values = parsed[valid].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"YesterdayBalance來源包含非有限／負數／非整數: {column}")
        data[column] = parsed.astype(np.float64)
    # The requested report must itself be an official session, not an inferred weekday.
    if len(data.index.difference(calendar)):
        raise ValueError("YesterdayBalance來源日期不在已驗證交易日曆")
    selected = data.reindex(previous_dates)
    values = selected[list(names)].to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & has_prior[:, None]
    # The physically absent/invalid feature is neutral only in the model tensor.
    values[~valid] = 0.0
    if not (np.isfinite(values).all() and np.all(values >= 0)):
        raise ValueError("確認餘額矩陣無效")
    return values, valid


def encode_margin_window(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-prediction 300-day log1p median-relative features + exact missing masks.

    Entire normalization scope ends at the prediction day; no global statistics,
    OOS labels or future dates are used.  True zero has mask=1 and is not imputed.
    """
    a = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if a.ndim != 2 or a.shape[1] != 2 or mask.shape != a.shape:
        raise ValueError("Margin window必須[N,2]，mask同尺寸")
    if np.any(mask & (~np.isfinite(a) | (a < 0))) or np.any(~mask & (a != 0)):
        raise ValueError("Margin window包含不合法來源／缺值非中性編碼")
    result = np.zeros((len(a), 4), dtype=np.float32)
    for col in range(2):
        seen = mask[:, col]
        if bool(seen.any()):
            logs = np.log1p(a[seen, col])
            median = np.median(logs)
            result[seen, col] = (logs - median).astype(np.float32)
        result[:, col + 2] = seen.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Margin feature含非有限值")
    return result


class MarginAugmentedDailyFeatureBank:
    """Lazy AO adjusted feature bank + four isolated margin observation channels.

    The base retains canonical target, sample and price inputs exactly; no mutation.
    """
    def __init__(self, base, arrays: tuple[tuple[np.ndarray, np.ndarray], ...]):
        if not hasattr(base, "_ticker_ids") or not hasattr(base, "_frame_dates") or not hasattr(base, "_source_positions"):
            raise ValueError("需要canonical LazyDailyFeatureBank 進行source/date alignment")
        self._base = base
        self._arrays = tuple(arrays)
        if len(self._arrays) != len(base._frame_dates):
            raise ValueError("Margin feature ticker count與AO不一致")
        for (v, m), dates in zip(self._arrays, base._frame_dates):
            if v.shape != (len(dates), 2) or m.shape != v.shape:
                raise ValueError("Margin feature dates與AO價格日期未對齊")
        if len(base.feature_columns) != 10:
            raise ValueError("MR-13BY必須重用AO的10欄原始feature")
        self.feature_columns = (*base.feature_columns, *FEATURE_COLUMNS)
        # Precompute causal rolling medians once per ticker, rather than running
        # two numpy median sorts for each of ~2 million observations every epoch.
        # The median for scoring bar D sees only observations through D.
        length = int(base.shape[1])
        self._log_values = []
        self._rolling_medians = []
        for values, mask in self._arrays:
            if (np.any(mask & (~np.isfinite(values) | (values < 0)))
                    or np.any(~mask & (values != 0))):
                raise ValueError("Margin來源陣列包含無效值或未遮罩缺失")
            logged = np.zeros_like(values, dtype=np.float64)
            logged[mask] = np.log1p(values[mask])
            median = (
                pd.DataFrame(np.where(mask, logged, np.nan))
                .rolling(window=length, min_periods=1)
                .median()
                .to_numpy(dtype=np.float64)
            )
            self._log_values.append(logged)
            self._rolling_medians.append(median)

    @property
    def shape(self):
        return (*self._base.shape[:2], len(self.feature_columns))

    @property
    def ndim(self):
        return 3

    def __len__(self):
        return len(self._base)

    def __getitem__(self, item):
        baseline = np.asarray(self._base[item], dtype=np.float32)
        scalar = baseline.ndim == 2
        ids = self._indices(item)
        result = np.zeros((len(ids), self.shape[1], self.shape[2]), dtype=np.float32)
        result[:, :, :baseline.shape[-1]] = baseline[None, :, :] if scalar else baseline
        window_len = self.shape[1]
        for k, group_id in enumerate(ids):
            ticker_id = int(self._base._ticker_ids[group_id])
            pos = int(self._base._source_positions[group_id])
            start = pos - window_len + 1
            if start < 0:
                raise ValueError("AO選樣沒有足夠300日觀察窗")
            vals, masks = self._arrays[ticker_id]
            seen = masks[start:pos + 1]
            logged = self._log_values[ticker_id][start:pos + 1]
            median = self._rolling_medians[ticker_id][pos]
            encoded = np.zeros((window_len, 4), dtype=np.float32)
            for col in range(2):
                if seen[:, col].any():
                    if not np.isfinite(median[col]):
                        raise ValueError("Margin視窗含有效餘額但中位數缺失")
                    encoded[seen[:, col], col] = (
                        logged[seen[:, col], col] - median[col]
                    ).astype(np.float32)
            encoded[:, 2:] = seen.astype(np.float32)
            result[k, :, baseline.shape[-1]:] = encoded
        return result[0] if scalar else result

    def _indices(self, item):
        if isinstance(item, (int, np.integer)):
            ids = np.asarray([item], dtype=np.int64)
        elif isinstance(item, slice):
            ids = np.arange(len(self))[item]
        else:
            ids = np.asarray(item)
            if ids.dtype == bool:
                if ids.ndim != 1 or len(ids) != len(self):
                    raise IndexError("feature-bank boolean index長度不一致")
                ids = np.flatnonzero(ids)
            ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if (ids < -len(self)).any() or (ids >= len(self)).any():
            raise IndexError("group index超出範圍")
        return np.where(ids < 0, ids + len(self), ids)


@dataclass(frozen=True)
class MarginSourceEvidence:
    arrays: tuple[tuple[np.ndarray, np.ndarray], ...]
    source_identity: dict[str, object]
    coverage: dict[str, object]


def load_margin_source_for_ao_bundle(
    project_root: Path,
    bundle,
    *,
    snapshot_fingerprint: str | None = None,
) -> MarginSourceEvidence:
    root = Path(project_root).resolve()
    contracts = [x for x in build_research_v2_pit_review_contracts() if x.dataset == DATASET]
    if len(contracts) != 1 or contracts[0].review_status != "REVIEW_REQUIRED":
        raise ValueError("此探索實驗要求Margin的正式Research PIT仍為REVIEW_REQUIRED")
    archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=snapshot_fingerprint)
    if archive.as_of_date < str(RESEARCH_MARKET_DATA_CUTOFF):
        raise ValueError("Provider Snapshot截止日不足")
    view = ProviderSnapshotView(project_root=root, archive=archive)
    if not view.dataset_artifacts(DATASET) or not view.dataset_artifacts(CALENDAR_DATASET):
        raise ValueError("READY Provider Snapshot缺融資融券或驗證日曆")
    calendar = _verified_calendar(view.iter_verified_frames(CALENDAR_DATASET, columns=("date",)))
    bank = bundle.feature_bank
    tids = np.asarray(bank._ticker_ids, dtype=np.int64)
    unique, indexes = np.unique(tids, return_index=True)
    if tids.ndim != 1 or (tids < 0).any() or (tids >= len(bank._frame_dates)).any():
        raise ValueError("AO ticker/frame index超出範圍")
    if isinstance(bundle.group_table["ticker"].dtype, pd.CategoricalDtype):
        # Canonical Daily ranker creates ticker labels by Categorical.from_codes
        # using ALL sanitized frames, including frames without target-eligible
        # groups.  Never assume unique score-ticker count == frame count.
        categories = list(bundle.group_table["ticker"].cat.categories)
        if len(categories) != len(bank._frame_dates):
            raise ValueError("AO source frames與categorical ticker語意不一致")
        tickers = tuple(str(value) for value in categories)
    else:
        # Synthetic / legacy object-string fixtures lack category metadata.
        # Frames without scoring groups cannot affect fitting, so leave them
        # as masked-missing without manufacturing an unknown ticker identity.
        observed = {
            int(ticker_id): str(bundle.group_table.iloc[int(index)]["ticker"])
            for ticker_id, index in zip(unique.tolist(), indexes.tolist())
        }
        tickers = tuple(observed.get(i) for i in range(len(bank._frame_dates)))
    if len(set(t for t in tickers if t is not None)) != len([t for t in tickers if t is not None]):
        raise ValueError("AO ticker名稱重複")
    for ticker_id, first_index in zip(unique.tolist(), indexes.tolist()):
        label = str(bundle.group_table.iloc[int(first_index)]["ticker"])
        if label != tickers[ticker_id]:
            raise ValueError("AO ticker ID與股票名稱來源索引不一致")
    wanted_tickers = tuple(ticker for ticker in tickers if ticker is not None)
    records: dict[str, list[pd.DataFrame]] = {}
    for fragment in view.iter_verified_frames_many_data_ids(DATASET, columns=SOURCE_COLUMNS, data_ids=wanted_tickers):
        if fragment.empty:
            continue
        if set(SOURCE_COLUMNS) - set(fragment.columns):
            raise ValueError("Provider margin parquet缺必要YesterdayBalance欄位")
        if pd.api.types.is_numeric_dtype(fragment["stock_id"]):
            raise ValueError("FinMind股票代碼數值化，無法保證前導零")
        ids = fragment["stock_id"].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any():
            raise ValueError("FinMind margin來源包含無效stock_id")
        work = fragment.loc[:, SOURCE_COLUMNS].copy()
        work["stock_id"] = ids.astype(str)
        for ticker, rows in work.groupby("stock_id", sort=False):
            if ticker not in wanted_tickers:
                continue
            records.setdefault(ticker, []).append(rows.drop(columns="stock_id"))
    results: list[tuple[np.ndarray, np.ndarray]] = []
    annual = {}
    for ticker, frame_dates in zip(tickers, bank._frame_dates):
        series = pd.concat(records[ticker], ignore_index=True) if ticker in records else pd.DataFrame(columns=["date", *SOURCE_COLUMNS[2:]])
        values, mask = align_confirmed_balances(pd.DatetimeIndex(frame_dates), calendar, series)
        results.append((values, mask))
        year = pd.DatetimeIndex(frame_dates).year
        for yr in np.unique(year):
            key = str(int(yr))
            a = annual.setdefault(key, {"stock_days": 0, "margin_available": 0, "short_available": 0, "both_available": 0})
            selected = year == yr
            a["stock_days"] += int(selected.sum())
            a["margin_available"] += int(mask[selected, 0].sum())
            a["short_available"] += int(mask[selected, 1].sum())
            a["both_available"] += int(mask[selected].all(axis=1).sum())
    identities = [
        {"dataset": dataset, "request_id": artifact.request_id, "sha256": artifact.content_sha256}
        for dataset in (DATASET, CALENDAR_DATASET)
        for artifact in view.dataset_artifacts(dataset)
    ]
    source_identity = {
        "source_contract": FEATURE_CONTRACT,
        "snapshot_fingerprint": archive.snapshot_fingerprint,
        "manifest_fingerprint": archive.manifest_fingerprint,
        "provider_artifact_inventory_sha256": canonical_json_sha256(identities),
        "research_pit": "REVIEW_REQUIRED",
        "historical_vintage_proven": False,
        "model_input_authorized_in_formal_workflow": False,
    }
    return MarginSourceEvidence(arrays=tuple(results), source_identity=source_identity,
                                coverage={"by_prediction_price_year": annual})
