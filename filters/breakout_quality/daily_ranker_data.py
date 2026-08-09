"""Daily-universal stock/date samples for cross-sectional ranker research.

The store is intentionally index-based: it never materializes one 300x10 tensor per
stock-day.  Canonical OHLCV frames remain the source of truth and sequence windows are
materialized on demand by the existing breakout-quality feature function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import get_breakout_quality_experiment_profile
from filters.breakout_quality.continuous_ranker_data import ContinuousRankerDataBundle
from filters.breakout_quality.continuous_target import (
    DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
    StrategyAlignedContinuousTargetSpec,
    build_daily_opportunity_no_time_contract,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY, FEATURE_COLUMNS
from filters.breakout_quality.features import build_breakout_quality_sequence_feature
from filters.breakout_quality.models.spec import get_model_spec, validate_model_sequence_length
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from filters.breakout_quality.workflow_io import (
    PROJECT_ROOT,
    discover_dataset_csv_inputs,
    load_dataset_frame,
    load_validated_dataset_bundle,
)


class LazyDailyFeatureBank:
    """Array-like stock/day feature bank backed by canonical sanitized OHLCV frames."""

    def __init__(
        self,
        *,
        frames: tuple[pd.DataFrame, ...],
        benchmark: pd.DataFrame,
        ticker_ids: np.ndarray,
        source_positions: np.ndarray,
        policy,
    ) -> None:
        self._frames = tuple(frames)
        self._benchmark = benchmark
        self._ticker_ids = np.asarray(ticker_ids, dtype=np.int32)
        self._source_positions = np.asarray(source_positions, dtype=np.int32)
        self._policy = policy
        if self._ticker_ids.ndim != 1 or self._source_positions.ndim != 1:
            raise ValueError("daily feature index必須是一維")
        if self._ticker_ids.shape != self._source_positions.shape:
            raise ValueError("daily feature ticker/source index長度不一致")
        if len(self._ticker_ids) and (
            int(self._ticker_ids.min()) < 0 or int(self._ticker_ids.max()) >= len(self._frames)
        ):
            raise ValueError("daily feature ticker index超出frame範圍")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(len(self._ticker_ids)),
            int(self._policy.feature_window_bars),
            int(len(FEATURE_COLUMNS)),
        )

    @property
    def ndim(self) -> int:
        return 3

    def __len__(self) -> int:
        return int(len(self._ticker_ids))

    def __getitem__(self, item):
        scalar = isinstance(item, (int, np.integer))
        ids = np.asarray([int(item)], dtype=np.int64) if scalar else np.arange(len(self))[item]
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        output = np.empty(
            (len(ids), int(self._policy.feature_window_bars), len(FEATURE_COLUMNS)),
            dtype=np.float32,
        )
        for out_index, group_id in enumerate(ids.tolist()):
            ticker_id = int(self._ticker_ids[group_id])
            source_pos = int(self._source_positions[group_id])
            sequence = build_breakout_quality_sequence_feature(
                self._frames[ticker_id],
                self._benchmark,
                event_pos=source_pos,
                policy=self._policy,
            )
            if sequence is None:
                raise RuntimeError(
                    "daily feature index與canonical feature builder不一致: "
                    f"group_index={group_id}, ticker_id={ticker_id}, source_pos={source_pos}"
                )
            output[out_index] = sequence
        return output[0] if scalar else output


@dataclass(frozen=True)
class DailyRankerSplit:
    inner_train_ids: np.ndarray
    validation_ids: np.ndarray
    selection_ids: np.ndarray
    oos_ids: np.ndarray
    report: dict[str, Any]


def _source_data_end(summary: dict[str, Any]) -> str:
    source_range = summary.get("source_data_date_range")
    if not isinstance(source_range, dict) or not str(source_range.get("end") or "").strip():
        raise ValueError("daily ranker需要dataset summary的source_data_date_range.end")
    return str(source_range["end"])


def _daily_targets_for_positions(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    spec: StrategyAlignedContinuousTargetSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize the exact adverse-first no-time target over one ticker.

    This is a compute optimization only.  Semantics match
    ``daily_opportunity_no_time_target_from_cached_path``: barrier-day high is
    excluded, the earliest maximum safe high is selected, and adverse excursion
    is measured through that selected peak bar.
    """

    source_positions = np.asarray(positions, dtype=np.int64)
    if source_positions.ndim != 1:
        raise ValueError("daily target positions必須是一維")
    count = int(len(source_positions))
    if count == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)

    horizon = int(spec.horizon_bars)
    offsets = np.arange(1, horizon + 1, dtype=np.int64)
    future_index = source_positions[:, None] + offsets[None, :]
    close = frame["Close"].to_numpy(dtype=np.float64, copy=False)
    high = frame["High"].to_numpy(dtype=np.float64, copy=False)[future_index]
    low = frame["Low"].to_numpy(dtype=np.float64, copy=False)[future_index]
    anchor = close[source_positions]

    valid = (
        np.isfinite(anchor)
        & (anchor > 0.0)
        & np.all(
            np.isfinite(high)
            & np.isfinite(low)
            & (high > 0.0)
            & (low > 0.0)
            & (high >= low),
            axis=1,
        )
    )
    target = np.full(count, np.nan, dtype=np.float64)
    if not bool(np.any(valid)):
        return target.astype(np.float32), valid

    valid_rows = np.flatnonzero(valid)
    a = anchor[valid_rows]
    h = high[valid_rows]
    l = low[valid_rows]
    risk_budget = float(spec.risk_budget_return)
    barrier = a[:, None] * (1.0 - risk_budget)
    breach = l <= barrier
    any_breach = breach.any(axis=1)
    first_breach_zero = np.where(any_breach, breach.argmax(axis=1), horizon)
    bar_index = np.arange(horizon, dtype=np.int64)[None, :]
    safe = bar_index < first_breach_zero[:, None]

    favorable_matrix = h / a[:, None] - 1.0
    safe_favorable = np.where(safe, favorable_matrix, -np.inf)
    has_safe_bar = safe.any(axis=1)
    best_zero = safe_favorable.argmax(axis=1)
    row_index = np.arange(len(valid_rows), dtype=np.int64)
    best_favorable = safe_favorable[row_index, best_zero]
    running_low = np.minimum.accumulate(l, axis=1)
    adverse = np.maximum(0.0, 1.0 - running_low[row_index, best_zero] / a)

    favorable = np.where(has_safe_bar, best_favorable, 0.0)
    adverse = np.where(has_safe_bar, adverse, risk_budget)
    target[valid_rows] = favorable / risk_budget - adverse / risk_budget
    finite = np.isfinite(target)
    valid &= finite
    target[~valid] = np.nan
    return target.astype(np.float32), valid


def load_daily_universal_ranker_data(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    preload_feature_bank: bool,
    allow_stale_source: bool,
    project_root: str | Path = PROJECT_ROOT,
) -> ContinuousRankerDataBundle:
    """Build a lightweight daily stock/day index and lazy feature provider.

    ``preload_feature_bank`` is intentionally ignored for this sample scope: expanding all
    windows would multiply storage/RAM by the number of eligible stock-days.  Sanitized
    ticker frames are kept in memory, while each 300x10 window is generated only when a
    training/evaluation batch requests it.
    """

    del preload_feature_bank
    root = Path(project_root)
    profile = get_breakout_quality_experiment_profile(experiment_profile)
    if str(profile.continuous_target_id) != DAILY_OPPORTUNITY_NO_TIME_TARGET_ID:
        raise ValueError("daily universal ranker target identity不一致")
    model_spec = get_model_spec(model_architecture)
    if (
        bool(model_spec.requires_market_set)
        or bool(model_spec.use_dataset_context)
        or bool(model_spec.derived_context_features)
    ):
        raise ValueError("daily universal ranker第一版只支援sequence-only architecture")

    # The existing official event dataset remains the source-selection/inventory truth.
    summary, _indexed_features, _context, _labels, event_rows = load_validated_dataset_bundle(
        filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(allow_stale_source),
    )
    dataset_profile = str(summary.get("dataset") or "").strip().lower()
    if dataset_profile not in {"reduced", "full"}:
        raise ValueError("daily ranker無法從dataset summary解析source dataset profile")
    outer_policy = resolve_breakout_quality_outer_policy(
        root,
        source_data_end_date=_source_data_end(summary),
    )
    sample_start = pd.Timestamp(str(outer_policy["selection_start_date"]))
    sample_end = pd.Timestamp(str(outer_policy["effective_oos_end_date"]))

    csv_inputs, _duplicate_lines = discover_dataset_csv_inputs(root, dataset_profile)
    input_map = {str(ticker): Path(path) for ticker, path in csv_inputs}
    benchmark_ticker = str(DEFAULT_LABEL_POLICY.benchmark_ticker)
    benchmark_path = input_map.get(benchmark_ticker)
    if benchmark_path is None:
        raise FileNotFoundError(f"daily ranker找不到benchmark ticker: {benchmark_ticker}")
    min_rows = max(
        int(DEFAULT_LABEL_POLICY.feature_window_bars) + 1,
        int(DEFAULT_LABEL_POLICY.label_horizon_bars) + 1,
    )
    benchmark = load_dataset_frame(benchmark_path, benchmark_ticker, min_rows=min_rows)
    benchmark_index = pd.DatetimeIndex(benchmark.index).normalize()

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    tickers: list[str] = []
    frames: list[pd.DataFrame] = []
    ticker_id_chunks: list[np.ndarray] = []
    source_pos_chunks: list[np.ndarray] = []
    date_chunks: list[np.ndarray] = []
    label_end_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    skipped_tickers = 0

    for ticker, path in csv_inputs:
        ticker = str(ticker)
        if ticker == benchmark_ticker:
            continue
        try:
            frame = load_dataset_frame(path, ticker, min_rows=min_rows)
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
            skipped_tickers += 1
            continue
        ticker_id = len(tickers)
        first_pos = int(DEFAULT_LABEL_POLICY.feature_window_bars) - 1
        last_pos = len(frame) - int(spec.horizon_bars) - 1
        if last_pos < first_pos:
            skipped_tickers += 1
            continue
        candidate_positions = np.arange(first_pos, last_pos + 1, dtype=np.int64)
        frame_dates = pd.DatetimeIndex(frame.index).normalize()
        candidate_dates = frame_dates.take(candidate_positions)
        in_period = (candidate_dates >= sample_start) & (candidate_dates <= sample_end)
        benchmark_pos = benchmark_index.get_indexer(candidate_dates)
        eligible = np.asarray(in_period, dtype=bool) & (benchmark_pos >= first_pos)
        local_positions = candidate_positions[eligible]
        if len(local_positions) == 0:
            skipped_tickers += 1
            continue
        local_targets, target_valid = _daily_targets_for_positions(
            frame, local_positions, spec=spec
        )
        local_positions = local_positions[target_valid]
        local_targets = local_targets[target_valid]
        if len(local_positions) == 0:
            skipped_tickers += 1
            continue
        local_dates_index = frame_dates.take(local_positions)
        local_label_end_index = frame_dates.take(
            local_positions + int(spec.horizon_bars)
        )
        tickers.append(ticker)
        frames.append(frame)
        count = len(local_positions)
        ticker_id_chunks.append(np.full(count, ticker_id, dtype=np.int32))
        source_pos_chunks.append(np.asarray(local_positions, dtype=np.int32))
        date_chunks.append(local_dates_index.to_numpy(dtype="datetime64[D]"))
        label_end_chunks.append(local_label_end_index.to_numpy(dtype="datetime64[D]"))
        target_chunks.append(np.asarray(local_targets, dtype=np.float32))

    if not target_chunks:
        raise ValueError("daily universal ranker沒有任何有效stock-day sample")

    ticker_ids = np.concatenate(ticker_id_chunks)
    source_positions = np.concatenate(source_pos_chunks)
    dates = np.concatenate(date_chunks)
    label_end_dates = np.concatenate(label_end_chunks)
    raw_target = np.concatenate(target_chunks)
    group_count = int(len(raw_target))
    group_index = np.arange(group_count, dtype=np.int64)
    ticker_categorical = pd.Categorical.from_codes(ticker_ids, categories=tickers)
    group_table = pd.DataFrame(
        {
            "ticker": ticker_categorical,
            "ticker_id": ticker_ids,
            "date": pd.to_datetime(dates),
            "group_index": group_index,
            "source_pos": source_positions,
            "label": np.full(group_count, -1, dtype=np.int8),
            "label_eval_end_date": pd.to_datetime(label_end_dates),
        }
    )
    feature_bank = LazyDailyFeatureBank(
        frames=tuple(frames),
        benchmark=benchmark,
        ticker_ids=ticker_ids,
        source_positions=source_positions,
        policy=DEFAULT_LABEL_POLICY,
    )
    validate_model_sequence_length(model_spec, int(feature_bank.shape[1]))
    target_contract = build_daily_opportunity_no_time_contract(DEFAULT_LABEL_POLICY)
    target_manifest = {
        "target_id": DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
        "target_contract": target_contract,
        "sample_scope": "daily_eligible_stock_days",
        "sample_count": group_count,
        "ticker_count": int(len(tickers)),
        "date_range": {
            "start": str(pd.Timestamp(group_table["date"].min()).date()),
            "end": str(pd.Timestamp(group_table["date"].max()).date()),
        },
        "feature_storage": "lazy_from_canonical_ohlcv_no_expanded_daily_feature_bank",
    }
    daily_summary = {
        "filter_id": filter_id,
        "dataset": dataset_profile,
        "source_dataset_artifacts": summary.get("dataset_artifacts"),
        "source_data_inventory": summary.get("source_data_inventory"),
        "source_data_date_range": summary.get("source_data_date_range"),
        "policy": summary.get("policy"),
        "sample_scope": "daily_eligible_stock_days",
        "sample_count": group_count,
        "ticker_count": int(len(tickers)),
        "skipped_ticker_count": int(skipped_tickers),
        "feature_storage": "lazy",
    }
    # ``events`` is retained only for the shared bundle interface; one row now equals one
    # stock-day rather than a breakout event.
    return ContinuousRankerDataBundle(
        summary=daily_summary,
        events=group_table.copy(),
        labels=np.full(group_count, -1, dtype=np.int8),
        feature_bank=feature_bank,
        event_group_index=group_index,
        group_table=group_table,
        group_context=np.empty((group_count, 0), dtype=np.float32),
        raw_target=raw_target,
        target_valid=np.ones(group_count, dtype=bool),
        target_manifest=target_manifest,
        outer_policy=outer_policy,
        profile=profile,
        model_spec=model_spec,
    )


def build_daily_ranker_split(bundle: ContinuousRankerDataBundle, *, inner_validation_months: int) -> DailyRankerSplit:
    """Mirror the canonical no-lookahead split directly at stock/day group level."""

    dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    label_end = pd.to_datetime(
        bundle.group_table["label_eval_end_date"], errors="raise"
    ).dt.normalize()
    policy = bundle.outer_policy
    selection_start = pd.Timestamp(str(policy["selection_start_date"]))
    selection_end = pd.Timestamp(str(policy["selection_end_date"]))
    oos_start = pd.Timestamp(str(policy["oos_start_date"]))
    oos_end = pd.Timestamp(str(policy["effective_oos_end_date"]))
    validation_start = (
        selection_end + pd.Timedelta(days=1) - pd.DateOffset(months=int(inner_validation_months))
    ).normalize()
    if not selection_start < validation_start <= selection_end:
        raise ValueError("daily ranker inner validation期間不合法")

    selection_mask = (dates >= selection_start) & (dates <= selection_end) & (label_end < oos_start)
    inner_train_mask = selection_mask & (dates < validation_start) & (label_end < validation_start)
    validation_mask = selection_mask & (dates >= validation_start)
    oos_mask = (dates >= oos_start) & (dates <= oos_end) & (label_end <= oos_end)

    def ids(mask) -> np.ndarray:
        return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)

    inner_train_ids = ids(inner_train_mask)
    validation_ids = ids(validation_mask)
    selection_ids = ids(selection_mask)
    oos_ids = ids(oos_mask)
    for name, values in (
        ("inner_train", inner_train_ids),
        ("validation", validation_ids),
        ("selection", selection_ids),
        ("oos", oos_ids),
    ):
        if len(values) < 20:
            raise ValueError(f"daily ranker {name}有效sample不足: {len(values)}")
    return DailyRankerSplit(
        inner_train_ids=inner_train_ids,
        validation_ids=validation_ids,
        selection_ids=selection_ids,
        oos_ids=oos_ids,
        report={
            "sample_scope": "daily_eligible_stock_days",
            "selection_start_date": str(selection_start.date()),
            "selection_end_date": str(selection_end.date()),
            "inner_validation_start_date": str(validation_start.date()),
            "oos_start_date": str(oos_start.date()),
            "oos_end_date": str(oos_end.date()),
            "counts": {
                "inner_train": int(len(inner_train_ids)),
                "validation": int(len(validation_ids)),
                "selection": int(len(selection_ids)),
                "oos": int(len(oos_ids)),
            },
            "no_lookahead": True,
            "selection_label_end_before_oos": True,
            "inner_train_label_end_before_validation": True,
            "oos_label_end_within_oos": True,
        },
    )


__all__ = [
    "DailyRankerSplit",
    "LazyDailyFeatureBank",
    "build_daily_ranker_split",
    "load_daily_universal_ranker_data",
]
