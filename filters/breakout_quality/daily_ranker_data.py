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
from filters.breakout_quality.continuous_ranker_data import (
    ContinuousRankerDataBundle,
    build_same_date_percentile_targets,
)
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
    DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID,
    DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
    StrategyAlignedContinuousTargetSpec,
    build_daily_full_horizon_opportunity_contract,
    build_daily_full_horizon_pure_mfe_contract,
    build_daily_full_horizon_low_adverse_contract,
    build_daily_full_horizon_equal_rank_mfe_low_adverse_contract,
    build_daily_opportunity_no_time_contract,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY, FEATURE_COLUMNS
from filters.breakout_quality.risk_normalized_target import (
    DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID,
    RISK_GEOMETRY_CONTEXT_FEATURES,
    build_risk_target_contract,
    compute_targets_and_context_for_positions,
    load_min_roos_risk_schedule,
)
from filters.breakout_quality.features import normalize_ohlcv_array_window
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
        benchmark_positions: np.ndarray,
        policy,
    ) -> None:
        self._frame_arrays = tuple(
            frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(
                dtype=np.float64,
                copy=True,
            )
            for frame in frames
        )
        self._benchmark_array = benchmark[
            ["Open", "High", "Low", "Close", "Volume"]
        ].to_numpy(dtype=np.float64, copy=True)
        self._ticker_ids = np.asarray(ticker_ids, dtype=np.int32)
        self._source_positions = np.asarray(source_positions, dtype=np.int32)
        self._benchmark_positions = np.asarray(benchmark_positions, dtype=np.int32)
        self._policy = policy
        self._benchmark_feature_cache: dict[int, np.ndarray] = {}
        if (
            self._ticker_ids.ndim != 1
            or self._source_positions.ndim != 1
            or self._benchmark_positions.ndim != 1
        ):
            raise ValueError("daily feature index必須是一維")
        if not (
            self._ticker_ids.shape
            == self._source_positions.shape
            == self._benchmark_positions.shape
        ):
            raise ValueError("daily feature ticker/source/benchmark index長度不一致")
        if len(self._ticker_ids) and (
            int(self._ticker_ids.min()) < 0
            or int(self._ticker_ids.max()) >= len(self._frame_arrays)
        ):
            raise ValueError("daily feature ticker index超出frame範圍")
        if len(self._benchmark_positions) and (
            int(self._benchmark_positions.min()) < 0
            or int(self._benchmark_positions.max()) >= len(self._benchmark_array)
        ):
            raise ValueError("daily feature benchmark index超出範圍")

    def _normalized_window(self, values: np.ndarray, source_pos: int) -> np.ndarray:
        feature_window = int(self._policy.feature_window_bars)
        start_pos = int(source_pos) - feature_window + 1
        if start_pos < 0 or int(source_pos) >= len(values):
            raise RuntimeError("daily feature source position沒有足夠history")
        window = values[start_pos : int(source_pos) + 1]
        if len(window) != feature_window:
            raise RuntimeError("daily feature window長度與policy不一致")
        anchor_close = float(values[int(source_pos), 3])
        normalized = normalize_ohlcv_array_window(window, anchor_close)
        if not np.isfinite(normalized).all():
            raise RuntimeError("daily feature window產生non-finite value")
        return normalized

    def _benchmark_feature(self, benchmark_pos: int) -> np.ndarray:
        position = int(benchmark_pos)
        cached = self._benchmark_feature_cache.get(position)
        if cached is not None:
            return cached
        feature = self._normalized_window(self._benchmark_array, position)
        self._benchmark_feature_cache[position] = feature
        return feature

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
        if scalar:
            ids = np.asarray([int(item)], dtype=np.int64)
        elif isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            ids = np.arange(start, stop, step, dtype=np.int64)
        else:
            ids = np.asarray(item)
            if ids.dtype == bool:
                if ids.ndim != 1 or len(ids) != len(self):
                    raise IndexError("daily feature boolean index長度不一致")
                ids = np.flatnonzero(ids)
            else:
                ids = np.asarray(ids, dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if bool(np.any(ids < -len(self))) or bool(np.any(ids >= len(self))):
            raise IndexError("daily feature group index超出範圍")
        ids = np.where(ids < 0, ids + len(self), ids)
        output = np.empty(
            (len(ids), int(self._policy.feature_window_bars), len(FEATURE_COLUMNS)),
            dtype=np.float32,
        )
        for out_index, group_id in enumerate(ids.tolist()):
            ticker_id = int(self._ticker_ids[group_id])
            source_pos = int(self._source_positions[group_id])
            benchmark_pos = int(self._benchmark_positions[group_id])
            output[out_index, :, :5] = self._normalized_window(
                self._frame_arrays[ticker_id],
                source_pos,
            )
            output[out_index, :, 5:] = self._benchmark_feature(benchmark_pos)
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


@dataclass(frozen=True)
class DailyOpportunityTargetBatch:
    target_raw_r: np.ndarray
    valid_mask: np.ndarray
    favorable_return: np.ndarray
    adverse_return_to_peak: np.ndarray
    opportunity_bar: np.ndarray
    first_risk_breach_bar: np.ndarray
    minimum_low_return: np.ndarray


def compute_daily_opportunity_target_batch(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    spec: StrategyAlignedContinuousTargetSpec,
    target_id: str,
) -> DailyOpportunityTargetBatch:
    """Vectorize the canonical daily opportunity targets over one ticker.

    ``daily_opportunity_no_time_r_v1`` keeps the historical adverse-first
    risk-barrier truncation. ``daily_full_horizon_opportunity_r_v1`` uses the
    same horizon/R scale/adverse-to-peak semantics but treats a barrier touch as
    diagnostic only. ``daily_full_horizon_pure_mfe_r_v1`` keeps the same full
    horizon and selected peak but does not deduct adverse-to-peak from target R.
    ``daily_full_horizon_low_adverse_r_v1`` keeps that same selected peak but
    ranks only negative adverse-to-peak R, so higher means a safer path.
    """

    source_positions = np.asarray(positions, dtype=np.int64)
    if source_positions.ndim != 1:
        raise ValueError("daily target positions必須是一維")
    count = int(len(source_positions))
    if count == 0:
        return DailyOpportunityTargetBatch(
            target_raw_r=np.empty(0, dtype=np.float32),
            valid_mask=np.empty(0, dtype=bool),
            favorable_return=np.empty(0, dtype=np.float32),
            adverse_return_to_peak=np.empty(0, dtype=np.float32),
            opportunity_bar=np.empty(0, dtype=np.int16),
            first_risk_breach_bar=np.empty(0, dtype=np.int16),
            minimum_low_return=np.empty(0, dtype=np.float32),
        )
    if target_id not in {
        DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
        DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
        DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
    }:
        raise ValueError(f"不支援的daily opportunity target: {target_id}")

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
    favorable = np.full(count, np.nan, dtype=np.float64)
    adverse = np.full(count, np.nan, dtype=np.float64)
    opportunity = np.full(count, -1, dtype=np.int16)
    first_breach = np.full(count, -1, dtype=np.int16)
    minimum_low_return = np.full(count, np.nan, dtype=np.float64)
    if not bool(np.any(valid)):
        return DailyOpportunityTargetBatch(
            target_raw_r=target.astype(np.float32),
            valid_mask=valid,
            favorable_return=favorable.astype(np.float32),
            adverse_return_to_peak=adverse.astype(np.float32),
            opportunity_bar=opportunity,
            first_risk_breach_bar=first_breach,
            minimum_low_return=minimum_low_return.astype(np.float32),
        )

    valid_rows = np.flatnonzero(valid)
    a = anchor[valid_rows]
    h = high[valid_rows]
    l = low[valid_rows]
    risk_budget = float(spec.risk_budget_return)
    barrier = a[:, None] * (1.0 - risk_budget)
    breach = l <= barrier
    any_breach = breach.any(axis=1)
    first_breach_zero = np.where(any_breach, breach.argmax(axis=1), horizon)
    first_breach_valid = np.where(any_breach, first_breach_zero + 1, -1).astype(np.int16)
    row_index = np.arange(len(valid_rows), dtype=np.int64)
    running_low = np.minimum.accumulate(l, axis=1)
    minimum_low_return_valid = np.min(l, axis=1) / a - 1.0
    favorable_matrix = h / a[:, None] - 1.0

    if target_id == DAILY_OPPORTUNITY_NO_TIME_TARGET_ID:
        bar_index = np.arange(horizon, dtype=np.int64)[None, :]
        safe = bar_index < first_breach_zero[:, None]
        safe_favorable = np.where(safe, favorable_matrix, -np.inf)
        has_safe_bar = safe.any(axis=1)
        best_zero = safe_favorable.argmax(axis=1)
        best_favorable = safe_favorable[row_index, best_zero]
        selected_adverse = np.maximum(
            0.0, 1.0 - running_low[row_index, best_zero] / a
        )
        selected_favorable = np.where(has_safe_bar, best_favorable, 0.0)
        selected_adverse = np.where(has_safe_bar, selected_adverse, risk_budget)
        selected_opportunity = np.where(
            has_safe_bar, best_zero + 1, np.where(any_breach, first_breach_zero + 1, 1)
        )
    else:
        best_zero = favorable_matrix.argmax(axis=1)
        selected_favorable = favorable_matrix[row_index, best_zero]
        selected_adverse = np.maximum(
            0.0, 1.0 - running_low[row_index, best_zero] / a
        )
        selected_opportunity = best_zero + 1

    if target_id == DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID:
        selected_target = selected_favorable / risk_budget
    elif target_id == DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID:
        selected_target = -selected_adverse / risk_budget
    else:
        selected_target = selected_favorable / risk_budget - selected_adverse / risk_budget
    target[valid_rows] = selected_target
    favorable[valid_rows] = selected_favorable
    adverse[valid_rows] = selected_adverse
    opportunity[valid_rows] = np.asarray(selected_opportunity, dtype=np.int16)
    first_breach[valid_rows] = first_breach_valid
    minimum_low_return[valid_rows] = minimum_low_return_valid
    finite = np.isfinite(target) & np.isfinite(favorable) & np.isfinite(adverse)
    valid &= finite
    target[~valid] = np.nan
    favorable[~valid] = np.nan
    adverse[~valid] = np.nan
    opportunity[~valid] = -1
    first_breach[~valid] = -1
    minimum_low_return[~valid] = np.nan
    return DailyOpportunityTargetBatch(
        target_raw_r=target.astype(np.float32),
        valid_mask=valid,
        favorable_return=favorable.astype(np.float32),
        adverse_return_to_peak=adverse.astype(np.float32),
        opportunity_bar=opportunity,
        first_risk_breach_bar=first_breach,
        minimum_low_return=minimum_low_return.astype(np.float32),
    )


def _daily_targets_for_positions(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    spec: StrategyAlignedContinuousTargetSpec,
    target_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    batch = compute_daily_opportunity_target_batch(
        frame, positions, spec=spec, target_id=target_id
    )
    return batch.target_raw_r, batch.valid_mask



def build_equal_rank_mfe_low_adverse_target(
    favorable_r: np.ndarray,
    adverse_r: np.ndarray,
    valid_mask: np.ndarray,
    group_dates: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build MR-13N from fixed equal weights on same-date component percentiles."""

    favorable = np.asarray(favorable_r, dtype=np.float64)
    adverse = np.asarray(adverse_r, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if favorable.shape != adverse.shape or favorable.shape != valid.shape:
        raise ValueError("MR-13N component target shape不一致")
    mfe_percentile = build_same_date_percentile_targets(favorable, valid, group_dates)
    low_adverse_percentile = build_same_date_percentile_targets(
        -adverse,
        valid,
        group_dates,
    )
    composite = (
        0.5 * mfe_percentile.astype(np.float64)
        + 0.5 * low_adverse_percentile.astype(np.float64)
    ).astype(np.float32)
    composite[~valid] = np.nan
    if bool(np.any(valid & ~np.isfinite(composite))):
        raise ValueError("MR-13N equal-rank target產生non-finite value")
    return composite, mfe_percentile, low_adverse_percentile


def resolve_daily_training_universe_start(
    benchmark_index: pd.DatetimeIndex,
    *,
    feature_window_bars: int,
) -> pd.Timestamp:
    """Return the earliest benchmark date with a complete causal feature window."""

    bars = int(feature_window_bars)
    if bars < 1:
        raise ValueError("feature_window_bars必須>=1")
    normalized = pd.DatetimeIndex(benchmark_index).normalize()
    first_pos = bars - 1
    if len(normalized) <= first_pos:
        raise ValueError("daily ranker benchmark不足以形成完整feature window")
    return pd.Timestamp(normalized[first_pos]).normalize()


def load_daily_universal_ranker_data(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    preload_feature_bank: bool,
    allow_stale_source: bool,
    extend_score_eligibility_to_source_tail: bool = False,
    project_root: str | Path = PROJECT_ROOT,
    progress_callback=None,
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
    target_id = str(profile.continuous_target_id or "").strip()
    if target_id not in {
        DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
        DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
        DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
        DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID,
        DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID,
    }:
        raise ValueError(f"daily universal ranker target identity不一致: {target_id!r}")
    model_spec = get_model_spec(model_architecture)
    if bool(model_spec.requires_market_set) or bool(model_spec.derived_context_features):
        raise ValueError("daily universal ranker不支援market-set／derived-context architecture")
    use_risk_context = str(model_spec.architecture) == "inception_time_risk_context_v1"
    if bool(model_spec.use_dataset_context) != bool(use_risk_context):
        raise ValueError("daily universal ranker architecture/context contract不一致")
    if use_risk_context and target_id != DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID:
        raise ValueError("risk-context architecture只允許MR-13I/J同源risk-normalized target")

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
    first_pos = int(DEFAULT_LABEL_POLICY.feature_window_bars) - 1
    # Daily Universal training universe must not inherit the optimizer's historical
    # selection_start cutoff.  The earliest legal stock-day is data-driven: both the
    # stock and benchmark must already have a complete feature window.  Per-stock
    # first_pos filtering below enforces the stock side; this benchmark date enforces
    # the shared market sequence side.
    sample_start = resolve_daily_training_universe_start(
        benchmark_index,
        feature_window_bars=int(DEFAULT_LABEL_POLICY.feature_window_bars),
    )
    if bool(extend_score_eligibility_to_source_tail):
        source_tail = pd.Timestamp(benchmark_index.max()).normalize()
        if source_tail > sample_end:
            sample_end = source_tail

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    risk_schedule = None
    risk_param_policy = None
    if target_id == DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID:
        from config.strategy_compare import get_strategy_comparison_settings

        risk_param_policy = str(
            get_strategy_comparison_settings("selection_pit").param_policy
        ).strip()
        risk_schedule = load_min_roos_risk_schedule(root, param_policy=risk_param_policy)
    tickers: list[str] = []
    frames: list[pd.DataFrame] = []

    # Keep target-valid rows first in the exact historical ticker/date order.  This preserves
    # the scientific training universe while allowing additional inference-only rows to be
    # appended without making score availability depend on future target completion.
    valid_ticker_id_chunks: list[np.ndarray] = []
    valid_source_pos_chunks: list[np.ndarray] = []
    valid_benchmark_pos_chunks: list[np.ndarray] = []
    valid_date_chunks: list[np.ndarray] = []
    valid_label_end_chunks: list[np.ndarray] = []
    valid_target_chunks: list[np.ndarray] = []
    valid_context_chunks: list[np.ndarray] = []
    valid_favorable_chunks: list[np.ndarray] = []
    valid_adverse_chunks: list[np.ndarray] = []
    valid_opportunity_bar_chunks: list[np.ndarray] = []
    valid_first_breach_bar_chunks: list[np.ndarray] = []
    valid_minimum_low_return_chunks: list[np.ndarray] = []

    inference_ticker_id_chunks: list[np.ndarray] = []
    inference_source_pos_chunks: list[np.ndarray] = []
    inference_benchmark_pos_chunks: list[np.ndarray] = []
    inference_date_chunks: list[np.ndarray] = []
    inference_context_chunks: list[np.ndarray] = []
    pending_inference_only_tickers: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]] = []
    skipped_tickers = 0
    processed_tickers = 0
    target_valid_so_far = 0
    stock_input_count = sum(1 for raw_ticker, _path in csv_inputs if str(raw_ticker) != benchmark_ticker)
    if progress_callback is not None:
        progress_callback(0, stock_input_count, 0, 0)

    stock_inputs = [(str(raw_ticker), path) for raw_ticker, path in csv_inputs if str(raw_ticker) != benchmark_ticker]
    for ticker, path in stock_inputs:
        processed_tickers += 1
        try:
            frame = load_dataset_frame(path, ticker, min_rows=min_rows)
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
            skipped_tickers += 1
            if progress_callback is not None:
                progress_callback(processed_tickers, stock_input_count, target_valid_so_far, skipped_tickers)
            continue

        if len(frame) <= first_pos:
            skipped_tickers += 1
            if progress_callback is not None:
                progress_callback(processed_tickers, stock_input_count, target_valid_so_far, skipped_tickers)
            continue
        candidate_positions = np.arange(first_pos, len(frame), dtype=np.int64)
        frame_dates = pd.DatetimeIndex(frame.index).normalize()
        candidate_dates = frame_dates.take(candidate_positions)
        in_period = (candidate_dates >= sample_start) & (candidate_dates <= sample_end)
        benchmark_pos = benchmark_index.get_indexer(candidate_dates)
        feature_eligible = np.asarray(in_period, dtype=bool) & (benchmark_pos >= first_pos)
        local_positions = candidate_positions[feature_eligible]
        local_benchmark_positions = benchmark_pos[feature_eligible]
        if len(local_positions) == 0:
            skipped_tickers += 1
            if progress_callback is not None:
                progress_callback(processed_tickers, stock_input_count, target_valid_so_far, skipped_tickers)
            continue

        # Future target completion is a training/evaluation property, not an inference
        # eligibility rule. MR-13I/J keep the MR-13E daily-universal stock-day universe;
        # Min ROOS contributes only historical-effective risk calibration.
        context_width = len(RISK_GEOMETRY_CONTEXT_FEATURES) if use_risk_context else 0
        local_context = np.empty((len(local_positions), context_width), dtype=np.float32)
        local_favorable = np.full(len(local_positions), np.nan, dtype=np.float32)
        local_adverse = np.full(len(local_positions), np.nan, dtype=np.float32)
        local_opportunity_bar = np.full(len(local_positions), -1, dtype=np.int16)
        local_first_breach_bar = np.full(len(local_positions), -1, dtype=np.int16)
        local_minimum_low_return = np.full(len(local_positions), np.nan, dtype=np.float32)
        if target_id in {
            DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
            DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
            DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
            DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
            DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID,
        }:
            local_targets = np.full(len(local_positions), np.nan, dtype=np.float32)
            local_target_valid = np.zeros(len(local_positions), dtype=bool)
            target_complete = local_positions + int(spec.horizon_bars) < len(frame)
            if bool(target_complete.any()):
                component_target_id = (
                    DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID
                    if target_id == DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID
                    else target_id
                )
                completed_batch = compute_daily_opportunity_target_batch(
                    frame,
                    local_positions[target_complete],
                    spec=spec,
                    target_id=component_target_id,
                )
                completed_indexes = np.flatnonzero(target_complete)
                local_targets[completed_indexes] = completed_batch.target_raw_r
                local_target_valid[completed_indexes] = completed_batch.valid_mask
                local_favorable[completed_indexes] = completed_batch.favorable_return
                local_adverse[completed_indexes] = completed_batch.adverse_return_to_peak
                local_opportunity_bar[completed_indexes] = completed_batch.opportunity_bar
                local_first_breach_bar[completed_indexes] = completed_batch.first_risk_breach_bar
                local_minimum_low_return[completed_indexes] = completed_batch.minimum_low_return
        else:
            local_targets, local_target_valid, risk_context, geometry_valid = (
                compute_targets_and_context_for_positions(
                    frame,
                    local_positions,
                    ticker=ticker,
                    schedule=risk_schedule,
                    horizon_bars=int(spec.horizon_bars),
                )
            )
            if use_risk_context:
                # MR-13J requires legal decision-time geometry to emit a score.  Filtering
                # applies only where no historical risk calibration exists; within covered
                # dates the stock-day universe remains the same as MR-13E/I.
                keep = np.asarray(geometry_valid, dtype=bool)
                local_positions = local_positions[keep]
                local_benchmark_positions = local_benchmark_positions[keep]
                local_targets = local_targets[keep]
                local_target_valid = local_target_valid[keep]
                local_context = np.asarray(risk_context[keep], dtype=np.float32)
                local_favorable = local_favorable[keep]
                local_adverse = local_adverse[keep]
                local_opportunity_bar = local_opportunity_bar[keep]
                local_first_breach_bar = local_first_breach_bar[keep]
                local_minimum_low_return = local_minimum_low_return[keep]
                if len(local_positions) == 0:
                    skipped_tickers += 1
                    if progress_callback is not None:
                        progress_callback(processed_tickers, stock_input_count, target_valid_so_far, skipped_tickers)
                    continue

        target_valid_so_far += int(np.count_nonzero(local_target_valid))
        if progress_callback is not None:
            progress_callback(processed_tickers, stock_input_count, target_valid_so_far, skipped_tickers)

        valid_positions = local_positions[local_target_valid]
        valid_benchmark_positions = local_benchmark_positions[local_target_valid]
        valid_targets = local_targets[local_target_valid]
        valid_context = local_context[local_target_valid]
        valid_favorable = local_favorable[local_target_valid]
        valid_adverse = local_adverse[local_target_valid]
        valid_opportunity_bar = local_opportunity_bar[local_target_valid]
        valid_first_breach_bar = local_first_breach_bar[local_target_valid]
        valid_minimum_low_return = local_minimum_low_return[local_target_valid]
        invalid_positions = local_positions[~local_target_valid]
        invalid_benchmark_positions = local_benchmark_positions[~local_target_valid]
        invalid_context = local_context[~local_target_valid]

        if len(valid_positions):
            ticker_id = len(tickers)
            tickers.append(ticker)
            frames.append(frame)
            count = len(valid_positions)
            valid_ticker_id_chunks.append(np.full(count, ticker_id, dtype=np.int32))
            valid_source_pos_chunks.append(np.asarray(valid_positions, dtype=np.int32))
            valid_benchmark_pos_chunks.append(
                np.asarray(valid_benchmark_positions, dtype=np.int32)
            )
            valid_date_chunks.append(
                frame_dates.take(valid_positions).to_numpy(dtype="datetime64[D]")
            )
            valid_label_end_chunks.append(
                frame_dates.take(valid_positions + int(spec.horizon_bars)).to_numpy(
                    dtype="datetime64[D]"
                )
            )
            valid_target_chunks.append(np.asarray(valid_targets, dtype=np.float32))
            valid_context_chunks.append(np.asarray(valid_context, dtype=np.float32))
            valid_favorable_chunks.append(np.asarray(valid_favorable, dtype=np.float32))
            valid_adverse_chunks.append(np.asarray(valid_adverse, dtype=np.float32))
            valid_opportunity_bar_chunks.append(np.asarray(valid_opportunity_bar, dtype=np.int16))
            valid_first_breach_bar_chunks.append(np.asarray(valid_first_breach_bar, dtype=np.int16))
            valid_minimum_low_return_chunks.append(np.asarray(valid_minimum_low_return, dtype=np.float32))
            if len(invalid_positions):
                inference_ticker_id_chunks.append(
                    np.full(len(invalid_positions), ticker_id, dtype=np.int32)
                )
                inference_source_pos_chunks.append(
                    np.asarray(invalid_positions, dtype=np.int32)
                )
                inference_benchmark_pos_chunks.append(
                    np.asarray(invalid_benchmark_positions, dtype=np.int32)
                )
                inference_date_chunks.append(
                    frame_dates.take(invalid_positions).to_numpy(dtype="datetime64[D]")
                )
                inference_context_chunks.append(np.asarray(invalid_context, dtype=np.float32))
        else:
            # A ticker with no target-valid row can still be score-eligible.  Append these
            # tickers after the historical training tickers so existing target-valid ticker
            # identities and group ordering remain stable.
            pending_inference_only_tickers.append(
                (
                    ticker,
                    frame,
                    np.asarray(invalid_positions, dtype=np.int32),
                    np.asarray(invalid_benchmark_positions, dtype=np.int32),
                    frame_dates,
                    np.asarray(invalid_context, dtype=np.float32),
                )
            )

    if not valid_target_chunks:
        raise ValueError("daily universal ranker沒有任何有效target stock-day sample")

    for ticker, frame, positions, benchmark_positions_local, frame_dates, pending_context in pending_inference_only_tickers:
        if len(positions) == 0:
            continue
        ticker_id = len(tickers)
        tickers.append(ticker)
        frames.append(frame)
        inference_ticker_id_chunks.append(
            np.full(len(positions), ticker_id, dtype=np.int32)
        )
        inference_source_pos_chunks.append(np.asarray(positions, dtype=np.int32))
        inference_benchmark_pos_chunks.append(
            np.asarray(benchmark_positions_local, dtype=np.int32)
        )
        inference_date_chunks.append(
            frame_dates.take(positions).to_numpy(dtype="datetime64[D]")
        )
        inference_context_chunks.append(np.asarray(pending_context, dtype=np.float32))

    valid_ticker_ids = np.concatenate(valid_ticker_id_chunks)
    valid_source_positions = np.concatenate(valid_source_pos_chunks)
    valid_benchmark_positions = np.concatenate(valid_benchmark_pos_chunks)
    valid_dates = np.concatenate(valid_date_chunks)
    valid_label_end_dates = np.concatenate(valid_label_end_chunks)
    valid_targets = np.concatenate(valid_target_chunks)
    valid_favorable = np.concatenate(valid_favorable_chunks)
    valid_adverse = np.concatenate(valid_adverse_chunks)
    valid_opportunity_bar = np.concatenate(valid_opportunity_bar_chunks)
    valid_first_breach_bar = np.concatenate(valid_first_breach_bar_chunks)
    valid_minimum_low_return = np.concatenate(valid_minimum_low_return_chunks)
    valid_mfe_percentile = None
    valid_low_adverse_percentile = None
    if target_id == DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID:
        component_valid = np.ones(len(valid_targets), dtype=bool)
        risk_budget_return = float(spec.risk_budget_return)
        valid_targets, valid_mfe_percentile, valid_low_adverse_percentile = (
            build_equal_rank_mfe_low_adverse_target(
                np.asarray(valid_favorable, dtype=np.float64) / risk_budget_return,
                np.asarray(valid_adverse, dtype=np.float64) / risk_budget_return,
                component_valid,
                valid_dates,
            )
        )
    context_width = len(RISK_GEOMETRY_CONTEXT_FEATURES) if use_risk_context else 0
    valid_context = np.concatenate(valid_context_chunks, axis=0)

    if inference_source_pos_chunks:
        inference_ticker_ids = np.concatenate(inference_ticker_id_chunks)
        inference_source_positions = np.concatenate(inference_source_pos_chunks)
        inference_benchmark_positions = np.concatenate(inference_benchmark_pos_chunks)
        inference_dates = np.concatenate(inference_date_chunks)
        inference_context = np.concatenate(inference_context_chunks, axis=0)
    else:
        inference_ticker_ids = np.empty(0, dtype=np.int32)
        inference_source_positions = np.empty(0, dtype=np.int32)
        inference_benchmark_positions = np.empty(0, dtype=np.int32)
        inference_dates = np.empty(0, dtype="datetime64[D]")
        inference_context = np.empty((0, context_width), dtype=np.float32)

    ticker_ids = np.concatenate([valid_ticker_ids, inference_ticker_ids])
    source_positions = np.concatenate([valid_source_positions, inference_source_positions])
    benchmark_positions = np.concatenate(
        [valid_benchmark_positions, inference_benchmark_positions]
    )
    dates = np.concatenate([valid_dates, inference_dates])
    group_context = np.concatenate([valid_context, inference_context], axis=0)
    raw_target = np.concatenate(
        [valid_targets, np.full(len(inference_dates), np.nan, dtype=np.float32)]
    )
    target_valid = np.concatenate(
        [
            np.ones(len(valid_targets), dtype=bool),
            np.zeros(len(inference_dates), dtype=bool),
        ]
    )
    label_end_dates = np.concatenate(
        [
            valid_label_end_dates.astype("datetime64[ns]"),
            np.full(len(inference_dates), np.datetime64("NaT"), dtype="datetime64[ns]"),
        ]
    )
    favorable_return = np.concatenate([valid_favorable, np.full(len(inference_dates), np.nan, dtype=np.float32)])
    adverse_return = np.concatenate([valid_adverse, np.full(len(inference_dates), np.nan, dtype=np.float32)])
    risk_budget_return = float(spec.risk_budget_return)
    favorable_r = favorable_return / risk_budget_return
    adverse_r = adverse_return / risk_budget_return
    opportunity_bar = np.concatenate([valid_opportunity_bar, np.full(len(inference_dates), -1, dtype=np.int16)])
    first_breach_bar = np.concatenate([valid_first_breach_bar, np.full(len(inference_dates), -1, dtype=np.int16)])
    minimum_low_return = np.concatenate([valid_minimum_low_return, np.full(len(inference_dates), np.nan, dtype=np.float32)])
    group_count = int(len(raw_target))
    target_valid_count = int(target_valid.sum())
    inference_only_count = int(group_count - target_valid_count)
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
            "target_favorable_return": favorable_return,
            "target_adverse_return_to_peak": adverse_return,
            "target_favorable_r": favorable_r,
            "target_adverse_r": adverse_r,
            "target_opportunity_bar": opportunity_bar,
            "target_first_risk_breach_bar": first_breach_bar,
            "target_minimum_low_return": minimum_low_return,
        }
    )
    if target_id == DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID:
        group_table["target_mfe_daily_percentile"] = np.concatenate(
            [
                np.asarray(valid_mfe_percentile, dtype=np.float32),
                np.full(len(inference_dates), np.nan, dtype=np.float32),
            ]
        )
        group_table["target_low_adverse_daily_percentile"] = np.concatenate(
            [
                np.asarray(valid_low_adverse_percentile, dtype=np.float32),
                np.full(len(inference_dates), np.nan, dtype=np.float32),
            ]
        )
        group_table["target_equal_rank_composite"] = raw_target
    feature_bank = LazyDailyFeatureBank(
        frames=tuple(frames),
        benchmark=benchmark,
        ticker_ids=ticker_ids,
        source_positions=source_positions,
        benchmark_positions=benchmark_positions,
        policy=DEFAULT_LABEL_POLICY,
    )
    validate_model_sequence_length(model_spec, int(feature_bank.shape[1]))
    if target_id == DAILY_OPPORTUNITY_NO_TIME_TARGET_ID:
        target_contract = build_daily_opportunity_no_time_contract(DEFAULT_LABEL_POLICY)
    elif target_id == DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID:
        target_contract = build_daily_full_horizon_opportunity_contract(DEFAULT_LABEL_POLICY)
    elif target_id == DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID:
        target_contract = build_daily_full_horizon_pure_mfe_contract(DEFAULT_LABEL_POLICY)
    elif target_id == DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID:
        target_contract = build_daily_full_horizon_low_adverse_contract(DEFAULT_LABEL_POLICY)
    elif target_id == DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID:
        target_contract = build_daily_full_horizon_equal_rank_mfe_low_adverse_contract(
            DEFAULT_LABEL_POLICY
        )
    else:
        target_contract = build_risk_target_contract(
            horizon_bars=int(spec.horizon_bars),
            param_policy=str(risk_param_policy),
        )
    target_manifest = {
        "target_id": target_id,
        "target_contract": target_contract,
        "sample_scope": "daily_eligible_stock_days",
        "sample_count": target_valid_count,
        "target_valid_sample_count": target_valid_count,
        "score_eligible_sample_count": group_count,
        "inference_only_sample_count": inference_only_count,
        "ticker_count": int(len(tickers)),
        "date_range": {
            "start": str(pd.Timestamp(group_table["date"].min()).date()),
            "end": str(pd.Timestamp(group_table["date"].max()).date()),
        },
        "feature_storage": "lazy_from_canonical_ohlcv_no_expanded_daily_feature_bank",
        "context_features": list(RISK_GEOMETRY_CONTEXT_FEATURES) if use_risk_context else [],
        "risk_param_coverage_start": (
            None if risk_schedule is None else min(item.start_date for item in risk_schedule).date().isoformat()
        ),
        "training_universe_start_date": str(pd.Timestamp(sample_start).date()),
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
        "score_eligible_sample_count": group_count,
        "target_valid_sample_count": target_valid_count,
        "inference_only_sample_count": inference_only_count,
        "ticker_count": int(len(tickers)),
        "skipped_ticker_count": int(skipped_tickers),
        "feature_storage": "lazy",
        "context_features": list(RISK_GEOMETRY_CONTEXT_FEATURES) if use_risk_context else [],
        "risk_param_coverage_start": (
            None if risk_schedule is None else min(item.start_date for item in risk_schedule).date().isoformat()
        ),
        "score_eligibility_extended_to_source_tail": bool(extend_score_eligibility_to_source_tail),
        "training_universe_start_date": str(pd.Timestamp(sample_start).date()),
        "score_eligibility_end_date": str(pd.Timestamp(sample_end).date()),
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
        group_context=np.asarray(group_context, dtype=np.float32),
        raw_target=raw_target,
        target_valid=target_valid,
        target_manifest=target_manifest,
        outer_policy=outer_policy,
        profile=profile,
        model_spec=model_spec,
    )


def select_breakout_candidate_group_ids(
    bundle: ContinuousRankerDataBundle,
    group_ids: np.ndarray,
    *,
    allow_stale_source: bool,
) -> np.ndarray:
    """Select official breakout ticker/date membership for post-model diagnostics."""

    _summary, _features, _context, _labels, events = load_validated_dataset_bundle(
        str(bundle.summary["filter_id"]),
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(allow_stale_source),
    )
    event_frame = events.loc[:, ["ticker", "date"]].copy()
    event_frame["ticker"] = event_frame["ticker"].astype(str)
    event_frame["date"] = pd.to_datetime(
        event_frame["date"], errors="raise"
    ).dt.normalize()
    event_keys = set(zip(event_frame["ticker"].tolist(), event_frame["date"].tolist()))
    ids = np.asarray(group_ids, dtype=np.int64)
    groups = bundle.group_table.iloc[ids]
    mask = np.fromiter(
        (
            (str(ticker), pd.Timestamp(date).normalize()) in event_keys
            for ticker, date in zip(groups["ticker"], groups["date"])
        ),
        dtype=bool,
        count=len(groups),
    )
    return ids[mask]


def build_daily_ranker_split(bundle: ContinuousRankerDataBundle, *, inner_validation_months: int) -> DailyRankerSplit:
    """Mirror the canonical no-lookahead split directly at stock/day group level."""

    dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    label_end = pd.to_datetime(
        bundle.group_table["label_eval_end_date"], errors="raise"
    ).dt.normalize()
    policy = bundle.outer_policy
    # Daily Universal Pre-Test must use every legally feature-complete historical
    # stock-day available before the frozen OOS cutoff.  The old optimizer
    # selection_start is an optimizer research boundary, not a DL data boundary.
    selection_start = pd.Timestamp(
        str(bundle.summary.get("training_universe_start_date") or policy["selection_start_date"])
    ).normalize()
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
    "DailyOpportunityTargetBatch",
    "DailyRankerSplit",
    "LazyDailyFeatureBank",
    "build_daily_ranker_split",
    "build_equal_rank_mfe_low_adverse_target",
    "compute_daily_opportunity_target_batch",
    "load_daily_universal_ranker_data",
    "resolve_daily_training_universe_start",
    "select_breakout_candidate_group_ids",
]
