"""Daily-universal *counterfactual* Min 40-session net-R target.

For each eligible stock/date, pretend a fresh Min setup originated at that
close.  Its normal limit is the canonical rounded close + ATR*tolerance;
thereafter the official A2 single-stock execution path owns order continuation,
fees, realized risk, stops, exits and the forced 40th exchange-session close.
The hypothetical *originating signal* is NOT a real Min breakout signal and
must never be used to create executable candidates or production orders.

No holding/cash/K/R0/selector state enters a training label. Unfilled,
unexecutable, immature and uncovered dates have a NULL supervised R; they are
not transformed into a fake loss or zero return. The score-eligible universe
remains daily-universal even when a future training label is unavailable.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config.execution_policy import DEFAULT_FIXED_RISK
from core.portfolio_param_runtime import load_portfolio_param_source_from_json
from core.price_utils import adjust_long_buy_limit_array
from core.strategy_params import V16StrategyParams
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.strategy_compare_sources import PARAM_POLICY_SPECS
from filters.breakout_quality.trade_path_label import (
    build_signal_cache,
    simulate_realized_trade_path_label,
)
from filters.breakout_quality.workflow_io import load_dataset_frame

TARGET_ID = "daily_universal_counterfactual_mintrade40_netr_v1"
LABEL_ID = "LABEL-daily_universal_counterfactual_mintrade40_netr_v1"
HORIZON_SESSIONS = 40
SOURCE_IDS = ("selection_min_roos", "min_best_broker_cash_rolling_experiment")
P16_HANDOFF_DATE = pd.Timestamp("2021-01-01")
SCHEMA = 1


@dataclass(frozen=True)
class ExecutableMinPeriod:
    start: pd.Timestamp
    end: pd.Timestamp | None
    params: V16StrategyParams
    source_id: str
    params_signature: str


@dataclass(frozen=True)
class CounterfactualTrade40Batch:
    target_raw_r: np.ndarray
    valid_mask: np.ndarray
    maturity_dates: np.ndarray
    reasons: tuple[str, ...]


def resolve_historical_min_teacher_paths(
    root: str | Path, *, settings=None,
) -> dict[str, Path]:
    """Resolve both teachers from the same Strategy Compare source SSOT as the label."""
    if settings is None:
        from core.strategy_compare_policy import get_strategy_comparison_settings
        settings = get_strategy_comparison_settings("selection_pit")
    policy = str(settings.param_policy).strip()
    specification = PARAM_POLICY_SPECS.get(policy)
    if policy == "auto" or not isinstance(specification, dict):
        raise ValueError("Counterfactual Trade40 requires resolved canonical Min param policy")
    filename = str(specification["filename"])
    paths: dict[str, Path] = {}
    for source_id in SOURCE_IDS:
        source = settings.parameter_sources.get(source_id)
        if source is None or not str(source.path_template or "").strip():
            raise ValueError(f"Counterfactual Trade40 missing canonical source {source_id}")
        paths[source_id] = Path(root).resolve() / str(source.path_template).format(
            param_filename=filename,
        )
    return paths


def load_historical_min_execution_schedule(
    root: str | Path, *, settings=None,
) -> tuple[tuple[ExecutableMinPeriod, ...], dict[str, Any]]:
    """Resolve PIT-effective historical Min teachers without fabricating P16 history.

    Earlier supervised rows can use independently archived P2 Min parameters.
    Starting in 2021 the teacher MUST switch to the canonical P16 Min source
    used by the Research portfolio; missing years stay unlabeled, never backfilled.
    Both canonical input files and every effective year enter target lineage.
    """
    if settings is None:
        from core.strategy_compare_policy import get_strategy_comparison_settings
        settings = get_strategy_comparison_settings("selection_pit")
    root = Path(root).resolve()
    paths = resolve_historical_min_teacher_paths(root, settings=settings)
    periods: list[ExecutableMinPeriod] = []
    provenance: dict[str, Any] = {"policy": str(settings.param_policy).strip(), "sources": []}
    for source_id, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Counterfactual Trade40 historical Min params missing: {source_id}: {path}")
        loaded = load_portfolio_param_source_from_json(path, fixed_risk=DEFAULT_FIXED_RISK)
        records = list(loaded.get("schedule") or ())
        if not records:
            raise ValueError(f"Counterfactual Trade40 missing historical effective Min schedule: {source_id}")
        for record in records:
            members = list(record.get("members") or ())
            if members:
                if len(members) != 1:
                    raise ValueError("Counterfactual Trade40 requires one canonical Min finalist, not consensus")
                params = members[0]["params_obj"]
                signature = str(members[0]["params_signature"])
            else:
                params = record["params_obj"]
                signature = str(record["params_signature"])
            if params.use_breakout_quality_filter or params.use_breakout_quality_ranking:
                raise ValueError("Counterfactual Trade40 teacher must be DL-off historical Min")
            start_raw = record.get("effective_date_text") or record.get("effective_date")
            end_raw = record.get("effective_end_date_text") or record.get("effective_end_date")
            if start_raw is None:
                raise ValueError("Counterfactual Trade40 historical Min effective date missing")
            start = pd.Timestamp(start_raw).normalize()
            end = None if end_raw in (None, "") else pd.Timestamp(end_raw).normalize()
            if end is not None and end < start:
                raise ValueError("Counterfactual Trade40 Min param period ends before start")
            # The P2 teacher is a legally historical training approximation,
            # NOT the P16 strategy that must be used for 2021+ OOS comparisons.
            if source_id == SOURCE_IDS[0]:
                if start >= P16_HANDOFF_DATE:
                    continue
                end = min(end, P16_HANDOFF_DATE - pd.Timedelta(days=1)) if end is not None else P16_HANDOFF_DATE - pd.Timedelta(days=1)
            else:
                if end is not None and end < P16_HANDOFF_DATE:
                    continue
                start = max(start, P16_HANDOFF_DATE)
            periods.append(ExecutableMinPeriod(start, end, params, source_id, signature))
        provenance["sources"].append({
            "source_id": source_id,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": compute_file_sha256(path),
        })
    periods.sort(key=lambda period: (period.start, period.source_id))
    for left, right in zip(periods, periods[1:]):
        # A newer effective year can supersede a previous open-ended archive.
        # Same-day conflicting truths are always illegal, not silently preferred.
        if left.start == right.start and left.params_signature != right.params_signature:
            raise ValueError("Counterfactual Trade40 Min sources have conflicting effective-date params")
        if (left.source_id == right.source_id and left.end is not None
                and right.start <= left.end):
            raise ValueError("Counterfactual Trade40 overlapping Min periods in one source")
    if not periods or not any(p.source_id == SOURCE_IDS[1] for p in periods):
        raise ValueError("Counterfactual Trade40 has no effective canonical P16 Min parameters for 2021+")
    provenance["periods"] = [
        {"start": str(period.start.date()),
         "end": None if period.end is None else str(period.end.date()),
         "source_id": period.source_id, "params_signature": period.params_signature}
        for period in periods
    ]
    return tuple(periods), provenance


def resolve_historical_min_period(
    schedule: tuple[ExecutableMinPeriod, ...], date: Any,
) -> ExecutableMinPeriod | None:
    day = pd.Timestamp(date).normalize()
    starts = [period.start for period in schedule]
    index = bisect_right(starts, day) - 1
    if index < 0:
        return None
    # A previous overlapping record may still be active if a more recent
    # time-window ended. Never borrow parameters from a *future* effective date.
    for period in reversed(schedule[:index + 1]):
        if period.start <= day and (period.end is None or day <= period.end):
            return period
    return None


def collect_canonical_exchange_sessions(
    csv_inputs: Iterable[tuple[str, Path]], *, benchmark_ticker: str,
) -> pd.DatetimeIndex:
    """Union the same *sanitized* stock quote sessions used by portfolio replay.

    A benchmark ETF is not an exchange calendar: it can lack a valid day. This
    extra prepass is confined to this new counterfactual training capability.
    """
    calendar: set[pd.Timestamp] = set()
    found_benchmark = False
    for ticker, path in csv_inputs:
        ticker = str(ticker)
        try:
            frame = load_dataset_frame(Path(path), ticker, min_rows=1)
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
            if ticker == benchmark_ticker:
                raise ValueError("Counterfactual Trade40 benchmark sanitized source invalid")
            continue
        if ticker == benchmark_ticker:
            found_benchmark = True
        calendar.update(pd.DatetimeIndex(frame.index).normalize().tolist())
    if not found_benchmark or len(calendar) <= HORIZON_SESSIONS:
        raise ValueError("Counterfactual Trade40 missing usable consolidated exchange calendar")
    return pd.DatetimeIndex(sorted(calendar)).normalize()


def counterfactual_trade40_contract(
    *, provenance: dict[str, Any], exchange_calendar: pd.DatetimeIndex,
) -> dict[str, Any]:
    days = pd.DatetimeIndex(exchange_calendar).normalize()
    if days.empty or days.has_duplicates or not days.is_monotonic_increasing:
        raise ValueError("Counterfactual Trade40 exchange calendar invalid")
    digest = sha256("\n".join(days.strftime("%Y-%m-%d")).encode("ascii")).hexdigest()
    return {
        "schema_version": SCHEMA, "target_id": TARGET_ID, "label_id": LABEL_ID,
        "sample_scope": "daily_eligible_stock_days",
        "target_semantics": "hypothetical_fresh_min_setup_each_stock_day",
        "horizon_sessions": HORIZON_SESSIONS,
        "calendar_policy": "union_of_canonical_sanitized_portfolio_stock_sessions",
        "exchange_calendar_sha256": digest,
        "teacher": provenance,
        "historical_teacher_handoff": "P2 before 2021-01-01; canonical P16 Min from 2021-01-01",
        "entry": "decision_date_close_plus_ATR_times_canonical_Min_tolerance_rounded_by_price_utils",
        "execution": "canonical_A2_trade_path_ledger_net_with_40th_exchange_session_closeout",
        "censored_labels": "no_fill_or_unexecutable_or_unmatured_is_null_not_zero",
        "inference_scope": "daily_eligible_all_stock_days_even_if_target_censored",
        "portfolio_state_inputs": False,
        "oos_fitted_parameters": False,
    }


def compute_counterfactual_trade40_batch(
    frame: pd.DataFrame, positions: np.ndarray, *, ticker: str,
    schedule: tuple[ExecutableMinPeriod, ...], exchange_calendar: pd.DatetimeIndex,
) -> CounterfactualTrade40Batch:
    """Compute continuous net R with *real* execution primitives, never proxy PnL."""
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("counterfactual Trade40 positions must be 1D")
    calendar = pd.DatetimeIndex(exchange_calendar).normalize()
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("counterfactual Trade40 requires strictly sorted exchange sessions")
    dates = pd.DatetimeIndex(frame.index).normalize()
    calendar_index = calendar.get_indexer(dates.take(positions))
    result = np.full(len(positions), np.nan, dtype=np.float32)
    valid = np.zeros(len(positions), dtype=bool)
    maturity_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[D]")
    reasons: list[str] = []
    signal_caches: dict[str, Any] = {}
    for row, (position, calendar_pos) in enumerate(zip(positions, calendar_index)):
        if calendar_pos < 0 or calendar_pos + HORIZON_SESSIONS >= len(calendar):
            reasons.append("unmatured_or_missing_exchange_session")
            continue
        expiry = calendar[calendar_pos + HORIZON_SESSIONS]
        maturity_dates[row] = expiry.to_datetime64().astype("datetime64[D]")
        period = resolve_historical_min_period(schedule, dates[position])
        if period is None:
            reasons.append("no_historical_effective_Min_param")
            continue
        params = period.params
        cache = signal_caches.get(period.params_signature)
        if cache is None:
            cache = build_signal_cache(frame, params, ticker=ticker)
            signal_caches[period.params_signature] = cache
        atr = float(cache[0][position])
        close = float(frame["Close"].iloc[position])
        if not math.isfinite(atr) or atr <= 0.0 or not math.isfinite(close) or close <= 0.0:
            reasons.append("no_decision_time_ATR_or_close")
            continue
        # Canonical generate_signals defines this same buy-limit formula on
        # *real* setups. We override only the hypothetical initial setup.
        limit = float(adjust_long_buy_limit_array(
            np.asarray([close + atr * float(params.atr_buy_tol)], dtype=np.float64),
            ticker=ticker,
        )[0])
        observation = simulate_realized_trade_path_label(
            frame, ticker=ticker, signal_pos=int(position), params=params,
            teacher_effective_date=str(period.start.date()),
            precomputed_signals=cache, horizon_end_date=expiry,
            counterfactual_entry_limit=limit,
        )
        if observation.realized_net_r is None:
            reasons.append(str(observation.reason))
            continue
        value = float(observation.realized_net_r)
        if not math.isfinite(value):
            raise ValueError("canonical trade path produced nonfinite realized R")
        result[row] = value
        valid[row] = True
        reasons.append("filled_executed_net_R")
    return CounterfactualTrade40Batch(result, valid, maturity_dates, tuple(reasons))


__all__ = [
    "TARGET_ID", "LABEL_ID", "HORIZON_SESSIONS", "SOURCE_IDS", "P16_HANDOFF_DATE",
    "ExecutableMinPeriod", "CounterfactualTrade40Batch",
    "resolve_historical_min_teacher_paths", "load_historical_min_execution_schedule",
    "resolve_historical_min_period",
    "collect_canonical_exchange_sessions", "counterfactual_trade40_contract",
    "compute_counterfactual_trade40_batch",
]
