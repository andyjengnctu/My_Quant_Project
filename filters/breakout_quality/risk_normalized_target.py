"""Universal risk-normalized 40D target and PIT risk/economic geometry.

MR-13I/J intentionally remain daily-universal.  The only strategy-derived input is the
historical-effective Min ROOS *risk* calibration (ATR length + initial-stop multiplier).
No high_len, breakout qualification, entry tolerance, trailing exit, portfolio cash, K,
R0, holdings, or selector state enters this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config.execution_policy import DEFAULT_FIXED_RISK, EXECUTION_POLICY_PARAM_SPECS
from core.exact_accounting import (
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    calc_planned_initial_risk_from_prices_milli,
    price_to_milli,
)
from core.order_lot_policy import apply_board_lot_preferred_qty
from core.portfolio_param_runtime import load_portfolio_param_source_from_json
from core.price_utils import calc_initial_stop_from_reference, calc_position_size
from core.signal_utils import tv_atr
from core.strategy_params import V16StrategyParams
from filters.breakout_quality.strategy_compare_sources import PARAM_POLICY_SPECS

DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID = (
    "daily_risk_normalized_net_opportunity_r_v1"
)
RISK_NORMALIZED_TARGET_SCHEMA_VERSION = 1
RISK_NORMALIZATION_SOURCE_IDS = ("selection_min_roos", "min_roos")
RISK_NORMALIZATION_FIELDS = ("atr_len", "atr_times_init")
RISK_GEOMETRY_CONTEXT_FEATURES = (
    "risk_distance_pct",
    "risk_distance_atr",
    "capital_per_risk",
    "cost_per_risk",
    "risk_capacity",
)
RISK_GEOMETRY_CONTEXT_VERSION = 1
STANDARDIZED_EQUITY = float(EXECUTION_POLICY_PARAM_SPECS["initial_capital"]["default"])


@dataclass(frozen=True)
class RiskParamPeriod:
    start_date: pd.Timestamp
    end_date: pd.Timestamp | None
    atr_len: int
    atr_times_init: float
    source_id: str
    params_signature: str


@dataclass(frozen=True)
class RiskGeometryResult:
    valid: bool
    reason: str
    reference_price: float
    atr: float
    stop_price: float
    qty: int
    planned_initial_risk_milli: int
    context: tuple[float, ...]


def _normalized_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _period_end(record: dict[str, Any]) -> pd.Timestamp | None:
    raw = record.get("effective_end_date_text") or record.get("effective_end_date")
    if raw in (None, ""):
        return None
    return _normalized_date(raw)


def _period_start(record: dict[str, Any]) -> pd.Timestamp:
    raw = record.get("effective_date_text") or record.get("effective_date")
    if raw in (None, ""):
        raise ValueError("Min ROOS risk schedule缺少effective date")
    return _normalized_date(raw)


def _record_risk_tuple(record: dict[str, Any]) -> tuple[int, float, str]:
    members = list(record.get("members") or [])
    if members:
        tuples = {
            (
                int(member["params_obj"].atr_len),
                float(member["params_obj"].atr_times_init),
            )
            for member in members
        }
        if len(tuples) != 1:
            raise ValueError(
                "Min ROOS ensemble同一effective period的risk fields不一致；"
                "universal risk target不得自行平均／投票"
            )
        atr_len, atr_times_init = next(iter(tuples))
        signatures = sorted(str(member.get("params_signature") or "") for member in members)
        return atr_len, atr_times_init, "+".join(signatures)
    params = record.get("params_obj")
    if params is None:
        raise ValueError("Min ROOS risk schedule record缺少params_obj")
    return int(params.atr_len), float(params.atr_times_init), str(record.get("params_signature") or "")


def load_min_roos_risk_schedule(
    project_root: str | Path,
    *,
    param_policy: str | None = None,
) -> tuple[RiskParamPeriod, ...]:
    """Load the canonical historical-effective Min ROOS risk-only schedule.

    Paths/policy are resolved from Strategy Compare SSOT.  Only ``atr_len`` and
    ``atr_times_init`` are extracted; all breakout/exit/portfolio semantics are discarded.
    """

    from config.strategy_compare import get_strategy_comparison_settings

    root = Path(project_root)
    settings = get_strategy_comparison_settings("selection_pit")
    policy = str(param_policy or settings.param_policy).strip()
    if policy == "auto":
        raise ValueError("risk-normalized target需要已解析的Strategy Compare param policy")
    spec = PARAM_POLICY_SPECS.get(policy)
    if not isinstance(spec, dict) or not str(spec.get("filename") or "").strip():
        raise ValueError(f"不支援的risk param policy: {policy}")
    filename = str(spec["filename"])
    periods: list[RiskParamPeriod] = []
    source_paths: list[str] = []
    for source_id in RISK_NORMALIZATION_SOURCE_IDS:
        source = settings.parameter_sources.get(source_id)
        if source is None or not source.path_template:
            raise ValueError(f"Strategy Compare缺少Min ROOS param source: {source_id}")
        path = root / str(source.path_template).format(param_filename=filename)
        source_paths.append(str(path.relative_to(root) if path.is_relative_to(root) else path))
        if not path.is_file():
            raise FileNotFoundError(
                f"MR-13I/J缺少Min ROOS risk param工件: {source_id} -> {path}"
            )
        loaded = load_portfolio_param_source_from_json(path, fixed_risk=DEFAULT_FIXED_RISK)
        schedule = list(loaded.get("schedule") or [])
        if not schedule:
            raise ValueError(f"MR-13I/J需要historical-effective Min ROOS schedule: {path}")
        for record in schedule:
            atr_len, atr_times_init, signature = _record_risk_tuple(record)
            if atr_len < 1 or not math.isfinite(atr_times_init) or atr_times_init <= 0.0:
                raise ValueError("Min ROOS risk fields必須是合法正值")
            periods.append(
                RiskParamPeriod(
                    start_date=_period_start(record),
                    end_date=_period_end(record),
                    atr_len=int(atr_len),
                    atr_times_init=float(atr_times_init),
                    source_id=source_id,
                    params_signature=signature,
                )
            )

    periods.sort(key=lambda item: (item.start_date, item.source_id))
    # Overlap is allowed only when the risk tuple is identical.  For a date with both
    # Selection and Forward sources, the more recent effective start wins deterministically.
    for left, right in zip(periods, periods[1:]):
        left_end = left.end_date or pd.Timestamp.max.normalize()
        if right.start_date <= left_end and (
            left.atr_len != right.atr_len
            or not math.isclose(left.atr_times_init, right.atr_times_init, rel_tol=0.0, abs_tol=1e-12)
        ):
            # Different rolling periods inside the same source are expected to overlap only
            # at malformed boundaries; different sources may meet at Selection/Forward edge.
            if left.source_id == right.source_id or right.start_date < left.start_date:
                raise ValueError(
                    "Min ROOS risk schedules存在衝突重疊: "
                    f"{left.source_id}@{left.start_date.date()} vs "
                    f"{right.source_id}@{right.start_date.date()}"
                )
    if not periods:
        raise ValueError(f"Min ROOS risk schedule為空: {source_paths}")
    return tuple(periods)


def resolve_risk_period(
    schedule: Iterable[RiskParamPeriod],
    decision_date: Any,
) -> RiskParamPeriod | None:
    date_value = _normalized_date(decision_date)
    matches = [
        period
        for period in schedule
        if period.start_date <= date_value
        and (period.end_date is None or date_value <= period.end_date)
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item.start_date, item.source_id))
    selected = matches[-1]
    # Same-start conflicts cannot be silently resolved by source name.
    same_start = [p for p in matches if p.start_date == selected.start_date]
    tuples = {(p.atr_len, round(p.atr_times_init, 12)) for p in same_start}
    if len(tuples) > 1:
        raise ValueError(f"Min ROOS risk schedule同日存在衝突: {date_value.date()}")
    return selected


def build_risk_target_contract(
    *,
    horizon_bars: int,
    param_policy: str,
) -> dict[str, object]:
    return {
        "schema_version": RISK_NORMALIZED_TARGET_SCHEMA_VERSION,
        "target_id": DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID,
        "objective_family": "daily_universal_cost_adjusted_risk_normalized_opportunity",
        "sample_scope": "daily_eligible_stock_days",
        "information_source": "fixed_future_40d_high_low_path_plus_pit_min_roos_risk_geometry",
        "horizon_bars": int(horizon_bars),
        "reference_price": "decision_date_close",
        "risk_param_policy": str(param_policy),
        "risk_param_source_ids": list(RISK_NORMALIZATION_SOURCE_IDS),
        "risk_fields": list(RISK_NORMALIZATION_FIELDS),
        "excluded_strategy_fields": [
            "high_len",
            "atr_buy_tol",
            "atr_times_trail",
            "breakout_qualification",
            "K",
            "R0",
            "cash",
            "holdings",
        ],
        "risk_distance": "reference_close - canonical_tick_rounded_initial_stop(reference_close, ATR(atr_len), atr_times_init)",
        "sizing": "canonical fixed-risk + canonical max-position-cap + board-lot preference at standardized equity",
        "standardized_equity": float(STANDARDIZED_EQUITY),
        "fixed_risk": float(DEFAULT_FIXED_RISK),
        "max_position_cap_pct": float(V16StrategyParams().max_position_cap_pct),
        "accounting": "canonical buy/sell fee + security-specific sell tax + minimum fee",
        "formula": (
            "[(peak-reference) - (reference-running_low_to_peak)]*qty "
            "- canonical_round_trip_cost(reference, peak, qty); divided by "
            "canonical_planned_initial_risk(reference, stop, qty)"
        ),
        "peak_rule": "earliest maximum high before first planned-stop touch",
        "risk_rule": "same-bar adverse-first; stop-touch bar high excluded",
        "no_safe_bar_rule": "canonical stop-out net PnL / planned initial risk",
        "normalization": "none",
        "clipping": "none",
        "split_derived_parameters": False,
        "oos_derived_parameters": False,
        "strategy_exit_path_used": False,
    }


def _params_for_period(period: RiskParamPeriod) -> V16StrategyParams:
    # Defaults supply canonical fees, fixed-risk and position-cap semantics.  Only the two
    # historical Min ROOS risk fields are replaced; no breakout/exit field may affect target.
    params = V16StrategyParams()
    params.atr_len = int(period.atr_len)
    params.atr_times_init = float(period.atr_times_init)
    params.fixed_risk = float(DEFAULT_FIXED_RISK)
    return params


def compute_risk_geometry(
    *,
    ticker: str,
    decision_date: Any,
    reference_price: float,
    atr: float,
    period: RiskParamPeriod | None,
    standardized_equity: float = STANDARDIZED_EQUITY,
) -> RiskGeometryResult:
    if period is None:
        return RiskGeometryResult(False, "missing_risk_period", math.nan, math.nan, math.nan, 0, 0, ())
    if not math.isfinite(reference_price) or reference_price <= 0.0:
        return RiskGeometryResult(False, "invalid_reference", math.nan, math.nan, math.nan, 0, 0, ())
    if not math.isfinite(atr) or atr <= 0.0:
        return RiskGeometryResult(False, "invalid_atr", float(reference_price), math.nan, math.nan, 0, 0, ())
    params = _params_for_period(period)
    stop = float(calc_initial_stop_from_reference(reference_price, atr, params, ticker=ticker))
    risk_distance = float(reference_price - stop)
    if not math.isfinite(stop) or stop <= 0.0 or risk_distance <= 0.0:
        return RiskGeometryResult(False, "invalid_stop", float(reference_price), float(atr), stop, 0, 0, ())
    qty = int(
        calc_position_size(
            reference_price,
            stop,
            standardized_equity,
            float(DEFAULT_FIXED_RISK),
            params,
            ticker=ticker,
            trade_date=pd.Timestamp(decision_date).date(),
        )
    )
    qty = int(apply_board_lot_preferred_qty(reference_price, qty, params))
    if qty <= 0:
        return RiskGeometryResult(False, "zero_qty", float(reference_price), float(atr), stop, 0, 0, ())
    planned_risk_milli = int(
        calc_planned_initial_risk_from_prices_milli(
            reference_price,
            stop,
            qty,
            params,
            ticker=ticker,
            trade_date=pd.Timestamp(decision_date).date(),
        )
    )
    if planned_risk_milli <= 0:
        return RiskGeometryResult(False, "non_positive_planned_risk", float(reference_price), float(atr), stop, qty, 0, ())
    buy = build_buy_ledger_from_price(reference_price, qty, params)
    flat_sell = build_sell_ledger_from_price(
        reference_price,
        qty,
        params,
        ticker=ticker,
        trade_date=pd.Timestamp(decision_date).date(),
    )
    entry_cost_milli = int(buy["net_buy_total_milli"])
    flat_round_trip_cost_milli = int(
        buy["buy_fee_milli"] + flat_sell["sell_fee_milli"] + flat_sell["tax_milli"]
    )
    risk_distance_pct = risk_distance / float(reference_price)
    risk_distance_atr = risk_distance / float(atr)
    capital_per_risk = entry_cost_milli / float(planned_risk_milli)
    cost_per_risk = flat_round_trip_cost_milli / float(planned_risk_milli)
    intended_risk_milli = float(standardized_equity) * 1000.0 * float(DEFAULT_FIXED_RISK)
    risk_capacity = min(1.0, planned_risk_milli / intended_risk_milli) if intended_risk_milli > 0 else 0.0
    context = (
        float(risk_distance_pct),
        float(risk_distance_atr),
        float(capital_per_risk),
        float(cost_per_risk),
        float(risk_capacity),
    )
    if not all(math.isfinite(value) for value in context):
        return RiskGeometryResult(False, "non_finite_context", float(reference_price), float(atr), stop, qty, planned_risk_milli, ())
    return RiskGeometryResult(
        True,
        "ok",
        float(reference_price),
        float(atr),
        float(stop),
        int(qty),
        int(planned_risk_milli),
        context,
    )


def risk_normalized_target_from_future_path(
    *,
    ticker: str,
    decision_date: Any,
    reference_price: float,
    stop_price: float,
    qty: int,
    planned_initial_risk_milli: int,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_dates: Iterable[Any],
    params: V16StrategyParams,
) -> tuple[float, bool, str]:
    highs = np.asarray(future_high, dtype=np.float64)
    lows = np.asarray(future_low, dtype=np.float64)
    dates = list(future_dates)
    if highs.ndim != 1 or lows.ndim != 1 or highs.shape != lows.shape or len(dates) != len(highs):
        return math.nan, False, "invalid_future_shape"
    if len(highs) == 0:
        return math.nan, False, "insufficient_future"
    valid = np.isfinite(highs) & np.isfinite(lows) & (highs > 0.0) & (lows > 0.0) & (highs >= lows)
    if not bool(np.all(valid)):
        return math.nan, False, "invalid_future_bar"
    buy = build_buy_ledger_from_price(reference_price, int(qty), params)
    running_low = float(reference_price)
    best_high = -math.inf
    best_low = math.nan
    best_date: Any = None
    for high_value, low_value, bar_date in zip(highs, lows, dates):
        low_value = float(low_value)
        high_value = float(high_value)
        if low_value <= float(stop_price):
            if not math.isfinite(best_high):
                stop_sell = build_sell_ledger_from_price(
                    stop_price,
                    int(qty),
                    params,
                    ticker=ticker,
                    trade_date=pd.Timestamp(bar_date).date(),
                )
                net_pnl_milli = int(stop_sell["net_sell_total_milli"]) - int(buy["net_buy_total_milli"])
                return float(net_pnl_milli / float(planned_initial_risk_milli)), True, "stop_first"
            break
        running_low = min(running_low, low_value)
        if high_value > best_high:
            best_high = high_value
            best_low = running_low
            best_date = bar_date
    if not math.isfinite(best_high):
        return math.nan, False, "no_safe_peak"
    sell = build_sell_ledger_from_price(
        best_high,
        int(qty),
        params,
        ticker=ticker,
        trade_date=pd.Timestamp(best_date).date(),
    )
    reference_milli = int(price_to_milli(reference_price))
    peak_milli = int(price_to_milli(best_high))
    adverse_low_milli = int(price_to_milli(best_low))
    gross_favorable_milli = int((peak_milli - reference_milli) * int(qty))
    gross_adverse_milli = int((reference_milli - adverse_low_milli) * int(qty))
    round_trip_cost_milli = int(
        buy["buy_fee_milli"] + sell["sell_fee_milli"] + sell["tax_milli"]
    )
    numerator_milli = gross_favorable_milli - gross_adverse_milli - round_trip_cost_milli
    target = numerator_milli / float(planned_initial_risk_milli)
    if not math.isfinite(target):
        return math.nan, False, "non_finite_target"
    return float(target), True, "ok"


def compute_targets_and_context_for_positions(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    ticker: str,
    schedule: tuple[RiskParamPeriod, ...],
    horizon_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return target, target-valid, context, geometry-valid arrays for one ticker."""

    positions = np.asarray(positions, dtype=np.int64)
    count = len(positions)
    target = np.full(count, np.nan, dtype=np.float32)
    target_valid = np.zeros(count, dtype=bool)
    context = np.zeros((count, len(RISK_GEOMETRY_CONTEXT_FEATURES)), dtype=np.float32)
    geometry_valid = np.zeros(count, dtype=bool)
    if count == 0:
        return target, target_valid, context, geometry_valid
    high = frame["High"].to_numpy(dtype=np.float64, copy=False)
    low = frame["Low"].to_numpy(dtype=np.float64, copy=False)
    close = frame["Close"].to_numpy(dtype=np.float64, copy=False)
    dates = pd.DatetimeIndex(frame.index).normalize()
    atr_cache: dict[int, np.ndarray] = {}
    params_cache: dict[tuple[int, float], V16StrategyParams] = {}
    for out_idx, source_pos in enumerate(positions.tolist()):
        decision_date = dates[source_pos]
        period = resolve_risk_period(schedule, decision_date)
        if period is None:
            continue
        atr_values = atr_cache.get(int(period.atr_len))
        if atr_values is None:
            atr_values = tv_atr(high, low, close, int(period.atr_len))
            atr_cache[int(period.atr_len)] = atr_values
        geometry = compute_risk_geometry(
            ticker=ticker,
            decision_date=decision_date,
            reference_price=float(close[source_pos]),
            atr=float(atr_values[source_pos]),
            period=period,
        )
        if not geometry.valid:
            continue
        geometry_valid[out_idx] = True
        context[out_idx] = np.asarray(geometry.context, dtype=np.float32)
        if source_pos + int(horizon_bars) >= len(frame):
            continue
        key = (int(period.atr_len), float(period.atr_times_init))
        params = params_cache.get(key)
        if params is None:
            params = _params_for_period(period)
            params_cache[key] = params
        start = source_pos + 1
        end = source_pos + int(horizon_bars) + 1
        value, valid_target, _reason = risk_normalized_target_from_future_path(
            ticker=ticker,
            decision_date=decision_date,
            reference_price=geometry.reference_price,
            stop_price=geometry.stop_price,
            qty=geometry.qty,
            planned_initial_risk_milli=geometry.planned_initial_risk_milli,
            future_high=high[start:end],
            future_low=low[start:end],
            future_dates=dates[start:end],
            params=params,
        )
        if valid_target:
            target[out_idx] = np.float32(value)
            target_valid[out_idx] = True
    return target, target_valid, context, geometry_valid


__all__ = [
    "DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID",
    "RISK_GEOMETRY_CONTEXT_FEATURES",
    "RISK_NORMALIZATION_FIELDS",
    "RISK_NORMALIZATION_SOURCE_IDS",
    "RiskGeometryResult",
    "RiskParamPeriod",
    "build_risk_target_contract",
    "compute_risk_geometry",
    "compute_targets_and_context_for_positions",
    "load_min_roos_risk_schedule",
    "resolve_risk_period",
    "risk_normalized_target_from_future_path",
]
