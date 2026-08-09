"""A2 point-in-time realized trade-path label contract and simulator."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.training_policy import (
    OUTER_ROLLING_OOS_HORIZON_MONTHS,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)

from core.backtest_finalize import finalize_open_position_at_end
from core.capital_policy import resolve_single_backtest_sizing_capital
from core.entry_plans import build_normal_entry_plan, execute_pre_market_entry_plan
from core.exact_accounting import calc_ratio_from_milli, milli_to_money
from core.extended_signals import (
    build_extended_entry_plan_from_signal,
    create_signal_tracking_state,
    should_clear_extended_signal,
)
from core.position_step import execute_bar_step
from core.signal_utils import generate_signals
from core.strategy_params import V16StrategyParams
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.contract import (
    LABEL_INVALID,
    LABEL_PASS,
    LABEL_REJECT,
    TRADE_PATH_LABEL_CONTRACT_VERSION,
    TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_LABEL_STATUS_PASS,
    TRADE_PATH_LABEL_STATUS_REJECT,
)

TRADE_PATH_BASE_FILTER_ID = "breakout_quality_v1"
TRADE_PATH_RESEARCH_FILTER_ID = "breakout_quality_a2_trade_path_v1"
TRADE_PATH_LABEL_ID = "a2_realized_trade_path_v1"
TRADE_PATH_SELECTION_BASELINE_PARAMS_RELATIVE_PATH = Path(
    "models/research/breakout_quality/selection_strategy_realization/roos_base_best.json"
)
TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE = "2014-01-01"
TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE = "2020-12-31"
TRADE_PATH_SELECTION_BASELINE_TRAIN_WINDOW_MONTHS = OUTER_ROLLING_TRAIN_WINDOW_MONTHS
TRADE_PATH_SELECTION_BASELINE_OOS_MONTHS = OUTER_ROLLING_OOS_HORIZON_MONTHS
TRADE_PATH_HISTORICAL_TEACHER_RELATIVE_DIR = Path(
    "models/research/breakout_quality/trade_path_label/a2_teacher_params"
)
TRADE_PATH_HISTORICAL_TEACHER_PARAMS_RELATIVE_PATH = (
    TRADE_PATH_HISTORICAL_TEACHER_RELATIVE_DIR
    / "p2_dl_off_trained"
    / "active_params"
    / "roos_base_best.json"
)
TRADE_PATH_FORWARD_TEACHER_PARAMS_RELATIVE_PATH = Path(
    "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
    "risk_only_rolling/p2_dl_off_trained/active_params/roos_base_best.json"
)

TRADE_PATH_REASON_REALIZED_NET_PROFIT = "realized_net_profit"
TRADE_PATH_REASON_REALIZED_NET_NONPOSITIVE = "realized_net_nonpositive"
TRADE_PATH_REASON_INSUFFICIENT_FUTURE = "insufficient_future"
TRADE_PATH_REASON_NOT_FORMAL_SETUP = "not_a2_formal_setup"
TRADE_PATH_REASON_INVALID_ENTRY_PLAN = "invalid_entry_plan"
TRADE_PATH_REASON_UNFILLED_SUPERSEDED = "unfilled_superseded_by_new_setup"
TRADE_PATH_REASON_UNFILLED_TERMINATED = "unfilled_terminated"
TRADE_PATH_REASON_UNFILLED_DATA_END = "unfilled_censored_at_data_end"
TRADE_PATH_REASON_TEACHER_UNAVAILABLE = "teacher_params_unavailable"
TRADE_PATH_REASON_INACTIVE_HIGH_LEN = "inactive_high_len_for_teacher"
TRADE_PATH_REASON_SIGNAL_DATE_MISSING = "signal_date_missing_from_source"

TRADE_PATH_LABEL_REASON_STATUS = {
    TRADE_PATH_REASON_REALIZED_NET_PROFIT: TRADE_PATH_LABEL_STATUS_PASS,
    TRADE_PATH_REASON_REALIZED_NET_NONPOSITIVE: TRADE_PATH_LABEL_STATUS_REJECT,
    TRADE_PATH_REASON_INSUFFICIENT_FUTURE: TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_REASON_NOT_FORMAL_SETUP: TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_REASON_INVALID_ENTRY_PLAN: TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_REASON_UNFILLED_SUPERSEDED: TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_REASON_UNFILLED_TERMINATED: TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_REASON_UNFILLED_DATA_END: TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_REASON_TEACHER_UNAVAILABLE: TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_REASON_INACTIVE_HIGH_LEN: TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_REASON_SIGNAL_DATE_MISSING: TRADE_PATH_LABEL_STATUS_EXCLUDED,
}


@dataclass(frozen=True)
class ActiveParamSchedule:
    source_path: Path
    source_sha256: str
    effective_dates: tuple[pd.Timestamp, ...]
    params_by_effective_date: dict[str, V16StrategyParams]

    @property
    def available_from(self) -> pd.Timestamp:
        return self.effective_dates[0]

    @property
    def available_through(self) -> pd.Timestamp:
        return self.effective_dates[-1]

    def resolve(self, value: Any) -> tuple[str, V16StrategyParams] | None:
        date_value = pd.Timestamp(value).normalize()
        index = bisect_right(self.effective_dates, date_value) - 1
        if index < 0:
            return None
        effective_date = self.effective_dates[index].strftime("%Y-%m-%d")
        return effective_date, self.params_by_effective_date[effective_date]


@dataclass(frozen=True)
class TradePathLabelResult:
    label: int
    status: str
    reason: str
    label_eval_start_date: str | None
    label_eval_end_date: str | None
    teacher_effective_date: str | None
    fill_type: str | None
    fill_date: str | None
    entry_price: float | None
    exit_date: str | None
    exit_price: float | None
    exit_reason: str | None
    continuation_wait_bars: int | None
    initial_missed_buy: bool
    forced_closeout: bool
    sizing_capital: float | None
    realized_net_pnl: float | None
    realized_net_r: float | None

    def __post_init__(self) -> None:
        expected_status = TRADE_PATH_LABEL_REASON_STATUS.get(str(self.reason))
        if expected_status is None:
            raise ValueError(f"未知trade-path label reason: {self.reason}")
        if str(self.status) != expected_status:
            raise ValueError(
                "trade-path label status／reason不一致: "
                f"status={self.status}, reason={self.reason}, expected={expected_status}"
            )
        expected_label = {
            TRADE_PATH_LABEL_STATUS_PASS: LABEL_PASS,
            TRADE_PATH_LABEL_STATUS_REJECT: LABEL_REJECT,
            TRADE_PATH_LABEL_STATUS_EXCLUDED: LABEL_INVALID,
        }[expected_status]
        if int(self.label) != int(expected_label):
            raise ValueError(
                "trade-path label value／status不一致: "
                f"label={self.label}, status={self.status}, expected={expected_label}"
            )

    def as_event_update(self) -> dict[str, Any]:
        return {
            "label": int(self.label),
            "label_status": str(self.status),
            "label_reason": str(self.reason),
            "label_eval_start_date": self.label_eval_start_date,
            "label_eval_end_date": self.label_eval_end_date,
            "teacher_effective_date": self.teacher_effective_date,
            "trade_path_fill_type": self.fill_type,
            "trade_path_fill_date": self.fill_date,
            "trade_path_entry_price": self.entry_price,
            "trade_path_exit_date": self.exit_date,
            "trade_path_exit_price": self.exit_price,
            "trade_path_exit_reason": self.exit_reason,
            "trade_path_continuation_wait_bars": self.continuation_wait_bars,
            "trade_path_initial_missed_buy": bool(self.initial_missed_buy),
            "trade_path_forced_closeout": bool(self.forced_closeout),
            "trade_path_sizing_capital": self.sizing_capital,
            "trade_path_realized_net_pnl": self.realized_net_pnl,
            "trade_path_realized_net_r": self.realized_net_r,
        }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取A2 teacher params: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"A2 teacher params根節點必須是object: {path}")
    return payload


def _extract_single_member_params(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ensemble = payload.get("params_ensemble_by_effective_date")
    if isinstance(ensemble, dict) and ensemble:
        output: dict[str, dict[str, Any]] = {}
        for raw_date, raw_members in ensemble.items():
            members = list(raw_members or [])
            if len(members) != 1:
                raise ValueError(
                    "trade-path teacher目前要求每個effective date恰有1個member: "
                    f"date={raw_date}, members={len(members)}"
                )
            params = dict((members[0] or {}).get("params") or {})
            if not params:
                raise ValueError(f"trade-path teacher member缺少params: {raw_date}")
            output[pd.Timestamp(raw_date).strftime("%Y-%m-%d")] = params
        return output
    mapping = payload.get("params_by_effective_date")
    if isinstance(mapping, dict) and mapping:
        return {
            pd.Timestamp(raw_date).strftime("%Y-%m-%d"): dict(raw_params or {})
            for raw_date, raw_params in mapping.items()
        }
    raise ValueError("A2 teacher params缺少params_ensemble_by_effective_date")


def _validate_teacher_param_contract(params: V16StrategyParams, *, effective_date: str) -> None:
    required_false = (
        "use_breakout_ema_filter",
        "use_bb",
        "use_kc",
        "use_vol",
        "use_breakout_return_filter",
        "use_breakout_false_filter",
        "use_history_threshold",
        "use_breakout_reclaim_reentry",
        "use_breakout_quality_filter",
        "use_breakout_quality_ranking",
    )
    enabled = [name for name in required_false if bool(getattr(params, name))]
    if enabled:
        raise ValueError(
            "A2 trade-path teacher必須是rule-based filters全關且DL關: "
            f"date={effective_date}, enabled={enabled}"
        )


def load_active_param_schedule(path: str | Path) -> ActiveParamSchedule:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到A2 teacher params: {resolved}")
    payload = _read_json_object(resolved)
    raw_mapping = _extract_single_member_params(payload)
    params_mapping: dict[str, V16StrategyParams] = {}
    for effective_date, raw_params in sorted(raw_mapping.items()):
        params = V16StrategyParams(**raw_params)
        _validate_teacher_param_contract(params, effective_date=effective_date)
        params_mapping[effective_date] = params
    dates = tuple(pd.Timestamp(value).normalize() for value in params_mapping)
    if not dates:
        raise ValueError("A2 teacher params沒有effective dates")
    return ActiveParamSchedule(
        source_path=resolved,
        source_sha256=compute_file_sha256(resolved),
        effective_dates=dates,
        params_by_effective_date=params_mapping,
    )


def merge_active_param_schedules(*schedules: ActiveParamSchedule) -> ActiveParamSchedule:
    if not schedules:
        raise ValueError("至少需要一份A2 teacher schedule")
    merged: dict[str, V16StrategyParams] = {}
    source_records = []
    for schedule in schedules:
        source_records.append(f"{schedule.source_path}:{schedule.source_sha256}")
        for effective_date, params in schedule.params_by_effective_date.items():
            if effective_date in merged:
                raise ValueError(f"A2 teacher schedule effective date重複: {effective_date}")
            merged[effective_date] = params
    ordered = dict(sorted(merged.items()))
    synthetic_source = Path(" | ".join(source_records))
    import hashlib

    synthetic_sha = hashlib.sha256("\n".join(source_records).encode("utf-8")).hexdigest()
    return ActiveParamSchedule(
        source_path=synthetic_source,
        source_sha256=synthetic_sha,
        effective_dates=tuple(pd.Timestamp(value).normalize() for value in ordered),
        params_by_effective_date=ordered,
    )


def build_signal_cache(stock_df: pd.DataFrame, params: V16StrategyParams, *, ticker: str):
    return generate_signals(stock_df, params, ticker=ticker)


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _excluded_result(
    reason: str,
    *,
    start_date: str | None,
    end_date: str | None,
    teacher_effective_date: str | None,
    fill_type: str | None = None,
    fill_date: str | None = None,
    entry_price: float | None = None,
    wait_bars: int | None = None,
    initial_missed_buy: bool = False,
    sizing_capital: float | None = None,
) -> TradePathLabelResult:
    return TradePathLabelResult(
        label=LABEL_INVALID,
        status=TRADE_PATH_LABEL_STATUS_EXCLUDED,
        reason=reason,
        label_eval_start_date=start_date,
        label_eval_end_date=end_date,
        teacher_effective_date=teacher_effective_date,
        fill_type=fill_type,
        fill_date=fill_date,
        entry_price=entry_price,
        exit_date=None,
        exit_price=None,
        exit_reason=None,
        continuation_wait_bars=wait_bars,
        initial_missed_buy=bool(initial_missed_buy),
        forced_closeout=False,
        sizing_capital=sizing_capital,
        realized_net_pnl=None,
        realized_net_r=None,
    )


def build_trade_path_excluded_event_update(
    reason: str,
    *,
    end_date: str | None,
    teacher_effective_date: str | None,
) -> dict[str, Any]:
    return _excluded_result(
        reason,
        start_date=None,
        end_date=end_date,
        teacher_effective_date=teacher_effective_date,
    ).as_event_update()


def _completed_result(
    *,
    realized_pnl: float,
    realized_r: float,
    eval_start_date: str,
    exit_date: str,
    exit_price: float | None,
    exit_reason: str,
    teacher_effective_date: str,
    fill_type: str,
    fill_date: str,
    entry_price: float,
    wait_bars: int,
    initial_missed_buy: bool,
    forced_closeout: bool,
    sizing_capital: float,
) -> TradePathLabelResult:
    numeric_fields = {
        "realized_pnl": realized_pnl,
        "realized_r": realized_r,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "sizing_capital": sizing_capital,
    }
    invalid_numeric = {
        name: value
        for name, value in numeric_fields.items()
        if value is None or not math.isfinite(float(value))
    }
    if invalid_numeric:
        raise ValueError(f"trade-path完整交易含無效數值: {invalid_numeric}")
    if float(entry_price) <= 0.0 or float(exit_price) <= 0.0:
        raise ValueError(
            "trade-path完整交易進出價格必須為正值: "
            f"entry={entry_price}, exit={exit_price}"
        )
    if float(sizing_capital) <= 0.0:
        raise ValueError(f"trade-path完整交易sizing capital必須為正值: {sizing_capital}")
    if not all(str(value or "").strip() for value in (fill_type, fill_date, exit_date, exit_reason)):
        raise ValueError("trade-path完整交易缺少fill／exit identity")

    label = LABEL_PASS if float(realized_r) > 0.0 else LABEL_REJECT
    status = (
        TRADE_PATH_LABEL_STATUS_PASS
        if label == LABEL_PASS
        else TRADE_PATH_LABEL_STATUS_REJECT
    )
    reason = (
        TRADE_PATH_REASON_REALIZED_NET_PROFIT
        if label == LABEL_PASS
        else TRADE_PATH_REASON_REALIZED_NET_NONPOSITIVE
    )
    return TradePathLabelResult(
        label=label,
        status=status,
        reason=reason,
        label_eval_start_date=eval_start_date,
        label_eval_end_date=exit_date,
        teacher_effective_date=teacher_effective_date,
        fill_type=fill_type,
        fill_date=fill_date,
        entry_price=float(entry_price),
        exit_date=exit_date,
        exit_price=None if exit_price is None else float(exit_price),
        exit_reason=exit_reason,
        continuation_wait_bars=int(wait_bars),
        initial_missed_buy=bool(initial_missed_buy),
        forced_closeout=bool(forced_closeout),
        sizing_capital=float(sizing_capital),
        realized_net_pnl=float(realized_pnl),
        realized_net_r=float(realized_r),
    )


def simulate_realized_trade_path_label(
    stock_df: pd.DataFrame,
    *,
    ticker: str,
    signal_pos: int,
    params: V16StrategyParams,
    teacher_effective_date: str,
    precomputed_signals,
    sizing_capital: float | None = None,
) -> TradePathLabelResult:
    if signal_pos < 0 or signal_pos >= len(stock_df) - 1:
        return _excluded_result(
            TRADE_PATH_REASON_INSUFFICIENT_FUTURE,
            start_date=None,
            end_date=None,
            teacher_effective_date=teacher_effective_date,
        )
    atr, buy_condition, sell_condition, buy_limits = precomputed_signals
    signal_date = stock_df.index[signal_pos]
    eval_start_date = _date_text(stock_df.index[signal_pos + 1])
    if not bool(buy_condition[signal_pos]) or not math.isfinite(float(atr[signal_pos])):
        return _excluded_result(
            TRADE_PATH_REASON_NOT_FORMAL_SETUP,
            start_date=eval_start_date,
            end_date=_date_text(signal_date),
            teacher_effective_date=teacher_effective_date,
        )

    signal_state = create_signal_tracking_state(
        buy_limits[signal_pos],
        atr[signal_pos],
        params,
        ticker=ticker,
        security_profile=stock_df.attrs.get("security_profile"),
        signal_date=signal_date,
    )
    if signal_state is None:
        return _excluded_result(
            TRADE_PATH_REASON_INVALID_ENTRY_PLAN,
            start_date=eval_start_date,
            end_date=_date_text(signal_date),
            teacher_effective_date=teacher_effective_date,
        )

    open_values = stock_df["Open"].to_numpy(dtype=np.float64, copy=False)
    high_values = stock_df["High"].to_numpy(dtype=np.float64, copy=False)
    low_values = stock_df["Low"].to_numpy(dtype=np.float64, copy=False)
    close_values = stock_df["Close"].to_numpy(dtype=np.float64, copy=False)
    volume_values = stock_df["Volume"].to_numpy(dtype=np.float64, copy=False)
    dates = stock_df.index
    resolved_sizing_capital = (
        resolve_single_backtest_sizing_capital(params, params.initial_capital)
        if sizing_capital is None
        else float(sizing_capital)
    )
    if not math.isfinite(float(resolved_sizing_capital)) or float(resolved_sizing_capital) <= 0.0:
        raise ValueError(
            "trade-path sizing capital必須為有限正值: "
            f"ticker={ticker}, signal_date={_date_text(signal_date)}, value={resolved_sizing_capital}"
        )
    active_signal = signal_state
    position: dict[str, Any] = {"qty": 0}
    fill_type: str | None = None
    fill_date: str | None = None
    entry_price: float | None = None
    wait_bars: int | None = None
    initial_missed_buy = False

    for current_pos in range(signal_pos + 1, len(stock_df)):
        if int(position.get("qty", 0) or 0) > 0:
            position, _freed_cash, _pnl, events = execute_bar_step(
                position,
                atr[current_pos - 1],
                sell_condition[current_pos - 1],
                close_values[current_pos - 1],
                open_values[current_pos],
                high_values[current_pos],
                low_values[current_pos],
                close_values[current_pos],
                volume_values[current_pos],
                params,
                current_date=dates[current_pos],
                y_high=high_values[current_pos - 1],
                return_milli=True,
                record_exec_contexts=True,
                sync_display_fields=False,
            )
            terminal_events = [value for value in ("STOP", "IND_SELL") if value in events]
            if terminal_events:
                terminal_event = terminal_events[0]
                terminal_context = next(
                    (
                        context
                        for context in reversed(position.get("_last_exec_contexts", []))
                        if str(context.get("event")) == terminal_event
                    ),
                    None,
                )
                realized_pnl_milli = int(position.get("realized_pnl_milli", 0) or 0)
                initial_risk_milli = int(position.get("initial_risk_total_milli", 0) or 0)
                realized_r = calc_ratio_from_milli(realized_pnl_milli, initial_risk_milli)
                exit_date = _date_text(dates[current_pos])
                return _completed_result(
                    realized_pnl=milli_to_money(realized_pnl_milli),
                    realized_r=float(realized_r),
                    eval_start_date=eval_start_date,
                    exit_date=exit_date,
                    exit_price=None if terminal_context is None else terminal_context.get("exec_price"),
                    exit_reason=terminal_event,
                    teacher_effective_date=teacher_effective_date,
                    fill_type=str(fill_type),
                    fill_date=str(fill_date),
                    entry_price=float(entry_price),
                    wait_bars=int(wait_bars or 0),
                    initial_missed_buy=initial_missed_buy,
                    forced_closeout=False,
                    sizing_capital=float(resolved_sizing_capital),
                )
            continue

        if current_pos > signal_pos + 1 and bool(buy_condition[current_pos - 1]):
            return _excluded_result(
                TRADE_PATH_REASON_UNFILLED_SUPERSEDED,
                start_date=eval_start_date,
                end_date=_date_text(dates[current_pos - 1]),
                teacher_effective_date=teacher_effective_date,
                wait_bars=current_pos - signal_pos - 1,
                initial_missed_buy=initial_missed_buy,
                sizing_capital=float(resolved_sizing_capital),
            )

        if current_pos == signal_pos + 1:
            entry_plan = build_normal_entry_plan(
                buy_limits[signal_pos],
                atr[signal_pos],
                resolved_sizing_capital,
                params,
                ticker=ticker,
                security_profile=stock_df.attrs.get("security_profile"),
                trade_date=dates[current_pos],
            )
            entry_type = "normal"
        else:
            entry_plan = build_extended_entry_plan_from_signal(
                active_signal,
                resolved_sizing_capital,
                params,
                y_close=close_values[current_pos - 1],
                ticker=ticker,
                security_profile=stock_df.attrs.get("security_profile"),
                trade_date=dates[current_pos],
            )
            entry_type = "extended"

        entry_result = execute_pre_market_entry_plan(
            entry_plan=entry_plan,
            t_open=open_values[current_pos],
            t_high=high_values[current_pos],
            t_low=low_values[current_pos],
            t_close=close_values[current_pos],
            t_volume=volume_values[current_pos],
            y_close=close_values[current_pos - 1],
            params=params,
            entry_type=entry_type,
            ticker=ticker,
            security_profile=stock_df.attrs.get("security_profile"),
            trade_date=dates[current_pos],
        )
        if current_pos == signal_pos + 1 and bool(entry_result.get("count_as_missed_buy")):
            initial_missed_buy = True
        if bool(entry_result.get("filled")):
            position = dict(entry_result["position"])
            position["signal_date"] = signal_date
            fill_type = "INITIAL_FILL" if current_pos == signal_pos + 1 else "CONTINUATION_FILL"
            fill_date = _date_text(dates[current_pos])
            entry_price = float(entry_result["entry_fill_price"])
            wait_bars = current_pos - signal_pos - 1
            active_signal = None
            continue

        if should_clear_extended_signal(
            active_signal,
            low_values[current_pos],
            high_values[current_pos],
            t_open=open_values[current_pos],
            t_close=close_values[current_pos],
            t_volume=volume_values[current_pos],
            y_close=close_values[current_pos - 1],
            y_high=high_values[current_pos - 1],
            y_atr=atr[current_pos - 1],
            y_ind_sell=sell_condition[current_pos - 1],
            sizing_capital=resolved_sizing_capital,
            current_date=dates[current_pos],
            params=params,
            copy_shadow_position=False,
        ):
            return _excluded_result(
                TRADE_PATH_REASON_UNFILLED_TERMINATED,
                start_date=eval_start_date,
                end_date=_date_text(dates[current_pos]),
                teacher_effective_date=teacher_effective_date,
                wait_bars=current_pos - signal_pos,
                initial_missed_buy=initial_missed_buy,
                sizing_capital=float(resolved_sizing_capital),
            )

    if fill_date is None:
        return _excluded_result(
            TRADE_PATH_REASON_UNFILLED_DATA_END,
            start_date=eval_start_date,
            end_date=_date_text(dates[-1]),
            teacher_effective_date=teacher_effective_date,
            wait_bars=wait_bars,
            initial_missed_buy=initial_missed_buy,
            sizing_capital=float(resolved_sizing_capital),
        )

    final_state = finalize_open_position_at_end(
        position=dict(position),
        ticker=ticker,
        final_close=close_values[-1],
        final_date=dates[-1],
        current_capital_milli=0,
        current_equity_milli=0,
        peak_capital_milli=0,
        max_drawdown_pct=0.0,
        trade_count=0,
        full_wins=0,
        total_profit_milli=0,
        total_loss_milli=0,
        total_r_multiple=0.0,
        total_r_win=0.0,
        total_r_loss=0.0,
        trade_logs=[],
        return_logs=False,
        params=params,
        collect_stats=False,
    )
    realized_pnl = final_state.get("final_trade_pnl")
    realized_r = final_state.get("final_trade_r_mult")
    if realized_pnl is None or realized_r is None:
        raise RuntimeError(
            "trade-path已成交但正式資料尾端結算未產生交易結果: "
            f"ticker={ticker}, signal_date={_date_text(signal_date)}"
        )
    return _completed_result(
        realized_pnl=float(realized_pnl),
        realized_r=float(realized_r),
        eval_start_date=eval_start_date,
        exit_date=_date_text(dates[-1]),
        exit_price=final_state.get("final_trade_exit_price"),
        exit_reason=str(final_state.get("final_trade_exit_reason") or "FORCED_CLOSEOUT"),
        teacher_effective_date=teacher_effective_date,
        fill_type=str(fill_type),
        fill_date=str(fill_date),
        entry_price=float(entry_price),
        wait_bars=int(wait_bars or 0),
        initial_missed_buy=initial_missed_buy,
        forced_closeout=True,
        sizing_capital=float(resolved_sizing_capital),
    )


__all__ = [
    "ActiveParamSchedule",
    "TRADE_PATH_BASE_FILTER_ID",
    "TRADE_PATH_FORWARD_TEACHER_PARAMS_RELATIVE_PATH",
    "TRADE_PATH_HISTORICAL_TEACHER_PARAMS_RELATIVE_PATH",
    "TRADE_PATH_HISTORICAL_TEACHER_RELATIVE_DIR",
    "TRADE_PATH_LABEL_ID",
    "TRADE_PATH_RESEARCH_FILTER_ID",
    "TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE",
    "TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE",
    "TRADE_PATH_SELECTION_BASELINE_TRAIN_WINDOW_MONTHS",
    "TRADE_PATH_SELECTION_BASELINE_OOS_MONTHS",
    "TRADE_PATH_SELECTION_BASELINE_PARAMS_RELATIVE_PATH",
    "TRADE_PATH_LABEL_CONTRACT_VERSION",
    "TRADE_PATH_LABEL_REASON_STATUS",
    "TRADE_PATH_LABEL_STATUS_EXCLUDED",
    "TRADE_PATH_LABEL_STATUS_PASS",
    "TRADE_PATH_LABEL_STATUS_REJECT",
    "TRADE_PATH_REASON_INACTIVE_HIGH_LEN",
    "TRADE_PATH_REASON_INSUFFICIENT_FUTURE",
    "TRADE_PATH_REASON_INVALID_ENTRY_PLAN",
    "TRADE_PATH_REASON_NOT_FORMAL_SETUP",
    "TRADE_PATH_REASON_REALIZED_NET_NONPOSITIVE",
    "TRADE_PATH_REASON_REALIZED_NET_PROFIT",
    "TRADE_PATH_REASON_SIGNAL_DATE_MISSING",
    "TRADE_PATH_REASON_TEACHER_UNAVAILABLE",
    "TRADE_PATH_REASON_UNFILLED_DATA_END",
    "TRADE_PATH_REASON_UNFILLED_SUPERSEDED",
    "TRADE_PATH_REASON_UNFILLED_TERMINATED",
    "TradePathLabelResult",
    "build_signal_cache",
    "build_trade_path_excluded_event_update",
    "load_active_param_schedule",
    "merge_active_param_schedules",
    "simulate_realized_trade_path_label",
]
