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
from filters.breakout_quality.contract import LABEL_INVALID, LABEL_PASS, LABEL_REJECT

TRADE_PATH_BASE_FILTER_ID = "breakout_quality_v1"
TRADE_PATH_RESEARCH_FILTER_ID = "breakout_quality_a2_trade_path_v1"
TRADE_PATH_LABEL_ID = "a2_realized_trade_path_v1"
TRADE_PATH_SELECTION_BASELINE_PARAMS_RELATIVE_PATH = Path(
    "models/research/breakout_quality/selection_strategy_realization/roos_base_best.json"
)
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
    reason: str
    label_eval_start_date: str | None
    label_eval_end_date: str | None
    teacher_effective_date: str | None
    fill_type: str | None
    fill_date: str | None
    exit_date: str | None
    exit_reason: str | None
    continuation_wait_bars: int | None
    realized_net_pnl: float | None
    realized_net_r: float | None

    def as_event_update(self) -> dict[str, Any]:
        return {
            "label": int(self.label),
            "label_reason": str(self.reason),
            "label_eval_start_date": self.label_eval_start_date,
            "label_eval_end_date": self.label_eval_end_date,
            "teacher_effective_date": self.teacher_effective_date,
            "trade_path_fill_type": self.fill_type,
            "trade_path_fill_date": self.fill_date,
            "trade_path_exit_date": self.exit_date,
            "trade_path_exit_reason": self.exit_reason,
            "trade_path_continuation_wait_bars": self.continuation_wait_bars,
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


def _invalid_result(
    reason: str,
    *,
    start_date: str | None,
    end_date: str | None,
    teacher_effective_date: str | None,
    fill_type: str | None = None,
    fill_date: str | None = None,
    wait_bars: int | None = None,
) -> TradePathLabelResult:
    return TradePathLabelResult(
        label=LABEL_INVALID,
        reason=reason,
        label_eval_start_date=start_date,
        label_eval_end_date=end_date,
        teacher_effective_date=teacher_effective_date,
        fill_type=fill_type,
        fill_date=fill_date,
        exit_date=None,
        exit_reason=None,
        continuation_wait_bars=wait_bars,
        realized_net_pnl=None,
        realized_net_r=None,
    )


def simulate_realized_trade_path_label(
    stock_df: pd.DataFrame,
    *,
    ticker: str,
    signal_pos: int,
    params: V16StrategyParams,
    teacher_effective_date: str,
    precomputed_signals,
) -> TradePathLabelResult:
    if signal_pos < 0 or signal_pos >= len(stock_df) - 1:
        return _invalid_result(
            "insufficient_future",
            start_date=None,
            end_date=None,
            teacher_effective_date=teacher_effective_date,
        )
    atr, buy_condition, sell_condition, buy_limits = precomputed_signals
    signal_date = stock_df.index[signal_pos]
    eval_start_date = _date_text(stock_df.index[signal_pos + 1])
    if not bool(buy_condition[signal_pos]) or not math.isfinite(float(atr[signal_pos])):
        return _invalid_result(
            "not_a2_formal_setup",
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
        return _invalid_result(
            "invalid_entry_plan",
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
    sizing_capital = resolve_single_backtest_sizing_capital(params, params.initial_capital)
    active_signal = signal_state
    position: dict[str, Any] = {"qty": 0}
    fill_type: str | None = None
    fill_date: str | None = None
    wait_bars: int | None = None

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
                record_exec_contexts=False,
                sync_display_fields=False,
            )
            terminal_events = [value for value in ("STOP", "IND_SELL") if value in events]
            if terminal_events:
                realized_pnl_milli = int(position.get("realized_pnl_milli", 0) or 0)
                initial_risk_milli = int(position.get("initial_risk_total_milli", 0) or 0)
                realized_r = calc_ratio_from_milli(realized_pnl_milli, initial_risk_milli)
                label = LABEL_PASS if realized_pnl_milli > 0 else LABEL_REJECT
                exit_date = _date_text(dates[current_pos])
                return TradePathLabelResult(
                    label=label,
                    reason=("realized_net_profit" if label == LABEL_PASS else "realized_net_nonpositive"),
                    label_eval_start_date=eval_start_date,
                    label_eval_end_date=exit_date,
                    teacher_effective_date=teacher_effective_date,
                    fill_type=fill_type,
                    fill_date=fill_date,
                    exit_date=exit_date,
                    exit_reason=terminal_events[0],
                    continuation_wait_bars=wait_bars,
                    realized_net_pnl=milli_to_money(realized_pnl_milli),
                    realized_net_r=float(realized_r),
                )
            continue

        if current_pos > signal_pos + 1 and bool(buy_condition[current_pos - 1]):
            return _invalid_result(
                "unfilled_superseded_by_new_setup",
                start_date=eval_start_date,
                end_date=_date_text(dates[current_pos - 1]),
                teacher_effective_date=teacher_effective_date,
                wait_bars=current_pos - signal_pos - 1,
            )

        if current_pos == signal_pos + 1:
            entry_plan = build_normal_entry_plan(
                buy_limits[signal_pos],
                atr[signal_pos],
                sizing_capital,
                params,
                ticker=ticker,
                security_profile=stock_df.attrs.get("security_profile"),
                trade_date=dates[current_pos],
            )
            entry_type = "normal"
        else:
            entry_plan = build_extended_entry_plan_from_signal(
                active_signal,
                sizing_capital,
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
        if bool(entry_result.get("filled")):
            position = dict(entry_result["position"])
            fill_type = "INITIAL_FILL" if current_pos == signal_pos + 1 else "CONTINUATION_FILL"
            fill_date = _date_text(dates[current_pos])
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
            sizing_capital=sizing_capital,
            current_date=dates[current_pos],
            params=params,
            copy_shadow_position=False,
        ):
            return _invalid_result(
                "unfilled_terminated",
                start_date=eval_start_date,
                end_date=_date_text(dates[current_pos]),
                teacher_effective_date=teacher_effective_date,
                wait_bars=current_pos - signal_pos,
            )

    return _invalid_result(
        "insufficient_future_after_fill" if fill_date else "unfilled_censored_at_data_end",
        start_date=eval_start_date,
        end_date=_date_text(dates[-1]),
        teacher_effective_date=teacher_effective_date,
        fill_type=fill_type,
        fill_date=fill_date,
        wait_bars=wait_bars,
    )


__all__ = [
    "ActiveParamSchedule",
    "TRADE_PATH_BASE_FILTER_ID",
    "TRADE_PATH_FORWARD_TEACHER_PARAMS_RELATIVE_PATH",
    "TRADE_PATH_HISTORICAL_TEACHER_PARAMS_RELATIVE_PATH",
    "TRADE_PATH_HISTORICAL_TEACHER_RELATIVE_DIR",
    "TRADE_PATH_LABEL_ID",
    "TRADE_PATH_RESEARCH_FILTER_ID",
    "TRADE_PATH_SELECTION_BASELINE_PARAMS_RELATIVE_PATH",
    "TradePathLabelResult",
    "build_signal_cache",
    "load_active_param_schedule",
    "merge_active_param_schedules",
    "simulate_realized_trade_path_label",
]
