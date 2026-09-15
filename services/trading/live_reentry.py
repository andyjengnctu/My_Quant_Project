"""Broker-truth Trading re-entry adapter over canonical strategy primitives.

Research/portfolio replay creates a re-entry watch only after an actual strategy STOP.
Trading must therefore never consume a STOP simulated inside the single-stock scanner as
live truth.  This module deterministically reconstructs the same per-member watch from
broker-confirmed STOP fills plus the immutable agreeing-voter Params frozen on the entry
order, then delegates all watch/signal/shadow semantics to the canonical core helpers.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.breakout_reentry import (
    BREAKOUT_REENTRY_SOURCE,
    create_breakout_reentry_signal_state,
    create_breakout_reentry_watch_state,
    is_breakout_reentry_enabled,
)
from core.capital_policy import resolve_scanner_live_capital
from core.extended_signals import (
    build_extended_candidate_plan_from_signal,
    build_extended_tbd_candidate_plan_from_state,
    has_extended_signal_orderable_context_for_day,
    has_extended_tbd_orderable_context_for_day,
    is_extended_tbd_display_day,
    is_extended_tbd_shadow_alive,
    should_clear_extended_signal,
)
from core.params_io import build_params_from_mapping
from core.portfolio_ensemble import aggregate_ensemble_candidate_rows, annotate_ensemble_candidate
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.signal_utils import generate_signals
from core.trading_order_state import (
    TRADING_ORDER_PURPOSE_ENTRY,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
    TRADING_ORDER_SIDE_BUY,
    TRADING_ORDER_SIDE_SELL,
)
from core.runtime_utils import is_insufficient_data_error
from core.data_utils import get_required_min_rows
from core.backtest_core import run_v16_backtest
from services.scanner.stock_processor import build_extended_scanner_row_from_plan
from services.trading.account_state import load_trading_account_state
from services.trading.market_data_consumer import load_trading_v2_sanitized_ohlcv_frame
from services.trading.market_data_v2_view import TradingMarketDataV2View
from services.trading.order_state import load_trading_order_state


_STOP_PURPOSES = {
    TRADING_ORDER_PURPOSE_PROTECTION_STOP,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
}


def _event_key(*, trade_date: object, confirmed_at: object) -> tuple[str, str]:
    return (str(trade_date or ""), str(confirmed_at or ""))


def _entry_voter_lineage(entry_order: Mapping[str, Any]) -> dict[str, Any]:
    raw_map = entry_order.get("ensemble_member_params_by_key") or {}
    member_count = int(entry_order.get("ensemble_member_count") or 1)
    representative_key = str(entry_order.get("ensemble_member_key") or "1").strip() or "1"
    if isinstance(raw_map, Mapping) and raw_map:
        member_params = {str(key): dict(value) for key, value in raw_map.items() if isinstance(value, Mapping)}
    else:
        frozen = entry_order.get("frozen_params")
        if member_count > 1:
            raise RuntimeError(
                "Trading legacy ensemble entry 缺少 agreeing-voter frozen Params；"
                f"entry_order_id={entry_order.get('order_id') or '-'}，禁止推測 live re-entry"
            )
        if not isinstance(frozen, Mapping):
            raise RuntimeError("Trading entry order 缺少 frozen Params，無法建立 live re-entry")
        member_params = {representative_key: dict(frozen)}

    member_keys = [str(key) for key in list(entry_order.get("ensemble_member_keys") or [])]
    if not member_keys:
        member_keys = sorted(member_params)
    missing = sorted(set(member_keys) - set(member_params))
    if missing:
        raise RuntimeError(f"Trading entry order voter Params lineage 不完整: missing={missing}")
    min_agree = int(entry_order.get("ensemble_min_agree") or 1)
    if min_agree < 1 or min_agree > len(member_keys):
        raise RuntimeError("Trading entry order re-entry min_agree lineage 不合法")
    return {
        "member_keys": member_keys,
        "member_params_by_key": member_params,
        "member_quality_rank_by_key": deepcopy(entry_order.get("ensemble_member_quality_rank_by_key") or {}),
        "min_agree": min_agree,
        "use_breakout_quality_ranking": bool(entry_order.get("use_breakout_quality_ranking", False)),
        "breakout_quality_ranking_policy": entry_order.get("breakout_quality_ranking_policy"),
        "breakout_quality_ranking_options": deepcopy(entry_order.get("breakout_quality_ranking_options") or {}),
    }


def _latest_buy_fill_keys(orders: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    latest: dict[str, tuple[str, str]] = {}
    for record in (orders.get("orders") or {}).values():
        if not isinstance(record, Mapping):
            continue
        if str(record.get("side") or "") != TRADING_ORDER_SIDE_BUY or str(record.get("purpose") or "") != TRADING_ORDER_PURPOSE_ENTRY:
            continue
        ticker = str(record.get("ticker") or "").strip()
        for fill in list(record.get("fills") or []):
            key = _event_key(trade_date=fill.get("trade_date"), confirmed_at=fill.get("confirmed_at"))
            if ticker and key > latest.get(ticker, ("", "")):
                latest[ticker] = key
    return latest


def build_trading_live_reentry_watch_records(project_root: str | Path) -> list[dict[str, Any]]:
    """Derive active live re-entry obligations exclusively from broker-confirmed STOP fills."""

    root = Path(project_root).resolve()
    order_state = load_trading_order_state(root, required=False)
    if not order_state:
        return []
    orders = order_state.get("orders") or {}
    account = load_trading_account_state(root, required=False) or {}
    held_tickers = {
        str(ticker)
        for ticker, position in (account.get("positions") or {}).items()
        if int(((position or {}).get("broker") or {}).get("qty") or 0) > 0
    }
    latest_buy = _latest_buy_fill_keys(order_state)

    latest_stop_by_ticker: dict[str, dict[str, Any]] = {}
    for sell_order in orders.values():
        if not isinstance(sell_order, Mapping):
            continue
        if str(sell_order.get("side") or "") != TRADING_ORDER_SIDE_SELL or str(sell_order.get("purpose") or "") not in _STOP_PURPOSES:
            continue
        ticker = str(sell_order.get("ticker") or "").strip()
        entry_order_id = str(sell_order.get("entry_order_id") or "").strip()
        entry_order = orders.get(entry_order_id)
        if not ticker or not isinstance(entry_order, Mapping):
            continue
        sell_fills = list(sell_order.get("fills") or [])
        first_stop_snapshot = next(
            (fill.get("strategy_position_before_fill") for fill in sell_fills if isinstance(fill.get("strategy_position_before_fill"), Mapping)),
            None,
        )
        for fill in sell_fills:
            position_qty_after_fill = fill.get("position_qty_after_fill")
            if position_qty_after_fill is None or int(position_qty_after_fill) != 0:
                continue
            position_snapshot = first_stop_snapshot or fill.get("strategy_position_before_fill")
            if not isinstance(position_snapshot, Mapping):
                continue
            key = _event_key(trade_date=fill.get("trade_date"), confirmed_at=fill.get("confirmed_at"))
            if latest_buy.get(ticker, ("", "")) > key:
                continue
            current = latest_stop_by_ticker.get(ticker)
            if current is not None and tuple(current["event_key"]) >= key:
                continue
            latest_stop_by_ticker[ticker] = {
                "ticker": ticker,
                "entry_order_id": entry_order_id,
                "stop_order_id": str(sell_order.get("order_id") or ""),
                "stop_fill_id": str(fill.get("fill_id") or ""),
                "stop_trade_date": str(fill.get("trade_date") or ""),
                "stop_confirmed_at": str(fill.get("confirmed_at") or ""),
                "exit_qty": int(sell_order.get("qty") or fill.get("position_qty_before_fill") or fill.get("qty") or 0),
                "position_snapshot": deepcopy(dict(position_snapshot)),
                "entry_order": deepcopy(dict(entry_order)),
                "event_key": key,
            }

    records: list[dict[str, Any]] = []
    for ticker in sorted(latest_stop_by_ticker):
        if ticker in held_tickers:
            continue
        record = latest_stop_by_ticker[ticker]
        lineage = _entry_voter_lineage(record["entry_order"])
        records.append({**record, **lineage})
    return records


def resolve_trading_live_reentry_required_tickers(project_root: str | Path) -> list[str]:
    return sorted({str(record["ticker"]) for record in build_trading_live_reentry_watch_records(project_root)})


def _replay_member_reentry_signal(
    *,
    frame: pd.DataFrame,
    ticker: str,
    information_date: str,
    stop_record: Mapping[str, Any],
    params,
    quality_rank: Mapping[str, Any] | None,
):
    atr, buy_condition, sell_condition, _buy_limits = generate_signals(frame, params, ticker=ticker)
    dates = pd.DatetimeIndex(frame.index)
    stop_date = pd.Timestamp(str(stop_record["stop_trade_date"]))
    info_date = pd.Timestamp(str(information_date))
    matches = np.flatnonzero(dates == stop_date)
    if len(matches) != 1:
        raise RuntimeError(f"Trading live re-entry STOP 日期不在 V2 frame: ticker={ticker}, date={stop_date.date()}")
    stop_idx = int(matches[0])
    info_matches = np.flatnonzero(dates <= info_date)
    if len(info_matches) == 0:
        return None, None, None
    info_idx = int(info_matches[-1])
    if info_idx < stop_idx or stop_idx <= 0:
        return None, None, None

    watch = create_breakout_reentry_watch_state(
        deepcopy(dict(stop_record["position_snapshot"])),
        exit_date=stop_date,
        params=params,
        exit_atr=atr[stop_idx - 1],
        exit_qty=int(stop_record.get("exit_qty") or 0),
        quality_rank=None if quality_rank is None else dict(quality_rank),
    )
    if watch is None:
        return None, None, None
    watch["source_entry_order_id"] = str(stop_record["entry_order_id"])

    signal = None
    signal_idx = None
    sizing_capital = resolve_scanner_live_capital(params)
    for idx in range(stop_idx, info_idx + 1):
        # A canonical normal setup supersedes a pending re-entry watch for this member.
        if bool(buy_condition[idx]):
            return None, None, None
        if signal is None:
            watch["last_checked_date"] = dates[idx].strftime("%Y-%m-%d")
            watch["bars_checked"] = int(watch.get("bars_checked", 0) or 0) + 1
            signal = create_breakout_reentry_signal_state(
                watch,
                close_price=float(frame["Close"].iloc[idx]),
                atr=atr[idx],
                params=params,
                ticker=ticker,
                security_profile=frame.attrs.get("security_profile"),
                signal_date=dates[idx],
            )
            if signal is not None:
                signal["source_entry_order_id"] = str(stop_record["entry_order_id"])
                signal_idx = idx
            elif int(watch.get("bars_checked", 0) or 0) >= int(watch.get("window_bars", 0) or 0):
                return None, None, None
            continue

        if idx <= int(signal_idx):
            continue
        if should_clear_extended_signal(
            signal,
            float(frame["Low"].iloc[idx]),
            float(frame["High"].iloc[idx]),
            t_open=float(frame["Open"].iloc[idx]),
            t_close=float(frame["Close"].iloc[idx]),
            t_volume=float(frame["Volume"].iloc[idx]),
            y_close=float(frame["Close"].iloc[idx - 1]),
            y_high=float(frame["High"].iloc[idx - 1]),
            y_atr=atr[idx - 1],
            y_ind_sell=bool(sell_condition[idx - 1]),
            sizing_capital=sizing_capital,
            current_date=dates[idx],
            params=params,
        ):
            return None, None, None

    if signal is None:
        return None, None, None
    low_last = float(frame["Low"].iloc[info_idx])
    close_last = float(frame["Close"].iloc[info_idx])
    tbd = bool(is_extended_tbd_display_day(signal, low_last) and is_extended_tbd_shadow_alive(signal))
    if tbd:
        plan = build_extended_tbd_candidate_plan_from_state(
            signal,
            sizing_capital,
            params,
            ticker=ticker,
            security_profile=frame.attrs.get("security_profile"),
            trade_date=dates[info_idx],
        )
        orderable = has_extended_tbd_orderable_context_for_day(
            signal,
            plan,
            close_last,
            ticker=ticker,
            security_profile=frame.attrs.get("security_profile"),
        )
    else:
        plan = build_extended_candidate_plan_from_signal(
            signal,
            sizing_capital,
            params,
            ticker=ticker,
            security_profile=frame.attrs.get("security_profile"),
            trade_date=dates[info_idx],
        )
        orderable = has_extended_signal_orderable_context_for_day(
            signal,
            plan,
            close_last,
            ticker=ticker,
            security_profile=frame.attrs.get("security_profile"),
        )
    return signal, plan, bool(orderable)


def build_trading_live_reentry_candidate_rows(
    project_root: str | Path,
    *,
    information_date: str,
    excluded_tickers: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build broker-truth re-entry candidates and re-apply the original entry ensemble consensus."""

    root = Path(project_root).resolve()
    excluded = {str(item) for item in (excluded_tickers or set())}
    records = build_trading_live_reentry_watch_records(root)
    if not records:
        return []
    view = TradingMarketDataV2View.open_as_of_target(
        root, target_date=str(information_date)
    )
    aggregated_rows: list[dict[str, Any]] = []

    for record in records:
        ticker = str(record["ticker"])
        if ticker in excluded or str(record["stop_trade_date"]) > str(information_date):
            continue
        member_params: dict[str, Any] = {}
        for member_key in record["member_keys"]:
            payload = record["member_params_by_key"].get(member_key)
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"Trading live re-entry 缺 member Params: ticker={ticker}, member={member_key}")
            member_params[member_key] = build_params_from_mapping(dict(payload))
        if not any(is_breakout_reentry_enabled(params) for params in member_params.values()):
            continue
        min_rows = max(get_required_min_rows(params) for params in member_params.values())
        try:
            frame = load_trading_v2_sanitized_ohlcv_frame(
                view,
                ticker=ticker,
                through_date=str(information_date),
                min_rows=min_rows,
            )
        except (ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            if is_insufficient_data_error(exc):
                continue
            raise

        member_rows: list[dict[str, Any]] = []
        for member_key in record["member_keys"]:
            params = member_params[member_key]
            if not is_breakout_reentry_enabled(params):
                continue
            quality_rank = (record.get("member_quality_rank_by_key") or {}).get(member_key)
            signal, plan, orderable = _replay_member_reentry_signal(
                frame=frame,
                ticker=ticker,
                information_date=str(information_date),
                stop_record=record,
                params=params,
                quality_rank=quality_rank,
            )
            if signal is None or plan is None:
                continue
            stats = run_v16_backtest(frame, params, ticker=ticker)
            row = build_extended_scanner_row_from_plan(
                ticker=ticker,
                stats=stats,
                params=params,
                trade_date=str(information_date),
                candidate_plan=plan,
                orderable_today=orderable,
                label_prefix="Re-entry",
                kind_if_orderable="reentry",
            )
            if row is None or str(row.get("kind") or "") == "candidate":
                continue
            row["param_lineage_source"] = "entry_order_frozen_ensemble"
            row["source_entry_order_id"] = str(record["entry_order_id"])
            row["use_breakout_quality_ranking"] = bool(record.get("use_breakout_quality_ranking", False))
            row["breakout_quality_ranking_policy"] = record.get("breakout_quality_ranking_policy")
            row["breakout_quality_ranking_options"] = deepcopy(record.get("breakout_quality_ranking_options") or {})
            if isinstance(quality_rank, Mapping):
                row["breakout_quality_rank"] = dict(quality_rank)
                row["breakout_quality_score"] = quality_rank.get("score")
                row["breakout_quality_score_date"] = quality_rank.get("score_date")
                row["breakout_quality_score_source"] = quality_rank.get("score_source")
            member = {
                "member_key": member_key,
                "params_obj": params,
                "params_signature": build_portfolio_params_signature(params),
            }
            member_rows.append(
                annotate_ensemble_candidate(
                    row,
                    member=member,
                    params_obj=params,
                    member_key=member_key,
                )
            )

        agreed = aggregate_ensemble_candidate_rows(member_rows, min_agree=int(record["min_agree"]))
        for row in agreed:
            row["param_lineage_source"] = "entry_order_frozen_ensemble"
            row["source_entry_order_id"] = str(record["entry_order_id"])
        aggregated_rows.extend(agreed)
    return aggregated_rows


__all__ = [
    "build_trading_live_reentry_watch_records",
    "resolve_trading_live_reentry_required_tickers",
    "build_trading_live_reentry_candidate_rows",
]
