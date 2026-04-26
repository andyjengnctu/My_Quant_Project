from __future__ import annotations

import io
import os
import re
import threading
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from tkinter import font as tkfont
from tkinter import messagebox, ttk
import tkinter as tk

import numpy as np
import pandas as pd

from core.buy_sort import calc_buy_sort_value, format_buy_sort_metric_value, get_buy_sort_metric_label, get_buy_sort_method
from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.model_paths import resolve_candidate_best_params_path, resolve_run_best_params_path
from core.entry_plans import build_position_from_entry_fill
from core.portfolio_fast_data import get_fast_close, get_fast_dates, get_fast_pos, get_fast_value
from core.position_step import execute_bar_step
from core.runtime_utils import parse_float_strict, parse_int_strict
from core.walk_forward_policy import load_walk_forward_policy
from tools.portfolio_sim.reporting import export_portfolio_reports, print_yearly_return_report
from tools.portfolio_sim.runtime import ensure_runtime_dirs, load_strict_params
from tools.portfolio_sim.simulation_runner import (
    PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
    load_portfolio_market_context,
    run_portfolio_simulation_prepared,
)
from tools.trade_analysis.charting import (
    bind_matplotlib_chart_navigation,
    build_chart_hover_snapshot,
    build_debug_chart_payload,
    create_debug_chart_context,
    create_matplotlib_trade_chart_figure,
    extract_trade_marker_indexes,
    record_active_levels,
    record_signal_annotation,
    record_trade_marker,
    scroll_chart_to_adjacent_trade,
    scroll_chart_to_latest,
)

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError as exc:  # pragma: no cover - GUI runtime fallback
    FigureCanvasTkAgg = None
    FIGURE_CANVAS_TKAGG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    FIGURE_CANVAS_TKAGG_IMPORT_ERROR = ""


WORKBENCH_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARAM_SOURCE_LABEL_TO_KEY = {
    "run_best | 目前參數": "run_best",
    "candidate_best | 候選參數": "candidate_best",
}
DEFAULT_PARAM_SOURCE_LABEL = "run_best | 目前參數"
ROTATION_LABEL_TO_BOOL = {
    "關閉 (穩定鎖倉)": False,
    "啟用 (強勢輪動)": True,
}
DEFAULT_ROTATION_LABEL = "關閉 (穩定鎖倉)"
FIXED_RISK_LABELS = ("0.01", "0.02", "自訂")
END_YEAR_LATEST_LABEL = "最新"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
PORTFOLIO_CONSOLE_COLORS = {
    "91": "#ff6174",
    "93": "#facc15",
    "96": "#4fd1ff",
    "92": "#5ee28a",
    "90": "#9aa7b6",
    "94": "#7fb3ff",
}


BUY_TRADE_TRACE_NAMES = ("買進", "買進(延續候選)")
SIDEBAR_SIGNAL_CHIP_TEXT = "出現買入訊號"
SIDEBAR_HISTORY_CHIP_TEXT = "符合歷史績效"
SIDEBAR_CHIP_ACTIVE_BG = "#2090ff"
SIDEBAR_HISTORY_CHIP_ACTIVE_BG = "#ff8a1c"
SIDEBAR_CHIP_INACTIVE_BG = "#04070c"
PERFORMANCE_STRATEGY_COLOR = "#ff3333"
PERFORMANCE_BENCHMARK_COLOR = "#4dabf5"
COMBOBOX_WIDTH_RULES = {
    "param_source": {"min_chars": 18, "max_chars": 24, "extra_px": 32},
    "rotation": {"min_chars": 12, "max_chars": 18, "extra_px": 30},
    "start_year": {"min_chars": 7, "max_chars": 8, "extra_px": 24},
    "end_year": {"min_chars": 7, "max_chars": 8, "extra_px": 24},
    "risk": {"min_chars": 6, "max_chars": 7, "extra_px": 22},
    "ticker": {"min_chars": 18, "max_chars": 44, "extra_px": 24},
}
PORTFOLIO_DROPDOWN_KIND_LABELS = {
    "normal": "新訊號",
    "extended": "延續",
    "extended_shadow": "延續",
}
PORTFOLIO_DROPDOWN_SORT_LABELS = {
    "EV": "EV",
    "預估投入": "投入",
    "勝率×次數": "勝×次",
    "資產成長": "成長",
}
PORTFOLIO_BUY_EV_PATTERN = re.compile(r"EV:([+-]?\d+(?:\.\d+)?)R")


def _warn_gui_fallback(action, exc):
    warnings.warn(f"GUI fallback {action}: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)


class _PortfolioConsoleWriter(io.TextIOBase):
    def __init__(self, panel: "PortfolioBacktestInspectorPanel"):
        super().__init__()
        self._panel = panel

    def write(self, text):
        if not text:
            return 0
        self._panel._append_console_stream(str(text))
        return len(text)

    def flush(self):
        return None



def _resolve_default_portfolio_start_year_hint():
    policy = load_walk_forward_policy(WORKBENCH_PROJECT_ROOT)
    return int(policy["search_train_end_year"]) + 1


def _coerce_float(value, default=np.nan):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _coerce_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _format_pct(value):
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _fast_data_to_price_df(fast_data):
    dates = pd.DatetimeIndex(pd.to_datetime(get_fast_dates(fast_data)))
    return pd.DataFrame(
        {
            "Open": np.asarray(fast_data["Open"], dtype=float),
            "High": np.asarray(fast_data["High"], dtype=float),
            "Low": np.asarray(fast_data["Low"], dtype=float),
            "Close": np.asarray(fast_data["Close"], dtype=float),
            "Volume": np.asarray(fast_data["Volume"], dtype=float),
        },
        index=dates,
    )


def _normalize_trade_action(row):
    raw_type = str(row.get("Type", "") or "").strip()
    entry_type = str(row.get("進場類型", "") or "").strip()
    if raw_type.startswith("買進"):
        return "買進(延續候選)" if entry_type == "extended" else "買進"
    if raw_type == "半倉停利":
        return "半倉停利"
    if raw_type == "全倉結算(停損)":
        return "停損殺出"
    if raw_type == "全倉結算(指標)":
        return "指標賣出"
    if raw_type.startswith("汰弱賣出"):
        return "指標賣出"
    if raw_type == "期末強制結算":
        return "期末強制結算"
    if raw_type == "錯失賣出":
        return "錯失賣出"
    return ""


def _is_actual_trade_row(row):
    action = _normalize_trade_action(row)
    return bool(action) and not str(row.get("Type", "") or "").startswith("錯失")


def _resolve_marker_price(price_df, row, action):
    price = _coerce_float(row.get("成交價"))
    if not pd.isna(price) and price > 0:
        return price
    date = pd.Timestamp(row.get("Date"))
    if date not in price_df.index:
        return np.nan
    if action in {"買進", "買進(延續候選)"}:
        return float(price_df.loc[date, "Low"])
    if action == "半倉停利":
        return float(price_df.loc[date, "High"])
    return float(price_df.loc[date, "Close"])




def _is_buy_trade_row(row):
    return _normalize_trade_action(row) in {"買進", "買進(延續候選)"}


def _is_full_exit_trade_row(row):
    return _normalize_trade_action(row) in {"停損殺出", "指標賣出", "期末強制結算"}


def _resolve_valid_float(value):
    numeric = _coerce_float(value)
    if pd.isna(numeric):
        return None
    return float(numeric)


def _resolve_valid_int(value):
    numeric = _coerce_int(value, default=0)
    return int(numeric) if numeric > 0 else None


def _safe_fast_pos(fast_data, date_value):
    try:
        return int(get_fast_pos(fast_data, pd.Timestamp(date_value)))
    except (TypeError, ValueError, KeyError):
        return -1


def _safe_fast_value(fast_data, field, *, pos):
    try:
        return _coerce_float(get_fast_value(fast_data, field, pos=pos))
    except (TypeError, ValueError, KeyError, IndexError):
        return np.nan


def _safe_fast_close(fast_data, *, pos):
    try:
        return _coerce_float(get_fast_close(fast_data, pos=pos))
    except (TypeError, ValueError, KeyError, IndexError):
        return np.nan


def _resolve_buy_limit_from_row(row, fast_data):
    explicit_limit = _resolve_valid_float(row.get("買入限價"))
    if explicit_limit is not None:
        return explicit_limit
    buy_date = pd.Timestamp(row.get("Date"))
    buy_pos = _safe_fast_pos(fast_data, buy_date)
    if buy_pos > 0:
        fast_limit = _resolve_valid_float(_safe_fast_value(fast_data, "buy_limit", pos=buy_pos - 1))
        if fast_limit is not None:
            return fast_limit
    entry_price = _resolve_valid_float(row.get("成交價"))
    return np.nan if entry_price is None else entry_price


def _calc_pct_from_capital(pnl_value, capital_value):
    pnl = _coerce_float(pnl_value)
    capital = _coerce_float(capital_value)
    if pd.isna(pnl) or pd.isna(capital) or capital <= 0:
        return None
    return float(pnl) * 100.0 / float(capital)


def _estimate_sell_capital(row):
    explicit_total = _resolve_valid_float(row.get("賣出總金額"))
    if explicit_total is not None:
        return explicit_total
    price = _resolve_valid_float(row.get("成交價"))
    qty = _resolve_valid_int(row.get("股數"))
    if price is None or qty is None:
        return None
    return float(price) * int(qty)


def _resolve_previous_trade_date(fast_data, trade_date):
    pos = _safe_fast_pos(fast_data, trade_date)
    if pos <= 0:
        return pd.Timestamp(trade_date)
    dates = pd.DatetimeIndex(pd.to_datetime(get_fast_dates(fast_data)))
    if pos >= len(dates):
        return pd.Timestamp(trade_date)
    return pd.Timestamp(dates[pos - 1])


def _apply_portfolio_stats_to_marker_meta(meta, actual_stats):
    enriched = dict(meta or {})
    trade_count = _coerce_int((actual_stats or {}).get("trade_count"), default=0)
    win_rate_pct = (actual_stats or {}).get("win_rate_pct")
    if trade_count > 0:
        enriched["trade_count"] = int(trade_count)
    if win_rate_pct is not None and not pd.isna(win_rate_pct):
        enriched["win_rate"] = float(win_rate_pct)
    return enriched


def _build_portfolio_ticker_actual_stats(df_tr, ticker):
    if df_tr is None or df_tr.empty or "Ticker" not in df_tr.columns:
        return {
            "buy_count": 0,
            "exit_count": 0,
            "event_count": 0,
            "win_count": 0,
            "win_rate_pct": None,
            "trade_count": 0,
            "total_buy_capital": 0.0,
            "total_pnl": 0.0,
            "asset_growth_pct": 0.0,
        }

    ticker_rows = df_tr[df_tr["Ticker"].astype(str) == str(ticker)].copy()
    actual_rows = ticker_rows[ticker_rows.apply(_is_actual_trade_row, axis=1)].copy() if not ticker_rows.empty else pd.DataFrame()
    if actual_rows.empty:
        return {
            "buy_count": 0,
            "exit_count": 0,
            "event_count": 0,
            "win_count": 0,
            "win_rate_pct": None,
            "trade_count": 0,
            "total_buy_capital": 0.0,
            "total_pnl": 0.0,
            "asset_growth_pct": 0.0,
        }

    records = actual_rows.to_dict("records")
    buy_records = [row for row in records if _is_buy_trade_row(row)]
    exit_records = [row for row in records if _is_full_exit_trade_row(row)]
    win_count = sum(1 for row in exit_records if _coerce_float(row.get("該筆總損益"), default=0.0) > 0)
    exit_count = len(exit_records)
    win_rate_pct = None if exit_count <= 0 else float(win_count) * 100.0 / float(exit_count)
    total_buy_capital = sum(_coerce_float(row.get("投入總金額"), default=0.0) for row in buy_records)
    total_pnl = sum(_coerce_float(row.get("該筆總損益"), default=0.0) for row in exit_records)
    asset_growth_pct = 0.0 if total_buy_capital <= 0 else float(total_pnl) * 100.0 / float(total_buy_capital)
    return {
        "buy_count": int(len(buy_records)),
        "exit_count": int(exit_count),
        "event_count": int(len(records)),
        "win_count": int(win_count),
        "win_rate_pct": win_rate_pct,
        "trade_count": int(exit_count),
        "total_buy_capital": float(total_buy_capital),
        "total_pnl": float(total_pnl),
        "asset_growth_pct": float(asset_growth_pct),
    }


def _build_portfolio_buy_marker_meta(row, *, fast_data, params):
    buy_capital = _coerce_float(row.get("投入總金額"), default=np.nan)
    meta = {
        "buy_capital": None if pd.isna(buy_capital) else float(buy_capital),
    }
    position = _build_position_from_portfolio_buy_row(row, fast_data=fast_data, params=params)
    if position is None:
        limit_price = _resolve_buy_limit_from_row(row, fast_data)
        entry_price = _resolve_valid_float(row.get("成交價"))
        if not pd.isna(limit_price):
            meta["limit_price"] = float(limit_price)
        if entry_price is not None:
            meta["entry_price"] = float(entry_price)
        return meta

    for source_key, target_key in (
        ("limit_price", "limit_price"),
        ("pure_buy_price", "entry_price"),
        ("initial_stop", "stop_price"),
        ("tp_half", "tp_price"),
    ):
        value = position.get(source_key)
        if value is not None and not pd.isna(value):
            meta[target_key] = float(value)

    entry_price = _resolve_valid_float(row.get("成交價"))
    if entry_price is not None:
        meta["entry_price"] = float(entry_price)
    return meta


def _build_portfolio_sell_marker_meta(row, active_entry, actual_stats=None):
    pnl_value = _resolve_valid_float(row.get("單筆損益"))
    total_pnl = _resolve_valid_float(row.get("該筆總損益"))
    sell_capital = _estimate_sell_capital(row)
    entry_capital = None if active_entry is None else active_entry.get("buy_capital")
    pnl_pct = _calc_pct_from_capital(total_pnl if total_pnl is not None else pnl_value, entry_capital)
    meta = {}
    if pnl_value is not None:
        meta["pnl_value"] = float(pnl_value)
    if total_pnl is not None:
        meta["total_pnl"] = float(total_pnl)
    if pnl_pct is not None:
        meta["pnl_pct"] = float(pnl_pct)
    if sell_capital is not None:
        meta["sell_capital"] = float(sell_capital)
    return _apply_portfolio_stats_to_marker_meta(meta, actual_stats or {})


def _extract_trade_marker_indexes(chart_payload):
    return extract_trade_marker_indexes(chart_payload, trace_names=BUY_TRADE_TRACE_NAMES)


def _build_position_from_portfolio_buy_row(row, *, fast_data, params):
    buy_date = pd.Timestamp(row.get("Date"))
    buy_pos = _safe_fast_pos(fast_data, buy_date)
    if buy_pos < 0:
        return None
    buy_price = _resolve_valid_float(row.get("成交價"))
    qty = _resolve_valid_int(row.get("股數"))
    if buy_price is None or qty is None:
        return None

    y_pos = buy_pos - 1
    entry_atr = _safe_fast_value(fast_data, "ATR", pos=y_pos) if y_pos >= 0 else np.nan
    limit_price = _resolve_buy_limit_from_row(row, fast_data)
    init_sl = _resolve_valid_float(row.get("初始停損價"))
    if init_sl is None:
        init_sl = _resolve_valid_float(row.get("停損價"))
    init_trail = _resolve_valid_float(row.get("初始移動停損價"))
    if init_trail is None:
        init_trail = init_sl
    target_price = _resolve_valid_float(row.get("半倉停利價"))
    try:
        return build_position_from_entry_fill(
            buy_price=buy_price,
            qty=qty,
            init_sl=init_sl,
            init_trail=init_trail,
            params=params,
            entry_type=str(row.get("進場類型", "normal") or "normal"),
            target_price=target_price,
            limit_price=limit_price,
            entry_atr=entry_atr,
            ticker=str(row.get("Ticker", "") or ""),
            security_profile=(fast_data or {}).get("security_profile"),
            trade_date=buy_date,
        )
    except (TypeError, ValueError, KeyError, RuntimeError):
        return None


def _record_portfolio_trade_annotations(chart_context, *, price_df, fast_data, row, action, marker_meta):
    trade_date = pd.Timestamp(row.get("Date"))
    if action in {"買進", "買進(延續候選)"}:
        signal_date = _resolve_previous_trade_date(fast_data, trade_date)
        if signal_date not in price_df.index:
            return
        limit_price = marker_meta.get("limit_price")
        reserved_capital = marker_meta.get("buy_capital")
        detail_lines = []
        qty = _coerce_int(row.get("股數"), default=0)
        if qty > 0:
            detail_lines.append(f"股數: {qty:,}")
        if limit_price is not None and not pd.isna(limit_price):
            detail_lines.append(f"限價: {float(limit_price):.2f}")
        if reserved_capital is not None and not pd.isna(reserved_capital):
            detail_lines.append(f"預留: {float(reserved_capital):,.0f}")
        record_signal_annotation(
            chart_context,
            current_date=signal_date,
            signal_type="buy",
            anchor_price=float(price_df.loc[signal_date, "Low"]),
            title="買訊",
            detail_lines=detail_lines,
            meta={
                "qty": qty,
                "reserved_capital": reserved_capital,
                "current_capital": None,
                "entry_price": marker_meta.get("entry_price"),
                "limit_price": limit_price,
            },
        )
        return
    if action == "指標賣出":
        signal_date = _resolve_previous_trade_date(fast_data, trade_date)
        if signal_date not in price_df.index:
            return
        record_signal_annotation(
            chart_context,
            current_date=signal_date,
            signal_type="sell",
            anchor_price=float(price_df.loc[signal_date, "Low"]),
            title="賣訊",
            detail_lines=[],
            meta={
                "profit_pct": marker_meta.get("pnl_pct"),
                "max_drawdown": marker_meta.get("max_drawdown", 0.0),
            },
        )


def _record_portfolio_active_level_segments(chart_context, *, fast_data, ticker_trades_df, params):
    buy_rows = [row for row in ticker_trades_df.to_dict("records") if _is_buy_trade_row(row)]
    if not buy_rows:
        return
    full_exit_rows = [row for row in ticker_trades_df.to_dict("records") if _is_full_exit_trade_row(row)]
    fast_dates = pd.DatetimeIndex(pd.to_datetime(get_fast_dates(fast_data)))

    for buy_idx, buy_row in enumerate(buy_rows):
        buy_date = pd.Timestamp(buy_row.get("Date"))
        buy_pos = _safe_fast_pos(fast_data, buy_date)
        if buy_pos < 0:
            continue
        next_buy_date = pd.Timestamp(buy_rows[buy_idx + 1].get("Date")) if buy_idx + 1 < len(buy_rows) else None
        exit_date = None
        for exit_row in full_exit_rows:
            candidate_exit_date = pd.Timestamp(exit_row.get("Date"))
            if candidate_exit_date < buy_date:
                continue
            if next_buy_date is not None and candidate_exit_date >= next_buy_date:
                continue
            exit_date = candidate_exit_date
            break

        end_pos = len(fast_dates) - 1 if exit_date is None else _safe_fast_pos(fast_data, exit_date)
        if end_pos < buy_pos:
            continue

        position = _build_position_from_portfolio_buy_row(buy_row, fast_data=fast_data, params=params)
        if position is None:
            continue

        for pos in range(buy_pos, end_pos + 1):
            current_date = fast_dates[pos]
            if pos > buy_pos:
                y_pos = pos - 1
                try:
                    position, _freed_cash, _pnl, _events = execute_bar_step(
                        position,
                        _safe_fast_value(fast_data, "ATR", pos=y_pos),
                        bool(_safe_fast_value(fast_data, "ind_sell_signal", pos=y_pos)),
                        _safe_fast_close(fast_data, pos=y_pos),
                        _safe_fast_value(fast_data, "Open", pos=pos),
                        _safe_fast_value(fast_data, "High", pos=pos),
                        _safe_fast_value(fast_data, "Low", pos=pos),
                        _safe_fast_close(fast_data, pos=pos),
                        _safe_fast_value(fast_data, "Volume", pos=pos),
                        params,
                        current_date=current_date,
                        y_high=_safe_fast_value(fast_data, "High", pos=y_pos),
                        return_milli=False,
                        record_exec_contexts=False,
                        sync_display_fields=True,
                    )
                except (TypeError, ValueError, KeyError, IndexError, RuntimeError):
                    break
            if int(position.get("qty", 0) or 0) <= 0:
                break
            record_active_levels(
                chart_context,
                current_date=current_date,
                stop_price=position.get("sl", np.nan),
                tp_half_price=position.get("tp_half", np.nan),
                limit_price=position.get("limit_price", np.nan),
                entry_price=position.get("pure_buy_price", np.nan),
            )
            if exit_date is not None and current_date >= exit_date:
                break


def _build_portfolio_ticker_chart_payload(*, ticker, fast_data, ticker_trades_df, params=None, ticker_dropdown_stats=None):
    price_df = _fast_data_to_price_df(fast_data)
    chart_context = create_debug_chart_context(price_df)
    dropdown_stats = dict(ticker_dropdown_stats or {})
    sort_text = str(dropdown_stats.get("sort_text") or "-")
    actual_stats = dict(dropdown_stats.get("portfolio_actual_stats") or _build_portfolio_ticker_actual_stats(ticker_trades_df, ticker))

    if params is not None:
        _record_portfolio_active_level_segments(
            chart_context,
            fast_data=fast_data,
            ticker_trades_df=ticker_trades_df,
            params=params,
        )

    active_entry = None
    sorted_records = sorted(
        ticker_trades_df.to_dict("records"),
        key=lambda row: (pd.Timestamp(row.get("Date")), 0 if _is_buy_trade_row(row) else 1),
    )
    for row in sorted_records:
        action = _normalize_trade_action(row)
        if not action:
            continue
        try:
            trade_date = pd.Timestamp(row.get("Date"))
        except (TypeError, ValueError):
            continue
        if trade_date not in price_df.index:
            continue
        marker_price = _resolve_marker_price(price_df, row, action)
        if pd.isna(marker_price):
            continue

        qty = _coerce_int(row.get("股數"), default=0)
        note = str(row.get("Type", "") or "").strip()
        if action in {"買進", "買進(延續候選)"}:
            marker_meta = _build_portfolio_buy_marker_meta(row, fast_data=fast_data, params=params)
            active_entry = {
                "buy_capital": marker_meta.get("buy_capital"),
                "qty": qty,
                "date": trade_date,
            }
        else:
            marker_meta = _build_portfolio_sell_marker_meta(row, active_entry, actual_stats)
            if action in {"停損殺出", "指標賣出", "期末強制結算"}:
                active_entry = None

        _record_portfolio_trade_annotations(
            chart_context,
            price_df=price_df,
            fast_data=fast_data,
            row=row,
            action=action,
            marker_meta=marker_meta,
        )
        record_trade_marker(
            chart_context,
            current_date=trade_date,
            action=action,
            price=marker_price,
            qty=qty,
            note=note,
            meta=marker_meta,
        )

    buy_count = int(actual_stats.get("buy_count", 0) or 0)
    exit_count = int(actual_stats.get("exit_count", 0) or 0)
    event_count = int(actual_stats.get("event_count", 0) or 0)
    total_pnl = float(actual_stats.get("total_pnl", 0.0) or 0.0)
    win_rate_pct = actual_stats.get("win_rate_pct")
    win_rate_text = "-" if win_rate_pct is None else f"{float(win_rate_pct):.1f}%"
    chart_context["summary_box"] = [
        "投組實際成交",
        f"{ticker} 買進 {buy_count} 次",
        f"完整出場 {exit_count} 次",
        f"交易事件 {event_count} 筆",
        f"投組勝率 {win_rate_text}",
        f"投組總損益 {total_pnl:+,.0f}",
        f"排序 {sort_text}",
    ]
    chart_context["status_box"] = {"lines": [SIDEBAR_SIGNAL_CHIP_TEXT, SIDEBAR_HISTORY_CHIP_TEXT], "ok": True}
    return build_debug_chart_payload(price_df, chart_context)



class PortfolioBacktestInspectorPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=4, style="Workbench.TFrame")
        self._ui_thread = threading.current_thread()
        self._console_writer = _PortfolioConsoleWriter(self)
        self._run_thread = None
        self._active_token = 0
        self._status_var = tk.StringVar(value="尚未執行")
        self._param_source_display_var = tk.StringVar(value=DEFAULT_PARAM_SOURCE_LABEL)
        self._rotation_display_var = tk.StringVar(value=DEFAULT_ROTATION_LABEL)
        self._max_positions_var = tk.StringVar(value="10")
        self._start_year_var = tk.StringVar(value=str(_resolve_default_portfolio_start_year_hint()))
        self._end_year_display_var = tk.StringVar(value=END_YEAR_LATEST_LABEL)
        self._fixed_risk_display_var = tk.StringVar(value="0.01")
        self._custom_fixed_risk_var = tk.StringVar(value="0.01")
        self._ticker_display_var = tk.StringVar()
        self._show_volume_var = tk.BooleanVar(value=False)
        self._result = None
        self._ticker_map = {}
        self._chart_canvas = None
        self._chart_figure = None
        self._performance_canvas = None
        self._performance_figure = None
        self._console_stream_buffer = ""
        self._console_stream_mode = "line"
        self._console_live_progress_start = None
        self._console_current_tag = "default"
        self._ticker_dropdown_stats = {}
        self._current_chart_trade_indexes = []
        self._current_chart_trade_cursor_index = None
        self._sidebar_signal_var = tk.StringVar(value=SIDEBAR_SIGNAL_CHIP_TEXT)
        self._sidebar_history_var = tk.StringVar(value=SIDEBAR_HISTORY_CHIP_TEXT)
        self._sidebar_summary_var = tk.StringVar(value="-")
        self._selected_date_var = tk.StringVar(value="選取日: -")
        self._selected_open_var = tk.StringVar(value="開: -")
        self._selected_high_var = tk.StringVar(value="高: -")
        self._selected_low_var = tk.StringVar(value="低: -")
        self._selected_close_var = tk.StringVar(value="收: -")
        self._selected_volume_var = tk.StringVar(value="量: -")
        self._selected_tp_var = tk.StringVar(value="停利: -")
        self._selected_limit_var = tk.StringVar(value="限價: -")
        self._selected_entry_var = tk.StringVar(value="成交: -")
        self._selected_stop_var = tk.StringVar(value="停損: -")
        self._selected_actual_spend_var = tk.StringVar(value="實支: -")
        self._build_ui()

    def destroy(self):
        self._clear_kline_chart()
        self._clear_performance_chart()
        super().destroy()

    def _build_ui(self):
        controls = ttk.Frame(self, padding=(8, 2, 8, 2), style="Workbench.TFrame")
        controls.pack(fill="x", pady=(0, 4))
        controls_bar = ttk.Frame(controls, style="Workbench.TFrame")
        controls_bar.pack(side="left", anchor="w")
        pady = (2, 2)

        ttk.Label(controls_bar, text="參數", style="Workbench.TLabel").grid(row=0, column=0, padx=(0, 6), pady=pady, sticky="w")
        self._param_source_combo = ttk.Combobox(
            controls_bar,
            state="readonly",
            width=20,
            textvariable=self._param_source_display_var,
            style="Workbench.TCombobox",
            values=list(PARAM_SOURCE_LABEL_TO_KEY.keys()),
        )
        self._autosize_combobox(self._param_source_combo, values=list(PARAM_SOURCE_LABEL_TO_KEY.keys()), current_text=self._param_source_display_var.get(), rule_key="param_source")
        self._param_source_combo.grid(row=0, column=1, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="汰弱換股", style="Workbench.TLabel").grid(row=0, column=2, padx=(0, 6), pady=pady, sticky="w")
        self._rotation_combo = ttk.Combobox(
            controls_bar,
            state="readonly",
            width=14,
            textvariable=self._rotation_display_var,
            style="Workbench.TCombobox",
            values=list(ROTATION_LABEL_TO_BOOL.keys()),
        )
        self._autosize_combobox(self._rotation_combo, values=list(ROTATION_LABEL_TO_BOOL.keys()), current_text=self._rotation_display_var.get(), rule_key="rotation")
        self._rotation_combo.grid(row=0, column=3, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="最大持股", style="Workbench.TLabel").grid(row=0, column=4, padx=(0, 6), pady=pady, sticky="w")
        ttk.Entry(controls_bar, textvariable=self._max_positions_var, width=5, style="Workbench.TEntry").grid(row=0, column=5, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="起始年", style="Workbench.TLabel").grid(row=0, column=6, padx=(0, 6), pady=pady, sticky="w")
        start_year_values = self._build_start_year_options()
        self._start_year_combo = ttk.Combobox(
            controls_bar,
            state="readonly",
            width=7,
            textvariable=self._start_year_var,
            style="Workbench.TCombobox",
            values=start_year_values,
        )
        self._autosize_combobox(self._start_year_combo, values=start_year_values, current_text=self._start_year_var.get(), rule_key="start_year")
        self._start_year_combo.grid(row=0, column=7, padx=(0, 10), pady=pady, sticky="w")
        self._start_year_combo.bind("<<ComboboxSelected>>", self._on_start_year_selected)

        ttk.Label(controls_bar, text="結束年", style="Workbench.TLabel").grid(row=0, column=8, padx=(0, 6), pady=pady, sticky="w")
        end_year_values = self._build_end_year_options()
        self._end_year_combo = ttk.Combobox(
            controls_bar,
            state="readonly",
            width=7,
            textvariable=self._end_year_display_var,
            style="Workbench.TCombobox",
            values=end_year_values,
        )
        self._autosize_combobox(self._end_year_combo, values=end_year_values, current_text=self._end_year_display_var.get(), rule_key="end_year")
        self._end_year_combo.grid(row=0, column=9, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="固定風險", style="Workbench.TLabel").grid(row=0, column=10, padx=(0, 6), pady=pady, sticky="w")
        self._risk_combo = ttk.Combobox(
            controls_bar,
            state="readonly",
            width=7,
            textvariable=self._fixed_risk_display_var,
            style="Workbench.TCombobox",
            values=FIXED_RISK_LABELS,
        )
        self._autosize_combobox(self._risk_combo, values=FIXED_RISK_LABELS, current_text=self._fixed_risk_display_var.get(), rule_key="risk")
        self._risk_combo.grid(row=0, column=11, padx=(0, 6), pady=pady, sticky="w")
        self._risk_combo.bind("<<ComboboxSelected>>", self._on_fixed_risk_selected)
        self._custom_fixed_risk_entry = ttk.Entry(controls_bar, textvariable=self._custom_fixed_risk_var, width=7, style="Workbench.TEntry")
        self._custom_fixed_risk_entry.grid(row=0, column=12, padx=(0, 10), pady=pady, sticky="w")
        self._custom_fixed_risk_entry.state(["disabled"])

        ttk.Button(controls_bar, text="執行投組回測", command=self._run_portfolio_backtest, style="Workbench.TButton").grid(row=0, column=13, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="K線股票", style="Workbench.TLabel").grid(row=0, column=14, padx=(0, 6), pady=pady, sticky="w")
        self._ticker_combo = ttk.Combobox(controls_bar, state="readonly", width=22, textvariable=self._ticker_display_var, style="Workbench.TCombobox", values=[])
        self._autosize_combobox(self._ticker_combo, values=[], current_text="", rule_key="ticker")
        self._ticker_combo.grid(row=0, column=15, padx=(0, 8), pady=pady, sticky="w")
        self._ticker_combo.bind("<<ComboboxSelected>>", self._on_ticker_selected)

        ttk.Checkbutton(
            controls_bar,
            text="顯示成交量",
            variable=self._show_volume_var,
            command=self._rerender_selected_ticker_chart,
            style="Workbench.TCheckbutton",
        ).grid(row=0, column=16, padx=(0, 0), pady=pady, sticky="w")

        notebook = ttk.Notebook(self, style="Workbench.TNotebook")
        notebook.pack(fill="both", expand=True)
        self._notebook = notebook

        kline_tab = ttk.Frame(notebook, padding=0, style="Workbench.TFrame")
        kline_tab.rowconfigure(0, weight=1)
        kline_tab.columnconfigure(0, weight=1)
        kline_tab.columnconfigure(1, weight=0)
        notebook.add(kline_tab, text="K 線圖")
        self._kline_host = tk.Frame(kline_tab, bg="#000000", highlightthickness=0, bd=0)
        self._kline_host.grid(row=0, column=0, sticky="nsew")
        self._kline_placeholder = self._make_placeholder(self._kline_host, "請先執行投組回測；選擇有成交過的股票後會顯示 K 線結果。")

        sidebar_outer = ttk.Frame(kline_tab, padding=(4, 4, 2, 4), width=188, style="Workbench.TFrame")
        sidebar_outer.grid(row=0, column=1, sticky="ns")
        sidebar_outer.grid_propagate(False)
        kline_tab.grid_columnconfigure(1, minsize=188)

        sidebar = ttk.Frame(sidebar_outer, padding=(2, 2), style="Workbench.TFrame")
        sidebar.pack(fill="both", expand=True)
        sidebar.columnconfigure(0, weight=1)
        sidebar_chip_font = ("Microsoft JhengHei", 13, "bold")
        sidebar_header_font = ("Microsoft JhengHei", 13, "bold")
        sidebar_body_font = ("Microsoft JhengHei", 12)
        self._signal_chip = tk.Label(sidebar, textvariable=self._sidebar_signal_var, bg="#04070c", fg="#ffffff", font=sidebar_chip_font, padx=6, pady=4, anchor="center")
        self._signal_chip.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self._history_chip = tk.Label(sidebar, textvariable=self._sidebar_history_var, bg="#04070c", fg="#ffffff", font=sidebar_chip_font, padx=6, pady=4, anchor="center")
        self._history_chip.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(sidebar, text="歷史績效表", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=2, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._sidebar_summary_var, style="Workbench.SidebarSummary.TLabel", font=sidebar_body_font, justify="left", anchor="nw", wraplength=168).grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(sidebar, text="選取日線值", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=4, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_date_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=5, column=0, sticky="w", pady=(2, 0))
        ttk.Label(sidebar, textvariable=self._selected_open_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=6, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_high_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=7, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_low_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=8, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_close_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=9, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_volume_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=10, column=0, sticky="w", pady=(0, 4))
        ttk.Label(sidebar, text="交易資訊", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=11, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_tp_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=12, column=0, sticky="w", pady=(2, 0))
        ttk.Label(sidebar, textvariable=self._selected_limit_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=13, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_entry_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=14, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_stop_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=15, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_actual_spend_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=16, column=0, sticky="w", pady=(0, 4))
        ttk.Button(sidebar, text="回到最新K線", command=self._move_kline_chart_to_latest, style="Workbench.TButton").grid(row=17, column=0, sticky="ew", pady=(4, 0))
        trade_nav = ttk.Frame(sidebar, style="Workbench.TFrame")
        trade_nav.grid(row=18, column=0, sticky="ew", pady=(4, 0))
        trade_nav.columnconfigure(0, weight=1)
        trade_nav.columnconfigure(1, weight=1)
        ttk.Button(trade_nav, text="前交易", command=self._move_kline_chart_to_previous_trade, style="Workbench.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(trade_nav, text="後交易", command=self._move_kline_chart_to_next_trade, style="Workbench.TButton").grid(row=0, column=1, sticky="ew", padx=(2, 0))
        sidebar.rowconfigure(19, weight=1)

        performance_tab = ttk.Frame(notebook, padding=0, style="Workbench.TFrame")
        performance_tab.rowconfigure(0, weight=1)
        performance_tab.columnconfigure(0, weight=1)
        notebook.add(performance_tab, text="績效圖")
        self._performance_host = tk.Frame(performance_tab, bg="#000000", highlightthickness=0, bd=0)
        self._performance_host.grid(row=0, column=0, sticky="nsew")
        self._performance_placeholder = self._make_placeholder(self._performance_host, "請先執行投組回測；績效圖會顯示投組與 0050 大盤。")

        console_tab = ttk.Frame(notebook, padding=10, style="Workbench.TFrame")
        console_tab.rowconfigure(0, weight=1)
        console_tab.columnconfigure(0, weight=1)
        notebook.add(console_tab, text="Console")
        self._console_text = tk.Text(console_tab, wrap="none", bg="#040a12", fg="#f7fbff", insertbackground="#f7fbff", relief="flat", bd=0, font=("Consolas", 10))
        self._console_text.grid(row=0, column=0, sticky="nsew")
        self._configure_console_tags()
        y_scroll = ttk.Scrollbar(console_tab, orient="vertical", command=self._console_text.yview, style="Workbench.Vertical.TScrollbar")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(console_tab, orient="horizontal", command=self._console_text.xview, style="Workbench.Horizontal.TScrollbar")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self._console_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        footer = ttk.Frame(self, style="Workbench.TFrame")
        footer.pack(fill="x", pady=(2, 0))
        ttk.Label(footer, textvariable=self._status_var, style="Workbench.TLabel").pack(anchor="w")
        self._notebook.select(kline_tab)

    def _make_placeholder(self, master, text):
        label_fg = ttk.Style(self).lookup("Workbench.TLabel", "foreground") or "#f7fbff"
        label = tk.Label(
            master,
            text=text,
            anchor="center",
            justify="center",
            bg="#000000",
            fg=label_fg,
            font=("Microsoft JhengHei", 12),
        )
        label.pack(fill="both", expand=True)
        return label

    def _get_workbench_combobox_font(self):
        if hasattr(self, "_workbench_combobox_font"):
            return self._workbench_combobox_font
        font_spec = ttk.Style(self).lookup("Workbench.TCombobox", "font") or ("Microsoft JhengHei", 11)
        try:
            self._workbench_combobox_font = tkfont.Font(font=font_spec)
        except tk.TclError as exc:
            _warn_gui_fallback('tkfont.Font(font=Workbench.TCombobox)', exc)
            self._workbench_combobox_font = tkfont.nametofont("TkDefaultFont")
        return self._workbench_combobox_font

    def _autosize_combobox(self, combo, *, values, current_text, rule_key):
        rule = COMBOBOX_WIDTH_RULES[rule_key]
        font_obj = self._get_workbench_combobox_font()
        text_candidates = [str(value or "") for value in list(values or [])]
        text_candidates.append(str(current_text or ""))
        try:
            live_text = combo.get()
        except tk.TclError as exc:
            _warn_gui_fallback('combobox.get() during autosize', exc)
            live_text = ""
        text_candidates.append(str(live_text or ""))
        longest_text = max(text_candidates, key=lambda text: font_obj.measure(text), default="")
        average_char_px = max(font_obj.measure("0"), 1)
        text_px = font_obj.measure(longest_text) + int(rule.get("extra_px") or 0)
        width_chars = max(int(rule.get("min_chars") or 0), (text_px + average_char_px - 1) // average_char_px)
        width_chars = min(width_chars, int(rule.get("max_chars") or width_chars))
        combo.configure(width=width_chars)

    def _build_start_year_options(self):
        default_year = int(_resolve_default_portfolio_start_year_hint())
        current_year = max(default_year, datetime.now().year)
        first_year = max(2000, default_year - 6)
        return [str(year) for year in range(first_year, current_year + 1)]

    def _build_end_year_options(self):
        start_year = parse_int_strict(self._start_year_var.get().strip(), "開始回測年份", min_value=1900)
        current_year = max(start_year, datetime.now().year)
        return [END_YEAR_LATEST_LABEL] + [str(year) for year in range(start_year, current_year + 1)]

    def _on_start_year_selected(self, _event=None):
        if not hasattr(self, "_end_year_combo"):
            return
        try:
            end_year_values = self._build_end_year_options()
        except ValueError:
            return
        current_end_year = self._end_year_display_var.get().strip()
        if current_end_year not in end_year_values:
            self._end_year_display_var.set(END_YEAR_LATEST_LABEL)
        self._end_year_combo.configure(values=end_year_values)
        self._autosize_combobox(self._end_year_combo, values=end_year_values, current_text=self._end_year_display_var.get(), rule_key="end_year")

    def _resolve_end_year(self, start_year):
        selected = self._end_year_display_var.get().strip()
        if selected == END_YEAR_LATEST_LABEL:
            return None
        end_year = parse_int_strict(selected, "結束回測年份", min_value=1900)
        if end_year < int(start_year):
            raise ValueError("結束回測年份不可早於開始回測年份")
        return end_year

    def _format_sidebar_line_value(self, label, value):
        return f"{label}: -" if value is None or pd.isna(value) else f"{label}: {float(value):.2f}"

    def _format_sidebar_amount_value(self, label, value):
        return f"{label}: -" if value is None or pd.isna(value) else f"{label}: {float(value):,.0f}"

    def _format_sidebar_ohlcv_value(self, label, value, *, volume=False):
        if value is None or pd.isna(value):
            return f"{label}: -"
        if volume:
            return f"{label}: {float(value) / 1_000_000:.2f}M"
        return f"{label}: {float(value):.2f}"

    def _update_selected_value_sidebar(self, snapshot):
        if not snapshot:
            self._selected_date_var.set("選取日: -")
            self._selected_open_var.set("開: -")
            self._selected_high_var.set("高: -")
            self._selected_low_var.set("低: -")
            self._selected_close_var.set("收: -")
            self._selected_volume_var.set("量: -")
            self._selected_tp_var.set("停利: -")
            self._selected_limit_var.set("限價: -")
            self._selected_entry_var.set("成交: -")
            self._selected_stop_var.set("停損: -")
            self._selected_actual_spend_var.set("實支: -")
            return
        self._selected_date_var.set(f"選取日: {snapshot.get('date_label', '-')}")
        self._selected_open_var.set(self._format_sidebar_ohlcv_value("開", snapshot.get("open")))
        self._selected_high_var.set(self._format_sidebar_ohlcv_value("高", snapshot.get("high")))
        self._selected_low_var.set(self._format_sidebar_ohlcv_value("低", snapshot.get("low")))
        self._selected_close_var.set(self._format_sidebar_ohlcv_value("收", snapshot.get("close")))
        self._selected_volume_var.set(self._format_sidebar_ohlcv_value("量", snapshot.get("volume"), volume=True))
        self._selected_tp_var.set(self._format_sidebar_line_value("停利", snapshot.get("tp_price")))
        self._selected_limit_var.set(self._format_sidebar_line_value("限價", snapshot.get("limit_price")))
        self._selected_entry_var.set(self._format_sidebar_line_value("成交", snapshot.get("entry_price")))
        self._selected_stop_var.set(self._format_sidebar_line_value("停損", snapshot.get("stop_price")))
        self._selected_actual_spend_var.set(self._format_sidebar_amount_value("實支", snapshot.get("buy_capital")))

    def _apply_sidebar_chip_styles(self, signal_active, history_active):
        self._signal_chip.configure(bg=SIDEBAR_CHIP_ACTIVE_BG if bool(signal_active) else SIDEBAR_CHIP_INACTIVE_BG)
        self._history_chip.configure(bg=SIDEBAR_HISTORY_CHIP_ACTIVE_BG if bool(history_active) else SIDEBAR_CHIP_INACTIVE_BG)

    @staticmethod
    def _resolve_sidebar_chip_states(status_lines):
        normalized_lines = [str(line).strip() for line in status_lines if str(line).strip()]
        signal_active = any(line == SIDEBAR_SIGNAL_CHIP_TEXT for line in normalized_lines)
        history_active = any(line in {SIDEBAR_HISTORY_CHIP_TEXT, "歷史績效符合", "歷績門檻符合"} for line in normalized_lines)
        return signal_active, history_active

    def _update_sidebar_from_chart_payload(self, chart_payload):
        chart_payload = dict(chart_payload or {})
        status_lines = list(((chart_payload.get("status_box") or {}).get("lines") or []))
        signal_active, history_active = self._resolve_sidebar_chip_states(status_lines)
        self._sidebar_signal_var.set(SIDEBAR_SIGNAL_CHIP_TEXT)
        self._sidebar_history_var.set(SIDEBAR_HISTORY_CHIP_TEXT)
        self._sidebar_summary_var.set("\n".join(str(line) for line in (chart_payload.get("summary_box") or []) if str(line).strip()) or "-")
        self._apply_sidebar_chip_styles(signal_active, history_active)
        dates = chart_payload.get("date_labels") or []
        if dates:
            idx = int((chart_payload.get("default_view") or {}).get("end_idx", len(dates) - 1))
            idx = max(0, min(idx, len(dates) - 1))
            self._update_selected_value_sidebar(build_chart_hover_snapshot(chart_payload, idx))
        else:
            self._update_selected_value_sidebar(None)

    def _build_gui_chart_payload(self, chart_payload):
        gui_payload = dict(chart_payload or {})
        gui_payload["summary_box"] = []
        gui_payload["status_box"] = {}
        return gui_payload

    def _move_kline_chart_to_latest(self):
        if self._chart_figure is None:
            return
        if scroll_chart_to_latest(self._chart_figure, redraw=True):
            state = getattr(self._chart_figure, "_stock_chart_navigation_state", None)
            if isinstance(state, dict):
                self._current_chart_trade_cursor_index = int(state.get("hover_last_index", 0) or 0)

    def _move_kline_chart_to_previous_trade(self):
        self._move_kline_chart_to_trade(direction=-1)

    def _move_kline_chart_to_next_trade(self):
        self._move_kline_chart_to_trade(direction=1)

    def _move_kline_chart_to_trade(self, *, direction):
        if self._chart_figure is None:
            return False
        trade_indexes = list(self._current_chart_trade_indexes or [])
        if not trade_indexes:
            state = getattr(self._chart_figure, "_stock_chart_navigation_state", None)
            chart_payload = state.get("chart_payload") if isinstance(state, dict) else None
            trade_indexes = _extract_trade_marker_indexes(chart_payload)
            self._current_chart_trade_indexes = trade_indexes
        if scroll_chart_to_adjacent_trade(self._chart_figure, trade_indexes, direction=direction, redraw=True):
            state = getattr(self._chart_figure, "_stock_chart_navigation_state", None)
            if isinstance(state, dict):
                self._current_chart_trade_cursor_index = int(state.get("hover_last_index", 0) or 0)
            return True
        return False

    def _configure_console_tags(self):
        self._console_text.tag_configure("default", foreground="#f7fbff")
        for code, color in PORTFOLIO_CONSOLE_COLORS.items():
            self._console_text.tag_configure(f"ansi_{code}", foreground=color)

    def _on_fixed_risk_selected(self, _event=None):
        if self._fixed_risk_display_var.get() == "自訂":
            self._custom_fixed_risk_entry.state(["!disabled"])
            self._custom_fixed_risk_entry.focus_set()
        else:
            self._custom_fixed_risk_entry.state(["disabled"])

    def _get_selected_param_source(self):
        return PARAM_SOURCE_LABEL_TO_KEY.get(self._param_source_display_var.get().strip(), "run_best")

    def _get_selected_params_path(self):
        if self._get_selected_param_source() == "candidate_best":
            return resolve_candidate_best_params_path(WORKBENCH_PROJECT_ROOT)
        return resolve_run_best_params_path(WORKBENCH_PROJECT_ROOT)

    def _resolve_fixed_risk(self):
        selected = self._fixed_risk_display_var.get().strip()
        raw_value = self._custom_fixed_risk_var.get().strip() if selected == "自訂" else selected
        return parse_float_strict(raw_value, "固定風險比例", min_value=0.0, max_value=1.0, strict_gt=True)

    def _resolve_user_options(self):
        max_positions = parse_int_strict(self._max_positions_var.get().strip(), "最大持倉數量", min_value=1)
        start_year = parse_int_strict(self._start_year_var.get().strip(), "開始回測年份", min_value=1900)
        end_year = self._resolve_end_year(start_year)
        return {
            "params_path": self._get_selected_params_path(),
            "param_source": self._get_selected_param_source(),
            "enable_rotation": ROTATION_LABEL_TO_BOOL.get(self._rotation_display_var.get().strip(), False),
            "max_positions": max_positions,
            "start_year": start_year,
            "end_year": end_year,
            "fixed_risk": self._resolve_fixed_risk(),
            "benchmark_ticker": PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
        }

    def _append_console_text(self, text):
        normalized_text = str(text or "")
        if not normalized_text:
            return
        if threading.current_thread() is not self._ui_thread:
            self.after(0, self._append_console_text, normalized_text)
            return
        self._flush_console_live_progress(force_newline=True)
        self._insert_ansi_text(normalized_text)
        self._console_text.see("end")

    def _append_console_stream(self, text):
        normalized_text = str(text or "").replace("\r\n", "\n")
        if not normalized_text:
            return
        if threading.current_thread() is not self._ui_thread:
            self.after(0, self._append_console_stream, normalized_text)
            return
        current = self._console_stream_buffer
        mode = self._console_stream_mode
        ended_with_carriage_return = False
        for char in normalized_text:
            if char == "\r":
                self._set_console_live_progress(current)
                current = ""
                mode = "progress"
                ended_with_carriage_return = True
                continue
            ended_with_carriage_return = False
            if char == "\n":
                if mode == "progress":
                    self._set_console_live_progress(current)
                    self._flush_console_live_progress(force_newline=True)
                else:
                    self._insert_ansi_text(current + "\n")
                    self._console_text.see("end")
                current = ""
                mode = "line"
                continue
            current += char
        self._console_stream_buffer = current
        self._console_stream_mode = mode
        if mode == "progress" and not ended_with_carriage_return:
            self._set_console_live_progress(current)

    def _insert_ansi_text(self, text):
        current_tag = self._console_current_tag
        pos = 0
        for match in ANSI_PATTERN.finditer(text):
            if match.start() > pos:
                self._console_text.insert("end", text[pos:match.start()], current_tag)
            sequence = match.group(0)
            codes = sequence[2:-1].split(";")
            if not codes or codes == ["0"] or "0" in codes:
                current_tag = "default"
            else:
                for code in codes:
                    if code in PORTFOLIO_CONSOLE_COLORS:
                        current_tag = f"ansi_{code}"
            pos = match.end()
        if pos < len(text):
            self._console_text.insert("end", text[pos:], current_tag)
        self._console_current_tag = current_tag

    def _set_console_live_progress(self, text):
        if self._console_live_progress_start is None:
            self._console_live_progress_start = self._console_text.index("end-1c")
            self._insert_ansi_text(str(text or ""))
        else:
            self._console_text.delete(self._console_live_progress_start, "end-1c")
            self._insert_ansi_text(str(text or ""))
        self._console_text.see("end")

    def _flush_console_live_progress(self, *, force_newline):
        if self._console_live_progress_start is None:
            return
        if force_newline:
            line_tail = self._console_text.get(
                f"{self._console_live_progress_start} lineend",
                f"{self._console_live_progress_start} lineend +1c",
            )
            if line_tail != "\n":
                self._console_text.insert("end", "\n", self._console_current_tag)
        self._console_live_progress_start = None
        self._console_text.see("end")

    def _prepare_console_for_new_task(self):
        self._flush_console_live_progress(force_newline=True)
        self._console_stream_buffer = ""
        self._console_stream_mode = "line"
        self._console_live_progress_start = None
        self._console_current_tag = "default"
        if self._console_text.compare("end-1c", ">", "1.0"):
            if self._console_text.get("end-2c", "end-1c") != "\n":
                self._console_text.insert("end", "\n", self._console_current_tag)
            self._console_text.insert("end", "\n" + ("=" * 80) + "\n", self._console_current_tag)
        self._console_text.see("end")

    def _clear_console(self):
        self._console_stream_buffer = ""
        self._console_stream_mode = "line"
        self._console_live_progress_start = None
        self._console_current_tag = "default"
        self._console_text.delete("1.0", "end")

    def _report_runtime_exception(self, context, exc, *, status_prefix, show_dialog=True):
        error_text = f"{status_prefix}：{type(exc).__name__}: {exc}"
        trace_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._append_console_text(f"[{context}]\n{trace_text}\n")
        self._status_var.set(error_text)
        self._notebook.select(2)
        if show_dialog:
            messagebox.showerror("股票工具工作台", error_text)
        return error_text

    def _run_portfolio_backtest(self):
        if self._run_thread is not None and self._run_thread.is_alive():
            status_text = "投組回測進行中：請等待目前回測完成"
            self._status_var.set(status_text)
            self._append_console_text(f"[portfolio] {status_text}\n")
            return
        try:
            options = self._resolve_user_options()
        except ValueError as exc:
            self._report_runtime_exception("resolve_options", exc, status_prefix="輸入錯誤", show_dialog=True)
            return

        self._active_token += 1
        request_token = self._active_token
        self._prepare_console_for_new_task()
        self._notebook.select(2)
        self._status_var.set("執行中：投組回測")
        self._append_console_text("[portfolio] 執行中：投組回測\n")
        run_thread = threading.Thread(
            target=self._run_portfolio_worker,
            args=(options, request_token),
            name="workbench-portfolio-backtest",
            daemon=True,
        )
        self._run_thread = run_thread
        run_thread.start()

    def _run_portfolio_worker(self, options, request_token):
        try:
            with redirect_stdout(self._console_writer), redirect_stderr(self._console_writer):
                result_payload = self._execute_portfolio_backtest(options)
        except Exception as exc:
            self.after(0, self._finish_portfolio_error, request_token, exc)
            return
        self.after(0, self._finish_portfolio_success, request_token, result_payload)

    def _execute_portfolio_backtest(self, options):
        data_dir = get_dataset_dir(WORKBENCH_PROJECT_ROOT, DEFAULT_DATASET_PROFILE)
        params = load_strict_params(options["params_path"])
        params.fixed_risk = float(options["fixed_risk"])

        print(f"{C_CYAN}================================================================================{C_RESET}")
        print(f"⚙️ {C_YELLOW}V16 投資組合模擬器：機構級實戰期望值 (終極模組化對齊版){C_RESET}")
        print(f"{C_CYAN}================================================================================{C_RESET}")
        print(f"{C_GRAY}📁 使用資料集: {get_dataset_profile_label(DEFAULT_DATASET_PROFILE)} | 來源: workbench | 路徑: {data_dir}{C_RESET}")
        print(f"{C_GRAY}ℹ️ 參數來源: {options['param_source']}{C_RESET}")
        print(f"\n{C_GREEN}✅ 成功載入 AI 訓練大腦！{C_RESET}")
        print(f"{C_GRAY}📦 參數檔: {options['params_path']}{C_RESET}")
        print(f"{C_GRAY}ℹ️ 單筆固定風險: {params.fixed_risk:.4f}{C_RESET}")

        ensure_runtime_dirs()
        start_time = time.time()
        context = load_portfolio_market_context(data_dir, params, verbose=True)
        result = run_portfolio_simulation_prepared(
            context["all_dfs_fast"],
            context["all_trade_logs"],
            context["sorted_dates"],
            params,
            max_positions=options["max_positions"],
            enable_rotation=options["enable_rotation"],
            start_year=options["start_year"],
            end_year=options["end_year"],
            benchmark_ticker=options["benchmark_ticker"],
            verbose=True,
        )
        end_time = time.time()

        (
            df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff,
            final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed,
            total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate,
            normal_trade_count, extended_trade_count, annual_trades,
            reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, pf_profile,
        ) = result

        mode_display = "開啟 (強勢輪動)" if options["enable_rotation"] else "關閉 (穩定鎖倉)"
        min_full_year_return_pct = pf_profile.get("min_full_year_return_pct", 0.0)
        bm_min_full_year_return_pct = pf_profile.get("bm_min_full_year_return_pct", 0.0)

        end_year_label = "最新資料" if options.get("end_year") is None else f"{options['end_year']} 年"
        print(f"\n{C_CYAN}================================================================================{C_RESET}")
        print(f"📊 【投資組合實戰模擬報告 ({options['start_year']} 年 ~ {end_year_label})】")
        print(f"{C_CYAN}================================================================================{C_RESET}")
        print(f"回測總耗時: {end_time - start_time:.2f} 秒")

        print_strategy_dashboard(
            params=params,
            title="績效與風險對比表",
            mode_display=mode_display,
            max_pos=options["max_positions"],
            trades=trade_count,
            missed_b=total_missed,
            missed_s=total_missed_sells,
            final_eq=final_eq,
            avg_exp=avg_exp,
            sys_ret=tot_ret,
            bm_ret=bm_ret,
            sys_mdd=mdd,
            bm_mdd=bm_mdd,
            win_rate=win_rate,
            payoff=pf_payoff,
            ev=pf_ev,
            benchmark_ticker=options["benchmark_ticker"],
            max_exp=max_exp,
            r_sq=r_sq,
            m_win_rate=m_win_rate,
            bm_r_sq=bm_r_sq,
            bm_m_win_rate=bm_m_win_rate,
            normal_trades=normal_trade_count,
            extended_trades=extended_trade_count,
            annual_trades=annual_trades,
            reserved_buy_fill_rate=reserved_buy_fill_rate,
            annual_return_pct=annual_return_pct,
            bm_annual_return_pct=bm_annual_return_pct,
            min_full_year_return_pct=min_full_year_return_pct,
            bm_min_full_year_return_pct=bm_min_full_year_return_pct,
        )

        df_yearly = print_yearly_return_report(
            pf_profile.get("yearly_return_rows", []),
            benchmark_yearly_return_rows=pf_profile.get("bm_yearly_return_rows", []),
            benchmark_ticker=options["benchmark_ticker"],
        )
        if pf_profile.get("full_year_count", 0) > 0:
            print(
                f"{C_GRAY}完整年度數: {pf_profile.get('full_year_count', 0)} | "
                f"最差完整年度報酬: {pf_profile.get('min_full_year_return_pct', 0.0):.2f}% | "
                f"大盤最差完整年度報酬: {pf_profile.get('bm_min_full_year_return_pct', 0.0):.2f}% | "
                f"年化報酬率: {annual_return_pct:.2f}%{C_RESET}"
            )

        export_portfolio_reports(df_eq, df_tr, df_yearly, options["benchmark_ticker"], options["start_year"], end_year=options["end_year"])
        return {
            "df_eq": df_eq,
            "df_tr": df_tr,
            "df_yearly": df_yearly,
            "params": params,
            "options": dict(options),
            "context": context,
            "metrics": {
                "total_return": tot_ret,
                "mdd": mdd,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "annual_return_pct": annual_return_pct,
                "bm_annual_return_pct": bm_annual_return_pct,
            },
        }

    def _finish_portfolio_success(self, request_token, result_payload):
        if request_token != self._active_token:
            return
        self._run_thread = None
        self._result = result_payload
        self._render_performance_chart(result_payload)
        self._refresh_trade_ticker_dropdown(result_payload)
        self._status_var.set("完成：投組回測")
        self._notebook.select(1)

    def _finish_portfolio_error(self, request_token, exc):
        if request_token != self._active_token:
            return
        self._run_thread = None
        self._report_runtime_exception("run_portfolio_backtest", exc, status_prefix="投組回測失敗")

    def _resolve_ticker_dropdown_stats(self, *, ticker, first_buy_row, result_payload):
        df_tr = result_payload.get("df_tr")
        actual_stats = _build_portfolio_ticker_actual_stats(df_tr, ticker)
        sort_method = get_buy_sort_method()
        raw_sort_metric_label = get_buy_sort_metric_label(sort_method)
        sort_metric_label = PORTFOLIO_DROPDOWN_SORT_LABELS.get(raw_sort_metric_label, raw_sort_metric_label)
        ev = self._extract_ev_from_buy_type(first_buy_row.get("Type"))
        projected_cost = _coerce_float(first_buy_row.get("投入總金額"), default=0.0)
        trade_count = int(actual_stats.get("trade_count", 0) or 0)
        win_rate_pct = actual_stats.get("win_rate_pct")
        win_rate_fraction = 0.0 if win_rate_pct is None else float(win_rate_pct) / 100.0
        asset_growth_pct = float(actual_stats.get("asset_growth_pct", 0.0) or 0.0)
        sort_value = calc_buy_sort_value(
            sort_method,
            ev,
            projected_cost,
            win_rate_fraction,
            trade_count,
            asset_growth_pct,
        )
        sort_value_text = format_buy_sort_metric_value(sort_value, sort_method)
        win_rate_text = "-" if win_rate_pct is None else f"{float(win_rate_pct):.1f}%"
        trade_count_text = "-" if trade_count <= 0 else str(trade_count)
        return {
            "sort_metric_label": sort_metric_label,
            "sort_value": float(sort_value),
            "sort_value_text": sort_value_text,
            "sort_text": f"{sort_metric_label} {sort_value_text}",
            "win_rate_text": win_rate_text,
            "trade_count_text": trade_count_text,
            "win_rate": win_rate_pct,
            "trade_count": trade_count,
            "portfolio_actual_stats": actual_stats,
        }


    @staticmethod
    def _extract_ev_from_buy_type(type_text):
        match = PORTFOLIO_BUY_EV_PATTERN.search(str(type_text or ""))
        if not match:
            return 0.0
        return _coerce_float(match.group(1), default=0.0)

    def _format_ticker_dropdown_label(self, *, ticker, first_buy_row, stats):
        entry_type = str(first_buy_row.get("進場類型", "normal") or "normal")
        kind_label = PORTFOLIO_DROPDOWN_KIND_LABELS.get(entry_type, entry_type or "-")
        return (
            f"{ticker}|{kind_label}|{stats.get('sort_text', '-')}"
            f"|勝率 {stats.get('win_rate_text', '-')}"
            f"|次 {stats.get('trade_count_text', '-')}"
        )

    def _refresh_trade_ticker_dropdown(self, result_payload):
        df_tr = result_payload.get("df_tr")
        if df_tr is None or df_tr.empty or "Ticker" not in df_tr.columns:
            self._ticker_map.clear()
            self._ticker_dropdown_stats.clear()
            self._ticker_combo.configure(values=[])
            self._ticker_display_var.set("")
            self._autosize_combobox(self._ticker_combo, values=[], current_text="", rule_key="ticker")
            return

        trade_rows = df_tr[df_tr.apply(_is_actual_trade_row, axis=1)].copy()
        buy_rows = trade_rows[trade_rows.apply(_is_buy_trade_row, axis=1)].copy() if not trade_rows.empty else pd.DataFrame()
        dropdown_entries = []
        seen = set()
        for row in buy_rows.to_dict("records"):
            ticker = str(row.get("Ticker", "") or "").strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            stats = self._resolve_ticker_dropdown_stats(ticker=ticker, first_buy_row=row, result_payload=result_payload)
            label = self._format_ticker_dropdown_label(ticker=ticker, first_buy_row=row, stats=stats)
            dropdown_entries.append((float(stats.get("sort_value", 0.0) or 0.0), ticker, label, stats))

        dropdown_entries.sort(key=lambda item: (-item[0], item[1]))
        values = [label for _sort_value, _ticker, label, _stats in dropdown_entries]
        label_to_ticker = {label: ticker for _sort_value, ticker, label, _stats in dropdown_entries}
        label_stats = {label: stats for _sort_value, _ticker, label, stats in dropdown_entries}

        self._ticker_map = label_to_ticker
        self._ticker_dropdown_stats = label_stats
        self._ticker_combo.configure(values=values)
        self._autosize_combobox(self._ticker_combo, values=values, current_text=values[0] if values else "", rule_key="ticker")
        if values:
            self._ticker_display_var.set(values[0])
            self._render_selected_ticker_chart(values[0])
        else:
            self._ticker_display_var.set("")
            self._clear_kline_chart()
            self._update_selected_value_sidebar(None)
            self._sidebar_summary_var.set("-")
            self._apply_sidebar_chip_styles(False, False)
            self._kline_placeholder.configure(text="投組回測完成，但沒有可顯示的成交股票。")

    def _on_ticker_selected(self, _event=None):
        self._render_selected_ticker_chart(self._ticker_display_var.get())

    def _rerender_selected_ticker_chart(self):
        if self._ticker_display_var.get().strip():
            self._render_selected_ticker_chart(self._ticker_display_var.get())

    def _render_selected_ticker_chart(self, display_label):
        if self._result is None:
            return
        display_key = str(display_label or "").strip()
        ticker = self._ticker_map.get(display_key)
        if not ticker:
            return
        df_tr = self._result.get("df_tr")
        context = self._result.get("context") or {}
        all_dfs_fast = context.get("all_dfs_fast") or {}
        fast_data = all_dfs_fast.get(ticker)
        if fast_data is None or df_tr is None or df_tr.empty:
            return
        ticker_trades = df_tr[(df_tr["Ticker"].astype(str) == ticker) & df_tr.apply(_is_actual_trade_row, axis=1)].copy()
        chart_payload = _build_portfolio_ticker_chart_payload(
            ticker=ticker,
            fast_data=fast_data,
            ticker_trades_df=ticker_trades,
            params=self._result.get("params"),
            ticker_dropdown_stats=self._ticker_dropdown_stats.get(display_key),
        )
        self._render_kline_chart({"ticker": ticker, "chart_payload": chart_payload})

    def _render_kline_chart(self, result):
        if FigureCanvasTkAgg is None:
            backend_error_text = "缺少 matplotlib TkAgg backend，無法內嵌圖表。"
            if FIGURE_CANVAS_TKAGG_IMPORT_ERROR:
                backend_error_text = f"{backend_error_text} {FIGURE_CANVAS_TKAGG_IMPORT_ERROR}"
            self._status_var.set(backend_error_text)
            return backend_error_text
        ticker = result.get("ticker", "")
        chart_payload = result.get("chart_payload")
        trade_indexes = _extract_trade_marker_indexes(chart_payload)
        self._current_chart_trade_indexes = trade_indexes
        self._current_chart_trade_cursor_index = None
        self._update_sidebar_from_chart_payload(chart_payload)
        try:
            figure = create_matplotlib_trade_chart_figure(
                chart_payload=self._build_gui_chart_payload(chart_payload),
                ticker=f"{ticker} 投組",
                show_volume=bool(self._show_volume_var.get()),
            )
        except Exception as exc:
            return self._report_runtime_exception("render_portfolio_kline.figure", exc, status_prefix="K線圖渲染失敗")

        try:
            self._clear_kline_chart()
            self._kline_placeholder.pack_forget()
            canvas = FigureCanvasTkAgg(figure, master=self._kline_host)
            bind_matplotlib_chart_navigation(figure, canvas)
            state = getattr(figure, "_stock_chart_navigation_state", None)
            if isinstance(state, dict):
                state["external_hover_callback"] = self._update_selected_value_sidebar
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.configure(background="#02050a", highlightthickness=0, bd=0, takefocus=1)
            widget.pack(fill="both", expand=True)
            widget.focus_set()
        except Exception as exc:
            self._clear_kline_chart()
            try:
                figure.clear()
            except Exception as clear_exc:
                self._append_console_text(f"[render_portfolio_kline.figure.clear]\n{clear_exc}\n")
            return self._report_runtime_exception("render_portfolio_kline.canvas", exc, status_prefix="K線圖嵌入失敗")

        self._chart_canvas = canvas
        self._chart_figure = figure
        self._current_chart_trade_indexes = trade_indexes
        self._current_chart_trade_cursor_index = None
        self._notebook.select(0)
        self._move_kline_chart_to_latest()
        return ""

    def _render_performance_chart(self, result_payload):
        df_eq = result_payload.get("df_eq")
        options = result_payload.get("options") or {}
        if df_eq is None or df_eq.empty:
            self._clear_performance_chart()
            self._performance_placeholder.configure(text="投組回測完成，但沒有績效曲線資料。")
            return
        if FigureCanvasTkAgg is None:
            self._status_var.set("缺少 matplotlib TkAgg backend，無法內嵌績效圖。")
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib import rcParams
            from matplotlib.font_manager import FontProperties
        except ImportError as exc:
            self._report_runtime_exception("render_performance.import", exc, status_prefix="績效圖渲染失敗")
            return

        rcParams["axes.unicode_minus"] = False
        dates = pd.to_datetime(df_eq["Date"])
        benchmark_ticker = options.get("benchmark_ticker", PORTFOLIO_DEFAULT_BENCHMARK_TICKER)
        bm_col = f"Benchmark_{benchmark_ticker}_Pct"
        font_prop = FontProperties(family="Microsoft JhengHei", size=11)
        title_font = FontProperties(family="Microsoft JhengHei", weight="bold", size=16)
        figure = Figure(figsize=(18.2, 10.6), dpi=96, facecolor="#000000")
        axis = figure.add_subplot(1, 1, 1)
        figure.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=0.08)
        axis.set_facecolor("#000000")
        axis.grid(True, color="#0a1824", alpha=0.22, linewidth=0.7)
        axis.plot(dates, df_eq["Strategy_Return_Pct"].astype(float), linewidth=3.0, color=PERFORMANCE_STRATEGY_COLOR, label="V16 尊爵系統報酬 (%)")
        if bm_col in df_eq.columns:
            axis.plot(dates, df_eq[bm_col].astype(float), linewidth=2.0, color=PERFORMANCE_BENCHMARK_COLOR, label=f"同期大盤 {benchmark_ticker} (%)", alpha=0.8)
        end_year_label = "至今" if options.get("end_year") is None else f"至 {options.get('end_year')}"
        axis.set_title(f"V16 投資組合實戰淨值 vs {benchmark_ticker} 大盤 ({options.get('start_year', '-')} {end_year_label})", color="#f7fbff", fontproperties=title_font)
        axis.set_xlabel("日期", color="#f7fbff", fontproperties=font_prop)
        axis.set_ylabel("累積報酬率 (%)", color="#f7fbff", fontproperties=font_prop)
        axis.tick_params(axis="x", colors="#f7fbff", labelsize=10)
        axis.tick_params(axis="y", colors="#f7fbff", labelsize=10)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#0a1824")
        legend = axis.legend(loc="upper left", frameon=False, prop=font_prop)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#f7fbff")

        self._clear_performance_chart()
        self._performance_placeholder.pack_forget()
        canvas = FigureCanvasTkAgg(figure, master=self._performance_host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(background="#02050a", highlightthickness=0, bd=0)
        widget.pack(fill="both", expand=True)
        self._performance_canvas = canvas
        self._performance_figure = figure

    def _clear_kline_chart(self):
        if self._chart_canvas is not None:
            self._chart_canvas.get_tk_widget().destroy()
            self._chart_canvas = None
        if self._chart_figure is not None:
            self._chart_figure.clear()
            self._chart_figure = None
        self._current_chart_trade_indexes = []
        self._current_chart_trade_cursor_index = None
        if hasattr(self, "_kline_placeholder") and not self._kline_placeholder.winfo_ismapped():
            self._kline_placeholder.pack(fill="both", expand=True)

    def _clear_performance_chart(self):
        if self._performance_canvas is not None:
            self._performance_canvas.get_tk_widget().destroy()
            self._performance_canvas = None
        if self._performance_figure is not None:
            self._performance_figure.clear()
            self._performance_figure = None
        if hasattr(self, "_performance_placeholder") and not self._performance_placeholder.winfo_ismapped():
            self._performance_placeholder.pack(fill="both", expand=True)
