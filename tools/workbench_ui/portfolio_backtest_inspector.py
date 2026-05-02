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

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.model_paths import resolve_candidate_best_params_path, resolve_run_best_params_path
from core.entry_plans import build_position_from_entry_fill
from core.portfolio_fast_data import get_fast_close, get_fast_dates, get_fast_pos, get_fast_value
from core.runtime_utils import parse_float_strict, parse_int_strict
from core.walk_forward_policy import load_walk_forward_policy
from tools.portfolio_sim.reporting import export_portfolio_reports, print_yearly_return_report
from tools.portfolio_sim.runtime import ensure_runtime_dirs, load_strict_params
from tools.trade_analysis.trade_log import run_ticker_analysis
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
    record_limit_order,
    record_signal_annotation,
    record_trade_marker,
    scroll_chart_to_adjacent_trade,
    scroll_chart_to_latest,
)
from tools.workbench_ui.workbench import (
    WORKBENCH_ACCENT,
    WORKBENCH_RIGHT_SIDEBAR_BODY_FONT,
    WORKBENCH_RIGHT_SIDEBAR_HEADER_FONT,
    WORKBENCH_RIGHT_SIDEBAR_WIDTH,
    WORKBENCH_RIGHT_SIDEBAR_WRAPLENGTH,
    WORKBENCH_SURFACE_ALT,
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
PARAM_SOURCE_KEY_TO_LABEL = {value: key for key, value in PARAM_SOURCE_LABEL_TO_KEY.items()}
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


BUY_TRADE_TRACE_NAMES = ("買進", "買進(延續候選)", "錯失買進(新訊號)", "錯失買進(延續候選)", "錯失賣出")
PERFORMANCE_STRATEGY_COLOR = "#ff3333"
PERFORMANCE_BENCHMARK_COLOR = "#4dabf5"
PERFORMANCE_TAB_CLOSE_HITBOX_PX = 32
PERFORMANCE_TAB_CLOSE_BUTTON_SIZE_PX = 18
PERFORMANCE_TAB_CLOSE_BUTTON_RIGHT_PAD_PX = 4
PERFORMANCE_CHART_TITLE_FONT_SIZE = 12
PERFORMANCE_CHART_TITLE_PAD_PX = 6
PERFORMANCE_CHART_SUBPLOT_TOP = 0.945
PERFORMANCE_CHART_CLOSE_BUTTON_SIZE_PX = 22
PERFORMANCE_CHART_CLOSE_BUTTON_MARGIN_PX = 5
PORTFOLIO_RIGHT_SIDEBAR_WIDTH_SCALE = 4 / 3
PORTFOLIO_RIGHT_SIDEBAR_WIDTH = int(round(WORKBENCH_RIGHT_SIDEBAR_WIDTH * PORTFOLIO_RIGHT_SIDEBAR_WIDTH_SCALE))
PORTFOLIO_RIGHT_SIDEBAR_WRAPLENGTH = PORTFOLIO_RIGHT_SIDEBAR_WIDTH - (WORKBENCH_RIGHT_SIDEBAR_WIDTH - WORKBENCH_RIGHT_SIDEBAR_WRAPLENGTH)
COMBOBOX_WIDTH_RULES = {
    "param_source": {"min_chars": 18, "max_chars": 24, "extra_px": 32},
    "rotation": {"min_chars": 12, "max_chars": 18, "extra_px": 30},
    "start_year": {"min_chars": 7, "max_chars": 8, "extra_px": 24},
    "end_year": {"min_chars": 7, "max_chars": 8, "extra_px": 24},
    "risk": {"min_chars": 6, "max_chars": 7, "extra_px": 22},
    "ticker": {"min_chars": 18, "max_chars": 60, "extra_px": 24},
}


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


def _normalize_entry_type_value(value, default="normal"):
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


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
    if raw_type.startswith("錯失買進"):
        return "錯失買進(延續候選)" if entry_type == "extended" or "延續" in raw_type else "錯失買進(新訊號)"
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


def _is_missed_buy_trade_row(row):
    return _normalize_trade_action(row).startswith("錯失買進")


def _is_missed_sell_trade_row(row):
    return _normalize_trade_action(row) == "錯失賣出"


def _is_portfolio_kline_event_row(row):
    return bool(_normalize_trade_action(row))


def _resolve_marker_price(price_df, row, action):
    price = _coerce_float(row.get("成交價"))
    if not pd.isna(price) and price > 0:
        return price
    if str(action).startswith("錯失買進"):
        limit_price = _coerce_float(row.get("買入限價"))
        return limit_price if not pd.isna(limit_price) and limit_price > 0 else np.nan
    if str(action) == "錯失賣出":
        for field in ("停損價", "參考收盤價"):
            reference_price = _coerce_float(row.get(field))
            if not pd.isna(reference_price) and reference_price > 0:
                return reference_price
    return np.nan




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


def _resolve_buy_signal_date_from_row(row, fast_data, trade_date):
    for field in ("買訊日", "Signal_Date", "signal_date"):
        value = row.get(field)
        if value is None or value == "":
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return pd.Timestamp(parsed)
    return _resolve_previous_trade_date(fast_data, trade_date)


def _apply_portfolio_stats_to_marker_meta(meta, actual_stats):
    enriched = dict(meta or {})
    win_rate_pct = (actual_stats or {}).get("win_rate_pct")
    if win_rate_pct is not None and not pd.isna(win_rate_pct):
        enriched["win_rate"] = float(win_rate_pct)
    return enriched


def _resolve_portfolio_event_capital(row):
    for key in ("資金", "權益", "總資產", "當日資產", "可用資金"):
        value = _resolve_valid_float(row.get(key))
        if value is not None:
            return float(value)
    return None


def _build_portfolio_equity_snapshot_index(df_eq):
    if df_eq is None or df_eq.empty or "Date" not in df_eq.columns or "Equity" not in df_eq.columns:
        return {}

    working = df_eq.copy()
    working["_workbench_date"] = pd.to_datetime(working["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["_workbench_equity"] = pd.to_numeric(working["Equity"], errors="coerce")
    working = working.dropna(subset=["_workbench_date", "_workbench_equity"])
    if working.empty:
        return {}

    running_peak = working["_workbench_equity"].cummax()
    drawdown = np.where(running_peak > 0, (running_peak - working["_workbench_equity"]) * 100.0 / running_peak, 0.0)
    working["_workbench_max_drawdown"] = pd.Series(drawdown, index=working.index).cummax()

    snapshots = {}
    for row in working.to_dict("records"):
        date_key = str(row.get("_workbench_date") or "").strip()
        if not date_key:
            continue
        snapshots[date_key] = {
            "current_capital": float(row.get("_workbench_equity")),
            "max_drawdown": float(row.get("_workbench_max_drawdown") or 0.0),
        }
    return snapshots


def _resolve_portfolio_equity_snapshot(equity_snapshots, date_value):
    if not equity_snapshots:
        return {}
    parsed = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed):
        return {}
    return dict(equity_snapshots.get(pd.Timestamp(parsed).strftime("%Y-%m-%d")) or {})


def _apply_portfolio_equity_snapshot_to_marker_meta(meta, row, equity_snapshots=None):
    enriched = dict(meta or {})
    snapshot = _resolve_portfolio_equity_snapshot(equity_snapshots, row.get("Date"))
    if enriched.get("current_capital") is None and snapshot.get("current_capital") is not None:
        enriched["current_capital"] = float(snapshot["current_capital"])
    if enriched.get("max_drawdown") is None and snapshot.get("max_drawdown") is not None:
        enriched["max_drawdown"] = float(snapshot["max_drawdown"])
    return enriched


def _empty_portfolio_ticker_actual_stats():
    return {
        "buy_count": 0,
        "exit_count": 0,
        "event_count": 0,
        "win_count": 0,
        "win_rate_pct": None,
        "trade_count": 0,
        "normal_trade_count": 0,
        "extended_trade_count": 0,
        "missed_buy_count": 0,
        "missed_sell_count": 0,
        "total_reserved_capital": 0.0,
        "total_buy_capital": 0.0,
        "total_pnl": 0.0,
        "asset_growth_pct": None,
    }


def _build_portfolio_ticker_actual_stats(df_tr, ticker):
    if df_tr is None or df_tr.empty or "Ticker" not in df_tr.columns:
        return _empty_portfolio_ticker_actual_stats()

    ticker_rows = df_tr[df_tr["Ticker"].astype(str) == str(ticker)].copy()
    if ticker_rows.empty:
        return _empty_portfolio_ticker_actual_stats()

    all_records = ticker_rows.to_dict("records")
    actual_rows = ticker_rows[ticker_rows.apply(_is_actual_trade_row, axis=1)].copy()
    actual_records = actual_rows.to_dict("records") if not actual_rows.empty else []
    buy_records = [row for row in actual_records if _is_buy_trade_row(row)]
    exit_records = [row for row in actual_records if _is_full_exit_trade_row(row)]
    missed_buy_records = [row for row in all_records if _is_missed_buy_trade_row(row)]
    missed_sell_records = [row for row in all_records if _is_missed_sell_trade_row(row)]

    win_count = sum(1 for row in exit_records if _coerce_float(row.get("該筆總損益"), default=0.0) > 0)
    exit_count = len(exit_records)
    normal_trade_count = 0
    extended_trade_count = 0
    active_entry_type = None
    for row in sorted(actual_records, key=lambda item: (str(item.get("Date", "") or ""), str(item.get("Type", "") or ""))):
        action = _normalize_trade_action(row)
        if action in {"買進", "買進(延續候選)"}:
            active_entry_type = _normalize_entry_type_value(row.get("進場類型"), default="normal")
            continue
        if action in {"停損殺出", "指標賣出", "期末強制結算"}:
            entry_type = _normalize_entry_type_value(row.get("進場類型"), default=active_entry_type or "normal")
            if entry_type == "extended":
                extended_trade_count += 1
            else:
                normal_trade_count += 1
            active_entry_type = None
    win_rate_pct = None if exit_count <= 0 else float(win_count) * 100.0 / float(exit_count)
    total_buy_capital = sum(_coerce_float(row.get("投入總金額"), default=0.0) for row in buy_records)
    total_reserved_capital = sum(_coerce_float(row.get("預留總金額"), default=0.0) for row in buy_records + missed_buy_records)
    total_pnl = sum(_coerce_float(row.get("該筆總損益"), default=0.0) for row in exit_records)
    asset_growth_pct = None if total_buy_capital <= 0 else float(total_pnl) * 100.0 / float(total_buy_capital)
    if normal_trade_count + extended_trade_count != exit_count:
        normal_trade_count = max(int(exit_count) - int(extended_trade_count), 0)
    return {
        "buy_count": int(len(buy_records)),
        "exit_count": int(exit_count),
        "event_count": int(len(actual_records) + len(missed_buy_records) + len(missed_sell_records)),
        "win_count": int(win_count),
        "win_rate_pct": win_rate_pct,
        "trade_count": int(exit_count),
        "normal_trade_count": int(normal_trade_count),
        "extended_trade_count": int(extended_trade_count),
        "missed_buy_count": int(len(missed_buy_records)),
        "missed_sell_count": int(len(missed_sell_records)),
        "total_reserved_capital": float(total_reserved_capital),
        "total_buy_capital": float(total_buy_capital),
        "total_pnl": float(total_pnl),
        "asset_growth_pct": asset_growth_pct,
    }


def _build_portfolio_buy_marker_meta(row, *, fast_data, params, equity_snapshots=None):
    buy_capital = _coerce_float(row.get("投入總金額"), default=np.nan)
    reserved_capital = _coerce_float(row.get("預留總金額"), default=np.nan)
    if _is_missed_buy_trade_row(row) and pd.isna(buy_capital):
        buy_capital = 0.0
    if pd.isna(reserved_capital):
        reserved_capital = buy_capital
    current_capital = _resolve_portfolio_event_capital(row)
    meta = {
        "buy_capital": None if pd.isna(buy_capital) else float(buy_capital),
        "reserved_capital": None if pd.isna(reserved_capital) else float(reserved_capital),
        "current_capital": current_capital,
    }
    position = _build_position_from_portfolio_buy_row(row, fast_data=fast_data, params=params)
    if position is None:
        limit_price = _resolve_buy_limit_from_row(row, fast_data)
        entry_price = _resolve_valid_float(row.get("成交價"))
        if not pd.isna(limit_price):
            meta["limit_price"] = float(limit_price)
        if entry_price is not None:
            meta["entry_price"] = float(entry_price)
        return _apply_portfolio_equity_snapshot_to_marker_meta(meta, row, equity_snapshots)

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
    return _apply_portfolio_equity_snapshot_to_marker_meta(meta, row, equity_snapshots)


def _build_portfolio_sell_marker_meta(row, active_entry, actual_stats=None, equity_snapshots=None):
    pnl_value = _resolve_valid_float(row.get("單筆損益"))
    total_pnl = _resolve_valid_float(row.get("該筆總損益"))
    sell_capital = _estimate_sell_capital(row)
    entry_capital = None if active_entry is None else active_entry.get("buy_capital")
    pnl_pct = _calc_pct_from_capital(total_pnl if total_pnl is not None else pnl_value, entry_capital)
    meta = {}
    current_capital = _resolve_portfolio_event_capital(row)
    if current_capital is not None:
        meta["current_capital"] = float(current_capital)
    if pnl_value is not None:
        meta["pnl_value"] = float(pnl_value)
    if total_pnl is not None:
        meta["total_pnl"] = float(total_pnl)
    if pnl_pct is not None:
        meta["pnl_pct"] = float(pnl_pct)
    if sell_capital is not None:
        meta["sell_capital"] = float(sell_capital)
    meta = _apply_portfolio_equity_snapshot_to_marker_meta(meta, row, equity_snapshots)
    return _apply_portfolio_stats_to_marker_meta(meta, actual_stats or {})


def _build_portfolio_missed_sell_marker_meta(row, actual_stats=None, equity_snapshots=None):
    meta = {}
    current_capital = _resolve_portfolio_event_capital(row)
    if current_capital is not None:
        meta["current_capital"] = float(current_capital)
    total_pnl = _resolve_valid_float(row.get("該筆總損益"))
    stop_price = _resolve_valid_float(row.get("停損價"))
    reference_price = _resolve_valid_float(row.get("參考收盤價"))
    if total_pnl is not None:
        meta["total_pnl"] = float(total_pnl)
    if stop_price is not None:
        meta["stop_price"] = float(stop_price)
    if reference_price is not None:
        meta["reference_price"] = float(reference_price)
    meta = _apply_portfolio_equity_snapshot_to_marker_meta(meta, row, equity_snapshots)
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
    if action in {"買進", "買進(延續候選)", "錯失買進(新訊號)", "錯失買進(延續候選)"}:
        signal_date = _resolve_buy_signal_date_from_row(row, fast_data, trade_date)
        if signal_date not in price_df.index:
            return
        limit_price = marker_meta.get("limit_price")
        reserved_capital = marker_meta.get("reserved_capital")
        qty = _coerce_int(row.get("股數"), default=0)
        _record_portfolio_signal_annotation_once(
            chart_context,
            current_date=signal_date,
            signal_type="buy",
            anchor_price=float(price_df.loc[signal_date, "Low"]),
            title="買訊(錯失)" if str(action).startswith("錯失買進") else "買訊",
            detail_lines=[],
            meta={
                "qty": qty,
                "reserved_capital": reserved_capital,
                "current_capital": marker_meta.get("current_capital"),
                "entry_price": marker_meta.get("entry_price"),
                "limit_price": limit_price,
            },
        )
        return
    if action in {"指標賣出", "錯失賣出"}:
        signal_date = _resolve_previous_trade_date(fast_data, trade_date)
        if signal_date not in price_df.index:
            return
        qty = _coerce_int(row.get("股數"), default=0)
        reference_price = _resolve_valid_float(row.get("參考收盤價"))
        if reference_price is None:
            reference_price = _resolve_valid_float(price_df.loc[signal_date, "Close"])
        record_signal_annotation(
            chart_context,
            current_date=signal_date,
            signal_type="sell",
            anchor_price=float(price_df.loc[signal_date, "Low"]),
            title="賣訊(錯失)" if action == "錯失賣出" else "賣訊",
            detail_lines=[],
            meta={
                "current_capital": marker_meta.get("current_capital"),
                "qty": qty,
                "reference_price": reference_price,
                "profit_pct": marker_meta.get("pnl_pct"),
                "max_drawdown": marker_meta.get("max_drawdown", 0.0),
            },
        )


def _record_portfolio_active_level_rows(chart_context, *, active_level_rows):
    for row in active_level_rows or []:
        try:
            current_date = pd.Timestamp(row.get("Date"))
        except (TypeError, ValueError):
            continue
        try:
            record_active_levels(
                chart_context,
                current_date=current_date,
                stop_price=_coerce_float(row.get("停損價")),
                tp_half_price=_coerce_float(row.get("半倉停利價")),
                limit_price=_coerce_float(row.get("買入限價")),
                entry_price=_coerce_float(row.get("成交價")),
            )
        except (TypeError, ValueError, KeyError, IndexError):
            continue


def _find_portfolio_signal_annotation_index(chart_context, *, current_date, signal_type, limit_price=None):
    annotations = chart_context.get("signal_annotations") if isinstance(chart_context, dict) else None
    if not annotations:
        return None
    target_date = pd.Timestamp(current_date)
    target_signal_type = str(signal_type or "").strip().lower()
    target_limit = _resolve_valid_float(limit_price)
    for idx, item in enumerate(annotations):
        try:
            item_date = pd.Timestamp(item.get("date"))
        except (TypeError, ValueError):
            continue
        if item_date != target_date:
            continue
        if str(item.get("signal_type", "")).strip().lower() != target_signal_type:
            continue
        if target_limit is not None:
            item_limit = _resolve_valid_float((item.get("meta") or {}).get("limit_price"))
            if item_limit is None or abs(float(item_limit) - float(target_limit)) > 1e-9:
                continue
        return idx
    return None


def _record_portfolio_signal_annotation_once(chart_context, *, current_date, signal_type, anchor_price, title, detail_lines, note="", meta=None):
    idx = _find_portfolio_signal_annotation_index(
        chart_context,
        current_date=current_date,
        signal_type=signal_type,
        limit_price=(meta or {}).get("limit_price"),
    )
    annotations = chart_context.get("signal_annotations") if isinstance(chart_context, dict) else None
    if idx is not None and annotations is not None:
        existing_title = str(annotations[idx].get("title", "") or "")
        new_title = str(title or "")
        existing_is_missed_buy = existing_title.startswith("買訊(錯失)")
        new_is_missed_buy = new_title.startswith("買訊(錯失)")
        if existing_title == new_title:
            return
        if str(signal_type or "").strip().lower() == "buy":
            if (not existing_is_missed_buy) and new_is_missed_buy:
                return
            if existing_is_missed_buy and (not new_is_missed_buy):
                del annotations[idx]
            else:
                return
        else:
            return
    record_signal_annotation(
        chart_context,
        current_date=current_date,
        signal_type=signal_type,
        anchor_price=anchor_price,
        title=title,
        detail_lines=detail_lines,
        note=note,
        meta=meta,
    )


def _apply_trade_sequence_to_marker_meta(marker_meta, trade_sequence):
    if trade_sequence is None:
        return marker_meta
    enriched = dict(marker_meta or {})
    enriched["trade_sequence"] = int(trade_sequence)
    return enriched


def _build_portfolio_ticker_chart_payload(*, ticker, fast_data, ticker_trades_df, params=None, ticker_dropdown_stats=None, active_level_rows=None, df_eq=None):
    price_df = _fast_data_to_price_df(fast_data)
    chart_context = create_debug_chart_context(price_df)
    dropdown_stats = dict(ticker_dropdown_stats or {})
    actual_stats = dict(dropdown_stats.get("portfolio_actual_stats") or _build_portfolio_ticker_actual_stats(ticker_trades_df, ticker))
    equity_snapshots = _build_portfolio_equity_snapshot_index(df_eq)

    _record_portfolio_active_level_rows(chart_context, active_level_rows=active_level_rows)

    active_entry = None
    next_trade_sequence = 0
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
            next_trade_sequence += 1
            marker_meta = _build_portfolio_buy_marker_meta(row, fast_data=fast_data, params=params, equity_snapshots=equity_snapshots)
            marker_meta["entry_type"] = _normalize_entry_type_value(row.get("進場類型"), default="normal")
            marker_meta["result"] = "成交"
            marker_meta = _apply_trade_sequence_to_marker_meta(marker_meta, next_trade_sequence)
            active_entry = {
                "buy_capital": marker_meta.get("buy_capital"),
                "qty": qty,
                "date": trade_date,
                "trade_sequence": next_trade_sequence,
            }
        elif str(action).startswith("錯失買進"):
            marker_meta = _build_portfolio_buy_marker_meta(row, fast_data=fast_data, params=params, equity_snapshots=equity_snapshots)
            marker_meta["entry_type"] = _normalize_entry_type_value(row.get("進場類型"), default="extended" if "延續" in str(action) else "normal")
            marker_meta["result"] = "未成交"
        elif action == "錯失賣出":
            marker_meta = _build_portfolio_missed_sell_marker_meta(row, actual_stats, equity_snapshots=equity_snapshots)
            marker_meta = _apply_trade_sequence_to_marker_meta(marker_meta, None if active_entry is None else active_entry.get("trade_sequence"))
        else:
            marker_meta = _build_portfolio_sell_marker_meta(row, active_entry, actual_stats, equity_snapshots=equity_snapshots)
            if active_entry is None and action in {"停損殺出", "指標賣出", "期末強制結算"}:
                next_trade_sequence += 1
                marker_meta = _apply_trade_sequence_to_marker_meta(marker_meta, next_trade_sequence)
            else:
                marker_meta = _apply_trade_sequence_to_marker_meta(marker_meta, None if active_entry is None else active_entry.get("trade_sequence"))
            if action in {"停損殺出", "指標賣出", "期末強制結算"}:
                active_entry = None

        if action in {"買進", "買進(延續候選)", "錯失買進(新訊號)", "錯失買進(延續候選)"}:
            limit_price = marker_meta.get("limit_price")
            if limit_price is not None and not pd.isna(limit_price):
                record_active_levels(
                    chart_context,
                    current_date=trade_date,
                    limit_price=float(limit_price),
                )
                record_limit_order(
                    chart_context,
                    current_date=trade_date,
                    limit_price=float(limit_price),
                    qty=qty,
                    entry_type=_normalize_entry_type_value(row.get("進場類型"), default="normal"),
                    status="missed" if str(action).startswith("錯失買進") else "filled",
                    note=str(row.get("備註", "") or "").strip(),
                )

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

    exit_count = int(actual_stats.get("exit_count", 0) or 0)
    normal_trade_count = int(actual_stats.get("normal_trade_count", 0) or 0)
    extended_trade_count = int(actual_stats.get("extended_trade_count", 0) or 0)
    missed_buy_count = int(actual_stats.get("missed_buy_count", 0) or 0)
    missed_sell_count = int(actual_stats.get("missed_sell_count", 0) or 0)
    total_pnl = float(actual_stats.get("total_pnl", 0.0) or 0.0)
    asset_growth_pct = actual_stats.get("asset_growth_pct")
    win_rate_pct = actual_stats.get("win_rate_pct")
    win_rate_text = "-" if win_rate_pct is None else f"{float(win_rate_pct):.1f}%"
    invested_return_text = "-" if asset_growth_pct is None else f"{float(asset_growth_pct):+.1f}%"
    chart_context["summary_box"] = [
        f"投報度: {invested_return_text}",
        f"總損益: {total_pnl:+,.0f}",
        f"交易次數: {exit_count} (正常: {normal_trade_count} | 延續: {extended_trade_count})",
        f"錯失買進: {missed_buy_count} | 錯失賣出: {missed_sell_count}",
        f"勝率: {win_rate_text}",
    ]
    chart_context["status_box"] = {}
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
        self._performance_tabs = []
        self._performance_tab_close_buttons = {}
        self._performance_tab_seq = 0
        self._console_stream_buffer = ""
        self._console_stream_mode = "line"
        self._console_live_progress_start = None
        self._console_current_tag = "default"
        self._ticker_dropdown_stats = {}
        self._history_summary_cache = {}
        self._current_chart_trade_indexes = []
        self._current_chart_trade_cursor_index = None
        self._history_summary_var = tk.StringVar(value="-")
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
        self._selected_reserved_capital_var = tk.StringVar(value="預留: -")
        self._selected_actual_spend_var = tk.StringVar(value="實支: -")
        self._build_ui()

    def destroy(self):
        self._clear_kline_chart()
        self._clear_all_performance_tabs()
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
        notebook.bind("<Configure>", self._refresh_performance_tab_close_buttons, add="+")
        notebook.bind("<Map>", self._refresh_performance_tab_close_buttons, add="+")
        notebook.bind("<<NotebookTabChanged>>", self._refresh_performance_tab_close_buttons, add="+")

        kline_tab = ttk.Frame(notebook, padding=0, style="Workbench.TFrame")
        kline_tab.rowconfigure(0, weight=1)
        kline_tab.columnconfigure(0, weight=1)
        kline_tab.columnconfigure(1, weight=0)
        notebook.add(kline_tab, text="K 線圖")
        self._kline_host = tk.Frame(kline_tab, bg="#000000", highlightthickness=0, bd=0)
        self._kline_host.grid(row=0, column=0, sticky="nsew")
        self._kline_placeholder = self._make_placeholder(self._kline_host, "請先執行投組回測；選擇有成交過的股票後會顯示 K 線結果。")

        sidebar_outer = ttk.Frame(kline_tab, padding=(0, 3, 0, 3), width=PORTFOLIO_RIGHT_SIDEBAR_WIDTH, style="Workbench.TFrame")
        sidebar_outer.grid(row=0, column=1, sticky="ns")
        sidebar_outer.grid_propagate(False)
        sidebar_outer.pack_propagate(False)
        kline_tab.grid_columnconfigure(1, minsize=PORTFOLIO_RIGHT_SIDEBAR_WIDTH)

        sidebar = ttk.Frame(sidebar_outer, padding=(0, 2), style="Workbench.TFrame")
        sidebar.pack(fill="both", expand=True)
        sidebar.columnconfigure(0, weight=1)
        sidebar_header_font = WORKBENCH_RIGHT_SIDEBAR_HEADER_FONT
        sidebar_body_font = WORKBENCH_RIGHT_SIDEBAR_BODY_FONT
        ttk.Label(sidebar, text="單股歷史績效表", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._history_summary_var, style="Workbench.SidebarSummary.TLabel", font=sidebar_body_font, justify="left", anchor="nw", wraplength=PORTFOLIO_RIGHT_SIDEBAR_WRAPLENGTH).grid(row=1, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(sidebar, text="投組實績摘要", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=2, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._sidebar_summary_var, style="Workbench.SidebarSummary.TLabel", font=sidebar_body_font, justify="left", anchor="nw", wraplength=PORTFOLIO_RIGHT_SIDEBAR_WRAPLENGTH).grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(sidebar, text="選取日線值", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=4, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_date_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=5, column=0, sticky="w", pady=(2, 0))
        ttk.Label(sidebar, textvariable=self._selected_open_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=6, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_high_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=7, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_low_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=8, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_close_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=9, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_volume_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=10, column=0, sticky="w", pady=(0, 4))
        ttk.Label(sidebar, text="投組交易資訊", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=11, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_tp_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=12, column=0, sticky="w", pady=(2, 0))
        ttk.Label(sidebar, textvariable=self._selected_limit_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=13, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_entry_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=14, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_stop_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=15, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_reserved_capital_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=16, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_actual_spend_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=17, column=0, sticky="w", pady=(0, 4))
        ttk.Button(sidebar, text="回到最新K線", command=self._move_kline_chart_to_latest, style="Workbench.Sidebar.TButton").grid(row=18, column=0, sticky="ew", pady=(4, 0))
        trade_nav = ttk.Frame(sidebar, style="Workbench.TFrame")
        trade_nav.grid(row=19, column=0, sticky="ew", pady=(0, 0))
        trade_nav.columnconfigure(0, weight=1)
        trade_nav.columnconfigure(1, weight=1)
        ttk.Button(trade_nav, text="前交易", command=self._move_kline_chart_to_previous_trade, style="Workbench.Sidebar.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 0))
        ttk.Button(trade_nav, text="後交易", command=self._move_kline_chart_to_next_trade, style="Workbench.Sidebar.TButton").grid(row=0, column=1, sticky="ew", padx=(0, 0))
        sidebar.rowconfigure(19, weight=1)

        console_tab = ttk.Frame(notebook, padding=10, style="Workbench.TFrame")
        self._console_tab = console_tab
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
            self._selected_reserved_capital_var.set("預留: -")
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
        self._selected_reserved_capital_var.set(self._format_sidebar_amount_value("預留", snapshot.get("reserved_capital")))
        self._selected_actual_spend_var.set(self._format_sidebar_amount_value("實支", snapshot.get("buy_capital")))

    def _resolve_single_stock_history_params(self):
        result_payload = self._result or {}
        params = result_payload.get("params")
        if params is not None:
            return params

        options = result_payload.get("options") or {}
        params_path = str(options.get("params_path") or "").strip()
        if not params_path:
            return None

        loaded_params = load_strict_params(params_path)
        if "fixed_risk" in options:
            loaded_params.fixed_risk = float(options["fixed_risk"])
        return loaded_params

    def _resolve_single_stock_history_summary_lines(self, ticker):
        resolved_ticker = str(ticker or "").strip()
        if not resolved_ticker:
            return ["-"]
        result_payload = self._result or {}
        options = result_payload.get("options") or {}
        params = self._resolve_single_stock_history_params()
        fixed_risk = getattr(params, "fixed_risk", options.get("fixed_risk", None))
        cache_key = (resolved_ticker, str(options.get("params_path") or "").strip(), None if fixed_risk is None else float(fixed_risk))
        if cache_key in self._history_summary_cache:
            return list(self._history_summary_cache[cache_key])

        try:
            analysis_result = run_ticker_analysis(
                resolved_ticker,
                dataset_profile_key=DEFAULT_DATASET_PROFILE,
                data_dir=get_dataset_dir(WORKBENCH_PROJECT_ROOT, DEFAULT_DATASET_PROFILE),
                params=params,
                export_excel=False,
                export_chart=False,
                return_chart_payload=True,
                verbose=False,
            )
            chart_payload = dict(analysis_result.get("chart_payload") or {})
            summary_lines = [str(line) for line in (chart_payload.get("summary_box") or []) if str(line).strip()]
            lines = list(summary_lines)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            warnings.warn(
                f"投組右側歷史績效表載入失敗: ticker={resolved_ticker} | {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            lines = [f"歷史績效表無法載入: {type(exc).__name__}"]

        if not lines:
            lines = ["-"]
        self._history_summary_cache[cache_key] = list(lines)
        return lines

    def _update_sidebar_from_chart_payload(self, chart_payload):
        chart_payload = dict(chart_payload or {})
        self._history_summary_var.set("\n".join(str(line) for line in (chart_payload.get("history_summary_box") or []) if str(line).strip()) or "-")
        self._sidebar_summary_var.set("\n".join(str(line) for line in (chart_payload.get("summary_box") or []) if str(line).strip()) or "-")
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
        param_source_label = self._param_source_display_var.get().strip() or DEFAULT_PARAM_SOURCE_LABEL
        return {
            "params_path": self._get_selected_params_path(),
            "param_source": self._get_selected_param_source(),
            "param_source_label": param_source_label,
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
        self._notebook.select(self._console_tab)
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
        self._notebook.select(self._console_tab)
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
            pit_stats_index=context.get("all_pit_stats_index"),
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
            "profile_stats": dict(pf_profile),
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
        self._history_summary_cache.clear()
        self._render_performance_chart(result_payload)
        self._refresh_trade_ticker_dropdown(result_payload)
        self._status_var.set("完成：投組回測")

    def _finish_portfolio_error(self, request_token, exc):
        if request_token != self._active_token:
            return
        self._run_thread = None
        self._report_runtime_exception("run_portfolio_backtest", exc, status_prefix="投組回測失敗")

    def _resolve_ticker_dropdown_stats(self, *, ticker, first_buy_row, result_payload):
        df_tr = result_payload.get("df_tr")
        actual_stats = _build_portfolio_ticker_actual_stats(df_tr, ticker)
        trade_count = int(actual_stats.get("trade_count", 0) or 0)
        buy_count = int(actual_stats.get("buy_count", 0) or 0)
        missed_buy_count = int(actual_stats.get("missed_buy_count", 0) or 0)
        missed_sell_count = int(actual_stats.get("missed_sell_count", 0) or 0)
        win_rate_pct = actual_stats.get("win_rate_pct")
        total_pnl = float(actual_stats.get("total_pnl", 0.0) or 0.0)
        total_buy_capital = float(actual_stats.get("total_buy_capital", 0.0) or 0.0)
        invested_return_pct = None if total_buy_capital <= 0 else total_pnl * 100.0 / total_buy_capital
        total_pnl_text = f"{total_pnl:+,.0f}"
        invested_return_text = "-" if invested_return_pct is None else f"{invested_return_pct:+.1f}%"
        win_rate_text = "-" if win_rate_pct is None else f"{float(win_rate_pct):.1f}%"
        trade_count_text = "-" if trade_count <= 0 else str(trade_count)
        has_actual_buy = buy_count > 0
        missed_event_count = missed_buy_count + missed_sell_count
        return {
            "sort_metric_label": "總損益",
            "sort_group": 0 if has_actual_buy else 1,
            "sort_value": float(total_pnl if has_actual_buy else missed_event_count),
            "sort_value_text": total_pnl_text,
            "total_pnl": float(total_pnl),
            "total_pnl_text": total_pnl_text,
            "invested_return_pct": invested_return_pct,
            "invested_return_text": invested_return_text,
            "win_rate_text": win_rate_text,
            "trade_count_text": trade_count_text,
            "buy_count": buy_count,
            "missed_buy_count": missed_buy_count,
            "missed_sell_count": missed_sell_count,
            "win_rate": win_rate_pct,
            "trade_count": trade_count,
            "portfolio_actual_stats": actual_stats,
        }


    def _format_ticker_dropdown_label(self, *, ticker, first_buy_row, stats):
        return (
            f"{ticker}"
            f" | 總損益 {stats.get('total_pnl_text', '-')}"
            f" | 勝率 {stats.get('win_rate_text', '-')}"
            f" | 交易次數 {stats.get('trade_count_text', '-')}"
        )

    def _refresh_trade_ticker_dropdown(self, result_payload):
        df_tr = result_payload.get("df_tr")
        if df_tr is None or df_tr.empty or "Ticker" not in df_tr.columns:
            self._ticker_map.clear()
            self._ticker_dropdown_stats.clear()
            self._ticker_combo.configure(values=[])
            self._ticker_display_var.set("")
            self._autosize_combobox(self._ticker_combo, values=[], current_text="", rule_key="ticker")
            self._history_summary_var.set("-")
            self._sidebar_summary_var.set("-")
            return

        event_rows = df_tr[df_tr.apply(_is_portfolio_kline_event_row, axis=1)].copy()
        dropdown_entries = []
        seen = set()
        for row in event_rows.to_dict("records"):
            ticker = str(row.get("Ticker", "") or "").strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            stats = self._resolve_ticker_dropdown_stats(ticker=ticker, first_buy_row=row, result_payload=result_payload)
            label = self._format_ticker_dropdown_label(ticker=ticker, first_buy_row=row, stats=stats)
            dropdown_entries.append((int(stats.get("sort_group", 1)), float(stats.get("sort_value", 0.0) or 0.0), ticker, label, stats))

        dropdown_entries.sort(key=lambda item: (item[0], -item[1], item[2]))
        values = [label for _sort_group, _sort_value, _ticker, label, _stats in dropdown_entries]
        label_to_ticker = {label: ticker for _sort_group, _sort_value, ticker, label, _stats in dropdown_entries}
        label_stats = {label: stats for _sort_group, _sort_value, _ticker, label, stats in dropdown_entries}

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
            self._history_summary_var.set("-")
            self._sidebar_summary_var.set("-")
            self._kline_placeholder.configure(text="投組回測完成，但沒有可顯示的成交或錯失事件股票。")

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
        ticker_trades = df_tr[(df_tr["Ticker"].astype(str) == ticker) & df_tr.apply(_is_portfolio_kline_event_row, axis=1)].copy()
        profile_stats = self._result.get("profile_stats") or {}
        active_level_rows = [
            row for row in (profile_stats.get("portfolio_active_level_rows") or [])
            if str(row.get("Ticker", "")).strip() == str(ticker)
        ]
        chart_payload = _build_portfolio_ticker_chart_payload(
            ticker=ticker,
            fast_data=fast_data,
            ticker_trades_df=ticker_trades,
            params=self._result.get("params"),
            ticker_dropdown_stats=self._ticker_dropdown_stats.get(display_key),
            active_level_rows=active_level_rows,
            df_eq=self._result.get("df_eq"),
        )
        chart_payload["history_summary_box"] = self._resolve_single_stock_history_summary_lines(ticker)
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

    def _resolve_performance_tab_title(self, options):
        self._performance_tab_seq += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        end_year_label = "至今" if options.get("end_year") is None else f"至{options.get('end_year')}"
        return f"績效 {self._performance_tab_seq}｜{timestamp}｜{options.get('start_year', '-')} {end_year_label}"

    def _get_workbench_notebook_font(self):
        if hasattr(self, "_workbench_notebook_font"):
            return self._workbench_notebook_font
        font_spec = ttk.Style(self).lookup("Workbench.TNotebook.Tab", "font") or ("Microsoft JhengHei", 11)
        try:
            self._workbench_notebook_font = tkfont.Font(font=font_spec)
        except tk.TclError as exc:
            _warn_gui_fallback("tkfont.Font(font=Workbench.TNotebook.Tab)", exc)
            self._workbench_notebook_font = tkfont.nametofont("TkDefaultFont")
        return self._workbench_notebook_font

    def _find_performance_tab_record(self, tab_id):
        for item in list(self._performance_tabs):
            if str(item.get("tab_id")) == str(tab_id):
                return item
        return None

    def _get_or_create_performance_close_button(self, tab_id):
        tab_id = str(tab_id)
        button = self._performance_tab_close_buttons.get(tab_id)
        if button is not None and button.winfo_exists():
            return button

        record = self._find_performance_tab_record(tab_id)
        if record is None:
            return None
        close_host = record.get("close_host") or record.get("frame")
        if close_host is None:
            return None

        button = tk.Button(
            close_host,
            text="×",
            command=lambda target_tab_id=tab_id: self._close_performance_tab_by_id(target_tab_id),
            bg="#243040",
            activebackground="#3a4a60",
            fg="#ffffff",
            activeforeground="#ffffff",
            cursor="hand2",
            font=("Microsoft JhengHei", 12, "bold"),
            bd=0,
            relief="flat",
            highlightthickness=0,
            padx=0,
            pady=0,
            takefocus=0,
        )
        self._performance_tab_close_buttons[tab_id] = button
        record["close_button"] = button
        return button

    def _refresh_performance_tab_close_buttons(self, _event=None):
        notebook = getattr(self, "_notebook", None)
        if notebook is None:
            return None

        live_tab_ids = {str(item.get("tab_id")) for item in getattr(self, "_performance_tabs", [])}
        for tab_id, button in list(getattr(self, "_performance_tab_close_buttons", {}).items()):
            if str(tab_id) in live_tab_ids:
                continue
            try:
                button.destroy()
            except tk.TclError as exc:
                _warn_gui_fallback("performance_tab.close_button.destroy", exc)
            self._performance_tab_close_buttons.pop(str(tab_id), None)

        notebook_tabs = [str(tab_id) for tab_id in notebook.tabs()]
        for item in list(getattr(self, "_performance_tabs", [])):
            tab_id = str(item.get("tab_id"))
            if tab_id not in notebook_tabs:
                self._destroy_performance_close_button(tab_id)
                continue
            button = item.get("close_button") or self._get_or_create_performance_close_button(tab_id)
            if button is None:
                continue
            try:
                button.place(
                    relx=1.0,
                    x=-int(PERFORMANCE_CHART_CLOSE_BUTTON_MARGIN_PX),
                    y=int(PERFORMANCE_CHART_CLOSE_BUTTON_MARGIN_PX),
                    width=int(PERFORMANCE_CHART_CLOSE_BUTTON_SIZE_PX),
                    height=int(PERFORMANCE_CHART_CLOSE_BUTTON_SIZE_PX),
                    anchor="ne",
                )
                button.lift()
            except tk.TclError as exc:
                _warn_gui_fallback("performance_tab.close_button.place_in_chart", exc)
                self._destroy_performance_close_button(tab_id)
        return None

    def _destroy_performance_close_button(self, tab_id):
        button = getattr(self, "_performance_tab_close_buttons", {}).pop(str(tab_id), None)
        if button is None:
            return
        try:
            button.destroy()
        except tk.TclError as exc:
            _warn_gui_fallback("performance_tab.close_button.destroy", exc)

    def _get_notebook_tab_index_at_event(self, *, notebook, event):
        try:
            return int(notebook.index(f"@{event.x},{event.y}"))
        except tk.TclError as exc:
            _warn_gui_fallback("performance_tab.index_at_event", exc)
        for index, _tab_id in enumerate(notebook.tabs()):
            try:
                bbox = notebook.bbox(index)
            except tk.TclError as exc:
                _warn_gui_fallback("performance_tab.bbox_at_event", exc)
                continue
            if not bbox:
                continue
            left, top, width, height = [int(value) for value in bbox]
            if left <= int(event.x) <= left + width and top <= int(event.y) <= top + height:
                return index
        return None

    def _is_performance_close_hit(self, *, notebook, tab_index, event, bbox):
        if not bbox:
            return False
        try:
            tab_id = notebook.tabs()[int(tab_index)]
        except (IndexError, TypeError, ValueError):
            return False
        if self._find_performance_tab_record(tab_id) is None:
            return False

        tab_left = int(bbox[0])
        tab_top = int(bbox[1])
        tab_width = int(bbox[2])
        tab_height = int(bbox[3])
        event_x = int(event.x)
        event_y = int(event.y)
        if not (tab_top <= event_y <= tab_top + tab_height):
            return False

        close_width = max(int(PERFORMANCE_TAB_CLOSE_HITBOX_PX), 24)
        close_left = tab_left + tab_width - close_width
        close_right = tab_left + tab_width
        return close_left <= event_x <= close_right

    def _on_performance_tab_click(self, event):
        notebook = getattr(self, "_notebook", None)
        if notebook is None:
            return None
        tab_index = self._get_notebook_tab_index_at_event(notebook=notebook, event=event)
        if tab_index is None:
            return None
        try:
            bbox = notebook.bbox(tab_index)
        except tk.TclError as exc:
            _warn_gui_fallback("performance_tab.bbox_click", exc)
            return None
        if self._is_performance_close_hit(notebook=notebook, tab_index=tab_index, event=event, bbox=bbox):
            self._close_performance_tab_by_index(tab_index)
            return "break"
        return None

    def _close_performance_tab_by_index(self, tab_index):
        notebook = getattr(self, "_notebook", None)
        if notebook is None:
            return
        try:
            tab_id = notebook.tabs()[int(tab_index)]
        except (IndexError, TypeError, ValueError):
            return
        self._close_performance_tab_by_id(tab_id)

    def _close_performance_tab_by_id(self, tab_id):
        notebook = getattr(self, "_notebook", None)
        if notebook is None:
            return
        tab_id = str(tab_id)
        record = self._find_performance_tab_record(tab_id)
        if record is None:
            return
        self._performance_tabs.remove(record)
        self._destroy_performance_close_button(tab_id)
        canvas = record.get("canvas")
        figure = record.get("figure")
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        if figure is not None:
            figure.clear()
        try:
            notebook.forget(tab_id)
        except tk.TclError as exc:
            _warn_gui_fallback("performance_tab.forget", exc)
        if not self._performance_tabs:
            try:
                notebook.select(self._console_tab)
            except tk.TclError as exc:
                _warn_gui_fallback("performance_tab.select(console)", exc)
        self.after_idle(self._refresh_performance_tab_close_buttons)

    def _resolve_performance_tab_insert_index(self):
        notebook = getattr(self, "_notebook", None)
        console_tab = getattr(self, "_console_tab", None)
        if notebook is None or console_tab is None:
            return "end"
        try:
            return notebook.index(console_tab)
        except tk.TclError as exc:
            _warn_gui_fallback("performance_tab.index(console)", exc)
            return "end"

    def _format_performance_param_value(self, value, *, pct=False):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value) if value is not None else "-"
        if pct:
            return f"{numeric * 100:.1f}%"
        if abs(numeric) >= 100:
            return f"{numeric:.0f}"
        return f"{numeric:.3g}"

    def _build_performance_setting_title(self, *, options):
        end_year_label = "至今" if options.get("end_year") is None else str(options.get("end_year"))
        rotation_label = "開" if options.get("enable_rotation") else "關"
        param_source = str(options.get("param_source_label") or PARAM_SOURCE_KEY_TO_LABEL.get(options.get("param_source"), options.get("param_source") or "-"))
        param_file = os.path.basename(str(options.get("params_path") or "")) or "-"
        return (
            f"設定：{param_source}｜{param_file}｜區間 {options.get('start_year', '-')}~{end_year_label}"
            f"｜持股 {options.get('max_positions', '-')}｜汰弱 {rotation_label}"
            f"｜固定風險 {self._format_performance_param_value(options.get('fixed_risk'))}"
            f"｜Benchmark {options.get('benchmark_ticker', PORTFOLIO_DEFAULT_BENCHMARK_TICKER)}"
        )

    def _render_performance_chart(self, result_payload):
        df_eq = result_payload.get("df_eq")
        options = result_payload.get("options") or {}
        if df_eq is None or df_eq.empty:
            self._status_var.set("完成：投組回測，但沒有績效曲線資料")
            self._append_console_text("[portfolio] 投組回測完成，但沒有績效曲線資料。\n")
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
        title_font = FontProperties(family="Microsoft JhengHei", weight="bold", size=PERFORMANCE_CHART_TITLE_FONT_SIZE)
        figure = Figure(figsize=(18.2, 10.6), dpi=96, facecolor="#000000")
        axis = figure.add_subplot(1, 1, 1)
        figure.subplots_adjust(left=0.055, right=0.985, top=PERFORMANCE_CHART_SUBPLOT_TOP, bottom=0.08)
        axis.set_facecolor("#000000")
        axis.grid(True, color="#0a1824", alpha=0.22, linewidth=0.7)
        axis.plot(dates, df_eq["Strategy_Return_Pct"].astype(float), linewidth=3.0, color=PERFORMANCE_STRATEGY_COLOR, label="V16 尊爵系統報酬 (%)")
        if bm_col in df_eq.columns:
            axis.plot(dates, df_eq[bm_col].astype(float), linewidth=2.0, color=PERFORMANCE_BENCHMARK_COLOR, label=f"同期大盤 {benchmark_ticker} (%)", alpha=0.8)
        axis.set_title(
            self._build_performance_setting_title(options=options),
            color="#f7fbff",
            fontproperties=title_font,
            pad=PERFORMANCE_CHART_TITLE_PAD_PX,
        )
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

        tab_frame = ttk.Frame(self._notebook, padding=0, style="Workbench.TFrame")
        tab_frame.rowconfigure(0, weight=1)
        tab_frame.columnconfigure(0, weight=1)
        host = tk.Frame(tab_frame, bg="#000000", highlightthickness=0, bd=0)
        host.grid(row=0, column=0, sticky="nsew")
        canvas = FigureCanvasTkAgg(figure, master=host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(background="#02050a", highlightthickness=0, bd=0)
        widget.pack(fill="both", expand=True)
        tab_title = self._resolve_performance_tab_title(options)
        self._notebook.insert(self._resolve_performance_tab_insert_index(), tab_frame, text=tab_title)
        tab_id = self._notebook.tabs()[self._notebook.index(tab_frame)]
        self._performance_tabs.append({"tab_id": tab_id, "frame": tab_frame, "close_host": host, "canvas": canvas, "figure": figure})
        self._get_or_create_performance_close_button(tab_id)
        self._notebook.select(tab_frame)
        self.after_idle(self._refresh_performance_tab_close_buttons)

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

    def _clear_all_performance_tabs(self):
        notebook = getattr(self, "_notebook", None)
        for item in list(getattr(self, "_performance_tabs", [])):
            canvas = item.get("canvas")
            figure = item.get("figure")
            if canvas is not None:
                canvas.get_tk_widget().destroy()
            if figure is not None:
                figure.clear()
            tab_id = str(item.get("tab_id"))
            self._destroy_performance_close_button(tab_id)
            if notebook is not None:
                try:
                    notebook.forget(tab_id)
                except tk.TclError as exc:
                    _warn_gui_fallback("performance_tab.forget", exc)
        self._performance_tabs = []
        self.after_idle(self._refresh_performance_tab_close_buttons)
