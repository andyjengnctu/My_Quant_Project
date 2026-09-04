"""Completed-daily-bar roll-forward for live Trading strategy positions.

This service advances only deterministic next-session management state from
completed Trading OHLCV.  It never infers broker fills, executes Stop/TP, or
uses current strategy params for an existing position; each holding is advanced
with the frozen params captured on its source entry order.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from core.data_utils import (
    discover_unique_csv_map,
    get_required_min_rows,
    sanitize_ohlcv_dataframe,
)
from core.params_io import build_params_from_mapping
from core.position_step import rollforward_position_management_from_completed_bar
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trading_account_state import (
    MANAGEMENT_STATUS_ACTIVE,
    POSITION_SOURCE_STRATEGY_FILL,
)
from core.trading_market_clock import latest_allowed_completed_daily_date
from core.trading_order_state import active_trading_entry_orders
from services.trading.account_state import (
    load_trading_account_state,
    rollforward_trading_strategy_management,
)
from services.trading.fill_reconciliation import recover_trading_fill_transaction
from services.trading.order_state import load_trading_order_state
from services.trading.protection_planning import build_trading_protection_plan

TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION = 1


def _normalize_date(value: object | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")


def _load_position_market_frame(*, file_path: str, ticker: str, params, allowed_date: str) -> pd.DataFrame:
    raw = pd.read_csv(file_path)
    df, _stats = sanitize_ohlcv_dataframe(
        raw,
        ticker,
        min_rows=get_required_min_rows(params),
    )
    allowed_ts = pd.Timestamp(allowed_date).normalize()
    if pd.Timestamp(df.index.max()).normalize() > allowed_ts:
        raise RuntimeError(
            f"Trading position data 含尚未完成日K: {ticker} "
            f"latest={pd.Timestamp(df.index.max()).strftime('%Y-%m-%d')} > allowed={allowed_date}"
        )
    return df


def _resolve_strategy_position_sources(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    account = load_trading_account_state(project_root, required=False)
    if account is None:
        return {}, {}, {}
    orders = load_trading_order_state(project_root, required=False)
    if orders is None:
        orders = {"orders": {}}
    has_strategy_positions = any(
        isinstance(record, dict) and record.get("source") == POSITION_SOURCE_STRATEGY_FILL
        for record in (account.get("positions") or {}).values()
    )
    if not has_strategy_positions:
        return account, orders, {}
    paths = resolve_runtime_domain_paths(project_root, domain=RUNTIME_DOMAIN_TRADING)
    csv_map, duplicate_issues = discover_unique_csv_map(paths.data_dir)
    if duplicate_issues:
        raise RuntimeError("Trading dataset 存在重複ticker CSV：" + "；".join(duplicate_issues[:5]))
    return account, orders, csv_map


def build_trading_position_rollforward_snapshot(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    account, orders, csv_map = _resolve_strategy_position_sources(root)
    allowed_date = latest_allowed_completed_daily_date()
    if not account:
        return {
            "schema_version": TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION,
            "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "allowed_completed_date": allowed_date,
            "strategy_position_count": 0,
            "due_count": 0,
            "due_tickers": [],
            "positions": [],
        }

    rows: list[dict[str, Any]] = []
    for ticker in sorted(account.get("positions") or {}):
        record = account["positions"][ticker]
        if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
            continue
        management = record.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise RuntimeError(f"Trading strategy position management 非 active: {ticker}")
        broker = record.get("broker") or {}
        entry_order_id = str(broker.get("entry_order_id") or "").strip()
        order = (orders.get("orders") or {}).get(entry_order_id)
        if not isinstance(order, dict) or not isinstance(order.get("frozen_params"), dict):
            raise RuntimeError(f"Trading position 缺少來源 entry order frozen params: {ticker}")
        file_path = csv_map.get(ticker)
        if not file_path:
            raise FileNotFoundError(f"Trading dataset 缺少持股 {ticker} CSV")
        params = build_params_from_mapping(order["frozen_params"])
        df = _load_position_market_frame(file_path=file_path, ticker=ticker, params=params, allowed_date=allowed_date)
        entry_date = _normalize_date(broker.get("entry_date"))
        last_rollforward = _normalize_date(management.get("last_rollforward_date"))
        eligible = df.index
        if entry_date is not None:
            eligible = eligible[eligible >= pd.Timestamp(entry_date)]
        if last_rollforward is not None:
            eligible = eligible[eligible > pd.Timestamp(last_rollforward)]
        target_date = None if len(eligible) == 0 else pd.Timestamp(eligible[-1]).strftime("%Y-%m-%d")
        rows.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "last_rollforward_date": last_rollforward,
                "target_rollforward_date": target_date,
                "due": target_date is not None,
            }
        )

    due_tickers = [row["ticker"] for row in rows if row["due"]]
    return {
        "schema_version": TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "allowed_completed_date": allowed_date,
        "strategy_position_count": len(rows),
        "due_count": len(due_tickers),
        "due_tickers": due_tickers,
        "positions": rows,
    }


def run_trading_position_rollforward(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    account, orders, csv_map = _resolve_strategy_position_sources(root)
    allowed_date = latest_allowed_completed_daily_date()
    if not account:
        return {
            "status": "NO_ACCOUNT",
            "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "processed_position_count": 0,
            "processed_bar_count": 0,
            "positions": [],
        }
    if active_trading_entry_orders(orders):
        raise RuntimeError("Trading 尚有 active ENTRY BUY；完成成交／取消 reconciliation 前禁止日終持股推進")

    updates: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    for ticker in sorted(account.get("positions") or {}):
        record = account["positions"][ticker]
        if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
            continue
        management = record.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise RuntimeError(f"Trading strategy position management 非 active: {ticker}")
        position = deepcopy(management.get("position_state"))
        if not isinstance(position, dict):
            raise RuntimeError(f"Trading strategy position state 缺失: {ticker}")
        broker = record.get("broker") or {}
        entry_order_id = str(broker.get("entry_order_id") or "").strip()
        order = (orders.get("orders") or {}).get(entry_order_id)
        if not isinstance(order, dict) or not isinstance(order.get("frozen_params"), dict):
            raise RuntimeError(f"Trading position 缺少來源 entry order frozen params: {ticker}")
        file_path = csv_map.get(ticker)
        if not file_path:
            raise FileNotFoundError(f"Trading dataset 缺少持股 {ticker} CSV")
        params = build_params_from_mapping(order["frozen_params"])
        df = _load_position_market_frame(file_path=file_path, ticker=ticker, params=params, allowed_date=allowed_date)
        precomputed = generate_signals(df, params, ticker=ticker)
        atr_values, _buy_values, _sell_values, _limits = unpack_precomputed_signals(precomputed)

        entry_date = _normalize_date(broker.get("entry_date"))
        last_rollforward = _normalize_date(management.get("last_rollforward_date"))
        processed_dates: list[str] = []
        previous_stop_milli = int(position.get("sl_milli") or 0)
        previous_high_milli = int(position.get("highest_high_since_entry_milli") or 0)
        for idx, date_value in enumerate(df.index):
            date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
            if entry_date is not None and date_text < entry_date:
                continue
            if last_rollforward is not None and date_text <= last_rollforward:
                continue
            rollforward_position_management_from_completed_bar(
                position,
                completed_high=float(df["High"].iloc[idx]),
                completed_atr=float(atr_values[idx]),
                params=params,
                sync_display_fields=True,
            )
            processed_dates.append(date_text)

        if not processed_dates:
            continue
        processed_through = processed_dates[-1]
        updates[ticker] = {
            "position_state": position,
            "processed_through_date": processed_through,
            "processed_bar_count": len(processed_dates),
        }
        result_rows.append(
            {
                "ticker": ticker,
                "processed_bar_count": len(processed_dates),
                "processed_from_date": processed_dates[0],
                "processed_through_date": processed_through,
                "previous_stop_milli": previous_stop_milli,
                "stop_milli": int(position.get("sl_milli") or 0),
                "previous_highest_high_milli": previous_high_milli,
                "highest_high_since_entry_milli": int(position.get("highest_high_since_entry_milli") or 0),
            }
        )

    if not updates:
        return {
            "status": "UP_TO_DATE",
            "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "account_revision": int(account["revision"]),
            "processed_position_count": 0,
            "processed_bar_count": 0,
            "positions": [],
        }

    updated_account = rollforward_trading_strategy_management(
        root,
        updates=updates,
        expected_revision=int(account["revision"]),
    )
    protection_refresh_error = None
    try:
        build_trading_protection_plan(root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        protection_refresh_error = f"{type(exc).__name__}: {exc}"

    return {
        "status": "READY",
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "account_revision": int(updated_account["revision"]),
        "allowed_completed_date": allowed_date,
        "processed_position_count": len(result_rows),
        "processed_bar_count": sum(int(row["processed_bar_count"]) for row in result_rows),
        "positions": result_rows,
        "protection_plan_refresh_error": protection_refresh_error,
    }


__all__ = [
    "TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION",
    "build_trading_position_rollforward_snapshot",
    "run_trading_position_rollforward",
]
