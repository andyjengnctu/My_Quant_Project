"""Completed-daily-bar roll-forward for live Trading strategy positions.

This service advances only deterministic next-session management state from
completed Trading OHLCV.  It never infers broker fills, executes Stop/TP, or
uses current strategy params for an existing position; each holding is advanced
with immutable position-level strategy-lineage params, with source-entry order
params retained only as legacy compatibility fallback.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from core.params_io import build_params_from_mapping
from core.position_step import rollforward_position_management_from_completed_bar
from core.runtime_domains import RUNTIME_DOMAIN_TRADING
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trading_account_state import (
    MANAGEMENT_STATUS_ACTIVE,
    POSITION_SOURCE_STRATEGY_FILL,
)
from core.trading_market_clock import latest_allowed_completed_daily_date
from core.trading_order_state import active_trading_entry_orders
from core.trading_stop_exit_progress import build_trading_stop_exit_progress
from services.trading.account_state import (
    load_trading_account_state,
    rollforward_trading_strategy_management,
)
from services.trading.fill_reconciliation import recover_trading_fill_transaction
from services.trading.position_market_context import (
    load_trading_position_market_frame,
    load_trading_position_market_frames,
    normalize_trading_date,
    resolve_trading_strategy_position_sources,
)
from services.trading.protection_planning import build_trading_protection_plan
from services.trading.strategy_param_runtime import resolve_trading_position_strategy_binding

TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION = 1


def build_trading_position_rollforward_snapshot(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    account, orders, market_view = resolve_trading_strategy_position_sources(root)
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

    prepared: list[dict[str, Any]] = []
    params_by_ticker: dict[str, object] = {}
    for ticker in sorted(account.get("positions") or {}):
        record = account["positions"][ticker]
        if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
            continue
        management = record.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise RuntimeError(f"Trading strategy position management 非 active: {ticker}")
        broker = record.get("broker") or {}
        binding = resolve_trading_position_strategy_binding(record, orders=orders)
        lineage_key = str(binding.get("lineage_key") or "").strip()
        legacy_entry_order_id = str(binding.get("entry_order_id") or "").strip()
        stop_progress = build_trading_stop_exit_progress(
            orders,
            ticker=ticker,
            entry_order_id=lineage_key,
            compatible_entry_order_ids=([legacy_entry_order_id] if legacy_entry_order_id else None),
        )
        if bool(stop_progress.get("triggered")):
            prepared.append({
                "ticker": ticker,
                "broker": broker,
                "management": management,
                "stop_forced_exit": True,
            })
            continue
        if market_view is None:
            raise RuntimeError("Trading V2 position market view 尚未就緒")
        params = build_params_from_mapping(binding["frozen_params"])
        params_by_ticker[ticker] = params
        prepared.append({
            "ticker": ticker,
            "broker": broker,
            "management": management,
            "stop_forced_exit": False,
            "params": params,
        })

    market_frames: dict[str, pd.DataFrame] = {}
    if params_by_ticker:
        try:
            market_frames = load_trading_position_market_frames(
                view=market_view,
                params_by_ticker=params_by_ticker,
                allowed_date=allowed_date,
            )
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError):
            # Preserve the prior deterministic per-ticker error path when a
            # batch fragment is unhealthy; only the healthy read strategy is
            # optimized.
            market_frames = {
                ticker: load_trading_position_market_frame(
                    view=market_view,
                    ticker=ticker,
                    params=params_by_ticker[ticker],
                    allowed_date=allowed_date,
                )
                for ticker in params_by_ticker
            }

    rows: list[dict[str, Any]] = []
    for item in prepared:
        ticker = str(item["ticker"])
        broker = dict(item["broker"])
        management = dict(item["management"])
        if bool(item["stop_forced_exit"]):
            rows.append({
                "ticker": ticker,
                "entry_date": normalize_trading_date(broker.get("entry_date")),
                "last_rollforward_date": normalize_trading_date(management.get("last_rollforward_date")),
                "target_rollforward_date": None,
                "due": False,
                "stop_forced_exit": True,
            })
            continue
        df = market_frames[ticker]
        entry_date = normalize_trading_date(broker.get("entry_date"))
        last_rollforward = normalize_trading_date(management.get("last_rollforward_date"))
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
    account, orders, market_view = resolve_trading_strategy_position_sources(root)
    allowed_date = latest_allowed_completed_daily_date()
    if not account:
        return {
            "status": "NO_ACCOUNT",
            "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "processed_position_count": 0,
            "processed_bar_count": 0,
            "positions": [],
        }
    # Broker OMS is compatibility-only.  Direct account positions must continue
    # daily strategy rollforward even if a stale legacy ENTRY order remains.

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
        binding = resolve_trading_position_strategy_binding(record, orders=orders)
        lineage_key = str(binding.get("lineage_key") or "").strip()
        legacy_entry_order_id = str(binding.get("entry_order_id") or "").strip()
        stop_progress = build_trading_stop_exit_progress(
            orders,
            ticker=ticker,
            entry_order_id=lineage_key,
            compatible_entry_order_ids=([legacy_entry_order_id] if legacy_entry_order_id else None),
        )
        if bool(stop_progress.get("triggered")):
            continue
        if market_view is None:
            raise RuntimeError("Trading V2 position market view 尚未就緒")
        params = build_params_from_mapping(binding["frozen_params"])
        df = load_trading_position_market_frame(view=market_view, ticker=ticker, params=params, allowed_date=allowed_date)
        precomputed = generate_signals(df, params, ticker=ticker)
        atr_values, _buy_values, _sell_values, _limits = unpack_precomputed_signals(precomputed)

        entry_date = normalize_trading_date(broker.get("entry_date"))
        last_rollforward = normalize_trading_date(management.get("last_rollforward_date"))
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
