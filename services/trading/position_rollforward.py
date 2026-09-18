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

from core.entry_plans import build_position_from_entry_fill
from core.exact_accounting import milli_to_price
from core.params_io import build_params_from_mapping
from core.position_step import (
    resolve_position_intraday_exit_hits,
    rollforward_position_management_from_completed_bar,
)
from core.runtime_domains import RUNTIME_DOMAIN_TRADING
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trading_account_state import (
    MANAGEMENT_STATUS_ACTIVE,
    MANAGEMENT_SELL_SIGNAL_INDICATOR,
    MANAGEMENT_SELL_SIGNAL_STOP,
    MANAGED_POSITION_SOURCES,
    POSITION_SOURCE_MANUAL_ADOPTED,
    POSITION_SOURCE_MANUAL_MANAGED,
)
from core.trading_market_clock import latest_allowed_completed_daily_date
from core.trading_order_state import active_trading_entry_orders
from core.trading_stop_exit_progress import build_trading_stop_exit_progress
from services.trading.account_state import (
    activate_existing_manual_trading_position_management,
    load_trading_account_state,
    record_trading_strategy_management_sell_signals,
    rollforward_trading_strategy_management,
)
from services.trading.account_trade_entry import build_manual_adopted_management_context
from services.trading.fill_reconciliation import recover_trading_fill_transaction
from services.trading.position_market_context import (
    load_trading_position_market_frame,
    load_trading_position_market_frames,
    normalize_trading_date,
    resolve_trading_strategy_position_sources,
)
from services.trading.protection_planning import build_trading_protection_plan
from services.trading.strategy_param_runtime import resolve_trading_position_management_binding

TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION = 1


def _build_initial_management_replay_position(
    record: dict[str, Any],
    *,
    binding: dict[str, Any],
    params,
) -> dict[str, Any]:
    broker = dict(record.get("broker") or {})
    management = dict(record.get("strategy_management") or {})
    current = dict(management.get("position_state") or {})
    seed = dict(binding.get("execution_plan_seed") or {})
    initial_qty = int(broker.get("initial_qty") or broker.get("qty") or 0)
    if initial_qty <= 0:
        raise RuntimeError(f"Trading managed position initial_qty 不合法: {record.get('ticker')}")
    entry_price = current.get("entry_fill_price")
    if entry_price is None:
        gross_milli = int(
            broker.get("initial_gross_buy_milli")
            or broker.get("initial_cost_basis_milli")
            or 0
        )
        if gross_milli <= 0:
            raise RuntimeError(f"Trading managed position 缺少 entry fill evidence: {record.get('ticker')}")
        entry_price = milli_to_price((gross_milli + initial_qty // 2) // initial_qty)
    entry_date = normalize_trading_date(broker.get("entry_date"))
    if entry_date is None:
        raise RuntimeError(f"Trading managed position 缺少 entry_date: {record.get('ticker')}")
    return build_position_from_entry_fill(
        buy_price=float(entry_price),
        qty=initial_qty,
        params=params,
        entry_type=str(seed.get("entry_type") or current.get("entry_type") or "normal"),
        init_sl=seed.get("init_sl", current.get("initial_stop")),
        init_trail=seed.get("init_trail", current.get("initial_stop")),
        target_price=seed.get("target_price", current.get("tp_half")),
        limit_price=seed.get("limit_price", current.get("limit_price")),
        entry_atr=seed.get("entry_atr", current.get("entry_atr")),
        target_reference_price=seed.get("target_reference_price"),
        ticker=str(record.get("ticker") or ""),
        security_profile=seed.get("security_profile", current.get("security_profile")),
        trade_date=entry_date,
    )


def _replay_position_sell_obligation(
    record: dict[str, Any],
    *,
    binding: dict[str, Any],
    frame: pd.DataFrame,
    params,
) -> dict[str, Any] | None:
    """Replay frozen management to the first canonical full-exit obligation.

    STOP uses the stop already active at the start of each session; only after
    that check is the completed bar allowed to tighten next-session trailing.
    This is the same timing contract as ``execute_bar_step`` and prevents a
    same-bar High from manufacturing a stop that did not exist when the Low
    traded.
    """
    management = dict(record.get("strategy_management") or {})
    existing = str(management.get("sell_signal") or "").strip()
    if existing:
        return None
    broker = dict(record.get("broker") or {})
    entry_date = normalize_trading_date(broker.get("entry_date"))
    management_start = normalize_trading_date(management.get("management_start_date"))
    effective_start = max(
        [value for value in (entry_date, management_start) if value is not None],
        default=None,
    )
    if effective_start is None or frame.empty:
        return None

    position = _build_initial_management_replay_position(record, binding=binding, params=params)
    atr_values, _buy_values, sell_values, _limits = unpack_precomputed_signals(
        generate_signals(frame, params, ticker=str(record.get("ticker") or ""))
    )
    for idx, date_value in enumerate(frame.index):
        date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        if date_text < effective_start:
            continue

        # AI: Trading account forbids same-session buy+sell. A management
        # takeover also starts after that completed bar, so its first executable
        # STOP session is strictly after the effective start date.
        if date_text > effective_start:
            stop_hit, _tp_hit = resolve_position_intraday_exit_hits(
                position,
                t_high=float(frame["High"].iloc[idx]),
                t_low=float(frame["Low"].iloc[idx]),
                params=params,
            )
            if stop_hit:
                return {
                    "sell_signal": MANAGEMENT_SELL_SIGNAL_STOP,
                    "sell_signal_date": date_text,
                    "sell_signal_trigger_price_milli": int(position["sl_milli"]),
                }

        rollforward_position_management_from_completed_bar(
            position,
            completed_high=float(frame["High"].iloc[idx]),
            completed_atr=float(atr_values[idx]),
            params=params,
            sync_display_fields=True,
        )
        if bool(sell_values[idx]):
            return {
                "sell_signal": MANAGEMENT_SELL_SIGNAL_INDICATOR,
                "sell_signal_date": date_text,
                "sell_signal_trigger_price_milli": None,
            }
    return None


def reconcile_trading_manual_position_management(project_root: str | Path) -> dict[str, Any]:
    """Normalize development-era manual holdings to one managed product contract.

    Broker quantity/cost/cash are never changed.  ``manual_adopted`` positions,
    plus prior compatibility activations whose management started later than the
    broker entry date, are rebuilt from the original entry date and advanced to
    the current finalized date with the same canonical management primitives.
    """
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    account = load_trading_account_state(root, required=False)
    if not account:
        return {"status": "NO_ACCOUNT", "promoted": [], "errors": {}}

    promoted: list[str] = []
    errors: dict[str, str] = {}
    state = account
    for ticker in sorted(state.get("positions") or {}):
        record = (state.get("positions") or {}).get(ticker)
        if not isinstance(record, dict):
            continue
        source = str(record.get("source") or "")
        broker = dict(record.get("broker") or {})
        entry_date = normalize_trading_date(broker.get("entry_date"))
        management = dict(record.get("strategy_management") or {})
        lineage = dict(record.get("management_lineage") or {})
        management_start = normalize_trading_date(management.get("management_start_date"))
        needs_normalization = source == POSITION_SOURCE_MANUAL_ADOPTED
        if source == POSITION_SOURCE_MANUAL_MANAGED:
            lineage_origin = str(lineage.get("origin") or "")
            legacy_origin = lineage_origin in {"manual_adopted_management", "manual_managed"}
            needs_normalization = bool(
                legacy_origin
                and entry_date is not None
                and management_start != entry_date
            )
        if not needs_normalization:
            continue
        try:
            context = build_manual_adopted_management_context(
                root,
                ticker=str(ticker),
                broker=broker,
            )
            state = activate_existing_manual_trading_position_management(
                root,
                ticker=str(ticker),
                management_lineage=dict(context["management_lineage"]),
                position_state=dict(context["position_state"]),
                management_start_date=context["management_start_date"],
                last_rollforward_date=context.get("last_rollforward_date"),
                initial_position_state=dict(context["initial_position_state"]),
                expected_revision=int(state["revision"]),
            )
            promoted.append(str(ticker))
        except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            errors[str(ticker)] = f"{type(exc).__name__}: {exc}"
            # Refresh after a failed optimistic mutation attempt so a later
            # ticker never reuses a stale revision.
            latest = load_trading_account_state(root, required=False)
            if latest is not None:
                state = latest
    return {
        "status": "OK" if not errors else ("PARTIAL" if promoted else "NO_CHANGE"),
        "promoted": promoted,
        "promoted_count": len(promoted),
        "errors": errors,
    }


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
            "managed_position_count": 0,
            "due_count": 0,
            "due_tickers": [],
            "positions": [],
        }

    prepared: list[dict[str, Any]] = []
    params_by_ticker: dict[str, object] = {}
    for ticker in sorted(account.get("positions") or {}):
        record = account["positions"][ticker]
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            continue
        management = record.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise RuntimeError(f"Trading managed position management 非 active: {ticker}")
        broker = record.get("broker") or {}
        management_sell_signal = str(management.get("sell_signal") or "").strip()
        if management_sell_signal:
            prepared.append({
                "ticker": ticker,
                "broker": broker,
                "management": management,
                "stop_forced_exit": False,
                "decision_obligation": True,
            })
            continue
        binding = resolve_trading_position_management_binding(record, orders=orders)
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
        if bool(item.get("decision_obligation")):
            rows.append({
                "ticker": ticker,
                "entry_date": normalize_trading_date(broker.get("entry_date")),
                "last_rollforward_date": normalize_trading_date(management.get("last_rollforward_date")),
                "target_rollforward_date": None,
                "due": False,
                "sell_signal": str(management.get("sell_signal") or ""),
                "sell_signal_date": normalize_trading_date(management.get("sell_signal_date")),
                "decision_obligation": True,
            })
            continue
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
        management_start_date = normalize_trading_date(management.get("management_start_date"))
        last_rollforward = normalize_trading_date(management.get("last_rollforward_date"))
        eligible = df.index
        effective_start = max(
            [value for value in (entry_date, management_start_date) if value is not None],
            default=None,
        )
        if effective_start is not None:
            eligible = eligible[eligible >= pd.Timestamp(effective_start)]
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
        "managed_position_count": len(rows),
        "due_count": len(due_tickers),
        "due_tickers": due_tickers,
        "positions": rows,
    }


def run_trading_position_rollforward(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    manual_management_reconcile = reconcile_trading_manual_position_management(root)
    account, orders, market_view = resolve_trading_strategy_position_sources(root)
    allowed_date = latest_allowed_completed_daily_date()
    if not account:
        return {
            "status": "NO_ACCOUNT",
            "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "manual_management_reconcile": manual_management_reconcile,
            "processed_position_count": 0,
            "processed_bar_count": 0,
            "positions": [],
        }
    # Broker OMS is compatibility-only.  Direct account positions must continue
    # daily strategy rollforward even if a stale legacy ENTRY order remains.

    sell_signal_updates: dict[str, dict[str, Any]] = {}
    for ticker in sorted(account.get("positions") or {}):
        record = account["positions"][ticker]
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            continue
        management = dict(record.get("strategy_management") or {})
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise RuntimeError(f"Trading managed position management 非 active: {ticker}")
        if str(management.get("sell_signal") or "").strip():
            continue
        position_state = dict(management.get("position_state") or {})
        binding = resolve_trading_position_management_binding(record, orders=orders)
        lineage_key = str(binding.get("lineage_key") or "").strip()
        legacy_entry_order_id = str(binding.get("entry_order_id") or "").strip()
        stop_progress = build_trading_stop_exit_progress(
            orders,
            ticker=ticker,
            entry_order_id=lineage_key,
            compatible_entry_order_ids=([legacy_entry_order_id] if legacy_entry_order_id else None),
        )
        if bool(stop_progress.get("triggered")) and stop_progress.get("trigger_trade_date"):
            stop_milli = int(position_state.get("sl_milli") or 0)
            if stop_milli <= 0:
                raise RuntimeError(f"Trading STOP EXIT 缺少有效停損: {ticker}")
            sell_signal_updates[ticker] = {
                "sell_signal": MANAGEMENT_SELL_SIGNAL_STOP,
                "sell_signal_date": str(stop_progress["trigger_trade_date"]),
                "sell_signal_trigger_price_milli": stop_milli,
            }
            continue
        if market_view is None:
            raise RuntimeError("Trading V2 position market view 尚未就緒")
        params = build_params_from_mapping(binding["frozen_params"])
        frame = load_trading_position_market_frame(
            view=market_view,
            ticker=ticker,
            params=params,
            allowed_date=allowed_date,
        )
        detected = _replay_position_sell_obligation(
            record,
            binding=binding,
            frame=frame,
            params=params,
        )
        if detected is not None:
            sell_signal_updates[ticker] = detected

    signal_updates_written = bool(sell_signal_updates)
    if signal_updates_written:
        account = record_trading_strategy_management_sell_signals(
            root,
            signals=sell_signal_updates,
            expected_revision=int(account["revision"]),
        )
        account, orders, market_view = resolve_trading_strategy_position_sources(root)

    updates: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    for ticker in sorted(account.get("positions") or {}):
        record = account["positions"][ticker]
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            continue
        management = record.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise RuntimeError(f"Trading managed position management 非 active: {ticker}")
        if str(management.get("sell_signal") or "").strip():
            continue
        position = deepcopy(management.get("position_state"))
        if not isinstance(position, dict):
            raise RuntimeError(f"Trading managed position state 缺失: {ticker}")
        broker = record.get("broker") or {}
        binding = resolve_trading_position_management_binding(record, orders=orders)
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
        management_start_date = normalize_trading_date(management.get("management_start_date"))
        effective_start = max(
            [value for value in (entry_date, management_start_date) if value is not None],
            default=None,
        )
        last_rollforward = normalize_trading_date(management.get("last_rollforward_date"))
        processed_dates: list[str] = []
        previous_stop_milli = int(position.get("sl_milli") or 0)
        previous_high_milli = int(position.get("highest_high_since_entry_milli") or 0)
        for idx, date_value in enumerate(df.index):
            date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
            if effective_start is not None and date_text < effective_start:
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
            "status": "READY" if signal_updates_written else "UP_TO_DATE",
            "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "manual_management_reconcile": manual_management_reconcile,
            "account_revision": int(account["revision"]),
            "processed_position_count": 0,
            "processed_bar_count": 0,
            "positions": [],
            "sell_signal_count": len(sell_signal_updates),
            "sell_signal_tickers": sorted(sell_signal_updates),
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
        "manual_management_reconcile": manual_management_reconcile,
        "account_revision": int(updated_account["revision"]),
        "allowed_completed_date": allowed_date,
        "processed_position_count": len(result_rows),
        "processed_bar_count": sum(int(row["processed_bar_count"]) for row in result_rows),
        "positions": result_rows,
        "sell_signal_count": len(sell_signal_updates),
        "sell_signal_tickers": sorted(sell_signal_updates),
        "protection_plan_refresh_error": protection_refresh_error,
    }


__all__ = [
    "TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION",
    "reconcile_trading_manual_position_management",
    "build_trading_position_rollforward_snapshot",
    "run_trading_position_rollforward",
]
