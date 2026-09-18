"""Automatic Trading lifecycle synchronization to the latest finalized state.

Pending orders and managed positions are two phases of the same frozen strategy
lineage.  This service advances both phases before Workbench read models are
rendered.  Historical broker/user facts (ticker, order date, fill date, fill
price/qty) are never rewritten; only derived planning/management state moves
forward.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.data_utils import get_required_min_rows
from core.entry_plans import build_cash_capped_entry_plan
from core.exact_accounting import build_buy_ledger, milli_to_money, price_to_milli
from core.params_io import build_params_from_mapping
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trade_lifecycle import build_prefill_lifecycle_timeline
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from services.trading.account_state import load_trading_account_state
from services.trading.market_data_consumer import (
    load_trading_v2_consumer_state,
    load_trading_v2_sanitized_ohlcv_frame,
    open_trading_v2_consumer_view,
)
from services.trading.pending_entry_state import (
    PENDING_ENTRY_STATUS_ACTIVE,
    load_trading_pending_entry_state,
    update_trading_pending_entry,
)
from services.trading.position_rollforward import (
    build_trading_position_rollforward_snapshot,
    run_trading_position_rollforward,
)

TRADING_LIFECYCLE_SYNC_SCHEMA_VERSION = 1
SYNC_STATUS_LATEST = "最新"
SYNC_STATUS_PENDING = "待同步"
SYNC_STATUS_FAILED = "同步失敗"


def _available_cash_for_pending(
    account: Mapping[str, Any] | None,
    active_entries: list[Mapping[str, Any]],
    current: Mapping[str, Any],
) -> float:
    if not isinstance(account, Mapping) or account.get("cash_milli") is None:
        raise RuntimeError("Trading cash 尚未設定，不能同步掛單規劃")
    current_id = str(current.get("pending_entry_id") or "")
    other_reserved = sum(
        int(row.get("reserved_cost_milli") or 0)
        for row in active_entries
        if str(row.get("pending_entry_id") or "") != current_id
    )
    available_milli = max(0, int(account.get("cash_milli") or 0) - other_reserved)
    return milli_to_money(available_milli)


def _build_pending_lifecycle_plan(entry: Mapping[str, Any]) -> dict[str, Any]:
    seed = dict(entry.get("execution_plan_seed") or {})
    return {
        "source": "trading_pending_frozen_lineage",
        "signal_date": entry.get("signal_date") or entry.get("information_date"),
        "information_date": entry.get("information_date"),
        "entry_type": seed.get("entry_type") or "normal",
        "limit_price": entry.get("limit_price"),
        "stop_price": entry.get("init_sl"),
        "init_trail": entry.get("init_trail"),
        "tp_price": entry.get("target_price"),
        "entry_atr": entry.get("entry_atr"),
        "ticker": entry.get("ticker"),
        "security_profile": deepcopy(seed.get("security_profile")),
        "planned_qty": entry.get("planned_qty"),
        "reserved_capital": entry.get("reserved_cost"),
    }


def _latest_shadow_state(
    project_root: Path,
    *,
    entry: Mapping[str, Any],
    latest_finalized_date: str,
    params,
) -> tuple[dict[str, Any] | None, str | None]:
    ticker = normalize_trading_ticker(entry.get("ticker"))
    view = open_trading_v2_consumer_view(project_root)
    frame = load_trading_v2_sanitized_ohlcv_frame(
        view,
        ticker=ticker,
        through_date=latest_finalized_date,
        min_rows=get_required_min_rows(params),
    )
    if frame.empty:
        raise RuntimeError(f"{ticker} 沒有可用 Trading adjusted 日K")
    signals = generate_signals(frame, params, ticker=ticker)
    atr_values, _buy_values, sell_values, _limits = unpack_precomputed_signals(signals)
    plan = _build_pending_lifecycle_plan(entry)
    timeline = build_prefill_lifecycle_timeline(
        date_labels=list(frame.index),
        open_values=list(frame["Open"]),
        high_values=list(frame["High"]),
        low_values=list(frame["Low"]),
        close_values=list(frame["Close"]),
        volume_values=list(frame["Volume"]) if "Volume" in frame.columns else None,
        atr_values=list(atr_values),
        sell_signals=list(sell_values),
        plan=plan,
        params=params,
    )
    if not timeline:
        return None, None
    latest_index = max(timeline)
    row = dict(timeline[latest_index])
    row_date = pd.Timestamp(frame.index[int(latest_index)]).strftime("%Y-%m-%d")
    return row, row_date


def _sync_one_pending(
    project_root: Path,
    *,
    entry: Mapping[str, Any],
    active_entries: list[Mapping[str, Any]],
    account: Mapping[str, Any] | None,
    latest_finalized_date: str,
) -> dict[str, Any]:
    entry_id = str(entry.get("pending_entry_id") or "")
    ticker = normalize_trading_ticker(entry.get("ticker"))
    evaluated = normalize_trading_date(
        entry.get("evaluated_through_date") or entry.get("information_date"),
        field_name="pending_entry.evaluated_through_date",
        allow_none=False,
    )
    if evaluated >= latest_finalized_date:
        return {"pending_entry_id": entry_id, "ticker": ticker, "status": SYNC_STATUS_LATEST, "changed": False}

    lineage = dict(entry.get("management_lineage") or {})
    frozen_params = lineage.get("frozen_params")
    if not isinstance(frozen_params, Mapping):
        raise RuntimeError(f"{ticker} 掛單缺少 frozen params")
    params = build_params_from_mapping(dict(frozen_params))
    shadow_row, shadow_date = _latest_shadow_state(
        project_root,
        entry=entry,
        latest_finalized_date=latest_finalized_date,
        params=params,
    )

    replacement = deepcopy(dict(entry))
    seed = deepcopy(dict(replacement.get("execution_plan_seed") or {}))
    if shadow_row is not None:
        shadow_state = dict(shadow_row.get("shadow_position_state") or {})
        if shadow_state.get("pending_exit_action") in {"STOP", "TP_HALF"}:
            raise RuntimeError(
                f"{ticker} frozen lineage 已於 {shadow_date or '-'} 觸發 shadow exit；ACTIVE 掛單需確認後結案"
            )
        if shadow_row.get("limit_price") is not None:
            seed["limit_price"] = float(shadow_row["limit_price"])
        if shadow_row.get("stop_price") is not None:
            seed["init_sl"] = float(shadow_row["stop_price"])
        if shadow_row.get("tp_price") is not None:
            seed["target_price"] = float(shadow_row["tp_price"])
        if shadow_state:
            seed["shadow_position_state"] = deepcopy(shadow_state)
            if shadow_state.get("trailing_stop") is not None:
                seed["init_trail"] = float(shadow_state["trailing_stop"])

    seed["trade_date"] = latest_finalized_date
    seed["planned_trade_date"] = entry.get("planned_trade_date")
    # Existing reservations own their slot/cash.  Automatic sync may tighten or
    # reduce a plan but must never steal additional allocation from sibling
    # pending orders; cap the refreshed sizing at the currently reserved qty.
    seed["max_qty"] = min(
        int(seed.get("max_qty") or int(entry.get("planned_qty") or 0)),
        int(entry.get("planned_qty") or 0),
    )
    available_cash = _available_cash_for_pending(account, active_entries, entry)
    refreshed = build_cash_capped_entry_plan(seed, available_cash, params)
    if refreshed is None or int(refreshed.get("qty") or 0) <= 0:
        raise RuntimeError(f"{ticker} 依 latest finalized state 已無可執行規劃股數")

    user_qty_override = seed.get("user_qty_override")
    if user_qty_override is not None:
        try:
            requested_qty = max(1, int(user_qty_override))
        except (TypeError, ValueError):
            requested_qty = int(refreshed["qty"])
        refreshed["qty"] = min(int(refreshed["qty"]), requested_qty)
        refreshed["reserved_cost_milli"] = int(
            build_buy_ledger(price_to_milli(refreshed["limit_price"]), int(refreshed["qty"]), params)["cash_buy_total_milli"]
        )
        refreshed["reserved_cost"] = milli_to_money(int(refreshed["reserved_cost_milli"]))

    replacement["execution_plan_seed"] = deepcopy(refreshed)
    replacement["planned_qty"] = int(refreshed["qty"])
    replacement["reserved_cost_milli"] = int(refreshed["reserved_cost_milli"])
    replacement["reserved_cost"] = float(refreshed["reserved_cost"])
    replacement["limit_price"] = float(refreshed["limit_price"])
    replacement["init_sl"] = float(refreshed["init_sl"])
    replacement["init_trail"] = float(refreshed["init_trail"])
    replacement["target_price"] = float(refreshed["target_price"])
    replacement["evaluated_through_date"] = latest_finalized_date
    replacement["sync_error"] = None
    updated = update_trading_pending_entry(
        project_root,
        pending_entry_id=entry_id,
        replacement=replacement,
    )
    return {
        "pending_entry_id": entry_id,
        "ticker": ticker,
        "status": SYNC_STATUS_LATEST,
        "changed": True,
        "evaluated_through_date": updated.get("evaluated_through_date"),
    }


def run_trading_lifecycle_sync(project_root: str | Path) -> dict[str, Any]:
    """Idempotently advance pending + position lifecycle state before reads."""

    root = Path(project_root).resolve()
    consumer_state = load_trading_v2_consumer_state(root, required=False)
    if consumer_state is None:
        return {
            "schema_version": TRADING_LIFECYCLE_SYNC_SCHEMA_VERSION,
            "latest_finalized_date": None,
            "status": SYNC_STATUS_PENDING,
            "pending_results": [],
            "pending_errors": {},
            "position_result": {"status": "NO_DATA"},
            "position_errors": {},
            "position_status_by_ticker": {},
            "position_due_tickers": [],
        }
    latest_finalized_date = normalize_trading_date(
        consumer_state.get("market_date"), field_name="latest_finalized_date", allow_none=False
    )

    pending_state = load_trading_pending_entry_state(root, required=False)
    active_entries = [] if pending_state is None else [
        deepcopy(row)
        for row in (pending_state.get("entries") or {}).values()
        if str(row.get("status") or "") == PENDING_ENTRY_STATUS_ACTIVE
    ]
    active_entries.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("pending_entry_id") or "")))
    account = load_trading_account_state(root, required=False)
    pending_results: list[dict[str, Any]] = []
    pending_errors: dict[str, str] = {}
    for entry in active_entries:
        entry_id = str(entry.get("pending_entry_id") or "")
        ticker = str(entry.get("ticker") or "")
        try:
            result = _sync_one_pending(
                root,
                entry=entry,
                active_entries=active_entries,
                account=account,
                latest_finalized_date=latest_finalized_date,
            )
            pending_results.append(result)
            if bool(result.get("changed")):
                # Later siblings must see the updated reservation amount.
                for index, current in enumerate(active_entries):
                    if str(current.get("pending_entry_id") or "") == entry_id:
                        latest_state = load_trading_pending_entry_state(root, required=True)
                        active_entries[index] = deepcopy(latest_state["entries"][entry_id])
                        break
        except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            pending_errors[entry_id] = f"{type(exc).__name__}: {exc}"
            pending_results.append({
                "pending_entry_id": entry_id,
                "ticker": ticker,
                "status": SYNC_STATUS_FAILED,
                "changed": False,
                "error": pending_errors[entry_id],
            })

    position_error = None
    try:
        position_result = run_trading_position_rollforward(root)
    except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        position_error = f"{type(exc).__name__}: {exc}"
        position_result = {"status": SYNC_STATUS_FAILED, "error": position_error}

    try:
        post_rollforward = build_trading_position_rollforward_snapshot(root)
    except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        post_rollforward = {"due_tickers": []}
        if position_error is None:
            position_error = f"{type(exc).__name__}: {exc}"

    due_tickers = {str(value) for value in list(post_rollforward.get("due_tickers") or [])}
    account_after = load_trading_account_state(root, required=False) or {}
    position_status_by_ticker: dict[str, str] = {}
    position_errors: dict[str, str] = {}
    for ticker in sorted((account_after.get("positions") or {}).keys()):
        if position_error is not None:
            position_status_by_ticker[ticker] = SYNC_STATUS_FAILED
            position_errors[ticker] = position_error
        elif ticker in due_tickers:
            position_status_by_ticker[ticker] = SYNC_STATUS_PENDING
        else:
            position_status_by_ticker[ticker] = SYNC_STATUS_LATEST

    overall = SYNC_STATUS_LATEST
    if pending_errors or position_error:
        overall = SYNC_STATUS_FAILED
    elif due_tickers:
        overall = SYNC_STATUS_PENDING
    return {
        "schema_version": TRADING_LIFECYCLE_SYNC_SCHEMA_VERSION,
        "latest_finalized_date": latest_finalized_date,
        "status": overall,
        "pending_results": pending_results,
        "pending_errors": pending_errors,
        "position_result": position_result,
        "position_errors": position_errors,
        "position_status_by_ticker": position_status_by_ticker,
        "position_due_tickers": sorted(due_tickers),
    }


__all__ = [
    "SYNC_STATUS_FAILED",
    "SYNC_STATUS_LATEST",
    "SYNC_STATUS_PENDING",
    "TRADING_LIFECYCLE_SYNC_SCHEMA_VERSION",
    "run_trading_lifecycle_sync",
]
