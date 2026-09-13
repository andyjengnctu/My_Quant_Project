"""Workbench-facing actual-account trade entry orchestration.

Workbench is a decision/accounting tool, not a broker OMS.  The user executes at
his broker and records the actual fill here.  Scanner-selected BUYs preserve the
strategy-management lineage; arbitrary BUYs remain manual account truth.  SELLs
are recorded directly against the selected broker inventory.
"""
from __future__ import annotations

from core.trading_order_state import TRADING_ACTIVE_ORDER_STATUSES, TRADING_ORDER_SIDE_SELL
from services.trading.account_state import (
    record_manual_trading_buy,
    record_manual_trading_sell,
    record_strategy_trading_buy,
)
from services.trading.order_state import get_trading_order_read_model
from services.trading.strategy_param_runtime import resolve_trading_candidate_frozen_params


def list_active_sell_orders_for_ticker(project_root, ticker: object) -> dict:
    """Legacy read-only helper retained for compatibility; the new UI does not use it."""
    ticker_key = str(ticker or "").strip().upper()
    snapshot = get_trading_order_read_model(project_root)
    rows = []
    for row in list(snapshot.get("orders") or []):
        if str(row.get("ticker") or "").strip().upper() != ticker_key:
            continue
        if str(row.get("side") or "").strip().upper() != TRADING_ORDER_SIDE_SELL:
            continue
        if str(row.get("status") or "") not in TRADING_ACTIVE_ORDER_STATUSES:
            continue
        rows.append(dict(row))
    return {"revision": snapshot.get("revision"), "orders": rows}


def record_trading_account_buy(
    project_root,
    *,
    ticker: object,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int | None = None,
    candidate: dict | None = None,
):
    ticker_key = str(ticker or "").strip().upper()
    if candidate is not None:
        candidate_row = dict(candidate)
        candidate_ticker = str(candidate_row.get("ticker") or "").strip().upper()
        if candidate_ticker != ticker_key:
            raise ValueError("選取的 Scanner candidate 與成交股票不一致")
        params, member = resolve_trading_candidate_frozen_params(candidate_row)
        # Strategy geometry is float-based while exact account ledgers accept
        # decimal-like inputs.  Normalize only at this strategy/account boundary
        # so UI Decimal input never leaks into stop/risk arithmetic.
        strategy_price = float(price)
        account = record_strategy_trading_buy(
            project_root,
            ticker=ticker_key,
            qty=int(qty),
            price=strategy_price,
            trade_date=trade_date,
            expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
            params=params,
            execution_plan_seed=dict(candidate_row.get("execution_plan_seed") or {}),
        )
        return {
            "route": "scanner_strategy_buy",
            "account": account,
            "strategy_member": dict(member or {}),
        }
    account = record_manual_trading_buy(
        project_root,
        ticker=ticker_key,
        qty=int(qty),
        price=price,
        trade_date=trade_date,
        expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
    )
    return {"route": "manual_account_buy", "account": account}


def record_trading_account_inventory_sell(
    project_root,
    *,
    ticker: object,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int | None = None,
    selected_order_id: str | None = None,
):
    """Record actual broker SELL directly; broker-order lifecycle is not required."""
    ticker_key = str(ticker or "").strip().upper()
    account = record_manual_trading_sell(
        project_root,
        ticker=ticker_key,
        qty=int(qty),
        price=price,
        trade_date=trade_date,
        expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
    )
    return {
        "route": "direct_account_sell",
        "account": account,
        "order": None,
        "refresh_errors": {},
    }


__all__ = [
    "list_active_sell_orders_for_ticker",
    "record_trading_account_buy",
    "record_trading_account_inventory_sell",
]
