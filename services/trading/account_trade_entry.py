"""Workbench-facing actual-account trade entry orchestration.

The accounting center may record an actual SELL against an inventory row.  This
service owns the small routing decision between a matching active broker order
and a standalone account-truth sell so UI panels do not duplicate order/fill
lineage rules.
"""
from __future__ import annotations

from core.trading_order_state import (
    TRADING_ACTIVE_ORDER_STATUSES,
    TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
    TRADING_ORDER_SIDE_SELL,
)
from services.trading.account_state import record_manual_trading_sell
from services.trading.fill_reconciliation import (
    confirm_trading_indicator_sell_order_fill,
    confirm_trading_protection_sell_order_fill,
)
from services.trading.order_state import get_trading_order_read_model
from services.trading.protection_planning import build_trading_protection_plan
from services.trading.indicator_exit_planning import build_trading_indicator_exit_plan


def list_active_sell_orders_for_ticker(project_root, ticker: object) -> dict:
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
    return {
        "revision": snapshot.get("revision"),
        "orders": rows,
    }


def record_trading_account_inventory_sell(
    project_root,
    *,
    ticker: object,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int,
    selected_order_id: str | None = None,
):
    ticker_key = str(ticker or "").strip().upper()
    order_state = list_active_sell_orders_for_ticker(project_root, ticker_key)
    active = list(order_state.get("orders") or [])

    selected = None
    if active:
        if selected_order_id:
            selected = next(
                (
                    row for row in active
                    if str(row.get("order_id") or "") == str(selected_order_id)
                    or str(row.get("broker_order_id") or "") == str(selected_order_id)
                ),
                None,
            )
            if selected is None:
                raise ValueError(f"指定的券商 SELL order 不在 {ticker_key} 目前 active orders 中")
        elif len(active) == 1:
            selected = active[0]
        else:
            labels = "、".join(str(row.get("broker_order_id") or row.get("order_id") or "-") for row in active)
            raise ValueError(f"{ticker_key} 有多筆 active SELL order（{labels}）；請先在帳務中心選擇實際成交的券商單")

    if selected is None:
        state = record_manual_trading_sell(
            project_root,
            ticker=ticker_key,
            qty=qty,
            price=price,
            trade_date=trade_date,
            expected_revision=expected_account_revision,
        )
        return {
            "route": "manual_account_sell",
            "account": state,
            "order": None,
        }

    remaining = int(selected.get("remaining_qty") or 0)
    if int(qty) > remaining:
        raise ValueError(f"本次成交股數不可超過券商單未成交股數 {remaining:,}")
    expected_order_revision = order_state.get("revision")
    if expected_order_revision is None:
        raise RuntimeError("Trading order revision 缺失，不能確認券商成交")
    order_id = str(selected.get("order_id") or "")
    if not order_id:
        raise RuntimeError("Trading active SELL order 缺少 order_id")

    if str(selected.get("purpose") or "") == TRADING_ORDER_PURPOSE_INDICATOR_EXIT:
        fill_fn = confirm_trading_indicator_sell_order_fill
    else:
        fill_fn = confirm_trading_protection_sell_order_fill
    result = fill_fn(
        project_root,
        order_id=order_id,
        fill_qty=int(qty),
        fill_price=price,
        trade_date=trade_date,
        expected_order_revision=int(expected_order_revision),
        expected_account_revision=int(expected_account_revision),
    )
    refresh_errors = {}
    try:
        build_trading_protection_plan(project_root)
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
        refresh_errors["protection"] = str(exc)
    try:
        build_trading_indicator_exit_plan(project_root)
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
        refresh_errors["indicator"] = str(exc)
    return {
        "route": "order_account_reconciliation",
        "account": result,
        "order": selected,
        "refresh_errors": refresh_errors,
    }


__all__ = [
    "list_active_sell_orders_for_ticker",
    "record_trading_account_inventory_sell",
]
