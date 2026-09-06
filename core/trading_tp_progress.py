"""Canonical live Trading TP-half progress across broker order attempts.

A TP_HALF strategy obligation is defined from the original confirmed entry
quantity.  Broker partial fills and explicit cancellation/retry may fragment
that obligation across multiple order records, but must never re-size the
strategy target from the already reduced remaining position.
"""
from __future__ import annotations

from typing import Any

from core.price_utils import calc_half_take_profit_sell_qty
from core.trading_identity import normalize_trading_ticker
from core.trading_order_state import (
    TRADING_ORDER_PURPOSE_PROTECTION_TP,
    validate_trading_order_state,
)


def build_trading_tp_half_progress(
    order_state: dict[str, Any],
    *,
    ticker: object,
    entry_order_id: object,
    initial_qty: int,
    tp_percent: float,
) -> dict[str, int]:
    """Return immutable target and cumulative confirmed TP fill progress."""

    validate_trading_order_state(order_state)
    ticker_key = normalize_trading_ticker(ticker)
    entry_id = str(entry_order_id or "").strip()
    if not ticker_key or not entry_id:
        raise ValueError("Trading TP progress 缺少 ticker/entry_order_id")
    initial_qty_int = int(initial_qty)
    if initial_qty_int <= 0:
        raise ValueError("Trading TP progress initial_qty 必須 > 0")

    target_qty = int(calc_half_take_profit_sell_qty(initial_qty_int, tp_percent))
    confirmed_qty = 0
    attempt_count = 0
    for row in (order_state.get("orders") or {}).values():
        if str(row.get("purpose") or "") != TRADING_ORDER_PURPOSE_PROTECTION_TP:
            continue
        if normalize_trading_ticker(row.get("ticker")) != ticker_key:
            continue
        if str(row.get("entry_order_id") or "").strip() != entry_id:
            continue
        attempt_count += 1
        confirmed_qty += int(row.get("filled_qty") or 0)

    remaining_qty = max(target_qty - confirmed_qty, 0)
    overfill_qty = max(confirmed_qty - target_qty, 0)
    return {
        "target_qty": target_qty,
        "confirmed_qty": confirmed_qty,
        "remaining_qty": remaining_qty,
        "overfill_qty": overfill_qty,
        "attempt_count": attempt_count,
    }


__all__ = ["build_trading_tp_half_progress"]
