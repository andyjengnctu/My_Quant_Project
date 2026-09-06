"""Canonical live Trading STOP-trigger progress across broker order attempts.

Once any confirmed fill exists on a strategy PROTECTION_STOP order, the STOP
has already triggered.  If the strategy position still has shares, that
remaining quantity is a persistent forced-exit obligation.  Cancellation or
restart must never turn it back into a fresh trigger-waiting STOP order.
"""
from __future__ import annotations

from typing import Any

from core.file_integrity import canonical_json_sha256
from core.trading_identity import normalize_trading_ticker
from core.trading_order_state import (
    TRADING_ACTIVE_ORDER_STATUSES,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
    validate_trading_order_state,
)


def build_trading_stop_exit_progress(
    order_state: dict[str, Any],
    *,
    ticker: object,
    entry_order_id: object,
) -> dict[str, Any]:
    """Return persistent STOP-trigger / forced-exit progress for one entry lineage."""

    validate_trading_order_state(order_state)
    ticker_key = normalize_trading_ticker(ticker)
    entry_id = str(entry_order_id or "").strip()
    if not ticker_key or not entry_id:
        raise ValueError("Trading STOP progress 缺少 ticker/entry_order_id")

    trigger_fills: list[tuple[str, str, str, str]] = []
    total_confirmed_qty = 0
    forced_attempt_count = 0
    active_original: list[dict[str, Any]] = []
    active_forced: list[dict[str, Any]] = []

    for raw in (order_state.get("orders") or {}).values():
        purpose = str(raw.get("purpose") or "")
        if purpose not in {
            TRADING_ORDER_PURPOSE_PROTECTION_STOP,
            TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
        }:
            continue
        if normalize_trading_ticker(raw.get("ticker")) != ticker_key:
            continue
        if str(raw.get("entry_order_id") or "").strip() != entry_id:
            continue
        row = dict(raw)
        total_confirmed_qty += int(row.get("filled_qty") or 0)
        if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER:
            forced_attempt_count += 1
        if str(row.get("status") or "") in TRADING_ACTIVE_ORDER_STATUSES:
            if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP:
                active_original.append(row)
            else:
                active_forced.append(row)
        if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP:
            for fill in list(row.get("fills") or []):
                trigger_fills.append(
                    (
                        str(fill.get("confirmed_at") or ""),
                        str(row.get("order_id") or ""),
                        str(fill.get("fill_id") or ""),
                        str(fill.get("trade_date") or ""),
                    )
                )

    if len(active_original) > 1 or len(active_forced) > 1 or (active_original and active_forced):
        raise RuntimeError(
            f"Trading {ticker_key} STOP exit 同時存在多個 active broker orders，必須先 reconcile"
        )

    if not trigger_fills:
        return {
            "triggered": False,
            "forced_exit_key": None,
            "trigger_order_id": None,
            "trigger_fill_id": None,
            "trigger_trade_date": None,
            "confirmed_stop_exit_qty": int(total_confirmed_qty),
            "forced_attempt_count": int(forced_attempt_count),
            "active_original_stop_order_ids": [str(row.get("order_id") or "") for row in active_original],
            "active_forced_exit_order_ids": [str(row.get("order_id") or "") for row in active_forced],
            "active_exit_order_ids": [str(row.get("order_id") or "") for row in [*active_original, *active_forced]],
            "active_exit_remaining_qty": sum(int(row.get("remaining_qty") or 0) for row in [*active_original, *active_forced]),
        }

    trigger_fills.sort()
    _confirmed_at, trigger_order_id, trigger_fill_id, trigger_trade_date = trigger_fills[0]
    forced_exit_key = canonical_json_sha256(
        {
            "ticker": ticker_key,
            "entry_order_id": entry_id,
            "trigger_order_id": trigger_order_id,
            "trigger_fill_id": trigger_fill_id,
            "trigger_trade_date": trigger_trade_date,
        }
    )
    active_rows = [*active_original, *active_forced]
    return {
        "triggered": True,
        "forced_exit_key": forced_exit_key,
        "trigger_order_id": trigger_order_id,
        "trigger_fill_id": trigger_fill_id,
        "trigger_trade_date": trigger_trade_date,
        "confirmed_stop_exit_qty": int(total_confirmed_qty),
        "forced_attempt_count": int(forced_attempt_count),
        "active_original_stop_order_ids": [str(row.get("order_id") or "") for row in active_original],
        "active_forced_exit_order_ids": [str(row.get("order_id") or "") for row in active_forced],
        "active_exit_order_ids": [str(row.get("order_id") or "") for row in active_rows],
        "active_exit_remaining_qty": sum(int(row.get("remaining_qty") or 0) for row in active_rows),
    }


__all__ = ["build_trading_stop_exit_progress"]
