"""Single owner of position-management timing and exit decisions.

AI: Execution adapters acknowledge a decision only after applying a simulated or
confirmed fill. This module never writes cash, cost basis or broker inventory.
Research, live replay and pre-fill shadow all use this same ordered transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import pandas as pd

from core.exit_priority import resolve_stop_tp_hits
from core.exact_accounting import milli_to_price, price_to_milli
from core.price_utils import (
    adjust_long_sell_fill_price, adjust_long_stop_price,
    calc_half_take_profit_sell_qty, get_exit_sell_block_reason,
)

# AI: Management projection is deliberately separate from fill/ledger fields.
POSITION_MANAGEMENT_FIELDS = (
    "sl_milli", "initial_stop_milli", "trailing_stop_milli", "tp_half_milli",
    "sl", "initial_stop", "trailing_stop", "tp_half", "sold_half",
    "highest_high_since_entry_milli", "highest_high_since_entry",
    "pending_exit_action", "pending_exit_trigger_price", "pending_exit_remaining_qty",
    "entry_day_stop_triggered", "entry_day_tp_triggered",
    "inherited_shadow_management",
    "shadow_entry_fill_price_milli", "shadow_entry_fill_price", "shadow_sold_half",
)

@dataclass(frozen=True)
class PositionExitDecision:
    event: str
    qty: int
    reference_price: float | None
    trigger_price: float | None = None
    deferred: bool = False
    trade_date: object = None
    block_reason: str | None = None

    @property
    def executable(self) -> bool:
        return self.block_reason is None and self.reference_price is not None and self.qty > 0


def _sync_trailing_stop_display_fields(position):
    position['highest_high_since_entry'] = milli_to_price(position['highest_high_since_entry_milli'])
    position['trailing_stop'] = milli_to_price(position['trailing_stop_milli'])
    position['sl'] = milli_to_price(position['sl_milli'])


def _update_trailing_stop(position, *, y_high, y_atr, params, sync_display_fields=True):
    previous_high_milli = int(position.get('highest_high_since_entry_milli', position['entry_fill_price_milli']))
    highest_high_milli = previous_high_milli
    made_new_high = False

    if not pd.isna(y_high):
        y_high_milli = price_to_milli(y_high)
        if y_high_milli > previous_high_milli:
            highest_high_milli = y_high_milli
            made_new_high = True

    position['highest_high_since_entry_milli'] = highest_high_milli

    if made_new_high and not pd.isna(y_atr):
        trail_reference = milli_to_price(highest_high_milli)
        candidate_trail = adjust_long_stop_price(
            trail_reference - (y_atr * params.atr_times_trail),
            ticker=position.get('ticker'),
            security_profile=position.get('security_profile'),
        )
        candidate_trail_milli = price_to_milli(candidate_trail)
        position['trailing_stop_milli'] = max(position.get('trailing_stop_milli', 0), candidate_trail_milli)

    position['sl_milli'] = max(position['initial_stop_milli'], position['trailing_stop_milli'])
    if sync_display_fields:
        _sync_trailing_stop_display_fields(position)


def rollforward_position_management_from_completed_bar(
    position,
    *,
    completed_high,
    completed_atr,
    params,
    sync_display_fields=True,
):
    """Advance only next-session trailing-stop state from one completed bar.

    This primitive is used by the canonical session transition and its
    next-session projection; adapters must not duplicate its calculation.  It intentionally does not infer any broker
    fill, execute Stop/TP, or consume next-session OHLC.
    """
    if int(position.get('qty', 0) or 0) <= 0:
        return position
    _update_trailing_stop(
        position,
        y_high=completed_high,
        y_atr=completed_atr,
        params=params,
        sync_display_fields=sync_display_fields,
    )
    return position


def resolve_position_intraday_exit_hits(position, *, t_high, t_low, params):
    """Resolve canonical STOP/TP touches against the stop active for this bar.

    The canonical session transition owns timing: touches use geometry from
    prior completed information, never a stop raised by the same bar's high.
    """
    is_stop_hit = price_to_milli(t_low) <= int(position['sl_milli'])
    half_sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    is_tp_hit = (
        price_to_milli(t_high) >= int(position['tp_half_milli'])
        and not position['sold_half']
        and half_sell_qty > 0
    )
    return resolve_stop_tp_hits(stop_hit=is_stop_hit, tp_hit=is_tp_hit)


def complete_position_entry_session(position, *, t_high, t_low, params):
    """Seal the acquisition session without a same-day sell or trailing reprice.

    AI: Preserve the established Research entry-day semantics: record that day's
    high-water, but do not tighten trailing from that already-consumed high.
    Inherited Shadow state is not reset. STOP retains priority over half TP.
    """
    if position is None or int(position.get("qty", 0) or 0) <= 0:
        return position
    position["entry_day_stop_triggered"] = False
    position["entry_day_tp_triggered"] = False
    highest = int(position.get("highest_high_since_entry_milli", position["entry_fill_price_milli"]))
    if t_high is not None and not pd.isna(t_high):
        highest = max(highest, price_to_milli(t_high))
    position["highest_high_since_entry_milli"] = highest
    position["highest_high_since_entry"] = milli_to_price(highest)
    stop = t_low is not None and not pd.isna(t_low) and price_to_milli(t_low) <= int(position["sl_milli"])
    tp = t_high is not None and not pd.isna(t_high) and price_to_milli(t_high) >= int(position["tp_half_milli"])
    stop, tp = resolve_stop_tp_hits(stop_hit=stop, tp_hit=tp)
    position["entry_day_stop_triggered"] = bool(stop)
    position["entry_day_tp_triggered"] = bool(tp)
    if stop:
        position["pending_exit_action"] = "STOP"
        position["pending_exit_trigger_price"] = milli_to_price(position["sl_milli"])
    elif tp and not position.get("sold_half", False) and calc_half_take_profit_sell_qty(position["qty"], params.tp_percent) > 0:
        position["pending_exit_action"] = "TP_HALF"
        position["pending_exit_trigger_price"] = milli_to_price(position["tp_half_milli"])
    return position


def inherit_shadow_management(position, shadow_position):
    """Transfer strategy management only; actual price/quantity/cost stay actual."""
    if position is None or shadow_position is None:
        return position
    for key in POSITION_MANAGEMENT_FIELDS:
        if key in shadow_position:
            position[key] = shadow_position[key]
    position["inherited_shadow_management"] = True
    position["shadow_entry_fill_price_milli"] = shadow_position.get("entry_fill_price_milli", 0)
    position["shadow_entry_fill_price"] = shadow_position.get("entry_fill_price", float("nan"))
    position["shadow_sold_half"] = bool(shadow_position.get("sold_half", False))
    return position


def acknowledge_position_exit(position, *, event, complete=True):
    """A fill acknowledgment, never a touch, fulfils an exit obligation."""
    if not complete:
        return
    if event == "TP_HALF":
        position["sold_half"] = True
    if position.get("pending_exit_action") == event or int(position.get("qty", 0) or 0) <= 0:
        position["pending_exit_action"] = None
        position["pending_exit_trigger_price"] = float("nan")
        position.pop("pending_exit_remaining_qty", None)


def step_position_management(
    position, *, y_atr, y_ind_sell, y_close, t_open, t_high, t_low,
    t_close, t_volume, params, on_decision: Callable[[PositionExitDecision], bool],
    current_date=None, y_high=None, sync_display_fields=True,
):
    """Run the sole ordered existing-position session transition.

    ``on_decision`` returns True only when the requested leg has actually been
    applied by its execution adapter. A read-only adapter returns False, leaving
    quantity, sold-half and deferred obligations untouched. An unacknowledged
    half-exit never suppresses a later full protection decision.
    """
    events = []
    if int(position.get("qty", 0) or 0) <= 0:
        return events

    def deliver(event, qty, reference, *, trigger=None, deferred=False, block=None):
        decision = PositionExitDecision(
            event=event, qty=int(qty), reference_price=reference,
            trigger_price=trigger, deferred=deferred,
            trade_date=current_date, block_reason=block,
        )
        acknowledged = bool(on_decision(decision))
        if acknowledged and not decision.executable:
            raise ValueError("Execution adapter acknowledged a blocked decision")
        if block is not None:
            events.extend(["MISSED_SELL", block])
        elif acknowledged:
            acknowledge_position_exit(position, event=event)
            if deferred:
                events.append("DEFERRED_" + event + "_ON_OPEN")
            events.append(event)
        return acknowledged

    pending = position.get("pending_exit_action")
    pending_tp_unfilled = False
    if pending in {"STOP", "TP_HALF"}:
        block = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=position.get("ticker"))
        qty = position["qty"] if pending == "STOP" else int(position.get("pending_exit_remaining_qty", calc_half_take_profit_sell_qty(position["qty"], params.tp_percent)))
        if block is not None:
            deliver(pending, qty, None, trigger=position.get("pending_exit_trigger_price"), deferred=True, block=block)
            return events
        if qty <= 0:
            position["pending_exit_action"] = None
            position["pending_exit_trigger_price"] = float("nan")
        else:
            qty_before_confirmation = int(position["qty"])
            acknowledged = deliver(
                pending, qty, adjust_long_sell_fill_price(t_open, ticker=position.get("ticker")),
                trigger=position.get("pending_exit_trigger_price"), deferred=True,
            )
            if pending == "STOP" or int(position.get("qty", 0) or 0) <= 0:
                return events
            pending_tp_unfilled = not acknowledged
            if pending_tp_unfilled:
                confirmed_qty = max(0, qty_before_confirmation - int(position["qty"]))
                position["pending_exit_remaining_qty"] = min(int(position["qty"]), max(0, qty - confirmed_qty))

    _update_trailing_stop(position, y_high=y_high, y_atr=y_atr, params=params, sync_display_fields=sync_display_fields)
    if y_ind_sell:
        block = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=position.get("ticker"))
        reference = None if block is not None else adjust_long_sell_fill_price(t_open, ticker=position.get("ticker"))
        deliver("IND_SELL", position["qty"], reference, block=block)
        return events

    stop, tp = resolve_position_intraday_exit_hits(position, t_high=t_high, t_low=t_low, params=params)
    if tp and not pending_tp_unfilled and not (pd.isna(t_volume) or t_volume <= 0):
        target = milli_to_price(position["tp_half_milli"])
        deliver("TP_HALF", calc_half_take_profit_sell_qty(position["qty"], params.tp_percent),
                adjust_long_sell_fill_price(max(target, t_open), ticker=position.get("ticker")), trigger=target)
    if stop and int(position.get("qty", 0) or 0) > 0:
        block = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=position.get("ticker"))
        stop_price = milli_to_price(position["sl_milli"])
        reference = None if block is not None else adjust_long_sell_fill_price(min(stop_price, t_open), ticker=position.get("ticker"))
        deliver("STOP", position["qty"], reference, trigger=stop_price, block=block)
    return events
