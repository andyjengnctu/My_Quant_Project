"""Canonical completed-session replay with externally confirmed quantity facts.

AI: Market observations can change management and create obligations, never
inventory. The caller supplies confirmed quantity events; no broker I/O occurs
here. The same session decision owner is used by Research's execution adapter.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import pandas as pd

from core.position_management import (
    PositionExitDecision, acknowledge_position_exit,
    complete_position_entry_session, rollforward_position_management_from_completed_bar,
    step_position_management,
)
from core.signal_utils import generate_signals, unpack_precomputed_signals

POSITION_REPLAY_CONTRACT_VERSION = 1


def replay_confirmed_position_management(
    initial_position: Mapping[str, Any], *, frame: pd.DataFrame, params,
    start_date: str | None = None,
    confirmed_quantity_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return management projections and decisions without changing any ledger.

    Confirmations are applied on their own trade dates, not retroactively to all
    bars. A full-exit obligation remains pending until inventory is actually gone.
    Entry-day high-water is sealed before subsequent-session trailing, matching
    the canonical acquisition transition rather than a second live algorithm.
    """
    position = deepcopy(dict(initial_position))
    start = str(start_date or position.get("entry_trade_date") or "")[:10]
    entry_date = str(position.get("entry_trade_date") or start)[:10]
    if not start:
        raise ValueError("Management replay requires an immutable entry/start date")
    if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Management replay requires an ordered unique completed-bar frame")
    labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in frame.index]
    if start not in labels:
        raise ValueError(f"Management replay is missing the acquisition bar: {start}")
    atr, _buy, sell, _limits = unpack_precomputed_signals(
        generate_signals(frame, params, ticker=position.get("ticker"))
    )
    by_date: dict[str, list[dict[str, Any]]] = {}
    for raw in confirmed_quantity_events:
        event = dict(raw)
        date = str(event.get("trade_date") or "")[:10]
        if not date or date < start:
            raise ValueError("Confirmation date precedes the management origin")
        by_date.setdefault(date, []).append(event)
    decisions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    first_full_exit = None
    completed = []

    consumed: set[int] = set()

    def apply_confirmation(event):
        delta = int(event.get("qty_delta") or 0)
        position["qty"] = int(position["qty"]) + delta
        if position["qty"] < 0:
            raise ValueError("Confirmed replay quantity became negative")
        if str(event.get("event") or "") == "TP_HALF" and "pending_exit_remaining_qty" in position:
            position["pending_exit_remaining_qty"] = max(0, int(position["pending_exit_remaining_qty"]) + delta)
        if event.get("tp_half_complete"):
            acknowledge_position_exit(position, event="TP_HALF")
        if position["qty"] == 0:
            acknowledge_position_exit(position, event="STOP")
        consumed.add(id(event))

    def observe(decision: PositionExitDecision) -> bool:
        decisions.append(asdict(decision))
        # AI: Confirmed legs are consumed at their canonical decision boundary.
        # This permits TP-on-open followed by a full indicator exit to see the
        # actual remaining quantity, without inferring a fill from a touch.
        matched = [e for e in by_date.get(str(decision.trade_date), [])
                   if id(e) not in consumed and str(e.get("event")) == decision.event]
        confirmed_qty = 0
        completed_tp = False
        for event in matched:
            confirmed_qty -= int(event.get("qty_delta") or 0)
            completed_tp = completed_tp or bool(event.get("tp_half_complete"))
            apply_confirmation(event)
        return decision.executable and (completed_tp or confirmed_qty >= decision.qty)

    for idx, date in enumerate(labels):
        if date < start:
            continue
        if int(position.get("qty", 0) or 0) <= 0:
            break
        bar = frame.iloc[idx]
        before_count = len(decisions)
        if date == entry_date:
            complete_position_entry_session(
                position, t_high=bar["High"], t_low=bar["Low"], params=params,
            )
        else:
            previous = frame.iloc[idx - 1]
            step_position_management(
                position, y_atr=atr[idx - 1], y_ind_sell=bool(sell[idx - 1]),
                y_close=previous["Close"], y_high=previous["High"],
                t_open=bar["Open"], t_high=bar["High"], t_low=bar["Low"],
                t_close=bar["Close"], t_volume=bar["Volume"], params=params,
                on_decision=observe, current_date=date,
            )
        # Confirmation facts are independent of simulated execution eligibility.
        # Explicit TP-completion is provided by the canonical fill/OMS contract.
        for event in by_date.get(date, []):
            if id(event) not in consumed:
                apply_confirmation(event)
        for decision in decisions[before_count:]:
            if decision["event"] in {"STOP", "IND_SELL"} and first_full_exit is None:
                first_full_exit = dict(decision)
                first_full_exit["signal_date"] = date
                first_full_exit["execution_after_date"] = None
        active_stop = int(position["sl_milli"])
        # AI: Next-session geometry is a READ projection of execution state.
        # Feeding it back would advance high-water early, including sessions
        # where a deferred exit is blocked. Only the session owner advances the
        # execution state; today's facts and tomorrow's view stay distinct.
        next_position = deepcopy(position)
        if date != entry_date and int(position.get("qty") or 0) > 0:
            rollforward_position_management_from_completed_bar(
                next_position, completed_high=bar["High"], completed_atr=atr[idx], params=params,
            )
        # Close-known indicator exits are obligations for a later session, not
        # inferred fills on this session. Entry-day STOP has priority.
        if first_full_exit is None and position.get("pending_exit_action") == "STOP":
            first_full_exit = {
                "event": "STOP", "qty": int(position["qty"]), "reference_price": None,
                "trigger_price": position.get("pending_exit_trigger_price"),
                "deferred": True, "trade_date": None, "block_reason": None,
                "signal_date": date, "execution_after_date": date,
            }
        if first_full_exit is None and bool(sell[idx]):
            first_full_exit = {
                "event": "IND_SELL", "qty": int(position["qty"]), "reference_price": None,
                "trigger_price": None, "deferred": False, "trade_date": None,
                "block_reason": None, "signal_date": date, "execution_after_date": date,
            }
        completed.append(date)
        rows.append({
            "date": date, "active_stop_milli": active_stop,
            "next_stop_milli": int(next_position["sl_milli"]),
            "confirmed_qty": int(position["qty"]),
            "pending_exit_action": position.get("pending_exit_action"),
            "decisions": decisions[before_count:],
            "position_state": next_position,
            "execution_position_state": deepcopy(position),
            "full_exit_obligation": deepcopy(first_full_exit),
        })
    unapplied = sorted(d for d in by_date if d <= labels[-1] and d not in completed)
    if unapplied:
        raise ValueError(f"Confirmed events have no completed market bar: {unapplied}")
    return {
        "position_state": deepcopy(rows[-1]["position_state"]) if rows else position,
        "execution_position_state": position, "decisions": decisions, "sessions": rows,
        "full_exit_obligation": first_full_exit, "processed_dates": completed,
        "processed_through_date": completed[-1] if completed else None,
    }
