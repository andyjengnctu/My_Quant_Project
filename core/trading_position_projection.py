"""Pure adapter of confirmed Trading facts into the common lifecycle replay.

AI: Shared by synchronization and read models. No storage, market I/O, UI or
strategy calculations live here; the domain session owner performs decisions.
"""
from copy import deepcopy
import pandas as pd
from core.trading_lifecycle_plans import resolve_confirmed_entry_plan_from_frame
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.params_io import build_params_from_mapping
from core.price_utils import calc_half_take_profit_sell_qty
from core.entry_plans import build_position_from_frozen_entry_plan
from core.exact_accounting import milli_to_price, price_to_milli
from core.trading_account_state import MANAGEMENT_SELL_SIGNAL_STOP, MANAGEMENT_SELL_SIGNAL_INDICATOR, TRADE_MUTATION_BUY, TRADE_MUTATION_MANUAL_MANAGED_BUY, TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_STRATEGY_BUY_INCREMENT

def build_confirmed_position_origin(record, *, binding, params, account_events=(), frame=None):
    broker = dict(record.get("broker") or {})
    management = dict(record.get("strategy_management") or {})
    origin = management.get("entry_position_state")
    entry_date = str(broker.get("entry_date") or "")
    buys = [dict(e.get("details") or {}) for e in account_events
        if str((e.get("details") or {}).get("ticker") or "") == str(record.get("ticker") or "")
        and str((e.get("details") or {}).get("trade_date") or "") == entry_date
        and e.get("mutation_type") in {TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_STRATEGY_BUY_INCREMENT, TRADE_MUTATION_MANUAL_MANAGED_BUY, TRADE_MUTATION_BUY}]
    if not isinstance(origin, dict):
        # AI: Pre-migration BUY events already froze the mechanical initial
        # position. Reuse that exact evidence, never a later rolling snapshot.
        for details in buys:
            snapshot = details.get("position_after") or {}
            initial = (snapshot.get("strategy_management") or {}).get("position_state")
            if isinstance(initial, dict) and str(initial.get("entry_trade_date") or "") == entry_date:
                origin = deepcopy(initial)
                break
    first_buy = None
    if buys:
        total_qty = sum(int(d.get("fill_qty") or d.get("qty") or 0) for d in buys)
        gross = sum(int(d.get("gross_buy_milli") or 0) for d in buys)
        if total_qty <= 0 or gross <= 0:
            raise ValueError("Confirmed acquisition journal is missing quantity/gross-price evidence")
        first_buy = {"qty": total_qty, "entry_fill_price_milli": (gross + total_qty // 2) // total_qty}
    if isinstance(origin, dict) and first_buy is None:
        if str(origin.get("entry_trade_date") or "") == entry_date and int(origin.get("initial_qty") or origin.get("qty") or 0) == int(broker.get("initial_qty") or broker.get("qty") or 0):
            return deepcopy(origin)
        # Explicit broker corrections invalidate old acquisition geometry too.
        # Do not silently adopt a later/current management state as its origin.
    current = dict(management.get("position_state") or {})
    seed = dict(management.get("entry_execution_plan") or binding.get("execution_plan_seed") or {})
    initial_qty = int((first_buy or {}).get("qty") or broker.get("initial_qty") or broker.get("qty") or 0)
    if initial_qty <= 0:
        raise RuntimeError("Managed position is missing its initial confirmed quantity")
    entry_price_milli = (first_buy or {}).get("entry_fill_price_milli") or (first_buy or {}).get("fill_price_milli")
    entry_price = milli_to_price(int(entry_price_milli)) if entry_price_milli else current.get("entry_fill_price")
    if isinstance(origin, dict) and int(origin.get("qty") or 0) == initial_qty and str(origin.get("entry_trade_date") or "") == entry_date and entry_price is not None and int(origin.get("entry_fill_price_milli") or 0) == price_to_milli(entry_price):
        return deepcopy(origin)
    if entry_price is None:
        gross = int(broker.get("initial_gross_buy_milli") or 0)
        if gross <= 0:
            raise RuntimeError("Managed position is missing original fill-price evidence")
        entry_price = milli_to_price((gross + initial_qty // 2) // initial_qty)
    old_entry_date = str((origin or current).get("entry_trade_date") or "")
    date_changed = bool((isinstance(origin, dict) or first_buy is not None) and old_entry_date and old_entry_date != entry_date)
    source_info = str(seed.get("management_information_date") or "")
    needs_prior_replay = date_changed or (source_info and source_info >= entry_date)
    if needs_prior_replay:
        if frame is None:
            raise RuntimeError("Corrected acquisition requires its frozen pre-entry market evidence")
        manual_direct = str(seed.get("entry_type") or "") in {"manual_direct_backfill", "manual"} and not binding.get("signal_date")
        if manual_direct:
            prior = frame.loc[frame.index < pd.Timestamp(entry_date)]
            if prior.empty:
                raise ValueError("Corrected manual acquisition has no pre-entry bar")
            atr, _buy, _sell, _limits = unpack_precomputed_signals(generate_signals(prior, params, ticker=record.get("ticker")))
            seed.pop("shadow_position_state", None)
            seed["entry_atr"] = float(atr[-1])
            seed["target_reference_price"] = float(prior["Close"].iloc[-1])
            seed["reference_market_date"] = prior.index[-1].strftime("%Y-%m-%d")
        else:
            signal_date = binding.get("signal_date") or seed.get("planned_trade_date") or seed.get("trade_date")
            entry = {"ticker": record.get("ticker"), "origin": "scanner_strategy",
                     "signal_date": signal_date, "information_date": signal_date,
                     "execution_plan_seed": seed, "management_lineage": binding}
            seed = resolve_confirmed_entry_plan_from_frame(entry=entry, frame=frame, params=params, fill_date=entry_date)
    return build_position_from_frozen_entry_plan(
        seed, buy_price=float(entry_price), qty=initial_qty, params=params,
        entry_type=str(seed.get("entry_type") or current.get("entry_type") or "normal"),
        ticker=str(record.get("ticker") or ""), trade_date=broker.get("entry_date"),
    )


def project_full_exit_obligation(obligation):
    if obligation is None:
        return None
    signal = MANAGEMENT_SELL_SIGNAL_STOP if obligation["event"] == "STOP" else MANAGEMENT_SELL_SIGNAL_INDICATOR
    trigger = obligation.get("trigger_price")
    return {
        "sell_signal": signal,
        "sell_signal_date": str(obligation.get("signal_date") or obligation.get("trade_date")),
        "sell_signal_trigger_price_milli": None if trigger is None else price_to_milli(trigger),
        "execution_after_date": obligation.get("execution_after_date"),
    }


def collect_confirmed_position_events(account_events, record, *, through_date):
    ticker = str(record.get("ticker") or "")
    entry_date = str((record.get("broker") or {}).get("entry_date") or "")
    events = []
    lineage = record.get("management_lineage") or record.get("strategy_lineage") or {}
    frozen = lineage.get("frozen_params")
    target_qty = None
    if isinstance(frozen, dict):
        params = build_params_from_mapping(frozen)
        target_qty = calc_half_take_profit_sell_qty(int((record.get("broker") or {}).get("initial_qty") or 0), params.tp_percent)
    cumulative_tp = 0
    ordered = sorted(account_events, key=lambda e: (str((e.get("details") or {}).get("trade_date") or ""), int((e.get("details") or {}).get("replacement_for_revision") or e.get("revision") or 0)))
    for event in ordered:
        details = dict(event.get("details") or {})
        date = str(details.get("trade_date") or "")
        if details.get("ticker") != ticker or not date or date < entry_date or date > through_date:
            continue
        if event.get("mutation_type") == "confirm_sell_fill":
            tp_complete = bool(details.get("tp_half_complete"))
            if str(details.get("event") or "") == "TP_HALF":
                cumulative_tp += int(details.get("fill_qty") or details.get("qty") or 0)
                if "tp_half_complete" not in details:
                    if target_qty is None:
                        raise ValueError("Legacy TP completion requires frozen original-quantity evidence")
                    tp_complete = target_qty > 0 and cumulative_tp >= target_qty
            events.append({
                "revision": event.get("revision"), "trade_date": date,
                "qty_delta": -int(details.get("fill_qty") or details.get("qty") or 0),
                "tp_half_complete": tp_complete,
                "event": str(details.get("event") or ""),
            })
        elif event.get("mutation_type") == TRADE_MUTATION_STRATEGY_BUY_INCREMENT and date != entry_date:
            raise ValueError("Canonical partial BUY fills must belong to the original acquisition session")
    return events



def collect_confirmed_position_cycles(account_events, *, ticker, current_position=None):
    """Partition effective, correction-ordered facts by acquisition, not ticker.

    AI: This is an evidence join, not an accounting or strategy replay. Only
    confirmed acquisitions/imports create a cycle. A later snapshot can supply
    its frozen binding but cannot invent a BUY or reopen a fully sold cycle.
    Each cycle carries only its own confirmations into the canonical replay.
    """
    from core.trading_account_state import (
        ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT, MANAGED_POSITION_SOURCES,
        POSITION_SOURCE_MANUAL_ADOPTED, POSITION_SOURCE_MANUAL_MANAGED,
        POSITION_SOURCE_STRATEGY_FILL,
    )

    buy_sources = {
        TRADE_MUTATION_BUY: POSITION_SOURCE_MANUAL_ADOPTED,
        TRADE_MUTATION_MANUAL_MANAGED_BUY: POSITION_SOURCE_MANUAL_MANAGED,
        TRADE_MUTATION_STRATEGY_BUY: POSITION_SOURCE_STRATEGY_FILL,
        TRADE_MUTATION_STRATEGY_BUY_INCREMENT: POSITION_SOURCE_STRATEGY_FILL,
    }
    cycles = []
    active = None

    def new_cycle(date, event, *, source, record=None):
        logical = (event.get("details") or {}).get("replacement_for_revision") or event.get("revision") or date
        cycle = {
            "cycle_id": f"{ticker}:{logical}:{date}", "entry_date": date,
            "exit_date": None, "events": [], "record": deepcopy(record or {}),
            "source": source, "qty": 0, "initial_qty": 0,
            "initial_gross_buy_milli": 0, "initial_cost_basis_milli": 0,
            "has_buy_event": False,
        }
        cycles.append(cycle)
        return cycle

    def attach(cycle, snapshot):
        if not isinstance(snapshot, dict):
            return
        previous = cycle["record"]
        # Prefer this cycle's acquisition/activation binding over a stale
        # pre-sale snapshot or a different, currently held acquisition.
        old_lineage = previous.get("management_lineage") or previous.get("strategy_lineage") or {}
        incoming = snapshot.get("management_lineage") or snapshot.get("strategy_lineage") or {}
        if old_lineage.get("lineage_id") and incoming.get("lineage_id") and old_lineage["lineage_id"] != incoming["lineage_id"]:
            return
        if previous.get("source") in MANAGED_POSITION_SOURCES and snapshot.get("source") not in MANAGED_POSITION_SOURCES:
            return
        merged = deepcopy(snapshot)
        previous_management = previous.get("strategy_management") or {}
        management = merged.setdefault("strategy_management", {})
        for field in ("entry_execution_plan", "entry_position_state"):
            if field not in management and field in previous_management:
                management[field] = deepcopy(previous_management[field])
        for key in ("management_lineage", "strategy_lineage"):
            if key not in merged and key in previous:
                merged[key] = deepcopy(previous[key])
        cycle["record"] = merged
        cycle["source"] = str(merged.get("source") or cycle["source"])

    for event in account_events:
        details = dict(event.get("details") or {})
        if str(details.get("ticker") or "") != str(ticker):
            continue
        mutation = str(event.get("mutation_type") or "")
        date = str(details.get("trade_date") or "")[:10]
        if mutation in buy_sources:
            qty = int(details.get("fill_qty") or details.get("qty") or 0)
            if qty <= 0 or not date:
                continue
            if active is None:
                active = new_cycle(date, event, source=buy_sources[mutation])
            active["events"].append(deepcopy(event))
            active["has_buy_event"] = True
            active["qty"] += qty
            active["initial_qty"] += qty
            active["initial_gross_buy_milli"] += int(details.get("gross_buy_milli") or 0)
            active["initial_cost_basis_milli"] += int(details.get("net_buy_total_milli") or 0)
            attach(active, details.get("position_after"))
            # Legacy acquisition events can carry lineage outside the snapshot.
            for key in ("management_lineage", "strategy_lineage"):
                if isinstance(details.get(key), dict):
                    active["record"][key] = deepcopy(details[key])
            continue
        if mutation == "adopt_manual_position":
            date = str(details.get("entry_date") or "")[:10]
            if not date:
                continue
            active = new_cycle(date, event, source=POSITION_SOURCE_MANUAL_ADOPTED)
            active["qty"] = active["initial_qty"] = int(details.get("qty") or 0)
            active["initial_cost_basis_milli"] = int(details.get("cost_basis_total_milli") or 0)
            active["events"].append(deepcopy(event))
            continue
        if mutation == ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT:
            snapshot = details.get("position_after")
            broker = (snapshot or {}).get("broker") or {}
            date = str(broker.get("entry_date") or "")[:10]
            if active is None:
                # Compatibility: an explicit retained adoption and matching
                # current inventory prove stock ownership, not a fictional fill.
                current_broker = (current_position or {}).get("broker") or {}
                if not date or current_broker.get("entry_date") != date:
                    continue
                active = new_cycle(date, event, source=POSITION_SOURCE_MANUAL_MANAGED, record=snapshot)
                active["qty"] = active["initial_qty"] = int(broker.get("initial_qty") or broker.get("qty") or 0)
                active["initial_gross_buy_milli"] = int(broker.get("initial_gross_buy_milli") or 0)
                active["initial_cost_basis_milli"] = int(broker.get("initial_cost_basis_milli") or 0)
            attach(active, snapshot)
            active["events"].append(deepcopy(event))
            continue
        if mutation == "confirm_sell_fill" and active is not None:
            active["events"].append(deepcopy(event))
            attach(active, details.get("position_before"))
            active["qty"] = int(details.get("remaining_qty", active["qty"] - int(details.get("qty") or 0)))
            if active["qty"] <= 0:
                active["exit_date"] = date
                active = None
        elif mutation in {"remove_manual_position", "remove_position_broker_truth"}:
            active = None

    if active is not None and isinstance(current_position, dict):
        if str((current_position.get("broker") or {}).get("entry_date") or "") == active["entry_date"]:
            attach(active, current_position)
    for cycle in cycles:
        record = cycle["record"]
        record["ticker"] = str(ticker)
        record["source"] = cycle["source"]
        broker = deepcopy(record.get("broker") or {})
        broker.update(qty=max(0, cycle["qty"]), initial_qty=cycle["initial_qty"], entry_date=cycle["entry_date"])
        for key in ("initial_gross_buy_milli", "initial_cost_basis_milli"):
            if cycle[key] > 0:
                broker[key] = cycle[key]
        record["broker"] = broker
    return cycles

