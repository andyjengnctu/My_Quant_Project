"""Canonical Trading state projection for the single-stock Workbench inspector.

The Trading single-stock chart may replay historical strategy geometry for visual
context, but broker/account truth must never be inferred from that replay.  This
module projects persisted Scanner, broker-order, fill, account-position and
roll-forward evidence into a date-aware sidebar state without duplicating entry
or stop formulas.
"""
from __future__ import annotations

from core.trade_lifecycle import build_prefill_lifecycle_from_frame

from copy import deepcopy
from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.entry_plans import build_position_from_frozen_entry_plan
from core.exact_accounting import calc_entry_total_cost, milli_to_money, milli_to_price, price_to_milli
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.position_replay import replay_confirmed_position_management
from core.trading_position_projection import build_confirmed_position_origin, collect_confirmed_position_events, collect_confirmed_position_cycles, project_full_exit_obligation
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trading_account_state import (
    ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT,
    ACCOUNT_MUTATION_RECORD_MANAGEMENT_SELL_SIGNAL,
    MANAGED_POSITION_SOURCES,
    POSITION_SOURCE_MANUAL_MANAGED,
    POSITION_SOURCE_STRATEGY_FILL,
    TRADE_MUTATION_MANUAL_MANAGED_BUY,
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_SELL,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
    effective_trading_account_events,
    project_trading_account_transactions,
)
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from core.trading_order_state import (
    TRADING_ORDER_PURPOSE_ENTRY,
    TRADING_ORDER_SIDE_BUY,
)
from services.trading.account_state import load_trading_account_state
from services.trading.pending_entry_links import (
    original_buy_event, pending_entry_matches_buy, resolve_pending_entry_for_buy_event, resolve_position_pending_entry,
)
from services.trading.pending_entry_state import (
    PENDING_ENTRY_STATUS_ACTIVE,
    PENDING_ENTRY_STATUS_CANCELLED_NO_FILL,
    PENDING_ENTRY_STATUS_CANCELLED_USER_DELETED,
    PENDING_ENTRY_STATUS_FILLED,
    load_trading_pending_entry_state,
    project_trading_pending_intent_entries,
)
from services.trading.accounting_policy import overlay_trading_accounting_params, build_standalone_trading_accounting_params
from services.trading.order_state import load_trading_order_state
from services.trading.protection_planning import get_trading_protection_plan_read_model
from services.trading.indicator_exit_planning import get_trading_indicator_exit_plan_read_model
from services.trading.strategy_param_runtime import resolve_trading_position_management_binding, resolve_trading_candidate_frozen_params, build_trading_order_strategy_lineage
from core.trading_lifecycle_plans import build_pending_prefill_plan
from core.trade_lifecycle import (
    TRADE_LIFECYCLE_POSITION,
    TRADE_LIFECYCLE_SHADOW,
    TRADE_LIFECYCLE_SIGNAL,
    TRADE_LIFECYCLE_PENDING,
    overlay_confirmed_order_intent,
    TRADE_TRANSACTION_LINE_KEYS,
    build_trade_lifecycle_row,
    build_prefill_lifecycle_timeline,
    iter_trade_lifecycle_line_values,
)


_LOGGER = logging.getLogger(__name__)


_BUY_MUTATIONS = {
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_MANUAL_MANAGED_BUY,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
}

_TRADING_REPLAY_TRANSACTION_TRACES = {
    "限價買進",
    "買進",
    "買進(延續候選)",
    "買進(重進)",
    "錯失買進",
    "錯失買進(延續候選)",
    "錯失買進(重進)",
    "停利",
    "停損賣出",
    "指標賣出",
    "錯失賣出",
    "強制結算",
}


def _assign_chart_float(
    lines: dict[str, list[float]],
    *,
    key: str,
    idx: int,
    value: object,
    evidence: str,
) -> None:
    """Assign a persisted Trading numeric value without hiding malformed evidence."""
    try:
        lines[key][idx] = float(value)
    except (TypeError, ValueError) as exc:
        _LOGGER.warning(
            "Skipping malformed Trading chart value | evidence=%s | key=%s | idx=%s | value=%r | error=%s",
            evidence,
            key,
            idx,
            value,
            exc,
        )


def _date_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return normalize_trading_date(text, field_name="inspection_date", allow_none=True)
    except (TypeError, ValueError):
        # Runtime timestamps are ISO-8601 strings while market evidence is a
        # calendar date.  Their first ten characters are the persisted local date.
        prefix = text[:10]
        try:
            datetime.strptime(prefix, "%Y-%m-%d")
        except ValueError:
            return None
        return prefix


def _ticker_matches(payload: Mapping[str, Any] | None, ticker: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    try:
        return normalize_trading_ticker(payload.get("ticker")) == ticker
    except (TypeError, ValueError):
        return False


def load_trading_single_stock_position_binding(project_root: str | Path, ticker: object) -> dict[str, Any] | None:
    """Resolve immutable frozen Params for an open strategy-managed position."""
    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    account = load_trading_account_state(root, required=False)
    if not account:
        return None
    record = (account.get("positions") or {}).get(ticker_key)
    if not isinstance(record, Mapping) or str(record.get("source") or "") not in MANAGED_POSITION_SOURCES:
        return None
    orders = load_trading_order_state(root, required=False) or {"orders": {}}
    return resolve_trading_position_management_binding(record, orders=orders)



def load_trading_single_stock_pending_binding(project_root: str | Path, ticker: object):
    """Resolve active pending Params before any current Scanner/production params."""
    ticker_key = normalize_trading_ticker(ticker)
    state = load_trading_pending_entry_state(project_root, required=False) or {}
    entries = [row for row in (state.get("entries") or {}).values()
               if row.get("ticker") == ticker_key and row.get("status") == "ACTIVE"]
    if not entries:
        return None
    if len(entries) != 1:
        raise RuntimeError("More than one active pending lineage for the same ticker")
    from services.trading.strategy_param_runtime import validate_trading_position_management_lineage
    return validate_trading_position_management_lineage(entries[0]["management_lineage"])

def _inspection_account_events(account):
    """Consume the SAME corrected economic rows as the account transaction table.

    AI: The immutable raw SELL may predate a BUY correction. Reusing its old
    cost/PnL (or appending the replacement BUY after SELL) is not account truth.
    Management journal metadata is retained; no fee/PnL formula is repeated here.
    """
    trades = {int(e["revision"]): e for e in project_trading_account_transactions(
        account, accounting_params=build_standalone_trading_accounting_params()
    )}
    events = []
    for raw in effective_trading_account_events(account):
        event = deepcopy(raw)
        trade = trades.get(int(event.get("revision") or 0))
        if trade is not None:
            event["details"] = deepcopy(trade["details"])
            if event.get("mutation_type") in _BUY_MUTATIONS:
                # AI: Corrections replace economics, not the original strategy
                # binding. Resolve the immutable acquisition, never today's
                # ticker template or another cycle's last SELL snapshot.
                origin = original_buy_event(raw, account.get("events") or [])
                original_details = origin.get("details") or {}
                for key in ("position_after", "management_lineage", "strategy_lineage", "entry_order_id"):
                    if key in original_details:
                        event["details"][key] = deepcopy(original_details[key])
        events.append(event)
    return sorted(events, key=lambda e: (
        int((e.get("details") or {}).get("effective_before_revision")
            or (e.get("details") or {}).get("replacement_for_revision") or e.get("revision") or 0),
        -1 if (e.get("details") or {}).get("effective_before_revision") else 0,
        int(e.get("revision") or 0),
    ))



def build_trading_single_stock_inspection(
    project_root: str | Path,
    ticker: object,
    *,
    candidate_row: Mapping[str, Any] | None = None,
    market_frame: pd.DataFrame | None = None,
    consumer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read canonical Trading evidence once for date-aware inspector rendering."""
    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    account = load_trading_account_state(root, required=False)
    orders = load_trading_order_state(root, required=False)
    pending_state = load_trading_pending_entry_state(root, required=False)
    pending_entries = [
        deepcopy(dict(row))
        for row in project_trading_pending_intent_entries(pending_state).values()
        if _ticker_matches(row, ticker_key)
    ]

    candidate = dict(candidate_row or {})
    if candidate and not _ticker_matches(candidate, ticker_key):
        candidate = {}

    account_events: list[dict[str, Any]] = []
    current_position = None
    position_binding = None
    if account:
        account_events = [
            deepcopy(event)
            for event in _inspection_account_events(account)
            if _ticker_matches(event.get("details") or {}, ticker_key)
            or any(
                _ticker_matches(row, ticker_key)
                for row in list((event.get("details") or {}).get("positions") or [])
            )
        ]
        current_position = deepcopy((account.get("positions") or {}).get(ticker_key))
        if isinstance(current_position, Mapping) and str(current_position.get("source") or "") in MANAGED_POSITION_SOURCES:
            position_binding = resolve_trading_position_management_binding(
                current_position,
                orders=(orders or {"orders": {}}),
            )

    entry_orders = []
    if orders:
        for record in (orders.get("orders") or {}).values():
            if not isinstance(record, Mapping):
                continue
            if not _ticker_matches(record, ticker_key):
                continue
            if str(record.get("side") or "") != TRADING_ORDER_SIDE_BUY:
                continue
            if str(record.get("purpose") or TRADING_ORDER_PURPOSE_ENTRY) != TRADING_ORDER_PURPOSE_ENTRY:
                continue
            entry_orders.append(deepcopy(dict(record)))
        entry_orders.sort(
            key=lambda row: (
                str(row.get("information_date") or ""),
                str(row.get("ordered_at") or ""),
                str(row.get("order_id") or ""),
            )
        )

    decision_errors: list[str] = []
    try:
        protection = get_trading_protection_plan_read_model(root, recover_pending_fill=False)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        protection = {"fresh": False, "positions": []}
        decision_errors.append(f"protection: {type(exc).__name__}: {exc}")
    try:
        indicator_exit = get_trading_indicator_exit_plan_read_model(root, recover_pending_fill=False)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        indicator_exit = {"fresh": False, "exits": []}
        decision_errors.append(f"indicator_exit: {type(exc).__name__}: {exc}")

    from services.trading.market_data_consumer import load_trading_v2_consumer_state
    # AI: The GUI already owns a verified, full consumer frame. Reuse that exact
    # pinned input for frozen management too; a second independent reload can
    # fail or switch generations after the chart has successfully loaded.
    if market_frame is not None and consumer_state is None:
        raise ValueError("Inspection market_frame requires its pinned consumer_state")
    consumer = deepcopy(dict(consumer_state)) if consumer_state is not None else load_trading_v2_consumer_state(root, required=False)
    # AI: Display candles may be float32 or a cropped window. Keep the already
    # pinned full-precision input available to pre-fill replay as well as POSITION.
    # This reference is operation-local, never a second persisted market source.
    if market_frame is not None:
        if market_frame.empty or pd.Timestamp(market_frame.index.max()).normalize() > pd.Timestamp(consumer["market_date"]):
            raise ValueError("Inspection frame exceeds the pinned finalized boundary")
        if not market_frame.index.is_unique or not market_frame.index.is_monotonic_increasing:
            raise ValueError("Inspection frame must have unique chronological dates")
    position_history = collect_confirmed_position_cycles(
        account_events, ticker=ticker_key, current_position=current_position,
    )
    management_projection = None
    context = None
    frame_cache = {}
    for cycle in position_history:
        record = cycle["record"]
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            continue
        try:
            is_current = (cycle.get("exit_date") is None and current_position is not None
                          and (current_position.get("broker") or {}).get("entry_date") == cycle["entry_date"])
            binding = position_binding if is_current else resolve_trading_position_management_binding(record, orders=orders)
            cycle["binding"] = binding
            if consumer is None:
                continue
            from services.trading.lifecycle_context import resolve_trading_lifecycle_context
            from services.trading.position_market_context import load_trading_position_market_frame
            from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
            params = build_params_from_mapping(binding["frozen_params"])
            through = min(str(consumer["market_date"]), cycle.get("exit_date") or str(consumer["market_date"]))
            if cycle["entry_date"] > through:
                continue
            signature = binding["params_signature"]
            if signature not in frame_cache:
                if market_frame is not None:
                    if market_frame.empty or pd.Timestamp(market_frame.index.max()).normalize() > pd.Timestamp(consumer["market_date"]):
                        raise ValueError("Inspection frame exceeds the pinned finalized boundary")
                    frame_cache[signature], _ = sanitize_ohlcv_dataframe(
                        market_frame.rename_axis("Date").reset_index(), ticker_key, min_rows=get_required_min_rows(params),
                    )
                else:
                    if context is None:
                        context = resolve_trading_lifecycle_context(root, consumer_state=consumer)
                    frame_cache[signature] = load_trading_position_market_frame(
                        view=context.view, ticker=ticker_key, params=params, allowed_date=context.finalized_date,
                    )
            frame = frame_cache[signature].loc[:through]
            cycle["projection"] = replay_confirmed_position_management(
                build_confirmed_position_origin(record, binding=binding, params=params,
                    account_events=cycle["events"], frame=frame,
                    prefill_entry=resolve_position_pending_entry(
                        record, pending_entries, account_events=cycle["events"],
                        account_audit_events=account.get("events") or () if account else (),
                    )),
                frame=frame, params=params, start_date=cycle["entry_date"],
                confirmed_quantity_events=collect_confirmed_position_events(cycle["events"], record, through_date=through),
            )
            if is_current:
                management_projection = cycle["projection"]
        except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            cycle["projection_error"] = f"management_projection[{cycle['cycle_id']}]: {type(exc).__name__}: {exc}"
            _LOGGER.warning("%s", cycle["projection_error"])
            decision_errors.append(cycle["projection_error"])
    return {
        "ticker": ticker_key,
        "finalized_date": None if consumer is None else consumer.get("market_date"),
        "candidate": candidate,
        "account_revision": None if not account else int(account.get("revision") or 0),
        "order_revision": None if not orders else int(orders.get("revision") or 0),
        "account_events": account_events,
        "account_audit_events": [] if not account else [
            deepcopy(event) for event in account.get("events", [])
            if _ticker_matches(event.get("details") or {}, ticker_key)
        ],
        "entry_orders": entry_orders,
        "pending_entries": pending_entries,
        "prefill_market_frame": market_frame,
        "current_position": deepcopy(current_position),
        "position_binding": deepcopy(position_binding),
        "management_projection": management_projection,
        "position_history": position_history,
        "protection": {
            "fresh": bool(protection.get("fresh")),
            "positions": [
                deepcopy(dict(row))
                for row in list(protection.get("positions") or [])
                if _ticker_matches(row, ticker_key)
            ],
        },
        "indicator_exit": {
            "fresh": bool(indicator_exit.get("fresh")),
            "exits": [
                deepcopy(dict(row))
                for row in list(indicator_exit.get("exits") or [])
                if _ticker_matches(row, ticker_key)
            ],
        },
        "decision_errors": decision_errors,
    }


def _order_start_date(order: Mapping[str, Any]) -> str | None:
    return _date_text(order.get("information_date")) or _date_text(order.get("ordered_at"))


def _order_cancel_date(order: Mapping[str, Any]) -> str | None:
    return _date_text(order.get("cancelled_at"))


def _fills_through(order: Mapping[str, Any], date_text: str) -> list[dict[str, Any]]:
    rows = []
    for raw in list(order.get("fills") or []):
        fill = dict(raw)
        fill_date = _date_text(fill.get("trade_date"))
        if fill_date is not None and fill_date <= date_text:
            rows.append(fill)
    rows.sort(key=lambda row: (str(row.get("trade_date") or ""), str(row.get("confirmed_at") or ""), str(row.get("fill_id") or "")))
    return rows


def _remaining_reserved_cost(order: Mapping[str, Any], remaining_qty: int) -> float | None:
    if remaining_qty <= 0:
        return 0.0
    qty = int(order.get("qty") or 0)
    if qty <= 0:
        return None
    if remaining_qty == qty:
        raw = order.get("reserved_cost_milli")
        return None if raw is None else milli_to_money(int(raw))
    frozen = order.get("frozen_params")
    limit_milli = order.get("limit_price_milli")
    if not isinstance(frozen, Mapping) or limit_milli is None:
        return None
    params = build_params_from_mapping(dict(frozen))
    return float(calc_entry_total_cost(milli_to_price(int(limit_milli)), int(remaining_qty), params))


def _order_entry_position(order: Mapping[str, Any], fills: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not fills:
        return None
    frozen = order.get("frozen_params")
    if not isinstance(frozen, Mapping):
        return None
    total_qty = sum(int(fill.get("qty") or 0) for fill in fills)
    if total_qty <= 0:
        return None
    weighted_milli = sum(int(fill.get("fill_price_milli") or 0) * int(fill.get("qty") or 0) for fill in fills)
    average_fill_price = milli_to_price((weighted_milli + total_qty // 2) // total_qty)
    params = overlay_trading_accounting_params(build_params_from_mapping(dict(frozen)))
    seed = build_trading_order_strategy_lineage(order)["execution_plan_seed"]
    return build_position_from_frozen_entry_plan(
        seed, buy_price=average_fill_price, qty=total_qty, params=params,
        entry_type=str(order.get("entry_type") or order.get("kind") or "normal"),
        ticker=str(order.get("ticker") or ""),
        trade_date=str(fills[0].get("trade_date") or ""),
    )


def _order_state_as_of(order: Mapping[str, Any], date_text: str) -> dict[str, Any] | None:
    start_date = _order_start_date(order)
    if start_date is None or date_text < start_date:
        return None
    cancel_date = _order_cancel_date(order)
    fills = _fills_through(order, date_text)
    filled_qty = sum(int(fill.get("qty") or 0) for fill in fills)
    qty = int(order.get("qty") or 0)
    remaining_qty = max(0, qty - filled_qty)
    full_fill_date = None
    if filled_qty >= qty > 0 and fills:
        full_fill_date = _date_text(fills[-1].get("trade_date"))

    # Once an order is terminal, it is no longer a reservation on later dates.
    if cancel_date is not None and date_text > cancel_date:
        return None
    if full_fill_date is not None and date_text > full_fill_date:
        return None

    if cancel_date is not None and date_text == cancel_date:
        state = "CANCELLED"
        reserved = 0.0
    elif filled_qty >= qty > 0:
        state = "FILLED"
        reserved = 0.0
    elif filled_qty > 0:
        state = "PARTIAL"
        reserved = _remaining_reserved_cost(order, remaining_qty)
    else:
        state = "ORDERED"
        reserved = _remaining_reserved_cost(order, remaining_qty)

    position = _order_entry_position(order, fills)
    actual_spend_milli = sum(int(fill.get("net_buy_total_milli") or 0) for fill in fills)
    return {
        "state": state,
        "source": "broker_entry_order",
        "order_id": str(order.get("order_id") or ""),
        "entry_type": str(order.get("entry_type") or order.get("kind") or "normal"),
        "limit_price": milli_to_price(int(order["limit_price_milli"])),
        "entry_price": None if position is None else position.get("entry_fill_price"),
        "initial_stop_price": (
            milli_to_price(int(order["init_sl_milli"])) if position is None else position.get("initial_stop")
        ),
        "trailing_stop_price": (
            milli_to_price(int(order["init_trail_milli"])) if position is None else position.get("trailing_stop")
        ),
        "stop_price": (
            milli_to_price(int(order["init_sl_milli"])) if position is None else position.get("sl")
        ),
        "tp_price": (
            milli_to_price(int(order["target_price_milli"])) if position is None else position.get("tp_half")
        ),
        "reserved_capital": reserved,
        "original_reserved_capital": milli_to_money(int(order.get("reserved_cost_milli") or 0)),
        "buy_capital": None if actual_spend_milli <= 0 else milli_to_money(actual_spend_milli),
        "buy_qty": filled_qty or None,
        "planned_qty": qty or None,
        "remaining_qty": remaining_qty,
        "position_qty": filled_qty or None,
        "remaining_order_qty": remaining_qty,
        "sell_signal": False,
    }


def _trade_event_date(event: Mapping[str, Any]) -> str | None:
    details = event.get("details") or {}
    mutation = str(event.get("mutation_type") or "")
    if mutation in _BUY_MUTATIONS or mutation == TRADE_MUTATION_SELL:
        return _date_text(details.get("trade_date"))
    if mutation == ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT:
        return _date_text(details.get("management_start_date")) or _date_text(event.get("timestamp"))
    return _date_text(event.get("timestamp"))


def _current_position_effective_date(
    inspection: Mapping[str, Any],
    *,
    ticker: str,
    entry_date: str,
    management_start_date: str | None = None,
) -> str:
    """Return the first date on which today's persisted open-position state is valid.

    The current account record may include a later partial fill/sell or roll-forward
    than the chart date being inspected.  Gate its use on the latest persisted
    event that contributed to this still-open position cycle so current state never
    leaks backward into an earlier point-in-time view.
    """
    effective_dates = [max(entry_date, management_start_date) if management_start_date is not None else entry_date]
    for event in list(inspection.get("account_events") or []):
        mutation = str(event.get("mutation_type") or "")
        details = dict(event.get("details") or {})
        if mutation == "rollforward_strategy_management":
            for row in list(details.get("positions") or []):
                if not _ticker_matches(row, ticker):
                    continue
                processed = _date_text(row.get("processed_through_date"))
                if processed is not None and processed >= entry_date:
                    effective_dates.append(processed)
            continue
        if not _ticker_matches(details, ticker):
            continue
        event_date = _trade_event_date(event)
        if event_date is not None and event_date >= entry_date:
            effective_dates.append(event_date)
    return max(effective_dates)


def _sell_signal_as_of(inspection: Mapping[str, Any], date_text: str, *, entry_date: str | None = None) -> str | None:
    """Project the same canonical account SELL obligation shown by Trading Center."""
    current_position = inspection.get("current_position")
    if isinstance(current_position, Mapping):
        current_entry = _date_text((current_position.get("broker") or {}).get("entry_date"))
        if entry_date is not None and current_entry != entry_date:
            return None
        management = dict(current_position.get("strategy_management") or {})
        signal = str(management.get("sell_signal") or "").strip()
        signal_date = _date_text(management.get("sell_signal_date"))
        if signal and signal_date is not None and signal_date <= date_text:
            return signal

    # AI: Compatibility fallback for runtime state created before account-level
    # SELL obligations existed. Lifecycle sync migrates current holdings, so new
    # state no longer depends on planner-specific SELL detection as a second SSOT.
    protection = inspection.get("protection") or {}
    if bool(protection.get("fresh")):
        for row in list(protection.get("positions") or []):
            if not bool(row.get("stop_forced_exit")):
                continue
            trigger_date = _date_text(row.get("stop_trigger_trade_date"))
            if trigger_date is not None and trigger_date <= date_text:
                return "STOP EXIT"

    indicator_exit = inspection.get("indicator_exit") or {}
    if bool(indicator_exit.get("fresh")):
        for row in list(indicator_exit.get("exits") or []):
            signal_date = _date_text(row.get("signal_information_date"))
            if signal_date is not None and signal_date <= date_text:
                return "INDICATOR SELL"
    return None


def _account_cycle_as_of(inspection: Mapping[str, Any], date_text: str) -> dict[str, Any] | None:
    ticker = str(inspection.get("ticker") or "")
    qty = 0
    entry_date = None
    gross_buy_milli = 0
    net_buy_milli = 0
    entry_qty = 0
    position_state = None
    lineage = None
    entry_order_id = None
    last_rollforward_date = None
    trailing_stop_exact = True
    closed_date = None
    management_sell_signal = None
    management_sell_signal_date = None

    for event in list(inspection.get("account_events") or []):
        mutation = str(event.get("mutation_type") or "")
        details = dict(event.get("details") or {})

        if mutation == "rollforward_strategy_management":
            for row in list(details.get("positions") or []):
                if not _ticker_matches(row, ticker):
                    continue
                processed = _date_text(row.get("processed_through_date"))
                if processed is None or processed > date_text or qty <= 0:
                    continue
                if position_state is not None:
                    stop_milli = int(row.get("stop_milli") or 0)
                    if stop_milli > 0:
                        position_state["sl_milli"] = stop_milli
                        position_state["sl"] = milli_to_price(stop_milli)
                        initial_stop_milli = int(position_state.get("initial_stop_milli") or 0)
                        if stop_milli > initial_stop_milli:
                            position_state["trailing_stop_milli"] = stop_milli
                            position_state["trailing_stop"] = milli_to_price(stop_milli)
                        else:
                            # The journal intentionally stores only effective stop.
                            # When it is still pinned to initial stop, historical
                            # trailing may have moved underneath it and cannot be
                            # reconstructed exactly from persisted evidence.
                            trailing_stop_exact = False
                last_rollforward_date = processed
            continue

        if mutation == ACCOUNT_MUTATION_RECORD_MANAGEMENT_SELL_SIGNAL:
            for row in list(details.get("positions") or []):
                if not _ticker_matches(row, ticker):
                    continue
                signal_date = _date_text(row.get("sell_signal_date"))
                if signal_date is None or signal_date > date_text or qty <= 0:
                    continue
                row_lineage_id = str(row.get("lineage_id") or "").strip()
                current_lineage_id = str((lineage or {}).get("lineage_id") or "").strip()
                if row_lineage_id and current_lineage_id and row_lineage_id != current_lineage_id:
                    continue
                management_sell_signal = str(row.get("sell_signal") or "").strip() or None
                management_sell_signal_date = signal_date
            continue

        if not _ticker_matches(details, ticker):
            continue
        event_date = _trade_event_date(event)
        if event_date is None or event_date > date_text:
            continue

        if mutation == ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT:
            position_after = details.get("position_after")
            if not isinstance(position_after, Mapping):
                continue
            broker_after = dict(position_after.get("broker") or {})
            management = dict(position_after.get("strategy_management") or {})
            management_start = _date_text(management.get("management_start_date"))
            if management_start is not None and date_text < management_start:
                continue
            managed_position = management.get("position_state")
            if not isinstance(managed_position, Mapping):
                continue
            qty = int(broker_after.get("qty") or 0)
            entry_qty = qty
            entry_date = _date_text(broker_after.get("entry_date")) or management_start or event_date
            gross_buy_milli = int(
                broker_after.get("initial_gross_buy_milli")
                or broker_after.get("initial_cost_basis_milli")
                or 0
            )
            net_buy_milli = int(broker_after.get("initial_cost_basis_milli") or 0)
            position_state = deepcopy(dict(managed_position))
            lineage = deepcopy(dict(position_after.get("management_lineage") or {}))
            entry_order_id = None
            last_rollforward_date = _date_text(management.get("last_rollforward_date"))
            trailing_stop_exact = True
            closed_date = None
            continue

        if mutation in _BUY_MUTATIONS:
            add_qty = int(details.get("qty") or details.get("fill_qty") or 0)
            if add_qty <= 0:
                continue
            if qty <= 0:
                qty = 0
                entry_date = event_date
                gross_buy_milli = 0
                net_buy_milli = 0
                entry_qty = 0
                position_state = None
                lineage = None
                entry_order_id = None
                last_rollforward_date = None
                trailing_stop_exact = True
                closed_date = None
                management_sell_signal = None
                management_sell_signal_date = None
            qty += add_qty
            entry_qty += add_qty
            gross_buy_milli += int(details.get("gross_buy_milli") or 0)
            net_buy_milli += int(details.get("net_buy_total_milli") or 0)
            if mutation in {TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_MANUAL_MANAGED_BUY}:
                position_after = details.get("position_after")
                if isinstance(position_after, Mapping):
                    management = position_after.get("strategy_management") or {}
                    management_start = _date_text(management.get("management_start_date"))
                    candidate_position = management.get("position_state")
                    management_lineage = details.get("management_lineage")
                    if not isinstance(management_lineage, Mapping):
                        management_lineage = position_after.get("management_lineage")
                    pending_managed_fill = (
                        mutation == TRADE_MUTATION_MANUAL_MANAGED_BUY
                        and isinstance(management_lineage, Mapping)
                        and str(management_lineage.get("origin") or "") == "manual_pending_entry"
                    )
                    if (
                        isinstance(candidate_position, Mapping)
                        and (
                            mutation == TRADE_MUTATION_STRATEGY_BUY
                            or pending_managed_fill
                            or management_start is None
                            or date_text >= management_start
                        )
                    ):
                        # The initial position state attached to a confirmed
                        # pending fill is mechanical fill-time evidence.  It may
                        # be rendered from the actual fill date even when daily
                        # roll-forward intentionally begins later at the frozen
                        # information/management-start boundary.  Direct manual
                        # backfills keep their prior "no management geometry
                        # before management_start" contract.
                        position_state = deepcopy(dict(candidate_position))
                lineage_key = "strategy_lineage" if mutation == TRADE_MUTATION_STRATEGY_BUY else "management_lineage"
                if isinstance(details.get(lineage_key), Mapping):
                    lineage = deepcopy(dict(details.get(lineage_key) or {}))
                entry_order_id = str(details.get("entry_order_id") or "") or None
            continue

        if mutation == TRADE_MUTATION_SELL and qty > 0:
            sell_qty = int(details.get("qty") or 0)
            qty = max(0, qty - sell_qty)
            if isinstance(position_state, Mapping) and qty > 0:
                position_state = deepcopy(dict(position_state))
                position_state["qty"] = qty
                if str(details.get("event") or "").upper() == "TP_HALF" and details.get("tp_half_complete", True):
                    # Historical account events persist the pre-sell position,
                    # not a second post-sell snapshot.  The confirmed TP event is
                    # sufficient evidence that subsequent dates must no longer
                    # show the half-take-profit line.
                    position_state["sold_half"] = True
            if qty <= 0:
                closed_date = event_date

    if entry_date is None or closed_date is not None and closed_date < date_text:
        return None

    # If this cycle was born from a broker ENTRY order, that order is the exact
    # source for weighted fills and post-fill entry geometry.
    matching_order = None
    if entry_order_id:
        for order in list(inspection.get("entry_orders") or []):
            if str(order.get("order_id") or "") == entry_order_id:
                matching_order = order
                break
    if matching_order is not None:
        fills = _fills_through(matching_order, date_text)
        order_position = _order_entry_position(matching_order, fills)
        if order_position is not None:
            if position_state is None:
                position_state = order_position
            else:
                # Preserve roll-forward stop evidence while taking entry geometry
                # from the complete broker-fill set for the entry date.
                rolled_stop = position_state.get("sl")
                position_state = order_position
                if last_rollforward_date is not None and rolled_stop is not None:
                    position_state["sl"] = rolled_stop

    current_position = inspection.get("current_position")
    if isinstance(current_position, Mapping):
        broker = current_position.get("broker") or {}
        management = current_position.get("strategy_management") or {}
        current_entry_date = _date_text(broker.get("entry_date"))
        current_last_roll = _date_text(management.get("last_rollforward_date"))
        current_management_start = _date_text(management.get("management_start_date"))
        current_state = management.get("position_state")
        current_effective_date = (
            _current_position_effective_date(
                inspection,
                ticker=ticker,
                entry_date=entry_date,
                management_start_date=current_management_start,
            )
            if current_entry_date == entry_date
            else None
        )
        if (
            current_entry_date == entry_date
            and isinstance(current_state, Mapping)
            and current_effective_date is not None
            and date_text >= current_effective_date
        ):
            position_state = deepcopy(dict(current_state))
            qty = int((current_position.get("broker") or {}).get("qty") or qty)
            last_rollforward_date = current_last_roll
            trailing_stop_exact = True

    average_entry = None
    if entry_qty > 0 and gross_buy_milli > 0:
        average_entry = milli_to_price((gross_buy_milli + entry_qty // 2) // entry_qty)
    if average_entry is None and isinstance(position_state, Mapping):
        average_entry = position_state.get("entry_fill_price")

    lineage_planned_qty = lineage.get("planned_qty") if isinstance(lineage, Mapping) else None
    lineage_planned_cost = lineage.get("planned_cost") if isinstance(lineage, Mapping) else None
    return {
        "state": TRADE_LIFECYCLE_POSITION,
        "display_state": "\u5df2\u51fa\u5834" if closed_date == date_text else "\u6301\u80a1",
        "closed_date": closed_date,
        "source": "canonical_account_position",
        "entry_date": entry_date,
        "entry_order_id": entry_order_id,
        "limit_price": None if not isinstance(position_state, Mapping) else position_state.get("limit_price"),
        "entry_price": average_entry,
        "initial_stop_price": None if not isinstance(position_state, Mapping) else position_state.get("initial_stop"),
        "trailing_stop_price": (
            None if not isinstance(position_state, Mapping) or not trailing_stop_exact
            else position_state.get("trailing_stop")
        ),
        "stop_price": None if not isinstance(position_state, Mapping) else position_state.get("sl"),
        "tp_price": (
            None
            if not isinstance(position_state, Mapping) or bool(position_state.get("sold_half", False))
            else position_state.get("tp_half")
        ),
        "reserved_capital": lineage_planned_cost,
        "original_reserved_capital": lineage_planned_cost,
        "buy_capital": None if net_buy_milli <= 0 else milli_to_money(net_buy_milli),
        "buy_qty": entry_qty or None,
        "planned_qty": lineage_planned_qty,
        "remaining_qty": qty,
        "position_qty": qty,
        "remaining_order_qty": 0,
        "last_rollforward_date": last_rollforward_date,
        "strategy_lineage": lineage,
        "management_lineage": lineage,
        "sold_half": bool(position_state.get("sold_half", False)) if isinstance(position_state, Mapping) else False,
        "sell_signal": management_sell_signal or _sell_signal_as_of(inspection, date_text, entry_date=entry_date),
        "sell_signal_date": management_sell_signal_date,
        "decision_errors": list(inspection.get("decision_errors") or []),
    }



def resolve_trading_single_stock_sidebar_state(
    inspection: Mapping[str, Any] | None,
    date_value: object,
    *,
    strategy_state: Mapping[str, Any] | None = None,
    last_date: str | None = None,
) -> dict[str, Any] | None:
    """Resolve one canonical user-facing Trading lifecycle state.

    Actual POSITION truth comes only from effective account events. Orders and
    Scanner rows are plan evidence. This distinction makes transaction deletion
    authoritative and prevents an immutable broker-order fill from resurrecting a
    voided trade in the single-stock inspector.
    """
    if not isinstance(inspection, Mapping):
        return None
    date_text = _date_text(date_value)
    if date_text is None:
        return None

    position_state = _account_cycle_as_of(inspection, date_text)
    if position_state is not None:
        # Order evidence may still contribute the unfilled reservation for a true
        # partial fill, but it never creates POSITION by itself.
        entry_order_id = str(position_state.get("entry_order_id") or "").strip()
        for order in reversed(list(inspection.get("entry_orders") or [])):
            # Historical ENTRY orders can belong to an earlier position cycle
            # that was later voided.  Only the order explicitly linked to this
            # account cycle may contribute planned/reservation metadata.
            if not entry_order_id or str(order.get("order_id") or "") != entry_order_id:
                continue
            order_state = _order_state_as_of(order, date_text)
            if order_state is None:
                continue
            raw_order_state = str(order_state.get("state") or "")
            position_state["fill_status"] = raw_order_state or None
            position_state["planned_qty"] = order_state.get("planned_qty")
            position_state["remaining_order_qty"] = order_state.get("remaining_order_qty")
            if raw_order_state == "PARTIAL":
                position_state["reserved_capital"] = order_state.get("reserved_capital")
                position_state["original_reserved_capital"] = order_state.get("original_reserved_capital")
            elif raw_order_state == "FILLED":
                position_state["reserved_capital"] = order_state.get("original_reserved_capital")
            break
        position_state["state"] = TRADE_LIFECYCLE_POSITION
        return position_state

    return _prefill_state_as_of(
        inspection,
        date_text,
        strategy_state=strategy_state,
        last_date=last_date,
    )

def _chart_date_labels(chart_payload: Mapping[str, Any]) -> list[str | None]:
    raw_labels = chart_payload.get("date_labels")
    labels = [] if raw_labels is None else list(raw_labels)
    if labels:
        return [_date_text(value) for value in labels]
    raw_dates = chart_payload.get("dates")
    return [_date_text(value) for value in ([] if raw_dates is None else list(raw_dates))]


def _chart_series_values(chart_payload: Mapping[str, Any], key: str) -> list[Any]:
    """Return one chart vector without evaluating array-like truthiness.

    GUI chart payloads can carry NumPy arrays.  ``array or []`` attempts a
    boolean conversion and raises for multi-element arrays, which previously
    prevented manual-managed historical lifecycle replay from rendering at all.
    """

    raw = chart_payload.get(key)
    return [] if raw is None else list(raw)



def _bind_chart_prefill_origin(plan: Mapping[str, Any], lineage: Mapping[str, Any]) -> dict[str, Any]:
    """AI: Unit/display adapters cannot independently date management geometry."""
    fallback_seed = {
        "trade_date": plan.get("information_date"), "entry_type": plan.get("entry_type"),
        "limit_price": plan.get("limit_price"), "init_sl": plan.get("stop_price"),
        "init_trail": plan.get("init_trail"), "target_price": plan.get("tp_price"),
        "entry_atr": plan.get("entry_atr"), "security_profile": plan.get("security_profile"),
        "shadow_position_state": plan.get("shadow_position_state"),
    }
    canonical = build_pending_prefill_plan({
        "origin": "scanner_strategy", "ticker": plan.get("ticker"),
        "planned_trade_date": plan.get("planned_trade_date") if plan.get("confirmed_pending_order") else None,
        "signal_date": plan.get("signal_date"), "information_date": plan.get("information_date"),
        "management_lineage": {**lineage, "frozen_params": lineage.get("frozen_params") or plan.get("frozen_params")},
        "execution_plan_seed": fallback_seed,
        "planned_qty": plan.get("planned_qty"), "reserved_cost": plan.get("reserved_capital"),
        "limit_price": plan.get("limit_price"), "init_sl": plan.get("stop_price"),
        "init_trail": plan.get("init_trail"), "target_price": plan.get("tp_price"),
        "entry_atr": plan.get("entry_atr"),
    })
    result = deepcopy(dict(plan))
    # Accepted quantities/labels are display facts. Only dated management comes
    # from the shared normalizer, just as in pending sync and acquisition.
    for key in ("order_intent", "signal_date", "origin_signal_date", "plan_as_of_date", "limit_price", "stop_price",
                "init_trail", "tp_price", "entry_atr", "shadow_position_state", "shadow_as_of_date"):
        result[key] = canonical[key]
    if canonical.get("origin_reconstruction"):
        result["origin_reconstruction"] = deepcopy(canonical["origin_reconstruction"])
    return result


def _shadow_plan_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize the current Scanner row into the common pre-fill plan contract."""
    if not isinstance(candidate, Mapping) or not candidate:
        return None
    seed = dict(candidate.get("execution_plan_seed") or {})
    information_date = _date_text(candidate.get("trade_date")) or _date_text(seed.get("trade_date"))
    signal_date = _date_text(candidate.get("signal_date")) or information_date
    if information_date is None or signal_date is None:
        return None
    # AI: A candidate's Params and its indicators are one provenance contract.
    # Legacy rows without parameter evidence are display hints only; they may
    # not create authoritative strategy geometry using today's unrelated Params.
    try:
        frozen_params, _member = resolve_trading_candidate_frozen_params(candidate)
    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
        _LOGGER.warning("Candidate has no verifiable frozen lifecycle Params: %s", exc)
        return None
    return _bind_chart_prefill_origin({
        "frozen_params": params_to_json_dict(frozen_params),
        "signal_date": signal_date,
        "information_date": information_date,
        "end_date": None,
        "end_before_date": None,
        "limit_price": seed.get("limit_price", candidate.get("limit_price")),
        "stop_price": seed.get("init_sl"),
        "init_trail": seed.get("init_trail"),
        "tp_price": seed.get("target_price"),
        "entry_atr": seed.get("entry_atr", seed.get("orig_atr")),
        "entry_price": seed.get("shadow_entry_price", seed.get("entry_ref_price")),
        "shadow_position_state": deepcopy(seed.get("shadow_position_state")),
        "planned_qty": candidate.get("proj_qty"),
        "reserved_capital": candidate.get("proj_cost"),
        "sizing_capital": candidate.get("sizing_capital"),
        "ticker": str(candidate.get("ticker") or seed.get("ticker") or ""),
        "security_profile": deepcopy(seed.get("security_profile")),
        "entry_type": str(seed.get("entry_type") or seed.get("entry_source") or candidate.get("kind") or "normal"),
        "source": "scanner_candidate",
        "priority": 10,
    }, {"execution_plan_seed": seed, "candidate_trade_date": information_date, "signal_origin": candidate.get("signal_lineage")})


def _shadow_plan_from_order(
    order: Mapping[str, Any],
    *,
    last_date: str | None,
    effective_fill_date: str | None = None,
) -> dict[str, Any] | None:
    """Normalize an ENTRY order as plan evidence, never as fill truth.

    Broker/order records describe the intended plan and reservation. Actual
    POSITION truth comes only from effective account transactions so a voided
    accounting trade cannot survive in the inspector merely because the immutable
    order audit trail still contains a historical fill.
    """
    if not isinstance(order, Mapping):
        return None
    information_date = _date_text(order.get("information_date")) or _date_text(order.get("ordered_at"))
    if information_date is None:
        return None
    signal_date = _date_text(order.get("signal_date")) or information_date
    cancel_date = _order_cancel_date(order)

    def _order_price(field: str):
        raw = order.get(field)
        if raw is None:
            return None
        try:
            return milli_to_price(int(raw))
        except (TypeError, ValueError):
            return None

    reserved_raw = order.get("reserved_cost_milli")
    return _bind_chart_prefill_origin({
        "signal_date": signal_date,
        "information_date": information_date,
        "end_date": None if effective_fill_date is not None else (cancel_date or last_date),
        "end_before_date": effective_fill_date,
        "limit_price": _order_price("limit_price_milli"),
        "stop_price": _order_price("init_sl_milli"),
        "init_trail": _order_price("init_trail_milli"),
        "tp_price": _order_price("target_price_milli"),
        "entry_atr": _order_price("entry_atr_milli"),
        "entry_price": None,
        "shadow_position_state": deepcopy(order.get("shadow_position_state")),
        "planned_qty": int(order.get("qty") or 0) or None,
        "reserved_capital": None if reserved_raw is None else milli_to_money(int(reserved_raw)),
        "sizing_capital": order.get("sizing_capital"),
        "ticker": str(order.get("ticker") or ""),
        "security_profile": deepcopy(order.get("security_profile")),
        "frozen_params": deepcopy(order.get("frozen_params")),
        "entry_type": str(order.get("entry_type") or order.get("kind") or "normal"),
        "source": "broker_entry_order",
        "planned_trade_date": _date_text(order.get("planned_trade_date")) or information_date,
        "confirmed_pending_order": True,
        "order_id": str(order.get("order_id") or ""),
        "priority": 20,
    }, {"execution_plan_seed": order.get("execution_plan_seed"), "candidate_trade_date": information_date, "signal_origin": order.get("signal_lineage")})


def _shadow_plan_from_strategy_buy_event(
    event: Mapping[str, Any],
    *,
    order_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Recover the pre-fill plan of an *effective* strategy buy from lineage."""
    mutation = str(event.get("mutation_type") or "")
    if mutation not in {TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_MANUAL_MANAGED_BUY}:
        return None
    details = dict(event.get("details") or {})
    entry_date = _date_text(details.get("trade_date"))
    if entry_date is None:
        return None
    lineage_field = "management_lineage" if mutation == TRADE_MUTATION_MANUAL_MANAGED_BUY else "strategy_lineage"
    lineage = details.get(lineage_field)
    if not isinstance(lineage, Mapping):
        position_after = details.get("position_after") or {}
        lineage = position_after.get(lineage_field) if isinstance(position_after, Mapping) else None
    lineage = dict(lineage or {})
    seed = dict(lineage.get("execution_plan_seed") or {})
    if not seed:
        return None
    order_id = str(details.get("entry_order_id") or "")
    matching_order = order_by_id.get(order_id) if order_id else None
    information_date = (
        _date_text(lineage.get("candidate_trade_date"))
        or _date_text(seed.get("trade_date"))
        or (_order_start_date(matching_order) if isinstance(matching_order, Mapping) else None)
        or entry_date
    )
    signal_date = _date_text(lineage.get("signal_date")) or information_date
    # AI: A direct backfill/adoption supplies management origin, not a pending
    # order or strategy signal. Its later registration/information date must
    # never create a new SIGNAL after the confirmed acquisition or exit.
    if signal_date >= entry_date:
        return None
    planned_qty = lineage.get("planned_qty")
    reserved_capital = lineage.get("planned_cost")
    limit_price = seed.get("limit_price")
    stop_price = seed.get("init_sl")
    tp_price = seed.get("target_price")
    entry_price = seed.get("shadow_entry_price", seed.get("entry_ref_price"))
    if isinstance(matching_order, Mapping):
        if planned_qty is None:
            planned_qty = int(matching_order.get("qty") or 0) or None
        if reserved_capital is None and matching_order.get("reserved_cost_milli") is not None:
            reserved_capital = milli_to_money(int(matching_order.get("reserved_cost_milli") or 0))

        def _fallback_order_price(current, field: str):
            if current is not None or matching_order.get(field) is None:
                return current
            try:
                return milli_to_price(int(matching_order.get(field)))
            except (TypeError, ValueError):
                return current

        # Older effective account events may carry only a partial lineage seed.
        # The matching canonical order is still plan evidence, so use it solely
        # to fill missing pre-fill geometry rather than inventing a second plan.
        limit_price = _fallback_order_price(limit_price, "limit_price_milli")
        stop_price = _fallback_order_price(stop_price, "init_sl_milli")
        tp_price = _fallback_order_price(tp_price, "target_price_milli")
    return _bind_chart_prefill_origin({
        "signal_date": signal_date,
        "information_date": information_date,
        "end_date": None,
        "end_before_date": entry_date,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "init_trail": seed.get("init_trail"),
        "tp_price": tp_price,
        "entry_atr": seed.get("entry_atr", seed.get("orig_atr")),
        "entry_price": entry_price,
        "shadow_position_state": deepcopy(seed.get("shadow_position_state")),
        "planned_qty": planned_qty,
        "reserved_capital": reserved_capital,
        "sizing_capital": lineage.get("sizing_capital"),
        "ticker": str(details.get("ticker") or seed.get("ticker") or ""),
        "security_profile": deepcopy(seed.get("security_profile")),
        "frozen_params": deepcopy(lineage.get("frozen_params")),
        "entry_type": str(seed.get("entry_type") or seed.get("entry_source") or lineage.get("candidate_kind") or "normal"),
        "source": "position_manual_management_lineage" if mutation == TRADE_MUTATION_MANUAL_MANAGED_BUY else "position_strategy_lineage",
        "order_id": order_id or None,
        "priority": 30,
    }, lineage)


def _shadow_plan_from_pending_entry(entry: Mapping[str, Any], *, last_date: str | None) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    status = str(entry.get("status") or "")
    supported_statuses = {
        PENDING_ENTRY_STATUS_ACTIVE,
        PENDING_ENTRY_STATUS_CANCELLED_NO_FILL,
        PENDING_ENTRY_STATUS_CANCELLED_USER_DELETED,
        PENDING_ENTRY_STATUS_FILLED,
    }
    if status not in supported_statuses:
        return None
    information_date = _date_text(entry.get("information_date"))
    if information_date is None:
        return None
    origin = str(entry.get("origin") or "")
    manual_pending = origin == "manual_selected"
    planned_trade_date = _date_text(entry.get("planned_trade_date"))
    # AI: Synchronization, acquisition and display share the same dated origin.
    # planned_trade_date remains an order fact, not geometry-availability proof.
    canonical_plan = build_pending_prefill_plan(entry)
    lineage = dict(entry.get("management_lineage") or {})
    seed = dict(lineage.get("execution_plan_seed") or entry.get("execution_plan_seed") or {})
    end_before = None
    end_date = last_date
    if status == PENDING_ENTRY_STATUS_FILLED:
        end_before = _date_text(entry.get("effective_fill_date")) or _date_text((entry.get("fill") or {}).get("trade_date"))
    elif status in {PENDING_ENTRY_STATUS_CANCELLED_NO_FILL, PENDING_ENTRY_STATUS_CANCELLED_USER_DELETED}:
        # Closed pending rows remain immutable audit evidence.  Preserve their
        # historical pre-fill plan only through the persisted order date; live
        # overlays still exclude them via _effective_pending_entries().
        end_date = planned_trade_date or information_date
    return {
        **canonical_plan,
        "end_date": end_date,
        "end_before_date": end_before,
        "entry_price": seed.get("shadow_entry_price", seed.get("entry_ref_price")),
        "planned_qty": int(entry.get("planned_qty") or 0) or None,
        "reserved_capital": entry.get("reserved_cost"),
        "sizing_capital": seed.get("sizing_capital"),
        "frozen_params": deepcopy(lineage.get("frozen_params")),
        "source": "manual_pending_entry" if manual_pending else "scanner_pending_entry",
        "planned_trade_date": planned_trade_date,
        "confirmed_pending_order": True,
        "pending_entry_id": str(entry.get("pending_entry_id") or ""),
        "priority": 40,
    }


def _pending_entry_is_effective_execution_evidence(
    entry: Mapping[str, Any],
    inspection: Mapping[str, Any],
) -> bool:
    """Return whether a pending-entry audit row may enter the live chart overlay.

    ACTIVE rows are current user-confirmed order intent.  FILLED rows remain
    visible only when the effective account journal still contains the managed
    BUY produced from the same frozen lineage.  A deleted/voided account BUY
    must not be resurrected merely because the immutable pending audit row is
    still marked FILLED.
    """

    if not isinstance(entry, Mapping):
        return False
    status = str(entry.get("status") or "")
    if status == PENDING_ENTRY_STATUS_ACTIVE:
        return True
    if status != PENDING_ENTRY_STATUS_FILLED:
        return False

    pending_lineage = entry.get("management_lineage")
    pending_lineage_id = (
        str((pending_lineage or {}).get("lineage_id") or "").strip()
        if isinstance(pending_lineage, Mapping)
        else ""
    )
    if not pending_lineage_id:
        # Current pending-entry schema always carries a frozen lineage.  Fail
        # closed for legacy/malformed FILLED rows rather than inventing a match
        # from ticker/date/price coincidence.
        return False

    return _matching_account_buy_event_for_pending(entry, inspection) is not None


def _matching_account_buy_event_for_pending(
    entry: Mapping[str, Any],
    inspection: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Match one FILLED pending audit row to its effective account BUY event.

    Lineage alone is insufficient for legacy manual pending rows because old
    lineage identities did not include ``planned_trade_date``.  Match the
    immutable lineage together with actual fill date/qty/price so an older order
    cannot be resurrected as a second chart annotation for the current position.
    """

    audit_events = inspection.get("account_audit_events") or inspection.get("account_events") or ()
    for event in inspection.get("account_events") or ():
        if not isinstance(event, Mapping):
            continue
        original = original_buy_event(event, audit_events)
        if pending_entry_matches_buy(entry, original):
            return event
    return None


def _effective_pending_entries(inspection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return current ACTIVE rows plus one canonical FILLED row per account BUY.

    Multiple legacy manual pending rows could share a lineage identity.  When
    they all matched the same effective account event the inspector drew more
    than one ``掛單`` annotation.  Select the audit row whose fill confirmation
    timestamp is closest to the account event timestamp; future rows include the
    planned order date in lineage identity and naturally remain unique.
    """

    # AI: Use the same original-event identity owner as correction validation.
    # An effective correction changes the fill end date, not its order origin.
    pending = [row for row in inspection.get("pending_entries") or () if isinstance(row, Mapping)]
    rows = [deepcopy(dict(row)) for row in pending if str(row.get("status") or "") == PENDING_ENTRY_STATUS_ACTIVE]
    audit = inspection.get("account_audit_events") or inspection.get("account_events") or ()
    seen = set()
    for event in inspection.get("account_events") or ():
        if not isinstance(event, Mapping) or str(event.get("mutation_type") or "") not in {TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_MANUAL_MANAGED_BUY}:
            continue
        matched = resolve_pending_entry_for_buy_event(event, pending, account_events=audit)
        if matched is None:
            continue
        entry_id = str(matched.get("pending_entry_id") or "")
        if entry_id in seen:
            continue
        seen.add(entry_id)
        entry = deepcopy(matched)
        entry["effective_fill_date"] = _trade_event_date(event)
        rows.append(entry)
    rows.sort(key=lambda row: (
        str(row.get("planned_trade_date") or row.get("information_date") or ""),
        str(row.get("pending_entry_id") or ""),
    ))
    return rows


def _plan_covers_date(plan: Mapping[str, Any], date_text: str) -> bool:
    signal_date = _date_text(plan.get("signal_date")) or _date_text(plan.get("information_date"))
    order_date = _date_text((plan.get("order_intent") or {}).get("planned_trade_date"))
    starts = [value for value in (signal_date, order_date) if value is not None]
    if not starts or date_text < min(starts):
        return False
    end_before = _date_text(plan.get("end_before_date"))
    if end_before is not None and date_text >= end_before:
        return False
    end_date = _date_text(plan.get("end_date"))
    return end_date is None or date_text <= end_date


def _build_shadow_plans(
    inspection: Mapping[str, Any],
    *,
    last_date: str | None = None,
) -> list[dict[str, Any]]:
    """Collect all persisted strategy-plan evidence under one normalized contract.

    Scanner, order and effective strategy-fill lineage are merely evidence
    producers. They no longer own separate UI state machines.
    """
    plans: list[dict[str, Any]] = []
    order_by_id = {
        str(row.get("order_id") or ""): row
        for row in list(inspection.get("entry_orders") or [])
        if isinstance(row, Mapping)
    }
    effective_fill_date_by_order: dict[str, str] = {}
    for event in list(inspection.get("account_events") or []):
        mutation = str(event.get("mutation_type") or "")
        if mutation not in _BUY_MUTATIONS:
            continue
        details = dict(event.get("details") or {})
        order_id = str(details.get("entry_order_id") or "")
        event_date = _trade_event_date(event)
        if order_id and event_date is not None:
            prior = effective_fill_date_by_order.get(order_id)
            if prior is None or event_date < prior:
                effective_fill_date_by_order[order_id] = event_date

    candidate_plan = _shadow_plan_from_candidate(dict(inspection.get("candidate") or {}))
    if candidate_plan is not None:
        if last_date is not None:
            candidate_plan["end_date"] = last_date
        plans.append(candidate_plan)

    for pending in _effective_pending_entries(inspection):
        pending_plan = _shadow_plan_from_pending_entry(pending, last_date=last_date)
        if pending_plan is not None:
            plans.append(pending_plan)

    for order in list(inspection.get("entry_orders") or []):
        order_id = str(order.get("order_id") or "")
        plan = _shadow_plan_from_order(
            order,
            last_date=last_date,
            effective_fill_date=effective_fill_date_by_order.get(order_id),
        )
        if plan is not None:
            # A broker fill whose account transaction was later voided remains
            # audit evidence, but it must not outrank the current Scanner plan
            # for the same signal.  With no Scanner candidate it still remains
            # available as SHADOW evidence, preserving historical visibility.
            if list(order.get("fills") or []) and order_id not in effective_fill_date_by_order:
                plan["priority"] = 5
                # AI: A voided fill's closed broker audit is not an outstanding
                # user order. It may retain strategy context, never intent rank.
                plan["order_intent"] = None
                plan["confirmed_pending_order"] = False
            plans.append(plan)

    for event in list(inspection.get("account_events") or []):
        plan = _shadow_plan_from_strategy_buy_event(event, order_by_id=order_by_id)
        if plan is not None:
            plans.append(plan)

    plans.sort(
        key=lambda row: (
            str(_date_text(row.get("signal_date")) or ""),
            int(row.get("priority") or 0),
            str(row.get("source") or ""),
        )
    )
    return plans


def _shadow_state_from_plan(
    plan: Mapping[str, Any],
    *,
    strategy_state: Mapping[str, Any] | None = None,
    lifecycle_state: str = TRADE_LIFECYCLE_SHADOW,
) -> dict[str, Any]:
    """Create one pre-fill state using Research strategy geometry when available."""
    strategy = dict(strategy_state or {})

    def _strategy_or_plan(field: str, plan_field: str):
        value = strategy.get(field)
        return plan.get(plan_field) if value is None else value

    planned_qty = plan.get("planned_qty")
    if planned_qty is None:
        planned_qty = strategy.get("planned_qty")
    reserved_capital = plan.get("reserved_capital")
    if reserved_capital is None:
        reserved_capital = strategy.get("reserved_capital")
    if lifecycle_state == TRADE_LIFECYCLE_SIGNAL:
        display_state = "掛單" if bool(plan.get("confirmed_pending_order")) else "買訊"
    else:
        display_state = "SHADOW"
    return {
        "state": lifecycle_state,
        "display_state": display_state,
        "source": str(plan.get("source") or "shadow_plan"),
        "entry_type": str(plan.get("entry_type") or "normal"),
        "signal_date": _date_text(plan.get("signal_date")),
        "information_date": _date_text(plan.get("information_date")),
        "limit_price": _strategy_or_plan("limit_price", "limit_price"),
        "entry_price": _strategy_or_plan("entry_price", "entry_price"),
        "initial_stop_price": _strategy_or_plan("stop_price", "stop_price"),
        "trailing_stop_price": None,
        "stop_price": _strategy_or_plan("stop_price", "stop_price"),
        "tp_price": _strategy_or_plan("tp_price", "tp_price"),
        "reserved_capital": reserved_capital,
        "buy_capital": None,
        "buy_qty": None,
        "planned_qty": planned_qty,
        "remaining_qty": planned_qty,
        "position_qty": None,
        "remaining_order_qty": planned_qty,
        "sell_signal": False,
    }


def _prefill_state_as_of(
    inspection: Mapping[str, Any],
    date_text: str,
    *,
    strategy_state: Mapping[str, Any] | None = None,
    last_date: str | None = None,
) -> dict[str, Any] | None:
    """Resolve SIGNAL/SHADOW from normalized plan evidence and Research strategy state."""
    candidates: list[dict[str, Any]] = []
    ranges = _account_cycle_ranges(inspection, last_date=last_date or date_text)
    for plan in _build_shadow_plans(inspection, last_date=last_date):
        origin = _date_text(plan.get("origin_signal_date")) or _date_text(plan.get("signal_date")) or _date_text(plan.get("information_date"))
        if not _prefill_lineage_available(origin, date_text, ranges):
            continue
        if _plan_covers_date(plan, date_text):
            candidates.append(plan)
    if not candidates:
        return None
    # Prefer the newest plan, then the strongest persisted evidence for the same
    # signal. This naturally chooses lineage > order > current scanner snapshot.
    plan = max(
        candidates,
        key=lambda row: (
            int(bool(overlay_confirmed_order_intent({}, date_labels=[date_text], plan=row))),
            str(_date_text(row.get("signal_date")) or ""),
            int(row.get("priority") or 0),
        ),
    )
    signal_date = _date_text(plan.get("signal_date")) or _date_text(plan.get("information_date"))
    management = {}
    if signal_date is not None and date_text >= signal_date:
        lifecycle_state = TRADE_LIFECYCLE_SIGNAL if signal_date == date_text else TRADE_LIFECYCLE_SHADOW
        management[0] = _shadow_state_from_plan(
            plan, strategy_state=strategy_state, lifecycle_state=lifecycle_state,
        )
    # AI: The full chart and this date-only compatibility query use the same
    # core order/management join. Never expose future geometry for an old order.
    return overlay_confirmed_order_intent(management, date_labels=[date_text], plan=plan).get(0)

def _account_cycle_ranges(inspection: Mapping[str, Any], *, last_date: str | None) -> list[tuple[str, str]]:
    ticker = str(inspection.get("ticker") or "")
    qty = 0
    cycle_start = None
    ranges: list[tuple[str, str]] = []
    for event in list(inspection.get("account_events") or []):
        mutation = str(event.get("mutation_type") or "")
        details = dict(event.get("details") or {})
        if not _ticker_matches(details, ticker):
            continue
        event_date = _trade_event_date(event)
        if event_date is None:
            continue
        if mutation in _BUY_MUTATIONS:
            add_qty = int(details.get("qty") or details.get("fill_qty") or 0)
            if add_qty <= 0:
                continue
            if qty <= 0:
                cycle_start = event_date
            qty += add_qty
            continue
        if mutation == TRADE_MUTATION_SELL and qty > 0:
            qty = max(0, qty - int(details.get("qty") or 0))
            if qty <= 0 and cycle_start is not None:
                ranges.append((cycle_start, event_date))
                cycle_start = None
    if qty > 0 and cycle_start is not None and last_date is not None:
        ranges.append((cycle_start, last_date))

    # A legacy/current position may predate the retained journal evidence.
    current = inspection.get("current_position")
    if isinstance(current, Mapping) and last_date is not None:
        broker = current.get("broker") or {}
        entry_date = _date_text(broker.get("entry_date"))
        if entry_date is not None and not any(start <= entry_date <= end for start, end in ranges):
            ranges.append((entry_date, last_date))
    return ranges


def _prefill_lineage_available(signal_date, date_text, position_ranges):
    """A real acquisition consumes prior intent; held-period signals never revive.

    AI: A genuinely fresh signal AFTER the completed exit remains eligible.
    Voided BUY events are absent from position_ranges and therefore do not
    suppress the historical unfilled lifecycle.
    """
    if signal_date is None or date_text is None:
        return False
    return not any(start <= date_text and signal_date <= end
                   for start, end in position_ranges)


def _sell_event_position_before(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if str(event.get("mutation_type") or "") != TRADE_MUTATION_SELL:
        return None
    details = event.get("details") or {}
    before = details.get("position_before") if isinstance(details, Mapping) else None
    if not isinstance(before, Mapping):
        return None
    management = before.get("strategy_management") or {}
    state = management.get("position_state") if isinstance(management, Mapping) else None
    return state if isinstance(state, Mapping) else None


def _entry_trace_name(entry_type: object) -> str:
    normalized = str(entry_type or "").strip().lower()
    if normalized in {"reentry", "re-entry", "重進"}:
        return "買進(重進)"
    if normalized in {"extended", "extended_candidate", "延續", "延續候選"}:
        return "買進(延續候選)"
    return "買進"


def _sell_trace_name(event_name: object) -> str:
    normalized = str(event_name or "").strip().upper()
    if normalized == "STOP":
        return "停損賣出"
    if normalized == "TP_HALF":
        return "停利"
    if normalized == "IND_SELL":
        return "\u6307\u6a19\u8ce3\u51fa"
    if normalized in {"MANUAL_ACCOUNT_SELL", "MANUAL_CONFIRMED_SELL", "ACCOUNT_CORRECTION_SELL"}:
        return "\u624b\u52d5\u8ce3\u51fa"
    return "\u8ce3\u51fa\u6210\u4ea4"


def _canonical_marker(
    *,
    trace_name: str,
    date_text: str,
    x: int,
    price: float,
    qty: int,
    note: str,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hover = (
        f"{trace_name}<br>日期: {date_text}<br>成交: {float(price):.2f}"
        f"<br>股數: {int(qty):,}"
    )
    if note:
        hover += f"<br>備註: {note}"
    return {
        "trace_name": trace_name,
        "date": date_text,
        "x": int(x),
        "price": float(price),
        "qty": int(qty),
        "note": str(note or ""),
        "hover_text": hover,
        "meta": {"canonical_trading": True, **dict(meta or {})},
    }



def _resolve_lifecycle_plan_params(plan: Mapping[str, Any], default_params):
    frozen = plan.get("frozen_params")
    if isinstance(frozen, Mapping):
        try:
            return build_params_from_mapping(dict(frozen))
        except (TypeError, ValueError, KeyError) as exc:
            # Old/partial persisted evidence must not crash the inspector or be
            # silently reinterpreted with today's Params.  Skip only this
            # persisted override; the shared strategy-prefill timeline remains
            # available as the canonical fallback for the same historical signal.
            _LOGGER.warning(
                "Ignoring malformed frozen lifecycle params | signal_date=%s | source=%s | error=%s",
                plan.get("signal_date"),
                plan.get("source"),
                exc,
            )
            return None
    return default_params


def _build_trading_prefill_lifecycle_timeline(
    inspection: Mapping[str, Any],
    chart_payload: Mapping[str, Any],
    *,
    params,
) -> dict[int, dict[str, Any]]:
    """Build pre-fill lifecycle solely from frozen strategy-plan evidence.

    Research chart geometry is intentionally not consulted here.  Both Research
    and Trading use the same core pre-fill engine; Trading differs only by the
    date on which effective account truth replaces that shadow path.
    """
    date_labels = _chart_date_labels(chart_payload)
    total = len(date_labels)
    if total <= 0:
        return {}
    authoritative_frame = inspection.get("prefill_market_frame")
    if isinstance(authoritative_frame, pd.DataFrame):
        frame = authoritative_frame
    else:
        # Compatibility for standalone projections without a supplied full frame.
        # The Workbench always supplies its pinned full-precision market input.
        frame = pd.DataFrame({
            name: list(chart_payload.get(key) if chart_payload.get(key) is not None else [float("nan")] * total)
            for name, key in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume"))
        }, index=pd.to_datetime(date_labels))
    chart_indices = {date: idx for idx, date in enumerate(date_labels)}
    finalized = inspection.get("finalized_date")
    if finalized:
        frame = frame.loc[frame.index <= pd.Timestamp(finalized)]
    indicator_cache = {}

    last_date = next((value for value in reversed(date_labels) if value is not None), None)
    ranges = _account_cycle_ranges(inspection, last_date=last_date)
    merged: dict[int, dict[str, Any]] = {}
    owner_key: dict[int, tuple[int, str, int]] = {}

    # Historical strategy signals are first-class pre-fill evidence.  They are
    # generated by the same canonical shadow engine during Research analysis and
    # survive even when there was never a Trading order/fill.  Persisted Trading
    # plan evidence below may override the same signal (for example current
    # Scanner sizing or broker-frozen params), but it must never erase unrelated
    # historical unfilled shadow lifecycles.
    for raw_idx, raw_row in dict(chart_payload.get("strategy_prefill_lifecycle_by_index") or {}).items():
        if not isinstance(raw_row, Mapping):
            continue
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= total:
            continue
        row = deepcopy(dict(raw_row))
        signal_key = str(_date_text(row.get("origin_signal_date")) or _date_text(row.get("signal_date")) or _date_text(row.get("information_date")) or "")
        if finalized and date_labels[idx] and date_labels[idx] > finalized:
            continue
        if not _prefill_lineage_available(signal_key, date_labels[idx], ranges):
            continue
        merged[idx] = row
        owner_key[idx] = (0, signal_key, 0)

    for plan in _build_shadow_plans(inspection, last_date=last_date):
        plan_params = _resolve_lifecycle_plan_params(plan, params)
        if plan_params is None and not plan.get("order_intent"):
            continue
        timeline = build_prefill_lifecycle_from_frame(
            frame=frame, plan=plan, params=plan_params, indicator_cache=indicator_cache,
        )
        signal_key = str(_date_text(plan.get("origin_signal_date")) or _date_text(plan.get("signal_date")) or _date_text(plan.get("information_date")) or "")
        priority = int(plan.get("priority") or 0)
        for source_idx, row in timeline.items():
            # The full source may start before the visible candle window.
            source_date = pd.Timestamp(frame.index[int(source_idx)]).strftime("%Y-%m-%d")
            idx = chart_indices.get(source_date)
            if idx is None:
                continue
            if not _prefill_lineage_available(signal_key, date_labels[idx], ranges):
                continue
            # AI: An accepted order owns its interval ahead of hypothetical
            # later scanner signals; actual POSITION still supersedes both.
            key = (int(bool(row.get("confirmed_pending_order"))), signal_key, priority)
            if idx not in owner_key or key >= owner_key[idx]:
                merged[int(idx)] = deepcopy(dict(row))
                owner_key[int(idx)] = key
    return merged


def _build_manual_managed_position_replay_timeline(
    inspection: Mapping[str, Any], chart_payload: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Compatibility name: project EVERY confirmed cycle with its own binding.

    AI: Closed histories must not disappear when current inventory is empty.
    An open later acquisition cannot supply Params, quantity or SELL obligations
    to an older one. Read-only compatibility replay uses the same domain owner.
    """
    labels = _chart_date_labels(chart_payload)
    indices = {label: idx for idx, label in enumerate(labels)}
    history = inspection.get("position_history")
    if history is None:
        history = collect_confirmed_position_cycles(
            list(inspection.get("account_events") or []), ticker=inspection.get("ticker"),
            current_position=inspection.get("current_position"),
        )
    result = {}

    for cycle in history:
        record = cycle["record"]
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            continue
        projection = cycle.get("projection")
        binding = cycle.get("binding")
        current = inspection.get("current_position") or {}
        is_current = (cycle.get("exit_date") is None and
            (current.get("broker") or {}).get("entry_date") == cycle["entry_date"])
        if is_current:
            binding = binding or inspection.get("position_binding")
            if projection is None:
                projection = inspection.get("management_projection")
        try:
            if binding is None:
                binding = resolve_trading_position_management_binding(record, orders={
                    "orders": {o.get("order_id"): o for o in inspection.get("entry_orders") or []}
                })
            if projection is None:
                required = {key: _chart_series_values(chart_payload, key) for key in ("open", "high", "low", "close", "volume")}
                if not labels or any(len(values) < len(labels) for values in required.values()):
                    raise ValueError("Confirmed management replay requires complete OHLCV evidence")
                frame = pd.DataFrame({key.title(): values[:len(labels)] for key, values in required.items()}, index=pd.to_datetime(labels))
                frame = frame.loc[~frame.index.isna()]
                through = min(filter(None, (inspection.get("finalized_date"), cycle.get("exit_date"), labels[-1])))
                frame = frame.loc[:through]
                if frame.empty or cycle["entry_date"] > through:
                    continue
                params = build_params_from_mapping(dict(binding["frozen_params"]))
                from core.data_utils import get_required_min_rows
                if inspection.get("position_history") is not None and len(frame) < get_required_min_rows(params):
                    raise ValueError("Cropped display frame is not full management history")
                projection = replay_confirmed_position_management(
                    build_confirmed_position_origin(record, binding=binding, params=params,
                        account_events=cycle["events"], frame=frame,
                        prefill_entry=resolve_position_pending_entry(
                            record, inspection.get("pending_entries") or (), account_events=cycle["events"],
                            account_audit_events=inspection.get("account_audit_events") or inspection.get("account_events") or (),
                        )),
                    frame=frame, params=params, start_date=cycle["entry_date"],
                    confirmed_quantity_events=collect_confirmed_position_events(cycle["events"], record, through_date=through),
                )
        except (TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            _LOGGER.warning("Managed cycle %s could not be replayed: %s", cycle["cycle_id"], exc)
            continue
        for session in projection["sessions"]:
            if session["date"] not in indices:
                continue
            position = session["position_state"]
            obligation = project_full_exit_obligation(session["full_exit_obligation"]) or {}
            row = build_trade_lifecycle_row(
                TRADE_LIFECYCLE_POSITION, source="confirmed_management_projection",
                information_date=session["date"], entry_type=position.get("entry_type"),
                limit_price=position.get("limit_price"), entry_price=position.get("entry_fill_price"),
                stop_price=position.get("sl"), tp_price=None if position.get("sold_half") else position.get("tp_half"),
                buy_qty=position.get("initial_qty"), position_qty=position.get("qty"), remaining_order_qty=0,
                entry_date=cycle["entry_date"], initial_stop_price=position.get("initial_stop"),
                trailing_stop_price=position.get("trailing_stop"), sell_signal=obligation.get("sell_signal"),
                sell_signal_date=obligation.get("sell_signal_date"), sold_half=bool(position.get("sold_half")),
                management_lineage=deepcopy(record.get("management_lineage") or record.get("strategy_lineage")),
                cycle_id=cycle["cycle_id"], management_binding_sha256=binding.get("frozen_params_sha256"),
                confirmed_inventory_evidence=not cycle.get("has_buy_event"),
            )
            row["position_qty"] = int(position.get("qty") or 0)
            result[indices[session["date"]]] = row
    return result


def build_trading_single_stock_lifecycle_timeline(
    inspection: Mapping[str, Any] | None,
    chart_payload: Mapping[str, Any] | None,
    *,
    params=None,
) -> dict[int, dict[str, Any]]:
    """Build the single canonical Trading timeline consumed by chart and sidebar.

    SIGNAL/SHADOW comes from the shared core lifecycle engine using frozen plan
    evidence.  POSITION comes only from effective account events.  No rendered
    Research line, simulated Research fill, or immutable broker fill can create
    an actual Trading position.
    """
    if not isinstance(inspection, Mapping) or not isinstance(chart_payload, Mapping):
        return {}
    date_labels = _chart_date_labels(chart_payload)
    prefill = _build_trading_prefill_lifecycle_timeline(
        inspection, chart_payload, params=params
    )
    manual_replay = _build_manual_managed_position_replay_timeline(inspection, chart_payload)
    timeline: dict[int, dict[str, Any]] = {}
    for idx, date_text in enumerate(date_labels):
        if date_text is None:
            continue
        position_state = _account_cycle_as_of(inspection, date_text)
        if position_state is None and int(idx) in manual_replay:
            # AI: A recorded broker adoption is actual inventory evidence even
            # before the later management-activation date. A replay by itself
            # can NEVER create POSITION or a BUY marker.
            row = manual_replay[int(idx)]
            if row.get("confirmed_inventory_evidence"):
                position_state = deepcopy(row)
                position_state["source"] = "confirmed_manual_adoption_projection"
                position_state["display_state"] = "\u5df2\u51fa\u5834" if row.get("position_qty") == 0 else "\u6301\u80a1"
        if position_state is not None:
            replay_state = manual_replay.get(int(idx))
            if isinstance(replay_state, Mapping):
                # Broker/event history remains authoritative for lifecycle and
                # quantity.  The frozen-Params replay owns date-aware management
                # geometry for every managed source.
                position_state["entry_price"] = replay_state.get("entry_price")
                position_state["initial_stop_price"] = replay_state.get("initial_stop_price")
                position_state["trailing_stop_price"] = replay_state.get("trailing_stop_price")
                position_state["stop_price"] = replay_state.get("stop_price")
                position_state["tp_price"] = replay_state.get("tp_price")
                position_state["sold_half"] = replay_state.get("sold_half", False)
                position_state["sell_signal"] = replay_state.get("sell_signal")
                position_state["sell_signal_date"] = replay_state.get("sell_signal_date")
                position_state["cycle_id"] = replay_state.get("cycle_id")
                position_state["management_binding_sha256"] = replay_state.get("management_binding_sha256")
                position_state["management_geometry_source"] = "confirmed_management_projection"
                position_state["decision_errors"] = [error for error in position_state.get("decision_errors") or []
                                                      if not str(error).startswith("management_projection[")]
            elif any(str(error).startswith("management_projection[") for error in inspection.get("decision_errors") or []):
                position_state["management_geometry_source"] = "persisted_account_evidence"
                position_state["management_projection_error"] = "Full management history unavailable; retained confirmed evidence"
            # Order evidence can contribute reservation metadata for a genuine
            # partial fill but never creates POSITION by itself.
            entry_order_id = str(position_state.get("entry_order_id") or "").strip()
            for order in reversed(list(inspection.get("entry_orders") or [])):
                if not entry_order_id or str(order.get("order_id") or "") != entry_order_id:
                    continue
                order_state = _order_state_as_of(order, date_text)
                if order_state is None:
                    continue
                raw_order_state = str(order_state.get("state") or "")
                position_state["fill_status"] = raw_order_state or None
                position_state["planned_qty"] = order_state.get("planned_qty")
                position_state["remaining_order_qty"] = order_state.get("remaining_order_qty")
                if raw_order_state == "PARTIAL":
                    position_state["reserved_capital"] = order_state.get("reserved_capital")
                    position_state["original_reserved_capital"] = order_state.get("original_reserved_capital")
                elif raw_order_state == "FILLED":
                    position_state["reserved_capital"] = order_state.get("original_reserved_capital")
                break
            if position_state.get("position_qty") == 0:
                position_state["sell_signal"] = None
                position_state["sell_signal_date"] = None
            position_state["state"] = TRADE_LIFECYCLE_POSITION
            timeline[int(idx)] = deepcopy(dict(position_state))
            continue
        state = prefill.get(int(idx))
        if isinstance(state, Mapping):
            timeline[int(idx)] = deepcopy(dict(state))
    return timeline

def _lifecycle_state_by_date(
    lifecycle_by_index: Mapping[int, Mapping[str, Any]],
    date_to_x: Mapping[str, int],
    date_text: str,
) -> dict[str, Any]:
    idx = date_to_x.get(date_text)
    if idx is None:
        return {}
    state = lifecycle_by_index.get(int(idx))
    return deepcopy(dict(state)) if isinstance(state, Mapping) else {}


def _canonical_account_marker_groups(
    inspection: Mapping[str, Any],
    *,
    date_to_x: Mapping[str, int],
    lifecycle_by_index: Mapping[int, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    ticker = str(inspection.get("ticker") or "")
    buy_buckets: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    order_by_id = {
        str(row.get("order_id") or ""): row
        for row in list(inspection.get("entry_orders") or [])
        if isinstance(row, Mapping)
    }

    for event in list(inspection.get("account_events") or []):
        mutation = str(event.get("mutation_type") or "")
        details = dict(event.get("details") or {})
        if not _ticker_matches(details, ticker):
            continue
        date_text = _trade_event_date(event)
        if date_text is None or date_text not in date_to_x:
            continue
        if mutation in _BUY_MUTATIONS:
            qty = int(details.get("qty") or details.get("fill_qty") or 0)
            price_milli = details.get("fill_price_milli")
            if price_milli is None:
                price_milli = details.get("entry_fill_price_milli")
            if qty <= 0 or price_milli is None:
                continue
            bucket = buy_buckets.setdefault(
                date_text,
                {"qty": 0, "weighted_price_milli": 0, "buy_capital_milli": 0, "entry_type": "normal", "order_id": ""},
            )
            bucket["qty"] += qty
            bucket["weighted_price_milli"] += int(price_milli) * qty
            bucket["buy_capital_milli"] += int(details.get("net_buy_total_milli") or 0)
            order_id = str(details.get("entry_order_id") or "")
            if order_id:
                bucket["order_id"] = order_id
                order = order_by_id.get(order_id) or {}
                bucket["entry_type"] = str(order.get("entry_type") or order.get("kind") or bucket["entry_type"])
            position_after = details.get("position_after")
            if isinstance(position_after, Mapping):
                state = ((position_after.get("strategy_management") or {}).get("position_state") or {})
                if isinstance(state, Mapping):
                    bucket["entry_type"] = str(state.get("entry_type") or bucket["entry_type"])
            continue

        if mutation != TRADE_MUTATION_SELL:
            continue
        qty = int(details.get("qty") or 0)
        price_milli = details.get("exec_price_milli")
        if qty <= 0 or price_milli is None:
            continue
        trace_name = _sell_trace_name(details.get("event"))
        allocated = int(details.get("allocated_cost_milli") or 0)
        pnl_milli = int(details.get("realized_pnl_milli") or 0)
        pnl_pct = None if allocated <= 0 else (pnl_milli / allocated) * 100.0
        marker = _canonical_marker(
            trace_name=trace_name,
            date_text=date_text,
            x=date_to_x[date_text],
            price=milli_to_price(int(price_milli)),
            qty=qty,
            note="Trading 實際賣出成交",
            meta={
                "sell_capital": milli_to_money(int(details.get("net_sell_total_milli") or 0)),
                "pnl_value": milli_to_money(pnl_milli),
                "pnl_pct": pnl_pct,
                "result": str(details.get("event") or "\u5be6\u969b\u6210\u4ea4"),
                "account_event_revision": event.get("revision"),
                "execution_reason": details.get("event"),
                "position_source": details.get("position_source"),
            },
        )
        groups.setdefault(trace_name, []).append(marker)

    for date_text, bucket in buy_buckets.items():
        qty = int(bucket["qty"])
        if qty <= 0:
            continue
        average_price = milli_to_price((int(bucket["weighted_price_milli"]) + qty // 2) // qty)
        state = _lifecycle_state_by_date(lifecycle_by_index, date_to_x, date_text)
        trace_name = _entry_trace_name(bucket.get("entry_type"))
        marker = _canonical_marker(
            trace_name=trace_name,
            date_text=date_text,
            x=date_to_x[date_text],
            price=average_price,
            qty=qty,
            note="Trading 實際買入成交",
            meta={
                "entry_type": bucket.get("entry_type") or "normal",
                "limit_price": state.get("limit_price"),
                "entry_price": average_price,
                "stop_price": state.get("stop_price"),
                "tp_price": state.get("tp_price"),
                "buy_capital": milli_to_money(int(bucket.get("buy_capital_milli") or 0)),
                "order_id": bucket.get("order_id") or "",
                "result": "實際成交",
            },
        )
        groups.setdefault(trace_name, []).append(marker)

    for markers in groups.values():
        markers.sort(key=lambda row: (int(row.get("x") or 0), str(row.get("trace_name") or "")))
    return groups


def _canonical_trading_sell_signal_annotations(
    *,
    lifecycle_by_index: Mapping[int, Mapping[str, Any]],
    date_labels: list[str | None],
    chart_payload: Mapping[str, Any],
    indicator_exit: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Project Trading SELL annotations from the same lifecycle used by sidebar.

    The single-stock sidebar consumes ``trading_lifecycle_by_index``.  SELL
    labels must consume that same state rather than independently deciding
    visibility from the current indicator-exit plan; otherwise a frozen-Params
    replay can correctly show ``INDICATOR SELL`` in the sidebar while the chart
    silently omits the signal annotation.

    Current indicator-exit rows remain useful audit metadata only.  Persistent
    obligations are de-duplicated by position cycle + SELL kind so one carried
    signal does not paint a label on every later bar.
    """

    exit_rows = []
    current_indicator = indicator_exit if isinstance(indicator_exit, Mapping) else {}
    if bool(current_indicator.get("fresh")):
        exit_rows = [
            dict(row)
            for row in list(current_indicator.get("exits") or [])
            if isinstance(row, Mapping)
        ]

    annotations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    high_values = _chart_series_values(chart_payload, "high")
    close_values = _chart_series_values(chart_payload, "close")

    for idx in sorted(int(value) for value in lifecycle_by_index):
        state = lifecycle_by_index.get(idx)
        if not isinstance(state, Mapping):
            continue
        if str(state.get("state") or "") != TRADE_LIFECYCLE_POSITION:
            continue
        sell_signal = str(state.get("sell_signal") or "").strip().upper()
        if not sell_signal:
            continue
        if idx < 0 or idx >= len(date_labels):
            continue
        signal_date = _date_text(date_labels[idx])
        if signal_date is None:
            continue

        cycle_key = str(state.get("entry_order_id") or "").strip()
        if not cycle_key:
            cycle_key = str(_date_text(state.get("entry_date")) or state.get("source") or "")
        if not cycle_key:
            cycle_key = f"position@{idx}"
        dedupe_key = (cycle_key, sell_signal)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        anchor = None
        for values in (high_values, close_values):
            try:
                candidate = float(values[idx])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(candidate):
                anchor = candidate
                break
        if anchor is None:
            for field in ("stop_price", "entry_price"):
                try:
                    candidate = float(state.get(field))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(candidate):
                    anchor = candidate
                    break
        if anchor is None:
            continue

        matched_exit = None
        same_date_exits = [
            row for row in exit_rows
            if _date_text(row.get("signal_information_date")) == signal_date
        ]
        state_entry = str(state.get("entry_order_id") or "").strip()
        for row in same_date_exits:
            row_entry = str(row.get("entry_order_id") or "").strip()
            if not state_entry or not row_entry or row_entry == state_entry:
                matched_exit = row
                break
        # The Indicator plan uses the canonical management-lineage key while
        # historical account replay can expose the broker ENTRY order id.  In a
        # single-ticker inspector a unique same-date exit row is unambiguous and
        # may safely enrich the lifecycle annotation with audit metadata.
        if matched_exit is None and len(same_date_exits) == 1:
            matched_exit = same_date_exits[0]

        meta = {
            "canonical_trading": True,
            "canonical_lifecycle_sell": True,
            "sell_signal": sell_signal,
            "entry_order_id": state.get("entry_order_id"),
        }
        if matched_exit is not None:
            meta.update({
                "canonical_indicator_exit": sell_signal == "INDICATOR SELL",
                "signal_key": matched_exit.get("signal_key"),
                "qty": matched_exit.get("qty"),
                "entry_order_id": matched_exit.get("entry_order_id") or state.get("entry_order_id"),
                "frozen_params_sha256": matched_exit.get("frozen_params_sha256"),
            })

        note = (
            "Trading frozen Params 指標賣出訊號"
            if sell_signal == "INDICATOR SELL"
            else f"Trading canonical SELL 訊號：{sell_signal}"
        )
        annotations.append({
            "date": signal_date,
            "x": idx,
            "anchor_price": anchor,
            "signal_type": "sell",
            "title": "賣出訊號",
            "detail_text": "",
            "note": note,
            "meta": meta,
        })
    return annotations


def project_trading_single_stock_chart_payload(
    chart_payload: Mapping[str, Any] | None,
    inspection: Mapping[str, Any] | None,
    *,
    params=None,
) -> dict[str, Any]:
    """Build the Trading execution layer on top of Research strategy context.

    Research remains the owner of K-lines, indicators, strategy signals and
    historical performance.  Its simulated transaction geometry is never mixed
    with Trading broker/account truth.  In Trading mode all transaction lines
    and transaction markers are rebuilt exclusively from one persisted Trading
    lifecycle timeline.  The sidebar consumes that same timeline through the
    chart hover snapshot, so chart and table cannot resolve separate states.
    """
    payload = deepcopy(dict(chart_payload or {}))
    if not payload or not isinstance(inspection, Mapping):
        return payload

    date_labels = _chart_date_labels(payload)
    total = len(date_labels)
    if total <= 0:
        return payload
    last_date = next((value for value in reversed(date_labels) if value is not None), None)
    date_to_x = {value: idx for idx, value in enumerate(date_labels) if value is not None}
    nan = float("nan")

    # Transaction lifecycle is built from frozen strategy-plan evidence plus the
    # shared canonical shadow engine.  Research rendered lines are never used as
    # input, so simulated fills cannot leak into Trading SHADOW geometry.
    lines = {key: [nan] * total for key in TRADE_TRANSACTION_LINE_KEYS}
    lifecycle_by_index = build_trading_single_stock_lifecycle_timeline(
        inspection,
        payload,
        params=params,
    )
    for idx, state_copy in lifecycle_by_index.items():
        lifecycle_state = str(state_copy.get("state") or "")
        for key, value in iter_trade_lifecycle_line_values(
            lifecycle_state,
            stop_price=state_copy.get("stop_price"),
            tp_price=state_copy.get("tp_price"),
            limit_price=state_copy.get("limit_price"),
            entry_price=state_copy.get("entry_price"),
            pending_limit_price=state_copy.get("pending_limit_line"),
        ):
            if value is not None:
                _assign_chart_float(
                    lines,
                    key=key,
                    idx=int(idx),
                    value=value,
                    evidence=f"lifecycle:{lifecycle_state}:{key}",
                )

    # A full-exit bar still has exact pre-fill position evidence even though the
    # post-fill account has no open position.  Draw only that Trading-owned
    # geometry; never fall back to Research's simulated position.
    for event in list(inspection.get("account_events") or []):
        event_date = _trade_event_date(event)
        if event_date is None or event_date not in date_to_x:
            continue
        state = _sell_event_position_before(event)
        if not isinstance(state, Mapping):
            continue
        idx = date_to_x[event_date]
        if (lifecycle_by_index.get(idx) or {}).get("management_geometry_source") == "confirmed_management_projection":
            continue
        values = {
            "entry_line": state.get("entry_fill_price"),
            "stop_line": state.get("sl", state.get("initial_stop")),
            "tp_line": None if bool(state.get("sold_half", False)) else state.get("tp_half"),
            "limit_line": state.get("limit_price"),
        }
        for key, value in values.items():
            if value is not None:
                _assign_chart_float(
                    lines, key=key, idx=idx, value=value,
                    evidence=f"sell_event_position_before:{key}",
                )

    for key, values in lines.items():
        payload[key] = values

    # Research future-preview is simulated execution evidence and must not leak
    # into Trading mode.  Rebuild it only from the latest Trading SIGNAL/SHADOW.
    payload["future_preview"] = {}
    if last_date is not None:
        last_idx = date_to_x.get(last_date)
        last_state = None if last_idx is None else lifecycle_by_index.get(last_idx)
        if isinstance(last_state, Mapping):
            last_lifecycle = str(last_state.get("state") or "")
            if last_lifecycle in {TRADE_LIFECYCLE_SIGNAL, TRADE_LIFECYCLE_PENDING}:
                # Match Research fresh-signal semantics exactly: after today's
                # completed signal/order bar, only the next-session buy limit is
                # previewable.  Stop/TP/entry are planned sidebar values but are
                # not active or shadow geometry until a later completed bar.
                payload["future_preview"] = {
                    "limit_price": last_state.get("limit_price"),
                    "stop_price": None,
                    "tp_half_price": None,
                    "entry_price": None,
                }
            elif last_lifecycle == TRADE_LIFECYCLE_SHADOW:
                payload["future_preview"] = {
                    "limit_price": last_state.get("limit_price"),
                    "stop_price": last_state.get("stop_price"),
                    "tp_half_price": last_state.get("tp_price"),
                    "entry_price": last_state.get("entry_price"),
                }

    # Strategy signals remain useful context.  Research simulated orders/fills
    # do not: Trading buy/sell icons are created only from confirmed account
    # events.  Remove transaction traces globally, not only on lifecycle bars.
    marker_groups = {
        str(trace_name): [deepcopy(dict(marker)) for marker in list(markers or [])
                          if not (marker.get("meta") or {}).get("canonical_trading")]
        for trace_name, markers in dict(payload.get("marker_groups") or {}).items()
        if str(trace_name) not in _TRADING_REPLAY_TRANSACTION_TRACES
    }
    canonical_groups = _canonical_account_marker_groups(
        inspection, date_to_x=date_to_x, lifecycle_by_index=lifecycle_by_index
    )
    for trace_name, markers in canonical_groups.items():
        marker_groups.setdefault(trace_name, []).extend(markers)
        marker_groups[trace_name].sort(key=lambda row: int(row.get("x") or 0))
    payload["marker_groups"] = marker_groups

    # Keep Research BUY anchors as context, but Trading SELL must come from the
    # current holding's frozen Params / canonical Indicator SELL obligation.
    # Reusing Research SELL annotations here could silently use a different
    # parameter lineage than the actual managed position.
    signal_annotations = []
    position_ranges = _account_cycle_ranges(inspection, last_date=last_date)
    for raw in list(payload.get("signal_annotations") or []):
        item = deepcopy(dict(raw))
        signal_type = str(item.get("signal_type") or "").lower()
        if signal_type == "sell":
            continue
        if signal_type == "buy":
            meta = dict(item.get("meta") or {})
            if meta.get("canonical_trading_pending"):
                continue
            signal_date = _date_text(item.get("date"))
            if signal_date is None and item.get("x") is not None and 0 <= int(item["x"]) < total:
                signal_date = date_labels[int(item["x"])]
            if signal_date and not _prefill_lineage_available(signal_date, signal_date, position_ranges):
                continue
            for key in ("qty", "reserved_capital", "buy_capital"):
                meta.pop(key, None)
            item["meta"] = meta
            item["detail_text"] = ""
        signal_annotations.append(item)

    signal_annotations.extend(
        _canonical_trading_sell_signal_annotations(
            lifecycle_by_index=lifecycle_by_index,
            date_labels=date_labels,
            chart_payload=payload,
            indicator_exit=inspection.get("indicator_exit"),
        )
    )

    # Pending-entry intent is a distinct Trading order layer, not a Research BUY
    # signal.  Keep Research signal anchors intact and add a dedicated order
    # annotation on the persisted planned_trade_date.  Cancelled/deleted rows
    # remain auditable in pending-entry state but are intentionally absent from
    # the current execution overlay.
    for pending in _effective_pending_entries(inspection):
        status = str(pending.get("status") or "")
        pending_date = _date_text(pending.get("planned_trade_date")) or _date_text(pending.get("information_date"))
        if pending_date is None or pending_date not in date_to_x:
            continue
        x = int(date_to_x[pending_date])
        meta = {
            "qty": int(pending.get("planned_qty") or 0),
            "reserved_capital": pending.get("reserved_cost"),
            "pending_entry_id": pending.get("pending_entry_id"),
            "status": status,
            "planned_trade_date": pending_date,
            "information_date": _date_text(pending.get("information_date")),
            "origin": pending.get("origin"),
            "limit_price": pending.get("limit_price"),
            "canonical_trading_pending": True,
        }
        highs = payload.get("high")
        anchor = None
        try:
            anchor = float(highs[x])
        except (TypeError, ValueError, IndexError):
            anchor = float(pending.get("limit_price") or 0.0)
        signal_annotations.append({
            "date": pending_date,
            "x": x,
            "anchor_price": anchor,
            "signal_type": "buy",
            "title": "掛單",
            "detail_text": "",
            "note": "Trading 掛單",
            "meta": meta,
        })
    signal_annotations.sort(key=lambda item: (int(item.get("x", -1)), str(item.get("title") or "")))
    payload["signal_annotations"] = signal_annotations

    # Navigation/focus must follow the visible Trading execution layer, not stale
    # Research simulated transactions that have just been removed.
    focus_positions = {
        int(item.get("x"))
        for item in signal_annotations
        if item.get("x") is not None
    }
    for markers in marker_groups.values():
        for marker in markers:
            if marker.get("x") is not None:
                focus_positions.add(int(marker.get("x")))
    payload["focus_positions"] = sorted(focus_positions)

    # Store the complete per-bar state.  build_chart_hover_snapshot carries this
    # exact object to the right sidebar, eliminating the former second resolver.
    payload["trading_lifecycle_by_index"] = {
        int(idx): deepcopy(dict(state)) for idx, state in lifecycle_by_index.items()
    }
    payload["trading_overlay_source"] = "canonical_execution_lifecycle"

    # Research computed these windows before Trading replaced transaction
    # geometry/focus/future-preview.  Keeping the stale window can place a new
    # Trading future preview outside the visible x-range, which looks like a
    # SIGNAL/SHADOW state with no L/S/TP lines.  Let the shared chart normalizer
    # recompute both windows from the projected Trading payload.
    payload.pop("default_view", None)
    payload.pop("gui_render_window", None)
    return payload


__all__ = [
    "build_trading_single_stock_inspection",
    "build_trading_single_stock_lifecycle_timeline",
    "load_trading_single_stock_position_binding",
    "project_trading_single_stock_chart_payload",
    "resolve_trading_single_stock_sidebar_state",
]
