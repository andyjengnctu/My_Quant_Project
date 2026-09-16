"""Canonical Trading state projection for the single-stock Workbench inspector.

The Trading single-stock chart may replay historical strategy geometry for visual
context, but broker/account truth must never be inferred from that replay.  This
module projects persisted Scanner, broker-order, fill, account-position and
roll-forward evidence into a date-aware sidebar state without duplicating entry
or stop formulas.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from core.entry_plans import build_position_from_entry_fill
from core.exact_accounting import calc_entry_total_cost, milli_to_money, milli_to_price
from core.params_io import build_params_from_mapping
from core.trading_account_state import (
    POSITION_SOURCE_STRATEGY_FILL,
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_SELL,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
    effective_trading_account_events,
)
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from core.trading_order_state import (
    TRADING_ORDER_PURPOSE_ENTRY,
    TRADING_ORDER_SIDE_BUY,
)
from services.trading.account_state import load_trading_account_state
from services.trading.accounting_policy import overlay_trading_accounting_params
from services.trading.order_state import load_trading_order_state
from services.trading.protection_planning import get_trading_protection_plan_read_model
from services.trading.indicator_exit_planning import get_trading_indicator_exit_plan_read_model
from services.trading.strategy_param_runtime import resolve_trading_position_strategy_binding


_BUY_MUTATIONS = {
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
}


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
    if not isinstance(record, Mapping) or str(record.get("source") or "") != POSITION_SOURCE_STRATEGY_FILL:
        return None
    orders = load_trading_order_state(root, required=False) or {"orders": {}}
    return resolve_trading_position_strategy_binding(record, orders=orders)


def build_trading_single_stock_inspection(
    project_root: str | Path,
    ticker: object,
    *,
    candidate_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read canonical Trading evidence once for date-aware inspector rendering."""
    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    account = load_trading_account_state(root, required=False)
    orders = load_trading_order_state(root, required=False)

    candidate = dict(candidate_row or {})
    if candidate and not _ticker_matches(candidate, ticker_key):
        candidate = {}

    account_events: list[dict[str, Any]] = []
    current_position = None
    position_binding = None
    if account:
        account_events = [
            deepcopy(event)
            for event in effective_trading_account_events(account)
            if _ticker_matches(event.get("details") or {}, ticker_key)
            or any(
                _ticker_matches(row, ticker_key)
                for row in list((event.get("details") or {}).get("positions") or [])
            )
        ]
        current_position = deepcopy((account.get("positions") or {}).get(ticker_key))
        if isinstance(current_position, Mapping) and str(current_position.get("source") or "") == POSITION_SOURCE_STRATEGY_FILL:
            position_binding = resolve_trading_position_strategy_binding(
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

    return {
        "ticker": ticker_key,
        "candidate": candidate,
        "account_revision": None if not account else int(account.get("revision") or 0),
        "order_revision": None if not orders else int(orders.get("revision") or 0),
        "account_events": account_events,
        "entry_orders": entry_orders,
        "current_position": deepcopy(current_position),
        "position_binding": deepcopy(position_binding),
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


def _candidate_state(candidate: Mapping[str, Any], date_text: str) -> dict[str, Any] | None:
    if not candidate:
        return None
    trade_date = _date_text(candidate.get("trade_date"))
    if trade_date != date_text:
        return None
    seed = dict(candidate.get("execution_plan_seed") or {})
    limit_price = seed.get("limit_price", candidate.get("limit_price"))
    return {
        "state": "SCANNER",
        "source": "scanner_candidate",
        "limit_price": limit_price,
        "entry_price": None,
        "initial_stop_price": seed.get("init_sl"),
        "trailing_stop_price": seed.get("init_trail"),
        "stop_price": seed.get("init_sl"),
        "tp_price": seed.get("target_price"),
        "reserved_capital": candidate.get("proj_cost"),
        "buy_capital": None,
        "buy_qty": None,
        "remaining_qty": candidate.get("proj_qty"),
        "sell_signal": False,
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
    entry_atr_milli = order.get("entry_atr_milli")
    return build_position_from_entry_fill(
        buy_price=average_fill_price,
        qty=total_qty,
        init_sl=milli_to_price(int(order["init_sl_milli"])),
        init_trail=milli_to_price(int(order["init_trail_milli"])),
        target_price=milli_to_price(int(order["target_price_milli"])),
        limit_price=milli_to_price(int(order["limit_price_milli"])),
        entry_atr=(None if entry_atr_milli is None else milli_to_price(int(entry_atr_milli))),
        params=params,
        entry_type=str(order.get("entry_type") or order.get("kind") or "normal"),
        ticker=str(order.get("ticker") or ""),
        security_profile=deepcopy(order.get("security_profile")),
        trade_date=str(fills[-1].get("trade_date") or ""),
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
        "remaining_qty": remaining_qty,
        "sell_signal": False,
    }


def _trade_event_date(event: Mapping[str, Any]) -> str | None:
    details = event.get("details") or {}
    mutation = str(event.get("mutation_type") or "")
    if mutation in _BUY_MUTATIONS or mutation == TRADE_MUTATION_SELL:
        return _date_text(details.get("trade_date"))
    return _date_text(event.get("timestamp"))


def _current_position_effective_date(
    inspection: Mapping[str, Any],
    *,
    ticker: str,
    entry_date: str,
) -> str:
    """Return the first date on which today's persisted open-position state is valid.

    The current account record may include a later partial fill/sell or roll-forward
    than the chart date being inspected.  Gate its use on the latest persisted
    event that contributed to this still-open position cycle so current state never
    leaks backward into an earlier point-in-time view.
    """
    effective_dates = [entry_date]
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


def _sell_signal_as_of(inspection: Mapping[str, Any], date_text: str) -> str | None:
    """Project the same current canonical SELL obligations shown by Trading Center."""
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

        if not _ticker_matches(details, ticker):
            continue
        event_date = _trade_event_date(event)
        if event_date is None or event_date > date_text:
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
            qty += add_qty
            entry_qty += add_qty
            gross_buy_milli += int(details.get("gross_buy_milli") or 0)
            net_buy_milli += int(details.get("net_buy_total_milli") or 0)
            if mutation == TRADE_MUTATION_STRATEGY_BUY:
                position_after = details.get("position_after")
                if isinstance(position_after, Mapping):
                    management = position_after.get("strategy_management") or {}
                    candidate_position = management.get("position_state")
                    if isinstance(candidate_position, Mapping):
                        position_state = deepcopy(dict(candidate_position))
                if isinstance(details.get("strategy_lineage"), Mapping):
                    lineage = deepcopy(dict(details.get("strategy_lineage") or {}))
                entry_order_id = str(details.get("entry_order_id") or "") or None
            continue

        if mutation == TRADE_MUTATION_SELL and qty > 0:
            sell_qty = int(details.get("qty") or 0)
            qty = max(0, qty - sell_qty)
            if qty <= 0:
                closed_date = event_date

    if entry_date is None or closed_date is not None and closed_date <= date_text:
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
        current_state = management.get("position_state")
        current_effective_date = (
            _current_position_effective_date(inspection, ticker=ticker, entry_date=entry_date)
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
    if isinstance(position_state, Mapping):
        average_entry = position_state.get("entry_fill_price", average_entry)

    return {
        "state": "POSITION",
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
        "tp_price": None if not isinstance(position_state, Mapping) else position_state.get("tp_half"),
        "reserved_capital": None,
        "buy_capital": None if net_buy_milli <= 0 else milli_to_money(net_buy_milli),
        "buy_qty": entry_qty or None,
        "remaining_qty": qty,
        "last_rollforward_date": last_rollforward_date,
        "strategy_lineage": lineage,
        "sell_signal": _sell_signal_as_of(inspection, date_text),
        "decision_errors": list(inspection.get("decision_errors") or []),
    }


def resolve_trading_single_stock_sidebar_state(
    inspection: Mapping[str, Any] | None,
    date_value: object,
) -> dict[str, Any] | None:
    """Resolve Trading transaction fields for one selected market date.

    Priority is broker/account truth, then active broker order, then the current
    Scanner plan.  No state transition is inferred merely because a calendar day
    followed a buy signal.
    """
    if not isinstance(inspection, Mapping):
        return None
    date_text = _date_text(date_value)
    if date_text is None:
        return None

    candidate = _candidate_state(dict(inspection.get("candidate") or {}), date_text)

    order_states = []
    for order in list(inspection.get("entry_orders") or []):
        state = _order_state_as_of(order, date_text)
        if state is not None:
            order_states.append(state)
    order_state = order_states[-1] if order_states else None

    position_state = _account_cycle_as_of(inspection, date_text)
    if position_state is not None:
        # A still-active PARTIAL order contributes the remaining reservation while
        # account state supplies the actual fill, stop and target truth.
        if order_state is not None and str(order_state.get("state") or "") == "PARTIAL":
            position_state["state"] = "PARTIAL"
            position_state["reserved_capital"] = order_state.get("reserved_capital")
            position_state["original_reserved_capital"] = order_state.get("original_reserved_capital")
        elif order_state is not None and str(order_state.get("state") or "") == "FILLED":
            # Preserve the original plan reservation on the actual fill date for
            # plan-vs-fill inspection; it is not a remaining cash lock.
            position_state["state"] = "FILLED"
            position_state["reserved_capital"] = order_state.get("original_reserved_capital")
        return position_state

    if order_state is not None:
        return order_state
    return candidate


__all__ = [
    "build_trading_single_stock_inspection",
    "load_trading_single_stock_position_binding",
    "resolve_trading_single_stock_sidebar_state",
]
