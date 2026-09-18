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
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.entry_plans import build_position_from_entry_fill
from core.exact_accounting import calc_entry_total_cost, milli_to_money, milli_to_price, price_to_milli
from core.params_io import build_params_from_mapping
from core.position_step import rollforward_position_management_from_completed_bar
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trading_account_state import (
    ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT,
    MANAGED_POSITION_SOURCES,
    POSITION_SOURCE_MANUAL_MANAGED,
    POSITION_SOURCE_STRATEGY_FILL,
    TRADE_MUTATION_MANUAL_MANAGED_BUY,
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
from services.trading.pending_entry_state import (
    PENDING_ENTRY_STATUS_ACTIVE,
    PENDING_ENTRY_STATUS_FILLED,
    load_trading_pending_entry_state,
)
from services.trading.accounting_policy import overlay_trading_accounting_params
from services.trading.order_state import load_trading_order_state
from services.trading.protection_planning import get_trading_protection_plan_read_model
from services.trading.indicator_exit_planning import get_trading_indicator_exit_plan_read_model
from services.trading.strategy_param_runtime import resolve_trading_position_management_binding
from core.trade_lifecycle import (
    TRADE_LIFECYCLE_POSITION,
    TRADE_LIFECYCLE_SHADOW,
    TRADE_LIFECYCLE_SIGNAL,
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
    pending_state = load_trading_pending_entry_state(root, required=False)
    pending_entries = [
        deepcopy(dict(row))
        for row in ((pending_state or {}).get("entries") or {}).values()
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
            for event in effective_trading_account_events(account)
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

    return {
        "ticker": ticker_key,
        "candidate": candidate,
        "account_revision": None if not account else int(account.get("revision") or 0),
        "order_revision": None if not orders else int(orders.get("revision") or 0),
        "account_events": account_events,
        "entry_orders": entry_orders,
        "pending_entries": pending_entries,
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
                if str(details.get("event") or "").upper() == "TP_HALF":
                    # Historical account events persist the pre-sell position,
                    # not a second post-sell snapshot.  The confirmed TP event is
                    # sufficient evidence that subsequent dates must no longer
                    # show the half-take-profit line.
                    position_state["sold_half"] = True
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
    if isinstance(position_state, Mapping):
        average_entry = position_state.get("entry_fill_price", average_entry)

    lineage_planned_qty = lineage.get("planned_qty") if isinstance(lineage, Mapping) else None
    lineage_planned_cost = lineage.get("planned_cost") if isinstance(lineage, Mapping) else None
    return {
        "state": TRADE_LIFECYCLE_POSITION,
        "display_state": "持股",
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
        "sell_signal": _sell_signal_as_of(inspection, date_text),
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



def _shadow_plan_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize the current Scanner row into the common pre-fill plan contract."""
    if not isinstance(candidate, Mapping) or not candidate:
        return None
    seed = dict(candidate.get("execution_plan_seed") or {})
    information_date = _date_text(candidate.get("trade_date")) or _date_text(seed.get("trade_date"))
    signal_date = _date_text(candidate.get("signal_date")) or information_date
    if information_date is None or signal_date is None:
        return None
    return {
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
    }


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
    return {
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
        "order_id": str(order.get("order_id") or ""),
        "priority": 20,
    }


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
    return {
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
    }


def _shadow_plan_from_pending_entry(entry: Mapping[str, Any], *, last_date: str | None) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    status = str(entry.get("status") or "")
    if status not in {PENDING_ENTRY_STATUS_ACTIVE, PENDING_ENTRY_STATUS_FILLED}:
        # Cancelled/deleted intent remains in the pending-entry audit log, but it
        # is no longer effective execution evidence and must not leak into the
        # current Trading chart/sidebar lifecycle.
        return None
    information_date = _date_text(entry.get("information_date"))
    if information_date is None:
        return None
    origin = str(entry.get("origin") or "")
    manual_pending = origin == "manual_selected"
    planned_trade_date = _date_text(entry.get("planned_trade_date"))
    # A manual pending row is explicit user-confirmed order intent.  Its chart
    # lifecycle therefore begins on the persisted planned order date, not on the
    # later Workbench bookkeeping/information date.  The frozen plan itself is
    # the audit evidence being replayed; no Research simulated fill is inferred.
    signal_date = information_date if manual_pending else (_date_text(entry.get("signal_date")) or information_date)
    seed = dict(entry.get("execution_plan_seed") or {})
    end_before = None
    end_date = last_date
    if status == PENDING_ENTRY_STATUS_FILLED:
        end_before = _date_text((entry.get("fill") or {}).get("trade_date"))
    return {
        "signal_date": signal_date,
        "information_date": information_date,
        "end_date": end_date,
        "end_before_date": end_before,
        "limit_price": seed.get("limit_price", entry.get("limit_price")),
        "stop_price": seed.get("init_sl", entry.get("init_sl")),
        "init_trail": seed.get("init_trail", entry.get("init_trail")),
        "tp_price": seed.get("target_price", entry.get("target_price")),
        "entry_atr": seed.get("entry_atr", entry.get("entry_atr")),
        "entry_price": seed.get("shadow_entry_price", seed.get("entry_ref_price")),
        "shadow_position_state": deepcopy(seed.get("shadow_position_state")),
        "planned_qty": int(entry.get("planned_qty") or 0) or None,
        "reserved_capital": entry.get("reserved_cost"),
        "sizing_capital": seed.get("sizing_capital"),
        "ticker": str(entry.get("ticker") or ""),
        "security_profile": deepcopy(seed.get("security_profile")),
        "frozen_params": deepcopy((entry.get("management_lineage") or {}).get("frozen_params")),
        "entry_type": str(seed.get("entry_type") or "manual"),
        "source": "manual_pending_entry" if manual_pending else "scanner_pending_entry",
        "planned_trade_date": planned_trade_date,
        "confirmed_pending_order": bool(manual_pending),
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

    if not isinstance(entry, Mapping) or str(entry.get("status") or "") != PENDING_ENTRY_STATUS_FILLED:
        return None
    pending_lineage = entry.get("management_lineage")
    pending_lineage_id = (
        str((pending_lineage or {}).get("lineage_id") or "").strip()
        if isinstance(pending_lineage, Mapping)
        else ""
    )
    if not pending_lineage_id:
        return None
    fill = dict(entry.get("fill") or {})
    fill_date = _date_text(fill.get("trade_date"))
    fill_qty = int(fill.get("qty") or 0)
    fill_price_milli = None
    if fill.get("price") is not None:
        try:
            fill_price_milli = int(price_to_milli(fill.get("price")))
        except (TypeError, ValueError):
            return None

    for event in list(inspection.get("account_events") or []):
        if not isinstance(event, Mapping):
            continue
        mutation = str(event.get("mutation_type") or "")
        if mutation == TRADE_MUTATION_STRATEGY_BUY:
            lineage_field = "strategy_lineage"
        elif mutation == TRADE_MUTATION_MANUAL_MANAGED_BUY:
            lineage_field = "management_lineage"
        else:
            continue
        details = dict(event.get("details") or {})
        lineage = details.get(lineage_field)
        if not isinstance(lineage, Mapping):
            position_after = details.get("position_after")
            lineage = (
                position_after.get(lineage_field)
                if isinstance(position_after, Mapping)
                else None
            )
        if not isinstance(lineage, Mapping):
            continue
        if str(lineage.get("lineage_id") or "").strip() != pending_lineage_id:
            continue
        if fill_date is not None and _trade_event_date(event) != fill_date:
            continue
        event_qty = int(details.get("qty") or details.get("fill_qty") or 0)
        if fill_qty > 0 and event_qty != fill_qty:
            continue
        event_price_milli = details.get("entry_fill_price_milli")
        if event_price_milli is None:
            event_price_milli = details.get("fill_price_milli")
        if fill_price_milli is not None and event_price_milli is not None and int(event_price_milli) != fill_price_milli:
            continue
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

    active: list[dict[str, Any]] = []
    filled_by_event: dict[tuple[object, ...], tuple[float, dict[str, Any]]] = {}
    for raw in list(inspection.get("pending_entries") or []):
        if not isinstance(raw, Mapping):
            continue
        entry = deepcopy(dict(raw))
        status = str(entry.get("status") or "")
        if status == PENDING_ENTRY_STATUS_ACTIVE:
            active.append(entry)
            continue
        if status != PENDING_ENTRY_STATUS_FILLED:
            continue
        event = _matching_account_buy_event_for_pending(entry, inspection)
        if not isinstance(event, Mapping):
            continue
        key = (
            event.get("revision"),
            event.get("mutation_id"),
            event.get("event_hash"),
            event.get("timestamp"),
            _trade_event_date(event),
        )
        event_ts = str(event.get("timestamp") or "")
        fill_ts = str((entry.get("fill") or {}).get("confirmed_at") or entry.get("closed_at") or "")
        distance = float("inf")
        try:
            event_dt = datetime.fromisoformat(event_ts)
            fill_dt = datetime.fromisoformat(fill_ts)
            distance = abs((event_dt - fill_dt).total_seconds())
        except (TypeError, ValueError):
            pass
        prior = filled_by_event.get(key)
        if prior is None or (distance, str(entry.get("pending_entry_id") or "")) < (
            prior[0], str(prior[1].get("pending_entry_id") or "")
        ):
            filled_by_event[key] = (distance, entry)
    rows = active + [pair[1] for pair in filled_by_event.values()]
    rows.sort(key=lambda row: (
        str(row.get("planned_trade_date") or row.get("information_date") or ""),
        str(row.get("pending_entry_id") or ""),
    ))
    return rows


def _plan_covers_date(plan: Mapping[str, Any], date_text: str) -> bool:
    signal_date = _date_text(plan.get("signal_date")) or _date_text(plan.get("information_date"))
    if signal_date is None or date_text < signal_date:
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
    return {
        "state": lifecycle_state,
        "display_state": (
            "掛單" if str(plan.get("source")) == "manual_pending_entry"
            else ("買訊" if lifecycle_state == TRADE_LIFECYCLE_SIGNAL else "SHADOW")
        ),
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
    for plan in _build_shadow_plans(inspection, last_date=last_date):
        if _plan_covers_date(plan, date_text):
            candidates.append(plan)
    if not candidates:
        return None
    # Prefer the newest plan, then the strongest persisted evidence for the same
    # signal. This naturally chooses lineage > order > current scanner snapshot.
    plan = max(
        candidates,
        key=lambda row: (
            str(_date_text(row.get("signal_date")) or ""),
            int(row.get("priority") or 0),
        ),
    )
    signal_date = _date_text(plan.get("signal_date")) or _date_text(plan.get("information_date"))
    lifecycle_state = (
        TRADE_LIFECYCLE_SHADOW
        if signal_date == date_text and bool(plan.get("confirmed_pending_order"))
        else (TRADE_LIFECYCLE_SIGNAL if signal_date == date_text else TRADE_LIFECYCLE_SHADOW)
    )
    return _shadow_state_from_plan(
        plan,
        strategy_state=strategy_state,
        lifecycle_state=lifecycle_state,
    )

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
        return "指標賣出"
    return "指標賣出"


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
    inputs = dict(chart_payload.get("strategy_lifecycle_inputs") or {})
    atr_values = inputs.get("atr")
    sell_signals = inputs.get("sell_signal")
    if atr_values is None:
        atr_values = [None] * total
    if sell_signals is None:
        sell_signals = [False] * total

    last_date = next((value for value in reversed(date_labels) if value is not None), None)
    merged: dict[int, dict[str, Any]] = {}
    owner_key: dict[int, tuple[str, int]] = {}

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
        signal_key = str(_date_text(row.get("signal_date")) or _date_text(row.get("information_date")) or "")
        merged[idx] = row
        owner_key[idx] = (signal_key, 0)

    for plan in _build_shadow_plans(inspection, last_date=last_date):
        plan_params = _resolve_lifecycle_plan_params(plan, params)
        if plan_params is None:
            continue
        if not any(value is not None for value in atr_values):
            fallback_atr = plan.get("entry_atr")
            atr_for_plan = [fallback_atr] * total
        else:
            atr_for_plan = atr_values
        timeline = build_prefill_lifecycle_timeline(
            date_labels=date_labels,
            open_values=list(chart_payload.get("open") if chart_payload.get("open") is not None else []),
            high_values=list(chart_payload.get("high") if chart_payload.get("high") is not None else []),
            low_values=list(chart_payload.get("low") if chart_payload.get("low") is not None else []),
            close_values=list(chart_payload.get("close") if chart_payload.get("close") is not None else []),
            volume_values=list(chart_payload.get("volume") if chart_payload.get("volume") is not None else []),
            atr_values=atr_for_plan,
            sell_signals=sell_signals,
            plan=plan,
            params=plan_params,
        )
        if bool(plan.get("confirmed_pending_order")):
            signal_date = _date_text(plan.get("signal_date")) or _date_text(plan.get("information_date"))
            planned_trade_date = _date_text(plan.get("planned_trade_date")) or signal_date
            for static_idx, static_date in enumerate(date_labels):
                if static_date is None or planned_trade_date is None or signal_date is None:
                    continue
                if static_date < planned_trade_date or static_date > signal_date:
                    continue
                # A confirmed manual pending order already owns immutable
                # planned L/S/TP geometry.  For a historical backfill, show that
                # persisted plan from its planned order date through the later
                # information date, but do not run today's frozen Params backward
                # over those historical bars.  The canonical shadow engine above
                # starts advancing only after the information bar.
                static_row = build_trade_lifecycle_row(
                    TRADE_LIFECYCLE_SHADOW,
                    source=str(plan.get("source") or "manual_pending_entry"),
                    signal_date=signal_date,
                    information_date=plan.get("information_date"),
                    entry_type=plan.get("entry_type"),
                    limit_price=plan.get("limit_price"),
                    entry_price=None,
                    stop_price=plan.get("stop_price"),
                    tp_price=plan.get("tp_price"),
                    reserved_capital=plan.get("reserved_capital"),
                    planned_qty=plan.get("planned_qty"),
                    remaining_order_qty=plan.get("planned_qty"),
                    confirmed_pending_order=True,
                )
                static_row["display_state"] = "掛單" if static_date == planned_trade_date else "SHADOW"
                timeline[int(static_idx)] = static_row
        signal_key = str(_date_text(plan.get("signal_date")) or _date_text(plan.get("information_date")) or "")
        priority = int(plan.get("priority") or 0)
        for idx, row in timeline.items():
            key = (signal_key, priority)
            if idx not in owner_key or key >= owner_key[idx]:
                merged[int(idx)] = deepcopy(dict(row))
                owner_key[int(idx)] = key
    return merged


def _build_manual_managed_position_replay_timeline(
    inspection: Mapping[str, Any],
    chart_payload: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Rebuild date-aware manual-managed geometry from frozen Params.

    Account truth remains the owner of inventory/cash/actual fills.  This helper
    only replays deterministic management geometry (stop/target/indicator SELL)
    so a manual holding has the same historical chart semantics as a strategy
    holding instead of exposing a development-era takeover-date gap.
    """

    record = inspection.get("current_position")
    if not isinstance(record, Mapping) or str(record.get("source") or "") != POSITION_SOURCE_MANUAL_MANAGED:
        return {}
    broker = dict(record.get("broker") or {})
    management = dict(record.get("strategy_management") or {})
    current_state = management.get("position_state")
    binding = inspection.get("position_binding")
    if not isinstance(binding, Mapping) or not isinstance(current_state, Mapping):
        return {}
    entry_date = _date_text(broker.get("entry_date"))
    if entry_date is None:
        return {}

    try:
        params = build_params_from_mapping(dict(binding.get("frozen_params") or {}))
    except (TypeError, ValueError, KeyError):
        return {}
    seed = dict(binding.get("execution_plan_seed") or {})
    ticker = str(inspection.get("ticker") or "")
    initial_qty = int(broker.get("initial_qty") or broker.get("qty") or 0)
    if initial_qty <= 0:
        return {}

    entry_price = current_state.get("entry_fill_price")
    if entry_price is None:
        gross_milli = broker.get("initial_gross_buy_milli")
        if gross_milli is None:
            gross_milli = broker.get("initial_cost_basis_milli")
        gross_milli = int(gross_milli or 0)
        if gross_milli <= 0:
            return {}
        entry_price = milli_to_price((gross_milli + initial_qty // 2) // initial_qty)

    try:
        position = build_position_from_entry_fill(
            buy_price=float(entry_price),
            qty=initial_qty,
            params=params,
            entry_type=str(seed.get("entry_type") or "manual"),
            init_sl=seed.get("init_sl"),
            init_trail=seed.get("init_trail"),
            target_price=seed.get("target_price"),
            limit_price=seed.get("limit_price"),
            entry_atr=seed.get("entry_atr"),
            ticker=ticker,
            security_profile=seed.get("security_profile"),
            trade_date=entry_date,
            target_reference_price=seed.get("target_reference_price"),
        )
    except (TypeError, ValueError, KeyError):
        return {}

    # Keep broker economics exact while replaying only strategy-management fields.
    position["qty"] = initial_qty
    position["initial_qty"] = initial_qty
    position["net_buy_total_milli"] = int(broker.get("initial_cost_basis_milli") or 0)
    position["remaining_cost_basis_milli"] = int(broker.get("initial_cost_basis_milli") or 0)

    date_labels = _chart_date_labels(chart_payload)
    total = len(date_labels)
    required = ("open", "high", "low", "close", "volume")
    chart_values = {key: _chart_series_values(chart_payload, key) for key in required}
    if total <= 0 or any(len(chart_values[key]) < total for key in required):
        return {}
    frame = pd.DataFrame(
        {
            "Open": chart_values["open"][:total],
            "High": chart_values["high"][:total],
            "Low": chart_values["low"][:total],
            "Close": chart_values["close"][:total],
            "Volume": chart_values["volume"][:total],
        },
        index=pd.to_datetime([value if value is not None else "NaT" for value in date_labels]),
    )
    valid_mask = ~frame.index.isna()
    if not bool(valid_mask.any()):
        return {}
    valid_frame = frame.loc[valid_mask].copy()
    valid_positions = [idx for idx, valid in enumerate(valid_mask.tolist()) if valid]
    try:
        atr_values, _buy_values, sell_values, _limits = unpack_precomputed_signals(
            generate_signals(valid_frame, params, ticker=ticker)
        )
    except (TypeError, ValueError, KeyError, IndexError):
        return {}

    replay: dict[int, dict[str, Any]] = {}
    for local_idx, chart_idx in enumerate(valid_positions):
        date_text = date_labels[chart_idx]
        if date_text is None or date_text < entry_date:
            continue
        rollforward_position_management_from_completed_bar(
            position,
            completed_high=float(valid_frame["High"].iloc[local_idx]),
            completed_atr=float(atr_values[local_idx]),
            params=params,
            sync_display_fields=True,
        )
        indicator_sell = bool(sell_values[local_idx]) and date_text > entry_date
        replay[int(chart_idx)] = build_trade_lifecycle_row(
            TRADE_LIFECYCLE_POSITION,
            source="manual_managed_frozen_replay",
            information_date=date_text,
            entry_type=str(position.get("entry_type") or seed.get("entry_type") or "manual"),
            limit_price=position.get("limit_price"),
            entry_price=position.get("entry_fill_price"),
            stop_price=position.get("sl"),
            tp_price=(None if bool(position.get("sold_half", False)) else position.get("tp_half")),
            buy_qty=initial_qty,
            position_qty=initial_qty,
            remaining_order_qty=0,
            entry_date=entry_date,
            initial_stop_price=position.get("initial_stop"),
            trailing_stop_price=position.get("trailing_stop"),
            sell_signal=("INDICATOR SELL" if indicator_sell else None),
            management_lineage=deepcopy(record.get("management_lineage")),
        )
    return replay


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
        if position_state is not None:
            replay_state = manual_replay.get(int(idx))
            if isinstance(replay_state, Mapping):
                # Broker/event history remains authoritative for lifecycle and
                # quantity.  The frozen-Params replay owns date-aware management
                # geometry for manual-managed positions.
                position_state["entry_price"] = replay_state.get("entry_price")
                position_state["initial_stop_price"] = replay_state.get("initial_stop_price")
                position_state["trailing_stop_price"] = replay_state.get("trailing_stop_price")
                position_state["stop_price"] = replay_state.get("stop_price")
                if not bool(position_state.get("sold_half", False)):
                    position_state["tp_price"] = replay_state.get("tp_price")
                position_state["sell_signal"] = replay_state.get("sell_signal")
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
            position_state["state"] = TRADE_LIFECYCLE_POSITION
            timeline[int(idx)] = deepcopy(dict(position_state))
            continue
        replay_state = manual_replay.get(int(idx))
        if isinstance(replay_state, Mapping):
            timeline[int(idx)] = deepcopy(dict(replay_state))
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
                "result": str(details.get("event") or "實際成交"),
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
        if isinstance(last_state, Mapping) and str(last_state.get("state") or "") in {
            TRADE_LIFECYCLE_SIGNAL, TRADE_LIFECYCLE_SHADOW,
        }:
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
        str(trace_name): [deepcopy(dict(marker)) for marker in list(markers or [])]
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
    for raw in list(payload.get("signal_annotations") or []):
        item = deepcopy(dict(raw))
        signal_type = str(item.get("signal_type") or "").lower()
        if signal_type == "sell":
            continue
        if signal_type == "buy":
            meta = dict(item.get("meta") or {})
            for key in ("qty", "reserved_capital", "buy_capital"):
                meta.pop(key, None)
            item["meta"] = meta
            item["detail_text"] = ""
        signal_annotations.append(item)

    indicator_exit = inspection.get("indicator_exit") or {}
    if bool(indicator_exit.get("fresh")):
        for exit_row in list(indicator_exit.get("exits") or []):
            signal_date = _date_text(exit_row.get("signal_information_date"))
            if signal_date is None or signal_date not in date_to_x:
                continue
            x = int(date_to_x[signal_date])
            anchor = None
            for series_key in ("high", "close"):
                series = payload.get(series_key)
                try:
                    anchor = float(series[x])
                except (TypeError, ValueError, IndexError):
                    anchor = None
                if anchor is not None:
                    break
            signal_annotations.append({
                "date": signal_date,
                "x": x,
                "anchor_price": anchor,
                "signal_type": "sell",
                "title": "賣出訊號",
                "detail_text": "",
                "note": "Trading frozen Params 指標賣出訊號",
                "meta": {
                    "canonical_trading": True,
                    "canonical_indicator_exit": True,
                    "signal_key": exit_row.get("signal_key"),
                    "qty": exit_row.get("qty"),
                    "entry_order_id": exit_row.get("entry_order_id"),
                    "frozen_params_sha256": exit_row.get("frozen_params_sha256"),
                },
            })

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
