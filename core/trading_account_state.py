"""Canonical Trading account-state schema and pure mutation semantics."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import math
from typing import Any

from core.entry_plans import build_position_from_entry_fill
from core.exact_accounting import (
    allocate_cost_basis_milli,
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    calc_average_price_from_total_milli,
    calc_initial_risk_total_milli,
    milli_to_money,
    milli_to_price,
    money_to_milli,
    rate_to_ppm,
    sync_position_display_fields,
)
from core.file_integrity import canonical_json_sha256
from core.position_step import execute_confirmed_position_sell_fill
from core.runtime_domains import RUNTIME_DOMAIN_TRADING

TRADING_ACCOUNT_SCHEMA_VERSION = 1
TRADING_RUNTIME_DOMAIN = RUNTIME_DOMAIN_TRADING
POSITION_SOURCE_MANUAL_ADOPTED = "manual_adopted"
POSITION_SOURCE_STRATEGY_FILL = "strategy_fill"
POSITION_SOURCES = (POSITION_SOURCE_MANUAL_ADOPTED, POSITION_SOURCE_STRATEGY_FILL)
MANAGEMENT_STATUS_UNMANAGED = "unmanaged"
MANAGEMENT_STATUS_ACTIVE = "active"
MANAGEMENT_STATUSES = (MANAGEMENT_STATUS_UNMANAGED, MANAGEMENT_STATUS_ACTIVE)


def _normalize_ticker(value: object) -> str:
    ticker = str(value or "").strip().upper()
    if not ticker:
        raise ValueError("ticker 必填")
    if any(ch.isspace() for ch in ticker):
        raise ValueError(f"ticker 不可包含空白: {ticker!r}")
    return ticker


def _normalize_iso_date(value: object | None, *, field_name: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} 必須是 YYYY-MM-DD: {value!r}") from exc


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def _event_hash_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "event_hash"}


def _build_event(
    *,
    revision: int,
    mutation_id: str,
    mutation_type: str,
    timestamp: str,
    details: dict[str, Any],
    prev_event_hash: str | None,
) -> dict[str, Any]:
    event = {
        "revision": int(revision),
        "mutation_id": str(mutation_id),
        "mutation_type": str(mutation_type),
        "timestamp": str(timestamp),
        "prev_event_hash": prev_event_hash,
        "details": _json_safe(details),
    }
    event["event_hash"] = canonical_json_sha256(_event_hash_payload(event))
    return event


def build_empty_trading_account_state(
    *,
    timestamp: str,
    mutation_id: str,
    cash: object | None = None,
) -> dict[str, Any]:
    cash_milli = None if cash is None else int(money_to_milli(cash))
    if cash_milli is not None and cash_milli < 0:
        raise ValueError("Trading cash 不可為負數")
    event = _build_event(
        revision=0,
        mutation_id=mutation_id,
        mutation_type="initialize",
        timestamp=timestamp,
        details={"cash_milli": cash_milli},
        prev_event_hash=None,
    )
    state = {
        "schema_version": TRADING_ACCOUNT_SCHEMA_VERSION,
        "runtime_domain": TRADING_RUNTIME_DOMAIN,
        "revision": 0,
        "created_at": str(timestamp),
        "updated_at": str(timestamp),
        "cash_milli": cash_milli,
        "positions": {},
        "events": [event],
    }
    validate_trading_account_state(state)
    return state


def _append_mutation(
    state: dict[str, Any],
    *,
    mutation_id: str,
    mutation_type: str,
    timestamp: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(state)
    previous_revision = int(updated["revision"])
    events = list(updated.get("events", []))
    previous_hash = events[-1]["event_hash"] if events else None
    revision = previous_revision + 1
    events.append(
        _build_event(
            revision=revision,
            mutation_id=mutation_id,
            mutation_type=mutation_type,
            timestamp=timestamp,
            details=details,
            prev_event_hash=previous_hash,
        )
    )
    updated["revision"] = revision
    updated["updated_at"] = str(timestamp)
    updated["events"] = events
    validate_trading_account_state(updated)
    return updated


def set_trading_account_cash(
    state: dict[str, Any],
    *,
    cash: object,
    timestamp: str,
    mutation_id: str,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_account_state(state)
    cash_milli = int(money_to_milli(cash))
    if cash_milli < 0:
        raise ValueError("Trading cash 不可為負數")
    previous_cash_milli = state.get("cash_milli")
    updated = deepcopy(state)
    updated["cash_milli"] = cash_milli
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="set_cash_balance",
        timestamp=timestamp,
        details={
            "previous_cash_milli": previous_cash_milli,
            "cash_milli": cash_milli,
            "note": None if note is None else str(note),
        },
    )


def adopt_manual_trading_position(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    cost_basis_total: object,
    timestamp: str,
    mutation_id: str,
    entry_date: object | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    qty = int(qty)
    if qty <= 0:
        raise ValueError("manual adopted position qty 必須 > 0")
    cost_basis_milli = int(money_to_milli(cost_basis_total))
    if cost_basis_milli <= 0:
        raise ValueError("manual adopted position cost_basis_total 必須 > 0")
    if ticker_key in state["positions"]:
        raise ValueError(f"Trading 已存在 open position: {ticker_key}")

    adopted_entry_date = _normalize_iso_date(entry_date, field_name="entry_date")
    updated = deepcopy(state)
    updated["positions"][ticker_key] = {
        "ticker": ticker_key,
        "source": POSITION_SOURCE_MANUAL_ADOPTED,
        "broker": {
            "qty": qty,
            "initial_qty": qty,
            "initial_cost_basis_milli": cost_basis_milli,
            "remaining_cost_basis_milli": cost_basis_milli,
            "realized_pnl_milli": 0,
            "entry_date": adopted_entry_date,
        },
        "strategy_management": {
            "status": MANAGEMENT_STATUS_UNMANAGED,
            "management_start_date": None,
            "position_state": None,
        },
    }
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="adopt_manual_position",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "qty": qty,
            "cost_basis_total_milli": cost_basis_milli,
            "entry_date": adopted_entry_date,
            "note": None if note is None else str(note),
            "cash_changed": False,
        },
    )


def _has_confirmed_sell_history(state: dict[str, Any], ticker: str) -> bool:
    ticker_key = _normalize_ticker(ticker)
    for event in state.get("events", []):
        if str(event.get("mutation_type") or "") != "confirm_sell_fill":
            continue
        details = event.get("details") or {}
        if _normalize_ticker(details.get("ticker")) == ticker_key:
            return True
    return False


def correct_manual_trading_position(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    cost_basis_total: object,
    timestamp: str,
    mutation_id: str,
    entry_date: object | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Correct broker truth for an unmanaged manually adopted holding.

    This is a state-reconciliation operation, not a trade.  It never changes
    cash and is intentionally restricted to holdings with no confirmed sell
    history so that historical accounting is not rewritten.
    """
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    record = state["positions"][ticker_key]
    if record.get("source") != POSITION_SOURCE_MANUAL_ADOPTED:
        raise ValueError(f"只有 manual_adopted position 可直接修正 broker truth: {ticker_key}")
    management = record.get("strategy_management") or {}
    if management.get("status") != MANAGEMENT_STATUS_UNMANAGED or management.get("position_state") is not None:
        raise ValueError(f"已由策略管理的 position 不可用 manual correction 修改: {ticker_key}")
    if _has_confirmed_sell_history(state, ticker_key):
        raise ValueError(f"已有 confirmed sell history 的 manual position 不可改寫 broker truth: {ticker_key}")

    qty = int(qty)
    if qty <= 0:
        raise ValueError("manual position qty 必須 > 0")
    cost_basis_milli = int(money_to_milli(cost_basis_total))
    if cost_basis_milli <= 0:
        raise ValueError("manual position cost_basis_total 必須 > 0")
    corrected_entry_date = _normalize_iso_date(entry_date, field_name="entry_date")

    updated = deepcopy(state)
    updated_record = updated["positions"][ticker_key]
    previous_broker = deepcopy(updated_record["broker"])
    updated_record["broker"] = {
        "qty": qty,
        "initial_qty": qty,
        "initial_cost_basis_milli": cost_basis_milli,
        "remaining_cost_basis_milli": cost_basis_milli,
        "realized_pnl_milli": 0,
        "entry_date": corrected_entry_date,
    }
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="correct_manual_position",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "previous_broker": previous_broker,
            "broker": deepcopy(updated_record["broker"]),
            "note": None if note is None else str(note),
            "cash_changed": False,
        },
    )


def remove_manual_trading_position(
    state: dict[str, Any],
    *,
    ticker: object,
    timestamp: str,
    mutation_id: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Remove an erroneous unmanaged manual holding without creating a trade."""
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    record = state["positions"][ticker_key]
    if record.get("source") != POSITION_SOURCE_MANUAL_ADOPTED:
        raise ValueError(f"只有 manual_adopted position 可直接移除: {ticker_key}")
    management = record.get("strategy_management") or {}
    if management.get("status") != MANAGEMENT_STATUS_UNMANAGED or management.get("position_state") is not None:
        raise ValueError(f"已由策略管理的 position 不可用 manual remove 移除: {ticker_key}")
    if _has_confirmed_sell_history(state, ticker_key):
        raise ValueError(f"已有 confirmed sell history 的 manual position 不可直接移除: {ticker_key}")

    updated = deepcopy(state)
    removed = deepcopy(updated["positions"].pop(ticker_key))
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="remove_manual_position",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "removed_position": removed,
            "note": None if note is None else str(note),
            "cash_changed": False,
        },
    )


def apply_confirmed_strategy_buy_fill(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    buy_price: object,
    params,
    timestamp: str,
    mutation_id: str,
    trade_date: object,
    init_sl=None,
    init_trail=None,
    target_price=None,
    limit_price=None,
    entry_atr=None,
    security_profile=None,
    entry_type: str = "normal",
    entry_order_id: str | None = None,
) -> dict[str, Any]:
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key in state["positions"]:
        raise ValueError(f"Trading 已存在 open position，不支援同ticker加碼: {ticker_key}")
    cash_milli = state.get("cash_milli")
    if cash_milli is None:
        raise ValueError("Trading cash 尚未設定，不能確認買入成交")
    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")

    position_state = build_position_from_entry_fill(
        buy_price=buy_price,
        qty=int(qty),
        init_sl=init_sl,
        init_trail=init_trail,
        params=params,
        entry_type=entry_type,
        target_price=target_price,
        limit_price=limit_price,
        entry_atr=entry_atr,
        ticker=ticker_key,
        security_profile=security_profile,
        trade_date=trade_date_text,
    )
    position_state = _json_safe(position_state)
    net_buy_total_milli = int(position_state["net_buy_total_milli"])
    if net_buy_total_milli > int(cash_milli):
        raise ValueError(
            f"Trading cash 不足：需要 {milli_to_money(net_buy_total_milli):.2f}，"
            f"可用 {milli_to_money(int(cash_milli)):.2f}"
        )

    updated = deepcopy(state)
    updated["cash_milli"] = int(cash_milli) - net_buy_total_milli
    updated["positions"][ticker_key] = {
        "ticker": ticker_key,
        "source": POSITION_SOURCE_STRATEGY_FILL,
        "broker": {
            "qty": int(position_state["qty"]),
            "initial_qty": int(position_state["initial_qty"]),
            "initial_cost_basis_milli": net_buy_total_milli,
            "remaining_cost_basis_milli": int(position_state["remaining_cost_basis_milli"]),
            "realized_pnl_milli": int(position_state.get("realized_pnl_milli", 0) or 0),
            "entry_date": trade_date_text,
            "entry_order_id": None if entry_order_id is None else str(entry_order_id),
        },
        "strategy_management": {
            "status": MANAGEMENT_STATUS_ACTIVE,
            "management_start_date": trade_date_text,
            "position_state": position_state,
        },
    }
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_strategy_buy_fill",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "qty": int(position_state["qty"]),
            "trade_date": trade_date_text,
            "entry_fill_price_milli": int(position_state["entry_fill_price_milli"]),
            "net_buy_total_milli": net_buy_total_milli,
            "cash_milli_after": int(updated["cash_milli"]),
            "entry_order_id": None if entry_order_id is None else str(entry_order_id),
        },
    )


def apply_confirmed_strategy_buy_fill_increment(
    state: dict[str, Any],
    *,
    entry_order_id: str,
    ticker: object,
    qty: int,
    buy_price: object,
    params,
    timestamp: str,
    mutation_id: str,
    trade_date: object,
    init_sl=None,
    init_trail=None,
    target_price=None,
    limit_price=None,
    entry_atr=None,
    security_profile=None,
    entry_type: str = "normal",
) -> dict[str, Any]:
    """Apply an additional confirmed fill for the same still-active BUY order.

    Partial fills are conservatively limited to one trade date. The broker/account
    cost basis accumulates the exact per-fill ledgers, while the strategy position
    is rebuilt from the gross weighted-average execution price and the original
    frozen entry ATR so stop/trail/target remain on the canonical entry-plan seam.
    """
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 不存在可累積 partial fill 的 open position: {ticker_key}")
    record = state["positions"][ticker_key]
    if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
        raise ValueError(f"Trading partial fill 只能累積到 strategy_fill position: {ticker_key}")
    broker = record.get("broker") or {}
    if str(broker.get("entry_order_id") or "") != str(entry_order_id or ""):
        raise ValueError(f"Trading partial fill order_id 與既有 position 不一致: {ticker_key}")
    if _has_confirmed_sell_history(state, ticker_key):
        raise ValueError(f"已有 confirmed sell history 的 position 不得再累積買入 partial fill: {ticker_key}")

    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")
    if str(broker.get("entry_date") or "") != trade_date_text:
        raise ValueError("同一 Trading order 的多次 partial fill 必須發生於同一交易日")
    fill_qty = int(qty)
    if fill_qty <= 0:
        raise ValueError("Trading partial fill qty 必須 > 0")
    cash_milli = state.get("cash_milli")
    if cash_milli is None:
        raise ValueError("Trading cash 尚未設定，不能確認買入成交")

    management = record.get("strategy_management") or {}
    position_state = management.get("position_state")
    if management.get("status") != MANAGEMENT_STATUS_ACTIVE or not isinstance(position_state, dict):
        raise ValueError(f"Trading strategy position state 不合法: {ticker_key}")

    new_ledger = build_buy_ledger_from_price(buy_price, fill_qty, params)
    new_cost_milli = int(new_ledger["net_buy_total_milli"])
    if new_cost_milli > int(cash_milli):
        raise ValueError(
            f"Trading cash 不足：需要 {milli_to_money(new_cost_milli):.2f}，"
            f"可用 {milli_to_money(int(cash_milli)):.2f}"
        )

    existing_qty = int(position_state["qty"])
    total_qty = existing_qty + fill_qty
    total_gross_milli = int(position_state.get("gross_buy_milli", 0) or 0) + int(new_ledger["gross_buy_milli"])
    total_fee_milli = int(position_state.get("buy_fee_milli", 0) or 0) + int(new_ledger["buy_fee_milli"])
    total_net_milli = int(position_state.get("net_buy_total_milli", 0) or 0) + new_cost_milli
    average_fill_price_milli = (total_gross_milli + total_qty // 2) // total_qty

    rebuilt = build_position_from_entry_fill(
        buy_price=milli_to_price(average_fill_price_milli),
        qty=total_qty,
        params=params,
        entry_type=entry_type,
        init_sl=init_sl,
        init_trail=init_trail,
        target_price=target_price,
        limit_price=limit_price,
        entry_atr=entry_atr,
        ticker=ticker_key,
        security_profile=security_profile,
        trade_date=trade_date_text,
    )
    rebuilt["gross_buy_milli"] = total_gross_milli
    rebuilt["buy_fee_milli"] = total_fee_milli
    rebuilt["net_buy_total_milli"] = total_net_milli
    rebuilt["remaining_cost_basis_milli"] = total_net_milli
    rebuilt["entry_fill_price_milli"] = int(average_fill_price_milli)
    rebuilt["entry_fill_price"] = milli_to_price(int(average_fill_price_milli))
    rebuilt["pure_buy_price_milli"] = int(average_fill_price_milli)
    rebuilt["pure_buy_price"] = milli_to_price(int(average_fill_price_milli))
    rebuilt["highest_high_since_entry_milli"] = max(
        int(position_state.get("highest_high_since_entry_milli", average_fill_price_milli) or average_fill_price_milli),
        int(average_fill_price_milli),
    )
    rebuilt["highest_high_since_entry"] = milli_to_price(int(rebuilt["highest_high_since_entry_milli"]))
    stop_ledger = build_sell_ledger_from_price(
        rebuilt["initial_stop"],
        total_qty,
        params,
        ticker=ticker_key,
        security_profile=security_profile,
        trade_date=trade_date_text,
    )
    rebuilt["initial_risk_total_milli"] = calc_initial_risk_total_milli(
        total_net_milli,
        int(stop_ledger["net_sell_total_milli"]),
        rate_to_ppm(params.fixed_risk),
    )
    rebuilt = sync_position_display_fields(rebuilt)

    updated = deepcopy(state)
    updated["cash_milli"] = int(cash_milli) - new_cost_milli
    target = updated["positions"][ticker_key]
    target_broker = target["broker"]
    target_broker["qty"] = total_qty
    target_broker["initial_qty"] = total_qty
    target_broker["initial_cost_basis_milli"] = total_net_milli
    target_broker["remaining_cost_basis_milli"] = total_net_milli
    target["strategy_management"]["position_state"] = _json_safe(rebuilt)
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_strategy_buy_fill_increment",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "entry_order_id": str(entry_order_id),
            "fill_qty": fill_qty,
            "fill_price_milli": int(new_ledger["fill_price_milli"]),
            "net_buy_total_milli": new_cost_milli,
            "cumulative_qty": total_qty,
            "cumulative_cost_basis_milli": total_net_milli,
            "cash_milli_after": int(updated["cash_milli"]),
        },
    )


def apply_trading_strategy_management_rollforward(
    state: dict[str, Any],
    *,
    updates: dict[str, dict[str, Any]],
    timestamp: str,
    mutation_id: str,
) -> dict[str, Any]:
    """Persist deterministic completed-bar strategy-management advances.

    ``updates`` is keyed by ticker and must contain a canonical ``position_state``
    plus the last actually processed completed market-data date.  Cash and broker
    accounting are intentionally immutable in this mutation.
    """
    validate_trading_account_state(state)
    if not isinstance(updates, dict) or not updates:
        raise ValueError("Trading rollforward updates 不可為空")

    updated = deepcopy(state)
    details_rows: list[dict[str, Any]] = []
    for ticker_raw in sorted(updates):
        ticker = _normalize_ticker(ticker_raw)
        payload = updates[ticker_raw]
        if not isinstance(payload, dict):
            raise ValueError(f"Trading rollforward payload 不合法: {ticker}")
        processed_bar_count = int(payload.get("processed_bar_count") or 0)
        if processed_bar_count <= 0:
            raise ValueError(f"Trading rollforward processed_bar_count 必須 > 0: {ticker}")
        if ticker not in updated["positions"]:
            raise ValueError(f"Trading rollforward position 不存在: {ticker}")
        record = updated["positions"][ticker]
        if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
            raise ValueError(f"Trading rollforward 只允許 strategy_fill position: {ticker}")
        management = record.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise ValueError(f"Trading rollforward position 尚未由策略 active 管理: {ticker}")
        new_position = deepcopy(payload.get("position_state"))
        if not isinstance(new_position, dict):
            raise ValueError(f"Trading rollforward position_state 不合法: {ticker}")
        broker = record.get("broker") or {}
        if int(new_position.get("qty", -1)) != int(broker.get("qty", -2)):
            raise ValueError(f"Trading rollforward 不得改變 broker qty: {ticker}")
        if int(new_position.get("remaining_cost_basis_milli", -1)) != int(broker.get("remaining_cost_basis_milli", -2)):
            raise ValueError(f"Trading rollforward 不得改變 broker cost basis: {ticker}")

        processed_through = _normalize_iso_date(
            payload.get("processed_through_date"), field_name="processed_through_date"
        )
        if processed_through is None:
            raise ValueError(f"Trading rollforward 缺少 processed_through_date: {ticker}")
        previous_date = _normalize_iso_date(
            management.get("last_rollforward_date"), field_name="last_rollforward_date"
        )
        if previous_date is not None and processed_through <= previous_date:
            raise ValueError(
                f"Trading rollforward date 必須前進: {ticker} {previous_date} -> {processed_through}"
            )

        before_position = management.get("position_state") or {}
        management["position_state"] = _json_safe(new_position)
        management["last_rollforward_date"] = processed_through
        record["strategy_management"] = management
        details_rows.append(
            {
                "ticker": ticker,
                "previous_rollforward_date": previous_date,
                "processed_through_date": processed_through,
                "processed_bar_count": processed_bar_count,
                "previous_stop_milli": int(before_position.get("sl_milli") or 0),
                "stop_milli": int(new_position.get("sl_milli") or 0),
                "previous_highest_high_milli": int(before_position.get("highest_high_since_entry_milli") or 0),
                "highest_high_since_entry_milli": int(new_position.get("highest_high_since_entry_milli") or 0),
                "cash_changed": False,
                "broker_accounting_changed": False,
            }
        )

    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="rollforward_strategy_management",
        timestamp=timestamp,
        details={"positions": details_rows},
    )


def apply_confirmed_sell_fill(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    exec_price: object,
    params,
    timestamp: str,
    mutation_id: str,
    trade_date: object,
    event: str = "MANUAL_CONFIRMED_SELL",
    mark_tp_half_complete: bool = False,
) -> dict[str, Any]:
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    cash_milli = state.get("cash_milli")
    if cash_milli is None:
        raise ValueError("Trading cash 尚未設定，不能確認賣出成交")
    qty = int(qty)
    if qty <= 0:
        raise ValueError("sell qty 必須 > 0")
    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")

    updated = deepcopy(state)
    record = updated["positions"][ticker_key]
    broker = record["broker"]
    held_qty = int(broker["qty"])
    if qty > held_qty:
        raise ValueError(f"sell qty 超過持股：sell={qty}, held={held_qty}")
    entry_date = broker.get("entry_date")
    if entry_date is not None and str(entry_date) == trade_date_text:
        raise ValueError("禁止同日買入又賣出同一股票")

    security_profile = None
    strategy_management = record.get("strategy_management") or {}
    strategy_position = strategy_management.get("position_state")
    if isinstance(strategy_position, dict):
        security_profile = strategy_position.get("security_profile")

    sell_ledger = build_sell_ledger_from_price(
        exec_price,
        qty,
        params,
        ticker=ticker_key,
        security_profile=security_profile,
        trade_date=trade_date_text,
    )
    allocated_cost_milli = allocate_cost_basis_milli(
        int(broker["remaining_cost_basis_milli"]), held_qty, qty
    )
    net_sell_total_milli = int(sell_ledger["net_sell_total_milli"])
    pnl_milli = net_sell_total_milli - int(allocated_cost_milli)

    broker["qty"] = held_qty - qty
    broker["remaining_cost_basis_milli"] = int(broker["remaining_cost_basis_milli"]) - int(allocated_cost_milli)
    broker["realized_pnl_milli"] = int(broker.get("realized_pnl_milli", 0) or 0) + pnl_milli
    updated["cash_milli"] = int(cash_milli) + net_sell_total_milli

    if isinstance(strategy_position, dict):
        strategy_position = deepcopy(strategy_position)
        freed_cash_milli, strategy_pnl_milli = execute_confirmed_position_sell_fill(
            strategy_position,
            exec_price=exec_price,
            sell_qty=qty,
            params=params,
            trade_date=trade_date_text,
            event=event,
        )
        if int(freed_cash_milli) != net_sell_total_milli or int(strategy_pnl_milli) != pnl_milli:
            raise RuntimeError("Trading broker/strategy sell accounting diverged from canonical position accounting")
        if mark_tp_half_complete and int(strategy_position.get("qty", 0) or 0) > 0:
            strategy_position["sold_half"] = True
        strategy_management["position_state"] = _json_safe(strategy_position)

    remaining_qty = int(broker["qty"])
    if remaining_qty <= 0:
        broker["qty"] = 0
        broker["remaining_cost_basis_milli"] = 0
        del updated["positions"][ticker_key]

    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_sell_fill",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "qty": qty,
            "trade_date": trade_date_text,
            "exec_price_milli": int(sell_ledger["exec_price_milli"]),
            "net_sell_total_milli": net_sell_total_milli,
            "allocated_cost_milli": int(allocated_cost_milli),
            "realized_pnl_milli": pnl_milli,
            "remaining_qty": max(remaining_qty, 0),
            "cash_milli_after": int(updated["cash_milli"]),
        },
    )


def validate_trading_account_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError("Trading account state 必須是 dict")
    if int(state.get("schema_version", -1)) != TRADING_ACCOUNT_SCHEMA_VERSION:
        raise ValueError("Trading account schema_version 不相容")
    if str(state.get("runtime_domain", "")) != TRADING_RUNTIME_DOMAIN:
        raise ValueError("Trading account runtime_domain 必須是 trading")
    revision = int(state.get("revision", -1))
    if revision < 0:
        raise ValueError("Trading account revision 不可為負數")
    cash_milli = state.get("cash_milli")
    if cash_milli is not None and int(cash_milli) < 0:
        raise ValueError("Trading account cash_milli 不可為負數")

    positions = state.get("positions")
    if not isinstance(positions, dict):
        raise ValueError("Trading account positions 必須是 object")
    for key, record in positions.items():
        ticker = _normalize_ticker(key)
        if ticker != _normalize_ticker(record.get("ticker")):
            raise ValueError(f"Trading position key/ticker 不一致: {key!r}")
        if record.get("source") not in POSITION_SOURCES:
            raise ValueError(f"Trading position source 不合法: {record.get('source')!r}")
        broker = record.get("broker")
        if not isinstance(broker, dict):
            raise ValueError(f"Trading position broker state 缺失: {ticker}")
        qty = int(broker.get("qty", 0) or 0)
        initial_qty = int(broker.get("initial_qty", 0) or 0)
        initial_cost = int(broker.get("initial_cost_basis_milli", 0) or 0)
        remaining_cost = int(broker.get("remaining_cost_basis_milli", 0) or 0)
        if qty <= 0 or initial_qty <= 0 or qty > initial_qty:
            raise ValueError(f"Trading position qty 不合法: {ticker}")
        if initial_cost <= 0 or remaining_cost < 0 or remaining_cost > initial_cost:
            raise ValueError(f"Trading position cost basis 不合法: {ticker}")
        _normalize_iso_date(broker.get("entry_date"), field_name="entry_date")
        entry_order_id = broker.get("entry_order_id")
        if entry_order_id is not None and not str(entry_order_id).strip():
            raise ValueError(f"Trading position entry_order_id 不合法: {ticker}")
        management = record.get("strategy_management")
        if not isinstance(management, dict) or management.get("status") not in MANAGEMENT_STATUSES:
            raise ValueError(f"Trading strategy management status 不合法: {ticker}")
        if record.get("source") == POSITION_SOURCE_MANUAL_ADOPTED:
            if management.get("status") != MANAGEMENT_STATUS_UNMANAGED or management.get("position_state") is not None:
                raise ValueError(f"manual adopted position 不得偽造 strategy state: {ticker}")
        if record.get("source") == POSITION_SOURCE_STRATEGY_FILL:
            position_state = management.get("position_state")
            if management.get("status") != MANAGEMENT_STATUS_ACTIVE or not isinstance(position_state, dict):
                raise ValueError(f"strategy fill 必須持有 active strategy position: {ticker}")
            if int(position_state.get("qty", -1)) != qty:
                raise ValueError(f"strategy/broker qty 不一致: {ticker}")
            if int(position_state.get("remaining_cost_basis_milli", -1)) != remaining_cost:
                raise ValueError(f"strategy/broker cost basis 不一致: {ticker}")
            last_rollforward_date = _normalize_iso_date(
                management.get("last_rollforward_date"), field_name="last_rollforward_date"
            )
            management_start_date = _normalize_iso_date(
                management.get("management_start_date"), field_name="management_start_date"
            )
            if (
                last_rollforward_date is not None
                and management_start_date is not None
                and last_rollforward_date < management_start_date
            ):
                raise ValueError(f"Trading last_rollforward_date 不得早於 management_start_date: {ticker}")

    events = state.get("events")
    if not isinstance(events, list) or len(events) != revision + 1:
        raise ValueError("Trading account event count 必須與 revision 連續一致")
    previous_hash = None
    for expected_revision, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("Trading account event 必須是 object")
        if int(event.get("revision", -1)) != expected_revision:
            raise ValueError("Trading account event revision 不連續")
        if event.get("prev_event_hash") != previous_hash:
            raise ValueError("Trading account event hash chain 斷裂")
        actual_hash = canonical_json_sha256(_event_hash_payload(event))
        if event.get("event_hash") != actual_hash:
            raise ValueError("Trading account event hash 不一致")
        previous_hash = actual_hash


def build_trading_account_read_model(state: dict[str, Any]) -> dict[str, Any]:
    validate_trading_account_state(state)
    positions = []
    for ticker in sorted(state["positions"]):
        record = state["positions"][ticker]
        broker = record["broker"]
        qty = int(broker["qty"])
        remaining_cost_milli = int(broker["remaining_cost_basis_milli"])
        positions.append(
            {
                "ticker": ticker,
                "source": record["source"],
                "qty": qty,
                "average_cost": calc_average_price_from_total_milli(remaining_cost_milli, qty),
                "remaining_cost_basis": milli_to_money(remaining_cost_milli),
                "realized_pnl": milli_to_money(int(broker.get("realized_pnl_milli", 0) or 0)),
                "entry_date": broker.get("entry_date"),
                "management_status": record["strategy_management"]["status"],
                "last_rollforward_date": record["strategy_management"].get("last_rollforward_date"),
                "effective_stop": (
                    None
                    if not isinstance(record["strategy_management"].get("position_state"), dict)
                    else milli_to_price(int(record["strategy_management"]["position_state"].get("sl_milli") or 0))
                ),
                "has_sell_history": _has_confirmed_sell_history(state, ticker),
            }
        )
    return {
        "schema_version": int(state["schema_version"]),
        "revision": int(state["revision"]),
        "cash": None if state.get("cash_milli") is None else milli_to_money(int(state["cash_milli"])),
        "position_count": len(positions),
        "positions": positions,
        "updated_at": state.get("updated_at"),
    }


__all__ = [
    "TRADING_ACCOUNT_SCHEMA_VERSION",
    "POSITION_SOURCE_MANUAL_ADOPTED",
    "POSITION_SOURCE_STRATEGY_FILL",
    "MANAGEMENT_STATUS_UNMANAGED",
    "MANAGEMENT_STATUS_ACTIVE",
    "build_empty_trading_account_state",
    "set_trading_account_cash",
    "adopt_manual_trading_position",
    "correct_manual_trading_position",
    "remove_manual_trading_position",
    "apply_confirmed_strategy_buy_fill",
    "apply_confirmed_strategy_buy_fill_increment",
    "apply_trading_strategy_management_rollforward",
    "apply_confirmed_sell_fill",
    "validate_trading_account_state",
    "build_trading_account_read_model",
]
