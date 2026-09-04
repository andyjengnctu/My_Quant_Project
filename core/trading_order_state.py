"""Canonical persistent Trading broker-order state and transitions."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.exact_accounting import milli_to_money, milli_to_price, price_to_milli
from core.file_integrity import canonical_json_sha256
from core.runtime_domains import RUNTIME_DOMAIN_TRADING

TRADING_ORDER_STATE_SCHEMA_VERSION = 1
TRADING_ORDER_STATE_FILENAME = "orders.json"
TRADING_ORDER_STATUS_ORDERED = "ORDERED"
TRADING_ORDER_STATUS_CANCELLED = "CANCELLED"
TRADING_ACTIVE_ORDER_STATUSES = frozenset({TRADING_ORDER_STATUS_ORDERED})
TRADING_ORDER_SIDE_BUY = "BUY"


def _event_hash_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "event_hash"}


def _normalize_optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_ticker(value) -> str:
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError("Trading order ticker 不可為空")
    return text


def _append_event(
    state: dict[str, Any],
    *,
    mutation_id: str,
    mutation_type: str,
    timestamp: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(state)
    revision = int(updated["revision"]) + 1
    previous_hash = updated["events"][-1]["event_hash"] if updated["events"] else None
    event = {
        "revision": revision,
        "mutation_id": str(mutation_id),
        "mutation_type": str(mutation_type),
        "timestamp": str(timestamp),
        "details": deepcopy(details),
        "prev_event_hash": previous_hash,
    }
    event["event_hash"] = canonical_json_sha256(_event_hash_payload(event))
    updated["revision"] = revision
    updated["updated_at"] = str(timestamp)
    updated["events"].append(event)
    return updated


def build_empty_trading_order_state(*, timestamp: str, mutation_id: str) -> dict[str, Any]:
    initial_event = {
        "revision": 0,
        "mutation_id": str(mutation_id),
        "mutation_type": "initialize_order_state",
        "timestamp": str(timestamp),
        "details": {},
        "prev_event_hash": None,
    }
    initial_event["event_hash"] = canonical_json_sha256(_event_hash_payload(initial_event))
    state = {
        "schema_version": TRADING_ORDER_STATE_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "revision": 0,
        "created_at": str(timestamp),
        "updated_at": str(timestamp),
        "orders": {},
        "events": [initial_event],
    }
    validate_trading_order_state(state)
    return state


def build_trading_proposal_key(*, plan_fingerprint: str, rank: int, ticker: str) -> str:
    return canonical_json_sha256(
        {
            "plan_fingerprint": str(plan_fingerprint),
            "rank": int(rank),
            "ticker": _normalize_ticker(ticker),
        }
    )


def append_ordered_trading_proposal(
    state: dict[str, Any],
    *,
    order_id: str,
    proposal: dict[str, Any],
    plan: dict[str, Any],
    timestamp: str,
    mutation_id: str,
    broker_order_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_order_state(state)
    order_id_text = str(order_id or "").strip()
    if not order_id_text:
        raise ValueError("Trading order_id 不可為空")
    if order_id_text in state["orders"]:
        raise ValueError(f"Trading order_id 已存在: {order_id_text}")

    plan_fingerprint = str(plan.get("plan_fingerprint") or "").strip()
    if not plan_fingerprint:
        raise ValueError("Trading proposed plan 缺少 plan_fingerprint")
    rank = int(proposal.get("rank") or 0)
    if rank <= 0:
        raise ValueError("Trading proposed order rank 必須 > 0")
    ticker = _normalize_ticker(proposal.get("ticker"))
    proposal_key = build_trading_proposal_key(
        plan_fingerprint=plan_fingerprint,
        rank=rank,
        ticker=ticker,
    )
    if any(str(row.get("proposal_key")) == proposal_key for row in state["orders"].values()):
        raise ValueError(f"同一 Trading proposal 已記錄過送單狀態: {ticker} rank={rank}")

    qty = int(proposal.get("qty") or 0)
    if qty <= 0:
        raise ValueError("Trading ordered qty 必須 > 0")
    reserved_cost_milli = int(proposal.get("reserved_cost_milli") or 0)
    if reserved_cost_milli <= 0:
        raise ValueError("Trading ordered reserved_cost_milli 必須 > 0")

    limit_price_milli = price_to_milli(proposal.get("limit_price"))
    init_sl_milli = price_to_milli(proposal.get("init_sl"))
    init_trail_milli = price_to_milli(proposal.get("init_trail"))
    target_price_milli = price_to_milli(proposal.get("target_price"))
    entry_atr_value = proposal.get("entry_atr")
    entry_atr_milli = None if entry_atr_value is None else price_to_milli(entry_atr_value)

    record = {
        "order_id": order_id_text,
        "proposal_key": proposal_key,
        "plan_fingerprint": plan_fingerprint,
        "information_date": str(plan.get("information_date") or ""),
        "account_revision": int(plan.get("account_revision")),
        "selected_params_sha256": str(plan.get("selected_params_sha256") or ""),
        "candidate_snapshot_sha256": str(plan.get("candidate_snapshot_sha256") or ""),
        "strategy_id": str(plan.get("strategy_id") or ""),
        "param_selector": str(plan.get("param_selector") or ""),
        "rank": rank,
        "side": TRADING_ORDER_SIDE_BUY,
        "ticker": ticker,
        "kind": str(proposal.get("kind") or ""),
        "entry_type": str(proposal.get("entry_type") or proposal.get("kind") or "normal"),
        "qty": qty,
        "limit_price_milli": limit_price_milli,
        "reserved_cost_milli": reserved_cost_milli,
        "init_sl_milli": init_sl_milli,
        "init_trail_milli": init_trail_milli,
        "target_price_milli": target_price_milli,
        "entry_atr_milli": entry_atr_milli,
        "security_profile": deepcopy(proposal.get("security_profile")),
        "status": TRADING_ORDER_STATUS_ORDERED,
        "ordered_at": str(timestamp),
        "broker_order_id": _normalize_optional_text(broker_order_id),
        "cancelled_at": None,
        "cancel_note": None,
        "note": _normalize_optional_text(note),
    }
    updated = deepcopy(state)
    updated["orders"][order_id_text] = record
    return _append_event(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_order_submission",
        timestamp=timestamp,
        details={
            "order_id": order_id_text,
            "proposal_key": proposal_key,
            "plan_fingerprint": plan_fingerprint,
            "ticker": ticker,
            "rank": rank,
            "from_status": "PROPOSED",
            "to_status": TRADING_ORDER_STATUS_ORDERED,
            "broker_order_id": record["broker_order_id"],
            "qty": qty,
            "limit_price_milli": limit_price_milli,
            "reserved_cost_milli": reserved_cost_milli,
        },
    )


def cancel_ordered_trading_order(
    state: dict[str, Any],
    *,
    order_id: str,
    timestamp: str,
    mutation_id: str,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_order_state(state)
    order_id_text = str(order_id or "").strip()
    if order_id_text not in state["orders"]:
        raise ValueError(f"Trading order 不存在: {order_id_text}")
    updated = deepcopy(state)
    record = updated["orders"][order_id_text]
    if record.get("status") != TRADING_ORDER_STATUS_ORDERED:
        raise ValueError(
            f"Trading order 只有 ORDERED 可確認取消: {order_id_text} status={record.get('status')}"
        )
    record["status"] = TRADING_ORDER_STATUS_CANCELLED
    record["cancelled_at"] = str(timestamp)
    record["cancel_note"] = _normalize_optional_text(note)
    return _append_event(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_order_cancellation",
        timestamp=timestamp,
        details={
            "order_id": order_id_text,
            "ticker": record["ticker"],
            "from_status": TRADING_ORDER_STATUS_ORDERED,
            "to_status": TRADING_ORDER_STATUS_CANCELLED,
            "broker_order_id": record.get("broker_order_id"),
            "note": record.get("cancel_note"),
        },
    )


def active_trading_orders(state: dict[str, Any]) -> list[dict[str, Any]]:
    validate_trading_order_state(state)
    return [
        deepcopy(row)
        for row in state["orders"].values()
        if str(row.get("status")) in TRADING_ACTIVE_ORDER_STATUSES
    ]


def has_active_trading_orders(state: dict[str, Any]) -> bool:
    return bool(active_trading_orders(state))


def validate_trading_order_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError("Trading order state 必須是 dict")
    if int(state.get("schema_version", -1)) != TRADING_ORDER_STATE_SCHEMA_VERSION:
        raise ValueError("Trading order state schema_version 不相容")
    if str(state.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading order state runtime_domain 必須是 trading")
    revision = int(state.get("revision", -1))
    if revision < 0:
        raise ValueError("Trading order state revision 不可為負數")
    orders = state.get("orders")
    if not isinstance(orders, dict):
        raise ValueError("Trading order state orders 必須是 object")
    proposal_keys: set[str] = set()
    for key, record in orders.items():
        if not isinstance(record, dict):
            raise ValueError("Trading order record 必須是 object")
        order_id = str(record.get("order_id") or "")
        if str(key) != order_id or not order_id:
            raise ValueError("Trading order key/order_id 不一致")
        proposal_key = str(record.get("proposal_key") or "")
        if not proposal_key or proposal_key in proposal_keys:
            raise ValueError("Trading order proposal_key 必須存在且不可重複")
        proposal_keys.add(proposal_key)
        if str(record.get("side") or "") != TRADING_ORDER_SIDE_BUY:
            raise ValueError("Trading Round 8 order side 只允許 BUY")
        _normalize_ticker(record.get("ticker"))
        if int(record.get("rank") or 0) <= 0 or int(record.get("qty") or 0) <= 0:
            raise ValueError("Trading order rank/qty 不合法")
        if int(record.get("limit_price_milli") or 0) <= 0:
            raise ValueError("Trading order limit_price_milli 不合法")
        if int(record.get("reserved_cost_milli") or 0) <= 0:
            raise ValueError("Trading order reserved_cost_milli 不合法")
        for field in ("init_sl_milli", "init_trail_milli", "target_price_milli"):
            if int(record.get(field) or 0) <= 0:
                raise ValueError(f"Trading order {field} 不合法")
        status = str(record.get("status") or "")
        if status not in {TRADING_ORDER_STATUS_ORDERED, TRADING_ORDER_STATUS_CANCELLED}:
            raise ValueError(f"Trading order status 不合法: {status}")
        if not str(record.get("ordered_at") or ""):
            raise ValueError("Trading ORDERED record 必須有 ordered_at")
        if status == TRADING_ORDER_STATUS_ORDERED and record.get("cancelled_at") is not None:
            raise ValueError("Trading ORDERED record 不得有 cancelled_at")
        if status == TRADING_ORDER_STATUS_CANCELLED and not str(record.get("cancelled_at") or ""):
            raise ValueError("Trading CANCELLED record 必須有 cancelled_at")
        for field in (
            "plan_fingerprint",
            "information_date",
            "selected_params_sha256",
            "candidate_snapshot_sha256",
            "strategy_id",
            "param_selector",
        ):
            if not str(record.get(field) or ""):
                raise ValueError(f"Trading order 缺少 immutable binding: {field}")

    events = state.get("events")
    if not isinstance(events, list) or len(events) != revision + 1:
        raise ValueError("Trading order event count 必須與 revision 連續一致")
    previous_hash = None
    for expected_revision, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("Trading order event 必須是 object")
        if int(event.get("revision", -1)) != expected_revision:
            raise ValueError("Trading order event revision 不連續")
        if event.get("prev_event_hash") != previous_hash:
            raise ValueError("Trading order event hash chain 斷裂")
        actual_hash = canonical_json_sha256(_event_hash_payload(event))
        if event.get("event_hash") != actual_hash:
            raise ValueError("Trading order event hash 不一致")
        previous_hash = actual_hash


def build_trading_order_read_model(state: dict[str, Any]) -> dict[str, Any]:
    validate_trading_order_state(state)
    rows = []
    for record in sorted(
        state["orders"].values(),
        key=lambda row: (str(row.get("information_date")), str(row.get("ordered_at")), str(row.get("order_id"))),
        reverse=True,
    ):
        rows.append(
            {
                "order_id": record["order_id"],
                "ticker": record["ticker"],
                "status": record["status"],
                "qty": int(record["qty"]),
                "limit_price": milli_to_price(int(record["limit_price_milli"])),
                "reserved_cost": milli_to_money(int(record["reserved_cost_milli"])),
                "broker_order_id": record.get("broker_order_id"),
                "information_date": record["information_date"],
                "ordered_at": record["ordered_at"],
                "cancelled_at": record.get("cancelled_at"),
                "plan_fingerprint": record["plan_fingerprint"],
            }
        )
    return {
        "schema_version": int(state["schema_version"]),
        "revision": int(state["revision"]),
        "active_order_count": sum(1 for row in rows if row["status"] in TRADING_ACTIVE_ORDER_STATUSES),
        "order_count": len(rows),
        "orders": rows,
        "updated_at": state.get("updated_at"),
    }


__all__ = [
    "TRADING_ORDER_STATE_SCHEMA_VERSION",
    "TRADING_ORDER_STATE_FILENAME",
    "TRADING_ORDER_STATUS_ORDERED",
    "TRADING_ORDER_STATUS_CANCELLED",
    "TRADING_ACTIVE_ORDER_STATUSES",
    "TRADING_ORDER_SIDE_BUY",
    "build_empty_trading_order_state",
    "build_trading_proposal_key",
    "append_ordered_trading_proposal",
    "cancel_ordered_trading_order",
    "active_trading_orders",
    "has_active_trading_orders",
    "validate_trading_order_state",
    "build_trading_order_read_model",
]
