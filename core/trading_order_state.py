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
TRADING_ORDER_STATUS_PARTIAL = "PARTIAL"
TRADING_ORDER_STATUS_FILLED = "FILLED"
TRADING_ORDER_STATUS_CANCELLED = "CANCELLED"
TRADING_ACTIVE_ORDER_STATUSES = frozenset({TRADING_ORDER_STATUS_ORDERED, TRADING_ORDER_STATUS_PARTIAL})
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
    frozen_params: dict[str, Any] | None = None,
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

    frozen_params_payload = None if frozen_params is None else deepcopy(frozen_params)
    frozen_params_sha256 = None if frozen_params_payload is None else canonical_json_sha256(frozen_params_payload)

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
        "filled_qty": 0,
        "remaining_qty": qty,
        "fills": [],
        "filled_at": None,
        "frozen_params": frozen_params_payload,
        "frozen_params_sha256": frozen_params_sha256,
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



def record_trading_buy_order_fill(
    state: dict[str, Any],
    *,
    order_id: str,
    fill_id: str,
    fill_qty: int,
    fill_price,
    trade_date: str,
    net_buy_total_milli: int,
    timestamp: str,
    mutation_id: str,
) -> dict[str, Any]:
    validate_trading_order_state(state)
    order_id_text = str(order_id or "").strip()
    if order_id_text not in state["orders"]:
        raise ValueError(f"Trading order 不存在: {order_id_text}")
    updated = deepcopy(state)
    record = updated["orders"][order_id_text]
    from_status = str(record.get("status") or "")
    if from_status not in TRADING_ACTIVE_ORDER_STATUSES:
        raise ValueError(
            f"Trading order 只有 ORDERED/PARTIAL 可確認成交: {order_id_text} status={from_status}"
        )
    qty = int(fill_qty)
    if qty <= 0:
        raise ValueError("Trading fill_qty 必須 > 0")
    remaining_before = int(record.get("remaining_qty") or 0)
    if qty > remaining_before:
        raise ValueError(
            f"Trading fill_qty 超過未成交股數: fill={qty}, remaining={remaining_before}"
        )
    fill_price_milli = price_to_milli(fill_price)
    if fill_price_milli <= 0:
        raise ValueError("Trading fill_price 必須 > 0")
    if fill_price_milli > int(record["limit_price_milli"]):
        raise ValueError("Trading BUY 實際成交價不可高於原始買入限價")
    trade_date_text = str(trade_date or "").strip()
    if not trade_date_text:
        raise ValueError("Trading fill trade_date 必填")
    fills = list(record.get("fills") or [])
    if fills and any(str(row.get("trade_date") or "") != trade_date_text for row in fills):
        raise ValueError("同一 Trading order 的多次 partial fill 必須發生於同一交易日")
    fill_id_text = str(fill_id or "").strip()
    if not fill_id_text:
        raise ValueError("Trading fill_id 不可為空")
    if any(str(row.get("fill_id") or "") == fill_id_text for row in fills):
        raise ValueError(f"Trading fill_id 已存在: {fill_id_text}")
    net_buy = int(net_buy_total_milli)
    if net_buy <= 0:
        raise ValueError("Trading fill net_buy_total_milli 必須 > 0")

    fills.append({
        "fill_id": fill_id_text,
        "qty": qty,
        "fill_price_milli": fill_price_milli,
        "trade_date": trade_date_text,
        "net_buy_total_milli": net_buy,
        "confirmed_at": str(timestamp),
    })
    filled_qty = sum(int(row["qty"]) for row in fills)
    remaining_qty = int(record["qty"]) - filled_qty
    if remaining_qty < 0:
        raise RuntimeError("Trading order cumulative fill 超過原始 qty")
    to_status = TRADING_ORDER_STATUS_FILLED if remaining_qty == 0 else TRADING_ORDER_STATUS_PARTIAL
    record["fills"] = fills
    record["filled_qty"] = filled_qty
    record["remaining_qty"] = remaining_qty
    record["status"] = to_status
    record["filled_at"] = str(timestamp) if to_status == TRADING_ORDER_STATUS_FILLED else None
    return _append_event(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_order_fill",
        timestamp=timestamp,
        details={
            "order_id": order_id_text,
            "ticker": record["ticker"],
            "fill_id": fill_id_text,
            "fill_qty": qty,
            "fill_price_milli": fill_price_milli,
            "trade_date": trade_date_text,
            "net_buy_total_milli": net_buy,
            "filled_qty": filled_qty,
            "remaining_qty": remaining_qty,
            "from_status": from_status,
            "to_status": to_status,
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
    from_status = str(record.get("status") or "")
    if from_status not in TRADING_ACTIVE_ORDER_STATUSES:
        raise ValueError(
            f"Trading order 只有 ORDERED/PARTIAL 可確認取消: {order_id_text} status={from_status}"
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
            "from_status": from_status,
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
            raise ValueError("Trading broker order side 目前只允許 BUY")
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
        allowed_statuses = {
            TRADING_ORDER_STATUS_ORDERED,
            TRADING_ORDER_STATUS_PARTIAL,
            TRADING_ORDER_STATUS_FILLED,
            TRADING_ORDER_STATUS_CANCELLED,
        }
        if status not in allowed_statuses:
            raise ValueError(f"Trading order status 不合法: {status}")
        if not str(record.get("ordered_at") or ""):
            raise ValueError("Trading order record 必須有 ordered_at")

        frozen_params = record.get("frozen_params")
        frozen_params_sha = record.get("frozen_params_sha256")
        if frozen_params is not None:
            if not isinstance(frozen_params, dict):
                raise ValueError("Trading order frozen_params 必須是 object")
            if str(frozen_params_sha or "") != canonical_json_sha256(frozen_params):
                raise ValueError("Trading order frozen_params hash 不一致")

        fills = record.get("fills", [])
        if not isinstance(fills, list):
            raise ValueError("Trading order fills 必須是 list")
        fill_ids = set()
        filled_qty = 0
        fill_trade_dates = set()
        for fill in fills:
            if not isinstance(fill, dict):
                raise ValueError("Trading order fill record 必須是 object")
            fill_id = str(fill.get("fill_id") or "")
            if not fill_id or fill_id in fill_ids:
                raise ValueError("Trading order fill_id 必須存在且不可重複")
            fill_ids.add(fill_id)
            fill_qty = int(fill.get("qty") or 0)
            if fill_qty <= 0 or int(fill.get("fill_price_milli") or 0) <= 0:
                raise ValueError("Trading order fill qty/price 不合法")
            if int(fill.get("net_buy_total_milli") or 0) <= 0:
                raise ValueError("Trading order fill net_buy_total_milli 不合法")
            trade_date = str(fill.get("trade_date") or "")
            if not trade_date or not str(fill.get("confirmed_at") or ""):
                raise ValueError("Trading order fill 缺少 trade_date/confirmed_at")
            fill_trade_dates.add(trade_date)
            filled_qty += fill_qty
        if len(fill_trade_dates) > 1:
            raise ValueError("同一 Trading order 的 partial fills 不得跨交易日")
        declared_filled = int(record.get("filled_qty", 0) or 0)
        declared_remaining = int(record.get("remaining_qty", int(record["qty"]) - declared_filled) or 0)
        if declared_filled != filled_qty or declared_remaining != int(record["qty"]) - filled_qty:
            raise ValueError("Trading order filled_qty/remaining_qty 與 fills 不一致")
        if filled_qty < 0 or filled_qty > int(record["qty"]):
            raise ValueError("Trading order filled_qty 不合法")
        if filled_qty > 0 and frozen_params is None:
            raise ValueError("Trading 已成交 order 必須持有 frozen_params")

        if status == TRADING_ORDER_STATUS_ORDERED:
            if filled_qty != 0 or record.get("cancelled_at") is not None or record.get("filled_at") is not None:
                raise ValueError("Trading ORDERED record 的 fill/cancel 狀態不一致")
        elif status == TRADING_ORDER_STATUS_PARTIAL:
            if not (0 < filled_qty < int(record["qty"])) or record.get("cancelled_at") is not None or record.get("filled_at") is not None:
                raise ValueError("Trading PARTIAL record 狀態不一致")
        elif status == TRADING_ORDER_STATUS_FILLED:
            if filled_qty != int(record["qty"]) or not str(record.get("filled_at") or "") or record.get("cancelled_at") is not None:
                raise ValueError("Trading FILLED record 狀態不一致")
        elif status == TRADING_ORDER_STATUS_CANCELLED:
            if filled_qty >= int(record["qty"]) or not str(record.get("cancelled_at") or "") or record.get("filled_at") is not None:
                raise ValueError("Trading CANCELLED record 狀態不一致")
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
                "filled_qty": int(record.get("filled_qty", 0) or 0),
                "remaining_qty": int(record.get("remaining_qty", record["qty"]) or 0),
                "average_fill_price": (
                    None
                    if not record.get("fills")
                    else milli_to_price(
                        sum(int(fill["fill_price_milli"]) * int(fill["qty"]) for fill in record["fills"])
                        // max(1, sum(int(fill["qty"]) for fill in record["fills"]))
                    )
                ),
                "limit_price": milli_to_price(int(record["limit_price_milli"])),
                "reserved_cost": milli_to_money(int(record["reserved_cost_milli"])),
                "broker_order_id": record.get("broker_order_id"),
                "information_date": record["information_date"],
                "ordered_at": record["ordered_at"],
                "cancelled_at": record.get("cancelled_at"),
                "filled_at": record.get("filled_at"),
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
    "TRADING_ORDER_STATUS_PARTIAL",
    "TRADING_ORDER_STATUS_FILLED",
    "TRADING_ORDER_STATUS_CANCELLED",
    "TRADING_ACTIVE_ORDER_STATUSES",
    "TRADING_ORDER_SIDE_BUY",
    "build_empty_trading_order_state",
    "build_trading_proposal_key",
    "append_ordered_trading_proposal",
    "record_trading_buy_order_fill",
    "cancel_ordered_trading_order",
    "active_trading_orders",
    "has_active_trading_orders",
    "validate_trading_order_state",
    "build_trading_order_read_model",
]
