"""Canonical persistent Trading broker-order state and transitions."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.event_hash_chain import compute_event_hash
from core.exact_accounting import milli_to_money, milli_to_price, price_to_milli
from core.file_integrity import canonical_json_sha256
from core.trading_identity import (
    normalize_trading_date,
    normalize_trading_ticker,
    require_trading_date_after,
    require_trading_date_not_before,
)
from core.runtime_domains import RUNTIME_DOMAIN_TRADING

TRADING_ORDER_STATE_SCHEMA_VERSION = 1
TRADING_ORDER_STATE_FILENAME = "orders.json"
TRADING_ORDER_STATUS_ORDERED = "ORDERED"
TRADING_ORDER_STATUS_PARTIAL = "PARTIAL"
TRADING_ORDER_STATUS_FILLED = "FILLED"
TRADING_ORDER_STATUS_CANCELLED = "CANCELLED"
TRADING_ACTIVE_ORDER_STATUSES = frozenset({TRADING_ORDER_STATUS_ORDERED, TRADING_ORDER_STATUS_PARTIAL})
TRADING_ORDER_SIDE_BUY = "BUY"
TRADING_ORDER_SIDE_SELL = "SELL"
TRADING_ORDER_PURPOSE_ENTRY = "ENTRY_BUY"
TRADING_ORDER_PURPOSE_PROTECTION_STOP = "PROTECTION_STOP"
TRADING_ORDER_PURPOSE_PROTECTION_TP = "PROTECTION_TP"
TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER = "PROTECTION_STOP_REMAINDER"
TRADING_ORDER_PURPOSE_INDICATOR_EXIT = "INDICATOR_EXIT"
TRADING_PROTECTION_ORDER_PURPOSES = frozenset({
    TRADING_ORDER_PURPOSE_PROTECTION_STOP,
    TRADING_ORDER_PURPOSE_PROTECTION_TP,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
})
TRADING_SELL_ORDER_PURPOSES = frozenset({
    *TRADING_PROTECTION_ORDER_PURPOSES,
    TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
})
TRADING_PROTECTION_ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
TRADING_PROTECTION_ORDER_TYPE_LIMIT = "LIMIT"
TRADING_PROTECTION_ORDER_TYPE_MARKET = "MARKET"
TRADING_INDICATOR_ORDER_TYPE_MARKET = "MARKET"


TRADING_PROTECTION_ACTION_STOP_FULL = "STOP_FULL"
TRADING_PROTECTION_ACTION_TP_HALF = "TP_HALF"
TRADING_PROTECTION_ACTION_STOP_REMAINDER_EXIT = "STOP_REMAINDER_EXIT"
TRADING_PROTECTION_ACTION_SPECS = {
    TRADING_PROTECTION_ACTION_STOP_FULL: {
        "purpose": TRADING_ORDER_PURPOSE_PROTECTION_STOP,
        "order_type": TRADING_PROTECTION_ORDER_TYPE_STOP_MARKET,
    },
    TRADING_PROTECTION_ACTION_TP_HALF: {
        "purpose": TRADING_ORDER_PURPOSE_PROTECTION_TP,
        "order_type": TRADING_PROTECTION_ORDER_TYPE_LIMIT,
    },
    TRADING_PROTECTION_ACTION_STOP_REMAINDER_EXIT: {
        "purpose": TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
        "order_type": TRADING_PROTECTION_ORDER_TYPE_MARKET,
    },
}


def get_trading_protection_action_spec(action: object) -> dict[str, str]:
    action_key = str(action or "").strip()
    spec = TRADING_PROTECTION_ACTION_SPECS.get(action_key)
    if spec is None:
        raise ValueError(f"Trading protection action 不合法: {action_key or '-'}")
    return dict(spec)


def _normalize_optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_normalize_ticker = normalize_trading_ticker


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
    event["event_hash"] = compute_event_hash(event)
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
    initial_event["event_hash"] = compute_event_hash(initial_event)
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
        "purpose": TRADING_ORDER_PURPOSE_ENTRY,
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




def build_trading_protection_key(
    *,
    protection_plan_fingerprint: str,
    position_plan_fingerprint: str,
    action: str,
    ticker: str,
) -> str:
    return canonical_json_sha256(
        {
            "protection_plan_fingerprint": str(protection_plan_fingerprint),
            "position_plan_fingerprint": str(position_plan_fingerprint),
            "action": str(action),
            "ticker": _normalize_ticker(ticker),
        }
    )


def build_trading_indicator_exit_key(*, signal_key: str, ticker: str, attempt: int) -> str:
    key = str(signal_key or "").strip()
    attempt_int = int(attempt)
    if not key:
        raise ValueError("Trading indicator exit signal_key 不可為空")
    if attempt_int <= 0:
        raise ValueError("Trading indicator exit attempt 必須 > 0")
    return canonical_json_sha256(
        {
            "signal_key": key,
            "ticker": _normalize_ticker(ticker),
            "purpose": TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
            "attempt": attempt_int,
        }
    )


def _build_ordered_protection_record(
    *,
    order_id: str,
    plan: dict[str, Any],
    position_plan: dict[str, Any],
    leg: dict[str, Any],
    account_revision: int,
    timestamp: str,
    broker_order_id: str | None,
    note: str | None,
    broker_oco_group_id: str | None,
    broker_native_oco_confirmed: bool,
) -> dict[str, Any]:
    order_id_text = str(order_id or "").strip()
    if not order_id_text:
        raise ValueError("Trading protection order_id 不可為空")
    plan_fingerprint = str(plan.get("plan_fingerprint") or "").strip()
    position_fp = str(position_plan.get("position_plan_fingerprint") or "").strip()
    if not plan_fingerprint or not position_fp:
        raise ValueError("Trading protection plan 缺少 fingerprint")
    ticker = _normalize_ticker(position_plan.get("ticker"))
    action = str(leg.get("action") or "").strip()
    action_spec = get_trading_protection_action_spec(action)
    purpose = action_spec["purpose"]
    expected_type = action_spec["order_type"]
    order_type = str(leg.get("order_type") or "").strip()
    if order_type != expected_type:
        raise ValueError(f"Trading protection order_type 與 action 不一致: {action}/{order_type}")
    qty = int(leg.get("qty") or 0)
    position_qty = int(position_plan.get("position_qty") or 0)
    if qty <= 0 or position_qty <= 0 or qty > position_qty:
        raise ValueError("Trading protection qty 不可超過目前實際持股")
    trigger_milli = leg.get("trigger_price_milli")
    limit_milli = leg.get("limit_price_milli")
    trigger_milli = None if trigger_milli is None else int(trigger_milli)
    limit_milli = None if limit_milli is None else int(limit_milli)
    if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP:
        if trigger_milli is None or trigger_milli <= 0 or limit_milli is not None:
            raise ValueError("Trading STOP protection 必須只有有效 trigger price")
    elif purpose == TRADING_ORDER_PURPOSE_PROTECTION_TP:
        if limit_milli is None or limit_milli <= 0 or trigger_milli is not None:
            raise ValueError("Trading TP protection 必須只有有效 limit price")
    else:
        if trigger_milli is not None or limit_milli is not None:
            raise ValueError("Trading STOP remainder forced exit 必須使用 MARKET 且不得有 trigger/limit")
    oco_group = _normalize_optional_text(broker_oco_group_id)
    oco_confirmed = bool(broker_native_oco_confirmed)
    if oco_confirmed and not oco_group:
        raise ValueError("確認 broker-native OCO 時必須提供券商 OCO/互斥群組識別")
    if not oco_confirmed and oco_group:
        raise ValueError("未確認 broker-native OCO 時不得記錄 OCO 群組")
    if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER and (oco_confirmed or oco_group):
        raise ValueError("Trading STOP remainder forced exit 不得標記為 OCO")
    protection_identity = build_trading_protection_key(
        protection_plan_fingerprint=plan_fingerprint,
        position_plan_fingerprint=position_fp,
        action=action,
        ticker=ticker,
    )
    proposal_key = canonical_json_sha256({"protection_identity": protection_identity, "order_id": order_id_text})
    return {
        "order_id": order_id_text,
        "proposal_key": proposal_key,
        "plan_fingerprint": plan_fingerprint,
        "protection_plan_fingerprint": plan_fingerprint,
        "position_plan_fingerprint": position_fp,
        "account_revision": int(account_revision),
        "side": TRADING_ORDER_SIDE_SELL,
        "purpose": purpose,
        "ticker": ticker,
        "rank": int(leg.get("priority") or 0),
        "qty": qty,
        "position_qty_at_submission": position_qty,
        "entry_order_id": str(position_plan.get("entry_order_id") or ""),
        "entry_trade_date": str(position_plan.get("entry_trade_date") or ""),
        "order_type": order_type,
        "trigger_price_milli": trigger_milli,
        "limit_price_milli": limit_milli,
        "reserved_cost_milli": 0,
        "status": TRADING_ORDER_STATUS_ORDERED,
        "filled_qty": 0,
        "remaining_qty": qty,
        "fills": [],
        "filled_at": None,
        "ordered_at": str(timestamp),
        "broker_order_id": _normalize_optional_text(broker_order_id),
        "broker_native_oco_confirmed": oco_confirmed,
        "broker_oco_group_id": oco_group,
        "stop_forced_exit_key": (
            str(position_plan.get("stop_forced_exit_key") or "")
            if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER
            else None
        ),
        "stop_trigger_order_id": (
            str(position_plan.get("stop_trigger_order_id") or "")
            if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER
            else None
        ),
        "stop_trigger_fill_id": (
            str(position_plan.get("stop_trigger_fill_id") or "")
            if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER
            else None
        ),
        "stop_trigger_trade_date": (
            str(position_plan.get("stop_trigger_trade_date") or "")
            if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER
            else None
        ),
        "stop_forced_exit_attempt": (
            int(position_plan.get("stop_forced_exit_attempt") or 0)
            if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER
            else None
        ),
        "cancelled_at": None,
        "cancel_note": None,
        "note": _normalize_optional_text(note),
    }


def append_ordered_trading_indicator_exit(
    state: dict[str, Any],
    *,
    order_id: str,
    exit_plan: dict[str, Any],
    plan: dict[str, Any],
    timestamp: str,
    mutation_id: str,
    broker_order_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_order_state(state)
    order_id_text = str(order_id or "").strip()
    if not order_id_text or order_id_text in state["orders"]:
        raise ValueError("Trading indicator exit order_id 不可為空或重複")
    plan_fingerprint = str(plan.get("plan_fingerprint") or "").strip()
    position_fp = str(exit_plan.get("position_plan_fingerprint") or "").strip()
    signal_key = str(exit_plan.get("signal_key") or "").strip()
    ticker = _normalize_ticker(exit_plan.get("ticker"))
    if not plan_fingerprint or not position_fp or not signal_key:
        raise ValueError("Trading indicator exit plan 缺少 immutable fingerprint/signal_key")
    qty = int(exit_plan.get("qty") or 0)
    position_qty = int(exit_plan.get("position_qty") or 0)
    if qty <= 0 or qty != position_qty:
        raise ValueError("Trading Indicator SELL 必須為送單時完整持股 qty")
    information_date = str(exit_plan.get("signal_information_date") or "").strip()
    entry_order_id = str(exit_plan.get("entry_order_id") or "").strip()
    if not information_date or not entry_order_id:
        raise ValueError("Trading indicator exit 缺少 signal date/entry_order_id")
    prior = [
        row for row in state["orders"].values()
        if str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_INDICATOR_EXIT
        and str(row.get("signal_key") or "") == signal_key
        and _normalize_ticker(row.get("ticker")) == ticker
    ]
    if any(str(row.get("status") or "") != TRADING_ORDER_STATUS_CANCELLED for row in prior):
        raise ValueError("同一 Trading indicator signal 已有未取消／已完成 order，不得重複送單")
    attempt = max([int(row.get("signal_attempt") or 0) for row in prior] or [0]) + 1
    proposal_key = build_trading_indicator_exit_key(signal_key=signal_key, ticker=ticker, attempt=attempt)
    if any(str(row.get("proposal_key") or "") == proposal_key for row in state["orders"].values()):
        raise ValueError("同一 Trading indicator exit attempt 已存在")
    record = {
        "order_id": order_id_text,
        "proposal_key": proposal_key,
        "plan_fingerprint": plan_fingerprint,
        "indicator_plan_fingerprint": plan_fingerprint,
        "position_plan_fingerprint": position_fp,
        "signal_key": signal_key,
        "signal_attempt": attempt,
        "information_date": information_date,
        "account_revision": int(plan.get("account_revision")),
        "side": TRADING_ORDER_SIDE_SELL,
        "purpose": TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
        "ticker": ticker,
        "rank": int(exit_plan.get("priority") or 1),
        "qty": qty,
        "position_qty_at_submission": position_qty,
        "entry_order_id": entry_order_id,
        "entry_trade_date": str(exit_plan.get("entry_trade_date") or ""),
        "order_type": TRADING_INDICATOR_ORDER_TYPE_MARKET,
        "trigger_price_milli": None,
        "limit_price_milli": None,
        "reserved_cost_milli": 0,
        "position_state_sha256": str(exit_plan.get("position_state_sha256") or ""),
        "frozen_params_sha256": str(exit_plan.get("frozen_params_sha256") or ""),
        "market_data_sha256": str(exit_plan.get("signal_origin_market_data_sha256") or exit_plan.get("market_data_sha256") or ""),
        "status": TRADING_ORDER_STATUS_ORDERED,
        "filled_qty": 0,
        "remaining_qty": qty,
        "fills": [],
        "filled_at": None,
        "ordered_at": str(timestamp),
        "broker_order_id": _normalize_optional_text(broker_order_id),
        "broker_native_oco_confirmed": False,
        "broker_oco_group_id": None,
        "cancelled_at": None,
        "cancel_note": None,
        "note": _normalize_optional_text(note),
    }
    updated = deepcopy(state)
    updated["orders"][order_id_text] = record
    return _append_event(
        updated, mutation_id=mutation_id, mutation_type="confirm_indicator_exit_submission", timestamp=timestamp,
        details={"order_id": order_id_text, "ticker": ticker, "signal_key": signal_key, "attempt": attempt,
                 "qty": qty, "from_status": "PROPOSED_INDICATOR_EXIT", "to_status": TRADING_ORDER_STATUS_ORDERED,
                 "broker_order_id": record["broker_order_id"]},
    )


def append_ordered_trading_protection_leg(
    state: dict[str, Any],
    *,
    order_id: str,
    plan: dict[str, Any],
    position_plan: dict[str, Any],
    leg: dict[str, Any],
    account_revision: int,
    timestamp: str,
    mutation_id: str,
    broker_order_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_order_state(state)
    position_payload = deepcopy(position_plan)
    if str(leg.get("action") or "") == "STOP_REMAINDER_EXIT":
        forced_key = str(position_payload.get("stop_forced_exit_key") or "").strip()
        if not forced_key:
            raise ValueError("Trading STOP remainder plan 缺少 persistent forced-exit key")
        prior = [
            row for row in state["orders"].values()
            if str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER
            and str(row.get("stop_forced_exit_key") or "") == forced_key
            and _normalize_ticker(row.get("ticker")) == _normalize_ticker(position_payload.get("ticker"))
        ]
        if any(str(row.get("status") or "") != TRADING_ORDER_STATUS_CANCELLED for row in prior):
            raise ValueError("同一 Trading STOP remainder obligation 已有未取消／已完成 order，不得重複送單")
        position_payload["stop_forced_exit_attempt"] = max(
            [int(row.get("stop_forced_exit_attempt") or 0) for row in prior] or [0]
        ) + 1
    record = _build_ordered_protection_record(
        order_id=order_id,
        plan=plan,
        position_plan=position_payload,
        leg=leg,
        account_revision=account_revision,
        timestamp=timestamp,
        broker_order_id=broker_order_id,
        note=note,
        broker_oco_group_id=None,
        broker_native_oco_confirmed=False,
    )
    if record["order_id"] in state["orders"]:
        raise ValueError(f"Trading order_id 已存在: {record['order_id']}")
    if any(str(row.get("proposal_key")) == record["proposal_key"] for row in state["orders"].values()):
        raise ValueError(f"同一 Trading protection leg 已記錄過送單狀態: {record['ticker']} {record['purpose']}")
    updated = deepcopy(state)
    updated["orders"][record["order_id"]] = record
    return _append_event(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_protection_order_submission",
        timestamp=timestamp,
        details={
            "order_id": record["order_id"],
            "ticker": record["ticker"],
            "side": record["side"],
            "purpose": record["purpose"],
            "qty": record["qty"],
            "from_status": "PROPOSED_PROTECTION",
            "to_status": TRADING_ORDER_STATUS_ORDERED,
            "broker_order_id": record.get("broker_order_id"),
        },
    )


def append_ordered_trading_protection_oco_group(
    state: dict[str, Any],
    *,
    stop_order_id: str,
    tp_order_id: str,
    plan: dict[str, Any],
    position_plan: dict[str, Any],
    stop_leg: dict[str, Any],
    tp_leg: dict[str, Any],
    account_revision: int,
    timestamp: str,
    mutation_id: str,
    broker_oco_group_id: str,
    stop_broker_order_id: str | None = None,
    tp_broker_order_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_order_state(state)
    group = _normalize_optional_text(broker_oco_group_id)
    if not group:
        raise ValueError("Stop+TP 同時送單只有在使用者明確確認券商 native OCO/互斥群組時才允許")
    records = [
        _build_ordered_protection_record(
            order_id=stop_order_id,
            plan=plan,
            position_plan=position_plan,
            leg=stop_leg,
            account_revision=account_revision,
            timestamp=timestamp,
            broker_order_id=stop_broker_order_id,
            note=note,
            broker_oco_group_id=group,
            broker_native_oco_confirmed=True,
        ),
        _build_ordered_protection_record(
            order_id=tp_order_id,
            plan=plan,
            position_plan=position_plan,
            leg=tp_leg,
            account_revision=account_revision,
            timestamp=timestamp,
            broker_order_id=tp_broker_order_id,
            note=note,
            broker_oco_group_id=group,
            broker_native_oco_confirmed=True,
        ),
    ]
    if {row["purpose"] for row in records} != {
        TRADING_ORDER_PURPOSE_PROTECTION_STOP,
        TRADING_ORDER_PURPOSE_PROTECTION_TP,
    }:
        raise ValueError("Trading OCO protection group 必須恰好包含 Stop 與 TP")
    existing_ids = set(state["orders"])
    if any(row["order_id"] in existing_ids for row in records):
        raise ValueError("Trading OCO protection order_id 已存在")
    existing_keys = {str(row.get("proposal_key")) for row in state["orders"].values()}
    if any(row["proposal_key"] in existing_keys for row in records):
        raise ValueError("同一 Trading protection leg 已記錄過送單狀態")
    updated = deepcopy(state)
    for row in records:
        updated["orders"][row["order_id"]] = row
    return _append_event(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_protection_oco_submission",
        timestamp=timestamp,
        details={
            "order_ids": [row["order_id"] for row in records],
            "ticker": records[0]["ticker"],
            "purposes": [row["purpose"] for row in records],
            "broker_oco_group_id": group,
            "from_status": "PROPOSED_PROTECTION",
            "to_status": TRADING_ORDER_STATUS_ORDERED,
            "broker_native_oco_confirmed": True,
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
    if str(record.get("side") or "") != TRADING_ORDER_SIDE_BUY:
        raise ValueError("Trading BUY fill reconciliation 只允許 ENTRY BUY order")
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
    trade_date_text = require_trading_date_after(
        trade_date,
        after=record.get("information_date"),
        field_name="BUY fill trade_date",
        after_field_name="information_date",
    )
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


def record_trading_sell_order_fill(
    state: dict[str, Any],
    *,
    order_id: str, fill_id: str, fill_qty: int, fill_price, trade_date: str,
    net_sell_total_milli: int, allocated_cost_milli: int, realized_pnl_milli: int,
    timestamp: str, mutation_id: str,
) -> dict[str, Any]:
    validate_trading_order_state(state)
    oid = str(order_id or "").strip()
    if oid not in state["orders"]:
        raise ValueError(f"Trading order 不存在: {oid}")
    updated = deepcopy(state)
    record = updated["orders"][oid]
    purpose = str(record.get("purpose") or "")
    if str(record.get("side") or "") != TRADING_ORDER_SIDE_SELL or purpose not in TRADING_SELL_ORDER_PURPOSES:
        raise ValueError("Trading SELL fill reconciliation 只允許 canonical SELL order")
    from_status = str(record.get("status") or "")
    if from_status not in TRADING_ACTIVE_ORDER_STATUSES:
        raise ValueError(f"Trading SELL 只有 ORDERED/PARTIAL 可確認成交: {oid} status={from_status}")
    qty = int(fill_qty)
    remaining_before = int(record.get("remaining_qty") or 0)
    if qty <= 0 or qty > remaining_before:
        raise ValueError(f"Trading SELL fill_qty 必須介於 1..{remaining_before}")
    fill_price_milli = price_to_milli(fill_price)
    if fill_price_milli <= 0:
        raise ValueError("Trading SELL fill_price 必須 > 0")
    if purpose == TRADING_ORDER_PURPOSE_PROTECTION_TP and fill_price_milli < int(record.get("limit_price_milli") or 0):
        raise ValueError("Trading TP 實際成交價不可低於原始賣出限價")
    trade_date_text = require_trading_date_after(
        trade_date,
        after=record.get("entry_trade_date"),
        field_name="SELL fill trade_date",
        after_field_name="entry_trade_date",
    )
    if purpose == TRADING_ORDER_PURPOSE_INDICATOR_EXIT:
        trade_date_text = require_trading_date_after(
            trade_date_text,
            after=record.get("information_date"),
            field_name="Indicator SELL fill trade_date",
            after_field_name="signal information_date",
        )
    if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER:
        trade_date_text = require_trading_date_not_before(
            trade_date_text,
            earliest=record.get("stop_trigger_trade_date"),
            field_name="STOP remainder fill trade_date",
            earliest_field_name="STOP trigger trade_date",
        )
    fill_id_text = str(fill_id or "").strip()
    fills = list(record.get("fills") or [])
    if not fill_id_text or any(str(x.get("fill_id") or "") == fill_id_text for x in fills):
        raise ValueError("Trading SELL fill_id 不可為空或重複")
    net_sell = int(net_sell_total_milli); allocated = int(allocated_cost_milli); pnl = int(realized_pnl_milli)
    if net_sell <= 0 or allocated < 0 or pnl != net_sell - allocated:
        raise ValueError("Trading SELL fill exact-accounting payload 不合法")
    fills.append({"fill_id": fill_id_text, "qty": qty, "fill_price_milli": fill_price_milli, "trade_date": trade_date_text,
                  "net_sell_total_milli": net_sell, "allocated_cost_milli": allocated, "realized_pnl_milli": pnl, "confirmed_at": str(timestamp)})
    filled_qty = sum(int(x["qty"]) for x in fills)
    remaining_qty = int(record["qty"]) - filled_qty
    to_status = TRADING_ORDER_STATUS_FILLED if remaining_qty == 0 else TRADING_ORDER_STATUS_PARTIAL
    record.update({"fills": fills, "filled_qty": filled_qty, "remaining_qty": remaining_qty, "status": to_status,
                   "filled_at": str(timestamp) if to_status == TRADING_ORDER_STATUS_FILLED else None})
    oco_cancelled=[]
    if purpose in TRADING_PROTECTION_ORDER_PURPOSES and bool(record.get("broker_native_oco_confirmed")):
        group=str(record.get("broker_oco_group_id") or "")
        for peer_id, peer in updated["orders"].items():
            if peer_id == oid: continue
            if str(peer.get("side") or "") != TRADING_ORDER_SIDE_SELL: continue
            if str(peer.get("purpose") or "") not in TRADING_PROTECTION_ORDER_PURPOSES: continue
            if str(peer.get("broker_oco_group_id") or "") != group: continue
            if str(peer.get("status") or "") in TRADING_ACTIVE_ORDER_STATUSES:
                peer["status"] = TRADING_ORDER_STATUS_CANCELLED
                peer["cancelled_at"] = str(timestamp)
                peer["cancel_note"] = f"broker-native OCO peer fill: {oid}"
                oco_cancelled.append(peer_id)
    mutation_type = "confirm_indicator_sell_fill" if purpose == TRADING_ORDER_PURPOSE_INDICATOR_EXIT else "confirm_protection_sell_fill"
    return _append_event(updated, mutation_id=mutation_id, mutation_type=mutation_type, timestamp=timestamp, details={
        "order_id": oid, "ticker": record["ticker"], "purpose": purpose, "fill_id": fill_id_text, "fill_qty": qty,
        "fill_price_milli": fill_price_milli, "trade_date": trade_date_text, "net_sell_total_milli": net_sell,
        "allocated_cost_milli": allocated, "realized_pnl_milli": pnl, "filled_qty": filled_qty, "remaining_qty": remaining_qty,
        "from_status": from_status, "to_status": to_status, "oco_cancelled_order_ids": oco_cancelled})



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


def active_trading_entry_orders(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in active_trading_orders(state)
        if str(row.get("side") or "") == TRADING_ORDER_SIDE_BUY
        and str(row.get("purpose") or TRADING_ORDER_PURPOSE_ENTRY) == TRADING_ORDER_PURPOSE_ENTRY
    ]


def active_trading_protection_orders(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in active_trading_orders(state)
        if str(row.get("side") or "") == TRADING_ORDER_SIDE_SELL
        and str(row.get("purpose") or "") in TRADING_PROTECTION_ORDER_PURPOSES
    ]


def active_trading_indicator_exit_orders(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in active_trading_orders(state)
        if str(row.get("side") or "") == TRADING_ORDER_SIDE_SELL
        and str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_INDICATOR_EXIT
    ]


def active_trading_sell_orders(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in active_trading_orders(state) if str(row.get("side") or "") == TRADING_ORDER_SIDE_SELL]


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
        side = str(record.get("side") or "")
        purpose = str(record.get("purpose") or (TRADING_ORDER_PURPOSE_ENTRY if side == TRADING_ORDER_SIDE_BUY else ""))
        if side not in {TRADING_ORDER_SIDE_BUY, TRADING_ORDER_SIDE_SELL}:
            raise ValueError(f"Trading broker order side 不合法: {side}")
        _normalize_ticker(record.get("ticker"))
        qty = int(record.get("qty") or 0)
        if qty <= 0:
            raise ValueError("Trading order qty 不合法")
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

        fills = record.get("fills", [])
        if not isinstance(fills, list):
            raise ValueError("Trading order fills 必須是 list")
        filled_qty = 0
        fill_ids: set[str] = set()
        fill_trade_dates: set[str] = set()
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
            trade_date = normalize_trading_date(
                fill.get("trade_date"), field_name="fill.trade_date", allow_none=False
            )
            if not str(fill.get("confirmed_at") or ""):
                raise ValueError("Trading order fill 缺少 confirmed_at")
            fill_trade_dates.add(trade_date)
            filled_qty += fill_qty
        declared_filled = int(record.get("filled_qty", 0) or 0)
        declared_remaining = int(record.get("remaining_qty", qty - declared_filled) or 0)
        if declared_filled != filled_qty or declared_remaining != qty - filled_qty:
            raise ValueError("Trading order filled_qty/remaining_qty 與 fills 不一致")
        if filled_qty < 0 or filled_qty > qty:
            raise ValueError("Trading order filled_qty 不合法")

        if side == TRADING_ORDER_SIDE_BUY:
            if purpose != TRADING_ORDER_PURPOSE_ENTRY:
                raise ValueError("Trading BUY order purpose 必須是 ENTRY_BUY")
            if int(record.get("rank") or 0) <= 0:
                raise ValueError("Trading BUY order rank 不合法")
            if int(record.get("limit_price_milli") or 0) <= 0 or int(record.get("reserved_cost_milli") or 0) <= 0:
                raise ValueError("Trading BUY order limit/reserved cost 不合法")
            for field in ("init_sl_milli", "init_trail_milli", "target_price_milli"):
                if int(record.get(field) or 0) <= 0:
                    raise ValueError(f"Trading BUY order {field} 不合法")
            frozen_params = record.get("frozen_params")
            frozen_params_sha = record.get("frozen_params_sha256")
            if frozen_params is not None:
                if not isinstance(frozen_params, dict):
                    raise ValueError("Trading order frozen_params 必須是 object")
                if str(frozen_params_sha or "") != canonical_json_sha256(frozen_params):
                    raise ValueError("Trading order frozen_params hash 不一致")
            if len(fill_trade_dates) > 1:
                raise ValueError("同一 Trading order 的 partial fills 不得跨交易日")
            for trade_date in fill_trade_dates:
                require_trading_date_after(
                    trade_date,
                    after=record.get("information_date"),
                    field_name="BUY fill trade_date",
                    after_field_name="information_date",
                )
            for fill in fills:
                if int(fill.get("net_buy_total_milli") or 0) <= 0:
                    raise ValueError("Trading BUY fill net_buy_total_milli 不合法")
            if filled_qty > 0 and frozen_params is None:
                raise ValueError("Trading 已成交 BUY order 必須持有 frozen_params")
            for field in (
                "plan_fingerprint", "information_date", "selected_params_sha256",
                "candidate_snapshot_sha256", "strategy_id", "param_selector",
            ):
                if not str(record.get(field) or ""):
                    raise ValueError(f"Trading BUY order 缺少 immutable binding: {field}")
        else:
            if purpose not in TRADING_SELL_ORDER_PURPOSES:
                raise ValueError("Trading SELL order purpose 不合法")
            for fill in fills:
                require_trading_date_after(
                    fill.get("trade_date"),
                    after=record.get("entry_trade_date"),
                    field_name="SELL fill trade_date",
                    after_field_name="entry_trade_date",
                )
                if int(fill.get("net_sell_total_milli") or 0) <= 0 or int(fill.get("allocated_cost_milli") or 0) < 0:
                    raise ValueError("Trading SELL fill exact-accounting payload 不合法")
                if int(fill.get("realized_pnl_milli") or 0) != int(fill.get("net_sell_total_milli") or 0) - int(fill.get("allocated_cost_milli") or 0):
                    raise ValueError("Trading SELL fill PnL reconciliation 不一致")
                if purpose == TRADING_ORDER_PURPOSE_INDICATOR_EXIT:
                    require_trading_date_after(
                        fill.get("trade_date"),
                        after=record.get("information_date"),
                        field_name="Indicator SELL fill trade_date",
                        after_field_name="signal information_date",
                    )
            if int(record.get("reserved_cost_milli") or 0) != 0:
                raise ValueError("Trading SELL 不得保留 BUY reserved cost")
            position_qty = int(record.get("position_qty_at_submission") or 0)
            if position_qty <= 0 or qty > position_qty:
                raise ValueError("Trading SELL qty 不得超過送單時實際持股")
            if not str(record.get("entry_order_id") or ""):
                raise ValueError("Trading SELL 缺少來源 entry_order_id")
            for field in ("plan_fingerprint", "position_plan_fingerprint"):
                if not str(record.get(field) or ""):
                    raise ValueError(f"Trading SELL 缺少 immutable binding: {field}")
            order_type = str(record.get("order_type") or "")
            trigger = record.get("trigger_price_milli")
            limit = record.get("limit_price_milli")
            if purpose in TRADING_PROTECTION_ORDER_PURPOSES:
                if not str(record.get("protection_plan_fingerprint") or ""):
                    raise ValueError("Trading protection SELL 缺少 protection plan fingerprint")
                if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP:
                    if order_type != TRADING_PROTECTION_ORDER_TYPE_STOP_MARKET or int(trigger or 0) <= 0 or limit is not None:
                        raise ValueError("Trading protection STOP order semantics 不合法")
                elif purpose == TRADING_ORDER_PURPOSE_PROTECTION_TP:
                    if order_type != TRADING_PROTECTION_ORDER_TYPE_LIMIT or int(limit or 0) <= 0 or trigger is not None:
                        raise ValueError("Trading protection TP order semantics 不合法")
                else:
                    if order_type != TRADING_PROTECTION_ORDER_TYPE_MARKET or trigger is not None or limit is not None:
                        raise ValueError("Trading STOP remainder forced exit 必須使用 MARKET 且不得有 trigger/limit")
                    for field in ("stop_forced_exit_key", "stop_trigger_order_id", "stop_trigger_fill_id", "stop_trigger_trade_date"):
                        if not str(record.get(field) or ""):
                            raise ValueError(f"Trading STOP remainder forced exit 缺少 immutable binding: {field}")
                    if int(record.get("stop_forced_exit_attempt") or 0) <= 0 or qty != position_qty:
                        raise ValueError("Trading STOP remainder forced exit attempt/qty 不合法")
                    for fill in fills:
                        require_trading_date_not_before(
                            fill.get("trade_date"),
                            earliest=record.get("stop_trigger_trade_date"),
                            field_name="STOP remainder fill trade_date",
                            earliest_field_name="STOP trigger trade_date",
                        )
                oco_confirmed = bool(record.get("broker_native_oco_confirmed"))
                oco_group = _normalize_optional_text(record.get("broker_oco_group_id"))
                if oco_confirmed != bool(oco_group):
                    raise ValueError("Trading protection OCO confirmed/group binding 不一致")
                if purpose == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER and (oco_confirmed or oco_group):
                    raise ValueError("Trading STOP remainder forced exit 不得標記為 OCO")
            else:
                for field in ("indicator_plan_fingerprint", "signal_key", "position_state_sha256", "frozen_params_sha256", "market_data_sha256"):
                    if not str(record.get(field) or ""):
                        raise ValueError(f"Trading Indicator SELL 缺少 immutable binding: {field}")
                if int(record.get("signal_attempt") or 0) <= 0 or not str(record.get("information_date") or ""):
                    raise ValueError("Trading Indicator SELL signal attempt/date 不合法")
                if order_type != TRADING_INDICATOR_ORDER_TYPE_MARKET or trigger is not None or limit is not None:
                    raise ValueError("Trading Indicator SELL 必須使用 MARKET 且不得有 trigger/limit")
                if qty != position_qty:
                    raise ValueError("Trading Indicator SELL 必須完整覆蓋送單時持股")
                if bool(record.get("broker_native_oco_confirmed")) or _normalize_optional_text(record.get("broker_oco_group_id")):
                    raise ValueError("Trading Indicator SELL 不得標記為 OCO")

        if status == TRADING_ORDER_STATUS_ORDERED:
            if filled_qty != 0 or record.get("cancelled_at") is not None or record.get("filled_at") is not None:
                raise ValueError("Trading ORDERED record 的 fill/cancel 狀態不一致")
        elif status == TRADING_ORDER_STATUS_PARTIAL:
            if not (0 < filled_qty < qty) or record.get("cancelled_at") is not None or record.get("filled_at") is not None:
                raise ValueError("Trading PARTIAL record 狀態不一致")
        elif status == TRADING_ORDER_STATUS_FILLED:
            if filled_qty != qty or not str(record.get("filled_at") or "") or record.get("cancelled_at") is not None:
                raise ValueError("Trading FILLED record 狀態不一致")
        elif status == TRADING_ORDER_STATUS_CANCELLED:
            if filled_qty >= qty or not str(record.get("cancelled_at") or "") or record.get("filled_at") is not None:
                raise ValueError("Trading CANCELLED record 狀態不一致")

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
        actual_hash = compute_event_hash(event)
        if event.get("event_hash") != actual_hash:
            raise ValueError("Trading order event hash 不一致")
        previous_hash = actual_hash


def build_trading_order_read_model(state: dict[str, Any]) -> dict[str, Any]:
    validate_trading_order_state(state)
    rows = []
    for record in sorted(
        state["orders"].values(),
        key=lambda row: (str(row.get("information_date") or row.get("entry_trade_date") or ""), str(row.get("ordered_at")), str(row.get("order_id"))),
        reverse=True,
    ):
        side = str(record.get("side") or "")
        limit_milli = record.get("limit_price_milli")
        trigger_milli = record.get("trigger_price_milli")
        rows.append(
            {
                "order_id": record["order_id"],
                "ticker": record["ticker"],
                "side": side,
                "purpose": record.get("purpose"),
                "order_type": record.get("order_type"),
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
                "limit_price": None if limit_milli is None else milli_to_price(int(limit_milli)),
                "trigger_price": None if trigger_milli is None else milli_to_price(int(trigger_milli)),
                "reserved_cost": None if side == TRADING_ORDER_SIDE_SELL else milli_to_money(int(record["reserved_cost_milli"])),
                "broker_order_id": record.get("broker_order_id"),
                "broker_native_oco_confirmed": bool(record.get("broker_native_oco_confirmed")),
                "broker_oco_group_id": record.get("broker_oco_group_id"),
                "entry_order_id": record.get("entry_order_id"),
                "information_date": record.get("information_date") or record.get("entry_trade_date"),
                "ordered_at": record["ordered_at"],
                "cancelled_at": record.get("cancelled_at"),
                "filled_at": record.get("filled_at"),
                "latest_fill_trade_date": max(
                    (str(fill.get("trade_date") or "") for fill in list(record.get("fills") or [])),
                    default="",
                ) or None,
                "plan_fingerprint": record["plan_fingerprint"],
            }
        )
    return {
        "schema_version": int(state["schema_version"]),
        "revision": int(state["revision"]),
        "active_order_count": sum(1 for row in rows if row["status"] in TRADING_ACTIVE_ORDER_STATUSES),
        "active_entry_order_count": sum(1 for row in rows if row["status"] in TRADING_ACTIVE_ORDER_STATUSES and row["side"] == TRADING_ORDER_SIDE_BUY),
        "active_sell_order_count": sum(1 for row in rows if row["status"] in TRADING_ACTIVE_ORDER_STATUSES and row["side"] == TRADING_ORDER_SIDE_SELL),
        "active_protection_order_count": sum(1 for row in rows if row["status"] in TRADING_ACTIVE_ORDER_STATUSES and row["purpose"] in TRADING_PROTECTION_ORDER_PURPOSES),
        "active_indicator_exit_order_count": sum(1 for row in rows if row["status"] in TRADING_ACTIVE_ORDER_STATUSES and row["purpose"] == TRADING_ORDER_PURPOSE_INDICATOR_EXIT),
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
    "TRADING_ORDER_SIDE_SELL",
    "TRADING_ORDER_PURPOSE_ENTRY",
    "TRADING_ORDER_PURPOSE_PROTECTION_STOP",
    "TRADING_ORDER_PURPOSE_PROTECTION_TP",
    "TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER",
    "TRADING_ORDER_PURPOSE_INDICATOR_EXIT",
    "TRADING_PROTECTION_ORDER_PURPOSES",
    "TRADING_SELL_ORDER_PURPOSES",
    "TRADING_PROTECTION_ORDER_TYPE_STOP_MARKET",
    "TRADING_PROTECTION_ORDER_TYPE_LIMIT",
    "TRADING_PROTECTION_ORDER_TYPE_MARKET",
    "TRADING_INDICATOR_ORDER_TYPE_MARKET",
    "TRADING_PROTECTION_ACTION_STOP_FULL",
    "TRADING_PROTECTION_ACTION_TP_HALF",
    "TRADING_PROTECTION_ACTION_STOP_REMAINDER_EXIT",
    "TRADING_PROTECTION_ACTION_SPECS",
    "get_trading_protection_action_spec",
    "build_empty_trading_order_state",
    "build_trading_proposal_key",
    "build_trading_protection_key",
    "build_trading_indicator_exit_key",
    "append_ordered_trading_proposal",
    "append_ordered_trading_protection_leg",
    "append_ordered_trading_protection_oco_group",
    "append_ordered_trading_indicator_exit",
    "record_trading_buy_order_fill",
    "record_trading_sell_order_fill",
    "cancel_ordered_trading_order",
    "active_trading_orders",
    "has_active_trading_orders",
    "active_trading_entry_orders",
    "active_trading_protection_orders",
    "active_trading_indicator_exit_orders",
    "active_trading_sell_orders",
    "validate_trading_order_state",
    "build_trading_order_read_model",
]
