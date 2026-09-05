"""Explicit broker submission of logical Trading protection legs.

The canonical broker-order truth remains ``state/trading/orders.json``.  This
service only records SELL protection orders after the user confirms that the
orders were actually submitted at the broker.  It never infers submission from
protection-plan existence and never mutates the Trading account.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from core.file_integrity import compute_file_sha256
from core.trading_account_state import POSITION_SOURCE_STRATEGY_FILL
from core.trading_order_state import (
    TRADING_ORDER_PURPOSE_PROTECTION_STOP,
    TRADING_ORDER_PURPOSE_PROTECTION_TP,
    active_trading_indicator_exit_orders,
    active_trading_protection_orders,
    append_ordered_trading_protection_leg,
    append_ordered_trading_protection_oco_group,
)
from services.trading.account_state import load_trading_account_state, resolve_trading_account_state_path
from services.trading.order_state import (
    mutate_trading_order_state,
)
from services.trading.protection_planning import (
    PROTECTION_STOP_ACTION,
    PROTECTION_TP_ACTION,
    load_current_trading_protection_plan,
    resolve_trading_protection_plan_json_path,
)


def _normalize_ticker(value: object) -> str:
    ticker = str(value or "").strip().upper()
    if not ticker:
        raise ValueError("Trading protection ticker 不可為空")
    return ticker


def _find_position_plan(plan: dict[str, Any], ticker: str) -> dict[str, Any]:
    ticker_key = _normalize_ticker(ticker)
    matches = [row for row in list(plan.get("positions") or []) if _normalize_ticker(row.get("ticker")) == ticker_key]
    if len(matches) != 1:
        raise ValueError(f"無法唯一定位 Trading protection position plan: {ticker_key}")
    return dict(matches[0])


def _find_leg(position_plan: dict[str, Any], action: str) -> dict[str, Any]:
    matches = [row for row in list(position_plan.get("legs") or []) if str(row.get("action") or "") == str(action)]
    if len(matches) != 1:
        raise ValueError(f"Trading protection plan 缺少唯一 {action} leg")
    return dict(matches[0])


def _load_position_truth(root: Path, position_plan: dict[str, Any]) -> tuple[dict[str, Any], int]:
    account = load_trading_account_state(root, required=True)
    ticker = _normalize_ticker(position_plan.get("ticker"))
    record = (account.get("positions") or {}).get(ticker)
    if not isinstance(record, dict) or record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
        raise RuntimeError(f"Trading protection SELL 只允許 strategy_fill 持股: {ticker}")
    broker = record.get("broker") or {}
    qty = int(broker.get("qty") or 0)
    if qty <= 0 or qty != int(position_plan.get("position_qty") or 0):
        raise RuntimeError(f"Trading protection plan 持股 qty 已變更，請先重新建立 plan: {ticker}")
    if str(broker.get("entry_order_id") or "") != str(position_plan.get("entry_order_id") or ""):
        raise RuntimeError(f"Trading protection plan entry order binding 已變更: {ticker}")
    return account, qty


def _effective_protection_exposure(rows: list[dict[str, Any]], *, ticker: str) -> int:
    ticker_key = _normalize_ticker(ticker)
    plain_total = 0
    oco_groups: dict[str, list[int]] = {}
    for row in rows:
        if _normalize_ticker(row.get("ticker")) != ticker_key:
            continue
        qty = int(row.get("remaining_qty") or row.get("qty") or 0)
        if qty <= 0:
            continue
        if bool(row.get("broker_native_oco_confirmed")):
            group = str(row.get("broker_oco_group_id") or "").strip()
            if not group:
                raise RuntimeError("Trading active OCO protection order 缺少 broker group ID")
            oco_groups.setdefault(group, []).append(qty)
        else:
            plain_total += qty
    return plain_total + sum(max(values) for values in oco_groups.values())


def _assert_no_active_duplicate(rows: list[dict[str, Any]], *, ticker: str, purpose: str) -> None:
    ticker_key = _normalize_ticker(ticker)
    for row in rows:
        if _normalize_ticker(row.get("ticker")) == ticker_key and str(row.get("purpose") or "") == purpose:
            raise RuntimeError(f"Trading {ticker_key} 已有 active {purpose} SELL order；請先確認券商取消舊單")


def _build_source_guard(root: Path):
    account_path = resolve_trading_account_state_path(root)
    protection_path = resolve_trading_protection_plan_json_path(root)
    account_sha = compute_file_sha256(account_path)
    protection_sha = compute_file_sha256(protection_path)

    def guard() -> None:
        if compute_file_sha256(account_path) != account_sha:
            raise RuntimeError("Trading account 在確認 protection SELL 送單期間已變更")
        if compute_file_sha256(protection_path) != protection_sha:
            raise RuntimeError("Trading protection plan 在確認送單期間已變更")

    return guard


def confirm_trading_protection_leg_submission(
    project_root,
    *,
    ticker: str,
    action: str,
    expected_order_revision: int,
    broker_order_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    plan = load_current_trading_protection_plan(root)
    position_plan = _find_position_plan(plan, ticker)
    leg = _find_leg(position_plan, action)
    _account, held_qty = _load_position_truth(root, position_plan)
    purpose = (
        TRADING_ORDER_PURPOSE_PROTECTION_STOP
        if action == PROTECTION_STOP_ACTION
        else TRADING_ORDER_PURPOSE_PROTECTION_TP
        if action == PROTECTION_TP_ACTION
        else None
    )
    if purpose is None:
        raise ValueError(f"Trading protection action 不合法: {action}")
    source_guard = _build_source_guard(root)

    def mutate(state, timestamp, mutation_id):
        if any(_normalize_ticker(row.get("ticker")) == _normalize_ticker(ticker) for row in active_trading_indicator_exit_orders(state)):
            raise RuntimeError("Trading 此持股已有 active Indicator MARKET SELL；完成成交／取消 reconciliation 前不得再掛 protection SELL")
        active = active_trading_protection_orders(state)
        _assert_no_active_duplicate(active, ticker=ticker, purpose=purpose)
        exposure_after = _effective_protection_exposure(active, ticker=ticker) + int(leg.get("qty") or 0)
        if exposure_after > held_qty:
            raise RuntimeError(
                f"Trading protection SELL 會超過實際持股：effective exposure={exposure_after}, held={held_qty}。"
                "若券商實際提供 native OCO/互斥群組，請改用明確 OCO 確認流程。"
            )
        return append_ordered_trading_protection_leg(
            state,
            order_id=uuid4().hex,
            plan=plan,
            position_plan=position_plan,
            leg=leg,
            account_revision=int(_account["revision"]),
            timestamp=timestamp,
            mutation_id=mutation_id,
            broker_order_id=broker_order_id,
            note=note,
        )

    updated = mutate_trading_order_state(
        root,
        expected_revision=int(expected_order_revision),
        mutator=mutate,
        pre_persist_guard=source_guard,
    )
    return updated


def confirm_trading_protection_oco_submission(
    project_root,
    *,
    ticker: str,
    expected_order_revision: int,
    broker_oco_group_id: str,
    stop_broker_order_id: str | None = None,
    tp_broker_order_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    group = str(broker_oco_group_id or "").strip()
    if not group:
        raise ValueError("必須輸入券商實際 OCO/互斥群組 ID，系統不會自行假設券商支援 OCO")
    plan = load_current_trading_protection_plan(root)
    position_plan = _find_position_plan(plan, ticker)
    stop_leg = _find_leg(position_plan, PROTECTION_STOP_ACTION)
    tp_leg = _find_leg(position_plan, PROTECTION_TP_ACTION)
    account, held_qty = _load_position_truth(root, position_plan)
    if max(int(stop_leg.get("qty") or 0), int(tp_leg.get("qty") or 0)) > held_qty:
        raise RuntimeError("Trading OCO protection leg qty 不得超過實際持股")
    source_guard = _build_source_guard(root)

    def mutate(state, timestamp, mutation_id):
        if any(_normalize_ticker(row.get("ticker")) == _normalize_ticker(ticker) for row in active_trading_indicator_exit_orders(state)):
            raise RuntimeError("Trading 此持股已有 active Indicator MARKET SELL；完成成交／取消 reconciliation 前不得再掛 protection SELL")
        active = active_trading_protection_orders(state)
        if any(_normalize_ticker(row.get("ticker")) == _normalize_ticker(ticker) for row in active):
            raise RuntimeError("Trading 此持股已有 active protection SELL；建立新 OCO 前必須先確認券商取消舊單")
        return append_ordered_trading_protection_oco_group(
            state,
            stop_order_id=uuid4().hex,
            tp_order_id=uuid4().hex,
            plan=plan,
            position_plan=position_plan,
            stop_leg=stop_leg,
            tp_leg=tp_leg,
            account_revision=int(account["revision"]),
            timestamp=timestamp,
            mutation_id=mutation_id,
            broker_oco_group_id=group,
            stop_broker_order_id=stop_broker_order_id,
            tp_broker_order_id=tp_broker_order_id,
            note=note,
        )

    return mutate_trading_order_state(
        root,
        expected_revision=int(expected_order_revision),
        mutator=mutate,
        pre_persist_guard=source_guard,
    )


__all__ = [
    "confirm_trading_protection_leg_submission",
    "confirm_trading_protection_oco_submission",
]
