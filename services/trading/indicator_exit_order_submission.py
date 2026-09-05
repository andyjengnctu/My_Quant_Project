"""Explicit broker submission confirmation for rule-based Indicator MARKET SELL."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from core.file_integrity import compute_file_sha256
from core.trading_account_state import POSITION_SOURCE_STRATEGY_FILL
from core.trading_order_state import (
    TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
    TRADING_ORDER_STATUS_CANCELLED,
    active_trading_indicator_exit_orders,
    active_trading_protection_orders,
    append_ordered_trading_indicator_exit,
)
from services.trading.account_state import load_trading_account_state, resolve_trading_account_state_path
from services.trading.indicator_exit_planning import load_current_trading_indicator_exit_plan, resolve_trading_indicator_exit_plan_json_path
from services.trading.order_state import mutate_trading_order_state


def _normalize_ticker(value: object) -> str:
    text=str(value or "").strip().upper()
    if not text: raise ValueError("Trading ticker 不可為空")
    return text


def _find_exit_plan(plan: dict[str, Any], signal_key: str) -> dict[str, Any]:
    rows=[row for row in plan.get("exits") or [] if str(row.get("signal_key") or "")==str(signal_key or "")]
    if len(rows)!=1: raise ValueError("Trading Indicator SELL signal_key 不存在或不唯一")
    return rows[0]


def _load_position_truth(root: Path, exit_plan: dict[str, Any]) -> tuple[dict[str, Any], int]:
    account=load_trading_account_state(root, required=True)
    ticker=_normalize_ticker(exit_plan.get("ticker")); record=(account.get("positions") or {}).get(ticker)
    if not isinstance(record,dict) or record.get("source")!=POSITION_SOURCE_STRATEGY_FILL: raise RuntimeError("Indicator SELL 只允許 strategy_fill 持股")
    broker=record.get("broker") or {}; management=record.get("strategy_management") or {}; position=management.get("position_state") or {}
    broker_qty=int(broker.get("qty") or 0); strategy_qty=int(position.get("qty") or 0)
    if broker_qty<=0 or broker_qty!=strategy_qty or broker_qty!=int(exit_plan.get("qty") or 0): raise RuntimeError("Indicator SELL plan qty 已與目前持股不一致")
    if str(broker.get("entry_order_id") or "")!=str(exit_plan.get("entry_order_id") or ""): raise RuntimeError("Indicator SELL entry_order_id binding 已改變")
    return account, broker_qty


def _build_source_guard(root: Path):
    account_path=resolve_trading_account_state_path(root); plan_path=resolve_trading_indicator_exit_plan_json_path(root)
    account_sha=compute_file_sha256(account_path); plan_sha=compute_file_sha256(plan_path)
    def guard() -> None:
        if compute_file_sha256(account_path)!=account_sha: raise RuntimeError("Trading account 在 Indicator SELL 送單確認期間已變更")
        if compute_file_sha256(plan_path)!=plan_sha: raise RuntimeError("Trading Indicator SELL plan 在送單確認期間已變更")
    return guard


def confirm_trading_indicator_exit_submission(project_root: str | Path, *, signal_key: str, expected_order_revision: int, broker_order_id: str | None=None, note: str | None=None) -> dict[str, Any]:
    root=Path(project_root).resolve(); plan=load_current_trading_indicator_exit_plan(root); exit_plan=_find_exit_plan(plan,signal_key); _account,qty=_load_position_truth(root,exit_plan); ticker=_normalize_ticker(exit_plan["ticker"])
    guard=_build_source_guard(root)
    def mutator(state: dict[str, Any], timestamp: str, mutation_id: str) -> dict[str, Any]:
        if any(_normalize_ticker(x.get("ticker"))==ticker for x in active_trading_protection_orders(state)):
            raise RuntimeError("Trading 此持股仍有 active Stop/TP protection SELL；必須先確認券商取消，才可送 Indicator MARKET SELL")
        if any(_normalize_ticker(x.get("ticker"))==ticker for x in active_trading_indicator_exit_orders(state)):
            raise RuntimeError("Trading 此持股已有 active Indicator MARKET SELL；不得重複送單")
        related=[x for x in (state.get("orders") or {}).values() if str(x.get("purpose") or "")==TRADING_ORDER_PURPOSE_INDICATOR_EXIT and str(x.get("signal_key") or "")==str(signal_key)]
        if related and any(str(x.get("status") or "")!=TRADING_ORDER_STATUS_CANCELLED for x in related):
            raise RuntimeError("同一 Indicator signal 只有取消剩餘量後才可建立下一 attempt")
        if qty!=int(exit_plan.get("position_qty") or 0): raise RuntimeError("Indicator SELL 必須完整覆蓋目前持股")
        return append_ordered_trading_indicator_exit(state, order_id=uuid4().hex, exit_plan=exit_plan, plan=plan, timestamp=timestamp, mutation_id=mutation_id, broker_order_id=broker_order_id, note=note)
    return mutate_trading_order_state(root, expected_revision=expected_order_revision, mutator=mutator, pre_persist_guard=guard)


__all__=["confirm_trading_indicator_exit_submission"]
