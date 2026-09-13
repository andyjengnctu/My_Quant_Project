"""Persistent Trading account-state service."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from core.file_integrity import atomic_write_json, load_json_strict
from services.trading.state_lock import serialized_trading_state_mutation
from core.runtime_utils import get_taipei_now
from core.trading_state_paths import (
    resolve_trading_account_state_path as _resolve_trading_account_state_path,
    resolve_trading_fill_transaction_path,
    resolve_trading_order_state_path,
)
from core.trading_order_state import (
    active_trading_entry_orders,
    has_active_trading_orders,
    validate_trading_order_state,
)
from core.trading_account_state import (
    TRADING_ACCOUNT_STATE_FILENAME,
    adopt_manual_trading_position,
    apply_trading_strategy_management_rollforward,
    apply_manual_trading_buy_fill,
    apply_confirmed_strategy_buy_fill,
    apply_confirmed_sell_fill,
    apply_strategy_account_buy_correction_fill,
    build_empty_trading_account_state,
    build_trading_account_read_model,
    correct_manual_trading_position,
    correct_trading_position_broker_truth,
    remove_manual_trading_position,
    remove_trading_position_broker_truth,
    rebuild_trading_account_economics,
    replace_trading_transaction,
    set_trading_account_cash,
    void_manual_trading_transaction,
    validate_trading_account_state,
)

from services.trading.accounting_policy import build_standalone_trading_accounting_params, overlay_trading_accounting_params

ACCOUNT_STATE_FILENAME = TRADING_ACCOUNT_STATE_FILENAME


class TradingAccountRevisionConflict(RuntimeError):
    pass


def resolve_trading_account_state_path(project_root) -> Path:
    return _resolve_trading_account_state_path(project_root)


def _timestamp() -> str:
    return get_taipei_now().isoformat(timespec="seconds")


def _mutation_id() -> str:
    return uuid4().hex


def _read_state_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    state = load_json_strict(path)
    validate_trading_account_state(state)
    return state, source_sha


def _read_order_guard_sha(project_root, *, allow_active_protection_orders: bool = False) -> str | None:
    fill_tx_path = resolve_trading_fill_transaction_path(project_root)
    if fill_tx_path.is_file():
        raise RuntimeError("Trading 尚有未完成 fill transaction；請先由 Workbench 重新整理以完成 recovery")
    order_path = resolve_trading_order_state_path(project_root)
    if not order_path.is_file():
        return None
    raw = order_path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    state = load_json_strict(order_path)
    validate_trading_order_state(state)
    if allow_active_protection_orders:
        if active_trading_entry_orders(state):
            raise RuntimeError("Trading 尚有 active ENTRY BUY；完成成交／取消 reconciliation 前禁止推進持股管理狀態")
    elif has_active_trading_orders(state):
        raise RuntimeError("Trading 尚有 ORDERED pending orders；完成成交／取消 reconciliation 前禁止修改 account state")
    return source_sha


def _assert_order_guard_unchanged(project_root, expected_sha: str | None) -> None:
    order_path = resolve_trading_order_state_path(project_root)
    if expected_sha is None:
        if order_path.exists():
            raise TradingAccountRevisionConflict("Trading order state 在 account mutation 期間已建立")
        return
    if not order_path.is_file():
        raise TradingAccountRevisionConflict("Trading order state 在 account mutation 期間被移除")
    current_sha = hashlib.sha256(order_path.read_bytes()).hexdigest()
    if current_sha != expected_sha:
        raise TradingAccountRevisionConflict("Trading order state 在 account mutation 期間已變更")


def load_trading_account_state(project_root, *, required: bool = True) -> dict[str, Any] | None:
    fill_tx_path = resolve_trading_fill_transaction_path(project_root)
    if fill_tx_path.is_file():
        raise RuntimeError("Trading 尚有未完成 fill transaction；必須先完成 recovery 才能讀取 account state")
    path = resolve_trading_account_state_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Trading account state 尚未初始化: {path}")
        return None
    state = load_json_strict(path)
    validate_trading_account_state(state)
    return state


@serialized_trading_state_mutation
def initialize_trading_account_state(project_root, *, cash=None) -> dict[str, Any]:
    path = resolve_trading_account_state_path(project_root)
    if path.exists():
        raise FileExistsError(f"Trading account state 已存在: {path}")
    state = build_empty_trading_account_state(
        timestamp=_timestamp(),
        mutation_id=_mutation_id(),
        cash=cash,
    )
    atomic_write_json(path, state)
    return load_trading_account_state(project_root)


@serialized_trading_state_mutation
def _mutate_account(
    project_root,
    *,
    expected_revision: int | None,
    mutator: Callable[[dict[str, Any], str, str], dict[str, Any]],
    allow_active_protection_orders: bool = False,
    expected_revision_increment: int | None = 1,
    guard_orders: bool = True,
    accounting_params=None,
) -> dict[str, Any]:
    order_guard_sha = (
        _read_order_guard_sha(project_root, allow_active_protection_orders=allow_active_protection_orders)
        if guard_orders else None
    )
    path = resolve_trading_account_state_path(project_root)
    state, source_sha = _read_state_with_sha(path)
    current_revision = int(state["revision"])
    if expected_revision is not None and int(expected_revision) != current_revision:
        raise TradingAccountRevisionConflict(
            f"Trading account revision 已變更：expected={expected_revision}, current={current_revision}"
        )
    if accounting_params is not None:
        state = rebuild_trading_account_economics(state, accounting_params=accounting_params)

    updated = mutator(state, _timestamp(), _mutation_id())
    validate_trading_account_state(updated)
    if expected_revision_increment is None:
        if int(updated["revision"]) <= current_revision:
            raise RuntimeError("Trading account mutation 必須至少增加 1 個 revision")
    else:
        increment = int(expected_revision_increment)
        if increment <= 0:
            raise ValueError("expected_revision_increment 必須 > 0")
        if int(updated["revision"]) != current_revision + increment:
            raise RuntimeError(f"Trading account mutation 必須恰好增加 {increment} 個 revision")

    latest_raw = path.read_bytes()
    latest_sha = hashlib.sha256(latest_raw).hexdigest()
    if latest_sha != source_sha:
        raise TradingAccountRevisionConflict("Trading account state 在寫入前已被其他流程修改")
    if guard_orders:
        _assert_order_guard_unchanged(project_root, order_guard_sha)
    atomic_write_json(path, updated)
    return load_trading_account_state(project_root)


def set_trading_cash_balance(project_root, *, cash, expected_revision: int, note: str | None = None):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        mutator=lambda state, timestamp, mutation_id: set_trading_account_cash(
            state,
            cash=cash,
            timestamp=timestamp,
            mutation_id=mutation_id,
            note=note,
        ),
    )


def adopt_existing_trading_position(
    project_root,
    *,
    ticker,
    qty: int,
    cost_basis_total,
    expected_revision: int,
    entry_date=None,
    note: str | None = None,
):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        mutator=lambda state, timestamp, mutation_id: adopt_manual_trading_position(
            state,
            ticker=ticker,
            qty=qty,
            cost_basis_total=cost_basis_total,
            timestamp=timestamp,
            mutation_id=mutation_id,
            entry_date=entry_date,
            note=note,
        ),
    )


def correct_existing_trading_position(
    project_root,
    *,
    ticker,
    qty: int,
    cost_basis_total,
    expected_revision: int,
    entry_date=None,
    note: str | None = None,
):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        mutator=lambda state, timestamp, mutation_id: correct_trading_position_broker_truth(
            state,
            ticker=ticker,
            qty=qty,
            cost_basis_total=cost_basis_total,
            timestamp=timestamp,
            mutation_id=mutation_id,
            entry_date=entry_date,
            note=note,
        ),
    )


def remove_existing_trading_position(
    project_root,
    *,
    ticker,
    expected_revision: int,
    note: str | None = None,
):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        mutator=lambda state, timestamp, mutation_id: remove_trading_position_broker_truth(
            state,
            ticker=ticker,
            timestamp=timestamp,
            mutation_id=mutation_id,
            note=note,
        ),
    )



def record_manual_trading_buy(
    project_root,
    *,
    ticker,
    qty: int,
    price,
    trade_date,
    expected_revision: int,
):
    params = build_standalone_trading_accounting_params()
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        mutator=lambda state, timestamp, mutation_id: apply_manual_trading_buy_fill(
            state,
            ticker=ticker,
            qty=qty,
            buy_price=price,
            params=params,
            timestamp=timestamp,
            mutation_id=mutation_id,
            trade_date=trade_date,
        ),
    )


def record_manual_trading_sell(
    project_root,
    *,
    ticker,
    qty: int,
    price,
    trade_date,
    expected_revision: int,
):
    params = build_standalone_trading_accounting_params()
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        mutator=lambda state, timestamp, mutation_id: apply_confirmed_sell_fill(
            state,
            ticker=ticker,
            qty=qty,
            exec_price=price,
            params=params,
            timestamp=timestamp,
            mutation_id=mutation_id,
            trade_date=trade_date,
            event="MANUAL_ACCOUNT_SELL",
        ),
    )




def record_strategy_trading_buy(
    project_root,
    *,
    ticker,
    qty: int,
    price,
    trade_date,
    expected_revision: int,
    params,
    execution_plan_seed: dict[str, Any] | None = None,
    strategy_lineage: dict[str, Any] | None = None,
):
    seed = dict(execution_plan_seed or {})
    accounting_params = overlay_trading_accounting_params(params)
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=accounting_params,
        mutator=lambda state, timestamp, mutation_id: apply_confirmed_strategy_buy_fill(
            state,
            ticker=ticker,
            qty=qty,
            buy_price=price,
            params=accounting_params,
            timestamp=timestamp,
            mutation_id=mutation_id,
            trade_date=trade_date,
            init_sl=seed.get("init_sl"),
            init_trail=seed.get("init_trail"),
            target_price=seed.get("target_price"),
            limit_price=seed.get("limit_price"),
            entry_atr=seed.get("entry_atr"),
            security_profile=seed.get("security_profile"),
            entry_type=str(seed.get("entry_type") or "normal"),
            entry_order_id=None,
            strategy_lineage=strategy_lineage,
        ),
    )



def delete_trading_transaction(
    project_root,
    *,
    transaction_revision: int,
    expected_revision: int | None,
    note: str | None = None,
):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        expected_revision_increment=None,
        mutator=lambda state, timestamp, mutation_id: void_manual_trading_transaction(
            state,
            target_revision=transaction_revision,
            timestamp=timestamp,
            mutation_id=mutation_id,
            note=note or "Workbench accounting center delete transaction",
            params=build_standalone_trading_accounting_params(),
        ),
    )


def correct_trading_transaction(
    project_root,
    *,
    transaction_revision: int,
    qty: int,
    price,
    trade_date,
    expected_revision: int | None,
):
    params = build_standalone_trading_accounting_params()
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        expected_revision_increment=None,
        mutator=lambda state, timestamp, mutation_id: replace_trading_transaction(
            state,
            target_revision=transaction_revision,
            qty=qty,
            price=price,
            trade_date=trade_date,
            params=params,
            timestamp=timestamp,
            mutation_id=mutation_id,
        ),
    )


def delete_manual_trading_transaction(*args, **kwargs):
    """Backward-compatible alias; strategy and manual trades are both supported."""
    return delete_trading_transaction(*args, **kwargs)


def correct_manual_trading_transaction(*args, **kwargs):
    """Backward-compatible alias; strategy and manual trades are both supported."""
    return correct_trading_transaction(*args, **kwargs)


def rollforward_trading_strategy_management(
    project_root,
    *,
    updates: dict[str, dict[str, Any]],
    expected_revision: int,
):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        guard_orders=False,
        accounting_params=build_standalone_trading_accounting_params(),
        mutator=lambda state, timestamp, mutation_id: apply_trading_strategy_management_rollforward(
            state,
            updates=updates,
            timestamp=timestamp,
            mutation_id=mutation_id,
        ),
    )

def get_trading_account_read_model(project_root) -> dict[str, Any]:
    state = load_trading_account_state(project_root)
    projected = rebuild_trading_account_economics(
        state, accounting_params=build_standalone_trading_accounting_params()
    )
    return build_trading_account_read_model(projected)


__all__ = [
    "TradingAccountRevisionConflict",
    "resolve_trading_account_state_path",
    "load_trading_account_state",
    "initialize_trading_account_state",
    "set_trading_cash_balance",
    "adopt_existing_trading_position",
    "correct_existing_trading_position",
    "remove_existing_trading_position",
    "record_manual_trading_buy",
    "record_strategy_trading_buy",
    "record_manual_trading_sell",
    "delete_trading_transaction",
    "correct_trading_transaction",
    "delete_manual_trading_transaction",
    "correct_manual_trading_transaction",
    "rollforward_trading_strategy_management",
    "get_trading_account_read_model",
]
