"""Persistent Trading account-state service."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from core.file_integrity import atomic_write_json, load_json_strict
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.runtime_utils import get_taipei_now
from core.trading_account_state import (
    adopt_manual_trading_position,
    apply_confirmed_sell_fill,
    apply_confirmed_strategy_buy_fill,
    build_empty_trading_account_state,
    build_trading_account_read_model,
    correct_manual_trading_position,
    remove_manual_trading_position,
    set_trading_account_cash,
    validate_trading_account_state,
)

ACCOUNT_STATE_FILENAME = "account.json"


class TradingAccountRevisionConflict(RuntimeError):
    pass


def resolve_trading_account_state_path(project_root) -> Path:
    paths = resolve_runtime_domain_paths(project_root, domain=RUNTIME_DOMAIN_TRADING)
    if paths.state_root is None:
        raise RuntimeError("Trading runtime domain 缺少 state_root")
    return Path(paths.state_root) / ACCOUNT_STATE_FILENAME


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


def load_trading_account_state(project_root, *, required: bool = True) -> dict[str, Any] | None:
    path = resolve_trading_account_state_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Trading account state 尚未初始化: {path}")
        return None
    state = load_json_strict(path)
    validate_trading_account_state(state)
    return state


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


def _mutate_account(
    project_root,
    *,
    expected_revision: int,
    mutator: Callable[[dict[str, Any], str, str], dict[str, Any]],
) -> dict[str, Any]:
    path = resolve_trading_account_state_path(project_root)
    state, source_sha = _read_state_with_sha(path)
    current_revision = int(state["revision"])
    if int(expected_revision) != current_revision:
        raise TradingAccountRevisionConflict(
            f"Trading account revision 已變更：expected={expected_revision}, current={current_revision}"
        )

    updated = mutator(state, _timestamp(), _mutation_id())
    validate_trading_account_state(updated)
    if int(updated["revision"]) != current_revision + 1:
        raise RuntimeError("Trading account mutation 必須恰好增加一個 revision")

    latest_raw = path.read_bytes()
    latest_sha = hashlib.sha256(latest_raw).hexdigest()
    if latest_sha != source_sha:
        raise TradingAccountRevisionConflict("Trading account state 在寫入前已被其他流程修改")
    atomic_write_json(path, updated)
    return load_trading_account_state(project_root)


def set_trading_cash_balance(project_root, *, cash, expected_revision: int, note: str | None = None):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
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
        mutator=lambda state, timestamp, mutation_id: correct_manual_trading_position(
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
        mutator=lambda state, timestamp, mutation_id: remove_manual_trading_position(
            state,
            ticker=ticker,
            timestamp=timestamp,
            mutation_id=mutation_id,
            note=note,
        ),
    )


def confirm_trading_strategy_buy_fill(
    project_root,
    *,
    ticker,
    qty: int,
    buy_price,
    params,
    trade_date,
    expected_revision: int,
    init_sl=None,
    init_trail=None,
    target_price=None,
    limit_price=None,
    entry_atr=None,
    security_profile=None,
    entry_type: str = "normal",
):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        mutator=lambda state, timestamp, mutation_id: apply_confirmed_strategy_buy_fill(
            state,
            ticker=ticker,
            qty=qty,
            buy_price=buy_price,
            params=params,
            timestamp=timestamp,
            mutation_id=mutation_id,
            trade_date=trade_date,
            init_sl=init_sl,
            init_trail=init_trail,
            target_price=target_price,
            limit_price=limit_price,
            entry_atr=entry_atr,
            security_profile=security_profile,
            entry_type=entry_type,
        ),
    )


def confirm_trading_sell_fill(
    project_root,
    *,
    ticker,
    qty: int,
    exec_price,
    params,
    trade_date,
    expected_revision: int,
    event: str = "MANUAL_CONFIRMED_SELL",
):
    return _mutate_account(
        project_root,
        expected_revision=expected_revision,
        mutator=lambda state, timestamp, mutation_id: apply_confirmed_sell_fill(
            state,
            ticker=ticker,
            qty=qty,
            exec_price=exec_price,
            params=params,
            timestamp=timestamp,
            mutation_id=mutation_id,
            trade_date=trade_date,
            event=event,
        ),
    )


def get_trading_account_read_model(project_root) -> dict[str, Any]:
    return build_trading_account_read_model(load_trading_account_state(project_root))


__all__ = [
    "TradingAccountRevisionConflict",
    "resolve_trading_account_state_path",
    "load_trading_account_state",
    "initialize_trading_account_state",
    "set_trading_cash_balance",
    "adopt_existing_trading_position",
    "correct_existing_trading_position",
    "remove_existing_trading_position",
    "confirm_trading_strategy_buy_fill",
    "confirm_trading_sell_fill",
    "get_trading_account_read_model",
]
