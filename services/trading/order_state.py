"""Persistent Trading broker-order state service."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.file_integrity import atomic_write_json, load_json_strict
from core.runtime_utils import get_taipei_now
from core.trading_state_paths import (
    resolve_trading_fill_transaction_path,
    resolve_trading_order_state_path as _resolve_trading_order_state_path,
)
from core.trading_order_state import (
    active_trading_orders,
    build_empty_trading_order_state,
    build_trading_order_read_model,
    cancel_ordered_trading_order,
    validate_trading_order_state,
)


class TradingOrderRevisionConflict(RuntimeError):
    pass


def resolve_trading_order_state_path(project_root) -> Path:
    return _resolve_trading_order_state_path(project_root)


def _assert_no_fill_transaction(project_root) -> None:
    tx_path = resolve_trading_fill_transaction_path(project_root)
    if tx_path.is_file():
        raise RuntimeError("Trading 尚有未完成 fill transaction；請先由 Workbench 重新整理以完成 recovery")


def _timestamp() -> str:
    return get_taipei_now().isoformat(timespec="seconds")


def _mutation_id() -> str:
    return uuid4().hex


def _read_state_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    state = load_json_strict(path)
    validate_trading_order_state(state)
    return state, source_sha


def load_trading_order_state(project_root, *, required: bool = False) -> dict[str, Any] | None:
    _assert_no_fill_transaction(project_root)
    path = resolve_trading_order_state_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Trading order state 尚未建立: {path}")
        return None
    state = load_json_strict(path)
    validate_trading_order_state(state)
    return state


def _load_or_initialize_for_mutation(project_root, *, expected_revision: int | None):
    _assert_no_fill_transaction(project_root)
    path = resolve_trading_order_state_path(project_root)
    if path.is_file():
        state, source_sha = _read_state_with_sha(path)
        current_revision = int(state["revision"])
        if expected_revision is None or int(expected_revision) != current_revision:
            raise TradingOrderRevisionConflict(
                f"Trading order revision 已變更：expected={expected_revision}, current={current_revision}"
            )
        return path, state, source_sha
    if expected_revision is not None:
        raise TradingOrderRevisionConflict(
            f"Trading order state 尚未建立，但 expected_revision={expected_revision}"
        )
    state = build_empty_trading_order_state(timestamp=_timestamp(), mutation_id=_mutation_id())
    return path, state, None


def _persist_mutation(
    project_root,
    path: Path,
    *,
    original_sha: str | None,
    original_revision: int,
    updated: dict[str, Any],
) -> dict[str, Any]:
    validate_trading_order_state(updated)
    if int(updated["revision"]) != int(original_revision) + 1:
        raise RuntimeError("Trading order mutation 必須恰好增加一個 revision")
    if original_sha is None:
        if path.exists():
            raise TradingOrderRevisionConflict("Trading order state 在初次寫入前已由其他流程建立")
    else:
        if not path.is_file():
            raise TradingOrderRevisionConflict("Trading order state 在寫入前被其他流程移除")
        latest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if latest_sha != original_sha:
            raise TradingOrderRevisionConflict("Trading order state 在寫入前已被其他流程修改")
    atomic_write_json(path, updated)
    return load_trading_order_state(project_root, required=True)


def mutate_trading_order_state(
    project_root,
    *,
    expected_revision: int | None,
    mutator,
    pre_persist_guard=None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    state_path, state, source_sha = _load_or_initialize_for_mutation(
        root,
        expected_revision=expected_revision,
    )
    original_revision = int(state["revision"])
    updated = mutator(state, _timestamp(), _mutation_id())
    if pre_persist_guard is not None:
        pre_persist_guard()
    return _persist_mutation(
        root,
        state_path,
        original_sha=source_sha,
        original_revision=original_revision,
        updated=updated,
    )


def confirm_trading_order_cancellation(
    project_root,
    *,
    order_id: str,
    expected_revision: int,
    note: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    state_path, state, source_sha = _load_or_initialize_for_mutation(
        root,
        expected_revision=expected_revision,
    )
    original_revision = int(state["revision"])
    updated = cancel_ordered_trading_order(
        state,
        order_id=order_id,
        timestamp=_timestamp(),
        mutation_id=_mutation_id(),
        note=note,
    )
    return _persist_mutation(
        root,
        state_path,
        original_sha=source_sha,
        original_revision=original_revision,
        updated=updated,
    )


def get_trading_order_read_model(project_root) -> dict[str, Any]:
    state = load_trading_order_state(project_root, required=False)
    if state is None:
        return {
            "schema_version": None,
            "revision": None,
            "active_order_count": 0,
            "order_count": 0,
            "orders": [],
            "updated_at": None,
        }
    return build_trading_order_read_model(state)


__all__ = [
    "TradingOrderRevisionConflict",
    "resolve_trading_order_state_path",
    "load_trading_order_state",
    "mutate_trading_order_state",
    "confirm_trading_order_cancellation",
    "get_trading_order_read_model",
]
