"""Persistent Trading broker-order state service."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.file_integrity import atomic_write_json, compute_file_sha256, load_json_strict
from core.params_io import params_to_json_dict
from core.portfolio_param_runtime import load_portfolio_param_source_from_json
from core.runtime_utils import get_taipei_now
from core.trading_state_paths import (
    resolve_trading_fill_transaction_path,
    resolve_trading_order_state_path as _resolve_trading_order_state_path,
)
from core.trading_policy import resolve_trading_selected_strategy_param_path
from core.trading_identity import normalize_trading_ticker
from core.trading_order_state import (
    active_trading_orders,
    active_trading_entry_orders,
    append_ordered_trading_proposal,
    build_empty_trading_order_state,
    build_trading_order_read_model,
    cancel_ordered_trading_order,
    validate_trading_order_state,
)
from services.trading.account_state import load_trading_account_state, resolve_trading_account_state_path
from services.trading.order_planning import (
    load_current_trading_proposed_order_plan,
    resolve_trading_proposed_orders_json_path,
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


def confirm_trading_order_submission(
    project_root,
    *,
    rank: int,
    ticker: str,
    expected_revision: int | None,
    broker_order_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    plan = load_current_trading_proposed_order_plan(root, require_current=True)
    account_state = load_trading_account_state(root, required=True)
    if int(account_state["revision"]) != int(plan["account_revision"]):
        raise RuntimeError("Trading account 已與建議掛單使用的 revision 不一致；請重新產生建議掛單")

    matches = [
        row
        for row in list(plan.get("orders") or [])
        if int(row.get("rank") or 0) == int(rank) and normalize_trading_ticker(row.get("ticker")) == normalize_trading_ticker(ticker)
    ]
    if len(matches) != 1:
        raise ValueError(f"無法唯一定位 Trading proposed order: rank={rank}, ticker={ticker}")
    proposal = dict(matches[0])

    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if not selected_path.is_file():
        raise FileNotFoundError("Trading selected strategy params 已不存在；禁止建立無法重現的 ORDERED")
    if compute_file_sha256(selected_path) != str(plan.get("selected_params_sha256") or ""):
        raise RuntimeError("Trading selected strategy params 已與 proposed plan 不一致；請重新建立建議掛單")
    param_source = load_portfolio_param_source_from_json(selected_path)
    if int(param_source.get("member_count") or 0) != 1:
        raise RuntimeError("Trading ORDERED 只能凍結單一參數 member")
    frozen_params = params_to_json_dict(param_source["primary_params"])

    state_path, state, source_sha = _load_or_initialize_for_mutation(
        root,
        expected_revision=expected_revision,
    )
    active = active_trading_entry_orders(state)
    active_plan_ids = {str(row.get("plan_fingerprint")) for row in active}
    if active_plan_ids and active_plan_ids != {str(plan["plan_fingerprint"])}:
        raise RuntimeError("Trading 尚有其他 plan 的 ORDERED 掛單；禁止混用不同盤前 allocation")

    original_revision = int(state["revision"])
    account_path = resolve_trading_account_state_path(root)
    account_sha_before = compute_file_sha256(account_path)
    proposed_path = resolve_trading_proposed_orders_json_path(root)
    proposed_sha_before = compute_file_sha256(proposed_path)
    updated = append_ordered_trading_proposal(
        state,
        order_id=uuid4().hex,
        proposal=proposal,
        plan=plan,
        timestamp=_timestamp(),
        mutation_id=_mutation_id(),
        broker_order_id=broker_order_id,
        note=note,
        frozen_params=frozen_params,
    )
    if compute_file_sha256(account_path) != account_sha_before:
        raise TradingOrderRevisionConflict("Trading account 在確認送單期間已變更")
    if compute_file_sha256(proposed_path) != proposed_sha_before:
        raise TradingOrderRevisionConflict("Trading proposed-order artifact 在確認送單期間已變更")
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
    "confirm_trading_order_submission",
    "confirm_trading_order_cancellation",
    "get_trading_order_read_model",
]
