"""Explicit broker submission confirmation for Trading ENTRY BUY orders.

The persisted order state remains owned by :mod:`services.trading.order_state`.
This orchestration layer re-validates the canonical Operations safety decision
at the moment a proposed BUY is actually confirmed as submitted to the broker.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from core.file_integrity import compute_file_sha256
from core.params_io import params_to_json_dict
from core.portfolio_param_runtime import load_portfolio_param_source_from_json
from core.trading_identity import normalize_trading_ticker
from core.trading_order_state import active_trading_entry_orders, append_ordered_trading_proposal
from core.trading_policy import resolve_trading_selected_strategy_param_path
from services.trading.account_state import load_trading_account_state, resolve_trading_account_state_path
from services.trading.operations_status import (
    assert_trading_proposed_submission_allowed,
    build_trading_operations_status,
)
from services.trading.order_state import mutate_trading_order_state
from services.trading.proposed_order_state import (
    load_current_trading_proposed_order_plan,
    resolve_trading_proposed_orders_json_path,
)


def confirm_trading_order_submission(
    project_root,
    *,
    rank: int,
    ticker: str,
    expected_revision: int | None,
    broker_order_id: str | None = None,
    note: str | None = None,
) -> dict:
    """Record one actually-submitted proposed BUY after rechecking Trading safety."""

    root = Path(project_root).resolve()
    plan = load_current_trading_proposed_order_plan(root, require_current=True)
    operations = build_trading_operations_status(root)
    assert_trading_proposed_submission_allowed(operations)

    account_state = load_trading_account_state(root, required=True)
    if int(account_state["revision"]) != int(plan["account_revision"]):
        raise RuntimeError("Trading account 已與建議掛單使用的 revision 不一致；請重新產生建議掛單")

    matches = [
        row
        for row in list(plan.get("orders") or [])
        if int(row.get("rank") or 0) == int(rank)
        and normalize_trading_ticker(row.get("ticker")) == normalize_trading_ticker(ticker)
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

    account_path = resolve_trading_account_state_path(root)
    account_sha_before = compute_file_sha256(account_path)
    proposed_path = resolve_trading_proposed_orders_json_path(root)
    proposed_sha_before = compute_file_sha256(proposed_path)

    def source_guard() -> None:
        latest_operations = build_trading_operations_status(root)
        assert_trading_proposed_submission_allowed(latest_operations)
        if compute_file_sha256(account_path) != account_sha_before:
            raise RuntimeError("Trading account 在確認送單期間已變更")
        if compute_file_sha256(proposed_path) != proposed_sha_before:
            raise RuntimeError("Trading proposed-order artifact 在確認送單期間已變更")

    def mutator(state, timestamp, mutation_id):
        active = active_trading_entry_orders(state)
        active_plan_ids = {str(row.get("plan_fingerprint")) for row in active}
        if active_plan_ids and active_plan_ids != {str(plan["plan_fingerprint"])}:
            raise RuntimeError("Trading 尚有其他 plan 的 ORDERED 掛單；禁止混用不同盤前 allocation")
        return append_ordered_trading_proposal(
            state,
            order_id=uuid4().hex,
            proposal=proposal,
            plan=plan,
            timestamp=timestamp,
            mutation_id=mutation_id,
            broker_order_id=broker_order_id,
            note=note,
            frozen_params=frozen_params,
        )

    return mutate_trading_order_state(
        root,
        expected_revision=expected_revision,
        mutator=mutator,
        pre_persist_guard=source_guard,
    )


__all__ = ["confirm_trading_order_submission"]
