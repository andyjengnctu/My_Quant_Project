"""Workbench-facing actual-account trade entry orchestration.

Workbench is a decision/accounting tool, not a broker OMS.  The user executes at
his broker and records the actual fill here.  Scanner-selected BUYs preserve the
strategy-management lineage; arbitrary BUYs remain manual account truth.  SELLs
are recorded directly against the selected broker inventory.
"""
from __future__ import annotations

from core.exact_accounting import price_to_milli
from core.file_integrity import canonical_json_sha256
from core.trading_account_state import (
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
    effective_trading_account_events,
)
from core.trading_identity import normalize_trading_date
from core.trading_order_state import TRADING_ACTIVE_ORDER_STATUSES, TRADING_ORDER_SIDE_SELL
from services.trading.account_state import (
    correct_trading_transaction,
    load_trading_account_state,
    record_manual_trading_buy,
    record_manual_trading_sell,
    record_strategy_trading_buy,
)
from services.trading.actual_fill_validation import validate_trading_actual_fill
from services.trading.order_state import get_trading_order_read_model
from services.trading.scanner_state import load_trading_candidate_snapshot_for_account
from services.trading.strategy_param_runtime import (
    build_trading_candidate_strategy_lineage,
    resolve_trading_candidate_frozen_params,
)


_BUY_MUTATIONS = {
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
}


def list_active_sell_orders_for_ticker(project_root, ticker: object) -> dict:
    """Legacy read-only helper retained for compatibility; the new UI does not use it."""
    ticker_key = str(ticker or "").strip().upper()
    snapshot = get_trading_order_read_model(project_root)
    rows = []
    for row in list(snapshot.get("orders") or []):
        if str(row.get("ticker") or "").strip().upper() != ticker_key:
            continue
        if str(row.get("side") or "").strip().upper() != TRADING_ORDER_SIDE_SELL:
            continue
        if str(row.get("status") or "") not in TRADING_ACTIVE_ORDER_STATUSES:
            continue
        rows.append(dict(row))
    return {"revision": snapshot.get("revision"), "orders": rows}


def _resolve_current_scanner_candidate(project_root, *, ticker: str, candidate_reference: dict) -> dict:
    """Resolve strategy semantics from the current snapshot, using UI data only as a stale guard."""
    reference = dict(candidate_reference or {})
    reference_ticker = str(reference.get("ticker") or "").strip().upper()
    if reference_ticker != ticker:
        raise ValueError("選取的 Scanner candidate 與成交股票不一致")
    reference_sha = canonical_json_sha256(reference)
    snapshot = load_trading_candidate_snapshot_for_account(project_root, require_current=True)
    matches = [
        dict(row)
        for row in list(snapshot.get("candidate_rows") or [])
        if str(row.get("ticker") or "").strip().upper() == ticker
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{ticker} 已不在目前 Scanner Pool；請重新整理交易中心")
    current = matches[0]
    if canonical_json_sha256(current) != reference_sha:
        raise RuntimeError(f"{ticker} Scanner candidate 已更新；請重新整理後再登錄成交")
    return current


def _scanner_buy_warnings_and_limits(*, candidate: dict, qty: int, price, trade_date) -> list[str]:
    warnings: list[str] = []
    seed = dict(candidate.get("execution_plan_seed") or {})
    information_date = normalize_trading_date(
        candidate.get("trade_date") or seed.get("trade_date"),
        field_name="candidate.trade_date",
        allow_none=False,
    )
    fill_date = normalize_trading_date(trade_date, field_name="trade_date", allow_none=False)
    if fill_date <= information_date:
        warnings.append(
            f"成交日 {fill_date} 早於或等於 Scanner 資訊日 {information_date}；"
            "若這是歷史補登可繼續，請確認日期無誤。"
        )

    limit_price = candidate.get("limit_price")
    if limit_price is None:
        limit_price = seed.get("limit_price")
    if limit_price is not None and int(price_to_milli(price)) > int(price_to_milli(limit_price)):
        raise ValueError(
            f"成交價 {price} 高於 Scanner 盤前買價上限 {limit_price}；"
            "不能標記為 Scanner 策略成交。若為自行交易，請取消 Scanner 選取後另行登錄。"
        )

    planned_qty = candidate.get("proj_qty")
    if planned_qty is not None and int(planned_qty) > 0 and int(qty) > int(planned_qty):
        raise ValueError(
            f"成交股數 {int(qty):,} 超過 Scanner 規劃股數 {int(planned_qty):,}；"
            "不能標記為 Scanner 策略成交。"
        )
    return warnings


def preview_trading_account_buy(
    project_root,
    *,
    ticker: object,
    qty: int,
    price,
    trade_date,
    candidate: dict | None = None,
) -> dict:
    """Validate one BUY without mutating account state; used by UI confirmation."""
    ticker_key = str(ticker or "").strip().upper()
    if not ticker_key:
        raise ValueError("股票代號必填")
    qty_int = int(qty)
    if qty_int <= 0:
        raise ValueError("成交股數必須 > 0")

    market_evidence = validate_trading_actual_fill(
        project_root,
        ticker=ticker_key,
        price=price,
        trade_date=trade_date,
    )
    current_candidate = None
    warnings: list[str] = []
    if candidate is not None:
        current_candidate = _resolve_current_scanner_candidate(
            project_root,
            ticker=ticker_key,
            candidate_reference=dict(candidate),
        )
        warnings.extend(
            _scanner_buy_warnings_and_limits(
                candidate=current_candidate,
                qty=qty_int,
                price=price,
                trade_date=trade_date,
            )
        )
    return {
        "route": "scanner_strategy_buy" if current_candidate is not None else "manual_account_buy",
        "candidate": current_candidate,
        "warnings": warnings,
        "market_evidence": market_evidence,
    }


def record_trading_account_buy(
    project_root,
    *,
    ticker: object,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int | None = None,
    candidate: dict | None = None,
):
    preview = preview_trading_account_buy(
        project_root,
        ticker=ticker,
        qty=qty,
        price=price,
        trade_date=trade_date,
        candidate=candidate,
    )
    ticker_key = str(ticker or "").strip().upper()
    candidate_row = preview.get("candidate")
    if candidate_row is not None:
        candidate_row = dict(candidate_row)
        params, member = resolve_trading_candidate_frozen_params(candidate_row)
        strategy_lineage = build_trading_candidate_strategy_lineage(candidate_row)
        # Strategy geometry is float-based while exact account ledgers accept
        # decimal-like inputs.  Normalize only at this strategy/account boundary
        # so UI Decimal input never leaks into stop/risk arithmetic.
        strategy_price = float(price)
        account = record_strategy_trading_buy(
            project_root,
            ticker=ticker_key,
            qty=int(qty),
            price=strategy_price,
            trade_date=trade_date,
            expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
            params=params,
            execution_plan_seed=dict(candidate_row.get("execution_plan_seed") or {}),
            strategy_lineage=strategy_lineage,
        )
        return {
            "route": "scanner_strategy_buy",
            "account": account,
            "strategy_member": dict(member or {}),
            "warnings": list(preview.get("warnings") or []),
            "market_evidence": dict(preview.get("market_evidence") or {}),
        }
    account = record_manual_trading_buy(
        project_root,
        ticker=ticker_key,
        qty=int(qty),
        price=price,
        trade_date=trade_date,
        expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
    )
    return {
        "route": "manual_account_buy",
        "account": account,
        "warnings": list(preview.get("warnings") or []),
        "market_evidence": dict(preview.get("market_evidence") or {}),
    }


def _effective_transaction_event(project_root, transaction_revision: int) -> dict:
    state = load_trading_account_state(project_root, required=True)
    target = int(transaction_revision)
    for event in effective_trading_account_events(state):
        if int(event.get("revision") or -1) == target:
            return dict(event)
    raise ValueError(f"找不到可修改的有效交易明細 revision={target}")


def correct_trading_account_transaction(
    project_root,
    *,
    transaction_revision: int,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int | None = None,
):
    """Correct an existing trade while preventing BUY edits from bypassing fill evidence checks."""
    event = _effective_transaction_event(project_root, int(transaction_revision))
    mutation = str(event.get("mutation_type") or "")
    details = dict(event.get("details") or {})
    if mutation in _BUY_MUTATIONS:
        validate_trading_actual_fill(
            project_root,
            ticker=details.get("ticker"),
            price=price,
            trade_date=trade_date,
        )
    return correct_trading_transaction(
        project_root,
        transaction_revision=int(transaction_revision),
        qty=int(qty),
        price=price,
        trade_date=trade_date,
        expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
    )


def record_trading_account_inventory_sell(
    project_root,
    *,
    ticker: object,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int | None = None,
    selected_order_id: str | None = None,
):
    """Record actual broker SELL directly; broker-order lifecycle is not required."""
    ticker_key = str(ticker or "").strip().upper()
    account = record_manual_trading_sell(
        project_root,
        ticker=ticker_key,
        qty=int(qty),
        price=price,
        trade_date=trade_date,
        expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
    )
    return {
        "route": "direct_account_sell",
        "account": account,
        "order": None,
        "refresh_errors": {},
    }


__all__ = [
    "correct_trading_account_transaction",
    "list_active_sell_orders_for_ticker",
    "preview_trading_account_buy",
    "record_trading_account_buy",
    "record_trading_account_inventory_sell",
]
