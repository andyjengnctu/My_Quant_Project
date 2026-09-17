"""Workbench-facing actual-account trade entry orchestration.

Workbench is a decision/accounting tool, not a broker OMS.  The user executes at
his broker and records the actual fill here.  Scanner-selected BUYs preserve the
strategy-management lineage; direct manual BUYs are explicit manual-managed
positions.  SELLs are recorded directly against the selected broker inventory.
"""
from __future__ import annotations

import pandas as pd

from core.data_utils import get_required_min_rows
from core.exact_accounting import infer_security_profile, price_to_milli
from core.params_io import build_params_from_mapping
from core.price_utils import calc_frozen_target_price, calc_initial_stop_from_reference, calc_initial_trailing_stop_from_reference
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.file_integrity import canonical_json_sha256
from core.trading_account_state import (
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_MANUAL_MANAGED_BUY,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
    effective_trading_account_events,
)
from core.trading_identity import normalize_trading_date
from core.trading_order_state import TRADING_ACTIVE_ORDER_STATUSES, TRADING_ORDER_SIDE_SELL
from services.trading.account_state import (
    correct_trading_transaction,
    load_trading_account_state,
    record_managed_manual_trading_buy,
    record_manual_trading_sell,
    record_strategy_trading_buy,
)
from services.trading.actual_fill_validation import validate_trading_actual_fill
from services.trading.market_data_consumer import load_trading_v2_sanitized_ohlcv_frame, open_trading_v2_consumer_view
from services.trading.order_state import get_trading_order_read_model
from services.trading.scanner_state import load_trading_candidate_snapshot_for_account, load_trading_scanner_runtime
from services.trading.strategy_param_runtime import (
    build_trading_candidate_strategy_lineage,
    build_trading_manual_management_lineage,
    resolve_trading_candidate_frozen_params,
)


_BUY_MUTATIONS = {
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_MANUAL_MANAGED_BUY,
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


def resolve_current_trading_scanner_candidate(project_root, *, ticker: str, candidate_reference: dict) -> dict:
    """Public stale-guarded Scanner candidate resolver for Trading entry workflows."""
    return _resolve_current_scanner_candidate(
        project_root, ticker=ticker, candidate_reference=candidate_reference
    )


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


def _build_direct_manual_managed_entry_context(project_root, *, ticker: str, trade_date: object) -> dict:
    """Freeze current primary Params while deriving historical geometry only from pre-entry data."""
    runtime = load_trading_scanner_runtime(project_root)
    params = runtime["params"]
    information_date = normalize_trading_date(runtime["latest_data_date"], field_name="information_date", allow_none=False)
    fill_date = normalize_trading_date(trade_date, field_name="trade_date", allow_none=False)
    if information_date < fill_date:
        raise RuntimeError(
            f"目前 Trading 資訊日 {information_date} 早於成交日 {fill_date}；請先更新 Trading 資料後再補登。"
        )
    view = open_trading_v2_consumer_view(project_root)
    frame = load_trading_v2_sanitized_ohlcv_frame(
        view,
        ticker=ticker,
        through_date=fill_date,
        min_rows=get_required_min_rows(params),
    )
    prior = frame.loc[frame.index < pd.Timestamp(fill_date)].copy()
    required_rows = int(get_required_min_rows(params))
    if len(prior) < required_rows:
        raise RuntimeError(
            f"{ticker} 買入日前歷史資料不足；需要至少 {required_rows} 筆，實際 {len(prior)} 筆"
        )
    # D1: derive historical manual-management geometry from pre-entry bars only.
    # The fill-date bar must not participate even indirectly in ATR/indicator state.
    signals = generate_signals(prior, params, ticker=ticker)
    atr_values, _buy, _sell, _limits = unpack_precomputed_signals(signals)
    prior_atr = float(atr_values[-1])
    prior_close = float(prior["Close"].iloc[-1])
    if prior_atr <= 0 or prior_close <= 0:
        raise RuntimeError(f"{ticker} 前一交易日 Close/ATR 不合法")
    security_profile = infer_security_profile(ticker)
    reference_stop = calc_initial_stop_from_reference(
        prior_close, prior_atr, params, ticker=ticker, security_profile=security_profile
    )
    reference_trail = calc_initial_trailing_stop_from_reference(
        prior_close, prior_atr, params, ticker=ticker, security_profile=security_profile
    )
    target_price = calc_frozen_target_price(
        prior_close, reference_stop, ticker=ticker, security_profile=security_profile
    )
    seed = {
        "entry_type": "manual_direct_backfill",
        "entry_atr": prior_atr,
        "init_sl": reference_stop,
        "init_trail": reference_trail,
        "target_price": target_price,
        "target_reference_price": prior_close,
        "limit_price": None,
        "security_profile": security_profile,
        "trade_date": fill_date,
        "reference_market_date": pd.Timestamp(prior.index[-1]).strftime("%Y-%m-%d"),
        "reference_close": prior_close,
    }
    lineage = build_trading_manual_management_lineage(
        params=params,
        execution_plan_seed=seed,
        information_date=information_date,
        origin="manual_direct_backfill",
        target_reference_close=prior_close,
    )
    return {
        "params": params,
        "information_date": information_date,
        "management_start_date": information_date,
        "execution_plan_seed": seed,
        "management_lineage": lineage,
        "reference_market_date": seed["reference_market_date"],
        "reference_close": prior_close,
        "reference_atr": prior_atr,
    }


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

    runtime = load_trading_scanner_runtime(project_root)
    latest_finalized_date = normalize_trading_date(
        runtime["latest_data_date"], field_name="latest_finalized_date", allow_none=False
    )
    market_evidence = validate_trading_actual_fill(
        project_root,
        ticker=ticker_key,
        price=price,
        trade_date=trade_date,
        latest_date=latest_finalized_date,
    )
    current_candidate = None
    manual_management = None
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
    else:
        manual_management = _build_direct_manual_managed_entry_context(
            project_root, ticker=ticker_key, trade_date=trade_date
        )
    return {
        "route": "scanner_strategy_buy" if current_candidate is not None else "manual_managed_buy",
        "candidate": current_candidate,
        "manual_management": manual_management,
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
    manual_management = dict(preview.get("manual_management") or {})
    if not manual_management:
        raise RuntimeError("手動補登缺少 managed-position planning context")
    account = record_managed_manual_trading_buy(
        project_root,
        ticker=ticker_key,
        qty=int(qty),
        price=float(price),
        trade_date=trade_date,
        expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
        params=manual_management["params"],
        execution_plan_seed=dict(manual_management["execution_plan_seed"]),
        management_lineage=dict(manual_management["management_lineage"]),
        management_start_date=manual_management["management_start_date"],
    )
    return {
        "route": "manual_managed_buy",
        "account": account,
        "manual_management": manual_management,
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
        runtime = load_trading_scanner_runtime(project_root)
        latest_finalized_date = normalize_trading_date(
            runtime["latest_data_date"], field_name="latest_finalized_date", allow_none=False
        )
        validate_trading_actual_fill(
            project_root,
            ticker=details.get("ticker"),
            price=price,
            trade_date=trade_date,
            latest_date=latest_finalized_date,
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
    "resolve_current_trading_scanner_candidate",
    "preview_trading_account_buy",
    "record_trading_account_buy",
    "record_trading_account_inventory_sell",
]
