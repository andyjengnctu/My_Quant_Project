"""Workbench-facing actual-account trade entry orchestration.

Workbench is a decision/accounting tool, not a broker OMS.  The user executes at
his broker and records the actual fill here.  Scanner-origin BUYs preserve the
strategy-management lineage even without a selected widget row; non-candidate
manual BUYs remain manual-managed positions.  SELLs are recorded directly against the selected broker inventory.
"""
from __future__ import annotations

import pandas as pd

from core.data_utils import get_required_min_rows
from core.exact_accounting import infer_security_profile, milli_to_price, price_to_milli, sync_position_display_fields
from core.entry_plans import build_position_from_frozen_entry_plan
from core.params_io import build_params_from_mapping
from core.position_replay import replay_confirmed_position_management
from core.price_utils import calc_frozen_target_price, calc_initial_stop_from_reference, calc_initial_trailing_stop_from_reference
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.file_integrity import canonical_json_sha256
from core.trading_account_state import (
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_SELL,
    TRADE_MUTATION_MANUAL_MANAGED_BUY,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
    effective_trading_account_events,
)
from core.trading_identity import normalize_trading_date
from core.trading_order_state import TRADING_ACTIVE_ORDER_STATUSES, TRADING_ORDER_SIDE_SELL
from services.trading.entry_lifecycle_context import resolve_trading_confirmed_entry_seed
from services.trading.account_state import (
    correct_trading_transaction,
    load_trading_account_state,
    record_managed_manual_trading_buy,
    record_manual_trading_sell,
    record_strategy_trading_buy,
)
from services.trading.actual_fill_validation import (
    validate_trading_actual_fill, validate_trading_fill_quantity, validate_pending_fill_terms,
)
from services.trading.state_lock import serialized_trading_state_mutation
from services.trading.pending_entry_state import load_trading_pending_entry_state
from services.trading.pending_entry_links import resolve_pending_entry_for_buy_event
from services.trading.market_data_consumer import load_trading_v2_sanitized_ohlcv_frame, open_trading_v2_consumer_view
from services.trading.order_state import get_trading_order_read_model
from services.trading.scanner_state import (
    load_trading_candidate_snapshot_for_account, load_trading_scanner_runtime,
    resolve_trading_candidate_snapshot_path,
)
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


def _resolve_direct_buy_candidate(project_root, *, ticker: str, candidate: dict | None) -> dict | None:
    """Resolve provenance from Scanner truth, never from widget selection alone.

    AI: Typing a Scanner ticker or losing a table selection cannot convert a
    strategy acquisition into a custom one. Discovery is read-only; an existing
    match still goes through the same currentness/hash guard as explicit picks.
    An absent snapshot means there is no Scanner evidence, not a failed match.
    Invalid or stale matching evidence must not silently downgrade provenance.
    """
    if candidate is not None:
        return _resolve_current_scanner_candidate(
            project_root, ticker=ticker, candidate_reference=dict(candidate)
        )
    if not resolve_trading_candidate_snapshot_path(project_root).is_file():
        return None
    snapshot = load_trading_candidate_snapshot_for_account(project_root, require_current=False)
    projection = dict(snapshot.get("entry_projection") or {})
    pending = set(projection.get("active_pending_tickers") or ())
    pending.update(snapshot.get("active_pending_candidate_tickers_skipped") or ())
    held = set(projection.get("held_tickers") or ())
    held.update(snapshot.get("held_candidate_tickers_skipped") or ())
    if ticker in pending:
        raise ValueError(f"{ticker} 已有掛單；請在掛單區確認成交，不可另建來源")
    if ticker in held:
        raise ValueError(f"{ticker} 已有持股；直接補單不可另建來源")
    matches = [dict(row) for row in snapshot.get("candidate_rows") or []
               if str(row.get("ticker") or "").strip().upper() == ticker]
    if len(matches) > 1:
        raise ValueError(f"{ticker} 的 Scanner 來源不唯一；請更新 Scanner 後再確認")
    if not matches:
        return None
    return _resolve_current_scanner_candidate(
        project_root, ticker=ticker, candidate_reference=matches[0]
    )


def resolve_trading_direct_buy_source(
    project_root,
    *,
    ticker: object,
    candidate: dict | None = None,
) -> dict:
    """Resolve direct-BUY provenance from canonical Scanner truth.

    The UI consumes this read-only resolver so the displayed source and the
    eventual account mutation use the same route decision.
    """
    ticker_key = str(ticker or "").strip().upper()
    if not ticker_key:
        raise ValueError("股票代號必填")
    current_candidate = _resolve_direct_buy_candidate(
        project_root, ticker=ticker_key, candidate=candidate
    )
    if current_candidate is not None:
        return {
            "route": "scanner_strategy_buy",
            "source": "strategy_fill",
            "candidate": current_candidate,
        }
    return {
        "route": "manual_managed_buy",
        "source": "manual_managed",
        "candidate": None,
    }


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
            "不能標記為 Scanner 策略成交。請確認原始成交資料；不能藉取消清單選取改變來源。"
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
        "management_start_date": fill_date,
        "execution_plan_seed": seed,
        "management_lineage": lineage,
        "reference_market_date": seed["reference_market_date"],
        "reference_close": prior_close,
        "reference_atr": prior_atr,
    }


def build_manual_adopted_management_context(
    project_root,
    *,
    ticker: str,
    broker: dict,
    lifecycle_context=None,
) -> dict:
    """Normalize an existing manual holding to the canonical managed contract.

    Broker truth remains authoritative.  The management geometry is rebuilt from
    the original broker entry date with the same canonical entry/roll-forward
    primitives used by ordinary managed manual positions, then advanced through
    the current finalized information date.  This is a compatibility migration
    for development-era account state, not a second product lifecycle.
    """
    ticker_key = str(ticker or "").strip().upper()
    if not ticker_key:
        raise ValueError("股票代號必填")
    broker_state = dict(broker or {})
    qty = int(broker_state.get("qty") or 0)
    remaining_cost_milli = int(broker_state.get("remaining_cost_basis_milli") or 0)
    if qty <= 0 or remaining_cost_milli <= 0:
        raise ValueError(f"{ticker_key} broker position qty/cost 不合法")
    entry_date = normalize_trading_date(
        broker_state.get("entry_date"), field_name="broker.entry_date", allow_none=False
    )

    runtime = load_trading_scanner_runtime(project_root)
    params = runtime["params"]
    information_date = normalize_trading_date(
        runtime["latest_data_date"], field_name="information_date", allow_none=False
    )
    if lifecycle_context is not None:
        information_date = lifecycle_context.finalized_date
    view = lifecycle_context.view if lifecycle_context is not None else open_trading_v2_consumer_view(project_root)
    frame = load_trading_v2_sanitized_ohlcv_frame(
        view,
        ticker=ticker_key,
        through_date=information_date,
        min_rows=get_required_min_rows(params),
    )
    if frame.empty:
        raise RuntimeError(f"{ticker_key} 沒有可用 Trading adjusted 日K")
    latest_market_date = pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d")
    if latest_market_date != information_date:
        raise RuntimeError(
            f"{ticker_key} 最新可用交易日 {latest_market_date} 與目前資訊日 {information_date} 不一致"
        )
    prior = frame.loc[frame.index < pd.Timestamp(entry_date)].copy()
    required_rows = int(get_required_min_rows(params))
    if len(prior) < required_rows:
        raise RuntimeError(
            f"{ticker_key} 原成交日前歷史資料不足；需要至少 {required_rows} 筆，實際 {len(prior)} 筆"
        )
    signals = generate_signals(frame, params, ticker=ticker_key)
    atr_values, _buy, _sell, _limits = unpack_precomputed_signals(signals)
    prior_index = int(frame.index.get_loc(prior.index[-1]))
    entry_atr = float(atr_values[prior_index])
    prior_close = float(prior["Close"].iloc[-1])
    if entry_atr <= 0 or prior_close <= 0:
        raise RuntimeError(f"{ticker_key} 原成交日前一交易日 Close/ATR 不合法")

    initial_qty = int(broker_state.get("initial_qty") or qty)
    price_basis_milli = broker_state.get("initial_gross_buy_milli")
    if price_basis_milli is None:
        price_basis_milli = broker_state.get("initial_cost_basis_milli")
    price_basis_milli = int(price_basis_milli or 0)
    if initial_qty <= 0 or price_basis_milli <= 0:
        raise ValueError(f"{ticker_key} 缺少可建立管理狀態的 broker 平均成本")
    average_entry_milli = max(1, (price_basis_milli + initial_qty // 2) // initial_qty)
    average_entry_price = milli_to_price(average_entry_milli)
    security_profile = infer_security_profile(ticker_key)
    initial_position_state = build_position_from_frozen_entry_plan(
        {"entry_atr": entry_atr, "target_reference_price": prior_close},
        buy_price=average_entry_price,
        qty=qty,
        params=params,
        entry_type="manual",
        ticker=ticker_key,
        security_profile=security_profile,
        trade_date=entry_date,
    )

    # Broker/account truth is never rewritten by management activation.
    initial_position_state["qty"] = qty
    initial_position_state["initial_qty"] = qty
    initial_position_state["net_buy_total_milli"] = int(broker_state.get("initial_cost_basis_milli") or remaining_cost_milli)
    initial_position_state["remaining_cost_basis_milli"] = remaining_cost_milli
    initial_position_state["realized_pnl_milli"] = int(broker_state.get("realized_pnl_milli") or 0)
    if broker_state.get("initial_gross_buy_milli") is not None:
        initial_position_state["gross_buy_milli"] = int(broker_state.get("initial_gross_buy_milli") or 0)
    if broker_state.get("initial_buy_fee_milli") is not None:
        initial_position_state["buy_fee_milli"] = int(broker_state.get("initial_buy_fee_milli") or 0)
    initial_position_state = sync_position_display_fields(initial_position_state)

    projection = replay_confirmed_position_management(
        initial_position_state, frame=frame, params=params, start_date=entry_date,
    )
    position_state = projection["position_state"]
    last_rollforward_date = projection["processed_through_date"]

    seed = {
        "entry_type": "manual",
        "entry_atr": entry_atr,
        "init_sl": initial_position_state.get("initial_stop"),
        "init_trail": initial_position_state.get("trailing_stop"),
        "target_price": initial_position_state.get("tp_half"),
        "target_reference_price": prior_close,
        "limit_price": None,
        "security_profile": security_profile,
        "trade_date": entry_date,
        "reference_market_date": pd.Timestamp(prior.index[-1]).strftime("%Y-%m-%d"),
        "reference_close": prior_close,
        "broker_average_entry_price": average_entry_price,
    }
    lineage = build_trading_manual_management_lineage(
        params=params,
        execution_plan_seed=seed,
        information_date=information_date,
        origin="manual_managed",
        planned_qty=qty,
        planned_cost=None,
        target_reference_close=prior_close,
    )
    return {
        "params": params,
        "information_date": information_date,
        "management_start_date": entry_date,
        "last_rollforward_date": last_rollforward_date,
        "execution_plan_seed": seed,
        "management_lineage": lineage,
        "initial_position_state": initial_position_state,
        "position_state": position_state,
        "reference_market_date": seed["reference_market_date"],
        "reference_close": prior_close,
        "reference_atr": entry_atr,
        "average_entry_price": average_entry_price,
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
    qty_int = validate_trading_fill_quantity(qty)
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
    source_resolution = resolve_trading_direct_buy_source(
        project_root, ticker=ticker_key, candidate=candidate
    )
    current_candidate = source_resolution["candidate"]
    manual_management = None
    warnings: list[str] = []
    if current_candidate is not None:
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
        "route": source_resolution["route"],
        "source": source_resolution["source"],
        "candidate": current_candidate,
        "manual_management": manual_management,
        "warnings": warnings,
        "market_evidence": market_evidence,
    }


@serialized_trading_state_mutation
def record_trading_account_buy(
    project_root,
    *,
    ticker: object,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int | None = None,
    candidate: dict | None = None,
    expected_route: str | None = None,
):
    preview = preview_trading_account_buy(
        project_root,
        ticker=ticker,
        qty=qty,
        price=price,
        trade_date=trade_date,
        candidate=candidate,
    )
    if expected_route is not None and preview["route"] != expected_route:
        raise RuntimeError("買入來源在預覽後已改變；請重新檢視來源並確認")
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
        acquisition_seed = resolve_trading_confirmed_entry_seed(
            project_root, ticker=ticker_key, lineage=strategy_lineage, params=params,
            fill_date=trade_date, information_date=candidate_row.get("trade_date"),
        )
        account = record_strategy_trading_buy(
            project_root,
            ticker=ticker_key,
            qty=int(qty),
            price=strategy_price,
            trade_date=trade_date,
            expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
            params=params,
            execution_plan_seed=acquisition_seed,
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


@serialized_trading_state_mutation
def correct_trading_account_transaction(
    project_root,
    *,
    transaction_revision: int,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int | None = None,
):
    """Correct a trade with the same evidence and original-intent guards as entry."""
    event = _effective_transaction_event(project_root, int(transaction_revision))
    mutation = str(event.get("mutation_type") or "")
    details = dict(event.get("details") or {})
    qty = validate_trading_fill_quantity(qty)
    if mutation in _BUY_MUTATIONS or mutation == TRADE_MUTATION_SELL:
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
    if mutation in _BUY_MUTATIONS:
        account = load_trading_account_state(project_root, required=True)
        pending = load_trading_pending_entry_state(project_root, required=False) or {}
        original_pending = resolve_pending_entry_for_buy_event(
            event, (pending.get("entries") or {}).values(), account_events=account.get("events") or (),
        )
        if original_pending is not None:
            validate_pending_fill_terms(original_pending, qty=qty, price=price, trade_date=trade_date)
    return correct_trading_transaction(
        project_root,
        transaction_revision=int(transaction_revision),
        qty=int(qty),
        price=price,
        trade_date=trade_date,
        expected_revision=(None if expected_account_revision is None else int(expected_account_revision)),
    )


@serialized_trading_state_mutation
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
    qty = validate_trading_fill_quantity(qty)
    runtime = load_trading_scanner_runtime(project_root)
    validate_trading_actual_fill(
        project_root, ticker=ticker_key, price=price, trade_date=trade_date,
        latest_date=runtime["latest_data_date"],
    )
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
    "build_manual_adopted_management_context",
    "correct_trading_account_transaction",
    "list_active_sell_orders_for_ticker",
    "resolve_current_trading_scanner_candidate",
    "resolve_trading_direct_buy_source",
    "preview_trading_account_buy",
    "record_trading_account_buy",
    "record_trading_account_inventory_sell",
]
