"""Workbench service for persistent pre-market entry intents.

A pending entry is a user-confirmed pre-market decision.  It freezes the exact
Params/entry geometry used at creation, reserves account resources, and is later
closed either as no-fill or transferred to a managed account position.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS
from core.capital_policy import resolve_scanner_live_capital
from core.data_utils import get_required_min_rows
from core.entry_plans import build_cash_capped_entry_plan, build_normal_candidate_plan
from core.exact_accounting import (
    build_buy_ledger,
    infer_security_profile,
    milli_to_money,
    price_to_milli,
    round_price_to_tick_milli,
)
from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.params_io import build_params_from_mapping
from core.price_utils import adjust_long_buy_limit
from core.runtime_utils import get_taipei_now
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from core.trading_account_state import (
    POSITION_SOURCE_MANUAL_MANAGED,
    POSITION_SOURCE_STRATEGY_FILL,
)
from core.trading_state_paths import resolve_trading_pending_entry_transaction_path
from services.trading.account_state import (
    load_trading_account_state,
    record_managed_manual_trading_buy,
    record_strategy_trading_buy,
)
from services.trading.account_trade_entry import resolve_current_trading_scanner_candidate
from services.trading.actual_fill_validation import validate_trading_actual_fill, validate_trading_fill_quantity, validate_pending_fill_terms
from services.trading.market_data_consumer import (
    load_trading_v2_sanitized_ohlcv_frame,
    open_trading_v2_consumer_view,
)
from services.trading.lifecycle_sync_status import SYNC_STATUS_LATEST, SYNC_STATUS_PENDING
from services.trading.order_form_constraints import (
    resolve_preferred_pending_order_date,
    validate_pending_order_limit_price,
    validate_pending_order_trade_date,
)
from services.trading.pending_entry_state import (
    PENDING_ENTRY_STATUS_ACTIVE,
    cancel_trading_pending_entry,
    create_trading_pending_entry,
    delete_trading_pending_entry,
    load_trading_pending_entry_state,
    mark_trading_pending_entry_filled,
    project_trading_pending_entry_state,
    update_trading_pending_entry,
)
from services.trading.scanner_state import load_trading_scanner_runtime
from services.trading.state_lock import serialized_trading_state_mutation
from services.trading.strategy_param_runtime import (
    build_trading_candidate_strategy_lineage,
    build_trading_manual_management_lineage,
    resolve_trading_candidate_frozen_params,
)

PENDING_ENTRY_ORIGIN_SCANNER = "scanner_strategy"
PENDING_ENTRY_ORIGIN_MANUAL = "manual_selected"
PENDING_ENTRY_TRANSACTION_SCHEMA_VERSION = 1


def _timestamp() -> str:
    return get_taipei_now().isoformat(timespec="seconds")


def _default_planned_trade_date() -> str:
    return get_taipei_now().date().isoformat()


def _account_resources(
    project_root,
    *,
    information_date: str,
    exclude_pending_entry_id: str | None = None,
) -> dict[str, Any]:
    account = load_trading_account_state(project_root, required=True)
    cash_milli = account.get("cash_milli")
    if cash_milli is None:
        raise RuntimeError("Trading cash 尚未設定，不能建立掛單")
    pending_state = load_trading_pending_entry_state(project_root, required=False)
    pending = project_trading_pending_entry_state(
        pending_state,
        current_information_date=information_date,
    )
    excluded_id = str(exclude_pending_entry_id or "").strip() or None
    stale_rows = [
        row for row in list(pending.get("stale_active_entries") or [])
        if str(row.get("pending_entry_id") or "") != excluded_id
    ]
    if stale_rows:
        tickers = [str(row.get("ticker")) for row in stale_rows]
        raise RuntimeError(
            "尚有前一資訊日未結案掛單；請先確認成交或刪除掛單：" + ", ".join(tickers)
        )
    held_tickers = {normalize_trading_ticker(t) for t in (account.get("positions") or {})}
    locked_rows = [
        row for row in list(pending.get("locked_entries") or [])
        if str(row.get("pending_entry_id") or "") != excluded_id
    ]
    locked_tickers = {normalize_trading_ticker(row.get("ticker")) for row in locked_rows}
    held_count = len(held_tickers)
    locked_count = len(locked_tickers)
    slot_quota = max(0, int(DEFAULT_PORTFOLIO_MAX_POSITIONS) - held_count)
    occupied = held_count + locked_count
    free_slots = max(0, slot_quota - locked_count)
    reserved_total_milli = sum(int(row.get("reserved_cost_milli") or 0) for row in locked_rows)
    available_cash_milli = int(cash_milli) - reserved_total_milli
    return {
        "account": account,
        "pending": pending,
        "held_tickers": held_tickers,
        "locked_tickers": locked_tickers,
        "held_count": held_count,
        "locked_count": locked_count,
        "slot_quota": slot_quota,
        "occupied": occupied,
        "max_positions": int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        "cash_limit_milli": int(cash_milli),
        "cash_limit": milli_to_money(int(cash_milli)),
        "reserved_total_milli": int(reserved_total_milli),
        "free_slots": free_slots,
        "available_cash_milli": max(0, available_cash_milli),
        "available_cash": milli_to_money(max(0, available_cash_milli)),
    }


def _assert_can_add_ticker(resources: Mapping[str, Any], *, ticker: str) -> None:
    if ticker in set(resources.get("held_tickers") or set()):
        raise ValueError(f"{ticker} 已在持股中，不能建立新的買入掛單")
    if ticker in set(resources.get("locked_tickers") or set()):
        raise ValueError(f"{ticker} 已有尚未釋放資源的掛單")
    if int(resources.get("free_slots") or 0) <= 0:
        raise ValueError(f"持股／掛單已達上限 {int(DEFAULT_PORTFOLIO_MAX_POSITIONS)}")
    if float(resources.get("available_cash") or 0.0) <= 0:
        raise ValueError("可用現金已被持股／掛單預留完畢")


def _pending_entry_payload(
    *,
    origin: str,
    ticker: str,
    information_date: str,
    plan: Mapping[str, Any],
    lineage: Mapping[str, Any],
    signal_date: str | None,
    planned_trade_date: str | None = None,
    candidate_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    seed = deepcopy(dict(plan))
    seed["trade_date"] = information_date
    seed["entry_type"] = str(seed.get("entry_type") or ("manual" if origin == PENDING_ENTRY_ORIGIN_MANUAL else "normal"))
    reserved_milli = int(seed.get("reserved_cost_milli") or 0)
    return {
        "origin": origin,
        "ticker": ticker,
        "information_date": information_date,
        "planned_trade_date": normalize_trading_date(
            planned_trade_date or _default_planned_trade_date(),
            field_name="planned_trade_date",
            allow_none=False,
        ),
        "signal_date": signal_date,
        "execution_plan_seed": seed,
        "planned_qty": int(seed["qty"]),
        "reserved_cost_milli": reserved_milli,
        "reserved_cost": float(seed.get("reserved_cost") or milli_to_money(reserved_milli)),
        "limit_price": float(seed["limit_price"]),
        "init_sl": float(seed["init_sl"]),
        "init_trail": float(seed["init_trail"]),
        "target_price": float(seed["target_price"]),
        "entry_atr": None if seed.get("entry_atr") is None else float(seed["entry_atr"]),
        "management_lineage": deepcopy(dict(lineage)),
        "candidate_reference_sha256": None if candidate_reference is None else canonical_json_sha256(dict(candidate_reference)),
    }


def _normalize_user_limit_price(*, ticker: str, raw_price, security_profile) -> float:
    try:
        price = float(raw_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("掛單限價必須是有效數字") from exc
    if pd.isna(price) or price <= 0:
        raise ValueError("掛單限價必須 > 0")
    price_milli = int(price_to_milli(price))
    rounded_milli = int(
        round_price_to_tick_milli(price, direction="nearest", ticker=ticker, security_profile=security_profile)
    )
    if price_milli != rounded_milli:
        raise ValueError(f"{ticker} 掛單限價 {raw_price} 不符合台股合法跳動單位")
    return price


def _apply_pending_draft_overrides(
    *,
    ticker: str,
    base_plan: Mapping[str, Any],
    params,
    resources: Mapping[str, Any],
    qty: int | None,
    limit_price,
) -> dict[str, Any]:
    seed = deepcopy(dict(base_plan or {}))
    if not seed:
        raise ValueError(f"{ticker} 缺少可用掛單計畫")
    security_profile = seed.get("security_profile") or infer_security_profile(ticker)
    original_limit = float(seed.get("limit_price") or 0.0)
    resolved_limit = original_limit if limit_price is None else _normalize_user_limit_price(
        ticker=ticker,
        raw_price=limit_price,
        security_profile=security_profile,
    )
    if resolved_limit <= 0:
        raise ValueError(f"{ticker} 缺少有效掛單限價")

    if limit_price is not None and price_to_milli(resolved_limit) != price_to_milli(original_limit):
        entry_atr = seed.get("entry_atr")
        if entry_atr is None or pd.isna(entry_atr) or float(entry_atr) <= 0:
            raise ValueError(f"{ticker} 缺少 entry ATR，不能修改掛單限價")
        rebuilt = build_normal_candidate_plan(
            resolved_limit,
            float(entry_atr),
            float(seed.get("sizing_capital") or resolve_scanner_live_capital(params)),
            params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=seed.get("trade_date"),
        )
        if rebuilt is None:
            raise ValueError(f"{ticker} 修改限價後無法建立掛單計畫")
        preserved = deepcopy(seed)
        preserved.update(rebuilt)
        seed = preserved
        seed["limit_price"] = resolved_limit

    auto_plan = build_cash_capped_entry_plan(seed, float(resources["available_cash"]), params)
    if auto_plan is None or int(auto_plan.get("qty") or 0) <= 0:
        raise ValueError(f"{ticker} 在目前可用現金下無可執行股數")
    max_safe_qty = int(auto_plan["qty"])
    resolved_qty = max_safe_qty if qty is None else int(qty)
    if resolved_qty <= 0:
        raise ValueError("掛單股數必須 > 0")
    if resolved_qty > max_safe_qty:
        raise ValueError(
            f"掛單股數 {resolved_qty:,} 超過目前參數／資金可執行上限 {max_safe_qty:,}"
        )
    reserved_cost_milli = int(
        build_buy_ledger(price_to_milli(auto_plan["limit_price"]), resolved_qty, params)["cash_buy_total_milli"]
    )
    if reserved_cost_milli > int(resources["available_cash_milli"]):
        raise ValueError("掛單預留成本超過目前可用現金")
    auto_plan["qty"] = resolved_qty
    auto_plan["reserved_cost_milli"] = reserved_cost_milli
    auto_plan["reserved_cost"] = milli_to_money(reserved_cost_milli)
    auto_plan["user_qty_override"] = None if qty is None else int(qty)
    auto_plan["user_limit_override"] = None if limit_price is None else float(resolved_limit)
    return auto_plan


def _resource_summary(resources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "held_count": int(resources.get("held_count") or 0),
        "locked_slots": int(resources.get("locked_count") or 0),
        "slot_quota": int(resources.get("slot_quota") or 0),
        "occupied": int(resources.get("occupied") or 0),
        "max_positions": int(resources.get("max_positions") or DEFAULT_PORTFOLIO_MAX_POSITIONS),
        "reserved_total_milli": int(resources.get("reserved_total_milli") or 0),
        "cash_limit_milli": int(resources.get("cash_limit_milli") or 0),
        "available_cash_milli": int(resources.get("available_cash_milli") or 0),
    }


def _prepare_scanner_pending_entry(
    project_root,
    *,
    candidate_reference: Mapping[str, Any],
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
    exclude_pending_entry_id: str | None = None,
) -> dict[str, Any]:
    reference = dict(candidate_reference or {})
    ticker = normalize_trading_ticker(reference.get("ticker"))
    candidate = resolve_current_trading_scanner_candidate(
        project_root,
        ticker=ticker,
        candidate_reference=reference,
    )
    information_date = normalize_trading_date(
        candidate.get("trade_date") or (candidate.get("execution_plan_seed") or {}).get("trade_date"),
        field_name="candidate.trade_date",
        allow_none=False,
    )
    runtime = load_trading_scanner_runtime(project_root)
    latest_finalized_date = normalize_trading_date(
        runtime["latest_data_date"], field_name="latest_finalized_date", allow_none=False
    )
    resources = _account_resources(
        project_root,
        information_date=information_date,
        exclude_pending_entry_id=exclude_pending_entry_id,
    )
    _assert_can_add_ticker(resources, ticker=ticker)
    params, _member = resolve_trading_candidate_frozen_params(candidate)
    seed = deepcopy(dict(candidate.get("execution_plan_seed") or {}))
    if not seed:
        raise RuntimeError(f"{ticker} Scanner candidate 缺少 execution_plan_seed")
    planned_qty = int(candidate.get("proj_qty") or seed.get("qty") or 0)
    if planned_qty <= 0:
        raise ValueError(f"{ticker} Scanner 規劃股數為 0，不能加入掛單")
    seed["max_qty"] = planned_qty
    # Scanner Pool and pending-order preview must share one sizing-capital SSOT.
    # Extended/TBD seeds can originate from a single-stock backtest capital; never
    # let that advisory backtest capital override the Scanner live-capital contract.
    seed["sizing_capital"] = float(resolve_scanner_live_capital(params))
    plan = _apply_pending_draft_overrides(
        ticker=ticker,
        base_plan=seed,
        params=params,
        resources=resources,
        qty=qty,
        limit_price=limit_price,
    )
    resolved_planned_date = (
        resolve_preferred_pending_order_date(
            project_root, ticker=ticker, latest_finalized_date=latest_finalized_date
        )
        if planned_trade_date is None
        else validate_pending_order_trade_date(
            project_root,
            ticker=ticker,
            latest_finalized_date=latest_finalized_date,
            planned_trade_date=planned_trade_date,
        )
    )
    plan["planned_trade_date"] = resolved_planned_date
    validate_pending_order_limit_price(
        project_root,
        ticker=ticker,
        planned_trade_date=resolved_planned_date,
        latest_finalized_date=latest_finalized_date,
        limit_price=plan["limit_price"],
    )
    candidate_for_lineage = deepcopy(candidate)
    candidate_for_lineage["execution_plan_seed"] = deepcopy(plan)
    candidate_for_lineage["proj_qty"] = int(plan["qty"])
    candidate_for_lineage["proj_cost"] = float(plan["reserved_cost"])
    lineage = build_trading_candidate_strategy_lineage(candidate_for_lineage)
    entry = _pending_entry_payload(
        origin=PENDING_ENTRY_ORIGIN_SCANNER,
        ticker=ticker,
        information_date=information_date,
        planned_trade_date=resolved_planned_date,
        plan=plan,
        lineage=lineage,
        signal_date=normalize_trading_date(candidate.get("signal_date"), field_name="signal_date", allow_none=True),
        candidate_reference=candidate,
    )
    entry["resource_summary"] = _resource_summary(resources)
    return entry


def preview_scanner_trading_pending_entry(
    project_root,
    *,
    candidate_reference: Mapping[str, Any],
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
) -> dict[str, Any]:
    return _prepare_scanner_pending_entry(
        project_root,
        candidate_reference=candidate_reference,
        qty=qty,
        limit_price=limit_price,
        planned_trade_date=planned_trade_date,
    )


@serialized_trading_state_mutation
def create_scanner_trading_pending_entry(
    project_root,
    *,
    candidate_reference: Mapping[str, Any],
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
) -> dict[str, Any]:
    entry = _prepare_scanner_pending_entry(
        project_root,
        candidate_reference=candidate_reference,
        qty=qty,
        limit_price=limit_price,
        planned_trade_date=planned_trade_date,
    )
    entry.pop("resource_summary", None)
    return create_trading_pending_entry(project_root, entry=entry)


def _manual_market_context(project_root, *, ticker: str, params, through_date: str) -> dict[str, Any]:
    view = open_trading_v2_consumer_view(project_root)
    frame = load_trading_v2_sanitized_ohlcv_frame(
        view,
        ticker=ticker,
        through_date=through_date,
        min_rows=get_required_min_rows(params),
    )
    if frame.empty:
        raise RuntimeError(f"{ticker} 沒有可用 Trading adjusted 日K")
    signals = generate_signals(frame, params, ticker=ticker)
    atr_values, _buy, _sell, _limits = unpack_precomputed_signals(signals)
    atr = float(atr_values[-1])
    close = float(frame["Close"].iloc[-1])
    if pd.isna(atr) or atr <= 0 or pd.isna(close) or close <= 0:
        raise RuntimeError(f"{ticker} 最新 completed bar 無有效 Close/ATR")
    date_text = pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d")
    if date_text != through_date:
        raise RuntimeError(f"{ticker} 最新可用交易日 {date_text} 與目前資訊日 {through_date} 不一致")
    return {"frame": frame, "close": close, "atr": atr, "date": date_text}


def _prepare_manual_pending_entry(
    project_root,
    *,
    ticker: object,
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
    exclude_pending_entry_id: str | None = None,
) -> dict[str, Any]:
    ticker_key = normalize_trading_ticker(ticker)
    runtime = load_trading_scanner_runtime(project_root)
    information_date = normalize_trading_date(runtime["latest_data_date"], allow_none=False)
    resources = _account_resources(
        project_root,
        information_date=information_date,
        exclude_pending_entry_id=exclude_pending_entry_id,
    )
    _assert_can_add_ticker(resources, ticker=ticker_key)
    params = runtime["params"]
    market = _manual_market_context(project_root, ticker=ticker_key, params=params, through_date=information_date)
    security_profile = infer_security_profile(ticker_key)
    user_limit_price = limit_price
    default_limit_price = adjust_long_buy_limit(
        market["close"] + market["atr"] * float(params.atr_buy_tol),
        ticker=ticker_key,
        security_profile=security_profile,
    )
    base_plan = build_normal_candidate_plan(
        default_limit_price,
        market["atr"],
        float(resolve_scanner_live_capital(params)),
        params,
        ticker=ticker_key,
        security_profile=security_profile,
        trade_date=information_date,
    )
    if base_plan is None:
        raise ValueError(f"{ticker_key} 無法建立盤前 entry plan")
    base_plan["entry_type"] = "manual"
    plan = _apply_pending_draft_overrides(
        ticker=ticker_key,
        base_plan=base_plan,
        params=params,
        resources=resources,
        qty=qty,
        limit_price=user_limit_price,
    )
    resolved_planned_date = (
        resolve_preferred_pending_order_date(
            project_root, ticker=ticker_key, latest_finalized_date=information_date
        )
        if planned_trade_date is None
        else validate_pending_order_trade_date(
            project_root,
            ticker=ticker_key,
            latest_finalized_date=information_date,
            planned_trade_date=planned_trade_date,
        )
    )
    plan["planned_trade_date"] = resolved_planned_date
    validate_pending_order_limit_price(
        project_root,
        ticker=ticker_key,
        planned_trade_date=resolved_planned_date,
        latest_finalized_date=information_date,
        limit_price=plan["limit_price"],
    )
    lineage = build_trading_manual_management_lineage(
        params=params,
        execution_plan_seed=plan,
        information_date=information_date,
        origin="manual_pending_entry",
        planned_qty=int(plan["qty"]),
        planned_cost=float(plan["reserved_cost"]),
    )
    entry = _pending_entry_payload(
        origin=PENDING_ENTRY_ORIGIN_MANUAL,
        ticker=ticker_key,
        information_date=information_date,
        planned_trade_date=resolved_planned_date,
        plan=plan,
        lineage=lineage,
        signal_date=None,
    )
    entry["resource_summary"] = _resource_summary(resources)
    return entry


def preview_manual_trading_pending_entry(
    project_root,
    *,
    ticker: object,
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
) -> dict[str, Any]:
    return _prepare_manual_pending_entry(
        project_root,
        ticker=ticker,
        qty=qty,
        limit_price=limit_price,
        planned_trade_date=planned_trade_date,
    )


@serialized_trading_state_mutation
def create_manual_trading_pending_entry(
    project_root,
    *,
    ticker: object,
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
) -> dict[str, Any]:
    entry = _prepare_manual_pending_entry(
        project_root,
        ticker=ticker,
        qty=qty,
        limit_price=limit_price,
        planned_trade_date=planned_trade_date,
    )
    entry.pop("resource_summary", None)
    return create_trading_pending_entry(project_root, entry=entry)


def _rebuild_existing_pending_lineage(entry: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    lineage = deepcopy(dict(entry.get("management_lineage") or {}))
    if not lineage:
        raise RuntimeError("既有掛單缺少 frozen management lineage")
    seed = deepcopy(dict(plan or {}))
    lineage["execution_plan_seed"] = seed
    lineage["planned_qty"] = int(seed.get("qty") or 0)
    lineage["planned_cost"] = float(seed.get("reserved_cost") or 0.0)
    origin = str(entry.get("origin") or "")
    if origin == PENDING_ENTRY_ORIGIN_SCANNER:
        identity_payload = {
            "params_signature": str(lineage.get("params_signature") or ""),
            "ensemble_member_key": str(lineage.get("ensemble_member_key") or ""),
            "frozen_params_sha256": str(lineage.get("frozen_params_sha256") or ""),
            "execution_plan_seed": seed,
        }
    elif origin == PENDING_ENTRY_ORIGIN_MANUAL:
        identity_payload = {
            "params_signature": str(lineage.get("params_signature") or ""),
            "frozen_params_sha256": str(lineage.get("frozen_params_sha256") or ""),
            "execution_plan_seed": seed,
            "information_date": str(entry.get("information_date") or lineage.get("candidate_trade_date") or ""),
            "origin": str(lineage.get("origin") or "manual_pending_entry"),
            "target_reference_close": lineage.get("target_reference_close"),
        }
    else:
        raise RuntimeError(f"未知掛單來源: {origin or '-'}")
    lineage["lineage_id"] = canonical_json_sha256(identity_payload)
    return lineage


def _prepare_existing_pending_entry_update(
    project_root,
    *,
    pending_entry_id: str,
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
) -> dict[str, Any]:
    current = _load_active_pending(project_root, pending_entry_id)
    ticker = normalize_trading_ticker(current.get("ticker"))
    information_date = normalize_trading_date(
        current.get("information_date"), field_name="information_date", allow_none=False
    )
    runtime = load_trading_scanner_runtime(project_root)
    latest_finalized_date = normalize_trading_date(
        runtime["latest_data_date"], field_name="latest_finalized_date", allow_none=False
    )
    resources = _account_resources(
        project_root, information_date=information_date, exclude_pending_entry_id=pending_entry_id
    )
    _assert_can_add_ticker(resources, ticker=ticker)
    lineage = deepcopy(dict(current.get("management_lineage") or {}))
    frozen_params = lineage.get("frozen_params")
    if not isinstance(frozen_params, Mapping):
        raise RuntimeError(f"{ticker} 掛單缺少 frozen params")
    params = build_params_from_mapping(dict(frozen_params))
    plan = _apply_pending_draft_overrides(
        ticker=ticker,
        base_plan=dict(current.get("execution_plan_seed") or {}),
        params=params,
        resources=resources,
        qty=qty,
        limit_price=limit_price,
    )
    requested_planned_date = planned_trade_date or current.get("planned_trade_date")
    resolved_planned_date = validate_pending_order_trade_date(
        project_root,
        ticker=ticker,
        latest_finalized_date=latest_finalized_date,
        planned_trade_date=requested_planned_date,
    )
    plan["planned_trade_date"] = resolved_planned_date
    lineage = _rebuild_existing_pending_lineage(current, plan)
    validate_pending_order_limit_price(
        project_root,
        ticker=ticker,
        planned_trade_date=resolved_planned_date,
        latest_finalized_date=latest_finalized_date,
        limit_price=plan["limit_price"],
    )
    replacement = _pending_entry_payload(
        origin=str(current.get("origin") or ""),
        ticker=ticker,
        information_date=information_date,
        planned_trade_date=resolved_planned_date,
        plan=plan,
        lineage=lineage,
        signal_date=normalize_trading_date(current.get("signal_date"), field_name="signal_date", allow_none=True),
    )
    replacement["candidate_reference_sha256"] = current.get("candidate_reference_sha256")
    replacement["resource_summary"] = _resource_summary(resources)
    return replacement


def preview_trading_pending_entry_update(
    project_root,
    *,
    pending_entry_id: str,
    origin: str,
    ticker: object,
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
    candidate_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = _load_active_pending(project_root, pending_entry_id)
    ticker_key = normalize_trading_ticker(ticker)
    current_ticker = normalize_trading_ticker(current.get("ticker"))
    current_origin = str(current.get("origin") or "")
    requested_origin = str(origin or "manual")
    normalized_origin = PENDING_ENTRY_ORIGIN_SCANNER if requested_origin in {"scanner", PENDING_ENTRY_ORIGIN_SCANNER} else PENDING_ENTRY_ORIGIN_MANUAL

    if ticker_key == current_ticker and normalized_origin == current_origin:
        return _prepare_existing_pending_entry_update(
            project_root,
            pending_entry_id=pending_entry_id,
            qty=qty,
            limit_price=limit_price,
            planned_trade_date=planned_trade_date,
        )
    if normalized_origin == PENDING_ENTRY_ORIGIN_SCANNER:
        if not candidate_reference:
            raise ValueError("Scanner 掛單修改缺少目前 candidate reference")
        return _prepare_scanner_pending_entry(
            project_root,
            candidate_reference=candidate_reference,
            qty=qty,
            limit_price=limit_price,
            planned_trade_date=planned_trade_date,
            exclude_pending_entry_id=pending_entry_id,
        )
    return _prepare_manual_pending_entry(
        project_root,
        ticker=ticker_key,
        qty=qty,
        limit_price=limit_price,
        planned_trade_date=planned_trade_date,
        exclude_pending_entry_id=pending_entry_id,
    )


@serialized_trading_state_mutation
def update_trading_pending_entry_intent(
    project_root,
    *,
    pending_entry_id: str,
    origin: str,
    ticker: object,
    qty: int | None = None,
    limit_price=None,
    planned_trade_date: str | None = None,
    candidate_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recover_trading_pending_entry_transaction(project_root)
    replacement = preview_trading_pending_entry_update(
        project_root,
        pending_entry_id=pending_entry_id,
        origin=origin,
        ticker=ticker,
        qty=qty,
        limit_price=limit_price,
        planned_trade_date=planned_trade_date,
        candidate_reference=candidate_reference,
    )
    replacement.pop("resource_summary", None)
    return update_trading_pending_entry(
        project_root, pending_entry_id=pending_entry_id, replacement=replacement
    )


def get_trading_pending_entry_read_model(project_root) -> dict[str, Any]:
    # Complete any durable account-commit -> pending-close recovery before exposing
    # the Workbench read model, so a process restart cannot leave a filled order
    # visible as ACTIVE indefinitely.
    recover_trading_pending_entry_transaction(project_root)
    try:
        runtime = load_trading_scanner_runtime(project_root)
        information_date = normalize_trading_date(runtime["latest_data_date"], allow_none=False)
    except (OSError, ValueError, RuntimeError, FileNotFoundError):
        information_date = None
    state = load_trading_pending_entry_state(project_root, required=False)
    projected = project_trading_pending_entry_state(state, current_information_date=information_date)
    account = load_trading_account_state(project_root, required=False)
    cash_limit_milli = None if not isinstance(account, Mapping) else account.get("cash_milli")
    held_count = 0 if not isinstance(account, Mapping) else len(account.get("positions") or {})
    locked_slots = int(projected.get("locked_count") or 0)
    projected["resource_usage"] = {
        "held_count": int(held_count),
        "locked_slots": locked_slots,
        "slot_quota": max(0, int(DEFAULT_PORTFOLIO_MAX_POSITIONS) - int(held_count)),
        "occupied": int(held_count) + locked_slots,
        "max_positions": int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        "reserved_total_milli": int(projected.get("reserved_total_milli") or 0),
        "cash_limit_milli": None if cash_limit_milli is None else int(cash_limit_milli),
    }
    rows = []
    for row in projected["entries"]:
        item = deepcopy(row)
        item["locked"] = any(
            str(locked.get("pending_entry_id")) == str(item.get("pending_entry_id"))
            for locked in projected["locked_entries"]
        )
        item["stale"] = any(
            str(stale.get("pending_entry_id")) == str(item.get("pending_entry_id"))
            for stale in projected["stale_active_entries"]
        )
        if str(item.get("status") or "") == PENDING_ENTRY_STATUS_ACTIVE:
            item["sync_status"] = SYNC_STATUS_PENDING if bool(item["stale"]) else SYNC_STATUS_LATEST
        else:
            item["sync_status"] = None
        rows.append(item)
    projected["entries"] = rows
    return projected


def _load_active_pending(project_root, pending_entry_id: str) -> dict[str, Any]:
    state = load_trading_pending_entry_state(project_root, required=True)
    entry_id = str(pending_entry_id or "").strip()
    row = (state.get("entries") or {}).get(entry_id)
    if not isinstance(row, dict):
        raise ValueError(f"找不到 Trading 掛單: {entry_id}")
    if str(row.get("status")) != PENDING_ENTRY_STATUS_ACTIVE:
        raise ValueError(f"只有 ACTIVE 掛單可確認成交: {entry_id}")
    return deepcopy(row)


def preview_trading_pending_entry_fill(
    project_root,
    *,
    pending_entry_id: str,
    qty: int,
    price,
    trade_date,
) -> dict[str, Any]:
    entry = _load_active_pending(project_root, pending_entry_id)
    qty_int = validate_trading_fill_quantity(qty)
    # AI: Original pending terms are shared by first fill and account corrections.
    validate_pending_fill_terms(entry, qty=qty_int, price=price, trade_date=trade_date)
    fill_date = normalize_trading_date(trade_date, field_name="trade_date", allow_none=False)
    runtime = load_trading_scanner_runtime(project_root)
    latest_finalized_date = normalize_trading_date(
        runtime["latest_data_date"], field_name="latest_finalized_date", allow_none=False
    )
    evidence = validate_trading_actual_fill(
        project_root,
        ticker=entry["ticker"],
        price=price,
        trade_date=fill_date,
        latest_date=latest_finalized_date,
    )
    info_date = normalize_trading_date(entry["information_date"], field_name="information_date", allow_none=False)
    warnings = []
    if fill_date <= info_date:
        warnings.append(
            f"成交日 {fill_date} 早於或等於掛單資訊日 {info_date}；若為歷史補登可繼續，請確認日期。"
        )
    account = load_trading_account_state(project_root, required=True)
    if normalize_trading_ticker(entry["ticker"]) in (account.get("positions") or {}):
        raise ValueError(f"{entry['ticker']} 已存在持股，不能再次由掛單轉入")
    return {"entry": entry, "warnings": warnings, "market_evidence": evidence}


def _write_pending_transfer_tx(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError("Trading 尚有未完成掛單轉持股 transaction；請先重新整理完成 recovery")
    atomic_write_json(path, dict(payload))


@serialized_trading_state_mutation
def recover_trading_pending_entry_transaction(project_root) -> dict[str, Any] | None:
    path = resolve_trading_pending_entry_transaction_path(project_root)
    if not path.is_file():
        return None
    tx = load_json_strict(path)
    if int(tx.get("schema_version") or -1) != PENDING_ENTRY_TRANSACTION_SCHEMA_VERSION:
        raise RuntimeError("Trading pending-entry transaction schema 不相容")
    entry_id = str(tx.get("pending_entry_id") or "")
    ticker = normalize_trading_ticker(tx.get("ticker"))
    pending = load_trading_pending_entry_state(project_root, required=True)
    row = (pending.get("entries") or {}).get(entry_id)
    if not isinstance(row, dict):
        raise RuntimeError("Trading pending-entry transaction 找不到對應掛單")
    if str(row.get("status")) != PENDING_ENTRY_STATUS_ACTIVE:
        path.unlink(missing_ok=True)
        return {"status": "ALREADY_CLOSED", "pending_entry_id": entry_id}
    account = load_trading_account_state(project_root, required=True)
    position = (account.get("positions") or {}).get(ticker)
    if isinstance(position, dict):
        broker = position.get("broker") or {}
        fill = dict(tx.get("fill") or {})
        if normalize_trading_date(broker.get("entry_date"), allow_none=False) != normalize_trading_date(fill.get("trade_date"), allow_none=False):
            raise RuntimeError("Trading pending-entry recovery 發現同 ticker 不一致持股，需人工檢查")
        origin = str(row.get("origin") or "")
        if origin == PENDING_ENTRY_ORIGIN_SCANNER:
            expected_source = POSITION_SOURCE_STRATEGY_FILL
            lineage_field = "strategy_lineage"
        elif origin == PENDING_ENTRY_ORIGIN_MANUAL:
            expected_source = POSITION_SOURCE_MANUAL_MANAGED
            lineage_field = "management_lineage"
        else:
            raise RuntimeError(f"Trading pending-entry recovery 未知掛單來源: {origin or '-'}")
        if str(position.get("source") or "") != expected_source:
            raise RuntimeError("Trading pending-entry recovery 發現持股來源與掛單來源不一致，需人工檢查")
        expected_lineage_id = str((row.get("management_lineage") or {}).get("lineage_id") or "")
        actual_lineage_id = str((position.get(lineage_field) or {}).get("lineage_id") or "")
        if not expected_lineage_id or actual_lineage_id != expected_lineage_id:
            raise RuntimeError("Trading pending-entry recovery 發現 frozen lineage 不一致，需人工檢查")
        if int(broker.get("initial_qty") or 0) != int(fill.get("qty") or 0):
            raise RuntimeError("Trading pending-entry recovery 發現原始成交股數不一致，需人工檢查")
        mark_trading_pending_entry_filled(project_root, pending_entry_id=entry_id, fill=fill)
        path.unlink(missing_ok=True)
        return {"status": "RECOVERED_FILLED", "pending_entry_id": entry_id}
    # Intent was durable but account commit never happened; safe to retry from ACTIVE.
    path.unlink(missing_ok=True)
    return {"status": "ROLLED_BACK_INTENT", "pending_entry_id": entry_id}


@serialized_trading_state_mutation
def fill_trading_pending_entry(
    project_root,
    *,
    pending_entry_id: str,
    qty: int,
    price,
    trade_date,
    expected_account_revision: int | None = None,
) -> dict[str, Any]:
    recover_trading_pending_entry_transaction(project_root)
    preview = preview_trading_pending_entry_fill(
        project_root,
        pending_entry_id=pending_entry_id,
        qty=qty,
        price=price,
        trade_date=trade_date,
    )
    entry = dict(preview["entry"])
    fill_date = normalize_trading_date(trade_date, field_name="trade_date", allow_none=False)
    fill = {
        "ticker": entry["ticker"],
        "qty": int(qty),
        "price": float(price),
        "trade_date": fill_date,
        "confirmed_at": _timestamp(),
        "market_evidence": deepcopy(dict(preview.get("market_evidence") or {})),
    }
    tx_path = resolve_trading_pending_entry_transaction_path(project_root)
    _write_pending_transfer_tx(tx_path, {
        "schema_version": PENDING_ENTRY_TRANSACTION_SCHEMA_VERSION,
        "pending_entry_id": entry["pending_entry_id"],
        "ticker": entry["ticker"],
        "fill": fill,
        "created_at": _timestamp(),
    })
    try:
        lineage = dict(entry["management_lineage"])
        params = build_params_from_mapping(lineage["frozen_params"])
        seed = dict(entry["execution_plan_seed"])
        if str(entry.get("origin")) == PENDING_ENTRY_ORIGIN_SCANNER:
            account = record_strategy_trading_buy(
                project_root,
                ticker=entry["ticker"],
                qty=int(qty),
                price=float(price),
                trade_date=fill_date,
                expected_revision=expected_account_revision,
                params=params,
                execution_plan_seed=seed,
                strategy_lineage=lineage,
            )
            route = "scanner_pending_fill"
        elif str(entry.get("origin")) == PENDING_ENTRY_ORIGIN_MANUAL:
            account = record_managed_manual_trading_buy(
                project_root,
                ticker=entry["ticker"],
                qty=int(qty),
                price=float(price),
                trade_date=fill_date,
                expected_revision=expected_account_revision,
                params=params,
                execution_plan_seed=seed,
                management_lineage=lineage,
                management_start_date=fill_date,
            )
            route = "manual_pending_fill"
        else:
            raise RuntimeError(f"未知 pending entry origin: {entry.get('origin')}")
        closed = mark_trading_pending_entry_filled(
            project_root,
            pending_entry_id=entry["pending_entry_id"],
            fill=fill,
        )
        tx_path.unlink(missing_ok=True)
        return {
            "route": route,
            "account": account,
            "pending_entry": closed,
            "warnings": list(preview.get("warnings") or []),
            "market_evidence": deepcopy(dict(preview.get("market_evidence") or {})),
        }
    except Exception:
        # Keep transaction evidence only if the account commit may already have happened.
        account = load_trading_account_state(project_root, required=True)
        if normalize_trading_ticker(entry["ticker"]) not in (account.get("positions") or {}):
            tx_path.unlink(missing_ok=True)
        raise


def cancel_pending_entry_no_fill(project_root, *, pending_entry_id: str, note: str | None = None) -> dict[str, Any]:
    recover_trading_pending_entry_transaction(project_root)
    return cancel_trading_pending_entry(project_root, pending_entry_id=pending_entry_id, note=note)


def delete_pending_entry(project_root, *, pending_entry_id: str, note: str | None = None) -> dict[str, Any]:
    recover_trading_pending_entry_transaction(project_root)
    return delete_trading_pending_entry(project_root, pending_entry_id=pending_entry_id, note=note)


__all__ = [
    "PENDING_ENTRY_ORIGIN_SCANNER",
    "PENDING_ENTRY_ORIGIN_MANUAL",
    "preview_scanner_trading_pending_entry",
    "preview_manual_trading_pending_entry",
    "create_scanner_trading_pending_entry",
    "create_manual_trading_pending_entry",
    "preview_trading_pending_entry_update",
    "update_trading_pending_entry_intent",
    "get_trading_pending_entry_read_model",
    "preview_trading_pending_entry_fill",
    "fill_trading_pending_entry",
    "cancel_pending_entry_no_fill",
    "delete_pending_entry",
    "recover_trading_pending_entry_transaction",
]
