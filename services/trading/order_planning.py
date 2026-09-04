"""Trading account-aware proposed-order planning using canonical portfolio reservation semantics."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS, DEFAULT_PORTFOLIO_ROTATION
from core.capital_policy import resolve_portfolio_sizing_equity
from core.console_report import project_relative_display_path
from core.data_utils import discover_unique_csv_map, sanitize_ohlcv_dataframe
from core.entry_plans import resize_candidate_plan_to_capital
from core.exact_accounting import (
    build_buy_ledger_from_price,
    calc_average_price_from_total_milli,
    milli_to_money,
)
from core.file_integrity import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    compute_file_sha256,
    load_json_strict,
)
from core.portfolio_entry_selection import (
    build_reserved_candidate_order_plan,
    reorder_candidates_for_resource_aware_quality,
    select_resource_aware_action_candidates,
)
from core.portfolio_fast_data import calc_mark_to_market_equity, pack_static_market_data
from core.runtime_domains import (
    RUNTIME_DOMAIN_TRADING,
    resolve_runtime_domain_paths,
    resolve_runtime_output_dir,
)
from core.trading_order_state import (
    TRADING_ORDER_STATE_FILENAME,
    active_trading_entry_orders,
    TRADING_ORDER_SIDE_BUY,
    validate_trading_order_state,
)
from services.trading.account_state import load_trading_account_state
from services.trading.daily_workflow import (
    load_trading_candidate_snapshot,
    load_trading_scanner_runtime,
    resolve_trading_candidate_snapshot_path,
)

PROPOSED_ORDER_SCHEMA_VERSION = 2
PROPOSED_ORDER_STATUS = "PROPOSED"


def resolve_trading_proposed_orders_dir(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return Path(resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="proposed_orders"))


def resolve_trading_proposed_orders_json_path(project_root: str | Path) -> Path:
    return resolve_trading_proposed_orders_dir(project_root) / "proposed_orders.json"


def resolve_trading_proposed_orders_text_path(project_root: str | Path) -> Path:
    return resolve_trading_proposed_orders_dir(project_root) / "proposed_orders.txt"


def _resolve_trading_order_state_path(project_root: str | Path) -> Path:
    paths = resolve_runtime_domain_paths(project_root, domain=RUNTIME_DOMAIN_TRADING)
    if paths.state_root is None:
        raise RuntimeError("Trading runtime domain 缺少 state_root")
    return Path(paths.state_root) / TRADING_ORDER_STATE_FILENAME


def _assert_order_state_allows_new_allocation(project_root: str | Path, *, information_date: str) -> None:
    order_state_path = _resolve_trading_order_state_path(project_root)
    if not order_state_path.is_file():
        return
    state = load_json_strict(order_state_path)
    validate_trading_order_state(state)
    if active_trading_entry_orders(state):
        raise RuntimeError("Trading 尚有 ORDERED pending orders（ENTRY BUY ORDERED/PARTIAL）；完成成交／取消 reconciliation 前禁止建立新的盤前建議掛單")
    if any(
        str(row.get("side") or "") == TRADING_ORDER_SIDE_BUY
        and str(row.get("information_date") or "") == str(information_date)
        for row in state.get("orders", {}).values()
    ):
        raise RuntimeError("Trading 本資訊日已存在實際 BUY 送單紀錄；依盤前資金鎖定原則禁止同日重新 allocation")


def load_current_trading_proposed_order_plan(
    project_root: str | Path,
    *,
    require_current: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = resolve_trading_proposed_orders_json_path(root)
    if not path.is_file():
        raise FileNotFoundError("Trading 建議掛單尚未產生；請先執行「4 建議掛單」。")
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise RuntimeError("Trading proposed-order payload 不合法")
    if int(payload.get("schema_version", -1)) != PROPOSED_ORDER_SCHEMA_VERSION:
        raise RuntimeError("Trading proposed-order schema_version 不相容，請重新產生建議掛單")
    if str(payload.get("status") or "") != PROPOSED_ORDER_STATUS:
        raise RuntimeError("Trading proposed-order status 不合法")
    if str(payload.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise RuntimeError("Trading proposed-order runtime domain 不合法")
    payload_core = {key: value for key, value in payload.items() if key != "plan_fingerprint"}
    expected_fingerprint = canonical_json_sha256(payload_core)
    if str(payload.get("plan_fingerprint") or "") != expected_fingerprint:
        raise RuntimeError("Trading proposed-order plan fingerprint 不一致")
    if not require_current:
        return payload

    runtime = load_trading_scanner_runtime(root)
    if str(payload.get("information_date") or "") != str(runtime["latest_data_date"]):
        raise RuntimeError("Trading 建議掛單資料日期已過期；請重新執行 3 Scanner 與 4 建議掛單")
    if str(payload.get("strategy_id") or "") != str(runtime["profile"].strategy_id):
        raise RuntimeError("Trading 建議掛單策略與目前設定不一致")
    if str(payload.get("param_selector") or "") != str(runtime["profile"].param_selector):
        raise RuntimeError("Trading 建議掛單 selector 與目前設定不一致")
    if str(payload.get("selected_params_sha256") or "") != compute_file_sha256(runtime["selected_path"]):
        raise RuntimeError("Trading 建議掛單對應的 params 已改變；請重新執行 Scanner／建議掛單")
    snapshot_path = resolve_trading_candidate_snapshot_path(root)
    if not snapshot_path.is_file() or str(payload.get("candidate_snapshot_sha256") or "") != compute_file_sha256(snapshot_path):
        raise RuntimeError("Trading 建議掛單對應的 Scanner snapshot 已改變；請重新執行建議掛單")
    account = load_trading_account_state(root, required=True)
    if int(payload.get("account_revision", -1)) != int(account["revision"]):
        raise RuntimeError("Trading account 已與建議掛單使用的 revision 不一致；請重新產生建議掛單")
    return payload


def get_trading_proposed_order_plan_read_model(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    json_path = resolve_trading_proposed_orders_json_path(root)
    text_path = resolve_trading_proposed_orders_text_path(root)
    if not json_path.is_file():
        return {
            "exists": False,
            "valid": False,
            "fresh": False,
            "status": None,
            "information_date": None,
            "order_count": 0,
            "account_revision": None,
            "plan_fingerprint": None,
            "error": None,
            "json_path": project_relative_display_path(json_path, project_root=root),
            "text_path": project_relative_display_path(text_path, project_root=root),
        }
    try:
        payload = load_current_trading_proposed_order_plan(root, require_current=False)
    except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        return {
            "exists": True,
            "valid": False,
            "fresh": False,
            "status": None,
            "information_date": None,
            "order_count": 0,
            "account_revision": None,
            "plan_fingerprint": None,
            "error": f"{type(exc).__name__}: {exc}",
            "json_path": project_relative_display_path(json_path, project_root=root),
            "text_path": project_relative_display_path(text_path, project_root=root),
        }
    freshness_error = None
    try:
        load_current_trading_proposed_order_plan(root, require_current=True)
    except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        freshness_error = f"{type(exc).__name__}: {exc}"
    return {
        "exists": True,
        "valid": True,
        "fresh": freshness_error is None,
        "status": payload.get("status"),
        "information_date": payload.get("information_date"),
        "order_count": len(payload.get("orders") or []),
        "account_revision": payload.get("account_revision"),
        "plan_fingerprint": payload.get("plan_fingerprint"),
        "reserved_total": payload.get("reserved_total"),
        "error": freshness_error,
        "json_path": project_relative_display_path(json_path, project_root=root),
        "text_path": project_relative_display_path(text_path, project_root=root),
    }


def _position_security_profile(record: dict[str, Any]):
    management = record.get("strategy_management") or {}
    state = management.get("position_state")
    if isinstance(state, dict):
        return state.get("security_profile")
    return None


def _build_account_mark_inputs(*, data_dir: Path, state: dict[str, Any], information_date: str):
    csv_map, duplicate_issues = discover_unique_csv_map(data_dir)
    if duplicate_issues:
        raise RuntimeError("Trading dataset 持股估值存在重複 ticker CSV，禁止在實盤 allocator 靜默選檔")

    target_date = pd.Timestamp(information_date)
    portfolio: dict[str, dict[str, Any]] = {}
    all_dfs_fast: dict[str, dict[pd.Timestamp, dict[str, float]]] = {}
    marks: list[dict[str, Any]] = []

    for ticker, record in sorted((state.get("positions") or {}).items()):
        broker = record.get("broker") or {}
        qty = int(broker.get("qty") or 0)
        if qty <= 0:
            continue
        file_path = csv_map.get(str(ticker))
        if not file_path:
            raise FileNotFoundError(f"Trading 持股 {ticker} 缺少市場資料，禁止估算 sizing equity")
        raw = pd.read_csv(file_path)
        df, _stats = sanitize_ohlcv_dataframe(raw, str(ticker), min_rows=1)
        usable = df.loc[df.index <= target_date]
        if usable.empty:
            raise RuntimeError(f"Trading 持股 {ticker} 在 {information_date} 前沒有合法收盤價")
        mark_date = pd.Timestamp(usable.index[-1])
        mark_close = float(usable["Close"].iloc[-1])
        remaining_cost_milli = int(broker.get("remaining_cost_basis_milli") or 0)
        average_cost = calc_average_price_from_total_milli(remaining_cost_milli, qty)
        portfolio[str(ticker)] = {
            "ticker": str(ticker),
            "qty": qty,
            "entry": average_cost,
            "pure_buy_price": average_cost,
            "last_px": mark_close,
            "security_profile": _position_security_profile(record),
        }
        all_dfs_fast[str(ticker)] = pack_static_market_data(usable)
        marks.append({
            "ticker": str(ticker),
            "qty": qty,
            "mark_date": mark_date.date().isoformat(),
            "mark_close": mark_close,
        })
    return portfolio, all_dfs_fast, marks


def _build_allocator_candidate(row: dict[str, Any], *, sizing_equity: float, params) -> dict[str, Any] | None:
    seed = row.get("execution_plan_seed")
    if not isinstance(seed, dict):
        raise RuntimeError(f"Trading Scanner 候選缺少 canonical execution_plan_seed: {row.get('ticker')}")
    candidate_plan = dict(seed)
    candidate_plan["sizing_capital"] = float(sizing_equity)
    resized = resize_candidate_plan_to_capital(candidate_plan, float(sizing_equity), params)
    if resized is None or int(resized.get("qty") or 0) <= 0:
        return None

    ticker = str(row.get("ticker") or resized.get("ticker") or "").strip()
    if not ticker:
        raise RuntimeError("Trading Scanner 候選缺少 ticker")
    qty = int(resized["qty"])
    limit_price = float(resized["limit_price"])
    ledger = build_buy_ledger_from_price(limit_price, qty, params)
    kind = str(row.get("kind") or "")
    candidate_type = "normal" if kind == "buy" else kind
    return {
        "ticker": ticker,
        "type": candidate_type,
        "kind": kind,
        "limit_px": limit_price,
        "init_sl": resized.get("init_sl"),
        "init_trail": resized.get("init_trail"),
        "target_price": resized.get("target_price"),
        "entry_atr": resized.get("entry_atr"),
        "security_profile": resized.get("security_profile"),
        "trade_date": resized.get("trade_date"),
        "qty": qty,
        "proj_cost_milli": int(ledger["net_buy_total_milli"]),
        "proj_cost": milli_to_money(int(ledger["net_buy_total_milli"])),
        "sizing_capital": float(sizing_equity),
        "max_qty": resized.get("max_qty"),
        "orig_limit": resized.get("orig_limit", limit_price),
        "orig_atr": resized.get("orig_atr", resized.get("entry_atr")),
        "entry_source": resized.get("entry_source", candidate_type),
        "is_orderable": True,
        "params_obj": params,
        "sort_value": row.get("sort_value"),
        "ev": row.get("expected_value", row.get("ev")),
        "expected_value": row.get("expected_value", row.get("ev")),
        "use_breakout_quality_ranking": bool(row.get("use_breakout_quality_ranking", False)),
        "breakout_quality_ranking_policy": row.get("breakout_quality_ranking_policy"),
        "breakout_quality_ranking_options": dict(row.get("breakout_quality_ranking_options") or {}),
    }


def _render_proposed_orders_text(payload: dict[str, Any]) -> str:
    lines = [
        "Trading Proposed Orders",
        "=" * 80,
        f"Status        : {payload['status']}",
        f"Information   : {payload['information_date']}",
        f"Account rev   : {payload['account_revision']}",
        f"Cash          : {payload['cash']:,.2f}",
        f"Sizing equity : {payload['sizing_equity']:,.2f}",
        f"Positions     : {payload['occupied_positions']}/{payload['max_positions']}",
        f"Free slots    : {payload['free_slots']}",
        f"Reserved      : {payload['reserved_total']:,.2f}",
        f"Cash remain   : {payload['cash_after_reservation']:,.2f}",
        "",
        "注意：這是盤前建議掛單，不代表已送單或已成交。",
        "",
    ]
    orders = list(payload.get("orders") or [])
    if not orders:
        lines.append("今日沒有可建立的建議買單。")
    else:
        lines.append("#  股票      類型          限價      股數       預留資金       Stop       Target")
        for row in orders:
            lines.append(
                f"{int(row['rank']):>2} {str(row['ticker']):<8} {str(row['kind']):<12} "
                f"{float(row['limit_price']):>9.2f} {int(row['qty']):>8,d} "
                f"{float(row['reserved_cost']):>13,.2f} "
                f"{float(row['init_sl']):>10.2f} {float(row['target_price']):>11.2f}"
            )
    return "\n".join(lines) + "\n"


def build_trading_proposed_order_plan(*, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    runtime = load_trading_scanner_runtime(root)
    _assert_order_state_allows_new_allocation(root, information_date=str(runtime["latest_data_date"]))
    snapshot_path = resolve_trading_candidate_snapshot_path(root)
    snapshot = load_trading_candidate_snapshot(root, require_current=True)
    current_param_sha = compute_file_sha256(runtime["selected_path"])

    state = load_trading_account_state(root, required=True)
    if state.get("cash_milli") is None:
        raise RuntimeError("Trading cash 尚未設定，不能建立建議掛單")
    account_revision = int(state["revision"])
    cash = milli_to_money(int(state["cash_milli"]))
    max_positions = int(DEFAULT_PORTFOLIO_MAX_POSITIONS)
    if str(DEFAULT_PORTFOLIO_ROTATION).strip().lower() != "off":
        raise RuntimeError("Trading proposed-order allocator 本輪只支援 canonical Rotation=off")
    occupied = len(state.get("positions") or {})
    free_slots = max(0, max_positions - occupied)

    portfolio, all_dfs_fast, marks = _build_account_mark_inputs(
        data_dir=Path(runtime["data_dir"]),
        state=state,
        information_date=str(runtime["latest_data_date"]),
    )
    sizing_equity = float(
        calc_mark_to_market_equity(
            cash,
            portfolio,
            all_dfs_fast,
            pd.Timestamp(runtime["latest_data_date"]),
            runtime["params"],
        )
    )
    sizing_equity = float(
        resolve_portfolio_sizing_equity(
            sizing_equity,
            float(runtime["params"].initial_capital),
            runtime["params"],
        )
    )

    held_tickers = {str(ticker) for ticker in (state.get("positions") or {}).keys()}
    allocator_candidates: list[dict[str, Any]] = []
    skipped_held: list[str] = []
    skipped_zero_qty: list[str] = []
    for raw_row in list(snapshot.get("candidate_rows") or []):
        row = dict(raw_row)
        ticker = str(row.get("ticker") or "").strip()
        if ticker in held_tickers:
            skipped_held.append(ticker)
            continue
        candidate = _build_allocator_candidate(row, sizing_equity=sizing_equity, params=runtime["params"])
        if candidate is None:
            skipped_zero_qty.append(ticker)
            continue
        allocator_candidates.append(candidate)

    resource_order, resource_diag = reorder_candidates_for_resource_aware_quality(
        allocator_candidates,
        available_cash=cash,
        sizing_equity=sizing_equity,
        pre_market_occupied=occupied,
        max_positions=max_positions,
        params=runtime["params"],
    )
    action_candidates = select_resource_aware_action_candidates(resource_order, resource_diag)
    reservation = build_reserved_candidate_order_plan(
        action_candidates,
        available_cash=cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=runtime["params"],
    )

    orders = []
    for rank, (row, plan) in enumerate(zip(reservation["selected_rows"], reservation["selected_plans"]), 1):
        orders.append({
            "rank": rank,
            "ticker": str(row["ticker"]),
            "kind": str(row.get("kind") or row.get("type") or ""),
            "limit_price": float(plan["limit_price"]),
            "qty": int(plan["qty"]),
            "reserved_cost_milli": int(plan["reserved_cost_milli"]),
            "reserved_cost": float(plan["reserved_cost"]),
            "init_sl": float(plan["init_sl"]),
            "init_trail": float(plan["init_trail"]),
            "target_price": float(plan["target_price"]),
            "entry_atr": None if plan.get("entry_atr") is None else float(plan["entry_atr"]),
            "entry_type": str(row.get("type") or row.get("kind") or "normal"),
            "security_profile": deepcopy(plan.get("security_profile")),
            "sort_value": row.get("sort_value"),
            "expected_value": row.get("expected_value", row.get("ev")),
        })

    latest_state = load_trading_account_state(root, required=True)
    if int(latest_state["revision"]) != account_revision:
        raise RuntimeError("Trading account 在建立建議掛單期間已變更；禁止使用 stale allocation，請重跑")

    reserved_total_milli = int(reservation["reserved_cost_milli"])
    payload_core = {
        "schema_version": PROPOSED_ORDER_SCHEMA_VERSION,
        "status": PROPOSED_ORDER_STATUS,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": runtime["profile"].strategy_id,
        "param_selector": runtime["profile"].param_selector,
        "information_date": str(runtime["latest_data_date"]),
        "account_revision": account_revision,
        "selected_params_sha256": current_param_sha,
        "candidate_snapshot_sha256": compute_file_sha256(snapshot_path),
        "max_positions": max_positions,
        "rotation": str(DEFAULT_PORTFOLIO_ROTATION),
        "occupied_positions": occupied,
        "free_slots": free_slots,
        "cash_milli": int(state["cash_milli"]),
        "cash": cash,
        "sizing_equity": sizing_equity,
        "reserved_total_milli": reserved_total_milli,
        "reserved_total": milli_to_money(reserved_total_milli),
        "cash_after_reservation_milli": int(reservation["remaining_cash_milli"]),
        "cash_after_reservation": milli_to_money(int(reservation["remaining_cash_milli"])),
        "held_tickers_skipped": sorted(set(skipped_held)),
        "zero_qty_candidates_skipped": sorted(set(skipped_zero_qty)),
        "position_marks": marks,
        "orders": orders,
        "account_mutated": False,
        "confirmed": False,
    }
    payload = {**payload_core, "plan_fingerprint": canonical_json_sha256(payload_core)}
    json_path = resolve_trading_proposed_orders_json_path(root)
    text_path = resolve_trading_proposed_orders_text_path(root)
    atomic_write_json(json_path, payload)
    atomic_write_text(text_path, _render_proposed_orders_text(payload))
    return {
        **deepcopy(payload),
        "json_path": project_relative_display_path(json_path, project_root=root),
        "text_path": project_relative_display_path(text_path, project_root=root),
    }


__all__ = [
    "PROPOSED_ORDER_SCHEMA_VERSION",
    "PROPOSED_ORDER_STATUS",
    "resolve_trading_proposed_orders_dir",
    "resolve_trading_proposed_orders_json_path",
    "resolve_trading_proposed_orders_text_path",
    "load_current_trading_proposed_order_plan",
    "get_trading_proposed_order_plan_read_model",
    "build_trading_proposed_order_plan",
]
