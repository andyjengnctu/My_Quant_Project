"""Read-only Trading operations readiness summary.

This module composes canonical Trading read models.  It does not download data,
train params, allocate capital, mutate account/order state, infer fills, or
submit broker orders.  The Workbench uses this single summary to explain the
current operational stage and the next legal action without re-deriving domain
rules in the UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.runtime_domains import RUNTIME_DOMAIN_TRADING
from core.trading_account_state import POSITION_SOURCE_STRATEGY_FILL
from core.trading_order_state import (
    TRADING_ACTIVE_ORDER_STATUSES,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP,
    TRADING_ORDER_PURPOSE_PROTECTION_TP,
    TRADING_ORDER_SIDE_BUY,
    TRADING_ORDER_SIDE_SELL,
)
from services.trading.account_state import (
    get_trading_account_read_model,
    load_trading_account_state,
)
from services.trading.daily_workflow import (
    build_trading_daily_workflow_snapshot,
    get_trading_candidate_snapshot_read_model,
)
from services.trading.fill_reconciliation import resolve_trading_fill_transaction_path
from services.trading.order_planning import get_trading_proposed_order_plan_read_model
from services.trading.order_state import get_trading_order_read_model
from services.trading.protection_planning import get_trading_protection_plan_read_model

TRADING_OPERATIONS_STATUS_SCHEMA_VERSION = 1

OPERATIONS_STATUS_BLOCKED = "BLOCKED"
OPERATIONS_STATUS_ACTION_REQUIRED = "ACTION_REQUIRED"
OPERATIONS_STATUS_READY = "READY"
OPERATIONS_STATUS_LOCKED_TODAY = "LOCKED_TODAY"
OPERATIONS_STATUS_IDLE = "IDLE"

NEXT_RECOVER_FILL = "RECOVER_FILL_TRANSACTION"
NEXT_INITIALIZE_ACCOUNT = "INITIALIZE_ACCOUNT"
NEXT_SET_CASH = "SET_CASH"
NEXT_REFRESH_PROTECTION = "REFRESH_PROTECTION_PLAN"
NEXT_SUBMIT_PROTECTION_STOP = "SUBMIT_PROTECTION_STOP"
NEXT_RECONCILE_ENTRY = "RECONCILE_ENTRY_ORDER"
NEXT_UPDATE_DATA = "UPDATE_DATA"
NEXT_UPDATE_PARAMS = "UPDATE_PARAMS"
NEXT_RUN_SCANNER = "RUN_SCANNER"
NEXT_DAY_LOCKED = "DAY_ALLOCATION_LOCKED"
NEXT_BUILD_PROPOSED = "BUILD_PROPOSED_ORDERS"
NEXT_SUBMIT_PROPOSED = "SUBMIT_PROPOSED_ORDER"
NEXT_NO_ENTRY = "NO_ENTRY_TODAY"
NEXT_MONITOR = "MONITOR_POSITIONS"
NEXT_READY = "READY"


def _empty_account() -> dict[str, Any]:
    return {
        "initialized": False,
        "revision": None,
        "cash": None,
        "position_count": 0,
        "positions": [],
        "updated_at": None,
    }


def _empty_orders() -> dict[str, Any]:
    return {
        "schema_version": None,
        "revision": None,
        "active_order_count": 0,
        "active_entry_order_count": 0,
        "active_protection_order_count": 0,
        "order_count": 0,
        "orders": [],
        "updated_at": None,
    }


def _empty_candidate() -> dict[str, Any]:
    return {
        "exists": False,
        "valid": False,
        "fresh": False,
        "candidate_count": 0,
        "information_date": None,
        "error": None,
        "path": None,
    }


def _empty_proposed() -> dict[str, Any]:
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
        "json_path": None,
        "text_path": None,
    }


def _empty_protection() -> dict[str, Any]:
    return {
        "exists": False,
        "fresh": False,
        "status": None,
        "broker_status": None,
        "position_count": 0,
        "positions": [],
        "manual_positions_skipped": [],
        "json_path": None,
        "text_path": None,
    }


def _workflow_action_availability(
    *,
    fill_transaction_pending: bool,
    workflow: dict[str, Any],
    account: dict[str, Any],
    candidate: dict[str, Any],
    active_entry_count: int,
    same_day_entry_locked: bool,
) -> dict[str, bool]:
    if fill_transaction_pending:
        return {"data": False, "params": False, "scanner": False, "orders": False, "all": False}
    latest_data_date = workflow.get("latest_data_date")
    account_ready = bool(account.get("initialized")) and account.get("cash") is not None
    return {
        "data": True,
        "params": bool(latest_data_date),
        "scanner": bool(workflow.get("params_ready_for_scan")),
        "orders": bool(
            account_ready
            and candidate.get("fresh")
            and active_entry_count == 0
            and not same_day_entry_locked
        ),
        "all": True,
    }


def derive_trading_operations_status(
    *,
    workflow: dict[str, Any],
    account: dict[str, Any],
    orders: dict[str, Any],
    candidate: dict[str, Any],
    proposed: dict[str, Any],
    protection: dict[str, Any],
    fill_transaction_pending: bool = False,
    component_errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Derive the operational stage from canonical read-model snapshots only."""

    errors = {str(k): str(v) for k, v in dict(component_errors or {}).items() if str(v).strip()}
    positions = [dict(row) for row in list(account.get("positions") or [])]
    strategy_tickers = sorted(
        str(row.get("ticker") or "")
        for row in positions
        if str(row.get("source") or "") == POSITION_SOURCE_STRATEGY_FILL and int(row.get("qty") or 0) > 0
    )
    manual_tickers = sorted(
        str(row.get("ticker") or "")
        for row in positions
        if str(row.get("source") or "") != POSITION_SOURCE_STRATEGY_FILL and int(row.get("qty") or 0) > 0
    )

    order_rows = [dict(row) for row in list(orders.get("orders") or [])]
    active_rows = [row for row in order_rows if str(row.get("status") or "") in TRADING_ACTIVE_ORDER_STATUSES]
    active_entry_rows = [row for row in active_rows if str(row.get("side") or "") == TRADING_ORDER_SIDE_BUY]
    active_protection_rows = [row for row in active_rows if str(row.get("side") or "") == TRADING_ORDER_SIDE_SELL]
    active_stop_tickers = sorted({
        str(row.get("ticker") or "")
        for row in active_protection_rows
        if str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_PROTECTION_STOP
    })
    active_tp_tickers = sorted({
        str(row.get("ticker") or "")
        for row in active_protection_rows
        if str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_PROTECTION_TP
    })
    missing_stop_tickers = sorted(set(strategy_tickers) - set(active_stop_tickers))

    latest_data_date = str(workflow.get("latest_data_date") or "")
    same_day_entry_locked = bool(
        latest_data_date
        and any(
            str(row.get("side") or "") == TRADING_ORDER_SIDE_BUY
            and str(row.get("information_date") or "") == latest_data_date
            for row in order_rows
        )
    )
    active_entry_count = len(active_entry_rows)
    active_protection_count = len(active_protection_rows)
    protection_fresh = bool(protection.get("exists") and protection.get("fresh"))

    availability = _workflow_action_availability(
        fill_transaction_pending=bool(fill_transaction_pending),
        workflow=workflow,
        account=account,
        candidate=candidate,
        active_entry_count=active_entry_count,
        same_day_entry_locked=same_day_entry_locked,
    )
    if "workflow" in errors:
        availability = {key: False for key in availability}

    warnings: list[str] = []
    blockers: list[str] = []
    if errors:
        warnings.extend(f"{name}: {message}" for name, message in sorted(errors.items()))
    if manual_tickers:
        warnings.append("manual adopted 未由策略接管: " + ",".join(manual_tickers))
    if proposed.get("exists") and not proposed.get("fresh"):
        warnings.append("既有建議掛單已 STALE")
    if candidate.get("exists") and not candidate.get("fresh"):
        warnings.append("既有 Scanner snapshot 已 STALE")
    if strategy_tickers and protection.get("exists") and not protection_fresh:
        warnings.append("既有保護單計畫已 STALE")

    if fill_transaction_pending:
        blockers.append("存在未完成 fill transaction，必須先完成 recovery")
        overall = OPERATIONS_STATUS_BLOCKED
        next_code = NEXT_RECOVER_FILL
        next_label = "完成成交 transaction recovery"
        next_detail = "重新整理 Trading 狀態；只允許依既有 write-ahead journal 完成 original→target recovery。"
    elif any(name in errors for name in ("workflow", "account", "orders")):
        blockers.extend(f"{name} 狀態不可讀" for name in ("workflow", "account", "orders") if name in errors)
        overall = OPERATIONS_STATUS_BLOCKED
        next_code = NEXT_READY
        next_label = "先修正 Trading state 讀取錯誤"
        next_detail = "Daily workflow／account／broker-order operational truth 必須可驗證後才可繼續交易操作。"
    elif not bool(account.get("initialized")):
        overall = OPERATIONS_STATUS_ACTION_REQUIRED
        next_code = NEXT_INITIALIZE_ACCOUNT
        next_label = "初始化 Trading account"
        next_detail = "先建立實際現金／持股 SSOT，再進行 account-aware allocation。"
    elif account.get("cash") is None:
        overall = OPERATIONS_STATUS_ACTION_REQUIRED
        next_code = NEXT_SET_CASH
        next_label = "設定 Trading 現金"
        next_detail = "Account 已建立但 cash 尚未設定，不能建立實際建議掛單。"
    elif missing_stop_tickers:
        overall = OPERATIONS_STATUS_ACTION_REQUIRED
        if protection_fresh:
            next_code = NEXT_SUBMIT_PROTECTION_STOP
            next_label = "確認未保護持股的 Stop／OCO 已送券商"
            next_detail = "缺少 active Stop: " + ",".join(missing_stop_tickers)
            tp_only = sorted(set(missing_stop_tickers) & set(active_tp_tickers))
            if tp_only:
                next_detail += "；其中已有 active TP 但無 Stop: " + ",".join(tp_only) + "，須先依實際券商狀態取消／重建或使用已確認 native OCO。"
        else:
            next_code = NEXT_REFRESH_PROTECTION
            next_label = "建立／刷新成交後保護單計畫"
            next_detail = "目前 strategy position 尚缺 active Stop: " + ",".join(missing_stop_tickers)
    elif active_entry_count:
        overall = OPERATIONS_STATUS_ACTION_REQUIRED
        next_code = NEXT_RECONCILE_ENTRY
        next_label = "確認 BUY 掛單成交或取消"
        next_detail = f"目前有 {active_entry_count} 筆 active ENTRY BUY；完成 reconciliation 前不得重新 allocation。"
    elif not workflow.get("latest_data_date"):
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_UPDATE_DATA
        next_label = "1 更新 Trading 資料"
        next_detail = "Trading dataset 尚無可用最新交易日。"
    elif not bool(workflow.get("params_ready_for_scan")):
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_UPDATE_PARAMS
        next_label = "2 更新 Trading Params"
        next_detail = "Trading Params 必須與目前 Trading data 同一最新交易日且解析為單一 runtime member。"
    elif not bool(candidate.get("fresh")):
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_RUN_SCANNER
        next_label = "3 Scanner 候選"
        next_detail = "建立綁定目前 Trading data／Params 的 fresh candidate snapshot。"
    elif same_day_entry_locked:
        overall = OPERATIONS_STATUS_LOCKED_TODAY
        next_code = NEXT_DAY_LOCKED
        next_label = "本資訊日盤前 allocation 已鎖定"
        next_detail = "本資訊日已存在實際 ENTRY BUY 送單紀錄；依 D4 不得同日重新配置資金。"
    elif not bool(proposed.get("fresh")):
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_BUILD_PROPOSED
        next_label = "4 建議掛單"
        next_detail = "以 fresh Scanner snapshot＋目前 account truth 建立盤前 account-aware allocation。"
    elif int(proposed.get("order_count") or 0) > 0:
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_SUBMIT_PROPOSED
        next_label = "依實際券商操作確認建議買單已送出"
        next_detail = f"目前有 {int(proposed.get('order_count') or 0)} 筆 fresh PROPOSED BUY；只有實際送券商後才可確認 ORDERED。"
    elif int(proposed.get("order_count") or 0) == 0:
        overall = OPERATIONS_STATUS_IDLE
        next_code = NEXT_NO_ENTRY
        next_label = "今日無可建立的新買單"
        next_detail = "Fresh account-aware allocation 已完成，沒有可執行 PROPOSED BUY。"
    elif strategy_tickers:
        overall = OPERATIONS_STATUS_IDLE
        next_code = NEXT_MONITOR
        next_label = "維持既有持股／掛單 reconciliation"
        next_detail = "目前沒有新的盤前動作；只處理實際券商成交／取消與機械保護單。"
    else:
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_READY
        next_label = "Trading 狀態可用"
        next_detail = "依每日流程與實際券商事件繼續操作。"

    return {
        "schema_version": TRADING_OPERATIONS_STATUS_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "overall_status": overall,
        "next_action_code": next_code,
        "next_action_label": next_label,
        "next_action_detail": next_detail,
        "fill_transaction_pending": bool(fill_transaction_pending),
        "latest_data_date": workflow.get("latest_data_date"),
        "params_ready_for_scan": bool(workflow.get("params_ready_for_scan")),
        "account_initialized": bool(account.get("initialized")),
        "account_revision": account.get("revision"),
        "cash": account.get("cash"),
        "strategy_position_count": len(strategy_tickers),
        "manual_position_count": len(manual_tickers),
        "strategy_tickers": strategy_tickers,
        "manual_tickers": manual_tickers,
        "active_entry_order_count": active_entry_count,
        "active_protection_order_count": active_protection_count,
        "active_stop_tickers": active_stop_tickers,
        "active_tp_tickers": active_tp_tickers,
        "missing_stop_tickers": missing_stop_tickers,
        "same_day_entry_locked": same_day_entry_locked,
        "candidate_snapshot_fresh": bool(candidate.get("fresh")),
        "candidate_count": int(candidate.get("candidate_count") or 0),
        "proposed_orders_fresh": bool(proposed.get("fresh")),
        "proposed_order_count": int(proposed.get("order_count") or 0),
        "protection_plan_fresh": protection_fresh,
        "workflow_action_availability": availability,
        "warnings": warnings,
        "blockers": blockers,
        "component_errors": errors,
        "components": {
            "workflow": workflow,
            "account": account,
            "orders": orders,
            "candidate": candidate,
            "proposed": proposed,
            "protection": protection,
        },
    }


def build_trading_operations_status(project_root: str | Path) -> dict[str, Any]:
    """Read all canonical Trading status owners and compose one Workbench summary."""

    root = Path(project_root).resolve()
    tx_path = resolve_trading_fill_transaction_path(root)
    fill_transaction_pending = tx_path.is_file()
    errors: dict[str, str] = {}

    try:
        workflow = build_trading_daily_workflow_snapshot(root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        errors["workflow"] = f"{type(exc).__name__}: {exc}"
        workflow = {
            "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "latest_data_date": None,
            "params_ready_for_scan": False,
            "param_selector": None,
        }

    try:
        candidate = get_trading_candidate_snapshot_read_model(root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        errors["candidate"] = f"{type(exc).__name__}: {exc}"
        candidate = _empty_candidate()

    if fill_transaction_pending:
        account = _empty_account()
        orders = _empty_orders()
        proposed = _empty_proposed()
        protection = _empty_protection()
    else:
        try:
            state = load_trading_account_state(root, required=False)
            if state is None:
                account = _empty_account()
            else:
                account = {"initialized": True, **get_trading_account_read_model(root)}
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            errors["account"] = f"{type(exc).__name__}: {exc}"
            account = _empty_account()

        try:
            orders = get_trading_order_read_model(root)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            errors["orders"] = f"{type(exc).__name__}: {exc}"
            orders = _empty_orders()

        try:
            proposed = get_trading_proposed_order_plan_read_model(root)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            errors["proposed"] = f"{type(exc).__name__}: {exc}"
            proposed = _empty_proposed()

        try:
            protection = get_trading_protection_plan_read_model(root, recover_pending_fill=False)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            errors["protection"] = f"{type(exc).__name__}: {exc}"
            protection = _empty_protection()

    return derive_trading_operations_status(
        workflow=workflow,
        account=account,
        orders=orders,
        candidate=candidate,
        proposed=proposed,
        protection=protection,
        fill_transaction_pending=fill_transaction_pending,
        component_errors=errors,
    )


__all__ = [
    "TRADING_OPERATIONS_STATUS_SCHEMA_VERSION",
    "OPERATIONS_STATUS_BLOCKED",
    "OPERATIONS_STATUS_ACTION_REQUIRED",
    "OPERATIONS_STATUS_READY",
    "OPERATIONS_STATUS_LOCKED_TODAY",
    "OPERATIONS_STATUS_IDLE",
    "NEXT_RECOVER_FILL",
    "NEXT_INITIALIZE_ACCOUNT",
    "NEXT_SET_CASH",
    "NEXT_REFRESH_PROTECTION",
    "NEXT_SUBMIT_PROTECTION_STOP",
    "NEXT_RECONCILE_ENTRY",
    "NEXT_UPDATE_DATA",
    "NEXT_UPDATE_PARAMS",
    "NEXT_RUN_SCANNER",
    "NEXT_DAY_LOCKED",
    "NEXT_BUILD_PROPOSED",
    "NEXT_SUBMIT_PROPOSED",
    "NEXT_NO_ENTRY",
    "NEXT_MONITOR",
    "NEXT_READY",
    "derive_trading_operations_status",
    "build_trading_operations_status",
]
