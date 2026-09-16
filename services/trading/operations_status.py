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

from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS

from core.runtime_domains import RUNTIME_DOMAIN_TRADING
from core.trading_account_state import POSITION_SOURCE_STRATEGY_FILL
from core.trading_order_state import (
    TRADING_ACTIVE_ORDER_STATUSES,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
    TRADING_ORDER_PURPOSE_PROTECTION_TP,
    TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
    TRADING_ORDER_SIDE_BUY,
    TRADING_ORDER_SIDE_SELL,
)
from core.trading_stop_exit_progress import is_trading_stop_exit_triggered_from_order_rows
from services.trading.account_state import (
    get_trading_account_read_model,
    load_trading_account_state,
)
from services.trading.daily_workflow import (
    build_trading_daily_workflow_snapshot,
    get_trading_candidate_snapshot_read_model,
)
from services.trading.fill_reconciliation import resolve_trading_fill_transaction_path
from services.trading.proposed_order_state import get_trading_proposed_order_plan_read_model
from services.trading.indicator_exit_planning import get_trading_indicator_exit_plan_read_model
from services.trading.order_state import get_trading_order_read_model
from services.trading.position_rollforward import build_trading_position_rollforward_snapshot
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
NEXT_ROLLFORWARD_POSITIONS = "ROLLFORWARD_POSITIONS"
NEXT_REPLACE_PROTECTION = "REPLACE_STALE_PROTECTION"
NEXT_REFRESH_PROTECTION = "REFRESH_PROTECTION_PLAN"
NEXT_SUBMIT_PROTECTION_STOP = "SUBMIT_PROTECTION_STOP"
NEXT_REFRESH_STOP_REMAINDER = "REFRESH_STOP_REMAINDER_PLAN"
NEXT_SUBMIT_STOP_REMAINDER = "SUBMIT_STOP_REMAINDER_MARKET"
NEXT_RECONCILE_STOP_REMAINDER = "RECONCILE_STOP_REMAINDER"
NEXT_CANCEL_SELL_FOR_STOP_REMAINDER = "CANCEL_SELL_FOR_STOP_REMAINDER"
NEXT_REFRESH_INDICATOR_EXIT = "REFRESH_INDICATOR_EXIT_PLAN"
NEXT_CANCEL_PROTECTION_FOR_INDICATOR = "CANCEL_PROTECTION_FOR_INDICATOR_EXIT"
NEXT_SUBMIT_INDICATOR_EXIT = "SUBMIT_INDICATOR_EXIT"
NEXT_RECONCILE_INDICATOR_EXIT = "RECONCILE_INDICATOR_EXIT"
NEXT_RECONCILE_ENTRY = "RECONCILE_ENTRY_ORDER"
NEXT_RECONCILE_ORPHAN_SELL = "RECONCILE_ORPHAN_SELL_ORDER"
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
        "latest_buy_trade_date": None,
        "latest_sell_trade_date": None,
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
        "candidate_tickers": [],
        "scanned_ticker_count": 0,
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
        "stale_active_protection_order_ids": [],
        "stale_active_protection_tickers": [],
        "json_path": None,
        "text_path": None,
    }


def _empty_indicator_exit() -> dict[str, Any]:
    return {
        "exists": False, "fresh": False, "status": None, "exit_count": 0, "exits": [],
        "active_indicator_exit_order_count": 0, "active_indicator_exit_tickers": [],
        "json_path": None, "text_path": None,
    }


def _empty_position_rollforward() -> dict[str, Any]:
    return {
        "schema_version": None,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "allowed_completed_date": None,
        "strategy_position_count": 0,
        "due_count": 0,
        "due_tickers": [],
        "positions": [],
    }


def _workflow_action_availability(
    *,
    fill_transaction_pending: bool,
    workflow: dict[str, Any],
    account: dict[str, Any],
    candidate: dict[str, Any],
    active_entry_count: int,
    active_indicator_count: int,
    indicator_due_count: int,
    same_day_entry_locked: bool,
    same_session_sell_locked: bool,
    rollforward_due_count: int,
    stale_active_protection_count: int,
    forced_stop_exit_count: int,
) -> dict[str, bool]:
    if fill_transaction_pending:
        return {"data": False, "rollforward": False, "params": False, "scanner": False, "orders": False, "all": False}
    latest_data_date = workflow.get("latest_data_date")
    trading_data_ready = bool(
        workflow.get("trading_data_ready", workflow.get("market_data_ready", bool(latest_data_date)))
    )
    account_ready = bool(account.get("initialized")) and account.get("cash") is not None
    return {
        "data": True,
        "rollforward": bool(account_ready and latest_data_date and rollforward_due_count > 0),
        "params": bool(latest_data_date and trading_data_ready),
        "scanner": bool(trading_data_ready and workflow.get("params_ready_for_scan")),
        # Primary Trading is not a broker OMS.  Legacy active/stale order state
        # cannot block account-aware allocation; only actual fills/account state,
        # strategy decisions and session locks may do so.
        "orders": bool(
            account_ready
            and candidate.get("fresh")
            and indicator_due_count == 0
            and rollforward_due_count == 0
            and not same_day_entry_locked
            and not same_session_sell_locked
        ),
        "all": bool(account_ready and indicator_due_count == 0),
    }


def derive_trading_operations_status(
    *,
    workflow: dict[str, Any],
    account: dict[str, Any],
    orders: dict[str, Any],
    candidate: dict[str, Any],
    proposed: dict[str, Any],
    protection: dict[str, Any],
    indicator_exit: dict[str, Any] | None = None,
    position_rollforward: dict[str, Any] | None = None,
    fill_transaction_pending: bool = False,
    component_errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Derive the operational stage from canonical read-model snapshots only."""

    errors = {str(k): str(v) for k, v in dict(component_errors or {}).items() if str(v).strip()}
    position_rollforward = dict(position_rollforward or _empty_position_rollforward())
    indicator_exit = dict(indicator_exit or _empty_indicator_exit())
    positions = [dict(row) for row in list(account.get("positions") or [])]
    strategy_lineage_by_ticker = {
        str(row.get("ticker") or ""): str(row.get("strategy_lineage_key") or row.get("entry_order_id") or "")
        for row in positions
        if str(row.get("source") or "") == POSITION_SOURCE_STRATEGY_FILL
        and int(row.get("qty") or 0) > 0
        and str(row.get("ticker") or "")
        and str(row.get("strategy_lineage_key") or row.get("entry_order_id") or "")
    }
    legacy_oms_lineage_by_ticker = {
        str(row.get("ticker") or ""): str(row.get("entry_order_id") or "")
        for row in positions
        if str(row.get("source") or "") == POSITION_SOURCE_STRATEGY_FILL
        and int(row.get("qty") or 0) > 0
        and str(row.get("ticker") or "")
        and str(row.get("entry_order_id") or "")
    }
    strategy_tickers = sorted(strategy_lineage_by_ticker)
    legacy_oms_tickers = sorted(legacy_oms_lineage_by_ticker)
    manual_tickers = sorted(
        str(row.get("ticker") or "")
        for row in positions
        if str(row.get("source") or "") != POSITION_SOURCE_STRATEGY_FILL and int(row.get("qty") or 0) > 0
    )

    order_rows = [dict(row) for row in list(orders.get("orders") or [])]
    active_rows = [row for row in order_rows if str(row.get("status") or "") in TRADING_ACTIVE_ORDER_STATUSES]
    active_entry_rows = [row for row in active_rows if str(row.get("side") or "") == TRADING_ORDER_SIDE_BUY]
    active_sell_rows = [row for row in active_rows if str(row.get("side") or "") == TRADING_ORDER_SIDE_SELL]

    def _matches_current_strategy_lineage(row: dict[str, Any]) -> bool:
        ticker = str(row.get("ticker") or "")
        return bool(
            ticker
            and ticker in strategy_lineage_by_ticker
            and str(row.get("entry_order_id") or "") == strategy_lineage_by_ticker[ticker]
        )

    current_active_sell_rows = [row for row in active_sell_rows if _matches_current_strategy_lineage(row)]
    orphan_active_sell_rows = [row for row in active_sell_rows if not _matches_current_strategy_lineage(row)]
    active_protection_rows = [
        row for row in current_active_sell_rows
        if str(row.get("purpose") or "") in {
            TRADING_ORDER_PURPOSE_PROTECTION_STOP,
            TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
            TRADING_ORDER_PURPOSE_PROTECTION_TP,
        }
    ]
    active_indicator_rows = [
        row for row in current_active_sell_rows
        if str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_INDICATOR_EXIT
    ]

    stop_triggered_by_ticker = {
        ticker: is_trading_stop_exit_triggered_from_order_rows(
            order_rows,
            ticker=ticker,
            entry_order_id=strategy_lineage_by_ticker[ticker],
            compatible_entry_order_ids=(
                [legacy_oms_lineage_by_ticker[ticker]]
                if legacy_oms_lineage_by_ticker.get(ticker) else None
            ),
        )
        for ticker in strategy_tickers
    }
    active_stop_tickers = sorted({
        str(row.get("ticker") or "")
        for row in active_protection_rows
        if str(row.get("purpose") or "") in {TRADING_ORDER_PURPOSE_PROTECTION_STOP, TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER}
    })
    forced_stop_exit_tickers = sorted(
        ticker for ticker, triggered in stop_triggered_by_ticker.items() if triggered
    )
    active_stop_exit_tickers = sorted(set(forced_stop_exit_tickers) & set(active_stop_tickers))
    active_tp_tickers = sorted({
        str(row.get("ticker") or "")
        for row in active_protection_rows
        if str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_PROTECTION_TP
    })
    missing_stop_tickers = sorted(set(legacy_oms_tickers) - set(active_stop_tickers))

    latest_data_date = str(workflow.get("latest_data_date") or "")
    same_day_entry_locked = bool(
        latest_data_date and str(account.get("latest_buy_trade_date") or "") == latest_data_date
    )
    active_entry_count = len(active_entry_rows)
    active_protection_count = len(active_protection_rows)
    protection_fresh = bool(protection.get("exists") and protection.get("fresh"))
    indicator_plan_fresh = bool(indicator_exit.get("exists") and indicator_exit.get("fresh"))
    indicator_due_rows = [
        dict(row) for row in list(indicator_exit.get("exits") or [])
        if _matches_current_strategy_lineage(dict(row))
    ] if indicator_plan_fresh else []
    indicator_due_tickers = sorted({str(row.get("ticker") or "") for row in indicator_due_rows if str(row.get("ticker") or "")})
    active_indicator_tickers = sorted({
        str(row.get("ticker") or "")
        for row in active_indicator_rows
        if str(row.get("ticker") or "") and _matches_current_strategy_lineage(row)
    })
    indicator_protection_conflict_tickers = sorted(set(indicator_due_tickers) & {str(row.get("ticker") or "") for row in active_protection_rows})
    forced_stop_conflict_tickers = sorted(
        set(forced_stop_exit_tickers)
        & (set(active_tp_tickers) | set(active_indicator_tickers))
    )
    forced_stop_unsubmitted_tickers = sorted(set(forced_stop_exit_tickers) - set(active_stop_exit_tickers))
    protection_forced_tickers = sorted({
        str(row.get("ticker") or "")
        for row in list(protection.get("positions") or [])
        if bool(row.get("stop_forced_exit")) and str(row.get("ticker") or "")
    })
    active_indicator_count = len(active_indicator_rows)
    orphan_active_sell_order_ids = sorted(str(row.get("order_id") or "") for row in orphan_active_sell_rows if str(row.get("order_id") or ""))
    orphan_active_sell_tickers = sorted({str(row.get("ticker") or "") for row in orphan_active_sell_rows if str(row.get("ticker") or "")})
    same_session_sell_locked = bool(
        latest_data_date
        and str(account.get("latest_sell_trade_date") or "")
        and str(account.get("latest_sell_trade_date") or "") > latest_data_date
    )
    rollforward_due_tickers = sorted(str(x) for x in list(position_rollforward.get("due_tickers") or []))
    stale_active_protection_order_ids = [str(x) for x in list(protection.get("stale_active_protection_order_ids") or []) if str(x)]
    stale_active_protection_tickers = sorted(str(x) for x in list(protection.get("stale_active_protection_tickers") or []))
    missing_stop_tickers = sorted(set(missing_stop_tickers) - set(indicator_due_tickers) - set(active_indicator_tickers) - set(forced_stop_exit_tickers))

    # Workbench is not the broker OMS.  Legacy broker-order inconsistencies stay
    # visible as warnings but do not block Scanner/allocation for position-lineage
    # managed accounts.  Strategy Stop/Target/SELL obligations are decision truth.
    sell_coverage_blockers: list[str] = []
    open_position_sell_coverage_safe = True

    market_data_ready = bool(workflow.get("market_data_ready", bool(workflow.get("latest_data_date"))))
    trading_data_ready = bool(workflow.get("trading_data_ready", market_data_ready))
    execution_pool_stats = dict(workflow.get("current_execution_pool_stats") or {})
    listed_ticker_count = execution_pool_stats.get("listed_count")
    if listed_ticker_count is not None:
        listed_ticker_count = int(listed_ticker_count)
    quick_filter_qualified_count = execution_pool_stats.get("qualified_count")
    if quick_filter_qualified_count is None:
        quick_filter_qualified_count = workflow.get("current_execution_pool_ticker_count")
    if quick_filter_qualified_count is not None:
        quick_filter_qualified_count = int(quick_filter_qualified_count)
    candidate_snapshot_available = bool(candidate.get("exists")) and bool(candidate.get("valid"))
    scanner_candidate_ticker_count = (
        int(candidate.get("candidate_count") or 0)
        if candidate_snapshot_available
        else None
    )
    open_position_tickers = {
        str(row.get("ticker") or "").strip().upper()
        for row in positions
        if int(row.get("qty") or 0) > 0 and str(row.get("ticker") or "").strip()
    }
    candidate_tickers = {
        str(ticker or "").strip().upper()
        for ticker in list(candidate.get("candidate_tickers") or [])
        if str(ticker or "").strip()
    }
    unheld_candidate_tickers = candidate_tickers - open_position_tickers
    portfolio_free_slot_count = max(0, int(DEFAULT_PORTFOLIO_MAX_POSITIONS) - len(open_position_tickers))
    scanner_buyable_ticker_count = None
    if (
        candidate_snapshot_available
        and bool(candidate.get("fresh"))
        and bool(account.get("initialized"))
        and account.get("cash") is not None
    ):
        scanner_buyable_ticker_count = min(len(unheld_candidate_tickers), portfolio_free_slot_count)
    trading_data_blockers = [str(item) for item in list(workflow.get("trading_data_blockers") or []) if str(item)]

    allocation_blockers: list[str] = []
    if fill_transaction_pending:
        allocation_blockers.append("存在未完成 fill transaction")
    if errors:
        allocation_blockers.append("Trading component state 不可完整驗證")
    if not bool(account.get("initialized")) or account.get("cash") is None:
        allocation_blockers.append("Trading account/cash 尚未就緒")
    if not trading_data_ready:
        allocation_blockers.append(
            "Trading data readiness 尚未就緒"
            + (("：" + "；".join(trading_data_blockers)) if trading_data_blockers else "")
        )
    if rollforward_due_tickers:
        allocation_blockers.append("strategy position 尚未完成日終推進")
    if same_day_entry_locked or same_session_sell_locked:
        allocation_blockers.append("本交易 session allocation 已鎖定")

    # ENTRY submission is a different lifecycle step from creating a new
    # allocation.  Existing same-plan ENTRY orders and the information-date
    # allocation lock must not prevent submitting the remaining orders that
    # were already frozen in the same PROPOSED plan.  All open-position exit
    # safety, component integrity, and plan freshness requirements still apply.
    entry_submission_blockers: list[str] = []
    if fill_transaction_pending:
        entry_submission_blockers.append("存在未完成 fill transaction")
    if errors:
        entry_submission_blockers.append("Trading component state 不可完整驗證")
    if not bool(account.get("initialized")) or account.get("cash") is None:
        entry_submission_blockers.append("Trading account/cash 尚未就緒")
    if not trading_data_ready:
        entry_submission_blockers.append(
            "Trading data readiness 尚未就緒"
            + (("：" + "；".join(trading_data_blockers)) if trading_data_blockers else "")
        )
    if orphan_active_sell_order_ids:
        entry_submission_blockers.append("存在 orphan active SELL order")
    if forced_stop_exit_tickers:
        entry_submission_blockers.append("存在 STOP forced-exit obligation")
    if rollforward_due_tickers:
        entry_submission_blockers.append("strategy position 尚未完成日終推進")
    if strategy_tickers and not indicator_plan_fresh:
        entry_submission_blockers.append("Indicator SELL plan 尚未 fresh")
    if active_indicator_count or indicator_due_tickers or indicator_protection_conflict_tickers:
        entry_submission_blockers.append("Indicator SELL lifecycle 尚未完成")
    if stale_active_protection_order_ids or missing_stop_tickers:
        entry_submission_blockers.append("open position protection 尚未安全完成")
    if not bool(proposed.get("fresh")) or int(proposed.get("order_count") or 0) <= 0:
        entry_submission_blockers.append("目前沒有可送出的 fresh PROPOSED BUY")

    availability = _workflow_action_availability(
        fill_transaction_pending=bool(fill_transaction_pending),
        workflow=workflow,
        account=account,
        candidate=candidate,
        active_entry_count=active_entry_count,
        active_indicator_count=active_indicator_count,
        indicator_due_count=len(indicator_due_tickers),
        same_day_entry_locked=same_day_entry_locked,
        same_session_sell_locked=same_session_sell_locked,
        rollforward_due_count=len(rollforward_due_tickers),
        stale_active_protection_count=len(stale_active_protection_order_ids),
        forced_stop_exit_count=len(forced_stop_exit_tickers),
    )
    if "workflow" in errors:
        availability = {key: False for key in availability}
    if "position_rollforward" in errors:
        availability["rollforward"] = False
        availability["orders"] = False
        availability["all"] = False
    if allocation_blockers:
        availability["orders"] = False
        availability["all"] = False

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
    if active_entry_count:
        warnings.append(f"Legacy broker OMS 尚有 {active_entry_count} 筆 active ENTRY BUY（不阻擋 primary Trading；以實際 account fill 為準）")
    if orphan_active_sell_order_ids:
        warnings.append("Legacy broker OMS 有未對應目前 position lineage 的 active SELL（不阻擋決策流程）: " + ",".join(orphan_active_sell_tickers))
    if stale_active_protection_order_ids:
        warnings.append("Legacy broker protection 已與目前 position plan 不一致（不阻擋決策流程）: " + ",".join(stale_active_protection_tickers))
    if same_session_sell_locked:
        warnings.append("本交易 session 已確認 SELL fill；Trading completed data 尚未追上成交日，依 D3/D4 禁止重新 allocation")
    if forced_stop_exit_tickers:
        warnings.append("STOP 已觸發且仍有剩餘持股，退出義務不得恢復為等待觸價 Stop: " + ",".join(forced_stop_exit_tickers))
    v2_archive_status = str(workflow.get("market_data_v2_archive_status") or "")
    if v2_archive_status not in {"", "SYNCED", "NOT_BOOTSTRAPPED"}:
        detail = str(workflow.get("market_data_v2_archive_error") or "").strip()
        warnings.append(
            "Market Data V2 archive sidecar 尚未同步；目前 active rule-based execution 不受阻擋"
            + (f": {detail}" if detail else "")
        )

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
    elif forced_stop_exit_tickers:
        overall = OPERATIONS_STATUS_ACTION_REQUIRED
        next_code = NEXT_SUBMIT_STOP_REMAINDER
        next_label = "依 Stop 退出訊號自行至券商處理"
        next_detail = "STOP 已觸發且仍有剩餘持股：" + ",".join(forced_stop_exit_tickers) + "；Workbench 不管理券商掛單，成交後回帳務中心登錄。"
    elif not workflow.get("latest_data_date") or not trading_data_ready:
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_UPDATE_DATA
        next_label = "1 更新 Trading 資料"
        next_detail = (
            "Trading data dependency readiness 尚未就緒；"
            + ("；".join(trading_data_blockers) if trading_data_blockers else "先由 canonical data workflow 更新／驗證資料。")
        )
    elif "position_rollforward" in errors:
        overall = OPERATIONS_STATUS_BLOCKED
        next_code = NEXT_ROLLFORWARD_POSITIONS
        next_label = "修正第 2 步持股日終推進狀態"
        next_detail = errors["position_rollforward"]
    elif rollforward_due_tickers:
        overall = OPERATIONS_STATUS_ACTION_REQUIRED
        next_code = NEXT_ROLLFORWARD_POSITIONS
        next_label = "2 持股日終推進"
        next_detail = "用已完成日K與各持股 immutable strategy lineage frozen params 更新下一交易日 trailing stop: " + ",".join(rollforward_due_tickers)
    elif "indicator_exit" in errors:
        overall = OPERATIONS_STATUS_BLOCKED
        next_code = NEXT_REFRESH_INDICATOR_EXIT
        next_label = "修正／刷新 Indicator SELL 計畫"
        next_detail = errors["indicator_exit"]
    elif strategy_tickers and not indicator_plan_fresh:
        overall = OPERATIONS_STATUS_ACTION_REQUIRED
        next_code = NEXT_REFRESH_INDICATOR_EXIT
        next_label = "建立／刷新 Indicator SELL 計畫"
        next_detail = "依最新 completed bar 與各持股 frozen params 判定 persistent indicator exit obligation。"
    elif indicator_due_tickers:
        overall = OPERATIONS_STATUS_ACTION_REQUIRED
        next_code = NEXT_SUBMIT_INDICATOR_EXIT
        next_label = "依 Indicator SELL 訊號自行至券商賣出"
        next_detail = "策略 SELL decision：" + ",".join(indicator_due_tickers) + "；Workbench 不管理券商掛單，成交後回帳務中心登錄。"
    elif not bool(workflow.get("params_ready_for_scan")):
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_UPDATE_PARAMS
        next_label = "3 套用 Trading Params"
        if bool(workflow.get("params_reusable")):
            next_detail = "可選擇沿用既有 Params 綁定至目前 Trading data，或重新訓練 Params；兩者皆須解析為單一 runtime member。"
        else:
            next_detail = "目前沒有可沿用的合法 Params；請先重新訓練並綁定目前 Trading data。"
    elif not bool(candidate.get("fresh")):
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_RUN_SCANNER
        next_label = "4 Scanner 候選"
        next_detail = "建立綁定目前 Trading data／Params 的 fresh candidate snapshot。"
    elif same_session_sell_locked:
        overall = OPERATIONS_STATUS_LOCKED_TODAY
        next_code = NEXT_DAY_LOCKED
        next_label = "本交易 session 資金重新配置已鎖定"
        next_detail = "已確認 SELL fill 的成交日晚於目前 completed information date；依 D3/D4 不得把盤中釋放資金重新配置。"
    elif same_day_entry_locked:
        overall = OPERATIONS_STATUS_LOCKED_TODAY
        next_code = NEXT_DAY_LOCKED
        next_label = "本資訊日盤前 allocation 已鎖定"
        next_detail = "本資訊日已存在實際 ENTRY BUY 送單紀錄；依 D4 不得同日重新配置資金。"
    elif not bool(proposed.get("fresh")):
        overall = OPERATIONS_STATUS_READY
        next_code = NEXT_BUILD_PROPOSED
        next_label = "5 建議掛單"
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
        next_label = "維持既有持股／每日決策監控"
        next_detail = "目前沒有新的盤前動作；依持股決策表的 Stop / 停利線 / SELL signal 自行處理券商交易。"
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
        "market_data_ready": market_data_ready,
        "trading_data_ready": trading_data_ready,
        "trading_data_readiness_status": workflow.get("trading_data_readiness_status"),
        "trading_data_dependency_fingerprint": workflow.get("trading_data_dependency_fingerprint"),
        "trading_data_required_v2_count": workflow.get("trading_data_required_v2_count"),
        "trading_data_ready_v2_count": workflow.get("trading_data_ready_v2_count"),
        "trading_data_blockers": trading_data_blockers,
        "market_data_consumer_state_sha256": workflow.get("market_data_consumer_state_sha256"),
        "market_data_source_view_fingerprint": workflow.get("market_data_source_view_fingerprint"),
        "market_data_v2_archive_status": workflow.get("market_data_v2_archive_status"),
        "market_data_v2_archive_latest_date": workflow.get("market_data_v2_archive_latest_date"),
        "market_data_v2_archive_error": workflow.get("market_data_v2_archive_error"),
        "listed_ticker_count": listed_ticker_count,
        "quick_filter_qualified_count": quick_filter_qualified_count,
        "scanner_candidate_ticker_count": scanner_candidate_ticker_count,
        "scanner_buyable_ticker_count": scanner_buyable_ticker_count,
        "scanner_unheld_candidate_ticker_count": (len(unheld_candidate_tickers) if candidate_snapshot_available else None),
        "portfolio_free_slot_count": (portfolio_free_slot_count if bool(account.get("initialized")) else None),
        "portfolio_max_positions": int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        "scanner_information_date": candidate.get("information_date"),
        "strategy_id": workflow.get("strategy_id"),
        "param_selector": workflow.get("param_selector"),
        "param_training_data_date": workflow.get("param_latest_data_date"),
        "param_member_count": int(workflow.get("param_member_count") or 0),
        "param_min_agree": int(workflow.get("param_min_agree") or 0),
        "params_ready_for_scan": bool(workflow.get("params_ready_for_scan")),
        "params_reusable": bool(workflow.get("params_reusable")),
        "param_usage_mode": workflow.get("param_usage_mode"),
        "param_error": workflow.get("param_error"),
        "param_binding_error": workflow.get("param_binding_error"),
        "account_initialized": bool(account.get("initialized")),
        "account_revision": account.get("revision"),
        "cash": account.get("cash"),
        "strategy_position_count": len(strategy_tickers),
        "manual_position_count": len(manual_tickers),
        "strategy_tickers": strategy_tickers,
        "manual_tickers": manual_tickers,
        "active_entry_order_count": active_entry_count,
        "active_protection_order_count": active_protection_count,
        "active_indicator_exit_order_count": active_indicator_count,
        "active_indicator_exit_tickers": active_indicator_tickers,
        "orphan_active_sell_order_ids": orphan_active_sell_order_ids,
        "orphan_active_sell_tickers": orphan_active_sell_tickers,
        "indicator_exit_due_count": len(indicator_due_tickers),
        "indicator_exit_due_tickers": indicator_due_tickers,
        "indicator_protection_conflict_tickers": indicator_protection_conflict_tickers,
        "active_stop_tickers": active_stop_tickers,
        "active_tp_tickers": active_tp_tickers,
        "forced_stop_exit_count": len(forced_stop_exit_tickers),
        "forced_stop_exit_tickers": forced_stop_exit_tickers,
        "active_stop_exit_tickers": active_stop_exit_tickers,
        "forced_stop_unsubmitted_tickers": forced_stop_unsubmitted_tickers,
        "forced_stop_conflict_tickers": forced_stop_conflict_tickers,
        "missing_stop_tickers": missing_stop_tickers,
        "rollforward_due_count": len(rollforward_due_tickers),
        "rollforward_due_tickers": rollforward_due_tickers,
        "stale_active_protection_order_ids": stale_active_protection_order_ids,
        "stale_active_protection_tickers": stale_active_protection_tickers,
        "same_day_entry_locked": same_day_entry_locked,
        "same_session_sell_locked": same_session_sell_locked,
        "candidate_snapshot_fresh": bool(candidate.get("fresh")),
        "candidate_count": int(candidate.get("candidate_count") or 0),
        "proposed_orders_fresh": bool(proposed.get("fresh")),
        "proposed_order_count": int(proposed.get("order_count") or 0),
        "protection_plan_fresh": protection_fresh,
        "indicator_exit_plan_fresh": indicator_plan_fresh,
        "workflow_action_availability": availability,
        "allocation_allowed": bool(availability.get("orders")),
        "allocation_blockers": allocation_blockers,
        "entry_submission_allowed": not entry_submission_blockers,
        "entry_submission_blockers": entry_submission_blockers,
        "open_position_sell_coverage_safe": open_position_sell_coverage_safe,
        "open_position_sell_coverage_blockers": sell_coverage_blockers,
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
            "indicator_exit": indicator_exit,
            "position_rollforward": position_rollforward,
        },
    }


def assert_trading_new_allocation_allowed(status: dict[str, Any]) -> None:
    """Fail closed unless canonical Operations Status permits Step-4 allocation."""

    if bool((status.get("workflow_action_availability") or {}).get("orders")):
        return
    reasons = [str(item) for item in list(status.get("allocation_blockers") or []) if str(item)]
    if not reasons:
        reasons = [str(status.get("next_action_detail") or status.get("next_action_label") or "Trading state 尚未允許 allocation")]
    raise RuntimeError("Trading 目前禁止建立新的盤前建議掛單：" + "；".join(reasons))


def assert_trading_proposed_submission_allowed(status: dict[str, Any]) -> None:
    """Fail closed unless a previously frozen PROPOSED BUY may be submitted."""

    if bool(status.get("entry_submission_allowed")):
        return
    reasons = [str(item) for item in list(status.get("entry_submission_blockers") or []) if str(item)]
    if not reasons:
        reasons = [str(status.get("next_action_detail") or status.get("next_action_label") or "Trading state 尚未允許 ENTRY submission")]
    raise RuntimeError("Trading 目前禁止確認新的 ENTRY BUY 送單：" + "；".join(reasons))



def derive_trading_operations_status_from_preloaded(
    *,
    workflow: dict[str, Any] | None,
    account: dict[str, Any] | None,
    orders: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    proposed: dict[str, Any] | None,
    protection: dict[str, Any] | None,
    indicator_exit: dict[str, Any] | None,
    position_rollforward: dict[str, Any] | None,
    fill_transaction_pending: bool = False,
    component_errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Derive Operations Status from one already-read refresh generation.

    This is an execution-only seam for Workbench initial load.  It preserves the
    same canonical derivation while avoiding a second pass over identical state
    files and V2 lineage.
    """

    workflow_value = dict(workflow or {
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "latest_data_date": None,
        "market_data_ready": False,
        "trading_data_ready": False,
        "trading_data_blockers": ["Trading workflow state 無法讀取"],
        "params_ready_for_scan": False,
        "param_selector": None,
    })
    return derive_trading_operations_status(
        workflow=workflow_value,
        account=dict(account or _empty_account()),
        orders=dict(orders or _empty_orders()),
        candidate=dict(candidate or _empty_candidate()),
        proposed=dict(proposed or _empty_proposed()),
        protection=dict(protection or _empty_protection()),
        indicator_exit=dict(indicator_exit or _empty_indicator_exit()),
        position_rollforward=dict(position_rollforward or _empty_position_rollforward()),
        fill_transaction_pending=bool(fill_transaction_pending),
        component_errors=component_errors,
    )


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
            "market_data_ready": False,
            "trading_data_ready": False,
            "trading_data_blockers": ["Trading workflow state 無法讀取"],
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
        indicator_exit = _empty_indicator_exit()
        position_rollforward = _empty_position_rollforward()
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
            position_rollforward = build_trading_position_rollforward_snapshot(root)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            errors["position_rollforward"] = f"{type(exc).__name__}: {exc}"
            position_rollforward = _empty_position_rollforward()

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

        try:
            indicator_exit = get_trading_indicator_exit_plan_read_model(root, recover_pending_fill=False)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            errors["indicator_exit"] = f"{type(exc).__name__}: {exc}"
            indicator_exit = _empty_indicator_exit()

    return derive_trading_operations_status(
        workflow=workflow,
        account=account,
        orders=orders,
        candidate=candidate,
        proposed=proposed,
        protection=protection,
        indicator_exit=indicator_exit,
        position_rollforward=position_rollforward,
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
    "NEXT_ROLLFORWARD_POSITIONS",
    "NEXT_REPLACE_PROTECTION",
    "NEXT_REFRESH_PROTECTION",
    "NEXT_SUBMIT_PROTECTION_STOP",
    "NEXT_REFRESH_STOP_REMAINDER",
    "NEXT_SUBMIT_STOP_REMAINDER",
    "NEXT_RECONCILE_STOP_REMAINDER",
    "NEXT_CANCEL_SELL_FOR_STOP_REMAINDER",
    "NEXT_REFRESH_INDICATOR_EXIT",
    "NEXT_CANCEL_PROTECTION_FOR_INDICATOR",
    "NEXT_SUBMIT_INDICATOR_EXIT",
    "NEXT_RECONCILE_INDICATOR_EXIT",
    "NEXT_RECONCILE_ENTRY",
    "NEXT_RECONCILE_ORPHAN_SELL",
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
    "derive_trading_operations_status_from_preloaded",
    "assert_trading_new_allocation_allowed",
    "assert_trading_proposed_submission_allowed",
    "build_trading_operations_status",
]
