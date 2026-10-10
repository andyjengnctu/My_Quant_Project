from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from config.trading import TRADING_WORKBENCH_INITIAL_READ_WORKERS
from core.console_report import project_relative_display_path
from core.trading_policy import get_trading_policy_snapshot
from services.trading.daily_workflow import TRADING_PARAM_MODE_REUSE, TRADING_PARAM_MODE_TRAIN, build_trading_daily_workflow_snapshot
from services.trading.proposed_order_state import get_trading_proposed_order_plan_read_model
from services.trading.scanner_state import get_trading_candidate_snapshot_read_model, load_trading_candidate_snapshot_for_account
from services.trading.position_rollforward import build_trading_position_rollforward_snapshot, reconcile_trading_manual_position_management
from services.trading.lifecycle_sync import run_trading_lifecycle_sync
from services.trading.pending_entry_service import get_trading_pending_entry_read_model
from services.trading.operations_status import derive_trading_operations_status_from_preloaded
from services.trading.market_data_consumer_promotion import reconcile_trading_v2_consumer_state_from_local_evidence
from services.trading.protection_planning import get_trading_protection_plan_read_model
from services.trading.indicator_exit_planning import get_trading_indicator_exit_plan_read_model
from services.trading.fill_reconciliation import resolve_trading_fill_transaction_path, recover_trading_fill_transaction
from services.trading.order_state import get_trading_order_read_model
from services.trading.account_dashboard import build_trading_account_dashboard_read_model, publish_trading_account_dashboard_snapshot
from services.trading.account_state import get_trading_account_read_model, load_trading_account_state, resolve_trading_account_state_path
from services.workbench_ui.workbench import run_workbench_local_reconcile_serialized

WORKBENCH_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_SOURCE = "manual_adopted"
UNMANAGED_STATUS = "unmanaged"
PARAM_MODE_REUSE_LABEL = "沿用既有 Params"
PARAM_MODE_TRAIN_LABEL = "重新訓練 Params"
PARAM_MODE_BY_LABEL = {
    PARAM_MODE_REUSE_LABEL: TRADING_PARAM_MODE_REUSE,
    PARAM_MODE_TRAIN_LABEL: TRADING_PARAM_MODE_TRAIN,
}

WORKFLOW_HINT = "每日操作：1 選擇／核對策略 → 2 選擇 Params 模式並訓練／套用 → 3 掃描候選；資料更新由資料中心負責。Trading 狀態刷新會自動同步掛單／持股 lifecycle，既有部位沿用 frozen params；券商操作由使用者自行完成。"
CASH_HINT = "初始化可留空；更新現金會留下 revision event，不直接改檔。"
MANUAL_POSITION_HINT = "修正／移除只適用尚未有賣出歷史、尚未由策略接管的 manual adopted 持股；不改 cash。"
FILL_HINT = "成交只接受券商實際股數／價格；PARTIAL 仍鎖定未成交餘額，FILLED 才解除 active order。"
PROPOSED_ORDER_HINT = "建議掛單的停利線是盤前策略參考：新訊號／再進場為停利線參考；延續／延續(TBD) 沿用 inherited shadow completion barrier。是否真的建立券商 TP，成交後仍只由該 entry order frozen params 的 tp_percent 決定；tp_percent=0 時不會建立 TP 券商單。"
PROTECTION_HINT = "只由 confirmed managed fill 的 canonical position state＋frozen params 機械派生；不讀成交後行情、不代表券商已掛出 Stop/TP。"
OCO_HINT = "系統不預設券商支援 OCO；只有你明確輸入實際券商 OCO/互斥群組 ID 時才允許 Stop full + TP 同時超額共享同一持股。尚未送券商的 logical plan 仍不是 broker truth。"
INDICATOR_HINT = "Signal 只由 completed bar + source entry frozen params 產生；計畫不是券商送單，實際成交仍須在掛單表輸入 broker fill。"
POSITION_DECISION_HINT = "左側 ▣ 可直接開啟單股回測檢視；停損／停利／賣出訊號是持股區決策資訊。"
SCANNER_HINT = "左側 ▣ 可直接開啟單股回測檢視；選取候選後會帶入掛單輸入框，可在正式送出前修改規劃股數／掛單日；買入限價由系統自動計算。"
BUY_ENTRY_HINT = "直接補登買入｜不經掛單區；成交日／成交價會先用 raw 市場證據檢查，手動股會凍結目前 Primary Params 並從 finalized information date 開始管理。"
PENDING_ENTRY_HINT = "掛單輸入｜輸入手動股票或從 Scanner Pool 選取；修改股票、規劃股數或掛單日會自動更新預覽。買入限價、預留成本、停損、停利由系統自動計算；選取既有掛單後可直接填成交價／成交日並確認成交；只有 ACTIVE 掛單占用 Params、資金與 slot。"



def holding_order_date_by_lineage(pending_entries) -> dict[tuple[str, str], str]:
    """Map FILLED pending lineage + fill date to the persisted order date."""

    result: dict[tuple[str, str], str] = {}
    for raw in pending_entries or ():
        row = dict(raw or {})
        if str(row.get("status") or "") != "FILLED":
            continue
        lineage = row.get("management_lineage")
        if not isinstance(lineage, dict):
            continue
        lineage_id = str(lineage.get("lineage_id") or "").strip()
        planned_trade_date = str(row.get("planned_trade_date") or "").strip()
        fill_date = str((row.get("fill") or {}).get("trade_date") or "").strip()
        if lineage_id and fill_date and planned_trade_date:
            result[(lineage_id, fill_date)] = planned_trade_date
    return result



def _pending_sync_failure_at_cell(tree, rows, event):
    """Resolve diagnostics by logical column, without selecting or mutating intent."""
    if event is None or str(tree.identify_region(event.x, event.y) or "") != "cell":
        return None
    row = rows.get(str(tree.identify_row(event.y) or ""))
    if not row or not row.get("sync_error"):
        return None
    column = tree.identify_column(event.x)
    return row if str(tree.column(column, "id")) == "status" else None



def parse_trading_money_text(raw_value, field_name: str, *, allow_blank: bool = False, allow_zero: bool = True):
    text = str(raw_value or "").strip().replace(",", "")
    if not text:
        if allow_blank:
            return None
        raise ValueError(f"{field_name} 必填")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} 必須是數字") from exc
    if not value.is_finite():
        raise ValueError(f"{field_name} 必須是有限數值")
    if value < 0 or (value == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{field_name} 必須 {comparator}")
    return value


def parse_trading_qty_text(raw_value, field_name: str = "持股數量") -> int:
    text = str(raw_value or "").strip().replace(",", "")
    if not text:
        raise ValueError(f"{field_name} 必填")
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} 必須是整數") from exc
    if value <= 0:
        raise ValueError(f"{field_name} 必須 > 0")
    return value


def format_trading_money(value) -> str:
    if value is None:
        return "未設定"
    return f"{float(value):,.3f}".rstrip("0").rstrip(".")


def build_trading_account_panel_snapshot(project_root=WORKBENCH_PROJECT_ROOT) -> dict[str, object]:
    recover_trading_fill_transaction(project_root)
    policy = get_trading_policy_snapshot(project_root)
    state = load_trading_account_state(project_root, required=False)
    account_path = resolve_trading_account_state_path(project_root)
    if state is None:
        return {
            "initialized": False,
            "state_path": project_relative_display_path(account_path, project_root=project_root),
            "policy": policy,
            "revision": None,
            "cash": None,
            "position_count": 0,
            "positions": [],
            "updated_at": None,
        }
    read_model = get_trading_account_read_model(project_root)
    return {
        "initialized": True,
        "state_path": project_relative_display_path(account_path, project_root=project_root),
        "policy": policy,
        **read_model,
    }


def _capture_initial_panel_value(loader):
    try:
        return True, loader()
    except Exception as exc:
        return False, exc


def build_trading_account_panel_initial_bundle(project_root=WORKBENCH_PROJECT_ROOT) -> dict[str, object]:
    """Read one coherent initial Trading page bundle without duplicate state I/O."""

    root = Path(project_root).resolve()
    bundle: dict[str, object] = {}
    # Consumer promotion can change the finalized market-data identity, so it
    # remains the only sequential prerequisite. Everything after this point is
    # read-only against the same finalized generation.
    def _reconcile_initial_state():
        market_data = run_workbench_local_reconcile_serialized(
            lambda: reconcile_trading_v2_consumer_state_from_local_evidence(root)
        )
        manual_management = reconcile_trading_manual_position_management(root)
        return {"market_data": market_data, "manual_management": manual_management}

    bundle["reconcile"] = _capture_initial_panel_value(_reconcile_initial_state)
    # Lifecycle state must be synchronized before any Trading read model is
    # rendered.  This is the canonical automatic replacement for the old
    # user-driven position rollforward step and also advances ACTIVE pending
    # plans with their own frozen lineage.
    bundle["lifecycle_sync"] = _capture_initial_panel_value(
        lambda: run_trading_lifecycle_sync(root)
    )
    # Account recovery historically ran before protection/indicator readers. Keep
    # that ordering, then parallelize independent canonical read models.
    bundle["account"] = _capture_initial_panel_value(lambda: build_trading_account_panel_snapshot(root))

    def _load_dashboard():
        snapshot = build_trading_account_dashboard_read_model(root)
        publish_trading_account_dashboard_snapshot(root, snapshot)
        return snapshot

    loaders = {
        # Display-only stock names are warmed independently by the panel and must
        # never gate the canonical Trading first paint.  A cold TaiwanStockInfo
        # verification can be much slower than the account/scanner read models.
        "dashboard": _load_dashboard,
        "candidate_read": lambda: get_trading_candidate_snapshot_read_model(root),
        "protection": lambda: get_trading_protection_plan_read_model(root, recover_pending_fill=False),
        "indicator": lambda: get_trading_indicator_exit_plan_read_model(root, recover_pending_fill=False),
        "workflow": lambda: build_trading_daily_workflow_snapshot(root),
        "orders": lambda: get_trading_order_read_model(root),
        "proposed": lambda: get_trading_proposed_order_plan_read_model(root),
        "position_rollforward": lambda: build_trading_position_rollforward_snapshot(root),
        "pending": lambda: get_trading_pending_entry_read_model(root),
    }
    worker_count = max(1, min(int(TRADING_WORKBENCH_INITIAL_READ_WORKERS), len(loaders)))
    parallel_results: dict[str, tuple[bool, object]] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="workbench-trading-read") as executor:
        futures = {
            key: executor.submit(_capture_initial_panel_value, loader)
            for key, loader in loaders.items()
        }
        for key, future in futures.items():
            parallel_results[key] = future.result()

    for key in ("dashboard", "candidate_read", "protection", "indicator", "workflow", "pending"):
        bundle[key] = parallel_results[key]

    candidate = bundle["candidate_read"]
    if candidate[0] and bool(candidate[1].get("fresh")):
        bundle["candidate_payload"] = _capture_initial_panel_value(
            lambda: load_trading_candidate_snapshot_for_account(root, require_current=False)
        )
    else:
        bundle["candidate_payload"] = (True, None)

    component_errors: dict[str, str] = {}

    def _preloaded_value(bundle_key: str, operations_key: str):
        if bundle_key == "account":
            ok, value = bundle["account"]
        else:
            ok, value = parallel_results[bundle_key]
        if ok:
            return value
        component_errors[operations_key] = f"{type(value).__name__}: {value}"
        return None

    fill_transaction_pending = Path(resolve_trading_fill_transaction_path(root)).is_file()
    preloaded_operations = {
        "fill_transaction_pending": fill_transaction_pending,
        "workflow": _preloaded_value("workflow", "workflow"),
        "candidate": _preloaded_value("candidate_read", "candidate"),
        "account": _preloaded_value("account", "account"),
        "position_rollforward": _preloaded_value("position_rollforward", "position_rollforward"),
        "orders": _preloaded_value("orders", "orders"),
        "proposed": _preloaded_value("proposed", "proposed"),
        "protection": _preloaded_value("protection", "protection"),
        "indicator_exit": _preloaded_value("indicator", "indicator_exit"),
    }
    bundle["operations"] = _capture_initial_panel_value(
        lambda: derive_trading_operations_status_from_preloaded(
            workflow=preloaded_operations["workflow"],
            account=preloaded_operations["account"],
            orders=preloaded_operations["orders"],
            candidate=preloaded_operations["candidate"],
            proposed=preloaded_operations["proposed"],
            protection=preloaded_operations["protection"],
            indicator_exit=preloaded_operations["indicator_exit"],
            position_rollforward=preloaded_operations["position_rollforward"],
            pending_entries=_preloaded_value("pending", "pending_entries"),
            fill_transaction_pending=bool(preloaded_operations["fill_transaction_pending"]),
            component_errors=component_errors,
        )
    )
    return bundle


def _load_panel_value(panel, key: str, loader):
    cache = getattr(panel, "_initial_preloaded", None)
    if isinstance(cache, dict) and key in cache:
        ok, value = cache.pop(key)
        if not ok:
            raise value
        return value
    return loader()


def _parse_candidate_date(value):
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None


def _candidate_signal_age_days(signal_date, candidate_date):
    signal = _parse_candidate_date(signal_date)
    candidate = _parse_candidate_date(candidate_date)
    if signal is None or candidate is None or candidate < signal:
        return None
    return int((candidate - signal).days)


def _candidate_signal_date_text(value):
    parsed = _parse_candidate_date(value)
    return "-" if parsed is None else parsed.isoformat()
