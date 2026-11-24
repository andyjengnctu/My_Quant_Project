from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from services.trading.daily_workflow import TRADING_PARAM_MODE_REUSE, TRADING_PARAM_MODE_TRAIN
from services.trading.workbench_snapshot import (
    build_trading_account_panel_snapshot as _build_account_snapshot,
    build_trading_account_panel_initial_bundle as _build_initial_bundle,
    _capture_initial_panel_value as capture_panel_value,
)

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


# Compatibility exports for existing panel consumers; orchestration is service-owned.
def build_trading_account_panel_snapshot(project_root=WORKBENCH_PROJECT_ROOT):
    return _build_account_snapshot(project_root)


def build_trading_account_panel_initial_bundle(project_root=WORKBENCH_PROJECT_ROOT):
    return _build_initial_bundle(project_root)


def _capture_initial_panel_value(loader):
    return capture_panel_value(loader)


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
