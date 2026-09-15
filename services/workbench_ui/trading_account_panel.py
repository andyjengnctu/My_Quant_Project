from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path
import queue
import re
import threading
import tkinter as tk
from tkinter import font as tkfont, messagebox, ttk

from config.trading import TRADING_WORKBENCH_INITIAL_READ_WORKERS
from core.console_report import project_relative_display_path
from core.trading_policy import get_trading_policy_snapshot
from core.trading_order_state import (
    TRADING_ACTIVE_ORDER_STATUSES,
    TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
    TRADING_ORDER_SIDE_BUY,
)
from services.trading.daily_workflow import (
    TRADING_PARAM_MODE_REUSE,
    TRADING_PARAM_MODE_TRAIN,
    build_trading_daily_workflow_snapshot,
    run_trading_candidate_scan,
    run_trading_daily_workflow,
    run_trading_market_data_update,
    run_trading_param_step,
)
from services.downloader.daily_console_progress import MarketDataDailyConsoleProgress
from services.trading.strategy_param_state import (
    TRADING_PARAM_USAGE_REUSE_EXISTING,
    TRADING_PARAM_USAGE_TRAINED_CURRENT,
)
from services.trading.order_planning import build_trading_proposed_order_plan
from services.trading.proposed_order_state import (
    get_trading_proposed_order_plan_read_model,
    load_current_trading_proposed_order_plan,
)
from services.trading.scanner_state import (
    get_trading_candidate_snapshot_read_model,
    load_trading_candidate_snapshot,
)
from services.trading.position_rollforward import build_trading_position_rollforward_snapshot, run_trading_position_rollforward
from services.trading.operations_status import build_trading_operations_status, derive_trading_operations_status_from_preloaded
from services.trading.market_data_consumer_promotion import reconcile_trading_v2_consumer_state_from_local_evidence
from services.trading.operational_audit import run_trading_operational_audit
from services.trading.protection_planning import (
    PROTECTION_STOP_REMAINDER_ACTION,
    build_trading_protection_plan,
    get_trading_protection_plan_read_model,
)
from services.trading.indicator_exit_planning import (
    build_trading_indicator_exit_plan,
    get_trading_indicator_exit_plan_read_model,
)
from services.trading.indicator_exit_order_submission import confirm_trading_indicator_exit_submission
from services.trading.protection_order_submission import (
    confirm_trading_protection_leg_submission,
    confirm_trading_protection_oco_submission,
)
from services.trading.entry_order_submission import confirm_trading_order_submission
from services.trading.fill_reconciliation import (
    TradingFillRevisionConflict,
    resolve_trading_fill_transaction_path,
    confirm_trading_buy_order_fill,
    confirm_trading_protection_sell_order_fill,
    confirm_trading_indicator_sell_order_fill,
    recover_trading_fill_transaction,
)
from services.trading.order_state import (
    TradingOrderRevisionConflict,
    confirm_trading_order_cancellation,
    get_trading_order_read_model,
)
from services.trading.account_dashboard import (
    build_trading_account_dashboard_read_model,
    publish_trading_account_dashboard_snapshot,
)
from services.trading.account_state import (
    TradingAccountRevisionConflict,
    adopt_existing_trading_position,
    correct_existing_trading_position,
    get_trading_account_read_model,
    initialize_trading_account_state,
    load_trading_account_state,
    remove_existing_trading_position,
    record_manual_trading_buy,
    resolve_trading_account_state_path,
    set_trading_cash_balance,
)
from services.workbench_ui.date_picker import DatePickerField
from services.workbench_ui.paged_table import PagedTable, TableColumn
from services.trading.account_trade_entry import record_trading_account_buy
from services.workbench_ui.workbench import (
    WORKBENCH_BG,
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_COMBO_STYLE,
    WORKBENCH_ERROR,
    WORKBENCH_ENTRY_STYLE,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_HSCROLL_STYLE,
    WORKBENCH_INFO,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_SUCCESS,
    WORKBENCH_TEXT,
    WORKBENCH_TREE_STYLE,
    WORKBENCH_UI_FONT,
    WORKBENCH_VSCROLL_STYLE,
    WORKBENCH_WARNING,
    _warn_gui_fallback,
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

WORKFLOW_HINT = "更新資料後，strategy_fill 持股用 position strategy_lineage 的 frozen params 做日終推進；Scanner 只接受已綁定目前 Trading data 的 Params。Stop/Target/SELL 為決策資訊，券商操作由使用者自行完成。"
CASH_HINT = "初始化可留空；更新現金會留下 revision event，不直接改檔。"
MANUAL_POSITION_HINT = "修正／移除只適用尚未有賣出歷史、尚未由策略接管的 manual adopted 持股；不改 cash。"
FILL_HINT = "成交只接受券商實際股數／價格；PARTIAL 仍鎖定未成交餘額，FILLED 才解除 active order。"
PROPOSED_ORDER_HINT = "建議掛單的 Target / 完成線是盤前策略參考：新訊號／再進場為 Target 參考；延續／延續(TBD) 為 inherited shadow completion barrier。是否真的建立券商 TP，成交後仍只由該 entry order frozen params 的 tp_percent 決定；tp_percent=0 時不會建立 TP 券商單。"
PROTECTION_HINT = "只由 confirmed strategy fill 的 canonical position state＋ORDERED 時 frozen params 機械派生；不讀成交後行情、不代表券商已掛出 Stop/TP。"
OCO_HINT = "系統不預設券商支援 OCO；只有你明確輸入實際券商 OCO/互斥群組 ID 時才允許 Stop full + TP 同時超額共享同一持股。尚未送券商的 logical plan 仍不是 broker truth。"
INDICATOR_HINT = "Signal 只由 completed bar + source entry frozen params 產生；計畫不是券商送單，實際成交仍須在掛單表輸入 broker fill。"
POSITION_DECISION_HINT = "左側 ▣ 可直接開啟單股回測檢視；Stop / Target / Trailing / SELL 訊號是持股決策資訊。"
SCANNER_HINT = "左側 ▣ 可直接開啟單股回測檢視；下單由你在券商端自行完成。"
BUY_ENTRY_HINT = "交易中心只登錄實際買入；賣出到帳務中心登錄。價金、手續費與持有成本由系統自動計算。"

_STATUS_TOKEN_TONES = {
    "READY": "success",
    "IDLE": "success",
    "FRESH": "success",
    "CLEAR": "success",
    "SYNCED": "success",
    "UPDATED": "success",
    "NO_DUE": "success",
    "FILLED": "success",
    "CANCELLED": "success",
    "完成": "success",
    "可沿用": "success",
    "NOT READY": "warning",
    "ACTION_REQUIRED": "warning",
    "NOT_BOOTSTRAPPED": "warning",
    "STALE/EMPTY": "warning",
    "STALE": "warning",
    "ORDERED": "warning",
    "PARTIAL": "warning",
    "PROPOSED": "warning",
    "未綁定": "warning",
    "尚無資料": "warning",
    "尚無 Params": "warning",
    "尚未": "warning",
    "注意": "warning",
    "WAIT": "warning",
    "BLOCKED": "error",
    "LIVE_BLOCKED": "error",
    "FAIL": "error",
    "FAILED": "error",
    "ERROR": "error",
    "BLOCK": "error",
    "失敗": "error",
    "執行中": "info",
    "沿用既有": "info",
    "重新訓練": "info",
    "ON": "info",
    "OFF": "info",
}
_STATUS_WORDS = sorted(_STATUS_TOKEN_TONES, key=len, reverse=True)
_STATUS_TOKEN_PATTERN = re.compile(
    "("
    + "|".join(re.escape(token) for token in _STATUS_WORDS)
    + r"|20\d{2}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:[.+-]\d{2}:?\d{2})?)?"
    + r"|(?<![A-Za-z0-9_])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:/\d+(?:\.\d+)?)?(?![A-Za-z0-9_])"
    + ")",
    re.IGNORECASE,
)


def build_trading_status_segments(text: str, *, default_tone: str = "text") -> list[tuple[str, str]]:
    """Split a status line into neutral text plus highlighted status/date/value tokens."""
    value = str(text or "")
    segments: list[tuple[str, str]] = []
    cursor = 0
    for match in _STATUS_TOKEN_PATTERN.finditer(value):
        if match.start() > cursor:
            segments.append((value[cursor:match.start()], default_tone))
        token = match.group(0)
        tone = _STATUS_TOKEN_TONES.get(token.upper())
        if tone is None:
            tone = _STATUS_TOKEN_TONES.get(token, "info")
        segments.append((token, tone))
        cursor = match.end()
    if cursor < len(value):
        segments.append((value[cursor:], default_tone))
    if not segments:
        segments.append((value, default_tone))
    return segments


class _TradingStatusLine(tk.Text):
    """Read-only wrapping status text with token-level emphasis instead of whole-line coloring."""

    _TONE_COLORS = {
        "text": WORKBENCH_TEXT,
        "muted": WORKBENCH_MUTED,
        "info": WORKBENCH_INFO,
        "success": WORKBENCH_SUCCESS,
        "warning": WORKBENCH_WARNING,
        "error": WORKBENCH_ERROR,
    }

    def __init__(self, master, *, textvariable: tk.StringVar, default_tone: str = "text", min_lines: int = 1, max_lines: int = 3):
        self._textvariable = textvariable
        self._default_tone = default_tone
        self._min_lines = max(1, int(min_lines))
        self._max_lines = max(self._min_lines, int(max_lines))
        self._trace_id = None
        super().__init__(
            master,
            height=self._min_lines,
            width=1,
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            background=WORKBENCH_BG,
            foreground=WORKBENCH_TEXT,
            insertbackground=WORKBENCH_TEXT,
            font=WORKBENCH_UI_FONT,
            takefocus=0,
            cursor="arrow",
        )
        self._measure_font = tkfont.Font(font=WORKBENCH_UI_FONT)
        for tone, color in self._TONE_COLORS.items():
            self.tag_configure(tone, foreground=color)
        self._trace_id = self._textvariable.trace_add("write", self._on_text_changed)
        self.bind("<Configure>", self._schedule_height_sync, add="+")
        self.set_status_text(self._textvariable.get())

    def _on_text_changed(self, *_args) -> None:
        self.set_status_text(self._textvariable.get())

    def set_status_text(self, text: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        for segment, tone in build_trading_status_segments(text, default_tone=self._default_tone):
            self.insert("end", segment, tone)
        self.configure(state="disabled")
        self._schedule_height_sync()

    def _schedule_height_sync(self, *_args) -> None:
        try:
            self.after_idle(self._sync_height)
        except tk.TclError as exc:
            _warn_gui_fallback("TradingStatusText.after_idle(_sync_height)", exc)
            return

    def _sync_height(self) -> None:
        try:
            if not self.winfo_exists():
                return
            available_width = int(self.winfo_width()) - 12
            if available_width <= 40:
                return
            display_lines = 0
            logical_lines = str(self._textvariable.get() or "").splitlines() or [""]
            for logical_line in logical_lines:
                measured = max(1, int(self._measure_font.measure(logical_line)))
                display_lines += max(1, (measured + available_width - 1) // available_width)
            target = max(self._min_lines, min(self._max_lines, display_lines))
            if int(self.cget("height")) != target:
                self.configure(height=target)
        except (tk.TclError, TypeError, ValueError) as exc:
            _warn_gui_fallback("TradingStatusText._sync_height()", exc)
            return

    def destroy(self) -> None:
        if self._trace_id is not None:
            try:
                self._textvariable.trace_remove("write", self._trace_id)
            except tk.TclError as exc:
                _warn_gui_fallback("TradingStatusText.trace_remove(write)", exc)
                pass
            self._trace_id = None
        super().destroy()


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
    policy = get_trading_policy_snapshot()
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
    bundle["reconcile"] = _capture_initial_panel_value(
        lambda: reconcile_trading_v2_consumer_state_from_local_evidence(root)
    )
    # Account recovery historically ran before protection/indicator readers. Keep
    # that ordering, then parallelize independent canonical read models.
    bundle["account"] = _capture_initial_panel_value(lambda: build_trading_account_panel_snapshot(root))

    def _load_dashboard():
        snapshot = build_trading_account_dashboard_read_model(root)
        publish_trading_account_dashboard_snapshot(root, snapshot)
        return snapshot

    loaders = {
        "dashboard": _load_dashboard,
        "candidate_read": lambda: get_trading_candidate_snapshot_read_model(root),
        "protection": lambda: get_trading_protection_plan_read_model(root, recover_pending_fill=False),
        "indicator": lambda: get_trading_indicator_exit_plan_read_model(root, recover_pending_fill=False),
        "workflow": lambda: build_trading_daily_workflow_snapshot(root),
        "orders": lambda: get_trading_order_read_model(root),
        "proposed": lambda: get_trading_proposed_order_plan_read_model(root),
        "position_rollforward": lambda: build_trading_position_rollforward_snapshot(root),
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

    for key in ("dashboard", "candidate_read", "protection", "indicator", "workflow"):
        bundle[key] = parallel_results[key]

    candidate = bundle["candidate_read"]
    if candidate[0] and bool(candidate[1].get("fresh")):
        bundle["candidate_payload"] = _capture_initial_panel_value(
            lambda: load_trading_candidate_snapshot(root, require_current=False)
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


class TradingAccountPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10, style=WORKBENCH_FRAME_STYLE)
        self._snapshot: dict[str, object] = {}
        self._account_dashboard_snapshot: dict[str, object] = {}
        self._performance_rows: list[dict[str, object]] = []
        self._position_rows: dict[str, dict[str, object]] = {}
        self._candidate_rows: list[dict[str, object]] = []
        self._candidate_payload: dict[str, object] = {}
        self._proposed_order_rows: list[dict[str, object]] = []
        self._order_snapshot: dict[str, object] = {}
        self._order_rows: dict[str, dict[str, object]] = {}
        self._protection_snapshot: dict[str, object] = {}
        self._protection_rows: list[dict[str, object]] = []
        self._indicator_snapshot: dict[str, object] = {}
        self._indicator_rows: list[dict[str, object]] = []
        self._command_thread = None
        self._command_token = 0
        self._command_results: queue.Queue = queue.Queue()
        self._command_poll_after_id = None
        self._command_context: dict[int, dict[str, object]] = {}
        self._command_saved_button_states: dict[object, str] = {}
        self._param_mode_user_selected = False
        self._workflow_buttons = []
        self._workflow_action_buttons: dict[str, object] = {}
        self._operations_snapshot: dict[str, object] = {}
        self._latest_workflow_snapshot: dict[str, object] = {}
        self._footer_hint_after_id = None
        self._initial_state_thread = None
        self._initial_state_token = 0
        self._initial_state_results: queue.Queue = queue.Queue()
        self._initial_state_poll_after_id = None
        self._initial_preloaded: dict[str, object] = {}
        self._suspend_operations_refresh = False
        self._build_ui()
        self._set_initial_loading_state()
        # AI: Paint the complete Trading page first.  Canonical state reads may validate
        # persisted lineage and touch multiple files, so do them off the Tk thread and
        # apply the already-read snapshots once they are ready.
        self.after(80, self._start_initial_state_load)

    def _set_initial_loading_state(self):
        self._operations_next_var.set("LOADING | Trading 狀態載入中…")
        self._operations_detail_var.set("canonical Trading state 正在背景讀取。")
        self._set_workflow_buttons_state("disabled")
        for button_name in ("_initialize_button", "_set_cash_button", "_add_button", "_correct_button", "_remove_button", "_confirm_ordered_button"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.configure(state="disabled")

    def _start_initial_state_load(self):
        if self._initial_state_thread is not None and self._initial_state_thread.is_alive():
            return
        self._initial_state_token += 1
        token = int(self._initial_state_token)
        thread = threading.Thread(
            target=self._initial_state_load_worker,
            args=(token,),
            name="workbench-trading-initial-state",
            daemon=True,
        )
        self._initial_state_thread = thread
        thread.start()
        self._schedule_initial_state_poll()

    def _initial_state_load_worker(self, token: int):
        bundle = build_trading_account_panel_initial_bundle(WORKBENCH_PROJECT_ROOT)
        self._initial_state_results.put((int(token), bundle))

    def _schedule_initial_state_poll(self):
        if self._initial_state_poll_after_id is None:
            self._initial_state_poll_after_id = self.after(25, self._drain_initial_state_results)

    def _drain_initial_state_results(self):
        self._initial_state_poll_after_id = None
        while True:
            try:
                token, bundle = self._initial_state_results.get_nowait()
            except queue.Empty:
                break
            self._finish_initial_state_load(token, bundle)
        if self._initial_state_thread is not None and self._initial_state_thread.is_alive():
            self._schedule_initial_state_poll()

    def _apply_state_bundle(self, bundle: dict[str, object]) -> None:
        """Apply an already-read canonical Trading state bundle on the Tk thread only."""
        self._initial_preloaded = dict(bundle or {})
        self._suspend_operations_refresh = True
        try:
            self.refresh_account()
            self.refresh_account_dashboard()
            self.refresh_candidate_snapshot_rows()
            self.refresh_protection_plan()
            self.refresh_indicator_exit_plan()
            self.refresh_daily_workflow()
        finally:
            self._suspend_operations_refresh = False
        self.refresh_operations_status()
        self._initial_preloaded.clear()

    def _finish_initial_state_load(self, token: int, bundle: dict[str, object]):
        if int(token) != int(self._initial_state_token):
            return
        self._initial_state_thread = None
        self._apply_state_bundle(bundle)

    def _iter_action_buttons(self):
        stack = [self]
        while stack:
            parent = stack.pop()
            for child in parent.winfo_children():
                stack.append(child)
                if isinstance(child, ttk.Button):
                    yield child

    def _set_command_busy(self, busy: bool, *, label: str = "") -> None:
        if busy:
            self._command_saved_button_states = {}
            for button in self._iter_action_buttons():
                try:
                    state = str(button.cget("state") or "normal")
                    self._command_saved_button_states[button] = state
                    button.configure(state="disabled")
                except tk.TclError as exc:
                    _warn_gui_fallback("Trading command disable button", exc)
            self._operations_next_var.set(f"BUSY | {label} 執行中…")
            self._operations_detail_var.set("背景執行 canonical Trading command；Workbench 可正常捲動與重繪，完成後只刷新一次 canonical state。")
            return
        saved = dict(self._command_saved_button_states)
        self._command_saved_button_states.clear()
        for button, state in saved.items():
            try:
                if button.winfo_exists():
                    button.configure(state=state)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading command restore button", exc)

    def _submit_trading_command(
        self,
        label: str,
        worker,
        *,
        on_success=None,
        on_error=None,
        error_title: str = "Trading 操作失敗",
        refresh_state: bool = True,
    ) -> bool:
        if self._initial_state_thread is not None and self._initial_state_thread.is_alive():
            self._operations_next_var.set("WAIT | Trading 初始狀態仍在載入中…")
            return False
        if self._command_thread is not None and self._command_thread.is_alive():
            self._operations_next_var.set(f"BUSY | {label} 尚未執行；目前已有 Trading command 執行中。")
            return False
        self._command_token += 1
        token = int(self._command_token)
        self._command_context[token] = {
            "label": str(label),
            "on_success": on_success,
            "on_error": on_error,
            "error_title": str(error_title),
        }
        self._set_command_busy(True, label=str(label))
        thread = threading.Thread(
            target=self._trading_command_worker,
            args=(token, worker, bool(refresh_state)),
            name=f"workbench-trading-command-{token}",
            daemon=True,
        )
        self._command_thread = thread
        thread.start()
        self._schedule_command_poll()
        return True

    def _trading_command_worker(self, token: int, worker, refresh_state: bool) -> None:
        result = None
        command_error = None
        try:
            result = worker()
        except Exception as exc:  # surfaced on Tk thread with the command context
            command_error = exc

        bundle = None
        bundle_error = None
        if refresh_state:
            try:
                bundle = build_trading_account_panel_initial_bundle(WORKBENCH_PROJECT_ROOT)
            except Exception as exc:
                bundle_error = exc
        self._command_results.put((int(token), result, command_error, bundle, bundle_error))

    def _schedule_command_poll(self) -> None:
        if self._command_poll_after_id is None:
            self._command_poll_after_id = self.after(25, self._drain_command_results)

    def _drain_command_results(self) -> None:
        self._command_poll_after_id = None
        while True:
            try:
                payload = self._command_results.get_nowait()
            except queue.Empty:
                break
            self._finish_trading_command(*payload)
        if self._command_thread is not None and self._command_thread.is_alive():
            self._schedule_command_poll()

    def _finish_trading_command(self, token: int, result, command_error, bundle, bundle_error) -> None:
        context = self._command_context.pop(int(token), {})
        if int(token) != int(self._command_token):
            return
        self._command_thread = None
        self._set_command_busy(False)
        if isinstance(bundle, dict):
            self._apply_state_bundle(bundle)

        on_error = context.get("on_error")
        on_success = context.get("on_success")
        if command_error is not None:
            if callable(on_error):
                on_error(command_error)
            else:
                messagebox.showerror(str(context.get("error_title") or "Trading 操作失敗"), str(command_error), parent=self)
            return
        if bundle_error is not None:
            messagebox.showwarning(
                "Trading 狀態刷新失敗",
                f"{context.get('label') or 'Trading command'} 已完成，但 canonical state 背景刷新失敗：{bundle_error}",
                parent=self,
            )
        if callable(on_success):
            on_success(result)

    def _request_state_refresh(self, label: str = "Trading 狀態刷新") -> None:
        self._submit_trading_command(label, lambda: None, refresh_state=True)

    def destroy(self):
        for attr_name, label in (("_initial_state_poll_after_id", "initial-state"), ("_command_poll_after_id", "command")):
            after_id = getattr(self, attr_name, None)
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except tk.TclError as exc:
                    _warn_gui_fallback(f"Trading {label} poll after_cancel", exc)
                setattr(self, attr_name, None)
        super().destroy()

    def _build_ui(self):
        # AI: The Trading page contains several independent detail tables.  A fixed-height
        # notebook tab used to squeeze the middle rows to ~0px on common 1080p screens,
        # making Scanner / proposed-order details effectively invisible.  Keep the
        # existing tables and semantics, but put the whole page in one vertical canvas.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self._page_canvas = tk.Canvas(self, background=WORKBENCH_BG, highlightthickness=0, borderwidth=0)
        self._page_scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self._page_canvas.yview, style=WORKBENCH_VSCROLL_STYLE
        )
        self._page_canvas.configure(yscrollcommand=self._page_scrollbar.set, yscrollincrement=36)
        self._page_canvas.grid(row=0, column=0, sticky="nsew")
        self._page_scrollbar.grid(row=0, column=1, sticky="ns")
        content = ttk.Frame(self._page_canvas, style=WORKBENCH_FRAME_STYLE)
        self._page_content = content
        self._page_window = self._page_canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", self._sync_page_scrollregion, add="+")
        self._page_canvas.bind("<Configure>", self._sync_page_content_width, add="+")
        content.columnconfigure(0, weight=1)

        self._footer_hint_var = tk.StringVar(value="")

        operations_box = ttk.LabelFrame(content, text="Trading 操作總覽", padding=10, style=WORKBENCH_LABELLF_STYLE)
        operations_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        operations_box.columnconfigure(0, weight=1)
        self._overview_vars = {
            key: tk.StringVar(value="-")
            for key in ("sync", "total", "quick", "candidates", "buyable", "strategy_params")
        }
        self._overview_detail_vars = {key: tk.StringVar(value="-") for key in self._overview_vars}
        self._overview_primary_labels: dict[str, tk.Label] = {}
        overview_grid = ttk.Frame(operations_box, style=WORKBENCH_FRAME_STYLE)
        overview_grid.grid(row=0, column=0, sticky="ew")
        for col, (title, key) in enumerate((
            ("同步狀態", "sync"),
            ("總股數", "total"),
            ("符合快篩數", "quick"),
            ("Scanner 候選數", "candidates"),
            ("剩餘可買數", "buyable"),
            ("策略 / Params", "strategy_params"),
        )):
            box = ttk.LabelFrame(overview_grid, text=title, padding=(8, 4), style=WORKBENCH_LABELLF_STYLE)
            box.grid(row=0, column=col, padx=(0 if col == 0 else 6, 0), sticky="nsew")
            primary = tk.Label(
                box,
                textvariable=self._overview_vars[key],
                background=WORKBENCH_BG,
                foreground=WORKBENCH_TEXT,
                font=(WORKBENCH_UI_FONT[0], WORKBENCH_UI_FONT[1], "bold"),
                justify="center",
            )
            primary.pack(fill="x")
            ttk.Label(
                box,
                textvariable=self._overview_detail_vars[key],
                style=WORKBENCH_LABEL_STYLE,
                foreground=WORKBENCH_MUTED,
                justify="center",
            ).pack(fill="x", pady=(1, 0))
            self._overview_primary_labels[key] = primary
            overview_grid.columnconfigure(col, weight=1)

        self._operations_next_var = tk.StringVar(value="下一步：-")
        self._operations_detail_var = tk.StringVar(value="")
        self._operations_next_label = _TradingStatusLine(operations_box, textvariable=self._operations_next_var, max_lines=1)
        self._operations_detail_label = _TradingStatusLine(operations_box, textvariable=self._operations_detail_var, default_tone="muted", max_lines=1)
        self._operations_next_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._operations_detail_label.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        # Trading Center still consumes the account dashboard read model for current
        # position valuation, but its duplicate account-summary cards belong only to
        # Accounting Center and are not mounted here.
        self._dashboard_metric_vars = {
            key: tk.StringVar(value="-")
            for key in ("cash", "market_value", "liquidation", "equity")
        }
        self._dashboard_metric_labels = {}
        self._dashboard_detail_var = tk.StringVar(value="account/ 衍生快照尚未載入")

        workflow_box = ttk.LabelFrame(content, text="每日 Trading 流程", padding=10, style=WORKBENCH_LABELLF_STYLE)
        workflow_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        workflow_box.columnconfigure(0, weight=1)
        param_mode_row = ttk.Frame(workflow_box, style=WORKBENCH_FRAME_STYLE)
        param_mode_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(param_mode_row, text="Params 模式", foreground=WORKBENCH_TEXT, style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._param_mode_var = tk.StringVar(value=PARAM_MODE_REUSE_LABEL)
        self._param_mode_combo = ttk.Combobox(
            param_mode_row,
            textvariable=self._param_mode_var,
            values=(PARAM_MODE_REUSE_LABEL, PARAM_MODE_TRAIN_LABEL),
            state="readonly",
            width=18,
            style=WORKBENCH_COMBO_STYLE,
        )
        self._param_mode_combo.pack(side="left", padx=(8, 0))
        self._param_mode_combo.bind("<<ComboboxSelected>>", self._on_param_mode_selected)
        # Params status/lineage is rendered once in Trading 操作總覽.

        workflow_buttons = ttk.Frame(workflow_box, style=WORKBENCH_FRAME_STYLE)
        workflow_buttons.grid(row=1, column=0, sticky="w", pady=(8, 0))
        for text, action in (
            ("1 更新資料", "data"),
            ("持股日終推進", "rollforward"),
            ("2 套用 Params", "params"),
            ("3 Scanner 候選", "scanner"),
            ("每日流程 1→2→3", "all"),
        ):
            button = ttk.Button(
                workflow_buttons,
                text=text,
                command=lambda selected=action: self._start_workflow_action(selected),
                style=WORKBENCH_BUTTON_STYLE,
            )
            button.pack(side="left", padx=(0 if not self._workflow_buttons else 8, 0))
            self._workflow_buttons.append(button)
            self._workflow_action_buttons[action] = button
        # Daily workflow dynamic status is centralized in Trading 操作總覽.

        header = ttk.LabelFrame(content, text="Trading 帳戶", padding=8, style=WORKBENCH_LABELLF_STYLE)
        header.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(header, text="重新整理帳戶", command=lambda: self._request_state_refresh("帳戶狀態刷新"), style=WORKBENCH_BUTTON_STYLE).pack(side="right")

        cash_box = ttk.LabelFrame(content, text="現金", padding=10, style=WORKBENCH_LABELLF_STYLE)
        cash_box.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(cash_box, text="帳戶現金", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        self._cash_var = tk.StringVar()
        self._cash_entry = ttk.Entry(cash_box, textvariable=self._cash_var, width=22, style=WORKBENCH_ENTRY_STYLE)
        self._cash_entry.grid(row=0, column=1, sticky="w", padx=(8, 8))
        self._initialize_button = ttk.Button(cash_box, text="初始化帳戶", command=self._initialize_account, style=WORKBENCH_BUTTON_STYLE)
        self._initialize_button.grid(row=0, column=2, padx=(0, 8))
        self._set_cash_button = ttk.Button(cash_box, text="更新現金", command=self._set_cash, style=WORKBENCH_BUTTON_STYLE)
        self._set_cash_button.grid(row=0, column=3)

        form = ttk.LabelFrame(content, text="既有持股（manual adopted broker truth）", padding=10, style=WORKBENCH_LABELLF_STYLE)
        form.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        labels = ("股票代號", "股數", "剩餘成本總額", "買入日 YYYY-MM-DD", "備註")
        for col, label in enumerate(labels):
            ttk.Label(form, text=label, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else 8, 0))
        self._ticker_var = tk.StringVar()
        self._qty_var = tk.StringVar()
        self._cost_var = tk.StringVar()
        self._entry_date_var = tk.StringVar()
        self._note_var = tk.StringVar()
        ttk.Entry(form, textvariable=self._ticker_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Entry(form, textvariable=self._qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Entry(form, textvariable=self._cost_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=(4, 0))
        DatePickerField(form, textvariable=self._entry_date_var, width=14).grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Entry(form, textvariable=self._note_var, width=36, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=4, sticky="ew", padx=(8, 0), pady=(4, 0))
        form.columnconfigure(4, weight=1)

        button_row = ttk.Frame(form, style=WORKBENCH_FRAME_STYLE)
        button_row.grid(row=2, column=0, columnspan=5, sticky="w", pady=(10, 0))
        self._add_button = ttk.Button(button_row, text="新增既有持股", command=self._adopt_position, style=WORKBENCH_BUTTON_STYLE)
        self._add_button.pack(side="left")
        self._correct_button = ttk.Button(button_row, text="修正選取持股", command=self._correct_position, style=WORKBENCH_BUTTON_STYLE)
        self._correct_button.pack(side="left", padx=(8, 0))
        self._remove_button = ttk.Button(button_row, text="移除選取持股", command=self._remove_position, style=WORKBENCH_BUTTON_STYLE)
        self._remove_button.pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="清除輸入", command=self._clear_position_form, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(8, 0))

        table_box = ttk.LabelFrame(content, text="持股決策", padding=8, style=WORKBENCH_LABELLF_STYLE)
        table_box.grid(row=5, column=0, sticky="nsew", pady=(0, 8))
        table_box.rowconfigure(0, weight=1)
        table_box.columnconfigure(0, weight=1)
        columns = ("open", "ticker", "qty", "avg_cost", "current", "stop", "target", "trailing", "sell_signal", "action")
        self._tree = ttk.Treeview(table_box, columns=columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse", height=7)
        headings = {
            "open": "↗", "ticker": "股票", "qty": "股數", "avg_cost": "均價", "current": "市價",
            "stop": "目前Stop", "target": "Target", "trailing": "Trailing",
            "sell_signal": "SELL訊號", "action": "建議動作",
        }
        widths = {"open": 36, "ticker": 80, "qty": 85, "avg_cost": 95, "current": 90, "stop": 95, "target": 95, "trailing": 95, "sell_signal": 125, "action": 220}
        for key in columns:
            self._tree.heading(key, text=headings[key])
            self._tree.column(key, width=widths[key], anchor="center", stretch=(key != "open"))
        self._tree.grid(row=0, column=0, sticky="nsew")
        self._tree.bind("<<TreeviewSelect>>", self._on_position_selected)
        self._tree.bind("<Button-1>", self._on_position_tree_click, add="+")
        self._tree.bind("<Double-1>", self._open_selected_position_in_inspector, add="+")

        candidate_box = ttk.LabelFrame(content, text="今日 Scanner Pool", padding=8, style=WORKBENCH_LABELLF_STYLE)
        candidate_box.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        candidate_box.rowconfigure(0, weight=1)
        candidate_box.columnconfigure(0, weight=1)
        self._candidate_tree = PagedTable(
            candidate_box,
            columns=self._build_candidate_table_columns([]),
            page_size=12,
            default_sort_key="rank",
            empty_text="目前沒有 Scanner 候選",
            on_select=self._on_candidate_selected,
            on_open_stock=self._open_candidate_ticker_in_inspector,
            on_mousewheel=self._on_page_mousewheel,
            on_pointer_enter=lambda _event: self._show_footer_hint(SCANNER_HINT),
            on_pointer_leave=lambda _event: self._schedule_footer_hint_clear(),
        )
        self._candidate_tree.grid(row=0, column=0, sticky="ew")
        trade_box = ttk.LabelFrame(content, text="買入成交登錄", padding=10, style=WORKBENCH_LABELLF_STYLE)
        trade_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        trade_labels = ("股票", "數量", "成交價", "成交日")
        self._trade_ticker_var = tk.StringVar()
        self._trade_qty_var = tk.StringVar()
        self._trade_price_var = tk.StringVar()
        self._trade_date_var = tk.StringVar()
        for col, label in enumerate(trade_labels):
            ttk.Label(trade_box, text=label, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else 8, 0))
        ttk.Entry(trade_box, textvariable=self._trade_ticker_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Entry(trade_box, textvariable=self._trade_qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Entry(trade_box, textvariable=self._trade_price_var, width=14, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=(4, 0))
        DatePickerField(trade_box, textvariable=self._trade_date_var, width=12).grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=(4, 0))
        trade_buttons = ttk.Frame(trade_box, style=WORKBENCH_FRAME_STYLE)
        trade_buttons.grid(row=1, column=4, sticky="w", padx=(12, 0), pady=(4, 0))
        ttk.Button(trade_buttons, text="登錄買入成交", command=lambda: self._record_simple_trade("BUY"), style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        performance_box = ttk.LabelFrame(content, text="帳戶績效統計", padding=8, style=WORKBENCH_LABELLF_STYLE)
        performance_box.grid(row=8, column=0, sticky="nsew", pady=(0, 8))
        performance_box.columnconfigure(0, weight=1)
        perf_columns = ("scope", "count", "cost", "pnl", "return", "profitable", "rate")
        self._performance_tree = ttk.Treeview(performance_box, columns=perf_columns, show="headings", style=WORKBENCH_TREE_STYLE, height=3)
        self._performance_tree.tag_configure("gain", foreground=WORKBENCH_ERROR)
        self._performance_tree.tag_configure("loss", foreground=WORKBENCH_SUCCESS)
        self._performance_tree.tag_configure("flat", foreground=WORKBENCH_TEXT)
        perf_headings = {"scope": "範圍", "count": "持股/交易數", "cost": "成本基礎", "pnl": "PnL", "return": "報酬率", "profitable": "獲利數", "rate": "獲利率/勝率"}
        perf_widths = {"scope": 120, "count": 100, "cost": 130, "pnl": 120, "return": 95, "profitable": 90, "rate": 110}
        for key in perf_columns:
            self._performance_tree.heading(key, text=perf_headings[key])
            self._performance_tree.column(key, width=perf_widths[key], anchor="center")
        self._performance_tree.grid(row=0, column=0, sticky="ew")
        self._performance_note_var = tk.StringVar(value="已賣出＝account event 中已完整平倉交易；持有中以最新 Trading 市價估值。")
        ttk.Label(performance_box, textvariable=self._performance_note_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).grid(row=1, column=0, sticky="w", pady=(5, 0))

        # AI: Trading Center owns selection and execution.  Account maintenance,
        # inventory detail, transaction history and performance live in the separate
        # top-level Accounting Center.  Keep these widgets instantiated for backward-
        # compatible refresh methods but remove them from the Trading Center layout.
        for accounting_section in (header, cash_box, form, performance_box):
            accounting_section.grid_remove()

        advanced_notebook = ttk.Notebook(content, style="Workbench.TNotebook")
        # Legacy broker-OMS widgets remain instantiated for backward-compatible state readers,
        # but are intentionally not mounted in Trading Center.  User executes at the broker.
        advanced_orders_tab = ttk.Frame(advanced_notebook, padding=6, style=WORKBENCH_FRAME_STYLE)
        advanced_protection_tab = ttk.Frame(advanced_notebook, padding=6, style=WORKBENCH_FRAME_STYLE)
        advanced_indicator_tab = ttk.Frame(advanced_notebook, padding=6, style=WORKBENCH_FRAME_STYLE)
        for tab in (advanced_orders_tab, advanced_protection_tab, advanced_indicator_tab):
            tab.columnconfigure(0, weight=1)
        advanced_notebook.add(advanced_orders_tab, text="進階｜掛單/成交")
        advanced_notebook.add(advanced_protection_tab, text="進階｜Stop / TP")
        advanced_notebook.add(advanced_indicator_tab, text="進階｜Indicator SELL")

        proposed_box = ttk.LabelFrame(advanced_orders_tab, text="建議掛單（尚未送單／尚未成交）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        proposed_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        proposed_box.rowconfigure(1, weight=1)
        proposed_box.columnconfigure(0, weight=1)
        self._proposed_status_var = tk.StringVar(value="尚未產生建議掛單。")
        self._proposed_status_label = _TradingStatusLine(proposed_box, textvariable=self._proposed_status_var, default_tone="muted", max_lines=2)
        self._proposed_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        proposed_columns = ("rank", "ticker", "kind", "agree", "limit", "qty", "reserved", "stop", "target")
        self._proposed_tree = ttk.Treeview(proposed_box, columns=proposed_columns, show="headings", style=WORKBENCH_TREE_STYLE, height=6)
        proposed_headings = {
            "rank": "順位", "ticker": "股票", "kind": "類型", "agree": "同意/成員",
            "limit": "買入限價", "qty": "股數", "reserved": "預留資金", "stop": "初始Stop", "target": "Target / 完成線"
        }
        proposed_widths = {"rank": 60, "ticker": 80, "kind": 110, "agree": 115, "limit": 100, "qty": 90, "reserved": 120, "stop": 100, "target": 135}
        for key in proposed_columns:
            self._proposed_tree.heading(key, text=proposed_headings[key])
            self._proposed_tree.column(key, width=proposed_widths[key], anchor="center")
        self._proposed_tree.grid(row=1, column=0, sticky="nsew")

        submit_row = ttk.Frame(proposed_box, style=WORKBENCH_FRAME_STYLE)
        submit_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(submit_row, text="4 建議掛單", command=lambda: self._start_workflow_action("orders"), style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(0, 12))
        ttk.Label(submit_row, text="券商委託號（可留空）", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._broker_order_id_var = tk.StringVar()
        ttk.Entry(submit_row, textvariable=self._broker_order_id_var, width=18, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(submit_row, text="備註", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._order_note_var = tk.StringVar()
        ttk.Entry(submit_row, textvariable=self._order_note_var, width=28, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10), fill="x", expand=True)
        self._confirm_ordered_button = ttk.Button(
            submit_row,
            text="確認選取已送單",
            command=self._confirm_selected_ordered,
            style=WORKBENCH_BUTTON_STYLE,
        )
        self._confirm_ordered_button.pack(side="right")

        pending_box = ttk.LabelFrame(advanced_orders_tab, text="券商掛單狀態（ORDERED / PARTIAL / FILLED / CANCELLED）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        pending_box.grid(row=1, column=0, sticky="nsew")
        pending_box.rowconfigure(1, weight=1)
        pending_box.columnconfigure(0, weight=1)
        self._order_status_var = tk.StringVar(value="尚無實際送單紀錄。")
        self._order_status_label = _TradingStatusLine(pending_box, textvariable=self._order_status_var, default_tone="muted", max_lines=2)
        self._order_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        order_columns = ("ticker", "side", "purpose", "status", "order_type", "price", "qty", "filled", "remaining", "avg_fill", "broker_id", "info_date", "ordered_at", "filled_at", "cancelled_at")
        self._order_tree = ttk.Treeview(pending_box, columns=order_columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse")
        order_headings = {
            "ticker": "股票", "side": "方向", "purpose": "用途", "status": "狀態", "order_type": "類型",
            "price": "委託/觸發價", "qty": "委託股數", "filled": "已成交", "remaining": "未成交",
            "avg_fill": "平均成交價", "broker_id": "券商委託號", "info_date": "資訊日",
            "ordered_at": "送單時間", "filled_at": "完成時間", "cancelled_at": "取消時間",
        }
        order_widths = {
            "ticker": 80, "side": 65, "purpose": 125, "status": 90, "order_type": 100, "price": 105,
            "qty": 95, "filled": 90, "remaining": 90, "avg_fill": 105, "broker_id": 130, "info_date": 105,
            "ordered_at": 160, "filled_at": 160, "cancelled_at": 160,
        }
        for key in order_columns:
            self._order_tree.heading(key, text=order_headings[key])
            self._order_tree.column(key, width=order_widths[key], anchor="center")
        self._order_tree.grid(row=1, column=0, sticky="nsew")
        fill_row = ttk.Frame(pending_box, style=WORKBENCH_FRAME_STYLE)
        fill_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(fill_row, text="本次成交股數", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_qty_var = tk.StringVar()
        ttk.Entry(fill_row, textvariable=self._fill_qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(fill_row, text="本次成交價", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_price_var = tk.StringVar()
        ttk.Entry(fill_row, textvariable=self._fill_price_var, width=12, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(fill_row, text="成交日", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_date_var = tk.StringVar()
        DatePickerField(fill_row, textvariable=self._fill_date_var, width=12).pack(side="left", padx=(6, 10))
        self._confirm_fill_button = ttk.Button(
            fill_row,
            text="確認選取成交",
            command=self._confirm_selected_fill,
            style=WORKBENCH_BUTTON_STYLE,
        )
        self._confirm_fill_button.pack(side="right")

        pending_buttons = ttk.Frame(pending_box, style=WORKBENCH_FRAME_STYLE)
        pending_buttons.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._cancel_order_button = ttk.Button(
            pending_buttons,
            text="確認選取剩餘委託已取消",
            command=self._cancel_selected_order,
            style=WORKBENCH_BUTTON_STYLE,
        )
        self._cancel_order_button.pack(side="left")
        ttk.Button(pending_buttons, text="刷新掛單狀態", command=lambda: self._request_state_refresh("券商掛單狀態刷新"), style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(8, 0))

        protection_box = ttk.LabelFrame(advanced_protection_tab, text="成交後 Stop / TP 保護單計畫（logical plan；送單狀態見券商掛單表）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        protection_box.grid(row=0, column=0, sticky="nsew")
        protection_box.rowconfigure(1, weight=1)
        protection_box.columnconfigure(0, weight=1)
        self._protection_status_var = tk.StringVar(value="尚未建立成交後保護單計畫。")
        self._protection_status_label = _TradingStatusLine(protection_box, textvariable=self._protection_status_var, default_tone="muted", max_lines=2)
        self._protection_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        protection_columns = ("ticker", "qty", "entry", "stop_qty", "stop", "tp_qty", "target", "entry_order_status", "priority")
        self._protection_tree = ttk.Treeview(
            protection_box,
            columns=protection_columns,
            show="headings",
            style=WORKBENCH_TREE_STYLE,
            selectmode="browse",
        )
        protection_headings = {
            "ticker": "股票",
            "qty": "目前持股",
            "entry": "實際成交均價",
            "stop_qty": "Stop股數",
            "stop": "Stop觸發價",
            "tp_qty": "TP股數",
            "target": "TP Limit",
            "entry_order_status": "買單狀態",
            "priority": "同bar優先序",
        }
        protection_widths = {
            "ticker": 80, "qty": 95, "entry": 110, "stop_qty": 95, "stop": 105,
            "tp_qty": 90, "target": 105, "entry_order_status": 95, "priority": 120,
        }
        for key in protection_columns:
            self._protection_tree.heading(key, text=protection_headings[key])
            self._protection_tree.column(key, width=protection_widths[key], anchor="center")
        self._protection_tree.grid(row=1, column=0, sticky="nsew")
        protection_buttons = ttk.Frame(protection_box, style=WORKBENCH_FRAME_STYLE)
        protection_buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(
            protection_buttons,
            text="建立／刷新保護單計畫",
            command=self._rebuild_protection_plan,
            style=WORKBENCH_BUTTON_STYLE,
        ).pack(side="left")
        ttk.Button(
            protection_buttons,
            text="刷新計畫狀態",
            command=lambda: self._request_state_refresh("保護單計畫狀態刷新"),
            style=WORKBENCH_BUTTON_STYLE,
        ).pack(side="left", padx=(8, 0))

        protection_submit = ttk.Frame(protection_box, style=WORKBENCH_FRAME_STYLE)
        protection_submit.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(protection_submit, text="單腿券商委託號", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._protection_broker_id_var = tk.StringVar()
        ttk.Entry(protection_submit, textvariable=self._protection_broker_id_var, width=16, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        self._confirm_stop_submitted_button = ttk.Button(
            protection_submit, text="確認 Stop 已送單", command=lambda: self._confirm_protection_leg_submitted("STOP_FULL"), style=WORKBENCH_BUTTON_STYLE
        )
        self._confirm_stop_submitted_button.pack(side="left")
        self._confirm_stop_remainder_submitted_button = ttk.Button(
            protection_submit, text="確認 Stop 剩餘 MARKET 已送單", command=lambda: self._confirm_protection_leg_submitted(PROTECTION_STOP_REMAINDER_ACTION), style=WORKBENCH_BUTTON_STYLE
        )
        self._confirm_stop_remainder_submitted_button.pack(side="left", padx=(8, 0))
        self._confirm_tp_submitted_button = ttk.Button(
            protection_submit, text="確認 TP 已送單", command=lambda: self._confirm_protection_leg_submitted("TP_HALF"), style=WORKBENCH_BUTTON_STYLE
        )
        self._confirm_tp_submitted_button.pack(side="left", padx=(8, 0))

        protection_oco = ttk.Frame(protection_box, style=WORKBENCH_FRAME_STYLE)
        protection_oco.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(protection_oco, text="券商 OCO/互斥群組 ID", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._protection_oco_group_var = tk.StringVar()
        ttk.Entry(protection_oco, textvariable=self._protection_oco_group_var, width=18, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(protection_oco, text="Stop委託號", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._protection_stop_broker_id_var = tk.StringVar()
        ttk.Entry(protection_oco, textvariable=self._protection_stop_broker_id_var, width=14, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(protection_oco, text="TP委託號", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._protection_tp_broker_id_var = tk.StringVar()
        ttk.Entry(protection_oco, textvariable=self._protection_tp_broker_id_var, width=14, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        self._confirm_oco_submitted_button = ttk.Button(
            protection_oco, text="確認 Stop+TP 已以券商 OCO 送單", command=self._confirm_protection_oco_submitted, style=WORKBENCH_BUTTON_STYLE
        )
        self._confirm_oco_submitted_button.pack(side="left")

        indicator_box = ttk.LabelFrame(advanced_indicator_tab, text="Indicator SELL 計畫", padding=8, style=WORKBENCH_LABELLF_STYLE)
        indicator_box.grid(row=0, column=0, sticky="nsew")
        indicator_box.rowconfigure(1, weight=1)
        indicator_box.columnconfigure(0, weight=1)
        self._indicator_status_var = tk.StringVar(value="尚未建立 Indicator SELL 計畫。")
        self._indicator_status_label = _TradingStatusLine(indicator_box, textvariable=self._indicator_status_var, default_tone="muted", max_lines=2)
        self._indicator_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        columns=("ticker","signal","qty","entry_date","type","carried")
        self._indicator_tree=ttk.Treeview(indicator_box, columns=columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse")
        headings={"ticker":"股票","signal":"Signal日","qty":"賣出股數","entry_date":"Entry日","type":"委託","carried":"狀態"}
        widths={"ticker":90,"signal":100,"qty":100,"entry_date":100,"type":90,"carried":110}
        for key in columns:
            self._indicator_tree.heading(key,text=headings[key]); self._indicator_tree.column(key,width=widths[key],anchor="center")
        self._indicator_tree.grid(row=1,column=0,sticky="nsew")
        buttons=ttk.Frame(indicator_box,style=WORKBENCH_FRAME_STYLE); buttons.grid(row=2,column=0,columnspan=2,sticky="w",pady=(8,0))
        ttk.Button(buttons,text="建立／刷新 Indicator SELL 計畫",command=self._rebuild_indicator_exit_plan,style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        ttk.Button(buttons,text="刷新 Indicator SELL 狀態",command=lambda: self._request_state_refresh("Indicator SELL 狀態刷新"),style=WORKBENCH_BUTTON_STYLE).pack(side="left",padx=(8,0))
        ttk.Label(buttons,text="券商委託號",style=WORKBENCH_LABEL_STYLE).pack(side="left",padx=(14,0))
        self._indicator_broker_id_var=tk.StringVar()
        ttk.Entry(buttons,textvariable=self._indicator_broker_id_var,width=16,style=WORKBENCH_ENTRY_STYLE).pack(side="left",padx=(6,8))
        ttk.Button(buttons,text="確認選取 MARKET SELL 已送單",command=self._confirm_indicator_exit_submitted,style=WORKBENCH_BUTTON_STYLE).pack(side="left")

        # Fixed bottom status line: contextual help never scrolls away or consumes page space.
        self._footer_bar = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        self._footer_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Separator(self._footer_bar, orient="horizontal").pack(fill="x", pady=(0, 3))
        footer_line = ttk.Frame(self._footer_bar, style=WORKBENCH_FRAME_STYLE)
        footer_line.pack(fill="x")
        ttk.Label(footer_line, text="操作提示｜", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="left")
        ttk.Button(footer_line, text="全狀態刷新", command=self._refresh_all_trading_state, style=WORKBENCH_BUTTON_STYLE).pack(side="right", padx=(8, 0))
        self._footer_hint_label = ttk.Label(footer_line, textvariable=self._footer_hint_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED)
        self._footer_hint_label.pack(side="left", fill="x", expand=True)
        self._bind_footer_hint(workflow_box, WORKFLOW_HINT)
        self._bind_footer_hint(table_box, POSITION_DECISION_HINT)
        self._bind_footer_hint(candidate_box, SCANNER_HINT)
        self._bind_footer_hint(trade_box, BUY_ENTRY_HINT)
        self._bind_page_mousewheel(content)

    def _sync_page_scrollregion(self, _event=None) -> None:
        try:
            self._page_canvas.configure(scrollregion=self._page_canvas.bbox("all"))
        except tk.TclError as exc:
            _warn_gui_fallback("Trading page scrollregion", exc)

    def _sync_page_content_width(self, event=None) -> None:
        try:
            width = max(1, int(event.width if event is not None else self._page_canvas.winfo_width()))
            self._page_canvas.itemconfigure(self._page_window, width=width)
        except (tk.TclError, TypeError, ValueError) as exc:
            _warn_gui_fallback("Trading page content width", exc)

    def _on_page_mousewheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120) if event.delta else 0
        if delta:
            self._page_canvas.yview_scroll(delta, "units")
            return "break"
        return None

    def _bind_page_mousewheel(self, widget) -> None:
        targets = [widget]
        index = 0
        while index < len(targets):
            target = targets[index]
            index += 1
            try:
                for child in target.winfo_children():
                    if child not in targets:
                        targets.append(child)
                target.bind("<MouseWheel>", self._on_page_mousewheel, add="+")
                target.bind("<Button-4>", self._on_page_mousewheel, add="+")
                target.bind("<Button-5>", self._on_page_mousewheel, add="+")
            except tk.TclError as exc:
                _warn_gui_fallback("Trading page mousewheel bind", exc)

    @staticmethod
    def _fit_tree_rows(tree, count):
        count = int(count or 0)
        if count <= 0:
            tree.grid_remove()
            return
        tree.grid()
        tree.configure(height=count)

    def _reload_protection_rows(self, rows):
        for item in self._protection_tree.get_children():
            self._protection_tree.delete(item)
        self._protection_rows = [dict(row) for row in list(rows or [])]
        self._fit_tree_rows(self._protection_tree, len(self._protection_rows))
        for row in self._protection_rows:
            self._protection_tree.insert(
                "",
                "end",
                iid=str(row.get("ticker") or ""),
                values=(
                    row.get("ticker") or "-",
                    f"{int(row.get('position_qty') or 0):,}",
                    self._format_candidate_number(row.get("entry_fill_price"), digits=2),
                    f"{int(row.get('stop_qty') or 0):,}",
                    ("MARKET" if bool(row.get("stop_forced_exit")) else self._format_candidate_number(row.get("stop_price"), digits=2)),
                    f"{int(row.get('tp_qty') or 0):,}",
                    self._format_candidate_number(row.get("target_price"), digits=2),
                    row.get("entry_order_status") or "-",
                    row.get("same_bar_priority") or "-",
                ),
            )

    def _selected_protection_row(self):
        selected = self._protection_tree.selection()
        if not selected:
            return None
        ticker = str(selected[0])
        for row in self._protection_rows:
            if str(row.get("ticker") or "") == ticker:
                return dict(row)
        return None

    def _confirm_protection_leg_submitted(self, action: str):
        row = self._selected_protection_row()
        if not row:
            messagebox.showerror("Trading 保護 SELL", "請先選取一筆成交後保護單計畫。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        label = (
            "Stop" if action == "STOP_FULL"
            else "Stop 剩餘 MARKET forced-exit" if action == PROTECTION_STOP_REMAINDER_ACTION
            else "TP"
        )
        if not messagebox.askyesno(
            "確認保護 SELL 已送券商",
            f"確認已在券商實際送出 {ticker} 的 {label} SELL？\n\n此動作只建立 ORDERED broker truth，不代表成交。若同時存在其他 SELL 委託，系統會依實際持股上限檢查。",
            parent=self,
        ):
            return
        expected_revision = int(self._current_order_revision())
        broker_order_id = self._protection_broker_id_var.get().strip() or None

        def worker():
            return confirm_trading_protection_leg_submission(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                action=action,
                expected_order_revision=expected_revision,
                broker_order_id=broker_order_id,
                note="Workbench confirmed protection SELL submission",
            )

        def on_success(_result):
            self._protection_broker_id_var.set("")
            messagebox.showinfo("Trading 保護 SELL", f"{ticker} {label} 已記錄為 ORDERED；account 未修改。", parent=self)

        self._submit_trading_command(
            f"{ticker} {label} 送單確認",
            worker,
            on_success=on_success,
            error_title="Trading 保護 SELL 送單失敗",
        )

    def _confirm_protection_oco_submitted(self):
        row = self._selected_protection_row()
        if not row:
            messagebox.showerror("Trading 保護 OCO", "請先選取一筆成交後保護單計畫。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        group_id = self._protection_oco_group_var.get().strip()
        if not group_id:
            messagebox.showerror("Trading 保護 OCO", "必須輸入券商實際 OCO/互斥群組 ID；系統不會自行假設券商支援 OCO。", parent=self)
            return
        if not messagebox.askyesno(
            "確認券商 OCO 送單",
            f"確認券商已將 {ticker} Stop + TP 以 native OCO/互斥群組送出？\n\n群組 ID：{group_id}\n\n只有實際券商具備互斥/共享持股語意時才可確認。此動作不代表成交。",
            parent=self,
        ):
            return
        expected_revision = int(self._current_order_revision())
        stop_broker_order_id = self._protection_stop_broker_id_var.get().strip() or None
        tp_broker_order_id = self._protection_tp_broker_id_var.get().strip() or None

        def worker():
            return confirm_trading_protection_oco_submission(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                expected_order_revision=expected_revision,
                broker_oco_group_id=group_id,
                stop_broker_order_id=stop_broker_order_id,
                tp_broker_order_id=tp_broker_order_id,
                note="Workbench confirmed broker-native OCO protection submission",
            )

        def on_success(_result):
            self._protection_oco_group_var.set("")
            self._protection_stop_broker_id_var.set("")
            self._protection_tp_broker_id_var.set("")
            messagebox.showinfo("Trading 保護 OCO", f"{ticker} Stop+TP 已依使用者確認記錄為券商 OCO ORDERED；account 未修改。", parent=self)

        self._submit_trading_command(
            f"{ticker} OCO 送單確認",
            worker,
            on_success=on_success,
            error_title="Trading 保護 OCO 送單失敗",
        )

    def refresh_protection_plan(self):
        try:
            snapshot = _load_panel_value(
                self, "protection", lambda: get_trading_protection_plan_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (ValueError, RuntimeError, OSError) as exc:
            self._protection_snapshot = {}
            self._reload_protection_rows([])
            self._protection_status_var.set(f"保護單計畫讀取失敗：{exc}")
            return
        self._protection_snapshot = snapshot
        self._reload_protection_rows(snapshot.get("positions") or [])
        if not snapshot.get("exists"):
            self._protection_status_var.set(
                "尚未建立保護單計畫"
            )
            self.refresh_operations_status()
            return
        freshness = "FRESH" if snapshot.get("fresh") else "STALE"
        skipped = list(snapshot.get("manual_positions_skipped") or [])
        suffix = f" | manual未接管 {','.join(skipped)}" if skipped else ""
        self._protection_status_var.set(
            f"{freshness} | {snapshot.get('status') or '-'} / {snapshot.get('broker_status') or '-'} | "
            f"positions {int(snapshot.get('position_count') or 0)}{suffix}"
        )
        self.refresh_operations_status()

    def _rebuild_protection_plan(self):
        def on_success(result):
            result = dict(result or {})
            self._protection_status_var.set(
                f"FRESH | {result.get('status')} / {result.get('broker_status')} | positions {len(result.get('positions') or [])}"
            )

        self._submit_trading_command(
            "建立／刷新保護單計畫",
            lambda: build_trading_protection_plan(WORKBENCH_PROJECT_ROOT),
            on_success=on_success,
            error_title="Trading 保護單計畫",
        )

    def _reload_indicator_rows(self, rows):
        for item in self._indicator_tree.get_children():
            self._indicator_tree.delete(item)
        self._indicator_rows=[dict(row) for row in list(rows or [])]
        self._fit_tree_rows(self._indicator_tree, len(self._indicator_rows))
        for row in self._indicator_rows:
            signal_key=str(row.get("signal_key") or "")
            iid=signal_key or f"{row.get('ticker')}:{row.get('signal_information_date')}"
            self._indicator_tree.insert("", "end", iid=iid, values=(row.get("ticker") or "-", row.get("signal_information_date") or "-", f"{int(row.get('qty') or 0):,}", row.get("entry_trade_date") or "-", row.get("order_type") or "MARKET", "CARRIED" if row.get("carried_forward") else "NEW"))

    def _selected_indicator_row(self):
        selected=self._indicator_tree.selection()
        if not selected: return None
        key=str(selected[0])
        for row in self._indicator_rows:
            if str(row.get("signal_key") or "")==key: return dict(row)
        return None

    def refresh_indicator_exit_plan(self):
        try:
            snapshot=_load_panel_value(
                self, "indicator", lambda: get_trading_indicator_exit_plan_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (ValueError,RuntimeError,OSError) as exc:
            self._indicator_snapshot={}; self._reload_indicator_rows([]); self._indicator_status_var.set(f"Indicator SELL 計畫讀取失敗：{exc}"); return
        self._indicator_snapshot=snapshot; self._reload_indicator_rows(snapshot.get("exits") or [])
        if not snapshot.get("exists"):
            self._indicator_status_var.set("尚未建立 Indicator SELL 計畫"); return
        self._indicator_status_var.set(f"{'FRESH' if snapshot.get('fresh') else 'STALE'} | exits {int(snapshot.get('exit_count') or 0)} | active broker Indicator SELL {int(snapshot.get('active_indicator_exit_order_count') or 0)}")

    def _rebuild_indicator_exit_plan(self):
        self._submit_trading_command(
            "建立／刷新 Indicator SELL 計畫",
            lambda: build_trading_indicator_exit_plan(WORKBENCH_PROJECT_ROOT),
            error_title="Trading Indicator SELL 計畫",
        )

    def _confirm_indicator_exit_submitted(self):
        row = self._selected_indicator_row()
        if not row:
            messagebox.showerror("Trading Indicator SELL", "請先選取一筆 Indicator SELL 計畫。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        if not messagebox.askyesno(
            "確認 Indicator MARKET SELL 已送券商",
            f"確認已在券商實際送出 {ticker} 全倉 MARKET SELL？\n\n此動作只建立 ORDERED broker truth，不代表成交；若仍有 active Stop/TP 必須先在券商取消並於掛單表確認。",
            parent=self,
        ):
            return
        signal_key = str(row.get("signal_key") or "")
        expected_revision = int(self._current_order_revision())
        broker_order_id = self._indicator_broker_id_var.get().strip() or None

        def worker():
            return confirm_trading_indicator_exit_submission(
                WORKBENCH_PROJECT_ROOT,
                signal_key=signal_key,
                expected_order_revision=expected_revision,
                broker_order_id=broker_order_id,
                note="Workbench confirmed Indicator MARKET SELL submission",
            )

        def on_success(_result):
            self._indicator_broker_id_var.set("")
            messagebox.showinfo("Trading Indicator SELL", f"{ticker} Indicator MARKET SELL 已記錄為 ORDERED；account 未修改。", parent=self)

        self._submit_trading_command(
            f"{ticker} Indicator SELL 送單確認",
            worker,
            on_success=on_success,
            error_title="Trading Indicator SELL 送單失敗",
        )

    def _set_overview_card(self, key: str, primary: object, detail: object = "", *, tone: str = "text") -> None:
        self._overview_vars[key].set(str(primary if primary not in (None, "") else "-"))
        self._overview_detail_vars[key].set(str(detail or ""))
        color = {
            "text": WORKBENCH_TEXT,
            "muted": WORKBENCH_MUTED,
            "info": WORKBENCH_INFO,
            "success": WORKBENCH_SUCCESS,
            "warning": WORKBENCH_WARNING,
            "error": WORKBENCH_ERROR,
        }.get(str(tone), WORKBENCH_TEXT)
        self._overview_primary_labels[key].configure(foreground=color)

    def refresh_operations_status(self):
        if bool(getattr(self, "_suspend_operations_refresh", False)):
            return
        try:
            snapshot = _load_panel_value(
                self, "operations", lambda: build_trading_operations_status(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self._operations_snapshot = {}
            for key in self._overview_vars:
                self._set_overview_card(
                    key,
                    "BLOCKED" if key == "sync" else "-",
                    "狀態讀取失敗" if key == "sync" else "",
                    tone="error" if key == "sync" else "muted",
                )
            self._operations_next_var.set("下一步：先修正 Trading 整體狀態讀取錯誤")
            self._operations_detail_var.set(f"FAIL：{exc}")
            self._set_workflow_buttons_state("disabled")
            return
        self._operations_snapshot = snapshot

        overall = str(snapshot.get("overall_status") or "-")
        latest_data_date = snapshot.get("latest_data_date") or "-"
        data_ready = bool(snapshot.get("trading_data_ready"))
        sync_status = "READY" if data_ready else "NOT READY"
        sync_tone = "success" if data_ready else ("error" if overall in {"BLOCKED", "LIVE_BLOCKED"} else "warning")
        self._set_overview_card(
            "sync",
            sync_status,
            f"資料日 {latest_data_date} | Trading Data {sync_status}",
            tone=sync_tone,
        )

        def count_text(value: object) -> str:
            return "-" if value is None else f"{int(value):,} 檔"

        listed_count = snapshot.get("listed_ticker_count")
        quick_count = snapshot.get("quick_filter_qualified_count")
        candidate_count = snapshot.get("scanner_candidate_ticker_count")
        buyable_count = snapshot.get("scanner_buyable_ticker_count")
        scanner_fresh = bool(snapshot.get("candidate_snapshot_fresh"))
        scanner_date = snapshot.get("scanner_information_date") or "-"
        self._set_overview_card(
            "total",
            count_text(listed_count),
            "當日市場 universe",
            tone="success" if listed_count is not None else "muted",
        )
        self._set_overview_card(
            "quick",
            count_text(quick_count),
            "快篩後可進 Scanner",
            tone="success" if quick_count is not None else "muted",
        )
        scanner_tone = "success" if scanner_fresh else ("warning" if candidate_count is not None else "muted")
        scanner_status = "FRESH" if scanner_fresh else ("STALE" if candidate_count is not None else "尚未掃描")
        self._set_overview_card(
            "candidates",
            count_text(candidate_count),
            f"{scanner_status} | {scanner_date}",
            tone=scanner_tone,
        )
        free_slots = snapshot.get("portfolio_free_slot_count")
        max_positions = snapshot.get("portfolio_max_positions")
        unheld_count = snapshot.get("scanner_unheld_candidate_ticker_count")
        buyable_detail = (
            f"未持有候選 {int(unheld_count):,} | 空位 {int(free_slots):,}/{int(max_positions):,}"
            if buyable_count is not None and unheld_count is not None and free_slots is not None and max_positions is not None
            else "需 fresh Scanner 與帳戶狀態"
        )
        self._set_overview_card(
            "buyable",
            count_text(buyable_count),
            buyable_detail,
            tone="success" if buyable_count is not None else "muted",
        )

        strategy_id = snapshot.get("strategy_id") or "-"
        policy = get_trading_policy_snapshot()
        dl_state = "OFF" if not policy.get("dl_filter_enabled") and not policy.get("dl_ranking_enabled") else "ON"
        param_training_date = snapshot.get("param_training_data_date") or "尚無 Params"
        usage_mode = {
            TRADING_PARAM_USAGE_REUSE_EXISTING: "沿用既有",
            TRADING_PARAM_USAGE_TRAINED_CURRENT: "重新訓練",
        }.get(snapshot.get("param_usage_mode"), "未綁定")
        param_detail = (
            f"{snapshot.get('param_selector') or '-'} | {usage_mode} | "
            f"訓練至 {param_training_date} | DL {dl_state}"
        )
        param_tone = "success" if snapshot.get("params_ready_for_scan") else ("info" if snapshot.get("params_reusable") else "warning")
        self._set_overview_card("strategy_params", strategy_id, param_detail, tone=param_tone)

        if not data_ready:
            next_text = "1 更新 Trading 資料"
        elif not snapshot.get("params_ready_for_scan"):
            next_text = "2 套用 Params"
        elif not scanner_fresh:
            next_text = "3 Scanner 候選"
        else:
            next_text = "查看 Scanner Pool；自行至券商交易，成交後回 Workbench 登錄"
        display_status = "READY" if data_ready and bool(snapshot.get("params_ready_for_scan")) and scanner_fresh else "WAIT"
        self._operations_next_var.set(f"{display_status} | 下一步：{next_text}")
        details = []
        param_error = snapshot.get("param_error")
        param_binding_error = snapshot.get("param_binding_error")
        if param_error:
            details.append(f"Params：{param_error}")
        elif param_binding_error and not snapshot.get("params_ready_for_scan"):
            details.append("Params 尚未綁定目前 Trading Data")
        trading_blockers = [str(item) for item in list(snapshot.get("trading_data_blockers") or []) if str(item)]
        if trading_blockers:
            details.append("Data：" + "；".join(trading_blockers[:2]))
        rollforward_due = list(snapshot.get("rollforward_due_tickers") or [])
        if rollforward_due:
            details.append("待持股日終推進: " + ",".join(rollforward_due))
        self._operations_detail_var.set(" | ".join(details))
        self._apply_workflow_action_availability()

    def _run_operational_audit(self):
        def on_success(audit):
            blockers = list((audit or {}).get("blockers") or [])
            warnings = list((audit or {}).get("warnings") or [])
            if blockers:
                summary = "；".join(blockers[:2])
                self._operations_detail_var.set(f"實盤就緒：{audit.get('status')}｜{summary}")
                messagebox.showwarning("Trading 尚不可實盤", summary, parent=self)
            else:
                suffix = f"｜警告 {len(warnings)} 項" if warnings else ""
                self._operations_detail_var.set(f"實盤就緒：{audit.get('status')}{suffix}")
                messagebox.showinfo("Trading 實盤就緒檢查", f"{audit.get('status')}{suffix}", parent=self)

        def on_error(exc):
            self._operations_detail_var.set(f"實盤就緒：LIVE_BLOCKED｜Audit 失敗：{exc}")
            messagebox.showerror("Trading 實盤就緒檢查失敗", str(exc), parent=self)

        self._submit_trading_command(
            "實盤就緒檢查",
            lambda: run_trading_operational_audit(WORKBENCH_PROJECT_ROOT),
            on_success=on_success,
            on_error=on_error,
            refresh_state=True,
        )

    def _refresh_all_trading_state(self):
        # Refresh decision/accounting read models only; broker OMS recovery is no
        # longer part of the user-facing Trading Center workflow.
        self._submit_trading_command(
            "全狀態刷新",
            lambda: {"status": "REFRESH"},
            error_title="Trading 狀態刷新失敗",
            refresh_state=True,
        )

    def _apply_workflow_action_availability(self):
        if self._command_thread is not None and self._command_thread.is_alive():
            self._set_workflow_buttons_state("disabled")
            return
        for _action, button in self._workflow_action_buttons.items():
            button.configure(state="normal")
        # Legacy broker-order controls are hidden from the simplified UI.
        self._confirm_ordered_button.configure(state="disabled")

    @staticmethod
    def _set_wraplength(labels, width: int) -> None:
        wraplength = max(320, int(width) - 32)
        for label in labels:
            label.configure(wraplength=wraplength)

    def _show_footer_hint(self, text: str) -> None:
        if self._footer_hint_after_id is not None:
            try:
                self.after_cancel(self._footer_hint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading footer hint after_cancel", exc)
            self._footer_hint_after_id = None
        self._footer_hint_var.set(str(text or ""))

    def _schedule_footer_hint_clear(self) -> None:
        if self._footer_hint_after_id is not None:
            try:
                self.after_cancel(self._footer_hint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading footer hint after_cancel", exc)
        self._footer_hint_after_id = self.after(80, self._clear_footer_hint)

    def _clear_footer_hint(self) -> None:
        self._footer_hint_after_id = None
        self._footer_hint_var.set("")

    def _bind_footer_hint(self, widget, text: str) -> None:
        targets = [widget]
        try:
            targets.extend(widget.winfo_children())
        except tk.TclError as exc:
            _warn_gui_fallback("Trading footer hint winfo_children", exc)
        index = 0
        while index < len(targets):
            target = targets[index]
            index += 1
            try:
                for child in target.winfo_children():
                    if child not in targets:
                        targets.append(child)
                target.bind("<Enter>", lambda _event, value=text: self._show_footer_hint(value), add="+")
                target.bind("<Leave>", lambda _event: self._schedule_footer_hint_clear(), add="+")
            except tk.TclError as exc:
                _warn_gui_fallback("Trading footer hint bind", exc)

    def _on_param_mode_selected(self, _event=None) -> None:
        self._param_mode_user_selected = True

    def _selected_param_mode(self) -> str:
        label = str(self._param_mode_var.get() or "").strip()
        try:
            return PARAM_MODE_BY_LABEL[label]
        except KeyError as exc:
            raise ValueError(f"不支援的 Params 模式：{label or '-'}") from exc

    def _set_workflow_buttons_state(self, state: str):
        for button in self._workflow_buttons:
            button.configure(state=state)

    def refresh_candidate_snapshot_rows(self):
        try:
            snapshot = _load_panel_value(
                self, "candidate_read", lambda: get_trading_candidate_snapshot_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self._reload_candidate_rows([], candidate_payload=None)
            self._operations_detail_var.set(f"Scanner snapshot 讀取失敗：{exc}")
            return
        if not bool(snapshot.get("valid")):
            self._reload_candidate_rows([], candidate_payload=None)
            self._operations_detail_var.set(
                f"Scanner snapshot 無效：{snapshot.get('error') or 'schema 不相容'}；請重新執行 Scanner。"
            )
            return
        if not bool(snapshot.get("fresh")):
            self._reload_candidate_rows([], candidate_payload=None)
            self._operations_detail_var.set(
                f"Scanner snapshot 已過期：{snapshot.get('error') or 'inputs 已變更'}；請重新執行 Scanner。"
            )
            return
        try:
            payload = _load_panel_value(
                self, "candidate_payload", lambda: load_trading_candidate_snapshot(WORKBENCH_PROJECT_ROOT, require_current=False)
            )
        except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
            self._reload_candidate_rows([], candidate_payload=None)
            self._operations_detail_var.set(f"Scanner snapshot 讀取失敗：{exc}")
            return
        self._reload_candidate_rows(payload.get("candidate_rows") or [], candidate_payload=payload)

    def refresh_proposed_order_plan(self):
        try:
            snapshot = _load_panel_value(
                self, "proposed_read", lambda: get_trading_proposed_order_plan_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set(f"建議掛單讀取失敗：{exc}")
            return
        if not snapshot.get("exists"):
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set("尚未產生建議掛單。")
            return
        if not snapshot.get("valid"):
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set(f"建議掛單 INVALID：{snapshot.get('error') or 'schema 不合法'}")
            return
        if not snapshot.get("fresh"):
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set("PROPOSED STALE｜請重新執行 3 Scanner 與 4 建議掛單。")
            return
        try:
            payload = _load_panel_value(
                self, "proposed_payload", lambda: load_current_trading_proposed_order_plan(WORKBENCH_PROJECT_ROOT, require_current=False)
            )
        except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set(f"建議掛單讀取失敗：{exc}")
            return
        self._reload_proposed_order_rows(payload.get("orders") or [])
        self._proposed_status_var.set(
            f"PROPOSED | account rev {payload.get('account_revision')} | equity {format_trading_money(payload.get('sizing_equity'))} | "
            f"預留 {format_trading_money(payload.get('reserved_total'))} | 餘額 {format_trading_money(payload.get('cash_after_reservation'))}"
        )

    def _refresh_workflow_views(self):
        self.refresh_daily_workflow()
        self.refresh_candidate_snapshot_rows()
        self.refresh_proposed_order_plan()

    def refresh_daily_workflow(self):
        try:
            snapshot = _load_panel_value(
                self, "workflow", lambda: build_trading_daily_workflow_snapshot(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self._latest_workflow_snapshot = {}
            self._operations_detail_var.set(f"FAIL：Workflow 狀態讀取失敗：{exc}")
            return
        self._latest_workflow_snapshot = dict(snapshot)
        reusable = bool(snapshot.get("params_reusable"))
        if not self._param_mode_user_selected:
            self._param_mode_var.set(PARAM_MODE_REUSE_LABEL if reusable else PARAM_MODE_TRAIN_LABEL)
        self.refresh_operations_status()

    def _start_workflow_action(self, action: str):
        param_mode = self._selected_param_mode()
        param_label = "沿用既有 Trading Params" if param_mode == TRADING_PARAM_MODE_REUSE else "重新訓練 Trading Params"
        labels = {
            "data": "更新 Trading 資料",
            "rollforward": "持股日終推進",
            "params": param_label,
            "scanner": "Scanner 候選",
            "orders": "產生建議掛單",
            "all": f"每日流程 1→2→3（{param_label}）",
        }
        if action not in labels:
            messagebox.showerror("Trading workflow", f"未知 workflow action: {action}", parent=self)
            return
        self._operations_detail_var.set(f"執行中：{labels[action]}")

        def on_success(result):
            self._finish_workflow_success(action, result)

        def on_error(exc):
            self._operations_detail_var.set(f"FAIL：{type(exc).__name__}: {exc}")
            messagebox.showerror("Trading workflow 失敗", f"{type(exc).__name__}: {exc}", parent=self)

        self._submit_trading_command(
            labels[action],
            lambda: self._run_workflow_command(action, param_mode),
            on_success=on_success,
            on_error=on_error,
            refresh_state=True,
        )

    @staticmethod
    def _run_workflow_command(action: str, param_mode: str):
        console_progress = MarketDataDailyConsoleProgress() if action in {"data", "all"} else None
        try:
            if action == "data":
                return run_trading_market_data_update(
                    project_root=WORKBENCH_PROJECT_ROOT,
                    progress_fn=console_progress.progress,
                    quota_wait_fn=console_progress.quota_wait,
                )
            if action == "rollforward":
                result = run_trading_position_rollforward(project_root=WORKBENCH_PROJECT_ROOT)
                if str(result.get("status") or "") != "NO_ACCOUNT":
                    build_trading_indicator_exit_plan(WORKBENCH_PROJECT_ROOT)
                return result
            if action == "params":
                return run_trading_param_step(project_root=WORKBENCH_PROJECT_ROOT, mode=param_mode)
            if action == "scanner":
                return run_trading_candidate_scan(project_root=WORKBENCH_PROJECT_ROOT)
            if action == "orders":
                return build_trading_proposed_order_plan(project_root=WORKBENCH_PROJECT_ROOT)
            return run_trading_daily_workflow(
                project_root=WORKBENCH_PROJECT_ROOT,
                param_mode=param_mode,
                data_progress_fn=console_progress.progress,
                data_quota_wait_fn=console_progress.quota_wait,
            )
        finally:
            if console_progress is not None:
                console_progress.close()

    @staticmethod
    def _format_candidate_number(value, *, digits=2):
        if value is None:
            return "-"
        try:
            return f"{float(value):,.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _candidate_static_columns_before_dynamic():
        return (
            TableColumn("rank", "順位", 5, sort_kind="numeric"),
            TableColumn("ticker", "股票", 7),
            TableColumn("kind_label", "類型", 10),
        )

    def _candidate_static_columns_after_dynamic(self):
        return (
            TableColumn("market_price", "市價", 9, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=2)),
            TableColumn("limit_price", "買入限價", 9, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=2)),
            TableColumn("stop_price", "初始Stop", 9, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=2)),
            TableColumn("target_price", "停利線", 9, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=2)),
            TableColumn("proj_qty", "參考股數", 9, sort_kind="numeric", formatter=lambda v, _r: "-" if v is None else f"{int(v):,}"),
            TableColumn("proj_cost", "參考投入", 10, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=0)),
            TableColumn("ev_value", "EV", 7, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=3)),
            TableColumn("win_rate", "歷史勝率", 8, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=1) + "%" if v is not None else "-"),
            TableColumn("trade_count", "交易次數", 8, sort_kind="numeric", formatter=lambda v, _r: f"{int(v or 0):,}"),
            TableColumn("asset_growth", "資產成長", 8, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=1) + "%" if v is not None else "-"),
        )

    @staticmethod
    def _candidate_metric_formatter(metric):
        spec = dict(metric or {})
        format_kind = str(spec.get("format_kind") or "")
        if format_kind == "fraction":
            denominator = int(spec.get("denominator") or 0)
            return lambda value, _row, total=denominator: "-" if value is None or total < 1 else f"{int(value)}/{total}"
        if format_kind == "integer":
            return lambda value, _row: "-" if value is None else str(int(value))
        if format_kind == "integer_grouped":
            return lambda value, _row: "-" if value is None else f"{int(round(float(value))):,}"
        if format_kind == "number":
            return lambda value, _row: "-" if value is None else f"{float(value):.2f}"
        if format_kind == "percent":
            return lambda value, _row: "-" if value is None else f"{float(value):.2f}%"
        if format_kind == "r":
            return lambda value, _row: "-" if value is None else f"{float(value):.2f}R"
        return lambda value, _row: "-" if value is None else str(value)

    def _build_candidate_table_columns(self, display_metrics):
        dynamic_columns = tuple(
            TableColumn(
                str(metric.get("key") or ""),
                str(metric.get("label") or ""),
                int(metric.get("width") or 8),
                sort_kind=str(metric.get("sort_kind") or "text"),
                formatter=self._candidate_metric_formatter(metric),
            )
            for metric in list(display_metrics or [])
            if str(metric.get("key") or "").strip() and str(metric.get("label") or "").strip()
        )
        return self._candidate_static_columns_before_dynamic() + dynamic_columns + self._candidate_static_columns_after_dynamic()

    @staticmethod
    def _candidate_display_metrics(candidate_payload):
        payload = dict(candidate_payload or {})
        metrics = payload.get("candidate_display_metrics")
        if isinstance(metrics, list):
            return [dict(item) for item in metrics if isinstance(item, dict)]
        return []

    def _reload_candidate_rows(self, rows, *, candidate_payload=None):
        self._candidate_payload = dict(candidate_payload or {})
        display_metrics = self._candidate_display_metrics(candidate_payload)
        self._candidate_tree.set_columns(self._build_candidate_table_columns(display_metrics))
        self._candidate_rows = [dict(row) for row in list(rows or [])]
        self._candidate_by_ticker = {
            str(row.get("ticker") or "").strip().upper(): row
            for row in self._candidate_rows
            if str(row.get("ticker") or "").strip()
        }
        kind_labels = {"buy": "新訊號", "extended": "延續", "extended_tbd": "延續(TBD)", "reentry": "再進場"}
        display_rows = []
        for idx, row in enumerate(self._candidate_rows, 1):
            seed = dict(row.get("execution_plan_seed") or {})
            ticker = str(row.get("ticker") or "-").strip().upper()
            display_row = {
                "_table_id": f"candidate:{ticker}",
                "rank": idx,
                "ticker": ticker,
                "kind_label": kind_labels.get(str(row.get("kind") or ""), str(row.get("kind") or "-")),
                "market_price": row.get("prev_close"),
                "limit_price": row.get("limit_price") if row.get("limit_price") is not None else seed.get("limit_price"),
                "stop_price": seed.get("init_sl"),
                "target_price": seed.get("target_price"),
                "proj_qty": row.get("proj_qty"),
                "proj_cost": row.get("proj_cost"),
                "ev_value": row.get("expected_value", row.get("ev")),
                "win_rate": row.get("win_rate"),
                "trade_count": row.get("trade_count"),
                "asset_growth": row.get("asset_growth"),
            }
            for metric in display_metrics:
                metric_key = str(metric.get("key") or "").strip()
                value_field = str(metric.get("value_field") or "").strip()
                if metric_key and value_field:
                    display_row[metric_key] = row.get(value_field)
            display_rows.append(display_row)
        self._candidate_tree.set_rows(display_rows, preserve_selection=True)

    def _selected_candidate_ticker(self):
        row = self._candidate_tree.selected_row()
        return str(row.get("ticker") or "").strip().upper() if row else None

    def _selected_candidate_row(self):
        ticker = self._selected_candidate_ticker()
        return dict(getattr(self, "_candidate_by_ticker", {}).get(ticker) or {}) if ticker else None

    def _on_candidate_selected(self, row):
        ticker = str((row or {}).get("ticker") or "").strip().upper()
        self._trade_ticker_var.set(ticker)

    def _open_ticker_in_inspector(self, ticker, *, candidate_row=None):
        ticker = str(ticker or "").strip().upper()
        if not ticker:
            return
        callback = getattr(self.winfo_toplevel(), "_open_single_stock_inspector", None)
        if not callable(callback):
            messagebox.showerror("單股回測檢視", "Workbench 尚未提供單股回測導覽。", parent=self)
            return
        callback(
            ticker,
            runtime_domain="trading",
            auto_run=True,
            candidate_row=dict(candidate_row or {}),
            candidate_rows=[dict(row) for row in list(self._candidate_rows or [])],
            candidate_latest_data_date=self._candidate_payload.get("latest_data_date"),
        )

    def _open_candidate_ticker_in_inspector(self, ticker):
        ticker = str(ticker or "").strip().upper()
        candidate_row = dict(getattr(self, "_candidate_by_ticker", {}).get(ticker) or {})
        self._open_ticker_in_inspector(ticker, candidate_row=candidate_row)

    def _open_selected_candidate_in_inspector(self, _event=None):
        ticker = self._selected_candidate_ticker()
        if ticker:
            candidate_row = dict(getattr(self, "_candidate_by_ticker", {}).get(ticker) or {})
            self._open_ticker_in_inspector(ticker, candidate_row=candidate_row)

    def _open_selected_position_in_inspector(self, _event=None):
        ticker = self._selected_ticker()
        if ticker:
            self._open_ticker_in_inspector(ticker)

    def _reload_proposed_order_rows(self, rows):
        for item in self._proposed_tree.get_children():
            self._proposed_tree.delete(item)
        self._proposed_order_rows = [dict(row) for row in list(rows or [])]
        self._fit_tree_rows(self._proposed_tree, len(self._proposed_order_rows))
        for row in self._proposed_order_rows:
            iid = f"{int(row.get('rank') or 0)}:{str(row.get('ticker') or '')}"
            self._proposed_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    int(row.get("rank") or 0),
                    row.get("ticker") or "-",
                    {"buy": "新訊號", "extended": "延續", "extended_tbd": "延續(TBD)", "reentry": "再進場"}.get(
                        str(row.get("kind") or ""), str(row.get("kind") or "-")
                    ),
                    f"{int(row.get('ensemble_vote_count') or 0)}/{int(row.get('ensemble_member_count') or 0)} (門檻{int(row.get('ensemble_min_agree') or 0)})",
                    self._format_candidate_number(row.get("limit_price"), digits=2),
                    f"{int(row.get('qty') or 0):,}",
                    self._format_candidate_number(row.get("reserved_cost"), digits=0),
                    self._format_candidate_number(row.get("init_sl"), digits=2),
                    (
                        ("完成 " if str(row.get("kind") or "") in {"extended", "extended_tbd"} else "Target ")
                        + self._format_candidate_number(row.get("target_price"), digits=2)
                    ),
                ),
            )

    def _selected_proposed_order(self):
        selected = self._proposed_tree.selection()
        if not selected:
            return None
        iid = str(selected[0])
        for row in self._proposed_order_rows:
            if f"{int(row.get('rank') or 0)}:{str(row.get('ticker') or '')}" == iid:
                return dict(row)
        return None

    def _current_order_revision(self):
        revision = self._order_snapshot.get("revision")
        return None if revision is None else int(revision)

    def _has_active_orders(self) -> bool:
        return int(self._order_snapshot.get("active_order_count") or 0) > 0

    def refresh_order_state(self):
        try:
            snapshot = _load_panel_value(
                self, "orders", lambda: get_trading_order_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (ValueError, RuntimeError, OSError) as exc:
            self._order_snapshot = {}
            self._order_rows = {}
            self._order_status_var.set(f"掛單狀態讀取失敗：{exc}")
            return
        self._order_snapshot = snapshot
        self._order_rows = {}
        for item in self._order_tree.get_children():
            self._order_tree.delete(item)
        for row in list(snapshot.get("orders") or []):
            order_id = str(row.get("order_id") or "")
            self._order_rows[order_id] = dict(row)
            self._order_tree.insert(
                "",
                "end",
                iid=order_id,
                values=(
                    row.get("ticker") or "-",
                    row.get("side") or "-",
                    row.get("purpose") or "-",
                    row.get("status") or "-",
                    row.get("order_type") or "-",
                    self._format_candidate_number(row.get("trigger_price") if row.get("trigger_price") is not None else row.get("limit_price"), digits=2),
                    f"{int(row.get('qty') or 0):,}",
                    f"{int(row.get('filled_qty') or 0):,}",
                    f"{int(row.get('remaining_qty') or 0):,}",
                    self._format_candidate_number(row.get("average_fill_price"), digits=2),
                    row.get("broker_order_id") or "-",
                    row.get("information_date") or "-",
                    row.get("ordered_at") or "-",
                    row.get("filled_at") or "-",
                    row.get("cancelled_at") or "-",
                ),
            )
        self._fit_tree_rows(self._order_tree, len(self._order_rows))
        active = int(snapshot.get("active_order_count") or 0)
        revision = snapshot.get("revision")
        self._order_status_var.set(
            f"{'ACTIVE' if active else 'CLEAR'} | revision {revision if revision is not None else '-'} | active {active} "
            f"(BUY {int(snapshot.get('active_entry_order_count') or 0)} / protection SELL {int(snapshot.get('active_protection_order_count') or 0)}) | "
            f"總紀錄 {int(snapshot.get('order_count') or 0)}"
        )
        self._apply_order_lock_to_account_controls()
        self.refresh_operations_status()

    def _apply_order_lock_to_account_controls(self):
        active = self._has_active_orders()
        initialized = bool(self._snapshot.get("initialized"))
        self._set_cash_button.configure(state="disabled" if active or not initialized else "normal")
        self._add_button.configure(state="disabled" if active or not initialized else "normal")
        selected = self._selected_ticker()
        self._set_position_action_state(self._position_rows.get(selected or ""))

    def _confirm_selected_ordered(self):
        row = self._selected_proposed_order()
        if not row:
            messagebox.showerror("Trading 掛單", "請先選取一筆建議掛單。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        qty = int(row.get("qty") or 0)
        limit_price = self._format_candidate_number(row.get("limit_price"), digits=2)
        if not messagebox.askyesno(
            "確認實際送單",
            f"確認已在券商送出 {ticker} 買單？\n\n股數：{qty:,}\n限價：{limit_price}\n\n此動作只建立 ORDERED broker truth，不會扣 cash、也不會建立持股。",
            parent=self,
        ):
            return
        rank = int(row.get("rank") or 0)
        expected_revision = self._current_order_revision()
        broker_order_id = self._broker_order_id_var.get().strip() or None
        note = self._order_note_var.get().strip() or None

        def worker():
            return confirm_trading_order_submission(
                WORKBENCH_PROJECT_ROOT,
                rank=rank,
                ticker=ticker,
                expected_revision=expected_revision,
                broker_order_id=broker_order_id,
                note=note,
            )

        def on_success(_result):
            self._broker_order_id_var.set("")
            self._order_note_var.set("")
            messagebox.showinfo("Trading 掛單", f"{ticker} 已記錄為 ORDERED；account cash／positions 未變更。", parent=self)

        def on_error(exc):
            title = "Trading 掛單已更新" if isinstance(exc, TradingOrderRevisionConflict) else "Trading 掛單失敗"
            suffix = "\n\n已重新讀取最新掛單狀態。" if isinstance(exc, TradingOrderRevisionConflict) else ""
            messagebox.showerror(title, f"{exc}{suffix}", parent=self)

        self._submit_trading_command(
            f"{ticker} BUY 送單確認",
            worker,
            on_success=on_success,
            on_error=on_error,
        )

    def _confirm_selected_fill(self):
        selected = self._order_tree.selection()
        if not selected:
            messagebox.showerror("Trading 成交", "請先選取一筆 ORDERED / PARTIAL 掛單。", parent=self)
            return
        order_id = str(selected[0])
        row = dict(self._order_rows.get(order_id) or {})
        if str(row.get("status")) not in TRADING_ACTIVE_ORDER_STATUSES:
            messagebox.showerror("Trading 成交", "只有 ORDERED / PARTIAL 掛單可確認成交。", parent=self)
            return
        try:
            fill_qty = parse_trading_qty_text(self._fill_qty_var.get(), "本次成交股數")
            fill_price = parse_trading_money_text(self._fill_price_var.get(), "本次成交價", allow_zero=False)
            trade_date = self._fill_date_var.get().strip()
            if not trade_date:
                raise ValueError("成交日必填")
        except ValueError as exc:
            messagebox.showerror("Trading 成交", str(exc), parent=self)
            return
        remaining = int(row.get("remaining_qty") or 0)
        if fill_qty > remaining:
            messagebox.showerror("Trading 成交", f"本次成交股數不可超過未成交股數 {remaining:,}", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        if not messagebox.askyesno(
            "確認券商成交",
            f"確認券商實際成交 {ticker} {row.get('side') or 'BUY'}？\n\n本次成交：{fill_qty:,} 股 @ {fill_price}\n成交日：{trade_date}\n\n只有實際券商成交才可確認；此動作會以 canonical exact accounting 更新 cash／持股。",
            parent=self,
        ):
            return
        if str(row.get("side") or "BUY") == "BUY":
            fill_fn = confirm_trading_buy_order_fill
        elif str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_INDICATOR_EXIT:
            fill_fn = confirm_trading_indicator_sell_order_fill
        else:
            fill_fn = confirm_trading_protection_sell_order_fill
        expected_order_revision = int(self._current_order_revision())
        expected_account_revision = int(self._current_revision())

        def worker():
            result = fill_fn(
                WORKBENCH_PROJECT_ROOT,
                order_id=order_id,
                fill_qty=fill_qty,
                fill_price=fill_price,
                trade_date=trade_date,
                expected_order_revision=expected_order_revision,
                expected_account_revision=expected_account_revision,
            )
            protection_error = None
            try:
                build_trading_protection_plan(WORKBENCH_PROJECT_ROOT)
            except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
                protection_error = str(exc)
            indicator_error = None
            try:
                build_trading_indicator_exit_plan(WORKBENCH_PROJECT_ROOT)
            except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
                indicator_error = str(exc)
            return {
                "fill_result": result,
                "protection_error": protection_error,
                "indicator_error": indicator_error,
            }

        def on_success(payload):
            payload = dict(payload or {})
            result = dict(payload.get("fill_result") or {})
            protection_error = payload.get("protection_error")
            indicator_error = payload.get("indicator_error")
            self._fill_qty_var.set("")
            self._fill_price_var.set("")
            self._fill_date_var.set("")
            message = (
                f"{ticker} 已更新為 {result.get('status')}；累計成交 {int(result.get('filled_qty') or 0):,}，"
                f"未成交 {int(result.get('remaining_qty') or 0):,}。"
            )
            if protection_error:
                message += f"\n\n成交已入帳，但保護單計畫建立失敗：{protection_error}"
            else:
                message += "\n\n已依實際成交後 canonical position state 機械刷新 Stop / TP 保護單計畫；尚未送券商。"
            if indicator_error:
                message += f"\n\nIndicator 計畫刷新失敗：{indicator_error}"
            messagebox.showinfo("Trading 成交", message, parent=self)

        def on_error(exc):
            title = "Trading 成交狀態已更新" if isinstance(exc, TradingFillRevisionConflict) else "Trading 成交失敗"
            suffix = "\n\n已重新讀取最新 account／order state。" if isinstance(exc, TradingFillRevisionConflict) else ""
            messagebox.showerror(title, f"{exc}{suffix}", parent=self)

        self._submit_trading_command(
            f"{ticker} 成交確認",
            worker,
            on_success=on_success,
            on_error=on_error,
        )

    def _cancel_selected_order(self):
        selected = self._order_tree.selection()
        if not selected:
            messagebox.showerror("Trading 掛單", "請先選取一筆 ORDERED / PARTIAL 掛單。", parent=self)
            return
        order_id = str(selected[0])
        row = dict(self._order_rows.get(order_id) or {})
        if str(row.get("status")) not in TRADING_ACTIVE_ORDER_STATUSES:
            messagebox.showerror("Trading 掛單", "只有 ORDERED / PARTIAL 掛單可確認取消剩餘委託。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        if not messagebox.askyesno(
            "確認券商取消",
            f"確認券商端 {ticker} 未成交剩餘委託已取消？\n\n已成交部位不回滾；此動作只結束未成交餘額。",
            parent=self,
        ):
            return
        expected_revision = int(self._current_order_revision())
        note = self._order_note_var.get().strip() or "Workbench confirmed broker cancellation"
        filled_qty = int(row.get("filled_qty") or 0)

        def worker():
            result = confirm_trading_order_cancellation(
                WORKBENCH_PROJECT_ROOT,
                order_id=order_id,
                expected_revision=expected_revision,
                note=note,
            )
            protection_refresh_error = None
            if filled_qty > 0:
                try:
                    build_trading_protection_plan(WORKBENCH_PROJECT_ROOT)
                except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
                    protection_refresh_error = str(exc)
            return {"cancel_result": result, "protection_error": protection_refresh_error}

        def on_success(payload):
            self._order_note_var.set("")
            protection_error = dict(payload or {}).get("protection_error")
            message = f"{ticker} 已記錄為 CANCELLED。"
            if protection_error:
                message += f"\n\n取消已記錄，但保護單計畫刷新失敗：{protection_error}"
            messagebox.showinfo("Trading 掛單", message, parent=self)

        def on_error(exc):
            title = "Trading 掛單已更新" if isinstance(exc, TradingOrderRevisionConflict) else "Trading 掛單失敗"
            suffix = "\n\n已重新讀取最新掛單狀態。" if isinstance(exc, TradingOrderRevisionConflict) else ""
            messagebox.showerror(title, f"{exc}{suffix}", parent=self)

        self._submit_trading_command(
            f"{ticker} 取消委託確認",
            worker,
            on_success=on_success,
            on_error=on_error,
        )

    def _finish_workflow_success(self, action: str, result):
        result = dict(result or {})
        scan_result = dict(result.get("scanner") or {}) if action == "all" else (dict(result) if action == "scanner" else {})
        if action == "data":
            v2 = dict(result.get("market_data_v2_archive") or {})
            self._operations_detail_var.set(
                f"資料更新完成：market {result.get('market_date') or '-'} | "
                f"新進場池 {result.get('current_execution_pool_ticker_count', 0)} 檔 | "
                f"訓練池 {result.get('training_ticker_count', 0)} 檔 | "
                f"V2 {v2.get('status') or '-'} | "
                f"data/usage requests {v2.get('data_requests', 0)}/{v2.get('usage_requests', 0)}"
            )
        elif action == "rollforward":
            self._operations_detail_var.set(
                f"持股日終推進完成：持股 {result.get('processed_position_count', 0)} 檔 | completed bars {result.get('processed_bar_count', 0)} | account rev {result.get('account_revision', '-')}"
            )
        elif action == "params":
            usage_mode = str(result.get("param_usage_mode") or "-")
            usage_label = "沿用既有" if usage_mode == TRADING_PARAM_USAGE_REUSE_EXISTING else "重新訓練"
            self._operations_detail_var.set(
                f"Params 套用完成（{usage_label}）：訓練資料至 {result.get('param_training_data_date') or '-'} | "
                f"Trading data {result.get('latest_data_date') or '-'}"
            )
        elif action == "orders":
            self._operations_detail_var.set(
                f"建議掛單完成：{len(result.get('orders') or [])} 筆 | 預留 {format_trading_money(result.get('reserved_total'))} | account rev {result.get('account_revision')}"
            )
        else:
            self._operations_detail_var.set(
                f"每日 Scanner 完成：候選 {len(scan_result.get('candidate_rows') or [])} 檔 | data {scan_result.get('latest_data_date') or '-'}"
            )


    def _current_revision(self) -> int:
        revision = self._snapshot.get("revision")
        if revision is None:
            raise RuntimeError("Trading account 尚未初始化")
        return int(revision)

    def _submit_account_mutation(self, label: str, worker, *, success_message: str, clear_form: bool = False) -> None:
        def on_success(_result):
            if clear_form:
                self._clear_position_form()
            messagebox.showinfo("Trading 帳戶", success_message, parent=self)

        def on_error(exc):
            if isinstance(exc, TradingAccountRevisionConflict):
                messagebox.showerror(
                    "Trading 帳戶已更新",
                    f"{exc}\n\n已重新讀取最新狀態，請確認後再操作。",
                    parent=self,
                )
            else:
                messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)

        self._submit_trading_command(
            label,
            worker,
            on_success=on_success,
            on_error=on_error,
            refresh_state=True,
        )

    def refresh_account_dashboard(self):
        try:
            snapshot = _load_panel_value(
                self, "dashboard", lambda: build_trading_account_dashboard_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            self._account_dashboard_snapshot = {}
            for variable in self._dashboard_metric_vars.values():
                variable.set("-")
            self._dashboard_detail_var.set(f"帳戶儀表板讀取失敗：{type(exc).__name__}: {exc}")
            self._reload_performance_rows([])
            self._reload_positions()
            return
        self._account_dashboard_snapshot = dict(snapshot or {})
        summary = dict(self._account_dashboard_snapshot.get("summary") or {})

        def money_text(value):
            return "-" if value is None else f"{float(value):,.0f}"

        self._dashboard_metric_vars["cash"].set(money_text(summary.get("cash")))
        self._dashboard_metric_vars["market_value"].set(money_text(summary.get("holdings_market_value")))
        self._dashboard_metric_vars["liquidation"].set(money_text(summary.get("holdings_net_liquidation")))
        self._dashboard_metric_vars["equity"].set(money_text(summary.get("equity")))
        warnings = list(self._account_dashboard_snapshot.get("warnings") or [])
        revision = self._account_dashboard_snapshot.get("source_account_revision")
        market_date = self._account_dashboard_snapshot.get("market_date") or "-"
        self._dashboard_detail_var.set(
            f"Trading 市價日 {market_date} | account revision {revision if revision is not None else '-'} | "
            f"持股 {int(summary.get('position_count') or 0)} | 已平倉 {int(summary.get('closed_trade_count') or 0)} | "
            f"唯讀衍生資料 outputs/trading/account/"
            + (f" | 注意 {len(warnings)} 項" if warnings else "")
        )
        self._reload_performance_rows(self._account_dashboard_snapshot.get("performance") or [])
        self._reload_positions()

    def _reload_performance_rows(self, rows):
        for item in self._performance_tree.get_children():
            self._performance_tree.delete(item)
        self._performance_rows = [dict(row) for row in list(rows or [])]
        self._fit_tree_rows(self._performance_tree, len(self._performance_rows))
        for row in self._performance_rows:
            return_pct = row.get("return_pct")
            profitable_rate = row.get("profitable_rate_pct")
            pnl = row.get("pnl")
            tag = "gain" if pnl is not None and float(pnl) > 0 else "loss" if pnl is not None and float(pnl) < 0 else "flat"
            self._performance_tree.insert(
                "",
                "end",
                tags=(tag,),
                values=(
                    row.get("scope") or "-",
                    f"{int(row.get('position_or_trade_count') or 0):,}",
                    self._format_candidate_number(row.get("cost_basis"), digits=0),
                    self._format_candidate_number(row.get("pnl"), digits=0),
                    "-" if return_pct is None else f"{float(return_pct):.2f}%",
                    f"{int(row.get('profitable_count') or 0):,}",
                    "-" if profitable_rate is None else f"{float(profitable_rate):.1f}%",
                ),
            )

    def refresh_account(self):
        try:
            snapshot = _load_panel_value(
                self, "account", lambda: build_trading_account_panel_snapshot(WORKBENCH_PROJECT_ROOT)
            )
        except (ValueError, RuntimeError, OSError) as exc:
            self._snapshot = {}
            self._operations_detail_var.set(f"Trading account 狀態讀取失敗：{exc}")
            self.refresh_operations_status()
            return
        self._snapshot = snapshot
        initialized = bool(snapshot.get("initialized"))
        self._initialize_button.configure(state="disabled" if initialized else "normal")
        self._set_cash_button.configure(state="normal" if initialized else "disabled")
        self._add_button.configure(state="normal" if initialized else "disabled")
        if initialized and snapshot.get("cash") is not None:
            self._cash_var.set(format_trading_money(snapshot.get("cash")))
        elif not initialized:
            self._cash_var.set("")
        self._reload_positions()
        self._apply_order_lock_to_account_controls()
        self.refresh_operations_status()

    def _reload_positions(self):
        if not hasattr(self, "_tree"):
            return
        selected = self._selected_ticker()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._position_rows = {}
        source_labels = {"manual_adopted": "手動既有", "strategy_fill": "策略成交"}
        management_labels = {"unmanaged": "未接管", "active": "策略管理"}
        dashboard_positions = {
            str(row.get("ticker") or ""): dict(row)
            for row in list(self._account_dashboard_snapshot.get("positions") or [])
        }
        for base_row in list(self._snapshot.get("positions") or []):
            ticker = str(base_row.get("ticker") or "")
            row = dict(base_row)
            row.update(dashboard_positions.get(ticker, {}))
            # Editing eligibility belongs to the canonical account read model.
            row["has_sell_history"] = bool(base_row.get("has_sell_history"))
            self._position_rows[ticker] = row
            indicator_due = {str(item.get("ticker") or "") for item in list(self._indicator_snapshot.get("exits") or [])}
            forced_stop = {
                str(item.get("ticker") or "")
                for item in list(self._protection_snapshot.get("positions") or [])
                if bool(item.get("stop_forced_exit"))
            }
            if ticker in forced_stop:
                sell_signal = "STOP EXIT"
                action = "建議賣出｜依 Stop 規則自行至券商處理"
            elif ticker in indicator_due:
                sell_signal = "INDICATOR SELL"
                action = "建議賣出｜自行至券商處理，成交後回帳務中心登錄"
            else:
                sell_signal = "-"
                action = "持有｜依目前 Stop / Target / Trailing 管理" if str(row.get("source")) == "strategy_fill" else "手動持股｜無策略接管"
            self._tree.insert(
                "", "end", iid=ticker,
                values=(
                    "▣", ticker, f"{int(row.get('qty') or 0):,}",
                    format_trading_money(row.get("average_cost")),
                    format_trading_money(row.get("current_price")),
                    format_trading_money(row.get("effective_stop")),
                    format_trading_money(row.get("target_price")),
                    format_trading_money(row.get("trailing_stop")),
                    sell_signal, action,
                ),
            )
        self._fit_tree_rows(self._tree, len(self._position_rows))
        if selected and selected in self._position_rows:
            self._tree.selection_set(selected)
            self._tree.focus(selected)
        else:
            self._set_position_action_state(None)

    def _on_position_tree_click(self, event):
        region = self._tree.identify_region(event.x, event.y)
        if region != "cell" or self._tree.identify_column(event.x) != "#1":
            return None
        item_id = str(self._tree.identify_row(event.y) or "")
        if not item_id:
            return "break"
        self._open_ticker_in_inspector(item_id)
        return "break"

    def _selected_ticker(self):
        selected = self._tree.selection()
        return str(selected[0]) if selected else None

    def _set_position_action_state(self, row):
        editable = bool(
            row
            and row.get("source") == MANUAL_SOURCE
            and row.get("management_status") == UNMANAGED_STATUS
            and not bool(row.get("has_sell_history"))
            and not self._has_active_orders()
        )
        state = "normal" if editable else "disabled"
        self._correct_button.configure(state=state)
        self._remove_button.configure(state=state)

    def _on_position_selected(self, _event=None):
        ticker = self._selected_ticker()
        row = self._position_rows.get(ticker or "")
        self._set_position_action_state(row)
        if not row:
            return
        self._ticker_var.set(str(row.get("ticker") or ""))
        self._qty_var.set(str(int(row.get("qty") or 0)))
        self._cost_var.set(format_trading_money(row.get("remaining_cost_basis")))
        self._entry_date_var.set(str(row.get("entry_date") or ""))
        self._note_var.set("")

    def _clear_position_form(self):
        for variable in (self._ticker_var, self._qty_var, self._cost_var, self._entry_date_var, self._note_var):
            variable.set("")
        self._tree.selection_remove(*self._tree.selection())
        self._set_position_action_state(None)

    def _initialize_account(self):
        try:
            cash = parse_trading_money_text(self._cash_var.get(), "帳戶現金", allow_blank=True, allow_zero=True)
        except ValueError as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        self._submit_account_mutation(
            "初始化 Trading account",
            lambda: initialize_trading_account_state(WORKBENCH_PROJECT_ROOT, cash=cash),
            success_message="Trading account 已初始化。",
        )

    def _set_cash(self):
        try:
            cash = parse_trading_money_text(self._cash_var.get(), "帳戶現金", allow_blank=False, allow_zero=True)
            expected_revision = self._current_revision()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        self._submit_account_mutation(
            "更新 Trading 現金",
            lambda: set_trading_cash_balance(
                WORKBENCH_PROJECT_ROOT,
                cash=cash,
                expected_revision=None,
                note="Workbench cash reconciliation",
            ),
            success_message="現金餘額已更新。",
        )

    def _simple_trade_values(self):
        ticker = self._trade_ticker_var.get().strip().upper()
        if not ticker:
            ticker = self._selected_candidate_ticker() or ""
        if not ticker:
            raise ValueError("股票代號必填")
        qty = parse_trading_qty_text(self._trade_qty_var.get(), "成交股數")
        price = parse_trading_money_text(self._trade_price_var.get(), "成交價", allow_zero=False)
        trade_date = self._trade_date_var.get().strip()
        if not trade_date:
            raise ValueError("成交日必填")
        return ticker, qty, price, trade_date

    def _record_simple_trade(self, side: str):
        try:
            ticker, qty, price, trade_date = self._simple_trade_values()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("成交登錄", str(exc), parent=self)
            return
        if str(side).upper() != "BUY":
            messagebox.showerror("成交登錄", "賣出成交請到帳務中心點選庫存後操作。", parent=self)
            return

        candidate = self._selected_candidate_row()
        if candidate and str(candidate.get("ticker") or "").strip().upper() != ticker:
            candidate = None
        source_text = "Scanner 策略買入" if candidate else "自行買入"
        if not messagebox.askyesno(
            "確認買入成交",
            f"{ticker}｜{qty:,} 股 @ {price}｜{trade_date}\n\n來源：{source_text}\n價金、買入手續費與持有成本由帳務 SSOT 自動計算。",
            parent=self,
        ):
            return

        worker = lambda: record_trading_account_buy(
            WORKBENCH_PROJECT_ROOT,
            ticker=ticker,
            qty=qty,
            price=price,
            trade_date=trade_date,
            expected_account_revision=None,
            candidate=candidate,
        )

        def on_success(_result):
            self._trade_ticker_var.set("")
            self._trade_qty_var.set("")
            self._trade_price_var.set("")
            self._trade_date_var.set("")
            callback = getattr(self.winfo_toplevel(), "_open_accounting_center", None)
            if callable(callback):
                callback(refresh=True)
            messagebox.showinfo(
                "成交登錄",
                f"{ticker} 買入成交已寫入帳戶 SSOT；已切換至帳務中心並重新整理。",
                parent=self.winfo_toplevel(),
            )

        self._submit_trading_command(
            f"{ticker} 買入成交登錄",
            worker,
            on_success=on_success,
            error_title="成交登錄失敗",
            # Buying only changes account truth.  Do not rebuild the historical
            # Trading/OMS bundle before switching to Accounting Center.
            refresh_state=False,
        )

    def _position_form_values(self):
        ticker = self._ticker_var.get().strip().upper()
        if not ticker:
            raise ValueError("股票代號必填")
        return {
            "ticker": ticker,
            "qty": parse_trading_qty_text(self._qty_var.get()),
            "cost_basis_total": parse_trading_money_text(self._cost_var.get(), "剩餘成本總額", allow_zero=False),
            "entry_date": self._entry_date_var.get().strip() or None,
            "note": self._note_var.get().strip() or None,
        }

    def _adopt_position(self):
        try:
            values = self._position_form_values()
            expected_revision = self._current_revision()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        self._submit_account_mutation(
            f"登記既有持股 {values['ticker']}",
            lambda: adopt_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                expected_revision=None,
                **values,
            ),
            success_message="既有持股已登記；現金未變更。",
            clear_form=True,
        )

    def _correct_position(self):
        selected = self._selected_ticker()
        if not selected:
            messagebox.showerror("Trading 帳戶操作失敗", "請先選取要修正的持股。", parent=self)
            return
        try:
            values = self._position_form_values()
            if values["ticker"] != selected:
                raise ValueError("修正持股時不可變更股票代號；如代號輸入錯誤，請移除後重新新增。")
            expected_revision = self._current_revision()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        self._submit_account_mutation(
            f"修正既有持股 {selected}",
            lambda: correct_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                expected_revision=None,
                **values,
            ),
            success_message="既有持股 broker truth 已修正；現金未變更。",
        )

    def _remove_position(self):
        selected = self._selected_ticker()
        if not selected:
            messagebox.showerror("Trading 帳戶操作失敗", "請先選取要移除的持股。", parent=self)
            return
        row = self._position_rows.get(selected) or {}
        if not messagebox.askyesno(
            "移除既有持股",
            f"確定要移除 {selected}（{int(row.get('qty') or 0):,} 股）？\n\n這是 broker truth 修正，不會產生賣出交易，也不會改變 cash。",
            parent=self,
        ):
            return
        try:
            expected_revision = self._current_revision()
        except RuntimeError as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        note = self._note_var.get().strip() or "Workbench manual position removal"
        self._submit_account_mutation(
            f"移除既有持股 {selected}",
            lambda: remove_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                ticker=selected,
                expected_revision=None,
                note=note,
            ),
            success_message="既有持股已移除；現金未變更。",
            clear_form=True,
        )


__all__ = [
    "TradingAccountPanel",
    "build_trading_account_panel_snapshot",
    "format_trading_money",
    "parse_trading_money_text",
    "parse_trading_qty_text",
]
