from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from core.console_report import project_relative_display_path
from core.trading_policy import get_trading_policy_snapshot
from core.trading_order_state import (
    TRADING_ACTIVE_ORDER_STATUSES,
    TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
    TRADING_ORDER_SIDE_BUY,
)
from services.trading.daily_workflow import (
    build_trading_daily_workflow_snapshot,
    run_trading_candidate_scan,
    run_trading_daily_workflow,
    run_trading_market_data_update,
)
from services.trading.strategy_param_training import run_trading_strategy_param_training
from services.trading.order_planning import build_trading_proposed_order_plan
from services.trading.position_rollforward import run_trading_position_rollforward
from services.trading.operations_status import build_trading_operations_status
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
    confirm_trading_buy_order_fill,
    confirm_trading_protection_sell_order_fill,
    confirm_trading_indicator_sell_order_fill,
    recover_trading_fill_transaction,
)
from services.trading.order_state import (
    TradingOrderRevisionConflict,
    confirm_trading_order_cancellation,
    get_trading_order_read_model,
    resolve_trading_order_state_path,
)
from services.trading.account_state import (
    TradingAccountRevisionConflict,
    adopt_existing_trading_position,
    correct_existing_trading_position,
    get_trading_account_read_model,
    initialize_trading_account_state,
    load_trading_account_state,
    remove_existing_trading_position,
    resolve_trading_account_state_path,
    set_trading_cash_balance,
)
from services.workbench_ui.workbench import (
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_ENTRY_STYLE,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_HSCROLL_STYLE,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_TREE_STYLE,
    WORKBENCH_VSCROLL_STYLE,
)

WORKBENCH_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_SOURCE = "manual_adopted"
UNMANAGED_STATUS = "unmanaged"


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


class TradingAccountPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10, style=WORKBENCH_FRAME_STYLE)
        self._snapshot: dict[str, object] = {}
        self._position_rows: dict[str, dict[str, object]] = {}
        self._candidate_rows: list[dict[str, object]] = []
        self._proposed_order_rows: list[dict[str, object]] = []
        self._order_snapshot: dict[str, object] = {}
        self._order_rows: dict[str, dict[str, object]] = {}
        self._protection_snapshot: dict[str, object] = {}
        self._protection_rows: list[dict[str, object]] = []
        self._indicator_snapshot: dict[str, object] = {}
        self._indicator_rows: list[dict[str, object]] = []
        self._workflow_thread = None
        self._workflow_token = 0
        self._workflow_buttons = []
        self._workflow_action_buttons: dict[str, object] = {}
        self._operations_snapshot: dict[str, object] = {}
        self._build_ui()
        self.refresh_account()
        self.refresh_order_state()
        self.refresh_protection_plan()
        self.refresh_indicator_exit_plan()
        self.refresh_daily_workflow()
        self.refresh_operations_status()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)
        self.rowconfigure(6, weight=1)
        self.rowconfigure(7, weight=1)
        self.rowconfigure(8, weight=1)
        self.rowconfigure(9, weight=1)
        self.rowconfigure(10, weight=1)

        operations_box = ttk.LabelFrame(self, text="Trading 操作總覽", padding=10, style=WORKBENCH_LABELLF_STYLE)
        operations_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        operations_box.columnconfigure(0, weight=1)
        self._operations_status_var = tk.StringVar(value="讀取 Trading 整體狀態...")
        self._operations_next_var = tk.StringVar(value="下一步：-")
        self._operations_detail_var = tk.StringVar(value="-")
        self._live_audit_var = tk.StringVar(value="實盤就緒：尚未執行實盤就緒檢查")
        ttk.Label(operations_box, textvariable=self._operations_status_var, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(operations_box, textvariable=self._operations_next_var, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(operations_box, textvariable=self._operations_detail_var, foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(operations_box, textvariable=self._live_audit_var, foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=3, column=0, sticky="w", pady=(4, 0))
        operations_buttons = ttk.Frame(operations_box, style=WORKBENCH_FRAME_STYLE)
        operations_buttons.grid(row=0, column=1, rowspan=4, padx=(12, 0), sticky="n")
        ttk.Button(operations_buttons, text="全狀態刷新", command=self._refresh_all_trading_state, style=WORKBENCH_BUTTON_STYLE).pack(fill="x")
        ttk.Button(operations_buttons, text="實盤就緒檢查", command=self._run_operational_audit, style=WORKBENCH_BUTTON_STYLE).pack(fill="x", pady=(6, 0))

        workflow_box = ttk.LabelFrame(self, text="每日 Trading 流程", padding=10, style=WORKBENCH_LABELLF_STYLE)
        workflow_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        workflow_box.columnconfigure(0, weight=1)
        self._workflow_status_var = tk.StringVar(value="讀取 Trading workflow 狀態...")
        self._workflow_freshness_var = tk.StringVar(value="-")
        ttk.Label(workflow_box, textvariable=self._workflow_status_var, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(workflow_box, textvariable=self._workflow_freshness_var, foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=0, sticky="w", pady=(4, 0))
        workflow_buttons = ttk.Frame(workflow_box, style=WORKBENCH_FRAME_STYLE)
        workflow_buttons.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))
        for text, action in (
            ("1 更新資料", "data"),
            ("持股日終推進", "rollforward"),
            ("2 更新 Params", "params"),
            ("3 Scanner 候選", "scanner"),
            ("4 建議掛單", "orders"),
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
        ttk.Button(workflow_buttons, text="刷新狀態", command=self.refresh_daily_workflow, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(8, 0))
        ttk.Label(
            workflow_box,
            text="更新資料後，既有 strategy_fill 持股先用各 entry order frozen params 做日終推進；Scanner 只在 Trading Params 與 Trading data 同一最新交易日執行。",
            foreground=WORKBENCH_MUTED,
            style=WORKBENCH_LABEL_STYLE,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        header = ttk.LabelFrame(self, text="Trading 帳戶", padding=10, style=WORKBENCH_LABELLF_STYLE)
        header.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        self._status_var = tk.StringVar(value="讀取中...")
        self._policy_var = tk.StringVar(value="-")
        self._path_var = tk.StringVar(value="-")
        ttk.Label(header, textvariable=self._status_var, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self._policy_var, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, textvariable=self._path_var, style=WORKBENCH_LABEL_STYLE).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Button(header, text="重新整理", command=self.refresh_account, style=WORKBENCH_BUTTON_STYLE).grid(row=0, column=1, rowspan=2, padx=(12, 0))

        cash_box = ttk.LabelFrame(self, text="現金", padding=10, style=WORKBENCH_LABELLF_STYLE)
        cash_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(cash_box, text="帳戶現金", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        self._cash_var = tk.StringVar()
        self._cash_entry = ttk.Entry(cash_box, textvariable=self._cash_var, width=22, style=WORKBENCH_ENTRY_STYLE)
        self._cash_entry.grid(row=0, column=1, sticky="w", padx=(8, 8))
        self._initialize_button = ttk.Button(cash_box, text="初始化帳戶", command=self._initialize_account, style=WORKBENCH_BUTTON_STYLE)
        self._initialize_button.grid(row=0, column=2, padx=(0, 8))
        self._set_cash_button = ttk.Button(cash_box, text="更新現金", command=self._set_cash, style=WORKBENCH_BUTTON_STYLE)
        self._set_cash_button.grid(row=0, column=3)
        ttk.Label(cash_box, text="初始化可留空；更新現金會留下 revision event，不直接改檔。", foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        form = ttk.LabelFrame(self, text="既有持股（manual adopted broker truth）", padding=10, style=WORKBENCH_LABELLF_STYLE)
        form.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        labels = ("股票代號", "股數", "剩餘成本總額", "買入日 YYYY-MM-DD", "備註")
        for col, label in enumerate(labels):
            ttk.Label(form, text=label, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else 8, 0))
        self._ticker_var = tk.StringVar()
        self._qty_var = tk.StringVar()
        self._cost_var = tk.StringVar()
        self._entry_date_var = tk.StringVar()
        self._note_var = tk.StringVar()
        entries = (
            (self._ticker_var, 12),
            (self._qty_var, 12),
            (self._cost_var, 18),
            (self._entry_date_var, 18),
            (self._note_var, 36),
        )
        for col, (variable, width) in enumerate(entries):
            ttk.Entry(form, textvariable=variable, width=width, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0), pady=(4, 0))
            form.columnconfigure(col, weight=1 if col == 4 else 0)

        button_row = ttk.Frame(form, style=WORKBENCH_FRAME_STYLE)
        button_row.grid(row=2, column=0, columnspan=5, sticky="w", pady=(10, 0))
        self._add_button = ttk.Button(button_row, text="新增既有持股", command=self._adopt_position, style=WORKBENCH_BUTTON_STYLE)
        self._add_button.pack(side="left")
        self._correct_button = ttk.Button(button_row, text="修正選取持股", command=self._correct_position, style=WORKBENCH_BUTTON_STYLE)
        self._correct_button.pack(side="left", padx=(8, 0))
        self._remove_button = ttk.Button(button_row, text="移除選取持股", command=self._remove_position, style=WORKBENCH_BUTTON_STYLE)
        self._remove_button.pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="清除輸入", command=self._clear_position_form, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(8, 0))
        ttk.Label(form, text="修正／移除只適用尚未有賣出歷史、尚未由策略接管的 manual adopted 持股；不改 cash。", foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))

        table_box = ttk.LabelFrame(self, text="目前持股", padding=8, style=WORKBENCH_LABELLF_STYLE)
        table_box.grid(row=5, column=0, sticky="nsew", pady=(0, 8))
        table_box.rowconfigure(0, weight=1)
        table_box.columnconfigure(0, weight=1)
        columns = ("ticker", "source", "qty", "avg_cost", "remaining_cost", "realized_pnl", "entry_date", "management")
        self._tree = ttk.Treeview(table_box, columns=columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse")
        headings = {
            "ticker": "股票",
            "source": "來源",
            "qty": "股數",
            "avg_cost": "平均成本",
            "remaining_cost": "剩餘成本",
            "realized_pnl": "已實現PnL",
            "entry_date": "買入日",
            "management": "策略管理",
        }
        widths = {"ticker": 90, "source": 120, "qty": 90, "avg_cost": 110, "remaining_cost": 130, "realized_pnl": 120, "entry_date": 120, "management": 100}
        for key in columns:
            self._tree.heading(key, text=headings[key])
            self._tree.column(key, width=widths[key], anchor="center")
        scroll = ttk.Scrollbar(table_box, orient="vertical", command=self._tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self._tree.bind("<<TreeviewSelect>>", self._on_position_selected)

        candidate_box = ttk.LabelFrame(self, text="今日 Scanner 候選（原始策略候選）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        candidate_box.grid(row=6, column=0, sticky="nsew")
        candidate_box.rowconfigure(0, weight=1)
        candidate_box.columnconfigure(0, weight=1)
        candidate_columns = ("rank", "ticker", "kind", "sort", "ev", "proj_cost", "detail")
        self._candidate_tree = ttk.Treeview(candidate_box, columns=candidate_columns, show="headings", style=WORKBENCH_TREE_STYLE)
        candidate_headings = {"rank": "排名", "ticker": "股票", "kind": "類型", "sort": "排序值", "ev": "EV", "proj_cost": "參考投入", "detail": "Scanner 摘要"}
        candidate_widths = {"rank": 60, "ticker": 80, "kind": 110, "sort": 100, "ev": 90, "proj_cost": 110, "detail": 700}
        for key in candidate_columns:
            self._candidate_tree.heading(key, text=candidate_headings[key])
            self._candidate_tree.column(key, width=candidate_widths[key], anchor="w" if key == "detail" else "center")
        candidate_y = ttk.Scrollbar(candidate_box, orient="vertical", command=self._candidate_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        candidate_x = ttk.Scrollbar(candidate_box, orient="horizontal", command=self._candidate_tree.xview, style=WORKBENCH_HSCROLL_STYLE)
        self._candidate_tree.configure(yscrollcommand=candidate_y.set, xscrollcommand=candidate_x.set)
        self._candidate_tree.grid(row=0, column=0, sticky="nsew")
        candidate_y.grid(row=0, column=1, sticky="ns")
        candidate_x.grid(row=1, column=0, sticky="ew")

        proposed_box = ttk.LabelFrame(self, text="建議掛單（尚未送單／尚未成交）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        proposed_box.grid(row=7, column=0, sticky="nsew", pady=(8, 0))
        proposed_box.rowconfigure(1, weight=1)
        proposed_box.columnconfigure(0, weight=1)
        self._proposed_status_var = tk.StringVar(value="尚未產生建議掛單。")
        ttk.Label(proposed_box, textvariable=self._proposed_status_var, foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w", pady=(0, 6))
        proposed_columns = ("rank", "ticker", "kind", "limit", "qty", "reserved", "stop", "target")
        self._proposed_tree = ttk.Treeview(proposed_box, columns=proposed_columns, show="headings", style=WORKBENCH_TREE_STYLE)
        proposed_headings = {"rank": "順位", "ticker": "股票", "kind": "類型", "limit": "買入限價", "qty": "股數", "reserved": "預留資金", "stop": "初始Stop", "target": "Target"}
        proposed_widths = {"rank": 60, "ticker": 80, "kind": 110, "limit": 100, "qty": 90, "reserved": 120, "stop": 100, "target": 100}
        for key in proposed_columns:
            self._proposed_tree.heading(key, text=proposed_headings[key])
            self._proposed_tree.column(key, width=proposed_widths[key], anchor="center")
        proposed_y = ttk.Scrollbar(proposed_box, orient="vertical", command=self._proposed_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        self._proposed_tree.configure(yscrollcommand=proposed_y.set)
        self._proposed_tree.grid(row=1, column=0, sticky="nsew")
        proposed_y.grid(row=1, column=1, sticky="ns")

        submit_row = ttk.Frame(proposed_box, style=WORKBENCH_FRAME_STYLE)
        submit_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
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

        pending_box = ttk.LabelFrame(self, text="券商掛單狀態（ORDERED / PARTIAL / FILLED / CANCELLED）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        pending_box.grid(row=8, column=0, sticky="nsew", pady=(8, 0))
        pending_box.rowconfigure(1, weight=1)
        pending_box.columnconfigure(0, weight=1)
        self._order_status_var = tk.StringVar(value="尚無實際送單紀錄。")
        ttk.Label(pending_box, textvariable=self._order_status_var, foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w", pady=(0, 6))
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
        order_y = ttk.Scrollbar(pending_box, orient="vertical", command=self._order_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        order_x = ttk.Scrollbar(pending_box, orient="horizontal", command=self._order_tree.xview, style=WORKBENCH_HSCROLL_STYLE)
        self._order_tree.configure(yscrollcommand=order_y.set, xscrollcommand=order_x.set)
        self._order_tree.grid(row=1, column=0, sticky="nsew")
        order_y.grid(row=1, column=1, sticky="ns")
        order_x.grid(row=2, column=0, sticky="ew")
        fill_row = ttk.Frame(pending_box, style=WORKBENCH_FRAME_STYLE)
        fill_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(fill_row, text="本次成交股數", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_qty_var = tk.StringVar()
        ttk.Entry(fill_row, textvariable=self._fill_qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(fill_row, text="本次成交價", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_price_var = tk.StringVar()
        ttk.Entry(fill_row, textvariable=self._fill_price_var, width=12, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(fill_row, text="成交日 YYYY-MM-DD", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_date_var = tk.StringVar()
        ttk.Entry(fill_row, textvariable=self._fill_date_var, width=14, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
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
        ttk.Button(pending_buttons, text="刷新掛單狀態", command=self.refresh_order_state, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(8, 0))
        ttk.Label(
            pending_buttons,
            text="成交只接受券商實際股數／價格；PARTIAL 仍鎖定未成交餘額，FILLED 才解除 active order。",
            foreground=WORKBENCH_MUTED,
            style=WORKBENCH_LABEL_STYLE,
        ).pack(side="left", padx=(12, 0))

        protection_box = ttk.LabelFrame(self, text="成交後 Stop / TP 保護單計畫（logical plan；送單狀態見券商掛單表）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        protection_box.grid(row=9, column=0, sticky="nsew", pady=(8, 0))
        protection_box.rowconfigure(1, weight=1)
        protection_box.columnconfigure(0, weight=1)
        self._protection_status_var = tk.StringVar(value="尚未建立成交後保護單計畫。")
        ttk.Label(
            protection_box,
            textvariable=self._protection_status_var,
            foreground=WORKBENCH_MUTED,
            style=WORKBENCH_LABEL_STYLE,
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
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
        protection_y = ttk.Scrollbar(
            protection_box, orient="vertical", command=self._protection_tree.yview, style=WORKBENCH_VSCROLL_STYLE
        )
        self._protection_tree.configure(yscrollcommand=protection_y.set)
        self._protection_tree.grid(row=1, column=0, sticky="nsew")
        protection_y.grid(row=1, column=1, sticky="ns")
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
            command=self.refresh_protection_plan,
            style=WORKBENCH_BUTTON_STYLE,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            protection_buttons,
            text="只由 confirmed strategy fill 的 canonical position state＋ORDERED 時 frozen params 機械派生；不讀成交後行情、不代表券商已掛出 Stop/TP。",
            foreground=WORKBENCH_MUTED,
            style=WORKBENCH_LABEL_STYLE,
        ).pack(side="left", padx=(12, 0))

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
        ttk.Label(
            protection_box,
            text="系統不預設券商支援 OCO；只有你明確輸入實際券商 OCO/互斥群組 ID 時才允許 Stop full + TP 同時超額共享同一持股。尚未送券商的 logical plan 仍不是 broker truth。",
            foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        indicator_box = ttk.LabelFrame(self, text="Indicator SELL 計畫", padding=8, style=WORKBENCH_LABELLF_STYLE)
        indicator_box.grid(row=10, column=0, sticky="nsew", pady=(8, 0))
        indicator_box.rowconfigure(1, weight=1)
        indicator_box.columnconfigure(0, weight=1)
        self._indicator_status_var = tk.StringVar(value="尚未建立 Indicator SELL 計畫。")
        ttk.Label(indicator_box, textvariable=self._indicator_status_var, foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w", pady=(0, 6))
        columns=("ticker","signal","qty","entry_date","type","carried")
        self._indicator_tree=ttk.Treeview(indicator_box, columns=columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse")
        headings={"ticker":"股票","signal":"Signal日","qty":"賣出股數","entry_date":"Entry日","type":"委託","carried":"狀態"}
        widths={"ticker":90,"signal":100,"qty":100,"entry_date":100,"type":90,"carried":110}
        for key in columns:
            self._indicator_tree.heading(key,text=headings[key]); self._indicator_tree.column(key,width=widths[key],anchor="center")
        iy=ttk.Scrollbar(indicator_box,orient="vertical",command=self._indicator_tree.yview,style=WORKBENCH_VSCROLL_STYLE)
        self._indicator_tree.configure(yscrollcommand=iy.set); self._indicator_tree.grid(row=1,column=0,sticky="nsew"); iy.grid(row=1,column=1,sticky="ns")
        buttons=ttk.Frame(indicator_box,style=WORKBENCH_FRAME_STYLE); buttons.grid(row=2,column=0,columnspan=2,sticky="w",pady=(8,0))
        ttk.Button(buttons,text="建立／刷新 Indicator SELL 計畫",command=self._rebuild_indicator_exit_plan,style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        ttk.Button(buttons,text="刷新 Indicator SELL 狀態",command=self.refresh_indicator_exit_plan,style=WORKBENCH_BUTTON_STYLE).pack(side="left",padx=(8,0))
        ttk.Label(buttons,text="券商委託號",style=WORKBENCH_LABEL_STYLE).pack(side="left",padx=(14,0))
        self._indicator_broker_id_var=tk.StringVar()
        ttk.Entry(buttons,textvariable=self._indicator_broker_id_var,width=16,style=WORKBENCH_ENTRY_STYLE).pack(side="left",padx=(6,8))
        ttk.Button(buttons,text="確認選取 MARKET SELL 已送單",command=self._confirm_indicator_exit_submitted,style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        ttk.Label(indicator_box,text="Signal 只由 completed bar + source entry frozen params 產生；計畫不是券商送單，實際成交仍須在掛單表輸入 broker fill。",foreground=WORKBENCH_MUTED,style=WORKBENCH_LABEL_STYLE).grid(row=3,column=0,columnspan=2,sticky="w",pady=(6,0))

    def _reload_protection_rows(self, rows):
        for item in self._protection_tree.get_children():
            self._protection_tree.delete(item)
        self._protection_rows = [dict(row) for row in list(rows or [])]
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
        try:
            confirm_trading_protection_leg_submission(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                action=action,
                expected_order_revision=int(self._current_order_revision()),
                broker_order_id=self._protection_broker_id_var.get().strip() or None,
                note="Workbench confirmed protection SELL submission",
            )
        except (TradingOrderRevisionConflict, ValueError, RuntimeError, FileNotFoundError) as exc:
            messagebox.showerror("Trading 保護 SELL 送單失敗", str(exc), parent=self)
            self.refresh_order_state()
            self.refresh_protection_plan()
            return
        self._protection_broker_id_var.set("")
        self.refresh_order_state()
        self.refresh_protection_plan()
        messagebox.showinfo("Trading 保護 SELL", f"{ticker} {label} 已記錄為 ORDERED；account 未修改。", parent=self)

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
        try:
            confirm_trading_protection_oco_submission(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                expected_order_revision=int(self._current_order_revision()),
                broker_oco_group_id=group_id,
                stop_broker_order_id=self._protection_stop_broker_id_var.get().strip() or None,
                tp_broker_order_id=self._protection_tp_broker_id_var.get().strip() or None,
                note="Workbench confirmed broker-native OCO protection submission",
            )
        except (TradingOrderRevisionConflict, ValueError, RuntimeError, FileNotFoundError) as exc:
            messagebox.showerror("Trading 保護 OCO 送單失敗", str(exc), parent=self)
            self.refresh_order_state()
            self.refresh_protection_plan()
            return
        self._protection_oco_group_var.set("")
        self._protection_stop_broker_id_var.set("")
        self._protection_tp_broker_id_var.set("")
        self.refresh_order_state()
        self.refresh_protection_plan()
        messagebox.showinfo("Trading 保護 OCO", f"{ticker} Stop+TP 已依使用者確認記錄為券商 OCO ORDERED；account 未修改。", parent=self)

    def refresh_protection_plan(self):
        try:
            snapshot = get_trading_protection_plan_read_model(WORKBENCH_PROJECT_ROOT)
        except (ValueError, RuntimeError, OSError) as exc:
            self._protection_snapshot = {}
            self._reload_protection_rows([])
            self._protection_status_var.set(f"保護單計畫讀取失敗：{exc}")
            return
        self._protection_snapshot = snapshot
        self._reload_protection_rows(snapshot.get("positions") or [])
        if not snapshot.get("exists"):
            self._protection_status_var.set(
                f"尚未建立保護單計畫 | {snapshot.get('json_path') or '-'}"
            )
            self.refresh_operations_status()
            return
        freshness = "FRESH" if snapshot.get("fresh") else "STALE"
        skipped = list(snapshot.get("manual_positions_skipped") or [])
        suffix = f" | manual未接管 {','.join(skipped)}" if skipped else ""
        self._protection_status_var.set(
            f"{freshness} | {snapshot.get('status') or '-'} / {snapshot.get('broker_status') or '-'} | "
            f"positions {int(snapshot.get('position_count') or 0)} | {snapshot.get('text_path') or '-'}{suffix}"
        )
        self.refresh_operations_status()

    def _rebuild_protection_plan(self):
        try:
            result = build_trading_protection_plan(WORKBENCH_PROJECT_ROOT)
        except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
            messagebox.showerror("Trading 保護單計畫", str(exc), parent=self)
            self.refresh_protection_plan()
            return None
        self.refresh_protection_plan()
        self._protection_status_var.set(
            f"FRESH | {result.get('status')} / {result.get('broker_status')} | positions {len(result.get('positions') or [])} | {result.get('text_path') or '-'}"
        )
        return result

    def _reload_indicator_rows(self, rows):
        for item in self._indicator_tree.get_children():
            self._indicator_tree.delete(item)
        self._indicator_rows=[dict(row) for row in list(rows or [])]
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
            snapshot=get_trading_indicator_exit_plan_read_model(WORKBENCH_PROJECT_ROOT)
        except (ValueError,RuntimeError,OSError) as exc:
            self._indicator_snapshot={}; self._reload_indicator_rows([]); self._indicator_status_var.set(f"Indicator SELL 計畫讀取失敗：{exc}"); return
        self._indicator_snapshot=snapshot; self._reload_indicator_rows(snapshot.get("exits") or [])
        if not snapshot.get("exists"):
            self._indicator_status_var.set(f"尚未建立 Indicator SELL 計畫 | {snapshot.get('json_path') or '-'}"); return
        self._indicator_status_var.set(f"{'FRESH' if snapshot.get('fresh') else 'STALE'} | exits {int(snapshot.get('exit_count') or 0)} | active broker Indicator SELL {int(snapshot.get('active_indicator_exit_order_count') or 0)} | {snapshot.get('text_path') or '-'}")

    def _rebuild_indicator_exit_plan(self):
        try:
            result=build_trading_indicator_exit_plan(WORKBENCH_PROJECT_ROOT)
        except (ValueError,RuntimeError,OSError,FileNotFoundError) as exc:
            messagebox.showerror("Trading Indicator SELL 計畫",str(exc),parent=self); self.refresh_indicator_exit_plan(); return None
        self.refresh_indicator_exit_plan(); self.refresh_operations_status(); return result

    def _confirm_indicator_exit_submitted(self):
        row=self._selected_indicator_row()
        if not row:
            messagebox.showerror("Trading Indicator SELL","請先選取一筆 Indicator SELL 計畫。",parent=self); return
        ticker=str(row.get("ticker") or "")
        if not messagebox.askyesno("確認 Indicator MARKET SELL 已送券商",f"確認已在券商實際送出 {ticker} 全倉 MARKET SELL？\n\n此動作只建立 ORDERED broker truth，不代表成交；若仍有 active Stop/TP 必須先在券商取消並於掛單表確認。",parent=self): return
        try:
            confirm_trading_indicator_exit_submission(WORKBENCH_PROJECT_ROOT,signal_key=str(row.get("signal_key") or ""),expected_order_revision=int(self._current_order_revision()),broker_order_id=self._indicator_broker_id_var.get().strip() or None,note="Workbench confirmed Indicator MARKET SELL submission")
        except (TradingOrderRevisionConflict,ValueError,RuntimeError,FileNotFoundError) as exc:
            messagebox.showerror("Trading Indicator SELL 送單失敗",str(exc),parent=self); self.refresh_order_state(); self.refresh_indicator_exit_plan(); return
        self._indicator_broker_id_var.set(""); self.refresh_order_state(); self.refresh_indicator_exit_plan(); self.refresh_operations_status()
        messagebox.showinfo("Trading Indicator SELL",f"{ticker} Indicator MARKET SELL 已記錄為 ORDERED；account 未修改。",parent=self)

    def refresh_operations_status(self):
        try:
            snapshot = build_trading_operations_status(WORKBENCH_PROJECT_ROOT)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self._operations_snapshot = {}
            self._operations_status_var.set(f"BLOCKED | Trading 整體狀態讀取失敗：{exc}")
            self._operations_next_var.set("下一步：先修正狀態讀取錯誤")
            self._operations_detail_var.set("-")
            self._set_workflow_buttons_state("disabled")
            return
        self._operations_snapshot = snapshot
        self._operations_status_var.set(
            f"{snapshot.get('overall_status') or '-'} | Data {snapshot.get('latest_data_date') or '-'} | "
            f"Account rev {snapshot.get('account_revision') if snapshot.get('account_revision') is not None else '-'} | "
            f"持股 strategy/manual {int(snapshot.get('strategy_position_count') or 0)}/{int(snapshot.get('manual_position_count') or 0)} | "
            f"Active BUY/protection/indicator {int(snapshot.get('active_entry_order_count') or 0)}/{int(snapshot.get('active_protection_order_count') or 0)}/{int(snapshot.get('active_indicator_exit_order_count') or 0)}"
        )
        self._operations_next_var.set(
            f"下一步：{snapshot.get('next_action_label') or '-'} | {snapshot.get('next_action_detail') or '-'}"
        )
        details = [
            f"Scanner {'FRESH' if snapshot.get('candidate_snapshot_fresh') else 'STALE/EMPTY'}({int(snapshot.get('candidate_count') or 0)})",
            f"Proposed {'FRESH' if snapshot.get('proposed_orders_fresh') else 'STALE/EMPTY'}({int(snapshot.get('proposed_order_count') or 0)})",
            f"Protection {'FRESH' if snapshot.get('protection_plan_fresh') else 'STALE/EMPTY'}",
            f"Indicator {'FRESH' if snapshot.get('indicator_exit_plan_fresh') else 'STALE/EMPTY'}",
        ]
        rollforward_due = list(snapshot.get('rollforward_due_tickers') or [])
        if rollforward_due:
            details.append("待持股日終推進: " + ",".join(rollforward_due))
        stale_protection = list(snapshot.get('stale_active_protection_tickers') or [])
        if stale_protection:
            details.append("保護單待取消/重送: " + ",".join(stale_protection))
        forced_stop = list(snapshot.get('forced_stop_exit_tickers') or [])
        if forced_stop:
            details.append("STOP已觸發/剩餘須退出: " + ",".join(forced_stop))
        missing_stop = list(snapshot.get('missing_stop_tickers') or [])
        if missing_stop:
            details.append("缺 active Stop: " + ",".join(missing_stop))
        warnings = list(snapshot.get('warnings') or [])
        blockers = list(snapshot.get('blockers') or [])
        if blockers:
            details.append("BLOCK: " + "；".join(blockers))
        elif warnings:
            details.append("注意: " + "；".join(warnings[:3]))
        self._operations_detail_var.set(" | ".join(details))
        self._apply_workflow_action_availability()

    def _run_operational_audit(self):
        try:
            audit = run_trading_operational_audit(WORKBENCH_PROJECT_ROOT)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self._live_audit_var.set(f"實盤就緒：LIVE_BLOCKED｜Audit 失敗：{exc}")
            messagebox.showerror("Trading 實盤就緒檢查失敗", str(exc), parent=self)
            return
        blockers = list(audit.get("blockers") or [])
        warnings = list(audit.get("warnings") or [])
        report_path = audit.get("markdown_path") or "-"
        if blockers:
            summary = "；".join(blockers[:2])
            self._live_audit_var.set(f"實盤就緒：{audit.get('status')}｜{summary}")
            messagebox.showwarning(
                "Trading 尚不可實盤",
                f"{summary}\n\n完整報告：{report_path}",
                parent=self,
            )
        else:
            suffix = f"｜警告 {len(warnings)} 項" if warnings else ""
            self._live_audit_var.set(f"實盤就緒：{audit.get('status')}{suffix}｜{report_path}")
            messagebox.showinfo("Trading 實盤就緒檢查", f"{audit.get('status')}\n\n報告：{report_path}", parent=self)

    def _refresh_all_trading_state(self):
        recovery_error = None
        try:
            recover_trading_fill_transaction(WORKBENCH_PROJECT_ROOT)
        except TradingFillRevisionConflict as exc:
            recovery_error = str(exc)
        self.refresh_account()
        self.refresh_order_state()
        self.refresh_protection_plan()
        self.refresh_indicator_exit_plan()
        self.refresh_daily_workflow()
        self.refresh_operations_status()
        if recovery_error:
            messagebox.showerror(
                "Trading fill recovery 失敗",
                recovery_error,
                parent=self,
            )

    def _apply_workflow_action_availability(self):
        if self._workflow_thread is not None and self._workflow_thread.is_alive():
            self._set_workflow_buttons_state("disabled")
            return
        availability = dict(self._operations_snapshot.get("workflow_action_availability") or {})
        for action, button in self._workflow_action_buttons.items():
            button.configure(state="normal" if bool(availability.get(action)) else "disabled")
        self._confirm_ordered_button.configure(
            state="normal" if bool(self._operations_snapshot.get("entry_submission_allowed")) else "disabled"
        )

    def _set_workflow_buttons_state(self, state: str):
        for button in self._workflow_buttons:
            button.configure(state=state)

    def refresh_daily_workflow(self):
        try:
            snapshot = build_trading_daily_workflow_snapshot(WORKBENCH_PROJECT_ROOT)
        except (OSError, ValueError, RuntimeError) as exc:
            self._workflow_status_var.set(f"Workflow 狀態讀取失敗：{exc}")
            self._workflow_freshness_var.set("-")
            return
        latest = snapshot.get("latest_data_date") or "尚無資料"
        param_latest = snapshot.get("param_latest_data_date") or "尚無 Params"
        ready = bool(snapshot.get("params_ready_for_scan"))
        self._workflow_status_var.set(
            f"{'READY' if ready else 'NOT READY'} | Data {latest} | Params {param_latest} | selector {snapshot.get('param_selector') or '-'}"
        )
        member_count = int(snapshot.get("param_member_count") or 0)
        param_error = snapshot.get("param_error")
        suffix = f" | member {member_count}" if snapshot.get("selected_params_exists") else ""
        if param_error:
            suffix += f" | {param_error}"
        self._workflow_freshness_var.set(
            f"Data: {snapshot.get('data_dir')} | Params: {snapshot.get('selected_params_path')} | Scanner: {snapshot.get('scanner_output_dir')}{suffix}"
        )
        self.refresh_operations_status()

    def _start_workflow_action(self, action: str):
        if self._workflow_thread is not None and self._workflow_thread.is_alive():
            self._workflow_status_var.set("Trading workflow 執行中；請等待目前工作完成。")
            return
        labels = {"data": "更新 Trading 資料", "rollforward": "持股日終推進", "params": "更新 Trading Params", "scanner": "Scanner 候選", "orders": "產生建議掛單", "all": "每日流程 1→2→3"}
        if action not in labels:
            messagebox.showerror("Trading workflow", f"未知 workflow action: {action}", parent=self)
            return
        availability = dict(self._operations_snapshot.get("workflow_action_availability") or {})
        if availability and not bool(availability.get(action)):
            messagebox.showerror(
                "Trading workflow 尚不可執行",
                f"{labels[action]} 目前被狀態契約阻擋。\n\n下一步：{self._operations_snapshot.get('next_action_label') or '-'}\n{self._operations_snapshot.get('next_action_detail') or ''}",
                parent=self,
            )
            return
        self._workflow_token += 1
        token = self._workflow_token
        self._set_workflow_buttons_state("disabled")
        self._workflow_status_var.set(f"執行中：{labels[action]}")
        thread = threading.Thread(
            target=self._run_workflow_worker,
            args=(action, token),
            name=f"workbench-trading-{action}",
            daemon=True,
        )
        self._workflow_thread = thread
        thread.start()

    def _run_workflow_worker(self, action: str, token: int):
        try:
            if action == "data":
                result = run_trading_market_data_update(project_root=WORKBENCH_PROJECT_ROOT)
            elif action == "rollforward":
                result = run_trading_position_rollforward(project_root=WORKBENCH_PROJECT_ROOT)
                if str(result.get("status") or "") != "NO_ACCOUNT":
                    build_trading_indicator_exit_plan(WORKBENCH_PROJECT_ROOT)
            elif action == "params":
                result = run_trading_strategy_param_training(project_root=WORKBENCH_PROJECT_ROOT)
            elif action == "scanner":
                result = run_trading_candidate_scan(project_root=WORKBENCH_PROJECT_ROOT)
            elif action == "orders":
                result = build_trading_proposed_order_plan(project_root=WORKBENCH_PROJECT_ROOT)
            else:
                result = run_trading_daily_workflow(project_root=WORKBENCH_PROJECT_ROOT)
        except Exception as exc:
            self.after(0, self._finish_workflow_error, action, token, exc)
            return
        self.after(0, self._finish_workflow_success, action, token, result)

    def _finish_workflow_error(self, action: str, token: int, exc: Exception):
        if token != self._workflow_token:
            return
        self._workflow_thread = None
        self.refresh_daily_workflow()
        self._workflow_status_var.set(f"FAIL：{type(exc).__name__}: {exc}")
        self.refresh_operations_status()
        messagebox.showerror("Trading workflow 失敗", f"{type(exc).__name__}: {exc}", parent=self)

    @staticmethod
    def _format_candidate_number(value, *, digits=2):
        if value is None:
            return "-"
        try:
            return f"{float(value):,.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def _reload_candidate_rows(self, rows):
        for item in self._candidate_tree.get_children():
            self._candidate_tree.delete(item)
        self._candidate_rows = [dict(row) for row in list(rows or [])]
        kind_labels = {"buy": "新訊號", "extended": "延續", "extended_tbd": "延續(TBD)"}
        for idx, row in enumerate(self._candidate_rows, 1):
            self._candidate_tree.insert(
                "",
                "end",
                values=(
                    idx,
                    row.get("ticker") or "-",
                    kind_labels.get(str(row.get("kind") or ""), str(row.get("kind") or "-")),
                    self._format_candidate_number(row.get("sort_value"), digits=4),
                    self._format_candidate_number(row.get("expected_value", row.get("ev")), digits=3),
                    self._format_candidate_number(row.get("proj_cost"), digits=0),
                    row.get("text") or "",
                ),
            )

    def _reload_proposed_order_rows(self, rows):
        for item in self._proposed_tree.get_children():
            self._proposed_tree.delete(item)
        self._proposed_order_rows = [dict(row) for row in list(rows or [])]
        for row in self._proposed_order_rows:
            iid = f"{int(row.get('rank') or 0)}:{str(row.get('ticker') or '')}"
            self._proposed_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    int(row.get("rank") or 0),
                    row.get("ticker") or "-",
                    row.get("kind") or "-",
                    self._format_candidate_number(row.get("limit_price"), digits=2),
                    f"{int(row.get('qty') or 0):,}",
                    self._format_candidate_number(row.get("reserved_cost"), digits=0),
                    self._format_candidate_number(row.get("init_sl"), digits=2),
                    self._format_candidate_number(row.get("target_price"), digits=2),
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
            snapshot = get_trading_order_read_model(WORKBENCH_PROJECT_ROOT)
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
        active = int(snapshot.get("active_order_count") or 0)
        revision = snapshot.get("revision")
        state_path = project_relative_display_path(resolve_trading_order_state_path(WORKBENCH_PROJECT_ROOT), project_root=WORKBENCH_PROJECT_ROOT)
        self._order_status_var.set(
            f"{'ACTIVE' if active else 'CLEAR'} | revision {revision if revision is not None else '-'} | active {active} "
            f"(BUY {int(snapshot.get('active_entry_order_count') or 0)} / protection SELL {int(snapshot.get('active_protection_order_count') or 0)}) | "
            f"總紀錄 {int(snapshot.get('order_count') or 0)} | {state_path}"
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
        try:
            confirm_trading_order_submission(
                WORKBENCH_PROJECT_ROOT,
                rank=int(row.get("rank") or 0),
                ticker=ticker,
                expected_revision=self._current_order_revision(),
                broker_order_id=self._broker_order_id_var.get().strip() or None,
                note=self._order_note_var.get().strip() or None,
            )
        except TradingOrderRevisionConflict as exc:
            messagebox.showerror("Trading 掛單已更新", f"{exc}\n\n已重新讀取最新掛單狀態。", parent=self)
            self.refresh_order_state()
            return
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            messagebox.showerror("Trading 掛單失敗", str(exc), parent=self)
            self.refresh_order_state()
            return
        self._broker_order_id_var.set("")
        self._order_note_var.set("")
        self.refresh_order_state()
        messagebox.showinfo("Trading 掛單", f"{ticker} 已記錄為 ORDERED；account cash／positions 未變更。", parent=self)

    def _confirm_selected_fill(self):
        selected = self._order_tree.selection()
        if not selected:
            messagebox.showerror("Trading 成交", "請先選取一筆 ORDERED / PARTIAL 掛單。", parent=self)
            return
        order_id = str(selected[0])
        row = self._order_rows.get(order_id) or {}
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
        try:
            if str(row.get("side") or "BUY") == "BUY":
                fill_fn = confirm_trading_buy_order_fill
            elif str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_INDICATOR_EXIT:
                fill_fn = confirm_trading_indicator_sell_order_fill
            else:
                fill_fn = confirm_trading_protection_sell_order_fill
            result = fill_fn(
                WORKBENCH_PROJECT_ROOT,
                order_id=order_id,
                fill_qty=fill_qty,
                fill_price=fill_price,
                trade_date=trade_date,
                expected_order_revision=int(self._current_order_revision()),
                expected_account_revision=int(self._current_revision()),
            )
        except TradingFillRevisionConflict as exc:
            messagebox.showerror("Trading 成交狀態已更新", f"{exc}\n\n已重新讀取最新 account／order state。", parent=self)
            self.refresh_account()
            self.refresh_order_state()
            return
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            messagebox.showerror("Trading 成交失敗", str(exc), parent=self)
            self.refresh_account()
            self.refresh_order_state()
            return
        self._fill_qty_var.set("")
        self._fill_price_var.set("")
        self._fill_date_var.set("")
        self.refresh_account()
        self.refresh_order_state()
        protection_error = None
        try:
            build_trading_protection_plan(WORKBENCH_PROJECT_ROOT)
        except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
            protection_error = str(exc)
        self.refresh_protection_plan()
        self.refresh_indicator_exit_plan()
        message = (
            f"{ticker} 已更新為 {result.get('status')}；累計成交 {int(result.get('filled_qty') or 0):,}，"
            f"未成交 {int(result.get('remaining_qty') or 0):,}。"
        )
        if protection_error:
            message += f"\n\n成交已入帳，但保護單計畫建立失敗：{protection_error}"
        else:
            message += "\n\n已依實際成交後 canonical position state 機械刷新 Stop / TP 保護單計畫；尚未送券商。"
        messagebox.showinfo("Trading 成交", message, parent=self)

    def _cancel_selected_order(self):
        selected = self._order_tree.selection()
        if not selected:
            messagebox.showerror("Trading 掛單", "請先選取一筆 ORDERED / PARTIAL 掛單。", parent=self)
            return
        order_id = str(selected[0])
        row = self._order_rows.get(order_id) or {}
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
        try:
            confirm_trading_order_cancellation(
                WORKBENCH_PROJECT_ROOT,
                order_id=order_id,
                expected_revision=int(self._current_order_revision()),
                note=self._order_note_var.get().strip() or "Workbench confirmed broker cancellation",
            )
        except TradingOrderRevisionConflict as exc:
            messagebox.showerror("Trading 掛單已更新", f"{exc}\n\n已重新讀取最新掛單狀態。", parent=self)
            self.refresh_order_state()
            return
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            messagebox.showerror("Trading 掛單失敗", str(exc), parent=self)
            self.refresh_order_state()
            return
        self._order_note_var.set("")
        self.refresh_order_state()
        protection_refresh_error = None
        if int(row.get("filled_qty") or 0) > 0:
            try:
                build_trading_protection_plan(WORKBENCH_PROJECT_ROOT)
            except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
                protection_refresh_error = str(exc)
        self.refresh_protection_plan()
        self.refresh_indicator_exit_plan()
        message = f"{ticker} 已記錄為 CANCELLED。"
        if protection_refresh_error:
            message += f"\n\n取消已記錄，但保護單計畫刷新失敗：{protection_refresh_error}"
        messagebox.showinfo("Trading 掛單", message, parent=self)

    def _finish_workflow_success(self, action: str, token: int, result):
        if token != self._workflow_token:
            return
        self._workflow_thread = None
        scan_result = dict(result.get("scanner") or {}) if action == "all" else (dict(result) if action == "scanner" else {})
        if scan_result:
            self._reload_candidate_rows(scan_result.get("candidate_rows") or [])
        if action == "orders":
            order_result = dict(result)
            self._reload_proposed_order_rows(order_result.get("orders") or [])
            self._proposed_status_var.set(
                f"PROPOSED | account rev {order_result.get('account_revision')} | equity {format_trading_money(order_result.get('sizing_equity'))} | "
                f"預留 {format_trading_money(order_result.get('reserved_total'))} | 餘額 {format_trading_money(order_result.get('cash_after_reservation'))} | {order_result.get('text_path') or '-'}"
            )
        self.refresh_daily_workflow()
        self.refresh_order_state()
        if action == "data":
            self._workflow_status_var.set(
                f"資料更新完成：market {result.get('market_date') or '-'} | 成功 {result.get('count_success', 0)} | 已最新 {result.get('count_skipped_latest', 0)} | 下載失敗 {result.get('download_error_count', 0)}"
            )
        elif action == "rollforward":
            self._workflow_status_var.set(
                f"持股日終推進完成：持股 {result.get('processed_position_count', 0)} 檔 | completed bars {result.get('processed_bar_count', 0)} | account rev {result.get('account_revision', '-')}"
            )
            self.refresh_account()
            self.refresh_protection_plan()
            self.refresh_indicator_exit_plan()
        elif action == "params":
            self._workflow_status_var.set(
                f"Params 更新完成：through {result.get('latest_data_date') or '-'} | {result.get('selected_policy') or result.get('param_selector') or '-'}"
            )
        elif action == "orders":
            self._workflow_status_var.set(
                f"建議掛單完成：{len(result.get('orders') or [])} 筆 | 預留 {format_trading_money(result.get('reserved_total'))} | account rev {result.get('account_revision')}"
            )
        else:
            self._workflow_status_var.set(
                f"每日 Scanner 完成：候選 {len(scan_result.get('candidate_rows') or [])} 檔 | data {scan_result.get('latest_data_date') or '-'}"
            )
        self.refresh_operations_status()


    def _current_revision(self) -> int:
        revision = self._snapshot.get("revision")
        if revision is None:
            raise RuntimeError("Trading account 尚未初始化")
        return int(revision)

    def _run_mutation(self, action):
        try:
            action()
        except TradingAccountRevisionConflict as exc:
            messagebox.showerror("Trading 帳戶已更新", f"{exc}\n\n已重新讀取最新狀態，請確認後再操作。", parent=self)
            self.refresh_account()
            self.refresh_order_state()
            return False
        except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            self.refresh_account()
            self.refresh_order_state()
            return False
        self.refresh_account()
        self.refresh_order_state()
        self.refresh_protection_plan()
        self.refresh_indicator_exit_plan()
        self._reload_proposed_order_rows([])
        self._proposed_status_var.set("帳戶已變更；既有建議掛單已失效，請重新執行 4 建議掛單。")
        self.refresh_operations_status()
        return True

    def refresh_account(self):
        try:
            snapshot = build_trading_account_panel_snapshot(WORKBENCH_PROJECT_ROOT)
        except (ValueError, RuntimeError, OSError) as exc:
            self._snapshot = {}
            self._status_var.set(f"狀態讀取失敗：{exc}")
            return
        self._snapshot = snapshot
        initialized = bool(snapshot.get("initialized"))
        policy = dict(snapshot.get("policy") or {})
        dl_state = "OFF" if not policy.get("dl_filter_enabled") and not policy.get("dl_ranking_enabled") else "ON"
        self._policy_var.set(
            f"策略: {policy.get('strategy_id', '-')} | Params: {policy.get('param_selector', '-')} | DL: {dl_state}"
        )
        self._path_var.set(f"State: {snapshot.get('state_path', '-')}")
        if initialized:
            self._status_var.set(
                f"READY | revision {snapshot.get('revision')} | 現金 {format_trading_money(snapshot.get('cash'))} | 持股 {snapshot.get('position_count', 0)} | 更新 {snapshot.get('updated_at') or '-'}"
            )
        else:
            self._status_var.set("尚未初始化 Trading account")
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
        selected = self._selected_ticker()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._position_rows = {}
        source_labels = {"manual_adopted": "手動既有", "strategy_fill": "策略成交"}
        management_labels = {"unmanaged": "未接管", "active": "策略管理"}
        for row in list(self._snapshot.get("positions") or []):
            ticker = str(row.get("ticker") or "")
            self._position_rows[ticker] = dict(row)
            self._tree.insert(
                "",
                "end",
                iid=ticker,
                values=(
                    ticker,
                    source_labels.get(str(row.get("source")), str(row.get("source") or "-")),
                    f"{int(row.get('qty') or 0):,}",
                    format_trading_money(row.get("average_cost")),
                    format_trading_money(row.get("remaining_cost_basis")),
                    format_trading_money(row.get("realized_pnl")),
                    row.get("entry_date") or "-",
                    management_labels.get(str(row.get("management_status")), str(row.get("management_status") or "-")),
                ),
            )
        if selected and selected in self._position_rows:
            self._tree.selection_set(selected)
            self._tree.focus(selected)
        else:
            self._set_position_action_state(None)

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
        def action():
            cash = parse_trading_money_text(self._cash_var.get(), "帳戶現金", allow_blank=True, allow_zero=True)
            initialize_trading_account_state(WORKBENCH_PROJECT_ROOT, cash=cash)
        if self._run_mutation(action):
            messagebox.showinfo("Trading 帳戶", "Trading account 已初始化。", parent=self)

    def _set_cash(self):
        def action():
            cash = parse_trading_money_text(self._cash_var.get(), "帳戶現金", allow_blank=False, allow_zero=True)
            set_trading_cash_balance(
                WORKBENCH_PROJECT_ROOT,
                cash=cash,
                expected_revision=self._current_revision(),
                note="Workbench cash reconciliation",
            )
        if self._run_mutation(action):
            messagebox.showinfo("Trading 帳戶", "現金餘額已更新。", parent=self)

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
        def action():
            values = self._position_form_values()
            adopt_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                expected_revision=self._current_revision(),
                **values,
            )
        if self._run_mutation(action):
            self._clear_position_form()
            messagebox.showinfo("Trading 帳戶", "既有持股已登記；現金未變更。", parent=self)

    def _correct_position(self):
        selected = self._selected_ticker()
        if not selected:
            messagebox.showerror("Trading 帳戶操作失敗", "請先選取要修正的持股。", parent=self)
            return
        def action():
            values = self._position_form_values()
            if values["ticker"] != selected:
                raise ValueError("修正持股時不可變更股票代號；如代號輸入錯誤，請移除後重新新增。")
            correct_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                expected_revision=self._current_revision(),
                **values,
            )
        if self._run_mutation(action):
            messagebox.showinfo("Trading 帳戶", "既有持股 broker truth 已修正；現金未變更。", parent=self)

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
        def action():
            remove_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                ticker=selected,
                expected_revision=self._current_revision(),
                note=self._note_var.get().strip() or "Workbench manual position removal",
            )
        if self._run_mutation(action):
            self._clear_position_form()
            messagebox.showinfo("Trading 帳戶", "既有持股已移除；現金未變更。", parent=self)


__all__ = [
    "TradingAccountPanel",
    "build_trading_account_panel_snapshot",
    "format_trading_money",
    "parse_trading_money_text",
    "parse_trading_qty_text",
]
