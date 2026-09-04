from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from core.console_report import project_relative_display_path
from core.trading_policy import get_trading_policy_snapshot
from services.trading.daily_workflow import (
    build_trading_daily_workflow_snapshot,
    run_trading_candidate_scan,
    run_trading_daily_workflow,
    run_trading_market_data_update,
)
from services.trading.strategy_param_training import run_trading_strategy_param_training
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
        self._workflow_thread = None
        self._workflow_token = 0
        self._workflow_buttons = []
        self._build_ui()
        self.refresh_account()
        self.refresh_daily_workflow()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(5, weight=1)

        workflow_box = ttk.LabelFrame(self, text="每日 Trading 流程", padding=10, style=WORKBENCH_LABELLF_STYLE)
        workflow_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        workflow_box.columnconfigure(0, weight=1)
        self._workflow_status_var = tk.StringVar(value="讀取 Trading workflow 狀態...")
        self._workflow_freshness_var = tk.StringVar(value="-")
        ttk.Label(workflow_box, textvariable=self._workflow_status_var, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(workflow_box, textvariable=self._workflow_freshness_var, foreground=WORKBENCH_MUTED, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=0, sticky="w", pady=(4, 0))
        workflow_buttons = ttk.Frame(workflow_box, style=WORKBENCH_FRAME_STYLE)
        workflow_buttons.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))
        for text, action in (
            ("1 更新資料", "data"),
            ("2 更新 Params", "params"),
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
        ttk.Button(workflow_buttons, text="刷新狀態", command=self.refresh_daily_workflow, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(8, 0))
        ttk.Label(
            workflow_box,
            text="Scanner 只在 Trading Params 與目前 Trading data 同一最新交易日，且 selector 解析為單一 member 時執行。",
            foreground=WORKBENCH_MUTED,
            style=WORKBENCH_LABEL_STYLE,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        header = ttk.LabelFrame(self, text="Trading 帳戶", padding=10, style=WORKBENCH_LABELLF_STYLE)
        header.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        self._status_var = tk.StringVar(value="讀取中...")
        self._policy_var = tk.StringVar(value="-")
        self._path_var = tk.StringVar(value="-")
        ttk.Label(header, textvariable=self._status_var, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self._policy_var, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, textvariable=self._path_var, style=WORKBENCH_LABEL_STYLE).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Button(header, text="重新整理", command=self.refresh_account, style=WORKBENCH_BUTTON_STYLE).grid(row=0, column=1, rowspan=2, padx=(12, 0))

        cash_box = ttk.LabelFrame(self, text="現金", padding=10, style=WORKBENCH_LABELLF_STYLE)
        cash_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
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
        form.grid(row=3, column=0, sticky="ew", pady=(0, 8))
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
        table_box.grid(row=4, column=0, sticky="nsew", pady=(0, 8))
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

        candidate_box = ttk.LabelFrame(self, text="今日 Scanner 候選（尚未做帳戶 allocator / 下單）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        candidate_box.grid(row=5, column=0, sticky="nsew")
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

    def _start_workflow_action(self, action: str):
        if self._workflow_thread is not None and self._workflow_thread.is_alive():
            self._workflow_status_var.set("Trading workflow 執行中；請等待目前工作完成。")
            return
        labels = {"data": "更新 Trading 資料", "params": "更新 Trading Params", "scanner": "Scanner 候選", "all": "每日流程 1→2→3"}
        if action not in labels:
            messagebox.showerror("Trading workflow", f"未知 workflow action: {action}", parent=self)
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
            elif action == "params":
                result = run_trading_strategy_param_training(project_root=WORKBENCH_PROJECT_ROOT)
            elif action == "scanner":
                result = run_trading_candidate_scan(project_root=WORKBENCH_PROJECT_ROOT)
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
        self._set_workflow_buttons_state("normal")
        self.refresh_daily_workflow()
        self._workflow_status_var.set(f"FAIL：{type(exc).__name__}: {exc}")
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

    def _finish_workflow_success(self, action: str, token: int, result):
        if token != self._workflow_token:
            return
        self._workflow_thread = None
        self._set_workflow_buttons_state("normal")
        scan_result = dict(result.get("scanner") or {}) if action == "all" else (dict(result) if action == "scanner" else {})
        if scan_result:
            self._reload_candidate_rows(scan_result.get("candidate_rows") or [])
        self.refresh_daily_workflow()
        if action == "data":
            self._workflow_status_var.set(
                f"資料更新完成：market {result.get('market_date') or '-'} | 成功 {result.get('count_success', 0)} | 已最新 {result.get('count_skipped_latest', 0)} | 下載失敗 {result.get('download_error_count', 0)}"
            )
        elif action == "params":
            self._workflow_status_var.set(
                f"Params 更新完成：through {result.get('latest_data_date') or '-'} | {result.get('selected_policy') or result.get('param_selector') or '-'}"
            )
        else:
            self._workflow_status_var.set(
                f"每日 Scanner 完成：候選 {len(scan_result.get('candidate_rows') or [])} 檔 | data {scan_result.get('latest_data_date') or '-'}"
            )


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
            return False
        except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            self.refresh_account()
            return False
        self.refresh_account()
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
