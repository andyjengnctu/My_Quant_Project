from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from services.trading.account_dashboard import (
    build_trading_account_dashboard_read_model,
    publish_trading_account_dashboard_snapshot,
)
from services.trading.account_state import (
    adopt_existing_trading_position,
    correct_existing_trading_position,
    delete_trading_transaction,
    get_trading_account_read_model,
    initialize_trading_account_state,
    remove_existing_trading_position,
    set_trading_cash_balance,
)
from services.trading.account_trade_entry import correct_trading_account_transaction, record_trading_account_inventory_sell
from services.trading.order_form_constraints import build_trading_actual_fill_form_constraints
from services.trading.scanner_state import load_trading_scanner_runtime
from services.workbench_ui.date_picker import DatePickerField
from services.workbench_ui.paged_table import PagedTable, TableColumn
from services.workbench_ui.state_sync import ACCOUNT_MUTATION_DOMAINS
from services.workbench_ui.trading_source_labels import trading_source_display_label
from services.workbench_ui.workbench import (
    WORKBENCH_BG,
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_COMBO_STYLE,
    WORKBENCH_ENTRY_STYLE,
    WORKBENCH_ERROR,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_INFO,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_SUCCESS,
    WORKBENCH_TEXT,
    WORKBENCH_VSCROLL_STYLE,
    _warn_gui_fallback,
)

WORKBENCH_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _amount(value):
    """Display TWD account amounts without decimal places."""
    if value is None:
        return "-"
    return f"{float(value):,.0f}"


def _price(value):
    if value is None:
        return "-"
    return f"{float(value):,.2f}"


def _money(value):
    # Backward-compatible alias for amount-like fields in this panel.
    return _amount(value)


def _pct(value):
    if value is None:
        return "-"
    return f"{float(value):.2f}%"


def _integer(value):
    if value is None:
        return "-"
    return f"{int(value):,}"


def _parse_positive_int(text: str, label: str) -> int:
    raw = str(text or "").replace(",", "").strip()
    if not raw:
        raise ValueError(f"{label}必填")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"{label}格式不正確") from exc
    if value != value.to_integral_value() or value <= 0:
        raise ValueError(f"{label}必須是正整數")
    return int(value)


def _parse_positive_money(text: str, label: str, *, allow_zero: bool = False) -> float:
    raw = str(text or "").replace(",", "").strip()
    if not raw:
        raise ValueError(f"{label}必填")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"{label}格式不正確") from exc
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{label}必須{'大於等於' if allow_zero else '大於'} 0")
    return float(value)


class AccountingCenterPanel(ttk.Frame):
    """Broker-truth inventory, actual fills and performance accounting."""

    def __init__(self, master):
        super().__init__(master, padding=10, style=WORKBENCH_FRAME_STYLE)
        self._snapshot: dict = {}
        self._dashboard: dict = {}
        self._refresh_results: queue.Queue = queue.Queue()
        self._refresh_thread = None
        self._refresh_pending = False
        self._all_buy_rows: list[dict] = []
        self._all_sell_rows: list[dict] = []
        self._footer_hint_var = tk.StringVar(value="")
        self._footer_hint_after_id = None
        self._sell_selected_ticker = ""
        self._sell_fill_constraints: dict[str, object] = {}
        self._sell_price_option_set: frozenset[str] = frozenset()
        self._sell_constraint_after_id = None
        self._sell_constraint_programmatic_update = False
        self._sell_constraint_token = 0
        self._sell_constraint_inflight = 0
        self._sell_constraint_results: queue.Queue = queue.Queue()
        self._sell_constraint_poll_after_id = None
        self._build_ui()
        self.after(80, self.refresh)

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self._canvas = tk.Canvas(self, background=WORKBENCH_BG, highlightthickness=0, borderwidth=0)
        self._page_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview, style=WORKBENCH_VSCROLL_STYLE)
        self._canvas.configure(yscrollcommand=self._page_scrollbar.set, yscrollincrement=36)
        self._canvas.grid(row=1, column=0, sticky="nsew")
        self._page_scrollbar.grid(row=1, column=1, sticky="ns")
        content = ttk.Frame(self._canvas, style=WORKBENCH_FRAME_STYLE)
        self._window = self._canvas.create_window((0, 0), window=content, anchor="nw")
        content.columnconfigure(0, weight=1)
        content.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")), add="+")
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._window, width=max(1, int(e.width))), add="+")

        inventory_hint = "左側 ▣ 可開啟單股回測檢視；選取庫存後可登錄實際賣出成交。"
        buy_hint = "左側 ▣ 可開啟單股回測檢視；買入明細可修改或刪除；原 event 保留，帳務由有效歷史重新計算。"
        sell_hint = "左側 ▣ 可開啟單股回測檢視；賣出明細可修改或刪除；沖抵成本、損益與現金會依有效歷史重算。"
        offset_hint = "沖抵明細跟隨上方選取的賣出紀錄。"

        dashboard = ttk.LabelFrame(self, text="帳戶儀表板", padding=10, style=WORKBENCH_LABELLF_STYLE)
        dashboard.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        dashboard_header = ttk.Frame(dashboard, style=WORKBENCH_FRAME_STYLE)
        dashboard_header.pack(fill="x", pady=(0, 6))
        self._refresh_status_var = tk.StringVar(value="")
        ttk.Label(dashboard_header, textvariable=self._refresh_status_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="left", fill="x", expand=True)
        self._status_var = tk.StringVar(value="-")
        self._market_date_var = tk.StringVar(value="-")

        grid = ttk.Frame(dashboard, style=WORKBENCH_FRAME_STYLE)
        grid.pack(fill="x")
        self._metric_vars = {key: tk.StringVar(value="-") for key in ("liquidation", "cash", "equity")}
        self._metric_labels = {}
        cards = (
            ("revision", "revision"),
            ("市價日", "market_date"),
            ("股票淨值", "liquidation"),
            ("現金餘額", "cash"),
            ("帳戶淨值", "equity"),
        )
        for col, (label, key) in enumerate(cards):
            box = ttk.LabelFrame(
                grid,
                text=label,
                padding=(8, 5),
                style=WORKBENCH_LABELLF_STYLE,
                labelanchor="n",
            )
            box.grid(row=0, column=col, padx=(0 if col == 0 else 6, 0), sticky="nsew")
            variable = self._status_var if key == "revision" else self._market_date_var if key == "market_date" else self._metric_vars[key]
            metric_label = ttk.Label(
                box,
                textvariable=variable,
                style=WORKBENCH_LABEL_STYLE,
                foreground=WORKBENCH_INFO if key == "market_date" else WORKBENCH_TEXT,
                anchor="center",
                justify="center",
            )
            metric_label.pack(fill="x")
            self._metric_labels[key] = metric_label
            grid.columnconfigure(col, weight=1)

        cash_box = ttk.LabelFrame(content, text="帳戶現金", padding=10, style=WORKBENCH_LABELLF_STYLE)
        cash_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(cash_box, text="現金餘額", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._cash_var = tk.StringVar()
        ttk.Entry(cash_box, textvariable=self._cash_var, width=22, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(8, 8))
        self._init_button = ttk.Button(cash_box, text="初始化帳戶", command=self._initialize, style=WORKBENCH_BUTTON_STYLE)
        self._init_button.pack(side="left")
        self._cash_button = ttk.Button(cash_box, text="更新現金", command=self._set_cash, style=WORKBENCH_BUTTON_STYLE)
        self._cash_button.pack(side="left", padx=(8, 0))

        inventory = ttk.LabelFrame(content, text="庫存股", padding=8, style=WORKBENCH_LABELLF_STYLE)
        inventory.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        inventory.columnconfigure(0, weight=1)
        self._inventory = PagedTable(
            inventory,
            columns=(
                TableColumn("source", "來源", 9, formatter=lambda v, _r: trading_source_display_label(source=v)),
                TableColumn("ticker", "股票", 8),
                TableColumn("average_cost", "均價", 10, sort_kind="numeric", formatter=lambda v, _r: _price(v)),
                TableColumn("qty", "股數", 9, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("holding_cost", "持有成本", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("current_price", "市價", 10, sort_kind="numeric", formatter=lambda v, _r: _price(v)),
                TableColumn("market_value", "市值", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("unrealized_pnl", "損益", 11, sort_kind="numeric", performance=True, formatter=lambda v, _r: _money(v)),
                TableColumn("holding_return_pct", "報酬率", 9, sort_kind="numeric", performance=True, formatter=lambda v, _r: _pct(v)),
            ),
            page_size=12,
            default_sort_key="ticker",
            empty_text="目前沒有庫存股",
            on_select=self._on_inventory_selected,
            on_open_stock=self._open_stock,
            on_mousewheel=self._on_mousewheel,
            on_pointer_enter=lambda _event: self._show_footer_hint(inventory_hint),
            on_pointer_leave=lambda _event: self._schedule_footer_hint_clear(),
        )
        self._inventory.grid(row=0, column=0, sticky="ew")

        inv_actions = ttk.Frame(inventory, style=WORKBENCH_FRAME_STYLE)
        inv_actions.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(inv_actions, text="新增既有庫存", command=self._add_inventory_dialog, style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        self._edit_inventory_button = ttk.Button(inv_actions, text="修改庫存", command=self._edit_inventory_dialog, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_inventory_button.pack(side="left", padx=(6, 0))
        self._delete_inventory_button = ttk.Button(inv_actions, text="刪除庫存", command=self._delete_inventory, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_inventory_button.pack(side="left", padx=(6, 0))
        inv_actions.grid_remove()

        sell_entry = ttk.LabelFrame(inventory, text="賣出成交登錄", padding=8, style=WORKBENCH_LABELLF_STYLE)
        sell_entry.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._sell_ticker_var = tk.StringVar(value="-")
        self._sell_qty_var = tk.StringVar()
        self._sell_price_var = tk.StringVar()
        self._sell_date_var = tk.StringVar()
        ttk.Label(sell_entry, text="股票", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(sell_entry, text="成交日", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(sell_entry, text="成交價", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(sell_entry, text="數量", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=3, sticky="w", padx=(8, 0))
        ttk.Label(sell_entry, textvariable=self._sell_ticker_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_TEXT).grid(row=1, column=0, sticky="ew")
        self._sell_date_field = DatePickerField(
            sell_entry, textvariable=self._sell_date_var, width=12, allowed_dates=()
        )
        self._sell_date_field.grid(row=1, column=1, sticky="ew", padx=(8, 0))
        self._sell_price_combo = ttk.Combobox(
            sell_entry,
            textvariable=self._sell_price_var,
            values=(),
            width=12,
            state="disabled",
            style=WORKBENCH_COMBO_STYLE,
        )
        self._sell_price_combo.grid(row=1, column=2, sticky="ew", padx=(8, 0))
        ttk.Entry(sell_entry, textvariable=self._sell_qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=3, sticky="ew", padx=(8, 0))
        self._sell_button = ttk.Button(sell_entry, text="登錄賣出成交", command=self._record_inventory_sell, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._sell_button.grid(row=1, column=4, sticky="e", padx=(8, 0))
        self._sell_date_var.trace_add("write", self._schedule_sell_fill_constraints)
        self._sell_price_var.trace_add("write", self._refresh_sell_button_state)
        self._sell_qty_var.trace_add("write", self._refresh_sell_button_state)

        buy_box = ttk.LabelFrame(content, text="買入明細", padding=8, style=WORKBENCH_LABELLF_STYLE)
        buy_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        buy_box.columnconfigure(0, weight=1)
        self._buys = PagedTable(
            buy_box,
            columns=(
                TableColumn("source", "來源", 9, formatter=lambda v, _r: trading_source_display_label(source=v)),
                TableColumn("ticker", "股票", 8),
                TableColumn("trade_date", "成交日", 11, sort_kind="date"),
                TableColumn("price", "成交價", 10, sort_kind="numeric", formatter=lambda v, _r: _price(v)),
                TableColumn("qty", "數量", 9, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("gross_amount", "價金", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("buy_fee", "買入手續費", 11, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("holding_cost", "持有成本", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
            ),
            page_size=12,
            default_sort_key="trade_date",
            default_desc=False,
            empty_text="目前沒有買入明細",
            on_select=self._on_buy_selected,
            on_open_stock=self._open_stock,
            on_mousewheel=self._on_mousewheel,
            on_pointer_enter=lambda _event: self._show_footer_hint(buy_hint),
            on_pointer_leave=lambda _event: self._schedule_footer_hint_clear(),
        )
        self._buys.grid(row=0, column=0, sticky="ew")
        buy_actions = ttk.Frame(buy_box, style=WORKBENCH_FRAME_STYLE)
        buy_actions.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._edit_buy_button = ttk.Button(buy_actions, text="修改選取買入", command=lambda: self._edit_selected_transaction("BUY"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_buy_button.pack(side="left")
        self._delete_buy_button = ttk.Button(buy_actions, text="刪除選取買入", command=lambda: self._delete_selected_transaction("BUY"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_buy_button.pack(side="left", padx=(6, 0))

        sell_box = ttk.LabelFrame(content, text="賣出明細", padding=8, style=WORKBENCH_LABELLF_STYLE)
        sell_box.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        sell_box.columnconfigure(0, weight=1)
        self._sells = PagedTable(
            sell_box,
            columns=(
                TableColumn("source", "來源", 9, formatter=lambda v, _r: trading_source_display_label(source=v)),
                TableColumn("ticker", "股票", 8),
                TableColumn("trade_date", "日期", 11, sort_kind="date"),
                TableColumn("qty", "股數", 9, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("price", "成交價", 10, sort_kind="numeric", formatter=lambda v, _r: _price(v)),
                TableColumn("gross_amount", "價金", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("sell_fee", "賣出手續費", 11, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("tax", "交易稅", 10, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("offset_holding_cost", "沖抵持有成本", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("pnl", "損益", 11, sort_kind="numeric", performance=True, formatter=lambda v, _r: _money(v)),
                TableColumn("return_pct", "報酬率", 9, sort_kind="numeric", performance=True, formatter=lambda v, _r: _pct(v)),
            ),
            page_size=12,
            default_sort_key="trade_date",
            default_desc=False,
            empty_text="目前沒有賣出明細",
            on_select=self._on_sell_selected,
            on_open_stock=self._open_stock,
            on_mousewheel=self._on_mousewheel,
            on_pointer_enter=lambda _event: self._show_footer_hint(sell_hint),
            on_pointer_leave=lambda _event: self._schedule_footer_hint_clear(),
        )
        self._sells.grid(row=0, column=0, sticky="ew")
        sell_actions = ttk.Frame(sell_box, style=WORKBENCH_FRAME_STYLE)
        sell_actions.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._edit_sell_button = ttk.Button(sell_actions, text="修改選取賣出", command=lambda: self._edit_selected_transaction("SELL"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_sell_button.pack(side="left")
        self._delete_sell_button = ttk.Button(sell_actions, text="刪除選取賣出", command=lambda: self._delete_selected_transaction("SELL"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_sell_button.pack(side="left", padx=(6, 0))

        offset_box = ttk.LabelFrame(content, text="沖抵明細", padding=8, style=WORKBENCH_LABELLF_STYLE)
        offset_box.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        offset_box.columnconfigure(0, weight=1)
        self._offset = PagedTable(
            offset_box,
            columns=(
                TableColumn("source", "來源", 9, formatter=lambda v, _r: trading_source_display_label(source=v)),
                TableColumn("ticker", "股票", 8),
                TableColumn("trade_date", "賣出日", 11, sort_kind="date"),
                TableColumn("qty", "賣出股數", 9, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("offset_gross_amount", "沖抵買入價金", 13, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("offset_buy_fee", "沖抵買入手續費", 13, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("offset_holding_cost", "沖抵持有成本", 13, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("remaining_qty", "賣出後剩餘股數", 12, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
            ),
            page_size=12,
            default_sort_key="trade_date",
            empty_text="選取賣出紀錄後顯示沖抵明細",
            on_open_stock=self._open_stock,
            on_mousewheel=self._on_mousewheel,
            on_pointer_enter=lambda _event: self._show_footer_hint(offset_hint),
            on_pointer_leave=lambda _event: self._schedule_footer_hint_clear(),
        )
        self._offset.grid(row=0, column=0, sticky="ew")

        perf = ttk.LabelFrame(content, text="績效統計", padding=8, style=WORKBENCH_LABELLF_STYLE)
        perf.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        perf.columnconfigure(0, weight=1)
        self._perf = PagedTable(
            perf,
            columns=(
                TableColumn("scope", "範圍", 9),
                TableColumn("stock_count", "股票檔數", 9, formatter=lambda v, _r: _integer(v)),
                TableColumn("value", "淨值", 12, formatter=lambda v, _r: _amount(v)),
                TableColumn("cost", "成本", 12, formatter=lambda v, _r: _amount(v)),
                TableColumn("pnl", "損益", 12, performance=True, formatter=lambda v, _r: _amount(v)),
                TableColumn("return_pct", "報酬率", 10, performance=True, formatter=lambda v, _r: _pct(v)),
                TableColumn("win_rate_pct", "勝率", 9, formatter=lambda v, _r: _pct(v)),
                TableColumn("expected_value_r", "實績 EV(R)", 11, performance=True, formatter=lambda v, _r: "N/A" if v is None else f"{float(v):.2f} R"),
                TableColumn("risk_reward_ratio", "實績賺賠比", 10, formatter=lambda v, _r: "N/A" if v is None else f"{float(v):.2f}"),
            ),
            page_size=12,
            stock_key=None,
            default_sort_key=None,
            sortable=False,
            empty_text="目前沒有可統計績效",
            on_mousewheel=self._on_mousewheel,
        )
        self._perf.grid(row=0, column=0, sticky="ew")

        # Fixed bottom status line: hints never consume scrollable page space.
        self._footer_bar = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        self._footer_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Separator(self._footer_bar, orient="horizontal").pack(fill="x", pady=(0, 3))
        footer_line = ttk.Frame(self._footer_bar, style=WORKBENCH_FRAME_STYLE)
        footer_line.pack(fill="x")
        ttk.Label(footer_line, text="操作提示｜", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="left")
        ttk.Button(footer_line, text="全狀態刷新", command=self.refresh, style=WORKBENCH_BUTTON_STYLE).pack(side="right", padx=(8, 0))
        ttk.Label(footer_line, textvariable=self._footer_hint_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="left", fill="x", expand=True)
        self._bind_footer_hint(cash_box, "現金欄只用於券商現金對帳；買賣成交會自動更新現金。")
        self._bind_footer_hint(inventory, inventory_hint)
        self._bind_footer_hint(sell_entry, "先在券商完成賣出，再登錄實際股數、成交價與成交日。")
        self._bind_footer_hint(buy_box, buy_hint)
        self._bind_footer_hint(sell_box, sell_hint)
        self._bind_footer_hint(offset_box, offset_hint)
        self._bind_page_mousewheel(dashboard)
        # Bind the whole panel, canvas, fixed footer and all existing descendants.
        # PagedTable separately binds cells created later during refresh.
        self._bind_page_mousewheel(self)

    def _on_mousewheel(self, event):
        # Tk reports ``event.num`` as the literal string "??" for native
        # <MouseWheel> events on some Windows/Tk builds.  Do not coerce it
        # to int; Button-4/5 remain numeric while MouseWheel uses delta.
        number = getattr(event, "num", None)
        if number == 4:
            units = -1
        elif number == 5:
            units = 1
        else:
            delta = getattr(event, "delta", 0)
            units = -1 * int(delta / 120) if isinstance(delta, (int, float)) and delta else 0
        if not units:
            return None
        self._canvas.yview_scroll(units, "units")
        return "break"

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
                target.bind("<MouseWheel>", self._on_mousewheel, add="+")
                target.bind("<Button-4>", self._on_mousewheel, add="+")
                target.bind("<Button-5>", self._on_mousewheel, add="+")
            except tk.TclError as exc:
                _warn_gui_fallback("AccountingCenterPanel.bind_page_mousewheel", exc)
                continue

    def _show_footer_hint(self, text: str) -> None:
        if self._footer_hint_after_id is not None:
            try:
                self.after_cancel(self._footer_hint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("AccountingCenterPanel.after_cancel(footer_hint)", exc)
            self._footer_hint_after_id = None
        self._footer_hint_var.set(str(text or ""))

    def _schedule_footer_hint_clear(self) -> None:
        if self._footer_hint_after_id is not None:
            try:
                self.after_cancel(self._footer_hint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("AccountingCenterPanel.after_cancel(scheduled_footer_hint)", exc)
        self._footer_hint_after_id = self.after(80, self._clear_footer_hint)

    def _clear_footer_hint(self) -> None:
        self._footer_hint_after_id = None
        self._footer_hint_var.set("")

    def _bind_footer_hint(self, widget, text: str) -> None:
        targets = [widget]
        index = 0
        while index < len(targets):
            target = targets[index]
            index += 1
            try:
                targets.extend(child for child in target.winfo_children() if child not in targets)
                target.bind("<Enter>", lambda _event, value=text: self._show_footer_hint(value), add="+")
                target.bind("<Leave>", lambda _event: self._schedule_footer_hint_clear(), add="+")
            except tk.TclError as exc:
                _warn_gui_fallback("AccountingCenterPanel.bind_footer_hint", exc)
                continue

    def refresh(self):
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            self._refresh_pending = True
            self._refresh_status_var.set("刷新排程中…")
            return
        self._refresh_pending = False
        self._refresh_status_var.set("刷新中…")
        self._refresh_thread = threading.Thread(target=self._load_worker, daemon=True, name="workbench-accounting-center-refresh")
        self._refresh_thread.start()
        self.after(30, self._poll_refresh)

    def _load_worker(self):
        try:
            account = get_trading_account_read_model(WORKBENCH_PROJECT_ROOT)
            dashboard = build_trading_account_dashboard_read_model(WORKBENCH_PROJECT_ROOT)
            publish_trading_account_dashboard_snapshot(WORKBENCH_PROJECT_ROOT, dashboard)
            self._refresh_results.put((account, dashboard, None))
        except Exception as exc:
            self._refresh_results.put((None, None, exc))

    def _poll_refresh(self):
        try:
            account, dashboard, error = self._refresh_results.get_nowait()
        except queue.Empty:
            if self._refresh_thread is not None and self._refresh_thread.is_alive():
                self.after(30, self._poll_refresh)
            return
        self._refresh_thread = None
        if error is not None:
            self._snapshot = {}
            self._dashboard = {}
            self._status_var.set("-")
            self._market_date_var.set("-")
            self._init_button.configure(state="normal")
            self._cash_button.configure(state="disabled")
            self._refresh_status_var.set(f"刷新失敗：{type(error).__name__}: {error}")
            if self._refresh_pending:
                self.after_idle(self.refresh)
            return
        self._snapshot = dict(account or {})
        self._dashboard = dict(dashboard or {})
        self._render()
        self._refresh_status_var.set("刷新完成")
        if self._refresh_pending:
            self.after_idle(self.refresh)

    def _render(self):
        summary = dict(self._dashboard.get("summary") or {})
        monetary_metrics = {
            "liquidation": summary.get("holdings_net_liquidation"),
            "cash": summary.get("cash"),
            "equity": summary.get("equity"),
        }
        for key, value in monetary_metrics.items():
            self._metric_vars[key].set(_amount(value))
            if value is None or float(value) == 0.0:
                color = WORKBENCH_TEXT
            elif float(value) > 0.0:
                color = WORKBENCH_ERROR
            else:
                color = WORKBENCH_SUCCESS
            self._metric_labels[key].configure(foreground=color)
        self._cash_var.set(_amount(self._snapshot.get("cash")) if self._snapshot.get("cash") is not None else "")
        self._init_button.configure(state="disabled")
        self._cash_button.configure(state="normal")
        self._status_var.set(str(int(self._snapshot.get("revision") or 0)))
        self._market_date_var.set(str(self._dashboard.get("market_date") or "-"))

        account_positions = {str(row.get("ticker") or ""): dict(row) for row in self._snapshot.get("positions") or []}
        inventory_rows = []
        for row in self._dashboard.get("positions") or []:
            item = dict(row)
            ticker = str(item.get("ticker") or "")
            account_row = account_positions.get(ticker, {})
            item["source"] = account_row.get("source", item.get("source"))
            item["management_status"] = account_row.get("management_status", item.get("management_status"))
            item["entry_date"] = account_row.get("entry_date", item.get("entry_date"))
            item["_table_id"] = ticker
            inventory_rows.append(item)
        self._inventory.set_rows(inventory_rows, preserve_selection=True)

        self._all_buy_rows = []
        for row in self._dashboard.get("buy_details") or []:
            item = dict(row)
            item["_table_id"] = f"buy-{int(item.get('revision') or 0)}"
            self._all_buy_rows.append(item)
        self._all_sell_rows = []
        for row in self._dashboard.get("sell_details") or []:
            item = dict(row)
            item["_table_id"] = f"sell-{int(item.get('revision') or 0)}"
            self._all_sell_rows.append(item)
        self._sells.set_rows(self._all_sell_rows, preserve_selection=True)
        self._perf.set_rows([
            {**dict(row), "_table_id": f"perf-{idx}"}
            for idx, row in enumerate(self._dashboard.get("performance") or [])
        ], preserve_selection=False)
        self._apply_inventory_filter()
        self._on_buy_selected(self._buys.selected_row())
        self._on_sell_selected(self._sells.selected_row())
        self._on_inventory_selected(self._inventory.selected_row(), rerender_buys=False)

    def _apply_inventory_filter(self):
        row = self._inventory.selected_row()
        if row:
            ticker = str(row.get("ticker") or "")
            rows = [
                item for item in self._all_buy_rows
                if str(item.get("ticker") or "") == ticker and bool(item.get("open_position_related"))
            ]
        else:
            rows = list(self._all_buy_rows)
        self._buys.set_rows(rows, preserve_selection=False)

    def refresh_for_state_domains(self, _domains) -> None:
        self.refresh()

    def _notify_workbench_state_changed(self) -> bool:
        callback = getattr(self.winfo_toplevel(), "_notify_workbench_state_changed", None)
        if callable(callback):
            callback(ACCOUNT_MUTATION_DOMAINS, source_panel_id="accounting_center")
            return True
        legacy = getattr(self.winfo_toplevel(), "_notify_trading_account_changed", None)
        if callable(legacy):
            legacy()
        return False

    def _refresh_after_account_mutation(self) -> None:
        # The source view rereads canonical truth itself; the shell only fans the same
        # committed mutation out to other mounted dependent views.
        self.refresh()
        self._notify_workbench_state_changed()

    def _parse_cash(self):
        text = self._cash_var.get().replace(",", "").strip()
        if not text:
            return None
        value = float(text)
        if value < 0:
            raise ValueError("現金不可為負數")
        return value

    def _initialize(self):
        try:
            initialize_trading_account_state(WORKBENCH_PROJECT_ROOT, cash=self._parse_cash())
        except Exception as exc:
            messagebox.showerror("帳務中心", str(exc), parent=self)
            return
        self._refresh_after_account_mutation()

    def _set_cash(self):
        try:
            cash = self._parse_cash()
            if cash is None:
                raise ValueError("現金必填")
            set_trading_cash_balance(
                WORKBENCH_PROJECT_ROOT,
                cash=cash,
                expected_revision=None,
                note="Workbench accounting center cash reconciliation",
            )
        except Exception as exc:
            messagebox.showerror("帳務中心", str(exc), parent=self)
            return
        self._refresh_after_account_mutation()

    def _selected_inventory(self):
        return self._inventory.selected_row()

    def _on_inventory_selected(self, row, *, rerender_buys=True):
        active = bool(row)
        self._edit_inventory_button.configure(state="normal" if active else "disabled")
        self._delete_inventory_button.configure(state="normal" if active else "disabled")
        ticker = str(row.get("ticker") or "").strip().upper() if row else ""
        self._sell_ticker_var.set(ticker or "-")
        if not row:
            self._sell_selected_ticker = ""
            self._sell_constraint_programmatic_update = True
            try:
                self._sell_qty_var.set("")
                self._sell_price_var.set("")
                self._sell_date_var.set("")
            finally:
                self._sell_constraint_programmatic_update = False
            self._clear_sell_fill_constraints()
        elif ticker != self._sell_selected_ticker:
            self._sell_selected_ticker = ticker
            self._sell_constraint_programmatic_update = True
            try:
                self._sell_qty_var.set(str(int(row.get("qty") or 0)))
                self._sell_price_var.set("")
                self._sell_date_var.set("")
            finally:
                self._sell_constraint_programmatic_update = False
            self._clear_sell_fill_constraints()
            self._schedule_sell_fill_constraints()
        self._refresh_sell_button_state()
        if rerender_buys:
            self._apply_inventory_filter()

    @staticmethod
    def _normalize_price_option_text(value) -> str:
        try:
            return f"{float(value):.3f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(value or "").strip()

    def _clear_sell_fill_constraints(self, *, invalidate: bool = True) -> None:
        if invalidate:
            self._sell_constraint_token += 1
            if self._sell_constraint_after_id is not None:
                try:
                    self.after_cancel(self._sell_constraint_after_id)
                except tk.TclError as exc:
                    _warn_gui_fallback("Accounting sell-fill constraint clear", exc)
                self._sell_constraint_after_id = None
        self._sell_fill_constraints = {}
        self._sell_price_option_set = frozenset()
        if hasattr(self, "_sell_price_combo"):
            self._sell_price_combo.configure(values=(), state="disabled")
        if hasattr(self, "_sell_date_field"):
            self._sell_date_field.set_allowed_dates(())

    def _schedule_sell_fill_constraints(self, *_args):
        if self._sell_constraint_programmatic_update:
            return None
        self._sell_constraint_token += 1
        if self._sell_constraint_after_id is not None:
            try:
                self.after_cancel(self._sell_constraint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Accounting sell-fill constraint debounce", exc)
        # A date change invalidates the previous day's price universe immediately.
        # Clear the actual-fill price before the async constraint refresh so a
        # stale tick can never remain submittable while the new date is loading.
        self._sell_constraint_programmatic_update = True
        try:
            self._sell_price_var.set("")
        finally:
            self._sell_constraint_programmatic_update = False
        self._sell_price_option_set = frozenset()
        self._sell_price_combo.configure(values=(), state="disabled")
        self._sell_constraint_after_id = self.after(120, self._start_sell_fill_constraints)
        self._refresh_sell_button_state()
        return None

    @staticmethod
    def _sell_fill_constraint_worker(
        token: int,
        ticker: str,
        selected_date: str | None,
        earliest_exclusive_date: str | None,
    ):
        result = None
        error = None
        try:
            runtime = load_trading_scanner_runtime(WORKBENCH_PROJECT_ROOT)
            latest = str(runtime.get("latest_data_date") or "").strip()
            result = build_trading_actual_fill_form_constraints(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                selected_date=selected_date or None,
                latest_finalized_date=latest or None,
                earliest_exclusive_date=earliest_exclusive_date or None,
            )
        except Exception as exc:
            error = exc
        return token, result, error

    def _start_sell_fill_constraints(self):
        self._sell_constraint_after_id = None
        row = self._selected_inventory()
        ticker = str((row or {}).get("ticker") or "").strip().upper()
        if not ticker:
            self._clear_sell_fill_constraints()
            self._refresh_sell_button_state()
            return
        selected_date = self._sell_date_var.get().strip() or None
        earliest_exclusive_date = str((row or {}).get("entry_date") or "").strip() or None
        self._sell_constraint_token += 1
        token = int(self._sell_constraint_token)
        self._sell_constraint_inflight += 1

        def worker():
            self._sell_constraint_results.put(
                self._sell_fill_constraint_worker(
                    token,
                    ticker,
                    selected_date,
                    earliest_exclusive_date,
                )
            )

        threading.Thread(
            target=worker,
            name=f"workbench-account-sell-fill-constraint-{token}",
            daemon=True,
        ).start()
        if self._sell_constraint_poll_after_id is None:
            self._sell_constraint_poll_after_id = self.after(30, self._drain_sell_fill_constraints)

    def _drain_sell_fill_constraints(self):
        self._sell_constraint_poll_after_id = None
        while True:
            try:
                token, result, error = self._sell_constraint_results.get_nowait()
            except queue.Empty:
                break
            self._sell_constraint_inflight = max(0, int(self._sell_constraint_inflight) - 1)
            if int(token) != int(self._sell_constraint_token):
                continue
            if error is not None:
                self._clear_sell_fill_constraints(invalidate=False)
                self._refresh_sell_button_state()
                continue
            payload = dict(result or {})
            self._sell_fill_constraints = payload
            allowed_dates = tuple(payload.get("allowed_dates") or ())
            prices = tuple(str(value) for value in (payload.get("price_options") or ()))
            self._sell_price_option_set = frozenset(
                self._normalize_price_option_text(value) for value in prices
            )
            self._sell_date_field.set_allowed_dates(allowed_dates)
            self._sell_price_combo.configure(
                values=prices,
                state="readonly" if prices else "disabled",
            )
            current_price = self._normalize_price_option_text(self._sell_price_var.get())
            if current_price and current_price not in self._sell_price_option_set:
                self._sell_constraint_programmatic_update = True
                try:
                    self._sell_price_var.set("")
                finally:
                    self._sell_constraint_programmatic_update = False
            if not self._sell_date_var.get().strip() and payload.get("preferred_fill_date"):
                # selected_date=None already returns the preferred day's price
                # options from the canonical constraint builder. Set the date
                # without launching a duplicate query for the same payload.
                self._sell_constraint_programmatic_update = True
                try:
                    self._sell_date_var.set(str(payload.get("preferred_fill_date")))
                finally:
                    self._sell_constraint_programmatic_update = False
            self._refresh_sell_button_state()
        if self._sell_constraint_inflight > 0:
            self._sell_constraint_poll_after_id = self.after(30, self._drain_sell_fill_constraints)

    def _refresh_sell_button_state(self, *_args) -> None:
        if not hasattr(self, "_sell_button"):
            return
        row = self._selected_inventory()
        if not row:
            self._sell_button.configure(state="disabled")
            return
        fill_date = self._sell_date_var.get().strip()
        allowed_dates = set(self._sell_fill_constraints.get("allowed_dates") or ())
        valid_date = bool(fill_date and fill_date in allowed_dates)
        price_text = self._normalize_price_option_text(self._sell_price_var.get())
        valid_price = bool(price_text and price_text in self._sell_price_option_set)
        try:
            qty = _parse_positive_int(self._sell_qty_var.get(), "數量")
            valid_qty = qty <= int(row.get("qty") or 0)
        except ValueError:
            valid_qty = False
        self._sell_button.configure(
            state="normal" if valid_date and valid_price and valid_qty else "disabled"
        )

    def _record_inventory_sell(self):
        row = self._selected_inventory()
        if not row:
            messagebox.showerror("賣出登錄", "請先選取庫存股", parent=self)
            return
        try:
            ticker = str(row.get("ticker") or "")
            qty = _parse_positive_int(self._sell_qty_var.get(), "數量")
            price = _parse_positive_money(self._sell_price_var.get(), "成交價")
            trade_date = self._sell_date_var.get().strip()
            if not trade_date:
                raise ValueError("成交日必填")
            allowed_dates = set(self._sell_fill_constraints.get("allowed_dates") or ())
            if trade_date not in allowed_dates:
                raise ValueError("成交日不是目前最新 finalized 範圍內、且晚於買入日的可選日期")
            normalized_price = self._normalize_price_option_text(price)
            if normalized_price not in self._sell_price_option_set:
                raise ValueError("成交價不在所選成交日的合法市場價格 ticks 內")
        except Exception as exc:
            messagebox.showerror("賣出登錄", str(exc), parent=self)
            return
        if not messagebox.askyesno(
            "確認賣出成交",
            f"{ticker}｜{qty:,} 股 @ {price:,.2f}｜{trade_date}\n\n費用、交易稅、沖抵成本與損益由帳務 SSOT 自動計算。",
            parent=self,
        ):
            return
        try:
            record_trading_account_inventory_sell(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                qty=qty,
                price=price,
                trade_date=trade_date,
                expected_account_revision=None,
            )
        except Exception as exc:
            messagebox.showerror("賣出登錄失敗", str(exc), parent=self)
            return
        self._sell_selected_ticker = ""
        self._sell_constraint_programmatic_update = True
        try:
            self._sell_qty_var.set("")
            self._sell_price_var.set("")
            self._sell_date_var.set("")
        finally:
            self._sell_constraint_programmatic_update = False
        self._clear_sell_fill_constraints()
        messagebox.showinfo("賣出登錄", f"{ticker} 實際賣出成交已寫入帳戶 SSOT。", parent=self)
        self._refresh_after_account_mutation()

    def _inventory_dialog(self, *, mode: str, row=None):
        top = tk.Toplevel(self)
        top.title("新增既有庫存" if mode == "add" else "修改庫存")
        top.configure(bg=WORKBENCH_BG)
        top.transient(self.winfo_toplevel())
        body = ttk.Frame(top, padding=12, style=WORKBENCH_FRAME_STYLE)
        body.pack(fill="both", expand=True)
        ticker_var = tk.StringVar(value="" if row is None else str(row.get("ticker") or ""))
        qty_var = tk.StringVar(value="" if row is None else str(int(row.get("qty") or 0)))
        cost_var = tk.StringVar(value="" if row is None else _money(row.get("holding_cost")))
        date_var = tk.StringVar(value="" if row is None else str(row.get("entry_date") or ""))
        for r, label in enumerate(("股票", "股數", "持有成本", "買入日")):
            ttk.Label(body, text=label, style=WORKBENCH_LABEL_STYLE).grid(row=r, column=0, sticky="w", pady=4)
        ticker_entry = ttk.Entry(body, textvariable=ticker_var, width=18, style=WORKBENCH_ENTRY_STYLE)
        ticker_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=4)
        if mode == "edit":
            ticker_entry.configure(state="readonly")
        ttk.Entry(body, textvariable=qty_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Entry(body, textvariable=cost_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)
        DatePickerField(body, textvariable=date_var, width=14).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=4)

        def save():
            try:
                ticker = ticker_var.get().strip().upper()
                if not ticker:
                    raise ValueError("股票必填")
                qty = _parse_positive_int(qty_var.get(), "股數")
                cost = _parse_positive_money(cost_var.get(), "持有成本")
                entry_date = date_var.get().strip() or None
                revision = int(self._snapshot["revision"])
                if mode == "add":
                    adopt_existing_trading_position(
                        WORKBENCH_PROJECT_ROOT,
                        ticker=ticker,
                        qty=qty,
                        cost_basis_total=cost,
                        entry_date=entry_date,
                        expected_revision=None,
                        note="Workbench accounting center inventory import",
                    )
                else:
                    correct_existing_trading_position(
                        WORKBENCH_PROJECT_ROOT,
                        ticker=ticker,
                        qty=qty,
                        cost_basis_total=cost,
                        entry_date=entry_date,
                        expected_revision=None,
                        note="Workbench accounting center inventory correction",
                    )
            except Exception as exc:
                messagebox.showerror("庫存操作失敗", str(exc), parent=top)
                return
            top.destroy()
            self._refresh_after_account_mutation()

        ttk.Button(body, text="儲存", command=save, style=WORKBENCH_BUTTON_STYLE).grid(row=5, column=0, columnspan=2, sticky="e")

    def _add_inventory_dialog(self):
        self._inventory_dialog(mode="add")

    def _edit_inventory_dialog(self):
        row = self._selected_inventory()
        if row:
            self._inventory_dialog(mode="edit", row=row)

    def _delete_inventory(self):
        row = self._selected_inventory()
        if not row:
            return
        ticker = str(row.get("ticker") or "")
        if not messagebox.askyesno(
            "刪除庫存",
            f"確定刪除 {ticker} 目前庫存？\n\n這是 broker truth 修正，不建立賣出交易，也不改變現金。",
            parent=self,
        ):
            return
        try:
            remove_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                expected_revision=None,
                note="Workbench accounting center inventory delete",
            )
        except Exception as exc:
            messagebox.showerror("刪除庫存失敗", str(exc), parent=self)
            return
        self._refresh_after_account_mutation()

    def _selected_transaction(self, side: str):
        return self._buys.selected_row() if side == "BUY" else self._sells.selected_row()

    @staticmethod
    def _row_can_edit_transaction(row) -> bool:
        return bool(row and row.get("editable", True))

    def _on_buy_selected(self, row):
        state = "normal" if self._row_can_edit_transaction(row) else "disabled"
        self._edit_buy_button.configure(state=state)
        self._delete_buy_button.configure(state=state)

    def _on_sell_selected(self, row):
        state = "normal" if self._row_can_edit_transaction(row) else "disabled"
        self._edit_sell_button.configure(state=state)
        self._delete_sell_button.configure(state=state)
        if row:
            offset_row = dict(row)
            offset_row["_table_id"] = f"offset-{int(row.get('revision') or 0)}"
            self._offset.set_rows([offset_row], preserve_selection=False)
        else:
            self._offset.set_rows([], preserve_selection=False)

    def _edit_selected_transaction(self, side: str):
        row = self._selected_transaction(side)
        if not self._row_can_edit_transaction(row):
            messagebox.showerror("修改明細", "請先選取要修改的成交明細。", parent=self)
            return
        top = tk.Toplevel(self)
        top.title("修改買入明細" if side == "BUY" else "修改賣出明細")
        top.configure(bg=WORKBENCH_BG)
        top.transient(self.winfo_toplevel())
        body = ttk.Frame(top, padding=12, style=WORKBENCH_FRAME_STYLE)
        body.pack(fill="both", expand=True)
        ticker_var = tk.StringVar(value=str(row.get("ticker") or ""))
        qty_var = tk.StringVar(value=str(int(row.get("qty") or 0)))
        price_var = tk.StringVar(value=_price(row.get("price")))
        date_var = tk.StringVar(value=str(row.get("trade_date") or ""))
        for r, label in enumerate(("股票", "數量", "成交價", "成交日")):
            ttk.Label(body, text=label, style=WORKBENCH_LABEL_STYLE).grid(row=r, column=0, sticky="w", pady=4)
        ttk.Label(body, textvariable=ticker_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_TEXT).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=4)
        ttk.Entry(body, textvariable=qty_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        price_widget = None
        date_field = None
        buy_constraint_state = {"allowed_dates": (), "price_options": ()}
        if side == "BUY":
            price_widget = ttk.Combobox(
                body, textvariable=price_var, values=(), width=18, state="normal", style=WORKBENCH_COMBO_STYLE
            )
            price_widget.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)
            date_field = DatePickerField(body, textvariable=date_var, width=14, allowed_dates=())
            date_field.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=4)

            def refresh_buy_constraints(*_args):
                try:
                    runtime = load_trading_scanner_runtime(WORKBENCH_PROJECT_ROOT)
                    constraints = build_trading_actual_fill_form_constraints(
                        WORKBENCH_PROJECT_ROOT,
                        ticker=ticker_var.get(),
                        selected_date=date_var.get().strip() or None,
                        latest_finalized_date=str(runtime.get("latest_data_date") or "").strip() or None,
                    )
                except Exception:
                    constraints = {"allowed_dates": (), "price_options": ()}
                buy_constraint_state.clear()
                buy_constraint_state.update(constraints)
                allowed = tuple(constraints.get("allowed_dates") or ())
                prices = tuple(str(value) for value in (constraints.get("price_options") or ()))
                date_field.set_allowed_dates(allowed)
                price_widget.configure(values=prices)

            date_var.trace_add("write", refresh_buy_constraints)
            refresh_buy_constraints()
        else:
            ttk.Entry(body, textvariable=price_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)
            date_field = DatePickerField(body, textvariable=date_var, width=14)
            date_field.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=4)

        def save():
            try:
                qty = _parse_positive_int(qty_var.get(), "數量")
                price = _parse_positive_money(price_var.get(), "成交價")
                trade_date = date_var.get().strip()
                if not trade_date:
                    raise ValueError("成交日必填")
                if side == "BUY":
                    allowed_dates = set(buy_constraint_state.get("allowed_dates") or ())
                    if trade_date not in allowed_dates:
                        raise ValueError("成交日不是目前最新 finalized 範圍內、且有正式市場證據的可選日期")
                    legal_prices = {float(value) for value in (buy_constraint_state.get("price_options") or ())}
                    if float(price) not in legal_prices:
                        raise ValueError("成交價不在所選成交日的合法市場價格 ticks 內")
                correct_trading_account_transaction(
                    WORKBENCH_PROJECT_ROOT,
                    transaction_revision=int(row.get("revision") or 0),
                    qty=qty,
                    price=price,
                    trade_date=trade_date,
                    expected_account_revision=None,
                )
            except Exception as exc:
                messagebox.showerror("修改明細失敗", str(exc), parent=top)
                return
            top.destroy()
            self._refresh_after_account_mutation()

        ttk.Button(body, text="儲存修改", command=save, style=WORKBENCH_BUTTON_STYLE).grid(row=5, column=0, columnspan=2, sticky="e")

    def _delete_selected_transaction(self, side: str):
        row = self._selected_transaction(side)
        if not self._row_can_edit_transaction(row):
            messagebox.showerror("刪除明細", "請先選取要刪除的成交明細。", parent=self)
            return
        label = "買入" if side == "BUY" else "賣出"
        if not messagebox.askyesno(
            "刪除明細",
            f"確定刪除 {row.get('ticker')} 這筆{label}明細？\n\n原始 event 仍保留供稽核，系統會追加 void 並回復現金/庫存。",
            parent=self,
        ):
            return
        try:
            delete_trading_transaction(
                WORKBENCH_PROJECT_ROOT,
                transaction_revision=int(row.get("revision") or 0),
                expected_revision=None,
                note=f"Workbench accounting center delete {label}",
            )
        except Exception as exc:
            messagebox.showerror("刪除明細失敗", str(exc), parent=self)
            return
        self._refresh_after_account_mutation()

    def _open_stock(self, ticker):
        callback = getattr(self.winfo_toplevel(), "_open_single_stock_inspector", None)
        if callable(callback):
            callback(str(ticker), runtime_domain="trading", auto_run=True)


__all__ = ["AccountingCenterPanel"]
