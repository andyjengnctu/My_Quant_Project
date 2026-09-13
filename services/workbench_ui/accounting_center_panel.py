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
    correct_trading_transaction,
    delete_trading_transaction,
    get_trading_account_read_model,
    initialize_trading_account_state,
    remove_existing_trading_position,
    set_trading_cash_balance,
)
from services.trading.account_trade_entry import record_trading_account_inventory_sell
from services.workbench_ui.date_picker import DatePickerField
from services.workbench_ui.paged_table import PagedTable, TableColumn
from services.workbench_ui.workbench import (
    WORKBENCH_BG,
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_ENTRY_STYLE,
    WORKBENCH_ERROR,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_SUCCESS,
    WORKBENCH_TEXT,
    WORKBENCH_VSCROLL_STYLE,
)

WORKBENCH_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _money(value):
    if value is None:
        return "-"
    return f"{float(value):,.2f}"


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
        self._all_buy_rows: list[dict] = []
        self._all_sell_rows: list[dict] = []
        self._build_ui()
        self.after(80, self.refresh)

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(self, background=WORKBENCH_BG, highlightthickness=0, borderwidth=0)
        page_scroll = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview, style=WORKBENCH_VSCROLL_STYLE)
        self._canvas.configure(yscrollcommand=page_scroll.set, yscrollincrement=36)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        page_scroll.grid(row=0, column=1, sticky="ns")
        content = ttk.Frame(self._canvas, style=WORKBENCH_FRAME_STYLE)
        self._window = self._canvas.create_window((0, 0), window=content, anchor="nw")
        content.columnconfigure(0, weight=1)
        content.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")), add="+")
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._window, width=max(1, int(e.width))), add="+")

        title = ttk.LabelFrame(content, text="帳務中心｜庫存、買賣明細、績效", padding=10, style=WORKBENCH_LABELLF_STYLE)
        title.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._status_var = tk.StringVar(value="帳務資料載入中…")
        ttk.Label(title, textvariable=self._status_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="left")
        ttk.Button(title, text="重新整理", command=self.refresh, style=WORKBENCH_BUTTON_STYLE).pack(side="right")

        dashboard = ttk.LabelFrame(content, text="帳戶儀表板", padding=10, style=WORKBENCH_LABELLF_STYLE)
        dashboard.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        grid = ttk.Frame(dashboard, style=WORKBENCH_FRAME_STYLE)
        grid.pack(fill="x")
        self._metric_vars = {key: tk.StringVar(value="-") for key in ("cash", "market", "equity", "unrealized", "risk", "count")}
        self._metric_labels = {}
        for col, (label, key) in enumerate((("現金", "cash"), ("持股市值", "market"), ("帳戶淨值", "equity"), ("未實現損益", "unrealized"), ("持倉 Stop 風險", "risk"), ("庫存檔數", "count"))):
            box = ttk.LabelFrame(grid, text=label, padding=(8, 5), style=WORKBENCH_LABELLF_STYLE)
            box.grid(row=0, column=col, padx=(0 if col == 0 else 6, 0), sticky="nsew")
            metric_label = ttk.Label(box, textvariable=self._metric_vars[key], style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_TEXT)
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

        inventory = ttk.LabelFrame(content, text="庫存股｜點一下選取，再點同一列取消選取", padding=8, style=WORKBENCH_LABELLF_STYLE)
        inventory.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        inventory.columnconfigure(0, weight=1)
        self._inventory = PagedTable(
            inventory,
            columns=(
                TableColumn("ticker", "股票", 8),
                TableColumn("average_cost", "均價", 10, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("qty", "股數", 9, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("holding_cost", "持有成本", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("current_price", "市價", 10, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("market_value", "市值", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("unrealized_pnl", "損益", 11, sort_kind="numeric", performance=True, formatter=lambda v, _r: _money(v)),
                TableColumn("holding_return_pct", "報酬率", 9, sort_kind="numeric", performance=True, formatter=lambda v, _r: _pct(v)),
            ),
            page_size=12,
            default_sort_key="ticker",
            empty_text="目前沒有庫存股",
            on_select=self._on_inventory_selected,
            on_open_stock=self._open_stock,
        )
        self._inventory.grid(row=0, column=0, sticky="ew")

        inv_actions = ttk.Frame(inventory, style=WORKBENCH_FRAME_STYLE)
        inv_actions.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(inv_actions, text="新增既有庫存", command=self._add_inventory_dialog, style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        self._edit_inventory_button = ttk.Button(inv_actions, text="修改庫存", command=self._edit_inventory_dialog, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_inventory_button.pack(side="left", padx=(6, 0))
        self._delete_inventory_button = ttk.Button(inv_actions, text="刪除庫存", command=self._delete_inventory, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_inventory_button.pack(side="left", padx=(6, 0))

        sell_entry = ttk.LabelFrame(inventory, text="選取庫存後登錄賣出成交", padding=8, style=WORKBENCH_LABELLF_STYLE)
        sell_entry.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._sell_ticker_var = tk.StringVar(value="-")
        self._sell_qty_var = tk.StringVar()
        self._sell_price_var = tk.StringVar()
        self._sell_date_var = tk.StringVar()
        ttk.Label(sell_entry, text="股票", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(sell_entry, text="數量", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(sell_entry, text="成交價", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(sell_entry, text="成交日", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=3, sticky="w", padx=(8, 0))
        ttk.Label(sell_entry, textvariable=self._sell_ticker_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_TEXT).grid(row=1, column=0, sticky="ew")
        ttk.Entry(sell_entry, textvariable=self._sell_qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=1, sticky="ew", padx=(8, 0))
        ttk.Entry(sell_entry, textvariable=self._sell_price_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=2, sticky="ew", padx=(8, 0))
        DatePickerField(sell_entry, textvariable=self._sell_date_var, width=12).grid(row=1, column=3, sticky="ew", padx=(8, 0))
        self._sell_button = ttk.Button(sell_entry, text="登錄賣出成交", command=self._record_inventory_sell, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._sell_button.grid(row=1, column=4, sticky="e", padx=(8, 0))
        ttk.Label(
            sell_entry,
            text="先在券商完成賣出，再登錄實際股數、成交價與成交日；系統不管理券商掛單。",
            style=WORKBENCH_LABEL_STYLE,
            foreground=WORKBENCH_MUTED,
        ).grid(row=2, column=0, columnspan=5, sticky="w", pady=(6, 0))

        buy_box = ttk.LabelFrame(content, text="買入明細｜未選庫存時顯示全部；選取庫存後只顯示目前庫存對應買入", padding=8, style=WORKBENCH_LABELLF_STYLE)
        buy_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        buy_box.columnconfigure(0, weight=1)
        self._buys = PagedTable(
            buy_box,
            columns=(
                TableColumn("ticker", "股票", 8),
                TableColumn("trade_date", "成交日", 11, sort_kind="date"),
                TableColumn("price", "成交價", 10, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("qty", "數量", 9, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("gross_amount", "價金", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("buy_fee", "買入手續費", 11, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("holding_cost", "持有成本", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("source", "來源", 9),
            ),
            page_size=12,
            default_sort_key="trade_date",
            default_desc=False,
            empty_text="目前沒有買入明細",
            on_select=self._on_buy_selected,
            on_open_stock=self._open_stock,
        )
        self._buys.grid(row=0, column=0, sticky="ew")
        buy_actions = ttk.Frame(buy_box, style=WORKBENCH_FRAME_STYLE)
        buy_actions.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._edit_buy_button = ttk.Button(buy_actions, text="修改選取買入", command=lambda: self._edit_selected_transaction("BUY"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_buy_button.pack(side="left")
        self._delete_buy_button = ttk.Button(buy_actions, text="刪除選取買入", command=lambda: self._delete_selected_transaction("BUY"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_buy_button.pack(side="left", padx=(6, 0))
        ttk.Label(buy_actions, text="手動與策略成交都可修正；若有後續交易，請從同股票最新一筆往回處理。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="right")

        sell_box = ttk.LabelFrame(content, text="賣出明細｜含沖抵持有成本", padding=8, style=WORKBENCH_LABELLF_STYLE)
        sell_box.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        sell_box.columnconfigure(0, weight=1)
        self._sells = PagedTable(
            sell_box,
            columns=(
                TableColumn("ticker", "股票", 8),
                TableColumn("trade_date", "日期", 11, sort_kind="date"),
                TableColumn("qty", "股數", 9, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("price", "成交價", 10, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
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
        )
        self._sells.grid(row=0, column=0, sticky="ew")
        sell_actions = ttk.Frame(sell_box, style=WORKBENCH_FRAME_STYLE)
        sell_actions.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._edit_sell_button = ttk.Button(sell_actions, text="修改選取賣出", command=lambda: self._edit_selected_transaction("SELL"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_sell_button.pack(side="left")
        self._delete_sell_button = ttk.Button(sell_actions, text="刪除選取賣出", command=lambda: self._delete_selected_transaction("SELL"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_sell_button.pack(side="left", padx=(6, 0))
        ttk.Label(sell_actions, text="修改/刪除保留原 event，另追加 correction/void；策略與手動成交採同一稽核規則。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="right")

        offset_box = ttk.LabelFrame(content, text="沖抵明細｜選取上方賣出紀錄", padding=8, style=WORKBENCH_LABELLF_STYLE)
        offset_box.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        offset_box.columnconfigure(0, weight=1)
        self._offset = PagedTable(
            offset_box,
            columns=(
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
        )
        self._offset.grid(row=0, column=0, sticky="ew")

        perf = ttk.LabelFrame(content, text="績效統計｜只對有方向的損益/報酬率著色：漲紅、跌綠、平白", padding=8, style=WORKBENCH_LABELLF_STYLE)
        perf.grid(row=7, column=0, sticky="ew")
        perf.columnconfigure(0, weight=1)
        self._perf = PagedTable(
            perf,
            columns=(
                TableColumn("scope", "範圍", 12),
                TableColumn("position_or_trade_count", "持股/交易數", 10, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("cost_basis", "成本基礎", 12, sort_kind="numeric", formatter=lambda v, _r: _money(v)),
                TableColumn("pnl", "損益", 12, sort_kind="numeric", performance=True, formatter=lambda v, _r: _money(v)),
                TableColumn("return_pct", "報酬率", 10, sort_kind="numeric", performance=True, formatter=lambda v, _r: _pct(v)),
                TableColumn("profitable_count", "獲利數", 9, sort_kind="numeric", formatter=lambda v, _r: _integer(v)),
                TableColumn("profitable_rate_pct", "獲利率/勝率", 11, sort_kind="numeric", formatter=lambda v, _r: _pct(v)),
            ),
            page_size=12,
            stock_key=None,
            default_sort_key="scope",
            empty_text="目前沒有可統計績效",
        )
        self._perf.grid(row=0, column=0, sticky="ew")

    def refresh(self):
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        self._status_var.set("帳務資料載入中…")
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
            self._status_var.set(f"帳務資料讀取失敗：{error}")
            self._init_button.configure(state="normal")
            self._cash_button.configure(state="disabled")
            return
        self._snapshot = dict(account or {})
        self._dashboard = dict(dashboard or {})
        self._render()

    def _render(self):
        summary = dict(self._dashboard.get("summary") or {})
        self._metric_vars["cash"].set(_money(summary.get("cash")))
        self._metric_vars["market"].set(_money(summary.get("holdings_market_value")))
        self._metric_vars["equity"].set(_money(summary.get("equity")))
        self._metric_vars["unrealized"].set(_money(summary.get("unrealized_pnl")))
        self._metric_vars["risk"].set(_money(summary.get("managed_open_risk")))
        self._metric_vars["count"].set(str(int(summary.get("position_count") or 0)))
        unrealized = summary.get("unrealized_pnl")
        self._metric_labels["unrealized"].configure(
            foreground=WORKBENCH_ERROR if unrealized is not None and float(unrealized) > 0 else WORKBENCH_SUCCESS if unrealized is not None and float(unrealized) < 0 else WORKBENCH_TEXT
        )
        self._cash_var.set(_money(self._snapshot.get("cash")) if self._snapshot.get("cash") is not None else "")
        self._init_button.configure(state="disabled")
        self._cash_button.configure(state="normal")
        self._status_var.set(f"revision {int(self._snapshot.get('revision') or 0)}｜市價日 {self._dashboard.get('market_date') or '-'}")

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
        self.refresh()

    def _set_cash(self):
        try:
            cash = self._parse_cash()
            if cash is None:
                raise ValueError("現金必填")
            set_trading_cash_balance(
                WORKBENCH_PROJECT_ROOT,
                cash=cash,
                expected_revision=int(self._snapshot["revision"]),
                note="Workbench accounting center cash reconciliation",
            )
        except Exception as exc:
            messagebox.showerror("帳務中心", str(exc), parent=self)
            return
        self.refresh()

    def _selected_inventory(self):
        return self._inventory.selected_row()

    def _on_inventory_selected(self, row, *, rerender_buys=True):
        active = bool(row)
        self._edit_inventory_button.configure(state="normal" if active else "disabled")
        self._delete_inventory_button.configure(state="normal" if active else "disabled")
        self._sell_button.configure(state="normal" if active else "disabled")
        self._sell_ticker_var.set(str(row.get("ticker") or "-") if row else "-")
        if not row:
            self._sell_qty_var.set("")
            self._sell_price_var.set("")
            self._sell_date_var.set("")
        if rerender_buys:
            self._apply_inventory_filter()

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
            expected_revision = int(self._snapshot["revision"])
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
                expected_account_revision=expected_revision,
            )
        except Exception as exc:
            messagebox.showerror("賣出登錄失敗", str(exc), parent=self)
            return
        self._sell_qty_var.set("")
        self._sell_price_var.set("")
        self._sell_date_var.set("")
        messagebox.showinfo("賣出登錄", f"{ticker} 實際賣出成交已寫入帳戶 SSOT。", parent=self)
        self.refresh()

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
        ttk.Label(body, text="這是券商庫存對帳修正，不建立買賣成交、也不變更現金；手動/策略庫存皆可修正。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 8))

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
                        expected_revision=revision,
                        note="Workbench accounting center inventory import",
                    )
                else:
                    correct_existing_trading_position(
                        WORKBENCH_PROJECT_ROOT,
                        ticker=ticker,
                        qty=qty,
                        cost_basis_total=cost,
                        entry_date=entry_date,
                        expected_revision=revision,
                        note="Workbench accounting center inventory correction",
                    )
            except Exception as exc:
                messagebox.showerror("庫存操作失敗", str(exc), parent=top)
                return
            top.destroy()
            self.refresh()

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
                expected_revision=int(self._snapshot["revision"]),
                note="Workbench accounting center inventory delete",
            )
        except Exception as exc:
            messagebox.showerror("刪除庫存失敗", str(exc), parent=self)
            return
        self.refresh()

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
        price_var = tk.StringVar(value=_money(row.get("price")))
        date_var = tk.StringVar(value=str(row.get("trade_date") or ""))
        for r, label in enumerate(("股票", "數量", "成交價", "成交日")):
            ttk.Label(body, text=label, style=WORKBENCH_LABEL_STYLE).grid(row=r, column=0, sticky="w", pady=4)
        ttk.Label(body, textvariable=ticker_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_TEXT).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=4)
        ttk.Entry(body, textvariable=qty_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Entry(body, textvariable=price_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)
        DatePickerField(body, textvariable=date_var, width=14).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Label(body, text="原紀錄保留於 event audit trail；策略與手動成交皆以 void + replacement 修正。若已有後續交易，請先從最新一筆往回修正。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 8))

        def save():
            try:
                qty = _parse_positive_int(qty_var.get(), "數量")
                price = _parse_positive_money(price_var.get(), "成交價")
                trade_date = date_var.get().strip()
                if not trade_date:
                    raise ValueError("成交日必填")
                correct_trading_transaction(
                    WORKBENCH_PROJECT_ROOT,
                    transaction_revision=int(row.get("revision") or 0),
                    qty=qty,
                    price=price,
                    trade_date=trade_date,
                    expected_revision=int(self._snapshot["revision"]),
                )
            except Exception as exc:
                messagebox.showerror("修改明細失敗", str(exc), parent=top)
                return
            top.destroy()
            self.refresh()

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
                expected_revision=int(self._snapshot["revision"]),
                note=f"Workbench accounting center delete {label}",
            )
        except Exception as exc:
            messagebox.showerror("刪除明細失敗", str(exc), parent=self)
            return
        self.refresh()

    def _open_stock(self, ticker):
        callback = getattr(self.winfo_toplevel(), "_open_single_stock_inspector", None)
        if callable(callback):
            callback(str(ticker), runtime_domain="trading", auto_run=True)


__all__ = ["AccountingCenterPanel"]
