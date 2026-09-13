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
    TradingAccountRevisionConflict,
    adopt_existing_trading_position,
    correct_existing_trading_position,
    correct_manual_trading_transaction,
    delete_manual_trading_transaction,
    get_trading_account_read_model,
    initialize_trading_account_state,
    remove_existing_trading_position,
    set_trading_cash_balance,
)
from services.trading.account_trade_entry import (
    list_active_sell_orders_for_ticker,
    record_trading_account_inventory_sell,
)
from services.workbench_ui.date_picker import DatePickerField
from services.workbench_ui.workbench import (
    WORKBENCH_BG,
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_COMBO_STYLE,
    WORKBENCH_ENTRY_STYLE,
    WORKBENCH_ERROR,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_SUCCESS,
    WORKBENCH_TEXT,
    WORKBENCH_TREE_STYLE,
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


def _direction_tag(value) -> str:
    if value is None:
        return "flat"
    value = float(value)
    if value > 0:
        return "gain"
    if value < 0:
        return "loss"
    return "flat"


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
    def __init__(self, master):
        super().__init__(master, padding=10, style=WORKBENCH_FRAME_STYLE)
        self._snapshot = {}
        self._dashboard = {}
        self._refresh_results: queue.Queue = queue.Queue()
        self._refresh_thread = None
        self._buy_rows = {}
        self._sell_rows = {}
        self._inventory_rows = {}
        self._sell_order_map = {}
        self._build_ui()
        self.after(80, self.refresh)

    def _configure_performance_tags(self, tree: ttk.Treeview) -> None:
        tree.tag_configure("gain", foreground=WORKBENCH_ERROR)   # Taiwan: 漲紅
        tree.tag_configure("loss", foreground=WORKBENCH_SUCCESS)  # Taiwan: 跌綠
        tree.tag_configure("flat", foreground=WORKBENCH_TEXT)

    def _make_empty_label(self, parent, text="目前無資料"):
        label = ttk.Label(parent, text=text, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED)
        label.grid(row=0, column=0, sticky="w", pady=4)
        return label

    @staticmethod
    def _fit_table(tree: ttk.Treeview, placeholder: ttk.Label, count: int) -> None:
        if int(count) <= 0:
            tree.grid_remove()
            placeholder.grid()
            return
        placeholder.grid_remove()
        tree.grid()
        tree.configure(height=max(1, int(count)))

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(self, background=WORKBENCH_BG, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview, style=WORKBENCH_VSCROLL_STYLE)
        self._canvas.configure(yscrollcommand=scrollbar.set, yscrollincrement=36)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
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
        for col, (label, key) in enumerate((("現金", "cash"), ("持股市值", "market"), ("帳戶淨值", "equity"), ("未實現損益", "unrealized"), ("持倉Stop風險", "risk"), ("庫存檔數", "count"))):
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

        inventory = ttk.LabelFrame(content, text="庫存股｜點選後可賣出、修改或刪除", padding=8, style=WORKBENCH_LABELLF_STYLE)
        inventory.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        inventory.columnconfigure(0, weight=1)
        cols = ("ticker", "avg", "qty", "cost", "price", "value", "pnl", "return")
        self._inventory = ttk.Treeview(inventory, columns=cols, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse", height=1)
        headings = {"ticker":"股票", "avg":"均價", "qty":"股數", "cost":"持有成本", "price":"市價", "value":"市值", "pnl":"損益", "return":"報酬率"}
        widths = {"ticker":85,"avg":105,"qty":90,"cost":125,"price":100,"value":125,"pnl":115,"return":95}
        for key in cols:
            self._inventory.heading(key, text=headings[key])
            self._inventory.column(key, width=widths[key], anchor="center")
        self._configure_performance_tags(self._inventory)
        self._inventory.grid(row=0, column=0, sticky="ew")
        self._inventory_empty = self._make_empty_label(inventory, "目前沒有庫存股")
        self._inventory.bind("<<TreeviewSelect>>", self._on_inventory_selected, add="+")
        self._inventory.bind("<Double-1>", self._open_holding, add="+")

        inv_actions = ttk.Frame(inventory, style=WORKBENCH_FRAME_STYLE)
        inv_actions.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(inv_actions, text="新增既有庫存", command=self._add_inventory_dialog, style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        self._edit_inventory_button = ttk.Button(inv_actions, text="修改庫存", command=self._edit_inventory_dialog, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_inventory_button.pack(side="left", padx=(6, 0))
        self._delete_inventory_button = ttk.Button(inv_actions, text="刪除庫存", command=self._delete_inventory, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_inventory_button.pack(side="left", padx=(6, 0))
        ttk.Button(inv_actions, text="單股檢視", command=self._open_holding, style=WORKBENCH_BUTTON_STYLE).pack(side="right")

        sell_entry = ttk.LabelFrame(inventory, text="選取庫存後登錄賣出成交", padding=8, style=WORKBENCH_LABELLF_STYLE)
        sell_entry.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._sell_ticker_var = tk.StringVar(value="-")
        self._sell_qty_var = tk.StringVar()
        self._sell_price_var = tk.StringVar()
        self._sell_date_var = tk.StringVar()
        self._sell_order_var = tk.StringVar()
        ttk.Label(sell_entry, text="股票", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(sell_entry, text="數量", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=1, sticky="w", padx=(8,0))
        ttk.Label(sell_entry, text="成交價", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=2, sticky="w", padx=(8,0))
        ttk.Label(sell_entry, text="成交日", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=3, sticky="w", padx=(8,0))
        ttk.Label(sell_entry, text="對應券商 SELL 單", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=4, sticky="w", padx=(8,0))
        ttk.Label(sell_entry, textvariable=self._sell_ticker_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_TEXT).grid(row=1, column=0, sticky="w")
        ttk.Entry(sell_entry, textvariable=self._sell_qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=1, sticky="ew", padx=(8,0))
        ttk.Entry(sell_entry, textvariable=self._sell_price_var, width=14, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=2, sticky="ew", padx=(8,0))
        DatePickerField(sell_entry, textvariable=self._sell_date_var, width=12).grid(row=1, column=3, sticky="ew", padx=(8,0))
        self._sell_order_combo = ttk.Combobox(sell_entry, textvariable=self._sell_order_var, state="readonly", width=26, style=WORKBENCH_COMBO_STYLE)
        self._sell_order_combo.grid(row=1, column=4, sticky="ew", padx=(8,0))
        self._sell_button = ttk.Button(sell_entry, text="登錄賣出成交", command=self._record_inventory_sell, style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._sell_button.grid(row=1, column=5, sticky="w", padx=(10,0))
        ttk.Label(sell_entry, text="股票由選取庫存自動帶入；只輸入數量、成交價、成交日，其餘費用/稅/損益自動計算。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).grid(row=2, column=0, columnspan=6, sticky="w", pady=(5,0))

        buy_box = ttk.LabelFrame(content, text="買入明細", padding=8, style=WORKBENCH_LABELLF_STYLE)
        buy_box.grid(row=4, column=0, sticky="ew", pady=(0, 8)); buy_box.columnconfigure(0, weight=1)
        buy_cols=("ticker","date","price","qty","gross","fee","cost","source")
        self._buys=ttk.Treeview(buy_box, columns=buy_cols, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse", height=1)
        buy_heads={"ticker":"股票","date":"成交日","price":"成交價","qty":"數量","gross":"價金","fee":"買入手續費","cost":"持有成本","source":"來源"}
        for key in buy_cols:
            self._buys.heading(key,text=buy_heads[key]); self._buys.column(key,width=110 if key not in {"ticker","qty","source"} else 90,anchor="center")
        self._buys.grid(row=0,column=0,sticky="ew")
        self._buy_empty = self._make_empty_label(buy_box, "目前沒有買入明細")
        self._buys.bind("<<TreeviewSelect>>", self._on_buy_selected, add="+")
        buy_actions = ttk.Frame(buy_box, style=WORKBENCH_FRAME_STYLE)
        buy_actions.grid(row=1, column=0, sticky="ew", pady=(6,0))
        self._edit_buy_button = ttk.Button(buy_actions, text="修改選取買入", command=lambda: self._edit_selected_transaction("BUY"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_buy_button.pack(side="left")
        self._delete_buy_button = ttk.Button(buy_actions, text="刪除選取買入", command=lambda: self._delete_selected_transaction("BUY"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_buy_button.pack(side="left", padx=(6,0))
        ttk.Label(buy_actions, text="只允許修正手動帳務成交；若該股票已有後續交易，請由最新一筆往回修正。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="right")

        sell_box = ttk.LabelFrame(content, text="賣出明細｜含沖抵持有成本", padding=8, style=WORKBENCH_LABELLF_STYLE)
        sell_box.grid(row=5, column=0, sticky="ew", pady=(0, 8)); sell_box.columnconfigure(0, weight=1)
        sell_cols=("ticker","date","qty","price","gross","fee","tax","offset","pnl","return")
        self._sells=ttk.Treeview(sell_box, columns=sell_cols, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse", height=1)
        sell_heads={"ticker":"股票","date":"日期","qty":"股數","price":"成交價","gross":"價金","fee":"賣出手續費","tax":"交易稅","offset":"沖抵持有成本","pnl":"損益","return":"報酬率"}
        for key in sell_cols:
            self._sells.heading(key,text=sell_heads[key]); self._sells.column(key,width=115 if key not in {"ticker","qty","return"} else 90,anchor="center")
        self._configure_performance_tags(self._sells)
        self._sells.grid(row=0,column=0,sticky="ew")
        self._sell_empty = self._make_empty_label(sell_box, "目前沒有賣出明細")
        self._sells.bind("<<TreeviewSelect>>", self._on_sell_selected, add="+")
        sell_actions = ttk.Frame(sell_box, style=WORKBENCH_FRAME_STYLE)
        sell_actions.grid(row=1, column=0, sticky="ew", pady=(6,0))
        self._edit_sell_button = ttk.Button(sell_actions, text="修改選取賣出", command=lambda: self._edit_selected_transaction("SELL"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._edit_sell_button.pack(side="left")
        self._delete_sell_button = ttk.Button(sell_actions, text="刪除選取賣出", command=lambda: self._delete_selected_transaction("SELL"), style=WORKBENCH_BUTTON_STYLE, state="disabled")
        self._delete_sell_button.pack(side="left", padx=(6,0))
        ttk.Label(sell_actions, text="策略 order/fill 成交不在帳務中心改寫；手動成交保留原 event 並追加 correction/void。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="right")

        offset_box = ttk.LabelFrame(content, text="沖抵明細｜選取上方賣出紀錄", padding=8, style=WORKBENCH_LABELLF_STYLE)
        offset_box.grid(row=6, column=0, sticky="ew", pady=(0, 8)); offset_box.columnconfigure(0, weight=1)
        offset_cols=("ticker","date","qty","gross","fee","cost","remaining")
        self._offset=ttk.Treeview(offset_box, columns=offset_cols, show="headings", style=WORKBENCH_TREE_STYLE, height=1)
        offset_heads={"ticker":"股票","date":"賣出日","qty":"賣出股數","gross":"沖抵買入價金","fee":"沖抵買入手續費","cost":"沖抵持有成本","remaining":"賣出後剩餘股數"}
        for key in offset_cols:
            self._offset.heading(key,text=offset_heads[key]); self._offset.column(key,width=125 if key not in {"ticker","qty","remaining"} else 100,anchor="center")
        self._offset.grid(row=0,column=0,sticky="ew")
        self._offset_empty = self._make_empty_label(offset_box, "選取賣出紀錄後顯示沖抵明細")

        perf = ttk.LabelFrame(content, text="績效統計｜漲紅、跌綠、平白", padding=8, style=WORKBENCH_LABELLF_STYLE)
        perf.grid(row=7, column=0, sticky="ew")
        perf.columnconfigure(0, weight=1)
        perf_cols=("scope","count","cost","pnl","return","wins","rate")
        self._perf=ttk.Treeview(perf, columns=perf_cols, show="headings", style=WORKBENCH_TREE_STYLE, height=1)
        perf_heads={"scope":"範圍","count":"持股/交易數","cost":"成本基礎","pnl":"損益","return":"報酬率","wins":"獲利數","rate":"獲利率/勝率"}
        for key in perf_cols:
            self._perf.heading(key,text=perf_heads[key]); self._perf.column(key,width=120,anchor="center")
        self._configure_performance_tags(self._perf)
        self._perf.grid(row=0, column=0, sticky="ew")
        self._perf_empty = self._make_empty_label(perf, "目前沒有可統計績效")

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
            self._dashboard = build_trading_account_dashboard_read_model(WORKBENCH_PROJECT_ROOT) if isinstance(error, FileNotFoundError) else {}
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

        for tree in (self._inventory, self._buys, self._sells, self._offset, self._perf):
            for item in tree.get_children():
                tree.delete(item)
        self._inventory_rows = {}
        account_positions = {
            str(row.get("ticker") or ""): dict(row)
            for row in self._snapshot.get("positions") or []
        }
        inventory_rows = [dict(row) for row in self._dashboard.get("positions") or []]
        for row in inventory_rows:
            ticker=str(row.get("ticker") or "")
            account_row = account_positions.get(ticker, {})
            row["source"] = account_row.get("source", row.get("source"))
            row["management_status"] = account_row.get("management_status", row.get("management_status"))
            row["has_sell_history"] = bool(account_row.get("has_sell_history"))
            row["entry_date"] = account_row.get("entry_date", row.get("entry_date"))
            row["inventory_editable"] = (
                str(row.get("source") or "") == "manual_adopted"
                and str(row.get("management_status") or "") == "unmanaged"
                and not bool(row.get("has_sell_history"))
            )
            self._inventory_rows[ticker] = row
            self._inventory.insert("","end",iid=ticker,tags=(_direction_tag(row.get("unrealized_pnl")),),values=(ticker,_money(row.get("average_cost")),f"{int(row.get('qty') or 0):,}",_money(row.get("holding_cost")),_money(row.get("current_price")),_money(row.get("market_value")),_money(row.get("unrealized_pnl")),_pct(row.get("holding_return_pct"))))
        self._fit_table(self._inventory, self._inventory_empty, len(inventory_rows))

        self._buy_rows = {}
        buy_rows = [dict(row) for row in self._dashboard.get("buy_details") or []]
        for row in buy_rows:
            iid=f"buy-{int(row.get('revision') or 0)}"
            self._buy_rows[iid] = row
            self._buys.insert("","end",iid=iid,values=(row.get("ticker"),row.get("trade_date") or "-",_money(row.get("price")),f"{int(row.get('qty') or 0):,}",_money(row.get("gross_amount")),_money(row.get("buy_fee")),_money(row.get("holding_cost")),row.get("source") or "-"))
        self._fit_table(self._buys, self._buy_empty, len(buy_rows))

        self._sell_rows = {}
        sell_rows = [dict(row) for row in self._dashboard.get("sell_details") or []]
        for row in sell_rows:
            iid=f"sell-{int(row.get('revision') or 0)}"
            self._sell_rows[iid] = row
            self._sells.insert("","end",iid=iid,tags=(_direction_tag(row.get("pnl")),),values=(row.get("ticker"),row.get("trade_date") or "-",f"{int(row.get('qty') or 0):,}",_money(row.get("price")),_money(row.get("gross_amount")),_money(row.get("sell_fee")),_money(row.get("tax")),_money(row.get("offset_holding_cost")),_money(row.get("pnl")),_pct(row.get("return_pct"))))
        self._fit_table(self._sells, self._sell_empty, len(sell_rows))

        perf_rows = [dict(row) for row in self._dashboard.get("performance") or []]
        for row in perf_rows:
            self._perf.insert("","end",tags=(_direction_tag(row.get("pnl")),),values=(row.get("scope"),int(row.get("position_or_trade_count") or 0),_money(row.get("cost_basis")),_money(row.get("pnl")),_pct(row.get("return_pct")),int(row.get("profitable_count") or 0),_pct(row.get("profitable_rate_pct"))))
        self._fit_table(self._perf, self._perf_empty, len(perf_rows))
        self._fit_table(self._offset, self._offset_empty, 0)
        self._on_inventory_selected()
        self._on_buy_selected()
        self._on_sell_selected()

    def _render_selected_offset(self):
        for item in self._offset.get_children():
            self._offset.delete(item)
        selected = self._sells.selection()
        row = self._sell_rows.get(str(selected[0])) if selected else None
        if not row:
            self._fit_table(self._offset, self._offset_empty, 0)
            return
        self._offset.insert("", "end", values=(
            row.get("ticker"), row.get("trade_date") or "-", f"{int(row.get('qty') or 0):,}",
            _money(row.get("offset_gross_amount")), _money(row.get("offset_buy_fee")),
            _money(row.get("offset_holding_cost")), f"{int(row.get('remaining_qty') or 0):,}",
        ))
        self._fit_table(self._offset, self._offset_empty, 1)

    def _parse_cash(self):
        text=self._cash_var.get().replace(",","").strip()
        if not text:
            return None
        value=float(text)
        if value < 0:
            raise ValueError("現金不可為負數")
        return value

    def _initialize(self):
        try:
            cash=self._parse_cash()
            initialize_trading_account_state(WORKBENCH_PROJECT_ROOT,cash=cash)
        except Exception as exc:
            messagebox.showerror("帳務中心",str(exc),parent=self); return
        self.refresh()

    def _set_cash(self):
        try:
            cash=self._parse_cash()
            if cash is None:
                raise ValueError("現金必填")
            revision=int(self._snapshot["revision"])
            set_trading_cash_balance(WORKBENCH_PROJECT_ROOT,cash=cash,expected_revision=revision,note="Workbench accounting center cash reconciliation")
        except Exception as exc:
            messagebox.showerror("帳務中心",str(exc),parent=self); return
        self.refresh()

    def _selected_inventory(self):
        selected = self._inventory.selection()
        if not selected:
            return None
        return self._inventory_rows.get(str(selected[0]))

    def _on_inventory_selected(self, _event=None):
        row = self._selected_inventory()
        active = bool(row)
        inventory_editable = bool(row and row.get("inventory_editable"))
        self._edit_inventory_button.configure(state="normal" if inventory_editable else "disabled")
        self._delete_inventory_button.configure(state="normal" if inventory_editable else "disabled")
        self._sell_button.configure(state="normal" if active else "disabled")
        self._sell_order_map = {}
        if not row:
            self._sell_ticker_var.set("-")
            self._sell_order_combo.configure(values=())
            self._sell_order_var.set("")
            return
        ticker = str(row.get("ticker") or "")
        self._sell_ticker_var.set(ticker)
        try:
            order_state = list_active_sell_orders_for_ticker(WORKBENCH_PROJECT_ROOT, ticker)
            values = []
            for order in order_state.get("orders") or []:
                label = f"{order.get('broker_order_id') or order.get('order_id')}｜{order.get('purpose') or '-'}｜剩 {int(order.get('remaining_qty') or 0):,}"
                values.append(label)
                self._sell_order_map[label] = str(order.get("order_id") or "")
            if values:
                self._sell_order_combo.configure(values=tuple(values))
                self._sell_order_var.set(values[0] if len(values) == 1 else "")
            else:
                label = "無 active SELL order｜直接帳務成交"
                self._sell_order_combo.configure(values=(label,))
                self._sell_order_var.set(label)
        except Exception as exc:
            self._sell_order_combo.configure(values=())
            self._sell_order_var.set("")
            self._status_var.set(f"券商 SELL order 讀取失敗：{exc}")

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
            order_label = self._sell_order_var.get().strip()
            active_values = list(self._sell_order_map)
            if len(active_values) > 1 and not order_label:
                raise ValueError("目前有多筆 active SELL order，請先選擇實際成交的券商單")
            selected_order_id = self._sell_order_map.get(order_label)
            expected_revision = int(self._snapshot["revision"])
        except Exception as exc:
            messagebox.showerror("賣出登錄", str(exc), parent=self)
            return
        if not messagebox.askyesno("確認賣出成交", f"{ticker}｜{qty:,} 股 @ {price:,.2f}｜{trade_date}\n\n費用、交易稅、沖抵成本與損益由帳務 SSOT 自動計算。", parent=self):
            return
        try:
            result = record_trading_account_inventory_sell(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                qty=qty,
                price=price,
                trade_date=trade_date,
                expected_account_revision=expected_revision,
                selected_order_id=selected_order_id,
            )
        except Exception as exc:
            messagebox.showerror("賣出登錄失敗", str(exc), parent=self)
            return
        self._sell_qty_var.set("")
        self._sell_price_var.set("")
        self._sell_date_var.set("")
        route = "已完成券商 order/account reconciliation" if result.get("route") == "order_account_reconciliation" else "已直接寫入帳戶 SSOT"
        refresh_errors = dict(result.get("refresh_errors") or {})
        extra = ""
        if refresh_errors.get("protection"):
            extra += f"\n保護單計畫刷新失敗：{refresh_errors['protection']}"
        if refresh_errors.get("indicator"):
            extra += f"\nIndicator 計畫刷新失敗：{refresh_errors['indicator']}"
        messagebox.showinfo("賣出登錄", f"{ticker} 賣出成交{route}。{extra}", parent=self)
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
        ticker_entry.grid(row=0, column=1, sticky="ew", padx=(8,0), pady=4)
        if mode == "edit":
            ticker_entry.configure(state="readonly")
        ttk.Entry(body, textvariable=qty_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=1, sticky="ew", padx=(8,0), pady=4)
        ttk.Entry(body, textvariable=cost_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=2, column=1, sticky="ew", padx=(8,0), pady=4)
        DatePickerField(body, textvariable=date_var, width=14).grid(row=3, column=1, sticky="ew", padx=(8,0), pady=4)
        ttk.Label(body, text="這是期初/券商庫存對帳修正，不產生買賣成交，也不變更現金。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6,8))

        def save():
            try:
                ticker=ticker_var.get().strip().upper()
                if not ticker:
                    raise ValueError("股票必填")
                qty=_parse_positive_int(qty_var.get(), "股數")
                cost=_parse_positive_money(cost_var.get(), "持有成本")
                entry_date=date_var.get().strip() or None
                revision=int(self._snapshot["revision"])
                if mode == "add":
                    adopt_existing_trading_position(WORKBENCH_PROJECT_ROOT,ticker=ticker,qty=qty,cost_basis_total=cost,entry_date=entry_date,expected_revision=revision,note="Workbench accounting center inventory import")
                else:
                    correct_existing_trading_position(WORKBENCH_PROJECT_ROOT,ticker=ticker,qty=qty,cost_basis_total=cost,entry_date=entry_date,expected_revision=revision,note="Workbench accounting center inventory correction")
            except Exception as exc:
                messagebox.showerror("庫存操作失敗", str(exc), parent=top); return
            top.destroy(); self.refresh()

        ttk.Button(body, text="儲存", command=save, style=WORKBENCH_BUTTON_STYLE).grid(row=5,column=0,columnspan=2,sticky="e")

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
        if not messagebox.askyesno("刪除庫存", f"確定刪除 {ticker} 庫存？\n\n這是 broker truth 修正，不會建立賣出交易，也不改變現金。", parent=self):
            return
        try:
            remove_existing_trading_position(WORKBENCH_PROJECT_ROOT,ticker=ticker,expected_revision=int(self._snapshot["revision"]),note="Workbench accounting center inventory delete")
        except Exception as exc:
            messagebox.showerror("刪除庫存失敗", str(exc), parent=self); return
        self.refresh()

    def _selected_transaction(self, side: str):
        if side == "BUY":
            selected = self._buys.selection()
            return self._buy_rows.get(str(selected[0])) if selected else None
        selected = self._sells.selection()
        return self._sell_rows.get(str(selected[0])) if selected else None

    @staticmethod
    def _row_can_edit_transaction(row) -> bool:
        return bool(row and row.get("editable") and row.get("is_latest_ticker_trade"))

    def _on_buy_selected(self, _event=None):
        row = self._selected_transaction("BUY")
        state = "normal" if self._row_can_edit_transaction(row) else "disabled"
        self._edit_buy_button.configure(state=state)
        self._delete_buy_button.configure(state=state)

    def _on_sell_selected(self, _event=None):
        row = self._selected_transaction("SELL")
        state = "normal" if self._row_can_edit_transaction(row) else "disabled"
        self._edit_sell_button.configure(state=state)
        self._delete_sell_button.configure(state=state)
        self._render_selected_offset()

    def _edit_selected_transaction(self, side: str):
        row = self._selected_transaction(side)
        if not self._row_can_edit_transaction(row):
            messagebox.showerror("修改明細", "只可修改該股票最新一筆有效手動帳務成交；策略成交不可在帳務中心改寫。", parent=self)
            return
        top = tk.Toplevel(self)
        top.title("修改買入明細" if side == "BUY" else "修改賣出明細")
        top.configure(bg=WORKBENCH_BG); top.transient(self.winfo_toplevel())
        body=ttk.Frame(top,padding=12,style=WORKBENCH_FRAME_STYLE); body.pack(fill="both",expand=True)
        ticker_var=tk.StringVar(value=str(row.get("ticker") or ""))
        qty_var=tk.StringVar(value=str(int(row.get("qty") or 0)))
        price_var=tk.StringVar(value=_money(row.get("price")))
        date_var=tk.StringVar(value=str(row.get("trade_date") or ""))
        for r,label in enumerate(("股票","數量","成交價","成交日")):
            ttk.Label(body,text=label,style=WORKBENCH_LABEL_STYLE).grid(row=r,column=0,sticky="w",pady=4)
        ttk.Label(body,textvariable=ticker_var,style=WORKBENCH_LABEL_STYLE,foreground=WORKBENCH_TEXT).grid(row=0,column=1,sticky="w",padx=(8,0),pady=4)
        ttk.Entry(body,textvariable=qty_var,width=18,style=WORKBENCH_ENTRY_STYLE).grid(row=1,column=1,sticky="ew",padx=(8,0),pady=4)
        ttk.Entry(body,textvariable=price_var,width=18,style=WORKBENCH_ENTRY_STYLE).grid(row=2,column=1,sticky="ew",padx=(8,0),pady=4)
        DatePickerField(body,textvariable=date_var,width=14).grid(row=3,column=1,sticky="ew",padx=(8,0),pady=4)
        ttk.Label(body,text="原紀錄保留於 event audit trail；修改會先 void 原交易，再以新數值重新入帳。",style=WORKBENCH_LABEL_STYLE,foreground=WORKBENCH_MUTED).grid(row=4,column=0,columnspan=2,sticky="w",pady=(6,8))

        def save():
            try:
                qty=_parse_positive_int(qty_var.get(),"數量")
                price=_parse_positive_money(price_var.get(),"成交價")
                trade_date=date_var.get().strip()
                if not trade_date:
                    raise ValueError("成交日必填")
                correct_manual_trading_transaction(WORKBENCH_PROJECT_ROOT,transaction_revision=int(row.get("revision") or 0),qty=qty,price=price,trade_date=trade_date,expected_revision=int(self._snapshot["revision"]))
            except Exception as exc:
                messagebox.showerror("修改明細失敗",str(exc),parent=top); return
            top.destroy(); self.refresh()

        ttk.Button(body,text="儲存修改",command=save,style=WORKBENCH_BUTTON_STYLE).grid(row=5,column=0,columnspan=2,sticky="e")

    def _delete_selected_transaction(self, side: str):
        row = self._selected_transaction(side)
        if not self._row_can_edit_transaction(row):
            messagebox.showerror("刪除明細", "只可刪除該股票最新一筆有效手動帳務成交；策略成交不可在帳務中心改寫。", parent=self)
            return
        label = "買入" if side == "BUY" else "賣出"
        if not messagebox.askyesno("刪除明細", f"確定刪除 {row.get('ticker')} 這筆{label}明細？\n\n原始 event 仍保留供稽核，系統會追加 void 並回復現金/庫存。", parent=self):
            return
        try:
            delete_manual_trading_transaction(WORKBENCH_PROJECT_ROOT,transaction_revision=int(row.get("revision") or 0),expected_revision=int(self._snapshot["revision"]),note=f"Workbench accounting center delete {label}")
        except Exception as exc:
            messagebox.showerror("刪除明細失敗",str(exc),parent=self); return
        self.refresh()

    def _open_holding(self, _event=None):
        selected=self._inventory.selection()
        if not selected:
            return
        callback=getattr(self.winfo_toplevel(),"_open_single_stock_inspector",None)
        if callable(callback):
            callback(str(selected[0]), mode="Trading")


__all__=["AccountingCenterPanel"]
