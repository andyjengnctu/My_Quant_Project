from __future__ import annotations

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
    get_trading_account_read_model,
    initialize_trading_account_state,
    set_trading_cash_balance,
)
from services.workbench_ui.workbench import (
    WORKBENCH_BG,
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_ENTRY_STYLE,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
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


class AccountingCenterPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10, style=WORKBENCH_FRAME_STYLE)
        self._snapshot = {}
        self._dashboard = {}
        self._refresh_results: queue.Queue = queue.Queue()
        self._refresh_thread = None
        self._build_ui()
        self.after(80, self.refresh)

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

        title = ttk.LabelFrame(content, text="帳務中心｜帳戶、庫存、買賣明細、績效", padding=10, style=WORKBENCH_LABELLF_STYLE)
        title.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._status_var = tk.StringVar(value="帳務資料載入中…")
        ttk.Label(title, textvariable=self._status_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="left")
        ttk.Button(title, text="重新整理", command=self.refresh, style=WORKBENCH_BUTTON_STYLE).pack(side="right")

        dashboard = ttk.LabelFrame(content, text="帳戶儀表板", padding=10, style=WORKBENCH_LABELLF_STYLE)
        dashboard.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        grid = ttk.Frame(dashboard, style=WORKBENCH_FRAME_STYLE)
        grid.pack(fill="x")
        self._metric_vars = {key: tk.StringVar(value="-") for key in ("cash", "market", "equity", "unrealized", "risk", "count")}
        for col, (label, key) in enumerate((("現金", "cash"), ("持股市值", "market"), ("帳戶淨值", "equity"), ("未實現損益", "unrealized"), ("持倉Stop風險", "risk"), ("庫存檔數", "count"))):
            box = ttk.LabelFrame(grid, text=label, padding=(8, 5), style=WORKBENCH_LABELLF_STYLE)
            box.grid(row=0, column=col, padx=(0 if col == 0 else 6, 0), sticky="nsew")
            ttk.Label(box, textvariable=self._metric_vars[key], style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_TEXT).pack(fill="x")
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
        inventory.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        inventory.columnconfigure(0, weight=1)
        cols = ("ticker", "avg", "qty", "cost", "price", "value", "pnl", "return")
        self._inventory = ttk.Treeview(inventory, columns=cols, show="headings", style=WORKBENCH_TREE_STYLE, height=8)
        headings = {"ticker":"股票", "avg":"均價", "qty":"股數", "cost":"持有成本", "price":"市價", "value":"市值", "pnl":"損益", "return":"報酬率"}
        widths = {"ticker":85,"avg":105,"qty":90,"cost":125,"price":100,"value":125,"pnl":115,"return":95}
        for key in cols:
            self._inventory.heading(key, text=headings[key]); self._inventory.column(key, width=widths[key], anchor="center")
        self._inventory.grid(row=0, column=0, sticky="nsew")
        self._inventory.bind("<Double-1>", self._open_holding, add="+")

        buy_box = ttk.LabelFrame(content, text="買入明細", padding=8, style=WORKBENCH_LABELLF_STYLE)
        buy_box.grid(row=4, column=0, sticky="nsew", pady=(0, 8)); buy_box.columnconfigure(0, weight=1)
        buy_cols=("ticker","date","price","qty","gross","fee","cost","source")
        self._buys=ttk.Treeview(buy_box, columns=buy_cols, show="headings", style=WORKBENCH_TREE_STYLE, height=7)
        buy_heads={"ticker":"股票","date":"成交日","price":"成交價","qty":"數量","gross":"價金","fee":"買入手續費","cost":"持有成本","source":"來源"}
        for key in buy_cols:
            self._buys.heading(key,text=buy_heads[key]); self._buys.column(key,width=110 if key not in {"ticker","qty","source"} else 90,anchor="center")
        self._buys.grid(row=0,column=0,sticky="nsew")

        sell_box = ttk.LabelFrame(content, text="賣出明細｜含沖抵持有成本", padding=8, style=WORKBENCH_LABELLF_STYLE)
        sell_box.grid(row=5, column=0, sticky="nsew", pady=(0, 8)); sell_box.columnconfigure(0, weight=1)
        sell_cols=("ticker","date","qty","price","gross","fee","tax","offset","pnl","return")
        self._sells=ttk.Treeview(sell_box, columns=sell_cols, show="headings", style=WORKBENCH_TREE_STYLE, height=7)
        sell_heads={"ticker":"股票","date":"日期","qty":"股數","price":"成交價","gross":"價金","fee":"賣出手續費","tax":"交易稅","offset":"沖抵持有成本","pnl":"損益","return":"報酬率"}
        for key in sell_cols:
            self._sells.heading(key,text=sell_heads[key]); self._sells.column(key,width=115 if key not in {"ticker","qty","return"} else 90,anchor="center")
        self._sells.grid(row=0,column=0,sticky="nsew")
        self._sells.bind("<<TreeviewSelect>>", self._render_selected_offset, add="+")
        ttk.Label(sell_box, text="沖抵目前依 canonical 平均成本比例分攤；不臆測券商未提供的 lot/FIFO 配對規則。", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).grid(row=1,column=0,sticky="w",pady=(5,0))

        offset_box = ttk.LabelFrame(content, text="沖抵明細｜選取上方賣出紀錄", padding=8, style=WORKBENCH_LABELLF_STYLE)
        offset_box.grid(row=6, column=0, sticky="ew", pady=(0, 8)); offset_box.columnconfigure(0, weight=1)
        offset_cols=("ticker","date","qty","gross","fee","cost","remaining")
        self._offset=ttk.Treeview(offset_box, columns=offset_cols, show="headings", style=WORKBENCH_TREE_STYLE, height=2)
        offset_heads={"ticker":"股票","date":"賣出日","qty":"賣出股數","gross":"沖抵買入價金","fee":"沖抵買入手續費","cost":"沖抵持有成本","remaining":"賣出後剩餘股數"}
        for key in offset_cols:
            self._offset.heading(key,text=offset_heads[key]); self._offset.column(key,width=125 if key not in {"ticker","qty","remaining"} else 100,anchor="center")
        self._offset.grid(row=0,column=0,sticky="ew")

        perf = ttk.LabelFrame(content, text="績效統計", padding=8, style=WORKBENCH_LABELLF_STYLE)
        perf.grid(row=7, column=0, sticky="ew")
        perf_cols=("scope","count","cost","pnl","return","wins","rate")
        self._perf=ttk.Treeview(perf, columns=perf_cols, show="headings", style=WORKBENCH_TREE_STYLE, height=3)
        perf_heads={"scope":"範圍","count":"持股/交易數","cost":"成本基礎","pnl":"損益","return":"報酬率","wins":"獲利數","rate":"獲利率/勝率"}
        for key in perf_cols:
            self._perf.heading(key,text=perf_heads[key]); self._perf.column(key,width=120,anchor="center")
        self._perf.pack(fill="x")

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
        self._cash_var.set(_money(self._snapshot.get("cash")) if self._snapshot.get("cash") is not None else "")
        self._init_button.configure(state="disabled")
        self._cash_button.configure(state="normal")
        self._status_var.set(f"revision {int(self._snapshot.get('revision') or 0)}｜市價日 {self._dashboard.get('market_date') or '-'}")

        for tree in (self._inventory, self._buys, self._sells, self._offset, self._perf):
            for item in tree.get_children(): tree.delete(item)
        for row in self._dashboard.get("positions") or []:
            ticker=str(row.get("ticker") or "")
            self._inventory.insert("","end",iid=ticker,values=(ticker,_money(row.get("average_cost")),f"{int(row.get('qty') or 0):,}",_money(row.get("holding_cost")),_money(row.get("current_price")),_money(row.get("market_value")),_money(row.get("unrealized_pnl")),_pct(row.get("holding_return_pct"))))
        for row in self._dashboard.get("buy_details") or []:
            self._buys.insert("","end",values=(row.get("ticker"),row.get("trade_date") or "-",_money(row.get("price")),f"{int(row.get('qty') or 0):,}",_money(row.get("gross_amount")),_money(row.get("buy_fee")),_money(row.get("holding_cost")),row.get("source") or "-"))
        for row in self._dashboard.get("sell_details") or []:
            iid=f"sell-{int(row.get('revision') or 0)}"
            self._sells.insert("","end",iid=iid,values=(row.get("ticker"),row.get("trade_date") or "-",f"{int(row.get('qty') or 0):,}",_money(row.get("price")),_money(row.get("gross_amount")),_money(row.get("sell_fee")),_money(row.get("tax")),_money(row.get("offset_holding_cost")),_money(row.get("pnl")),_pct(row.get("return_pct"))))
        for row in self._dashboard.get("performance") or []:
            self._perf.insert("","end",values=(row.get("scope"),int(row.get("position_or_trade_count") or 0),_money(row.get("cost_basis")),_money(row.get("pnl")),_pct(row.get("return_pct")),int(row.get("profitable_count") or 0),_pct(row.get("profitable_rate_pct"))))


    def _render_selected_offset(self, _event=None):
        for item in self._offset.get_children():
            self._offset.delete(item)
        selected = self._sells.selection()
        if not selected:
            return
        try:
            revision = int(str(selected[0]).removeprefix("sell-"))
        except ValueError:
            return
        row = next((dict(item) for item in (self._dashboard.get("sell_details") or []) if int(item.get("revision") or 0) == revision), None)
        if not row:
            return
        self._offset.insert("", "end", values=(
            row.get("ticker"),
            row.get("trade_date") or "-",
            f"{int(row.get('qty') or 0):,}",
            _money(row.get("offset_gross_amount")),
            _money(row.get("offset_buy_fee")),
            _money(row.get("offset_holding_cost")),
            f"{int(row.get('remaining_qty') or 0):,}",
        ))

    def _parse_cash(self):
        text=self._cash_var.get().replace(",","").strip()
        if not text: return None
        value=float(text)
        if value < 0: raise ValueError("現金不可為負數")
        return value

    def _initialize(self):
        try: cash=self._parse_cash()
        except ValueError as exc: messagebox.showerror("帳務中心",str(exc),parent=self); return
        try: initialize_trading_account_state(WORKBENCH_PROJECT_ROOT,cash=cash)
        except Exception as exc: messagebox.showerror("帳務中心",str(exc),parent=self); return
        self.refresh()

    def _set_cash(self):
        try:
            cash=self._parse_cash()
            if cash is None: raise ValueError("現金必填")
            revision=int(self._snapshot["revision"])
            set_trading_cash_balance(WORKBENCH_PROJECT_ROOT,cash=cash,expected_revision=revision,note="Workbench accounting center cash reconciliation")
        except Exception as exc: messagebox.showerror("帳務中心",str(exc),parent=self); return
        self.refresh()

    def _open_holding(self, _event=None):
        selected=self._inventory.selection()
        if not selected: return
        callback=getattr(self.winfo_toplevel(),"_open_single_stock_inspector",None)
        if callable(callback): callback(str(selected[0]), mode="Trading")


__all__=["AccountingCenterPanel"]
