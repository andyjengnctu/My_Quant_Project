from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

import pandas as pd

from core.dataset_profiles import DATASET_PROFILE_REDUCED, DEFAULT_DATASET_PROFILE, get_dataset_profile_label
from tools.debug.trade_log import run_debug_ticker_analysis


DATASET_OPTIONS = (
    (DEFAULT_DATASET_PROFILE, get_dataset_profile_label(DEFAULT_DATASET_PROFILE)),
    (DATASET_PROFILE_REDUCED, get_dataset_profile_label(DATASET_PROFILE_REDUCED)),
)


SUMMARY_FIELDS = (
    ("dataset_label", "資料集"),
    ("file_path", "資料來源"),
    ("excel_path", "交易明細"),
    ("chart_path", "K線檢視"),
    ("trade_count", "明細列數"),
    ("buy_count", "買進列數"),
    ("sell_count", "賣出列數"),
    ("missed_buy_count", "錯失買進列數"),
    ("missed_sell_count", "錯失賣出列數"),
)


class SingleStockBacktestInspectorPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self._result = None
        self._summary_vars = {key: tk.StringVar(value="-") for key, _label in SUMMARY_FIELDS}
        self._status_var = tk.StringVar(value="尚未執行")
        self._dataset_var = tk.StringVar(value=DEFAULT_DATASET_PROFILE)
        self._ticker_var = tk.StringVar()
        self._columns = []
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 10))

        controls = ttk.LabelFrame(top, text="執行參數", padding=10)
        controls.pack(fill="x")
        controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="股票代號").grid(row=0, column=0, sticky="w")
        ticker_entry = ttk.Entry(controls, textvariable=self._ticker_var, width=20)
        ticker_entry.grid(row=0, column=1, padx=(6, 14), sticky="w")
        ticker_entry.focus_set()

        ttk.Label(controls, text="資料集").grid(row=0, column=2, sticky="w")
        dataset_combo = ttk.Combobox(
            controls,
            state="readonly",
            width=10,
            values=[label for _key, label in DATASET_OPTIONS],
        )
        dataset_combo.grid(row=0, column=3, padx=(6, 14), sticky="w")
        dataset_combo.current(0)

        def _sync_dataset(*_args):
            selected_label = dataset_combo.get()
            for key, label in DATASET_OPTIONS:
                if label == selected_label:
                    self._dataset_var.set(key)
                    break

        dataset_combo.bind("<<ComboboxSelected>>", _sync_dataset)
        _sync_dataset()

        ttk.Button(controls, text="執行回測", command=self._run_analysis).grid(row=0, column=4, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟 K 線圖", command=self._open_chart).grid(row=0, column=5, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟 Excel", command=self._open_excel).grid(row=0, column=6, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟輸出資料夾", command=self._open_output_dir).grid(row=0, column=7, sticky="w")

        ttk.Label(top, textvariable=self._status_var).pack(anchor="w", pady=(8, 0))

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True)

        summary_frame = ttk.LabelFrame(body, text="執行摘要", padding=10)
        body.add(summary_frame, weight=1)
        for row_idx, (key, label) in enumerate(SUMMARY_FIELDS):
            ttk.Label(summary_frame, text=label).grid(row=row_idx, column=0, sticky="nw", pady=2)
            ttk.Label(summary_frame, textvariable=self._summary_vars[key], wraplength=360, justify="left").grid(
                row=row_idx,
                column=1,
                sticky="nw",
                padx=(8, 0),
                pady=2,
            )
        summary_frame.columnconfigure(1, weight=1)

        table_frame = ttk.LabelFrame(body, text="交易明細", padding=10)
        body.add(table_frame, weight=3)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(table_frame, show="headings")
        self._tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self._tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self._tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self._tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    def _run_analysis(self):
        ticker = self._ticker_var.get().strip()
        if not ticker:
            messagebox.showerror("股票工具工作台", "請先輸入股票代號。")
            return

        dataset_key = self._dataset_var.get().strip() or DEFAULT_DATASET_PROFILE
        self._status_var.set(f"執行中：{ticker} / {get_dataset_profile_label(dataset_key)}")
        self.update_idletasks()

        try:
            result = run_debug_ticker_analysis(
                ticker,
                dataset_profile_key=dataset_key,
                export_excel=True,
                export_chart=True,
                verbose=False,
            )
        except Exception as exc:
            self._status_var.set(f"執行失敗：{type(exc).__name__}: {exc}")
            messagebox.showerror("股票工具工作台", str(exc))
            return

        self._result = result
        self._render_result(result)
        self._status_var.set(f"完成：{ticker} / {get_dataset_profile_label(dataset_key)}")

    def _render_result(self, result):
        trade_logs_df = result.get("trade_logs_df")
        trade_count = 0 if trade_logs_df is None else int(len(trade_logs_df))
        buy_count = 0
        sell_count = 0
        missed_buy_count = 0
        missed_sell_count = 0
        if trade_logs_df is not None and not trade_logs_df.empty:
            actions = trade_logs_df["動作"].astype(str)
            buy_count = int(actions.str.contains("買進").sum())
            sell_count = int(actions.isin(["半倉停利", "停損殺出", "指標賣出", "期末強制結算"]).sum())
            missed_buy_count = int(actions.str.startswith("錯失買進").sum())
            missed_sell_count = int((actions == "錯失賣出").sum())

        summary_values = {
            "dataset_label": result.get("dataset_label", "-"),
            "file_path": result.get("file_path", "-"),
            "excel_path": result.get("excel_path") or "-",
            "chart_path": result.get("chart_path") or "-",
            "trade_count": str(trade_count),
            "buy_count": str(buy_count),
            "sell_count": str(sell_count),
            "missed_buy_count": str(missed_buy_count),
            "missed_sell_count": str(missed_sell_count),
        }
        for key, value in summary_values.items():
            self._summary_vars[key].set(value)

        self._render_trade_table(trade_logs_df)

    def _render_trade_table(self, trade_logs_df: pd.DataFrame | None):
        self._tree.delete(*self._tree.get_children())
        if trade_logs_df is None or trade_logs_df.empty:
            self._tree.configure(columns=("message",))
            self._tree.heading("message", text="訊息")
            self._tree.column("message", width=520, anchor="w")
            self._tree.insert("", "end", values=("這檔股票沒有任何交易紀錄。",))
            self._columns = ["message"]
            return

        columns = list(trade_logs_df.columns)
        self._tree.configure(columns=columns)
        self._columns = columns
        for column in columns:
            self._tree.heading(column, text=column)
            width = 120 if column not in {"備註", "日期", "動作"} else (260 if column == "備註" else 120)
            self._tree.column(column, width=width, anchor="center")

        normalized_df = trade_logs_df.copy()
        normalized_df = normalized_df.where(pd.notna(normalized_df), "")
        for row in normalized_df.to_dict("records"):
            self._tree.insert("", "end", values=[row.get(column, "") for column in columns])

    def _open_chart(self):
        self._open_result_path("chart_path")

    def _open_excel(self):
        self._open_result_path("excel_path")

    def _open_output_dir(self):
        if self._result is None:
            messagebox.showinfo("股票工具工作台", "尚未產生輸出。")
            return
        output_dir = None
        for key in ("excel_path", "chart_path"):
            path = self._result.get(key)
            if path:
                output_dir = os.path.dirname(path)
                break
        if not output_dir:
            messagebox.showinfo("股票工具工作台", "目前沒有可開啟的輸出資料夾。")
            return
        self._open_path(output_dir)

    def _open_result_path(self, key):
        if self._result is None:
            messagebox.showinfo("股票工具工作台", "請先執行回測。")
            return
        path = self._result.get(key)
        if not path:
            messagebox.showinfo("股票工具工作台", "目前沒有可開啟的輸出。")
            return
        self._open_path(path)

    def _open_path(self, path):
        normalized_path = os.path.abspath(path)
        if not os.path.exists(normalized_path):
            messagebox.showerror("股票工具工作台", f"找不到檔案或資料夾：{normalized_path}")
            return

        try:
            if os.name == "nt":
                os.startfile(normalized_path)  # type: ignore[attr-defined]
                return
            if sys.platform == "darwin":
                subprocess.Popen(["open", normalized_path])
                return
            webbrowser.open(f"file://{normalized_path}")
        except Exception as exc:
            messagebox.showerror("股票工具工作台", f"開啟失敗：{type(exc).__name__}: {exc}")
