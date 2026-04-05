from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

import pandas as pd

from core.dataset_profiles import DATASET_PROFILE_REDUCED, DEFAULT_DATASET_PROFILE, get_dataset_profile_label
from tools.debug.charting import bind_matplotlib_chart_navigation, create_matplotlib_debug_chart_figure
from tools.debug.trade_log import run_debug_ticker_analysis

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:  # pragma: no cover - GUI runtime fallback
    FigureCanvasTkAgg = None


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
        super().__init__(master, padding=4, style="Workbench.TFrame")
        self._result = None
        self._summary_vars = {key: tk.StringVar(value="-") for key, _label in SUMMARY_FIELDS}
        self._status_var = tk.StringVar(value="尚未執行")
        self._chart_hint_var = tk.StringVar(value="預設顯示最近 18 個月；滑鼠滾輪縮放、按住左鍵拖曳平移、左右鍵逐根移動時間軸，左上即時顯示滑鼠所在 K 棒數值。")
        self._dataset_var = tk.StringVar(value=DEFAULT_DATASET_PROFILE)
        self._ticker_var = tk.StringVar()
        self._show_volume_var = tk.BooleanVar(value=False)
        self._columns = []
        self._chart_canvas = None
        self._chart_figure = None
        self._build_ui()

    def destroy(self):
        self._clear_embedded_chart()
        super().destroy()

    def _build_ui(self):
        controls = ttk.LabelFrame(self, text="執行參數", padding=8, style="Workbench.TLabelframe")
        controls.pack(fill="x", pady=(0, 4))
        controls.columnconfigure(10, weight=1)

        ttk.Label(controls, text="股票代號", style="Workbench.TLabel").grid(row=0, column=0, sticky="w")
        ticker_entry = ttk.Entry(controls, textvariable=self._ticker_var, width=18, style="Workbench.TEntry")
        ticker_entry.grid(row=0, column=1, padx=(6, 12), sticky="w")
        ticker_entry.focus_set()

        ttk.Label(controls, text="資料集", style="Workbench.TLabel").grid(row=0, column=2, sticky="w")
        dataset_combo = ttk.Combobox(
            controls,
            state="readonly",
            width=10,
            values=[label for _key, label in DATASET_OPTIONS],
            style="Workbench.TCombobox",
        )
        dataset_combo.grid(row=0, column=3, padx=(6, 12), sticky="w")
        dataset_combo.current(0)

        def _sync_dataset(*_args):
            selected_label = dataset_combo.get()
            for key, label in DATASET_OPTIONS:
                if label == selected_label:
                    self._dataset_var.set(key)
                    break

        dataset_combo.bind("<<ComboboxSelected>>", _sync_dataset)
        _sync_dataset()

        ttk.Button(controls, text="執行回測", command=self._run_analysis, style="Workbench.TButton").grid(row=0, column=4, padx=(0, 8), sticky="w")
        ttk.Checkbutton(controls, text="顯示成交量", variable=self._show_volume_var, command=self._rerender_current_chart, style="Workbench.TCheckbutton").grid(row=0, column=5, padx=(0, 12), sticky="w")
        ttk.Button(controls, text="開啟 HTML K 線圖", command=self._open_chart, style="Workbench.TButton").grid(row=0, column=6, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟 Excel", command=self._open_excel, style="Workbench.TButton").grid(row=0, column=7, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟輸出資料夾", command=self._open_output_dir, style="Workbench.TButton").grid(row=0, column=8, sticky="w")

        notebook = ttk.Notebook(self, style="Workbench.TNotebook")
        notebook.pack(fill="both", expand=True)
        self._notebook = notebook

        chart_tab = ttk.Frame(notebook, padding=1, style="Workbench.TFrame")
        chart_tab.rowconfigure(0, weight=1)
        chart_tab.columnconfigure(0, weight=1)
        notebook.add(chart_tab, text="K 線圖")

        self._chart_canvas_host = tk.Frame(chart_tab, bg=ttk.Style(self).lookup("Workbench.TFrame", "background") or "#05090e", highlightthickness=0, bd=0)
        self._chart_canvas_host.grid(row=0, column=0, sticky="nsew")
        chart_tab.configure(style="Workbench.TFrame")
        self._chart_placeholder = tk.Label(
            self._chart_canvas_host,
            text="請先執行回測；K 線圖會直接顯示在此。",
            anchor="center",
            justify="center",
            bg=ttk.Style(self).lookup("Workbench.TFrame", "background") or "#05090e",
            fg=ttk.Style(self).lookup("Workbench.TLabel", "foreground") or "#f7fbff",
        )
        self._chart_placeholder.pack(fill="both", expand=True)

        summary_tab = ttk.Frame(notebook, padding=10, style="Workbench.TFrame")
        notebook.add(summary_tab, text="執行摘要")
        for row_idx, (key, label) in enumerate(SUMMARY_FIELDS):
            ttk.Label(summary_tab, text=label, style="Workbench.TLabel").grid(row=row_idx, column=0, sticky="nw", pady=3)
            ttk.Label(summary_tab, textvariable=self._summary_vars[key], wraplength=860, justify="left", style="Workbench.TLabel").grid(
                row=row_idx,
                column=1,
                sticky="nw",
                padx=(8, 0),
                pady=3,
            )
        summary_tab.columnconfigure(1, weight=1)

        table_tab = ttk.Frame(notebook, padding=10, style="Workbench.TFrame")
        notebook.add(table_tab, text="交易明細")
        table_tab.rowconfigure(0, weight=1)
        table_tab.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(table_tab, show="headings", style="Workbench.Treeview")
        self._tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(table_tab, orient="vertical", command=self._tree.yview, style="Workbench.Vertical.TScrollbar")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(table_tab, orient="horizontal", command=self._tree.xview, style="Workbench.Horizontal.TScrollbar")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self._tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        footer = ttk.Frame(self, style="Workbench.TFrame")
        footer.pack(fill="x", pady=(4, 0))
        ttk.Label(footer, textvariable=self._status_var, style="Workbench.TLabel").pack(anchor="w")
        ttk.Label(footer, textvariable=self._chart_hint_var, wraplength=1500, justify="left", style="Workbench.TLabel").pack(anchor="w", pady=(2, 0))

        self._notebook.select(chart_tab)

    def _run_analysis(self):
        ticker = self._ticker_var.get().strip()
        if not ticker:
            messagebox.showerror("股票工具工作台", "請先輸入股票代號。")
            return

        dataset_key = self._dataset_var.get().strip() or DEFAULT_DATASET_PROFILE
        self._status_var.set(f"執行中：{ticker} / {get_dataset_profile_label(dataset_key)}")
        self._chart_hint_var.set("載入 K 線圖中…")
        self.update_idletasks()

        try:
            result = run_debug_ticker_analysis(
                ticker,
                dataset_profile_key=dataset_key,
                export_excel=True,
                export_chart=False,
                return_chart_payload=True,
                verbose=False,
            )
        except Exception as exc:
            self._status_var.set(f"執行失敗：{type(exc).__name__}: {exc}")
            self._chart_hint_var.set("執行失敗，無法顯示 K 線圖。")
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
        self._render_embedded_chart(result)

    def _render_trade_table(self, trade_logs_df: pd.DataFrame | None):
        self._tree.delete(*self._tree.get_children())
        if trade_logs_df is None or trade_logs_df.empty:
            self._tree.configure(columns=("message",))
            self._tree.heading("message", text="訊息")
            self._tree.column("message", width=920, anchor="w")
            self._tree.insert("", "end", values=("這檔股票沒有任何交易紀錄。",))
            self._columns = ["message"]
            return

        columns = list(trade_logs_df.columns)
        self._tree.configure(columns=columns)
        self._columns = columns
        for column in columns:
            self._tree.heading(column, text=column)
            width = 120 if column not in {"備註", "日期", "動作"} else (340 if column == "備註" else 120)
            self._tree.column(column, width=width, anchor="center")

        normalized_df = trade_logs_df.copy()
        normalized_df = normalized_df.where(pd.notna(normalized_df), "")
        for row in normalized_df.to_dict("records"):
            self._tree.insert("", "end", values=[row.get(column, "") for column in columns])

    def _rerender_current_chart(self):
        if self._result is not None:
            self._render_embedded_chart(self._result)

    def _render_embedded_chart(self, result):
        chart_payload = result.get("chart_payload")
        ticker = result.get("ticker", "")
        if chart_payload is None:
            self._clear_embedded_chart()
            self._chart_hint_var.set("目前沒有可顯示的 K 線圖。")
            return

        if FigureCanvasTkAgg is None:
            self._clear_embedded_chart()
            self._chart_hint_var.set("環境缺少 matplotlib Tk backend，無法在 GUI 內嵌顯示；可改開啟 HTML K 線圖。")
            return

        try:
            figure = create_matplotlib_debug_chart_figure(
                chart_payload=chart_payload,
                ticker=ticker,
                show_volume=bool(self._show_volume_var.get()),
            )
        except Exception as exc:
            self._clear_embedded_chart()
            self._chart_hint_var.set(f"K 線圖建立失敗：{type(exc).__name__}: {exc}")
            return

        self._clear_embedded_chart()
        self._chart_placeholder.pack_forget()

        canvas = FigureCanvasTkAgg(figure, master=self._chart_canvas_host)
        bind_matplotlib_chart_navigation(figure, canvas)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.configure(background="#02050a", highlightthickness=0, bd=0, takefocus=1)
        canvas_widget.pack(fill="both", expand=True)
        canvas_widget.focus_set()

        self._chart_canvas = canvas
        self._chart_figure = figure
        self._notebook.select(0)
        if self._show_volume_var.get():
            self._chart_hint_var.set("K 線圖已內嵌於 GUI；預設從最近 18 個月開始，滑鼠滾輪縮放、左鍵拖曳平移、左右鍵逐根移動，成交量為同圖 overlay。")
        else:
            self._chart_hint_var.set("K 線圖已內嵌於 GUI；預設顯示最近 18 個月，滑鼠滾輪縮放、左鍵拖曳平移、左右鍵逐根移動，隱藏成交量以保留大圖版面。")

    def _clear_embedded_chart(self):
        if self._chart_canvas is not None:
            widget = self._chart_canvas.get_tk_widget()
            widget.destroy()
            self._chart_canvas = None
        if self._chart_figure is not None:
            self._chart_figure.clear()
            self._chart_figure = None
        if not self._chart_placeholder.winfo_ismapped():
            self._chart_placeholder.pack(fill="both", expand=True)

    def _open_chart(self):
        if self._result is None:
            messagebox.showinfo("股票工具工作台", "請先執行回測。")
            return
        if not self._result.get("chart_path"):
            ticker = self._result.get("ticker")
            dataset_key = self._result.get("dataset_profile_key", DEFAULT_DATASET_PROFILE)
            try:
                export_result = run_debug_ticker_analysis(
                    ticker,
                    dataset_profile_key=dataset_key,
                    export_excel=False,
                    export_chart=True,
                    return_chart_payload=False,
                    verbose=False,
                )
            except Exception as exc:
                messagebox.showerror("股票工具工作台", f"建立 HTML K 線圖失敗：{type(exc).__name__}: {exc}")
                return
            self._result["chart_path"] = export_result.get("chart_path")
            self._summary_vars["chart_path"].set(self._result.get("chart_path") or "-")
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
