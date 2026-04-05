from __future__ import annotations

import io
import os
import subprocess
import sys
import tkinter as tk
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from tkinter import messagebox, ttk

import pandas as pd

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_profile_label
from tools.debug.charting import bind_matplotlib_chart_navigation, create_matplotlib_debug_chart_figure, scroll_chart_to_latest
from tools.debug.trade_log import load_params, resolve_debug_data_dir, run_debug_ticker_analysis
from tools.scanner.scan_runner import run_daily_scanner

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:  # pragma: no cover - GUI runtime fallback
    FigureCanvasTkAgg = None


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


class _ConsoleWriter(io.TextIOBase):
    def __init__(self, panel: "SingleStockBacktestInspectorPanel"):
        super().__init__()
        self._panel = panel

    def write(self, text):
        if not text:
            return 0
        self._panel._append_console_text(str(text))
        return len(text)

    def flush(self):
        return None


class SingleStockBacktestInspectorPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=4, style="Workbench.TFrame")
        self._result = None
        self._summary_vars = {key: tk.StringVar(value="-") for key, _label in SUMMARY_FIELDS}
        self._status_var = tk.StringVar(value="尚未執行")
        self._chart_hint_var = tk.StringVar(value="預設顯示最近 18 個月；滑鼠滾輪縮放、按住左鍵拖曳平移、左右鍵逐根移動時間軸，左上即時顯示滑鼠所在 K 棒數值。")
        self._ticker_var = tk.StringVar()
        self._show_volume_var = tk.BooleanVar(value=False)
        self._candidate_display_var = tk.StringVar()
        self._candidate_map = {}
        self._columns = []
        self._chart_canvas = None
        self._chart_figure = None
        self._console_writer = _ConsoleWriter(self)
        self._sidebar_signal_var = tk.StringVar(value="無買入訊號")
        self._sidebar_history_var = tk.StringVar(value="歷史績效未達")
        self._sidebar_summary_var = tk.StringVar(value="-")
        self._selected_date_var = tk.StringVar(value="選取日：-")
        self._selected_tp_var = tk.StringVar(value="停利：-")
        self._selected_limit_var = tk.StringVar(value="限價：-")
        self._selected_entry_var = tk.StringVar(value="成交：-")
        self._selected_stop_var = tk.StringVar(value="停損：-")
        self._build_ui()

    def destroy(self):
        self._clear_embedded_chart()
        super().destroy()

    def _build_ui(self):
        controls = ttk.LabelFrame(self, text="執行參數", padding=8, style="Workbench.TLabelframe")
        controls.pack(fill="x", pady=(0, 4))
        controls.columnconfigure(12, weight=1)

        ttk.Label(controls, text="股票代號", style="Workbench.TLabel").grid(row=0, column=0, sticky="w")
        ticker_entry = ttk.Entry(controls, textvariable=self._ticker_var, width=18, style="Workbench.TEntry")
        ticker_entry.grid(row=0, column=1, padx=(6, 12), sticky="w")
        ticker_entry.focus_set()

        ticker_entry.bind("<Return>", self._on_ticker_enter)

        ttk.Button(controls, text="計算候選股", command=self._run_scanner, style="Workbench.TButton").grid(row=0, column=2, padx=(0, 8), sticky="w")
        self._candidate_combo = ttk.Combobox(controls, state="readonly", width=34, textvariable=self._candidate_display_var, style="Workbench.TCombobox", values=[])
        self._candidate_combo.grid(row=0, column=3, padx=(0, 12), sticky="w")
        self._candidate_combo.bind("<<ComboboxSelected>>", self._on_candidate_selected)

        ttk.Checkbutton(controls, text="顯示成交量", variable=self._show_volume_var, command=self._rerender_current_chart, style="Workbench.TCheckbutton").grid(row=0, column=4, padx=(0, 12), sticky="w")
        ttk.Button(controls, text="開啟 HTML K 線圖", command=self._open_chart, style="Workbench.TButton").grid(row=0, column=5, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟 Excel", command=self._open_excel, style="Workbench.TButton").grid(row=0, column=6, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟輸出資料夾", command=self._open_output_dir, style="Workbench.TButton").grid(row=0, column=7, sticky="w")

        notebook = ttk.Notebook(self, style="Workbench.TNotebook")
        notebook.pack(fill="both", expand=True)
        self._notebook = notebook

        chart_tab = ttk.Frame(notebook, padding=1, style="Workbench.TFrame")
        chart_tab.rowconfigure(0, weight=1)
        chart_tab.columnconfigure(0, weight=1)
        chart_tab.columnconfigure(1, weight=0)
        notebook.add(chart_tab, text="K 線圖")

        chart_bg = ttk.Style(self).lookup("Workbench.TFrame", "background") or "#05090e"
        label_fg = ttk.Style(self).lookup("Workbench.TLabel", "foreground") or "#f7fbff"
        self._chart_canvas_host = tk.Frame(chart_tab, bg=chart_bg, highlightthickness=0, bd=0)
        self._chart_canvas_host.grid(row=0, column=0, sticky="nsew")
        chart_tab.configure(style="Workbench.TFrame")
        self._chart_placeholder = tk.Label(
            self._chart_canvas_host,
            text="請先執行回測；K 線圖會直接顯示在此。",
            anchor="center",
            justify="center",
            bg=chart_bg,
            fg=label_fg,
            font=("Microsoft JhengHei", 12),
        )
        self._chart_placeholder.pack(fill="both", expand=True)

        sidebar = ttk.Frame(chart_tab, padding=(10, 12), style="Workbench.TFrame")
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.configure(width=320)
        self._signal_chip = tk.Label(sidebar, textvariable=self._sidebar_signal_var, bg="#2090ff", fg="#ffffff", font=("Microsoft JhengHei", 20, "bold"), padx=10, pady=8, anchor="center")
        self._signal_chip.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._history_chip = tk.Label(sidebar, textvariable=self._sidebar_history_var, bg="#ff8a1c", fg="#ffffff", font=("Microsoft JhengHei", 20, "bold"), padx=10, pady=8, anchor="center")
        self._history_chip.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(sidebar, text="歷史績效表", style="Workbench.SidebarHeader.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._sidebar_summary_var, style="Workbench.SidebarSummary.TLabel", justify="left", anchor="nw", wraplength=280).grid(row=3, column=0, sticky="ew", pady=(4, 14))
        ttk.Label(sidebar, text="選取日線值", style="Workbench.SidebarHeader.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_date_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=5, column=0, sticky="w", pady=(4, 0))
        ttk.Label(sidebar, textvariable=self._selected_tp_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=6, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_limit_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=7, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_entry_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=8, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_stop_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=9, column=0, sticky="w", pady=(0, 12))
        ttk.Button(sidebar, text="回到最新K線", command=self._move_chart_to_latest, style="Workbench.TButton").grid(row=10, column=0, sticky="ew")
        sidebar.rowconfigure(11, weight=1)

        summary_tab = ttk.Frame(notebook, padding=10, style="Workbench.TFrame")
        notebook.add(summary_tab, text="執行摘要")
        for row_idx, (key, label) in enumerate(SUMMARY_FIELDS):
            ttk.Label(summary_tab, text=label, style="Workbench.TLabel").grid(row=row_idx, column=0, sticky="nw", pady=3)
            ttk.Label(summary_tab, textvariable=self._summary_vars[key], wraplength=860, justify="left", style="Workbench.TLabel").grid(row=row_idx, column=1, sticky="nw", padx=(8, 0), pady=3)
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

        console_tab = ttk.Frame(notebook, padding=10, style="Workbench.TFrame")
        console_tab.rowconfigure(0, weight=1)
        console_tab.columnconfigure(0, weight=1)
        notebook.add(console_tab, text="Console")
        self._console_text = tk.Text(console_tab, wrap="word", bg="#040a12", fg="#f7fbff", insertbackground="#f7fbff", relief="flat", bd=0, font=("Consolas", 10))
        self._console_text.grid(row=0, column=0, sticky="nsew")
        console_scroll = ttk.Scrollbar(console_tab, orient="vertical", command=self._console_text.yview, style="Workbench.Vertical.TScrollbar")
        console_scroll.grid(row=0, column=1, sticky="ns")
        self._console_text.configure(yscrollcommand=console_scroll.set)

        footer = ttk.Frame(self, style="Workbench.TFrame")
        footer.pack(fill="x", pady=(4, 0))
        ttk.Label(footer, textvariable=self._status_var, style="Workbench.TLabel").pack(anchor="w")
        ttk.Label(footer, textvariable=self._chart_hint_var, wraplength=1500, justify="left", style="Workbench.TLabel").pack(anchor="w", pady=(2, 0))

        self._notebook.select(chart_tab)

    def _append_console_text(self, text):
        self._console_text.insert("end", text)
        self._console_text.see("end")
        self.update_idletasks()

    def _clear_console(self):
        self._console_text.delete("1.0", "end")

    def _on_candidate_selected(self, _event=None):
        selected = self._candidate_display_var.get().strip()
        ticker = self._candidate_map.get(selected)
        if ticker:
            self._ticker_var.set(ticker)
            self.after_idle(self._run_analysis)

    def _on_ticker_enter(self, _event=None):
        self._run_analysis()

    def _run_scanner(self):
        self._clear_console()
        self._notebook.select(3)
        self._status_var.set("執行中：掃描候選股")
        self.update_idletasks()
        try:
            data_dir = resolve_debug_data_dir(DEFAULT_DATASET_PROFILE)
            params = load_params(verbose=False)
            with redirect_stdout(self._console_writer), redirect_stderr(self._console_writer):
                scan_result = run_daily_scanner(data_dir, params)
        except Exception as exc:
            self._status_var.set(f"掃描失敗：{type(exc).__name__}: {exc}")
            messagebox.showerror("股票工具工作台", str(exc))
            return

        candidate_rows = list((scan_result or {}).get("candidate_rows") or [])
        candidate_rows.sort(key=lambda item: (item.get("sort_value") or 0.0, item.get("ticker") or ""), reverse=True)
        display_values = []
        self._candidate_map = {}
        for item in candidate_rows:
            label = f"{item.get('ticker', '')} | {'新訊號' if item.get('kind') == 'buy' else '延續候選'}"
            display_values.append(label)
            self._candidate_map[label] = item.get("ticker")
        self._candidate_combo.configure(values=display_values)
        if display_values:
            self._candidate_display_var.set(display_values[0])
            self._ticker_var.set(self._candidate_map[display_values[0]])
        self._status_var.set(f"掃描完成：候選股 {len(display_values)} 檔")

    def _run_analysis(self):
        ticker = self._ticker_var.get().strip()
        if not ticker:
            messagebox.showerror("股票工具工作台", "請先輸入股票代號。")
            return

        self._status_var.set(f"執行中：{ticker} / {get_dataset_profile_label(DEFAULT_DATASET_PROFILE)}")
        self._chart_hint_var.set("載入 K 線圖中…")
        self.update_idletasks()

        try:
            result = run_debug_ticker_analysis(
                ticker,
                dataset_profile_key=DEFAULT_DATASET_PROFILE,
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
        self._status_var.set(f"完成：{ticker} / {get_dataset_profile_label(DEFAULT_DATASET_PROFILE)}")

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

        self._update_sidebar_from_result(result)
        self._render_trade_table(trade_logs_df)
        self._render_embedded_chart(result)

    def _format_sidebar_line_value(self, label, value):
        return f"{label}：-" if value is None or pd.isna(value) else f"{label}：{float(value):.2f}"

    def _update_selected_value_sidebar(self, snapshot):
        if not snapshot:
            self._selected_date_var.set("選取日：-")
            self._selected_tp_var.set("停利：-")
            self._selected_limit_var.set("限價：-")
            self._selected_entry_var.set("成交：-")
            self._selected_stop_var.set("停損：-")
            return
        self._selected_date_var.set(f"選取日：{snapshot.get('date_label', '-')}")
        self._selected_tp_var.set(self._format_sidebar_line_value("停利", snapshot.get("tp_price")))
        self._selected_limit_var.set(self._format_sidebar_line_value("限價", snapshot.get("limit_price")))
        self._selected_entry_var.set(self._format_sidebar_line_value("成交", snapshot.get("entry_price")))
        self._selected_stop_var.set(self._format_sidebar_line_value("停損", snapshot.get("stop_price")))

    def _update_sidebar_from_result(self, result):
        chart_payload = dict(result.get("chart_payload") or {})
        status_lines = list(((chart_payload.get("status_box") or {}).get("lines") or []))
        signal_text = next((line for line in status_lines if "買" in str(line) or "賣" in str(line) or "候選" in str(line)), "無買入訊號")
        history_text = next((line for line in status_lines if "歷史績效" in str(line) or "歷績門檻" in str(line)), "歷史績效未達")
        self._sidebar_signal_var.set(signal_text)
        self._sidebar_history_var.set(history_text)
        self._sidebar_summary_var.set("\n".join(str(line) for line in (chart_payload.get("summary_box") or []) if str(line).strip()) or "-")
        dates = chart_payload.get("date_labels") or []
        if dates:
            idx = int((chart_payload.get("default_view") or {}).get("end_idx", len(dates) - 1))
            idx = max(0, min(idx, len(dates) - 1))
            snapshot = {
                "date_label": dates[idx],
                "tp_price": chart_payload.get("tp_line", [None])[idx] if len(chart_payload.get("tp_line", [])) > idx else None,
                "limit_price": chart_payload.get("limit_line", [None])[idx] if len(chart_payload.get("limit_line", [])) > idx else None,
                "entry_price": chart_payload.get("entry_line", [None])[idx] if len(chart_payload.get("entry_line", [])) > idx else None,
                "stop_price": chart_payload.get("stop_line", [None])[idx] if len(chart_payload.get("stop_line", [])) > idx else None,
            }
            self._update_selected_value_sidebar(snapshot)
        else:
            self._update_selected_value_sidebar(None)

    def _build_gui_chart_payload(self, result):
        chart_payload = dict(result.get("chart_payload") or {})
        chart_payload["summary_box"] = []
        chart_payload["status_box"] = {}
        return chart_payload

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

        normalized_df = trade_logs_df.copy().where(pd.notna(trade_logs_df), "")
        for row in normalized_df.to_dict("records"):
            self._tree.insert("", "end", values=[row.get(column, "") for column in columns])

    def _rerender_current_chart(self):
        if self._result is not None:
            self._render_embedded_chart(self._result)

    def _move_chart_to_latest(self):
        if self._chart_figure is None:
            return
        scroll_chart_to_latest(self._chart_figure, redraw=True)

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
            figure = create_matplotlib_debug_chart_figure(chart_payload=self._build_gui_chart_payload(result), ticker=ticker, show_volume=bool(self._show_volume_var.get()))
        except Exception as exc:
            self._clear_embedded_chart()
            self._chart_hint_var.set(f"K 線圖建立失敗：{type(exc).__name__}: {exc}")
            return

        self._clear_embedded_chart()
        self._chart_placeholder.pack_forget()
        canvas = FigureCanvasTkAgg(figure, master=self._chart_canvas_host)
        bind_matplotlib_chart_navigation(figure, canvas)
        state = getattr(figure, "_stock_chart_navigation_state", None)
        if isinstance(state, dict):
            state["external_hover_callback"] = self._update_selected_value_sidebar
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.configure(background="#02050a", highlightthickness=0, bd=0, takefocus=1)
        canvas_widget.pack(fill="both", expand=True)
        canvas_widget.focus_set()

        self._chart_canvas = canvas
        self._chart_figure = figure
        self._notebook.select(0)
        self._move_chart_to_latest()
        self._chart_hint_var.set("K 線圖已內嵌於 GUI；預設顯示最近 18 個月，可滑動至完整歷史，右側可檢視狀態、績效與選取日線值。")

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
            try:
                export_result = run_debug_ticker_analysis(
                    ticker,
                    dataset_profile_key=DEFAULT_DATASET_PROFILE,
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
