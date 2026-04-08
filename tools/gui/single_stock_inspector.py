from __future__ import annotations

import io
import os
import subprocess
import sys
import tkinter as tk
from contextlib import redirect_stderr, redirect_stdout
from tkinter import messagebox, ttk

import pandas as pd

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_profile_label
from tools.debug.charting import bind_matplotlib_chart_navigation, build_chart_hover_snapshot, create_matplotlib_debug_chart_figure, scroll_chart_to_latest
from tools.debug.trade_log import load_params, resolve_debug_data_dir, run_debug_ticker_analysis
from tools.scanner.scan_runner import run_daily_scanner

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError as exc:  # pragma: no cover - GUI runtime fallback
    FigureCanvasTkAgg = None
    FIGURE_CANVAS_TKAGG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    FIGURE_CANVAS_TKAGG_IMPORT_ERROR = ""


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


SIDEBAR_SIGNAL_CHIP_TEXT = "出現買入訊號"
SIDEBAR_HISTORY_CHIP_TEXT = "符合歷史績效"
SIDEBAR_CHIP_ACTIVE_BG = "#2090ff"
SIDEBAR_HISTORY_CHIP_ACTIVE_BG = "#ff8a1c"
SIDEBAR_CHIP_INACTIVE_BG = "#04070c"


class SingleStockBacktestInspectorPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=4, style="Workbench.TFrame")
        self._result = None
        self._status_var = tk.StringVar(value="尚未執行")
        self._ticker_var = tk.StringVar()
        self._show_volume_var = tk.BooleanVar(value=False)
        self._candidate_display_var = tk.StringVar()
        self._candidate_map = {}
        self._columns = []
        self._chart_canvas = None
        self._chart_figure = None
        self._console_writer = _ConsoleWriter(self)
        self._sidebar_signal_var = tk.StringVar(value=SIDEBAR_SIGNAL_CHIP_TEXT)
        self._sidebar_history_var = tk.StringVar(value=SIDEBAR_HISTORY_CHIP_TEXT)
        self._sidebar_summary_var = tk.StringVar(value="-")
        self._selected_date_var = tk.StringVar(value="選取日: -")
        self._selected_open_var = tk.StringVar(value="開: -")
        self._selected_high_var = tk.StringVar(value="高: -")
        self._selected_low_var = tk.StringVar(value="低: -")
        self._selected_close_var = tk.StringVar(value="收: -")
        self._selected_volume_var = tk.StringVar(value="量: -")
        self._selected_signal_capital_var = tk.StringVar(value="資金: -")
        self._selected_signal_qty_var = tk.StringVar(value="股數: -")
        self._selected_limit_var = tk.StringVar(value="限價: -")
        self._selected_reserved_var = tk.StringVar(value="預留: -")
        self._selected_tp_var = tk.StringVar(value="停利: -")
        self._selected_entry_var = tk.StringVar(value="成交: -")
        self._selected_stop_var = tk.StringVar(value="停損: -")
        self._selected_spend_var = tk.StringVar(value="實支: -")
        self._build_ui()

    def destroy(self):
        self._clear_embedded_chart()
        super().destroy()

    def _build_ui(self):
        controls = ttk.Frame(self, padding=(8, 2, 8, 2), style="Workbench.TFrame")
        controls.pack(fill="x", pady=(0, 4))
        controls.columnconfigure(10, weight=1)

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
        ttk.Button(controls, text="開啟 Excel", command=self._open_excel, style="Workbench.TButton").grid(row=0, column=5, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟輸出資料夾", command=self._open_output_dir, style="Workbench.TButton").grid(row=0, column=6, sticky="w")

        notebook = ttk.Notebook(self, style="Workbench.TNotebook")
        notebook.pack(fill="both", expand=True)
        self._notebook = notebook

        chart_tab = ttk.Frame(notebook, padding=0, style="Workbench.TFrame")
        chart_tab.rowconfigure(0, weight=1)
        chart_tab.columnconfigure(0, weight=1)
        chart_tab.columnconfigure(1, weight=0)
        notebook.add(chart_tab, text="K 線圖")

        chart_bg = "#000000"
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

        sidebar = ttk.Frame(chart_tab, padding=(10, 6), style="Workbench.TFrame")
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.configure(width=320)
        sidebar.columnconfigure(1, weight=1)
        self._signal_chip = tk.Label(sidebar, textvariable=self._sidebar_signal_var, bg="#04070c", fg="#ffffff", font=("Microsoft JhengHei", 19, "bold"), padx=8, pady=7, anchor="center")
        self._signal_chip.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._history_chip = tk.Label(sidebar, textvariable=self._sidebar_history_var, bg="#04070c", fg="#ffffff", font=("Microsoft JhengHei", 19, "bold"), padx=8, pady=7, anchor="center")
        self._history_chip.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ttk.Label(sidebar, text="歷史績效表", style="Workbench.SidebarHeader.TLabel").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(sidebar, textvariable=self._sidebar_summary_var, style="Workbench.SidebarSummary.TLabel", justify="left", anchor="nw", wraplength=270).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 14))
        ttk.Label(sidebar, text="選取日線值", style="Workbench.SidebarHeader.TLabel").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_date_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(sidebar, textvariable=self._selected_open_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_high_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=7, column=0, columnspan=2, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_low_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_close_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=9, column=0, columnspan=2, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_volume_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=10, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(sidebar, text="買訊資訊", style="Workbench.SidebarHeader.TLabel").grid(row=11, column=0, columnspan=2, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_signal_capital_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=12, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(sidebar, textvariable=self._selected_signal_qty_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=13, column=0, columnspan=2, sticky="w")
        self._limit_icon = tk.Label(sidebar, text="┅┅", bg="#05090e", fg="#4f86ff", font=("Microsoft JhengHei", 13, "bold"))
        self._limit_icon.grid(row=14, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_limit_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=14, column=1, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_reserved_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=15, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(sidebar, text="買入資訊", style="Workbench.SidebarHeader.TLabel").grid(row=16, column=0, columnspan=2, sticky="w")
        self._tp_icon = tk.Label(sidebar, text="━━", bg="#05090e", fg="#facc15", font=("Microsoft JhengHei", 13, "bold"))
        self._tp_icon.grid(row=17, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_tp_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=17, column=1, sticky="w", pady=(4, 0))
        self._entry_icon = tk.Label(sidebar, text="━━", bg="#05090e", fg="#2f6df6", font=("Microsoft JhengHei", 13, "bold"))
        self._entry_icon.grid(row=18, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_entry_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=18, column=1, sticky="w")
        self._stop_icon = tk.Label(sidebar, text="━━", bg="#05090e", fg="#ff4d4f", font=("Microsoft JhengHei", 13, "bold"))
        self._stop_icon.grid(row=19, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_stop_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=19, column=1, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_spend_var, style="Workbench.SidebarValue.TLabel", justify="left").grid(row=20, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Button(sidebar, text="回到最新K線", command=self._move_chart_to_latest, style="Workbench.TButton").grid(row=21, column=0, columnspan=2, sticky="ew")
        sidebar.rowconfigure(22, weight=1)

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
        footer.pack(fill="x", pady=(2, 0))
        ttk.Label(footer, textvariable=self._status_var, style="Workbench.TLabel").pack(anchor="w")

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
        self._notebook.select(2)
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
            messagebox.showerror("股票工具工作台", str(exc))
            return

        self._result = result
        self._render_result(result)
        self._status_var.set(f"完成：{ticker} / {get_dataset_profile_label(DEFAULT_DATASET_PROFILE)}")

    def _render_result(self, result):
        trade_logs_df = result.get("trade_logs_df")
        self._update_sidebar_from_result(result)
        self._render_trade_table(trade_logs_df)
        self._render_embedded_chart(result)

    def _format_sidebar_line_value(self, label, value):
        return f"{label}: -" if value is None or pd.isna(value) else f"{label}: {float(value):.2f}"

    def _format_sidebar_amount_value(self, label, value):
        return f"{label}: -" if value is None or pd.isna(value) else f"{label}: {float(value):,.0f}"

    def _format_sidebar_qty_value(self, label, value):
        return f"{label}: -" if value is None or pd.isna(value) else f"{label}: {int(value):,}"

    def _format_sidebar_ohlcv_value(self, label, value, *, volume=False):
        if value is None or pd.isna(value):
            return f"{label}: -"
        if volume:
            return f"{label}: {float(value) / 1_000_000:.2f}M"
        return f"{label}: {float(value):.2f}"

    def _update_selected_value_sidebar(self, snapshot):
        if not snapshot:
            self._selected_date_var.set("選取日: -")
            self._selected_open_var.set("開: -")
            self._selected_high_var.set("高: -")
            self._selected_low_var.set("低: -")
            self._selected_close_var.set("收: -")
            self._selected_volume_var.set("量: -")
            self._selected_signal_capital_var.set("資金: -")
            self._selected_signal_qty_var.set("股數: -")
            self._selected_limit_var.set("限價: -")
            self._selected_reserved_var.set("預留: -")
            self._selected_tp_var.set("停利: -")
            self._selected_entry_var.set("成交: -")
            self._selected_stop_var.set("停損: -")
            self._selected_spend_var.set("實支: -")
            return
        self._selected_date_var.set(f"選取日: {snapshot.get('date_label', '-')}")
        self._selected_open_var.set(self._format_sidebar_ohlcv_value("開", snapshot.get("open")))
        self._selected_high_var.set(self._format_sidebar_ohlcv_value("高", snapshot.get("high")))
        self._selected_low_var.set(self._format_sidebar_ohlcv_value("低", snapshot.get("low")))
        self._selected_close_var.set(self._format_sidebar_ohlcv_value("收", snapshot.get("close")))
        self._selected_volume_var.set(self._format_sidebar_ohlcv_value("量", snapshot.get("volume"), volume=True))
        self._selected_signal_capital_var.set(self._format_sidebar_amount_value("資金", snapshot.get("signal_capital")))
        self._selected_signal_qty_var.set(self._format_sidebar_qty_value("股數", snapshot.get("signal_qty")))
        self._selected_limit_var.set(self._format_sidebar_line_value("限價", snapshot.get("limit_price")))
        self._selected_reserved_var.set(self._format_sidebar_amount_value("預留", snapshot.get("reserved_capital")))
        self._selected_tp_var.set(self._format_sidebar_line_value("停利", snapshot.get("tp_price")))
        self._selected_entry_var.set(self._format_sidebar_line_value("成交", snapshot.get("entry_price")))
        self._selected_stop_var.set(self._format_sidebar_line_value("停損", snapshot.get("stop_price")))
        self._selected_spend_var.set(self._format_sidebar_amount_value("實支", snapshot.get("buy_capital")))

    def _apply_sidebar_chip_styles(self, signal_active, history_active):
        self._signal_chip.configure(bg=SIDEBAR_CHIP_ACTIVE_BG if bool(signal_active) else SIDEBAR_CHIP_INACTIVE_BG)
        self._history_chip.configure(bg=SIDEBAR_HISTORY_CHIP_ACTIVE_BG if bool(history_active) else SIDEBAR_CHIP_INACTIVE_BG)

    @staticmethod
    def _resolve_sidebar_chip_states(status_lines):
        normalized_lines = [str(line).strip() for line in status_lines if str(line).strip()]
        signal_active = any(line == SIDEBAR_SIGNAL_CHIP_TEXT for line in normalized_lines)
        history_active = any(line in {SIDEBAR_HISTORY_CHIP_TEXT, "歷史績效符合", "歷績門檻符合"} for line in normalized_lines)
        return signal_active, history_active

    @staticmethod
    def _resolve_chart_payload_value(chart_payload, key, idx):
        values = chart_payload.get(key, [])
        if values is None or len(values) <= idx:
            return None
        return values[idx]

    def _update_sidebar_from_result(self, result):
        chart_payload = dict(result.get("chart_payload") or {})
        status_lines = list(((chart_payload.get("status_box") or {}).get("lines") or []))
        signal_active, history_active = self._resolve_sidebar_chip_states(status_lines)
        self._sidebar_signal_var.set(SIDEBAR_SIGNAL_CHIP_TEXT)
        self._sidebar_history_var.set(SIDEBAR_HISTORY_CHIP_TEXT)
        self._sidebar_summary_var.set("\n".join(str(line) for line in (chart_payload.get("summary_box") or []) if str(line).strip()) or "-")
        self._apply_sidebar_chip_styles(signal_active, history_active)
        dates = chart_payload.get("date_labels") or []
        if dates:
            idx = int((chart_payload.get("default_view") or {}).get("end_idx", len(dates) - 1))
            idx = max(0, min(idx, len(dates) - 1))
            snapshot = build_chart_hover_snapshot(chart_payload, idx)
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
            return
        if FigureCanvasTkAgg is None:
            self._clear_embedded_chart()
            backend_error_text = "缺少 matplotlib TkAgg backend，無法內嵌圖表。"
            if FIGURE_CANVAS_TKAGG_IMPORT_ERROR:
                backend_error_text = f"{backend_error_text} {FIGURE_CANVAS_TKAGG_IMPORT_ERROR}"
            self._status_var.set(backend_error_text)
            return
        try:
            figure = create_matplotlib_debug_chart_figure(chart_payload=self._build_gui_chart_payload(result), ticker=ticker, show_volume=bool(self._show_volume_var.get()))
        except Exception as exc:
            self._clear_embedded_chart()
            error_text = f"圖表渲染失敗：{type(exc).__name__}: {exc}"
            self._status_var.set(error_text)
            messagebox.showerror("股票工具工作台", error_text)
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

    def _open_excel(self):
        self._open_result_path("excel_path")

    def _open_output_dir(self):
        if self._result is None:
            messagebox.showinfo("股票工具工作台", "尚未產生輸出。")
            return
        output_dir = None
        for key in ("excel_path",):
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
            subprocess.Popen(["xdg-open", normalized_path])
        except Exception as exc:
            messagebox.showerror("股票工具工作台", f"開啟失敗：{type(exc).__name__}: {exc}")
