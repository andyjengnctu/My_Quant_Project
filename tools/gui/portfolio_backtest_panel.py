from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label
from tools.portfolio_sim.reporting import DASHBOARD_HTML_PATH, OUTPUT_DIR, REPORT_XLSX_PATH, export_portfolio_reports, print_yearly_return_report
from tools.portfolio_sim.runtime import BEST_PARAMS_PATH, ensure_runtime_dirs, load_strict_params, run_portfolio_simulation

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:  # pragma: no cover - GUI runtime fallback
    FigureCanvasTkAgg = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ThreadSafeConsoleWriter(io.TextIOBase):
    def __init__(self, panel: "PortfolioBacktestPanel"):
        super().__init__()
        self._panel = panel

    def write(self, text):
        text = str(text or "")
        if not text:
            return 0
        self._panel.after(0, self._panel._append_console, text)
        return len(text)

    def flush(self):
        return None


class PortfolioBacktestPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=6, style="Workbench.TFrame")
        self._status_var = tk.StringVar(value="尚未執行投組回測")
        self._max_pos_var = tk.StringVar(value="10")
        self._start_year_var = tk.StringVar(value="2015")
        self._benchmark_var = tk.StringVar(value="0050")
        self._rotation_var = tk.BooleanVar(value=False)
        self._summary_var = tk.StringVar(value="尚未執行投組回測")
        self._dataset_var = tk.StringVar(value=f"資料集：{get_dataset_profile_label(DEFAULT_DATASET_PROFILE)}")
        self._worker = None
        self._chart_canvas = None
        self._chart_figure = None
        self._console_writer = _ThreadSafeConsoleWriter(self)
        self._build_ui()

    def destroy(self):
        self._clear_embedded_chart()
        super().destroy()

    def _build_ui(self):
        controls = ttk.Frame(self, padding=(4, 4, 4, 6), style="Workbench.TFrame")
        controls.pack(fill="x", pady=(0, 4))
        for idx in range(11):
            controls.columnconfigure(idx, weight=0)
        controls.columnconfigure(10, weight=1)

        ttk.Label(controls, text="最大持倉", style="Workbench.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self._max_pos_var, width=10, style="Workbench.TEntry").grid(row=0, column=1, padx=(6, 12), sticky="w")
        ttk.Label(controls, text="開始年份", style="Workbench.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self._start_year_var, width=10, style="Workbench.TEntry").grid(row=0, column=3, padx=(6, 12), sticky="w")
        ttk.Label(controls, text="大盤標的", style="Workbench.TLabel").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self._benchmark_var, width=10, style="Workbench.TEntry").grid(row=0, column=5, padx=(6, 12), sticky="w")
        ttk.Checkbutton(controls, text="啟用汰弱換股", variable=self._rotation_var, style="Workbench.TCheckbutton").grid(row=0, column=6, padx=(0, 12), sticky="w")
        ttk.Button(controls, text="執行投組回測", command=self._run_portfolio_sim, style="Workbench.TButton").grid(row=0, column=7, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟 Excel", command=lambda: self._open_path(REPORT_XLSX_PATH), style="Workbench.TButton").grid(row=0, column=8, padx=(0, 8), sticky="w")
        ttk.Button(controls, text="開啟輸出資料夾", command=lambda: self._open_path(OUTPUT_DIR), style="Workbench.TButton").grid(row=0, column=9, sticky="w")

        progress_row = ttk.Frame(self, padding=(4, 0, 4, 6), style="Workbench.TFrame")
        progress_row.pack(fill="x")
        progress_row.columnconfigure(1, weight=1)
        ttk.Label(progress_row, textvariable=self._dataset_var, style="Workbench.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 12))
        self._progress = ttk.Progressbar(progress_row, mode="indeterminate", style="Workbench.Horizontal.TProgressbar")
        self._progress.grid(row=0, column=1, sticky="ew")

        main_pane = ttk.Panedwindow(self, orient="horizontal")
        main_pane.pack(fill="both", expand=True)

        chart_section = ttk.Frame(main_pane, style="Workbench.TFrame")
        chart_section.rowconfigure(0, weight=1)
        chart_section.columnconfigure(0, weight=1)
        main_pane.add(chart_section, weight=5)

        chart_host = tk.Frame(chart_section, bg="#05090e", highlightthickness=0, bd=0)
        chart_host.grid(row=0, column=0, sticky="nsew")
        self._chart_host = chart_host
        self._chart_placeholder = tk.Label(chart_host, text="執行投組回測後，這裡會顯示淨值曲線與大盤對照。", bg="#05090e", fg="#f7fbff", font=("Microsoft JhengHei", 13), anchor="center", justify="center")
        self._chart_placeholder.pack(fill="both", expand=True)

        sidebar = ttk.Frame(main_pane, padding=(10, 10), style="Workbench.TFrame")
        sidebar.configure(width=340)
        main_pane.add(sidebar, weight=2)
        sidebar.columnconfigure(0, weight=1)
        ttk.Label(sidebar, text="投組回測摘要", style="Workbench.SidebarHeader.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._summary_var, style="Workbench.SidebarSummary.TLabel", justify="left", anchor="nw", wraplength=310).grid(row=1, column=0, sticky="nsew", pady=(4, 10))
        ttk.Button(sidebar, text="開啟投組儀表板", command=lambda: self._open_path(DASHBOARD_HTML_PATH), style="Workbench.TButton").grid(row=2, column=0, sticky="ew")
        sidebar.rowconfigure(3, weight=1)

        notebook = ttk.Notebook(self, style="Workbench.TNotebook")
        notebook.pack(fill="both", expand=True, pady=(6, 0))
        self._notebook = notebook

        trades_tab = ttk.Frame(notebook, padding=8, style="Workbench.TFrame")
        trades_tab.rowconfigure(0, weight=1)
        trades_tab.columnconfigure(0, weight=1)
        notebook.add(trades_tab, text="交易明細")
        self._trade_tree = ttk.Treeview(trades_tab, show="headings", style="Workbench.Treeview")
        self._trade_tree.grid(row=0, column=0, sticky="nsew")
        trade_y = ttk.Scrollbar(trades_tab, orient="vertical", command=self._trade_tree.yview, style="Workbench.Vertical.TScrollbar")
        trade_y.grid(row=0, column=1, sticky="ns")
        trade_x = ttk.Scrollbar(trades_tab, orient="horizontal", command=self._trade_tree.xview, style="Workbench.Horizontal.TScrollbar")
        trade_x.grid(row=1, column=0, sticky="ew")
        self._trade_tree.configure(yscrollcommand=trade_y.set, xscrollcommand=trade_x.set)

        yearly_tab = ttk.Frame(notebook, padding=8, style="Workbench.TFrame")
        yearly_tab.rowconfigure(0, weight=1)
        yearly_tab.columnconfigure(0, weight=1)
        notebook.add(yearly_tab, text="年度報酬")
        self._yearly_tree = ttk.Treeview(yearly_tab, show="headings", style="Workbench.Treeview")
        self._yearly_tree.grid(row=0, column=0, sticky="nsew")
        yearly_y = ttk.Scrollbar(yearly_tab, orient="vertical", command=self._yearly_tree.yview, style="Workbench.Vertical.TScrollbar")
        yearly_y.grid(row=0, column=1, sticky="ns")
        yearly_x = ttk.Scrollbar(yearly_tab, orient="horizontal", command=self._yearly_tree.xview, style="Workbench.Horizontal.TScrollbar")
        yearly_x.grid(row=1, column=0, sticky="ew")
        self._yearly_tree.configure(yscrollcommand=yearly_y.set, xscrollcommand=yearly_x.set)

        console_tab = ttk.Frame(notebook, padding=8, style="Workbench.TFrame")
        console_tab.rowconfigure(0, weight=1)
        console_tab.columnconfigure(0, weight=1)
        notebook.add(console_tab, text="Console")
        self._console = tk.Text(console_tab, wrap="word", bg="#040a12", fg="#f7fbff", insertbackground="#f7fbff", relief="flat", bd=0, font=("Consolas", 10))
        self._console.grid(row=0, column=0, sticky="nsew")
        console_y = ttk.Scrollbar(console_tab, orient="vertical", command=self._console.yview, style="Workbench.Vertical.TScrollbar")
        console_y.grid(row=0, column=1, sticky="ns")
        self._console.configure(yscrollcommand=console_y.set)

        footer = ttk.Frame(self, style="Workbench.TFrame")
        footer.pack(fill="x", pady=(4, 0))
        ttk.Label(footer, textvariable=self._status_var, style="Workbench.TLabel").pack(anchor="w")

    def _append_console(self, text):
        self._console.insert("end", text)
        self._console.see("end")
        self.update_idletasks()

    def _set_status(self, text):
        self._status_var.set(text)
        self.update_idletasks()

    def _set_running(self, running, status_text):
        if running:
            self._progress.start(10)
        else:
            self._progress.stop()
        self._set_status(status_text)

    def _render_table(self, tree, frame: pd.DataFrame | None):
        tree.delete(*tree.get_children())
        if frame is None or frame.empty:
            tree.configure(columns=("message",))
            tree.heading("message", text="訊息")
            tree.column("message", width=960, anchor="w")
            tree.insert("", "end", values=("目前沒有資料。",))
            return
        columns = list(frame.columns)
        tree.configure(columns=columns)
        normalized = frame.copy().where(pd.notna(frame), "")
        for column in columns:
            tree.heading(column, text=str(column))
            width = 140 if str(column) not in {"Date", "日期", "start_date", "end_date", "動作", "備註"} else (160 if str(column) in {"Date", "日期", "start_date", "end_date"} else 260)
            tree.column(column, width=width, anchor="center")
        for row in normalized.to_dict("records"):
            tree.insert("", "end", values=[row.get(column, "") for column in columns])

    def _build_portfolio_figure(self, df_eq, benchmark_ticker):
        if FigureCanvasTkAgg is None:
            raise RuntimeError("環境缺少 matplotlib Tk backend，無法在 GUI 內嵌顯示投組淨值曲線。")
        from matplotlib.figure import Figure

        figure = Figure(figsize=(12.4, 5.9), dpi=96, facecolor="#02050a")
        axis = figure.add_subplot(1, 1, 1)
        axis.set_facecolor("#02050a")
        axis.grid(True, color="#14304a", alpha=0.10, linewidth=0.68)
        axis.tick_params(colors="#fbfdff", labelsize=10)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#14304a")
        axis.spines["bottom"].set_color("#14304a")
        axis.plot(df_eq["Date"], df_eq["Strategy_Return_Pct"], color="#ff5b6e", linewidth=2.2, label="投組報酬 (%)")
        bm_col = f"Benchmark_{benchmark_ticker}_Pct"
        if bm_col in df_eq.columns:
            axis.plot(df_eq["Date"], df_eq[bm_col], color="#4f86ff", linewidth=1.6, label=f"{benchmark_ticker} (%)")
        legend = axis.legend(loc="upper left", frameon=False)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#fbfdff")
        figure.subplots_adjust(left=0.055, right=0.99, top=0.97, bottom=0.11)
        return figure

    def _render_embedded_chart(self, figure):
        self._clear_embedded_chart()
        self._chart_placeholder.pack_forget()
        canvas = FigureCanvasTkAgg(figure, master=self._chart_host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(background="#02050a", highlightthickness=0, bd=0)
        widget.pack(fill="both", expand=True)
        self._chart_canvas = canvas
        self._chart_figure = figure

    def _clear_embedded_chart(self):
        if self._chart_canvas is not None:
            self._chart_canvas.get_tk_widget().destroy()
            self._chart_canvas = None
        if self._chart_figure is not None:
            self._chart_figure.clear()
            self._chart_figure = None
        if hasattr(self, "_chart_placeholder") and not self._chart_placeholder.winfo_ismapped():
            self._chart_placeholder.pack(fill="both", expand=True)

    def _render_result(self, *, result_tuple, benchmark_ticker, df_yearly):
        (
            df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff,
            final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed,
            total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate,
            normal_trade_count, extended_trade_count, annual_trades,
            reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, pf_profile,
        ) = result_tuple
        summary_lines = [
            f"總報酬: {tot_ret:.2f}%",
            f"年化報酬: {annual_return_pct:.2f}%",
            f"最大回撤: {mdd:.2f}%",
            f"勝率: {win_rate:.2f}%",
            f"風報比: {pf_payoff:.2f}",
            f"EV: {pf_ev:.2f} R",
            f"交易次數: {int(trade_count)}",
            f"正常交易: {int(normal_trade_count)} / 延續交易: {int(extended_trade_count)}",
            f"買入成交率: {reserved_buy_fill_rate:.2f}%",
            f"平均曝險: {avg_exp:.2f}% | 最大曝險: {max_exp:.2f}%",
            f"R²: {r_sq:.4f} | 月勝率: {m_win_rate:.2f}%",
            f"大盤報酬: {bm_ret:.2f}% | 大盤 MDD: {bm_mdd:.2f}%",
            f"大盤年化: {bm_annual_return_pct:.2f}% | 大盤 R²: {bm_r_sq:.4f}",
            f"錯失買點: {int(total_missed)} | 錯失賣出: {int(total_missed_sells)}",
            f"完整年度數: {int(pf_profile.get('full_year_count', 0) or 0)} | 年均交易: {annual_trades:.2f}",
            f"最終權益: {final_eq:,.0f}",
        ]
        self._summary_var.set("\n".join(summary_lines))
        self._render_table(self._trade_tree, df_tr)
        self._render_table(self._yearly_tree, df_yearly)
        self._render_embedded_chart(self._build_portfolio_figure(df_eq, benchmark_ticker))
        self._notebook.select(0)

    def _run_portfolio_sim(self):
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("股票工具工作台", "投組回測執行中，請稍候。")
            return
        self._console.delete("1.0", "end")
        self._summary_var.set("執行中…")
        self._set_running(True, "執行中：投組回測")
        benchmark_ticker = self._benchmark_var.get().strip() or "0050"
        max_positions = int(self._max_pos_var.get().strip() or "10")
        start_year = int(self._start_year_var.get().strip() or "2015")
        enable_rotation = bool(self._rotation_var.get())
        data_dir = get_dataset_dir(str(PROJECT_ROOT), DEFAULT_DATASET_PROFILE)

        def _worker():
            try:
                self.after(0, self._dataset_var.set, f"資料集：{get_dataset_profile_label(DEFAULT_DATASET_PROFILE)} | 路徑：{data_dir}")
                self.after(0, self._append_console, f"資料集：{data_dir}\n")
                with redirect_stdout(self._console_writer), redirect_stderr(self._console_writer):
                    ensure_runtime_dirs()
                    params = load_strict_params(BEST_PARAMS_PATH)
                    self.after(0, self._set_status, "執行中：載入市場資料")
                    result_tuple = run_portfolio_simulation(
                        data_dir,
                        params,
                        max_positions=max_positions,
                        enable_rotation=enable_rotation,
                        start_year=start_year,
                        benchmark_ticker=benchmark_ticker,
                        verbose=True,
                    )
                    pf_profile = result_tuple[-1]
                    df_yearly = print_yearly_return_report(pf_profile.get("yearly_return_rows", []))
                    export_portfolio_reports(result_tuple[0], result_tuple[1], df_yearly, benchmark_ticker, start_year)
                self.after(0, lambda: self._render_result(result_tuple=result_tuple, benchmark_ticker=benchmark_ticker, df_yearly=df_yearly))
                self.after(0, lambda: self._set_running(False, "投組回測完成"))
            except Exception as exc:
                self.after(0, lambda: self._append_console(f"執行失敗：{type(exc).__name__}: {exc}\n"))
                self.after(0, lambda: self._summary_var.set(f"執行失敗：{type(exc).__name__}: {exc}"))
                self.after(0, lambda: self._set_running(False, f"投組回測失敗：{type(exc).__name__}: {exc}"))

        self._worker = threading.Thread(target=_worker, daemon=True)
        self._worker.start()

    def _open_path(self, path):
        normalized_path = os.path.abspath(str(path))
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
