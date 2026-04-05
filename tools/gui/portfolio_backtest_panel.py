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
from core.display_common import _pad_display
from core.strategy_dashboard import build_strategy_dashboard_sections
from tools.debug.charting import get_matplotlib_cjk_font_candidates
from tools.portfolio_sim.reporting import DASHBOARD_HTML_PATH, OUTPUT_DIR, REPORT_XLSX_PATH, export_portfolio_reports, print_yearly_return_report
from tools.portfolio_sim.runtime import BEST_PARAMS_PATH, ensure_runtime_dirs, load_strict_params, run_portfolio_simulation

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:  # pragma: no cover - GUI runtime fallback
    FigureCanvasTkAgg = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_PROGRESS_LOAD_WEIGHT = 0.35
PORTFOLIO_PROGRESS_SIM_WEIGHT = 0.60
PORTFOLIO_PROGRESS_EXPORT_WEIGHT = 0.05
SUMMARY_BORDER_TAG = "summary_border"
SUMMARY_HEADER_TAG = "summary_header"
SUMMARY_NEUTRAL_TAG = "summary_neutral"
SUMMARY_POSITIVE_TAG = "summary_positive"
SUMMARY_NEGATIVE_TAG = "summary_negative"
SUMMARY_CAUTION_TAG = "summary_caution"
SUMMARY_MUTED_TAG = "summary_muted"
SUMMARY_ITEM_WIDTH = 16
SUMMARY_VALUE_WIDTH = 16


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
        self._dataset_var = tk.StringVar(value=f"資料集：{get_dataset_profile_label(DEFAULT_DATASET_PROFILE)}")
        self._progress_detail_var = tk.StringVar(value="進度：尚未執行")
        self._chart_hover_var = tk.StringVar(value="滑鼠移到曲線上可查看日期、投組報酬與大盤報酬。")
        self._worker = None
        self._chart_canvas = None
        self._chart_figure = None
        self._chart_hover_binding_id = None
        self._console_writer = _ThreadSafeConsoleWriter(self)
        self._build_ui()
        self._render_summary_placeholder("尚未執行投組回測")

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
        ttk.Label(progress_row, textvariable=self._progress_detail_var, style="Workbench.TLabel").grid(row=0, column=1, sticky="e")
        self._progress = ttk.Progressbar(progress_row, mode="determinate", maximum=100.0, style="Workbench.Horizontal.TProgressbar")
        self._progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        notebook = ttk.Notebook(self, style="Workbench.TNotebook")
        notebook.pack(fill="both", expand=True, pady=(6, 0))
        self._notebook = notebook

        summary_tab = ttk.Frame(notebook, padding=8, style="Workbench.TFrame")
        summary_tab.rowconfigure(0, weight=1)
        summary_tab.columnconfigure(0, weight=1)
        notebook.add(summary_tab, text="結果彙整")

        summary_pane = ttk.Panedwindow(summary_tab, orient="horizontal")
        summary_pane.grid(row=0, column=0, sticky="nsew")

        chart_section = ttk.Frame(summary_pane, style="Workbench.TFrame")
        chart_section.rowconfigure(0, weight=1)
        chart_section.columnconfigure(0, weight=1)
        summary_pane.add(chart_section, weight=5)

        chart_host = tk.Frame(chart_section, bg="#05090e", highlightthickness=0, bd=0)
        chart_host.grid(row=0, column=0, sticky="nsew")
        self._chart_host = chart_host
        self._chart_placeholder = tk.Label(
            chart_host,
            text="執行投組回測後，這裡會顯示投組淨值曲線與大盤對照。",
            bg="#05090e",
            fg="#f7fbff",
            font=("Microsoft JhengHei", 13),
            anchor="center",
            justify="center",
        )
        self._chart_placeholder.pack(fill="both", expand=True)

        chart_footer = ttk.Frame(chart_section, padding=(2, 8, 2, 0), style="Workbench.TFrame")
        chart_footer.grid(row=1, column=0, sticky="ew")
        ttk.Label(chart_footer, textvariable=self._chart_hover_var, style="Workbench.TLabel").pack(anchor="w")

        compare_section = ttk.Frame(summary_pane, style="Workbench.TFrame")
        compare_section.columnconfigure(0, weight=1)
        compare_section.rowconfigure(1, weight=1)
        summary_pane.add(compare_section, weight=4)
        ttk.Label(compare_section, text="對比表", style="Workbench.SidebarHeader.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self._summary_text = tk.Text(
            compare_section,
            wrap="none",
            bg="#040a12",
            fg="#f7fbff",
            insertbackground="#f7fbff",
            relief="flat",
            bd=0,
            padx=8,
            pady=8,
            font=("Consolas", 10),
        )
        self._summary_text.grid(row=1, column=0, sticky="nsew")
        self._summary_text.configure(state="disabled")
        self._summary_text.tag_configure(SUMMARY_BORDER_TAG, foreground="#9aa9bd")
        self._summary_text.tag_configure(SUMMARY_HEADER_TAG, foreground="#f7fbff", font=("Consolas", 10, "bold"))
        self._summary_text.tag_configure(SUMMARY_NEUTRAL_TAG, foreground="#f7fbff")
        self._summary_text.tag_configure(SUMMARY_POSITIVE_TAG, foreground="#1fe08f")
        self._summary_text.tag_configure(SUMMARY_NEGATIVE_TAG, foreground="#ff6b78")
        self._summary_text.tag_configure(SUMMARY_CAUTION_TAG, foreground="#ffd84d")
        self._summary_text.tag_configure(SUMMARY_MUTED_TAG, foreground="#9aa9bd")
        summary_y = ttk.Scrollbar(compare_section, orient="vertical", command=self._summary_text.yview, style="Workbench.Vertical.TScrollbar")
        summary_y.grid(row=1, column=1, sticky="ns")
        summary_x = ttk.Scrollbar(compare_section, orient="horizontal", command=self._summary_text.xview, style="Workbench.Horizontal.TScrollbar")
        summary_x.grid(row=2, column=0, sticky="ew")
        self._summary_text.configure(yscrollcommand=summary_y.set, xscrollcommand=summary_x.set)
        ttk.Button(compare_section, text="開啟投組儀表板", command=lambda: self._open_path(DASHBOARD_HTML_PATH), style="Workbench.TButton").grid(row=3, column=0, sticky="ew", pady=(8, 0))

        yearly_tab = ttk.Frame(notebook, padding=8, style="Workbench.TFrame")
        yearly_tab.rowconfigure(0, weight=1)
        yearly_tab.columnconfigure(0, weight=1)
        notebook.add(yearly_tab, text="年度報酬%")
        self._yearly_tree = ttk.Treeview(yearly_tab, show="headings", style="Workbench.Treeview")
        self._yearly_tree.grid(row=0, column=0, sticky="nsew")
        yearly_y = ttk.Scrollbar(yearly_tab, orient="vertical", command=self._yearly_tree.yview, style="Workbench.Vertical.TScrollbar")
        yearly_y.grid(row=0, column=1, sticky="ns")
        yearly_x = ttk.Scrollbar(yearly_tab, orient="horizontal", command=self._yearly_tree.xview, style="Workbench.Horizontal.TScrollbar")
        yearly_x.grid(row=1, column=0, sticky="ew")
        self._yearly_tree.configure(yscrollcommand=yearly_y.set, xscrollcommand=yearly_x.set)

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
        if not running and float(self._progress["value"]) <= 0.0:
            self._progress.configure(value=100.0)
        self._set_status(status_text)

    def _set_progress(self, value, detail):
        self._progress.configure(value=max(0.0, min(100.0, float(value))))
        self._progress_detail_var.set(str(detail))
        self.update_idletasks()

    def _report_progress(self, event, payload):
        payload = dict(payload or {})
        stage = str(payload.get("stage", ""))
        current = float(payload.get("current", 0) or 0)
        total = max(float(payload.get("total", 0) or 0), 1.0)
        ratio = current / total
        detail = self._progress_detail_var.get()
        progress_value = float(self._progress["value"])

        if stage == "load_market":
            progress_value = ratio * PORTFOLIO_PROGRESS_LOAD_WEIGHT * 100.0
            detail = f"預載入資料 [{int(current)}/{int(total)}] | 成功 {int(payload.get('success', 0) or 0)} | 資料不足 {int(payload.get('skipped', 0) or 0)}"
        elif stage == "simulate_timeline":
            progress_value = (PORTFOLIO_PROGRESS_LOAD_WEIGHT + ratio * PORTFOLIO_PROGRESS_SIM_WEIGHT) * 100.0
            date_text = str(payload.get("date", "-") or "-")
            equity = float(payload.get("equity", 0.0) or 0.0)
            exposure = float(payload.get("exposure_pct", 0.0) or 0.0)
            detail = f"模擬推進 [{int(current)}/{int(total)}] | {date_text} | 資產 {equity:,.0f} | 水位 {exposure:.1f}%"
        elif stage == "export_reports":
            progress_value = (PORTFOLIO_PROGRESS_LOAD_WEIGHT + PORTFOLIO_PROGRESS_SIM_WEIGHT + ratio * PORTFOLIO_PROGRESS_EXPORT_WEIGHT) * 100.0
            detail = f"輸出報表 [{int(current)}/{int(total)}]"

        self.after(0, lambda: self._set_progress(progress_value, detail))

    def _render_table(self, tree, frame: pd.DataFrame | None, *, yearly_mode=False):
        tree.delete(*tree.get_children())
        if frame is None or frame.empty:
            tree.configure(columns=("message",))
            tree.heading("message", text="訊息")
            tree.column("message", width=960, anchor="w")
            tree.insert("", "end", values=("目前沒有資料。",))
            return

        normalized = frame.copy().where(pd.notna(frame), "")
        if yearly_mode and "year_return_pct" in normalized.columns:
            normalized["year_return_pct"] = normalized["year_return_pct"].map(lambda value: "" if value == "" else f"{float(value):.2f}%")
        if yearly_mode and "is_full_year" in normalized.columns:
            normalized["is_full_year"] = normalized["is_full_year"].map(lambda value: "完整" if bool(value) else "非完整")

        columns = list(normalized.columns)
        tree.configure(columns=columns)
        for column in columns:
            tree.heading(column, text=str(column))
            if str(column) in {"Date", "日期", "start_date", "end_date"}:
                width = 160
                anchor = "center"
            elif str(column) in {"動作", "備註", "year_return_pct"}:
                width = 180 if str(column) == "year_return_pct" else 260
                anchor = "center" if str(column) == "year_return_pct" else "w"
            else:
                width = 140
                anchor = "center"
            tree.column(column, width=width, anchor=anchor)
        for row in normalized.to_dict("records"):
            tree.insert("", "end", values=[row.get(column, "") for column in columns])

    def _clear_summary_text(self):
        self._summary_text.configure(state="normal")
        self._summary_text.delete("1.0", "end")
        self._summary_text.configure(state="disabled")

    def _append_summary_segments(self, segments):
        self._summary_text.configure(state="normal")
        for text, tag in segments:
            self._summary_text.insert("end", str(text), tag)
        self._summary_text.configure(state="disabled")

    def _append_summary_plain_line(self, text, tag=SUMMARY_NEUTRAL_TAG):
        self._append_summary_segments([(f"{text}\n", tag)])

    def _summary_value_tag(self, value, *, item, column):
        text = str(value or "").strip()
        if not text or text == "-":
            return SUMMARY_MUTED_TAG
        if item == "最大回撤 (MDD)":
            if column == "system":
                return SUMMARY_CAUTION_TAG
            if column == "alpha":
                if text.startswith("少跌"):
                    return SUMMARY_POSITIVE_TAG
                if text.startswith("多跌"):
                    return SUMMARY_NEGATIVE_TAG
        if column == "benchmark":
            return SUMMARY_NEUTRAL_TAG
        if text.startswith("+"):
            return SUMMARY_POSITIVE_TAG
        if text.startswith("-"):
            return SUMMARY_NEGATIVE_TAG
        return SUMMARY_NEUTRAL_TAG

    def _build_summary_metric_segments(self, item, system, benchmark, alpha, *, header=False):
        row_tag = SUMMARY_HEADER_TAG if header else SUMMARY_NEUTRAL_TAG
        value_tag_system = row_tag if header else self._summary_value_tag(system, item=item, column="system")
        value_tag_benchmark = row_tag if header else self._summary_value_tag(benchmark, item=item, column="benchmark")
        value_tag_alpha = row_tag if header else self._summary_value_tag(alpha, item=item, column="alpha")
        return [
            ("| ", SUMMARY_BORDER_TAG),
            (_pad_display(item, SUMMARY_ITEM_WIDTH), row_tag),
            (" | ", SUMMARY_BORDER_TAG),
            (_pad_display(system, SUMMARY_VALUE_WIDTH), value_tag_system),
            (" | ", SUMMARY_BORDER_TAG),
            (_pad_display(benchmark, SUMMARY_VALUE_WIDTH), value_tag_benchmark),
            (" | ", SUMMARY_BORDER_TAG),
            (_pad_display(alpha, SUMMARY_VALUE_WIDTH), value_tag_alpha),
            (" |\n", SUMMARY_BORDER_TAG),
        ]

    def _render_summary_placeholder(self, message):
        self._clear_summary_text()
        self._append_summary_plain_line(str(message), tag=SUMMARY_MUTED_TAG)

    def _render_summary_table(self, *, result_tuple, params, benchmark_ticker, max_positions, enable_rotation):
        self._clear_summary_text()
        (
            _df_eq, _df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff,
            final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed,
            total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate,
            normal_trade_count, extended_trade_count, annual_trades,
            reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, pf_profile,
        ) = result_tuple

        sections = build_strategy_dashboard_sections(
            params=params,
            mode_display="開啟 (強勢輪動)" if enable_rotation else "關閉 (穩定鎖倉)",
            max_pos=max_positions,
            trades=trade_count,
            missed_b=total_missed,
            missed_s=total_missed_sells,
            final_eq=final_eq,
            avg_exp=avg_exp,
            sys_ret=tot_ret,
            bm_ret=bm_ret,
            sys_mdd=mdd,
            bm_mdd=bm_mdd,
            win_rate=win_rate,
            payoff=pf_payoff,
            ev=pf_ev,
            benchmark_ticker=benchmark_ticker,
            max_exp=max_exp,
            r_sq=r_sq,
            m_win_rate=m_win_rate,
            bm_r_sq=bm_r_sq,
            bm_m_win_rate=bm_m_win_rate,
            normal_trades=normal_trade_count,
            extended_trades=extended_trade_count,
            annual_trades=annual_trades,
            reserved_buy_fill_rate=reserved_buy_fill_rate,
            annual_return_pct=annual_return_pct,
            bm_annual_return_pct=bm_annual_return_pct,
            min_full_year_return_pct=pf_profile.get("min_full_year_return_pct", 0.0),
            bm_min_full_year_return_pct=pf_profile.get("bm_min_full_year_return_pct", 0.0),
        )

        border_line = "-" * 80
        self._append_summary_plain_line(sections["global_strategy_line"], tag=SUMMARY_HEADER_TAG)
        for line in sections["overview_lines"]:
            self._append_summary_plain_line(line)
        self._append_summary_plain_line(border_line, tag=SUMMARY_BORDER_TAG)
        self._append_summary_segments(
            self._build_summary_metric_segments(
                "指標項目",
                "V16 尊爵系統",
                f"同期大盤 ({sections['benchmark_ticker']})",
                "差異 (Alpha)",
                header=True,
            )
        )
        for row in sections["metric_rows"]:
            self._append_summary_segments(
                self._build_summary_metric_segments(
                    row["item"],
                    row["system"],
                    row["benchmark"],
                    row["alpha"],
                )
            )
        self._append_summary_plain_line(border_line, tag=SUMMARY_BORDER_TAG)
        self._append_summary_plain_line("【訓練參數】", tag=SUMMARY_HEADER_TAG)
        for row in sections["training_rows"]:
            self._append_summary_plain_line(f"{row['item']} : {row['value']}")
        self._append_summary_plain_line(border_line, tag=SUMMARY_BORDER_TAG)
        self._append_summary_plain_line("【共用硬門檻】", tag=SUMMARY_HEADER_TAG)
        for row in sections["threshold_rows"]:
            self._append_summary_plain_line(f"{row['item']} : {row['value']}")
        self._summary_text.see("1.0")

    def _resolve_matplotlib_font_family(self):
        try:
            from matplotlib import font_manager
        except ImportError:
            return None
        for family in get_matplotlib_cjk_font_candidates():
            try:
                font_manager.findfont(family, fallback_to_default=False)
                return family
            except Exception:
                continue
        return None

    def _build_portfolio_figure(self, df_eq, benchmark_ticker):
        if FigureCanvasTkAgg is None:
            raise RuntimeError("環境缺少 matplotlib Tk backend，無法在 GUI 內嵌顯示投組淨值曲線。")
        from matplotlib import rcParams
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties

        font_family = self._resolve_matplotlib_font_family()
        rcParams["axes.unicode_minus"] = False
        if font_family:
            rcParams["font.sans-serif"] = [font_family, *get_matplotlib_cjk_font_candidates()]
        label_font = FontProperties(family=font_family, size=11) if font_family else None
        title_font = FontProperties(family=font_family, size=13, weight="bold") if font_family else None

        figure = Figure(figsize=(12.4, 6.3), dpi=96, facecolor="#02050a")
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
            axis.plot(df_eq["Date"], df_eq[bm_col], color="#4f86ff", linewidth=1.6, label=f"同期大盤 {benchmark_ticker} (%)")
        axis.set_title(f"投組淨值 vs {benchmark_ticker}", color="#fbfdff", fontproperties=title_font)
        axis.set_ylabel("累積報酬率 (%)", color="#fbfdff", fontproperties=label_font)
        legend = axis.legend(loc="upper left", frameon=False, prop=label_font)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#fbfdff")
        if label_font is not None:
            for tick in axis.get_xticklabels() + axis.get_yticklabels():
                tick.set_fontproperties(label_font)
        figure.subplots_adjust(left=0.070, right=0.985, top=0.93, bottom=0.12)
        return figure

    def _update_chart_hover_label(self, df_eq, benchmark_ticker, index):
        idx = max(0, min(int(index), len(df_eq) - 1))
        row = df_eq.iloc[idx]
        bm_col = f"Benchmark_{benchmark_ticker}_Pct"
        bm_text = f" | 大盤 {benchmark_ticker} {float(row[bm_col]):.2f}%" if bm_col in df_eq.columns and pd.notna(row[bm_col]) else ""
        self._chart_hover_var.set(
            f"{pd.Timestamp(row['Date']).strftime('%Y-%m-%d')} | 投組 {float(row['Strategy_Return_Pct']):.2f}%{bm_text}"
        )

    def _bind_chart_hover(self, canvas, figure, df_eq, benchmark_ticker):
        if figure.canvas is None:
            return
        try:
            import matplotlib.dates as mdates
        except ImportError:
            return

        x_values = mdates.date2num(pd.to_datetime(df_eq["Date"]).dt.to_pydatetime())
        axis = figure.axes[0] if figure.axes else None
        if axis is None:
            return

        self._update_chart_hover_label(df_eq, benchmark_ticker, len(df_eq) - 1)

        def _on_motion(event):
            if event.inaxes is not axis or event.xdata is None:
                return
            nearest = int(abs(x_values - float(event.xdata)).argmin())
            self._update_chart_hover_label(df_eq, benchmark_ticker, nearest)

        self._chart_hover_binding_id = canvas.mpl_connect("motion_notify_event", _on_motion)

    def _render_embedded_chart(self, figure, *, df_eq, benchmark_ticker):
        self._clear_embedded_chart()
        self._chart_placeholder.pack_forget()
        canvas = FigureCanvasTkAgg(figure, master=self._chart_host)
        canvas.draw()
        self._bind_chart_hover(canvas, figure, df_eq, benchmark_ticker)
        widget = canvas.get_tk_widget()
        widget.configure(background="#02050a", highlightthickness=0, bd=0)
        widget.pack(fill="both", expand=True)
        self._chart_canvas = canvas
        self._chart_figure = figure

    def _clear_embedded_chart(self):
        if self._chart_canvas is not None:
            if self._chart_hover_binding_id is not None:
                try:
                    self._chart_canvas.mpl_disconnect(self._chart_hover_binding_id)
                except Exception:
                    pass
            self._chart_canvas.get_tk_widget().destroy()
            self._chart_canvas = None
            self._chart_hover_binding_id = None
        if self._chart_figure is not None:
            self._chart_figure.clear()
            self._chart_figure = None
        if hasattr(self, "_chart_placeholder") and not self._chart_placeholder.winfo_ismapped():
            self._chart_placeholder.pack(fill="both", expand=True)
        self._chart_hover_var.set("滑鼠移到曲線上可查看日期、投組報酬與大盤報酬。")

    def _render_result(self, *, result_tuple, benchmark_ticker, df_yearly, params, max_positions, enable_rotation):
        df_eq = result_tuple[0]
        df_tr = result_tuple[1]
        self._render_summary_table(
            result_tuple=result_tuple,
            params=params,
            benchmark_ticker=benchmark_ticker,
            max_positions=max_positions,
            enable_rotation=enable_rotation,
        )
        self._render_table(self._trade_tree, df_tr)
        self._render_table(self._yearly_tree, df_yearly, yearly_mode=True)
        self._render_embedded_chart(self._build_portfolio_figure(df_eq, benchmark_ticker), df_eq=df_eq, benchmark_ticker=benchmark_ticker)
        self._notebook.select(0)

    def _run_portfolio_sim(self):
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("股票工具工作台", "投組回測執行中，請稍候。")
            return

        try:
            max_positions = int(self._max_pos_var.get().strip() or "10")
            start_year = int(self._start_year_var.get().strip() or "2015")
        except ValueError:
            messagebox.showerror("股票工具工作台", "最大持倉與開始年份必須是整數。")
            return

        benchmark_ticker = self._benchmark_var.get().strip() or "0050"
        enable_rotation = bool(self._rotation_var.get())
        data_dir = get_dataset_dir(str(PROJECT_ROOT), DEFAULT_DATASET_PROFILE)

        self._console.delete("1.0", "end")
        self._render_summary_placeholder("執行中…")
        self._clear_embedded_chart()
        self._render_table(self._trade_tree, None)
        self._render_table(self._yearly_tree, None)
        self._set_progress(0.0, "進度：初始化")
        self._set_running(True, "執行中：投組回測")

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
                        progress_callback=self._report_progress,
                    )
                    pf_profile = result_tuple[-1]
                    df_yearly = print_yearly_return_report(pf_profile.get("yearly_return_rows", []))
                    self._report_progress("export_start", {"stage": "export_reports", "current": 0, "total": 2})
                    export_portfolio_reports(result_tuple[0], result_tuple[1], df_yearly, benchmark_ticker, start_year)
                    self._report_progress("export_finish", {"stage": "export_reports", "current": 2, "total": 2})
                self.after(
                    0,
                    lambda: self._render_result(
                        result_tuple=result_tuple,
                        benchmark_ticker=benchmark_ticker,
                        df_yearly=df_yearly,
                        params=params,
                        max_positions=max_positions,
                        enable_rotation=enable_rotation,
                    ),
                )
                self.after(0, lambda: self._set_progress(100.0, "完成 [100%]"))
                self.after(0, lambda: self._set_running(False, "投組回測完成"))
            except Exception as exc:
                self.after(0, lambda: self._append_console(f"執行失敗：{type(exc).__name__}: {exc}\n"))
                self.after(0, lambda: self._render_summary_placeholder(f"執行失敗：{type(exc).__name__}: {exc}"))
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
