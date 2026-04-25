from __future__ import annotations

import io
import os
import re
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from tkinter import messagebox, ttk
import tkinter as tk

import numpy as np
import pandas as pd

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.model_paths import resolve_candidate_best_params_path, resolve_run_best_params_path
from core.portfolio_fast_data import get_fast_dates
from core.runtime_utils import parse_float_strict, parse_int_strict
from core.walk_forward_policy import load_walk_forward_policy
from tools.portfolio_sim.reporting import export_portfolio_reports, print_yearly_return_report
from tools.portfolio_sim.runtime import ensure_runtime_dirs, load_strict_params
from tools.portfolio_sim.simulation_runner import (
    PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
    load_portfolio_market_context,
    run_portfolio_simulation_prepared,
)
from tools.trade_analysis.charting import (
    bind_matplotlib_chart_navigation,
    build_chart_hover_snapshot,
    build_debug_chart_payload,
    create_debug_chart_context,
    create_matplotlib_trade_chart_figure,
    record_trade_marker,
    scroll_chart_to_latest,
)

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError as exc:  # pragma: no cover - GUI runtime fallback
    FigureCanvasTkAgg = None
    FIGURE_CANVAS_TKAGG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    FIGURE_CANVAS_TKAGG_IMPORT_ERROR = ""


WORKBENCH_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARAM_SOURCE_LABEL_TO_KEY = {
    "run_best | 目前參數": "run_best",
    "candidate_best | 候選參數": "candidate_best",
}
DEFAULT_PARAM_SOURCE_LABEL = "run_best | 目前參數"
ROTATION_LABEL_TO_BOOL = {
    "關閉 (穩定鎖倉)": False,
    "啟用 (強勢輪動)": True,
}
DEFAULT_ROTATION_LABEL = "關閉 (穩定鎖倉)"
FIXED_RISK_LABELS = ("0.01", "0.02", "自訂")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
PORTFOLIO_CONSOLE_COLORS = {
    "91": "#ff6174",
    "93": "#facc15",
    "96": "#4fd1ff",
    "92": "#5ee28a",
    "90": "#9aa7b6",
    "94": "#7fb3ff",
}


class _PortfolioConsoleWriter(io.TextIOBase):
    def __init__(self, panel: "PortfolioBacktestInspectorPanel"):
        super().__init__()
        self._panel = panel

    def write(self, text):
        if not text:
            return 0
        self._panel._append_console_stream(str(text))
        return len(text)

    def flush(self):
        return None



def _resolve_default_portfolio_start_year_hint():
    policy = load_walk_forward_policy(WORKBENCH_PROJECT_ROOT)
    return int(policy["search_train_end_year"]) + 1


def _coerce_float(value, default=np.nan):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _coerce_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _format_pct(value):
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _fast_data_to_price_df(fast_data):
    dates = pd.DatetimeIndex(pd.to_datetime(get_fast_dates(fast_data)))
    return pd.DataFrame(
        {
            "Open": np.asarray(fast_data["Open"], dtype=float),
            "High": np.asarray(fast_data["High"], dtype=float),
            "Low": np.asarray(fast_data["Low"], dtype=float),
            "Close": np.asarray(fast_data["Close"], dtype=float),
            "Volume": np.asarray(fast_data["Volume"], dtype=float),
        },
        index=dates,
    )


def _normalize_trade_action(row):
    raw_type = str(row.get("Type", "") or "").strip()
    entry_type = str(row.get("進場類型", "") or "").strip()
    if raw_type.startswith("買進"):
        return "買進(延續候選)" if entry_type == "extended" else "買進"
    if raw_type == "半倉停利":
        return "半倉停利"
    if raw_type == "全倉結算(停損)":
        return "停損殺出"
    if raw_type == "全倉結算(指標)":
        return "指標賣出"
    if raw_type.startswith("汰弱賣出"):
        return "指標賣出"
    if raw_type == "期末強制結算":
        return "期末強制結算"
    if raw_type == "錯失賣出":
        return "錯失賣出"
    return ""


def _is_actual_trade_row(row):
    action = _normalize_trade_action(row)
    return bool(action) and not str(row.get("Type", "") or "").startswith("錯失")


def _resolve_marker_price(price_df, row, action):
    price = _coerce_float(row.get("成交價"))
    if not pd.isna(price) and price > 0:
        return price
    date = pd.Timestamp(row.get("Date"))
    if date not in price_df.index:
        return np.nan
    if action in {"買進", "買進(延續候選)"}:
        return float(price_df.loc[date, "Low"])
    if action == "半倉停利":
        return float(price_df.loc[date, "High"])
    return float(price_df.loc[date, "Close"])


def _build_portfolio_ticker_chart_payload(*, ticker, fast_data, ticker_trades_df):
    price_df = _fast_data_to_price_df(fast_data)
    chart_context = create_debug_chart_context(price_df)
    for row in ticker_trades_df.to_dict("records"):
        action = _normalize_trade_action(row)
        if not action:
            continue
        try:
            trade_date = pd.Timestamp(row.get("Date"))
        except (TypeError, ValueError):
            continue
        if trade_date not in price_df.index:
            continue
        marker_price = _resolve_marker_price(price_df, row, action)
        if pd.isna(marker_price):
            continue
        qty = _coerce_int(row.get("股數"), default=0)
        note_parts = [str(row.get("Type", "") or "").strip()]
        pnl = row.get("該筆總損益", row.get("單筆損益", None))
        if pnl is not None and not pd.isna(pnl):
            note_parts.append(f"損益 {float(pnl):,.0f}")
        record_trade_marker(
            chart_context,
            current_date=trade_date,
            action=action,
            price=marker_price,
            qty=qty,
            note=" | ".join(part for part in note_parts if part),
            meta={"portfolio_view": True},
        )
    chart_context["summary_box"] = [
        "投組實際成交",
        f"{ticker} 共 {len(ticker_trades_df)} 筆事件",
    ]
    chart_context["status_box"] = {"lines": ["投組成交檢視"], "ok": True}
    return build_debug_chart_payload(price_df, chart_context)


class PortfolioBacktestInspectorPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=4, style="Workbench.TFrame")
        self._ui_thread = threading.current_thread()
        self._console_writer = _PortfolioConsoleWriter(self)
        self._run_thread = None
        self._active_token = 0
        self._status_var = tk.StringVar(value="尚未執行")
        self._param_source_display_var = tk.StringVar(value=DEFAULT_PARAM_SOURCE_LABEL)
        self._rotation_display_var = tk.StringVar(value=DEFAULT_ROTATION_LABEL)
        self._max_positions_var = tk.StringVar(value="10")
        self._start_year_var = tk.StringVar(value=str(_resolve_default_portfolio_start_year_hint()))
        self._fixed_risk_display_var = tk.StringVar(value="0.01")
        self._custom_fixed_risk_var = tk.StringVar(value="0.01")
        self._ticker_display_var = tk.StringVar()
        self._show_volume_var = tk.BooleanVar(value=False)
        self._result = None
        self._ticker_map = {}
        self._chart_canvas = None
        self._chart_figure = None
        self._performance_canvas = None
        self._performance_figure = None
        self._console_stream_buffer = ""
        self._console_stream_mode = "line"
        self._console_live_progress_start = None
        self._console_current_tag = "default"
        self._build_ui()

    def destroy(self):
        self._clear_kline_chart()
        self._clear_performance_chart()
        super().destroy()

    def _build_ui(self):
        controls = ttk.Frame(self, padding=(8, 2, 8, 2), style="Workbench.TFrame")
        controls.pack(fill="x", pady=(0, 4))
        controls_bar = ttk.Frame(controls, style="Workbench.TFrame")
        controls_bar.pack(side="left", anchor="w")
        pady = (2, 2)

        ttk.Label(controls_bar, text="參數", style="Workbench.TLabel").grid(row=0, column=0, padx=(0, 6), pady=pady, sticky="w")
        ttk.Combobox(
            controls_bar,
            state="readonly",
            width=20,
            textvariable=self._param_source_display_var,
            style="Workbench.TCombobox",
            values=list(PARAM_SOURCE_LABEL_TO_KEY.keys()),
        ).grid(row=0, column=1, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="汰弱換股", style="Workbench.TLabel").grid(row=0, column=2, padx=(0, 6), pady=pady, sticky="w")
        ttk.Combobox(
            controls_bar,
            state="readonly",
            width=14,
            textvariable=self._rotation_display_var,
            style="Workbench.TCombobox",
            values=list(ROTATION_LABEL_TO_BOOL.keys()),
        ).grid(row=0, column=3, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="最大持股", style="Workbench.TLabel").grid(row=0, column=4, padx=(0, 6), pady=pady, sticky="w")
        ttk.Entry(controls_bar, textvariable=self._max_positions_var, width=5, style="Workbench.TEntry").grid(row=0, column=5, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="起始年", style="Workbench.TLabel").grid(row=0, column=6, padx=(0, 6), pady=pady, sticky="w")
        ttk.Entry(controls_bar, textvariable=self._start_year_var, width=7, style="Workbench.TEntry").grid(row=0, column=7, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="固定風險", style="Workbench.TLabel").grid(row=0, column=8, padx=(0, 6), pady=pady, sticky="w")
        risk_combo = ttk.Combobox(
            controls_bar,
            state="readonly",
            width=7,
            textvariable=self._fixed_risk_display_var,
            style="Workbench.TCombobox",
            values=FIXED_RISK_LABELS,
        )
        risk_combo.grid(row=0, column=9, padx=(0, 6), pady=pady, sticky="w")
        risk_combo.bind("<<ComboboxSelected>>", self._on_fixed_risk_selected)
        self._custom_fixed_risk_entry = ttk.Entry(controls_bar, textvariable=self._custom_fixed_risk_var, width=7, style="Workbench.TEntry")
        self._custom_fixed_risk_entry.grid(row=0, column=10, padx=(0, 10), pady=pady, sticky="w")
        self._custom_fixed_risk_entry.state(["disabled"])

        ttk.Button(controls_bar, text="執行投組回測", command=self._run_portfolio_backtest, style="Workbench.TButton").grid(row=0, column=11, padx=(0, 10), pady=pady, sticky="w")

        ttk.Label(controls_bar, text="K線股票", style="Workbench.TLabel").grid(row=0, column=12, padx=(0, 6), pady=pady, sticky="w")
        self._ticker_combo = ttk.Combobox(controls_bar, state="readonly", width=22, textvariable=self._ticker_display_var, style="Workbench.TCombobox", values=[])
        self._ticker_combo.grid(row=0, column=13, padx=(0, 8), pady=pady, sticky="w")
        self._ticker_combo.bind("<<ComboboxSelected>>", self._on_ticker_selected)

        ttk.Checkbutton(
            controls_bar,
            text="顯示成交量",
            variable=self._show_volume_var,
            command=self._rerender_selected_ticker_chart,
            style="Workbench.TCheckbutton",
        ).grid(row=0, column=14, padx=(0, 0), pady=pady, sticky="w")

        notebook = ttk.Notebook(self, style="Workbench.TNotebook")
        notebook.pack(fill="both", expand=True)
        self._notebook = notebook

        kline_tab = ttk.Frame(notebook, padding=0, style="Workbench.TFrame")
        kline_tab.rowconfigure(0, weight=1)
        kline_tab.columnconfigure(0, weight=1)
        notebook.add(kline_tab, text="K 線圖")
        self._kline_host = tk.Frame(kline_tab, bg="#000000", highlightthickness=0, bd=0)
        self._kline_host.grid(row=0, column=0, sticky="nsew")
        self._kline_placeholder = self._make_placeholder(self._kline_host, "請先執行投組回測；選擇有成交過的股票後會顯示 K 線結果。")

        performance_tab = ttk.Frame(notebook, padding=0, style="Workbench.TFrame")
        performance_tab.rowconfigure(0, weight=1)
        performance_tab.columnconfigure(0, weight=1)
        notebook.add(performance_tab, text="績效圖")
        self._performance_host = tk.Frame(performance_tab, bg="#000000", highlightthickness=0, bd=0)
        self._performance_host.grid(row=0, column=0, sticky="nsew")
        self._performance_placeholder = self._make_placeholder(self._performance_host, "請先執行投組回測；績效圖會顯示投組與 0050 大盤。")

        console_tab = ttk.Frame(notebook, padding=10, style="Workbench.TFrame")
        console_tab.rowconfigure(0, weight=1)
        console_tab.columnconfigure(0, weight=1)
        notebook.add(console_tab, text="Console")
        self._console_text = tk.Text(console_tab, wrap="none", bg="#040a12", fg="#f7fbff", insertbackground="#f7fbff", relief="flat", bd=0, font=("Consolas", 10))
        self._console_text.grid(row=0, column=0, sticky="nsew")
        self._configure_console_tags()
        y_scroll = ttk.Scrollbar(console_tab, orient="vertical", command=self._console_text.yview, style="Workbench.Vertical.TScrollbar")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(console_tab, orient="horizontal", command=self._console_text.xview, style="Workbench.Horizontal.TScrollbar")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self._console_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        footer = ttk.Frame(self, style="Workbench.TFrame")
        footer.pack(fill="x", pady=(2, 0))
        ttk.Label(footer, textvariable=self._status_var, style="Workbench.TLabel").pack(anchor="w")
        self._notebook.select(kline_tab)

    def _make_placeholder(self, master, text):
        label_fg = ttk.Style(self).lookup("Workbench.TLabel", "foreground") or "#f7fbff"
        label = tk.Label(
            master,
            text=text,
            anchor="center",
            justify="center",
            bg="#000000",
            fg=label_fg,
            font=("Microsoft JhengHei", 12),
        )
        label.pack(fill="both", expand=True)
        return label

    def _configure_console_tags(self):
        self._console_text.tag_configure("default", foreground="#f7fbff")
        for code, color in PORTFOLIO_CONSOLE_COLORS.items():
            self._console_text.tag_configure(f"ansi_{code}", foreground=color)

    def _on_fixed_risk_selected(self, _event=None):
        if self._fixed_risk_display_var.get() == "自訂":
            self._custom_fixed_risk_entry.state(["!disabled"])
            self._custom_fixed_risk_entry.focus_set()
        else:
            self._custom_fixed_risk_entry.state(["disabled"])

    def _get_selected_param_source(self):
        return PARAM_SOURCE_LABEL_TO_KEY.get(self._param_source_display_var.get().strip(), "run_best")

    def _get_selected_params_path(self):
        if self._get_selected_param_source() == "candidate_best":
            return resolve_candidate_best_params_path(WORKBENCH_PROJECT_ROOT)
        return resolve_run_best_params_path(WORKBENCH_PROJECT_ROOT)

    def _resolve_fixed_risk(self):
        selected = self._fixed_risk_display_var.get().strip()
        raw_value = self._custom_fixed_risk_var.get().strip() if selected == "自訂" else selected
        return parse_float_strict(raw_value, "固定風險比例", min_value=0.0, max_value=1.0, strict_gt=True)

    def _resolve_user_options(self):
        max_positions = parse_int_strict(self._max_positions_var.get().strip(), "最大持倉數量", min_value=1)
        start_year = parse_int_strict(self._start_year_var.get().strip(), "開始回測年份", min_value=1900)
        return {
            "params_path": self._get_selected_params_path(),
            "param_source": self._get_selected_param_source(),
            "enable_rotation": ROTATION_LABEL_TO_BOOL.get(self._rotation_display_var.get().strip(), False),
            "max_positions": max_positions,
            "start_year": start_year,
            "fixed_risk": self._resolve_fixed_risk(),
            "benchmark_ticker": PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
        }

    def _append_console_text(self, text):
        normalized_text = str(text or "")
        if not normalized_text:
            return
        if threading.current_thread() is not self._ui_thread:
            self.after(0, self._append_console_text, normalized_text)
            return
        self._flush_console_live_progress(force_newline=True)
        self._insert_ansi_text(normalized_text)
        self._console_text.see("end")

    def _append_console_stream(self, text):
        normalized_text = str(text or "").replace("\r\n", "\n")
        if not normalized_text:
            return
        if threading.current_thread() is not self._ui_thread:
            self.after(0, self._append_console_stream, normalized_text)
            return
        current = self._console_stream_buffer
        mode = self._console_stream_mode
        ended_with_carriage_return = False
        for char in normalized_text:
            if char == "\r":
                self._set_console_live_progress(current)
                current = ""
                mode = "progress"
                ended_with_carriage_return = True
                continue
            ended_with_carriage_return = False
            if char == "\n":
                if mode == "progress":
                    self._set_console_live_progress(current)
                    self._flush_console_live_progress(force_newline=True)
                else:
                    self._insert_ansi_text(current + "\n")
                    self._console_text.see("end")
                current = ""
                mode = "line"
                continue
            current += char
        self._console_stream_buffer = current
        self._console_stream_mode = mode
        if mode == "progress" and not ended_with_carriage_return:
            self._set_console_live_progress(current)

    def _insert_ansi_text(self, text):
        current_tag = self._console_current_tag
        pos = 0
        for match in ANSI_PATTERN.finditer(text):
            if match.start() > pos:
                self._console_text.insert("end", text[pos:match.start()], current_tag)
            sequence = match.group(0)
            codes = sequence[2:-1].split(";")
            if not codes or codes == ["0"] or "0" in codes:
                current_tag = "default"
            else:
                for code in codes:
                    if code in PORTFOLIO_CONSOLE_COLORS:
                        current_tag = f"ansi_{code}"
            pos = match.end()
        if pos < len(text):
            self._console_text.insert("end", text[pos:], current_tag)
        self._console_current_tag = current_tag

    def _set_console_live_progress(self, text):
        if self._console_live_progress_start is None:
            self._console_live_progress_start = self._console_text.index("end-1c")
            self._insert_ansi_text(str(text or ""))
        else:
            self._console_text.delete(self._console_live_progress_start, "end-1c")
            self._insert_ansi_text(str(text or ""))
        self._console_text.see("end")

    def _flush_console_live_progress(self, *, force_newline):
        if self._console_live_progress_start is None:
            return
        if force_newline:
            line_tail = self._console_text.get(
                f"{self._console_live_progress_start} lineend",
                f"{self._console_live_progress_start} lineend +1c",
            )
            if line_tail != "\n":
                self._console_text.insert("end", "\n", self._console_current_tag)
        self._console_live_progress_start = None
        self._console_text.see("end")

    def _clear_console(self):
        self._console_stream_buffer = ""
        self._console_stream_mode = "line"
        self._console_live_progress_start = None
        self._console_current_tag = "default"
        self._console_text.delete("1.0", "end")

    def _report_runtime_exception(self, context, exc, *, status_prefix, show_dialog=True):
        error_text = f"{status_prefix}：{type(exc).__name__}: {exc}"
        trace_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._append_console_text(f"[{context}]\n{trace_text}\n")
        self._status_var.set(error_text)
        self._notebook.select(2)
        if show_dialog:
            messagebox.showerror("股票工具工作台", error_text)
        return error_text

    def _run_portfolio_backtest(self):
        if self._run_thread is not None and self._run_thread.is_alive():
            status_text = "投組回測進行中：請等待目前回測完成"
            self._status_var.set(status_text)
            self._append_console_text(f"[portfolio] {status_text}\n")
            return
        try:
            options = self._resolve_user_options()
        except ValueError as exc:
            self._report_runtime_exception("resolve_options", exc, status_prefix="輸入錯誤", show_dialog=True)
            return

        self._active_token += 1
        request_token = self._active_token
        self._clear_console()
        self._notebook.select(2)
        self._status_var.set("執行中：投組回測")
        self._append_console_text("[portfolio] 執行中：投組回測\n")
        run_thread = threading.Thread(
            target=self._run_portfolio_worker,
            args=(options, request_token),
            name="workbench-portfolio-backtest",
            daemon=True,
        )
        self._run_thread = run_thread
        run_thread.start()

    def _run_portfolio_worker(self, options, request_token):
        try:
            with redirect_stdout(self._console_writer), redirect_stderr(self._console_writer):
                result_payload = self._execute_portfolio_backtest(options)
        except Exception as exc:
            self.after(0, self._finish_portfolio_error, request_token, exc)
            return
        self.after(0, self._finish_portfolio_success, request_token, result_payload)

    def _execute_portfolio_backtest(self, options):
        data_dir = get_dataset_dir(WORKBENCH_PROJECT_ROOT, DEFAULT_DATASET_PROFILE)
        params = load_strict_params(options["params_path"])
        params.fixed_risk = float(options["fixed_risk"])

        print(f"{C_CYAN}================================================================================{C_RESET}")
        print(f"⚙️ {C_YELLOW}V16 投資組合模擬器：機構級實戰期望值 (終極模組化對齊版){C_RESET}")
        print(f"{C_CYAN}================================================================================{C_RESET}")
        print(f"{C_GRAY}📁 使用資料集: {get_dataset_profile_label(DEFAULT_DATASET_PROFILE)} | 來源: workbench | 路徑: {data_dir}{C_RESET}")
        print(f"{C_GRAY}ℹ️ 參數來源: {options['param_source']}{C_RESET}")
        print(f"\n{C_GREEN}✅ 成功載入 AI 訓練大腦！{C_RESET}")
        print(f"{C_GRAY}📦 參數檔: {options['params_path']}{C_RESET}")
        print(f"{C_GRAY}ℹ️ 單筆固定風險: {params.fixed_risk:.4f}{C_RESET}")

        ensure_runtime_dirs()
        start_time = time.time()
        context = load_portfolio_market_context(data_dir, params, verbose=True)
        result = run_portfolio_simulation_prepared(
            context["all_dfs_fast"],
            context["all_trade_logs"],
            context["sorted_dates"],
            params,
            max_positions=options["max_positions"],
            enable_rotation=options["enable_rotation"],
            start_year=options["start_year"],
            benchmark_ticker=options["benchmark_ticker"],
            verbose=True,
        )
        end_time = time.time()

        (
            df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff,
            final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed,
            total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate,
            normal_trade_count, extended_trade_count, annual_trades,
            reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, pf_profile,
        ) = result

        mode_display = "開啟 (強勢輪動)" if options["enable_rotation"] else "關閉 (穩定鎖倉)"
        min_full_year_return_pct = pf_profile.get("min_full_year_return_pct", 0.0)
        bm_min_full_year_return_pct = pf_profile.get("bm_min_full_year_return_pct", 0.0)

        print(f"\n{C_CYAN}================================================================================{C_RESET}")
        print(f"📊 【投資組合實戰模擬報告 (自 {options['start_year']} 年起算)】")
        print(f"{C_CYAN}================================================================================{C_RESET}")
        print(f"回測總耗時: {end_time - start_time:.2f} 秒")

        print_strategy_dashboard(
            params=params,
            title="績效與風險對比表",
            mode_display=mode_display,
            max_pos=options["max_positions"],
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
            benchmark_ticker=options["benchmark_ticker"],
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
            min_full_year_return_pct=min_full_year_return_pct,
            bm_min_full_year_return_pct=bm_min_full_year_return_pct,
        )

        df_yearly = print_yearly_return_report(
            pf_profile.get("yearly_return_rows", []),
            benchmark_yearly_return_rows=pf_profile.get("bm_yearly_return_rows", []),
            benchmark_ticker=options["benchmark_ticker"],
        )
        if pf_profile.get("full_year_count", 0) > 0:
            print(
                f"{C_GRAY}完整年度數: {pf_profile.get('full_year_count', 0)} | "
                f"最差完整年度報酬: {pf_profile.get('min_full_year_return_pct', 0.0):.2f}% | "
                f"大盤最差完整年度報酬: {pf_profile.get('bm_min_full_year_return_pct', 0.0):.2f}% | "
                f"年化報酬率: {annual_return_pct:.2f}%{C_RESET}"
            )

        export_portfolio_reports(df_eq, df_tr, df_yearly, options["benchmark_ticker"], options["start_year"])
        return {
            "df_eq": df_eq,
            "df_tr": df_tr,
            "df_yearly": df_yearly,
            "params": params,
            "options": dict(options),
            "context": context,
            "metrics": {
                "total_return": tot_ret,
                "mdd": mdd,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "annual_return_pct": annual_return_pct,
                "bm_annual_return_pct": bm_annual_return_pct,
            },
        }

    def _finish_portfolio_success(self, request_token, result_payload):
        if request_token != self._active_token:
            return
        self._run_thread = None
        self._result = result_payload
        self._render_performance_chart(result_payload)
        self._refresh_trade_ticker_dropdown(result_payload)
        self._status_var.set("完成：投組回測")
        self._notebook.select(1)

    def _finish_portfolio_error(self, request_token, exc):
        if request_token != self._active_token:
            return
        self._run_thread = None
        self._report_runtime_exception("run_portfolio_backtest", exc, status_prefix="投組回測失敗")

    def _refresh_trade_ticker_dropdown(self, result_payload):
        df_tr = result_payload.get("df_tr")
        if df_tr is None or df_tr.empty or "Ticker" not in df_tr.columns:
            self._ticker_map.clear()
            self._ticker_combo.configure(values=[])
            self._ticker_display_var.set("")
            return
        trade_rows = df_tr[df_tr.apply(_is_actual_trade_row, axis=1)].copy()
        if trade_rows.empty:
            values = []
        else:
            order = []
            counts = {}
            for row in trade_rows.to_dict("records"):
                ticker = str(row.get("Ticker", "") or "").strip()
                if not ticker:
                    continue
                if ticker not in counts:
                    order.append(ticker)
                    counts[ticker] = 0
                counts[ticker] += 1
            values = [f"{ticker}|成交 {counts[ticker]}" for ticker in order]
        self._ticker_map = {label: label.split("|", 1)[0].strip() for label in values}
        self._ticker_combo.configure(values=values)
        if values:
            self._ticker_display_var.set(values[0])
            self._render_selected_ticker_chart(values[0])
        else:
            self._ticker_display_var.set("")
            self._clear_kline_chart()
            self._kline_placeholder.configure(text="投組回測完成，但沒有可顯示的成交股票。")

    def _on_ticker_selected(self, _event=None):
        self._render_selected_ticker_chart(self._ticker_display_var.get())

    def _rerender_selected_ticker_chart(self):
        if self._ticker_display_var.get().strip():
            self._render_selected_ticker_chart(self._ticker_display_var.get())

    def _render_selected_ticker_chart(self, display_label):
        if self._result is None:
            return
        ticker = self._ticker_map.get(str(display_label or "").strip())
        if not ticker:
            return
        df_tr = self._result.get("df_tr")
        context = self._result.get("context") or {}
        all_dfs_fast = context.get("all_dfs_fast") or {}
        fast_data = all_dfs_fast.get(ticker)
        if fast_data is None or df_tr is None or df_tr.empty:
            return
        ticker_trades = df_tr[(df_tr["Ticker"].astype(str) == ticker) & df_tr.apply(_is_actual_trade_row, axis=1)].copy()
        chart_payload = _build_portfolio_ticker_chart_payload(
            ticker=ticker,
            fast_data=fast_data,
            ticker_trades_df=ticker_trades,
        )
        self._render_kline_chart({"ticker": ticker, "chart_payload": chart_payload})

    def _render_kline_chart(self, result):
        if FigureCanvasTkAgg is None:
            backend_error_text = "缺少 matplotlib TkAgg backend，無法內嵌圖表。"
            if FIGURE_CANVAS_TKAGG_IMPORT_ERROR:
                backend_error_text = f"{backend_error_text} {FIGURE_CANVAS_TKAGG_IMPORT_ERROR}"
            self._status_var.set(backend_error_text)
            return backend_error_text
        ticker = result.get("ticker", "")
        chart_payload = result.get("chart_payload")
        try:
            figure = create_matplotlib_trade_chart_figure(
                chart_payload=chart_payload,
                ticker=f"{ticker} 投組",
                show_volume=bool(self._show_volume_var.get()),
            )
        except Exception as exc:
            return self._report_runtime_exception("render_portfolio_kline.figure", exc, status_prefix="K線圖渲染失敗")

        try:
            self._clear_kline_chart()
            self._kline_placeholder.pack_forget()
            canvas = FigureCanvasTkAgg(figure, master=self._kline_host)
            bind_matplotlib_chart_navigation(figure, canvas)
            state = getattr(figure, "_stock_chart_navigation_state", None)
            if isinstance(state, dict):
                state["external_hover_callback"] = lambda snapshot: None if snapshot is None else build_chart_hover_snapshot(chart_payload, int(snapshot.get("index", 0))) if isinstance(snapshot, dict) and "index" in snapshot else None
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.configure(background="#02050a", highlightthickness=0, bd=0, takefocus=1)
            widget.pack(fill="both", expand=True)
            widget.focus_set()
        except Exception as exc:
            self._clear_kline_chart()
            try:
                figure.clear()
            except Exception as clear_exc:
                self._append_console_text(f"[render_portfolio_kline.figure.clear]\n{clear_exc}\n")
            return self._report_runtime_exception("render_portfolio_kline.canvas", exc, status_prefix="K線圖嵌入失敗")

        self._chart_canvas = canvas
        self._chart_figure = figure
        self._notebook.select(0)
        scroll_chart_to_latest(self._chart_figure, redraw=True)
        return ""

    def _render_performance_chart(self, result_payload):
        df_eq = result_payload.get("df_eq")
        options = result_payload.get("options") or {}
        if df_eq is None or df_eq.empty:
            self._clear_performance_chart()
            self._performance_placeholder.configure(text="投組回測完成，但沒有績效曲線資料。")
            return
        if FigureCanvasTkAgg is None:
            self._status_var.set("缺少 matplotlib TkAgg backend，無法內嵌績效圖。")
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib import rcParams
            from matplotlib.font_manager import FontProperties
        except ImportError as exc:
            self._report_runtime_exception("render_performance.import", exc, status_prefix="績效圖渲染失敗")
            return

        rcParams["axes.unicode_minus"] = False
        dates = pd.to_datetime(df_eq["Date"])
        benchmark_ticker = options.get("benchmark_ticker", PORTFOLIO_DEFAULT_BENCHMARK_TICKER)
        bm_col = f"Benchmark_{benchmark_ticker}_Pct"
        font_prop = FontProperties(family="Microsoft JhengHei", size=11)
        title_font = FontProperties(family="Microsoft JhengHei", weight="bold", size=16)
        figure = Figure(figsize=(18.2, 10.6), dpi=96, facecolor="#000000")
        axis = figure.add_subplot(1, 1, 1)
        figure.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=0.08)
        axis.set_facecolor("#000000")
        axis.grid(True, color="#0a1824", alpha=0.22, linewidth=0.7)
        axis.plot(dates, df_eq["Strategy_Return_Pct"].astype(float), linewidth=2.5, label="V16 尊爵系統報酬 (%)")
        if bm_col in df_eq.columns:
            axis.plot(dates, df_eq[bm_col].astype(float), linewidth=1.8, label=f"同期大盤 {benchmark_ticker} (%)", alpha=0.88)
        axis.set_title(f"V16 投資組合實戰淨值 vs {benchmark_ticker} 大盤 ({options.get('start_year', '-') } 至今)", color="#f7fbff", fontproperties=title_font)
        axis.set_xlabel("日期", color="#f7fbff", fontproperties=font_prop)
        axis.set_ylabel("累積報酬率 (%)", color="#f7fbff", fontproperties=font_prop)
        axis.tick_params(axis="x", colors="#f7fbff", labelsize=10)
        axis.tick_params(axis="y", colors="#f7fbff", labelsize=10)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#0a1824")
        legend = axis.legend(loc="upper left", frameon=False, prop=font_prop)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#f7fbff")

        self._clear_performance_chart()
        self._performance_placeholder.pack_forget()
        canvas = FigureCanvasTkAgg(figure, master=self._performance_host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(background="#02050a", highlightthickness=0, bd=0)
        widget.pack(fill="both", expand=True)
        self._performance_canvas = canvas
        self._performance_figure = figure

    def _clear_kline_chart(self):
        if self._chart_canvas is not None:
            self._chart_canvas.get_tk_widget().destroy()
            self._chart_canvas = None
        if self._chart_figure is not None:
            self._chart_figure.clear()
            self._chart_figure = None
        if hasattr(self, "_kline_placeholder") and not self._kline_placeholder.winfo_ismapped():
            self._kline_placeholder.pack(fill="both", expand=True)

    def _clear_performance_chart(self):
        if self._performance_canvas is not None:
            self._performance_canvas.get_tk_widget().destroy()
            self._performance_canvas = None
        if self._performance_figure is not None:
            self._performance_figure.clear()
            self._performance_figure = None
        if hasattr(self, "_performance_placeholder") and not self._performance_placeholder.winfo_ismapped():
            self._performance_placeholder.pack(fill="both", expand=True)
