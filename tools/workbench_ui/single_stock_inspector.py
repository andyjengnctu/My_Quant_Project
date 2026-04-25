from __future__ import annotations

import csv
import io
import json
import os
import re
import ssl
import subprocess
import sys
import traceback
import threading
import tkinter as tk
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from html import unescape
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label
from core.output_paths import ensure_output_dir
from core.buy_sort import format_buy_sort_metric_value, get_buy_sort_metric_label, get_buy_sort_method
from core.scanner_display import build_scanner_sort_probe_text
from core.model_paths import resolve_candidate_best_params_path, resolve_run_best_params_path
from tools.trade_analysis.charting import (
    bind_matplotlib_chart_navigation,
    build_chart_hover_snapshot,
    create_matplotlib_trade_chart_figure,
    extract_trade_marker_indexes,
    scroll_chart_to_index,
    scroll_chart_to_latest,
)
from tools.trade_analysis.trade_log import load_params, resolve_trade_analysis_data_dir, run_ticker_analysis
from tools.scanner.scan_runner import run_daily_scanner, run_history_qualified_scanner

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError as exc:  # pragma: no cover - GUI runtime fallback
    FigureCanvasTkAgg = None
    FIGURE_CANVAS_TKAGG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    FIGURE_CANVAS_TKAGG_IMPORT_ERROR = ""


def _warn_gui_fallback(action, exc):
    warnings.warn(f"GUI fallback {action}: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)


class _ConsoleWriter(io.TextIOBase):
    def __init__(self, panel: "SingleStockBacktestInspectorPanel"):
        super().__init__()
        self._panel = panel

    def write(self, text):
        if not text:
            return 0
        self._panel._append_console_stream(str(text))
        return len(text)

    def flush(self):
        return None


BUY_TRADE_TRACE_NAMES = ("買進", "買進(延續候選)")
SIDEBAR_SIGNAL_CHIP_TEXT = "出現買入訊號"
SIDEBAR_HISTORY_CHIP_TEXT = "符合歷史績效"
SIDEBAR_CHIP_ACTIVE_BG = "#2090ff"
SIDEBAR_HISTORY_CHIP_ACTIVE_BG = "#ff8a1c"
SIDEBAR_CHIP_INACTIVE_BG = "#04070c"
WORKBENCH_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKBENCH_OUTPUT_CATEGORY = "workbench_ui"
WORKBENCH_CACHE_FILENAME = "reduced_stock_company_names_cache.json"
PARAM_SOURCE_LABEL_TO_KEY = {
    "run_best | 目前參數": "run_best",
    "candidate_best | 候選參數": "candidate_best",
}
DEFAULT_PARAM_SOURCE_LABEL = "run_best | 目前參數"
COMBOBOX_WIDTH_RULES = {
    "reduced": {"min_chars": 16, "max_chars": 24, "extra_px": 34},
    "candidate": {"min_chars": 18, "max_chars": 44, "extra_px": 24},
    "history": {"min_chars": 18, "max_chars": 44, "extra_px": 24},
    "param_source": {"min_chars": 18, "max_chars": 24, "extra_px": 32},
}
SCAN_DROPDOWN_KIND_LABELS = {
    "buy": "新訊號",
    "extended": "延續",
    "extended_tbd": "延續TBD",
    "candidate": "歷績",
}
SCAN_DROPDOWN_SORT_LABELS = {
    "EV": "EV",
    "預估投入": "投入",
    "勝率×次數": "勝×次",
    "資產成長": "成長",
}
SCAN_DROPDOWN_WIN_RATE_PATTERN = re.compile(r"勝率\s+(-?\d+(?:\.\d+)?)%")
SCAN_DROPDOWN_TRADE_COUNT_PATTERN = re.compile(r"交易\s+([0-9]+)")
SCAN_DROPDOWN_ASSET_GROWTH_PATTERN = re.compile(r"資產成長\s+(-?\d+(?:\.\d+)?)%")

OFFICIAL_COMPANY_NAME_SOURCE_SPECS = (
    {
        "kind": "csv",
        "label": "MOPS 上市公司基本資料",
        "urls": (
            "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv",
            "http://mopsfin.twse.com.tw/opendata/t187ap03_L.csv",
        ),
    },
    {
        "kind": "csv",
        "label": "MOPS 上櫃公司基本資料",
        "urls": (
            "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv",
            "http://mopsfin.twse.com.tw/opendata/t187ap03_O.csv",
        ),
    },
    {
        "kind": "html",
        "label": "ISIN 上市證券名錄",
        "urls": (
            "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
            "http://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        ),
    },
    {
        "kind": "html",
        "label": "ISIN 上櫃證券名錄",
        "urls": (
            "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
            "http://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
        ),
    },
    {
        "kind": "html",
        "label": "ISIN 興櫃證券名錄",
        "urls": (
            "https://isin.twse.com.tw/isin/C_public.jsp?strMode=5",
            "http://isin.twse.com.tw/isin/C_public.jsp?strMode=5",
        ),
    },
    {
        "kind": "html",
        "label": "ISIN 公開發行證券名錄",
        "urls": (
            "https://isin.twse.com.tw/isin/C_public.jsp?strMode=1",
            "http://isin.twse.com.tw/isin/C_public.jsp?strMode=1",
        ),
    },
)
SECURITY_CODE_PATTERN = re.compile(r"^[0-9A-Z]{4,7}$")
ANSI_ESCAPE_SEQUENCE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FALLBACK_REDUCED_STOCK_DISPLAY_NAME_MAP = {
    "0050": "元大台灣50",
    "00631L": "元大台灣50正2",
    "00632R": "元大台灣50反1",
    "00635U": "期元大S&P黃金",
    "00679B": "元大美債20年",
    "1101": "台泥",
    "1216": "統一",
    "2330": "台積電",
    "2603": "長榮",
    "2881": "富邦金",
}


def _normalize_security_code(value):
    return str(value or "").strip().upper()


def _build_workbench_cache_path():
    output_dir = ensure_output_dir(WORKBENCH_PROJECT_ROOT, WORKBENCH_OUTPUT_CATEGORY)
    return os.path.join(output_dir, WORKBENCH_CACHE_FILENAME)


def _load_reduced_stock_company_name_cache():
    cache_path = _build_workbench_cache_path()
    if not os.path.isfile(cache_path):
        return {"ticker_to_name": {}, "reduced_members": []}

    try:
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return {"ticker_to_name": {}, "reduced_members": []}

    ticker_to_name = {
        _normalize_security_code(ticker): str(name).strip()
        for ticker, name in dict(payload.get("ticker_to_name") or {}).items()
        if str(name).strip()
    }
    reduced_members = [
        _normalize_security_code(ticker)
        for ticker in list(payload.get("reduced_members") or [])
        if _normalize_security_code(ticker)
    ]
    return {"ticker_to_name": ticker_to_name, "reduced_members": reduced_members}


def _save_reduced_stock_company_name_cache(*, ticker_to_name, reduced_members):
    cache_path = _build_workbench_cache_path()
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_urls": [url for spec in OFFICIAL_COMPANY_NAME_SOURCE_SPECS for url in spec["urls"]],
        "reduced_members": sorted({_normalize_security_code(ticker) for ticker in reduced_members if _normalize_security_code(ticker)}),
        "ticker_to_name": {
            _normalize_security_code(ticker): str(name).strip()
            for ticker, name in sorted(dict(ticker_to_name).items())
            if _normalize_security_code(ticker) and str(name).strip()
        },
    }
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, ensure_ascii=False, indent=2, sort_keys=True)


def _normalize_company_name(raw_name):
    return re.sub(r"\s+", " ", str(raw_name or "").strip())


def _extract_company_name_pair_from_cells(cells):
    normalized_cells = [_normalize_company_name(cell) for cell in cells if _normalize_company_name(cell)]
    if not normalized_cells:
        return None

    first_cell = normalized_cells[0]
    if first_cell in {"有價證券代號及名稱", "證券代號", "證券名稱", "公司代號", "公司簡稱", "公司名稱"}:
        return None

    combined_match = re.match(r"^([0-9A-Z]{4,7})\s+(.+)$", first_cell)
    if combined_match:
        ticker = _normalize_security_code(combined_match.group(1))
        company_name = _normalize_company_name(combined_match.group(2))
        if SECURITY_CODE_PATTERN.fullmatch(ticker) and company_name:
            return ticker, company_name

    if len(normalized_cells) >= 2:
        ticker = _normalize_security_code(normalized_cells[0])
        company_name = _normalize_company_name(normalized_cells[1])
        if SECURITY_CODE_PATTERN.fullmatch(ticker) and company_name and not SECURITY_CODE_PATTERN.fullmatch(company_name):
            return ticker, company_name
    return None


def _decode_official_text(raw_bytes, *, declared_charset=""):
    candidate_charsets = []
    normalized_declared_charset = str(declared_charset or "").strip()
    if normalized_declared_charset:
        candidate_charsets.append(normalized_declared_charset)
    candidate_charsets.extend(["utf-8-sig", "utf-8", "cp950", "big5", "latin-1"])
    for charset in candidate_charsets:
        try:
            return raw_bytes.decode(charset)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _should_retry_without_ssl_verification(exc):
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, ssl.SSLCertVerificationError)


def _build_http_fallback_url(source_url):
    parts = urlsplit(source_url)
    if parts.scheme != "https":
        return None
    return urlunsplit(("http", parts.netloc, parts.path, parts.query, parts.fragment))


def _fetch_official_text(source_url, *, timeout_seconds=6):
    request_headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/json,text/csv,application/xml;q=0.9,*/*;q=0.8",
    }
    request = Request(source_url, headers=request_headers)
    tried_unverified_ssl = False
    candidates = [source_url]
    http_fallback_url = _build_http_fallback_url(source_url)
    if http_fallback_url and http_fallback_url not in candidates:
        candidates.append(http_fallback_url)

    last_error = None
    for candidate_url in candidates:
        request = Request(candidate_url, headers=request_headers)
        ssl_contexts = [None]
        if candidate_url.startswith("https://"):
            ssl_contexts.append(ssl._create_unverified_context())
        for ssl_context in ssl_contexts:
            if ssl_context is not None and tried_unverified_ssl:
                continue
            try:
                open_kwargs = {"timeout": timeout_seconds}
                if ssl_context is not None:
                    open_kwargs["context"] = ssl_context
                with urlopen(request, **open_kwargs) as response:
                    raw_bytes = response.read()
                    charset = response.headers.get_content_charset() or ""
                if ssl_context is not None:
                    tried_unverified_ssl = True
                return _decode_official_text(raw_bytes, declared_charset=charset)
            except URLError as exc:
                last_error = exc
                if ssl_context is None and candidate_url.startswith("https://") and _should_retry_without_ssl_verification(exc):
                    continue
                break
            except (HTTPError, OSError) as exc:
                last_error = exc
                break
    if last_error is None:
        raise URLError(f"無法讀取官方來源：{source_url}")
    raise last_error


def _extract_company_name_pairs_from_html(html_text):
    tables = pd.read_html(io.StringIO(html_text))
    company_name_map = {}
    for table in tables:
        normalized_columns = [_normalize_company_name(column) for column in list(table.columns)]
        if not any("代號" in column or "名稱" in column for column in normalized_columns):
            continue
        normalized_table = table.copy().fillna("")
        for row in normalized_table.itertuples(index=False):
            pair = _extract_company_name_pair_from_cells(list(row))
            if pair is None:
                continue
            ticker, company_name = pair
            company_name_map.setdefault(ticker, company_name)
    return company_name_map


def _extract_company_name_pairs_from_csv(csv_text):
    reader = csv.DictReader(io.StringIO(csv_text))
    company_name_map = {}
    for row in reader:
        normalized_row = {
            _normalize_company_name(key): _normalize_company_name(value)
            for key, value in dict(row or {}).items()
        }
        ticker = _normalize_security_code(
            normalized_row.get("公司代號")
            or normalized_row.get("公司代碼")
            or normalized_row.get("證券代號")
            or normalized_row.get("股票代號")
        )
        company_name = _normalize_company_name(
            normalized_row.get("公司簡稱")
            or normalized_row.get("公司名稱")
            or normalized_row.get("證券名稱")
            or normalized_row.get("股票名稱")
        )
        if SECURITY_CODE_PATTERN.fullmatch(ticker) and company_name:
            company_name_map.setdefault(ticker, company_name)
    return company_name_map


def _fetch_company_names_from_official_sources(timeout_seconds=6):
    company_name_map = {}
    source_failures = []
    for source_spec in OFFICIAL_COMPANY_NAME_SOURCE_SPECS:
        last_source_error = None
        for source_url in source_spec["urls"]:
            try:
                payload_text = _fetch_official_text(source_url, timeout_seconds=timeout_seconds)
                if source_spec["kind"] == "csv":
                    company_name_map.update(_extract_company_name_pairs_from_csv(payload_text))
                else:
                    company_name_map.update(_extract_company_name_pairs_from_html(unescape(payload_text)))
                last_source_error = None
                break
            except (ValueError, HTTPError, URLError, OSError) as exc:
                last_source_error = exc
        if last_source_error is not None:
            source_failures.append(f"{source_spec['label']}: {type(last_source_error).__name__}: {last_source_error}")
    if not company_name_map:
        raise URLError("; ".join(source_failures) or "所有官方來源皆查詢失敗")
    return company_name_map, source_failures


def _load_reduced_stock_tickers():
    data_dir = get_dataset_dir(WORKBENCH_PROJECT_ROOT, "reduced")
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        _normalize_security_code(os.path.splitext(name)[0])
        for name in os.listdir(data_dir)
        if name.lower().endswith(".csv")
    )


def _build_reduced_stock_dropdown_options(*, company_name_map=None):
    tickers = _load_reduced_stock_tickers()
    normalized_name_map = {
        _normalize_security_code(ticker): _normalize_company_name(name)
        for ticker, name in dict(company_name_map or {}).items()
        if _normalize_company_name(name)
    }
    display_values = []
    display_map = {}
    for ticker in tickers:
        company_name = normalized_name_map.get(ticker, "")
        display_label = f"{ticker} | {company_name}" if company_name else ticker
        display_values.append(display_label)
        display_map[display_label] = ticker
    return tickers, display_values, display_map


class SingleStockBacktestInspectorPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=4, style="Workbench.TFrame")
        self._result = None
        self._status_var = tk.StringVar(value="尚未執行")
        self._ticker_var = tk.StringVar()
        self._reduced_stock_display_var = tk.StringVar()
        self._param_source_display_var = tk.StringVar(value=DEFAULT_PARAM_SOURCE_LABEL)
        self._reduced_stock_map = {}
        self._reduced_stock_company_name_map = {}
        self._show_volume_var = tk.BooleanVar(value=False)
        self._candidate_display_var = tk.StringVar()
        self._candidate_map = {}
        self._history_display_var = tk.StringVar()
        self._history_map = {}
        self._columns = []
        self._chart_canvas = None
        self._chart_figure = None
        self._current_chart_trade_indexes = []
        self._current_chart_trade_cursor_index = None
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
        self._selected_tp_var = tk.StringVar(value="停利: -")
        self._selected_limit_var = tk.StringVar(value="限價: -")
        self._selected_entry_var = tk.StringVar(value="成交: -")
        self._selected_stop_var = tk.StringVar(value="停損: -")
        self._selected_actual_spend_var = tk.StringVar(value="實支: -")
        self._ui_thread = threading.current_thread()
        self._console_stream_buffer = ""
        self._console_stream_mode = "line"
        self._console_live_progress_start = None
        self._analysis_thread = None
        self._analysis_pending_request = None
        self._analysis_active_token = 0
        self._scanner_thread = None
        self._scanner_active_token = 0
        self._company_name_refresh_inflight = False
        self._build_ui()

    def destroy(self):
        self._clear_embedded_chart()
        super().destroy()

    def _build_ui(self):
        controls = ttk.Frame(self, padding=(8, 2, 8, 2), style="Workbench.TFrame")
        controls.pack(fill="x", pady=(0, 4))

        controls_bar = ttk.Frame(controls, style="Workbench.TFrame")
        controls_bar.pack(side="left", anchor="w")
        uniform_pady = (2, 2)

        ttk.Label(controls_bar, text="股票代號", style="Workbench.TLabel").grid(row=0, column=0, padx=(0, 6), pady=uniform_pady, sticky="w")
        ticker_entry = ttk.Entry(controls_bar, textvariable=self._ticker_var, width=12, style="Workbench.TEntry")
        ticker_entry.grid(row=0, column=1, padx=(0, 10), pady=uniform_pady, sticky="w")
        ticker_entry.focus_set()

        ticker_entry.bind("<Return>", self._on_ticker_enter)

        ttk.Label(controls_bar, text="常用股票", style="Workbench.TLabel").grid(row=0, column=2, padx=(0, 6), pady=uniform_pady, sticky="w")
        self._reduced_stock_company_name_map = self._build_initial_reduced_stock_company_name_map()
        _, reduced_display_values, self._reduced_stock_map = _build_reduced_stock_dropdown_options(company_name_map=self._reduced_stock_company_name_map)
        self._reduced_stock_combo = ttk.Combobox(controls_bar, state="readonly", width=18, textvariable=self._reduced_stock_display_var, style="Workbench.TCombobox", values=reduced_display_values)
        self._autosize_combobox(self._reduced_stock_combo, values=reduced_display_values, current_text=self._reduced_stock_display_var.get(), rule_key="reduced")
        self._reduced_stock_combo.grid(row=0, column=3, padx=(0, 12), pady=uniform_pady, sticky="w")
        self._reduced_stock_combo.bind("<<ComboboxSelected>>", self._on_reduced_stock_selected)

        ttk.Button(controls_bar, text="計算候選股", command=self._run_scanner, style="Workbench.TButton").grid(row=0, column=4, padx=(0, 8), pady=uniform_pady, sticky="w")
        self._candidate_combo = ttk.Combobox(controls_bar, state="readonly", width=22, textvariable=self._candidate_display_var, style="Workbench.TCombobox", values=[])
        self._autosize_combobox(self._candidate_combo, values=[], current_text=self._candidate_display_var.get(), rule_key="candidate")
        self._candidate_combo.grid(row=0, column=5, padx=(0, 12), pady=uniform_pady, sticky="w")
        self._candidate_combo.bind("<<ComboboxSelected>>", self._on_candidate_selected)

        ttk.Button(controls_bar, text="計算歷史績效股", command=self._run_history_scanner, style="Workbench.TButton").grid(row=0, column=6, padx=(0, 8), pady=uniform_pady, sticky="w")
        self._history_combo = ttk.Combobox(controls_bar, state="readonly", width=30, textvariable=self._history_display_var, style="Workbench.TCombobox", values=[])
        self._autosize_combobox(self._history_combo, values=[], current_text=self._history_display_var.get(), rule_key="history")
        self._history_combo.grid(row=0, column=7, padx=(0, 14), pady=uniform_pady, sticky="w")
        self._history_combo.bind("<<ComboboxSelected>>", self._on_history_selected)

        ttk.Label(controls_bar, text="參數", style="Workbench.TLabel").grid(row=0, column=8, padx=(0, 6), pady=uniform_pady, sticky="w")
        self._param_source_combo = ttk.Combobox(
            controls_bar,
            state="readonly",
            width=20,
            textvariable=self._param_source_display_var,
            style="Workbench.TCombobox",
            values=list(PARAM_SOURCE_LABEL_TO_KEY.keys()),
        )
        self._autosize_combobox(self._param_source_combo, values=list(PARAM_SOURCE_LABEL_TO_KEY.keys()), current_text=self._param_source_display_var.get(), rule_key="param_source")
        self._param_source_combo.grid(row=0, column=9, padx=(0, 10), pady=uniform_pady, sticky="w")
        self._param_source_combo.bind("<<ComboboxSelected>>", self._on_param_source_selected)
        ttk.Checkbutton(
            controls_bar,
            text="顯示成交量",
            variable=self._show_volume_var,
            command=self._rerender_current_chart,
            style="Workbench.TCheckbutton",
        ).grid(row=0, column=10, padx=(0, 0), pady=uniform_pady, sticky="w")

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

        sidebar_outer = ttk.Frame(chart_tab, padding=(4, 4, 2, 4), width=188, style="Workbench.TFrame")
        sidebar_outer.grid(row=0, column=1, sticky="ns")
        sidebar_outer.grid_propagate(False)
        chart_tab.grid_columnconfigure(1, minsize=188)

        sidebar = ttk.Frame(sidebar_outer, padding=(2, 2), style="Workbench.TFrame")
        sidebar.pack(fill="both", expand=True)
        sidebar.columnconfigure(0, weight=1)
        sidebar_chip_font = ("Microsoft JhengHei", 13, "bold")
        sidebar_header_font = ("Microsoft JhengHei", 13, "bold")
        sidebar_body_font = ("Microsoft JhengHei", 12)
        self._signal_chip = tk.Label(sidebar, textvariable=self._sidebar_signal_var, bg="#04070c", fg="#ffffff", font=sidebar_chip_font, padx=6, pady=4, anchor="center")
        self._signal_chip.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self._history_chip = tk.Label(sidebar, textvariable=self._sidebar_history_var, bg="#04070c", fg="#ffffff", font=sidebar_chip_font, padx=6, pady=4, anchor="center")
        self._history_chip.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(sidebar, text="歷史績效表", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=2, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._sidebar_summary_var, style="Workbench.SidebarSummary.TLabel", font=sidebar_body_font, justify="left", anchor="nw", wraplength=168).grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(sidebar, text="選取日線值", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=4, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_date_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=5, column=0, sticky="w", pady=(2, 0))
        ttk.Label(sidebar, textvariable=self._selected_open_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=6, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_high_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=7, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_low_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=8, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_close_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=9, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_volume_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=10, column=0, sticky="w", pady=(0, 4))
        ttk.Label(sidebar, text="交易資訊", style="Workbench.SidebarHeader.TLabel", font=sidebar_header_font).grid(row=11, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_tp_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=12, column=0, sticky="w", pady=(2, 0))
        ttk.Label(sidebar, textvariable=self._selected_limit_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=13, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_entry_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=14, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_stop_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=15, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self._selected_actual_spend_var, style="Workbench.SidebarValue.TLabel", font=sidebar_body_font, justify="left").grid(row=16, column=0, sticky="w", pady=(0, 4))
        ttk.Button(sidebar, text="回到最新K線", command=self._move_chart_to_latest, style="Workbench.TButton").grid(row=17, column=0, sticky="ew", pady=(4, 0))
        trade_nav = ttk.Frame(sidebar, style="Workbench.TFrame")
        trade_nav.grid(row=18, column=0, sticky="ew", pady=(4, 0))
        trade_nav.columnconfigure(0, weight=1)
        trade_nav.columnconfigure(1, weight=1)
        ttk.Button(trade_nav, text="前交易", command=self._move_chart_to_previous_trade, style="Workbench.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(trade_nav, text="後交易", command=self._move_chart_to_next_trade, style="Workbench.TButton").grid(row=0, column=1, sticky="ew", padx=(2, 0))
        sidebar.rowconfigure(19, weight=1)

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
        self.after_idle(self._refresh_reduced_stock_company_names_if_needed)

    def _get_workbench_combobox_font(self):
        if hasattr(self, "_workbench_combobox_font"):
            return self._workbench_combobox_font
        font_spec = ttk.Style(self).lookup("Workbench.TCombobox", "font") or ("Microsoft JhengHei", 11)
        try:
            self._workbench_combobox_font = tkfont.Font(font=font_spec)
        except tk.TclError as exc:
            _warn_gui_fallback('tkfont.Font(font=Workbench.TCombobox)', exc)
            self._workbench_combobox_font = tkfont.nametofont("TkDefaultFont")
        return self._workbench_combobox_font

    def _autosize_combobox(self, combo, *, values, current_text, rule_key):
        rule = COMBOBOX_WIDTH_RULES[rule_key]
        font_obj = self._get_workbench_combobox_font()
        text_candidates = [str(value or "") for value in list(values or [])]
        text_candidates.append(str(current_text or ""))
        try:
            live_text = combo.get()
        except tk.TclError as exc:
            _warn_gui_fallback('combobox.get() during autosize', exc)
            live_text = ""
        text_candidates.append(str(live_text or ""))
        longest_text = max(text_candidates, key=lambda text: font_obj.measure(text), default="")
        average_char_px = max(font_obj.measure("0"), 1)
        text_px = font_obj.measure(longest_text) + int(rule.get("extra_px") or 0)
        width_chars = max(
            int(rule.get("min_chars") or 0),
            (text_px + average_char_px - 1) // average_char_px,
        )
        width_chars = min(width_chars, int(rule.get("max_chars") or width_chars))
        combo.configure(width=width_chars)

    def _build_initial_reduced_stock_company_name_map(self):
        name_map = dict(FALLBACK_REDUCED_STOCK_DISPLAY_NAME_MAP)
        cache_payload = _load_reduced_stock_company_name_cache()
        name_map.update(dict(cache_payload.get("ticker_to_name") or {}))
        return name_map

    def _apply_reduced_stock_dropdown_values(self, *, preferred_ticker=None):
        current_ticker = _normalize_security_code(preferred_ticker or self._ticker_var.get())
        _, display_values, display_map = _build_reduced_stock_dropdown_options(company_name_map=self._reduced_stock_company_name_map)
        self._reduced_stock_map = display_map
        self._reduced_stock_combo.configure(values=display_values)
        selected_display = ""
        if current_ticker:
            for display_label, ticker in display_map.items():
                if ticker == current_ticker:
                    selected_display = display_label
                    break
        self._reduced_stock_display_var.set(selected_display)
        self._autosize_combobox(self._reduced_stock_combo, values=display_values, current_text=selected_display or current_ticker, rule_key="reduced")

    def _append_reduced_stock_lookup_message(self, message):
        normalized_message = str(message or "").strip()
        if not normalized_message:
            return
        self._append_console_text(f"[常用股票] {normalized_message}\n")

    def _refresh_reduced_stock_company_names_if_needed(self):
        if self._company_name_refresh_inflight:
            return
        reduced_tickers = _load_reduced_stock_tickers()
        current_reduced_set = set(reduced_tickers)
        cache_payload = _load_reduced_stock_company_name_cache()
        cache_ticker_to_name = dict(cache_payload.get("ticker_to_name") or {})
        cached_reduced_members = {
            _normalize_security_code(ticker)
            for ticker in list(cache_payload.get("reduced_members") or [])
            if _normalize_security_code(ticker)
        }
        missing_tickers = sorted(
            ticker for ticker in reduced_tickers
            if not _normalize_company_name(self._reduced_stock_company_name_map.get(ticker) or cache_ticker_to_name.get(ticker))
        )
        reduced_members_changed = current_reduced_set != cached_reduced_members
        if not missing_tickers and not reduced_members_changed:
            return

        if reduced_members_changed:
            self._append_reduced_stock_lookup_message(
                f"偵測到 reduced 代碼組變動，目前 {len(reduced_tickers)} 檔。"
            )
        if missing_tickers:
            self._append_reduced_stock_lookup_message(
                f"缺少中文名稱，嘗試查詢：{', '.join(missing_tickers)}"
            )

        self._company_name_refresh_inflight = True
        refresh_thread = threading.Thread(
            target=self._refresh_reduced_stock_company_names_worker,
            args=(tuple(reduced_tickers), dict(cache_ticker_to_name)),
            name="workbench-reduced-name-refresh",
            daemon=True,
        )
        refresh_thread.start()

    def _refresh_reduced_stock_company_names_worker(self, reduced_tickers, cache_ticker_to_name):
        try:
            fetched_company_name_map, source_failures = _fetch_company_names_from_official_sources()
            updated_ticker_to_name = dict(cache_ticker_to_name)
            newly_resolved_tickers = []
            unresolved_tickers = []
            for ticker in reduced_tickers:
                fetched_company_name = _normalize_company_name(fetched_company_name_map.get(ticker))
                if fetched_company_name:
                    if updated_ticker_to_name.get(ticker) != fetched_company_name:
                        newly_resolved_tickers.append(ticker)
                    updated_ticker_to_name[ticker] = fetched_company_name
                    continue
                if not _normalize_company_name(self._reduced_stock_company_name_map.get(ticker) or updated_ticker_to_name.get(ticker)):
                    unresolved_tickers.append(ticker)
            self.after(
                0,
                self._finish_reduced_stock_company_names_refresh,
                dict(updated_ticker_to_name),
                list(reduced_tickers),
                list(source_failures),
                list(newly_resolved_tickers),
                list(unresolved_tickers),
                None,
            )
        except (ValueError, HTTPError, URLError, OSError) as exc:
            self.after(
                0,
                self._finish_reduced_stock_company_names_refresh,
                None,
                list(reduced_tickers),
                [],
                [],
                [],
                exc,
            )

    def _finish_reduced_stock_company_names_refresh(
        self,
        updated_ticker_to_name,
        reduced_tickers,
        source_failures,
        newly_resolved_tickers,
        unresolved_tickers,
        error,
    ):
        self._company_name_refresh_inflight = False
        if error is not None:
            self._append_reduced_stock_lookup_message(
                f"官方中文名稱查詢失敗，保留既有快取：{type(error).__name__}: {error}"
            )
            return

        try:
            _save_reduced_stock_company_name_cache(
                ticker_to_name=updated_ticker_to_name,
                reduced_members=reduced_tickers,
            )
        except OSError as exc:
            self._append_reduced_stock_lookup_message(
                f"寫回常用股票中文名稱快取失敗：{type(exc).__name__}: {exc}"
            )
            return

        self._reduced_stock_company_name_map.update(dict(updated_ticker_to_name or {}))
        self._apply_reduced_stock_dropdown_values()
        if source_failures:
            self._append_reduced_stock_lookup_message(
                f"部分官方來源查詢失敗，但已改用可用來源完成補名：{' | '.join(source_failures)}"
            )
        if newly_resolved_tickers:
            self._append_reduced_stock_lookup_message(
                f"已更新中文名稱：{', '.join(sorted(set(newly_resolved_tickers)))}"
            )
        if unresolved_tickers:
            self._append_reduced_stock_lookup_message(
                f"仍查不到中文名稱：{', '.join(unresolved_tickers)}"
            )

    def _append_console_text(self, text):
        normalized_text = str(text or "")
        if not normalized_text:
            return
        if threading.current_thread() is not self._ui_thread:
            self.after(0, self._append_console_text, normalized_text)
            return
        self._flush_console_live_progress(force_newline=True)
        self._console_text.insert("end", normalized_text)
        self._console_text.see("end")

    def _append_console_stream(self, text):
        normalized_text = ANSI_ESCAPE_SEQUENCE_PATTERN.sub("", str(text or "")).replace("\r\n", "\n")
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
                    self._console_text.insert("end", current + "\n")
                    self._console_text.see("end")
                current = ""
                mode = "line"
                continue
            current += char
        self._console_stream_buffer = current
        self._console_stream_mode = mode
        if mode == "progress" and not ended_with_carriage_return:
            self._set_console_live_progress(current)

    def _set_console_live_progress(self, text):
        progress_text = str(text or "")
        if self._console_live_progress_start is None:
            self._console_live_progress_start = self._console_text.index("end-1c")
            self._console_text.insert("end", progress_text)
        else:
            self._console_text.delete(self._console_live_progress_start, "end-1c")
            self._console_text.insert(self._console_live_progress_start, progress_text)
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
                self._console_text.insert("end", "\n")
        self._console_live_progress_start = None
        self._console_text.see("end")

    def _clear_console(self):
        self._console_stream_buffer = ""
        self._console_stream_mode = "line"
        self._console_live_progress_start = None
        self._console_text.delete("1.0", "end")

    def _report_runtime_exception(self, context, exc, *, status_prefix, show_dialog=True, switch_to_console=False):
        error_text = f"{status_prefix}：{type(exc).__name__}: {exc}"
        trace_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._append_console_text(f"[{context}]\n{trace_text}\n")
        self._status_var.set(error_text)
        if switch_to_console:
            self._notebook.select(2)
        if show_dialog:
            messagebox.showerror("股票工具工作台", error_text)
        return error_text

    def _get_selected_param_source(self):
        selected_label = self._param_source_display_var.get().strip()
        return PARAM_SOURCE_LABEL_TO_KEY.get(selected_label, "run_best")

    def _get_selected_params_path(self):
        param_source = self._get_selected_param_source()
        if param_source == "candidate_best":
            return resolve_candidate_best_params_path(WORKBENCH_PROJECT_ROOT)
        return resolve_run_best_params_path(WORKBENCH_PROJECT_ROOT)

    def _get_selected_params(self):
        return load_params(self._get_selected_params_path(), verbose=False)

    def _get_selected_param_source_label(self):
        return self._param_source_display_var.get().strip() or DEFAULT_PARAM_SOURCE_LABEL

    def _on_reduced_stock_selected(self, _event=None):
        selected = self._reduced_stock_display_var.get().strip()
        ticker = self._reduced_stock_map.get(selected)
        if ticker:
            self._ticker_var.set(ticker)
            self.after_idle(self._run_analysis)

    def _on_candidate_selected(self, _event=None):
        selected = self._candidate_display_var.get().strip()
        ticker = self._candidate_map.get(selected)
        if ticker:
            self._ticker_var.set(ticker)
            self.after_idle(self._run_analysis)

    def _on_history_selected(self, _event=None):
        selected = self._history_display_var.get().strip()
        ticker = self._history_map.get(selected)
        if ticker:
            self._ticker_var.set(ticker)
            self.after_idle(self._run_analysis)

    def _on_param_source_selected(self, _event=None):
        ticker = self._ticker_var.get().strip()
        if not ticker:
            self._status_var.set(f"參數已切換：{self._get_selected_param_source_label()}")
            return
        self.after_idle(self._run_analysis)

    def _on_ticker_enter(self, _event=None):
        self._run_analysis()

    def _on_sidebar_canvas_configure(self, event=None):
        if event is None or not hasattr(self, "_sidebar_canvas"):
            return
        canvas_width = max(int(event.width), 1)
        self._sidebar_canvas.itemconfigure(self._sidebar_inner_window, width=canvas_width)

    def _on_sidebar_frame_configure(self, _event=None):
        if hasattr(self, "_sidebar_canvas"):
            self._sidebar_canvas.configure(scrollregion=self._sidebar_canvas.bbox("all"))

    @staticmethod
    def _coerce_optional_float(value):
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(numeric_value):
            return None
        return numeric_value

    @staticmethod
    def _coerce_optional_int(value):
        numeric_value = SingleStockBacktestInspectorPanel._coerce_optional_float(value)
        if numeric_value is None:
            return None
        return int(numeric_value)

    @staticmethod
    def _extract_optional_float(pattern, text):
        match = pattern.search(str(text or ""))
        if not match:
            return None
        return SingleStockBacktestInspectorPanel._coerce_optional_float(match.group(1))

    @staticmethod
    def _extract_optional_int(pattern, text):
        numeric_value = SingleStockBacktestInspectorPanel._extract_optional_float(pattern, text)
        if numeric_value is None:
            return None
        return int(numeric_value)

    def _resolve_scan_dropdown_win_rate_pct(self, item):
        win_rate = self._coerce_optional_float(item.get("win_rate"))
        if win_rate is None:
            win_rate = self._extract_optional_float(SCAN_DROPDOWN_WIN_RATE_PATTERN, item.get("text"))
        if win_rate is None:
            return None
        if win_rate <= 1.0:
            return win_rate * 100.0
        return win_rate

    def _resolve_scan_dropdown_trade_count(self, item):
        trade_count = self._coerce_optional_int(item.get("trade_count"))
        if trade_count is None:
            trade_count = self._extract_optional_int(SCAN_DROPDOWN_TRADE_COUNT_PATTERN, item.get("text"))
        return trade_count

    def _resolve_scan_dropdown_asset_growth_pct(self, item):
        asset_growth = self._coerce_optional_float(item.get("asset_growth"))
        if asset_growth is None:
            asset_growth = self._extract_optional_float(SCAN_DROPDOWN_ASSET_GROWTH_PATTERN, item.get("text"))
        return asset_growth

    def _resolve_scan_dropdown_sort_probe_text(self, item, *, sort_method, sort_value, win_rate, trade_count):
        expected_value = self._coerce_optional_float(item.get("expected_value"))
        if expected_value is None:
            expected_value = self._coerce_optional_float(item.get("ev"))
        asset_growth = self._resolve_scan_dropdown_asset_growth_pct(item)
        if any(value is None for value in (expected_value, win_rate, trade_count, asset_growth, sort_value)):
            return ""
        return build_scanner_sort_probe_text(
            ev=expected_value,
            win_rate=win_rate,
            trade_count=trade_count,
            asset_growth_pct=asset_growth,
            sort_value=sort_value,
            method=sort_method,
        )

    def _format_scan_dropdown_label(self, item):
        ticker = str(item.get("ticker") or "").strip()
        kind = str(item.get("kind") or "candidate").strip()
        kind_label = SCAN_DROPDOWN_KIND_LABELS.get(kind, kind or "-")
        sort_method = get_buy_sort_method()
        raw_sort_metric_label = get_buy_sort_metric_label(sort_method)
        sort_metric_label = SCAN_DROPDOWN_SORT_LABELS.get(raw_sort_metric_label, raw_sort_metric_label)
        sort_value = self._coerce_optional_float(item.get("sort_value"))
        win_rate = self._resolve_scan_dropdown_win_rate_pct(item)
        trade_count = self._resolve_scan_dropdown_trade_count(item)
        sort_probe_text = self._resolve_scan_dropdown_sort_probe_text(
            item,
            sort_method=sort_method,
            sort_value=sort_value,
            win_rate=win_rate,
            trade_count=trade_count,
        )
        sort_value_text = "-" if sort_value is None else format_buy_sort_metric_value(sort_value, sort_method)
        win_rate_text = "-" if win_rate is None else f"{win_rate:.1f}%"
        trade_count_text = "-" if trade_count is None else str(trade_count)
        if sort_probe_text:
            sort_probe_payload = {"text": sort_probe_text}
            probe_win_rate = self._resolve_scan_dropdown_win_rate_pct(sort_probe_payload)
            probe_trade_count = self._resolve_scan_dropdown_trade_count(sort_probe_payload)
            win_rate_text = "-" if probe_win_rate is None else f"{probe_win_rate:.1f}%"
            trade_count_text = "-" if probe_trade_count is None else str(probe_trade_count)
        return f"{ticker}|{kind_label}|{sort_metric_label} {sort_value_text}|勝率 {win_rate_text}|次 {trade_count_text}"

    def _apply_scan_dropdown(self, *, combo, value_var, mapping, display_values, rule_key):
        mapping.clear()
        combo.configure(values=display_values)
        if display_values:
            value_var.set(display_values[0])
            mapping.update({label: label.split("|", 1)[0].strip() for label in display_values})
            self._ticker_var.set(mapping[display_values[0]])
            self._autosize_combobox(combo, values=display_values, current_text=display_values[0], rule_key=rule_key)
            return
        value_var.set("")
        self._autosize_combobox(combo, values=[], current_text="", rule_key=rule_key)

    def _run_scanner_worker(self, mode, params_path, request_token):
        try:
            data_dir = resolve_trade_analysis_data_dir(DEFAULT_DATASET_PROFILE)
            params = load_params(params_path, verbose=False)
            with redirect_stdout(self._console_writer), redirect_stderr(self._console_writer):
                if mode == "candidate":
                    scan_result = run_daily_scanner(data_dir, params)
                else:
                    scan_result = run_history_qualified_scanner(data_dir, params)
        except Exception as exc:
            self.after(0, self._finish_scanner_error, mode, request_token, exc)
            return
        self.after(0, self._finish_scanner_success, mode, request_token, scan_result)

    def _start_scanner(self, mode):
        if self._scanner_thread is not None and self._scanner_thread.is_alive():
            status_text = "掃描進行中：請等待目前掃描完成"
            self._status_var.set(status_text)
            self._append_console_text(f"[scanner] {status_text}\n")
            return
        self._scanner_active_token += 1
        request_token = self._scanner_active_token
        params_path = self._get_selected_params_path()
        self._clear_console()
        self._notebook.select(2)
        param_source = self._get_selected_param_source_label()
        status_text = (
            f"執行中：掃描候選股 ({param_source})"
            if mode == "candidate"
            else f"執行中：掃描歷史績效股 ({param_source})"
        )
        self._status_var.set(status_text)
        self._append_console_text(f"[scanner] {status_text}\n")
        scanner_thread = threading.Thread(
            target=self._run_scanner_worker,
            args=(mode, params_path, request_token),
            name=f"workbench-scanner-{mode}",
            daemon=True,
        )
        self._scanner_thread = scanner_thread
        scanner_thread.start()

    def _finish_scanner_success(self, mode, request_token, scan_result):
        if request_token != self._scanner_active_token:
            return
        self._scanner_thread = None
        if mode == "candidate":
            candidate_rows = list((scan_result or {}).get("candidate_rows") or [])
            candidate_rows.sort(key=lambda item: (item.get("sort_value") or 0.0, item.get("ticker") or ""), reverse=True)
            display_values = [self._format_scan_dropdown_label(item) for item in candidate_rows]
            self._apply_scan_dropdown(
                combo=self._candidate_combo,
                value_var=self._candidate_display_var,
                mapping=self._candidate_map,
                display_values=display_values,
                rule_key="candidate",
            )
            self._status_var.set(f"掃描完成：候選股 {len(display_values)} 檔")
            return

        history_rows = list((scan_result or {}).get("history_qualified_rows") or [])
        history_rows.sort(key=lambda item: (item.get("sort_value") or 0.0, item.get("ticker") or ""), reverse=True)
        display_values = [self._format_scan_dropdown_label(item) for item in history_rows]
        self._apply_scan_dropdown(
            combo=self._history_combo,
            value_var=self._history_display_var,
            mapping=self._history_map,
            display_values=display_values,
            rule_key="history",
        )
        self._status_var.set(f"掃描完成：歷史績效股 {len(display_values)} 檔")

    def _finish_scanner_error(self, mode, request_token, exc):
        if request_token != self._scanner_active_token:
            return
        self._scanner_thread = None
        context = "run_scanner" if mode == "candidate" else "run_history_scanner"
        self._report_runtime_exception(context, exc, status_prefix="掃描失敗", switch_to_console=True)

    def _run_scanner(self):
        self._start_scanner("candidate")

    def _run_history_scanner(self):
        self._start_scanner("history")

    def _run_analysis_worker(self, ticker, params_path, request_token):
        try:
            result = run_ticker_analysis(
                ticker,
                dataset_profile_key=DEFAULT_DATASET_PROFILE,
                params=load_params(params_path, verbose=False),
                export_excel=True,
                export_chart=False,
                return_chart_payload=True,
                verbose=False,
            )
        except Exception as exc:
            self.after(0, self._finish_analysis_error, ticker, params_path, request_token, exc)
            return
        self.after(0, self._finish_analysis_success, ticker, params_path, request_token, result)

    def _start_analysis(self, ticker, *, params_path=None):
        ticker = str(ticker or "").strip()
        if not ticker:
            return
        resolved_params_path = str(params_path or self._get_selected_params_path())
        self._analysis_active_token += 1
        request_token = self._analysis_active_token
        self._status_var.set(f"執行中：{ticker} / {get_dataset_profile_label(DEFAULT_DATASET_PROFILE)}")
        analysis_thread = threading.Thread(
            target=self._run_analysis_worker,
            args=(ticker, resolved_params_path, request_token),
            name=f"workbench-analysis-{ticker}",
            daemon=True,
        )
        self._analysis_thread = analysis_thread
        analysis_thread.start()

    def _consume_pending_analysis_request(self, completed_ticker, completed_params_path):
        pending_request = self._analysis_pending_request
        self._analysis_pending_request = None
        if not pending_request:
            return
        pending_ticker, pending_params_path = pending_request
        if (
            str(pending_ticker or "").strip() == str(completed_ticker or "").strip()
            and str(pending_params_path or "").strip() == str(completed_params_path or "").strip()
        ):
            return
        self.after_idle(lambda ticker=pending_ticker, path=pending_params_path: self._start_analysis(ticker, params_path=path))

    def _finish_analysis_success(self, ticker, params_path, request_token, result):
        if request_token != self._analysis_active_token:
            return
        self._analysis_thread = None
        self._result = result
        render_error_text = self._render_result(result)
        if not render_error_text:
            self._status_var.set(f"完成：{ticker} / {get_dataset_profile_label(DEFAULT_DATASET_PROFILE)}")
        self._consume_pending_analysis_request(ticker, params_path)

    def _finish_analysis_error(self, ticker, params_path, request_token, exc):
        if request_token != self._analysis_active_token:
            return
        self._analysis_thread = None
        self._report_runtime_exception("run_analysis", exc, status_prefix="執行失敗")
        self._consume_pending_analysis_request(ticker, params_path)

    def _run_analysis(self):
        ticker = self._ticker_var.get().strip()
        if not ticker:
            messagebox.showerror("股票工具工作台", "請先輸入股票代號。")
            return

        params_path = self._get_selected_params_path()
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            self._analysis_pending_request = (ticker, params_path)
            self._status_var.set(
                f"查股進行中，已排入最新請求：{ticker} / {get_dataset_profile_label(DEFAULT_DATASET_PROFILE)}"
            )
            return

        self._analysis_pending_request = None
        self._start_analysis(ticker, params_path=params_path)

    def _render_result(self, result):
        trade_logs_df = result.get("trade_logs_df")
        self._update_sidebar_from_result(result)
        self._render_trade_table(trade_logs_df)
        return self._render_embedded_chart(result)

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
            self._selected_tp_var.set("停利: -")
            self._selected_limit_var.set("限價: -")
            self._selected_entry_var.set("成交: -")
            self._selected_stop_var.set("停損: -")
            self._selected_actual_spend_var.set("實支: -")
            return
        self._selected_date_var.set(f"選取日: {snapshot.get('date_label', '-')}")
        self._selected_open_var.set(self._format_sidebar_ohlcv_value("開", snapshot.get("open")))
        self._selected_high_var.set(self._format_sidebar_ohlcv_value("高", snapshot.get("high")))
        self._selected_low_var.set(self._format_sidebar_ohlcv_value("低", snapshot.get("low")))
        self._selected_close_var.set(self._format_sidebar_ohlcv_value("收", snapshot.get("close")))
        self._selected_volume_var.set(self._format_sidebar_ohlcv_value("量", snapshot.get("volume"), volume=True))
        self._selected_tp_var.set(self._format_sidebar_line_value("停利", snapshot.get("tp_price")))
        self._selected_limit_var.set(self._format_sidebar_line_value("限價", snapshot.get("limit_price")))
        self._selected_entry_var.set(self._format_sidebar_line_value("成交", snapshot.get("entry_price")))
        self._selected_stop_var.set(self._format_sidebar_line_value("停損", snapshot.get("stop_price")))
        self._selected_actual_spend_var.set(self._format_sidebar_amount_value("實支", snapshot.get("buy_capital")))

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
        if scroll_chart_to_latest(self._chart_figure, redraw=True):
            state = getattr(self._chart_figure, "_stock_chart_navigation_state", None)
            if isinstance(state, dict):
                self._current_chart_trade_cursor_index = int(state.get("hover_last_index", 0) or 0)

    def _move_chart_to_previous_trade(self):
        self._move_chart_to_trade(direction=-1)

    def _move_chart_to_next_trade(self):
        self._move_chart_to_trade(direction=1)

    def _move_chart_to_trade(self, *, direction):
        if self._chart_figure is None or not self._current_chart_trade_indexes:
            return False
        state = getattr(self._chart_figure, "_stock_chart_navigation_state", None)
        current_index = None
        if isinstance(state, dict):
            current_index = state.get("hover_last_index")
        if current_index is None:
            current_index = self._current_chart_trade_cursor_index
        if current_index is None:
            current_index = self._current_chart_trade_indexes[-1]
        current_index = int(current_index)

        buy_indexes = sorted(set(int(idx) for idx in self._current_chart_trade_indexes))
        if int(direction) < 0:
            candidates = [idx for idx in buy_indexes if idx < current_index]
            target_index = candidates[-1] if candidates else buy_indexes[0]
        else:
            candidates = [idx for idx in buy_indexes if idx > current_index]
            target_index = candidates[0] if candidates else buy_indexes[-1]
        if scroll_chart_to_index(self._chart_figure, target_index, redraw=True):
            self._current_chart_trade_cursor_index = int(target_index)
            return True
        return False

    def _render_embedded_chart(self, result):
        chart_payload = result.get("chart_payload")
        ticker = result.get("ticker", "")
        if chart_payload is None:
            self._clear_embedded_chart()
            self._append_console_text(f"[render_embedded_chart] ticker={ticker or '-'} | chart_payload=None\n")
            self._status_var.set(f"圖表缺失：{ticker or '-'} 沒有 chart_payload")
            return self._status_var.get()
        if FigureCanvasTkAgg is None:
            self._clear_embedded_chart()
            backend_error_text = "缺少 matplotlib TkAgg backend，無法內嵌圖表。"
            if FIGURE_CANVAS_TKAGG_IMPORT_ERROR:
                backend_error_text = f"{backend_error_text} {FIGURE_CANVAS_TKAGG_IMPORT_ERROR}"
            self._status_var.set(backend_error_text)
            return backend_error_text
        trade_indexes = extract_trade_marker_indexes(chart_payload, trace_names=BUY_TRADE_TRACE_NAMES)
        self._current_chart_trade_indexes = trade_indexes
        self._current_chart_trade_cursor_index = None
        payload_bar_count = 0
        payload_dates = chart_payload.get("date_labels") if isinstance(chart_payload, dict) else None
        if payload_dates is not None:
            try:
                payload_bar_count = len(payload_dates)
            except TypeError:
                payload_bar_count = 0
        self._append_console_text(
            f"[render_embedded_chart] ticker={ticker or '-'} | chart_payload_bars={payload_bar_count} | show_volume={int(bool(self._show_volume_var.get()))}\n"
        )
        try:
            figure = create_matplotlib_trade_chart_figure(chart_payload=self._build_gui_chart_payload(result), ticker=ticker, show_volume=bool(self._show_volume_var.get()))
        except Exception as exc:
            self._clear_embedded_chart()
            error_text = self._report_runtime_exception("render_embedded_chart.figure", exc, status_prefix="圖表渲染失敗")
            return error_text

        try:
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
        except Exception as exc:
            self._clear_embedded_chart()
            try:
                figure.clear()
            except Exception as clear_exc:
                clear_trace = "".join(traceback.format_exception(type(clear_exc), clear_exc, clear_exc.__traceback__))
                self._append_console_text(f"[render_embedded_chart.figure.clear]\n{clear_trace}\n")
            error_text = self._report_runtime_exception("render_embedded_chart.canvas", exc, status_prefix="圖表嵌入失敗")
            return error_text

        self._chart_canvas = canvas
        self._chart_figure = figure
        self._current_chart_trade_indexes = trade_indexes
        self._current_chart_trade_cursor_index = None
        self._notebook.select(0)
        self._move_chart_to_latest()
        return ""

    def _clear_embedded_chart(self):
        if self._chart_canvas is not None:
            widget = self._chart_canvas.get_tk_widget()
            widget.destroy()
            self._chart_canvas = None
        if self._chart_figure is not None:
            self._chart_figure.clear()
            self._chart_figure = None
        self._current_chart_trade_indexes = []
        self._current_chart_trade_cursor_index = None
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
            self._report_runtime_exception("open_path", exc, status_prefix="開啟失敗", show_dialog=True)
