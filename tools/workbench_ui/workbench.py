from __future__ import annotations

import importlib
import tkinter as tk
import warnings
from tkinter import ttk


WORKBENCH_TITLE = "股票工具工作台"
WORKBENCH_GEOMETRY = "1920x1180"
WORKBENCH_DEFAULT_MIN_WIDTH = 1520
WORKBENCH_DEFAULT_MIN_HEIGHT = 920
WORKBENCH_SAFE_MIN_WIDTH = 900
WORKBENCH_SAFE_MIN_HEIGHT = 650
WORKBENCH_SCREEN_MARGIN_X = 40
WORKBENCH_SCREEN_MARGIN_Y = 80
WORKBENCH_BG = "#05090e"
WORKBENCH_SURFACE = "#0a1220"
WORKBENCH_SURFACE_ALT = "#0d1626"
WORKBENCH_BORDER = "#18324a"
WORKBENCH_TEXT = "#f7fbff"
WORKBENCH_MUTED = "#d6dfeb"
WORKBENCH_ACCENT = "#2d7ff9"
WORKBENCH_SIDEBAR_TITLE_BLUE = "#2d7ff9"
WORKBENCH_FRAME_STYLE = "Workbench.TFrame"
WORKBENCH_LABELLF_STYLE = "Workbench.TLabelframe"
WORKBENCH_LABEL_STYLE = "Workbench.TLabel"
WORKBENCH_BUTTON_STYLE = "Workbench.TButton"
WORKBENCH_CHECK_STYLE = "Workbench.TCheckbutton"
WORKBENCH_ENTRY_STYLE = "Workbench.TEntry"
WORKBENCH_COMBO_STYLE = "Workbench.TCombobox"
WORKBENCH_NOTEBOOK_STYLE = "Workbench.TNotebook"
WORKBENCH_TREE_STYLE = "Workbench.Treeview"
WORKBENCH_VSCROLL_STYLE = "Workbench.Vertical.TScrollbar"
WORKBENCH_HSCROLL_STYLE = "Workbench.Horizontal.TScrollbar"
WORKBENCH_SIDEBAR_SIGNAL_STYLE = "Workbench.SidebarSignal.TLabel"
WORKBENCH_SIDEBAR_GATE_STYLE = "Workbench.SidebarGate.TLabel"
WORKBENCH_SIDEBAR_HEADER_STYLE = "Workbench.SidebarHeader.TLabel"
WORKBENCH_SIDEBAR_SUMMARY_STYLE = "Workbench.SidebarSummary.TLabel"
WORKBENCH_SIDEBAR_VALUE_STYLE = "Workbench.SidebarValue.TLabel"
WORKBENCH_SIDEBAR_BUTTON_STYLE = "Workbench.Sidebar.TButton"
WORKBENCH_UI_FONT = ("Microsoft JhengHei", 11)
WORKBENCH_NOTEBOOK_FONT = ("Microsoft JhengHei", 11)
WORKBENCH_RIGHT_SIDEBAR_WIDTH = 195
WORKBENCH_RIGHT_SIDEBAR_WRAPLENGTH = 182
WORKBENCH_RIGHT_SIDEBAR_FONT_SCALE = 0.81


def _scale_right_sidebar_font_size(base_size):
    return max(1, int(round(float(base_size) * WORKBENCH_RIGHT_SIDEBAR_FONT_SCALE)))


WORKBENCH_RIGHT_SIDEBAR_BUTTON_FONT = ("Microsoft JhengHei", _scale_right_sidebar_font_size(13))
WORKBENCH_RIGHT_SIDEBAR_CHIP_FONT = ("Microsoft JhengHei", _scale_right_sidebar_font_size(15), "bold")
WORKBENCH_RIGHT_SIDEBAR_HEADER_FONT = ("Microsoft JhengHei", _scale_right_sidebar_font_size(15), "bold")
WORKBENCH_RIGHT_SIDEBAR_BODY_FONT = ("Microsoft JhengHei", _scale_right_sidebar_font_size(14))


def _warn_gui_fallback(action, exc):
    warnings.warn(f"GUI fallback {action}: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)


def _load_panel_factory(factory_path):
    module_name, _, attr_name = str(factory_path).partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"不合法的 panel_factory_path: {factory_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


PANEL_SPECS = (
    {
        "panel_id": "single_stock_backtest_inspector",
        "tab_label": "單股回測檢視",
        "backend_runner": "tools.trade_analysis.trade_log.run_ticker_analysis",
        "artifact_keys": ("excel_path",),
        "inline_chart_backend": "tools.trade_analysis.charting.create_matplotlib_trade_chart_figure",
        "default_show_volume": False,
        "jump_to_trade_enabled": True,
        "panel_factory_path": "tools.workbench_ui.single_stock_inspector:SingleStockBacktestInspectorPanel",
    },
    {
        "panel_id": "portfolio_backtest_inspector",
        "tab_label": "投組回測檢視",
        "backend_runner": "tools.portfolio_sim.simulation_runner.run_portfolio_simulation_prepared",
        "artifact_keys": ("dashboard_html_path", "report_xlsx_path"),
        "inline_chart_backend": "tools.trade_analysis.charting.create_matplotlib_trade_chart_figure",
        "default_show_volume": False,
        "jump_to_trade_enabled": True,
        "panel_factory_path": "tools.workbench_ui.portfolio_backtest_inspector:PortfolioBacktestInspectorPanel",
    },
)


def build_workbench_spec():
    return {
        "app_id": "stock_tools_workbench",
        "entry_module": "apps.workbench",
        "title": WORKBENCH_TITLE,
        "geometry": WORKBENCH_GEOMETRY,
        "startup_window_mode": "maximized",
        "ui_theme": "deep_dark",
        "panels": [
            {
                "panel_id": panel["panel_id"],
                "tab_label": panel["tab_label"],
                "backend_runner": panel["backend_runner"],
                "artifact_keys": list(panel["artifact_keys"]),
                "inline_chart_backend": panel["inline_chart_backend"],
                "default_show_volume": panel["default_show_volume"],
                "default_dataset": "full",
                "scanner_dropdown_enabled": True,
                "console_tab_enabled": True,
                "jump_to_latest_enabled": True,
                "jump_to_trade_enabled": bool(panel.get("jump_to_trade_enabled", False)),
            }
            for panel in PANEL_SPECS
        ],
    }


def configure_workbench_theme(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError as exc:
        _warn_gui_fallback('style.theme_use("clam")', exc)
    root.configure(bg=WORKBENCH_BG)
    try:
        root.tk_setPalette(background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, activeBackground=WORKBENCH_SURFACE_ALT, activeForeground=WORKBENCH_TEXT, highlightColor=WORKBENCH_ACCENT, selectColor=WORKBENCH_ACCENT, selectBackground=WORKBENCH_ACCENT, selectForeground=WORKBENCH_TEXT)
    except tk.TclError as exc:
        _warn_gui_fallback('root.tk_setPalette(...)', exc)
    root.option_add("*Background", WORKBENCH_BG)
    root.option_add("*Foreground", WORKBENCH_TEXT)
    root.option_add("*TCombobox*Listbox.background", WORKBENCH_SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", WORKBENCH_TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", WORKBENCH_ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", WORKBENCH_TEXT)

    style.configure(".", background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, fieldbackground=WORKBENCH_SURFACE, bordercolor=WORKBENCH_BORDER, lightcolor=WORKBENCH_BORDER, darkcolor=WORKBENCH_BORDER, troughcolor=WORKBENCH_BG)
    style.configure(WORKBENCH_FRAME_STYLE, background=WORKBENCH_BG)
    style.configure(WORKBENCH_LABELLF_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, bordercolor=WORKBENCH_BORDER)
    style.configure(f"{WORKBENCH_LABELLF_STYLE}.Label", background=WORKBENCH_BG, foreground=WORKBENCH_TEXT)
    style.configure(WORKBENCH_LABEL_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, font=WORKBENCH_UI_FONT)
    style.configure(WORKBENCH_BUTTON_STYLE, background=WORKBENCH_SURFACE_ALT, foreground=WORKBENCH_TEXT, bordercolor=WORKBENCH_BORDER, focusthickness=1, focuscolor=WORKBENCH_BORDER, padding=(10, 4), font=WORKBENCH_UI_FONT)
    style.map(WORKBENCH_BUTTON_STYLE, background=[("active", WORKBENCH_ACCENT), ("pressed", WORKBENCH_ACCENT)])
    style.configure(WORKBENCH_SIDEBAR_BUTTON_STYLE, background=WORKBENCH_SURFACE_ALT, foreground=WORKBENCH_TEXT, bordercolor=WORKBENCH_BORDER, focusthickness=1, focuscolor=WORKBENCH_BORDER, padding=(2, 3), font=WORKBENCH_RIGHT_SIDEBAR_BUTTON_FONT)
    style.map(WORKBENCH_SIDEBAR_BUTTON_STYLE, background=[("active", WORKBENCH_ACCENT), ("pressed", WORKBENCH_ACCENT)])
    style.configure(WORKBENCH_CHECK_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, font=WORKBENCH_UI_FONT, padding=(0, 2))
    style.map(WORKBENCH_CHECK_STYLE, foreground=[("active", WORKBENCH_TEXT)])
    style.configure(WORKBENCH_ENTRY_STYLE, fieldbackground=WORKBENCH_SURFACE, foreground=WORKBENCH_TEXT, insertcolor=WORKBENCH_TEXT, font=WORKBENCH_UI_FONT)
    style.configure(WORKBENCH_COMBO_STYLE, fieldbackground=WORKBENCH_SURFACE, foreground=WORKBENCH_TEXT, arrowcolor=WORKBENCH_TEXT, font=WORKBENCH_UI_FONT)
    style.map(
        WORKBENCH_COMBO_STYLE,
        fieldbackground=[("readonly", WORKBENCH_SURFACE), ("disabled", WORKBENCH_SURFACE_ALT)],
        foreground=[("readonly", WORKBENCH_TEXT), ("disabled", WORKBENCH_MUTED)],
        selectbackground=[("readonly", WORKBENCH_SURFACE)],
        selectforeground=[("readonly", WORKBENCH_TEXT)],
        arrowcolor=[("readonly", WORKBENCH_TEXT), ("disabled", WORKBENCH_MUTED)],
    )
    style.configure(WORKBENCH_NOTEBOOK_STYLE, background=WORKBENCH_BG, borderwidth=0)
    style.configure(f"{WORKBENCH_NOTEBOOK_STYLE}.Tab", background=WORKBENCH_SURFACE_ALT, foreground=WORKBENCH_TEXT, padding=(10, 5), font=WORKBENCH_NOTEBOOK_FONT)
    style.map(f"{WORKBENCH_NOTEBOOK_STYLE}.Tab", background=[("selected", WORKBENCH_ACCENT)], foreground=[("selected", WORKBENCH_TEXT)])
    style.configure(WORKBENCH_TREE_STYLE, background=WORKBENCH_SURFACE, fieldbackground=WORKBENCH_SURFACE, foreground=WORKBENCH_TEXT, bordercolor=WORKBENCH_BORDER)
    style.configure(f"{WORKBENCH_TREE_STYLE}.Heading", background=WORKBENCH_SURFACE_ALT, foreground=WORKBENCH_TEXT, font=WORKBENCH_UI_FONT)
    style.map(WORKBENCH_TREE_STYLE, background=[("selected", WORKBENCH_ACCENT)], foreground=[("selected", WORKBENCH_TEXT)])
    style.configure(WORKBENCH_VSCROLL_STYLE, background=WORKBENCH_SURFACE_ALT, troughcolor=WORKBENCH_BG, arrowcolor=WORKBENCH_TEXT)
    style.configure(WORKBENCH_HSCROLL_STYLE, background=WORKBENCH_SURFACE_ALT, troughcolor=WORKBENCH_BG, arrowcolor=WORKBENCH_TEXT)
    style.configure(WORKBENCH_SIDEBAR_SIGNAL_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, anchor="center", padding=(10, 8), font=WORKBENCH_RIGHT_SIDEBAR_CHIP_FONT)
    style.configure(WORKBENCH_SIDEBAR_GATE_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, anchor="center", padding=(10, 8), font=WORKBENCH_RIGHT_SIDEBAR_CHIP_FONT)
    style.configure(WORKBENCH_SIDEBAR_HEADER_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_SIDEBAR_TITLE_BLUE, font=WORKBENCH_RIGHT_SIDEBAR_HEADER_FONT)
    style.configure(WORKBENCH_SIDEBAR_SUMMARY_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, font=WORKBENCH_RIGHT_SIDEBAR_BODY_FONT, anchor="nw")
    style.configure(WORKBENCH_SIDEBAR_VALUE_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, font=WORKBENCH_RIGHT_SIDEBAR_BODY_FONT)

def _resolve_screen_limited_size(root, *, desired_width, desired_height):
    screen_width = max(1, int(root.winfo_screenwidth()))
    screen_height = max(1, int(root.winfo_screenheight()))
    available_width = max(1, screen_width - WORKBENCH_SCREEN_MARGIN_X)
    available_height = max(1, screen_height - WORKBENCH_SCREEN_MARGIN_Y)
    return min(int(desired_width), available_width), min(int(desired_height), available_height)


def _resolve_screen_limited_min_size(root):
    min_width, min_height = _resolve_screen_limited_size(
        root,
        desired_width=WORKBENCH_DEFAULT_MIN_WIDTH,
        desired_height=WORKBENCH_DEFAULT_MIN_HEIGHT,
    )
    if min_width >= WORKBENCH_SAFE_MIN_WIDTH:
        min_width = max(WORKBENCH_SAFE_MIN_WIDTH, min_width)
    if min_height >= WORKBENCH_SAFE_MIN_HEIGHT:
        min_height = max(WORKBENCH_SAFE_MIN_HEIGHT, min_height)
    return min_width, min_height


def _apply_responsive_window_size(root):
    initial_width, initial_height = _resolve_screen_limited_size(root, desired_width=1920, desired_height=1180)
    root.geometry(f"{initial_width}x{initial_height}+0+0")
    root.minsize(*_resolve_screen_limited_min_size(root))


def _maximize_root_window(root):
    try:
        root.state("zoomed")
        return "zoomed"
    except tk.TclError as exc:
        _warn_gui_fallback('root.state("zoomed")', exc)

    try:
        root.attributes("-zoomed", True)
        return "attributes-zoomed"
    except tk.TclError as exc:
        _warn_gui_fallback('root.attributes("-zoomed", True)', exc)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    return "geometry-screen"


def build_workbench_scrollable_sidebar(master, *, width, row=0, column=1, outer_padding=(0, 3, 0, 3), inner_padding=(0, 2)):
    outer = ttk.Frame(master, padding=outer_padding, width=width, style=WORKBENCH_FRAME_STYLE)
    outer.grid(row=row, column=column, sticky="ns")
    outer.grid_propagate(False)
    outer.pack_propagate(False)
    outer.rowconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)

    canvas = tk.Canvas(outer, bg=WORKBENCH_BG, highlightthickness=0, bd=0)
    y_scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style=WORKBENCH_VSCROLL_STYLE)
    canvas.configure(yscrollcommand=y_scroll.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")

    inner = ttk.Frame(canvas, padding=inner_padding, style=WORKBENCH_FRAME_STYLE)
    inner_window = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _refresh_scrollregion(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _sync_inner_width(event=None):
        canvas_width = 1 if event is None else max(1, int(event.width))
        canvas.itemconfigure(inner_window, width=canvas_width)

    def _on_mousewheel(event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120) if event.delta else 0
        if delta:
            canvas.yview_scroll(delta, "units")

    inner.bind("<Configure>", _refresh_scrollregion)
    canvas.bind("<Configure>", _sync_inner_width)
    for widget in (canvas, inner):
        widget.bind("<MouseWheel>", _on_mousewheel, add="+")
        widget.bind("<Button-4>", _on_mousewheel, add="+")
        widget.bind("<Button-5>", _on_mousewheel, add="+")

    outer._workbench_sidebar_canvas = canvas
    outer._workbench_sidebar_inner = inner
    return outer, inner


WORKBENCH_SELECTED_OHLCV_UP_OR_FLAT_ORDER = ("date", "high", "close", "open", "low", "volume")
WORKBENCH_SELECTED_OHLCV_DOWN_ORDER = ("date", "high", "open", "close", "low", "volume")
WORKBENCH_CAPITAL_MODE_RESERVED = "reserved"
WORKBENCH_CAPITAL_MODE_ACTUAL = "actual"


def _coerce_optional_float_for_sidebar(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def resolve_workbench_selected_ohlcv_order(*, open_value, close_value):
    open_number = _coerce_optional_float_for_sidebar(open_value)
    close_number = _coerce_optional_float_for_sidebar(close_value)
    if open_number is not None and close_number is not None and close_number < open_number:
        return WORKBENCH_SELECTED_OHLCV_DOWN_ORDER
    return WORKBENCH_SELECTED_OHLCV_UP_OR_FLAT_ORDER


def grid_workbench_selected_ohlcv_labels(label_widgets, *, open_value=None, close_value=None, start_row=5):
    order = resolve_workbench_selected_ohlcv_order(open_value=open_value, close_value=close_value)
    for widget in label_widgets.values():
        widget.grid_forget()
    for offset, key in enumerate(order):
        widget = label_widgets.get(key)
        if widget is None:
            continue
        pady = (2, 0) if offset == 0 else ((0, 4) if offset == len(order) - 1 else (0, 0))
        widget.grid(row=start_row + offset, column=0, sticky="w", pady=pady)
    return order


def refresh_workbench_capital_display(target):
    mode = getattr(target, "_selected_capital_display_mode", WORKBENCH_CAPITAL_MODE_RESERVED)
    if mode not in {WORKBENCH_CAPITAL_MODE_RESERVED, WORKBENCH_CAPITAL_MODE_ACTUAL}:
        mode = WORKBENCH_CAPITAL_MODE_RESERVED
        target._selected_capital_display_mode = mode
    reserved_text = getattr(target, "_selected_reserved_capital_text", "預留: -")
    actual_text = getattr(target, "_selected_actual_spend_text", "實支: -")
    target._selected_capital_var.set(actual_text if mode == WORKBENCH_CAPITAL_MODE_ACTUAL else reserved_text)


def resolve_workbench_capital_display_mode_for_snapshot(snapshot):
    if not snapshot:
        return WORKBENCH_CAPITAL_MODE_RESERVED
    line_sources = dict(snapshot.get("line_value_sources") or {})
    buy_capital = _coerce_optional_float_for_sidebar(snapshot.get("buy_capital"))
    if line_sources.get("entry_price") == "actual" and buy_capital is not None:
        return WORKBENCH_CAPITAL_MODE_ACTUAL
    return WORKBENCH_CAPITAL_MODE_RESERVED


def set_workbench_capital_display_text(target, *, reserved_text, actual_text, display_mode=None):
    target._selected_reserved_capital_text = str(reserved_text)
    target._selected_actual_spend_text = str(actual_text)
    if display_mode in {WORKBENCH_CAPITAL_MODE_RESERVED, WORKBENCH_CAPITAL_MODE_ACTUAL}:
        target._selected_capital_display_mode = display_mode
    refresh_workbench_capital_display(target)


class StockToolsWorkbench:
    def __init__(self):
        self.root = tk.Tk()
        configure_workbench_theme(self.root)
        self.root.title(WORKBENCH_TITLE)
        _apply_responsive_window_size(self.root)
        self.root.update_idletasks()
        self._maximize_mode = _maximize_root_window(self.root)
        self._build_ui()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=4, style=WORKBENCH_FRAME_STYLE)
        container.pack(fill="both", expand=True)

        notebook = ttk.Notebook(container, style=WORKBENCH_NOTEBOOK_STYLE)
        notebook.pack(fill="both", expand=True)

        for panel_spec in PANEL_SPECS:
            panel_factory = _load_panel_factory(panel_spec["panel_factory_path"])
            panel = panel_factory(notebook)
            notebook.add(panel, text=panel_spec["tab_label"])

    def run(self):
        self.root.mainloop()



def launch_workbench():
    app = StockToolsWorkbench()
    app.run()
