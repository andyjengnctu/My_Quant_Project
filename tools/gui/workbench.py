from __future__ import annotations

import tkinter as tk
import warnings
from tkinter import ttk

from tools.gui.single_stock_inspector import SingleStockBacktestInspectorPanel


WORKBENCH_TITLE = "股票工具工作台"
WORKBENCH_GEOMETRY = "1920x1180"
WORKBENCH_BG = "#05090e"
WORKBENCH_SURFACE = "#0a1220"
WORKBENCH_SURFACE_ALT = "#0d1626"
WORKBENCH_BORDER = "#18324a"
WORKBENCH_TEXT = "#f7fbff"
WORKBENCH_MUTED = "#d6dfeb"
WORKBENCH_ACCENT = "#2d7ff9"
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


def _warn_gui_fallback(action, exc):
    warnings.warn(f"GUI fallback {action}: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)


PANEL_SPECS = (
    {
        "panel_id": "single_stock_backtest_inspector",
        "tab_label": "單股回測檢視",
        "backend_runner": "tools.debug.trade_log.run_debug_ticker_analysis",
        "artifact_keys": ("excel_path",),
        "inline_chart_backend": "tools.debug.charting.create_matplotlib_debug_chart_figure",
        "default_show_volume": False,
        "panel_factory": SingleStockBacktestInspectorPanel,
    },
)


def build_workbench_spec():
    return {
        "app_id": "stock_tools_workbench",
        "entry_module": "apps.gui",
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
    style.configure(WORKBENCH_LABEL_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT)
    style.configure(WORKBENCH_BUTTON_STYLE, background=WORKBENCH_SURFACE_ALT, foreground=WORKBENCH_TEXT, bordercolor=WORKBENCH_BORDER, focusthickness=1, focuscolor=WORKBENCH_BORDER, padding=(10, 4))
    style.map(WORKBENCH_BUTTON_STYLE, background=[("active", WORKBENCH_ACCENT), ("pressed", WORKBENCH_ACCENT)])
    style.configure(WORKBENCH_CHECK_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT)
    style.map(WORKBENCH_CHECK_STYLE, foreground=[("active", WORKBENCH_TEXT)])
    style.configure(WORKBENCH_ENTRY_STYLE, fieldbackground=WORKBENCH_SURFACE, foreground=WORKBENCH_TEXT, insertcolor=WORKBENCH_TEXT)
    style.configure(WORKBENCH_COMBO_STYLE, fieldbackground=WORKBENCH_SURFACE, foreground=WORKBENCH_TEXT, arrowcolor=WORKBENCH_TEXT)
    style.configure(WORKBENCH_NOTEBOOK_STYLE, background=WORKBENCH_BG, borderwidth=0)
    style.configure(f"{WORKBENCH_NOTEBOOK_STYLE}.Tab", background=WORKBENCH_SURFACE_ALT, foreground=WORKBENCH_TEXT, padding=(10, 5))
    style.map(f"{WORKBENCH_NOTEBOOK_STYLE}.Tab", background=[("selected", WORKBENCH_ACCENT)], foreground=[("selected", WORKBENCH_TEXT)])
    style.configure(WORKBENCH_TREE_STYLE, background=WORKBENCH_SURFACE, fieldbackground=WORKBENCH_SURFACE, foreground=WORKBENCH_TEXT, bordercolor=WORKBENCH_BORDER)
    style.configure(f"{WORKBENCH_TREE_STYLE}.Heading", background=WORKBENCH_SURFACE_ALT, foreground=WORKBENCH_TEXT)
    style.map(WORKBENCH_TREE_STYLE, background=[("selected", WORKBENCH_ACCENT)], foreground=[("selected", WORKBENCH_TEXT)])
    style.configure(WORKBENCH_VSCROLL_STYLE, background=WORKBENCH_SURFACE_ALT, troughcolor=WORKBENCH_BG, arrowcolor=WORKBENCH_TEXT)
    style.configure(WORKBENCH_HSCROLL_STYLE, background=WORKBENCH_SURFACE_ALT, troughcolor=WORKBENCH_BG, arrowcolor=WORKBENCH_TEXT)
    style.configure(WORKBENCH_SIDEBAR_SIGNAL_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, anchor="center", padding=(10, 8), font=("Microsoft JhengHei", 20, "bold"))
    style.configure(WORKBENCH_SIDEBAR_GATE_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, anchor="center", padding=(10, 8), font=("Microsoft JhengHei", 20, "bold"))
    style.configure(WORKBENCH_SIDEBAR_HEADER_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, font=("Microsoft JhengHei", 16, "bold"))
    style.configure(WORKBENCH_SIDEBAR_SUMMARY_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, font=("Microsoft JhengHei", 14), anchor="nw")
    style.configure(WORKBENCH_SIDEBAR_VALUE_STYLE, background=WORKBENCH_BG, foreground=WORKBENCH_TEXT, font=("Microsoft JhengHei", 14))

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


class StockToolsWorkbench:
    def __init__(self):
        self.root = tk.Tk()
        configure_workbench_theme(self.root)
        self.root.title(WORKBENCH_TITLE)
        self.root.geometry(WORKBENCH_GEOMETRY)
        self.root.minsize(1520, 920)
        self.root.update_idletasks()
        self._maximize_mode = _maximize_root_window(self.root)
        self._build_ui()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=4, style=WORKBENCH_FRAME_STYLE)
        container.pack(fill="both", expand=True)

        notebook = ttk.Notebook(container, style=WORKBENCH_NOTEBOOK_STYLE)
        notebook.pack(fill="both", expand=True)

        for panel_spec in PANEL_SPECS:
            panel = panel_spec["panel_factory"](notebook)
            notebook.add(panel, text=panel_spec["tab_label"])

    def run(self):
        self.root.mainloop()



def launch_workbench():
    app = StockToolsWorkbench()
    app.run()
