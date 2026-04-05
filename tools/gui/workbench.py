from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from tools.gui.single_stock_inspector import SingleStockBacktestInspectorPanel


WORKBENCH_TITLE = "股票工具工作台"
WORKBENCH_GEOMETRY = "1680x980"


PANEL_SPECS = (
    {
        "panel_id": "single_stock_backtest_inspector",
        "tab_label": "單股回測檢視",
        "backend_runner": "tools.debug.trade_log.run_debug_ticker_analysis",
        "artifact_keys": ("excel_path", "chart_path"),
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
        "panels": [
            {
                "panel_id": panel["panel_id"],
                "tab_label": panel["tab_label"],
                "backend_runner": panel["backend_runner"],
                "artifact_keys": list(panel["artifact_keys"]),
                "inline_chart_backend": panel["inline_chart_backend"],
                "default_show_volume": panel["default_show_volume"],
            }
            for panel in PANEL_SPECS
        ],
    }


class StockToolsWorkbench:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(WORKBENCH_TITLE)
        self.root.geometry(WORKBENCH_GEOMETRY)
        self.root.minsize(1400, 860)
        self._build_ui()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=8)
        container.pack(fill="both", expand=True)

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)

        for panel_spec in PANEL_SPECS:
            panel = panel_spec["panel_factory"](notebook)
            notebook.add(panel, text=panel_spec["tab_label"])

    def run(self):
        self.root.mainloop()



def launch_workbench():
    app = StockToolsWorkbench()
    app.run()
