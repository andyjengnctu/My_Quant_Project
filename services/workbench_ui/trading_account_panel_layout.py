from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from services.trading.protection_planning import PROTECTION_STOP_REMAINDER_ACTION
from services.workbench_ui.date_picker import DatePickerField
from services.workbench_ui.paged_table import PagedTable
from services.workbench_ui.stock_names import workbench_stock_name
from services.workbench_ui.trading_source_labels import TRADING_SOURCE_CUSTOM_LABEL
from services.workbench_ui.selection_behavior import bind_treeview_toggle_selection
from services.workbench_ui.workbench import WORKBENCH_BG, WORKBENCH_BUTTON_STYLE, WORKBENCH_COMBO_STYLE, WORKBENCH_ERROR, WORKBENCH_ENTRY_STYLE, WORKBENCH_FRAME_STYLE, WORKBENCH_INFO_LABEL_STYLE, WORKBENCH_LABEL_STYLE, WORKBENCH_LABELLF_STYLE, WORKBENCH_MUTED, WORKBENCH_SUCCESS, WORKBENCH_TEXT, WORKBENCH_TREE_STYLE, WORKBENCH_UI_FONT, WORKBENCH_VSCROLL_STYLE, _warn_gui_fallback
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    PARAM_MODE_REUSE_LABEL,
    PARAM_MODE_TRAIN_LABEL,
    WORKFLOW_HINT,
    POSITION_DECISION_HINT,
    SCANNER_HINT,
    BUY_ENTRY_HINT,
    PENDING_ENTRY_HINT,
    _parse_candidate_date,
)
from services.workbench_ui.trading_account_panel_status import (
    _TradingStatusLine,
)

class TradingAccountLayoutMixin:
    """Trading Workbench TradingAccountLayout responsibilities; composed by TradingAccountPanel."""

    def _register_sortable_table(
        self,
        tree: ttk.Treeview,
        *,
        headings: dict[str, str],
        sort_kinds: dict[str, str],
        default_column: str = "source",
        default_ascending: bool = True,
        excluded_columns: tuple[str, ...] = ("open",),
    ) -> None:
        self._table_sort_titles[tree] = dict(headings)
        self._table_sort_kinds[tree] = dict(sort_kinds)
        self._table_sort_values[tree] = {}
        self._table_sort_state[tree] = (str(default_column), bool(default_ascending))
        excluded = {str(value) for value in excluded_columns}
        for column in headings:
            command = None
            if column not in excluded:
                command = lambda selected_column=column, selected_tree=tree: self._sort_table_by_column(
                    selected_tree, selected_column
                )
            tree.heading(column, command=command)
        self._refresh_sortable_table_headings(tree)

    def _refresh_sortable_table_headings(self, tree: ttk.Treeview) -> None:
        titles = self._table_sort_titles.get(tree, {})
        active_column, ascending = self._table_sort_state.get(tree, ("", True))
        for column, title in titles.items():
            suffix = " ▲" if column == active_column and ascending else " ▼" if column == active_column else ""
            tree.heading(column, text=f"{title}{suffix}")

    @staticmethod
    def _normalize_table_sort_value(value: object, *, sort_kind: str) -> object | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text == "-":
            return None
        if sort_kind == "numeric":
            try:
                return float(text.replace(",", ""))
            except ValueError:
                return None
        if sort_kind == "date":
            parsed = _parse_candidate_date(text)
            return None if parsed is None else parsed.isoformat()
        return text.casefold()

    def _set_table_sort_values(self, tree: ttk.Treeview, iid: str, values: dict[str, object]) -> None:
        kinds = self._table_sort_kinds.get(tree, {})
        self._table_sort_values.setdefault(tree, {})[str(iid)] = {
            str(column): self._normalize_table_sort_value(
                value,
                sort_kind=str(kinds.get(str(column)) or "text"),
            )
            for column, value in values.items()
        }

    def _clear_table_sort_values(self, tree: ttk.Treeview) -> None:
        self._table_sort_values.setdefault(tree, {}).clear()

    def _apply_table_sort(self, tree: ttk.Treeview, column: str, ascending: bool) -> None:
        row_values = self._table_sort_values.get(tree, {})
        present: list[tuple[str, object]] = []
        missing: list[str] = []
        for iid in tree.get_children(""):
            value = row_values.get(str(iid), {}).get(str(column))
            if value is None:
                missing.append(str(iid))
            else:
                present.append((str(iid), value))
        present.sort(key=lambda item: item[1], reverse=not bool(ascending))
        ordered = [iid for iid, _value in present] + missing
        for index, iid in enumerate(ordered):
            tree.move(iid, "", index)
        self._table_sort_state[tree] = (str(column), bool(ascending))
        self._refresh_sortable_table_headings(tree)

    def _apply_current_table_sort(self, tree: ttk.Treeview) -> None:
        column, ascending = self._table_sort_state.get(tree, ("", True))
        if column:
            self._apply_table_sort(tree, column, ascending)

    def _sort_table_by_column(self, tree: ttk.Treeview, column: str) -> None:
        active_column, active_ascending = self._table_sort_state.get(tree, ("", True))
        ascending = not active_ascending if active_column == str(column) else True
        self._apply_table_sort(tree, str(column), ascending)

    def _build_ui(self):
        # AI: The Trading page contains several independent detail tables.  A fixed-height
        # notebook tab used to squeeze the middle rows to ~0px on common 1080p screens,
        # making Scanner / proposed-order details effectively invisible.  Keep the
        # existing tables and semantics, but put the whole page in one vertical canvas.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self._page_canvas = tk.Canvas(self, background=WORKBENCH_BG, highlightthickness=0, borderwidth=0)
        self._page_scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self._page_canvas.yview, style=WORKBENCH_VSCROLL_STYLE
        )
        self._page_canvas.configure(yscrollcommand=self._page_scrollbar.set, yscrollincrement=36)
        self._page_canvas.grid(row=1, column=0, sticky="nsew")
        self._page_scrollbar.grid(row=1, column=1, sticky="ns")
        content = ttk.Frame(self._page_canvas, style=WORKBENCH_FRAME_STYLE)
        self._page_content = content
        self._page_window = self._page_canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", self._sync_page_scrollregion, add="+")
        self._page_canvas.bind("<Configure>", self._sync_page_content_width, add="+")
        content.columnconfigure(0, weight=1)

        self._footer_hint_var = tk.StringVar(value="")

        operations_box = ttk.LabelFrame(self, text="Trading 儀表板", padding=10, style=WORKBENCH_LABELLF_STYLE)
        operations_box.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        operations_box.columnconfigure(0, weight=1)
        self._overview_vars = {
            key: tk.StringVar(value="-")
            for key in ("total", "quick", "candidates", "slots", "funds", "strategy_params")
        }
        self._overview_detail_vars = {key: tk.StringVar(value="-") for key in self._overview_vars}
        self._overview_primary_labels: dict[str, tk.Label] = {}
        self._overview_detail_labels: dict[str, ttk.Label] = {}
        overview_grid = ttk.Frame(operations_box, style=WORKBENCH_FRAME_STYLE)
        overview_grid.grid(row=0, column=0, sticky="ew")
        for col, (title, key) in enumerate((
            ("總股數", "total"),
            ("符合快篩數", "quick"),
            ("Scanner 候選數", "candidates"),
            ("可操作檔位", "slots"),
            ("可操作資金", "funds"),
            ("策略 / Params", "strategy_params"),
        )):
            box = ttk.LabelFrame(
                overview_grid,
                text=title,
                padding=(8, 4),
                style=WORKBENCH_LABELLF_STYLE,
                labelanchor="n",
            )
            box.grid(row=0, column=col, padx=(0 if col == 0 else 6, 0), sticky="nsew")
            primary = tk.Label(
                box,
                textvariable=self._overview_vars[key],
                background=WORKBENCH_BG,
                foreground=WORKBENCH_TEXT,
                font=(WORKBENCH_UI_FONT[0], WORKBENCH_UI_FONT[1], "bold"),
                anchor="center",
                justify="center",
            )
            primary.pack(fill="x")
            detail_label = ttk.Label(
                box,
                textvariable=self._overview_detail_vars[key],
                style=WORKBENCH_LABEL_STYLE,
                foreground=WORKBENCH_MUTED,
                anchor="center",
                justify="center",
            )
            detail_label.pack(fill="x", pady=(1, 0))
            self._overview_primary_labels[key] = primary
            self._overview_detail_labels[key] = detail_label
            overview_grid.columnconfigure(col, weight=1)

        self._operations_next_var = tk.StringVar(value="下一步：-")
        self._operations_next_label = _TradingStatusLine(operations_box, textvariable=self._operations_next_var, max_lines=1)
        self._operations_next_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # Trading Center still consumes the account dashboard read model for current
        # position valuation, but its duplicate account-summary cards belong only to
        # Accounting Center and are not mounted here.
        self._dashboard_metric_vars = {
            key: tk.StringVar(value="-")
            for key in ("cash", "market_value", "liquidation", "equity")
        }
        self._dashboard_metric_labels = {}
        self._dashboard_detail_var = tk.StringVar(value="account/ 衍生快照尚未載入")

        # AI: Keep the three daily decisions on one compact row. Data updates live
        # in 資料中心; the separate market-data and batch runners remain available
        # internally but are not exposed as daily Trading actions.
        workflow_box = ttk.LabelFrame(content, text="每日 Trading 流程", padding=(8, 4), style=WORKBENCH_LABELLF_STYLE)
        workflow_box.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        workflow_bar = ttk.Frame(workflow_box, style=WORKBENCH_FRAME_STYLE)
        workflow_bar.pack(fill="x")

        ttk.Button(
            workflow_bar, text="1 策略選擇",
            command=self._open_strategy_promotion_dialog,
            style=WORKBENCH_BUTTON_STYLE,
        ).pack(side="left")
        ttk.Separator(workflow_bar, orient="vertical").pack(side="left", fill="y", padx=12, pady=2)
        ttk.Label(workflow_bar, text="Params 模式", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._param_mode_var = tk.StringVar(value=PARAM_MODE_REUSE_LABEL)
        self._param_mode_combo = ttk.Combobox(
            workflow_bar,
            textvariable=self._param_mode_var,
            values=(PARAM_MODE_REUSE_LABEL, PARAM_MODE_TRAIN_LABEL),
            state="readonly",
            width=18,
            style=WORKBENCH_COMBO_STYLE,
        )
        self._param_mode_combo.pack(side="left", padx=(6, 0))
        self._param_mode_combo.bind("<<ComboboxSelected>>", self._on_param_mode_selected)
        params_button = ttk.Button(
            workflow_bar, text="2 訓練／套用 Params",
            command=lambda: self._start_workflow_action("params"),
            style=WORKBENCH_BUTTON_STYLE,
        )
        params_button.pack(side="left", padx=(8, 0))
        self._workflow_buttons.append(params_button)
        self._workflow_action_buttons["params"] = params_button

        ttk.Separator(workflow_bar, orient="vertical").pack(side="left", fill="y", padx=12, pady=2)
        scanner_button = ttk.Button(
            workflow_bar, text="3 Scanner 候選",
            command=lambda: self._start_workflow_action("scanner"),
            style=WORKBENCH_BUTTON_STYLE,
        )
        scanner_button.pack(side="left")
        self._workflow_buttons.append(scanner_button)
        self._workflow_action_buttons["scanner"] = scanner_button
        # AI: Params lineage and workflow status are already visible in Trading 儀表板.

        header = ttk.LabelFrame(content, text="Trading 帳戶", padding=8, style=WORKBENCH_LABELLF_STYLE)
        header.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(header, text="重新整理帳戶", command=lambda: self._request_state_refresh("帳戶狀態刷新"), style=WORKBENCH_BUTTON_STYLE).pack(side="right")

        cash_box = ttk.LabelFrame(content, text="現金", padding=10, style=WORKBENCH_LABELLF_STYLE)
        cash_box.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(cash_box, text="帳戶現金", style=WORKBENCH_LABEL_STYLE).grid(row=0, column=0, sticky="w")
        self._cash_var = tk.StringVar()
        self._cash_entry = ttk.Entry(cash_box, textvariable=self._cash_var, width=22, style=WORKBENCH_ENTRY_STYLE)
        self._cash_entry.grid(row=0, column=1, sticky="w", padx=(8, 8))
        self._initialize_button = ttk.Button(cash_box, text="初始化帳戶", command=self._initialize_account, style=WORKBENCH_BUTTON_STYLE)
        self._initialize_button.grid(row=0, column=2, padx=(0, 8))
        self._set_cash_button = ttk.Button(cash_box, text="更新現金", command=self._set_cash, style=WORKBENCH_BUTTON_STYLE)
        self._set_cash_button.grid(row=0, column=3)

        form = ttk.LabelFrame(content, text="既有持股（manual adopted broker truth）", padding=10, style=WORKBENCH_LABELLF_STYLE)
        form.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        labels = ("股票代號", "名稱", "股數", "剩餘成本總額", "買入日 YYYY-MM-DD", "備註")
        for col, label in enumerate(labels):
            ttk.Label(form, text=label, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else 8, 0))
        self._ticker_var = tk.StringVar()
        self._ticker_name_var = tk.StringVar(value="-")
        self._ticker_var.trace_add("write", lambda *_args: self._ticker_name_var.set(workbench_stock_name(WORKBENCH_PROJECT_ROOT, self._ticker_var.get())))
        self._qty_var = tk.StringVar()
        self._cost_var = tk.StringVar()
        self._entry_date_var = tk.StringVar()
        self._note_var = tk.StringVar()
        ttk.Entry(form, textvariable=self._ticker_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(form, textvariable=self._ticker_name_var, width=14, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Entry(form, textvariable=self._qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Entry(form, textvariable=self._cost_var, width=18, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=(4, 0))
        DatePickerField(form, textvariable=self._entry_date_var, width=14).grid(row=1, column=4, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Entry(form, textvariable=self._note_var, width=36, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=5, sticky="ew", padx=(8, 0), pady=(4, 0))
        form.columnconfigure(5, weight=1)

        button_row = ttk.Frame(form, style=WORKBENCH_FRAME_STYLE)
        button_row.grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 0))
        self._add_button = ttk.Button(button_row, text="新增既有持股", command=self._adopt_position, style=WORKBENCH_BUTTON_STYLE)
        self._add_button.pack(side="left")
        self._correct_button = ttk.Button(button_row, text="修正選取持股", command=self._correct_position, style=WORKBENCH_BUTTON_STYLE)
        self._correct_button.pack(side="left", padx=(8, 0))
        self._remove_button = ttk.Button(button_row, text="移除選取持股", command=self._remove_position, style=WORKBENCH_BUTTON_STYLE)
        self._remove_button.pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="清除輸入", command=self._clear_position_form, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(8, 0))

        table_box = ttk.LabelFrame(content, text="持股區", padding=8, style=WORKBENCH_LABELLF_STYLE)
        table_box.grid(row=5, column=0, sticky="nsew", pady=(0, 8))
        table_box.rowconfigure(0, weight=1)
        table_box.columnconfigure(0, weight=1)
        columns = ("open", "source", "ticker", "stock_name", "order_date", "entry_date", "qty", "avg_cost", "current", "stop", "target", "sell_signal", "status")
        self._tree = ttk.Treeview(table_box, columns=columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse", height=7)
        headings = {
            "open": "↗", "source": "來源", "ticker": "股票", "stock_name": "名稱", "order_date": "掛單日", "entry_date": "成交日", "qty": "股數", "avg_cost": "均價", "current": "市價",
            "stop": "停損", "target": "停利", "sell_signal": "賣出訊號", "status": "狀態",
        }
        widths = {"open": 36, "source": 90, "ticker": 80, "stock_name": 120, "order_date": 100, "entry_date": 100, "qty": 85, "avg_cost": 95, "current": 90, "stop": 95, "target": 95, "sell_signal": 125, "status": 88}
        for key in columns:
            self._tree.heading(key, text=headings[key])
            self._tree.column(key, width=widths[key], anchor="center", stretch=(key != "open"))
        self._register_sortable_table(
            self._tree,
            headings=headings,
            sort_kinds={
                "source": "text",
                "ticker": "text",
                "stock_name": "text",
                "order_date": "date",
                "entry_date": "date",
                "qty": "numeric",
                "avg_cost": "numeric",
                "current": "numeric",
                "stop": "numeric",
                "target": "numeric",
                "sell_signal": "text",
                "status": "text",
            },
            default_column="source",
            default_ascending=True,
        )
        self._tree.grid(row=0, column=0, sticky="nsew")
        self._tree.bind("<<TreeviewSelect>>", self._on_position_selected)
        self._tree.bind("<Button-1>", self._on_position_tree_click, add="+")
        self._tree.bind("<Double-1>", self._open_selected_position_in_inspector, add="+")

        candidate_box = ttk.LabelFrame(content, text="侯選區", padding=8, style=WORKBENCH_LABELLF_STYLE)
        candidate_box.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        candidate_box.rowconfigure(0, weight=1)
        candidate_box.columnconfigure(0, weight=1)
        self._candidate_tree = PagedTable(
            candidate_box,
            columns=self._build_candidate_table_columns([]),
            page_size=12,
            default_sort_key="rank",
            empty_text="目前沒有 Scanner 候選",
            on_select=self._on_candidate_selected,
            on_open_stock=self._open_candidate_ticker_in_inspector,
            on_mousewheel=self._on_page_mousewheel,
            on_pointer_enter=lambda _event: self._show_footer_hint(SCANNER_HINT),
            on_pointer_leave=lambda _event: self._schedule_footer_hint_clear(),
        )
        self._candidate_tree.grid(row=0, column=0, sticky="ew")
        pending_box = ttk.LabelFrame(content, text="掛單區", padding=8, style=WORKBENCH_LABELLF_STYLE)
        pending_box.grid(row=4, column=0, sticky="nsew", pady=(0, 8))
        pending_box.columnconfigure(0, weight=1)
        pending_columns = ("open", "source", "ticker", "stock_name", "date", "qty", "reserved", "limit", "stop", "target", "status")
        self._pending_tree = ttk.Treeview(
            pending_box, columns=pending_columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse", height=5
        )
        pending_headings = {
            "open": "↗", "source": "來源", "ticker": "股票", "stock_name": "名稱", "date": "掛單日", "qty": "規劃股數",
            "reserved": "預留成本", "limit": "買入限價", "stop": "停損", "target": "停利", "status": "狀態",
        }
        pending_widths = {
            "open": 36, "source": 86, "ticker": 75, "stock_name": 120, "date": 96, "qty": 88,
            "reserved": 110, "limit": 90, "stop": 86, "target": 86, "status": 100,
        }
        for key in pending_columns:
            self._pending_tree.heading(key, text=pending_headings[key])
            self._pending_tree.column(key, width=pending_widths[key], anchor="center", stretch=(key != "open"))
        self._register_sortable_table(
            self._pending_tree,
            headings=pending_headings,
            sort_kinds={
                "source": "text",
                "ticker": "text",
                "stock_name": "text",
                "date": "date",
                "qty": "numeric",
                "reserved": "numeric",
                "limit": "numeric",
                "stop": "numeric",
                "target": "numeric",
                "status": "text",
            },
            default_column="source",
            default_ascending=True,
        )
        self._pending_tree.grid(row=0, column=0, sticky="ew")
        self._pending_tree.bind("<<TreeviewSelect>>", self._on_pending_selected)
        self._pending_tree.bind("<Button-1>", self._on_pending_tree_click, add="+")
        self._pending_tree.bind("<Double-1>", self._open_selected_pending_in_inspector, add="+")

        pending_actions = ttk.LabelFrame(
            pending_box,
            text="掛單輸入",
            padding=8,
            style=WORKBENCH_LABELLF_STYLE,
        )
        pending_actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self._pending_order_source_var = tk.StringVar(value=TRADING_SOURCE_CUSTOM_LABEL)
        self._pending_order_ticker_var = tk.StringVar()
        self._pending_order_stock_name_var = tk.StringVar(value="-")
        self._pending_order_ticker_var.trace_add("write", lambda *_args: self._pending_order_stock_name_var.set(workbench_stock_name(WORKBENCH_PROJECT_ROOT, self._pending_order_ticker_var.get())))
        self._pending_order_qty_var = tk.StringVar()
        self._pending_order_limit_var = tk.StringVar()
        self._pending_fill_price_var = tk.StringVar()
        self._pending_order_date_var = tk.StringVar(value="")
        self._pending_fill_date_var = tk.StringVar(value="")
        draft_labels = (
            ("來源", 10),
            ("股票", 10),
            ("名稱", 14),
            ("掛單日", 12),
            ("規劃股數", 10),
            ("成交日", 12),
            ("成交價", 12),
        )
        for col, (label, _width) in enumerate(draft_labels):
            ttk.Label(pending_actions, text=label, style=WORKBENCH_LABEL_STYLE).grid(
                row=0, column=col, sticky="w", padx=(0 if col == 0 else 8, 0)
            )
        ttk.Label(
            pending_actions,
            textvariable=self._pending_order_source_var,
            width=10,
            style=WORKBENCH_INFO_LABEL_STYLE,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._pending_order_ticker_entry = ttk.Entry(
            pending_actions, textvariable=self._pending_order_ticker_var, width=10, style=WORKBENCH_ENTRY_STYLE
        )
        self._pending_order_ticker_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Label(pending_actions, textvariable=self._pending_order_stock_name_var, width=14, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._pending_order_ticker_entry.bind("<Return>", self._on_manual_pending_ticker_commit)
        self._pending_order_ticker_entry.bind("<KeyRelease>", self._schedule_manual_pending_ticker_preview)
        self._pending_order_date_field = DatePickerField(
            pending_actions, textvariable=self._pending_order_date_var, width=12, allowed_dates=()
        )
        self._pending_order_date_field.grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._pending_order_qty_entry = ttk.Entry(
            pending_actions, textvariable=self._pending_order_qty_var, width=10, style=WORKBENCH_ENTRY_STYLE
        )
        self._pending_order_qty_entry.grid(row=1, column=4, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._pending_fill_date_field = DatePickerField(
            pending_actions, textvariable=self._pending_fill_date_var, width=12, allowed_dates=()
        )
        self._pending_fill_date_field.grid(row=1, column=5, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._pending_fill_price_combo = ttk.Combobox(
            pending_actions,
            textvariable=self._pending_fill_price_var,
            values=(),
            width=12,
            state="disabled",
            style=WORKBENCH_COMBO_STYLE,
        )
        self._pending_fill_price_combo.grid(row=1, column=6, sticky="ew", padx=(8, 0), pady=(4, 0))
        for draft_var in (self._pending_order_qty_var, self._pending_order_date_var):
            draft_var.trace_add("write", self._schedule_pending_value_preview)
        self._pending_fill_date_var.trace_add("write", self._schedule_pending_fill_preview)
        self._pending_fill_price_var.trace_add("write", self._refresh_pending_fill_button_state)
        pending_draft_buttons = ttk.Frame(pending_actions, style=WORKBENCH_FRAME_STYLE)
        pending_draft_buttons.grid(row=1, column=7, sticky="w", padx=(12, 0), pady=(4, 0))
        self._pending_submit_button = ttk.Button(
            pending_draft_buttons, text="確認掛單", command=self._confirm_submit_pending_draft, style=WORKBENCH_BUTTON_STYLE
        )
        self._pending_submit_button.pack(side="left")
        self._pending_fill_button = ttk.Button(
            pending_draft_buttons,
            text="確認成交",
            command=self._confirm_pending_fill,
            style=WORKBENCH_BUTTON_STYLE,
            state="disabled",
        )
        self._pending_fill_button.pack(side="left", padx=(8, 0))
        self._pending_delete_button = ttk.Button(
            pending_draft_buttons,
            text="刪除掛單",
            command=self._delete_selected_pending,
            style=WORKBENCH_BUTTON_STYLE,
            state="disabled",
        )
        self._pending_delete_button.pack(side="left", padx=(8, 0))
        self._pending_draft_preview_var = tk.StringVar(value="")
        _TradingStatusLine(
            pending_actions,
            textvariable=self._pending_draft_preview_var,
            default_tone="muted",
            max_lines=2,
        ).grid(row=2, column=0, columnspan=8, sticky="ew", pady=(6, 0))

        trade_box = ttk.LabelFrame(content, text="直接補登買入", padding=10, style=WORKBENCH_LABELLF_STYLE)
        trade_box.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        trade_labels = ("來源", "股票", "名稱", "數量", "成交日", "成交價")
        self._trade_ticker_var = tk.StringVar()
        self._trade_stock_name_var = tk.StringVar(value="-")
        self._trade_ticker_var.trace_add("write", lambda *_args: self._trade_stock_name_var.set(workbench_stock_name(WORKBENCH_PROJECT_ROOT, self._trade_ticker_var.get())))
        self._trade_qty_var = tk.StringVar()
        self._trade_price_var = tk.StringVar()
        self._trade_date_var = tk.StringVar()
        self._trade_source_var = tk.StringVar(value="-")
        for col, label in enumerate(trade_labels):
            ttk.Label(trade_box, text=label, style=WORKBENCH_LABEL_STYLE).grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else 8, 0))
        ttk.Label(
            trade_box,
            textvariable=self._trade_source_var,
            width=10,
            style=WORKBENCH_INFO_LABEL_STYLE,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._trade_ticker_entry = ttk.Entry(trade_box, textvariable=self._trade_ticker_var, width=12, style=WORKBENCH_ENTRY_STYLE)
        self._trade_ticker_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Label(trade_box, textvariable=self._trade_stock_name_var, width=14, style=WORKBENCH_LABEL_STYLE).grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Entry(trade_box, textvariable=self._trade_qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._trade_date_field = DatePickerField(trade_box, textvariable=self._trade_date_var, width=12, allowed_dates=())
        self._trade_date_field.grid(row=1, column=4, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._trade_price_combo = ttk.Combobox(
            trade_box, textvariable=self._trade_price_var, values=(), width=14, state="normal", style=WORKBENCH_COMBO_STYLE
        )
        self._trade_price_combo.grid(row=1, column=5, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._trade_ticker_var.trace_add("write", self._schedule_direct_buy_ticker_constraints)
        self._trade_date_var.trace_add("write", self._schedule_direct_fill_constraints)
        trade_buttons = ttk.Frame(trade_box, style=WORKBENCH_FRAME_STYLE)
        trade_buttons.grid(row=1, column=6, sticky="w", padx=(12, 0), pady=(4, 0))
        ttk.Button(trade_buttons, text="登錄買入", command=lambda: self._record_simple_trade("BUY"), style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        performance_box = ttk.LabelFrame(content, text="帳戶績效統計", padding=8, style=WORKBENCH_LABELLF_STYLE)
        performance_box.grid(row=8, column=0, sticky="nsew", pady=(0, 8))
        performance_box.columnconfigure(0, weight=1)
        perf_columns = ("scope", "count", "cost", "pnl", "return", "profitable", "rate")
        self._performance_tree = ttk.Treeview(performance_box, columns=perf_columns, show="headings", style=WORKBENCH_TREE_STYLE, height=3)
        self._performance_tree.tag_configure("gain", foreground=WORKBENCH_ERROR)
        self._performance_tree.tag_configure("loss", foreground=WORKBENCH_SUCCESS)
        self._performance_tree.tag_configure("flat", foreground=WORKBENCH_TEXT)
        perf_headings = {"scope": "範圍", "count": "持股/交易數", "cost": "成本基礎", "pnl": "PnL", "return": "報酬率", "profitable": "獲利數", "rate": "獲利率/勝率"}
        perf_widths = {"scope": 120, "count": 100, "cost": 130, "pnl": 120, "return": 95, "profitable": 90, "rate": 110}
        for key in perf_columns:
            self._performance_tree.heading(key, text=perf_headings[key])
            self._performance_tree.column(key, width=perf_widths[key], anchor="center")
        self._performance_tree.grid(row=0, column=0, sticky="ew")
        bind_treeview_toggle_selection(self._performance_tree)
        self._performance_note_var = tk.StringVar(value="已賣出＝account event 中已完整平倉交易；持有中以最新 Trading 市價估值。")
        ttk.Label(performance_box, textvariable=self._performance_note_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).grid(row=1, column=0, sticky="w", pady=(5, 0))

        # AI: Trading Center owns selection and execution.  Account maintenance,
        # inventory detail, transaction history and performance live in the separate
        # top-level Accounting Center.  Keep these widgets instantiated for backward-
        # compatible refresh methods but remove them from the Trading Center layout.
        for accounting_section in (header, cash_box, form, performance_box):
            accounting_section.grid_remove()

        advanced_notebook = ttk.Notebook(content, style="Workbench.TNotebook")
        # Legacy broker-OMS widgets remain instantiated for backward-compatible state readers,
        # but are intentionally not mounted in Trading Center.  User executes at the broker.
        advanced_orders_tab = ttk.Frame(advanced_notebook, padding=6, style=WORKBENCH_FRAME_STYLE)
        advanced_protection_tab = ttk.Frame(advanced_notebook, padding=6, style=WORKBENCH_FRAME_STYLE)
        advanced_indicator_tab = ttk.Frame(advanced_notebook, padding=6, style=WORKBENCH_FRAME_STYLE)
        for tab in (advanced_orders_tab, advanced_protection_tab, advanced_indicator_tab):
            tab.columnconfigure(0, weight=1)
        advanced_notebook.add(advanced_orders_tab, text="進階｜掛單/成交")
        advanced_notebook.add(advanced_protection_tab, text="進階｜Stop / TP")
        advanced_notebook.add(advanced_indicator_tab, text="進階｜Indicator SELL")

        proposed_box = ttk.LabelFrame(advanced_orders_tab, text="建議掛單（尚未送單／尚未成交）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        proposed_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        proposed_box.rowconfigure(1, weight=1)
        proposed_box.columnconfigure(0, weight=1)
        self._proposed_status_var = tk.StringVar(value="尚未產生建議掛單。")
        self._proposed_status_label = _TradingStatusLine(proposed_box, textvariable=self._proposed_status_var, default_tone="muted", max_lines=2)
        self._proposed_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        proposed_columns = ("rank", "ticker", "stock_name", "kind", "agree", "limit", "qty", "reserved", "stop", "target")
        self._proposed_tree = ttk.Treeview(proposed_box, columns=proposed_columns, show="headings", style=WORKBENCH_TREE_STYLE, height=6)
        proposed_headings = {
            "rank": "順位", "ticker": "股票", "stock_name": "名稱", "kind": "類型", "agree": "同意/成員",
            "limit": "買入限價", "qty": "股數", "reserved": "預留資金", "stop": "初始Stop", "target": "停利線"
        }
        proposed_widths = {"rank": 60, "ticker": 80, "stock_name": 120, "kind": 110, "agree": 115, "limit": 100, "qty": 90, "reserved": 120, "stop": 100, "target": 135}
        for key in proposed_columns:
            self._proposed_tree.heading(key, text=proposed_headings[key])
            self._proposed_tree.column(key, width=proposed_widths[key], anchor="center")
        self._proposed_tree.grid(row=1, column=0, sticky="nsew")
        bind_treeview_toggle_selection(self._proposed_tree)

        submit_row = ttk.Frame(proposed_box, style=WORKBENCH_FRAME_STYLE)
        submit_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(submit_row, text="5 建議掛單", command=lambda: self._start_workflow_action("orders"), style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(0, 12))
        ttk.Label(submit_row, text="券商委託號（可留空）", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._broker_order_id_var = tk.StringVar()
        ttk.Entry(submit_row, textvariable=self._broker_order_id_var, width=18, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(submit_row, text="備註", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._order_note_var = tk.StringVar()
        ttk.Entry(submit_row, textvariable=self._order_note_var, width=28, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10), fill="x", expand=True)
        self._confirm_ordered_button = ttk.Button(
            submit_row,
            text="確認選取已送單",
            command=self._confirm_selected_ordered,
            style=WORKBENCH_BUTTON_STYLE,
        )
        self._confirm_ordered_button.pack(side="right")

        pending_box = ttk.LabelFrame(advanced_orders_tab, text="券商掛單狀態（ORDERED / PARTIAL / FILLED / CANCELLED）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        pending_box.grid(row=1, column=0, sticky="nsew")
        pending_box.rowconfigure(1, weight=1)
        pending_box.columnconfigure(0, weight=1)
        self._order_status_var = tk.StringVar(value="尚無實際送單紀錄。")
        self._order_status_label = _TradingStatusLine(pending_box, textvariable=self._order_status_var, default_tone="muted", max_lines=2)
        self._order_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        order_columns = ("ticker", "stock_name", "side", "purpose", "status", "order_type", "price", "qty", "filled", "remaining", "avg_fill", "broker_id", "info_date", "ordered_at", "filled_at", "cancelled_at")
        self._order_tree = ttk.Treeview(pending_box, columns=order_columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse")
        order_headings = {
            "ticker": "股票", "stock_name": "名稱", "side": "方向", "purpose": "用途", "status": "狀態", "order_type": "類型",
            "price": "委託/觸發價", "qty": "委託股數", "filled": "已成交", "remaining": "未成交",
            "avg_fill": "平均成交價", "broker_id": "券商委託號", "info_date": "資訊日",
            "ordered_at": "送單時間", "filled_at": "完成時間", "cancelled_at": "取消時間",
        }
        order_widths = {
            "ticker": 80, "stock_name": 120, "side": 65, "purpose": 125, "status": 90, "order_type": 100, "price": 105,
            "qty": 95, "filled": 90, "remaining": 90, "avg_fill": 105, "broker_id": 130, "info_date": 105,
            "ordered_at": 160, "filled_at": 160, "cancelled_at": 160,
        }
        for key in order_columns:
            self._order_tree.heading(key, text=order_headings[key])
            self._order_tree.column(key, width=order_widths[key], anchor="center")
        self._order_tree.grid(row=1, column=0, sticky="nsew")
        bind_treeview_toggle_selection(self._order_tree)
        fill_row = ttk.Frame(pending_box, style=WORKBENCH_FRAME_STYLE)
        fill_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(fill_row, text="本次成交股數", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_qty_var = tk.StringVar()
        ttk.Entry(fill_row, textvariable=self._fill_qty_var, width=12, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(fill_row, text="本次成交價", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_price_var = tk.StringVar()
        ttk.Entry(fill_row, textvariable=self._fill_price_var, width=12, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(fill_row, text="成交日", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._fill_date_var = tk.StringVar()
        DatePickerField(fill_row, textvariable=self._fill_date_var, width=12).pack(side="left", padx=(6, 10))
        self._confirm_fill_button = ttk.Button(
            fill_row,
            text="確認選取成交",
            command=self._confirm_selected_fill,
            style=WORKBENCH_BUTTON_STYLE,
        )
        self._confirm_fill_button.pack(side="right")

        pending_buttons = ttk.Frame(pending_box, style=WORKBENCH_FRAME_STYLE)
        pending_buttons.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._cancel_order_button = ttk.Button(
            pending_buttons,
            text="確認選取剩餘委託已取消",
            command=self._cancel_selected_order,
            style=WORKBENCH_BUTTON_STYLE,
        )
        self._cancel_order_button.pack(side="left")
        ttk.Button(pending_buttons, text="刷新掛單狀態", command=lambda: self._request_state_refresh("券商掛單狀態刷新"), style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(8, 0))

        protection_box = ttk.LabelFrame(advanced_protection_tab, text="成交後 Stop / TP 保護單計畫（logical plan；送單狀態見券商掛單表）", padding=8, style=WORKBENCH_LABELLF_STYLE)
        protection_box.grid(row=0, column=0, sticky="nsew")
        protection_box.rowconfigure(1, weight=1)
        protection_box.columnconfigure(0, weight=1)
        self._protection_status_var = tk.StringVar(value="尚未建立成交後保護單計畫。")
        self._protection_status_label = _TradingStatusLine(protection_box, textvariable=self._protection_status_var, default_tone="muted", max_lines=2)
        self._protection_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        protection_columns = ("ticker", "stock_name", "qty", "entry", "stop_qty", "stop", "tp_qty", "target", "entry_order_status", "priority")
        self._protection_tree = ttk.Treeview(
            protection_box,
            columns=protection_columns,
            show="headings",
            style=WORKBENCH_TREE_STYLE,
            selectmode="browse",
        )
        protection_headings = {
            "ticker": "股票",
            "stock_name": "名稱",
            "qty": "目前持股",
            "entry": "實際成交均價",
            "stop_qty": "Stop股數",
            "stop": "Stop觸發價",
            "tp_qty": "TP股數",
            "target": "TP Limit",
            "entry_order_status": "買單狀態",
            "priority": "同bar優先序",
        }
        protection_widths = {
            "ticker": 80, "stock_name": 120, "qty": 95, "entry": 110, "stop_qty": 95, "stop": 105,
            "tp_qty": 90, "target": 105, "entry_order_status": 95, "priority": 120,
        }
        for key in protection_columns:
            self._protection_tree.heading(key, text=protection_headings[key])
            self._protection_tree.column(key, width=protection_widths[key], anchor="center")
        self._protection_tree.grid(row=1, column=0, sticky="nsew")
        bind_treeview_toggle_selection(self._protection_tree)
        protection_buttons = ttk.Frame(protection_box, style=WORKBENCH_FRAME_STYLE)
        protection_buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(
            protection_buttons,
            text="建立／刷新保護單計畫",
            command=self._rebuild_protection_plan,
            style=WORKBENCH_BUTTON_STYLE,
        ).pack(side="left")
        ttk.Button(
            protection_buttons,
            text="刷新計畫狀態",
            command=lambda: self._request_state_refresh("保護單計畫狀態刷新"),
            style=WORKBENCH_BUTTON_STYLE,
        ).pack(side="left", padx=(8, 0))

        protection_submit = ttk.Frame(protection_box, style=WORKBENCH_FRAME_STYLE)
        protection_submit.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(protection_submit, text="單腿券商委託號", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._protection_broker_id_var = tk.StringVar()
        ttk.Entry(protection_submit, textvariable=self._protection_broker_id_var, width=16, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        self._confirm_stop_submitted_button = ttk.Button(
            protection_submit, text="確認 Stop 已送單", command=lambda: self._confirm_protection_leg_submitted("STOP_FULL"), style=WORKBENCH_BUTTON_STYLE
        )
        self._confirm_stop_submitted_button.pack(side="left")
        self._confirm_stop_remainder_submitted_button = ttk.Button(
            protection_submit, text="確認 Stop 剩餘 MARKET 已送單", command=lambda: self._confirm_protection_leg_submitted(PROTECTION_STOP_REMAINDER_ACTION), style=WORKBENCH_BUTTON_STYLE
        )
        self._confirm_stop_remainder_submitted_button.pack(side="left", padx=(8, 0))
        self._confirm_tp_submitted_button = ttk.Button(
            protection_submit, text="確認 TP 已送單", command=lambda: self._confirm_protection_leg_submitted("TP_HALF"), style=WORKBENCH_BUTTON_STYLE
        )
        self._confirm_tp_submitted_button.pack(side="left", padx=(8, 0))

        protection_oco = ttk.Frame(protection_box, style=WORKBENCH_FRAME_STYLE)
        protection_oco.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(protection_oco, text="券商 OCO/互斥群組 ID", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._protection_oco_group_var = tk.StringVar()
        ttk.Entry(protection_oco, textvariable=self._protection_oco_group_var, width=18, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(protection_oco, text="Stop委託號", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._protection_stop_broker_id_var = tk.StringVar()
        ttk.Entry(protection_oco, textvariable=self._protection_stop_broker_id_var, width=14, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        ttk.Label(protection_oco, text="TP委託號", style=WORKBENCH_LABEL_STYLE).pack(side="left")
        self._protection_tp_broker_id_var = tk.StringVar()
        ttk.Entry(protection_oco, textvariable=self._protection_tp_broker_id_var, width=14, style=WORKBENCH_ENTRY_STYLE).pack(side="left", padx=(6, 10))
        self._confirm_oco_submitted_button = ttk.Button(
            protection_oco, text="確認 Stop+TP 已以券商 OCO 送單", command=self._confirm_protection_oco_submitted, style=WORKBENCH_BUTTON_STYLE
        )
        self._confirm_oco_submitted_button.pack(side="left")

        indicator_box = ttk.LabelFrame(advanced_indicator_tab, text="Indicator SELL 計畫", padding=8, style=WORKBENCH_LABELLF_STYLE)
        indicator_box.grid(row=0, column=0, sticky="nsew")
        indicator_box.rowconfigure(1, weight=1)
        indicator_box.columnconfigure(0, weight=1)
        self._indicator_status_var = tk.StringVar(value="尚未建立 Indicator SELL 計畫。")
        self._indicator_status_label = _TradingStatusLine(indicator_box, textvariable=self._indicator_status_var, default_tone="muted", max_lines=2)
        self._indicator_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        columns=("ticker","stock_name","signal","qty","entry_date","type","carried")
        self._indicator_tree=ttk.Treeview(indicator_box, columns=columns, show="headings", style=WORKBENCH_TREE_STYLE, selectmode="browse")
        headings={"ticker":"股票","stock_name":"名稱","signal":"Signal日","qty":"賣出股數","entry_date":"Entry日","type":"委託","carried":"狀態"}
        widths={"ticker":90,"stock_name":120,"signal":100,"qty":100,"entry_date":100,"type":90,"carried":110}
        for key in columns:
            self._indicator_tree.heading(key,text=headings[key]); self._indicator_tree.column(key,width=widths[key],anchor="center")
        self._indicator_tree.grid(row=1,column=0,sticky="nsew")
        bind_treeview_toggle_selection(self._indicator_tree)
        buttons=ttk.Frame(indicator_box,style=WORKBENCH_FRAME_STYLE); buttons.grid(row=2,column=0,columnspan=2,sticky="w",pady=(8,0))
        ttk.Button(buttons,text="建立／刷新 Indicator SELL 計畫",command=self._rebuild_indicator_exit_plan,style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        ttk.Button(buttons,text="刷新 Indicator SELL 狀態",command=lambda: self._request_state_refresh("Indicator SELL 狀態刷新"),style=WORKBENCH_BUTTON_STYLE).pack(side="left",padx=(8,0))
        ttk.Label(buttons,text="券商委託號",style=WORKBENCH_LABEL_STYLE).pack(side="left",padx=(14,0))
        self._indicator_broker_id_var=tk.StringVar()
        ttk.Entry(buttons,textvariable=self._indicator_broker_id_var,width=16,style=WORKBENCH_ENTRY_STYLE).pack(side="left",padx=(6,8))
        ttk.Button(buttons,text="確認選取 MARKET SELL 已送單",command=self._confirm_indicator_exit_submitted,style=WORKBENCH_BUTTON_STYLE).pack(side="left")

        # Fixed bottom status line: contextual help never scrolls away or consumes page space.
        self._footer_bar = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        self._footer_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Separator(self._footer_bar, orient="horizontal").pack(fill="x", pady=(0, 3))
        footer_line = ttk.Frame(self._footer_bar, style=WORKBENCH_FRAME_STYLE)
        footer_line.pack(fill="x")
        ttk.Label(footer_line, text="操作提示｜", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).pack(side="left")
        ttk.Button(footer_line, text="全狀態刷新", command=self._refresh_all_trading_state, style=WORKBENCH_BUTTON_STYLE).pack(side="right", padx=(8, 0))
        self._footer_hint_label = ttk.Label(footer_line, textvariable=self._footer_hint_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED)
        self._footer_hint_label.pack(side="left", fill="x", expand=True)
        self._bind_footer_hint(workflow_box, WORKFLOW_HINT)
        self._bind_footer_hint(table_box, POSITION_DECISION_HINT)
        self._bind_footer_hint(candidate_box, SCANNER_HINT)
        self._bind_footer_hint(pending_box, PENDING_ENTRY_HINT)
        self._bind_footer_hint(trade_box, BUY_ENTRY_HINT)
        self._bind_page_mousewheel(operations_box)
        self._bind_page_mousewheel(content)

    def _sync_page_scrollregion(self, _event=None) -> None:
        try:
            self._page_canvas.configure(scrollregion=self._page_canvas.bbox("all"))
        except tk.TclError as exc:
            _warn_gui_fallback("Trading page scrollregion", exc)

    def _sync_page_content_width(self, event=None) -> None:
        try:
            width = max(1, int(event.width if event is not None else self._page_canvas.winfo_width()))
            self._page_canvas.itemconfigure(self._page_window, width=width)
        except (tk.TclError, TypeError, ValueError) as exc:
            _warn_gui_fallback("Trading page content width", exc)

    def _on_page_mousewheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120) if event.delta else 0
        if delta:
            self._page_canvas.yview_scroll(delta, "units")
            return "break"
        return None

    def _bind_page_mousewheel(self, widget) -> None:
        targets = [widget]
        index = 0
        while index < len(targets):
            target = targets[index]
            index += 1
            try:
                for child in target.winfo_children():
                    if child not in targets:
                        targets.append(child)
                target.bind("<MouseWheel>", self._on_page_mousewheel, add="+")
                target.bind("<Button-4>", self._on_page_mousewheel, add="+")
                target.bind("<Button-5>", self._on_page_mousewheel, add="+")
            except tk.TclError as exc:
                _warn_gui_fallback("Trading page mousewheel bind", exc)
