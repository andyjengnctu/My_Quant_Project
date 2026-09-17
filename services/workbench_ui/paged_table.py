from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Iterable

from services.workbench_ui.selection_behavior import toggled_row_selection
from services.workbench_ui.workbench import (
    WORKBENCH_ACCENT,
    WORKBENCH_BG,
    WORKBENCH_BORDER,
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_ERROR,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_SUCCESS,
    WORKBENCH_TEXT,
)


@dataclass(frozen=True)
class TableColumn:
    key: str
    label: str
    width: int = 10
    anchor: str = "center"
    sort_kind: str = "text"  # text | numeric | date
    performance: bool = False
    formatter: Callable[[Any, dict[str, Any]], str] | None = None


class PagedTable(ttk.Frame):
    """Workbench table with shared sort/select/page/open-stock semantics.

    The implementation intentionally uses individual cell labels instead of a
    ttk.Treeview so performance colouring can be applied to specific cells only.
    """

    def __init__(
        self,
        master,
        *,
        columns: Iterable[TableColumn],
        page_size: int = 12,
        id_key: str = "_table_id",
        stock_key: str | None = "ticker",
        default_sort_key: str | None = None,
        default_desc: bool = False,
        empty_text: str = "目前無資料",
        on_select: Callable[[dict[str, Any] | None], None] | None = None,
        on_open_stock: Callable[[str], None] | None = None,
        sortable: bool = True,
        on_mousewheel: Callable[[Any], Any] | None = None,
        on_pointer_enter: Callable[[Any], Any] | None = None,
        on_pointer_leave: Callable[[Any], Any] | None = None,
    ):
        super().__init__(master, style=WORKBENCH_FRAME_STYLE)
        self.columns = tuple(columns)
        self.page_size = max(1, int(page_size))
        self.id_key = str(id_key)
        self.stock_key = stock_key
        self.default_sort_key = default_sort_key
        self._sort_key = default_sort_key
        self._sort_desc = bool(default_desc)
        self._sort_user_clicked = False
        self.empty_text = str(empty_text)
        self.on_select = on_select
        self.on_open_stock = on_open_stock
        self.sortable = bool(sortable)
        self.on_mousewheel = on_mousewheel
        self.on_pointer_enter = on_pointer_enter
        self.on_pointer_leave = on_pointer_leave
        self._rows: list[dict[str, Any]] = []
        self._selected_id: str | None = None
        self._page = 0
        self._body_widgets: list[tk.Widget] = []

        self.columnconfigure(0, weight=1)
        self._grid = tk.Frame(self, bg=WORKBENCH_BG, highlightthickness=0, bd=0)
        self._grid.grid(row=0, column=0, sticky="ew")
        self._empty = ttk.Label(self, text=self.empty_text, foreground=WORKBENCH_MUTED)
        self._pager = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        self._prev = ttk.Button(self._pager, text="上一頁", command=lambda: self._change_page(-1), style=WORKBENCH_BUTTON_STYLE)
        self._page_var = tk.StringVar(value="")
        self._page_label = ttk.Label(self._pager, textvariable=self._page_var, foreground=WORKBENCH_MUTED)
        self._next = ttk.Button(self._pager, text="下一頁", command=lambda: self._change_page(1), style=WORKBENCH_BUTTON_STYLE)
        self._prev.pack(side="left")
        self._page_label.pack(side="left", padx=10)
        self._next.pack(side="left")
        for widget in (self, self._grid, self._empty, self._pager, self._prev, self._page_label, self._next):
            self._bind_mousewheel(widget)
            self._bind_pointer_callbacks(widget)
        self._render()

    def _bind_mousewheel(self, widget) -> None:
        if not callable(self.on_mousewheel):
            return
        widget.bind("<MouseWheel>", self.on_mousewheel, add="+")
        widget.bind("<Button-4>", self.on_mousewheel, add="+")
        widget.bind("<Button-5>", self.on_mousewheel, add="+")

    def _bind_pointer_callbacks(self, widget) -> None:
        if callable(self.on_pointer_enter):
            widget.bind("<Enter>", self.on_pointer_enter, add="+")
        if callable(self.on_pointer_leave):
            widget.bind("<Leave>", self.on_pointer_leave, add="+")

    def set_columns(self, columns: Iterable[TableColumn]) -> None:
        """Replace the visible schema while preserving rows/selection when possible."""
        normalized = tuple(columns)
        keys = {column.key for column in normalized}
        self.columns = normalized
        if self._sort_key not in keys:
            self._sort_key = self.default_sort_key if self.default_sort_key in keys else None
            self._sort_desc = False
            self._sort_user_clicked = False
        self._page = min(self._page, max(0, self.page_count - 1))
        self._render()

    def set_rows(self, rows: Iterable[dict[str, Any]], *, preserve_selection: bool = True) -> None:
        normalized: list[dict[str, Any]] = []
        for idx, raw in enumerate(rows):
            row = dict(raw)
            if not row.get(self.id_key):
                row[self.id_key] = f"row:{idx}:{row.get(self.stock_key or '', '')}:{row.get('revision', '')}"
            normalized.append(row)
        self._rows = normalized
        if not preserve_selection or self._selected_id not in {str(row.get(self.id_key)) for row in normalized}:
            self._selected_id = None
        self._page = min(self._page, max(0, self.page_count - 1))
        self._render()

    @property
    def page_count(self) -> int:
        return max(1, math.ceil(len(self._rows) / self.page_size))

    def clear_selection(self, *, notify: bool = True) -> None:
        changed = self._selected_id is not None
        self._selected_id = None
        self._render()
        if changed and notify and callable(self.on_select):
            self.on_select(None)

    def selected_row(self) -> dict[str, Any] | None:
        if self._selected_id is None:
            return None
        for row in self._rows:
            if str(row.get(self.id_key)) == self._selected_id:
                return dict(row)
        return None

    def selected_id(self) -> str | None:
        return self._selected_id

    def _sort_value(self, row: dict[str, Any], column: TableColumn):
        value = row.get(column.key)
        missing = value is None or value == ""
        if missing:
            return (1, 0)
        if column.sort_kind == "numeric":
            try:
                return (0, float(value))
            except (TypeError, ValueError):
                return (0, 0.0)
        if column.sort_kind == "date":
            text = str(value)
            try:
                return (0, datetime.strptime(text[:10], "%Y-%m-%d").date().toordinal())
            except ValueError:
                return (0, text)
        return (0, str(value).casefold())

    def _sorted_rows(self) -> list[dict[str, Any]]:
        rows = list(self._rows)
        if not self._sort_key:
            return rows
        column = next((col for col in self.columns if col.key == self._sort_key), None)
        if column is None:
            return rows
        return sorted(rows, key=lambda row: self._sort_value(row, column), reverse=self._sort_desc)

    def _format(self, column: TableColumn, row: dict[str, Any]) -> str:
        value = row.get(column.key)
        if column.formatter is not None:
            return str(column.formatter(value, row))
        return "-" if value is None else str(value)

    @staticmethod
    def _performance_colour(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return WORKBENCH_TEXT
        if number > 0:
            return WORKBENCH_ERROR  # Taiwan market convention: gain red
        if number < 0:
            return WORKBENCH_SUCCESS  # loss green
        return WORKBENCH_TEXT

    def _header_click(self, key: str) -> None:
        if not self.sortable:
            return
        # First user click on any column is ascending; the next click on that same
        # column reverses it.  A default ascending sort (for example trade date)
        # does not consume that first-click contract.
        if not self._sort_user_clicked or self._sort_key != key:
            self._sort_key = key
            self._sort_desc = False
        else:
            self._sort_desc = not self._sort_desc
        self._sort_user_clicked = True
        self._page = 0
        self._render()

    def _row_click(self, row_id: str) -> None:
        self._selected_id = toggled_row_selection(self._selected_id, row_id)
        selected = self.selected_row()
        self._render()
        if callable(self.on_select):
            self.on_select(selected)

    def _open_stock(self, ticker: Any) -> None:
        ticker_text = str(ticker or "").strip().upper()
        if ticker_text and callable(self.on_open_stock):
            self.on_open_stock(ticker_text)

    def _change_page(self, delta: int) -> None:
        new_page = min(max(0, self._page + int(delta)), max(0, self.page_count - 1))
        if new_page != self._page:
            self._page = new_page
            self._render()

    def _clear_grid(self) -> None:
        for widget in self._grid.winfo_children():
            widget.destroy()

    def _render(self) -> None:
        self._clear_grid()
        self._empty.grid_remove()
        self._pager.grid_remove()
        rows = self._sorted_rows()
        if not rows:
            self._empty.grid(row=0, column=0, sticky="w", pady=4)
            return

        visible_columns: list[tuple[str, TableColumn | None]] = []
        if self.stock_key:
            visible_columns.append(("__open__", None))
        visible_columns.extend((col.key, col) for col in self.columns)

        for col_idx, (key, column) in enumerate(visible_columns):
            self._grid.grid_columnconfigure(col_idx, weight=0 if key == "__open__" else 1)
            if key == "__open__":
                text = "↗"
                width = 3
                cursor = "arrow"
                command = None
            else:
                arrow = ""
                if self.sortable and self._sort_key == key:
                    arrow = " ▼" if self._sort_desc else " ▲"
                text = f"{column.label}{arrow}"
                width = max(4, int(column.width))
                cursor = "hand2" if self.sortable else "arrow"
                command = (lambda _event=None, sort_key=key: self._header_click(sort_key)) if self.sortable else None
            label = tk.Label(
                self._grid,
                text=text,
                width=width,
                anchor="center",
                bg="#101a25",
                fg=WORKBENCH_TEXT,
                relief="solid",
                borderwidth=1,
                padx=3,
                pady=4,
                cursor=cursor,
            )
            label.grid(row=0, column=col_idx, sticky="nsew")
            self._bind_mousewheel(label)
            self._bind_pointer_callbacks(label)
            if command is not None:
                label.bind("<Button-1>", command)

        start = self._page * self.page_size
        page_rows = rows[start : start + self.page_size]
        for row_idx, row in enumerate(page_rows, start=1):
            row_id = str(row.get(self.id_key))
            selected = row_id == self._selected_id
            bg = WORKBENCH_ACCENT if selected else WORKBENCH_BG
            for col_idx, (key, column) in enumerate(visible_columns):
                if key == "__open__":
                    text = "▣"
                    fg = "#7db7ff"
                    anchor = "center"
                    width = 3
                else:
                    text = self._format(column, row)
                    fg = self._performance_colour(row.get(column.key)) if column.performance else WORKBENCH_TEXT
                    anchor = column.anchor
                    width = max(4, int(column.width))
                cell = tk.Label(
                    self._grid,
                    text=text,
                    width=width,
                    anchor=anchor,
                    bg=bg,
                    fg=fg,
                    relief="solid",
                    borderwidth=1,
                    padx=3,
                    pady=3,
                    cursor="hand2",
                )
                cell.grid(row=row_idx, column=col_idx, sticky="nsew")
                self._bind_mousewheel(cell)
                self._bind_pointer_callbacks(cell)
                if key == "__open__":
                    cell.bind("<Button-1>", lambda _event, ticker=row.get(self.stock_key): self._open_stock(ticker))
                else:
                    cell.bind("<Button-1>", lambda _event, rid=row_id: self._row_click(rid))

        if len(rows) > self.page_size:
            self._pager.grid(row=1, column=0, sticky="w", pady=(6, 0))
            self._page_var.set(f"第 {self._page + 1}/{self.page_count} 頁")
            self._prev.configure(state="normal" if self._page > 0 else "disabled")
            self._next.configure(state="normal" if self._page < self.page_count - 1 else "disabled")


__all__ = ["TableColumn", "PagedTable"]
