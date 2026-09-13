from __future__ import annotations

import calendar
from datetime import date
import tkinter as tk
from tkinter import ttk

from core.runtime_utils import get_taipei_now
from services.workbench_ui.workbench import (
    WORKBENCH_BG,
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_ENTRY_STYLE,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_TEXT,
    _warn_gui_fallback,
)


class DatePickerField(ttk.Frame):
    """YYYY-MM-DD entry with a dependency-free calendar picker."""

    def __init__(self, master, *, textvariable: tk.StringVar, width: int = 14):
        super().__init__(master, style=WORKBENCH_FRAME_STYLE)
        self.variable = textvariable
        self.entry = ttk.Entry(self, textvariable=self.variable, width=width, style=WORKBENCH_ENTRY_STYLE)
        self.entry.pack(side="left", fill="x", expand=True)
        # The date field itself is the picker trigger; keep the textvariable editable
        # so YYYY-MM-DD may still be pasted/typed when that is faster.
        self.entry.bind("<Button-1>", self._on_entry_click, add="+")
        self._popup = None
        self._calendar_year = None
        self._calendar_month = None

    def _on_entry_click(self, _event=None):
        self.after_idle(self._open_calendar)

    def _initial_date(self) -> date:
        raw = self.variable.get().strip()
        if raw:
            parts = raw.split("-")
            if len(parts) == 3 and all(part.isdigit() for part in parts):
                year, month, day = (int(part) for part in parts)
                if 1 <= month <= 12:
                    last_day = calendar.monthrange(year, month)[1]
                    if 1 <= day <= last_day:
                        return date(year, month, day)
        return get_taipei_now().date()

    def _open_calendar(self) -> None:
        if self._popup is not None and self._popup.winfo_exists():
            self._popup.lift()
            return
        selected = self._initial_date()
        self._calendar_year = selected.year
        self._calendar_month = selected.month
        popup = tk.Toplevel(self)
        popup.title("選擇日期")
        popup.configure(bg=WORKBENCH_BG)
        popup.resizable(False, False)
        popup.transient(self.winfo_toplevel())
        self._popup = popup
        body = ttk.Frame(popup, padding=8, style=WORKBENCH_FRAME_STYLE)
        body.pack(fill="both", expand=True)
        self._calendar_body = body
        self._render_calendar()
        popup.protocol("WM_DELETE_WINDOW", self._close_calendar)
        try:
            popup.grab_set()
        except tk.TclError as exc:
            _warn_gui_fallback("DatePickerField.grab_set", exc)

    def _close_calendar(self) -> None:
        popup = self._popup
        self._popup = None
        if popup is not None and popup.winfo_exists():
            try:
                popup.grab_release()
            except tk.TclError as exc:
                _warn_gui_fallback("DatePickerField.grab_release", exc)
            popup.destroy()

    def _shift_month(self, delta: int) -> None:
        year = int(self._calendar_year)
        month = int(self._calendar_month) + int(delta)
        if month < 1:
            year -= 1
            month = 12
        elif month > 12:
            year += 1
            month = 1
        self._calendar_year = year
        self._calendar_month = month
        self._render_calendar()

    def _select_day(self, day: int) -> None:
        chosen = date(int(self._calendar_year), int(self._calendar_month), int(day))
        self.variable.set(chosen.isoformat())
        self._close_calendar()

    def _render_calendar(self) -> None:
        body = self._calendar_body
        for child in body.winfo_children():
            child.destroy()

        header = ttk.Frame(body, style=WORKBENCH_FRAME_STYLE)
        header.grid(row=0, column=0, columnspan=7, sticky="ew", pady=(0, 6))
        ttk.Button(header, text="‹", width=3, command=lambda: self._shift_month(-1), style=WORKBENCH_BUTTON_STYLE).pack(side="left")
        ttk.Label(
            header,
            text=f"{int(self._calendar_year):04d}-{int(self._calendar_month):02d}",
            style=WORKBENCH_LABEL_STYLE,
            foreground=WORKBENCH_TEXT,
        ).pack(side="left", expand=True, padx=12)
        ttk.Button(header, text="›", width=3, command=lambda: self._shift_month(1), style=WORKBENCH_BUTTON_STYLE).pack(side="right")

        for col, label in enumerate(("一", "二", "三", "四", "五", "六", "日")):
            ttk.Label(body, text=label, width=4, anchor="center", style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED).grid(row=1, column=col, padx=1, pady=1)

        month_rows = calendar.Calendar(firstweekday=0).monthdayscalendar(int(self._calendar_year), int(self._calendar_month))
        for row_idx, week in enumerate(month_rows, start=2):
            for col_idx, day in enumerate(week):
                if not day:
                    ttk.Label(body, text="", width=4, style=WORKBENCH_LABEL_STYLE).grid(row=row_idx, column=col_idx, padx=1, pady=1)
                    continue
                ttk.Button(
                    body,
                    text=str(day),
                    width=4,
                    command=lambda value=day: self._select_day(value),
                    style=WORKBENCH_BUTTON_STYLE,
                ).grid(row=row_idx, column=col_idx, padx=1, pady=1)


__all__ = ["DatePickerField"]
