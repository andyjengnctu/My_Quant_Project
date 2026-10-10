from __future__ import annotations

import re
import tkinter as tk
from tkinter import font as tkfont
from services.workbench_ui.workbench import WORKBENCH_BG, WORKBENCH_ERROR, WORKBENCH_INFO, WORKBENCH_MUTED, WORKBENCH_SUCCESS, WORKBENCH_TEXT, WORKBENCH_UI_FONT, WORKBENCH_WARNING, _warn_gui_fallback

_STATUS_TOKEN_TONES = {
    "READY": "success",
    "IDLE": "success",
    "FRESH": "success",
    "CLEAR": "success",
    "SYNCED": "success",
    "UPDATED": "success",
    "NO_DUE": "success",
    "FILLED": "success",
    "CANCELLED": "success",
    "完成": "success",
    "可沿用": "success",
    "NOT READY": "warning",
    "ACTION_REQUIRED": "warning",
    "NOT_BOOTSTRAPPED": "warning",
    "STALE/EMPTY": "warning",
    "STALE": "warning",
    "ORDERED": "warning",
    "PARTIAL": "warning",
    "PROPOSED": "warning",
    "未綁定": "warning",
    "尚無資料": "warning",
    "尚無 Params": "warning",
    "尚未": "warning",
    "注意": "warning",
    "WAIT": "warning",
    "BLOCKED": "error",
    "LIVE_BLOCKED": "error",
    "FAIL": "error",
    "FAILED": "error",
    "ERROR": "error",
    "BLOCK": "error",
    "失敗": "error",
    "執行中": "info",
    "沿用既有": "info",
    "重新訓練": "info",
    "ON": "info",
    "OFF": "info",
}
_STATUS_WORDS = sorted(_STATUS_TOKEN_TONES, key=len, reverse=True)
_STATUS_TOKEN_REGEXES = [
    (
        rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
        if re.search(r"[A-Za-z]", token)
        else re.escape(token)
    )
    for token in _STATUS_WORDS
]
_STATUS_TOKEN_PATTERN = re.compile(
    "("
    + "|".join(_STATUS_TOKEN_REGEXES)
    + r"|20\d{2}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:[.+-]\d{2}:?\d{2})?)?"
    + r"|(?<![A-Za-z0-9_])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:/\d+(?:\.\d+)?)?(?![A-Za-z0-9_])"
    + ")",
    re.IGNORECASE,
)


def build_trading_status_segments(text: str, *, default_tone: str = "text") -> list[tuple[str, str]]:
    """Split a status line into neutral text plus highlighted status/date/value tokens."""
    value = str(text or "")
    segments: list[tuple[str, str]] = []
    cursor = 0
    for match in _STATUS_TOKEN_PATTERN.finditer(value):
        if match.start() > cursor:
            segments.append((value[cursor:match.start()], default_tone))
        token = match.group(0)
        tone = _STATUS_TOKEN_TONES.get(token.upper())
        if tone is None:
            tone = _STATUS_TOKEN_TONES.get(token, "info")
        segments.append((token, tone))
        cursor = match.end()
    if cursor < len(value):
        segments.append((value[cursor:], default_tone))
    if not segments:
        segments.append((value, default_tone))
    return segments

class _TradingStatusLine(tk.Text):
    """Read-only wrapping status text with token-level emphasis instead of whole-line coloring."""

    _TONE_COLORS = {
        "text": WORKBENCH_TEXT,
        "muted": WORKBENCH_MUTED,
        "info": WORKBENCH_INFO,
        "success": WORKBENCH_SUCCESS,
        "warning": WORKBENCH_WARNING,
        "error": WORKBENCH_ERROR,
    }

    def __init__(self, master, *, textvariable: tk.StringVar, default_tone: str = "text", min_lines: int = 1, max_lines: int = 3):
        self._textvariable = textvariable
        self._default_tone = default_tone
        self._min_lines = max(1, int(min_lines))
        self._max_lines = max(self._min_lines, int(max_lines))
        self._trace_id = None
        super().__init__(
            master,
            height=self._min_lines,
            width=1,
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            background=WORKBENCH_BG,
            foreground=WORKBENCH_TEXT,
            insertbackground=WORKBENCH_TEXT,
            font=WORKBENCH_UI_FONT,
            takefocus=0,
            cursor="arrow",
        )
        self._measure_font = tkfont.Font(font=WORKBENCH_UI_FONT)
        for tone, color in self._TONE_COLORS.items():
            self.tag_configure(tone, foreground=color)
        self._trace_id = self._textvariable.trace_add("write", self._on_text_changed)
        self.bind("<Configure>", self._schedule_height_sync, add="+")
        self.set_status_text(self._textvariable.get())

    def _on_text_changed(self, *_args) -> None:
        self.set_status_text(self._textvariable.get())

    def set_status_text(self, text: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        for segment, tone in build_trading_status_segments(text, default_tone=self._default_tone):
            self.insert("end", segment, tone)
        self.configure(state="disabled")
        self._schedule_height_sync()

    def _schedule_height_sync(self, *_args) -> None:
        try:
            self.after_idle(self._sync_height)
        except tk.TclError as exc:
            _warn_gui_fallback("TradingStatusText.after_idle(_sync_height)", exc)
            return

    def _sync_height(self) -> None:
        try:
            if not self.winfo_exists():
                return
            available_width = int(self.winfo_width()) - 12
            if available_width <= 40:
                return
            display_lines = 0
            logical_lines = str(self._textvariable.get() or "").splitlines() or [""]
            for logical_line in logical_lines:
                measured = max(1, int(self._measure_font.measure(logical_line)))
                display_lines += max(1, (measured + available_width - 1) // available_width)
            target = max(self._min_lines, min(self._max_lines, display_lines))
            if int(self.cget("height")) != target:
                self.configure(height=target)
        except (tk.TclError, TypeError, ValueError) as exc:
            _warn_gui_fallback("TradingStatusText._sync_height()", exc)
            return

    def destroy(self) -> None:
        if self._trace_id is not None:
            try:
                self._textvariable.trace_remove("write", self._trace_id)
            except tk.TclError as exc:
                _warn_gui_fallback("TradingStatusText.trace_remove(write)", exc)
                pass
            self._trace_id = None
        super().destroy()
