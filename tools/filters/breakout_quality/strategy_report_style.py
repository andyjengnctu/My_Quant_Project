"""Shared readable/color semantics for breakout-quality strategy diagnostics."""

from __future__ import annotations

import math
from typing import Any

from core.display import C_GREEN, C_GRAY, C_RED, C_RESET, C_YELLOW
from filters.breakout_quality.console_report import console_color_enabled

SIGNAL_POSITIVE = "positive"
SIGNAL_NEGATIVE = "negative"
SIGNAL_WARNING = "warning"
SIGNAL_NEUTRAL = "neutral"

_SIGNAL_MARKERS = {
    SIGNAL_POSITIVE: "🟢",
    SIGNAL_NEGATIVE: "🔴",
    SIGNAL_WARNING: "🟡",
    SIGNAL_NEUTRAL: "⚪",
}
_SIGNAL_LABELS = {
    SIGNAL_POSITIVE: "改善",
    SIGNAL_NEGATIVE: "惡化",
    SIGNAL_WARNING: "注意",
    SIGNAL_NEUTRAL: "中性",
}
_SIGNAL_COLORS = {
    SIGNAL_POSITIVE: C_GREEN,
    SIGNAL_NEGATIVE: C_RED,
    SIGNAL_WARNING: C_YELLOW,
    SIGNAL_NEUTRAL: C_GRAY,
}


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def signal_for_delta(
    value: Any,
    *,
    preference: str,
    tolerance: float = 1e-12,
    warning_threshold: float | None = None,
) -> str:
    """Classify a delta using an explicit metric direction contract."""

    numeric = finite_number(value)
    if numeric is None or abs(numeric) <= float(tolerance):
        return SIGNAL_NEUTRAL
    if preference == "higher":
        return SIGNAL_POSITIVE if numeric > 0 else SIGNAL_NEGATIVE
    if preference == "lower":
        return SIGNAL_POSITIVE if numeric < 0 else SIGNAL_NEGATIVE
    if preference == "attention":
        threshold = float(warning_threshold or 0.0)
        return SIGNAL_WARNING if abs(numeric) > threshold else SIGNAL_NEUTRAL
    if preference == "neutral":
        return SIGNAL_NEUTRAL
    raise ValueError(f"不支援的metric preference: {preference!r}")


def signal_marker(signal: str, *, include_label: bool = True) -> str:
    normalized = str(signal)
    marker = _SIGNAL_MARKERS.get(normalized, _SIGNAL_MARKERS[SIGNAL_NEUTRAL])
    if not include_label:
        return marker
    return f"{marker} {_SIGNAL_LABELS.get(normalized, _SIGNAL_LABELS[SIGNAL_NEUTRAL])}"


def terminal_signal(text: str, signal: str, *, enabled: bool | None = None) -> str:
    use_color = console_color_enabled() if enabled is None else bool(enabled)
    if not use_color:
        return str(text)
    color = _SIGNAL_COLORS.get(str(signal), C_GRAY)
    return f"{color}{text}{C_RESET}"


__all__ = [
    "SIGNAL_POSITIVE",
    "SIGNAL_NEGATIVE",
    "SIGNAL_WARNING",
    "SIGNAL_NEUTRAL",
    "finite_number",
    "signal_for_delta",
    "signal_marker",
    "terminal_signal",
]
