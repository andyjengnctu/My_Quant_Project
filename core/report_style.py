"""Project-wide semantic color and judgment rules for human-readable reports.

Renderers must classify a metric using an explicit metric contract first, then use
this module to map that semantic signal to markers/terminal colors.  A numeric
sign alone is not enough unless the metric contract defines it as meaningful.
"""

from __future__ import annotations

import math
from typing import Any

from core.display_common import C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, console_color_enabled

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
_SIGNAL_TONES = {
    SIGNAL_POSITIVE: "green",
    SIGNAL_NEGATIVE: "red",
    SIGNAL_WARNING: "yellow",
    SIGNAL_NEUTRAL: "gray",
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
    """Classify a delta using an explicit metric-direction contract."""

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


def signal_for_signed_value(value: Any, *, tolerance: float = 1e-12) -> str:
    """Classify a value whose sign is itself the documented quality contract."""

    numeric = finite_number(value)
    if numeric is None or abs(numeric) <= float(tolerance):
        return SIGNAL_NEUTRAL
    return SIGNAL_POSITIVE if numeric > 0 else SIGNAL_NEGATIVE


def signal_for_coverage(value: Any, *, tolerance: float = 1e-12) -> str:
    """Complete coverage is positive, partial coverage warns, zero is negative."""

    numeric = finite_number(value)
    if numeric is None:
        return SIGNAL_NEUTRAL
    if numeric >= 1.0 - float(tolerance):
        return SIGNAL_POSITIVE
    if numeric > 0.0:
        return SIGNAL_WARNING
    return SIGNAL_NEGATIVE


def signal_for_ratio(numerator: Any, denominator: Any) -> str:
    """Full ratio is positive, >=50% warns, below 50% is negative."""

    top = finite_number(numerator)
    bottom = finite_number(denominator)
    if top is None or bottom is None or bottom <= 0.0:
        return SIGNAL_NEUTRAL
    ratio = top / bottom
    if ratio >= 1.0 - 1e-12:
        return SIGNAL_POSITIVE
    if ratio >= 0.5:
        return SIGNAL_WARNING
    return SIGNAL_NEGATIVE


def signal_for_auc(value: Any) -> str:
    """AUC above random is positive, below random negative, exactly 0.5 neutral."""

    numeric = finite_number(value)
    if numeric is None or abs(numeric - 0.5) <= 1e-12:
        return SIGNAL_NEUTRAL
    return SIGNAL_POSITIVE if numeric > 0.5 else SIGNAL_NEGATIVE


def signal_marker(signal: str, *, include_label: bool = True) -> str:
    normalized = str(signal)
    marker = _SIGNAL_MARKERS.get(normalized, _SIGNAL_MARKERS[SIGNAL_NEUTRAL])
    if not include_label:
        return marker
    return f"{marker} {_SIGNAL_LABELS.get(normalized, _SIGNAL_LABELS[SIGNAL_NEUTRAL])}"


def tone_for_signal(signal: str) -> str:
    return _SIGNAL_TONES.get(str(signal), _SIGNAL_TONES[SIGNAL_NEUTRAL])


def terminal_color_for_signal(signal: str) -> str:
    """Return the shared ANSI color code for an already-classified signal."""

    return _SIGNAL_COLORS.get(str(signal), _SIGNAL_COLORS[SIGNAL_NEUTRAL])


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
    "signal_for_signed_value",
    "signal_for_coverage",
    "signal_for_ratio",
    "signal_for_auc",
    "signal_marker",
    "tone_for_signal",
    "terminal_color_for_signal",
    "terminal_signal",
]
