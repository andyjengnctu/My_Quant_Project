"""Project-wide semantic color and judgment rules for human-readable reports.

Renderers must classify a metric using an explicit metric contract first, then use
this module to map that semantic signal to text color. A numeric sign alone is
not enough unless the metric contract defines it as meaningful.

Semantic judgment must never rely on traffic-light / circle emoji. Console
reports color the judgment text/value itself with ANSI. Markdown reports use
inline HTML text color so the same semantic palette is preserved without icons.
"""

from __future__ import annotations

import math
import re
from typing import Any

from core.display_common import C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, console_color_enabled

SIGNAL_POSITIVE = "positive"
SIGNAL_NEGATIVE = "negative"
SIGNAL_WARNING = "warning"
SIGNAL_NEUTRAL = "neutral"

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
_SIGNAL_MARKDOWN_COLORS = {
    SIGNAL_POSITIVE: "#188038",
    SIGNAL_NEGATIVE: "#C62828",
    SIGNAL_WARNING: "#B06000",
    SIGNAL_NEUTRAL: "#667085",
}
_TONE_MARKDOWN_COLORS = {
    "green": "#188038",
    "red": "#C62828",
    "yellow": "#B06000",
    "light_yellow": "#D6B53A",
    "gray": "#667085",
    "cyan": "#42A5F5",
    "blue": "#42A5F5",
}


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None



def signal_for_workflow_status(value: Any) -> str:
    """Classify shared workflow/action status text for human-readable reports.

    This is the SSOT for execution-plan status colors.  Only statuses with a
    clear readiness/failure meaning receive positive/warning/negative signals;
    execution actions without an inherent good/bad meaning remain neutral.
    """

    status = str(value or "").strip().upper()
    positive = {"READY", "REUSE", "DONE", "PASS"}
    warning = {
        "PREPARABLE", "BUILD", "REBUILD", "RESUME", "MIGRATE", "DERIVE",
        "CHECK", "WARN", "WARNING", "PARTIAL",
    }
    negative = {"BLOCKED", "NOT_RUN", "FAIL", "FAILED", "ERROR"}
    neutral_actions = {"RUN", "REPORT", "RUN/REUSE", "PARAM+REPLAY", "TRAIN+REPLAY"}
    if status in neutral_actions:
        return SIGNAL_NEUTRAL
    if status in positive:
        return SIGNAL_POSITIVE
    if status in warning:
        return SIGNAL_WARNING
    if status in negative:
        return SIGNAL_NEGATIVE

    # Progress/status labels often add a scope prefix (for example
    # ``[TRAIN DONE]`` or ``[BASELINE REUSE]``).  Parse semantic status tokens
    # here instead of teaching every renderer a second color vocabulary.
    tokens = set(re.findall(r"[A-Z][A-Z0-9_]*", status))
    if tokens & negative:
        return SIGNAL_NEGATIVE
    if tokens & warning:
        return SIGNAL_WARNING
    if tokens & positive:
        return SIGNAL_POSITIVE
    return SIGNAL_NEUTRAL


def styled_workflow_status(
    value: Any,
    *,
    target: str = "console",
    bold: bool = True,
) -> str:
    """Style a workflow/action status through the project-wide semantic palette."""

    return styled_signal(
        value,
        signal_for_workflow_status(value),
        target=target,
        bold=bold,
    )


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


def best_worst_signals(
    values_by_id: dict[str, Any],
    *,
    preference: str,
    tolerance: float = 1e-12,
) -> dict[str, str]:
    """Mark only the best and worst comparable values; all others stay unstyled.

    This is the canonical multi-arm aggregate color contract. Metrics without a
    universal higher/lower preference intentionally return no signal.
    """

    if preference not in {"higher", "lower"}:
        return {}
    finite = {str(key): finite_number(value) for key, value in values_by_id.items()}
    finite = {key: value for key, value in finite.items() if value is not None}
    if len(finite) < 2:
        return {}
    numeric_values = list(finite.values())
    best = max(numeric_values) if preference == "higher" else min(numeric_values)
    worst = min(numeric_values) if preference == "higher" else max(numeric_values)
    if math.isclose(best, worst, rel_tol=0.0, abs_tol=float(tolerance)):
        return {}
    signals: dict[str, str] = {}
    for key, value in finite.items():
        if math.isclose(value, best, rel_tol=0.0, abs_tol=float(tolerance)):
            signals[key] = SIGNAL_POSITIVE
        elif math.isclose(value, worst, rel_tol=0.0, abs_tol=float(tolerance)):
            signals[key] = SIGNAL_NEGATIVE
    return signals


def signal_label(signal: str) -> str:
    """Return the plain semantic judgment word, without any icon."""

    return _SIGNAL_LABELS.get(str(signal), _SIGNAL_LABELS[SIGNAL_NEUTRAL])


def signal_marker(signal: str, *, include_label: bool = True) -> str:
    """Backward-compatible alias with icon-free semantics.

    New renderers should prefer :func:`signal_label` and color the text/value
    itself.  ``include_label=False`` intentionally returns an empty string so
    legacy call sites can never re-introduce traffic-light glyphs.
    """

    return signal_label(signal) if include_label else ""


def tone_for_signal(signal: str) -> str:
    return _SIGNAL_TONES.get(str(signal), _SIGNAL_TONES[SIGNAL_NEUTRAL])


def terminal_color_for_signal(signal: str) -> str:
    """Return the shared ANSI color code for an already-classified signal."""

    return _SIGNAL_COLORS.get(str(signal), _SIGNAL_COLORS[SIGNAL_NEUTRAL])


def terminal_signal(text: str, signal: str, *, enabled: bool | None = None) -> str:
    """Color the text itself for console output; never prefix an icon."""

    use_color = console_color_enabled() if enabled is None else bool(enabled)
    if not use_color:
        return str(text)
    color = _SIGNAL_COLORS.get(str(signal), C_GRAY)
    return f"{color}{text}{C_RESET}"


def markdown_tone(text: Any, tone: str, *, bold: bool = False) -> str:
    """Color Markdown text with the project-wide HTML palette, without icons."""

    raw = str(text)
    color = _TONE_MARKDOWN_COLORS.get(str(tone), _TONE_MARKDOWN_COLORS["gray"])
    weight = "font-weight:700;" if bool(bold) else ""
    return f'<span style="color:{color};{weight}">{raw}</span>'


def markdown_signal(text: Any, signal: str, *, bold: bool = False) -> str:
    """Color Markdown text using the same semantic signal palette as console."""

    raw = str(text)
    color = _SIGNAL_MARKDOWN_COLORS.get(str(signal), _SIGNAL_MARKDOWN_COLORS[SIGNAL_NEUTRAL])
    weight = "font-weight:700;" if bool(bold) else ""
    return f'<span style="color:{color};{weight}">{raw}</span>'


def styled_signal(
    text: Any,
    signal: str,
    *,
    target: str,
    enabled: bool | None = None,
    bold: bool = False,
) -> str:
    """Render semantic text consistently for console, Markdown or plain output."""

    normalized = str(target).strip().lower()
    if normalized == "console":
        return terminal_signal(str(text), signal, enabled=enabled)
    if normalized == "markdown":
        return markdown_signal(text, signal, bold=bold)
    if normalized == "plain":
        return str(text)
    raise ValueError(f"不支援的report target: {target!r}")


__all__ = [
    "SIGNAL_POSITIVE",
    "SIGNAL_NEGATIVE",
    "SIGNAL_WARNING",
    "SIGNAL_NEUTRAL",
    "finite_number",
    "signal_for_workflow_status",
    "signal_for_delta",
    "signal_for_signed_value",
    "signal_for_coverage",
    "signal_for_ratio",
    "signal_for_auc",
    "best_worst_signals",
    "signal_label",
    "signal_marker",
    "tone_for_signal",
    "terminal_color_for_signal",
    "terminal_signal",
    "markdown_tone",
    "markdown_signal",
    "styled_signal",
    "styled_workflow_status",
]
