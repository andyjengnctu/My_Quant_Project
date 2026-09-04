from __future__ import annotations

import math

from core.display_policy import format_system_score_for_display, scale_system_score_for_display
from services.optimizer.study_utils import INVALID_TRIAL_VALUE


def _coerce_finite_score(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_optimizer_score_for_display(value, *, decimals: int = 3, invalid_text: str = "N/A") -> str:
    """Format an internal optimizer system score for human-readable output only."""
    number = _coerce_finite_score(value)
    if number is None:
        return str(invalid_text)
    precision = max(0, int(decimals))
    if number <= float(INVALID_TRIAL_VALUE):
        return f"{number:.{precision}f}"
    return format_system_score_for_display(number, decimals=precision)


def scale_optimizer_score_for_display(value, *, default: float = 0.0) -> float:
    """Scale a finite optimizer system score while preserving the invalid sentinel."""
    number = _coerce_finite_score(value)
    if number is None:
        return float(default)
    if number <= float(INVALID_TRIAL_VALUE):
        return number
    return scale_system_score_for_display(number)
