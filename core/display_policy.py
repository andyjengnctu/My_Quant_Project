"""Runtime helpers for display-only policy.

The user-adjustable display value is owned by ``config.display_policy``.  This
module owns coercion, formatting, and snapshot construction so config stays
declarative.
"""

from config.display_policy import SYSTEM_SCORE_DISPLAY_MULTIPLIER


def scale_system_score_for_display(score) -> float:
    try:
        raw_score = float(score)
    except (TypeError, ValueError):
        raw_score = 0.0
    return raw_score * float(SYSTEM_SCORE_DISPLAY_MULTIPLIER)


def format_system_score_for_display(score, *, decimals: int = 2) -> str:
    precision = max(0, int(decimals))
    return f"{scale_system_score_for_display(score):.{precision}f}"


def build_display_policy_snapshot() -> dict:
    return {
        "SYSTEM_SCORE_DISPLAY_MULTIPLIER": float(SYSTEM_SCORE_DISPLAY_MULTIPLIER),
    }
