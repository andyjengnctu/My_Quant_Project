"""Backward-compatible alias for training display policy.

Display-only settings live in config.training_display_policy.  Keep this
module as a thin alias so legacy imports cannot drift to a different
SYSTEM_SCORE_DISPLAY_MULTIPLIER.
"""

from config.training_display_policy import (  # noqa: F401
    SYSTEM_SCORE_DISPLAY_MULTIPLIER,
    build_display_policy_snapshot,
    format_system_score_for_display,
    scale_system_score_for_display,
)

__all__ = [
    "SYSTEM_SCORE_DISPLAY_MULTIPLIER",
    "scale_system_score_for_display",
    "format_system_score_for_display",
    "build_display_policy_snapshot",
]
