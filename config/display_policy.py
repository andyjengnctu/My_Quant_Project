"""Console / report display policy.

This module is the single source for display-only settings.  It must not
change trading rules, optimizer scoring, replay, or selection behavior.
"""

# 系統得分顯示倍率，僅影響 console/report 顯示，不影響 score 公式或排序。
SYSTEM_SCORE_DISPLAY_MULTIPLIER = 100000.0


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
