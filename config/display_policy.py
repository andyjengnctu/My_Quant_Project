"""Console / report display policy.

This module is the single source for display-only settings.  It must not
change trading rules, optimizer scoring, replay, or selection behavior.
"""

# 系統得分顯示倍率，僅影響 console/report 顯示，不影響 score 公式或排序。
SYSTEM_SCORE_DISPLAY_MULTIPLIER = 100000.0


def build_display_policy_snapshot() -> dict:
    return {
        "SYSTEM_SCORE_DISPLAY_MULTIPLIER": float(SYSTEM_SCORE_DISPLAY_MULTIPLIER),
    }
