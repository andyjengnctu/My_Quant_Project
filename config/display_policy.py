"""Console / report display policy.

This module is the single source for display-only settings.  It must not
change trading rules, optimizer scoring, replay, or selection behavior.
"""

# 系統得分顯示倍率，僅影響 console/report 顯示，不影響 score 公式或排序。
SYSTEM_SCORE_DISPLAY_MULTIPLIER = 100000.0

# 非 rolling 訓練結果表格顯示開關。
# True  = 訓練結束後顯示 candidate / seed ensemble 結果表格。
# False = 仍計算與輸出正式 artifacts，但不顯示結果表格，降低 console 洗版。
OPTIMIZER_NONROLLING_TRAIN_RESULT_TABLE_ENABLED = False


def is_optimizer_nonrolling_train_result_table_enabled() -> bool:
    return bool(OPTIMIZER_NONROLLING_TRAIN_RESULT_TABLE_ENABLED)


def build_display_policy_snapshot() -> dict:
    return {
        "SYSTEM_SCORE_DISPLAY_MULTIPLIER": float(SYSTEM_SCORE_DISPLAY_MULTIPLIER),
        "OPTIMIZER_NONROLLING_TRAIN_RESULT_TABLE_ENABLED": is_optimizer_nonrolling_train_result_table_enabled(),
    }
