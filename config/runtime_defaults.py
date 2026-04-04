# 純設定資料：全域戰略開關、顯示倍率與 runtime 預設值；score/threshold 已移至 training_policy。

# ==========================================
# 🌟 全域戰略切換開關 (System-Wide Strategy Switches)
# ==========================================
# 1. 期望值 (EV) 算法切換
# 'A' = 嚴格 R_Multiple 期望值 (Mean R)
# 'B' = 傳統實際盈虧期望值 (Win% * Payoff - Loss%)
EV_CALC_METHOD = 'A'

# 2. 買入優先序切換開關
# 'PROJ_COST' = 優先買入能消耗最多資金的標的 (資金效率極大化)
# 'EV'        = 優先買入期望值最高的標的 (單筆質量極大化)
# 'HIST_WIN_X_TRADES' = 優先買入歷史勝率 × 交易次數最高的標的 (穩定度 × 樣本數)
BUY_SORT_METHOD = 'HIST_WIN_X_TRADES'

# 3. score / threshold 類全域訓練政策已集中至 config/training_policy.py
# ==========================================

# # (AI註: 僅影響顯示，不改實際排序與優化邏輯)
SYSTEM_SCORE_DISPLAY_MULTIPLIER = 1000.0

RUNTIME_PARAM_SPECS = {
    'optimizer_max_workers': {'type': int, 'default': None, 'allow_none': True, 'min_value': 1},
    'scanner_max_workers': {'type': int, 'default': None, 'allow_none': True, 'min_value': 1},
    'scanner_live_capital': {'type': float, 'default': 2_000_000.0, 'allow_none': False, 'min_value': 0.0},
}

RUNTIME_PARAM_DEFAULTS = {field_name: spec['default'] for field_name, spec in RUNTIME_PARAM_SPECS.items()}
RUNTIME_PARAM_TYPES = {field_name: spec['type'] for field_name, spec in RUNTIME_PARAM_SPECS.items()}
