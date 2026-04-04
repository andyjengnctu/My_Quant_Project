# 純設定資料：全域戰略開關、threshold 與 runtime 預設值。

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

# 3. 系統評分 (Score) 算法切換
# 'LOG_R2' = 結合對數 R 平方與月度勝率的不對稱模型 (容許暴漲，尋找平穩向上的聖杯)
# 'RoMD'   = 傳統報酬回撤比風格的基底分數
SCORE_CALC_METHOD = 'RoMD'

# 3-1. 系統評分分子切換
# 'ANNUAL_RETURN' = 分子使用年化報酬率
# 'TOTAL_RETURN'  = 分子使用總報酬率
SCORE_NUMERATOR_METHOD = 'ANNUAL_RETURN'

# 4-1. Optimizer 投組績效門檻
MIN_FULL_YEAR_RETURN_PCT = -30.0
MIN_ANNUAL_TRADES = 5.0
MIN_BUY_FILL_RATE = 70.0
MIN_TRADE_WIN_RATE = 35.0
MAX_PORTFOLIO_MDD_PCT = 45.0
MIN_MONTHLY_WIN_RATE = 35.0
MIN_EQUITY_CURVE_R_SQUARED = 0.40
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
