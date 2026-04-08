
# 期望值 (EV) 算法切換
# 'A' = 嚴格 R_Multiple 期望值 (Mean R)
# 'B' = 傳統實際盈虧期望值 (Win% * Payoff - Loss%)
EV_CALC_METHOD = 'A' 

# 買入優先序切換開關
# 'EV' = 優先買入期望值最高的標的 (單筆質量極大化)
# 'PROJ_COST' = 優先買入能消耗最多資金的標的 (資金效率極大化)
# 'HIST_WIN_X_TRADES' = 優先買入歷史勝率 × 交易次數最高的標的 (穩定度 × 樣本數)
BUY_SORT_METHOD = 'HIST_WIN_X_TRADES'  

# 系統評分 (Score) 算法切換
# 'RoMD' = 傳統報酬回撤比風格的基底分數
# 'LOG_R2' = 結合對數 R 平方與月度勝率的不對稱模型
SCORE_CALC_METHOD = 'RoMD'  

# 系統評分分子切換
# 'TOTAL_RETURN' = 分子使用總報酬率
# 'ANNUAL_RETURN' = 分子使用年化報酬率
SCORE_NUMERATOR_METHOD = 'TOTAL_RETURN'  

SYSTEM_SCORE_DISPLAY_MULTIPLIER = 1000.0  # 系統得分顯示倍率，僅影響 console/report 顯示)

# 停利比例固定開關
# None = 由 optimizer 搜尋 tp_percent
# 0.0 = 固定關閉停利
# 其他數值 = 固定停利比例
OPTIMIZER_FIXED_TP_PERCENT = 0.505

# 共用硬門檻 (投組期未績效門檻)
# 交易頻率
MIN_ANNUAL_TRADES = 5.0  # 最小年化交易次數門檻
MIN_BUY_FILL_RATE = 70.0  # 最小保留後買進成交率門檻
MIN_TRADE_WIN_RATE = 30.0  # 最小完整交易勝率門檻
# 績效風險
MIN_FULL_YEAR_RETURN_PCT = -30.0  # 完整年度最差報酬率下限
MAX_PORTFOLIO_MDD_PCT = 45.0  # 投組最大回撤上限
# 穩定度
MIN_MONTHLY_WIN_RATE = 35.0  # 最小月勝率門檻
MIN_EQUITY_CURVE_R_SQUARED = 0.40  # 權益曲線最小 R 平方門檻

# 共用訓練參數 (單股歷史績效門檻)
SELECTION_POLICY_PARAM_SPECS = {
    "min_history_trades": {"type": int, "default": 0, "min_value": 0},  # 歷史績效最少交易次數門檻
    "min_history_ev": {"type": float, "default": -1.0},  # 歷史績效最小期望值門檻
    "min_history_win_rate": {"type": float, "default": 0.30, "min_value": 0.0, "max_value": 1.0},  # 歷史績效最小勝率門檻
}



def build_training_threshold_snapshot():
    return {
        "MIN_FULL_YEAR_RETURN_PCT": MIN_FULL_YEAR_RETURN_PCT,
        "MIN_ANNUAL_TRADES": MIN_ANNUAL_TRADES,
        "MIN_BUY_FILL_RATE": MIN_BUY_FILL_RATE,
        "MIN_TRADE_WIN_RATE": MIN_TRADE_WIN_RATE,
        "MAX_PORTFOLIO_MDD_PCT": MAX_PORTFOLIO_MDD_PCT,
        "MIN_MONTHLY_WIN_RATE": MIN_MONTHLY_WIN_RATE,
        "MIN_EQUITY_CURVE_R_SQUARED": MIN_EQUITY_CURVE_R_SQUARED,
    }


def build_training_score_policy_snapshot():
    return {
        "EV_CALC_METHOD": EV_CALC_METHOD,
        "BUY_SORT_METHOD": BUY_SORT_METHOD,
        "SCORE_CALC_METHOD": SCORE_CALC_METHOD,
        "SCORE_NUMERATOR_METHOD": SCORE_NUMERATOR_METHOD,
        "SYSTEM_SCORE_DISPLAY_MULTIPLIER": SYSTEM_SCORE_DISPLAY_MULTIPLIER,
        "OPTIMIZER_FIXED_TP_PERCENT": OPTIMIZER_FIXED_TP_PERCENT,
    }

def build_selection_policy_snapshot():
    return {field_name: spec["default"] for field_name, spec in SELECTION_POLICY_PARAM_SPECS.items()}
