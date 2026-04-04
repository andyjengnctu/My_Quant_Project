"""全域訓練政策：score、排序/EV 戰略開關、顯示倍率與 optimizer 硬門檻。"""

# 期望值 (EV) 算法切換
# 'A' = 嚴格 R_Multiple 期望值 (Mean R)
# 'B' = 傳統實際盈虧期望值 (Win% * Payoff - Loss%)
EV_CALC_METHOD = 'A'  # (AI註: 期望值算法切換，預設 'A')

# 買入優先序切換開關
# 'PROJ_COST' = 優先買入能消耗最多資金的標的 (資金效率極大化)
# 'EV' = 優先買入期望值最高的標的 (單筆質量極大化)
# 'HIST_WIN_X_TRADES' = 優先買入歷史勝率 × 交易次數最高的標的 (穩定度 × 樣本數)
BUY_SORT_METHOD = 'HIST_WIN_X_TRADES'  # (AI註: 買入優先序切換，預設 'HIST_WIN_X_TRADES')

# 系統評分 (Score) 算法切換
# 'LOG_R2' = 結合對數 R 平方與月度勝率的不對稱模型
# 'RoMD' = 傳統報酬回撤比風格的基底分數
SCORE_CALC_METHOD = 'RoMD'  # (AI註: 系統評分算法，預設 'RoMD')

# 系統評分分子切換
# 'ANNUAL_RETURN' = 分子使用年化報酬率
# 'TOTAL_RETURN' = 分子使用總報酬率
SCORE_NUMERATOR_METHOD = 'ANNUAL_RETURN'  # (AI註: 系統評分分子來源，預設 'ANNUAL_RETURN')

SYSTEM_SCORE_DISPLAY_MULTIPLIER = 1000.0  # (AI註: 系統得分顯示倍率，僅影響 console/report 顯示，預設 1000.0)

MIN_FULL_YEAR_RETURN_PCT = -30.0  # (AI註: 完整年度最差報酬率下限，預設 -30.0)
MIN_ANNUAL_TRADES = 5.0  # (AI註: 最小年化交易次數門檻，預設 5.0)
MIN_BUY_FILL_RATE = 70.0  # (AI註: 最小保留後買進成交率門檻，預設 70.0)
MIN_TRADE_WIN_RATE = 35.0  # (AI註: 最小完整交易勝率門檻，預設 35.0)
MAX_PORTFOLIO_MDD_PCT = 45.0  # (AI註: 投組最大回撤上限，預設 45.0)
MIN_MONTHLY_WIN_RATE = 35.0  # (AI註: 最小月勝率門檻，預設 35.0)
MIN_EQUITY_CURVE_R_SQUARED = 0.40  # (AI註: 權益曲線最小 R 平方門檻，預設 0.40)


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
    }
