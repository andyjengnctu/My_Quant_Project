"""全域訓練政策：score 設定與 optimizer 硬門檻。"""

# 系統評分 (Score) 算法切換
# 'LOG_R2' = 結合對數 R 平方與月度勝率的不對稱模型
# 'RoMD'   = 傳統報酬回撤比風格的基底分數
SCORE_CALC_METHOD = 'RoMD'

# 系統評分分子切換
# 'ANNUAL_RETURN' = 分子使用年化報酬率
# 'TOTAL_RETURN'  = 分子使用總報酬率
SCORE_NUMERATOR_METHOD = 'ANNUAL_RETURN'

# Optimizer 投組績效門檻
MIN_FULL_YEAR_RETURN_PCT = -30.0
MIN_ANNUAL_TRADES = 5.0
MIN_BUY_FILL_RATE = 70.0
MIN_TRADE_WIN_RATE = 35.0
MAX_PORTFOLIO_MDD_PCT = 45.0
MIN_MONTHLY_WIN_RATE = 35.0
MIN_EQUITY_CURVE_R_SQUARED = 0.40


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
        "SCORE_CALC_METHOD": SCORE_CALC_METHOD,
        "SCORE_NUMERATOR_METHOD": SCORE_NUMERATOR_METHOD,
    }
