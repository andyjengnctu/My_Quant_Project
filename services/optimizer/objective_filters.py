from core.config import (
    MAX_PORTFOLIO_MDD_PCT,
    MIN_ANNUAL_TRADES,
    MIN_BUY_FILL_RATE,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_FULL_YEAR_RETURN_PCT,
    MIN_MONTHLY_WIN_RATE,
    MIN_TRADE_WIN_RATE,
)


def apply_filter_rules(metrics, thresholds=None):
    thresholds = dict(thresholds or {})
    max_mdd = float(thresholds.get("MAX_PORTFOLIO_MDD_PCT", MAX_PORTFOLIO_MDD_PCT))
    min_annual_trades = float(thresholds.get("MIN_ANNUAL_TRADES", MIN_ANNUAL_TRADES))
    min_buy_fill_rate = float(thresholds.get("MIN_BUY_FILL_RATE", MIN_BUY_FILL_RATE))
    min_full_year_return = float(
        thresholds.get("MIN_FULL_YEAR_RETURN_PCT", MIN_FULL_YEAR_RETURN_PCT)
    )
    min_trade_win_rate = float(thresholds.get("MIN_TRADE_WIN_RATE", MIN_TRADE_WIN_RATE))
    min_monthly_win_rate = float(
        thresholds.get("MIN_MONTHLY_WIN_RATE", MIN_MONTHLY_WIN_RATE)
    )
    min_r_squared = float(
        thresholds.get("MIN_EQUITY_CURVE_R_SQUARED", MIN_EQUITY_CURVE_R_SQUARED)
    )
    if metrics["mdd"] > max_mdd:
        return f"回撤過大 ({metrics['mdd']:.1f}%)"
    if metrics["annual_trades"] < min_annual_trades:
        return f"年化交易次數過低 ({metrics['annual_trades']:.2f}次/年)"
    if metrics["reserved_buy_fill_rate"] < min_buy_fill_rate:
        return f"保留後買進成交率過低 ({metrics['reserved_buy_fill_rate']:.2f}%)"
    if metrics["annual_return_pct"] <= 0:
        return f"年化報酬率非正 ({metrics['annual_return_pct']:.2f}%)"
    if metrics["full_year_count"] <= 0:
        return "無完整年度可驗證 min{r_y}"
    if metrics["min_full_year_return_pct"] <= min_full_year_return:
        return (
            f"完整年度最差報酬未大於 {min_full_year_return:.2f}% "
            f"({metrics['min_full_year_return_pct']:.2f}%)"
        )
    if metrics["win_rate"] < min_trade_win_rate:
        return f"實戰勝率偏低 ({metrics['win_rate']:.2f}%)"
    if metrics["m_win_rate"] < min_monthly_win_rate:
        return f"月勝率偏低 ({metrics['m_win_rate']:.0f}%)"
    if metrics["r_sq"] < min_r_squared:
        return f"曲線過度震盪 (R²={metrics['r_sq']:.2f})"
    return None
