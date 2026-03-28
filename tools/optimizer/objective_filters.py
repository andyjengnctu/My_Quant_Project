from core.v16_config import (
    MAX_PORTFOLIO_MDD_PCT,
    MIN_ANNUAL_TRADES,
    MIN_BUY_FILL_RATE,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_FULL_YEAR_RETURN_PCT,
    MIN_MONTHLY_WIN_RATE,
    MIN_TRADE_WIN_RATE,
)


def apply_filter_rules(metrics):
    if metrics["mdd"] > MAX_PORTFOLIO_MDD_PCT:
        return f"回撤過大 ({metrics['mdd']:.1f}%)"
    if metrics["annual_trades"] < MIN_ANNUAL_TRADES:
        return f"年化交易次數過低 ({metrics['annual_trades']:.2f}次/年)"
    if metrics["reserved_buy_fill_rate"] < MIN_BUY_FILL_RATE:
        return f"保留後買進成交率過低 ({metrics['reserved_buy_fill_rate']:.2f}%)"
    if metrics["annual_return_pct"] <= 0:
        return f"年化報酬率非正 ({metrics['annual_return_pct']:.2f}%)"
    if metrics["full_year_count"] <= 0:
        return "無完整年度可驗證 min{r_y}"
    if metrics["min_full_year_return_pct"] <= MIN_FULL_YEAR_RETURN_PCT:
        return (
            f"完整年度最差報酬未大於 {MIN_FULL_YEAR_RETURN_PCT:.2f}% "
            f"({metrics['min_full_year_return_pct']:.2f}%)"
        )
    if metrics["win_rate"] < MIN_TRADE_WIN_RATE:
        return f"實戰勝率偏低 ({metrics['win_rate']:.2f}%)"
    if metrics["m_win_rate"] < MIN_MONTHLY_WIN_RATE:
        return f"月勝率偏低 ({metrics['m_win_rate']:.0f}%)"
    if metrics["r_sq"] < MIN_EQUITY_CURVE_R_SQUARED:
        return f"曲線過度震盪 (R²={metrics['r_sq']:.2f})"
    return None
