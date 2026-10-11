"""Reconcile post-closeout equity into the existing portfolio calendar ledgers.

This phase runs *after* canonical closeout execution. Broker rebate receivables
contribute to equity, never to the cash available for pre-market orders.
"""

from typing import NamedTuple

from core.exact_accounting import milli_to_money
from core.fee_rebate import get_fee_rebate_receivable_milli


class PortfolioTerminalEquity(NamedTuple):
    final_equity_money: float
    total_return: float
    max_drawdown: float


def reconcile_portfolio_terminal_equity(
    *,
    closeout_equity_milli,
    fee_rebate_state,
    last_date,
    year_end_equity,
    month_end_equity,
    quarter_end_equity,
    peak_equity,
    max_drawdown,
    sorted_dates,
    start_idx,
    monthly_equities,
    benchmark_loop_enabled,
    current_bm_px,
    bm_monthly_equities,
    initial_capital,
    equity_curve,
    is_training,
    capture_equity_curve,
) -> PortfolioTerminalEquity:
    # # (AI註: 先完成期末結算，再將退款應收款計入權益；不得併入盤前可用現金。)
    final_equity = int(closeout_equity_milli) + get_fee_rebate_receivable_milli(fee_rebate_state)
    if last_date is not None and last_date.year in year_end_equity:
        year_end_equity[last_date.year] = milli_to_money(final_equity)
        last_month_key = (int(last_date.year), int(last_date.month))
        if last_month_key in month_end_equity:
            month_end_equity[last_month_key] = milli_to_money(final_equity)
        last_quarter_key = (int(last_date.year), int((last_date.month - 1) // 3 + 1))
        if last_quarter_key in quarter_end_equity:
            quarter_end_equity[last_quarter_key] = milli_to_money(final_equity)

    # # (AI註: 期末強制結算後補做一次 peak / drawdown 更新，避免 final closeout 對 MDD 漏算)
    if final_equity > peak_equity:
        peak_equity = final_equity
    drawdown = (peak_equity - final_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    if drawdown > max_drawdown:
        max_drawdown = drawdown

    # # (AI註: 月底權益以期末強制結算後的真實 final equity 為準。)
    if len(sorted_dates) > start_idx:
        monthly_equities.append(milli_to_money(final_equity))
        if benchmark_loop_enabled and current_bm_px is not None:
            bm_monthly_equities.append(current_bm_px)

    final_equity_money = milli_to_money(final_equity)
    total_return = (final_equity_money - initial_capital) / initial_capital * 100

    if ((not is_training) or capture_equity_curve) and len(equity_curve) > 0:
        equity_curve[-1]['Equity'] = final_equity_money
        equity_curve[-1]['Strategy_Return_Pct'] = total_return
        equity_curve[-1]['Invested_Amount'] = 0.0
        equity_curve[-1]['Exposure_Pct'] = 0.0

    return PortfolioTerminalEquity(final_equity_money, total_return, max_drawdown)
