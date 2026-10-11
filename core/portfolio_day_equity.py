"""Portfolio daily mark-to-market/exposure; exact accounting remains canonical.

The equity inputs are all values known by this trading day. Accrued but
unsettled Research broker rebates are a receivable, not extra spendable cash.
"""

from typing import NamedTuple

from core.exact_accounting import milli_to_money
from core.fee_rebate import get_fee_rebate_receivable_milli
from core.portfolio_fast_data import calc_mark_to_market_equity


class PortfolioDayEquityMark(NamedTuple):
    equity_milli: int
    equity_money: float
    cash_money: float


class PortfolioDayExposure(NamedTuple):
    invested_capital: float
    exposure_pct: float
    strategy_ret_pct: float


def mark_portfolio_day_equity(
    *, cash, portfolio, all_dfs_fast, today, params, fee_rebate_state,
) -> PortfolioDayEquityMark:
    """Mark held inventory and outstanding rebate using canonical ledger units."""
    rebate_receivable_milli = get_fee_rebate_receivable_milli(fee_rebate_state)
    if portfolio:
        equity = calc_mark_to_market_equity(
            cash,
            portfolio,
            all_dfs_fast,
            today,
            params,
            fee_rebate_receivable_milli=rebate_receivable_milli,
            use_cached_last_px=True,
        )
    else:
        equity = cash + rebate_receivable_milli
    return PortfolioDayEquityMark(equity, milli_to_money(equity), milli_to_money(cash))


def measure_portfolio_day_exposure(
    *, current_equity_money, cash_money, initial_capital, fee_rebate_state,
) -> PortfolioDayExposure:
    """Return display/exposure statistics without changing the cash ledger."""
    rebate_receivable_money = milli_to_money(get_fee_rebate_receivable_milli(fee_rebate_state))
    invested_capital = max(0.0, current_equity_money - cash_money - rebate_receivable_money)
    exposure_pct = (invested_capital / current_equity_money) * 100 if current_equity_money > 0 else 0
    strategy_ret_pct = (current_equity_money - initial_capital) / initial_capital * 100
    return PortfolioDayExposure(invested_capital, exposure_pct, strategy_ret_pct)
