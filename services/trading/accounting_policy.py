"""Actual-broker accounting policy for Trading account records.

Research/backtest may apply a separate economic fee-rebate policy, while this
module always consumes the shared raw broker settlement constants.  Real
Trading account valuation and manually/broker-confirmed fills therefore keep
the broker cash semantics unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

from config.broker_accounting import (
    BROKER_CHARGE_ROUNDING_MODE,
    BROKER_FEE_RATE,
    BROKER_MIN_FEE_TWD,
    BROKER_STOCK_TAX_RATE,
)
from config.execution_policy import EXECUTION_POLICY_PARAM_SPECS

TRADING_ACCOUNT_BROKER_FEE_RATE = float(BROKER_FEE_RATE)
# User-defined broker accounting: fee = consideration * 0.001425, then discard
# all fractional TWD.  Do not apply strategy/backtest discount or an unstated
# minimum fee to actual account records.
TRADING_ACCOUNT_MIN_FEE = float(BROKER_MIN_FEE_TWD)
TRADING_ACCOUNT_STOCK_TAX_RATE = float(BROKER_STOCK_TAX_RATE)
TRADING_ACCOUNT_FIXED_RISK_FALLBACK = float(EXECUTION_POLICY_PARAM_SPECS["fixed_risk"]["default"])


class _TradingAccountingParamsProxy:
    """Read-through params proxy with actual-broker accounting overrides."""

    def __init__(self, base):
        self._base = base
        self.buy_fee = TRADING_ACCOUNT_BROKER_FEE_RATE
        self.sell_fee = TRADING_ACCOUNT_BROKER_FEE_RATE
        self.min_fee = TRADING_ACCOUNT_MIN_FEE
        self.charge_rounding_mode = BROKER_CHARGE_ROUNDING_MODE

    def __getattr__(self, name):
        return getattr(self._base, name)


def overlay_trading_accounting_params(params):
    """Keep strategy geometry but replace only actual-broker accounting fields."""
    return _TradingAccountingParamsProxy(params)


def build_standalone_trading_accounting_params(*, fixed_risk=None):
    """Minimal params object accepted by canonical exact-accounting helpers."""
    return SimpleNamespace(
        buy_fee=TRADING_ACCOUNT_BROKER_FEE_RATE,
        sell_fee=TRADING_ACCOUNT_BROKER_FEE_RATE,
        tax_rate=TRADING_ACCOUNT_STOCK_TAX_RATE,
        min_fee=TRADING_ACCOUNT_MIN_FEE,
        fixed_risk=(
            TRADING_ACCOUNT_FIXED_RISK_FALLBACK
            if fixed_risk is None
            else float(fixed_risk)
        ),
        charge_rounding_mode=BROKER_CHARGE_ROUNDING_MODE,
    )


__all__ = [
    "TRADING_ACCOUNT_BROKER_FEE_RATE",
    "TRADING_ACCOUNT_MIN_FEE",
    "TRADING_ACCOUNT_STOCK_TAX_RATE",
    "overlay_trading_accounting_params",
    "build_standalone_trading_accounting_params",
]
