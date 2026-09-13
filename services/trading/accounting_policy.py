"""Actual-broker accounting policy for Trading account records.

Research/backtest execution policy intentionally remains unchanged.  Only real
Trading account valuation and manually/broker-confirmed fills use the broker
fee rate declared here.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from config.execution_policy import BROKER_FEE_RATE, EXECUTION_POLICY_PARAM_SPECS

TRADING_ACCOUNT_BROKER_FEE_RATE = float(BROKER_FEE_RATE)
# User-defined broker accounting: fee = consideration * 0.001425.  Do not apply
# the strategy/backtest discount or an unstated minimum fee to account records.
TRADING_ACCOUNT_MIN_FEE = 0.0
TRADING_ACCOUNT_STOCK_TAX_RATE = float(EXECUTION_POLICY_PARAM_SPECS["tax_rate"]["default"])
TRADING_ACCOUNT_FIXED_RISK_FALLBACK = float(EXECUTION_POLICY_PARAM_SPECS["fixed_risk"]["default"])


def overlay_trading_accounting_params(params):
    """Keep strategy geometry but replace only actual-broker accounting fields."""
    return replace(
        params,
        buy_fee=TRADING_ACCOUNT_BROKER_FEE_RATE,
        sell_fee=TRADING_ACCOUNT_BROKER_FEE_RATE,
        min_fee=TRADING_ACCOUNT_MIN_FEE,
    )


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
    )


__all__ = [
    "TRADING_ACCOUNT_BROKER_FEE_RATE",
    "TRADING_ACCOUNT_MIN_FEE",
    "TRADING_ACCOUNT_STOCK_TAX_RATE",
    "overlay_trading_accounting_params",
    "build_standalone_trading_accounting_params",
]
