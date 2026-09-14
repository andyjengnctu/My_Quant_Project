"""Canonical broker settlement constants shared by Research and Trading.

These values describe the broker-facing cash settlement layer only.  Research
may layer an economic fee-rebate policy on top, but must not redefine the raw
broker fee/tax/rounding inputs.
"""

from decimal import Decimal

BROKER_FEE_RATE = Decimal("0.001425")
BROKER_MIN_FEE_TWD = Decimal("0")
BROKER_STOCK_TAX_RATE = Decimal("0.003")
BROKER_CHARGE_ROUNDING_MODE = "floor_twd"

__all__ = [
    "BROKER_FEE_RATE",
    "BROKER_MIN_FEE_TWD",
    "BROKER_STOCK_TAX_RATE",
    "BROKER_CHARGE_ROUNDING_MODE",
]
