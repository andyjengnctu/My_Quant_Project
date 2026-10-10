"""Canonical broker settlement constants shared by Research and Trading.

These values describe the broker-facing cash settlement layer only.  Research
may layer an economic fee-rebate policy on top, but must not redefine the raw
broker fee/tax/rounding inputs.
"""

from decimal import Decimal

BROKER_FEE_RATE = Decimal("0.001425")
BROKER_MIN_FEE_TWD = Decimal("20")
BROKER_ODD_LOT_MIN_FEE_TWD = Decimal("1")
BROKER_BOARD_LOT_SHARES = 1000
# Electronic commission is rebated after the raw broker charge is deducted.
# This is an economic entitlement, not cash available for immediate Trading use.
BROKER_ELECTRONIC_FEE_EFFECTIVE_RATIO = Decimal("0.60")
BROKER_ACCOUNTING_POLICY_ID = "yuanta_electronic_six_tenths_raw_cash_v1"
BROKER_STOCK_TAX_RATE = Decimal("0.003")
BROKER_CHARGE_ROUNDING_MODE = "floor_twd"

__all__ = [
    "BROKER_FEE_RATE",
    "BROKER_MIN_FEE_TWD",
    "BROKER_ODD_LOT_MIN_FEE_TWD",
    "BROKER_BOARD_LOT_SHARES",
    "BROKER_ELECTRONIC_FEE_EFFECTIVE_RATIO",
    "BROKER_ACCOUNTING_POLICY_ID",
    "BROKER_STOCK_TAX_RATE",
    "BROKER_CHARGE_ROUNDING_MODE",
]
