"""Research fee-rebate cash timing primitives.

The exact-accounting ledger owns transaction economics.  This module owns only
when accrued fee-rebate receivables become spendable cash.  A receivable is an
economic asset immediately, but is transferred into cash only at the next
month boundary.
"""

from __future__ import annotations


SETTLEMENT_BASIS_BROKER_CASH = "broker_cash"
SETTLEMENT_BASIS_LEDGER_NET = "ledger_net"
_VALID_SETTLEMENT_BASES = {SETTLEMENT_BASIS_BROKER_CASH, SETTLEMENT_BASIS_LEDGER_NET}


def validate_fee_rebate_settlement_basis(settlement_basis, fee_rebate_state):
    if settlement_basis not in _VALID_SETTLEMENT_BASES:
        raise ValueError(
            "settlement_basis must be explicitly set to broker_cash or ledger_net"
        )
    if settlement_basis == SETTLEMENT_BASIS_LEDGER_NET and fee_rebate_state is not None:
        raise ValueError("ledger_net settlement must not mutate fee_rebate_state")
    return settlement_basis


def create_fee_rebate_state():
    return {
        "receivable_milli": 0,
        "settled_milli": 0,
    }


def accrue_fee_rebate(state, amount_milli: int) -> int:
    amount_milli = max(0, int(amount_milli or 0))
    if amount_milli <= 0:
        return 0
    if state is None:
        raise ValueError("fee_rebate_state is required when a positive Research fee rebate accrues")
    state["receivable_milli"] = int(state.get("receivable_milli", 0) or 0) + amount_milli
    return amount_milli


def get_fee_rebate_receivable_milli(state) -> int:
    if state is None:
        return 0
    return max(0, int(state.get("receivable_milli", 0) or 0))


def settle_fee_rebate(state) -> int:
    if state is None:
        raise ValueError("fee_rebate_state is required for monthly Research fee-rebate settlement")
    amount_milli = get_fee_rebate_receivable_milli(state)
    state["receivable_milli"] = 0
    state["settled_milli"] = int(state.get("settled_milli", 0) or 0) + amount_milli
    return amount_milli


def get_fee_rebate_settled_milli(state) -> int:
    if state is None:
        return 0
    return max(0, int(state.get("settled_milli", 0) or 0))


__all__ = [
    "SETTLEMENT_BASIS_BROKER_CASH",
    "SETTLEMENT_BASIS_LEDGER_NET",
    "validate_fee_rebate_settlement_basis",
    "create_fee_rebate_state",
    "accrue_fee_rebate",
    "get_fee_rebate_receivable_milli",
    "settle_fee_rebate",
    "get_fee_rebate_settled_milli",
]
