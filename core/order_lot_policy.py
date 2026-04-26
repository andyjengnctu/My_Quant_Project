import pandas as pd

from core.exact_accounting import money_to_milli, price_to_milli


TAIWAN_BOARD_LOT_SHARES = 1000


def calc_entry_notional_milli(limit_price, qty):
    if qty is None or pd.isna(limit_price):
        return 0
    try:
        qty_int = int(qty)
    except (TypeError, ValueError, OverflowError):
        return 0
    if qty_int <= 0:
        return 0
    return price_to_milli(limit_price) * qty_int


def get_min_entry_notional_milli(params):
    return money_to_milli(float(getattr(params, "min_entry_notional", 0.0) or 0.0))


def entry_notional_meets_minimum(limit_price, qty, params):
    min_notional_milli = get_min_entry_notional_milli(params)
    if min_notional_milli <= 0:
        return True
    return calc_entry_notional_milli(limit_price, qty) >= min_notional_milli


def apply_board_lot_preferred_qty(limit_price, qty, params):
    try:
        qty_int = int(qty)
    except (TypeError, ValueError, OverflowError):
        return 0
    if qty_int <= 0:
        return 0

    board_lot_qty = (qty_int // TAIWAN_BOARD_LOT_SHARES) * TAIWAN_BOARD_LOT_SHARES
    if board_lot_qty > 0 and entry_notional_meets_minimum(limit_price, board_lot_qty, params):
        return board_lot_qty
    if entry_notional_meets_minimum(limit_price, qty_int, params):
        return qty_int
    return 0
