import math
import numpy as np

from core.exact_accounting import round_money_for_display


def get_debug_tp_half_price(tp_half, qty=None, params=None):
    if tp_half is None:
        return np.nan
    try:
        return np.nan if np.isnan(tp_half) else tp_half
    except (TypeError, ValueError):
        return tp_half


def _round_money_for_log(value):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return value
    return round_money_for_display(value)


def append_debug_trade_row(
    trade_logs,
    *,
    date_str,
    action,
    price,
    net_price,
    qty,
    gross_amount,
    stop_price,
    tp_half_price,
    atr_prev,
    pnl,
    note="",
):
    row = {
        "日期": date_str,
        "動作": action,
        "成交價": price,
        "成本均價": net_price,
        "股數": qty,
        "投入總金額": _round_money_for_log(gross_amount),
        "設定停損價": stop_price,
        "半倉停利價": tp_half_price,
        "ATR(前日)": atr_prev,
        "單筆實質損益": _round_money_for_log(pnl),
    }
    if note:
        row["備註"] = note
    trade_logs.append(row)
