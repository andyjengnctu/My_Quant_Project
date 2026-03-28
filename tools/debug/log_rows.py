import numpy as np

from core.backtest_core import can_execute_half_take_profit


def get_debug_tp_half_price(tp_half, qty, params):
    return tp_half if can_execute_half_take_profit(qty, params.tp_percent) else np.nan


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
        "含息成本價": net_price,
        "股數": qty,
        "投入總金額": gross_amount,
        "設定停損價": stop_price,
        "半倉停利價": tp_half_price,
        "ATR(前日)": atr_prev,
        "單筆實質損益": pnl,
    }
    if note:
        row["備註"] = note
    trade_logs.append(row)
