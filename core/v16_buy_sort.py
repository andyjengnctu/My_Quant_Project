from core.v16_config import BUY_SORT_METHOD


def calc_buy_sort_value(method, ev, proj_cost, win_rate, trade_count):
    if method == 'EV':
        return float(ev)
    if method == 'PROJ_COST':
        return float(proj_cost)
    if method == 'HIST_WIN_X_TRADES':
        return float(win_rate) * float(trade_count)
    raise ValueError(f"未知的 BUY_SORT_METHOD: {method}")


def calc_active_buy_sort_value(ev, proj_cost, win_rate, trade_count):
    return calc_buy_sort_value(BUY_SORT_METHOD, ev, proj_cost, win_rate, trade_count)


def get_buy_sort_title(method=None):
    active_method = BUY_SORT_METHOD if method is None else method
    if active_method == 'EV':
        return '按期望值 (EV) 由大到小排序'
    if active_method == 'PROJ_COST':
        return '按預估投入資金由大到小排序'
    if active_method == 'HIST_WIN_X_TRADES':
        return '按歷史勝率 × 交易次數由大到小排序'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")