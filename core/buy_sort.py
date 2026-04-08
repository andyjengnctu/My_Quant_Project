from core.config import get_buy_sort_method


def calc_buy_sort_value(method, ev, proj_cost, win_rate, trade_count, asset_growth_pct=0.0):
    if method == 'EV':
        return float(ev)
    if method == 'PROJ_COST':
        return float(proj_cost)
    if method == 'HIST_WIN_X_TRADES':
        return float(win_rate) * float(trade_count)
    if method == 'ASSET_GROWTH':
        return float(asset_growth_pct)
    raise ValueError(f"未知的 BUY_SORT_METHOD: {method}")


def calc_active_buy_sort_value(ev, proj_cost, win_rate, trade_count, asset_growth_pct=0.0):
    return calc_buy_sort_value(
        get_buy_sort_method(),
        ev,
        proj_cost,
        win_rate,
        trade_count,
        asset_growth_pct,
    )


def get_buy_sort_title(method=None):
    active_method = get_buy_sort_method() if method is None else method
    if active_method == 'EV':
        return '按期望值 (EV) 由大到小排序'
    if active_method == 'PROJ_COST':
        return '按預估投入資金由大到小排序'
    if active_method == 'HIST_WIN_X_TRADES':
        return '按歷史勝率 × 交易次數由大到小排序'
    if active_method == 'ASSET_GROWTH':
        return '按資產成長由大到小排序'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")
