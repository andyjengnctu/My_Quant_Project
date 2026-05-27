import math

from core.config import get_buy_sort_method


BUY_LIMIT_OVERAGE_SORT_METHOD = 'BUY_LIMIT_OVERAGE_THEN_PROJ_COST'


def _as_finite_float(value, *, default):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric_value):
        return default
    return numeric_value


def calc_buy_limit_overage_pct(prev_close, limit_price):
    resolved_prev_close = _as_finite_float(prev_close, default=math.inf)
    resolved_limit_price = _as_finite_float(limit_price, default=math.nan)
    if not math.isfinite(resolved_prev_close) or not math.isfinite(resolved_limit_price) or resolved_limit_price <= 0:
        return math.inf
    return max(0.0, (resolved_prev_close / resolved_limit_price - 1.0) * 100.0)


def is_buy_limit_overage_sort(method=None):
    active_method = get_buy_sort_method() if method is None else method
    return active_method == BUY_LIMIT_OVERAGE_SORT_METHOD


def calc_buy_sort_value(method, ev, proj_cost, win_rate, trade_count, asset_growth_pct=0.0, prev_close=None, limit_price=None):
    if method == 'EV':
        return float(ev)
    if method == 'PROJ_COST':
        return float(proj_cost)
    if method == 'HIST_WIN_X_TRADES':
        return float(win_rate) * float(trade_count)
    if method == 'ASSET_GROWTH':
        return float(asset_growth_pct)
    if method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return calc_buy_limit_overage_pct(prev_close, limit_price)
    raise ValueError(f"未知的 BUY_SORT_METHOD: {method}")


def calc_active_buy_sort_value(ev, proj_cost, win_rate, trade_count, asset_growth_pct=0.0, prev_close=None, limit_price=None):
    return calc_buy_sort_value(
        get_buy_sort_method(),
        ev,
        proj_cost,
        win_rate,
        trade_count,
        asset_growth_pct,
        prev_close=prev_close,
        limit_price=limit_price,
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
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return '按買入限價超出幅度由小到大，再按預估投入資金排序'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def get_buy_sort_metric_label(method=None):
    active_method = get_buy_sort_method() if method is None else method
    if active_method == 'EV':
        return 'EV'
    if active_method == 'PROJ_COST':
        return '預估投入'
    if active_method == 'HIST_WIN_X_TRADES':
        return '勝率×次數'
    if active_method == 'ASSET_GROWTH':
        return '資產成長'
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return '超限幅'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def format_buy_sort_metric_value(value, method=None):
    active_method = get_buy_sort_method() if method is None else method
    numeric_value = float(value)
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD and not math.isfinite(numeric_value):
        return 'N/A'
    if active_method == 'EV':
        return f'{numeric_value:.2f}R'
    if active_method == 'PROJ_COST':
        return f'{numeric_value:,.0f}'
    if active_method == 'HIST_WIN_X_TRADES':
        return f'{numeric_value:.2f}'
    if active_method == 'ASSET_GROWTH':
        return f'{numeric_value:.2f}%'
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return f'{numeric_value:.2f}%'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def sort_candidate_rows(rows, method=None):
    active_method = get_buy_sort_method() if method is None else method
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        rows.sort(
            key=lambda item: (
                _as_finite_float(item.get('sort_value'), default=math.inf),
                -_as_finite_float(item.get('proj_cost'), default=0.0),
                str(item.get('ticker') or ''),
            )
        )
        return rows
    rows.sort(
        key=lambda item: (
            _as_finite_float(item.get('sort_value'), default=-math.inf),
            str(item.get('ticker') or ''),
        ),
        reverse=True,
    )
    return rows



def is_sort_value_better(candidate_value, incumbent_value, method=None):
    active_method = get_buy_sort_method() if method is None else method
    candidate_numeric = _as_finite_float(candidate_value, default=math.inf if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD else -math.inf)
    incumbent_numeric = _as_finite_float(incumbent_value, default=math.inf if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD else -math.inf)
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return candidate_numeric < incumbent_numeric
    return candidate_numeric > incumbent_numeric
