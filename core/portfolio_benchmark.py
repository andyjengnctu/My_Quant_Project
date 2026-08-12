import threading
from collections import OrderedDict

from core.portfolio_fast_data import get_fast_close, get_fast_close_on_or_before, has_fast_date
from core.portfolio_stats import calc_annual_return_pct, calc_curve_stats, calc_sim_years


BENCHMARK_PERIOD_STATS_CACHE_MAX_ITEMS = 64
_BENCHMARK_PERIOD_STATS_CACHE = OrderedDict()
_BENCHMARK_PERIOD_STATS_CACHE_LOCK = threading.RLock()


def _make_benchmark_period_cache_key(*, benchmark_data, sorted_dates, start_idx):
    if benchmark_data is None or not sorted_dates or start_idx >= len(sorted_dates):
        return None
    first_date = sorted_dates[start_idx]
    last_date = sorted_dates[-1]
    return (id(benchmark_data), len(sorted_dates), int(start_idx), first_date, last_date)


def _build_benchmark_period_stats(*, benchmark_data, sorted_dates, start_idx):
    benchmark_start_price, benchmark_anchor_date = get_fast_close_on_or_before(benchmark_data, sorted_dates[start_idx])
    if benchmark_start_price is None:
        return {
            'benchmark_start_price': None,
            'bm_ret_pct': 0.0,
            'bm_max_drawdown': 0.0,
            'bm_annual_return_pct': 0.0,
            'bm_r_squared': 0.0,
            'bm_monthly_win_rate': 0.0,
            'bm_full_year_count': 0,
            'bm_min_full_year_return_pct': 0.0,
            'bm_yearly_return_rows': [],
            'bm_full_month_count': 0,
            'bm_min_month_return_pct': 0.0,
            'bm_monthly_return_rows': [],
            'bm_full_quarter_count': 0,
            'bm_min_quarter_return_pct': 0.0,
            'bm_quarterly_return_rows': [],
        }

    current_bm_px = benchmark_start_price if benchmark_anchor_date == sorted_dates[start_idx] else None
    yesterday_bm_px = current_bm_px
    bm_peak_price = benchmark_start_price
    bm_max_drawdown = 0.0
    bm_ret_pct = 0.0
    bm_monthly_equities = [benchmark_start_price]
    current_month = sorted_dates[start_idx].month

    for today in sorted_dates[start_idx:]:
        if has_fast_date(benchmark_data, today):
            current_bm_px = get_fast_close(benchmark_data, date=today)
            bm_ret_pct = (current_bm_px - benchmark_start_price) / benchmark_start_price * 100 if benchmark_start_price > 0 else 0.0
            if current_bm_px > bm_peak_price:
                bm_peak_price = current_bm_px
            current_bm_drawdown = (bm_peak_price - current_bm_px) / bm_peak_price * 100 if bm_peak_price > 0 else 0.0
            if current_bm_drawdown > bm_max_drawdown:
                bm_max_drawdown = current_bm_drawdown

        if today.month != current_month:
            if yesterday_bm_px is not None:
                bm_monthly_equities.append(yesterday_bm_px)
            current_month = today.month
        yesterday_bm_px = current_bm_px

    if current_bm_px is not None:
        bm_monthly_equities.append(current_bm_px)

    sim_years = calc_sim_years(sorted_dates, start_idx)
    bm_end_value = benchmark_start_price * (1.0 + bm_ret_pct / 100.0) if benchmark_start_price > 0 else 0.0
    bm_annual_return_pct = calc_annual_return_pct(benchmark_start_price, bm_end_value, sim_years)
    bm_r_squared, bm_monthly_win_rate = calc_curve_stats(bm_monthly_equities)

    benchmark_yearly_rows = []
    benchmark_monthly_rows = []
    benchmark_quarterly_rows = []
    year_market_bounds = {}
    month_market_bounds = {}
    quarter_market_bounds = {}
    for dt in sorted_dates[start_idx:]:
        year = dt.year
        if year not in year_market_bounds:
            year_market_bounds[year] = {'first': dt, 'last': dt}
        else:
            year_market_bounds[year]['last'] = dt
        month_key = (int(year), int(dt.month))
        if month_key not in month_market_bounds:
            month_market_bounds[month_key] = {'first': dt, 'last': dt}
        else:
            month_market_bounds[month_key]['last'] = dt
        quarter = int((dt.month - 1) // 3 + 1)
        quarter_key = (int(year), int(quarter))
        if quarter_key not in quarter_market_bounds:
            quarter_market_bounds[quarter_key] = {'first': dt, 'last': dt}
        else:
            quarter_market_bounds[quarter_key]['last'] = dt

    for year, bounds in sorted(year_market_bounds.items()):
        if not has_fast_date(benchmark_data, bounds['first']) or not has_fast_date(benchmark_data, bounds['last']):
            continue
        start_value = get_fast_close(benchmark_data, date=bounds['first'])
        end_value = get_fast_close(benchmark_data, date=bounds['last'])
        if start_value is None or end_value is None or start_value <= 0:
            continue
        benchmark_yearly_rows.append({
            'year': int(year),
            'year_return_pct': float((end_value / start_value - 1.0) * 100.0),
            'is_full_year': True,
            'start_date': bounds['first'].strftime('%Y-%m-%d'),
            'end_date': bounds['last'].strftime('%Y-%m-%d'),
        })
    for (year, month), bounds in sorted(month_market_bounds.items()):
        if not has_fast_date(benchmark_data, bounds['first']) or not has_fast_date(benchmark_data, bounds['last']):
            continue
        start_value = get_fast_close(benchmark_data, date=bounds['first'])
        end_value = get_fast_close(benchmark_data, date=bounds['last'])
        if start_value is None or end_value is None or start_value <= 0:
            continue
        benchmark_monthly_rows.append({
            'year': int(year),
            'month': int(month),
            'period': f"{int(year):04d}-{int(month):02d}",
            'month_return_pct': float((end_value / start_value - 1.0) * 100.0),
            'is_full_month': True,
            'start_date': bounds['first'].strftime('%Y-%m-%d'),
            'end_date': bounds['last'].strftime('%Y-%m-%d'),
        })
    for (year, quarter), bounds in sorted(quarter_market_bounds.items()):
        if not has_fast_date(benchmark_data, bounds['first']) or not has_fast_date(benchmark_data, bounds['last']):
            continue
        start_value = get_fast_close(benchmark_data, date=bounds['first'])
        end_value = get_fast_close(benchmark_data, date=bounds['last'])
        if start_value is None or end_value is None or start_value <= 0:
            continue
        benchmark_quarterly_rows.append({
            'year': int(year),
            'quarter': int(quarter),
            'period': f"{int(year)}Q{int(quarter)}",
            'quarter_return_pct': float((end_value / start_value - 1.0) * 100.0),
            'is_full_quarter': True,
            'start_date': bounds['first'].strftime('%Y-%m-%d'),
            'end_date': bounds['last'].strftime('%Y-%m-%d'),
        })

    bm_min_full_year_return_pct = min((row['year_return_pct'] for row in benchmark_yearly_rows), default=0.0)
    bm_min_month_return_pct = min((row['month_return_pct'] for row in benchmark_monthly_rows), default=0.0)
    bm_min_quarter_return_pct = min((row['quarter_return_pct'] for row in benchmark_quarterly_rows), default=0.0)

    return {
        'benchmark_start_price': float(benchmark_start_price),
        'bm_ret_pct': float(bm_ret_pct),
        'bm_max_drawdown': float(bm_max_drawdown),
        'bm_annual_return_pct': float(bm_annual_return_pct),
        'bm_r_squared': float(bm_r_squared),
        'bm_monthly_win_rate': float(bm_monthly_win_rate),
        'bm_full_year_count': int(len(benchmark_yearly_rows)),
        'bm_min_full_year_return_pct': float(bm_min_full_year_return_pct),
        'bm_yearly_return_rows': benchmark_yearly_rows,
        'bm_full_month_count': int(len(benchmark_monthly_rows)),
        'bm_min_month_return_pct': float(bm_min_month_return_pct),
        'bm_monthly_return_rows': benchmark_monthly_rows,
        'bm_full_quarter_count': int(len(benchmark_quarterly_rows)),
        'bm_min_quarter_return_pct': float(bm_min_quarter_return_pct),
        'bm_quarterly_return_rows': benchmark_quarterly_rows,
    }


def _get_benchmark_period_stats(*, benchmark_data, sorted_dates, start_idx):
    cache_key = _make_benchmark_period_cache_key(
        benchmark_data=benchmark_data,
        sorted_dates=sorted_dates,
        start_idx=start_idx,
    )
    if cache_key is None:
        return None
    with _BENCHMARK_PERIOD_STATS_CACHE_LOCK:
        cached = _BENCHMARK_PERIOD_STATS_CACHE.get(cache_key)
        if cached is not None:
            _BENCHMARK_PERIOD_STATS_CACHE.move_to_end(cache_key)
            return cached
    stats = _build_benchmark_period_stats(
        benchmark_data=benchmark_data,
        sorted_dates=sorted_dates,
        start_idx=start_idx,
    )
    with _BENCHMARK_PERIOD_STATS_CACHE_LOCK:
        cached = _BENCHMARK_PERIOD_STATS_CACHE.get(cache_key)
        if cached is not None:
            _BENCHMARK_PERIOD_STATS_CACHE.move_to_end(cache_key)
            return cached
        _BENCHMARK_PERIOD_STATS_CACHE[cache_key] = stats
        _BENCHMARK_PERIOD_STATS_CACHE.move_to_end(cache_key)
        while len(_BENCHMARK_PERIOD_STATS_CACHE) > BENCHMARK_PERIOD_STATS_CACHE_MAX_ITEMS:
            _BENCHMARK_PERIOD_STATS_CACHE.popitem(last=False)
    return stats


