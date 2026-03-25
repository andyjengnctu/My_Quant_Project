import pandas as pd
import numpy as np
import bisect
import time
from core.v16_core import generate_signals, adjust_long_stop_price, adjust_long_sell_fill_price, adjust_long_buy_fill_price, adjust_long_target_price, calc_net_sell_price, calc_position_size, calc_entry_price, calc_initial_risk_total, execute_bar_step, evaluate_chase_condition, run_v16_backtest, build_normal_entry_plan
from core.v16_config import EV_CALC_METHOD, BUY_SORT_METHOD
from core.v16_buy_sort import calc_buy_sort_value


def calc_portfolio_score(sys_ret, sys_mdd, m_win_rate, r_sq, annual_return_pct=None):
    from core.v16_config import SCORE_CALC_METHOD
    base_return = sys_ret if annual_return_pct is None else annual_return_pct
    annualized_romd = base_return / (abs(sys_mdd) + 0.0001)
    if SCORE_CALC_METHOD == 'LOG_R2':
        return annualized_romd * (m_win_rate / 100.0) * r_sq
    return annualized_romd

def calc_curve_stats(eq_list):
    r_squared, monthly_win_rate = 0.0, 0.0
    if len(eq_list) > 2:
        eq_array = np.array(eq_list)
        x = np.arange(len(eq_array))
        if np.std(eq_array) > 0:
            valid_idx = eq_array > 0
            if np.any(valid_idx):
                log_eq = np.log(eq_array[valid_idx])
                x_valid = x[valid_idx]
                if np.std(log_eq) > 0 and len(x_valid) > 1:
                    r_matrix = np.corrcoef(x_valid, log_eq)
                    r_squared = r_matrix[0, 1] ** 2 if not np.isnan(r_matrix[0, 1]) else 0.0
        eq_series = pd.Series(eq_list)
        monthly_rets = eq_series.pct_change().dropna()
        if len(monthly_rets) > 0:
            monthly_win_rate = (len(monthly_rets[monthly_rets > 0]) / len(monthly_rets)) * 100
    return r_squared, monthly_win_rate

# # (AI註: 只用完整年度做 min{r_y} > 0 的檢查；不把起始殘年與當前未完整年度算進去)
def build_full_year_return_stats(sorted_dates, year_start_equity, year_end_equity, year_first_sim_date, year_last_sim_date):
    yearly_return_rows = []
    year_market_bounds = {}

    for dt in sorted_dates:
        year = dt.year
        if year not in year_market_bounds:
            year_market_bounds[year] = {"first": dt, "last": dt}
        else:
            year_market_bounds[year]["last"] = dt

    for year in sorted(year_start_equity.keys()):
        start_equity = float(year_start_equity.get(year, 0.0))
        end_equity = float(year_end_equity.get(year, 0.0))
        first_sim_date = year_first_sim_date.get(year)
        last_sim_date = year_last_sim_date.get(year)
        market_bounds = year_market_bounds.get(year)

        if start_equity <= 0 or end_equity <= 0 or market_bounds is None or first_sim_date is None or last_sim_date is None:
            continue

        year_return_pct = (end_equity / start_equity - 1.0) * 100.0
        is_full_year = (first_sim_date == market_bounds["first"]) and (last_sim_date == market_bounds["last"])

        yearly_return_rows.append({
            "year": int(year),
            "year_return_pct": float(year_return_pct),
            "is_full_year": bool(is_full_year),
            "start_date": first_sim_date.strftime("%Y-%m-%d"),
            "end_date": last_sim_date.strftime("%Y-%m-%d"),
        })

    full_year_rows = [row for row in yearly_return_rows if row["is_full_year"]]
    min_full_year_return_pct = min((row["year_return_pct"] for row in full_year_rows), default=0.0)

    return {
        "full_year_count": len(full_year_rows),
        "min_full_year_return_pct": float(min_full_year_return_pct),
        "yearly_return_rows": yearly_return_rows,
    }

def build_benchmark_full_year_return_stats(sorted_dates, benchmark_data, yearly_return_rows):
    if benchmark_data is None or not yearly_return_rows:
        return {
            "bm_full_year_count": 0,
            "bm_min_full_year_return_pct": 0.0,
            "bm_yearly_return_rows": []
        }

    year_market_bounds = {}
    for dt in sorted_dates:
        year = dt.year
        if year not in year_market_bounds:
            year_market_bounds[year] = {"first": dt, "last": dt}
        else:
            year_market_bounds[year]["last"] = dt

    bm_yearly_rows = []
    full_years = [row["year"] for row in yearly_return_rows if row["is_full_year"]]

    for year in full_years:
        bounds = year_market_bounds.get(year)
        if bounds is None:
            continue
        if not has_fast_date(benchmark_data, bounds["first"]) or not has_fast_date(benchmark_data, bounds["last"]):
            continue

        start_value = get_fast_close(benchmark_data, date=bounds["first"])
        end_value = get_fast_close(benchmark_data, date=bounds["last"])
        if start_value is None or end_value is None or start_value <= 0:
            continue

        bm_yearly_rows.append({
            "year": int(year),
            "year_return_pct": float((end_value / start_value - 1.0) * 100.0),
            "is_full_year": True,
            "start_date": bounds["first"].strftime("%Y-%m-%d"),
            "end_date": bounds["last"].strftime("%Y-%m-%d"),
        })

    bm_min_full_year_return_pct = min((row["year_return_pct"] for row in bm_yearly_rows), default=0.0)

    return {
        "bm_full_year_count": len(bm_yearly_rows),
        "bm_min_full_year_return_pct": float(bm_min_full_year_return_pct),
        "bm_yearly_return_rows": bm_yearly_rows
    }

# # (AI註: 年化報酬率與年化交易次數共用同一個回測期間口徑，避免統計不一致)
def calc_sim_years(sorted_dates, start_idx):
    if not sorted_dates or start_idx >= len(sorted_dates):
        return 0.0
    first_dt = pd.Timestamp(sorted_dates[start_idx])
    last_dt = pd.Timestamp(sorted_dates[-1])
    span_days = (last_dt - first_dt).days + 1
    if span_days <= 0:
        return 0.0
    return span_days / 365.25


# # (AI註: 用 CAGR 口徑統一系統與大盤年化報酬率；若期末值非正，直接回傳 -100% 避免數學異常)
def calc_annual_return_pct(start_value, end_value, years):
    if start_value <= 0 or years <= 0:
        return 0.0
    if end_value <= 0:
        return -100.0
    return ((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0


# # (AI註: 單一真理來源 - 浮動權益估值與 pending_chase 的 next-day sizing 共用同一口徑)
def calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params):
    equity = cash

    for ticker in sorted(portfolio.keys()):
        pos = portfolio[ticker]
        pt_data = all_dfs_fast[ticker]
        pt_pos = get_fast_pos(pt_data, today)

        if pt_pos >= 0:
            px = get_fast_close(pt_data, pos=pt_pos)
        else:
            px = pos.get('last_px', pos.get('pure_buy_price', pos['entry']))

        pos['last_px'] = px

        floating_exec_price = adjust_long_sell_fill_price(px)
        net_px = calc_net_sell_price(floating_exec_price, pos['qty'], params)
        equity += pos['qty'] * net_px

    return float(equity)

FAST_FLOAT_FIELDS = ('Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'buy_limit')
FAST_BOOL_FIELDS = ('is_setup', 'ind_sell_signal')


def pack_prepared_stock_data(df):
    packed = {
        '_packed_market_data': True,
        'dates': tuple(df.index.tolist()),
        'date_to_pos': {dt: i for i, dt in enumerate(df.index)},
    }
    for field in FAST_FLOAT_FIELDS:
        packed[field] = df[field].to_numpy(dtype=np.float64, copy=True)
    for field in FAST_BOOL_FIELDS:
        packed[field] = df[field].to_numpy(dtype=bool, copy=True)
    return packed


def is_packed_market_data(data):
    return isinstance(data, dict) and data.get('_packed_market_data', False) is True


def get_fast_dates(data):
    if is_packed_market_data(data):
        return data['dates']
    return tuple(sorted(data.keys()))


def get_fast_pos(data, date):
    if is_packed_market_data(data):
        return data['date_to_pos'].get(date, -1)
    if date not in data:
        return -1
    d_list = get_fast_dates(data)
    return bisect.bisect_left(d_list, date)


def has_fast_date(data, date):
    if is_packed_market_data(data):
        return date in data['date_to_pos']
    return date in data


def get_fast_value(data, field, *, pos=None, date=None):
    if is_packed_market_data(data):
        if pos is None:
            pos = data['date_to_pos'].get(date, -1)
        if pos < 0:
            raise KeyError(date)
        value = data[field][pos]
        if field in FAST_BOOL_FIELDS:
            return bool(value)
        return float(value)
    if date is None:
        raise KeyError('dict market data 需要 date')
    return data[date][field]


def get_fast_close(data, *, pos=None, date=None):
    return get_fast_value(data, 'Close', pos=pos, date=date)


# # (AI註: benchmark 起算必須與模擬起點對齊；若起點當日無資料，取起點當日或之前最近一筆收盤價，避免之後才開始算 benchmark)
def get_fast_close_on_or_before(data, date):
    dates = get_fast_dates(data)
    pos = bisect.bisect_right(dates, date) - 1
    if pos < 0:
        return None, None

    anchor_date = dates[pos]
    return get_fast_close(data, date=anchor_date), anchor_date


def build_normal_setup_index(all_dfs_fast):
    setup_index = {}
    for ticker, fast_df in all_dfs_fast.items():
        dates = get_fast_dates(fast_df)
        if len(dates) < 2:
            continue

        if is_packed_market_data(fast_df):
            setup_positions = np.flatnonzero(fast_df['is_setup'][:-1])
            for y_pos in setup_positions:
                today = dates[y_pos + 1]
                setup_index.setdefault(today, []).append((ticker, y_pos, y_pos + 1))
        else:
            for i in range(1, len(dates)):
                y_date = dates[i - 1]
                if fast_df[y_date]['is_setup']:
                    setup_index.setdefault(dates[i], []).append((ticker, i - 1, i))
    return setup_index

def prep_stock_data_and_trades(df, params, profile_stats=None):
    t_total_start = time.perf_counter() if profile_stats is not None else None

    t0 = time.perf_counter() if profile_stats is not None else None
    df = df.copy()
    if profile_stats is not None:
        profile_stats['copy_sec'] = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_stats is not None else None
    precomputed_signals = generate_signals(df, params)
    ATR_main, buyCondition, sellCondition, buy_limits = precomputed_signals
    if profile_stats is not None:
        profile_stats['generate_signals_sec'] = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_stats is not None else None
    df['ATR'] = ATR_main
    df['is_setup'] = buyCondition
    df['ind_sell_signal'] = sellCondition
    df['buy_limit'] = buy_limits
    if profile_stats is not None:
        profile_stats['assign_columns_sec'] = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_stats is not None else None
    _, standalone_logs = run_v16_backtest(df, params, return_logs=True, precomputed_signals=precomputed_signals)
    if profile_stats is not None:
        profile_stats['run_backtest_sec'] = time.perf_counter() - t0
        profile_stats['total_sec'] = time.perf_counter() - t_total_start

    return df, standalone_logs

def build_trade_stats_index(trade_logs):
    if not trade_logs:
        return {
            'exit_dates': [],
            'cum_trade_count': np.array([], dtype=np.int32),
            'cum_win_count': np.array([], dtype=np.int32),
            'cum_win_r_sum': np.array([], dtype=np.float64),
            'cum_loss_r_sum': np.array([], dtype=np.float64),
            'cum_total_r_sum': np.array([], dtype=np.float64),
        }

    ordered_logs = sorted(trade_logs, key=lambda t: t['exit_date'])

    exit_dates = []
    cum_trade_count = []
    cum_win_count = []
    cum_win_r_sum = []
    cum_loss_r_sum = []
    cum_total_r_sum = []

    trade_count = 0
    win_count = 0
    win_r_sum = 0.0
    loss_r_sum = 0.0
    total_r_sum = 0.0

    for trade in ordered_logs:
        trade_count += 1

        pnl = float(trade.get('pnl', 0.0))
        r_mult = float(trade.get('r_mult', 0.0))
        total_r_sum += r_mult

        if pnl > 0:
            win_count += 1
            win_r_sum += r_mult
        else:
            loss_r_sum += r_mult

        exit_dates.append(trade['exit_date'])
        cum_trade_count.append(trade_count)
        cum_win_count.append(win_count)
        cum_win_r_sum.append(win_r_sum)
        cum_loss_r_sum.append(loss_r_sum)
        cum_total_r_sum.append(total_r_sum)

    return {
        'exit_dates': exit_dates,
        'cum_trade_count': np.array(cum_trade_count, dtype=np.int32),
        'cum_win_count': np.array(cum_win_count, dtype=np.int32),
        'cum_win_r_sum': np.array(cum_win_r_sum, dtype=np.float64),
        'cum_loss_r_sum': np.array(cum_loss_r_sum, dtype=np.float64),
        'cum_total_r_sum': np.array(cum_total_r_sum, dtype=np.float64),
    }

def get_pit_stats_from_index(stats_index, current_date, params):
    cutoff = bisect.bisect_left(stats_index['exit_dates'], current_date)
    trade_count = int(stats_index['cum_trade_count'][cutoff - 1]) if cutoff > 0 else 0
    min_trades_req = getattr(params, 'min_history_trades', 0)
    min_ev_req = getattr(params, 'min_history_ev', 0.0)
    min_win_rate_req = getattr(params, 'min_history_win_rate', 0.30)

    # # (AI註: 與單股回測完全一致；零樣本只有在所有歷史門檻都被明確放寬時才可通過)
    allow_zero_history = (
        (min_trades_req == 0) and
        (min_ev_req <= 0) and
        (min_win_rate_req <= 0)
    )

    if trade_count < min_trades_req:
        return False, 0.0, 0.0, trade_count
    if trade_count == 0:
        return allow_zero_history, 0.0, 0.0, trade_count

    win_count = int(stats_index['cum_win_count'][cutoff - 1])
    win_rate = win_count / trade_count

    if EV_CALC_METHOD == 'B':
        avg_win_r = (
            stats_index['cum_win_r_sum'][cutoff - 1] / win_count
            if win_count > 0 else 0.0
        )

        loss_count = trade_count - win_count
        loss_r_sum = stats_index['cum_loss_r_sum'][cutoff - 1] if loss_count > 0 else 0.0
        avg_loss_r = abs(loss_r_sum / loss_count) if loss_count > 0 else 0.0

        if avg_loss_r > 0:
            payoff_for_ev = min(10.0, avg_win_r / avg_loss_r)
        elif avg_win_r > 0:
            payoff_for_ev = 99.9
        else:
            payoff_for_ev = 0.0

        ev_to_sort = (win_rate * payoff_for_ev) - (1 - win_rate)
    else:
        ev_to_sort = stats_index['cum_total_r_sum'][cutoff - 1] / trade_count

    is_candidate = (
        (ev_to_sort > getattr(params, 'min_history_ev', 0.0))
        and
        (win_rate >= getattr(params, 'min_history_win_rate', 0.30))
    )
    return is_candidate, ev_to_sort, win_rate, trade_count

def is_extended_entry_type(entry_type):
    return entry_type in ('extended', 'chase')

def run_portfolio_timeline(all_dfs_fast, all_standalone_logs, sorted_dates, start_year, params, max_positions, enable_rotation, benchmark_ticker="0050", benchmark_data=None, is_training=True, profile_stats=None, verbose=True):
    t_portfolio_start = time.perf_counter() if profile_stats is not None else None
    candidate_scan_sec = 0.0
    day_loop_sec = 0.0
    rotation_sec = 0.0
    settle_sec = 0.0
    buy_sec = 0.0
    equity_mark_sec = 0.0
    build_trade_index_sec = 0.0

    start_dt = pd.to_datetime(f"{start_year}-01-01")
    start_idx = next((i for i, d in enumerate(sorted_dates) if d >= start_dt), len(sorted_dates))
    start_idx = max(1, start_idx)

    t0 = time.perf_counter() if profile_stats is not None else None
    pit_stats_index = {t: build_trade_stats_index(logs) for t, logs in all_standalone_logs.items()}
    normal_setup_index = build_normal_setup_index(all_dfs_fast)
    if profile_stats is not None:
        build_trade_index_sec = time.perf_counter() - t0

    initial_capital = params.initial_capital
    cash = initial_capital
    portfolio = {}
    pending_chases = {}
    trade_history, equity_curve, closed_trades_stats = [], [], []
    normal_trade_count, extended_trade_count = 0, 0
    peak_equity, max_drawdown, current_equity = initial_capital, 0.0, initial_capital
    total_exposure, sim_days, total_missed_buys, total_missed_sells, max_exp = 0.0, 0, 0, 0, 0.0
    monthly_equities, bm_monthly_equities = [], []
    yesterday_equity = initial_capital
    year_start_equity, year_end_equity = {}, {}
    year_first_sim_date, year_last_sim_date = {}, {}

    current_month = sorted_dates[start_idx].month if start_idx < len(sorted_dates) else 1

    current_bm_px = None
    benchmark_start_price = None
    bm_peak_price = None
    if benchmark_data and start_idx < len(sorted_dates):
        benchmark_start_price, benchmark_anchor_date = get_fast_close_on_or_before(benchmark_data, sorted_dates[start_idx])
        if benchmark_start_price is not None:
            bm_peak_price = benchmark_start_price
            if benchmark_anchor_date == sorted_dates[start_idx]:
                current_bm_px = benchmark_start_price

    yesterday_bm_px, bm_max_drawdown, bm_ret_pct = current_bm_px, 0.0, 0.0

    for i in range(start_idx, len(sorted_dates)):
        t_day_start = time.perf_counter() if profile_stats is not None else None
        sim_days += 1
        today = sorted_dates[i]

        if today.year not in year_start_equity:
            year_start_equity[today.year] = current_equity
            year_first_sim_date[today.year] = pd.Timestamp(today)

        available_cash = cash
        sizing_equity = current_equity if getattr(params, 'use_compounding', True) else initial_capital

        sold_today = set()

        if verbose and (not is_training) and i % 20 == 0:
            exp = ((current_equity - cash) / current_equity) * 100 if current_equity > 0 else 0
            print(f"\033[90m⏳ 推進中: {today.strftime('%Y-%m')} | 資產: {current_equity:,.0f} | 水位: {exp:>5.1f}%...\033[0m", end="\r", flush=True)

        t0 = time.perf_counter() if profile_stats is not None else None
        candidates_today = []

        for ticker, y_pos, t_pos in sorted(normal_setup_index.get(today, []), key=lambda x: x[0]):
            if ticker in portfolio or ticker in sold_today:
                continue

            fast_df = all_dfs_fast[ticker]
            if ticker in pending_chases:
                del pending_chases[ticker]

            is_candidate, ev, win_rate, trade_count = get_pit_stats_from_index(
                pit_stats_index[ticker], today, params
            )
            if not is_candidate:
                continue

            y_buy_limit = get_fast_value(fast_df, 'buy_limit', pos=y_pos)
            y_atr = get_fast_value(fast_df, 'ATR', pos=y_pos)
            entry_plan = build_normal_entry_plan(y_buy_limit, y_atr, sizing_equity, params)
            if entry_plan is None:
                continue

            est_init_sl = entry_plan['init_sl']
            est_init_trail = entry_plan['init_trail']
            est_qty = entry_plan['qty']

            est_cost = calc_entry_price(y_buy_limit, est_qty, params) * est_qty
            sort_value = calc_buy_sort_value(BUY_SORT_METHOD, ev, est_cost, win_rate, trade_count)
            candidates_today.append({
                'ticker': ticker,
                'type': 'normal',
                'limit_px': y_buy_limit,
                'ev': ev,
                'y_atr': y_atr,
                'today_pos': t_pos,
                'yesterday_pos': y_pos,
                'qty': est_qty,
                'proj_cost': est_cost,
                'sort_value': sort_value,
                'hist_win_rate': win_rate,
                'hist_trade_count': trade_count,
                'init_sl': est_init_sl,
                'init_trail': est_init_trail,
            })

        for ticker in sorted(list(pending_chases.keys())):
            if ticker in portfolio or ticker in sold_today:
                continue

            fast_df = all_dfs_fast.get(ticker)
            if fast_df is None:
                continue

            t_pos = get_fast_pos(fast_df, today)
            if t_pos <= 0:
                continue
            y_pos = t_pos - 1

            chase = pending_chases[ticker]
            is_candidate, ev, win_rate, trade_count = get_pit_stats_from_index(pit_stats_index[ticker], today, params)
            if is_candidate:
                est_qty = chase['qty']
                if est_qty > 0:
                    est_cost = calc_entry_price(chase['chase_price'], est_qty, params) * est_qty
                    sort_value = calc_buy_sort_value(BUY_SORT_METHOD, ev, est_cost, win_rate, trade_count)
                    candidates_today.append({
                        'ticker': ticker,
                        'type': 'chase',
                        'limit_px': chase['chase_price'],
                        'ev': ev,
                        'y_atr': chase['orig_atr'],
                        'today_pos': t_pos,
                        'yesterday_pos': y_pos,
                        'qty': est_qty,
                        'proj_cost': est_cost,
                        'sort_value': sort_value,
                        'hist_win_rate': win_rate,
                        'hist_trade_count': trade_count,
                        'chase_data': chase,
                        'init_sl': chase['sl'],
                        'init_trail': chase['sl'],
                        'orig_limit': chase['orig_limit'],
                    })
            else:
                del pending_chases[ticker]

        candidates_today.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
        if profile_stats is not None:
            candidate_scan_sec += time.perf_counter() - t0

        t0 = time.perf_counter() if profile_stats is not None else None
        if len(portfolio) == max_positions and enable_rotation and candidates_today:
            for cand_idx in range(len(candidates_today)):
                cand = candidates_today[cand_idx]
                weakest_ticker = None
                lowest_ret = float('inf')

                for pt in sorted(portfolio.keys()):
                    pos = portfolio[pt]
                    pt_data = all_dfs_fast[pt]
                    pt_pos = get_fast_pos(pt_data, today)
                    if pt_pos <= 0:
                        continue
                    pt_y_pos = pt_pos - 1
                    pt_y_close = get_fast_close(pt_data, pos=pt_y_pos)
                    ret = (pt_y_close - pos['entry']) / pos['entry']

                    holding_cost = pos['entry'] * pos['qty']
                    _, holding_ev, holding_win_rate, holding_trade_count = get_pit_stats_from_index(pit_stats_index[pt], today, params)
                    holding_sort_value = calc_buy_sort_value(BUY_SORT_METHOD, holding_ev, holding_cost, holding_win_rate, holding_trade_count)
                    is_strategically_better = cand['sort_value'] > holding_sort_value

                    if is_strategically_better and ret < lowest_ret:
                        lowest_ret = ret
                        weakest_ticker = pt

                if weakest_ticker:
                    w_data = all_dfs_fast[weakest_ticker]
                    w_pos = get_fast_pos(w_data, today)
                    if w_pos <= 0:
                        continue
                    w_y_pos = w_pos - 1
                    w_open = get_fast_value(w_data, 'Open', pos=w_pos)
                    w_high = get_fast_value(w_data, 'High', pos=w_pos)
                    w_low = get_fast_value(w_data, 'Low', pos=w_pos)
                    w_close = get_fast_close(w_data, pos=w_pos)
                    w_y_close = get_fast_close(w_data, pos=w_y_pos)
                    is_locked_down = (
                        (w_open == w_high) and
                        (w_high == w_low) and
                        (w_low == w_close) and
                        (w_close < w_y_close)
                    )

                    if not is_locked_down:
                        pos = portfolio[weakest_ticker]
                        est_sell_px = adjust_long_sell_fill_price(w_open)
                        est_freed_cash = calc_net_sell_price(est_sell_px, pos['qty'], params) * pos['qty']

                        # # (AI註: 問題3 版本B - rotation 改成 T 日只賣弱股、T+1 才重新評估可買標的，不用今天候選的 proj_cost 綁定是否先賣弱股)
                        pnl = est_freed_cash - (pos['entry'] * pos['qty'])
                        cash += est_freed_cash

                        total_pnl = pos['realized_pnl'] + pnl
                        total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
                        closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')})
                        if is_extended_entry_type(pos.get('entry_type', 'normal')):
                            extended_trade_count += 1
                        else:
                            normal_trade_count += 1

                        if not is_training:
                            trade_history.append({
                                "Date": today.strftime('%Y-%m-%d'),
                                "Ticker": weakest_ticker,
                                "Type": "汰弱賣出(Open, T+1再評估買進)",
                                "單筆損益": pnl,
                                "該筆總損益": total_pnl,
                                "R_Multiple": total_r,
                                "Risk": params.fixed_risk
                            })

                        del portfolio[weakest_ticker]
                        sold_today.add(weakest_ticker)
                        break
        if profile_stats is not None:
            rotation_sec += time.perf_counter() - t0

        t0 = time.perf_counter() if profile_stats is not None else None
        tickers_to_remove = []
        for ticker in sorted(portfolio.keys()):
            pos = portfolio[ticker]
            fast_df = all_dfs_fast[ticker]
            t_pos = get_fast_pos(fast_df, today)
            if t_pos <= 0:
                continue
            y_pos = t_pos - 1

            pos, freed_cash, pnl_realized, events = execute_bar_step(
                pos,
                get_fast_value(fast_df, 'ATR', pos=y_pos),
                get_fast_value(fast_df, 'ind_sell_signal', pos=y_pos),
                get_fast_close(fast_df, pos=y_pos),
                get_fast_value(fast_df, 'Open', pos=t_pos),
                get_fast_value(fast_df, 'High', pos=t_pos),
                get_fast_value(fast_df, 'Low', pos=t_pos),
                get_fast_close(fast_df, pos=t_pos),
                get_fast_value(fast_df, 'Volume', pos=t_pos),
                params,
            )
            cash += freed_cash

            if 'TP_HALF' in events and not is_training:
                trade_history.append({
                    "Date": today.strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Type": "半倉停利",
                    "單筆損益": pnl_realized,
                    "R_Multiple": 0.0,
                    "Risk": params.fixed_risk
                })

            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl = pos['realized_pnl']
                total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
                closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')})
                if is_extended_entry_type(pos.get('entry_type', 'normal')):
                    extended_trade_count += 1
                else:
                    normal_trade_count += 1

                if not is_training:
                    t_type = "全倉結算(停損)" if 'STOP' in events else "全倉結算(指標)"
                    trade_history.append({
                        "Date": today.strftime('%Y-%m-%d'),
                        "Ticker": ticker,
                        "Type": t_type,
                        "單筆損益": pnl_realized,
                        "該筆總損益": total_pnl,
                        "R_Multiple": total_r,
                        "Risk": params.fixed_risk
                    })

                tickers_to_remove.append(ticker)
            elif 'LOCKED_DOWN' in events:
                total_missed_sells += 1

        for t in tickers_to_remove:
            del portfolio[t]
            sold_today.add(t)
        if profile_stats is not None:
            settle_sec += time.perf_counter() - t0

        t0 = time.perf_counter() if profile_stats is not None else None
        # # (AI註: 問題3 版本B - sold_today 也包含 rotation 當日賣出，名額必須凍結到下一交易日)
        pre_market_occupied = len(portfolio) + len(sold_today)

        for cand in candidates_today:
            fast_df = all_dfs_fast[cand['ticker']]
            t_pos = cand['today_pos']
            y_pos = cand['yesterday_pos']
            t_open = get_fast_value(fast_df, 'Open', pos=t_pos)
            t_high = get_fast_value(fast_df, 'High', pos=t_pos)
            t_low = get_fast_value(fast_df, 'Low', pos=t_pos)
            t_close = get_fast_close(fast_df, pos=t_pos)
            t_volume = get_fast_value(fast_df, 'Volume', pos=t_pos)
            y_close = get_fast_close(fast_df, pos=y_pos)
            is_locked_up = (
                (t_open == t_high) and
                (t_high == t_low) and
                (t_low == t_close) and
                (t_close > y_close)
            )

            is_normal_worse_than_sl = False

            if pre_market_occupied >= max_positions or cand['proj_cost'] > available_cash:
                if cand['type'] == 'normal':
                    if t_volume > 0:
                        next_day_sizing_equity = (
                            calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params)
                            if getattr(params, 'use_compounding', True) else initial_capital
                        )
                        chase_res = evaluate_chase_condition(
                            t_close, cand['limit_px'], cand['y_atr'], next_day_sizing_equity, params
                        )
                        if chase_res:
                            chase_res['orig_limit'] = cand['limit_px']
                            chase_res['orig_atr'] = cand['y_atr']
                            pending_chases[cand['ticker']] = chase_res

                elif cand['type'] == 'chase':
                    # # (AI註: chase 候選即使因滿倉/資金不足未買，也必須依當日收盤更新、失效或保留 pending_chase)
                    if t_volume <= 0:
                        pass
                    elif cand['chase_data']['sl'] < t_close < cand['chase_data']['tp']:
                        next_day_sizing_equity = (
                            calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params)
                            if getattr(params, 'use_compounding', True) else initial_capital
                        )
                        chase_res = evaluate_chase_condition(
                            t_close, cand['orig_limit'], cand['y_atr'], next_day_sizing_equity, params
                        )
                        if chase_res:
                            chase_res['orig_limit'] = cand['orig_limit']
                            chase_res['orig_atr'] = cand['y_atr']
                            pending_chases[cand['ticker']] = chase_res
                        elif cand['ticker'] in pending_chases:
                            del pending_chases[cand['ticker']]
                    elif cand['ticker'] in pending_chases:
                        del pending_chases[cand['ticker']]

                continue

            available_cash -= cand['proj_cost']
            pre_market_occupied += 1

            buyTriggered = False

            if t_volume > 0 and t_low <= cand['limit_px'] and not is_locked_up:
                buy_price = adjust_long_buy_fill_price(min(t_open, cand['limit_px']))
                if buy_price > cand['init_sl']:
                    qty = cand['qty']
                    actual_cost_per_share = calc_entry_price(buy_price, qty, params)
                    actual_total_cost = actual_cost_per_share * qty
                    cash -= actual_total_cost

                    net_sl_per_share = calc_net_sell_price(cand['init_sl'], qty, params)
                    if cand['type'] == 'normal':
                        tp_px = adjust_long_target_price(buy_price + (actual_cost_per_share - net_sl_per_share))
                    else:
                        tp_px = cand['chase_data']['tp']

                    initial_risk = calc_initial_risk_total(actual_cost_per_share, net_sl_per_share, qty, params)
                    portfolio[cand['ticker']] = {
                        'qty': qty,
                        'entry': actual_cost_per_share,
                        'sl': max(cand['init_sl'], cand['init_trail']),
                        'initial_stop': cand['init_sl'],
                        'trailing_stop': cand['init_trail'],
                        'tp_half': tp_px,
                        'sold_half': False,
                        'pure_buy_price': buy_price,
                        'realized_pnl': 0.0,
                        'initial_risk_total': initial_risk,
                        'entry_type': cand['type'],
                    }
                    buyTriggered = True

                    if cand['ticker'] in pending_chases:
                        del pending_chases[cand['ticker']]
                    if not is_training:
                        trade_history.append({
                            "Date": today.strftime('%Y-%m-%d'),
                            "Ticker": cand['ticker'],
                            "Type": f"買進 (EV:{cand['ev']:.2f}R)",
                            "單筆損益": 0.0,
                            "該筆總損益": 0.0,
                            "R_Multiple": 0.0,
                            "Risk": params.fixed_risk
                        })
                elif cand['type'] == 'normal':
                    is_normal_worse_than_sl = True

            if not buyTriggered:
                if cand['type'] == 'normal':
                    total_missed_buys += 1
                    if not is_normal_worse_than_sl:
                        if t_volume > 0:
                            next_day_sizing_equity = (
                                calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params)
                                if getattr(params, 'use_compounding', True) else initial_capital
                            )
                            chase_res = evaluate_chase_condition(
                                t_close, cand['limit_px'], cand['y_atr'], next_day_sizing_equity, params
                            )
                            if chase_res:
                                chase_res['orig_limit'] = cand['limit_px']
                                chase_res['orig_atr'] = cand['y_atr']
                                pending_chases[cand['ticker']] = chase_res
                elif cand['type'] == 'chase':
                    # # (AI註: 零成交量日不可用 stale close 更新續追；維持原 pending_chase 到下一個可交易日)
                    if t_volume <= 0:
                        pass
                    elif cand['chase_data']['sl'] < t_close < cand['chase_data']['tp']:
                        next_day_sizing_equity = (
                            calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params)
                            if getattr(params, 'use_compounding', True) else initial_capital
                        )
                        chase_res = evaluate_chase_condition(
                            t_close, cand['orig_limit'], cand['y_atr'], next_day_sizing_equity, params
                        )
                        if chase_res:
                            chase_res['orig_limit'] = cand['orig_limit']
                            chase_res['orig_atr'] = cand['y_atr']
                            pending_chases[cand['ticker']] = chase_res
                        elif cand['ticker'] in pending_chases:
                            del pending_chases[cand['ticker']]
                    elif cand['ticker'] in pending_chases:
                        del pending_chases[cand['ticker']]
        if profile_stats is not None:
            buy_sec += time.perf_counter() - t0

        t0 = time.perf_counter() if profile_stats is not None else None
        today_equity = calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params)
        if profile_stats is not None:
            equity_mark_sec += time.perf_counter() - t0

        current_equity = today_equity
        invested_capital = today_equity - cash
        exposure_pct = (invested_capital / today_equity) * 100 if today_equity > 0 else 0
        total_exposure += exposure_pct
        if exposure_pct > max_exp:
            max_exp = exposure_pct

        strategy_ret_pct = (today_equity - initial_capital) / initial_capital * 100

        if benchmark_data and benchmark_start_price is not None and has_fast_date(benchmark_data, today):
            current_bm_px = get_fast_close(benchmark_data, date=today)
            bm_ret_pct = (current_bm_px - benchmark_start_price) / benchmark_start_price * 100 if benchmark_start_price > 0 else 0.0

            if bm_peak_price is not None:
                if current_bm_px > bm_peak_price:
                    bm_peak_price = current_bm_px
                current_bm_drawdown = (bm_peak_price - current_bm_px) / bm_peak_price * 100
                if current_bm_drawdown > bm_max_drawdown:
                    bm_max_drawdown = current_bm_drawdown

        if today.month != current_month:
            monthly_equities.append(yesterday_equity)
            if yesterday_bm_px is not None:
                bm_monthly_equities.append(yesterday_bm_px)
            current_month = today.month

        yesterday_equity = today_equity
        yesterday_bm_px = current_bm_px
        year_end_equity[today.year] = today_equity
        year_last_sim_date[today.year] = pd.Timestamp(today)

        if not is_training:
            equity_curve.append({
                "Date": today.strftime('%Y-%m-%d'),
                "Equity": today_equity,
                "Invested_Amount": invested_capital,
                "Exposure_Pct": exposure_pct,
                "Strategy_Return_Pct": strategy_ret_pct,
                f"Benchmark_{benchmark_ticker}_Pct": bm_ret_pct
            })

        if today_equity > peak_equity:
            peak_equity = today_equity
        drawdown = (peak_equity - today_equity) / peak_equity * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        if profile_stats is not None:
            day_loop_sec += time.perf_counter() - t_day_start

    # # (AI註: 月底權益應以期末強制結算後的真實 final equity 為準，故移到 closeout 後再 append)

    final_cash = cash
    last_date = sorted_dates[-1] if len(sorted_dates) > 0 else None

    for ticker in sorted(list(portfolio.keys())):
        pos = portfolio[ticker]
        raw_exit_price = pos.get('last_px', pos.get('pure_buy_price', pos['entry']))
        exec_price = adjust_long_sell_fill_price(raw_exit_price)
        net_price = calc_net_sell_price(exec_price, pos['qty'], params)
        final_cash += net_price * pos['qty']

        pnl = (net_price - pos['entry']) * pos['qty']
        total_pnl = pos['realized_pnl'] + pnl
        total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
        closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')})
        if is_extended_entry_type(pos.get('entry_type', 'normal')):
            extended_trade_count += 1
        else:
            normal_trade_count += 1

        if not is_training:
            trade_history.append({
                "Date": last_date.strftime('%Y-%m-%d') if last_date else "",
                "Ticker": ticker,
                "Type": "期末強制結算",
                "單筆損益": pnl,
                "該筆總損益": total_pnl,
                "R_Multiple": total_r,
                "Risk": params.fixed_risk
            })

    today_equity = final_cash
    if last_date is not None and last_date.year in year_end_equity:
        year_end_equity[last_date.year] = today_equity

    # # (AI註: 期末強制結算後補做一次 peak / drawdown 更新，避免 final closeout 對 MDD 漏算)
    if today_equity > peak_equity:
        peak_equity = today_equity
    drawdown = (peak_equity - today_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    if drawdown > max_drawdown:
        max_drawdown = drawdown

    if len(sorted_dates) > start_idx:
        monthly_equities.append(today_equity)
        if current_bm_px is not None:
            bm_monthly_equities.append(current_bm_px)

    total_return = (today_equity - initial_capital) / initial_capital * 100

    if not is_training and len(equity_curve) > 0:
        equity_curve[-1]['Equity'] = today_equity
        equity_curve[-1]['Strategy_Return_Pct'] = total_return
        equity_curve[-1]['Invested_Amount'] = 0.0
        equity_curve[-1]['Exposure_Pct'] = 0.0

    t0 = time.perf_counter() if profile_stats is not None else None
    r_squared, monthly_win_rate = calc_curve_stats(monthly_equities)
    bm_r_squared, bm_monthly_win_rate = calc_curve_stats(bm_monthly_equities)
    curve_stats_sec = (time.perf_counter() - t0) if profile_stats is not None else 0.0

    trade_count = len(closed_trades_stats)
    if trade_count > 0:
        wins = [t for t in closed_trades_stats if t['pnl'] > 0]
        losses = [t for t in closed_trades_stats if t['pnl'] <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / trade_count) * 100
        avg_win_amount = sum(t['pnl'] for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss_amount = abs(sum(t['pnl'] for t in losses) / loss_count) if loss_count > 0 else 0.0
        pf_payoff = (avg_win_amount / avg_loss_amount) if avg_loss_amount > 0 else (99.9 if avg_win_amount > 0 else 0.0)

        avg_win_r = sum(t['r_mult'] for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss_r = abs(sum(t['r_mult'] for t in losses) / loss_count) if loss_count > 0 else 0.0
        if EV_CALC_METHOD == 'B':
            payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
            pf_ev = (win_rate / 100.0 * payoff_for_ev) - (1 - win_rate / 100.0)
        else:
            pf_ev = sum(t['r_mult'] for t in closed_trades_stats) / trade_count
    else:
        win_rate, pf_ev, pf_payoff = 0.0, 0.0, 0.0

    avg_exp = total_exposure / sim_days if sim_days > 0 else 0.0
    sim_years = calc_sim_years(sorted_dates, start_idx)
    annual_trades = (trade_count / sim_years) if sim_years > 0 else 0.0
    reserved_buy_fill_rate = (normal_trade_count / (normal_trade_count + total_missed_buys) * 100.0) if (normal_trade_count + total_missed_buys) > 0 else 0.0
    annual_return_pct = calc_annual_return_pct(initial_capital, today_equity, sim_years)

    bm_start_value = float(benchmark_start_price) if benchmark_start_price is not None else 0.0
    bm_end_value = bm_start_value * (1.0 + bm_ret_pct / 100.0) if bm_start_value > 0 else 0.0
    bm_annual_return_pct = calc_annual_return_pct(bm_start_value, bm_end_value, sim_years)

    yearly_stats = build_full_year_return_stats(
        sorted_dates=sorted_dates,
        year_start_equity=year_start_equity,
        year_end_equity=year_end_equity,
        year_first_sim_date=year_first_sim_date,
        year_last_sim_date=year_last_sim_date,
    )
    bm_yearly_stats = build_benchmark_full_year_return_stats(
        sorted_dates=sorted_dates,
        benchmark_data=benchmark_data,
        yearly_return_rows=yearly_stats['yearly_return_rows'],
    )

    if profile_stats is not None:
        profile_stats['portfolio_wall_sec'] = time.perf_counter() - t_portfolio_start
        profile_stats['portfolio_ticker_dates_sec'] = 0.0
        profile_stats['portfolio_build_trade_index_sec'] = build_trade_index_sec
        profile_stats['portfolio_day_loop_sec'] = day_loop_sec
        profile_stats['portfolio_candidate_scan_sec'] = candidate_scan_sec
        profile_stats['portfolio_rotation_sec'] = rotation_sec
        profile_stats['portfolio_settle_sec'] = settle_sec
        profile_stats['portfolio_buy_sec'] = buy_sec
        profile_stats['portfolio_equity_mark_sec'] = equity_mark_sec
        profile_stats['portfolio_closeout_sec'] = 0.0
        profile_stats['curve_stats_sec'] = curve_stats_sec
        profile_stats['sim_years'] = sim_years
        profile_stats['annual_return_pct'] = annual_return_pct
        profile_stats['bm_annual_return_pct'] = bm_annual_return_pct
        profile_stats.update(yearly_stats)
        profile_stats.update(bm_yearly_stats)

    if is_training:
        chase_trade_count = extended_trade_count  # legacy tuple alias
        return total_return, max_drawdown, trade_count, today_equity, avg_exp, max_exp, bm_ret_pct, bm_max_drawdown, win_rate, pf_ev, pf_payoff, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate, normal_trade_count, chase_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct

    df_equity = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trade_history)
    final_bm_return = df_equity.iloc[-1][f"Benchmark_{benchmark_ticker}_Pct"] if not df_equity.empty else 0.0
    return df_equity, df_trades, total_return, max_drawdown, trade_count, win_rate, pf_ev, pf_payoff, today_equity, avg_exp, max_exp, final_bm_return, bm_max_drawdown, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate, normal_trade_count, chase_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct