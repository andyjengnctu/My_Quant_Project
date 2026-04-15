import bisect
import time

import numpy as np

from core.backtest_core import run_v16_backtest
from core.history_filters import evaluate_history_candidate_metrics
from core.exact_accounting import build_sell_ledger_from_price, coerce_money_like_to_milli, restore_money_like_from_milli
from core.price_utils import adjust_long_sell_fill_price
from core.signal_utils import generate_signals


# # (AI註: 單一真理來源 - 浮動權益估值與延續候選的 next-day sizing 共用同一口徑)
def calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params):
    cash_template = cash
    equity_milli = coerce_money_like_to_milli(cash)

    for ticker in sorted(portfolio.keys()):
        pos = portfolio[ticker]
        pt_data = all_dfs_fast[ticker]
        pt_pos = get_fast_pos(pt_data, today)

        if pt_pos >= 0:
            px = get_fast_close(pt_data, pos=pt_pos)
        else:
            px = pos.get('last_px', pos.get('pure_buy_price', pos['entry']))

        pos['last_px'] = px

        floating_exec_price = adjust_long_sell_fill_price(px, ticker=ticker)
        sell_ledger = build_sell_ledger_from_price(
            floating_exec_price,
            pos['qty'],
            params,
            ticker=ticker,
            security_profile=pos.get('security_profile'),
            trade_date=today,
        )
        equity_milli += sell_ledger['net_sell_total_milli']

    return restore_money_like_from_milli(equity_milli, cash_template)


FAST_STATIC_FLOAT_FIELDS = ('Open', 'High', 'Low', 'Close', 'Volume')
FAST_DYNAMIC_FLOAT_FIELDS = ('ATR', 'buy_limit')
FAST_DYNAMIC_BOOL_FIELDS = ('is_setup', 'ind_sell_signal')
FAST_FLOAT_FIELDS = FAST_STATIC_FLOAT_FIELDS + FAST_DYNAMIC_FLOAT_FIELDS
FAST_BOOL_FIELDS = FAST_DYNAMIC_BOOL_FIELDS



def pack_static_market_data(df):
    packed = {
        '_packed_market_data': True,
        'dates': tuple(df.index.tolist()),
        'date_to_pos': {dt: i for i, dt in enumerate(df.index)},
        'security_profile': df.attrs.get('security_profile'),
    }
    for field in FAST_STATIC_FLOAT_FIELDS:
        packed[field] = df[field].to_numpy(dtype=np.float64, copy=True)
    return packed


def pack_optimizer_dynamic_data(precomputed_signals):
    atr_main, buy_condition, sell_condition, buy_limits = precomputed_signals
    return {
        'ATR': np.asarray(atr_main, dtype=np.float64),
        'buy_limit': np.asarray(buy_limits, dtype=np.float64),
        'is_setup': np.asarray(buy_condition, dtype=bool),
        'ind_sell_signal': np.asarray(sell_condition, dtype=bool),
    }


def merge_static_market_with_dynamic(static_data, dynamic_data):
    merged = dict(static_data)
    merged.update(dynamic_data)
    return merged


def pack_prepared_stock_data(df):
    packed = pack_static_market_data(df)
    for field in FAST_DYNAMIC_FLOAT_FIELDS:
        packed[field] = df[field].to_numpy(dtype=np.float64, copy=True)
    for field in FAST_DYNAMIC_BOOL_FIELDS:
        packed[field] = df[field].to_numpy(dtype=bool, copy=True)
    return packed


def is_packed_market_data(data):
    return isinstance(data, dict) and data.get('_packed_market_data', False) is True


def get_fast_security_profile(data):
    if is_packed_market_data(data):
        return data.get('security_profile')
    return None


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


def prep_optimizer_stock_data_bundle(df, params, profile_stats=None, ticker=None):
    t_total_start = time.perf_counter() if profile_stats is not None else None
    resolved_ticker = ticker or df.attrs.get('ticker')

    if profile_stats is not None:
        profile_stats['copy_sec'] = 0.0
        profile_stats['assign_columns_sec'] = 0.0

    t0 = time.perf_counter() if profile_stats is not None else None
    precomputed_signals = generate_signals(df, params, ticker=resolved_ticker)
    if profile_stats is not None:
        profile_stats['generate_signals_sec'] = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_stats is not None else None
    _stats_unused, standalone_logs = run_v16_backtest(
        df,
        params,
        return_logs=True,
        precomputed_signals=precomputed_signals,
        ticker=resolved_ticker,
        collect_stats=False,
    )
    if profile_stats is not None:
        profile_stats['run_backtest_sec'] = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_stats is not None else None
    dynamic_data = pack_optimizer_dynamic_data(precomputed_signals)
    if profile_stats is not None:
        profile_stats['to_dict_sec'] = time.perf_counter() - t0
        profile_stats['total_sec'] = time.perf_counter() - t_total_start

    return dynamic_data, standalone_logs


def prep_stock_data_and_trades(df, params, profile_stats=None, return_stats=False, ticker=None):
    t_total_start = time.perf_counter() if profile_stats is not None else None

    t0 = time.perf_counter() if profile_stats is not None else None
    df = df.copy()
    resolved_ticker = ticker or df.attrs.get('ticker')
    if profile_stats is not None:
        profile_stats['copy_sec'] = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_stats is not None else None
    precomputed_signals = generate_signals(df, params, ticker=resolved_ticker)
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
    if return_stats:
        stats_dict, standalone_logs = run_v16_backtest(
            df,
            params,
            return_logs=True,
            precomputed_signals=precomputed_signals,
            ticker=resolved_ticker,
            collect_stats=True,
        )
    else:
        _stats_unused, standalone_logs = run_v16_backtest(
            df,
            params,
            return_logs=True,
            precomputed_signals=precomputed_signals,
            ticker=resolved_ticker,
            collect_stats=False,
        )
        stats_dict = None
    if profile_stats is not None:
        profile_stats['run_backtest_sec'] = time.perf_counter() - t0
        profile_stats['total_sec'] = time.perf_counter() - t_total_start

    if return_stats:
        return df, standalone_logs, stats_dict
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
            'cum_pnl_sum': np.array([], dtype=np.float64),
        }

    ordered_logs = sorted(trade_logs, key=lambda t: t['exit_date'])

    exit_dates = []
    cum_trade_count = []
    cum_win_count = []
    cum_win_r_sum = []
    cum_loss_r_sum = []
    cum_total_r_sum = []
    cum_pnl_sum = []

    trade_count = 0
    win_count = 0
    win_r_sum = 0.0
    loss_r_sum = 0.0
    total_r_sum = 0.0
    pnl_sum = 0.0

    for trade in ordered_logs:
        trade_count += 1

        pnl = float(trade.get('pnl', 0.0))
        r_mult = float(trade.get('r_mult', 0.0))
        total_r_sum += r_mult
        pnl_sum += pnl

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
        cum_pnl_sum.append(pnl_sum)

    return {
        'exit_dates': exit_dates,
        'cum_trade_count': np.array(cum_trade_count, dtype=np.int32),
        'cum_win_count': np.array(cum_win_count, dtype=np.int32),
        'cum_win_r_sum': np.array(cum_win_r_sum, dtype=np.float64),
        'cum_loss_r_sum': np.array(cum_loss_r_sum, dtype=np.float64),
        'cum_total_r_sum': np.array(cum_total_r_sum, dtype=np.float64),
        'cum_pnl_sum': np.array(cum_pnl_sum, dtype=np.float64),
    }


def get_pit_stats_from_index(stats_index, current_date, params):
    cutoff = bisect.bisect_left(stats_index['exit_dates'], current_date)
    if cutoff <= 0:
        is_candidate, expected_value, history_win_rate, history_trade_count = evaluate_history_candidate_metrics(
            0,
            0,
            0.0,
            0.0,
            0.0,
            params,
        )
        return is_candidate, expected_value, history_win_rate, history_trade_count, 0.0

    trade_count = int(stats_index['cum_trade_count'][cutoff - 1])
    win_count = int(stats_index['cum_win_count'][cutoff - 1])
    total_r_sum = float(stats_index['cum_total_r_sum'][cutoff - 1])
    win_r_sum = float(stats_index['cum_win_r_sum'][cutoff - 1])
    loss_r_sum = float(stats_index['cum_loss_r_sum'][cutoff - 1])
    pnl_sum = float(stats_index['cum_pnl_sum'][cutoff - 1])
    is_candidate, expected_value, history_win_rate, history_trade_count = evaluate_history_candidate_metrics(
        trade_count,
        win_count,
        total_r_sum,
        win_r_sum,
        loss_r_sum,
        params,
    )
    asset_growth_pct = (pnl_sum / float(params.initial_capital) * 100.0) if float(params.initial_capital) > 0 else 0.0
    return is_candidate, expected_value, history_win_rate, history_trade_count, asset_growth_pct


def is_extended_entry_type(entry_type):
    return entry_type == 'extended'
