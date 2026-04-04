import numpy as np

from core.capital_policy import resolve_single_backtest_sizing_capital
from core.signal_utils import generate_signals
from tools.debug.entry_flow import process_debug_entry_for_day
from tools.debug.exit_flow import append_debug_forced_closeout, process_debug_position_step
from tools.debug.reporting import finalize_debug_trade_logs


def _extract_precomputed_signals(df):
    required_columns = {'ATR', 'is_setup', 'ind_sell_signal', 'buy_limit'}
    if not required_columns.issubset(df.columns):
        return None
    return (
        df['ATR'].to_numpy(copy=False),
        df['is_setup'].to_numpy(copy=False),
        df['ind_sell_signal'].to_numpy(copy=False),
        df['buy_limit'].to_numpy(copy=False),
    )


def run_debug_backtest(df, ticker, params, output_dir, colors, export_excel=True, verbose=True, precomputed_signals=None):
    """以正式核心邏輯為準，輸出可讀交易明細的除錯工具"""
    h = df['High'].to_numpy(dtype=np.float64, copy=False)
    l = df['Low'].to_numpy(dtype=np.float64, copy=False)
    c = df['Close'].to_numpy(dtype=np.float64, copy=False)
    o = df['Open'].to_numpy(dtype=np.float64, copy=False)
    v = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    dates = df.index

    if precomputed_signals is None:
        precomputed_signals = _extract_precomputed_signals(df)
    if precomputed_signals is None:
        precomputed_signals = generate_signals(df, params)
    atr_main, buy_condition, sell_condition, buy_limits = precomputed_signals

    position = {'qty': 0}
    active_extended_signal = None
    current_capital = params.initial_capital
    trade_logs = []

    for j in range(1, len(c)):
        if np.isnan(atr_main[j - 1]):
            continue

        pos_qty_start_of_bar = position['qty']

        if pos_qty_start_of_bar > 0:
            position, pnl_realized = process_debug_position_step(
                position=position,
                atr_prev=atr_main[j - 1],
                sell_condition_prev=sell_condition[j - 1],
                close_prev=c[j - 1],
                t_open=o[j],
                t_high=h[j],
                t_low=l[j],
                t_close=c[j],
                t_volume=v[j],
                current_date=dates[j],
                params=params,
                trade_logs=trade_logs,
            )
            current_capital += pnl_realized

        sizing_cap = resolve_single_backtest_sizing_capital(params, current_capital)
        position, active_extended_signal = process_debug_entry_for_day(
            position=position,
            pos_qty_start_of_bar=pos_qty_start_of_bar,
            active_extended_signal=active_extended_signal,
            buy_condition_prev=buy_condition[j - 1],
            buy_limit_prev=buy_limits[j - 1],
            atr_prev=atr_main[j - 1],
            close_prev=c[j - 1],
            sizing_cap=sizing_cap,
            t_open=o[j],
            t_high=h[j],
            t_low=l[j],
            t_close=c[j],
            t_volume=v[j],
            current_date=dates[j],
            params=params,
            trade_logs=trade_logs,
        )

    if position['qty'] > 0:
        position['close_price'] = c[-1]
        append_debug_forced_closeout(
            position=position,
            current_date=dates[-1],
            atr_last=atr_main[-1] if len(atr_main) > 0 else np.nan,
            params=params,
            trade_logs=trade_logs,
        )

    return finalize_debug_trade_logs(
        trade_logs=trade_logs,
        ticker=ticker,
        output_dir=output_dir,
        colors=colors,
        export_excel=export_excel,
        verbose=verbose,
    )
