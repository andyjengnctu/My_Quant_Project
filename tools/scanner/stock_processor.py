import pandas as pd

from core.buy_sort import calc_buy_sort_value
from core.config import BUY_SORT_METHOD
from core.backtest_core import (
    adjust_long_target_price,
    calc_entry_price,
    calc_net_sell_price,
    calc_reference_candidate_qty,
    can_execute_half_take_profit,
    run_v16_backtest,
)
from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from .runtime_common import is_insufficient_data_error


def build_scanner_response_from_stats(*, ticker, stats, params, sanitize_stats):
    if not stats or not stats['is_candidate']:
        return None

    invalid_row_count = int(sanitize_stats.get('invalid_row_count', 0))
    duplicate_date_count = int(sanitize_stats.get('duplicate_date_count', 0))
    dropped_row_count = int(sanitize_stats.get('dropped_row_count', 0))

    sanitize_issue = None
    if dropped_row_count > 0:
        sanitize_issue = (
            f"{ticker}: 清洗移除 {dropped_row_count} 列 "
            f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
        )

    stat_str = f"勝率:{stats['win_rate']:>5.1f}% | 期望值:{stats['expected_value']:>5.2f}R | 交易:{stats['trade_count']:>3}次 | MDD:{stats['max_drawdown']:>5.1f}%"

    if stats['is_setup_today']:
        proj_qty = calc_reference_candidate_qty(stats['buy_limit'], stats['stop_loss'], params)
        if proj_qty == 0:
            return ('candidate', None, None, None, None, ticker, sanitize_issue)

        proj_cost = calc_entry_price(stats['buy_limit'], proj_qty, params) * proj_qty

        if can_execute_half_take_profit(proj_qty, params.tp_percent):
            actual_cost_per_share = calc_entry_price(stats['buy_limit'], proj_qty, params)
            net_sl_per_share = calc_net_sell_price(stats['stop_loss'], proj_qty, params)
            est_target = adjust_long_target_price(stats['buy_limit'] + (actual_cost_per_share - net_sl_per_share))
            buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 停利(預估):{est_target:>6.2f} | 參考投入:{proj_cost:>7,.0f}"
        elif params.tp_percent <= 0:
            buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 半倉停利:關閉 | 參考投入:{proj_cost:>7,.0f}"
        else:
            buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 半倉停利:股數不足 | 參考投入:{proj_cost:>7,.0f}"
        msg = f"{ticker:<6} | {stat_str} | {buy_str}"

        sort_value = calc_buy_sort_value(BUY_SORT_METHOD, stats['expected_value'], proj_cost, stats['win_rate'] / 100.0, stats['trade_count'])
        return ('buy', proj_cost, stats['expected_value'], sort_value, msg, ticker, sanitize_issue)

    extended_candidate = stats.get('extended_candidate_today')
    if extended_candidate is not None:
        limit_price = extended_candidate.get('limit_price')
        init_sl = extended_candidate.get('init_sl')
        if limit_price is None or init_sl is None:
            return ('candidate', None, None, None, None, ticker, sanitize_issue)

        proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params)
        if proj_qty == 0:
            return ('candidate', None, None, None, None, ticker, sanitize_issue)

        proj_cost = calc_entry_price(limit_price, proj_qty, params) * proj_qty

        if can_execute_half_take_profit(proj_qty, params.tp_percent):
            actual_cost_per_share = calc_entry_price(limit_price, proj_qty, params)
            net_sl_per_share = calc_net_sell_price(init_sl, proj_qty, params)
            est_target = adjust_long_target_price(limit_price + (actual_cost_per_share - net_sl_per_share))
            buy_str = f"延續掛單:{limit_price:>6.2f} | 停損:{init_sl:>6.2f} | 停利(預估):{est_target:>6.2f} | 參考投入:{proj_cost:>7,.0f}"
        elif params.tp_percent <= 0:
            buy_str = f"延續掛單:{limit_price:>6.2f} | 停損:{init_sl:>6.2f} | 半倉停利:關閉 | 參考投入:{proj_cost:>7,.0f}"
        else:
            buy_str = f"延續掛單:{limit_price:>6.2f} | 停損:{init_sl:>6.2f} | 半倉停利:股數不足 | 參考投入:{proj_cost:>7,.0f}"
        msg = f"{ticker:<6} | {stat_str} | {buy_str}"

        sort_value = calc_buy_sort_value(BUY_SORT_METHOD, stats['expected_value'], proj_cost, stats['win_rate'] / 100.0, stats['trade_count'])
        return ('extended', proj_cost, stats['expected_value'], sort_value, msg, ticker, sanitize_issue)

    return ('candidate', None, None, None, None, ticker, sanitize_issue)


def process_prepared_stock(df, ticker, params, sanitize_stats=None):
    sanitize_stats = sanitize_stats or {}
    precomputed_signals = None
    required_columns = {'ATR', 'is_setup', 'ind_sell_signal', 'buy_limit'}
    if required_columns.issubset(df.columns):
        precomputed_signals = (
            df['ATR'].to_numpy(copy=False),
            df['is_setup'].to_numpy(copy=False),
            df['ind_sell_signal'].to_numpy(copy=False),
            df['buy_limit'].to_numpy(copy=False),
        )
    stats = run_v16_backtest(df, params, precomputed_signals=precomputed_signals)
    return build_scanner_response_from_stats(ticker=ticker, stats=stats, params=params, sanitize_stats=sanitize_stats)


def process_single_stock(file_path, ticker, params):
    try:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
        return process_prepared_stock(df, ticker, params, sanitize_stats=sanitize_stats)

    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
        if is_insufficient_data_error(e):
            return ('skip_insufficient', None, None, None, None, ticker, None)
        raise RuntimeError(f"{ticker} 處理失敗 | {type(e).__name__}: {e}") from e


# # (AI註: 相容舊名稱，避免外部直接引用時中斷)
def _build_scanner_response_from_stats(*, ticker, stats, params, sanitize_stats):
    return build_scanner_response_from_stats(ticker=ticker, stats=stats, params=params, sanitize_stats=sanitize_stats)
