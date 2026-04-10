import pandas as pd

from core.backtest_core import run_v16_backtest
from core.buy_sort import calc_buy_sort_value
from core.config import get_buy_sort_method
from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.exact_accounting import calc_entry_total_cost
from core.price_utils import calc_entry_price, calc_reference_candidate_qty, can_execute_half_take_profit
from core.scanner_display import build_scanner_sort_probe_text
from .runtime_common import is_insufficient_data_error


def _build_sanitize_issue(ticker, sanitize_stats):
    invalid_row_count = int(sanitize_stats.get('invalid_row_count', 0))
    duplicate_date_count = int(sanitize_stats.get('duplicate_date_count', 0))
    dropped_row_count = int(sanitize_stats.get('dropped_row_count', 0))
    if dropped_row_count <= 0:
        return None
    return (
        f"{ticker}: 清洗移除 {dropped_row_count} 列 "
        f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
    )


def _build_stat_str(*, expected_value, win_rate_pct, trade_count, asset_growth_pct, sort_value):
    return build_scanner_sort_probe_text(
        ev=expected_value,
        win_rate=win_rate_pct,
        trade_count=trade_count,
        asset_growth_pct=asset_growth_pct,
        sort_value=sort_value,
    )


def _calc_sort_value(*, expected_value, proj_cost, win_rate_pct, trade_count, asset_growth_pct):
    return calc_buy_sort_value(
        get_buy_sort_method(),
        expected_value,
        0.0 if proj_cost is None else proj_cost,
        win_rate_pct / 100.0,
        trade_count,
        asset_growth_pct,
    )


def build_history_qualified_row_from_stats(*, ticker, stats, params, sanitize_stats):
    if not stats or not stats['is_candidate']:
        return None

    sanitize_issue = _build_sanitize_issue(ticker, sanitize_stats)
    expected_value = float(stats['expected_value'])
    win_rate_pct = float(stats['win_rate'])
    trade_count = int(stats['trade_count'])
    asset_growth_pct = float(stats.get('asset_growth', 0.0))

    if stats['is_setup_today']:
        proj_qty = calc_reference_candidate_qty(stats['buy_limit'], stats['stop_loss'], params)
        if proj_qty > 0:
            proj_cost = calc_entry_total_cost(stats['buy_limit'], proj_qty, params)
            half_tp_note = " | 半倉停利:股數不足" if not can_execute_half_take_profit(proj_qty, params.tp_percent) else ""
            detail = f"限價買進:{stats['buy_limit']:>6.2f} | 參考投入:{proj_cost:>7,.0f}{half_tp_note}"
            kind = 'buy'
        else:
            proj_cost = 0.0
            detail = "新訊號成立 | 股數不足，今日不掛單"
            kind = 'candidate'
        sort_value = _calc_sort_value(
            expected_value=expected_value,
            proj_cost=proj_cost,
            win_rate_pct=win_rate_pct,
            trade_count=trade_count,
            asset_growth_pct=asset_growth_pct,
        )
        stat_str = _build_stat_str(
            expected_value=expected_value,
            win_rate_pct=win_rate_pct,
            trade_count=trade_count,
            asset_growth_pct=asset_growth_pct,
            sort_value=sort_value,
        )
        return {
            'kind': kind,
            'ticker': ticker,
            'proj_cost': proj_cost,
            'ev': expected_value,
            'expected_value': expected_value,
            'sort_value': sort_value,
            'text': f"{ticker:<6} | {stat_str} | {detail}",
            'sanitize_issue': sanitize_issue,
            'win_rate': win_rate_pct,
            'trade_count': trade_count,
            'asset_growth': asset_growth_pct,
        }

    extended_candidate = stats.get('extended_candidate_today')
    extended_orderable_today = bool(stats.get('extended_orderable_today', extended_candidate is not None))
    if extended_candidate is not None:
        limit_price = extended_candidate.get('limit_price')
        init_sl = extended_candidate.get('init_sl')
        proj_cost = None
        barrier_parts = []
        if limit_price is not None and not pd.isna(limit_price):
            barrier_parts.append(f"延續掛單:{float(limit_price):>6.2f}")
        else:
            barrier_parts.append("延續掛單:-")
        invalidation_barrier = extended_candidate.get('continuation_invalidation_barrier')
        completion_barrier = extended_candidate.get('continuation_completion_barrier')
        if invalidation_barrier is not None and not pd.isna(invalidation_barrier):
            barrier_parts.append(f"失效線:{float(invalidation_barrier):>6.2f}")
        if completion_barrier is not None and not pd.isna(completion_barrier):
            barrier_parts.append(f"達標線:{float(completion_barrier):>6.2f}")

        proj_qty = 0
        if limit_price is not None and init_sl is not None and not pd.isna(limit_price) and not pd.isna(init_sl):
            proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params)
            if proj_qty > 0:
                proj_cost = calc_entry_total_cost(limit_price, proj_qty, params)
                barrier_parts.append(f"參考投入:{proj_cost:>7,.0f}")
                if not can_execute_half_take_profit(proj_qty, params.tp_percent):
                    barrier_parts.append("半倉停利:股數不足")

        if extended_orderable_today and proj_qty > 0:
            kind = 'extended'
        else:
            kind = 'candidate'
            if not extended_orderable_today:
                barrier_parts.append("今日不可掛單")
            elif proj_qty == 0:
                barrier_parts.append("股數不足，今日不掛單")
            else:
                barrier_parts.append("今日延續觀察")

        sort_value = _calc_sort_value(
            expected_value=expected_value,
            proj_cost=proj_cost,
            win_rate_pct=win_rate_pct,
            trade_count=trade_count,
            asset_growth_pct=asset_growth_pct,
        )
        stat_str = _build_stat_str(
            expected_value=expected_value,
            win_rate_pct=win_rate_pct,
            trade_count=trade_count,
            asset_growth_pct=asset_growth_pct,
            sort_value=sort_value,
        )
        return {
            'kind': kind,
            'ticker': ticker,
            'proj_cost': proj_cost,
            'ev': expected_value,
            'expected_value': expected_value,
            'sort_value': sort_value,
            'text': f"{ticker:<6} | {stat_str} | {' | '.join(barrier_parts)}",
            'sanitize_issue': sanitize_issue,
            'win_rate': win_rate_pct,
            'trade_count': trade_count,
            'asset_growth': asset_growth_pct,
        }

    sort_value = _calc_sort_value(
        expected_value=expected_value,
        proj_cost=None,
        win_rate_pct=win_rate_pct,
        trade_count=trade_count,
        asset_growth_pct=asset_growth_pct,
    )
    stat_str = _build_stat_str(
        expected_value=expected_value,
        win_rate_pct=win_rate_pct,
        trade_count=trade_count,
        asset_growth_pct=asset_growth_pct,
        sort_value=sort_value,
    )
    return {
        'kind': 'candidate',
        'ticker': ticker,
        'proj_cost': None,
        'ev': expected_value,
        'expected_value': expected_value,
        'sort_value': sort_value,
        'text': f"{ticker:<6} | {stat_str} | 僅歷績符合：今日無新訊號 / 延續掛單",
        'sanitize_issue': sanitize_issue,
        'win_rate': win_rate_pct,
        'trade_count': trade_count,
        'asset_growth': asset_growth_pct,
    }


def build_scanner_response_from_stats(*, ticker, stats, params, sanitize_stats):
    history_row = build_history_qualified_row_from_stats(
        ticker=ticker,
        stats=stats,
        params=params,
        sanitize_stats=sanitize_stats,
    )
    if history_row is None:
        return None

    if history_row['kind'] not in ('buy', 'extended'):
        return ('candidate', None, None, None, None, ticker, history_row['sanitize_issue'])

    return (
        history_row['kind'],
        history_row['proj_cost'],
        history_row['expected_value'],
        history_row['sort_value'],
        history_row['text'],
        history_row['ticker'],
        history_row['sanitize_issue'],
    )


def _build_precomputed_signals(df):
    required_columns = {'ATR', 'is_setup', 'ind_sell_signal', 'buy_limit'}
    if not required_columns.issubset(df.columns):
        return None
    return (
        df['ATR'].to_numpy(copy=False),
        df['is_setup'].to_numpy(copy=False),
        df['ind_sell_signal'].to_numpy(copy=False),
        df['buy_limit'].to_numpy(copy=False),
    )


def process_prepared_stock(df, ticker, params, sanitize_stats=None):
    sanitize_stats = sanitize_stats or {}
    precomputed_signals = _build_precomputed_signals(df)
    stats = run_v16_backtest(df, params, precomputed_signals=precomputed_signals, ticker=ticker)
    return build_scanner_response_from_stats(ticker=ticker, stats=stats, params=params, sanitize_stats=sanitize_stats)


def process_prepared_history_qualified_stock(df, ticker, params, sanitize_stats=None):
    sanitize_stats = sanitize_stats or {}
    precomputed_signals = _build_precomputed_signals(df)
    stats = run_v16_backtest(df, params, precomputed_signals=precomputed_signals, ticker=ticker)
    history_row = build_history_qualified_row_from_stats(
        ticker=ticker,
        stats=stats,
        params=params,
        sanitize_stats=sanitize_stats,
    )
    if history_row is None:
        return None
    history_row['status'] = history_row['kind']
    return history_row


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


def process_single_stock_history_qualified(file_path, ticker, params):
    try:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
        return process_prepared_history_qualified_stock(df, ticker, params, sanitize_stats=sanitize_stats)

    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
        if is_insufficient_data_error(e):
            return {'status': 'skip_insufficient', 'ticker': ticker, 'sanitize_issue': None}
        raise RuntimeError(f"{ticker} 處理失敗 | {type(e).__name__}: {e}") from e


# # (AI註: 相容舊名稱，避免外部直接引用時中斷)
def _build_scanner_response_from_stats(*, ticker, stats, params, sanitize_stats):
    return build_scanner_response_from_stats(ticker=ticker, stats=stats, params=params, sanitize_stats=sanitize_stats)
