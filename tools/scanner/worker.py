import os
import pandas as pd

from core.v16_buy_sort import calc_buy_sort_value
from core.v16_config import BUY_SORT_METHOD
from core.v16_core import (
    adjust_long_target_price,
    calc_entry_price,
    calc_net_sell_price,
    calc_reference_candidate_qty,
    can_execute_half_take_profit,
    run_v16_backtest,
)
from core.v16_data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.v16_display import C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_scanner_header
from core.v16_log_utils import format_exception_summary
from core.v16_params_io import load_params_from_json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "v16_best_params.json")
SCANNER_PROGRESS_EVERY = 25
DEFAULT_SCANNER_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1) // 2))


def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def resolve_scanner_max_workers(params):
    configured = getattr(params, 'scanner_max_workers', DEFAULT_SCANNER_MAX_WORKERS)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_SCANNER_MAX_WORKERS
    return max(1, configured)


def load_strict_params(json_file):
    return load_params_from_json(json_file)


def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))


def process_single_stock(file_path, ticker, params):
    try:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

        invalid_row_count = sanitize_stats['invalid_row_count']
        duplicate_date_count = sanitize_stats['duplicate_date_count']
        dropped_row_count = sanitize_stats['dropped_row_count']

        stats = run_v16_backtest(df, params)

        if not stats or not stats['is_candidate']:
            return None

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
                raise KeyError("extended candidate 缺少 limit_price/init_sl")

            proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params)
            if proj_qty == 0:
                return ('candidate', None, None, None, None, ticker, sanitize_issue)

            proj_cost = calc_entry_price(limit_price, proj_qty, params) * proj_qty
            extended_str = f"延續限價:{limit_price:>6.2f} | 停損:{init_sl:>6.2f} | 參考投入:{proj_cost:>7,.0f}"
            msg = f"{ticker:<6} | {stat_str} | {extended_str}"
            sort_value = calc_buy_sort_value(BUY_SORT_METHOD, stats['expected_value'], proj_cost, stats['win_rate'] / 100.0, stats['trade_count'])
            return ('extended', proj_cost, stats['expected_value'], sort_value, msg, ticker, sanitize_issue)

        return ('candidate', None, None, None, None, ticker, sanitize_issue)

    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
        if is_insufficient_data_error(e):
            return ('skip_insufficient', None, None, None, None, ticker, None)
        raise RuntimeError(
            f"scanner 處理失敗: ticker={ticker} | {format_exception_summary(e)}"
        ) from e
