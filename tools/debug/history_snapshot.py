import bisect

from core.history_filters import evaluate_history_candidate_metrics


def compute_payoff_ratio_from_trade_index(stats_index, cutoff):
    if cutoff <= 0:
        return 0.0
    trade_count = int(stats_index['cum_trade_count'][cutoff - 1])
    win_count = int(stats_index['cum_win_count'][cutoff - 1])
    loss_count = max(0, trade_count - win_count)
    win_r_sum = float(stats_index['cum_win_r_sum'][cutoff - 1])
    loss_r_sum = float(stats_index['cum_loss_r_sum'][cutoff - 1])
    avg_win_r = (win_r_sum / win_count) if win_count > 0 else 0.0
    avg_loss_r = abs(loss_r_sum / loss_count) if loss_count > 0 else 0.0
    if avg_loss_r > 0:
        return avg_win_r / avg_loss_r
    return 99.9 if avg_win_r > 0 else 0.0


def build_pit_history_snapshot(
    stats_index,
    current_date,
    params,
    current_capital,
    overall_max_drawdown=0.0,
    *,
    include_current_date_exits=False,
):
    exit_dates = stats_index.get('exit_dates', []) if stats_index is not None else []
    if include_current_date_exits:
        cutoff = bisect.bisect_right(exit_dates, current_date) if exit_dates else 0
    else:
        cutoff = bisect.bisect_left(exit_dates, current_date) if exit_dates else 0
    trade_count = 0
    win_rate = 0.0
    expected_value = 0.0
    payoff_ratio = 0.0
    is_candidate = False
    if cutoff > 0:
        trade_count = int(stats_index['cum_trade_count'][cutoff - 1])
        win_count = int(stats_index['cum_win_count'][cutoff - 1])
        total_r_sum = float(stats_index['cum_total_r_sum'][cutoff - 1])
        win_r_sum = float(stats_index['cum_win_r_sum'][cutoff - 1])
        loss_r_sum = float(stats_index['cum_loss_r_sum'][cutoff - 1])
        is_candidate, expected_value, win_rate, _ = evaluate_history_candidate_metrics(
            trade_count,
            win_count,
            total_r_sum,
            win_r_sum,
            loss_r_sum,
            params,
        )
        payoff_ratio = compute_payoff_ratio_from_trade_index(stats_index, cutoff)
    initial_capital = float(params.initial_capital)
    asset_growth_pct = ((float(current_capital) - initial_capital) / initial_capital * 100.0) if initial_capital > 0 else 0.0
    return {
        'trade_count': trade_count,
        'win_rate': float(win_rate) * 100.0,
        'expected_value': float(expected_value),
        'payoff_ratio': float(payoff_ratio),
        'is_candidate': bool(is_candidate),
        'asset_growth_pct': float(asset_growth_pct),
        'current_capital': float(current_capital),
        'max_drawdown': float(overall_max_drawdown),
    }
