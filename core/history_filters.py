from core.config import EV_CALC_METHOD


def evaluate_history_candidate_metrics(trade_count, win_count, total_r_sum, win_r_sum, loss_r_sum, params):
    min_trades_req = getattr(params, "min_history_trades", 0)
    min_ev_req = getattr(params, "min_history_ev", 0.0)
    min_win_rate_req = getattr(params, "min_history_win_rate", 0.30)

    allow_zero_history = (
        (min_trades_req == 0)
        and (min_ev_req <= 0)
        and (min_win_rate_req <= 0)
    )

    if trade_count < min_trades_req:
        return False, 0.0, 0.0, trade_count

    if trade_count == 0:
        return allow_zero_history, 0.0, 0.0, trade_count

    win_rate = win_count / trade_count

    if EV_CALC_METHOD == "B":
        avg_win_r = (win_r_sum / win_count) if win_count > 0 else 0.0
        loss_count = trade_count - win_count
        avg_loss_r = abs(loss_r_sum / loss_count) if loss_count > 0 else 0.0

        if avg_loss_r > 0:
            payoff_for_ev = min(10.0, avg_win_r / avg_loss_r)
        elif avg_win_r > 0:
            payoff_for_ev = 99.9
        else:
            payoff_for_ev = 0.0

        expected_value = (win_rate * payoff_for_ev) - (1 - win_rate)
    else:
        expected_value = total_r_sum / trade_count

    is_candidate = (win_rate >= min_win_rate_req) and (expected_value >= min_ev_req)
    return is_candidate, expected_value, win_rate, trade_count
