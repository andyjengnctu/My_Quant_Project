from core.config import get_ev_calc_method
from core.param_access import get_param_value


_get_param = get_param_value


def _history_threshold_payload_is_neutral(params) -> bool:
    try:
        min_trades = int(_get_param(params, "min_history_trades", 0))
        min_ev = float(_get_param(params, "min_history_ev", -1.0))
        min_win_rate = float(_get_param(params, "min_history_win_rate", 0.0))
    except (TypeError, ValueError):
        return False
    return min_trades == 0 and min_ev <= 0.0 and min_win_rate <= 0.0


def history_threshold_is_enabled(params) -> bool:
    explicit_value = _get_param(params, "use_history_threshold", None)
    if explicit_value is not None:
        return bool(explicit_value)
    return not _history_threshold_payload_is_neutral(params)


def evaluate_history_candidate_metrics(trade_count, win_count, total_r_sum, win_r_sum, loss_r_sum, params):
    use_history_threshold = history_threshold_is_enabled(params)
    min_trades_req = _get_param(params, "min_history_trades", 0)
    min_ev_req = _get_param(params, "min_history_ev", 0.0)
    min_win_rate_req = _get_param(params, "min_history_win_rate", 0.30)

    allow_zero_history = (
        not use_history_threshold
        or (
            (min_trades_req == 0)
            and (min_ev_req <= 0)
            and (min_win_rate_req <= 0)
        )
    )

    if use_history_threshold and trade_count < min_trades_req:
        return False, 0.0, 0.0, trade_count

    if trade_count == 0:
        return allow_zero_history, 0.0, 0.0, trade_count

    win_rate = win_count / trade_count

    if get_ev_calc_method() == "B":
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

    is_candidate = True
    if use_history_threshold:
        is_candidate = (win_rate >= min_win_rate_req) and (expected_value >= min_ev_req)
    return is_candidate, expected_value, win_rate, trade_count
