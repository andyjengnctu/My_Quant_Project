from core.backtest_core import (
    build_cash_capped_entry_plan,
    calc_entry_price,
    resize_candidate_plan_to_capital,
    evaluate_history_candidate_metrics,
)

from .checks import add_check, make_synthetic_validation_params


def validate_synthetic_history_ev_threshold_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.min_history_trades = 1
    params.min_history_ev = 0.5
    params.min_history_win_rate = 0.5

    case_id = "SYNTH_HISTORY_EV_THRESHOLD"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    is_candidate, expected_value, win_rate, trade_count = evaluate_history_candidate_metrics(
        trade_count=1,
        win_count=1,
        total_r_sum=0.5,
        win_r_sum=0.5,
        loss_r_sum=0.0,
        params=params,
    )

    add_check(results, "synthetic_history_ev_threshold", case_id, "expected_value_equals_threshold", 0.5, expected_value)
    add_check(results, "synthetic_history_ev_threshold", case_id, "win_rate_equals_threshold", 1.0, win_rate)
    add_check(results, "synthetic_history_ev_threshold", case_id, "trade_count_preserved", 1, trade_count)
    add_check(results, "synthetic_history_ev_threshold", case_id, "candidate_allowed_when_ev_equals_threshold", True, is_candidate)

    summary["candidate_allowed"] = bool(is_candidate)
    summary["expected_value"] = expected_value
    return results, summary



def validate_synthetic_proj_cost_cash_capped_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.initial_capital = 1_000_000.0
    params.fixed_risk = 0.01

    case_id = "SYNTH_PROJ_COST_CASH_CAPPED_ORDER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    sizing_equity = 1_000_000.0
    available_cash = 50_000.0
    candidate_rows = [
        {"ticker": "9801", "limit_px": 10.0, "init_sl": 9.5, "init_trail": 9.0},
        {"ticker": "9802", "limit_px": 14.0, "init_sl": 13.3, "init_trail": 12.8},
    ]

    estimated_rank_rows = []
    for cand in candidate_rows:
        est_plan = resize_candidate_plan_to_capital(
            {
                "limit_price": cand["limit_px"],
                "init_sl": cand["init_sl"],
                "init_trail": cand["init_trail"],
            },
            sizing_equity,
            params,
        )
        est_reserved_cost = calc_entry_price(est_plan["limit_price"], est_plan["qty"], params) * est_plan["qty"]
        estimated_rank_rows.append({
            "ticker": cand["ticker"],
            "reserved_cost": est_reserved_cost,
        })

    estimated_rank_rows.sort(key=lambda x: (x["reserved_cost"], x["ticker"]), reverse=True)
    stale_top_ticker = estimated_rank_rows[0]["ticker"]

    cash_capped_rank_rows = []
    for cand in candidate_rows:
        cash_capped_plan = build_cash_capped_entry_plan(
            {
                "limit_price": cand["limit_px"],
                "init_sl": cand["init_sl"],
                "init_trail": cand["init_trail"],
            },
            available_cash,
            params,
        )
        if cash_capped_plan is None:
            continue

        cash_capped_rank_rows.append({
            "ticker": cand["ticker"],
            "reserved_cost": cash_capped_plan["reserved_cost"],
            "qty": cash_capped_plan["qty"],
        })

    cash_capped_rank_rows.sort(key=lambda x: (x["reserved_cost"], x["ticker"]), reverse=True)
    cash_capped_top_ticker = cash_capped_rank_rows[0]["ticker"] if cash_capped_rank_rows else None
    cash_capped_top_reserved_cost = cash_capped_rank_rows[0]["reserved_cost"] if cash_capped_rank_rows else None
    cash_capped_top_qty = cash_capped_rank_rows[0]["qty"] if cash_capped_rank_rows else None

    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "stale_proj_cost_top_ticker",
        "9802",
        stale_top_ticker,
    )
    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "cash_capped_proj_cost_top_ticker",
        "9801",
        cash_capped_top_ticker,
    )
    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "proj_cost_order_reversal_detected",
        True,
        stale_top_ticker != cash_capped_top_ticker,
    )
    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "cash_capped_reserved_cost_within_available_cash",
        True,
        cash_capped_top_reserved_cost is not None and cash_capped_top_reserved_cost <= available_cash,
    )
    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "cash_capped_qty_positive",
        True,
        cash_capped_top_qty is not None and cash_capped_top_qty > 0,
    )

    summary["stale_top_ticker"] = stale_top_ticker
    summary["cash_capped_top_ticker"] = cash_capped_top_ticker
    summary["available_cash"] = available_cash
    return results, summary
