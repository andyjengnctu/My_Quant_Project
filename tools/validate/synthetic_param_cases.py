from core.v16_config import V16StrategyParams
from core.v16_core import (
    build_cash_capped_entry_plan,
    calc_entry_price,
    calc_reference_candidate_qty,
    evaluate_history_candidate_metrics,
    resize_candidate_plan_to_capital,
    run_v16_backtest,
)
from core.v16_params_io import build_params_from_mapping, params_to_json_dict

from .checks import (
    add_check,
    add_fail_result,
    make_consistency_params,
    make_synthetic_validation_params,
)
from .synthetic_fixtures import build_synthetic_param_guardrail_case as fixture_build_synthetic_param_guardrail_case


def build_synthetic_param_guardrail_case(base_params):
    return fixture_build_synthetic_param_guardrail_case(base_params, lambda p: params_to_json_dict(make_consistency_params(p)))


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

def validate_synthetic_param_guardrail_case(base_params):
    case = build_synthetic_param_guardrail_case(base_params)
    results = []
    summary = {"ticker": case["case_id"], "synthetic": True}

    valid_params = build_params_from_mapping(case["base_payload"])
    add_check(results, "synthetic_param_guardrail", case["case_id"], "valid_payload_loads", True, isinstance(valid_params, V16StrategyParams))

    invalid_cases = [
        ("tp_percent_ge_1_rejected", {**case["base_payload"], "tp_percent": 1.0}, "tp_percent"),
        ("fixed_risk_zero_rejected", {**case["base_payload"], "fixed_risk": 0.0}, "fixed_risk"),
        ("min_history_win_rate_gt_1_rejected", {**case["base_payload"], "min_history_win_rate": 1.1}, "min_history_win_rate"),
        ("vol_long_len_lt_short_rejected", {**case["base_payload"], "vol_short_len": 10, "vol_long_len": 5}, "vol_long_len"),
        ("use_bb_string_type_rejected", {**case["base_payload"], "use_bb": "abc"}, "use_bb"),
    ]

    runtime_valid_payload = {**case["base_payload"], "optimizer_max_workers": 3, "scanner_max_workers": 4}
    runtime_params = build_params_from_mapping(runtime_valid_payload)
    add_check(results, "synthetic_param_guardrail", case["case_id"], "runtime_optimizer_max_workers_loads", 3, getattr(runtime_params, "optimizer_max_workers", None))
    add_check(results, "synthetic_param_guardrail", case["case_id"], "runtime_scanner_max_workers_loads", 4, getattr(runtime_params, "scanner_max_workers", None))

    runtime_invalid_cases = [
        ("optimizer_max_workers_zero_rejected", {**case["base_payload"], "optimizer_max_workers": 0}, "optimizer_max_workers"),
        ("scanner_max_workers_zero_rejected", {**case["base_payload"], "scanner_max_workers": 0}, "scanner_max_workers"),
    ]

    for metric_name, payload, expected_field in invalid_cases + runtime_invalid_cases:
        try:
            build_params_from_mapping(payload)
            add_fail_result(
                results,
                "synthetic_param_guardrail",
                case["case_id"],
                metric_name,
                f"ValueError containing {expected_field}",
                "loaded_ok",
                "非法參數不應成功載入。"
            )
        except ValueError as e:
            add_check(results, "synthetic_param_guardrail", case["case_id"], metric_name, True, expected_field in str(e))

    for metric_name, payload, expected_field in invalid_cases:
        try:
            V16StrategyParams(**payload)
            add_fail_result(
                results,
                "synthetic_param_guardrail",
                case["case_id"],
                f"direct_dataclass_{metric_name}",
                f"ValueError containing {expected_field}",
                "loaded_ok",
                "直接建立 V16StrategyParams 也不應繞過 guardrail。"
            )
        except ValueError as e:
            add_check(
                results,
                "synthetic_param_guardrail",
                case["case_id"],
                f"direct_dataclass_{metric_name}",
                True,
                expected_field in str(e)
            )

    runtime_mutation_params = V16StrategyParams()
    try:
        runtime_mutation_params.optimizer_max_workers = 0
        add_fail_result(
            results,
            "synthetic_param_guardrail",
            case["case_id"],
            "direct_runtime_attr_guardrail",
            "ValueError containing optimizer_max_workers",
            "setattr_ok",
            "runtime worker 設定直接改欄位也不應繞過 guardrail。"
        )
    except ValueError as e:
        add_check(results, "synthetic_param_guardrail", case["case_id"], "direct_runtime_attr_guardrail", True, "optimizer_max_workers" in str(e))

    invalid_direct_setattr_cases = [
        ("direct_setattr_use_bb_string_rejected", "use_bb", "abc", "use_bb"),
        ("direct_setattr_high_len_string_rejected", "high_len", "10", "high_len"),
    ]

    for metric_name, field_name, invalid_value, expected_field in invalid_direct_setattr_cases:
        mutation_target = V16StrategyParams()
        try:
            setattr(mutation_target, field_name, invalid_value)
            add_fail_result(
                results,
                "synthetic_param_guardrail",
                case["case_id"],
                metric_name,
                f"ValueError containing {expected_field}",
                "setattr_ok",
                "直接改 dataclass 欄位不應繞過型別 guardrail。"
            )
        except ValueError as e:
            add_check(results, "synthetic_param_guardrail", case["case_id"], metric_name, True, expected_field in str(e))

    mutation_params = V16StrategyParams()
    try:
        mutation_params.tp_precent = 0.3
        add_fail_result(
            results,
            "synthetic_param_guardrail",
            case["case_id"],
            "unknown_attr_typo_rejected",
            "AttributeError containing tp_precent",
            "setattr_ok",
            "未知屬性 typo 不應靜默掛到 params 物件上。"
        )
    except AttributeError as e:
        add_check(results, "synthetic_param_guardrail", case["case_id"], "unknown_attr_typo_rejected", True, "tp_precent" in str(e))

    default_params_arg = run_v16_backtest.__defaults__[0] if run_v16_backtest.__defaults__ else None
    add_check(results, "synthetic_param_guardrail", case["case_id"], "run_v16_backtest_default_params_is_none", True, default_params_arg is None)

    summary["guardrail_cases"] = (len(invalid_cases) * 2) + len(runtime_invalid_cases) + len(invalid_direct_setattr_cases) + 4
    return results, summary
