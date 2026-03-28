from core.v16_config import V16StrategyParams
from core.v16_core import run_v16_backtest
from core.v16_params_io import build_params_from_mapping, params_to_json_dict

from .checks import (
    add_check,
    add_fail_result,
    make_consistency_params,
)
from .synthetic_fixtures import build_synthetic_param_guardrail_case as fixture_build_synthetic_param_guardrail_case


def build_synthetic_param_guardrail_case(base_params):
    return fixture_build_synthetic_param_guardrail_case(
        base_params,
        lambda p: params_to_json_dict(make_consistency_params(p)),
    )



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
