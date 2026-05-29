from .synthetic_frame_utils import build_synthetic_baseline_frame, set_synthetic_bar


def build_synthetic_half_tp_full_year_case(base_params, make_params):
    params = make_params(base_params, tp_percent=0.5)
    df = build_synthetic_baseline_frame("2024-01-01", 320)
    trigger_idx = 270
    set_synthetic_bar(df, trigger_idx, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df, trigger_idx + 1, open_price=103.8, high_price=105.0, low_price=103.4, close_price=104.2)
    set_synthetic_bar(df, trigger_idx + 2, open_price=104.3, high_price=107.5, low_price=104.0, close_price=106.5)
    set_synthetic_bar(df, trigger_idx + 3, open_price=106.5, high_price=107.0, low_price=106.1, close_price=106.8)
    set_synthetic_bar(df, trigger_idx + 4, open_price=106.6, high_price=107.1, low_price=106.2, close_price=106.9)
    for idx in range(trigger_idx + 5, len(df)):
        base_close = 106.8 + (idx - (trigger_idx + 5)) * 0.01
        set_synthetic_bar(df, idx, open_price=base_close - 0.2, high_price=base_close + 0.3, low_price=base_close - 0.4, close_price=base_close)
    return {
        "case_id": "SYNTH_HALF_TP_FULL_YEAR",
        "params": params,
        "frames": {"9201": df},
        "benchmark_ticker": "9201",
        "max_positions": 1,
        "enable_rotation": False,
        "start_year": 2024,
        "primary_ticker": "9201",
    }


def build_synthetic_extended_miss_buy_case(base_params, make_params):
    params = make_params(base_params, tp_percent=0.0)
    df = build_synthetic_baseline_frame("2024-01-01", 56)
    set_synthetic_bar(df, 54, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df, 55, open_price=103.8, high_price=104.0, low_price=103.6, close_price=103.9, volume=0)
    return {
        "case_id": "SYNTH_EXTENDED_MISS_BUY",
        "params": params,
        "frames": {"9301": df},
        "benchmark_ticker": "9301",
        "max_positions": 1,
        "enable_rotation": False,
        "start_year": 2024,
        "primary_ticker": "9301",
    }


def build_synthetic_competing_candidates_case(base_params, make_params):
    params = make_params(base_params, tp_percent=0.0)

    def build_frame():
        df = build_synthetic_baseline_frame("2024-01-01", 60)
        set_synthetic_bar(df, 55, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
        set_synthetic_bar(df, 56, open_price=103.8, high_price=105.0, low_price=103.4, close_price=104.2)
        set_synthetic_bar(df, 57, open_price=104.8, high_price=105.3, low_price=104.4, close_price=105.0)
        set_synthetic_bar(df, 58, open_price=105.3, high_price=105.8, low_price=105.0, close_price=105.5)
        set_synthetic_bar(df, 59, open_price=105.8, high_price=106.3, low_price=105.6, close_price=106.0)
        return df

    return {
        "case_id": "SYNTH_COMPETING_CANDIDATES",
        "params": params,
        "frames": {"9401": build_frame(), "9402": build_frame()},
        "benchmark_ticker": "9401",
        "max_positions": 1,
        "enable_rotation": False,
        "start_year": 2024,
        "primary_ticker": "9402",
    }


def build_synthetic_same_day_sell_block_case(base_params, make_params):
    params = make_params(base_params, tp_percent=0.0)

    df_a = build_synthetic_baseline_frame("2024-01-01", 60)
    set_synthetic_bar(df_a, 55, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df_a, 56, open_price=103.8, high_price=105.0, low_price=103.4, close_price=104.2)
    set_synthetic_bar(df_a, 57, open_price=102.5, high_price=103.0, low_price=100.5, close_price=101.5)
    set_synthetic_bar(df_a, 58, open_price=101.4, high_price=101.9, low_price=101.1, close_price=101.6)
    set_synthetic_bar(df_a, 59, open_price=101.5, high_price=102.0, low_price=101.2, close_price=101.7)

    df_b = build_synthetic_baseline_frame("2024-01-01", 60)
    set_synthetic_bar(df_b, 56, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df_b, 57, open_price=103.8, high_price=105.0, low_price=103.4, close_price=104.2)
    set_synthetic_bar(df_b, 58, open_price=104.0, high_price=104.4, low_price=103.7, close_price=103.9)
    set_synthetic_bar(df_b, 59, open_price=103.8, high_price=104.1, low_price=103.5, close_price=103.7)

    return {
        "case_id": "SYNTH_SAME_DAY_SELL_BLOCK",
        "params": params,
        "frames": {"9501": df_a, "9502": df_b},
        "benchmark_ticker": "9501",
        "max_positions": 1,
        "enable_rotation": False,
        "start_year": 2024,
        "primary_ticker": "9501",
    }


def build_synthetic_unexecutable_half_tp_case(base_params, make_params):
    params = make_params(base_params, tp_percent=0.5)
    params.initial_capital = 130.0
    params.fixed_risk = 1.0

    df = build_synthetic_baseline_frame("2024-01-01", 320)
    trigger_idx = 270
    set_synthetic_bar(df, trigger_idx, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df, trigger_idx + 1, open_price=103.8, high_price=105.0, low_price=103.4, close_price=104.2)
    set_synthetic_bar(df, trigger_idx + 2, open_price=104.3, high_price=107.5, low_price=104.0, close_price=106.5)
    set_synthetic_bar(df, trigger_idx + 3, open_price=106.5, high_price=107.0, low_price=106.1, close_price=106.8)
    set_synthetic_bar(df, trigger_idx + 4, open_price=106.6, high_price=107.1, low_price=106.2, close_price=106.9)
    for idx in range(trigger_idx + 5, len(df)):
        base_close = 106.8 + (idx - (trigger_idx + 5)) * 0.01
        set_synthetic_bar(df, idx, open_price=base_close - 0.2, high_price=base_close + 0.3, low_price=base_close - 0.4, close_price=base_close)

    return {
        "case_id": "SYNTH_UNEXECUTABLE_HALF_TP",
        "params": params,
        "frames": {"9601": df},
        "benchmark_ticker": "9601",
        "max_positions": 1,
        "enable_rotation": False,
        "start_year": 2024,
        "primary_ticker": "9601",
    }


def build_synthetic_rotation_t_plus_one_case(base_params, make_params):
    params = make_params(base_params, tp_percent=0.0)

    df_weak = build_synthetic_baseline_frame("2024-01-01", 140)
    for idx in range(20, 36):
        set_synthetic_bar(df_weak, idx, open_price=100.0, high_price=100.4, low_price=99.6, close_price=100.0)

    for idx in range(36, len(df_weak)):
        set_synthetic_bar(df_weak, idx, open_price=49.8, high_price=50.3, low_price=49.5, close_price=50.0)

    set_synthetic_bar(df_weak, 70, open_price=50.0, high_price=55.0, low_price=48.0, close_price=50.8)
    set_synthetic_bar(df_weak, 71, open_price=50.6, high_price=51.2, low_price=50.4, close_price=50.8)
    for idx in range(72, len(df_weak)):
        set_synthetic_bar(df_weak, idx, open_price=50.8, high_price=51.0, low_price=50.6, close_price=50.8)

    df_strong = build_synthetic_baseline_frame("2024-01-01", 140)
    for idx in range(20, 100):
        set_synthetic_bar(df_strong, idx, open_price=100.0, high_price=100.4, low_price=99.6, close_price=100.0)

    set_synthetic_bar(df_strong, 100, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df_strong, 101, open_price=103.8, high_price=104.0, low_price=103.7, close_price=103.9, volume=0)
    # # (AI註: rotation T+1 case 需要在汰弱賣出後的首個可評估交易日先產生一次延續候選 miss buy，
    # #        再於後一日成功延後買進；因此 day 102 必須保持 reachable 但 volume=0。)
    set_synthetic_bar(df_strong, 102, open_price=103.9, high_price=104.1, low_price=103.8, close_price=104.0, volume=0)
    set_synthetic_bar(df_strong, 103, open_price=104.2, high_price=104.6, low_price=104.0, close_price=104.4)
    set_synthetic_bar(df_strong, 104, open_price=104.3, high_price=104.7, low_price=104.1, close_price=104.4)
    for idx in range(105, len(df_strong)):
        set_synthetic_bar(df_strong, idx, open_price=104.4, high_price=104.7, low_price=104.2, close_price=104.4)

    return {
        "case_id": "SYNTH_ROTATION_T_PLUS_ONE",
        "params": params,
        "frames": {"9701": df_weak, "9702": df_strong},
        "benchmark_ticker": "9701",
        "max_positions": 1,
        "enable_rotation": True,
        "start_year": 2024,
        "weak_ticker": "9701",
        "strong_ticker": "9702",
    }


def build_synthetic_param_guardrail_case(base_params, params_to_payload):
    return {
        "case_id": "SYNTH_PARAM_GUARDRAIL",
        "base_payload": params_to_payload(base_params),
    }
