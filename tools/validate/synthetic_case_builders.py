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
    set_synthetic_bar(df_weak, 20, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df_weak, 21, open_price=103.8, high_price=105.0, low_price=103.4, close_price=104.2)
    set_synthetic_bar(df_weak, 22, open_price=102.5, high_price=103.0, low_price=100.5, close_price=101.5)
    set_synthetic_bar(df_weak, 23, open_price=101.4, high_price=101.9, low_price=101.1, close_price=101.6)

    set_synthetic_bar(df_weak, 70, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df_weak, 71, open_price=103.8, high_price=105.0, low_price=103.4, close_price=104.2)
    for idx in range(72, len(df_weak)):
        set_synthetic_bar(df_weak, idx, open_price=104.2, high_price=104.5, low_price=103.9, close_price=104.2)

    df_strong = build_synthetic_baseline_frame("2024-01-01", 140)
    winning_bars = {
        20: (103.0, 104.5, 102.8, 104.0),
        21: (103.8, 105.0, 103.4, 104.2),
        22: (104.05535005017578, 105.7658272705768, 103.9476570515015, 105.31105462881263),
        23: (105.19304226445999, 105.8322019147044, 104.84077547539177, 105.58924143223415),
        24: (105.59205354572458, 106.13076304128805, 105.04437370111694, 105.78073381761503),
        25: (105.63103762243249, 106.23421634010937, 105.1143070306091, 105.4853284286143),
        26: (105.72662799887804, 107.34936535415748, 104.99919808701634, 106.7596940685349),
        27: (106.87008442768416, 107.19759311749098, 105.99576001905243, 106.53506299183323),
        28: (106.6015951758995, 106.77584050530604, 106.27455989056097, 106.71541978046501),
        29: (106.934605737128, 108.57791123422221, 106.42921391453338, 107.99794741355976),
        30: (107.70637243365817, 108.48096410976757, 107.3199586072383, 107.64917818953965),
        31: (107.75007011027877, 108.6877340308841, 107.15427120325072, 108.44843990554953),
        32: (108.29478643168181, 108.74458662542872, 106.93073528853994, 107.45129695383561),
        33: (107.491803398208, 107.9735796931008, 107.14966760580265, 107.26430786070408),
        34: (107.23308960356542, 107.84483201087816, 106.3789400006742, 106.86084768224212),
        35: (107.12054797677827, 107.05288044504564, 105.75719784311111, 106.06196222862346),
    }
    for idx, (o, h, l, c) in winning_bars.items():
        set_synthetic_bar(df_strong, idx, open_price=o, high_price=h, low_price=l, close_price=c)

    for idx in range(36, 100):
        set_synthetic_bar(df_strong, idx, open_price=100.0, high_price=100.4, low_price=99.6, close_price=100.0)

    set_synthetic_bar(df_strong, 100, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df_strong, 101, open_price=103.8, high_price=104.0, low_price=103.7, close_price=103.9, volume=0)
    set_synthetic_bar(df_strong, 102, open_price=103.9, high_price=104.1, low_price=103.8, close_price=104.0)
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
