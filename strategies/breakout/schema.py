"""breakout 策略專屬參數契約。"""

BREAKOUT_PARAM_SPECS = {
    "high_len": {"type": int, "default": 201, "min_value": 1},  # (AI註: 突破新高觀察窗長度，預設 201)
    "atr_len": {"type": int, "default": 14, "min_value": 1},  # (AI註: ATR 計算窗長，預設 14)
    "atr_buy_tol": {"type": float, "default": 1.5, "min_value": 0.0},  # (AI註: 買點容忍 ATR 倍數，預設 1.5)
    "atr_times_init": {"type": float, "default": 2.0, "min_value": 0.0, "strict_gt": True},  # (AI註: 初始停損 ATR 倍數，預設 2.0)
    "atr_times_trail": {"type": float, "default": 3.5, "min_value": 0.0, "strict_gt": True},  # (AI註: 移動停損 ATR 倍數，預設 3.5)
    "tp_percent": {"type": float, "default": 0.5, "min_value": 0.0, "max_value": 1.0, "max_exclusive": True},  # (AI註: 半倉停利比例，預設 0.5)
    "use_bb": {"type": bool, "default": True},  # (AI註: 是否啟用布林通道濾網，預設 True)
    "bb_len": {"type": int, "default": 20, "min_value": 1},  # (AI註: 布林通道長度，預設 20)
    "bb_mult": {"type": float, "default": 2.0, "min_value": 0.0, "strict_gt": True},  # (AI註: 布林通道倍數，預設 2.0)
    "use_kc": {"type": bool, "default": False},  # (AI註: 是否啟用肯特納通道濾網，預設 False)
    "kc_len": {"type": int, "default": 20, "min_value": 1},  # (AI註: 肯特納通道長度，預設 20)
    "kc_mult": {"type": float, "default": 2.0, "min_value": 0.0, "strict_gt": True},  # (AI註: 肯特納通道倍數，預設 2.0)
    "use_vol": {"type": bool, "default": True},  # (AI註: 是否啟用量能濾網，預設 True)
    "vol_short_len": {"type": int, "default": 5, "min_value": 1},  # (AI註: 短期量能窗長，預設 5)
    "vol_long_len": {"type": int, "default": 19, "min_value": 1},  # (AI註: 長期量能窗長，預設 19)
}


def validate_breakout_param_ranges(param_values, *, build_rule_text):
    for field_name, spec in BREAKOUT_PARAM_SPECS.items():
        value = param_values[field_name]
        min_value = spec.get("min_value")
        max_value = spec.get("max_value")
        strict_gt = spec.get("strict_gt", False)
        max_exclusive = spec.get("max_exclusive", False)

        if min_value is not None:
            condition = value > min_value if strict_gt else value >= min_value
            if not condition:
                raise ValueError(f"參數 {field_name} 驗證失敗: {build_rule_text(spec)}，收到 {value!r}")

        if max_value is not None:
            condition = value < max_value if max_exclusive else value <= max_value
            if not condition:
                raise ValueError(f"參數 {field_name} 驗證失敗: {build_rule_text(spec)}，收到 {value!r}")

    if param_values["vol_long_len"] < param_values["vol_short_len"]:
        raise ValueError(
            f"參數 vol_long_len 驗證失敗: 需 >= vol_short_len，收到 {param_values['vol_long_len']!r} < {param_values['vol_short_len']!r}"
        )
    return param_values
