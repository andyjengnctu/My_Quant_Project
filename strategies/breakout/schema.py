"""breakout 策略專屬參數契約。"""

from config.breakout_policy import BREAKOUT_DEFAULT_HIGH_LEN
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
)

BREAKOUT_PARAM_SPECS = {
    "use_breakout_buy": {"type": bool, "default": True},
    "high_len": {"type": int, "default": BREAKOUT_DEFAULT_HIGH_LEN, "min_value": 1},  # (AI註: 突破新高觀察窗長度；預設由 config.breakout_policy 單一提供)
    "use_breakout_ema_filter": {"type": bool, "default": True},  # (AI註: 是否啟用突破 EMA 濾網，預設 True 以相容舊策略口徑)
    "breakout_ema_len": {"type": int, "default": 240, "min_value": 1},  # (AI註: 突破 EMA 濾網長度，僅 use_breakout_ema_filter=True 時套用)
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
    "use_vol": {"type": bool, "default": True},  # (AI註: 是否啟用突破日放量濾網，預設 True)
    "vol_short_len": {"type": int, "default": 5, "min_value": 1},  # (AI註: 舊版量能欄位，僅保留 JSON 相容；正式訊號不再使用)
    "vol_long_len": {"type": int, "default": 20, "min_value": 1},  # (AI註: 突破日前均量窗長，預設 20)
    "vol_breakout_mult": {"type": float, "default": 1.5, "min_value": 0.0, "strict_gt": True},  # (AI註: 突破日量需大於前均量的倍數，預設 1.5)
    "use_breakout_return_filter": {"type": bool, "default": False},  # (AI註: 是否啟用突破日漲幅濾網，預設 False 以相容舊模型)
    "breakout_return_min": {"type": float, "default": 0.0, "min_value": 0.0},  # (AI註: 突破日收盤相對前收漲幅門檻，0.03 代表 3%)
    "use_breakout_false_filter": {"type": bool, "default": False},  # (AI註: 是否啟用假突破濾網，預設 False 以相容舊模型)
    "breakout_false_filter_atr_pct_min": {"type": float, "default": 0.045, "min_value": 0.0, "strict_gt": True},  # (AI註: 假突破濾網個股 ATR/Close 下限)
    "use_breakout_quality_filter": {"type": bool, "default": False},  # (AI註: 是否啟用 breakout quality pass/reject score table 濾網，預設 False 以相容舊模型)
    "breakout_quality_filter_id": {"type": str, "default": BREAKOUT_QUALITY_DEFAULT_FILTER_ID},  # (AI註: breakout quality filter 模型/score table ID)
    "breakout_quality_score_threshold": {"type": float, "default": BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD, "min_value": 0.0, "max_value": 1.0},  # (AI註: 正式 runtime 以 dl_quality_score >= threshold 作為唯一通過判斷)
    "use_breakout_reclaim_reentry": {"type": bool, "default": False},  # (AI註: 是否啟用突破停損後 reclaim re-entry，預設 False 以相容舊模型)
    "breakout_reclaim_window_bars": {"type": int, "default": 20, "min_value": 1},  # (AI註: 停損後可觸發 re-entry 的觀察交易日數)
    "breakout_reclaim_confirm_atr": {"type": float, "default": 0.8, "min_value": 0.0, "strict_gt": True},  # (AI註: Close 重新站回本次 STOP line + N ATR 才產生 re-entry candidate)
}


def validate_breakout_param_ranges(param_values, *, build_rule_text):
    filter_id = str(param_values["breakout_quality_filter_id"]).strip()
    if not filter_id:
        raise ValueError("參數 breakout_quality_filter_id 不可為空白")

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

    return param_values
