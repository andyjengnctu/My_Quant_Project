from core.data_utils import get_required_min_rows_from_lookbacks
from strategies.breakout.adapter import build_breakout_strategy_params
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS


BREAKOUT_OPTIMIZER_SEARCH_SPACE = {
    "use_breakout_buy": {"kind": "categorical", "choices": [True]},
    "use_breakout_reclaim_reentry": {"kind": "categorical", "choices": [True, False]},  # (AI註: 停損後 reclaim re-entry 開關搜尋)
    "use_bb": {"kind": "categorical", "choices": [True, False]},  # (AI註: 布林通道濾網開關搜尋)
    "use_kc": {"kind": "categorical", "choices": [True, False],},  # (AI註: 肯特納通道濾網開關搜尋)
    "use_vol": {"kind": "categorical", "choices": [True, False]},  # (AI註: 突破日放量濾網開關搜尋)
    "use_breakout_return_filter": {"kind": "categorical", "choices": [True, False]},  # (AI註: 突破日漲幅濾網開關搜尋)
    "use_breakout_false_filter": {"kind": "categorical", "choices": [False]},  # (AI註: 假突破濾網開關搜尋)
    "use_breakout_ema_filter": {"kind": "categorical", "choices": [True, False]},  # (AI註: 突破 EMA 濾網開關搜尋)
    "use_breakout_quality_filter": {"kind": "categorical", "choices": [False]},  # (AI註: breakout quality filter 開關；score table 建好後可手動改成 [True, False])
    "use_history_threshold": {"kind": "categorical", "choices": [False]},  # (AI註: 歷史門檻開關搜尋)
    "high_len": {"kind": "int", "low": 60, "high": 350, "step": 5},  # (AI註: 突破新高觀察窗長搜尋，預設區間 100~300、步長 5)
    "breakout_ema_len": {"kind": "int", "low": 60, "high": 350, "step": 5, "enabled_by": "use_breakout_ema_filter"},  # (AI註: 突破 EMA 濾網長度搜尋)
    "atr_len": {"kind": "int", "low": 3, "high": 30},  # (AI註: ATR 窗長搜尋範圍，預設區間 3~25)
    "atr_times_init": {"kind": "float", "low": 1.0, "high": 4.5, "step": 0.1},  # (AI註: 初始停損 ATR 倍數搜尋，預設區間 1.0~4.5)
    "atr_times_trail": {"kind": "float", "low": 1.0, "high": 4.5, "step": 0.1},  # (AI註: 移動停損 ATR 倍數搜尋，預設區間 1.0~4.5)
    "atr_buy_tol": {"kind": "float", "low": 1.0, "high": 4.5, "step": 0.1},  # (AI註: 買點容忍 ATR 倍數搜尋，預設區間 1.0~4.5)
    "bb_len": {"kind": "int", "low": 3, "high": 30, "step": 1, "enabled_by": "use_bb"},  # (AI註: 布林通道長度搜尋，僅 use_bb=True 啟用)
    "bb_mult": {"kind": "float", "low": 1.0, "high": 3.0, "step": 0.1, "enabled_by": "use_bb"},  # (AI註: 布林通道倍數搜尋，僅 use_bb=True 啟用)
    "kc_len": {"kind": "int", "low": 3, "high": 30, "step": 1, "enabled_by": "use_kc"},  # (AI註: 肯特納通道長度搜尋，僅 use_kc=True 啟用)
    "kc_mult": {"kind": "float", "low": 1.0, "high": 3.0, "step": 0.1, "enabled_by": "use_kc"},  # (AI註: 肯特納通道倍數搜尋，僅 use_kc=True 啟用)
    "vol_long_len": {"kind": "int", "low": 5, "high": 30, "step": 1, "enabled_by": "use_vol"},  # (AI註: 突破日前均量窗長搜尋，僅 use_vol=True 啟用)
    "vol_breakout_mult": {"kind": "float", "low": 1.0, "high": 3.0, "step": 0.1, "enabled_by": "use_vol"},  # (AI註: 突破日量相對前均量倍數搜尋，僅 use_vol=True 啟用)
    "breakout_return_min": {"kind": "float", "low": 0.0, "high": 0.08, "step": 0.005, "enabled_by": "use_breakout_return_filter"},  # (AI註: 突破日收盤相對前收漲幅門檻搜尋)
    "breakout_false_filter_atr_pct_min": {"kind": "float", "low": 0.05, "high": 0.4, "step": 0.05, "enabled_by": "use_breakout_false_filter"},  # (AI註: 假突破濾網 ATR/Close 下限搜尋)
    "breakout_reclaim_window_bars": {"kind": "int", "low": 5, "high": 60, "step": 5, "enabled_by": "use_breakout_reclaim_reentry"},  # (AI註: re-entry 觀察窗搜尋)
    "breakout_reclaim_confirm_atr": {"kind": "float", "low": 0.1, "high": 2.5, "step": 0.1, "enabled_by": "use_breakout_reclaim_reentry"},  # (AI註: re-entry 重新站回本次 STOP line + N ATR 門檻搜尋；需與 schema > 0 契約一致)
    "min_history_trades": {"kind": "int", "low": 5, "high": 5, "enabled_by": "use_history_threshold"},  # (AI註: 歷史績效最少交易次數門檻搜尋)
    "min_history_ev": {"kind": "float", "low": -1.0, "high": 0.5, "step": 0.1, "enabled_by": "use_history_threshold"},  # (AI註: 歷史績效最小期望值門檻搜尋)
    "min_history_win_rate": {"kind": "float", "low": 0.0, "high": 0.75, "step": 0.05, "enabled_by": "use_history_threshold"},  # (AI註: 歷史績效最小勝率門檻搜尋)
}


def _resolve_optimizer_categorical_choices(field_name):
    choices = BREAKOUT_OPTIMIZER_SEARCH_SPACE[field_name]["choices"]
    if isinstance(choices, bool):
        return [choices]
    normalized = list(choices)
    if not normalized:
        raise ValueError(f"參數 {field_name} 的 choices 不可為空")
    return normalized


def _suggest_optimizer_switch(trial, field_name):
    return bool(trial.suggest_categorical(field_name, _resolve_optimizer_categorical_choices(field_name)))


def _suggest_optimizer_int(trial, field_name):
    spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE[field_name]
    return trial.suggest_int(
        field_name,
        int(spec["low"]),
        int(spec["high"]),
        step=int(spec.get("step", 1)),
    )


def _suggest_optimizer_float(trial, field_name):
    spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE[field_name]
    return trial.suggest_float(
        field_name,
        float(spec["low"]),
        float(spec["high"]),
        step=spec.get("step"),
    )


def build_trial_params(session, trial):
    ai_use_breakout_buy = _suggest_optimizer_switch(trial, "use_breakout_buy")
    ai_use_bb = _suggest_optimizer_switch(trial, "use_bb")
    ai_use_kc = _suggest_optimizer_switch(trial, "use_kc")
    ai_use_vol = _suggest_optimizer_switch(trial, "use_vol")
    ai_use_breakout_return_filter = _suggest_optimizer_switch(trial, "use_breakout_return_filter")
    ai_use_breakout_ema_filter = _suggest_optimizer_switch(trial, "use_breakout_ema_filter")
    ai_use_breakout_false_filter = _suggest_optimizer_switch(trial, "use_breakout_false_filter")
    ai_use_breakout_quality_filter = _suggest_optimizer_switch(trial, "use_breakout_quality_filter")
    ai_use_breakout_reclaim_reentry = _suggest_optimizer_switch(trial, "use_breakout_reclaim_reentry")
    ai_use_history_threshold = _suggest_optimizer_switch(trial, "use_history_threshold")

    if ai_use_vol:
        vol_long_len = _suggest_optimizer_int(trial, "vol_long_len")
        vol_breakout_mult = _suggest_optimizer_float(trial, "vol_breakout_mult")
    else:
        vol_long_len = BREAKOUT_PARAM_SPECS["vol_long_len"]["default"]
        vol_breakout_mult = BREAKOUT_PARAM_SPECS["vol_breakout_mult"]["default"]

    breakout_return_min = (
        _suggest_optimizer_float(trial, "breakout_return_min")
        if ai_use_breakout_return_filter
        else BREAKOUT_PARAM_SPECS["breakout_return_min"]["default"]
    )

    breakout_ema_len = (
        _suggest_optimizer_int(trial, "breakout_ema_len")
        if ai_use_breakout_ema_filter
        else BREAKOUT_PARAM_SPECS["breakout_ema_len"]["default"]
    )

    breakout_false_filter_atr_pct_min = (
        _suggest_optimizer_float(trial, "breakout_false_filter_atr_pct_min")
        if ai_use_breakout_false_filter
        else BREAKOUT_PARAM_SPECS["breakout_false_filter_atr_pct_min"]["default"]
    )
    breakout_reclaim_window_bars = (
        _suggest_optimizer_int(trial, "breakout_reclaim_window_bars")
        if ai_use_breakout_reclaim_reentry
        else BREAKOUT_PARAM_SPECS["breakout_reclaim_window_bars"]["default"]
    )
    breakout_reclaim_confirm_atr = (
        _suggest_optimizer_float(trial, "breakout_reclaim_confirm_atr")
        if ai_use_breakout_reclaim_reentry
        else BREAKOUT_PARAM_SPECS["breakout_reclaim_confirm_atr"]["default"]
    )
    min_history_trades = _suggest_optimizer_int(trial, "min_history_trades") if ai_use_history_threshold else 0
    min_history_ev = _suggest_optimizer_float(trial, "min_history_ev") if ai_use_history_threshold else -1.0
    min_history_win_rate = _suggest_optimizer_float(trial, "min_history_win_rate") if ai_use_history_threshold else 0.0

    return build_breakout_strategy_params(
        atr_len=_suggest_optimizer_int(trial, "atr_len"),
        atr_times_init=_suggest_optimizer_float(trial, "atr_times_init"),
        atr_times_trail=_suggest_optimizer_float(trial, "atr_times_trail"),
        atr_buy_tol=_suggest_optimizer_float(trial, "atr_buy_tol"),
        use_breakout_buy=ai_use_breakout_buy,
        high_len=_suggest_optimizer_int(trial, "high_len"),
        use_breakout_ema_filter=ai_use_breakout_ema_filter,
        breakout_ema_len=breakout_ema_len,
        tp_percent=session.resolve_optimizer_tp_percent(trial, fixed_tp_percent=session.optimizer_fixed_tp_percent),
        use_bb=ai_use_bb,
        use_kc=ai_use_kc,
        use_vol=ai_use_vol,
        use_breakout_return_filter=ai_use_breakout_return_filter,
        use_breakout_false_filter=ai_use_breakout_false_filter,
        breakout_false_filter_atr_pct_min=breakout_false_filter_atr_pct_min,
        use_breakout_quality_filter=ai_use_breakout_quality_filter,
        breakout_quality_filter_id=BREAKOUT_PARAM_SPECS["breakout_quality_filter_id"]["default"],
        use_breakout_reclaim_reentry=ai_use_breakout_reclaim_reentry,
        breakout_reclaim_window_bars=breakout_reclaim_window_bars,
        breakout_reclaim_confirm_atr=breakout_reclaim_confirm_atr,
        use_history_threshold=ai_use_history_threshold,
        bb_len=(
            _suggest_optimizer_int(trial, "bb_len")
            if ai_use_bb
            else BREAKOUT_PARAM_SPECS["bb_len"]["default"]
        ),
        bb_mult=(
            _suggest_optimizer_float(trial, "bb_mult")
            if ai_use_bb
            else BREAKOUT_PARAM_SPECS["bb_mult"]["default"]
        ),
        kc_len=(
            _suggest_optimizer_int(trial, "kc_len")
            if ai_use_kc
            else BREAKOUT_PARAM_SPECS["kc_len"]["default"]
        ),
        kc_mult=(
            _suggest_optimizer_float(trial, "kc_mult")
            if ai_use_kc
            else BREAKOUT_PARAM_SPECS["kc_mult"]["default"]
        ),
        vol_long_len=vol_long_len,
        vol_breakout_mult=vol_breakout_mult,
        breakout_return_min=breakout_return_min,
        min_history_trades=min_history_trades,
        min_history_ev=min_history_ev,
        min_history_win_rate=min_history_win_rate,
        use_compounding=True,
    )


BREAKOUT_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELDS = frozenset({
    "high_len",
    "use_breakout_ema_filter",
    "breakout_ema_len",
    "atr_len",
    "atr_times_trail",
    "atr_buy_tol",
    "bb_len",
    "bb_mult",
    "kc_len",
    "kc_mult",
    "vol_long_len",
    "vol_breakout_mult",
    "use_breakout_return_filter",
    "breakout_return_min",
    "use_breakout_false_filter",
    "breakout_false_filter_atr_pct_min",
})

BREAKOUT_LOCAL_MIN_PORTFOLIO_DEPENDENCY_FIELDS = frozenset({
    "atr_times_init",
    "tp_percent",
    "use_breakout_reclaim_reentry",
    "breakout_reclaim_window_bars",
    "breakout_reclaim_confirm_atr",
})


def classify_breakout_local_min_dependency_layer(field_name: str) -> str:
    field = str(field_name or "")
    if field in BREAKOUT_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELDS:
        return "signal"
    if field in BREAKOUT_LOCAL_MIN_PORTFOLIO_DEPENDENCY_FIELDS:
        return "portfolio"
    return "unknown"


def get_breakout_local_min_candidate_fields(trial, *, center_payload):
    candidate_fields = [
        "atr_len",
        "atr_times_init",
        "atr_times_trail",
        "atr_buy_tol",
    ]
    if bool(center_payload.get("use_breakout_buy", True)):
        candidate_fields.append("high_len")
        if bool(center_payload.get("use_breakout_ema_filter", True)):
            candidate_fields.append("breakout_ema_len")
    if "tp_percent" in getattr(trial, "params", {}):
        candidate_fields.append("tp_percent")
    if bool(center_payload.get("use_bb", False)):
        candidate_fields.extend(("bb_len", "bb_mult"))
    if bool(center_payload.get("use_kc", False)):
        candidate_fields.extend(("kc_len", "kc_mult"))
    if bool(center_payload.get("use_vol", False)):
        candidate_fields.extend(("vol_long_len", "vol_breakout_mult"))
    if bool(center_payload.get("use_breakout_return_filter", False)):
        candidate_fields.append("breakout_return_min")
    if bool(center_payload.get("use_breakout_false_filter", False)):
        candidate_fields.append("breakout_false_filter_atr_pct_min")
    if bool(center_payload.get("use_breakout_reclaim_reentry", False)):
        candidate_fields.extend(("breakout_reclaim_window_bars", "breakout_reclaim_confirm_atr"))
    return tuple(candidate_fields)


def resolve_breakout_neighbor_spec(field_name, *, center_payload=None):
    spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE[field_name]
    kind = str(spec["kind"])
    step = spec.get("step", 1)
    low_value = spec["low"]
    high_value = spec["high"]
    return step, kind, low_value, high_value


def get_breakout_optimizer_required_min_rows():
    return get_required_min_rows_from_lookbacks(
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["breakout_ema_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_long_len"]["high"],
    )
