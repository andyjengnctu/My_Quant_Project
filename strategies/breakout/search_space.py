from core.data_utils import get_required_min_rows_from_lookbacks
from strategies.breakout.adapter import build_breakout_strategy_params
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS


BREAKOUT_OPTIMIZER_SEARCH_SPACE = {
    "use_bb": {"kind": "categorical", "choices": [True, False]},  # (AI註: 布林通道濾網開關搜尋)
    "use_kc": {"kind": "categorical", "choices": [True, False],},  # (AI註: 肯特納通道濾網開關搜尋)
    "use_vol": {"kind": "categorical", "choices": [True, False]},  # (AI註: 量能濾網開關搜尋)
    "use_ema_pullback": {"kind": "categorical", "choices": [True]},
    "high_len": {"kind": "int", "low": 100, "high": 300, "step": 5},  # (AI註: 突破新高觀察窗長搜尋，預設區間 40~250、步長 5)
    "atr_len": {"kind": "int", "low": 3, "high": 25},  # (AI註: ATR 窗長搜尋範圍，預設區間 3~25)
    "atr_times_init": {"kind": "float", "low": 1.0, "high": 4.5, "step": 0.1},  # (AI註: 初始停損 ATR 倍數搜尋，預設區間 1.0~3.5)
    "atr_times_trail": {"kind": "float", "low": 1.0, "high": 4.5, "step": 0.1},  # (AI註: 移動停損 ATR 倍數搜尋，預設區間 2.0~4.5)
    "atr_buy_tol": {"kind": "float", "low": 1.0, "high": 4.5, "step": 0.1},  # (AI註: 買點容忍 ATR 倍數搜尋，預設區間 0.1~3.5)
    "bb_len": {"kind": "int", "low": 3, "high": 30, "step": 1, "enabled_by": "use_bb"},  # (AI註: 布林通道長度搜尋，僅 use_bb=True 啟用)
    "bb_mult": {"kind": "float", "low": 1.0, "high": 3.0, "step": 0.1, "enabled_by": "use_bb"},  # (AI註: 布林通道倍數搜尋，僅 use_bb=True 啟用)
    "kc_len": {"kind": "int", "low": 3, "high": 30, "step": 1, "enabled_by": "use_kc"},  # (AI註: 肯特納通道長度搜尋，僅 use_kc=True 啟用)
    "kc_mult": {"kind": "float", "low": 1.0, "high": 3.0, "step": 0.1, "enabled_by": "use_kc"},  # (AI註: 肯特納通道倍數搜尋，僅 use_kc=True 啟用)
    "vol_short_len": {"kind": "int", "low": 1, "high": 10, "enabled_by": "use_vol"},  # (AI註: 短期量能窗長搜尋，僅 use_vol=True 啟用)
    "vol_long_len": {"kind": "int", "high": 30, "depends_on": "vol_short_len", "enabled_by": "use_vol"},  # (AI註: 長期量能窗長搜尋，僅 use_vol=True 啟用且下限跟隨 vol_short_len)
    "ema_pullback_short_len": {"kind": "int", "low": 3, "high": 20, "enabled_by": "use_ema_pullback"},
    "ema_pullback_mid_len": {"kind": "int", "low": 10, "high": 60, "enabled_by": "use_ema_pullback"},
    "ema_pullback_long_len": {"kind": "int", "low": 80, "high": 240, "enabled_by": "use_ema_pullback"},
    "min_history_trades": {"kind": "int", "low": 3, "high": 3},  # (AI註: 歷史績效最少交易次數搜尋，預設區間 0~5)
    "min_history_ev": {"kind": "float", "low": 0.0, "high": 0.0, "step": 0.1},  # (AI註: 歷史績效最小期望值搜尋，預設區間 -1.0~0.5)
    "min_history_win_rate": {"kind": "float", "low": 0.45, "high": 0.45, "step": 0.05},  # (AI註: 歷史績效最小勝率搜尋，預設區間 0.0~0.6)
}


def _suggest_ema_pullback_lengths(trial):
    if not trial.suggest_categorical("use_ema_pullback", BREAKOUT_OPTIMIZER_SEARCH_SPACE["use_ema_pullback"]["choices"]):
        return (
            False,
            BREAKOUT_PARAM_SPECS["ema_pullback_short_len"]["default"],
            BREAKOUT_PARAM_SPECS["ema_pullback_mid_len"]["default"],
            BREAKOUT_PARAM_SPECS["ema_pullback_long_len"]["default"],
        )

    short_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["ema_pullback_short_len"]
    mid_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["ema_pullback_mid_len"]
    long_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["ema_pullback_long_len"]
    short_len = trial.suggest_int("ema_pullback_short_len", short_spec["low"], short_spec["high"])
    mid_low = max(int(mid_spec["low"]), int(short_len) + 1)
    mid_len = trial.suggest_int("ema_pullback_mid_len", mid_low, int(mid_spec["high"]))
    long_low = max(int(long_spec["low"]), int(mid_len) + 1)
    long_len = trial.suggest_int("ema_pullback_long_len", long_low, int(long_spec["high"]))
    return True, short_len, mid_len, long_len


def build_trial_params(session, trial):
    ai_use_bb = trial.suggest_categorical("use_bb", BREAKOUT_OPTIMIZER_SEARCH_SPACE["use_bb"]["choices"])
    ai_use_kc = trial.suggest_categorical("use_kc", BREAKOUT_OPTIMIZER_SEARCH_SPACE["use_kc"]["choices"])
    ai_use_vol = trial.suggest_categorical("use_vol", BREAKOUT_OPTIMIZER_SEARCH_SPACE["use_vol"]["choices"])
    ai_use_ema_pullback, ema_short_len, ema_mid_len, ema_long_len = _suggest_ema_pullback_lengths(trial)

    if ai_use_vol:
        vol_short_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_short_len"]
        vol_short_len = trial.suggest_int("vol_short_len", vol_short_spec["low"], vol_short_spec["high"])
        vol_long_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_long_len"]
        vol_long_len = trial.suggest_int("vol_long_len", vol_short_len, vol_long_spec["high"])
    else:
        vol_short_len = BREAKOUT_PARAM_SPECS["vol_short_len"]["default"]
        vol_long_len = BREAKOUT_PARAM_SPECS["vol_long_len"]["default"]

    return build_breakout_strategy_params(
        atr_len=trial.suggest_int("atr_len", BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"]["high"]),
        atr_times_init=trial.suggest_float("atr_times_init", BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_init"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_init"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_init"]["step"]),
        atr_times_trail=trial.suggest_float("atr_times_trail", BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_trail"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_trail"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_trail"]["step"]),
        atr_buy_tol=trial.suggest_float("atr_buy_tol", BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_buy_tol"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_buy_tol"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_buy_tol"]["step"]),
        high_len=trial.suggest_int("high_len", BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]["step"]),
        tp_percent=session.resolve_optimizer_tp_percent(trial, fixed_tp_percent=session.optimizer_fixed_tp_percent),
        use_bb=ai_use_bb,
        use_kc=ai_use_kc,
        use_vol=ai_use_vol,
        use_ema_pullback=ai_use_ema_pullback,
        bb_len=(
            trial.suggest_int("bb_len", BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_len"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_len"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_len"]["step"])
            if ai_use_bb
            else BREAKOUT_PARAM_SPECS["bb_len"]["default"]
        ),
        bb_mult=(
            trial.suggest_float("bb_mult", BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_mult"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_mult"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_mult"]["step"])
            if ai_use_bb
            else BREAKOUT_PARAM_SPECS["bb_mult"]["default"]
        ),
        kc_len=(
            trial.suggest_int("kc_len", BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_len"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_len"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_len"]["step"])
            if ai_use_kc
            else BREAKOUT_PARAM_SPECS["kc_len"]["default"]
        ),
        kc_mult=(
            trial.suggest_float("kc_mult", BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_mult"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_mult"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_mult"]["step"])
            if ai_use_kc
            else BREAKOUT_PARAM_SPECS["kc_mult"]["default"]
        ),
        vol_short_len=vol_short_len,
        vol_long_len=vol_long_len,
        ema_pullback_short_len=ema_short_len,
        ema_pullback_mid_len=ema_mid_len,
        ema_pullback_long_len=ema_long_len,
        min_history_trades=trial.suggest_int("min_history_trades", BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_trades"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_trades"]["high"]),
        min_history_ev=trial.suggest_float("min_history_ev", BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_ev"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_ev"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_ev"]["step"]),
        min_history_win_rate=trial.suggest_float("min_history_win_rate", BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_win_rate"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_win_rate"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_win_rate"]["step"]),
        use_compounding=True,
    )


BREAKOUT_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELDS = frozenset({
    "high_len",
    "atr_len",
    "atr_times_trail",
    "atr_buy_tol",
    "bb_len",
    "bb_mult",
    "kc_len",
    "kc_mult",
    "vol_short_len",
    "vol_long_len",
    "ema_pullback_short_len",
    "ema_pullback_mid_len",
    "ema_pullback_long_len",
})

BREAKOUT_LOCAL_MIN_PORTFOLIO_DEPENDENCY_FIELDS = frozenset({
    "atr_times_init",
    "tp_percent",
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
        "high_len",
        "atr_len",
        "atr_times_init",
        "atr_times_trail",
        "atr_buy_tol",
    ]
    if "tp_percent" in getattr(trial, "params", {}):
        candidate_fields.append("tp_percent")
    if bool(center_payload.get("use_bb", False)):
        candidate_fields.extend(("bb_len", "bb_mult"))
    if bool(center_payload.get("use_kc", False)):
        candidate_fields.extend(("kc_len", "kc_mult"))
    if bool(center_payload.get("use_vol", False)):
        candidate_fields.extend(("vol_short_len", "vol_long_len"))
    if bool(center_payload.get("use_ema_pullback", False)):
        candidate_fields.extend(("ema_pullback_short_len", "ema_pullback_mid_len", "ema_pullback_long_len"))
    return tuple(candidate_fields)


def resolve_breakout_neighbor_spec(field_name, *, center_payload=None):
    spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE[field_name]
    kind = str(spec["kind"])
    step = spec.get("step", 1)
    high_value = spec["high"]

    if field_name == "vol_long_len":
        floor_value = int(BREAKOUT_PARAM_SPECS["vol_long_len"].get("min_value", 1))
        anchor_value = floor_value
        if center_payload is not None:
            anchor_value = max(anchor_value, int(center_payload["vol_short_len"]))
        low_value = anchor_value
    elif field_name == "ema_pullback_short_len":
        low_value = spec["low"]
        if center_payload is not None:
            high_value = min(int(high_value), int(center_payload["ema_pullback_mid_len"]) - 1)
    elif field_name == "ema_pullback_mid_len":
        low_value = spec["low"]
        if center_payload is not None:
            low_value = max(int(low_value), int(center_payload["ema_pullback_short_len"]) + 1)
            high_value = min(int(high_value), int(center_payload["ema_pullback_long_len"]) - 1)
    elif field_name == "ema_pullback_long_len":
        low_value = spec["low"]
        if center_payload is not None:
            low_value = max(int(low_value), int(center_payload["ema_pullback_mid_len"]) + 1)
    else:
        low_value = spec["low"]

    return step, kind, low_value, high_value


def get_breakout_optimizer_required_min_rows():
    return get_required_min_rows_from_lookbacks(
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_short_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_long_len"]["high"],
        BREAKOUT_OPTIMIZER_SEARCH_SPACE["ema_pullback_long_len"]["high"] + 20,
    )
