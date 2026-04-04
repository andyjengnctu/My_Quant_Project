from strategies.breakout.schema import BREAKOUT_PARAM_SPECS, V16StrategyParams


BREAKOUT_OPTIMIZER_SEARCH_SPACE = {
    "use_bb": {"kind": "categorical", "choices": [True, False]},
    "use_kc": {"kind": "categorical", "choices": [True, False]},
    "use_vol": {"kind": "categorical", "choices": [True, False]},
    "atr_len": {"kind": "int", "low": 3, "high": 25},
    "atr_times_init": {"kind": "float", "low": 1.0, "high": 3.5, "step": 0.1},
    "atr_times_trail": {"kind": "float", "low": 2.0, "high": 4.5, "step": 0.1},
    "atr_buy_tol": {"kind": "float", "low": 0.1, "high": 3.5, "step": 0.1},
    "bb_len": {"kind": "int", "low": 10, "high": 30, "step": 1, "enabled_by": "use_bb"},
    "bb_mult": {"kind": "float", "low": 1.0, "high": 2.5, "step": 0.1, "enabled_by": "use_bb"},
    "kc_len": {"kind": "int", "low": 3, "high": 30, "step": 1, "enabled_by": "use_kc"},
    "kc_mult": {"kind": "float", "low": 1.0, "high": 3.0, "step": 0.1, "enabled_by": "use_kc"},
    "vol_short_len": {"kind": "int", "low": 1, "high": 10, "enabled_by": "use_vol"},
    "vol_long_len": {"kind": "int", "high": 30, "depends_on": "vol_short_len", "enabled_by": "use_vol"},
    "min_history_trades": {"kind": "int", "low": 0, "high": 5},
    "min_history_ev": {"kind": "float", "low": -1.0, "high": 0.5, "step": 0.1},
    "min_history_win_rate": {"kind": "float", "low": 0.0, "high": 0.6, "step": 0.01},
}


def build_trial_params(session, trial):
    ai_use_bb = trial.suggest_categorical("use_bb", BREAKOUT_OPTIMIZER_SEARCH_SPACE["use_bb"]["choices"])
    ai_use_kc = trial.suggest_categorical("use_kc", BREAKOUT_OPTIMIZER_SEARCH_SPACE["use_kc"]["choices"])
    ai_use_vol = trial.suggest_categorical("use_vol", BREAKOUT_OPTIMIZER_SEARCH_SPACE["use_vol"]["choices"])

    if ai_use_vol:
        vol_short_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_short_len"]
        vol_short_len = trial.suggest_int("vol_short_len", vol_short_spec["low"], vol_short_spec["high"])
        vol_long_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_long_len"]
        vol_long_len = trial.suggest_int("vol_long_len", vol_short_len, vol_long_spec["high"])
    else:
        vol_short_len = BREAKOUT_PARAM_SPECS["vol_short_len"]["default"]
        vol_long_len = BREAKOUT_PARAM_SPECS["vol_long_len"]["default"]

    return V16StrategyParams(
        atr_len=trial.suggest_int("atr_len", BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"]["high"]),
        atr_times_init=trial.suggest_float("atr_times_init", BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_init"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_init"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_init"]["step"]),
        atr_times_trail=trial.suggest_float("atr_times_trail", BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_trail"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_trail"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_times_trail"]["step"]),
        atr_buy_tol=trial.suggest_float("atr_buy_tol", BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_buy_tol"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_buy_tol"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_buy_tol"]["step"]),
        high_len=trial.suggest_int("high_len", session.optimizer_high_len_min, session.optimizer_high_len_max, step=session.optimizer_high_len_step),
        tp_percent=session.resolve_optimizer_tp_percent(trial, fixed_tp_percent=session.optimizer_fixed_tp_percent),
        use_bb=ai_use_bb,
        use_kc=ai_use_kc,
        use_vol=ai_use_vol,
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
        min_history_trades=trial.suggest_int("min_history_trades", BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_trades"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_trades"]["high"]),
        min_history_ev=trial.suggest_float("min_history_ev", BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_ev"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_ev"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_ev"]["step"]),
        min_history_win_rate=trial.suggest_float("min_history_win_rate", BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_win_rate"]["low"], BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_win_rate"]["high"], step=BREAKOUT_OPTIMIZER_SEARCH_SPACE["min_history_win_rate"]["step"]),
        use_compounding=True,
    )
