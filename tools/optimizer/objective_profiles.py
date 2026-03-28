from core.v16_config import V16StrategyParams


def build_trial_params(session, trial):
    ai_use_bb = trial.suggest_categorical("use_bb", [True, False])
    ai_use_kc = trial.suggest_categorical("use_kc", [True, False])
    ai_use_vol = trial.suggest_categorical("use_vol", [True, False])

    if ai_use_vol:
        vol_short_len = trial.suggest_int("vol_short_len", 1, 10)
        vol_long_len = trial.suggest_int("vol_long_len", vol_short_len, 30)
    else:
        vol_short_len = 5
        vol_long_len = 19

    return V16StrategyParams(
        atr_len=trial.suggest_int("atr_len", 3, 25),
        atr_times_init=trial.suggest_float("atr_times_init", 1.0, 3.5, step=0.1),
        atr_times_trail=trial.suggest_float("atr_times_trail", 2.0, 4.5, step=0.1),
        atr_buy_tol=trial.suggest_float("atr_buy_tol", 0.1, 3.5, step=0.1),
        high_len=trial.suggest_int("high_len", session.optimizer_high_len_min, session.optimizer_high_len_max, step=session.optimizer_high_len_step),
        tp_percent=session.resolve_optimizer_tp_percent(trial, fixed_tp_percent=session.optimizer_fixed_tp_percent),
        use_bb=ai_use_bb,
        use_kc=ai_use_kc,
        use_vol=ai_use_vol,
        bb_len=trial.suggest_int("bb_len", 10, 30, step=1) if ai_use_bb else 20,
        bb_mult=trial.suggest_float("bb_mult", 1.0, 2.5, step=0.1) if ai_use_bb else 2.0,
        kc_len=trial.suggest_int("kc_len", 3, 30, step=1) if ai_use_kc else 20,
        kc_mult=trial.suggest_float("kc_mult", 1.0, 3.0, step=0.1) if ai_use_kc else 2.0,
        vol_short_len=vol_short_len,
        vol_long_len=vol_long_len,
        min_history_trades=trial.suggest_int("min_history_trades", 0, 5),
        min_history_ev=trial.suggest_float("min_history_ev", -1.0, 0.5, step=0.1),
        min_history_win_rate=trial.suggest_float("min_history_win_rate", 0.0, 0.6, step=0.01),
        use_compounding=True,
    )


def build_initial_profile_row(trial_number, prep_wall_sec, prep_profile):
    return {
        "trial_number": trial_number + 1,
        "objective_wall_sec": 0.0,
        "prep_wall_sec": prep_wall_sec,
        "prep_worker_total_sum_sec": prep_profile["worker_total_sum_sec"],
        "prep_worker_copy_sum_sec": prep_profile["copy_sum_sec"],
        "prep_worker_generate_signals_sum_sec": prep_profile["generate_signals_sum_sec"],
        "prep_worker_assign_sum_sec": prep_profile["assign_sum_sec"],
        "prep_worker_run_backtest_sum_sec": prep_profile["run_backtest_sum_sec"],
        "prep_worker_to_dict_sum_sec": prep_profile["to_dict_sum_sec"],
        "prep_ok_count": prep_profile["ok_count"],
        "prep_fail_count": prep_profile["fail_count"],
        "prep_avg_per_ok_sec": (prep_profile["prep_total_sum_sec"] / prep_profile["ok_count"]) if prep_profile["ok_count"] > 0 else 0.0,
        "sort_dates_sec": 0.0,
        "portfolio_wall_sec": 0.0,
        "portfolio_total_sec": 0.0,
        "portfolio_ticker_dates_sec": 0.0,
        "portfolio_build_trade_index_sec": 0.0,
        "portfolio_day_loop_sec": 0.0,
        "portfolio_candidate_scan_sec": 0.0,
        "portfolio_rotation_sec": 0.0,
        "portfolio_settle_sec": 0.0,
        "portfolio_buy_sec": 0.0,
        "portfolio_equity_mark_sec": 0.0,
        "portfolio_closeout_sec": 0.0,
        "portfolio_curve_stats_sec": 0.0,
        "filter_rules_sec": 0.0,
        "score_calc_sec": 0.0,
        "ret_pct": 0.0,
        "mdd": 0.0,
        "trade_count": 0,
        "annual_return_pct": 0.0,
        "annual_trades": 0.0,
        "reserved_buy_fill_rate": 0.0,
        "full_year_count": 0,
        "min_full_year_return_pct": 0.0,
        "m_win_rate": 0.0,
        "r_squared": 0.0,
        "base_score": 0.0,
        "trial_value": -9999.0,
        "fail_reason": "",
    }
