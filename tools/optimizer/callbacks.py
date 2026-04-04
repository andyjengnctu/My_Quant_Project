from tools.optimizer.study_utils import is_qualified_trial_value

def run_optimizer_monitoring_callback(session, study, trial):
    session.current_session_trial += 1
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    prep_mode = trial.user_attrs.get("prep_mode", "parallel")
    mode_suffix = " [fallback]" if prep_mode == "sequential_fallback" else ""

    if trial.value is None:
        state_name = getattr(trial.state, "name", str(trial.state))
        status_text, score_text = f"{session.colors['yellow']}{state_name}{mode_suffix}{session.colors['reset']}", "N/A"
    elif not is_qualified_trial_value(trial.value):
        fail_msg = trial.user_attrs.get("fail_reason", "策略無效")
        status_text, score_text = f"{session.colors['yellow']}淘汰 [{fail_msg}]{mode_suffix}{session.colors['reset']}", "N/A"
    else:
        status_text, score_text = f"{session.colors['green']}進化中{mode_suffix}{session.colors['reset']}", f"{trial.value:.3f}"

    total_trials_display = str(session.n_trials) if isinstance(session.n_trials, int) and session.n_trials > 0 else "?"
    print(
        f"\r{session.colors['gray']}⏳ [累積 {trial.number + 1:>4} | 本輪 {session.current_session_trial:>3}/{total_trials_display}] "
        f"耗時: {duration:>5.1f}s | 系統評分: {score_text:>7} | 狀態: {status_text}{session.colors['reset']}\033[K",
        end="",
        flush=True,
    )

    if session.profile_recorder.enabled and session.profile_recorder.console_print and (
        session.current_session_trial % session.profile_recorder.print_every_n_trials == 0
    ):
        profile = trial.user_attrs.get("profile_row", {})
        print()
        print(
            f"{session.colors['gray']}   [Profile] total={float(profile.get('objective_wall_sec', 0.0)):.3f}s | "
            f"prep_wall={float(profile.get('prep_wall_sec', 0.0)):.3f}s | "
            f"pf_wall={float(profile.get('portfolio_wall_sec', 0.0)):.3f}s | "
            f"gen_sum={float(profile.get('prep_worker_generate_signals_sum_sec', 0.0)):.3f}s | "
            f"backtest_sum={float(profile.get('prep_worker_run_backtest_sum_sec', 0.0)):.3f}s | "
            f"to_dict_sum={float(profile.get('prep_worker_to_dict_sum_sec', 0.0)):.3f}s | "
            f"pf_loop={float(profile.get('portfolio_day_loop_sec', 0.0)):.3f}s{session.colors['reset']}"
        )

    best_completed_trial = session.get_best_completed_trial_or_none(study)
    if (
        best_completed_trial is not None
        and best_completed_trial.number == trial.number
        and is_qualified_trial_value(trial.value)
    ):
        print()
        attrs = trial.user_attrs
        params = session.build_optimizer_trial_params(trial.params, attrs, fixed_tp_percent=session.optimizer_fixed_tp_percent)
        mode_display = "啟用 (汰弱換強)" if session.train_enable_rotation else "關閉 (穩定鎖倉)"
        print(f"\n{session.colors['red']}🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試){session.colors['reset']}")
        session.print_strategy_dashboard(
            params=params,
            title="績效與風險對比表",
            mode_display=mode_display,
            max_pos=session.train_max_positions,
            trades=attrs["pf_trades"],
            missed_b=attrs.get("missed_buys", 0),
            missed_s=attrs.get("missed_sells", 0),
            final_eq=attrs["final_equity"],
            avg_exp=attrs["avg_exposure"],
            max_exp=attrs.get("max_exposure", None),
            sys_ret=attrs["pf_return"],
            bm_ret=attrs["bm_return"],
            sys_mdd=attrs["pf_mdd"],
            bm_mdd=attrs["bm_mdd"],
            win_rate=attrs["win_rate"],
            payoff=attrs["pf_payoff"],
            ev=attrs["pf_ev"],
            r_sq=attrs["r_squared"],
            m_win_rate=attrs["m_win_rate"],
            bm_r_sq=attrs.get("bm_r_squared", 0.0),
            bm_m_win_rate=attrs.get("bm_m_win_rate", 0.0),
            normal_trades=attrs.get("normal_trades", attrs["pf_trades"]),
            extended_trades=attrs.get("extended_trades", 0),
            annual_trades=attrs.get("annual_trades", 0.0),
            reserved_buy_fill_rate=attrs.get("reserved_buy_fill_rate", 0.0),
            annual_return_pct=attrs.get("annual_return_pct", 0.0),
            bm_annual_return_pct=attrs.get("bm_annual_return_pct", 0.0),
            min_full_year_return_pct=attrs.get("min_full_year_return_pct", 0.0),
            bm_min_full_year_return_pct=attrs.get("bm_min_full_year_return_pct", 0.0),
        )
        print(
            f"{session.colors['gray']}   年化報酬率: {attrs.get('annual_return_pct', 0.0):.2f}% | "
            f"年化交易次數: {attrs.get('annual_trades', 0.0):.1f} 次/年 | "
            f"保留後買進成交率: {attrs.get('reserved_buy_fill_rate', 0.0):.1f}% | "
            f"完整年度數: {attrs.get('full_year_count', 0)} | "
            f"最差完整年度: {attrs.get('min_full_year_return_pct', 0.0):.2f}%{session.colors['reset']}"
        )
