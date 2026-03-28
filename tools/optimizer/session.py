import time

from core.v16_config import (
    MAX_PORTFOLIO_MDD_PCT,
    MIN_ANNUAL_TRADES,
    MIN_BUY_FILL_RATE,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_FULL_YEAR_RETURN_PCT,
    MIN_MONTHLY_WIN_RATE,
    MIN_TRADE_WIN_RATE,
    V16StrategyParams,
)
from core.v16_portfolio_engine import run_portfolio_timeline
from core.v16_portfolio_stats import calc_portfolio_score
from tools.optimizer.prep import is_insufficient_data_message, prepare_trial_inputs


def close_study_storage(study):
    storage = getattr(study, "_storage", None)
    if storage is None:
        return
    remove_session = getattr(storage, "remove_session", None)
    if callable(remove_session):
        remove_session()
    backend = getattr(storage, "_backend", None)
    backend_remove_session = getattr(backend, "remove_session", None)
    if callable(backend_remove_session):
        backend_remove_session()
    engine = getattr(backend, "engine", None)
    dispose = getattr(engine, "dispose", None)
    if callable(dispose):
        dispose()


class OptimizerSession:
    def __init__(
        self,
        *,
        output_dir,
        session_ts,
        profile_recorder_cls,
        build_optimizer_trial_params,
        get_best_completed_trial_or_none,
        resolve_optimizer_tp_percent,
        print_strategy_dashboard,
        colors,
        optimizer_fixed_tp_percent,
        train_max_positions,
        train_start_year,
        train_enable_rotation,
        optimizer_high_len_min,
        optimizer_high_len_max,
        optimizer_high_len_step,
        default_max_workers,
        enable_optimizer_profiling,
        enable_profile_console_print,
        profile_print_every_n_trials,
    ):
        self.raw_data_cache = {}
        self.raw_data_cache_data_dir = None
        self.current_session_trial = 0
        self.n_trials = 0
        self.prep_summary = {
            "trials_with_insufficient": 0,
            "insufficient_count_total": 0,
        }
        self.output_dir = output_dir
        self.session_ts = session_ts
        self.build_optimizer_trial_params = build_optimizer_trial_params
        self.get_best_completed_trial_or_none = get_best_completed_trial_or_none
        self.resolve_optimizer_tp_percent = resolve_optimizer_tp_percent
        self.print_strategy_dashboard = print_strategy_dashboard
        self.colors = colors
        self.optimizer_fixed_tp_percent = optimizer_fixed_tp_percent
        self.train_max_positions = train_max_positions
        self.train_start_year = train_start_year
        self.train_enable_rotation = train_enable_rotation
        self.optimizer_high_len_min = optimizer_high_len_min
        self.optimizer_high_len_max = optimizer_high_len_max
        self.optimizer_high_len_step = optimizer_high_len_step
        self.default_max_workers = default_max_workers
        self.profile_recorder = profile_recorder_cls(
            output_dir=output_dir,
            session_ts=session_ts,
            enabled=enable_optimizer_profiling,
            console_print=enable_profile_console_print,
            print_every_n_trials=profile_print_every_n_trials,
        )

    def load_raw_data(self, data_dir, *, load_all_raw_data, required_min_rows):
        self.raw_data_cache = load_all_raw_data(
            data_dir=data_dir,
            required_min_rows=required_min_rows,
            output_dir=self.output_dir,
        )
        self.raw_data_cache_data_dir = data_dir

    def record_optimizer_prep_failures(self, insufficient_failures):
        insufficient_count = len(insufficient_failures)
        if insufficient_count <= 0:
            return
        self.prep_summary["trials_with_insufficient"] += 1
        self.prep_summary["insufficient_count_total"] += insufficient_count

    def print_optimizer_prep_summary(self):
        insufficient_total = self.prep_summary["insufficient_count_total"]
        if insufficient_total == 0:
            return
        print(
            f"{self.colors['yellow']}⚠️ 本輪預處理摘要："
            f"資料不足 trial={self.prep_summary['trials_with_insufficient']} 次 / 累計 {insufficient_total} 檔。{self.colors['reset']}"
        )

    def _build_trial_params(self, trial):
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
            high_len=trial.suggest_int("high_len", self.optimizer_high_len_min, self.optimizer_high_len_max, step=self.optimizer_high_len_step),
            tp_percent=self.resolve_optimizer_tp_percent(trial, fixed_tp_percent=self.optimizer_fixed_tp_percent),
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

    def _build_initial_profile_row(self, trial_number, prep_wall_sec, prep_profile):
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

    def _apply_filter_rules(self, metrics):
        if metrics["mdd"] > MAX_PORTFOLIO_MDD_PCT:
            return f"回撤過大 ({metrics['mdd']:.1f}%)"
        if metrics["annual_trades"] < MIN_ANNUAL_TRADES:
            return f"年化交易次數過低 ({metrics['annual_trades']:.2f}次/年)"
        if metrics["reserved_buy_fill_rate"] < MIN_BUY_FILL_RATE:
            return f"保留後買進成交率過低 ({metrics['reserved_buy_fill_rate']:.2f}%)"
        if metrics["annual_return_pct"] <= 0:
            return f"年化報酬率非正 ({metrics['annual_return_pct']:.2f}%)"
        if metrics["full_year_count"] <= 0:
            return "無完整年度可驗證 min{r_y}"
        if metrics["min_full_year_return_pct"] <= MIN_FULL_YEAR_RETURN_PCT:
            return (
                f"完整年度最差報酬未大於 {MIN_FULL_YEAR_RETURN_PCT:.2f}% "
                f"({metrics['min_full_year_return_pct']:.2f}%)"
            )
        if metrics["win_rate"] < MIN_TRADE_WIN_RATE:
            return f"實戰勝率偏低 ({metrics['win_rate']:.2f}%)"
        if metrics["m_win_rate"] < MIN_MONTHLY_WIN_RATE:
            return f"月勝率偏低 ({metrics['m_win_rate']:.0f}%)"
        if metrics["r_sq"] < MIN_EQUITY_CURVE_R_SQUARED:
            return f"曲線過度震盪 (R²={metrics['r_sq']:.2f})"
        return None

    def objective(self, trial):
        objective_start = time.perf_counter()
        ai_params = self._build_trial_params(trial)
        prep_result = prepare_trial_inputs(
            raw_data_cache=self.raw_data_cache,
            params=ai_params,
            default_max_workers=self.default_max_workers,
        )
        trial.set_user_attr("prep_mode", prep_result["prep_mode"])
        trial.set_user_attr("prep_start_method", prep_result["pool_start_method"] or "default")
        if prep_result["pool_error_text"] is not None:
            trial.set_user_attr("prep_pool_error", prep_result["pool_error_text"])

        prep_failures = prep_result["prep_failures"]
        if prep_failures:
            insufficient_failures = [item for item in prep_failures if is_insufficient_data_message(item[1])]
            self.record_optimizer_prep_failures(insufficient_failures)

        profile_row = self._build_initial_profile_row(trial.number, prep_result["prep_wall_sec"], prep_result["prep_profile"])
        master_dates = prep_result["master_dates"]
        if not master_dates:
            profile_row["objective_wall_sec"] = time.perf_counter() - objective_start
            profile_row["fail_reason"] = "無有效資料"
            self.profile_recorder.append_row(profile_row)
            trial.set_user_attr("profile_row", profile_row)
            return -9999.0

        sort_start = time.perf_counter()
        sorted_dates = sorted(master_dates)
        profile_row["sort_dates_sec"] = time.perf_counter() - sort_start
        all_dfs_fast = prep_result["all_dfs_fast"]
        all_trade_logs = prep_result["all_trade_logs"]
        benchmark_data = all_dfs_fast.get("0050", None)

        pf_profile = {}
        portfolio_start = time.perf_counter()
        (
            ret_pct,
            mdd,
            trade_count,
            final_eq,
            avg_exp,
            max_exp,
            bm_ret,
            bm_mdd,
            win_rate,
            pf_ev,
            pf_payoff,
            total_missed,
            total_missed_sells,
            r_sq,
            m_win_rate,
            bm_r_sq,
            bm_m_win_rate,
            normal_trade_count,
            extended_trade_count,
            annual_trades,
            reserved_buy_fill_rate,
            annual_return_pct,
            bm_annual_return_pct,
        ) = run_portfolio_timeline(
            all_dfs_fast,
            all_trade_logs,
            sorted_dates,
            self.train_start_year,
            ai_params,
            self.train_max_positions,
            self.train_enable_rotation,
            benchmark_ticker="0050",
            benchmark_data=benchmark_data,
            is_training=True,
            profile_stats=pf_profile,
            verbose=False,
        )
        profile_row["portfolio_wall_sec"] = time.perf_counter() - portfolio_start
        profile_row["portfolio_total_sec"] = float(pf_profile.get("portfolio_wall_sec", 0.0))
        profile_row["portfolio_ticker_dates_sec"] = float(pf_profile.get("portfolio_ticker_dates_sec", 0.0))
        profile_row["portfolio_build_trade_index_sec"] = float(pf_profile.get("portfolio_build_trade_index_sec", 0.0))
        profile_row["portfolio_day_loop_sec"] = float(pf_profile.get("portfolio_day_loop_sec", 0.0))
        profile_row["portfolio_candidate_scan_sec"] = float(pf_profile.get("portfolio_candidate_scan_sec", 0.0))
        profile_row["portfolio_rotation_sec"] = float(pf_profile.get("portfolio_rotation_sec", 0.0))
        profile_row["portfolio_settle_sec"] = float(pf_profile.get("portfolio_settle_sec", 0.0))
        profile_row["portfolio_buy_sec"] = float(pf_profile.get("portfolio_buy_sec", 0.0))
        profile_row["portfolio_equity_mark_sec"] = float(pf_profile.get("portfolio_equity_mark_sec", 0.0))
        profile_row["portfolio_closeout_sec"] = float(pf_profile.get("portfolio_closeout_sec", 0.0))
        profile_row["portfolio_curve_stats_sec"] = float(pf_profile.get("curve_stats_sec", 0.0))
        full_year_count = int(pf_profile.get("full_year_count", 0))
        min_full_year_return_pct = float(pf_profile.get("min_full_year_return_pct", 0.0))
        bm_min_full_year_return_pct = float(pf_profile.get("bm_min_full_year_return_pct", 0.0))
        profile_row["ret_pct"] = ret_pct
        profile_row["mdd"] = mdd
        profile_row["trade_count"] = trade_count
        profile_row["annual_return_pct"] = annual_return_pct
        profile_row["annual_trades"] = annual_trades
        profile_row["reserved_buy_fill_rate"] = reserved_buy_fill_rate
        profile_row["full_year_count"] = full_year_count
        profile_row["min_full_year_return_pct"] = min_full_year_return_pct
        profile_row["m_win_rate"] = m_win_rate
        profile_row["r_squared"] = r_sq

        metrics = {
            "mdd": mdd,
            "annual_trades": annual_trades,
            "reserved_buy_fill_rate": reserved_buy_fill_rate,
            "annual_return_pct": annual_return_pct,
            "full_year_count": full_year_count,
            "min_full_year_return_pct": min_full_year_return_pct,
            "win_rate": win_rate,
            "m_win_rate": m_win_rate,
            "r_sq": r_sq,
        }
        filter_start = time.perf_counter()
        fail_reason = self._apply_filter_rules(metrics)
        profile_row["filter_rules_sec"] = time.perf_counter() - filter_start
        if fail_reason is not None:
            trial.set_user_attr("fail_reason", fail_reason)
            profile_row["fail_reason"] = fail_reason
            profile_row["trial_value"] = -9999.0
            profile_row["objective_wall_sec"] = time.perf_counter() - objective_start
            self.profile_recorder.append_row(profile_row)
            trial.set_user_attr("profile_row", profile_row)
            return -9999.0

        score_start = time.perf_counter()
        base_score = calc_portfolio_score(ret_pct, mdd, m_win_rate, r_sq, annual_return_pct=annual_return_pct)
        final_score = base_score
        profile_row["score_calc_sec"] = time.perf_counter() - score_start
        profile_row["base_score"] = base_score

        trial.set_user_attr("pf_return", ret_pct)
        trial.set_user_attr("pf_mdd", mdd)
        trial.set_user_attr("pf_trades", trade_count)
        trial.set_user_attr("final_equity", final_eq)
        trial.set_user_attr("avg_exposure", avg_exp)
        trial.set_user_attr("max_exposure", max_exp)
        trial.set_user_attr("bm_return", bm_ret)
        trial.set_user_attr("bm_mdd", bm_mdd)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("pf_ev", pf_ev)
        trial.set_user_attr("pf_payoff", pf_payoff)
        trial.set_user_attr("missed_buys", total_missed)
        trial.set_user_attr("missed_sells", total_missed_sells)
        trial.set_user_attr("normal_trades", normal_trade_count)
        trial.set_user_attr("extended_trades", extended_trade_count)
        trial.set_user_attr("annual_trades", annual_trades)
        trial.set_user_attr("reserved_buy_fill_rate", reserved_buy_fill_rate)
        trial.set_user_attr("annual_return_pct", annual_return_pct)
        trial.set_user_attr("bm_annual_return_pct", bm_annual_return_pct)
        trial.set_user_attr("full_year_count", full_year_count)
        trial.set_user_attr("min_full_year_return_pct", min_full_year_return_pct)
        trial.set_user_attr("yearly_return_rows", pf_profile.get("yearly_return_rows", []))
        trial.set_user_attr("base_score", base_score)
        trial.set_user_attr("bm_min_full_year_return_pct", bm_min_full_year_return_pct)
        trial.set_user_attr("r_squared", r_sq)
        trial.set_user_attr("m_win_rate", m_win_rate)
        trial.set_user_attr("bm_r_squared", bm_r_sq)
        trial.set_user_attr("bm_m_win_rate", bm_m_win_rate)
        profile_row["trial_value"] = final_score
        profile_row["objective_wall_sec"] = time.perf_counter() - objective_start
        self.profile_recorder.append_row(profile_row)
        trial.set_user_attr("profile_row", profile_row)
        return final_score

    def monitoring_callback(self, study, trial):
        self.current_session_trial += 1
        duration = trial.duration.total_seconds() if trial.duration else 0.0
        prep_mode = trial.user_attrs.get("prep_mode", "parallel")
        mode_suffix = " [fallback]" if prep_mode == "sequential_fallback" else ""

        if trial.value is None:
            state_name = getattr(trial.state, "name", str(trial.state))
            status_text, score_text = f"{self.colors['yellow']}{state_name}{mode_suffix}{self.colors['reset']}", "N/A"
        elif trial.value <= -9000:
            fail_msg = trial.user_attrs.get("fail_reason", "策略無效")
            status_text, score_text = f"{self.colors['yellow']}淘汰 [{fail_msg}]{mode_suffix}{self.colors['reset']}", "N/A"
        else:
            status_text, score_text = f"{self.colors['green']}進化中{mode_suffix}{self.colors['reset']}", f"{trial.value:.3f}"

        total_trials_display = str(self.n_trials) if isinstance(self.n_trials, int) and self.n_trials > 0 else "?"
        print(
            f"\r{self.colors['gray']}⏳ [累積 {trial.number + 1:>4} | 本輪 {self.current_session_trial:>3}/{total_trials_display}] "
            f"耗時: {duration:>5.1f}s | 系統評分: {score_text:>7} | 狀態: {status_text}{self.colors['reset']}\033[K",
            end="",
            flush=True,
        )

        if self.profile_recorder.enabled and self.profile_recorder.console_print and (
            self.current_session_trial % self.profile_recorder.print_every_n_trials == 0
        ):
            profile = trial.user_attrs.get("profile_row", {})
            print()
            print(
                f"{self.colors['gray']}   [Profile] total={float(profile.get('objective_wall_sec', 0.0)):.3f}s | "
                f"prep_wall={float(profile.get('prep_wall_sec', 0.0)):.3f}s | "
                f"pf_wall={float(profile.get('portfolio_wall_sec', 0.0)):.3f}s | "
                f"gen_sum={float(profile.get('prep_worker_generate_signals_sum_sec', 0.0)):.3f}s | "
                f"backtest_sum={float(profile.get('prep_worker_run_backtest_sum_sec', 0.0)):.3f}s | "
                f"to_dict_sum={float(profile.get('prep_worker_to_dict_sum_sec', 0.0)):.3f}s | "
                f"pf_loop={float(profile.get('portfolio_day_loop_sec', 0.0)):.3f}s{self.colors['reset']}"
            )

        best_completed_trial = self.get_best_completed_trial_or_none(study)
        if (
            best_completed_trial is not None
            and best_completed_trial.number == trial.number
            and trial.value is not None
            and trial.value > -9000
        ):
            print()
            attrs = trial.user_attrs
            params = self.build_optimizer_trial_params(trial.params, attrs, fixed_tp_percent=self.optimizer_fixed_tp_percent)
            mode_display = "啟用 (汰弱換強)" if self.train_enable_rotation else "關閉 (穩定鎖倉)"
            print(f"\n{self.colors['red']}🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試){self.colors['reset']}")
            self.print_strategy_dashboard(
                params=params,
                title="績效與風險對比表",
                mode_display=mode_display,
                max_pos=self.train_max_positions,
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
                f"{self.colors['gray']}   年化報酬率: {attrs.get('annual_return_pct', 0.0):.2f}% | "
                f"年化交易次數: {attrs.get('annual_trades', 0.0):.1f} 次/年 | "
                f"保留後買進成交率: {attrs.get('reserved_buy_fill_rate', 0.0):.1f}% | "
                f"完整年度數: {attrs.get('full_year_count', 0)} | "
                f"最差完整年度: {attrs.get('min_full_year_return_pct', 0.0):.2f}%{self.colors['reset']}"
            )
