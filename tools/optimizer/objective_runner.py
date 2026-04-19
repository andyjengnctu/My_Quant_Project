import time

import pandas as pd

from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_stats import calc_portfolio_score
from core.strategy_params import build_runtime_param_raw_value
from tools.optimizer.objective_filters import apply_filter_rules
from core.walk_forward_policy import filter_search_train_dates
from tools.optimizer.objective_profiles import build_initial_profile_row, build_trial_params
from tools.optimizer.prep import is_insufficient_data_message, prepare_trial_inputs
from tools.optimizer.study_utils import (
    INVALID_TRIAL_VALUE,
    OBJECTIVE_MODE_LEGACY_BASE_SCORE,
    OBJECTIVE_MODE_SPLIT_TRAIN_ROMD,
    normalize_objective_mode,
)


def _append_invalid_profile_row(*, session, trial, profile_row, fail_reason: str, objective_start: float):
    trial.set_user_attr("fail_reason", fail_reason)
    profile_row["fail_reason"] = fail_reason
    profile_row["trial_value"] = INVALID_TRIAL_VALUE
    profile_row["objective_wall_sec"] = time.perf_counter() - objective_start
    session.profile_recorder.append_row(profile_row)
    trial.set_user_attr("profile_row", profile_row)
    return INVALID_TRIAL_VALUE


def run_optimizer_objective(session, trial):
    objective_start = time.perf_counter()
    ai_params = build_trial_params(session, trial)
    prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(ai_params, "optimizer_max_workers"))
    prep_result = prepare_trial_inputs(
        raw_data_cache=session.raw_data_cache,
        params=ai_params,
        default_max_workers=session.default_max_workers,
        executor_bundle=prep_executor_bundle,
        static_fast_cache=session.static_fast_cache,
        static_master_dates=session.master_dates,
    )
    trial.set_user_attr("prep_mode", prep_result["prep_mode"])
    trial.set_user_attr("prep_start_method", prep_result["pool_start_method"] or "default")
    if prep_result["pool_error_text"] is not None:
        trial.set_user_attr("prep_pool_error", prep_result["pool_error_text"])

    prep_failures = prep_result["prep_failures"]
    if prep_failures:
        insufficient_failures = [item for item in prep_failures if is_insufficient_data_message(item[1])]
        session.record_optimizer_prep_failures(insufficient_failures)

    profile_row = build_initial_profile_row(trial.number, prep_result["prep_wall_sec"], prep_result["prep_profile"])
    mode = normalize_objective_mode(session.objective_mode)
    profile_row["objective_mode"] = str(mode)
    profile_row["search_train_end_year"] = int(session.search_train_end_year)
    master_dates = prep_result["master_dates"]
    if not master_dates:
        profile_row["objective_wall_sec"] = time.perf_counter() - objective_start
        profile_row["fail_reason"] = "無有效資料"
        session.profile_recorder.append_row(profile_row)
        trial.set_user_attr("profile_row", profile_row)
        return INVALID_TRIAL_VALUE

    sort_start = time.perf_counter()
    sorted_dates = sorted(master_dates)
    profile_row["sort_dates_sec"] = time.perf_counter() - sort_start
    use_full_history_search = mode == OBJECTIVE_MODE_LEGACY_BASE_SCORE
    if use_full_history_search:
        search_train_dates = list(sorted_dates)
        effective_search_train_end_year = int(pd.Timestamp(sorted_dates[-1]).year)
    else:
        search_train_dates = filter_search_train_dates(
            sorted_dates=sorted_dates,
            train_start_year=int(session.train_start_year),
            search_train_end_year=int(session.search_train_end_year),
        )
        effective_search_train_end_year = int(session.search_train_end_year)
    profile_row["search_train_end_year"] = int(effective_search_train_end_year)
    profile_row["search_train_date_count"] = int(len(search_train_dates))
    trial.set_user_attr("objective_mode", str(mode))
    trial.set_user_attr("search_train_end_year", int(effective_search_train_end_year))
    trial.set_user_attr("search_train_date_count", int(len(search_train_dates)))
    if not search_train_dates:
        return _append_invalid_profile_row(
            session=session,
            trial=trial,
            profile_row=profile_row,
            fail_reason="主搜尋 train 區間無有效資料",
            objective_start=objective_start,
        )

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
        search_train_dates,
        session.train_start_year,
        ai_params,
        session.train_max_positions,
        session.train_enable_rotation,
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
    fail_reason = apply_filter_rules(metrics)
    profile_row["filter_rules_sec"] = time.perf_counter() - filter_start
    if fail_reason is not None:
        return _append_invalid_profile_row(
            session=session,
            trial=trial,
            profile_row=profile_row,
            fail_reason=fail_reason,
            objective_start=objective_start,
        )

    score_start = time.perf_counter()
    base_score = calc_portfolio_score(ret_pct, mdd, m_win_rate, r_sq, annual_return_pct=annual_return_pct)
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

    final_score = float(base_score)
    if mode not in {OBJECTIVE_MODE_LEGACY_BASE_SCORE, OBJECTIVE_MODE_SPLIT_TRAIN_ROMD}:
        return _append_invalid_profile_row(
            session=session,
            trial=trial,
            profile_row=profile_row,
            fail_reason=f"未知 objective_mode: {session.objective_mode}",
            objective_start=objective_start,
        )

    profile_row["trial_value"] = float(final_score)
    profile_row["objective_wall_sec"] = time.perf_counter() - objective_start
    session.profile_recorder.append_row(profile_row)
    trial.set_user_attr("profile_row", profile_row)
    return float(final_score)
