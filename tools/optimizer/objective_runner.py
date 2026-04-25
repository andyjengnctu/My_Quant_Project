import time
from typing import Any

import pandas as pd

from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_stats import calc_portfolio_score
from core.strategy_params import build_runtime_param_raw_value
from tools.optimizer.objective_filters import apply_filter_rules
from core.walk_forward_policy import filter_search_train_dates
from tools.optimizer.objective_profiles import build_initial_profile_row, build_trial_params
from tools.optimizer.param_cache import build_full_evaluation_cache_key, build_prep_cache_key
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


def resolve_search_train_scope(session, master_dates, *, objective_mode: str | None = None):
    mode = normalize_objective_mode(session.objective_mode if objective_mode is None else objective_mode)
    if not master_dates:
        return {
            "mode": str(mode),
            "sorted_dates": [],
            "search_train_dates": [],
            "effective_search_train_end_year": int(session.search_train_end_year),
            "fail_reason": "無有效資料",
        }

    sort_start = time.perf_counter()
    sorted_dates = sorted(master_dates)
    sort_dates_sec = time.perf_counter() - sort_start
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

    fail_reason = None if search_train_dates else "主搜尋 train 區間無有效資料"
    return {
        "mode": str(mode),
        "sorted_dates": sorted_dates,
        "search_train_dates": search_train_dates,
        "effective_search_train_end_year": int(effective_search_train_end_year),
        "sort_dates_sec": float(sort_dates_sec),
        "fail_reason": fail_reason,
    }


def evaluate_prepared_train_score(session, *, ai_params, prep_result, search_scope: dict[str, Any], profile_stats=None):
    fail_reason = str(search_scope.get("fail_reason") or "").strip()
    if fail_reason:
        return {
            "ok": False,
            "fail_reason": fail_reason,
            "score": float(INVALID_TRIAL_VALUE),
            "profile_stats": {},
        }

    all_dfs_fast = prep_result["all_dfs_fast"]
    all_trade_logs = prep_result["all_trade_logs"]
    all_pit_stats_index = prep_result.get("all_pit_stats_index")
    benchmark_data = all_dfs_fast.get("0050", None)
    pf_profile = {} if profile_stats is None else profile_stats

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
        list(search_scope["search_train_dates"]),
        session.train_start_year,
        ai_params,
        session.train_max_positions,
        session.train_enable_rotation,
        benchmark_ticker="0050",
        benchmark_data=benchmark_data,
        is_training=True,
        profile_stats=pf_profile,
        verbose=False,
        pit_stats_index=all_pit_stats_index,
    )

    full_year_count = int(pf_profile.get("full_year_count", 0))
    min_full_year_return_pct = float(pf_profile.get("min_full_year_return_pct", 0.0))
    bm_min_full_year_return_pct = float(pf_profile.get("bm_min_full_year_return_pct", 0.0))
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
    objective_fail_reason = apply_filter_rules(metrics)
    if objective_fail_reason is not None:
        return {
            "ok": False,
            "fail_reason": str(objective_fail_reason),
            "score": float(INVALID_TRIAL_VALUE),
            "profile_stats": pf_profile,
            "ret_pct": ret_pct,
            "mdd": mdd,
            "trade_count": trade_count,
            "final_equity": final_eq,
            "avg_exposure": avg_exp,
            "max_exposure": max_exp,
            "bm_return": bm_ret,
            "bm_mdd": bm_mdd,
            "win_rate": win_rate,
            "pf_ev": pf_ev,
            "pf_payoff": pf_payoff,
            "missed_buys": total_missed,
            "missed_sells": total_missed_sells,
            "normal_trades": normal_trade_count,
            "extended_trades": extended_trade_count,
            "annual_trades": annual_trades,
            "reserved_buy_fill_rate": reserved_buy_fill_rate,
            "annual_return_pct": annual_return_pct,
            "bm_annual_return_pct": bm_annual_return_pct,
            "full_year_count": full_year_count,
            "min_full_year_return_pct": min_full_year_return_pct,
            "yearly_return_rows": pf_profile.get("yearly_return_rows", []),
            "bm_min_full_year_return_pct": bm_min_full_year_return_pct,
            "r_squared": r_sq,
            "m_win_rate": m_win_rate,
            "bm_r_squared": bm_r_sq,
            "bm_m_win_rate": bm_m_win_rate,
            "base_score": float(INVALID_TRIAL_VALUE),
        }

    base_score = calc_portfolio_score(ret_pct, mdd, m_win_rate, r_sq, annual_return_pct=annual_return_pct)
    return {
        "ok": True,
        "fail_reason": None,
        "score": float(base_score),
        "profile_stats": pf_profile,
        "ret_pct": ret_pct,
        "mdd": mdd,
        "trade_count": trade_count,
        "final_equity": final_eq,
        "avg_exposure": avg_exp,
        "max_exposure": max_exp,
        "bm_return": bm_ret,
        "bm_mdd": bm_mdd,
        "win_rate": win_rate,
        "pf_ev": pf_ev,
        "pf_payoff": pf_payoff,
        "missed_buys": total_missed,
        "missed_sells": total_missed_sells,
        "normal_trades": normal_trade_count,
        "extended_trades": extended_trade_count,
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "annual_return_pct": annual_return_pct,
        "bm_annual_return_pct": bm_annual_return_pct,
        "full_year_count": full_year_count,
        "min_full_year_return_pct": min_full_year_return_pct,
        "yearly_return_rows": pf_profile.get("yearly_return_rows", []),
        "bm_min_full_year_return_pct": bm_min_full_year_return_pct,
        "r_squared": r_sq,
        "m_win_rate": m_win_rate,
        "bm_r_squared": bm_r_sq,
        "bm_m_win_rate": bm_m_win_rate,
        "base_score": float(base_score),
    }


def run_optimizer_objective(session, trial):
    objective_start = time.perf_counter()
    ai_params = build_trial_params(session, trial)
    prep_cache_key = build_prep_cache_key(ai_params)
    get_cached_prep = getattr(session, "get_prepared_trial_inputs_from_cache", None)
    prep_result = get_cached_prep(prep_cache_key) if callable(get_cached_prep) else None
    if prep_result is None:
        prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(ai_params, "optimizer_max_workers"))
        prep_result = prepare_trial_inputs(
            raw_data_cache=session.raw_data_cache,
            params=ai_params,
            default_max_workers=session.default_max_workers,
            executor_bundle=prep_executor_bundle,
            static_fast_cache=session.static_fast_cache,
            static_master_dates=session.master_dates,
            include_trade_logs=False,
            include_pit_stats_index=True,
        )
        cache_prep = getattr(session, "cache_prepared_trial_inputs", None)
        if callable(cache_prep):
            cache_prep(prep_cache_key, prep_result)
    trial.set_user_attr("prep_mode", prep_result["prep_mode"])
    trial.set_user_attr("prep_start_method", prep_result["pool_start_method"] or "default")
    if prep_result["pool_error_text"] is not None:
        trial.set_user_attr("prep_pool_error", prep_result["pool_error_text"])

    prep_failures = prep_result["prep_failures"]
    if prep_failures:
        insufficient_failures = [item for item in prep_failures if is_insufficient_data_message(item[1])]
        session.record_optimizer_prep_failures(insufficient_failures)

    profile_row = build_initial_profile_row(trial.number, prep_result["prep_wall_sec"], prep_result["prep_profile"])
    search_scope = resolve_search_train_scope(session, prep_result["master_dates"])
    mode = str(search_scope["mode"])
    profile_row["objective_mode"] = mode
    profile_row["search_train_end_year"] = int(search_scope["effective_search_train_end_year"])
    profile_row["sort_dates_sec"] = float(search_scope.get("sort_dates_sec", 0.0))
    profile_row["search_train_date_count"] = int(len(search_scope["search_train_dates"]))
    trial.set_user_attr("objective_mode", mode)
    trial.set_user_attr("search_train_end_year", int(search_scope["effective_search_train_end_year"]))
    trial.set_user_attr("search_train_date_count", int(len(search_scope["search_train_dates"])))
    if search_scope.get("fail_reason") is not None:
        return _append_invalid_profile_row(
            session=session,
            trial=trial,
            profile_row=profile_row,
            fail_reason=str(search_scope["fail_reason"]),
            objective_start=objective_start,
        )

    full_evaluation_cache_key = build_full_evaluation_cache_key(
        ai_params,
        objective_mode=mode,
        train_start_year=session.train_start_year,
        search_train_end_year=int(search_scope["effective_search_train_end_year"]),
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
    )
    get_cached_evaluation = getattr(session, "get_full_evaluation_from_cache", None)
    evaluation = get_cached_evaluation(full_evaluation_cache_key) if callable(get_cached_evaluation) else None

    pf_profile = {}
    portfolio_start = time.perf_counter()
    score_start = time.perf_counter()
    if evaluation is None:
        evaluation = evaluate_prepared_train_score(
            session,
            ai_params=ai_params,
            prep_result=prep_result,
            search_scope=search_scope,
            profile_stats=pf_profile,
        )
        cache_evaluation = getattr(session, "cache_full_evaluation", None)
        if callable(cache_evaluation):
            cache_evaluation(full_evaluation_cache_key, evaluation)
    profile_row["score_calc_sec"] = time.perf_counter() - score_start
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
    profile_row["ret_pct"] = float(evaluation.get("ret_pct", 0.0))
    profile_row["mdd"] = float(evaluation.get("mdd", 0.0))
    profile_row["trade_count"] = int(evaluation.get("trade_count", 0))
    profile_row["annual_return_pct"] = float(evaluation.get("annual_return_pct", 0.0))
    profile_row["annual_trades"] = float(evaluation.get("annual_trades", 0.0))
    profile_row["reserved_buy_fill_rate"] = float(evaluation.get("reserved_buy_fill_rate", 0.0))
    profile_row["full_year_count"] = int(evaluation.get("full_year_count", 0))
    profile_row["min_full_year_return_pct"] = float(evaluation.get("min_full_year_return_pct", 0.0))
    profile_row["m_win_rate"] = float(evaluation.get("m_win_rate", 0.0))
    profile_row["r_squared"] = float(evaluation.get("r_squared", 0.0))
    profile_row["base_score"] = float(evaluation.get("base_score", INVALID_TRIAL_VALUE))
    if not evaluation["ok"]:
        return _append_invalid_profile_row(
            session=session,
            trial=trial,
            profile_row=profile_row,
            fail_reason=str(evaluation["fail_reason"]),
            objective_start=objective_start,
        )

    trial.set_user_attr("pf_return", evaluation["ret_pct"])
    trial.set_user_attr("pf_mdd", evaluation["mdd"])
    trial.set_user_attr("pf_trades", evaluation["trade_count"])
    trial.set_user_attr("final_equity", evaluation["final_equity"])
    trial.set_user_attr("avg_exposure", evaluation["avg_exposure"])
    trial.set_user_attr("max_exposure", evaluation["max_exposure"])
    trial.set_user_attr("bm_return", evaluation["bm_return"])
    trial.set_user_attr("bm_mdd", evaluation["bm_mdd"])
    trial.set_user_attr("win_rate", evaluation["win_rate"])
    trial.set_user_attr("pf_ev", evaluation["pf_ev"])
    trial.set_user_attr("pf_payoff", evaluation["pf_payoff"])
    trial.set_user_attr("missed_buys", evaluation["missed_buys"])
    trial.set_user_attr("missed_sells", evaluation["missed_sells"])
    trial.set_user_attr("normal_trades", evaluation["normal_trades"])
    trial.set_user_attr("extended_trades", evaluation["extended_trades"])
    trial.set_user_attr("annual_trades", evaluation["annual_trades"])
    trial.set_user_attr("reserved_buy_fill_rate", evaluation["reserved_buy_fill_rate"])
    trial.set_user_attr("annual_return_pct", evaluation["annual_return_pct"])
    trial.set_user_attr("bm_annual_return_pct", evaluation["bm_annual_return_pct"])
    trial.set_user_attr("full_year_count", evaluation["full_year_count"])
    trial.set_user_attr("min_full_year_return_pct", evaluation["min_full_year_return_pct"])
    trial.set_user_attr("yearly_return_rows", evaluation["yearly_return_rows"])
    trial.set_user_attr("base_score", evaluation["base_score"])
    trial.set_user_attr("bm_min_full_year_return_pct", evaluation["bm_min_full_year_return_pct"])
    trial.set_user_attr("r_squared", evaluation["r_squared"])
    trial.set_user_attr("m_win_rate", evaluation["m_win_rate"])
    trial.set_user_attr("bm_r_squared", evaluation["bm_r_squared"])
    trial.set_user_attr("bm_m_win_rate", evaluation["bm_m_win_rate"])

    cache_trial_milestone_inputs = getattr(session, "cache_trial_milestone_inputs", None)
    if callable(cache_trial_milestone_inputs):
        cache_trial_milestone_inputs(
            trial.number,
            sorted_master_dates=sorted(prep_result["master_dates"]),
            all_pit_stats_index=prep_result.get("all_pit_stats_index"),
            all_dfs_fast=prep_result.get("all_dfs_fast"),
        )

    final_score = float(evaluation["base_score"])
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
