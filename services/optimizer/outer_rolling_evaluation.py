from __future__ import annotations

import pandas as pd

from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS
from core.params_io import build_params_from_mapping
from core.portfolio_stats import calc_plain_romd
from core.seed_ensemble_policy import normalize_seed_ensemble_members
from core.strategy_params import build_runtime_param_raw_value
from services.optimizer.outer_rolling_active_replay import _extract_active_replay_metrics
from services.optimizer.outer_rolling_params import (
    _materialize_fixed_strategy_param_overrides_in_members,
    build_effective_trial_params_payload,
)
from services.optimizer.outer_rolling_policy import _is_policy_ensemble_item
from services.optimizer.outer_rolling_policy_replay import (
    _build_single_period_ensemble_payload,
    _policy_metrics_from_ensemble_metrics,
)
from services.optimizer.param_cache import build_prep_cache_key
from services.optimizer.prep import prepare_trial_inputs
from services.optimizer.walk_forward import evaluate_walk_forward

def _prep_result_has_pit_index(prep_result) -> bool:
    if not isinstance(prep_result, dict):
        return False
    pit_stats_index = prep_result.get("all_pit_stats_index")
    return isinstance(pit_stats_index, dict) and bool(pit_stats_index)

def _get_or_prepare_oos_inputs(*, session, params):
    # AI註: OOS diagnostics only needs dynamic data + PIT stats index for
    # portfolio replay.  Standalone trade logs are unnecessary when the PIT
    # index is already available, so reuse the normal optimizer prep cache.
    prep_cache_key = build_prep_cache_key(params)
    get_cached_prep = getattr(session, "get_prepared_trial_inputs_from_cache", None)
    prep_result = get_cached_prep(prep_cache_key) if callable(get_cached_prep) else None
    if _prep_result_has_pit_index(prep_result):
        return prep_result

    prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(params, "optimizer_max_workers"))
    prep_result = prepare_trial_inputs(
        raw_data_cache=session.raw_data_cache,
        params=params,
        default_max_workers=session.default_max_workers,
        executor_bundle=prep_executor_bundle,
        static_fast_cache=session.static_fast_cache,
        static_master_dates=session.master_dates,
        include_trade_logs=False,
        include_pit_stats_index=True,
        profile_enabled=False,
    )
    cache_prep = getattr(session, "cache_prepared_trial_inputs", None)
    if callable(cache_prep):
        cache_prep(prep_cache_key, prep_result)
    return prep_result

def _evaluate_period_oos(*, session, trial, oos_year: int, include_equity_curve: bool = False, oos_start_date: str | None = None, oos_end_date: str | None = None):
    payload = build_effective_trial_params_payload(session=session, trial=trial)
    params = build_params_from_mapping(payload)
    prep_result = _get_or_prepare_oos_inputs(session=session, params=params)
    all_dates = sorted(prep_result["master_dates"])
    policy = dict(getattr(session, "walk_forward_policy", {}) or {})
    start_text = str(oos_start_date or policy.get("oos_start_date") or f"{int(str(oos_year)[:4])}-01-01")
    end_text = str(oos_end_date or policy.get("oos_end_date") or f"{int(str(oos_year)[:4])}-12-31")
    oos_start = pd.Timestamp(start_text).normalize()
    oos_end = pd.Timestamp(end_text).normalize()
    test_dates = [dt for dt in all_dates if oos_start <= pd.Timestamp(dt).normalize() <= oos_end]
    if not test_dates:
        raise RuntimeError(f"OOS {start_text}~{end_text} 無有效交易日期")
    train_start_text = str(policy.get("train_start_date") or f"{int(session.train_start_year)}-01-01")
    train_end_text = str(policy.get("search_train_end_date") or (oos_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    holdout_period = {
        "label": f"OOS-{start_text}~{end_text}",
        "train_start": train_start_text,
        "train_end": train_end_text,
        "oos_start": pd.Timestamp(test_dates[0]).strftime("%Y-%m-%d"),
        "oos_end": pd.Timestamp(test_dates[-1]).strftime("%Y-%m-%d"),
        "test_dates": test_dates,
    }
    return evaluate_walk_forward(
        all_dfs_fast=prep_result["all_dfs_fast"],
        all_trade_logs=prep_result["all_trade_logs"],
        sorted_dates=all_dates,
        params=params,
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
        benchmark_ticker="0050",
        train_start_year=int(pd.Timestamp(train_start_text).year),
        min_train_years=int(getattr(session, "walk_forward_policy", {}).get("min_train_years", 1) or 1),
        oos_start_year=int(oos_start.year),
        pit_stats_index=prep_result.get("all_pit_stats_index"),
        holdout_period=holdout_period,
        include_equity_curve=bool(include_equity_curve),
    )

def _extract_period_metrics(report: dict) -> dict:
    period = dict(report.get("period") or {})
    return {
        "oos_score": float(period.get("test_score_romd", 0.0)),
        "oos_plain_romd_score": calc_plain_romd(float(period.get("ret_pct", 0.0)), float(period.get("mdd", 0.0))),
        "ret_pct": float(period.get("ret_pct", 0.0)),
        "mdd_pct": float(period.get("mdd", 0.0)),
        "trades": int(period.get("trade_count", 0) or 0),
        "portfolio_total_r": float(period.get("portfolio_total_r", 0.0)),
        "portfolio_median_r": float(period.get("portfolio_median_r", 0.0)),
        "score_total_r": float(period.get("score_total_r", period.get("single_stock_total_r", 0.0)) or 0.0),
        "score_median_r": float(period.get("score_median_r", period.get("single_stock_median_r", 0.0)) or 0.0),
        "score_r_source": str(period.get("score_r_source", "single_stock")),
        "benchmark_oos_score": float(period.get("benchmark_score_romd", 0.0)),
        "benchmark_return_pct": float(period.get("benchmark_return_pct", 0.0)),
        "benchmark_mdd_pct": float(period.get("benchmark_mdd", 0.0)),
        "initial_capital": float(period.get("initial_capital", 0.0)),
        "equity_curve": list(period.get("equity_curve") or []),
    }

def _evaluate_finalist_ensemble_oos_metrics(*, session, item: dict, policy_name: str, oos_year: int, oos_start_date: str | None = None, oos_end_date: str | None = None) -> dict:
    from services.portfolio_replay import run_portfolio_simulation_with_param_ensemble

    members = _materialize_fixed_strategy_param_overrides_in_members(
        item.get("params_ensemble"),
        getattr(session, "fixed_strategy_param_overrides", None),
    )
    if not members:
        return {}
    data_dir = getattr(session, "raw_data_cache_data_dir", None)
    if not data_dir:
        raise RuntimeError("session 尚未載入 data_dir，無法建立 finalist agree OOS diagnostics")
    start_text = str(oos_start_date or f"{int(str(oos_year)[:4])}-01-01")
    end_text = str(oos_end_date or f"{int(str(oos_year)[:4])}-12-31")
    payload = _build_single_period_ensemble_payload(
        members=members,
        effective_start=start_text,
        effective_end=end_text,
        oos_year=int(oos_year),
        policy_name=str(policy_name),
        raw_universe_required_min_rows=getattr(session, "raw_data_cache_required_min_rows", None),
    )
    result = run_portfolio_simulation_with_param_ensemble(
        str(data_dir),
        payload,
        max_positions=int(getattr(session, "train_max_positions", DEFAULT_PORTFOLIO_MAX_POSITIONS) or DEFAULT_PORTFOLIO_MAX_POSITIONS),
        enable_rotation=bool(getattr(session, "train_enable_rotation", False)),
        start_year=int(pd.Timestamp(start_text).year),
        end_year=int(pd.Timestamp(end_text).year),
        start_date=start_text,
        end_date=end_text,
        benchmark_ticker="0050",
        verbose=False,
        use_prepared_cache=False,
        write_prepared_cache=False,
    )
    return _extract_active_replay_metrics(result)

def _evaluate_finalist_oos_diagnostics(*, session, finalists: list[dict], policy_items: dict[str, dict | None], oos_year: int, oos_start_date: str | None = None, oos_end_date: str | None = None) -> dict:
    best_score = float("-inf")
    best_trial_number = None
    best_trial = None
    best_metrics: dict = {}
    report_cache: dict[int, dict] = {}
    metrics_by_trial: dict[int, dict] = {}
    benchmark_score = 0.0
    benchmark_return_pct = 0.0
    benchmark_mdd_pct = 0.0
    curve_trial_numbers = {
        int(item["trial"].number)
        for item in dict(policy_items or {}).values()
        if item is not None and item.get("trial") is not None and not _is_policy_ensemble_item(item)
    }

    def _load_trial_metrics(trial, *, include_equity_curve: bool) -> dict:
        trial_number = int(trial.number)
        cached_entry = report_cache.get(trial_number)
        report = None
        if cached_entry is not None:
            cached_has_curve = bool(cached_entry.get("include_equity_curve", False))
            if cached_has_curve or not bool(include_equity_curve):
                report = cached_entry.get("report")
        if report is None:
            report = _evaluate_period_oos(
                session=session,
                trial=trial,
                oos_year=int(oos_year),
                include_equity_curve=bool(include_equity_curve),
                oos_start_date=oos_start_date,
                oos_end_date=oos_end_date,
            )
            report_cache[trial_number] = {
                "include_equity_curve": bool(include_equity_curve),
                "report": report,
            }
        metrics = _extract_period_metrics(report)
        if bool(include_equity_curve) or trial_number not in metrics_by_trial:
            metrics_by_trial[trial_number] = dict(metrics)
        return dict(metrics)

    for item in list(finalists or []):
        trial = item.get("trial")
        if trial is None:
            continue
        trial_number = int(trial.number)
        metrics = _load_trial_metrics(trial, include_equity_curve=trial_number in curve_trial_numbers)
        benchmark_score = float(metrics.get("benchmark_oos_score", benchmark_score))
        benchmark_return_pct = float(metrics.get("benchmark_return_pct", benchmark_return_pct))
        benchmark_mdd_pct = float(metrics.get("benchmark_mdd_pct", benchmark_mdd_pct))
        score = float(metrics["oos_score"])
        if score > best_score:
            best_score = score
            best_trial_number = trial_number
            best_trial = trial
            best_metrics = dict(metrics)
    if best_score == float("-inf"):
        best_score = 0.0
        best_metrics = {}
    elif best_trial is not None:
        best_metrics = _load_trial_metrics(best_trial, include_equity_curve=True)
    policies: dict[str, dict] = {}
    for policy_name, item in dict(policy_items or {}).items():
        if item is not None and _is_policy_ensemble_item(item):
            ensemble_metrics = _evaluate_finalist_ensemble_oos_metrics(
                session=session,
                item=item,
                policy_name=str(policy_name),
                oos_year=int(oos_year),
                oos_start_date=oos_start_date,
                oos_end_date=oos_end_date,
            )
            benchmark_score = float(ensemble_metrics.get("benchmark_oos_score", benchmark_score))
            benchmark_return_pct = float(ensemble_metrics.get("benchmark_return_pct", benchmark_return_pct))
            benchmark_mdd_pct = float(ensemble_metrics.get("benchmark_mdd_pct", benchmark_mdd_pct))
            policy_metrics = _policy_metrics_from_ensemble_metrics(
                ensemble_metrics,
                best_score=float(best_score),
                benchmark_score=float(benchmark_score),
            )
            policy_metrics["member_count"] = int(item.get("member_count") or len(normalize_seed_ensemble_members(item.get("params_ensemble"))))
            policy_metrics["min_agree"] = int(item.get("min_agree") or 1)
            policies[policy_name] = policy_metrics
            continue
        if item is None or item.get("trial") is None:
            policies[policy_name] = {
                "available": False,
                "rank_1_trial": None,
                "rank_1_oos": 0.0,
                "rank_1_plain_romd": 0.0,
                "rank_1_return_pct": 0.0,
                "rank_1_mdd_pct": 0.0,
                "rank_1_trades": 0,
                "best_gap": 0.0 - float(best_score),
                "benchmark_0050_gap": 0.0 - float(benchmark_score),
                "rank_1_initial_capital": 0.0,
                "rank_1_equity_curve": [],
            }
            continue
        trial_number = int(item["trial"].number)
        metrics = _load_trial_metrics(item["trial"], include_equity_curve=True)
        rank_1_oos = float(metrics.get("oos_score", 0.0))
        rank_1_plain_romd = float(metrics.get("oos_plain_romd_score", calc_plain_romd(metrics.get("ret_pct", 0.0), metrics.get("mdd_pct", 0.0))))
        benchmark_score = float(metrics.get("benchmark_oos_score", benchmark_score))
        benchmark_return_pct = float(metrics.get("benchmark_return_pct", benchmark_return_pct))
        benchmark_mdd_pct = float(metrics.get("benchmark_mdd_pct", benchmark_mdd_pct))
        policies[policy_name] = {
            "available": True,
            "rank_1_trial": trial_number + 1,
            "rank_1_oos": rank_1_oos,
            "rank_1_plain_romd": rank_1_plain_romd,
            "rank_1_return_pct": float(metrics.get("ret_pct", 0.0)),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_trades": int(metrics.get("trades", 0) or 0),
            "rank_1_total_r": float(metrics.get("portfolio_total_r", 0.0)),
            "rank_1_median_r": float(metrics.get("portfolio_median_r", 0.0)),
            "rank_1_score_total_r": float(metrics.get("score_total_r", 0.0)),
            "rank_1_score_median_r": float(metrics.get("score_median_r", 0.0)),
            "rank_1_score_r_source": str(metrics.get("score_r_source", "single_stock")),
            "best_gap": rank_1_oos - float(best_score),
            "benchmark_0050_gap": rank_1_oos - float(benchmark_score),
            "benchmark_0050_plain_romd_gap": rank_1_plain_romd - float(benchmark_score),
            "rank_1_initial_capital": float(metrics.get("initial_capital", 0.0)),
            "rank_1_equity_curve": list(metrics.get("equity_curve") or []),
        }
    return {
        "best_finalist_oos_score": float(best_score),
        "best_finalist_plain_romd_score": float(best_metrics.get("oos_plain_romd_score", calc_plain_romd(best_metrics.get("ret_pct", 0.0), best_metrics.get("mdd_pct", 0.0)))) if best_metrics else 0.0,
        "best_finalist_trial": int(best_trial_number) + 1 if best_trial_number is not None else None,
        "best_finalist_return_pct": float(best_metrics.get("ret_pct", 0.0)),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "best_finalist_trades": int(best_metrics.get("trades", 0) or 0),
        "best_finalist_total_r": float(best_metrics.get("portfolio_total_r", 0.0)),
        "best_finalist_median_r": float(best_metrics.get("portfolio_median_r", 0.0)),
        "best_finalist_score_total_r": float(best_metrics.get("score_total_r", 0.0)),
        "best_finalist_score_median_r": float(best_metrics.get("score_median_r", 0.0)),
        "best_finalist_score_r_source": str(best_metrics.get("score_r_source", "single_stock")),
        "best_finalist_initial_capital": float(best_metrics.get("initial_capital", 0.0)),
        "best_finalist_equity_curve": list(best_metrics.get("equity_curve") or []),
        "best_finalist_params": (
            build_effective_trial_params_payload(session=session, trial=best_trial)
            if best_trial is not None
            else {}
        ),
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(benchmark_return_pct),
        "benchmark_mdd_pct": float(benchmark_mdd_pct),
        "policies": policies,
    }

