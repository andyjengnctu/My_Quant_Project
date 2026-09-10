from __future__ import annotations

from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS

import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from contextlib import redirect_stderr, redirect_stdout
from collections import OrderedDict

PARALLEL_FOLD_HEARTBEAT_INTERVAL_SEC = 2.0

import pandas as pd

from config.training_policy import (
    OPTIMIZER_FIXED_TP_PERCENT,
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
    OUTER_ROLLING_OOS_HORIZON_MONTHS,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)
from core.training_policy import (
    is_optimizer_local_min_review_enabled,
)
from core.training_performance import (
    is_optimizer_active_replay_include_pit_stats_index_enabled,
    is_optimizer_active_replay_include_trade_logs_enabled,
    is_optimizer_active_replay_use_prepared_cache_enabled,
    is_optimizer_active_replay_write_prepared_cache_enabled,
    resolve_optimizer_active_replay_prep_workers,
    resolve_optimizer_random_seed_ensemble_parallel_backend_default,
    resolve_optimizer_random_seed_ensemble_parallel_workers_default,
    is_optimizer_policy_replay_context_reuse_enabled_default,
    is_optimizer_policy_replay_dedup_by_signature_enabled_default,
    resolve_optimizer_policy_replay_parallel_backend_default,
    resolve_optimizer_policy_replay_parallel_workers_default,
)
from services.optimizer.outer_rolling_performance import (
    _ResourceUsageSampler,
    _apply_outer_rolling_resource_env_defaults,
    _build_training_performance_alignment_rows,
    _env_value_for_display,
    _format_parallel_settings_line,
    _is_outer_rolling_sqlite_storage_enabled,
    _is_rolling_fold_parallel_enabled,
    _mark_resource_probe_fallback,
    _resolve_parallel_worker_prep_cache_max_items,
    _resolve_resource_sample_interval_sec,
    _resolve_rolling_fold_workers,
    _resolve_rolling_shared_prep_cache_max_items,
    _resolve_single_fold_search_parallel_trials,
    _shutdown_rolling_shared_prep_executor_holder,
)
from services.optimizer.outer_rolling_formatting import (
    OOS_SCORE_DECIMALS,
    _color_numeric_text,
    _display_compact_month_period,
    _display_month_period,
    _display_month_value,
    _display_short_date_period,
    _display_short_date_value,
    _extract_cli_value,
    _fmt_duration,
    _fmt_duration_compact,
    _format_compare,
    _format_compare_plain,
    _format_plain_score,
    _format_system_score_compare_plain,
    _has_cli_flag,
    _month_start,
    _pad_ansi,
    _parse_oos_boundary,
    _period_key,
    _period_label,
    _prompt_int,
    _safe_float,
    _short_month_period_end,
    _strip_ansi,
    _timestamp_or_none,
    _visible_len,
    format_optimizer_output_file_lines,
    print_optimizer_output_files,
)
from services.optimizer.outer_rolling_search_progress import _SearchProgress
from services.optimizer.outer_rolling_progress import (
    PARALLEL_FOLD_PROGRESS_PREFIX,
    _collect_seed_progress_phase_metrics,
    _count_local_min_completed_from_progress,
    _count_local_min_completed_neighbors_from_progress,
    _format_parallel_fold_progress_line,
    _format_seed_ensemble_progress_line,
    _optimizer_resource_usage_suffix,
    _parallel_fold_seed_log_paths,
    _read_latest_parallel_fold_seed_progresses,
    _read_latest_parallel_fold_seed_progresses_for_task,
    _safe_progress_float,
    _safe_progress_json_loads,
    _safe_progress_ts,
    _seed_progress_context_key,
    _write_parallel_fold_progress_event,
    build_optimizer_seed_ensemble_live_lines,
    format_optimizer_final_performance_summary,
    format_optimizer_seed_ensemble_progress_header,
    read_optimizer_seed_progresses_from_log_paths,
    render_optimizer_fold_progress_line,
    render_optimizer_seed_progress_line,
    write_optimizer_seed_progress_event,
    _compact_policy_for_live_result,
    _compact_row_for_live_result,
    _seed_progress_context_from_task,
    _FoldLogSearchProgress,
)
from services.optimizer.outer_rolling_fold_context import (
    _fold_label,
    _fold_label_display,
    _fold_selection_label,
    _fold_selection_label_display,
    build_optimizer_seed_ensemble_fold_context,
    normalize_optimizer_seed_ensemble_fold_row,
    normalize_optimizer_seed_ensemble_fold_rows,
    optimizer_seed_ensemble_row_sort_key,
)
from services.optimizer.outer_rolling_timing import (
    _build_outer_timing_row,
    _merge_compact_timing_count_text,
    _parse_compact_count_text,
    _profile_avg_float,
    _sum_int_timing_rows,
    _sum_timing_rows,
    _write_outer_timing_summary,
)
from services.optimizer.outer_rolling_runtime import (
    _apply_outer_rolling_process_environ,
    _ensure_study_runtime_identity_compatible,
    _format_exception_summary,
    _is_non_retryable_fold_failure,
    _optimizer_runtime_context_spec,
    _outer_rolling_db_member_suffix,
    _outer_rolling_runtime_identity,
    _remaining_optimizer_trials,
    _resolve_outer_rolling_db_file,
    _tail_text_file,
    _validate_optimizer_runtime_context,
)
from services.optimizer.outer_rolling_parallel_progress import (
    PARALLEL_FOLD_LOG_STATUS_MAX_CHARS,
    _ParallelFoldProgressLogFilter,
    _build_parallel_fold_failure_message,
    _collect_parallel_fold_replay_phase_metrics,
    _latest_parallel_fold_log_status,
    _merge_live_result_rows,
    _parallel_fold_log_path_for_fallback,
    _read_latest_parallel_fold_progress,
    _read_parallel_fold_replay_phase_metrics,
    _read_parallel_fold_result_row,
)
from services.optimizer.outer_rolling_params import (
    _materialize_fixed_strategy_param_overrides_in_members,
    build_effective_trial_params_payload,
    materialize_fixed_strategy_param_overrides_in_active_param_payload,
    materialize_fixed_strategy_param_overrides_in_payload,
)
from services.optimizer.outer_rolling_policy import (
    ALL_REPORT_POLICY_NAMES,
    BASE_FINALISTS_AGREE_POLICY_NAME,
    BASE_FINALIST_BEST_POLICY_NAME,
    BASE_RETENTION_COMPARISON_POLICY_LABELS,
    BASE_RETENTION_COMPARISON_POLICY_NAMES,
    BASE_RETENTION_COMPARISON_POLICY_THRESHOLDS,
    BASE_RETENTION_COMPARISON_THRESHOLDS,
    CHAIN_POLICY_NAMES,
    FINALISTS_AGREE_POLICY_NAMES,
    FINALISTS_AGREE_RESULT_POLICY_NAMES,
    FINALISTS_AGREE_TABLE_TITLE,
    FINALIST_BEST_POLICY_NAMES,
    FINALIST_BEST_RESULT_POLICY_NAMES,
    FINALIST_BEST_TABLE_TITLE,
    LOCAL_FINALISTS_AGREE_POLICY_NAME,
    LOCAL_FINALIST_BEST_POLICY_NAME,
    LOCAL_DEPENDENT_POLICY_NAMES,
    NONROLLING_PARAMSET_FILENAME_BY_POLICY,
    NONROLLING_PARAMSET_FILENAME_PREFIX_BY_MODE,
    PARAMSET_FILENAME_BY_POLICY,
    POLICY_OUTPUT_LABELS,
    REPORT_POLICY_LABELS,
    REPORT_POLICY_NAMES,
    RETENTION_FINALISTS_AGREE_POLICY_NAME,
    RETENTION_FINALIST_BEST_POLICY_NAME,
    SEED_ENSEMBLE_OOS_TABLE_TITLE,
    SEED_ENSEMBLE_POLICY_NAMES,
    SEED_ENSEMBLE_RESULTS_TABLE_TITLE,
    SEED_ENSEMBLE_RESULT_POLICY_NAMES,
    SEED_ENSEMBLE_RETENTION_TABLE_TITLE,
    STALE_POLICY_PARAMSET_FILENAMES,
    _active_optimizer_table_titles,
    _build_finalists_agree_member_payloads,
    _build_local_rank_map,
    _build_policy_items,
    _build_policy_schedule_entry,
    _build_retention_rank_map,
    _finalist_best_policy_sort_key,
    _finalists_agree_member_order_key,
    _finalists_agree_metadata,
    _finalists_agree_metadata_keys,
    _finalists_agree_policy_config,
    _finalists_agree_seed_group_sort_key,
    _is_finalist_best_policy,
    _is_finalists_agree_policy,
    _is_local_finalist_best_policy,
    _is_local_finalists_agree_policy,
    _is_policy_ensemble_item,
    _is_retention_finalist_best_policy,
    _is_retention_finalists_agree_policy,
    _is_rolling_random_seed_ensemble_enabled,
    _policy_is_available,
    _policy_plain_romd_score,
    _rank_base_finalist_items,
    _rank_local_finalist_items,
    _rank_retention_finalist_items,
    _resolve_finalists_agree_min_agree,
    _resolve_report_policy_names,
    _safe_float_for_finalists_agree_sort,
    _safe_int_for_finalists_agree_sort,
    _select_base_finalists_agree_item,
    _select_base_rank1_item,
    _select_finalists_agree_item,
    _select_local_finalists_agree_item,
    _select_local_rank1_item,
    _select_retention_finalists_agree_item,
    _select_retention_rank1_item,
    _select_winner,
    build_optimizer_policy_members_from_finalists,
    get_optimizer_nonrolling_policy_paramset_filename,
    get_optimizer_paramset_policy_names,
    get_optimizer_policy_output_label,
    get_optimizer_policy_paramset_filename,
    optimizer_seed_ensemble_table_titles,
    select_base_finalists_agree_members,
    select_finalist_best_members,
    select_finalists_agree_members,
    select_local_finalists_agree_members,
    select_retention_finalists_agree_members,
)
from services.optimizer.outer_rolling_plan import (
    OuterRollingConfig,
    _build_rolling_folds,
    _confirm_plan,
    _print_plan,
    _resolve_config,
    _resolve_latest_date_from_csv_data_dir,
)
from services.optimizer.outer_rolling_curve_metrics import (
    _calc_full_month_return_metrics_from_curve,
    _calc_full_quarter_return_metrics_from_curve,
    _calc_full_year_return_metrics_from_curve,
    _calc_stitched_curve_metrics,
    _derive_curve_initial_capital,
    _month_end_equities_from_curve,
    _normalize_equity_curve_rows,
    _stitch_benchmark_equity_curve,
    _stitch_strategy_equity_curves,
)
from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
    ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
    get_active_param_ensemble_policy,
    is_active_param_ensemble_payload,
)
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RESET, C_YELLOW
from core.file_integrity import atomic_write_json
from core.params_io import build_params_from_mapping
from core.model_paths import resolve_models_dir
from core.portfolio_stats import calc_plain_romd, calc_portfolio_score
from core.portfolio_param_runtime import (
    build_active_param_objects_from_payload,
    build_active_param_ensemble_objects_from_payload,
)
from core.raw_universe_contract import (
    RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD,
    build_raw_universe_contract_fields as _raw_universe_contract_fields,
    coerce_raw_universe_required_min_rows,
    resolve_raw_universe_required_min_rows,
)
from core.runtime_utils import (
    get_process_pool_executor_kwargs,
    get_taipei_now,
    stdout_supports_inline_progress,
    write_inline_progress,
)
from core.rolling_oos_params import ROLLING_OOS_PARAM_SET_SCHEMA_TYPE, ROLLING_OOS_USAGE
from core.seed_ensemble_policy import (
    build_seed_ensemble_policy_snapshot,
    generate_random_seed_ensemble,
    normalize_seed_ensemble_members,
    renumber_seed_ensemble_members,
)
from core.strategy_params import build_runtime_param_raw_value
from core.strategy_param_artifacts import normalize_strategy_param_payload_for_persistence
from core.walk_forward_policy import build_optimizer_runtime_policy
from services.optimizer.param_cache import build_prep_cache_key
from services.optimizer.prep import prepare_trial_inputs
from services.optimizer.robustness import (
    list_local_min_score_finalists,
)
from services.optimizer.score_display import format_optimizer_score_for_display
from services.optimizer.study_utils import (
    INVALID_TRIAL_VALUE,
)
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








def _build_active_param_replay_payload_from_rows(rows: list[dict], *, policy_name: str | None = None, best_finalist: bool = False, raw_universe_required_min_rows=None) -> dict:
    params_by_oos_year: dict[str, dict] = {}
    params_by_effective_date: dict[str, dict] = {}
    params_ensemble_by_effective_date: dict[str, list[dict]] = {}
    fold_entries: list[dict] = []
    has_ensemble_members = False
    for row in sorted((normalize_optimizer_seed_ensemble_fold_row(item) for item in list(rows or [])), key=optimizer_seed_ensemble_row_sort_key):
        try:
            oos_key = int(row.get("oos_year"))
        except (TypeError, ValueError):
            continue
        effective_start = str(row.get("oos_start_date") or f"{str(oos_key)[:4]}-01-01")
        effective_end = str(row.get("oos_end_date") or f"{str(oos_key)[:4]}-12-31")
        if best_finalist:
            params_payload = dict(row.get("best_finalist_params") or {})
            members = renumber_seed_ensemble_members(row.get("best_finalist_params_ensemble"))
        else:
            schedule = dict((row.get("policy_schedules") or {}).get(str(policy_name)) or {})
            params_payload = dict(schedule.get("params") or {})
            effective_start = str(schedule.get("effective_start") or effective_start)
            effective_end = str(schedule.get("effective_end") or effective_end)
            members = renumber_seed_ensemble_members(_build_params_ensemble_members_for_schedule(schedule))
        if not params_payload and members:
            params_payload = dict(members[0].get("params") or {})
        if not params_payload:
            continue
        params_by_oos_year[str(oos_key)] = params_payload
        params_by_effective_date[effective_start] = params_payload
        if members:
            params_ensemble_by_effective_date[effective_start] = members
            if len(members) > 1:
                has_ensemble_members = True
        fold_entries.append({
            "oos_year": int(oos_key),
            "effective_start": effective_start,
            "effective_end": effective_end,
            "oos_start_date": str(row.get("oos_start_date") or effective_start),
            "oos_end_date": str(row.get("oos_end_date") or effective_end),
        })
    if has_ensemble_members:
        return {
            **_raw_universe_contract_fields(raw_universe_required_min_rows),
            "schema_type": ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
            "schema_version": 1,
            "mode": ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
            "type": "outer_rolling_oos_param_set",
            "active_param_policy": "daily_active_param_ensemble",
            "random_seed_ensemble": _build_effective_seed_ensemble_policy_payload(params_ensemble_by_effective_date, policy_name=policy_name),
            "params_ensemble_by_effective_date": params_ensemble_by_effective_date,
            "params_by_oos_year": params_by_oos_year,
            "params_by_effective_date": params_by_effective_date,
            "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
            "folds": fold_entries,
        }
    return {
        **_raw_universe_contract_fields(raw_universe_required_min_rows),
        "schema_type": ROLLING_OOS_PARAM_SET_SCHEMA_TYPE,
        "schema_version": 1,
        "usage": ROLLING_OOS_USAGE,
        "type": "outer_rolling_oos_param_set",
        "active_param_policy": "daily_active_param",
        "params_by_oos_year": params_by_oos_year,
        "params_by_effective_date": params_by_effective_date,
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        "folds": fold_entries,
    }


def _active_replay_payload_has_params(payload: dict) -> bool:
    if is_active_param_ensemble_payload(payload):
        return bool(dict(payload.get("params_ensemble_by_effective_date") or {}) or list(payload.get("params_ensemble") or []))
    return bool(dict(payload.get("params_by_effective_date") or {}) or dict(payload.get("params_by_oos_year") or {}))


def _extract_active_replay_metrics(result) -> dict:
    if not isinstance(result, tuple) or len(result) < 25:
        raise RuntimeError("active-param replay 回傳格式不完整，無法建立 OOS_CHAIN")
    ret_pct = float(result[2])
    mdd_pct = float(result[3])
    trade_count = int(result[4] or 0)
    win_rate = float(result[5] or 0.0)
    bm_ret_pct = float(result[11])
    bm_mdd_pct = float(result[12])
    r_squared = float(result[15])
    monthly_win_rate = float(result[16])
    bm_r_squared = float(result[17])
    bm_monthly_win_rate = float(result[18])
    annual_return_pct = float(result[23])
    bm_annual_return_pct = float(result[24])
    profile = dict(result[-1]) if isinstance(result[-1], dict) else {}
    equity_curve_points = len(profile.get("equity_curve") or [])
    full_year_count = int(profile.get("full_year_count", 0) or 0)
    min_full_year_return_pct = float(profile.get("min_full_year_return_pct", 0.0) or 0.0)
    yearly_return_rows = list(profile.get("yearly_return_rows") or [])
    full_month_count = int(profile.get("full_month_count", 0) or 0)
    min_month_return_pct = float(profile.get("min_month_return_pct", 0.0) or 0.0)
    monthly_return_rows = list(profile.get("monthly_return_rows") or [])
    full_quarter_count = int(profile.get("full_quarter_count", 0) or 0)
    min_quarter_return_pct = float(profile.get("min_quarter_return_pct", 0.0) or 0.0)
    quarterly_return_rows = list(profile.get("quarterly_return_rows") or [])
    benchmark_full_year_count = int(profile.get("bm_full_year_count", 0) or 0)
    benchmark_min_full_year_return_pct = float(profile.get("bm_min_full_year_return_pct", 0.0) or 0.0)
    benchmark_yearly_return_rows = list(profile.get("bm_yearly_return_rows") or [])
    benchmark_full_month_count = int(profile.get("bm_full_month_count", 0) or 0)
    benchmark_min_month_return_pct = float(profile.get("bm_min_month_return_pct", 0.0) or 0.0)
    benchmark_monthly_return_rows = list(profile.get("bm_monthly_return_rows") or [])
    benchmark_full_quarter_count = int(profile.get("bm_full_quarter_count", 0) or 0)
    benchmark_min_quarter_return_pct = float(profile.get("bm_min_quarter_return_pct", 0.0) or 0.0)
    benchmark_quarterly_return_rows = list(profile.get("bm_quarterly_return_rows") or [])
    portfolio_total_r = float(profile.get("portfolio_total_r", 0.0) or 0.0)
    portfolio_median_r = float(profile.get("portfolio_median_r", 0.0) or 0.0)
    score_total_r = float(profile.get("score_total_r", profile.get("single_stock_total_r", 0.0)) or 0.0)
    score_median_r = float(profile.get("score_median_r", profile.get("single_stock_median_r", 0.0)) or 0.0)
    score = calc_portfolio_score(
        ret_pct,
        mdd_pct,
        monthly_win_rate,
        r_squared,
        annual_return_pct=annual_return_pct,
        trade_win_rate_pct=win_rate,
        min_full_year_return_pct=min_full_year_return_pct,
        min_month_return_pct=min_month_return_pct,
        min_quarter_return_pct=min_quarter_return_pct,
        total_r=score_total_r,
        median_r=score_median_r,
    )
    benchmark_score = calc_plain_romd(bm_ret_pct, bm_mdd_pct)
    plain_romd_score = calc_plain_romd(ret_pct, mdd_pct)
    return {
        "score": float(score),
        "plain_romd_score": float(plain_romd_score),
        "return_pct": float(ret_pct),
        "mdd_pct": float(mdd_pct),
        "annual_return_pct": float(annual_return_pct),
        "full_year_count": int(full_year_count),
        "min_full_year_return_pct": float(min_full_year_return_pct),
        "yearly_return_rows": yearly_return_rows,
        "full_month_count": int(full_month_count),
        "min_month_return_pct": float(min_month_return_pct),
        "monthly_return_rows": monthly_return_rows,
        "full_quarter_count": int(full_quarter_count),
        "min_quarter_return_pct": float(min_quarter_return_pct),
        "quarterly_return_rows": quarterly_return_rows,
        "r_squared": float(r_squared),
        "monthly_win_rate": float(monthly_win_rate),
        "trade_count": int(trade_count),
        "win_rate": float(win_rate),
        "portfolio_total_r": float(portfolio_total_r),
        "portfolio_median_r": float(portfolio_median_r),
        "score_total_r": float(score_total_r),
        "score_median_r": float(score_median_r),
        "score_r_source": str(profile.get("score_r_source", "single_stock")),
        "single_stock_total_r": float(profile.get("single_stock_total_r", score_total_r) or 0.0),
        "single_stock_median_r": float(profile.get("single_stock_median_r", score_median_r) or 0.0),
        "curve_points": int(equity_curve_points),
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(bm_ret_pct),
        "benchmark_mdd_pct": float(bm_mdd_pct),
        "benchmark_annual_return_pct": float(bm_annual_return_pct),
        "benchmark_full_year_count": int(benchmark_full_year_count),
        "benchmark_min_full_year_return_pct": float(benchmark_min_full_year_return_pct),
        "benchmark_yearly_return_rows": benchmark_yearly_return_rows,
        "benchmark_full_month_count": int(benchmark_full_month_count),
        "benchmark_min_month_return_pct": float(benchmark_min_month_return_pct),
        "benchmark_monthly_return_rows": benchmark_monthly_return_rows,
        "benchmark_full_quarter_count": int(benchmark_full_quarter_count),
        "benchmark_min_quarter_return_pct": float(benchmark_min_quarter_return_pct),
        "benchmark_quarterly_return_rows": benchmark_quarterly_return_rows,
        "benchmark_r_squared": float(bm_r_squared),
        "benchmark_monthly_win_rate": float(bm_monthly_win_rate),
    }


def _build_active_replay_schedule_records(payload: dict) -> list[dict]:
    if not _active_replay_payload_has_params(payload):
        return []
    raw_universe_required_min_rows = resolve_raw_universe_required_min_rows(payload)
    if is_active_param_ensemble_payload(payload):
        policy = get_active_param_ensemble_policy(payload)
        records = list(build_active_param_ensemble_objects_from_payload(payload, fixed_risk=None))
        for record in records:
            record[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = raw_universe_required_min_rows
            record["ensemble_min_agree"] = int(policy.get("min_agree", 1) or 1)
            record["ensemble_seed_count"] = int(policy.get("seed_count", len(record.get("members") or []) or 1) or 1)
        return records
    records = list(build_active_param_objects_from_payload(payload, fixed_risk=None))
    for record in records:
        record[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = raw_universe_required_min_rows
    return records


def _row_oos_start_date(row: dict) -> str:
    try:
        return str(row.get("oos_start_date") or f"{str(int(row.get('oos_year')) )[:4]}-01-01")
    except (TypeError, ValueError):
        return str(row.get("oos_start_date") or "")


def _policy_has_complete_active_schedule(rows: list[dict], policy_name: str) -> bool:
    source_rows = sorted(
        (normalize_optimizer_seed_ensemble_fold_row(item) for item in list(rows or [])),
        key=optimizer_seed_ensemble_row_sort_key,
    )
    if not source_rows:
        return False
    first_oos_start = None
    active_starts = []
    for row in source_rows:
        oos_start = _row_oos_start_date(row)
        if oos_start:
            try:
                oos_start_ts = pd.Timestamp(oos_start).normalize()
            except (TypeError, ValueError):
                return False
            if first_oos_start is None or oos_start_ts < first_oos_start:
                first_oos_start = oos_start_ts
        schedule = dict((row.get("policy_schedules") or {}).get(str(policy_name)) or {})
        if not dict(schedule.get("params") or {}):
            continue
        effective_start = str(schedule.get("effective_start") or oos_start or "")
        try:
            effective_start_ts = pd.Timestamp(effective_start).normalize()
        except (TypeError, ValueError):
            return False
        if oos_start and effective_start_ts > oos_start_ts:
            return False
        active_starts.append(effective_start_ts)
    if first_oos_start is None or not active_starts:
        return False
    return min(active_starts) <= first_oos_start


def _empty_unavailable_chain_metrics() -> dict:
    return {
        "available": False,
        "score": 0.0,
        "return_pct": 0.0,
        "mdd_pct": 0.0,
        "annual_return_pct": 0.0,
        "full_month_count": 0,
        "min_month_return_pct": 0.0,
        "full_quarter_count": 0,
        "min_quarter_return_pct": 0.0,
        "r_squared": 0.0,
        "monthly_win_rate": 0.0,
        "trade_count": 0,
        "curve_points": 0,
        "benchmark_oos_score": 0.0,
        "benchmark_return_pct": 0.0,
        "benchmark_mdd_pct": 0.0,
        "benchmark_annual_return_pct": 0.0,
        "benchmark_full_month_count": 0,
        "benchmark_min_month_return_pct": 0.0,
        "benchmark_full_quarter_count": 0,
        "benchmark_min_quarter_return_pct": 0.0,
        "benchmark_r_squared": 0.0,
        "benchmark_monthly_win_rate": 0.0,
    }


def _iter_active_replay_context_records(*, policy_name: str, group_records: list[dict]):
    for record in list(group_records or []):
        members = list(record.get("members") or [])
        if members:
            for member in members:
                item = dict(record)
                item.pop("members", None)
                item.update({
                    "params_obj": member.get("params_obj"),
                    "params_signature": member.get("params_signature"),
                    "member_index": member.get("member_index"),
                    "seed": member.get("seed"),
                    "selected_trial": member.get("selected_trial"),
                })
                item["_ensemble_parent_effective_date"] = record.get("effective_date_text")
                yield item
        else:
            yield dict(record)


def _load_active_replay_contexts_by_signature(
    *,
    data_dir: str,
    schedule_groups: dict[str, list[dict]],
    output_dir: str,
    first_year: int | None = None,
    last_year: int | None = None,
    overall_start: float | None = None,
) -> dict[str, dict]:
    from core.data_utils import get_required_min_rows
    from core.portfolio_fast_data import build_normal_setup_index, build_trade_stats_index, pack_static_market_data
    from services.optimizer.raw_cache import load_all_raw_data
    from services.portfolio_replay import load_portfolio_market_context

    contexts_by_signature: dict[str, dict] = {}
    records: list[dict] = []
    by_signature: dict[str, dict] = {}
    for policy_name, group_records in dict(schedule_groups or {}).items():
        for record in _iter_active_replay_context_records(policy_name=str(policy_name), group_records=list(group_records or [])):
            signature = str(record.get("params_signature") or "")
            if not signature:
                continue
            existing = by_signature.get(signature)
            if existing is None:
                existing = dict(record)
                existing["_policies"] = []
                by_signature[signature] = existing
                records.append(existing)
            policies = existing.setdefault("_policies", [])
            if str(policy_name) not in policies:
                policies.append(str(policy_name))

    def _record_year(item: dict) -> int:
        try:
            return int(item.get("year", 0) or 0)
        except (TypeError, ValueError):
            text = str(item.get("effective_date_text") or "")
            try:
                return int(text[:4])
            except (TypeError, ValueError):
                return 0

    records.sort(key=lambda item: (_record_year(item), str(item.get("effective_date_text") or ""), str(item.get("params_signature") or "")))
    total = len(records)
    previous_width = 0
    supports_inline = stdout_supports_inline_progress()
    policy_names = "/".join(sorted({policy for record in records for policy in record.get("_policies", [])})) or "N/A"
    replay_context_start = time.perf_counter()
    years_label = ""
    if first_year and last_year:
        years_label = f" | years={int(first_year)}~{int(last_year)}"

    include_trade_logs = is_optimizer_active_replay_include_trade_logs_enabled()
    include_pit_stats_index = is_optimizer_active_replay_include_pit_stats_index_enabled()
    use_prepared_cache = is_optimizer_active_replay_use_prepared_cache_enabled()
    write_prepared_cache = is_optimizer_active_replay_write_prepared_cache_enabled()

    raw_data_cache = None
    static_fast_cache = None
    static_master_dates = None
    active_prep_workers = None
    if total and not (use_prepared_cache or write_prepared_cache):
        params_required_min_rows = max(get_required_min_rows(record["params_obj"]) for record in records)
        contract_min_rows = max((int(record.get(RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD) or 0) for record in records), default=0)
        required_min_rows = max(int(params_required_min_rows), int(contract_min_rows))
        raw_data_cache = load_all_raw_data(data_dir, required_min_rows, output_dir, verbose=False)
        static_fast_cache = {ticker: pack_static_market_data(df) for ticker, df in raw_data_cache.items()}
        static_master_dates = set()
        for df in raw_data_cache.values():
            static_master_dates.update(df.index)
        active_prep_workers = resolve_optimizer_active_replay_prep_workers(len(raw_data_cache))

    def _covers_text(index: int, item: dict) -> str:
        year = _record_year(item)
        if year <= 0:
            return "N/A"
        next_year = None
        for follow in records[index:]:
            candidate = _record_year(follow)
            if candidate > year:
                next_year = candidate
                break
        end_year = int(last_year) if last_year else year
        if next_year is not None:
            end_year = min(end_year, int(next_year) - 1)
        if end_year <= year:
            return str(year)
        return f"{year}~{end_year}"

    for idx, record in enumerate(records, start=1):
        signature = str(record["params_signature"])
        policies_text = ",".join(record.get("_policies", [])) or "N/A"
        chain_elapsed = max(0.0, time.perf_counter() - replay_context_start)
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        message = (
            f"{C_CYAN}⏱️ OOS_CHAIN active replay context [{idx}/{total}] | "
            f"covers={_covers_text(idx, record)} | policies={policies_text} | effective={record.get('effective_date_text')} | "
            f"signature={signature[:8]}{years_label} | chain={_fmt_duration(chain_elapsed)}{total_elapsed_text}{C_RESET}"
        )
        if supports_inline:
            previous_width = write_inline_progress(message, previous_width=previous_width)
        else:
            print(message)

        if raw_data_cache is not None:
            prep_result = prepare_trial_inputs(
                raw_data_cache=raw_data_cache,
                params=record["params_obj"],
                default_max_workers=int(active_prep_workers or 1),
                static_fast_cache=static_fast_cache,
                static_master_dates=static_master_dates,
                include_trade_logs=bool(include_trade_logs),
                include_pit_stats_index=bool(include_pit_stats_index),
                profile_enabled=False,
            )
            context = {
                "all_dfs_fast": prep_result.get("all_dfs_fast") or {},
                "all_trade_logs": prep_result.get("all_trade_logs") or {},
                "all_pit_stats_index": prep_result.get("all_pit_stats_index") or {},
                "sorted_dates": sorted(prep_result.get("master_dates") or []),
                "prep_wall_sec": float(prep_result.get("prep_wall_sec", 0.0) or 0.0),
                "prep_mode": f"active_replay_shared_raw_{prep_result.get('prep_mode')}",
            }
        else:
            context = load_portfolio_market_context(
                data_dir,
                record["params_obj"],
                verbose=False,
                use_prepared_cache=use_prepared_cache,
                write_prepared_cache=write_prepared_cache,
                raw_universe_required_min_rows=record.get(RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD),
            )
            context = dict(context)

        if not context.get("all_pit_stats_index"):
            if not context.get("all_trade_logs"):
                raise RuntimeError("active replay context 缺少 PIT stats index；請保留 OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX=1")
            context["all_pit_stats_index"] = {
                ticker: build_trade_stats_index(logs)
                for ticker, logs in (context.get("all_trade_logs") or {}).items()
            }
        context["normal_setup_index"] = build_normal_setup_index(context.get("all_dfs_fast") or {})
        contexts_by_signature[signature] = context

    if total:
        chain_elapsed = max(0.0, time.perf_counter() - replay_context_start)
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        summary = (
            f"{C_CYAN}⏱️ OOS_CHAIN active replay context 完成 | "
            f"contexts={total}/{total}{years_label} | policies={policy_names} | "
            f"chain={_fmt_duration(chain_elapsed)}{total_elapsed_text}{C_RESET}"
        )
        if supports_inline:
            write_inline_progress(summary, previous_width=previous_width)
            print()
        else:
            print(summary)
    return contexts_by_signature

def _merge_active_replay_market_dates(contexts_by_signature: dict[str, dict]) -> list:
    market_dates = set()
    for context in dict(contexts_by_signature or {}).values():
        market_dates.update(context.get("sorted_dates") or [])
    return sorted(market_dates)


def _filter_market_dates_by_date_range(market_dates, *, start_date: str | None, end_date: str | None):
    start_ts = pd.Timestamp(start_date).normalize() if start_date else None
    end_ts = pd.Timestamp(end_date).normalize() if end_date else None
    resolved = []
    for raw_date in list(market_dates or []):
        try:
            ts = pd.Timestamp(raw_date).normalize()
        except (TypeError, ValueError):
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        resolved.append(raw_date)
    return sorted(resolved)


def _run_active_replay_metrics_from_schedule_records(
    *,
    schedule_records: list[dict],
    contexts_by_signature: dict[str, dict],
    start_year: int,
    end_year: int,
    max_positions: int,
    enable_rotation: bool,
    benchmark_ticker: str = "0050",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    if not schedule_records:
        return {}
    from core.portfolio_engine import run_portfolio_timeline
    from core.portfolio_stats import find_sim_start_idx
    from services.portfolio_replay import (
        _apply_active_single_stock_score_stats,
        _filter_market_dates_by_end_year,
        _resolve_active_schedule_record,
    )

    all_market_dates = _merge_active_replay_market_dates(contexts_by_signature)
    if start_date or end_date:
        resolved_sorted_dates = _filter_market_dates_by_date_range(
            all_market_dates,
            start_date=start_date or f"{int(start_year)}-01-01",
            end_date=end_date or f"{int(end_year)}-12-31",
        )
        resolved_start_year = int(pd.Timestamp(resolved_sorted_dates[0]).year) if resolved_sorted_dates else int(start_year)
    else:
        resolved_sorted_dates = _filter_market_dates_by_end_year(
            all_market_dates,
            start_year=int(start_year),
            end_year=int(end_year),
        )
        resolved_start_year = int(start_year)
    if not resolved_sorted_dates:
        raise ValueError("active-param replay 沒有可回測日期")
    first_sim_idx = find_sim_start_idx(resolved_sorted_dates, int(resolved_start_year))
    if first_sim_idx >= len(resolved_sorted_dates):
        raise ValueError("active-param replay 起始日期沒有可回測日期")
    first_record = _resolve_active_schedule_record(schedule_records, resolved_sorted_dates[first_sim_idx])
    use_param_ensemble = bool(first_record.get("members"))
    if use_param_ensemble:
        first_members = list(first_record.get("members") or [])
        if not first_members:
            raise ValueError("active-param ensemble schedule 缺少 members")
        base_contexts = [contexts_by_signature[str(member["params_signature"])] for member in first_members]
        base_context = base_contexts[0]
        base_params = first_members[0]["params_obj"]
        benchmark_data = (base_context.get("all_dfs_fast") or {}).get(benchmark_ticker)
        ensemble_min_agree = int(first_record.get("ensemble_min_agree", len(first_members)) or len(first_members))

        def active_param_ensemble_resolver(trade_date):
            return _resolve_active_schedule_record(schedule_records, trade_date)["members"]

        def active_context_ensemble_resolver(trade_date):
            record = _resolve_active_schedule_record(schedule_records, trade_date)
            return [contexts_by_signature[str(member["params_signature"])] for member in list(record.get("members") or [])]

        pf_profile = {
            "param_policy": "active_param_ensemble_replay",
            "capture_equity_curve": True,
            "active_param_ensemble": {
                "seed_count": int(first_record.get("ensemble_seed_count", len(first_members)) or len(first_members)),
                "min_agree": int(ensemble_min_agree),
            },
            "active_param_ensemble_schedule": [
                {
                    "effective_date": str(record.get("effective_date_text")),
                    "year": int(record.get("year", 0) or 0),
                    "member_count": int(len(record.get("members") or [])),
                }
                for record in schedule_records
            ],
        }
        contexts_by_effective_date = {
            str(record.get("effective_date_text")): [
                contexts_by_signature[str(member["params_signature"])]
                for member in list(record.get("members") or [])
            ]
            for record in schedule_records
        }
        _apply_active_single_stock_score_stats(
            pf_profile,
            schedule_records,
            contexts_by_effective_date,
            resolved_sorted_dates,
            ensemble=True,
        )
        result = run_portfolio_timeline(
            base_context.get("all_dfs_fast") or {},
            base_context.get("all_trade_logs") or {},
            resolved_sorted_dates,
            int(resolved_start_year),
            base_params,
            int(max_positions),
            bool(enable_rotation),
            benchmark_ticker=str(benchmark_ticker),
            benchmark_data=benchmark_data,
            is_training=False,
            profile_stats=pf_profile,
            verbose=False,
            pit_stats_index=base_context.get("all_pit_stats_index"),
            active_param_ensemble_resolver=active_param_ensemble_resolver,
            active_context_ensemble_resolver=active_context_ensemble_resolver,
            ensemble_min_agree=int(ensemble_min_agree),
        )
        return _extract_active_replay_metrics((*result, pf_profile))

    base_context = contexts_by_signature[str(first_record["params_signature"])]
    benchmark_data = (base_context.get("all_dfs_fast") or {}).get(benchmark_ticker)

    def active_params_resolver(trade_date):
        return _resolve_active_schedule_record(schedule_records, trade_date)["params_obj"]

    def active_context_resolver(trade_date):
        record = _resolve_active_schedule_record(schedule_records, trade_date)
        return contexts_by_signature[str(record["params_signature"])]

    pf_profile = {
        "param_policy": "active_param_replay",
        "capture_equity_curve": True,
        "active_param_schedule": [
            {
                "effective_date": str(record.get("effective_date_text")),
                "year": int(record.get("year", 0) or 0),
                "params_signature": str(record.get("params_signature")),
            }
            for record in schedule_records
        ],
    }
    contexts_by_effective_date = {
        str(record.get("effective_date_text")): contexts_by_signature[str(record["params_signature"])]
        for record in schedule_records
    }
    _apply_active_single_stock_score_stats(
        pf_profile,
        schedule_records,
        contexts_by_effective_date,
        resolved_sorted_dates,
        ensemble=False,
    )
    result = run_portfolio_timeline(
        base_context.get("all_dfs_fast") or {},
        base_context.get("all_trade_logs") or {},
        resolved_sorted_dates,
        int(resolved_start_year),
        first_record["params_obj"],
        int(max_positions),
        bool(enable_rotation),
        benchmark_ticker=str(benchmark_ticker),
        benchmark_data=benchmark_data,
        is_training=False,
        profile_stats=pf_profile,
        verbose=False,
        pit_stats_index=base_context.get("all_pit_stats_index"),
        active_params_resolver=active_params_resolver,
        active_context_resolver=active_context_resolver,
    )
    return _extract_active_replay_metrics((*result, pf_profile))


def _build_active_replay_chained_oos_summary(
    *,
    rows: list[dict],
    config: OuterRollingConfig,
    selected_data_dir: str,
    output_dir: str,
    max_positions: int,
    enable_rotation: bool,
    overall_start: float | None = None,
) -> dict:
    if not rows:
        return {}
    first_oos_start = min(pd.Timestamp(row.get("oos_start_date") or f"{str(row.get('oos_year'))[:4]}-01-01").normalize() for row in rows)
    last_oos_end = max(pd.Timestamp(row.get("oos_end_date") or f"{str(row.get('oos_year'))[:4]}-12-31").normalize() for row in rows)
    first_year = int(first_oos_start.year)
    last_year = int(last_oos_end.year)
    selection_start_ts = min(pd.Timestamp(row.get("selection_start_date") or f"{int(row.get('selection_start_year', first_year))}-01-01").normalize() for row in rows)
    selection_end_ts = max(pd.Timestamp(row.get("selection_end_date") or f"{int(row.get('selection_end_year', last_year))}-12-31").normalize() for row in rows)
    selection_period_label = _period_label(selection_start_ts, selection_end_ts)
    oos_period_label = _period_label(first_oos_start, last_oos_end)

    payloads = {
        "best": _build_active_param_replay_payload_from_rows(
            rows,
            best_finalist=True,
            raw_universe_required_min_rows=config.raw_universe_required_min_rows,
        )
    }
    skipped_chain_policies: dict[str, str] = {}
    for policy_name in CHAIN_POLICY_NAMES:
        if _policy_has_complete_active_schedule(rows, policy_name):
            payloads[policy_name] = _build_active_param_replay_payload_from_rows(
                rows,
                policy_name=policy_name,
                raw_universe_required_min_rows=config.raw_universe_required_min_rows,
            )
        else:
            skipped_chain_policies[policy_name] = "incomplete_oos_schedule"
    total_elapsed_text = ""
    if overall_start is not None:
        total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
    print(
        f"{C_CYAN}⏳ OOS_CHAIN active replay | schedule policies={len(payloads)} | "
        f"period={oos_period_label}{total_elapsed_text}{C_RESET}",
        flush=True,
    )
    schedule_groups = {name: _build_active_replay_schedule_records(payload) for name, payload in payloads.items()}
    contexts_by_signature = _load_active_replay_contexts_by_signature(
        data_dir=selected_data_dir,
        schedule_groups=schedule_groups,
        output_dir=output_dir,
        first_year=first_year,
        last_year=last_year,
        overall_start=overall_start,
    )

    replay_metrics: dict[str, dict] = {}
    replay_started = time.perf_counter()
    replay_total = len(schedule_groups)
    replay_previous_width = 0
    replay_inline = stdout_supports_inline_progress()
    for replay_idx, (name, records) in enumerate(schedule_groups.items(), start=1):
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        replay_message = (
            f"{C_CYAN}⏳ OOS_CHAIN active replay [{replay_idx}/{replay_total}] | "
            f"policy={name} | schedule={len(records)} | "
            f"chain={_fmt_duration(time.perf_counter() - replay_started)}{total_elapsed_text}{C_RESET}"
        )
        if replay_inline:
            replay_previous_width = write_inline_progress(replay_message, previous_width=replay_previous_width)
        else:
            print(replay_message, flush=True)
        replay_metrics[name] = _run_active_replay_metrics_from_schedule_records(
            schedule_records=records,
            contexts_by_signature=contexts_by_signature,
            start_year=first_year,
            end_year=last_year,
            start_date=first_oos_start.strftime("%Y-%m-%d"),
            end_date=last_oos_end.strftime("%Y-%m-%d"),
            max_positions=max_positions,
            enable_rotation=enable_rotation,
        )
    if replay_total:
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        replay_done_message = (
            f"{C_CYAN}⏳ OOS_CHAIN active replay 完成 | "
            f"policies={replay_total}/{replay_total} | chain={_fmt_duration(time.perf_counter() - replay_started)}"
            f"{total_elapsed_text}{C_RESET}"
        )
        if replay_inline:
            write_inline_progress(replay_done_message, previous_width=replay_previous_width)
            print()
        else:
            print(replay_done_message, flush=True)
    for policy_name, reason in skipped_chain_policies.items():
        metrics = _empty_unavailable_chain_metrics()
        metrics["unavailable_reason"] = str(reason)
        replay_metrics[policy_name] = metrics

    best_metrics = dict(replay_metrics.get("best") or {})
    benchmark_source = dict(best_metrics)
    for policy_name in CHAIN_POLICY_NAMES:
        if not benchmark_source and replay_metrics.get(policy_name):
            benchmark_source = dict(replay_metrics[policy_name])
            break

    best_score = float(best_metrics.get("score", 0.0))
    best_return = float(best_metrics.get("return_pct", 0.0))
    benchmark_score = float(benchmark_source.get("benchmark_oos_score", 0.0))
    benchmark_return = float(benchmark_source.get("benchmark_return_pct", 0.0))
    summary = {
        "method": "continuous_active_param_replay",
        "score_aggregation_method": "continuous_active_param_replay_recomputed_score",
        "return_aggregation_method": "continuous_active_param_replay_total_return",
        "note": "OOS_CHAIN 由 portfolio_sim active-param replay 連續重跑後重算 score；持股、現金與 benchmark 跨 fold 延續，不使用獨立 OOS closeout stitch 或區間 score mean。",
        "selection_period": selection_period_label,
        "oos_period": oos_period_label,
        "max_positions": int(max_positions),
        "enable_rotation": bool(enable_rotation),
        "best_finalist_oos_score": float(best_score),
        "benchmark_oos_score": float(benchmark_score),
        "best_finalist_return_pct": float(best_return),
        "benchmark_return_pct": float(benchmark_return),
        "benchmark_alpha_pct": float(best_return - benchmark_return),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "benchmark_mdd_pct": float(benchmark_source.get("benchmark_mdd_pct", 0.0)),
        "best_finalist_annual_return_pct": float(best_metrics.get("annual_return_pct", 0.0)),
        "benchmark_annual_return_pct": float(benchmark_source.get("benchmark_annual_return_pct", 0.0)),
        "best_finalist_curve_points": int(best_metrics.get("curve_points", 0)),
        "benchmark_curve_points": int(benchmark_source.get("curve_points", 0)),
        "yearly_best_finalist_oos_score": [float(row.get("best_finalist_oos_score", 0.0)) for row in rows],
        "yearly_benchmark_oos_score": [float(row.get("benchmark_oos_score", 0.0)) for row in rows],
        "skipped_chain_policies": dict(skipped_chain_policies),
    }
    for policy_name in CHAIN_POLICY_NAMES:
        metrics = dict(replay_metrics.get(policy_name) or {})
        chain_score = float(metrics.get("score", 0.0))
        chain_plain_romd = float(metrics.get("plain_romd_score", calc_plain_romd(metrics.get("return_pct", 0.0), metrics.get("mdd_pct", 0.0))))
        chain_return = float(metrics.get("return_pct", 0.0))
        summary[policy_name] = {
            "available": bool(metrics.get("available", int(metrics.get("curve_points", 0) or 0) > 0)),
            "rank_1_oos": float(chain_score),
            "rank_1_plain_romd": float(chain_plain_romd),
            "best_gap": float(chain_score - best_score),
            "benchmark_0050_gap": float(chain_score - benchmark_score),
            "benchmark_0050_plain_romd_gap": float(chain_plain_romd - benchmark_score),
            "rank_1_return_pct": float(chain_return),
            "best_gap_pct": float(chain_return - best_return),
            "benchmark_0050_gap_pct": float(chain_return - benchmark_return),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_annual_return_pct": float(metrics.get("annual_return_pct", 0.0)),
            "rank_1_curve_points": int(metrics.get("curve_points", 0)),
            "rank_1_trades": int(metrics.get("trade_count", 0)),
            "unavailable_reason": str(metrics.get("unavailable_reason") or skipped_chain_policies.get(policy_name, "")),
            "yearly_oos_score": [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows],
            "yearly_return_pct": [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows],
        }
    return summary

def _build_chained_oos_summary(rows: list[dict], *, chained_override: dict | None = None) -> dict:
    if chained_override is not None:
        return dict(chained_override)
    if not rows:
        return {}
    first_oos_start = min(pd.Timestamp(row.get("oos_start_date") or f"{str(row.get('oos_year'))[:4]}-01-01").normalize() for row in rows)
    last_oos_end = max(pd.Timestamp(row.get("oos_end_date") or f"{str(row.get('oos_year'))[:4]}-12-31").normalize() for row in rows)
    first_year = int(first_oos_start.year)
    last_year = int(last_oos_end.year)
    selection_start_ts = min(pd.Timestamp(row.get("selection_start_date") or f"{int(row.get('selection_start_year', first_year))}-01-01").normalize() for row in rows)
    selection_end_ts = max(pd.Timestamp(row.get("selection_end_date") or f"{int(row.get('selection_end_year', last_year))}-12-31").normalize() for row in rows)
    selection_period_label = _period_label(selection_start_ts, selection_end_ts)
    oos_period_label = _period_label(first_oos_start, last_oos_end)

    best_stitched = _stitch_strategy_equity_curves(rows, best_finalist=True)
    benchmark_stitched = _stitch_benchmark_equity_curve(rows)
    best_metrics = _calc_stitched_curve_metrics(best_stitched)
    benchmark_metrics = _calc_stitched_curve_metrics(benchmark_stitched, benchmark_plain_romd=True)
    best_score = float(best_metrics.get("score", 0.0))
    benchmark_score = float(benchmark_metrics.get("score", 0.0))
    best_return = float(best_metrics.get("return_pct", 0.0))
    benchmark_return = float(benchmark_metrics.get("return_pct", 0.0))

    summary = {
        "method": "fallback_period_closeout_stitched_daily_equity",
        "score_aggregation_method": "fallback_period_closeout_stitched_daily_equity",
        "return_aggregation_method": "fallback_period_closeout_stitched_daily_equity_total_return",
        "note": "fallback：由各 OOS period daily equity 串接後重新計算 score；正式輸出應優先使用 continuous_active_param_replay。",
        "selection_period": selection_period_label,
        "oos_period": oos_period_label,
        "best_finalist_oos_score": float(best_score),
        "benchmark_oos_score": float(benchmark_score),
        "best_finalist_return_pct": float(best_return),
        "benchmark_return_pct": float(benchmark_return),
        "benchmark_alpha_pct": float(best_return - benchmark_return),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "benchmark_mdd_pct": float(benchmark_metrics.get("mdd_pct", 0.0)),
        "best_finalist_annual_return_pct": float(best_metrics.get("annual_return_pct", 0.0)),
        "benchmark_annual_return_pct": float(benchmark_metrics.get("annual_return_pct", 0.0)),
        "best_finalist_curve_points": int(best_metrics.get("curve_points", 0)),
        "benchmark_curve_points": int(benchmark_metrics.get("curve_points", 0)),
        "yearly_best_finalist_oos_score": [float(row.get("best_finalist_oos_score", 0.0)) for row in rows],
        "yearly_benchmark_oos_score": [float(row.get("benchmark_oos_score", 0.0)) for row in rows],
    }
    for policy_name in CHAIN_POLICY_NAMES:
        stitched = _stitch_strategy_equity_curves(rows, policy_name=policy_name)
        metrics = _calc_stitched_curve_metrics(stitched)
        chain_score = float(metrics.get("score", 0.0))
        chain_plain_romd = float(metrics.get("plain_romd_score", calc_plain_romd(metrics.get("return_pct", 0.0), metrics.get("mdd_pct", 0.0))))
        chain_return = float(metrics.get("return_pct", 0.0))
        summary[policy_name] = {
            "available": int(metrics.get("curve_points", 0) or 0) > 0,
            "rank_1_oos": float(chain_score),
            "rank_1_plain_romd": float(chain_plain_romd),
            "best_gap": float(chain_score - best_score),
            "benchmark_0050_gap": float(chain_score - benchmark_score),
            "benchmark_0050_plain_romd_gap": float(chain_plain_romd - benchmark_score),
            "rank_1_return_pct": float(chain_return),
            "best_gap_pct": float(chain_return - best_return),
            "benchmark_0050_gap_pct": float(chain_return - benchmark_return),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_annual_return_pct": float(metrics.get("annual_return_pct", 0.0)),
            "rank_1_curve_points": int(metrics.get("curve_points", 0)),
            "yearly_oos_score": [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows],
            "yearly_return_pct": [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows],
        }
    return summary


def _resolve_chain_elapsed_sec(rows: list[dict], chained: dict) -> float:
    """Return the OOS_CHAIN elapsed value for display/reporting.

    In serial mode the chain row historically used the sum of fold elapsed time. In
    parallel-fold timing mode, however, the sum of fold elapsed time is total work
    time, not user-visible wall-clock time. When the caller provides an
    ``elapsed_sec`` override in the chained summary, prefer it; otherwise fall back
    to the serial-compatible fold sum.
    """
    try:
        override = chained.get("elapsed_sec")
    except AttributeError:
        override = None
    if override is not None:
        try:
            return max(0.0, float(override))
        except (TypeError, ValueError) as exc:
            _mark_resource_probe_fallback(exc)
    total = 0.0
    for item in list(rows or []):
        try:
            total += float(item.get("elapsed_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return max(0.0, total)


def _with_chain_elapsed_override(chained_override: dict | None, *, elapsed_sec: float | None) -> dict:
    payload = dict(chained_override or {})
    if elapsed_sec is not None:
        try:
            payload["elapsed_sec"] = max(0.0, float(elapsed_sec))
        except (TypeError, ValueError) as exc:
            _mark_resource_probe_fallback(exc)
    return payload


def _build_chained_oos_row(rows: list[dict], *, chained_override: dict | None = None) -> dict | None:
    if not rows:
        return None
    chained = _build_chained_oos_summary(rows, chained_override=chained_override)
    row = {
        "fold": "OOS_CHAIN",
        "selection_period": chained.get("selection_period", ""),
        "oos_year": chained.get("oos_period", ""),
        "best_finalist_oos_score": float(chained.get("best_finalist_oos_score", 0.0)),
        "benchmark_oos_score": float(chained.get("benchmark_oos_score", 0.0)),
        "best_finalist_return_pct": float(chained.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": float(chained.get("benchmark_return_pct", 0.0)),
        "elapsed_sec": _resolve_chain_elapsed_sec(rows, chained),
        "aggregation_method": chained.get("score_aggregation_method") or chained.get("method"),
    }
    for policy_name in REPORT_POLICY_NAMES:
        item = dict(chained.get(policy_name) or {})
        available = _policy_is_available(item)
        rank_score = float(item.get("rank_1_oos", 0.0)) if available else 0.0
        plain_romd = _policy_plain_romd_score(item) if available else 0.0
        row[policy_name] = {
            "available": bool(available),
            "rank_1_trial": None,
            "rank_1_oos": rank_score,
            "rank_1_plain_romd": plain_romd,
            "rank_1_return_pct": float(item.get("rank_1_return_pct", 0.0)) if available else 0.0,
            "rank_1_mdd_pct": float(item.get("rank_1_mdd_pct", 0.0)) if available else 0.0,
            "best_gap": (rank_score - float(chained.get("best_finalist_oos_score", 0.0))) if available else 0.0,
            "benchmark_0050_gap": (rank_score - float(chained.get("benchmark_oos_score", 0.0))) if available else 0.0,
            "benchmark_0050_plain_romd_gap": (plain_romd - float(chained.get("benchmark_oos_score", 0.0))) if available else 0.0,
            "unavailable_reason": item.get("unavailable_reason") or item.get("skip_reason") or "",
        }
    return row



def _rows_period_bounds(rows: list[dict]) -> dict:
    source_rows = [
        normalize_optimizer_seed_ensemble_fold_row(row)
        for row in list(rows or [])
        if str((row or {}).get("fold", "")).upper() not in {"OOS_CHAIN", "OOS_AVG"}
    ]
    if not source_rows:
        return {}

    def _date_or_latest(value, fallback):
        text = str(value or "").strip()
        if text.lower() == "latest" or not text:
            return pd.Timestamp(fallback).normalize()
        return pd.Timestamp(text).normalize()

    selection_starts = [pd.Timestamp(row.get("selection_start_date")).normalize() for row in source_rows if row.get("selection_start_date")]
    selection_ends = [pd.Timestamp(row.get("selection_end_date")).normalize() for row in source_rows if row.get("selection_end_date")]
    oos_starts = [pd.Timestamp(row.get("oos_start_date")).normalize() for row in source_rows if row.get("oos_start_date")]
    oos_ends = [_date_or_latest(row.get("oos_end_date"), pd.Timestamp.today().normalize()) for row in source_rows if row.get("oos_start_date") or row.get("oos_end_date")]
    oos_keys = [int(row.get("oos_year", 0) or 0) for row in source_rows if int(row.get("oos_year", 0) or 0) > 0]

    selection_period = ""
    if selection_starts and selection_ends:
        selection_period = _period_label(min(selection_starts), max(selection_ends))
    oos_period = ""
    if oos_starts:
        oos_period = _period_label(min(oos_starts), max(oos_ends) if oos_ends else min(oos_starts))
    return {
        "selection_period": selection_period,
        "oos_period": oos_period,
        "first_oos_key": min(oos_keys) if oos_keys else 0,
        "last_oos_key": max(oos_keys) if oos_keys else 0,
    }


def _avg_float_from_rows(rows: list[dict], getter, default: float = 0.0) -> float:
    values: list[float] = []
    for row in list(rows or []):
        try:
            value = float(getter(row))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
        values.append(value)
    return (sum(values) / float(len(values))) if values else float(default)


def _build_oos_avg_row(rows: list[dict]) -> dict | None:
    source_rows = [dict(row) for row in list(rows or []) if str(row.get("fold", "")).upper() not in {"OOS_CHAIN", "OOS_AVG"}]
    if not source_rows:
        return None
    bounds = _rows_period_bounds(source_rows)
    row = {
        "fold": "OOS_AVG",
        "selection_period": str(bounds.get("selection_period", "")),
        "oos_year": str(bounds.get("oos_period", "")),
        "oos_period": str(bounds.get("oos_period", "")),
        "best_finalist_oos_score": _avg_float_from_rows(source_rows, lambda item: item.get("best_finalist_oos_score", 0.0)),
        "benchmark_oos_score": _avg_float_from_rows(source_rows, lambda item: item.get("benchmark_oos_score", 0.0)),
        "best_finalist_return_pct": _avg_float_from_rows(source_rows, lambda item: item.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": _avg_float_from_rows(source_rows, lambda item: item.get("benchmark_return_pct", 0.0)),
        "elapsed_sec": None,
    }
    for policy_name in REPORT_POLICY_NAMES:
        available_rows = [
            item
            for item in source_rows
            if _policy_is_available(dict((item.get(policy_name) or {})))
        ]
        if not available_rows:
            row[policy_name] = {
                "available": False,
                "rank_1_trial": None,
                "rank_1_oos": 0.0,
                "rank_1_plain_romd": 0.0,
                "rank_1_return_pct": 0.0,
                "rank_1_mdd_pct": 0.0,
                "best_gap": 0.0,
                "benchmark_0050_gap": 0.0,
                "benchmark_0050_plain_romd_gap": 0.0,
                "unavailable_reason": "no_available_period",
            }
            continue
        rank_score = _avg_float_from_rows(available_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_oos", 0.0))
        plain_romd = _avg_float_from_rows(available_rows, lambda item, name=policy_name: _policy_plain_romd_score((item.get(name) or {})))
        row[policy_name] = {
            "available": True,
            "rank_1_trial": None,
            "rank_1_oos": rank_score,
            "rank_1_plain_romd": plain_romd,
            "rank_1_return_pct": _avg_float_from_rows(available_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_return_pct", 0.0)),
            "rank_1_mdd_pct": _avg_float_from_rows(available_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_mdd_pct", 0.0)),
            "best_gap": rank_score - float(row["best_finalist_oos_score"]),
            "benchmark_0050_gap": rank_score - float(row["benchmark_oos_score"]),
            "benchmark_0050_plain_romd_gap": plain_romd - float(row["benchmark_oos_score"]),
        }
    return row






def _policy_cell_text(policy_row: dict, *, best_score: float, benchmark_score: float, color: bool = True) -> tuple[str, str]:
    _ = best_score
    if not _policy_is_available(policy_row):
        return "N/A", "N/A"
    rank_1_romd = _policy_plain_romd_score(policy_row)
    if color:
        return (
            _format_plain_score(rank_1_romd),
            _format_compare(benchmark_score, rank_1_romd),
        )
    return (
        f"{rank_1_romd:.{OOS_SCORE_DECIMALS}f}",
        _format_compare_plain(benchmark_score, rank_1_romd),
    )


def _table_separator(width: int = 218) -> str:
    return "-" * int(width)


def _render_results_table(rows: list[dict], *, color: bool = True, include_chain: bool = True, include_oos_avg: bool = False, chained_override: dict | None = None, table_title: str = "ROLLING MONTHLY OOS RESULTS", policy_names: tuple[str, ...] | None = None) -> str:
    display_rows = normalize_optimizer_seed_ensemble_fold_rows(list(rows or []))
    if include_oos_avg and display_rows:
        avg_row = _build_oos_avg_row(display_rows)
        if avg_row is not None:
            display_rows = display_rows + [avg_row]
    if include_chain:
        chain_row = _build_chained_oos_row(display_rows, chained_override=chained_override)
        if chain_row is not None:
            display_rows = display_rows + [chain_row]
    if not display_rows:
        return ""
    active_policy_names = tuple(policy_names or REPORT_POLICY_NAMES)
    if not active_policy_names:
        return ""
    widths = {
        "fold": 9,
        "selection": 19,
        "oos_year": 19,
        "romd": 8,
        "bench": 15,
        "elapsed": 8,
    }
    policy_group_width = widths["romd"] + widths["bench"] + 3
    lines: list[str] = []
    lines.append(str(table_title or "ROLLING MONTHLY OOS RESULTS"))
    policy_header = " | ".join(_pad_ansi(REPORT_POLICY_LABELS[name], policy_group_width, align="^") for name in active_policy_names)
    policy_subheader = " | ".join(
        f"{_pad_ansi('RoMD', widths['romd'], align='^')} | {_pad_ansi('0050', widths['bench'], align='^')}"
        for _ in active_policy_names
    )
    header1 = (
        f"{_pad_ansi('fold', widths['fold'], align='^')} | {_pad_ansi('train', widths['selection'], align='^')} | {_pad_ansi('oos_period', widths['oos_year'], align='^')} | "
        f"{policy_header} | {_pad_ansi('elapsed', widths['elapsed'], align='^')}"
    )
    header2 = (
        f"{_pad_ansi('', widths['fold'])} | {_pad_ansi('', widths['selection'])} | {_pad_ansi('', widths['oos_year'])} | "
        f"{policy_subheader} | {_pad_ansi('', widths['elapsed'])}"
    )
    separator = _table_separator(max(_visible_len(header1), _visible_len(header2), 120))
    lines.append(separator)
    lines.append(header1)
    lines.append(separator)
    lines.append(header2)
    lines.append(separator)
    total = len(rows or [])
    for idx, row in enumerate(display_rows, start=1):
        fold_kind = str(row.get("fold", "")).upper()
        is_chain = fold_kind == "OOS_CHAIN"
        is_avg = fold_kind == "OOS_AVG"
        fold_text = "OOS_CHAIN" if is_chain else ("OOS_AVG" if is_avg else str(row.get("fold") or f"{idx}/{total}"))
        best_score = float(row.get("best_finalist_oos_score", 0.0))
        benchmark_score = float(row.get("benchmark_oos_score", 0.0))
        policy_cells: list[str] = []
        for policy_name in active_policy_names:
            romd_text, bench_text = _policy_cell_text(row.get(policy_name) or {}, best_score=best_score, benchmark_score=benchmark_score, color=color)
            policy_cells.append(
                f"{_pad_ansi(romd_text, widths['romd'], align='>')} | "
                f"{_pad_ansi(bench_text, widths['bench'], align='>')}"
            )
        line = (
            f"{_pad_ansi(fold_text, widths['fold'])} | {_pad_ansi(_display_compact_month_period(row.get('selection_period', '')), widths['selection'])} | {_pad_ansi(_display_compact_month_period(row.get('oos_period') or row.get('oos_year', '')), widths['oos_year'])} | "
            f"{' | '.join(policy_cells)} | "
            f"{_pad_ansi('' if row.get('elapsed_sec') is None else _fmt_duration(row.get('elapsed_sec', 0.0)), widths['elapsed'], align='>')}"
        )
        lines.append(line)
    lines.append(separator)
    return "\n".join(lines)


def _render_optimizer_results_tables(rows: list[dict], *, color: bool = True, include_chain: bool = True, include_oos_avg: bool = False, chained_override: dict | None = None, main_table_title: str = "ROLLING MONTHLY OOS RESULTS", retention_table_title: str = "") -> str:
    _ = (main_table_title, retention_table_title)
    finalist_best_table = _render_results_table(
        rows,
        color=color,
        include_chain=include_chain,
        include_oos_avg=include_oos_avg,
        chained_override=chained_override,
        table_title=FINALIST_BEST_TABLE_TITLE,
        policy_names=FINALIST_BEST_RESULT_POLICY_NAMES,
    )
    finalists_agree_table = _render_results_table(
        rows,
        color=color,
        include_chain=include_chain,
        include_oos_avg=include_oos_avg,
        chained_override=chained_override,
        table_title=FINALISTS_AGREE_TABLE_TITLE,
        policy_names=FINALISTS_AGREE_RESULT_POLICY_NAMES,
    )
    seed_ensemble_table = ""
    if _is_rolling_random_seed_ensemble_enabled():
        seed_ensemble_table = _render_results_table(
            rows,
            color=color,
            include_chain=include_chain,
            include_oos_avg=include_oos_avg,
            chained_override=chained_override,
            table_title=SEED_ENSEMBLE_RESULTS_TABLE_TITLE,
            policy_names=SEED_ENSEMBLE_RESULT_POLICY_NAMES,
        )
    return "\n\n".join(part for part in (finalist_best_table, finalists_agree_table, seed_ensemble_table) if part)


def render_optimizer_results_tables(rows: list[dict], *, color: bool = True, include_chain: bool = True, include_oos_avg: bool = False, chained_override: dict | None = None, main_table_title: str = "", retention_table_title: str = "") -> str:
    """Render optimizer OOS result tables through the rolling-OOS table source.

    Non-rolling summary display intentionally uses this public wrapper with a
    single fold row so rolling and non-rolling console tables cannot drift.
    """
    return _render_optimizer_results_tables(
        rows,
        color=color,
        include_chain=include_chain,
        include_oos_avg=include_oos_avg,
        chained_override=chained_override,
        main_table_title=main_table_title,
        retention_table_title=retention_table_title,
    )




def _print_completed_results(rows: list[dict]):
    if not rows:
        return
    print("\n" + _render_optimizer_results_tables(rows, color=True, include_chain=True))








def _build_seed_ensemble_policy_payload() -> dict:
    return build_seed_ensemble_policy_snapshot(
        enabled=OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
        seed_count=OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
        min_agree=OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
    )


_build_random_seed_ensemble_policy_payload = _build_seed_ensemble_policy_payload




def _build_effective_seed_ensemble_policy_payload(params_ensemble_by_effective_date: dict, *, policy_name: str | None = None) -> dict:
    requested_policy = _build_random_seed_ensemble_policy_payload()
    member_counts = [
        len(normalize_seed_ensemble_members(members))
        for members in (params_ensemble_by_effective_date or {}).values()
    ]
    member_counts = [int(count) for count in member_counts if int(count) > 0]
    actual_min_members = max(1, min(member_counts) if member_counts else 1)
    actual_max_members = max(member_counts) if member_counts else actual_min_members
    requested_count = int(requested_policy.get("seed_count", 1) or 1)
    if _is_finalists_agree_policy(str(policy_name or "")):
        finalists_agree_config = _finalists_agree_policy_config(str(policy_name or ""))
        policy = build_seed_ensemble_policy_snapshot(
            enabled=actual_min_members > 1,
            seed_count=actual_min_members,
            min_agree=_resolve_finalists_agree_min_agree(str(policy_name or ""), actual_min_members),
        )
        policy["selection_rule"] = str(finalists_agree_config["selection_rule"])
        policy[str(finalists_agree_config["min_agree_requested_key"])] = finalists_agree_config["min_agree_requested"]
        policy["member_selection"] = str(finalists_agree_config["member_selection"])
        policy["intra_seed_agree"] = "selected_seed_finalists"
        policy["requested_random_seed_ensemble"] = dict(requested_policy)
    elif actual_min_members == actual_max_members == requested_count:
        policy = dict(requested_policy)
    else:
        policy = build_seed_ensemble_policy_snapshot(
            enabled=actual_min_members > 1,
            seed_count=actual_min_members,
            min_agree=requested_policy.get("min_agree_requested", requested_policy.get("min_agree", "auto")),
        )
        policy["requested_random_seed_ensemble"] = dict(requested_policy)
        policy["generation_note"] = (
            "outer rolling 目前此 policy schedule 實際產出的 members 數量與設定 N 不一致；"
            "正式 replay 口徑以 JSON 實際 members 數量為準，避免 min_agree 大於可用 members。"
        )
    policy["policy_name"] = str(policy_name or "")
    policy["member_count_min"] = int(actual_min_members)
    policy["member_count_max"] = int(actual_max_members)
    policy["member_count_mismatch"] = bool(actual_min_members != requested_count or actual_max_members != requested_count)
    return policy


def _build_params_ensemble_members_for_schedule(schedule: dict) -> list[dict]:
    explicit_members = normalize_seed_ensemble_members(schedule.get("params_ensemble"))
    if explicit_members:
        return explicit_members
    params_payload = dict(schedule.get("params") or {})
    if not params_payload:
        return []
    member = {
        "member_index": 1,
        "seed": schedule.get("optimizer_seed"),
        "selected_trial": schedule.get("selected_trial"),
        "params": params_payload,
    }
    for key in (
        "base_score",
        "base_rank",
        "selection_rule",
        "policy_type",
        "base_agree_seed_finalist_count",
        "base_agree_seed_base_score_sum",
        "base_agree_seed_selected_trials",
        "base_agree_min_agree_requested",
        "local_agree_seed_finalist_count",
        "local_agree_seed_local_min_sum",
        "local_agree_seed_selected_trials",
        "local_agree_min_agree_requested",
        "retention_agree_seed_finalist_count",
        "retention_agree_seed_retention_sum",
        "retention_agree_seed_selected_trials",
        "retention_agree_min_agree_requested",
        "min_agree",
        "min_agree_requested",
        "member_count",
    ):
        if key in schedule:
            member[key] = schedule.get(key)
    return [member]


def _build_policy_paramset_payload(*, policy_name: str, rows: list[dict], config: OuterRollingConfig, summary: dict) -> dict:
    rows = normalize_optimizer_seed_ensemble_fold_rows(list(rows or []))
    params_by_oos_year = {}
    params_by_effective_date = {}
    params_ensemble_by_effective_date = {}
    requested_seed_ensemble_policy = _build_random_seed_ensemble_policy_payload()
    fold_entries = []
    for row in rows:
        schedule = dict((row.get("policy_schedules") or {}).get(policy_name) or {})
        if not schedule:
            continue
        oos_year = str(int(schedule.get("oos_year") or row.get("oos_year")))
        params_payload = dict(schedule.get("params") or {})
        effective_start_key = str(schedule.get("effective_start") or row.get("oos_start_date") or f"{str(oos_year)[:4]}-01-01")
        params_by_oos_year[oos_year] = params_payload
        params_by_effective_date[effective_start_key] = params_payload
        params_ensemble_members = _build_params_ensemble_members_for_schedule(schedule)
        if params_ensemble_members:
            params_ensemble_by_effective_date[effective_start_key] = renumber_seed_ensemble_members(params_ensemble_members)
        policy_metrics = dict(row.get(policy_name) or {})
        fold_entries.append({
            "fold": row.get("fold"),
            "selection_period": row.get("selection_period"),
            "oos_year": int(row.get("oos_year")),
            "oos_period": row.get("oos_period"),
            "selection_start_date": row.get("selection_start_date"),
            "selection_end_date": row.get("selection_end_date"),
            "oos_start_date": row.get("oos_start_date"),
            "oos_end_date": row.get("oos_end_date"),
            "effective_start": schedule.get("effective_start"),
            "effective_end": schedule.get("effective_end"),
            "selected_trial": schedule.get("selected_trial"),
            "optimizer_seed": schedule.get("optimizer_seed"),
            "member_count": schedule.get("member_count"),
            "min_agree": schedule.get("min_agree"),
            "min_agree_requested": schedule.get("min_agree_requested"),
            "base_score": schedule.get("base_score"),
            "base_rank": schedule.get("base_rank"),
            "selection_rule": schedule.get("selection_rule"),
            "policy_type": schedule.get("policy_type"),
            "base_agree_seed_finalist_count": schedule.get("base_agree_seed_finalist_count"),
            "base_agree_seed_base_score_sum": schedule.get("base_agree_seed_base_score_sum"),
            "base_agree_seed_selected_trials": schedule.get("base_agree_seed_selected_trials"),
            "base_agree_min_agree_requested": schedule.get("base_agree_min_agree_requested"),
            "local_agree_seed_finalist_count": schedule.get("local_agree_seed_finalist_count"),
            "local_agree_seed_local_min_sum": schedule.get("local_agree_seed_local_min_sum"),
            "local_agree_seed_selected_trials": schedule.get("local_agree_seed_selected_trials"),
            "local_agree_min_agree_requested": schedule.get("local_agree_min_agree_requested"),
            "retention_agree_seed_finalist_count": schedule.get("retention_agree_seed_finalist_count"),
            "retention_agree_seed_retention_sum": schedule.get("retention_agree_seed_retention_sum"),
            "retention_agree_seed_selected_trials": schedule.get("retention_agree_seed_selected_trials"),
            "retention_agree_min_agree_requested": schedule.get("retention_agree_min_agree_requested"),
            "local_min": schedule.get("local_min"),
            "local_min_review_enabled": schedule.get("local_min_review_enabled", bool(is_optimizer_local_min_review_enabled())),
            "local_min_review_mode": schedule.get("local_min_review_mode"),
            "local_min_exact": schedule.get("local_min_exact"),
            "local_rank": schedule.get("local_rank"),
            "retention": schedule.get("retention"),
            "retention_rank": schedule.get("retention_rank"),
            "oos_score": policy_metrics.get("rank_1_oos"),
            "plain_romd_score": policy_metrics.get("rank_1_plain_romd"),
            "return_pct": policy_metrics.get("rank_1_return_pct"),
            "mdd_pct": policy_metrics.get("rank_1_mdd_pct"),
            "trades": policy_metrics.get("rank_1_trades"),
            "benchmark_return_pct": row.get("benchmark_return_pct"),
            "benchmark_oos_score": row.get("benchmark_oos_score"),
            "best_finalist_return_pct": row.get("best_finalist_return_pct"),
            "best_finalist_oos_score": row.get("best_finalist_oos_score"),
        })
    seed_ensemble_policy = _build_effective_seed_ensemble_policy_payload(params_ensemble_by_effective_date, policy_name=policy_name)
    chain_all = dict(summary.get("chained_oos") or {})
    chain_policy = dict(chain_all.get(policy_name) or {})
    chain_policy_available = _policy_is_available(chain_policy)
    chained_oos = {
        "available": bool(chain_policy_available),
        "unavailable_reason": str(chain_policy.get("unavailable_reason") or "") if not chain_policy_available else "",
        "method": chain_all.get("method"),
        "score_aggregation_method": chain_all.get("score_aggregation_method"),
        "return_aggregation_method": chain_all.get("return_aggregation_method"),
        "note": chain_all.get("note"),
        "selection_period": chain_all.get("selection_period"),
        "oos_period": chain_all.get("oos_period"),
        "rank_1_oos_score": float(chain_policy.get("rank_1_oos", 0.0)) if chain_policy_available else 0.0,
        "rank_1_plain_romd_score": _policy_plain_romd_score(chain_policy) if chain_policy_available else 0.0,
        "best_finalist_oos_score": float(chain_all.get("best_finalist_oos_score", 0.0)),
        "benchmark_oos_score": float(chain_all.get("benchmark_oos_score", 0.0)),
        "alpha_oos_score": float(chain_policy.get("benchmark_0050_gap", 0.0)) if chain_policy_available else 0.0,
        "alpha_plain_romd_score": float(chain_policy.get("benchmark_0050_plain_romd_gap", 0.0)) if chain_policy_available else 0.0,
        "best_gap_score": float(chain_policy.get("best_gap", 0.0)) if chain_policy_available else 0.0,
        "rank_1_return_pct": float(chain_policy.get("rank_1_return_pct", 0.0)) if chain_policy_available else 0.0,
        "best_finalist_return_pct": float(chain_all.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": float(chain_all.get("benchmark_return_pct", 0.0)),
        "alpha_return_pct": float(chain_policy.get("benchmark_0050_gap_pct", 0.0)) if chain_policy_available else 0.0,
        "best_gap_pct": float(chain_policy.get("best_gap_pct", 0.0)) if chain_policy_available else 0.0,
    }
    policy_summary = dict(summary.get(policy_name) or {})
    return {
        **_raw_universe_contract_fields(config.raw_universe_required_min_rows),
        "schema_type": ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
        "schema_version": 1,
        "mode": ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
        "legacy_schema_type": ROLLING_OOS_PARAM_SET_SCHEMA_TYPE,
        "usage": ROLLING_OOS_USAGE,
        "type": "outer_rolling_oos_param_set",
        "created_at": get_taipei_now().isoformat(),
        "selector": str(policy_name),
        "meta": {
            "window_mode": str(config.window_mode),
            "train_window_years": int(config.train_window_years),
            "first_oos_date": str(config.first_oos_date),
            "last_oos_date": str(config.last_oos_date),
            "train_window_months": int(config.train_window_months),
            "oos_horizon_months": int(config.oos_horizon_months),
            "training_start_year": int(config.training_start_year),
            "first_oos_year": int(config.first_oos_year),
            "last_oos_year": int(config.last_oos_year),
            "oos_horizon": f"next_{int(config.oos_horizon_months)}m",
            "oos_feedback_used": False,
            "promotion_enabled": False,
            "trials_per_fold": int(config.trials_per_fold),
            "live_trading_param": False,
            "active_param_policy": "daily_active_param_ensemble",
            "active_param_policy_note": "驗證 replay 時，每個交易日所有決策都使用該日期已生效的 active-param ensemble；實盤同理使用當下正式 promote 的最新 ensemble param.json。",
            "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
            "random_seed_ensemble": seed_ensemble_policy,
            "requested_random_seed_ensemble": requested_seed_ensemble_policy,
        },
        "summary": {
            "folds": int(summary.get("folds", 0)),
            "selection_period": summary.get("selection_period"),
            "oos_period": summary.get("oos_period"),
            "aggregation_method": summary.get("aggregation_method"),
            "return_aggregation_method": summary.get("return_aggregation_method"),
            "selector": str(policy_name),
            "available": bool(policy_summary.get("available", True)),
            "chained_oos_score": float(policy_summary.get("chained_oos_score", 0.0)),
            "chained_plain_romd_score": float(policy_summary.get("chained_plain_romd_score", 0.0)),
            "chained_return_pct": float(policy_summary.get("chained_return_pct", 0.0)),
            "chained_gap_vs_0050_pct": float(policy_summary.get("chained_gap_vs_0050_pct", 0.0)),
            "chained_plain_romd_gap_vs_0050": float(policy_summary.get("chained_plain_romd_gap_vs_0050", 0.0)),
            "chained_unavailable_reason": str(policy_summary.get("chained_unavailable_reason") or ""),
            "period_avg_oos_score": float(policy_summary.get("period_avg_oos_score", policy_summary.get("avg_oos_score", 0.0))),
            "period_avg_plain_romd_score": float(policy_summary.get("period_avg_plain_romd_score", policy_summary.get("avg_plain_romd_score", 0.0))),
            "yearly_avg_oos_score": float(policy_summary.get("period_avg_oos_score", policy_summary.get("yearly_avg_oos_score", policy_summary.get("avg_oos_score", 0.0)))),
            "yearly_avg_plain_romd_score": float(policy_summary.get("period_avg_plain_romd_score", policy_summary.get("yearly_avg_plain_romd_score", policy_summary.get("avg_plain_romd_score", 0.0)))),
            "avg_oos_score": float(policy_summary.get("avg_oos_score", 0.0)),
            "avg_plain_romd_score": float(policy_summary.get("avg_plain_romd_score", 0.0)),
            "median_oos_score": float(policy_summary.get("median_oos_score", 0.0)),
            "median_plain_romd_score": float(policy_summary.get("median_plain_romd_score", 0.0)),
            "worst_oos_score": float(policy_summary.get("worst_oos_score", 0.0)),
            "worst_plain_romd_score": float(policy_summary.get("worst_plain_romd_score", 0.0)),
            "positive_years": int(policy_summary.get("positive_years", 0)),
            "total_years": int(policy_summary.get("total_years", 0)),
        },
        "active_param_policy": "daily_active_param_ensemble",
        "active_param_policy_note": "每日決策使用該日 active-param ensemble；rolling 只是用歷史 effective date replay，不代表實盤使用固定單期參數組。",
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        "random_seed_ensemble": seed_ensemble_policy,
        "chained_oos": chained_oos,
        "params_by_effective_date": params_by_effective_date,
        "params_ensemble_by_effective_date": params_ensemble_by_effective_date,
        "params_by_oos_year": params_by_oos_year,
        "folds": fold_entries,
    }


def _remove_stale_policy_paramset_files(models_dir: str) -> None:
    for filename in STALE_POLICY_PARAMSET_FILENAMES:
        path = os.path.join(models_dir, str(filename))
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            try:
                display_path = os.path.relpath(path, os.path.dirname(models_dir)).replace(os.sep, "/")
            except ValueError:
                display_path = os.path.basename(path).replace(os.sep, "/")
            print(f"{C_YELLOW}注意：無法移除舊 policy 檔：{display_path}｜{type(exc).__name__}: {exc}{C_RESET}")


def _write_policy_paramset_files(
    *,
    models_dir: str,
    rows: list[dict],
    config: OuterRollingConfig,
    summary: dict,
    canonical_strategy_param_family: str | None = None,
) -> dict:
    os.makedirs(models_dir, exist_ok=True)
    _remove_stale_policy_paramset_files(models_dir)
    paths = {}
    for policy_name in REPORT_POLICY_NAMES:
        if canonical_strategy_param_family:
            from core.strategy_param_artifacts import POLICY_FILENAME_BY_NAME

            base = str(POLICY_FILENAME_BY_NAME.get(policy_name, f"{policy_name}.json"))
            filename = f"{canonical_strategy_param_family}_{base}"
        else:
            filename = str(PARAMSET_FILENAME_BY_POLICY.get(policy_name, f"roos_{policy_name}.json"))
        path = os.path.join(models_dir, filename)
        payload = normalize_strategy_param_payload_for_persistence(
            _build_policy_paramset_payload(
                policy_name=policy_name, rows=rows, config=config, summary=summary
            )
        )
        atomic_write_json(path, payload)
        paths[policy_name] = path
    return paths


def _write_reports(
    *,
    project_root: str,
    output_dir: str,
    session_ts: str,
    rows: list[dict],
    config: OuterRollingConfig,
    chained_override: dict | None = None,
    models_dir: str | None = None,
    canonical_strategy_param_family: str | None = None,
) -> dict:
    _ = (output_dir, session_ts)
    resolved_models_dir = str(models_dir or resolve_models_dir(project_root))
    default_models_dir = os.path.abspath(resolve_models_dir(project_root))
    requested_abs = os.path.abspath(resolved_models_dir)
    canonical_family = (
        None if canonical_strategy_param_family in (None, "")
        else str(canonical_strategy_param_family).strip().lower()
    )
    if canonical_family not in (None, "full", "min"):
        raise ValueError(f"不支援的canonical strategy parameter family: {canonical_family!r}")
    if canonical_family is not None:
        from core.strategy_param_artifacts import resolve_strategy_param_dir
        resolved_models_dir = str(resolve_strategy_param_dir(
            project_root, family=canonical_family, evaluation_mode="rolling"
        ))
    elif requested_abs == default_models_dir:
        # Historical direct Outer-Rolling entry is the Full canonical producer.
        from core.strategy_param_artifacts import resolve_strategy_param_dir
        canonical_family = "full"
        resolved_models_dir = str(resolve_strategy_param_dir(
            project_root, family="full", evaluation_mode="rolling"
        ))
    canonical_current_output = canonical_family is not None
    summary = _build_summary(rows, config=config, chained_override=chained_override)
    paramset_paths = _write_policy_paramset_files(
        models_dir=resolved_models_dir,
        rows=rows,
        config=config,
        summary=summary,
        canonical_strategy_param_family=(str(canonical_family) if canonical_current_output else None),
    )
    if canonical_current_output:
        from services.optimizer.strategy_param_repository import refresh_strategy_parameter_manifest

        refresh_strategy_parameter_manifest(
            project_root, family=str(canonical_family), evaluation_mode="rolling"
        )
    return {"paramsets": paramset_paths}

def _build_summary(rows: list[dict], *, config: OuterRollingConfig | None = None, chained_override: dict | None = None) -> dict:
    if not rows:
        return {"folds": 0}
    rows = normalize_optimizer_seed_ensemble_fold_rows(list(rows or []))
    bounds = _rows_period_bounds(rows)
    chained = _build_chained_oos_summary(rows, chained_override=chained_override)
    summary = {
        "folds": len(rows),
        "selection_period": str(bounds.get("selection_period", "")),
        "oos_period": str(bounds.get("oos_period", "")),
        "aggregation_method": chained.get("score_aggregation_method") or chained.get("method"),
        "return_aggregation_method": chained.get("return_aggregation_method"),
        "note": chained.get("note"),
        "chained_oos": chained,
    }
    for policy_name in CHAIN_POLICY_NAMES:
        available_rows = [row for row in rows if _policy_is_available(dict(row.get(policy_name) or {}))]
        scores = [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in available_rows]
        plain_romds = [_policy_plain_romd_score((row.get(policy_name) or {})) for row in available_rows]
        returns = [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in available_rows]
        benchmark_gaps = [float((row.get(policy_name) or {}).get("benchmark_0050_gap", 0.0)) for row in available_rows]
        benchmark_plain_romd_gaps = [
            _policy_plain_romd_score((row.get(policy_name) or {})) - float(row.get("benchmark_oos_score", 0.0))
            for row in available_rows
        ]
        chain_policy = dict(chained.get(policy_name) or {})
        chain_available = _policy_is_available(chain_policy)
        period_avg_score = (sum(scores) / float(len(scores))) if scores else 0.0
        period_avg_plain_romd = (sum(plain_romds) / float(len(plain_romds))) if plain_romds else 0.0
        summary[policy_name] = {
            "available": bool(chain_available),
            "chained_oos_score": float(chain_policy.get("rank_1_oos", 0.0)) if chain_available else 0.0,
            "chained_plain_romd_score": _policy_plain_romd_score(chain_policy) if chain_available else 0.0,
            "chained_return_pct": float(chain_policy.get("rank_1_return_pct", 0.0)) if chain_available else 0.0,
            "chained_gap_vs_best_pct": float(chain_policy.get("best_gap_pct", 0.0)) if chain_available else 0.0,
            "chained_gap_vs_0050_pct": float(chain_policy.get("benchmark_0050_gap_pct", 0.0)) if chain_available else 0.0,
            "chained_plain_romd_gap_vs_0050": float(chain_policy.get("benchmark_0050_plain_romd_gap", 0.0)) if chain_available else 0.0,
            "chained_unavailable_reason": str(chain_policy.get("unavailable_reason") or "") if not chain_available else "",
            "period_avg_oos_score": float(period_avg_score),
            "period_avg_plain_romd_score": float(period_avg_plain_romd),
            "yearly_avg_oos_score": float(period_avg_score),
            "yearly_avg_plain_romd_score": float(period_avg_plain_romd),
            "avg_oos_score": float(period_avg_score),
            "avg_plain_romd_score": float(period_avg_plain_romd),
            "median_oos_score": float(statistics.median(scores)) if scores else 0.0,
            "median_plain_romd_score": float(statistics.median(plain_romds)) if plain_romds else 0.0,
            "worst_oos_score": min(scores) if scores else 0.0,
            "worst_plain_romd_score": min(plain_romds) if plain_romds else 0.0,
            "positive_periods": sum(1 for value in returns if value > 0.0),
            "positive_years": sum(1 for value in returns if value > 0.0),
            "win_vs_0050_score": sum(1 for gap in benchmark_gaps if gap > 0.0),
            "win_vs_0050_plain_romd": sum(1 for gap in benchmark_plain_romd_gaps if gap > 0.0),
            "available_periods": len(scores),
            "available_years": len(scores),
            "total_periods": len(rows),
            "total_years": len(rows),
            "period_return_pct": returns,
            "period_oos_score": scores,
            "period_plain_romd_score": plain_romds,
            "yearly_return_pct": returns,
            "yearly_oos_score": scores,
            "yearly_plain_romd_score": plain_romds,
        }
    benchmark_returns = [float(row.get("benchmark_return_pct", 0.0)) for row in rows]
    benchmark_scores = [float(row.get("benchmark_oos_score", 0.0)) for row in rows]
    benchmark_period_avg_score = sum(benchmark_scores) / float(len(benchmark_scores))
    summary["benchmark_0050"] = {
        "chained_oos_score": float(chained.get("benchmark_oos_score", 0.0)),
        "chained_return_pct": float(chained.get("benchmark_return_pct", 0.0)),
        "period_avg_oos_score": float(benchmark_period_avg_score),
        "yearly_avg_oos_score": float(benchmark_period_avg_score),
        "avg_oos_score": float(benchmark_period_avg_score),
        "positive_periods": sum(1 for value in benchmark_returns if value > 0.0),
        "positive_years": sum(1 for value in benchmark_returns if value > 0.0),
        "total_periods": len(benchmark_returns),
        "total_years": len(benchmark_returns),
        "period_return_pct": benchmark_returns,
        "period_oos_score": benchmark_scores,
        "yearly_return_pct": benchmark_returns,
        "yearly_oos_score": benchmark_scores,
    }
    if config is not None:
        summary["window_mode"] = str(config.window_mode)
        summary["train_window_years"] = int(config.train_window_years)
        summary["train_window_months"] = int(config.train_window_months)
        summary["oos_horizon_months"] = int(config.oos_horizon_months)
    return summary

def _format_final_report(rows: list[dict], summary: dict, *, color: bool = False) -> str:
    main_title, retention_title = _active_optimizer_table_titles()
    rendered = _render_optimizer_results_tables(
        rows,
        color=color,
        include_chain=True,
        chained_override=summary.get("chained_oos"),
        main_table_title=main_title,
        retention_table_title=retention_title,
    )
    lines = []
    if rendered:
        lines.append(rendered)
    return "\n".join(lines) + "\n"


























































































































class OptimizerSeedEnsembleProgressBoard:
    """Render fold × seed progress lines from one source for rolling and non-rolling."""

    def __init__(self, contexts: list[dict], *, header: str = "", header_factory=None):
        self.contexts = sorted(
            [dict(item) for item in list(contexts or [])],
            key=lambda item: (int(item.get("fold_idx", 0) or 0), int(item.get("seed_index", 0) or 0)),
        )
        self.header = str(header or "")
        self.header_factory = header_factory
        self.inline = stdout_supports_inline_progress()
        self.rendered_lines = 0
        self.progress_by_key: dict[tuple[int, int], dict] = {}
        self.completed_trials_by_key: dict[tuple[int, int], int] = {}
        self.completed_local_min_trials_by_key: dict[tuple[int, int], int] = {}
        self.completed_local_min_neighbors_by_key: dict[tuple[int, int], int] = {}
        self.search_started_ts_by_key: dict[tuple[int, int], float] = {}
        self.search_last_done_ts_by_key: dict[tuple[int, int], float] = {}
        self.local_min_started_ts_by_key: dict[tuple[int, int], float] = {}
        self.local_min_last_done_ts_by_key: dict[tuple[int, int], float] = {}
        self.fold_progress_by_idx: dict[int, dict] = {}
        self.result_rows_by_fold: dict[int, dict] = {}
        self.completed_replays_by_fold: dict[int, int] = {}
        self.replay_started_ts_by_fold: dict[int, float] = {}
        self.replay_last_done_ts_by_fold: dict[int, float] = {}
        self.last_lines: list[str] = []

    def _context_key(self, context: dict) -> tuple[int, int]:
        return (int(context.get("fold_idx", 0) or 0), int(context.get("seed_index", 0) or 0))

    def _record_progress_metrics(self, key: tuple[int, int], progress: dict) -> None:
        data = dict(progress or {})
        if "ts" not in data:
            data["ts"] = time.time()
        try:
            completed = int(data.get("completed", 0) or 0)
        except (TypeError, ValueError):
            completed = 0
        if completed > int(self.completed_trials_by_key.get(key, 0) or 0):
            self.completed_trials_by_key[key] = int(completed)
        local_completed = _count_local_min_completed_from_progress(data)
        if local_completed > int(self.completed_local_min_trials_by_key.get(key, 0) or 0):
            self.completed_local_min_trials_by_key[key] = int(local_completed)
        local_neighbor_completed = _count_local_min_completed_neighbors_from_progress(data)
        if local_neighbor_completed > int(self.completed_local_min_neighbors_by_key.get(key, 0) or 0):
            self.completed_local_min_neighbors_by_key[key] = int(local_neighbor_completed)
        explicit_search_start = _safe_progress_float(data, "search_started_ts")
        if explicit_search_start is None and str(data.get("stage") or "").upper() == "OPTIMIZER_SEARCH" and completed <= 0:
            explicit_search_start = _safe_progress_ts(data)
        if explicit_search_start is not None:
            previous = self.search_started_ts_by_key.get(key)
            self.search_started_ts_by_key[key] = explicit_search_start if previous is None else min(previous, explicit_search_start)
        explicit_search_done = _safe_progress_float(data, "search_last_done_ts")
        if explicit_search_done is None and str(data.get("stage") or "").upper() == "OPTIMIZER_SEARCH" and completed > 0:
            explicit_search_done = _safe_progress_ts(data)
        if explicit_search_done is not None:
            self.search_last_done_ts_by_key[key] = max(float(self.search_last_done_ts_by_key.get(key, 0.0) or 0.0), explicit_search_done)
        explicit_local_start = _safe_progress_float(data, "local_min_started_ts")
        if explicit_local_start is not None:
            previous = self.local_min_started_ts_by_key.get(key)
            self.local_min_started_ts_by_key[key] = explicit_local_start if previous is None else min(previous, explicit_local_start)
        explicit_local_done = _safe_progress_float(data, "local_min_last_neighbor_done_ts")
        if explicit_local_done is None:
            explicit_local_done = _safe_progress_float(data, "local_min_last_done_ts")
        if explicit_local_done is None and local_neighbor_completed > 0:
            explicit_local_done = _safe_progress_ts(data)
        if explicit_local_done is not None:
            self.local_min_last_done_ts_by_key[key] = max(float(self.local_min_last_done_ts_by_key.get(key, 0.0) or 0.0), explicit_local_done)

    def get_completed_trial_count(self) -> int:
        return sum(int(value or 0) for value in self.completed_trials_by_key.values())

    def get_completed_local_min_trial_count(self) -> int:
        # avg_local is per local-min neighbor, not per finalist.  Keep the method
        # name for caller compatibility while returning the finalized neighbor units.
        return sum(int(value or 0) for value in self.completed_local_min_neighbors_by_key.values())

    def get_completed_local_min_finalist_count(self) -> int:
        return sum(int(value or 0) for value in self.completed_local_min_trials_by_key.values())

    def get_search_wall_elapsed_sec(self) -> float | None:
        if not self.search_started_ts_by_key or not self.search_last_done_ts_by_key:
            return None
        return max(0.0, max(self.search_last_done_ts_by_key.values()) - min(self.search_started_ts_by_key.values()))

    def get_local_min_wall_elapsed_sec(self) -> float | None:
        if not self.local_min_started_ts_by_key or not self.local_min_last_done_ts_by_key:
            return None
        return max(0.0, max(self.local_min_last_done_ts_by_key.values()) - min(self.local_min_started_ts_by_key.values()))

    def get_completed_fold_count(self) -> int:
        return len(self.result_rows_by_fold)

    def _record_replay_metrics(self, fold_idx: int, progress: dict) -> None:
        data = dict(progress or {})
        replay_started = _safe_progress_float(data, "replay_started_ts")
        if replay_started is not None:
            previous = self.replay_started_ts_by_fold.get(fold_idx)
            self.replay_started_ts_by_fold[fold_idx] = replay_started if previous is None else min(previous, replay_started)
        replay_done_ts = _safe_progress_float(data, "replay_last_done_ts")
        ts = _safe_progress_ts(data)
        try:
            replay_done = int(data.get("replay_done", 0) or 0)
        except (TypeError, ValueError):
            replay_done = 0
        if replay_done > int(self.completed_replays_by_fold.get(fold_idx, 0) or 0):
            self.completed_replays_by_fold[fold_idx] = int(replay_done)
        if replay_done_ts is None and replay_done > 0:
            replay_done_ts = ts
        if replay_done_ts is not None:
            self.replay_last_done_ts_by_fold[fold_idx] = max(float(self.replay_last_done_ts_by_fold.get(fold_idx, 0.0) or 0.0), replay_done_ts)

    def get_completed_replay_count(self) -> int:
        return sum(int(value or 0) for value in self.completed_replays_by_fold.values())

    def get_replay_wall_elapsed_sec(self) -> float | None:
        if not self.replay_started_ts_by_fold or not self.replay_last_done_ts_by_fold:
            return None
        return max(0.0, max(self.replay_last_done_ts_by_fold.values()) - min(self.replay_started_ts_by_fold.values()))

    def update_fold_progress(self, *, fold_idx: int, progress: dict, force: bool = False) -> None:
        idx = int(fold_idx)
        payload = dict(progress or {})
        payload.setdefault("fold_idx", idx)
        self.fold_progress_by_idx[idx] = payload
        self._record_replay_metrics(idx, payload)
        self.render(force=force)

    def update_result_row(self, row: dict, *, force: bool = False) -> None:
        payload = dict(row or {})
        try:
            fold_idx = int(payload.get("fold_idx", 0) or 0)
        except (TypeError, ValueError):
            fold_idx = 0
        if fold_idx <= 0:
            fold_idx = 1
        self.result_rows_by_fold[fold_idx] = payload
        self.fold_progress_by_idx[fold_idx] = {
            "stage": "FOLD_RESULT",
            "status": "fold result ready",
            "fold_idx": fold_idx,
            "fold_count": max((int(ctx.get("fold_count", 0) or 0) for ctx in self.contexts), default=fold_idx),
            "oos_year": int(payload.get("oos_year", 0) or 0),
            "elapsed_sec": payload.get("elapsed_sec"),
        }
        self.render(force=force)

    def update(self, *, fold_idx: int, seed_index: int, progress: dict, force: bool = False) -> None:
        key = (int(fold_idx), int(seed_index))
        self.progress_by_key[key] = dict(progress or {})
        self._record_progress_metrics(key, dict(progress or {}))
        self.render(force=force)

    def update_many(self, progress_map: dict[tuple[int, int], dict], *, force: bool = False) -> None:
        for key, progress in dict(progress_map or {}).items():
            try:
                normalized_key = (int(key[0]), int(key[1]))
            except (TypeError, ValueError, IndexError):
                continue
            self.progress_by_key[normalized_key] = dict(progress or {})
            self._record_progress_metrics(normalized_key, dict(progress or {}))
        self.render(force=force)

    def _build_lines(self) -> list[str]:
        header_text = str(self.header_factory(self) if callable(self.header_factory) else self.header)
        progress_by_key = {
            self._context_key(context): dict(self.progress_by_key.get(self._context_key(context)) or {"stage": "QUEUED", "status": "queued"})
            for context in self.contexts
        }
        fold_progress_lines: list[str] = []
        context_by_fold = {int(ctx.get("fold_idx", 0) or 0): dict(ctx) for ctx in self.contexts}
        for fold_idx, progress in sorted(self.fold_progress_by_idx.items()):
            stage = str((progress or {}).get("stage") or "").upper()
            if stage not in {"ENSEMBLE_REPLAY", "FOLD_RESULT"}:
                continue
            context = dict(context_by_fold.get(int(fold_idx)) or {})
            task = {
                "fold_idx": int(fold_idx),
                "fold_count": int(context.get("fold_count", progress.get("fold_count", 0)) or 0),
                "oos_year": int(context.get("oos_year", progress.get("oos_year", 0)) or 0),
                "oos_period": str(context.get("oos_period") or progress.get("oos_period") or ""),
                "selection_period": str(context.get("selection_period") or context.get("selection_start") or ""),
                "selection_start_date": str(context.get("selection_start") or ""),
                "selection_end_date": str(context.get("selection_end") or ""),
                "show_oos": bool(context.get("show_oos", True)),
            }
            fold_progress_lines.append(_format_parallel_fold_progress_line(task, dict(progress or {})))
        table_text = ""
        if self.result_rows_by_fold:
            main_title, retention_title = _active_optimizer_table_titles()
            show_oos_avg = any(bool(ctx.get("show_oos", True)) for ctx in self.contexts)
            table_text = _render_optimizer_results_tables(
                sorted((normalize_optimizer_seed_ensemble_fold_row(row) for row in self.result_rows_by_fold.values()), key=optimizer_seed_ensemble_row_sort_key),
                color=True,
                include_chain=False,
                include_oos_avg=show_oos_avg,
                main_table_title=main_title,
                retention_table_title=retention_title,
            )
        return build_optimizer_seed_ensemble_live_lines(
            header_text=header_text,
            seed_contexts=self.contexts,
            seed_progress_by_key=progress_by_key,
            fold_progress_lines=fold_progress_lines,
            table_text=table_text,
            color=True,
        )

    def render(self, *, force: bool = False) -> None:
        lines = self._build_lines()
        if not force and lines == self.last_lines:
            return
        self.last_lines = list(lines)
        if self.inline:
            if self.rendered_lines > 0:
                sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            for line in lines:
                sys.stdout.write("\r" + line + "\x1b[K\n")
            if self.rendered_lines > len(lines):
                for _ in range(self.rendered_lines - len(lines)):
                    sys.stdout.write("\r\x1b[K\n")
            sys.stdout.flush()
            self.rendered_lines = len(lines)
        else:
            print("\n".join(lines), flush=True)
            self.rendered_lines = 0

    def close(self) -> None:
        if self.inline and self.rendered_lines > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.rendered_lines = 0








class _ParallelFoldLiveBoard:
    """Render parallel-fold progress and completed results as one refresh block."""

    def __init__(self, tasks: list[dict], *, overall_start: float | None = None, raw_data_load_sec: float = 0.0):
        self.tasks = sorted(list(tasks or []), key=lambda item: int(item.get("fold_idx", 0) or 0))
        self.inline = stdout_supports_inline_progress()
        self.started_at = time.perf_counter()
        self.overall_start = float(overall_start) if overall_start is not None else self.started_at
        self.raw_data_load_sec = max(0.0, float(raw_data_load_sec or 0.0))
        self.rendered_lines = 0
        self.last_lines: list[str] = []
        self.last_render_key: list[str] = []

    def _build_lines(self, *, pending: set, future_map: dict, completed_rows: list[dict], fold_timing_rows: list[dict] | None = None) -> list[str]:
        completed_rows_sorted = sorted((normalize_optimizer_seed_ensemble_fold_row(item) for item in list(completed_rows or [])), key=optimizer_seed_ensemble_row_sort_key)
        completed_oos = {int(row.get("oos_year", 0) or 0) for row in completed_rows_sorted}
        fold_wall_elapsed = max(0.0, time.perf_counter() - self.started_at)
        total_elapsed = max(0.0, time.perf_counter() - self.overall_start)
        setup_elapsed = max(0.0, self.started_at - self.overall_start)
        # parallel fold 模式下，raw data 由各 fold worker 自行載入，
        # worker raw 時間已包含在 fold_time wall 裡；不要再以 raw/other 顯示，避免與總時間對帳時重複或缺項。
        total_folds = max((int(task.get("fold_count", 0) or 0) for task in self.tasks), default=len(self.tasks)) or len(self.tasks)
        seed_ensemble_display = _is_rolling_random_seed_ensemble_enabled()
        live_result_rows = _merge_live_result_rows(completed_rows_sorted, self.tasks)
        table_text = ""
        if live_result_rows:
            main_title, retention_title = _active_optimizer_table_titles()
            table_text = _render_optimizer_results_tables(
                live_result_rows,
                color=True,
                include_chain=False,
                include_oos_avg=True,
                main_table_title=main_title,
                retention_table_title=retention_title,
            )
        if seed_ensemble_display:
            seed_policy = _build_rolling_seed_ensemble_policy_payload()
            seed_count = int(seed_policy.get("seed_count", 1) or 1)
            seed_progress_maps = {id(task): _read_latest_parallel_fold_seed_progresses_for_task(task) for task in self.tasks}
            phase_metrics = _collect_seed_progress_phase_metrics(
                progress
                for progress_map in seed_progress_maps.values()
                for progress in dict(progress_map or {}).values()
            )
            completed_trials = int(phase_metrics.get("completed_trials", 0) or 0)
            replay_metrics = _collect_parallel_fold_replay_phase_metrics(self.tasks)
            header = format_optimizer_seed_ensemble_progress_header(
                folds=total_folds,
                seeds=seed_count,
                min_agree=int(seed_policy.get("min_agree", seed_count) or seed_count),
                parallel_workers=resolve_optimizer_random_seed_ensemble_parallel_workers_default(seed_count),
                backend=resolve_optimizer_random_seed_ensemble_parallel_backend_default(),
                completed_folds=len(completed_rows_sorted),
                pending_folds=len(pending),
                total_elapsed_sec=total_elapsed,
                fold_elapsed_sec=fold_wall_elapsed,
                setup_elapsed_sec=setup_elapsed,
                completed_trials=completed_trials,
                search_wall_elapsed_sec=phase_metrics.get("search_wall_elapsed_sec"),
                completed_local_min_trials=int(phase_metrics.get("completed_local_min_trials", 0) or 0),
                local_min_wall_elapsed_sec=phase_metrics.get("local_min_wall_elapsed_sec"),
                completed_replays=int(replay_metrics.get("completed_replays", 0) or 0),
                replay_wall_elapsed_sec=replay_metrics.get("replay_wall_elapsed_sec"),
            )
            seed_contexts: list[dict] = []
            seed_progress_by_key: dict[tuple[int, int], dict] = {}
            fold_progress_lines: list[str] = []
            for task in self.tasks:
                seed_progresses = dict(seed_progress_maps.get(id(task)) or {})
                fold_completed = int(task.get("oos_year", 0) or 0) in completed_oos
                for seed_index in range(1, seed_count + 1):
                    progress = dict(seed_progresses.get(seed_index) or {})
                    if not progress:
                        progress = {"stage": "DONE" if fold_completed else "QUEUED", "status": "done" if fold_completed else "queued"}
                    context = {
                        "fold_idx": int(task.get("fold_idx", 0) or 0),
                        "fold_count": int(task.get("fold_count", 0) or 0),
                        "seed_index": int(seed_index),
                        "seed_count": int(seed_count),
                        "seed": progress.get("seed"),
                        "oos_year": int(task.get("oos_year", 0) or 0),
                        "oos_period": str(task.get("oos_period") or ""),
                        "selection_start": str(task.get("selection_start_date") or task.get("selection_period") or ""),
                        "selection_end": str(task.get("selection_end_date") or ""),
                        "selection_period": str(task.get("selection_period") or ""),
                    }
                    seed_contexts.append(context)
                    seed_progress_by_key[_seed_progress_context_key(context)] = progress
                fold_progress = _read_latest_parallel_fold_progress(str(task.get("log_path") or ""))
                fold_stage = str(fold_progress.get("stage") or "").upper()
                if fold_progress and fold_stage in {"ENSEMBLE_REPLAY", "FOLD_RESULT"}:
                    fold_progress_lines.append(_format_parallel_fold_progress_line(task, fold_progress))
            lines = build_optimizer_seed_ensemble_live_lines(
                header_text=header,
                seed_contexts=seed_contexts,
                seed_progress_by_key=seed_progress_by_key,
                fold_progress_lines=fold_progress_lines,
                table_text=table_text,
                color=True,
            )
        else:
            _ = (pending, fold_wall_elapsed, setup_elapsed)
            header = (
                f"⏱️ Rolling fold parallel | completed={len(completed_rows_sorted)}/{total_folds} | "
                f"total_time={_fmt_duration(total_elapsed)}"
            )
            lines = [f"{C_CYAN}{header}{C_RESET}"]
            for task in self.tasks:
                log_path = str(task.get("log_path") or "")
                progress = _read_latest_parallel_fold_progress(log_path)
                if int(task.get("oos_year", 0) or 0) in completed_oos and not progress:
                    progress = {
                        "stage": "DONE",
                        "status": "done",
                        "fold_idx": int(task.get("fold_idx", 0) or 0),
                        "fold_count": int(task.get("fold_count", 0) or 0),
                        "oos_year": int(task.get("oos_year", 0) or 0),
                    }
                log_status = _latest_parallel_fold_log_status(log_path)
                lines.append(f"{C_GRAY}  {_format_parallel_fold_progress_line(task, progress, log_status=log_status)}{C_RESET}")
        if (not seed_ensemble_display) and table_text:
            lines.append("")
            lines.extend(table_text.splitlines())
        return lines

    @staticmethod
    def _stable_render_key(lines: list[str]) -> list[str]:
        # Ignore elapsed-only heartbeat changes; refresh when fold progress or completed results change.
        key: list[str] = []
        for line in list(lines or []):
            text = str(line)
            if "⏱️ 耗時摘要" in text:
                text = "⏱️ 耗時摘要"
            elif "seed ensemble |" in text or "⏱️ Rolling fold parallel" in text or text.lstrip().startswith("folds="):
                for marker in (" | total_time=", " | total=", " | elapsed="):
                    if marker in text:
                        text = text.split(marker, 1)[0]
                        break
            elif " | elapsed=" in text:
                text = text.split(" | elapsed=", 1)[0]
            key.append(text)
        return key

    def render(self, *, pending: set, future_map: dict, completed_rows: list[dict], fold_timing_rows: list[dict] | None = None, force: bool = False) -> None:
        lines = self._build_lines(pending=pending, future_map=future_map, completed_rows=completed_rows, fold_timing_rows=fold_timing_rows)
        render_key = self._stable_render_key(lines)
        if not force and render_key == self.last_render_key:
            return
        self.last_lines = list(lines)
        self.last_render_key = list(render_key)
        if self.inline:
            if self.rendered_lines > 0:
                sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            for line in lines:
                sys.stdout.write("\r" + line + "\x1b[K\n")
            if self.rendered_lines > len(lines):
                for _ in range(self.rendered_lines - len(lines)):
                    sys.stdout.write("\r\x1b[K\n")
            sys.stdout.flush()
            self.rendered_lines = len(lines)
        else:
            # Non-interactive outputs cannot refresh safely; print only on meaningful changes.
            print("\n".join(lines), flush=True)
            self.rendered_lines = 0

    def close(self) -> None:
        if self.inline and self.rendered_lines > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.rendered_lines = 0



def _print_parallel_fold_result(row: dict) -> None:
    if not row:
        return
    print(f"{C_CYAN}📌 Parallel fold result | fold={row.get('fold')} | selection={_display_month_period(row.get('selection_period'))} | OOS={_display_month_period(row.get('oos_period') or row.get('oos_year'))}{C_RESET}", flush=True)
    rendered = _render_optimizer_results_tables([row], color=True, include_chain=False)
    if rendered:
        print(rendered, flush=True)


def _consume_parallel_fold_result(*, result: dict, rows: list[dict], fold_timing_rows: list[dict], chain_state: dict) -> None:
    result = dict(result or {})
    row = result.get("row")
    if row is not None:
        rows.append(row)
    if result.get("timing_row") is not None:
        fold_timing_rows.append(result["timing_row"])
    if result.get("chain_max_positions") is not None:
        chain_state["chain_max_positions"] = int(result.get("chain_max_positions"))
    if result.get("chain_enable_rotation") is not None:
        chain_state["chain_enable_rotation"] = bool(result.get("chain_enable_rotation"))


def _consume_parallel_fold_future(*, future, task: dict, rows: list[dict], fold_timing_rows: list[dict], chain_state: dict) -> None:
    try:
        result = future.result()
    except Exception as exc:
        raise RuntimeError(_build_parallel_fold_failure_message(task=task, exc=exc)) from exc
    _consume_parallel_fold_result(result=result, rows=rows, fold_timing_rows=fold_timing_rows, chain_state=chain_state)


def _cancel_parallel_fold_executor_after_failure(executor, pending: set) -> None:
    for future in list(pending or []):
        try:
            future.cancel()
        except Exception as exc:
            _format_exception_summary(exc)
    processes = getattr(executor, "_processes", None)
    live_processes = []
    if isinstance(processes, dict):
        for process in list(processes.values()):
            try:
                if process is not None and process.is_alive():
                    live_processes.append(process)
                    process.terminate()
            except Exception as exc:
                _format_exception_summary(exc)
        for process in list(live_processes):
            try:
                process.join(timeout=5.0)
            except Exception as exc:
                _format_exception_summary(exc)
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(wait=False, cancel_futures=True)
        except TypeError:
            try:
                shutdown(wait=False)
            except Exception as exc:
                _format_exception_summary(exc)
        except Exception as exc:
            _format_exception_summary(exc)


def _run_missing_parallel_fold_tasks_sequentially(*, tasks: list[dict], rows: list[dict], fold_timing_rows: list[dict], chain_state: dict, cause: BaseException | None = None) -> dict:
    completed_oos = {int((row or {}).get("oos_year", 0) or 0) for row in list(rows or []) if row}
    missing_tasks = [dict(task) for task in list(tasks or []) if int(task.get("oos_year", 0) or 0) not in completed_oos]
    if not missing_tasks:
        return chain_state
    if _is_non_retryable_fold_failure(cause):
        raise RuntimeError(
            "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR: parallel fold發生runtime identity／manifest錯誤；"
            "序列fallback不會改變結果，已直接中止。"
            f" cause={_format_exception_summary(cause)}"
        ) from cause
    cause_text = f" | cause={_format_exception_summary(cause)}" if cause is not None else ""
    print(f"{C_YELLOW}⚠️ parallel fold 中斷，改用單 fold fallback 跑完剩餘 {len(missing_tasks)} 個 fold{cause_text}{C_RESET}", flush=True)
    for task in missing_tasks:
        fallback_task = dict(task)
        fallback_task["log_path"] = _parallel_fold_log_path_for_fallback(str(task.get("log_path") or ""))
        fallback_task["force_safe_sequential"] = True
        fallback_task["safe_sequential_fallback"] = True
        fallback_task["fold_workers"] = 1
        print(
            f"{C_CYAN}↪ fallback fold | fold={int(task.get('fold_idx', 0) or 0)}/{int(task.get('fold_count', 0) or 0)} "
            f"| OOS={_display_month_period(task.get('oos_period') or task.get('oos_year'))} | log={fallback_task.get('log_path')}{C_RESET}",
            flush=True,
        )
        try:
            result = _run_outer_rolling_oos_fold_task(fallback_task)
        except Exception as exc:
            raise RuntimeError(_build_parallel_fold_failure_message(task=fallback_task, exc=exc, label="sequential fallback fold failed")) from exc
        _consume_parallel_fold_result(result=result, rows=rows, fold_timing_rows=fold_timing_rows, chain_state=chain_state)
        row = (result or {}).get("row")
        if row is not None:
            _print_parallel_fold_result(row)
    return chain_state


def _run_parallel_fold_futures(*, executor, tasks: list[dict], rows: list[dict], fold_timing_rows: list[dict], overall_start: float | None = None, raw_data_load_sec: float = 0.0) -> dict:
    future_map = {executor.submit(_run_outer_rolling_oos_fold_task, task): task for task in tasks}
    pending = set(future_map)
    chain_state = {"chain_max_positions": None, "chain_enable_rotation": None}
    live_board = _ParallelFoldLiveBoard(tasks, overall_start=overall_start, raw_data_load_sec=raw_data_load_sec)
    live_board.render(pending=pending, future_map=future_map, completed_rows=rows, fold_timing_rows=fold_timing_rows, force=True)
    try:
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: int(future_map[item].get("fold_idx", 0) or 0)):
                try:
                    _consume_parallel_fold_future(
                        future=future,
                        task=future_map[future],
                        rows=rows,
                        fold_timing_rows=fold_timing_rows,
                        chain_state=chain_state,
                    )
                except Exception as exc:
                    _cancel_parallel_fold_executor_after_failure(executor, pending)
                    raise RuntimeError(_build_parallel_fold_failure_message(task=future_map[future], exc=exc)) from exc
            live_board.render(pending=pending, future_map=future_map, completed_rows=rows, fold_timing_rows=fold_timing_rows, force=bool(done))
    finally:
        live_board.close()
    if _is_rolling_random_seed_ensemble_enabled():
        phase_metrics = _collect_seed_progress_phase_metrics(
            progress
            for task in list(tasks or [])
            for progress in dict(_read_latest_parallel_fold_seed_progresses_for_task(task) or {}).values()
        )
        replay_metrics = _collect_parallel_fold_replay_phase_metrics(list(tasks or []))
        chain_state["seed_ensemble_performance_metrics"] = {
            **dict(phase_metrics or {}),
            **dict(replay_metrics or {}),
        }
    return chain_state









def _effective_rolling_seed_ensemble_policy_payload() -> dict:
    """Return the effective runtime seed policy, not merely the configured capacity."""
    requested = _build_seed_ensemble_policy_payload()
    if _is_rolling_random_seed_ensemble_enabled():
        return dict(requested)
    effective = build_seed_ensemble_policy_snapshot(enabled=False, seed_count=1, min_agree=1)
    effective["seed_mode"] = "single_canonical_optimizer_seed"
    effective["requested_random_seed_ensemble"] = dict(requested)
    return effective


_build_rolling_seed_ensemble_policy_payload = _build_seed_ensemble_policy_payload


def _extract_seed_from_policy_schedules(row: dict) -> int | None:
    for schedule in dict(row.get("policy_schedules") or {}).values():
        if not isinstance(schedule, dict):
            continue
        raw_seed = schedule.get("optimizer_seed")
        if raw_seed is None:
            continue
        try:
            return int(raw_seed)
        except (TypeError, ValueError):
            continue
    return None


def _build_members_from_seed_rows_for_policy(seed_rows: list[dict], policy_name: str) -> list[dict]:
    members: list[dict] = []
    for idx, row in enumerate(list(seed_rows or []), start=1):
        schedule = dict((row.get("policy_schedules") or {}).get(str(policy_name)) or {})
        explicit_members = _build_params_ensemble_members_for_schedule(schedule) if schedule.get("params_ensemble") else []
        if explicit_members:
            fallback_seed = schedule.get("optimizer_seed")
            if fallback_seed is None:
                fallback_seed = _extract_seed_from_policy_schedules(row)
            for raw_member in explicit_members:
                member = dict(raw_member)
                member["member_index"] = int(len(members) + 1)
                if member.get("seed") is None:
                    member["seed"] = fallback_seed
                if member.get("optimizer_seed") is None:
                    member["optimizer_seed"] = fallback_seed
                member["policy"] = str(policy_name)
                members.append(member)
            continue
        params_payload = dict(schedule.get("params") or {})
        if not params_payload:
            continue
        seed = schedule.get("optimizer_seed")
        if seed is None:
            seed = _extract_seed_from_policy_schedules(row)
        member = {
            "member_index": int(len(members) + 1),
            "seed": seed,
            "selected_trial": schedule.get("selected_trial"),
            "params": params_payload,
        }
        for key in (
            "base_score",
            "base_rank",
            "local_min",
            "local_rank",
            "retention",
            "retention_rank",
            "local_min_review_enabled",
            "local_min_review_mode",
            "local_min_exact",
            "selection_rule",
            "policy_type",
            "base_agree_seed_finalist_count",
            "base_agree_seed_base_score_sum",
            "base_agree_seed_selected_trials",
            "base_agree_min_agree_requested",
            "local_agree_seed_finalist_count",
            "local_agree_seed_local_min_sum",
            "local_agree_seed_selected_trials",
            "local_agree_min_agree_requested",
            "retention_agree_seed_finalist_count",
            "retention_agree_seed_retention_sum",
            "retention_agree_seed_selected_trials",
            "retention_agree_min_agree_requested",
        ):
            if key in schedule:
                member[key] = schedule.get(key)
        members.append(member)
    normalized = renumber_seed_ensemble_members(members)
    if _is_finalists_agree_policy(policy_name):
        return select_finalists_agree_members(normalized, policy_name=str(policy_name))
    return normalized




def _build_single_period_ensemble_payload(*, members: list[dict], effective_start: str, effective_end: str, oos_year: int, policy_name: str | None = None, raw_universe_required_min_rows=None) -> dict:
    normalized_members = renumber_seed_ensemble_members(members)
    if not normalized_members:
        raise ValueError("rolling seed ensemble 缺少可用 params members")
    params_ensemble_by_effective_date = {str(effective_start): normalized_members}
    return {
        **_raw_universe_contract_fields(raw_universe_required_min_rows),
        "schema_type": ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
        "schema_version": 1,
        "mode": ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
        "type": "outer_rolling_oos_param_set",
        "selector": str(policy_name or ""),
        "active_param_policy": "daily_active_param_ensemble",
        "random_seed_ensemble": _build_effective_seed_ensemble_policy_payload(
            params_ensemble_by_effective_date,
            policy_name=str(policy_name or ""),
        ),
        "params_ensemble_by_effective_date": params_ensemble_by_effective_date,
        "folds": [{
            "oos_year": int(oos_year),
            "effective_start": str(effective_start),
            "effective_end": str(effective_end),
            "oos_start_date": str(effective_start),
            "oos_end_date": str(effective_end),
        }],
    }




def _build_policy_replay_context(
    *,
    selected_data_dir: str,
    oos_year: int,
    oos_start_date: str,
    oos_end_date: str,
    max_positions: int,
    enable_rotation: bool,
    raw_universe_required_min_rows=None,
) -> dict:
    """Build the single replay context shared by all policies in a fold.

    The process backend cannot share large in-memory market contexts across workers;
    those are still reused through the existing portfolio prepared/raw cache.  This
    object is the canonical policy-replay input schema so rolling/non-rolling do not
    rebuild date/position inputs differently.
    """
    _ = is_optimizer_policy_replay_context_reuse_enabled_default()
    return {
        "selected_data_dir": str(selected_data_dir),
        "oos_year": int(oos_year),
        "oos_start_date": str(oos_start_date),
        "oos_end_date": str(oos_end_date),
        "max_positions": int(max_positions),
        "enable_rotation": bool(enable_rotation),
        RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD: coerce_raw_universe_required_min_rows(raw_universe_required_min_rows, allow_none=True),
        "start_year": int(pd.Timestamp(oos_start_date).year),
        "end_year": int(pd.Timestamp(oos_end_date).year),
        "benchmark_ticker": "0050",
    }


def _stable_policy_replay_signature(payload: dict) -> str:
    material = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _build_policy_replay_payload_from_members(*, members: list[dict], replay_context: dict, policy_name: str | None = None) -> dict:
    return _build_single_period_ensemble_payload(
        members=members,
        effective_start=str(replay_context["oos_start_date"]),
        effective_end=str(replay_context["oos_end_date"]),
        oos_year=int(replay_context["oos_year"]),
        policy_name=str(policy_name or ""),
        raw_universe_required_min_rows=replay_context.get(RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD),
    )


def _evaluate_active_param_ensemble_replay_payload_task(task: dict) -> dict:
    from services.portfolio_replay import run_portfolio_simulation_with_param_ensemble

    payload = dict(task.get("payload") or {})
    replay_context = dict(task.get("replay_context") or {})
    result = run_portfolio_simulation_with_param_ensemble(
        str(replay_context["selected_data_dir"]),
        payload,
        max_positions=int(replay_context["max_positions"]),
        enable_rotation=bool(replay_context["enable_rotation"]),
        start_year=int(replay_context["start_year"]),
        end_year=int(replay_context["end_year"]),
        start_date=str(replay_context["oos_start_date"]),
        end_date=str(replay_context["oos_end_date"]),
        benchmark_ticker=str(replay_context.get("benchmark_ticker") or "0050"),
        verbose=False,
        use_prepared_cache=False,
        write_prepared_cache=False,
    )
    return {
        "signature": str(task.get("signature") or ""),
        "policy_names": list(task.get("policy_names") or []),
        "metrics": _extract_active_replay_metrics(result),
    }


def _policy_replay_executor_class(backend: str):
    return ThreadPoolExecutor if str(backend or "").strip().lower() == "thread" else ProcessPoolExecutor


def _is_safe_sequential_fold_task(task: dict | None) -> bool:
    return bool((task or {}).get("force_safe_sequential") or (task or {}).get("safe_sequential_fallback"))


def _should_disable_nested_parallelism(task: dict | None) -> bool:
    if _is_safe_sequential_fold_task(task):
        return True
    try:
        return int((task or {}).get("fold_workers", 1) or 1) > 1
    except (TypeError, ValueError):
        return False


def _resolve_fold_seed_ensemble_parallel_workers(task: dict | None, seed_count: int) -> int:
    workers = resolve_optimizer_random_seed_ensemble_parallel_workers_default(int(seed_count))
    if _should_disable_nested_parallelism(task):
        return 1
    return max(1, int(workers))


def _run_policy_replay_tasks(
    replay_tasks: list[dict],
    *,
    progress_emit=None,
) -> dict[str, dict]:
    tasks = [dict(task) for task in list(replay_tasks or [])]
    if not tasks:
        return {}
    replay_count = len(tasks)
    backend = resolve_optimizer_policy_replay_parallel_backend_default()
    force_serial = any(bool(task.get("force_serial_policy_replay")) for task in tasks)
    workers = 1 if force_serial else resolve_optimizer_policy_replay_parallel_workers_default(replay_count)
    workers = max(1, min(int(workers), replay_count))
    results: dict[str, dict] = {}
    completed = 0

    def _label(task: dict) -> str:
        names = [str(name) for name in list(task.get("policy_names") or []) if str(name)]
        return "/".join(names) if names else str(task.get("signature") or "policy")[:12]

    if workers <= 1:
        for task in tasks:
            if callable(progress_emit):
                progress_emit(_label(task), done=completed, total=replay_count, status="RUN")
            output = _evaluate_active_param_ensemble_replay_payload_task(task)
            completed += 1
            results[str(output.get("signature") or task.get("signature"))] = dict(output.get("metrics") or {})
            if callable(progress_emit):
                progress_emit(_label(task), done=completed, total=replay_count, status="DONE")
        return results

    executor_class = _policy_replay_executor_class(backend)
    if executor_class is ProcessPoolExecutor:
        executor_kwargs, _start_method = get_process_pool_executor_kwargs()
    else:
        executor_kwargs = {}
    with executor_class(max_workers=workers, **executor_kwargs) as executor:
        future_to_task = {}
        for task in tasks:
            if callable(progress_emit):
                progress_emit(_label(task), done=completed, total=replay_count, status="RUN")
            future_to_task[executor.submit(_evaluate_active_param_ensemble_replay_payload_task, task)] = task
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            output = future.result()
            completed += 1
            results[str(output.get("signature") or task.get("signature"))] = dict(output.get("metrics") or {})
            if callable(progress_emit):
                progress_emit(_label(task), done=completed, total=replay_count, status="DONE")
    return results


def _policy_metrics_from_ensemble_metrics(metrics: dict, *, best_score: float, benchmark_score: float) -> dict:
    rank_1_oos = float(metrics.get("score", 0.0))
    rank_1_plain_romd = float(metrics.get("plain_romd_score", calc_plain_romd(metrics.get("return_pct", 0.0), metrics.get("mdd_pct", 0.0))))
    return {
        "available": True,
        "rank_1_trial": None,
        "rank_1_oos": rank_1_oos,
        "rank_1_plain_romd": rank_1_plain_romd,
        "rank_1_return_pct": float(metrics.get("return_pct", 0.0)),
        "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
        "rank_1_trades": int(metrics.get("trade_count", 0) or 0),
        "rank_1_total_r": float(metrics.get("portfolio_total_r", 0.0)),
        "rank_1_median_r": float(metrics.get("portfolio_median_r", 0.0)),
        "rank_1_score_total_r": float(metrics.get("score_total_r", 0.0)),
        "rank_1_score_median_r": float(metrics.get("score_median_r", 0.0)),
        "rank_1_score_r_source": str(metrics.get("score_r_source", "single_stock")),
        "best_gap": rank_1_oos - float(best_score),
        "benchmark_0050_gap": rank_1_oos - float(benchmark_score),
        "benchmark_0050_plain_romd_gap": rank_1_plain_romd - float(benchmark_score),
        "rank_1_initial_capital": float(metrics.get("initial_equity", 0.0) or 0.0),
        "rank_1_equity_curve": [],
        "ensemble_replay": True,
    }


def _build_ensemble_policy_schedule_from_seed_rows(*, seed_rows: list[dict], policy_name: str, oos_year: int, selection_period: str, oos_start_date: str, oos_end_date: str) -> dict:
    members = _build_members_from_seed_rows_for_policy(seed_rows, policy_name)
    if _is_finalist_best_policy(policy_name):
        members = select_finalist_best_members(members, policy_name=policy_name)
    members = renumber_seed_ensemble_members(members)
    if not members:
        return {}
    first_member = dict(members[0])
    optimizer_seeds = sorted({member.get("seed") for member in members if member.get("seed") is not None})
    schedule = {
        "effective_start": str(oos_start_date),
        "effective_end": str(oos_end_date),
        "selection": str(selection_period),
        "oos_year": int(oos_year),
        "oos_period": f"{oos_start_date}~{oos_end_date}",
        "policy": str(policy_name),
        "selected_trial": first_member.get("selected_trial"),
        "optimizer_seed": optimizer_seeds[0] if optimizer_seeds else first_member.get("seed"),
        "optimizer_seeds": optimizer_seeds,
        "member_count": int(len(members)),
        "params": dict(first_member.get("params") or {}),
        "params_ensemble": members,
    }
    if _is_finalists_agree_policy(policy_name):
        finalists_agree_config = _finalists_agree_policy_config(policy_name)
        schedule["policy_type"] = "selected_seed_finalist_ensemble"
        schedule["selection_rule"] = str(finalists_agree_config["selection_rule"])
        schedule["min_agree"] = int(_resolve_finalists_agree_min_agree(policy_name, len(members)))
        schedule["min_agree_requested"] = finalists_agree_config["min_agree_requested"]
        schedule[str(finalists_agree_config["min_agree_requested_key"])] = finalists_agree_config["min_agree_requested"]
        schedule["member_selection"] = str(finalists_agree_config["member_selection"])
        schedule["intra_seed_agree"] = "selected_seed_finalists"
        for key in _finalists_agree_metadata_keys():
            if key in first_member:
                schedule[key] = first_member.get(key)
    return schedule


def _aggregate_seed_ensemble_fold_results(*, task: dict, seed_results: list[dict]) -> dict:
    seed_rows = [dict(result.get("row") or {}) for result in list(seed_results or []) if result.get("row")]
    timing_rows = [dict(result.get("timing_row") or {}) for result in list(seed_results or []) if result.get("timing_row")]
    if not seed_rows:
        first = dict(seed_results[0]) if seed_results else {}
        first["status"] = "skipped_no_finalist"
        return first
    if len(seed_rows) != len(list(seed_results or [])):
        raise RuntimeError(
            f"rolling seed ensemble 需要完整 N 組 params，但只有 {len(seed_rows)}/{len(list(seed_results or []))} 個 seed 產生可用 row"
        )

    selected_data_dir = str(task["selected_data_dir"])
    fold_idx = int(task["fold_idx"])
    fold_count = int(task["fold_count"])
    oos_year = int(task["oos_year"])
    selection_period = str(task.get("selection_period") or seed_rows[0].get("selection_period") or "")
    selection_start = str(task.get("selection_start_date") or seed_rows[0].get("selection_start_date") or "")
    selection_end = str(task.get("selection_end_date") or seed_rows[0].get("selection_end_date") or "")
    oos_start_date = str(task.get("oos_start_date") or seed_rows[0].get("oos_start_date") or f"{oos_year}-01-01")
    oos_end_date = str(task.get("oos_end_date") or seed_rows[0].get("oos_end_date") or f"{oos_year}-12-31")
    chain_max_positions = int((seed_results[0] or {}).get("chain_max_positions", DEFAULT_PORTFOLIO_MAX_POSITIONS) or DEFAULT_PORTFOLIO_MAX_POSITIONS)
    chain_enable_rotation = bool((seed_results[0] or {}).get("chain_enable_rotation", False))

    eval_started = time.perf_counter()
    replay_started_ts = time.time()
    replay_last_done_ts: float | None = None

    replay_context = _build_policy_replay_context(
        selected_data_dir=selected_data_dir,
        oos_year=oos_year,
        oos_start_date=oos_start_date,
        oos_end_date=oos_end_date,
        max_positions=chain_max_positions,
        enable_rotation=chain_enable_rotation,
        raw_universe_required_min_rows=task.get("optimizer_required_min_rows"),
    )

    policy_schedules: dict[str, dict] = {}
    policy_jobs_by_name: dict[str, dict] = {}
    unique_jobs_by_signature: OrderedDict[str, dict] = OrderedDict()
    dedup_enabled = is_optimizer_policy_replay_dedup_by_signature_enabled_default()
    for policy_name in CHAIN_POLICY_NAMES:
        members = _build_members_from_seed_rows_for_policy(seed_rows, policy_name)
        if _is_finalist_best_policy(policy_name):
            members = select_finalist_best_members(members, policy_name=policy_name)
        if not members:
            continue
        schedule = _build_ensemble_policy_schedule_from_seed_rows(
            seed_rows=seed_rows,
            policy_name=policy_name,
            oos_year=oos_year,
            selection_period=selection_period,
            oos_start_date=oos_start_date,
            oos_end_date=oos_end_date,
        )
        policy_schedules[policy_name] = schedule
        payload = _build_policy_replay_payload_from_members(members=members, replay_context=replay_context, policy_name=policy_name)
        signature = _stable_policy_replay_signature(payload) if dedup_enabled else f"{policy_name}:{_stable_policy_replay_signature(payload)}"
        job = policy_jobs_by_name[policy_name] = {
            "signature": signature,
            "policy_name": policy_name,
            "policy_names": [policy_name],
            "payload": payload,
            "replay_context": replay_context,
            "members": members,
            "force_serial_policy_replay": bool(_should_disable_nested_parallelism(task)),
        }
        if signature in unique_jobs_by_signature:
            unique_jobs_by_signature[signature].setdefault("policy_names", []).append(policy_name)
            continue
        unique_jobs_by_signature[signature] = dict(job)

    replay_total = len(unique_jobs_by_signature)
    replay_done = 0

    def _emit_ensemble_replay_progress(policy_name: str, *, done: int | None = None, total: int | None = None, status: str = "RUN") -> None:
        nonlocal replay_last_done_ts, replay_done
        done_value = int(replay_done if done is None else done)
        if done_value > 0:
            replay_last_done_ts = time.time()
        _write_parallel_fold_progress_event(
            stage="ENSEMBLE_REPLAY",
            fold_idx=fold_idx,
            fold_count=fold_count,
            oos_year=oos_year,
            selection_start=selection_start,
            selection_end=selection_end,
            status=str(status),
            policy=str(policy_name),
            replay_done=int(done_value),
            replay_total=int(replay_total if total is None else total),
            replay_started_ts=float(replay_started_ts),
            replay_last_done_ts=float(replay_last_done_ts) if replay_last_done_ts is not None else None,
            elapsed_sec=max(0.0, time.perf_counter() - eval_started),
        )

    metrics_by_signature = _run_policy_replay_tasks(
        list(unique_jobs_by_signature.values()),
        progress_emit=_emit_ensemble_replay_progress,
    )
    replay_done = int(len(metrics_by_signature))

    policy_raw_metrics: dict[str, dict] = {}
    for policy_name, job in policy_jobs_by_name.items():
        policy_raw_metrics[policy_name] = dict(metrics_by_signature.get(str(job.get("signature"))) or {})

    valid_policy_metrics = [metrics for metrics in policy_raw_metrics.values() if metrics]
    best_metrics = max(valid_policy_metrics, key=lambda item: float(item.get("score", 0.0)), default={})
    best_score = float(best_metrics.get("score", 0.0))
    benchmark_source = next((metrics for metrics in valid_policy_metrics if metrics.get("benchmark_oos_score") is not None), {})
    benchmark_score = float(benchmark_source.get("benchmark_oos_score", 0.0))
    benchmark_return_pct = float(benchmark_source.get("benchmark_return_pct", 0.0))
    benchmark_mdd_pct = float(benchmark_source.get("benchmark_mdd_pct", 0.0))

    best_policy_name = next((name for name, metrics in policy_raw_metrics.items() if metrics is best_metrics), "")
    best_members = list((policy_jobs_by_name.get(best_policy_name) or {}).get("members") or [])

    policy_metrics: dict[str, dict] = {}
    for policy_name, metrics in policy_raw_metrics.items():
        if not metrics:
            continue
        policy_metrics[policy_name] = _policy_metrics_from_ensemble_metrics(
            metrics,
            best_score=best_score,
            benchmark_score=benchmark_score,
        )

    ensemble_eval_sec = max(0.0, time.perf_counter() - eval_started)
    fold_elapsed = sum(float(row.get("elapsed_sec", 0.0) or 0.0) for row in seed_rows) + ensemble_eval_sec
    row = {
        "fold": f"{fold_idx}/{fold_count}",
        "oos_year": int(oos_year),
        "selection_period": selection_period,
        "selection_start_year": int(pd.Timestamp(selection_start).year),
        "selection_end_year": int(pd.Timestamp(selection_end).year),
        "selection_start_date": selection_start,
        "selection_end_date": selection_end,
        "oos_start_date": oos_start_date,
        "oos_end_date": oos_end_date,
        "oos_period": str(task.get("oos_period") or f"{oos_start_date}~{oos_end_date}"),
        "best_finalist_oos_score": float(best_score),
        "best_finalist_trial": None,
        "best_finalist_return_pct": float(best_metrics.get("return_pct", 0.0)),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "best_finalist_trades": int(best_metrics.get("trade_count", 0) or 0),
        "best_finalist_initial_capital": float(best_metrics.get("initial_equity", 0.0) or 0.0),
        "best_finalist_equity_curve": [],
        "best_finalist_params": dict((best_members[0] or {}).get("params") or {}) if best_members else {},
        "best_finalist_params_ensemble": best_members,
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(benchmark_return_pct),
        "benchmark_mdd_pct": float(benchmark_mdd_pct),
        **{policy_name: policy_metrics.get(policy_name, {}) for policy_name in REPORT_POLICY_NAMES},
        **{policy_name: policy_metrics.get(policy_name, {}) for policy_name in BASE_RETENTION_COMPARISON_POLICY_NAMES},
        "policy_schedules": policy_schedules,
        "elapsed_sec": float(fold_elapsed),
        "optimizer_search_sec": sum(float(row.get("optimizer_search_sec", 0.0) or 0.0) for row in seed_rows),
        "local_min_review_sec": sum(float(row.get("local_min_review_sec", 0.0) or 0.0) for row in seed_rows),
        "oos_diagnostics_sec": sum(float(row.get("oos_diagnostics_sec", 0.0) or 0.0) for row in seed_rows) + ensemble_eval_sec,
        "install_shared_cache_sec": sum(float(row.get("install_shared_cache_sec", 0.0) or 0.0) for row in seed_rows),
        "study_create_sec": sum(float(row.get("study_create_sec", 0.0) or 0.0) for row in seed_rows),
        "random_seed_ensemble": _build_rolling_seed_ensemble_policy_payload(),
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        "optimizer_seeds": [_extract_seed_from_policy_schedules(row) for row in seed_rows],
    }

    _write_parallel_fold_progress_event(
        stage="FOLD_RESULT",
        fold_idx=fold_idx,
        fold_count=fold_count,
        oos_year=oos_year,
        selection_start=selection_start,
        selection_end=selection_end,
        status="row ready",
        result_row=_compact_row_for_live_result(row),
        elapsed_sec=float(row.get("elapsed_sec", 0.0) or 0.0),
    )

    timing_row = dict(timing_rows[0]) if timing_rows else {
        "fold": f"{fold_idx}/{fold_count}",
        "fold_idx": fold_idx,
        "fold_count": fold_count,
        "oos_year": oos_year,
        "selection_period": selection_period,
        "db_file": "",
    }
    timing_row.update({
        "status": "done_seed_ensemble",
        "requested_trials": sum(int(row.get("requested_trials", 0) or 0) for row in timing_rows) or int(task.get("config", {}).get("trials_per_fold", 0) or 0) * len(seed_rows),
        "completed_trials": sum(int(row.get("completed_trials", 0) or 0) for row in timing_rows),
        "finalists_count": sum(int(row.get("finalists_count", 0) or 0) for row in timing_rows),
        "install_shared_cache_sec": float(row["install_shared_cache_sec"]),
        "study_create_sec": float(row["study_create_sec"]),
        "optimize_sec": float(row["optimizer_search_sec"]),
        "local_min_review_sec": float(row["local_min_review_sec"]),
        "oos_diagnostics_sec": float(row["oos_diagnostics_sec"]),
        "fold_total_sec": float(row["elapsed_sec"]),
        "seed_ensemble_members": int(len(seed_rows)),
        "seed_ensemble_seeds": ",".join(str(seed) for seed in row.get("optimizer_seeds", []) if seed is not None),
    })
    return {
        "status": "done",
        "fold_idx": fold_idx,
        "oos_year": int(oos_year),
        "row": row,
        "timing_row": timing_row,
        "chain_max_positions": chain_max_positions,
        "chain_enable_rotation": chain_enable_rotation,
        "log_path": str(task.get("log_path") or ""),
    }


def _run_outer_rolling_oos_fold_ensemble_task(task: dict) -> dict:
    policy = _build_rolling_seed_ensemble_policy_payload()
    seeds = generate_random_seed_ensemble(int(policy.get("seed_count", 1) or 1))
    parallel_workers = _resolve_fold_seed_ensemble_parallel_workers(task, len(seeds))
    log_path = str(task.get("log_path") or "")
    fold_idx = int(task.get("fold_idx", 0) or 0)
    fold_count = int(task.get("fold_count", 0) or 0)
    oos_year = int(task.get("oos_year", 0) or 0)
    selection_start = str(task.get("selection_start_date") or "")
    selection_end = str(task.get("selection_end_date") or "")
    _write_parallel_fold_progress_event(
        stage="START",
        fold_idx=fold_idx,
        fold_count=fold_count,
        oos_year=oos_year,
        selection_start=selection_start,
        selection_end=selection_end,
        status=f"seed ensemble START N={len(seeds)} min_agree={policy.get('min_agree')}",
        elapsed_sec=0.0,
    )
    if int(parallel_workers) > 1:
        _write_parallel_fold_progress_event(
            stage="START",
            fold_idx=fold_idx,
            fold_count=fold_count,
            oos_year=oos_year,
            selection_start=selection_start,
            selection_end=selection_end,
            status=f"seed ensemble parallel workers={int(parallel_workers)}",
            elapsed_sec=0.0,
        )
    elif _should_disable_nested_parallelism(task):
        _write_parallel_fold_progress_event(
            stage="START",
            fold_idx=fold_idx,
            fold_count=fold_count,
            oos_year=oos_year,
            selection_start=selection_start,
            selection_end=selection_end,
            status="seed ensemble safe sequential workers=1",
            elapsed_sec=0.0,
        )

    started = time.perf_counter()

    def _build_member_task(member_index: int, seed: int) -> dict:
        member_task = dict(task)
        member_task["seed_ensemble_member"] = True
        member_task["optimizer_seed"] = int(seed)
        member_task["seed_ensemble_member_index"] = int(member_index)
        member_task["seed_ensemble_member_count"] = int(len(seeds))
        if log_path:
            root, ext = os.path.splitext(log_path)
            member_task["log_path"] = f"{root}_seed{int(member_index):02d}_{int(seed)}{ext or '.log'}"
        return member_task

    def _run_member(member_index: int, seed: int) -> dict:
        _write_parallel_fold_progress_event(
            stage="START",
            fold_idx=fold_idx,
            fold_count=fold_count,
            oos_year=oos_year,
            selection_start=selection_start,
            selection_end=selection_end,
            seed_ensemble_member_index=int(member_index),
            seed_ensemble_member_count=int(len(seeds)),
            seed=int(seed),
            status="START",
            elapsed_sec=max(0.0, time.perf_counter() - started),
        )
        return _run_outer_rolling_oos_fold_task(_build_member_task(member_index, int(seed)))

    seed_results: list[dict] = []
    parallel_backend = resolve_optimizer_random_seed_ensemble_parallel_backend_default()
    if int(parallel_workers) <= 1:
        for member_index, seed in enumerate(seeds, start=1):
            seed_results.append(_run_member(int(member_index), int(seed)))
    else:
        if parallel_backend == "process":
            executor_kwargs, _start_method = get_process_pool_executor_kwargs()
            with ProcessPoolExecutor(max_workers=int(parallel_workers), **executor_kwargs) as executor:
                future_map = {
                    executor.submit(_run_outer_rolling_oos_fold_task, _build_member_task(int(member_index), int(seed))): (int(member_index), int(seed))
                    for member_index, seed in enumerate(seeds, start=1)
                }
                for future in as_completed(future_map):
                    member_index, _seed = future_map[future]
                    result = future.result()
                    result["seed_ensemble_member_index"] = int(member_index)
                    seed_results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
                future_map = {
                    executor.submit(_run_member, int(member_index), int(seed)): (int(member_index), int(seed))
                    for member_index, seed in enumerate(seeds, start=1)
                }
                for future in as_completed(future_map):
                    member_index, _seed = future_map[future]
                    result = future.result()
                    result["seed_ensemble_member_index"] = int(member_index)
                    seed_results.append(result)
        seed_results.sort(key=lambda item: int(item.get("seed_ensemble_member_index", 0) or 0))

    result = _aggregate_seed_ensemble_fold_results(task=task, seed_results=seed_results)
    _write_parallel_fold_progress_event(
        stage="DONE",
        fold_idx=fold_idx,
        fold_count=fold_count,
        oos_year=oos_year,
        selection_start=selection_start,
        selection_end=selection_end,
        status=f"seed ensemble done N={len(seeds)}",
        elapsed_sec=max(0.0, time.perf_counter() - started),
    )
    return result


def _run_outer_rolling_oos_fold_task(task: dict) -> dict:
    """Run one rolling fold in an isolated process for timing-mode fold parallelism."""
    log_path = str((task or {}).get("log_path") or "")

    def _execute() -> dict:
        from services.optimizer.session_factory import (
            build_optimizer_session,
            build_optimizer_session_from_spec,
            configure_optuna_logging,
            ensure_study_effective_policy_compatible,
        )
        from services.optimizer.prep import load_all_raw_data
        from services.optimizer.runtime import create_optimizer_study
        from services.optimizer.session import close_study_storage

        configure_optuna_logging()
        base_policy = dict(task["base_policy"])
        config_payload = dict(task["config"])
        config = OuterRollingConfig(
            training_start_year=int(config_payload["training_start_year"]),
            first_oos_year=int(config_payload["first_oos_year"]),
            last_oos_year=int(config_payload["last_oos_year"]),
            trials_per_fold=int(config_payload["trials_per_fold"]),
            window_mode=str(config_payload.get("window_mode", "fixed")),
            train_window_years=int(config_payload.get("train_window_years", 5)),
            confirm=False,
            first_oos_date=str(config_payload.get("first_oos_date", "")),
            last_oos_date=str(config_payload.get("last_oos_date", "")),
            train_window_months=int(config_payload.get("train_window_months", OUTER_ROLLING_TRAIN_WINDOW_MONTHS)),
            oos_horizon_months=int(config_payload.get("oos_horizon_months", OUTER_ROLLING_OOS_HORIZON_MONTHS)),
            raw_universe_required_min_rows=coerce_raw_universe_required_min_rows(
                config_payload.get(RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD),
                allow_none=True,
            ),
        )
        output_dir = str(task["output_dir"])
        selected_data_dir = str(task["selected_data_dir"])
        session_ts = str(task["session_ts"])
        optimizer_seed = task.get("optimizer_seed")
        sampler_kind = str(task.get("sampler_kind", "random") or "random")
        fold_idx = int(task["fold_idx"])
        fold_count = int(task["fold_count"])
        oos_year = int(task["oos_year"])
        optimizer_required_min_rows = int(task["optimizer_required_min_rows"])
        fold_workers = int(task.get("fold_workers", 1) or 1)
        timing_mode = bool(task.get("timing_mode", False))
        if not str(os.environ.get("OPTIMIZER_PREP_CACHE_MAX_ITEMS", "")).strip():
            os.environ["OPTIMIZER_PREP_CACHE_MAX_ITEMS"] = str(_resolve_parallel_worker_prep_cache_max_items(os.environ))

        fold_start = time.perf_counter()
        seed_context = _seed_progress_context_from_task(task)
        selection_start = str(task.get("selection_start_date") or "")
        selection_end = str(task.get("selection_end_date") or "")
        selection_period = str(task.get("selection_period") or f"{selection_start}~{selection_end}")
        selection_period_display = _display_month_period(selection_period)
        oos_period = str(task.get("oos_period") or str(oos_year))
        oos_period_display = _display_month_period(oos_period)
        oos_start_date = str(task.get("oos_start_date") or f"{str(oos_year)[:4]}-01-01")
        oos_end_date = str(task.get("oos_end_date") or f"{str(oos_year)[:4]}-12-31")
        print(f"[{fold_idx}/{fold_count}] OOS {oos_period_display} | parallel fold START | selection={selection_period_display}", flush=True)
        _write_parallel_fold_progress_event(
            stage="START",
            fold_idx=fold_idx,
            fold_count=fold_count,
            oos_year=oos_year,
            selection_start=selection_start,
            selection_end=selection_end,
            **seed_context,
            status="START",
            elapsed_sec=0.0,
        )
        fold_policy = dict(base_policy)
        fold_policy["selection_start_year"] = int(pd.Timestamp(selection_start).year)
        fold_policy["train_start_year"] = int(pd.Timestamp(selection_start).year)
        fold_policy["search_train_end_year"] = int(pd.Timestamp(selection_end).year)
        fold_policy["oos_start_year"] = int(pd.Timestamp(oos_start_date).year)
        fold_policy["selection_start_date"] = selection_start
        fold_policy["train_start_date"] = selection_start
        fold_policy["search_train_end_date"] = selection_end
        fold_policy["oos_start_date"] = oos_start_date
        fold_policy["oos_end_date"] = oos_end_date
        fold_policy = build_optimizer_runtime_policy(fold_policy, "split")
        fold_policy[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = int(optimizer_required_min_rows)
        objective_mode = str(fold_policy.get("objective_mode", "split_train_romd"))
        session_spec = resolve_optimizer_session_spec_for_fold(
            dict(task.get("optimizer_session_spec") or {}),
            oos_start_date=oos_start_date,
        )
        session = (
            build_optimizer_session_from_spec(
                walk_forward_policy=fold_policy,
                spec=session_spec,
            )
            if session_spec
            else build_optimizer_session(walk_forward_policy=fold_policy)
        )
        _validate_optimizer_runtime_context(session, session_spec)
        session.rolling_fold_workers_max = int(fold_workers)
        session.rolling_fold_parallel = True
        reset_prep_cache_stats = getattr(session, "reset_prep_cache_stats", None)
        if callable(reset_prep_cache_stats):
            reset_prep_cache_stats()
        chain_max_positions = int(session.train_max_positions)
        chain_enable_rotation = bool(session.train_enable_rotation)
        session.n_trials = int(config.trials_per_fold)
        session.disable_milestone_dashboard = True
        session.timing_mode = bool(timing_mode)
        sqlite_storage_enabled = _is_outer_rolling_sqlite_storage_enabled(os.environ)
        db_file = ""
        db_name = None
        resumed_db = False
        if sqlite_storage_enabled:
            db_file, resumed_db = _resolve_outer_rolling_db_file(
                output_dir=output_dir,
                session_ts=session_ts,
                oos_year=int(oos_year),
                task=task,
                environ=os.environ,
            )
            db_name = f"sqlite:///{db_file}"
        study = None
        try:
            install_started = time.perf_counter()
            session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
            install_shared_cache_sec = max(0.0, time.perf_counter() - install_started)
            print(f"[{fold_idx}/{fold_count}] OOS {oos_period_display} | raw data loaded | elapsed={_fmt_duration(install_shared_cache_sec)}", flush=True)
            _write_parallel_fold_progress_event(
                stage="RAW_DATA",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                **seed_context,
                status="raw data loaded",
                elapsed_sec=max(0.0, time.perf_counter() - fold_start),
            )
            session.profile_recorder.init_output_files()
            session.profile_recorder.mark_run_started()
            study_started = time.perf_counter()
            study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=sampler_kind)
            ensure_study_effective_policy_compatible(study=study, walk_forward_policy=fold_policy)
            _ensure_study_runtime_identity_compatible(study, session)
            study_create_sec = max(0.0, time.perf_counter() - study_started)
            seed_context = _seed_progress_context_from_task(task)
            progress = _FoldLogSearchProgress(
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                total_trials=int(config.trials_per_fold),
                seed_context=seed_context,
            )
            existing_trials, remaining_trials = _remaining_optimizer_trials(
                study, int(config.trials_per_fold)
            )
            session.current_session_trial = int(existing_trials)
            progress.emit(int(existing_trials), force=True)
            if resumed_db and existing_trials:
                print(
                    f"[{fold_idx}/{fold_count}] OOS {oos_period_display} | "
                    f"resume study {existing_trials}/{int(config.trials_per_fold)} trials",
                    flush=True,
                )
            optimize_started = time.perf_counter()
            search_parallel_trials = _resolve_single_fold_search_parallel_trials(os.environ, sampler_kind=sampler_kind)
            if remaining_trials > 0:
                study.optimize(
                    session.objective,
                    n_trials=int(remaining_trials),
                    n_jobs=int(search_parallel_trials),
                    callbacks=[progress.callback(session)],
                )
            search_done_ts = time.time()
            optimize_sec = max(0.0, time.perf_counter() - optimize_started)
            trial_count = len(list(getattr(study, "trials", []) or []))
            session.current_session_trial = int(trial_count or config.trials_per_fold)
            progress.done(int(session.current_session_trial))
            best_base_score = None if progress.best_score == float("-inf") else float(progress.best_score)
            best_base_text = format_optimizer_score_for_display(best_base_score, decimals=3)
            print(f"[{fold_idx}/{fold_count}] OOS {oos_period_display} | optimizer search DONE | best_base_score={best_base_text} | elapsed={_fmt_duration(optimize_sec)}", flush=True)

            local_started = time.perf_counter()
            local_started_ts = time.time()

            def _parallel_local_min_progress_sink(event: dict) -> None:
                data = dict(event or {})
                _write_parallel_fold_progress_event(
                    stage="LOCAL_MIN_REVIEW",
                    fold_idx=fold_idx,
                    fold_count=fold_count,
                    oos_year=oos_year,
                    selection_start=selection_start,
                    selection_end=selection_end,
                    **seed_context,
                    finalist_idx=int(data.get("finalist_idx", 0) or 0),
                    finalist_total=int(data.get("finalist_total", 0) or 0),
                    trial_number=data.get("trial_number"),
                    neighbor_done=int(data.get("neighbor_done", 0) or 0),
                    neighbor_total=int(data.get("neighbor_total", 0) or 0),
                    current=data.get("current"),
                    best=data.get("best"),
                    completed=int(config.trials_per_fold),
                    total=int(config.trials_per_fold),
                    best_base_score=best_base_score,
                    status=str(data.get("status") or "RUN"),
                    elapsed_sec=max(0.0, time.perf_counter() - local_started),
                    search_started_ts=float(progress.stage_start_ts),
                    search_last_done_ts=search_done_ts,
                    local_min_started_ts=data.get("local_min_started_ts") or local_started_ts,
                    local_min_completed=int(data.get("local_min_completed", 0) or 0),
                    local_min_neighbor_completed=int(data.get("local_min_neighbor_completed", 0) or 0),
                    local_min_last_done_ts=data.get("local_min_last_done_ts"),
                    local_min_last_neighbor_done_ts=data.get("local_min_last_neighbor_done_ts") or data.get("local_min_last_done_ts"),
                )

            session.outer_rolling_parallel_progress_sink = _parallel_local_min_progress_sink
            session.outer_rolling_local_progress_context = {
                "fold_idx": int(fold_idx),
                "fold_count": int(fold_count),
                "oos_year": int(oos_year),
                "selection_start": str(selection_start),
                "selection_end": str(selection_end),
                "best_base_score": best_base_score,
                "completed_results": [],
                "overall_start": fold_start,
            }
            finalists = list_local_min_score_finalists(
                study,
                session=session,
                objective_mode=objective_mode,
                include_trial=None,
                show_progress=True,
                include_oos_diagnostics=False,
                single_finalist_fast_path=True,
                selection_pruning=True,
            )
            if hasattr(session, "outer_rolling_parallel_progress_sink"):
                delattr(session, "outer_rolling_parallel_progress_sink")
            if hasattr(session, "outer_rolling_local_progress_context"):
                delattr(session, "outer_rolling_local_progress_context")
            local_elapsed = time.perf_counter() - local_started
            best_local_min_score = max((float(item.get("local_min_score", INVALID_TRIAL_VALUE)) for item in list(finalists or [])), default=None)
            best_local_min_text = format_optimizer_score_for_display(best_local_min_score, decimals=3)
            local_stage_label = "local-min review DONE" if is_optimizer_local_min_review_enabled() else "local-min disabled (base equivalent)"
            print(f"[{fold_idx}/{fold_count}] OOS {oos_period_display} | {local_stage_label} | finalists={len(finalists)} | best_local_min={best_local_min_text} | elapsed={_fmt_duration(local_elapsed)}", flush=True)
            _write_parallel_fold_progress_event(
                stage="LOCAL_MIN_REVIEW",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                finalist_idx=len(finalists),
                finalist_total=len(finalists),
                neighbor_done=1,
                neighbor_total=1,
                current=best_local_min_score,
                best=best_local_min_score,
                completed=int(config.trials_per_fold),
                total=int(config.trials_per_fold),
                best_base_score=best_base_score,
                best_local_min_score=best_local_min_score,
                status="DONE",
                elapsed_sec=local_elapsed,
                search_started_ts=float(progress.stage_start_ts),
                search_last_done_ts=search_done_ts,
                local_min_started_ts=local_started_ts,
                local_min_completed=len(finalists),
                local_min_neighbor_completed=sum(
                    max(0, int((item.get("local_min_meta") or {}).get("evaluated_neighbors", 0) or 0))
                    for item in list(finalists or [])
                ),
                local_min_last_done_ts=time.time(),
                local_min_last_neighbor_done_ts=time.time(),
            )
            policy_items = _build_policy_items(finalists, objective_mode=objective_mode)
            if not any(item is not None for item in policy_items.values()):
                fold_elapsed = time.perf_counter() - fold_start
                timing_row = _build_outer_timing_row(
                    fold_idx=fold_idx,
                    fold_count=fold_count,
                    oos_year=int(oos_year),
                    selection_start=selection_start,
                    selection_end=selection_end,
                    status="skipped_no_finalist",
                    session=session,
                    db_file=db_file,
                    install_shared_cache_sec=install_shared_cache_sec,
                    study_create_sec=study_create_sec,
                    optimize_sec=optimize_sec,
                    local_min_review_sec=local_elapsed,
                    oos_diagnostics_sec=0.0,
                    fold_total_sec=fold_elapsed,
                    finalists_count=len(finalists),
                )
                return {
                    "status": "skipped_no_finalist",
                    "fold_idx": fold_idx,
                    "oos_year": int(oos_year),
                    "row": None,
                    "timing_row": timing_row,
                    "chain_max_positions": chain_max_positions,
                    "chain_enable_rotation": chain_enable_rotation,
                    "log_path": log_path,
                }
            local_rank_map = _build_local_rank_map(finalists)
            retention_rank_map = _build_retention_rank_map(finalists)
            diagnostics_started = time.perf_counter()
            _write_parallel_fold_progress_event(
                stage="OOS_DIAGNOSTICS",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                status="RUN",
                best_base_score=best_base_score,
                best_local_min_score=best_local_min_score,
                elapsed_sec=0.0,
            )
            with session.optimizer_runtime_context():
                diagnostics = _evaluate_finalist_oos_diagnostics(
                    session=session,
                    finalists=finalists,
                    policy_items=policy_items,
                    oos_year=int(oos_year),
                    oos_start_date=oos_start_date,
                    oos_end_date=oos_end_date,
                )
            oos_diagnostics_sec = max(0.0, time.perf_counter() - diagnostics_started)
            print(f"[{fold_idx}/{fold_count}] OOS {oos_period_display} | OOS diagnostics DONE | elapsed={_fmt_duration(oos_diagnostics_sec)}", flush=True)
            _write_parallel_fold_progress_event(
                stage="OOS_DIAGNOSTICS",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                **seed_context,
                status="DONE",
                best_base_score=best_base_score,
                best_local_min_score=best_local_min_score,
                elapsed_sec=oos_diagnostics_sec,
            )
            fold_elapsed = time.perf_counter() - fold_start
            selection_period = str(selection_period)
            policy_schedules = {
                name: _build_policy_schedule_entry(
                    item=item,
                    policy_name=name,
                    oos_year=int(oos_year),
                    selection_period=selection_period,
                    local_rank_map=local_rank_map,
                    retention_rank_map=retention_rank_map,
                    oos_start_date=oos_start_date,
                    oos_end_date=oos_end_date,
                    optimizer_seed=optimizer_seed,
                    fixed_strategy_param_overrides=getattr(
                        session, "fixed_strategy_param_overrides", None
                    ),
                    fixed_tp_percent=getattr(
                        session,
                        "optimizer_fixed_tp_percent",
                        OPTIMIZER_FIXED_TP_PERCENT,
                    ),
                )
                for name, item in policy_items.items()
                if item is not None
            }
            row = {
                "fold": f"{fold_idx}/{fold_count}",
                "oos_year": int(oos_year),
                "selection_period": selection_period,
                "selection_start_year": int(pd.Timestamp(selection_start).year),
                "selection_end_year": int(pd.Timestamp(selection_end).year),
                "selection_start_date": str(selection_start),
                "selection_end_date": str(selection_end),
                "oos_start_date": str(oos_start_date),
                "oos_end_date": str(oos_end_date),
                "oos_period": str(oos_period),
                "best_finalist_oos_score": float(diagnostics.get("best_finalist_oos_score", 0.0)),
                "best_finalist_trial": diagnostics.get("best_finalist_trial"),
                "best_finalist_return_pct": float(diagnostics.get("best_finalist_return_pct", 0.0)),
                "best_finalist_mdd_pct": float(diagnostics.get("best_finalist_mdd_pct", 0.0)),
                "best_finalist_trades": int(diagnostics.get("best_finalist_trades", 0) or 0),
                "best_finalist_initial_capital": float(diagnostics.get("best_finalist_initial_capital", 0.0)),
                "best_finalist_equity_curve": list(diagnostics.get("best_finalist_equity_curve") or []),
                "best_finalist_params": dict(diagnostics.get("best_finalist_params") or {}),
                "benchmark_oos_score": float(diagnostics.get("benchmark_oos_score", 0.0)),
                "benchmark_return_pct": float(diagnostics.get("benchmark_return_pct", 0.0)),
                "benchmark_mdd_pct": float(diagnostics.get("benchmark_mdd_pct", 0.0)),
                **{
                    policy_name: diagnostics.get("policies", {}).get(policy_name, {})
                    for policy_name in REPORT_POLICY_NAMES
                },
                **{
                    policy_name: diagnostics.get("policies", {}).get(policy_name, {})
                    for policy_name in BASE_RETENTION_COMPARISON_POLICY_NAMES
                },
                "policy_schedules": policy_schedules,
                "elapsed_sec": float(fold_elapsed),
                "optimizer_search_sec": float(optimize_sec),
                "local_min_review_sec": float(local_elapsed),
                "oos_diagnostics_sec": float(oos_diagnostics_sec),
                "install_shared_cache_sec": float(install_shared_cache_sec),
                "study_create_sec": float(study_create_sec),
            }
            timing_row = _build_outer_timing_row(
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=int(oos_year),
                selection_start=selection_start,
                selection_end=selection_end,
                status="done",
                session=session,
                db_file=db_file,
                install_shared_cache_sec=install_shared_cache_sec,
                study_create_sec=study_create_sec,
                optimize_sec=optimize_sec,
                local_min_review_sec=local_elapsed,
                oos_diagnostics_sec=oos_diagnostics_sec,
                fold_total_sec=fold_elapsed,
                finalists_count=len(finalists),
            )
            _write_parallel_fold_progress_event(
                stage="DONE",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                **seed_context,
                status="done",
                best_base_score=best_base_score,
                best_local_min_score=best_local_min_score,
                elapsed_sec=fold_elapsed,
            )
            return {
                "status": "done",
                "fold_idx": fold_idx,
                "oos_year": int(oos_year),
                "row": row,
                "timing_row": timing_row,
                "chain_max_positions": chain_max_positions,
                "chain_enable_rotation": chain_enable_rotation,
                "log_path": log_path,
            }
        finally:
            session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    def _execute_with_logged_exception() -> dict:
        try:
            if not bool((task or {}).get("seed_ensemble_member")) and _is_rolling_random_seed_ensemble_enabled():
                return _run_outer_rolling_oos_fold_ensemble_task(task)
            return _execute()
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(_build_parallel_fold_failure_message(task=task, exc=exc, label="fold task failed")) from exc

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_handle:
            progress_log = _ParallelFoldProgressLogFilter(log_handle)
            with redirect_stdout(progress_log), redirect_stderr(log_handle):
                return _execute_with_logged_exception()
    return _execute_with_logged_exception()



FOLD_FIXED_STRATEGY_OVERRIDES_KEY = (
    "fixed_strategy_param_overrides_by_effective_date"
)


def resolve_optimizer_session_spec_for_fold(
    optimizer_session_spec: dict | None,
    *,
    oos_start_date: str | None,
) -> dict:
    """Merge one fold's fixed strategy values into the process-safe session spec.

    The generic optimizer session keeps one global fixed override mapping.  Research
    workflows that freeze non-search parameters to an existing rolling policy need
    values that may differ by effective date.  This resolver applies the matching
    fold mapping before session creation, so the objective, local-min review, OOS
    diagnostics and exported active params all see the same effective contract.
    """

    payload = dict(optimizer_session_spec or {})
    raw_mapping = payload.pop(FOLD_FIXED_STRATEGY_OVERRIDES_KEY, None)
    base_overrides = dict(payload.get("fixed_strategy_param_overrides") or {})
    if raw_mapping is None:
        if base_overrides:
            payload["fixed_strategy_param_overrides"] = base_overrides
        return payload
    if not isinstance(raw_mapping, dict):
        raise ValueError(
            f"{FOLD_FIXED_STRATEGY_OVERRIDES_KEY} 必須是 effective-date mapping"
        )
    if oos_start_date is None:
        if base_overrides:
            payload["fixed_strategy_param_overrides"] = base_overrides
        return payload

    normalized_date = pd.Timestamp(str(oos_start_date)).normalize().strftime("%Y-%m-%d")
    fold_overrides = raw_mapping.get(normalized_date)
    if fold_overrides is None:
        raise ValueError(
            "optimizer 缺少 rolling fold 固定參數："
            f"effective_date={normalized_date}"
        )
    if not isinstance(fold_overrides, dict):
        raise ValueError(
            "optimizer rolling fold 固定參數必須是 mapping："
            f"effective_date={normalized_date}"
        )
    base_overrides.update(dict(fold_overrides))
    payload["fixed_strategy_param_overrides"] = base_overrides
    return payload


def run_outer_rolling_oos(
    *,
    argv,
    environ,
    project_root: str,
    output_dir: str,
    base_policy: dict,
    selected_data_dir: str,
    dataset_label: str,
    load_all_raw_data,
    optimizer_required_min_rows: int,
    build_optimizer_session,
    create_optimizer_study,
    ensure_study_effective_policy_compatible,
    configure_optuna_logging,
    optimizer_seed=None,
    optimizer_session_spec: dict | None = None,
    default_trials: int = OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    timing_mode: bool = False,
    paramset_models_dir: str | None = None,
    canonical_strategy_param_family: str | None = None,
    output_files_title: str = "💾 輸出檔案",
) -> int:
    from services.optimizer.session import close_study_storage
    from services.optimizer.session_factory import build_optimizer_session_from_spec

    session_spec = dict(optimizer_session_spec or {})

    def _build_session(walk_forward_policy, *, oos_start_date=None):
        resolved_session_spec = resolve_optimizer_session_spec_for_fold(
            session_spec,
            oos_start_date=oos_start_date,
        )
        if resolved_session_spec:
            return build_optimizer_session_from_spec(
                walk_forward_policy=walk_forward_policy,
                spec=resolved_session_spec,
            )
        return build_optimizer_session(walk_forward_policy=walk_forward_policy)

    session_ts = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(output_dir, "outer_rolling_oos"), exist_ok=True)

    latest_date = _resolve_latest_date_from_csv_data_dir(selected_data_dir)
    latest_year = None if latest_date is None else int(latest_date.year)

    config = _resolve_config(argv, environ, base_policy=base_policy, latest_year=latest_year, latest_date=latest_date, default_trials=default_trials, timing_mode=bool(timing_mode))
    config.raw_universe_required_min_rows = int(optimizer_required_min_rows)
    folds = _build_rolling_folds(config)
    fold_count = int(len(folds))
    _apply_outer_rolling_resource_env_defaults(environ, timing_mode=bool(timing_mode), fold_count=fold_count)
    _apply_outer_rolling_process_environ(environ)
    fold_workers = _resolve_rolling_fold_workers(environ, timing_mode=bool(timing_mode), fold_count=fold_count)
    fold_parallel_enabled = _is_rolling_fold_parallel_enabled(environ, timing_mode=bool(timing_mode), fold_count=fold_count)
    sampler_kind = "random" if bool(timing_mode) else "tpe"
    _print_plan(
        config,
        parallel_settings_line=_format_parallel_settings_line(environ, fold_workers=int(fold_workers), sampler_kind=sampler_kind)
        if (bool(timing_mode) and fold_parallel_enabled)
        else None,
    )
    if not _confirm_plan(config):
        print(f"{C_YELLOW}已取消 outer rolling OOS。{C_RESET}")
        return 0

    configure_optuna_logging()
    rows: list[dict] = []
    fold_timing_rows: list[dict] = []
    chain_max_positions: int | None = None
    chain_enable_rotation: bool | None = None
    rolling_shared_prep_cache = OrderedDict()
    rolling_shared_prep_cache_max_items = _resolve_rolling_shared_prep_cache_max_items(environ)
    rolling_shared_local_min_order_score_cache = {}
    rolling_shared_local_min_field_order_score_cache = {}
    rolling_shared_prep_executor_holder = {}
    overall_start = time.perf_counter()
    resource_sampler = _ResourceUsageSampler(interval_sec=_resolve_resource_sample_interval_sec(environ))
    resource_sampler.start()
    performance_alignment_rows = _build_training_performance_alignment_rows(
        environ,
        fold_count=fold_count,
        fold_workers=int(fold_workers),
        sampler_kind=sampler_kind,
    )
    seed_ensemble_performance_metrics: dict = {}
    shared_raw_context = None
    if fold_parallel_enabled:
        raw_data_load_sec = 0.0
    else:
        shared_load_start = time.perf_counter()
        shared_data_policy = build_optimizer_runtime_policy(dict(base_policy), "split")
        shared_data_policy[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = int(optimizer_required_min_rows)
        shared_data_session = _build_session(shared_data_policy)
        try:
            shared_data_session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
            shared_raw_context = {
                "raw_data_cache": dict(shared_data_session.raw_data_cache),
                "static_fast_cache": dict(shared_data_session.static_fast_cache),
                "master_dates": set(shared_data_session.master_dates),
                "sorted_master_dates": list(shared_data_session.sorted_master_dates),
            }
        finally:
            shared_data_session.close_trial_prep_executor()
        raw_data_load_sec = max(0.0, time.perf_counter() - shared_load_start)
        print(f"{C_CYAN}⏱️ Rolling 共用資料快取完成：raw_data_load_once={raw_data_load_sec:.3f}s | folds={fold_count}{C_RESET}")

    if fold_parallel_enabled:
        log_dir = os.path.join(output_dir, "outer_rolling_oos", "fold_logs", session_ts)
        os.makedirs(log_dir, exist_ok=True)
        config_payload = {
            "training_start_year": int(config.training_start_year),
            "first_oos_year": int(config.first_oos_year),
            "last_oos_year": int(config.last_oos_year),
            "trials_per_fold": int(config.trials_per_fold),
            "window_mode": str(config.window_mode),
            "train_window_years": int(config.train_window_years),
            "first_oos_date": str(config.first_oos_date),
            "last_oos_date": str(config.last_oos_date),
            "train_window_months": int(config.train_window_months),
            "oos_horizon_months": int(config.oos_horizon_months),
            RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD: int(config.raw_universe_required_min_rows),
        }
        tasks = []
        for fold_idx, fold in enumerate(folds, start=1):
            oos_year = int(fold["oos_year"])
            selection_start = str(fold["selection_start_date"])
            selection_end = str(fold["selection_end_date"])
            tasks.append({
                "project_root": str(project_root),
                "output_dir": str(output_dir),
                "base_policy": dict(base_policy),
                "selected_data_dir": str(selected_data_dir),
                "optimizer_required_min_rows": int(optimizer_required_min_rows),
                "session_ts": str(session_ts),
                "config": dict(config_payload),
                "fold_idx": int(fold_idx),
                "fold_count": fold_count,
                "oos_year": int(oos_year),
                "oos_start_date": str(fold["oos_start_date"]),
                "oos_end_date": str(fold["oos_end_date"]),
                "oos_period": str(fold["oos_period"]),
                "selection_start_date": str(fold["selection_start_date"]),
                "selection_end_date": str(fold["selection_end_date"]),
                "selection_period": str(fold["selection_period"]),
                "optimizer_seed": optimizer_seed,
                "sampler_kind": str(sampler_kind),
                "timing_mode": bool(timing_mode),
                "fold_workers": int(fold_workers),
                "optimizer_session_spec": dict(session_spec),
                "log_path": os.path.join(log_dir, f"fold_{int(fold_idx):02d}_oos_{int(oos_year)}.log"),
            })
            # Fold log path is kept internally for diagnostics, but not printed during normal progress.
        chain_state = {"chain_max_positions": None, "chain_enable_rotation": None}
        try:
            with ProcessPoolExecutor(max_workers=int(fold_workers)) as executor:
                chain_state = _run_parallel_fold_futures(
                    executor=executor,
                    tasks=tasks,
                    rows=rows,
                    fold_timing_rows=fold_timing_rows,
                    overall_start=overall_start,
                    raw_data_load_sec=raw_data_load_sec,
                )
        except Exception as exc:
            chain_state = _run_missing_parallel_fold_tasks_sequentially(
                tasks=tasks,
                rows=rows,
                fold_timing_rows=fold_timing_rows,
                chain_state=chain_state,
                cause=exc,
            )
        if chain_state.get("chain_max_positions") is not None:
            chain_max_positions = int(chain_state.get("chain_max_positions"))
        if chain_state.get("chain_enable_rotation") is not None:
            chain_enable_rotation = bool(chain_state.get("chain_enable_rotation"))
        seed_ensemble_performance_metrics = dict(chain_state.get("seed_ensemble_performance_metrics") or {})
        rows[:] = sorted((normalize_optimizer_seed_ensemble_fold_row(item) for item in rows), key=optimizer_seed_ensemble_row_sort_key)
        fold_timing_rows.sort(key=lambda item: int(item.get("fold_idx", 0) or 0))
        for idx, row in enumerate(rows, start=1):
            row["fold"] = f"{idx}/{fold_count}"

    folds_to_run = [] if fold_parallel_enabled else folds

    for fold_idx, fold in enumerate(folds_to_run, start=1):
        fold_start = time.perf_counter()
        oos_year = int(fold["oos_year"])
        oos_period = str(fold["oos_period"])
        selection_period = str(fold["selection_period"])
        oos_period_display = _display_month_period(oos_period)
        selection_period_display = _display_month_period(selection_period)
        selection_start = str(fold["selection_start_date"])
        selection_end = str(fold["selection_end_date"])
        oos_start_date = str(fold["oos_start_date"])
        oos_end_date = str(fold["oos_end_date"])
        print(f"[{fold_idx}/{fold_count}] selection={selection_period_display} | OOS {oos_period_display} | fold START", flush=True)
        if _is_rolling_random_seed_ensemble_enabled():
            log_dir = os.path.join(output_dir, "outer_rolling_oos", "fold_logs", session_ts)
            os.makedirs(log_dir, exist_ok=True)
            config_payload = {
                "training_start_year": int(config.training_start_year),
                "first_oos_year": int(config.first_oos_year),
                "last_oos_year": int(config.last_oos_year),
                "trials_per_fold": int(config.trials_per_fold),
                "window_mode": str(config.window_mode),
                "train_window_years": int(config.train_window_years),
                "first_oos_date": str(config.first_oos_date),
                "last_oos_date": str(config.last_oos_date),
                "train_window_months": int(config.train_window_months),
                "oos_horizon_months": int(config.oos_horizon_months),
                RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD: int(config.raw_universe_required_min_rows),
            }
            task = {
                "project_root": str(project_root),
                "output_dir": str(output_dir),
                "base_policy": dict(base_policy),
                "selected_data_dir": str(selected_data_dir),
                "optimizer_required_min_rows": int(optimizer_required_min_rows),
                "session_ts": str(session_ts),
                "config": dict(config_payload),
                "fold_idx": int(fold_idx),
                "fold_count": int(fold_count),
                "oos_year": int(oos_year),
                "oos_start_date": str(oos_start_date),
                "oos_end_date": str(oos_end_date),
                "oos_period": str(oos_period),
                "selection_start_date": str(selection_start),
                "selection_end_date": str(selection_end),
                "selection_period": str(selection_period),
                "optimizer_seed": optimizer_seed,
                "sampler_kind": str(sampler_kind),
                "timing_mode": bool(timing_mode),
                "fold_workers": int(fold_workers),
                "optimizer_session_spec": dict(session_spec),
                "log_path": os.path.join(log_dir, f"fold_{int(fold_idx):02d}_oos_{int(oos_year)}.log"),
            }
            result = _run_outer_rolling_oos_fold_task(task)
            chain_state = {"chain_max_positions": chain_max_positions, "chain_enable_rotation": chain_enable_rotation}
            _consume_parallel_fold_result(result=result, rows=rows, fold_timing_rows=fold_timing_rows, chain_state=chain_state)
            if chain_state.get("chain_max_positions") is not None:
                chain_max_positions = int(chain_state.get("chain_max_positions"))
            if chain_state.get("chain_enable_rotation") is not None:
                chain_enable_rotation = bool(chain_state.get("chain_enable_rotation"))
            row = (result or {}).get("row")
            if row is not None:
                local_policy = dict(row.get("local") or {})
                local_oos = float(local_policy.get("rank_1_oos", 0.0))
                local_romd = _policy_plain_romd_score(local_policy)
                print(
                    f"{C_GREEN}[{fold_idx}/{fold_count}] selection={selection_period_display} | OOS {oos_period_display} | DONE seed ensemble | "
                    f"local_score={format_optimizer_score_for_display(local_oos, decimals=OOS_SCORE_DECIMALS)} | "
                    f"local_RoMD={local_romd:.{OOS_SCORE_DECIMALS}f} | "
                    f"best_score={_format_system_score_compare_plain(row['best_finalist_oos_score'], local_oos)} | "
                    f"0050_RoMD={_format_compare_plain(row['benchmark_oos_score'], local_romd)} | elapsed={_fmt_duration(row.get('elapsed_sec'))}{C_RESET}"
                )
                _print_completed_results(rows)
            continue
        fold_policy = dict(base_policy)
        fold_policy["selection_start_year"] = int(fold["selection_start_year"])
        fold_policy["train_start_year"] = int(fold["selection_start_year"])
        fold_policy["search_train_end_year"] = int(fold["selection_end_year"])
        fold_policy["oos_start_year"] = int(pd.Timestamp(fold["oos_start_date"]).year)
        fold_policy["selection_start_date"] = str(fold["selection_start_date"])
        fold_policy["train_start_date"] = str(fold["selection_start_date"])
        fold_policy["search_train_end_date"] = str(fold["selection_end_date"])
        fold_policy["oos_start_date"] = str(fold["oos_start_date"])
        fold_policy["oos_end_date"] = str(fold["oos_end_date"])
        fold_policy = build_optimizer_runtime_policy(fold_policy, "split")
        fold_policy[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = int(optimizer_required_min_rows)
        objective_mode = str(fold_policy.get("objective_mode", "split_train_romd"))
        resolved_session_spec = resolve_optimizer_session_spec_for_fold(
            session_spec,
            oos_start_date=oos_start_date,
        )
        session = _build_session(
            fold_policy,
            oos_start_date=oos_start_date,
        )
        _validate_optimizer_runtime_context(session, resolved_session_spec)
        attach_shared_executor = getattr(session, "attach_shared_trial_prep_executor_holder", None)
        if callable(attach_shared_executor):
            attach_shared_executor(rolling_shared_prep_executor_holder)
        if rolling_shared_prep_cache_max_items > 0:
            attach_shared_cache = getattr(session, "attach_shared_prepared_trial_input_cache", None)
            if callable(attach_shared_cache):
                attach_shared_cache(rolling_shared_prep_cache, max_items=rolling_shared_prep_cache_max_items)
        attach_order_score_cache = getattr(session, "attach_shared_local_min_order_score_cache", None)
        if callable(attach_order_score_cache):
            attach_order_score_cache(rolling_shared_local_min_order_score_cache)
        attach_field_order_score_cache = getattr(session, "attach_shared_local_min_field_order_score_cache", None)
        if callable(attach_field_order_score_cache):
            attach_field_order_score_cache(rolling_shared_local_min_field_order_score_cache)
        reset_prep_cache_stats = getattr(session, "reset_prep_cache_stats", None)
        if callable(reset_prep_cache_stats):
            reset_prep_cache_stats()
        chain_max_positions = int(session.train_max_positions)
        chain_enable_rotation = bool(session.train_enable_rotation)
        session.n_trials = int(config.trials_per_fold)
        session.disable_milestone_dashboard = True
        session.timing_mode = bool(timing_mode)
        sqlite_storage_enabled = _is_outer_rolling_sqlite_storage_enabled(os.environ)
        db_file = ""
        db_name = None
        resumed_db = False
        if sqlite_storage_enabled:
            sequential_task = {
                "optimizer_seed": optimizer_seed,
                "seed_ensemble_member": False,
                "runtime_cache_identity": str(
                    dict(session_spec or {}).get("runtime_cache_identity") or ""
                ),
            }
            db_file, resumed_db = _resolve_outer_rolling_db_file(
                output_dir=output_dir,
                session_ts=session_ts,
                oos_year=int(oos_year),
                task=sequential_task,
                environ=environ,
            )
            db_name = f"sqlite:///{db_file}"
        study = None
        try:
            print(f"\n{C_CYAN}[{fold_idx}/{fold_count}] selection={selection_period_display} | OOS {oos_period_display}{C_RESET}")
            install_started = time.perf_counter()
            session.install_raw_data_cache(
                selected_data_dir,
                shared_raw_context["raw_data_cache"],
                static_fast_cache=shared_raw_context["static_fast_cache"],
                master_dates=shared_raw_context["master_dates"],
                sorted_master_dates=shared_raw_context["sorted_master_dates"],
                required_min_rows=optimizer_required_min_rows,
            )
            install_shared_cache_sec = max(0.0, time.perf_counter() - install_started)
            session.profile_recorder.init_output_files()
            session.profile_recorder.mark_run_started()
            study_started = time.perf_counter()
            study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=sampler_kind)
            ensure_study_effective_policy_compatible(study=study, walk_forward_policy=fold_policy)
            _ensure_study_runtime_identity_compatible(study, session)
            study_create_sec = max(0.0, time.perf_counter() - study_started)
            progress = _SearchProgress(
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                total_trials=config.trials_per_fold,
                completed_results=rows,
                overall_start=overall_start,
            )
            existing_trials, remaining_trials = _remaining_optimizer_trials(
                study, int(config.trials_per_fold)
            )
            session.current_session_trial = int(existing_trials)
            if resumed_db and existing_trials:
                print(
                    f"[{fold_idx}/{fold_count}] OOS {oos_period_display} | "
                    f"resume study {existing_trials}/{int(config.trials_per_fold)} trials",
                    flush=True,
                )
            optimize_started = time.perf_counter()
            search_parallel_trials = _resolve_single_fold_search_parallel_trials(os.environ, sampler_kind=sampler_kind)
            if remaining_trials > 0:
                study.optimize(
                    session.objective,
                    n_trials=int(remaining_trials),
                    n_jobs=int(search_parallel_trials),
                    callbacks=[progress.callback(session)],
                )
            optimize_sec = max(0.0, time.perf_counter() - optimize_started)
            progress.done(int(session.current_session_trial))
            best_base_score = None if progress.best_score == float("-inf") else float(progress.best_score)

            local_started = time.perf_counter()
            session.outer_rolling_local_progress_context = {
                "fold_idx": int(fold_idx),
                "fold_count": fold_count,
                "oos_year": int(oos_year),
                "selection_start": str(selection_start),
                "selection_end": str(selection_end),
                "best_base_score": best_base_score,
                "completed_results": rows,
                "overall_start": overall_start,
            }
            finalists = list_local_min_score_finalists(
                study,
                session=session,
                objective_mode=objective_mode,
                include_trial=None,
                show_progress=True,
                include_oos_diagnostics=False,
                single_finalist_fast_path=True,
                selection_pruning=True,
            )
            if hasattr(session, "outer_rolling_local_progress_context"):
                delattr(session, "outer_rolling_local_progress_context")
            local_elapsed = time.perf_counter() - local_started
            prep_cache_stats = session.get_prep_cache_stats() if hasattr(session, "get_prep_cache_stats") else {}
            print(
                f"[{fold_idx}/{fold_count}] selection={selection_period_display} | OOS {oos_period_display} | "
                f"{'LOCAL_MIN_REVIEW DONE' if is_optimizer_local_min_review_enabled() else 'LOCAL_MIN_DISABLED'} | "
                f"finalists={len(finalists)} | best_local={format_optimizer_score_for_display(float(finalists[0].get('local_min_score', 0.0)) if finalists else 0.0, decimals=3)} "
                f"#{int(finalists[0]['trial'].number) + 1 if finalists else 0} | "
                f"prep_cache_hit/miss/evict={int(prep_cache_stats.get('hits', 0))}/{int(prep_cache_stats.get('misses', 0))}/{int(prep_cache_stats.get('evictions', 0))} | "
                f"elapsed={_fmt_duration(local_elapsed)}"
            )
            policy_items = _build_policy_items(finalists, objective_mode=objective_mode)
            if not any(item is not None for item in policy_items.values()):
                fold_elapsed = time.perf_counter() - fold_start
                fold_timing_rows.append(_build_outer_timing_row(
                    fold_idx=fold_idx,
                    fold_count=fold_count,
                    oos_year=int(oos_year),
                    selection_start=selection_start,
                    selection_end=selection_end,
                    status="skipped_no_finalist",
                    session=session,
                    db_file=db_file,
                    install_shared_cache_sec=install_shared_cache_sec,
                    study_create_sec=study_create_sec,
                    optimize_sec=optimize_sec,
                    local_min_review_sec=local_elapsed,
                    oos_diagnostics_sec=0.0,
                    fold_total_sec=fold_elapsed,
                    finalists_count=len(finalists),
                ))
                print(f"{C_YELLOW}[{fold_idx}/{fold_count}] selection={selection_period_display} | OOS {oos_period_display} | 無可用 finalist，略過。{C_RESET}")
                continue
            local_rank_map = _build_local_rank_map(finalists)
            retention_rank_map = _build_retention_rank_map(finalists)
            diagnostics_started = time.perf_counter()
            with session.optimizer_runtime_context():
                diagnostics = _evaluate_finalist_oos_diagnostics(
                    session=session,
                    finalists=finalists,
                    policy_items=policy_items,
                    oos_year=int(oos_year),
                    oos_start_date=oos_start_date,
                    oos_end_date=oos_end_date,
                )
            oos_diagnostics_sec = max(0.0, time.perf_counter() - diagnostics_started)
            print(f"[{fold_idx}/{fold_count}] OOS {oos_period_display} | OOS diagnostics DONE | elapsed={_fmt_duration(oos_diagnostics_sec)}", flush=True)
            fold_elapsed = time.perf_counter() - fold_start
            selection_period = str(selection_period)
            policy_schedules = {
                name: _build_policy_schedule_entry(
                    item=item,
                    policy_name=name,
                    oos_year=int(oos_year),
                    selection_period=selection_period,
                    local_rank_map=local_rank_map,
                    retention_rank_map=retention_rank_map,
                    oos_start_date=oos_start_date,
                    oos_end_date=oos_end_date,
                    optimizer_seed=optimizer_seed,
                    fixed_strategy_param_overrides=getattr(
                        session, "fixed_strategy_param_overrides", None
                    ),
                    fixed_tp_percent=getattr(
                        session,
                        "optimizer_fixed_tp_percent",
                        OPTIMIZER_FIXED_TP_PERCENT,
                    ),
                )
                for name, item in policy_items.items()
                if item is not None
            }
            row = {
                "fold": f"{fold_idx}/{fold_count}",
                "oos_year": int(oos_year),
                "selection_period": selection_period,
                "selection_start_year": int(pd.Timestamp(selection_start).year),
                "selection_end_year": int(pd.Timestamp(selection_end).year),
                "selection_start_date": str(selection_start),
                "selection_end_date": str(selection_end),
                "oos_start_date": str(oos_start_date),
                "oos_end_date": str(oos_end_date),
                "oos_period": str(oos_period),
                "best_finalist_oos_score": float(diagnostics.get("best_finalist_oos_score", 0.0)),
                "best_finalist_trial": diagnostics.get("best_finalist_trial"),
                "best_finalist_return_pct": float(diagnostics.get("best_finalist_return_pct", 0.0)),
                "best_finalist_mdd_pct": float(diagnostics.get("best_finalist_mdd_pct", 0.0)),
                "best_finalist_trades": int(diagnostics.get("best_finalist_trades", 0) or 0),
                "best_finalist_initial_capital": float(diagnostics.get("best_finalist_initial_capital", 0.0)),
                "best_finalist_equity_curve": list(diagnostics.get("best_finalist_equity_curve") or []),
                "best_finalist_params": dict(diagnostics.get("best_finalist_params") or {}),
                "benchmark_oos_score": float(diagnostics.get("benchmark_oos_score", 0.0)),
                "benchmark_return_pct": float(diagnostics.get("benchmark_return_pct", 0.0)),
                "benchmark_mdd_pct": float(diagnostics.get("benchmark_mdd_pct", 0.0)),
                **{
                    policy_name: diagnostics.get("policies", {}).get(policy_name, {})
                    for policy_name in REPORT_POLICY_NAMES
                },
                **{
                    policy_name: diagnostics.get("policies", {}).get(policy_name, {})
                    for policy_name in BASE_RETENTION_COMPARISON_POLICY_NAMES
                },
                "policy_schedules": policy_schedules,
                "elapsed_sec": float(fold_elapsed),
                "optimizer_search_sec": float(optimize_sec),
                "local_min_review_sec": float(local_elapsed),
                "oos_diagnostics_sec": float(oos_diagnostics_sec),
                "install_shared_cache_sec": float(install_shared_cache_sec),
                "study_create_sec": float(study_create_sec),
            }
            rows.append(row)
            fold_timing_rows.append(_build_outer_timing_row(
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=int(oos_year),
                selection_start=selection_start,
                selection_end=selection_end,
                status="done",
                session=session,
                db_file=db_file,
                install_shared_cache_sec=install_shared_cache_sec,
                study_create_sec=study_create_sec,
                optimize_sec=optimize_sec,
                local_min_review_sec=local_elapsed,
                oos_diagnostics_sec=oos_diagnostics_sec,
                fold_total_sec=fold_elapsed,
                finalists_count=len(finalists),
            ))
            local_policy = dict(row.get("local") or {})
            local_oos = float(local_policy.get("rank_1_oos", 0.0))
            local_romd = _policy_plain_romd_score(local_policy)
            print(
                f"{C_GREEN}[{fold_idx}/{fold_count}] selection={selection_period_display} | OOS {oos_period_display} | DONE | "
                f"local_score={format_optimizer_score_for_display(local_oos, decimals=OOS_SCORE_DECIMALS)} | "
                f"local_RoMD={local_romd:.{OOS_SCORE_DECIMALS}f} | "
                f"best_score={_format_system_score_compare_plain(row['best_finalist_oos_score'], local_oos)} | "
                f"0050_RoMD={_format_compare_plain(row['benchmark_oos_score'], local_romd)} | elapsed={_fmt_duration(fold_elapsed)}{C_RESET}"
            )
            _print_completed_results(rows)
        finally:
            flush_field_order_cache = getattr(session, "flush_shared_local_min_field_order_score_cache", None)
            if callable(flush_field_order_cache):
                flush_field_order_cache()
            session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    _shutdown_rolling_shared_prep_executor_holder(rolling_shared_prep_executor_holder)

    resolved_chain_max_positions = int(chain_max_positions if chain_max_positions is not None else DEFAULT_PORTFOLIO_MAX_POSITIONS)
    resolved_chain_enable_rotation = bool(chain_enable_rotation if chain_enable_rotation is not None else False)
    active_replay_started = time.perf_counter()
    if rows:
        chain_policy = build_optimizer_runtime_policy(dict(base_policy), "split")
        chain_policy[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = int(optimizer_required_min_rows)
        chain_session = _build_session(chain_policy)
        try:
            _validate_optimizer_runtime_context(chain_session, session_spec)
            with chain_session.optimizer_runtime_context():
                active_replay_chained = _build_active_replay_chained_oos_summary(
                    rows=rows,
                    config=config,
                    selected_data_dir=selected_data_dir,
                    output_dir=output_dir,
                    max_positions=resolved_chain_max_positions,
                    enable_rotation=resolved_chain_enable_rotation,
                    overall_start=overall_start,
                )
        finally:
            chain_session.close_trial_prep_executor()
    else:
        active_replay_chained = {}
    active_replay_chain_sec = max(0.0, time.perf_counter() - active_replay_started)
    final_report_chain_elapsed_sec = max(0.0, time.perf_counter() - overall_start)
    active_replay_chained_for_report = _with_chain_elapsed_override(
        active_replay_chained,
        elapsed_sec=final_report_chain_elapsed_sec,
    )
    report_write_started = time.perf_counter()
    paths = {}
    if not bool(timing_mode):
        paths = _write_reports(
            project_root=project_root,
            output_dir=output_dir,
            session_ts=session_ts,
            rows=rows,
            config=config,
            chained_override=active_replay_chained_for_report,
            models_dir=(
                str(paramset_models_dir)
                if paramset_models_dir is not None
                else resolve_models_dir(project_root, environ=environ)
            ),
            canonical_strategy_param_family=canonical_strategy_param_family,
        )
    report_write_sec = max(0.0, time.perf_counter() - report_write_started)
    resource_sampler.stop()
    timing_paths = _write_outer_timing_summary(
        output_dir=output_dir,
        session_ts=session_ts,
        dataset_label=dataset_label,
        config=config,
        timing_mode=bool(timing_mode),
        optimizer_seed=optimizer_seed,
        raw_data_load_sec=raw_data_load_sec,
        active_replay_chain_sec=active_replay_chain_sec,
        report_write_sec=report_write_sec,
        overall_sec=max(0.0, time.perf_counter() - overall_start),
        fold_timing_rows=fold_timing_rows,
        resource_summary=resource_sampler.summary(),
        resource_samples=resource_sampler.samples,
        performance_alignment_rows=performance_alignment_rows,
    )
    final_report = _format_final_report(rows, _build_summary(rows, config=config, chained_override=active_replay_chained_for_report), color=True)
    if final_report.strip():
        print("\n" + final_report)
    timing_payload = dict((timing_paths or {}).get("payload") or {})
    timing_summary = dict(timing_payload.get("summary") or {})
    seed_policy = _effective_rolling_seed_ensemble_policy_payload()
    seed_ensemble_enabled = bool(seed_policy.get("enabled", False))
    effective_seed_count = int(seed_policy.get("seed_count", 1) or 1)
    effective_min_agree = int(seed_policy.get("min_agree", effective_seed_count) or effective_seed_count)
    if not seed_ensemble_performance_metrics:
        seed_ensemble_performance_metrics = {
            "completed_trials": int(timing_summary.get("completed_trials", 0) or 0),
            "search_wall_elapsed_sec": float(timing_summary.get("optimize_sum_sec", 0.0) or 0.0),
            "completed_local_min_trials": int(timing_summary.get("local_min_neighbors_evaluated", 0) or 0),
            "local_min_wall_elapsed_sec": float(timing_summary.get("local_min_review_sum_sec", 0.0) or 0.0),
            "completed_replays": 0,
            "replay_wall_elapsed_sec": None,
        }
    print(format_optimizer_final_performance_summary(
        folds=int(fold_count),
        seeds=int(effective_seed_count),
        min_agree=int(effective_min_agree),
        completed_folds=int(len(rows)),
        total_elapsed_sec=float(timing_summary.get("overall_sec", max(0.0, time.perf_counter() - overall_start)) or 0.0),
        completed_trials=int(seed_ensemble_performance_metrics.get("completed_trials", 0) or 0),
        search_wall_elapsed_sec=seed_ensemble_performance_metrics.get("search_wall_elapsed_sec"),
        completed_local_min_trials=int(seed_ensemble_performance_metrics.get("completed_local_min_trials", 0) or 0),
        local_min_wall_elapsed_sec=seed_ensemble_performance_metrics.get("local_min_wall_elapsed_sec"),
        completed_replays=int(seed_ensemble_performance_metrics.get("completed_replays", 0) or 0),
        replay_wall_elapsed_sec=seed_ensemble_performance_metrics.get("replay_wall_elapsed_sec"),
        resource_summary=timing_summary,
        seed_ensemble_enabled=bool(seed_ensemble_enabled),
        color=True,
    ))
    if bool(timing_mode):
        pass
    if not bool(timing_mode):
        visible_paramsets = [
            (get_optimizer_policy_output_label(str(policy_name)), str(paramset_path))
            for policy_name, paramset_path in dict(paths.get("paramsets") or {}).items()
        ]
        print_optimizer_output_files(
            visible_paramsets, title=str(output_files_title or "💾 輸出檔案"), project_root=project_root
        )
    return 0
