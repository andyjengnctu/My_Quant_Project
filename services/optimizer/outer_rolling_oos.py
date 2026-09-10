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
from services.optimizer.outer_rolling_results import (
    _avg_float_from_rows,
    _build_chained_oos_row,
    _build_chained_oos_summary,
    _build_oos_avg_row,
    _policy_cell_text,
    _print_completed_results,
    _render_optimizer_results_tables,
    _render_results_table,
    _resolve_chain_elapsed_sec,
    _rows_period_bounds,
    _table_separator,
    _with_chain_elapsed_override,
    render_optimizer_results_tables,
)
from services.optimizer.outer_rolling_artifacts import (
    _build_effective_seed_ensemble_policy_payload,
    _build_params_ensemble_members_for_schedule,
    _build_policy_paramset_payload,
    _build_random_seed_ensemble_policy_payload,
    _build_seed_ensemble_policy_payload,
    _build_summary,
    _format_final_report,
    _remove_stale_policy_paramset_files,
    _write_policy_paramset_files,
    _write_reports,
)
from services.optimizer.outer_rolling_live_boards import (
    OptimizerSeedEnsembleProgressBoard,
    _ParallelFoldLiveBoard,
)
from services.optimizer.outer_rolling_parallel_runner import (
    _cancel_parallel_fold_executor_after_failure,
    _consume_parallel_fold_future,
    _consume_parallel_fold_result,
    _print_parallel_fold_result,
)
from services.optimizer.outer_rolling_evaluation import (
    _evaluate_finalist_ensemble_oos_metrics,
    _evaluate_finalist_oos_diagnostics,
    _evaluate_period_oos,
    _extract_period_metrics,
    _get_or_prepare_oos_inputs,
    _prep_result_has_pit_index,
)
from services.optimizer.outer_rolling_active_replay_runtime import (
    _build_active_replay_chained_oos_summary,
    _load_active_replay_contexts_by_signature,
    _run_active_replay_metrics_from_schedule_records,
)
from services.optimizer.outer_rolling_active_replay import (
    _active_replay_payload_has_params,
    _build_active_param_replay_payload_from_rows,
    _build_active_replay_schedule_records,
    _empty_unavailable_chain_metrics,
    _extract_active_replay_metrics,
    _filter_market_dates_by_date_range,
    _iter_active_replay_context_records,
    _merge_active_replay_market_dates,
    _policy_has_complete_active_schedule,
    _row_oos_start_date,
)
from services.optimizer.outer_rolling_policy_replay import (
    _build_ensemble_policy_schedule_from_seed_rows,
    _build_members_from_seed_rows_for_policy,
    _build_policy_replay_context,
    _build_policy_replay_payload_from_members,
    _build_rolling_seed_ensemble_policy_payload,
    _build_single_period_ensemble_payload,
    _effective_rolling_seed_ensemble_policy_payload,
    _evaluate_active_param_ensemble_replay_payload_task,
    _extract_seed_from_policy_schedules,
    _is_safe_sequential_fold_task,
    _policy_metrics_from_ensemble_metrics,
    _policy_replay_executor_class,
    _resolve_fold_seed_ensemble_parallel_workers,
    _run_policy_replay_tasks,
    _should_disable_nested_parallelism,
    _stable_policy_replay_signature,
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
    FOLD_FIXED_STRATEGY_OVERRIDES_KEY,
    OuterRollingConfig,
    resolve_optimizer_session_spec_for_fold,
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
