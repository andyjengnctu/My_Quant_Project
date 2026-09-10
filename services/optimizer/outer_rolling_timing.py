"""Outer-rolling timing row construction and timing-summary reporting.

This service owns execution timing aggregation/report persistence only.  It does
not own fold construction, replay, selection, seed semantics, or scientific
identity.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from core.runtime_utils import get_taipei_now, resolve_environment_flag as _env_flag
from core.training_performance import (
    is_optimizer_active_replay_include_pit_stats_index_enabled,
    is_optimizer_active_replay_include_trade_logs_enabled,
    is_optimizer_active_replay_use_prepared_cache_enabled,
    is_optimizer_active_replay_write_prepared_cache_enabled,
    is_optimizer_profile_write_files_enabled,
    is_optimizer_single_fold_tpe_parallel_search_allowed_default,
)
from services.optimizer.outer_rolling_performance import (
    _is_outer_rolling_sqlite_storage_enabled,
    _resolve_parallel_worker_prep_cache_max_items,
    _resolve_single_fold_search_parallel_trials,
)

OuterRollingConfig = Any


def _profile_avg_float(profile_summary: dict, key: str) -> float:
    avg = dict((profile_summary or {}).get("avg") or {})
    try:
        return float(avg.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0

def _build_outer_timing_row(
    *,
    fold_idx: int,
    fold_count: int,
    oos_year: int,
    selection_start: int,
    selection_end: int,
    status: str,
    session,
    db_file: str,
    install_shared_cache_sec: float,
    study_create_sec: float,
    optimize_sec: float,
    local_min_review_sec: float,
    oos_diagnostics_sec: float,
    fold_total_sec: float,
    finalists_count: int = 0,
) -> dict:
    profile_summary = session.profile_recorder.build_summary_payload()
    completed_trials = int(getattr(session, "current_session_trial", 0) or 0)
    get_prep_cache_stats = getattr(session, "get_prep_cache_stats", None)
    prep_cache_stats = get_prep_cache_stats() if callable(get_prep_cache_stats) else {}
    get_local_min_stats = getattr(session, "get_local_min_review_stats", None)
    local_min_stats = get_local_min_stats() if callable(get_local_min_stats) else {}
    return {
        "fold": f"{int(fold_idx)}/{int(fold_count)}",
        "fold_idx": int(fold_idx),
        "fold_count": int(fold_count),
        "oos_year": int(oos_year),
        "selection_period": f"{selection_start}~{selection_end}",
        "status": str(status),
        "requested_trials": int(getattr(session, "n_trials", 0) or 0),
        "completed_trials": completed_trials,
        "finalists_count": int(finalists_count),
        "db_file": str(db_file or "memory"),
        "profile_csv_path": str(session.profile_recorder.csv_path if getattr(session.profile_recorder, "write_files", True) else ""),
        "profile_summary_path": str(session.profile_recorder.summary_path if getattr(session.profile_recorder, "write_files", True) else ""),
        "profile_write_files": bool(getattr(session.profile_recorder, "write_files", True)),
        "study_storage": "sqlite" if db_file else "memory",
        "parallel_fold_log_mode": "progress_only",
        "install_shared_cache_sec": float(install_shared_cache_sec),
        "study_create_sec": float(study_create_sec),
        "optimize_sec": float(optimize_sec),
        "local_min_review_sec": float(local_min_review_sec),
        "oos_diagnostics_sec": float(oos_diagnostics_sec),
        "fold_total_sec": float(fold_total_sec),
        "avg_trial_total_wall_sec": _profile_avg_float(profile_summary, "trial_total_wall_sec"),
        "avg_objective_wall_sec": _profile_avg_float(profile_summary, "objective_wall_sec"),
        "avg_prep_wall_sec": _profile_avg_float(profile_summary, "prep_wall_sec"),
        "avg_portfolio_wall_sec": _profile_avg_float(profile_summary, "portfolio_wall_sec"),
        "avg_score_calc_sec": _profile_avg_float(profile_summary, "score_calc_sec"),
        "avg_filter_rules_sec": _profile_avg_float(profile_summary, "filter_rules_sec"),
        "first_trial_completed_wall_sec": profile_summary.get("first_trial_completed_wall_sec"),
        "prep_cache_hits": int(prep_cache_stats.get("hits", 0) or 0),
        "prep_cache_misses": int(prep_cache_stats.get("misses", 0) or 0),
        "prep_cache_stores": int(prep_cache_stats.get("stores", 0) or 0),
        "prep_cache_evictions": int(prep_cache_stats.get("evictions", 0) or 0),
        "prep_cache_items": int(prep_cache_stats.get("items", 0) or 0),
        "prep_cache_max_items": int(prep_cache_stats.get("max_items", 0) or 0),
        "prep_cache_shared": bool(prep_cache_stats.get("shared", False)),
        "prep_executor_created": int(prep_cache_stats.get("executor_created", 0) or 0),
        "prep_executor_reused": int(prep_cache_stats.get("executor_reused", 0) or 0),
        "prep_executor_shared": bool(prep_cache_stats.get("executor_shared", False)),
        "prep_call_count": int(prep_cache_stats.get("prep_call_count", 0) or 0),
        "prep_wall_sum_sec": float(prep_cache_stats.get("prep_wall_sum_sec", 0.0) or 0.0),
        "prep_worker_total_sum_sec": float(prep_cache_stats.get("prep_worker_total_sum_sec", 0.0) or 0.0),
        "prep_total_sum_sec": float(prep_cache_stats.get("prep_total_sum_sec", 0.0) or 0.0),
        "prep_copy_sum_sec": float(prep_cache_stats.get("prep_copy_sum_sec", 0.0) or 0.0),
        "prep_generate_signals_sum_sec": float(prep_cache_stats.get("prep_generate_signals_sum_sec", 0.0) or 0.0),
        "prep_assign_sum_sec": float(prep_cache_stats.get("prep_assign_sum_sec", 0.0) or 0.0),
        "prep_run_backtest_sum_sec": float(prep_cache_stats.get("prep_run_backtest_sum_sec", 0.0) or 0.0),
        "prep_to_dict_sum_sec": float(prep_cache_stats.get("prep_to_dict_sum_sec", 0.0) or 0.0),
        "prep_static_pack_sum_sec": float(prep_cache_stats.get("prep_static_pack_sum_sec", 0.0) or 0.0),
        "prep_executor_setup_sum_sec": float(prep_cache_stats.get("prep_executor_setup_sum_sec", 0.0) or 0.0),
        "prep_executor_submit_sum_sec": float(prep_cache_stats.get("prep_executor_submit_sum_sec", 0.0) or 0.0),
        "prep_executor_collect_sum_sec": float(prep_cache_stats.get("prep_executor_collect_sum_sec", 0.0) or 0.0),
        "prep_executor_shutdown_sum_sec": float(prep_cache_stats.get("prep_executor_shutdown_sum_sec", 0.0) or 0.0),
        "prep_merge_sum_sec": float(prep_cache_stats.get("prep_merge_sum_sec", 0.0) or 0.0),
        "prep_master_union_sum_sec": float(prep_cache_stats.get("prep_master_union_sum_sec", 0.0) or 0.0),
        "prep_ticker_count_sum": int(prep_cache_stats.get("prep_ticker_count_sum", 0) or 0),
        "prep_batch_count_sum": int(prep_cache_stats.get("prep_batch_count_sum", 0) or 0),
        "prep_ok_count_sum": int(prep_cache_stats.get("prep_ok_count_sum", 0) or 0),
        "prep_fail_count_sum": int(prep_cache_stats.get("prep_fail_count_sum", 0) or 0),
        "prep_feature_bank_hits": int(prep_cache_stats.get("prep_feature_bank_hits", 0) or 0),
        "prep_feature_bank_misses": int(prep_cache_stats.get("prep_feature_bank_misses", 0) or 0),
        "prep_feature_bank_size_max": int(prep_cache_stats.get("prep_feature_bank_size_max", 0) or 0),
        "prep_feature_bank_max_items": int(prep_cache_stats.get("prep_feature_bank_max_items", 0) or 0),
        "local_min_neighbors_total": int(local_min_stats.get("total_neighbors", 0) or 0),
        "local_min_neighbors_evaluated": int(local_min_stats.get("evaluated_neighbors", 0) or 0),
        "local_min_neighbors_skipped": int(local_min_stats.get("skipped_neighbors", 0) or 0),
        "local_min_payload_score_cache_hits": int(local_min_stats.get("payload_score_cache_hits", 0) or 0),
        "local_min_prep_cache_prioritized": int(local_min_stats.get("prep_cache_prioritized", 0) or 0),
        "local_min_order_score_prioritized": int(local_min_stats.get("order_score_prioritized", 0) or 0),
        "local_min_field_order_score_prioritized": int(local_min_stats.get("field_order_score_prioritized", 0) or 0),
        "local_min_portfolio_dependency_deprioritized": int(local_min_stats.get("portfolio_dependency_deprioritized", 0) or 0),
        "local_min_signal_dependency_field_prioritized": int(local_min_stats.get("signal_dependency_field_prioritized", 0) or 0),
        "local_min_early_stops": int(local_min_stats.get("early_stops", 0) or 0),
        "local_min_selection_prunes": int(local_min_stats.get("selection_prunes", 0) or 0),
        "local_min_parallel_workers_max": int(local_min_stats.get("parallel_workers_max", 0) or 0),
        "local_min_parallel_submitted": int(local_min_stats.get("parallel_submitted", 0) or 0),
        "local_min_parallel_completed": int(local_min_stats.get("parallel_completed", 0) or 0),
        "local_min_parallel_cancelled": int(local_min_stats.get("parallel_cancelled", 0) or 0),
        "local_min_hard_fail_stops": int(local_min_stats.get("hard_fail_stops", 0) or 0),
        "local_min_hard_fail_neighbors_skipped": int(local_min_stats.get("hard_fail_neighbors_skipped", 0) or 0),
        "local_min_hard_fail_cancelled": int(local_min_stats.get("hard_fail_cancelled", 0) or 0),
        "local_min_dependency_signal_total": int(local_min_stats.get("dependency_signal_total", 0) or 0),
        "local_min_dependency_signal_evaluated": int(local_min_stats.get("dependency_signal_evaluated", 0) or 0),
        "local_min_dependency_portfolio_total": int(local_min_stats.get("dependency_portfolio_total", 0) or 0),
        "local_min_dependency_portfolio_evaluated": int(local_min_stats.get("dependency_portfolio_evaluated", 0) or 0),
        "local_min_dependency_mixed_total": int(local_min_stats.get("dependency_mixed_total", 0) or 0),
        "local_min_dependency_mixed_evaluated": int(local_min_stats.get("dependency_mixed_evaluated", 0) or 0),
        "local_min_dependency_unknown_total": int(local_min_stats.get("dependency_unknown_total", 0) or 0),
        "local_min_dependency_unknown_evaluated": int(local_min_stats.get("dependency_unknown_evaluated", 0) or 0),
        "local_min_signal_reuse_candidate_total": int(local_min_stats.get("signal_reuse_candidate_total", 0) or 0),
        "local_min_signal_reuse_candidate_evaluated": int(local_min_stats.get("signal_reuse_candidate_evaluated", 0) or 0),
        "local_min_signal_recompute_required_total": int(local_min_stats.get("signal_recompute_required_total", 0) or 0),
        "local_min_signal_recompute_required_evaluated": int(local_min_stats.get("signal_recompute_required_evaluated", 0) or 0),
        "local_min_dependency_field_total_counts": str(local_min_stats.get("dependency_field_total_counts", "") or ""),
        "local_min_dependency_field_evaluated_counts": str(local_min_stats.get("dependency_field_evaluated_counts", "") or ""),
        "rolling_fold_workers_max": int(getattr(session, "rolling_fold_workers_max", 1) or 1),
        "rolling_fold_parallel": bool(getattr(session, "rolling_fold_parallel", False)),
    }

def _sum_timing_rows(rows: list[dict], key: str) -> float:
    total = 0.0
    for row in list(rows or []):
        try:
            total += float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total

def _sum_int_timing_rows(rows: list[dict], key: str) -> int:
    total = 0
    for row in list(rows or []):
        try:
            total += int(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return int(total)

def _parse_compact_count_text(value) -> dict[str, int]:
    counts: dict[str, int] = {}
    text = str(value or "").strip()
    if not text:
        return counts
    for part in text.split(";"):
        if not part or ":" not in part:
            continue
        name, raw_count = part.rsplit(":", 1)
        field_name = str(name or "unknown")
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        counts[field_name] = int(counts.get(field_name, 0) or 0) + int(count)
    return counts

def _merge_compact_timing_count_text(rows: list[dict], key: str, *, max_items: int = 24) -> str:
    merged: dict[str, int] = {}
    for row in list(rows or []):
        row_counts = _parse_compact_count_text(row.get(key, ""))
        for field_name, count in row_counts.items():
            merged[field_name] = int(merged.get(field_name, 0) or 0) + int(count)
    items = [(field_name, count) for field_name, count in merged.items() if int(count or 0) > 0]
    items.sort(key=lambda item: (-item[1], item[0]))
    if max_items is not None:
        items = items[:max(1, int(max_items))]
    return ";".join(f"{field_name}:{count}" for field_name, count in items)

def _write_outer_timing_summary(
    *,
    output_dir: str,
    session_ts: str,
    dataset_label: str,
    config: OuterRollingConfig,
    timing_mode: bool,
    optimizer_seed,
    raw_data_load_sec: float,
    active_replay_chain_sec: float,
    report_write_sec: float,
    overall_sec: float,
    fold_timing_rows: list[dict],
    resource_summary: dict | None = None,
    resource_samples: list[dict] | None = None,
    performance_alignment_rows: list[dict] | None = None,
    write_files: bool = False,
) -> dict:
    report_dir = os.path.join(output_dir, "outer_rolling_oos")
    base = os.path.join(report_dir, f"outer_rolling_oos_timing_{session_ts}")
    csv_path = base + ".csv" if bool(write_files) else ""
    json_path = base + ".json" if bool(write_files) else ""
    resource_write_csv = bool(write_files) and _env_flag(os.environ, "OPTIMIZER_RESOURCE_WRITE_CSV", False)
    resource_csv_path = os.path.join(report_dir, f"outer_rolling_oos_resource_{session_ts}.csv") if resource_write_csv else ""
    if bool(write_files):
        os.makedirs(report_dir, exist_ok=True)
    sampler_kind = "random" if bool(timing_mode) else "tpe"
    single_fold_search_parallel_trials = _resolve_single_fold_search_parallel_trials(os.environ, sampler_kind=sampler_kind)
    single_fold_allow_tpe_parallel_search = _env_flag(
        os.environ,
        "OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH",
        is_optimizer_single_fold_tpe_parallel_search_allowed_default(),
    )

    if bool(write_files) and fold_timing_rows:
        fieldnames = list(fold_timing_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fold_timing_rows)

    samples_for_csv = list(resource_samples or [])
    if bool(write_files) and resource_write_csv and samples_for_csv:
        sample_fieldnames = list(samples_for_csv[0].keys())
        with open(resource_csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sample_fieldnames)
            writer.writeheader()
            writer.writerows(samples_for_csv)
    else:
        resource_csv_path = ""

    optimize_sec = _sum_timing_rows(fold_timing_rows, "optimize_sec")
    local_min_review_sec = _sum_timing_rows(fold_timing_rows, "local_min_review_sec")
    oos_diagnostics_sec = _sum_timing_rows(fold_timing_rows, "oos_diagnostics_sec")
    install_shared_cache_sec = _sum_timing_rows(fold_timing_rows, "install_shared_cache_sec")
    study_create_sec = _sum_timing_rows(fold_timing_rows, "study_create_sec")
    fold_total_sec = _sum_timing_rows(fold_timing_rows, "fold_total_sec")
    completed_trials = sum(int(row.get("completed_trials", 0) or 0) for row in list(fold_timing_rows or []))
    prep_cache_hits = sum(int(row.get("prep_cache_hits", 0) or 0) for row in list(fold_timing_rows or []))
    prep_cache_misses = sum(int(row.get("prep_cache_misses", 0) or 0) for row in list(fold_timing_rows or []))
    prep_cache_stores = sum(int(row.get("prep_cache_stores", 0) or 0) for row in list(fold_timing_rows or []))
    prep_cache_evictions = sum(int(row.get("prep_cache_evictions", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_neighbors_total = sum(int(row.get("local_min_neighbors_total", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_neighbors_evaluated = sum(int(row.get("local_min_neighbors_evaluated", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_payload_score_cache_hits = sum(int(row.get("local_min_payload_score_cache_hits", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_prep_cache_prioritized = sum(int(row.get("local_min_prep_cache_prioritized", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_order_score_prioritized = sum(int(row.get("local_min_order_score_prioritized", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_field_order_score_prioritized = sum(int(row.get("local_min_field_order_score_prioritized", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_portfolio_dependency_deprioritized = sum(int(row.get("local_min_portfolio_dependency_deprioritized", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_signal_dependency_field_prioritized = sum(int(row.get("local_min_signal_dependency_field_prioritized", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_early_stops = sum(int(row.get("local_min_early_stops", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_selection_prunes = sum(int(row.get("local_min_selection_prunes", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_parallel_workers_max = max((int(row.get("local_min_parallel_workers_max", 0) or 0) for row in list(fold_timing_rows or [])), default=0)
    local_min_parallel_submitted = sum(int(row.get("local_min_parallel_submitted", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_parallel_completed = sum(int(row.get("local_min_parallel_completed", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_parallel_cancelled = sum(int(row.get("local_min_parallel_cancelled", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_hard_fail_stops = sum(int(row.get("local_min_hard_fail_stops", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_hard_fail_neighbors_skipped = sum(int(row.get("local_min_hard_fail_neighbors_skipped", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_hard_fail_cancelled = sum(int(row.get("local_min_hard_fail_cancelled", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_dependency_signal_total = _sum_int_timing_rows(fold_timing_rows, "local_min_dependency_signal_total")
    local_min_dependency_signal_evaluated = _sum_int_timing_rows(fold_timing_rows, "local_min_dependency_signal_evaluated")
    local_min_dependency_portfolio_total = _sum_int_timing_rows(fold_timing_rows, "local_min_dependency_portfolio_total")
    local_min_dependency_portfolio_evaluated = _sum_int_timing_rows(fold_timing_rows, "local_min_dependency_portfolio_evaluated")
    local_min_dependency_mixed_total = _sum_int_timing_rows(fold_timing_rows, "local_min_dependency_mixed_total")
    local_min_dependency_mixed_evaluated = _sum_int_timing_rows(fold_timing_rows, "local_min_dependency_mixed_evaluated")
    local_min_dependency_unknown_total = _sum_int_timing_rows(fold_timing_rows, "local_min_dependency_unknown_total")
    local_min_dependency_unknown_evaluated = _sum_int_timing_rows(fold_timing_rows, "local_min_dependency_unknown_evaluated")
    local_min_signal_reuse_candidate_total = _sum_int_timing_rows(fold_timing_rows, "local_min_signal_reuse_candidate_total")
    local_min_signal_reuse_candidate_evaluated = _sum_int_timing_rows(fold_timing_rows, "local_min_signal_reuse_candidate_evaluated")
    local_min_signal_recompute_required_total = _sum_int_timing_rows(fold_timing_rows, "local_min_signal_recompute_required_total")
    local_min_signal_recompute_required_evaluated = _sum_int_timing_rows(fold_timing_rows, "local_min_signal_recompute_required_evaluated")
    local_min_dependency_field_total_counts = _merge_compact_timing_count_text(fold_timing_rows, "local_min_dependency_field_total_counts")
    local_min_dependency_field_evaluated_counts = _merge_compact_timing_count_text(fold_timing_rows, "local_min_dependency_field_evaluated_counts")
    rolling_fold_workers_max = max((int(row.get("rolling_fold_workers_max", 1) or 1) for row in list(fold_timing_rows or [])), default=1)
    rolling_fold_parallel = any(bool(row.get("rolling_fold_parallel", False)) for row in list(fold_timing_rows or []))
    prep_executor_created = sum(int(row.get("prep_executor_created", 0) or 0) for row in list(fold_timing_rows or []))
    prep_executor_reused = sum(int(row.get("prep_executor_reused", 0) or 0) for row in list(fold_timing_rows or []))
    prep_call_count = sum(int(row.get("prep_call_count", 0) or 0) for row in list(fold_timing_rows or []))
    prep_ticker_count_sum = sum(int(row.get("prep_ticker_count_sum", 0) or 0) for row in list(fold_timing_rows or []))
    prep_batch_count_sum = sum(int(row.get("prep_batch_count_sum", 0) or 0) for row in list(fold_timing_rows or []))
    prep_ok_count_sum = sum(int(row.get("prep_ok_count_sum", 0) or 0) for row in list(fold_timing_rows or []))
    prep_fail_count_sum = sum(int(row.get("prep_fail_count_sum", 0) or 0) for row in list(fold_timing_rows or []))
    prep_feature_bank_hits = sum(int(row.get("prep_feature_bank_hits", 0) or 0) for row in list(fold_timing_rows or []))
    prep_feature_bank_misses = sum(int(row.get("prep_feature_bank_misses", 0) or 0) for row in list(fold_timing_rows or []))
    prep_feature_bank_size_max = max((int(row.get("prep_feature_bank_size_max", 0) or 0) for row in list(fold_timing_rows or [])), default=0)
    prep_feature_bank_max_items = max((int(row.get("prep_feature_bank_max_items", 0) or 0) for row in list(fold_timing_rows or [])), default=0)
    payload = {
        "type": "outer_rolling_oos_timing",
        "version": 1,
        "created_at": get_taipei_now().isoformat(),
        "timing_mode": bool(timing_mode),
        "dataset_label": str(dataset_label),
        "optimizer_seed": optimizer_seed,
        "sampler_kind": sampler_kind,
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
            "trials_per_fold": int(config.trials_per_fold),
            "fold_count": len(fold_timing_rows),
            "completed_trials": int(completed_trials),
        },
        "training_performance_alignment": list(performance_alignment_rows or []),
        "summary": {
            "overall_sec": float(overall_sec),
            "raw_data_load_once_sec": float(raw_data_load_sec),
            "install_shared_cache_sum_sec": float(install_shared_cache_sec),
            "study_create_sum_sec": float(study_create_sec),
            "optimize_sum_sec": float(optimize_sec),
            "local_min_review_sum_sec": float(local_min_review_sec),
            "oos_diagnostics_sum_sec": float(oos_diagnostics_sec),
            "active_replay_chain_sec": float(active_replay_chain_sec),
            "report_write_sec": float(report_write_sec),
            "fold_total_sum_sec": float(fold_total_sec),
            "avg_optimize_sec_per_completed_trial": (float(optimize_sec) / float(completed_trials)) if completed_trials > 0 else 0.0,
            "single_fold_search_parallel_trials": int(single_fold_search_parallel_trials),
            "single_fold_allow_tpe_parallel_search": bool(single_fold_allow_tpe_parallel_search),
            "avg_fold_total_sec": (float(fold_total_sec) / float(len(fold_timing_rows))) if fold_timing_rows else 0.0,
            "fold_wall_sec": (max((float(row.get("fold_total_sec", 0.0) or 0.0) for row in list(fold_timing_rows or [])), default=0.0) if bool(rolling_fold_parallel) else float(fold_total_sec)),
            "other_overhead_sec": max(0.0, float(overall_sec) - (max((float(row.get("fold_total_sec", 0.0) or 0.0) for row in list(fold_timing_rows or [])), default=0.0) if bool(rolling_fold_parallel) else float(fold_total_sec)) - float(active_replay_chain_sec) - float(raw_data_load_sec) - float(report_write_sec)),
            "prep_cache_hits": int(prep_cache_hits),
            "prep_cache_misses": int(prep_cache_misses),
            "prep_cache_stores": int(prep_cache_stores),
            "prep_cache_evictions": int(prep_cache_evictions),
            "prep_cache_hit_rate": (float(prep_cache_hits) / float(prep_cache_hits + prep_cache_misses)) if (prep_cache_hits + prep_cache_misses) > 0 else 0.0,
            "prep_call_count": int(prep_call_count),
            "prep_wall_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_wall_sum_sec")),
            "prep_worker_total_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_worker_total_sum_sec")),
            "prep_total_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_total_sum_sec")),
            "prep_copy_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_copy_sum_sec")),
            "prep_generate_signals_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_generate_signals_sum_sec")),
            "prep_assign_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_assign_sum_sec")),
            "prep_run_backtest_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_run_backtest_sum_sec")),
            "prep_to_dict_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_to_dict_sum_sec")),
            "prep_static_pack_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_static_pack_sum_sec")),
            "prep_executor_setup_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_executor_setup_sum_sec")),
            "prep_executor_submit_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_executor_submit_sum_sec")),
            "prep_executor_collect_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_executor_collect_sum_sec")),
            "prep_executor_shutdown_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_executor_shutdown_sum_sec")),
            "prep_merge_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_merge_sum_sec")),
            "prep_master_union_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_master_union_sum_sec")),
            "prep_ticker_count_sum": int(prep_ticker_count_sum),
            "prep_batch_count_sum": int(prep_batch_count_sum),
            "prep_ok_count_sum": int(prep_ok_count_sum),
            "prep_fail_count_sum": int(prep_fail_count_sum),
            "prep_feature_bank_hits": int(prep_feature_bank_hits),
            "prep_feature_bank_misses": int(prep_feature_bank_misses),
            "prep_feature_bank_hit_rate": (float(prep_feature_bank_hits) / float(prep_feature_bank_hits + prep_feature_bank_misses)) if (prep_feature_bank_hits + prep_feature_bank_misses) > 0 else 0.0,
            "prep_feature_bank_size_max": int(prep_feature_bank_size_max),
            "prep_feature_bank_max_items": int(prep_feature_bank_max_items),
            "local_min_neighbors_total": int(local_min_neighbors_total),
            "local_min_neighbors_evaluated": int(local_min_neighbors_evaluated),
            "local_min_neighbors_skipped": max(0, int(local_min_neighbors_total) - int(local_min_neighbors_evaluated)),
            "local_min_payload_score_cache_hits": int(local_min_payload_score_cache_hits),
            "local_min_prep_cache_prioritized": int(local_min_prep_cache_prioritized),
            "local_min_order_score_prioritized": int(local_min_order_score_prioritized),
            "local_min_field_order_score_prioritized": int(local_min_field_order_score_prioritized),
            "local_min_portfolio_dependency_deprioritized": int(local_min_portfolio_dependency_deprioritized),
            "local_min_signal_dependency_field_prioritized": int(local_min_signal_dependency_field_prioritized),
            "local_min_early_stops": int(local_min_early_stops),
            "local_min_selection_prunes": int(local_min_selection_prunes),
            "local_min_parallel_workers_max": int(local_min_parallel_workers_max),
            "local_min_parallel_submitted": int(local_min_parallel_submitted),
            "local_min_parallel_completed": int(local_min_parallel_completed),
            "local_min_parallel_cancelled": int(local_min_parallel_cancelled),
            "local_min_hard_fail_stops": int(local_min_hard_fail_stops),
            "local_min_hard_fail_neighbors_skipped": int(local_min_hard_fail_neighbors_skipped),
            "local_min_hard_fail_cancelled": int(local_min_hard_fail_cancelled),
            "local_min_dependency_signal_total": int(local_min_dependency_signal_total),
            "local_min_dependency_signal_evaluated": int(local_min_dependency_signal_evaluated),
            "local_min_dependency_portfolio_total": int(local_min_dependency_portfolio_total),
            "local_min_dependency_portfolio_evaluated": int(local_min_dependency_portfolio_evaluated),
            "local_min_dependency_mixed_total": int(local_min_dependency_mixed_total),
            "local_min_dependency_mixed_evaluated": int(local_min_dependency_mixed_evaluated),
            "local_min_dependency_unknown_total": int(local_min_dependency_unknown_total),
            "local_min_dependency_unknown_evaluated": int(local_min_dependency_unknown_evaluated),
            "local_min_signal_reuse_candidate_total": int(local_min_signal_reuse_candidate_total),
            "local_min_signal_reuse_candidate_evaluated": int(local_min_signal_reuse_candidate_evaluated),
            "local_min_signal_recompute_required_total": int(local_min_signal_recompute_required_total),
            "local_min_signal_recompute_required_evaluated": int(local_min_signal_recompute_required_evaluated),
            "local_min_dependency_field_total_counts": str(local_min_dependency_field_total_counts),
            "local_min_dependency_field_evaluated_counts": str(local_min_dependency_field_evaluated_counts),
            "rolling_fold_workers_max": int(rolling_fold_workers_max),
            "rolling_fold_parallel": bool(rolling_fold_parallel),
            "profile_write_files": is_optimizer_profile_write_files_enabled(outer_rolling=True),
            "study_storage": "sqlite" if _is_outer_rolling_sqlite_storage_enabled(os.environ) else "memory",
            "parallel_fold_log_mode": "progress_only" if bool(rolling_fold_parallel) else "normal",
            "parallel_worker_prep_cache_max_items": _resolve_parallel_worker_prep_cache_max_items(os.environ),
            "resource_write_csv": bool(resource_write_csv),
            "resource_csv_rows": int(len(samples_for_csv)) if resource_write_csv and samples_for_csv else 0,
            "active_replay_include_trade_logs": is_optimizer_active_replay_include_trade_logs_enabled(),
            "active_replay_include_pit_stats_index": is_optimizer_active_replay_include_pit_stats_index_enabled(),
            "active_replay_use_prepared_cache": is_optimizer_active_replay_use_prepared_cache_enabled(),
            "active_replay_write_prepared_cache": is_optimizer_active_replay_write_prepared_cache_enabled(),
            **dict(resource_summary or {}),
            "prep_executor_created": int(prep_executor_created),
            "prep_executor_reused": int(prep_executor_reused),
        },
        "folds": fold_timing_rows,
        "csv_path": csv_path if fold_timing_rows else "",
        "resource_csv_path": resource_csv_path,
        "json_path": json_path,
    }
    if bool(write_files) and json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    return {"json": json_path, "csv": csv_path if (bool(write_files) and fold_timing_rows) else "", "resource_csv": resource_csv_path, "payload": payload}
