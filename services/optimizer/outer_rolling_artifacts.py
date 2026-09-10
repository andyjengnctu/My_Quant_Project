from __future__ import annotations

import os
import statistics

from config.training_policy import (
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
)
from core.active_param_ensemble import ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING, ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE
from core.display import C_RESET, C_YELLOW
from core.file_integrity import atomic_write_json
from core.model_paths import resolve_models_dir
from core.raw_universe_contract import build_raw_universe_contract_fields as _raw_universe_contract_fields
from core.rolling_oos_params import ROLLING_OOS_PARAM_SET_SCHEMA_TYPE, ROLLING_OOS_USAGE
from core.runtime_utils import get_taipei_now
from core.seed_ensemble_policy import (
    build_seed_ensemble_policy_snapshot,
    normalize_seed_ensemble_members,
    renumber_seed_ensemble_members,
)
from core.strategy_param_artifacts import normalize_strategy_param_payload_for_persistence
from core.training_policy import is_optimizer_local_min_review_enabled
from services.optimizer.outer_rolling_fold_context import normalize_optimizer_seed_ensemble_fold_rows
from services.optimizer.outer_rolling_plan import OuterRollingConfig
from services.optimizer.outer_rolling_policy import (
    CHAIN_POLICY_NAMES,
    PARAMSET_FILENAME_BY_POLICY,
    REPORT_POLICY_NAMES,
    STALE_POLICY_PARAMSET_FILENAMES,
    _active_optimizer_table_titles,
    _finalists_agree_policy_config,
    _is_finalists_agree_policy,
    _policy_is_available,
    _policy_plain_romd_score,
    _resolve_finalists_agree_min_agree,
)
from services.optimizer.outer_rolling_results import (
    _build_chained_oos_summary,
    _render_optimizer_results_tables,
    _rows_period_bounds,
)

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
