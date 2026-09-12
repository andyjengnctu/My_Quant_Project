from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd

from core.active_param_ensemble import ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING, ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE
from core.portfolio_stats import calc_plain_romd
from core.raw_universe_contract import (
    RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD,
    build_raw_universe_contract_fields as _raw_universe_contract_fields,
    coerce_raw_universe_required_min_rows,
)
from core.runtime_utils import get_process_pool_executor_kwargs
from core.seed_ensemble_policy import build_seed_ensemble_policy_snapshot, renumber_seed_ensemble_members
from core.training_performance import (
    is_optimizer_policy_replay_context_reuse_enabled_default,
    resolve_optimizer_policy_replay_parallel_backend_default,
    resolve_optimizer_policy_replay_parallel_workers_default,
    resolve_optimizer_random_seed_ensemble_parallel_workers_default,
)
from services.optimizer.outer_rolling_active_replay import _extract_active_replay_metrics
from services.optimizer.outer_rolling_artifacts import (
    _build_effective_seed_ensemble_policy_payload,
    _build_params_ensemble_members_for_schedule,
    _build_seed_ensemble_policy_payload,
)
from services.optimizer.outer_rolling_policy import (
    _finalists_agree_metadata_keys,
    _finalists_agree_policy_config,
    _is_finalist_best_policy,
    _is_finalists_agree_policy,
    _is_rolling_random_seed_ensemble_enabled,
    _resolve_finalists_agree_min_agree,
    select_finalist_best_members,
    select_finalists_agree_members,
)

_build_rolling_seed_ensemble_policy_payload = _build_seed_ensemble_policy_payload

def _effective_rolling_seed_ensemble_policy_payload() -> dict:
    """Return the effective runtime seed policy, not merely the configured capacity."""
    requested = _build_seed_ensemble_policy_payload()
    if _is_rolling_random_seed_ensemble_enabled():
        return dict(requested)
    effective = build_seed_ensemble_policy_snapshot(enabled=False, seed_count=1, min_agree=1)
    effective["seed_mode"] = "single_canonical_optimizer_seed"
    effective["requested_random_seed_ensemble"] = dict(requested)
    return effective

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
    raw_data_loader_path: str | None = None,
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
        "raw_data_loader_path": str(raw_data_loader_path or ""),
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
    raw_data_loader = None
    raw_data_loader_path = str(replay_context.get("raw_data_loader_path") or "").strip()
    if raw_data_loader_path:
        from services.optimizer.callable_ref import load_callable_from_import_path

        raw_data_loader = load_callable_from_import_path(raw_data_loader_path)
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
        raw_data_loader=raw_data_loader,
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

