from __future__ import annotations

from decimal import Decimal
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


from config.training_performance_policy import (
    is_optimizer_local_min_dependency_stats_enabled,
    resolve_optimizer_local_min_portfolio_dependency_order,
    resolve_optimizer_local_min_signal_dependency_field_order,
    resolve_optimizer_single_fold_local_min_parallel_workers_default,
)
from config.training_policy import (
    OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE,
    OPTIMIZER_INNER_VALIDATE_MIN_SCORE,
    is_optimizer_local_min_review_enabled,
    resolve_optimizer_local_min_score_finalist_top_k,
)
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.runtime_utils import choose_inline_progress_message, stdout_supports_inline_progress, write_inline_progress
from core.strategy_params import build_runtime_param_raw_value
from strategies.breakout.search_space import (
    classify_breakout_local_min_dependency_layer,
    get_breakout_local_min_candidate_fields,
    resolve_breakout_neighbor_spec,
)
from tools.optimizer.objective_runner import (
    evaluate_prepared_inner_validate_score,
    evaluate_prepared_train_score,
    resolve_inner_validate_scope,
    resolve_search_train_scope,
)
from tools.optimizer.param_cache import build_full_evaluation_cache_key, build_prep_cache_key
from tools.optimizer.prep import prepare_trial_inputs
from tools.optimizer.walk_forward import evaluate_walk_forward
from tools.optimizer.study_utils import (
    INVALID_TRIAL_VALUE,
    OBJECTIVE_MODE_SPLIT_TRAIN_ROMD,
    OPTIMIZER_TP_PERCENT_SEARCH_SPEC,
    build_best_params_payload_from_trial,
    is_qualified_trial_value,
    list_completed_study_trials,
    normalize_objective_mode,
    objective_modes_are_compatible,
)



def _get_progress_colors(session):
    colors = getattr(session, "colors", None)
    return colors if isinstance(colors, dict) else {}


def _resolve_local_min_progress_min_interval_sec():
    raw_value = os.environ.get("OPTIMIZER_LOCAL_MIN_PROGRESS_MIN_INTERVAL_SEC", "1.0")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(10.0, float(value)))


def _get_local_min_score_cache(session):
    if not hasattr(session, "local_min_score_cache"):
        session.local_min_score_cache = {}
    return session.local_min_score_cache


def _get_local_min_payload_score_cache(session):
    if not hasattr(session, "local_min_payload_score_cache"):
        session.local_min_payload_score_cache = {}
    return session.local_min_payload_score_cache


def _get_local_min_order_score_cache(session):
    cache = getattr(session, "local_min_order_score_cache", None)
    if isinstance(cache, dict):
        return cache
    session.local_min_order_score_cache = {}
    return session.local_min_order_score_cache


def _get_dominant_year_dependency_cache(session):
    if not hasattr(session, "dominant_year_dependency_cache"):
        session.dominant_year_dependency_cache = {}
    return session.dominant_year_dependency_cache


def _get_inner_validate_cache(session):
    if not hasattr(session, "inner_validate_cache"):
        session.inner_validate_cache = {}
    return session.inner_validate_cache


def _get_oos_cache(session):
    if not hasattr(session, "oos_cache"):
        session.oos_cache = {}
    return session.oos_cache


def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', str(text))


def _format_year_range(start_year, end_year) -> str:
    if start_year in (None, '', 0) and end_year in (None, '', 0):
        return 'N/A'
    if end_year in (None, '', 0):
        return f"{start_year}~latest"
    if start_year == end_year:
        return str(start_year)
    return f"{start_year}~{end_year}"


def _format_progress_value(value, *, compact: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return "N/A"
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        year, month, _day = match.groups()
        return f"{year[-2:]}-{month}" if compact else f"{year}-{month}"
    if re.fullmatch(r"\d{6}", text):
        return f"{text[-4:-2]}-{text[-2:]}" if compact else f"{text[:4]}-{text[4:]}"
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return text[-5:] if compact else text
    if re.fullmatch(r"\d{4}", text):
        return text[-2:] if compact else text
    return text


def _format_progress_range(start_value, end_value, *, compact: bool = False) -> str:
    start_text = _format_progress_value(start_value, compact=compact)
    end_text = _format_progress_value(end_value, compact=compact)
    if start_text == "N/A" and end_text == "N/A":
        return "N/A"
    if end_text == "N/A":
        return f"{start_text}~latest"
    if start_text == end_text:
        return start_text
    return f"{start_text}~{end_text}"


def _resolve_report_periods(session, *, objective_mode: str) -> dict:
    inner_policy = resolve_inner_validate_scope(session, getattr(session, 'sorted_master_dates', []), objective_mode=objective_mode).get('policy', {})
    train_start_year = int(getattr(session, 'train_start_year', 0) or 0)
    search_train_end_year = int(getattr(session, 'search_train_end_year', 0) or 0)
    if bool(inner_policy.get('enabled', False)):
        training_period = _format_year_range(train_start_year, inner_policy.get('inner_train_end_year'))
        validate_period = _format_year_range(inner_policy.get('validate_start_year'), inner_policy.get('validate_end_year'))
    else:
        training_period = _format_year_range(train_start_year, search_train_end_year)
        validate_period = 'disabled'
    latest_year = None
    sorted_master_dates = list(getattr(session, 'sorted_master_dates', []) or [])
    if sorted_master_dates:
        latest_year = int(getattr(sorted_master_dates[-1], 'year', 0) or 0) or None
    oos_start_year = getattr(session, 'walk_forward_policy', {}).get('oos_start_year')
    if oos_start_year is None:
        oos_period = 'disabled'
    else:
        oos_period = _format_year_range(int(oos_start_year), latest_year)
    return {
        'training_period': training_period,
        'validate_period': validate_period,
        'oos_test_period': oos_period,
    }


def _build_gate_status_rows(finalists: list[dict], *, objective_mode: str) -> list[dict]:
    gate_rows: list[dict] = []
    gate_rows.append({
        'name': 'LOCAL_MIN_SCORE',
        'enabled': bool(is_optimizer_local_min_review_enabled()),
        'ratio_text': '-',
    })
    inner_enabled = is_inner_validate_anti_overfit_enabled(objective_mode)
    ratio_text = '-'
    if inner_enabled:
        pass_items = [
            item for item in finalists
            if bool(item.get('gate_pass', False))
            and isinstance(item.get('inner_validate_diagnostics'), dict)
        ]
        allowed_count = 0
        total_count = len(pass_items)
        if pass_items:
            diagnostics = pass_items[0].get('inner_validate_diagnostics') or {}
            allowed_count = int(diagnostics.get('inner_validate_rank_cutoff', 0) or 0)
        ratio_pct = float(OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE) * 100.0
        ratio_text = f"{ratio_pct:.0f}%"
        if total_count > 0:
            ratio_text += f" ({allowed_count}/{total_count})"
    gate_rows.append({
        'name': 'INNER_VALIDATE_RANK',
        'enabled': bool(inner_enabled),
        'ratio_text': ratio_text,
    })
    gate_rows.append({
        'name': 'DOMINANT_YEAR_DEPENDENCY',
        'enabled': bool(is_dominant_year_dependency_anti_overfit_enabled()),
        'ratio_text': '-',
    })
    return gate_rows


def _build_payload_score_cache_key(payload: dict):
    canonical_payload = params_to_json_dict(build_params_from_mapping(payload))
    return tuple(sorted(canonical_payload.items()))


def _get_local_min_field_order_score_cache(session):
    cache = getattr(session, "local_min_field_order_score_cache", None)
    if isinstance(cache, dict):
        return cache
    session.local_min_field_order_score_cache = {}
    return session.local_min_field_order_score_cache


def _get_local_min_field_order_score_update_cache(session):
    cache = getattr(session, "local_min_field_order_score_update_cache", None)
    if isinstance(cache, dict):
        return cache
    session.local_min_field_order_score_update_cache = {}
    return session.local_min_field_order_score_update_cache


def _infer_neighbor_delta(center_payload: dict, payload: dict) -> tuple[str, int] | None:
    changed_fields = [
        field_name for field_name in sorted(set(center_payload) | set(payload))
        if center_payload.get(field_name) != payload.get(field_name)
    ]
    if len(changed_fields) != 1:
        return None
    field_name = changed_fields[0]
    old_value = center_payload.get(field_name)
    new_value = payload.get(field_name)
    try:
        direction = 1 if float(new_value) > float(old_value) else -1
    except (TypeError, ValueError):
        direction = 1 if str(new_value) > str(old_value) else -1
    return str(field_name), int(direction)


def _infer_neighbor_changed_fields(center_payload: dict, payload: dict) -> list[str]:
    return [
        str(field_name) for field_name in sorted(set(center_payload or {}) | set(payload or {}))
        if (center_payload or {}).get(field_name) != (payload or {}).get(field_name)
    ]


def _classify_local_min_neighbor_dependency(center_payload: dict, payload: dict) -> tuple[str, str]:
    changed_fields = _infer_neighbor_changed_fields(center_payload, payload)
    if len(changed_fields) == 1:
        field_name = str(changed_fields[0])
        layer = str(classify_breakout_local_min_dependency_layer(field_name))
        if layer not in {"signal", "portfolio"}:
            layer = "unknown"
        return layer, field_name
    if len(changed_fields) > 1:
        return "mixed", "+".join(changed_fields)
    return "unknown", "unchanged"


def _empty_local_min_dependency_runtime_stats() -> dict:
    return {
        "dependency_signal_total": 0,
        "dependency_signal_evaluated": 0,
        "dependency_portfolio_total": 0,
        "dependency_portfolio_evaluated": 0,
        "dependency_mixed_total": 0,
        "dependency_mixed_evaluated": 0,
        "dependency_unknown_total": 0,
        "dependency_unknown_evaluated": 0,
        "signal_reuse_candidate_total": 0,
        "signal_reuse_candidate_evaluated": 0,
        "signal_recompute_required_total": 0,
        "signal_recompute_required_evaluated": 0,
        "dependency_field_total_counts": {},
        "dependency_field_evaluated_counts": {},
    }


def _record_local_min_dependency(stats: dict, *, layer: str, field_name: str, bucket: str) -> None:
    if not isinstance(stats, dict):
        return
    normalized_bucket = "evaluated" if str(bucket) == "evaluated" else "total"
    normalized_layer = str(layer or "unknown")
    if normalized_layer not in {"signal", "portfolio", "mixed", "unknown"}:
        normalized_layer = "unknown"
    stats[f"dependency_{normalized_layer}_{normalized_bucket}"] = int(stats.get(f"dependency_{normalized_layer}_{normalized_bucket}", 0) or 0) + 1
    if normalized_layer == "portfolio":
        stats[f"signal_reuse_candidate_{normalized_bucket}"] = int(stats.get(f"signal_reuse_candidate_{normalized_bucket}", 0) or 0) + 1
    else:
        stats[f"signal_recompute_required_{normalized_bucket}"] = int(stats.get(f"signal_recompute_required_{normalized_bucket}", 0) or 0) + 1
    field_counts_key = f"dependency_field_{normalized_bucket}_counts"
    if not isinstance(stats.get(field_counts_key), dict):
        stats[field_counts_key] = {}
    resolved_field = str(field_name or "unknown")
    stats[field_counts_key][resolved_field] = int(stats[field_counts_key].get(resolved_field, 0) or 0) + 1


def _build_local_min_dependency_total_stats(center_payload: dict, neighbor_payloads: list[dict]) -> dict:
    stats = _empty_local_min_dependency_runtime_stats()
    if not is_optimizer_local_min_dependency_stats_enabled():
        return stats
    for payload in list(neighbor_payloads or []):
        layer, field_name = _classify_local_min_neighbor_dependency(center_payload, payload)
        _record_local_min_dependency(stats, layer=layer, field_name=field_name, bucket="total")
    return stats


def _record_local_min_dependency_evaluated(stats: dict, center_payload: dict, payload: dict) -> None:
    if not is_optimizer_local_min_dependency_stats_enabled():
        return
    layer, field_name = _classify_local_min_neighbor_dependency(center_payload, payload)
    _record_local_min_dependency(stats, layer=layer, field_name=field_name, bucket="evaluated")


def _get_local_min_dependency_sort_bucket(center_payload: dict, payload: dict) -> int:
    order_mode = resolve_optimizer_local_min_portfolio_dependency_order()
    if order_mode != "last":
        return 0
    layer, _field_name = _classify_local_min_neighbor_dependency(center_payload, payload)
    return 1 if str(layer) == "portfolio" else 0


def _get_local_min_signal_dependency_field_priority(center_payload: dict, payload: dict) -> tuple[int, bool]:
    layer, field_name = _classify_local_min_neighbor_dependency(center_payload, payload)
    if str(layer) != "signal":
        return 0, False
    ordered_fields = tuple(resolve_optimizer_local_min_signal_dependency_field_order() or ())
    if not ordered_fields:
        return 0, False
    priority_map = {str(name): idx for idx, name in enumerate(ordered_fields)}
    field = str(field_name or "")
    if field in priority_map:
        return int(priority_map[field]), True
    return len(priority_map), False


def _rank_local_min_neighbor_payloads(session, center_payload: dict, neighbor_payloads: list[dict], payload_score_cache: dict) -> tuple[list[dict], dict]:
    """Evaluate cheap / likely-pruning neighbors first without changing local-min semantics.

    The local-min score is the minimum over the same neighbor set, so order does
    not affect exact reviews.  In outer rolling selection-pruning mode, an early
    low neighbor can safely stop the review because additional neighbors can only
    keep or lower the minimum.  Cross-fold order scores are used only as ordering
    hints; they are never reused as current-fold scores.
    """

    has_prep_cache = getattr(session, "has_prepared_trial_inputs_in_cache", None)
    order_score_cache = _get_local_min_order_score_cache(session)
    field_order_score_cache = _get_local_min_field_order_score_cache(session)
    ranked_items = []
    prep_cache_prioritized = 0
    payload_score_candidates = 0
    order_score_prioritized = 0
    field_order_score_prioritized = 0
    portfolio_dependency_deprioritized = 0
    signal_dependency_field_prioritized = 0
    for original_idx, payload in enumerate(list(neighbor_payloads or [])):
        dependency_sort_bucket = _get_local_min_dependency_sort_bucket(center_payload, payload)
        if dependency_sort_bucket > 0:
            portfolio_dependency_deprioritized += 1
        signal_dependency_field_priority, has_signal_dependency_field_priority = _get_local_min_signal_dependency_field_priority(center_payload, payload)
        if has_signal_dependency_field_priority:
            signal_dependency_field_prioritized += 1
        payload_cache_key = _build_payload_score_cache_key(payload)
        cached_payload_score = payload_score_cache.get(payload_cache_key)
        if cached_payload_score is not None:
            payload_score_candidates += 1
            try:
                cached_score_sort_value = float(cached_payload_score)
            except (TypeError, ValueError):
                cached_score_sort_value = float(INVALID_TRIAL_VALUE)
            ranked_items.append((0, cached_score_sort_value, 0, signal_dependency_field_priority, original_idx, payload))
            continue

        order_score = order_score_cache.get(payload_cache_key)
        has_order_score = order_score is not None
        try:
            order_score_sort_value = float(order_score) if has_order_score else 0.0
        except (TypeError, ValueError):
            has_order_score = False
            order_score_sort_value = 0.0

        delta_key = _infer_neighbor_delta(center_payload, payload)
        field_order_score = field_order_score_cache.get(delta_key) if delta_key is not None else None
        has_field_order_score = field_order_score is not None
        try:
            field_order_score_sort_value = float(field_order_score) if has_field_order_score else 0.0
        except (TypeError, ValueError):
            has_field_order_score = False
            field_order_score_sort_value = 0.0

        prep_cached = False
        if callable(has_prep_cache):
            try:
                ai_params = build_params_from_mapping(payload)
                prep_cached = bool(has_prep_cache(build_prep_cache_key(ai_params)))
            except (TypeError, ValueError, KeyError, AttributeError):
                prep_cached = False

        if prep_cached:
            prep_cache_prioritized += 1
            if has_order_score:
                order_score_prioritized += 1
                ranked_items.append((1, order_score_sort_value, dependency_sort_bucket, signal_dependency_field_priority, original_idx, payload))
            elif has_field_order_score:
                field_order_score_prioritized += 1
                ranked_items.append((2, field_order_score_sort_value, dependency_sort_bucket, signal_dependency_field_priority, original_idx, payload))
            else:
                ranked_items.append((3, 0.0, dependency_sort_bucket, signal_dependency_field_priority, original_idx, payload))
        elif has_order_score:
            order_score_prioritized += 1
            ranked_items.append((4, order_score_sort_value, dependency_sort_bucket, signal_dependency_field_priority, original_idx, payload))
        elif has_field_order_score:
            field_order_score_prioritized += 1
            ranked_items.append((5, field_order_score_sort_value, dependency_sort_bucket, signal_dependency_field_priority, original_idx, payload))
        else:
            ranked_items.append((6, 0.0, dependency_sort_bucket, signal_dependency_field_priority, original_idx, payload))

    ranked_items.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return [item[5] for item in ranked_items], {
        "payload_score_candidates": int(payload_score_candidates),
        "prep_cache_prioritized": int(prep_cache_prioritized),
        "order_score_prioritized": int(order_score_prioritized),
        "field_order_score_prioritized": int(field_order_score_prioritized),
        "portfolio_dependency_deprioritized": int(portfolio_dependency_deprioritized),
        "signal_dependency_field_prioritized": int(signal_dependency_field_prioritized),
    }


def _resolve_local_min_parallel_workers(session) -> int:
    default_workers = resolve_optimizer_single_fold_local_min_parallel_workers_default()
    raw_value = os.environ.get("OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS", str(default_workers))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = int(default_workers)
    # AI註: local-min 平行化採小型 ordered prefetch window，避免重演 batch 過度超前計算。
    return max(1, min(4, int(value)))


def _get_local_min_thread_lock(session):
    lock = getattr(session, "local_min_thread_lock", None)
    if lock is None:
        import threading
        lock = threading.RLock()
        session.local_min_thread_lock = lock
    return lock


def _read_payload_score_cache(payload_score_cache: dict, payload_cache_key):
    try:
        cached_score = payload_score_cache.get(payload_cache_key)
    except AttributeError:
        return None
    return cached_score


def _evaluate_local_min_neighbor_payload(session, payload: dict, payload_score_cache: dict) -> dict:
    payload_cache_key = _build_payload_score_cache_key(payload)
    lock = _get_local_min_thread_lock(session)
    with lock:
        cached_payload_score = _read_payload_score_cache(payload_score_cache, payload_cache_key)
    if cached_payload_score is not None:
        return {
            "payload": payload,
            "payload_cache_key": payload_cache_key,
            "score": float(cached_payload_score),
            "payload_score_cache_hit": True,
        }

    ai_params = build_params_from_mapping(payload)
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
            profile_enabled=False,
        )
        cache_prep = getattr(session, "cache_prepared_trial_inputs", None)
        if callable(cache_prep):
            cache_prep(prep_cache_key, prep_result)
    search_scope = resolve_search_train_scope(session, prep_result["master_dates"], objective_mode=session.objective_mode)
    full_evaluation_cache_key = build_full_evaluation_cache_key(
        ai_params,
        objective_mode=search_scope.get("mode", session.objective_mode),
        train_start_year=session.train_start_year,
        search_train_end_year=int(search_scope.get("effective_search_train_end_year", session.search_train_end_year)),
        train_start_date=(getattr(session, "walk_forward_policy", {}) or {}).get("train_start_date"),
        search_train_end_date=search_scope.get("effective_search_train_end_date"),
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
    )
    get_cached_evaluation = getattr(session, "get_full_evaluation_from_cache", None)
    evaluation = get_cached_evaluation(full_evaluation_cache_key) if callable(get_cached_evaluation) else None
    if evaluation is None:
        evaluation = evaluate_prepared_train_score(
            session,
            ai_params=ai_params,
            prep_result=prep_result,
            search_scope=search_scope,
            profile_stats=None,
        )
        cache_evaluation = getattr(session, "cache_full_evaluation", None)
        if callable(cache_evaluation):
            cache_evaluation(full_evaluation_cache_key, evaluation)
    score = float(evaluation["score"])
    with lock:
        payload_score_cache[payload_cache_key] = float(score)
    return {
        "payload": payload,
        "payload_cache_key": payload_cache_key,
        "score": float(score),
        "payload_score_cache_hit": False,
    }


def _update_local_min_order_hints(session, center_payload: dict, payload: dict, payload_cache_key, score: float):
    lock = _get_local_min_thread_lock(session)
    with lock:
        _get_local_min_order_score_cache(session)[payload_cache_key] = float(score)
        delta_key = _infer_neighbor_delta(center_payload, payload)
        if delta_key is not None:
            field_score_cache = _get_local_min_field_order_score_update_cache(session)
            prior_score = field_score_cache.get(delta_key)
            try:
                should_update_field_score = prior_score is None or float(score) < float(prior_score)
            except (TypeError, ValueError):
                should_update_field_score = True
            if bool(should_update_field_score):
                field_score_cache[delta_key] = float(score)


def _is_local_min_hard_fail_score(score) -> bool:
    try:
        return float(score) <= float(INVALID_TRIAL_VALUE)
    except (TypeError, ValueError):
        return False


def _should_stop_local_min(local_min_score: float, selection_prune_floor) -> tuple[bool, bool]:
    if _is_local_min_hard_fail_score(local_min_score):
        return True, False
    if float(local_min_score) <= 0.0:
        return True, False
    if selection_prune_floor is not None and float(local_min_score) < float(selection_prune_floor):
        return True, True
    return False, False


def _evaluate_local_min_neighbors_ordered(
    session,
    *,
    center_payload: dict,
    neighbor_payloads: list[dict],
    payload_score_cache: dict,
    total_neighbors: int,
    selection_prune_floor,
    on_neighbor=None,
) -> dict:
    workers = _resolve_local_min_parallel_workers(session)
    dependency_stats = _build_local_min_dependency_total_stats(center_payload, neighbor_payloads)
    if workers <= 1 or int(total_neighbors) <= 1:
        local_min_score = float("inf")
        evaluated_neighbors = 0
        payload_score_cache_hit_count = 0
        early_stopped = False
        selection_pruned = False
        hard_fail_stopped = False
        hard_fail_neighbors_skipped = 0
        hard_fail_cancelled = 0
        for neighbor_idx, payload in enumerate(neighbor_payloads, start=1):
            evaluated_neighbors = neighbor_idx
            result = _evaluate_local_min_neighbor_payload(session, payload, payload_score_cache)
            _record_local_min_dependency_evaluated(dependency_stats, center_payload, payload)
            score = float(result["score"])
            if bool(result.get("payload_score_cache_hit", False)):
                payload_score_cache_hit_count += 1
            _update_local_min_order_hints(session, center_payload, payload, result["payload_cache_key"], score)
            if score < local_min_score:
                local_min_score = score
            if on_neighbor is not None:
                current_local_min = None if local_min_score == float("inf") else float(local_min_score)
                on_neighbor(neighbor_idx, total_neighbors, current_local_min)
            should_stop, stopped_by_prune = _should_stop_local_min(local_min_score, selection_prune_floor)
            if bool(should_stop):
                early_stopped = neighbor_idx < total_neighbors
                selection_pruned = bool(stopped_by_prune) and neighbor_idx < total_neighbors
                hard_fail_stopped = bool(_is_local_min_hard_fail_score(local_min_score)) and neighbor_idx < total_neighbors
                if bool(hard_fail_stopped):
                    hard_fail_neighbors_skipped = max(0, int(total_neighbors) - int(neighbor_idx))
                break
        return {
            "local_min_score": local_min_score,
            "evaluated_neighbors": int(evaluated_neighbors),
            "payload_score_cache_hits": int(payload_score_cache_hit_count),
            "early_stopped": bool(early_stopped),
            "selection_pruned": bool(selection_pruned),
            "parallel_workers": 1,
            "parallel_submitted": int(evaluated_neighbors),
            "parallel_completed": int(evaluated_neighbors),
            "parallel_cancelled": 0,
            "hard_fail_stopped": bool(hard_fail_stopped),
            "hard_fail_neighbors_skipped": int(hard_fail_neighbors_skipped),
            "hard_fail_cancelled": int(hard_fail_cancelled),
            **dependency_stats,
        }

    local_min_score = float("inf")
    evaluated_neighbors = 0
    payload_score_cache_hit_count = 0
    early_stopped = False
    selection_pruned = False
    hard_fail_stopped = False
    hard_fail_neighbors_skipped = 0
    hard_fail_cancelled = 0
    submitted = 0
    completed = 0
    cancelled = 0
    pending = {}
    next_submit_idx = 1

    executor = ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="local-min")
    try:
        while next_submit_idx <= total_neighbors and len(pending) < int(workers):
            payload = neighbor_payloads[next_submit_idx - 1]
            pending[next_submit_idx] = executor.submit(_evaluate_local_min_neighbor_payload, session, payload, payload_score_cache)
            submitted += 1
            next_submit_idx += 1

        neighbor_idx = 1
        while neighbor_idx <= total_neighbors:
            future = pending.pop(neighbor_idx, None)
            if future is None:
                break
            result = future.result()
            completed += 1
            evaluated_neighbors = neighbor_idx
            payload = neighbor_payloads[neighbor_idx - 1]
            _record_local_min_dependency_evaluated(dependency_stats, center_payload, payload)
            score = float(result["score"])
            if bool(result.get("payload_score_cache_hit", False)):
                payload_score_cache_hit_count += 1
            _update_local_min_order_hints(session, center_payload, payload, result["payload_cache_key"], score)
            if score < local_min_score:
                local_min_score = score
            if on_neighbor is not None:
                current_local_min = None if local_min_score == float("inf") else float(local_min_score)
                on_neighbor(neighbor_idx, total_neighbors, current_local_min)
            should_stop, stopped_by_prune = _should_stop_local_min(local_min_score, selection_prune_floor)
            if bool(should_stop):
                early_stopped = neighbor_idx < total_neighbors
                selection_pruned = bool(stopped_by_prune) and neighbor_idx < total_neighbors
                hard_fail_stopped = bool(_is_local_min_hard_fail_score(local_min_score)) and neighbor_idx < total_neighbors
                if bool(hard_fail_stopped):
                    hard_fail_neighbors_skipped = max(0, int(total_neighbors) - int(neighbor_idx))
                for pending_future in list(pending.values()):
                    if pending_future.cancel():
                        cancelled += 1
                        if bool(hard_fail_stopped):
                            hard_fail_cancelled += 1
                break

            while next_submit_idx <= total_neighbors and len(pending) < int(workers):
                payload = neighbor_payloads[next_submit_idx - 1]
                pending[next_submit_idx] = executor.submit(_evaluate_local_min_neighbor_payload, session, payload, payload_score_cache)
                submitted += 1
                next_submit_idx += 1
            neighbor_idx += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return {
        "local_min_score": local_min_score,
        "evaluated_neighbors": int(evaluated_neighbors),
        "payload_score_cache_hits": int(payload_score_cache_hit_count),
        "early_stopped": bool(early_stopped),
        "selection_pruned": bool(selection_pruned),
        "parallel_workers": int(workers),
        "parallel_submitted": int(submitted),
        "parallel_completed": int(completed),
        "parallel_cancelled": int(cancelled),
        "hard_fail_stopped": bool(hard_fail_stopped),
        "hard_fail_neighbors_skipped": int(hard_fail_neighbors_skipped),
        "hard_fail_cancelled": int(hard_fail_cancelled),
        **dependency_stats,
    }


def _get_seeded_payload_trial_numbers(session):
    if not hasattr(session, "local_min_seeded_payload_trial_numbers"):
        session.local_min_seeded_payload_trial_numbers = set()
    return session.local_min_seeded_payload_trial_numbers


def _seed_payload_score_cache_from_study(session, study, objective_mode: str):
    cache = _get_local_min_payload_score_cache(session)
    seeded_trial_numbers = _get_seeded_payload_trial_numbers(session)
    for completed_trial in _list_qualified_trials_for_objective(study, objective_mode):
        trial_number = int(completed_trial.number)
        if trial_number in seeded_trial_numbers:
            continue
        try:
            payload = build_best_params_payload_from_trial(
                completed_trial,
                fixed_tp_percent=session.optimizer_fixed_tp_percent,
            )
            cache[_build_payload_score_cache_key(payload)] = float(completed_trial.value)
            seeded_trial_numbers.add(trial_number)
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    return cache


def _get_best_trial_resolver_cache(session):
    if not hasattr(session, "local_min_best_trial_cache"):
        session.local_min_best_trial_cache = {}
    return session.local_min_best_trial_cache


def _build_best_trial_cache_key(study, objective_mode: str, top_k: int):
    return (id(study), str(normalize_objective_mode(objective_mode)), int(top_k))


def _print_progress_line(session, message: str):
    print(message, flush=True)


def _emit_local_min_progress_event(session, event: dict) -> None:
    sink = getattr(session, "outer_rolling_parallel_progress_sink", None)
    if not callable(sink):
        return
    try:
        sink(dict(event or {}))
    except (TypeError, ValueError, RuntimeError, OSError) as exc:
        # AI註: progress sink 僅用於 timing 顯示；失敗不可中斷 optimizer 主流程，但需可追蹤。
        if not hasattr(session, "outer_rolling_parallel_progress_sink_error"):
            session.outer_rolling_parallel_progress_sink_error = repr(exc)


class _FinalistProgressBoard:
    def __init__(self, session, finalists: list[dict]):
        self.session = session
        self.finalists = finalists
        self.lines: list[str] = []
        self.rendered = False
        self.single_line_context = getattr(session, "outer_rolling_local_progress_context", None)
        self.event_only = bool(getattr(session, "local_min_progress_event_only", False))
        self.inline_progress_enabled = bool(self.single_line_context) and stdout_supports_inline_progress()
        self.inline_progress_width = 0
        self.inline_progress_closed = False
        self.stage_start = time.perf_counter()
        self.stage_start_ts = time.time()
        self.best_local_score = float("-inf")
        self.best_local_trial = None
        self.completed_status: dict[int, tuple[float, bool, bool]] = {}
        self.progress_min_interval_sec = _resolve_local_min_progress_min_interval_sec()
        self._last_progress_render_at = 0.0

    def _format_duration(self, seconds) -> str:
        if seconds is None:
            return "N/A"
        total = max(0, int(float(seconds)))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _format_duration_compact(self, seconds) -> str:
        text = self._format_duration(seconds)
        if text.startswith("00:"):
            return text[3:]
        return text

    def _emit_progress_event(self, idx: int, *, current_neighbor: int, total_neighbors: int, current_local_min, status_text: str):
        if not self.finalists:
            return
        safe_idx = min(max(0, int(idx)), len(self.finalists) - 1)
        trial = self.finalists[safe_idx]["trial"]
        best_score = None if self.best_local_score == float("-inf") else float(self.best_local_score)
        _emit_local_min_progress_event(
            self.session,
            {
                "finalist_idx": int(safe_idx) + 1,
                "finalist_total": len(self.finalists),
                "trial_number": int(trial.number) + 1,
                "neighbor_done": int(current_neighbor or 0),
                "neighbor_total": int(total_neighbors or 0),
                "current": None if current_local_min is None else float(current_local_min),
                "best": best_score,
                "status": str(status_text or "RUN"),
                "local_min_started_ts": float(self.stage_start_ts),
                "local_min_completed": int(len(self.completed_status)),
                "local_min_last_done_ts": time.time() if int(len(self.completed_status)) > 0 else None,
            },
        )

    def _format_line(self, idx: int, *, prefix: str, progress_text: str, local_text: str, status_text: str):
        finalist = self.finalists[idx]
        trial = finalist["trial"]
        base_score = float(finalist["base_score"])
        return (
            f"{prefix} finalist {idx + 1}/{len(self.finalists)} | trial #{int(trial.number) + 1} "
            f"| base_score={base_score:.3f} | {progress_text} | local_min_score={local_text} | {status_text}"
        )

    def initialize(self):
        self._emit_progress_event(0, current_neighbor=0, total_neighbors=0, current_local_min=None, status_text="WAIT")
        if self.event_only:
            return
        if self.single_line_context and self.inline_progress_enabled:
            self._render_single_line(0, current_neighbor=0, total_neighbors=0, local_min_score=None, status_text="WAIT")
            return
        self.lines = [
            self._format_line(idx, prefix="⏳", progress_text="進度 0/0", local_text="N/A", status_text="等待中")
            for idx in range(len(self.finalists))
        ]
        self._render()

    def update_pending(self, idx: int, *, total_neighbors: int):
        self._emit_progress_event(idx, current_neighbor=0, total_neighbors=total_neighbors, current_local_min=None, status_text="RUN")
        if self.event_only:
            return
        if self.single_line_context and self.inline_progress_enabled:
            self._render_single_line(idx, current_neighbor=0, total_neighbors=total_neighbors, local_min_score=None, status_text="RUN")
            return
        self.lines[idx] = self._format_line(
            idx,
            prefix="⏳",
            progress_text=f"進度 0/{int(total_neighbors)}",
            local_text="N/A",
            status_text="分析中",
        )
        self._render()

    def _should_render_progress_update(self, *, current_neighbor: int, total_neighbors: int) -> bool:
        current_neighbor = int(current_neighbor or 0)
        total_neighbors = int(total_neighbors or 0)
        if current_neighbor <= 1 or (total_neighbors > 0 and current_neighbor >= total_neighbors):
            self._last_progress_render_at = time.perf_counter()
            return True
        min_interval = float(self.progress_min_interval_sec)
        if min_interval <= 0.0:
            return True
        now = time.perf_counter()
        if now - float(self._last_progress_render_at or 0.0) >= min_interval:
            self._last_progress_render_at = now
            return True
        return False

    def update_neighbor(self, idx: int, *, current_neighbor: int, total_neighbors: int, current_local_min):
        if not self._should_render_progress_update(current_neighbor=current_neighbor, total_neighbors=total_neighbors):
            return
        self._emit_progress_event(idx, current_neighbor=current_neighbor, total_neighbors=total_neighbors, current_local_min=current_local_min, status_text="RUN")
        if self.event_only:
            return
        if self.single_line_context and self.inline_progress_enabled:
            self._render_single_line(
                idx,
                current_neighbor=current_neighbor,
                total_neighbors=total_neighbors,
                local_min_score=current_local_min,
                status_text="RUN",
            )
            return
        local_text = "N/A" if current_local_min is None else f"{float(current_local_min):.3f}"
        self.lines[idx] = self._format_line(
            idx,
            prefix="⏳",
            progress_text=f"進度 {int(current_neighbor)}/{int(total_neighbors)}",
            local_text=local_text,
            status_text="分析中",
        )
        self._render()

    def update_cache(self, idx: int, *, total_neighbors: int, local_min_score: float):
        self._record_done(idx, local_min_score=float(local_min_score), early_stopped=False)
        self._emit_progress_event(idx, current_neighbor=total_neighbors, total_neighbors=total_neighbors, current_local_min=local_min_score, status_text="cache")
        if self.event_only:
            self._close_single_line_if_finished()
            return
        if self.single_line_context and self.inline_progress_enabled:
            gate_status = "PASS" if float(local_min_score) > 0.0 else "FAIL"
            self._render_single_line(
                idx,
                current_neighbor=total_neighbors,
                total_neighbors=total_neighbors,
                local_min_score=local_min_score,
                status_text=f"{gate_status} cache",
                done=True,
            )
            self._close_single_line_if_finished()
            return
        gate_status = "PASS" if float(local_min_score) > 0.0 else "FAIL"
        self.lines[idx] = self._format_line(
            idx,
            prefix="ℹ️",
            progress_text=f"進度 {int(total_neighbors)}/{int(total_neighbors)}",
            local_text=f"{float(local_min_score):.3f}",
            status_text=f"{gate_status} | 快取",
        )
        self._render()

    def update_done(self, idx: int, *, evaluated_neighbors: int, total_neighbors: int, local_min_score: float, early_stopped: bool = False):
        self._record_done(idx, local_min_score=float(local_min_score), early_stopped=bool(early_stopped))
        self._emit_progress_event(idx, current_neighbor=evaluated_neighbors, total_neighbors=total_neighbors, current_local_min=local_min_score, status_text="early_stop" if bool(early_stopped) else "DONE")
        if self.event_only:
            self._close_single_line_if_finished()
            return
        if self.single_line_context and self.inline_progress_enabled:
            gate_status = "PASS" if float(local_min_score) > 0.0 else "FAIL"
            stop_text = " early_stop" if bool(early_stopped) else ""
            self._render_single_line(
                idx,
                current_neighbor=evaluated_neighbors,
                total_neighbors=total_neighbors,
                local_min_score=local_min_score,
                status_text=f"{gate_status}{stop_text}",
                done=True,
            )
            self._close_single_line_if_finished()
            return
        gate_status = "PASS" if float(local_min_score) > 0.0 else "FAIL"
        stop_text = " | early stop" if bool(early_stopped) else ""
        self.lines[idx] = self._format_line(
            idx,
            prefix="✅",
            progress_text=f"進度 {int(evaluated_neighbors)}/{int(total_neighbors)}",
            local_text=f"{float(local_min_score):.3f}",
            status_text=f"{gate_status}{stop_text}",
        )
        self._render()

    def _record_done(self, idx: int, *, local_min_score: float, early_stopped: bool):
        score = float(local_min_score)
        passed = bool(score > 0.0)
        self.completed_status[int(idx)] = (score, passed, bool(early_stopped))
        if score > self.best_local_score:
            self.best_local_score = score
            self.best_local_trial = self.finalists[idx]["trial"]

    def _estimate_single_line_eta(self, idx: int, current_neighbor: int, total_neighbors: int):
        total_finalists = max(1, len(self.finalists))
        current_units = float(len(self.completed_status))
        if int(idx) not in self.completed_status and int(total_neighbors) > 0:
            current_units = max(current_units, float(idx) + max(0.0, min(1.0, float(current_neighbor) / float(total_neighbors))))
        if current_units <= 0.0:
            return None
        elapsed = max(0.0, time.perf_counter() - self.stage_start)
        avg = elapsed / current_units
        return avg * max(0.0, float(total_finalists) - current_units)

    def _estimate_total_eta(self, eta_stage):
        ctx = self.single_line_context or {}
        completed_results = list(ctx.get("completed_results") or [])
        fold_count = int(ctx.get("fold_count", 0) or 0)
        fold_idx = int(ctx.get("fold_idx", 0) or 0)
        if not completed_results:
            return None
        avg_done = sum(float(row.get("elapsed_sec", 0.0)) for row in completed_results) / float(len(completed_results))
        return float(eta_stage or 0.0) + avg_done * max(0, fold_count - fold_idx)

    def _close_single_line_if_finished(self):
        if (
            self.inline_progress_enabled
            and not self.inline_progress_closed
            and len(self.completed_status) >= len(self.finalists)
        ):
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.inline_progress_closed = True

    def _render_single_line(self, idx: int, *, current_neighbor: int, total_neighbors: int, local_min_score, status_text: str, done: bool = False):
        if not self.inline_progress_enabled or self.inline_progress_closed or not self.finalists:
            return
        ctx = self.single_line_context or {}
        safe_idx = min(max(0, int(idx)), len(self.finalists) - 1)
        trial = self.finalists[safe_idx]["trial"]
        current_text = "N/A" if local_min_score is None else f"{float(local_min_score):.3f}"
        pass_count = sum(1 for _, passed, _ in self.completed_status.values() if passed)
        fail_count = sum(1 for _, passed, _ in self.completed_status.values() if not passed)
        early_count = sum(1 for _, _, early in self.completed_status.values() if early)
        best_text = "N/A"
        if self.best_local_trial is not None and self.best_local_score != float("-inf"):
            best_text = f"{self.best_local_score:.3f} #{int(self.best_local_trial.number) + 1}"
        best_base_score = ctx.get("best_base_score")
        best_base_text = "N/A" if best_base_score is None else f"{float(best_base_score):.3f}"
        eta_stage = self._estimate_single_line_eta(safe_idx, int(current_neighbor), int(total_neighbors))
        eta_total = self._estimate_total_eta(eta_stage)
        elapsed = time.perf_counter() - self.stage_start
        selection_start = ctx.get('selection_start', '')
        selection_end = ctx.get('selection_end', '')
        oos_value = ctx.get('oos_period') or ctx.get('oos_year', '')
        selection_text = _format_progress_range(selection_start, selection_end, compact=False)
        selection_compact = _format_progress_range(selection_start, selection_end, compact=True)
        oos_text = _format_progress_value(oos_value, compact=False)
        oos_compact = _format_progress_value(oos_value, compact=True)
        readable_status = str(status_text).strip()
        compact_status = readable_status.replace(" early_stop", " early").replace(" cache", " cache")
        best_compact = best_text.replace(" #", "#")
        elapsed_text = self._format_duration_compact(elapsed)
        eta_stage_text = self._format_duration_compact(eta_stage)
        eta_total_text = self._format_duration_compact(eta_total)
        line = choose_inline_progress_message((
            (
                f"[{int(ctx.get('fold_idx', 0) or 0)}/{int(ctx.get('fold_count', 0) or 0)}] "
                f"selection={selection_text} | OOS={oos_text} | "
                f"LOCAL_MIN_REVIEW={safe_idx + 1}/{len(self.finalists)} | trial=#{int(trial.number) + 1} | "
                f"進度={int(current_neighbor)}/{int(total_neighbors)} | "
                f"current={current_text} {readable_status} | best_base={best_base_text} | best_local_min={best_text} | "
                f"pass/fail/early={pass_count}/{fail_count}/{early_count} | "
                f"elapsed={elapsed_text} | eta={eta_stage_text}/{eta_total_text}"
            ),
            (
                f"[{int(ctx.get('fold_idx', 0) or 0)}/{int(ctx.get('fold_count', 0) or 0)}] "
                f"selection={selection_compact} | OOS={oos_compact} | "
                f"review={safe_idx + 1}/{len(self.finalists)} | trial=#{int(trial.number) + 1} | "
                f"進度={int(current_neighbor)}/{int(total_neighbors)} | "
                f"current={current_text} {readable_status} | best_base={best_base_text} | best_local_min={best_text} | "
                f"pass/fail/early={pass_count}/{fail_count}/{early_count} | "
                f"elapsed={elapsed_text} | eta={eta_stage_text}/{eta_total_text}"
            ),
            (
                f"[{int(ctx.get('fold_idx', 0) or 0)}/{int(ctx.get('fold_count', 0) or 0)}] "
                f"{selection_compact}>OOS{oos_compact} | "
                f"review {safe_idx + 1}/{len(self.finalists)} #{int(trial.number) + 1} | "
                f"{int(current_neighbor)}/{int(total_neighbors)} | "
                f"current={current_text} {compact_status} | base={best_base_text} | local={best_compact} | "
                f"pass/fail/early={pass_count}/{fail_count}/{early_count}"
            ),
        ))
        self.inline_progress_width = write_inline_progress(line, previous_width=self.inline_progress_width)

    def _render(self):
        if not stdout_supports_inline_progress():
            return
        out = sys.stdout
        if self.rendered and self.lines:
            out.write(f"\x1b[{len(self.lines)}F")
        for line in self.lines:
            out.write(f"\r{line}\x1b[K\n")
        out.flush()
        self.rendered = True




def _compute_local_retention(base_score: float, local_min_score: float) -> float:
    base = float(base_score)
    if base <= 0.0:
        return float("-inf")
    return float(local_min_score) / base


def is_dominant_year_dependency_anti_overfit_enabled() -> bool:
    return bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED)


def is_inner_validate_anti_overfit_enabled(objective_mode: str | None = None) -> bool:
    if not bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED):
        return False
    if objective_mode is None:
        return True
    return normalize_objective_mode(objective_mode) == OBJECTIVE_MODE_SPLIT_TRAIN_ROMD


def _inner_validate_score_is_positive(item: dict) -> bool:
    diagnostics = item.get("inner_validate_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    try:
        score = float(diagnostics.get("inner_validate_score", float("-inf")))
    except (TypeError, ValueError):
        return False
    return score > float(OPTIMIZER_INNER_VALIDATE_MIN_SCORE)


def _has_inner_validate_pass(item: dict) -> bool:
    if not _inner_validate_score_is_positive(item):
        return False
    diagnostics = item.get("inner_validate_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    return bool(diagnostics.get("inner_validate_rank_gate", True))


def _annotate_inner_validate_ranks(finalists: list[dict]) -> list[dict]:
    score_items = [
        item for item in finalists
        if bool(item.get("gate_pass", False))
        and isinstance(item.get("inner_validate_diagnostics"), dict)
    ]
    if not score_items:
        return finalists
    score_items.sort(
        key=lambda item: (
            float(item["inner_validate_diagnostics"].get("inner_validate_score", float("-inf"))),
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    cutoff = max(1, int(
        (Decimal(len(score_items)) * Decimal(str(OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE)))
        .to_integral_value(rounding="ROUND_CEILING")
    ))
    for rank, item in enumerate(score_items, start=1):
        diagnostics = item["inner_validate_diagnostics"]
        score_gate = _inner_validate_score_is_positive(item)
        diagnostics["inner_validate_rank"] = int(rank)
        diagnostics["inner_validate_rank_cutoff"] = int(cutoff)
        diagnostics["inner_validate_rank_total"] = int(len(score_items))
        diagnostics["inner_validate_rank_gate"] = bool(rank <= cutoff)
        diagnostics["inner_validate_gate"] = bool(score_gate and rank <= cutoff)
    return finalists


def _has_dependency_warning(item: dict) -> bool:
    diagnostics = item.get("dominant_year_dependency_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    return bool(diagnostics.get("dependency_warning", False))


def _format_dependency_reason(diagnostics: dict) -> str:
    reasons = diagnostics.get("dependency_reason") if isinstance(diagnostics, dict) else []
    if not isinstance(reasons, list) or not reasons:
        return "-"
    label_map = {
        "dominant_year_positive_pnl_share_gte_high": "year",
        "dominant_year_positive_trade_count_lte_narrow": "trades",
        "dominant_year_positive_symbol_count_lte_narrow": "symbols",
        "top_trade_pnl_share_in_dominant_year_gte_outlier": "top_trade",
    }
    labels = [label_map.get(str(reason), str(reason)) for reason in reasons]
    return "+".join(labels)


def _sort_finalists_by_local_min(finalists: list[dict]) -> list[dict]:
    return sorted(
        finalists,
        key=lambda item: (
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("local_retention", float("-inf"))),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )


def _select_best_finalist_by_local_min_score(finalists: list[dict], *, use_inner_validate: bool | None = None):
    eligible = [item for item in finalists if bool(item.get("gate_pass", False))]
    if not eligible:
        return None
    if use_inner_validate is None:
        use_inner_validate = any(
            isinstance(item.get("inner_validate_diagnostics"), dict)
            and bool(item["inner_validate_diagnostics"].get("enabled", False))
            for item in finalists
        )
    if bool(use_inner_validate):
        eligible = [item for item in eligible if _has_inner_validate_pass(item)]
        if not eligible:
            return None
    if is_dominant_year_dependency_anti_overfit_enabled():
        safe_eligible = [item for item in eligible if not _has_dependency_warning(item)]
        if safe_eligible:
            return _sort_finalists_by_local_min(safe_eligible)[0]
    return _sort_finalists_by_local_min(eligible)[0]


def select_best_finalist_by_local_retention(finalists: list[dict]):
    eligible = [item for item in finalists if bool(item.get("gate_pass", False))]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            float(item.get("local_retention", float("-inf"))),
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return eligible[0]


def select_best_finalist_by_inner_validate_score(finalists: list[dict]):
    eligible = []
    for item in finalists:
        if not bool(item.get("gate_pass", False)):
            continue
        diagnostics = item.get("inner_validate_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        try:
            validate_score = float(diagnostics.get("inner_validate_score", float("-inf")))
        except (TypeError, ValueError):
            continue
        eligible.append((validate_score, item))
    if not eligible:
        return None
    eligible.sort(
        key=lambda pair: (
            pair[0],
            float(pair[1].get("local_min_score", INVALID_TRIAL_VALUE)),
            float(pair[1].get("local_retention", float("-inf"))),
            float(pair[1].get("base_score", INVALID_TRIAL_VALUE)),
            -int(pair[1]["trial"].number),
        ),
        reverse=True,
    )
    return eligible[0][1]


def _trial_matches_objective_mode(trial, objective_mode: str) -> bool:
    expected = normalize_objective_mode(objective_mode)
    actual = str(trial.user_attrs.get("objective_mode", "")).strip()
    if actual:
        return objective_modes_are_compatible(actual, expected)
    return expected == normalize_objective_mode("")


def _resolve_neighbor_step(session, field_name: str, *, center_payload=None):
    if field_name == "tp_percent":
        spec = OPTIMIZER_TP_PERCENT_SEARCH_SPEC
        return float(spec["step"]), "float", float(spec["low"]), float(spec["high"])

    return resolve_breakout_neighbor_spec(field_name, center_payload=center_payload)


def _apply_step(field_name: str, current_value, step_value, direction: int, kind: str):
    if kind == "int":
        return int(current_value) + int(direction) * int(step_value)

    candidate = Decimal(str(current_value)) + (Decimal(str(step_value)) * Decimal(direction))
    return float(candidate)


def _build_neighbor_candidates(session, trial):
    center_payload = build_best_params_payload_from_trial(trial, fixed_tp_percent=session.optimizer_fixed_tp_percent)
    candidate_fields = get_breakout_local_min_candidate_fields(trial, center_payload=center_payload)

    neighbors = []
    seen_payload_keys = set()
    for field_name in candidate_fields:
        step_value, kind, low_value, high_value = _resolve_neighbor_step(session, field_name, center_payload=center_payload)
        current_value = center_payload[field_name]
        for direction in (-1, 1):
            candidate_value = _apply_step(field_name, current_value, step_value, direction, kind)
            if candidate_value < low_value or candidate_value > high_value:
                continue
            candidate_payload = dict(center_payload)
            candidate_payload[field_name] = candidate_value
            try:
                build_params_from_mapping(candidate_payload)
            except ValueError:
                continue
            payload_key = tuple(sorted(candidate_payload.items()))
            if payload_key in seen_payload_keys:
                continue
            seen_payload_keys.add(payload_key)
            neighbors.append(candidate_payload)
    return neighbors


def compute_local_min_score(
    session,
    trial,
    *,
    progress_label: str | None = None,
    show_cache_hit: bool = False,
    on_cache_hit=None,
    on_start=None,
    on_neighbor=None,
    on_finish=None,
    stop_below_local_min_score: float | None = None,
):
    if not bool(is_optimizer_local_min_review_enabled()):
        score = float(getattr(trial, "user_attrs", {}).get("base_score", getattr(trial, "value", INVALID_TRIAL_VALUE)))
        if on_finish is not None:
            on_finish(0, 0, float(score), False)
        elif progress_label:
            _print_progress_line(session, f"ℹ️ {progress_label}: local_min review disabled，使用 base_score={score:.3f}")
        return float(score)

    cache = _get_local_min_score_cache(session)
    payload_score_cache = _get_local_min_payload_score_cache(session)
    cache_key = int(trial.number)
    cached = cache.get(cache_key)
    if cached is not None:
        if isinstance(cached, dict):
            cached_score = float(cached.get("score", INVALID_TRIAL_VALUE))
            cached_total_neighbors = int(cached.get("total_neighbors", 0))
        else:
            cached_score = float(cached)
            cached_total_neighbors = 0
        if on_cache_hit is not None:
            on_cache_hit(float(cached_score), int(cached_total_neighbors))
        elif progress_label and show_cache_hit:
            _print_progress_line(session, f"ℹ️ {progress_label}: 使用快取 local_min_score={cached_score:.3f}")
        return cached_score

    neighbor_payloads = _build_neighbor_candidates(session, trial)
    if not neighbor_payloads:
        local_min_score = float(INVALID_TRIAL_VALUE)
        cache[cache_key] = {"score": float(local_min_score), "total_neighbors": 0, "evaluated_neighbors": 0, "early_stopped": False}
        if on_finish is not None:
            on_finish(0, 0, float(local_min_score), False)
        elif progress_label:
            _print_progress_line(session, f"❌ {progress_label}: 無合法鄰點，local_min_score={float(local_min_score):.3f} | gate=FAIL")
        return float(local_min_score)

    total_neighbors = len(neighbor_payloads)
    center_payload = build_best_params_payload_from_trial(trial, fixed_tp_percent=session.optimizer_fixed_tp_percent)
    neighbor_payloads, neighbor_rank_stats = _rank_local_min_neighbor_payloads(session, center_payload, neighbor_payloads, payload_score_cache)
    if on_start is not None:
        on_start(total_neighbors)
    elif progress_label:
        _print_progress_line(session, f"⏳ {progress_label}: 開始 local_min_score 分析，共 {total_neighbors} 個鄰點")

    try:
        selection_prune_floor = float(stop_below_local_min_score) if stop_below_local_min_score is not None else None
    except (TypeError, ValueError):
        selection_prune_floor = None

    evaluation_result = _evaluate_local_min_neighbors_ordered(
        session,
        center_payload=center_payload,
        neighbor_payloads=neighbor_payloads,
        payload_score_cache=payload_score_cache,
        total_neighbors=total_neighbors,
        selection_prune_floor=selection_prune_floor,
        on_neighbor=on_neighbor,
    )
    local_min_score = float(evaluation_result.get("local_min_score", float("inf")))
    evaluated_neighbors = int(evaluation_result.get("evaluated_neighbors", 0) or 0)
    payload_score_cache_hit_count = int(evaluation_result.get("payload_score_cache_hits", 0) or 0)
    early_stopped = bool(evaluation_result.get("early_stopped", False))
    selection_pruned = bool(evaluation_result.get("selection_pruned", False))
    hard_fail_stopped = bool(evaluation_result.get("hard_fail_stopped", False))
    hard_fail_neighbors_skipped = int(evaluation_result.get("hard_fail_neighbors_skipped", 0) or 0)
    hard_fail_cancelled = int(evaluation_result.get("hard_fail_cancelled", 0) or 0)

    if local_min_score == float("inf"):
        local_min_score = float(INVALID_TRIAL_VALUE)
    if not selection_pruned:
        cache[cache_key] = {
            "score": float(local_min_score),
            "total_neighbors": total_neighbors,
            "evaluated_neighbors": evaluated_neighbors,
            "early_stopped": bool(early_stopped),
        }
    record_local_min_stats = getattr(session, "record_local_min_review_stats", None)
    if callable(record_local_min_stats):
        record_local_min_stats(
            total_neighbors=total_neighbors,
            evaluated_neighbors=evaluated_neighbors,
            payload_score_cache_hits=payload_score_cache_hit_count,
            prep_cache_prioritized=int(neighbor_rank_stats.get("prep_cache_prioritized", 0) or 0),
            order_score_prioritized=int(neighbor_rank_stats.get("order_score_prioritized", 0) or 0),
            field_order_score_prioritized=int(neighbor_rank_stats.get("field_order_score_prioritized", 0) or 0),
            portfolio_dependency_deprioritized=int(neighbor_rank_stats.get("portfolio_dependency_deprioritized", 0) or 0),
            signal_dependency_field_prioritized=int(neighbor_rank_stats.get("signal_dependency_field_prioritized", 0) or 0),
            early_stopped=bool(early_stopped),
            selection_pruned=bool(selection_pruned),
            parallel_workers=int(evaluation_result.get("parallel_workers", 1) or 1),
            parallel_submitted=int(evaluation_result.get("parallel_submitted", evaluated_neighbors) or 0),
            parallel_completed=int(evaluation_result.get("parallel_completed", evaluated_neighbors) or 0),
            parallel_cancelled=int(evaluation_result.get("parallel_cancelled", 0) or 0),
            hard_fail_stopped=bool(hard_fail_stopped),
            hard_fail_neighbors_skipped=int(hard_fail_neighbors_skipped),
            hard_fail_cancelled=int(hard_fail_cancelled),
            dependency_signal_total=int(evaluation_result.get("dependency_signal_total", 0) or 0),
            dependency_signal_evaluated=int(evaluation_result.get("dependency_signal_evaluated", 0) or 0),
            dependency_portfolio_total=int(evaluation_result.get("dependency_portfolio_total", 0) or 0),
            dependency_portfolio_evaluated=int(evaluation_result.get("dependency_portfolio_evaluated", 0) or 0),
            dependency_mixed_total=int(evaluation_result.get("dependency_mixed_total", 0) or 0),
            dependency_mixed_evaluated=int(evaluation_result.get("dependency_mixed_evaluated", 0) or 0),
            dependency_unknown_total=int(evaluation_result.get("dependency_unknown_total", 0) or 0),
            dependency_unknown_evaluated=int(evaluation_result.get("dependency_unknown_evaluated", 0) or 0),
            signal_reuse_candidate_total=int(evaluation_result.get("signal_reuse_candidate_total", 0) or 0),
            signal_reuse_candidate_evaluated=int(evaluation_result.get("signal_reuse_candidate_evaluated", 0) or 0),
            signal_recompute_required_total=int(evaluation_result.get("signal_recompute_required_total", 0) or 0),
            signal_recompute_required_evaluated=int(evaluation_result.get("signal_recompute_required_evaluated", 0) or 0),
            dependency_field_total_counts=dict(evaluation_result.get("dependency_field_total_counts", {}) or {}),
            dependency_field_evaluated_counts=dict(evaluation_result.get("dependency_field_evaluated_counts", {}) or {}),
        )
    if on_finish is not None:
        on_finish(evaluated_neighbors, total_neighbors, float(local_min_score), bool(early_stopped))
    elif progress_label:
        gate_status = "PASS" if float(local_min_score) > 0.0 else "FAIL"
        stop_text = " | early stop" if bool(early_stopped) else ""
        _print_progress_line(session, f"✅ {progress_label}: local_min_score={float(local_min_score):.3f} | gate={gate_status}{stop_text}")
    return float(local_min_score)



_CURRENT_DEPENDENCY_DIAGNOSTIC_REQUIRED_KEYS = frozenset({
    "dependency_warning",
    "dependency_status",
    "dependency_reason",
    "positive_total_pnl",
    "dominant_entry_year",
    "dominant_year_positive_pnl",
    "dominant_year_positive_pnl_share",
    "dominant_year_positive_trade_count",
    "dominant_year_positive_symbol_count",
    "top_trade_pnl_share_in_dominant_year",
})

_CURRENT_DEPENDENCY_DIAGNOSTIC_OUTPUT_KEYS = (
    "dependency_warning",
    "dependency_status",
    "dependency_reason",
    "positive_total_pnl",
    "unassigned_positive_pnl",
    "dominant_entry_year",
    "dominant_year_positive_pnl",
    "dominant_year_positive_pnl_share",
    "dominant_year_positive_trade_count",
    "dominant_year_positive_symbol_count",
    "top_trade_pnl_share_in_dominant_year",
    "thresholds",
)


def _normalize_current_dependency_diagnostics(value):
    if not isinstance(value, dict) or not _CURRENT_DEPENDENCY_DIAGNOSTIC_REQUIRED_KEYS.issubset(value.keys()):
        return None
    return {key: value.get(key) for key in _CURRENT_DEPENDENCY_DIAGNOSTIC_OUTPUT_KEYS if key in value}


def _resolve_trial_dependency_diagnostics(session, trial, objective_mode: str):
    existing = _normalize_current_dependency_diagnostics(trial.user_attrs.get("dominant_year_dependency_diagnostics"))
    if existing is not None:
        return existing

    payload = build_best_params_payload_from_trial(
        trial,
        fixed_tp_percent=session.optimizer_fixed_tp_percent,
    )
    cache = _get_dominant_year_dependency_cache(session)
    cache_key = _build_payload_score_cache_key(payload)
    cached = _normalize_current_dependency_diagnostics(cache.get(cache_key))
    if cached is not None:
        return cached

    ai_params = build_params_from_mapping(payload)
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
        profile_enabled=False,
    )
    search_scope = resolve_search_train_scope(session, prep_result["master_dates"], objective_mode=objective_mode)
    evaluation = evaluate_prepared_train_score(
        session,
        ai_params=ai_params,
        prep_result=prep_result,
        search_scope=search_scope,
        profile_stats={"_timing_enabled": False},
    )
    diagnostics = _normalize_current_dependency_diagnostics(evaluation.get("dominant_year_dependency_diagnostics", {}))
    if diagnostics is None:
        diagnostics = {}
    cache[cache_key] = diagnostics
    return diagnostics


def _normalize_inner_validate_diagnostics(value):
    if not isinstance(value, dict) or not bool(value.get("enabled", False)):
        return None
    if "inner_validate_score" not in value:
        return None
    output_keys = (
        "enabled",
        "inner_validate_score",
        "inner_validate_gate",
        "inner_validate_rank",
        "inner_validate_rank_cutoff",
        "inner_validate_rank_total",
        "inner_validate_rank_gate",
        "validate_year",
        "validate_start_year",
        "validate_end_year",
        "inner_train_end_year",
        "ret_pct",
        "mdd",
        "trade_count",
        "annual_return_pct",
        "monthly_win_rate",
        "r_squared",
        "normal_trades",
        "extended_trades",
        "reserved_buy_fill_rate",
        "fail_reason",
    )
    return {key: value.get(key) for key in output_keys if key in value}


def _resolve_trial_inner_validate_diagnostics(session, trial, objective_mode: str):
    payload = build_best_params_payload_from_trial(
        trial,
        fixed_tp_percent=session.optimizer_fixed_tp_percent,
    )
    cache = _get_inner_validate_cache(session)
    cache_key = _build_payload_score_cache_key(payload)
    cached = _normalize_inner_validate_diagnostics(cache.get(cache_key))
    if cached is not None:
        return cached

    ai_params = build_params_from_mapping(payload)
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
        profile_enabled=False,
    )
    validate_scope = resolve_inner_validate_scope(session, prep_result["master_dates"], objective_mode=objective_mode)
    diagnostics = evaluate_prepared_inner_validate_score(
        session,
        ai_params=ai_params,
        prep_result=prep_result,
        validate_scope=validate_scope,
        profile_stats={"_timing_enabled": False},
    )
    diagnostics = _normalize_inner_validate_diagnostics(diagnostics) or {
        "enabled": True,
        "inner_validate_score": float(INVALID_TRIAL_VALUE),
        "inner_validate_gate": False,
        "fail_reason": "inner validation diagnostics 無法計算",
    }
    cache[cache_key] = diagnostics
    return diagnostics


def _normalize_oos_diagnostics(value):
    if not isinstance(value, dict) or not bool(value.get('enabled', False)):
        return None
    return {
        'enabled': True,
        'oos_score': float(value.get('oos_score', INVALID_TRIAL_VALUE)),
        'oos_start': str(value.get('oos_start') or ''),
        'oos_end': str(value.get('oos_end') or ''),
        'oos_start_year': value.get('oos_start_year'),
        'period_count': int(value.get('period_count', 0) or 0),
    }


def _resolve_trial_oos_diagnostics(session, trial):
    oos_start_year = getattr(session, 'walk_forward_policy', {}).get('oos_start_year')
    if oos_start_year is None:
        return {'enabled': False, 'oos_score': float(INVALID_TRIAL_VALUE)}

    payload = build_best_params_payload_from_trial(
        trial,
        fixed_tp_percent=session.optimizer_fixed_tp_percent,
    )
    cache = _get_oos_cache(session)
    cache_key = _build_payload_score_cache_key(payload)
    cached = _normalize_oos_diagnostics(cache.get(cache_key))
    if cached is not None:
        return cached

    ai_params = build_params_from_mapping(payload)
    prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(ai_params, 'optimizer_max_workers'))
    prep_result = prepare_trial_inputs(
        raw_data_cache=session.raw_data_cache,
        params=ai_params,
        default_max_workers=session.default_max_workers,
        executor_bundle=prep_executor_bundle,
        static_fast_cache=session.static_fast_cache,
        static_master_dates=session.master_dates,
        include_trade_logs=True,
        include_pit_stats_index=True,
        profile_enabled=False,
    )
    report = evaluate_walk_forward(
        all_dfs_fast=prep_result['all_dfs_fast'],
        all_trade_logs=prep_result['all_trade_logs'],
        sorted_dates=sorted(prep_result['master_dates']),
        params=ai_params,
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
        min_train_years=int(getattr(session, 'walk_forward_policy', {}).get('min_train_years', 1) or 1),
        train_start_year=getattr(session, 'train_start_year', None),
        oos_start_year=int(oos_start_year),
        train_start_date=(getattr(session, 'walk_forward_policy', {}) or {}).get('train_start_date'),
        oos_start_date=(getattr(session, 'walk_forward_policy', {}) or {}).get('oos_start_date'),
        oos_end_date=(getattr(session, 'walk_forward_policy', {}) or {}).get('oos_end_date'),
        pit_stats_index=prep_result.get('all_pit_stats_index'),
    )
    summary = dict(report.get('summary') or {})
    diagnostics = {
        'enabled': True,
        'oos_score': float(summary.get('test_score_romd', INVALID_TRIAL_VALUE)),
        'oos_start': str(summary.get('oos_start') or ''),
        'oos_end': str(summary.get('oos_end') or ''),
        'oos_start_year': int(oos_start_year),
        'period_count': int(summary.get('period_count', 0) or 0),
    }
    cache[cache_key] = diagnostics
    return diagnostics


def _list_qualified_trials_for_objective(study, objective_mode: str):
    qualified_trials = [
        trial
        for trial in list_completed_study_trials(study)
        if is_qualified_trial_value(trial.value) and _trial_matches_objective_mode(trial, objective_mode)
    ]
    return sorted(qualified_trials, key=lambda trial: (float(trial.value), -int(trial.number)), reverse=True)


def _build_display_finalists(sorted_trials, *, top_k: int, include_trial=None):
    selected_trials = list(sorted_trials[: max(1, int(top_k))])
    if include_trial is not None and all(int(trial.number) != int(include_trial.number) for trial in selected_trials):
        if len(selected_trials) >= max(1, int(top_k)):
            selected_trials = list(selected_trials[: max(0, int(top_k) - 1)]) + [include_trial]
        else:
            selected_trials.append(include_trial)
    base_rank_map = {int(trial.number): idx for idx, trial in enumerate(sorted_trials, start=1)}
    return [
        {
            "trial": trial,
            "base_rank": int(base_rank_map.get(int(trial.number), 0)),
            "base_score": float(trial.user_attrs.get("base_score", trial.value)),
        }
        for trial in selected_trials
    ]


def _build_local_min_disabled_finalist_item(item: dict) -> dict:
    trial = item["trial"]
    base_score = float(item["base_score"])
    return {
        "trial": trial,
        "base_rank": int(item.get("base_rank", 0)),
        "base_score": base_score,
        "local_min_score": base_score,
        "local_retention": _compute_local_retention(base_score, base_score),
        "gate_pass": bool(base_score > 0.0),
        "local_min_review_enabled": False,
        "local_min_review_mode": "disabled_base_score_equivalent",
        "local_min_exact": False,
    }


def _build_local_min_disabled_finalists(
    finalists: list[dict],
    *,
    session,
    objective_mode: str,
    include_oos_diagnostics: bool,
) -> list[dict]:
    enriched_finalists: list[dict] = []
    for item in finalists:
        enriched_item = _build_local_min_disabled_finalist_item(item)
        trial = enriched_item["trial"]
        if is_inner_validate_anti_overfit_enabled(objective_mode):
            enriched_item["inner_validate_diagnostics"] = _resolve_trial_inner_validate_diagnostics(
                session,
                trial,
                objective_mode,
            )
        if is_dominant_year_dependency_anti_overfit_enabled():
            enriched_item["dominant_year_dependency_diagnostics"] = _resolve_trial_dependency_diagnostics(
                session,
                trial,
                objective_mode,
            )
        if bool(include_oos_diagnostics):
            enriched_item["oos_diagnostics"] = _resolve_trial_oos_diagnostics(session, trial)
        enriched_finalists.append(enriched_item)
    if is_inner_validate_anti_overfit_enabled(objective_mode):
        _annotate_inner_validate_ranks(enriched_finalists)
    enriched_finalists.sort(
        key=lambda item: (
            float(item["local_min_score"]),
            float(item["local_retention"]),
            float(item["base_score"]),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return enriched_finalists


def _resolve_local_min_score_finalist_top_k(session, top_k=None):
    if top_k is not None:
        return max(1, int(top_k))
    return resolve_optimizer_local_min_score_finalist_top_k(getattr(session, "n_trials", 0))


def list_local_min_score_finalists(
    study,
    *,
    session,
    objective_mode: str,
    top_k=None,
    include_trial=None,
    show_progress: bool = False,
    include_oos_diagnostics: bool = True,
    single_finalist_fast_path: bool = False,
    selection_pruning: bool = False,
):
    resolved_top_k = _resolve_local_min_score_finalist_top_k(session, top_k)
    sorted_trials = _list_qualified_trials_for_objective(study, objective_mode)
    if not sorted_trials:
        return []

    _seed_payload_score_cache_from_study(session, study, objective_mode)
    finalists = _build_display_finalists(sorted_trials, top_k=resolved_top_k, include_trial=include_trial)
    if not bool(is_optimizer_local_min_review_enabled()):
        return _build_local_min_disabled_finalists(
            finalists,
            session=session,
            objective_mode=objective_mode,
            include_oos_diagnostics=include_oos_diagnostics,
        )
    gates_require_exact_review = bool(is_inner_validate_anti_overfit_enabled(objective_mode) or is_dominant_year_dependency_anti_overfit_enabled())
    if (
        bool(single_finalist_fast_path)
        and include_trial is None
        and not bool(include_oos_diagnostics)
        and not gates_require_exact_review
        and len(finalists) == 1
    ):
        item = finalists[0]
        base_score = float(item["base_score"])
        return [{
            "trial": item["trial"],
            "base_rank": int(item.get("base_rank", 0)),
            "base_score": base_score,
            "local_min_score": base_score,
            "local_retention": _compute_local_retention(base_score, base_score),
            "gate_pass": bool(base_score > 0.0),
            "local_min_review_enabled": True,
            "local_min_review_mode": "single_finalist_selection_equivalent_fast_path",
            "local_min_exact": False,
        }]
    progress_board = None
    if show_progress:
        progress_board = _FinalistProgressBoard(session, finalists)
        progress_board.initialize()

    enriched_finalists = []
    best_selection_local_min = float("-inf")
    best_selection_retention = float("-inf")
    allow_selection_pruning = bool(selection_pruning) and not gates_require_exact_review
    for finalist_idx, item in enumerate(finalists):
        trial = item["trial"]
        base_score = float(item["base_score"])
        selection_prune_floor = None
        if allow_selection_pruning and best_selection_local_min != float("-inf") and best_selection_retention != float("-inf") and base_score > 0.0:
            retention_floor = float(best_selection_retention) * float(base_score)
            selection_prune_floor = min(float(best_selection_local_min), float(retention_floor))
        if progress_board is not None:
            local_min_score = compute_local_min_score(
                session,
                trial,
                on_cache_hit=lambda score, total, idx=finalist_idx: progress_board.update_cache(idx, total_neighbors=total, local_min_score=score),
                on_start=lambda total, idx=finalist_idx: progress_board.update_pending(idx, total_neighbors=total),
                on_neighbor=lambda current, total, current_local_min, idx=finalist_idx: progress_board.update_neighbor(
                    idx,
                    current_neighbor=current,
                    total_neighbors=total,
                    current_local_min=current_local_min,
                ),
                on_finish=lambda evaluated, total, score, early_stopped, idx=finalist_idx: progress_board.update_done(
                    idx,
                    evaluated_neighbors=evaluated,
                    total_neighbors=total,
                    local_min_score=score,
                    early_stopped=early_stopped,
                ),
                stop_below_local_min_score=selection_prune_floor,
            )
        else:
            local_min_score = compute_local_min_score(
                session,
                trial,
                stop_below_local_min_score=selection_prune_floor,
            )
        enriched_item = {
            "trial": trial,
            "base_rank": int(item.get("base_rank", 0)),
            "base_score": base_score,
            "local_min_score": float(local_min_score),
            "local_retention": _compute_local_retention(base_score, float(local_min_score)),
            "gate_pass": bool(local_min_score > 0.0),
            "local_min_review_enabled": True,
            "local_min_review_mode": "exact",
            "local_min_exact": True,
        }
        if is_inner_validate_anti_overfit_enabled(objective_mode):
            enriched_item["inner_validate_diagnostics"] = _resolve_trial_inner_validate_diagnostics(
                session,
                trial,
                objective_mode,
            )
        if is_dominant_year_dependency_anti_overfit_enabled():
            enriched_item["dominant_year_dependency_diagnostics"] = _resolve_trial_dependency_diagnostics(
                session,
                trial,
                objective_mode,
            )
        if bool(include_oos_diagnostics):
            enriched_item["oos_diagnostics"] = _resolve_trial_oos_diagnostics(session, trial)
        enriched_finalists.append(enriched_item)
        if bool(enriched_item.get("gate_pass", False)):
            best_selection_local_min = max(best_selection_local_min, float(enriched_item["local_min_score"]))
            best_selection_retention = max(best_selection_retention, float(enriched_item["local_retention"]))
    if is_inner_validate_anti_overfit_enabled(objective_mode):
        _annotate_inner_validate_ranks(enriched_finalists)
    enriched_finalists.sort(
        key=lambda item: (
            float(item["local_min_score"]),
            float(item["local_retention"]),
            float(item["base_score"]),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return enriched_finalists


def print_local_min_score_finalist_review(study, *, session, objective_mode: str, colors: dict, winner_trial=None, top_k=None, emit_table: bool = True, show_progress: bool | None = None):
    progress_enabled = bool(emit_table) if show_progress is None else bool(show_progress)
    finalists = list_local_min_score_finalists(
        study,
        session=session,
        objective_mode=objective_mode,
        top_k=top_k,
        include_trial=winner_trial,
        show_progress=progress_enabled,
        include_oos_diagnostics=True,
    )
    if not finalists:
        return [], winner_trial
    if winner_trial is None:
        best_finalist = _select_best_finalist_by_local_min_score(
            finalists,
            use_inner_validate=is_inner_validate_anti_overfit_enabled(objective_mode),
        )
        winner_trial = None if best_finalist is None else best_finalist["trial"]
    if not bool(emit_table):
        return finalists, winner_trial
    if not bool(is_optimizer_local_min_review_enabled()):
        _print_progress_line(session, "ℹ️ local_min review disabled：local_min_score 使用 base_score 等價值，retention=1.0")

    gray = colors.get("gray", "")
    green = colors.get("green", "")
    red = colors.get("red", "")
    yellow = colors.get("yellow", "")
    reset = colors.get("reset", "")

    inner_validate_enabled = is_inner_validate_anti_overfit_enabled(objective_mode)
    dependency_enabled = is_dominant_year_dependency_anti_overfit_enabled()
    periods = _resolve_report_periods(session, objective_mode=objective_mode)
    gate_rows = _build_gate_status_rows(finalists, objective_mode=objective_mode)

    print(f"{gray}{'-' * 108}{reset}")
    period_gate_pairs = [
        ('training period', periods['training_period'], gate_rows[0]),
        ('validate period', periods['validate_period'], gate_rows[1]),
        ('oos test period', periods['oos_test_period'], gate_rows[2]),
    ]
    for period_label, period_value, gate_row in period_gate_pairs:
        enabled_value = bool(gate_row.get('enabled', False))
        enabled_text = 'True' if enabled_value else 'False'
        enabled_color = green if enabled_value else red
        print(
            f"{period_label:<18}: {period_value:<18}"
            f"{str(gate_row.get('name', 'gate')):<28}: {enabled_color}{enabled_text}{reset}"
        )
    print(f"{gray}{'-' * 108}{reset}")

    header = (
        f"{'trial':<8} | "
        f"{'base_score':>12} {'base_rank':>10} | "
        f"{'local_min':>12} {'local_rank':>12} {'local_gate':>12}"
    )
    if inner_validate_enabled:
        header += f" | {'val_score':>10} {'val_rank':>10} {'val_gate':>10}"
    if dependency_enabled:
        header += f" | {'dep_gate':>10} {'dep_reason':>18}"
    header += f" | {'result':>10} {'oos_score':>10}"
    separator_width = len(_strip_ansi(header))
    print(header)
    print(f"{gray}{'-' * separator_width}{reset}")

    for local_rank, item in enumerate(finalists, start=1):
        trial = item['trial']
        gate_pass = bool(item.get('gate_pass'))
        is_winner = winner_trial is not None and int(winner_trial.number) == int(trial.number)
        has_inner_validate_fail = inner_validate_enabled and not _has_inner_validate_pass(item)
        has_dependency_warning = dependency_enabled and _has_dependency_warning(item)
        if is_winner:
            result_text = 'winner'
            result_color = yellow
        elif not gate_pass:
            result_text = 'reject'
            result_color = red
        elif has_inner_validate_fail:
            result_text = 'val_skip'
            result_color = red
        elif has_dependency_warning:
            result_text = 'dep_skip'
            result_color = red
        else:
            result_text = 'keep'
            result_color = green

        local_gate_text = 'PASS' if gate_pass else 'FAIL'
        local_gate_color = green if gate_pass else red
        base_rank_text = f"#{int(item.get('base_rank', 0))}"
        local_rank_text = f"#{int(local_rank)}"
        line = (
            f"#{int(trial.number) + 1:<7} | "
            f"{float(item['base_score']):>12.3f} {base_rank_text:>10} | "
            f"{float(item['local_min_score']):>12.3f} {local_rank_text:>12} "
            f"{local_gate_color}{local_gate_text:>12}{reset}"
        )
        if inner_validate_enabled:
            val_diag = item.get('inner_validate_diagnostics') if isinstance(item.get('inner_validate_diagnostics'), dict) else {}
            try:
                validate_score = float(val_diag.get('inner_validate_score', INVALID_TRIAL_VALUE))
            except (TypeError, ValueError):
                validate_score = float(INVALID_TRIAL_VALUE)
            rank_value = val_diag.get('inner_validate_rank')
            rank_cutoff = val_diag.get('inner_validate_rank_cutoff')
            if rank_value is None or rank_cutoff is None:
                validate_rank_text = 'N/A'
            else:
                validate_rank_text = f"#{int(rank_value)}/{int(rank_cutoff)}"
            validate_gate = _has_inner_validate_pass(item)
            validate_text = 'PASS' if validate_gate else 'FAIL'
            validate_color = green if validate_gate else red
            line += (
                f" | {validate_score:>10.3f} {validate_rank_text:>10} "
                f"{validate_color}{validate_text:>10}{reset}"
            )
        if dependency_enabled:
            diagnostics = item.get('dominant_year_dependency_diagnostics') if isinstance(item.get('dominant_year_dependency_diagnostics'), dict) else {}
            dependency_pass = not bool(diagnostics.get('dependency_warning', False))
            dep_text = 'PASS' if dependency_pass else 'FAIL'
            dep_color = green if dependency_pass else red
            dep_reason_text = _format_dependency_reason(diagnostics)
            line += f" | {dep_color}{dep_text:>10}{reset} {dep_reason_text:>18}"
        oos_diag = item.get('oos_diagnostics') if isinstance(item.get('oos_diagnostics'), dict) else {}
        try:
            oos_score = float(oos_diag.get('oos_score', INVALID_TRIAL_VALUE))
        except (TypeError, ValueError):
            oos_score = float(INVALID_TRIAL_VALUE)
        oos_score_text = 'N/A' if not bool(oos_diag.get('enabled', False)) else f"{oos_score:.3f}"
        line += f" | {result_color}{result_text:>10}{reset} {oos_score_text:>10}"
        print(line)

    print(f"{gray}{'=' * separator_width}{reset}")
    winner_count = 1 if winner_trial is not None else 0
    keep_count = 0
    reject_count = 0
    for item in finalists:
        item_trial = item['trial']
        if winner_trial is not None and int(item_trial.number) == int(winner_trial.number):
            continue
        gate_pass = bool(item.get('gate_pass'))
        has_inner_validate_fail = inner_validate_enabled and not _has_inner_validate_pass(item)
        has_dependency_warning = dependency_enabled and _has_dependency_warning(item)
        if gate_pass and not has_inner_validate_fail and not has_dependency_warning:
            keep_count += 1
        else:
            reject_count += 1
    best_local_item = finalists[0] if finalists else None
    best_local_text = '-'
    if best_local_item is not None:
        best_local_text = f"{float(best_local_item.get('local_min_score', INVALID_TRIAL_VALUE)):.3f} (trial #{int(best_local_item['trial'].number) + 1})"
    best_val_text = '-'
    if inner_validate_enabled:
        val_best = select_best_finalist_by_inner_validate_score(finalists)
        if val_best is not None:
            val_diag = val_best.get('inner_validate_diagnostics') if isinstance(val_best.get('inner_validate_diagnostics'), dict) else {}
            best_val_text = f"{float(val_diag.get('inner_validate_score', INVALID_TRIAL_VALUE)):.3f} (trial #{int(val_best['trial'].number) + 1})"
    best_oos_item = None
    oos_enabled_items = [item for item in finalists if bool((item.get('oos_diagnostics') or {}).get('enabled', False))]
    if oos_enabled_items:
        best_oos_item = max(
            oos_enabled_items,
            key=lambda item: (
                float((item.get('oos_diagnostics') or {}).get('oos_score', INVALID_TRIAL_VALUE)),
                float(item.get('local_min_score', INVALID_TRIAL_VALUE)),
                -int(item['trial'].number),
            ),
        )
    best_oos_text = '-'
    if best_oos_item is not None:
        best_oos_text = f"{float((best_oos_item.get('oos_diagnostics') or {}).get('oos_score', INVALID_TRIAL_VALUE)):.3f} (trial #{int(best_oos_item['trial'].number) + 1})"
    print(
        "summary: "
        f"{yellow}winner={winner_count}{reset}  "
        f"{green}keep={keep_count}{reset}  "
        f"{red}reject={reject_count}{reset}"
        f" | best_local_min={best_local_text}"
        f" | best_val_score={best_val_text}"
        f" | best_oos_score={best_oos_text}"
    )
    return finalists, winner_trial


def print_local_min_score_winner_summary(*, winner_trial, session, colors: dict):
    return None


def resolve_best_completed_trial_with_local_min_score_or_none(study, *, session, objective_mode: str, show_progress: bool = True, top_k=None):
    resolved_top_k = _resolve_local_min_score_finalist_top_k(session, top_k)
    resolver_cache = _get_best_trial_resolver_cache(session)
    cache_key = _build_best_trial_cache_key(study, objective_mode, resolved_top_k)
    cached_trial = resolver_cache.get(cache_key)
    if cached_trial is not None:
        return cached_trial

    finalists = list_local_min_score_finalists(
        study,
        session=session,
        objective_mode=objective_mode,
        top_k=resolved_top_k,
        include_trial=None,
        show_progress=show_progress,
    )
    best_finalist = _select_best_finalist_by_local_min_score(
        finalists,
        use_inner_validate=is_inner_validate_anti_overfit_enabled(objective_mode),
    )
    if best_finalist is None:
        if show_progress:
            _print_progress_line(session, "ℹ️ 沒有任何 finalist 通過 local_min / inner validation / dependency gate")
        resolver_cache[cache_key] = None
        return None
    trial = best_finalist["trial"]
    if show_progress:
        _print_progress_line(
            session,
            f"🏁 winner({'local_min' if is_optimizer_local_min_review_enabled() else 'base_equivalent'}{' + inner_val_rank' if is_inner_validate_anti_overfit_enabled(objective_mode) else ''}{' + dependency_safe' if is_dominant_year_dependency_anti_overfit_enabled() else ''}): trial #{int(trial.number) + 1} | base_score={float(best_finalist['base_score']):.3f} | local_min_score={float(best_finalist['local_min_score']):.3f} | retention={float(best_finalist['local_retention']):.3f}"
        )
    resolver_cache[cache_key] = trial
    return trial


def build_local_min_score_best_trial_resolver(*, session, objective_mode: str) -> Callable:
    def _resolver(study):
        return resolve_best_completed_trial_with_local_min_score_or_none(
            study,
            session=session,
            objective_mode=objective_mode,
            show_progress=False,
            top_k=None,
        )

    return _resolver
