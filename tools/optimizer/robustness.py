from __future__ import annotations

from decimal import Decimal
import sys
from typing import Callable


from config.training_policy import (
    OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE,
    OPTIMIZER_INNER_VALIDATE_MIN_SCORE,
    resolve_optimizer_local_min_score_finalist_top_k,
)
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.strategy_params import build_runtime_param_raw_value
from strategies.breakout.search_space import get_breakout_local_min_candidate_fields, resolve_breakout_neighbor_spec
from tools.optimizer.objective_runner import (
    evaluate_prepared_inner_validate_score,
    evaluate_prepared_train_score,
    resolve_inner_validate_scope,
    resolve_search_train_scope,
)
from tools.optimizer.prep import prepare_trial_inputs
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


def _get_local_min_score_cache(session):
    if not hasattr(session, "local_min_score_cache"):
        session.local_min_score_cache = {}
    return session.local_min_score_cache


def _get_local_min_payload_score_cache(session):
    if not hasattr(session, "local_min_payload_score_cache"):
        session.local_min_payload_score_cache = {}
    return session.local_min_payload_score_cache


def _get_dominant_year_dependency_cache(session):
    if not hasattr(session, "dominant_year_dependency_cache"):
        session.dominant_year_dependency_cache = {}
    return session.dominant_year_dependency_cache


def _get_inner_validate_cache(session):
    if not hasattr(session, "inner_validate_cache"):
        session.inner_validate_cache = {}
    return session.inner_validate_cache


def _build_payload_score_cache_key(payload: dict):
    canonical_payload = params_to_json_dict(build_params_from_mapping(payload))
    return tuple(sorted(canonical_payload.items()))


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


class _FinalistProgressBoard:
    def __init__(self, session, finalists: list[dict]):
        self.session = session
        self.finalists = finalists
        self.lines: list[str] = []
        self.rendered = False

    def _format_line(self, idx: int, *, prefix: str, progress_text: str, local_text: str, status_text: str):
        finalist = self.finalists[idx]
        trial = finalist["trial"]
        base_score = float(finalist["base_score"])
        return (
            f"{prefix} finalist {idx + 1}/{len(self.finalists)} | trial #{int(trial.number) + 1} "
            f"| base_score={base_score:.3f} | {progress_text} | local_min_score={local_text} | {status_text}"
        )

    def initialize(self):
        self.lines = [
            self._format_line(idx, prefix="⏳", progress_text="進度 0/0", local_text="N/A", status_text="等待中")
            for idx in range(len(self.finalists))
        ]
        self._render()

    def update_pending(self, idx: int, *, total_neighbors: int):
        self.lines[idx] = self._format_line(
            idx,
            prefix="⏳",
            progress_text=f"進度 0/{int(total_neighbors)}",
            local_text="N/A",
            status_text="分析中",
        )
        self._render()

    def update_neighbor(self, idx: int, *, current_neighbor: int, total_neighbors: int, current_local_min):
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

    def _render(self):
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
):
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
    if on_start is not None:
        on_start(total_neighbors)
    elif progress_label:
        _print_progress_line(session, f"⏳ {progress_label}: 開始 local_min_score 分析，共 {total_neighbors} 個鄰點")

    local_min_score = float("inf")
    evaluated_neighbors = 0
    early_stopped = False
    for neighbor_idx, payload in enumerate(neighbor_payloads, start=1):
        evaluated_neighbors = neighbor_idx
        payload_cache_key = _build_payload_score_cache_key(payload)
        cached_payload_score = payload_score_cache.get(payload_cache_key)
        if cached_payload_score is None:
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
            )
            search_scope = resolve_search_train_scope(session, prep_result["master_dates"], objective_mode=session.objective_mode)
            evaluation = evaluate_prepared_train_score(
                session,
                ai_params=ai_params,
                prep_result=prep_result,
                search_scope=search_scope,
                profile_stats=None,
            )
            score = float(evaluation["score"])
            payload_score_cache[payload_cache_key] = score
        else:
            score = float(cached_payload_score)

        if score < local_min_score:
            local_min_score = score
        if on_neighbor is not None:
            current_local_min = None if local_min_score == float("inf") else float(local_min_score)
            on_neighbor(neighbor_idx, total_neighbors, current_local_min)
        if local_min_score <= 0.0:
            early_stopped = neighbor_idx < total_neighbors
            break

    if local_min_score == float("inf"):
        local_min_score = float(INVALID_TRIAL_VALUE)
    cache[cache_key] = {
        "score": float(local_min_score),
        "total_neighbors": total_neighbors,
        "evaluated_neighbors": evaluated_neighbors,
        "early_stopped": bool(early_stopped),
    }
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
    )
    search_scope = resolve_search_train_scope(session, prep_result["master_dates"], objective_mode=objective_mode)
    evaluation = evaluate_prepared_train_score(
        session,
        ai_params=ai_params,
        prep_result=prep_result,
        search_scope=search_scope,
        profile_stats={},
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
    )
    validate_scope = resolve_inner_validate_scope(session, prep_result["master_dates"], objective_mode=objective_mode)
    diagnostics = evaluate_prepared_inner_validate_score(
        session,
        ai_params=ai_params,
        prep_result=prep_result,
        validate_scope=validate_scope,
        profile_stats={},
    )
    diagnostics = _normalize_inner_validate_diagnostics(diagnostics) or {
        "enabled": True,
        "inner_validate_score": float(INVALID_TRIAL_VALUE),
        "inner_validate_gate": False,
        "fail_reason": "inner validation diagnostics 無法計算",
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


def _resolve_local_min_score_finalist_top_k(session, top_k=None):
    if top_k is not None:
        return max(1, int(top_k))
    return resolve_optimizer_local_min_score_finalist_top_k(getattr(session, "n_trials", 0))


def list_local_min_score_finalists(study, *, session, objective_mode: str, top_k=None, include_trial=None, show_progress: bool = False):
    resolved_top_k = _resolve_local_min_score_finalist_top_k(session, top_k)
    sorted_trials = _list_qualified_trials_for_objective(study, objective_mode)
    if not sorted_trials:
        return []

    _seed_payload_score_cache_from_study(session, study, objective_mode)
    finalists = _build_display_finalists(sorted_trials, top_k=resolved_top_k, include_trial=include_trial)
    progress_board = None
    if show_progress:
        progress_board = _FinalistProgressBoard(session, finalists)
        progress_board.initialize()

    enriched_finalists = []
    for finalist_idx, item in enumerate(finalists):
        trial = item["trial"]
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
            )
        else:
            local_min_score = compute_local_min_score(session, trial)
        base_score = float(item["base_score"])
        enriched_item = {
            "trial": trial,
            "base_rank": int(item.get("base_rank", 0)),
            "base_score": base_score,
            "local_min_score": float(local_min_score),
            "local_retention": _compute_local_retention(base_score, float(local_min_score)),
            "gate_pass": bool(local_min_score > 0.0),
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


def print_local_min_score_finalist_review(study, *, session, objective_mode: str, colors: dict, winner_trial=None, top_k=None):
    finalists = list_local_min_score_finalists(
        study,
        session=session,
        objective_mode=objective_mode,
        top_k=top_k,
        include_trial=winner_trial,
        show_progress=True,
    )
    if not finalists:
        return [], winner_trial
    retention_best_finalist = select_best_finalist_by_local_retention(finalists)
    retention_best_trial = None if retention_best_finalist is None else retention_best_finalist["trial"]
    if winner_trial is None:
        best_finalist = _select_best_finalist_by_local_min_score(
            finalists,
            use_inner_validate=is_inner_validate_anti_overfit_enabled(objective_mode),
        )
        winner_trial = None if best_finalist is None else best_finalist["trial"]

    gray = colors.get("gray", "")
    green = colors.get("green", "")
    red = colors.get("red", "")
    cyan = colors.get("cyan", "")
    yellow = colors.get("yellow", "")
    reset = colors.get("reset", "")

    inner_validate_enabled = is_inner_validate_anti_overfit_enabled(objective_mode)
    dependency_enabled = is_dominant_year_dependency_anti_overfit_enabled()
    separator_width = 96 + (46 if inner_validate_enabled else 0) + (72 if dependency_enabled else 0)
    print(f"{gray}{'-' * separator_width}{reset}")
    header = (
        f"{'trial':<14}{'rank':>8}{'base_score':>14}{'local_min':>14}{'retention':>12}"
        f"{'local_gate':>14}"
    )
    if inner_validate_enabled:
        header += f"{'val_year':>10}{'val_score':>12}{'val_rank':>10}{'val_gate':>12}"
    if dependency_enabled:
        header += f"{'dep':>8}{'dom_year':>10}{'dom_pnl%':>10}{'pos_trades':>12}{'pos_symbols':>13}{'top_trade%':>12}{'dep_reason':>14}"
    header += f"{'結果':>12}"
    print(header)
    for idx, item in enumerate(finalists, start=1):
        trial = item["trial"]
        gate_pass = bool(item["gate_pass"])
        status_text = 'PASS' if gate_pass else 'FAIL'
        status_color = green if gate_pass else red
        is_winner = winner_trial is not None and int(winner_trial.number) == int(trial.number)
        is_retention_ref = (
            retention_best_trial is not None
            and int(retention_best_trial.number) == int(trial.number)
            and not is_winner
        )
        has_inner_validate_fail = inner_validate_enabled and not _has_inner_validate_pass(item)
        has_dependency_warning = dependency_enabled and _has_dependency_warning(item)
        if is_winner:
            result_text = 'winner'
        elif gate_pass and has_inner_validate_fail:
            result_text = 'val_skip'
        elif gate_pass and has_dependency_warning:
            result_text = 'dep_skip'
        elif is_retention_ref:
            result_text = 'ret_ref'
        else:
            result_text = '保留' if gate_pass else '淘汰'
        result_color = yellow if result_text in {'winner', 'ret_ref'} else (red if result_text in {'val_skip', 'dep_skip'} else status_color)

        line = (
            f"#{int(trial.number) + 1:<13}"
            f"#{int(item.get('base_rank', idx)):>7}"
            f"{float(item['base_score']):>14.3f}"
            f"{float(item['local_min_score']):>14.3f}"
            f"{float(item['local_retention']):>12.3f}"
            f"{status_color}{status_text:>14}{reset}"
        )
        if inner_validate_enabled:
            val_diag = item.get("inner_validate_diagnostics") if isinstance(item.get("inner_validate_diagnostics"), dict) else {}
            validate_year = val_diag.get('validate_year')
            validate_year_text = 'N/A' if validate_year is None else str(validate_year)
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
                f"{validate_year_text:>10}"
                f"{validate_score:>12.3f}"
                f"{validate_rank_text:>10}"
                f"{validate_color}{validate_text:>12}{reset}"
            )
        if dependency_enabled:
            diagnostics = item.get("dominant_year_dependency_diagnostics") if isinstance(item.get("dominant_year_dependency_diagnostics"), dict) else {}
            dep_text = 'WARN' if bool(diagnostics.get('dependency_warning', False)) else str(diagnostics.get('dependency_status') or 'OK')
            dep_color = red if dep_text == 'WARN' else green
            dominant_year = diagnostics.get('dominant_entry_year')
            dominant_year_text = 'N/A' if dominant_year is None else str(dominant_year)
            dominant_pnl_pct = float(diagnostics.get('dominant_year_positive_pnl_share', 0.0)) * 100.0
            top_trade_pct = float(diagnostics.get('top_trade_pnl_share_in_dominant_year', 0.0)) * 100.0
            positive_trade_count = int(diagnostics.get('dominant_year_positive_trade_count', 0) or 0)
            positive_symbol_count = int(diagnostics.get('dominant_year_positive_symbol_count', 0) or 0)
            dep_reason_text = _format_dependency_reason(diagnostics)
            line += (
                f"{dep_color}{dep_text:>8}{reset}"
                f"{dominant_year_text:>10}"
                f"{dominant_pnl_pct:>10.1f}"
                f"{positive_trade_count:>12}"
                f"{positive_symbol_count:>13}"
                f"{top_trade_pct:>12.1f}"
                f"{dep_reason_text:>14}"
            )
        line += f"{result_color}{result_text:>12}{reset}"
        print(line)
    print(f"{gray}{'=' * separator_width}{reset}")
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
            f"🏁 winner(local_min{' + inner_val_rank' if is_inner_validate_anti_overfit_enabled(objective_mode) else ''}{' + dependency_safe' if is_dominant_year_dependency_anti_overfit_enabled() else ''}): trial #{int(trial.number) + 1} | base_score={float(best_finalist['base_score']):.3f} | local_min_score={float(best_finalist['local_min_score']):.3f} | retention={float(best_finalist['local_retention']):.3f}"
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
