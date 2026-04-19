from __future__ import annotations

from decimal import Decimal
from typing import Callable

FINALIST_LOCAL_REVIEW_TOP_K = 3

from core.params_io import build_params_from_mapping
from core.strategy_params import build_runtime_param_raw_value
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from tools.optimizer.objective_runner import evaluate_prepared_train_score, resolve_search_train_scope
from tools.optimizer.prep import prepare_trial_inputs
from tools.optimizer.study_utils import (
    INVALID_TRIAL_VALUE,
    OPTIMIZER_TP_PERCENT_SEARCH_SPEC,
    build_best_params_payload_from_trial,
    is_qualified_trial_value,
    list_completed_study_trials,
    normalize_objective_mode,
    objective_modes_are_compatible,
)

LOCAL_MIN_CORE_FIELDS = (
    "high_len",
    "atr_len",
    "atr_times_init",
    "atr_times_trail",
    "atr_buy_tol",
)


def _trial_matches_objective_mode(trial, objective_mode: str) -> bool:
    expected = normalize_objective_mode(objective_mode)
    actual = str(trial.user_attrs.get("objective_mode", "")).strip()
    if actual:
        return objective_modes_are_compatible(actual, expected)
    return expected == normalize_objective_mode("")


def _resolve_neighbor_step(session, field_name: str):
    if field_name == "high_len":
        return int(session.optimizer_high_len_step), "int", int(session.optimizer_high_len_min), int(session.optimizer_high_len_max)
    if field_name == "tp_percent":
        spec = OPTIMIZER_TP_PERCENT_SEARCH_SPEC
        return float(spec["step"]), "float", float(spec["low"]), float(spec["high"])

    spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE[field_name]
    kind = str(spec["kind"])
    step = spec.get("step", 1)
    if field_name == "vol_long_len":
        low_value = 1
    else:
        low_value = spec["low"]
    return step, kind, low_value, spec["high"]


def _apply_step(field_name: str, current_value, step_value, direction: int, kind: str):
    if kind == "int":
        return int(current_value) + int(direction) * int(step_value)

    candidate = Decimal(str(current_value)) + (Decimal(str(step_value)) * Decimal(direction))
    return float(candidate)


def _build_neighbor_candidates(session, trial):
    center_payload = build_best_params_payload_from_trial(trial, fixed_tp_percent=session.optimizer_fixed_tp_percent)
    candidate_fields = list(LOCAL_MIN_CORE_FIELDS)
    if "tp_percent" in trial.params:
        candidate_fields.append("tp_percent")
    if bool(center_payload.get("use_bb", False)):
        candidate_fields.extend(("bb_len", "bb_mult"))
    if bool(center_payload.get("use_kc", False)):
        candidate_fields.extend(("kc_len", "kc_mult"))
    if bool(center_payload.get("use_vol", False)):
        candidate_fields.extend(("vol_short_len", "vol_long_len"))

    neighbors = []
    seen_payload_keys = set()
    for field_name in candidate_fields:
        step_value, kind, low_value, high_value = _resolve_neighbor_step(session, field_name)
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


def compute_local_min_score(session, trial):
    if not hasattr(session, "local_min_score_cache"):
        session.local_min_score_cache = {}
    cache_key = int(trial.number)
    cached = session.local_min_score_cache.get(cache_key)
    if cached is not None:
        return float(cached)

    neighbor_payloads = _build_neighbor_candidates(session, trial)
    if not neighbor_payloads:
        base_score = trial.user_attrs.get("base_score", trial.value)
        try:
            local_min_score = float(base_score)
        except (TypeError, ValueError):
            local_min_score = float(INVALID_TRIAL_VALUE)
        session.local_min_score_cache[cache_key] = float(local_min_score)
        return float(local_min_score)

    local_min_score = float("inf")
    for payload in neighbor_payloads:
        ai_params = build_params_from_mapping(payload)
        prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(ai_params, "optimizer_max_workers"))
        prep_result = prepare_trial_inputs(
            raw_data_cache=session.raw_data_cache,
            params=ai_params,
            default_max_workers=session.default_max_workers,
            executor_bundle=prep_executor_bundle,
            static_fast_cache=session.static_fast_cache,
            static_master_dates=session.master_dates,
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
        if score < local_min_score:
            local_min_score = score

    if local_min_score == float("inf"):
        local_min_score = float(INVALID_TRIAL_VALUE)
    session.local_min_score_cache[cache_key] = float(local_min_score)
    return float(local_min_score)




def list_local_min_score_finalists(study, *, session, objective_mode: str, top_k: int = FINALIST_LOCAL_REVIEW_TOP_K, include_trial=None):
    qualified_trials = [
        trial
        for trial in list_completed_study_trials(study)
        if is_qualified_trial_value(trial.value) and _trial_matches_objective_mode(trial, objective_mode)
    ]
    if not qualified_trials:
        return []

    sorted_trials = sorted(qualified_trials, key=lambda trial: (float(trial.value), -int(trial.number)), reverse=True)
    selected_trials = list(sorted_trials[: max(1, int(top_k))])
    if include_trial is not None and all(int(trial.number) != int(include_trial.number) for trial in selected_trials):
        selected_trials.append(include_trial)

    finalists = []
    for trial in selected_trials:
        local_min_score = compute_local_min_score(session, trial)
        finalists.append({
            "trial": trial,
            "base_score": float(trial.user_attrs.get("base_score", trial.value)),
            "local_min_score": float(local_min_score),
            "gate_pass": bool(local_min_score > 0.0),
        })
    return finalists


def print_local_min_score_finalist_review(study, *, session, objective_mode: str, colors: dict, winner_trial=None, top_k: int = FINALIST_LOCAL_REVIEW_TOP_K):
    finalists = list_local_min_score_finalists(
        study,
        session=session,
        objective_mode=objective_mode,
        top_k=top_k,
        include_trial=winner_trial,
    )
    if not finalists:
        return []

    gray = colors.get("gray", "")
    green = colors.get("green", "")
    red = colors.get("red", "")
    cyan = colors.get("cyan", "")
    yellow = colors.get("yellow", "")
    reset = colors.get("reset", "")

    print(f"{gray}{'=' * 96}{reset}")
    print(f"{cyan}🔍 Finalist Local Robustness Review{reset}")
    print(f"{gray}Top finalists by base_score:{reset}")
    for item in finalists:
        trial = item["trial"]
        print(f"  #{int(trial.number) + 1:<4} base_score={float(item['base_score']):.3f}")
    print(f"{gray}{'-' * 96}{reset}")
    print(f"{'trial':<14}{'base_score':>14}{'local_min':>14}{'local_gate':>14}{'結果':>12}")
    for item in finalists:
        trial = item["trial"]
        gate_pass = bool(item["gate_pass"])
        status_text = 'PASS' if gate_pass else 'FAIL'
        status_color = green if gate_pass else red
        result_text = 'winner' if winner_trial is not None and int(winner_trial.number) == int(trial.number) else ('保留' if gate_pass else '淘汰')
        result_color = yellow if result_text == 'winner' else status_color
        print(
            f"#{int(trial.number) + 1:<13}"
            f"{float(item['base_score']):>14.3f}"
            f"{float(item['local_min_score']):>14.3f}"
            f"{status_color}{status_text:>14}{reset}"
            f"{result_color}{result_text:>12}{reset}"
        )
    print(f"{gray}{'=' * 96}{reset}")
    return finalists


def print_local_min_score_winner_summary(*, winner_trial, session, colors: dict):
    gray = colors.get("gray", "")
    green = colors.get("green", "")
    yellow = colors.get("yellow", "")
    reset = colors.get("reset", "")
    local_min_score = compute_local_min_score(session, winner_trial)
    gate_status = 'PASS' if local_min_score > 0.0 else 'FAIL'
    gate_color = green if gate_status == 'PASS' else colors.get('red', '')
    print(f"{gray}{'=' * 96}{reset}")
    print(f"{green}✅ Final winner after local robustness gate{reset}")
    print(
        f"winner trial       : #{int(winner_trial.number) + 1}\n"
        f"base_score         : {float(winner_trial.user_attrs.get('base_score', winner_trial.value)):.3f}\n"
        f"local_min_score    : {float(local_min_score):.3f}\n"
        f"local gate         : {gate_color}{gate_status}{reset}"
    )
    print(f"{gray}{'=' * 96}{reset}")


def resolve_best_completed_trial_with_local_min_score_or_none(study, *, session, objective_mode: str):
    qualified_trials = [
        trial
        for trial in list_completed_study_trials(study)
        if is_qualified_trial_value(trial.value) and _trial_matches_objective_mode(trial, objective_mode)
    ]
    if not qualified_trials:
        return None

    sorted_trials = sorted(qualified_trials, key=lambda trial: (float(trial.value), -int(trial.number)), reverse=True)
    for trial in sorted_trials:
        local_min_score = compute_local_min_score(session, trial)
        if local_min_score > 0.0:
            return trial
    return None


def build_local_min_score_best_trial_resolver(*, session, objective_mode: str) -> Callable:
    def _resolver(study):
        return resolve_best_completed_trial_with_local_min_score_or_none(
            study,
            session=session,
            objective_mode=objective_mode,
        )

    return _resolver
