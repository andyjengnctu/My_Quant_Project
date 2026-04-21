from __future__ import annotations

from decimal import Decimal
import sys
from typing import Callable


from config.training_policy import OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K
from core.params_io import build_params_from_mapping
from core.strategy_params import build_runtime_param_raw_value
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE, resolve_breakout_neighbor_spec
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



def _get_progress_colors(session):
    colors = getattr(session, "colors", None)
    return colors if isinstance(colors, dict) else {}


def _get_local_min_score_cache(session):
    if not hasattr(session, "local_min_score_cache"):
        session.local_min_score_cache = {}
    return session.local_min_score_cache


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

    def update_done(self, idx: int, *, total_neighbors: int, local_min_score: float):
        gate_status = "PASS" if float(local_min_score) > 0.0 else "FAIL"
        self.lines[idx] = self._format_line(
            idx,
            prefix="✅",
            progress_text=f"進度 {int(total_neighbors)}/{int(total_neighbors)}",
            local_text=f"{float(local_min_score):.3f}",
            status_text=gate_status,
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


def _select_best_finalist_by_local_retention(finalists: list[dict]):
    eligible = [item for item in finalists if bool(item.get("gate_pass", False))]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            float(item.get("local_retention", float("-inf"))),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return eligible[0]

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
        base_score = trial.user_attrs.get("base_score", trial.value)
        try:
            local_min_score = float(base_score)
        except (TypeError, ValueError):
            local_min_score = float(INVALID_TRIAL_VALUE)
        cache[cache_key] = {"score": float(local_min_score), "total_neighbors": 0}
        if on_finish is not None:
            on_finish(0, float(local_min_score))
        elif progress_label:
            _print_progress_line(session, f"✅ {progress_label}: 無合法鄰點，local_min_score={float(local_min_score):.3f}")
        return float(local_min_score)

    if on_start is not None:
        on_start(len(neighbor_payloads))
    elif progress_label:
        _print_progress_line(session, f"⏳ {progress_label}: 開始 local_min_score 分析，共 {len(neighbor_payloads)} 個鄰點")

    local_min_score = float("inf")
    for neighbor_idx, payload in enumerate(neighbor_payloads, start=1):
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
        if on_neighbor is not None:
            current_local_min = None if local_min_score == float("inf") else float(local_min_score)
            on_neighbor(neighbor_idx, len(neighbor_payloads), current_local_min)

    if local_min_score == float("inf"):
        local_min_score = float(INVALID_TRIAL_VALUE)
    cache[cache_key] = {"score": float(local_min_score), "total_neighbors": len(neighbor_payloads)}
    if on_finish is not None:
        on_finish(len(neighbor_payloads), float(local_min_score))
    elif progress_label:
        gate_status = "PASS" if float(local_min_score) > 0.0 else "FAIL"
        _print_progress_line(session, f"✅ {progress_label}: local_min_score={float(local_min_score):.3f} | gate={gate_status}")
    return float(local_min_score)




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


def list_local_min_score_finalists(study, *, session, objective_mode: str, top_k: int = OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K, include_trial=None, show_progress: bool = False):
    sorted_trials = _list_qualified_trials_for_objective(study, objective_mode)
    if not sorted_trials:
        return []

    finalists = _build_display_finalists(sorted_trials, top_k=top_k, include_trial=include_trial)
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
                on_finish=lambda total, score, idx=finalist_idx: progress_board.update_done(
                    idx,
                    total_neighbors=total,
                    local_min_score=score,
                ),
            )
        else:
            local_min_score = compute_local_min_score(session, trial)
        base_score = float(item["base_score"])
        enriched_finalists.append({
            "trial": trial,
            "base_rank": int(item.get("base_rank", 0)),
            "base_score": base_score,
            "local_min_score": float(local_min_score),
            "local_retention": _compute_local_retention(base_score, float(local_min_score)),
            "gate_pass": bool(local_min_score > 0.0),
        })
    enriched_finalists.sort(
        key=lambda item: (
            float(item["local_retention"]),
            float(item["base_score"]),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return enriched_finalists


def print_local_min_score_finalist_review(study, *, session, objective_mode: str, colors: dict, winner_trial=None, top_k: int = OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K):
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
    if winner_trial is None:
        best_finalist = _select_best_finalist_by_local_retention(finalists)
        winner_trial = None if best_finalist is None else best_finalist["trial"]

    gray = colors.get("gray", "")
    green = colors.get("green", "")
    red = colors.get("red", "")
    cyan = colors.get("cyan", "")
    yellow = colors.get("yellow", "")
    reset = colors.get("reset", "")

    print(f"{gray}{'-' * 96}{reset}")
    print(f"{'trial':<14}{'rank':>8}{'base_score':>14}{'local_min':>14}{'retention':>12}{'local_gate':>14}{'結果':>12}")
    for idx, item in enumerate(finalists, start=1):
        trial = item["trial"]
        gate_pass = bool(item["gate_pass"])
        status_text = 'PASS' if gate_pass else 'FAIL'
        status_color = green if gate_pass else red
        result_text = 'winner' if winner_trial is not None and int(winner_trial.number) == int(trial.number) else ('保留' if gate_pass else '淘汰')
        result_color = yellow if result_text == 'winner' else status_color
        print(
            f"#{int(trial.number) + 1:<13}"
            f"#{int(item.get('base_rank', idx)):>7}"
            f"{float(item['base_score']):>14.3f}"
            f"{float(item['local_min_score']):>14.3f}"
            f"{float(item['local_retention']):>12.3f}"
            f"{status_color}{status_text:>14}{reset}"
            f"{result_color}{result_text:>12}{reset}"
        )
    print(f"{gray}{'=' * 96}{reset}")
    return finalists, winner_trial


def print_local_min_score_winner_summary(*, winner_trial, session, colors: dict):
    return None


def resolve_best_completed_trial_with_local_min_score_or_none(study, *, session, objective_mode: str, show_progress: bool = True, top_k: int = OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K):
    resolver_cache = _get_best_trial_resolver_cache(session)
    cache_key = _build_best_trial_cache_key(study, objective_mode, top_k)
    cached_trial = resolver_cache.get(cache_key)
    if cached_trial is not None:
        return cached_trial

    finalists = list_local_min_score_finalists(
        study,
        session=session,
        objective_mode=objective_mode,
        top_k=top_k,
        include_trial=None,
        show_progress=show_progress,
    )
    best_finalist = _select_best_finalist_by_local_retention(finalists)
    if best_finalist is None:
        if show_progress:
            _print_progress_line(session, "ℹ️ 沒有任何 finalist 通過 local_min_score gate")
        resolver_cache[cache_key] = None
        return None
    trial = best_finalist["trial"]
    if show_progress:
        _print_progress_line(
            session,
            f"🏁 winner: trial #{int(trial.number) + 1} | base_score={float(best_finalist['base_score']):.3f} | local_min_score={float(best_finalist['local_min_score']):.3f} | retention={float(best_finalist['local_retention']):.3f}"
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
            top_k=OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K,
        )

    return _resolver
