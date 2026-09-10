from __future__ import annotations

import math
from collections import OrderedDict

from config.training_policy import (
    OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE,
    OPTIMIZER_FIXED_TP_PERCENT,
    OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
    OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE,
)
from core.portfolio_stats import calc_plain_romd
from core.seed_ensemble_policy import normalize_seed_ensemble_members, renumber_seed_ensemble_members
from core.training_policy import (
    is_optimizer_local_min_review_enabled,
    resolve_optimizer_base_finalists_agree_min_agree,
    resolve_optimizer_enabled_policy_indicators,
    resolve_optimizer_local_finalists_agree_min_agree,
    resolve_optimizer_retention_finalists_agree_min_agree,
)
from services.optimizer.outer_rolling_params import _materialize_fixed_strategy_param_overrides_in_members
from services.optimizer.robustness import (
    _has_dependency_warning,
    _has_inner_validate_pass,
    is_dominant_year_dependency_anti_overfit_enabled,
    is_inner_validate_anti_overfit_enabled,
)
from services.optimizer.study_utils import INVALID_TRIAL_VALUE, build_best_params_payload_from_trial


BASE_FINALIST_BEST_POLICY_NAME = "base_finalist_best"
LOCAL_FINALIST_BEST_POLICY_NAME = "local_finalist_best"
RETENTION_FINALIST_BEST_POLICY_NAME = "retention_finalist_best"
FINALIST_BEST_POLICY_NAMES = (
    BASE_FINALIST_BEST_POLICY_NAME,
    LOCAL_FINALIST_BEST_POLICY_NAME,
    RETENTION_FINALIST_BEST_POLICY_NAME,
)

BASE_FINALISTS_AGREE_POLICY_NAME = "base_finalists_agree"
LOCAL_FINALISTS_AGREE_POLICY_NAME = "local_finalists_agree"
RETENTION_FINALISTS_AGREE_POLICY_NAME = "retention_finalists_agree"
FINALISTS_AGREE_POLICY_NAMES = (
    BASE_FINALISTS_AGREE_POLICY_NAME,
    LOCAL_FINALISTS_AGREE_POLICY_NAME,
    RETENTION_FINALISTS_AGREE_POLICY_NAME,
)

SEED_ENSEMBLE_POLICY_NAMES = ("base", "local", "retention")
LOCAL_DEPENDENT_POLICY_NAMES = frozenset({
    LOCAL_FINALIST_BEST_POLICY_NAME,
    RETENTION_FINALIST_BEST_POLICY_NAME,
    LOCAL_FINALISTS_AGREE_POLICY_NAME,
    RETENTION_FINALISTS_AGREE_POLICY_NAME,
    "local",
    "retention",
})
ALL_REPORT_POLICY_NAMES = (
    *FINALIST_BEST_POLICY_NAMES,
    *FINALISTS_AGREE_POLICY_NAMES,
    *SEED_ENSEMBLE_POLICY_NAMES,
)


def _resolve_report_policy_names(policy_names: tuple[str, ...]) -> tuple[str, ...]:
    enabled = set(resolve_optimizer_enabled_policy_indicators(policy_names))
    local_enabled = bool(is_optimizer_local_min_review_enabled())
    return tuple(
        name
        for name in policy_names
        if name in enabled and (local_enabled or name not in LOCAL_DEPENDENT_POLICY_NAMES)
    )


REPORT_POLICY_NAMES = _resolve_report_policy_names(ALL_REPORT_POLICY_NAMES)
REPORT_POLICY_LABELS = {
    BASE_FINALIST_BEST_POLICY_NAME: "base best",
    LOCAL_FINALIST_BEST_POLICY_NAME: "local best",
    RETENTION_FINALIST_BEST_POLICY_NAME: "retention best",
    BASE_FINALISTS_AGREE_POLICY_NAME: "base agree",
    LOCAL_FINALISTS_AGREE_POLICY_NAME: "local agree",
    RETENTION_FINALISTS_AGREE_POLICY_NAME: "retention agree",
    "base": "base ensemble",
    "local": "local ensemble",
    "retention": "retention ensemble",
}
FINALIST_BEST_RESULT_POLICY_NAMES = tuple(name for name in FINALIST_BEST_POLICY_NAMES if name in set(REPORT_POLICY_NAMES))
FINALISTS_AGREE_RESULT_POLICY_NAMES = tuple(name for name in FINALISTS_AGREE_POLICY_NAMES if name in set(REPORT_POLICY_NAMES))
SEED_ENSEMBLE_RESULT_POLICY_NAMES = tuple(name for name in SEED_ENSEMBLE_POLICY_NAMES if name in set(REPORT_POLICY_NAMES))
FINALIST_BEST_TABLE_TITLE = "FINALIST BEST RESULTS"
FINALISTS_AGREE_TABLE_TITLE = "FINALIST AGREE RESULTS"
SEED_ENSEMBLE_RESULTS_TABLE_TITLE = "SEED ENSEMBLE RESULTS"

# Retention-threshold variants were retired from the official OOS/ROOS contract.
# Keep these names as empty compatibility anchors so old helper imports do not
# reintroduce threshold replay, reports, or paramset output.
BASE_RETENTION_COMPARISON_THRESHOLDS = ()
BASE_RETENTION_COMPARISON_POLICY_THRESHOLDS = OrderedDict()
BASE_RETENTION_COMPARISON_POLICY_NAMES = ()
BASE_RETENTION_COMPARISON_POLICY_LABELS = {}
CHAIN_POLICY_NAMES = REPORT_POLICY_NAMES

PARAMSET_FILENAME_BY_POLICY = {
    BASE_FINALIST_BEST_POLICY_NAME: "roos_base_best.json",
    LOCAL_FINALIST_BEST_POLICY_NAME: "roos_local_best.json",
    RETENTION_FINALIST_BEST_POLICY_NAME: "roos_retention_best.json",
    BASE_FINALISTS_AGREE_POLICY_NAME: "roos_base_finalists_agree.json",
    LOCAL_FINALISTS_AGREE_POLICY_NAME: "roos_local_finalists_agree.json",
    RETENTION_FINALISTS_AGREE_POLICY_NAME: "roos_retention_finalists_agree.json",
    "base": "roos_ensemble_base.json",
    "local": "roos_ensemble_local.json",
    "retention": "roos_ensemble_retention.json",
}

NONROLLING_PARAMSET_FILENAME_BY_POLICY = {
    BASE_FINALIST_BEST_POLICY_NAME: "base_best.json",
    LOCAL_FINALIST_BEST_POLICY_NAME: "local_best.json",
    RETENTION_FINALIST_BEST_POLICY_NAME: "retention_best.json",
    BASE_FINALISTS_AGREE_POLICY_NAME: "base_finalists_agree.json",
    LOCAL_FINALISTS_AGREE_POLICY_NAME: "local_finalists_agree.json",
    RETENTION_FINALISTS_AGREE_POLICY_NAME: "retention_finalists_agree.json",
    "base": "base.json",
    "local": "local.json",
    "retention": "retention.json",
}

NONROLLING_PARAMSET_FILENAME_PREFIX_BY_MODE = {
    "study": "",
    "full": "full_",
    "oos": "oos_",
    "trade": "trade_",
}

POLICY_OUTPUT_LABELS = {
    BASE_FINALIST_BEST_POLICY_NAME: "base_best",
    LOCAL_FINALIST_BEST_POLICY_NAME: "local_best",
    RETENTION_FINALIST_BEST_POLICY_NAME: "retention_best",
    BASE_FINALISTS_AGREE_POLICY_NAME: "base_agree",
    LOCAL_FINALISTS_AGREE_POLICY_NAME: "local_agree",
    RETENTION_FINALISTS_AGREE_POLICY_NAME: "retention_agree",
    "base": "base_ensemble",
    "local": "local_ensemble",
    "retention": "retention_ensemble",
}

STALE_POLICY_PARAMSET_FILENAMES = (
    "roos_base_r.json",
    "base_r.json",
    "oos_base_r.json",
    "trade_base_r.json",
    "base_r0.json",
    "base_r05.json",
    "oos_base_r0.json",
    "oos_base_r05.json",
    "trade_base_r0.json",
    "trade_base_r05.json",
    "roos_base_r0.json",
    "roos_base_r05.json",
    "roos_base_retention_gt_0_0.json",
    "roos_base_retention_gt_0_2.json",
    "roos_base_retention_gt_0_4.json",
    "roos_base_retention_gt_0_6.json",
    "roos_base_retention_gt_0_8.json",
    "base_retention_gt_0_0.json",
    "base_retention_gt_0_2.json",
    "base_retention_gt_0_4.json",
    "base_retention_gt_0_6.json",
    "base_retention_gt_0_8.json",
    "oos_base_retention_gt_0_0.json",
    "oos_base_retention_gt_0_2.json",
    "oos_base_retention_gt_0_4.json",
    "oos_base_retention_gt_0_6.json",
    "oos_base_retention_gt_0_8.json",
    "trade_base_retention_gt_0_0.json",
    "trade_base_retention_gt_0_2.json",
    "trade_base_retention_gt_0_4.json",
    "trade_base_retention_gt_0_6.json",
    "trade_base_retention_gt_0_8.json",
    "roos_base_agree.json",
    "base_agree.json",
    "oos_base_agree.json",
    "trade_base_agree.json",
    "roos_local_agree.json",
    "local_agree.json",
    "oos_local_agree.json",
    "trade_local_agree.json",
    "roos_retention_agree.json",
    "retention_agree.json",
    "oos_retention_agree.json",
    "trade_retention_agree.json",
    "full_base.json",
    "full_local.json",
    "full_retention.json",
    "oos_base.json",
    "oos_local.json",
    "oos_retention.json",
    "trade_base.json",
    "trade_local.json",
    "trade_retention.json",
    "roos_base.json",
    "roos_local.json",
    "roos_retention.json",
)


SEED_ENSEMBLE_OOS_TABLE_TITLE = SEED_ENSEMBLE_RESULTS_TABLE_TITLE
SEED_ENSEMBLE_RETENTION_TABLE_TITLE = ""


def optimizer_seed_ensemble_table_titles() -> tuple[str, str]:
    return SEED_ENSEMBLE_OOS_TABLE_TITLE, SEED_ENSEMBLE_RETENTION_TABLE_TITLE

def _is_rolling_random_seed_ensemble_enabled() -> bool:
    return bool(OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED) and int(OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE or 1) > 1

def _active_optimizer_table_titles() -> tuple[str, str]:
    if _is_rolling_random_seed_ensemble_enabled():
        return optimizer_seed_ensemble_table_titles()
    return SEED_ENSEMBLE_RESULTS_TABLE_TITLE, ""

def _select_winner(finalists: list[dict], *, objective_mode: str):
    eligible = [item for item in finalists if bool(item.get("gate_pass", False))]
    if is_inner_validate_anti_overfit_enabled(objective_mode):
        eligible = [item for item in eligible if _has_inner_validate_pass(item)]
    if is_dominant_year_dependency_anti_overfit_enabled():
        safe = [item for item in eligible if not _has_dependency_warning(item)]
        if safe:
            eligible = safe
    if not eligible:
        return None
    eligible = sorted(
        eligible,
        key=lambda item: (
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("local_retention", float("-inf"))),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return eligible[0]


def _build_local_rank_map(finalists: list[dict]) -> dict[int, int]:
    return {int(item["trial"].number): rank for rank, item in enumerate(finalists or [], start=1) if item.get("trial") is not None}


def _build_retention_rank_map(finalists: list[dict]) -> dict[int, int]:
    ranked = sorted(
        [item for item in list(finalists or []) if item.get("trial") is not None],
        key=lambda item: (
            float(item.get("local_retention", float("-inf"))),
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return {int(item["trial"].number): rank for rank, item in enumerate(ranked, start=1)}


def _select_base_rank1_item(finalists: list[dict]):
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    if not items:
        return None
    return min(items, key=lambda item: (int(item.get("base_rank", 10**9) or 10**9), -float(item.get("base_score", INVALID_TRIAL_VALUE)), int(item["trial"].number)))



def _rank_base_finalist_items(finalists: list[dict]) -> list[dict]:
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    return sorted(
        items,
        key=lambda item: (
            int(item.get("base_rank", 10**9) or 10**9),
            -float(item.get("base_score", INVALID_TRIAL_VALUE)),
            int(item["trial"].number),
        ),
    )


def _rank_local_finalist_items(finalists: list[dict]) -> list[dict]:
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    return sorted(
        items,
        key=lambda item: (
            -float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            -float(item.get("local_retention", float("-inf"))),
            -float(item.get("base_score", INVALID_TRIAL_VALUE)),
            int(item["trial"].number),
        ),
    )


def _rank_retention_finalist_items(finalists: list[dict]) -> list[dict]:
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    return sorted(
        items,
        key=lambda item: (
            -float(item.get("local_retention", float("-inf"))),
            -float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            -float(item.get("base_score", INVALID_TRIAL_VALUE)),
            int(item["trial"].number),
        ),
    )




def _is_local_finalists_agree_policy(policy_name: str) -> bool:
    return str(policy_name) == LOCAL_FINALISTS_AGREE_POLICY_NAME


def _is_retention_finalists_agree_policy(policy_name: str) -> bool:
    return str(policy_name) == RETENTION_FINALISTS_AGREE_POLICY_NAME


def _is_finalists_agree_policy(policy_name: str) -> bool:
    return str(policy_name) in set(FINALISTS_AGREE_POLICY_NAMES)




def _is_local_finalist_best_policy(policy_name: str) -> bool:
    return str(policy_name) == LOCAL_FINALIST_BEST_POLICY_NAME


def _is_retention_finalist_best_policy(policy_name: str) -> bool:
    return str(policy_name) == RETENTION_FINALIST_BEST_POLICY_NAME


def _is_finalist_best_policy(policy_name: str) -> bool:
    return str(policy_name) in set(FINALIST_BEST_POLICY_NAMES)


def _finalist_best_policy_sort_key(member: dict, *, policy_name: str) -> tuple:
    item = dict(member or {})
    selected_trial = int(item.get("selected_trial", 0) or 0)
    optimizer_seed = int(item.get("optimizer_seed", item.get("seed", 0)) or 0)
    if _is_local_finalist_best_policy(policy_name):
        return (
            -float(item.get("local_min_score", item.get("local_min", INVALID_TRIAL_VALUE))),
            -float(item.get("retention", 0.0)),
            -float(item.get("base_score", INVALID_TRIAL_VALUE)),
            selected_trial,
            optimizer_seed,
        )
    if _is_retention_finalist_best_policy(policy_name):
        return (
            -float(item.get("retention", 0.0)),
            -float(item.get("local_min_score", item.get("local_min", INVALID_TRIAL_VALUE))),
            -float(item.get("base_score", INVALID_TRIAL_VALUE)),
            selected_trial,
            optimizer_seed,
        )
    return (
        int(item.get("base_rank", 10**9) or 10**9),
        -float(item.get("base_score", INVALID_TRIAL_VALUE)),
        selected_trial,
        optimizer_seed,
    )


def select_finalist_best_members(members: list[dict], *, policy_name: str) -> list[dict]:
    candidates = [dict(item) for item in list(members or []) if isinstance(item, dict)]
    if not candidates:
        return []
    winner = min(candidates, key=lambda item: _finalist_best_policy_sort_key(item, policy_name=policy_name))
    winner["policy"] = str(policy_name)
    winner["member_index"] = 1
    return [winner]


def _is_policy_ensemble_item(item: dict | None) -> bool:
    return isinstance(item, dict) and bool(normalize_seed_ensemble_members(item.get("params_ensemble")))


def _finalists_agree_policy_config(policy_name: str) -> dict:
    name = str(policy_name)
    if _is_local_finalists_agree_policy(name):
        return {
            "policy_name": LOCAL_FINALISTS_AGREE_POLICY_NAME,
            "label": "local_agree",
            "score_field": "local_min_score",
            "seed_score_sum_key": "local_agree_seed_local_min_sum",
            "seed_finalist_count_key": "local_agree_seed_finalist_count",
            "seed_selected_trials_key": "local_agree_seed_selected_trials",
            "min_agree_requested_key": "local_agree_min_agree_requested",
            "min_agree_requested": OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE,
            "selection_rule": "all_finalists_local_min_sum_best_seed_finalist_agree",
            "member_selection": "seed_with_max_all_finalists_local_min_sum",
            "ranker": _rank_local_finalist_items,
            "resolver": resolve_optimizer_local_finalists_agree_min_agree,
        }
    if _is_retention_finalists_agree_policy(name):
        return {
            "policy_name": RETENTION_FINALISTS_AGREE_POLICY_NAME,
            "label": "retention_agree",
            "score_field": "local_retention",
            "seed_score_sum_key": "retention_agree_seed_retention_sum",
            "seed_finalist_count_key": "retention_agree_seed_finalist_count",
            "seed_selected_trials_key": "retention_agree_seed_selected_trials",
            "min_agree_requested_key": "retention_agree_min_agree_requested",
            "min_agree_requested": OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE,
            "selection_rule": "all_finalists_retention_sum_best_seed_finalist_agree",
            "member_selection": "seed_with_max_all_finalists_retention_sum",
            "ranker": _rank_retention_finalist_items,
            "resolver": resolve_optimizer_retention_finalists_agree_min_agree,
        }
    return {
        "policy_name": BASE_FINALISTS_AGREE_POLICY_NAME,
        "label": "base_agree",
        "score_field": "base_score",
        "seed_score_sum_key": "base_agree_seed_base_score_sum",
        "seed_finalist_count_key": "base_agree_seed_finalist_count",
        "seed_selected_trials_key": "base_agree_seed_selected_trials",
        "min_agree_requested_key": "base_agree_min_agree_requested",
        "min_agree_requested": OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE,
        "selection_rule": "all_finalists_base_score_sum_best_seed_finalist_agree",
        "member_selection": "seed_with_max_all_finalists_base_score_sum",
        "ranker": _rank_base_finalist_items,
        "resolver": resolve_optimizer_base_finalists_agree_min_agree,
    }


def _resolve_finalists_agree_min_agree(policy_name: str, member_count: int) -> int:
    config = _finalists_agree_policy_config(policy_name)
    return int(config["resolver"](member_count, config["min_agree_requested"]))


def _finalists_agree_metadata(finalists: list[dict], *, policy_name: str) -> dict:
    config = _finalists_agree_policy_config(policy_name)
    ranked = config["ranker"](finalists)
    member_count = int(len(ranked))
    score_sum = float(sum(float(item.get(config["score_field"], INVALID_TRIAL_VALUE)) for item in ranked)) if ranked else 0.0
    selected_trials = [int(item["trial"].number) + 1 for item in ranked if item.get("trial") is not None]
    metadata = {
        "policy_type": "selected_seed_finalist_ensemble",
        "selection_rule": str(config["selection_rule"]),
        str(config["seed_finalist_count_key"]): int(member_count),
        str(config["seed_score_sum_key"]): float(score_sum),
        str(config["seed_selected_trials_key"]): selected_trials,
        str(config["min_agree_requested_key"]): config["min_agree_requested"],
        "member_selection": str(config["member_selection"]),
        "intra_seed_agree": "selected_seed_finalists",
        "member_count": int(member_count),
        "min_agree": _resolve_finalists_agree_min_agree(policy_name, member_count) if member_count > 0 else 0,
    }
    return metadata








def _build_finalists_agree_member_payloads(
    finalists: list[dict],
    *,
    policy_name: str,
    member_index: int = 1,
    seed: int | None = None,
    local_rank_map: dict[int, int] | None = None,
    retention_rank_map: dict[int, int] | None = None,
) -> list[dict]:
    config = _finalists_agree_policy_config(policy_name)
    ranked = config["ranker"](finalists)
    if not ranked:
        return []
    metadata = _finalists_agree_metadata(ranked, policy_name=policy_name)
    local_ranks = dict(local_rank_map or _build_local_rank_map(ranked))
    retention_ranks = dict(retention_rank_map or _build_retention_rank_map(ranked))
    members: list[dict] = []
    for offset, item in enumerate(ranked):
        trial = item.get("trial")
        if trial is None:
            continue
        trial_number = int(trial.number)
        member_payload = {
            "member_index": int(member_index) + int(offset),
            "seed": None if seed is None else int(seed),
            "policy": str(config["policy_name"]),
            "selected_trial": trial_number + 1,
            "optimizer_seed": None if seed is None else int(seed),
            "score": float(item.get(config["score_field"], INVALID_TRIAL_VALUE)),
            "base_score": float(item.get("base_score", INVALID_TRIAL_VALUE)),
            "base_rank": int(item.get("base_rank", 0) or 0),
            "local_min_score": float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            "local_min": float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            "local_min_review_enabled": bool(item.get("local_min_review_enabled", is_optimizer_local_min_review_enabled())),
            "local_min_exact": bool(item.get("local_min_exact", bool(is_optimizer_local_min_review_enabled()))),
            "local_min_review_mode": str(item.get("local_min_review_mode", "exact")),
            "local_rank": int(local_ranks.get(trial_number, 0)),
            "retention": float(item.get("local_retention", 0.0)),
            "retention_rank": int(retention_ranks.get(trial_number, 0)),
            "local_gate": bool(item.get("gate_pass", False)),
            "params": build_best_params_payload_from_trial(trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT),
        }
        member_payload.update(metadata)
        members.append(member_payload)
    return renumber_seed_ensemble_members(members)










def _safe_float_for_finalists_agree_sort(value, default: float = INVALID_TRIAL_VALUE) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int_for_finalists_agree_sort(value, default: int = 10**9) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _finalists_agree_seed_group_sort_key(group: list[dict], *, policy_name: str) -> tuple:
    config = _finalists_agree_policy_config(policy_name)
    members = [dict(member) for member in list(group or []) if isinstance(member, dict)]
    if not members:
        return (float("-inf"), float("-inf"), float("-inf"), 0, -10**9)
    seed_score_sum = max(
        _safe_float_for_finalists_agree_sort(member.get(config["seed_score_sum_key"]), INVALID_TRIAL_VALUE)
        for member in members
    )
    best_retention = max(_safe_float_for_finalists_agree_sort(member.get("retention"), float("-inf")) for member in members)
    best_local_min = max(_safe_float_for_finalists_agree_sort(member.get("local_min_score", member.get("local_min")), INVALID_TRIAL_VALUE) for member in members)
    best_base_score = max(_safe_float_for_finalists_agree_sort(member.get("base_score"), INVALID_TRIAL_VALUE) for member in members)
    best_base_rank = min(_safe_int_for_finalists_agree_sort(member.get("base_rank"), 10**9) for member in members)
    seed_values = [
        _safe_int_for_finalists_agree_sort(member.get("seed", member.get("optimizer_seed")), 10**9)
        for member in members
    ]
    seed_value = min(seed_values) if seed_values else 10**9
    if _is_local_finalists_agree_policy(policy_name):
        return (seed_score_sum, best_local_min, best_retention, best_base_score, -best_base_rank, -seed_value)
    if _is_retention_finalists_agree_policy(policy_name):
        return (seed_score_sum, best_retention, best_local_min, best_base_score, -best_base_rank, -seed_value)
    return (seed_score_sum, best_base_score, best_local_min, best_retention, -best_base_rank, -seed_value)


def _finalists_agree_member_order_key(member: dict, *, policy_name: str) -> tuple:
    data = dict(member or {})
    if _is_local_finalists_agree_policy(policy_name):
        return (
            _safe_int_for_finalists_agree_sort(data.get("local_rank"), 10**9),
            -_safe_float_for_finalists_agree_sort(data.get("local_min_score", data.get("local_min")), INVALID_TRIAL_VALUE),
            -_safe_float_for_finalists_agree_sort(data.get("retention"), float("-inf")),
            -_safe_float_for_finalists_agree_sort(data.get("base_score"), INVALID_TRIAL_VALUE),
            _safe_int_for_finalists_agree_sort(data.get("selected_trial"), 10**9),
            _safe_int_for_finalists_agree_sort(data.get("member_index"), 10**9),
        )
    if _is_retention_finalists_agree_policy(policy_name):
        return (
            _safe_int_for_finalists_agree_sort(data.get("retention_rank"), 10**9),
            -_safe_float_for_finalists_agree_sort(data.get("retention"), float("-inf")),
            -_safe_float_for_finalists_agree_sort(data.get("local_min_score", data.get("local_min")), INVALID_TRIAL_VALUE),
            -_safe_float_for_finalists_agree_sort(data.get("base_score"), INVALID_TRIAL_VALUE),
            _safe_int_for_finalists_agree_sort(data.get("selected_trial"), 10**9),
            _safe_int_for_finalists_agree_sort(data.get("member_index"), 10**9),
        )
    return (
        _safe_int_for_finalists_agree_sort(data.get("base_rank"), 10**9),
        -_safe_float_for_finalists_agree_sort(data.get("base_score"), INVALID_TRIAL_VALUE),
        _safe_int_for_finalists_agree_sort(data.get("selected_trial"), 10**9),
        _safe_int_for_finalists_agree_sort(data.get("member_index"), 10**9),
    )


def select_finalists_agree_members(members: list[dict], *, policy_name: str) -> list[dict]:
    if not _is_finalists_agree_policy(policy_name):
        return renumber_seed_ensemble_members(list(members or []))
    config = _finalists_agree_policy_config(policy_name)
    candidates = [dict(member) for member in list(members or []) if isinstance(member, dict) and dict(member).get("params")]
    if not candidates:
        return []
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for member in candidates:
        seed_key = member.get("seed", member.get("optimizer_seed"))
        if seed_key is None:
            seed_key = f"member:{member.get('member_index', len(groups) + 1)}"
        groups.setdefault(str(seed_key), []).append(member)
    selected_group = max(groups.values(), key=lambda group: _finalists_agree_seed_group_sort_key(group, policy_name=policy_name))
    selected_members = sorted((dict(member) for member in selected_group), key=lambda member: _finalists_agree_member_order_key(member, policy_name=policy_name))
    resolved_min_agree = _resolve_finalists_agree_min_agree(policy_name, len(selected_members))
    first_selected = dict(selected_members[0]) if selected_members else {}
    metadata = {key: first_selected.get(key) for key in _finalists_agree_metadata_keys() if key in first_selected}
    metadata.update({
        "selection_rule": str(config["selection_rule"]),
        "policy_type": "selected_seed_finalist_ensemble",
        "member_selection": str(config["member_selection"]),
        "intra_seed_agree": "selected_seed_finalists",
        "member_count": int(len(selected_members)),
        "min_agree": int(resolved_min_agree),
        str(config["min_agree_requested_key"]): config["min_agree_requested"],
    })
    for member in selected_members:
        member.update(metadata)
        member["policy"] = str(config["policy_name"])
    return renumber_seed_ensemble_members(selected_members)


def select_base_finalists_agree_members(members: list[dict]) -> list[dict]:
    return select_finalists_agree_members(members, policy_name=BASE_FINALISTS_AGREE_POLICY_NAME)


def select_local_finalists_agree_members(members: list[dict]) -> list[dict]:
    return select_finalists_agree_members(members, policy_name=LOCAL_FINALISTS_AGREE_POLICY_NAME)


def select_retention_finalists_agree_members(members: list[dict]) -> list[dict]:
    return select_finalists_agree_members(members, policy_name=RETENTION_FINALISTS_AGREE_POLICY_NAME)


def _select_finalists_agree_item(finalists: list[dict], *, policy_name: str) -> dict | None:
    config = _finalists_agree_policy_config(policy_name)
    ranked = config["ranker"](finalists)
    if not ranked:
        return None
    members = _build_finalists_agree_member_payloads(ranked, policy_name=policy_name)
    if not members:
        return None
    metadata = _finalists_agree_metadata(ranked, policy_name=policy_name)
    first_member = dict(members[0])
    item = {
        "params_ensemble": members,
        "params": dict(first_member.get("params") or {}),
        "trial": ranked[0].get("trial"),
        "base_score": float(first_member.get("base_score", INVALID_TRIAL_VALUE)),
        "base_rank": int(first_member.get("base_rank", 0) or 0),
        "local_min_score": float(first_member.get("local_min_score", INVALID_TRIAL_VALUE)),
        "local_retention": float(first_member.get("retention", 0.0)),
    }
    item.update(metadata)
    return item


def _select_base_finalists_agree_item(finalists: list[dict]) -> dict | None:
    return _select_finalists_agree_item(finalists, policy_name=BASE_FINALISTS_AGREE_POLICY_NAME)


def _select_local_finalists_agree_item(finalists: list[dict]) -> dict | None:
    return _select_finalists_agree_item(finalists, policy_name=LOCAL_FINALISTS_AGREE_POLICY_NAME)


def _select_retention_finalists_agree_item(finalists: list[dict]) -> dict | None:
    return _select_finalists_agree_item(finalists, policy_name=RETENTION_FINALISTS_AGREE_POLICY_NAME)


def _select_local_rank1_item(finalists: list[dict], *, objective_mode: str):
    winner = _select_winner(finalists, objective_mode=objective_mode)
    if winner is not None:
        return winner
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    return items[0] if items else None


def _select_retention_rank1_item(finalists: list[dict]):
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    if not items:
        return None
    return max(
        items,
        key=lambda item: (
            float(item.get("local_retention", float("-inf"))),
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
    )


def _build_policy_items(finalists: list[dict], *, objective_mode: str) -> dict[str, dict | None]:
    base_rank1 = _select_base_rank1_item(finalists)
    local_rank1 = _select_local_rank1_item(finalists, objective_mode=objective_mode)
    retention_rank1 = _select_retention_rank1_item(finalists)
    items = {
        BASE_FINALIST_BEST_POLICY_NAME: base_rank1,
        LOCAL_FINALIST_BEST_POLICY_NAME: local_rank1,
        RETENTION_FINALIST_BEST_POLICY_NAME: retention_rank1,
        BASE_FINALISTS_AGREE_POLICY_NAME: _select_base_finalists_agree_item(finalists),
        LOCAL_FINALISTS_AGREE_POLICY_NAME: _select_local_finalists_agree_item(finalists),
        RETENTION_FINALISTS_AGREE_POLICY_NAME: _select_retention_finalists_agree_item(finalists),
        "base": base_rank1,
        "local": local_rank1,
        "retention": retention_rank1,
    }
    return {name: item for name, item in items.items() if name in set(REPORT_POLICY_NAMES)}


def get_optimizer_paramset_policy_names() -> tuple[str, ...]:
    """Return the policy set that writes first-class optimizer param JSON files."""
    return tuple(REPORT_POLICY_NAMES)


def get_optimizer_policy_paramset_filename(policy_name: str) -> str:
    """Return the canonical JSON filename for a rolling-OOS optimizer policy paramset."""
    return str(PARAMSET_FILENAME_BY_POLICY.get(str(policy_name), f"roos_{policy_name}.json"))


def get_optimizer_nonrolling_policy_paramset_filename(policy_name: str, *, mode: str | None = None) -> str:
    """Return the canonical JSON filename for a non-rolling optimizer policy paramset.

    Study mode keeps the legacy first-class filenames (base/local/retention) as
    the single-seed study artifact.  Prefixed modes put the aggregation dimension
    before the policy target, e.g. ``full_ensemble_base.json`` and
    ``oos_ensemble_base.json``, so console labels and filenames use the same
    ``<mode>_<method>_<target>.json`` contract.
    """
    policy_key = str(policy_name)
    base_filename = str(NONROLLING_PARAMSET_FILENAME_BY_POLICY.get(policy_key, f"{policy_key}.json"))
    normalized_mode = str(mode or "study").strip().lower()
    if normalized_mode == "split":
        normalized_mode = "oos"
    if normalized_mode != "study" and policy_key in {"base", "local", "retention"}:
        return f"{normalized_mode}_ensemble_{policy_key}.json"
    prefix = str(NONROLLING_PARAMSET_FILENAME_PREFIX_BY_MODE.get(normalized_mode, ""))
    if prefix == "":
        return base_filename
    return f"{prefix}{base_filename}"


def get_optimizer_policy_output_label(policy_name: str) -> str:
    """Return the canonical console label for an optimizer policy paramset."""
    return str(POLICY_OUTPUT_LABELS.get(str(policy_name), str(policy_name)))


def build_optimizer_policy_members_from_finalists(
    finalists: list[dict],
    *,
    objective_mode: str,
    member_index: int,
    seed: int | None = None,
) -> dict[str, dict | list[dict]]:
    """Build static active-param ensemble members with the same policy selectors as rolling OOS.

    Non-rolling training is a single-fold case, so it must reuse rolling's policy
    selection rules instead of re-implementing base/local/retention choices.
    """
    policy_items = _build_policy_items(finalists, objective_mode=objective_mode)
    local_rank_map = _build_local_rank_map(finalists)
    retention_rank_map = _build_retention_rank_map(finalists)
    members: dict[str, dict | list[dict]] = {}
    for policy_name in CHAIN_POLICY_NAMES:
        item = policy_items.get(policy_name)
        if item is None:
            continue
        if _is_finalists_agree_policy(policy_name):
            finalists_agree_members = normalize_seed_ensemble_members(item.get("params_ensemble"))
            if not finalists_agree_members:
                finalists_agree_members = _build_finalists_agree_member_payloads(
                    finalists,
                    policy_name=str(policy_name),
                    member_index=int(member_index),
                    seed=seed,
                    local_rank_map=local_rank_map,
                    retention_rank_map=retention_rank_map,
                )
            for offset, raw_member in enumerate(finalists_agree_members):
                member_payload = dict(raw_member)
                member_payload["member_index"] = int(member_index) + int(offset)
                member_payload["seed"] = None if seed is None else int(seed)
                member_payload["optimizer_seed"] = None if seed is None else int(seed)
                member_payload["policy"] = str(policy_name)
                existing_members = members.setdefault(str(policy_name), [])
                if isinstance(existing_members, list):
                    existing_members.append(member_payload)
            continue
        if item.get("trial") is None:
            continue
        trial = item["trial"]
        trial_number = int(trial.number)
        params_payload = build_best_params_payload_from_trial(trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
        member_payload = {
            "member_index": int(member_index),
            "seed": None if seed is None else int(seed),
            "policy": str(policy_name),
            "selected_trial": trial_number + 1,
            "optimizer_seed": None if seed is None else int(seed),
            "score": float(item.get("base_score", INVALID_TRIAL_VALUE)),
            "base_score": float(item.get("base_score", INVALID_TRIAL_VALUE)),
            "base_rank": int(item.get("base_rank", 0) or 0),
            "local_min_score": float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            "local_min": float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            "local_min_review_enabled": bool(item.get("local_min_review_enabled", is_optimizer_local_min_review_enabled())),
            "local_min_exact": bool(item.get("local_min_exact", bool(is_optimizer_local_min_review_enabled()))),
            "local_min_review_mode": str(item.get("local_min_review_mode", "exact")),
            "local_rank": int(local_rank_map.get(trial_number, 0)),
            "retention": float(item.get("local_retention", 0.0)),
            "retention_rank": int(retention_rank_map.get(trial_number, 0)),
            "local_gate": bool(item.get("gate_pass", False)),
            "params": dict(params_payload),
        }
        if _is_finalists_agree_policy(policy_name):
            for key in _finalists_agree_metadata_keys():
                if key in item:
                    member_payload[key] = item.get(key)
        members[str(policy_name)] = member_payload
    return members



def _finalists_agree_metadata_keys() -> tuple[str, ...]:
    return (
        "selection_rule",
        "policy_type",
        "member_selection",
        "intra_seed_agree",
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
    )



def _build_policy_schedule_entry(
    *,
    item: dict,
    policy_name: str,
    oos_year: int,
    selection_period: str,
    local_rank_map: dict[int, int],
    retention_rank_map: dict[int, int],
    oos_start_date: str | None = None,
    oos_end_date: str | None = None,
    optimizer_seed: int | None = None,
    fixed_strategy_param_overrides: dict | None = None,
    fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
) -> dict:
    effective_start = str(oos_start_date or f"{str(oos_year)[:4]}-01-01")
    effective_end = str(oos_end_date or f"{str(oos_year)[:4]}-12-31")
    ensemble_members = _materialize_fixed_strategy_param_overrides_in_members(
        item.get("params_ensemble"),
        fixed_strategy_param_overrides,
    )
    if ensemble_members:
        if optimizer_seed is not None:
            for member in ensemble_members:
                if member.get("seed") is None:
                    member["seed"] = int(optimizer_seed)
                if member.get("optimizer_seed") is None:
                    member["optimizer_seed"] = int(optimizer_seed)
        optimizer_seed_values = sorted({member.get("seed") for member in ensemble_members if member.get("seed") is not None})
        if not optimizer_seed_values and optimizer_seed is not None:
            optimizer_seed_values = [int(optimizer_seed)]
        first_member = dict(ensemble_members[0])
        return {
            "effective_start": effective_start,
            "effective_end": effective_end,
            "selection": str(selection_period),
            "oos_year": int(oos_year),
            "oos_period": f"{effective_start}~{effective_end}",
            "policy": str(policy_name),
            "selected_trial": first_member.get("selected_trial"),
            "optimizer_seed": None if optimizer_seed is None else int(optimizer_seed),
            "optimizer_seeds": optimizer_seed_values,
            "member_count": int(len(ensemble_members)),
            "min_agree": int(item.get("min_agree") or (_resolve_finalists_agree_min_agree(policy_name, len(ensemble_members)) if _is_finalists_agree_policy(policy_name) else 1)),
            "min_agree_requested": item.get("min_agree_requested", item.get("base_agree_min_agree_requested", item.get("local_agree_min_agree_requested", item.get("retention_agree_min_agree_requested")))),
            "base_agree_min_agree_requested": item.get("base_agree_min_agree_requested"),
            "local_agree_min_agree_requested": item.get("local_agree_min_agree_requested"),
            "retention_agree_min_agree_requested": item.get("retention_agree_min_agree_requested"),
            "base_score": float(item.get("base_score", first_member.get("base_score", INVALID_TRIAL_VALUE))),
            "base_rank": int(item.get("base_rank", first_member.get("base_rank", 0)) or 0),
            "local_min": float(item.get("local_min_score", first_member.get("local_min_score", INVALID_TRIAL_VALUE))),
            "local_min_review_enabled": bool(item.get("local_min_review_enabled", is_optimizer_local_min_review_enabled())),
            "local_min_exact": bool(item.get("local_min_exact", bool(is_optimizer_local_min_review_enabled()))),
            "local_min_review_mode": str(item.get("local_min_review_mode", "exact")),
            "retention": float(item.get("local_retention", first_member.get("retention", 0.0))),
            "params": dict(first_member.get("params") or {}),
            "params_ensemble": ensemble_members,
            "selection_rule": item.get("selection_rule", first_member.get("selection_rule")),
            "policy_type": item.get("policy_type", first_member.get("policy_type")),
            "base_agree_seed_finalist_count": item.get("base_agree_seed_finalist_count", first_member.get("base_agree_seed_finalist_count")),
            "base_agree_seed_base_score_sum": item.get("base_agree_seed_base_score_sum", first_member.get("base_agree_seed_base_score_sum")),
            "base_agree_seed_selected_trials": item.get("base_agree_seed_selected_trials", first_member.get("base_agree_seed_selected_trials")),
            "local_agree_seed_finalist_count": item.get("local_agree_seed_finalist_count", first_member.get("local_agree_seed_finalist_count")),
            "local_agree_seed_local_min_sum": item.get("local_agree_seed_local_min_sum", first_member.get("local_agree_seed_local_min_sum")),
            "local_agree_seed_selected_trials": item.get("local_agree_seed_selected_trials", first_member.get("local_agree_seed_selected_trials")),
            "retention_agree_seed_finalist_count": item.get("retention_agree_seed_finalist_count", first_member.get("retention_agree_seed_finalist_count")),
            "retention_agree_seed_retention_sum": item.get("retention_agree_seed_retention_sum", first_member.get("retention_agree_seed_retention_sum")),
            "retention_agree_seed_selected_trials": item.get("retention_agree_seed_selected_trials", first_member.get("retention_agree_seed_selected_trials")),
        }
    trial = item["trial"]
    trial_number = int(trial.number)
    entry = {
        "effective_start": effective_start,
        "effective_end": effective_end,
        "selection": str(selection_period),
        "oos_year": int(oos_year),
        "oos_period": f"{effective_start}~{effective_end}",
        "policy": str(policy_name),
        "selected_trial": trial_number + 1,
        "optimizer_seed": None if optimizer_seed is None else int(optimizer_seed),
        "base_score": float(item.get("base_score", INVALID_TRIAL_VALUE)),
        "base_rank": int(item.get("base_rank", 0) or 0),
        "local_min": float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
        "local_min_review_enabled": bool(item.get("local_min_review_enabled", is_optimizer_local_min_review_enabled())),
        "local_min_exact": bool(item.get("local_min_exact", bool(is_optimizer_local_min_review_enabled()))),
        "local_min_review_mode": str(item.get("local_min_review_mode", "exact")),
        "local_rank": int(local_rank_map.get(trial_number, 0)),
        "retention": float(item.get("local_retention", 0.0)),
        "retention_rank": int(retention_rank_map.get(trial_number, 0)),
        "params": build_best_params_payload_from_trial(
            trial,
            fixed_tp_percent=fixed_tp_percent,
            fixed_strategy_param_overrides=fixed_strategy_param_overrides,
        ),
    }
    if _is_finalists_agree_policy(policy_name):
        for key in _finalists_agree_metadata_keys():
            if key in item:
                entry[key] = item.get(key)
    return entry

def _policy_is_available(policy_row: dict) -> bool:
    if not policy_row:
        return False
    if policy_row.get("available") is False:
        return False
    try:
        float(policy_row.get("rank_1_oos"))
    except (TypeError, ValueError):
        return False
    return True

def _policy_plain_romd_score(policy_row: dict) -> float:
    payload = dict(policy_row or {})
    for key in ("rank_1_plain_romd", "rank_1_romd", "rank_1_oos_romd", "plain_romd_score"):
        if key not in payload:
            continue
        try:
            value = float(payload.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return calc_plain_romd(payload.get("rank_1_return_pct", 0.0), payload.get("rank_1_mdd_pct", 0.0))
