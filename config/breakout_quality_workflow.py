"""User-adjustable orchestration settings for breakout-quality research workflows.

This module describes which existing model/profile/artifacts a workflow uses. It does not
redefine model architecture, labels, training mathematics, or portfolio accounting rules.
The selected experiment profile is the single source of truth for routing the interactive
model workflow to binary classification or continuous point-in-time ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.breakout_quality_experiments import (
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    get_breakout_quality_experiment_profile,
)
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)


WORKFLOW_STRATEGY_MODE_AUTO = "auto"
WORKFLOW_STRATEGY_MODE_HARD_FILTER = "hard-filter"
WORKFLOW_STRATEGY_MODE_SCORE_RANKING = "score-ranking"
SUPPORTED_WORKFLOW_STRATEGY_MODES = (
    WORKFLOW_STRATEGY_MODE_AUTO,
    WORKFLOW_STRATEGY_MODE_HARD_FILTER,
    WORKFLOW_STRATEGY_MODE_SCORE_RANKING,
)

WORKFLOW_SCORE_SOURCE_AUTO = "auto"
WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME = "selection_point_in_time"
WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS = "final_selection_model_oos"
WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME = "canonical_runtime"
SUPPORTED_WORKFLOW_SCORE_SOURCES = (
    WORKFLOW_SCORE_SOURCE_AUTO,
    WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS,
    WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
)

WORKFLOW_BUY_SORT_AUTO = "auto"
WORKFLOW_BUY_SORT_ORIGINAL = "original"
WORKFLOW_BUY_SORT_SCORE_DESC = "breakout_quality_score_desc"


# =============================================================================
# 1. Model research workflow identity
# =============================================================================

BREAKOUT_QUALITY_WORKFLOW_FILTER_ID = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE = BREAKOUT_QUALITY_MODEL_ARCHITECTURE
BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE = (
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
)
BREAKOUT_QUALITY_WORKFLOW_SEED = 1


# =============================================================================
# 2. Selection point-in-time score contract
# =============================================================================

# These settings are used only when the selected experiment profile is a continuous ranker.
# The first score date may be later than the model dataset Selection start so earlier history can
# train the first fold. None means use the canonical Selection end from the outer split policy.
BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE = "2014-01-01"
BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE: str | None = None
BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS = 12
BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS = 24
BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS = 20
BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS = 20
BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS = 1
BREAKOUT_QUALITY_POINT_IN_TIME_RESUME = True


# =============================================================================
# 3. Strategy workflow defaults
# =============================================================================

BREAKOUT_QUALITY_STRATEGY_DATASET = "full"
BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY = "base-finalist-best"
BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS = 10
BREAKOUT_QUALITY_STRATEGY_ROTATION = "off"

# "auto" resolves from the selected experiment profile:
# - binary classification -> hard-filter / canonical_runtime / original
# - continuous ranker     -> score-ranking / selection_point_in_time /
#                            breakout_quality_score_desc
BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE = WORKFLOW_STRATEGY_MODE_AUTO
BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE = WORKFLOW_SCORE_SOURCE_AUTO
BREAKOUT_QUALITY_STRATEGY_BUY_SORT = WORKFLOW_BUY_SORT_AUTO


@dataclass(frozen=True)
class BreakoutQualityWorkflowSettings:
    filter_id: str
    model_architecture: str
    experiment_profile: str
    training_objective: str
    continuous_target_id: str | None
    training_label_scope: str
    seed: int
    point_in_time_score_start_date: str
    point_in_time_score_end_date: str | None
    point_in_time_fold_months: int
    point_in_time_inner_validation_months: int
    point_in_time_min_train_groups: int
    point_in_time_min_validation_groups: int
    point_in_time_min_score_groups: int
    point_in_time_resume: bool
    strategy_dataset: str
    strategy_param_policy: str
    strategy_max_positions: int
    strategy_rotation: str
    strategy_comparison_mode: str
    strategy_score_source: str
    strategy_buy_sort: str

    @property
    def is_binary_classification(self) -> bool:
        return self.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION

    @property
    def is_continuous_ranker(self) -> bool:
        return self.training_objective == TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION

    def as_manifest_payload(self) -> dict[str, Any]:
        return {
            "filter_id": self.filter_id,
            "model_architecture": self.model_architecture,
            "experiment_profile": self.experiment_profile,
            "training_objective": self.training_objective,
            "continuous_target_id": self.continuous_target_id,
            "training_label_scope": self.training_label_scope,
            "seed": int(self.seed),
            "point_in_time": {
                "enabled": bool(self.is_continuous_ranker),
                "score_start_date": self.point_in_time_score_start_date,
                "score_end_date": self.point_in_time_score_end_date,
                "fold_months": int(self.point_in_time_fold_months),
                "inner_validation_months": int(
                    self.point_in_time_inner_validation_months
                ),
                "min_train_groups": int(self.point_in_time_min_train_groups),
                "min_validation_groups": int(
                    self.point_in_time_min_validation_groups
                ),
                "min_score_groups": int(self.point_in_time_min_score_groups),
                "resume": bool(self.point_in_time_resume),
            },
            "strategy": {
                "dataset": self.strategy_dataset,
                "param_policy": self.strategy_param_policy,
                "max_positions": int(self.strategy_max_positions),
                "rotation": self.strategy_rotation,
                "comparison_mode": self.strategy_comparison_mode,
                "score_source": self.strategy_score_source,
                "buy_sort": self.strategy_buy_sort,
            },
        }


def _resolve_strategy_defaults(training_objective: str) -> tuple[str, str, str]:
    if training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
        return (
            WORKFLOW_STRATEGY_MODE_HARD_FILTER,
            WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
            WORKFLOW_BUY_SORT_ORIGINAL,
        )
    if training_objective == TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION:
        return (
            WORKFLOW_STRATEGY_MODE_SCORE_RANKING,
            WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            WORKFLOW_BUY_SORT_SCORE_DESC,
        )
    raise ValueError(f"不支援的 workflow training objective: {training_objective!r}")


def _resolve_auto(value: str, *, auto_value: str, resolved_default: str) -> str:
    normalized = str(value).strip()
    return resolved_default if normalized == auto_value else normalized


def get_breakout_quality_workflow_settings() -> BreakoutQualityWorkflowSettings:
    profile = get_breakout_quality_experiment_profile(
        BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE
    )
    if profile.training_objective not in {
        TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
        TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    }:
        raise ValueError(
            "workflow experiment profile必須是binary classification或continuous ranker"
        )
    if int(BREAKOUT_QUALITY_WORKFLOW_SEED) < 0:
        raise ValueError("workflow seed 必須 >= 0")
    if int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS) < 1:
        raise ValueError("point-in-time fold months 必須 >= 1")
    if int(BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS) < 1:
        raise ValueError("point-in-time inner validation months 必須 >= 1")
    if min(
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS),
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS),
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS),
    ) < 1:
        raise ValueError("point-in-time minimum group counts 必須 >= 1")
    if BREAKOUT_QUALITY_STRATEGY_DATASET not in {"reduced", "full"}:
        raise ValueError("strategy dataset 必須是 reduced 或 full")
    if BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY not in {
        "auto",
        "base-finalist-best",
        "base-finalists-agree",
    }:
        raise ValueError(
            "strategy param policy 必須是auto、base-finalist-best或base-finalists-agree"
        )
    if int(BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS) < 1:
        raise ValueError("strategy max positions 必須 >= 1")
    if BREAKOUT_QUALITY_STRATEGY_ROTATION not in {"off", "on"}:
        raise ValueError("strategy rotation 必須是 off 或 on")

    raw_comparison_mode = str(BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE).strip()
    if raw_comparison_mode not in SUPPORTED_WORKFLOW_STRATEGY_MODES:
        raise ValueError(
            "strategy comparison mode 必須是auto、hard-filter或score-ranking"
        )
    raw_score_source = str(BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE).strip()
    if raw_score_source not in SUPPORTED_WORKFLOW_SCORE_SOURCES:
        raise ValueError(
            "strategy score source 必須是auto、selection_point_in_time、"
            "final_selection_model_oos或canonical_runtime"
        )
    raw_buy_sort = str(BREAKOUT_QUALITY_STRATEGY_BUY_SORT).strip()
    if not raw_buy_sort:
        raise ValueError("strategy buy sort不可為空白")

    default_mode, default_score_source, default_buy_sort = _resolve_strategy_defaults(
        profile.training_objective
    )
    strategy_comparison_mode = _resolve_auto(
        raw_comparison_mode,
        auto_value=WORKFLOW_STRATEGY_MODE_AUTO,
        resolved_default=default_mode,
    )
    strategy_score_source = _resolve_auto(
        raw_score_source,
        auto_value=WORKFLOW_SCORE_SOURCE_AUTO,
        resolved_default=default_score_source,
    )
    strategy_buy_sort = _resolve_auto(
        raw_buy_sort,
        auto_value=WORKFLOW_BUY_SORT_AUTO,
        resolved_default=default_buy_sort,
    )

    if strategy_comparison_mode == WORKFLOW_STRATEGY_MODE_HARD_FILTER:
        if strategy_score_source != WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME:
            raise ValueError("hard-filter策略比較只接受canonical_runtime score source")
        if strategy_buy_sort != WORKFLOW_BUY_SORT_ORIGINAL:
            raise ValueError("hard-filter策略比較必須沿用original buy-sort")
    elif strategy_comparison_mode == WORKFLOW_STRATEGY_MODE_SCORE_RANKING:
        if strategy_score_source not in {
            WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS,
            WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
        }:
            raise ValueError("score-ranking策略比較缺少合法score source")
        if strategy_buy_sort != WORKFLOW_BUY_SORT_SCORE_DESC:
            raise ValueError(
                "score-ranking策略比較目前只支援breakout_quality_score_desc"
            )
    else:
        raise ValueError(
            f"不支援的 resolved strategy comparison mode: {strategy_comparison_mode!r}"
        )

    return BreakoutQualityWorkflowSettings(
        filter_id=str(BREAKOUT_QUALITY_WORKFLOW_FILTER_ID),
        model_architecture=str(BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE),
        experiment_profile=str(BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE),
        training_objective=str(profile.training_objective),
        continuous_target_id=(
            None
            if profile.continuous_target_id is None
            else str(profile.continuous_target_id)
        ),
        training_label_scope=str(profile.training_label_scope),
        seed=int(BREAKOUT_QUALITY_WORKFLOW_SEED),
        point_in_time_score_start_date=str(
            BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE
        ),
        point_in_time_score_end_date=(
            None
            if BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE is None
            else str(BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE)
        ),
        point_in_time_fold_months=int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS),
        point_in_time_inner_validation_months=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ),
        point_in_time_min_train_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS
        ),
        point_in_time_min_validation_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS
        ),
        point_in_time_min_score_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS
        ),
        point_in_time_resume=bool(BREAKOUT_QUALITY_POINT_IN_TIME_RESUME),
        strategy_dataset=str(BREAKOUT_QUALITY_STRATEGY_DATASET),
        strategy_param_policy=str(BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY),
        strategy_max_positions=int(BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS),
        strategy_rotation=str(BREAKOUT_QUALITY_STRATEGY_ROTATION),
        strategy_comparison_mode=strategy_comparison_mode,
        strategy_score_source=strategy_score_source,
        strategy_buy_sort=strategy_buy_sort,
    )


__all__ = [
    "BREAKOUT_QUALITY_STRATEGY_BUY_SORT",
    "BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE",
    "BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE",
    "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
    "BreakoutQualityWorkflowSettings",
    "SUPPORTED_WORKFLOW_SCORE_SOURCES",
    "SUPPORTED_WORKFLOW_STRATEGY_MODES",
    "WORKFLOW_BUY_SORT_AUTO",
    "WORKFLOW_BUY_SORT_ORIGINAL",
    "WORKFLOW_BUY_SORT_SCORE_DESC",
    "WORKFLOW_SCORE_SOURCE_AUTO",
    "WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME",
    "WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS",
    "WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME",
    "WORKFLOW_STRATEGY_MODE_AUTO",
    "WORKFLOW_STRATEGY_MODE_HARD_FILTER",
    "WORKFLOW_STRATEGY_MODE_SCORE_RANKING",
    "get_breakout_quality_workflow_settings",
]
