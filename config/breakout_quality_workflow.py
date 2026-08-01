"""User-adjustable orchestration settings for breakout-quality research workflows.

This module describes which existing model/profile/artifacts a workflow uses.  It does not
redefine model architecture, labels, training mathematics, or portfolio accounting rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.breakout_quality_experiments import (
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    get_breakout_quality_experiment_profile,
)
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)


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

# The first score date may be later than the model dataset Selection start so earlier history can
# train the first fold.  None means use the canonical Selection end from the outer split policy.
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
BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE = "selection_point_in_time"
BREAKOUT_QUALITY_STRATEGY_BUY_SORT = "breakout_quality_score_desc"


@dataclass(frozen=True)
class BreakoutQualityWorkflowSettings:
    filter_id: str
    model_architecture: str
    experiment_profile: str
    continuous_target_id: str
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
    strategy_score_source: str
    strategy_buy_sort: str

    def as_manifest_payload(self) -> dict[str, Any]:
        return {
            "filter_id": self.filter_id,
            "model_architecture": self.model_architecture,
            "experiment_profile": self.experiment_profile,
            "continuous_target_id": self.continuous_target_id,
            "training_label_scope": self.training_label_scope,
            "seed": int(self.seed),
            "point_in_time": {
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
                "score_source": self.strategy_score_source,
                "buy_sort": self.strategy_buy_sort,
            },
        }


def get_breakout_quality_workflow_settings() -> BreakoutQualityWorkflowSettings:
    profile = get_breakout_quality_experiment_profile(
        BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE
    )
    if profile.training_objective != TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION:
        raise ValueError("workflow experiment profile 必須是 continuous ranker")
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
    if BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE not in {
        "selection_point_in_time",
        "final_selection_model_oos",
        "canonical_runtime",
    }:
        raise ValueError(
            "strategy score source 必須是selection_point_in_time、"
            "final_selection_model_oos或canonical_runtime"
        )
    if not str(BREAKOUT_QUALITY_STRATEGY_BUY_SORT).strip():
        raise ValueError("strategy buy sort不可為空白")

    return BreakoutQualityWorkflowSettings(
        filter_id=str(BREAKOUT_QUALITY_WORKFLOW_FILTER_ID),
        model_architecture=str(BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE),
        experiment_profile=str(BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE),
        continuous_target_id=str(profile.continuous_target_id),
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
        strategy_score_source=str(BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE),
        strategy_buy_sort=str(BREAKOUT_QUALITY_STRATEGY_BUY_SORT),
    )


__all__ = [
    "BreakoutQualityWorkflowSettings",
    "get_breakout_quality_workflow_settings",
]
