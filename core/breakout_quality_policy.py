"""Derived breakout-quality settings and runtime/workflow resolvers.

This module interprets declarative values from ``config.breakout_quality`` against the
scientific/profile SSOT in ``core.breakout_quality_registry``.  It does not own current
user selections or scientific identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from config.breakout_policy import build_breakout_optimizer_high_len_values
from config.execution_policy import DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT
from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
import config.breakout_quality as cfg
from core.breakout_quality_registry import (
    CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
    SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLE_SCOPES,
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
    get_breakout_quality_pretraining_profile,
    get_continuous_ranker_research_spec,
    normalize_breakout_quality_experiment_profile,
)
from core.breakout_quality_runtime import (
    ContinuousRankerExecutionRecipe,
    build_continuous_ranker_execution_recipe,
)

def get_breakout_quality_model_workflow_profile_names() -> tuple[str, ...]:
    """Return the single configured target set for current model workflows.

    [1]/[2] consume ``BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE`` while
    [3]～[6] consume ``BREAKOUT_QUALITY_MODEL_TEST_PROFILES``.  Membership in either
    current work-item SSOT is itself the authorization for current Forward/Rolling
    model evaluation; experiment-history flags remain compatibility evidence rather
    than a second current-workflow selector.
    """

    ordered = [str(cfg.BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE).strip()]
    ordered.extend(str(profile).strip() for _model_id, profile in cfg.BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
    return tuple(dict.fromkeys(value for value in ordered if value))


def is_breakout_quality_model_workflow_profile(profile_name: str) -> bool:
    normalized = normalize_breakout_quality_experiment_profile(profile_name)
    return normalized in set(get_breakout_quality_model_workflow_profile_names())


def is_breakout_quality_model_test_profile(profile_name: str) -> bool:
    """Whether a profile is selected by the one [3]～[6] compare/test SSOT."""

    normalized = normalize_breakout_quality_experiment_profile(profile_name)
    return normalized in {
        normalize_breakout_quality_experiment_profile(profile)
        for _model_id, profile in cfg.BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    }


def resolve_breakout_quality_random_seed() -> int:
    """Return the single configured breakout-quality random seed."""

    resolved = int(cfg.BREAKOUT_QUALITY_RANDOM_SEED)
    if resolved < 0:
        raise ValueError("breakout quality random seed 必須是>=0的整數")
    return resolved

# =============================================================================
# DERIVED VALUES AND HELPER FUNCTIONS — do not edit unless changing implementation
# =============================================================================

# Workflow filter and architecture intentionally follow the active canonical identity.
# Strategy tools use the validated workflow identity; the model-research menu has its own active profile above.
BREAKOUT_QUALITY_WORKFLOW_FILTER_ID = cfg.BREAKOUT_QUALITY_DEFAULT_FILTER_ID
BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE = cfg.BREAKOUT_QUALITY_MODEL_ARCHITECTURE


# =============================================================================
# 9. Derived values and helper calculations (not user-adjustable)
# =============================================================================

# 下列 pretraining 欄位由具名 profile 自動展開，請調整 profile 定義而非直接修改衍生值。
_BREAKOUT_QUALITY_PRETRAINING_SETTINGS = get_breakout_quality_pretraining_profile(
    cfg.BREAKOUT_QUALITY_PRETRAINING_PROFILE
)
BREAKOUT_QUALITY_PRETRAINING_FAMILY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.family
BREAKOUT_QUALITY_PRETRAINING_EPOCHS = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.epochs
BREAKOUT_QUALITY_PRETRAINING_BATCH_SIZE = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.batch_size
BREAKOUT_QUALITY_PRETRAINING_LEARNING_RATE = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.learning_rate
BREAKOUT_QUALITY_PRETRAINING_WEIGHT_DECAY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.weight_decay
BREAKOUT_QUALITY_PRETRAINING_GRADIENT_CLIP_NORM = (
    _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.gradient_clip_norm
)
BREAKOUT_QUALITY_PRETRAINING_MIN_CROP_BARS = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.min_crop_bars
BREAKOUT_QUALITY_PRETRAINING_MASK_PROBABILITY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.mask_probability
BREAKOUT_QUALITY_PRETRAINING_CONTRASTIVE_ALPHA = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.contrastive_alpha
BREAKOUT_QUALITY_PRETRAINING_TEMPORAL_UNIT = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.temporal_unit


def build_breakout_quality_default_high_len_values() -> tuple[int, ...]:
    values = set(build_breakout_optimizer_high_len_values())
    for value in cfg.BREAKOUT_QUALITY_EXTRA_HIGH_LENS:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("BREAKOUT_QUALITY_EXTRA_HIGH_LENS 只能包含正整數")
        values.add(normalized)
    return tuple(sorted(values))


def _nearest_positive_odd(value: float, *, minimum: int = 3) -> int:
    lower = max(int(minimum), int(math.floor(float(value))))
    if lower % 2 == 0:
        lower -= 1
    lower = max(int(minimum), lower)
    if lower % 2 == 0:
        lower += 1
    upper = lower + 2
    if abs(float(value) - lower) <= abs(upper - float(value)):
        return int(lower)
    return int(upper)


def build_breakout_quality_inception_kernel_sizes() -> tuple[int, int, int]:
    depth = int(cfg.BREAKOUT_QUALITY_INCEPTION_DEPTH)
    target = int(cfg.BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS)
    residual_every = int(cfg.BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY)
    feature_window = int(cfg.BREAKOUT_QUALITY_FEATURE_WINDOW_BARS)
    if depth < 1:
        raise ValueError("BREAKOUT_QUALITY_INCEPTION_DEPTH 必須 >= 1")
    if residual_every < 1 or depth % residual_every != 0:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY 必須 >= 1，且必須整除 INCEPTION_DEPTH"
        )
    minimum_target = 1 + depth * 8
    if target < minimum_target:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS 過小；"
            f"depth={depth} 時至少需要 {minimum_target} bars，才能保留三尺度 kernels"
        )
    if target > feature_window:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS 不得大於 "
            "BREAKOUT_QUALITY_FEATURE_WINDOW_BARS；模型不能從不存在的更早歷史學習"
        )

    minimum_max_kernel = math.ceil(1.0 + (target - 1) / depth)
    max_kernel = int(minimum_max_kernel)
    if max_kernel % 2 == 0:
        max_kernel += 1
    middle_kernel = _nearest_positive_odd(max_kernel / 2.0)
    short_kernel = _nearest_positive_odd(max_kernel / 4.0)
    kernels = (int(max_kernel), int(middle_kernel), int(short_kernel))
    if len(set(kernels)) != 3 or not (kernels[0] > kernels[1] > kernels[2] >= 3):
        raise ValueError(f"InceptionTime 自動產生的 kernels 不合法: {kernels}")
    return kernels


def resolve_breakout_quality_inception_receptive_field_bars() -> int:
    kernels = build_breakout_quality_inception_kernel_sizes()
    return 1 + int(cfg.BREAKOUT_QUALITY_INCEPTION_DEPTH) * (max(kernels) - 1)

def get_continuous_ranker_execution_recipe(
    experiment_profile: str,
) -> ContinuousRankerExecutionRecipe:
    """Resolve declarative profile identity into the canonical runtime contract."""

    profile_name = normalize_breakout_quality_experiment_profile(experiment_profile)
    profile = get_breakout_quality_experiment_profile(profile_name)
    spec = get_continuous_ranker_research_spec(profile_name)
    if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        raise ValueError(f"continuous ranker recipe只接受continuous profile: {profile_name}")
    return build_continuous_ranker_execution_recipe(
        profile_name=profile_name,
        profile=profile,
        spec=spec,
        requires_continuous_target_artifact=(
            profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
        ),
    )


@dataclass(frozen=True)
class BreakoutQualityContinuousRankerComparisonSettings:
    enabled: bool
    menu_label: str
    model_profiles: tuple[tuple[str, str], ...]
    reference_arm: str
    summary_pair: tuple[str, str]
    fixed_k_values: tuple[int, ...]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model_id for model_id, _profile in self.model_profiles)


def get_breakout_quality_continuous_ranker_comparison_settings(
) -> BreakoutQualityContinuousRankerComparisonSettings:
    model_profiles = tuple(
        (str(model_id).strip(), str(profile).strip())
        for model_id, profile in cfg.BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_PROFILES
    )
    if len(model_profiles) < 2:
        raise ValueError("continuous ranker comparison至少需要兩個model profile")
    model_ids = tuple(model_id for model_id, _profile in model_profiles)
    profiles = tuple(profile for _model_id, profile in model_profiles)
    if any(not model_id for model_id in model_ids) or any(not profile for profile in profiles):
        raise ValueError("continuous ranker comparison model id/profile不得為空")
    if len(set(model_ids)) != len(model_ids) or len(set(profiles)) != len(profiles):
        raise ValueError("continuous ranker comparison model id/profile不可重複")
    for profile in profiles:
        experiment = get_breakout_quality_experiment_profile(profile)
        if experiment.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            raise ValueError(
                "continuous ranker comparison只允許continuous ranker profile: "
                f"{profile}"
            )

    menu_label = str(cfg.BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL).strip()
    if not menu_label:
        raise ValueError("continuous ranker comparison menu label不得為空")
    reference_arm = str(
        cfg.BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_REFERENCE_ARM
    ).strip()
    if not reference_arm:
        raise ValueError("continuous ranker comparison reference arm不得為空")

    raw_summary_pair = tuple(
        str(value).strip()
        for value in cfg.BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_SUMMARY_PAIR
    )
    if len(raw_summary_pair) != 2 or raw_summary_pair[0] == raw_summary_pair[1]:
        raise ValueError("continuous ranker comparison summary pair必須是兩個不同model id")
    if any(model_id not in model_ids for model_id in raw_summary_pair):
        raise ValueError("continuous ranker comparison summary pair必須存在於model profiles")
    available_contrasts = tuple(zip(model_ids[1:], model_ids[:-1]))
    if raw_summary_pair not in available_contrasts:
        raise ValueError(
            "continuous ranker comparison summary pair必須符合model profiles的相鄰比較順序"
        )

    fixed_k_values = tuple(
        int(value)
        for value in cfg.BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_FIXED_K_VALUES
    )
    if not fixed_k_values or any(value < 1 for value in fixed_k_values):
        raise ValueError("continuous ranker comparison fixed K sweep只能包含正整數")
    if len(set(fixed_k_values)) != len(fixed_k_values):
        raise ValueError("continuous ranker comparison fixed K sweep不可重複")
    if tuple(sorted(fixed_k_values)) != fixed_k_values:
        raise ValueError("continuous ranker comparison fixed K sweep必須遞增排序")

    return BreakoutQualityContinuousRankerComparisonSettings(
        enabled=bool(cfg.BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_ENABLED),
        menu_label=menu_label,
        model_profiles=model_profiles,
        reference_arm=reference_arm,
        summary_pair=(raw_summary_pair[0], raw_summary_pair[1]),
        fixed_k_values=fixed_k_values,
    )


@dataclass(frozen=True)
class BreakoutQualityModelTestSettings:
    model_profiles: tuple[tuple[str, str], ...]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model_id for model_id, _profile in self.model_profiles)


def get_breakout_quality_model_test_settings() -> BreakoutQualityModelTestSettings:
    model_profiles = tuple(
        (str(model_id).strip(), str(profile).strip())
        for model_id, profile in cfg.BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    )
    required_training_model = tuple(
        str(value).strip() for value in cfg.BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
    )
    if len(required_training_model) != 2 or required_training_model not in model_profiles:
        raise ValueError(
            "模型比較／測試清單必須自動包含[1]/[2] canonical Training Model: "
            f"{required_training_model!r}"
        )
    if len(model_profiles) < 2:
        raise ValueError("模型比較／測試清單至少需要兩個model profile")
    model_ids = tuple(model_id for model_id, _profile in model_profiles)
    profiles = tuple(profile for _model_id, profile in model_profiles)
    if any(not value for value in (*model_ids, *profiles)):
        raise ValueError("模型比較／測試清單model id/profile不得為空")
    if len(set(model_ids)) != len(model_ids) or len(set(profiles)) != len(profiles):
        raise ValueError("模型比較／測試清單model id/profile不可重複")
    for model_id, profile_name in model_profiles:
        experiment = get_breakout_quality_experiment_profile(profile_name)
        if experiment.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            raise ValueError(
                "模型比較／測試清單只允許continuous ranker profile: "
                f"{profile_name}"
            )
        research = get_continuous_ranker_research_spec(profile_name)
        if str(research.model_research_id) != str(model_id):
            raise ValueError(
                "模型比較／測試清單model id/profile research identity不一致: "
                f"configured={model_id}, resolved={research.model_research_id}, profile={profile_name}"
            )
    return BreakoutQualityModelTestSettings(model_profiles=model_profiles)


@dataclass(frozen=True)
class BreakoutQualityStandardModelComparisonSettings:
    menu_label: str
    model_profiles: tuple[tuple[str, str], ...]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model_id for model_id, _profile in self.model_profiles)


def get_breakout_quality_standard_model_comparison_settings(
) -> BreakoutQualityStandardModelComparisonSettings:
    model_profiles = get_breakout_quality_model_test_settings().model_profiles
    menu_label = str(cfg.BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_MENU_LABEL).strip()
    if not menu_label:
        raise ValueError("Standard Model SOP比較menu label不得為空")
    return BreakoutQualityStandardModelComparisonSettings(
        menu_label=menu_label,
        model_profiles=model_profiles,
    )


@dataclass(frozen=True)
class BreakoutQualityRollingTestModeSettings:
    mode_id: str
    label: str
    score_start_date: str
    score_end_date: str | None
    fold_months: int
    fold_anchor_date: str | None
    single_score_block: bool
    point_in_time_dirname: str | None


def get_breakout_quality_rolling_test_modes() -> tuple[BreakoutQualityRollingTestModeSettings, ...]:
    rows: list[BreakoutQualityRollingTestModeSettings] = []
    seen: set[str] = set()
    for mode_id, raw in dict(cfg.BREAKOUT_QUALITY_ROLLING_TEST_MODES).items():
        normalized_id = str(mode_id).strip()
        if not normalized_id or normalized_id in seen:
            raise ValueError(f"Rolling Test mode id空白或重複: {mode_id!r}")
        label = str(dict(raw).get("label") or "").strip()
        score_start_date = str(dict(raw).get("score_start_date") or "").strip()
        score_end_raw = dict(raw).get("score_end_date")
        score_end_date = None if score_end_raw in (None, "") else str(score_end_raw).strip()
        fold_months = int(dict(raw).get("fold_months", 0) or 0)
        single_score_block = bool(dict(raw).get("single_score_block", False))
        anchor_raw = dict(raw).get("fold_anchor_date")
        fold_anchor_date = None if anchor_raw in (None, "") else str(anchor_raw).strip()
        if fold_anchor_date is not None:
            try:
                date.fromisoformat(fold_anchor_date)
            except ValueError as exc:
                raise ValueError(
                    f"Rolling Test fold_anchor_date必須是YYYY-MM-DD: {fold_anchor_date!r}"
                ) from exc
        dirname_raw = dict(raw).get("point_in_time_dirname")
        dirname = None if dirname_raw in (None, "") else str(dirname_raw).strip()
        if not label:
            raise ValueError(f"Rolling Test mode label不可空白: {normalized_id}")
        if not score_start_date:
            raise ValueError(f"Rolling Test score_start_date不可空白: {normalized_id}")
        try:
            date.fromisoformat(score_start_date)
        except ValueError as exc:
            raise ValueError(
                f"Rolling Test score_start_date必須是YYYY-MM-DD: {score_start_date!r}"
            ) from exc
        if score_end_date is not None and score_end_date.lower() != "auto":
            try:
                date.fromisoformat(score_end_date)
            except ValueError as exc:
                raise ValueError(
                    f"Rolling Test score_end_date必須是YYYY-MM-DD或auto: {score_end_date!r}"
                ) from exc
        if fold_months < 1:
            raise ValueError(f"Rolling Test fold_months必須>=1: {normalized_id}")
        if dirname is not None and (Path(dirname).name != dirname or dirname in {".", ".."}):
            raise ValueError(f"Rolling Test point_in_time_dirname必須是安全單一資料夾名稱: {dirname!r}")
        rows.append(
            BreakoutQualityRollingTestModeSettings(
                mode_id=normalized_id,
                label=label,
                score_start_date=score_start_date,
                score_end_date=score_end_date,
                fold_months=fold_months,
                fold_anchor_date=fold_anchor_date,
                single_score_block=single_score_block,
                point_in_time_dirname=dirname,
            )
        )
        seen.add(normalized_id)
    if len(rows) < 2:
        raise ValueError("Rolling Test至少需要兩種執行深度")
    default_mode = str(cfg.BREAKOUT_QUALITY_DEFAULT_ROLLING_TEST_MODE).strip()
    if default_mode not in seen:
        raise ValueError(f"Rolling Test default mode不存在: {default_mode!r}")
    return tuple(rows)


def get_breakout_quality_rolling_test_mode(
    mode_id: str | None = None,
) -> BreakoutQualityRollingTestModeSettings:
    selected = str(mode_id or cfg.BREAKOUT_QUALITY_DEFAULT_ROLLING_TEST_MODE).strip()
    for item in get_breakout_quality_rolling_test_modes():
        if item.mode_id == selected:
            return item
    raise ValueError(f"未知Rolling Test mode: {selected!r}")


@dataclass(frozen=True)
class BreakoutQualityRollingTimingSettings:
    experiment_profile: str
    seed: int
    score_years: tuple[int, ...]


def get_breakout_quality_rolling_timing_settings() -> BreakoutQualityRollingTimingSettings:
    profile_name = str(
        cfg.BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE
        if cfg.BREAKOUT_QUALITY_ROLLING_TIMING_EXPERIMENT_PROFILE is None
        else cfg.BREAKOUT_QUALITY_ROLLING_TIMING_EXPERIMENT_PROFILE
    ).strip()
    if not profile_name:
        raise ValueError("Rolling Timing experiment profile不可空白")
    profile = get_breakout_quality_experiment_profile(profile_name)
    if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        raise ValueError("Rolling Timing只支援continuous ranker profile")

    workflow = get_breakout_quality_workflow_settings(experiment_profile=profile_name)
    if not workflow.rolling_authorized:
        raise ValueError(
            f"Rolling Timing profile尚未授權current Rolling: {profile_name}"
        )

    seed = (
        resolve_breakout_quality_random_seed()
        if cfg.BREAKOUT_QUALITY_ROLLING_TIMING_SEED is None
        else int(cfg.BREAKOUT_QUALITY_ROLLING_TIMING_SEED)
    )
    if seed < 0:
        raise ValueError("Rolling Timing seed必須 >= 0")

    years = tuple(int(value) for value in cfg.BREAKOUT_QUALITY_ROLLING_TIMING_SCORE_YEARS)
    if not years:
        raise ValueError("Rolling Timing至少需要一個score year")
    if len(set(years)) != len(years):
        raise ValueError("Rolling Timing score years不可重複")
    if tuple(sorted(years)) != years:
        raise ValueError("Rolling Timing score years必須遞增排序")
    formal_start_raw = str(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE).strip()
    formal_start = (
        None
        if formal_start_raw.lower() == "auto"
        else date.fromisoformat(formal_start_raw)
    )
    formal_end_raw = (
        ""
        if cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE is None
        else str(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE).strip()
    )
    formal_end = (
        None
        if formal_end_raw.lower() in {"", "auto"}
        else date.fromisoformat(formal_end_raw)
    )
    for year in years:
        if year < 1900 or year > 2200:
            raise ValueError(f"Rolling Timing score year不合法: {year}")
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        if formal_start is not None and start < formal_start:
            raise ValueError(
                f"Rolling Timing year早於current Rolling起點: {year} < {formal_start.year}"
            )
        if formal_end is not None and end > formal_end:
            raise ValueError(
                f"Rolling Timing year晚於current完整Rolling終點: {year} > {formal_end.year}"
            )

    return BreakoutQualityRollingTimingSettings(
        experiment_profile=profile_name,
        seed=int(seed),
        score_years=years,
    )


# =============================================================================
# WORKFLOW RESOLUTION AND VALIDATION — do not edit unless changing implementation
# =============================================================================

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

@dataclass(frozen=True)
class BreakoutQualityWorkflowSettings:
    filter_id: str
    model_architecture: str
    experiment_profile: str
    training_objective: str
    continuous_target_id: str | None
    training_label_scope: str
    training_sample_scope: str
    seed: int
    point_in_time_score_start_date: str
    point_in_time_coverage_reference_start_date: str
    point_in_time_score_end_date: str | None
    point_in_time_fold_months: int
    point_in_time_inner_validation_months: int
    point_in_time_train_window_months: int | None
    point_in_time_min_train_groups: int
    point_in_time_min_validation_groups: int
    point_in_time_min_score_groups: int
    point_in_time_resume: bool
    strategy_dataset: str
    strategy_param_policy: str
    strategy_max_positions: int
    strategy_rotation: str
    strategy_trials_per_fold: int
    strategy_fixed_risk: float
    strategy_max_position_cap_pct: float
    strategy_comparison_mode: str
    strategy_score_source: str
    strategy_buy_sort: str
    runtime_strategy_enabled: bool
    runtime_ranking_policy: str
    runtime_ranking_options: dict[str, Any]

    @property
    def is_binary_classification(self) -> bool:
        return self.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION

    @property
    def is_continuous_ranker(self) -> bool:
        return self.training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES

    @property
    def supports_point_in_time_scores(self) -> bool:
        return bool(
            self.is_continuous_ranker
            and self.training_sample_scope
            in SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLE_SCOPES
        )

    @property
    def rolling_authorized(self) -> bool:
        """Whether current Rolling model evidence is authorized for this profile.

        Current model-workflow membership is the work-item SSOT: the active Training
        Profile drives [1]/[2], and the shared Model Compare/Test List drives [3]～[6].
        Historical recipe authorization remains accepted for compatibility, but it is
        no longer a second selector that can block a currently configured model target.
        """

        if not self.supports_point_in_time_scores:
            return False
        recipe = get_continuous_ranker_execution_recipe(self.experiment_profile)
        # B313/B325: current workflow membership is the sole current-mode selector.
        # Historical recipe authorization remains readable for profiles outside the
        # current workflow, but it cannot veto a configured Training/Compare member.
        return bool(
            recipe.current_time_validation_authorized
            or is_breakout_quality_model_workflow_profile(self.experiment_profile)
        )

    @property
    def robustness_authorized(self) -> bool:
        """Whether current multi-seed model evidence is authorized for this profile.

        Robustness is a property of shared Model Compare/Test membership.  The current
        Training Model is structurally injected into that same list, so no per-profile
        authorization flag may create a second selector.
        """

        return bool(
            self.is_continuous_ranker
            and is_breakout_quality_model_test_profile(self.experiment_profile)
        )

    def as_manifest_payload(self) -> dict[str, Any]:
        payload = {
            "filter_id": self.filter_id,
            "model_architecture": self.model_architecture,
            "experiment_profile": self.experiment_profile,
            "training_objective": self.training_objective,
            "continuous_target_id": self.continuous_target_id,
            "training_label_scope": self.training_label_scope,
            "seed": int(self.seed),
            "point_in_time": {
                "enabled": bool(self.rolling_authorized),
                "structurally_supported": bool(self.supports_point_in_time_scores),
                "research_authorized": bool(self.rolling_authorized),
                "score_start_date": self.point_in_time_score_start_date,
                "coverage_reference_start_date": (
                    self.point_in_time_coverage_reference_start_date
                ),
                "score_end_date": self.point_in_time_score_end_date,
                "fold_months": int(self.point_in_time_fold_months),
                "inner_validation_months": int(
                    self.point_in_time_inner_validation_months
                ),
                "train_window_months": (
                    None
                    if self.point_in_time_train_window_months is None
                    else int(self.point_in_time_train_window_months)
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
                "adaptation": {
                    "trials_per_fold": int(self.strategy_trials_per_fold),
                    "trial_source": "OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT",
                    "fixed_risk": float(self.strategy_fixed_risk),
                    "max_position_cap_pct": float(
                        self.strategy_max_position_cap_pct
                    ),
                },
                "comparison_mode": self.strategy_comparison_mode,
                "score_source": self.strategy_score_source,
                "buy_sort": self.strategy_buy_sort,
            },
            "runtime": {
                "enabled": bool(self.runtime_strategy_enabled),
                "score_source": WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
                "ranking_policy": self.runtime_ranking_policy,
                "ranking_options": dict(self.runtime_ranking_options),
            },
        }
        if self.training_sample_scope != TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
            payload["training_sample_scope"] = self.training_sample_scope
        return payload



def _resolve_strategy_defaults(training_objective: str) -> tuple[str, str, str]:
    if training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
        return (
            WORKFLOW_STRATEGY_MODE_HARD_FILTER,
            WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
            WORKFLOW_BUY_SORT_ORIGINAL,
        )
    if training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        return (
            WORKFLOW_STRATEGY_MODE_SCORE_RANKING,
            WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            WORKFLOW_BUY_SORT_SCORE_DESC,
        )
    raise ValueError(f"不支援的 workflow training objective: {training_objective!r}")


def _resolve_auto(value: str, *, auto_value: str, resolved_default: str) -> str:
    normalized = str(value).strip()
    return resolved_default if normalized == auto_value else normalized


def get_breakout_quality_workflow_settings(
    *, experiment_profile: str | None = None
) -> BreakoutQualityWorkflowSettings:
    random_seed = resolve_breakout_quality_random_seed()
    resolved_experiment_profile = str(
        cfg.BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE
        if experiment_profile is None
        else experiment_profile
    ).strip()
    profile = get_breakout_quality_experiment_profile(resolved_experiment_profile)
    if profile.training_objective not in {
        TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
        *CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
    }:
        raise ValueError(
            "workflow experiment profile必須是binary classification或continuous ranker"
        )
    try:
        coverage_reference_start = date.fromisoformat(
            str(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_COVERAGE_REFERENCE_START_DATE).strip()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "point-in-time coverage reference start date必須是YYYY-MM-DD合法日期"
        ) from exc
    if int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS) < 1:
        raise ValueError("point-in-time fold months 必須 >= 1")
    if int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS) < 1:
        raise ValueError("point-in-time inner validation months 必須 >= 1")
    if cfg.BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS is not None:
        if int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS) <= int(
            cfg.BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ):
            raise ValueError(
                "point-in-time fixed train window必須大於inner validation months"
            )
    if min(
        int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS),
        int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS),
        int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS),
    ) < 1:
        raise ValueError("point-in-time minimum group counts 必須 >= 1")
    if cfg.BREAKOUT_QUALITY_STRATEGY_DATASET not in {"reduced", "full"}:
        raise ValueError("strategy dataset 必須是 reduced 或 full")
    if cfg.BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY not in {
        "auto",
        "base-finalist-best",
        "base-finalists-agree",
    }:
        raise ValueError(
            "strategy param policy 必須是auto、base-finalist-best或base-finalists-agree"
        )
    if int(cfg.BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS) < 1:
        raise ValueError("strategy max positions 必須 >= 1")
    if cfg.BREAKOUT_QUALITY_STRATEGY_ROTATION not in {"off", "on"}:
        raise ValueError("strategy rotation 必須是 off 或 on")
    if int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT) < 1:
        raise ValueError("outer rolling optimizer trials per fold 必須 >= 1")
    if not 0.0 < float(DEFAULT_FIXED_RISK) <= 1.0:
        raise ValueError("strategy fixed risk 必須介於0與1")
    if not 0.0 < float(DEFAULT_MAX_POSITION_CAP_PCT) <= 1.0:
        raise ValueError("strategy max position cap pct 必須介於0與1")

    raw_comparison_mode = str(cfg.BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE).strip()
    if raw_comparison_mode not in SUPPORTED_WORKFLOW_STRATEGY_MODES:
        raise ValueError(
            "strategy comparison mode 必須是auto、hard-filter或score-ranking"
        )
    raw_score_source = str(cfg.BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE).strip()
    if raw_score_source not in SUPPORTED_WORKFLOW_SCORE_SOURCES:
        raise ValueError(
            "strategy score source 必須是auto、selection_point_in_time、"
            "final_selection_model_oos或canonical_runtime"
        )
    raw_buy_sort = str(cfg.BREAKOUT_QUALITY_STRATEGY_BUY_SORT).strip()
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

    runtime_ranking_policy = str(cfg.BREAKOUT_QUALITY_RUNTIME_RANKING_POLICY).strip()
    runtime_ranking_options = dict(cfg.BREAKOUT_QUALITY_RUNTIME_RANKING_OPTIONS or {})
    runtime_strategy_enabled = bool(
        cfg.BREAKOUT_QUALITY_RUNTIME_STRATEGY_ENABLED
        and resolved_experiment_profile == str(cfg.BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE).strip()
        and profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )
    if runtime_strategy_enabled:
        if not runtime_ranking_policy:
            raise ValueError("正式runtime ranking policy不可為空白")
        expected_runtime_options = {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": False,
        }
        if runtime_ranking_options != expected_runtime_options:
            raise ValueError(
                "正式runtime必須維持Gate已驗證的exact K/R0 contract: "
                f"expected={expected_runtime_options}, actual={runtime_ranking_options}"
            )

    return BreakoutQualityWorkflowSettings(
        filter_id=str(BREAKOUT_QUALITY_WORKFLOW_FILTER_ID),
        model_architecture=str(profile.model_architecture or BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE),
        experiment_profile=resolved_experiment_profile,
        training_objective=str(profile.training_objective),
        continuous_target_id=(
            None
            if profile.continuous_target_id is None
            else str(profile.continuous_target_id)
        ),
        training_label_scope=str(profile.training_label_scope),
        training_sample_scope=str(profile.training_sample_scope),
        seed=random_seed,
        point_in_time_score_start_date=str(
            cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE
        ),
        point_in_time_coverage_reference_start_date=(
            coverage_reference_start.isoformat()
        ),
        point_in_time_score_end_date=(
            None
            if cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE is None
            else str(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE)
        ),
        point_in_time_fold_months=int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS),
        point_in_time_inner_validation_months=int(
            cfg.BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ),
        point_in_time_train_window_months=(
            None
            if cfg.BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS is None
            else int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS)
        ),
        point_in_time_min_train_groups=int(
            cfg.BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS
        ),
        point_in_time_min_validation_groups=int(
            cfg.BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS
        ),
        point_in_time_min_score_groups=int(
            cfg.BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS
        ),
        point_in_time_resume=bool(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_RESUME),
        strategy_dataset=str(cfg.BREAKOUT_QUALITY_STRATEGY_DATASET),
        strategy_param_policy=str(cfg.BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY),
        strategy_max_positions=int(cfg.BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS),
        strategy_rotation=str(cfg.BREAKOUT_QUALITY_STRATEGY_ROTATION),
        strategy_trials_per_fold=int(
            OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
        ),
        strategy_fixed_risk=float(DEFAULT_FIXED_RISK),
        strategy_max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        strategy_comparison_mode=strategy_comparison_mode,
        strategy_score_source=strategy_score_source,
        strategy_buy_sort=strategy_buy_sort,
        runtime_strategy_enabled=runtime_strategy_enabled,
        runtime_ranking_policy=runtime_ranking_policy,
        runtime_ranking_options=runtime_ranking_options,
    )


@dataclass(frozen=True)
class BreakoutQualityContinuousRankerPITGateSettings:
    model_profiles: tuple[tuple[str, str], ...]
    seed: int
    point_in_time_score_start_date: str
    point_in_time_score_end_date: str | None
    point_in_time_fold_months: int
    point_in_time_inner_validation_months: int

    def __post_init__(self) -> None:
        if len(self.model_profiles) < 2:
            raise ValueError("PIT Gate batch至少需要兩個model profile")
        seen_ids: set[str] = set()
        seen_profiles: set[str] = set()
        reference = None
        for model_id, profile_name in self.model_profiles:
            model_id = str(model_id).strip()
            profile_name = normalize_breakout_quality_experiment_profile(profile_name)
            if not model_id or model_id in seen_ids:
                raise ValueError(f"PIT Gate model ID空白或重複: {model_id!r}")
            if profile_name in seen_profiles:
                raise ValueError(f"PIT Gate experiment profile重複: {profile_name}")
            seen_ids.add(model_id)
            seen_profiles.add(profile_name)
            spec = get_continuous_ranker_research_spec(profile_name)
            if spec.model_research_id != model_id:
                raise ValueError(
                    "PIT Gate model ID/profile research identity不一致: "
                    f"{model_id} != {spec.model_research_id}"
                )
            workflow = get_breakout_quality_workflow_settings(experiment_profile=profile_name)
            if not workflow.rolling_authorized:
                raise ValueError(f"PIT Gate profile尚未授權Rolling PIT: {profile_name}")
            contract = (
                workflow.filter_id,
                workflow.model_architecture,
                workflow.continuous_target_id,
                workflow.training_sample_scope,
                workflow.seed,
                workflow.point_in_time_score_start_date,
                workflow.point_in_time_score_end_date,
                workflow.point_in_time_fold_months,
                workflow.point_in_time_inner_validation_months,
            )
            if reference is None:
                reference = contract
            elif contract != reference:
                raise ValueError(
                    "PIT Gate profiles的Dataset/Target/Seed/period/fold contract不一致"
                )


def get_breakout_quality_continuous_ranker_pit_gate_settings(
) -> BreakoutQualityContinuousRankerPITGateSettings:
    return BreakoutQualityContinuousRankerPITGateSettings(
        model_profiles=tuple(
            (str(model_id), str(profile_name))
            for model_id, profile_name in cfg.BREAKOUT_QUALITY_CONTINUOUS_RANKER_PIT_GATE_PROFILES
        ),
        seed=resolve_breakout_quality_random_seed(),
        point_in_time_score_start_date=str(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE),
        point_in_time_score_end_date=(
            None
            if cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE is None
            else str(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE)
        ),
        point_in_time_fold_months=int(cfg.BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS),
        point_in_time_inner_validation_months=int(
            cfg.BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ),
    )


def get_breakout_quality_model_research_settings() -> BreakoutQualityWorkflowSettings:
    """Return the canonical [1]/[2] training identity without changing strategy defaults."""

    model_id, profile_name = (
        str(value).strip() for value in cfg.BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
    )
    if not model_id or not profile_name:
        raise ValueError("canonical Training Model model id/profile不得為空")
    research = get_continuous_ranker_research_spec(profile_name)
    if str(research.model_research_id).strip() != model_id:
        raise ValueError(
            "canonical Training Model model id/profile research identity不一致: "
            f"configured={model_id}, resolved={research.model_research_id}, "
            f"profile={profile_name}"
        )
    return get_breakout_quality_workflow_settings(experiment_profile=profile_name)
