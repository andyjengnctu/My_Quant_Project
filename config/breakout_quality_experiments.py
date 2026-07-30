"""Named breakout-quality training experiment profiles.

Model architecture versions describe network/input changes. Training experiments such as
optimizer, learning-rate schedule, augmentation, and training sampling are selected
independently here so experiments do not create fake model versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BASELINE_EXPERIMENT_PROFILE = "baseline"
ADAMW_ONLY_EXPERIMENT_PROFILE = "adamw_only"
ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE = "adam_warmup_cosine"
HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE = "history_masking_only"
UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE = "unique_group_sampling"
UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE = "unique_group_date_balanced"
STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE = "strategy_aligned_daily_percentile_mse"
STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE = "strategy_aligned_no_time_pass_magnitude_mse"
TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE = "ts2vec_selection_only"

TRAINING_SAMPLING_ALL_EVENT_ROWS = "all_event_rows_group_weighted"
TRAINING_SAMPLING_UNIQUE_TICKER_DATE = "unique_ticker_date"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES = (
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
)

TIME_WEIGHT_MODE_NONE = "none"
TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT = "year_balanced_sqrt"
TIME_WEIGHT_MODE_DATE_BALANCED = "date_balanced"
SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES = (
    TIME_WEIGHT_MODE_NONE,
    TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT,
    TIME_WEIGHT_MODE_DATE_BALANCED,
)

TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM = "batch_weight_sum"
TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE = "fixed_batch_size"

TRAINING_LABEL_SCOPE_ALL = "all_labels"
TRAINING_LABEL_SCOPE_PASS_ONLY = "pass_only"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES = (
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
)

TRAINING_OBJECTIVE_BINARY_CLASSIFICATION = "binary_classification"
TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION = "daily_percentile_regression"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES = (
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
)
SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS = (
    TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM,
    TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
)

LR_SCHEDULE_NONE = "none"
LR_SCHEDULE_LINEAR_WARMUP_COSINE = "linear_warmup_cosine"
AUGMENTATION_NONE = "none"
AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK = "old_history_contiguous_mask"

SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS = ("adam", "adamw")
SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES = (
    LR_SCHEDULE_NONE,
    LR_SCHEDULE_LINEAR_WARMUP_COSINE,
)
SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS = (
    AUGMENTATION_NONE,
    AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
)


@dataclass(frozen=True)
class BreakoutQualityExperimentProfile:
    name: str
    optimizer_name: str
    lr_schedule_name: str = LR_SCHEDULE_NONE
    augmentation_name: str = "none"
    augmentation_probability: float = 0.0
    augmentation_protected_recent_bars: int = 0
    augmentation_min_mask_bars: int = 0
    augmentation_max_mask_bars: int = 0
    lr_warmup_fraction: float = 0.0
    lr_minimum_ratio: float = 1.0
    training_sampling_mode: str = TRAINING_SAMPLING_ALL_EVENT_ROWS
    time_weight_mode: str | None = None
    training_weight_reduction: str = TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM
    training_objective: str = TRAINING_OBJECTIVE_BINARY_CLASSIFICATION
    continuous_target_id: str | None = None
    loss_name: str = "cross_entropy"
    epoch_selection_metric: str = "validation_loss"
    training_label_scope: str = TRAINING_LABEL_SCOPE_ALL

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("experiment profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("experiment profile name 必須是安全的單一資料夾名稱")
        if self.optimizer_name not in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS:
            raise ValueError(f"不支援的 optimizer: {self.optimizer_name!r}")
        if self.lr_schedule_name not in SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES:
            raise ValueError(f"不支援的 LR schedule: {self.lr_schedule_name!r}")
        if self.augmentation_name not in SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS:
            raise ValueError(f"不支援的 augmentation: {self.augmentation_name!r}")
        if self.training_sampling_mode not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES:
            raise ValueError(
                f"不支援的 training sampling mode: {self.training_sampling_mode!r}"
            )
        if (
            self.time_weight_mode is not None
            and self.time_weight_mode not in SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES
        ):
            raise ValueError(f"不支援的 time weight mode: {self.time_weight_mode!r}")
        if self.training_weight_reduction not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS:
            raise ValueError(
                f"不支援的 training weight reduction: {self.training_weight_reduction!r}"
            )
        if self.training_objective not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES:
            raise ValueError(f"不支援的 training objective: {self.training_objective!r}")
        if self.training_label_scope not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES:
            raise ValueError(f"不支援的 training label scope: {self.training_label_scope!r}")
        if self.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
            if self.continuous_target_id is not None:
                raise ValueError("binary classification profile 不得指定 continuous_target_id")
            if self.loss_name != "cross_entropy" or self.epoch_selection_metric != "validation_loss":
                raise ValueError("binary classification profile 必須使用 cross_entropy / validation_loss")
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                raise ValueError("binary classification profile 必須使用all_labels scope")
        elif self.training_objective == TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION:
            if not str(self.continuous_target_id or "").strip():
                raise ValueError("daily percentile regression profile 必須指定 continuous_target_id")
            if self.loss_name != "mse":
                raise ValueError("daily percentile regression profile 必須使用 mse")
            if self.epoch_selection_metric != "mean_daily_spearman":
                raise ValueError("daily percentile regression profile 必須以 mean_daily_spearman 選 epoch")
            if self.training_sampling_mode != TRAINING_SAMPLING_UNIQUE_TICKER_DATE:
                raise ValueError("daily percentile regression 只允許 unique ticker/date sampling")
            if self.time_weight_mode not in {None, TIME_WEIGHT_MODE_NONE}:
                raise ValueError("daily percentile regression 第一版不允許 time weighting")
            if self.training_weight_reduction != TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
                raise ValueError("daily percentile regression 第一版只允許 batch_weight_sum")
        if (
            self.training_weight_reduction == TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE
            and self.time_weight_mode != TIME_WEIGHT_MODE_DATE_BALANCED
        ):
            raise ValueError(
                "fixed_batch_size training weight reduction 目前只允許 date_balanced profile"
            )
        warmup_fraction = float(self.lr_warmup_fraction)
        minimum_ratio = float(self.lr_minimum_ratio)
        if self.lr_schedule_name == LR_SCHEDULE_NONE:
            if warmup_fraction != 0.0 or minimum_ratio != 1.0:
                raise ValueError("無 LR schedule 時 warmup 必須為 0、minimum ratio 必須為 1")
        elif self.lr_schedule_name == LR_SCHEDULE_LINEAR_WARMUP_COSINE:
            if not 0.0 < warmup_fraction < 1.0:
                raise ValueError("linear warmup fraction 必須介於 0 與 1 之間")
            if not 0.0 < minimum_ratio <= 1.0:
                raise ValueError("minimum LR ratio 必須介於 0 與 1 之間")
        augmentation_parameters = self.augmentation_parameters()
        if self.augmentation_name == AUGMENTATION_NONE:
            if (
                float(self.augmentation_probability) != 0.0
                or int(self.augmentation_protected_recent_bars) != 0
                or int(self.augmentation_min_mask_bars) != 0
                or int(self.augmentation_max_mask_bars) != 0
                or augmentation_parameters
            ):
                raise ValueError("augmentation=none 時不可帶 augmentation 參數")
        elif self.augmentation_name == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
            probability = float(self.augmentation_probability)
            protected_recent_bars = int(self.augmentation_protected_recent_bars)
            min_mask_bars = int(self.augmentation_min_mask_bars)
            max_mask_bars = int(self.augmentation_max_mask_bars)
            if not 0.0 < probability <= 1.0:
                raise ValueError("masking augmentation probability 必須介於 0 與 1 之間")
            if protected_recent_bars < 1:
                raise ValueError("masking protected_recent_bars 必須 >=1")
            if min_mask_bars < 1 or max_mask_bars < min_mask_bars:
                raise ValueError("masking bars 必須滿足 1 <= min <= max")

    def lr_schedule_parameters(self) -> dict[str, float]:
        if self.lr_schedule_name == LR_SCHEDULE_NONE:
            return {}
        return {
            "warmup_fraction": float(self.lr_warmup_fraction),
            "minimum_lr_ratio": float(self.lr_minimum_ratio),
        }

    def augmentation_parameters(self) -> dict[str, int | float]:
        if self.augmentation_name == AUGMENTATION_NONE:
            return {}
        if self.augmentation_name == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
            return {
                "probability": float(self.augmentation_probability),
                "protected_recent_bars": int(self.augmentation_protected_recent_bars),
                "min_mask_bars": int(self.augmentation_min_mask_bars),
                "max_mask_bars": int(self.augmentation_max_mask_bars),
            }
        raise ValueError(f"不支援的 augmentation: {self.augmentation_name!r}")

    def as_manifest_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "optimizer_name": self.optimizer_name,
            "lr_schedule_name": self.lr_schedule_name,
            "augmentation_name": self.augmentation_name,
        }
        schedule_parameters = self.lr_schedule_parameters()
        if schedule_parameters:
            payload["lr_schedule_parameters"] = schedule_parameters
        augmentation_parameters = self.augmentation_parameters()
        if augmentation_parameters:
            payload["augmentation_parameters"] = augmentation_parameters
        if self.training_sampling_mode != TRAINING_SAMPLING_ALL_EVENT_ROWS:
            payload["training_sampling_mode"] = self.training_sampling_mode
        if self.time_weight_mode is not None:
            payload["time_weight_mode"] = self.time_weight_mode
        if self.training_weight_reduction != TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
            payload["training_weight_reduction"] = self.training_weight_reduction
        if self.training_objective != TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
            payload.update({
                "training_objective": self.training_objective,
                "continuous_target_id": self.continuous_target_id,
                "loss_name": self.loss_name,
                "epoch_selection_metric": self.epoch_selection_metric,
            })
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                payload["training_label_scope"] = self.training_label_scope
        return payload



@dataclass(frozen=True)
class BreakoutQualityPretrainingProfile:
    name: str
    family: str
    optimizer_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    min_crop_bars: int
    mask_probability: float
    contrastive_alpha: float
    temporal_unit: int

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("pretraining profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("pretraining profile name 必須是安全的單一資料夾名稱")
        if str(self.family).strip().lower() != self.family or not self.family:
            raise ValueError("pretraining family 必須是非空白小寫名稱")
        if any(token in self.family for token in ("/", "\\", "\x00")):
            raise ValueError("pretraining family 必須是安全的單一資料夾名稱")
        if self.optimizer_name not in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS:
            raise ValueError(f"不支援的 pretraining optimizer: {self.optimizer_name!r}")
        if int(self.epochs) < 1 or int(self.batch_size) < 2:
            raise ValueError("pretraining epochs 必須 >=1 且 batch_size 必須 >=2")
        if (
            float(self.learning_rate) <= 0.0
            or float(self.weight_decay) < 0.0
            or float(self.gradient_clip_norm) < 0.0
        ):
            raise ValueError(
                "pretraining learning_rate 必須 >0，weight_decay與gradient_clip_norm必須 >=0"
            )
        if int(self.min_crop_bars) < 2 or int(self.temporal_unit) < 0:
            raise ValueError("pretraining min_crop_bars 必須 >=2 且 temporal_unit 必須 >=0")
        if not 0.0 <= float(self.mask_probability) < 1.0:
            raise ValueError("pretraining mask_probability 必須介於0（含）與1（不含）")
        if not 0.0 <= float(self.contrastive_alpha) <= 1.0:
            raise ValueError("pretraining contrastive_alpha 必須介於0與1")

    def as_manifest_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "optimizer_name": self.optimizer_name,
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "min_crop_bars": int(self.min_crop_bars),
            "mask_probability": float(self.mask_probability),
            "contrastive_alpha": float(self.contrastive_alpha),
            "temporal_unit": int(self.temporal_unit),
        }


_PRETRAINING_PROFILES = {
    TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE: BreakoutQualityPretrainingProfile(
        name=TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE,
        family="ts2vec_v1",
        optimizer_name="adamw",
        epochs=10,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        min_crop_bars=60,
        mask_probability=0.5,
        contrastive_alpha=0.5,
        temporal_unit=0,
    ),
}
SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES = tuple(_PRETRAINING_PROFILES)


def normalize_breakout_quality_pretraining_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _PRETRAINING_PROFILES:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES)
        raise ValueError(
            f"不支援的 breakout quality pretraining profile: {value!r}；可用值: {allowed}"
        )
    return normalized


def get_breakout_quality_pretraining_profile(
    value: str,
) -> BreakoutQualityPretrainingProfile:
    return _PRETRAINING_PROFILES[normalize_breakout_quality_pretraining_profile(value)]


def build_breakout_quality_pretraining_profile_payload(
    value: str,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    gradient_clip_norm: float | None = None,
    min_crop_bars: int | None = None,
    mask_probability: float | None = None,
    contrastive_alpha: float | None = None,
    temporal_unit: int | None = None,
) -> dict[str, Any]:
    """Return the named profile payload with explicit CLI overrides applied.

    Formal workflow runs use the profile defaults. The override path remains available for
    isolated development experiments, while downstream canonical training can reject an
    encoder whose stored payload differs from the active named profile.
    """

    profile = get_breakout_quality_pretraining_profile(value)
    resolved = BreakoutQualityPretrainingProfile(
        name=profile.name,
        family=profile.family,
        optimizer_name=profile.optimizer_name,
        epochs=profile.epochs if epochs is None else int(epochs),
        batch_size=profile.batch_size if batch_size is None else int(batch_size),
        learning_rate=(
            profile.learning_rate if learning_rate is None else float(learning_rate)
        ),
        weight_decay=profile.weight_decay if weight_decay is None else float(weight_decay),
        gradient_clip_norm=(
            profile.gradient_clip_norm
            if gradient_clip_norm is None
            else float(gradient_clip_norm)
        ),
        min_crop_bars=(
            profile.min_crop_bars if min_crop_bars is None else int(min_crop_bars)
        ),
        mask_probability=(
            profile.mask_probability
            if mask_probability is None
            else float(mask_probability)
        ),
        contrastive_alpha=(
            profile.contrastive_alpha
            if contrastive_alpha is None
            else float(contrastive_alpha)
        ),
        temporal_unit=profile.temporal_unit if temporal_unit is None else int(temporal_unit),
    )
    return resolved.as_manifest_payload()

_EXPERIMENT_PROFILES = {
    BASELINE_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=BASELINE_EXPERIMENT_PROFILE,
        optimizer_name="adam",
    ),
    ADAMW_ONLY_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=ADAMW_ONLY_EXPERIMENT_PROFILE,
        optimizer_name="adamw",
    ),
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        lr_schedule_name=LR_SCHEDULE_LINEAR_WARMUP_COSINE,
        lr_warmup_fraction=0.05,
        lr_minimum_ratio=0.10,
    ),
    HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        augmentation_name=AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
        augmentation_probability=0.50,
        augmentation_protected_recent_bars=60,
        augmentation_min_mask_bars=10,
        augmentation_max_mask_bars=30,
    ),
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    ),
    UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        time_weight_mode=TIME_WEIGHT_MODE_DATE_BALANCED,
        training_weight_reduction=TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
    ),
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
    ),
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
    ),
}

SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES = tuple(_EXPERIMENT_PROFILES)
SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES = tuple(
    name
    for name, profile in _EXPERIMENT_PROFILES.items()
    if profile.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION
)


def normalize_breakout_quality_experiment_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _EXPERIMENT_PROFILES:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES)
        raise ValueError(
            f"不支援的 breakout quality experiment profile: {value!r}；可用值: {allowed}"
        )
    return normalized


def get_breakout_quality_experiment_profile(
    value: str,
) -> BreakoutQualityExperimentProfile:
    return _EXPERIMENT_PROFILES[
        normalize_breakout_quality_experiment_profile(value)
    ]


__all__ = [
    "ADAMW_ONLY_EXPERIMENT_PROFILE",
    "ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE",
    "AUGMENTATION_NONE",
    "AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK",
    "BASELINE_EXPERIMENT_PROFILE",
    "HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE",
    "UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE",
    "UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE",
    "STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE",
    "STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE",
    "TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE",
    "BreakoutQualityExperimentProfile",
    "BreakoutQualityPretrainingProfile",
    "LR_SCHEDULE_LINEAR_WARMUP_COSINE",
    "LR_SCHEDULE_NONE",
    "SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS",
    "SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES",
    "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES",
    "SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES",
    "SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS",
    "SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES",
    "SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES",
    "SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES",
    "SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES",
    "TRAINING_LABEL_SCOPE_ALL",
    "TRAINING_LABEL_SCOPE_PASS_ONLY",
    "TRAINING_OBJECTIVE_BINARY_CLASSIFICATION",
    "TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION",
    "TRAINING_SAMPLING_ALL_EVENT_ROWS",
    "TRAINING_SAMPLING_UNIQUE_TICKER_DATE",
    "TIME_WEIGHT_MODE_DATE_BALANCED",
    "TIME_WEIGHT_MODE_NONE",
    "TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT",
    "SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES",
    "TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM",
    "TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE",
    "SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS",
    "get_breakout_quality_experiment_profile",
    "get_breakout_quality_pretraining_profile",
    "build_breakout_quality_pretraining_profile_payload",
    "normalize_breakout_quality_experiment_profile",
    "normalize_breakout_quality_pretraining_profile",
]
