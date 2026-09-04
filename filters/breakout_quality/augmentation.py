"""Deterministic training-only augmentation for breakout-quality sequence inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.breakout_quality_registry import (
    AUGMENTATION_NONE,
    AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
    SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS,
)


@dataclass(frozen=True)
class TrainingAugmentationPlan:
    name: str
    probability: float = 0.0
    protected_recent_bars: int = 0
    min_mask_bars: int = 0
    max_mask_bars: int = 0

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS:
            raise ValueError(f"不支援的 training augmentation: {self.name!r}")
        probability = float(self.probability)
        protected_recent_bars = int(self.protected_recent_bars)
        min_mask_bars = int(self.min_mask_bars)
        max_mask_bars = int(self.max_mask_bars)
        if self.name == AUGMENTATION_NONE:
            if (
                probability != 0.0
                or protected_recent_bars != 0
                or min_mask_bars != 0
                or max_mask_bars != 0
            ):
                raise ValueError("augmentation=none 時參數必須全部為 0")
            return
        if not 0.0 < probability <= 1.0:
            raise ValueError("augmentation probability 必須介於 0 與 1 之間")
        if protected_recent_bars < 1:
            raise ValueError("protected_recent_bars 必須 >=1")
        if min_mask_bars < 1 or max_mask_bars < min_mask_bars:
            raise ValueError("mask bars 必須滿足 1 <= min <= max")

    def as_parameters(self) -> dict[str, int | float]:
        if self.name == AUGMENTATION_NONE:
            return {}
        return {
            "probability": float(self.probability),
            "protected_recent_bars": int(self.protected_recent_bars),
            "min_mask_bars": int(self.min_mask_bars),
            "max_mask_bars": int(self.max_mask_bars),
        }


def build_training_augmentation_plan(
    *,
    name: str,
    parameters: dict[str, Any] | None = None,
) -> TrainingAugmentationPlan:
    normalized = str(name).strip().lower()
    values = dict(parameters or {})
    if normalized == AUGMENTATION_NONE:
        if values:
            raise ValueError("augmentation=none 不可帶參數")
        return TrainingAugmentationPlan(name=AUGMENTATION_NONE)
    if normalized == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
        required = {
            "probability",
            "protected_recent_bars",
            "min_mask_bars",
            "max_mask_bars",
        }
        if set(values) != required:
            raise ValueError(
                "old_history_contiguous_mask 參數必須完整且不可多餘: "
                f"expected={sorted(required)}, actual={sorted(values)}"
            )
        return TrainingAugmentationPlan(
            name=normalized,
            probability=float(values["probability"]),
            protected_recent_bars=int(values["protected_recent_bars"]),
            min_mask_bars=int(values["min_mask_bars"]),
            max_mask_bars=int(values["max_mask_bars"]),
        )
    raise ValueError(f"不支援的 training augmentation: {name!r}")


def _validate_sequence_geometry(
    plan: TrainingAugmentationPlan,
    *,
    sequence_length: int,
) -> tuple[int, int, int]:
    protected_start = int(sequence_length) - int(plan.protected_recent_bars)
    if protected_start <= 1:
        raise ValueError(
            "sequence length 不足以同時保留最近區段與左右插值邊界: "
            f"sequence_length={sequence_length}, "
            f"protected_recent_bars={plan.protected_recent_bars}"
        )
    max_legal_mask = protected_start - 1
    if int(plan.max_mask_bars) > max_legal_mask:
        raise ValueError(
            "max_mask_bars 超過可 augmentation 的舊歷史區段: "
            f"max_mask_bars={plan.max_mask_bars}, max_legal={max_legal_mask}"
        )
    return protected_start, int(plan.min_mask_bars), int(plan.max_mask_bars)


def validate_training_augmentation_sequence_length(
    plan: TrainingAugmentationPlan,
    *,
    sequence_length: int,
) -> None:
    if int(sequence_length) < 1:
        raise ValueError("sequence_length 必須 >=1")
    if plan.name == AUGMENTATION_NONE:
        return
    _validate_sequence_geometry(plan, sequence_length=int(sequence_length))


def apply_training_augmentation(
    features: np.ndarray,
    *,
    plan: TrainingAugmentationPlan,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Apply training-only masking without mutating the feature bank.

    A selected contiguous block is replaced by per-channel linear interpolation
    between the immediate left and right boundary bars. All channels share the
    same time block for one sample, and the protected recent tail is never edited.
    """

    array = np.asarray(features)
    if array.ndim != 3:
        raise ValueError(
            "training augmentation features 必須是 [batch, sequence, channel]: "
            f"shape={array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"training augmentation features 必須是浮點數: dtype={array.dtype}")
    if not bool(np.isfinite(array).all()):
        raise ValueError("training augmentation 輸入 features 必須全部有限")

    batch_size, sequence_length, _channel_count = array.shape
    if plan.name == AUGMENTATION_NONE or batch_size == 0:
        return array, {
            "name": plan.name,
            "sample_count": int(batch_size),
            "augmented_sample_count": 0,
            "masked_bar_count": 0,
        }

    protected_start, min_mask_bars, max_mask_bars = _validate_sequence_geometry(
        plan,
        sequence_length=int(sequence_length),
    )
    output = np.array(array, dtype=array.dtype, copy=True, order="C")
    selected = rng.random(int(batch_size)) < float(plan.probability)
    augmented_sample_count = 0
    masked_bar_count = 0

    for sample_index in np.flatnonzero(selected):
        mask_length = int(rng.integers(min_mask_bars, max_mask_bars + 1))
        latest_start = protected_start - mask_length
        if latest_start < 1:
            raise ValueError(
                "mask geometry 無法保留左右插值邊界: "
                f"protected_start={protected_start}, mask_length={mask_length}"
            )
        mask_start = int(rng.integers(1, latest_start + 1))
        mask_stop = mask_start + mask_length
        left = output[sample_index, mask_start - 1, :]
        right = output[sample_index, mask_stop, :]
        fractions = (
            np.arange(1, mask_length + 1, dtype=output.dtype)
            / np.asarray(mask_length + 1, dtype=output.dtype)
        )[:, None]
        output[sample_index, mask_start:mask_stop, :] = (
            left[None, :] * (1.0 - fractions) + right[None, :] * fractions
        )
        augmented_sample_count += 1
        masked_bar_count += mask_length

    if not bool(np.isfinite(output).all()):
        raise ValueError("training augmentation 輸出 features 必須全部有限")
    if not np.array_equal(
        output[:, protected_start:, :],
        array[:, protected_start:, :],
    ):
        raise ValueError("training augmentation 不可修改受保護的最近 bars")
    return output, {
        "name": plan.name,
        "sample_count": int(batch_size),
        "augmented_sample_count": int(augmented_sample_count),
        "masked_bar_count": int(masked_bar_count),
    }


__all__ = [
    "AUGMENTATION_NONE",
    "AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK",
    "TrainingAugmentationPlan",
    "apply_training_augmentation",
    "build_training_augmentation_plan",
    "validate_training_augmentation_sequence_length",
]
