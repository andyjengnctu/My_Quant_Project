"""Step-based learning-rate schedule helpers for breakout-quality training artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from core.breakout_quality_registry import (
    LR_SCHEDULE_LINEAR_WARMUP_COSINE,
    LR_SCHEDULE_NONE,
)


def build_learning_rate_schedule_plan(
    *,
    schedule_name: str,
    base_learning_rate: float,
    total_optimizer_steps: int,
    warmup_fraction: float,
    minimum_lr_ratio: float,
) -> dict[str, object]:
    normalized = str(schedule_name).strip().lower()
    total_steps = int(total_optimizer_steps)
    base_lr = float(base_learning_rate)
    if total_steps < 1:
        raise ValueError("LR schedule total_optimizer_steps 必須 >=1")
    if not math.isfinite(base_lr) or base_lr <= 0.0:
        raise ValueError("LR schedule base learning rate 必須為有限正數")
    if normalized == LR_SCHEDULE_NONE:
        return {
            "name": LR_SCHEDULE_NONE,
            "base_learning_rate": base_lr,
            "minimum_learning_rate": base_lr,
            "minimum_lr_ratio": 1.0,
            "warmup_fraction": 0.0,
            "warmup_steps": 0,
            "total_optimizer_steps": total_steps,
        }
    if normalized != LR_SCHEDULE_LINEAR_WARMUP_COSINE:
        raise ValueError(f"不支援的 LR schedule: {schedule_name!r}")
    warmup = float(warmup_fraction)
    minimum_ratio = float(minimum_lr_ratio)
    if not 0.0 < warmup < 1.0:
        raise ValueError("LR warmup fraction 必須介於 0 與 1 之間")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum LR ratio 必須介於 0 與 1 之間")
    raw_warmup_steps = max(1, int(math.ceil(total_steps * warmup)))
    warmup_steps = (
        min(raw_warmup_steps, total_steps - 1)
        if total_steps > 1
        else 1
    )
    return {
        "name": normalized,
        "base_learning_rate": base_lr,
        "minimum_learning_rate": base_lr * minimum_ratio,
        "minimum_lr_ratio": minimum_ratio,
        "warmup_fraction": warmup,
        "warmup_steps": int(warmup_steps),
        "total_optimizer_steps": total_steps,
    }


def learning_rate_for_optimizer_step(
    schedule_plan: Mapping[str, object],
    optimizer_step_index: int,
) -> float:
    step_index = int(optimizer_step_index)
    total_steps = int(schedule_plan["total_optimizer_steps"])
    if step_index < 0 or step_index >= total_steps:
        raise ValueError(
            "optimizer step 超出 LR schedule 範圍: "
            f"step={step_index}, total={total_steps}"
        )
    base_lr = float(schedule_plan["base_learning_rate"])
    if str(schedule_plan["name"]) == LR_SCHEDULE_NONE:
        return base_lr
    warmup_steps = int(schedule_plan["warmup_steps"])
    if step_index < warmup_steps:
        return base_lr * float(step_index + 1) / float(warmup_steps)
    minimum_lr = float(schedule_plan["minimum_learning_rate"])
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return minimum_lr
    decay_index = step_index - warmup_steps
    progress = float(decay_index) / float(decay_steps - 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_lr + (base_lr - minimum_lr) * cosine_factor


def validate_learning_rate_schedule_record(
    record: Mapping[str, Any],
    *,
    schedule_name: str,
    schedule_parameters: Mapping[str, Any],
    base_learning_rate: float,
    expected_total_optimizer_steps: int,
    expected_actual_optimizer_steps: int,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("learning-rate schedule plan 必須是 object")
    expected_plan = build_learning_rate_schedule_plan(
        schedule_name=schedule_name,
        base_learning_rate=base_learning_rate,
        total_optimizer_steps=expected_total_optimizer_steps,
        warmup_fraction=float(schedule_parameters.get("warmup_fraction", 0.0)),
        minimum_lr_ratio=float(schedule_parameters.get("minimum_lr_ratio", 1.0)),
    )
    for field_name, expected_value in expected_plan.items():
        actual_value = record.get(field_name)
        if isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                "learning-rate schedule plan 與 profile/steps 不一致: "
                f"field={field_name}, actual={actual_value}, expected={expected_value}"
            )
    actual_steps = int(record.get("actual_optimizer_steps", -1))
    if actual_steps != int(expected_actual_optimizer_steps):
        raise ValueError(
            "learning-rate schedule actual optimizer steps 不一致: "
            f"actual={actual_steps}, expected={expected_actual_optimizer_steps}"
        )
    if actual_steps < 1 or actual_steps > int(expected_total_optimizer_steps):
        raise ValueError("learning-rate schedule actual optimizer steps 超出合法範圍")
    expected_last_lr = learning_rate_for_optimizer_step(
        expected_plan,
        actual_steps - 1,
    )
    try:
        actual_last_lr = float(record.get("last_applied_learning_rate"))
    except (TypeError, ValueError) as exc:
        raise ValueError("learning-rate schedule 缺少合法 last_applied_learning_rate") from exc
    if not math.isclose(actual_last_lr, expected_last_lr, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "learning-rate schedule last_applied_learning_rate 不一致: "
            f"actual={actual_last_lr}, expected={expected_last_lr}"
        )


__all__ = [
    "build_learning_rate_schedule_plan",
    "learning_rate_for_optimizer_step",
    "validate_learning_rate_schedule_record",
]
