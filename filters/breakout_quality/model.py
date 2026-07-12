"""Backward-compatible breakout quality model factory facade."""

from __future__ import annotations

from filters.breakout_quality.models import (
    build_model,
    count_trainable_parameters,
    get_model_spec,
    require_torch,
    resolve_model_spec,
)

__all__ = [
    "build_model",
    "count_trainable_parameters",
    "get_model_spec",
    "require_torch",
    "resolve_model_spec",
]
