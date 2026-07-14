"""Breakout quality model implementations and versioned specifications."""

from filters.breakout_quality.models.factory import (
    build_model,
    count_trainable_parameters,
    require_torch,
    resolve_model_spec,
)
from filters.breakout_quality.models.spec import (
    MULTISCALE_CNN_V1,
    MULTISCALE_CNN_V2,
    MULTISCALE_CNN_V3,
    RESIDUAL_TCN_V1,
    SUPPORTED_MODEL_ARCHITECTURES,
    TINY_CNN_V1,
    BreakoutQualityModelSpec,
    get_model_spec,
    model_spec_from_manifest,
    normalize_model_architecture,
)

__all__ = [
    "BreakoutQualityModelSpec",
    "MULTISCALE_CNN_V1",
    "MULTISCALE_CNN_V2",
    "MULTISCALE_CNN_V3",
    "RESIDUAL_TCN_V1",
    "SUPPORTED_MODEL_ARCHITECTURES",
    "TINY_CNN_V1",
    "build_model",
    "count_trainable_parameters",
    "get_model_spec",
    "model_spec_from_manifest",
    "normalize_model_architecture",
    "require_torch",
    "resolve_model_spec",
]
