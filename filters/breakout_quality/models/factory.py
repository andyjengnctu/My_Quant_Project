"""Single model factory for breakout quality training and inference."""

from __future__ import annotations

from typing import Mapping

from config.breakout_quality_policy import BREAKOUT_QUALITY_MODEL_ARCHITECTURE
from filters.breakout_quality.models.multiscale_cnn import build_multiscale_cnn
from filters.breakout_quality.models.residual_tcn import build_residual_tcn
from filters.breakout_quality.models.spec import (
    MULTISCALE_CNN_V1,
    MULTISCALE_CNN_V2,
    MULTISCALE_CNN_V3,
    MULTISCALE_CNN_V4,
    MULTISCALE_CNN_V5,
    MULTISCALE_CNN_V6,
    MULTISCALE_CNN_V7,
    MULTISCALE_CNN_V8,
    MULTISCALE_CNN_V10,
    RESIDUAL_TCN_V1,
    TINY_CNN_V1,
    get_model_spec,
    model_spec_from_manifest,
    normalize_model_architecture,
)
from filters.breakout_quality.models.tiny_cnn import build_tiny_cnn


def require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError as exc:
        raise RuntimeError("breakout quality DL 訓練需要 PyTorch；請先安裝 torch") from exc
    return torch, nn


def resolve_model_spec(
    *,
    architecture: str | None = None,
    model_spec: Mapping[str, object] | None = None,
):
    if model_spec is not None:
        resolved = model_spec_from_manifest(model_spec)
        if architecture is not None and normalize_model_architecture(architecture) != resolved.architecture:
            raise ValueError("model architecture 與 model_spec.architecture 不一致")
        return resolved
    return get_model_spec(architecture or BREAKOUT_QUALITY_MODEL_ARCHITECTURE)


def build_model(
    feature_count: int,
    context_count: int,
    *,
    architecture: str | None = None,
    model_spec: Mapping[str, object] | None = None,
):
    torch, nn = require_torch()
    spec = resolve_model_spec(architecture=architecture, model_spec=model_spec)
    if spec.architecture == TINY_CNN_V1:
        return build_tiny_cnn(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
        )
    if spec.architecture in {
        MULTISCALE_CNN_V1,
        MULTISCALE_CNN_V2,
        MULTISCALE_CNN_V3,
        MULTISCALE_CNN_V4,
        MULTISCALE_CNN_V5,
        MULTISCALE_CNN_V6,
        MULTISCALE_CNN_V7,
        MULTISCALE_CNN_V8,
        MULTISCALE_CNN_V10,
    }:
        return build_multiscale_cnn(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == RESIDUAL_TCN_V1:
        return build_residual_tcn(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    raise AssertionError(f"未處理的 model architecture: {spec.architecture}")


def count_trainable_parameters(model) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "build_model",
    "count_trainable_parameters",
    "require_torch",
    "resolve_model_spec",
]
