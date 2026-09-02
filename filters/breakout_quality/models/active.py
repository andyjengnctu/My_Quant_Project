"""Active breakout quality model API used by formal new training workflows."""

from __future__ import annotations

from typing import Mapping

from filters.breakout_quality.models.architectures import (
    ACTIVE_MODEL_ARCHITECTURES,
    get_architecture_descriptor,
)
from filters.breakout_quality.models.runtime import count_trainable_parameters, require_torch
from filters.breakout_quality.models.runtime_registry import build_registered_model
from filters.breakout_quality.models.spec import (
    get_model_spec,
    model_spec_from_manifest,
    normalize_active_model_architecture,
)


def get_active_model_spec(architecture: str):
    normalized = normalize_active_model_architecture(architecture)
    return get_model_spec(normalized)


def resolve_active_model_spec(
    *,
    architecture: str | None = None,
    model_spec: Mapping[str, object] | None = None,
):
    if model_spec is not None:
        resolved = model_spec_from_manifest(model_spec)
        normalize_active_model_architecture(resolved.architecture)
        if architecture is not None and normalize_active_model_architecture(architecture) != resolved.architecture:
            raise ValueError("model architecture 與 model_spec.architecture 不一致")
        return resolved
    if architecture is None:
        raise ValueError("active model architecture 不得為空")
    return get_active_model_spec(architecture)


def build_active_model(
    feature_count: int,
    context_count: int,
    *,
    architecture: str,
    model_spec: Mapping[str, object] | None = None,
    pretrained_encoder_state: Mapping[str, object] | None = None,
):
    """Build only architectures that are allowed for formal new training."""

    if pretrained_encoder_state is not None:
        raise ValueError("active architecture 不接受 legacy pretrained encoder state")
    spec = resolve_active_model_spec(architecture=architecture, model_spec=model_spec)
    descriptor = get_architecture_descriptor(spec.architecture)
    if not descriptor.active:
        raise AssertionError(f"active resolver取得非active architecture: {spec.architecture}")
    torch, nn = require_torch()
    return build_registered_model(
        nn,
        torch,
        descriptor=descriptor,
        feature_count=int(feature_count),
        context_count=int(context_count),
        spec=spec,
        pretrained_encoder_state=None,
    )


__all__ = [
    "ACTIVE_MODEL_ARCHITECTURES",
    "build_active_model",
    "count_trainable_parameters",
    "get_active_model_spec",
    "require_torch",
    "resolve_active_model_spec",
]
