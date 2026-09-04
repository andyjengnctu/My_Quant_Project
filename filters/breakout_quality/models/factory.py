"""Compatibility model factory for active and historical breakout quality artifacts."""

from __future__ import annotations

from typing import Mapping

from config.breakout_quality import (
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from filters.breakout_quality.models.active import build_active_model
from filters.breakout_quality.models.runtime import (
    count_trainable_parameters,
    require_torch,
)
from filters.breakout_quality.models.spec import (
    ACTIVE_MODEL_ARCHITECTURES,
    get_model_spec,
    model_spec_from_manifest,
    normalize_model_architecture,
)


def resolve_model_spec(
    *,
    architecture: str | None = None,
    model_spec: Mapping[str, object] | None = None,
    pretrained_encoder_state: Mapping[str, object] | None = None,
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
    pretrained_encoder_state: Mapping[str, object] | None = None,
):
    """Build active models or lazily reconstruct historical artifact models."""

    spec = resolve_model_spec(architecture=architecture, model_spec=model_spec)
    if spec.architecture in ACTIVE_MODEL_ARCHITECTURES:
        return build_active_model(
            feature_count,
            context_count,
            architecture=spec.architecture,
            model_spec=spec.as_manifest_payload(),
            pretrained_encoder_state=pretrained_encoder_state,
        )

    from filters.breakout_quality.models.legacy_compatibility import build_legacy_model

    return build_legacy_model(
        feature_count,
        context_count,
        architecture=spec.architecture,
        model_spec=spec.as_manifest_payload(),
        pretrained_encoder_state=pretrained_encoder_state,
    )


__all__ = [
    "build_model",
    "count_trainable_parameters",
    "require_torch",
    "resolve_model_spec",
]
