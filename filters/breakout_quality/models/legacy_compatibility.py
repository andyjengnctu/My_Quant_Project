"""Read-only historical model construction compatibility for archived artifacts."""

from __future__ import annotations

from typing import Mapping

from filters.breakout_quality.models.architectures import (
    LEGACY_MODEL_ARCHITECTURES,
    get_architecture_descriptor,
)
from filters.breakout_quality.models.runtime import require_torch
from filters.breakout_quality.models.runtime_registry import build_registered_model
from filters.breakout_quality.models.spec import (
    get_model_spec,
    model_spec_from_manifest,
    normalize_model_architecture,
)


def resolve_legacy_model_spec(
    *,
    architecture: str | None = None,
    model_spec: Mapping[str, object] | None = None,
):
    if model_spec is not None:
        resolved = model_spec_from_manifest(model_spec)
        if get_architecture_descriptor(resolved.architecture).active:
            raise ValueError(f"architecture {resolved.architecture!r} 不是 historical compatibility architecture")
        if architecture is not None and normalize_model_architecture(architecture) != resolved.architecture:
            raise ValueError("model architecture 與 model_spec.architecture 不一致")
        return resolved
    if architecture is None:
        raise ValueError("legacy model architecture 不得為空")
    normalized = normalize_model_architecture(architecture)
    if get_architecture_descriptor(normalized).active:
        raise ValueError(f"architecture {normalized!r} 不是 historical compatibility architecture")
    return get_model_spec(normalized)


def build_legacy_model(
    feature_count: int,
    context_count: int,
    *,
    architecture: str | None = None,
    model_spec: Mapping[str, object] | None = None,
    pretrained_encoder_state: Mapping[str, object] | None = None,
):
    """Build historical architecture only for artifact reconstruction/inference."""

    torch, nn = require_torch()
    spec = resolve_legacy_model_spec(architecture=architecture, model_spec=model_spec)
    descriptor = get_architecture_descriptor(spec.architecture)
    return build_registered_model(
        nn,
        torch,
        descriptor=descriptor,
        feature_count=int(feature_count),
        context_count=int(context_count),
        spec=spec,
        pretrained_encoder_state=pretrained_encoder_state,
    )


__all__ = ["build_legacy_model", "resolve_legacy_model_spec"]
