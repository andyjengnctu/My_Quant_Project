"""Read-only historical model construction compatibility for archived artifacts."""

from __future__ import annotations

from typing import Mapping

from filters.breakout_quality.models.runtime import require_torch
from filters.breakout_quality.models.spec import (
    INCEPTION_TIME_GROUP_NORM_V1,
    INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
    INCEPTION_TIME_MARKET_SET_V1,
    LEGACY_MODEL_ARCHITECTURES,
    MANTIS_V2_FROZEN_LINEAR_V1,
    MOMENT_1_BASE_FROZEN_LINEAR_V1,
    MODERN_TCN_V1,
    MULTISCALE_CNN_REGIME_CONTEXT_V1,
    MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1,
    MULTISCALE_CNN_V1,
    MULTISCALE_CNN_V2,
    MULTISCALE_CNN_V3,
    MULTISCALE_CNN_V4,
    MULTISCALE_CNN_V5,
    MULTISCALE_CNN_V6,
    MULTISCALE_CNN_V7,
    MULTISCALE_CNN_V8,
    PATCH_TRANSFORMER_V1,
    RESIDUAL_TCN_V1,
    TINY_CNN_V1,
    TS2VEC_FROZEN_LINEAR_V1,
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
        if resolved.architecture not in LEGACY_MODEL_ARCHITECTURES:
            raise ValueError(
                f"architecture {resolved.architecture!r} 不是 historical compatibility architecture"
            )
        if architecture is not None and normalize_model_architecture(architecture) != resolved.architecture:
            raise ValueError("model architecture 與 model_spec.architecture 不一致")
        return resolved
    if architecture is None:
        raise ValueError("legacy model architecture 不得為空")
    normalized = normalize_model_architecture(architecture)
    if normalized not in LEGACY_MODEL_ARCHITECTURES:
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

    if spec.architecture == MOMENT_1_BASE_FROZEN_LINEAR_V1:
        from filters.breakout_quality.models.moment import build_moment_frozen_linear

        return build_moment_frozen_linear(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
            pretrained_encoder_state=pretrained_encoder_state,
        )
    if spec.architecture == MANTIS_V2_FROZEN_LINEAR_V1:
        from filters.breakout_quality.models.mantis_v2 import build_mantis_v2_frozen_linear

        return build_mantis_v2_frozen_linear(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
            pretrained_encoder_state=pretrained_encoder_state,
        )
    if spec.architecture == TS2VEC_FROZEN_LINEAR_V1:
        from filters.breakout_quality.models.ts2vec import build_ts2vec_frozen_linear

        return build_ts2vec_frozen_linear(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
            pretrained_encoder_state=pretrained_encoder_state,
        )
    if spec.architecture == PATCH_TRANSFORMER_V1:
        from filters.breakout_quality.models.patch_transformer import build_patch_transformer

        return build_patch_transformer(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == MODERN_TCN_V1:
        from filters.breakout_quality.models.modern_tcn import build_modern_tcn

        return build_modern_tcn(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture in {
        INCEPTION_TIME_MARKET_SET_V1,
        INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
    }:
        from filters.breakout_quality.models.inception_time_market_set import (
            build_inception_time_market_set,
        )

        return build_inception_time_market_set(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == INCEPTION_TIME_GROUP_NORM_V1:
        from filters.breakout_quality.models.inception_time import build_inception_time

        return build_inception_time(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == TINY_CNN_V1:
        from filters.breakout_quality.models.tiny_cnn import build_tiny_cnn

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
        MULTISCALE_CNN_REGIME_CONTEXT_V1,
        MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1,
    }:
        from filters.breakout_quality.models.multiscale_cnn import build_multiscale_cnn

        return build_multiscale_cnn(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == RESIDUAL_TCN_V1:
        from filters.breakout_quality.models.residual_tcn import build_residual_tcn

        return build_residual_tcn(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    raise AssertionError(f"未處理的 historical model architecture: {spec.architecture}")


__all__ = [
    "build_legacy_model",
    "resolve_legacy_model_spec",
]
