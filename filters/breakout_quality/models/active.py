"""Active breakout quality model API used by formal new training workflows."""

from __future__ import annotations

from typing import Mapping

from filters.breakout_quality.models.inception_time import build_inception_time
from filters.breakout_quality.models.large_kernel_tcn_joint_min import build_modern_tcn_joint_min
from filters.breakout_quality.models.multiscale_cnn import build_multiscale_cnn
from filters.breakout_quality.models.patch_token_joint_min import (
    build_patch_token_joint_min,
    build_patch_token_ranker,
)
from filters.breakout_quality.models.runtime import (
    count_trainable_parameters,
    require_torch,
)
from filters.breakout_quality.models.spec import (
    ACTIVE_MODEL_ARCHITECTURES,
    INCEPTION_TIME_V1,
    INCEPTION_TIME_RISK_CONTEXT_V1,
    INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1,
    INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    PATCH_TOKEN_TRANSFORMER_RANKER_V1,
    MULTISCALE_CNN_SEQUENCE_ONLY_V1,
    get_model_spec,
    model_spec_from_manifest,
    normalize_active_model_architecture,
)


def get_active_model_spec(architecture: str):
    """Resolve a model spec only when the architecture is valid for new training."""

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
        if (
            architecture is not None
            and normalize_active_model_architecture(architecture) != resolved.architecture
        ):
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
    spec = resolve_active_model_spec(
        architecture=architecture,
        model_spec=model_spec,
    )
    torch, nn = require_torch()
    if spec.architecture in {
        INCEPTION_TIME_V1,
        INCEPTION_TIME_RISK_CONTEXT_V1,
        INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1,
        INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1,
        INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1,
        INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1,
        INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    }:
        return build_inception_time(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1:
        return build_modern_tcn_joint_min(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1:
        return build_patch_token_joint_min(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == PATCH_TOKEN_TRANSFORMER_RANKER_V1:
        return build_patch_token_ranker(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    if spec.architecture == MULTISCALE_CNN_SEQUENCE_ONLY_V1:
        return build_multiscale_cnn(
            nn,
            torch,
            feature_count=int(feature_count),
            context_count=int(context_count),
            spec=spec,
        )
    raise AssertionError(f"未處理的 active model architecture: {spec.architecture}")


__all__ = [
    "ACTIVE_MODEL_ARCHITECTURES",
    "build_active_model",
    "count_trainable_parameters",
    "get_active_model_spec",
    "require_torch",
    "resolve_active_model_spec",
]
