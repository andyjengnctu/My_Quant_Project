"""Registry-driven breakout-quality model-spec resolver."""

from __future__ import annotations

from filters.breakout_quality.models.architectures import *
from filters.breakout_quality.models.spec_builders import (
    ModelSpecBuilder,
    build_inception_group_norm_spec,
    build_inception_hmhs_mlp_spec,
    build_inception_joint_spec,
    build_inception_variant_spec,
    build_mantis_spec,
    build_market_set_spec,
    build_modern_tcn_joint_spec,
    build_modern_tcn_spec,
    build_moment_spec,
    build_multiscale_cnn_spec,
    build_patch_token_joint_spec,
    build_patch_token_ranker_spec,
    build_patch_transformer_spec,
    build_residual_tcn_spec,
    build_tiny_cnn_spec,
    build_ts2vec_spec,
)
from filters.breakout_quality.models.spec_contract import BreakoutQualityModelSpec


MODEL_SPEC_BUILDERS: dict[str, ModelSpecBuilder] = {
    TINY_CNN_V1: build_tiny_cnn_spec,
    **{architecture: build_multiscale_cnn_spec for architecture in (
        MULTISCALE_CNN_V1, MULTISCALE_CNN_V2, MULTISCALE_CNN_V3, MULTISCALE_CNN_V4,
        MULTISCALE_CNN_V5, MULTISCALE_CNN_V6, MULTISCALE_CNN_V7, MULTISCALE_CNN_V8,
        MULTISCALE_CNN_REGIME_CONTEXT_V1, MULTISCALE_CNN_SEQUENCE_ONLY_V1,
        MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1,
    )},
    PATCH_TRANSFORMER_V1: build_patch_transformer_spec,
    MOMENT_1_BASE_FROZEN_LINEAR_V1: build_moment_spec,
    MANTIS_V2_FROZEN_LINEAR_V1: build_mantis_spec,
    TS2VEC_FROZEN_LINEAR_V1: build_ts2vec_spec,
    MODERN_TCN_V1: build_modern_tcn_spec,
    MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1: build_modern_tcn_joint_spec,
    PATCH_TOKEN_TRANSFORMER_RANKER_V1: build_patch_token_ranker_spec,
    PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1: build_patch_token_joint_spec,
    INCEPTION_TIME_MARKET_SET_V1: build_market_set_spec,
    INCEPTION_TIME_MARKET_SET_CANDIDATE_V1: build_market_set_spec,
    **{architecture: build_inception_variant_spec for architecture in (
        INCEPTION_TIME_V1,
        INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1,
        INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1,
        INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1,
        INCEPTION_TIME_PREDICTED_UPSIDE_CONTEXT_V1,
        INCEPTION_TIME_PREDICTED_SAFETY_CONTEXT_V1,
        INCEPTION_TIME_RISK_CONTEXT_V1,
    )},
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1: build_inception_hmhs_mlp_spec,
    INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1: build_inception_joint_spec,
    INCEPTION_TIME_GROUP_NORM_V1: build_inception_group_norm_spec,
    RESIDUAL_TCN_V1: build_residual_tcn_spec,
}

if set(MODEL_SPEC_BUILDERS) != set(SUPPORTED_MODEL_ARCHITECTURES):
    missing = sorted(set(SUPPORTED_MODEL_ARCHITECTURES) - set(MODEL_SPEC_BUILDERS))
    extra = sorted(set(MODEL_SPEC_BUILDERS) - set(SUPPORTED_MODEL_ARCHITECTURES))
    raise RuntimeError(f"model spec registry mismatch: missing={missing}, extra={extra}")


def get_model_spec(architecture: str) -> BreakoutQualityModelSpec:
    normalized = normalize_model_architecture(architecture)
    return MODEL_SPEC_BUILDERS[normalized](normalized)
