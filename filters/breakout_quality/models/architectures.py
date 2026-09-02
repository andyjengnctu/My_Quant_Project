"""Canonical breakout-quality model architecture identities."""

from __future__ import annotations

TINY_CNN_V1 = "tiny_cnn_v1"
MULTISCALE_CNN_V1 = "multiscale_cnn_v1"
MULTISCALE_CNN_V2 = "multiscale_cnn_v2"
MULTISCALE_CNN_V3 = "multiscale_cnn_v3"
MULTISCALE_CNN_V4 = "multiscale_cnn_v4"
MULTISCALE_CNN_V5 = "multiscale_cnn_v5"
MULTISCALE_CNN_V6 = "multiscale_cnn_v6"
MULTISCALE_CNN_V7 = "multiscale_cnn_v7"
MULTISCALE_CNN_V8 = "multiscale_cnn_v8"
MULTISCALE_CNN_REGIME_CONTEXT_V1 = "multiscale_cnn_regime_context_v1"
MULTISCALE_CNN_SEQUENCE_ONLY_V1 = "multiscale_cnn_sequence_only_v1"
MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1 = "multiscale_cnn_sequence_only_dual_path_v1"
INCEPTION_TIME_V1 = "inception_time_v1"
INCEPTION_TIME_RISK_CONTEXT_V1 = "inception_time_risk_context_v1"
INCEPTION_TIME_PREDICTED_UPSIDE_CONTEXT_V1 = "inception_time_predicted_upside_context_v1"
INCEPTION_TIME_PREDICTED_SAFETY_CONTEXT_V1 = "inception_time_predicted_safety_context_v1"
INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1 = "inception_time_conditional_mfe_safety_v1"
INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1 = "inception_time_safety_conditional_mfe_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_V1 = "inception_time_shared_safety_mfe_v1"
INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1 = "inception_time_task_specific_safety_mfe_v1"
INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1 = "inception_time_safety_raw_mfe_hmhs_v1"
INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1 = "inception_time_safety_raw_mfe_hmhs_mlp_v1"
INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1 = "inception_time_safety_raw_mfe_joint_attn_mlp_v1"
INCEPTION_TIME_MARKET_SET_V1 = "inception_time_market_set_v1"
INCEPTION_TIME_MARKET_SET_CANDIDATE_V1 = "inception_time_market_set_candidate_v1"
INCEPTION_TIME_GROUP_NORM_V1 = "inception_time_group_norm_v1"
MODERN_TCN_V1 = "modern_tcn_v1"
MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1 = "modern_tcn_safety_raw_mfe_joint_attn_mlp_v1"
PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1 = (
    "patch_token_transformer_safety_raw_mfe_joint_attn_mlp_v1"
)
PATCH_TOKEN_TRANSFORMER_RANKER_V1 = "patch_token_transformer_ranker_v1"
MANTIS_V2_FROZEN_LINEAR_V1 = "mantis_v2_frozen_linear_v1"
MOMENT_1_BASE_FROZEN_LINEAR_V1 = "moment_1_base_frozen_linear_v1"
PATCH_TRANSFORMER_V1 = "patch_transformer_v1"
TS2VEC_FROZEN_LINEAR_V1 = "ts2vec_frozen_linear_v1"
RESIDUAL_TCN_V1 = "residual_tcn_v1"

SUPPORTED_MODEL_ARCHITECTURES = (
    TINY_CNN_V1,
    MULTISCALE_CNN_V1,
    MULTISCALE_CNN_V2,
    MULTISCALE_CNN_V3,
    MULTISCALE_CNN_V4,
    MULTISCALE_CNN_V5,
    MULTISCALE_CNN_V6,
    MULTISCALE_CNN_V7,
    MULTISCALE_CNN_V8,
    MULTISCALE_CNN_REGIME_CONTEXT_V1,
    MULTISCALE_CNN_SEQUENCE_ONLY_V1,
    MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1,
    INCEPTION_TIME_V1,
    INCEPTION_TIME_RISK_CONTEXT_V1,
    INCEPTION_TIME_PREDICTED_UPSIDE_CONTEXT_V1,
    INCEPTION_TIME_PREDICTED_SAFETY_CONTEXT_V1,
    INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1,
    INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1,
    INCEPTION_TIME_SHARED_SAFETY_MFE_V1,
    INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    INCEPTION_TIME_MARKET_SET_V1,
    INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
    INCEPTION_TIME_GROUP_NORM_V1,
    MODERN_TCN_V1,
    MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    PATCH_TOKEN_TRANSFORMER_RANKER_V1,
    MANTIS_V2_FROZEN_LINEAR_V1,
    MOMENT_1_BASE_FROZEN_LINEAR_V1,
    PATCH_TRANSFORMER_V1,
    TS2VEC_FROZEN_LINEAR_V1,
    RESIDUAL_TCN_V1,
)

ACTIVE_MODEL_ARCHITECTURES = (
    INCEPTION_TIME_V1,
    INCEPTION_TIME_RISK_CONTEXT_V1,
    INCEPTION_TIME_PREDICTED_UPSIDE_CONTEXT_V1,
    INCEPTION_TIME_PREDICTED_SAFETY_CONTEXT_V1,
    INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1,
    INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1,
    INCEPTION_TIME_SHARED_SAFETY_MFE_V1,
    INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
    PATCH_TOKEN_TRANSFORMER_RANKER_V1,
    MULTISCALE_CNN_SEQUENCE_ONLY_V1,
)

LEGACY_MODEL_ARCHITECTURES = tuple(
    architecture
    for architecture in SUPPORTED_MODEL_ARCHITECTURES
    if architecture not in ACTIVE_MODEL_ARCHITECTURES
)


def normalize_model_architecture(value: str) -> str:
    architecture = str(value).strip().lower()
    if architecture not in SUPPORTED_MODEL_ARCHITECTURES:
        allowed = ", ".join(SUPPORTED_MODEL_ARCHITECTURES)
        raise ValueError(
            f"不支援的 breakout quality model architecture: {value!r}；可用值: {allowed}"
        )
    return architecture


def normalize_active_model_architecture(value: str) -> str:
    architecture = normalize_model_architecture(value)
    if architecture not in ACTIVE_MODEL_ARCHITECTURES:
        allowed = ", ".join(ACTIVE_MODEL_ARCHITECTURES)
        raise ValueError(
            "breakout quality 正式新訓練只允許 active architecture；"
            f"收到 {architecture!r}，可用值: {allowed}。歷史版本僅供舊工件重現。"
        )
    return architecture
