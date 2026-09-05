"""Public compatibility façade for versioned breakout-quality model specs.

Architecture identities live in :mod:`architectures`; construction lives in the
registry/builders modules.  Existing imports through ``models.spec`` remain
stable for artifact reconstruction and callers.
"""

from __future__ import annotations

from typing import Mapping

from filters.breakout_quality.models.architectures import *
from filters.breakout_quality.models.spec_contract import BreakoutQualityModelSpec
from filters.breakout_quality.models.spec_registry import get_model_spec


def validate_model_sequence_length(
    model_spec: BreakoutQualityModelSpec, sequence_length: int
) -> None:
    normalized_length = int(sequence_length)
    if normalized_length < 1:
        raise ValueError("model sequence_length 必須 >= 1")
    if model_spec.input_window_bars is not None and int(model_spec.input_window_bars) != normalized_length:
        raise ValueError(
            "model sequence_length 與 architecture input window 不一致: "
            f"sequence_length={normalized_length}, expected={int(model_spec.input_window_bars)}"
        )
    if bool(model_spec.requires_market_set):
        market_history = int(model_spec.market_set_history_bars or 0)
        if market_history != normalized_length:
            raise ValueError(
                "Market Set 第一版要求 candidate sequence_length 與 market history 一致: "
                f"candidate={normalized_length}, market={market_history}"
            )
    if model_spec.family not in {
        "patch_transformer",
        "patch_token_transformer_ranker",
        "patch_token_transformer_safety_raw_mfe_joint_attn_mlp",
        "patch_transformer_safety_inception_mfe",
    }:
        return
    patch_size = int(model_spec.patch_transformer_patch_size or 0)
    patch_stride = int(model_spec.patch_transformer_patch_stride or 0)
    if patch_size < 1 or patch_stride != patch_size:
        raise ValueError("Patch Transformer 必須使用有效的非重疊 patch spec")
    if normalized_length < patch_size or normalized_length % patch_size != 0:
        raise ValueError(
            "Patch Transformer sequence_length 必須可被 patch size 整除: "
            f"sequence_length={normalized_length}, patch_size={patch_size}"
        )


def model_spec_from_manifest(payload: Mapping[str, object]) -> BreakoutQualityModelSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("breakout quality model_spec 必須是 object")
    architecture = normalize_model_architecture(str(payload.get("architecture", "")))
    expected = get_model_spec(architecture)
    actual = dict(payload)
    if actual != expected.as_manifest_payload():
        raise ValueError(
            "breakout quality model_spec 與版本化正式規格不一致: "
            f"architecture={architecture}, expected={expected.as_manifest_payload()}, actual={actual}"
        )
    return expected


__all__ = [
    "ACTIVE_MODEL_ARCHITECTURES",
    "BreakoutQualityModelSpec",
    "INCEPTION_TIME_GROUP_NORM_V1",
    "INCEPTION_TIME_MARKET_SET_CANDIDATE_V1",
    "INCEPTION_TIME_MARKET_SET_V1",
    "INCEPTION_TIME_V1",
    "INCEPTION_TIME_RISK_CONTEXT_V1",
    "INCEPTION_TIME_PREDICTED_UPSIDE_CONTEXT_V1",
    "INCEPTION_TIME_PREDICTED_SAFETY_CONTEXT_V1",
    "INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1",
    "INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_DYNAMIC_HYPERGRAPH_MFE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_PRICE_VOLUME_STRUCTURE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_PRICE_VOLUME_STRUCTURE_LOCAL_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_PRICE_VOLUME_MULTISCALE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_PRICE_VOLUME_POSITION_AWARE_MULTISCALE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_PAIRWISE_RELATION_MFE_V1",
    "INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_ATTN_MFE_V1",
    "INCEPTION_TIME_TASK_SPECIFIC_SAFETY_ATTN_MFE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_SELF_ATTN_MFE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_FULL_WINDOW_RF_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_WIDE_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_600BAR_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_DEEP_V1",
    "INCEPTION_TIME_SHARED_SAFETY_MFE_600BAR_WIDE_V1",
    "DAY_TOKEN_TRANSFORMER_SHARED_SAFETY_MFE_V1",
    "GRU_SHARED_SAFETY_MFE_V1",
    "GRU_SHARED_SAFETY_MFE_V2",
    "GRU_SHARED_SAFETY_MFE_V3",
    "GRU_SHARED_SAFETY_MFE_V4",
    "PATCH_TRANSFORMER_SAFETY_INCEPTION_MFE_V1",
    "INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1",
    "INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1",
    "INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1",
    "LEGACY_MODEL_ARCHITECTURES",
    "MULTISCALE_CNN_V1",
    "MULTISCALE_CNN_V2",
    "MULTISCALE_CNN_V3",
    "MULTISCALE_CNN_V4",
    "MULTISCALE_CNN_V5",
    "MULTISCALE_CNN_V6",
    "MULTISCALE_CNN_V7",
    "MULTISCALE_CNN_V8",
    "MULTISCALE_CNN_REGIME_CONTEXT_V1",
    "MULTISCALE_CNN_SEQUENCE_ONLY_V1",
    "MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1",
    "MODERN_TCN_V1",
    "MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1",
    "PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1",
    "PATCH_TOKEN_TRANSFORMER_RANKER_V1",
    "MANTIS_V2_FROZEN_LINEAR_V1",
    "MOMENT_1_BASE_FROZEN_LINEAR_V1",
    "PATCH_TRANSFORMER_V1",
    "TS2VEC_FROZEN_LINEAR_V1",
    "RESIDUAL_TCN_V1",
    "SUPPORTED_MODEL_ARCHITECTURES",
    "TINY_CNN_V1",
    "get_model_spec",
    "model_spec_from_manifest",
    "normalize_active_model_architecture",
    "normalize_model_architecture",
    "validate_model_sequence_length",
]
