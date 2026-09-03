"""Descriptor-driven breakout-quality model-spec resolver."""

from __future__ import annotations

from filters.breakout_quality.models.architectures import (
    SUPPORTED_MODEL_ARCHITECTURES,
    get_architecture_descriptor,
    normalize_model_architecture,
)
from filters.breakout_quality.models.spec_builders import (
    ModelSpecBuilder,
    build_day_token_transformer_shared_safety_mfe_spec,
    build_inception_group_norm_spec,
    build_inception_hmhs_mlp_spec,
    build_inception_joint_spec,
    build_hybrid_safety_patch_mfe_inception_spec,
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


_SPEC_BUILDERS_BY_KEY: dict[str, ModelSpecBuilder] = {
    "tiny_cnn": build_tiny_cnn_spec,
    "multiscale_cnn": build_multiscale_cnn_spec,
    "patch_transformer": build_patch_transformer_spec,
    "hybrid_safety_patch_mfe_inception": build_hybrid_safety_patch_mfe_inception_spec,
    "patch_token_ranker": build_patch_token_ranker_spec,
    "patch_token_joint": build_patch_token_joint_spec,
    "moment": build_moment_spec,
    "mantis": build_mantis_spec,
    "ts2vec": build_ts2vec_spec,
    "modern_tcn": build_modern_tcn_spec,
    "modern_tcn_joint": build_modern_tcn_joint_spec,
    "inception_variant": build_inception_variant_spec,
    "inception_hmhs_mlp": build_inception_hmhs_mlp_spec,
    "inception_joint": build_inception_joint_spec,
    "inception_group_norm": build_inception_group_norm_spec,
    "day_token_transformer_shared_safety_mfe": build_day_token_transformer_shared_safety_mfe_spec,
    "market_set": build_market_set_spec,
    "residual_tcn": build_residual_tcn_spec,
}

# Compatibility view for callers/tests that historically consumed architecture -> builder.
# It is derived from the descriptor registry rather than maintained independently.
MODEL_SPEC_BUILDERS: dict[str, ModelSpecBuilder] = {
    architecture: _SPEC_BUILDERS_BY_KEY[get_architecture_descriptor(architecture).spec_builder_key]
    for architecture in SUPPORTED_MODEL_ARCHITECTURES
}


def get_model_spec(architecture: str) -> BreakoutQualityModelSpec:
    normalized = normalize_model_architecture(architecture)
    descriptor = get_architecture_descriptor(normalized)
    try:
        builder = _SPEC_BUILDERS_BY_KEY[descriptor.spec_builder_key]
    except KeyError as exc:
        raise RuntimeError(
            f"architecture descriptor spec builder 未註冊: {descriptor.spec_builder_key!r}"
        ) from exc
    return builder(normalized)
