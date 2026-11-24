"""Descriptor-driven breakout-quality model-spec resolver."""

from __future__ import annotations

from filters.breakout_quality.models.architectures import (
    SUPPORTED_MODEL_ARCHITECTURES,
    get_architecture_descriptor,
    normalize_model_architecture,
)
from filters.breakout_quality.models.spec_builders import (
    ModelSpecBuilder,
    build_inception_variant_spec,
    build_ts2vec_spec,
)
from filters.breakout_quality.models.spec_contract import BreakoutQualityModelSpec


_SPEC_BUILDERS_BY_KEY: dict[str, ModelSpecBuilder] = {
    "ts2vec": build_ts2vec_spec,
    "inception_variant": build_inception_variant_spec,
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
