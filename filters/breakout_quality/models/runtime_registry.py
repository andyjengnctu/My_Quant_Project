"""Runtime builder capabilities used by architecture descriptors.

The architecture registry owns architecture -> runtime_builder_key.  This module
owns only reusable builder-key implementations, so adding another architecture
that reuses an existing family does not add an architecture-ID branch here.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

RuntimeBuilder = Callable[..., Any]


def _build_tiny(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.tiny_cnn import build_tiny_cnn
    return build_tiny_cnn(nn, torch, feature_count=int(feature_count), context_count=int(context_count))


def _build_multiscale(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.multiscale_cnn import build_multiscale_cnn
    return build_multiscale_cnn(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)


def _build_inception(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.inception_time import build_inception_time
    return build_inception_time(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)


def _build_market_set(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.inception_time_market_set import build_inception_time_market_set
    return build_inception_time_market_set(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)


def _build_modern_tcn(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.modern_tcn import build_modern_tcn
    return build_modern_tcn(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)


def _build_modern_tcn_joint(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.large_kernel_tcn_joint_min import build_modern_tcn_joint_min
    return build_modern_tcn_joint_min(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)



def _build_hybrid_safety_patch_mfe_inception(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.hybrid_safety_patch_mfe import build_safety_patch_mfe_inception
    return build_safety_patch_mfe_inception(
        nn,
        torch,
        feature_count=int(feature_count),
        context_count=int(context_count),
        spec=spec,
    )

def _build_patch_transformer(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.patch_transformer import build_patch_transformer
    return build_patch_transformer(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)


def _build_patch_token_ranker(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.patch_token_joint_min import build_patch_token_ranker
    return build_patch_token_ranker(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)


def _build_patch_token_joint(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.patch_token_joint_min import build_patch_token_joint_min
    return build_patch_token_joint_min(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)



def _build_day_token_transformer_shared_safety_mfe(
    nn,
    torch,
    *,
    feature_count,
    context_count,
    spec,
    pretrained_encoder_state=None,
):
    from filters.breakout_quality.models.day_token_transformer import (
        build_day_token_transformer_shared_safety_mfe,
    )
    return build_day_token_transformer_shared_safety_mfe(
        nn,
        torch,
        feature_count=int(feature_count),
        context_count=int(context_count),
        spec=spec,
    )

def _build_residual_tcn(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.residual_tcn import build_residual_tcn
    return build_residual_tcn(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)


def _build_moment(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.moment import build_moment_frozen_linear
    return build_moment_frozen_linear(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec, pretrained_encoder_state=pretrained_encoder_state)


def _build_mantis(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.mantis_v2 import build_mantis_v2_frozen_linear
    return build_mantis_v2_frozen_linear(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec, pretrained_encoder_state=pretrained_encoder_state)


def _build_ts2vec(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.ts2vec import build_ts2vec_frozen_linear
    return build_ts2vec_frozen_linear(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec, pretrained_encoder_state=pretrained_encoder_state)


_RUNTIME_BUILDERS_BY_KEY: dict[str, RuntimeBuilder] = {
    "tiny_cnn": _build_tiny,
    "multiscale_cnn": _build_multiscale,
    "inception_time": _build_inception,
    "market_set": _build_market_set,
    "modern_tcn": _build_modern_tcn,
    "modern_tcn_joint": _build_modern_tcn_joint,
    "patch_transformer": _build_patch_transformer,
    "hybrid_safety_patch_mfe_inception": _build_hybrid_safety_patch_mfe_inception,
    "patch_token_ranker": _build_patch_token_ranker,
    "patch_token_joint": _build_patch_token_joint,
    "day_token_transformer_shared_safety_mfe": _build_day_token_transformer_shared_safety_mfe,
    "residual_tcn": _build_residual_tcn,
    "moment": _build_moment,
    "mantis": _build_mantis,
    "ts2vec": _build_ts2vec,
}


def build_registered_model(
    nn,
    torch,
    *,
    descriptor,
    feature_count: int,
    context_count: int,
    spec,
    pretrained_encoder_state: Mapping[str, object] | None = None,
):
    try:
        builder = _RUNTIME_BUILDERS_BY_KEY[descriptor.runtime_builder_key]
    except KeyError as exc:
        raise RuntimeError(
            f"architecture descriptor runtime builder 未註冊: {descriptor.runtime_builder_key!r}"
        ) from exc
    return builder(
        nn,
        torch,
        feature_count=int(feature_count),
        context_count=int(context_count),
        spec=spec,
        pretrained_encoder_state=pretrained_encoder_state,
    )


def registered_runtime_builder_keys() -> tuple[str, ...]:
    return tuple(_RUNTIME_BUILDERS_BY_KEY)
