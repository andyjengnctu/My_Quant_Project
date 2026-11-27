"""Runtime builder capabilities used by architecture descriptors.

The architecture registry owns architecture -> runtime_builder_key.  This module
owns only reusable builder-key implementations, so adding another architecture
that reuses an existing family does not add an architecture-ID branch here.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

RuntimeBuilder = Callable[..., Any]


def _build_inception(nn, torch, *, feature_count, context_count, spec, pretrained_encoder_state=None):
    from filters.breakout_quality.models.inception_time import build_inception_time
    return build_inception_time(nn, torch, feature_count=int(feature_count), context_count=int(context_count), spec=spec)




_RUNTIME_BUILDERS_BY_KEY: dict[str, RuntimeBuilder] = {
    "inception_time": _build_inception,
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
