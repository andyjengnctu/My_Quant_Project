"""Dependency-inverted Core -> breakout-quality filter runtime port.

The Filter adapter binds its two callables once when loaded by a service
composition root.  Pure Core modules never import filters/ or services/.
Filter-specific source and score semantics continue to live at their original
canonical runtime owner.  If a consumer requests the optional capability
without binding the adapter, fail closed instead of inventing pass-through data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BreakoutQualityRuntimePort:
    filter_pass_condition: Callable
    candidate_rank_resolver: Callable


_RUNTIME_PORT: BreakoutQualityRuntimePort | None = None


def bind_breakout_quality_runtime_port(*, filter_pass_condition: Callable, candidate_rank_resolver: Callable) -> None:
    global _RUNTIME_PORT
    if not callable(filter_pass_condition) or not callable(candidate_rank_resolver):
        raise TypeError("breakout-quality Core port 必須由正式Filter runtime提供callable")
    existing = _RUNTIME_PORT
    if existing is not None:
        if (existing.filter_pass_condition is not filter_pass_condition or
                existing.candidate_rank_resolver is not candidate_rank_resolver):
            raise RuntimeError("breakout-quality Core port 不得被第二個runtime覆蓋")
        return
    _RUNTIME_PORT = BreakoutQualityRuntimePort(
        filter_pass_condition=filter_pass_condition,
        candidate_rank_resolver=candidate_rank_resolver,
    )


def _require_runtime_port() -> BreakoutQualityRuntimePort:
    port = _RUNTIME_PORT
    if port is None:
        raise RuntimeError(
            "breakout-quality runtime 尚未由正式service綁定；不得略過PIT評分或降級為pass-through"
        )
    return port


def build_breakout_quality_filter_pass_condition(*args, **kwargs):
    """Compatibility patch seam; dispatch to the sole registered Filter producer."""
    return _require_runtime_port().filter_pass_condition(*args, **kwargs)


def resolve_breakout_quality_candidate_rank(*args, **kwargs):
    """Compatibility patch seam; dispatch to the sole registered Filter producer."""
    return _require_runtime_port().candidate_rank_resolver(*args, **kwargs)
