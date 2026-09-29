"""Lightweight runtime binding for a resolved scientific contract.

This module deliberately has no dependency on Optimizer/services code.  A
producer may bind one already-resolved Strategy Parameter scientific payload for
the duration of a computation so deep core helpers can read the same immutable
semantics without re-reading mutable config.  Outside the binding, existing
config-backed behavior is unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Mapping

_ACTIVE_SCIENTIFIC_CONTRACT: ContextVar[dict[str, Any] | None] = ContextVar(
    "active_strategy_param_scientific_contract",
    default=None,
)


def normalize_scientific_contract(payload) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    scientific = payload.get("scientific_contract")
    if isinstance(scientific, Mapping):
        return dict(scientific)
    if "schema_version" in payload and "strategy" in payload:
        return dict(payload)
    return None




def get_scientific_section(payload, section_name: str) -> dict[str, Any] | None:
    normalized = normalize_scientific_contract(payload)
    if not isinstance(normalized, dict):
        return None
    section = normalized.get(str(section_name))
    return dict(section) if isinstance(section, Mapping) else None

def get_active_scientific_contract() -> dict[str, Any] | None:
    payload = _ACTIVE_SCIENTIFIC_CONTRACT.get()
    return dict(payload) if isinstance(payload, dict) else None


def get_active_scientific_section(section_name: str) -> dict[str, Any] | None:
    payload = _ACTIVE_SCIENTIFIC_CONTRACT.get()
    if not isinstance(payload, dict):
        return None
    section = payload.get(str(section_name))
    return dict(section) if isinstance(section, Mapping) else None


@contextmanager
def scientific_runtime_context(payload):
    normalized = normalize_scientific_contract(payload)
    if normalized is None:
        yield
        return
    token = _ACTIVE_SCIENTIFIC_CONTRACT.set(normalized)
    try:
        yield
    finally:
        _ACTIVE_SCIENTIFIC_CONTRACT.reset(token)


__all__ = [
    "get_active_scientific_contract",
    "get_active_scientific_section",
    "get_scientific_section",
    "normalize_scientific_contract",
    "scientific_runtime_context",
]
