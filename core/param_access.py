"""Small canonical helpers for reading parameter payloads or objects."""
from __future__ import annotations

from typing import Any


def get_param_value(params: Any, key: str, default=None):
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)


__all__ = ["get_param_value"]
