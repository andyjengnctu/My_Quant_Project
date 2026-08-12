"""Small serialization primitives shared by strategy and audit output adapters."""

from __future__ import annotations

import math
from typing import Any


def clean_optional_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def json_native_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return json_native_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_native_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native_value(item) for item in value]
    return str(value)


__all__ = ["clean_optional_text", "json_native_value"]
