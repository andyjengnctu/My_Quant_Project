"""Importable callable references shared by optimizer process boundaries."""
from __future__ import annotations

import importlib


def resolve_importable_callable_path(func) -> str | None:
    """Return ``module:qualname`` only when *func* can be re-imported identically."""
    module_name = str(getattr(func, "__module__", "") or "").strip()
    qualname = str(getattr(func, "__qualname__", "") or "").strip()
    if not module_name or not qualname or "<locals>" in qualname:
        return None
    try:
        value = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError):
        return None
    if value is not func:
        return None
    return f"{module_name}:{qualname}"


def load_callable_from_import_path(path: str):
    """Resolve an importable callable reference created by the companion resolver."""
    module_name, separator, qualname = str(path or "").partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError(f"不合法的 callable import path: {path!r}")
    value = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"import path 不是 callable: {path}")
    return value


__all__ = ["resolve_importable_callable_path", "load_callable_from_import_path"]
