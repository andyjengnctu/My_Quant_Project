"""Cross-platform path parsing primitives shared by core and validation tooling."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def contains_any_path_separator(value: str) -> bool:
    return ("/" in value) or ("\\" in value)


def is_windows_absolute_path(raw_value: str) -> bool:
    return bool(_WINDOWS_ABSOLUTE_PATH_RE.match(raw_value)) or raw_value.startswith("\\\\")


def split_cross_platform_parts(raw_value: str) -> tuple[str, tuple[str, ...]]:
    normalized = raw_value.replace("\\", "/")
    return normalized, PurePosixPath(normalized).parts


__all__ = [
    "contains_any_path_separator",
    "is_windows_absolute_path",
    "split_cross_platform_parts",
]
