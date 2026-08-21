"""Cross-platform path parsing primitives shared by core and validation tooling."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def contains_any_path_separator(value: str) -> bool:
    return ("/" in value) or ("\\" in value)


def is_windows_absolute_path(raw_value: str) -> bool:
    return bool(_WINDOWS_ABSOLUTE_PATH_RE.match(raw_value)) or raw_value.startswith("\\\\")



def project_relative_display_path(
    path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> str:
    """Render project-scoped paths from repository root using forward slashes."""

    raw = os.fspath(path)
    candidate = Path(raw)
    root = Path(project_root)
    try:
        relative = candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        if candidate.is_absolute():
            return raw.replace("\\", "/")
        relative = candidate
    text = relative.as_posix()
    return text if text not in {"", "."} else "."

def split_cross_platform_parts(raw_value: str) -> tuple[str, tuple[str, ...]]:
    normalized = raw_value.replace("\\", "/")
    return normalized, PurePosixPath(normalized).parts


__all__ = [
    "contains_any_path_separator",
    "is_windows_absolute_path",
    "project_relative_display_path",
    "split_cross_platform_parts",
]
