from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class _FileSignature:
    mtime_ns: int
    size: int


_TEXT_CACHE: Dict[Path, Tuple[_FileSignature, str]] = {}
_AST_CACHE: Dict[Path, Tuple[_FileSignature, ast.AST]] = {}


def _normalize_path(path: Path | str) -> Path:
    return Path(path).resolve()


def _signature(path: Path) -> _FileSignature:
    stat = path.stat()
    return _FileSignature(mtime_ns=int(stat.st_mtime_ns), size=int(stat.st_size))


def read_source_text(path: Path | str, *, encoding: str = "utf-8") -> str:
    """Read source text once per unchanged file within the current process."""

    normalized = _normalize_path(path)
    signature = _signature(normalized)
    cached = _TEXT_CACHE.get(normalized)
    if cached is not None and cached[0] == signature:
        return cached[1]

    text = normalized.read_text(encoding=encoding)
    _TEXT_CACHE[normalized] = (signature, text)
    _AST_CACHE.pop(normalized, None)
    return text


def read_source_ast(path: Path | str, *, encoding: str = "utf-8") -> ast.AST:
    """Return a cached immutable-by-contract AST for an unchanged Python file."""

    normalized = _normalize_path(path)
    signature = _signature(normalized)
    cached = _AST_CACHE.get(normalized)
    if cached is not None and cached[0] == signature:
        return cached[1]

    text = read_source_text(normalized, encoding=encoding)
    tree = ast.parse(text, filename=str(normalized))
    _AST_CACHE[normalized] = (signature, tree)
    return tree


def clear_source_index_cache() -> None:
    """Clear process-local source caches; intended for isolated validator fixtures."""

    _TEXT_CACHE.clear()
    _AST_CACHE.clear()
