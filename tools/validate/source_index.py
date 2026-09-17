from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class _FileSignature:
    mtime_ns: int
    size: int


_TEXT_CACHE: Dict[Path, Tuple[_FileSignature, str]] = {}
_AST_CACHE: Dict[Path, Tuple[_FileSignature, ast.AST]] = {}
_IMPORT_NODE_CACHE: Dict[Path, Tuple[_FileSignature, Tuple[ast.Import | ast.ImportFrom, ...]]] = {}
_TOP_LEVEL_FUNCTION_CACHE: Dict[Path, Tuple[_FileSignature, Tuple[str, ...]]] = {}
_EXCEPTION_HANDLER_CACHE: Dict[Path, Tuple[_FileSignature, Tuple[ast.ExceptHandler, ...]]] = {}


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
    _IMPORT_NODE_CACHE.pop(normalized, None)
    _TOP_LEVEL_FUNCTION_CACHE.pop(normalized, None)
    _EXCEPTION_HANDLER_CACHE.pop(normalized, None)
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


def _populate_source_structure(path: Path | str, *, encoding: str = "utf-8") -> tuple[
    _FileSignature,
    Tuple[ast.Import | ast.ImportFrom, ...],
    Tuple[str, ...],
    Tuple[ast.ExceptHandler, ...],
]:
    normalized = _normalize_path(path)
    signature = _signature(normalized)
    cached_imports = _IMPORT_NODE_CACHE.get(normalized)
    cached_functions = _TOP_LEVEL_FUNCTION_CACHE.get(normalized)
    cached_exception_handlers = _EXCEPTION_HANDLER_CACHE.get(normalized)
    if (
        cached_imports is not None
        and cached_imports[0] == signature
        and cached_functions is not None
        and cached_functions[0] == signature
        and cached_exception_handlers is not None
        and cached_exception_handlers[0] == signature
    ):
        return (
            signature,
            cached_imports[1],
            cached_functions[1],
            cached_exception_handlers[1],
        )

    tree = read_source_ast(normalized, encoding=encoding)
    import_nodes = []
    exception_handlers = []

    # Imports and exception handlers can only occur in statement-bearing AST
    # containers.  ast.walk() also traverses every expression, constant,
    # argument and operator node; that work dominates project-wide source
    # contracts under coverage.  Keep the same breadth-first ordering as
    # ast.walk(), but do not enqueue branches that cannot contain statements.
    queue = deque([tree])
    statement_container_types = (ast.stmt, ast.ExceptHandler)
    match_case_type = getattr(ast, "match_case", None)
    while queue:
        node = queue.popleft()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
        if isinstance(node, ast.ExceptHandler):
            exception_handlers.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, statement_container_types) or (
                match_case_type is not None and isinstance(child, match_case_type)
            ):
                queue.append(child)

    function_names = tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    import_nodes_tuple = tuple(import_nodes)
    exception_handlers_tuple = tuple(exception_handlers)
    _IMPORT_NODE_CACHE[normalized] = (signature, import_nodes_tuple)
    _TOP_LEVEL_FUNCTION_CACHE[normalized] = (signature, function_names)
    _EXCEPTION_HANDLER_CACHE[normalized] = (signature, exception_handlers_tuple)
    return signature, import_nodes_tuple, function_names, exception_handlers_tuple


def read_source_import_nodes(path: Path | str, *, encoding: str = "utf-8") -> Tuple[ast.Import | ast.ImportFrom, ...]:
    """Return cached static import nodes for an unchanged Python file."""

    _signature_value, import_nodes, _function_names, _exception_handlers = _populate_source_structure(
        path,
        encoding=encoding,
    )
    return import_nodes


def read_source_top_level_function_names(path: Path | str, *, encoding: str = "utf-8") -> Tuple[str, ...]:
    """Return cached top-level function names for an unchanged Python file."""

    _signature_value, _import_nodes, function_names, _exception_handlers = _populate_source_structure(
        path,
        encoding=encoding,
    )
    return function_names


def read_source_exception_handlers(path: Path | str, *, encoding: str = "utf-8") -> Tuple[ast.ExceptHandler, ...]:
    """Return cached exception-handler nodes for an unchanged Python file."""

    _signature_value, _import_nodes, _function_names, exception_handlers = _populate_source_structure(
        path,
        encoding=encoding,
    )
    return exception_handlers


def clear_source_index_cache() -> None:
    """Clear process-local source caches; intended for isolated validator fixtures."""

    _TEXT_CACHE.clear()
    _AST_CACHE.clear()
    _IMPORT_NODE_CACHE.clear()
    _TOP_LEVEL_FUNCTION_CACHE.clear()
    _EXCEPTION_HANDLER_CACHE.clear()
