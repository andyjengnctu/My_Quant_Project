from __future__ import annotations

import ast
import gc
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class _FileSignature:
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class SourceDefinition:
    name: str
    kind: str
    lineno: int
    end_lineno: int
    decorated: bool


@dataclass(frozen=True)
class _SourceStructure:
    imports: tuple
    function_names: tuple[str, ...]
    exception_handlers: tuple
    bindings: dict
    definitions: tuple[SourceDefinition, ...]
    branch_literals: tuple | None


_TEXT_CACHE: Dict[Path, Tuple[_FileSignature, str]] = {}
_AST_CACHE: Dict[Path, Tuple[_FileSignature, ast.AST]] = {}
_STRUCTURE_CACHE: Dict[Path, Tuple[_FileSignature, _SourceStructure]] = {}
_STATEMENT_CHILD_FIELDS: dict[type, tuple[str, ...]] = {}


def _normalize_path(path: Path | str) -> Path:
    return Path(path).resolve()


def _signature(path: Path) -> _FileSignature:
    stat = path.stat()
    return _FileSignature(mtime_ns=int(stat.st_mtime_ns), size=int(stat.st_size))


def _source_text_at_signature(normalized: Path, signature: _FileSignature, *, encoding: str) -> str:
    """Internal fast path: the caller already resolved and stat-ed the file."""
    cached = _TEXT_CACHE.get(normalized)
    if cached is not None and cached[0] == signature:
        return cached[1]

    text = normalized.read_text(encoding=encoding)
    _TEXT_CACHE[normalized] = (signature, text)
    _AST_CACHE.pop(normalized, None)
    _STRUCTURE_CACHE.pop(normalized, None)
    return text


def read_source_text(path: Path | str, *, encoding: str = "utf-8") -> str:
    """Read source text once per unchanged file within the current process."""
    normalized = _normalize_path(path)
    return _source_text_at_signature(normalized, _signature(normalized), encoding=encoding)


@contextmanager
def _defer_source_cycle_collection():
    # A freshly parsed AST has only child references, not cycles. Its many
    # short-lived nodes need reference counting, but not repeated cyclic-GC
    # scans of every older cached tree during construction. Defer only that
    # bounded source inspection; restore the caller's GC state on every exit.
    collect_cycles = gc.isenabled()
    if collect_cycles:
        gc.disable()
    try:
        yield
    finally:
        if collect_cycles:
            gc.enable()


@_defer_source_cycle_collection()
def _source_ast_at_signature(normalized: Path, signature: _FileSignature, *, encoding: str, retain: bool = True) -> ast.AST:
    cached = _AST_CACHE.get(normalized)
    if cached is not None and cached[0] == signature:
        return cached[1]

    text = _source_text_at_signature(normalized, signature, encoding=encoding)
    tree = ast.parse(text, filename=str(normalized))
    if retain:
        _AST_CACHE[normalized] = (signature, tree)
    return tree


def read_source_ast(path: Path | str, *, encoding: str = "utf-8", retain: bool = True) -> ast.AST:
    """Return a cached immutable-by-contract AST for an unchanged Python file."""
    normalized = _normalize_path(path)
    return _source_ast_at_signature(normalized, _signature(normalized), encoding=encoding, retain=retain)


def _iter_statement_nodes(tree):
    queue = deque([tree])
    while queue:
        node = queue.popleft()
        yield node
        node_type = type(node)
        fields = _STATEMENT_CHILD_FIELDS.get(node_type)
        if fields is None:
            fields = tuple(name for name in node._fields
                           if name in {"body", "orelse", "finalbody", "handlers", "cases"})
            _STATEMENT_CHILD_FIELDS[node_type] = fields
        for name in fields:
            queue.extend(getattr(node, name))


def _iter_structure_nodes(tree):
    """Full expression reachability, excluding leaves that cannot hold a branch."""
    queue = deque([tree])
    leaves = (ast.Name, ast.Constant, ast.expr_context, ast.operator, ast.unaryop,
              ast.boolop, ast.cmpop, ast.alias)
    while queue:
        node = queue.popleft()
        yield node
        queue.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, leaves))


@_defer_source_cycle_collection()
def _populate_source_structure(path: Path | str, *, encoding: str = "utf-8", include_branches: bool = False) -> _SourceStructure:
    normalized = _normalize_path(path)
    signature = _signature(normalized)
    cached = _STRUCTURE_CACHE.get(normalized)
    if cached is not None and cached[0] == signature:
        if not include_branches or cached[1].branch_literals is not None:
            return cached[1]

    # Structural scans need imports, handlers and declarations, not all of the
    # function bodies. Retaining every project AST makes later allocations pay
    # for repeated GC scans of millions of otherwise-unused syntax nodes.
    tree = _source_ast_at_signature(normalized, signature, encoding=encoding, retain=False)
    import_nodes = []
    exception_handlers = []
    branch_literals = []

    # Imports and exception handlers can only occur in statement-bearing AST
    # containers.  ast.walk() also traverses every expression, constant,
    # argument and operator node; that work dominates project-wide source
    # contracts under coverage.  Keep the same breadth-first ordering as
    # ast.walk(), but do not enqueue branches that cannot contain statements.
    # An expression-level inspection requests the full walk once, while also
    # populating the import/declaration facts used by the remaining consumers.
    nodes = _iter_structure_nodes(tree) if include_branches else _iter_statement_nodes(tree)
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
        if isinstance(node, ast.ExceptHandler):
            exception_handlers.append(node)
        if include_branches:
            if isinstance(node, ast.Match):
                tests = [case.pattern for case in node.cases]
            elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
                tests = [node.test]
            else:
                continue
            literals = tuple(child.value for test in tests for child in ast.walk(test)
                             if isinstance(child, ast.Constant) and isinstance(child.value, str))
            branch_literals.append((int(node.lineno), literals))

    function_names = tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    exports = set()
    functions = []
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
            definitions.append(SourceDefinition(node.name, type(node).__name__, node.lineno,
                                                node.end_lineno, bool(node.decorator_list)))
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        elif isinstance(node, ast.Assign):
            exports.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.ImportFrom):
            exports.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            exports.update(alias.asname or alias.name.split(".")[-1] for alias in node.names)
    structure = _SourceStructure(
        tuple(import_nodes), function_names, tuple(exception_handlers),
        {"exports": frozenset(exports), "functions": tuple(functions)},
        tuple(definitions), tuple(branch_literals) if include_branches else None,
    )
    _STRUCTURE_CACHE[normalized] = (signature, structure)
    return structure


def read_source_import_nodes(path: Path | str, *, encoding: str = "utf-8") -> Tuple[ast.Import | ast.ImportFrom, ...]:
    """Return cached static import nodes for an unchanged Python file."""

    return _populate_source_structure(path, encoding=encoding).imports


def imports_symbol_from(path: Path | str, module: str, symbol: str) -> bool:
    """Check ownership independent of whitespace, parentheses, and aliases."""
    return any(
        isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == module
        and any(alias.name == symbol for alias in node.names)
        for node in read_source_import_nodes(path)
    )


def contains_source_expression(path: Path | str, expression: str) -> bool:
    """Match an expression's syntax tree, excluding comments and formatting."""
    expected_node = ast.parse(expression, mode="eval").body
    expected = ast.dump(expected_node, include_attributes=False)
    return any(isinstance(node, type(expected_node)) and ast.dump(node, include_attributes=False) == expected
               for node in ast.walk(read_source_ast(path)))


def read_source_top_level_function_names(path: Path | str, *, encoding: str = "utf-8") -> Tuple[str, ...]:
    """Return cached top-level function names for an unchanged Python file."""

    return _populate_source_structure(path, encoding=encoding).function_names


def read_source_exception_handlers(path: Path | str, *, encoding: str = "utf-8") -> Tuple[ast.ExceptHandler, ...]:
    """Return cached exception-handler nodes for an unchanged Python file."""

    return _populate_source_structure(path, encoding=encoding).exception_handlers


def read_source_top_level_bindings(path: Path | str, *, encoding: str = "utf-8") -> dict:
    """Small immutable values; no retained function body or cached PASS result."""
    return dict(_populate_source_structure(path, encoding=encoding).bindings)


def read_source_definitions(path: Path | str, *, encoding: str = "utf-8") -> Tuple[SourceDefinition, ...]:
    """Top-level declaration locations, without retaining their function bodies."""
    return _populate_source_structure(path, encoding=encoding).definitions


def read_source_branch_literals(path: Path | str, *, encoding: str = "utf-8") -> tuple:
    """Branch locations/literal strings, including nested ternaries and match patterns."""
    return _populate_source_structure(path, encoding=encoding, include_branches=True).branch_literals


def clear_source_index_cache() -> None:
    """Clear process-local source caches; intended for isolated validator fixtures."""

    _TEXT_CACHE.clear()
    _AST_CACHE.clear()
    _STRUCTURE_CACHE.clear()
