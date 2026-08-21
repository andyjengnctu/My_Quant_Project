"""Advisory maintenance scan for disposable research/test code.

This is intentionally not a second formal gate.  It identifies high-signal
slimming candidates so completed one-off research does not silently become a
permanent maintenance burden.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

TRANSIENT_MODULE_PREFIXES = (
    "tools.audit",
    "config.compatibility",
)
TEST_MODULE_PREFIXES = (
    "tools.validate",
    "tools.local_regression",
)
TRANSIENT_TEST_LINE_THRESHOLD = 2000


def _module_name(project_root: Path, path: Path) -> str:
    rel = path.relative_to(project_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_index(project_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root_name in ("apps", "config", "core", "filters", "services", "strategies", "tools"):
        root = project_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            module = _module_name(project_root, path)
            if module:
                index[module] = path
    return index


def _resolve_import_from(current_module: str, current_path: Path, module: str | None, level: int) -> str:
    if level <= 0:
        return str(module or "")
    package_parts = current_module.split(".") if current_path.name == "__init__.py" else current_module.split(".")[:-1]
    trim = max(level - 1, 0)
    if trim:
        package_parts = package_parts[:-trim] if trim <= len(package_parts) else []
    suffix = module.split(".") if module else []
    return ".".join([*package_parts, *suffix])


def _import_edges(project_root: Path, index: dict[str, Path]) -> dict[str, set[str]]:
    names = set(index)
    graph: dict[str, set[str]] = {name: set() for name in names}
    for module, path in index.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in names:
                        graph[module].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_import_from(module, path, node.module, node.level)
                for alias in node.names:
                    if alias.name == "*":
                        if base in names:
                            graph[module].add(base)
                        continue
                    full = f"{base}.{alias.name}" if base else alias.name
                    if full in names:
                        graph[module].add(full)
                    elif base in names:
                        graph[module].add(base)
    return graph


def _active_dynamic_audit_modules() -> set[str]:
    from config.audit import get_audit_module_ids, get_enabled_audit_definitions
    from tools.audit.catalog import AUDIT_CATALOG, get_audit_entry

    # Research-mode catalog commands are dynamically routed by the model application
    # even though they are not config-driven formal Audits.  Treat every exposed CLI
    # entry as current reachability, then add enabled formal handlers.
    modules = {
        entry.module
        for entry in AUDIT_CATALOG.values()
        if entry.cli_command
    }
    for module_id in get_audit_module_ids(enabled_only=True):
        for definition in get_enabled_audit_definitions(module_id):
            modules.add(get_audit_entry(definition.audit_type).module)
    return modules


def _reachable_from_production(graph: dict[str, set[str]], dynamic_roots: set[str]) -> set[str]:
    roots = {
        module
        for module in graph
        if not module.startswith(TRANSIENT_MODULE_PREFIXES + TEST_MODULE_PREFIXES)
    }
    roots.update(dynamic_roots)
    reachable: set[str] = set()
    queue: deque[str] = deque(sorted(roots))
    while queue:
        module = queue.popleft()
        if module in reachable:
            continue
        reachable.add(module)
        for child in graph.get(module, ()):
            if child not in reachable:
                queue.append(child)
    return reachable


def _catalog_modes_by_module() -> dict[str, list[str]]:
    from tools.audit.catalog import AUDIT_CATALOG

    modes: dict[str, list[str]] = defaultdict(list)
    for entry in AUDIT_CATALOG.values():
        modes[entry.module].append(entry.mode)
    return {key: sorted(set(values)) for key, values in modes.items()}


def _retired_dedicated_test_candidates(
    project_root: Path,
    index: dict[str, Path],
) -> list[dict[str, str]]:
    """Find retired checklist tests whose dedicated validator still exists.

    The checklist lifecycle is the source of retirement state.  A validator reused by
    any currently active T-row is not a candidate, so shared generic contracts are not
    penalized when one historical requirement is retired.
    """

    checklist = project_root / "doc" / "TEST_SUITE_CHECKLIST.md"
    if not checklist.is_file():
        return []
    text = checklist.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    active_validators: set[str] = set()
    in_current_tests = False
    for line in lines:
        if line.startswith("### T."):
            in_current_tests = True
            continue
        if in_current_tests and line.startswith("## G."):
            break
        if in_current_tests and line.startswith("| T"):
            active_validators.update(
                re.findall(r"`?(validate_[A-Za-z0-9_]+)`?", line)
            )

    latest_status: dict[str, str] = {}
    validators_by_test: dict[str, set[str]] = defaultdict(set)
    for line in lines:
        if not line.startswith("| 20"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[1].startswith("T"):
            continue
        test_id = cells[1]
        status = cells[3].split("->")[-1].strip()
        latest_status[test_id] = status
        validators_by_test[test_id].update(
            re.findall(r"`?(validate_[A-Za-z0-9_]+)`?", line)
        )

    function_paths: dict[str, str] = {}
    for module, path in index.items():
        if not module.startswith("tools.validate"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_paths[node.name] = path.relative_to(project_root).as_posix()

    candidates: list[dict[str, str]] = []
    for test_id, status in sorted(latest_status.items()):
        if status != "N/A":
            continue
        for validator in sorted(validators_by_test.get(test_id, ())):
            path = function_paths.get(validator)
            if path is None or validator in active_validators:
                continue
            candidates.append(
                {
                    "test_id": test_id,
                    "validator": validator,
                    "path": path,
                    "reason": "retired_checklist_test_has_unshared_validator_definition",
                }
            )
    return candidates


def summarize_transient_code_maintenance(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    index = _module_index(root)
    graph = _import_edges(root, index)
    dynamic_roots = _active_dynamic_audit_modules()
    reachable = _reachable_from_production(graph, dynamic_roots)
    catalog_modes = _catalog_modes_by_module()

    stale_modules: list[dict[str, Any]] = []
    for module, path in sorted(index.items()):
        if not module.startswith(TRANSIENT_MODULE_PREFIXES):
            continue
        if module in reachable:
            continue
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(root).as_posix()
        stale_modules.append(
            {
                "module": module,
                "path": rel,
                "line_count": len(path.read_text(encoding="utf-8", errors="ignore").splitlines()),
                "catalog_modes": catalog_modes.get(module, []),
                "reason": "not_reachable_from_current_runtime_or_active_formal_audit",
            }
        )

    from config.audit import AUDIT_MODULES

    disabled_audits: list[dict[str, str]] = []
    for module_id, raw_module in AUDIT_MODULES.items():
        raw_audits = raw_module.get("audits") if isinstance(raw_module, dict) else None
        if not isinstance(raw_audits, dict):
            continue
        for audit_id, raw_audit in raw_audits.items():
            if not isinstance(raw_audit, dict) or bool(raw_audit.get("enabled", False)):
                continue
            disabled_audits.append(
                {
                    "module_id": str(module_id),
                    "audit_id": str(audit_id),
                    "audit_type": str(raw_audit.get("audit_type") or ""),
                }
            )

    oversized_transient_tests: list[dict[str, Any]] = []
    validate_root = root / "tools" / "validate"
    if validate_root.is_dir():
        for path in sorted(validate_root.glob("synthetic_*audit*_cases.py")):
            line_count = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
            if line_count > TRANSIENT_TEST_LINE_THRESHOLD:
                oversized_transient_tests.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "line_count": line_count,
                        "threshold": TRANSIENT_TEST_LINE_THRESHOLD,
                    }
                )

    retired_dedicated_tests = _retired_dedicated_test_candidates(root, index)

    candidate_count = (
        len(stale_modules)
        + len(disabled_audits)
        + len(oversized_transient_tests)
        + len(retired_dedicated_tests)
    )
    return {
        "status": "REVIEW" if candidate_count else "CLEAN",
        "needs_slimming": bool(candidate_count),
        "candidate_count": candidate_count,
        "stale_modules": stale_modules,
        "disabled_audits": disabled_audits,
        "oversized_transient_tests": oversized_transient_tests,
        "retired_dedicated_tests": retired_dedicated_tests,
        "advisory_only": True,
        "policy": (
            "Remove only after the research decision is recorded and the module is no longer "
            "reachable from current runtime or an active formal Audit. Retired dedicated "
            "validators should be replaced by reusable generic contracts when their math remains useful."
        ),
    }


__all__ = [
    "TRANSIENT_TEST_LINE_THRESHOLD",
    "summarize_transient_code_maintenance",
]
