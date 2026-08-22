"""Advisory maintenance scan for disposable research/test code.

This is intentionally not a second formal gate.  It identifies high-signal
slimming candidates so completed one-off research does not silently become a
permanent maintenance burden.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections import Counter
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
GROWTH_PYTHON_DELTA_REVIEW = 1500
GROWTH_VALIDATE_DELTA_REVIEW = 800
SOURCE_SHAPE_DELTA_REVIEW = 250
SOURCE_SHAPE_NEW_VALIDATOR_REVIEW = 120
EXPERIMENT_VALIDATOR_PATTERN = re.compile(r"(?:^|_)(?:mr\d+[a-z]?|sr(?:_|\d)|c\d+)(?:_|$)", re.IGNORECASE)
EXPERIMENT_ID_PATTERN = re.compile(r"\b(?:MR|SR)-\d+[A-Z]?\b|\bC\d+\b")
SOURCE_SHAPE_TOKENS = (
    "read_text(",
    "read_source_text(",
    "read_source_ast(",
    "ast.parse(",
    "inspect.getsource(",
)


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



def _iter_python_paths(project_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root_name in ("apps", "config", "core", "filters", "services", "strategies", "tools"):
        root = project_root / root_name
        if root.is_dir():
            paths.extend(sorted(root.rglob("*.py")))
    return paths


def _private_zero_caller_candidates(project_root: Path) -> list[dict[str, Any]]:
    """Return conservative top-level private definitions with no lexical caller.

    This is advisory only.  It intentionally ignores decorated definitions and any name
    that appears anywhere else in the Python tree, including tests and compatibility code.
    """

    paths = _iter_python_paths(project_root)
    token_re = re.compile(r"\b[A-Za-z_]\w*\b")
    counts: Counter[str] = Counter()
    sources: dict[Path, str] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        sources[path] = text
        counts.update(token_re.findall(text))

    candidates: list[dict[str, Any]] = []
    for path, text in sources.items():
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if not node.name.startswith("_") or node.name.startswith("__"):
                continue
            if getattr(node, "decorator_list", None) or counts[node.name] != 1:
                continue
            candidates.append({
                "path": path.relative_to(project_root).as_posix(),
                "name": node.name,
                "line_count": int(node.end_lineno - node.lineno + 1),
                "reason": "top_level_private_definition_has_no_other_lexical_reference",
            })
    return candidates


def _literal_strings(node: ast.AST) -> list[str]:
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.append(child.value)
    return values


def _experiment_execution_branch_candidates(project_root: Path) -> list[dict[str, Any]]:
    """Find MR/SR/Cxx identity literals that directly control executable branches."""

    candidates: list[dict[str, Any]] = []
    for root_name in ("core", "filters", "services"):
        root = project_root / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                condition: ast.AST | None = None
                if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    condition = node.test
                elif isinstance(node, ast.Match):
                    condition = node.subject
                    match_literals = []
                    for case in node.cases:
                        match_literals.extend(_literal_strings(case.pattern))
                    literals = match_literals
                    hits = sorted({hit for value in literals for hit in EXPERIMENT_ID_PATTERN.findall(value)})
                    if hits:
                        candidates.append({
                            "path": path.relative_to(project_root).as_posix(),
                            "line": int(node.lineno),
                            "identities": hits,
                            "reason": "experiment_identity_literal_controls_execution_branch",
                        })
                    continue
                if condition is None:
                    continue
                hits = sorted({hit for value in _literal_strings(condition) for hit in EXPERIMENT_ID_PATTERN.findall(value)})
                if hits:
                    candidates.append({
                        "path": path.relative_to(project_root).as_posix(),
                        "line": int(node.lineno),
                        "identities": hits,
                        "reason": "experiment_identity_literal_controls_execution_branch",
                    })
    return candidates


def _source_shape_validator_inventory(project_root: Path) -> dict[str, Any]:
    functions: dict[str, dict[str, Any]] = {}
    total_loc = 0
    validate_root = project_root / "tools" / "validate"
    for path in sorted(validate_root.glob("synthetic*_cases.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("validate_"):
                continue
            source = "\n".join(lines[node.lineno - 1:node.end_lineno])
            if not any(token in source for token in SOURCE_SHAPE_TOKENS):
                continue
            line_count = int(node.end_lineno - node.lineno + 1)
            key = f"{path.relative_to(project_root).as_posix()}::{node.name}"
            functions[key] = {
                "path": path.relative_to(project_root).as_posix(),
                "validator": node.name,
                "line_count": line_count,
            }
            total_loc += line_count
    return {"line_count": total_loc, "validator_count": len(functions), "functions": functions}


def _git_text(project_root: Path, revision: str, rel_path: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "show", f"{revision}:{rel_path}"],
            cwd=project_root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _git_parent_available(project_root: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^"],
            cwd=project_root, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _git_growth_summary(project_root: Path, current_source_shape: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "available": False, "python_added": 0, "python_deleted": 0, "python_delta": 0,
        "validate_added": 0, "validate_deleted": 0, "validate_delta": 0,
        "new_python_files": 0, "source_shape_delta": 0, "new_experiment_validators": [],
        "new_large_source_shape_validators": [], "review_reasons": [],
    }
    if not _git_parent_available(project_root):
        return summary
    try:
        proc = subprocess.run(
            ["git", "diff", "--numstat", "HEAD^", "HEAD", "--", "*.py"],
            cwd=project_root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return summary
    if proc.returncode != 0:
        return summary
    summary["available"] = True
    changed_paths: list[str] = []
    for raw in proc.stdout.splitlines():
        parts = raw.split("\t")
        if len(parts) != 3:
            continue
        add_text, del_text, rel_path = parts
        if not add_text.isdigit() or not del_text.isdigit():
            continue
        added, deleted = int(add_text), int(del_text)
        summary["python_added"] += added
        summary["python_deleted"] += deleted
        if rel_path.startswith("tools/validate/"):
            summary["validate_added"] += added
            summary["validate_deleted"] += deleted
        if deleted == 0 and added > 0 and _git_text(project_root, "HEAD^", rel_path) is None:
            summary["new_python_files"] += 1
        changed_paths.append(rel_path)
    summary["python_delta"] = summary["python_added"] - summary["python_deleted"]
    summary["validate_delta"] = summary["validate_added"] - summary["validate_deleted"]

    previous_shape_loc = 0
    previous_validator_names: set[str] = set()
    for rel_path in sorted(set(changed_paths)):
        if not rel_path.startswith("tools/validate/") or not rel_path.endswith("_cases.py"):
            continue
        old = _git_text(project_root, "HEAD^", rel_path)
        if old is None:
            continue
        try:
            tree = ast.parse(old)
        except SyntaxError:
            continue
        lines = old.splitlines()
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("validate_"):
                continue
            previous_validator_names.add(node.name)
            source = "\n".join(lines[node.lineno - 1:node.end_lineno])
            if any(token in source for token in SOURCE_SHAPE_TOKENS):
                previous_shape_loc += int(node.end_lineno - node.lineno + 1)

    current_changed_shape_loc = 0
    for rel_path in sorted(set(changed_paths)):
        if not rel_path.startswith("tools/validate/") or not rel_path.endswith("_cases.py"):
            continue
        path = project_root / rel_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("validate_"):
                continue
            source = "\n".join(lines[node.lineno - 1:node.end_lineno])
            is_source_shape = any(token in source for token in SOURCE_SHAPE_TOKENS)
            if is_source_shape:
                line_count = int(node.end_lineno - node.lineno + 1)
                current_changed_shape_loc += line_count
                if node.name not in previous_validator_names and line_count >= SOURCE_SHAPE_NEW_VALIDATOR_REVIEW:
                    summary["new_large_source_shape_validators"].append({
                        "path": rel_path, "validator": node.name, "line_count": line_count,
                    })
            if node.name not in previous_validator_names and EXPERIMENT_VALIDATOR_PATTERN.search(node.name):
                summary["new_experiment_validators"].append({"path": rel_path, "validator": node.name})
    summary["source_shape_delta"] = current_changed_shape_loc - previous_shape_loc

    if summary["python_delta"] > GROWTH_PYTHON_DELTA_REVIEW:
        summary["review_reasons"].append("python_loc_delta_above_review_threshold")
    if summary["validate_delta"] > GROWTH_VALIDATE_DELTA_REVIEW:
        summary["review_reasons"].append("validation_loc_delta_above_review_threshold")
    if summary["source_shape_delta"] > SOURCE_SHAPE_DELTA_REVIEW:
        summary["review_reasons"].append("source_shape_validator_growth_above_review_threshold")
    if summary["new_experiment_validators"]:
        summary["review_reasons"].append("new_experiment_specific_validator")
    if summary["new_large_source_shape_validators"]:
        summary["review_reasons"].append("new_large_source_shape_validator")
    return summary

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
    private_zero_callers = _private_zero_caller_candidates(root)
    experiment_execution_branches = _experiment_execution_branch_candidates(root)
    source_shape_inventory = _source_shape_validator_inventory(root)
    growth = _git_growth_summary(root, source_shape_inventory)
    growth_reviews = [
        {"reason": reason}
        for reason in growth.get("review_reasons", [])
    ]

    candidate_count = (
        len(stale_modules)
        + len(disabled_audits)
        + len(oversized_transient_tests)
        + len(retired_dedicated_tests)
        + len(private_zero_callers)
        + len(experiment_execution_branches)
        + len(growth_reviews)
    )
    return {
        "status": "REVIEW" if candidate_count else "CLEAN",
        "needs_slimming": bool(candidate_count),
        "candidate_count": candidate_count,
        "stale_modules": stale_modules,
        "disabled_audits": disabled_audits,
        "oversized_transient_tests": oversized_transient_tests,
        "retired_dedicated_tests": retired_dedicated_tests,
        "private_zero_callers": private_zero_callers,
        "experiment_execution_branches": experiment_execution_branches,
        "source_shape_inventory": {
            "line_count": source_shape_inventory["line_count"],
            "validator_count": source_shape_inventory["validator_count"],
        },
        "growth": growth,
        "growth_reviews": growth_reviews,
        "advisory_only": True,
        "policy": (
            "Remove only after the research decision is recorded and the module is no longer "
            "reachable from current runtime or an active formal Audit. Retired dedicated "
            "validators should be replaced by reusable generic contracts when their math remains useful. "
            "Growth, source-shape, experiment-specific validator and zero-caller findings are REVIEW-only; "
            "they are never deletion authority by themselves."
        ),
    }


__all__ = [
    "TRANSIENT_TEST_LINE_THRESHOLD",
    "summarize_transient_code_maintenance",
]
