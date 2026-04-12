from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

CMD_SINGLE_ENTRY_TEXT = "正式對外入口為 `apps/test_suite.py`"
ARCHITECTURE_SINGLE_ENTRY_TEXT = "`apps/test_suite.py` 是日常唯一建議使用的一鍵測試入口"
LEGACY_APP_ENTRY_PATHS = ("apps/local_regression.py", "apps/validate_consistency.py")
LEGACY_DOC_GUIDANCE_FILES = ("doc/CMD.md", "doc/ARCHITECTURE.md")
SUSPICIOUS_APP_ENTRY_PATTERN = re.compile(r"(?:test|validate|regression|consistency)", re.IGNORECASE)

CRITICAL_HELPER_SINGLE_SOURCE_SPECS: Dict[str, Tuple[str, ...]] = {
    "core/price_utils.py": (
        "calc_entry_price",
        "calc_net_sell_price",
        "calc_position_size",
        "calc_initial_risk_total",
    ),
    "core/capital_policy.py": (
        "resolve_single_backtest_sizing_capital",
        "resolve_portfolio_sizing_equity",
        "resolve_portfolio_entry_budget",
        "resolve_scanner_live_capital",
    ),
    "core/exact_accounting.py": (
        "build_buy_ledger",
        "build_sell_ledger",
        "allocate_cost_basis_milli",
        "round_price_milli_to_tick",
    ),
}


def extract_markdown_table_rows(text: str, heading: str) -> List[List[str]]:
    lines = text.splitlines()
    heading_candidates = (f"### {heading}", f"## {heading}")
    heading_index = None
    for idx, raw_line in enumerate(lines):
        if raw_line.strip() in heading_candidates:
            heading_index = idx
            break
    if heading_index is None:
        raise AssertionError(f"找不到表格段落: {heading}")

    data_lines: List[List[str]] = []
    seen_table = False
    cursor = heading_index + 1
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if stripped.startswith(("## ", "### ")):
            break
        if not stripped.startswith("|"):
            cursor += 1
            continue

        block_lines = []
        while cursor < len(lines) and lines[cursor].strip().startswith("|"):
            block_lines.append(lines[cursor].strip())
            cursor += 1
        if block_lines:
            seen_table = True
            has_header = (
                len(block_lines) >= 2
                and set(block_lines[1].replace("|", "").replace(" ", "")) <= {"-"}
            )
            content_lines = block_lines[2:] if has_header else block_lines
            for line in content_lines:
                cols = [part.strip() for part in line.strip("|").split("|")]
                data_lines.append(cols)

    if not seen_table:
        raise AssertionError(f"找不到表格內容: {heading}")
    return data_lines


def _read_python_ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _load_named_string_dict_keys(module_path: Path, constant_name: str) -> List[str]:
    tree = _read_python_ast(module_path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if constant_name not in target_names or not isinstance(node.value, ast.Dict):
            continue
        parsed_keys: List[str] = []
        for key_node in node.value.keys:
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                parsed_keys.append(key_node.value)
        return parsed_keys
    return []


def summarize_no_reverse_app_import_contract(project_root: Path) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []
    for rel_dir in ("core", "tools"):
        for path in sorted((project_root / rel_dir).rglob("*.py")):
            tree = _read_python_ast(path)
            for node in ast.walk(tree):
                modules: List[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module_name in modules:
                    if module_name == "apps" or module_name.startswith("apps."):
                        violations.append({
                            "path": str(path.relative_to(project_root)).replace("\\", "/"),
                            "lineno": getattr(node, "lineno", 0),
                            "module": module_name,
                        })
    return {"violations": violations}


def _module_name_from_path(project_root: Path, path: Path) -> str:
    rel_path = path.relative_to(project_root).with_suffix("")
    parts = list(rel_path.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _load_project_module_index(project_root: Path) -> Dict[str, Path]:
    module_index: Dict[str, Path] = {}
    for rel_dir in ("apps", "core", "tools"):
        for path in sorted((project_root / rel_dir).rglob("*.py")):
            module_name = _module_name_from_path(project_root, path)
            if module_name:
                module_index[module_name] = path
    return module_index


def _resolve_import_from_module(current_module: str, current_path: Path, module: str | None, level: int) -> str:
    if current_path.name == "__init__.py":
        package_parts = current_module.split(".") if current_module else []
    else:
        package_parts = current_module.split(".")[:-1] if current_module else []

    if level > 0:
        trim_count = max(level - 1, 0)
        if trim_count:
            package_parts = package_parts[:-trim_count] if trim_count <= len(package_parts) else []

    module_parts = module.split(".") if module else []
    resolved_parts = [part for part in [*package_parts, *module_parts] if part]
    return ".".join(resolved_parts)


def _iter_project_import_edges(
    *,
    current_module: str,
    current_path: Path,
    node: ast.AST,
    project_module_names: Set[str],
) -> List[str]:
    targets: List[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            module_name = alias.name
            if module_name in project_module_names:
                targets.append(module_name)
    elif isinstance(node, ast.ImportFrom):
        base_module = _resolve_import_from_module(current_module, current_path, node.module, node.level)
        for alias in node.names:
            if alias.name == "*":
                if base_module in project_module_names:
                    targets.append(base_module)
                continue
            full_module = f"{base_module}.{alias.name}" if base_module else alias.name
            if full_module in project_module_names:
                targets.append(full_module)
            elif base_module in project_module_names:
                targets.append(base_module)
    return targets


def _compute_strongly_connected_components(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    index = 0
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[Set[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in sorted(graph.get(node, set())):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: Set[str] = set()
            while stack:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            components.append(component)

    for node in sorted(graph):
        if node not in indices:
            strongconnect(node)
    return components


def summarize_no_top_level_import_cycles_contract(project_root: Path) -> Dict[str, Any]:
    module_index = _load_project_module_index(project_root)
    project_module_names = set(module_index)
    graph: Dict[str, Set[str]] = {module_name: set() for module_name in module_index}

    for module_name, path in module_index.items():
        tree = _read_python_ast(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for imported_module in _iter_project_import_edges(
                current_module=module_name,
                current_path=path,
                node=node,
                project_module_names=project_module_names,
            ):
                graph[module_name].add(imported_module)

    violations: List[Dict[str, Any]] = []
    for component in _compute_strongly_connected_components(graph):
        if len(component) == 1:
            only_module = next(iter(component))
            if only_module not in graph.get(only_module, set()):
                continue
        cycle_modules = sorted(component)
        violations.append({
            "modules": cycle_modules,
            "paths": [str(module_index[name].relative_to(project_root)).replace("\\", "/") for name in cycle_modules],
            "size": len(cycle_modules),
        })

    return {
        "module_count": len(module_index),
        "violations": violations,
    }


def load_imported_validate_names_from_synthetic_main_entry(project_root: Path) -> Set[str]:
    source_path = project_root / "tools" / "validate" / "synthetic_cases.py"
    tree = _read_python_ast(source_path)
    imported_names: Set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.module or not node.module.startswith("synthetic"):
            continue
        for alias in node.names:
            imported_name = alias.asname or alias.name
            if imported_name.startswith("validate_"):
                imported_names.add(imported_name)
    return imported_names


def summarize_synthetic_cases_import_target_resolution_contract(project_root: Path) -> Dict[str, Any]:
    source_path = project_root / "tools" / "validate" / "synthetic_cases.py"
    tree = _read_python_ast(source_path)
    validate_dir = project_root / "tools" / "validate"

    invalid_imports: List[str] = []
    checked_imports: List[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 1 or not (node.module or "").startswith("synthetic_"):
            continue
        target_path = validate_dir / f"{node.module}.py"
        checked_imports.append(node.module or "")
        if not target_path.exists():
            invalid_imports.append(f"{node.module}:missing_module")
            continue
        target_tree = _read_python_ast(target_path)
        exported: Set[str] = set()
        for target_node in target_tree.body:
            if isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exported.add(target_node.name)
            elif isinstance(target_node, ast.Assign):
                for target in target_node.targets:
                    if isinstance(target, ast.Name):
                        exported.add(target.id)
            elif isinstance(target_node, ast.ImportFrom):
                for alias in target_node.names:
                    exported.add(alias.asname or alias.name)
            elif isinstance(target_node, ast.Import):
                for alias in target_node.names:
                    exported.add(alias.asname or alias.name.split(".")[-1])
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name not in exported:
                invalid_imports.append(f"{node.module}:{alias.name}")

    return {
        "source_path": str(source_path.relative_to(project_root)).replace("\\", "/"),
        "checked_module_imports": checked_imports,
        "invalid_imports": invalid_imports,
    }


def _extract_constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_constant_string_sequence(node: ast.AST | None) -> list[str]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return []
    values: list[str] = []
    for element in node.elts:
        value = _extract_constant_string(element)
        if value is None:
            return []
        values.append(value)
    return values


def load_synthetic_registry_entries_from_source(project_root: Path) -> list[Dict[str, Any]]:
    source_path = project_root / "tools" / "validate" / "synthetic_cases.py"
    tree = _read_python_ast(source_path)

    registry_function = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_synthetic_validator_entries":
            registry_function = node
            break
    if registry_function is None:
        return []

    entries: list[Dict[str, Any]] = []
    for node in ast.walk(registry_function):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_entry":
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        keyword_map = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        entries.append({
            "name": node.args[0].id,
            "layer": _extract_constant_string(keyword_map.get("layer")),
            "cost_class": _extract_constant_string(keyword_map.get("cost_class")),
            "impacted_modules": tuple(_extract_constant_string_sequence(keyword_map.get("impacted_modules"))),
        })
    return entries


def load_defined_validate_names_from_synthetic_case_modules(project_root: Path) -> Set[str]:
    validate_dir = project_root / "tools" / "validate"
    defined_names: Set[str] = set()
    for path in sorted(validate_dir.glob("synthetic*_cases.py")):
        tree = _read_python_ast(path)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("validate_"):
                defined_names.add(node.name)
    return defined_names


def summarize_test_suite_registry_contract(project_root: Path, formal_step_order: List[str] | tuple[str, ...]) -> Dict[str, Any]:
    source_path = project_root / "apps" / "test_suite.py"
    tree = _read_python_ast(source_path)

    imported_name_sources: Dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        for alias in node.names:
            imported_name_sources[alias.asname or alias.name] = (node.module, alias.name)

    regression_step_order_aliases_registry = False
    step_labels_keys: List[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "REGRESSION_STEP_ORDER" in target_names:
            regression_step_order_aliases_registry = (
                isinstance(node.value, ast.Name) and node.value.id == "FORMAL_STEP_ORDER"
            )
        if "STEP_LABELS" in target_names:
            if isinstance(node.value, ast.Dict):
                for key_node in node.value.keys:
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        step_labels_keys.append(key_node.value)
            elif isinstance(node.value, ast.Name):
                imported_source = imported_name_sources.get(node.value.id)
                if imported_source == ("core.test_suite_reporting", "TEST_SUITE_STEP_LABELS"):
                    step_labels_keys = _load_named_string_dict_keys(
                        project_root / "core" / "test_suite_reporting.py",
                        "TEST_SUITE_STEP_LABELS",
                    )

    formal_step_order_list = list(formal_step_order)
    missing_step_labels = [step for step in formal_step_order_list if step not in step_labels_keys]
    required_extra_labels = ["preflight", "dataset_prepare", "manifest"]
    missing_extra_labels = [label for label in required_extra_labels if label not in step_labels_keys]

    return {
        "regression_step_order_aliases_registry": regression_step_order_aliases_registry,
        "step_labels_keys": step_labels_keys,
        "missing_step_labels": missing_step_labels,
        "missing_extra_labels": missing_extra_labels,
    }


def summarize_single_formal_test_entry_contract(project_root: Path) -> Dict[str, Any]:
    app_dir = project_root / "apps"
    app_py_files = sorted(path.name for path in app_dir.glob("*.py"))
    cmd_text = (project_root / "doc" / "CMD.md").read_text(encoding="utf-8")
    architecture_text = (project_root / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    legacy_entry_paths = [path for path in LEGACY_APP_ENTRY_PATHS if (project_root / path).exists()]

    suspicious_app_entries: List[str] = []
    for path in sorted(app_dir.glob("*.py")):
        rel_path = str(path.relative_to(project_root)).replace("\\", "/")
        if rel_path == "apps/test_suite.py":
            continue
        if SUSPICIOUS_APP_ENTRY_PATTERN.search(path.stem):
            suspicious_app_entries.append(rel_path)

    return {
        "test_suite_exists": (project_root / "apps" / "test_suite.py").exists(),
        "cmd_declares_single_entry": CMD_SINGLE_ENTRY_TEXT in cmd_text,
        "architecture_declares_single_entry": ARCHITECTURE_SINGLE_ENTRY_TEXT in architecture_text,
        "legacy_entry_paths": legacy_entry_paths,
        "suspicious_app_entries": suspicious_app_entries,
        "app_py_files": app_py_files,
    }



def summarize_critical_helper_single_source_contract(project_root: Path) -> Dict[str, Any]:
    top_level_definitions: Dict[str, List[str]] = {}
    scanned_files: List[str] = []
    for rel_dir in ("apps", "core", "tools", "config", "strategies"):
        base_dir = project_root / rel_dir
        if not base_dir.is_dir():
            continue
        for path in sorted(base_dir.rglob("*.py")):
            rel_path = str(path.relative_to(project_root)).replace("\\", "/")
            scanned_files.append(rel_path)
            tree = _read_python_ast(path)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    top_level_definitions.setdefault(node.name, []).append(rel_path)

    missing_definitions: List[str] = []
    duplicate_definitions: List[str] = []
    canonical_definitions: Dict[str, str] = {}
    for canonical_path, helper_names in CRITICAL_HELPER_SINGLE_SOURCE_SPECS.items():
        for helper_name in helper_names:
            canonical_definitions[helper_name] = canonical_path
            definition_paths = sorted(top_level_definitions.get(helper_name, []))
            if canonical_path not in definition_paths:
                missing_definitions.append(f"{helper_name} -> {canonical_path}")
                continue
            extra_paths = [path for path in definition_paths if path != canonical_path]
            if extra_paths:
                duplicate_definitions.append(f"{helper_name} -> {', '.join(extra_paths)}")

    return {
        "scanned_files": scanned_files,
        "canonical_definitions": canonical_definitions,
        "top_level_definitions": {key: sorted(value) for key, value in sorted(top_level_definitions.items()) if key in canonical_definitions},
        "missing_definitions": missing_definitions,
        "duplicate_definitions": duplicate_definitions,
    }


def summarize_legacy_app_entry_doc_reference_contract(project_root: Path) -> Dict[str, Any]:
    legacy_doc_reference_lines: List[str] = []
    manual_delete_guidance_lines: List[str] = []

    for rel_path in LEGACY_DOC_GUIDANCE_FILES:
        path = project_root / rel_path
        for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw_line.strip()
            if any(legacy_path in stripped for legacy_path in LEGACY_APP_ENTRY_PATHS):
                legacy_doc_reference_lines.append(f"{rel_path}:{lineno}:{stripped}")
            if "手動刪除" in stripped and ("apps/" in stripped or "刪檔" in stripped):
                manual_delete_guidance_lines.append(f"{rel_path}:{lineno}:{stripped}")

    return {
        "legacy_doc_reference_lines": legacy_doc_reference_lines,
        "manual_delete_guidance_lines": manual_delete_guidance_lines,
    }
