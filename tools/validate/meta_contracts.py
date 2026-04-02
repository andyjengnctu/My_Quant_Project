from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_SETTINGS_SINGLE_ENTRY_TEXT = "`apps/test_suite.py` 為所有已實作測試的單一正式入口"
PROJECT_SETTINGS_REGISTRY_SOURCE_TEXT = "`tools/local_regression/formal_pipeline.py` 為單一真理來源"
CMD_SINGLE_ENTRY_TEXT = "正式對外入口為 `apps/test_suite.py`"
ARCHITECTURE_SINGLE_ENTRY_TEXT = "`apps/test_suite.py` 是日常唯一建議使用的一鍵測試入口"
LEGACY_APP_ENTRY_PATHS = ("apps/local_regression.py", "apps/validate_consistency.py")
SUSPICIOUS_APP_ENTRY_PATTERN = re.compile(r"(?:test|validate|regression|consistency)", re.IGNORECASE)


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
        if "STEP_LABELS" in target_names and isinstance(node.value, ast.Dict):
            parsed_keys: List[str] = []
            for key_node in node.value.keys:
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    parsed_keys.append(key_node.value)
            step_labels_keys = parsed_keys

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
    apps_dir = project_root / "apps"
    app_py_files = sorted(path.name for path in apps_dir.glob("*.py") if path.is_file())
    suspicious_app_entries = [
        name
        for name in app_py_files
        if name != "test_suite.py" and SUSPICIOUS_APP_ENTRY_PATTERN.search(Path(name).stem)
    ]
    legacy_entry_paths = [path for path in LEGACY_APP_ENTRY_PATHS if (project_root / path).exists()]

    project_settings_text = (project_root / "doc" / "PROJECT_SETTINGS.md").read_text(encoding="utf-8")
    cmd_text = (project_root / "doc" / "CMD.md").read_text(encoding="utf-8")
    architecture_text = (project_root / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    return {
        "test_suite_exists": (project_root / "apps" / "test_suite.py").exists(),
        "app_py_files": app_py_files,
        "suspicious_app_entries": suspicious_app_entries,
        "legacy_entry_paths": legacy_entry_paths,
        "project_settings_declares_single_entry": PROJECT_SETTINGS_SINGLE_ENTRY_TEXT in project_settings_text,
        "project_settings_declares_registry_source": PROJECT_SETTINGS_REGISTRY_SOURCE_TEXT in project_settings_text,
        "cmd_declares_single_entry": CMD_SINGLE_ENTRY_TEXT in cmd_text,
        "architecture_declares_single_entry": ARCHITECTURE_SINGLE_ENTRY_TEXT in architecture_text,
    }
