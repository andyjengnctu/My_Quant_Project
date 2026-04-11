import ast
from pathlib import Path
import importlib
import json
import re
import shlex
import tempfile
from unittest.mock import patch

from .checks import add_check
from .module_loader import build_project_absolute_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_PATH = PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md"
CMD_PATH = PROJECT_ROOT / "doc" / "CMD.md"
SYNTHETIC_VALIDATE_DIR = PROJECT_ROOT / "tools" / "validate"

from .meta_contracts import (
    extract_markdown_table_rows,
    load_defined_validate_names_from_synthetic_case_modules,
    load_imported_validate_names_from_synthetic_main_entry,
    load_synthetic_registry_entries_from_source,
    summarize_critical_helper_single_source_contract,
    summarize_legacy_app_entry_doc_reference_contract,
    summarize_no_reverse_app_import_contract,
    summarize_no_top_level_import_cycles_contract,
    summarize_project_settings_dynamic_test_boundary_contract,
    summarize_single_formal_test_entry_contract,
    summarize_synthetic_cases_import_target_resolution_contract,
)
from tools.local_regression.formal_pipeline import FORMAL_STEP_SPECS
from tools.local_regression.meta_quality_coverage import build_coverage_summary
from tools.local_regression.meta_quality_targets import (
    CORE_TRADING_COVERAGE_TARGETS,
    COVERAGE_BRANCH_MIN_FLOOR,
    COVERAGE_LINE_MIN_FLOOR,
    COVERAGE_MAX_LINE_BRANCH_GAP,
    COVERAGE_TARGETS,
    CRITICAL_COVERAGE_BRANCH_MIN_FLOOR,
    CRITICAL_COVERAGE_LINE_MIN_FLOOR,
    CRITICAL_COVERAGE_TARGETS,
    ENTRY_PATH_CRITICAL_COVERAGE_TARGETS,
    FORMAL_STEP_ENTRY_COVERAGE_TARGETS,
    FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS,
    POLICY_CONTRACT_COVERAGE_TARGETS,
    TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS,
)


def _load_main_table_statuses():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    statuses = {}
    headings = [
        ("B1. 專案設定對應清單（不含暫時特例）", 3),
        ("B2. 未明列於專案設定，但正式 test suite 應納入", 4),
        ("B3. 可隨策略升級調整的測試", 4),
    ]
    for heading, status_idx in headings:
        rows = extract_markdown_table_rows(text, heading)
        for cols in rows:
            if len(cols) > status_idx:
                statuses[cols[0]] = cols[status_idx]
    return statuses


def _load_main_table_catalog():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    catalog = {}
    for cols in extract_markdown_table_rows(text, "B1. 專案設定對應清單（不含暫時特例）"):
        if len(cols) > 5:
            catalog[cols[0]] = {
                "kind": "規則",
                "item": cols[2],
                "entry": cols[5],
                "status": cols[3],
            }
    for cols in extract_markdown_table_rows(text, "B2. 未明列於專案設定，但正式 test suite 應納入"):
        if len(cols) > 6:
            catalog[cols[0]] = {
                "kind": cols[2],
                "item": cols[3],
                "entry": cols[6],
                "status": cols[4],
            }
    for cols in extract_markdown_table_rows(text, "B3. 可隨策略升級調整的測試"):
        if len(cols) > 6:
            catalog[cols[0]] = {
                "kind": cols[2],
                "item": cols[3],
                "entry": cols[6],
                "status": cols[4],
            }
    return catalog


def _load_done_test_rows():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    rows = extract_markdown_table_rows(text, "T. 目前所有 `DONE` 的建議測試項目摘要")
    parsed = []
    for cols in rows:
        if len(cols) < 3:
            continue
        parsed.append(
            {
                "id": cols[0],
                "name": cols[1].replace("`", "").strip(),
                "b_id": cols[2],
            }
        )
    return parsed


def _load_done_b_rows():
    catalog = _load_main_table_catalog()
    return [
        {
            "kind": row["kind"],
            "b_id": b_id,
            "item": row["item"],
            "entry": row["entry"],
        }
        for b_id, row in sorted(catalog.items())
        if row.get("status") == "DONE"
    ]


def _load_convergence_latest_statuses():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    rows = extract_markdown_table_rows(text, "G. 逐項收斂紀錄")
    statuses = {}
    for cols in rows:
        if len(cols) < 4:
            continue
        item_id = cols[1].strip()
        transition = cols[3].strip()
        if not item_id or not transition:
            continue
        statuses[item_id] = transition.split("->")[-1].strip()
    return statuses


def _extract_cmd_python_commands():
    commands = []
    for raw_line in CMD_PATH.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "- ", "```")):
            continue
        if not stripped.startswith("python "):
            continue
        command = stripped.split("#", 1)[0].strip()
        if command:
            commands.append(command)
    return commands


def _replace_markdown_table_row(text: str, *, heading: str, row_id: str, id_col_idx: int, update_cols, match_index: int = 0):
    original_had_trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    in_target_section = False
    matched_count = 0
    for line_idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            current_heading = stripped.lstrip("#").strip()
            if in_target_section and current_heading != heading:
                break
            in_target_section = current_heading == heading
            continue
        if not in_target_section or not stripped.startswith("|"):
            continue
        cols = [part.strip() for part in raw_line.split("|")[1:-1]]
        if len(cols) <= id_col_idx or cols[id_col_idx] != row_id:
            continue
        if matched_count != match_index:
            matched_count += 1
            continue
        updated_cols = update_cols(list(cols))
        lines[line_idx] = "| " + " | ".join(updated_cols) + " |"
        updated_text = "\n".join(lines)
        if original_had_trailing_newline:
            updated_text += "\n"
        return updated_text
    raise ValueError(f"找不到 checklist heading={heading}, row_id={row_id}, match_index={match_index}")


def _swap_markdown_table_rows(text: str, *, heading: str, row_id_a: str, row_id_b: str, id_col_idx: int):
    original_had_trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    in_target_section = False
    row_idx_a = None
    row_idx_b = None
    for line_idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            current_heading = stripped.lstrip("#").strip()
            if in_target_section and current_heading != heading:
                break
            in_target_section = current_heading == heading
            continue
        if not in_target_section or not stripped.startswith("|"):
            continue
        cols = [part.strip() for part in raw_line.split("|")[1:-1]]
        if len(cols) <= id_col_idx:
            continue
        if cols[id_col_idx] == row_id_a and row_idx_a is None:
            row_idx_a = line_idx
        elif cols[id_col_idx] == row_id_b and row_idx_b is None:
            row_idx_b = line_idx
    if row_idx_a is None or row_idx_b is None:
        raise ValueError(f"找不到 checklist heading={heading}, row_id_a={row_id_a}, row_id_b={row_id_b}")
    lines[row_idx_a], lines[row_idx_b] = lines[row_idx_b], lines[row_idx_a]
    swapped_text = "\n".join(lines)
    if original_had_trailing_newline:
        swapped_text += "\n"
    return swapped_text


def _read_summary_value(result: dict, key: str, default=None):
    if key in result:
        return result.get(key)
    extra = result.get("extra")
    if isinstance(extra, dict) and key in extra:
        return extra.get(key)
    return default


def _parsed_module_uses_numpy_alias(parsed_module: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np"
        for node in ast.walk(parsed_module)
    )


def _parsed_module_declares_numpy_alias_import(parsed_module: ast.AST) -> bool:
    for node in getattr(parsed_module, "body", []):
        if isinstance(node, ast.Import):
            if any(alias.name == "numpy" and alias.asname == "np" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numpy" and any(alias.asname == "np" for alias in node.names):
                return True
    return False


def _find_nonpositive_initial_capital_assignments(parsed_module: ast.AST):
    hits = []
    for node in ast.walk(parsed_module):
        target = None
        value = None
        if isinstance(node, ast.Assign):
            value = node.value
            for candidate in node.targets:
                if isinstance(candidate, ast.Attribute) and candidate.attr == "initial_capital":
                    target = candidate
                    break
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            if isinstance(node.target, ast.Attribute) and node.target.attr == "initial_capital":
                target = node.target
        if target is None or value is None:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, (int, float)):
            continue
        if float(value.value) <= 0.0:
            hits.append({"lineno": getattr(node, "lineno", None), "value": float(value.value)})
    return hits


def _parsed_module_declares_specific_from_import(parsed_module: ast.AST, *, module_name: str, imported_name: str) -> bool:
    for node in getattr(parsed_module, "body", []):
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            if any(alias.name == imported_name for alias in node.names):
                return True
    return False


def validate_cmd_document_contract_case(_base_params):
    from tools.local_regression.run_all import STEP_NAMES
    from tools.local_regression.run_quick_gate import HELP_TARGETS
    from tools.validate.preflight_env import _LOCAL_REGRESSION_STEP_ORDER

    case_id = "META_CMD_DOCUMENT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    commands = _extract_cmd_python_commands()
    unique_commands = list(dict.fromkeys(commands))
    help_script_paths = {str(cmd[1]).replace("\\", "/") for cmd, _ in HELP_TARGETS if len(cmd) >= 2 and str(cmd[1]).endswith(".py")}
    project_command_count = 0

    add_check(results, "meta_cmd_contract", case_id, "cmd_python_commands_nonempty", True, bool(unique_commands))

    for command in unique_commands:
        command = command.replace("\\", "/")
        parts = shlex.split(command, posix=True)
        if len(parts) < 2 or parts[0] != "python":
            continue
        if parts[1] == "-m":
            if parts[2:4] == ["pip", "install"] and "-r" in parts:
                req_idx = parts.index("-r")
                req_path = PROJECT_ROOT / parts[req_idx + 1]
                add_check(results, "meta_cmd_contract", case_id, "cmd_requirements_lock_exists", True, req_path.exists())
            continue

        script_rel = parts[1].replace("\\", "/")
        project_command_count += 1
        script_path = PROJECT_ROOT / script_rel
        metric_prefix = script_rel.replace("/", "_").replace(".", "_")
        add_check(results, "meta_cmd_contract", case_id, f"{metric_prefix}_script_exists", True, script_path.is_file())
        add_check(results, "meta_cmd_contract", case_id, f"{metric_prefix}_covered_by_help_target", True, script_rel in help_script_paths)

        if "--dataset" in parts:
            dataset_value = parts[parts.index("--dataset") + 1]
            add_check(results, "meta_cmd_contract", case_id, f"{metric_prefix}_dataset_value_valid", True, dataset_value in {"full", "reduced"})

        if script_rel == "tools/local_regression/run_all.py" and "--only" in parts:
            only_value = parts[parts.index("--only") + 1]
            only_steps = [token.strip() for token in only_value.split(",") if token.strip()]
            invalid_steps = [step for step in only_steps if step not in STEP_NAMES]
            add_check(results, "meta_cmd_contract", case_id, f"{metric_prefix}_only_steps_valid", [], invalid_steps)

        if script_rel == "tools/validate/preflight_env.py" and "--steps" in parts:
            steps_value = parts[parts.index("--steps") + 1]
            step_names = [token.strip() for token in steps_value.split(",") if token.strip()]
            invalid_steps = [step for step in step_names if step not in _LOCAL_REGRESSION_STEP_ORDER]
            add_check(results, "meta_cmd_contract", case_id, f"{metric_prefix}_preflight_steps_valid", [], invalid_steps)

    summary["command_count"] = len(unique_commands)
    summary["project_command_count"] = project_command_count
    return results, summary

def validate_no_reverse_app_layer_dependencies_case(_base_params):
    case_id = "META_NO_REVERSE_APP_LAYER_DEPENDENCIES"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    contract = summarize_no_reverse_app_import_contract(PROJECT_ROOT)
    violations = [
        f"{item['path']}:{item['lineno']} -> {item['module']}"
        for item in contract["violations"]
    ]
    add_check(results, "meta_entry_contract", case_id, "core_and_tools_do_not_import_apps", [], violations)

    summary["violation_count"] = len(violations)
    summary["violations"] = violations
    return results, summary


def validate_critical_helper_single_source_contract_case(_base_params):
    case_id = "META_CRITICAL_HELPER_SINGLE_SOURCE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    contract = summarize_critical_helper_single_source_contract(PROJECT_ROOT)
    add_check(results, "meta_entry_contract", case_id, "critical_helpers_defined_in_canonical_modules", [], contract["missing_definitions"])
    add_check(results, "meta_entry_contract", case_id, "critical_helpers_not_redefined_outside_canonical_modules", [], contract["duplicate_definitions"])

    summary["tracked_helper_count"] = len(contract["canonical_definitions"])
    summary["missing_definitions"] = contract["missing_definitions"]
    summary["duplicate_definitions"] = contract["duplicate_definitions"]
    return results, summary


def validate_no_top_level_import_cycles_case(_base_params):
    case_id = "META_NO_TOP_LEVEL_IMPORT_CYCLES"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    contract = summarize_no_top_level_import_cycles_contract(PROJECT_ROOT)
    violations = [
        " | ".join(item["modules"])
        for item in contract["violations"]
    ]
    add_check(results, "meta_entry_contract", case_id, "project_has_no_top_level_import_cycles", [], violations)

    summary["module_count"] = contract["module_count"]
    summary["cycle_count"] = len(contract["violations"])
    summary["cycles"] = violations
    return results, summary


def validate_single_formal_test_entry_contract_case(_base_params):
    case_id = "META_SINGLE_FORMAL_TEST_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    contract = summarize_single_formal_test_entry_contract(PROJECT_ROOT)
    add_check(results, "meta_entry_contract", case_id, "test_suite_entry_file_exists", True, contract["test_suite_exists"])
    add_check(results, "meta_entry_contract", case_id, "project_settings_declares_single_entry", True, contract["project_settings_declares_single_entry"])
    add_check(results, "meta_entry_contract", case_id, "cmd_declares_single_entry", True, contract["cmd_declares_single_entry"])
    add_check(results, "meta_entry_contract", case_id, "architecture_declares_single_entry", True, contract["architecture_declares_single_entry"])
    add_check(results, "meta_entry_contract", case_id, "no_legacy_app_test_entries", [], contract["legacy_entry_paths"])
    add_check(results, "meta_entry_contract", case_id, "no_suspicious_alternate_app_test_entries", [], contract["suspicious_app_entries"])

    summary["app_py_files"] = contract["app_py_files"]
    summary["legacy_entry_paths"] = contract["legacy_entry_paths"]
    summary["suspicious_app_entries"] = contract["suspicious_app_entries"]
    return results, summary



def validate_project_settings_dynamic_test_boundary_case(_base_params):
    case_id = "META_PROJECT_SETTINGS_DYNAMIC_TEST_BOUNDARY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    contract = summarize_project_settings_dynamic_test_boundary_contract(PROJECT_ROOT)
    add_check(results, "meta_entry_contract", case_id, "project_settings_forbids_dynamic_test_rerun", True, contract["project_settings_declares_no_dynamic_test_rerun"])
    add_check(results, "meta_entry_contract", case_id, "project_settings_forbids_formal_step_validator_bypass", True, contract["project_settings_declares_no_formal_step_bypass"])

    summary["project_settings_declares_no_dynamic_test_rerun"] = contract["project_settings_declares_no_dynamic_test_rerun"]
    summary["project_settings_declares_no_formal_step_bypass"] = contract["project_settings_declares_no_formal_step_bypass"]
    return results, summary



def validate_project_settings_init_sl_frozen_plan_principle_case(_base_params):
    case_id = "META_PROJECT_SETTINGS_PHYSICAL_TRADING_PRINCIPLES"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    project_settings_text = (PROJECT_ROOT / "doc" / "PROJECT_SETTINGS.md").read_text(encoding="utf-8")
    checklist_text = (PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md").read_text(encoding="utf-8")

    governance_text = "細部契約與驗證細節一律下沉到 `doc/TEST_SUITE_CHECKLIST.md`"
    minimal_constraint_text = "最小必要約束"
    l_only_text = "`L` 只作進場上限 / 最壞風險 sizing 上界"
    pfill_text = "`P_fill + ATR_t`"
    continuation_barrier_text = "固定反事實 `P' = min(Open, L)`"
    inclusive_hit_text = "長倉 hit 採 `Low <= line` / `High >= line`"

    add_check(results, "meta_entry_contract", case_id, "project_settings_declares_minimal_physical_constraint_priority", True, minimal_constraint_text in project_settings_text)
    add_check(results, "meta_entry_contract", case_id, "project_settings_declares_trading_detail_sink_to_checklist", True, governance_text in project_settings_text)
    add_check(results, "meta_entry_contract", case_id, "checklist_declares_l_is_entry_and_sizing_only", True, l_only_text in checklist_text)
    add_check(results, "meta_entry_contract", case_id, "checklist_declares_first_actionable_stop_uses_pfill_and_atr", True, pfill_text in checklist_text)
    add_check(results, "meta_entry_contract", case_id, "checklist_declares_extended_candidate_fixed_counterfactual_barrier", True, continuation_barrier_text in checklist_text)
    add_check(results, "meta_entry_contract", case_id, "checklist_declares_inclusive_hit_semantics", True, inclusive_hit_text in checklist_text)

    summary["project_settings_declares_minimal_physical_constraint_priority"] = minimal_constraint_text in project_settings_text
    summary["project_settings_declares_trading_detail_sink_to_checklist"] = governance_text in project_settings_text
    summary["checklist_declares_l_is_entry_and_sizing_only"] = l_only_text in checklist_text
    summary["checklist_declares_first_actionable_stop_uses_pfill_and_atr"] = pfill_text in checklist_text
    summary["checklist_declares_extended_candidate_fixed_counterfactual_barrier"] = continuation_barrier_text in checklist_text
    summary["checklist_declares_inclusive_hit_semantics"] = inclusive_hit_text in checklist_text
    return results, summary


def validate_gui_tcl_fallback_traceability_contract_case(_base_params):
    case_id = "META_GUI_TCL_FALLBACK_TRACEABILITY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    scan_targets = sorted((PROJECT_ROOT / "tools" / "gui").rglob("*.py"))
    syntax_errors = []
    gui_tcl_fallback_traceability_failures = []
    scanned_files = []

    def _is_gui_tcl_exception_type(node):
        if isinstance(node, ast.Name):
            return node.id == "TclError"
        if isinstance(node, ast.Attribute):
            return ast.unparse(node) in {"tk.TclError", "tkinter.TclError"}
        if isinstance(node, ast.Tuple):
            return any(_is_gui_tcl_exception_type(element) for element in node.elts)
        return False

    def _handler_uses_exception_name(handler, exception_name):
        handler_module = ast.Module(body=handler.body, type_ignores=[])
        return any(
            isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id == exception_name
            for child in ast.walk(handler_module)
        )

    def _handler_reraises(handler):
        return any(isinstance(child, ast.Raise) for child in ast.walk(ast.Module(body=handler.body, type_ignores=[])))

    for path in scan_targets:
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        scanned_files.append(rel_path)
        try:
            source_text = path.read_text(encoding="utf-8")
            parsed = ast.parse(source_text, filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(f"{rel_path}:{exc.lineno}: {exc.msg}")
            continue

        for node in ast.walk(parsed):
            if not isinstance(node, ast.ExceptHandler) or node.type is None or not _is_gui_tcl_exception_type(node.type):
                continue
            if not node.name:
                gui_tcl_fallback_traceability_failures.append(f"{rel_path}:{node.lineno}: GUI TclError fallback must bind exception name")
                continue
            if _handler_reraises(node):
                continue
            if not _handler_uses_exception_name(node, node.name):
                gui_tcl_fallback_traceability_failures.append(f"{rel_path}:{node.lineno}: GUI TclError fallback must use bound exception or re-raise")

    add_check(results, "meta_contract", case_id, "gui_tcl_fallback_handler_files_parse", [], syntax_errors)
    add_check(results, "meta_contract", case_id, "gui_tcl_fallbacks_bind_and_trace_or_reraise", [], gui_tcl_fallback_traceability_failures)

    summary["scanned_file_count"] = len(scanned_files)
    summary["failure_count"] = len(gui_tcl_fallback_traceability_failures)
    summary["syntax_error_count"] = len(syntax_errors)
    return results, summary


def validate_optional_dependency_fallback_traceability_contract_case(_base_params):
    case_id = "META_OPTIONAL_DEPENDENCY_FALLBACK_TRACEABILITY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    scan_targets = [
        PROJECT_ROOT / "tools" / "trade_analysis" / "charting.py",
        PROJECT_ROOT / "tools" / "downloader" / "runtime.py",
        PROJECT_ROOT / "tools" / "workbench_ui" / "single_stock_inspector.py",
        PROJECT_ROOT / "tools" / "validate" / "main.py",
    ]
    syntax_errors = []
    optional_fallback_traceability_failures = []
    scanned_files = []

    def _is_optional_import_exception_type(node):
        if isinstance(node, ast.Name):
            return node.id in {"ImportError", "ModuleNotFoundError"}
        if isinstance(node, ast.Attribute):
            return ast.unparse(node) in {"ImportError", "ModuleNotFoundError"}
        if isinstance(node, ast.Tuple):
            return any(_is_optional_import_exception_type(element) for element in node.elts)
        return False

    def _handler_uses_exception_name(handler, exception_name):
        handler_module = ast.Module(body=handler.body, type_ignores=[])
        return any(
            isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id == exception_name
            for child in ast.walk(handler_module)
        )

    def _handler_reraises(handler):
        return any(isinstance(child, ast.Raise) for child in ast.walk(ast.Module(body=handler.body, type_ignores=[])))

    for path in scan_targets:
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        scanned_files.append(rel_path)
        try:
            source_text = path.read_text(encoding="utf-8")
            parsed = ast.parse(source_text, filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(f"{rel_path}:{exc.lineno}: {exc.msg}")
            continue

        for node in ast.walk(parsed):
            if not isinstance(node, ast.ExceptHandler) or node.type is None or not _is_optional_import_exception_type(node.type):
                continue
            if not node.name:
                optional_fallback_traceability_failures.append(f"{rel_path}:{node.lineno}: optional dependency fallback must bind exception name")
                continue
            if _handler_reraises(node):
                continue
            if not _handler_uses_exception_name(node, node.name):
                optional_fallback_traceability_failures.append(f"{rel_path}:{node.lineno}: optional dependency fallback must use bound exception or re-raise")

    add_check(results, "meta_contract", case_id, "optional_dependency_fallback_handler_files_parse", [], syntax_errors)
    add_check(results, "meta_contract", case_id, "optional_dependency_fallbacks_bind_and_trace_or_reraise", [], optional_fallback_traceability_failures)

    summary["scanned_file_count"] = len(scanned_files)
    summary["failure_count"] = len(optional_fallback_traceability_failures)
    summary["syntax_error_count"] = len(syntax_errors)
    return results, summary


def validate_specific_pass_only_exception_traceability_contract_case(_base_params):
    case_id = "META_SPECIFIC_PASS_ONLY_EXCEPTION_TRACEABILITY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    scan_roots = [
        PROJECT_ROOT / "apps",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "core",
        PROJECT_ROOT / "strategies",
        PROJECT_ROOT / "tools",
    ]
    syntax_errors = []
    specific_pass_only_failures = []
    scanned_files = []
    allowed_pass_only_exception_names = {"FileNotFoundError"}

    def _exception_type_names(node):
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            return [node.attr]
        if isinstance(node, ast.Tuple):
            names = []
            for element in node.elts:
                names.extend(_exception_type_names(element))
            return names
        return []

    def _is_traceability_contract_exempt(path):
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        return rel_path.startswith("tools/validate/synthetic_")

    for scan_root in scan_roots:
        for path in sorted(scan_root.rglob("*.py")):
            if _is_traceability_contract_exempt(path):
                continue
            rel_path = path.relative_to(PROJECT_ROOT).as_posix()
            scanned_files.append(rel_path)
            try:
                source_text = path.read_text(encoding="utf-8")
                parsed = ast.parse(source_text, filename=str(path))
            except SyntaxError as exc:
                syntax_errors.append(f"{rel_path}:{exc.lineno}: {exc.msg}")
                continue

            for node in ast.walk(parsed):
                if not isinstance(node, ast.ExceptHandler) or node.type is None:
                    continue
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
                    continue
                exception_names = set(_exception_type_names(node.type))
                if not exception_names:
                    continue
                if exception_names & {"Exception", "BaseException", "ImportError", "ModuleNotFoundError", "TclError"}:
                    continue
                if exception_names <= allowed_pass_only_exception_names:
                    continue
                specific_pass_only_failures.append(
                    f"{rel_path}:{node.lineno}: pass-only specific exception handler must trace, re-raise, or use an allowed control-flow exception"
                )

    add_check(results, "meta_contract", case_id, "specific_pass_only_exception_handler_files_parse", [], syntax_errors)
    add_check(results, "meta_contract", case_id, "specific_pass_only_exception_handlers_forbidden", [], specific_pass_only_failures)

    summary["scanned_file_count"] = len(scanned_files)
    summary["failure_count"] = len(specific_pass_only_failures)
    summary["syntax_error_count"] = len(syntax_errors)
    return results, summary


def validate_broad_exception_traceability_contract_case(_base_params):
    case_id = "META_BROAD_EXCEPTION_TRACEABILITY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    scan_roots = [
        PROJECT_ROOT / "apps",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "core",
        PROJECT_ROOT / "strategies",
        PROJECT_ROOT / "tools",
    ]
    syntax_errors = []
    broad_exception_traceability_failures = []
    scanned_files = []

    def _is_broad_exception_type(node):
        if isinstance(node, ast.Name):
            return node.id in {"Exception", "BaseException"}
        if isinstance(node, ast.Tuple):
            return any(_is_broad_exception_type(element) for element in node.elts)
        return False

    def _handler_uses_exception_name(handler, exception_name):
        handler_module = ast.Module(body=handler.body, type_ignores=[])
        return any(
            isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id == exception_name
            for child in ast.walk(handler_module)
        )

    def _handler_reraises(handler):
        return any(isinstance(child, ast.Raise) for child in ast.walk(ast.Module(body=handler.body, type_ignores=[])))

    for scan_root in scan_roots:
        for path in sorted(scan_root.rglob("*.py")):
            rel_path = path.relative_to(PROJECT_ROOT).as_posix()
            scanned_files.append(rel_path)
            try:
                source_text = path.read_text(encoding="utf-8")
                parsed = ast.parse(source_text, filename=str(path))
            except SyntaxError as exc:
                syntax_errors.append(f"{rel_path}:{exc.lineno}: {exc.msg}")
                continue

            for node in ast.walk(parsed):
                if not isinstance(node, ast.ExceptHandler) or node.type is None or not _is_broad_exception_type(node.type):
                    continue
                if not node.name:
                    broad_exception_traceability_failures.append(f"{rel_path}:{node.lineno}: broad exception handler must bind exception name")
                    continue
                if _handler_reraises(node):
                    continue
                if not _handler_uses_exception_name(node, node.name):
                    broad_exception_traceability_failures.append(f"{rel_path}:{node.lineno}: broad exception handler must use bound exception or re-raise")

    add_check(results, "meta_contract", case_id, "broad_exception_handler_files_parse", [], syntax_errors)
    add_check(results, "meta_contract", case_id, "broad_exception_handlers_bind_and_trace_or_reraise", [], broad_exception_traceability_failures)

    summary["scanned_file_count"] = len(scanned_files)
    summary["failure_count"] = len(broad_exception_traceability_failures)
    summary["syntax_error_count"] = len(syntax_errors)
    return results, summary


def validate_no_legacy_app_entry_doc_references_case(_base_params):
    case_id = "META_NO_LEGACY_APP_ENTRY_DOC_REFERENCES"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    contract = summarize_legacy_app_entry_doc_reference_contract(PROJECT_ROOT)
    add_check(results, "meta_entry_contract", case_id, "cmd_and_architecture_have_no_legacy_app_entry_references", [], contract["legacy_doc_reference_lines"])
    add_check(results, "meta_entry_contract", case_id, "docs_have_no_manual_delete_guidance_for_app_entries", [], contract["manual_delete_guidance_lines"])

    summary["legacy_doc_reference_lines"] = contract["legacy_doc_reference_lines"]
    summary["manual_delete_guidance_lines"] = contract["manual_delete_guidance_lines"]
    return results, summary


def validate_app_thin_wrapper_export_contract_case(_base_params):
    case_id = "META_APP_THIN_WRAPPER_EXPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    module_names = [
        "apps.portfolio_sim",
        "apps.vip_scanner",
    ]
    import_failures = []
    duplicated_lazy_export_failures = []
    missing_all_export_failures = []
    unresolved_lazy_export_failures = []

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
        except Exception as exc:
            import_failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue

        lazy_exports = list(getattr(module, "LAZY_EXPORTS", ()))
        exported_names = set(getattr(module, "__all__", []))
        duplicated_lazy_exports = sorted(name for name in set(lazy_exports) if lazy_exports.count(name) > 1)
        missing_all_exports = sorted(set(lazy_exports) - exported_names)
        unresolved_lazy_exports = []
        for export_name in sorted(set(lazy_exports)):
            try:
                getattr(module, export_name)
            except AttributeError:
                unresolved_lazy_exports.append(export_name)

        if duplicated_lazy_exports:
            duplicated_lazy_export_failures.append(f"{module_name}: {duplicated_lazy_exports}")
        if missing_all_exports:
            missing_all_export_failures.append(f"{module_name}: {missing_all_exports}")
        if unresolved_lazy_exports:
            unresolved_lazy_export_failures.append(f"{module_name}: {unresolved_lazy_exports}")

    add_check(results, "meta_entry_contract", case_id, "thin_wrapper_modules_importable", [], import_failures)
    add_check(results, "meta_entry_contract", case_id, "thin_wrapper_lazy_exports_unique", [], duplicated_lazy_export_failures)
    add_check(results, "meta_entry_contract", case_id, "thin_wrapper_lazy_exports_listed_in___all__", [], missing_all_export_failures)
    add_check(results, "meta_entry_contract", case_id, "thin_wrapper_lazy_exports_resolvable", [], unresolved_lazy_export_failures)

    summary["module_names"] = module_names
    summary["import_failures"] = import_failures
    summary["duplicated_lazy_export_failures"] = duplicated_lazy_export_failures
    summary["missing_all_export_failures"] = missing_all_export_failures
    summary["unresolved_lazy_export_failures"] = unresolved_lazy_export_failures
    return results, summary


def validate_synthetic_registry_metadata_contract_case(_base_params):
    case_id = "META_SYNTHETIC_REGISTRY_METADATA"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    entries = load_synthetic_registry_entries_from_source(PROJECT_ROOT)
    entry_names = [entry["name"] for entry in entries]
    allowed_layers = {
        "core_invariant",
        "unit_boundary",
        "meta_contract",
        "output_contract",
        "error_path",
        "data_quality",
        "cli_contract",
        "strategy_contract",
        "regression_contract",
    }
    allowed_cost_classes = {"fast", "medium", "heavy"}

    invalid_names = sorted(name for name in entry_names if not name.startswith("validate_"))
    invalid_layers = sorted(entry["name"] for entry in entries if entry["layer"] not in allowed_layers)
    invalid_cost_classes = sorted(entry["name"] for entry in entries if entry["cost_class"] not in allowed_cost_classes)
    missing_impacted_modules = sorted(entry["name"] for entry in entries if not entry["impacted_modules"])
    invalid_impacted_modules = sorted(
        f"{entry['name']}:{module_path}"
        for entry in entries
        for module_path in entry["impacted_modules"]
        if (not module_path)
        or (module_path.strip() != module_path)
        or module_path.startswith("/")
        or "\\" in module_path
        or not module_path.endswith((".py", ".md", ".json"))
    )
    duplicated_impacted_modules = sorted(
        entry["name"] for entry in entries if len(entry["impacted_modules"]) != len(set(entry["impacted_modules"]))
    )
    duplicate_entry_names = sorted(name for name in set(entry_names) if entry_names.count(name) > 1)

    add_check(results, "meta_registry", case_id, "registry_metadata_not_empty", True, len(entries) > 0)
    add_check(results, "meta_registry", case_id, "registry_metadata_names_unique", [], duplicate_entry_names)
    add_check(results, "meta_registry", case_id, "registry_metadata_validator_names_prefixed", [], invalid_names)
    add_check(results, "meta_registry", case_id, "registry_metadata_layers_valid", [], invalid_layers)
    add_check(results, "meta_registry", case_id, "registry_metadata_cost_classes_valid", [], invalid_cost_classes)
    add_check(results, "meta_registry", case_id, "registry_metadata_impacted_modules_present", [], missing_impacted_modules)
    add_check(results, "meta_registry", case_id, "registry_metadata_impacted_modules_normalized", [], invalid_impacted_modules)
    add_check(results, "meta_registry", case_id, "registry_metadata_impacted_modules_unique_per_entry", [], duplicated_impacted_modules)

    layer_counts = {}
    for entry in entries:
        layer_counts[entry["layer"]] = layer_counts.get(entry["layer"], 0) + 1
    summary["validator_count"] = len(entries)
    summary["layer_counts"] = layer_counts
    return results, summary


def validate_checklist_g_single_note_entry_delimiter_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_G_SINGLE_NOTE_ENTRY_DELIMITER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="G. 逐項收斂紀錄",
            row_id="B38",
            id_col_idx=1,
            update_cols=lambda cols: cols[:4] + ["`run_meta_quality.py` / `meta_contracts.py`"],
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_g_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_g_note_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    g_note_result = result_by_name.get("checklist_g_rows_use_single_note_entry", {})
    invalid_rows = g_note_result.get("invalid_note_rows")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(g_note_result, "invalid_note_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_note_single_entry_guard_fails", "FAIL", g_note_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_note_reports_multiple_entries", True, any(row.get("id") == "B38" for row in invalid_rows))

    summary["guard_status"] = g_note_result.get("status")
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
    return results, summary


def validate_checklist_f2_formal_command_single_entry_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_F2_FORMAL_COMMAND_SINGLE_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    command_entry = "tools/validate/cli.py --dataset reduced"
    parsed_entries = meta_quality_module._extract_checklist_test_entries(f"`{command_entry}`")
    add_check(results, "meta_checklist", case_id, "formal_command_entry_parses_as_single_entry", [command_entry], parsed_entries)

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="T. 目前所有 `DONE` 的建議測試項目摘要",
            row_id="T107",
            id_col_idx=0,
            update_cols=lambda cols: [cols[0], f"`{command_entry}`", cols[2]],
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_f2_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_f2_formal_command_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    f2_result = result_by_name.get("checklist_f_rows_use_single_test_entry", {})
    invalid_rows = f2_result.get("invalid_entries")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(f2_result, "invalid_entries", [])

    add_check(results, "meta_checklist", case_id, "mutated_f2_formal_command_single_entry_guard_passes", "PASS", f2_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_f2_formal_command_not_reported_invalid", False, any(row.get("id") == "T107" for row in invalid_rows))

    summary["guard_status"] = f2_result.get("status")
    summary["parsed_entries"] = parsed_entries
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
    return results, summary


def validate_checklist_done_test_summary_markdown_structure_case(_base_params):
    case_id = "META_CHECKLIST_DONE_TEST_SUMMARY_MARKDOWN_STRUCTURE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    lines = CHECKLIST_PATH.read_text(encoding="utf-8").splitlines()
    heading = "### T. 目前所有 `DONE` 的建議測試項目摘要"
    try:
        heading_index = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        add_check(results, "meta_checklist", case_id, "done_test_summary_heading_exists", True, False)
        return results, summary

    header_index = None
    for cursor in range(heading_index + 1, len(lines)):
        stripped = lines[cursor].strip()
        if stripped.startswith(("## ", "### ")):
            break
        if stripped.startswith("|"):
            header_index = cursor
            break
    if header_index is None:
        add_check(results, "meta_checklist", case_id, "done_test_summary_table_header_exists", True, False)
        return results, summary

    separator_line = lines[header_index + 1].strip() if header_index + 1 < len(lines) else ""
    has_separator = separator_line.startswith("|") and set(separator_line.replace("|", "").replace(" ", "").replace(":", "")) <= {"-"}
    add_check(results, "meta_checklist", case_id, "done_test_summary_table_has_markdown_separator_row", True, has_separator)

    done_test_rows = _load_done_test_rows()
    invalid_ids = [row["id"] for row in done_test_rows if not re.fullmatch(r"T\d+", row["id"])]
    invalid_b_ids = [row["b_id"] for row in done_test_rows if not re.fullmatch(r"B\d+", row["b_id"])]
    add_check(results, "meta_checklist", case_id, "done_test_summary_rows_use_valid_t_ids", [], invalid_ids)
    add_check(results, "meta_checklist", case_id, "done_test_summary_rows_use_valid_b_ids", [], invalid_b_ids)

    summary["invalid_ids"] = invalid_ids
    summary["invalid_b_ids"] = invalid_b_ids
    return results, summary


def validate_checklist_f2_single_entry_delimiter_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_F2_SINGLE_ENTRY_DELIMITER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="T. 目前所有 `DONE` 的建議測試項目摘要",
            row_id="T107",
            id_col_idx=0,
            update_cols=lambda cols: [cols[0], "`validate_checklist_g_single_note_entry_delimiter_case` / `tools/local_regression/run_meta_quality.py`", cols[2]],
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_f2_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_f2_entry_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    f2_result = result_by_name.get("checklist_f_rows_use_single_test_entry", {})
    invalid_rows = f2_result.get("invalid_entries")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(f2_result, "invalid_entries", [])

    add_check(results, "meta_checklist", case_id, "mutated_f2_single_entry_guard_fails", "FAIL", f2_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_f2_reports_multiple_entries", True, any(row.get("id") == "T107" for row in invalid_rows))

    summary["guard_status"] = f2_result.get("status")
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
    return results, summary


def validate_checklist_no_legacy_f1_section_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_NO_LEGACY_F1_SECTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    if "### F1. 目前所有 `DONE` 的主表項目摘要" in original_text:
        add_check(results, "meta_checklist", case_id, "baseline_has_no_legacy_f1_section", True, False)
        return results, summary

    insertion_target = "### T. 目前所有 `DONE` 的建議測試項目摘要"
    if insertion_target not in original_text:
        add_check(results, "meta_checklist", case_id, "target_f_section_exists_for_mutation", True, False)
        return results, summary

    legacy_block = (
        "### F1. 目前所有 `DONE` 的主表項目摘要\n\n"
        "| 類型 | ID | 項目 |\n"
        "|---|---|---|\n"
        "| Meta | B26 | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） |\n\n"
    )
    mutated_text = original_text.replace(insertion_target, legacy_block + insertion_target, 1)

    with tempfile.TemporaryDirectory(prefix="meta_checklist_legacy_f1_section_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    legacy_result = result_by_name.get("checklist_has_no_legacy_f1_section", {})
    legacy_present = legacy_result.get("legacy_f1_section_present")
    if legacy_present is None:
        legacy_present = _read_summary_value(legacy_result, "legacy_f1_section_present")

    add_check(results, "meta_checklist", case_id, "mutated_legacy_f1_section_guard_fails", "FAIL", legacy_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_legacy_f1_section_reported", True, bool(legacy_present))

    summary["guard_status"] = legacy_result.get("status")
    summary["legacy_f1_section_present"] = legacy_present
    return results, summary


def validate_checklist_no_legacy_d_section_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_NO_LEGACY_D_SECTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    if "## D. 建議先補的測試項目" in original_text:
        add_check(results, "meta_checklist", case_id, "baseline_has_no_legacy_d_section", True, False)
        return results, summary

    mutated_block = (
        "## D. 建議先補的測試項目\n\n"
        "| ID | 測試 / 工具 | 說明 |\n"
        "|---|---|---|\n"
        "| D97 | `validate_core_trading_modules_in_coverage_targets_case` | legacy backlog section should not return |\n\n"
    )
    insertion_target = "## E. 未完成缺口摘要"
    if insertion_target not in original_text:
        add_check(results, "meta_checklist", case_id, "target_e_section_exists_for_mutation", True, False)
        return results, summary
    mutated_text = original_text.replace(insertion_target, mutated_block + insertion_target, 1)

    with tempfile.TemporaryDirectory(prefix="meta_checklist_legacy_d_section_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    legacy_result = result_by_name.get("checklist_has_no_legacy_d_section", {})
    legacy_present = legacy_result.get("legacy_d_section_present")
    if legacy_present is None:
        legacy_present = _read_summary_value(legacy_result, "legacy_d_section_present")

    add_check(results, "meta_checklist", case_id, "mutated_legacy_d_section_guard_fails", "FAIL", legacy_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_legacy_d_section_reports_presence", True, bool(legacy_present))

    summary["guard_status"] = legacy_result.get("status")
    summary["legacy_d_section_present"] = bool(legacy_present)
    return results, summary


def validate_checklist_g_transition_format_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_G_TRANSITION_FORMAT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="G. 逐項收斂紀錄",
            row_id="B38",
            id_col_idx=1,
            update_cols=lambda cols: cols[:3] + ["DONE"] + cols[4:],
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_g_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_g_transition_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    g_transition_result = result_by_name.get("checklist_g_rows_have_valid_status_transition", {})
    invalid_rows = g_transition_result.get("invalid_transition_rows")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(g_transition_result, "invalid_transition_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_transition_guard_fails", "FAIL", g_transition_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_transition_reports_invalid_row", True, any(row.get("id") == "B38" for row in invalid_rows))

    summary["guard_status"] = g_transition_result.get("status")
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
    return results, summary


def validate_checklist_g_new_transition_first_occurrence_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_G_NEW_TRANSITION_FIRST_OCCURRENCE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    g_rows = extract_markdown_table_rows(original_text, "G. 逐項收斂紀錄")
    b26_occurrence_count = sum(1 for cols in g_rows if len(cols) > 1 and cols[1].strip() == "B26")
    target_match_index = 1 if b26_occurrence_count >= 2 else None
    add_check(results, "meta_checklist", case_id, "target_g_row_has_nonfirst_occurrence_for_mutation", True, target_match_index is not None)
    if target_match_index is None:
        return results, summary

    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="G. 逐項收斂紀錄",
            row_id="B26",
            id_col_idx=1,
            update_cols=lambda cols: cols[:3] + ["NEW -> DONE"] + cols[4:],
            match_index=target_match_index,
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_g_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_g_new_transition_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    g_new_result = result_by_name.get("checklist_g_new_transition_only_on_first_occurrence", {})
    invalid_rows = g_new_result.get("invalid_new_transition_rows")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(g_new_result, "invalid_new_transition_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_new_transition_guard_fails", "FAIL", g_new_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_new_transition_reports_target_row", True, any(row.get("id") == "B26" for row in invalid_rows))

    summary["b26_occurrence_count"] = b26_occurrence_count
    summary["guard_status"] = g_new_result.get("status")
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
    summary["target_match_index"] = target_match_index
    return results, summary
def validate_checklist_first_nonempty_line_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_FIRST_NONEMPTY_LINE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    mutated_text = "<!-- synthetic mutation -->\n" + original_text

    with tempfile.TemporaryDirectory(prefix="meta_checklist_first_line_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    first_line_result = result_by_name.get("checklist_first_nonempty_line_matches_title", {})

    add_check(results, "meta_checklist", case_id, "mutated_first_nonempty_line_guard_fails", "FAIL", first_line_result.get("status"))
    match_flag = _read_summary_value(first_line_result, "matches")

    add_check(results, "meta_checklist", case_id, "mutated_first_nonempty_line_reports_false_match", False, match_flag)

    summary["guard_status"] = first_line_result.get("status")
    summary["match_flag"] = match_flag
    return results, summary


def validate_checklist_g_note_validate_reference_exists_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_G_NOTE_VALIDATE_REFERENCE_EXISTS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="G. 逐項收斂紀錄",
            row_id="T138",
            id_col_idx=1,
            update_cols=lambda cols: cols[:4] + ["`validate_nonexistent_retired_case`"],
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_g_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_g_validate_note_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    g_note_validate_result = result_by_name.get("checklist_g_note_validate_entries_exist", {})
    invalid_rows = g_note_validate_result.get("invalid_note_validate_rows")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(g_note_validate_result, "invalid_note_validate_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_note_validate_ref_guard_fails", "FAIL", g_note_validate_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_note_validate_ref_reports_target_row", True, any(row.get("id") == "T138" for row in invalid_rows))

    summary["guard_status"] = g_note_validate_result.get("status")
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
    return results, summary


def validate_checklist_g_ordering_case(_base_params):
    import tools.local_regression.run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_G_ORDERING"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="G. 逐項收斂紀錄",
            row_id="T01",
            id_col_idx=1,
            update_cols=lambda cols: ["2026-04-05"] + cols[1:],
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_g_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_g_order_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    g_order_result = result_by_name.get("checklist_g_rows_sorted_by_date_then_id", {})
    invalid_rows = g_order_result.get("invalid_order_rows")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(g_order_result, "invalid_order_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_order_guard_fails", "FAIL", g_order_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_order_reports_invalid_pair", True, bool(invalid_rows))

    summary["guard_status"] = g_order_result.get("status")
    summary["invalid_order_rows"] = invalid_rows
    return results, summary



def validate_checklist_summary_tables_sorted_by_id_case(_base_params):
    from tools.local_regression import run_meta_quality as meta_quality_module

    case_id = "META_CHECKLIST_SUMMARY_TABLE_ORDER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    mutated_text = _swap_markdown_table_rows(
        original_text,
        heading="T. 目前所有 `DONE` 的建議測試項目摘要",
        row_id_a="T01",
        row_id_b="T02",
        id_col_idx=0,
    )

    with tempfile.TemporaryDirectory(prefix="meta_checklist_summary_order_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    order_result = result_by_name.get("checklist_summary_tables_sorted_by_id", {})
    invalid_rows = _read_summary_value(order_result, "invalid_summary_table_orders", [])

    add_check(results, "meta_checklist", case_id, "mutated_summary_table_order_guard_fails", "FAIL", order_result.get("status"))
    add_check(
        results,
        "meta_checklist",
        case_id,
        "mutated_summary_table_order_reports_t_table",
        True,
        any(row.get("table") == "T" for row in invalid_rows),
    )

    summary["guard_status"] = order_result.get("status")
    summary["invalid_summary_table_orders"] = invalid_rows
    return results, summary




def validate_synthetic_case_numpy_alias_import_contract_case(_base_params):
    case_id = "META_SYNTHETIC_CASE_NUMPY_ALIAS_IMPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    modules_using_numpy_alias = []
    missing_numpy_alias_import = []
    for source_path in sorted(SYNTHETIC_VALIDATE_DIR.glob("synthetic*_cases.py")):
        source_text = source_path.read_text(encoding="utf-8")
        parsed = ast.parse(source_text, filename=str(source_path))
        if not _parsed_module_uses_numpy_alias(parsed):
            continue
        modules_using_numpy_alias.append(source_path.name)
        if not _parsed_module_declares_numpy_alias_import(parsed):
            missing_numpy_alias_import.append(source_path.name)

    add_check(results, "meta_contract", case_id, "synthetic_case_numpy_alias_usage_detected", True, bool(modules_using_numpy_alias))
    add_check(results, "meta_contract", case_id, "synthetic_case_numpy_alias_imports_declared", [], missing_numpy_alias_import)

    summary["modules_using_numpy_alias"] = modules_using_numpy_alias
    summary["missing_numpy_alias_import"] = missing_numpy_alias_import
    return results, summary


def validate_synthetic_case_numpy_alias_scan_ignores_string_literals_contract_case(_base_params):
    case_id = "META_SYNTHETIC_CASE_NUMPY_ALIAS_SCAN_IGNORES_STRING_LITERALS_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    string_only_source = """
message = "np. should stay inside string literal"
def describe():
    return message
"""
    actual_usage_source = """
import numpy as np

def make_array():
    return np.array([1, 2, 3])
"""

    parsed_string_only = ast.parse(string_only_source, filename="string_only_source.py")
    parsed_actual_usage = ast.parse(actual_usage_source, filename="actual_usage_source.py")

    add_check(
        results,
        "meta_contract",
        case_id,
        "numpy_alias_scan_ignores_string_literal_only_source",
        False,
        _parsed_module_uses_numpy_alias(parsed_string_only),
    )
    add_check(
        results,
        "meta_contract",
        case_id,
        "numpy_alias_scan_detects_actual_ast_usage",
        True,
        _parsed_module_uses_numpy_alias(parsed_actual_usage),
    )
    add_check(
        results,
        "meta_contract",
        case_id,
        "numpy_alias_scan_detects_numpy_alias_import",
        True,
        _parsed_module_declares_numpy_alias_import(parsed_actual_usage),
    )

    summary["string_literal_only_detected_as_usage"] = _parsed_module_uses_numpy_alias(parsed_string_only)
    summary["actual_ast_usage_detected"] = _parsed_module_uses_numpy_alias(parsed_actual_usage)
    return results, summary

def validate_synthetic_case_non_error_initial_capital_contract_case(_base_params):
    case_id = "SYNTHETIC_CASE_NON_ERROR_INITIAL_CAPITAL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_paths = sorted(build_project_absolute_path("tools", "validate").glob("synthetic_*cases.py"))
    invalid_assignments = []
    scanned_modules = []
    for source_path in source_paths:
        if source_path.name == "synthetic_error_cases.py":
            continue
        scanned_modules.append(source_path.name)
        parsed = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for hit in _find_nonpositive_initial_capital_assignments(parsed):
            invalid_assignments.append(f"{source_path.name}:{hit['lineno']}={hit['value']}")

    add_check(results, "meta_contract", case_id, "non_error_synthetic_cases_forbid_nonpositive_initial_capital_literals", [], invalid_assignments)
    summary["scanned_modules"] = scanned_modules
    summary["invalid_assignments"] = invalid_assignments
    return results, summary


def validate_synthetic_meta_cases_build_project_absolute_path_import_contract_case(_base_params):
    case_id = "META_SYNTHETIC_META_CASES_BUILD_PROJECT_ABSOLUTE_PATH_IMPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = SYNTHETIC_VALIDATE_DIR / "synthetic_meta_cases.py"
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    uses_helper_symbol = any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "build_project_absolute_path"
        for node in ast.walk(parsed)
    )
    has_explicit_import = _parsed_module_declares_specific_from_import(
        parsed,
        module_name="module_loader",
        imported_name="build_project_absolute_path",
    )

    add_check(
        results,
        "meta_contract",
        case_id,
        "synthetic_meta_cases_uses_build_project_absolute_path_symbol",
        True,
        uses_helper_symbol,
    )
    add_check(
        results,
        "meta_contract",
        case_id,
        "synthetic_meta_cases_imports_build_project_absolute_path",
        True,
        has_explicit_import,
    )

    summary["source_file"] = source_path.name
    summary["uses_build_project_absolute_path"] = uses_helper_symbol
    summary["has_explicit_import"] = has_explicit_import
    return results, summary


def validate_synthetic_case_normalize_chart_payload_literal_x_contract_case(_base_params):
    case_id = "META_SYNTHETIC_CASE_NORMALIZE_CHART_PAYLOAD_LITERAL_X_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = SYNTHETIC_VALIDATE_DIR / "synthetic_contract_cases.py"
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    invalid_calls = []
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "normalize_chart_payload_contract":
            continue
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        key_names = []
        for key_node in node.args[0].keys:
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key_names.append(key_node.value)
        if "x" not in key_names:
            invalid_calls.append(f"{source_path.name}:{node.lineno}")

    add_check(
        results,
        "meta_contract",
        case_id,
        "normalize_chart_payload_literal_contracts_require_x_field",
        [],
        invalid_calls,
    )

    summary["source_file"] = source_path.name
    summary["invalid_calls"] = invalid_calls
    return results, summary


def validate_synthetic_cases_import_target_resolution_contract_case(_base_params):
    case_id = "META_SYNTHETIC_CASES_IMPORT_TARGET_RESOLUTION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    contract = summarize_synthetic_cases_import_target_resolution_contract(PROJECT_ROOT)
    invalid_imports = contract["invalid_imports"]

    add_check(
        results,
        "meta_contract",
        case_id,
        "synthetic_cases_import_targets_resolve_to_declaring_module",
        [],
        invalid_imports,
    )

    summary["source_file"] = Path(contract["source_path"]).name
    summary["checked_module_imports"] = contract["checked_module_imports"]
    summary["invalid_imports"] = invalid_imports
    return results, summary


def validate_quick_gate_synthetic_registry_import_targets_contract_case(_base_params):
    case_id = "META_QUICK_GATE_SYNTHETIC_REGISTRY_IMPORT_TARGETS_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    quick_gate_path = PROJECT_ROOT / "tools" / "local_regression" / "run_quick_gate.py"
    source_text = quick_gate_path.read_text(encoding="utf-8")
    has_static_check_name = '"synthetic_registry_import_targets"' in source_text
    uses_shared_helper = "summarize_synthetic_cases_import_target_resolution_contract" in source_text

    add_check(
        results,
        "meta_contract",
        case_id,
        "quick_gate_registers_synthetic_registry_import_targets_static_check",
        True,
        has_static_check_name,
    )
    add_check(
        results,
        "meta_contract",
        case_id,
        "quick_gate_reuses_shared_import_target_resolution_helper",
        True,
        uses_shared_helper,
    )

    summary["source_file"] = quick_gate_path.name
    summary["has_static_check_name"] = has_static_check_name
    summary["uses_shared_helper"] = uses_shared_helper
    return results, summary


def validate_debug_backtest_history_snapshot_patch_seam_contract_case(_base_params):
    case_id = "META_DEBUG_BACKTEST_HISTORY_SNAPSHOT_PATCH_SEAM_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "trade_analysis", "backtest.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    top_level_names = set()
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level_names.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                top_level_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top_level_names.add(alias.asname or alias.name.split(".")[-1])

    add_check(
        results,
        "meta_contract",
        case_id,
        "debug_backtest_exports_history_snapshot_patch_seam",
        True,
        "_build_pit_history_snapshot" in top_level_names,
    )
    add_check(
        results,
        "meta_contract",
        case_id,
        "debug_backtest_run_debug_analysis_uses_history_snapshot_patch_seam",
        True,
        "signal_history_snapshot = _build_pit_history_snapshot(" in source_text
        and "latest_history_snapshot = _build_pit_history_snapshot(" in source_text,
    )

    summary["source_file"] = source_path.name
    return results, summary


def validate_synthetic_case_chart_navigation_binder_import_contract_case(_base_params):
    case_id = "META_SYNTHETIC_CASE_CHART_NAV_BINDER_IMPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = SYNTHETIC_VALIDATE_DIR / "synthetic_contract_cases.py"
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    uses_binder_symbol = any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "bind_matplotlib_chart_navigation"
        for node in ast.walk(parsed)
    )
    has_explicit_import = _parsed_module_declares_specific_from_import(
        parsed,
        module_name="tools.trade_analysis.charting",
        imported_name="bind_matplotlib_chart_navigation",
    )

    add_check(
        results,
        "meta_contract",
        case_id,
        "synthetic_contract_cases_uses_chart_navigation_binder_symbol",
        True,
        uses_binder_symbol,
    )
    add_check(
        results,
        "meta_contract",
        case_id,
        "synthetic_contract_cases_imports_chart_navigation_binder",
        True,
        has_explicit_import,
    )

    summary["source_file"] = source_path.name
    summary["uses_binder_symbol"] = uses_binder_symbol
    summary["has_explicit_import"] = has_explicit_import
    return results, summary


def validate_gui_buy_signal_annotation_helper_import_contract_case(_base_params):
    case_id = "META_GUI_BUY_SIGNAL_ANNOTATION_HELPER_IMPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = SYNTHETIC_VALIDATE_DIR / "synthetic_contract_cases.py"
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    uses_helper_symbol = any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "_record_buy_signal_annotation"
        for node in ast.walk(parsed)
    )
    has_explicit_import = _parsed_module_declares_specific_from_import(
        parsed,
        module_name="tools.trade_analysis.backtest",
        imported_name="_record_buy_signal_annotation",
    )

    add_check(
        results,
        "meta_contract",
        case_id,
        "synthetic_contract_cases_uses_buy_signal_annotation_helper_symbol",
        True,
        uses_helper_symbol,
    )
    add_check(
        results,
        "meta_contract",
        case_id,
        "synthetic_contract_cases_imports_buy_signal_annotation_helper",
        True,
        has_explicit_import,
    )

    summary["source_file"] = source_path.name
    summary["uses_helper_symbol"] = uses_helper_symbol
    summary["has_explicit_import"] = has_explicit_import
    return results, summary


def validate_synthetic_meta_cases_summary_value_accessor_contract_case(_base_params):
    case_id = "META_SUMMARY_VALUE_ACCESSOR_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_text = Path(__file__).read_text(encoding="utf-8")
    helper_defined = "def _read_summary_value(" in source_text
    direct_extra_access_patterns = []
    for quote in ('"', "'"):
        pattern = f".get({quote}extra{quote}, {{}}).get("
        if pattern in source_text:
            direct_extra_access_patterns.append(pattern)

    add_check(results, "meta_contract", case_id, "summary_value_accessor_helper_defined", True, helper_defined)
    add_check(results, "meta_contract", case_id, "summary_value_accessor_no_direct_extra_get_chain", [], direct_extra_access_patterns)

    summary["helper_defined"] = helper_defined
    summary["direct_extra_access_patterns"] = direct_extra_access_patterns
    return results, summary

def validate_registry_checklist_entry_consistency_case(_base_params):
    case_id = "META_REGISTRY_CHECKLIST_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validator_entries = load_synthetic_registry_entries_from_source(PROJECT_ROOT)
    validator_names = [entry["name"] for entry in validator_entries]
    validator_name_set = set(validator_names)
    imported_validate_names = load_imported_validate_names_from_synthetic_main_entry(PROJECT_ROOT)
    defined_validate_names = load_defined_validate_names_from_synthetic_case_modules(PROJECT_ROOT)
    convergence_statuses = _load_convergence_latest_statuses()
    done_test_rows = _load_done_test_rows()
    done_b_rows = _load_done_b_rows()
    main_statuses = _load_main_table_statuses()

    add_check(results, "meta_registry", case_id, "validator_registry_not_empty", True, len(validator_entries) > 0)
    add_check(results, "meta_registry", case_id, "validator_registry_names_unique", len(validator_names), len(validator_name_set))

    missing_imported_names = sorted(imported_validate_names - validator_name_set)
    extra_registry_names = sorted(validator_name_set - imported_validate_names)
    missing_defined_names = sorted(defined_validate_names - validator_name_set)
    orphan_registry_names = sorted(validator_name_set - defined_validate_names)

    add_check(results, "meta_registry", case_id, "imported_validate_cases_all_registered", [], missing_imported_names)
    add_check(results, "meta_registry", case_id, "registry_has_no_unimported_validate_case_names", [], extra_registry_names)
    add_check(results, "meta_registry", case_id, "defined_validate_cases_all_registered", [], missing_defined_names)
    add_check(results, "meta_registry", case_id, "registry_has_no_orphan_validate_case_names", [], orphan_registry_names)

    done_test_names = [row["name"] for row in done_test_rows]
    done_test_name_set = set(done_test_names)
    add_check(results, "meta_registry", case_id, "done_test_names_unique", len(done_test_names), len(done_test_name_set))

    formal_step_commands = [spec.command for spec in FORMAL_STEP_SPECS]
    missing_formal_step_commands = sorted(command for command in formal_step_commands if command not in done_test_name_set)
    add_check(
        results,
        "meta_registry",
        case_id,
        "all_formal_step_commands_listed_in_done_t_summary",
        [],
        missing_formal_step_commands,
    )

    done_test_validate_name_set = {
        row["name"].split()[0]
        for row in done_test_rows
        if row["name"].split() and row["name"].split()[0].startswith("validate_")
    }
    missing_done_test_validator_names = sorted(validator_name_set - done_test_validate_name_set)
    add_check(
        results,
        "meta_registry",
        case_id,
        "all_registered_validate_cases_listed_in_done_f_summary",
        [],
        missing_done_test_validator_names,
    )

    for row in done_test_rows:
        test_name = row["name"]
        first_token = test_name.split()[0] if test_name.split() else test_name
        if first_token.startswith("validate_"):
            add_check(
                results,
                "meta_registry",
                case_id,
                f"{row['id']}_registered_in_main_entry",
                True,
                first_token in validator_name_set,
            )
        elif test_name.startswith("run_") or ".py" in test_name:
            declared_script = first_token
            if declared_script == "run_meta_quality.py":
                declared_script = "tools/local_regression/run_meta_quality.py"
            add_check(
                results,
                "meta_registry",
                case_id,
                f"{row['id']}_declared_non_synthetic_entry_exists",
                True,
                (PROJECT_ROOT / declared_script).exists(),
            )
        else:
            add_check(
                results,
                "meta_registry",
                case_id,
                f"{row['id']}_entry_name_recognized",
                True,
                False,
            )

        add_check(
            results,
            "meta_registry",
            case_id,
            f"{row['id']}_done_summary_matches_convergence_status",
            "DONE",
            convergence_statuses.get(row["id"], ""),
        )

    done_b_ids = [row["b_id"] for row in done_b_rows]
    mapped_b_ids = {row["b_id"] for row in done_test_rows}
    for row in done_b_rows:
        add_check(
            results,
            "meta_registry",
            case_id,
            f"{row['b_id']}_done_summary_status_matches_main_table",
            "DONE",
            main_statuses.get(row["b_id"]),
        )
        add_check(
            results,
            "meta_registry",
            case_id,
            f"{row['b_id']}_done_summary_matches_convergence_status",
            "DONE",
            convergence_statuses.get(row["b_id"], "DONE"),
        )
        if row["entry"] == "既有 synthetic case":
            continue
        add_check(
            results,
            "meta_registry",
            case_id,
            f"{row['b_id']}_done_summary_has_done_test_mapping",
            True,
            row["b_id"] in mapped_b_ids,
        )
        entry_candidates = re.findall(r"(?:apps|core|tools)/[A-Za-z0-9_./-]+\.py", row["entry"])
        if entry_candidates:
            entry_path = entry_candidates[0]
            add_check(
                results,
                "meta_registry",
                case_id,
                f"{row['b_id']}_declared_entry_file_exists",
                True,
                (PROJECT_ROOT / entry_path).exists(),
            )

    summary["done_test_count"] = len(done_test_rows)
    summary["done_b_count"] = len(done_b_ids)
    summary["validator_count"] = len(validator_entries)
    summary["missing_imported_names"] = missing_imported_names
    summary["missing_defined_names"] = missing_defined_names
    summary["missing_done_test_validator_names"] = missing_done_test_validator_names
    return results, summary



def _count_failures(results):
    return sum(1 for row in results if row.get("status") == "FAIL")

def _summary_result_by_name(results, name):
    for row in results:
        if row.get("name") == name:
            return row
    raise AssertionError(f"result not found: {name}")


def _build_meta_quality_reuse_payload(*, line_percent=70.0, branch_percent=65.0, critical_line_percent=35.0, critical_branch_percent=30.0):
    files = {}
    for rel_path in COVERAGE_TARGETS:
        summary = {
            "covered_lines": 8,
            "num_statements": 10,
            "percent_covered": 80.0,
            "covered_branches": 4,
            "num_branches": 5,
        }
        if rel_path in CRITICAL_COVERAGE_TARGETS:
            summary = {
                "covered_lines": int(critical_line_percent),
                "num_statements": 100,
                "percent_covered": float(critical_line_percent),
                "covered_branches": int(critical_branch_percent),
                "num_branches": 100,
            }
        files[rel_path] = {"summary": summary}
    return {
        "totals": {
            "covered_lines": int(line_percent),
            "num_statements": 100,
            "covered_branches": int(branch_percent),
            "num_branches": 100,
            "percent_covered": float((line_percent + branch_percent) / 2.0),
        },
        "files": files,
    }



def validate_core_trading_modules_in_coverage_targets_case(_base_params):
    case_id = "META_CORE_TRADING_MODULES_IN_COVERAGE_TARGETS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    expected_targets = list(CORE_TRADING_COVERAGE_TARGETS)
    declared_targets = list(COVERAGE_TARGETS)
    missing_targets = sorted(path for path in expected_targets if path not in declared_targets)
    missing_files = sorted(path for path in expected_targets if not (PROJECT_ROOT / path).is_file())

    module_export_expectations = {
        "core.portfolio_ops": {"execute_reserved_entries_for_day", "settle_portfolio_positions", "closeout_open_positions"},
        "core.trade_plans": {"build_normal_candidate_plan", "execute_pre_market_entry_plan", "evaluate_history_candidate_metrics"},
    }
    module_import_failures = []
    module_export_failures = []
    reloaded_modules = []
    for module_name, expected_exports in module_export_expectations.items():
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
        except Exception as exc:
            module_import_failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        exported_names = set(getattr(module, "__all__", []))
        missing_exports = sorted(expected_exports - exported_names)
        if missing_exports:
            module_export_failures.append(f"{module_name}: {missing_exports}")
        reloaded_modules.append(module_name)

    add_check(results, "meta_coverage", case_id, "core_trading_coverage_targets_exist", [], missing_files)
    add_check(results, "meta_coverage", case_id, "core_trading_coverage_targets_declared", [], missing_targets)
    add_check(results, "meta_coverage", case_id, "core_wrapper_modules_importable_for_coverage_probe", [], module_import_failures)
    add_check(results, "meta_coverage", case_id, "core_wrapper_modules_export_expected_symbols", [], module_export_failures)

    summary["expected_target_count"] = len(expected_targets)
    summary["missing_targets"] = missing_targets
    summary["reloaded_modules"] = reloaded_modules
    return results, summary





def validate_policy_contract_modules_in_coverage_targets_case(_base_params):
    case_id = "META_POLICY_CONTRACT_MODULES_IN_COVERAGE_TARGETS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    expected_targets = list(POLICY_CONTRACT_COVERAGE_TARGETS)
    declared_targets = list(COVERAGE_TARGETS)
    missing_targets = sorted(path for path in expected_targets if path not in declared_targets)
    missing_files = sorted(path for path in expected_targets if not (PROJECT_ROOT / path).is_file())

    module_symbol_expectations = {
        "core.capital_policy": {
            "resolve_single_backtest_sizing_capital",
            "resolve_portfolio_sizing_equity",
            "resolve_portfolio_entry_budget",
            "resolve_scanner_live_capital",
        },
        "core.strategy_params": {
            "V16StrategyParams",
            "validate_strategy_param_ranges",
            "normalize_runtime_param_value",
        },
        "core.params_io": {
            "build_params_from_mapping",
            "load_params_from_json",
            "params_to_json_dict",
        },
        "config.execution_policy": {
            "EXECUTION_POLICY_PARAM_SPECS",
            "RUNTIME_PARAM_SPECS",
            "build_execution_policy_snapshot",
        },
        "config.training_policy": {
            "SELECTION_POLICY_PARAM_SPECS",
            "build_training_threshold_snapshot",
            "build_training_score_policy_snapshot",
        },
    }
    module_import_failures = []
    module_symbol_failures = []
    reloaded_modules = []
    for module_name, expected_symbols in module_symbol_expectations.items():
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
        except Exception as exc:
            module_import_failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        missing_symbols = sorted(symbol for symbol in expected_symbols if not hasattr(module, symbol))
        if missing_symbols:
            module_symbol_failures.append(f"{module_name}: {missing_symbols}")
        reloaded_modules.append(module_name)

    add_check(results, "meta_coverage", case_id, "policy_contract_coverage_targets_exist", [], missing_files)
    add_check(results, "meta_coverage", case_id, "policy_contract_coverage_targets_declared", [], missing_targets)
    add_check(results, "meta_coverage", case_id, "policy_contract_modules_importable_for_coverage_probe", [], module_import_failures)
    add_check(results, "meta_coverage", case_id, "policy_contract_modules_expose_expected_symbols", [], module_symbol_failures)

    summary["expected_target_count"] = len(expected_targets)
    summary["missing_targets"] = missing_targets
    summary["reloaded_modules"] = reloaded_modules
    return results, summary


def validate_peak_traced_memory_tracker_context_management_case(_base_params):
    case_id = "META_PEAK_TRACED_MEMORY_TRACKER_CONTEXT_MANAGEMENT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    tracked_files = [
        PROJECT_ROOT / "tools/local_regression/run_chain_checks.py",
        PROJECT_ROOT / "tools/local_regression/run_meta_quality.py",
        PROJECT_ROOT / "tools/local_regression/run_ml_smoke.py",
        PROJECT_ROOT / "tools/local_regression/run_quick_gate.py",
        PROJECT_ROOT / "tools/validate/main.py",
    ]

    manual_lifecycle_files = []
    missing_with_context_files = []
    syntax_errors = []
    scanned_files = []
    for path in tracked_files:
        rel_path = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        scanned_files.append(rel_path)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(f"{rel_path}: {exc.msg}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"__enter__", "__exit__"}:
                manual_lifecycle_files.append(rel_path)
                break

        if "with PeakTracedMemoryTracker() as tracker:" not in source:
            missing_with_context_files.append(rel_path)

    add_check(results, "meta_contract", case_id, "peak_traced_memory_tracker_files_parse", [], syntax_errors)
    add_check(results, "meta_contract", case_id, "peak_traced_memory_tracker_manual_lifecycle_forbidden", [], sorted(set(manual_lifecycle_files)))
    add_check(results, "meta_contract", case_id, "peak_traced_memory_tracker_uses_with_context", [], sorted(set(missing_with_context_files)))

    summary["tracked_files"] = scanned_files
    return results, summary


def validate_synthetic_contract_cases_no_legacy_price_df_case_key_contract_case(_base_params):
    case_id = "META_SYNTHETIC_CONTRACT_CASES_NO_LEGACY_PRICE_DF_CASE_KEY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "validate", "synthetic_contract_cases.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    legacy_hits = []
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "case":
            continue
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and slice_node.value == "price_df":
            legacy_hits.append(f"L{node.lineno}")

    add_check(results, "meta_contract", case_id, "synthetic_contract_cases_no_legacy_case_price_df_access", [], legacy_hits)

    summary["legacy_hits"] = legacy_hits
    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_gui_trade_count_contract_no_legacy_exit_snippet_case(_base_params):
    case_id = "META_GUI_TRADE_COUNT_CONTRACT_NO_LEGACY_EXIT_SNIPPET"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "validate", "synthetic_contract_cases.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    function_node = None
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "validate_gui_trade_count_and_sidebar_sync_contract_case":
            function_node = node
            break

    if function_node is None:
        function_source = ""
        legacy_literals = ["missing_validate_gui_trade_count_and_sidebar_sync_contract_case"]
    else:
        function_source = "\n".join(source_text.splitlines()[function_node.lineno - 1:function_node.end_lineno])
        legacy_literals = []
        for legacy_literal in (
            "'trade_count': _resolve_completed_trade_count(history_snapshot, include_current_round_trip=True)",
            '"history_snapshot=latest_history_snapshot"',
            "history_snapshot=latest_history_snapshot",
        ):
            if legacy_literal in function_source:
                legacy_literals.append(legacy_literal)

    add_check(results, "meta_contract", case_id, "gui_trade_count_contract_has_no_legacy_exit_snippet_literals", [], legacy_literals)
    add_check(results, "meta_contract", case_id, "gui_trade_count_contract_uses_forced_close_behavior_probe", True, "append_debug_forced_closeout(" in function_source and "build_trade_stats_index(" in function_source)

    summary["legacy_literals"] = legacy_literals
    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_single_backtest_stats_legacy_schema_contract_case(_base_params):
    case_id = "META_SINGLE_BACKTEST_STATS_LEGACY_SCHEMA_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("core", "backtest_finalize.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    function_node = None
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_backtest_stats":
            function_node = node
            break

    required_keys = {
        "trade_count",
        "win_rate",
        "expected_value",
        "asset_growth",
        "max_drawdown",
        "missed_buys",
        "missed_sells",
        "is_setup_today",
        "buy_limit",
        "stop_loss",
        "extended_candidate_today",
        "extended_orderable_today",
        "current_position",
        "payoff_ratio",
        "score",
    }

    if function_node is None:
        declared_keys = set()
        missing_keys = sorted(required_keys)
    else:
        declared_keys = set()
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Dict):
                continue
            for key_node in node.keys:
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    declared_keys.add(key_node.value)
        missing_keys = sorted(required_keys - declared_keys)

    add_check(results, "meta_contract", case_id, "single_backtest_stats_declares_legacy_schema_keys", [], missing_keys)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_signature_accepts_final_date", True, "final_date=None" in source_text)

    backtest_core_path = build_project_absolute_path("core", "backtest_core.py")
    backtest_core_source = backtest_core_path.read_text(encoding="utf-8")
    add_check(results, "meta_contract", case_id, "single_backtest_stats_empty_path_threads_final_date_none", True, "final_date=None" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_final_path_threads_dates_last", True, "final_date=Dates[-1]" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_extended_candidate_uses_threaded_final_date", True, "trade_date=final_date" in source_text)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_preview_resolves_security_profile", True, 'resolved_security_profile = (active_extended_signal or {}).get("security_profile")' in source_text)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_stop_preview_uses_shared_initial_stop_helper", True, "stop_loss = calc_initial_stop_from_reference(close_last, atr_last, params, ticker=resolved_ticker, security_profile=resolved_security_profile)" in source_text)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_tp_preview_uses_shared_target_helper", True, "tp_price = calc_frozen_target_price(close_last, stop_loss, ticker=resolved_ticker, security_profile=resolved_security_profile)" in source_text)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_has_no_manual_tp_preview_formula", False, "tp_price = close_last + (close_last - (close_last - atr_last * params.atr_times_init))" in source_text)

    summary["missing_keys"] = missing_keys
    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary





def validate_debug_backtest_entry_cash_path_contract_case(_base_params):
    case_id = "META_DEBUG_BACKTEST_ENTRY_CASH_PATH_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    def _get_function_source(rel_path, func_name):
        source_path = build_project_absolute_path(*rel_path.split('/'))
        source_text = source_path.read_text(encoding="utf-8")
        parsed = ast.parse(source_text, filename=str(source_path))
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
                return source_path, func_source
        return source_path, ""

    debug_backtest_path, debug_backtest_source = _get_function_source("tools/trade_analysis/backtest.py", "run_debug_analysis")
    debug_entry_path, debug_entry_source = _get_function_source("tools/trade_analysis/entry_flow.py", "process_debug_entry_for_day")

    add_check(results, "meta_contract", case_id, "debug_backtest_entry_flow_returns_spent_cash", True, "position, active_extended_signal, spent_cash = process_debug_entry_for_day(" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_entry_cash_subtracts_spent_cash", True, "current_capital -= spent_cash" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_passes_ticker_to_entry_flow", True, "ticker=ticker" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_signature_accepts_ticker", True, "ticker=None" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_signature_accepts_security_profile", True, "security_profile=None" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_passes_trade_date_to_entry_flow", True, "trade_date=dates[j]" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_signature_accepts_trade_date", True, "trade_date=None" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_threads_ticker_to_normal_signal_state", True, "create_signal_tracking_state(buy_limit_prev, atr_prev, params, ticker=ticker, security_profile=security_profile)" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_threads_ticker_to_normal_entry_plan", True, "build_normal_entry_plan(buy_limit_prev, atr_prev, sizing_cap, params, ticker=ticker, security_profile=security_profile, trade_date=effective_trade_date)" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_threads_ticker_to_extended_entry_plan", True, "build_extended_entry_plan_from_signal(" in debug_entry_source and "ticker=ticker" in debug_entry_source and "security_profile=security_profile" in debug_entry_source and "trade_date=effective_trade_date" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_uses_exact_entry_total_helper", True, "spent_cash = _resolve_display_entry_total(entry_result, qty=entry_plan['qty'], params=params)" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_returns_spent_cash", True, "return position, active_extended_signal, spent_cash" in debug_entry_source)
    summary["source_paths"] = [
        str(debug_backtest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(debug_entry_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    ]
    return results, summary




def validate_display_money_rounding_helper_contract_case(_base_params):
    case_id = "META_DISPLAY_MONEY_ROUNDING_HELPER_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    exact_path = build_project_absolute_path("core", "exact_accounting.py")
    portfolio_path = build_project_absolute_path("core", "portfolio_exits.py")
    log_rows_path = build_project_absolute_path("tools", "trade_analysis", "log_rows.py")

    exact_text = exact_path.read_text(encoding="utf-8")
    portfolio_text = portfolio_path.read_text(encoding="utf-8")
    log_rows_text = log_rows_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "shared_display_rounding_uses_decimal_half_up", True, 'quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)' in exact_text)
    add_check(results, "meta_contract", case_id, "portfolio_history_rounding_delegates_to_shared_helper", True, 'return round_money_for_display(value)' in portfolio_text)
    add_check(results, "meta_contract", case_id, "debug_log_rounding_delegates_to_shared_helper", True, 'return round_money_for_display(value)' in log_rows_text)
    add_check(results, "meta_contract", case_id, "portfolio_history_rounding_has_no_legacy_builtin_round", False, 'return round(float(value), 2)' in portfolio_text)
    add_check(results, "meta_contract", case_id, "debug_log_rounding_has_no_legacy_builtin_round", False, 'return round(float(value), 2)' in log_rows_text)

    summary["source_paths"] = [
        str(exact_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(portfolio_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(log_rows_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    ]
    return results, summary


def validate_real_case_completed_trade_rounding_oracle_contract_case(_base_params):
    case_id = "META_REAL_CASE_COMPLETED_TRADE_ROUNDING_ORACLE_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "validate", "real_case_assertions.py")
    source_text = source_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "real_case_assertions_imports_shared_rounding_helper", True, 'from core.exact_accounting import round_money_for_display' in source_text)
    add_check(results, "meta_contract", case_id, "real_case_assertions_rounds_expected_trade_pnls_with_shared_helper", True, 'expected_trade_pnls = [round_money_for_display(log["pnl"]) for log in standalone_logs]' in source_text)
    add_check(results, "meta_contract", case_id, "real_case_assertions_rounds_expected_realized_sum_with_shared_helper", True, 'expected_realized_pnl_sum = round_money_for_display(sum(expected_trade_pnls))' in source_text)
    add_check(results, "meta_contract", case_id, "real_case_assertions_has_no_legacy_builtin_round_for_trade_pnls", False, 'expected_trade_pnls = [round(float(log["pnl"]), 2) for log in standalone_logs]' in source_text)
    add_check(results, "meta_contract", case_id, "real_case_assertions_has_no_legacy_builtin_round_for_realized_sum", False, 'expected_realized_pnl_sum = round(sum(expected_trade_pnls), 2)' in source_text)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary

def validate_trade_rebuild_rounding_helper_contract_case(_base_params):
    case_id = "META_TRADE_REBUILD_ROUNDING_HELPER_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "validate", "trade_rebuild.py")
    source_text = source_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "trade_rebuild_imports_shared_rounding_helper", True, 'from core.exact_accounting import round_money_for_display' in source_text)
    add_check(results, "meta_contract", case_id, "trade_rebuild_normalizes_row_pnl_with_shared_helper", True, 'realized_pnl = round_money_for_display(getattr(row, pnl_col))' in source_text)
    add_check(results, "meta_contract", case_id, "trade_rebuild_half_exit_accumulates_with_shared_helper", True, 'active_trade["total_pnl"] = round_money_for_display(active_trade["total_pnl"] + realized_pnl)' in source_text)
    add_check(results, "meta_contract", case_id, "trade_rebuild_full_exit_accumulates_with_shared_helper", True, 'active_trade["total_pnl"] = round_money_for_display(active_trade["total_pnl"] + realized_pnl)' in source_text)
    add_check(results, "meta_contract", case_id, "trade_rebuild_has_no_legacy_builtin_round_total_pnl", False, 'active_trade["total_pnl"] = round(active_trade["total_pnl"] + realized_pnl, 2)' in source_text)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_single_backtest_exact_cash_path_contract_case(_base_params):
    case_id = "META_SINGLE_BACKTEST_EXACT_CASH_PATH_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    def _get_function_source(rel_path, func_name):
        source_path = build_project_absolute_path(*rel_path.split('/'))
        source_text = source_path.read_text(encoding="utf-8")
        parsed = ast.parse(source_text, filename=str(source_path))
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
                return source_path, func_source
        return source_path, ""

    core_path, backtest_core_source = _get_function_source("core/backtest_core.py", "run_v16_backtest")
    finalize_path, finalize_source = _get_function_source("core/backtest_finalize.py", "finalize_open_position_at_end")
    debug_backtest_path, debug_backtest_source = _get_function_source("tools/trade_analysis/backtest.py", "run_debug_analysis")
    debug_exit_path, debug_exit_source = _get_function_source("tools/trade_analysis/exit_flow.py", "process_debug_position_step")
    forced_close_path, forced_close_source = _get_function_source("tools/trade_analysis/exit_flow.py", "append_debug_forced_closeout")

    add_check(results, "meta_contract", case_id, "single_backtest_exit_cash_adds_freed_cash", True, "currentCapital_milli += freed_cash_milli" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_exit_cash_has_no_legacy_pnl_addition", False, "currentCapital_milli += pnl_realized_milli" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_mark_to_market_equity_uses_net_liquidation_value", True, "currentEquity_milli = currentCapital_milli + floating_sell_ledger['net_sell_total_milli']" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_mark_to_market_equity_has_no_legacy_floating_pnl_path", False, "currentEquity_milli = currentCapital_milli + floating_pnl_milli" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_closeout_cash_adds_net_sell_total", True, "current_capital_milli += sell_ledger['net_sell_total_milli']" in finalize_source)
    add_check(results, "meta_contract", case_id, "single_backtest_closeout_cash_has_no_legacy_pnl_addition", False, "current_capital_milli += pnl_milli" in finalize_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_exit_cash_adds_freed_cash", True, "current_capital += freed_cash" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_exit_snapshot_uses_freed_cash", True, "current_capital_after_exit = None if current_capital_before_event is None else float(current_capital_before_event) + float(freed_cash)" in debug_exit_source)
    add_check(results, "meta_contract", case_id, "debug_forced_closeout_snapshot_uses_net_sell_total", True, "current_capital_after_exit = None if current_capital_before_event is None else float(current_capital_before_event) + milli_to_money(sell_ledger['net_sell_total_milli'])" in forced_close_source)

    summary["source_paths"] = [
        str(core_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(finalize_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(debug_backtest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(debug_exit_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(forced_close_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    ]
    return results, summary



def validate_debug_forced_closeout_exact_total_pnl_contract_case(_base_params):
    case_id = "META_DEBUG_FORCED_CLOSEOUT_EXACT_TOTAL_PNL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    forced_close_path = build_project_absolute_path("tools", "trade_analysis", "exit_flow.py")
    source_text = forced_close_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(forced_close_path))
    function_source = ""
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "append_debug_forced_closeout":
            function_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
            break

    add_check(results, "meta_contract", case_id, "forced_closeout_total_pnl_uses_integer_ledger_path", True, "total_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0) + int(final_leg_actual_pnl_milli)" in function_source)
    add_check(results, "meta_contract", case_id, "forced_closeout_total_pnl_derives_display_from_milli", True, "total_pnl = milli_to_money(total_pnl_milli)" in function_source)
    add_check(results, "meta_contract", case_id, "forced_closeout_has_no_legacy_float_total_pnl_path", False, "total_pnl = float(position.get('realized_pnl', 0.0) + final_leg_actual_pnl)" in function_source)

    summary["source_path"] = forced_close_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary

def validate_unit_display_rounding_helper_contract_case(_base_params):
    case_id = "META_UNIT_DISPLAY_ROUNDING_HELPER_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "validate", "synthetic_unit_cases.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))
    function_source = ""
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "validate_exact_accounting_display_leg_reconciliation_case":
            function_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
            break

    add_check(results, "meta_contract", case_id, "unit_display_reconciliation_uses_shared_rounding_helper", True, 'round_money_for_display(position["display_realized_pnl_sum"] + reconciled_exit_pnl)' in function_source)
    add_check(results, "meta_contract", case_id, "unit_display_reconciliation_has_no_legacy_builtin_round", False, 'round(position["display_realized_pnl_sum"] + reconciled_exit_pnl, 2)' in function_source)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary



def validate_debug_entry_display_capital_uses_exact_total_contract_case(_base_params):
    case_id = "META_DEBUG_ENTRY_DISPLAY_CAPITAL_USES_EXACT_TOTAL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "trade_analysis", "entry_flow.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    helper_source = ""
    step_source = ""
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_display_entry_total":
            helper_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.FunctionDef) and node.name == "process_debug_entry_for_day":
            step_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])

    add_check(results, "meta_contract", case_id, "debug_entry_has_exact_entry_total_helper", True, bool(helper_source))
    add_check(results, "meta_contract", case_id, "debug_entry_helper_prefers_position_net_buy_total_milli", True, 'exact_entry_total_milli = int(position.get("net_buy_total_milli", 0) or 0)' in helper_source)
    add_check(results, "meta_contract", case_id, "debug_entry_helper_uses_entry_cost_before_any_recompute", True, 'display_entry_cost = float(entry_result.get("entry_cost", 0.0) or 0.0)' in helper_source)
    add_check(results, "meta_contract", case_id, "debug_entry_helper_final_fallback_uses_exact_total_helper", True, 'return calc_entry_total_cost(buy_price, int(qty or 0), params)' in helper_source)
    add_check(results, "meta_contract", case_id, "debug_entry_process_uses_exact_entry_total_helper", True, "spent_cash = _resolve_display_entry_total(entry_result, qty=entry_plan['qty'], params=params)" in step_source)
    add_check(results, "meta_contract", case_id, "debug_entry_has_no_legacy_per_share_entry_cost_fallback", False, "entry_result.get('entry_cost', entry_result['entry_price'] * entry_plan['qty'])" in step_source)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary

def validate_debug_half_exit_leg_return_pct_uses_allocated_cost_contract_case(_base_params):
    case_id = "META_DEBUG_HALF_EXIT_LEG_RETURN_PCT_ALLOCATED_COST_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "trade_analysis", "exit_flow.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))
    helper_source = ""
    step_source = ""
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_display_leg_return_pct":
            helper_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.FunctionDef) and node.name == "process_debug_position_step":
            step_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])

    add_check(results, "meta_contract", case_id, "debug_half_exit_has_exact_leg_return_helper", True, bool(helper_source))
    add_check(results, "meta_contract", case_id, "debug_half_exit_leg_return_prefers_allocated_cost_milli", True, "allocated_cost_milli = 0 if exit_context is None else int(exit_context.get('allocated_cost_milli', 0) or 0)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_leg_return_uses_pnl_milli", True, "pnl_milli = 0 if exit_context is None else int(exit_context.get('pnl_milli', 0) or 0)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_leg_return_converts_allocated_cost_from_exact_ledger", True, "return float(pnl_milli * 100.0 / allocated_cost_milli)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_leg_return_has_no_legacy_money_ratio_path", False, "return float(milli_to_money(pnl_milli) / milli_to_money(allocated_cost_milli) * 100.0)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_marker_uses_exact_leg_return_helper", True, "'pnl_pct': _resolve_display_leg_return_pct(" in step_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_has_no_legacy_per_share_return_pct_formula", False, "(sell_net_price_half - float(position.get('entry', exec_sell_price_half))) / float(position.get('entry', exec_sell_price_half)) * 100.0" in step_source)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary

def validate_debug_exit_display_capital_uses_ledger_totals_contract_case(_base_params):
    case_id = "META_DEBUG_EXIT_DISPLAY_CAPITAL_LEDGER_TOTALS_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "trade_analysis", "exit_flow.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))
    helper_source = ""
    step_source = ""
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_full_entry_capital_milli":
            helper_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.FunctionDef) and node.name == "process_debug_position_step":
            step_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])

    add_check(results, "meta_contract", case_id, "debug_exit_total_return_prefers_exact_entry_total", True, "exact_entry_total_milli = int(position.get('net_buy_total_milli', 0) or 0)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_gross_amount_uses_exact_sell_total", True, "gross_amount=tp_sell_total" in step_source)
    add_check(results, "meta_contract", case_id, "debug_full_exit_gross_amount_uses_exact_sell_total", True, "gross_amount=sell_total_amount" in step_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_marker_sell_capital_uses_exact_sell_total", True, "'sell_capital': float(tp_sell_total)" in step_source)
    add_check(results, "meta_contract", case_id, "debug_full_exit_marker_sell_capital_uses_exact_sell_total", True, "'sell_capital': float(sell_total_amount)" in step_source)
    add_check(results, "meta_contract", case_id, "debug_exit_has_no_legacy_half_exit_per_share_times_qty", False, "gross_amount=sell_net_price_half * sold_qty" in step_source)
    add_check(results, "meta_contract", case_id, "debug_exit_has_no_legacy_full_exit_per_share_times_qty", False, "gross_amount=sell_net_price * final_exit_qty" in step_source)
    add_check(results, "meta_contract", case_id, "debug_exit_has_no_legacy_full_exit_marker_per_share_times_qty", False, "'sell_capital': float(sell_net_price * final_exit_qty)" in step_source)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary



def validate_debug_exit_entry_capital_fallback_contract_case(_base_params):
    case_id = "META_DEBUG_EXIT_ENTRY_CAPITAL_FALLBACK_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "trade_analysis", "exit_flow.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))
    entry_helper_source = ""
    sell_helper_source = ""
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_full_entry_capital_milli":
            entry_helper_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_display_sell_total_milli":
            sell_helper_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])

    step_source = ""
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "process_debug_position_step":
            step_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
            break

    compact_step_source = re.sub(r"\s+", "", step_source)

    add_check(results, "meta_contract", case_id, "debug_exit_entry_capital_prefers_display_total_before_exact_total_fallback", True, "display_entry_capital = float(position.get('entry_capital_total', 0.0) or 0.0)" in entry_helper_source)
    add_check(results, "meta_contract", case_id, "debug_exit_entry_capital_coerces_display_total_with_shared_helper", True, "return coerce_money_like_to_milli(round_money_for_display(display_entry_capital))" in entry_helper_source)
    add_check(results, "meta_contract", case_id, "debug_exit_entry_capital_final_fallback_uses_average_price_total_helper", True, "return calc_total_from_average_price_milli(entry_price, initial_qty)" in entry_helper_source)
    add_check(results, "meta_contract", case_id, "debug_exit_display_sell_total_signature_accepts_position_and_current_date", True, "def _resolve_display_sell_total_milli(exit_context, *, position, current_date, sell_price, qty, params):" in source_text)
    add_check(results, "meta_contract", case_id, "debug_exit_display_sell_total_helper_uses_explicit_context", True, "ticker=position.get('ticker')" in sell_helper_source and "trade_date=current_date" in sell_helper_source)
    add_check(results, "meta_contract", case_id, "debug_exit_tp_marker_threads_position_and_current_date", True, "tp_sell_total=_resolve_display_sell_total(tp_context,position=position,current_date=current_date," in compact_step_source)
    add_check(results, "meta_contract", case_id, "debug_exit_full_sell_threads_position_and_current_date", True, "sell_total_amount=_resolve_display_sell_total(exit_context,position=position,current_date=current_date," in compact_step_source)
    add_check(results, "meta_contract", case_id, "debug_exit_leg_return_threads_current_date", True, "current_date=current_date," in step_source and "fallback_sell_price=exec_sell_price_half" in step_source)
    add_check(results, "meta_contract", case_id, "debug_exit_display_sell_total_has_no_free_position_reference", False, "ticker=position.get('ticker')" in sell_helper_source and "def _resolve_display_sell_total_milli(exit_context, *, sell_price, qty, params):" in source_text)
    add_check(results, "meta_contract", case_id, "debug_exit_entry_capital_has_no_legacy_gross_price_helper_on_net_average_entry", False, "return calc_entry_total_cost(entry_price, initial_qty, params)" in entry_helper_source)
    add_check(results, "meta_contract", case_id, "debug_exit_entry_capital_has_no_legacy_per_share_fallback", False, "return round_money_for_display(entry_price * initial_qty)" in entry_helper_source)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary

def validate_synthetic_meta_source_path_binding_contract_case(_base_params):
    case_id = "META_SYNTHETIC_META_SOURCE_PATH_BINDING_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "validate", "synthetic_meta_cases.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))

    offenders = []
    for node in parsed.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("validate_"):
            continue
        assigned_names = {arg.arg for arg in node.args.args}
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for target in inner.targets:
                    if isinstance(target, ast.Name):
                        assigned_names.add(target.id)
            elif isinstance(inner, ast.AnnAssign) and isinstance(inner.target, ast.Name):
                assigned_names.add(inner.target.id)
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Assign):
                continue
            if len(inner.targets) != 1 or not isinstance(inner.targets[0], ast.Subscript):
                continue
            sub = inner.targets[0]
            if not isinstance(sub.value, ast.Name) or sub.value.id != "summary":
                continue
            slice_node = sub.slice
            if not (isinstance(slice_node, ast.Constant) and slice_node.value == "source_path"):
                continue
            value = inner.value
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
                continue
            relative_call = value.func.value
            if not isinstance(relative_call, ast.Call) or not isinstance(relative_call.func, ast.Attribute):
                continue
            if relative_call.func.attr != "relative_to":
                continue
            base_obj = relative_call.func.value
            if isinstance(base_obj, ast.Name) and base_obj.id not in assigned_names:
                offenders.append(f"{node.name}:{base_obj.id}:L{inner.lineno}")

    add_check(results, "meta_contract", case_id, "synthetic_meta_source_path_assignments_bind_declared_locals", [], offenders)
    summary["offenders"] = offenders
    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary



def validate_debug_sell_signal_profit_pct_uses_exact_mark_to_market_contract_case(_base_params):
    case_id = "META_DEBUG_SELL_SIGNAL_PROFIT_PCT_EXACT_MARK_TO_MARKET_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "trade_analysis", "backtest.py")
    source_text = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_text, filename=str(source_path))
    helper_source = ""
    sell_signal_source = ""
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_sell_signal_profit_pct":
            helper_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.FunctionDef) and node.name == "_record_sell_signal_annotation":
            sell_signal_source = "\n".join(source_text.splitlines()[node.lineno - 1:node.end_lineno])

    add_check(results, "meta_contract", case_id, "debug_sell_signal_has_exact_profit_pct_helper", True, "def _resolve_sell_signal_profit_pct(position, signal_close, params, signal_date=None):" in source_text)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_prefers_net_buy_total_milli", True, "full_entry_total_milli = int(position.get('net_buy_total_milli', 0) or 0)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_final_fallback_uses_average_price_total_helper", True, "initial_qty = int(position.get('initial_qty', remaining_qty) or remaining_qty or 0)" in helper_source and "full_entry_total_milli = calc_total_from_average_price_milli(entry_price, initial_qty)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_has_no_legacy_gross_price_helper_on_net_average_entry", False, "full_entry_total_milli = coerce_money_like_to_milli(calc_entry_total_cost(entry_price, initial_qty, params))" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_uses_remaining_cost_basis_mark_to_market", True, "floating_pnl_milli = signal_sell_ledger['net_sell_total_milli'] - remaining_cost_basis_milli" in helper_source and "total_trade_pnl_milli = realized_pnl_milli + floating_pnl_milli" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_divides_integer_totals_directly", True, "return float(total_trade_pnl_milli * 100.0 / full_entry_total_milli)" in helper_source and "return float(signal_trade_pnl_milli * 100.0 / full_entry_total_milli)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_has_no_legacy_milli_to_money_ratio_path", False, "milli_to_money(total_trade_pnl_milli) / milli_to_money(full_entry_total_milli)" in helper_source or "milli_to_money(signal_trade_pnl_milli) / milli_to_money(full_entry_total_milli)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_records_helper_output", True, "signal_trade_pct = _resolve_sell_signal_profit_pct(position, signal_close, params, signal_date=signal_date)" in sell_signal_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_has_no_legacy_raw_close_minus_entry_formula", False, "((float(signal_close) - entry_price) / entry_price * 100.0)" in sell_signal_source or "((float(signal_close) - entry_price) / entry_price * 100.0)" in helper_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_has_no_legacy_per_share_entry_total_fallback", False, "coerce_money_like_to_milli(round_money_for_display(entry_price * remaining_qty))" in helper_source)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_debug_exact_fallback_helpers_contract_case(_base_params):
    case_id = "META_DEBUG_EXACT_FALLBACK_HELPERS_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    exit_path = build_project_absolute_path("tools", "trade_analysis", "exit_flow.py")
    exit_source = exit_path.read_text(encoding="utf-8")
    backtest_path = build_project_absolute_path("tools", "trade_analysis", "backtest.py")
    backtest_source = backtest_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "debug_exit_entry_capital_fallback_uses_average_price_total_helper", True, "return calc_total_from_average_price_milli(entry_price, initial_qty)" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_exit_entry_capital_has_no_legacy_gross_price_helper_on_net_average_entry", False, "return calc_entry_total_cost(entry_price, initial_qty, params)" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_leg_return_fallback_uses_average_price_total_helper", True, "entry_total_milli = calc_total_from_average_price_milli(entry_price, sold_qty)" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_leg_return_fallback_divides_integer_totals_directly", True, "return float((sell_total_milli - entry_total_milli) * 100.0 / entry_total_milli)" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_leg_return_has_no_legacy_money_ratio_path", False, "return float((sell_total - entry_total) / entry_total * 100.0)" in exit_source or "milli_to_money(pnl_milli) / milli_to_money(allocated_cost_milli)" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_half_exit_leg_return_fallback_has_no_legacy_per_share_formula", False, "return float((float(fallback_net_price) - entry_price) / entry_price * 100.0)" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_entry_total_fallback_uses_average_price_total_helper", True, "full_entry_total_milli = calc_total_from_average_price_milli(entry_price, initial_qty)" in backtest_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_entry_total_has_no_legacy_gross_price_helper_on_net_average_entry", False, "full_entry_total_milli = coerce_money_like_to_milli(calc_entry_total_cost(entry_price, initial_qty, params))" in backtest_source)

    summary["source_path"] = exit_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary



def validate_average_price_total_helper_contract_case(_base_params):
    case_id = "META_AVERAGE_PRICE_TOTAL_HELPER_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    exact_path = build_project_absolute_path("core", "exact_accounting.py")
    exact_source = exact_path.read_text(encoding="utf-8")
    price_path = build_project_absolute_path("core", "price_utils.py")
    price_source = price_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "exact_accounting_defines_average_price_total_milli_helper", True, "def calc_total_from_average_price_milli(avg_price, qty: int) -> int:" in exact_source)
    add_check(results, "meta_contract", case_id, "exact_accounting_defines_average_price_total_helper", True, "def calc_total_from_average_price(avg_price, qty: int) -> float:" in exact_source)
    add_check(results, "meta_contract", case_id, "initial_risk_total_uses_average_price_total_helper_for_entry", True, "entry_total_milli = calc_total_from_average_price_milli(entry_price, qty)" in price_source)
    add_check(results, "meta_contract", case_id, "initial_risk_total_uses_average_price_total_helper_for_stop", True, "stop_net_total_milli = calc_total_from_average_price_milli(net_stop_price, qty)" in price_source)
    add_check(results, "meta_contract", case_id, "initial_risk_total_has_no_legacy_entry_price_times_qty_float_path", False, "entry_total_milli = money_to_milli(entry_price * qty)" in price_source)
    add_check(results, "meta_contract", case_id, "initial_risk_total_has_no_legacy_stop_price_times_qty_float_path", False, "stop_net_total_milli = money_to_milli(net_stop_price * qty)" in price_source)

    summary["source_path"] = exact_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_price_utils_average_price_total_import_contract_case(_base_params):
    case_id = "META_PRICE_UTILS_AVERAGE_PRICE_TOTAL_IMPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    price_path = build_project_absolute_path("core", "price_utils.py")
    price_source = price_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "price_utils_imports_average_price_total_milli_helper", True, "calc_total_from_average_price_milli," in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_initial_risk_total_entry_uses_imported_average_price_total_helper", True, "entry_total_milli = calc_total_from_average_price_milli(entry_price, qty)" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_initial_risk_total_stop_uses_imported_average_price_total_helper", True, "stop_net_total_milli = calc_total_from_average_price_milli(net_stop_price, qty)" in price_source)

    summary["source_path"] = price_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_price_utils_array_tick_normalization_contract_case(_base_params):
    case_id = "META_PRICE_UTILS_ARRAY_TICK_NORMALIZATION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    price_path = build_project_absolute_path("core", "price_utils.py")
    price_source = price_path.read_text(encoding="utf-8")
    signal_path = build_project_absolute_path("core", "signal_utils.py")
    signal_source = signal_path.read_text(encoding="utf-8")
    backtest_path = build_project_absolute_path("core", "backtest_core.py")
    backtest_source = backtest_path.read_text(encoding="utf-8")
    entry_path = build_project_absolute_path("core", "portfolio_entries.py")
    entry_source = entry_path.read_text(encoding="utf-8")
    entry_plans_path = build_project_absolute_path("core", "entry_plans.py")
    entry_plans_source = entry_plans_path.read_text(encoding="utf-8")
    scanner_processor_path = build_project_absolute_path("tools", "scanner", "stock_processor.py")
    scanner_processor_source = scanner_processor_path.read_text(encoding="utf-8")
    position_step_path = build_project_absolute_path("core", "position_step.py")
    position_step_source = position_step_path.read_text(encoding="utf-8")
    portfolio_exits_path = build_project_absolute_path("core", "portfolio_exits.py")
    portfolio_exits_source = portfolio_exits_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "price_utils_imports_shared_security_profile_inference", True, "infer_security_profile" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_scalar_tick_size_uses_shared_raw_price_tick_helper", True, "milli_to_price(get_tick_milli_from_price(price, security_profile=resolved_profile))" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_array_tick_size_uses_shared_raw_price_tick_helper", True, "milli_to_price(get_tick_milli_from_price(price, security_profile=resolved_profile)) for price in prices[valid]" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_scalar_rounding_uses_shared_raw_price_tick_helper", True, "milli_to_price(round_price_to_tick_milli(price, direction=direction, security_profile=resolved_profile))" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_array_rounding_uses_shared_raw_price_tick_helper", True, "milli_to_price(round_price_to_tick_milli(price, direction=direction, security_profile=resolved_profile))" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_tick_lookup_resolves_security_profile_from_ticker", True, "resolved_profile = infer_security_profile(ticker) if security_profile is None else security_profile" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_tick_lookup_has_no_price_to_milli_prequantize_path", False, "get_tick_milli(price_to_milli(price))" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_rounding_has_no_price_to_milli_prequantize_path", False, "round_price_milli_to_tick(price_to_milli(price), direction=direction)" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_has_no_stock_only_global_tick_ladder_lookup", False, "MILLI_TICK_LADDER = STOCK_MILLI_TICK_LADDER" in price_source)
    add_check(results, "meta_contract", case_id, "signal_utils_buy_limit_array_routes_ticker_to_shared_tick_helper", True, "adjust_long_buy_limit_array(raw_buy_limits[valid_buy_mask], ticker=resolved_ticker)" in signal_source)
    add_check(results, "meta_contract", case_id, "backtest_core_generate_signals_routes_ticker", True, "generate_signals(df, params, ticker=resolved_ticker)" in backtest_source)
    add_check(results, "meta_contract", case_id, "portfolio_entry_seed_preserves_ticker_for_execution", True, "'ticker': candidate_row.get('ticker')" in entry_source)
    add_check(results, "meta_contract", case_id, "entry_plan_resize_threads_ticker_security_profile_and_trade_date", True, 'ticker=candidate_plan.get("ticker")' in entry_plans_source and 'security_profile=candidate_plan.get("security_profile")' in entry_plans_source and 'trade_date=candidate_plan.get("trade_date")' in entry_plans_source)
    add_check(results, "meta_contract", case_id, "scanner_projected_qty_threads_ticker_and_trade_date", True, "calc_reference_candidate_qty(stats['buy_limit'], stats['stop_loss'], params, ticker=ticker, trade_date=trade_date)" in scanner_processor_source and "calc_reference_candidate_qty(limit_price, init_sl, params, ticker=ticker, trade_date=trade_date)" in scanner_processor_source)
    add_check(results, "meta_contract", case_id, "scanner_response_threads_latest_trade_date", True, 'resolve_latest_trade_date_from_frame' in scanner_processor_source and 'trade_date = resolve_latest_trade_date_from_frame(df)' in scanner_processor_source and 'build_scanner_response_from_stats(ticker=ticker, stats=stats, params=params, sanitize_stats=sanitize_stats, trade_date=trade_date)' in scanner_processor_source)
    add_check(results, "meta_contract", case_id, "position_step_exit_path_uses_position_ticker", True, 'ticker=position.get("ticker")' in position_step_source)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_exit_path_uses_weakest_ticker", True, 'adjust_long_sell_fill_price(w_open, ticker=weakest_ticker)' in portfolio_exits_source)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_exit_path_has_no_undefined_ticker_reference", False, 'adjust_long_sell_fill_price(w_open, ticker=ticker)' in portfolio_exits_source)
    add_check(results, "meta_contract", case_id, "price_utils_position_size_routes_tax_schedule_by_security_profile", True, 'tax_ppm = resolve_sell_tax_ppm(params, ticker=ticker, security_profile=security_profile, trade_date=trade_date)' in price_source)
    add_check(results, "meta_contract", case_id, "backtest_core_floating_sell_ledger_threads_trade_date_and_profile", True, "trade_date=Dates[j]" in backtest_source and "security_profile=position.get('security_profile')" in backtest_source)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_exit_path_threads_trade_date_and_profile", True, 'trade_date=today' in portfolio_exits_source and "security_profile=pos.get('security_profile')" in portfolio_exits_source)
    add_check(results, "meta_contract", case_id, "price_utils_array_rounding_has_no_legacy_float_ratio_path", False, "ratios = valid_prices / ticks" in price_source)
    add_check(results, "meta_contract", case_id, "price_utils_array_rounding_has_no_legacy_numpy_ceil_floor_tick_path", False, "np.ceil(ratios - 1e-12) * ticks" in price_source or "np.floor(ratios + 1e-12) * ticks" in price_source or "np.floor(ratios + 0.5) * ticks" in price_source)

    summary["source_path"] = price_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_exact_ledger_return_ratio_no_money_float_division_contract_case(_base_params):
    case_id = "META_EXACT_LEDGER_RETURN_RATIO_NO_MONEY_FLOAT_DIVISION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    exit_path = build_project_absolute_path("tools", "trade_analysis", "exit_flow.py")
    exit_source = exit_path.read_text(encoding="utf-8")
    backtest_path = build_project_absolute_path("tools", "trade_analysis", "backtest.py")
    backtest_source = backtest_path.read_text(encoding="utf-8")
    portfolio_path = build_project_absolute_path("core", "portfolio_exits.py")
    portfolio_source = portfolio_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "debug_exit_total_return_pct_uses_integer_totals", True, "total_return_pct = float(total_pnl_milli * 100.0 / full_entry_capital_milli) if full_entry_capital_milli > 0 else 0.0" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_exit_leg_return_pct_uses_integer_totals", True, "return float(pnl_milli * 100.0 / allocated_cost_milli)" in exit_source and "return float((sell_total_milli - entry_total_milli) * 100.0 / entry_total_milli)" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_exit_has_no_legacy_money_ratio_division", False, "milli_to_money(pnl_milli) / milli_to_money(allocated_cost_milli)" in exit_source or "(sell_total - entry_total) / entry_total" in exit_source or "(total_pnl / full_entry_capital * 100.0)" in exit_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_uses_integer_totals", True, "return float(total_trade_pnl_milli * 100.0 / full_entry_total_milli)" in backtest_source and "return float(signal_trade_pnl_milli * 100.0 / full_entry_total_milli)" in backtest_source)
    add_check(results, "meta_contract", case_id, "debug_sell_signal_profit_pct_has_no_legacy_money_ratio_division", False, "milli_to_money(total_trade_pnl_milli) / milli_to_money(full_entry_total_milli)" in backtest_source or "milli_to_money(signal_trade_pnl_milli) / milli_to_money(full_entry_total_milli)" in backtest_source)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_mark_to_market_uses_integer_totals", True, "return total_trade_pnl_milli / full_entry_total_milli" in portfolio_source)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_mark_to_market_has_no_legacy_money_ratio_division", False, "milli_to_money(total_trade_pnl_milli) / milli_to_money(full_entry_total_milli)" in portfolio_source)

    summary["source_path"] = exit_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case(_base_params):
    case_id = "META_TEST_SUITE_SUMMARY_COMMENT_COVERS_LATEST_EXACT_CONTRACT_IDS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("apps", "test_suite.py")
    source_text = source_path.read_text(encoding="utf-8")
    summary_comment_line = next(
        (
            line.strip()
            for line in source_text.splitlines()
            if line.strip().startswith("# consistency step 透過 synthetic registry 覆蓋 debug exact-fallback / exit milli-binding helper contracts")
        ),
        "",
    )

    add_check(results, "meta_contract", case_id, "test_suite_summary_comment_block_present", True, bool(summary_comment_line))
    add_check(results, "meta_contract", case_id, "test_suite_summary_comment_lists_t225_through_t236", True, "T225/T226/T229/T230/T231/T232/T233/T234/T235/T236" in summary_comment_line)
    add_check(results, "meta_contract", case_id, "test_suite_summary_comment_explicitly_mentions_t234", True, "T234" in summary_comment_line)
    add_check(results, "meta_contract", case_id, "test_suite_summary_comment_explicitly_mentions_t235", True, "T235" in summary_comment_line)
    add_check(results, "meta_contract", case_id, "test_suite_summary_comment_explicitly_mentions_t236", True, "T236" in summary_comment_line)
    add_check(results, "meta_contract", case_id, "test_suite_summary_comment_has_no_stale_missing_t234_t235_t236_list", False, "T225/T226/T229/T230/T231/T232/T233）。" in summary_comment_line)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_core_r_multiple_exact_ledger_contract_case(_base_params):
    case_id = "META_CORE_R_MULTIPLE_EXACT_LEDGER_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    exact_path = build_project_absolute_path("core", "exact_accounting.py")
    exact_source = exact_path.read_text(encoding="utf-8")
    backtest_core_path = build_project_absolute_path("core", "backtest_core.py")
    backtest_core_source = backtest_core_path.read_text(encoding="utf-8")
    backtest_finalize_path = build_project_absolute_path("core", "backtest_finalize.py")
    backtest_finalize_source = backtest_finalize_path.read_text(encoding="utf-8")
    portfolio_exits_path = build_project_absolute_path("core", "portfolio_exits.py")
    portfolio_exits_source = portfolio_exits_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "exact_accounting_exposes_shared_milli_ratio_helper", True, "def calc_ratio_from_milli(numerator_milli: int, denominator_milli: int) -> float:" in exact_source)
    add_check(results, "meta_contract", case_id, "backtest_core_r_multiple_uses_exact_ledger_helper", True, "trade_r_mult = calc_ratio_from_milli(position['realized_pnl_milli'], position.get('initial_risk_total_milli', 0))" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "backtest_finalize_r_multiple_uses_exact_ledger_helper", True, "trade_r_mult = calc_ratio_from_milli(total_pnl_milli, position.get('initial_risk_total_milli', 0))" in backtest_finalize_source)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_r_multiple_uses_exact_ledger_helper", True, "total_r = calc_ratio_from_milli(total_pnl_milli, pos.get('initial_risk_total_milli', 0))" in portfolio_exits_source)
    add_check(results, "meta_contract", case_id, "portfolio_exit_r_multiple_uses_exact_ledger_helper", True, "total_r = calc_ratio_from_milli(pos.get('realized_pnl_milli', 0), pos.get('initial_risk_total_milli', 0))" in portfolio_exits_source)
    add_check(results, "meta_contract", case_id, "core_has_no_legacy_float_total_pnl_divided_by_initial_risk_total", False, "total_pnl / position['initial_risk_total']" in backtest_core_source or "total_pnl / position['initial_risk_total']" in backtest_finalize_source or "total_pnl / pos['initial_risk_total']" in portfolio_exits_source)

    summary["source_path"] = backtest_core_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_debug_exit_total_return_milli_binding_contract_case(_base_params):
    case_id = "META_DEBUG_EXIT_TOTAL_RETURN_MILLI_BINDING_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("tools", "trade_analysis", "exit_flow.py")
    source_text = source_path.read_text(encoding="utf-8")

    process_start = source_text.index("def process_debug_position_step(")
    forced_closeout_start = source_text.index("def append_debug_forced_closeout(")
    process_source = source_text[process_start:forced_closeout_start]
    forced_closeout_source = source_text[forced_closeout_start:]

    process_milli_binding = "total_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0)"
    forced_closeout_milli_binding = "total_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0) + int(final_leg_actual_pnl_milli)"
    total_return_stmt = "total_return_pct = float(total_pnl_milli * 100.0 / full_entry_capital_milli) if full_entry_capital_milli > 0 else 0.0"
    process_binding_index = process_source.find(process_milli_binding)
    process_total_return_index = process_source.find(total_return_stmt)
    forced_closeout_binding_index = forced_closeout_source.find(forced_closeout_milli_binding)
    forced_closeout_total_return_index = forced_closeout_source.find(total_return_stmt)

    add_check(results, "meta_contract", case_id, "debug_exit_total_return_pct_binds_total_pnl_milli_before_use", True, process_binding_index != -1 and process_total_return_index != -1 and process_binding_index < process_total_return_index)
    add_check(results, "meta_contract", case_id, "debug_exit_total_pnl_display_derives_from_milli_binding", True, "total_pnl = milli_to_money(total_pnl_milli)" in process_source)
    add_check(results, "meta_contract", case_id, "debug_exit_total_return_pct_has_no_unbound_total_pnl_milli_path", False, "total_pnl = float(position.get('realized_pnl', pnl_realized))" in process_source)
    add_check(results, "meta_contract", case_id, "debug_forced_closeout_total_return_pct_binds_total_pnl_milli_before_use", True, forced_closeout_binding_index != -1 and forced_closeout_total_return_index != -1 and forced_closeout_binding_index < forced_closeout_total_return_index)
    add_check(results, "meta_contract", case_id, "debug_forced_closeout_total_pnl_display_derives_from_milli_binding", True, "total_pnl = milli_to_money(total_pnl_milli)" in forced_closeout_source)
    add_check(results, "meta_contract", case_id, "debug_forced_closeout_total_return_pct_has_no_legacy_float_total_pnl_path", False, "total_pnl = float(position.get('realized_pnl', 0.0) + final_leg_actual_pnl)" in forced_closeout_source)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary

def validate_formal_step_entry_coverage_targets_case(_base_params):
    case_id = "META_FORMAL_STEP_ENTRY_COVERAGE_TARGETS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    expected_step_scripts = [spec.command.split()[0] for spec in FORMAL_STEP_SPECS]
    declared_entry_targets = list(FORMAL_STEP_ENTRY_COVERAGE_TARGETS)
    declared_targets = list(COVERAGE_TARGETS)
    missing_declared_entry_targets = sorted(path for path in declared_entry_targets if path not in declared_targets)
    missing_step_scripts = sorted(path for path in expected_step_scripts if path not in declared_targets)
    missing_entry_files = sorted(path for path in declared_entry_targets if not (PROJECT_ROOT / path).is_file())

    module_symbol_expectations = {
        "tools.local_regression.run_quick_gate": {"HELP_TARGETS", "main", "run_static_checks"},
        "tools.validate.cli": {"main"},
    }
    module_import_failures = []
    module_symbol_failures = []
    reloaded_modules = []
    for module_name, expected_symbols in module_symbol_expectations.items():
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
        except Exception as exc:
            module_import_failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        missing_symbols = sorted(symbol for symbol in expected_symbols if not hasattr(module, symbol))
        if missing_symbols:
            module_symbol_failures.append(f"{module_name}: {missing_symbols}")
        reloaded_modules.append(module_name)

    add_check(results, "meta_coverage", case_id, "formal_step_entry_coverage_targets_exist", [], missing_entry_files)
    add_check(results, "meta_coverage", case_id, "formal_step_entry_coverage_targets_declared", [], missing_declared_entry_targets)
    add_check(results, "meta_coverage", case_id, "formal_step_scripts_declared_in_overall_coverage_targets", [], missing_step_scripts)
    add_check(results, "meta_coverage", case_id, "formal_step_entry_modules_importable_for_coverage_probe", [], module_import_failures)
    add_check(results, "meta_coverage", case_id, "formal_step_entry_modules_expose_expected_symbols", [], module_symbol_failures)

    summary["expected_step_scripts"] = expected_step_scripts
    summary["declared_entry_targets"] = declared_entry_targets
    summary["missing_step_scripts"] = missing_step_scripts
    summary["reloaded_modules"] = reloaded_modules
    return results, summary


def validate_formal_step_implementation_coverage_targets_case(_base_params):
    case_id = "META_FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    expected_targets = list(FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS)
    declared_targets = list(COVERAGE_TARGETS)
    missing_targets = sorted(path for path in expected_targets if path not in declared_targets)
    missing_files = sorted(path for path in expected_targets if not (PROJECT_ROOT / path).is_file())

    module_symbol_expectations = {
        "tools.validate.main": {
            "main",
            "_run_synthetic_suite_with_optional_coverage",
            "resolve_validate_dataset_profile_key",
        },
    }
    module_import_failures = []
    module_symbol_failures = []
    reloaded_modules = []
    for module_name, expected_symbols in module_symbol_expectations.items():
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
        except Exception as exc:
            module_import_failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        missing_symbols = sorted(symbol for symbol in expected_symbols if not hasattr(module, symbol))
        if missing_symbols:
            module_symbol_failures.append(f"{module_name}: {missing_symbols}")
        reloaded_modules.append(module_name)

    add_check(results, "meta_coverage", case_id, "formal_step_implementation_coverage_targets_exist", [], missing_files)
    add_check(results, "meta_coverage", case_id, "formal_step_implementation_coverage_targets_declared", [], missing_targets)
    add_check(results, "meta_coverage", case_id, "formal_step_implementation_modules_importable_for_coverage_probe", [], module_import_failures)
    add_check(results, "meta_coverage", case_id, "formal_step_implementation_modules_expose_expected_symbols", [], module_symbol_failures)

    summary["expected_target_count"] = len(expected_targets)
    summary["missing_targets"] = missing_targets
    summary["reloaded_modules"] = reloaded_modules
    return results, summary


def validate_test_suite_orchestrator_coverage_targets_case(_base_params):
    case_id = "META_TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    expected_targets = list(TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS)
    declared_targets = list(COVERAGE_TARGETS)
    missing_targets = sorted(path for path in expected_targets if path not in declared_targets)
    missing_files = sorted(path for path in expected_targets if not (PROJECT_ROOT / path).is_file())

    module_symbol_expectations = {
        "tools.local_regression.common": {"ensure_reduced_dataset", "build_artifacts_manifest", "write_json"},
        "tools.local_regression.formal_pipeline": {"FORMAL_STEP_ORDER", "FORMAL_SINGLE_ENTRY"},
        "tools.local_regression.meta_quality_targets": {"COVERAGE_TARGETS", "CORE_TRADING_COVERAGE_TARGETS", "TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS"},
        "tools.local_regression.meta_quality_coverage": {"build_coverage_summary", "_coverage_threshold_policy_ok"},
        "tools.local_regression.run_meta_quality": {"COVERAGE_TARGETS", "TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS"},
        "tools.validate.preflight_env": {"REQUIREMENTS_PATH", "format_preflight_summary", "run_preflight"},
        "core.test_suite_reporting": {"print_test_suite_human_summary", "TEST_SUITE_STEP_LABELS"},
    }
    module_import_failures = []
    module_symbol_failures = []
    reloaded_modules = []
    for module_name, expected_symbols in module_symbol_expectations.items():
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
        except Exception as exc:
            module_import_failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        missing_symbols = sorted(symbol for symbol in expected_symbols if not hasattr(module, symbol))
        if missing_symbols:
            module_symbol_failures.append(f"{module_name}: {missing_symbols}")
        reloaded_modules.append(module_name)

    add_check(results, "meta_coverage", case_id, "test_suite_orchestrator_coverage_targets_exist", [], missing_files)
    add_check(results, "meta_coverage", case_id, "test_suite_orchestrator_coverage_targets_declared", [], missing_targets)
    add_check(results, "meta_coverage", case_id, "test_suite_orchestrator_modules_importable_for_coverage_probe", [], module_import_failures)
    add_check(results, "meta_coverage", case_id, "test_suite_orchestrator_modules_expose_expected_symbols", [], module_symbol_failures)

    summary["expected_target_count"] = len(expected_targets)
    summary["missing_targets"] = missing_targets
    summary["reloaded_modules"] = reloaded_modules
    return results, summary


def validate_critical_file_coverage_minimum_gate_case(_base_params):
    case_id = "META_CRITICAL_FILE_COVERAGE_MINIMUM_GATE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="meta_critical_cov_") as temp_dir:
        run_dir = Path(temp_dir)
        coverage_dir = run_dir / "coverage_artifacts"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        payload = _build_meta_quality_reuse_payload(line_percent=72.0, branch_percent=68.0, critical_line_percent=35.0, critical_branch_percent=30.0)
        payload["files"][CRITICAL_COVERAGE_TARGETS[0]]["summary"]["covered_lines"] = 10
        payload["files"][CRITICAL_COVERAGE_TARGETS[0]]["summary"]["percent_covered"] = 10.0
        payload["files"][CRITICAL_COVERAGE_TARGETS[1]]["summary"]["covered_branches"] = 5
        payload["files"][CRITICAL_COVERAGE_TARGETS[1]]["summary"]["num_branches"] = 100
        write_path = coverage_dir / "coverage_synthetic.json"
        write_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (coverage_dir / "coverage_run_info.json").write_text(json.dumps({
            "source": "validate_consistency",
            "returncode": 0,
            "stdout": "cached",
            "stderr": "",
            "timed_out": False,
            "synthetic_fail_count": 0,
            "synthetic_case_count": 99,
            "json_generated": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "coverage_line_min_percent": 55.0,
            "coverage_branch_min_percent": 50.0,
            "coverage_critical_line_min_percent": 30.0,
            "coverage_critical_branch_min_percent": 25.0,
        }
        coverage_summary = build_coverage_summary(run_dir, manifest)

    line_gate_result = _summary_result_by_name(coverage_summary["results"], "coverage_critical_files_line_percent_within_minimum")
    branch_gate_result = _summary_result_by_name(coverage_summary["results"], "coverage_critical_files_branch_percent_within_minimum")
    line_fail_targets = coverage_summary.get("critical_under_line_targets", [])
    branch_fail_targets = coverage_summary.get("critical_under_branch_targets", [])
    add_check(results, "meta_coverage", case_id, "critical_file_line_gate_detects_undercovered_file", True, CRITICAL_COVERAGE_TARGETS[0] in line_fail_targets and line_gate_result.get("status") == "FAIL")
    add_check(results, "meta_coverage", case_id, "critical_file_branch_gate_detects_undercovered_file", True, CRITICAL_COVERAGE_TARGETS[1] in branch_fail_targets and branch_gate_result.get("status") == "FAIL")
    add_check(results, "meta_coverage", case_id, "critical_file_gate_blocks_overall_pass", False, coverage_summary.get("ok"))

    summary["line_fail_targets"] = line_fail_targets
    summary["branch_fail_targets"] = branch_fail_targets
    return results, summary


def validate_coverage_threshold_floor_case(_base_params):
    import tools.local_regression.common as common_module

    case_id = "META_COVERAGE_THRESHOLD_FLOOR"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    loaded_manifest = common_module.load_manifest()
    expected_line_floor = int(COVERAGE_LINE_MIN_FLOOR)
    expected_branch_floor = int(COVERAGE_BRANCH_MIN_FLOOR)

    add_check(results, "meta_coverage", case_id, "manifest_line_floor_respects_formal_baseline", True, int(loaded_manifest["coverage_line_min_percent"]) >= expected_line_floor)
    add_check(results, "meta_coverage", case_id, "manifest_branch_floor_respects_formal_baseline", True, int(loaded_manifest["coverage_branch_min_percent"]) >= expected_branch_floor)
    add_check(results, "meta_coverage", case_id, "manifest_branch_floor_priority_gap_valid", True, float(loaded_manifest["coverage_line_min_percent"]) - float(loaded_manifest["coverage_branch_min_percent"]) <= float(COVERAGE_MAX_LINE_BRANCH_GAP))

    with tempfile.TemporaryDirectory(prefix="meta_cov_floor_") as temp_dir:
        run_dir = Path(temp_dir)
        coverage_dir = run_dir / "coverage_artifacts"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        payload = _build_meta_quality_reuse_payload()
        (coverage_dir / "coverage_synthetic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (coverage_dir / "coverage_run_info.json").write_text(json.dumps({
            "source": "validate_consistency",
            "returncode": 0,
            "stdout": "cached",
            "stderr": "",
            "timed_out": False,
            "synthetic_fail_count": 0,
            "synthetic_case_count": 99,
            "json_generated": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        failing_manifest = {
            "coverage_line_min_percent": 50.0,
            "coverage_branch_min_percent": 45.0,
            "coverage_critical_line_min_percent": 30.0,
            "coverage_critical_branch_min_percent": 25.0,
        }
        coverage_summary = build_coverage_summary(run_dir, failing_manifest)

    threshold_policy_result = _summary_result_by_name(coverage_summary["results"], "coverage_thresholds_respect_formal_floor")
    add_check(results, "meta_coverage", case_id, "coverage_threshold_floor_blocks_regression", False, coverage_summary.get("ok"))
    add_check(results, "meta_coverage", case_id, "coverage_threshold_floor_detects_below_baseline_manifest", "FAIL", threshold_policy_result.get("status"))

    summary["threshold_policy_status"] = threshold_policy_result.get("status")
    return results, summary


def validate_critical_coverage_threshold_floor_case(_base_params):
    import tools.local_regression.common as common_module

    case_id = "META_CRITICAL_COVERAGE_THRESHOLD_FLOOR"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    loaded_manifest = common_module.load_manifest()
    expected_line_floor = int(CRITICAL_COVERAGE_LINE_MIN_FLOOR)
    expected_branch_floor = int(CRITICAL_COVERAGE_BRANCH_MIN_FLOOR)

    add_check(results, "meta_coverage", case_id, "manifest_critical_line_floor_respects_formal_baseline", True, int(loaded_manifest["coverage_critical_line_min_percent"]) >= expected_line_floor)
    add_check(results, "meta_coverage", case_id, "manifest_critical_branch_floor_respects_formal_baseline", True, int(loaded_manifest["coverage_critical_branch_min_percent"]) >= expected_branch_floor)
    add_check(results, "meta_coverage", case_id, "manifest_critical_branch_floor_priority_gap_valid", True, float(loaded_manifest["coverage_critical_line_min_percent"]) - float(loaded_manifest["coverage_critical_branch_min_percent"]) <= float(COVERAGE_MAX_LINE_BRANCH_GAP))

    with tempfile.TemporaryDirectory(prefix="meta_critical_cov_floor_") as temp_dir:
        run_dir = Path(temp_dir)
        coverage_dir = run_dir / "coverage_artifacts"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        payload = _build_meta_quality_reuse_payload(line_percent=72.0, branch_percent=68.0, critical_line_percent=35.0, critical_branch_percent=30.0)
        (coverage_dir / "coverage_synthetic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (coverage_dir / "coverage_run_info.json").write_text(json.dumps({
            "source": "validate_consistency",
            "returncode": 0,
            "stdout": "cached",
            "stderr": "",
            "timed_out": False,
            "synthetic_fail_count": 0,
            "synthetic_case_count": 100,
            "json_generated": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        failing_manifest = {
            "coverage_line_min_percent": 55.0,
            "coverage_branch_min_percent": 50.0,
            "coverage_critical_line_min_percent": 25.0,
            "coverage_critical_branch_min_percent": 20.0,
        }
        coverage_summary = build_coverage_summary(run_dir, failing_manifest)

    threshold_policy_result = _summary_result_by_name(coverage_summary["results"], "coverage_critical_thresholds_respect_formal_floor")
    add_check(results, "meta_coverage", case_id, "critical_coverage_threshold_floor_blocks_regression", False, coverage_summary.get("ok"))
    add_check(results, "meta_coverage", case_id, "critical_coverage_threshold_floor_detects_below_baseline_manifest", "FAIL", threshold_policy_result.get("status"))

    summary["threshold_policy_status"] = threshold_policy_result.get("status")
    return results, summary


def validate_entry_path_critical_coverage_gate_case(_base_params):
    case_id = "META_ENTRY_PATH_CRITICAL_COVERAGE_GATE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    expected_targets = list(ENTRY_PATH_CRITICAL_COVERAGE_TARGETS)
    declared_targets = list(CRITICAL_COVERAGE_TARGETS)
    missing_targets = sorted(path for path in expected_targets if path not in declared_targets)
    missing_files = sorted(path for path in expected_targets if not (PROJECT_ROOT / path).is_file())

    module_expectations = {
        "core.portfolio_entries": {"execute_reserved_entries_for_day", "cleanup_extended_signals_for_day"},
        "core.entry_plans": {"build_cash_capped_entry_plan", "execute_pre_market_entry_plan", "should_count_miss_buy"},
    }
    module_import_failures = []
    module_symbol_failures = []
    reloaded_modules = []
    for module_name, expected_symbols in module_expectations.items():
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
        except Exception as exc:
            module_import_failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        missing_symbols = sorted(symbol for symbol in expected_symbols if not hasattr(module, symbol))
        if missing_symbols:
            module_symbol_failures.append(f"{module_name}: {missing_symbols}")
        reloaded_modules.append(module_name)

    add_check(results, "meta_coverage", case_id, "entry_path_critical_coverage_targets_exist", [], missing_files)
    add_check(results, "meta_coverage", case_id, "entry_path_critical_coverage_targets_declared", [], missing_targets)
    add_check(results, "meta_coverage", case_id, "entry_path_modules_importable_for_coverage_probe", [], module_import_failures)
    add_check(results, "meta_coverage", case_id, "entry_path_modules_expose_expected_symbols", [], module_symbol_failures)

    summary["expected_target_count"] = len(expected_targets)
    summary["missing_targets"] = missing_targets
    summary["reloaded_modules"] = reloaded_modules
    return results, summary


def validate_known_bad_fault_injection_case(base_params):
    case_id = "META_KNOWN_BAD_FAULT_INJECTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.validate.synthetic_flow_cases import validate_synthetic_same_day_buy_sell_forbidden_case
    from tools.validate.synthetic_history_cases import validate_synthetic_portfolio_history_filter_only_case
    from tools.validate.synthetic_take_profit_cases import (
        validate_synthetic_fee_tax_net_equity_case,
        validate_synthetic_same_bar_stop_priority_case,
    )

    def broken_same_day_sell_stats(*args, **kwargs):
        stats = {
            "trade_count": 1,
            "portfolio_missed_buy_rows": 0,
            "total_missed": 0,
            "df_trades": __import__("pandas").DataFrame(
                [
                    {"Date": "2024-02-26", "Type": "買進(一般)"},
                    {"Date": "2024-02-26", "Type": "全倉結算(停損)"},
                ]
            ),
        }
        return stats

    with patch('tools.validate.synthetic_flow_cases.run_portfolio_core_check_for_dir', side_effect=broken_same_day_sell_stats), \
         patch('tools.validate.synthetic_flow_cases.run_portfolio_sim_tool_check_for_dir', side_effect=broken_same_day_sell_stats), \
         patch('tools.validate.synthetic_flow_cases.add_portfolio_stats_equality_checks', lambda *args, **kwargs: None):
        fault_results, _summary = validate_synthetic_same_day_buy_sell_forbidden_case(base_params)
        add_check(results, 'meta_fault_injection', case_id, 'same_day_sell_fault_detected', True, _count_failures(fault_results) > 0)

    def broken_execute_bar_step(*args, **kwargs):
        position = dict(args[0])
        position['qty'] = 5
        position['sold_half'] = True
        position['realized_pnl'] = 25.0
        return position, 500.0, 25.0, ['TP_HALF']

    with patch('tools.validate.synthetic_take_profit_cases.execute_bar_step', side_effect=broken_execute_bar_step):
        fault_results, _summary = validate_synthetic_same_bar_stop_priority_case(base_params)
        add_check(results, 'meta_fault_injection', case_id, 'same_bar_stop_priority_fault_detected', True, _count_failures(fault_results) > 0)

    def broken_run_portfolio_timeline(*args, **kwargs):
        module = __import__('tools.validate.synthetic_take_profit_cases', fromlist=['run_portfolio_timeline'])
        original = broken_run_portfolio_timeline._original
        timeline = list(original(*args, **kwargs))
        df_equity = timeline[0].copy()
        df_trades = timeline[1].copy()
        if not df_equity.empty:
            df_equity.loc[df_equity.index[-1], 'Equity'] = float(df_equity.iloc[-1]['Equity']) + 50.0
        if not df_trades.empty:
            mask = df_trades['Type'].fillna('').isin(['全倉結算(停損)', '全倉結算(指標)'])
            if mask.any():
                idx = df_trades[mask].index[0]
                df_trades.loc[idx, '該筆總損益'] = float(df_trades.loc[idx, '該筆總損益']) + 50.0
        timeline[0] = df_equity
        timeline[1] = df_trades
        timeline[2] = float(timeline[2]) + 0.5
        timeline[8] = float(timeline[8]) + 50.0
        return tuple(timeline)

    import tools.validate.synthetic_take_profit_cases as _tp_module
    broken_run_portfolio_timeline._original = _tp_module.run_portfolio_timeline
    with patch('tools.validate.synthetic_take_profit_cases.run_portfolio_timeline', side_effect=broken_run_portfolio_timeline):
        fault_results, _summary = validate_synthetic_fee_tax_net_equity_case(base_params)
        add_check(results, 'meta_fault_injection', case_id, 'fee_tax_fault_detected', True, _count_failures(fault_results) > 0)

    def broken_run_v16_backtest(*args, **kwargs):
        return {
            'trade_count': 0,
            'is_candidate': False,
            'current_position': 0,
            'asset_growth': 0.0,
        }

    with patch('tools.validate.synthetic_history_cases.run_v16_backtest', side_effect=broken_run_v16_backtest):
        fault_results, _summary = validate_synthetic_portfolio_history_filter_only_case(base_params)
        add_check(results, 'meta_fault_injection', case_id, 'history_filter_misuse_fault_detected', True, _count_failures(fault_results) > 0)

    summary['fault_injections_checked'] = 4
    return results, summary


def validate_portfolio_rotation_mark_to_market_return_contract_case(_base_params):
    case_id = "META_PORTFOLIO_ROTATION_MARK_TO_MARKET_RETURN"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = PROJECT_ROOT / "core" / "portfolio_exits.py"
    source_text = source_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "portfolio_rotation_has_mark_to_market_helper", True, "def _calc_position_mark_to_market_return(" in source_text)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_uses_mark_to_market_helper", True, "ret = _calc_position_mark_to_market_return(pos, pt_y_close, params, trade_date=today)" in source_text)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_has_no_legacy_raw_close_minus_entry_formula", False, "ret = (pt_y_close - pos['entry']) / pos['entry']" in source_text)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_same_bar_stop_priority_oracle_snapshots_pre_exit_cost_basis_contract_case(_base_params):
    case_id = "META_SAME_BAR_STOP_PRIORITY_ORACLE_SNAPSHOT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = PROJECT_ROOT / "tools" / "validate" / "synthetic_take_profit_cases.py"
    source_text = source_path.read_text(encoding="utf-8")

    add_check(results, "meta_contract", case_id, "same_bar_stop_priority_snapshots_original_cost_basis_before_execute", True, 'original_cost_basis_milli = int(position["remaining_cost_basis_milli"])' in source_text)
    add_check(results, "meta_contract", case_id, "same_bar_stop_priority_expected_pnl_uses_original_cost_basis_snapshot", True, 'expected_pnl = milli_to_money(expected_sell_ledger["net_sell_total_milli"] - original_cost_basis_milli)' in source_text)
    add_check(results, "meta_contract", case_id, "same_bar_stop_priority_has_no_mutated_remaining_cost_basis_oracle", False, 'expected_pnl = milli_to_money(expected_sell_ledger["net_sell_total_milli"] - position["remaining_cost_basis_milli"])' in source_text)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary


def validate_validator_oracles_use_exact_ledger_totals_contract_case(_base_params):
    case_id = "META_VALIDATOR_ORACLES_USE_EXACT_LEDGER_TOTALS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    unit_path = PROJECT_ROOT / "tools" / "validate" / "synthetic_unit_cases.py"
    unit_text = unit_path.read_text(encoding="utf-8")
    tp_path = PROJECT_ROOT / "tools" / "validate" / "synthetic_take_profit_cases.py"
    tp_text = tp_path.read_text(encoding="utf-8")
    scanner_path = PROJECT_ROOT / "tools" / "validate" / "scanner_expectations.py"
    scanner_text = scanner_path.read_text(encoding="utf-8")
    scanner_compact_text = re.sub(r"\s+", "", scanner_text)
    contract_path = PROJECT_ROOT / "tools" / "validate" / "synthetic_contract_cases.py"
    contract_text = contract_path.read_text(encoding="utf-8")
    contract_compact_text = re.sub(r"\s+", "", contract_text)
    tp_compact_text = re.sub(r"\s+", "", tp_text)

    add_check(results, "meta_contract", case_id, "unit_oracle_net_sell_uses_sell_ledger", True, 'build_sell_ledger_from_price(price, int(qty), params)' in unit_text)
    add_check(results, "meta_contract", case_id, "unit_oracle_position_size_uses_integer_risk_budget", True, 'risk_budget_milli = calc_risk_budget_milli(cap_milli, risk_fraction)' in unit_text)
    add_check(results, "meta_contract", case_id, "unit_oracle_has_no_legacy_float_gross_times_qty", False, 'gross = float(price) * int(qty)' in unit_text)
    add_check(results, "meta_contract", case_id, "unit_oracle_has_no_legacy_float_risk_budget", False, 'risk_budget = capital * risk_fraction' in unit_text)
    add_check(results, "meta_contract", case_id, "take_profit_case_uses_sell_ledger_for_expected_cash", True, 'expected_sell_ledger = build_sell_ledger_from_price(expected_exec_price, qty, params)' in tp_text)
    add_check(results, "meta_contract", case_id, "take_profit_case_has_no_legacy_expected_freed_cash_formula", False, 'expected_freed_cash = expected_net_price * qty' in tp_text)
    add_check(results, "meta_contract", case_id, "take_profit_case_has_no_legacy_expected_pnl_formula", False, 'expected_pnl = (expected_net_price - entry_price) * qty' in tp_text)
    add_check(results, "meta_contract", case_id, "scanner_expectations_threads_trade_date_from_clean_df", True, 'resolve_latest_trade_date_from_frame' in scanner_text and 'stats["trade_date"] = resolve_latest_trade_date_from_frame(df)' in scanner_text and 'stats["trade_date"] = resolve_latest_trade_date_from_frame(clean_df)' in scanner_text)
    add_check(results, "meta_contract", case_id, "scanner_expectations_threads_ticker_and_trade_date_into_projected_qty", True, 'ticker=ticker' in scanner_compact_text and 'trade_date=trade_date' in scanner_compact_text and 'calc_reference_candidate_qty(scanner_ref_stats["buy_limit"],scanner_ref_stats["stop_loss"],params,' in scanner_compact_text)
    add_check(results, "meta_contract", case_id, "scanner_payload_builder_threads_trade_date_to_status_oracle", True, 'status=derive_expected_scanner_status(scanner_ref_stats,params,ticker=ticker,trade_date=trade_date)' in scanner_compact_text)
    add_check(results, "meta_contract", case_id, "scanner_expectations_reuses_shared_trade_date_helper", False, 'def _resolve_trade_date_from_clean_df(' in scanner_text)
    add_check(results, "meta_contract", case_id, "scanner_live_capital_contract_threads_ticker_and_trade_date", True, 'calc_reference_candidate_qty(buy_limit,stop_loss,params,ticker="2330",trade_date=pd.Timestamp("2026-01-02"))' in contract_compact_text)
    add_check(results, "meta_contract", case_id, "scanner_half_tp_case_threads_ticker_and_trade_date", True, 'calc_reference_candidate_qty(scanner_ref_stats["buy_limit"],scanner_ref_stats["stop_loss"],scanner_case["params"],ticker=scanner_ticker,trade_date=scanner_ref_stats.get("trade_date"))' in tp_compact_text)

    summary["source_path"] = unit_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary
