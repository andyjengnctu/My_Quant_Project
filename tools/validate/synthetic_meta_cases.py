import ast
from pathlib import Path
import importlib
import json
import re
import shlex
import tempfile
from unittest.mock import patch

from .checks import add_check


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_PATH = PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md"
CMD_PATH = PROJECT_ROOT / "doc" / "CMD.md"

from .meta_contracts import (
    extract_markdown_table_rows,
    load_defined_validate_names_from_synthetic_case_modules,
    load_imported_validate_names_from_synthetic_main_entry,
    load_synthetic_registry_entries_from_source,
    summarize_legacy_app_entry_doc_reference_contract,
    summarize_no_reverse_app_import_contract,
    summarize_no_top_level_import_cycles_contract,
    summarize_single_formal_test_entry_contract,
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
    TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS,
)


def _load_main_table_statuses():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    statuses = {}
    headings = [
        ("B1. 專案設定對應清單（不含暫時特例）", 3),
        ("B2. 未明列於專案設定，但正式 test suite 應納入", 4),
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
    return catalog


def _load_done_d_rows():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    rows = extract_markdown_table_rows(text, "F. 目前所有 `DONE` 的建議測試項目摘要")
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


def _replace_markdown_table_row(text: str, *, heading: str, row_id: str, id_col_idx: int, update_cols):
    original_had_trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    in_target_section = False
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
        updated_cols = update_cols(list(cols))
        lines[line_idx] = "| " + " | ".join(updated_cols) + " |"
        updated_text = "\n".join(lines)
        if original_had_trailing_newline:
            updated_text += "\n"
        return updated_text
    raise ValueError(f"找不到 checklist heading={heading}, row_id={row_id}")


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
        invalid_rows = g_note_result.get("extra", {}).get("invalid_note_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_note_single_entry_guard_fails", "FAIL", g_note_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_note_reports_multiple_entries", True, any(row.get("id") == "B38" for row in invalid_rows))

    summary["guard_status"] = g_note_result.get("status")
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
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
            heading="F. 目前所有 `DONE` 的建議測試項目摘要",
            row_id="D111",
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
        invalid_rows = f2_result.get("extra", {}).get("invalid_entries", [])

    add_check(results, "meta_checklist", case_id, "mutated_f2_single_entry_guard_fails", "FAIL", f2_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_f2_reports_multiple_entries", True, any(row.get("id") == "D111" for row in invalid_rows))

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

    insertion_target = "### F. 目前所有 `DONE` 的建議測試項目摘要"
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
        legacy_present = legacy_result.get("extra", {}).get("legacy_f1_section_present")

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
        legacy_present = legacy_result.get("extra", {}).get("legacy_d_section_present")

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
        invalid_rows = g_transition_result.get("extra", {}).get("invalid_transition_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_transition_guard_fails", "FAIL", g_transition_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_transition_reports_invalid_row", True, any(row.get("id") == "B38" for row in invalid_rows))

    summary["guard_status"] = g_transition_result.get("status")
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
            row_id="D01",
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
        invalid_rows = g_order_result.get("extra", {}).get("invalid_order_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_order_guard_fails", "FAIL", g_order_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_order_reports_invalid_pair", True, bool(invalid_rows))

    summary["guard_status"] = g_order_result.get("status")
    summary["invalid_order_rows"] = invalid_rows
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
    done_d_rows = _load_done_d_rows()
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

    done_d_names = [row["name"] for row in done_d_rows]
    done_d_name_set = set(done_d_names)
    add_check(results, "meta_registry", case_id, "done_d_names_unique", len(done_d_names), len(done_d_name_set))

    done_d_validate_name_set = {
        row["name"].split()[0]
        for row in done_d_rows
        if row["name"].split() and row["name"].split()[0].startswith("validate_")
    }
    missing_done_d_validator_names = sorted(validator_name_set - done_d_validate_name_set)
    add_check(
        results,
        "meta_registry",
        case_id,
        "all_registered_validate_cases_listed_in_done_f2_summary",
        [],
        missing_done_d_validator_names,
    )

    for row in done_d_rows:
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
    mapped_b_ids = {row["b_id"] for row in done_d_rows}
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
            f"{row['b_id']}_done_summary_has_done_d_mapping",
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

    summary["done_d_count"] = len(done_d_rows)
    summary["done_b_count"] = len(done_b_ids)
    summary["validator_count"] = len(validator_entries)
    summary["missing_imported_names"] = missing_imported_names
    summary["missing_defined_names"] = missing_defined_names
    summary["missing_done_d_validator_names"] = missing_done_d_validator_names
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
