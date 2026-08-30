import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import importlib
import json
import re
import shlex
import tempfile
from unittest.mock import patch

from .checks import add_check
from core.model_paths import (
    RUN_BEST_PARAMS_PATH_ENV_VAR,
    resolve_default_primary_param_source_path,
    resolve_models_dir,
    discover_model_param_sources,
)
from .module_loader import build_project_absolute_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_PATH = PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md"
CMD_PATH = PROJECT_ROOT / "doc" / "CMD.md"
SYNTHETIC_VALIDATE_DIR = PROJECT_ROOT / "tools" / "validate"

from .source_index import read_source_ast, read_source_text
from .meta_contracts import (
    extract_markdown_table_rows,
    load_defined_validate_names_from_synthetic_case_modules,
    load_imported_validate_names_from_synthetic_main_entry,
    load_synthetic_registry_entries_from_source,
    summarize_critical_helper_single_source_contract,
    summarize_dependency_direction_contract,
    summarize_legacy_app_entry_doc_reference_contract,
    summarize_no_reverse_app_import_contract,
    summarize_no_top_level_import_cycles_contract,
    summarize_single_formal_test_entry_contract,
    summarize_synthetic_cases_import_target_resolution_contract,
)
from tools.local_regression.checklist_contract import (
    load_convergence_latest_statuses,
    load_done_b_rows,
    load_done_test_rows,
    load_main_catalog,
    load_checklist_tables,
    load_main_statuses,
)
from tools.local_regression.common import partition_result_statuses
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
    checked_script_paths = set()

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
        if parts[1] == "-c":
            continue

        script_rel = parts[1].replace("\\", "/")
        project_command_count += 1
        script_path = PROJECT_ROOT / script_rel
        metric_prefix = script_rel.replace("/", "_").replace(".", "_")
        if script_rel not in checked_script_paths:
            checked_script_paths.add(script_rel)
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


def validate_gui_workbench_documentation_sync_case(_base_params):
    case_id = "META_GUI_WORKBENCH_DOCUMENTATION_SYNC"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    cmd_text = (PROJECT_ROOT / "doc" / "CMD.md").read_text(encoding="utf-8")
    architecture_text = (PROJECT_ROOT / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    inspector_source = (PROJECT_ROOT / "services" / "workbench_ui" / "single_stock_inspector.py").read_text(encoding="utf-8")

    cmd_required_fragment = "交易明細與 Console 為獨立分頁"
    cmd_forbidden_fragment = "執行摘要、交易明細與 Console 為獨立分頁"
    architecture_apps_required_fragment = "交易明細與 Console 改以獨立分頁承接"
    architecture_ui_required_fragment = "交易明細與 Console 以獨立分頁承接"
    architecture_forbidden_fragment = "執行摘要與交易明細改以獨立分頁承接"
    architecture_ui_forbidden_fragment = "摘要與明細 / Console 以獨立分頁承接"

    add_check(results, "meta_cmd_contract", case_id, "inspector_notebook_has_trade_detail_tab", True, 'text="交易明細"' in inspector_source)
    add_check(results, "meta_cmd_contract", case_id, "inspector_notebook_has_console_tab", True, 'text="Console"' in inspector_source)
    add_check(results, "meta_cmd_contract", case_id, "inspector_notebook_omits_summary_tab", False, 'text="執行摘要"' in inspector_source)
    add_check(results, "meta_cmd_contract", case_id, "cmd_workbench_mentions_trade_detail_console_tabs", True, cmd_required_fragment in cmd_text)
    add_check(results, "meta_cmd_contract", case_id, "cmd_workbench_omits_summary_tab_description", False, cmd_forbidden_fragment in cmd_text)
    add_check(results, "meta_cmd_contract", case_id, "architecture_apps_workbench_mentions_trade_detail_console_tabs", True, architecture_apps_required_fragment in architecture_text)
    add_check(results, "meta_cmd_contract", case_id, "architecture_ui_workbench_mentions_trade_detail_console_tabs", True, architecture_ui_required_fragment in architecture_text)
    add_check(results, "meta_cmd_contract", case_id, "architecture_workbench_omits_summary_tab_description", False, architecture_forbidden_fragment in architecture_text)
    add_check(results, "meta_cmd_contract", case_id, "architecture_ui_workbench_omits_summary_tab_description", False, architecture_ui_forbidden_fragment in architecture_text)

    summary["cmd_required_fragment"] = cmd_required_fragment
    summary["architecture_required_fragments"] = [architecture_apps_required_fragment, architecture_ui_required_fragment]
    summary["tabs"] = ["K 線圖", "交易明細", "Console"]
    return results, summary

def validate_architecture_workbench_entry_file_tree_sync_case(_base_params):
    case_id = "META_ARCHITECTURE_WORKBENCH_ENTRY_FILE_TREE_SYNC"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    architecture_text = (PROJECT_ROOT / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    workbench_source = (PROJECT_ROOT / "apps" / "workbench.py").read_text(encoding="utf-8")

    required_tree_fragment = "│  └─ workbench.py                    # GUI 工作台正式入口（薄入口）"
    stale_missing_tree_fragment = "│  └─ vip_scanner.py                  # 掃描器正式入口（薄入口）"

    add_check(results, "meta_architecture_contract", case_id, "architecture_apps_file_tree_lists_workbench_entry", True, required_tree_fragment in architecture_text)
    add_check(results, "meta_architecture_contract", case_id, "architecture_apps_file_tree_has_no_stale_missing_workbench_tail", False, stale_missing_tree_fragment in architecture_text)
    workbench_entry_fragments = [
        "`apps/workbench.py`：GUI / workbench 正式入口",
        "`apps/workbench.py` 為單一 GUI 啟用入口",
    ]

    add_check(
        results,
        "meta_architecture_contract",
        case_id,
        "architecture_apps_section_mentions_workbench_single_gui_entry",
        True,
        any(fragment in architecture_text for fragment in workbench_entry_fragments),
    )
    add_check(results, "meta_architecture_contract", case_id, "workbench_app_remains_thin_gui_entry", True, "from services.workbench_ui import main" in workbench_source and '__all__ = ["main"]' in workbench_source)

    summary["required_tree_fragment"] = required_tree_fragment
    summary["source_paths"] = ["doc/ARCHITECTURE.md", "apps/workbench.py"]
    return results, summary


def validate_model_param_source_resolution_contract_case(_base_params):
    case_id = "META_MODEL_PARAM_SOURCE_RESOLUTION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    architecture_text = (PROJECT_ROOT / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    models_dir = Path(resolve_models_dir(PROJECT_ROOT, environ={}))

    required_fragments = [
        "strategy_params/",
        "models root不再放current策略JSON",
        "formal_primary_params.json",
        "V16_RUN_BEST_PARAMS_PATH",
        "models/strategy_params/canonical/run_best_params.json",
    ]
    stale_exact_tree_fragments = [
        "│  └─ <optimizer parameter artifacts>.json",
        "預設參數 fallback 仍解析到 `models/run_best_params.json`",
    ]

    for idx, fragment in enumerate(required_fragments, start=1):
        add_check(
            results,
            "meta_architecture_contract",
            case_id,
            f"architecture_documents_param_source_contract_{idx}",
            True,
            fragment in architecture_text,
        )
    for idx, fragment in enumerate(stale_exact_tree_fragments, start=1):
        add_check(
            results,
            "meta_architecture_contract",
            case_id,
            f"architecture_omits_optional_exact_param_artifact_{idx}",
            False,
            fragment in architecture_text,
        )

    default_param_source_path = Path(
        resolve_default_primary_param_source_path(PROJECT_ROOT, environ={})
    )
    expected_default_path = models_dir / "strategy_params" / "canonical" / "run_best_params.json"
    add_check(
        results,
        "meta_architecture_contract",
        case_id,
        "default_primary_param_is_canonical_named_run_best",
        expected_default_path.resolve(),
        default_param_source_path.resolve(),
    )

    with tempfile.TemporaryDirectory(prefix="canonical_param_discovery_contract_") as temp_dir:
        temp_root = Path(temp_dir)
        legacy_root = temp_root / "models"
        legacy_root.mkdir(parents=True, exist_ok=True)
        (legacy_root / "run_best_params.json").write_text("{}", encoding="utf-8")
        add_check(
            results, "meta_architecture_contract", case_id,
            "legacy_models_root_strategy_json_is_not_current_discovery_source",
            [],
            discover_model_param_sources(str(temp_root), environ={}),
        )
        canonical_active = temp_root / "models" / "strategy_params" / "canonical" / "run_best_params.json"
        canonical_active.parent.mkdir(parents=True, exist_ok=True)
        canonical_active.write_text("{}", encoding="utf-8")
        discovered = discover_model_param_sources(str(temp_root), environ={})
        add_check(
            results, "meta_architecture_contract", case_id,
            "canonical_named_run_best_is_discovered",
            True,
            any(Path(record["path"]).resolve() == canonical_active.resolve() for record in discovered),
        )

    with tempfile.TemporaryDirectory(prefix="formal_param_override_contract_") as temp_dir:
        override_path = Path(temp_dir) / "formal_primary_params.json"
        override_path.write_text("{}", encoding="utf-8")
        override_env = {RUN_BEST_PARAMS_PATH_ENV_VAR: str(override_path)}
        resolved_override_path = Path(
            resolve_default_primary_param_source_path(PROJECT_ROOT, environ=override_env)
        )
        add_check(
            results,
            "meta_architecture_contract",
            case_id,
            "runtime_primary_param_override_resolves_exact_path",
            override_path.resolve(),
            resolved_override_path.resolve(),
        )
        add_check(
            results,
            "meta_architecture_contract",
            case_id,
            "runtime_primary_param_override_may_live_outside_models",
            True,
            resolved_override_path.parent != models_dir,
        )
        add_check(
            results,
            "meta_architecture_contract",
            case_id,
            "runtime_primary_param_override_exists",
            True,
            resolved_override_path.is_file(),
        )

    summary["default_fallback_path"] = str(default_param_source_path)
    summary["models_dir"] = str(models_dir)
    summary["source_paths"] = [
        "doc/ARCHITECTURE.md",
        "core/model_paths.py",
        "tools/local_regression/run_all.py",
    ]
    return results, summary



def validate_architecture_local_regression_meta_quality_file_tree_sync_case(_base_params):
    case_id = "META_ARCHITECTURE_LOCAL_REGRESSION_META_QUALITY_FILE_TREE_SYNC"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    architecture_text = (PROJECT_ROOT / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    required_tree_fragment = "├── run_meta_quality.py"
    stale_malformed_fragment = "├── run_meta_quality.py（含 `run_all.py` helper path coverage probe）"
    required_role_fragment = "- `run_meta_quality.py`：meta quality 工具；"

    add_check(results, "meta_architecture_contract", case_id, "architecture_local_regression_file_tree_lists_run_meta_quality_entry", True, required_tree_fragment in architecture_text)
    add_check(results, "meta_architecture_contract", case_id, "architecture_local_regression_file_tree_has_no_malformed_run_meta_quality_entry", False, stale_malformed_fragment in architecture_text)
    add_check(results, "meta_architecture_contract", case_id, "architecture_local_regression_role_section_mentions_run_meta_quality_tooling", True, required_role_fragment in architecture_text)
    add_check(results, "meta_architecture_contract", case_id, "repo_ships_tools_local_regression_run_meta_quality_py", True, (PROJECT_ROOT / "tools" / "local_regression" / "run_meta_quality.py").exists())

    summary["required_tree_fragment"] = required_tree_fragment
    summary["source_paths"] = ["doc/ARCHITECTURE.md", "tools/local_regression/run_meta_quality.py"]
    return results, summary


def validate_trade_analysis_legacy_naming_documentation_contract_case(_base_params):
    case_id = "META_TRADE_ANALYSIS_LEGACY_NAMING_DOCUMENTATION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    cmd_text = (PROJECT_ROOT / "doc" / "CMD.md").read_text(encoding="utf-8")
    architecture_text = (PROJECT_ROOT / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    trade_log_text = (PROJECT_ROOT / "services" / "trade_analysis" / "trade_log.py").read_text(encoding="utf-8")

    cmd_has_legacy_debug_labels = all(fragment in cmd_text for fragment in ("legacy `run_debug_*`", "`debug_trade_log`"))
    architecture_has_legacy_debug_labels = all(fragment in architecture_text for fragment in ("legacy `run_debug_*`", "`debug_trade_log`"))
    architecture_marks_legacy_as_compatibility = ("為維持相容性" in architecture_text) or ("以維持相容性" in architecture_text)
    architecture_formal_entry_section = architecture_text.split("## 正式入口", 1)[1].split("## ", 1)[0] if "## 正式入口" in architecture_text else ""

    add_check(results, "meta_cmd_contract", case_id, "cmd_workbench_is_single_user_entry_for_trade_analysis", True, "`apps/workbench.py` 為 GUI 正式入口，也是單股 trade-analysis 的單一使用者入口" in cmd_text)
    add_check(results, "meta_cmd_contract", case_id, "cmd_trade_analysis_helper_described_as_backend_not_formal_entry", True, "`services/trade_analysis/trade_log.py` 提供單股 trade-analysis 共用 backend / 開發輔助 CLI；正式使用者入口仍為 `apps/workbench.py`" in cmd_text)
    add_check(results, "meta_cmd_contract", case_id, "cmd_trade_analysis_mentions_legacy_debug_api_labels", True, cmd_has_legacy_debug_labels)
    add_check(results, "meta_cmd_contract", case_id, "cmd_trade_analysis_output_dir_explicitly_marked_legacy", True, "`outputs/debug_trade_log/`：`trade_analysis` 單股分析輸出；為維持既有工具鏈相容，暫沿用 legacy 目錄名 `debug_trade_log`" in cmd_text)
    add_check(results, "meta_cmd_contract", case_id, "cmd_trade_analysis_retention_section_marks_legacy_output_dir", True, "`outputs/debug_trade_log/`（trade_analysis legacy output dir）" in cmd_text)
    add_check(results, "meta_cmd_contract", case_id, "cmd_has_no_trade_log_formal_entry_label", False, "`services/trade_analysis/trade_log.py` 為單股 trade-analysis 正式入口" in cmd_text)
    add_check(results, "meta_cmd_contract", case_id, "cmd_has_no_stale_debug_formal_entry_label", False, "為 debug 正式入口" in cmd_text)

    add_check(results, "meta_architecture_contract", case_id, "architecture_trade_analysis_described_as_subsystem", True, "- `services/trade_analysis/`：單股 trade-analysis 子系統；" in architecture_text)
    add_check(results, "meta_architecture_contract", case_id, "architecture_trade_analysis_bound_to_workbench_entry", True, "由 `apps/workbench.py` 經 `services/workbench_ui/` 觸發" in architecture_text and "提供共用 backend / 開發輔助 CLI" in architecture_text)
    add_check(results, "meta_architecture_contract", case_id, "architecture_trade_analysis_mentions_legacy_debug_api_labels", True, architecture_has_legacy_debug_labels and architecture_marks_legacy_as_compatibility)
    add_check(results, "meta_architecture_contract", case_id, "architecture_output_section_marks_legacy_output_dir", True, "`outputs/debug_trade_log/`（trade_analysis legacy output dir）" in architecture_text)
    add_check(results, "meta_architecture_contract", case_id, "architecture_has_no_trade_log_formal_entry_label", False, "`services/trade_analysis/trade_log.py` 為單股 trade-analysis 正式入口" in architecture_text)
    add_check(results, "meta_architecture_contract", case_id, "architecture_formal_entry_section_excludes_trade_log_helper", False, "`services/trade_analysis/trade_log.py`" in architecture_formal_entry_section)
    add_check(results, "meta_architecture_contract", case_id, "architecture_has_no_stale_debug_subsystem_label", False, "交易除錯子系統" in architecture_text)

    add_check(results, "meta_trade_analysis_cli_contract", case_id, "trade_log_prompt_uses_analysis_wording", True, "請輸入要分析的股票代號" in trade_log_text)
    add_check(results, "meta_trade_analysis_cli_contract", case_id, "trade_log_banner_uses_trade_analysis_wording", True, "單股 trade-analysis 交易明細工具" in trade_log_text)
    add_check(results, "meta_trade_analysis_cli_contract", case_id, "trade_log_help_marks_workbench_as_formal_user_entry", True, "正式使用者入口為 apps/workbench.py" in trade_log_text)
    add_check(results, "meta_trade_analysis_cli_contract", case_id, "trade_log_has_no_stale_debug_prompt_or_banner", False, ("請輸入要除錯的股票代號" in trade_log_text) or ("交易明細除錯工具" in trade_log_text))

    summary["source_paths"] = ["doc/CMD.md", "doc/ARCHITECTURE.md", "services/trade_analysis/trade_log.py"]
    summary["architecture_has_legacy_debug_labels"] = architecture_has_legacy_debug_labels
    summary["architecture_marks_legacy_as_compatibility"] = architecture_marks_legacy_as_compatibility
    return results, summary


def validate_validate_runtime_tmp_output_staging_contract_case(_base_params):
    case_id = "META_VALIDATE_RUNTIME_TMP_OUTPUT_STAGING_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    error_cases_text = (PROJECT_ROOT / "tools" / "validate" / "synthetic_error_cases.py").read_text(encoding="utf-8")
    regression_cases_text = (PROJECT_ROOT / "tools" / "validate" / "synthetic_regression_cases.py").read_text(encoding="utf-8")
    architecture_text = (PROJECT_ROOT / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    cmd_text = (PROJECT_ROOT / "doc" / "CMD.md").read_text(encoding="utf-8")
    run_all_text = (PROJECT_ROOT / "tools" / "local_regression" / "run_all.py").read_text(encoding="utf-8")

    required_staging_fragment = 'output_dir_path(PROJECT_ROOT, "local_regression") / "_staging" / "validate_runtime"'
    stale_root_fragment = 'PROJECT_ROOT / "outputs" / "validate"'

    add_check(results, "meta_output_contract", case_id, "synthetic_error_cases_use_local_regression_staging_runtime_root", True, required_staging_fragment in error_cases_text)
    add_check(results, "meta_output_contract", case_id, "synthetic_regression_cases_use_local_regression_staging_runtime_root", True, required_staging_fragment in regression_cases_text)
    add_check(results, "meta_output_contract", case_id, "validate_runtime_paths_have_no_outputs_validate_root", False, (stale_root_fragment in error_cases_text) or (stale_root_fragment in regression_cases_text) or ("outputs' / 'validate'" in regression_cases_text))
    add_check(results, "meta_output_contract", case_id, "run_all_retention_keeps_local_regression_staging_cleanup_rule", True, 'name="local_regression_staging"' in run_all_text and 'output_dir_path(PROJECT_ROOT, "local_regression") / "_staging"' in run_all_text)
    add_check(results, "meta_output_contract", case_id, "cmd_documents_local_regression_staging_runtime_area", True, '`outputs/local_regression/_staging/`：formal / validate 暫存 staging；屬 `local_regression` 內部子目錄，會由 retention 自動清理。' in cmd_text)
    add_check(results, "meta_output_contract", case_id, "architecture_documents_no_outputs_validate_root_category", True, '`outputs/local_regression/_staging/` 為 local regression / validate 共用暫存 staging 子目錄；不新增 `outputs/validate/` 根分類。' in architecture_text)

    summary["source_paths"] = [
        "tools/validate/synthetic_error_cases.py",
        "tools/validate/synthetic_regression_cases.py",
        "tools/local_regression/run_all.py",
        "doc/CMD.md",
        "doc/ARCHITECTURE.md",
    ]
    return results, summary


def validate_trade_analysis_canonical_alias_export_contract_case(_base_params):
    case_id = "META_TRADE_ANALYSIS_CANONICAL_ALIAS_EXPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    package_text = (PROJECT_ROOT / "services" / "trade_analysis" / "__init__.py").read_text(encoding="utf-8")
    trade_log_text = (PROJECT_ROOT / "services" / "trade_analysis" / "trade_log.py").read_text(encoding="utf-8")

    canonical_aliases = [
        "run_trade_analysis",
        "run_trade_backtest",
        "run_prepared_trade_backtest",
        "run_ticker_analysis",
    ]
    legacy_aliases = [
        "run_debug_analysis",
        "run_debug_backtest",
        "run_debug_prepared_backtest",
        "run_debug_ticker_analysis",
    ]

    for alias_name in canonical_aliases:
        add_check(results, "meta_contract", case_id, f"package_exports_{alias_name}", True, f'def {alias_name}(' in package_text and f'"{alias_name}"' in package_text)
        add_check(results, "meta_contract", case_id, f"trade_log_exports_{alias_name}", True, f'def {alias_name}(' in trade_log_text and f'"{alias_name}"' in trade_log_text)

    for alias_name in legacy_aliases:
        add_check(results, "meta_contract", case_id, f"package_keeps_legacy_{alias_name}", True, f'def {alias_name}(' in package_text and f'"{alias_name}"' in package_text)
        add_check(results, "meta_contract", case_id, f"trade_log_keeps_legacy_{alias_name}", True, f'def {alias_name}(' in trade_log_text and f'"{alias_name}"' in trade_log_text)

    summary["canonical_aliases"] = canonical_aliases
    summary["legacy_aliases"] = legacy_aliases
    return results, summary



def validate_no_reverse_app_layer_dependencies_case(_base_params):
    case_id = "META_NO_REVERSE_APP_LAYER_DEPENDENCIES"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    reverse_contract = summarize_no_reverse_app_import_contract(PROJECT_ROOT)
    reverse_violations = [
        f"{item['path']}:{item['lineno']} -> {item['module']}"
        for item in reverse_contract["violations"]
    ]
    direction_contract = summarize_dependency_direction_contract(PROJECT_ROOT)
    direction_violations = [
        f"{item['path']}:{item['lineno']} -> {item['module']}"
        for item in direction_contract["violations"]
    ]
    add_check(results, "meta_entry_contract", case_id, "core_filters_services_and_tools_do_not_import_apps_or_tools_upward", [], reverse_violations)
    add_check(results, "meta_entry_contract", case_id, "formal_apps_do_not_depend_on_runtime_tools_and_filters_do_not_depend_on_services", [], direction_violations)

    summary["violation_count"] = len(reverse_violations) + len(direction_violations)
    summary["violations"] = [*reverse_violations, *direction_violations]
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

    with tempfile.TemporaryDirectory(prefix="top_level_cycle_contract_") as temp_dir:
        fixture_root = Path(temp_dir)
        (fixture_root / "apps").mkdir()
        (fixture_root / "core").mkdir()
        (fixture_root / "tools").mkdir()
        (fixture_root / "core" / "a.py").write_text("from core.b import B\nA = 1\n", encoding="utf-8")
        (fixture_root / "core" / "b.py").write_text("from core.a import A\nB = 1\n", encoding="utf-8")
        absolute_cycle_contract = summarize_no_top_level_import_cycles_contract(fixture_root)

        (fixture_root / "core" / "b.py").write_text(
            "def load_a():\n"
            "    from core.a import A\n"
            "    return A\n"
            "B = 1\n",
            encoding="utf-8",
        )
        lazy_import_contract = summarize_no_top_level_import_cycles_contract(fixture_root)

    absolute_cycle_modules = [item["modules"] for item in absolute_cycle_contract["violations"]]
    add_check(
        results,
        "meta_entry_contract",
        case_id,
        "absolute_import_top_level_cycle_fixture_detected",
        [["core.a", "core.b"]],
        absolute_cycle_modules,
    )
    lazy_cycle_modules = [item["modules"] for item in lazy_import_contract["violations"]]
    add_check(
        results,
        "meta_entry_contract",
        case_id,
        "function_local_lazy_import_cycle_fixture_detected",
        [["core.a", "core.b"]],
        lazy_cycle_modules,
    )

    summary["module_count"] = contract["module_count"]
    summary["cycle_count"] = len(contract["violations"])
    summary["cycles"] = violations
    summary["absolute_cycle_fixture_count"] = len(absolute_cycle_contract["violations"])
    summary["lazy_import_fixture_count"] = len(lazy_import_contract["violations"])
    return results, summary


def validate_single_formal_test_entry_contract_case(_base_params):
    case_id = "META_SINGLE_FORMAL_TEST_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    contract = summarize_single_formal_test_entry_contract(PROJECT_ROOT)
    add_check(results, "meta_entry_contract", case_id, "test_suite_entry_file_exists", True, contract["test_suite_exists"])
    add_check(results, "meta_entry_contract", case_id, "cmd_declares_single_entry", True, contract["cmd_declares_single_entry"])
    add_check(results, "meta_entry_contract", case_id, "architecture_declares_single_entry", True, contract["architecture_declares_single_entry"])
    add_check(results, "meta_entry_contract", case_id, "no_legacy_app_test_entries", [], contract["legacy_entry_paths"])
    add_check(results, "meta_entry_contract", case_id, "no_suspicious_alternate_app_test_entries", [], contract["suspicious_app_entries"])

    summary["app_py_files"] = contract["app_py_files"]
    summary["legacy_entry_paths"] = contract["legacy_entry_paths"]
    summary["suspicious_app_entries"] = contract["suspicious_app_entries"]
    return results, summary



def validate_checklist_physical_trading_principles_contract_case(_base_params):
    case_id = "META_CHECKLIST_PHYSICAL_TRADING_PRINCIPLES_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    checklist_text = (PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md").read_text(encoding="utf-8")

    l_only_text = "`L` 只作進場上限 / 最壞風險 sizing 上界"
    pfill_text = "`P_fill + ATR_t`"
    continuation_barrier_text = "固定反事實 `P' = min(Open, L)`"
    inclusive_hit_text = "長倉 hit 採 `Low <= line` / `High >= line`"

    add_check(results, "meta_entry_contract", case_id, "checklist_declares_l_is_entry_and_sizing_only", True, l_only_text in checklist_text)
    add_check(results, "meta_entry_contract", case_id, "checklist_declares_first_actionable_stop_uses_pfill_and_atr", True, pfill_text in checklist_text)
    add_check(results, "meta_entry_contract", case_id, "checklist_declares_extended_candidate_fixed_counterfactual_barrier", True, continuation_barrier_text in checklist_text)
    add_check(results, "meta_entry_contract", case_id, "checklist_declares_inclusive_hit_semantics", True, inclusive_hit_text in checklist_text)

    summary["checklist_declares_l_is_entry_and_sizing_only"] = l_only_text in checklist_text
    summary["checklist_declares_first_actionable_stop_uses_pfill_and_atr"] = pfill_text in checklist_text
    summary["checklist_declares_extended_candidate_fixed_counterfactual_barrier"] = continuation_barrier_text in checklist_text
    summary["checklist_declares_inclusive_hit_semantics"] = inclusive_hit_text in checklist_text
    return results, summary



def _exception_type_names(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.Tuple):
        return [name for element in node.elts for name in _exception_type_names(element)]
    return []


def _scan_exception_handlers(paths, *, accepted_names=None, pass_only=False, exempt_synthetic=False):
    syntax_errors, failures, scanned = [], [], []
    for path in paths:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if exempt_synthetic and rel.startswith("tools/validate/synthetic_"):
            continue
        scanned.append(rel)
        try:
            parsed = read_source_ast(path)
        except SyntaxError as exc:
            syntax_errors.append(f"{rel}:{exc.lineno}: {exc.msg}")
            continue
        for node in ast.walk(parsed):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            names = set(_exception_type_names(node.type))
            if not names or (accepted_names is not None and not names.intersection(accepted_names)):
                continue
            if pass_only:
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
                    continue
                if names <= {"FileNotFoundError"} or names.intersection({"Exception", "BaseException", "ImportError", "ModuleNotFoundError", "TclError"}):
                    continue
                failures.append(f"{rel}:{node.lineno}: pass-only specific exception handler must trace, re-raise, or use an allowed control-flow exception")
                continue
            body = ast.Module(body=node.body, type_ignores=[])
            if any(isinstance(child, ast.Raise) for child in ast.walk(body)):
                continue
            if not node.name:
                failures.append(f"{rel}:{node.lineno}: exception fallback must bind exception name unless it re-raises")
                continue
            if not any(isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id == node.name for child in ast.walk(body)):
                failures.append(f"{rel}:{node.lineno}: exception fallback must use bound exception or re-raise")
    return syntax_errors, failures, scanned


def _exception_traceability_result(case_id, paths, accepted_names, parse_metric, failure_metric, *, pass_only=False, target_metric=None, exempt_synthetic=False):
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    syntax_errors, failures, scanned = _scan_exception_handlers(
        paths, accepted_names=None if accepted_names is None else set(accepted_names), pass_only=pass_only, exempt_synthetic=exempt_synthetic,
    )
    if target_metric:
        add_check(results, "meta_contract", case_id, target_metric, True, bool(scanned))
    add_check(results, "meta_contract", case_id, parse_metric, [], syntax_errors)
    add_check(results, "meta_contract", case_id, failure_metric, [], failures)
    summary.update(scanned_file_count=len(scanned), failure_count=len(failures), syntax_error_count=len(syntax_errors))
    return results, summary

def validate_gui_tcl_fallback_traceability_contract_case(_base_params):
    paths = sorted((PROJECT_ROOT / "services" / "workbench_ui").rglob("*.py"))
    results, summary = _exception_traceability_result(
        "META_GUI_TCL_FALLBACK_TRACEABILITY_CONTRACT", paths, {"TclError"},
        "gui_tcl_fallback_handler_files_parse", "gui_tcl_fallbacks_bind_and_trace_or_reraise",
        target_metric="gui_tcl_fallback_scan_targets_present",
    )
    summary["scan_root"] = "services/workbench_ui"
    return results, summary


def validate_optional_dependency_fallback_traceability_contract_case(_base_params):
    paths = [PROJECT_ROOT / rel for rel in (
        "services/trade_analysis/charting.py", "services/downloader/runtime.py",
        "services/workbench_ui/single_stock_inspector.py", "tools/validate/main.py",
    )]
    return _exception_traceability_result(
        "META_OPTIONAL_DEPENDENCY_FALLBACK_TRACEABILITY_CONTRACT", paths, {"ImportError", "ModuleNotFoundError"},
        "optional_dependency_fallback_handler_files_parse", "optional_dependency_fallbacks_bind_and_trace_or_reraise",
    )


def validate_specific_pass_only_exception_traceability_contract_case(_base_params):
    paths = [path for root_name in ("apps", "config", "core", "filters", "services", "strategies", "tools")
             for path in sorted((PROJECT_ROOT / root_name).rglob("*.py"))]
    return _exception_traceability_result(
        "META_SPECIFIC_PASS_ONLY_EXCEPTION_TRACEABILITY_CONTRACT", paths, None,
        "specific_pass_only_exception_handler_files_parse", "specific_pass_only_exception_handlers_forbidden",
        pass_only=True, exempt_synthetic=True,
    )


def validate_broad_exception_traceability_contract_case(_base_params):
    paths = [path for root_name in ("apps", "config", "core", "strategies", "tools")
             for path in sorted((PROJECT_ROOT / root_name).rglob("*.py"))]
    return _exception_traceability_result(
        "META_BROAD_EXCEPTION_TRACEABILITY_CONTRACT", paths, {"Exception", "BaseException"},
        "broad_exception_handler_files_parse", "broad_exception_handlers_bind_and_trace_or_reraise",
    )

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
    duplicate_entry_names = sorted(name for name in set(entry_names) if entry_names.count(name) > 1)

    registry_source = (SYNTHETIC_VALIDATE_DIR / "synthetic_cases.py").read_text(encoding="utf-8")
    registry_tree = ast.parse(registry_source)
    location_coupled_keywords = sorted({
        keyword.arg
        for node in ast.walk(registry_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_entry"
        for keyword in node.keywords
        if keyword.arg in {"impacted_modules", "coverage_targets", "module_paths", "file_paths"}
    })

    add_check(results, "meta_registry", case_id, "registry_metadata_not_empty", True, len(entries) > 0)
    add_check(results, "meta_registry", case_id, "registry_metadata_names_unique", [], duplicate_entry_names)
    add_check(results, "meta_registry", case_id, "registry_metadata_validator_names_prefixed", [], invalid_names)
    add_check(results, "meta_registry", case_id, "registry_metadata_layers_valid", [], invalid_layers)
    add_check(results, "meta_registry", case_id, "registry_metadata_cost_classes_valid", [], invalid_cost_classes)
    add_check(
        results,
        "meta_registry",
        case_id,
        "registry_metadata_is_location_independent",
        [],
        location_coupled_keywords,
    )

    breakout_quality_case_modules = (
        "synthetic_breakout_quality_policy_cases",
        "synthetic_breakout_quality_artifact_cases",
        "synthetic_breakout_quality_model_cases",
        "synthetic_breakout_quality_audit_cases",
        "synthetic_breakout_quality_pit_cases",
        "synthetic_breakout_quality_strategy_cases",
        "synthetic_breakout_quality_strategy_app_cases",
        "synthetic_breakout_quality_strategy_plan_cases",
    )
    breakout_quality_module_paths = [
        SYNTHETIC_VALIDATE_DIR / f"{module_name}.py"
        for module_name in breakout_quality_case_modules
    ]
    missing_breakout_quality_case_modules = [
        path.name
        for path in breakout_quality_module_paths
        if not path.exists()
    ]

    declared_breakout_quality_validators = []
    for path in breakout_quality_module_paths:
        if not path.exists():
            continue
        module_tree = read_source_ast(path)
        declared_breakout_quality_validators.extend(
            node.name
            for node in module_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("validate_")
        )
    duplicate_breakout_quality_validator_owners = sorted(
        name
        for name in set(declared_breakout_quality_validators)
        if declared_breakout_quality_validators.count(name) > 1
    )

    breakout_quality_facade_path = SYNTHETIC_VALIDATE_DIR / "synthetic_breakout_quality_cases.py"
    breakout_quality_facade_exists = breakout_quality_facade_path.exists()

    synthetic_cases_source = read_source_text(SYNTHETIC_VALIDATE_DIR / "synthetic_cases.py")
    synthetic_cases_uses_breakout_quality_facade = (
        "from .synthetic_breakout_quality_cases import (" in synthetic_cases_source
    )
    missing_direct_breakout_quality_case_imports = sorted(
        module_name
        for module_name in breakout_quality_case_modules
        if f"from .{module_name} import (" not in synthetic_cases_source
    )
    declared_coverage_targets = set(COVERAGE_TARGETS)
    breakout_quality_validator_paths = {
        "tools/validate/synthetic_breakout_quality_support.py",
        *(f"tools/validate/{module_name}.py" for module_name in breakout_quality_case_modules),
    }
    unexpected_breakout_quality_validator_coverage_targets = sorted(
        breakout_quality_validator_paths & declared_coverage_targets
    )
    unexpected_synthetic_implementation_coverage_targets = sorted(
        rel_path
        for rel_path in declared_coverage_targets
        if rel_path.startswith("tools/validate/synthetic")
    )
    breakout_quality_facade_coverage_target = (
        "tools/validate/synthetic_breakout_quality_cases.py" in declared_coverage_targets
    )

    add_check(
        results,
        "meta_registry",
        case_id,
        "breakout_quality_case_modules_exist",
        [],
        missing_breakout_quality_case_modules,
    )
    add_check(
        results,
        "meta_registry",
        case_id,
        "breakout_quality_validators_have_single_domain_owner",
        [],
        duplicate_breakout_quality_validator_owners,
    )
    add_check(
        results,
        "meta_registry",
        case_id,
        "breakout_quality_retired_compatibility_facade_absent",
        False,
        breakout_quality_facade_exists,
    )
    add_check(
        results,
        "meta_registry",
        case_id,
        "synthetic_registry_imports_breakout_quality_domain_owners_directly",
        False,
        synthetic_cases_uses_breakout_quality_facade,
    )
    add_check(
        results,
        "meta_registry",
        case_id,
        "synthetic_registry_has_all_breakout_quality_domain_imports",
        [],
        missing_direct_breakout_quality_case_imports,
    )
    add_check(
        results,
        "meta_registry",
        case_id,
        "breakout_quality_validator_implementations_are_not_key_coverage_targets",
        [],
        unexpected_breakout_quality_validator_coverage_targets,
    )
    add_check(
        results,
        "meta_registry",
        case_id,
        "synthetic_validator_implementations_are_not_key_coverage_targets",
        [],
        unexpected_synthetic_implementation_coverage_targets,
    )
    add_check(
        results,
        "meta_registry",
        case_id,
        "breakout_quality_compatibility_facade_is_not_key_coverage_target",
        False,
        breakout_quality_facade_coverage_target,
    )

    retired_unused_paths = (
        "filters/breakout_quality/mantis_pretrained.py",
        "filters/breakout_quality/moment_pretrained.py",
        "filters/registry.py",
        "tools/local_regression/run_scanner_terminal_guard.py",
        "tools/validate/synthetic_breakout_quality_cases.py",
        "tools/filters/breakout_quality/train_daily_ranker.py",
        "tools/optimizer/objective.py",
        "tools/optimizer/objective_filters.py",
        "tools/optimizer/objective_profiles.py",
        "tools/optimizer/objective_runner.py",
        "tools/optimizer/param_cache.py",
        "tools/optimizer/profile.py",
        "tools/optimizer/trial_inputs.py",
        "services/portfolio_sim/runtime_common.py",
    )
    resurrected_retired_paths = [
        rel_path
        for rel_path in retired_unused_paths
        if (SYNTHETIC_VALIDATE_DIR.parents[1] / rel_path).exists()
    ]
    add_check(
        results,
        "meta_registry",
        case_id,
        "retired_unused_modules_remain_absent",
        [],
        resurrected_retired_paths,
    )

    with tempfile.TemporaryDirectory(prefix="source_index_cache_contract_") as temp_dir_text:
        cache_probe_path = Path(temp_dir_text) / "probe.py"
        cache_probe_path.write_text("VALUE = 1\n", encoding="utf-8")
        first_tree = read_source_ast(cache_probe_path)
        second_tree = read_source_ast(cache_probe_path)
        cache_reuses_unchanged_ast = first_tree is second_tree

        cache_probe_path.write_text("VALUE = 22\n", encoding="utf-8")
        third_tree = read_source_ast(cache_probe_path)
        third_value = None
        for statement in third_tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "VALUE"
                for target in statement.targets
            ):
                continue
            if isinstance(statement.value, ast.Constant):
                third_value = statement.value.value
                break
        cache_invalidates_changed_file = bool(
            third_tree is not second_tree and third_value == 22
        )

    add_check(
        results,
        "meta_registry",
        case_id,
        "source_index_reuses_unchanged_ast",
        True,
        cache_reuses_unchanged_ast,
    )
    add_check(
        results,
        "meta_registry",
        case_id,
        "source_index_invalidates_changed_file",
        True,
        cache_invalidates_changed_file,
    )

    layer_counts = {}
    for entry in entries:
        layer_counts[entry["layer"]] = layer_counts.get(entry["layer"], 0) + 1
    summary["validator_count"] = len(entries)
    summary["layer_counts"] = layer_counts
    return results, summary



def validate_checklist_t_formal_command_single_entry_case(_base_params):
    meta_quality_module = importlib.import_module("tools.local_regression.run_meta_quality")

    case_id = "META_CHECKLIST_T_FORMAL_COMMAND_SINGLE_ENTRY"
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
            row_id="T108",
            id_col_idx=0,
            update_cols=lambda cols: [cols[0], f"`{command_entry}`", cols[2]],
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_t_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_t_formal_command_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    t_result = result_by_name.get("checklist_t_rows_use_single_test_entry", {})
    invalid_rows = t_result.get("invalid_entries")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(t_result, "invalid_entries", [])

    add_check(results, "meta_checklist", case_id, "mutated_t_formal_command_single_entry_guard_passes", "PASS", t_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_t_formal_command_not_reported_invalid", False, any(row.get("id") == "T108" for row in invalid_rows))

    summary["guard_status"] = t_result.get("status")
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

    done_test_rows = load_done_test_rows(CHECKLIST_PATH)
    invalid_ids = [row["id"] for row in done_test_rows if not re.fullmatch(r"T\d+", row["id"])]
    invalid_b_ids = [row["b_id"] for row in done_test_rows if not re.fullmatch(r"B\d+", row["b_id"])]
    add_check(results, "meta_checklist", case_id, "done_test_summary_rows_use_valid_t_ids", [], invalid_ids)
    add_check(results, "meta_checklist", case_id, "done_test_summary_rows_use_valid_b_ids", [], invalid_b_ids)

    summary["invalid_ids"] = invalid_ids
    summary["invalid_b_ids"] = invalid_b_ids
    return results, summary


def validate_checklist_t_single_entry_delimiter_case(_base_params):
    meta_quality_module = importlib.import_module("tools.local_regression.run_meta_quality")

    case_id = "META_CHECKLIST_T_SINGLE_ENTRY_DELIMITER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="T. 目前所有 `DONE` 的建議測試項目摘要",
            row_id="T108",
            id_col_idx=0,
            update_cols=lambda cols: [cols[0], "`tools/local_regression/run_meta_quality.py` / `tools/validate/meta_contracts.py`", cols[2]],
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_t_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_t_entry_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    t_result = result_by_name.get("checklist_t_rows_use_single_test_entry", {})
    invalid_rows = t_result.get("invalid_entries")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(t_result, "invalid_entries", [])

    add_check(results, "meta_checklist", case_id, "mutated_t_single_entry_guard_fails", "FAIL", t_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_t_reports_multiple_entries", True, any(row.get("id") == "T108" for row in invalid_rows))

    summary["guard_status"] = t_result.get("status")
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
    return results, summary



def validate_checklist_g_transition_format_case(_base_params):
    meta_quality_module = importlib.import_module("tools.local_regression.run_meta_quality")

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
    meta_quality_module = importlib.import_module("tools.local_regression.run_meta_quality")

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


def validate_checklist_g_transition_sequence_case(_base_params):
    meta_quality_module = importlib.import_module("tools.local_regression.run_meta_quality")

    case_id = "META_CHECKLIST_G_TRANSITION_SEQUENCE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    original_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    g_rows = extract_markdown_table_rows(original_text, "G. 逐項收斂紀錄")
    b26_occurrence_count = sum(1 for cols in g_rows if len(cols) > 1 and cols[1].strip() == "B26")
    target_match_index = 2 if b26_occurrence_count >= 3 else None
    add_check(results, "meta_checklist", case_id, "target_g_row_has_followup_occurrence_for_mutation", True, target_match_index is not None)
    if target_match_index is None:
        return results, summary

    try:
        mutated_text = _replace_markdown_table_row(
            original_text,
            heading="G. 逐項收斂紀錄",
            row_id="B26",
            id_col_idx=1,
            update_cols=lambda cols: cols[:3] + ["PARTIAL -> DONE"] + cols[4:],
            match_index=target_match_index,
        )
    except ValueError:
        add_check(results, "meta_checklist", case_id, "target_g_row_exists_for_mutation", True, False)
        return results, summary

    with tempfile.TemporaryDirectory(prefix="meta_checklist_g_chain_") as temp_dir:
        mutated_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        mutated_path.write_text(mutated_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", mutated_path):
            consistency = meta_quality_module._summarize_checklist_consistency()

    result_by_name = {item.get("name"): item for item in consistency.get("results", [])}
    g_chain_result = result_by_name.get("checklist_g_rows_follow_previous_status_chain", {})
    invalid_rows = g_chain_result.get("invalid_transition_sequence_rows")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(g_chain_result, "invalid_transition_sequence_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_transition_sequence_guard_fails", "FAIL", g_chain_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_transition_sequence_reports_target_row", True, any(row.get("id") == "B26" for row in invalid_rows))

    summary["b26_occurrence_count"] = b26_occurrence_count
    summary["guard_status"] = g_chain_result.get("status")
    summary["invalid_row_ids"] = [row.get("id") for row in invalid_rows]
    summary["target_match_index"] = target_match_index
    return results, summary

def validate_checklist_g_ordering_case(_base_params):
    meta_quality_module = importlib.import_module("tools.local_regression.run_meta_quality")

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
    g_order_result = result_by_name.get("checklist_g_dates_non_decreasing", {})
    invalid_rows = g_order_result.get("invalid_date_rows")
    if invalid_rows is None:
        invalid_rows = _read_summary_value(g_order_result, "invalid_date_rows", [])

    add_check(results, "meta_checklist", case_id, "mutated_g_chronology_guard_fails", "FAIL", g_order_result.get("status"))
    add_check(results, "meta_checklist", case_id, "mutated_g_chronology_reports_invalid_pair", True, bool(invalid_rows))

    same_day_reordered_text = _swap_markdown_table_rows(
        original_text,
        heading="G. 逐項收斂紀錄",
        row_id_a="T01",
        row_id_b="T02",
        id_col_idx=1,
    )
    with tempfile.TemporaryDirectory(prefix="meta_checklist_g_same_day_") as temp_dir:
        same_day_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        same_day_path.write_text(same_day_reordered_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", same_day_path):
            same_day_consistency = meta_quality_module._summarize_checklist_consistency()
    same_day_result = {item.get("name"): item for item in same_day_consistency.get("results", [])}.get(
        "checklist_g_dates_non_decreasing", {}
    )
    add_check(results, "meta_checklist", case_id, "same_day_tracking_id_reorder_is_non_blocking", "PASS", same_day_result.get("status"))

    summary["guard_status"] = g_order_result.get("status")
    summary["invalid_date_rows"] = invalid_rows
    summary["same_day_guard_status"] = same_day_result.get("status")
    return results, summary



def validate_checklist_summary_tables_sorted_by_id_case(_base_params):
    meta_quality_module = importlib.import_module("tools.local_regression.run_meta_quality")

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

    legacy_summary_headings = [
        "### E1. 目前所有 `PARTIAL` 的主表項目摘要",
        "### E2. 目前所有 `TODO` 的主表項目摘要",
        "### E3. 目前所有未完成的建議測試項目摘要",
    ]
    add_check(
        results,
        "meta_checklist",
        case_id,
        "persisted_partial_todo_summary_tables_removed",
        [],
        [heading for heading in legacy_summary_headings if heading in original_text],
    )

    derived_text = _replace_markdown_table_row(
        original_text,
        heading="B2. 長期固定補充契約",
        row_id="B22",
        id_col_idx=0,
        update_cols=lambda cols: cols[:4] + ["PARTIAL"] + cols[5:],
    )
    with tempfile.TemporaryDirectory(prefix="meta_checklist_derived_summary_") as temp_dir:
        derived_path = Path(temp_dir) / "TEST_SUITE_CHECKLIST.md"
        derived_path.write_text(derived_text, encoding="utf-8")
        with patch.object(meta_quality_module, "CHECKLIST_PATH", derived_path):
            derived_consistency = meta_quality_module._summarize_checklist_consistency()
    add_check(
        results,
        "meta_checklist",
        case_id,
        "partial_ids_are_derived_directly_from_main_table",
        True,
        "B22" in derived_consistency.get("partial_ids", []),
    )
    legacy_summary_checks = {
        "checklist_partial_summary_matches_main_table",
        "checklist_todo_summary_matches_main_table",
        "checklist_unfinished_test_summary_matches_convergence_unfinished_records",
    }
    actual_result_names = {item.get("name") for item in derived_consistency.get("results", [])}
    add_check(
        results,
        "meta_checklist",
        case_id,
        "derived_summary_contract_has_no_duplicate_table_sync_checks",
        set(),
        legacy_summary_checks & actual_result_names,
    )

    summary["guard_status"] = order_result.get("status")
    summary["invalid_summary_table_orders"] = invalid_rows
    summary["derived_partial_ids"] = derived_consistency.get("partial_ids", [])
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


def validate_registry_checklist_entry_consistency_case(_base_params):
    case_id = "META_REGISTRY_CHECKLIST_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validator_entries = load_synthetic_registry_entries_from_source(PROJECT_ROOT)
    validator_names = [entry["name"] for entry in validator_entries]
    validator_name_set = set(validator_names)
    imported_validate_names = load_imported_validate_names_from_synthetic_main_entry(PROJECT_ROOT)
    defined_validate_names = load_defined_validate_names_from_synthetic_case_modules(PROJECT_ROOT)
    convergence_statuses = load_convergence_latest_statuses(CHECKLIST_PATH)
    done_test_rows = load_done_test_rows(CHECKLIST_PATH)
    done_b_rows = load_done_b_rows(CHECKLIST_PATH)
    main_statuses = load_main_statuses(load_checklist_tables(CHECKLIST_PATH))

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
        entry_candidates = list(dict.fromkeys(
            re.findall(r"(?:apps|config|core|filters|tools)/[A-Za-z0-9_./-]+\.py", row["entry"])
        ))
        if entry_candidates:
            missing_entry_paths = [
                entry_path
                for entry_path in entry_candidates
                if not (PROJECT_ROOT / entry_path).exists()
            ]
            add_check(
                results,
                "meta_registry",
                case_id,
                f"{row['b_id']}_declared_entry_file_exists",
                True,
                not missing_entry_paths,
                note=(
                    "missing=" + ",".join(missing_entry_paths)
                    if missing_entry_paths
                    else ""
                ),
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


def _write_meta_quality_coverage_reuse_artifacts(
    coverage_dir: Path,
    payload,
    *,
    returncode=0,
    synthetic_fail_count=0,
    synthetic_case_count=99,
    stderr="",
    suite_completed=True,
):
    (coverage_dir / "coverage_synthetic.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (coverage_dir / "coverage_run_info.json").write_text(
        json.dumps(
            {
                "source": "validate_consistency",
                "returncode": int(returncode),
                "stdout": "cached",
                "stderr": str(stderr),
                "timed_out": False,
                "synthetic_fail_count": int(synthetic_fail_count),
                "synthetic_case_count": int(synthetic_case_count),
                "suite_completed": bool(suite_completed),
                "json_generated": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )



def validate_meta_quality_coverage_threshold_uses_target_scope_case(_base_params):
    case_id = "META_QUALITY_COVERAGE_THRESHOLD_USES_TARGET_SCOPE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="meta_cov_target_scope_") as temp_dir:
        run_dir = Path(temp_dir)
        coverage_dir = run_dir / "coverage_artifacts"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        payload = _build_meta_quality_reuse_payload(
            line_percent=5.0,
            branch_percent=4.0,
            critical_line_percent=80.0,
            critical_branch_percent=70.0,
        )
        payload["files"]["tools/optimizer/untracked_helper.py"] = {
            "summary": {
                "covered_lines": 0,
                "num_statements": 1000,
                "percent_covered": 0.0,
                "covered_branches": 0,
                "num_branches": 1000,
            }
        }
        _write_meta_quality_coverage_reuse_artifacts(coverage_dir, payload)
        manifest = {
            "coverage_line_min_percent": 55.0,
            "coverage_branch_min_percent": 50.0,
            "coverage_critical_line_min_percent": 30.0,
            "coverage_critical_branch_min_percent": 25.0,
        }
        coverage_summary = build_coverage_summary(run_dir, manifest)

    with tempfile.TemporaryDirectory(prefix="meta_cov_incomplete_suite_") as temp_dir:
        run_dir = Path(temp_dir)
        coverage_dir = run_dir / "coverage_artifacts"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        incomplete_payload = {"totals": {}, "files": {}}
        _write_meta_quality_coverage_reuse_artifacts(
            coverage_dir,
            incomplete_payload,
            returncode=1,
            synthetic_fail_count=0,
            synthetic_case_count=0,
            stderr="NameError: synthetic fixture aborted before suite completion",
            suite_completed=False,
        )
        incomplete_summary = build_coverage_summary(run_dir, manifest)

    with tempfile.TemporaryDirectory(prefix="meta_cov_completed_failure_") as temp_dir:
        run_dir = Path(temp_dir)
        coverage_dir = run_dir / "coverage_artifacts"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        completed_failure_payload = _build_meta_quality_reuse_payload(
            line_percent=72.0,
            branch_percent=68.0,
            critical_line_percent=80.0,
            critical_branch_percent=70.0,
        )
        _write_meta_quality_coverage_reuse_artifacts(
            coverage_dir,
            completed_failure_payload,
            returncode=1,
            synthetic_fail_count=1,
            synthetic_case_count=99,
            suite_completed=True,
        )
        completed_failure_summary = build_coverage_summary(run_dir, manifest)

    line_gate_result = _summary_result_by_name(coverage_summary["results"], "coverage_line_percent_within_minimum")
    branch_gate_result = _summary_result_by_name(coverage_summary["results"], "coverage_branch_percent_within_minimum")
    totals = coverage_summary.get("totals", {})

    add_check(results, "meta_coverage", case_id, "coverage_threshold_scope_is_declared_targets", "coverage_targets", totals.get("scope"))
    add_check(results, "meta_coverage", case_id, "coverage_target_scope_passes_line_gate", "PASS", line_gate_result.get("status"))
    add_check(results, "meta_coverage", case_id, "coverage_target_scope_passes_branch_gate", "PASS", branch_gate_result.get("status"))
    add_check(results, "meta_coverage", case_id, "coverage_raw_payload_totals_preserved_for_diagnostics", True, totals.get("raw_project_line_percent_covered", 0.0) < 55.0 and totals.get("raw_project_branch_percent_covered", 0.0) < 50.0)
    add_check(results, "meta_coverage", case_id, "coverage_untracked_files_do_not_create_missing_targets", [], coverage_summary.get("missing_targets", []))

    incomplete_results = incomplete_summary["results"]
    add_check(
        results,
        "meta_coverage",
        case_id,
        "coverage_incomplete_suite_is_single_root_failure",
        "FAIL",
        _summary_result_by_name(incomplete_results, "coverage_synthetic_suite_runs_successfully").get("status"),
    )
    for dependent_name in (
        "coverage_overall_nonzero",
        "coverage_line_percent_within_minimum",
        "coverage_branch_percent_within_minimum",
        "coverage_key_targets_present",
        "coverage_key_targets_hit",
        "coverage_critical_files_line_percent_within_minimum",
        "coverage_critical_files_branch_percent_within_minimum",
    ):
        add_check(
            results,
            "meta_coverage",
            case_id,
            f"incomplete_suite_blocks_{dependent_name}",
            "BLOCKED",
            _summary_result_by_name(incomplete_results, dependent_name).get("status"),
        )
    add_check(
        results,
        "meta_coverage",
        case_id,
        "incomplete_suite_does_not_block_static_threshold_policy",
        "PASS",
        _summary_result_by_name(incomplete_results, "coverage_thresholds_respect_formal_floor").get("status"),
    )

    completed_failure_results = completed_failure_summary["results"]
    add_check(
        results,
        "meta_coverage",
        case_id,
        "completed_failing_suite_keeps_synthetic_failure_as_root",
        "FAIL",
        _summary_result_by_name(completed_failure_results, "coverage_synthetic_suite_runs_successfully").get("status"),
    )
    add_check(
        results,
        "meta_coverage",
        case_id,
        "completed_failing_suite_still_evaluates_line_coverage",
        "PASS",
        _summary_result_by_name(completed_failure_results, "coverage_line_percent_within_minimum").get("status"),
    )
    add_check(
        results,
        "meta_coverage",
        case_id,
        "completed_failing_suite_still_evaluates_key_target_coverage",
        "PASS",
        _summary_result_by_name(completed_failure_results, "coverage_key_targets_hit").get("status"),
    )

    partitioned = partition_result_statuses(incomplete_results)
    add_check(
        results,
        "meta_coverage",
        case_id,
        "meta_quality_failure_count_excludes_blocked_dependents",
        ["coverage_synthetic_suite_runs_successfully"],
        partitioned.get("failures"),
    )
    add_check(
        results,
        "meta_coverage",
        case_id,
        "meta_quality_blocked_dependents_are_reported_separately",
        True,
        len(partitioned.get("blocked", [])) >= 7,
    )

    summary["scope"] = totals.get("scope")
    summary["target_line_percent"] = totals.get("line_percent_covered")
    summary["target_branch_percent"] = totals.get("branch_percent_covered")
    summary["raw_project_line_percent"] = totals.get("raw_project_line_percent_covered")
    summary["raw_project_branch_percent"] = totals.get("raw_project_branch_percent_covered")
    return results, summary



def validate_portfolio_core_module_boundary_contract_case(_base_params):
    case_id = "META_PORTFOLIO_CORE_MODULE_BOUNDARY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    import ast as _ast
    from core import portfolio_engine as engine
    from core import portfolio_benchmark as benchmark
    from core import portfolio_ensemble as ensemble
    from core import portfolio_entries as entries
    from core import portfolio_entry_plans as entry_plans
    from core import portfolio_entry_selection as selection
    from core import portfolio_entry_selection_common as selection_common
    from core import portfolio_entry_selection_max_dl as selection_max_dl
    from core import portfolio_levels as levels
    from core import portfolio_replay_support as replay_support
    from core.trade_plans import clone_shadow_position as canonical_clone_shadow_position

    required_paths = [
        "core/portfolio_benchmark.py",
        "core/portfolio_replay_support.py",
        "core/portfolio_levels.py",
        "core/portfolio_ensemble.py",
        "core/portfolio_entry_plans.py",
        "core/portfolio_entry_selection.py",
        "core/portfolio_entry_selection_common.py",
        "core/portfolio_entry_selection_max_dl.py",
    ]
    missing_paths = sorted(path for path in required_paths if not (PROJECT_ROOT / path).is_file())

    engine_source = read_source_text(PROJECT_ROOT / "core" / "portfolio_engine.py")
    engine_tree = _ast.parse(engine_source)
    engine_defs = sorted(
        node.name for node in engine_tree.body
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
    )
    entries_source = read_source_text(PROJECT_ROOT / "core" / "portfolio_entries.py")
    entries_tree = _ast.parse(entries_source)
    entries_defs = sorted(
        node.name for node in entries_tree.body
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
    )
    expected_entries_defs = sorted([
        "_candidate_kind_label",
        "_format_candidate_date",
        "cleanup_extended_signals_for_day",
        "execute_reserved_entries_for_day",
    ])

    owner_expectations = {
        "benchmark": engine._get_benchmark_period_stats.__module__ == benchmark.__name__,
        "replay_snapshot": engine._candidate_replay_snapshot.__module__ == replay_support.__name__,
        "levels": engine._append_portfolio_active_level_rows.__module__ == levels.__name__,
        "ensemble": engine._aggregate_ensemble_candidate_rows.__module__ == ensemble.__name__,
        "entry_plan": entries.build_candidate_plan_seed.__module__ == entry_plans.__name__,
        "selection_router": entries.reorder_candidates_for_resource_aware_quality.__module__ == selection.__name__,
        "selection_common": entries._simulate_reserved_candidate_order.__module__ == selection_common.__name__,
        "selection_max_dl": entries._reorder_resource_aware_continuous_max_dl.__module__ == selection_max_dl.__name__,
        "entry_execution": entries.execute_reserved_entries_for_day.__module__ == entries.__name__,
    }
    alias_expectations = {
        "engine_benchmark_alias": engine._get_benchmark_period_stats is benchmark._get_benchmark_period_stats,
        "engine_ensemble_alias": engine._aggregate_ensemble_candidate_rows is ensemble._aggregate_ensemble_candidate_rows,
        "engine_replay_alias": engine._candidate_replay_snapshot is replay_support._candidate_replay_snapshot,
        "engine_levels_alias": engine._append_portfolio_active_level_rows is levels._append_portfolio_active_level_rows,
        "entries_plan_alias": entries.build_candidate_plan_seed is entry_plans.build_candidate_plan_seed,
        "entries_selection_alias": entries.reorder_candidates_for_resource_aware_quality is selection.reorder_candidates_for_resource_aware_quality,
        "entries_clone_shadow_compat": entries.clone_shadow_position is canonical_clone_shadow_position,
    }

    forbidden_reverse_imports = []
    for rel_path in required_paths:
        source = read_source_text(PROJECT_ROOT / rel_path)
        if "from core.portfolio_engine" in source or "import core.portfolio_engine" in source:
            forbidden_reverse_imports.append(f"{rel_path}:portfolio_engine")
        if rel_path.startswith("core/portfolio_entry_") and (
            "from core.portfolio_entries" in source or "import core.portfolio_entries" in source
        ):
            forbidden_reverse_imports.append(f"{rel_path}:portfolio_entries")

    add_check(results, "meta_contract", case_id, "portfolio_core_split_modules_exist", [], missing_paths)
    add_check(results, "meta_contract", case_id, "portfolio_engine_is_timeline_orchestrator_only", ["run_portfolio_timeline"], engine_defs)
    add_check(results, "meta_contract", case_id, "portfolio_entries_keeps_only_entry_state_transition_defs", expected_entries_defs, entries_defs)
    add_check(results, "meta_contract", case_id, "portfolio_split_canonical_owners_match_responsibility", True, all(owner_expectations.values()), note=str(owner_expectations))
    add_check(results, "meta_contract", case_id, "portfolio_legacy_import_aliases_share_same_function_objects", True, all(alias_expectations.values()), note=str(alias_expectations))
    add_check(results, "meta_contract", case_id, "portfolio_split_has_no_reverse_engine_or_entries_import", [], forbidden_reverse_imports)

    summary["required_module_count"] = len(required_paths)
    summary["engine_defs"] = engine_defs
    summary["entries_defs"] = entries_defs
    summary["owner_expectations"] = owner_expectations
    return results, summary


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
        "config.display_policy": {
            "SYSTEM_SCORE_DISPLAY_MULTIPLIER",
            "build_display_policy_snapshot",
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


def validate_peak_process_memory_tracker_context_management_case(_base_params):
    case_id = "META_PEAK_PROCESS_MEMORY_TRACKER_CONTEXT_MANAGEMENT"
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

        if "with PeakProcessMemoryTracker() as tracker:" not in source:
            missing_with_context_files.append(rel_path)

    from core.runtime_utils import PeakProcessMemoryTracker

    with PeakProcessMemoryTracker() as tracker:
        measured_peak_mb = tracker.snapshot_peak_mb()
        measurement_mode = tracker.measurement_mode
    runtime_utils_source = (PROJECT_ROOT / "core" / "runtime_utils.py").read_text(encoding="utf-8")
    runtime_utils_tree = ast.parse(runtime_utils_source)
    imports_tracemalloc = any(
        (isinstance(node, ast.Import) and any(alias.name == "tracemalloc" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "tracemalloc")
        for node in ast.walk(runtime_utils_tree)
    )

    add_check(results, "meta_contract", case_id, "peak_process_memory_tracker_files_parse", [], syntax_errors)
    add_check(results, "meta_contract", case_id, "peak_process_memory_tracker_manual_lifecycle_forbidden", [], sorted(set(manual_lifecycle_files)))
    add_check(results, "meta_contract", case_id, "peak_process_memory_tracker_uses_with_context", [], sorted(set(missing_with_context_files)))
    add_check(results, "meta_contract", case_id, "peak_process_memory_tracker_reports_positive_peak", True, measured_peak_mb > 0.0)
    add_check(results, "meta_contract", case_id, "peak_process_memory_tracker_declares_rss_mode", "process_peak_rss", measurement_mode)
    add_check(results, "meta_contract", case_id, "formal_memory_tracker_does_not_reenable_tracemalloc", False, imports_tracemalloc)

    summary["tracked_files"] = scanned_files
    summary["measurement_mode"] = measurement_mode
    summary["measured_peak_mb"] = measured_peak_mb
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
    add_check(results, "meta_contract", case_id, "single_backtest_stats_signature_accepts_security_profile", True, "security_profile=None" in source_text)

    backtest_core_path = build_project_absolute_path("core", "backtest_core.py")
    backtest_core_source = backtest_core_path.read_text(encoding="utf-8")
    add_check(results, "meta_contract", case_id, "single_backtest_stats_empty_path_threads_final_date_none", True, "final_date=None" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_empty_path_threads_security_profile", True, "security_profile=resolved_security_profile" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_final_path_threads_dates_last", True, "final_date=Dates[-1]" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_final_path_threads_security_profile", True, "security_profile=resolved_security_profile" in backtest_core_source)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_extended_candidate_uses_threaded_final_date", True, "trade_date=final_date" in source_text)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_extended_candidate_threads_security_profile", True, "security_profile=resolved_security_profile" in source_text)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_preview_resolves_security_profile", True, 'resolved_security_profile = security_profile or (active_extended_signal or {}).get("security_profile")' in source_text)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_stop_preview_uses_shared_initial_stop_helper", True, "stop_loss = calc_initial_stop_from_reference(buy_limit, atr_last, params, ticker=resolved_ticker, security_profile=resolved_security_profile)" in source_text)
    add_check(results, "meta_contract", case_id, "single_backtest_stats_tp_preview_uses_shared_target_helper", True, "tp_price = calc_frozen_target_price(buy_limit, stop_loss, ticker=resolved_ticker, security_profile=resolved_security_profile)" in source_text)
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

    debug_backtest_path, debug_backtest_source = _get_function_source("services/trade_analysis/backtest.py", "run_debug_analysis")
    debug_entry_path, debug_entry_source = _get_function_source("services/trade_analysis/entry_flow.py", "process_debug_entry_for_day")

    add_check(results, "meta_contract", case_id, "debug_backtest_entry_flow_returns_spent_cash", True, "position, active_extended_signal, spent_cash = process_debug_entry_for_day(" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_entry_cash_subtracts_spent_cash", True, "current_capital -= spent_cash" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_passes_ticker_to_entry_flow", True, "ticker=ticker" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_passes_security_profile_to_entry_flow", True, "security_profile=resolved_security_profile" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_signature_accepts_ticker", True, "ticker=None" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_signature_accepts_security_profile", True, "security_profile=None" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_passes_trade_date_to_entry_flow", True, "trade_date=dates[j]" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_signature_accepts_trade_date", True, "trade_date=None" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_threads_ticker_to_normal_signal_state", True, "create_signal_tracking_state(" in debug_entry_source and "buy_limit_prev" in debug_entry_source and "atr_prev" in debug_entry_source and "ticker=ticker" in debug_entry_source and "security_profile=security_profile" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_threads_ticker_to_normal_entry_plan", True, "build_normal_entry_plan(buy_limit_prev, atr_prev, sizing_cap, params, ticker=ticker, security_profile=security_profile, trade_date=effective_trade_date)" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_threads_ticker_to_extended_entry_plan", True, "build_extended_entry_plan_from_signal(" in debug_entry_source and "ticker=ticker" in debug_entry_source and "security_profile=security_profile" in debug_entry_source and "trade_date=effective_trade_date" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_buy_signal_preview_threads_security_profile", True, "entry_plan_preview = build_normal_entry_plan(" in debug_backtest_source and "security_profile=resolved_security_profile" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_latest_raw_signal_preview_threads_security_profile", True, "latest_entry_plan_preview = build_normal_candidate_plan(" in debug_backtest_source and "security_profile=resolved_security_profile" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_backtest_latest_extended_preview_threads_security_profile", True, "latest_extended_preview = build_extended_candidate_plan_from_signal(" in debug_backtest_source and "security_profile=resolved_security_profile" in debug_backtest_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_uses_exact_entry_total_helper", True, "spent_cash = _resolve_display_entry_total(entry_result, qty=entry_plan['qty'], params=params)" in debug_entry_source)
    add_check(results, "meta_contract", case_id, "debug_entry_flow_returns_spent_cash", True, "return position, active_extended_signal, spent_cash" in debug_entry_source)
    summary["source_paths"] = [
        str(debug_backtest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        str(debug_entry_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    ]
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
    entry_plan_seed_path = build_project_absolute_path("core", "portfolio_entry_plans.py")
    entry_plan_seed_source = entry_plan_seed_path.read_text(encoding="utf-8")
    candidate_path = build_project_absolute_path("core", "portfolio_candidates.py")
    candidate_source = candidate_path.read_text(encoding="utf-8")
    fast_data_path = build_project_absolute_path("core", "portfolio_fast_data.py")
    fast_data_source = fast_data_path.read_text(encoding="utf-8")
    entry_plans_path = build_project_absolute_path("core", "entry_plans.py")
    entry_plans_source = entry_plans_path.read_text(encoding="utf-8")
    scanner_processor_path = build_project_absolute_path("services", "scanner", "stock_processor.py")
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
    add_check(results, "meta_contract", case_id, "backtest_core_normal_signal_state_threads_security_profile", True, "create_signal_tracking_state(" in backtest_source and "security_profile=resolved_security_profile" in backtest_source)
    add_check(results, "meta_contract", case_id, "backtest_core_normal_entry_plan_threads_security_profile", True, "entry_plan = build_normal_entry_plan(" in backtest_source and "security_profile=resolved_security_profile" in backtest_source)
    add_check(results, "meta_contract", case_id, "backtest_core_stats_thread_security_profile", True, "build_backtest_stats(" in backtest_source and "security_profile=resolved_security_profile" in backtest_source)
    add_check(results, "meta_contract", case_id, "portfolio_fast_data_pack_preserves_security_profile", True, "'security_profile': df.attrs.get('security_profile')" in fast_data_source)
    add_check(results, "meta_contract", case_id, "portfolio_candidates_read_packed_security_profile", True, "get_fast_security_profile" in candidate_source and "security_profile = get_fast_security_profile(fast_df)" in candidate_source)
    add_check(results, "meta_contract", case_id, "portfolio_candidates_normal_candidate_threads_security_profile", True, "candidate_plan = build_normal_candidate_plan(" in candidate_source and "security_profile=security_profile" in candidate_source)
    add_check(results, "meta_contract", case_id, "portfolio_candidates_normal_signal_state_threads_security_profile", True, "signal_state = create_signal_tracking_state(" in candidate_source and "security_profile=security_profile" in candidate_source)
    add_check(results, "meta_contract", case_id, "portfolio_candidates_extended_candidate_threads_security_profile", True, "candidate_plan = build_extended_candidate_plan_from_signal(" in candidate_source and "security_profile=security_profile" in candidate_source)
    add_check(results, "meta_contract", case_id, "portfolio_entry_seed_preserves_ticker_for_execution", True, "'ticker': candidate_row.get('ticker')" in entry_plan_seed_source)
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


def _capture_test_suite_help_output():
    test_suite_module = importlib.import_module("apps.test_suite")
    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        exit_code = test_suite_module.main(["apps/test_suite.py", "--help"])
    help_text = stdout_buffer.getvalue()
    help_line = next((line.strip() for line in help_text.splitlines() if line.strip().startswith("說明:")), "")
    return exit_code, help_text, help_line


def validate_test_suite_help_text_mentions_stable_theme_tokens_case(_base_params):
    case_id = "META_TEST_SUITE_HELP_TEXT_MENTIONS_STABLE_THEME_TOKENS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_path = build_project_absolute_path("apps", "test_suite.py")
    exit_code, help_text, help_line = _capture_test_suite_help_output()

    add_check(results, "meta_contract", case_id, "test_suite_help_exit_code_zero", True, exit_code == 0)
    add_check(results, "meta_contract", case_id, "test_suite_help_output_present", True, bool(help_text.strip()))
    add_check(results, "meta_contract", case_id, "test_suite_help_text_line_present", True, bool(help_line))
    add_check(results, "meta_contract", case_id, "test_suite_help_text_mentions_trading_consistency_theme", True, "交易口徑一致性" in help_line)
    add_check(results, "meta_contract", case_id, "test_suite_help_text_mentions_conservative_exit_interpretation_theme", True, "保守出場解讀" in help_line)
    add_check(results, "meta_contract", case_id, "test_suite_help_text_mentions_debug_backtest_cash_path_theme", True, "debug-backtest 現金路徑" in help_line)

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
        "tools.local_regression.checklist_contract": {"derive_checklist_state", "load_checklist_tables", "load_done_test_rows"},
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
        _write_meta_quality_coverage_reuse_artifacts(coverage_dir, payload)
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
        _write_meta_quality_coverage_reuse_artifacts(coverage_dir, payload)
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
        _write_meta_quality_coverage_reuse_artifacts(coverage_dir, payload, synthetic_case_count=100)
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
    add_check(results, "meta_contract", case_id, "portfolio_rotation_uses_mark_to_market_helper", True, "ret = _calc_position_mark_to_market_return(pos, pt_y_close, pos_params, trade_date=today)" in source_text)
    add_check(results, "meta_contract", case_id, "portfolio_rotation_has_no_legacy_raw_close_minus_entry_formula", False, "ret = (pt_y_close - pos['entry']) / pos['entry']" in source_text)

    summary["source_path"] = source_path.relative_to(PROJECT_ROOT).as_posix()
    return results, summary




def validate_research_report_contract_freeze_case(_base_params):
    """Persistent Research report/menu contracts stay frozen and cross-profile comparable."""

    from types import SimpleNamespace

    import config.breakout_quality as bq
    from core.research_report_contract import (
        APPROVED_PERSISTENT_REPORT_CONTRACT_FINGERPRINTS,
        MODEL_EXTENSION_SCHEMAS,
        MODEL_STANDARD_COMPARISON,
        MODEL_STANDARD_SOP,
        MODEL_MODE_EXTENSION_SCHEMAS,
        extension_contract,
        mode_extension_contract,
        persistent_report_contract_fingerprints,
        table_contract,
        validate_approved_persistent_report_contracts,
    )
    from services.research import breakout_quality_application as app
    from config.breakout_quality import (
        TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    )

    case_id = "RESEARCH_REPORT_CONTRACT_FREEZE"
    results = []

    def check_true(name, condition, detail=""):
        add_check(results, "synthetic_meta", case_id, name, True, bool(condition), note=detail)

    current = persistent_report_contract_fingerprints()
    errors = validate_approved_persistent_report_contracts()
    check_true(
        "persistent_research_report_contract_fingerprints_match_user_approved_freeze",
        not errors and current == dict(APPROVED_PERSISTENT_REPORT_CONTRACT_FINGERPRINTS),
        detail=str(errors),
    )
    check_true(
        "standard_model_sop_v7_common_section_order_is_contiguous_1_to_6",
        int(MODEL_STANDARD_SOP.version) == 7
        and [(section.number, section.title) for section in MODEL_STANDARD_SOP.sections]
        == [
            (1, "Learnability"),
            (2, "Generalization"),
            (3, "Upside / Downside Alignment"),
            (4, "Top-tail Economic Quality"),
            (5, "Ranking / Boundary"),
            (6, "Evidence Coverage"),
        ],
    )
    check_true(
        "standard_multi_model_comparison_schema_is_derived_from_standard_sop",
        int(MODEL_STANDARD_COMPARISON.version) == 6
        and [(section.number, section.title) for section in MODEL_STANDARD_COMPARISON.sections]
        == [(section.number, section.title) for section in MODEL_STANDARD_SOP.sections]
        and all(
            comparison_table.headers
            == (
                "Model",
                *tuple(
                    column.label
                    for column in standard_table.columns
                    if column.key not in {"split", "comparison"}
                ),
            )
            for standard_section, comparison_section in zip(
                MODEL_STANDARD_SOP.sections, MODEL_STANDARD_COMPARISON.sections
            )
            for standard_table, comparison_table in zip(
                standard_section.tables, comparison_section.tables
            )
        ),
    )

    check_true(
        "rolling_and_robustness_use_mode_extensions_not_duplicate_persistent_report_contracts",
        set(MODEL_MODE_EXTENSION_SCHEMAS) == {"rolling_stability", "robustness_stability"}
        and mode_extension_contract("rolling_stability").title
            == "Rolling-specific Extension｜Fold / Year Stability"
        and mode_extension_contract("robustness_stability").title
            == "Robustness-specific Extension｜Across-seed Stability"
        and "model.rolling_standard_sop" not in current
        and "model.rolling_standard_comparison" not in current,
    )

    learn_headers = table_contract("model.standard_sop", "learnability", "learnability").headers
    alignment_headers = table_contract(
        "model.standard_sop", "upside_downside_alignment", "upside_downside_alignment"
    ).headers
    tail_headers = table_contract(
        "model.standard_sop", "top_tail_economic_quality", "top_tail_economic_quality"
    ).headers
    ranking_headers = table_contract(
        "model.standard_sop", "ranking_boundary", "ranking_boundary"
    ).headers
    check_true(
        "standard_sop_v5_common_columns_match_user_approved_semantics",
        "Top-Bottom Target" in learn_headers
        and alignment_headers == (
            "Split", "Target→MFE rho", "Target→Safety rho", "Score→MFE rho", "Score→Safety rho"
        )
        and "Pred-Safety→Target rho" not in alignment_headers
        and "Pred-Safety→Score rho" not in alignment_headers
        and "Top10 Low-Adverse" in tail_headers
        and "HM/HS" in tail_headers and "HM/LS" in tail_headers
        and "LM/HS" in tail_headers and "LM/LS" in tail_headers
        and "HM/HS ×" not in tail_headers
        and "Top-K Target" not in ranking_headers
        and "Top-K Lift" in ranking_headers
        and "競爭日 / Pool日" in ranking_headers,
        detail=(
            f"learn={learn_headers}; alignment={alignment_headers}; "
            f"tail={tail_headers}; ranking={ranking_headers}"
        ),
    )

    base_metrics = {
        "validation": {
            "group_count": 10,
            "mean_daily_spearman": 0.30,
            "global_spearman_vs_raw_target": 0.31,
            "pairwise_concordance": 0.60,
            "top_score_decile_raw_target_mean": 1.2,
            "bottom_score_decile_raw_target_mean": 0.3,
            "top_k_quality": {
                "top_k": 10, "boundary_width": 3, "ndcg_at_k": 0.70,
                "top_k_raw_target_lift": 0.50, "oracle_top_k_overlap": 0.10,
                "boundary_concordance": 0.53, "boundary_raw_target_gap": 0.10,
                "competition_date_count": 4, "all_date_count": 6,
            },
        },
        "oos": {
            "group_count": 10,
            "mean_daily_spearman": 0.28,
            "global_spearman_vs_raw_target": 0.29,
            "pairwise_concordance": 0.58,
            "top_score_decile_raw_target_mean": 1.1,
            "bottom_score_decile_raw_target_mean": 0.35,
            "top_k_quality": {
                "top_k": 10, "boundary_width": 3, "ndcg_at_k": 0.68,
                "top_k_raw_target_lift": 0.45, "oracle_top_k_overlap": 0.09,
                "boundary_concordance": 0.52, "boundary_raw_target_gap": 0.08,
                "competition_date_count": 5, "all_date_count": 7,
            },
        },
        "breakout_candidate_oos": {
            "group_count": 5,
            "mean_daily_spearman": 0.25,
            "global_spearman_vs_raw_target": 0.26,
            "pairwise_concordance": 0.56,
            "top_score_decile_raw_target_mean": 1.0,
            "bottom_score_decile_raw_target_mean": 0.4,
            "top_k_quality": {
                "top_k": 10, "boundary_width": 3, "ndcg_at_k": 0.75,
                "top_k_raw_target_lift": 0.40, "oracle_top_k_overlap": 0.50,
                "boundary_concordance": 0.54, "boundary_raw_target_gap": 0.12,
                "competition_date_count": 3, "all_date_count": 5,
            },
        },
    }

    def alignment_scope(*, target_mfe, target_safety, score_mfe, score_safety, top_n):
        return {
            "target_to_full_mfe_daily_spearman": target_mfe,
            "target_to_low_adverse_daily_spearman": target_safety,
            "score_to_full_mfe_daily_spearman": score_mfe,
            "score_to_low_adverse_daily_spearman": score_safety,
            "population": {
                "full_mfe_r_mean": 1.0,
                "adverse_r_mean": 0.50,
                "low_adverse_r_mean": -0.50,
                "high_mfe_pct": 50.0,
                "high_safety_pct": 50.0,
                "quadrants": {"hmhs": 25.0, "hmls": 25.0, "lmhs": 25.0, "lmls": 25.0},
            },
            "top_10pct": {
                "n": top_n,
                "full_mfe_r_mean": 2.0,
                "adverse_r_mean": 0.40,
                "low_adverse_r_mean": -0.40,
                "high_mfe_pct": 80.0,
                "high_safety_pct": 70.0,
                "hmhs_pct": 60.0,
                "hmhs_enrichment": 2.40,
                "quadrants": {
                    "hmhs": {"pct": 60.0, "population_pct": 25.0, "enrichment": 2.40},
                    "hmls": {"pct": 20.0, "population_pct": 25.0, "enrichment": 0.80},
                    "lmhs": {"pct": 10.0, "population_pct": 25.0, "enrichment": 0.40},
                    "lmls": {"pct": 10.0, "population_pct": 25.0, "enrichment": 0.40},
                },
            },
        }

    def payload(model_id, objective):
        standard = {
            "schema": "standard_model_sop_v7",
            "evaluation_mode": "forward_oos",
            "training": {"objective": objective},
            "split_metrics": json.loads(json.dumps(base_metrics)),
            "upside_downside_alignment_evaluation": {
                "validation": alignment_scope(
                    target_mfe=0.30, target_safety=0.20, score_mfe=0.28, score_safety=0.18, top_n=2
                ),
                "oos": alignment_scope(
                    target_mfe=0.27, target_safety=0.19, score_mfe=0.25, score_safety=0.17, top_n=2
                ),
                "breakout_candidate_oos": alignment_scope(
                    target_mfe=0.24, target_safety=0.18, score_mfe=0.22, score_safety=0.16, top_n=1
                ),
            },
            "mode_extensions": {},
        }
        return {
            "model_research_id": model_id,
            "training": {"objective": objective},
            "standard_model_sop": standard,
        }

    control_payload = payload("MODEL-CONTROL", TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING)
    joint_payload = payload("MODEL-JOINT", TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING)
    h_only_payload = payload("MODEL-HONLY", TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING)
    joint_min_payload = payload(
        "MODEL-JOINT-MIN",
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    )
    control_payload["safety_raw_mfe_evaluation"] = {
        "validation": {
            "raw_safety": dict(base_metrics["validation"]),
            "raw_mfe": dict(base_metrics["validation"]),
        },
        "oos": {
            "raw_safety": dict(base_metrics["oos"]),
            "raw_mfe": dict(base_metrics["oos"]),
            "model_gate": {
                "predicted_safety_to_raw_mfe_mean_daily_spearman": -0.80,
                "joint_product_to_actual_hmhs_mean_daily_spearman": 0.03,
                "upper_right_s5_m5": {"n": 0},
                "actual_truth_geometry": {
                    "safety_to_mfe_mean_daily_spearman": -0.12,
                    "s5_m5": {"n": 4, "population_pct": 4.0, "independence_enrichment": 0.9},
                    "s4plus_m4plus": {"n": 15, "population_pct": 15.0, "independence_enrichment": 0.95},
                    "actual_joint_geometry": [],
                },
                "predicted_joint_geometry": [],
                "safety_cohorts": [],
            },
        },
        "breakout_candidate_oos": {
            "raw_safety": dict(base_metrics["breakout_candidate_oos"]),
            "raw_mfe": dict(base_metrics["breakout_candidate_oos"]),
        },
    }
    joint_payload["safety_raw_mfe_hmhs_evaluation"] = {
        "validation": {
            "joint_hmhs": {
                "population_hmhs_pct": 22.0,
                "pairwise_concordance": 0.57,
                "global_average_precision": 0.25,
                "top_10pct": {"hmhs_pct": 26.0, "hmhs_enrichment": 1.18},
            },
            "joint_product_control": {
                "pairwise_concordance": 0.56,
                "global_average_precision": 0.24,
                "top_10pct": {"hmhs_pct": 25.0, "hmhs_enrichment": 1.12},
            },
        }
    }
    for key, pct, pair, ap, daily_ap, top10, top20 in (
        ("validation", 22.0, 0.57, 0.25, 0.24, 1.17, 1.10),
        ("oos", 22.0, 0.54, 0.23, 0.22, 1.11, 1.08),
        ("breakout_candidate_oos", 24.0, 0.50, 0.24, 0.23, 1.03, 1.02),
    ):
        h_only_payload["standard_model_sop"]["split_metrics"][key].update({
            "population_hmhs_pct": pct,
            "global_average_precision": ap,
            "mean_daily_average_precision": daily_ap,
            "top_10pct": {"hmhs_pct": pct + 1.0, "hmhs_enrichment": top10},
            "top_20pct": {"hmhs_pct": pct + 0.5, "hmhs_enrichment": top20},
            "pairwise_concordance": pair,
        })
    joint_min_payload["safety_raw_mfe_joint_min_evaluation"] = {
        "validation": {
            "raw_safety": dict(base_metrics["validation"]),
            "raw_mfe": dict(base_metrics["validation"]),
            "joint_min": {
                "population_joint_min_mean": 0.33,
                "mean_daily_spearman": 0.41,
                "pairwise_concordance": 0.63,
                "top_10pct": {
                    "mean_joint_min": 0.61,
                    "mean_safety": 0.72,
                    "mean_mfe": 0.74,
                    "hmhs_pct": 62.0,
                    "hmhs_enrichment": 2.7,
                },
                "top_20pct": {"mean_joint_min": 0.55, "hmhs_enrichment": 2.1},
            },
        }
    }

    rendered = {
        "control": app._render_continuous_ranker_simple_console(control_payload),
        "joint": app._render_continuous_ranker_simple_console(joint_payload),
        "h_only": app._render_continuous_ranker_simple_console(h_only_payload),
        "joint_min": app._render_continuous_ranker_simple_console(joint_min_payload),
    }
    standard_lines = {
        key: "\n".join(line for line in text.splitlines() if line.startswith("標準模型 SOP｜"))
        for key, text in rendered.items()
    }
    extension_expectations = {
        "multi_head_learnability": ("control", "MODEL-CONTROL"),
        "truth_prediction_geometry": ("control", "MODEL-CONTROL"),
        "direct_hmhs_joint_retrieval": ("joint", "MODEL-JOINT"),
        "direct_hmhs_h_only": ("h_only", "MODEL-HONLY"),
        "joint_min_retrieval": ("joint_min", "MODEL-JOINT-MIN"),
    }
    check_true(
        "model_extension_registry_is_fully_covered_by_capability_payloads",
        set(extension_expectations) == set(MODEL_EXTENSION_SCHEMAS),
        detail=f"covered={sorted(extension_expectations)}, registered={sorted(MODEL_EXTENSION_SCHEMAS)}",
    )
    check_true(
        "model_extensions_cannot_mutate_standard_model_sop_namespace",
        all(
            extension_contract(extension_id).title not in standard_lines[render_key]
            and f"Model-specific Extension｜{model_id}｜{extension_contract(extension_id).title}"
            in rendered[render_key]
            for extension_id, (render_key, model_id) in extension_expectations.items()
        )
        and "Δ HM/HS Pair" not in standard_lines["h_only"],
    )
    check_true(
        "cross_profile_standard_v6_common_headers_and_section_order_are_invariant",
        all(
            "Top-Bottom Target" in text
            and "Validation → OOS" in text
            and "Primary score" not in text
            and "標準模型 SOP｜3. Upside / Downside Alignment" in text
            and "標準模型 SOP｜4. Top-tail Economic Quality" in text
            and "標準模型 SOP｜5. Ranking / Boundary" in text
            and "標準模型 SOP｜6. Evidence Coverage" in text
            and "標準模型 SOP｜3. Multi-head Learnability" not in text
            and "標準模型 SOP｜6. Truth / Prediction Geometry" not in text
            and "Target→Safety rho" in text
            and "Score→Safety rho" in text
            and "Pred-Safety→Target rho" not in text
            and "Pred-Safety→Score rho" not in text
            and "Top10 Low-Adverse" in text
            and "60.00% (2.40×)" in text
            and "HM/LS" in text and "LM/HS" in text and "LM/LS" in text
            and "Top-K Target" not in text
            and "Top-K Lift" in text
            and "競爭日 / Pool日" in text
            and text.find("標準模型 SOP｜6. Evidence Coverage")
                > text.find("標準模型 SOP｜5. Ranking / Boundary")
            for text in rendered.values()
        ),
    )

    comparison_text = app._render_standard_model_comparison(
        [
            {"model_id": "MODEL-A", "payload": control_payload},
            {"model_id": "MODEL-B", "payload": control_payload},
            {"model_id": "MODEL-C", "payload": control_payload},
        ],
        target="console",
    )
    check_true(
        "multi_model_comparison_uses_single_section_title_with_oos_then_breakout_tables",
        all(model in comparison_text for model in ("MODEL-A", "MODEL-B", "MODEL-C"))
        and comparison_text.count("模型比較 SOP｜1. Learnability") == 1
        and "模型比較 SOP｜1. Learnability｜Forward OOS" not in comparison_text
        and "模型比較 SOP｜1. Learnability｜Breakout slice" not in comparison_text
        and comparison_text.count("模型比較 SOP｜3. Upside / Downside Alignment") == 1
        and comparison_text.count("模型比較 SOP｜4. Top-tail Economic Quality") == 1
        and comparison_text.count("模型比較 SOP｜5. Ranking / Boundary") == 1
        and "Forward OOS" in comparison_text
        and "Breakout slice" in comparison_text
        and "Validation → OOS" in comparison_text
        and "OOS → Breakout slice" in comparison_text
        and "Comparison" not in comparison_text
        and "Target→Safety rho" in comparison_text
        and "Top10 Low-Adverse" in comparison_text
        and "Top-K Lift" in comparison_text
        and "競爭日 / Pool日" in comparison_text
        and "模型比較 SOP｜6. Evidence Coverage" in comparison_text
        and comparison_text.find("模型比較 SOP｜1. Learnability")
            < comparison_text.find("模型比較 SOP｜2. Generalization")
            < comparison_text.find("模型比較 SOP｜3. Upside / Downside Alignment")
            < comparison_text.find("模型比較 SOP｜4. Top-tail Economic Quality")
            < comparison_text.find("模型比較 SOP｜5. Ranking / Boundary")
            < comparison_text.find("模型比較 SOP｜6. Evidence Coverage"),
    )

    section3 = comparison_text.split("模型比較 SOP｜3. Upside / Downside Alignment", 1)[1].split("模型比較 SOP｜4. Top-tail Economic Quality", 1)[0]
    section4 = comparison_text.split("模型比較 SOP｜4. Top-tail Economic Quality", 1)[1].split("模型比較 SOP｜5. Ranking / Boundary", 1)[0]
    check_true(
        "multi_model_comparison_never_silently_drops_a_configured_model_from_common_sections",
        all(section3.count(model) == 2 for model in ("MODEL-A", "MODEL-B", "MODEL-C"))
        and all(section4.count(model) == 2 for model in ("MODEL-A", "MODEL-B", "MODEL-C")),
        detail=f"section3={section3}; section4={section4}",
    )

    comparison_payloads = []
    for idx, model_id in enumerate(("MODEL-LOW", "MODEL-MID", "MODEL-HIGH")):
        candidate = json.loads(json.dumps(control_payload))
        bump = idx * 0.01
        for split_key in ("oos", "breakout_candidate_oos"):
            candidate["standard_model_sop"]["split_metrics"][split_key]["mean_daily_spearman"] += bump
            candidate["standard_model_sop"]["split_metrics"][split_key]["global_spearman_vs_raw_target"] += bump
        comparison_payloads.append({"model_id": model_id, "payload": candidate})
    comparison_markdown = app._render_standard_model_comparison(
        comparison_payloads,
        target="markdown",
    )
    check_true(
        "multi_model_comparison_uses_best_green_worst_red_blue_titles_light_yellow_subtitles_and_existing_status_palette",
        "#42A5F5" in comparison_markdown
        and "#188038" in comparison_markdown
        and "#C62828" in comparison_markdown
        and "#D6B53A" in comparison_markdown
        and "### <span" in comparison_markdown
        and "Forward OOS</span>" in comparison_markdown
        and "Breakout slice</span>" in comparison_markdown
        and "Validation → OOS</span>" in comparison_markdown
        and "OOS → Breakout slice</span>" in comparison_markdown
        and "| Model | Groups | Daily rho |" in comparison_markdown
        and "| Model | Δ Daily rho | Δ Pair | Δ Top-Bottom |" in comparison_markdown,
    )

    rolling_payload = json.loads(json.dumps(control_payload))
    rolling_payload.pop("safety_raw_mfe_evaluation", None)
    rolling_standard = rolling_payload["standard_model_sop"]
    rolling_standard["evaluation_mode"] = "rolling_oos"
    rolling_standard["mode_extensions"] = {
        "rolling": {
            "fold_count": 5,
            "fold_months": 12,
            "direction_summary": {
                "valid_year_count": 5,
                "positive_spearman_year_count": 4,
                "positive_spread_year_count": 3,
            },
            "fold_drift": {
                "max_adjacent_mean_shift_in_pooled_std": 0.42,
                "drift_flag": False,
            },
        }
    }
    rolling_simple = app._render_continuous_ranker_simple_console(rolling_payload)
    rolling_comparison = app._render_standard_model_comparison(
        [
            {"model_id": "ROLL-A", "payload": rolling_payload},
            {"model_id": "ROLL-B", "payload": rolling_payload},
        ],
        target="console",
    )
    check_true(
        "rolling_reports_use_same_common_sop_and_append_only_fold_year_stability_extension",
        "標準模型 SOP｜1. Learnability" in rolling_simple
        and "標準模型 SOP｜6. Evidence Coverage" in rolling_simple
        and "validation" in rolling_simple.lower()
        and "Validation → OOS" in rolling_simple
        and "OOS → Breakout slice" in rolling_simple
        and "Rolling-specific Extension｜Fold / Year Stability" in rolling_simple
        and rolling_comparison.count("模型比較 SOP｜1. Learnability") == 1
        and "Rolling OOS" in rolling_comparison
        and "Breakout slice" in rolling_comparison
        and "Validation → OOS" in rolling_comparison
        and "OOS → Breakout slice" in rolling_comparison
        and "Rolling-specific Extension｜Fold / Year Stability" in rolling_comparison
        and "Fold count" in rolling_comparison
        and all(model in rolling_comparison for model in ("ROLL-A", "ROLL-B")),
        detail=rolling_comparison,
    )

    from services.breakout_quality.standard_model_sop import aggregate_standard_model_sop_robustness
    seed_a = json.loads(json.dumps(control_payload["standard_model_sop"]))
    seed_b = json.loads(json.dumps(control_payload["standard_model_sop"]))
    seed_b["split_metrics"]["oos"]["mean_daily_spearman"] = 0.32
    seed_b["split_metrics"]["oos"]["pairwise_concordance"] = 0.62
    rolling_seed_a = json.loads(json.dumps(rolling_standard))
    rolling_seed_b = json.loads(json.dumps(rolling_standard))
    rolling_seed_a["mode_extensions"]["rolling"]["fold_drift"].update({
        "criterion": "adjacent fold score mean shift >= 1.0 pooled score standard deviation",
        "drift_flag": False,
        "flagged_folds": [],
    })
    rolling_seed_b["mode_extensions"]["rolling"]["fold_drift"].update({
        "criterion": "adjacent fold score mean shift >= 1.0 pooled score standard deviation",
        "max_adjacent_mean_shift_in_pooled_std": 0.84,
        "drift_flag": True,
        "flagged_folds": ["fold_20240101_20241231", "fold_20220101_20221231"],
    })
    rolling_robust = aggregate_standard_model_sop_robustness(
        [rolling_seed_a, rolling_seed_b], seeds=(11, 22)
    )
    rolling_drift = dict((rolling_robust.get("mode_extensions") or {}).get("rolling", {}).get("fold_drift") or {})
    check_true(
        "rolling_robustness_aggregates_seed_dependent_flagged_folds_without_contract_drift",
        rolling_drift.get("flagged_folds")
        == ["fold_20220101_20221231", "fold_20240101_20241231"]
        and rolling_drift.get("drift_flag") is True
        and abs(float(rolling_drift.get("max_adjacent_mean_shift_in_pooled_std")) - 0.63) < 1e-12
        and rolling_drift.get("criterion")
        == "adjacent fold score mean shift >= 1.0 pooled score standard deviation",
        detail=str(rolling_drift),
    )

    aggregated = aggregate_standard_model_sop_robustness([seed_a, seed_b], seeds=(11, 22))
    robust_payload = {"model_research_id": "ROBUST-A", "standard_model_sop": aggregated}
    robust_comparison = app._render_standard_model_comparison(
        [
            {"model_id": "ROBUST-A", "payload": robust_payload},
            {"model_id": "ROBUST-B", "payload": robust_payload},
        ],
        target="console",
    )
    check_true(
        "robustness_reports_keep_same_common_sop_and_append_only_across_seed_extension",
        "模型比較 SOP｜1. Learnability" in robust_comparison
        and "模型比較 SOP｜6. Evidence Coverage" in robust_comparison
        and "Validation → OOS" in robust_comparison
        and "OOS → Breakout slice" in robust_comparison
        and "Robustness-specific Extension｜Across-seed Stability" in robust_comparison
        and "OOS Daily rho σ" in robust_comparison
        and all(model in robust_comparison for model in ("ROBUST-A", "ROBUST-B")),
        detail=robust_comparison,
    )

    from services.breakout_quality.train_daily_ranker import _render_markdown as _render_detailed_model_markdown
    detailed_payload = {
        "experiment": "synthetic",
        "experiment_profile": "synthetic",
        "model_research_id": "MODEL-SYNTHETIC",
        "training": {
            "sample_scope": "daily_universal",
            "target": "synthetic_target",
            "selected_epoch": 1,
            "objective": "daily_pairwise_ranking",
        },
        "score_semantic_id": "synthetic_score",
        "source_dataset": {},
        "target_manifest": {},
        "split_metrics": {
            "validation": dict(base_metrics["validation"]),
            "oos": dict(base_metrics["oos"]),
        },
    }
    detailed_markdown = _render_detailed_model_markdown(detailed_payload)
    check_true(
        "detailed_standard_model_report_uses_validation_to_oos_primary_generalization_label",
        "Validation → OOS" in detailed_markdown
        and "Validation → Forward OOS" not in detailed_markdown
        and "Primary score" not in detailed_markdown,
    )

    # Standard SOP must not inject MR-13M Pred-Safety as a reporting-only dependency.
    captured = {}

    class _Settings:
        filter_id = "breakout_quality_v1"
        model_architecture = "inception_time_v1"
        experiment_profile = "synthetic_profile"

    sentinel_plan = object()

    def _capture_plan(*args, **kwargs):
        captured.update(kwargs)
        return sentinel_plan

    with patch.object(app, "collect_model_upstream_preparation_plan", side_effect=_capture_plan):
        actual_plan = app._collect_continuous_research_input_plan(_Settings())
    check_true(
        "standard_sop_has_no_model_specific_pred_safety_reporting_dependency",
        actual_plan is sentinel_plan and "include_standard_report_references" not in captured,
        detail=str(captured),
    )

    # [1][1]: validated model/manifest/report plus complete current Standard-SOP
    # evidence is reusable; a missing OOS score CSV alone must not force retraining.
    reusable_contract = SimpleNamespace(seed=42, report=control_payload)
    settings = SimpleNamespace(
        filter_id="breakout_quality_v1",
        model_architecture="inception_time_v1",
        experiment_profile="synthetic_profile",
        seed=42,
    )
    with patch.object(
        app, "load_continuous_ranker_oos_contract", return_value=reusable_contract
    ) as report_loader:
        loaded, reason = app._load_reusable_continuous_forward_contract(settings)
    check_true(
        "report_reuse_validates_model_manifest_report_without_requiring_oos_score_file",
        loaded is reusable_contract
        and reason is None
        and report_loader.call_args.kwargs.get("require_scores") is False,
        detail=str(report_loader.call_args),
    )
    with patch.object(app, "_load_reusable_continuous_forward_contract", return_value=(reusable_contract, None)), \
         patch.object(app, "_run_command", side_effect=AssertionError("REUSE path must not train")):
        code, contract, action = app._ensure_continuous_forward_model_report(
            "synthetic", model_id="MODEL-REUSE", settings=settings,
            prompt_for_build=False,
        )
    check_true(
        "forward_standard_sop_reuses_valid_complete_model_report_without_training",
        code == 0 and contract is reusable_contract and action == "REUSE",
    )

    incomplete_payload = json.loads(json.dumps(control_payload))
    incomplete_payload["standard_model_sop"].pop("upside_downside_alignment_evaluation", None)
    incomplete_contract = SimpleNamespace(seed=42, report=incomplete_payload)
    with patch.object(
        app, "load_continuous_ranker_oos_contract", return_value=incomplete_contract
    ):
        loaded, reason = app._load_reusable_continuous_forward_contract(settings)
    check_true(
        "standard_sop_missing_common_evidence_is_not_reusable_and_requires_canonical_rebuild",
        loaded is None
        and "Standard SOP共通evidence不完整" in str(reason)
        and "upside_downside_alignment" in str(reason),
        detail=str(reason),
    )

    # Missing/stale/incomplete artifact contract automatically takes canonical BUILD path.
    plan = SimpleNamespace(blocked=False)
    with patch.object(
        app, "_load_reusable_continuous_forward_contract",
        side_effect=[(None, "synthetic missing"), (reusable_contract, None)],
    ), patch.object(app, "_collect_continuous_research_input_plan", return_value=plan), \
         patch.object(app, "_render_continuous_research_input_plan", return_value=None), \
         patch.object(app, "_prepare_continuous_research_inputs", return_value=0), \
         patch.object(app, "_run_command", return_value=0) as train_call, \
         patch.object(app, "_clear_continuous_forward_reuse_caches", return_value=None):
        code, contract, action = app._ensure_continuous_forward_model_report(
            "synthetic", model_id="MODEL-BUILD", settings=settings,
            prompt_for_build=False,
        )
    check_true(
        "forward_standard_sop_auto_builds_only_when_complete_contract_is_not_reusable",
        code == 0 and contract is reusable_contract and action == "BUILD"
        and train_call.call_count == 1
        and train_call.call_args.args[0] == "train-continuous-ranker",
    )

    # Current requested comparison set is H / AF / AH, while the generic config accepts 3+ arms.
    current_pairs = tuple(bq.get_breakout_quality_model_test_settings().model_profiles)
    check_true(
        "current_standard_model_comparison_set_is_requested_h_af_ah",
        current_pairs == (
            ("MR-13H", bq.DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE),
            ("MR-13AF", bq.DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE),
            ("MR-13AH", bq.DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE),
        ),
        detail=str(current_pairs),
    )

    # Comparison config accepts 3+ arbitrary valid model/profile pairs rather than a fixed two-arm flow.
    extra_profile = bq.DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    extra_id = str(bq.get_continuous_ranker_research_spec(extra_profile).model_research_id)
    synthetic_pairs = (*current_pairs, (extra_id, extra_profile))
    with patch.object(bq, "BREAKOUT_QUALITY_MODEL_TEST_PROFILES", synthetic_pairs):
        resolved_shared = bq.get_breakout_quality_model_test_settings()
        resolved = bq.get_breakout_quality_standard_model_comparison_settings()
    check_true(
        "standard_model_comparison_config_supports_more_than_three_models",
        len(resolved.model_profiles) == len(synthetic_pairs) == 4
        and tuple(resolved.model_profiles) == tuple(synthetic_pairs)
        and tuple(resolved_shared.model_profiles) == tuple(synthetic_pairs),
        detail=str(resolved.model_profiles),
    )

    return results, {"ticker": case_id, "synthetic": True, "training_performed": False}

