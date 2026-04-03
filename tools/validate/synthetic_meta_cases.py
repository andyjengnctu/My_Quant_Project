from pathlib import Path
import importlib
import re
import shlex
from unittest.mock import patch

from .checks import add_check


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_PATH = PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md"
CMD_PATH = PROJECT_ROOT / "doc" / "CMD.md"

from .meta_contracts import (
    extract_markdown_table_rows,
    load_defined_validate_names_from_synthetic_case_modules,
    load_imported_validate_names_from_synthetic_main_entry,
    summarize_no_reverse_app_import_contract,
    summarize_no_top_level_import_cycles_contract,
    summarize_single_formal_test_entry_contract,
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


def _load_done_d_rows():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    rows = extract_markdown_table_rows(text, "F2. 目前所有 `DONE` 的建議測試項目摘要")
    parsed = []
    for cols in rows:
        if len(cols) < 4:
            continue
        parsed.append(
            {
                "id": cols[0],
                "name": cols[1].replace("`", "").strip(),
                "b_id": cols[2],
                "done_date": cols[3],
            }
        )
    return parsed


def _load_done_b_rows():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    rows = extract_markdown_table_rows(text, "F1. 目前所有 `DONE` 的主表項目摘要")
    parsed = []
    for cols in rows:
        if len(cols) < 5:
            continue
        parsed.append(
            {
                "kind": cols[0],
                "b_id": cols[1],
                "item": cols[2],
                "entry": cols[3],
                "done_date": cols[4],
            }
        )
    return parsed


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


def validate_registry_checklist_entry_consistency_case(_base_params):
    from tools.validate.synthetic_cases import get_synthetic_validators

    case_id = "META_REGISTRY_CHECKLIST_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validators = get_synthetic_validators()
    validator_names = [validator.__name__ for validator in validators]
    validator_name_set = set(validator_names)
    imported_validate_names = load_imported_validate_names_from_synthetic_main_entry(PROJECT_ROOT)
    defined_validate_names = load_defined_validate_names_from_synthetic_case_modules(PROJECT_ROOT)
    convergence_statuses = _load_convergence_latest_statuses()
    done_d_rows = _load_done_d_rows()
    done_b_rows = _load_done_b_rows()
    main_statuses = _load_main_table_statuses()

    add_check(results, "meta_registry", case_id, "validator_registry_not_empty", True, len(validators) > 0)
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
        entry_path = row["entry"].split(",")[0].strip().strip("`")
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
    summary["validator_count"] = len(validators)
    summary["missing_imported_names"] = missing_imported_names
    summary["missing_defined_names"] = missing_defined_names
    summary["missing_done_d_validator_names"] = missing_done_d_validator_names
    return results, summary



def _count_failures(results):
    return sum(1 for row in results if row.get("status") == "FAIL")


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
