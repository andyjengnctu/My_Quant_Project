from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import patch

from .checks import bind_checks

from .synthetic_breakout_quality_support import add_check


def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import (
        AUDIT_OUTPUT_ROOT,
        get_audit_definitions,
        get_audit_module_ids,
        get_enabled_audit_definitions,
        validate_audit_config,
    )
    from services.audit.catalog import (
        get_domain_cli_commands,
        validate_audit_catalog,
    )
    from services.audit.runner import collect_audit_status

    case_id = "AUDIT_FRAMEWORK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    validate_audit_config()
    definitions = get_audit_definitions("breakout_quality")
    enabled = get_enabled_audit_definitions("breakout_quality")
    validate_audit_catalog(definitions)

    check_true(
        "formal_audit_config_is_config_driven_and_uses_stable_output_root",
        bool(
                    get_audit_module_ids()[0] == "breakout_quality"
                    and len(definitions) >= len(enabled)
                    and AUDIT_OUTPUT_ROOT == "outputs/audit"
                ),
    )

    status = collect_audit_status(module_id="breakout_quality")
    expected_overall = (
        "DISABLED"
        if not enabled
        else "READY"
        if all(
            row.get("status") == "READY"
            for row in (status.get("rows") or [])
            if bool(row.get("enabled"))
        )
        else "BLOCKED"
    )
    check(
        "audit_runner_overall_status_follows_current_enabled_definitions_without_fixed_audit_id",
        expected_overall,
        status.get("overall_status"),
    )

    commands = get_domain_cli_commands("breakout_quality")
    check_true(
        "research_audit_utilities_remain_catalogued_alongside_formal_gate",
        bool("regime-audit" in commands and "audit-point-in-time-scores" in commands),
    )

    import apps.research as research_app

    audit_menu_console = io.StringIO()
    with patch.object(research_app, "get_enabled_audit_definitions", return_value=()), patch(
        "builtins.input", return_value="0"
    ), redirect_stdout(audit_menu_console):
        audit_menu_rc = research_app._audit_menu()
    audit_menu_text = audit_menu_console.getvalue()
    check_true(
        "research_audit_menu_hides_non_executable_run_action_when_no_formal_audit_is_enabled",
        audit_menu_rc == 0
        and "目前沒有啟用的正式 Audit" in audit_menu_text
        and "執行目前 Audit 設定" not in audit_menu_text,
    )

    summary["workflow"] = "config_driven_formal_audit_topology"
    return results, summary
