from __future__ import annotations

from .synthetic_breakout_quality_support import add_check


def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import (
        AUDIT_OUTPUT_ROOT,
        get_audit_definitions,
        get_audit_module_ids,
        get_enabled_audit_definitions,
        validate_audit_config,
    )
    from tools.audit.catalog import (
        get_domain_cli_commands,
        validate_audit_catalog,
    )
    from tools.audit.runner import collect_audit_status

    case_id = "AUDIT_FRAMEWORK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validate_audit_config()
    definitions = get_audit_definitions("breakout_quality")
    enabled = get_enabled_audit_definitions("breakout_quality")
    validate_audit_catalog(definitions)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "formal_audit_config_is_config_driven_and_uses_stable_output_root",
        True,
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
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "audit_runner_overall_status_follows_current_enabled_definitions_without_fixed_audit_id",
        expected_overall,
        status.get("overall_status"),
    )

    commands = get_domain_cli_commands("breakout_quality")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "research_audit_utilities_remain_catalogued_alongside_formal_gate",
        True,
        bool("regime-audit" in commands and "audit-point-in-time-scores" in commands),
    )

    summary["workflow"] = "config_driven_formal_audit_topology"
    return results, summary
