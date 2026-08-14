from __future__ import annotations

from .synthetic_breakout_quality_support import add_check, tempfile, Path

def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import (
        AUDIT_OUTPUT_ROOT,
        get_audit_definitions,
        get_audit_module_ids,
        get_enabled_audit_definitions,
        validate_audit_config,
    )
    from tools.audit.catalog import validate_audit_catalog
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
        "formal_audit_config_has_no_stale_completed_decision_audits",
        True,
        bool(
            get_audit_module_ids() == ("breakout_quality",)
            and len(definitions) == 0
            and len(enabled) == 0
            and AUDIT_OUTPUT_ROOT == "outputs/audit"
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        status_payload = collect_audit_status(
            module_id="breakout_quality",
            project_root=Path(tmp),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "formal_audit_runner_reports_disabled_when_no_current_decision_audit_exists",
        ("DISABLED", 0),
        (status_payload.get("overall_status"), len(status_payload.get("rows") or [])),
    )

    summary["workflow"] = "current_minimal_audit_framework"
    return results, summary
