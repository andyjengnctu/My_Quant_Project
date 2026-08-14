from __future__ import annotations

from .synthetic_breakout_quality_support import add_check, math, tempfile, Path

def validate_breakout_quality_minimum_repair_mechanism_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_MINIMUM_REPAIR_MECHANISM_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config.audit import get_audit_definitions
    from tools.audit.catalog import get_audit_entry
    from tools.audit.breakout_quality.minimum_repair_mechanism import (
        _conclusion,
        collect_minimum_repair_mechanism_status,
    )

    definitions = {item.audit_id: item for item in get_audit_definitions("breakout_quality")}
    definition = definitions["mr13e-minimum-repair-mechanism"]
    entry = get_audit_entry("minimum_repair_mechanism")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "minimum_repair_mechanism_audit_is_active_config_driven_read_only_scalable_search_certificate_step",
        True,
        bool(
            len(definitions) == 1
            and definition.enabled
            and definition.audit_type == "minimum_repair_mechanism"
            and definition.dimensions.get("exact_one_swap_certificate") is True
            and definition.dimensions.get("multi_step_path_unresolved") is True
            and definition.outcomes.get("frozen_score_only_search_certificate") is True
            and definition.outcomes.get("no_numeric_threshold") is True
            and entry.formal
            and entry.read_only
            and entry.module == "tools.audit.breakout_quality.minimum_repair_mechanism"
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        blocked = collect_minimum_repair_mechanism_status(definition, project_root=Path(tmp))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "minimum_repair_mechanism_audit_blocks_cleanly_without_search_certificate_sidecars",
        True,
        blocked.get("status") == "BLOCKED",
    )

    def arm(*, one_step, multi_step, fallback, one_contrib, multi_contrib, fallback_contrib):
        return {
            "status": "AVAILABLE",
            "classification_counts": {
                "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT": int(one_step),
                "MULTI_STEP_GREEDY_PATH_UNRESOLVED": int(multi_step),
                "BASELINE_FALLBACK_AFTER_GREEDY_REPAIR": int(fallback),
            },
            "class_target_delta_contribution": {
                "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT": float(one_contrib),
                "MULTI_STEP_GREEDY_PATH_UNRESOLVED": float(multi_contrib),
                "BASELINE_FALLBACK_AFTER_GREEDY_REPAIR": float(fallback_contrib),
            },
        }

    payload = {
        "phases": {
            "forward_oos": {
                "arms": {
                    "C20": arm(one_step=4, multi_step=2, fallback=0, one_contrib=-0.10, multi_contrib=-0.05, fallback_contrib=0.0),
                    "C29": arm(one_step=3, multi_step=3, fallback=0, one_contrib=-0.15, multi_contrib=-0.20, fallback_contrib=0.0),
                    "C36": arm(one_step=5, multi_step=2, fallback=0, one_contrib=-0.40, multi_contrib=-0.10, fallback_contrib=0.0),
                }
            }
        }
    }
    conclusion = _conclusion(payload)
    extra = dict(conclusion.get("mr13e_minus_mr12b_class_contribution_r") or {})
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "minimum_repair_conclusion_separates_exact_one_swap_resource_days_from_multi_step_unresolved_contribution_without_threshold",
        True,
        bool(
            conclusion.get("classification") == "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT_DOMINATES"
            and conclusion.get("dominant_class") == "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT"
            and math.isclose(
                float(extra.get("EXACT_ONE_SWAP_RESOURCE_CONSTRAINT")),
                -0.30,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(extra.get("MULTI_STEP_GREEDY_PATH_UNRESOLVED")),
                -0.05,
                abs_tol=1e-12,
            )
        ),
    )

    summary["workflow"] = "mr13e_minimum_repair_mechanism_audit"
    return results, summary

def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import (
        AUDIT_OUTPUT_ROOT,
        get_audit_definitions,
        get_audit_module_ids,
        get_enabled_audit_definitions,
        validate_audit_config,
    )
    from tools.audit.catalog import get_audit_entry, validate_audit_catalog
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
        "formal_audit_config_contains_only_current_decision_relevant_definition",
        True,
        bool(
            get_audit_module_ids() == ("breakout_quality",)
            and len(definitions) == 1
            and len(enabled) == 1
            and definitions[0].audit_type == "minimum_repair_mechanism"
            and AUDIT_OUTPUT_ROOT == "outputs/audit"
        ),
    )
    entry = get_audit_entry(definitions[0].audit_type)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "current_formal_audit_catalog_entry_is_read_only_and_loadable",
        True,
        bool(
            entry.formal
            and entry.read_only
            and callable(entry.load_status_handler())
            and callable(entry.load_run_handler())
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
        "formal_audit_runner_reports_current_audit_without_historical_handlers",
        True,
        bool(
            len(status_payload.get("rows") or []) == 1
            and status_payload["rows"][0].get("audit_id") == definitions[0].audit_id
            and status_payload["rows"][0].get("status") == "BLOCKED"
        ),
    )

    summary["workflow"] = "current_minimal_audit_framework"
    return results, summary
