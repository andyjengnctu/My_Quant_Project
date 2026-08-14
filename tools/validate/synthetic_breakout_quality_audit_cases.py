from __future__ import annotations

from .synthetic_breakout_quality_support import add_check, math, tempfile, Path

def validate_breakout_quality_direct_r_calibration_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_DIRECT_R_CALIBRATION_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    import pandas as pd

    from config.audit import get_audit_definitions
    from tools.audit.catalog import get_audit_entry
    from tools.audit.breakout_quality.direct_r_calibration import (
        _audit_metrics,
        collect_direct_r_calibration_status,
    )

    definitions = {item.audit_id: item for item in get_audit_definitions("breakout_quality")}
    definition = definitions["mr13f-direct-r-calibration"]
    entry = get_audit_entry("direct_r_calibration")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "direct_r_calibration_audit_is_single_active_config_driven_read_only_posthoc_step",
        True,
        bool(
            len(definitions) == 1
            and definition.enabled
            and definition.audit_type == "direct_r_calibration"
            and definition.dimensions.get("frozen_constant_baseline") == "selection_inner_validation_target_mean"
            and definition.outcomes.get("oos_target_posthoc_only") is True
            and definition.outcomes.get("posthoc_fit_runtime_eligible") is False
            and definition.outcomes.get("no_numeric_gate_threshold") is True
            and entry.formal
            and entry.read_only
            and entry.module == "tools.audit.breakout_quality.direct_r_calibration"
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        blocked = collect_direct_r_calibration_status(definition, project_root=Path(tmp))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "direct_r_calibration_audit_blocks_cleanly_without_frozen_forward_artifacts",
        True,
        blocked.get("status") == "BLOCKED",
    )

    frame = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "date": ["2021-01-04"] * 4,
        "predicted_r": [-1.0, 0.0, 1.0, 2.0],
        "target_raw_r": [-0.5, 0.5, 1.5, 2.5],
    })
    metrics, buckets, signs = _audit_metrics(
        frame,
        frozen_constant_r=0.0,
        huber_delta_r=1.0,
    )
    line = dict(metrics.get("posthoc_calibration_line") or {})
    conclusion = dict(metrics.get("conclusion") or {})
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "direct_r_calibration_metrics_compare_frozen_constant_and_recover_posthoc_affine_structure_without_runtime_fit",
        True,
        bool(
            math.isclose(float(metrics["model_error"]["bias_r"]), -0.5, abs_tol=1e-12)
            and float(metrics["model_error"]["huber_raw_r"]) < float(metrics["frozen_constant_error"]["huber_raw_r"])
            and float(metrics["model_error"]["mae_r"]) < float(metrics["frozen_constant_error"]["mae_r"])
            and math.isclose(float(line["slope"]), 1.0, abs_tol=1e-12)
            and math.isclose(float(line["intercept_r"]), 0.5, abs_tol=1e-12)
            and conclusion.get("classification") == "MAGNITUDE_ERROR_BEATS_FROZEN_CONSTANT_WITH_POSITIVE_STRUCTURE"
            and len(buckets) == 4
            and set(signs["predicted_sign"].tolist()) == {"> 0", "<= 0"}
        ),
    )

    summary["workflow"] = "mr13f_direct_r_calibration_audit"
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
            and definitions[0].audit_type == "direct_r_calibration"
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
