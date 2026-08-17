from __future__ import annotations

import numpy as np
import pandas as pd

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


def validate_breakout_quality_frozen_rank_fusion_audit_contract_case(_base_params):
    from tools.audit.breakout_quality.frozen_rank_fusion import compute_frozen_rank_fusion

    case_id = "AUDIT_FROZEN_RANK_FUSION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    per_day = 12
    dates = pd.to_datetime(["2025-01-02"] * per_day + ["2025-01-03"] * per_day)
    tickers = [f"T{i:02d}" for i in range(per_day)] * 2
    group_index = np.arange(per_day * 2, dtype=np.int64)
    first = np.arange(per_day, 0, -1, dtype=np.float64)
    second = np.arange(1, per_day + 1, dtype=np.float64)
    economic_target = np.concatenate([first, second])
    mfe_target = economic_target + 1.0
    low_adverse_target = economic_target / float(per_day)
    alternating_noise = np.asarray([3.0, -3.0] * (per_day // 2), dtype=np.float64)
    noise = np.concatenate([alternating_noise, alternating_noise[::-1]])
    mfe_score = economic_target + noise
    low_adverse_score = economic_target - noise

    def frame(score, target, *, reference_target=None):
        payload = {
            "ticker": tickers,
            "date": dates,
            "group_index": group_index,
            "model_score": np.asarray(score, dtype=np.float64),
            "target_raw_r": np.asarray(target, dtype=np.float64),
        }
        if reference_target is not None:
            payload["reference_target_raw_r"] = np.asarray(reference_target, dtype=np.float64)
        return pd.DataFrame(payload)

    mfe = frame(mfe_score, mfe_target, reference_target=economic_target)
    low = frame(low_adverse_score, low_adverse_target)
    payload = compute_frozen_rank_fusion(mfe, low)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "frozen_rank_fusion_uses_all_common_rows_without_training",
        (24, False, False),
        (
            payload.get("row_count"),
            (payload.get("decision") or {}).get("used_for_training_or_epoch_selection"),
            (payload.get("decision") or {}).get("weights_fitted"),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "fixed_equal_rank_fusion_can_preserve_joint_signal_without_weight_fit",
        "GO_MULTI_EXPERT_FUSION",
        (payload.get("decision") or {}).get("status"),
    )
    fused = (payload.get("metrics") or {}).get("fused_equal_rank") or {}
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "frozen_rank_fusion_reports_positive_economic_daily_rho_pair_and_topk_lift",
        True,
        bool(
            float(fused.get("mean_daily_spearman") or 0.0) > 0.0
            and float(fused.get("pairwise_concordance") or 0.0) > 0.5
            and float(fused.get("top_k_lift") or 0.0) > 0.0
        ),
    )
    summary["workflow"] = "read_only_fixed_equal_rank_frozen_score_fusion"
    return results, summary
