"""Read-only Standard Model SOP evidence validation, independent of producers."""
from __future__ import annotations
import math
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY


def standard_model_sop_completeness_issues(payload: dict) -> tuple[str, ...]:
    """Block publication of missing/legacy truth; never backfill Joint Quality from averages."""
    source = dict(payload or {})
    standard = dict(source.get("standard_model_sop") or {})
    if not standard and str(source.get("schema") or "").startswith("standard_model_sop_v"):
        standard = source
    issues = []
    expected_schema = {"standard_model_sop_v10", "standard_model_sop_v10_robustness_mean"}
    if str(standard.get("schema") or "") not in expected_schema:
        issues.append("Standard SOP evaluation truth version v10 missing: refresh from scored stock-day evidence (do not relabel averages)")
    truth = dict(standard.get("evaluation_truth") or {})
    try:
        horizon = int(truth.get("horizon_bars"))
    except (TypeError, ValueError, OverflowError):
        horizon = 0
    if horizon != int(DEFAULT_LABEL_POLICY.label_horizon_bars):
        issues.append(
            "evaluation truth mismatch: horizon_bars must match the canonical common 40-bar "
            "MFE/Adverse evaluation (refresh audit from validated scores)"
        )
    for key, expected in (
        ("mfe_source", "target_favorable_return"),
        ("adverse_source", "target_adverse_return_to_peak"),
        ("joint_quality_id", "same_day_min_mfe_safety_percentile_v1"),
    ):
        if truth.get(key) != expected:
            issues.append(f"evaluation_truth:{key} mismatch")
    cohorts = dict(truth.get("cohort_identity") or {})
    independent_price_truth = dict(truth.get("price_truth_identity") or {})
    for split in ("validation", "oos", "breakout_candidate_oos"):
        price_digest = independent_price_truth.get(split)
        if (not isinstance(price_digest, str) or len(price_digest) != 64
                or any(char not in "0123456789abcdef" for char in price_digest)):
            issues.append(f"evaluation_truth:{split}:independent 40D price SHA256 missing")
        if not isinstance(cohorts.get(split), str) or len(cohorts[split]) != 64:
            issues.append(f"evaluation_truth:{split}:cohort SHA256 missing")
        learn = dict((standard.get("split_metrics") or {}).get(split) or {})
        alignment = dict((standard.get("upside_downside_alignment_evaluation") or {}).get(split) or {})
        top = dict(alignment.get("top_10pct") or {})
        for key in ("group_count", "mean_daily_spearman", "global_spearman_vs_raw_target",
                    "pairwise_concordance", "top_score_decile_raw_target_mean", "bottom_score_decile_raw_target_mean"):
            value = learn.get(key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or (key == "group_count" and int(value) < 1)):
                issues.append(f"M1:{split}:{key} invalid or missing")
        for key in ("target_to_full_mfe_daily_spearman", "target_to_low_adverse_daily_spearman",
                    "score_to_full_mfe_daily_spearman", "score_to_low_adverse_daily_spearman"):
            value = alignment.get(key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value))):
                issues.append(f"M3:{split}:{key} invalid or missing")
        if alignment.get("joint_quality_evaluation_id") != "same_day_min_mfe_safety_percentile_v1":
            issues.append(f"M4:{split}:joint_quality_evaluation_id mismatch")
        for key in ("n", "mfe_pct_mean", "adverse_pct_mean", "joint_quality_pct"):
            value = top.get(key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value))):
                issues.append(f"M4:{split}:{key} invalid or missing")
        if isinstance(top.get("n"), (int, float)) and math.isfinite(float(top["n"])):
            if float(top["n"]) < 1:
                issues.append(f"M4:{split}:top10 sample empty")
        joint_quality = top.get("joint_quality_pct")
        if isinstance(joint_quality, (int, float)) and math.isfinite(float(joint_quality)):
            if not (0 <= float(joint_quality) <= 100):
                issues.append(f"M4:{split}:Joint Quality out of range")
    # M2 requires both transitions, directly from the canonical split metrics.
    # A renderer is not an evidence authority and may not validate its own projection.
    splits = dict(standard.get("split_metrics") or {})
    if not all(splits.get(key) for key in ("validation", "oos", "breakout_candidate_oos")):
        issues.append("M2:both transitions missing")
    return tuple(dict.fromkeys(issues))
