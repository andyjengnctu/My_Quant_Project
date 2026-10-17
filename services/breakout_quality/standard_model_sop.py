"""Single-source Standard Model SOP metric builder for continuous DL evaluation modes.

Forward, Rolling, and robustness consumers must provide evaluation score frames to this
module instead of reimplementing Standard SOP metrics.  Mode-specific evidence (for
example Rolling fold/year stability) is carried as Standard Mode Evidence and never
changes the common four-section metric payload.
"""

from __future__ import annotations

from typing import Any, Mapping
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd

from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from services.breakout_quality import ranker_training as ranker_api

STANDARD_SPLIT_KEYS = ("validation", "oos", "breakout_candidate_oos")

# Read-only evaluation metric, never a model Target or fitting/selection input.
JOINT_QUALITY_EVALUATION_ID = "same_day_min_mfe_safety_percentile_v1"
STANDARD_SOP_SCHEMA = "standard_model_sop_v10"

# Standard M3/M4 always evaluate the same canonical 40-bar price outcome.
# This is NOT the model's fitting target: shorter-label models keep their own
# 20-bar target, but are judged on the common 40-bar MFE/Adverse outcome.
STANDARD_EVALUATION_HORIZON_BARS = int(DEFAULT_LABEL_POLICY.label_horizon_bars)


def resolve_standard_evaluation_truth_table(
    bundle,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    project_root: Path,
    allow_stale_source: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Resolve canonical *evaluation* prices independently of each fitted label.

    Reuse the original bundle when its target spans the common evaluation horizon.
    For a 20-bar fitted label constrained to the same 40-bar mature universe,
    load a declared 40-bar pure-MFE target through the existing canonical data
    producer.  Never reconstitute prices from a 20-bar label, adjust score rows,
    retrain, or infer an evaluation horizon from the model identity.
    """
    from core.breakout_quality_runtime import DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from core.breakout_quality_policy import get_breakout_quality_model_test_settings
    from services.breakout_quality.continuous_ranker_pipeline import load_continuous_ranker_data

    common_horizon = STANDARD_EVALUATION_HORIZON_BARS
    manifest = dict(bundle.target_manifest or {})
    source_horizon = int(dict(manifest.get("target_contract") or {}).get("horizon_bars", 0))
    if source_horizon == common_horizon:
        return bundle.group_table, common_horizon
    if int(manifest.get("sample_universe_horizon_bars") or 0) != common_horizon:
        raise ValueError(
            "Standard Model SOP cross-horizon evaluation requires exactly the "
            "canonical 40-bar mature universe, not a shortened 20-bar cohort"
        )
    source_recipe = get_continuous_ranker_execution_recipe(str(experiment_profile))
    references = []
    for _model_id, profile in get_breakout_quality_model_test_settings().model_profiles:
        recipe = get_continuous_ranker_execution_recipe(str(profile))
        if (str(recipe.continuous_target_id) == DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID
                and recipe.context_policy.source == source_recipe.context_policy.source):
            references.append(str(profile))
    if not references:
        raise ValueError("Standard Model SOP missing compatible canonical 40-bar evaluation producer")
    reference = load_continuous_ranker_data(
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=references[0],
        preload_feature_bank=False,
        allow_stale_source=bool(allow_stale_source),
        project_root=Path(project_root),
    )
    reference_manifest = dict(reference.target_manifest or {})
    reference_contract = dict(reference_manifest.get("target_contract") or {})
    if (str(reference_contract.get("target_id")) != DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID
            or int(reference_contract.get("horizon_bars", 0)) != common_horizon):
        raise ValueError("Standard Model SOP canonical evaluation source target/horizon mismatch")

    source = bundle.group_table
    target = reference.group_table
    if len(source) != len(target):
        raise ValueError("Standard Model SOP 20/40 evaluation stock-day cohort size mismatch")
    # Checking group_index alone is NOT enough: the same integer could point to
    # another stock or date in independently materialized target bundles.
    for key in ("group_index", "ticker", "date"):
        if key not in source.columns or key not in target.columns:
            raise ValueError(f"Standard Model SOP evaluation key missing: {key}")
        if key == "date":
            match = pd.to_datetime(source[key]).reset_index(drop=True).equals(
                pd.to_datetime(target[key]).reset_index(drop=True)
            )
        else:
            match = source[key].astype(str).reset_index(drop=True).equals(
                target[key].astype(str).reset_index(drop=True)
            )
        if not match:
            raise ValueError(f"Standard Model SOP 20/40 evaluation {key} cohort mismatch")
    mature_source = np.asarray(bundle.target_valid, dtype=bool)
    if not np.array_equal(mature_source, np.asarray(reference.target_valid, dtype=bool)):
        raise ValueError("Standard Model SOP 20/40 evaluation mature cohort mismatch")
    # A 20-bar label does not grant earlier 20-bar publication. The matched
    # comparison cohort is explicitly mature at the 40th future trading bar.
    if ("label_eval_end_date" not in source.columns
            or "label_eval_end_date" not in target.columns):
        raise ValueError("Standard Model SOP shared evaluation maturity evidence missing")
    maturity_ids = np.flatnonzero(mature_source)
    source_end = pd.to_datetime(source.iloc[maturity_ids]["label_eval_end_date"], errors="raise")
    target_end = pd.to_datetime(target.iloc[maturity_ids]["label_eval_end_date"], errors="raise")
    source_start = pd.to_datetime(source.iloc[maturity_ids]["date"], errors="raise")
    if (source_end.isna().any() or target_end.isna().any()
            or not source_end.reset_index(drop=True).equals(target_end.reset_index(drop=True))
            or bool((source_end.to_numpy() <= source_start.to_numpy()).any())):
        raise ValueError("Standard Model SOP 20/40 evaluation maturity date mismatch")
    return target, common_horizon


def _ranking_dates(frame: pd.DataFrame) -> np.ndarray:
    """Return ranking-group dates, isolating overlapping Rolling fold/date groups."""

    if "rank_group" not in frame.columns:
        return pd.to_datetime(frame["date"], errors="raise").to_numpy()
    codes, _uniques = pd.factorize(frame["rank_group"].astype(str), sort=True)
    # The existing canonical ranking helpers group on datetime.  Only equality matters;
    # map arbitrary fold/date group keys to deterministic unique timestamps.
    base = pd.Timestamp("1900-01-01")
    return (base + pd.to_timedelta(codes, unit="D")).to_numpy()


def calculate_upside_downside_alignment_metrics(
    group_ids: np.ndarray,
    group_table: pd.DataFrame,
    raw_target: np.ndarray,
    scores: np.ndarray,
    *,
    ranking_dates: np.ndarray | None = None,
) -> dict[str, Any]:
    """Canonical actual-truth MFE/Safety diagnostics for every Standard Model SOP."""

    ids = np.asarray(group_ids, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    if score.shape != ids.shape:
        raise ValueError("Standard MFE/Safety score/group length mismatch")
    target_all = np.asarray(raw_target, dtype=np.float64)
    if target_all.ndim != 1 or len(target_all) != len(group_table):
        raise ValueError("Standard MFE/Safety raw_target length mismatch")

    favorable_all = pd.to_numeric(group_table["target_favorable_r"], errors="coerce").to_numpy(dtype=np.float64)
    adverse_all = pd.to_numeric(group_table["target_adverse_r"], errors="coerce").to_numpy(dtype=np.float64)
    mfe_return_all = pd.to_numeric(group_table["target_favorable_return"], errors="coerce").to_numpy(dtype=np.float64)
    adverse_return_all = pd.to_numeric(group_table["target_adverse_return_to_peak"], errors="coerce").to_numpy(dtype=np.float64)
    actual_dates_all = pd.to_datetime(group_table["date"], errors="raise").to_numpy()
    actual_valid_all = (
        np.isfinite(favorable_all) & np.isfinite(adverse_all)
        & np.isfinite(mfe_return_all) & np.isfinite(adverse_return_all)
    )
    # Joint Quality must measure the same price-return MFE / Adverse shown in
    # the M4 cells, not an R-normalized target whose cross-stock risk scale may
    # reorder the absolute opportunity or adverse excursion.
    mfe_pct_all = build_same_date_percentile_targets(
        mfe_return_all, actual_valid_all, actual_dates_all
    ).astype(np.float64)
    safety_pct_all = build_same_date_percentile_targets(
        -adverse_return_all, actual_valid_all, actual_dates_all
    ).astype(np.float64)

    actual_dates = actual_dates_all[ids]
    rank_dates = (
        actual_dates
        if ranking_dates is None
        else np.asarray(ranking_dates)
    )
    if len(rank_dates) != len(ids):
        raise ValueError("Standard MFE/Safety ranking_dates/group length mismatch")
    target = target_all[ids]
    favorable = favorable_all[ids]
    adverse = adverse_all[ids]
    mfe_return = mfe_return_all[ids]
    adverse_return = adverse_return_all[ids]
    mfe_pct = mfe_pct_all[ids]
    safety_pct = safety_pct_all[ids]

    def daily_rho(left: np.ndarray, right: np.ndarray) -> float | None:
        valid = np.isfinite(left) & np.isfinite(right)
        if int(valid.sum()) < 2:
            return None
        return ranker_api.daily_rank_metrics(
            rank_dates[valid], left[valid], right[valid]
        ).get("mean_daily_spearman")

    base_valid = (
        np.isfinite(score)
        & np.isfinite(target)
        & np.isfinite(favorable)
        & np.isfinite(adverse)
        & np.isfinite(mfe_return)
        & np.isfinite(adverse_return)
        & np.isfinite(mfe_pct)
        & np.isfinite(safety_pct)
    )
    if int(base_valid.sum()) < 2:
        return {
            "available": False,
            "group_count": int(base_valid.sum()),
            "not_evaluated_reason": "MFE/Adverse可評估sample不足",
        }

    valid_ids = np.flatnonzero(base_valid)
    # Exact per-rank-group top decile (ceil(10% N)), with deterministic ticker
    # tie-breaking.  Never apply a global multi-date score cutoff.
    rank_frame = pd.DataFrame({
        "index": np.flatnonzero(base_valid),
        "rank_group": pd.Series(rank_dates[base_valid]).astype(str).to_numpy(),
        "score": score[base_valid],
        "ticker": group_table.iloc[ids[base_valid]]["ticker"].astype(str).to_numpy(),
    })
    rank_frame = rank_frame.sort_values(
        ["rank_group", "score", "ticker", "index"],
        ascending=[True, False, True, True], kind="mergesort",
    )
    top = np.sort(np.concatenate([
        day.iloc[:max(1, int(np.ceil(len(day) * 0.10)))]["index"].to_numpy(dtype=np.int64)
        for _key, day in rank_frame.groupby("rank_group", sort=True)
    ]))
    if len(top) == 0:
        return {
            "available": False,
            "group_count": int(base_valid.sum()),
            "not_evaluated_reason": "同rank-group Top10 score cohort為空",
        }

    hm = mfe_pct >= 0.50
    hs = safety_pct >= 0.50
    quadrants = {
        "hmhs": hm & hs,
        "hmls": hm & ~hs,
        "lmhs": ~hm & hs,
        "lmls": ~hm & ~hs,
    }

    def quadrant_payload(mask: np.ndarray, cohort: np.ndarray) -> dict[str, float | None]:
        population_pct = float(np.mean(mask[valid_ids]) * 100.0)
        cohort_pct = float(np.mean(mask[cohort]) * 100.0)
        return {
            "pct": cohort_pct,
            "population_pct": population_pct,
            "enrichment": None if population_pct <= 0.0 else float(cohort_pct / population_pct),
        }

    top_quadrants = {key: quadrant_payload(mask, top) for key, mask in quadrants.items()}
    population_quadrants = {
        key: float(np.mean(mask[valid_ids]) * 100.0) for key, mask in quadrants.items()
    }
    return {
        "available": True,
        "group_count": int(base_valid.sum()),
        "joint_quality_evaluation_id": JOINT_QUALITY_EVALUATION_ID,
        "target_to_full_mfe_daily_spearman": daily_rho(target, favorable),
        "target_to_low_adverse_daily_spearman": daily_rho(target, -adverse),
        "score_to_full_mfe_daily_spearman": daily_rho(score, favorable),
        "score_to_low_adverse_daily_spearman": daily_rho(score, -adverse),
        "population": {
            "full_mfe_r_mean": float(np.mean(favorable[valid_ids])),
            "adverse_r_mean": float(np.mean(adverse[valid_ids])),
            "low_adverse_r_mean": float(np.mean(-adverse[valid_ids])),
            "high_mfe_pct": float(np.mean(hm[valid_ids]) * 100.0),
            "high_safety_pct": float(np.mean(hs[valid_ids]) * 100.0),
            "quadrants": population_quadrants,
        },
        "top_10pct": {
            "n": int(len(top)),
            "mfe_pct_mean": float(np.mean(mfe_return[top]) * 100.0),
            "adverse_pct_mean": float(np.mean(adverse_return[top]) * 100.0),
            "joint_quality_pct": float(np.mean(np.minimum(mfe_pct[top], safety_pct[top])) * 100.0),
            "full_mfe_r_mean": float(np.mean(favorable[top])),
            "adverse_r_mean": float(np.mean(adverse[top])),
            "low_adverse_r_mean": float(np.mean(-adverse[top])),
            "high_mfe_pct": float(np.mean(hm[top]) * 100.0),
            "high_safety_pct": float(np.mean(hs[top]) * 100.0),
            "quadrants": top_quadrants,
            "hmhs_pct": top_quadrants["hmhs"]["pct"],
            "hmhs_enrichment": top_quadrants["hmhs"]["enrichment"],
        },
        "status": "standard_sop_actual_truth_diagnostic_only_no_fit_no_selection",
    }


def _canonical_percentile_target(group_table: pd.DataFrame, raw_target: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_target, dtype=np.float64)
    dates = pd.to_datetime(group_table["date"], errors="raise").to_numpy()
    return build_same_date_percentile_targets(values, np.isfinite(values), dates).astype(np.float32)


def _split_metrics(
    frame: pd.DataFrame,
    *,
    group_table: pd.DataFrame,
    raw_target: np.ndarray,
    percentile_target: np.ndarray,
    evaluation_group_table: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if frame.empty:
        raise ValueError("Standard SOP split score frame不得為空")
    ids = pd.to_numeric(frame["group_index"], errors="raise").to_numpy(dtype=np.int64)
    scores = pd.to_numeric(frame["breakout_quality_score"], errors="raise").to_numpy(dtype=np.float64)
    if np.any((ids < 0) | (ids >= len(group_table))) or not np.isfinite(scores).all():
        raise ValueError("Standard SOP scored stock-day有非法group_index或非有限score")
    for truth_column in ("target_favorable_r", "target_adverse_r", "target_favorable_return",
                         "target_adverse_return_to_peak"):
        truth = pd.to_numeric(group_table.iloc[ids][truth_column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(truth).all():
            raise ValueError(f"Standard SOP尚未成熟或缺少actual truth: {truth_column}")
    rank_dates = _ranking_dates(frame)
    metrics = ranker_api.split_metrics(
        ids,
        group_table,
        raw_target,
        percentile_target,
        scores,
        include_top_k_quality=True,
        ranking_dates=rank_dates,
    )
    alignment = calculate_upside_downside_alignment_metrics(
        ids,
        evaluation_group_table,
        raw_target,
        scores,
        ranking_dates=rank_dates,
    )
    return metrics, alignment


def _score_cohort_digest(frame: pd.DataFrame, group_table: pd.DataFrame) -> str:
    """Stable stock-day/fold identity across independently trained model score tables."""
    ids = pd.to_numeric(frame["group_index"], errors="raise").to_numpy(dtype=np.int64)
    source = group_table.iloc[ids]
    date = pd.to_datetime(source["date"], errors="raise").dt.strftime("%Y-%m-%d")
    fold = (frame["rank_group"].astype(str).to_numpy()
            if "rank_group" in frame.columns else np.full(len(ids), "", dtype=object))
    stock = source["ticker"].astype(str).to_numpy()
    keys = sorted(f"{d}|{t}|{g}" for d,t,g in zip(date.to_numpy(),stock,fold))
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8") + b"\n")
    return digest.hexdigest()


def build_standard_model_sop(
    *,
    group_table: pd.DataFrame,
    raw_target: np.ndarray,
    training_objective: str,
    validation_scores: pd.DataFrame,
    oos_scores: pd.DataFrame,
    breakout_scores: pd.DataFrame,
    evaluation_mode: str,
    evaluation_horizon_bars: int,
    evaluation_group_table: pd.DataFrame | None = None,
    mode_extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the sole machine payload consumed by all Standard Model SOP renderers."""

    if int(evaluation_horizon_bars) < 2:
        raise ValueError("Standard SOP evaluation horizon must have at least two bars")
    shared_truth = group_table if evaluation_group_table is None else evaluation_group_table
    if len(shared_truth) != len(group_table):
        raise ValueError("Standard SOP evaluation stock-day population size mismatch")
    percentile_target = _canonical_percentile_target(group_table, raw_target)
    cohort_identity = {
        key: _score_cohort_digest(frame, group_table)
        for key, frame in (
            ("validation", validation_scores),
            ("oos", oos_scores),
            ("breakout_candidate_oos", breakout_scores),
        )
    }
    split_metrics: dict[str, Any] = {}
    alignment: dict[str, Any] = {}
    for key, frame in (
        ("validation", validation_scores),
        ("oos", oos_scores),
        ("breakout_candidate_oos", breakout_scores),
    ):
        split_metrics[key], alignment[key] = _split_metrics(
            frame,
            group_table=group_table,
            raw_target=raw_target,
            percentile_target=percentile_target,
            evaluation_group_table=shared_truth,
        )
    return {
        "schema": STANDARD_SOP_SCHEMA,
        "evaluation_mode": str(evaluation_mode),
        "training": {"objective": str(training_objective)},
        "evaluation_truth": {
            "horizon_bars": int(evaluation_horizon_bars),
            "mfe_source": "target_favorable_return",
            "adverse_source": "target_adverse_return_to_peak",
            "joint_quality_id": JOINT_QUALITY_EVALUATION_ID,
            "cohort_identity": cohort_identity,
        },
        "split_metrics": split_metrics,
        "upside_downside_alignment_evaluation": alignment,
        "mode_extensions": dict(mode_extensions or {}),
    }


def migrate_legacy_forward_standard_model_sop(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """One-way artifact migration for pre-v7 Forward reports; never used by renderers.

    This copies already-persisted canonical metric evidence into the new nested payload.
    It performs no metric calculation and exists only so a report-schema upgrade does not
    retrain an unchanged model.  Once persisted, all consumers use ``standard_model_sop``.
    """

    report = dict(payload or {})
    current = report.get("standard_model_sop")
    if isinstance(current, dict) and current:
        return dict(current)
    splits = dict(report.get("split_metrics") or {})
    alignment = dict(report.get("upside_downside_alignment_evaluation") or {})
    required = {"validation", "oos", "breakout_candidate_oos"}
    if not required.issubset(splits) or not required.issubset(alignment):
        return None
    return {
        "schema": "standard_model_sop_v9_migrated_metrics_only",
        "evaluation_mode": "forward_oos",
        "training": {"objective": str((report.get("training") or {}).get("objective") or "")},
        "split_metrics": {key: dict(splits[key]) for key in STANDARD_SPLIT_KEYS},
        "upside_downside_alignment_evaluation": {
            key: dict(alignment[key]) for key in STANDARD_SPLIT_KEYS
        },
        "mode_extensions": {},
        "migration": {"source": "pre_v7_forward_report", "metric_recalculation": False},
    }


def _aggregate_scalar_values(values: list[Any]) -> Any:
    """Aggregate one Standard-SOP leaf across robustness seeds.

    Scientific labels/booleans must agree exactly. Numeric evidence is aggregated by
    arithmetic mean; integer-valued leaves that are identical across seeds remain ints.
    """

    if not values:
        return None
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Robustness Standard SOP seed payload欄位不完整")
    if all(isinstance(value, bool) for value in values):
        if len(set(values)) != 1:
            raise ValueError("Robustness Standard SOP boolean contract跨seed不一致")
        return bool(values[0])
    numeric = all(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) for value in values)
    if numeric:
        floats = [float(value) for value in values]
        if not all(np.isfinite(floats)):
            raise ValueError("Robustness Standard SOP含非有限數值")
        if all(isinstance(value, (int, np.integer)) for value in values) and len(set(int(value) for value in values)) == 1:
            return int(values[0])
        return float(np.mean(floats))
    if all(isinstance(value, str) for value in values):
        if len(set(values)) != 1:
            raise ValueError("Robustness Standard SOP文字contract跨seed不一致")
        return str(values[0])
    return values[0] if all(value == values[0] for value in values) else None


def _aggregate_mapping_values(payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("Robustness Standard SOP至少需要一個seed payload")
    key_sets = [set(payload.keys()) for payload in payloads]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Robustness Standard SOP payload schema跨seed不一致")
    result: dict[str, Any] = {}
    for key in sorted(key_sets[0]):
        values = [payload[key] for payload in payloads]
        if all(isinstance(value, Mapping) for value in values):
            result[key] = _aggregate_mapping_values([dict(value) for value in values])
        elif all(isinstance(value, list) for value in values):
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"Robustness Standard SOP list contract跨seed不一致: {key}")
            result[key] = list(values[0])
        else:
            result[key] = _aggregate_scalar_values(values)
    return result


def _aggregate_rolling_fold_drift(payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate Rolling fold-drift diagnostics across robustness seeds.

    The drift criterion is a contract and must remain identical. The observed flagged
    folds and drift flag are seed-dependent evidence, so they cannot be treated as
    cross-seed schema invariants. Preserve the existing rolling-extension schema by
    reporting the deterministic union of flagged folds, conservative any-seed drift,
    and the arithmetic mean of numeric drift magnitude.
    """

    if not payloads:
        raise ValueError("Rolling robustness fold_drift至少需要一個seed payload")
    normalized = [dict(payload or {}) for payload in payloads]
    key_sets = [set(payload.keys()) for payload in normalized]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Rolling robustness fold_drift schema跨seed不一致")
    result: dict[str, Any] = {}
    for key in sorted(key_sets[0]):
        values = [payload[key] for payload in normalized]
        if key == "flagged_folds":
            if not all(isinstance(value, list) for value in values):
                raise ValueError("Rolling robustness flagged_folds必須為list")
            if any(not all(isinstance(item, str) for item in value) for value in values):
                raise ValueError("Rolling robustness flagged_folds只允許fold-id文字")
            result[key] = sorted({item for value in values for item in value})
        elif key == "drift_flag":
            if not all(isinstance(value, bool) for value in values):
                raise ValueError("Rolling robustness drift_flag必須為boolean")
            result[key] = bool(any(values))
        else:
            result[key] = _aggregate_scalar_values(values)
    if "drift_flag" in result and "flagged_folds" in result:
        result["drift_flag"] = bool(result["drift_flag"] or result["flagged_folds"])
    return result


def _aggregate_rolling_extension(payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate Rolling extension while keeping fold-drift outcome semantics explicit."""

    if not payloads:
        raise ValueError("Rolling robustness extension至少需要一個seed payload")
    normalized = [dict(payload or {}) for payload in payloads]
    key_sets = [set(payload.keys()) for payload in normalized]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Rolling robustness extension schema跨seed不一致")
    if "fold_drift" not in key_sets[0]:
        raise ValueError("Rolling robustness extension缺少fold_drift")
    common_payloads = []
    for payload in normalized:
        common = dict(payload)
        common.pop("fold_drift", None)
        common_payloads.append(common)
    result = _aggregate_mapping_values(common_payloads)
    result["fold_drift"] = _aggregate_rolling_fold_drift(
        [dict(payload["fold_drift"]) for payload in normalized]
    )
    return result


def _top_bottom_gap(split: Mapping[str, Any]) -> float | None:
    top = split.get("top_score_decile_raw_target_mean")
    bottom = split.get("bottom_score_decile_raw_target_mean")
    if top is None or bottom is None:
        return None
    return float(top) - float(bottom)


def aggregate_standard_model_sop_robustness(
    seed_payloads: list[Mapping[str, Any]],
    *,
    seeds: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Aggregate identical Standard-SOP schemas across benchmark seeds.

    The four Standard sections remain the exact same schema used by single-seed
    Forward/Rolling reports.  Only the mode extension adds across-seed dispersion.
    """

    if len(seed_payloads) < 2:
        raise ValueError("Robustness Standard SOP至少需要兩個seed")
    if len(seed_payloads) != len(seeds):
        raise ValueError("Robustness Standard SOP seed/payload數量不一致")
    normalized = [dict(payload or {}) for payload in seed_payloads]
    if any(not payload for payload in normalized):
        raise ValueError("Robustness Standard SOP seed payload不得為空")
    modes = {str(payload.get("evaluation_mode") or "") for payload in normalized}
    if len(modes) != 1:
        raise ValueError("Robustness Standard SOP evaluation_mode跨seed不一致")
    objectives = {str((payload.get("training") or {}).get("objective") or "") for payload in normalized}
    if len(objectives) != 1:
        raise ValueError("Robustness Standard SOP training objective跨seed不一致")

    truths = [dict(payload.get("evaluation_truth") or {}) for payload in normalized]
    if not truths[0] or any(truth != truths[0] for truth in truths[1:]):
        raise ValueError("Robustness Standard SOP score/true outcome cohort identity differs across seeds")
    split_payloads = [dict(payload.get("split_metrics") or {}) for payload in normalized]
    alignment_payloads = [dict(payload.get("upside_downside_alignment_evaluation") or {}) for payload in normalized]
    if any(set(payload) != set(STANDARD_SPLIT_KEYS) for payload in split_payloads):
        raise ValueError("Robustness Standard SOP split_metrics不完整")
    if any(set(payload) != set(STANDARD_SPLIT_KEYS) for payload in alignment_payloads):
        raise ValueError("Robustness Standard SOP alignment不完整")

    split_metrics = {
        key: _aggregate_mapping_values([dict(payload[key]) for payload in split_payloads])
        for key in STANDARD_SPLIT_KEYS
    }
    # Same evaluated stock-day cohort is a necessary but not sufficient seed
    # identity: reject missing score/target rows rather than silently averaging
    # differing Top10 sample sizes and truncating an artificial fractional N.
    for split in STANDARD_SPLIT_KEYS:
        counts = []
        for seed_alignment in alignment_payloads:
            raw = dict(dict(seed_alignment[split]).get("top_10pct") or {}).get("n")
            if (not isinstance(raw, (int, float, np.integer, np.floating))
                    or not np.isfinite(raw) or int(raw) < 1 or int(raw) != float(raw)):
                raise ValueError(f"Robustness Standard SOP {split} Top10 N invalid")
            counts.append(int(raw))
        if len(set(counts)) != 1:
            raise ValueError(f"Robustness Standard SOP {split} Top10 N differs across seeds")
    alignment = {
        key: _aggregate_mapping_values([dict(payload[key]) for payload in alignment_payloads])
        for key in STANDARD_SPLIT_KEYS
    }

    oos_rows = [dict(payload["oos"]) for payload in split_payloads]
    daily = np.asarray([float(row["mean_daily_spearman"]) for row in oos_rows], dtype=np.float64)
    pair = np.asarray([float(row["pairwise_concordance"]) for row in oos_rows], dtype=np.float64)
    gaps = np.asarray([float(_top_bottom_gap(row)) for row in oos_rows], dtype=np.float64)
    joint_qualities = np.asarray([
        float(dict(dict(payload["oos"]).get("top_10pct") or {})["joint_quality_pct"])
        for payload in alignment_payloads
    ], dtype=np.float64)
    if not np.isfinite(joint_qualities).all() or ((joint_qualities < 0) | (joint_qualities > 100)).any():
        raise ValueError("Robustness Standard SOP Joint Quality evidence非法")
    robustness = {
        "seed_count": int(len(seeds)),
        "seeds": [int(seed) for seed in seeds],
        "aggregation": "arithmetic_mean_across_benchmark_seeds",
        "oos_daily_rho_std": float(np.std(daily, ddof=0)),
        "oos_pair_std": float(np.std(pair, ddof=0)),
        "oos_top_bottom_std": float(np.std(gaps, ddof=0)),
        "oos_top10_joint_quality_std": float(np.std(joint_qualities, ddof=0)),
    }

    mode_extensions: dict[str, Any] = {"robustness": robustness}
    if str(next(iter(modes))) == "rolling_oos":
        rolling_payloads = [dict((payload.get("mode_extensions") or {}).get("rolling") or {}) for payload in normalized]
        if any(not payload for payload in rolling_payloads):
            raise ValueError("Rolling robustness Standard SOP缺少rolling extension")
        mode_extensions["rolling"] = _aggregate_rolling_extension(rolling_payloads)

    if any(str(payload.get("schema")) != STANDARD_SOP_SCHEMA for payload in normalized):
        raise ValueError("Robustness Standard SOP包含舊版evaluation schema，需由合法score重算")
    return {
        "schema": "standard_model_sop_v10_robustness_mean",
        "evaluation_mode": str(next(iter(modes))),
        "training": {"objective": str(next(iter(objectives)))},
        "evaluation_truth": dict(truths[0]),
        "split_metrics": split_metrics,
        "upside_downside_alignment_evaluation": alignment,
        "mode_extensions": mode_extensions,
    }
