"""Read-only MR-13AB vs MR-13K frozen-score survival-increment control.

This one-shot Audit answers the final Model-Gate uncertainty for MR-13AB: did
training on Pure-MFE before the first canonical risk breach add a measurable
first-passage survival ordering, or did the frozen model remain effectively the
same Pure-MFE ranker because most target rows are unchanged?

The Audit consumes the canonical MR-13AB Forward-OOS score plus the MR-13K frozen
Forward score.  If the historical MR-13K score CSV was pruned, the shared Research
dependency layer may deterministically reconstruct only those scores from the existing
MR-13K frozen checkpoint before this read-only Audit runs.  It never fits parameters,
rebuilds targets, creates PIT scores, or replays a strategy.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.path_utils import project_relative_display_path
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from services.breakout_quality.frozen_daily_ranker_score_rebuild import (
    rebuild_frozen_daily_ranker_scores_for_keys,
)
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    resolve_continuous_ranker_oos_score_path,
)

SUPPORTED_AUDIT_TYPE = "mr13ab_survival_increment"


def _relative(path: Path, root: Path) -> str:
    return project_relative_display_path(Path(path), project_root=Path(root))


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return None
    xr = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _mean_daily_spearman(frame: pd.DataFrame, score_col: str, target_col: str) -> dict[str, Any]:
    values: list[float] = []
    for _date, day in frame.groupby("date", sort=True):
        corr = _safe_spearman(
            day[score_col].to_numpy(dtype=np.float64),
            day[target_col].to_numpy(dtype=np.float64),
        )
        if corr is not None:
            values.append(float(corr))
    return {
        "rankable_date_count": int(len(values)),
        "mean_daily_spearman": float(np.mean(values)) if values else None,
        "median_daily_spearman": float(np.median(values)) if values else None,
    }


def _add_daily_percentile(frame: pd.DataFrame, value_col: str, output_col: str) -> None:
    grouped = frame.groupby("date", sort=False)[value_col]
    average_rank = grouped.rank(method="average") - 1.0
    size = grouped.transform("size").astype(float)
    denominator = size - 1.0
    percentile = np.full(len(frame), 0.5, dtype=np.float64)
    np.divide(
        average_rank.to_numpy(dtype=np.float64),
        denominator.to_numpy(dtype=np.float64),
        out=percentile,
        where=denominator.to_numpy(dtype=np.float64) > 0.0,
    )
    frame[output_col] = percentile


def _pair_concordance(score_diff: np.ndarray, target_diff: np.ndarray) -> tuple[float, int]:
    products = np.asarray(score_diff, dtype=np.float64) * np.asarray(target_diff, dtype=np.float64)
    count = int(len(products))
    if count == 0:
        return math.nan, 0
    concordant = float((products > 0.0).sum()) + 0.5 * float((products == 0.0).sum())
    return float(concordant / count), count


def _conflict_pair_metrics(day: pd.DataFrame, *, tolerance_r: float) -> dict[str, Any]:
    changed = day["target_changed"].to_numpy(dtype=bool)
    changed_idx = np.flatnonzero(changed)
    unchanged_idx = np.flatnonzero(~changed)
    if len(changed_idx) == 0:
        return {"pair_count": 0}

    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    if len(unchanged_idx):
        left_parts.append(np.repeat(changed_idx, len(unchanged_idx)))
        right_parts.append(np.tile(unchanged_idx, len(changed_idx)))
    if len(changed_idx) >= 2:
        a, b = np.triu_indices(len(changed_idx), k=1)
        left_parts.append(changed_idx[a])
        right_parts.append(changed_idx[b])
    if not left_parts:
        return {"pair_count": 0}

    i = np.concatenate(left_parts)
    j = np.concatenate(right_parts)
    candidate_target = day["candidate_target_r"].to_numpy(dtype=np.float64)
    reference_target = day["reference_target_r"].to_numpy(dtype=np.float64)
    candidate_score = day["candidate_score"].to_numpy(dtype=np.float64)
    reference_score = day["reference_score"].to_numpy(dtype=np.float64)
    candidate_diff = candidate_target[i] - candidate_target[j]
    reference_diff = reference_target[i] - reference_target[j]
    conflict = (
        ((candidate_diff > tolerance_r) & (reference_diff < -tolerance_r))
        | ((candidate_diff < -tolerance_r) & (reference_diff > tolerance_r))
    )
    if not bool(conflict.any()):
        return {"pair_count": 0}

    candidate_diff = candidate_diff[conflict]
    reference_diff = reference_diff[conflict]
    i = i[conflict]
    j = j[conflict]
    candidate_conc, pair_count = _pair_concordance(candidate_score[i] - candidate_score[j], candidate_diff)
    reference_conc, _ = _pair_concordance(reference_score[i] - reference_score[j], candidate_diff)
    candidate_ref_truth, _ = _pair_concordance(candidate_score[i] - candidate_score[j], reference_diff)
    reference_ref_truth, _ = _pair_concordance(reference_score[i] - reference_score[j], reference_diff)
    return {
        "pair_count": int(pair_count),
        "candidate_model_candidate_truth_concordance": candidate_conc,
        "reference_model_candidate_truth_concordance": reference_conc,
        "candidate_model_reference_truth_concordance": candidate_ref_truth,
        "reference_model_reference_truth_concordance": reference_ref_truth,
    }


def analyze_frozen_score_frames(
    candidate_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    *,
    tolerance_r: float = 1e-6,
    top_fraction: float = 0.10,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return exact changed-row/conflict-pair evidence from two frozen OOS score tables."""

    if not (0.0 < float(top_fraction) < 1.0):
        raise ValueError("top_fraction必須介於0與1")
    tolerance = float(tolerance_r)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_r必須是有限非負數")

    candidate_required = {
        "ticker", "date", "group_index", "target_raw_r", "reference_target_raw_r", "model_score"
    }
    reference_required = {"ticker", "date", "group_index", "target_raw_r", "model_score"}
    missing_candidate = sorted(candidate_required - set(candidate_frame.columns))
    missing_reference = sorted(reference_required - set(reference_frame.columns))
    if missing_candidate:
        raise ValueError(f"MR-13AB OOS score缺少欄位: {missing_candidate}")
    if missing_reference:
        raise ValueError(f"MR-13K OOS score缺少欄位: {missing_reference}")

    candidate = candidate_frame[list(candidate_required)].copy()
    reference = reference_frame[list(reference_required)].copy()
    candidate["date"] = pd.to_datetime(candidate["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    reference["date"] = pd.to_datetime(reference["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for table, label in ((candidate, "MR-13AB"), (reference, "MR-13K")):
        if bool(table["date"].isna().any()):
            raise ValueError(f"{label} OOS score含無效date")
        table["ticker"] = table["ticker"].astype(str).str.strip()
        table["group_index"] = pd.to_numeric(table["group_index"], errors="coerce")
        if bool(table["group_index"].isna().any()):
            raise ValueError(f"{label} OOS score含無效group_index")
        table["group_index"] = table["group_index"].astype(np.int64)

    for column in ("target_raw_r", "reference_target_raw_r", "model_score"):
        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
    for column in ("target_raw_r", "model_score"):
        reference[column] = pd.to_numeric(reference[column], errors="coerce")

    candidate = candidate.loc[
        np.isfinite(candidate["target_raw_r"])
        & np.isfinite(candidate["reference_target_raw_r"])
        & np.isfinite(candidate["model_score"])
    ].copy()
    reference = reference.loc[
        np.isfinite(reference["target_raw_r"])
        & np.isfinite(reference["model_score"])
    ].copy()
    keys = ["ticker", "date", "group_index"]
    for table, label in ((candidate, "MR-13AB"), (reference, "MR-13K")):
        if bool(table.duplicated(keys).any()):
            raise ValueError(f"{label} OOS evaluable identity不唯一")
    if len(candidate) != len(reference):
        raise ValueError(
            f"MR-13AB/MR-13K OOS evaluable row數不一致: {len(candidate)} != {len(reference)}"
        )

    joined = candidate.merge(reference, on=keys, how="inner", validate="one_to_one", suffixes=("__ab", "__k"))
    if len(joined) != len(candidate):
        raise ValueError("MR-13AB/MR-13K OOS evaluable identity集合不一致")
    joined = joined.rename(columns={
        "target_raw_r__ab": "candidate_target_r",
        "reference_target_raw_r": "embedded_reference_target_r",
        "model_score__ab": "candidate_score",
        "target_raw_r__k": "reference_target_r",
        "model_score__k": "reference_score",
    }).sort_values(["date", "ticker", "group_index"], kind="stable").reset_index(drop=True)

    embedded = joined["embedded_reference_target_r"].to_numpy(dtype=np.float64)
    reference_target = joined["reference_target_r"].to_numpy(dtype=np.float64)
    if not np.allclose(embedded, reference_target, rtol=0.0, atol=max(tolerance, 2e-6), equal_nan=False):
        max_delta = float(np.max(np.abs(embedded - reference_target)))
        raise ValueError(f"MR-13AB內嵌reference target與MR-13K target不一致: max_abs_delta={max_delta:.8g}R")

    candidate_target = joined["candidate_target_r"].to_numpy(dtype=np.float64)
    if bool(np.any(candidate_target > reference_target + max(tolerance, 2e-6))):
        raise ValueError("MR-13AB first-breach target出現大於full-horizon Pure-MFE的非法row")
    correction = reference_target - candidate_target
    changed = correction > tolerance
    joined["target_correction_r"] = correction
    joined["target_changed"] = changed

    _add_daily_percentile(joined, "candidate_score", "candidate_score_percentile")
    _add_daily_percentile(joined, "reference_score", "reference_score_percentile")
    joined["candidate_minus_reference_score_percentile"] = (
        joined["candidate_score_percentile"] - joined["reference_score_percentile"]
    )

    cross_target = {
        "candidate_model_vs_candidate_target": _mean_daily_spearman(joined, "candidate_score", "candidate_target_r"),
        "reference_model_vs_candidate_target": _mean_daily_spearman(joined, "reference_score", "candidate_target_r"),
        "candidate_model_vs_reference_target": _mean_daily_spearman(joined, "candidate_score", "reference_target_r"),
        "reference_model_vs_reference_target": _mean_daily_spearman(joined, "reference_score", "reference_target_r"),
    }

    changed_rows = joined.loc[joined["target_changed"]].copy()
    unchanged_rows = joined.loc[~joined["target_changed"]].copy()
    top_cutoff = 1.0 - float(top_fraction)
    changed_k_top = changed_rows["reference_score_percentile"] >= top_cutoff
    changed_ab_top = changed_rows["candidate_score_percentile"] >= top_cutoff
    changed_top_reference_count = int(changed_k_top.sum())
    changed_top_retained_count = int((changed_k_top & changed_ab_top).sum())

    changed_summary = {
        "row_count": int(len(changed_rows)),
        "row_rate": float(len(changed_rows) / len(joined)) if len(joined) else None,
        "mean_target_correction_r": float(changed_rows["target_correction_r"].mean()) if len(changed_rows) else None,
        "median_target_correction_r": float(changed_rows["target_correction_r"].median()) if len(changed_rows) else None,
        "mean_candidate_score_percentile": float(changed_rows["candidate_score_percentile"].mean()) if len(changed_rows) else None,
        "mean_reference_score_percentile": float(changed_rows["reference_score_percentile"].mean()) if len(changed_rows) else None,
        "mean_candidate_minus_reference_score_percentile": float(changed_rows["candidate_minus_reference_score_percentile"].mean()) if len(changed_rows) else None,
        "median_candidate_minus_reference_score_percentile": float(changed_rows["candidate_minus_reference_score_percentile"].median()) if len(changed_rows) else None,
        "correction_vs_reference_minus_candidate_percentile_spearman": _safe_spearman(
            changed_rows["target_correction_r"].to_numpy(dtype=np.float64),
            (changed_rows["reference_score_percentile"] - changed_rows["candidate_score_percentile"]).to_numpy(dtype=np.float64),
        ) if len(changed_rows) else None,
        "reference_model_top_fraction_changed_rows": changed_top_reference_count,
        "candidate_model_top_fraction_changed_rows": int(changed_ab_top.sum()),
        "reference_top_changed_retention_rate_under_candidate": (
            float(changed_top_retained_count / changed_top_reference_count)
            if changed_top_reference_count else None
        ),
    }
    unchanged_summary = {
        "row_count": int(len(unchanged_rows)),
        "mean_candidate_minus_reference_score_percentile": (
            float(unchanged_rows["candidate_minus_reference_score_percentile"].mean())
            if len(unchanged_rows) else None
        ),
    }

    conflict_rows: list[dict[str, Any]] = []
    total_pairs = 0
    candidate_concordant_weight = 0.0
    reference_concordant_weight = 0.0
    candidate_reference_truth_weight = 0.0
    reference_reference_truth_weight = 0.0
    for date_value, day in joined.groupby("date", sort=True):
        metrics = _conflict_pair_metrics(day, tolerance_r=tolerance)
        pair_count = int(metrics.get("pair_count", 0) or 0)
        if pair_count <= 0:
            continue
        row = {"date": str(date_value), **metrics}
        conflict_rows.append(row)
        total_pairs += pair_count
        candidate_concordant_weight += float(metrics["candidate_model_candidate_truth_concordance"]) * pair_count
        reference_concordant_weight += float(metrics["reference_model_candidate_truth_concordance"]) * pair_count
        candidate_reference_truth_weight += float(metrics["candidate_model_reference_truth_concordance"]) * pair_count
        reference_reference_truth_weight += float(metrics["reference_model_reference_truth_concordance"]) * pair_count

    conflict_by_date = pd.DataFrame(conflict_rows)
    if total_pairs:
        candidate_candidate = candidate_concordant_weight / total_pairs
        reference_candidate = reference_concordant_weight / total_pairs
        candidate_reference = candidate_reference_truth_weight / total_pairs
        reference_reference = reference_reference_truth_weight / total_pairs
    else:
        candidate_candidate = reference_candidate = candidate_reference = reference_reference = None
    conflict_summary = {
        "pair_count": int(total_pairs),
        "date_count": int(len(conflict_by_date)),
        "candidate_model_candidate_truth_concordance": candidate_candidate,
        "reference_model_candidate_truth_concordance": reference_candidate,
        "candidate_minus_reference_candidate_truth_concordance": (
            float(candidate_candidate - reference_candidate)
            if candidate_candidate is not None and reference_candidate is not None else None
        ),
        "candidate_model_reference_truth_concordance": candidate_reference,
        "reference_model_reference_truth_concordance": reference_reference,
    }

    metrics = {
        "common_oos_rows": int(len(joined)),
        "date_count": int(joined["date"].nunique()),
        "target_invariant": {
            "candidate_never_exceeds_reference": True,
            "embedded_reference_matches_reference_model_target": True,
            "tolerance_r": tolerance,
        },
        "cross_target_daily_spearman": cross_target,
        "changed_rows": changed_summary,
        "unchanged_rows": unchanged_summary,
        "conflict_pairs": conflict_summary,
        "top_fraction": float(top_fraction),
        "percentile_method": "same_date_average_zero_based_rank_over_common_oos",
    }
    return metrics, changed_rows, conflict_by_date


def _read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取continuous ranker report: {path.name}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"continuous ranker report根節點必須是object: {path.name}")
    return payload


def _validate_profile_artifacts(
    *,
    root: Path,
    filter_id: str,
    architecture: str,
    profile_id: str,
    research_id: str,
    target_id: str,
    seed: int,
    expected_reference_profile: str | None = None,
) -> dict[str, Any]:
    score_path = resolve_continuous_ranker_oos_score_path(root, filter_id, architecture, profile_id)
    report_path = score_path.parent / CONTINUOUS_RANKER_REPORT_FILENAME
    if not score_path.is_file():
        raise FileNotFoundError("缺少frozen Forward OOS score: " + _relative(score_path, root))
    if not report_path.is_file():
        raise FileNotFoundError("缺少continuous ranker report: " + _relative(report_path, root))
    report = _read_report(report_path)
    expected = {
        "filter_id": filter_id,
        "model_architecture": architecture,
        "experiment_profile": profile_id,
        "model_research_id": research_id,
    }
    for field, value in expected.items():
        if str(report.get(field) or "") != str(value):
            raise ValueError(f"{research_id} report {field}不一致")
    training = dict(report.get("training") or {})
    if str(training.get("target") or "") != target_id:
        raise ValueError(f"{research_id} report target不一致")
    if int(training.get("seed", -1)) != int(seed):
        raise ValueError(f"{research_id} report seed不一致")
    artifact = dict(dict(report.get("artifacts") or {}).get("oos_scores_gzip") or {})
    if str(artifact.get("filename") or "") != score_path.name:
        raise ValueError(f"{research_id} report OOS score filename不一致")
    expected_hash = str(artifact.get("sha256") or "").lower()
    actual_hash = compute_file_sha256(score_path).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(f"{research_id} frozen OOS score SHA256與report不一致")
    if expected_reference_profile is not None:
        reference_eval = dict(report.get("reference_target_evaluation") or {})
        if not bool(reference_eval.get("available", False)):
            raise ValueError(f"{research_id} report缺少reference target evaluation")
        if str(reference_eval.get("reference_profile") or "") != str(expected_reference_profile):
            raise ValueError(f"{research_id} report reference profile不一致")
        if bool(reference_eval.get("used_for_training_or_epoch_selection", True)):
            raise ValueError(f"{research_id} reference target不得參與training/epoch selection")
    return {
        "score_path": score_path,
        "report_path": report_path,
        "score_sha256": actual_hash,
        "report": report,
    }



def _reference_cache_paths(definition: AuditDefinition, root: Path) -> tuple[Path, Path]:
    base = Path(root) / AUDIT_OUTPUT_ROOT / definition.output_subdir / "source_cache"
    return base / "mr13k_frozen_oos_scores.csv.gz", base / "mr13k_frozen_oos_scores_manifest.json"


def _validate_reference_model_artifacts(
    *,
    root: Path,
    filter_id: str,
    architecture: str,
    profile_id: str,
    research_id: str,
    target_id: str,
    seed: int,
) -> dict[str, Any]:
    artifacts = resolve_filter_artifact_paths(root, filter_id, architecture, profile_id)
    report_path = resolve_filter_model_output_dir(root, filter_id, architecture, profile_id) / CONTINUOUS_RANKER_REPORT_FILENAME
    for label, path in (("frozen model", artifacts.model_path), ("model manifest", artifacts.manifest_path), ("continuous ranker report", report_path)):
        if not path.is_file():
            raise FileNotFoundError(f"MR-13K缺少{label}，Audit不得以重新訓練取代: {_relative(path, root)}")
    report = _read_report(report_path)
    manifest = _read_report(artifacts.manifest_path)
    expected = {
        "filter_id": filter_id,
        "model_architecture": architecture,
        "experiment_profile": profile_id,
        "model_research_id": research_id,
    }
    for field, value in expected.items():
        if str(report.get(field) or "") != str(value):
            raise ValueError(f"{research_id} report {field}不一致")
    for field, value in {
        "filter_id": filter_id,
        "model_architecture": architecture,
        "experiment_profile": profile_id,
    }.items():
        if str(manifest.get(field) or "") != str(value):
            raise ValueError(f"{research_id} manifest {field}不一致")
    if str(manifest.get("continuous_target_id") or "") != str(target_id):
        raise ValueError(f"{research_id} manifest continuous_target_id不一致")
    training = dict(report.get("training") or {})
    if str(training.get("target") or "") != target_id:
        raise ValueError(f"{research_id} report target不一致")
    if int(training.get("seed", -1)) != int(seed):
        raise ValueError(f"{research_id} report seed不一致")
    model_hash = compute_file_sha256(artifacts.model_path).lower()
    recorded_model = dict((report.get("artifacts") or {}).get("model") or {})
    if recorded_model:
        if str(recorded_model.get("filename") or "") != artifacts.model_path.name:
            raise ValueError(f"{research_id} report model filename不一致")
        if str(recorded_model.get("sha256") or "").lower() != model_hash:
            raise ValueError(f"{research_id} frozen model SHA256與report不一致")
    manifest_model = dict(manifest.get("model") or {})
    if manifest_model:
        if str(manifest_model.get("filename") or "") != artifacts.model_path.name:
            raise ValueError(f"{research_id} manifest model filename不一致")
        if str(manifest_model.get("sha256") or "").lower() != model_hash:
            raise ValueError(f"{research_id} frozen model SHA256與manifest不一致")
    return {
        "model_path": artifacts.model_path,
        "manifest_path": artifacts.manifest_path,
        "report_path": report_path,
        "model_sha256": model_hash,
        "manifest_sha256": compute_file_sha256(artifacts.manifest_path).lower(),
        "report_sha256": compute_file_sha256(report_path).lower(),
        "report": report,
    }


def _load_valid_reference_cache(
    *,
    definition: AuditDefinition,
    root: Path,
    candidate: Mapping[str, Any],
    reference_meta: Mapping[str, Any],
) -> dict[str, Any]:
    score_path, cache_manifest_path = _reference_cache_paths(definition, root)
    if not score_path.is_file() or not cache_manifest_path.is_file():
        raise FileNotFoundError(
            "缺少MR-13K frozen score derived cache；可由既有frozen checkpoint做deterministic inference建立: "
            + _relative(score_path, root)
        )
    cache = _read_report(cache_manifest_path)
    expected_sources = {
        "candidate_score_sha256": str(candidate["score_sha256"]),
        "reference_model_sha256": str(reference_meta["model_sha256"]),
        "reference_manifest_sha256": str(reference_meta["manifest_sha256"]),
        "reference_report_sha256": str(reference_meta["report_sha256"]),
    }
    for field, expected in expected_sources.items():
        if str(cache.get(field) or "").lower() != str(expected).lower():
            raise ValueError(f"MR-13K derived score cache source identity stale: {field}")
    actual_hash = compute_file_sha256(score_path).lower()
    if str(cache.get("score_sha256") or "").lower() != actual_hash:
        raise ValueError("MR-13K derived score cache SHA256不一致")
    return {
        "score_path": score_path,
        "report_path": Path(reference_meta["report_path"]),
        "score_sha256": actual_hash,
        "report": reference_meta["report"],
        "derived_from_frozen_checkpoint": True,
        "cache_manifest_path": cache_manifest_path,
    }


def prepare_reference_frozen_scores(definition: AuditDefinition, *, project_root: Path) -> int:
    """Build only a missing MR-13K row-level score cache from its frozen checkpoint."""

    root = Path(project_root).resolve()
    source = definition.source
    filter_id = str(source.get("filter_id") or "").strip()
    architecture = str(source.get("model_architecture") or "").strip()
    candidate_profile = str(source.get("candidate_profile_id") or "").strip()
    reference_profile = str(source.get("reference_profile_id") or "").strip()
    seed = int(source.get("seed", 42))
    candidate = _validate_profile_artifacts(
        root=root,
        filter_id=filter_id,
        architecture=architecture,
        profile_id=candidate_profile,
        research_id=str(source.get("candidate_research_id") or ""),
        target_id=str(source.get("candidate_target_id") or ""),
        seed=seed,
        expected_reference_profile=reference_profile,
    )
    reference_meta = _validate_reference_model_artifacts(
        root=root,
        filter_id=filter_id,
        architecture=architecture,
        profile_id=reference_profile,
        research_id=str(source.get("reference_research_id") or ""),
        target_id=str(source.get("reference_target_id") or ""),
        seed=seed,
    )
    candidate_frame = read_breakout_quality_csv(candidate["score_path"])
    required = ["ticker", "date", "group_index", "target_raw_r", "reference_target_raw_r", "model_score"]
    missing = [name for name in required if name not in candidate_frame.columns]
    if missing:
        raise ValueError(f"MR-13AB OOS score缺少欄位: {missing}")
    numeric = candidate_frame.copy()
    for col in ("target_raw_r", "reference_target_raw_r", "model_score"):
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    numeric = numeric.loc[
        np.isfinite(numeric["target_raw_r"])
        & np.isfinite(numeric["reference_target_raw_r"])
        & np.isfinite(numeric["model_score"])
    ].copy()
    score_path, cache_manifest_path = _reference_cache_paths(definition, root)
    rebuilt = rebuild_frozen_daily_ranker_scores_for_keys(
        project_root=root,
        filter_id=filter_id,
        model_architecture=architecture,
        experiment_profile=reference_profile,
        key_frame=numeric[["ticker", "date", "group_index"]],
        output_path=score_path,
        expected_target_raw_r=numeric["reference_target_raw_r"].to_numpy(dtype=np.float64),
        target_tolerance_r=max(float(definition.dimensions.get("changed_tolerance_r", 1e-6)), 2e-6),
    )
    cache_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache_manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "mr13k_frozen_forward_score_reconstruction_for_mr13ab_audit",
        "candidate_score_sha256": candidate["score_sha256"],
        "reference_model_sha256": reference_meta["model_sha256"],
        "reference_manifest_sha256": reference_meta["manifest_sha256"],
        "reference_report_sha256": reference_meta["report_sha256"],
        "score_sha256": str(rebuilt["score_artifact"]["sha256"]).lower(),
        "row_count": int(rebuilt["row_count"]),
        "scientific_semantics": "inference_only_from_existing_frozen_checkpoint_no_fit_no_target_rebuild",
        "torch_execution": rebuilt["torch_execution"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _source_contract(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source = definition.source
    filter_id = str(source.get("filter_id") or "").strip()
    architecture = str(source.get("model_architecture") or "").strip()
    candidate_profile = str(source.get("candidate_profile_id") or "").strip()
    reference_profile = str(source.get("reference_profile_id") or "").strip()
    candidate_research_id = str(source.get("candidate_research_id") or "").strip()
    reference_research_id = str(source.get("reference_research_id") or "").strip()
    candidate_target_id = str(source.get("candidate_target_id") or "").strip()
    reference_target_id = str(source.get("reference_target_id") or "").strip()
    seed = int(source.get("seed", 42))
    required = {
        "filter_id": filter_id,
        "model_architecture": architecture,
        "candidate_profile_id": candidate_profile,
        "reference_profile_id": reference_profile,
        "candidate_research_id": candidate_research_id,
        "reference_research_id": reference_research_id,
        "candidate_target_id": candidate_target_id,
        "reference_target_id": reference_target_id,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"{definition.audit_id} source contract缺少: {missing}")
    candidate = _validate_profile_artifacts(
        root=root, filter_id=filter_id, architecture=architecture,
        profile_id=candidate_profile, research_id=candidate_research_id,
        target_id=candidate_target_id, seed=seed,
        expected_reference_profile=reference_profile,
    )
    try:
        reference = _validate_profile_artifacts(
            root=root, filter_id=filter_id, architecture=architecture,
            profile_id=reference_profile, research_id=reference_research_id,
            target_id=reference_target_id, seed=seed,
        )
        reference["derived_from_frozen_checkpoint"] = False
    except FileNotFoundError as original_score_error:
        reference_meta = _validate_reference_model_artifacts(
            root=root, filter_id=filter_id, architecture=architecture,
            profile_id=reference_profile, research_id=reference_research_id,
            target_id=reference_target_id, seed=seed,
        )
        try:
            reference = _load_valid_reference_cache(
                definition=definition, root=root, candidate=candidate, reference_meta=reference_meta
            )
        except FileNotFoundError as cache_error:
            raise FileNotFoundError(f"{original_score_error}; {cache_error}") from cache_error
    return {
        **required,
        "seed": seed,
        "candidate": candidate,
        "reference": reference,
    }


def preflight(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        contract = _source_contract(definition, project_root=root)
    except (ValueError, OSError, FileNotFoundError) as exc:
        cache_score_path, _cache_manifest_path = _reference_cache_paths(definition, root)
        preparable = False
        try:
            source = definition.source
            candidate = _validate_profile_artifacts(
                root=root,
                filter_id=str(source.get("filter_id") or "").strip(),
                architecture=str(source.get("model_architecture") or "").strip(),
                profile_id=str(source.get("candidate_profile_id") or "").strip(),
                research_id=str(source.get("candidate_research_id") or "").strip(),
                target_id=str(source.get("candidate_target_id") or "").strip(),
                seed=int(source.get("seed", 42)),
                expected_reference_profile=str(source.get("reference_profile_id") or "").strip(),
            )
            _validate_reference_model_artifacts(
                root=root,
                filter_id=str(source.get("filter_id") or "").strip(),
                architecture=str(source.get("model_architecture") or "").strip(),
                profile_id=str(source.get("reference_profile_id") or "").strip(),
                research_id=str(source.get("reference_research_id") or "").strip(),
                target_id=str(source.get("reference_target_id") or "").strip(),
                seed=int(source.get("seed", 42)),
            )
            preparable = bool(candidate)
        except (ValueError, OSError, FileNotFoundError):
            preparable = False
        return {
            "status": "BLOCKED",
            "blockers": [str(exc)],
            "source_paths": [],
            "source_path": _relative(cache_score_path, root),
            "preparable": preparable,
        }
    return {
        "status": "READY",
        "blockers": [],
        "source_paths": [
            _relative(Path(contract["candidate"]["score_path"]), root),
            _relative(Path(contract["candidate"]["report_path"]), root),
            _relative(Path(contract["reference"]["score_path"]), root),
            _relative(Path(contract["reference"]["report_path"]), root),
        ],
        "source_path": _relative(Path(contract["reference"]["score_path"]), root),
        "preparable": False,
    }


def collect_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    state = preflight(definition, project_root=Path(project_root))
    return {
        "status": str(state.get("status") or "BLOCKED"),
        "reason": "；".join(str(v) for v in state.get("blockers", ())),
        "source": {
            "display": "MR-13AB frozen OOS + MR-13K frozen checkpoint/score (missing score可inference-only derive)",
            "path": str(state.get("source_path") or ""),
            "paths": list(state.get("source_paths", ())),
            "preparable": bool(state.get("preparable", False)),
        },
    }


def _fingerprint(definition: AuditDefinition, contract: Mapping[str, Any]) -> str:
    payload = {
        "audit": definition.as_dict(),
        "candidate_score_sha256": contract["candidate"]["score_sha256"],
        "reference_score_sha256": contract["reference"]["score_sha256"],
        "reference_score_derived_from_frozen_checkpoint": bool(contract["reference"].get("derived_from_frozen_checkpoint", False)),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "-" if not math.isfinite(number) else f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def render_result(result: Mapping[str, Any]) -> str:
    metrics = dict(result.get("metrics") or {})
    cross = dict(metrics.get("cross_target_daily_spearman") or {})
    changed = dict(metrics.get("changed_rows") or {})
    conflict = dict(metrics.get("conflict_pairs") or {})
    lines = [
        "=" * 100,
        " MR-13AB Survival Increment｜Frozen Model Control",
        "=" * 100,
        f"Audit     ：{result.get('audit_id', '')}",
        "Contract  ：READ ONLY / frozen Forward OOS artifacts only / no training / no PIT / no replay",
        f"Common OOS：{int(metrics.get('common_oos_rows', 0)):,}",
        "",
        "1. Cross-target Daily rho",
        "-------------------------",
        "Model      MR-13AB first-breach target   MR-13K full-horizon target",
        "---------  ----------------------------  --------------------------",
        "MR-13AB    {:>28}  {:>26}".format(
            _fmt(dict(cross.get("candidate_model_vs_candidate_target") or {}).get("mean_daily_spearman")),
            _fmt(dict(cross.get("candidate_model_vs_reference_target") or {}).get("mean_daily_spearman")),
        ),
        "MR-13K     {:>28}  {:>26}".format(
            _fmt(dict(cross.get("reference_model_vs_candidate_target") or {}).get("mean_daily_spearman")),
            _fmt(dict(cross.get("reference_model_vs_reference_target") or {}).get("mean_daily_spearman")),
        ),
        "",
        "2. Changed-row demotion",
        "------------------------",
        f"Changed rows                         ：{int(changed.get('row_count', 0)):,} ({_pct(changed.get('row_rate'))})",
        f"Mean target correction               ：{_fmt(changed.get('mean_target_correction_r'))}R",
        f"Mean score percentile｜MR-13K        ：{_fmt(changed.get('mean_reference_score_percentile'))}",
        f"Mean score percentile｜MR-13AB       ：{_fmt(changed.get('mean_candidate_score_percentile'))}",
        f"AB − K percentile                    ：{_fmt(changed.get('mean_candidate_minus_reference_score_percentile'))}",
        f"Correction ↔ (K−AB percentile) rho   ：{_fmt(changed.get('correction_vs_reference_minus_candidate_percentile_spearman'))}",
        f"K top-{metrics.get('top_fraction', 0.1):.0%} changed retained by AB ：{_pct(changed.get('reference_top_changed_retention_rate_under_candidate'))}",
        "",
        "3. Target-order conflict pairs",
        "------------------------------",
        f"Conflict pairs                       ：{int(conflict.get('pair_count', 0)):,}",
        f"Conflict dates                       ：{int(conflict.get('date_count', 0)):,}",
        f"MR-13AB concordance vs AB target     ：{_pct(conflict.get('candidate_model_candidate_truth_concordance'))}",
        f"MR-13K concordance vs AB target      ：{_pct(conflict.get('reference_model_candidate_truth_concordance'))}",
        f"AB − K                               ：{_pct(conflict.get('candidate_minus_reference_candidate_truth_concordance'))}",
        "",
        "Decision boundary：只判斷MR-13AB是否在target真正改寫／ordering衝突處增加survival ordering；",
        "不以本Audit結果調barrier、loss、threshold、architecture或portfolio rule。",
    ]
    return "\n".join(lines)


def _render_markdown(result: Mapping[str, Any]) -> str:
    return "```text\n" + render_result(result) + "\n```\n"


def run_audit(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract = _source_contract(definition, project_root=root)
    candidate_frame = read_breakout_quality_csv(contract["candidate"]["score_path"])
    reference_frame = read_breakout_quality_csv(contract["reference"]["score_path"])
    tolerance = float(definition.dimensions.get("changed_tolerance_r", 1e-6))
    top_fraction = float(definition.dimensions.get("top_fraction", 0.10))
    metrics, changed_rows, conflict_by_date = analyze_frozen_score_frames(
        candidate_frame, reference_frame, tolerance_r=tolerance, top_fraction=top_fraction
    )

    fingerprint = _fingerprint(definition, contract)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_dir = output_root / "runs" / f"{timestamp}_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "mr13ab_survival_increment.json"
    report_path = run_dir / "report.md"
    changed_path = run_dir / "changed_rows.csv.gz"
    conflict_path = run_dir / "conflict_by_date.csv"
    manifest_path = run_dir / "manifest.json"

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "module_id": definition.module_id,
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "decision_question": str(definition.outcomes.get("decision_question") or ""),
        "critical_uncertainty": str(definition.outcomes.get("critical_uncertainty") or ""),
        "stopping_condition": str(definition.outcomes.get("stopping_condition") or ""),
        "source_identity": {
            "candidate_research_id": contract["candidate_research_id"],
            "candidate_profile_id": contract["candidate_profile_id"],
            "candidate_target_id": contract["candidate_target_id"],
            "reference_research_id": contract["reference_research_id"],
            "reference_profile_id": contract["reference_profile_id"],
            "reference_target_id": contract["reference_target_id"],
            "model_architecture": contract["model_architecture"],
            "seed": int(contract["seed"]),
        },
        "metrics": metrics,
        "interpretation_boundary": {
            "changed_rows": "Only rows whose MR-13AB first-breach target is strictly below MR-13K full-horizon Pure-MFE target.",
            "conflict_pairs": "Only same-date pairs where the two target definitions demand opposite strict ordering; at least one row must be changed.",
            "causality": "Frozen-score diagnostic supports or rejects incremental survival ordering; it is not a strategy-performance claim.",
            "future_use": "No audit-derived threshold, weight, calibration or fitted coefficient may enter training/PIT/runtime.",
        },
        "artifacts": {
            "完整JSON": _relative(json_path, root),
            "詳細Markdown": _relative(report_path, root),
            "Changed Rows": _relative(changed_path, root),
            "Conflict By Date": _relative(conflict_path, root),
            "Manifest": _relative(manifest_path, root),
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_markdown(result), encoding="utf-8")
    changed_export = changed_rows[[
        "ticker", "date", "group_index", "candidate_target_r", "reference_target_r",
        "target_correction_r", "candidate_score", "reference_score",
        "candidate_score_percentile", "reference_score_percentile",
        "candidate_minus_reference_score_percentile",
    ]].copy()
    changed_export.to_csv(changed_path, index=False, encoding="utf-8-sig", compression="gzip")
    conflict_by_date.to_csv(conflict_path, index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": 1,
        "audit_definition": definition.as_dict(),
        "config_fingerprint": fingerprint,
        "source_refs": {
            "candidate_score": {
                "path": _relative(contract["candidate"]["score_path"], root),
                "sha256": contract["candidate"]["score_sha256"],
            },
            "reference_score": {
                "path": _relative(contract["reference"]["score_path"], root),
                "sha256": contract["reference"]["score_sha256"],
                "derived_from_frozen_checkpoint": bool(contract["reference"].get("derived_from_frozen_checkpoint", False)),
            },
        },
        "artifacts": result["artifacts"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest.json").write_text(json.dumps({
        "run_dir": _relative(run_dir, root),
        "report": _relative(report_path, root),
        "result": _relative(json_path, root),
        "config_fingerprint": fingerprint,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_dir = output_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "audit.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "audit.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "changed_rows.csv.gz").write_bytes(changed_path.read_bytes())
    (latest_dir / "conflict_by_date.csv").write_bytes(conflict_path.read_bytes())
    (latest_dir / "manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    return result


def run_formal_audit(
    definition: AuditDefinition,
    *,
    project_root: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    result = run_audit(definition, project_root=Path(project_root))
    if not quiet:
        print(render_result(result))
    return result


__all__ = [
    "SUPPORTED_AUDIT_TYPE",
    "analyze_frozen_score_frames",
    "collect_status",
    "prepare_reference_frozen_scores",
    "preflight",
    "render_result",
    "run_audit",
    "run_formal_audit",
]
