"""Read-only MR-13AB frozen-score survival-increment control.

This one-shot Audit answers the final MR-13AB Model-Gate uncertainty without
requiring a historical MR-13K checkpoint.  MR-13AB's canonical Forward-OOS score
artifact already stores both truths for the exact same stock-day rows:

* target_raw_r: first-risk-breach Pure-MFE (MR-13AB)
* reference_target_raw_r: full-horizon Pure-MFE (MR-13K target semantics)

The Audit therefore isolates rows whose target was actually corrected and
same-date pairs where those two target definitions demand opposite ordering.
It then asks whether the frozen MR-13AB score follows the first-breach ordering
rather than the full-horizon ordering.  No fitting, target rebuild, PIT score,
historical-model reconstruction, or strategy replay is performed.
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
from filters.breakout_quality.csv_io import read_breakout_quality_csv
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
    """Measure only pairs where MR-13AB and MR-13K target semantics disagree in sign."""

    changed_idx = np.flatnonzero(day["target_changed"].to_numpy(dtype=bool))
    unchanged_idx = np.flatnonzero(~day["target_changed"].to_numpy(dtype=bool))
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

    candidate_diff = candidate_target[i] - candidate_target[j]
    reference_diff = reference_target[i] - reference_target[j]
    conflict = (
        ((candidate_diff > tolerance_r) & (reference_diff < -tolerance_r))
        | ((candidate_diff < -tolerance_r) & (reference_diff > tolerance_r))
    )
    if not bool(conflict.any()):
        return {"pair_count": 0}

    i = i[conflict]
    j = j[conflict]
    candidate_diff = candidate_diff[conflict]
    reference_diff = reference_diff[conflict]
    score_diff = candidate_score[i] - candidate_score[j]

    candidate_truth_conc, pair_count = _pair_concordance(score_diff, candidate_diff)
    reference_truth_conc, _ = _pair_concordance(score_diff, reference_diff)
    return {
        "pair_count": int(pair_count),
        "candidate_model_candidate_truth_concordance": candidate_truth_conc,
        "candidate_model_reference_truth_concordance": reference_truth_conc,
        "candidate_truth_advantage": (
            float(candidate_truth_conc - reference_truth_conc)
            if math.isfinite(candidate_truth_conc) and math.isfinite(reference_truth_conc)
            else None
        ),
    }


def analyze_candidate_score_frame(
    candidate_frame: pd.DataFrame,
    *,
    tolerance_r: float = 1e-6,
    top_fraction: float = 0.10,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return survival-correction evidence from MR-13AB's frozen OOS score artifact only."""

    if not (0.0 < float(top_fraction) < 1.0):
        raise ValueError("top_fraction必須介於0與1")
    tolerance = float(tolerance_r)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_r必須是有限非負數")

    required = {
        "ticker",
        "date",
        "group_index",
        "target_raw_r",
        "reference_target_raw_r",
        "model_score",
    }
    missing = sorted(required - set(candidate_frame.columns))
    if missing:
        raise ValueError(f"MR-13AB OOS score缺少欄位: {missing}")

    frame = candidate_frame[list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if bool(frame["date"].isna().any()):
        raise ValueError("MR-13AB OOS score含無效date")
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="coerce")
    if bool(frame["group_index"].isna().any()):
        raise ValueError("MR-13AB OOS score含無效group_index")
    frame["group_index"] = frame["group_index"].astype(np.int64)
    if bool(frame.duplicated(["ticker", "date", "group_index"]).any()):
        raise ValueError("MR-13AB OOS evaluable identity不唯一")

    for column in ("target_raw_r", "reference_target_raw_r", "model_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame["target_raw_r"])
        & np.isfinite(frame["reference_target_raw_r"])
        & np.isfinite(frame["model_score"])
    ].copy()
    if frame.empty:
        raise ValueError("MR-13AB OOS score沒有可評估row")

    frame = frame.rename(columns={
        "target_raw_r": "candidate_target_r",
        "reference_target_raw_r": "reference_target_r",
        "model_score": "candidate_score",
    }).sort_values(["date", "ticker", "group_index"], kind="stable").reset_index(drop=True)

    candidate_target = frame["candidate_target_r"].to_numpy(dtype=np.float64)
    reference_target = frame["reference_target_r"].to_numpy(dtype=np.float64)
    contract_tolerance = max(tolerance, 2e-6)

    # Historical MR-13K Pure-MFE is max(future high / anchor - 1) / R and is not
    # floored at zero. MR-13AB intentionally defines an empty pre-breach path
    # (first-bar breach) as 0R. Therefore AB may exceed K only in the narrow
    # zero-floor case: AB ~= 0R while the historical K reference is negative.
    exceeds_reference = candidate_target > reference_target + contract_tolerance
    legal_zero_floor_exception = (
        exceeds_reference
        & (np.abs(candidate_target) <= contract_tolerance)
        & (reference_target < -contract_tolerance)
    )
    illegal_exceeds_reference = exceeds_reference & ~legal_zero_floor_exception
    if bool(np.any(illegal_exceeds_reference)):
        raise ValueError(
            "MR-13AB first-breach target大於full-horizon Pure-MFE，且不符合"
            "first-bar empty-prebreach=0R / historical negative Pure-MFE例外"
        )

    correction = reference_target - candidate_target
    frame["target_correction_r"] = correction
    frame["target_changed"] = np.abs(correction) > tolerance
    frame["zero_floor_exception"] = legal_zero_floor_exception

    _add_daily_percentile(frame, "candidate_score", "candidate_score_percentile")
    _add_daily_percentile(frame, "candidate_target_r", "candidate_target_percentile")
    _add_daily_percentile(frame, "reference_target_r", "reference_target_percentile")
    frame["target_rank_correction"] = (
        frame["candidate_target_percentile"] - frame["reference_target_percentile"]
    )
    frame["score_minus_reference_rank"] = (
        frame["candidate_score_percentile"] - frame["reference_target_percentile"]
    )

    cross_target = {
        "candidate_model_vs_candidate_target": _mean_daily_spearman(
            frame, "candidate_score", "candidate_target_r"
        ),
        "candidate_model_vs_reference_target": _mean_daily_spearman(
            frame, "candidate_score", "reference_target_r"
        ),
        "candidate_target_vs_reference_target": _mean_daily_spearman(
            frame, "candidate_target_r", "reference_target_r"
        ),
    }

    changed_rows = frame.loc[frame["target_changed"]].copy()
    unchanged_rows = frame.loc[~frame["target_changed"]].copy()
    zero_floor_rows = frame.loc[frame["zero_floor_exception"]].copy()
    demoted_rows = frame.loc[
        frame["target_changed"] & (frame["candidate_target_r"] < frame["reference_target_r"] - tolerance)
    ].copy()
    top_cutoff = 1.0 - float(top_fraction)
    reference_only_top = (
        (changed_rows["reference_target_percentile"] >= top_cutoff)
        & (changed_rows["candidate_target_percentile"] < top_cutoff)
    )
    reference_only_top_count = int(reference_only_top.sum())
    model_retains_reference_only_top = (
        reference_only_top & (changed_rows["candidate_score_percentile"] >= top_cutoff)
    )

    changed_summary = {
        "row_count": int(len(changed_rows)),
        "row_rate": float(len(changed_rows) / len(frame)),
        "mean_target_correction_r": (
            float(changed_rows["target_correction_r"].mean()) if len(changed_rows) else None
        ),
        "median_target_correction_r": (
            float(changed_rows["target_correction_r"].median()) if len(changed_rows) else None
        ),
        "mean_candidate_score_percentile": (
            float(changed_rows["candidate_score_percentile"].mean()) if len(changed_rows) else None
        ),
        "mean_candidate_target_percentile": (
            float(changed_rows["candidate_target_percentile"].mean()) if len(changed_rows) else None
        ),
        "mean_reference_target_percentile": (
            float(changed_rows["reference_target_percentile"].mean()) if len(changed_rows) else None
        ),
        "mean_target_rank_correction": (
            float(changed_rows["target_rank_correction"].mean()) if len(changed_rows) else None
        ),
        "mean_score_minus_reference_rank": (
            float(changed_rows["score_minus_reference_rank"].mean()) if len(changed_rows) else None
        ),
        "rank_correction_alignment_spearman": (
            _safe_spearman(
                changed_rows["target_rank_correction"].to_numpy(dtype=np.float64),
                changed_rows["score_minus_reference_rank"].to_numpy(dtype=np.float64),
            )
            if len(changed_rows)
            else None
        ),
        "reference_only_top_fraction_rows": reference_only_top_count,
        "candidate_model_retains_reference_only_top_fraction_rows": int(
            model_retains_reference_only_top.sum()
        ),
        "reference_only_top_retention_rate_under_candidate_model": (
            float(model_retains_reference_only_top.sum() / reference_only_top_count)
            if reference_only_top_count
            else None
        ),
        "survival_demotion_row_count": int(len(demoted_rows)),
        "zero_floor_exception_row_count": int(len(zero_floor_rows)),
        "zero_floor_exception_row_rate": float(len(zero_floor_rows) / len(frame)),
    }
    unchanged_summary = {
        "row_count": int(len(unchanged_rows)),
        "mean_candidate_score_percentile": (
            float(unchanged_rows["candidate_score_percentile"].mean())
            if len(unchanged_rows)
            else None
        ),
    }

    conflict_rows: list[dict[str, Any]] = []
    total_pairs = 0
    candidate_truth_weight = 0.0
    reference_truth_weight = 0.0
    daily_candidate_truth: list[float] = []
    for date_value, day in frame.groupby("date", sort=True):
        day_metrics = _conflict_pair_metrics(day, tolerance_r=tolerance)
        pair_count = int(day_metrics.get("pair_count", 0) or 0)
        if pair_count <= 0:
            continue
        conflict_rows.append({"date": str(date_value), **day_metrics})
        total_pairs += pair_count
        candidate_conc = float(day_metrics["candidate_model_candidate_truth_concordance"])
        reference_conc = float(day_metrics["candidate_model_reference_truth_concordance"])
        candidate_truth_weight += candidate_conc * pair_count
        reference_truth_weight += reference_conc * pair_count
        daily_candidate_truth.append(candidate_conc)

    conflict_by_date = pd.DataFrame(conflict_rows)
    if total_pairs:
        pooled_candidate = candidate_truth_weight / total_pairs
        pooled_reference = reference_truth_weight / total_pairs
        conflict_summary = {
            "pair_count": int(total_pairs),
            "date_count": int(len(conflict_rows)),
            "candidate_model_candidate_truth_concordance": float(pooled_candidate),
            "candidate_model_reference_truth_concordance": float(pooled_reference),
            "candidate_truth_advantage": float(pooled_candidate - pooled_reference),
            "mean_daily_candidate_truth_concordance": float(np.mean(daily_candidate_truth)),
            "median_daily_candidate_truth_concordance": float(np.median(daily_candidate_truth)),
            "date_share_candidate_truth_above_half": float(
                np.mean(np.asarray(daily_candidate_truth, dtype=np.float64) > 0.5)
            ),
            "natural_null_concordance": 0.5,
        }
    else:
        conflict_summary = {
            "pair_count": 0,
            "date_count": 0,
            "candidate_model_candidate_truth_concordance": None,
            "candidate_model_reference_truth_concordance": None,
            "candidate_truth_advantage": None,
            "mean_daily_candidate_truth_concordance": None,
            "median_daily_candidate_truth_concordance": None,
            "date_share_candidate_truth_above_half": None,
            "natural_null_concordance": 0.5,
        }

    metrics = {
        "common_oos_rows": int(len(frame)),
        "target_contract": {
            "candidate_never_exceeds_reference_except_empty_prebreach_zero_floor": True,
            "historical_reference_pure_mfe_has_zero_floor": False,
            "legal_zero_floor_exception": (
                "candidate~=0R and reference<0R only; this is first-bar empty-prebreach "
                "semantics versus historical MR-13K unfloored Pure-MFE"
            ),
            "zero_floor_exception_row_count": int(len(zero_floor_rows)),
            "reference_truth_source": "MR-13AB frozen OOS embedded reference_target_raw_r",
            "tolerance_r": tolerance,
        },
        "cross_target_daily_spearman": cross_target,
        "changed_rows": changed_summary,
        "unchanged_rows": unchanged_summary,
        "conflict_pairs": conflict_summary,
        "top_fraction": float(top_fraction),
        "percentile_method": "same_date_average_zero_based_rank_over_candidate_oos",
    }
    return metrics, changed_rows, conflict_by_date


def _read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"無法讀取continuous ranker report: {path.name}; {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"continuous ranker report根節點必須是object: {path.name}")
    return payload


def _validate_candidate_artifacts(
    *,
    root: Path,
    filter_id: str,
    architecture: str,
    profile_id: str,
    research_id: str,
    target_id: str,
    seed: int,
    expected_reference_profile: str,
    expected_reference_target: str,
) -> dict[str, Any]:
    score_path = resolve_continuous_ranker_oos_score_path(
        root, filter_id, architecture, profile_id
    )
    report_path = score_path.parent / CONTINUOUS_RANKER_REPORT_FILENAME
    if not score_path.is_file():
        raise FileNotFoundError("缺少MR-13AB frozen Forward OOS score: " + _relative(score_path, root))
    if not report_path.is_file():
        raise FileNotFoundError("缺少MR-13AB continuous ranker report: " + _relative(report_path, root))

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

    reference_eval = dict(report.get("reference_target_evaluation") or {})
    if not bool(reference_eval.get("available", False)):
        raise ValueError(f"{research_id} report缺少reference target evaluation")
    if str(reference_eval.get("reference_profile") or "") != str(expected_reference_profile):
        raise ValueError(f"{research_id} report reference profile不一致")
    if str(reference_eval.get("reference_target_id") or "") != str(expected_reference_target):
        raise ValueError(f"{research_id} report reference target不一致")
    if bool(reference_eval.get("used_for_training_or_epoch_selection", True)):
        raise ValueError(f"{research_id} reference target不得參與training/epoch selection")

    artifact = dict(dict(report.get("artifacts") or {}).get("oos_scores_gzip") or {})
    if str(artifact.get("filename") or "") != score_path.name:
        raise ValueError(f"{research_id} report OOS score filename不一致")
    expected_hash = str(artifact.get("sha256") or "").lower()
    actual_hash = compute_file_sha256(score_path).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(f"{research_id} frozen OOS score SHA256與report不一致")
    return {
        "score_path": score_path,
        "report_path": report_path,
        "score_sha256": actual_hash,
        "report_sha256": compute_file_sha256(report_path).lower(),
        "report": report,
    }


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

    candidate = _validate_candidate_artifacts(
        root=root,
        filter_id=filter_id,
        architecture=architecture,
        profile_id=candidate_profile,
        research_id=candidate_research_id,
        target_id=candidate_target_id,
        seed=seed,
        expected_reference_profile=reference_profile,
        expected_reference_target=reference_target_id,
    )
    return {**required, "seed": seed, "candidate": candidate}


def preflight(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        contract = _source_contract(definition, project_root=root)
    except (ValueError, OSError, FileNotFoundError) as exc:
        source = definition.source
        score_path = resolve_continuous_ranker_oos_score_path(
            root,
            str(source.get("filter_id") or "").strip(),
            str(source.get("model_architecture") or "").strip(),
            str(source.get("candidate_profile_id") or "").strip(),
        )
        return {
            "status": "BLOCKED",
            "blockers": [str(exc)],
            "source_paths": [],
            "source_path": _relative(score_path, root),
            "preparable": False,
        }
    return {
        "status": "READY",
        "blockers": [],
        "source_paths": [
            _relative(Path(contract["candidate"]["score_path"]), root),
            _relative(Path(contract["candidate"]["report_path"]), root),
        ],
        "source_path": _relative(Path(contract["candidate"]["score_path"]), root),
        "preparable": False,
    }


def collect_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    state = preflight(definition, project_root=Path(project_root))
    return {
        "status": str(state.get("status") or "BLOCKED"),
        "reason": "；".join(str(v) for v in state.get("blockers", ())),
        "source": {
            "display": "MR-13AB frozen OOS score/report + embedded MR-13K reference target truth",
            "path": str(state.get("source_path") or ""),
            "paths": list(state.get("source_paths", ())),
            "preparable": False,
        },
    }


def _fingerprint(definition: AuditDefinition, contract: Mapping[str, Any]) -> str:
    payload = {
        "audit": definition.as_dict(),
        "candidate_score_sha256": contract["candidate"]["score_sha256"],
        "candidate_report_sha256": contract["candidate"]["report_sha256"],
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
        " MR-13AB Survival Increment｜Reference-Target Controlled Frozen Score Audit",
        "=" * 100,
        f"Audit     ：{result.get('audit_id', '')}",
        "Contract  ：READ ONLY / MR-13AB frozen Forward OOS only / embedded MR-13K target truth / no training / no PIT / no replay",
        f"Common OOS：{int(metrics.get('common_oos_rows', 0)):,}",
        "",
        "1. Frozen score vs two target truths",
        "------------------------------------",
        "Truth target                     Mean Daily rho",
        "-------------------------------  --------------",
        "MR-13AB first-breach Pure-MFE    {:>14}".format(
            _fmt(dict(cross.get("candidate_model_vs_candidate_target") or {}).get("mean_daily_spearman"))
        ),
        "MR-13K full-horizon Pure-MFE     {:>14}".format(
            _fmt(dict(cross.get("candidate_model_vs_reference_target") or {}).get("mean_daily_spearman"))
        ),
        "Target↔Target                    {:>14}".format(
            _fmt(dict(cross.get("candidate_target_vs_reference_target") or {}).get("mean_daily_spearman"))
        ),
        "",
        "2. Changed-row correction",
        "-------------------------",
        f"Changed rows                              ：{int(changed.get('row_count', 0)):,} ({_pct(changed.get('row_rate'))})",
        f"Survival-demotion rows                    ：{int(changed.get('survival_demotion_row_count', 0)):,}",
        f"Empty-prebreach zero-floor rows           ：{int(changed.get('zero_floor_exception_row_count', 0)):,} ({_pct(changed.get('zero_floor_exception_row_rate'))})",
        f"Mean target correction (K - AB)           ：{_fmt(changed.get('mean_target_correction_r'))}R",
        f"Mean reference-target percentile          ：{_fmt(changed.get('mean_reference_target_percentile'))}",
        f"Mean first-breach-target percentile       ：{_fmt(changed.get('mean_candidate_target_percentile'))}",
        f"Mean frozen-score percentile              ：{_fmt(changed.get('mean_candidate_score_percentile'))}",
        f"Mean target rank correction               ：{_fmt(changed.get('mean_target_rank_correction'))}",
        f"Score residual vs reference-target rank   ：{_fmt(changed.get('mean_score_minus_reference_rank'))}",
        f"Rank-correction alignment rho             ：{_fmt(changed.get('rank_correction_alignment_spearman'))}",
        f"Reference-only top-{metrics.get('top_fraction', 0.1):.0%} rows          ：{int(changed.get('reference_only_top_fraction_rows', 0)):,}",
        f"Still top-{metrics.get('top_fraction', 0.1):.0%} under MR-13AB score      ：{_pct(changed.get('reference_only_top_retention_rate_under_candidate_model'))}",
        "",
        "3. Target-order conflict pairs",
        "------------------------------",
        f"Conflict pairs                            ：{int(conflict.get('pair_count', 0)):,}",
        f"Conflict dates                            ：{int(conflict.get('date_count', 0)):,}",
        f"MR-13AB score concordance vs AB truth     ：{_pct(conflict.get('candidate_model_candidate_truth_concordance'))}",
        f"MR-13AB score concordance vs K truth      ：{_pct(conflict.get('candidate_model_reference_truth_concordance'))}",
        f"AB-truth advantage                        ：{_pct(conflict.get('candidate_truth_advantage'))}",
        f"Mean daily concordance vs AB truth        ：{_pct(conflict.get('mean_daily_candidate_truth_concordance'))}",
        f"Conflict dates > 50% AB-truth concordance ：{_pct(conflict.get('date_share_candidate_truth_above_half'))}",
        "",
        "Target edge：historical MR-13K Pure-MFE不做0R floor；first-bar breach使AB empty-prebreach=0R時，AB可合法高於負值K reference。",
        "Natural null：在兩Target要求相反排序的pair上，50%表示frozen score沒有偏向first-breach或full-horizon ordering。",
        "Decision boundary：只判斷MR-13AB frozen score是否在target真正改寫／ordering衝突處呈現first-passage survival ordering；",
        "本Audit不產生barrier、loss、threshold、architecture、portfolio rule或任何可回流fitting的參數。",
    ]
    return "\n".join(lines)


def _render_markdown(result: Mapping[str, Any]) -> str:
    return "```text\n" + render_result(result) + "\n```\n"


def run_audit(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract = _source_contract(definition, project_root=root)
    candidate_frame = read_breakout_quality_csv(contract["candidate"]["score_path"])
    tolerance = float(definition.dimensions.get("changed_tolerance_r", 1e-6))
    top_fraction = float(definition.dimensions.get("top_fraction", 0.10))
    metrics, changed_rows, conflict_by_date = analyze_candidate_score_frame(
        candidate_frame, tolerance_r=tolerance, top_fraction=top_fraction
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
        "schema_version": 2,
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
            "reference_artifact_requirement": "none; reference truth is embedded in candidate frozen score",
        },
        "metrics": metrics,
        "interpretation_boundary": {
            "changed_rows": "Only rows whose MR-13AB first-breach target is strictly below its embedded MR-13K full-horizon Pure-MFE reference target.",
            "conflict_pairs": "Only same-date pairs where the two target definitions demand opposite strict ordering; at least one row must be changed.",
            "increment_test": "On conflict pairs, concordance above the natural 50% null indicates the MR-13AB frozen score follows first-breach ordering more often than full-horizon ordering.",
            "causality": "This frozen-score diagnostic is evidence of learned survival ordering; it is not a strategy-performance claim.",
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
        "ticker",
        "date",
        "group_index",
        "candidate_target_r",
        "reference_target_r",
        "target_correction_r",
        "candidate_score",
        "candidate_score_percentile",
        "candidate_target_percentile",
        "reference_target_percentile",
        "target_rank_correction",
        "score_minus_reference_rank",
    ]].copy()
    changed_export.to_csv(changed_path, index=False, encoding="utf-8-sig", compression="gzip")
    conflict_by_date.to_csv(conflict_path, index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": 2,
        "audit_definition": definition.as_dict(),
        "config_fingerprint": fingerprint,
        "source_refs": {
            "candidate_score": {
                "path": _relative(contract["candidate"]["score_path"], root),
                "sha256": contract["candidate"]["score_sha256"],
            },
            "candidate_report": {
                "path": _relative(contract["candidate"]["report_path"], root),
                "sha256": contract["candidate"]["report_sha256"],
            },
            "reference_truth": {
                "research_id": contract["reference_research_id"],
                "profile_id": contract["reference_profile_id"],
                "target_id": contract["reference_target_id"],
                "storage": "candidate_score.reference_target_raw_r",
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
    "analyze_candidate_score_frame",
    "collect_status",
    "preflight",
    "render_result",
    "run_audit",
    "run_formal_audit",
]
