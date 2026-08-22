"""Deterministic Selection-PIT builder for frozen MR-13E rank -> Expected Excess-R."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.console_report import project_relative_display_path
from core.serialization_utils import json_native_value as _json_native
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.excess_r_calibration import (
    EXPECTED_EXCESS_R_CALIBRATION_METHOD,
    EXPECTED_EXCESS_R_CALIBRATION_SCHEMA_VERSION,
    resolve_expected_excess_r_calibration_paths,
)
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.rank_calibration import (
    add_daily_excess_r,
    add_daily_score_percentile,
    build_canonical_target_frame,
    fit_weighted_increasing_isotonic,
    mature_target_rows,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_selection_point_in_time_ranking_contract,
    load_selection_point_in_time_score_table,
)



def _selection_score_frame(
    root: Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> tuple[pd.DataFrame, Any]:
    contract = load_selection_point_in_time_ranking_contract(
        str(root), filter_id, model_architecture, experiment_profile
    )
    table = load_selection_point_in_time_score_table(
        str(root), filter_id, model_architecture, experiment_profile
    ).reset_index()
    return table, contract


def _fit_isotonic_excess(
    frame: pd.DataFrame,
    *,
    canonical_targets: pd.DataFrame,
    cutoff_exclusive: str,
):
    mature = mature_target_rows(frame, cutoff_exclusive=cutoff_exclusive)
    mature_targets = add_daily_excess_r(
        mature_target_rows(canonical_targets, cutoff_exclusive=cutoff_exclusive)
    )
    mature = mature.drop(
        columns=["daily_target_mean_r", "target_excess_r"], errors="ignore"
    ).merge(
        mature_targets[["group_index", "daily_target_mean_r", "target_excess_r"]],
        on="group_index",
        how="left",
        validate="one_to_one",
    )
    mature = mature[
        np.isfinite(mature["daily_score_percentile"])
        & np.isfinite(mature["target_excess_r"])
    ].copy()
    if len(mature) < 2:
        raise ValueError(
            f"Expected Excess-R calibration成熟樣本不足: cutoff={cutoff_exclusive}, n={len(mature)}"
        )
    curve, summary = fit_weighted_increasing_isotonic(
        mature["daily_score_percentile"].to_numpy(dtype=np.float64),
        mature["target_excess_r"].to_numpy(dtype=np.float64),
    )
    summary = {
        "cutoff_exclusive": pd.Timestamp(cutoff_exclusive).normalize().strftime("%Y-%m-%d"),
        "mature_date_count": int(mature["date"].nunique()),
        **summary,
    }
    return curve, summary


def _render_report(*, fit_rows: list[dict[str, Any]], lookup_count: int, source_score: str) -> str:
    lines = [
        "# Frozen MR-13E Expected Excess-R Calibration",
        "",
        "- Phase: `selection_pit`",
        f"- Method: `{EXPECTED_EXCESS_R_CALIBRATION_METHOD}`",
        "- Input: frozen MR-13E daily score percentile",
        "- Target: daily raw R minus same-day daily-eligible target-valid mean R",
        "- Output: Expected Excess R (relative alpha, not absolute Expected R)",
        f"- Lookup rows: {lookup_count:,}",
        f"- Score source: `{source_score}`",
        "- Mapping: expanding/PIT nondecreasing isotonic PAVA; no percentile bins, slope, intercept, lambda or OOS fit.",
        "- K / R0 / sizing / cash / execution are not part of calibration.",
        "",
        "| cutoff exclusive | samples | dates | unique pct | blocks | target mean | pred mean | pred min | pred max | MAE R | RMSE R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fit_rows:
        lines.append(
            f"| {row['cutoff_exclusive']} | {row['sample_count']:,} | {row['mature_date_count']:,} | "
            f"{row['unique_percentile_count']:,} | {row['isotonic_block_count']:,} | "
            f"{row['target_mean_r']:.6f} | {row['prediction_mean_r']:.6f} | "
            f"{row['prediction_min_r']:.6f} | {row['prediction_max_r']:.6f} | "
            f"{row['mae_r']:.6f} | {row['rmse_r']:.6f} |"
        )
    lines.extend([
        "",
        "Each strategy year is fitted only from rows whose label evaluation ended before that year's cutoff.",
        "The same-day target mean is used only after that day's target has fully matured; it never enters runtime candidate ranking directly.",
        "Forward is intentionally not implemented for SR-C39. A new Forward arm may be created only after Selection decision.",
        "",
    ])
    return "\n".join(lines)


def build_expected_excess_r_calibration_artifact(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    phase_id: str,
    selection_runtime_start_date: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if str(phase_id).strip() != "selection_pit":
        raise ValueError("SR-C39 Expected Excess-R第一階段只允許selection_pit")
    selection_start = pd.Timestamp(selection_runtime_start_date).normalize()

    scores, score_contract = _selection_score_frame(
        root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )
    scores = add_daily_score_percentile(scores)
    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    targets = build_canonical_target_frame(bundle)
    fit_frame = scores.merge(
        targets[[
            "group_index",
            "label_eval_end_date",
            "target_raw_r",
            "target_valid",
        ]],
        on="group_index",
        how="left",
        validate="one_to_one",
    )
    if fit_frame["target_valid"].isna().any():
        raise ValueError("Expected Excess-R score group_index無法完整對應canonical target")

    runtime_dates = pd.to_datetime(scores["date"], errors="raise")
    runtime_mask = runtime_dates >= selection_start
    years = sorted({int(value) for value in runtime_dates.loc[runtime_mask].dt.year})
    fit_rows: list[dict[str, Any]] = []
    output_parts: list[pd.DataFrame] = []
    for year in years:
        cutoff_ts = max(selection_start, pd.Timestamp(year=year, month=1, day=1))
        cutoff = cutoff_ts.strftime("%Y-%m-%d")
        curve, summary = _fit_isotonic_excess(
            fit_frame, canonical_targets=targets, cutoff_exclusive=cutoff
        )
        fit_rows.append(summary)
        part = scores[(runtime_dates.dt.year == year) & runtime_mask].copy()
        if part.empty:
            continue
        part["expected_excess_r"] = curve.predict(
            part["daily_score_percentile"].to_numpy(dtype=np.float64)
        )
        part["calibration_cutoff_exclusive"] = cutoff
        part["calibration_sample_count"] = int(summary["sample_count"])
        output_parts.append(part)

    if not output_parts:
        raise ValueError("Expected Excess-R calibration沒有可輸出的runtime rows")
    lookup = pd.concat(output_parts, ignore_index=True)
    required = [
        "ticker",
        "date",
        "group_index",
        "breakout_quality_score",
        "daily_score_percentile",
        "expected_excess_r",
        "calibration_cutoff_exclusive",
        "calibration_sample_count",
    ]
    lookup = lookup[required].copy()
    lookup["ticker"] = lookup["ticker"].fillna("").astype(str).str.strip()
    lookup["date"] = pd.to_datetime(lookup["date"], errors="raise").dt.strftime("%Y-%m-%d")
    lookup = lookup.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    if lookup.duplicated(["ticker", "date"]).any():
        raise ValueError("Expected Excess-R output同ticker/date重複")
    if not np.isfinite(lookup["expected_excess_r"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Expected Excess-R output包含非有限值")

    paths = resolve_expected_excess_r_calibration_paths(
        root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        phase_id="selection_pit",
    )
    paths["dir"].mkdir(parents=True, exist_ok=True)
    lookup.to_csv(paths["lookup"], index=False, compression="gzip")

    score_path = Path(score_contract.score_path).resolve()
    manifest = {
        "schema_version": EXPECTED_EXCESS_R_CALIBRATION_SCHEMA_VERSION,
        "method": EXPECTED_EXCESS_R_CALIBRATION_METHOD,
        "identity": {
            "filter_id": filter_id,
            "model_architecture": model_architecture,
            "experiment_profile": experiment_profile,
            "phase_id": "selection_pit",
        },
        "source_artifacts": {
            "runtime_score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            "runtime_score_path": project_relative_display_path(score_path, project_root=root),
            "runtime_score_sha256": compute_file_sha256(score_path),
            "fit_score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            "fit_score_path": project_relative_display_path(score_path, project_root=root),
            "fit_score_sha256": compute_file_sha256(score_path),
        },
        "target_contract": _json_native(bundle.target_manifest),
        "fit_contract": {
            "selection_expanding_by_calendar_year": True,
            "selection_runtime_start_date": selection_start.strftime("%Y-%m-%d"),
            "target_maturity_rule": "label_eval_end_date < calibration_cutoff_exclusive",
            "target_semantic": "daily_raw_r_minus_same_day_daily_eligible_target_mean",
            "same_day_mean_scope": "daily_eligible_target_valid_rows_mature_at_cutoff",
            "mapping_family": "weighted_pava_piecewise_linear_interpolation",
            "monotonic_direction": "nondecreasing",
            "score_percentile_uses_target": False,
            "forward_target_used_for_fit": False,
            "absolute_expected_r_not_claimed": True,
        },
        "fit_rows": _json_native(fit_rows),
        "lookup_period": {"start": str(lookup["date"].min()), "end": str(lookup["date"].max())},
        "lookup_row_count": int(len(lookup)),
        "lookup_artifact": {
            "filename": paths["lookup"].name,
            "sha256": compute_file_sha256(paths["lookup"]),
            "size_bytes": int(paths["lookup"].stat().st_size),
        },
    }
    paths["manifest"].write_text(
        json.dumps(_json_native(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(
        _render_report(
            fit_rows=fit_rows,
            lookup_count=len(lookup),
            source_score=project_relative_display_path(score_path, project_root=root),
        ),
        encoding="utf-8",
    )
    return {"manifest": manifest, "paths": paths}


__all__ = ["build_expected_excess_r_calibration_artifact"]
