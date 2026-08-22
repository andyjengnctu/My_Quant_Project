"""Deterministic builder for frozen MR-13E percentile -> Expected-R artifacts."""

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
from filters.breakout_quality.expected_r_calibration import (
    EXPECTED_R_CALIBRATION_METHOD,
    EXPECTED_R_CALIBRATION_SCHEMA_VERSION,
    resolve_expected_r_calibration_paths,
)
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.rank_calibration import (
    add_daily_score_percentile,
    build_canonical_target_frame,
    mature_target_rows,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_continuous_ranker_oos_contract,
    load_continuous_ranker_oos_score_table,
    load_selection_point_in_time_ranking_contract,
    load_selection_point_in_time_score_table,
)



def _fit_nonnegative_affine(frame: pd.DataFrame, *, cutoff_exclusive: str) -> dict[str, Any]:
    cutoff = pd.Timestamp(cutoff_exclusive).normalize()
    eligible = mature_target_rows(frame, cutoff_exclusive=cutoff.strftime("%Y-%m-%d"))
    eligible = eligible[np.isfinite(eligible["daily_score_percentile"]) & np.isfinite(eligible["target_raw_r"])]
    if len(eligible) < 2:
        raise ValueError(
            f"Expected-R calibration成熟樣本不足: cutoff={cutoff.date()}, n={len(eligible)}"
        )
    x = eligible["daily_score_percentile"].to_numpy(dtype=np.float64)
    y = eligible["target_raw_r"].to_numpy(dtype=np.float64)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    variance = float(np.mean((x - x_mean) ** 2))
    if variance <= 0.0:
        raise ValueError("Expected-R calibration percentile沒有變異")
    unconstrained_slope = float(np.mean((x - x_mean) * (y - y_mean)) / variance)
    slope = max(0.0, unconstrained_slope)
    intercept = float(y_mean - slope * x_mean)
    fitted = intercept + slope * x
    mae = float(np.mean(np.abs(fitted - y)))
    mse = float(np.mean((fitted - y) ** 2))
    return {
        "cutoff_exclusive": str(cutoff.date()),
        "sample_count": int(len(eligible)),
        "intercept": intercept,
        "slope": slope,
        "unconstrained_slope": unconstrained_slope,
        "slope_constrained_to_nonnegative": bool(unconstrained_slope < 0.0),
        "target_mean_r": y_mean,
        "mae_r": mae,
        "rmse_r": float(math.sqrt(mse)),
    }


def _selection_score_frame(root: Path, *, filter_id: str, model_architecture: str, experiment_profile: str) -> tuple[pd.DataFrame, Any]:
    contract = load_selection_point_in_time_ranking_contract(
        str(root), filter_id, model_architecture, experiment_profile
    )
    table = load_selection_point_in_time_score_table(
        str(root), filter_id, model_architecture, experiment_profile
    ).reset_index()
    table = table.rename(columns={"breakout_quality_score": "breakout_quality_score"})
    return table, contract


def _runtime_score_frame(root: Path, *, filter_id: str, model_architecture: str, experiment_profile: str, phase_id: str) -> tuple[pd.DataFrame, Any, str]:
    if phase_id == "selection_pit":
        table, contract = _selection_score_frame(
            root, filter_id=filter_id, model_architecture=model_architecture, experiment_profile=experiment_profile
        )
        return table, contract, SCORE_SOURCE_SELECTION_POINT_IN_TIME
    contract = load_continuous_ranker_oos_contract(
        str(root), filter_id, model_architecture, experiment_profile
    )
    table = load_continuous_ranker_oos_score_table(
        str(root), filter_id, model_architecture, experiment_profile
    ).reset_index().rename(columns={"model_score": "breakout_quality_score"})
    return table, contract, SCORE_SOURCE_CONTINUOUS_RANKER_OOS


def _render_report(
    *,
    phase_id: str,
    method: str,
    fit_rows: list[dict[str, Any]],
    lookup_count: int,
    source_score: str,
    fit_score: str,
    forward_target_used: bool,
    forward_frozen_cutoff_exclusive: str | None,
) -> str:
    lines = [
        "# Frozen MR-13E Expected-R Calibration",
        "",
        f"- Phase: `{phase_id}`",
        f"- Method: `{method}`",
        "- Input: frozen MR-13E daily score percentile",
        "- Output: Expected R",
        f"- Lookup rows: {lookup_count:,}",
        f"- Runtime score source: `{source_score}`",
        f"- Fit score source: `{fit_score}`",
        f"- Forward target used for fit: `{forward_target_used}`",
        "- Portfolio sizing / K / R0 / execution are not part of calibration.",
        "",
        "| cutoff exclusive | samples | intercept | slope | raw slope | MAE R | RMSE R | constrained |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in fit_rows:
        lines.append(
            f"| {row['cutoff_exclusive']} | {row['sample_count']:,} | {row['intercept']:.6f} | "
            f"{row['slope']:.6f} | {row['unconstrained_slope']:.6f} | {row['mae_r']:.6f} | "
            f"{row['rmse_r']:.6f} | {row['slope_constrained_to_nonnegative']} |"
        )
    lines.extend([
        "",
        "Selection uses expanding/PIT calibration: each strategy year is fitted only from scores whose target evaluation completed before that year.",
        (
            "Forward uses one mapping frozen at runtime execution start "
            f"({forward_frozen_cutoff_exclusive}) from Selection PIT scores/targets only; "
            "Forward labels/targets are never read for fitting."
            if forward_frozen_cutoff_exclusive
            else "Forward frozen cutoff is not applicable to Selection PIT calibration."
        ),
        "",
    ])
    return "\n".join(lines)


def build_expected_r_calibration_artifact(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    phase_id: str,
    selection_runtime_start_date: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    phase = str(phase_id).strip()
    if phase not in {"selection_pit", "forward_oos"}:
        raise ValueError(f"Expected-R phase不支援: {phase!r}")

    fit_scores, fit_contract = _selection_score_frame(
        root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )
    runtime_scores, runtime_contract, runtime_source = _runtime_score_frame(
        root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        phase_id=phase,
    )
    fit_scores = add_daily_score_percentile(fit_scores)
    runtime_scores = add_daily_score_percentile(runtime_scores)

    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    targets = build_canonical_target_frame(bundle)
    fit_frame = fit_scores.merge(
        targets[["group_index", "label_eval_end_date", "target_raw_r", "target_valid"]],
        on="group_index",
        how="left",
        validate="one_to_one",
    )
    if fit_frame["target_valid"].isna().any():
        raise ValueError("Expected-R fit score group_index無法完整對應canonical target")

    fit_rows: list[dict[str, Any]] = []
    output_parts: list[pd.DataFrame] = []
    forward_frozen_cutoff_exclusive: str | None = None
    if phase == "selection_pit":
        if not selection_runtime_start_date:
            raise ValueError("Selection Expected-R calibration缺少策略比較start_date")
        selection_start = pd.Timestamp(selection_runtime_start_date).normalize()
        runtime_dates = pd.to_datetime(runtime_scores["date"], errors="raise")
        runtime_mask = runtime_dates >= selection_start
        years = sorted({int(value) for value in runtime_dates.loc[runtime_mask].dt.year})
        for year in years:
            year_start = pd.Timestamp(year=year, month=1, day=1)
            cutoff_ts = max(selection_start, year_start)
            cutoff = cutoff_ts.strftime("%Y-%m-%d")
            params = _fit_nonnegative_affine(fit_frame, cutoff_exclusive=cutoff)
            fit_rows.append(params)
            part = runtime_scores[(runtime_dates.dt.year == year) & runtime_mask].copy()
            if part.empty:
                continue
            part["expected_r"] = params["intercept"] + params["slope"] * part["daily_score_percentile"]
            part["calibration_cutoff_exclusive"] = params["cutoff_exclusive"]
            part["calibration_sample_count"] = params["sample_count"]
            output_parts.append(part)
    else:
        execution_start = str(getattr(runtime_contract, "execution_start", "") or "").strip()
        if not execution_start:
            raise ValueError("Forward Expected-R calibration缺少frozen OOS execution_start contract")
        forward_frozen_cutoff_exclusive = pd.Timestamp(execution_start).normalize().strftime("%Y-%m-%d")
        params = _fit_nonnegative_affine(
            fit_frame,
            cutoff_exclusive=forward_frozen_cutoff_exclusive,
        )
        fit_rows.append(params)
        part = runtime_scores.copy()
        part["expected_r"] = params["intercept"] + params["slope"] * part["daily_score_percentile"]
        part["calibration_cutoff_exclusive"] = params["cutoff_exclusive"]
        part["calibration_sample_count"] = params["sample_count"]
        output_parts.append(part)

    if not output_parts:
        raise ValueError("Expected-R calibration沒有可輸出的runtime rows")
    lookup = pd.concat(output_parts, ignore_index=True)
    required = [
        "ticker", "date", "group_index", "breakout_quality_score", "daily_score_percentile",
        "expected_r", "calibration_cutoff_exclusive", "calibration_sample_count",
    ]
    lookup = lookup[required].copy()
    lookup["date"] = pd.to_datetime(lookup["date"], errors="raise").dt.strftime("%Y-%m-%d")
    lookup = lookup.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    if lookup.duplicated(["ticker", "date"]).any():
        raise ValueError("Expected-R output同ticker/date重複")
    if not np.isfinite(lookup["expected_r"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Expected-R output包含非有限值")

    paths = resolve_expected_r_calibration_paths(
        root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        phase_id=phase,
    )
    paths["dir"].mkdir(parents=True, exist_ok=True)
    lookup.to_csv(paths["lookup"], index=False, compression="gzip")

    runtime_score_path = Path(runtime_contract.score_path).resolve()
    fit_score_path = Path(fit_contract.score_path).resolve()
    manifest = {
        "schema_version": EXPECTED_R_CALIBRATION_SCHEMA_VERSION,
        "method": EXPECTED_R_CALIBRATION_METHOD,
        "identity": {
            "filter_id": filter_id,
            "model_architecture": model_architecture,
            "experiment_profile": experiment_profile,
            "phase_id": phase,
        },
        "source_artifacts": {
            "runtime_score_source": runtime_source,
            "runtime_score_path": project_relative_display_path(runtime_score_path, project_root=root),
            "runtime_score_sha256": compute_file_sha256(runtime_score_path),
            "fit_score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            "fit_score_path": project_relative_display_path(fit_score_path, project_root=root),
            "fit_score_sha256": compute_file_sha256(fit_score_path),
        },
        "target_contract": _json_native(bundle.target_manifest),
        "fit_contract": {
            "selection_expanding_by_calendar_year": phase == "selection_pit",
            "selection_runtime_start_date": (
                pd.Timestamp(selection_runtime_start_date).normalize().strftime("%Y-%m-%d")
                if phase == "selection_pit" and selection_runtime_start_date
                else None
            ),
            "target_maturity_rule": "label_eval_end_date < calibration_cutoff_exclusive",
            "forward_frozen_cutoff_exclusive": forward_frozen_cutoff_exclusive,
            "forward_target_used_for_fit": False,
            "score_percentile_uses_target": False,
            "negative_expected_r_allowed": True,
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
    report = _render_report(
        phase_id=phase,
        method=EXPECTED_R_CALIBRATION_METHOD,
        fit_rows=fit_rows,
        lookup_count=len(lookup),
        source_score=project_relative_display_path(runtime_score_path, project_root=root),
        fit_score=project_relative_display_path(fit_score_path, project_root=root),
        forward_target_used=False,
        forward_frozen_cutoff_exclusive=forward_frozen_cutoff_exclusive,
    )
    paths["report"].write_text(report, encoding="utf-8")
    return {"manifest": manifest, "paths": paths}


__all__ = ["build_expected_r_calibration_artifact"]
