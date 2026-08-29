"""Shared PIT-safe Stage-1 predicted-context producer for stacked daily rankers."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from config.breakout_quality import get_breakout_quality_workflow_settings
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.dataset_readiness import collect_dataset_readiness
from filters.breakout_quality.predicted_context_artifact import (
    get_predicted_context_artifact_spec,
    resolve_predicted_context_dir,
)
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
)
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from services.breakout_quality.point_in_time_scores import build_cross_fitted_context_scores


def load_stage1_context_scores(
    path: Path,
    *,
    phase: str,
    predicted_score_column: str,
    context_column: str,
    context_label: str,
) -> pd.DataFrame:
    """Load one Stage-1 PIT score block and convert the score to same-date percentile."""

    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 {context_label} score不存在: {path}")
    frame = pd.read_csv(
        path, encoding="utf-8-sig", dtype={"ticker": "string"}, low_memory=False
    )
    required = {
        "ticker", "date", "breakout_quality_score", "fold_id", "model_information_cutoff"
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage-1 {context_label} score缺少欄位: {missing}")
    frame = frame[list(required)].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["model_information_cutoff"] = pd.to_datetime(
        frame["model_information_cutoff"], errors="raise"
    ).dt.normalize()
    frame[predicted_score_column] = pd.to_numeric(
        frame.pop("breakout_quality_score"), errors="raise"
    )
    score = frame[predicted_score_column].to_numpy(dtype=np.float64)
    if not np.isfinite(score).all() or bool(((score < 0.0) | (score > 1.0)).any()):
        raise ValueError(f"Stage-1 {context_label} score必須finite且位於[0,1]")
    valid = np.ones(len(frame), dtype=bool)
    frame[context_column] = build_same_date_percentile_targets(
        score, valid, frame["date"]
    )
    frame["context_phase"] = str(phase)
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError(f"Stage-1 {context_label} score ticker/date重複")
    return frame


def build_predicted_context(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    dataset: str,
    max_tickers: int,
    stage1_profile: str,
    stage1_architecture: str,
    stage1_research_id: str,
    stage1_seed: int,
    context_dir_resolver: Callable[..., Path],
    context_filename: str,
    context_manifest_filename: str,
    predicted_score_column: str,
    context_column: str,
    context_contract: dict[str, Any],
    context_label: str,
) -> Path:
    """Build Selection cross-fit + single fixed-forward Stage-1 context artifact."""

    root = Path(project_root).resolve()
    readiness = collect_dataset_readiness(
        root,
        filter_id=str(filter_id),
        dataset=str(dataset),
        max_tickers=int(max_tickers),
        model_architecture=str(stage1_architecture),
    )
    if not readiness.ready or readiness.summary is None:
        raise RuntimeError(f"{context_label} context需要READY canonical Dataset")
    source_range = dict(readiness.summary.get("source_data_date_range") or {})
    source_end = str(source_range.get("end") or "").strip()
    if not source_end:
        raise ValueError("Dataset summary缺少source_data_date_range.end")
    outer = resolve_breakout_quality_outer_policy(root, source_data_end_date=source_end)
    selection_end = str(outer["selection_end_date"])
    oos_start = str(outer["oos_start_date"])
    oos_end = str(outer["effective_oos_end_date"])

    directory = context_dir_resolver(
        root,
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
    )
    selection_dir = directory / "stage1_selection_crossfit"
    forward_dir = directory / "stage1_forward_fixed"
    directory.mkdir(parents=True, exist_ok=True)

    stage1_settings = get_breakout_quality_workflow_settings(
        experiment_profile=str(stage1_profile)
    )
    common = dict(
        filter_id=str(filter_id),
        model_architecture=str(stage1_architecture),
        experiment_profile=str(stage1_profile),
        inner_validation_months=int(stage1_settings.point_in_time_inner_validation_months),
        seed=int(stage1_seed),
        resume=True,
        allow_stale_source=False,
    )
    build_cross_fitted_context_scores(
        **common,
        score_start_date="auto",
        score_end_date=selection_end,
        fold_months=int(stage1_settings.point_in_time_fold_months),
        point_in_time_dir_override=str(selection_dir),
        single_score_block=False,
    )
    build_cross_fitted_context_scores(
        **common,
        score_start_date=oos_start,
        score_end_date=oos_end,
        fold_months=int(stage1_settings.point_in_time_fold_months),
        point_in_time_dir_override=str(forward_dir),
        single_score_block=True,
    )

    selection_score_path = selection_dir / SELECTION_POINT_IN_TIME_SCORE_FILENAME
    selection_manifest_path = selection_dir / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
    forward_score_path = forward_dir / SELECTION_POINT_IN_TIME_SCORE_FILENAME
    forward_manifest_path = forward_dir / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
    selection = load_stage1_context_scores(
        selection_score_path,
        phase="selection_crossfit",
        predicted_score_column=predicted_score_column,
        context_column=context_column,
        context_label=context_label,
    )
    forward = load_stage1_context_scores(
        forward_score_path,
        phase="forward_fixed_pre_oos",
        predicted_score_column=predicted_score_column,
        context_column=context_column,
        context_label=context_label,
    )
    if bool((selection["date"] > pd.Timestamp(selection_end)).any()):
        raise ValueError("Selection Stage-1 context超出selection_end")
    if bool(((forward["date"] < pd.Timestamp(oos_start)) | (forward["date"] > pd.Timestamp(oos_end))).any()):
        raise ValueError("Forward Stage-1 context超出OOS period")
    if bool((selection["model_information_cutoff"] >= selection["date"]).any()):
        raise ValueError("Selection Stage-1 context不是PIT-safe")
    if len(forward["fold_id"].astype(str).drop_duplicates()) != 1:
        raise ValueError("Forward Stage-1 context必須只有一個fixed score block")
    if len(forward["model_information_cutoff"].drop_duplicates()) != 1:
        raise ValueError("Forward Stage-1 context必須只有一個pre-OOS information cutoff")
    if not bool((forward["model_information_cutoff"] < pd.Timestamp(oos_start)).all()):
        raise ValueError("Forward Stage-1 model information cutoff必須早於OOS start")

    combined = pd.concat([selection, forward], ignore_index=True)
    if combined.duplicated(["ticker", "date"]).any():
        raise ValueError("Selection/Forward Stage-1 context出現ticker/date重疊")
    combined = combined.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    score_path = directory / str(context_filename)
    combined.to_csv(score_path, index=False, encoding="utf-8-sig", compression="gzip")
    manifest = {
        "schema_version": int(context_contract["schema_version"]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT",
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
        "contract": dict(context_contract),
        "dataset_policy": readiness.summary.get("policy"),
        "dataset_artifact_source": readiness.summary.get("dataset_artifacts"),
        "periods": {
            "selection_end": selection_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
        },
        "rows": {
            "total": int(len(combined)),
            "selection_crossfit": int(len(selection)),
            "forward_fixed_pre_oos": int(len(forward)),
        },
        "source_stage1": {
            "research_id": str(stage1_research_id),
            "profile": str(stage1_profile),
            "architecture": str(stage1_architecture),
            "seed": int(stage1_seed),
            "selection_score": build_file_manifest(selection_score_path),
            "selection_manifest": build_file_manifest(selection_manifest_path),
            "forward_score": build_file_manifest(forward_score_path),
            "forward_manifest": build_file_manifest(forward_manifest_path),
        },
        "artifact": build_file_manifest(score_path),
    }
    (directory / str(context_manifest_filename)).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return score_path


def build_registered_predicted_context(
    source: str,
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    dataset: str,
    max_tickers: int = 0,
) -> Path:
    """Build one registered predicted-context capability without consumer branching."""

    spec = get_predicted_context_artifact_spec(source)
    owner_architecture = spec.resolve_owner_architecture(model_architecture)
    owner_profile = spec.resolve_owner_profile(experiment_profile)
    return build_predicted_context(
        project_root=project_root,
        filter_id=filter_id,
        model_architecture=owner_architecture,
        experiment_profile=owner_profile,
        dataset=dataset,
        max_tickers=max_tickers,
        stage1_profile=spec.stage1_profile,
        stage1_architecture=spec.stage1_architecture,
        stage1_research_id=spec.stage1_research_id,
        stage1_seed=spec.stage1_seed,
        context_dir_resolver=partial(resolve_predicted_context_dir, spec.source),
        context_filename=spec.filename,
        context_manifest_filename=spec.manifest_filename,
        predicted_score_column=spec.predicted_score_column,
        context_column=spec.context_column,
        context_contract=spec.contract(),
        context_label=spec.producer_label,
    )


__all__ = ["build_predicted_context", "build_registered_predicted_context", "load_stage1_context_scores"]
