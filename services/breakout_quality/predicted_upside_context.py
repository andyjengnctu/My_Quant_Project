"""Build the isolated PIT-safe Stage-1 upside context required by MR-13AC."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.breakout_quality import get_breakout_quality_workflow_settings
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.dataset_readiness import collect_dataset_readiness
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
)
from filters.breakout_quality.predicted_upside_context import (
    CONTEXT_COLUMN,
    PREDICTED_UPSIDE_CONTEXT_FILENAME,
    PREDICTED_UPSIDE_CONTEXT_MANIFEST_FILENAME,
    STAGE1_ARCHITECTURE,
    STAGE1_PROFILE,
    STAGE1_RESEARCH_ID,
    STAGE1_SEED,
    predicted_upside_context_contract,
    resolve_predicted_upside_context_dir,
)
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from services.breakout_quality.point_in_time_scores import build_cross_fitted_context_scores


def _load_stage1_scores(path: Path, *, phase: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 predicted-upside score不存在: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "ticker", "date", "breakout_quality_score", "fold_id", "model_information_cutoff"
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage-1 predicted-upside score缺少欄位: {missing}")
    frame = frame[list(required)].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["model_information_cutoff"] = pd.to_datetime(
        frame["model_information_cutoff"], errors="raise"
    ).dt.normalize()
    frame["predicted_upside_score"] = pd.to_numeric(
        frame.pop("breakout_quality_score"), errors="raise"
    )
    score = frame["predicted_upside_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(score).all() or bool(((score < 0.0) | (score > 1.0)).any()):
        raise ValueError("Stage-1 predicted-upside score必須finite且位於[0,1]")
    valid = np.ones(len(frame), dtype=bool)
    frame[CONTEXT_COLUMN] = build_same_date_percentile_targets(
        score, valid, frame["date"]
    )
    frame["context_phase"] = str(phase)
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Stage-1 predicted-upside score ticker/date重複")
    return frame


def build_predicted_upside_context(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    dataset: str,
    max_tickers: int = 0,
) -> Path:
    """Build/rebuild the MR-13AC context from isolated MR-13K PIT-safe scoring.

    Selection rows use annual expanding cross-fits.  Forward OOS rows use one
    fixed pre-OOS Stage-1 fit across the entire OOS block.  No Stage-2 label or
    OOS statistic enters Stage-1 fitting or the context percentile transform.
    """

    root = Path(project_root).resolve()
    readiness = collect_dataset_readiness(
        root,
        filter_id=str(filter_id),
        dataset=str(dataset),
        max_tickers=int(max_tickers),
        model_architecture=str(STAGE1_ARCHITECTURE),
    )
    if not readiness.ready or readiness.summary is None:
        raise RuntimeError("MR-13AC predicted-upside context需要READY canonical Dataset")
    source_range = dict(readiness.summary.get("source_data_date_range") or {})
    source_end = str(source_range.get("end") or "").strip()
    if not source_end:
        raise ValueError("Dataset summary缺少source_data_date_range.end")
    outer = resolve_breakout_quality_outer_policy(root, source_data_end_date=source_end)
    selection_end = str(outer["selection_end_date"])
    oos_start = str(outer["oos_start_date"])
    oos_end = str(outer["effective_oos_end_date"])

    directory = resolve_predicted_upside_context_dir(
        root,
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
    )
    selection_dir = directory / "stage1_selection_crossfit"
    forward_dir = directory / "stage1_forward_fixed"
    directory.mkdir(parents=True, exist_ok=True)

    stage1_settings = get_breakout_quality_workflow_settings(
        experiment_profile=STAGE1_PROFILE
    )
    common = dict(
        filter_id=str(filter_id),
        model_architecture=STAGE1_ARCHITECTURE,
        experiment_profile=STAGE1_PROFILE,
        inner_validation_months=int(stage1_settings.point_in_time_inner_validation_months),
        seed=int(STAGE1_SEED),
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
    selection = _load_stage1_scores(selection_score_path, phase="selection_crossfit")
    forward = _load_stage1_scores(forward_score_path, phase="forward_fixed_pre_oos")
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
    score_path = directory / PREDICTED_UPSIDE_CONTEXT_FILENAME
    combined.to_csv(score_path, index=False, encoding="utf-8-sig", compression="gzip")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT",
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
        "contract": predicted_upside_context_contract(),
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
            "research_id": STAGE1_RESEARCH_ID,
            "profile": STAGE1_PROFILE,
            "architecture": STAGE1_ARCHITECTURE,
            "seed": int(STAGE1_SEED),
            "selection_score": build_file_manifest(selection_score_path),
            "selection_manifest": build_file_manifest(selection_manifest_path),
            "forward_score": build_file_manifest(forward_score_path),
            "forward_manifest": build_file_manifest(forward_manifest_path),
        },
        "artifact": build_file_manifest(score_path),
    }
    (directory / PREDICTED_UPSIDE_CONTEXT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return score_path


__all__ = ["build_predicted_upside_context"]
