"""Rebuild continuous-ranker Forward-OOS score coverage from a frozen checkpoint.

This service never fits model weights.  It exists so model-work can repair a score-only
contract change without retraining a scientifically unchanged Selection-fitted checkpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
from core.console_report import project_relative_display_path
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.models.factory import build_model
from filters.breakout_quality.paths import resolve_filter_artifact_paths, resolve_filter_model_output_dir
from filters.breakout_quality.ranker_sample_contract import (
    build_score_eligibility_contract,
    resolve_forward_oos_score_group_ids,
    resolve_forward_oos_target_evaluable_group_ids,
)
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    DAILY_RANKER_OOS_SCORE_FILENAME,
    resolve_continuous_ranker_oos_score_path,
)
from filters.breakout_quality.workflow_io import write_json
from tools.filters.breakout_quality.continuous_ranker_pipeline import (
    build_percentile_target,
    build_training_scope_mask,
    load_continuous_ranker_data,
    predict_scores,
    resolve_ranker_execution_plan,
)
from tools.filters.breakout_quality import train_continuous_ranker as ranker_impl


def _read_json(
    path: Path, *, label: str, project_root: Path
) -> dict[str, Any]:
    display_path = project_relative_display_path(path, project_root=project_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"無法讀取{label}: {display_path}; {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}根節點必須是object: {display_path}")
    return payload


def _source_contract(*, dataset_summary: dict[str, Any], target_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_policy": dataset_summary.get("policy"),
        "dataset_storage_schema_version": dataset_summary.get("dataset_storage_schema_version"),
        "source_data_inventory": dataset_summary.get("source_data_inventory"),
        "dataset_artifacts": dataset_summary.get("dataset_artifacts"),
        "target_schema_version": target_manifest.get("schema_version"),
        "target_contract": target_manifest.get("target_contract"),
        "target_artifacts": target_manifest.get("artifacts"),
    }


def _validate_frozen_checkpoint(*, checkpoint: dict[str, Any], bundle, profile_name: str, manifest: dict[str, Any]) -> None:
    expected_profile = bundle.profile.as_manifest_payload()
    if str(checkpoint.get("experiment_profile") or "") != str(profile_name):
        raise ValueError("Forward score-only rebuild checkpoint experiment_profile不一致")
    if checkpoint.get("experiment_settings") != expected_profile:
        raise ValueError("Forward score-only rebuild checkpoint experiment settings不一致")
    if str(checkpoint.get("training_objective") or "") != str(bundle.profile.training_objective):
        raise ValueError("Forward score-only rebuild checkpoint training objective不一致")
    if checkpoint.get("model_spec") != bundle.model_spec.as_manifest_payload():
        raise ValueError("Forward score-only rebuild checkpoint model spec不一致")
    if checkpoint.get("continuous_target_contract") != bundle.target_manifest.get("target_contract"):
        raise ValueError("Forward score-only rebuild checkpoint continuous target contract不一致")
    expected_shape = (
        int(bundle.feature_bank.shape[1]),
        int(bundle.feature_bank.shape[2]),
        int(bundle.group_context.shape[1]),
    )
    actual_shape = tuple(
        int(checkpoint.get(field)) if checkpoint.get(field) is not None else -1
        for field in ("sequence_length", "feature_count", "context_count")
    )
    if actual_shape != expected_shape:
        raise ValueError(
            "Forward score-only rebuild checkpoint input shape不一致: "
            f"expected={expected_shape}, actual={actual_shape}"
        )
    selected_epoch = int(checkpoint.get("selected_epoch") or manifest.get("selected_epoch") or 0)
    if selected_epoch < 1:
        raise ValueError("Forward score-only rebuild checkpoint缺少合法selected_epoch")


def rebuild_forward_oos_scores_from_frozen_checkpoint(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    seed: int,
) -> Path:
    """Rebuild Forward score rows without fitting or changing the frozen model."""

    root = Path(project_root).resolve()
    args = ranker_impl.parse_args(
        [
            "--filter-id", str(filter_id),
            "--model-architecture", str(model_architecture),
            "--experiment-profile", str(experiment_profile),
            "--seed", str(int(seed)),
        ]
    )
    bundle = load_continuous_ranker_data(
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
        preload_feature_bank=bool(args.preload_feature_bank),
        allow_stale_source=False,
        project_root=root,
    )
    artifacts = resolve_filter_artifact_paths(root, str(filter_id), str(model_architecture), str(experiment_profile))
    output_dir = resolve_filter_model_output_dir(root, str(filter_id), str(model_architecture), str(experiment_profile))
    report_path = (output_dir / CONTINUOUS_RANKER_REPORT_FILENAME).resolve()
    score_path = resolve_continuous_ranker_oos_score_path(
        root, str(filter_id), str(model_architecture), str(experiment_profile)
    )
    for label, path in (
        ("frozen model", artifacts.model_path),
        ("manifest", artifacts.manifest_path),
        ("report", report_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                "Forward score-only rebuild找不到"
                f"{label}: {project_relative_display_path(path, project_root=root)}"
            )

    manifest = _read_json(
        artifacts.manifest_path,
        label="continuous ranker manifest",
        project_root=root,
    )
    report = _read_json(
        report_path,
        label="continuous ranker report",
        project_root=root,
    )
    expected_identity = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
    }
    for field, expected in expected_identity.items():
        if str(manifest.get(field) or "") != expected or str(report.get(field) or "") != expected:
            raise ValueError(f"Forward score-only rebuild identity不一致: {field}")
    if build_file_manifest(artifacts.model_path) != manifest.get("model"):
        raise ValueError("Forward score-only rebuild frozen model雜湊與manifest不一致")
    manifest_source_contract = _source_contract(
        dataset_summary=dict(manifest.get("source_dataset") or {}),
        target_manifest=dict(manifest.get("source_continuous_target") or {}),
    )
    current_source_contract = _source_contract(
        dataset_summary=dict(bundle.summary or {}),
        target_manifest=dict(bundle.target_manifest or {}),
    )
    if manifest_source_contract != current_source_contract:
        raise ValueError("Forward score-only rebuild Dataset／Target source contract與frozen model不一致")

    torch, plan = resolve_ranker_execution_plan(args)
    checkpoint = torch.load(artifacts.model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Forward score-only rebuild checkpoint根節點必須是object")
    _validate_frozen_checkpoint(
        checkpoint=checkpoint,
        bundle=bundle,
        profile_name=str(experiment_profile),
        manifest=manifest,
    )
    model = build_model(
        int(bundle.feature_bank.shape[2]),
        int(bundle.group_context.shape[1]),
        model_spec=checkpoint["model_spec"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(plan.device)
    model.eval()

    inference_ids = resolve_forward_oos_score_group_ids(bundle)
    evaluable_ids = resolve_forward_oos_target_evaluable_group_ids(bundle)
    scores = predict_scores(
        torch,
        model,
        bundle,
        inference_ids,
        batch_size=int(args.evaluation_batch_size),
        plan=plan,
    )
    if len(scores) != len(inference_ids):
        raise ValueError("Forward score-only rebuild inference count不一致")
    training_scope = build_training_scope_mask(bundle)
    percentile_ids = evaluable_ids[training_scope[evaluable_ids]]
    percentile_target = build_percentile_target(bundle, percentile_ids)
    is_daily = (
        str(bundle.profile.training_sample_scope)
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )

    evaluable_inference_mask = np.isin(inference_ids, evaluable_ids)
    if is_daily:
        frame = bundle.group_table.iloc[inference_ids][["ticker", "date", "group_index"]].copy()
        frame["target_raw_r"] = np.nan
        frame["target_daily_percentile"] = np.nan
        frame.loc[evaluable_inference_mask, "target_raw_r"] = bundle.raw_target[
            inference_ids[evaluable_inference_mask]
        ]
        frame.loc[evaluable_inference_mask, "target_daily_percentile"] = percentile_target[
            inference_ids[evaluable_inference_mask]
        ]
        frame["model_score"] = scores
        frame.to_csv(score_path, index=False, encoding="utf-8-sig", compression="gzip")
        score_key = "oos_scores_gzip"
    else:
        frame = bundle.group_table.iloc[inference_ids][["ticker", "date", "group_index", "label"]].copy()
        frame["split"] = "oos"
        frame["selection_role"] = "not_applicable"
        frame["in_training_label_scope"] = training_scope[inference_ids]
        frame["target_raw_r"] = np.nan
        frame["target_daily_percentile"] = np.nan
        frame.loc[evaluable_inference_mask, "target_raw_r"] = bundle.raw_target[
            inference_ids[evaluable_inference_mask]
        ]
        frame.loc[evaluable_inference_mask, "target_daily_percentile"] = percentile_target[
            inference_ids[evaluable_inference_mask]
        ]
        frame["model_score"] = scores
        selection_frame = None
        if score_path.is_file():
            prior = read_breakout_quality_csv(score_path)
            if "split" in prior.columns:
                selection_frame = prior[prior["split"].astype(str) != "oos"].copy()
        if selection_frame is not None and not selection_frame.empty:
            frame = pd.concat([selection_frame, frame], ignore_index=True, sort=False)
        frame = frame.sort_values(["date", "ticker", "group_index"], kind="mergesort").reset_index(drop=True)
        frame.to_csv(score_path, index=False, encoding="utf-8-sig")
        score_key = "scores"

    eligibility = build_score_eligibility_contract(bundle.profile)
    coverage = {
        "inference_eligible_groups": int(len(inference_ids)),
        "target_evaluable_groups": int(len(evaluable_ids)),
        "future_target_required_for_score": False,
    }
    now = datetime.now(timezone.utc).isoformat()
    report["generated_at_utc"] = now
    report["score_eligibility_contract"] = eligibility
    report["forward_score_coverage"] = coverage
    report["score_only_rebuild"] = {
        "performed": True,
        "frozen_checkpoint_reused": True,
        "model_weights_changed": False,
        "reason": "future_target_independent_forward_score_eligibility",
    }
    report_artifacts = dict(report.get("artifacts") or {})
    report_artifacts[score_key] = build_file_manifest(score_path)
    report["artifacts"] = report_artifacts
    write_json(report_path, report)

    manifest["score_eligibility_contract"] = eligibility
    manifest["forward_score_coverage"] = coverage
    manifest["score_only_rebuild"] = dict(report["score_only_rebuild"])
    research_outputs = dict(manifest.get("research_outputs") or {})
    research_outputs[score_key] = build_file_manifest(score_path)
    research_outputs["report_json"] = build_file_manifest(report_path)
    manifest["research_outputs"] = research_outputs
    write_json(artifacts.manifest_path, manifest)
    return score_path


__all__ = ["rebuild_forward_oos_scores_from_frozen_checkpoint"]
