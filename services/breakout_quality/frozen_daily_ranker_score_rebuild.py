"""Deterministically reconstruct daily-universal frozen ranker scores for explicit keys.

This is an upstream model-domain producer.  It never fits model parameters or changes
scientific identity; it loads an existing frozen checkpoint and evaluates only the
requested canonical daily stock-day rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    get_breakout_quality_experiment_profile,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.daily_ranker_data import load_daily_universal_ranker_data
from filters.breakout_quality.model import build_model, require_torch
from filters.breakout_quality.paths import resolve_filter_artifact_paths
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
from services.breakout_quality import ranker_training as ranker_api


def rebuild_frozen_daily_ranker_scores_for_keys(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    key_frame: pd.DataFrame,
    output_path: str | Path,
    expected_target_raw_r: np.ndarray | None = None,
    target_tolerance_r: float = 2e-6,
) -> dict[str, Any]:
    """Evaluate an existing frozen checkpoint on exact requested daily keys."""

    root = Path(project_root).resolve()
    required = ["ticker", "date", "group_index"]
    missing = [name for name in required if name not in key_frame.columns]
    if missing:
        raise ValueError(f"Frozen daily score rebuild缺少key欄位: {missing}")
    keys = key_frame[required].copy()
    keys["ticker"] = keys["ticker"].astype(str).str.strip()
    keys["date"] = pd.to_datetime(keys["date"], errors="raise").dt.strftime("%Y-%m-%d")
    keys["group_index"] = pd.to_numeric(keys["group_index"], errors="raise").astype(np.int64)
    if keys.duplicated(required).any():
        raise ValueError("Frozen daily score rebuild requested keys不唯一")

    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    artifacts = resolve_filter_artifact_paths(
        root, str(filter_id), str(model_architecture), str(experiment_profile)
    )
    if not artifacts.model_path.is_file() or not artifacts.manifest_path.is_file():
        raise FileNotFoundError(
            "缺少frozen model/checkpoint identity，不能以Audit重訓取代: "
            f"{artifacts.model_path} / {artifacts.manifest_path}"
        )

    bundle = load_daily_universal_ranker_data(
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
        preload_feature_bank=bool(BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK),
        allow_stale_source=True,
        project_root=root,
    )
    universe = bundle.group_table[["ticker", "date", "group_index"]].copy()
    universe["ticker"] = universe["ticker"].astype(str).str.strip()
    universe["date"] = pd.to_datetime(universe["date"], errors="raise").dt.strftime("%Y-%m-%d")
    universe["group_index"] = pd.to_numeric(universe["group_index"], errors="raise").astype(np.int64)
    universe["__bundle_pos"] = np.arange(len(universe), dtype=np.int64)
    mapped = keys.merge(universe, on=required, how="left", validate="one_to_one")
    if mapped["__bundle_pos"].isna().any():
        raise ValueError("Frozen daily score rebuild requested keys與canonical daily universe不一致")
    group_ids = mapped["__bundle_pos"].to_numpy(dtype=np.int64)
    target_raw = np.asarray(bundle.raw_target[group_ids], dtype=np.float64)
    if expected_target_raw_r is not None:
        expected = np.asarray(expected_target_raw_r, dtype=np.float64)
        if expected.shape != target_raw.shape or not np.allclose(
            target_raw, expected, rtol=0.0, atol=float(target_tolerance_r), equal_nan=False
        ):
            max_delta = float(np.nanmax(np.abs(target_raw - expected))) if len(target_raw) else 0.0
            raise ValueError(
                "Frozen daily score rebuild canonical target與requested reference truth不一致: "
                f"max_abs_delta={max_delta:.8g}R"
            )

    torch, _nn = require_torch()
    checkpoint = torch.load(artifacts.model_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Frozen daily ranker checkpoint根節點必須是object")
    if str(checkpoint.get("experiment_profile") or "") != str(experiment_profile):
        raise ValueError("Frozen daily ranker checkpoint experiment_profile不一致")
    if str(checkpoint.get("training_objective") or "") != str(profile.training_objective):
        raise ValueError("Frozen daily ranker checkpoint training_objective不一致")
    if checkpoint.get("experiment_settings") != profile.as_manifest_payload():
        raise ValueError("Frozen daily ranker checkpoint experiment_settings不一致")
    expected_spec = bundle.model_spec.as_manifest_payload()
    if checkpoint.get("model_spec") != expected_spec:
        raise ValueError("Frozen daily ranker checkpoint model_spec與canonical daily provider不一致")

    plan = resolve_torch_execution_plan(
        torch,
        requested_device=str(BREAKOUT_QUALITY_TORCH_DEVICE),
        mixed_precision=bool(BREAKOUT_QUALITY_USE_MIXED_PRECISION),
        mixed_precision_dtype=str(BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE),
        deterministic_algorithms=bool(BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS),
        allow_tf32=bool(BREAKOUT_QUALITY_ALLOW_TF32),
    )
    model = build_model(
        int(checkpoint["feature_count"]),
        int(checkpoint["context_count"]),
        architecture=str(model_architecture),
        model_spec=expected_spec,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(plan.device)
    model.eval()
    scores = ranker_api.predict_scores(
        torch,
        model,
        bundle.feature_bank,
        bundle.group_context,
        group_ids,
        batch_size=int(BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE),
        plan=plan,
        training_objective=str(profile.training_objective),
    )
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) != len(keys) or not np.isfinite(scores).all():
        raise ValueError("Frozen daily score rebuild inference輸出長度或finite contract失敗")

    result = keys.copy()
    result["target_raw_r"] = target_raw.astype(np.float32)
    result["model_score"] = scores
    path = Path(output_path)
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
    return {
        "score_path": path,
        "score_artifact": build_file_manifest(path),
        "model_artifact": build_file_manifest(artifacts.model_path),
        "manifest_artifact": build_file_manifest(artifacts.manifest_path),
        "row_count": int(len(result)),
        "torch_execution": plan.as_manifest_payload(),
    }


__all__ = ["rebuild_frozen_daily_ranker_scores_for_keys"]
