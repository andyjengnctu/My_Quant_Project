"""Read-only MR joint-quadrant proper scores from validated canonical OOS artifacts.

No model is re-fitted, re-selected or re-scored; no OOS observation feeds fitting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import get_continuous_ranker_training_policy
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.daily_ranker_data import (
    load_daily_universal_ranker_data, load_official_breakout_candidate_keys,
)
from filters.breakout_quality.joint_mfe_d40 import (
    build_joint_mfe_d40_quadrants, JOINT_FAVORABLE_AND_LOW_MAE_CLASS,
)

from services.breakout_quality.joint_event_incremental_evidence import (
    build_joint_event_incremental_extensions,
)

EXTENSION_ID = "joint_mfe_d40_quadrant_proper_scores"


def _proper_scores(frame: pd.DataFrame, *, split: str, columns: tuple[str, ...]) -> dict[str, Any]:
    truth = frame["joint_truth"].to_numpy(dtype=np.int64)
    probs = frame[list(columns)].to_numpy(dtype=np.float64)
    if not len(truth):
        raise ValueError("joint quadrant OOS proper-scoring cohort is empty")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("joint quadrant OOS probabilities must be finite [0,1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=2e-5, rtol=0):
        raise ValueError("joint quadrant OOS probabilities must sum to 1")
    if (truth < 0).any() or (truth >= len(columns)).any():
        raise ValueError("joint quadrant OOS truth must use registered categories")
    p_joint = probs[:, JOINT_FAVORABLE_AND_LOW_MAE_CLASS]
    joint_true = (truth == JOINT_FAVORABLE_AND_LOW_MAE_CLASS).astype(np.float64)
    independent = (probs[:, 2] + probs[:, 3]) * (probs[:, 1] + probs[:, 3])
    eps = np.finfo(np.float64).tiny
    ce = -np.log(np.clip(probs[np.arange(len(truth)), truth], eps, 1.0)).mean()
    top_hit = []
    for _date, day in frame.groupby("date", sort=True):
        if len(day) < 10:
            continue
        ranked = day.sort_values([columns[3], "ticker"], ascending=[False, True], kind="mergesort")
        n = max(1, int(np.ceil(len(day) * 0.1)))
        top_hit.extend((ranked.head(n)["joint_truth"] == JOINT_FAVORABLE_AND_LOW_MAE_CLASS).tolist())
    return {
        "split": split,
        "stock_days": int(len(truth)),
        "dates": int(frame["date"].nunique()),
        "multiclass_logloss": float(ce),
        "joint_brier": float(np.mean((p_joint - joint_true) ** 2)),
        "independent_brier": float(np.mean((independent - joint_true) ** 2)),
        "joint_base_rate_pct": float(100.0 * joint_true.mean()),
        "joint_predicted_pct": float(100.0 * p_joint.mean()),
        "top10_joint_hit_pct": None if not top_hit else float(100.0 * np.mean(top_hit)),
    }


def _validated_rows(item: Mapping[str, Any], *, mode: str, project_root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame, Any, set]:
    settings = item["settings"]
    policy = get_continuous_ranker_training_policy(str(settings.training_objective))
    if policy.categorical_class_count is None:
        return []
    if int(policy.categorical_class_count) != 4 or int(policy.categorical_primary_class) != 3:
        raise ValueError("unsupported categorical joint event contract")
    columns = tuple(policy.categorical_score_columns)
    contract = item.get("contract")
    if contract is None or not hasattr(contract, "score_path"):
        raise ValueError("joint quadrant OOS validated score contract missing")
    score_path = Path(contract.score_path)
    if mode == "forward_oos":
        recorded = dict((contract.report.get("artifacts") or {}).get("oos_scores_gzip") or {})
        if not recorded or build_file_manifest(score_path) != recorded:
            raise ValueError("joint quadrant Forward OOS score SHA/manifest mismatch")
    score = pd.read_csv(score_path, encoding="utf-8-sig", dtype={"ticker": str})
    primary_score_column = "model_score" if mode == "forward_oos" else "breakout_quality_score"
    required = {"ticker", "date", "group_index", primary_score_column, *columns}
    if mode == "forward_oos":
        required.add("target_raw_r")
    else:
        required.update(("fold_id", "model_information_cutoff"))
    missing = sorted(required - set(score))
    if missing:
        raise ValueError(f"joint quadrant OOS score missing declared fields: {missing}")
    if mode == "forward_oos":
        score = score[pd.to_numeric(score["target_raw_r"], errors="coerce").notna()].copy()
    score["date"] = pd.to_datetime(score["date"], errors="raise").dt.normalize()
    score["ticker"] = score["ticker"].astype(str)
    score["group_index"] = pd.to_numeric(score["group_index"], errors="raise").astype("int64")
    if score.duplicated(["ticker", "date"]).any() or score["group_index"].duplicated().any():
        raise ValueError("joint quadrant OOS duplicated ticker/date or group_index")
    if mode == "rolling_oos":
        cutoff = pd.to_datetime(score["model_information_cutoff"], errors="raise").dt.normalize()
        if (score["date"] <= cutoff).any() or score["fold_id"].isna().any():
            raise ValueError("joint quadrant Rolling score violates information cutoff/fold")
    if not np.allclose(
        pd.to_numeric(score[primary_score_column], errors="raise").to_numpy(dtype=np.float64),
        pd.to_numeric(score[columns[3]], errors="raise").to_numpy(dtype=np.float64),
        atol=1e-6, rtol=0,
    ):
        raise ValueError("joint quadrant OOS primary score is not class-3 probability")
    bundle = load_daily_universal_ranker_data(
        filter_id=str(settings.filter_id),
        model_architecture=str(settings.model_architecture),
        experiment_profile=str(settings.experiment_profile),
        preload_feature_bank=False, allow_stale_source=False, project_root=project_root,
    )
    truth = build_joint_mfe_d40_quadrants(bundle.group_table, np.asarray(bundle.target_valid, dtype=bool))
    canonical = bundle.group_table[["ticker", "date", "group_index"]].copy()
    canonical["joint_truth"] = truth
    canonical = canonical[np.isfinite(canonical["joint_truth"])].copy()
    canonical["ticker"] = canonical["ticker"].astype(str)
    canonical["date"] = pd.to_datetime(canonical["date"], errors="raise").dt.normalize()
    canonical["group_index"] = pd.to_numeric(canonical["group_index"], errors="raise").astype("int64")
    score = score.merge(canonical, how="inner", on=["ticker", "date", "group_index"], validate="one_to_one")
    if score.empty:
        raise ValueError("joint quadrant validated OOS scores have no mature canonical 40D truth")
    for column in columns:
        score[column] = pd.to_numeric(score[column], errors="raise")
    label = "Rolling OOS" if mode == "rolling_oos" else "Forward OOS"
    rows = [_proper_scores(score, split=label, columns=columns)]
    membership = load_official_breakout_candidate_keys(
        str(settings.filter_id), allow_stale_source=False
    )
    eligible = pd.Series(list(zip(score["ticker"], score["date"]))).isin(membership)
    breakout = score[eligible.to_numpy(dtype=bool)]
    if not breakout.empty:
        rows.append(_proper_scores(breakout, split="Breakout slice", columns=columns))
    return rows, score, bundle, membership, truth


def attach_joint_mfe_d40_extension(
    models: Sequence[Mapping[str, Any]], *, evaluation_mode: str, project_root: Path,
) -> list[dict[str, Any]]:
    """Model-specific post-fit evidence; safe in both RUN and REUSE comparisons."""
    output = [{**item, "payload": dict(item.get("payload") or {})} for item in models]
    if evaluation_mode not in {"forward_oos", "rolling_oos"}:
        return output
    for item in output:
        policy = get_continuous_ranker_training_policy(str(item["settings"].training_objective))
        if policy.categorical_class_count is None:
            continue
        rows, scored, bundle, membership, truth = _validated_rows(
            item, mode=evaluation_mode, project_root=Path(project_root),
        )
        if rows:
            old = [dict(ext) for ext in (item["payload"].get("comparison_extensions") or [])
                   if str(dict(ext).get("id")) != EXTENSION_ID]
            item["payload"]["comparison_extensions"] = [*old, {"id": EXTENSION_ID, "rows": rows}]
            incremental = build_joint_event_incremental_extensions(
                models=output, categorical_item=item, scored=scored, bundle=bundle,
                candidate_keys=membership, mode=evaluation_mode, full_truth=truth,
                categorical_columns=tuple(policy.categorical_score_columns),
            )
            for model in output:
                evidence = incremental.get(str(model["model_id"]))
                if evidence is None:
                    continue
                extensions = [dict(ext) for ext in model["payload"].get("comparison_extensions", [])
                              if str(ext.get("id")) != evidence["id"]]
                model["payload"]["comparison_extensions"] = [*extensions, evidence]
    return output
