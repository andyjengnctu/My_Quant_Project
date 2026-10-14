"""Post-fit paired D40 evidence for canonical Model Compare / Test.

This is a Research-only read of validated OOS predictions and a mature canonical
future-path truth. It never updates the training target, selects an epoch, or
creates an inference/strategy runtime source. No model identity is hard-coded:
applicability comes from training/output capability and Safety supervision policy.
"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE,
    get_continuous_ranker_score_output_columns,
)
from core.breakout_quality_registry import get_continuous_ranker_research_spec
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.daily_ranker_data import load_official_breakout_candidate_keys
from filters.breakout_quality.daily_ranker_data import load_daily_universal_ranker_data
from filters.breakout_quality.continuous_target import (
    StrategyAlignedContinuousTargetSpec,
    full_horizon_mae_safety_r_from_minimum_low_return,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from services.breakout_quality.train_continuous_ranker import (
    HS_QUALIFICATION_P45_P55_BAND, daily_rank_metrics,
)

EXTENSION_ID = "full_horizon_mae_common_truth"
SHARED_BOUNDARY_EXTENSION_ID = "ao_safety_p45_p55_shared_cohort"


def _score_frame(item: Mapping[str, Any], *, evaluation_mode: str) -> pd.DataFrame | None:
    """Consume existing validated RUN/REUSE score artifact, never re-score a model."""
    settings = item["settings"]
    spec = get_continuous_ranker_research_spec(str(settings.experiment_profile))
    objective = str(settings.training_objective)
    score_column = get_continuous_ranker_score_output_columns(objective).get("raw_safety")
    if not score_column:
        return None
    contract = item.get("contract")
    if contract is None or not hasattr(contract, "score_path"):
        raise ValueError("D40 paired evidence缺少已驗證的score artifact contract")
    score_path = Path(contract.score_path)
    if evaluation_mode == "forward_oos":
        recorded = dict((contract.report.get("artifacts") or {}).get("oos_scores_gzip") or {})
        if not recorded or build_file_manifest(score_path) != recorded:
            raise ValueError("D40 Forward原始OOS score sidecar不存在或與validated report hash不一致")
    frame = pd.read_csv(score_path, encoding="utf-8-sig", dtype={"ticker": str})
    required = {"ticker", "date", "group_index", score_column}
    if evaluation_mode == "forward_oos":
        required.add("target_raw_r")
    else:
        required.update({"fold_id", "model_information_cutoff"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"D40 paired score缺少已宣告的欄位: {sorted(missing)}")
    if evaluation_mode == "forward_oos":
        # Forward sidecar also holds inference-only post-OOS rows. They have no
        # target_raw_r and MUST NEVER enter a hindsight-labelled comparison.
        frame = frame[pd.to_numeric(frame["target_raw_r"], errors="coerce").notna()].copy()
    if frame.empty:
        raise ValueError("D40 paired evidence找不到可評估的OOS分數")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="raise").astype("int64")
    frame["safety_prediction"] = pd.to_numeric(frame[score_column], errors="raise").astype("float64")
    if frame[["ticker", "date"]].duplicated().any() or frame["group_index"].duplicated().any():
        raise ValueError("D40 paired evidence同模型分數存在重複ticker/date或group_index")
    predictions = frame["safety_prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(predictions).all() or ((predictions < 0.0) | (predictions > 1.0)).any():
        raise ValueError("D40 paired evidence Raw Safety score必須是finite [0,1]")
    if evaluation_mode == "rolling_oos":
        cutoff = pd.to_datetime(frame["model_information_cutoff"], errors="raise").dt.normalize()
        if bool((frame["date"] <= cutoff).any()):
            raise ValueError("D40 paired evidence Rolling score未通過逐row information cutoff")
        if frame["fold_id"].isna().any():
            raise ValueError("D40 paired evidence Rolling fold_id缺失")
    return frame[["ticker", "date", "group_index", "safety_prediction"]].copy()


def _canonical_truth(bundle) -> pd.DataFrame:
    group = bundle.group_table
    required = {"ticker", "date", "group_index", "label_eval_end_date", "target_minimum_low_return", "target_adverse_r"}
    missing = required - set(group.columns)
    if missing:
        raise ValueError(f"D40 common truth缺少canonical future-path欄位: {sorted(missing)}")
    valid = np.asarray(bundle.target_valid, dtype=bool)
    if valid.ndim != 1 or len(valid) != len(group):
        raise ValueError("D40 common truth target-valid mask shape不一致")
    table = group.loc[valid, list(required)].copy()
    if table.empty:
        raise ValueError("D40 common truth沒有成熟標籤")
    table["ticker"] = table["ticker"].astype(str)
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.normalize()
    table["label_eval_end_date"] = pd.to_datetime(table["label_eval_end_date"], errors="raise").dt.normalize()
    if bool((table["label_eval_end_date"] < table["date"]).any()):
        raise ValueError("D40 common truth發現未來標籤結束日期早於signal date")
    min_return = pd.to_numeric(table["target_minimum_low_return"], errors="raise").to_numpy(dtype=np.float64)
    adverse = pd.to_numeric(table["target_adverse_r"], errors="raise").to_numpy(dtype=np.float64)
    risk_budget = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY).risk_budget_return
    safety = full_horizon_mae_safety_r_from_minimum_low_return(min_return, risk_budget_return=risk_budget)
    d40 = -safety
    if not np.isfinite(d40).all() or not np.isfinite(adverse).all():
        raise ValueError("D40 common truth有效標籤中有非有限值")
    if (d40 < 0).any() or (adverse < 0).any():
        raise ValueError("D40 common truth adverse定義不合法")
    if bool(np.any(d40 + 1e-5 < adverse)):
        raise ValueError("D40應不少於同窗口adverse-to-peak，價格或風險分母不一致")
    table["d40_r"] = d40
    table["old_safety_r"] = -adverse
    return table[["ticker", "date", "group_index", "d40_r", "old_safety_r"]]


def _daily_top_half_precision_and_top_decile_mae(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    successes = 0
    selected_count = 0
    top_decile_values: list[np.ndarray] = []
    for _, day in frame.groupby("date", sort=True):
        n = len(day)
        if n < 2:
            continue
        scored = day.sort_values(["safety_prediction", "ticker"], ascending=[False, True], kind="mergesort")
        p50 = max(1, math.ceil(n * 0.5))
        predicted = scored.head(p50)
        selected_count += len(predicted)
        successes += int(predicted["d40_true_hs"].sum())
        top_decile_values.append(scored.head(max(1, math.ceil(n * 0.1)))["d40_r"].to_numpy(dtype=np.float64))
    return (
        None if not selected_count else float(successes * 100.0 / selected_count),
        None if not top_decile_values else float(np.concatenate(top_decile_values).mean()),
    )


def _shared_ao_boundary_rows(
    common: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    split_label: str,
    breakout_candidate_keys: set[tuple[str, pd.Timestamp]],
) -> dict[str, list[dict[str, Any]]]:
    """Compare Raw Safety scores within exactly the same original-AO truth band.

    The AO adverse-to-peak percentile is constructed on the *full* mature
    daily universe, before intersecting model OOS scores and before filtering
    breakout membership. Neither D40 truth nor the model score selects rows.
    """
    low, high = HS_QUALIFICATION_P45_P55_BAND
    boundary = common["ao_safety_percentile"].between(low, high).to_numpy(dtype=bool)
    candidate_mask = np.fromiter(
        (
            (str(ticker), pd.Timestamp(date).normalize()) in breakout_candidate_keys
            for ticker, date in zip(common["ticker"], common["date"])
        ),
        dtype=bool, count=len(common),
    )
    by_model: dict[str, list[dict[str, Any]]] = {model_id: [] for model_id in model_ids}
    for scope, scope_mask in ((split_label, boundary), ("Breakout slice", boundary & candidate_mask)):
        cohort = common.loc[scope_mask].copy()
        ordered_keys = (
            f"{date:%Y-%m-%d}|{ticker}"
            for ticker, date in zip(cohort["ticker"], cohort["date"])
        )
        digest = hashlib.sha256("\n".join(ordered_keys).encode("utf-8")).hexdigest()
        dates = cohort["date"].to_numpy()
        truth = cohort["old_true_hs"].to_numpy(dtype=np.float32)
        for model_id in model_ids:
            scores = cohort[f"prediction_{model_id}"].to_numpy(dtype=np.float64)
            metrics = daily_rank_metrics(dates, scores, truth)
            by_model[model_id].append({
                "split": scope,
                "ao_boundary_stock_days": int(len(cohort)),
                "ao_boundary_dates": int(cohort["date"].nunique()),
                "rankable_dates": int(metrics["rankable_date_count"]),
                "comparable_pairs": int(metrics["comparable_pair_count"]),
                "ao_boundary_p45_p55_pair": metrics["pairwise_concordance"],
                "cohort_sha256": digest,
            })
    return by_model


def build_paired_full_horizon_mae_rows(
    *,
    truth: pd.DataFrame,
    predictions_by_model: Mapping[str, pd.DataFrame],
    split_label: str,
    breakout_candidate_keys: set[tuple[str, pd.Timestamp]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Paired intersection, common truth and canonical rank metrics across models."""
    if len(predictions_by_model) < 2:
        return {}, {}
    common = truth.copy()
    if common.duplicated(["ticker", "date"]).any() or common["group_index"].duplicated().any():
        raise ValueError("D40 common truth存在重複stock-day identity")
    common["date"] = pd.to_datetime(common["date"], errors="raise").dt.normalize()
    # Preserve original AO true-HS interpretation on the complete mature
    # day-universe. Cross-model score coverage cannot relabel the truth.
    full_mask = np.ones(len(common), dtype=bool)
    common["d40_true_hs"] = build_same_date_percentile_targets(
        -common["d40_r"].to_numpy(dtype=np.float64), full_mask, common["date"]
    ) >= 0.5
    common["ao_safety_percentile"] = build_same_date_percentile_targets(
        common["old_safety_r"].to_numpy(dtype=np.float64), full_mask, common["date"]
    )
    common["old_true_hs"] = common["ao_safety_percentile"] >= 0.5
    for model_id, raw_frame in predictions_by_model.items():
        score = raw_frame.copy()
        score["date"] = pd.to_datetime(score["date"], errors="raise").dt.normalize()
        if score.duplicated(["ticker", "date"]).any():
            raise ValueError(f"D40 {model_id}有重複ticker/date")
        if "group_index" in score.columns:
            score = score.rename(columns={"group_index": f"group_index_{model_id}"})
        common = common.merge(
            score.rename(columns={"safety_prediction": f"prediction_{model_id}"}),
            on=["ticker", "date"], how="inner", validate="one_to_one", sort=False,
        )
        group_id_column = f"group_index_{model_id}"
        if group_id_column in common:
            if bool((common[group_id_column] != common["group_index"]).any()):
                raise ValueError(f"D40 {model_id}與canonical truth group_index不一致")
            common.drop(columns=[group_id_column], inplace=True)
    if len(common) < 2:
        raise ValueError("D40 paired model comparison共同成熟stock-day不足")
    common.sort_values(["date", "ticker"], kind="mergesort", inplace=True)
    dates = common["date"].to_numpy()
    safety_truth = -common["d40_r"].to_numpy(dtype=np.float64)
    old_truth = common["old_safety_r"].to_numpy(dtype=np.float64)
    if not np.isfinite(safety_truth).all() or not np.isfinite(old_truth).all():
        raise ValueError("D40 paired truth有非有限值")
    p50_agreement = float(np.mean(common["d40_true_hs"] == common["old_true_hs"]) * 100.0)
    extra = (-safety_truth) - (-old_truth)
    if bool((extra < -1e-5).any()):
        raise ValueError("D40 paired truth與原adverse-to-peak風險比較不一致")
    all_dates = int(common["date"].nunique())
    sample = {
        "split": split_label,
        "paired_stock_days": int(len(common)),
        "paired_dates": all_dates,
        "ao_vs_d40_hs_p50_agreement_pct": p50_agreement,
        "d40_strictly_worse_than_peak_pct": float(np.mean(extra > 1e-5) * 100.0),
        "median_extra_adverse_r": float(np.median(np.maximum(extra, 0.0))),
        "truth_semantic": "-max(0,-minimum_future_low_return)/fixed_train_known_risk_budget; 40 mature bars",
        "cohort_semantic": "same-date P50 truth on full mature daily universe, evaluated on AO/CC common OOS ticker/date",
    }
    result: dict[str, dict[str, Any]] = {}
    for model_id in predictions_by_model:
        scored = common[["ticker", "date", "d40_r", "d40_true_hs", f"prediction_{model_id}"]].rename(
            columns={f"prediction_{model_id}": "safety_prediction"}
        )
        scores = scored["safety_prediction"].to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError(f"D40 {model_id}共同樣本包含non-finite score")
        rank = daily_rank_metrics(dates, scores, safety_truth)
        precision, top10 = _daily_top_half_precision_and_top_decile_mae(scored)
        result[model_id] = {
            "split": split_label,
            "paired_stock_days": sample["paired_stock_days"],
            "paired_dates": sample["paired_dates"],
            "d40_daily_rho": rank.get("mean_daily_spearman"),
            "d40_pair": rank.get("pairwise_concordance"),
            "top10_mean_d40_r": top10,
            "pred_p50_true_low_mae_precision_pct": precision,
            "ao_vs_d40_hs_p50_agreement_pct": p50_agreement,
            "d40_strictly_worse_than_peak_pct": sample["d40_strictly_worse_than_peak_pct"],
            "median_extra_adverse_r": sample["median_extra_adverse_r"],
        }
    if breakout_candidate_keys is not None:
        # Private transient handoff: the permanent D40 Extension sample payload
        # remains byte/schema compatible with all earlier model reports.
        sample["_shared_ao_boundary_rows"] = _shared_ao_boundary_rows(
            common, model_ids=tuple(predictions_by_model),
            split_label=split_label,
            breakout_candidate_keys=breakout_candidate_keys,
        )
    return result, sample


def attach_full_horizon_mae_common_truth_extension(
    models: Sequence[Mapping[str, Any]],
    *,
    evaluation_mode: str,
    project_root: Path,
) -> list[dict[str, Any]]:
    """Add ephemeral comparison evidence to report payloads, never source artifacts.

    The source models keep identical fitting identities and their original
    validated reports. Both RUN and REUSE call this same post-fit composition.
    """
    copied = [{**item, "payload": dict(item.get("payload") or {})} for item in models]
    if evaluation_mode not in {"forward_oos", "rolling_oos"}:
        return copied
    participants: dict[str, pd.DataFrame] = {}
    # A1 is a single-variable control: identical training objective / score-head
    # recipe. Do not silently shrink AO/CC's paired sample to AK/BZ or unrelated
    # auxiliary-loss models just because they also persist a Raw Safety head.
    full_horizon_model = next(
        (
            item for item in copied
            if str(get_continuous_ranker_research_spec(
                str(item["settings"].experiment_profile)
            ).safety_supervision_source) == CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE
        ),
        None,
    )
    if full_horizon_model is None:
        return copied
    control_objective = str(full_horizon_model["settings"].training_objective)
    for item in copied:
        settings = item["settings"]
        if str(settings.training_objective) != control_objective:
            continue
        if "raw_safety" not in get_continuous_ranker_score_output_columns(control_objective):
            continue
        frame = _score_frame(item, evaluation_mode=evaluation_mode)
        if frame is not None:
            participants[str(item["model_id"])] = frame
    if full_horizon_model is None or len(participants) < 2:
        return copied
    reference = full_horizon_model["settings"]
    comparable_target_id = str(full_horizon_model["contract"].continuous_target_id)
    for item in copied:
        if str(item["model_id"]) in participants:
            if str(item["contract"].continuous_target_id) != comparable_target_id:
                raise ValueError("D40 paired model comparison混入不同continuous target id")
            if str(item["settings"].filter_id) != str(reference.filter_id):
                raise ValueError("D40 paired model comparison混入不同filter id")
    # The AO/CC full-horizon target is embedded in the canonical daily stock-day
    # bundle.  Event-group Continuous Target manifests are neither required nor
    # produced for this sample scope; loading the event-group bundle here would
    # wrongly require an unrelated physical target artifact after OOS reports exist.
    bundle = load_daily_universal_ranker_data(
        filter_id=str(reference.filter_id),
        model_architecture=str(reference.model_architecture),
        experiment_profile=str(reference.experiment_profile),
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=project_root,
    )
    truth = _canonical_truth(bundle)
    label = "Rolling OOS" if evaluation_mode == "rolling_oos" else "Forward OOS"
    # Same official breakout event membership as the per-model SOP, filtering
    # the daily-universal cohort only; never re-rank Safety within the slice.
    candidate_keys = load_official_breakout_candidate_keys(
        str(reference.filter_id), allow_stale_source=False
    )
    rows, sample = build_paired_full_horizon_mae_rows(
        truth=truth, predictions_by_model=participants, split_label=label,
        breakout_candidate_keys=candidate_keys,
    )
    boundary_rows = sample.pop("_shared_ao_boundary_rows")
    for item in copied:
        model_id = str(item["model_id"])
        if model_id not in rows:
            continue
        payload = item["payload"]
        persisted = [
            dict(ext) for ext in list(payload.get("comparison_extensions") or [])
            if str(dict(ext).get("id") or "") not in {EXTENSION_ID, SHARED_BOUNDARY_EXTENSION_ID}
        ]
        persisted.append({"id": EXTENSION_ID, "rows": [rows[model_id]], "paired_truth": sample})
        persisted.append({
            "id": SHARED_BOUNDARY_EXTENSION_ID,
            "rows": boundary_rows[model_id],
            "cohort_semantic": "AO original adverse-to-peak P45-P55 from full mature daily universe; "
                "same joined OOS ticker/date and original AO True-HS; breakout membership filter only",
        })
        payload["comparison_extensions"] = persisted
    return copied


__all__ = [
    "EXTENSION_ID", "SHARED_BOUNDARY_EXTENSION_ID", "build_paired_full_horizon_mae_rows",
    "attach_full_horizon_mae_common_truth_extension",
]
