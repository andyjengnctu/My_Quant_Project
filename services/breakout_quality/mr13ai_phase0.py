"""MR-13AI Phase-0 read-only local-conflict boundary diagnostic.

This is a disposable research diagnostic.  It never trains, refits, or writes model
artifacts.  The selection rule uses only frozen MR-13AF Forward-OOS score ranks and
the PIT-safe predicted-Safety context available on the same date.  Realized MFE and
Safety truth are attached only after the ranking decision for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K,
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    get_continuous_ranker_research_spec,
)
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.continuous_ranker_quality import daily_top_k_metrics
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.daily_ranker_data import load_official_breakout_candidate_keys
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.ranking_score_store import load_continuous_ranker_oos_contract
from filters.breakout_quality.workflow_io import PROJECT_ROOT
from services.breakout_quality import ranker_training as ranker_api

SCHEMA_VERSION = 1
DIAGNOSTIC_ID = "MR-13AI-PHASE0"
OUTPUT_DIRNAME = "mr13ai_phase0_local_conflict"
REPORT_JSON_FILENAME = "mr13ai_phase0_local_conflict.json"
REPORT_MARKDOWN_FILENAME = "mr13ai_phase0_local_conflict.md"
SOURCE_PROFILE = DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE

PHASE0_CONTRACT = {
    "research_id": "MR-13AI",
    "phase": "phase0_zero_retrain",
    "source_model": "MR-13AF",
    "source_profile": SOURCE_PROFILE,
    "decision_scope": "single_k_boundary_pair_per_date",
    "mfe_confidence_proxy": "same_date_frozen_af_model_score_gap_in_0_1",
    "safety_input": "pit_safe_same_date_predicted_safety_percentile",
    "preference_formula": "g = d_score + (1 - abs(d_score)) * d_safety",
    "swap_rule": "swap_rank_k_with_rank_k_plus_1_iff_g_lt_0",
    "tie_rule": "g_eq_0_keeps_af_order",
    "uses_future_target_for_selection": False,
    "realized_truth_role": "evaluation_only_after_selection",
    "training_performed": False,
    "model_artifact_written": False,
}

_REQUIRED_TRUTH_COLUMNS = (
    "target_favorable_r",
    "target_adverse_r",
    "target_mfe_daily_percentile",
    "target_low_adverse_daily_percentile",
    "predicted_safety_percentile",
)


@dataclass(frozen=True)
class BoundaryDecision:
    score_gap: float
    safety_gap: float
    preference_margin: float
    swap: bool


def boundary_preference_margin(score_gap: float, safety_gap: float) -> float:
    """Return the parameter-free local-conflict margin used only at the K boundary."""

    d_score = float(score_gap)
    d_safety = float(safety_gap)
    if not math.isfinite(d_score) or not math.isfinite(d_safety):
        raise ValueError("MR-13AI boundary gap必須finite")
    if d_score < -1e-12 or d_score > 1.0 + 1e-12:
        raise ValueError("MR-13AI AF model-score gap必須位於[0,1]")
    if d_safety < -1.0 - 1e-12 or d_safety > 1.0 + 1e-12:
        raise ValueError("MR-13AI Safety percentile gap必須位於[-1,1]")
    d_score = min(1.0, max(0.0, d_score))
    d_safety = min(1.0, max(-1.0, d_safety))
    return float(d_score + (1.0 - abs(d_score)) * d_safety)


def boundary_decision(
    *,
    inside_score: float,
    outside_score: float,
    inside_safety_percentile: float,
    outside_safety_percentile: float,
) -> BoundaryDecision:
    """Compare only AF rank K versus K+1 using prediction-time information."""

    score_gap = float(inside_score) - float(outside_score)
    safety_gap = float(inside_safety_percentile) - float(outside_safety_percentile)
    margin = boundary_preference_margin(score_gap, safety_gap)
    return BoundaryDecision(
        score_gap=float(score_gap),
        safety_gap=float(safety_gap),
        preference_margin=float(margin),
        swap=bool(margin < 0.0),
    )


def apply_boundary_reorder(frame: pd.DataFrame, *, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one deterministic K/K+1 exchange per date at most.

    The returned ranking scores are ordinal and exist only to evaluate the resulting
    ordering.  No realized target column participates in the boundary decision.
    """

    k = int(top_k)
    if k < 1:
        raise ValueError("MR-13AI top_k必須>=1")
    required = {"ticker", "date", "group_index", "model_score", "predicted_safety_percentile"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MR-13AI輸入缺少欄位: {missing}")
    work = frame.copy().reset_index(drop=True)
    work["ticker"] = work["ticker"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    work["group_index"] = pd.to_numeric(work["group_index"], errors="raise").astype(int)
    work["model_score"] = pd.to_numeric(work["model_score"], errors="raise").astype(float)
    work["predicted_safety_percentile"] = pd.to_numeric(
        work["predicted_safety_percentile"], errors="raise"
    ).astype(float)
    safety = work["predicted_safety_percentile"].to_numpy(dtype=np.float64)
    if not np.isfinite(safety).all() or bool(((safety < 0.0) | (safety > 1.0)).any()):
        raise ValueError("MR-13AI predicted Safety percentile必須finite且位於[0,1]")
    if bool(work.duplicated(["ticker", "date", "group_index"]).any()):
        raise ValueError("MR-13AI輸入row identity重複")

    work["af_rank"] = np.nan
    work["ai_rank"] = np.nan
    swap_rows: list[dict[str, Any]] = []

    for date, day in work.groupby("date", sort=True):
        ordered = day.sort_values(
            ["model_score", "ticker", "group_index"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        indices = ordered.index.to_numpy(dtype=np.int64)
        count = int(len(indices))
        ranks = np.arange(1, count + 1, dtype=np.int64)
        work.loc[indices, "af_rank"] = ranks
        ai_indices = indices.copy()
        decision: BoundaryDecision | None = None
        inside_index: int | None = None
        outside_index: int | None = None
        if count > k:
            inside_index = int(indices[k - 1])
            outside_index = int(indices[k])
            decision = boundary_decision(
                inside_score=float(work.at[inside_index, "model_score"]),
                outside_score=float(work.at[outside_index, "model_score"]),
                inside_safety_percentile=float(work.at[inside_index, "predicted_safety_percentile"]),
                outside_safety_percentile=float(work.at[outside_index, "predicted_safety_percentile"]),
            )
            if decision.swap:
                ai_indices[k - 1], ai_indices[k] = ai_indices[k], ai_indices[k - 1]
        work.loc[ai_indices, "ai_rank"] = ranks

        if decision is not None and inside_index is not None and outside_index is not None:
            inside = work.loc[inside_index]
            outside = work.loc[outside_index]
            row = {
                "date": pd.Timestamp(date),
                "candidate_count": count,
                "inside_ticker": str(inside["ticker"]),
                "outside_ticker": str(outside["ticker"]),
                "inside_group_index": int(inside["group_index"]),
                "outside_group_index": int(outside["group_index"]),
                "inside_af_score": float(inside["model_score"]),
                "outside_af_score": float(outside["model_score"]),
                                "inside_predicted_safety_percentile": float(inside["predicted_safety_percentile"]),
                "outside_predicted_safety_percentile": float(outside["predicted_safety_percentile"]),
                "score_gap": float(decision.score_gap),
                "safety_gap": float(decision.safety_gap),
                "preference_margin": float(decision.preference_margin),
                "swap": bool(decision.swap),
            }
            for column in _REQUIRED_TRUTH_COLUMNS[:-1]:
                if column in work.columns:
                    row[f"inside_{column}"] = float(inside[column])
                    row[f"outside_{column}"] = float(outside[column])
            swap_rows.append(row)

    if bool(work[["af_rank", "ai_rank"]].isna().any().any()):
        raise ValueError("MR-13AI無法建立完整rank")
    work["af_rank"] = work["af_rank"].astype(int)
    work["ai_rank"] = work["ai_rank"].astype(int)
    # Higher score means better rank.  Daily Spearman/top-K consumers only require
    # a monotone ranking score, not the original model-score magnitude.
    day_size = work.groupby("date", sort=False)["ticker"].transform("size").astype(float)
    work["af_order_score"] = day_size - work["af_rank"].astype(float) + 1.0
    work["ai_order_score"] = day_size - work["ai_rank"].astype(float) + 1.0
    return work, pd.DataFrame(swap_rows)


def _selected_metrics(frame: pd.DataFrame, *, score_column: str, top_k: int) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    competition_dates = 0
    for _date, day in frame.groupby("date", sort=True):
        if len(day) <= int(top_k):
            continue
        competition_dates += 1
        ordered = day.sort_values(
            [score_column, "ticker", "group_index"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        rows.append(ordered.iloc[: int(top_k)].copy())
    if not rows:
        return {"competition_date_count": 0, "selected_row_count": 0}
    selected = pd.concat(rows, ignore_index=True)
    mfe = selected["target_favorable_r"].to_numpy(dtype=np.float64)
    adverse = selected["target_adverse_r"].to_numpy(dtype=np.float64)
    mfe_pct = selected["target_mfe_daily_percentile"].to_numpy(dtype=np.float64)
    safety_pct = selected["target_low_adverse_daily_percentile"].to_numpy(dtype=np.float64)
    hm = mfe_pct >= 0.50
    hs = safety_pct >= 0.50
    return {
        "competition_date_count": int(competition_dates),
        "selected_row_count": int(len(selected)),
        "top_k_mfe_r_mean": float(np.mean(mfe)),
        "top_k_adverse_r_mean": float(np.mean(adverse)),
        "top_k_high_mfe_pct": float(np.mean(hm) * 100.0),
        "top_k_high_safety_pct": float(np.mean(hs) * 100.0),
        "top_k_hmhs_pct": float(np.mean(hm & hs) * 100.0),
    }


def _ranking_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_k: int,
    boundary_width: int,
) -> dict[str, Any]:
    dates = frame["date"].to_numpy()
    scores = frame[score_column].to_numpy(dtype=np.float64)
    mfe_pct = frame["target_mfe_daily_percentile"].to_numpy(dtype=np.float64)
    safety_pct = frame["target_low_adverse_daily_percentile"].to_numpy(dtype=np.float64)
    favorable = frame["target_favorable_r"].to_numpy(dtype=np.float64)
    adverse = frame["target_adverse_r"].to_numpy(dtype=np.float64)
    predicted_safety = frame["predicted_safety_percentile"].to_numpy(dtype=np.float64)
    mfe_rank = ranker_api.daily_rank_metrics(dates, scores, mfe_pct)
    actual_safety_rank = ranker_api.daily_rank_metrics(dates, scores, -adverse)
    predicted_safety_rank = ranker_api.daily_rank_metrics(dates, predicted_safety, scores)
    top_k_quality = daily_top_k_metrics(
        dates,
        scores,
        favorable,
        mfe_pct,
        top_k=int(top_k),
        boundary_width=int(boundary_width),
    )
    selected = _selected_metrics(frame, score_column=score_column, top_k=int(top_k))
    return {
        "group_count": int(len(frame)),
        "mfe_mean_daily_spearman": mfe_rank.get("mean_daily_spearman"),
        "mfe_pairwise_concordance": mfe_rank.get("pairwise_concordance"),
        "score_to_low_adverse_mean_daily_spearman": actual_safety_rank.get("mean_daily_spearman"),
        "predicted_safety_to_score_mean_daily_spearman": predicted_safety_rank.get("mean_daily_spearman"),
        "top_k_quality": top_k_quality,
        "selected": selected,
    }


def _swap_metrics(swap_rows: pd.DataFrame) -> dict[str, Any]:
    if swap_rows.empty:
        return {
            "boundary_date_count": 0,
            "swap_date_count": 0,
            "swap_rate_pct": None,
        }
    swapped = swap_rows.loc[swap_rows["swap"].astype(bool)].copy()
    result: dict[str, Any] = {
        "boundary_date_count": int(len(swap_rows)),
        "swap_date_count": int(len(swapped)),
        "swap_rate_pct": float(len(swapped) / len(swap_rows) * 100.0),
        "mean_score_gap_all_boundary_dates": float(swap_rows["score_gap"].mean()),
        "mean_abs_safety_gap_all_boundary_dates": float(swap_rows["safety_gap"].abs().mean()),
    }
    if swapped.empty:
        return result

    def delta(column: str) -> np.ndarray:
        return (
            swapped[f"outside_{column}"].to_numpy(dtype=np.float64)
            - swapped[f"inside_{column}"].to_numpy(dtype=np.float64)
        )

    mfe_delta = delta("target_favorable_r")
    adverse_delta = delta("target_adverse_r")
    hm_inside = swapped["inside_target_mfe_daily_percentile"].to_numpy(dtype=np.float64) >= 0.50
    hm_outside = swapped["outside_target_mfe_daily_percentile"].to_numpy(dtype=np.float64) >= 0.50
    hs_inside = swapped["inside_target_low_adverse_daily_percentile"].to_numpy(dtype=np.float64) >= 0.50
    hs_outside = swapped["outside_target_low_adverse_daily_percentile"].to_numpy(dtype=np.float64) >= 0.50
    result.update(
        {
            "mean_realized_mfe_delta_r_ai_minus_af": float(np.mean(mfe_delta)),
            "mean_realized_adverse_delta_r_ai_minus_af": float(np.mean(adverse_delta)),
            "actual_safety_improved_swap_pct": float(np.mean(adverse_delta < 0.0) * 100.0),
            "mfe_retained_or_improved_swap_pct": float(np.mean(mfe_delta >= 0.0) * 100.0),
            "high_mfe_delta_pp": float((np.mean(hm_outside) - np.mean(hm_inside)) * 100.0),
            "high_safety_delta_pp": float((np.mean(hs_outside) - np.mean(hs_inside)) * 100.0),
            "hmhs_delta_pp": float(
                (np.mean(hm_outside & hs_outside) - np.mean(hm_inside & hs_inside)) * 100.0
            ),
            "hmhs_conversion_swap_pct": float(
                np.mean((hm_outside & hs_outside) & ~(hm_inside & hs_inside)) * 100.0
            ),
        }
    )
    return result


def evaluate_phase0_frame(
    frame: pd.DataFrame,
    *,
    top_k: int,
    boundary_width: int,
) -> dict[str, Any]:
    required = {
        "ticker",
        "date",
        "group_index",
        "model_score",
        *_REQUIRED_TRUTH_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MR-13AI evaluation frame缺少欄位: {missing}")
    evaluable = frame.copy()
    numeric_columns = [
        "model_score",
        "predicted_safety_percentile",
        "target_favorable_r",
        "target_adverse_r",
        "target_mfe_daily_percentile",
        "target_low_adverse_daily_percentile",
    ]
    for column in numeric_columns:
        evaluable[column] = pd.to_numeric(evaluable[column], errors="coerce")
    finite = np.ones(len(evaluable), dtype=bool)
    for column in numeric_columns:
        finite &= np.isfinite(evaluable[column].to_numpy(dtype=np.float64))
    evaluable = evaluable.loc[finite].copy()
    if evaluable.empty:
        raise ValueError("MR-13AI沒有可評估rows")
    ranked, swap_rows = apply_boundary_reorder(evaluable, top_k=int(top_k))
    af = _ranking_metrics(
        ranked,
        score_column="af_order_score",
        top_k=int(top_k),
        boundary_width=int(boundary_width),
    )
    ai = _ranking_metrics(
        ranked,
        score_column="ai_order_score",
        top_k=int(top_k),
        boundary_width=int(boundary_width),
    )

    def diff(left: Any, right: Any) -> float | None:
        if left is None or right is None:
            return None
        return float(right) - float(left)

    af_selected = dict(af.get("selected") or {})
    ai_selected = dict(ai.get("selected") or {})
    af_topk = dict(af.get("top_k_quality") or {})
    ai_topk = dict(ai.get("top_k_quality") or {})
    comparison = {
        "delta_mfe_mean_daily_spearman": diff(af.get("mfe_mean_daily_spearman"), ai.get("mfe_mean_daily_spearman")),
        "delta_mfe_pairwise_concordance": diff(af.get("mfe_pairwise_concordance"), ai.get("mfe_pairwise_concordance")),
        "delta_predicted_safety_to_score_mean_daily_spearman": diff(
            af.get("predicted_safety_to_score_mean_daily_spearman"),
            ai.get("predicted_safety_to_score_mean_daily_spearman"),
        ),
        "delta_score_to_low_adverse_mean_daily_spearman": diff(
            af.get("score_to_low_adverse_mean_daily_spearman"),
            ai.get("score_to_low_adverse_mean_daily_spearman"),
        ),
        "delta_top_k_mfe_r": diff(af_selected.get("top_k_mfe_r_mean"), ai_selected.get("top_k_mfe_r_mean")),
        "delta_top_k_adverse_r": diff(af_selected.get("top_k_adverse_r_mean"), ai_selected.get("top_k_adverse_r_mean")),
        "delta_top_k_high_mfe_pp": diff(af_selected.get("top_k_high_mfe_pct"), ai_selected.get("top_k_high_mfe_pct")),
        "delta_top_k_high_safety_pp": diff(af_selected.get("top_k_high_safety_pct"), ai_selected.get("top_k_high_safety_pct")),
        "delta_top_k_hmhs_pp": diff(af_selected.get("top_k_hmhs_pct"), ai_selected.get("top_k_hmhs_pct")),
        "delta_top_k_lift_r": diff(af_topk.get("top_k_raw_target_lift"), ai_topk.get("top_k_raw_target_lift")),
        "delta_boundary_concordance": diff(af_topk.get("boundary_concordance"), ai_topk.get("boundary_concordance")),
    }
    return {
        "population_n": int(len(ranked)),
        "af": af,
        "ai_phase0": ai,
        "comparison_ai_minus_af": comparison,
        "boundary_swaps": _swap_metrics(swap_rows),
        "swap_rows": swap_rows,
    }


def _load_forward_oos_frame(
    *,
    project_root: Path,
    filter_id: str,
    model_architecture: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_continuous_ranker_research_spec(SOURCE_PROFILE)
    if str(spec.model_research_id) != "MR-13AF":
        raise ValueError("MR-13AI Phase-0 source profile不再是MR-13AF")
    contract = load_continuous_ranker_oos_contract(
        str(project_root), str(filter_id), str(model_architecture), SOURCE_PROFILE
    )
    score_frame = read_breakout_quality_csv(contract.score_path).copy()
    required_score = {
        "ticker",
        "date",
        "group_index",
        "split",
        "target_raw_r",
        "target_daily_percentile",
        "model_score",
    }
    missing = sorted(required_score - set(score_frame.columns))
    if missing:
        raise ValueError(f"MR-13AF OOS score artifact缺少欄位: {missing}")
    score_frame = score_frame.loc[score_frame["split"].astype(str).eq("oos")].copy()
    score_frame["ticker"] = score_frame["ticker"].astype(str)
    score_frame["date"] = pd.to_datetime(score_frame["date"], errors="raise").dt.normalize()
    score_frame["group_index"] = pd.to_numeric(score_frame["group_index"], errors="raise").astype(int)
    for column in ("target_raw_r", "target_daily_percentile", "model_score"):
        score_frame[column] = pd.to_numeric(score_frame[column], errors="coerce")
    score_frame = score_frame.loc[
        np.isfinite(score_frame["target_raw_r"].to_numpy(dtype=np.float64))
        & np.isfinite(score_frame["target_daily_percentile"].to_numpy(dtype=np.float64))
        & np.isfinite(score_frame["model_score"].to_numpy(dtype=np.float64))
    ].copy()
    if score_frame.empty:
        raise ValueError("MR-13AF沒有target-evaluable Forward OOS rows")

    bundle = load_profile_continuous_ranker_data(
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=SOURCE_PROFILE,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=project_root,
    )
    groups = bundle.group_table.copy()
    missing_truth = sorted(set(_REQUIRED_TRUTH_COLUMNS) - set(groups.columns))
    if missing_truth:
        raise ValueError(f"MR-13AF canonical daily bundle缺少Phase-0 truth/context欄位: {missing_truth}")
    groups["ticker"] = groups["ticker"].astype(str)
    groups["date"] = pd.to_datetime(groups["date"], errors="raise").dt.normalize()
    truth = groups[["ticker", "date", "group_index", *_REQUIRED_TRUTH_COLUMNS]].copy()
    merged = score_frame.merge(
        truth,
        on=["ticker", "date", "group_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(score_frame):
        raise ValueError(
            "MR-13AI AF score與canonical daily bundle universe不一致: "
            f"score={len(score_frame)}, merged={len(merged)}"
        )
    if not np.allclose(
        merged["target_raw_r"].to_numpy(dtype=np.float64),
        merged["target_favorable_r"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("MR-13AI AF score raw target與canonical Pure-MFE truth不一致")
    if not np.allclose(
        merged["target_daily_percentile"].to_numpy(dtype=np.float64),
        merged["target_mfe_daily_percentile"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("MR-13AI AF score percentile target與canonical MFE percentile不一致")
    source = {
        "model_research_id": str(spec.model_research_id),
        "profile": SOURCE_PROFILE,
        "score_path": project_relative_display_path(contract.score_path, project_root=project_root),
        "report_path": project_relative_display_path(contract.report_path, project_root=project_root),
        "model_information_cutoff": contract.model_information_cutoff,
        "continuous_target_id": contract.continuous_target_id,
        "forward_oos_target_evaluable_rows": int(len(merged)),
    }
    return merged.sort_values(["date", "ticker", "group_index"], kind="mergesort").reset_index(drop=True), source


def build_phase0_diagnostic(
    *,
    project_root: Path,
    filter_id: str,
    model_architecture: str,
    top_k: int,
    boundary_width: int,
) -> dict[str, Any]:
    frame, source = _load_forward_oos_frame(
        project_root=project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
    )
    candidate_keys = load_official_breakout_candidate_keys(
        str(filter_id), allow_stale_source=False
    )
    candidate_mask = np.fromiter(
        (
            (str(ticker), pd.Timestamp(date).normalize()) in candidate_keys
            for ticker, date in zip(frame["ticker"], frame["date"])
        ),
        dtype=bool,
        count=len(frame),
    )
    breakout = frame.loc[candidate_mask].copy()
    if breakout.empty:
        raise ValueError("MR-13AI breakout candidate OOS slice為空")

    daily_result = evaluate_phase0_frame(
        frame,
        top_k=int(top_k),
        boundary_width=int(boundary_width),
    )
    breakout_result = evaluate_phase0_frame(
        breakout,
        top_k=int(top_k),
        boundary_width=int(boundary_width),
    )
    # Detailed pair rows can be large/noisy and are not needed in the primary JSON.
    daily_swap_rows = daily_result.pop("swap_rows")
    breakout_swap_rows = breakout_result.pop("swap_rows")
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "contract": dict(PHASE0_CONTRACT),
        "settings": {
            "top_k": int(top_k),
            "boundary_width": int(boundary_width),
        },
        "source": source,
        "scopes": {
            "forward_oos": daily_result,
            "breakout_candidate_oos": breakout_result,
        },
        "detail_rows": {
            "forward_boundary_rows": int(len(daily_swap_rows)),
            "forward_swap_rows": int(daily_swap_rows["swap"].sum()) if not daily_swap_rows.empty else 0,
            "breakout_boundary_rows": int(len(breakout_swap_rows)),
            "breakout_swap_rows": int(breakout_swap_rows["swap"].sum()) if not breakout_swap_rows.empty else 0,
        },
    }


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def _pct(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}%"


def _pp(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):+.{digits}f}pp"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MR-13AI Phase-0｜AF Local-Conflict Boundary Diagnostic",
        "",
        "此報表只讀MR-13AF frozen Forward-OOS score與canonical PIT-safe predicted-Safety context；不訓練、不refit、不建立PIT或Strategy arm。",
        "Boundary decision只看同日AF frozen model-score gap與Pred-Safety percentile；realized MFE/Safety只在決策完成後用於評估，無前視。",
        "",
        "## Contract",
        "",
        "- Scope：每天僅比較AF rank K與K+1；其餘order保持AF。",
        "- `d_score = AF_score(K) - AF_score(K+1)`；AF score本身為canonical `[0,1]` frozen model score。",
        "- `d_safety = S(K) - S(K+1)`。",
        "- `g = d_score + (1 - |d_score|) × d_safety`；只有`g < 0`才交換K/K+1，`g = 0`保留AF。",
        "- 不使用realized MFE gap作decision input；它只用於post-selection evidence。",
        "",
        "## AF vs MR-13AI Phase-0",
        "",
        "| Scope | Ordering | MFE Daily rho | Pair | Pred-Safety→Score rho | Score→Low-Adverse rho | Top-K MFE | Top-K Adverse | High-MFE | High-Safety | HM/HS | Top-K Lift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope_label, scope_key in (
        ("Forward OOS", "forward_oos"),
        ("Breakout candidate", "breakout_candidate_oos"),
    ):
        scope = dict((payload.get("scopes") or {}).get(scope_key) or {})
        for ordering_label, ordering_key in (("AF", "af"), ("AI Phase-0", "ai_phase0")):
            row = dict(scope.get(ordering_key) or {})
            selected = dict(row.get("selected") or {})
            topk = dict(row.get("top_k_quality") or {})
            pair = row.get("mfe_pairwise_concordance")
            lines.append(
                f"| {scope_label} | {ordering_label} "
                f"| {_fmt(row.get('mfe_mean_daily_spearman'))} "
                f"| {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
                f"| {_fmt(row.get('predicted_safety_to_score_mean_daily_spearman'))} "
                f"| {_fmt(row.get('score_to_low_adverse_mean_daily_spearman'))} "
                f"| {_fmt(selected.get('top_k_mfe_r_mean'))}R "
                f"| {_fmt(selected.get('top_k_adverse_r_mean'))}R "
                f"| {_pct(selected.get('top_k_high_mfe_pct'))} "
                f"| {_pct(selected.get('top_k_high_safety_pct'))} "
                f"| {_pct(selected.get('top_k_hmhs_pct'))} "
                f"| {_fmt(topk.get('top_k_raw_target_lift'))}R |"
            )
    lines.extend([
        "",
        "## AI − AF Delta",
        "",
        "| Scope | Δ MFE rho | Δ Pair | Δ Pred-Safety→Score | Δ Score→Safety | Δ Top-K MFE | Δ Adverse | Δ High-MFE | Δ High-Safety | Δ HM/HS | Δ Top-K Lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scope_label, scope_key in (
        ("Forward OOS", "forward_oos"),
        ("Breakout candidate", "breakout_candidate_oos"),
    ):
        delta = dict(((payload.get("scopes") or {}).get(scope_key) or {}).get("comparison_ai_minus_af") or {})
        lines.append(
            f"| {scope_label} | {_fmt(delta.get('delta_mfe_mean_daily_spearman'))} "
            f"| {_pp(None if delta.get('delta_mfe_pairwise_concordance') is None else float(delta['delta_mfe_pairwise_concordance'])*100.0)} "
            f"| {_fmt(delta.get('delta_predicted_safety_to_score_mean_daily_spearman'))} "
            f"| {_fmt(delta.get('delta_score_to_low_adverse_mean_daily_spearman'))} "
            f"| {_fmt(delta.get('delta_top_k_mfe_r'))}R "
            f"| {_fmt(delta.get('delta_top_k_adverse_r'))}R "
            f"| {_pp(delta.get('delta_top_k_high_mfe_pp'))} "
            f"| {_pp(delta.get('delta_top_k_high_safety_pp'))} "
            f"| {_pp(delta.get('delta_top_k_hmhs_pp'))} "
            f"| {_fmt(delta.get('delta_top_k_lift_r'))}R |"
        )
    lines.extend([
        "",
        "## Boundary Conflict Conversion",
        "",
        "| Scope | Boundary days | Swaps | Swap rate | Mean score gap | Mean abs Safety gap | Δ realized MFE | Δ realized Adverse | Actual Safety improved | MFE retained/improved | Δ High-Safety | Δ HM/HS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scope_label, scope_key in (
        ("Forward OOS", "forward_oos"),
        ("Breakout candidate", "breakout_candidate_oos"),
    ):
        row = dict(((payload.get("scopes") or {}).get(scope_key) or {}).get("boundary_swaps") or {})
        lines.append(
            f"| {scope_label} | {int(row.get('boundary_date_count', 0) or 0):,} "
            f"| {int(row.get('swap_date_count', 0) or 0):,} "
            f"| {_pct(row.get('swap_rate_pct'))} "
            f"| {_fmt(row.get('mean_score_gap_all_boundary_dates'))} "
            f"| {_fmt(row.get('mean_abs_safety_gap_all_boundary_dates'))} "
            f"| {_fmt(row.get('mean_realized_mfe_delta_r_ai_minus_af'))}R "
            f"| {_fmt(row.get('mean_realized_adverse_delta_r_ai_minus_af'))}R "
            f"| {_pct(row.get('actual_safety_improved_swap_pct'))} "
            f"| {_pct(row.get('mfe_retained_or_improved_swap_pct'))} "
            f"| {_pp(row.get('high_safety_delta_pp'))} "
            f"| {_pp(row.get('hmhs_delta_pp'))} |"
        )
    lines.extend([
        "",
        "## Research Boundary",
        "",
        "- Phase-0只回答AF的K-boundary local Safety conflict是否有joint value；不代表已建立可訓練MR-13AI模型。",
        "- 若Top-K MFE僅小幅下降且Adverse/High-Safety/HMHS有實質改善，才值得定義Phase-1 trainable pair semantics。",
        "- 若連此PIT-safe zero-retrain boundary reorder都不能改善joint outcome，MR-13AI直接STOP。",
        "",
    ])
    return "\n".join(lines)


def _write_outputs(payload: dict[str, Any], *, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_FILENAME
    markdown_path = output_dir / REPORT_MARKDOWN_FILENAME
    json_path.write_text(
        json.dumps(_json_native(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "MR-13AI Phase-0：只讀MR-13AF Forward-OOS score與PIT-safe Pred-Safety，"
            "在每日K/K+1 boundary做parameter-free local-conflict交換診斷；不訓練。"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument("--model-architecture", default=BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    parser.add_argument("--project-root", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path(PROJECT_ROOT).resolve()
    started = datetime.now(timezone.utc)
    payload = build_phase0_diagnostic(
        project_root=root,
        filter_id=str(args.filter_id),
        model_architecture=str(args.model_architecture),
        top_k=int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K),
        boundary_width=int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH),
    )
    output_dir = (
        resolve_filter_output_dir(root, filter_id=str(args.filter_id))
        / "research_diagnostics"
        / OUTPUT_DIRNAME
    )
    json_path, markdown_path = _write_outputs(payload, output_dir=output_dir)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    print(render_title("MR-13AI Phase-0｜AF Local-Conflict Boundary Diagnostic"))
    print(render_key_values([
        ("Source", "MR-13AF frozen Forward-OOS"),
        ("Top-K", str(BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K)),
        ("Decision", "只比較每日rank K / K+1；g<0才swap"),
        ("Lookahead", "無；realized MFE/Safety只作post-selection evaluation"),
        ("Training", "NONE"),
        ("Elapsed", f"{elapsed:.1f}s"),
    ]))
    rows = []
    for label, key in (("Forward OOS", "forward_oos"), ("Breakout", "breakout_candidate_oos")):
        scope = dict((payload.get("scopes") or {}).get(key) or {})
        delta = dict(scope.get("comparison_ai_minus_af") or {})
        swaps = dict(scope.get("boundary_swaps") or {})
        rows.append([
            label,
            f"{int(swaps.get('swap_date_count', 0) or 0):,}/{int(swaps.get('boundary_date_count', 0) or 0):,}",
            _pct(swaps.get("swap_rate_pct")),
            _fmt(delta.get("delta_top_k_mfe_r")),
            _fmt(delta.get("delta_top_k_adverse_r")),
            "-" if delta.get("delta_top_k_high_safety_pp") is None else f"{float(delta['delta_top_k_high_safety_pp']):+.2f}pp",
            "-" if delta.get("delta_top_k_hmhs_pp") is None else f"{float(delta['delta_top_k_hmhs_pp']):+.2f}pp",
        ])
    print(render_section("Boundary conversion｜AI − AF"))
    print(render_table(
        ["Scope", "Swaps", "Rate", "ΔMFE R", "ΔAdverse R", "ΔHigh-Safety", "ΔHM/HS"],
        rows,
    ))
    print_artifact_paths(
        [
            ("Markdown", markdown_path),
            ("JSON", json_path),
        ],
        project_root=root,
    )
    return 0


__all__ = [
    "PHASE0_CONTRACT",
    "SOURCE_PROFILE",
    "BoundaryDecision",
    "apply_boundary_reorder",
    "boundary_decision",
    "boundary_preference_margin",
    "build_phase0_diagnostic",
    "evaluate_phase0_frame",
    "main",
    "render_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
