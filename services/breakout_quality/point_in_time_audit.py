"""Audit Rolling point-in-time continuous-ranker predictive ordering."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K,
    get_breakout_quality_workflow_settings,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_target import (
    TARGET_MANIFEST_FILENAME,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.contract import (
    DEFAULT_MODEL_FILENAME,
    LABEL_PASS,
    LABEL_REJECT,
)
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME,
    SELECTION_POINT_IN_TIME_AUDIT_MARKDOWN_FILENAME,
    SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
    resolve_filter_output_dir,
    resolve_filter_point_in_time_fold_dir,
    resolve_selection_point_in_time_audit_json_path,
    resolve_selection_point_in_time_audit_markdown_path,
    resolve_selection_point_in_time_coverage_path,
    resolve_selection_point_in_time_manifest_path,
    resolve_selection_point_in_time_score_path,
)
from filters.breakout_quality.ranking_score_store import (
    clear_selection_point_in_time_ranking_contract_cache,
    derive_point_in_time_model_validation_gate,
)
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from services.breakout_quality.standard_model_sop import build_standard_model_sop
from services.breakout_quality import ranker_training as ranker_api
from services.breakout_quality.point_in_time_scores import (
    FOLD_MANIFEST_FILENAME,
    FOLD_SCORE_FILENAME,
    FOLD_VALIDATION_SCORE_FILENAME,
    POINT_IN_TIME_SCHEMA_VERSION,
)
from config.breakout_quality import (
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
)
from filters.breakout_quality.daily_ranker_data import select_breakout_candidate_group_ids
from services.breakout_quality.continuous_ranker_pipeline import (
    calculate_descriptive_rank_quality,
    calculate_spearman,
    load_continuous_ranker_data,
    primary_audit_metric_scope,
)
from config.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.console_report import (
    compact_console_enabled,
    console_color_enabled,
    paint,
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from core.report_style import (
    signal_for_auc,
    signal_for_coverage,
    signal_for_ratio,
    signal_for_signed_value,
    tone_for_signal,
)

AUDIT_SCHEMA_VERSION = 4
DRIFT_MEAN_SHIFT_STD_THRESHOLD = 1.0
_ORDERABLE_PIT_SCORE_COLUMN = "__pit_breakout_quality_score"


def parse_args(argv=None) -> argparse.Namespace:
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "驗證Rolling point-in-time Score對continuous target的排序能力；"
            "不執行策略optimizer，不以actual selected R作主要否決依據。"
        )
    )
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument(
        "--orderable-candidates",
        default=None,
        help="選填orderable candidate CSV；未指定時自動尋找既有Selection realization工件",
    )
    parser.add_argument(
        "--point-in-time-dir-override",
        default=None,
        help="指定隔離Rolling PIT根目錄；供Fixed-Window Stability audit使用",
    )
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="只供離線重現；預設要求來源CSV inventory與dataset一致",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取PIT manifest: {path}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"PIT manifest根節點必須是object: {path}")
    return payload


def _pit_artifact_paths(args) -> dict[str, Path]:
    override = str(args.point_in_time_dir_override or "").strip()
    if override:
        base = Path(override)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return {
            "score": base / SELECTION_POINT_IN_TIME_SCORE_FILENAME,
            "coverage": base / SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
            "manifest": base / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
            "audit_json": base / SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME,
            "audit_markdown": base / SELECTION_POINT_IN_TIME_AUDIT_MARKDOWN_FILENAME,
        }
    return {
        "score": resolve_selection_point_in_time_score_path(
            PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
        ),
        "coverage": resolve_selection_point_in_time_coverage_path(
            PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
        ),
        "manifest": resolve_selection_point_in_time_manifest_path(
            PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
        ),
        "audit_json": resolve_selection_point_in_time_audit_json_path(
            PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
        ),
        "audit_markdown": resolve_selection_point_in_time_audit_markdown_path(
            PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
        ),
    }


def _validate_score_artifacts(args) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = _pit_artifact_paths(args)
    score_path = paths["score"]
    coverage_path = paths["coverage"]
    manifest_path = paths["manifest"]
    if not score_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "找不到Rolling point-in-time score工件；請先執行 build-point-in-time-scores"
        )
    manifest = _read_json(manifest_path)
    expected_identity = {
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
    }
    actual_identity = {key: str(manifest.get(key, "")) for key in expected_identity}
    if actual_identity != expected_identity:
        raise ValueError(
            "Rolling PIT manifest identity與CLI不一致: "
            f"expected={expected_identity}, actual={actual_identity}"
        )
    if (
        int(manifest.get("schema_version", -1)) != POINT_IN_TIME_SCHEMA_VERSION
        or manifest.get("status") != "BUILT"
    ):
        raise ValueError("Rolling PIT manifest schema/status不支援")
    if manifest.get("score_column") != "breakout_quality_score":
        raise ValueError("Rolling PIT manifest score column不一致")
    lookahead = manifest.get("lookahead_contract") or {}
    if not bool(lookahead.get("every_score_uses_model_not_trained_on_scored_event")):
        raise ValueError("Rolling PIT manifest未宣告未見事件評分契約")
    if not bool(lookahead.get("training_requires_label_eval_end_before_score_start")):
        raise ValueError("Rolling PIT manifest未宣告label completion cutoff")
    if bool(lookahead.get("oos_rows_or_target_statistics_used_for_training_or_epoch_selection")):
        raise ValueError("Rolling PIT manifest宣告使用OOS訓練或選epoch")
    if bool(lookahead.get("future_target_in_score_table")):
        raise ValueError("Rolling PIT manifest宣告Score表包含Future Target")
    runtime_eligibility = manifest.get("runtime_eligibility") or {}
    legacy_runtime_contract = bool(
        runtime_eligibility.get("not_eligible_for_forward_oos_runtime")
    )
    rolling_runtime_contract = bool(
        runtime_eligibility.get("per_score_fold_information_cutoff_required")
    )
    if not (legacy_runtime_contract or rolling_runtime_contract):
        raise ValueError("Rolling PIT manifest缺少可驗證的runtime information-cutoff契約")

    artifacts = manifest.get("artifacts") or {}
    if build_file_manifest(score_path) != artifacts.get("scores"):
        raise ValueError("Rolling point-in-time score hash與manifest不一致")
    if artifacts.get("coverage") is not None:
        if not coverage_path.is_file():
            raise FileNotFoundError(f"Rolling PIT coverage工件不存在: {coverage_path}")
        if build_file_manifest(coverage_path) != artifacts.get("coverage"):
            raise ValueError("Rolling point-in-time coverage hash與manifest不一致")

    frame = pd.read_csv(
        score_path,
        encoding="utf-8-sig",
        dtype={
            "ticker": "string",
            "fold_id": "string",
            "model_information_cutoff": "string",
        },
    )
    required = {
        "ticker",
        "date",
        "group_index",
        "breakout_quality_score",
        "fold_id",
        "model_information_cutoff",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rolling PIT score缺少欄位: {missing}")
    forbidden = sorted(
        column
        for column in frame.columns
        if column in {"label", "target_raw_r", "target_daily_percentile"}
        or column.startswith("future_")
    )
    if forbidden:
        raise ValueError(f"Rolling PIT runtime score表不得包含Future Target欄位: {forbidden}")

    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="raise").astype(np.int64)
    frame["breakout_quality_score"] = pd.to_numeric(
        frame["breakout_quality_score"], errors="raise"
    ).astype(np.float64)
    frame["fold_id"] = frame["fold_id"].astype(str)
    frame["model_information_cutoff"] = pd.to_datetime(
        frame["model_information_cutoff"], errors="raise"
    ).dt.normalize()
    if bool(frame["group_index"].duplicated().any()):
        raise ValueError("Rolling PIT score group_index重複")
    if bool(frame.duplicated(["ticker", "date"]).any()):
        raise ValueError("Rolling PIT score ticker/date重複")
    if not np.isfinite(frame["breakout_quality_score"]).all():
        raise ValueError("Rolling PIT score含非有限值")
    if bool(
        (
            (frame["breakout_quality_score"] < 0.0)
            | (frame["breakout_quality_score"] > 1.0)
        ).any()
    ):
        raise ValueError("Rolling PIT score超出[0,1]")
    if bool((frame["model_information_cutoff"] >= frame["date"]).any()):
        raise ValueError("Rolling PIT score存在information cutoff未早於score date")

    score_period = manifest.get("score_period") or {}
    score_start = pd.Timestamp(score_period.get("start"))
    score_end = pd.Timestamp(score_period.get("end"))
    if pd.isna(score_start) or pd.isna(score_end) or score_start > score_end:
        raise ValueError("Rolling PIT manifest score period無效")
    if bool(((frame["date"] < score_start) | (frame["date"] > score_end)).any()):
        raise ValueError("Rolling PIT score日期超出manifest score period")

    fold_records = manifest.get("folds")
    if not isinstance(fold_records, list) or not fold_records:
        raise ValueError("Rolling PIT manifest缺少fold records")
    if int(manifest.get("fold_count", -1)) != len(fold_records):
        raise ValueError("Rolling PIT manifest fold_count不一致")
    fold_ids = [str(item.get("fold_id") or "").strip() for item in fold_records]
    if any(not fold_id for fold_id in fold_ids) or len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Rolling PIT manifest fold_id重複或空白")
    fold_contract = dict(zip(fold_ids, fold_records))
    if set(frame["fold_id"].unique()) != set(fold_contract):
        raise ValueError("Rolling PIT score fold集合與manifest不一致")
    for fold_id, fold_frame in frame.groupby("fold_id", sort=False):
        record = fold_contract[fold_id]
        expected_count = int((record.get("group_counts") or {}).get("score", -1))
        if len(fold_frame) != expected_count:
            raise ValueError(f"Rolling PIT {fold_id} score count與manifest不一致")
        expected_cutoff = pd.Timestamp(record.get("model_information_cutoff"))
        if set(fold_frame["model_information_cutoff"].unique()) != {expected_cutoff}:
            raise ValueError(f"Rolling PIT {fold_id} information cutoff與manifest不一致")
        periods = record.get("planned_periods") or {}
        fold_start = pd.Timestamp(periods.get("score_start"))
        fold_end = pd.Timestamp(periods.get("score_end"))
        if bool(((fold_frame["date"] < fold_start) | (fold_frame["date"] > fold_end)).any()):
            raise ValueError(f"Rolling PIT {fold_id} score日期超出fold期間")
        override = str(args.point_in_time_dir_override or "").strip()
        fold_dir = (
            (_pit_artifact_paths(args)["manifest"].parent / "folds" / fold_id)
            if override
            else resolve_filter_point_in_time_fold_dir(
                PROJECT_ROOT,
                args.filter_id,
                fold_id,
                args.model_architecture,
                args.experiment_profile,
            )
        )
        fold_manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
        checkpoint_path = fold_dir / DEFAULT_MODEL_FILENAME
        fold_score_path = fold_dir / FOLD_SCORE_FILENAME
        if build_file_manifest(fold_manifest_path) != record.get("fold_manifest"):
            raise ValueError(f"Rolling PIT {fold_id} fold manifest hash不一致")
        fold_artifacts = record.get("artifacts") or {}
        if build_file_manifest(checkpoint_path) != fold_artifacts.get("checkpoint"):
            raise ValueError(f"Rolling PIT {fold_id} checkpoint hash不一致")
        if build_file_manifest(fold_score_path) != fold_artifacts.get("scores"):
            raise ValueError(f"Rolling PIT {fold_id} fold score hash不一致")

    coverage = manifest.get("coverage") or {}
    if int(coverage.get("scored_group_count", -1)) != len(frame):
        raise ValueError("Rolling PIT scored_group_count與score rows不一致")
    if int(coverage.get("duplicate_group_count", -1)) != 0:
        raise ValueError("Rolling PIT manifest宣告存在duplicate groups")
    if int(coverage.get("missing_group_count", -1)) != 0:
        raise ValueError("Rolling PIT manifest宣告存在missing groups")
    if int(coverage.get("extra_group_count", -1)) != 0:
        raise ValueError("Rolling PIT manifest宣告存在extra groups")
    return frame, manifest


def _primary_scope_key(payload: dict[str, Any]) -> str:
    return str(
        (payload.get("decision_contract") or {}).get(
            "primary_metric_scope", "pass_only_target"
        )
    )


def _primary_scope_label(payload: dict[str, Any]) -> str:
    return str(
        (payload.get("decision_contract") or {}).get(
            "primary_metric_label", "PASS-only"
        )
    )


def _primary_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    key = _primary_scope_key(payload)
    return dict((payload.get("metrics") or {}).get(key) or {})


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [LABEL_REJECT, LABEL_PASS])
    y = y[valid]
    s = s[valid]
    positive = int((y == LABEL_PASS).sum())
    negative = int((y == LABEL_REJECT).sum())
    if positive == 0 or negative == 0:
        return None
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_positive = float(ranks[y == LABEL_PASS].sum())
    return float(
        (rank_sum_positive - positive * (positive + 1) / 2.0)
        / float(positive * negative)
    )


def _decile_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "group_count": 0,
            "decile_group_count": 0,
            "top_decile_target_mean": None,
            "bottom_decile_target_mean": None,
            "top_bottom_target_spread": None,
        }
    ordered = frame.sort_values(
        ["breakout_quality_score", "date", "ticker", "group_index"],
        kind="mergesort",
    )
    count = max(1, int(math.ceil(len(ordered) * 0.10)))
    top = float(ordered.tail(count)["target_raw_r"].mean())
    bottom = float(ordered.head(count)["target_raw_r"].mean())
    return {
        "group_count": int(len(ordered)),
        "decile_group_count": int(count),
        "top_decile_target_mean": top,
        "bottom_decile_target_mean": bottom,
        "top_bottom_target_spread": float(top - bottom),
    }


def _scope_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "group_count": 0,
            "global_spearman": None,
            "eligible_day_count": 0,
            "valid_spearman_day_count": 0,
            "mean_daily_spearman": None,
            "median_daily_spearman": None,
            "pairwise_concordance": None,
            "comparable_pair_count": 0,
            "top_k_quality": None,
            **_decile_metrics(frame),
        }
    rank_quality = calculate_descriptive_rank_quality(
        frame["date"].to_numpy(),
        frame["breakout_quality_score"].to_numpy(dtype=np.float64),
        frame["target_raw_r"].to_numpy(dtype=np.float64),
        top_k=BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K,
        boundary_width=BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH,
    )
    return {
        "group_count": int(len(frame)),
        "global_spearman": calculate_spearman(
            frame["breakout_quality_score"].to_numpy(dtype=np.float64),
            frame["target_raw_r"].to_numpy(dtype=np.float64),
        ),
        **rank_quality,
        **_decile_metrics(frame),
    }



def _build_safety_raw_mfe_evaluation(
    bundle,
    merged: pd.DataFrame,
    candidate_ids: np.ndarray,
) -> dict[str, Any]:
    """Build Rolling multi-head evidence from persisted capability-owned score sidecars."""

    raw_head_columns = {"raw_safety_score", "raw_mfe_score"}
    if not raw_head_columns.issubset(set(merged.columns)):
        return {}

    work = merged.copy()
    for column in sorted(raw_head_columns):
        work[column] = pd.to_numeric(work[column], errors="raise").astype(np.float64)
        if not np.isfinite(work[column]).all():
            raise ValueError(f"Rolling PIT multi-head sidecar含非有限值: {column}")
        if bool(((work[column] < 0.0) | (work[column] > 1.0)).any()):
            raise ValueError(f"Rolling PIT multi-head sidecar超出[0,1]: {column}")

    target_scope_mask = np.asarray(
        bundle.target_valid & np.isfinite(bundle.raw_target), dtype=bool
    )
    primary_mfe_percentile = ranker_api.build_daily_percentile_targets(
        bundle.raw_target, target_scope_mask, bundle.group_table["date"]
    )
    raw_mfe_targets = ranker_api.build_conditional_mfe_opportunity_targets_for_training(
        bundle.group_table, primary_mfe_percentile
    )

    def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
        ids = frame["group_index"].to_numpy(dtype=np.int64)
        scores = {
            "raw_safety": frame["raw_safety_score"].to_numpy(dtype=np.float64),
            "raw_mfe": frame["raw_mfe_score"].to_numpy(dtype=np.float64),
        }
        return ranker_api.safety_raw_mfe_metrics(
            ids, bundle.group_table, raw_mfe_targets, scores,
            include_top_k_quality=True,
        )

    valid_target = work[work["target_available"]].copy()
    evaluation: dict[str, Any] = {}
    if len(valid_target) >= 2:
        evaluation["oos"] = evaluate(valid_target)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if len(candidate_ids) >= 2:
        breakout = valid_target[valid_target["group_index"].isin(candidate_ids)].copy()
        if len(breakout) >= 2:
            evaluation["breakout_candidate_oos"] = evaluate(breakout)
    return evaluation


def _build_hs_conditional_mfe_evaluation(
    bundle,
    merged: pd.DataFrame,
    candidate_ids: np.ndarray,
    *,
    experiment_profile: str,
) -> dict[str, Any]:
    """Build Rolling HS/Conditional-MFE evidence from persisted score sidecars.

    The Rolling score artifact owns only decision-time model outputs.  This audit
    reconstructs evaluation-only targets from the same canonical target builder;
    no checkpoint fitting or score reranking is performed.
    """

    recipe = get_continuous_ranker_execution_recipe(str(experiment_profile))
    if "hs_conditional_mfe" not in set(recipe.training_policy.report_evidence_families):
        return {}
    if "raw_safety_score" not in merged.columns:
        return {}
    threshold = recipe.objective_policy.secondary_pair_scope_threshold
    if threshold is None:
        raise ValueError("HS-Conditional Rolling evidence缺少canonical true-HS threshold")

    work = merged.copy()
    work["raw_safety_score"] = pd.to_numeric(work["raw_safety_score"], errors="raise").astype(np.float64)
    work["breakout_quality_score"] = pd.to_numeric(
        work["breakout_quality_score"], errors="raise"
    ).astype(np.float64)
    for column in ("raw_safety_score", "breakout_quality_score"):
        if not np.isfinite(work[column]).all():
            raise ValueError(f"Rolling PIT HS-Conditional sidecar含非有限值: {column}")
        if bool(((work[column] < 0.0) | (work[column] > 1.0)).any()):
            raise ValueError(f"Rolling PIT HS-Conditional sidecar超出[0,1]: {column}")

    hs_targets = ranker_api.build_hs_conditional_mfe_targets(
        bundle.group_table,
        np.isfinite(np.asarray(bundle.raw_target, dtype=np.float32)),
        true_hs_percentile_cutoff=float(threshold),
    )
    safety_pct = ranker_api.build_daily_percentile_targets(
        work["raw_safety_score"].to_numpy(dtype=np.float32),
        np.ones(len(work), dtype=bool),
        work["date"].reset_index(drop=True),
    )
    work["__predicted_safety_percentile"] = safety_pct

    def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
        ids = frame["group_index"].to_numpy(dtype=np.int64)
        heads = {
            "raw_safety": frame["raw_safety_score"].to_numpy(dtype=np.float64),
            "conditional_mfe": frame["breakout_quality_score"].to_numpy(dtype=np.float64),
        }
        return ranker_api.hs_conditional_mfe_metrics(
            ids,
            bundle.group_table,
            hs_targets,
            heads,
            include_top_k_quality=True,
            predicted_safety_percentile=frame["__predicted_safety_percentile"].to_numpy(
                dtype=np.float32
            ),
        )

    valid_target = work[work["target_available"]].copy()
    evaluation: dict[str, Any] = {}
    if len(valid_target) >= 2:
        evaluation["oos"] = evaluate(valid_target)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if len(candidate_ids) >= 2:
        breakout = valid_target[valid_target["group_index"].isin(candidate_ids)].copy()
        if len(breakout) >= 2:
            evaluation["breakout_candidate_oos"] = evaluate(breakout)
    return evaluation


def _yearly_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    work["year"] = work["date"].dt.year.astype(int)
    for year, year_frame in work.groupby("year", sort=True):
        metrics = _scope_metrics(year_frame)
        rows.append({"year": int(year), **metrics})
    return rows


def _fold_metrics(
    frame: pd.DataFrame, *, primary_metric_scope: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pooled_std = float(frame["breakout_quality_score"].std(ddof=0)) if len(frame) else math.nan
    previous_mean: float | None = None
    max_shift = 0.0
    drift_folds: list[str] = []
    for fold_id, fold_frame in frame.groupby("fold_id", sort=True):
        score = fold_frame["breakout_quality_score"].to_numpy(dtype=np.float64)
        mean = float(np.mean(score))
        shift = None
        if previous_mean is not None and math.isfinite(pooled_std) and pooled_std > 0.0:
            shift = float(abs(mean - previous_mean) / pooled_std)
            max_shift = max(max_shift, shift)
            if shift >= DRIFT_MEAN_SHIFT_STD_THRESHOLD:
                drift_folds.append(str(fold_id))
        if primary_metric_scope == "all_valid_target":
            target_frame = fold_frame[fold_frame["target_available"]].copy()
        else:
            target_frame = fold_frame[
                fold_frame["target_available"] & (fold_frame["label"] == LABEL_PASS)
            ].copy()
        rows.append(
            {
                "fold_id": str(fold_id),
                "group_count": int(len(fold_frame)),
                "score_mean": mean,
                "score_std": float(np.std(score)),
                "score_p10": float(np.quantile(score, 0.10)),
                "score_p50": float(np.quantile(score, 0.50)),
                "score_p90": float(np.quantile(score, 0.90)),
                "adjacent_mean_shift_in_pooled_std": shift,
                "primary_target_spearman": calculate_spearman(
                    target_frame["breakout_quality_score"].to_numpy(dtype=np.float64),
                    target_frame["target_raw_r"].to_numpy(dtype=np.float64),
                )
                if len(target_frame) >= 2
                else None,
                "pass_target_spearman": (
                    calculate_spearman(
                        target_frame["breakout_quality_score"].to_numpy(dtype=np.float64),
                        target_frame["target_raw_r"].to_numpy(dtype=np.float64),
                    )
                    if primary_metric_scope == "pass_only_target" and len(target_frame) >= 2
                    else None
                ),
            }
        )
        previous_mean = mean
    return rows, {
        "criterion": (
            "adjacent fold score mean shift >= "
            f"{DRIFT_MEAN_SHIFT_STD_THRESHOLD:.1f} pooled score standard deviation"
        ),
        "drift_flag": bool(drift_folds),
        "flagged_folds": drift_folds,
        "max_adjacent_mean_shift_in_pooled_std": float(max_shift),
    }


def _default_orderable_path(filter_id: str, target_id: str) -> Path:
    return (
        resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
        / "continuous_targets"
        / target_id
        / "selection_strategy_realization_audit"
        / "selection_orderable_candidates.csv"
    )


def _orderable_coverage(
    score_frame: pd.DataFrame,
    *,
    requested_path: str | None,
    filter_id: str,
    target_id: str,
) -> dict[str, Any]:
    path = (
        Path(requested_path).expanduser().resolve()
        if requested_path
        else _default_orderable_path(filter_id, target_id)
    )
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "orderable candidate工件不存在"}
    candidates = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": "string"})
    date_column = next(
        (
            column
            for column in ("target_date", "signal_date", "candidate_date", "date")
            if column in candidates.columns
        ),
        None,
    )
    if "ticker" not in candidates.columns or date_column is None:
        return {
            "available": False,
            "path": str(path),
            "reason": (
                "orderable candidate缺少ticker及事件日期欄位"
                "（target_date/signal_date/candidate_date/date）"
            ),
        }
    if _ORDERABLE_PIT_SCORE_COLUMN in candidates.columns:
        raise ValueError(
            "orderable candidate工件使用了PIT audit保留欄位: "
            f"{_ORDERABLE_PIT_SCORE_COLUMN}"
        )
    candidate_has_existing_score = "breakout_quality_score" in candidates.columns
    lookup = score_frame[["ticker", "date", "breakout_quality_score"]].copy().rename(
        columns={"breakout_quality_score": _ORDERABLE_PIT_SCORE_COLUMN}
    )
    lookup["date"] = lookup["date"].dt.strftime("%Y-%m-%d")
    work = candidates.copy()
    work["ticker"] = work["ticker"].astype(str)
    work["date"] = pd.to_datetime(work[date_column], errors="raise").dt.strftime("%Y-%m-%d")
    work = work.merge(lookup, how="left", on=["ticker", "date"], validate="many_to_one")
    scored = np.isfinite(pd.to_numeric(work[_ORDERABLE_PIT_SCORE_COLUMN], errors="coerce"))
    return {
        "available": True,
        "path": str(path),
        "candidate_count": int(len(work)),
        "scored_candidate_count": int(scored.sum()),
        "coverage_rate": float(scored.mean()) if len(work) else None,
        "unscored_candidate_count": int((~scored).sum()),
        "candidate_artifact_has_existing_breakout_quality_score": bool(
            candidate_has_existing_score
        ),
        "coverage_score_source": "selection_point_in_time_scores",
    }


def _fmt_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def _fmt_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric * 100.0:.{digits}f}%"


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _signed_tone(value: Any) -> str:
    return tone_for_signal(signal_for_signed_value(value))


def _coverage_tone(value: Any) -> str:
    return tone_for_signal(signal_for_coverage(value))


def _ratio_tone(numerator: Any, denominator: Any) -> str:
    return tone_for_signal(signal_for_ratio(numerator, denominator))


def _auc_tone(value: Any) -> str:
    return tone_for_signal(signal_for_auc(value))


def _colored_section(title: str, *, number: int, color: bool) -> str:
    return render_section(
        paint(title, "cyan", enabled=color, bold=True),
        number=number,
    )


def _render_console_table(headers, rows):
    return render_table(headers, rows).splitlines()


def _direction_summary(yearly_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in yearly_rows if row.get("global_spearman") is not None]
    positive_spearman = [row for row in valid_rows if float(row["global_spearman"]) > 0.0]
    positive_spread = [
        row
        for row in valid_rows
        if row.get("top_bottom_target_spread") is not None
        and float(row["top_bottom_target_spread"]) > 0.0
    ]
    return {
        "valid_year_count": int(len(valid_rows)),
        "positive_spearman_year_count": int(len(positive_spearman)),
        "positive_spearman_year_rate": (
            float(len(positive_spearman) / len(valid_rows)) if valid_rows else None
        ),
        "positive_spread_year_count": int(len(positive_spread)),
        "positive_spread_year_rate": (
            float(len(positive_spread) / len(valid_rows)) if valid_rows else None
        ),
    }


def render_console_summary(payload: dict[str, Any], *, color: bool = False) -> str:
    primary = _primary_metrics(payload)
    all_target = payload["metrics"]["all_valid_target"]
    candidate_target = dict((payload.get("metrics") or {}).get("breakout_candidate_target") or {})
    secondary_label = (
        "Breakout candidates"
        if _primary_scope_key(payload) == "all_valid_target"
        else "All valid"
    )
    secondary = candidate_target if secondary_label == "Breakout candidates" else all_target
    classification = payload["classification_overlap"]
    drift = payload["fold_drift"]
    direction = payload["direction_summary"]
    score_coverage = payload.get("score_coverage") or {}
    workflow = payload.get("workflow") or {}
    score_coverage_text = (
        f"{int(score_coverage.get('scored_group_count', 0)):,}/"
        f"{int(score_coverage.get('expected_group_count', 0)):,} "
        f"({_fmt_percent(score_coverage.get('coverage_rate'))})"
    )

    lines = [
        "",
        render_title(
            paint(
                "Breakout Quality Selection Point-in-time 模型評估報表",
                "cyan",
                enabled=color,
                bold=True,
            )
        ),
        render_key_values((
            ("Filter ID", payload["filter_id"]),
            ("Architecture", payload["model_architecture"]),
            ("Experiment Profile", payload["experiment_profile"]),
            ("Continuous Target", payload["continuous_target_id"]),
            ("Training Scope", workflow.get("training_label_scope", "-")),
            ("Random Seed", workflow.get("seed", "-")),
            ("Score Period", f"{payload['score_period']['start']} ～ {payload['score_period']['end']}"),
            ("PIT Folds", workflow.get("fold_count", "-")),
            (
                "Fold／Validation",
                f"{workflow.get('fold_months', '-')}／"
                f"{workflow.get('inner_validation_months', '-')} months",
            ),
            (
                "Score Coverage",
                paint(
                    score_coverage_text,
                    _coverage_tone(score_coverage.get("coverage_rate")),
                    enabled=color,
                    bold=True,
                ),
            ),
            (
                "Primary Evidence",
                paint(
                    f"{_primary_scope_label(payload)} No-time Target ordering",
                    "cyan",
                    enabled=color,
                    bold=True,
                ),
            ),
            (
                "Strategy Result",
                paint(
                    "尚未執行；本報表只評估模型排序能力",
                    "yellow",
                    enabled=color,
                    bold=True,
                ),
            ),
        )),
        _colored_section("核心排序能力", number=1, color=color),
    ]
    lines.extend(
        _render_console_table(
            ["Scope", "Groups", "Spearman", "Daily rho", "Top", "Bottom", "Spread"],
            [
                [
                    paint(_primary_scope_label(payload), "cyan", enabled=color, bold=True),
                    f"{int(primary['group_count']):,}",
                    paint(
                        _fmt_metric(primary.get("global_spearman")),
                        _signed_tone(primary.get("global_spearman")),
                        enabled=color,
                        bold=True,
                    ),
                    paint(
                        _fmt_metric(primary.get("mean_daily_spearman")),
                        _signed_tone(primary.get("mean_daily_spearman")),
                        enabled=color,
                        bold=True,
                    ),
                    _fmt_metric(primary.get("top_decile_target_mean")),
                    _fmt_metric(primary.get("bottom_decile_target_mean")),
                    paint(
                        _fmt_metric(primary.get("top_bottom_target_spread")),
                        _signed_tone(primary.get("top_bottom_target_spread")),
                        enabled=color,
                        bold=True,
                    ),
                ],
                [
                    secondary_label,
                    f"{int(secondary.get('group_count', 0) or 0):,}",
                    paint(
                        _fmt_metric(secondary.get("global_spearman")),
                        _signed_tone(secondary.get("global_spearman")),
                        enabled=color,
                    ),
                    paint(
                        _fmt_metric(secondary.get("mean_daily_spearman")),
                        _signed_tone(secondary.get("mean_daily_spearman")),
                        enabled=color,
                    ),
                    _fmt_metric(secondary.get("top_decile_target_mean")),
                    _fmt_metric(secondary.get("bottom_decile_target_mean")),
                    paint(
                        _fmt_metric(secondary.get("top_bottom_target_spread")),
                        _signed_tone(secondary.get("top_bottom_target_spread")),
                        enabled=color,
                    ),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            _colored_section(f"年度穩定性（{_primary_scope_label(payload)}）", number=2, color=color),
            (
                "正向 Spearman 年度："
                + paint(
                    f"{direction['positive_spearman_year_count']}/{direction['valid_year_count']} "
                    f"({_fmt_percent(direction['positive_spearman_year_rate'])})",
                    _ratio_tone(
                        direction["positive_spearman_year_count"],
                        direction["valid_year_count"],
                    ),
                    enabled=color,
                    bold=True,
                )
                + "；"
                "正向 Top-bottom spread 年度："
                + paint(
                    f"{direction['positive_spread_year_count']}/{direction['valid_year_count']} "
                    f"({_fmt_percent(direction['positive_spread_year_rate'])})",
                    _ratio_tone(
                        direction["positive_spread_year_count"],
                        direction["valid_year_count"],
                    ),
                    enabled=color,
                    bold=True,
                )
            ),
        ]
    )
    yearly_rows = [
        [
            str(row["year"]),
            f"{int(row['group_count']):,}",
            paint(
                _fmt_metric(row.get("global_spearman")),
                _signed_tone(row.get("global_spearman")),
                enabled=color,
                bold=True,
            ),
            paint(
                _fmt_metric(row.get("mean_daily_spearman")),
                _signed_tone(row.get("mean_daily_spearman")),
                enabled=color,
            ),
            paint(
                _fmt_metric(row.get("top_bottom_target_spread")),
                _signed_tone(row.get("top_bottom_target_spread")),
                enabled=color,
                bold=True,
            ),
        ]
        for row in payload.get("yearly_primary", payload.get("yearly_pass_only", []))
    ]
    lines.extend(
        _render_console_table(
            ["Year", "Groups", "Spearman", "Daily rho", "Spread"],
            yearly_rows,
        )
    )

    lines.append(_colored_section("Fold 分布與漂移", number=3, color=color))
    fold_rows = [
        [
            str(row["fold_id"]),
            f"{int(row['group_count']):,}",
            _fmt_metric(row.get("score_mean")),
            _fmt_metric(row.get("score_std")),
            _fmt_metric(row.get("score_p10")),
            _fmt_metric(row.get("score_p50")),
            _fmt_metric(row.get("score_p90")),
            paint(
                _fmt_metric(row.get("primary_target_spearman", row.get("pass_target_spearman"))),
                _signed_tone(row.get("primary_target_spearman", row.get("pass_target_spearman"))),
                enabled=color,
                bold=True,
            ),
        ]
        for row in payload["fold_metrics"]
    ]
    lines.extend(
        _render_console_table(
            ["Fold", "Groups", "Mean", "Std", "P10", "P50", "P90", "Primary rho"],
            fold_rows,
        )
    )
    lines.extend(
        [
            "Drift flag         : "
            + paint(
                str(bool(drift["drift_flag"])),
                "red" if bool(drift["drift_flag"]) else "green",
                enabled=color,
                bold=True,
            ),
            "Max adjacent shift : "
            + paint(
                _fmt_metric(drift.get("max_adjacent_mean_shift_in_pooled_std")),
                (
                    "red"
                    if float(drift.get("max_adjacent_mean_shift_in_pooled_std") or 0.0)
                    >= DRIFT_MEAN_SHIFT_STD_THRESHOLD
                    else "green"
                ),
                enabled=color,
                bold=True,
            )
            + " pooled SD",
            "Flagged folds      : "
            + paint(
                ", ".join(drift["flagged_folds"]) if drift["flagged_folds"] else "-",
                "red" if drift["flagged_folds"] else "gray",
                enabled=color,
            ),
            "",
            _colored_section("PASS／REJECT 重疊診斷", number=4, color=color),
            "Score vs PASS AUC          : "
            + paint(
                _fmt_metric(classification.get("score_vs_pass_reject_auc")),
                _auc_tone(classification.get("score_vs_pass_reject_auc")),
                enabled=color,
                bold=True,
            ),
            f"Overall PASS share         : {_fmt_percent(classification.get('overall_pass_share'))}",
            "Top score decile PASS share: "
            + paint(
                _fmt_percent(classification.get("top_score_decile_pass_share")),
                _signed_tone(
                    (_finite_number(classification.get("top_score_decile_pass_share")) or 0.0)
                    - (_finite_number(classification.get("overall_pass_share")) or 0.0)
                ),
                enabled=color,
                bold=True,
            ),
            f"{_primary_scope_label(payload)} Target Spearman  : "
            + paint(
                _fmt_metric(primary.get("global_spearman")),
                _signed_tone(primary.get("global_spearman")),
                enabled=color,
                bold=True,
            ),
            "",
            _colored_section("Orderable candidate Score coverage", number=5, color=color),
        ]
    )
    orderable = payload["orderable_candidate_coverage"]
    if orderable.get("available"):
        orderable_text = (
            f"{int(orderable['scored_candidate_count']):,}/{int(orderable['candidate_count']):,} "
            f"({_fmt_percent(orderable.get('coverage_rate'))})；"
            f"未評分={int(orderable['unscored_candidate_count']):,}"
        )
        lines.append(
            paint(
                orderable_text,
                _coverage_tone(orderable.get("coverage_rate")),
                enabled=color,
                bold=True,
            )
        )
    else:
        unavailable_text = f"未提供：{orderable.get('reason')}"
        if not compact_console_enabled():
            orderable_path = project_relative_display_path(
                orderable.get("path") or "-", project_root=PROJECT_ROOT
            )
            unavailable_text += f"；{orderable_path}"
        lines.append(
            paint(
                unavailable_text,
                "yellow",
                enabled=color,
                bold=True,
            )
        )

    lines.extend(
        [
            "",
            _colored_section("綜合狀態", number=6, color=color),
            "- 報表狀態："
            + paint(
                "RESULT_AVAILABLE_PENDING_REVIEW",
                "yellow",
                enabled=color,
                bold=True,
            ),
            "- 策略 optimizer："
            + paint("未執行", "yellow", enabled=color, bold=True),
            "- Future Target runtime sort："
            + paint("未使用", "green", enabled=color, bold=True),
            "- Forward-OOS runtime："
            + paint(
                "本 PIT 工件不可直接使用",
                "yellow",
                enabled=color,
                bold=True,
            ),
            "- 下一步："
            + paint(
                "先審閱本報表；只有排序能力在多數年份穩定為正，才進入策略績效驗證。",
                "cyan",
                enabled=color,
                bold=True,
            ),
        ]
    )
    return "\n".join(lines)


def render_compact_console_summary(
    payload: dict[str, Any], *, color: bool = False
) -> str:
    """Render the interactive workflow summary without repeating full audit detail."""

    primary = _primary_metrics(payload)
    classification = payload["classification_overlap"]
    direction = payload["direction_summary"]
    drift = payload["fold_drift"]
    score_coverage = payload.get("score_coverage") or {}
    score_period = payload.get("score_period") or {}
    workflow = payload.get("workflow") or {}
    orderable = payload["orderable_candidate_coverage"]
    gate = derive_point_in_time_model_validation_gate(payload)
    candidate = dict((payload.get("metrics") or {}).get("breakout_candidate_target") or {})

    gate_passed = gate["status"] == "PASS"
    orderable_text = (
        f"{int(orderable['scored_candidate_count']):,}/"
        f"{int(orderable['candidate_count']):,} "
        f"({_fmt_percent(orderable.get('coverage_rate'))})"
        if orderable.get("available")
        else "尚未提供；策略 replay 時建立"
    )
    rows = (
        (
            "期間／Coverage",
            f"{score_period.get('start')}～{score_period.get('end')}"
            f" | folds={int(workflow.get('fold_count') or 0)}"
            f" | {int(score_coverage.get('scored_group_count', 0)):,}/"
            f"{int(score_coverage.get('expected_group_count', 0)):,}"
            f" ({_fmt_percent(score_coverage.get('coverage_rate'))})",
        ),
        (
            "核心排序",
            paint(
                f"{_primary_scope_label(payload)} rho={_fmt_metric(primary.get('global_spearman'))}"
                f" | daily rho={_fmt_metric(primary.get('mean_daily_spearman'))}"
                f" | pair={_fmt_percent(primary.get('pairwise_concordance'))}"
                f" | spread={_fmt_metric(primary.get('top_bottom_target_spread'))}R",
                "green" if gate_passed else "red",
                enabled=color,
                bold=True,
            ),
        ),
        (
            "年度穩定",
            paint(
                f"rho>0：{direction['positive_spearman_year_count']}/"
                f"{direction['valid_year_count']}"
                f" | spread>0：{direction['positive_spread_year_count']}/"
                f"{direction['valid_year_count']}"
                f" | drift={bool(drift['drift_flag'])}",
                "green" if gate_passed and not bool(drift["drift_flag"]) else "yellow",
                enabled=color,
                bold=True,
            ),
        ),
        *(
            ((
                "Breakout slice",
                f"rho={_fmt_metric(candidate.get('global_spearman'))}"
                f" | daily rho={_fmt_metric(candidate.get('mean_daily_spearman'))}"
                f" | pair={_fmt_percent(candidate.get('pairwise_concordance'))}"
                f" | top-K lift={_fmt_metric((candidate.get('top_k_quality') or {}).get('top_k_raw_target_lift'))}R"
                f" | boundary={_fmt_percent((candidate.get('top_k_quality') or {}).get('boundary_concordance'))}",
            ),)
            if int(candidate.get("group_count", 0) or 0) > 0
            else ()
        ),
        (
            "分類重疊",
            (
                f"AUC={_fmt_metric(classification.get('score_vs_pass_reject_auc'))}"
                f" | Top decile PASS={_fmt_percent(classification.get('top_score_decile_pass_share'))}"
                f" | Overall={_fmt_percent(classification.get('overall_pass_share'))}"
                if int(classification.get("valid_label_group_count", 0) or 0) > 0
                else "不適用（daily sample無PASS／REJECT label）"
            ),
        ),
        ("Orderable coverage", orderable_text),
        (
            "模型 Gate",
            paint(
                gate["status"],
                "green" if gate_passed else "red",
                enabled=color,
                bold=True,
            ),
        ),
        (
            "下一步",
            paint(
                "執行策略績效驗證"
                if gate_passed
                else "停止策略驗證，先檢查模型排序失敗項目",
                "cyan" if gate_passed else "red",
                enabled=color,
                bold=True,
            ),
        ),
    )
    return "\n".join(
        (
            render_section(
                paint("PIT 模型驗證", "cyan", enabled=color, bold=True)
            ),
            render_key_values(rows),
        )
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    primary = _primary_metrics(payload)
    all_target = payload["metrics"]["all_valid_target"]
    classification = payload["classification_overlap"]
    direction = payload["direction_summary"]
    drift = payload["fold_drift"]
    workflow = payload.get("workflow") or {}
    coverage = payload.get("score_coverage") or {}

    lines = [
        "# Breakout Quality Rolling Point-in-time 模型評估報表",
        "",
        "> 本報表只評估 Rolling point-in-time Score 對 Future Target 的離線排序能力。",
        "> 未執行策略 optimizer、未使用 Future Target 作 runtime buy-sort；OOS合法性由每個score fold的information cutoff決定。",
        "",
        "## 1. 執行設定與資料範圍",
        "",
        "| 項目 | 內容 |",
        "|---|---|",
        f"| Filter ID | `{payload['filter_id']}` |",
        f"| Architecture | `{payload['model_architecture']}` |",
        f"| Experiment profile | `{payload['experiment_profile']}` |",
        f"| Continuous Target | `{payload['continuous_target_id']}` |",
        f"| Training scope | `{workflow.get('training_label_scope', '-')}` |",
        f"| Sample scope | `{workflow.get('training_sample_scope', '-')}` |",
        f"| Seed | `{workflow.get('seed', '-')}` |",
        f"| Score period | `{payload['score_period']['start']} ~ {payload['score_period']['end']}` |",
        f"| PIT folds | `{workflow.get('fold_count', '-')}` |",
        f"| Fold／Validation | `{workflow.get('fold_months', '-')}／{workflow.get('inner_validation_months', '-')} months` |",
        f"| Evaluation policy | `{(workflow.get('evaluation_policy') or {}).get('mode', '-')}`；train window=`{(workflow.get('evaluation_policy') or {}).get('train_window_months', 'expanding')}` |",
        f"| Score coverage | `{int(coverage.get('scored_group_count', 0)):,}/{int(coverage.get('expected_group_count', 0)):,}`（{_fmt_percent(coverage.get('coverage_rate'))}） |",
        "",
        "## 2. 核心排序能力",
        "",
        "| Scope | Groups | Global Spearman | Mean daily Spearman | Pair | Top decile Target | Bottom decile Target | Top-bottom spread |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| **{_primary_scope_label(payload)}（主要判讀）** | {primary['group_count']:,} | {_fmt_metric(primary['global_spearman'])} | {_fmt_metric(primary['mean_daily_spearman'])} | {_fmt_percent(primary.get('pairwise_concordance'))} | {_fmt_metric(primary['top_decile_target_mean'])} | {_fmt_metric(primary['bottom_decile_target_mean'])} | {_fmt_metric(primary['top_bottom_target_spread'])} |",
        (
            f"| All valid labels | {all_target['group_count']:,} | {_fmt_metric(all_target['global_spearman'])} | {_fmt_metric(all_target['mean_daily_spearman'])} | {_fmt_percent(all_target.get('pairwise_concordance'))} | {_fmt_metric(all_target['top_decile_target_mean'])} | {_fmt_metric(all_target['bottom_decile_target_mean'])} | {_fmt_metric(all_target['top_bottom_target_spread'])} |"
            if _primary_scope_key(payload) != "all_valid_target"
            else f"| Breakout candidate diagnostic | {int((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('group_count', 0)):,} | {_fmt_metric((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('global_spearman'))} | {_fmt_metric((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('mean_daily_spearman'))} | {_fmt_percent((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('pairwise_concordance'))} | {_fmt_metric((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_decile_target_mean'))} | {_fmt_metric((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('bottom_decile_target_mean'))} | {_fmt_metric((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_bottom_target_spread'))} |"
        ),
        "",
        "### Top-K / K-boundary",
        "",
        "| Scope | NDCG@K | Top-K Target | Lift | Oracle overlap | Boundary | Boundary gap | 競爭日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {_primary_scope_label(payload)} | {_fmt_metric((primary.get('top_k_quality') or {}).get('ndcg_at_k'))} | "
            f"{_fmt_metric((primary.get('top_k_quality') or {}).get('top_k_raw_target_mean'))} | "
            f"{_fmt_metric((primary.get('top_k_quality') or {}).get('top_k_raw_target_lift'))} | "
            f"{_fmt_percent((primary.get('top_k_quality') or {}).get('oracle_top_k_overlap'))} | "
            f"{_fmt_percent((primary.get('top_k_quality') or {}).get('boundary_concordance'))} | "
            f"{_fmt_metric((primary.get('top_k_quality') or {}).get('boundary_raw_target_gap'))} | "
            f"{int((primary.get('top_k_quality') or {}).get('competition_date_count', 0) or 0):,} |"
        ),
        (
            f"| Breakout candidate diagnostic | {_fmt_metric(((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_k_quality') or {}).get('ndcg_at_k'))} | "
            f"{_fmt_metric(((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_k_quality') or {}).get('top_k_raw_target_mean'))} | "
            f"{_fmt_metric(((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_k_quality') or {}).get('top_k_raw_target_lift'))} | "
            f"{_fmt_percent(((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_k_quality') or {}).get('oracle_top_k_overlap'))} | "
            f"{_fmt_percent(((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_k_quality') or {}).get('boundary_concordance'))} | "
            f"{_fmt_metric(((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_k_quality') or {}).get('boundary_raw_target_gap'))} | "
            f"{int(((payload.get('metrics') or {}).get('breakout_candidate_target', {}).get('top_k_quality') or {}).get('competition_date_count', 0) or 0):,} |"
        ),
        "",
        f"## 3. 年度穩定性（{_primary_scope_label(payload)}）",
        "",
        f"- 有效年度：**{direction['valid_year_count']}**",
        f"- Spearman 為正：**{direction['positive_spearman_year_count']}/{direction['valid_year_count']}**（{_fmt_percent(direction['positive_spearman_year_rate'])}）",
        f"- Top-bottom spread 為正：**{direction['positive_spread_year_count']}/{direction['valid_year_count']}**（{_fmt_percent(direction['positive_spread_year_rate'])}）",
        "",
        "| Year | Groups | Spearman | Mean daily Spearman | Top-bottom spread |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("yearly_primary", payload.get("yearly_pass_only", [])):
        lines.append(
            f"| {row['year']} | {row['group_count']:,} | {_fmt_metric(row['global_spearman'])} | "
            f"{_fmt_metric(row['mean_daily_spearman'])} | {_fmt_metric(row['top_bottom_target_spread'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. Fold 分布與漂移",
            "",
            "| Fold | Groups | Mean | Std | P10 | P50 | P90 | Primary Target Spearman | Adjacent mean shift（pooled SD） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["fold_metrics"]:
        lines.append(
            f"| `{row['fold_id']}` | {row['group_count']:,} | {_fmt_metric(row['score_mean'])} | "
            f"{_fmt_metric(row['score_std'])} | {_fmt_metric(row['score_p10'])} | "
            f"{_fmt_metric(row['score_p50'])} | {_fmt_metric(row['score_p90'])} | "
            f"{_fmt_metric(row.get('primary_target_spearman', row.get('pass_target_spearman')))} | "
            f"{_fmt_metric(row['adjacent_mean_shift_in_pooled_std'])} |"
        )
    lines.extend(
        [
            "",
            f"- Drift flag：`{drift['drift_flag']}`",
            f"- 最大相鄰 fold 平均值位移：{_fmt_metric(drift['max_adjacent_mean_shift_in_pooled_std'])} pooled SD",
            f"- 被標記 fold：{', '.join(drift['flagged_folds']) if drift['flagged_folds'] else '-'}",
            f"- 判定口徑：{drift['criterion']}",
            "",
            "## 5. PASS／REJECT 重疊診斷",
            "",
            "| 指標 | 結果 |",
            "|---|---:|",
            f"| Score vs PASS／REJECT AUC | {_fmt_metric(classification['score_vs_pass_reject_auc'])} |",
            f"| Overall PASS share | {_fmt_percent(classification['overall_pass_share'])} |",
            f"| Top score decile PASS share | {_fmt_percent(classification['top_score_decile_pass_share'])} |",
            f"| {_primary_scope_label(payload)} Target Spearman | {_fmt_metric(primary['global_spearman'])} |",
            "",
            classification["interpretation_contract"],
            "",
            "## 6. Orderable candidate Score coverage",
            "",
        ]
    )
    orderable = payload["orderable_candidate_coverage"]
    if orderable.get("available"):
        lines.extend(
            [
                f"- 已評分：**{orderable['scored_candidate_count']:,}/{orderable['candidate_count']:,}**（{_fmt_percent(orderable['coverage_rate'])}）",
                f"- 未評分：**{orderable['unscored_candidate_count']:,}**",
                f"- 候選工件：`{orderable['path']}`",
                f"- Coverage Score source：`{orderable['coverage_score_source']}`",
            ]
        )
    else:
        lines.append(f"- 尚不可用：{orderable.get('reason')}；`{orderable.get('path')}`")

    sources = payload.get("source_artifacts") or {}
    outputs = payload.get("report_artifacts") or {}

    def source_path(name: str) -> str:
        record = sources.get(name, "-")
        if isinstance(record, dict):
            return str(
                record.get("path")
                or record.get("filename")
                or record.get("source")
                or "-"
            )
        return str(record)

    lines.extend(
        [
            "",
            "## 7. 研究邊界與下一步",
            "",
            f"- 主要證據：{_primary_scope_label(payload)} No-time Target ordering。",
            "- Actual selected R 不作本階段主要否決依據。",
            "- 策略 optimizer：**未執行**。",
            "- Future Target runtime sort：**未使用**。",
            "- OOS legality：以每個score fold的information cutoff為準，不要求Frozen Forward。",
            "- 只有本報表顯示多數年度具有穩定正向排序能力後，才進入策略績效驗證。",
            "",
            "## 8. 工件",
            "",
            f"- PIT manifest：`{source_path('point_in_time_manifest')}`",
            f"- PIT Scores：`{source_path('point_in_time_scores')}`",
            f"- PIT coverage：`{source_path('point_in_time_coverage')}`",
            f"- Continuous Target source：`{source_path('continuous_target_manifest')}`",
            f"- Markdown 易讀報表：`{outputs.get('markdown', '-')}`",
            f"- 完整指標 JSON：`{outputs.get('json', '-')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_point_in_time_scores_audit(
    args: argparse.Namespace,
    *,
    authorization_mode: str = "rolling",
    strategy_compare_profile_id: str | None = None,
    strategy_compare_source_id: str | None = None,
    strategy_compare_seed: int | None = None,
) -> int:
    settings = get_breakout_quality_workflow_settings(
        experiment_profile=str(args.experiment_profile)
    )
    mode = str(authorization_mode).strip().lower()
    if mode == "rolling":
        if not settings.rolling_authorized:
            raise ValueError(
                f"目前profile未授權Rolling PIT scores: {args.experiment_profile}"
            )
    elif mode == "strategy_compare":
        from config.strategy_compare import (
            validate_single_seed_strategy_conversion_authorization,
        )

        if strategy_compare_seed is None:
            raise ValueError("Strategy Compare PIT audit缺少canonical seed")
        validate_single_seed_strategy_conversion_authorization(
            strategy_profile_id=str(strategy_compare_profile_id or ""),
            dl_id=str(strategy_compare_source_id or ""),
            filter_id=str(args.filter_id),
            model_architecture=str(args.model_architecture),
            experiment_profile=str(args.experiment_profile),
            seed=int(strategy_compare_seed),
        )
    else:
        raise ValueError(f"未知PIT audit authorization mode: {authorization_mode!r}")
    score_frame, manifest = _validate_score_artifacts(args)
    bundle = load_continuous_ranker_data(
        filter_id=args.filter_id,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
        preload_feature_bank=False,
        allow_stale_source=bool(args.allow_stale_source),
        project_root=PROJECT_ROOT,
    )
    if str(bundle.profile.continuous_target_id) != str(manifest.get("continuous_target_id")):
        raise ValueError("PIT score manifest target與目前profile不一致")
    manifest_sample_scope = str(
        manifest.get("training_sample_scope")
        or TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
    )
    if manifest_sample_scope != str(bundle.profile.training_sample_scope):
        raise ValueError(
            "PIT score manifest sample scope與目前profile不一致: "
            f"manifest={manifest_sample_scope}, "
            f"profile={bundle.profile.training_sample_scope}"
        )
    groups = bundle.group_table[["group_index", "ticker", "date", "label"]].copy()
    groups["ticker"] = groups["ticker"].astype(str)
    groups["date"] = pd.to_datetime(groups["date"], errors="raise").dt.normalize()
    groups["target_raw_r"] = bundle.raw_target
    groups["target_available"] = bundle.target_valid & np.isfinite(bundle.raw_target)
    merged = score_frame.merge(
        groups,
        how="left",
        on="group_index",
        suffixes=("", "_expected"),
        validate="one_to_one",
    )
    if bool(merged["label"].isna().any()):
        raise ValueError("PIT score存在無法對應dataset group的列")
    identity_mismatch = (
        merged["ticker"].astype(str) != merged["ticker_expected"].astype(str)
    ) | (merged["date"] != merged["date_expected"])
    if bool(identity_mismatch.any()):
        raise ValueError(
            "PIT score ticker/date與dataset group identity不一致: "
            f"mismatch_groups={int(identity_mismatch.sum())}"
        )
    merged = merged.drop(columns=["ticker_expected", "date_expected"])

    valid_target = merged[merged["target_available"]].copy()
    pass_target = valid_target[valid_target["label"] == LABEL_PASS].copy()
    reject_target = valid_target[valid_target["label"] == LABEL_REJECT].copy()
    all_metrics = _scope_metrics(valid_target)
    pass_metrics = _scope_metrics(pass_target)
    reject_metrics = _scope_metrics(reject_target)
    primary_metric_scope = primary_audit_metric_scope(bundle)
    primary_frame = valid_target if primary_metric_scope == "all_valid_target" else pass_target
    primary_metric_label = (
        "All eligible stock-days"
        if primary_metric_scope == "all_valid_target"
        else "PASS-only"
    )

    candidate_metrics = _scope_metrics(pd.DataFrame(columns=valid_target.columns))
    candidate_ids = np.empty(0, dtype=np.int64)
    if (
        bundle.profile.training_sample_scope
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        candidate_ids = select_breakout_candidate_group_ids(
            bundle,
            merged["group_index"].to_numpy(dtype=np.int64),
            allow_stale_source=bool(args.allow_stale_source),
        )
        candidate_metrics = _scope_metrics(
            valid_target[valid_target["group_index"].isin(candidate_ids)].copy()
        )

    safety_raw_mfe_evaluation = _build_safety_raw_mfe_evaluation(
        bundle, merged, candidate_ids
    )
    hs_conditional_mfe_evaluation = _build_hs_conditional_mfe_evaluation(
        bundle, merged, candidate_ids, experiment_profile=str(args.experiment_profile)
    )

    validation_frames: list[pd.DataFrame] = []
    fold_records_by_id = {str(item["fold_id"]): dict(item) for item in manifest.get("folds", [])}
    for fold_id, record in fold_records_by_id.items():
        override = str(args.point_in_time_dir_override or "").strip()
        fold_dir = (
            (_pit_artifact_paths(args)["manifest"].parent / "folds" / fold_id)
            if override
            else resolve_filter_point_in_time_fold_dir(
                PROJECT_ROOT, args.filter_id, fold_id,
                args.model_architecture, args.experiment_profile,
            )
        )
        validation_path = fold_dir / FOLD_VALIDATION_SCORE_FILENAME
        artifacts = dict(record.get("artifacts") or {})
        if not validation_path.is_file():
            raise FileNotFoundError(f"Rolling Standard SOP缺少validation score sidecar: {validation_path}")
        if build_file_manifest(validation_path) != artifacts.get("validation_scores"):
            raise ValueError(f"Rolling Standard SOP {fold_id} validation score hash不一致")
        required = {"group_index", "date", "breakout_quality_score", "fold_id"}
        header_columns = set(pd.read_csv(validation_path, encoding="utf-8-sig", nrows=0).columns)
        missing = sorted(required.difference(header_columns))
        if missing:
            raise ValueError(f"Rolling Standard SOP {fold_id} validation sidecar缺欄: {missing}")
        validation_frame = pd.read_csv(
            validation_path,
            encoding="utf-8-sig",
            usecols=sorted(required),
            dtype={"fold_id": "string"},
        )
        validation_frame["date"] = pd.to_datetime(validation_frame["date"], errors="raise").dt.normalize()
        validation_frame["rank_group"] = (
            validation_frame["fold_id"].astype(str) + "|" + validation_frame["date"].dt.strftime("%Y-%m-%d")
        )
        validation_frames.append(validation_frame)
    if not validation_frames:
        raise ValueError("Rolling Standard SOP缺少validation score evidence")
    standard_validation_frame = pd.concat(validation_frames, ignore_index=True)
    standard_oos_frame = valid_target[["group_index", "date", "breakout_quality_score"]].copy()
    if bundle.profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        standard_breakout_frame = valid_target[valid_target["group_index"].isin(candidate_ids)][
            ["group_index", "date", "breakout_quality_score"]
        ].copy()
    else:
        standard_breakout_frame = standard_oos_frame.copy()

    label_valid = merged[merged["label"].isin([LABEL_REJECT, LABEL_PASS])].copy()
    ordered = label_valid.sort_values("breakout_quality_score", kind="mergesort")
    decile_count = max(1, int(math.ceil(len(ordered) * 0.10))) if len(ordered) else 0
    classification_overlap = {
        "valid_label_group_count": int(len(label_valid)),
        "score_vs_pass_reject_auc": _rank_auc(
            label_valid["label"].to_numpy(dtype=np.int64),
            label_valid["breakout_quality_score"].to_numpy(dtype=np.float64),
        ),
        "overall_pass_share": float((label_valid["label"] == LABEL_PASS).mean())
        if len(label_valid)
        else None,
        "top_score_decile_pass_share": float(
            (ordered.tail(decile_count)["label"] == LABEL_PASS).mean()
        )
        if decile_count
        else None,
        "interpretation_contract": (
            "AUC/decile PASS share衡量Score是否主要重複binary分類；"
            "PASS-only target Spearman衡量分類內部magnitude排序能力。"
            if len(label_valid)
            else "Daily-universal sample沒有PASS／REJECT binary label；此診斷不適用。"
        ),
    }
    yearly_primary = _yearly_metrics(primary_frame)
    yearly_pass_only = _yearly_metrics(pass_target)
    fold_rows, fold_drift = _fold_metrics(
        merged, primary_metric_scope=primary_metric_scope
    )
    orderable = _orderable_coverage(
        score_frame,
        requested_path=args.orderable_candidates,
        filter_id=args.filter_id,
        target_id=str(bundle.profile.continuous_target_id),
    )
    score_period = dict(manifest.get("score_period") or {})
    paths = _pit_artifact_paths(args)
    output_json = paths["audit_json"]
    output_markdown = paths["audit_markdown"]
    score_path = paths["score"]
    manifest_path = paths["manifest"]
    coverage_path = paths["coverage"]
    target_manifest_path = (
        resolve_continuous_target_dir(
            PROJECT_ROOT,
            args.filter_id,
            target_id=str(bundle.profile.continuous_target_id),
        )
        / TARGET_MANIFEST_FILENAME
    )
    if (
        bundle.profile.training_sample_scope
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        target_source_artifact = {
            "path": None,
            "source": "embedded_in_point_in_time_manifest",
            "target_id": str(bundle.profile.continuous_target_id),
            "target_contract": bundle.target_manifest.get("target_contract"),
        }
    else:
        if not target_manifest_path.is_file():
            raise FileNotFoundError(
                f"缺少Continuous Target manifest: {target_manifest_path}"
            )
        target_source_artifact = {
            "path": str(target_manifest_path),
            **build_file_manifest(target_manifest_path),
        }
    standard_model_sop = build_standard_model_sop(
        group_table=bundle.group_table,
        raw_target=bundle.raw_target,
        training_objective=str(bundle.profile.training_objective),
        validation_scores=standard_validation_frame,
        oos_scores=standard_oos_frame,
        breakout_scores=standard_breakout_frame,
        evaluation_mode="rolling_oos",
        mode_extensions={
            "rolling": {
                "fold_count": int(manifest.get("fold_count", 0) or 0),
                "fold_months": int(manifest.get("fold_months", 0) or 0),
                "direction_summary": _direction_summary(yearly_primary),
                "fold_drift": fold_drift,
            }
        },
    )

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "continuous_target_id": str(bundle.profile.continuous_target_id),
        "score_period": score_period,
        "score_group_count": int(len(merged)),
        "score_coverage": manifest.get("coverage"),
        "workflow": {
            "training_label_scope": manifest.get("training_label_scope"),
            "training_sample_scope": manifest.get("training_sample_scope"),
            "seed": manifest.get("seed"),
            "fold_count": manifest.get("fold_count"),
            "fold_months": manifest.get("fold_months"),
            "inner_validation_months": manifest.get("inner_validation_months"),
            "selection_period": manifest.get("selection_period"),
            "available_history_period": manifest.get("available_history_period"),
            "evaluation_policy": manifest.get("evaluation_policy") or {
                "mode": "legacy_expanding_annual_refit",
                "train_window_months": None,
            },
            "torch_execution": manifest.get("torch_execution"),
            "elapsed_sec": manifest.get("elapsed_sec"),
        },
        "standard_model_sop": standard_model_sop,
        "safety_raw_mfe_evaluation": safety_raw_mfe_evaluation,
        "hs_conditional_mfe_evaluation": hs_conditional_mfe_evaluation,
        "metrics": {
            "pass_only_target": pass_metrics,
            "reject_only_target": reject_metrics,
            "all_valid_target": all_metrics,
            "breakout_candidate_target": candidate_metrics,
        },
        "yearly_primary": yearly_primary,
        "yearly_pass_only": yearly_pass_only,
        "direction_summary": _direction_summary(yearly_primary),
        "fold_metrics": fold_rows,
        "fold_drift": fold_drift,
        "classification_overlap": classification_overlap,
        "orderable_candidate_coverage": orderable,
        "decision_contract": {
            "primary_metric_scope": primary_metric_scope,
            "primary_metric_label": primary_metric_label,
            "primary_evidence": f"{primary_metric_label} no-time target ordering",
            "actual_selected_r_is_primary_veto": False,
            "strategy_optimizer_executed": False,
            "future_target_used_for_runtime_sort": False,
        },
        "source_artifacts": {
            "point_in_time_manifest": {
                "path": str(manifest_path),
                **build_file_manifest(manifest_path),
            },
            "point_in_time_scores": {
                "path": str(score_path),
                **build_file_manifest(score_path),
            },
            "point_in_time_coverage": {
                "path": str(coverage_path),
                **build_file_manifest(coverage_path),
            },
            "continuous_target_manifest": target_source_artifact,
        },
        "report_artifacts": {
            "markdown": str(output_markdown),
            "json": str(output_json),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, payload)
    output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    clear_selection_point_in_time_ranking_contract_cache()
    color_enabled = console_color_enabled()
    if compact_console_enabled():
        print(render_compact_console_summary(payload, color=color_enabled))
    else:
        print(render_console_summary(payload, color=color_enabled))
    print_artifact_paths(
        (("Markdown", output_markdown), ("完整指標 JSON", output_json)),
        project_root=PROJECT_ROOT,
    )
    return 0


def audit_selection_point_in_time_scores(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    orderable_candidates: str | None = None,
    point_in_time_dir_override: str | None = None,
    allow_stale_source: bool = False,
    strategy_compare_profile_id: str | None = None,
    strategy_compare_source_id: str | None = None,
    strategy_compare_seed: int | None = None,
) -> int:
    """Programmatic PIT audit service used by formal artifact workflows."""

    argv = [
        "--filter-id", str(filter_id),
        "--model-architecture", str(model_architecture),
        "--experiment-profile", str(experiment_profile),
    ]
    if orderable_candidates is not None:
        argv.extend(["--orderable-candidates", str(orderable_candidates)])
    if point_in_time_dir_override is not None:
        argv.extend(["--point-in-time-dir-override", str(point_in_time_dir_override)])
    if allow_stale_source:
        argv.append("--allow-stale-source")
    strategy_scoped = bool(
        str(strategy_compare_profile_id or "").strip()
        and str(strategy_compare_source_id or "").strip()
    )
    if strategy_scoped != (strategy_compare_seed is not None):
        raise ValueError(
            "Strategy Compare PIT audit authorization必須同時提供profile/source/seed"
        )
    return _run_point_in_time_scores_audit(
        parse_args(argv),
        authorization_mode=("strategy_compare" if strategy_scoped else "rolling"),
        strategy_compare_profile_id=strategy_compare_profile_id,
        strategy_compare_source_id=strategy_compare_source_id,
        strategy_compare_seed=strategy_compare_seed,
    )


def main(argv=None) -> int:
    return _run_point_in_time_scores_audit(parse_args(argv))


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "audit_selection_point_in_time_scores",
    "main",
    "parse_args",
    "render_compact_console_summary",
    "render_console_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
