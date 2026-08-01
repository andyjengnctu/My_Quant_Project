"""Audit Selection point-in-time continuous-ranker predictive ordering."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import get_breakout_quality_workflow_settings
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import (
    DEFAULT_MODEL_FILENAME,
    LABEL_PASS,
    LABEL_REJECT,
)
from filters.breakout_quality.paths import (
    resolve_filter_output_dir,
    resolve_filter_point_in_time_fold_dir,
    resolve_selection_point_in_time_audit_json_path,
    resolve_selection_point_in_time_audit_markdown_path,
    resolve_selection_point_in_time_coverage_path,
    resolve_selection_point_in_time_manifest_path,
    resolve_selection_point_in_time_score_path,
)
from tools.filters.breakout_quality.common import PROJECT_ROOT, write_json
from tools.filters.breakout_quality.build_point_in_time_scores import (
    FOLD_MANIFEST_FILENAME,
    FOLD_SCORE_FILENAME,
    POINT_IN_TIME_SCHEMA_VERSION,
)
from tools.filters.breakout_quality.continuous_ranker_pipeline import (
    calculate_spearman,
    load_continuous_ranker_data,
)

AUDIT_SCHEMA_VERSION = 1
DRIFT_MEAN_SHIFT_STD_THRESHOLD = 1.0
_ORDERABLE_PIT_SCORE_COLUMN = "__pit_breakout_quality_score"


def parse_args(argv=None) -> argparse.Namespace:
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "驗證Selection point-in-time Score對continuous target的排序能力；"
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


def _validate_score_artifacts(args) -> tuple[pd.DataFrame, dict[str, Any]]:
    score_path = resolve_selection_point_in_time_score_path(
        PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
    )
    coverage_path = resolve_selection_point_in_time_coverage_path(
        PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
    )
    manifest_path = resolve_selection_point_in_time_manifest_path(
        PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
    )
    if not score_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "找不到Selection point-in-time score工件；請先執行 build-point-in-time-scores"
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
            "Selection PIT manifest identity與CLI不一致: "
            f"expected={expected_identity}, actual={actual_identity}"
        )
    if (
        int(manifest.get("schema_version", -1)) != POINT_IN_TIME_SCHEMA_VERSION
        or manifest.get("status") != "BUILT"
    ):
        raise ValueError("Selection PIT manifest schema/status不支援")
    if manifest.get("score_column") != "breakout_quality_score":
        raise ValueError("Selection PIT manifest score column不一致")
    lookahead = manifest.get("lookahead_contract") or {}
    if not bool(lookahead.get("every_score_uses_model_not_trained_on_scored_event")):
        raise ValueError("Selection PIT manifest未宣告未見事件評分契約")
    if not bool(lookahead.get("training_requires_label_eval_end_before_score_start")):
        raise ValueError("Selection PIT manifest未宣告label completion cutoff")
    if bool(lookahead.get("oos_rows_or_target_statistics_used_for_training_or_epoch_selection")):
        raise ValueError("Selection PIT manifest宣告使用OOS訓練或選epoch")
    if bool(lookahead.get("future_target_in_score_table")):
        raise ValueError("Selection PIT manifest宣告Score表包含Future Target")
    runtime_eligibility = manifest.get("runtime_eligibility") or {}
    if bool(runtime_eligibility.get("eligible")):
        raise ValueError("Selection PIT builder工件不得在模型audit前標記為策略可用")
    if not bool(runtime_eligibility.get("not_eligible_for_forward_oos_runtime")):
        raise ValueError("Selection PIT工件必須明確禁止forward-OOS runtime")

    artifacts = manifest.get("artifacts") or {}
    if build_file_manifest(score_path) != artifacts.get("scores"):
        raise ValueError("Selection point-in-time score hash與manifest不一致")
    if artifacts.get("coverage") is not None:
        if not coverage_path.is_file():
            raise FileNotFoundError(f"Selection PIT coverage工件不存在: {coverage_path}")
        if build_file_manifest(coverage_path) != artifacts.get("coverage"):
            raise ValueError("Selection point-in-time coverage hash與manifest不一致")

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
        raise ValueError(f"Selection PIT score缺少欄位: {missing}")
    forbidden = sorted(
        column
        for column in frame.columns
        if column in {"label", "target_raw_r", "target_daily_percentile"}
        or column.startswith("future_")
    )
    if forbidden:
        raise ValueError(f"Selection PIT runtime score表不得包含Future Target欄位: {forbidden}")

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
        raise ValueError("Selection PIT score group_index重複")
    if bool(frame.duplicated(["ticker", "date"]).any()):
        raise ValueError("Selection PIT score ticker/date重複")
    if not np.isfinite(frame["breakout_quality_score"]).all():
        raise ValueError("Selection PIT score含非有限值")
    if bool(
        (
            (frame["breakout_quality_score"] < 0.0)
            | (frame["breakout_quality_score"] > 1.0)
        ).any()
    ):
        raise ValueError("Selection PIT score超出[0,1]")
    if bool((frame["model_information_cutoff"] >= frame["date"]).any()):
        raise ValueError("Selection PIT score存在information cutoff未早於score date")

    score_period = manifest.get("score_period") or {}
    score_start = pd.Timestamp(score_period.get("start"))
    score_end = pd.Timestamp(score_period.get("end"))
    if pd.isna(score_start) or pd.isna(score_end) or score_start > score_end:
        raise ValueError("Selection PIT manifest score period無效")
    if bool(((frame["date"] < score_start) | (frame["date"] > score_end)).any()):
        raise ValueError("Selection PIT score日期超出manifest score period")

    fold_records = manifest.get("folds")
    if not isinstance(fold_records, list) or not fold_records:
        raise ValueError("Selection PIT manifest缺少fold records")
    if int(manifest.get("fold_count", -1)) != len(fold_records):
        raise ValueError("Selection PIT manifest fold_count不一致")
    fold_ids = [str(item.get("fold_id") or "").strip() for item in fold_records]
    if any(not fold_id for fold_id in fold_ids) or len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Selection PIT manifest fold_id重複或空白")
    fold_contract = dict(zip(fold_ids, fold_records))
    if set(frame["fold_id"].unique()) != set(fold_contract):
        raise ValueError("Selection PIT score fold集合與manifest不一致")
    for fold_id, fold_frame in frame.groupby("fold_id", sort=False):
        record = fold_contract[fold_id]
        expected_count = int((record.get("group_counts") or {}).get("score", -1))
        if len(fold_frame) != expected_count:
            raise ValueError(f"Selection PIT {fold_id} score count與manifest不一致")
        expected_cutoff = pd.Timestamp(record.get("model_information_cutoff"))
        if set(fold_frame["model_information_cutoff"].unique()) != {expected_cutoff}:
            raise ValueError(f"Selection PIT {fold_id} information cutoff與manifest不一致")
        periods = record.get("planned_periods") or {}
        fold_start = pd.Timestamp(periods.get("score_start"))
        fold_end = pd.Timestamp(periods.get("score_end"))
        if bool(((fold_frame["date"] < fold_start) | (fold_frame["date"] > fold_end)).any()):
            raise ValueError(f"Selection PIT {fold_id} score日期超出fold期間")
        fold_dir = resolve_filter_point_in_time_fold_dir(
            PROJECT_ROOT,
            args.filter_id,
            fold_id,
            args.model_architecture,
            args.experiment_profile,
        )
        fold_manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
        checkpoint_path = fold_dir / DEFAULT_MODEL_FILENAME
        fold_score_path = fold_dir / FOLD_SCORE_FILENAME
        if build_file_manifest(fold_manifest_path) != record.get("fold_manifest"):
            raise ValueError(f"Selection PIT {fold_id} fold manifest hash不一致")
        fold_artifacts = record.get("artifacts") or {}
        if build_file_manifest(checkpoint_path) != fold_artifacts.get("checkpoint"):
            raise ValueError(f"Selection PIT {fold_id} checkpoint hash不一致")
        if build_file_manifest(fold_score_path) != fold_artifacts.get("scores"):
            raise ValueError(f"Selection PIT {fold_id} fold score hash不一致")

    coverage = manifest.get("coverage") or {}
    if int(coverage.get("scored_group_count", -1)) != len(frame):
        raise ValueError("Selection PIT scored_group_count與score rows不一致")
    if int(coverage.get("duplicate_group_count", -1)) != 0:
        raise ValueError("Selection PIT manifest宣告存在duplicate groups")
    if int(coverage.get("missing_group_count", -1)) != 0:
        raise ValueError("Selection PIT manifest宣告存在missing groups")
    if int(coverage.get("extra_group_count", -1)) != 0:
        raise ValueError("Selection PIT manifest宣告存在extra groups")
    return frame, manifest


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


def _daily_spearman(frame: pd.DataFrame) -> dict[str, Any]:
    values: list[float] = []
    eligible_days = 0
    for _date, day in frame.groupby("date", sort=True):
        if len(day) < 2:
            continue
        eligible_days += 1
        value = calculate_spearman(
            day["breakout_quality_score"].to_numpy(dtype=np.float64),
            day["target_raw_r"].to_numpy(dtype=np.float64),
        )
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return {
        "eligible_day_count": int(eligible_days),
        "valid_spearman_day_count": int(len(values)),
        "mean_daily_spearman": float(np.mean(values)) if values else None,
        "median_daily_spearman": float(np.median(values)) if values else None,
    }


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
            **_daily_spearman(frame),
            **_decile_metrics(frame),
        }
    return {
        "group_count": int(len(frame)),
        "global_spearman": calculate_spearman(
            frame["breakout_quality_score"].to_numpy(dtype=np.float64),
            frame["target_raw_r"].to_numpy(dtype=np.float64),
        ),
        **_daily_spearman(frame),
        **_decile_metrics(frame),
    }


def _yearly_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    work["year"] = work["date"].dt.year.astype(int)
    for year, year_frame in work.groupby("year", sort=True):
        metrics = _scope_metrics(year_frame)
        rows.append({"year": int(year), **metrics})
    return rows


def _fold_metrics(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                "pass_target_spearman": calculate_spearman(
                    target_frame["breakout_quality_score"].to_numpy(dtype=np.float64),
                    target_frame["target_raw_r"].to_numpy(dtype=np.float64),
                )
                if len(target_frame) >= 2
                else None,
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


def _render_markdown(payload: dict[str, Any]) -> str:
    def fmt(value, digits=4):
        return "-" if value is None else f"{float(value):.{digits}f}"

    primary = payload["metrics"]["pass_only_target"]
    all_target = payload["metrics"]["all_valid_target"]
    classification = payload["classification_overlap"]
    lines = [
        "# Selection Point-in-time Score Audit",
        "",
        f"- Architecture：`{payload['model_architecture']}`",
        f"- Experiment profile：`{payload['experiment_profile']}`",
        f"- Target：`{payload['continuous_target_id']}`",
        f"- Score period：`{payload['score_period']['start']} ~ {payload['score_period']['end']}`",
        "- Primary decision scope：PASS-only target ordering。",
        "",
        "## 1. Predictive ordering",
        "",
        "| Scope | Groups | Global Spearman | Mean daily Spearman | Top target | Bottom target | Spread |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| PASS-only | {primary['group_count']:,} | {fmt(primary['global_spearman'])} | {fmt(primary['mean_daily_spearman'])} | {fmt(primary['top_decile_target_mean'])} | {fmt(primary['bottom_decile_target_mean'])} | {fmt(primary['top_bottom_target_spread'])} |",
        f"| All valid labels | {all_target['group_count']:,} | {fmt(all_target['global_spearman'])} | {fmt(all_target['mean_daily_spearman'])} | {fmt(all_target['top_decile_target_mean'])} | {fmt(all_target['bottom_decile_target_mean'])} | {fmt(all_target['top_bottom_target_spread'])} |",
        "",
        "## 2. PASS／REJECT overlap",
        "",
        f"- Score binary AUC：{fmt(classification['score_vs_pass_reject_auc'])}",
        f"- Overall PASS share：{fmt(classification['overall_pass_share'])}",
        f"- Top score decile PASS share：{fmt(classification['top_score_decile_pass_share'])}",
        f"- PASS-only target Spearman：{fmt(primary['global_spearman'])}",
        "",
        "## 3. Yearly PASS-only target ordering",
        "",
        "| Year | Groups | Spearman | Mean daily Spearman | Top-bottom spread |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["yearly_pass_only"]:
        lines.append(
            f"| {row['year']} | {row['group_count']:,} | {fmt(row['global_spearman'])} | "
            f"{fmt(row['mean_daily_spearman'])} | {fmt(row['top_bottom_target_spread'])} |"
        )
    drift = payload["fold_drift"]
    lines.extend(
        [
            "",
            "## 4. Fold drift",
            "",
            f"- Drift flag：`{drift['drift_flag']}`",
            f"- Max adjacent mean shift：{fmt(drift['max_adjacent_mean_shift_in_pooled_std'])} pooled SD",
            f"- Flagged folds：{', '.join(drift['flagged_folds']) if drift['flagged_folds'] else '-'}",
            "",
            "## 5. Orderable candidate coverage",
            "",
        ]
    )
    orderable = payload["orderable_candidate_coverage"]
    if orderable.get("available"):
        lines.append(
            f"- {orderable['scored_candidate_count']:,}/{orderable['candidate_count']:,} "
            f"({fmt(orderable['coverage_rate'])})"
        )
    else:
        lines.append(f"- unavailable：{orderable.get('reason')}; `{orderable.get('path')}`")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
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
        ),
    }
    fold_rows, fold_drift = _fold_metrics(merged)
    orderable = _orderable_coverage(
        score_frame,
        requested_path=args.orderable_candidates,
        filter_id=args.filter_id,
        target_id=str(bundle.profile.continuous_target_id),
    )
    score_period = dict(manifest.get("score_period") or {})
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
        "metrics": {
            "pass_only_target": pass_metrics,
            "reject_only_target": reject_metrics,
            "all_valid_target": all_metrics,
        },
        "yearly_pass_only": _yearly_metrics(pass_target),
        "fold_metrics": fold_rows,
        "fold_drift": fold_drift,
        "classification_overlap": classification_overlap,
        "orderable_candidate_coverage": orderable,
        "decision_contract": {
            "primary_evidence": "PASS-only no-time target ordering",
            "actual_selected_r_is_primary_veto": False,
            "strategy_optimizer_executed": False,
            "future_target_used_for_runtime_sort": False,
        },
        "source_artifacts": {
            "point_in_time_manifest": str(
                resolve_selection_point_in_time_manifest_path(
                    PROJECT_ROOT,
                    args.filter_id,
                    args.model_architecture,
                    args.experiment_profile,
                )
            ),
            "point_in_time_scores": str(
                resolve_selection_point_in_time_score_path(
                    PROJECT_ROOT,
                    args.filter_id,
                    args.model_architecture,
                    args.experiment_profile,
                )
            ),
        },
    }
    output_json = resolve_selection_point_in_time_audit_json_path(
        PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
    )
    output_markdown = resolve_selection_point_in_time_audit_markdown_path(
        PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, payload)
    output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print("Selection point-in-time model audit完成")
    print(
        f"PASS-only groups={pass_metrics['group_count']:,} "
        f"Spearman={pass_metrics['global_spearman']} "
        f"mean_daily={pass_metrics['mean_daily_spearman']}"
    )
    print(f"已輸出: {output_markdown}")
    print(f"已輸出: {output_json}")
    return 0


__all__ = ["AUDIT_SCHEMA_VERSION", "main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
