"""Read-only frozen-score fusion audit for MR-13K / MR-13M against MR-13H economics."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from config.breakout_quality import (
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K,
)
from core.console_report import project_relative_display_path, render_key_values, render_table, render_title
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.continuous_ranker_quality import daily_top_k_metrics
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.ranking_score_store import resolve_continuous_ranker_oos_score_path
from services.breakout_quality.train_continuous_ranker import calculate_spearman, daily_rank_metrics

AUDIT_SCHEMA_VERSION = 1
_REQUIRED_SCORE_COLUMNS = ("ticker", "date", "group_index", "model_score", "target_raw_r")


def _source_profiles(definition: AuditDefinition) -> tuple[str, str, str]:
    source = dict(definition.source or {})
    mfe_profile = str(source.get("mfe_profile") or "").strip()
    low_adverse_profile = str(source.get("low_adverse_profile") or "").strip()
    economic_reference_profile = str(source.get("economic_reference_profile") or "").strip()
    if not mfe_profile or not low_adverse_profile or not economic_reference_profile:
        raise ValueError("Frozen rank fusion Audit source缺少mfe/low_adverse/economic_reference profile")
    if len({mfe_profile, low_adverse_profile, economic_reference_profile}) != 3:
        raise ValueError("Frozen rank fusion Audit三個profile必須互不相同")
    return mfe_profile, low_adverse_profile, economic_reference_profile


def _score_paths(definition: AuditDefinition, *, project_root: Path) -> dict[str, Path]:
    mfe_profile, low_adverse_profile, _reference_profile = _source_profiles(definition)
    filter_id = str((definition.source or {}).get("filter_id") or "breakout_quality_v1")
    architecture = str((definition.source or {}).get("architecture") or "inception_time_v1")
    return {
        "mfe": resolve_continuous_ranker_oos_score_path(
            project_root, filter_id, architecture, mfe_profile
        ),
        "low_adverse": resolve_continuous_ranker_oos_score_path(
            project_root, filter_id, architecture, low_adverse_profile
        ),
    }


def _score_schema_issue(path: Path, *, label: str, require_reference: bool) -> str | None:
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        return f"{label} frozen score header讀取失敗: {exc}"
    required = set(_REQUIRED_SCORE_COLUMNS)
    if require_reference:
        required.add("reference_target_raw_r")
    missing = sorted(required - columns)
    if missing:
        return f"{label} frozen score缺少欄位: {missing}"
    return None


def audit_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = _score_paths(definition, project_root=root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    issues: list[str] = []
    if not missing:
        mfe_issue = _score_schema_issue(paths["mfe"], label="MR-13K", require_reference=True)
        low_adverse_issue = _score_schema_issue(
            paths["low_adverse"], label="MR-13M", require_reference=False
        )
        issues.extend(issue for issue in (mfe_issue, low_adverse_issue) if issue)
    blocked_reasons = []
    if missing:
        blocked_reasons.append(f"缺少 frozen Forward score: {', '.join(missing)}")
    blocked_reasons.extend(issues)
    return {
        "status": "READY" if not blocked_reasons else "BLOCKED",
        "reason": "；".join(blocked_reasons),
        "source": {
            "display": (
                "MR-13K + MR-13M frozen Forward scores → "
                "MR-13K artifact內嵌MR-13H economic reference target"
            ),
            "artifacts": {
                name: project_relative_display_path(path, project_root=root)
                for name, path in paths.items()
            },
        },
    }


def _load_score_frame(path: Path, *, label: str) -> pd.DataFrame:
    frame = read_breakout_quality_csv(path)
    missing = [column for column in _REQUIRED_SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} frozen score缺少欄位: {missing}")
    columns = list(_REQUIRED_SCORE_COLUMNS)
    if "reference_target_raw_r" in frame.columns:
        columns.append("reference_target_raw_r")
    result = frame[columns].copy()
    result["ticker"] = result["ticker"].astype(str)
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["group_index"] = pd.to_numeric(result["group_index"], errors="raise").astype(np.int64)
    result["model_score"] = pd.to_numeric(result["model_score"], errors="coerce")
    result["target_raw_r"] = pd.to_numeric(result["target_raw_r"], errors="coerce")
    if "reference_target_raw_r" in result.columns:
        result["reference_target_raw_r"] = pd.to_numeric(
            result["reference_target_raw_r"], errors="coerce"
        )
    if result.duplicated(["ticker", "date", "group_index"]).any():
        raise ValueError(f"{label} frozen score存在重複ticker/date/group_index")
    return result.sort_values(["date", "ticker", "group_index"], kind="mergesort").reset_index(drop=True)


def _aligned_frame(mfe: pd.DataFrame, low_adverse: pd.DataFrame) -> pd.DataFrame:
    keys = ["ticker", "date", "group_index"]
    if "reference_target_raw_r" not in mfe.columns:
        raise ValueError("MR-13K frozen score缺少MR-13H reference_target_raw_r")
    merged = mfe.rename(
        columns={
            "model_score": "mfe_score",
            "target_raw_r": "mfe_target_r",
            "reference_target_raw_r": "economic_target_r",
        }
    ).merge(
        low_adverse.rename(
            columns={"model_score": "low_adverse_score", "target_raw_r": "low_adverse_target_r"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    finite = np.ones(len(merged), dtype=bool)
    for column in (
        "mfe_score",
        "low_adverse_score",
        "mfe_target_r",
        "low_adverse_target_r",
        "economic_target_r",
    ):
        finite &= np.isfinite(merged[column].to_numpy(dtype=np.float64))
    merged = merged.loc[finite].copy().reset_index(drop=True)
    if len(merged) < 2:
        raise ValueError("Frozen rank fusion Audit共同OOS evaluable rows不足")
    return merged


def _decile_means(scores: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    count = max(1, int(math.ceil(len(order) * 0.10)))
    raw = np.asarray(target, dtype=np.float64)
    bottom = float(raw[order[:count]].mean())
    top = float(raw[order[-count:]].mean())
    return top, bottom, float(top - bottom)


def _evaluate_score(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    dates = frame["date"].to_numpy()
    scores = frame[score_column].to_numpy(dtype=np.float64)
    target = frame["economic_target_r"].to_numpy(dtype=np.float64)
    target_pct = build_same_date_percentile_targets(
        target,
        np.ones(len(frame), dtype=bool),
        frame["date"],
    )
    daily = daily_rank_metrics(dates, scores, target)
    top, bottom, spread = _decile_means(scores, target)
    top_k = daily_top_k_metrics(
        dates,
        scores,
        target,
        target_pct,
        top_k=int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K),
        boundary_width=int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH),
    )
    return {
        "global_spearman": calculate_spearman(scores, target),
        "mean_daily_spearman": daily.get("mean_daily_spearman"),
        "pairwise_concordance": daily.get("pairwise_concordance"),
        "top_decile_target_mean": top,
        "bottom_decile_target_mean": bottom,
        "top_bottom_target_spread": spread,
        "top_k_lift": top_k.get("top_k_raw_target_lift"),
        "boundary_concordance": top_k.get("boundary_concordance"),
        "boundary_gap": top_k.get("boundary_raw_target_gap"),
        "competition_date_count": top_k.get("competition_date_count"),
    }


def compute_frozen_rank_fusion(mfe: pd.DataFrame, low_adverse: pd.DataFrame) -> dict[str, Any]:
    """Compute fixed equal-rank frozen expert fusion without fitting any parameter."""

    frame = _aligned_frame(mfe, low_adverse)
    valid = np.ones(len(frame), dtype=bool)
    frame["mfe_score_pct"] = build_same_date_percentile_targets(
        frame["mfe_score"].to_numpy(dtype=np.float64), valid, frame["date"]
    )
    frame["low_adverse_score_pct"] = build_same_date_percentile_targets(
        frame["low_adverse_score"].to_numpy(dtype=np.float64), valid, frame["date"]
    )
    frame["fused_score"] = 0.5 * frame["mfe_score_pct"] + 0.5 * frame["low_adverse_score_pct"]

    frame["mfe_target_pct"] = build_same_date_percentile_targets(
        frame["mfe_target_r"].to_numpy(dtype=np.float64), valid, frame["date"]
    )
    frame["low_adverse_target_pct"] = build_same_date_percentile_targets(
        frame["low_adverse_target_r"].to_numpy(dtype=np.float64), valid, frame["date"]
    )
    frame["equal_rank_target"] = 0.5 * frame["mfe_target_pct"] + 0.5 * frame["low_adverse_target_pct"]

    true_component_relation = daily_rank_metrics(
        frame["date"].to_numpy(),
        frame["mfe_target_r"].to_numpy(dtype=np.float64),
        frame["low_adverse_target_r"].to_numpy(dtype=np.float64),
    )
    frozen_score_relation = daily_rank_metrics(
        frame["date"].to_numpy(),
        frame["mfe_score"].to_numpy(dtype=np.float64),
        frame["low_adverse_score"].to_numpy(dtype=np.float64),
    )
    equal_rank_vs_economic = daily_rank_metrics(
        frame["date"].to_numpy(),
        frame["equal_rank_target"].to_numpy(dtype=np.float64),
        frame["economic_target_r"].to_numpy(dtype=np.float64),
    )

    metrics = {
        "mfe_mr13k": _evaluate_score(frame, "mfe_score"),
        "low_adverse_mr13m": _evaluate_score(frame, "low_adverse_score"),
        "fused_equal_rank": _evaluate_score(frame, "fused_score"),
    }
    fused = metrics["fused_equal_rank"]
    primary_keys = ("mean_daily_spearman", "pairwise_concordance", "top_k_lift")
    primary_go = True
    best_single: dict[str, float] = {}
    for key in primary_keys:
        single_values = [
            metrics["mfe_mr13k"].get(key),
            metrics["low_adverse_mr13m"].get(key),
        ]
        if fused.get(key) is None or any(value is None for value in single_values):
            primary_go = False
            continue
        best_single[key] = max(float(value) for value in single_values)
        if not float(fused[key]) > best_single[key]:
            primary_go = False
    return {
        "row_count": int(len(frame)),
        "date_start": str(frame["date"].min().date()),
        "date_end": str(frame["date"].max().date()),
        "true_component_relation": {
            "global_spearman": calculate_spearman(
                frame["mfe_target_r"].to_numpy(dtype=np.float64),
                frame["low_adverse_target_r"].to_numpy(dtype=np.float64),
            ),
            **true_component_relation,
        },
        "frozen_score_relation": {
            "global_spearman": calculate_spearman(
                frame["mfe_score"].to_numpy(dtype=np.float64),
                frame["low_adverse_score"].to_numpy(dtype=np.float64),
            ),
            **frozen_score_relation,
        },
        "equal_rank_target_vs_economic": {
            "global_spearman": calculate_spearman(
                frame["equal_rank_target"].to_numpy(dtype=np.float64),
                frame["economic_target_r"].to_numpy(dtype=np.float64),
            ),
            **equal_rank_vs_economic,
            "equal_rank_target_std": float(frame["equal_rank_target"].std(ddof=1)),
        },
        "metrics": metrics,
        "decision": {
            "status": "GO_MULTI_EXPERT_FUSION" if primary_go else "REJECT_EQUAL_RANK_SCORE_FUSION",
            "rule": "fused Forward economic Daily rho, Pair and Top-K Lift must all strictly exceed the better MR-13K/MR-13M single expert on each metric",
            "best_single_expert_primary_metrics": best_single,
            "used_for_training_or_epoch_selection": False,
            "weights_fitted": False,
            "weights": {"mfe_score_percentile": 0.5, "low_adverse_score_percentile": 0.5},
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.2f}%"


def _markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    rows = []
    for label, key in (
        ("MR-13K Pure-MFE", "mfe_mr13k"),
        ("MR-13M Low-Adverse", "low_adverse_mr13m"),
        ("Equal-rank frozen fusion", "fused_equal_rank"),
    ):
        row = metrics[key]
        rows.append(
            f"| {label} | {_fmt(row['mean_daily_spearman'])} | {_fmt_pct(row['pairwise_concordance'])} | "
            f"{_fmt(row['top_k_lift'])} | {_fmt(row['top_bottom_target_spread'])} | {_fmt(row['boundary_gap'])} |"
        )
    relation = payload["true_component_relation"]
    score_relation = payload["frozen_score_relation"]
    equal_rank = payload["equal_rank_target_vs_economic"]
    return "\n".join(
        [
            "# MR-13K / MR-13M Frozen Rank Fusion Audit",
            "",
            f"- Status：`{payload['decision']['status']}`",
            f"- Period：`{payload['date_start']} ～ {payload['date_end']}`",
            f"- Common evaluable rows：`{payload['row_count']:,}`",
            "- Contract：只讀frozen Forward scores；兩個score各自同日percentile後固定0.5/0.5，不fit權重、不重訓。",
            "",
            "## Economic reference quality",
            "",
            "| Score | Daily rho | Pair | Top-K Lift | Top-Bottom R | Boundary gap |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Component geometry",
            "",
            f"- True MFE vs Low-Adverse：Global rho=`{_fmt(relation['global_spearman'])}`；Daily rho=`{_fmt(relation['mean_daily_spearman'])}`；Pair=`{_fmt_pct(relation['pairwise_concordance'])}`。",
            f"- Frozen MR-13K score vs MR-13M score：Global rho=`{_fmt(score_relation['global_spearman'])}`；Daily rho=`{_fmt(score_relation['mean_daily_spearman'])}`；Pair=`{_fmt_pct(score_relation['pairwise_concordance'])}`。",
            f"- True equal-rank Target vs MR-13H economic Target：Global rho=`{_fmt(equal_rank['global_spearman'])}`；Daily rho=`{_fmt(equal_rank['mean_daily_spearman'])}`；Pair=`{_fmt_pct(equal_rank['pairwise_concordance'])}`；Target std=`{_fmt(equal_rank['equal_rank_target_std'])}`。",
            "",
            "## Decision contract",
            "",
            f"`{payload['decision']['rule']}`",
            "",
        ]
    )


def run_audit(
    definition: AuditDefinition,
    *,
    project_root: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status = audit_status(definition, project_root=root)
    if status["status"] != "READY":
        raise RuntimeError(status["reason"])
    paths = _score_paths(definition, project_root=root)
    payload = compute_frozen_rank_fusion(
        _load_score_frame(paths["mfe"], label="MR-13K"),
        _load_score_frame(paths["low_adverse"], label="MR-13M"),
    )
    payload.update(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_id": definition.audit_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": status["source"],
        }
    )
    output_dir = (
        root
        / Path(AUDIT_OUTPUT_ROOT)
        / Path(definition.output_subdir)
        / "latest"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")

    if not quiet:
        metrics = payload["metrics"]
        table_rows = []
        for label, key in (
            ("MR-13K", "mfe_mr13k"),
            ("MR-13M", "low_adverse_mr13m"),
            ("K+M 50/50", "fused_equal_rank"),
        ):
            row = metrics[key]
            table_rows.append(
                (
                    label,
                    _fmt(row.get("mean_daily_spearman")),
                    _fmt_pct(row.get("pairwise_concordance")),
                    _fmt(row.get("top_k_lift")),
                    _fmt(row.get("top_bottom_target_spread")),
                )
            )
        print(
            "\n\n".join(
                (
                    render_title("MR-13K / MR-13M Frozen Rank Fusion Audit"),
                    render_key_values(
                        (
                            ("狀態", payload["decision"]["status"]),
                            ("期間", f"{payload['date_start']}～{payload['date_end']}"),
                            ("Common rows", f"{payload['row_count']:,}"),
                            ("權重", "MFE rank 0.5 + Low-Adverse rank 0.5（固定）"),
                            ("訓練", "無；只讀既有frozen Forward scores"),
                        )
                    ),
                    render_table(
                        ("Score", "Daily rho", "Pair", "Top-K Lift", "Top-Bottom R"),
                        table_rows,
                    ),
                    render_key_values(
                        (
                            (
                                "True component Daily rho",
                                _fmt(payload["true_component_relation"].get("mean_daily_spearman")),
                            ),
                            (
                                "Frozen score Daily rho",
                                _fmt(payload["frozen_score_relation"].get("mean_daily_spearman")),
                            ),
                            (
                                "Equal-rank Target → economic Daily rho",
                                _fmt(payload["equal_rank_target_vs_economic"].get("mean_daily_spearman")),
                            ),
                            (
                                "Markdown",
                                project_relative_display_path(markdown_path, project_root=root),
                            ),
                        )
                    ),
                )
            )
        )
    return payload


__all__ = ["audit_status", "compute_frozen_rank_fusion", "run_audit"]
