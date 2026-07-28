"""Audit breakout-event coverage and model quality across market regimes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.breakout_quality_experiments import SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
from config.breakout_quality_policy import BREAKOUT_QUALITY_EXPERIMENT_PROFILE
from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    FEATURE_COLUMNS,
    LABEL_PASS,
    LABEL_REJECT,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_SELECTION,
    SCORE_COLUMN,
)
from filters.breakout_quality.paths import ensure_filter_report_dir
from filters.breakout_quality.splits import KEY_COLUMNS, normalize_split_assignment_keys
from tools.filters.breakout_quality.common import load_validated_dataset_bundle
from tools.filters.breakout_quality.evaluate import (
    GROUP_SCORE_NUMERICAL_NOISE_ATOL,
    evaluate_frame_metrics,
    prepare_evaluation_context,
)

REGIME_AUDIT_SCHEMA_VERSION = 1
ANNUALIZATION_BARS = 252
VOLATILITY_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
TREND_STATES = ("bull", "transition", "bear")
DRAWDOWN_STATES = ("near_high", "correction", "deep_drawdown")
VOLATILITY_STATES = ("low", "medium", "high")
REGIME_DIMENSIONS = ("trend_state", "drawdown_state", "volatility_state", "combined_regime")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "分析 Selection 與 OOS 的 breakout event 市場狀態覆蓋；"
            "所有 regime 只使用事件日及以前的 0050 序列，OOS 僅作診斷不得回頭調參"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--experiment-profile",
        choices=SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
        default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    )
    parser.add_argument("--score-path", default=None, help="自訂 research_scores.csv；通常不需指定")
    parser.add_argument(
        "--min-selection-groups",
        type=int,
        default=100,
        help="combined regime 在 Selection 少於此 group 數時標記為 low support",
    )
    parser.add_argument(
        "--min-support-share-ratio",
        type=float,
        default=0.5,
        help="Selection share / OOS share 低於此值時標記為 underrepresented",
    )
    return parser.parse_args(argv)


def _validate_audit_args(*, min_selection_groups: int, min_support_share_ratio: float) -> None:
    if int(min_selection_groups) < 1:
        raise ValueError("min_selection_groups 必須 >= 1")
    ratio = float(min_support_share_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("min_support_share_ratio 必須是有限正數")


def derive_benchmark_regime_features(
    feature_bank: np.ndarray,
    group_indices: np.ndarray,
) -> pd.DataFrame:
    """Derive event-date observable 0050 regime features for unique feature groups."""

    bank = np.asarray(feature_bank)
    indices = np.asarray(group_indices, dtype=np.int64)
    if bank.ndim != 3:
        raise ValueError(f"feature_bank 必須是 3D: {bank.shape}")
    if indices.ndim != 1:
        raise ValueError(f"group_indices 必須是 1D: {indices.shape}")
    if bank.shape[1] < 253:
        raise ValueError("regime audit 需要至少 253 bars 的 feature window")
    if bank.shape[2] != len(FEATURE_COLUMNS):
        raise ValueError("feature_bank feature 維度與正式 contract 不一致")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(bank)):
        raise ValueError("group_indices 超出 feature_bank 範圍")

    benchmark_close_index = FEATURE_COLUMNS.index("benchmark_close_norm")
    relative_close = (
        np.asarray(bank[indices, :, benchmark_close_index], dtype=np.float64) + 1.0
    )
    if relative_close.size == 0:
        return pd.DataFrame(
            columns=[
                "group_index",
                "benchmark_return_20",
                "benchmark_return_60",
                "benchmark_return_120",
                "benchmark_volatility_20",
                "benchmark_volatility_60",
                "benchmark_close_to_sma_200",
                "benchmark_drawdown_252",
            ]
        )
    if not np.isfinite(relative_close).all() or np.any(relative_close <= 0.0):
        raise ValueError("benchmark close feature 無法還原為有限正價格")

    log_close = np.log(relative_close)
    daily_log_return = np.diff(log_close, axis=1)

    def _return(lookback: int) -> np.ndarray:
        return relative_close[:, -1] / relative_close[:, -(int(lookback) + 1)] - 1.0

    def _volatility(lookback: int) -> np.ndarray:
        window = daily_log_return[:, -int(lookback):]
        return np.std(window, axis=1, ddof=1) * math.sqrt(ANNUALIZATION_BARS)

    sma_200 = np.mean(relative_close[:, -200:], axis=1)
    high_252 = np.max(relative_close[:, -252:], axis=1)
    result = pd.DataFrame(
        {
            "group_index": indices,
            "benchmark_return_20": _return(20),
            "benchmark_return_60": _return(60),
            "benchmark_return_120": _return(120),
            "benchmark_volatility_20": _volatility(20),
            "benchmark_volatility_60": _volatility(60),
            "benchmark_close_to_sma_200": relative_close[:, -1] / sma_200 - 1.0,
            "benchmark_drawdown_252": relative_close[:, -1] / high_252 - 1.0,
        }
    )
    numeric = result.drop(columns=["group_index"]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("regime audit 衍生值含 NaN 或 infinite")
    return result


def assign_market_regimes(group_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Assign semantic trend/drawdown states and Selection-defined volatility bins."""

    required = {
        "outer_split",
        "benchmark_return_60",
        "benchmark_volatility_20",
        "benchmark_close_to_sma_200",
        "benchmark_drawdown_252",
    }
    missing = sorted(required - set(group_frame.columns))
    if missing:
        raise ValueError(f"regime assignment 缺少欄位: {missing}")

    frame = group_frame.copy()
    selection_vol = frame.loc[
        frame["outer_split"] == OUTER_SPLIT_SELECTION,
        "benchmark_volatility_20",
    ].to_numpy(dtype=np.float64)
    if selection_vol.size == 0:
        raise ValueError("regime audit 找不到 Selection group")
    low_cut, high_cut = np.quantile(selection_vol, VOLATILITY_QUANTILES)
    if not (math.isfinite(float(low_cut)) and math.isfinite(float(high_cut))):
        raise ValueError("Selection volatility quantile 無效")

    above_sma = frame["benchmark_close_to_sma_200"].to_numpy(dtype=np.float64) >= 0.0
    positive_60 = frame["benchmark_return_60"].to_numpy(dtype=np.float64) >= 0.0
    frame["trend_state"] = np.select(
        [above_sma & positive_60, (~above_sma) & (~positive_60)],
        ["bull", "bear"],
        default="transition",
    )

    drawdown = frame["benchmark_drawdown_252"].to_numpy(dtype=np.float64)
    frame["drawdown_state"] = np.select(
        [drawdown >= -0.10, drawdown >= -0.20],
        ["near_high", "correction"],
        default="deep_drawdown",
    )

    volatility = frame["benchmark_volatility_20"].to_numpy(dtype=np.float64)
    frame["volatility_state"] = np.select(
        [volatility <= float(low_cut), volatility <= float(high_cut)],
        ["low", "medium"],
        default="high",
    )
    frame["combined_regime"] = (
        frame["trend_state"].astype(str)
        + "|"
        + frame["drawdown_state"].astype(str)
        + "|"
        + frame["volatility_state"].astype(str)
    )
    thresholds = {
        "volatility_feature": "benchmark_volatility_20",
        "volatility_quantile_source": "selection_only",
        "volatility_quantiles": [round(float(value), 8) for value in VOLATILITY_QUANTILES],
        "volatility_low_max": round(float(low_cut), 8),
        "volatility_medium_max": round(float(high_cut), 8),
        "trend_definition": {
            "bull": "close_to_sma_200 >= 0 and return_60 >= 0",
            "bear": "close_to_sma_200 < 0 and return_60 < 0",
            "transition": "mixed signs",
        },
        "drawdown_definition": {
            "near_high": "drawdown_252 >= -0.10",
            "correction": "-0.20 <= drawdown_252 < -0.10",
            "deep_drawdown": "drawdown_252 < -0.20",
        },
    }
    return frame, thresholds


def _collapse_to_group_frame(
    merged_events: pd.DataFrame,
    *,
    require_identical_group_scores: bool,
) -> tuple[pd.DataFrame, dict]:
    required = {
        "ticker",
        "date",
        "high_len",
        "group_index",
        "label",
        "outer_split",
        SCORE_COLUMN,
    }
    missing = sorted(required - set(merged_events.columns))
    if missing:
        raise ValueError(f"regime audit event table 缺少欄位: {missing}")

    valid = merged_events[merged_events["label"].isin([LABEL_PASS, LABEL_REJECT])].copy()
    allowed_span = 0.0 if require_identical_group_scores else GROUP_SCORE_NUMERICAL_NOISE_ATOL
    rows: list[pd.Series] = []
    multirow = 0
    nonidentical = 0
    max_span = 0.0
    for (_ticker, _date), group in valid.groupby(["ticker", "date"], sort=False):
        for column in ("group_index", "label", "outer_split"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"同一 ticker/date 的 {column} 不一致")
        scores = pd.to_numeric(group[SCORE_COLUMN], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("regime audit score 含 NaN 或 infinite")
        span = float(scores.max() - scores.min())
        if span > allowed_span:
            raise ValueError(
                "同一 ticker/date 的 regime audit score 違反 group 契約: "
                f"span={span:.10f}, allowed={allowed_span:.10f}"
            )
        if len(group) > 1:
            multirow += 1
        if span > 0.0:
            nonidentical += 1
            max_span = max(max_span, span)
        rows.append(group.iloc[0])
    if not rows:
        raise ValueError("regime audit 沒有有效 PASS／REJECT group")
    frame = pd.DataFrame(rows).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    diagnostics = {
        "group_score_reduction": (
            "single_unique_group_inference_broadcast"
            if require_identical_group_scores
            else "first_event_row_with_bounded_numerical_noise"
        ),
        "group_score_numerical_noise_atol": allowed_span,
        "multirow_group_count": int(multirow),
        "nonidentical_score_group_count": int(nonidentical),
        "max_within_group_score_span": round(float(max_span), 10),
    }
    return frame, diagnostics


def _metric_subset(frame: pd.DataFrame, threshold: float) -> dict:
    if frame.empty:
        return {"group_count": 0}
    metrics = evaluate_frame_metrics(
        frame,
        threshold=float(threshold),
        group_weighted=False,
        require_identical_group_scores=False,
    )
    ranking = metrics.get("ranking_and_calibration") or {}
    return {
        "group_count": int(metrics.get("group_count", 0)),
        "base_pass_rate": metrics.get("base_pass_rate"),
        "model_pass_rate": metrics.get("acceptance_rate"),
        "pass_precision": metrics.get("pass_precision"),
        "pass_recall": metrics.get("pass_recall"),
        "accuracy": metrics.get("accuracy"),
        "average_score": metrics.get("avg_score"),
        "pr_auc": ranking.get("average_precision_pr_auc"),
        "precision_at_50pct_coverage": (ranking.get("precision_at_coverage") or {}).get("0.50"),
        "precision_at_60pct_coverage": (ranking.get("precision_at_coverage") or {}).get("0.60"),
        "precision_at_70pct_coverage": (ranking.get("precision_at_coverage") or {}).get("0.70"),
    }


def build_regime_audit_payload(
    group_frame: pd.DataFrame,
    *,
    threshold: float,
    min_selection_groups: int,
    min_support_share_ratio: float,
    regime_thresholds: dict,
    score_group_diagnostics: dict,
    metadata: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _validate_audit_args(
        min_selection_groups=min_selection_groups,
        min_support_share_ratio=min_support_share_ratio,
    )
    frame = group_frame.copy()
    selection = frame[frame["outer_split"] == OUTER_SPLIT_SELECTION].copy()
    oos = frame[frame["outer_split"] == OUTER_SPLIT_OOS].copy()
    if selection.empty or oos.empty:
        raise ValueError("regime audit 需要同時存在 Selection 與 OOS group")

    dimension_rows: list[dict] = []
    dimension_payload: dict[str, list[dict]] = {}
    for dimension in REGIME_DIMENSIONS:
        states = sorted(set(frame[dimension].astype(str)))
        records: list[dict] = []
        selection_total = float(len(selection))
        oos_total = float(len(oos))
        for state in states:
            selection_slice = selection[selection[dimension] == state]
            oos_slice = oos[oos[dimension] == state]
            selection_share = len(selection_slice) / selection_total
            oos_share = len(oos_slice) / oos_total
            share_ratio = selection_share / oos_share if oos_share > 0.0 else None
            record = {
                "dimension": dimension,
                "state": state,
                "selection_share": round(float(selection_share), 6),
                "oos_share": round(float(oos_share), 6),
                "selection_to_oos_share_ratio": (
                    round(float(share_ratio), 6) if share_ratio is not None else None
                ),
                "selection": _metric_subset(selection_slice, threshold),
                "oos": _metric_subset(oos_slice, threshold),
            }
            records.append(record)
            flat = {
                "dimension": dimension,
                "state": state,
                "selection_share": record["selection_share"],
                "oos_share": record["oos_share"],
                "selection_to_oos_share_ratio": record["selection_to_oos_share_ratio"],
            }
            for split_name in ("selection", "oos"):
                for key, value in record[split_name].items():
                    flat[f"{split_name}_{key}"] = value
            dimension_rows.append(flat)
        dimension_payload[dimension] = records

    combined = dimension_payload["combined_regime"]
    support_by_regime = {
        str(record["state"]): {
            "selection_groups": int(record["selection"].get("group_count", 0)),
            "selection_to_oos_share_ratio": record["selection_to_oos_share_ratio"],
        }
        for record in combined
    }
    frame["selection_support_groups"] = frame["combined_regime"].map(
        lambda value: support_by_regime[str(value)]["selection_groups"]
    )
    frame["selection_to_oos_share_ratio"] = frame["combined_regime"].map(
        lambda value: support_by_regime[str(value)]["selection_to_oos_share_ratio"]
    )
    frame["low_selection_support"] = (
        frame["selection_support_groups"] < int(min_selection_groups)
    )
    ratio_values = pd.to_numeric(frame["selection_to_oos_share_ratio"], errors="coerce")
    frame["underrepresented_in_selection"] = (
        ratio_values < float(min_support_share_ratio)
    ).fillna(False)

    yearly: list[dict] = []
    for year, year_frame in oos.groupby(oos["date"].dt.year, sort=True):
        year_enriched = frame.loc[year_frame.index]
        metrics = _metric_subset(year_enriched, threshold)
        metrics.update(
            {
                "year": int(year),
                "low_selection_support_rate": round(
                    float(year_enriched["low_selection_support"].mean()), 6
                ),
                "underrepresented_in_selection_rate": round(
                    float(year_enriched["underrepresented_in_selection"].mean()), 6
                ),
                "bear_event_rate": round(
                    float((year_enriched["trend_state"] == "bear").mean()), 6
                ),
                "deep_drawdown_event_rate": round(
                    float((year_enriched["drawdown_state"] == "deep_drawdown").mean()), 6
                ),
                "high_volatility_event_rate": round(
                    float((year_enriched["volatility_state"] == "high").mean()), 6
                ),
            }
        )
        yearly.append(metrics)

    payload = {
        "schema_version": REGIME_AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "fixed_threshold": float(threshold),
        "diagnostic_only": True,
        "oos_tuning_allowed": False,
        "regime_feature_contract": {
            "information_cutoff": "event_date_close_or_earlier",
            "source": "benchmark columns in canonical 300-bar feature bank",
            "annualization_bars": ANNUALIZATION_BARS,
            "thresholds": regime_thresholds,
        },
        "support_policy": {
            "min_selection_groups": int(min_selection_groups),
            "min_support_share_ratio": float(min_support_share_ratio),
            "share_ratio_formula": "selection regime share / OOS regime share",
        },
        "score_group_contract": score_group_diagnostics,
        "overall": {
            "selection": _metric_subset(selection, threshold),
            "oos": _metric_subset(oos, threshold),
        },
        "dimensions": dimension_payload,
        "oos_yearly": yearly,
    }
    dimension_frame = pd.DataFrame(dimension_rows)
    return payload, dimension_frame, frame


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _decimal(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def render_regime_audit_markdown(payload: dict) -> str:
    metadata = payload["metadata"]
    thresholds = payload["regime_feature_contract"]["thresholds"]
    lines = [
        "# Breakout Quality Regime Coverage Audit",
        "",
        f"- Filter：`{metadata['filter_id']}`",
        f"- Architecture／Profile：`{metadata['model_architecture']}` / `{metadata['experiment_profile']}`",
        f"- 固定 threshold：`{payload['fixed_threshold']}`",
        "- 用途：診斷 Selection 是否涵蓋 OOS 的市場狀態；不得依 OOS 結果回頭調整模型、Label、threshold 或 regime 開關。",
        "",
        "## 1. Regime 契約",
        "",
        "- 所有市場狀態只使用事件日收盤及以前的 canonical 0050 300-bar sequence。",
        "- Trend：0050 相對 200 日均線與 60 日報酬同正為 bull、同負為 bear，其餘為 transition。",
        "- Drawdown：距 252 日高點小於 10% 為 near_high、10%～20% 為 correction、超過 20% 為 deep_drawdown。",
        "- Volatility：只用 Selection 的 20 日年化波動率三分位數切 low／medium／high。",
        f"- Selection volatility cutoffs：low ≤ `{thresholds['volatility_low_max']:.6f}`；medium ≤ `{thresholds['volatility_medium_max']:.6f}`。",
        "",
        "## 2. OOS 年度覆蓋與模型品質",
        "",
        "| 年度 | Groups | 原始 PASS | 模型 PASS | Precision | Recall | PR-AUC | Bear事件 | 深回撤事件 | Low support | Selection低代表性 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["oos_yearly"]:
        lines.append(
            "| {year} | {groups:,} | {base} | {model} | {precision} | {recall} | {pr_auc} | {bear} | {deep} | {low} | {under} |".format(
                year=int(row["year"]),
                groups=int(row.get("group_count", 0)),
                base=_pct(row.get("base_pass_rate")),
                model=_pct(row.get("model_pass_rate")),
                precision=_pct(row.get("pass_precision")),
                recall=_pct(row.get("pass_recall")),
                pr_auc=_decimal(row.get("pr_auc")),
                bear=_pct(row.get("bear_event_rate")),
                deep=_pct(row.get("deep_drawdown_event_rate")),
                low=_pct(row.get("low_selection_support_rate")),
                under=_pct(row.get("underrepresented_in_selection_rate")),
            )
        )

    section_names = {
        "trend_state": "Trend state",
        "drawdown_state": "Drawdown state",
        "volatility_state": "Volatility state",
        "combined_regime": "Combined regime",
    }
    section_number = 3
    for dimension in REGIME_DIMENSIONS:
        lines.extend(
            [
                "",
                f"## {section_number}. {section_names[dimension]}",
                "",
                "| State | Selection Groups | OOS Groups | Sel Share | OOS Share | Sel/OOS Share | Sel PASS | OOS PASS | OOS Precision | OOS Recall | OOS PR-AUC |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        records = sorted(
            payload["dimensions"][dimension],
            key=lambda row: int(row["oos"].get("group_count", 0)),
            reverse=True,
        )
        for row in records:
            lines.append(
                "| {state} | {sg:,} | {og:,} | {ss} | {os} | {ratio} | {sp} | {op} | {precision} | {recall} | {pr_auc} |".format(
                    state=row["state"],
                    sg=int(row["selection"].get("group_count", 0)),
                    og=int(row["oos"].get("group_count", 0)),
                    ss=_pct(row.get("selection_share")),
                    os=_pct(row.get("oos_share")),
                    ratio=_decimal(row.get("selection_to_oos_share_ratio"), 3),
                    sp=_pct(row["selection"].get("base_pass_rate")),
                    op=_pct(row["oos"].get("base_pass_rate")),
                    precision=_pct(row["oos"].get("pass_precision")),
                    recall=_pct(row["oos"].get("pass_recall")),
                    pr_auc=_decimal(row["oos"].get("pr_auc")),
                )
            )
        section_number += 1

    lines.extend(
        [
            "",
            "## 判讀限制",
            "",
            "- `low support` 只表示同一 combined regime 的 Selection group 數低於命令列門檻，不等於該年度必然失效。",
            "- `Selection低代表性` 表示該 regime 在 Selection 的占比，相對 OOS 占比低於指定比例；它是支撐缺口證據，不是可部署的年度／regime gate。",
            "- 若 2022 同時呈現高 bear／deep-drawdown 占比、高 low-support，以及低 PR-AUC，才支持『熊市 breakout 訓練支撐不足』的假說。",
            "",
        ]
    )
    return "\n".join(lines)


def run_regime_audit(
    *,
    filter_id: str,
    experiment_profile: str,
    score_path: str | Path | None = None,
    min_selection_groups: int = 100,
    min_support_share_ratio: float = 0.5,
) -> tuple[dict, dict[str, Path]]:
    _validate_audit_args(
        min_selection_groups=min_selection_groups,
        min_support_share_ratio=min_support_share_ratio,
    )
    evaluation_context = prepare_evaluation_context(
        filter_id=str(filter_id),
        experiment_profile=str(experiment_profile),
        score_path=score_path,
    )
    model_manifest = evaluation_context["model_manifest"]
    policy = model_manifest.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("model manifest 缺少 policy")
    dataset_summary, indexed_features, _context, _labels, events = load_validated_dataset_bundle(
        str(filter_id),
        expected_policy=policy,
        require_current_source=False,
    )

    event_keys = normalize_split_assignment_keys(
        events[["ticker", "date", "high_len", "group_index"]].copy()
    )
    merged = evaluation_context["merged"].merge(
        event_keys,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if merged["group_index"].isna().any():
        raise ValueError("research score 有事件找不到 dataset group_index")
    merged["group_index"] = pd.to_numeric(merged["group_index"], errors="raise").astype(int)

    model_spec = model_manifest.get("model_spec")
    if not isinstance(model_spec, dict):
        raise ValueError("model manifest 缺少 model_spec")
    require_identical = not bool(model_spec.get("use_dataset_context", True))
    group_frame, score_group_diagnostics = _collapse_to_group_frame(
        merged,
        require_identical_group_scores=require_identical,
    )

    unique_indices = group_frame["group_index"].to_numpy(dtype=np.int64)
    regime_features = derive_benchmark_regime_features(
        indexed_features.feature_bank,
        unique_indices,
    )
    group_frame = group_frame.merge(
        regime_features,
        on="group_index",
        how="left",
        validate="one_to_one",
    )
    if group_frame["benchmark_return_60"].isna().any():
        raise ValueError("regime features 未完整覆蓋 score groups")
    group_frame, regime_thresholds = assign_market_regimes(group_frame)

    metadata = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_manifest.get("model_architecture") or ""),
        "experiment_profile": str(experiment_profile),
        "score_path": str(evaluation_context["score_path"]),
        "dataset_profile": str(dataset_summary.get("dataset") or ""),
        "dataset_feature_group_count": int(dataset_summary.get("feature_group_count", 0)),
        "audited_group_count": int(len(group_frame)),
    }
    payload, dimension_frame, enriched_groups = build_regime_audit_payload(
        group_frame,
        threshold=float(evaluation_context["threshold"]),
        min_selection_groups=int(min_selection_groups),
        min_support_share_ratio=float(min_support_share_ratio),
        regime_thresholds=regime_thresholds,
        score_group_diagnostics=score_group_diagnostics,
        metadata=metadata,
    )

    report_dir = ensure_filter_report_dir(
        PROJECT_ROOT,
        str(filter_id),
        experiment_profile=str(experiment_profile),
    )
    paths = {
        "markdown": report_dir / "regime_coverage_audit.md",
        "json": report_dir / "regime_coverage_audit.json",
        "cells_csv": report_dir / "regime_coverage_cells.csv",
        "groups_csv": report_dir / "regime_event_groups.csv",
    }
    paths["markdown"].write_text(render_regime_audit_markdown(payload), encoding="utf-8")
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    dimension_frame.to_csv(paths["cells_csv"], index=False, encoding="utf-8-sig")
    output_columns = [
        "ticker",
        "date",
        "group_index",
        "outer_split",
        "selection_role",
        "label",
        SCORE_COLUMN,
        "benchmark_return_20",
        "benchmark_return_60",
        "benchmark_return_120",
        "benchmark_volatility_20",
        "benchmark_volatility_60",
        "benchmark_close_to_sma_200",
        "benchmark_drawdown_252",
        "trend_state",
        "drawdown_state",
        "volatility_state",
        "combined_regime",
        "selection_support_groups",
        "selection_to_oos_share_ratio",
        "low_selection_support",
        "underrepresented_in_selection",
    ]
    enriched_groups[output_columns].sort_values(
        ["date", "ticker"], kind="mergesort"
    ).to_csv(paths["groups_csv"], index=False, encoding="utf-8-sig")
    return payload, paths


def main(argv=None) -> int:
    args = parse_args(argv)
    payload, paths = run_regime_audit(
        filter_id=str(args.filter_id),
        experiment_profile=str(args.experiment_profile),
        score_path=args.score_path,
        min_selection_groups=int(args.min_selection_groups),
        min_support_share_ratio=float(args.min_support_share_ratio),
    )
    print(render_regime_audit_markdown(payload))
    print("已輸出：")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    return 0


__all__ = [
    "REGIME_AUDIT_SCHEMA_VERSION",
    "assign_market_regimes",
    "build_regime_audit_payload",
    "derive_benchmark_regime_features",
    "render_regime_audit_markdown",
    "run_regime_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())
