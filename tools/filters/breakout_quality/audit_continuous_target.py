"""Build and audit the fixed 11A strategy-aligned continuous target."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_target import (
    CONTINUOUS_TARGET_SCHEMA_VERSION,
    STRATEGY_ALIGNED_TARGET_ID,
    TARGET_ADVERSE_RETURN_FILENAME,
    TARGET_AUDIT_JSON_FILENAME,
    TARGET_AUDIT_MARKDOWN_FILENAME,
    TARGET_DAILY_CSV_FILENAME,
    TARGET_FAVORABLE_RETURN_FILENAME,
    TARGET_MANIFEST_FILENAME,
    TARGET_OPPORTUNITY_BAR_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_RISK_BREACH_BAR_FILENAME,
    TARGET_TRADE_MATCHES_CSV_FILENAME,
    TARGET_VALID_MASK_FILENAME,
    StrategyAlignedContinuousTargetSpec,
    build_strategy_aligned_group_targets,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    DEFAULT_LABEL_POLICY,
    LABEL_PASS,
    LABEL_REJECT,
)
from filters.breakout_quality.dataset_store import load_npy, save_npy_atomic
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    resolve_breakout_quality_outer_policy,
)
from tools.filters.breakout_quality.common import (
    dataset_paths,
    load_validated_dataset_bundle,
)

TARGET_AUDIT_SCHEMA_VERSION = 1
_SPLIT_ORDER = ("inner_train", "validation", "selection", "oos")
_PERCENTILES = (0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "建立11A固定連續target arrays，並稽核Selection／Validation／OOS分布、"
            "同日排序可學性與選填的實際Round-trip R方向；本命令不訓練模型"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--round-trips",
        default=None,
        help=(
            "選填 no_filter_round_trips.csv；未指定時只嘗試active 9A strategy_compare標準路徑，"
            "找不到不視為錯誤"
        ),
    )
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="允許來源CSV inventory已更新時仍稽核既有Dataset；預設fail-fast",
    )
    return parser.parse_args(argv)


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_native(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _source_data_end(summary: dict[str, Any], events: pd.DataFrame) -> str:
    source_range = summary.get("source_data_date_range")
    if isinstance(source_range, dict):
        value = str(source_range.get("end") or "").strip()
        if value:
            return pd.Timestamp(value).strftime("%Y-%m-%d")
    return str(pd.to_datetime(events["label_eval_end_date"], errors="raise").max().date())


def _collapse_group_frame(
    events: pd.DataFrame,
    event_group_index: np.ndarray,
    targets: dict[str, np.ndarray],
    split_group_indices: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = events.copy()
    frame["group_index"] = np.asarray(event_group_index, dtype=np.int64)
    checked_columns = (
        "ticker",
        "date",
        "label",
        "max_upside_return",
        "max_downside_return",
        "decision_mfe_return",
        "decision_mae_return",
        "decision_reward_risk_ratio",
        "first_hit_bar",
    )
    for column in checked_columns:
        if column not in frame.columns:
            raise ValueError(f"continuous target audit events.csv 缺少欄位: {column}")
        nonunique = frame.groupby("group_index", sort=False)[column].nunique(dropna=False)
        if bool((nonunique > 1).any()):
            sample = nonunique[nonunique > 1].index[:5].tolist()
            raise ValueError(f"同一group_index的{column}不一致: sample={sample}")

    group_frame = (
        frame.sort_index()
        .drop_duplicates("group_index", keep="first")
        .sort_values("group_index")
        .reset_index(drop=True)
    )
    expected = np.arange(len(group_frame), dtype=np.int64)
    observed = group_frame["group_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("continuous target group frame未完整覆蓋連續group_index")

    group_frame["date"] = pd.to_datetime(group_frame["date"], errors="raise").dt.normalize()
    for name, array in targets.items():
        if len(array) != len(group_frame):
            raise ValueError(f"continuous target array長度不一致: {name}")
        group_frame[name] = np.asarray(array)

    for split_name, indices in split_group_indices.items():
        mask = np.zeros(len(group_frame), dtype=bool)
        mask[np.asarray(indices, dtype=np.int64)] = True
        group_frame[f"is_{split_name}"] = mask
    return group_frame


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float | None:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 3:
        return None
    x_rank = pd.Series(x_values[valid]).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_values[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _binary_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [LABEL_REJECT, LABEL_PASS])
    y = y[valid]
    s = s[valid]
    positive_count = int((y == LABEL_PASS).sum())
    negative_count = int((y == LABEL_REJECT).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_positive = float(ranks[y == LABEL_PASS].sum())
    auc = (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)
    return float(auc)


def _positive_sum_concentration(values: np.ndarray, top_fraction: float) -> float | None:
    positive = np.sort(values[values > 0.0])
    if positive.size == 0:
        return None
    top_count = max(1, int(math.ceil(float(positive.size) * float(top_fraction))))
    denominator = float(positive.sum())
    if denominator <= 0.0:
        return None
    return float(positive[-top_count:].sum() / denominator)


def _distribution_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    values = _finite(frame["target_raw_r"])
    if values.size == 0:
        return {"group_count": 0}
    quantiles = np.quantile(values, _PERCENTILES)
    quantile_payload = {
        f"p{int(round(percentile * 100)):02d}": float(value)
        for percentile, value in zip(_PERCENTILES, quantiles)
    }
    labels = frame.loc[np.isfinite(frame["target_raw_r"]), "label"].to_numpy(dtype=np.int64)
    pass_values = values[labels == LABEL_PASS]
    reject_values = values[labels == LABEL_REJECT]
    return {
        "group_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "quantiles": quantile_payload,
        "positive_rate": float((values > 0.0).mean()),
        "nonpositive_rate": float((values <= 0.0).mean()),
        "exact_zero_rate": float((values == 0.0).mean()),
        "unique_value_count": int(np.unique(values).size),
        "unique_value_ratio": float(np.unique(values).size / values.size),
        "top_1pct_positive_sum_share": _positive_sum_concentration(values, 0.01),
        "top_5pct_positive_sum_share": _positive_sum_concentration(values, 0.05),
        "pass_target_mean": float(pass_values.mean()) if pass_values.size else None,
        "pass_target_median": float(np.median(pass_values)) if pass_values.size else None,
        "reject_target_mean": float(reject_values.mean()) if reject_values.size else None,
        "reject_target_median": float(np.median(reject_values)) if reject_values.size else None,
        "binary_label_auc": _binary_auc(labels, values),
        "spearman_with_max_upside": _spearman(frame["target_raw_r"], frame["max_upside_return"]),
        "spearman_with_decision_mfe": _spearman(frame["target_raw_r"], frame["decision_mfe_return"]),
        "spearman_with_negative_decision_mae": _spearman(
            frame["target_raw_r"],
            -pd.to_numeric(frame["decision_mae_return"], errors="coerce"),
        ),
    }


def _tied_pair_count(counts: np.ndarray) -> int:
    values = np.asarray(counts, dtype=np.int64)
    return int(np.sum(values * (values - 1) // 2))


def _daily_rankability(frame: pd.DataFrame, *, split_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for date_value, day in frame.groupby("date", sort=True):
        values = _finite(day["target_raw_r"])
        if values.size == 0:
            continue
        unique, counts = np.unique(values, return_counts=True)
        pair_count = int(values.size * (values.size - 1) // 2)
        tied_pairs = _tied_pair_count(counts)
        rows.append(
            {
                "split": str(split_name),
                "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "group_count": int(values.size),
                "unique_target_count": int(unique.size),
                "unique_target_ratio": float(unique.size / values.size),
                "pair_count": pair_count,
                "tied_pair_count": int(tied_pairs),
                "comparable_pair_count": int(pair_count - tied_pairs),
                "target_min": float(values.min()),
                "target_max": float(values.max()),
                "target_spread": float(values.max() - values.min()),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"date_count": 0}, daily
    multi = daily[daily["group_count"] >= 2].copy()
    if multi.empty:
        return {
            "date_count": int(len(daily)),
            "multi_candidate_date_count": 0,
        }, daily
    total_pairs = int(multi["pair_count"].sum())
    comparable_pairs = int(multi["comparable_pair_count"].sum())
    rankable = multi["unique_target_count"] >= 2
    return {
        "date_count": int(len(daily)),
        "multi_candidate_date_count": int(len(multi)),
        "multi_candidate_date_rate": float(len(multi) / len(daily)),
        "rankable_date_count": int(rankable.sum()),
        "rankable_date_rate": float(rankable.mean()),
        "median_candidates_per_multi_candidate_date": float(multi["group_count"].median()),
        "p90_candidates_per_multi_candidate_date": float(multi["group_count"].quantile(0.90)),
        "mean_unique_target_ratio": float(multi["unique_target_ratio"].mean()),
        "pairwise_non_tie_rate": (
            float(comparable_pairs / total_pairs) if total_pairs > 0 else None
        ),
        "median_target_spread": float(multi["target_spread"].median()),
        "p10_target_spread": float(multi["target_spread"].quantile(0.10)),
    }, daily


def _same_day_binary_concordance(frame: pd.DataFrame) -> dict[str, Any]:
    date_scores: list[float] = []
    weighted_numerator = 0.0
    weighted_denominator = 0
    for _date, day in frame.groupby("date", sort=False):
        pass_values = _finite(day.loc[day["label"] == LABEL_PASS, "target_raw_r"])
        reject_values = _finite(day.loc[day["label"] == LABEL_REJECT, "target_raw_r"])
        if pass_values.size == 0 or reject_values.size == 0:
            continue
        comparisons = pass_values[:, None] - reject_values[None, :]
        score = float((comparisons > 0.0).mean() + 0.5 * (comparisons == 0.0).mean())
        pair_count = int(comparisons.size)
        date_scores.append(score)
        weighted_numerator += score * pair_count
        weighted_denominator += pair_count
    return {
        "mixed_label_date_count": int(len(date_scores)),
        "mean_date_concordance": float(np.mean(date_scores)) if date_scores else None,
        "median_date_concordance": float(np.median(date_scores)) if date_scores else None,
        "pair_weighted_concordance": (
            float(weighted_numerator / weighted_denominator)
            if weighted_denominator > 0
            else None
        ),
        "comparable_pass_reject_pair_count": int(weighted_denominator),
    }


def _resolve_round_trip_path(filter_id: str, explicit_path: str | None) -> tuple[Path | None, str]:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到round-trip檔案: {path}")
        return path, "explicit"
    candidate = (
        resolve_filter_model_output_dir(
            PROJECT_ROOT,
            filter_id,
            BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        )
        / "strategy_compare"
        / "no_filter_round_trips.csv"
    )
    return (candidate, "active_9a_standard_path") if candidate.is_file() else (None, "not_found")


def _first_nonempty_date(row: pd.Series) -> tuple[str, str]:
    for column in ("quality_score_date", "signal_date", "candidate_date"):
        if column not in row.index:
            continue
        raw_value = row.get(column)
        if pd.isna(raw_value):
            continue
        value = str(raw_value).strip()
        if value:
            return pd.Timestamp(value).strftime("%Y-%m-%d"), column
    return "", ""


def _trade_alignment_diagnostic(
    group_frame: pd.DataFrame,
    *,
    filter_id: str,
    explicit_round_trips: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    path, path_source = _resolve_round_trip_path(filter_id, explicit_round_trips)
    if path is None:
        return {
            "available": False,
            "reason": "no_filter_round_trips.csv not found",
            "path_source": path_source,
            "formula_tuned_from_trade_r": False,
            "diagnostic_only": True,
        }, pd.DataFrame()

    trades = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted({"ticker", "r_multiple"} - set(trades.columns))
    if missing:
        raise ValueError(f"round-trip檔案缺少欄位: {missing}")
    lookup = group_frame[["ticker", "date", "target_raw_r"]].copy()
    lookup["ticker"] = lookup["ticker"].astype(str)
    lookup["target_date"] = lookup["date"].dt.strftime("%Y-%m-%d")
    lookup = lookup.drop(columns=["date"])
    if lookup.duplicated(["ticker", "target_date"], keep=False).any():
        raise ValueError("continuous target trade lookup的ticker/date必須唯一")

    matched = trades.copy()
    resolved = matched.apply(_first_nonempty_date, axis=1)
    matched["target_date"] = [item[0] for item in resolved]
    matched["target_date_source"] = [item[1] for item in resolved]
    matched["ticker"] = matched["ticker"].astype(str)
    matched["r_multiple"] = pd.to_numeric(matched["r_multiple"], errors="coerce")
    matched = matched.merge(lookup, how="left", on=["ticker", "target_date"], validate="many_to_one")
    valid = matched[
        np.isfinite(pd.to_numeric(matched["r_multiple"], errors="coerce"))
        & np.isfinite(pd.to_numeric(matched["target_raw_r"], errors="coerce"))
    ].copy()
    if valid.empty:
        return {
            "available": True,
            "path": str(path),
            "path_source": path_source,
            "trade_count": int(len(matched)),
            "matched_trade_count": 0,
            "coverage_rate": 0.0,
            "formula_tuned_from_trade_r": False,
            "diagnostic_only": True,
        }, matched

    valid = valid.sort_values("target_raw_r", kind="mergesort").reset_index(drop=True)
    target_values = valid["target_raw_r"].to_numpy(dtype=np.float64)
    r_values = valid["r_multiple"].to_numpy(dtype=np.float64)
    top_count = max(1, int(math.ceil(len(valid) * 0.10)))
    bottom_count = top_count
    top = valid.tail(top_count)
    bottom = valid.head(bottom_count)
    positive_r_total = float(np.maximum(r_values, 0.0).sum())
    top_positive_r = float(np.maximum(top["r_multiple"].to_numpy(dtype=np.float64), 0.0).sum())
    large_winner_mask = r_values >= 2.0
    median_target = float(np.median(target_values))
    result = {
        "available": True,
        "path": str(path),
        "path_source": path_source,
        "trade_count": int(len(matched)),
        "matched_trade_count": int(len(valid)),
        "coverage_rate": float(len(valid) / len(matched)) if len(matched) else None,
        "spearman_target_vs_r_multiple": _spearman(target_values, r_values),
        "target_positive_vs_trade_positive_accuracy": float(((target_values > 0.0) == (r_values > 0.0)).mean()),
        "all_trade_average_r": float(r_values.mean()),
        "top_target_decile_average_r": float(top["r_multiple"].mean()),
        "bottom_target_decile_average_r": float(bottom["r_multiple"].mean()),
        "top_target_decile_positive_r_share": (
            float(top_positive_r / positive_r_total) if positive_r_total > 0.0 else None
        ),
        "large_winner_count_r_ge_2": int(large_winner_mask.sum()),
        "large_winner_top_half_target_retention": (
            float((target_values[large_winner_mask] >= median_target).mean())
            if bool(large_winner_mask.any())
            else None
        ),
        "formula_tuned_from_trade_r": False,
        "diagnostic_only": True,
    }
    return result, matched


def build_continuous_target_audit(
    group_frame: pd.DataFrame,
    *,
    target_contract: dict[str, Any],
    split_report: dict[str, Any],
    dataset_summary: dict[str, Any],
    trade_alignment: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits: dict[str, Any] = {}
    daily_frames: list[pd.DataFrame] = []
    for split_name in _SPLIT_ORDER:
        subset = group_frame[group_frame[f"is_{split_name}"] & group_frame["valid_mask"]].copy()
        distribution = _distribution_metrics(subset)
        rankability, daily = _daily_rankability(subset, split_name=split_name)
        distribution["same_day_rankability"] = rankability
        distribution["same_day_binary_concordance"] = _same_day_binary_concordance(subset)
        splits[split_name] = distribution
        daily_frames.append(daily)

    all_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    payload = {
        "schema_version": TARGET_AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "11A Strategy-aligned Continuous Outcome Target",
        "status": "IMPLEMENTED_AUDIT_ONLY",
        "training_performed": False,
        "target_contract": target_contract,
        "dataset": {
            "filter_id": str(dataset_summary.get("filter_id") or DEFAULT_FILTER_ID),
            "dataset_profile": dataset_summary.get("dataset"),
            "event_count": int(dataset_summary.get("event_count", len(group_frame))),
            "feature_group_count": int(dataset_summary.get("feature_group_count", len(group_frame))),
            "source_data_date_range": dataset_summary.get("source_data_date_range"),
            "label_policy": dataset_summary.get("label_policy"),
        },
        "split_contract": split_report,
        "split_metrics": splits,
        "trade_r_alignment": trade_alignment,
        "interpretation_contract": {
            "selection_only_for_future_training_and_epoch_selection": True,
            "oos_statistics_are_iterative_research_evidence_only": True,
            "target_formula_uses_no_split_or_oos_statistics": True,
            "this_audit_does_not_authorize_training_or_runtime_deployment": True,
        },
    }
    return payload, all_daily


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value) * 100:.2f}%"


def render_continuous_target_audit_markdown(payload: dict[str, Any]) -> str:
    contract = payload["target_contract"]
    lines = [
        "# 11A Strategy-aligned Continuous Target Audit",
        "",
        f"- Target：`{contract['target_id']}`",
        f"- Horizon：`{contract['horizon_bars']}` trading bars",
        f"- Risk budget：`{float(contract['risk_budget_return']) * 100:.2f}%`",
        f"- Full-horizon time penalty：`{contract['full_horizon_time_penalty_r']:.4f}R`",
        "- 狀態：只完成target arrays與可學性稽核；本輪沒有訓練模型。",
        f"- 無前視邊界：target由各事件固定未來{int(contract['horizon_bars'])}交易日建立；公式不讀取Selection／Validation／OOS統計量。",
        "",
        "## 1. 固定公式",
        "",
        "```text",
        "target_raw_r",
        f"= 突破後、首次-{float(contract['risk_budget_return']) * 100:.2f}%風險觸發前的最大有利漲幅 ÷ {float(contract['risk_budget_return']) * 100:.2f}%風險預算",
        f"- 到達該高點前必須承受的最大不利跌幅 ÷ {float(contract['risk_budget_return']) * 100:.2f}%風險預算",
        f"- {float(contract['full_horizon_time_penalty_r']):.4f}R × 到達高點所用時間比例",
        "```",
        "",
        "同一日High／Low先後未知時採保守adverse-first；若Low先觸及風險障礙，該日High不計入可用機會。",
        "",
        "## 2. 分布與同日排序可學性",
        "",
        "| 區段 | Groups | Mean | P01 | P50 | P99 | 正值率 | Unique率 | Binary AUC | 同日可排序日 | Pair非Tie率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in _SPLIT_ORDER:
        metrics = payload["split_metrics"][split_name]
        q = metrics.get("quantiles") or {}
        rankability = metrics.get("same_day_rankability") or {}
        lines.append(
            f"| {split_name} | {int(metrics.get('group_count', 0)):,} "
            f"| {_fmt(metrics.get('mean'))} | {_fmt(q.get('p01'))} | {_fmt(q.get('p50'))} "
            f"| {_fmt(q.get('p99'))} | {_pct(metrics.get('positive_rate'))} "
            f"| {_pct(metrics.get('unique_value_ratio'))} | {_fmt(metrics.get('binary_label_auc'))} "
            f"| {_pct(rankability.get('rankable_date_rate'))} | {_pct(rankability.get('pairwise_non_tie_rate'))} |"
        )

    lines += [
        "",
        "## 3. 與既有二元Label的關係",
        "",
        "| 區段 | PASS target平均 | REJECT target平均 | 同日PASS/REJECT pair concordance | Top 1%正target貢獻 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in _SPLIT_ORDER:
        metrics = payload["split_metrics"][split_name]
        concordance = metrics.get("same_day_binary_concordance") or {}
        lines.append(
            f"| {split_name} | {_fmt(metrics.get('pass_target_mean'))} "
            f"| {_fmt(metrics.get('reject_target_mean'))} "
            f"| {_fmt(concordance.get('pair_weighted_concordance'))} "
            f"| {_pct(metrics.get('top_1pct_positive_sum_share'))} |"
        )

    trade = payload.get("trade_r_alignment") or {}
    lines += ["", "## 4. 實際Round-trip R方向診斷", ""]
    if not bool(trade.get("available")):
        lines.append(f"- 未取得可用no-filter round-trip檔案：`{trade.get('reason', 'not available')}`。")
        lines.append("- 這不影響target分布與同日可排序稽核，但尚不能確認target與實際交易R方向一致。")
    else:
        lines += [
            f"- 來源：`{trade.get('path')}`",
            f"- 配對：`{trade.get('matched_trade_count', 0)}` / `{trade.get('trade_count', 0)}`；coverage `{_pct(trade.get('coverage_rate'))}`",
            f"- Spearman(target, realized R)：`{_fmt(trade.get('spearman_target_vs_r_multiple'))}`",
            f"- Top target decile平均R：`{_fmt(trade.get('top_target_decile_average_r'))}`；Bottom decile：`{_fmt(trade.get('bottom_target_decile_average_r'))}`",
            f"- ≥2R大贏家位於target上半部比例：`{_pct(trade.get('large_winner_top_half_target_retention'))}`",
            "- 本診斷不參與target公式、normalization、clip或任何參數擬合。",
        ]

    lines += [
        "",
        "## 5. 本輪邊界",
        "",
        "- 不建立新模型architecture，不訓練、不選epoch、不調threshold。",
        "- 不使用OOS統計定義target、normalization或loss。",
        "- 下一步是否進入11A regression training，必須依本稽核的target分布、tie／極端值與實際R方向結果再判定。",
        "",
    ]
    return "\n".join(lines)


def _artifact_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "target_raw_r": target_dir / TARGET_RAW_FILENAME,
        "favorable_return": target_dir / TARGET_FAVORABLE_RETURN_FILENAME,
        "adverse_return_to_peak": target_dir / TARGET_ADVERSE_RETURN_FILENAME,
        "opportunity_bar": target_dir / TARGET_OPPORTUNITY_BAR_FILENAME,
        "first_risk_breach_bar": target_dir / TARGET_RISK_BREACH_BAR_FILENAME,
        "valid_mask": target_dir / TARGET_VALID_MASK_FILENAME,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    summary, indexed_features, _context, labels, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(args.allow_stale_source),
    )
    paths = dataset_paths(args.filter_id)
    event_group_index = np.asarray(indexed_features.event_group_index, dtype=np.int64)
    group_anchor_prices = load_npy(paths.group_anchor_prices)
    future_high_prices = load_npy(paths.future_high_prices)
    future_low_prices = load_npy(paths.future_low_prices)
    future_available_bars = load_npy(paths.future_available_bars)

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    targets = build_strategy_aligned_group_targets(
        group_anchor_prices,
        future_high_prices,
        future_low_prices,
        future_available_bars,
        spec=spec,
    )
    group_count = int(len(group_anchor_prices))
    valid_count = int(np.asarray(targets["valid_mask"], dtype=bool).sum())
    if valid_count == 0:
        raise ValueError("11A continuous target沒有任何有效group")

    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=_source_data_end(summary, events),
    )
    (
        split_assignments,
        inner_train_idx,
        validation_idx,
        selection_idx,
        oos_idx,
        split_report,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=bool(BREAKOUT_QUALITY_USE_INNER_VALIDATION),
        inner_validation_months=int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS),
        early_stopping_enabled=bool(
            BREAKOUT_QUALITY_USE_INNER_VALIDATION
            and int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE) > 0
        ),
    )
    del split_assignments
    split_group_indices = {
        "inner_train": np.unique(event_group_index[inner_train_idx]),
        "validation": np.unique(event_group_index[validation_idx]),
        "selection": np.unique(event_group_index[selection_idx]),
        "oos": np.unique(event_group_index[oos_idx]),
    }
    group_frame = _collapse_group_frame(events, event_group_index, targets, split_group_indices)
    trade_alignment, trade_matches = _trade_alignment_diagnostic(
        group_frame,
        filter_id=args.filter_id,
        explicit_round_trips=args.round_trips,
    )
    payload, daily_frame = build_continuous_target_audit(
        group_frame,
        target_contract=spec.contract_payload(),
        split_report=split_report,
        dataset_summary={**summary, "filter_id": args.filter_id},
        trade_alignment=trade_alignment,
    )

    target_dir = resolve_continuous_target_dir(PROJECT_ROOT, args.filter_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    array_paths = _artifact_paths(target_dir)
    for name, path in array_paths.items():
        save_npy_atomic(path, np.asarray(targets[name]))

    audit_json_path = target_dir / TARGET_AUDIT_JSON_FILENAME
    audit_markdown_path = target_dir / TARGET_AUDIT_MARKDOWN_FILENAME
    daily_csv_path = target_dir / TARGET_DAILY_CSV_FILENAME
    trade_matches_path = target_dir / TARGET_TRADE_MATCHES_CSV_FILENAME
    _write_json(audit_json_path, payload)
    audit_markdown_path.write_text(
        render_continuous_target_audit_markdown(payload),
        encoding="utf-8",
    )
    daily_frame.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")
    if not trade_matches.empty:
        trade_matches.to_csv(trade_matches_path, index=False, encoding="utf-8-sig")
    elif trade_matches_path.exists():
        trade_matches_path.unlink()

    source_artifacts = summary.get("dataset_artifacts")
    manifest = {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": str(args.filter_id),
        "target_contract": spec.contract_payload(),
        "group_count": group_count,
        "valid_group_count": valid_count,
        "invalid_group_count": int(group_count - valid_count),
        "dataset_policy": summary.get("policy"),
        "dataset_artifact_source": {
            key: value
            for key, value in dict(source_artifacts or {}).items()
            if key in {
                "group_anchor_prices",
                "future_high_prices",
                "future_low_prices",
                "future_available_bars",
                "event_group_index",
                "events_csv",
            }
        },
        "split_policy": outer_policy,
        "split_report": split_report,
        "artifacts": {name: build_file_manifest(path) for name, path in array_paths.items()},
        "audit_outputs": {
            "json": build_file_manifest(audit_json_path),
            "markdown": build_file_manifest(audit_markdown_path),
            "daily_csv": build_file_manifest(daily_csv_path),
            **(
                {"trade_matches_csv": build_file_manifest(trade_matches_path)}
                if trade_matches_path.is_file()
                else {}
            ),
        },
        "training_performed": False,
        "runtime_eligible": False,
    }
    manifest_path = target_dir / TARGET_MANIFEST_FILENAME
    _write_json(manifest_path, manifest)

    print("11A continuous target audit完成")
    print(f"target={spec.target_id} groups={group_count:,} valid={valid_count:,}")
    for split_name in _SPLIT_ORDER:
        metrics = payload["split_metrics"][split_name]
        rankability = metrics.get("same_day_rankability") or {}
        print(
            f"- {split_name:<11} groups={int(metrics.get('group_count', 0)):,} "
            f"mean={_fmt(metrics.get('mean'))} binary_auc={_fmt(metrics.get('binary_label_auc'))} "
            f"rankable_dates={_pct(rankability.get('rankable_date_rate'))}"
        )
    if trade_alignment.get("available"):
        print(
            "- trade R: "
            f"matched={trade_alignment.get('matched_trade_count')}/{trade_alignment.get('trade_count')} "
            f"spearman={_fmt(trade_alignment.get('spearman_target_vs_r_multiple'))}"
        )
    else:
        print("- trade R: no_filter_round_trips.csv未找到，略過實際R方向診斷")
    print(f"已輸出: {manifest_path}")
    print(f"已輸出: {audit_markdown_path}")
    return 0


__all__ = [
    "TARGET_AUDIT_SCHEMA_VERSION",
    "build_continuous_target_audit",
    "main",
    "parse_args",
    "render_continuous_target_audit_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
