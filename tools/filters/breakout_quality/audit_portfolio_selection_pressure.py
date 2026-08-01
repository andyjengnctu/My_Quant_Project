"""11K read-only portfolio selection-pressure attribution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import BREAKOUT_QUALITY_DEFAULT_FILTER_ID
from filters.breakout_quality.continuous_target import STRATEGY_ALIGNED_NO_TIME_TARGET_ID
from tools.filters.breakout_quality.audit_selection_strategy_realization import (
    AUDIT_JSON_FILENAME as SOURCE_AUDIT_JSON_FILENAME,
    _output_dir as selection_strategy_output_dir,
)
from filters.breakout_quality.console_report import print_artifact_paths
from tools.filters.breakout_quality.common import PROJECT_ROOT, write_json
from tools.filters.breakout_quality.train_continuous_ranker import _spearman

AUDIT_SCHEMA_VERSION = 1
EXPERIMENT_NAME = "11K Portfolio Selection-pressure Attribution Audit"
AUDIT_DIRNAME = "portfolio_selection_pressure_audit"
AUDIT_JSON_FILENAME = "portfolio_selection_pressure_audit.json"
AUDIT_MARKDOWN_FILENAME = "portfolio_selection_pressure_audit.md"
SIGNALS_FILENAME = "selection_pressure_orderable_occurrences.csv"
DAILY_FILENAME = "selection_pressure_daily.csv"
BUCKETS_FILENAME = "selection_pressure_buckets.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11K research-only portfolio selection-pressure attribution；"
            "只讀11I orderable candidates與actual trades，不重播策略、不建立counterfactual"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"11K JSON根節點必須是object: {path}")
    return payload


def _artifact_path(source_dir: Path, source_payload: dict[str, Any], key: str) -> Path:
    artifact = dict((source_payload.get("artifacts") or {}).get(key) or {})
    filename = str(artifact.get("filename") or "").strip()
    expected_hash = str(artifact.get("sha256") or "").strip().lower()
    if not filename or not expected_hash:
        raise ValueError(f"11K 11I artifact缺少filename／sha256: key={key}")
    path = source_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"11K找不到11I artifact: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash.lower() != expected_hash:
        raise ValueError(
            f"11K偵測到11I artifact SHA256不一致: key={key}, "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    return path


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    values = series.fillna("").astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "yes", "y"})


def _date_text_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")


def _prepare_orderable_occurrences(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "trade_date", "target_raw_r", "target_match"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"11K orderable artifact缺少欄位: {missing}")
    out = frame.copy()
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip()
    out["trade_date"] = _date_text_series(out["trade_date"])
    out["target_raw_r"] = pd.to_numeric(out["target_raw_r"], errors="coerce")
    out["target_match"] = _coerce_bool(out["target_match"])
    out = out[
        (out["ticker"] != "")
        & (out["trade_date"] != "")
        & out["target_match"]
        & np.isfinite(out["target_raw_r"])
    ].copy()
    duplicate = out.duplicated(["ticker", "trade_date"], keep=False)
    if bool(duplicate.any()):
        sample = out.loc[duplicate, ["ticker", "trade_date", "target_raw_r"]].head(10).to_dict("records")
        raise ValueError(f"11K同ticker／trade_date存在多筆orderable occurrence: sample={sample}")
    return out.sort_values(["trade_date", "ticker"], kind="mergesort").reset_index(drop=True)


def _prepare_actual_trades(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "entry_date", "target_raw_r", "target_match", "r_multiple"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"11K actual trade artifact缺少欄位: {missing}")
    out = frame.copy()
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip()
    out["entry_date"] = _date_text_series(out["entry_date"])
    out["target_raw_r"] = pd.to_numeric(out["target_raw_r"], errors="coerce")
    out["r_multiple"] = pd.to_numeric(out["r_multiple"], errors="coerce")
    out["target_match"] = _coerce_bool(out["target_match"])
    out = out[
        (out["ticker"] != "")
        & (out["entry_date"] != "")
        & out["target_match"]
        & np.isfinite(out["target_raw_r"])
        & np.isfinite(out["r_multiple"])
    ].copy()
    duplicate = out.duplicated(["ticker", "entry_date"], keep=False)
    if bool(duplicate.any()):
        sample = out.loc[duplicate, ["ticker", "entry_date", "r_multiple"]].head(10).to_dict("records")
        raise ValueError(f"11K同ticker／entry_date存在多筆actual trade: sample={sample}")
    return out.sort_values(["entry_date", "ticker"], kind="mergesort").reset_index(drop=True)


def _pressure_bucket(candidate_count: int) -> str:
    count = int(candidate_count)
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    if count <= 10:
        return "6-10"
    return "11+"


def _build_selection_pressure_tables(
    orderable_frame: pd.DataFrame,
    trade_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    orderable = _prepare_orderable_occurrences(orderable_frame)
    trades = _prepare_actual_trades(trade_frame)
    if orderable.empty:
        raise ValueError("11K沒有可用orderable target-matched occurrences")
    if trades.empty:
        raise ValueError("11K沒有可用actual target-matched trades")

    selected = trades[["ticker", "entry_date", "r_multiple"]].rename(columns={"entry_date": "trade_date"})
    selected["selected"] = 1
    merged = orderable.merge(
        selected,
        how="left",
        on=["ticker", "trade_date"],
        validate="one_to_one",
    )
    matched_count = int(pd.to_numeric(merged["selected"], errors="coerce").fillna(0).sum())
    if matched_count != len(trades):
        available = set(zip(orderable["ticker"], orderable["trade_date"]))
        missing = [
            {"ticker": row.ticker, "entry_date": row.entry_date}
            for row in trades.itertuples(index=False)
            if (row.ticker, row.entry_date) not in available
        ][:10]
        raise ValueError(
            "11K actual trades不是11I orderable occurrences的完整子集: "
            f"expected={len(trades)}, matched={matched_count}, missing_sample={missing}"
        )
    merged["selected"] = pd.to_numeric(merged["selected"], errors="coerce").fillna(0).astype(int).astype(bool)
    merged["r_multiple"] = pd.to_numeric(merged["r_multiple"], errors="coerce")
    merged["candidate_count"] = merged.groupby("trade_date", sort=False)["ticker"].transform("size").astype(int)
    merged["target_percentile_within_day"] = merged.groupby("trade_date", sort=False)["target_raw_r"].rank(
        method="average", pct=True, ascending=True
    )
    merged["global_target_percentile"] = merged["target_raw_r"].rank(method="average", pct=True, ascending=True)
    merged["global_target_decile"] = np.minimum(
        10,
        np.maximum(1, np.ceil(merged["global_target_percentile"] * 10.0).astype(int)),
    )
    merged["pressure_bucket"] = merged["candidate_count"].map(_pressure_bucket)

    daily_rows: list[dict[str, Any]] = []
    for trade_date, group in merged.groupby("trade_date", sort=True):
        selected_group = group[group["selected"]]
        candidate_count = int(len(group))
        selected_count = int(len(selected_group))
        top_k_retention = None
        target_opportunity_gap = None
        top_k_overlap_count = 0
        if selected_count > 0:
            top_k = group.sort_values(
                ["target_raw_r", "ticker"], ascending=[False, True], kind="mergesort"
            ).head(selected_count)
            actual_keys = set(selected_group["ticker"].astype(str))
            top_keys = set(top_k["ticker"].astype(str))
            top_k_overlap_count = int(len(actual_keys.intersection(top_keys)))
            top_k_retention = float(top_k_overlap_count / selected_count)
            target_opportunity_gap = float(top_k["target_raw_r"].mean() - selected_group["target_raw_r"].mean())
        daily_rows.append({
            "trade_date": str(trade_date),
            "candidate_count": candidate_count,
            "selected_count": selected_count,
            "pressure_bucket": _pressure_bucket(candidate_count),
            "orderable_target_mean": float(group["target_raw_r"].mean()),
            "selected_target_mean": float(selected_group["target_raw_r"].mean()) if selected_count else None,
            "selected_target_percentile_mean": (
                float(selected_group["target_percentile_within_day"].mean()) if selected_count else None
            ),
            "selected_top_half_rate": (
                float((selected_group["target_percentile_within_day"] >= 0.5).mean()) if selected_count else None
            ),
            "selected_top_quartile_rate": (
                float((selected_group["target_percentile_within_day"] >= 0.75).mean()) if selected_count else None
            ),
            "top_k_overlap_count": top_k_overlap_count,
            "top_k_retention": top_k_retention,
            "target_opportunity_gap_r": target_opportunity_gap,
            "actual_r_mean": float(selected_group["r_multiple"].mean()) if selected_count else None,
        })
    daily = pd.DataFrame(daily_rows)

    bucket_rows: list[dict[str, Any]] = []
    bucket_order = ["1", "2-3", "4-5", "6-10", "11+"]
    for bucket in bucket_order:
        group = merged[merged["pressure_bucket"] == bucket]
        day_group = daily[daily["pressure_bucket"] == bucket]
        selected_group = group[group["selected"]]
        competition_selected = selected_group[selected_group["candidate_count"] >= 2]
        bucket_rows.append({
            "pressure_bucket": bucket,
            "day_count": int(group["trade_date"].nunique()),
            "orderable_occurrence_count": int(len(group)),
            "selected_trade_count": int(len(selected_group)),
            "selected_rate": float(len(selected_group) / len(group)) if len(group) else None,
            "selected_target_percentile_mean": (
                float(competition_selected["target_percentile_within_day"].mean())
                if len(competition_selected) else None
            ),
            "top_k_retention": (
                float(day_group["top_k_overlap_count"].sum() / day_group["selected_count"].sum())
                if int(day_group["selected_count"].sum()) > 0 else None
            ),
            "target_opportunity_gap_r": (
                float(day_group["target_opportunity_gap_r"].dropna().mean())
                if day_group["target_opportunity_gap_r"].notna().any() else None
            ),
            "target_vs_realized_r_spearman": (
                _spearman(
                    selected_group["target_raw_r"].to_numpy(dtype=np.float64),
                    selected_group["r_multiple"].to_numpy(dtype=np.float64),
                ) if len(selected_group) >= 2 else None
            ),
        })
    buckets = pd.DataFrame(bucket_rows)

    selected_rows = merged[merged["selected"]].copy()
    unselected_rows = merged[~merged["selected"]].copy()
    competition_selected = selected_rows[selected_rows["candidate_count"] >= 2].copy()
    competition_daily = daily[(daily["candidate_count"] >= 2) & (daily["selected_count"] > 0)].copy()
    total_selected_on_competition_days = int(competition_daily["selected_count"].sum())
    metrics = {
        "orderable_occurrence_count": int(len(merged)),
        "orderable_day_count": int(merged["trade_date"].nunique()),
        "selected_trade_count": int(len(selected_rows)),
        "competition_day_count": int((daily["candidate_count"] >= 2).sum()),
        "competition_day_with_trade_count": int(len(competition_daily)),
        "selected_trade_count_on_competition_days": int(len(competition_selected)),
        "selected_target_mean": float(selected_rows["target_raw_r"].mean()),
        "unselected_target_mean": float(unselected_rows["target_raw_r"].mean()) if len(unselected_rows) else None,
        "selected_minus_unselected_target_mean_r": (
            float(selected_rows["target_raw_r"].mean() - unselected_rows["target_raw_r"].mean())
            if len(unselected_rows) else None
        ),
        "target_vs_selected_spearman": _spearman(
            merged["target_raw_r"].to_numpy(dtype=np.float64),
            merged["selected"].astype(float).to_numpy(dtype=np.float64),
        ),
        "selected_target_percentile_mean_competition": (
            float(competition_selected["target_percentile_within_day"].mean())
            if len(competition_selected) else None
        ),
        "selected_target_percentile_median_competition": (
            float(competition_selected["target_percentile_within_day"].median())
            if len(competition_selected) else None
        ),
        "selected_top_half_rate_competition": (
            float((competition_selected["target_percentile_within_day"] >= 0.5).mean())
            if len(competition_selected) else None
        ),
        "selected_top_quartile_rate_competition": (
            float((competition_selected["target_percentile_within_day"] >= 0.75).mean())
            if len(competition_selected) else None
        ),
        "top_k_retention_occurrence_weighted": (
            float(competition_daily["top_k_overlap_count"].sum() / total_selected_on_competition_days)
            if total_selected_on_competition_days > 0 else None
        ),
        "top_k_retention_date_weighted": (
            float(competition_daily["top_k_retention"].mean()) if len(competition_daily) else None
        ),
        "target_opportunity_gap_r_date_weighted": (
            float(competition_daily["target_opportunity_gap_r"].mean()) if len(competition_daily) else None
        ),
        "within_day_target_percentile_vs_realized_r_spearman": (
            _spearman(
                competition_selected["target_percentile_within_day"].to_numpy(dtype=np.float64),
                competition_selected["r_multiple"].to_numpy(dtype=np.float64),
            ) if len(competition_selected) >= 2 else None
        ),
        "selected_top_quartile_average_r": (
            float(competition_selected.loc[
                competition_selected["target_percentile_within_day"] >= 0.75, "r_multiple"
            ].mean())
            if bool((competition_selected["target_percentile_within_day"] >= 0.75).any()) else None
        ),
        "selected_bottom_quartile_average_r": (
            float(competition_selected.loc[
                competition_selected["target_percentile_within_day"] <= 0.25, "r_multiple"
            ].mean())
            if bool((competition_selected["target_percentile_within_day"] <= 0.25).any()) else None
        ),
        "selection_rate_by_global_target_decile": {
            str(decile): {
                "rows": int(len(group)),
                "selected": int(group["selected"].sum()),
                "selection_rate": float(group["selected"].mean()),
            }
            for decile, group in merged.groupby("global_target_decile", sort=True)
        },
    }
    return merged, daily, buckets, metrics


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    number = float(value)
    return "-" if not math.isfinite(number) else f"{number:.{digits}f}"


def _render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# 11K Portfolio Selection-pressure Attribution Audit",
        "",
        "- Runtime：research-only、read-only；只讀11I凍結工件，不重播市場、不建立counterfactual、不訓練。",
        "- 目的：檢查同日候選競爭下，actual portfolio是否選到較高No-time Target候選。",
        "- 未交易候選沒有realized R；本audit不填0R，也不宣稱反事實績效。",
        "",
        "## 1. Selection pressure",
        "",
        f"- Orderable occurrences：`{metrics['orderable_occurrence_count']:,}`；actual matched trades：`{metrics['selected_trade_count']:,}`。",
        f"- Competition days：`{metrics['competition_day_count']:,}`；其中有actual trade：`{metrics['competition_day_with_trade_count']:,}`。",
        f"- Selected／unselected Target mean：`{_fmt(metrics['selected_target_mean'])}`／`{_fmt(metrics['unselected_target_mean'])}`R。",
        f"- Target↔selected indicator：`{_fmt(metrics['target_vs_selected_spearman'])}`。",
        "",
        "## 2. Same-day ranking",
        "",
        f"- Actual selected Target percentile mean／median：`{_fmt(metrics['selected_target_percentile_mean_competition'])}`／`{_fmt(metrics['selected_target_percentile_median_competition'])}`。",
        f"- Actual selected位於同日Target top half／top quartile：`{_fmt(metrics['selected_top_half_rate_competition'], 2)}`／`{_fmt(metrics['selected_top_quartile_rate_competition'], 2)}`。",
        f"- 依actual買入數k計算的Target top-k retention：`{_fmt(metrics['top_k_retention_occurrence_weighted'])}`。",
        f"- Target top-k相對actual selection的平均機會差：`{_fmt(metrics['target_opportunity_gap_r_date_weighted'])}`R。",
        "",
        "## 3. Realized-R diagnostic among selected trades",
        "",
        f"- 同日Target percentile↔realized R：`{_fmt(metrics['within_day_target_percentile_vs_realized_r_spearman'])}`。",
        f"- Selected top／bottom quartile平均R：`{_fmt(metrics['selected_top_quartile_average_r'])}`／`{_fmt(metrics['selected_bottom_quartile_average_r'])}`。",
        "",
        "## 4. Boundary",
        "",
        "- 若selected percentile與top-k retention偏高，代表現有portfolio選擇已傾向高Target；下一步應檢查成交後capture／exit path，不再研究候選擠出。",
        "- 若兩者偏低且Target opportunity gap明顯為正，代表現有buy-sort／capacity allocation會排除高Target候選；下一步才值得做固定ranking-proxy audit。",
        "- 本結果不授權使用future Target作runtime排序，也不授權模型、threshold或optimizer調整。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    filter_id = str(args.filter_id)
    started = time.perf_counter()
    source_dir = selection_strategy_output_dir(filter_id)
    source_json_path = source_dir / SOURCE_AUDIT_JSON_FILENAME
    if not source_json_path.is_file():
        raise FileNotFoundError(f"11K需要先完成11I: {source_json_path}")
    source = _read_json(source_json_path)
    if str(source.get("filter_id") or "") != filter_id:
        raise ValueError("11K 11I filter_id不一致")
    trade_metrics = dict(source.get("trade_metrics") or {})
    overall_rho = trade_metrics.get("spearman_target_vs_realized_r")
    pass_rho = dict(trade_metrics.get("label_conditional") or {}).get("PASS", {}).get(
        "spearman_target_vs_realized_r"
    )
    coverage = dict(source.get("coverage") or {})
    actual_coverage = coverage.get("actual_trade_coverage_vs_qualified")
    if overall_rho is None or pass_rho is None or float(overall_rho) <= 0 or float(pass_rho) <= 0:
        raise ValueError("11K需要11I overall與PASS Target↔R均為正")
    if actual_coverage is None or not (0.0 < float(actual_coverage) < 1.0):
        raise ValueError("11K需要11I actual trade coverage介於0與1")

    orderable_path = _artifact_path(source_dir, source, "orderable")
    trades_path = _artifact_path(source_dir, source, "trade_matches")
    orderable = pd.read_csv(orderable_path, encoding="utf-8-sig")
    trades = pd.read_csv(trades_path, encoding="utf-8-sig")
    signals, daily, buckets, metrics = _build_selection_pressure_tables(orderable, trades)
    expected_matched = int(trade_metrics.get("matched_trade_count") or 0)
    if int(metrics["selected_trade_count"]) != expected_matched:
        raise ValueError(
            "11K selected trade count與11I不一致: "
            f"expected={expected_matched}, actual={metrics['selected_trade_count']}"
        )

    output_dir = source_dir / AUDIT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path = output_dir / SIGNALS_FILENAME
    daily_path = output_dir / DAILY_FILENAME
    buckets_path = output_dir / BUCKETS_FILENAME
    signals.to_csv(signals_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    buckets.to_csv(buckets_path, index=False, encoding="utf-8-sig")
    artifacts = {}
    for key, path in (("signals", signals_path), ("daily", daily_path), ("buckets", buckets_path)):
        artifacts[key] = {
            "filename": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": float(time.perf_counter() - started),
        "filter_id": filter_id,
        "target_id": STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
        "source_11i": {
            "path": str(source_json_path),
            "sha256": _sha256_file(source_json_path),
            "qualified_unique_signal_count": int(coverage.get("qualified_unique_signal_count") or 0),
            "orderable_unique_signal_count": int(coverage.get("orderable_unique_signal_count") or 0),
            "matched_trade_count": expected_matched,
            "actual_trade_coverage_vs_qualified": float(actual_coverage),
            "target_vs_r_spearman": float(overall_rho),
            "pass_target_vs_r_spearman": float(pass_rho),
        },
        "metrics": metrics,
        "pressure_buckets": buckets.to_dict("records"),
        "artifacts": artifacts,
        "training_performed": False,
        "strategy_replay_performed": False,
        "counterfactual_performed": False,
        "runtime_eligible": False,
        "interpretation_contract": {
            "unselected_realized_r_not_imputed": True,
            "association_not_counterfactual_performance": True,
            "future_target_not_runtime_eligible": True,
            "audit_does_not_authorize_training_or_ranking_change": True,
        },
    }
    json_path = output_dir / AUDIT_JSON_FILENAME
    markdown_path = output_dir / AUDIT_MARKDOWN_FILENAME
    write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    print("11K portfolio selection-pressure attribution audit完成")
    print(
        f"orderable_occurrences={metrics['orderable_occurrence_count']:,} "
        f"selected_trades={metrics['selected_trade_count']:,} "
        f"competition_days={metrics['competition_day_count']:,}"
    )
    print(
        "- Selection alignment: "
        f"selected_percentile={metrics['selected_target_percentile_mean_competition']} "
        f"top_k_retention={metrics['top_k_retention_occurrence_weighted']} "
        f"target_gap_r={metrics['target_opportunity_gap_r_date_weighted']}"
    )
    print_artifact_paths(
        [("Markdown 報表", markdown_path), ("完整 JSON", json_path)],
        project_root=PROJECT_ROOT,
    )
    return 0


__all__ = [
    "AUDIT_JSON_FILENAME",
    "AUDIT_MARKDOWN_FILENAME",
    "BUCKETS_FILENAME",
    "DAILY_FILENAME",
    "SIGNALS_FILENAME",
    "_build_selection_pressure_tables",
    "main",
    "parse_args",
]
