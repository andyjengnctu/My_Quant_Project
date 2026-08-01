"""11C research-only audit of the strategy-qualified candidate decision surface."""

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

from config.breakout_quality import STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from filters.breakout_quality.continuous_target import (
    STRATEGY_ALIGNED_TARGET_ID,
    TARGET_TRADE_MATCHES_CSV_FILENAME,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from tools.filters.breakout_quality.common import PROJECT_ROOT, write_json
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_HARD_FILTER,
    _first_existing_comparison_dir,
    _scenario_summary as summarize_strategy_scenario,
    run_no_filter_candidate_replay_from_metadata,
)
from tools.filters.breakout_quality.train_continuous_ranker import (
    RANKER_REPORT_JSON_FILENAME,
    RANKER_SCORE_FILENAME,
    _daily_rank_metrics,
    _spearman,
)

AUDIT_SCHEMA_VERSION = 1
AUDIT_DIRNAME = "qualified_candidate_set_audit"
AUDIT_JSON_FILENAME = "qualified_candidate_set_audit.json"
AUDIT_MARKDOWN_FILENAME = "qualified_candidate_set_audit.md"
QUALIFIED_OCCURRENCES_FILENAME = "qualified_candidate_occurrences.csv"
ORDERABLE_OCCURRENCES_FILENAME = "orderable_candidate_occurrences.csv"
QUALIFIED_GROUPS_FILENAME = "qualified_candidate_groups.csv"
ORDERABLE_GROUPS_FILENAME = "orderable_candidate_groups.csv"
DAILY_COVERAGE_FILENAME = "qualified_candidate_daily_coverage.csv"
ACTUAL_TRADE_MATCHES_FILENAME = "qualified_candidate_actual_trade_matches.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11C research-only Qualified Candidate-set Coverage Audit；"
            "重播正式no-filter historical active-param OOS候選鏈，"
            "比較全部OOS breakout、qualified、orderable與actual trades"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--ranker-profile",
        default=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        choices=(STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,),
    )
    parser.add_argument(
        "--strategy-compare-dir",
        default=None,
        help="選填hard-filter strategy_compare輸出目錄；預設使用active 9A標準路徑",
    )
    parser.add_argument("--quiet", action="store_true", help="降低portfolio replay進度輸出")
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取JSON失敗: {path}｜{type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root必須是object: {path}")
    return payload


def _strategy_compare_dir(filter_id: str, explicit_dir: str | None) -> Path:
    if explicit_dir:
        path = Path(explicit_dir).expanduser().resolve()
    else:
        output_root = resolve_filter_model_output_dir(
            PROJECT_ROOT,
            filter_id,
            BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        )
        path = _first_existing_comparison_dir(
            output_root,
            comparison_mode=COMPARISON_MODE_HARD_FILTER,
        )
    if not path.is_dir():
        raise FileNotFoundError(f"找不到11C strategy_compare目錄: {path}")
    return path


def _ranker_dir(filter_id: str, ranker_profile: str) -> Path:
    return resolve_filter_model_output_dir(
        PROJECT_ROOT,
        filter_id,
        BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        ranker_profile,
    )


def _validate_strategy_metadata(metadata: dict[str, Any], *, filter_id: str) -> None:
    expected = {
        "comparison_mode": "hard-filter",
        "comparison_design": "historical_active_param_oos",
        "filter_id": filter_id,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if str(metadata.get(key) or "") != str(value)
    }
    if mismatches:
        raise ValueError(f"11C strategy_compare metadata不一致: {mismatches}")
    if not bool(metadata.get("lookahead_safe_active_param_schedule")):
        raise ValueError("11C只接受lookahead-safe historical active-param comparison")
    if bool(metadata.get("threshold_used_as_gate")) is not True:
        raise ValueError("11C來源必須是hard-filter controlled comparison")


def _load_ranker_scores(filter_id: str, ranker_profile: str) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    ranker_dir = _ranker_dir(filter_id, ranker_profile)
    score_path = ranker_dir / RANKER_SCORE_FILENAME
    report_path = ranker_dir / RANKER_REPORT_JSON_FILENAME
    if not score_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "11C需要既有11B research scores與report；請先執行 "
            f"`python apps/breakout_quality.py train-continuous-ranker --filter-id {filter_id}`"
        )
    report = _read_json(report_path)
    if str(report.get("experiment_profile") or "") != ranker_profile:
        raise ValueError("11C ranker report profile不一致")
    if bool((report.get("runtime_eligibility") or {}).get("eligible")):
        raise ValueError("11C預期11B維持research-only")
    score_manifest = ((report.get("artifacts") or {}).get("scores") or {})
    expected_hash = str(score_manifest.get("sha256") or "")
    if expected_hash and _sha256_file(score_path) != expected_hash:
        raise ValueError("11C偵測到11B scores SHA256不一致")
    scores = pd.read_csv(score_path, encoding="utf-8-sig")
    required = {
        "ticker", "date", "group_index", "label", "split",
        "target_raw_r", "target_daily_percentile", "model_score",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"11C ranker scores缺少欄位: {missing}")
    scores["ticker"] = scores["ticker"].astype(str)
    scores["date"] = pd.to_datetime(scores["date"], errors="raise").dt.strftime("%Y-%m-%d")
    for column in ("target_raw_r", "target_daily_percentile", "model_score"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    oos = scores[scores["split"].astype(str) == "oos"].copy()
    if oos.empty:
        raise ValueError("11C ranker scores沒有OOS rows")
    if bool(oos.duplicated(["ticker", "date"]).any()):
        raise ValueError("11C OOS ranker score ticker/date必須唯一")
    valid = np.isfinite(oos["target_raw_r"].to_numpy(dtype=np.float64)) & np.isfinite(
        oos["model_score"].to_numpy(dtype=np.float64)
    )
    if not bool(valid.all()):
        raise ValueError("11C OOS ranker scores含非有限target或score")
    return oos.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True), report, score_path


def _resolve_target_date(frame: pd.DataFrame) -> pd.Series:
    signal = frame.get("signal_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    candidate = frame.get("candidate_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    trade = frame.get("trade_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    target = signal.where(signal != "", candidate)
    target = target.where(target != "", trade)
    return pd.to_datetime(target, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")


def _attach_ranker_scores(occurrences: pd.DataFrame, oos_scores: pd.DataFrame, *, layer: str) -> pd.DataFrame:
    frame = occurrences.copy()
    frame["target_date"] = _resolve_target_date(frame)
    lookup = oos_scores.rename(columns={"date": "target_date"})[
        [
            "ticker", "target_date", "group_index", "label", "target_raw_r",
            "target_daily_percentile", "model_score",
        ]
    ]
    merged = frame.merge(lookup, how="left", on=["ticker", "target_date"], validate="many_to_one")
    merged.insert(0, "layer", str(layer))
    merged["target_match"] = np.isfinite(pd.to_numeric(merged["target_raw_r"], errors="coerce"))
    return merged.sort_values(
        ["trade_date", "ticker", "target_date", "candidate_type"],
        kind="mergesort",
    ).reset_index(drop=True)


def _unique_groups(matched_occurrences: pd.DataFrame, *, layer: str) -> pd.DataFrame:
    valid = matched_occurrences[matched_occurrences["target_match"]].copy()
    valid["occurrence_count"] = valid.groupby(["ticker", "target_date"])["ticker"].transform("size")
    unique = valid.sort_values(
        ["ticker", "target_date", "trade_date", "candidate_type"],
        kind="mergesort",
    ).drop_duplicates(["ticker", "target_date"], keep="first")
    unique["layer"] = str(layer)
    return unique.reset_index(drop=True)


def _decile_target_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "top_score_decile_target_mean": None,
            "bottom_score_decile_target_mean": None,
        }
    ordered = frame.sort_values("model_score", kind="mergesort")
    count = max(1, int(math.ceil(len(ordered) * 0.10)))
    return {
        "top_score_decile_target_mean": float(ordered.tail(count)["target_raw_r"].mean()),
        "bottom_score_decile_target_mean": float(ordered.head(count)["target_raw_r"].mean()),
    }


def _layer_metrics(
    group_frame: pd.DataFrame,
    *,
    occurrence_count: int,
    occurrence_date_count: int,
) -> dict[str, Any]:
    frame = group_frame.copy()
    if frame.empty:
        return {
            "occurrence_count": int(occurrence_count),
            "matched_unique_group_count": 0,
            "date_count": int(occurrence_date_count),
            "avg_occurrences_per_date": None,
            "global_spearman_score_vs_target": None,
            "rankable_date_count": 0,
            "rankable_date_rate": None,
            "mean_daily_spearman_score_vs_target": None,
            "pairwise_concordance_score_vs_target": None,
            **_decile_target_metrics(frame),
        }
    dates = pd.to_datetime(frame["target_date"], errors="raise").to_numpy()
    scores = frame["model_score"].to_numpy(dtype=np.float64)
    targets = frame["target_raw_r"].to_numpy(dtype=np.float64)
    daily = _daily_rank_metrics(dates, scores, targets)
    date_count = int(pd.Series(dates).nunique())
    return {
        "occurrence_count": int(occurrence_count),
        "matched_unique_group_count": int(len(frame)),
        "date_count": date_count,
        "avg_occurrences_per_date": (
            float(occurrence_count / occurrence_date_count) if occurrence_date_count else None
        ),
        "global_spearman_score_vs_target": _spearman(scores, targets),
        "rankable_date_count": int(daily["rankable_date_count"]),
        "rankable_date_rate": (
            float(daily["rankable_date_count"] / date_count) if date_count else None
        ),
        "mean_daily_spearman_score_vs_target": daily["mean_daily_spearman"],
        "median_daily_spearman_score_vs_target": daily["median_daily_spearman"],
        "pairwise_concordance_score_vs_target": daily["pairwise_concordance"],
        "comparable_pair_count": int(daily["comparable_pair_count"]),
        **_decile_target_metrics(frame),
    }


REPLAY_STABILITY_KEYS = (
    "total_return_pct",
    "max_drawdown_pct",
    "return_over_max_drawdown",
    "annual_return_pct",
    "trade_count",
    "final_equity",
    "avg_exposure_pct",
    "max_exposure_pct",
    "missed_buy_count",
    "missed_sell_count",
    "normal_trade_count",
    "extended_trade_count",
    "portfolio_total_r",
)


def _assert_replay_matches_strategy_summary(
    expected_summary: dict[str, Any],
    replay_payload: dict[str, Any],
) -> dict[str, Any]:
    replay_summary = summarize_strategy_scenario(replay_payload)
    mismatches: dict[str, dict[str, Any]] = {}
    for key in REPLAY_STABILITY_KEYS:
        expected = expected_summary.get(key)
        actual = replay_summary.get(key)
        if isinstance(expected, bool) or isinstance(actual, bool):
            same = expected is actual
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            same = math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-8)
        else:
            same = expected == actual
        if not same:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"11C candidate replay改變正式no-filter策略結果: {mismatches}")
    return {key: replay_summary.get(key) for key in REPLAY_STABILITY_KEYS}


def _signal_key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame.empty or not {"ticker", "target_date"}.issubset(frame.columns):
        return set()
    work = frame[["ticker", "target_date"]].copy()
    work["ticker"] = work["ticker"].astype(str).str.strip()
    work["target_date"] = work["target_date"].fillna("").astype(str).str.strip()
    work = work[(work["ticker"] != "") & (work["target_date"] != "")]
    return set(zip(work["ticker"], work["target_date"]))


def _actual_trade_metrics(actual: pd.DataFrame, oos_scores: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = actual.copy()
    required = {"ticker", "target_date", "target_raw_r", "r_multiple"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"11C actual trade matches缺少欄位: {missing}")
    frame["ticker"] = frame["ticker"].astype(str)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    frame["r_multiple"] = pd.to_numeric(frame["r_multiple"], errors="coerce")
    frame["target_raw_r"] = pd.to_numeric(frame["target_raw_r"], errors="coerce")
    lookup = oos_scores.rename(columns={"date": "target_date"})[["ticker", "target_date", "model_score"]]
    frame = frame.merge(lookup, how="left", on=["ticker", "target_date"], validate="many_to_one")
    valid = frame[
        np.isfinite(frame["r_multiple"].to_numpy(dtype=np.float64))
        & np.isfinite(frame["target_raw_r"].to_numpy(dtype=np.float64))
        & np.isfinite(frame["model_score"].to_numpy(dtype=np.float64))
    ].copy()
    if valid.empty:
        return {
            "trade_count": int(len(frame)),
            "matched_trade_count": 0,
            "coverage_rate": 0.0,
        }, frame
    r_values = valid["r_multiple"].to_numpy(dtype=np.float64)
    target_values = valid["target_raw_r"].to_numpy(dtype=np.float64)
    score_values = valid["model_score"].to_numpy(dtype=np.float64)
    count = max(1, int(math.ceil(len(valid) * 0.10)))
    by_target = valid.sort_values("target_raw_r", kind="mergesort")
    by_score = valid.sort_values("model_score", kind="mergesort")
    large = r_values >= 2.0
    median_target = float(np.median(target_values))
    median_score = float(np.median(score_values))
    metrics = {
        "trade_count": int(len(frame)),
        "matched_trade_count": int(len(valid)),
        "coverage_rate": float(len(valid) / len(frame)) if len(frame) else None,
        "spearman_target_vs_realized_r": _spearman(target_values, r_values),
        "spearman_score_vs_target": _spearman(score_values, target_values),
        "spearman_score_vs_realized_r": _spearman(score_values, r_values),
        "top_target_decile_average_r": float(by_target.tail(count)["r_multiple"].mean()),
        "bottom_target_decile_average_r": float(by_target.head(count)["r_multiple"].mean()),
        "top_score_decile_average_r": float(by_score.tail(count)["r_multiple"].mean()),
        "bottom_score_decile_average_r": float(by_score.head(count)["r_multiple"].mean()),
        "large_winner_count_r_ge_2": int(large.sum()),
        "large_winner_top_half_target_retention": (
            float((target_values[large] >= median_target).mean()) if bool(large.any()) else None
        ),
        "large_winner_top_half_score_retention": (
            float((score_values[large] >= median_score).mean()) if bool(large.any()) else None
        ),
    }
    return metrics, frame


def _daily_coverage(qualified: pd.DataFrame, orderable: pd.DataFrame) -> pd.DataFrame:
    def _summarize(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=[
                "trade_date", f"{prefix}_occurrences", f"{prefix}_unique_signals",
                f"{prefix}_matched_occurrences",
            ])
        work = frame.copy()
        work["matched"] = work["target_match"].astype(bool)
        work["signal_key"] = (
            work["ticker"].astype(str).str.strip()
            + "|"
            + work["target_date"].fillna("").astype(str).str.strip()
        )
        return (
            work.groupby("trade_date", sort=True)
            .agg(
                **{
                    f"{prefix}_occurrences": ("ticker", "size"),
                    f"{prefix}_unique_signals": ("signal_key", "nunique"),
                    f"{prefix}_matched_occurrences": ("matched", "sum"),
                }
            )
            .reset_index()
        )

    left = _summarize(qualified, "qualified")
    right = _summarize(orderable, "orderable")
    out = left.merge(right, how="outer", on="trade_date").fillna(0)
    count_columns = [column for column in out.columns if column != "trade_date"]
    for column in count_columns:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
    return out.sort_values("trade_date", kind="mergesort").reset_index(drop=True)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number * 100:.2f}%" if math.isfinite(number) else "-"


def render_markdown(payload: dict[str, Any]) -> str:
    layers = payload["layers"]
    trade = payload["actual_trade_alignment"]
    coverage = payload["coverage"]
    lines = [
        "# 11C Qualified Candidate-set Coverage Audit",
        "",
        "- 狀態：只讀診斷；本輪沒有訓練、選epoch、調threshold或修改runtime。",
        f"- Strategy compare：`{payload['strategy_compare']['path']}`",
        f"- Comparison period：`{payload['strategy_compare']['period']['start']}` ～ `{payload['strategy_compare']['period']['end']}`",
        f"- 11B Profile：`{payload['ranker_profile']}`（已淘汰模型，只供失敗歸因）",
        "",
        "## 1. 母體與覆蓋",
        "",
        "| Layer | Occurrences | Matched unique groups | Dates | Rankable dates | Score↔Target global ρ | Daily ρ | Pair concordance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title in (
        ("all_oos_breakouts", "All OOS breakouts"),
        ("qualified_candidates", "Qualified candidates"),
        ("orderable_candidates", "Orderable candidates"),
    ):
        row = layers[key]
        lines.append(
            f"| {title} | {int(row['occurrence_count']):,} | {int(row['matched_unique_group_count']):,} "
            f"| {int(row['date_count']):,} | {int(row['rankable_date_count']):,} ({_pct(row.get('rankable_date_rate'))}) "
            f"| {_fmt(row.get('global_spearman_score_vs_target'))} "
            f"| {_fmt(row.get('mean_daily_spearman_score_vs_target'))} "
            f"| {_fmt(row.get('pairwise_concordance_score_vs_target'))} |"
        )
    lines += [
        "",
        f"- Qualified unique group coverage vs all OOS：`{_pct(coverage.get('qualified_unique_group_coverage_vs_all_oos'))}`",
        f"- Qualified membership in all OOS：`{_pct(coverage.get('qualified_unique_group_membership_rate_in_all_oos'))}`",
        f"- Orderable unique group coverage vs qualified：`{_pct(coverage.get('orderable_unique_group_coverage_vs_qualified'))}`",
        f"- Orderable membership in qualified：`{_pct(coverage.get('orderable_unique_group_membership_rate_in_qualified'))}`",
        f"- Actual trade unique signal coverage vs qualified：`{_pct(coverage.get('actual_trade_unique_signal_coverage_vs_qualified'))}`",
        f"- Actual trade membership in qualified／orderable：`{_pct(coverage.get('actual_trade_unique_signal_membership_rate_in_qualified'))}` / `{_pct(coverage.get('actual_trade_unique_signal_membership_rate_in_orderable'))}`",
        f"- Qualified occurrence target-match coverage：`{_pct(coverage.get('qualified_occurrence_target_match_rate'))}`",
        "",
        "## 2. Score對Target的子集漂移",
        "",
        "| Layer | Top score decile target | Bottom score decile target | Top−Bottom |",
        "|---|---:|---:|---:|",
    ]
    for key, title in (
        ("all_oos_breakouts", "All OOS breakouts"),
        ("qualified_candidates", "Qualified candidates"),
        ("orderable_candidates", "Orderable candidates"),
    ):
        row = layers[key]
        top = row.get("top_score_decile_target_mean")
        bottom = row.get("bottom_score_decile_target_mean")
        delta = None if top is None or bottom is None else float(top) - float(bottom)
        lines.append(f"| {title} | {_fmt(top)} | {_fmt(bottom)} | {_fmt(delta)} |")
    lines += [
        "",
        "## 3. Actual Round-trip R",
        "",
        f"- 配對：`{trade.get('matched_trade_count', 0)}` / `{trade.get('trade_count', 0)}`；coverage `{_pct(trade.get('coverage_rate'))}`",
        f"- Spearman(Target, realized R)：`{_fmt(trade.get('spearman_target_vs_realized_r'))}`",
        f"- Spearman(11B Score, Target)：`{_fmt(trade.get('spearman_score_vs_target'))}`",
        f"- Spearman(11B Score, realized R)：`{_fmt(trade.get('spearman_score_vs_realized_r'))}`",
        f"- Target top／bottom decile realized R：`{_fmt(trade.get('top_target_decile_average_r'))}` / `{_fmt(trade.get('bottom_target_decile_average_r'))}`",
        f"- Score top／bottom decile realized R：`{_fmt(trade.get('top_score_decile_average_r'))}` / `{_fmt(trade.get('bottom_score_decile_average_r'))}`",
        f"- ≥2R贏家位於Target／Score上半部：`{_pct(trade.get('large_winner_top_half_target_retention'))}` / `{_pct(trade.get('large_winner_top_half_score_retention'))}`",
        "",
        "## 4. 判定邊界",
        "",
        "- 本audit只用固定OOS做迭代研究失敗歸因，不建立loss、sample weight、normalization或任何新模型參數。",
        "- Qualified candidate由正式portfolio replay產生，不另寫策略資格規則。",
        "- 是否建立qualified-candidate training profile，必須等本報表結果後另行決定。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    strategy_dir = _strategy_compare_dir(args.filter_id, args.strategy_compare_dir)
    comparison_path = strategy_dir / "strategy_comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"11C缺少strategy_comparison.json: {comparison_path}")
    comparison = _read_json(comparison_path)
    metadata = dict(comparison.get("metadata") or {})
    _validate_strategy_metadata(metadata, filter_id=args.filter_id)

    oos_scores, ranker_report, ranker_score_path = _load_ranker_scores(
        args.filter_id,
        args.ranker_profile,
    )
    replay_payload, qualified_raw, orderable_raw = run_no_filter_candidate_replay_from_metadata(
        metadata,
        quiet=bool(args.quiet),
    )
    expected_no_filter_summary = dict(comparison.get("no_filter") or {})
    replay_consistency = _assert_replay_matches_strategy_summary(
        expected_no_filter_summary,
        replay_payload,
    )
    expected_trade_count = int(expected_no_filter_summary.get("trade_count", 0) or 0)
    replay_trade_count = int(replay_consistency.get("trade_count", 0) or 0)

    qualified = _attach_ranker_scores(qualified_raw, oos_scores, layer="qualified")
    orderable = _attach_ranker_scores(orderable_raw, oos_scores, layer="orderable")
    qualified_groups = _unique_groups(qualified, layer="qualified")
    orderable_groups = _unique_groups(orderable, layer="orderable")

    all_oos_groups = oos_scores.rename(columns={"date": "target_date"}).copy()
    all_oos_groups["occurrence_count"] = 1
    all_metrics = _layer_metrics(
        all_oos_groups,
        occurrence_count=len(all_oos_groups),
        occurrence_date_count=int(all_oos_groups["target_date"].nunique()),
    )
    qualified_metrics = _layer_metrics(
        qualified_groups,
        occurrence_count=len(qualified),
        occurrence_date_count=int(qualified["trade_date"].nunique()) if len(qualified) else 0,
    )
    orderable_metrics = _layer_metrics(
        orderable_groups,
        occurrence_count=len(orderable),
        occurrence_date_count=int(orderable["trade_date"].nunique()) if len(orderable) else 0,
    )

    target_dir = resolve_continuous_target_dir(
        PROJECT_ROOT,
        args.filter_id,
        target_id=STRATEGY_ALIGNED_TARGET_ID,
    )
    trade_match_path = target_dir / TARGET_TRADE_MATCHES_CSV_FILENAME
    if not trade_match_path.is_file():
        raise FileNotFoundError(
            f"11C需要11A actual trade matches: {trade_match_path}；請先執行audit-continuous-target"
        )
    actual_raw = pd.read_csv(trade_match_path, encoding="utf-8-sig")
    actual_metrics, actual_matches = _actual_trade_metrics(actual_raw, oos_scores)
    all_oos_keys = _signal_key_set(all_oos_groups)
    qualified_keys = _signal_key_set(qualified_groups)
    orderable_keys = _signal_key_set(orderable_groups)
    actual_valid = actual_matches[
        np.isfinite(pd.to_numeric(actual_matches.get("model_score"), errors="coerce"))
    ].copy()
    actual_keys = _signal_key_set(actual_valid)
    qualified_in_all = qualified_keys & all_oos_keys
    orderable_in_qualified = orderable_keys & qualified_keys
    actual_in_qualified = actual_keys & qualified_keys
    actual_in_orderable = actual_keys & orderable_keys

    coverage = {
        "qualified_unique_group_coverage_vs_all_oos": (
            float(len(qualified_in_all) / len(all_oos_keys)) if all_oos_keys else None
        ),
        "qualified_unique_group_membership_rate_in_all_oos": (
            float(len(qualified_in_all) / len(qualified_keys)) if qualified_keys else None
        ),
        "orderable_unique_group_coverage_vs_qualified": (
            float(len(orderable_in_qualified) / len(qualified_keys)) if qualified_keys else None
        ),
        "orderable_unique_group_membership_rate_in_qualified": (
            float(len(orderable_in_qualified) / len(orderable_keys)) if orderable_keys else None
        ),
        "actual_trade_unique_signal_coverage_vs_qualified": (
            float(len(actual_in_qualified) / len(qualified_keys)) if qualified_keys else None
        ),
        "actual_trade_unique_signal_membership_rate_in_qualified": (
            float(len(actual_in_qualified) / len(actual_keys)) if actual_keys else None
        ),
        "actual_trade_unique_signal_membership_rate_in_orderable": (
            float(len(actual_in_orderable) / len(actual_keys)) if actual_keys else None
        ),
        "all_oos_unique_group_count": int(len(all_oos_keys)),
        "qualified_unique_group_count": int(len(qualified_keys)),
        "orderable_unique_group_count": int(len(orderable_keys)),
        "actual_trade_unique_signal_count": int(len(actual_keys)),
        "qualified_occurrence_target_match_rate": (
            float(qualified["target_match"].mean()) if len(qualified) else None
        ),
        "orderable_occurrence_target_match_rate": (
            float(orderable["target_match"].mean()) if len(orderable) else None
        ),
        "qualified_unmatched_occurrence_count": int((~qualified["target_match"]).sum()) if len(qualified) else 0,
        "orderable_unmatched_occurrence_count": int((~orderable["target_match"]).sum()) if len(orderable) else 0,
    }
    daily = _daily_coverage(qualified, orderable)

    output_dir = _ranker_dir(args.filter_id, args.ranker_profile) / AUDIT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    qualified_path = output_dir / QUALIFIED_OCCURRENCES_FILENAME
    orderable_path = output_dir / ORDERABLE_OCCURRENCES_FILENAME
    qualified_groups_path = output_dir / QUALIFIED_GROUPS_FILENAME
    orderable_groups_path = output_dir / ORDERABLE_GROUPS_FILENAME
    daily_path = output_dir / DAILY_COVERAGE_FILENAME
    actual_path = output_dir / ACTUAL_TRADE_MATCHES_FILENAME
    report_json_path = output_dir / AUDIT_JSON_FILENAME
    report_markdown_path = output_dir / AUDIT_MARKDOWN_FILENAME
    qualified.to_csv(qualified_path, index=False, encoding="utf-8-sig")
    orderable.to_csv(orderable_path, index=False, encoding="utf-8-sig")
    qualified_groups.to_csv(qualified_groups_path, index=False, encoding="utf-8-sig")
    orderable_groups.to_csv(orderable_groups_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    actual_matches.to_csv(actual_path, index=False, encoding="utf-8-sig")

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "11C Qualified Candidate-set Coverage Audit",
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "training_performed": False,
        "filter_id": args.filter_id,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "ranker_profile": args.ranker_profile,
        "ranker_status": ranker_report.get("status"),
        "ranker_score_path": str(ranker_score_path),
        "ranker_score_sha256": _sha256_file(ranker_score_path),
        "continuous_target_id": STRATEGY_ALIGNED_TARGET_ID,
        "strategy_compare": {
            "path": str(strategy_dir),
            "metadata_path": str(comparison_path),
            "metadata_sha256": _sha256_file(comparison_path),
            "comparison_design": metadata.get("comparison_design"),
            "param_source_kind": metadata.get("param_source_kind"),
            "param_selector": metadata.get("param_selector"),
            "max_positions": metadata.get("max_positions"),
            "enable_rotation": metadata.get("enable_rotation"),
            "period": metadata.get("comparison_period"),
            "expected_trade_count": expected_trade_count,
            "replay_trade_count": replay_trade_count,
            "replay_consistency": replay_consistency,
        },
        "layers": {
            "all_oos_breakouts": all_metrics,
            "qualified_candidates": qualified_metrics,
            "orderable_candidates": orderable_metrics,
        },
        "coverage": coverage,
        "actual_trade_alignment": actual_metrics,
        "artifacts": {
            "qualified_occurrences": {"path": str(qualified_path), "sha256": _sha256_file(qualified_path)},
            "orderable_occurrences": {"path": str(orderable_path), "sha256": _sha256_file(orderable_path)},
            "qualified_groups": {"path": str(qualified_groups_path), "sha256": _sha256_file(qualified_groups_path)},
            "orderable_groups": {"path": str(orderable_groups_path), "sha256": _sha256_file(orderable_groups_path)},
            "daily_coverage": {"path": str(daily_path), "sha256": _sha256_file(daily_path)},
            "actual_trade_matches": {"path": str(actual_path), "sha256": _sha256_file(actual_path)},
        },
        "interpretation_contract": {
            "research_only": True,
            "oos_iterative_research_evidence_only": True,
            "qualified_candidates_from_canonical_portfolio_replay": True,
            "no_training_epoch_selection_threshold_or_runtime_change": True,
            "audit_does_not_authorize_new_model": True,
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    write_json(report_json_path, payload)
    report_markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    print("11C qualified candidate-set audit完成")
    print(
        f"all_oos={all_metrics['matched_unique_group_count']:,} "
        f"qualified={qualified_metrics['matched_unique_group_count']:,} "
        f"orderable={orderable_metrics['matched_unique_group_count']:,} "
        f"trades={actual_metrics.get('matched_trade_count', 0):,}"
    )
    print(
        "- Score↔Target ρ: "
        f"all={_fmt(all_metrics.get('global_spearman_score_vs_target'))} "
        f"qualified={_fmt(qualified_metrics.get('global_spearman_score_vs_target'))} "
        f"orderable={_fmt(orderable_metrics.get('global_spearman_score_vs_target'))}"
    )
    print(
        "- Actual R: "
        f"Target↔R={_fmt(actual_metrics.get('spearman_target_vs_realized_r'))} "
        f"Score↔R={_fmt(actual_metrics.get('spearman_score_vs_realized_r'))}"
    )
    print(f"已輸出: {report_markdown_path}")
    print(f"已輸出: {report_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
