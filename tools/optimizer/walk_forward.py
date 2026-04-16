from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from statistics import median
from typing import Iterable

import pandas as pd

from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_stats import calc_portfolio_score
from core.runtime_utils import get_taipei_now

WF_MIN_TRAIN_YEARS = 8
WF_TEST_WINDOW_MONTHS = 6
WF_REGIME_UP_THRESHOLD_PCT = 8.0
WF_REGIME_DOWN_THRESHOLD_PCT = -8.0
WF_MIN_WINDOW_BARS = 20


def _safe_median(values: Iterable[float]) -> float:
    normalized = [float(value) for value in values]
    return float(median(normalized)) if normalized else 0.0


def _safe_min(values: Iterable[float]) -> float:
    normalized = [float(value) for value in values]
    return float(min(normalized)) if normalized else 0.0


def _safe_max(values: Iterable[float]) -> float:
    normalized = [float(value) for value in values]
    return float(max(normalized)) if normalized else 0.0


def _align_to_half_year_boundary(candidate_start: pd.Timestamp) -> pd.Timestamp:
    candidate_start = pd.Timestamp(candidate_start).normalize()
    jan_boundary = pd.Timestamp(year=candidate_start.year, month=1, day=1)
    jul_boundary = pd.Timestamp(year=candidate_start.year, month=7, day=1)
    if candidate_start <= jan_boundary:
        return jan_boundary
    if candidate_start <= jul_boundary:
        return jul_boundary
    return pd.Timestamp(year=candidate_start.year + 1, month=1, day=1)


def _build_window_label(start_ts: pd.Timestamp) -> str:
    half = 1 if int(start_ts.month) <= 6 else 2
    return f"{int(start_ts.year)}H{half}"


def build_walk_forward_windows(
    sorted_dates,
    *,
    min_train_years: int = WF_MIN_TRAIN_YEARS,
    test_window_months: int = WF_TEST_WINDOW_MONTHS,
    min_window_bars: int = WF_MIN_WINDOW_BARS,
    train_start_year: int | None = None,
):
    if not sorted_dates:
        return []

    sorted_timestamps = [pd.Timestamp(dt).normalize() for dt in sorted_dates]
    first_market_date = sorted_timestamps[0]
    last_date = sorted_timestamps[-1]
    if train_start_year is None:
        first_date = first_market_date
    else:
        requested_start = pd.Timestamp(year=int(train_start_year), month=1, day=1)
        first_date = max(first_market_date, requested_start)
    first_oos_candidate = _align_to_half_year_boundary(first_date + pd.DateOffset(years=int(min_train_years)))

    windows = []
    oos_start = first_oos_candidate
    while oos_start <= last_date:
        oos_end = (oos_start + pd.DateOffset(months=int(test_window_months))) - pd.Timedelta(days=1)
        window_dates = [dt for dt in sorted_dates if oos_start <= pd.Timestamp(dt).normalize() <= oos_end]
        train_start = first_date
        train_end = oos_start - pd.Timedelta(days=1)
        if len(window_dates) >= int(min_window_bars) and train_start <= train_end:
            windows.append(
                {
                    "label": _build_window_label(oos_start),
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "oos_start": oos_start.strftime("%Y-%m-%d"),
                    "oos_end": min(pd.Timestamp(oos_end), last_date).strftime("%Y-%m-%d"),
                    "window_dates": window_dates,
                }
            )
        oos_start = oos_start + pd.DateOffset(months=int(test_window_months))

    return windows


def classify_benchmark_regime(benchmark_return_pct: float) -> str:
    benchmark_return_pct = float(benchmark_return_pct)
    if benchmark_return_pct >= WF_REGIME_UP_THRESHOLD_PCT:
        return "up"
    if benchmark_return_pct <= WF_REGIME_DOWN_THRESHOLD_PCT:
        return "down"
    return "flat"


def evaluate_walk_forward(
    *,
    all_dfs_fast,
    all_trade_logs,
    sorted_dates,
    params,
    max_positions,
    enable_rotation,
    benchmark_ticker: str = "0050",
    min_train_years: int = WF_MIN_TRAIN_YEARS,
    test_window_months: int = WF_TEST_WINDOW_MONTHS,
    train_start_year: int | None = None,
):
    benchmark_data = all_dfs_fast.get(str(benchmark_ticker), None)
    windows = build_walk_forward_windows(
        sorted_dates,
        min_train_years=min_train_years,
        test_window_months=test_window_months,
        train_start_year=train_start_year,
    )
    rows = []

    for window in windows:
        window_dates = window["window_dates"]
        if not window_dates:
            continue
        (
            ret_pct,
            mdd,
            trade_count,
            final_eq,
            avg_exp,
            max_exp,
            bm_ret,
            bm_mdd,
            win_rate,
            pf_ev,
            pf_payoff,
            total_missed,
            total_missed_sells,
            r_sq,
            m_win_rate,
            bm_r_sq,
            bm_m_win_rate,
            normal_trade_count,
            extended_trade_count,
            annual_trades,
            reserved_buy_fill_rate,
            annual_return_pct,
            bm_annual_return_pct,
        ) = run_portfolio_timeline(
            all_dfs_fast,
            all_trade_logs,
            window_dates,
            pd.Timestamp(window_dates[0]).year,
            params,
            max_positions,
            enable_rotation,
            benchmark_ticker=benchmark_ticker,
            benchmark_data=benchmark_data,
            is_training=True,
            profile_stats=None,
            verbose=False,
        )
        window_score = calc_portfolio_score(
            ret_pct,
            mdd,
            m_win_rate,
            r_sq,
            annual_return_pct=annual_return_pct,
        )
        regime = classify_benchmark_regime(bm_ret)
        rows.append(
            {
                "label": window["label"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "oos_start": window["oos_start"],
                "oos_end": window["oos_end"],
                "regime": regime,
                "window_score": float(window_score),
                "ret_pct": float(ret_pct),
                "annual_return_pct": float(annual_return_pct),
                "mdd": float(mdd),
                "trade_count": int(trade_count),
                "annual_trades": float(annual_trades),
                "reserved_buy_fill_rate": float(reserved_buy_fill_rate),
                "benchmark_return_pct": float(bm_ret),
                "benchmark_annual_return_pct": float(bm_annual_return_pct),
                "benchmark_mdd": float(bm_mdd),
                "win_rate": float(win_rate),
                "pf_ev": float(pf_ev),
                "r_squared": float(r_sq),
                "monthly_win_rate": float(m_win_rate),
                "final_equity": float(final_eq),
                "avg_exposure": float(avg_exp),
                "max_exposure": float(max_exp),
                "missed_buys": int(total_missed),
                "missed_sells": int(total_missed_sells),
                "normal_trades": int(normal_trade_count),
                "extended_trades": int(extended_trade_count),
            }
        )

    regime_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        regime_groups[str(row["regime"])].append(row)

    regime_summary = {}
    for regime_name in ("up", "flat", "down"):
        grouped_rows = regime_groups.get(regime_name, [])
        regime_summary[regime_name] = {
            "window_count": int(len(grouped_rows)),
            "median_score": _safe_median(row["window_score"] for row in grouped_rows),
            "median_ret_pct": _safe_median(row["ret_pct"] for row in grouped_rows),
            "worst_ret_pct": _safe_min(row["ret_pct"] for row in grouped_rows),
            "max_mdd": _safe_max(row["mdd"] for row in grouped_rows),
        }

    summary = {
        "window_count": int(len(rows)),
        "min_train_years": int(min_train_years),
        "train_start_year": None if train_start_year is None else int(train_start_year),
        "test_window_months": int(test_window_months),
        "regime_up_threshold_pct": float(WF_REGIME_UP_THRESHOLD_PCT),
        "regime_down_threshold_pct": float(WF_REGIME_DOWN_THRESHOLD_PCT),
        "median_window_score": _safe_median(row["window_score"] for row in rows),
        "median_ret_pct": _safe_median(row["ret_pct"] for row in rows),
        "worst_ret_pct": _safe_min(row["ret_pct"] for row in rows),
        "max_mdd": _safe_max(row["mdd"] for row in rows),
        "median_annual_trades": _safe_median(row["annual_trades"] for row in rows),
        "median_fill_rate": _safe_median(row["reserved_buy_fill_rate"] for row in rows),
    }

    return {
        "summary": summary,
        "regime_summary": regime_summary,
        "windows": rows,
    }


def _format_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def write_walk_forward_report(
    *,
    output_dir: str,
    params_payload: dict,
    dataset_label: str,
    report: dict,
    best_trial_number: int | None,
    source_db_path: str,
    session_ts: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    resolved_session_ts = session_ts or get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"walk_forward_report_{resolved_session_ts}"
    json_path = os.path.join(output_dir, f"{base_name}.json")
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    md_path = os.path.join(output_dir, f"{base_name}.md")

    payload = {
        "meta": {
            "dataset_label": str(dataset_label),
            "best_trial_number": None if best_trial_number is None else int(best_trial_number),
            "source_db_path": str(source_db_path),
            "session_ts": str(resolved_session_ts),
        },
        "params": dict(params_payload),
        "summary": dict(report.get("summary") or {}),
        "regime_summary": dict(report.get("regime_summary") or {}),
        "windows": list(report.get("windows") or []),
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    windows = list(report.get("windows") or [])
    if windows:
        fieldnames = list(windows[0].keys())
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(windows)
    else:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label", "regime", "window_score", "ret_pct", "mdd", "annual_trades", "reserved_buy_fill_rate"])

    summary = payload["summary"]
    regime_summary = payload["regime_summary"]
    lines = [
        "# Walk-Forward 驗證報表（MVP）",
        "",
        f"- 資料集：{dataset_label}",
        f"- 最佳 trial：{payload['meta']['best_trial_number'] if payload['meta']['best_trial_number'] is not None else 'N/A'}",
        f"- 記憶庫：`{source_db_path}`",
        f"- 產生時間：{resolved_session_ts}",
        "",
        "## Summary",
        "",
        "| 指標 | 數值 |",
        "|---|---:|",
        f"| 視窗數 | {int(summary.get('window_count', 0))} |",
        f"| 視窗分數中位數 | {float(summary.get('median_window_score', 0.0)):.3f} |",
        f"| 視窗報酬率中位數 | {_format_pct(summary.get('median_ret_pct', 0.0))} |",
        f"| 最差視窗報酬率 | {_format_pct(summary.get('worst_ret_pct', 0.0))} |",
        f"| 最大視窗 MDD | {_format_pct(summary.get('max_mdd', 0.0))} |",
        f"| 年化交易次數中位數 | {float(summary.get('median_annual_trades', 0.0)):.2f} |",
        f"| 買進成交率中位數 | {_format_pct(summary.get('median_fill_rate', 0.0))} |",
        "",
        "## Regime Summary",
        "",
        "| Regime | 視窗數 | 分數中位數 | 報酬率中位數 | 最差報酬率 | 最大 MDD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for regime_name in ("up", "flat", "down"):
        row = regime_summary.get(regime_name, {})
        lines.append(
            f"| {regime_name} | {int(row.get('window_count', 0))} | {float(row.get('median_score', 0.0)):.3f} | "
            f"{_format_pct(row.get('median_ret_pct', 0.0))} | {_format_pct(row.get('worst_ret_pct', 0.0))} | {_format_pct(row.get('max_mdd', 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## OOS Windows",
            "",
            "| 視窗 | OOS 區間 | Regime | 分數 | 報酬率 | MDD | 年化交易次數 | 買進成交率 | 0050 報酬率 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in windows:
        lines.append(
            f"| {row['label']} | {row['oos_start']} ~ {row['oos_end']} | {row['regime']} | {float(row['window_score']):.3f} | "
            f"{_format_pct(row['ret_pct'])} | {_format_pct(row['mdd'])} | {float(row['annual_trades']):.2f} | "
            f"{_format_pct(row['reserved_buy_fill_rate'])} | {_format_pct(row['benchmark_return_pct'])} |"
        )

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "md_path": md_path,
    }
