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
WF_GATE_MIN_MEDIAN_SCORE = 0.0
WF_GATE_MIN_WORST_RET_PCT = -8.0
WF_GATE_MIN_FLAT_MEDIAN_SCORE = 0.0


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


def classify_benchmark_regime(benchmark_return_pct: float, *, regime_up_threshold_pct: float = WF_REGIME_UP_THRESHOLD_PCT, regime_down_threshold_pct: float = WF_REGIME_DOWN_THRESHOLD_PCT) -> str:
    benchmark_return_pct = float(benchmark_return_pct)
    if benchmark_return_pct >= float(regime_up_threshold_pct):
        return "up"
    if benchmark_return_pct <= float(regime_down_threshold_pct):
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
    regime_up_threshold_pct: float = WF_REGIME_UP_THRESHOLD_PCT,
    regime_down_threshold_pct: float = WF_REGIME_DOWN_THRESHOLD_PCT,
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
        regime = classify_benchmark_regime(
            bm_ret,
            regime_up_threshold_pct=regime_up_threshold_pct,
            regime_down_threshold_pct=regime_down_threshold_pct,
        )
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
        "regime_up_threshold_pct": float(regime_up_threshold_pct),
        "regime_down_threshold_pct": float(regime_down_threshold_pct),
        "median_window_score": _safe_median(row["window_score"] for row in rows),
        "median_ret_pct": _safe_median(row["ret_pct"] for row in rows),
        "worst_ret_pct": _safe_min(row["ret_pct"] for row in rows),
        "max_mdd": _safe_max(row["mdd"] for row in rows),
        "median_annual_trades": _safe_median(row["annual_trades"] for row in rows),
        "median_fill_rate": _safe_median(row["reserved_buy_fill_rate"] for row in rows),
    }
    upgrade_gate = build_upgrade_gate_assessment(summary=summary, regime_summary=regime_summary)

    return {
        "summary": summary,
        "regime_summary": regime_summary,
        "upgrade_gate": upgrade_gate,
        "windows": rows,
    }


def _build_gate_check(*, name: str, actual: float | int | bool, threshold: str, passed: bool, severity: str, note: str) -> dict:
    return {
        "name": str(name),
        "actual": actual,
        "threshold": str(threshold),
        "passed": bool(passed),
        "severity": str(severity),
        "note": str(note),
    }




def build_upgrade_gate_assessment(*, summary: dict, regime_summary: dict) -> dict:
    median_window_score = float(summary.get("median_window_score", 0.0) or 0.0)
    worst_ret_pct = float(summary.get("worst_ret_pct", 0.0) or 0.0)
    flat_row = dict(regime_summary.get("flat") or {})
    down_row = dict(regime_summary.get("down") or {})
    flat_window_count = int(flat_row.get("window_count", 0) or 0)
    flat_median_score = float(flat_row.get("median_score", 0.0) or 0.0)
    down_window_count = int(down_row.get("window_count", 0) or 0)

    quality_checks = [
        _build_gate_check(
            name="median_window_score",
            actual=median_window_score,
            threshold="> 0",
            passed=median_window_score > WF_GATE_MIN_MEDIAN_SCORE,
            severity="quality",
            note="整體 OOS 典型視窗需為正分。",
        ),
        _build_gate_check(
            name="worst_ret_pct",
            actual=worst_ret_pct,
            threshold=f">= {WF_GATE_MIN_WORST_RET_PCT:.1f}%",
            passed=worst_ret_pct >= WF_GATE_MIN_WORST_RET_PCT,
            severity="quality",
            note="避免單一 OOS 視窗災難性失真。",
        ),
        _build_gate_check(
            name="flat_median_score",
            actual=flat_median_score,
            threshold=">= 0 (當 flat 視窗存在時)",
            passed=(flat_window_count == 0) or (flat_median_score >= WF_GATE_MIN_FLAT_MEDIAN_SCORE),
            severity="quality",
            note="盤整期不可連續結構性失分。",
        ),
    ]
    coverage_checks = [
        _build_gate_check(
            name="down_regime_coverage",
            actual=down_window_count,
            threshold=">= 1",
            passed=down_window_count >= 1,
            severity="coverage",
            note="若無 down 視窗，只能視為 regime 證據不足。",
        ),
    ]
    quality_pass = all(bool(check["passed"]) for check in quality_checks)
    coverage_pass = all(bool(check["passed"]) for check in coverage_checks)
    quality_status = "pass" if quality_pass else "fail"
    coverage_status = "pass" if coverage_pass else "watch"
    if quality_pass and coverage_pass:
        status = "pass"
    elif quality_pass:
        status = "watch"
    else:
        status = "fail"
    recommendation = {
        "pass": "可列為候選升版版本，且具基本跨盤勢證據。",
        "watch": "品質門檻已過，但 regime 覆蓋仍有限，只能視為有限證據。",
        "fail": "暫不建議僅依此報表升版；應先改善品質門檻失敗項目。",
    }[status]
    return {
        "status": status,
        "recommended_for_promotion": bool(quality_pass and coverage_pass),
        "cross_regime_claim_allowed": bool(coverage_pass),
        "quality_gate": {"status": quality_status, "checks": quality_checks},
        "coverage_gate": {"status": coverage_status, "checks": coverage_checks},
        "checks": quality_checks + coverage_checks,
        "recommendation": recommendation,
    }

def _format_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def _format_gate_actual(name: str, actual) -> str:
    if name in ("down_regime_coverage", "down_regime_coverage_vs_champion"):
        return str(int(actual))
    if "score" in name:
        return f"{float(actual):.3f}"
    if "trades" in name:
        return f"{float(actual):.2f}"
    return _format_pct(actual)


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

    summary = dict(report.get("summary") or {})
    regime_summary = dict(report.get("regime_summary") or {})
    upgrade_gate = dict(report.get("upgrade_gate") or build_upgrade_gate_assessment(summary=summary, regime_summary=regime_summary))
    payload = {
        "meta": {
            "dataset_label": str(dataset_label),
            "best_trial_number": None if best_trial_number is None else int(best_trial_number),
            "source_db_path": str(source_db_path),
            "session_ts": str(resolved_session_ts),
        },
        "params": dict(params_payload),
        "summary": summary,
        "regime_summary": regime_summary,
        "upgrade_gate": upgrade_gate,
        "windows": list(report.get("windows") or []),
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    windows = list(payload.get("windows") or [])
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
        "## 品質 / 覆蓋 Gate（MVP，僅報表判讀，不阻擋匯出）",
        "",
        f"- 狀態：`{upgrade_gate.get('status', 'fail')}`",
        f"- 建議：{upgrade_gate.get('recommendation', 'N/A')}",
        f"- 可宣稱跨盤勢穩健：{'是' if upgrade_gate.get('cross_regime_claim_allowed', False) else '否'}",
        "",
        "| Gate | 檢查項目 | 實際值 | 門檻 | 結果 | 說明 |",
        "|---|---|---:|---:|---|---|",
    ]
    for gate_name, gate_payload in (("quality", upgrade_gate.get("quality_gate") or {}), ("coverage", upgrade_gate.get("coverage_gate") or {})):
        for check in list(gate_payload.get("checks") or []):
            lines.append(
                f"| {gate_name} | {check['name']} | {_format_gate_actual(str(check['name']), check.get('actual'))} | {check.get('threshold', '')} | "
                f"{'PASS' if check.get('passed') else ('WATCH' if gate_name == 'coverage' else 'FAIL')} | {check.get('note', '')} |"
            )

    lines.extend(
        [
            "",
            "## Regime Summary",
            "",
            "| Regime | 視窗數 | 分數中位數 | 報酬率中位數 | 最差報酬率 | 最大 MDD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
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



def _parse_iso_date(value) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(str(value))
    except Exception as exc:
        _ = exc
        return None


def build_oos_total_performance(rows: list[dict], *, ret_key: str) -> dict:
    ordered_rows = sorted(list(rows or []), key=lambda row: str(row.get("oos_start") or ""))
    if not ordered_rows:
        return {
            "window_count": 0,
            "linked_total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "positive_window_rate": 0.0,
            "ending_equity_factor": 1.0,
            "oos_start": "",
            "oos_end": "",
        }

    equity = 1.0
    peak = 1.0
    max_drawdown_pct = 0.0
    positive_windows = 0
    start_ts = _parse_iso_date(ordered_rows[0].get("oos_start"))
    end_ts = _parse_iso_date(ordered_rows[-1].get("oos_end"))

    for row in ordered_rows:
        ret_pct = _safe_float(row.get(ret_key, 0.0))
        if ret_pct > 0.0:
            positive_windows += 1
        equity *= 1.0 + (ret_pct / 100.0)
        peak = max(peak, equity)
        if peak > 0.0:
            drawdown_pct = (peak - equity) / peak * 100.0
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

    years = 0.0
    if start_ts is not None and end_ts is not None:
        day_span = max(1, int((end_ts - start_ts).days) + 1)
        years = day_span / 365.25
    annualized_return_pct = 0.0
    if years > 0.0 and equity > 0.0:
        annualized_return_pct = (equity ** (1.0 / years) - 1.0) * 100.0

    return {
        "window_count": int(len(ordered_rows)),
        "linked_total_return_pct": (equity - 1.0) * 100.0,
        "annualized_return_pct": float(annualized_return_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "positive_window_rate": float(positive_windows / len(ordered_rows) * 100.0),
        "ending_equity_factor": float(equity),
        "oos_start": "" if start_ts is None else start_ts.strftime("%Y-%m-%d"),
        "oos_end": "" if end_ts is None else end_ts.strftime("%Y-%m-%d"),
    }


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _window_row_by_label(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("label")): dict(row) for row in rows}




def build_compare_assessment(*, champion_report: dict, challenger_report: dict) -> dict:
    champion_summary = dict(champion_report.get("summary") or {})
    challenger_summary = dict(challenger_report.get("summary") or {})
    champion_regime = dict(champion_report.get("regime_summary") or {})
    challenger_regime = dict(challenger_report.get("regime_summary") or {})

    champion_flat_score = _safe_float((champion_regime.get("flat") or {}).get("median_score", 0.0))
    challenger_flat_score = _safe_float((challenger_regime.get("flat") or {}).get("median_score", 0.0))
    champion_down_count = int((champion_regime.get("down") or {}).get("window_count", 0) or 0)
    challenger_down_count = int((challenger_regime.get("down") or {}).get("window_count", 0) or 0)

    quality_checks = [
        _build_gate_check(
            name="median_window_score_vs_champion",
            actual=_safe_float(challenger_summary.get("median_window_score", 0.0)),
            threshold=f">= {_safe_float(champion_summary.get('median_window_score', 0.0)):.3f}",
            passed=_safe_float(challenger_summary.get("median_window_score", 0.0)) >= _safe_float(champion_summary.get("median_window_score", 0.0)),
            severity="quality",
            note="候選版整體 OOS 典型視窗分數不得低於現役版。",
        ),
        _build_gate_check(
            name="flat_median_score_vs_champion",
            actual=challenger_flat_score,
            threshold=f">= {champion_flat_score:.3f}",
            passed=challenger_flat_score >= champion_flat_score,
            severity="quality",
            note="候選版盤整期中位分數不得低於現役版。",
        ),
        _build_gate_check(
            name="worst_ret_pct_vs_champion",
            actual=_safe_float(challenger_summary.get("worst_ret_pct", 0.0)),
            threshold=f">= {(_safe_float(champion_summary.get('worst_ret_pct', 0.0)) - 1.0):.2f}%",
            passed=_safe_float(challenger_summary.get("worst_ret_pct", 0.0)) >= (_safe_float(champion_summary.get("worst_ret_pct", 0.0)) - 1.0),
            severity="quality",
            note="候選版最差視窗報酬不可比現役版惡化超過 1%。",
        ),
        _build_gate_check(
            name="max_mdd_vs_champion",
            actual=_safe_float(challenger_summary.get("max_mdd", 0.0)),
            threshold=f"<= {(_safe_float(champion_summary.get('max_mdd', 0.0)) + 2.0):.2f}%",
            passed=_safe_float(challenger_summary.get("max_mdd", 0.0)) <= (_safe_float(champion_summary.get("max_mdd", 0.0)) + 2.0),
            severity="quality",
            note="候選版最大視窗 MDD 不可比現役版惡化超過 2%。",
        ),
    ]
    coverage_checks = [
        _build_gate_check(
            name="down_regime_coverage_vs_champion",
            actual=challenger_down_count,
            threshold=f">= {champion_down_count}",
            passed=challenger_down_count >= champion_down_count,
            severity="coverage",
            note="候選版 down 視窗覆蓋不可少於現役版。",
        ),
    ]
    quality_pass = all(bool(check["passed"]) for check in quality_checks)
    coverage_pass = all(bool(check["passed"]) for check in coverage_checks)
    quality_status = "pass" if quality_pass else "fail"
    coverage_status = "pass" if coverage_pass else "watch"
    if quality_pass and coverage_pass:
        status = "pass"
    elif quality_pass:
        status = "watch"
    else:
        status = "fail"

    recommendation = {
        "pass": "候選版整體優於現役版，可列為升版候選。",
        "watch": "候選版品質優於現役版，但 regime 覆蓋證據仍有限，建議審慎升版。",
        "fail": "候選版尚未穩定優於現役版，暫不建議升版。",
    }[status]

    return {
        "status": status,
        "recommended_for_promotion": bool(quality_pass and coverage_pass),
        "quality_gate": {"status": quality_status, "checks": quality_checks},
        "coverage_gate": {"status": coverage_status, "checks": coverage_checks},
        "checks": quality_checks + coverage_checks,
        "recommendation": recommendation,
    }



def _link_total_return_pct(rows: list[dict]) -> float:
    equity = 1.0
    for row in rows:
        equity *= 1.0 + (float(row.get("ret_pct", 0.0) or 0.0) / 100.0)
    return (equity - 1.0) * 100.0


def _annualized_return_pct(*, linked_total_return_pct: float, start_date: str | None, end_date: str | None) -> float:
    if not start_date or not end_date:
        return 0.0
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    days = max((end_ts - start_ts).days + 1, 1)
    years = days / 365.25
    if years <= 0:
        return float(linked_total_return_pct)
    growth = 1.0 + float(linked_total_return_pct) / 100.0
    if growth <= 0:
        return -100.0
    return (growth ** (1.0 / years) - 1.0) * 100.0


def _positive_window_rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    positives = sum(1 for row in rows if float(row.get(key, 0.0) or 0.0) > 0.0)
    return positives * 100.0 / len(rows)


def _metric_triplet(*, champion, challenger, benchmark=None) -> dict:
    payload = {
        "champion": float(champion),
        "challenger": float(challenger),
        "delta": float(challenger) - float(champion),
    }
    if benchmark is not None:
        payload.update({
            "benchmark": float(benchmark),
            "delta_vs_champion": float(challenger) - float(champion),
            "delta_vs_benchmark": float(challenger) - float(benchmark),
        })
    return payload


def build_walk_forward_compare_payload(*, champion_payload: dict, champion_report: dict, challenger_payload: dict, challenger_report: dict, dataset_label: str, source_db_path: str, session_ts: str | None = None) -> dict:
    champion_windows = list((champion_report or {}).get("windows") or [])
    challenger_windows = list((challenger_report or {}).get("windows") or [])
    champion_summary = dict((champion_report or {}).get("summary") or {})
    challenger_summary = dict((challenger_report or {}).get("summary") or {})
    champion_regime = dict((champion_report or {}).get("regime_summary") or {})
    challenger_regime = dict((challenger_report or {}).get("regime_summary") or {})

    champion_by_label = {str(row.get("label")): dict(row) for row in champion_windows}
    challenger_by_label = {str(row.get("label")): dict(row) for row in challenger_windows}
    ordered_labels = sorted(set(champion_by_label) | set(challenger_by_label), key=lambda x: (int(x[:4]), x[-2:]))
    window_compare_rows = []
    for label in ordered_labels:
        c = dict(champion_by_label.get(label) or {})
        h = dict(challenger_by_label.get(label) or {})
        regime = str(h.get("regime") or c.get("regime") or "flat")
        oos_start = str(h.get("oos_start") or c.get("oos_start") or "")
        oos_end = str(h.get("oos_end") or c.get("oos_end") or "")
        row = {
            "label": label,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "regime": regime,
            "champion_window_score": float(c.get("window_score", 0.0) or 0.0),
            "challenger_window_score": float(h.get("window_score", 0.0) or 0.0),
            "delta_window_score": float(h.get("window_score", 0.0) or 0.0) - float(c.get("window_score", 0.0) or 0.0),
            "champion_ret_pct": float(c.get("ret_pct", 0.0) or 0.0),
            "challenger_ret_pct": float(h.get("ret_pct", 0.0) or 0.0),
            "delta_ret_pct": float(h.get("ret_pct", 0.0) or 0.0) - float(c.get("ret_pct", 0.0) or 0.0),
            "champion_mdd": float(c.get("mdd", 0.0) or 0.0),
            "challenger_mdd": float(h.get("mdd", 0.0) or 0.0),
            "delta_mdd": float(h.get("mdd", 0.0) or 0.0) - float(c.get("mdd", 0.0) or 0.0),
        }
        window_compare_rows.append(row)

    flat_champion = dict(champion_regime.get("flat") or {})
    flat_challenger = dict(challenger_regime.get("flat") or {})
    down_champion = dict(champion_regime.get("down") or {})
    down_challenger = dict(challenger_regime.get("down") or {})

    summary_compare = {
        "median_window_score": _metric_triplet(champion=champion_summary.get("median_window_score", 0.0), challenger=challenger_summary.get("median_window_score", 0.0)),
        "median_ret_pct": _metric_triplet(champion=champion_summary.get("median_ret_pct", 0.0), challenger=challenger_summary.get("median_ret_pct", 0.0)),
        "worst_ret_pct": _metric_triplet(champion=champion_summary.get("worst_ret_pct", 0.0), challenger=challenger_summary.get("worst_ret_pct", 0.0)),
        "max_mdd": _metric_triplet(champion=champion_summary.get("max_mdd", 0.0), challenger=challenger_summary.get("max_mdd", 0.0)),
        "median_annual_trades": _metric_triplet(champion=champion_summary.get("median_annual_trades", 0.0), challenger=challenger_summary.get("median_annual_trades", 0.0)),
        "median_fill_rate": _metric_triplet(champion=champion_summary.get("median_fill_rate", 0.0), challenger=challenger_summary.get("median_fill_rate", 0.0)),
        "flat_median_score": _metric_triplet(champion=flat_champion.get("median_score", 0.0), challenger=flat_challenger.get("median_score", 0.0)),
        "down_window_count": {
            "champion": int(down_champion.get("window_count", 0) or 0),
            "challenger": int(down_challenger.get("window_count", 0) or 0),
            "delta": int(down_challenger.get("window_count", 0) or 0) - int(down_champion.get("window_count", 0) or 0),
        },
    }

    regime_compare = {}
    for regime_name in ("up", "flat", "down"):
        c = dict(champion_regime.get(regime_name) or {})
        h = dict(challenger_regime.get(regime_name) or {})
        regime_compare[regime_name] = {
            "window_count": {"champion": int(c.get("window_count", 0) or 0), "challenger": int(h.get("window_count", 0) or 0), "delta": int(h.get("window_count", 0) or 0) - int(c.get("window_count", 0) or 0)},
            "median_score": _metric_triplet(champion=c.get("median_score", 0.0), challenger=h.get("median_score", 0.0)),
            "median_ret_pct": _metric_triplet(champion=c.get("median_ret_pct", 0.0), challenger=h.get("median_ret_pct", 0.0)),
            "worst_ret_pct": _metric_triplet(champion=c.get("worst_ret_pct", 0.0), challenger=h.get("worst_ret_pct", 0.0)),
            "max_mdd": _metric_triplet(champion=c.get("max_mdd", 0.0), challenger=h.get("max_mdd", 0.0)),
        }

    labels_with_dates = [row for row in window_compare_rows if row.get("oos_start") and row.get("oos_end")]
    start_date = labels_with_dates[0]["oos_start"] if labels_with_dates else ""
    end_date = labels_with_dates[-1]["oos_end"] if labels_with_dates else ""
    champion_linked = _link_total_return_pct(champion_windows)
    challenger_linked = _link_total_return_pct(challenger_windows)
    benchmark_linked = _link_total_return_pct([{"ret_pct": row.get("benchmark_return_pct", 0.0)} for row in challenger_windows])
    oos_total_compare = {
        "oos_range": {"start": start_date, "end": end_date},
        "metrics": {
            "linked_total_return_pct": _metric_triplet(champion=champion_linked, challenger=challenger_linked, benchmark=benchmark_linked),
            "annualized_return_pct": _metric_triplet(
                champion=_annualized_return_pct(linked_total_return_pct=champion_linked, start_date=start_date, end_date=end_date),
                challenger=_annualized_return_pct(linked_total_return_pct=challenger_linked, start_date=start_date, end_date=end_date),
                benchmark=_annualized_return_pct(linked_total_return_pct=benchmark_linked, start_date=start_date, end_date=end_date),
            ),
            "max_drawdown_pct": _metric_triplet(
                champion=_safe_max(float(r.get("mdd", 0.0) or 0.0) for r in champion_windows),
                challenger=_safe_max(float(r.get("mdd", 0.0) or 0.0) for r in challenger_windows),
                benchmark=_safe_max(float(r.get("benchmark_mdd", 0.0) or 0.0) for r in challenger_windows),
            ),
            "positive_window_rate": _metric_triplet(
                champion=_positive_window_rate(champion_windows, "ret_pct"),
                challenger=_positive_window_rate(challenger_windows, "ret_pct"),
                benchmark=_positive_window_rate(challenger_windows, "benchmark_return_pct"),
            ),
        },
    }

    quality_checks = [
        _build_gate_check(name="median_window_score_vs_champion", actual=float(summary_compare["median_window_score"]["challenger"]), threshold=f">= {float(summary_compare['median_window_score']['champion']):.3f}", passed=float(summary_compare["median_window_score"]["challenger"]) >= float(summary_compare["median_window_score"]["champion"]), severity="quality", note="候選版整體 OOS 典型視窗分數不得低於現役版。"),
        _build_gate_check(name="flat_median_score_vs_champion", actual=float(summary_compare["flat_median_score"]["challenger"]), threshold=f">= {float(summary_compare['flat_median_score']['champion']):.3f}", passed=float(summary_compare["flat_median_score"]["challenger"]) >= float(summary_compare["flat_median_score"]["champion"]), severity="quality", note="候選版盤整期中位分數不得低於現役版。"),
        _build_gate_check(name="worst_ret_pct_vs_champion", actual=float(summary_compare["worst_ret_pct"]["challenger"]), threshold=f">= {float(summary_compare['worst_ret_pct']['champion']) - 1.0:.2f}%", passed=float(summary_compare["worst_ret_pct"]["challenger"]) >= (float(summary_compare["worst_ret_pct"]["champion"]) - 1.0), severity="quality", note="候選版最差視窗報酬不可比現役版惡化超過 1%。"),
        _build_gate_check(name="max_mdd_vs_champion", actual=float(summary_compare["max_mdd"]["challenger"]), threshold=f"<= {float(summary_compare['max_mdd']['champion']) + 2.0:.2f}%", passed=float(summary_compare["max_mdd"]["challenger"]) <= (float(summary_compare["max_mdd"]["champion"]) + 2.0), severity="quality", note="候選版最大視窗 MDD 不可比現役版惡化超過 2%。"),
    ]
    coverage_checks = [
        _build_gate_check(name="down_regime_coverage_vs_champion", actual=int(summary_compare["down_window_count"]["challenger"]), threshold=f">= {int(summary_compare['down_window_count']['champion'])}", passed=int(summary_compare["down_window_count"]["challenger"]) >= int(summary_compare["down_window_count"]["champion"]), severity="coverage", note="候選版 down 視窗覆蓋不可少於現役版。"),
    ]
    quality_pass = all(bool(check["passed"]) for check in quality_checks)
    coverage_pass = all(bool(check["passed"]) for check in coverage_checks)
    quality_status = "pass" if quality_pass else "fail"
    coverage_status = "pass" if coverage_pass else "watch"
    status = "pass" if (quality_pass and coverage_pass) else ("watch" if quality_pass else "fail")
    recommendation = {
        "pass": "候選版整體優於現役版，可列為升版候選。",
        "watch": "候選版品質優於現役版，但 regime 覆蓋證據仍有限。",
        "fail": "候選版相對現役版仍不具穩定升級優勢。",
    }[status]
    compare_assessment = {
        "status": status,
        "recommended_for_promotion": bool(quality_pass and coverage_pass),
        "quality_gate": {"status": quality_status, "checks": quality_checks},
        "coverage_gate": {"status": coverage_status, "checks": coverage_checks},
        "checks": quality_checks + coverage_checks,
        "recommendation": recommendation,
    }

    return {
        "meta": {
            "dataset_label": str(dataset_label),
            "source_db_path": str(source_db_path),
            "session_ts": str(session_ts or get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")),
            "champion_params_path": "models/champion_params.json",
            "challenger_run_best_params_path": "models/run_best_params.json",
        },
        "champion_params": dict(champion_payload or {}),
        "challenger_params": dict(challenger_payload or {}),
        "summary_compare": summary_compare,
        "oos_total_compare": oos_total_compare,
        "regime_compare": regime_compare,
        "window_compare_rows": window_compare_rows,
        "compare_assessment": compare_assessment,
    }

def write_walk_forward_compare_report(*, output_dir: str, compare_payload: dict):
    os.makedirs(output_dir, exist_ok=True)
    session_ts = str((compare_payload.get("meta") or {}).get("session_ts") or get_taipei_now().strftime("%Y%m%d_%H%M%S_%f"))
    base_name = f"walk_forward_compare_{session_ts}"
    json_path = os.path.join(output_dir, f"{base_name}.json")
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    md_path = os.path.join(output_dir, f"{base_name}.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(compare_payload, handle, indent=2, ensure_ascii=False)

    rows = list(compare_payload.get("window_compare_rows") or [])
    fieldnames = [
        "label", "oos_start", "oos_end", "regime",
        "champion_window_score", "challenger_window_score", "delta_window_score",
        "champion_ret_pct", "challenger_ret_pct", "delta_ret_pct",
        "champion_mdd", "challenger_mdd", "delta_mdd",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    assessment = dict(compare_payload.get("compare_assessment") or {})
    summary_compare = dict(compare_payload.get("summary_compare") or {})
    oos_total_compare = dict(compare_payload.get("oos_total_compare") or {})
    regime_compare = dict(compare_payload.get("regime_compare") or {})

    lines = [
        "# Walk-Forward 升版比較報表（Champion vs Challenger）",
        "",
        f"- 資料集：{(compare_payload.get('meta') or {}).get('dataset_label', 'N/A')}",
        f"- 記憶庫：`{(compare_payload.get('meta') or {}).get('source_db_path', '')}`",
        f"- 現役正式版：`{(compare_payload.get('meta') or {}).get('champion_params_path', 'models/champion_params.json')}`",
        f"- 本輪最佳：`{(compare_payload.get('meta') or {}).get('challenger_run_best_params_path', 'models/run_best_params.json')}`",
        f"- 產生時間：{session_ts}",
        "",
        "## 升版比較結論",
        "",
        f"- 狀態：`{assessment.get('status', 'fail')}`",
        f"- 建議：{assessment.get('recommendation', 'N/A')}",
        f"- 品質 gate：`{(assessment.get('quality_gate') or {}).get('status', 'fail')}`",
        f"- 覆蓋 gate：`{(assessment.get('coverage_gate') or {}).get('status', 'watch')}`",
        "",
        "| 指標 | Champion | Challenger | Delta |",
        "|---|---:|---:|---:|",
    ]
    ordered_summary_keys = [
        "median_window_score", "median_ret_pct", "worst_ret_pct", "max_mdd",
        "median_annual_trades", "median_fill_rate", "flat_median_score", "down_window_count",
    ]
    for metric_key in ordered_summary_keys:
        metric = dict(summary_compare.get(metric_key) or {})
        if metric_key in ("median_window_score", "flat_median_score"):
            champ = f"{metric.get('champion', 0.0):.3f}"
            chall = f"{metric.get('challenger', 0.0):.3f}"
            delt = f"{metric.get('delta', 0.0):+.3f}"
        elif metric_key in ("median_annual_trades",):
            champ = f"{metric.get('champion', 0.0):.2f}"
            chall = f"{metric.get('challenger', 0.0):.2f}"
            delt = f"{metric.get('delta', 0.0):+.2f}"
        elif metric_key in ("down_window_count",):
            champ = str(int(metric.get('champion', 0) or 0))
            chall = str(int(metric.get('challenger', 0) or 0))
            delt = f"{int(metric.get('delta', 0) or 0):+d}"
        else:
            champ = _format_pct(metric.get('champion', 0.0))
            chall = _format_pct(metric.get('challenger', 0.0))
            delt = _format_pct(metric.get('delta', 0.0))
        lines.append(f"| {metric_key} | {champ} | {chall} | {delt} |")

    lines.extend([
        "",
        "## OOS 總績效比較",
        "",
        f"- OOS 區間：{(oos_total_compare.get('oos_range') or {}).get('start', '')} ~ {(oos_total_compare.get('oos_range') or {}).get('end', '')}",
        "",
        "| 指標 | Champion | Challenger | 0050 | Challenger - Champion | Challenger - 0050 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric_key in ("linked_total_return_pct", "annualized_return_pct", "max_drawdown_pct", "positive_window_rate"):
        metric = dict((oos_total_compare.get("metrics") or {}).get(metric_key) or {})
        lines.append(
            f"| {metric_key} | {_format_pct(metric.get('champion', 0.0))} | {_format_pct(metric.get('challenger', 0.0))} | {_format_pct(metric.get('benchmark', 0.0))} | {_format_pct(metric.get('delta_vs_champion', 0.0))} | {_format_pct(metric.get('delta_vs_benchmark', 0.0))} |"
        )

    lines.extend([
        "",
        "## 品質 / 覆蓋 Gate",
        "",
        "| Gate | 檢查項目 | 實際值 | 門檻 | 結果 | 說明 |",
        "|---|---|---:|---:|---|---|",
    ])
    for gate_name, gate_payload in (("quality", assessment.get("quality_gate") or {}), ("coverage", assessment.get("coverage_gate") or {})):
        for check in list(gate_payload.get("checks") or []):
            lines.append(
                f"| {gate_name} | {check['name']} | {_format_gate_actual(str(check['name']), check.get('actual'))} | {check.get('threshold', '')} | "
                f"{'PASS' if check.get('passed') else ('WATCH' if gate_name == 'coverage' else 'FAIL')} | {check.get('note', '')} |"
            )

    lines.extend([
        "",
        "## Regime 比較",
        "",
        "| Regime | 指標 | Champion | Challenger | Delta |",
        "|---|---|---:|---:|---:|",
    ])
    for regime_name in ("up", "flat", "down"):
        metrics = dict(regime_compare.get(regime_name) or {})
        for metric_key in ("window_count", "median_score", "median_ret_pct", "worst_ret_pct", "max_mdd"):
            metric = dict(metrics.get(metric_key) or {})
            if metric_key == "window_count":
                champ = str(int(metric.get('champion', 0) or 0))
                chall = str(int(metric.get('challenger', 0) or 0))
                delt = f"{int(metric.get('delta', 0) or 0):+d}"
            elif metric_key == "median_score":
                champ = f"{metric.get('champion', 0.0):.3f}"
                chall = f"{metric.get('challenger', 0.0):.3f}"
                delt = f"{metric.get('delta', 0.0):+.3f}"
            else:
                champ = _format_pct(metric.get('champion', 0.0))
                chall = _format_pct(metric.get('challenger', 0.0))
                delt = _format_pct(metric.get('delta', 0.0))
            lines.append(f"| {regime_name} | {metric_key} | {champ} | {chall} | {delt} |")

    lines.extend([
        "",
        "## OOS 視窗比較",
        "",
        "| 視窗 | OOS 區間 | Regime | Champion 分數 | Challenger 分數 | Delta | Champion 報酬率 | Challenger 報酬率 | Delta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['oos_start']} ~ {row['oos_end']} | {row['regime']} | {row['champion_window_score']:.3f} | {row['challenger_window_score']:.3f} | {row['delta_window_score']:+.3f} | "
            f"{_format_pct(row['champion_ret_pct'])} | {_format_pct(row['challenger_ret_pct'])} | {_format_pct(row['delta_ret_pct'])} |"
        )

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return {"json_path": json_path, "csv_path": csv_path, "md_path": md_path}
