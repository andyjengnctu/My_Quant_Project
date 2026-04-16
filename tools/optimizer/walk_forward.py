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
    window_count = int(summary.get("window_count", 0) or 0)
    median_window_score = float(summary.get("median_window_score", 0.0) or 0.0)
    worst_ret_pct = float(summary.get("worst_ret_pct", 0.0) or 0.0)
    flat_row = dict(regime_summary.get("flat") or {})
    down_row = dict(regime_summary.get("down") or {})
    flat_window_count = int(flat_row.get("window_count", 0) or 0)
    flat_median_score = float(flat_row.get("median_score", 0.0) or 0.0)
    down_window_count = int(down_row.get("window_count", 0) or 0)

    checks = [
        _build_gate_check(
            name="median_window_score",
            actual=median_window_score,
            threshold="> 0",
            passed=median_window_score > WF_GATE_MIN_MEDIAN_SCORE,
            severity="hard",
            note="整體 OOS 典型視窗需為正分。",
        ),
        _build_gate_check(
            name="worst_ret_pct",
            actual=worst_ret_pct,
            threshold=f">= {WF_GATE_MIN_WORST_RET_PCT:.1f}%",
            passed=worst_ret_pct >= WF_GATE_MIN_WORST_RET_PCT,
            severity="hard",
            note="避免單一 OOS 視窗災難性失真。",
        ),
        _build_gate_check(
            name="flat_median_score",
            actual=flat_median_score,
            threshold=">= 0 (當 flat 視窗存在時)",
            passed=(flat_window_count == 0) or (flat_median_score >= WF_GATE_MIN_FLAT_MEDIAN_SCORE),
            severity="hard",
            note="盤整期不可連續結構性失分。",
        ),
        _build_gate_check(
            name="down_regime_coverage",
            actual=down_window_count,
            threshold=">= 1",
            passed=down_window_count >= 1,
            severity="coverage",
            note="若無 down 視窗，不得宣稱已具跨盤勢穩健性。",
        ),
    ]

    hard_pass = all(bool(check["passed"]) for check in checks if check["severity"] == "hard")
    coverage_pass = all(bool(check["passed"]) for check in checks if check["severity"] == "coverage")
    if hard_pass and coverage_pass:
        status = "pass"
    elif hard_pass:
        status = "watch"
    else:
        status = "fail"

    recommendation = {
        "pass": "可列為候選升版版本，但仍建議與現役版做對照。",
        "watch": "核心 OOS 門檻已過，但 regime 覆蓋不足，只能視為有限證據。",
        "fail": "暫不建議僅依此報表升版；應先改善失敗項目。",
    }[status]

    return {
        "status": status,
        "recommended_for_promotion": bool(hard_pass and coverage_pass),
        "cross_regime_claim_allowed": bool(coverage_pass),
        "checks": checks,
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
        "## 升版門檻（MVP，僅報表判讀，不阻擋匯出）",
        "",
        f"- 狀態：`{upgrade_gate.get('status', 'fail')}`",
        f"- 建議：{upgrade_gate.get('recommendation', 'N/A')}",
        f"- 可宣稱跨盤勢穩健：{'是' if upgrade_gate.get('cross_regime_claim_allowed', False) else '否'}",
        "",
        "| 檢查項目 | 實際值 | 門檻 | 結果 | 說明 |",
        "|---|---:|---:|---|---|",
    ]
    for check in list(upgrade_gate.get("checks") or []):
        lines.append(
            f"| {check['name']} | {_format_gate_actual(str(check['name']), check.get('actual'))} | {check.get('threshold', '')} | "
            f"{'PASS' if check.get('passed') else 'FAIL'} | {check.get('note', '')} |"
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

    checks = [
        _build_gate_check(
            name="median_window_score_vs_champion",
            actual=_safe_float(challenger_summary.get("median_window_score", 0.0)),
            threshold=f">= { _safe_float(champion_summary.get('median_window_score', 0.0)):.3f}",
            passed=_safe_float(challenger_summary.get("median_window_score", 0.0)) >= _safe_float(champion_summary.get("median_window_score", 0.0)),
            severity="hard",
            note="候選版整體 OOS 典型視窗分數不得低於現役版。",
        ),
        _build_gate_check(
            name="flat_median_score_vs_champion",
            actual=challenger_flat_score,
            threshold=f">= {champion_flat_score:.3f}",
            passed=challenger_flat_score >= champion_flat_score,
            severity="hard",
            note="候選版盤整期中位分數不得低於現役版。",
        ),
        _build_gate_check(
            name="worst_ret_pct_vs_champion",
            actual=_safe_float(challenger_summary.get("worst_ret_pct", 0.0)),
            threshold=f">= {(_safe_float(champion_summary.get('worst_ret_pct', 0.0)) - 1.0):.2f}%",
            passed=_safe_float(challenger_summary.get("worst_ret_pct", 0.0)) >= (_safe_float(champion_summary.get("worst_ret_pct", 0.0)) - 1.0),
            severity="hard",
            note="候選版最差視窗報酬不可比現役版惡化超過 1%。",
        ),
        _build_gate_check(
            name="max_mdd_vs_champion",
            actual=_safe_float(challenger_summary.get("max_mdd", 0.0)),
            threshold=f"<= {(_safe_float(champion_summary.get('max_mdd', 0.0)) + 2.0):.2f}%",
            passed=_safe_float(challenger_summary.get("max_mdd", 0.0)) <= (_safe_float(champion_summary.get("max_mdd", 0.0)) + 2.0),
            severity="hard",
            note="候選版最大視窗 MDD 不可比現役版惡化超過 2%。",
        ),
        _build_gate_check(
            name="down_regime_coverage_vs_champion",
            actual=challenger_down_count,
            threshold=f">= {champion_down_count}",
            passed=challenger_down_count >= champion_down_count,
            severity="coverage",
            note="候選版 down 視窗覆蓋不可少於現役版。",
        ),
    ]

    hard_pass = all(bool(check["passed"]) for check in checks if check["severity"] == "hard")
    coverage_pass = all(bool(check["passed"]) for check in checks if check["severity"] == "coverage")
    if hard_pass and coverage_pass:
        status = "pass"
    elif hard_pass:
        status = "watch"
    else:
        status = "fail"

    recommendation = {
        "pass": "候選版整體優於現役版，可列為升版候選。",
        "watch": "候選版核心指標不差，但 regime 覆蓋未明顯優於現役版，需審慎人工判讀。",
        "fail": "候選版尚未穩定優於現役版，暫不建議升版。",
    }[status]

    return {
        "status": status,
        "recommended_for_promotion": bool(hard_pass and coverage_pass),
        "checks": checks,
        "recommendation": recommendation,
    }


def build_walk_forward_compare_payload(*, champion_payload: dict, champion_report: dict, challenger_payload: dict, challenger_report: dict, dataset_label: str, source_db_path: str, session_ts: str) -> dict:
    champion_summary = dict(champion_report.get("summary") or {})
    challenger_summary = dict(challenger_report.get("summary") or {})
    champion_regime = dict(champion_report.get("regime_summary") or {})
    challenger_regime = dict(challenger_report.get("regime_summary") or {})
    champion_windows = list(champion_report.get("windows") or [])
    challenger_windows = list(challenger_report.get("windows") or [])
    compare_assessment = build_compare_assessment(champion_report=champion_report, challenger_report=challenger_report)

    def delta(a, b):
        return _safe_float(b) - _safe_float(a)

    summary_compare = {
        "median_window_score": {"champion": _safe_float(champion_summary.get("median_window_score", 0.0)), "challenger": _safe_float(challenger_summary.get("median_window_score", 0.0)), "delta": delta(champion_summary.get("median_window_score", 0.0), challenger_summary.get("median_window_score", 0.0))},
        "median_ret_pct": {"champion": _safe_float(champion_summary.get("median_ret_pct", 0.0)), "challenger": _safe_float(challenger_summary.get("median_ret_pct", 0.0)), "delta": delta(champion_summary.get("median_ret_pct", 0.0), challenger_summary.get("median_ret_pct", 0.0))},
        "worst_ret_pct": {"champion": _safe_float(champion_summary.get("worst_ret_pct", 0.0)), "challenger": _safe_float(challenger_summary.get("worst_ret_pct", 0.0)), "delta": delta(champion_summary.get("worst_ret_pct", 0.0), challenger_summary.get("worst_ret_pct", 0.0))},
        "max_mdd": {"champion": _safe_float(champion_summary.get("max_mdd", 0.0)), "challenger": _safe_float(challenger_summary.get("max_mdd", 0.0)), "delta": delta(champion_summary.get("max_mdd", 0.0), challenger_summary.get("max_mdd", 0.0))},
        "median_annual_trades": {"champion": _safe_float(champion_summary.get("median_annual_trades", 0.0)), "challenger": _safe_float(challenger_summary.get("median_annual_trades", 0.0)), "delta": delta(champion_summary.get("median_annual_trades", 0.0), challenger_summary.get("median_annual_trades", 0.0))},
        "median_fill_rate": {"champion": _safe_float(champion_summary.get("median_fill_rate", 0.0)), "challenger": _safe_float(challenger_summary.get("median_fill_rate", 0.0)), "delta": delta(champion_summary.get("median_fill_rate", 0.0), challenger_summary.get("median_fill_rate", 0.0))},
        "flat_median_score": {"champion": _safe_float((champion_regime.get("flat") or {}).get("median_score", 0.0)), "challenger": _safe_float((challenger_regime.get("flat") or {}).get("median_score", 0.0)), "delta": delta((champion_regime.get("flat") or {}).get("median_score", 0.0), (challenger_regime.get("flat") or {}).get("median_score", 0.0))},
        "down_window_count": {"champion": int((champion_regime.get("down") or {}).get("window_count", 0) or 0), "challenger": int((challenger_regime.get("down") or {}).get("window_count", 0) or 0), "delta": int((challenger_regime.get("down") or {}).get("window_count", 0) or 0) - int((champion_regime.get("down") or {}).get("window_count", 0) or 0)},
    }

    regime_compare = {}
    for regime_name in ("up", "flat", "down"):
        champ = dict(champion_regime.get(regime_name) or {})
        chall = dict(challenger_regime.get(regime_name) or {})
        regime_compare[regime_name] = {
            "window_count": {"champion": int(champ.get("window_count", 0) or 0), "challenger": int(chall.get("window_count", 0) or 0), "delta": int(chall.get("window_count", 0) or 0) - int(champ.get("window_count", 0) or 0)},
            "median_score": {"champion": _safe_float(champ.get("median_score", 0.0)), "challenger": _safe_float(chall.get("median_score", 0.0)), "delta": delta(champ.get("median_score", 0.0), chall.get("median_score", 0.0))},
            "median_ret_pct": {"champion": _safe_float(champ.get("median_ret_pct", 0.0)), "challenger": _safe_float(chall.get("median_ret_pct", 0.0)), "delta": delta(champ.get("median_ret_pct", 0.0), chall.get("median_ret_pct", 0.0))},
            "worst_ret_pct": {"champion": _safe_float(champ.get("worst_ret_pct", 0.0)), "challenger": _safe_float(chall.get("worst_ret_pct", 0.0)), "delta": delta(champ.get("worst_ret_pct", 0.0), chall.get("worst_ret_pct", 0.0))},
            "max_mdd": {"champion": _safe_float(champ.get("max_mdd", 0.0)), "challenger": _safe_float(chall.get("max_mdd", 0.0)), "delta": delta(champ.get("max_mdd", 0.0), chall.get("max_mdd", 0.0))},
        }

    champion_oos_total = build_oos_total_performance(champion_windows, ret_key="ret_pct")
    challenger_oos_total = build_oos_total_performance(challenger_windows, ret_key="ret_pct")
    benchmark_oos_total = build_oos_total_performance(challenger_windows or champion_windows, ret_key="benchmark_return_pct")
    reference_oos_total = challenger_oos_total if challenger_windows else champion_oos_total
    oos_total_compare = {
        "oos_range": {
            "start": str(reference_oos_total.get("oos_start", "")),
            "end": str(reference_oos_total.get("oos_end", "")),
        },
        "metrics": {},
    }
    for metric_key in ("linked_total_return_pct", "annualized_return_pct", "max_drawdown_pct", "positive_window_rate"):
        oos_total_compare["metrics"][metric_key] = {
            "champion": _safe_float(champion_oos_total.get(metric_key, 0.0)),
            "challenger": _safe_float(challenger_oos_total.get(metric_key, 0.0)),
            "benchmark": _safe_float(benchmark_oos_total.get(metric_key, 0.0)),
            "delta_vs_champion": delta(champion_oos_total.get(metric_key, 0.0), challenger_oos_total.get(metric_key, 0.0)),
            "delta_vs_benchmark": delta(benchmark_oos_total.get(metric_key, 0.0), challenger_oos_total.get(metric_key, 0.0)),
        }

    champion_by_label = _window_row_by_label(champion_windows)
    challenger_by_label = _window_row_by_label(challenger_windows)
    ordered_labels = []
    for source_rows in (champion_windows, challenger_windows):
        for row in source_rows:
            label = str(row.get("label"))
            if label not in ordered_labels:
                ordered_labels.append(label)
    window_compare_rows = []
    for label in ordered_labels:
        champ = champion_by_label.get(label, {})
        chall = challenger_by_label.get(label, {})
        oos_start = chall.get("oos_start") or champ.get("oos_start") or ""
        oos_end = chall.get("oos_end") or champ.get("oos_end") or ""
        regime = chall.get("regime") or champ.get("regime") or ""
        window_compare_rows.append({
            "label": label,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "regime": regime,
            "champion_window_score": _safe_float(champ.get("window_score", 0.0)),
            "challenger_window_score": _safe_float(chall.get("window_score", 0.0)),
            "delta_window_score": delta(champ.get("window_score", 0.0), chall.get("window_score", 0.0)),
            "champion_ret_pct": _safe_float(champ.get("ret_pct", 0.0)),
            "challenger_ret_pct": _safe_float(chall.get("ret_pct", 0.0)),
            "delta_ret_pct": delta(champ.get("ret_pct", 0.0), chall.get("ret_pct", 0.0)),
            "champion_mdd": _safe_float(champ.get("mdd", 0.0)),
            "challenger_mdd": _safe_float(chall.get("mdd", 0.0)),
            "delta_mdd": delta(champ.get("mdd", 0.0), chall.get("mdd", 0.0)),
        })

    return {
        "meta": {
            "dataset_label": str(dataset_label),
            "source_db_path": str(source_db_path),
            "session_ts": str(session_ts),
            "champion_params_path": "models/champion_params.json",
            "challenger_run_best_params_path": "models/run_best_params.json",
        },
        "champion": {"params": dict(champion_payload), "summary": champion_summary, "regime_summary": champion_regime},
        "challenger": {"params": dict(challenger_payload), "summary": challenger_summary, "regime_summary": challenger_regime},
        "compare_assessment": compare_assessment,
        "summary_compare": summary_compare,
        "oos_total_compare": oos_total_compare,
        "regime_compare": regime_compare,
        "window_compare_rows": window_compare_rows,
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
        "## 升版檢查項目",
        "",
        "| 檢查項目 | 實際值 | 門檻 | 結果 | 說明 |",
        "|---|---:|---:|---|---|",
    ])
    for check in list(assessment.get("checks") or []):
        lines.append(
            f"| {check['name']} | {_format_gate_actual(str(check['name']), check.get('actual'))} | {check.get('threshold', '')} | "
            f"{'PASS' if check.get('passed') else 'FAIL'} | {check.get('note', '')} |"
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
