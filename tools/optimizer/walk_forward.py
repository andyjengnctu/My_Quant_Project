from __future__ import annotations

import csv
import json
import os
from statistics import median
from typing import Iterable

import pandas as pd

from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_stats import calc_portfolio_score
from core.runtime_utils import get_taipei_now

WF_MIN_TRAIN_YEARS = 8
WF_COMPARE_WORST_RET_TOLERANCE_PCT = 1.0
WF_COMPARE_MAX_MDD_TOLERANCE_PCT = 2.0


def _safe_median(values: Iterable[float]) -> float:
    normalized = [float(value) for value in values]
    return float(median(normalized)) if normalized else 0.0


def _safe_min(values: Iterable[float]) -> float:
    normalized = [float(value) for value in values]
    return float(min(normalized)) if normalized else 0.0


def _safe_max(values: Iterable[float]) -> float:
    normalized = [float(value) for value in values]
    return float(max(normalized)) if normalized else 0.0


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _align_to_half_year_boundary(candidate_start: pd.Timestamp) -> pd.Timestamp:
    candidate_start = pd.Timestamp(candidate_start).normalize()
    jan_boundary = pd.Timestamp(year=candidate_start.year, month=1, day=1)
    jul_boundary = pd.Timestamp(year=candidate_start.year, month=7, day=1)
    if candidate_start <= jan_boundary:
        return jan_boundary
    if candidate_start <= jul_boundary:
        return jul_boundary
    return pd.Timestamp(year=candidate_start.year + 1, month=1, day=1)


def resolve_first_walk_forward_test_boundary(
    sorted_dates,
    *,
    min_train_years: int = WF_MIN_TRAIN_YEARS,
    train_start_year: int | None = None,
) -> pd.Timestamp | None:
    if not sorted_dates:
        return None

    sorted_timestamps = [pd.Timestamp(dt).normalize() for dt in sorted_dates]
    first_market_date = sorted_timestamps[0]
    if train_start_year is None:
        first_train_date = first_market_date
    else:
        requested_start = pd.Timestamp(year=int(train_start_year), month=1, day=1)
        first_train_date = max(first_market_date, requested_start)
    return _align_to_half_year_boundary(first_train_date + pd.DateOffset(years=int(min_train_years)))


def build_walk_forward_windows(
    sorted_dates,
    *,
    min_train_years: int = WF_MIN_TRAIN_YEARS,
    test_window_months: int | None = None,
    min_window_bars: int | None = None,
    train_start_year: int | None = None,
):
    _ = test_window_months
    _ = min_window_bars
    if not sorted_dates:
        return []

    first_test_start = resolve_first_walk_forward_test_boundary(
        sorted_dates,
        min_train_years=min_train_years,
        train_start_year=train_start_year,
    )
    if first_test_start is None:
        return []

    sorted_timestamps = [pd.Timestamp(dt).normalize() for dt in sorted_dates]
    first_market_date = sorted_timestamps[0]
    last_date = sorted_timestamps[-1]
    requested_start = first_market_date if train_start_year is None else max(first_market_date, pd.Timestamp(year=int(train_start_year), month=1, day=1))
    train_end = first_test_start - pd.Timedelta(days=1)
    if train_end < requested_start:
        return []

    window_dates = [dt for dt in sorted_timestamps if first_test_start <= dt <= last_date]
    if not window_dates:
        return []

    return [{
        'label': 'TEST',
        'train_start': requested_start.strftime('%Y-%m-%d'),
        'train_end': train_end.strftime('%Y-%m-%d'),
        'oos_start': first_test_start.strftime('%Y-%m-%d'),
        'oos_end': last_date.strftime('%Y-%m-%d'),
        'window_dates': window_dates,
    }]


def evaluate_walk_forward(
    *,
    all_dfs_fast,
    all_trade_logs,
    sorted_dates,
    params,
    max_positions,
    enable_rotation,
    benchmark_ticker: str = '0050',
    min_train_years: int = WF_MIN_TRAIN_YEARS,
    test_window_months: int | None = None,
    train_start_year: int | None = None,
    regime_up_threshold_pct: float | None = None,
    regime_down_threshold_pct: float | None = None,
    min_window_bars: int | None = None,
    gate_min_median_score: float | None = None,
    gate_min_worst_ret_pct: float | None = None,
    gate_min_flat_median_score: float | None = None,
):
    _ = regime_up_threshold_pct
    _ = regime_down_threshold_pct
    _ = gate_min_median_score
    _ = gate_min_worst_ret_pct
    _ = gate_min_flat_median_score
    benchmark_data = all_dfs_fast.get(str(benchmark_ticker), None)
    windows = build_walk_forward_windows(
        sorted_dates,
        min_train_years=min_train_years,
        test_window_months=test_window_months,
        min_window_bars=min_window_bars,
        train_start_year=train_start_year,
    )
    rows = []
    for window in windows:
        window_dates = list(window['window_dates'])
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
        rows.append({
            'label': window['label'],
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'oos_start': window['oos_start'],
            'oos_end': window['oos_end'],
            'regime': 'test',
            'window_score': float(window_score),
            'ret_pct': float(ret_pct),
            'annual_return_pct': float(annual_return_pct),
            'mdd': float(mdd),
            'trade_count': int(trade_count),
            'annual_trades': float(annual_trades),
            'reserved_buy_fill_rate': float(reserved_buy_fill_rate),
            'benchmark_return_pct': float(bm_ret),
            'benchmark_annual_return_pct': float(bm_annual_return_pct),
            'benchmark_mdd': float(bm_mdd),
            'win_rate': float(win_rate),
            'pf_payoff': float(pf_payoff),
            'pf_ev': float(pf_ev),
            'r_squared': float(r_sq),
            'monthly_win_rate': float(m_win_rate),
            'benchmark_r_squared': float(bm_r_sq),
            'benchmark_monthly_win_rate': float(bm_m_win_rate),
            'final_equity': float(final_eq),
            'avg_exposure': float(avg_exp),
            'max_exposure': float(max_exp),
            'missed_buys': int(total_missed),
            'missed_sells': int(total_missed_sells),
            'normal_trades': int(normal_trade_count),
            'extended_trades': int(extended_trade_count),
        })

    summary = {
        'window_count': int(len(rows)),
        'train_start_year': None if train_start_year is None else int(train_start_year),
        'min_train_years': int(min_train_years),
        'median_window_score': _safe_median(row['window_score'] for row in rows),
        'median_ret_pct': _safe_median(row['ret_pct'] for row in rows),
        'worst_ret_pct': _safe_min(row['ret_pct'] for row in rows),
        'max_mdd': _safe_max(row['mdd'] for row in rows),
        'median_annual_trades': _safe_median(row['annual_trades'] for row in rows),
        'median_fill_rate': _safe_median(row['reserved_buy_fill_rate'] for row in rows),
    }
    upgrade_gate = build_upgrade_gate_assessment(summary=summary, regime_summary={})
    return {
        'summary': summary,
        'regime_summary': {},
        'upgrade_gate': upgrade_gate,
        'windows': rows,
    }


def _build_gate_check(*, name: str, actual, threshold: str, passed: bool, severity: str, note: str) -> dict:
    return {
        'name': str(name),
        'actual': actual,
        'threshold': str(threshold),
        'passed': bool(passed),
        'severity': str(severity),
        'note': str(note),
    }


def build_upgrade_gate_assessment(*, summary: dict, regime_summary: dict, gate_min_median_score: float | None = None, gate_min_worst_ret_pct: float | None = None, gate_min_flat_median_score: float | None = None) -> dict:
    _ = regime_summary
    _ = gate_min_median_score
    _ = gate_min_worst_ret_pct
    _ = gate_min_flat_median_score
    window_count = int(summary.get('window_count', 0) or 0)
    quality_checks = [
        _build_gate_check(
            name='test_period_available',
            actual=window_count,
            threshold='>= 1',
            passed=window_count >= 1,
            severity='quality',
            note='必須成功產生 split 測試期間績效。',
        )
    ]
    quality_pass = all(bool(check['passed']) for check in quality_checks)
    status = 'pass' if quality_pass else 'fail'
    recommendation = '已產生 split 測試期間績效，可進入後續比較。' if quality_pass else '無法產生 split 測試期間績效，暫不可比較。'
    return {
        'status': status,
        'recommended_for_promotion': bool(quality_pass),
        'cross_regime_claim_allowed': False,
        'quality_gate': {'status': status, 'checks': quality_checks},
        'coverage_gate': {'status': 'pass', 'checks': []},
        'checks': quality_checks,
        'recommendation': recommendation,
    }


def _resolve_report_upgrade_gate(report: dict) -> dict:
    report_payload = dict(report or {})
    summary = dict(report_payload.get('summary') or {})
    return dict(report_payload.get('upgrade_gate') or build_upgrade_gate_assessment(summary=summary, regime_summary={}))


def _format_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def _format_gate_actual(name: str, actual) -> str:
    if 'score' in name:
        return f"{float(actual):.3f}"
    if 'trades' in name or 'available' in name:
        return str(int(actual))
    if isinstance(actual, bool):
        return 'PASS' if actual else 'FAIL'
    return _format_pct(actual)


def write_walk_forward_report(*, output_dir: str, params_payload: dict, dataset_label: str, report: dict, best_trial_number: int | None, source_db_path: str, session_ts: str | None = None):
    os.makedirs(output_dir, exist_ok=True)
    resolved_session_ts = session_ts or get_taipei_now().strftime('%Y%m%d_%H%M%S_%f')
    base_name = f'walk_forward_report_{resolved_session_ts}'
    json_path = os.path.join(output_dir, f'{base_name}.json')
    csv_path = os.path.join(output_dir, f'{base_name}.csv')
    md_path = os.path.join(output_dir, f'{base_name}.md')

    summary = dict(report.get('summary') or {})
    upgrade_gate = dict(report.get('upgrade_gate') or build_upgrade_gate_assessment(summary=summary, regime_summary={}))
    oos_total = build_oos_total_performance(list(report.get('windows') or []), ret_key='ret_pct')
    benchmark_oos_total = build_oos_total_performance(list(report.get('windows') or []), ret_key='benchmark_return_pct')
    oos_total['score_romd'] = float(oos_total.get('linked_total_return_pct', 0.0)) / (abs(float(oos_total.get('max_drawdown_pct', 0.0))) + 0.0001)
    benchmark_oos_total['score_romd'] = float(benchmark_oos_total.get('linked_total_return_pct', 0.0)) / (abs(float(benchmark_oos_total.get('max_drawdown_pct', 0.0))) + 0.0001)
    payload = {
        'meta': {
            'dataset_label': str(dataset_label),
            'best_trial_number': None if best_trial_number is None else int(best_trial_number),
            'source_db_path': str(source_db_path),
            'session_ts': str(resolved_session_ts),
        },
        'params': dict(params_payload),
        'summary': summary,
        'regime_summary': {},
        'upgrade_gate': upgrade_gate,
        'oos_total': oos_total,
        'benchmark_oos_total': benchmark_oos_total,
        'windows': list(report.get('windows') or []),
    }
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    windows = list(payload.get('windows') or [])
    if windows:
        fieldnames = list(windows[0].keys())
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(windows)
    else:
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['label', 'oos_start', 'oos_end', 'window_score', 'ret_pct', 'mdd'])

    lines = [
        '# 測試期間報表',
        '',
        f'- 資料集：{dataset_label}',
        f"- 最佳 trial：{payload['meta']['best_trial_number'] if payload['meta']['best_trial_number'] is not None else 'N/A'}",
        f'- 記憶庫：`{source_db_path}`',
        f'- 產生時間：{resolved_session_ts}',
        '',
        '## Policy',
        '',
        '| 項目 | 數值 |',
        '|---|---:|',
        f"| train_start_year | {summary.get('train_start_year', 'N/A')} |",
        f"| min_train_years | {int(summary.get('min_train_years', 0))} |",
        '',
        '## Test Period Total',
        '',
        '| 指標 | 策略 | 0050 |',
        '|---|---:|---:|',
        f"| 測試區間 | {oos_total.get('oos_start', '')} ~ {oos_total.get('oos_end', '')} | {benchmark_oos_total.get('oos_start', '')} ~ {benchmark_oos_total.get('oos_end', '')} |",
        f"| 測試 RoMD | {float(oos_total.get('score_romd', 0.0)):.3f} | {float(benchmark_oos_total.get('score_romd', 0.0)):.3f} |",
        f"| 串接總報酬率 | {_format_pct(oos_total.get('linked_total_return_pct', 0.0))} | {_format_pct(benchmark_oos_total.get('linked_total_return_pct', 0.0))} |",
        f"| 年化報酬率 | {_format_pct(oos_total.get('annualized_return_pct', 0.0))} | {_format_pct(benchmark_oos_total.get('annualized_return_pct', 0.0))} |",
        f"| 最大回撤 | {_format_pct(oos_total.get('max_drawdown_pct', 0.0))} | {_format_pct(benchmark_oos_total.get('max_drawdown_pct', 0.0))} |",
        '',
        '## Gate',
        '',
        f"- 狀態：`{upgrade_gate.get('status', 'fail')}`",
        f"- 建議：{upgrade_gate.get('recommendation', 'N/A')}",
        '',
        '| Gate | 檢查項目 | 實際值 | 門檻 | 結果 | 說明 |',
        '|---|---|---:|---:|---|---|',
    ]
    for gate_name, gate_payload in (('quality', upgrade_gate.get('quality_gate') or {}), ('coverage', upgrade_gate.get('coverage_gate') or {})):
        for check in list(gate_payload.get('checks') or []):
            lines.append(
                f"| {gate_name} | {check['name']} | {_format_gate_actual(str(check['name']), check.get('actual'))} | {check.get('threshold', '')} | {'PASS' if check.get('passed') else 'FAIL'} | {check.get('note', '')} |"
            )
    lines.extend(['', '## Test Window', '', '| 視窗 | 訓練區間 | 測試區間 | 分數 | 報酬率 | MDD | 年化交易次數 | 買進成交率 | 0050 報酬率 |', '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for row in windows:
        lines.append(
            f"| {row['label']} | {row['train_start']} ~ {row['train_end']} | {row['oos_start']} ~ {row['oos_end']} | {float(row['window_score']):.3f} | {_format_pct(row['ret_pct'])} | {_format_pct(row['mdd'])} | {float(row['annual_trades']):.2f} | {_format_pct(row['reserved_buy_fill_rate'])} | {_format_pct(row['benchmark_return_pct'])} |"
        )
    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return {'json_path': json_path, 'csv_path': csv_path, 'md_path': md_path}


def _parse_iso_date(value) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(str(value))
    except Exception:
        return None


def build_oos_total_performance(rows: list[dict], *, ret_key: str) -> dict:
    ordered_rows = sorted(list(rows or []), key=lambda row: str(row.get('oos_start') or ''))
    if not ordered_rows:
        return {
            'window_count': 0,
            'linked_total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'positive_window_rate': 0.0,
            'ending_equity_factor': 1.0,
            'oos_start': '',
            'oos_end': '',
        }
    equity = 1.0
    peak = 1.0
    max_drawdown_pct = 0.0
    positive_windows = 0
    start_ts = _parse_iso_date(ordered_rows[0].get('oos_start'))
    end_ts = _parse_iso_date(ordered_rows[-1].get('oos_end'))
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
        'window_count': int(len(ordered_rows)),
        'linked_total_return_pct': (equity - 1.0) * 100.0,
        'annualized_return_pct': float(annualized_return_pct),
        'max_drawdown_pct': float(max_drawdown_pct),
        'positive_window_rate': float(positive_windows / len(ordered_rows) * 100.0),
        'ending_equity_factor': float(equity),
        'oos_start': '' if start_ts is None else start_ts.strftime('%Y-%m-%d'),
        'oos_end': '' if end_ts is None else end_ts.strftime('%Y-%m-%d'),
    }


def _metric_triplet(*, champion, challenger, benchmark=None) -> dict:
    payload = {
        'champion': float(champion),
        'challenger': float(challenger),
        'delta': float(challenger) - float(champion),
    }
    if benchmark is not None:
        payload['benchmark'] = float(benchmark)
    return payload


def build_compare_assessment(*, champion_report: dict, challenger_report: dict, compare_worst_ret_tolerance_pct: float = WF_COMPARE_WORST_RET_TOLERANCE_PCT, compare_max_mdd_tolerance_pct: float = WF_COMPARE_MAX_MDD_TOLERANCE_PCT) -> dict:
    champion_summary = dict((champion_report or {}).get('summary') or {})
    challenger_summary = dict((challenger_report or {}).get('summary') or {})
    challenger_upgrade_gate = _resolve_report_upgrade_gate(challenger_report)
    challenger_quality_pass = str(((challenger_upgrade_gate.get('quality_gate') or {}).get('status') or 'fail')).lower() == 'pass'
    quality_checks = [
        _build_gate_check(
            name='challenger_upgrade_quality_gate',
            actual=challenger_quality_pass,
            threshold='PASS',
            passed=challenger_quality_pass,
            severity='quality',
            note='候選版自身測試期間報表必須成功產生。',
        ),
        _build_gate_check(
            name='median_window_score_vs_champion',
            actual=_safe_float(challenger_summary.get('median_window_score', 0.0)),
            threshold=f"> {_safe_float(champion_summary.get('median_window_score', 0.0)):.3f}",
            passed=_safe_float(challenger_summary.get('median_window_score', 0.0)) > _safe_float(champion_summary.get('median_window_score', 0.0)),
            severity='quality',
            note='候選版測試 RoMD 必須優於現役版。',
        ),
        _build_gate_check(
            name='worst_ret_pct_vs_champion',
            actual=_safe_float(challenger_summary.get('worst_ret_pct', 0.0)),
            threshold=f">= {(_safe_float(champion_summary.get('worst_ret_pct', 0.0)) - float(compare_worst_ret_tolerance_pct)):.2f}%",
            passed=_safe_float(challenger_summary.get('worst_ret_pct', 0.0)) >= (_safe_float(champion_summary.get('worst_ret_pct', 0.0)) - float(compare_worst_ret_tolerance_pct)),
            severity='quality',
            note=f"候選版測試報酬不可比現役版惡化超過 {float(compare_worst_ret_tolerance_pct):.2f}%。",
        ),
        _build_gate_check(
            name='max_mdd_vs_champion',
            actual=_safe_float(challenger_summary.get('max_mdd', 0.0)),
            threshold=f"<= {(_safe_float(champion_summary.get('max_mdd', 0.0)) + float(compare_max_mdd_tolerance_pct)):.2f}%",
            passed=_safe_float(challenger_summary.get('max_mdd', 0.0)) <= (_safe_float(champion_summary.get('max_mdd', 0.0)) + float(compare_max_mdd_tolerance_pct)),
            severity='quality',
            note=f"候選版測試期最大回撤不可比現役版惡化超過 {float(compare_max_mdd_tolerance_pct):.2f}%。",
        ),
    ]
    quality_pass = all(bool(check['passed']) for check in quality_checks)
    status = 'pass' if quality_pass else 'fail'
    recommendation = '候選版測試期優於現役版，可列為升版候選。' if quality_pass else '候選版尚未穩定超越現役版，維持現況。'
    return {
        'status': status,
        'recommended_for_promotion': bool(quality_pass),
        'cross_regime_claim_allowed': False,
        'quality_gate': {'status': status, 'checks': quality_checks},
        'coverage_gate': {'status': 'pass', 'checks': []},
        'challenger_upgrade_gate': challenger_upgrade_gate,
        'checks': quality_checks,
        'recommendation': recommendation,
    }


def build_walk_forward_compare_payload(*, champion_payload: dict, champion_report: dict, challenger_payload: dict, challenger_report: dict, dataset_label: str, source_db_path: str, session_ts: str | None = None, compare_worst_ret_tolerance_pct: float = WF_COMPARE_WORST_RET_TOLERANCE_PCT, compare_max_mdd_tolerance_pct: float = WF_COMPARE_MAX_MDD_TOLERANCE_PCT) -> dict:
    champion_windows = list((champion_report or {}).get('windows') or [])
    challenger_windows = list((challenger_report or {}).get('windows') or [])
    champion_summary = dict((champion_report or {}).get('summary') or {})
    challenger_summary = dict((challenger_report or {}).get('summary') or {})
    champion_by_label = {str(row.get('label')): dict(row) for row in champion_windows}
    challenger_by_label = {str(row.get('label')): dict(row) for row in challenger_windows}
    ordered_labels = sorted(set(champion_by_label) | set(challenger_by_label))
    window_compare_rows = []
    for label in ordered_labels:
        c = dict(champion_by_label.get(label) or {})
        h = dict(challenger_by_label.get(label) or {})
        window_compare_rows.append({
            'label': label,
            'oos_start': str(h.get('oos_start') or c.get('oos_start') or ''),
            'oos_end': str(h.get('oos_end') or c.get('oos_end') or ''),
            'regime': str(h.get('regime') or c.get('regime') or 'test'),
            'champion_window_score': float(c.get('window_score', 0.0) or 0.0),
            'challenger_window_score': float(h.get('window_score', 0.0) or 0.0),
            'delta_window_score': float(h.get('window_score', 0.0) or 0.0) - float(c.get('window_score', 0.0) or 0.0),
            'champion_ret_pct': float(c.get('ret_pct', 0.0) or 0.0),
            'challenger_ret_pct': float(h.get('ret_pct', 0.0) or 0.0),
            'delta_ret_pct': float(h.get('ret_pct', 0.0) or 0.0) - float(c.get('ret_pct', 0.0) or 0.0),
            'champion_mdd': float(c.get('mdd', 0.0) or 0.0),
            'challenger_mdd': float(h.get('mdd', 0.0) or 0.0),
            'delta_mdd': float(h.get('mdd', 0.0) or 0.0) - float(c.get('mdd', 0.0) or 0.0),
        })
    summary_compare = {
        'median_window_score': _metric_triplet(champion=champion_summary.get('median_window_score', 0.0), challenger=challenger_summary.get('median_window_score', 0.0)),
        'median_ret_pct': _metric_triplet(champion=champion_summary.get('median_ret_pct', 0.0), challenger=challenger_summary.get('median_ret_pct', 0.0)),
        'worst_ret_pct': _metric_triplet(champion=champion_summary.get('worst_ret_pct', 0.0), challenger=challenger_summary.get('worst_ret_pct', 0.0)),
        'max_mdd': _metric_triplet(champion=champion_summary.get('max_mdd', 0.0), challenger=challenger_summary.get('max_mdd', 0.0)),
        'median_annual_trades': _metric_triplet(champion=champion_summary.get('median_annual_trades', 0.0), challenger=challenger_summary.get('median_annual_trades', 0.0)),
        'median_fill_rate': _metric_triplet(champion=champion_summary.get('median_fill_rate', 0.0), challenger=challenger_summary.get('median_fill_rate', 0.0)),
        'flat_median_score': _metric_triplet(champion=champion_summary.get('median_window_score', 0.0), challenger=challenger_summary.get('median_window_score', 0.0)),
        'down_window_count': {'champion': 0, 'challenger': 0, 'delta': 0},
    }
    labels_with_dates = [row for row in window_compare_rows if row.get('oos_start') and row.get('oos_end')]
    start_date = labels_with_dates[0]['oos_start'] if labels_with_dates else ''
    end_date = labels_with_dates[-1]['oos_end'] if labels_with_dates else ''
    champion_total = build_oos_total_performance(champion_windows, ret_key='ret_pct')
    challenger_total = build_oos_total_performance(challenger_windows, ret_key='ret_pct')
    benchmark_total = build_oos_total_performance(challenger_windows, ret_key='benchmark_return_pct')
    oos_total_compare = {
        'oos_range': {'start': start_date, 'end': end_date},
        'metrics': {
            'linked_total_return_pct': _metric_triplet(champion=champion_total['linked_total_return_pct'], challenger=challenger_total['linked_total_return_pct'], benchmark=benchmark_total['linked_total_return_pct']),
            'annualized_return_pct': _metric_triplet(champion=champion_total['annualized_return_pct'], challenger=challenger_total['annualized_return_pct'], benchmark=benchmark_total['annualized_return_pct']),
            'max_drawdown_pct': _metric_triplet(champion=champion_total['max_drawdown_pct'], challenger=challenger_total['max_drawdown_pct'], benchmark=benchmark_total['max_drawdown_pct']),
            'positive_window_rate': _metric_triplet(champion=champion_total['positive_window_rate'], challenger=challenger_total['positive_window_rate'], benchmark=benchmark_total['positive_window_rate']),
        },
    }
    compare_assessment = build_compare_assessment(
        champion_report=champion_report,
        challenger_report=challenger_report,
        compare_worst_ret_tolerance_pct=compare_worst_ret_tolerance_pct,
        compare_max_mdd_tolerance_pct=compare_max_mdd_tolerance_pct,
    )
    return {
        'meta': {
            'dataset_label': str(dataset_label),
            'source_db_path': str(source_db_path),
            'session_ts': str(session_ts or get_taipei_now().strftime('%Y%m%d_%H%M%S_%f')),
            'champion_params_path': 'models/champion_params.json',
            'challenger_run_best_params_path': 'models/run_best_params.json',
            'compare_worst_ret_tolerance_pct': float(compare_worst_ret_tolerance_pct),
            'compare_max_mdd_tolerance_pct': float(compare_max_mdd_tolerance_pct),
        },
        'champion_params': dict(champion_payload or {}),
        'challenger_params': dict(challenger_payload or {}),
        'summary_compare': summary_compare,
        'oos_total_compare': oos_total_compare,
        'regime_compare': {},
        'window_compare_rows': window_compare_rows,
        'compare_assessment': compare_assessment,
    }


def write_walk_forward_compare_report(*, output_dir: str, compare_payload: dict):
    os.makedirs(output_dir, exist_ok=True)
    session_ts = str((compare_payload.get('meta') or {}).get('session_ts') or get_taipei_now().strftime('%Y%m%d_%H%M%S_%f'))
    base_name = f'walk_forward_compare_{session_ts}'
    json_path = os.path.join(output_dir, f'{base_name}.json')
    csv_path = os.path.join(output_dir, f'{base_name}.csv')
    md_path = os.path.join(output_dir, f'{base_name}.md')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(compare_payload, handle, indent=2, ensure_ascii=False)
    rows = list(compare_payload.get('window_compare_rows') or [])
    fieldnames = [
        'label', 'oos_start', 'oos_end', 'regime',
        'champion_window_score', 'challenger_window_score', 'delta_window_score',
        'champion_ret_pct', 'challenger_ret_pct', 'delta_ret_pct',
        'champion_mdd', 'challenger_mdd', 'delta_mdd',
    ]
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    assessment = dict(compare_payload.get('compare_assessment') or {})
    summary_compare = dict(compare_payload.get('summary_compare') or {})
    oos_total_compare = dict(compare_payload.get('oos_total_compare') or {})
    lines = [
        '# 測試期間升版比較報表（Champion vs Challenger）',
        '',
        f"- 資料集：{(compare_payload.get('meta') or {}).get('dataset_label', 'N/A')}",
        f"- 記憶庫：`{(compare_payload.get('meta') or {}).get('source_db_path', '')}`",
        f"- 現役正式版：`{(compare_payload.get('meta') or {}).get('champion_params_path', 'models/champion_params.json')}`",
        f"- 本輪最佳：`{(compare_payload.get('meta') or {}).get('challenger_run_best_params_path', 'models/run_best_params.json')}`",
        f'- 產生時間：{session_ts}',
        '',
        '## 升版比較結論',
        '',
        f"- 狀態：`{assessment.get('status', 'fail')}`",
        f"- 建議：{assessment.get('recommendation', 'N/A')}",
        f"- 品質 gate：`{(assessment.get('quality_gate') or {}).get('status', 'fail')}`",
        '',
        '| 指標 | Champion | Challenger | Delta |',
        '|---|---:|---:|---:|',
    ]
    for metric_key in ('median_window_score', 'worst_ret_pct', 'max_mdd', 'median_annual_trades'):
        metric = dict(summary_compare.get(metric_key) or {})
        lines.append(f"| {metric_key} | {metric.get('champion', 0.0)} | {metric.get('challenger', 0.0)} | {metric.get('delta', 0.0)} |")
    lines.extend(['', '## OOS Total', '', '| 指標 | Champion | Challenger | 0050 |', '|---|---:|---:|---:|'])
    for metric_key in ('linked_total_return_pct', 'annualized_return_pct', 'max_drawdown_pct', 'positive_window_rate'):
        metric = dict((oos_total_compare.get('metrics') or {}).get(metric_key) or {})
        lines.append(f"| {metric_key} | {metric.get('champion', 0.0)} | {metric.get('challenger', 0.0)} | {metric.get('benchmark', 0.0)} |")
    lines.extend(['', '## Test Window', '', '| 視窗 | 區間 | Champion 分數 | Challenger 分數 | Delta |', '|---|---|---:|---:|---:|'])
    for row in rows:
        lines.append(f"| {row['label']} | {row['oos_start']} ~ {row['oos_end']} | {float(row['champion_window_score']):.3f} | {float(row['challenger_window_score']):.3f} | {float(row['delta_window_score']):+.3f} |")
    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return {'json_path': json_path, 'csv_path': csv_path, 'md_path': md_path}
