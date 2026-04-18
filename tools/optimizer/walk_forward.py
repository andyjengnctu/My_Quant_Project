from __future__ import annotations

import csv
import json
import os
from typing import Iterable

import pandas as pd

from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_stats import calc_portfolio_score
from core.runtime_utils import get_taipei_now

WF_MIN_TRAIN_YEARS = 8


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def build_test_holdout_period(
    sorted_dates,
    *,
    min_train_years: int = WF_MIN_TRAIN_YEARS,
    train_start_year: int | None = None,
) -> dict | None:
    if not sorted_dates:
        return None

    sorted_timestamps = [pd.Timestamp(dt).normalize() for dt in sorted_dates]
    first_market_date = sorted_timestamps[0]
    last_date = sorted_timestamps[-1]
    requested_start = first_market_date if train_start_year is None else max(first_market_date, pd.Timestamp(year=int(train_start_year), month=1, day=1))
    first_test_start = resolve_first_walk_forward_test_boundary(
        sorted_timestamps,
        min_train_years=min_train_years,
        train_start_year=train_start_year,
    )
    if first_test_start is None:
        return None

    train_end = first_test_start - pd.Timedelta(days=1)
    if train_end < requested_start:
        return None

    test_dates = [dt for dt in sorted_timestamps if first_test_start <= dt <= last_date]
    if not test_dates:
        return None

    return {
        'label': 'TEST',
        'train_start': requested_start.strftime('%Y-%m-%d'),
        'train_end': train_end.strftime('%Y-%m-%d'),
        'oos_start': first_test_start.strftime('%Y-%m-%d'),
        'oos_end': last_date.strftime('%Y-%m-%d'),
        'test_dates': test_dates,
    }


def _calc_romd(ret_pct: float, mdd_pct: float) -> float:
    return float(ret_pct) / (abs(float(mdd_pct)) + 0.0001)


def _evaluate_single_holdout_period(
    *,
    holdout_period: dict,
    all_dfs_fast,
    all_trade_logs,
    params,
    max_positions,
    enable_rotation,
    benchmark_ticker: str,
) -> dict | None:
    test_dates = list(holdout_period.get('test_dates') or [])
    if not test_dates:
        return None
    benchmark_data = all_dfs_fast.get(str(benchmark_ticker), None)
    pf_profile = {}
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
        test_dates,
        pd.Timestamp(test_dates[0]).year,
        params,
        max_positions,
        enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=benchmark_data,
        is_training=True,
        profile_stats=pf_profile,
        verbose=False,
    )
    test_score_romd = calc_portfolio_score(
        ret_pct,
        mdd,
        m_win_rate,
        r_sq,
        annual_return_pct=annual_return_pct,
    )
    return {
        'label': str(holdout_period.get('label') or 'TEST'),
        'train_start': str(holdout_period.get('train_start') or ''),
        'train_end': str(holdout_period.get('train_end') or ''),
        'oos_start': str(holdout_period.get('oos_start') or ''),
        'oos_end': str(holdout_period.get('oos_end') or ''),
        'test_score_romd': float(test_score_romd),
        'ret_pct': float(ret_pct),
        'annual_return_pct': float(annual_return_pct),
        'min_full_year_return_pct': float(pf_profile.get('min_full_year_return_pct', 0.0)),
        'full_year_count': int(pf_profile.get('full_year_count', 0)),
        'mdd': float(mdd),
        'trade_count': int(trade_count),
        'normal_trades': int(normal_trade_count),
        'extended_trades': int(extended_trade_count),
        'annual_trades': float(annual_trades),
        'reserved_buy_fill_rate': float(reserved_buy_fill_rate),
        'win_rate': float(win_rate),
        'pf_payoff': float(pf_payoff),
        'pf_ev': float(pf_ev),
        'r_squared': float(r_sq),
        'monthly_win_rate': float(m_win_rate),
        'final_equity': float(final_eq),
        'avg_exposure': float(avg_exp),
        'max_exposure': float(max_exp),
        'missed_buys': int(total_missed),
        'missed_sells': int(total_missed_sells),
        'benchmark_return_pct': float(bm_ret),
        'benchmark_annual_return_pct': float(bm_annual_return_pct),
        'benchmark_min_full_year_return_pct': float(pf_profile.get('bm_min_full_year_return_pct', 0.0)),
        'benchmark_mdd': float(bm_mdd),
        'benchmark_r_squared': float(bm_r_sq),
        'benchmark_monthly_win_rate': float(bm_m_win_rate),
        'benchmark_score_romd': float(_calc_romd(bm_ret, bm_mdd)),
    }


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
    train_start_year: int | None = None,
):
    holdout_period = build_test_holdout_period(
        sorted_dates,
        min_train_years=min_train_years,
        train_start_year=train_start_year,
    )
    period_metrics = None
    if holdout_period is not None:
        period_metrics = _evaluate_single_holdout_period(
            holdout_period=holdout_period,
            all_dfs_fast=all_dfs_fast,
            all_trade_logs=all_trade_logs,
            params=params,
            max_positions=max_positions,
            enable_rotation=enable_rotation,
            benchmark_ticker=benchmark_ticker,
        )

    summary = {
        'period_count': 1 if period_metrics is not None else 0,
        'train_start_year': None if train_start_year is None else int(train_start_year),
        'min_train_years': int(min_train_years),
        'oos_start': '' if period_metrics is None else str(period_metrics.get('oos_start') or ''),
        'oos_end': '' if period_metrics is None else str(period_metrics.get('oos_end') or ''),
        'test_score_romd': 0.0 if period_metrics is None else float(period_metrics.get('test_score_romd', 0.0)),
        'test_return_pct': 0.0 if period_metrics is None else float(period_metrics.get('ret_pct', 0.0)),
        'test_mdd': 0.0 if period_metrics is None else float(period_metrics.get('mdd', 0.0)),
        'test_annual_trades': 0.0 if period_metrics is None else float(period_metrics.get('annual_trades', 0.0)),
        'test_fill_rate': 0.0 if period_metrics is None else float(period_metrics.get('reserved_buy_fill_rate', 0.0)),
    }
    upgrade_gate = build_upgrade_gate_assessment(summary=summary)
    return {
        'summary': summary,
        'upgrade_gate': upgrade_gate,
        'period': period_metrics,
    }


def build_upgrade_gate_assessment(*, summary: dict) -> dict:
    period_count = int(summary.get('period_count', 0) or 0)
    quality_checks = [
        {
            'name': 'test_period_available',
            'actual': period_count,
            'threshold': '>= 1',
            'passed': period_count >= 1,
            'severity': 'quality',
            'note': '必須成功產生單一連續 test holdout 績效。',
        }
    ]
    quality_pass = all(bool(check['passed']) for check in quality_checks)
    status = 'pass' if quality_pass else 'fail'
    recommendation = '已產生單一連續 test holdout 績效，可進入後續比較。' if quality_pass else '無法產生單一連續 test holdout 績效，暫不可比較。'
    return {
        'status': status,
        'recommended_for_promotion': bool(quality_pass),
        'quality_gate': {'status': status, 'checks': quality_checks},
        'coverage_gate': {'status': 'pass', 'checks': []},
        'checks': quality_checks,
        'recommendation': recommendation,
    }


def _empty_test_period_metrics() -> dict:
    return {
        'period_count': 0,
        'oos_start': '',
        'oos_end': '',
        'total_return_pct': 0.0,
        'annualized_return_pct': 0.0,
        'min_full_year_return_pct': 0.0,
        'max_drawdown_pct': 0.0,
        'test_score_romd': 0.0,
        'trade_count': 0,
        'normal_trades': 0,
        'extended_trades': 0,
        'annual_trades': 0.0,
        'fill_rate': 0.0,
        'win_rate': 0.0,
        'payoff': 0.0,
        'ev': 0.0,
        'r_squared': 0.0,
        'monthly_win_rate': 0.0,
        'avg_exposure': 0.0,
        'final_equity': 0.0,
        'benchmark_total_return_pct': 0.0,
        'benchmark_annualized_return_pct': 0.0,
        'benchmark_min_full_year_return_pct': 0.0,
        'benchmark_max_drawdown_pct': 0.0,
        'benchmark_score_romd': 0.0,
        'benchmark_r_squared': 0.0,
        'benchmark_monthly_win_rate': 0.0,
    }


def build_test_period_metrics(report: dict | None) -> dict:
    period = dict((report or {}).get('period') or {})
    if not period:
        return _empty_test_period_metrics()
    return {
        'period_count': 1,
        'oos_start': str(period.get('oos_start') or ''),
        'oos_end': str(period.get('oos_end') or ''),
        'total_return_pct': float(period.get('ret_pct', 0.0)),
        'annualized_return_pct': float(period.get('annual_return_pct', 0.0)),
        'min_full_year_return_pct': float(period.get('min_full_year_return_pct', 0.0)),
        'max_drawdown_pct': float(period.get('mdd', 0.0)),
        'test_score_romd': float(period.get('test_score_romd', 0.0)),
        'trade_count': int(period.get('trade_count', 0) or 0),
        'normal_trades': int(period.get('normal_trades', 0) or 0),
        'extended_trades': int(period.get('extended_trades', 0) or 0),
        'annual_trades': float(period.get('annual_trades', 0.0)),
        'fill_rate': float(period.get('reserved_buy_fill_rate', 0.0)),
        'win_rate': float(period.get('win_rate', 0.0)),
        'payoff': float(period.get('pf_payoff', 0.0)),
        'ev': float(period.get('pf_ev', 0.0)),
        'r_squared': float(period.get('r_squared', 0.0)),
        'monthly_win_rate': float(period.get('monthly_win_rate', 0.0)),
        'avg_exposure': float(period.get('avg_exposure', 0.0)),
        'final_equity': float(period.get('final_equity', 0.0)),
        'benchmark_total_return_pct': float(period.get('benchmark_return_pct', 0.0)),
        'benchmark_annualized_return_pct': float(period.get('benchmark_annual_return_pct', 0.0)),
        'benchmark_min_full_year_return_pct': float(period.get('benchmark_min_full_year_return_pct', 0.0)),
        'benchmark_max_drawdown_pct': float(period.get('benchmark_mdd', 0.0)),
        'benchmark_score_romd': float(period.get('benchmark_score_romd', 0.0)),
        'benchmark_r_squared': float(period.get('benchmark_r_squared', 0.0)),
        'benchmark_monthly_win_rate': float(period.get('benchmark_monthly_win_rate', 0.0)),
    }


def _format_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def _format_gate_actual(name: str, actual) -> str:
    if 'score' in name:
        return f"{float(actual):.3f}"
    if 'trades' in name or 'available' in name or 'count' in name:
        return str(int(actual))
    if isinstance(actual, bool):
        return 'PASS' if actual else 'FAIL'
    return _format_pct(actual)


def write_walk_forward_report(*, output_dir: str, params_payload: dict, dataset_label: str, report: dict, best_trial_number: int | None, source_db_path: str, session_ts: str | None = None):
    os.makedirs(output_dir, exist_ok=True)
    resolved_session_ts = session_ts or get_taipei_now().strftime('%Y%m%d_%H%M%S_%f')
    base_name = f'test_period_report_{resolved_session_ts}'
    json_path = os.path.join(output_dir, f'{base_name}.json')
    csv_path = os.path.join(output_dir, f'{base_name}.csv')
    md_path = os.path.join(output_dir, f'{base_name}.md')

    summary = dict(report.get('summary') or {})
    upgrade_gate = dict(report.get('upgrade_gate') or build_upgrade_gate_assessment(summary=summary))
    test_total = build_test_period_metrics(report)
    benchmark_test_total = {
        'oos_start': str(test_total.get('oos_start') or ''),
        'oos_end': str(test_total.get('oos_end') or ''),
        'total_return_pct': float(test_total.get('benchmark_total_return_pct', 0.0)),
        'annualized_return_pct': float(test_total.get('benchmark_annualized_return_pct', 0.0)),
        'min_full_year_return_pct': float(test_total.get('benchmark_min_full_year_return_pct', 0.0)),
        'max_drawdown_pct': float(test_total.get('benchmark_max_drawdown_pct', 0.0)),
        'test_score_romd': float(test_total.get('benchmark_score_romd', 0.0)),
        'r_squared': float(test_total.get('benchmark_r_squared', 0.0)),
        'monthly_win_rate': float(test_total.get('benchmark_monthly_win_rate', 0.0)),
    }
    payload = {
        'meta': {
            'dataset_label': str(dataset_label),
            'best_trial_number': None if best_trial_number is None else int(best_trial_number),
            'source_db_path': str(source_db_path),
            'session_ts': str(resolved_session_ts),
        },
        'params': dict(params_payload),
        'summary': summary,
        'upgrade_gate': upgrade_gate,
        'test_total': test_total,
        'benchmark_test_total': benchmark_test_total,
        'period': dict(report.get('period') or {}),
    }
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    period = dict(payload.get('period') or {})
    if period:
        fieldnames = list(period.keys())
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(period)
    else:
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['label', 'train_start', 'train_end', 'oos_start', 'oos_end', 'test_score_romd', 'ret_pct', 'mdd'])

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
        f"| 測試區間 | {test_total.get('oos_start', '')} ~ {test_total.get('oos_end', '')} | {benchmark_test_total.get('oos_start', '')} ~ {benchmark_test_total.get('oos_end', '')} |",
        f"| 測試 RoMD | {float(test_total.get('test_score_romd', 0.0)):.3f} | {float(benchmark_test_total.get('test_score_romd', 0.0)):.3f} |",
        f"| 測試總報酬率 | {_format_pct(test_total.get('total_return_pct', 0.0))} | {_format_pct(benchmark_test_total.get('total_return_pct', 0.0))} |",
        f"| 年化報酬率 | {_format_pct(test_total.get('annualized_return_pct', 0.0))} | {_format_pct(benchmark_test_total.get('annualized_return_pct', 0.0))} |",
        f"| 完整年度最差報酬 | {_format_pct(test_total.get('min_full_year_return_pct', 0.0))} | {_format_pct(benchmark_test_total.get('min_full_year_return_pct', 0.0))} |",
        f"| 最大回撤 | {_format_pct(test_total.get('max_drawdown_pct', 0.0))} | {_format_pct(benchmark_test_total.get('max_drawdown_pct', 0.0))} |",
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
    lines.extend([
        '',
        '## Test Period Detail',
        '',
        '| 區間 | 訓練區間 | 測試區間 | 測試 RoMD | 總報酬率 | MDD | 年化交易次數 | 買進成交率 | 0050 報酬率 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ])
    if period:
        lines.append(
            f"| {period.get('label', 'TEST')} | {period.get('train_start', '')} ~ {period.get('train_end', '')} | {period.get('oos_start', '')} ~ {period.get('oos_end', '')} | {float(period.get('test_score_romd', 0.0)):.3f} | {_format_pct(period.get('ret_pct', 0.0))} | {_format_pct(period.get('mdd', 0.0))} | {float(period.get('annual_trades', 0.0)):.2f} | {_format_pct(period.get('reserved_buy_fill_rate', 0.0))} | {_format_pct(period.get('benchmark_return_pct', 0.0))} |"
        )
    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines) + "\n")
    return {'json_path': json_path, 'csv_path': csv_path, 'md_path': md_path}


def _metric_triplet(*, champion, challenger, benchmark=None) -> dict:
    payload = {
        'champion': float(champion),
        'challenger': float(challenger),
        'delta': float(challenger) - float(champion),
    }
    if benchmark is not None:
        payload['benchmark'] = float(benchmark)
    return payload


def build_compare_assessment(*, champion_report: dict, challenger_report: dict) -> dict:
    champion_total = build_test_period_metrics(champion_report)
    challenger_total = build_test_period_metrics(challenger_report)
    challenger_upgrade_gate = dict((challenger_report or {}).get('upgrade_gate') or {})
    both_available = int(champion_total.get('period_count', 0)) >= 1 and int(challenger_total.get('period_count', 0)) >= 1
    challenger_score = float(challenger_total.get('test_score_romd', 0.0))
    champion_score = float(champion_total.get('test_score_romd', 0.0))
    checks = [
        {
            'name': 'test_period_available',
            'actual': 1 if both_available else 0,
            'threshold': '兩者都必須可計算',
            'passed': both_available,
            'severity': 'quality',
            'note': 'Champion 與 Challenger 都必須有單一連續 test holdout 績效。',
        },
        {
            'name': 'test_score_romd_vs_champion',
            'actual': challenger_score,
            'threshold': f"> {champion_score:.3f}",
            'passed': both_available and challenger_score > champion_score,
            'severity': 'quality',
            'note': '單一連續 test holdout 的 RoMD 必須超越現役 Champion。',
        },
    ]
    quality_checks = [check for check in checks if str(check.get('severity')) == 'quality']
    quality_pass = all(bool(check.get('passed')) for check in quality_checks)
    status = 'pass' if quality_pass else 'fail'
    recommendation = '候選版單一連續 test holdout RoMD 較高，建議升版。' if quality_pass else '候選版未超越現役 Champion 的單一連續 test holdout RoMD，維持現況。'
    return {
        'status': status,
        'recommended_for_promotion': bool(quality_pass),
        'quality_gate': {'status': status, 'checks': quality_checks},
        'coverage_gate': {'status': 'pass', 'checks': []},
        'checks': checks,
        'recommendation': recommendation,
        'challenger_upgrade_gate': challenger_upgrade_gate,
    }


def build_test_period_compare_payload(*, champion_payload: dict, champion_report: dict, challenger_payload: dict, challenger_report: dict, dataset_label: str, source_db_path: str, session_ts: str | None = None) -> dict:
    champion_total = build_test_period_metrics(champion_report)
    challenger_total = build_test_period_metrics(challenger_report)
    compare_assessment = build_compare_assessment(
        champion_report=champion_report,
        challenger_report=challenger_report,
    )
    return {
        'meta': {
            'dataset_label': str(dataset_label),
            'source_db_path': str(source_db_path),
            'session_ts': str(session_ts or get_taipei_now().strftime('%Y%m%d_%H%M%S_%f')),
            'champion_params_path': 'models/champion_params.json',
            'challenger_run_best_params_path': 'models/run_best_params.json',
        },
        'champion_params': dict(champion_payload or {}),
        'challenger_params': dict(challenger_payload or {}),
        'test_metrics_compare': {
            'test_score_romd': _metric_triplet(champion=champion_total.get('test_score_romd', 0.0), challenger=challenger_total.get('test_score_romd', 0.0), benchmark=challenger_total.get('benchmark_score_romd', 0.0)),
            'total_return_pct': _metric_triplet(champion=champion_total.get('total_return_pct', 0.0), challenger=challenger_total.get('total_return_pct', 0.0), benchmark=challenger_total.get('benchmark_total_return_pct', 0.0)),
            'annualized_return_pct': _metric_triplet(champion=champion_total.get('annualized_return_pct', 0.0), challenger=challenger_total.get('annualized_return_pct', 0.0), benchmark=challenger_total.get('benchmark_annualized_return_pct', 0.0)),
            'min_full_year_return_pct': _metric_triplet(champion=champion_total.get('min_full_year_return_pct', 0.0), challenger=challenger_total.get('min_full_year_return_pct', 0.0), benchmark=challenger_total.get('benchmark_min_full_year_return_pct', 0.0)),
            'max_drawdown_pct': _metric_triplet(champion=champion_total.get('max_drawdown_pct', 0.0), challenger=challenger_total.get('max_drawdown_pct', 0.0), benchmark=challenger_total.get('benchmark_max_drawdown_pct', 0.0)),
            'annual_trades': _metric_triplet(champion=champion_total.get('annual_trades', 0.0), challenger=challenger_total.get('annual_trades', 0.0)),
            'fill_rate': _metric_triplet(champion=champion_total.get('fill_rate', 0.0), challenger=challenger_total.get('fill_rate', 0.0)),
        },
        'compare_assessment': compare_assessment,
        'oos_range': {
            'start': str(challenger_total.get('oos_start') or champion_total.get('oos_start') or ''),
            'end': str(challenger_total.get('oos_end') or champion_total.get('oos_end') or ''),
        },
    }


def write_test_period_compare_report(*, output_dir: str, compare_payload: dict):
    os.makedirs(output_dir, exist_ok=True)
    session_ts = str((compare_payload.get('meta') or {}).get('session_ts') or get_taipei_now().strftime('%Y%m%d_%H%M%S_%f'))
    base_name = f'test_period_compare_{session_ts}'
    json_path = os.path.join(output_dir, f'{base_name}.json')
    csv_path = os.path.join(output_dir, f'{base_name}.csv')
    md_path = os.path.join(output_dir, f'{base_name}.md')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(compare_payload, handle, indent=2, ensure_ascii=False)

    metrics = dict(compare_payload.get('test_metrics_compare') or {})
    csv_rows = []
    for metric_name, metric in metrics.items():
        csv_rows.append({
            'metric': metric_name,
            'champion': metric.get('champion', 0.0),
            'challenger': metric.get('challenger', 0.0),
            'benchmark': metric.get('benchmark', ''),
            'delta': metric.get('delta', 0.0),
        })
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['metric', 'champion', 'challenger', 'benchmark', 'delta'])
        writer.writeheader()
        writer.writerows(csv_rows)

    assessment = dict(compare_payload.get('compare_assessment') or {})
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
        '',
        '| 指標 | Champion | Challenger | 0050 | Delta |',
        '|---|---:|---:|---:|---:|',
    ]
    for metric_name, metric in metrics.items():
        benchmark = metric.get('benchmark', '-')
        if benchmark == '':
            benchmark = '-'
        lines.append(f"| {metric_name} | {metric.get('champion', 0.0)} | {metric.get('challenger', 0.0)} | {benchmark} | {metric.get('delta', 0.0)} |")
    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines) + "\n")
    return {'json_path': json_path, 'csv_path': csv_path, 'md_path': md_path}
