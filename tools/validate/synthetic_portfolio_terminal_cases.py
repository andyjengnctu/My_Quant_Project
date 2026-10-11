"""Behavior contracts for the terminal portfolio phases; not a second runner."""

import ast
import inspect
from unittest.mock import patch

from pandas import Timestamp

from .checks import bind_synthetic_case


def validate_portfolio_terminal_owner_contract_case(_base_params):
    from core import (
        portfolio_engine, portfolio_terminal_equity,
        portfolio_terminal_statistics, portfolio_terminal_reporting,
    )
    from core.portfolio_run_result import PortfolioTrainingResult, PortfolioReportingResult
    from core.portfolio_terminal_statistics import PortfolioPeriodReports

    case_id = 'META_PORTFOLIO_TERMINAL_OWNER'
    results, summary, check, _ = bind_synthetic_case(case_id, 'portfolio_terminal')

    engine = ast.parse(inspect.getsource(portfolio_engine.run_portfolio_timeline))
    calls = [(node.lineno, node.func.id) for node in ast.walk(engine)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    stages = (
        'closeout_open_positions', 'reconcile_portfolio_terminal_equity',
        'summarize_portfolio_terminal_metrics', 'build_portfolio_period_reports',
        'publish_portfolio_profile_stats', 'assemble_portfolio_timeline_result',
    )
    positions = [next((line for line, name in calls if name == stage), None) for stage in stages]
    check('closeout_then_account_then_publish', True,
          all(item is not None for item in positions) and positions == sorted(positions))
    check('one_terminal_owner_per_stage', [1] * len(stages),
          [sum(name == stage for _, name in calls) for stage in stages])
    check('terminal_equity_owner_identical', True,
          portfolio_engine.reconcile_portfolio_terminal_equity is portfolio_terminal_equity.reconcile_portfolio_terminal_equity)
    check('terminal_metric_owner_identical', True,
          portfolio_engine.summarize_portfolio_terminal_metrics is portfolio_terminal_statistics.summarize_portfolio_terminal_metrics)
    check('terminal_period_owner_identical', True,
          portfolio_engine.build_portfolio_period_reports is portfolio_terminal_statistics.build_portfolio_period_reports)
    check('terminal_sidecar_owner_identical', True,
          portfolio_engine.publish_portfolio_profile_stats is portfolio_terminal_reporting.publish_portfolio_profile_stats)
    check('terminal_return_owner_identical', True,
          portfolio_engine.assemble_portfolio_timeline_result is portfolio_terminal_reporting.assemble_portfolio_timeline_result)

    from tools.local_regression.meta_quality_targets import CORE_TRADING_COVERAGE_TARGETS
    check('new_terminal_owners_are_covered_not_just_facade', True, all(
        'core/' + module + '.py' in CORE_TRADING_COVERAGE_TARGETS
        for module in ('portfolio_terminal_equity', 'portfolio_terminal_statistics',
                       'portfolio_terminal_reporting')
    ))

    last = Timestamp('2024-06-28')
    y_end, m_end, q_end = {2024: 1000.0}, {(2024, 6): 1000.0}, {(2024, 2): 1000.0}
    monthly, bm_monthly = [1000.0], [100.0]
    curve = [{'Equity': 1000.0, 'Strategy_Return_Pct': 0,
              'Invested_Amount': 42, 'Exposure_Pct': 4.2}]
    input_args = dict(
        closeout_equity_milli=1000000, fee_rebate_state={}, last_date=last,
        year_end_equity=y_end, month_end_equity=m_end, quarter_end_equity=q_end,
        peak_equity=1050000, max_drawdown=2.0, sorted_dates=[last], start_idx=0,
        monthly_equities=monthly, benchmark_loop_enabled=True,
        current_bm_px=102.0, bm_monthly_equities=bm_monthly,
        initial_capital=1000.0, equity_curve=curve,
        is_training=False, capture_equity_curve=False,
    )
    # Accrued fees are receivables in terminal equity, never spendable cash.
    with patch.object(portfolio_terminal_equity, 'get_fee_rebate_receivable_milli', return_value=1200):
        out = portfolio_terminal_equity.reconcile_portfolio_terminal_equity(**input_args)
    check('rebate_is_in_equity', 1001.2, out.final_equity_money)
    check('terminal_return_exact_basis', (1001.2 - 1000.0) / 1000.0 * 100, out.total_return)
    check('terminal_closeout_mdd_updated', (1050000 - 1001200) / 1050000 * 100, out.max_drawdown)
    check('year_end_after_closeout', 1001.2, y_end[2024])
    check('month_end_after_closeout', 1001.2, m_end[(2024, 6)])
    check('quarter_end_after_closeout', 1001.2, q_end[(2024, 2)])
    check('monthly_last_closeout_count', [1000.0, 1001.2], monthly)
    check('benchmark_last_closeout_count', [100.0, 102.0], bm_monthly)
    check('nontraining_curve_final_mark', (1001.2, 0.0, 0.0),
          (curve[-1]['Equity'], curve[-1]['Invested_Amount'], curve[-1]['Exposure_Pct']))
    check('original_closeout_cash_immutable', 1000000, input_args['closeout_equity_milli'])
    # Empty timeline must not create fictitious period rows or curve samples.
    with patch.object(portfolio_terminal_equity, 'get_fee_rebate_receivable_milli', return_value=0):
        empty_curve, empty_monthly = [], [1000.0]
        out_empty = portfolio_terminal_equity.reconcile_portfolio_terminal_equity(**{
            **input_args, 'last_date': None, 'sorted_dates': [], 'year_end_equity': {},
            'month_end_equity': {}, 'quarter_end_equity': {}, 'monthly_equities': empty_monthly,
            'equity_curve': empty_curve, 'is_training': True,
        })
    check('empty_timeline_return', 1000.0, out_empty.final_equity_money)
    check('empty_timeline_has_no_added_periods', [1000.0], empty_monthly)
    check('empty_timeline_curve_unchanged', [], empty_curve)

    trades = [
        {'pnl': 100.0, 'r_mult': 1.5, 'entry_type': 'normal'},
        {'pnl': -25.0, 'r_mult': -0.5, 'entry_type': 'reentry'},
    ]
    metrics_args = dict(
        closed_trades_stats=trades, pit_stats_index={}, sorted_dates=[last],
        start_idx=0, profile_requested=False, active_context_resolver=None,
        active_context_ensemble_resolver=None, total_exposure=25.0, sim_days=1,
        portfolio_entry_stats={'filled_buy_count': 2}, total_missed_buys=2,
        initial_capital=1000.0, final_equity_money=1100.0,
        benchmark_start_price=100.0, bm_ret_pct=5.0, benchmark_period_stats=None,
    )
    by_method = {}
    for method in ('A', 'B'):
        by_method[method] = portfolio_terminal_statistics.summarize_portfolio_terminal_metrics(
            **metrics_args, ev_calc_method_getter=lambda m=method: m,
        )
    check('risk_ev_a_keeps_average_r', 0.5, by_method['A'].pf_ev)
    check('risk_ev_b_keeps_payoff_formula', 1.0, by_method['B'].pf_ev)
    check('net_payoff_uses_pnl', 4.0, by_method['A'].pf_payoff)
    check('round_trip_win_rate', 50.0, by_method['A'].win_rate)
    check('entry_type_counts_from_canonical_owner', (1, 1, 1),
          (by_method['A'].normal_trade_count, by_method['A'].reentry_trade_count,
           by_method['A'].breakout_trade_count))
    check('reserved_fill_rate_same_denominator', 50.0, by_method['A'].reserved_buy_fill_rate)
    check('exposure_from_daily_accumulator', 25.0, by_method['A'].avg_exp)
    # Zero trades must not probe an EV method or fabricate a trade count.
    zero = portfolio_terminal_statistics.summarize_portfolio_terminal_metrics(
        **{**metrics_args, 'closed_trades_stats': [], 'total_missed_buys': 0,
           'portfolio_entry_stats': {'filled_buy_count': 0}},
        ev_calc_method_getter=lambda: (_ for _ in ()).throw(AssertionError('unexpected EV probe')),
    )
    check('zero_trades_keep_zero_stats', (0, 0.0, 0.0, 0.0),
          (zero.trade_count, zero.win_rate, zero.pf_ev, zero.pf_payoff))
    check('zero_orders_keep_zero_fill_rate', 0.0, zero.reserved_buy_fill_rate)

    # Cached benchmark period rows are filtered to identical full periods;
    # noncached paths must delegate to the canonical per-period owners.
    yearly = {'yearly_return_rows': [
        {'year': 2024, 'is_full_year': True}, {'year': 2025, 'is_full_year': False},
    ]}
    monthly_rows = {'monthly_return_rows': [
        {'year': 2024, 'month': 5, 'is_full_month': True},
        {'year': 2024, 'month': 6, 'is_full_month': False},
    ]}
    quarterly = {'quarterly_return_rows': [
        {'year': 2024, 'quarter': 2, 'is_full_quarter': True},
    ]}
    cached = dict(
        bm_yearly_return_rows=[{'year': 2024, 'year_return_pct': 3.0}, {'year': 2025, 'year_return_pct': -7.0}],
        bm_monthly_return_rows=[{'year': 2024, 'month': 5, 'month_return_pct': 2.0},
                                {'year': 2024, 'month': 6, 'month_return_pct': -8.0}],
        bm_quarterly_return_rows=[{'year': 2024, 'quarter': 2, 'quarter_return_pct': 4.0}],
    )
    periods_args = dict(
        sorted_dates=[last], year_start_equity={}, year_end_equity={},
        year_first_sim_date={}, year_last_sim_date={},
        month_start_equity={}, month_end_equity={}, month_first_sim_date={},
        month_last_sim_date={}, quarter_start_equity={}, quarter_end_equity={},
        quarter_first_sim_date={}, quarter_last_sim_date={}, benchmark_period_stats=cached,
        benchmark_data=None,
    )
    with (patch.object(portfolio_terminal_statistics, 'build_full_year_return_stats', return_value=yearly),
          patch.object(portfolio_terminal_statistics, 'build_full_month_return_stats', return_value=monthly_rows),
          patch.object(portfolio_terminal_statistics, 'build_full_quarter_return_stats', return_value=quarterly)):
        periods = portfolio_terminal_statistics.build_portfolio_period_reports(**periods_args)
    check('cached_benchmark_years_match_full_only', [2024],
          [row['year'] for row in periods.bm_yearly_stats['bm_yearly_return_rows']])
    check('cached_benchmark_months_match_full_only', [5],
          [row['month'] for row in periods.bm_monthly_stats['bm_monthly_return_rows']])
    check('cached_benchmark_quarters_match_full_only', [2],
          [row['quarter'] for row in periods.bm_quarterly_stats['bm_quarterly_return_rows']])
    check('cached_benchmark_year_min', 3.0, periods.bm_yearly_stats['bm_min_full_year_return_pct'])
    check('cached_benchmark_month_min', 2.0, periods.bm_monthly_stats['bm_min_month_return_pct'])

    args = dict(
        is_training=True, equity_curve=[], trade_history=[], benchmark_ticker='0050',
        terminal_metrics=by_method['A'], total_return=10.0, max_drawdown=5.0,
        final_equity_money=1100.0, max_exp=40.0, bm_ret_pct=5.0,
        bm_max_drawdown=8.0, total_missed_buys=2, total_missed_sells=1,
        r_squared=0.25, monthly_win_rate=50.0,
        bm_r_squared=0.16, bm_monthly_win_rate=33.0,
    )
    training_tuple = portfolio_terminal_reporting.assemble_portfolio_timeline_result(**args)
    check('training_return_preserves_23_field_order', tuple(PortfolioTrainingResult._fields),
          tuple(PortfolioTrainingResult._fields[:len(training_tuple)]))
    check('training_return_is_plain_tuple', tuple, type(training_tuple))
    check('training_return_metric_positions', (10.0, 5.0, 2, 1100.0, 0.5, 4.0),
          (training_tuple[0], training_tuple[1], training_tuple[2], training_tuple[3],
           training_tuple[9], training_tuple[10]))
    reporting_tuple = portfolio_terminal_reporting.assemble_portfolio_timeline_result(**{
        **args, 'is_training': False, 'equity_curve': [
            {'Equity': 1100.0, 'Benchmark_0050_Pct': 5.0, 'Date': last}
        ], 'trade_history': [{'Ticker': '2330', 'NetPnL': 75.0}],
    })
    check('report_return_is_plain_tuple', tuple, type(reporting_tuple))
    check('report_return_has_25_fields', len(PortfolioReportingResult._fields), len(reporting_tuple))
    check('report_benchmark_comes_from_final_curve', 5.0, reporting_tuple[11])
    check('report_trade_row_is_preserved', '2330', reporting_tuple[1].iloc[0]['Ticker'])

    timings = {key: 0.0 for key in (
        'ticker_dates_sec', 'build_trade_index_sec', 'day_loop_sec', 'candidate_scan_sec',
        'day_setup_sec', 'reentry_sec', 'resource_select_sec', 'cleanup_sec',
        'bookkeeping_sec', 'rotation_sec', 'settle_sec', 'buy_sec', 'equity_mark_sec',
        'closeout_sec', 'curve_stats_sec',
    )}
    profile = {}
    fee_state = {}
    with (patch.object(portfolio_terminal_reporting, 'get_fee_rebate_settled_milli', return_value=4500),
          patch.object(portfolio_terminal_reporting, 'get_fee_rebate_receivable_milli', return_value=1200)):
        portfolio_terminal_reporting.publish_portfolio_profile_stats(
            profile_stats=profile, fee_rebate_state=fee_state, timing_enabled=False,
            portfolio_start=None, timings=timings, candidate_subprofile=None,
            idle_fast_path_days=7, closed_trades_stats=trades, terminal_metrics=by_method['A'],
            period_reports=PortfolioPeriodReports(*({'period_marker': i} for i in range(6))),
            is_training=False, active_level_rows=[], capacity_rows=[],
            capture_equity_curve=False, equity_curve=[],
        )
    check('settled_rebate_only_when_settled', 4.5, profile['fee_rebate_settled'])
    check('receivable_reported_separately', 1.2, profile['fee_rebate_receivable_end'])
    check('profile_does_not_resolve_rebate_to_cash', 'fee_rebate_receivable_end',
          next(key for key in profile if key.startswith('fee_rebate_receivable')))
    check('profile_closed_trades_is_audit_copy', False,
          profile['closed_trade_rows'][0] is trades[0])
    check('profile_timing_disabled_is_zero', (0.0, 0.0),
          (profile['portfolio_wall_sec'], profile['portfolio_day_other_sec']))
    check('profile_full_period_rows_merge', 5, profile['period_marker'])
    check('profile_keeps_closed_trade_types', (1, 1),
          (profile['normal_trades'], profile['reentry_trades']))
    check('training_tuple_length_is_23', 23, len(training_tuple))
    check('report_tuple_length_is_25', 25, len(reporting_tuple))

    summary['contract_assertions'] = len(results)
    return results, summary
