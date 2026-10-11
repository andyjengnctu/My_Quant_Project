"""Candidate gate and Replay-output owner invariants for Portfolio timelines.

These tests exercise each reusable capability with isolated inputs, not current
strategy identity or a second portfolio simulator.
"""

import ast
import inspect
from types import SimpleNamespace

from pandas import Timestamp

from .checks import bind_synthetic_case


def validate_portfolio_candidate_replay_owner_contract_case(_base_params):
    from core import portfolio_engine, portfolio_day_candidates, portfolio_replay_diagnostics

    case_id = 'META_PORTFOLIO_CANDIDATE_REPLAY_OWNER'
    results, summary, check, _ = bind_synthetic_case(case_id, 'portfolio_candidate_replay')
    today = Timestamp('2025-01-15')
    context = SimpleNamespace(
        ensemble_members=[], ensemble_contexts=[], ensemble_min_agree=None,
        use_param_ensemble=False, params=_base_params,
        all_dfs_fast={'2330': object()}, pit_stats_index={'2330': object()},
        pit_stats_cursor={}, normal_setup_index={today: [{'ticker': '2330'}]},
    )

    # The engine coordinates one owner for eligibility and another for Replay;
    # the canonical daily execution/mark phases must remain ordered after them.
    engine_tree = ast.parse(inspect.getsource(portfolio_engine.run_portfolio_timeline))
    calls = [(node.lineno, node.func.id) for node in ast.walk(engine_tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    chronological = (
        'resolve_portfolio_candidate_gate', 'build_portfolio_day_candidate_rows',
        'select_portfolio_candidate_orders', 'append_portfolio_selector_trace',
        'record_portfolio_candidate_replay', 'rotate_portfolio_day',
        'settle_and_execute_portfolio_day', 'mark_portfolio_day_equity',
        'append_portfolio_trade_replay',
    )
    positions = [next((line for line, name in calls if name == stage), None)
                 for stage in chronological]
    check('candidate_then_event_then_replay_chronology', True,
          all(item is not None for item in positions) and positions == sorted(positions))
    check('one_call_per_candidate_replay_phase', [1] * len(chronological),
          [sum(name == stage for _, name in calls) for stage in chronological])
    check('candidate_owners_match_engine_imports', True,
          portfolio_engine.resolve_portfolio_candidate_gate is portfolio_day_candidates.resolve_portfolio_candidate_gate
          and portfolio_engine.build_portfolio_day_candidate_rows is portfolio_day_candidates.build_portfolio_day_candidate_rows
          and portfolio_engine.select_portfolio_candidate_orders is portfolio_day_candidates.select_portfolio_candidate_orders)
    check('replay_owners_match_engine_imports', True,
          portfolio_engine.append_portfolio_selector_trace is portfolio_replay_diagnostics.append_portfolio_selector_trace
          and portfolio_engine.record_portfolio_candidate_replay is portfolio_replay_diagnostics.record_portfolio_candidate_replay
          and portfolio_engine.append_portfolio_trade_replay is portfolio_replay_diagnostics.append_portfolio_trade_replay)

    gate_args = dict(
        today=today, day_context=context, portfolio={'held': {}}, sold_today={'sold'},
        normal_setup_entries_today=[{'ticker': '2330'}], active_extended_signals={},
        active_extended_signals_by_member={}, is_training=True, replay_counts=None,
        enable_rotation=False, max_positions=2,
    )
    gate = portfolio_day_candidates.resolve_portfolio_candidate_gate(**gate_args)
    check('pre_market_slots_include_sold_today', 2, gate.pre_market_occupied)
    check('no_capacity_training_gate', (True, True), (gate.has_candidates, gate.no_entry_capacity))
    replay_gate = portfolio_day_candidates.resolve_portfolio_candidate_gate(**{
        **gate_args, 'replay_counts': {},
    })
    check('replay_does_not_disable_candidate_build', False, replay_gate.no_entry_capacity)
    rotation_gate = portfolio_day_candidates.resolve_portfolio_candidate_gate(**{
        **gate_args, 'enable_rotation': True,
    })
    check('rotation_preserves_candidate_build', False, rotation_gate.no_entry_capacity)
    room_gate = portfolio_day_candidates.resolve_portfolio_candidate_gate(**{
        **gate_args, 'max_positions': 3,
    })
    check('available_slot_is_not_full', False, room_gate.no_entry_capacity)
    no_source = portfolio_day_candidates.resolve_portfolio_candidate_gate(**{
        **gate_args, 'normal_setup_entries_today': [],
    })
    check('empty_single_sources', False, no_source.has_candidates)
    extended = portfolio_day_candidates.resolve_portfolio_candidate_gate(**{
        **gate_args, 'normal_setup_entries_today': [],
        'active_extended_signals': {'2330': object()},
    })
    check('extended_signals_are_candidate_source', True, extended.has_candidates)
    ensemble_context = SimpleNamespace(**{
        **vars(context), 'use_param_ensemble': True,
        'ensemble_members': [{'seed': 11, 'params_obj': _base_params},
                             {'seed': 12, 'params_obj': _base_params}],
        'ensemble_contexts': [{'normal_setup_index': {}},
                              {'normal_setup_index': {today: [{'ticker': '2330'}]}}],
        'ensemble_min_agree': 2,
    })
    ensemble_gate = portfolio_day_candidates.resolve_portfolio_candidate_gate(**{
        **gate_args, 'day_context': ensemble_context,
        'normal_setup_entries_today': [],
    })
    check('ensemble_context_membership_instead_of_single', True, ensemble_gate.has_candidates)
    ensemble_signal = portfolio_day_candidates.resolve_portfolio_candidate_gate(**{
        **gate_args, 'day_context': SimpleNamespace(**{
            **vars(ensemble_context), 'ensemble_contexts': []}),
        'normal_setup_entries_today': [],
        'active_extended_signals_by_member': {'11': {'2330': object()}},
    })
    check('ensemble_extended_signal_membership', True, ensemble_signal.has_candidates)

    # The sole producer remains the existing single / consensus callback;
    # the wrapper cannot silently change PIT cursor, setup rows or model identity.
    producer_calls = []
    callback_single, callback_ensemble = object(), object()
    candidate_rows, orderable_rows = [{'ticker': '2330'}], [{'ticker': '2330', 'qty': 2}]
    def replay_stub(day, phase, callback, **kwargs):
        producer_calls.append((day, phase, callback, kwargs))
        return candidate_rows, orderable_rows, {'2330'}

    build_args = dict(
        today=today, day_context=context, portfolio={}, sold_today=set(),
        active_extended_signals={}, active_extended_signals_by_member={},
        current_equity_money=100.0, initial_capital=100.0, sizing_equity=100.0,
        replay_counts={}, replay_execution_rows=[], replay_selector_trace_rows=[],
        enable_rotation=False, is_training=False, verbose=False,
        extended_day_market_cache={}, persistent_sizing_fee_context_cache={},
        persistent_sizing_qty_cache={}, candidate_subprofile={},
        replay_phase=replay_stub, single_candidate_builder=callback_single,
        ensemble_candidate_builder=callback_ensemble,
    )
    plain = portfolio_day_candidates.build_portfolio_day_candidate_rows(**build_args)
    check('single_builder_keeps_row_objects', True,
          plain[0] is candidate_rows and plain[1] is orderable_rows)
    check('single_source_callback_identity', ('build_daily_candidates', callback_single),
          (producer_calls[-1][1], producer_calls[-1][2]))
    check('single_builder_preserves_exact_pit_cursor', True,
          producer_calls[-1][3]['pit_stats_cursor'] is context.pit_stats_cursor)
    check('single_builder_collects_replay_candidates', True,
          producer_calls[-1][3]['collect_all_candidates'])
    check('single_builder_no_compact_replay', False,
          producer_calls[-1][3]['compact_training_rows'])
    training = portfolio_day_candidates.build_portfolio_day_candidate_rows(**{
        **build_args, 'is_training': True, 'replay_counts': None,
        'replay_execution_rows': None, 'replay_selector_trace_rows': None,
    })
    check('training_without_ranking_uses_compact_rows', True,
          producer_calls[-1][3]['compact_training_rows'])
    consensus = portfolio_day_candidates.build_portfolio_day_candidate_rows(**{
        **build_args, 'day_context': ensemble_context,
    })
    check('ensemble_builder_reuses_canonical_callback',
          ('build_daily_ensemble_candidates', callback_ensemble),
          (producer_calls[-1][1], producer_calls[-1][2]))
    check('ensemble_vote_threshold_from_context', 2,
          producer_calls[-1][3]['min_agree'])
    check('ensemble_builder_passes_members_by_reference', True,
          producer_calls[-1][3]['ensemble_members'] is ensemble_context.ensemble_members)
    check('builder_does_not_mutate_source_rows', (candidate_rows, orderable_rows),
          (consensus[0], training[1]))

    cached_rows = {'2330': orderable_rows[0]}
    orderable_cache = {('__meta__', 'orderable_candidate_rows_by_ticker'): cached_rows}
    check('cleanup_reuses_producer_index_identity', True,
          portfolio_day_candidates.resolve_cleanup_candidate_rows(
              orderable_rows, is_ensemble=False,
              extended_day_market_cache=orderable_cache) is cached_rows)
    check('cleanup_fallback_ticker_mapping', cached_rows,
          portfolio_day_candidates.resolve_cleanup_candidate_rows(
              orderable_rows, is_ensemble=False,
              extended_day_market_cache={}))
    check('ensemble_does_not_consume_single_member_cleanup', None,
          portfolio_day_candidates.resolve_cleanup_candidate_rows(
              orderable_rows, is_ensemble=True,
              extended_day_market_cache=orderable_cache))

    selector_calls = []
    def selector_phase(day, phase, callback, rows, **kwargs):
        selector_calls.append((day, phase, callback, rows, kwargs))
        return list(reversed(rows)), {'pre_market_order_limit': 3}
    reorder_callback, action_callback = object(), lambda rows, _diag: rows[:1]
    select_args = dict(
        today=today, orderable_rows=orderable_rows, replay_selector_trace_rows=None,
        replay_execution_rows=None, replay_counts=None, is_training=True,
        params=_base_params, available_cash=950, sizing_equity=1000,
        pre_market_occupied=1, max_positions=2, replay_phase=selector_phase,
        reorder_callback=reorder_callback, select_action_callback=action_callback,
    )
    fast_selection = portfolio_day_candidates.select_portfolio_candidate_orders(**select_args)
    check('training_noop_preserves_same_rows_object', True,
          fast_selection.orderable_rows is orderable_rows
          and fast_selection.action_rows is orderable_rows)
    check('training_noop_never_calls_selector', [], selector_calls)
    selected = portfolio_day_candidates.select_portfolio_candidate_orders(**{
        **select_args, 'replay_counts': {}, 'replay_selector_trace_rows': [],
    })
    check('resource_selector_uses_canonical_phase_name', 'resource_aware_binary_order',
          selector_calls[-1][1])
    check('resource_selector_callbacks_not_duplicated', reorder_callback,
          selector_calls[-1][2])
    check('resource_selector_uses_pre_market_cash', (950, 1000, 1),
          (selector_calls[-1][4]['available_cash'],
           selector_calls[-1][4]['sizing_equity'],
           selector_calls[-1][4]['pre_market_occupied']))
    check('selector_trace_has_pre_order_copy', orderable_rows,
          selected.trace_input_rows)
    check('selector_trace_copy_not_alias', True,
          selected.trace_input_rows is not orderable_rows)
    check('selector_action_rows_use_original_decision_callback', orderable_rows,
          selected.action_rows)
    check('selector_diagnostics_are_forwarded', 3,
          selected.diagnostic['pre_market_order_limit'])

    # Replay is output-only. Selector repair trace is kept byte-shape-equivalent
    # and does not change the action list or its resource selection metrics.
    repair_calls = []
    def repair_builder(rows, **kwargs):
        repair_calls.append((rows, kwargs))
        return {'repair_steps_trace': [
            {'step': 1, 'out_row': {'ticker': '2330'}, 'in_row': {'ticker': '2317'},
             'before_selected_count': 2, 'after_selected_count': 2}
        ], 'diagnostic_token': 'unchanged'}
    def snapshot(candidate, *, fallback_trade_date, is_orderable):
        return {'ticker': candidate['ticker'], 'is_orderable': is_orderable,
                'trade_date': str(fallback_trade_date.date())}
    traces = []
    diag = {'pre_market_order_limit': 2, 'direct_score_order_feasible': False,
            'max_dl_repair_steps': 1, 'max_dl_feasible_ascent_steps': 0,
            '_selector_trace_baskets': {'raw_top_n': [{'ticker': '2330'}],
                                        'feasible_ascent_final': [{'ticker': '2317'}]}}
    trace_args = dict(
        today=today, replay_selector_trace_rows=traces,
        resource_selection_input_rows=orderable_rows,
        resource_selection_diag=diag, available_cash=950,
        sizing_equity=1000, params=_base_params,
        repair_mechanism_builder=repair_builder, candidate_snapshot=snapshot,
    )
    portfolio_replay_diagnostics.append_portfolio_selector_trace(**trace_args)
    check('repair_and_basket_trace_order',
          ['repair_summary', 'repair_swap', 'repair_swap', 'basket_stage', 'basket_stage'],
          [row['trace_kind'] for row in traces])
    check('repair_summary_does_not_leak_internal_steps_trace', False,
          'repair_steps_trace' in traces[0])
    check('repair_row_tracks_in_and_out', ['out', 'in'],
          [row['repair_role'] for row in traces[1:3]])
    check('basket_stage_rank_starts_at_one', [1, 1],
          [row['stage_rank'] for row in traces[3:]])
    check('selector_diagnostic_is_read_only', 2, diag['pre_market_order_limit'])
    check('repair_builder_gets_original_orderable_rows', True,
          repair_calls[0][0] is orderable_rows)
    portfolio_replay_diagnostics.append_portfolio_selector_trace(**{
        **trace_args, 'replay_selector_trace_rows': None,
        'repair_mechanism_builder': lambda *_args, **_kwargs:
            (_ for _ in ()).throw(AssertionError('trace should be disabled')),
    })
    check('disabled_trace_has_no_side_effect', 1, len(repair_calls))

    buckets = {'2330': None}
    portfolio_replay_diagnostics.initialize_portfolio_replay_counts(buckets)
    check('existing_malformed_bucket_recovers', True, isinstance(buckets['2330'], dict))
    check('bucket_keys_in_original_order',
          ['candidate_dates', 'orderable_dates', 'candidate_rows', 'orderable_rows', 'trade_rows'],
          list(buckets['2330']))
    trades = [{'Ticker': '2330', 'Net': 1}]
    execution_rows = []
    def execution_snapshot(candidate, *, fallback_trade_date, all_dfs_fast, sizing_equity):
        return {'ticker': candidate['ticker'], 'date': str(fallback_trade_date.date()),
                'cash_basis': sizing_equity}
    captured_start = portfolio_replay_diagnostics.record_portfolio_candidate_replay(
        today=today, replay_counts=buckets,
        candidates_today=[{'ticker': '2330'}, {'ticker': '2317'}],
        orderable_candidates_today=[{'ticker': '2330'}],
        replay_execution_rows=execution_rows, trade_history=trades,
        all_dfs_fast=context.all_dfs_fast, sizing_equity=800,
        candidate_snapshot=snapshot, execution_snapshot=execution_snapshot,
    )
    check('capture_start_before_new_trade', 1, captured_start)
    check('capture_candidate_by_ticker', (1, 1),
          (len(buckets['2330']['candidate_rows']), len(buckets['2317']['candidate_rows'])))
    check('capture_only_orderable_tickers', (1, 0),
          (len(buckets['2330']['orderable_rows']), len(buckets['2317']['orderable_rows'])))
    check('capture_execution_cash_basis', 800, execution_rows[0]['cash_basis'])
    trades.extend([{'Ticker': '2330', 'Net': 2}, {'Ticker': '2317', 'Net': -1}])
    portfolio_replay_diagnostics.append_portfolio_trade_replay(
        replay_counts=buckets, before_trade_rows=captured_start, trade_history=trades,
    )
    check('trade_replay_only_appends_new_post_start_trades',
          ([2], [-1]),
          ([row['Net'] for row in buckets['2330']['trade_rows']],
           [row['Net'] for row in buckets['2317']['trade_rows']]))
    disabled_start = portfolio_replay_diagnostics.record_portfolio_candidate_replay(
        today=today, replay_counts=None, candidates_today=[],
        orderable_candidates_today=[], replay_execution_rows=None,
        trade_history=trades, all_dfs_fast={}, sizing_equity=0,
        candidate_snapshot=snapshot, execution_snapshot=execution_snapshot,
    )
    check('disabled_capture_returns_minus_one', -1, disabled_start)
    portfolio_replay_diagnostics.append_portfolio_trade_replay(
        replay_counts=None, before_trade_rows=-1, trade_history=trades,
    )
    check('disabled_replay_keeps_trade_list_unmodified', 3, len(trades))

    summary['contract_assertions'] = len(results)
    return results, summary
