"""Candidate gate and Replay-output owner invariants for Portfolio timelines.

These tests exercise each reusable capability with isolated inputs, not current
strategy identity or a second portfolio simulator.
"""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import patch

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

    _check_normal_setup_state_single_owner(check, _base_params)
    summary['contract_assertions'] = len(results)
    return results, summary


def _check_normal_setup_state_single_owner(check, _base_params):
    """The full-capacity tracking path and candidate path share PIT/state ownership."""
    from core import portfolio_candidates as candidates

    # Regression on *real* exported producers with isolated data providers.
    # Never replace the decision routines themselves or derive expected states
    # from today's engine output.
    today = Timestamp('2025-01-15')
    entries = [
        ('GOOD', 1, 2), ('HELD', 1, 2), ('SOLD', 1, 2),
        ('PIT_REJECT', 1, 2), ('NO_PLAN', 1, 2),
    ]
    all_fast = {ticker: ticker for ticker, _, _ in entries}
    pit_index = {ticker: ticker for ticker, _, _ in entries}
    cursor = {'known_as_of': '2025-01-14'}
    params = SimpleNamespace(
        use_breakout_quality_ranking=True,
        breakout_quality_filter_id='isolated-filter',
        high_len=260,
    )
    events = []

    def signal_values(ticker, y_pos):
        events.append(('signal', ticker))
        return (100.0, 2.0, 99.0, Timestamp('2025-01-14'))

    def security_profile(ticker):
        events.append(('security', ticker))
        return {'family': 'stock', 'ticker': ticker}

    def pit_stats(index, trade_date, params_obj, *, cursor_state, ticker):
        events.append(('pit', ticker))
        if trade_date != today or cursor_state is not cursor or ticker != index:
            raise AssertionError('normal setup PIT inputs were changed')
        return (ticker != 'PIT_REJECT', 2.0, 0.6, 10, 4.0)

    def quality_rank(*, params, ticker, signal_date, information_date):
        events.append(('rank', ticker))
        if signal_date != information_date:
            raise AssertionError('normal setup ranking information date drift')
        return {'ticker': ticker}

    def create_state(limit, atr, params_obj, *, ticker, security_profile, signal_date):
        events.append(('state', ticker))
        return {'source': 'normal', 'ticker': ticker, 'signal_date': signal_date}

    def attach(state, quality):
        events.append(('attach', state['ticker']))
        state['quality'] = quality

    def normal_plan(limit, atr, equity, params_obj, *, ticker, **kwargs):
        events.append(('plan', ticker))
        if ticker == 'NO_PLAN':
            return None
        return {
            'limit_price': 100.0, 'init_sl': 95.0, 'init_trail': 96.0,
            'target_price': 110.0, 'entry_atr': 2.0, 'qty': 10,
            'is_orderable': True, 'sizing_capital': equity,
        }

    def row_builder(**kwargs):
        events.append(('row', kwargs['ticker']))
        return {'ticker': kwargs['ticker'], 'is_orderable': kwargs['is_orderable']}

    stubs = {
        'get_fast_normal_signal_values': signal_values,
        'get_fast_security_profile': security_profile,
        'get_pit_stats_from_index': pit_stats,
        '_resolve_candidate_quality_ranking': quality_rank,
        'create_signal_tracking_state': create_state,
        'attach_breakout_quality_rank': attach,
        'build_normal_candidate_plan': normal_plan,
        '_resolve_candidate_sizing_fee_context': lambda *a, **kw: ('fee',),
        '_make_candidate_row': row_builder,
    }
    with patch.multiple(candidates, **stubs):
        common = dict(
            normal_setup_entries=entries, portfolio={'HELD': {}},
            sold_today={'SOLD'}, all_dfs_fast=all_fast,
            pit_stats_index=pit_index, pit_stats_cursor=cursor,
            today=today, params=params, profile_stats={},
        )
        initial = {ticker: {'source': 'old'} for ticker in all_fast}
        initial['PREVIOUS'] = {'source': 'keep'}
        full_active = {ticker: dict(value) for ticker, value in initial.items()}
        all_rows, orderable, setup_tickers = candidates._collect_normal_candidates(
            **{**common, 'active_extended_signals': full_active},
            buy_sort_method='ev', sizing_equity=10000.0,
            sizing_capital_context=('capital',), sizing_fee_context_cache={},
            collect_all_candidates=True, ranking_context=None,
        )
        full_events = events[:]
        events.clear()
        track_active = {ticker: dict(value) for ticker, value in initial.items()}
        tracked_tickers = candidates.track_normal_setup_signals_for_day(
            **{**common, 'active_extended_signals': track_active},
        )
        track_events = events[:]

        expected_tickers = {ticker for ticker, _, _ in entries}
        check('normal_setup_full_ticker_coverage', expected_tickers, setup_tickers)
        check('normal_setup_track_ticker_coverage', expected_tickers, tracked_tickers)
        check('normal_setup_full_only_orderable_good', ['GOOD'], [r['ticker'] for r in orderable])
        check('normal_setup_full_nonorderable_state_not_created', ['GOOD'], [r['ticker'] for r in all_rows])
        check('normal_setup_full_state_only_after_plan', ['GOOD'], sorted(k for k in full_active if k != 'PREVIOUS'))
        check('normal_setup_track_keeps_pit_qualified_unorderable', ['GOOD', 'NO_PLAN'],
              sorted(k for k in track_active if k != 'PREVIOUS'))
        check('normal_setup_prior_unrelated_state_full', {'source': 'keep'}, full_active['PREVIOUS'])
        check('normal_setup_prior_unrelated_state_track', {'source': 'keep'}, track_active['PREVIOUS'])
        check('normal_setup_full_and_track_good_state_same', full_active['GOOD'], track_active['GOOD'])
        check('normal_setup_full_and_track_source_rank', {'ticker': 'GOOD'}, full_active['GOOD']['quality'])
        check('normal_setup_full_pit_validity_before_plan',
              [('pit', 'GOOD'), ('plan', 'GOOD'), ('rank', 'GOOD'), ('state', 'GOOD')],
              [event for event in full_events if event[1] == 'GOOD' and event[0] in {'pit', 'plan', 'rank', 'state'}])
        check('normal_setup_track_pit_before_rank_state',
              [('pit', 'GOOD'), ('rank', 'GOOD'), ('state', 'GOOD')],
              [event for event in track_events if event[1] == 'GOOD' and event[0] in {'pit', 'rank', 'state'}])
        check('normal_setup_rejected_never_ranked_full', False, ('rank', 'PIT_REJECT') in full_events)
        check('normal_setup_rejected_never_ranked_track', False, ('rank', 'PIT_REJECT') in track_events)
        check('normal_setup_held_no_pit_full', False, ('pit', 'HELD') in full_events)
        check('normal_setup_sold_no_pit_track', False, ('pit', 'SOLD') in track_events)
        check('normal_setup_track_never_sizes_any_plan', False, any(event[0] == 'plan' for event in track_events))
        check('normal_setup_unorderable_plan_has_no_new_rank_full', False, ('rank', 'NO_PLAN') in full_events)
        check('normal_setup_unorderable_pit_allows_rank_track', True, ('rank', 'NO_PLAN') in track_events)
        check('normal_setup_pit_lookup_once_per_qualified_ticker_full', 1,
              sum(event == ('pit', 'GOOD') for event in full_events))
        check('normal_setup_pit_lookup_once_per_qualified_ticker_track', 1,
              sum(event == ('pit', 'GOOD') for event in track_events))
        check('normal_setup_full_pit_profile_event', True,
              'portfolio_candidate_pit_stats_sec' in common['profile_stats'])

        def one_qualified_setup(*, full, signals):
            args = {**common, 'normal_setup_entries': [('GOOD', 1, 2)],
                    'active_extended_signals': signals}
            if full:
                return candidates._collect_normal_candidates(
                    **args, buy_sort_method='ev', sizing_equity=10000.0,
                    sizing_capital_context=('capital',), sizing_fee_context_cache={},
                    collect_all_candidates=True, ranking_context=None,
                )
            return candidates.track_normal_setup_signals_for_day(**args)

        # A missing ranking provider is never silently treated as an accepted
        # signal, and an old extended setup must already have been superseded.
        with patch.object(candidates, '_resolve_candidate_quality_ranking',
                          side_effect=ValueError('PIT score unavailable')):
            for label, full in (('full', True), ('tracking', False)):
                signals = {'GOOD': {'source': 'stale'}, 'PREVIOUS': {'source': 'keep'}}
                caught = None
                try:
                    one_qualified_setup(full=full, signals=signals)
                except ValueError as exc:
                    caught = str(exc)
                check(f'normal_setup_{label}_ranking_provider_fail_closed',
                      'PIT score unavailable', caught)
                check(f'normal_setup_{label}_ranking_failure_removes_old_state',
                      {'PREVIOUS': {'source': 'keep'}}, signals)

        # A rejected or absent new tracking state must not resurrect a previous
        # continuation.  The full and full-capacity branches are equivalent.
        with patch.object(candidates, 'create_signal_tracking_state', return_value=None):
            for label, full in (('full', True), ('tracking', False)):
                signals = {'GOOD': {'source': 'stale'}}
                events.clear()
                one_qualified_setup(full=full, signals=signals)
                check(f'normal_setup_{label}_empty_state_clears_stale', {}, signals)
                check(f'normal_setup_{label}_empty_state_never_attaches_rank', False,
                      any(name == 'attach' for name, *_ in events))

        # Ranking-disabled mode cannot invoke a provider or retain old score
        # metadata simply because tracking is called instead of candidate build.
        params.use_breakout_quality_ranking = False
        with patch.object(candidates, '_resolve_candidate_quality_ranking',
                          side_effect=AssertionError('ranking must be disabled')):
            for label, full in (('full', True), ('tracking', False)):
                signals = {'GOOD': {'source': 'stale'}}
                one_qualified_setup(full=full, signals=signals)
                check(f'normal_setup_{label}_ranking_disabled_state', None,
                      signals['GOOD']['quality'])

    # No independent PIT/continuation branches can silently grow in either
    # caller: their only seam is the shared same-module helper.
    source = ast.parse(inspect.getsource(candidates))
    functions = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
    for name in ('_collect_normal_candidates', 'track_normal_setup_signals_for_day'):
        calls = [node.func.id for node in ast.walk(functions[name])
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        check(f'{name}_single_qualified_setup_owner', 1, calls.count('_resolve_qualified_normal_setup'))
        check(f'{name}_single_signal_state_owner', 1, calls.count('_record_normal_setup_tracking_state'))
        check(f'{name}_no_duplicate_pit_lookup', False, 'get_pit_stats_from_index' in calls)
        check(f'{name}_no_duplicate_tracking_state', False, 'create_signal_tracking_state' in calls)
