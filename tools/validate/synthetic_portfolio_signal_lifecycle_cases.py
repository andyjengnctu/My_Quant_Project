"""PIT-safe daily signal lifecycle contracts for the Portfolio orchestrator.

The producer callbacks remain canonical; tests check routing, identity and
mutation boundaries instead of copying Re-entry or continuation calculations.
"""

import ast
import inspect
from types import SimpleNamespace

from pandas import Timestamp

from .checks import bind_synthetic_case


def validate_portfolio_signal_lifecycle_owner_contract_case(_base_params):
    from core import portfolio_day_signal_lifecycle as signals, portfolio_engine

    results, summary, check, _ = bind_synthetic_case(
        "META_PORTFOLIO_SIGNAL_LIFECYCLE", "portfolio_signal_lifecycle"
    )
    today = Timestamp("2025-04-15")
    portfolio, sold_today = {}, set()
    watchlist, extended = {"2330": object()}, {"2317": object()}
    member_watchlists, member_extended = {"s1": watchlist}, {"s1": extended}
    fast_data, stats, pit_cursor = {"2330": object()}, {"2330": object()}, {}
    normal_setup = [("2330", 3, 4)]
    members = [{"seed": 1, "params_obj": _base_params}]
    contexts = [{"normal_setup_index": {today: normal_setup}}]
    base = SimpleNamespace(
        use_param_ensemble=False, ensemble_members=members,
        ensemble_contexts=contexts, all_dfs_fast=fast_data,
        pit_stats_index=stats, pit_stats_cursor=pit_cursor, params=_base_params,
    )
    consensus = SimpleNamespace(**{**vars(base), "use_param_ensemble": True})

    # The orchestrator's exact chronology is more important than module LOC.
    tree = ast.parse(inspect.getsource(portfolio_engine.run_portfolio_timeline))
    calls = [(node.lineno, node.func.id) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    ordered_phases = (
        "activate_portfolio_reentry_for_day", "resolve_portfolio_candidate_gate",
        "track_portfolio_full_capacity_signals", "build_portfolio_day_candidate_rows",
        "settle_and_execute_portfolio_day",
        "cleanup_portfolio_extended_signals_for_day", "mark_portfolio_day_equity",
    )
    indexes = [next((line for line, name in calls if name == phase), None)
               for phase in ordered_phases]
    check("reentry_tracking_settle_cleanup_mark_order", True,
          all(index is not None for index in indexes) and indexes == sorted(indexes))
    check("phases_have_single_engine_call", [1] * len(ordered_phases),
          [sum(name == phase for _, name in calls) for phase in ordered_phases])
    check("single_signal_lifecycle_owner_imports", True,
          all(getattr(portfolio_engine, n) is getattr(signals, n) for n in (
              "activate_portfolio_reentry_for_day",
              "track_portfolio_full_capacity_signals",
              "cleanup_portfolio_extended_signals_for_day",
          )))
    check("engine_does_not_directly_invoke_signal_producers", [], [
        name for _, name in calls if name in (
            "activate_breakout_reentry_signals_for_day",
            "_activate_ensemble_reentry_signals_for_day",
            "track_normal_setup_signals_for_day",
            "_track_ensemble_normal_setup_signals_for_day",
            "cleanup_extended_signals_for_day",
            "_cleanup_ensemble_extended_signals_for_day",
        )
    ])

    calls_made = []
    callbacks = {name: object() for name in (
        "single_reentry", "ensemble_reentry", "single_track", "ensemble_track",
        "single_cleanup", "ensemble_cleanup",
    )}

    def replay_phase(day, phase, callback, **kw):
        calls_made.append((day, phase, callback, kw))
        return "callback-result"

    reentry_args = dict(
        today=today, day_context=base, portfolio=portfolio,
        sold_today=sold_today, active_reentry_watchlist=watchlist,
        active_reentry_watchlists_by_member=member_watchlists,
        active_extended_signals=extended,
        active_extended_signals_by_member=member_extended,
        is_training=True, replay_counts=None, replay_execution_rows=None,
        replay_selector_trace_rows=None, replay_phase=replay_phase,
        single_reentry_callback=callbacks["single_reentry"],
        ensemble_reentry_callback=callbacks["ensemble_reentry"],
    )
    res = signals.activate_portfolio_reentry_for_day(**reentry_args)
    day, phase, callback, kw = calls_made[-1]
    check("single_reentry_callback_result", "callback-result", res)
    check("single_reentry_phase", (today, "activate_reentry", callbacks["single_reentry"]),
          (day, phase, callback))
    check("single_reentry_watch_reference", True, kw["active_reentry_watchlist"] is watchlist)
    check("single_reentry_extended_reference", True, kw["active_extended_signals"] is extended)
    check("single_reentry_market_reference", True, kw["all_dfs_fast"] is fast_data)
    check("single_reentry_param_reference", True, kw["params"] is _base_params)
    check("single_reentry_position_reference", True,
          kw["portfolio"] is portfolio and kw["sold_today"] is sold_today)
    check("single_reentry_training_fast_path", True, kw["training_fast_path"])
    check("single_reentry_does_not_pass_ensemble", False,
          "active_reentry_watchlists_by_member" in kw)
    signals.activate_portfolio_reentry_for_day(**{
        **reentry_args, "replay_counts": {},
    })
    check("replay_disables_reentry_fast_path", False, calls_made[-1][3]["training_fast_path"])
    signals.activate_portfolio_reentry_for_day(**{
        **reentry_args, "replay_execution_rows": [],
    })
    check("execution_capture_disables_reentry_fast_path", False, calls_made[-1][3]["training_fast_path"])
    signals.activate_portfolio_reentry_for_day(**{
        **reentry_args, "replay_selector_trace_rows": [],
    })
    check("selector_capture_disables_reentry_fast_path", False, calls_made[-1][3]["training_fast_path"])
    signals.activate_portfolio_reentry_for_day(**{
        **reentry_args, "is_training": False,
    })
    check("reporting_disables_reentry_fast_path", False, calls_made[-1][3]["training_fast_path"])

    signals.activate_portfolio_reentry_for_day(**{
        **reentry_args, "day_context": consensus,
    })
    day, phase, callback, kw = calls_made[-1]
    check("ensemble_reentry_phase", (today, "activate_ensemble_reentry", callbacks["ensemble_reentry"]),
          (day, phase, callback))
    check("ensemble_reentry_members_reference", True, kw["ensemble_members"] is members)
    check("ensemble_reentry_contexts_reference", True, kw["ensemble_contexts"] is contexts)
    check("ensemble_reentry_watch_reference", True,
          kw["active_reentry_watchlists_by_member"] is member_watchlists)
    check("ensemble_reentry_extended_reference", True,
          kw["active_extended_signals_by_member"] is member_extended)
    check("ensemble_reentry_no_single_watch", False, "active_reentry_watchlist" in kw)
    check("ensemble_reentry_no_training_fast_path_override", False, "training_fast_path" in kw)

    track_args = dict(
        today=today, day_context=base,
        normal_setup_entries_today=normal_setup,
        portfolio=portfolio, sold_today=sold_today,
        active_extended_signals=extended,
        active_extended_signals_by_member=member_extended,
        candidate_subprofile={"_timing_enabled": True},
        replay_phase=replay_phase,
        single_track_callback=callbacks["single_track"],
        ensemble_track_callback=callbacks["ensemble_track"],
    )
    check("tracking_runs_single_setup", True,
          signals.track_portfolio_full_capacity_signals(**track_args))
    day, phase, callback, kw = calls_made[-1]
    check("single_tracking_phase", (today, "track_normal_setup", callbacks["single_track"]),
          (day, phase, callback))
    check("single_tracking_uses_same_setup_rows", True,
          kw["normal_setup_entries"] is normal_setup)
    check("single_tracking_uses_exact_pit_index", True, kw["pit_stats_index"] is stats)
    check("single_tracking_uses_exact_pit_cursor", True, kw["pit_stats_cursor"] is pit_cursor)
    check("single_tracking_uses_extended_state", True, kw["active_extended_signals"] is extended)
    check("single_tracking_uses_param_reference", True, kw["params"] is _base_params)
    check("single_tracking_uses_profile_reference", True,
          kw["profile_stats"] is track_args["candidate_subprofile"])
    last_count = len(calls_made)
    check("single_empty_setup_skips_callback", False,
          signals.track_portfolio_full_capacity_signals(**{
              **track_args, "normal_setup_entries_today": []
          }))
    check("single_empty_setup_no_side_effect", last_count, len(calls_made))
    check("ensemble_tracking_runs_context_setup", True,
          signals.track_portfolio_full_capacity_signals(**{
              **track_args, "day_context": consensus,
              "normal_setup_entries_today": [],
          }))
    day, phase, callback, kw = calls_made[-1]
    check("ensemble_tracking_phase", (today, "track_ensemble_normal_setup", callbacks["ensemble_track"]),
          (day, phase, callback))
    check("ensemble_tracking_member_and_context_refs", True,
          kw["ensemble_members"] is members and kw["ensemble_contexts"] is contexts)
    check("ensemble_tracking_extended_and_profile_refs", True,
          kw["active_extended_signals_by_member"] is member_extended
          and kw["profile_stats"] is track_args["candidate_subprofile"])
    empty_consensus = SimpleNamespace(**{**vars(consensus),
                                         "ensemble_contexts": [{"normal_setup_index": {}}]})
    last_count = len(calls_made)
    check("ensemble_empty_context_skips_callback", False,
          signals.track_portfolio_full_capacity_signals(**{
              **track_args, "day_context": empty_consensus,
          }))
    check("ensemble_empty_context_no_side_effect", last_count, len(calls_made))

    candidate_rows = {"2330": {"ticker": "2330"}}
    market_cache = {"a": object()}
    cleanup_args = dict(
        today=today, day_context=base, portfolio=portfolio,
        active_extended_signals=extended,
        active_extended_signals_by_member=member_extended,
        current_equity_money=120.0, initial_capital=100.0,
        sizing_equity=110.0, verbose=True,
        is_training=True, replay_counts=None, replay_execution_rows=None,
        replay_selector_trace_rows=None,
        extended_day_market_cache=market_cache,
        cleanup_candidate_rows_by_ticker=candidate_rows,
        replay_phase=replay_phase,
        single_cleanup_callback=callbacks["single_cleanup"],
        ensemble_cleanup_callback=callbacks["ensemble_cleanup"],
    )
    result = signals.cleanup_portfolio_extended_signals_for_day(**cleanup_args)
    day, phase, callback, kw = calls_made[-1]
    check("cleanup_callback_result", "callback-result", result)
    check("single_cleanup_phase", (today, "cleanup_extended_signals", callbacks["single_cleanup"]),
          (day, phase, callback))
    check("single_cleanup_uses_current_sizing_equity", 110.0, kw["sizing_capital"])
    check("single_cleanup_uses_original_market_cache", True,
          kw["extended_day_market_cache"] is market_cache)
    check("single_cleanup_uses_candidate_snapshot_index", True,
          kw["candidate_rows_by_ticker"] is candidate_rows)
    check("single_cleanup_uses_active_signals", True,
          kw["active_extended_signals"] is extended)
    check("single_cleanup_fast_path_training", True, kw["training_fast_path"])
    signals.cleanup_portfolio_extended_signals_for_day(**{
        **cleanup_args, "replay_counts": {},
    })
    check("cleanup_replay_disables_fast_path", False, calls_made[-1][3]["training_fast_path"])
    signals.cleanup_portfolio_extended_signals_for_day(**{
        **cleanup_args, "replay_execution_rows": [],
    })
    check("cleanup_execution_capture_disables_fast_path", False, calls_made[-1][3]["training_fast_path"])
    signals.cleanup_portfolio_extended_signals_for_day(**{
        **cleanup_args, "replay_selector_trace_rows": [],
    })
    check("cleanup_selector_capture_disables_fast_path", False, calls_made[-1][3]["training_fast_path"])
    signals.cleanup_portfolio_extended_signals_for_day(**{
        **cleanup_args, "is_training": False,
    })
    check("cleanup_reporting_disables_fast_path", False, calls_made[-1][3]["training_fast_path"])
    signals.cleanup_portfolio_extended_signals_for_day(**{
        **cleanup_args, "day_context": consensus,
    })
    day, phase, callback, kw = calls_made[-1]
    check("ensemble_cleanup_phase", (today, "cleanup_ensemble_extended_signals", callbacks["ensemble_cleanup"]),
          (day, phase, callback))
    check("ensemble_cleanup_extended_reference", True,
          kw["active_extended_signals_by_member"] is member_extended)
    check("ensemble_cleanup_context_references", True,
          kw["ensemble_members"] is members and kw["ensemble_contexts"] is contexts)
    check("ensemble_cleanup_keeps_exact_equity_inputs", (120.0, 100.0),
          (kw["current_equity_money"], kw["initial_capital"]))
    check("ensemble_training_verbosity_suppressed", False, kw["verbose"])
    signals.cleanup_portfolio_extended_signals_for_day(**{
        **cleanup_args, "day_context": consensus, "is_training": False,
    })
    check("ensemble_reporting_verbose_forwarded", True, calls_made[-1][3]["verbose"])
    check("ensemble_cleanup_does_not_resolve_single_sizing", False,
          "sizing_capital" in calls_made[-1][3])

    def raising_phase(day, phase, callback, **kwargs):
        raise ValueError(f"PIT routing failed: {phase}")
    try:
        signals.activate_portfolio_reentry_for_day(**{
            **reentry_args, "replay_phase": raising_phase,
        })
    except ValueError as exc:
        check("replay_phase_error_propagates", "PIT routing failed: activate_reentry", str(exc))
    else:
        check("replay_phase_error_propagates", True, False)

    summary["contract_assertions"] = len(results)
    return results, summary
