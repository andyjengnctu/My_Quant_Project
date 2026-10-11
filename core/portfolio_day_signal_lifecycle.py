"""Portfolio daily re-entry/continuation lifecycle routing (no trading formulas).

The engine determines the event chronology; these functions only dispatch to
existing single-member and ensemble signal producers.  The callbacks are
provided by the engine to retain the original test/compatibility patch seam.
"""


def activate_portfolio_reentry_for_day(
    *, today, day_context, portfolio, sold_today,
    active_reentry_watchlist, active_reentry_watchlists_by_member,
    active_extended_signals, active_extended_signals_by_member,
    is_training, replay_counts, replay_execution_rows,
    replay_selector_trace_rows, replay_phase,
    single_reentry_callback, ensemble_reentry_callback,
):
    """Activate PIT-valid re-entry before evaluating today's entry candidates."""
    if day_context.use_param_ensemble:
        return replay_phase(
            today,
            "activate_ensemble_reentry",
            ensemble_reentry_callback,
            ensemble_members=day_context.ensemble_members,
            ensemble_contexts=day_context.ensemble_contexts,
            active_reentry_watchlists_by_member=active_reentry_watchlists_by_member,
            active_extended_signals_by_member=active_extended_signals_by_member,
            portfolio=portfolio,
            sold_today=sold_today,
            today=today,
        )
    return replay_phase(
        today,
        "activate_reentry",
        single_reentry_callback,
        active_reentry_watchlist=active_reentry_watchlist,
        active_extended_signals=active_extended_signals,
        portfolio=portfolio,
        sold_today=sold_today,
        all_dfs_fast=day_context.all_dfs_fast,
        today=today,
        params=day_context.params,
        training_fast_path=bool(
            is_training
            and replay_counts is None
            and replay_execution_rows is None
            and replay_selector_trace_rows is None
        ),
    )


def track_portfolio_full_capacity_signals(
    *, today, day_context, normal_setup_entries_today,
    portfolio, sold_today, active_extended_signals,
    active_extended_signals_by_member, candidate_subprofile,
    replay_phase, single_track_callback, ensemble_track_callback,
) -> bool:
    """Track new setups even when no same-day capacity is available.

    Returns whether a tracking producer ran, for the engine's timing sidecar.
    Normal setup supersession and the PIT checks stay in the original producer.
    """
    if day_context.use_param_ensemble:
        has_ensemble_normal_setup = any(
            bool((ctx.get("normal_setup_index") or {}).get(today, []))
            for ctx in day_context.ensemble_contexts
        )
        if not has_ensemble_normal_setup:
            return False
        replay_phase(
            today,
            "track_ensemble_normal_setup",
            ensemble_track_callback,
            ensemble_members=day_context.ensemble_members,
            ensemble_contexts=day_context.ensemble_contexts,
            active_extended_signals_by_member=active_extended_signals_by_member,
            portfolio=portfolio,
            sold_today=sold_today,
            today=today,
            profile_stats=candidate_subprofile,
        )
        return True
    if not normal_setup_entries_today:
        return False
    replay_phase(
        today,
        "track_normal_setup",
        single_track_callback,
        normal_setup_entries=normal_setup_entries_today,
        portfolio=portfolio,
        sold_today=sold_today,
        all_dfs_fast=day_context.all_dfs_fast,
        active_extended_signals=active_extended_signals,
        pit_stats_index=day_context.pit_stats_index,
        pit_stats_cursor=day_context.pit_stats_cursor,
        today=today,
        params=day_context.params,
        profile_stats=candidate_subprofile,
    )
    return True


def cleanup_portfolio_extended_signals_for_day(
    *, today, day_context, portfolio,
    active_extended_signals, active_extended_signals_by_member,
    current_equity_money, initial_capital, sizing_equity,
    verbose, is_training, replay_counts, replay_execution_rows,
    replay_selector_trace_rows, extended_day_market_cache,
    cleanup_candidate_rows_by_ticker, replay_phase,
    single_cleanup_callback, ensemble_cleanup_callback,
):
    """Cleanup continuation signals only after the settlement/entry event."""
    if day_context.use_param_ensemble:
        return replay_phase(
            today,
            "cleanup_ensemble_extended_signals",
            ensemble_cleanup_callback,
            ensemble_members=day_context.ensemble_members,
            ensemble_contexts=day_context.ensemble_contexts,
            active_extended_signals_by_member=active_extended_signals_by_member,
            portfolio=portfolio,
            today=today,
            current_equity_money=current_equity_money,
            initial_capital=initial_capital,
            verbose=bool(verbose and (not is_training)),
        )
    return replay_phase(
        today,
        "cleanup_extended_signals",
        single_cleanup_callback,
        active_extended_signals=active_extended_signals,
        portfolio=portfolio,
        all_dfs_fast=day_context.all_dfs_fast,
        today=today,
        params=day_context.params,
        sizing_capital=sizing_equity,
        extended_day_market_cache=extended_day_market_cache,
        candidate_rows_by_ticker=cleanup_candidate_rows_by_ticker,
        training_fast_path=bool(
            is_training
            and replay_counts is None
            and replay_execution_rows is None
            and replay_selector_trace_rows is None
        ),
    )
