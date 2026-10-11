"""Canonical Portfolio daily candidate gate, construction, and resource ordering.

Keeps single- and multi-member candidate eligibility in one place.  The
engine still owns the event timeline and supplies the existing producer and
selection callbacks; no scoring or trade formulas are implemented here.
"""

from typing import NamedTuple


class PortfolioCandidateGate(NamedTuple):
    pre_market_occupied: int
    has_candidates: bool
    no_entry_capacity: bool


class PortfolioCandidateSelection(NamedTuple):
    orderable_rows: list
    action_rows: list
    diagnostic: dict
    trace_input_rows: list | None


def resolve_portfolio_candidate_gate(
    *, today, day_context, portfolio, sold_today, normal_setup_entries_today,
    active_extended_signals, active_extended_signals_by_member,
    is_training, replay_counts, enable_rotation, max_positions,
) -> PortfolioCandidateGate:
    """Gate after reentry activation, before sell/fill phases or price updates."""
    occupied = len(portfolio) + len(sold_today)
    if day_context.use_param_ensemble:
        has_candidates = (
            any(bool((ctx.get("normal_setup_index") or {}).get(today, []))
                for ctx in day_context.ensemble_contexts)
            or any(bool(signals) for signals in active_extended_signals_by_member.values())
        )
    else:
        has_candidates = bool(normal_setup_entries_today) or bool(active_extended_signals)
    no_capacity = (
        bool(is_training)
        and replay_counts is None
        and (not bool(enable_rotation))
        and occupied >= int(max_positions)
    )
    return PortfolioCandidateGate(occupied, has_candidates, no_capacity)


def build_portfolio_day_candidate_rows(
    *, today, day_context, portfolio, sold_today,
    active_extended_signals, active_extended_signals_by_member,
    current_equity_money, initial_capital, sizing_equity,
    replay_counts, replay_execution_rows, replay_selector_trace_rows,
    enable_rotation, is_training, verbose, extended_day_market_cache,
    persistent_sizing_fee_context_cache, persistent_sizing_qty_cache,
    candidate_subprofile, replay_phase, single_candidate_builder,
    ensemble_candidate_builder,
):
    """Delegate to original candidate producers without replacing PIT semantics."""
    if day_context.use_param_ensemble:
        return replay_phase(
            today,
            "build_daily_ensemble_candidates",
            ensemble_candidate_builder,
            ensemble_members=day_context.ensemble_members,
            ensemble_contexts=day_context.ensemble_contexts,
            active_extended_signals_by_member=active_extended_signals_by_member,
            portfolio=portfolio,
            sold_today=sold_today,
            today=today,
            current_equity_money=current_equity_money,
            initial_capital=initial_capital,
            collect_all_candidates=replay_counts is not None,
            min_agree=int(day_context.ensemble_min_agree or 1),
            verbose=bool(verbose and (not is_training)),
            profile_stats=candidate_subprofile,
        )
    return replay_phase(
        today,
        "build_daily_candidates",
        single_candidate_builder,
        normal_setup_index=day_context.normal_setup_index,
        active_extended_signals=active_extended_signals,
        portfolio=portfolio,
        sold_today=sold_today,
        all_dfs_fast=day_context.all_dfs_fast,
        pit_stats_index=day_context.pit_stats_index,
        pit_stats_cursor=day_context.pit_stats_cursor,
        today=today,
        sizing_equity=sizing_equity,
        params=day_context.params,
        collect_all_candidates=replay_counts is not None,
        extended_day_market_cache=extended_day_market_cache,
        compact_training_rows=bool(
            is_training
            and replay_counts is None
            and replay_execution_rows is None
            and replay_selector_trace_rows is None
            and not bool(enable_rotation)
            and not bool(getattr(day_context.params, "use_breakout_quality_ranking", False))
        ),
        persistent_sizing_fee_context_cache=persistent_sizing_fee_context_cache,
        persistent_sizing_qty_cache=persistent_sizing_qty_cache,
        profile_stats=candidate_subprofile,
    )


def resolve_cleanup_candidate_rows(orderable_rows, *, is_ensemble, extended_day_market_cache):
    """Reuse the producer's row index when available; never change row order."""
    if is_ensemble or not orderable_rows:
        return None
    by_ticker = (
        None if extended_day_market_cache is None
        else extended_day_market_cache.get(("__meta__", "orderable_candidate_rows_by_ticker"))
    )
    if by_ticker is None:
        by_ticker = {
            row.get("ticker"): row for row in orderable_rows
            if row.get("ticker") is not None
        }
    return by_ticker


def select_portfolio_candidate_orders(
    *, today, orderable_rows, replay_selector_trace_rows,
    replay_execution_rows, replay_counts, is_training,
    params, available_cash, sizing_equity, pre_market_occupied,
    max_positions, replay_phase, reorder_callback, select_action_callback,
) -> PortfolioCandidateSelection:
    """Use the existing resource selector, preserving its training no-op path."""
    trace_input = (
        list(orderable_rows) if replay_selector_trace_rows is not None else None
    )
    skip_noop = bool(
        is_training
        and replay_counts is None
        and replay_execution_rows is None
        and replay_selector_trace_rows is None
        and not bool(getattr(params, "use_breakout_quality_ranking", False))
    )
    diagnostic = {}
    if skip_noop:
        # Training without DL/ranking: identity selector; keep the same list.
        action_rows = orderable_rows
    else:
        orderable_rows, diagnostic = replay_phase(
            today,
            "resource_aware_binary_order",
            reorder_callback,
            orderable_rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            pre_market_occupied=pre_market_occupied,
            max_positions=max_positions,
            params=params,
        )
        action_rows = select_action_callback(orderable_rows, diagnostic)
    return PortfolioCandidateSelection(orderable_rows, action_rows, diagnostic, trace_input)
