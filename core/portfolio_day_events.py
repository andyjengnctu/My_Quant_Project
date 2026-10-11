"""Daily Portfolio event phases; the Portfolio Engine owns their ordering.

All callbacks remain the canonical exit/entry implementations. The pre-market
``available_cash`` is intentionally passed unchanged to the entry phase: proceeds
from intraday exits must not fund a discretionary replacement order.
"""

import time
from typing import NamedTuple

from core.portfolio_ops import (
    execute_reserved_entries_for_day,
    settle_portfolio_positions,
    try_rotate_weakest_position,
)
from core.portfolio_replay_support import _run_portfolio_replay_phase


class PortfolioRotationResult(NamedTuple):
    cash: int
    normal_trade_count: int
    extended_trade_count: int
    elapsed_sec: float


class PortfolioSettleEntryResult(NamedTuple):
    cash: int
    total_missed_sells: int
    total_missed_buys: int
    normal_trade_count: int
    extended_trade_count: int
    settle_sec: float
    buy_sec: float


def rotate_portfolio_day(
    *, today, day_context, portfolio, resource_action_candidates_today,
    max_positions, enable_rotation, sold_today, cash, closed_trades_stats,
    trade_history, is_training, normal_trade_count, extended_trade_count,
    active_level_rows, fee_rebate_state, profile_timing_enabled,
) -> PortfolioRotationResult:
    """Rotate only after the pre-market resource selection has completed."""
    t0 = time.perf_counter() if profile_timing_enabled else None
    cash, normal_trade_count, extended_trade_count = _run_portfolio_replay_phase(
        today,
        "rotate_weakest_position",
        try_rotate_weakest_position,
        portfolio=portfolio,
        orderable_candidates_today=resource_action_candidates_today,
        max_positions=max_positions,
        enable_rotation=enable_rotation,
        sold_today=sold_today,
        all_dfs_fast=day_context.all_dfs_fast,
        today=today,
        pit_stats_index=day_context.pit_stats_index,
        params=day_context.params,
        cash=cash,
        closed_trades_stats=closed_trades_stats,
        trade_history=trade_history,
        is_training=is_training,
        normal_trade_count=normal_trade_count,
        extended_trade_count=extended_trade_count,
        active_level_rows=active_level_rows,
        fee_rebate_state=fee_rebate_state,
    )
    elapsed = time.perf_counter() - t0 if profile_timing_enabled else 0.0
    return PortfolioRotationResult(cash, normal_trade_count, extended_trade_count, elapsed)


def settle_and_execute_portfolio_day(
    *, today, day_context, portfolio, sold_today, cash, closed_trades_stats,
    trade_history, is_training, total_missed_sells, total_missed_buys,
    normal_trade_count, extended_trade_count, active_level_rows,
    active_reentry_watchlist, active_reentry_watchlists_by_member,
    fee_rebate_state, no_entry_capacity_today, resource_action_candidates_today,
    max_positions, active_extended_signals, available_cash, sizing_equity,
    portfolio_entry_stats, replay_execution_rows, extended_day_market_cache,
    profile_timing_enabled,
) -> PortfolioSettleEntryResult:
    """Exit first; then fill only the *already reserved* pre-market orders.

    The entry gate sees the final portfolio/sold_today state but uses the
    pre-market available_cash. No day-trading or intraday reallocations.
    """
    t0 = time.perf_counter() if profile_timing_enabled else None
    cash, total_missed_sells, normal_trade_count, extended_trade_count = _run_portfolio_replay_phase(
        today,
        "settle_positions",
        settle_portfolio_positions,
        portfolio=portfolio,
        sold_today=sold_today,
        all_dfs_fast=day_context.all_dfs_fast,
        today=today,
        params=day_context.params,
        cash=cash,
        closed_trades_stats=closed_trades_stats,
        trade_history=trade_history,
        is_training=is_training,
        total_missed_sells=total_missed_sells,
        normal_trade_count=normal_trade_count,
        extended_trade_count=extended_trade_count,
        active_level_rows=active_level_rows,
        active_reentry_watchlist=(
            active_reentry_watchlist if not day_context.use_param_ensemble else None
        ),
        active_reentry_watchlists_by_member=(
            active_reentry_watchlists_by_member if day_context.use_param_ensemble else None
        ),
        fee_rebate_state=fee_rebate_state,
    )
    settle_sec = time.perf_counter() - t0 if profile_timing_enabled else 0.0

    # Settled exits may free position slots, but the day's eligible buy basket
    # and its spending cap are frozen before the market opens.
    can_try_entries_today = (
        (not no_entry_capacity_today)
        and bool(resource_action_candidates_today)
        and (len(portfolio) + len(sold_today)) < int(max_positions)
    )
    buy_sec = 0.0
    if can_try_entries_today:
        t0 = time.perf_counter() if profile_timing_enabled else None
        cash, total_missed_buys = _run_portfolio_replay_phase(
            today,
            "execute_reserved_entries",
            execute_reserved_entries_for_day,
            portfolio=portfolio,
            active_extended_signals=active_extended_signals,
            orderable_candidates_today=resource_action_candidates_today,
            sold_today=sold_today,
            all_dfs_fast=day_context.all_dfs_fast,
            today=today,
            params=day_context.params,
            cash=cash,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            max_positions=max_positions,
            trade_history=trade_history,
            is_training=is_training,
            total_missed_buys=total_missed_buys,
            entry_stats=portfolio_entry_stats,
            replay_execution_rows=replay_execution_rows,
            fee_rebate_state=fee_rebate_state,
            extended_day_market_cache=extended_day_market_cache,
        )
        buy_sec = time.perf_counter() - t0 if profile_timing_enabled else 0.0

    return PortfolioSettleEntryResult(
        cash, total_missed_sells, total_missed_buys,
        normal_trade_count, extended_trade_count, settle_sec, buy_sec,
    )
