"""Behavioral guards for the daily event sequence and exact equity mark."""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import patch

from pandas import Timestamp

from .checks import add_check, bind_synthetic_case


def validate_portfolio_day_event_equity_owner_contract_case(_base_params):
    from core import portfolio_day_events, portfolio_day_equity, portfolio_engine

    case_id = "META_PORTFOLIO_EVENT_EQUITY_OWNER"
    results, summary, check, _ = bind_synthetic_case(case_id, "portfolio_event_equity")
    today = Timestamp("2025-01-15")
    context = SimpleNamespace(
        all_dfs_fast={"sample": object()},
        pit_stats_index={"sample": object()},
        params=_base_params,
        use_param_ensemble=False,
    )
    portfolio, sold_today = {}, set()
    rows, closed, active_levels = [], [], []
    fee_rebate = {}
    stages, snapshots = [], []
    pre_market_cash = 950

    # A later intraday sale may increase cash, but the reserved buying budget
    # must remain exactly the pre-market value.
    def replay_stub(day, phase, callback, **kwargs):
        stages.append(phase)
        snapshots.append((phase, dict(kwargs), callback))
        if phase == "rotate_weakest_position":
            return kwargs["cash"] + 20, kwargs["normal_trade_count"] + 1, kwargs["extended_trade_count"] + 2
        if phase == "settle_positions":
            return (kwargs["cash"] + 200, kwargs["total_missed_sells"] + 1,
                    kwargs["normal_trade_count"] + 1, kwargs["extended_trade_count"] + 1)
        if phase == "execute_reserved_entries":
            return kwargs["cash"] - 50, kwargs["total_missed_buys"] + 2
        raise AssertionError(f"Unexpected portfolio phase: {phase}")

    rotate_args = dict(
        today=today, day_context=context, portfolio=portfolio,
        resource_action_candidates_today=[{"ticker": "2330"}],
        max_positions=2, enable_rotation=True, sold_today=sold_today,
        cash=1000, closed_trades_stats=closed, trade_history=rows,
        is_training=False, normal_trade_count=0, extended_trade_count=0,
        active_level_rows=active_levels, fee_rebate_state=fee_rebate,
        profile_timing_enabled=False,
    )
    settle_args = dict(
        today=today, day_context=context, portfolio=portfolio,
        sold_today=sold_today, cash=1020, closed_trades_stats=closed,
        trade_history=rows, is_training=False, total_missed_sells=3,
        total_missed_buys=4, normal_trade_count=1, extended_trade_count=2,
        active_level_rows=active_levels, active_reentry_watchlist={},
        active_reentry_watchlists_by_member={}, fee_rebate_state=fee_rebate,
        no_entry_capacity_today=False,
        resource_action_candidates_today=[{"ticker": "2330"}],
        max_positions=2, active_extended_signals={},
        available_cash=pre_market_cash, sizing_equity=1000,
        portfolio_entry_stats={}, replay_execution_rows=[],
        extended_day_market_cache={}, profile_timing_enabled=False,
    )
    # A future orchestrator edit must not invert the pre-market rotation,
    # settlement, entry, mark-to-market, and exposure chronology.
    engine_tree = ast.parse(inspect.getsource(portfolio_engine.run_portfolio_timeline))
    engine_calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(engine_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    expected_sequence = (
        "rotate_portfolio_day", "settle_and_execute_portfolio_day",
        "mark_portfolio_day_equity", "measure_portfolio_day_exposure",
    )
    phase_positions = [
        next((lineno for lineno, name in engine_calls if name == phase), None)
        for phase in expected_sequence
    ]
    check("engine_daily_phase_chronology_is_frozen", True,
          all(item is not None for item in phase_positions)
          and phase_positions == sorted(phase_positions))
    check("engine_does_not_duplicate_canonical_event_calls", [], [
        name for _, name in engine_calls if name in (
            "try_rotate_weakest_position", "settle_portfolio_positions",
            "execute_reserved_entries_for_day",
        )
    ])
    check("canonical_event_owner_import", True,
          portfolio_engine.rotate_portfolio_day is portfolio_day_events.rotate_portfolio_day
          and portfolio_engine.settle_and_execute_portfolio_day is portfolio_day_events.settle_and_execute_portfolio_day)
    check("canonical_equity_owner_import", True,
          portfolio_engine.mark_portfolio_day_equity is portfolio_day_equity.mark_portfolio_day_equity
          and portfolio_engine.measure_portfolio_day_exposure is portfolio_day_equity.measure_portfolio_day_exposure)

    with patch.object(portfolio_day_events, "_run_portfolio_replay_phase", side_effect=replay_stub):
        rotated = portfolio_day_events.rotate_portfolio_day(**rotate_args)
        completed = portfolio_day_events.settle_and_execute_portfolio_day(**settle_args)
        check("event_sequence_order", ["rotate_weakest_position", "settle_positions", "execute_reserved_entries"], stages)
        check("event_mutations_are_forwarded", (1020, 1, 2), tuple(rotated[:3]))
        check("exit_then_entry_balances", (1170, 4, 6, 2, 3), tuple(completed[:5]))
        check("pre_market_buying_budget_never_uses_exit_proceeds", (1220, pre_market_cash),
              (snapshots[-1][1]["cash"], snapshots[-1][1]["available_cash"]))
        check("canon_exit_and_entry_callbacks", True,
              snapshots[0][2] is portfolio_day_events.try_rotate_weakest_position
              and snapshots[1][2] is portfolio_day_events.settle_portfolio_positions
              and snapshots[2][2] is portfolio_day_events.execute_reserved_entries_for_day)
        check("event_clock_off_is_zero", (0.0, 0.0, 0.0),
              (rotated.elapsed_sec, completed.settle_sec, completed.buy_sec))
        check("one_member_reentry_routing", (settle_args["active_reentry_watchlist"], None),
              (snapshots[1][1]["active_reentry_watchlist"],
               snapshots[1][1]["active_reentry_watchlists_by_member"]))

        stages.clear()
        portfolio_day_events.settle_and_execute_portfolio_day(**{**settle_args, "no_entry_capacity_today": True})
        check("capacity_full_does_not_allow_buy", ["settle_positions"], stages)
        stages.clear()
        portfolio_day_events.settle_and_execute_portfolio_day(**{
            **settle_args, "resource_action_candidates_today": [],
        })
        check("no_premarket_order_does_not_allow_buy", ["settle_positions"], stages)
        stages.clear()
        portfolio_day_events.settle_and_execute_portfolio_day(**{
            **settle_args, "sold_today": {"a", "b"},
        })
        check("same_day_sales_cannot_free_same_day_buy_slots", ["settle_positions"], stages)
        stages.clear()
        ensemble_watchlists = {"seed0": {"item": "watched"}}
        portfolio_day_events.settle_and_execute_portfolio_day(**{
            **settle_args,
            "day_context": SimpleNamespace(**{**vars(context), "use_param_ensemble": True}),
            "active_reentry_watchlists_by_member": ensemble_watchlists,
            "no_entry_capacity_today": True,
        })
        check("multi_seed_reentry_routes_only_ensemble", (None, ensemble_watchlists),
              (snapshots[-1][1]["active_reentry_watchlist"],
               snapshots[-1][1]["active_reentry_watchlists_by_member"]))

    # Settlement errors must retain canonical replay phase/date context.
    with patch.object(portfolio_day_events, "settle_portfolio_positions", side_effect=ValueError("broken-sell")):
        try:
            portfolio_day_events.settle_and_execute_portfolio_day(**settle_args)
            error = "NOT_RAISED"
        except RuntimeError as exc:
            error = str(exc)
    check("settlement_errors_keep_phase_context", True,
          "date=2025-01-15" in error and "phase=settle_positions" in error
          and "broken-sell" in error)

    # A fee-rebate receivable increases equity but never the spendable cash.
    with patch.object(portfolio_day_equity, "get_fee_rebate_receivable_milli", return_value=2500):
        cash_only = portfolio_day_equity.mark_portfolio_day_equity(
            cash=100000, portfolio={}, all_dfs_fast={}, today=today,
            params=_base_params, fee_rebate_state=fee_rebate,
        )
        check("cash_only_mark_includes_receivable", (102500, 102.5, 100.0), tuple(cash_only))
        with patch.object(portfolio_day_equity, "calc_mark_to_market_equity", return_value=130000) as mtm:
            held = portfolio_day_equity.mark_portfolio_day_equity(
                cash=100000, portfolio={"held": object()}, all_dfs_fast={},
                today=today, params=_base_params, fee_rebate_state=fee_rebate,
            )
            check("held_mark_uses_exact_accounting_owner", (130000, 130.0, 100.0), tuple(held))
            check("held_mark_preserves_receivable_and_cached_px", (2500, True),
                  (mtm.call_args.kwargs["fee_rebate_receivable_milli"],
                   mtm.call_args.kwargs["use_cached_last_px"]))
        measured = portfolio_day_equity.measure_portfolio_day_exposure(
            current_equity_money=130.0, cash_money=100.0, initial_capital=100.0,
            fee_rebate_state=fee_rebate,
        )
        check("mark_exposure_excludes_receivable_from_invested_amount", 27.5, measured.invested_capital)
        check("mark_exposure_uses_same_equity_denominator", 27.5 / 130.0 * 100.0, measured.exposure_pct)
        check("strategy_return_uses_total_equity", 30.0, measured.strategy_ret_pct)
    summary["contract_assertions"] = len(results)
    return results, summary
