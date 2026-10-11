import pandas as pd
import time

from core.exact_accounting import milli_to_money, money_to_milli
from core.fee_rebate import (
    create_fee_rebate_state,
    get_fee_rebate_receivable_milli,
    get_fee_rebate_settled_milli,
    settle_fee_rebate,
)
from core.capital_policy import resolve_portfolio_sizing_equity
from core.seed_ensemble_policy import resolve_seed_ensemble_min_agree
from core.portfolio_day_context import resolve_portfolio_day_context, portfolio_has_day_work
from core.portfolio_day_signal_lifecycle import (
    activate_portfolio_reentry_for_day,
    track_portfolio_full_capacity_signals,
    cleanup_portfolio_extended_signals_for_day,
)
from core.portfolio_day_candidates import (
    resolve_portfolio_candidate_gate,
    build_portfolio_day_candidate_rows,
    resolve_cleanup_candidate_rows,
    select_portfolio_candidate_orders,
)
from core.portfolio_replay_diagnostics import (
    append_portfolio_selector_trace,
    record_portfolio_candidate_replay,
    initialize_portfolio_replay_counts,
    append_portfolio_trade_replay,
)
from core.portfolio_capacity_rows import build_portfolio_capacity_day_row
from core.portfolio_day_events import rotate_portfolio_day, settle_and_execute_portfolio_day
from core.portfolio_day_equity import mark_portfolio_day_equity, measure_portfolio_day_exposure
from core.portfolio_terminal_equity import reconcile_portfolio_terminal_equity
from core.portfolio_terminal_statistics import (
    build_portfolio_period_reports, summarize_portfolio_terminal_metrics,
)
from core.portfolio_terminal_reporting import (
    assemble_portfolio_timeline_result, publish_portfolio_profile_stats,
)
from core.breakout_reentry import activate_breakout_reentry_signals_for_day
from core.config import get_ev_calc_method
from core.portfolio_fast_data import (
    build_score_single_stock_profile_fields,
    build_normal_setup_index,
    build_trade_stats_index,
    calc_mark_to_market_equity,
    get_fast_close,
    get_fast_close_on_or_before,
    has_fast_date,
    summarize_single_stock_trade_stats_from_pit_index,
)
from core.portfolio_stats import (
    build_benchmark_full_month_return_stats,
    build_benchmark_full_quarter_return_stats,
    build_benchmark_full_year_return_stats,
    build_full_month_return_stats,
    build_full_quarter_return_stats,
    build_full_year_return_stats,
    calc_annual_return_pct,
    calc_curve_stats,
    calc_sim_years,
    find_sim_start_idx,
    summarize_closed_trade_entry_type_counts,
    summarize_closed_trade_r_stats,
)
from core.portfolio_attribution import build_dominant_year_dependency_diagnostics
from core.portfolio_candidates import build_daily_candidates, track_normal_setup_signals_for_day
from core.portfolio_ops import (
    cleanup_extended_signals_for_day,
    closeout_open_positions,
    execute_reserved_entries_for_day,
    reorder_candidates_for_resource_aware_quality,
    select_resource_aware_action_candidates,
    settle_portfolio_positions,
    try_rotate_weakest_position,
)

# Compatibility re-exports: canonical implementations live in focused portfolio modules.
from core.portfolio_benchmark import (
    BENCHMARK_PERIOD_STATS_CACHE_MAX_ITEMS,
    _BENCHMARK_PERIOD_STATS_CACHE,
    _BENCHMARK_PERIOD_STATS_CACHE_LOCK,
    _build_benchmark_period_stats,
    _get_benchmark_period_stats,
    _make_benchmark_period_cache_key,
)
from core.portfolio_ensemble import (
    _activate_ensemble_reentry_signals_for_day,
    _aggregate_ensemble_candidate_rows,
    _annotate_ensemble_candidate,
    _build_daily_ensemble_candidates,
    _cleanup_ensemble_extended_signals_for_day,
    _flatten_ensemble_extended_signals,
    _iter_ensemble_member_pairs,
    _median_candidate_row,
    _resolve_ensemble_member_key,
    _track_ensemble_normal_setup_signals_for_day,
)
from core.portfolio_levels import (
    _append_portfolio_active_level_rows,
    _append_portfolio_extended_shadow_level_rows,
    _is_portfolio_shadow_level_visible_day,
)
from core.portfolio_replay_support import (
    _candidate_execution_replay_snapshot,
    _candidate_replay_snapshot,
    _format_replay_date,
    _format_replay_elapsed,
    _print_slow_ensemble_phase_heartbeat,
    _run_portfolio_replay_phase,
)
from core.portfolio_entry_selection_max_dl import (
    build_max_dl_repair_mechanism_diagnostic,
)


def run_portfolio_timeline(
    all_dfs_fast,
    all_standalone_logs,
    sorted_dates,
    start_year,
    params,
    max_positions,
    enable_rotation,
    benchmark_ticker="0050",
    benchmark_data=None,
    is_training=True,
    profile_stats=None,
    verbose=True,
    replay_counts=None,
    replay_execution_rows=None,
    pit_stats_index=None,
    active_params_resolver=None,
    active_context_resolver=None,
    active_param_ensemble_resolver=None,
    active_context_ensemble_resolver=None,
    ensemble_min_agree=None,
    replay_selector_trace_rows=None,
):
    profile_timing_enabled = bool(profile_stats.get("_timing_enabled", True)) if profile_stats is not None else False
    capture_equity_curve = bool(profile_stats.get("capture_equity_curve", False)) if profile_stats is not None else False
    t_portfolio_start = time.perf_counter() if profile_timing_enabled else None
    candidate_scan_sec = 0.0
    candidate_subprofile = (
        {}
        if profile_timing_enabled
        and bool(profile_stats.get("_candidate_timing_enabled", False))
        else None
    )
    day_loop_sec = 0.0
    rotation_sec = 0.0
    settle_sec = 0.0
    buy_sec = 0.0
    equity_mark_sec = 0.0
    day_setup_sec = 0.0
    reentry_sec = 0.0
    resource_select_sec = 0.0
    cleanup_sec = 0.0
    bookkeeping_sec = 0.0
    build_trade_index_sec = 0.0
    ticker_dates_sec = 0.0
    closeout_sec = 0.0
    idle_fast_path_days = 0

    t0 = time.perf_counter() if profile_timing_enabled else None
    start_idx = find_sim_start_idx(sorted_dates, start_year)
    benchmark_period_stats = None
    use_benchmark_period_cache = bool(is_training and (not capture_equity_curve) and replay_counts is None and benchmark_data is not None and start_idx < len(sorted_dates))
    if use_benchmark_period_cache:
        benchmark_period_stats = _get_benchmark_period_stats(
            benchmark_data=benchmark_data,
            sorted_dates=sorted_dates,
            start_idx=start_idx,
        )
    if profile_timing_enabled:
        ticker_dates_sec = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_timing_enabled else None
    if active_context_resolver is None:
        if pit_stats_index is None:
            pit_stats_index = {t: build_trade_stats_index(logs) for t, logs in all_standalone_logs.items()}
        else:
            pit_stats_index = dict(pit_stats_index)
            if all_standalone_logs:
                for ticker, logs in all_standalone_logs.items():
                    if pit_stats_index.get(ticker) is None:
                        pit_stats_index[ticker] = build_trade_stats_index(logs)
        normal_setup_index = build_normal_setup_index(all_dfs_fast)
    else:
        pit_stats_index = dict(pit_stats_index or {})
        normal_setup_index = build_normal_setup_index(all_dfs_fast)
    if profile_timing_enabled:
        build_trade_index_sec = time.perf_counter() - t0

    initial_capital = params.initial_capital
    initial_capital_milli = money_to_milli(initial_capital)
    cash = initial_capital_milli
    fee_rebate_state = create_fee_rebate_state()
    rebate_month_key = None
    portfolio = {}
    active_extended_signals = {}
    active_extended_signals_by_member = {}
    active_reentry_watchlist = {}
    active_reentry_watchlists_by_member = {}
    # Optimizer single-member replay repeatedly sizes the same continuation geometry
    # across adjacent days.  Fee semantics are immutable for ordinary stocks/funds,
    # and exact qty is a pure function of the frozen geometry + capital/fee context.
    # Keep these caches trial-local so no state can leak across scientific runs.
    candidate_sizing_fee_context_cache = {}
    candidate_sizing_qty_cache = {}
    trade_history, equity_curve, closed_trades_stats = [], [], []
    normal_trade_count, extended_trade_count = 0, 0
    portfolio_entry_stats = {'filled_buy_count': 0}
    active_level_rows = [] if (profile_stats is not None and not is_training) else None
    capacity_rows = [] if (profile_stats is not None and not is_training) else None
    peak_equity, max_drawdown, current_equity = initial_capital_milli, 0.0, initial_capital_milli
    total_exposure, sim_days, total_missed_buys, total_missed_sells, max_exp = 0.0, 0, 0, 0, 0.0
    monthly_equities, bm_monthly_equities = [initial_capital], []
    yesterday_equity = initial_capital
    year_start_equity, year_end_equity = {}, {}
    year_first_sim_date, year_last_sim_date = {}, {}
    month_start_equity, month_end_equity = {}, {}
    month_first_sim_date, month_last_sim_date = {}, {}
    quarter_start_equity, quarter_end_equity = {}, {}
    quarter_first_sim_date, quarter_last_sim_date = {}, {}

    current_month = sorted_dates[start_idx].month if start_idx < len(sorted_dates) else 1

    current_bm_px = None
    benchmark_start_price = None
    bm_peak_price = None
    benchmark_loop_enabled = bool(benchmark_data and start_idx < len(sorted_dates) and benchmark_period_stats is None)
    if benchmark_period_stats is not None:
        benchmark_start_price = benchmark_period_stats.get('benchmark_start_price')
        bm_ret_pct = float(benchmark_period_stats.get('bm_ret_pct', 0.0))
        bm_max_drawdown = float(benchmark_period_stats.get('bm_max_drawdown', 0.0))
    elif benchmark_loop_enabled:
        benchmark_start_price, benchmark_anchor_date = get_fast_close_on_or_before(benchmark_data, sorted_dates[start_idx])
        if benchmark_start_price is not None:
            bm_peak_price = benchmark_start_price
            bm_monthly_equities.append(benchmark_start_price)
            if benchmark_anchor_date == sorted_dates[start_idx]:
                current_bm_px = benchmark_start_price
        yesterday_bm_px, bm_max_drawdown, bm_ret_pct = current_bm_px, 0.0, 0.0
    else:
        yesterday_bm_px, bm_max_drawdown, bm_ret_pct = None, 0.0, 0.0

    pit_stats_cursor = {}

    initialize_portfolio_replay_counts(replay_counts)

    training_idle_fast_path_enabled = bool(
        is_training
        and replay_counts is None
        and active_params_resolver is None
        and active_context_resolver is None
        and active_param_ensemble_resolver is None
        and active_context_ensemble_resolver is None
        and (not capture_equity_curve)
        and benchmark_period_stats is not None
    )

    replay_progress_start = time.perf_counter()
    replay_total_days = max(0, len(sorted_dates) - start_idx)
    for i in range(start_idx, len(sorted_dates)):
        t_day_start = time.perf_counter() if profile_timing_enabled else None
        sim_days += 1
        today = sorted_dates[i]
        today_rebate_month_key = (int(today.year), int(today.month))
        if rebate_month_key is None:
            rebate_month_key = today_rebate_month_key
        elif today_rebate_month_key != rebate_month_key:
            cash += settle_fee_rebate(fee_rebate_state)
            rebate_month_key = today_rebate_month_key
        day_context = resolve_portfolio_day_context(
            today,
            params=params,
            all_dfs_fast=all_dfs_fast,
            pit_stats_index=pit_stats_index,
            normal_setup_index=normal_setup_index,
            pit_stats_cursor=pit_stats_cursor,
            active_params_resolver=active_params_resolver,
            active_context_resolver=active_context_resolver,
            active_param_ensemble_resolver=active_param_ensemble_resolver,
            active_context_ensemble_resolver=active_context_ensemble_resolver,
            ensemble_min_agree=ensemble_min_agree,
        )
        day_ensemble_members = day_context.ensemble_members
        day_ensemble_contexts = day_context.ensemble_contexts
        day_ensemble_min_agree = day_context.ensemble_min_agree
        use_param_ensemble = day_context.use_param_ensemble
        day_params = day_context.params
        day_all_dfs_fast = day_context.all_dfs_fast
        day_pit_stats_index = day_context.pit_stats_index
        day_normal_setup_index = day_context.normal_setup_index
        day_pit_stats_cursor = day_context.pit_stats_cursor

        if today.year not in year_start_equity:
            year_start_equity[today.year] = milli_to_money(current_equity)
            year_first_sim_date[today.year] = pd.Timestamp(today)
        month_key = (int(today.year), int(today.month))
        if month_key not in month_start_equity:
            month_start_equity[month_key] = milli_to_money(current_equity)
            month_first_sim_date[month_key] = pd.Timestamp(today)
        quarter_key = (int(today.year), int((today.month - 1) // 3 + 1))
        if quarter_key not in quarter_start_equity:
            quarter_start_equity[quarter_key] = milli_to_money(current_equity)
            quarter_first_sim_date[quarter_key] = pd.Timestamp(today)

        normal_setup_entries_today = day_normal_setup_index.get(today, [])
        has_portfolio_work_today = portfolio_has_day_work(
            today,
            day_context=day_context,
            portfolio=portfolio,
            active_extended_signals=active_extended_signals,
            active_extended_signals_by_member=active_extended_signals_by_member,
            active_reentry_watchlist=active_reentry_watchlist,
            active_reentry_watchlists_by_member=active_reentry_watchlists_by_member,
            normal_setup_entries_today=normal_setup_entries_today,
        )

        if profile_timing_enabled:
            day_setup_sec += time.perf_counter() - t_day_start

        if bool(training_idle_fast_path_enabled) and not bool(has_portfolio_work_today):
            t0 = time.perf_counter() if profile_timing_enabled else None
            current_equity_money = milli_to_money(current_equity)
            if today.month != current_month:
                monthly_equities.append(yesterday_equity)
                current_month = today.month
            yesterday_equity = current_equity_money
            year_end_equity[today.year] = current_equity_money
            year_last_sim_date[today.year] = pd.Timestamp(today)
            month_end_equity[month_key] = current_equity_money
            month_last_sim_date[month_key] = pd.Timestamp(today)
            quarter_end_equity[quarter_key] = current_equity_money
            quarter_last_sim_date[quarter_key] = pd.Timestamp(today)
            idle_fast_path_days += 1
            if profile_timing_enabled:
                bookkeeping_sec += time.perf_counter() - t0
                day_loop_sec += time.perf_counter() - t_day_start
            continue

        current_equity_money = milli_to_money(current_equity)
        cash_money = milli_to_money(cash)

        sold_today = set()
        pre_market_position_count = len(portfolio)
        daily_filled_buy_count_before = int(portfolio_entry_stats.get('filled_buy_count', 0) or 0)
        daily_missed_buy_count_before = int(total_missed_buys)
        candidate_sources_today = False
        orderable_candidates_today = []
        cleanup_candidate_rows_by_ticker = None
        resource_action_candidates_today = []
        resource_selection_diag = {}

        if verbose and (not is_training) and ((i - start_idx) % 5 == 0):
            rebate_receivable_money = milli_to_money(get_fee_rebate_receivable_milli(fee_rebate_state))
            invested_for_display = max(0.0, current_equity_money - cash_money - rebate_receivable_money)
            exp = (invested_for_display / current_equity_money) * 100 if current_equity_money > 0 else 0
            replay_day_number = i - start_idx + 1
            replay_elapsed = _format_replay_elapsed(time.perf_counter() - replay_progress_start)
            print(
                f"\033[90m⏳ 推進中: {_format_replay_date(today)} | "
                f"日序: {replay_day_number}/{replay_total_days} | "
                f"資產: {current_equity_money:,.0f} | 水位: {exp:>5.1f}% | "
                f"elapsed={replay_elapsed}...\033[0m",
                end="\r",
                flush=True,
            )

        if has_portfolio_work_today:
            available_cash = cash
            sizing_equity = resolve_portfolio_sizing_equity(current_equity_money, initial_capital, day_params)
            extended_day_market_cache = {} if not use_param_ensemble else None
            t0 = time.perf_counter() if profile_timing_enabled else None
            activate_portfolio_reentry_for_day(
                today=today,
                day_context=day_context,
                portfolio=portfolio,
                sold_today=sold_today,
                active_reentry_watchlist=active_reentry_watchlist,
                active_reentry_watchlists_by_member=active_reentry_watchlists_by_member,
                active_extended_signals=active_extended_signals,
                active_extended_signals_by_member=active_extended_signals_by_member,
                is_training=is_training,
                replay_counts=replay_counts,
                replay_execution_rows=replay_execution_rows,
                replay_selector_trace_rows=replay_selector_trace_rows,
                replay_phase=_run_portfolio_replay_phase,
                single_reentry_callback=activate_breakout_reentry_signals_for_day,
                ensemble_reentry_callback=_activate_ensemble_reentry_signals_for_day,
            )
            if profile_timing_enabled:
                reentry_sec += time.perf_counter() - t0
            gate = resolve_portfolio_candidate_gate(
                today=today,
                day_context=day_context,
                portfolio=portfolio,
                sold_today=sold_today,
                normal_setup_entries_today=normal_setup_entries_today,
                active_extended_signals=active_extended_signals,
                active_extended_signals_by_member=active_extended_signals_by_member,
                is_training=is_training,
                replay_counts=replay_counts,
                enable_rotation=enable_rotation,
                max_positions=max_positions,
            )
            pre_market_occupied = gate.pre_market_occupied
            candidate_sources_today = gate.has_candidates
            no_entry_capacity_today = gate.no_entry_capacity

            if no_entry_capacity_today:
                t0 = time.perf_counter() if profile_timing_enabled else None
                did_track = track_portfolio_full_capacity_signals(
                    today=today,
                    day_context=day_context,
                    normal_setup_entries_today=normal_setup_entries_today,
                    portfolio=portfolio,
                    sold_today=sold_today,
                    active_extended_signals=active_extended_signals,
                    active_extended_signals_by_member=active_extended_signals_by_member,
                    candidate_subprofile=candidate_subprofile,
                    replay_phase=_run_portfolio_replay_phase,
                    single_track_callback=track_normal_setup_signals_for_day,
                    ensemble_track_callback=_track_ensemble_normal_setup_signals_for_day,
                )
                if profile_timing_enabled and did_track:
                    candidate_scan_sec += time.perf_counter() - t0
                before_trade_rows = -1
            elif candidate_sources_today:
                t0 = time.perf_counter() if profile_timing_enabled else None
                candidates_today, orderable_candidates_today, normal_setup_tickers_today = build_portfolio_day_candidate_rows(
                    today=today,
                    day_context=day_context,
                    portfolio=portfolio,
                    sold_today=sold_today,
                    active_extended_signals=active_extended_signals,
                    active_extended_signals_by_member=active_extended_signals_by_member,
                    current_equity_money=current_equity_money,
                    initial_capital=initial_capital,
                    sizing_equity=sizing_equity,
                    replay_counts=replay_counts,
                    replay_execution_rows=replay_execution_rows,
                    replay_selector_trace_rows=replay_selector_trace_rows,
                    enable_rotation=enable_rotation,
                    is_training=is_training,
                    verbose=verbose,
                    extended_day_market_cache=extended_day_market_cache,
                    persistent_sizing_fee_context_cache=candidate_sizing_fee_context_cache,
                    persistent_sizing_qty_cache=candidate_sizing_qty_cache,
                    candidate_subprofile=candidate_subprofile,
                    replay_phase=_run_portfolio_replay_phase,
                    single_candidate_builder=build_daily_candidates,
                    ensemble_candidate_builder=_build_daily_ensemble_candidates,
                )
                if profile_timing_enabled:
                    candidate_scan_sec += time.perf_counter() - t0

                cleanup_candidate_rows_by_ticker = resolve_cleanup_candidate_rows(
                    orderable_candidates_today,
                    is_ensemble=use_param_ensemble,
                    extended_day_market_cache=extended_day_market_cache,
                )

                t0 = time.perf_counter() if profile_timing_enabled else None
                selected = select_portfolio_candidate_orders(
                    today=today,
                    orderable_rows=orderable_candidates_today,
                    replay_selector_trace_rows=replay_selector_trace_rows,
                    replay_execution_rows=replay_execution_rows,
                    replay_counts=replay_counts,
                    is_training=is_training,
                    params=day_params,
                    available_cash=available_cash,
                    sizing_equity=sizing_equity,
                    pre_market_occupied=pre_market_occupied,
                    max_positions=max_positions,
                    replay_phase=_run_portfolio_replay_phase,
                    reorder_callback=reorder_candidates_for_resource_aware_quality,
                    select_action_callback=select_resource_aware_action_candidates,
                )
                orderable_candidates_today = selected.orderable_rows
                resource_action_candidates_today = selected.action_rows
                resource_selection_diag = selected.diagnostic
                resource_selection_input_rows = selected.trace_input_rows
                if profile_timing_enabled:
                    resource_select_sec += time.perf_counter() - t0

                append_portfolio_selector_trace(
                    today=today,
                    replay_selector_trace_rows=replay_selector_trace_rows,
                    resource_selection_input_rows=resource_selection_input_rows,
                    resource_selection_diag=resource_selection_diag,
                    available_cash=available_cash,
                    sizing_equity=sizing_equity,
                    params=day_params,
                    repair_mechanism_builder=build_max_dl_repair_mechanism_diagnostic,
                    candidate_snapshot=_candidate_replay_snapshot,
                )

                before_trade_rows = record_portfolio_candidate_replay(
                    today=today,
                    replay_counts=replay_counts,
                    candidates_today=candidates_today,
                    orderable_candidates_today=orderable_candidates_today,
                    replay_execution_rows=replay_execution_rows,
                    trade_history=trade_history,
                    all_dfs_fast=day_all_dfs_fast,
                    sizing_equity=sizing_equity,
                    candidate_snapshot=_candidate_replay_snapshot,
                    execution_snapshot=_candidate_execution_replay_snapshot,
                )

                if resource_action_candidates_today:
                    rotation_result = rotate_portfolio_day(
                        today=today,
                        day_context=day_context,
                        portfolio=portfolio,
                        resource_action_candidates_today=resource_action_candidates_today,
                        max_positions=max_positions,
                        enable_rotation=enable_rotation,
                        sold_today=sold_today,
                        cash=cash,
                        closed_trades_stats=closed_trades_stats,
                        trade_history=trade_history,
                        is_training=is_training,
                        normal_trade_count=normal_trade_count,
                        extended_trade_count=extended_trade_count,
                        active_level_rows=active_level_rows,
                        fee_rebate_state=fee_rebate_state,
                        profile_timing_enabled=profile_timing_enabled,
                    )
                    cash = rotation_result.cash
                    normal_trade_count = rotation_result.normal_trade_count
                    extended_trade_count = rotation_result.extended_trade_count
                    rotation_sec += rotation_result.elapsed_sec
            else:
                before_trade_rows = len(trade_history) if replay_counts is not None else -1

            event_result = settle_and_execute_portfolio_day(
                today=today,
                day_context=day_context,
                portfolio=portfolio,
                sold_today=sold_today,
                cash=cash,
                closed_trades_stats=closed_trades_stats,
                trade_history=trade_history,
                is_training=is_training,
                total_missed_sells=total_missed_sells,
                total_missed_buys=total_missed_buys,
                normal_trade_count=normal_trade_count,
                extended_trade_count=extended_trade_count,
                active_level_rows=active_level_rows,
                active_reentry_watchlist=active_reentry_watchlist,
                active_reentry_watchlists_by_member=active_reentry_watchlists_by_member,
                fee_rebate_state=fee_rebate_state,
                no_entry_capacity_today=no_entry_capacity_today,
                resource_action_candidates_today=resource_action_candidates_today,
                max_positions=max_positions,
                active_extended_signals=active_extended_signals,
                available_cash=available_cash,
                sizing_equity=sizing_equity,
                portfolio_entry_stats=portfolio_entry_stats,
                replay_execution_rows=replay_execution_rows,
                extended_day_market_cache=extended_day_market_cache,
                profile_timing_enabled=profile_timing_enabled,
            )
            cash = event_result.cash
            total_missed_sells = event_result.total_missed_sells
            total_missed_buys = event_result.total_missed_buys
            normal_trade_count = event_result.normal_trade_count
            extended_trade_count = event_result.extended_trade_count
            settle_sec += event_result.settle_sec
            buy_sec += event_result.buy_sec
            t0 = time.perf_counter() if profile_timing_enabled else None
            cleanup_portfolio_extended_signals_for_day(
                today=today,
                day_context=day_context,
                portfolio=portfolio,
                active_extended_signals=active_extended_signals,
                active_extended_signals_by_member=active_extended_signals_by_member,
                current_equity_money=current_equity_money,
                initial_capital=initial_capital,
                sizing_equity=sizing_equity,
                verbose=verbose,
                is_training=is_training,
                replay_counts=replay_counts,
                replay_execution_rows=replay_execution_rows,
                replay_selector_trace_rows=replay_selector_trace_rows,
                extended_day_market_cache=extended_day_market_cache,
                cleanup_candidate_rows_by_ticker=cleanup_candidate_rows_by_ticker,
                replay_phase=_run_portfolio_replay_phase,
                single_cleanup_callback=cleanup_extended_signals_for_day,
                ensemble_cleanup_callback=_cleanup_ensemble_extended_signals_for_day,
            )
            if profile_timing_enabled:
                cleanup_sec += time.perf_counter() - t0
        elif replay_counts is not None:
            before_trade_rows = len(trade_history)
        else:
            before_trade_rows = -1

        t0 = time.perf_counter() if profile_timing_enabled else None
        day_mark = mark_portfolio_day_equity(
            cash=cash,
            portfolio=portfolio,
            all_dfs_fast=day_all_dfs_fast,
            today=today,
            params=day_params,
            fee_rebate_state=fee_rebate_state,
        )
        today_equity = day_mark.equity_milli
        today_equity_money = day_mark.equity_money
        cash_money = day_mark.cash_money
        if profile_timing_enabled:
            equity_mark_sec += time.perf_counter() - t0

        t0 = time.perf_counter() if profile_timing_enabled else None
        if active_level_rows is not None:
            _append_portfolio_active_level_rows(active_level_rows, portfolio, today)
            if use_param_ensemble:
                _append_portfolio_extended_shadow_level_rows(active_level_rows, _flatten_ensemble_extended_signals(active_extended_signals_by_member), portfolio, today)
            else:
                _append_portfolio_extended_shadow_level_rows(active_level_rows, active_extended_signals, portfolio, today)

        if capacity_rows is not None:
            capacity_rows.append(build_portfolio_capacity_day_row(
                today,
                max_positions=max_positions,
                pre_market_position_count=pre_market_position_count,
                candidate_sources_today=candidate_sources_today,
                orderable_candidates_today=orderable_candidates_today,
                portfolio=portfolio,
                portfolio_entry_stats=portfolio_entry_stats,
                daily_filled_buy_count_before=daily_filled_buy_count_before,
                total_missed_buys=total_missed_buys,
                daily_missed_buy_count_before=daily_missed_buy_count_before,
                resource_selection_diag=resource_selection_diag,
            ))

        current_equity = today_equity
        current_equity_money = today_equity_money
        day_exposure = measure_portfolio_day_exposure(
            current_equity_money=current_equity_money,
            cash_money=cash_money,
            initial_capital=initial_capital,
            fee_rebate_state=fee_rebate_state,
        )
        invested_capital = day_exposure.invested_capital
        exposure_pct = day_exposure.exposure_pct
        total_exposure += exposure_pct
        if exposure_pct > max_exp:
            max_exp = exposure_pct
        strategy_ret_pct = day_exposure.strategy_ret_pct

        if benchmark_loop_enabled and benchmark_start_price is not None and has_fast_date(benchmark_data, today):
            current_bm_px = get_fast_close(benchmark_data, date=today)
            bm_ret_pct = (current_bm_px - benchmark_start_price) / benchmark_start_price * 100 if benchmark_start_price > 0 else 0.0

            if bm_peak_price is not None:
                if current_bm_px > bm_peak_price:
                    bm_peak_price = current_bm_px
                current_bm_drawdown = (bm_peak_price - current_bm_px) / bm_peak_price * 100
                if current_bm_drawdown > bm_max_drawdown:
                    bm_max_drawdown = current_bm_drawdown

        if today.month != current_month:
            monthly_equities.append(yesterday_equity)
            if benchmark_loop_enabled and yesterday_bm_px is not None:
                bm_monthly_equities.append(yesterday_bm_px)
            current_month = today.month

        yesterday_equity = current_equity_money
        if benchmark_loop_enabled:
            yesterday_bm_px = current_bm_px
        year_end_equity[today.year] = current_equity_money
        year_last_sim_date[today.year] = pd.Timestamp(today)
        month_end_equity[month_key] = current_equity_money
        month_last_sim_date[month_key] = pd.Timestamp(today)
        quarter_end_equity[quarter_key] = current_equity_money
        quarter_last_sim_date[quarter_key] = pd.Timestamp(today)

        if (not is_training) or capture_equity_curve:
            equity_curve.append({
                "Date": today.strftime('%Y-%m-%d'),
                "Equity": current_equity_money,
                "Invested_Amount": invested_capital,
                "Exposure_Pct": exposure_pct,
                "Strategy_Return_Pct": strategy_ret_pct,
                f"Benchmark_{benchmark_ticker}_Pct": bm_ret_pct
            })

        if current_equity > peak_equity:
            peak_equity = current_equity
        drawdown = (peak_equity - current_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        append_portfolio_trade_replay(
            replay_counts=replay_counts,
            before_trade_rows=before_trade_rows,
            trade_history=trade_history,
        )

        if profile_timing_enabled:
            bookkeeping_sec += time.perf_counter() - t0
            day_loop_sec += time.perf_counter() - t_day_start

    # # (AI註: 月底權益應以期末強制結算後的真實 final equity 為準，故移到 closeout 後再 append)

    t0 = time.perf_counter() if profile_timing_enabled else None
    last_date = sorted_dates[-1] if len(sorted_dates) > 0 else None
    closeout_params = active_params_resolver(last_date) if (active_params_resolver is not None and last_date is not None) else params
    today_equity, normal_trade_count, extended_trade_count = closeout_open_positions(
        portfolio=portfolio,
        cash=cash,
        params=closeout_params,
        trade_history=trade_history,
        is_training=is_training,
        closed_trades_stats=closed_trades_stats,
        normal_trade_count=normal_trade_count,
        extended_trade_count=extended_trade_count,
        last_date=last_date,
        active_level_rows=active_level_rows,
        fee_rebate_state=fee_rebate_state,
    )

    if profile_timing_enabled:
        closeout_sec = time.perf_counter() - t0

    terminal_equity = reconcile_portfolio_terminal_equity(
        closeout_equity_milli=today_equity,
        fee_rebate_state=fee_rebate_state,
        last_date=last_date,
        year_end_equity=year_end_equity,
        month_end_equity=month_end_equity,
        quarter_end_equity=quarter_end_equity,
        peak_equity=peak_equity,
        max_drawdown=max_drawdown,
        sorted_dates=sorted_dates,
        start_idx=start_idx,
        monthly_equities=monthly_equities,
        benchmark_loop_enabled=benchmark_loop_enabled,
        current_bm_px=current_bm_px,
        bm_monthly_equities=bm_monthly_equities,
        initial_capital=initial_capital,
        equity_curve=equity_curve,
        is_training=is_training,
        capture_equity_curve=capture_equity_curve,
    )
    final_equity_money = terminal_equity.final_equity_money
    total_return = terminal_equity.total_return
    max_drawdown = terminal_equity.max_drawdown

    t0 = time.perf_counter() if profile_timing_enabled else None
    r_squared, monthly_win_rate = calc_curve_stats(monthly_equities)
    if benchmark_period_stats is not None:
        bm_r_squared = float(benchmark_period_stats.get('bm_r_squared', 0.0))
        bm_monthly_win_rate = float(benchmark_period_stats.get('bm_monthly_win_rate', 0.0))
    else:
        bm_r_squared, bm_monthly_win_rate = calc_curve_stats(bm_monthly_equities)
    curve_stats_sec = (time.perf_counter() - t0) if profile_timing_enabled else 0.0

    terminal_metrics = summarize_portfolio_terminal_metrics(
        closed_trades_stats=closed_trades_stats,
        pit_stats_index=pit_stats_index,
        sorted_dates=sorted_dates,
        start_idx=start_idx,
        profile_requested=profile_stats is not None,
        active_context_resolver=active_context_resolver,
        active_context_ensemble_resolver=active_context_ensemble_resolver,
        ev_calc_method_getter=get_ev_calc_method,
        total_exposure=total_exposure,
        sim_days=sim_days,
        portfolio_entry_stats=portfolio_entry_stats,
        total_missed_buys=total_missed_buys,
        initial_capital=initial_capital,
        final_equity_money=final_equity_money,
        benchmark_start_price=benchmark_start_price,
        bm_ret_pct=bm_ret_pct,
        benchmark_period_stats=benchmark_period_stats,
    )
    period_reports = build_portfolio_period_reports(
        sorted_dates=sorted_dates,
        year_start_equity=year_start_equity,
        year_end_equity=year_end_equity,
        year_first_sim_date=year_first_sim_date,
        year_last_sim_date=year_last_sim_date,
        month_start_equity=month_start_equity,
        month_end_equity=month_end_equity,
        month_first_sim_date=month_first_sim_date,
        month_last_sim_date=month_last_sim_date,
        quarter_start_equity=quarter_start_equity,
        quarter_end_equity=quarter_end_equity,
        quarter_first_sim_date=quarter_first_sim_date,
        quarter_last_sim_date=quarter_last_sim_date,
        benchmark_period_stats=benchmark_period_stats,
        benchmark_data=benchmark_data,
    )

    if profile_stats is not None:
        # Per-run instrumentation is sidecar-only; no timing enters scientific identity.
        profile_timings = {
            'ticker_dates_sec': ticker_dates_sec,
            'build_trade_index_sec': build_trade_index_sec,
            'day_loop_sec': day_loop_sec,
            'candidate_scan_sec': candidate_scan_sec,
            'day_setup_sec': day_setup_sec,
            'reentry_sec': reentry_sec,
            'resource_select_sec': resource_select_sec,
            'cleanup_sec': cleanup_sec,
            'bookkeeping_sec': bookkeeping_sec,
            'rotation_sec': rotation_sec,
            'settle_sec': settle_sec,
            'buy_sec': buy_sec,
            'equity_mark_sec': equity_mark_sec,
            'closeout_sec': closeout_sec,
            'curve_stats_sec': curve_stats_sec,
        }
        publish_portfolio_profile_stats(
            profile_stats=profile_stats,
            fee_rebate_state=fee_rebate_state,
            timing_enabled=profile_timing_enabled,
            portfolio_start=t_portfolio_start,
            timings=profile_timings,
            candidate_subprofile=candidate_subprofile,
            idle_fast_path_days=idle_fast_path_days,
            closed_trades_stats=closed_trades_stats,
            terminal_metrics=terminal_metrics,
            period_reports=period_reports,
            is_training=is_training,
            active_level_rows=active_level_rows,
            capacity_rows=capacity_rows,
            capture_equity_curve=capture_equity_curve,
            equity_curve=equity_curve,
        )

    return assemble_portfolio_timeline_result(
        is_training=is_training,
        equity_curve=equity_curve,
        trade_history=trade_history,
        benchmark_ticker=benchmark_ticker,
        terminal_metrics=terminal_metrics,
        total_return=total_return,
        max_drawdown=max_drawdown,
        final_equity_money=final_equity_money,
        max_exp=max_exp,
        bm_ret_pct=bm_ret_pct,
        bm_max_drawdown=bm_max_drawdown,
        total_missed_buys=total_missed_buys,
        total_missed_sells=total_missed_sells,
        r_squared=r_squared,
        monthly_win_rate=monthly_win_rate,
        bm_r_squared=bm_r_squared,
        bm_monthly_win_rate=bm_monthly_win_rate,
    )
