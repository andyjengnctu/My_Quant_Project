import pandas as pd
import time

from core.exact_accounting import milli_to_money, money_to_milli
from core.capital_policy import resolve_portfolio_sizing_equity
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
from core.seed_ensemble_policy import resolve_seed_ensemble_min_agree

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
):
    profile_timing_enabled = bool(profile_stats.get("_timing_enabled", True)) if profile_stats is not None else False
    capture_equity_curve = bool(profile_stats.get("capture_equity_curve", False)) if profile_stats is not None else False
    t_portfolio_start = time.perf_counter() if profile_timing_enabled else None
    candidate_scan_sec = 0.0
    day_loop_sec = 0.0
    rotation_sec = 0.0
    settle_sec = 0.0
    buy_sec = 0.0
    equity_mark_sec = 0.0
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
    portfolio = {}
    active_extended_signals = {}
    active_extended_signals_by_member = {}
    active_reentry_watchlist = {}
    active_reentry_watchlists_by_member = {}
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

    if replay_counts is not None:
        for ticker, bucket in replay_counts.items():
            if not isinstance(bucket, dict):
                replay_counts[ticker] = {}
                bucket = replay_counts[ticker]
            bucket.setdefault("candidate_dates", [])
            bucket.setdefault("orderable_dates", [])
            bucket.setdefault("candidate_rows", [])
            bucket.setdefault("orderable_rows", [])
            bucket.setdefault("trade_rows", [])

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
        day_ensemble_members = list(active_param_ensemble_resolver(today) or []) if active_param_ensemble_resolver is not None else []
        day_ensemble_contexts = list(active_context_ensemble_resolver(today) or []) if active_context_ensemble_resolver is not None else []
        day_ensemble_min_agree = resolve_seed_ensemble_min_agree(
            len(day_ensemble_members),
            ensemble_min_agree if ensemble_min_agree is not None else "auto",
        ) if day_ensemble_members else None
        single_member_identity_replay = bool(
            len(day_ensemble_members) == 1
            and int(day_ensemble_min_agree or 1) <= 1
        )
        use_param_ensemble = bool(day_ensemble_members) and not single_member_identity_replay
        if day_ensemble_members:
            day_params = day_ensemble_members[0].get("params_obj") or params
            day_context = day_ensemble_contexts[0] if day_ensemble_contexts else None
        else:
            day_params = active_params_resolver(today) if active_params_resolver is not None else params
            day_context = active_context_resolver(today) if active_context_resolver is not None else None
        if day_context is None:
            day_all_dfs_fast = all_dfs_fast
            day_pit_stats_index = pit_stats_index
            day_normal_setup_index = normal_setup_index
            day_pit_stats_cursor = pit_stats_cursor
        else:
            day_all_dfs_fast = day_context.get('all_dfs_fast') or all_dfs_fast
            day_pit_stats_index = day_context.get('all_pit_stats_index') or pit_stats_index
            day_normal_setup_index = day_context.get('normal_setup_index') or normal_setup_index
            day_cursor_key = id(day_pit_stats_index)
            day_pit_stats_cursor = pit_stats_cursor.setdefault(day_cursor_key, {})

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

        sold_today = set()
        pre_market_position_count = len(portfolio)
        daily_filled_buy_count_before = int(portfolio_entry_stats.get('filled_buy_count', 0) or 0)
        daily_missed_buy_count_before = int(total_missed_buys)
        candidate_sources_today = False
        orderable_candidates_today = []
        resource_action_candidates_today = []
        resource_selection_diag = {
            'enabled': False,
            'mode': 'inactive',
            'free_slots': max(0, int(max_positions) - int(pre_market_position_count)),
            'candidate_count': 0,
            'baseline_selected_count': 0,
            'baseline_pass_count': 0,
            'baseline_reserved_cost_milli': 0,
            'selected_count': 0,
            'selected_pass_count': 0,
            'reserved_cost_milli': 0,
            'promoted_pass_count': 0,
            'changed': False,
            'baseline_scored_selected_count': 0,
            'selected_scored_count': 0,
            'baseline_selected_score_sum': 0.0,
            'selected_score_sum': 0.0,
            'baseline_selected_score_mean': None,
            'selected_score_mean': None,
            'promoted_score_orders': 0,
            'direct_score_order_feasible': False,
            'pre_market_order_limit': None,
            'max_dl_eligible': False,
            'max_dl_repair_steps': 0,
            'max_dl_repair_evaluations': 0,
            'max_dl_fallback_to_baseline': False,
            'max_dl_seed_fallback': False,
            'max_dl_feasible_ascent_steps': 0,
            'max_dl_feasible_ascent_evaluations': 0,
            'max_dl_feasible_ascent_local_optimum': False,
            'stale_score_membership_guard_enabled': False,
            'stale_score_membership_guard_max_age_days': None,
            'stale_score_candidate_count': 0,
            'stale_score_guard_triggered': False,
            'stale_score_guard_seed_blocked': False,
            'stale_score_guard_blocked_swaps': 0,
            'selector_elapsed_ns': 0,
        }
        normal_setup_entries_today = day_normal_setup_index.get(today, [])
        if use_param_ensemble:
            has_ensemble_normal_setup = any(bool((ctx.get("normal_setup_index") or {}).get(today, [])) for ctx in day_ensemble_contexts)
            has_ensemble_extended = any(bool(member_signals) for member_signals in active_extended_signals_by_member.values())
            has_ensemble_reentry_watch = any(bool(member_watchlist) for member_watchlist in active_reentry_watchlists_by_member.values())
            has_portfolio_work_today = bool(portfolio) or bool(has_ensemble_extended) or bool(has_ensemble_normal_setup) or bool(has_ensemble_reentry_watch)
        else:
            has_portfolio_work_today = bool(portfolio) or bool(active_extended_signals) or bool(active_reentry_watchlist) or bool(normal_setup_entries_today)

        if bool(training_idle_fast_path_enabled) and not bool(has_portfolio_work_today):
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
                day_loop_sec += time.perf_counter() - t_day_start
            continue

        current_equity_money = milli_to_money(current_equity)
        cash_money = milli_to_money(cash)

        if verbose and (not is_training) and ((i - start_idx) % 5 == 0):
            exp = ((current_equity_money - cash_money) / current_equity_money) * 100 if current_equity_money > 0 else 0
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
            if use_param_ensemble:
                _run_portfolio_replay_phase(
                    today,
                    "activate_ensemble_reentry",
                    _activate_ensemble_reentry_signals_for_day,
                    ensemble_members=day_ensemble_members,
                    ensemble_contexts=day_ensemble_contexts,
                    active_reentry_watchlists_by_member=active_reentry_watchlists_by_member,
                    active_extended_signals_by_member=active_extended_signals_by_member,
                    portfolio=portfolio,
                    sold_today=sold_today,
                    today=today,
                )
            else:
                _run_portfolio_replay_phase(
                    today,
                    "activate_reentry",
                    activate_breakout_reentry_signals_for_day,
                    active_reentry_watchlist=active_reentry_watchlist,
                    active_extended_signals=active_extended_signals,
                    portfolio=portfolio,
                    sold_today=sold_today,
                    all_dfs_fast=day_all_dfs_fast,
                    today=today,
                    params=day_params,
                )
            pre_market_occupied = len(portfolio) + len(sold_today)
            if use_param_ensemble:
                candidate_sources_today = any(bool((ctx.get("normal_setup_index") or {}).get(today, [])) for ctx in day_ensemble_contexts) or any(bool(member_signals) for member_signals in active_extended_signals_by_member.values())
            else:
                candidate_sources_today = bool(normal_setup_entries_today) or bool(active_extended_signals)
            no_entry_capacity_today = (
                bool(is_training)
                and replay_counts is None
                and (not bool(enable_rotation))
                and pre_market_occupied >= int(max_positions)
            )

            if no_entry_capacity_today:
                if use_param_ensemble:
                    has_ensemble_normal_setup = any(bool((ctx.get("normal_setup_index") or {}).get(today, [])) for ctx in day_ensemble_contexts)
                    if has_ensemble_normal_setup:
                        t0 = time.perf_counter() if profile_timing_enabled else None
                        _run_portfolio_replay_phase(
                            today,
                            "track_ensemble_normal_setup",
                            _track_ensemble_normal_setup_signals_for_day,
                            ensemble_members=day_ensemble_members,
                            ensemble_contexts=day_ensemble_contexts,
                            active_extended_signals_by_member=active_extended_signals_by_member,
                            portfolio=portfolio,
                            sold_today=sold_today,
                            today=today,
                        )
                        if profile_timing_enabled:
                            candidate_scan_sec += time.perf_counter() - t0
                elif normal_setup_entries_today:
                    t0 = time.perf_counter() if profile_timing_enabled else None
                    _run_portfolio_replay_phase(
                        today,
                        "track_normal_setup",
                        track_normal_setup_signals_for_day,
                        normal_setup_entries=normal_setup_entries_today,
                        portfolio=portfolio,
                        sold_today=sold_today,
                        all_dfs_fast=day_all_dfs_fast,
                        active_extended_signals=active_extended_signals,
                        pit_stats_index=day_pit_stats_index,
                        pit_stats_cursor=day_pit_stats_cursor,
                        today=today,
                        params=day_params,
                    )
                    if profile_timing_enabled:
                        candidate_scan_sec += time.perf_counter() - t0
                before_trade_rows = -1
            elif candidate_sources_today:
                t0 = time.perf_counter() if profile_timing_enabled else None
                if use_param_ensemble:
                    candidates_today, orderable_candidates_today, normal_setup_tickers_today = _run_portfolio_replay_phase(
                        today,
                        "build_daily_ensemble_candidates",
                        _build_daily_ensemble_candidates,
                        ensemble_members=day_ensemble_members,
                        ensemble_contexts=day_ensemble_contexts,
                        active_extended_signals_by_member=active_extended_signals_by_member,
                        portfolio=portfolio,
                        sold_today=sold_today,
                        today=today,
                        current_equity_money=current_equity_money,
                        initial_capital=initial_capital,
                        collect_all_candidates=replay_counts is not None,
                        min_agree=int(day_ensemble_min_agree or 1),
                        verbose=bool(verbose and (not is_training)),
                    )
                else:
                    candidates_today, orderable_candidates_today, normal_setup_tickers_today = _run_portfolio_replay_phase(
                        today,
                        "build_daily_candidates",
                        build_daily_candidates,
                        normal_setup_index=day_normal_setup_index,
                        active_extended_signals=active_extended_signals,
                        portfolio=portfolio,
                        sold_today=sold_today,
                        all_dfs_fast=day_all_dfs_fast,
                        pit_stats_index=day_pit_stats_index,
                        pit_stats_cursor=day_pit_stats_cursor,
                        today=today,
                        sizing_equity=sizing_equity,
                        params=day_params,
                        collect_all_candidates=replay_counts is not None,
                    )
                if profile_timing_enabled:
                    candidate_scan_sec += time.perf_counter() - t0

                orderable_candidates_today, resource_selection_diag = _run_portfolio_replay_phase(
                    today,
                    "resource_aware_binary_order",
                    reorder_candidates_for_resource_aware_quality,
                    orderable_candidates_today,
                    available_cash=available_cash,
                    sizing_equity=sizing_equity,
                    pre_market_occupied=pre_market_occupied,
                    max_positions=max_positions,
                    params=day_params,
                )
                resource_action_candidates_today = select_resource_aware_action_candidates(
                    orderable_candidates_today,
                    resource_selection_diag,
                )

                qualified_candidate_snapshots_today = []
                orderable_candidate_snapshots_today = []
                if replay_counts is not None:
                    for candidate in candidates_today:
                        snapshot = _candidate_replay_snapshot(
                            candidate,
                            fallback_trade_date=today,
                            is_orderable=False,
                        )
                        qualified_candidate_snapshots_today.append(snapshot)
                        ticker = str(snapshot.get("ticker", ""))
                        bucket = replay_counts.setdefault(
                            ticker,
                            {
                                "candidate_dates": [],
                                "orderable_dates": [],
                                "candidate_rows": [],
                                "orderable_rows": [],
                                "trade_rows": [],
                            },
                        )
                        bucket.setdefault("candidate_dates", []).append(today)
                        bucket.setdefault("candidate_rows", []).append(snapshot)
                    for candidate in orderable_candidates_today:
                        snapshot = _candidate_replay_snapshot(
                            candidate,
                            fallback_trade_date=today,
                            is_orderable=True,
                        )
                        orderable_candidate_snapshots_today.append(snapshot)
                        ticker = str(snapshot.get("ticker", ""))
                        bucket = replay_counts.setdefault(
                            ticker,
                            {
                                "candidate_dates": [],
                                "orderable_dates": [],
                                "candidate_rows": [],
                                "orderable_rows": [],
                                "trade_rows": [],
                            },
                        )
                        bucket.setdefault("orderable_dates", []).append(today)
                        bucket.setdefault("orderable_rows", []).append(snapshot)
                    before_trade_rows = len(trade_history)
                else:
                    before_trade_rows = -1

                if replay_execution_rows is not None:
                    for candidate in orderable_candidates_today:
                        replay_execution_rows.append(
                            _candidate_execution_replay_snapshot(
                                candidate,
                                fallback_trade_date=today,
                                all_dfs_fast=day_all_dfs_fast,
                                sizing_equity=sizing_equity,
                            )
                        )

                if resource_action_candidates_today:
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
                        all_dfs_fast=day_all_dfs_fast,
                        today=today,
                        pit_stats_index=day_pit_stats_index,
                        params=day_params,
                        cash=cash,
                        closed_trades_stats=closed_trades_stats,
                        trade_history=trade_history,
                        is_training=is_training,
                        normal_trade_count=normal_trade_count,
                        extended_trade_count=extended_trade_count,
                        active_level_rows=active_level_rows,
                    )
                    if profile_timing_enabled:
                        rotation_sec += time.perf_counter() - t0
            else:
                before_trade_rows = len(trade_history) if replay_counts is not None else -1

            t0 = time.perf_counter() if profile_timing_enabled else None
            cash, total_missed_sells, normal_trade_count, extended_trade_count = _run_portfolio_replay_phase(
                today,
                "settle_positions",
                settle_portfolio_positions,
                portfolio=portfolio,
                sold_today=sold_today,
                all_dfs_fast=day_all_dfs_fast,
                today=today,
                params=day_params,
                cash=cash,
                closed_trades_stats=closed_trades_stats,
                trade_history=trade_history,
                is_training=is_training,
                total_missed_sells=total_missed_sells,
                normal_trade_count=normal_trade_count,
                extended_trade_count=extended_trade_count,
                active_level_rows=active_level_rows,
                active_reentry_watchlist=active_reentry_watchlist if not use_param_ensemble else None,
                active_reentry_watchlists_by_member=active_reentry_watchlists_by_member if use_param_ensemble else None,
            )
            if profile_timing_enabled:
                settle_sec += time.perf_counter() - t0

            can_try_entries_today = (
                (not no_entry_capacity_today)
                and bool(resource_action_candidates_today)
                and (len(portfolio) + len(sold_today)) < int(max_positions)
            )
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
                    all_dfs_fast=day_all_dfs_fast,
                    today=today,
                    params=day_params,
                    cash=cash,
                    available_cash=available_cash,
                    sizing_equity=sizing_equity,
                    max_positions=max_positions,
                    trade_history=trade_history,
                    is_training=is_training,
                    total_missed_buys=total_missed_buys,
                    entry_stats=portfolio_entry_stats,
                    replay_execution_rows=replay_execution_rows,
                )
                if profile_timing_enabled:
                    buy_sec += time.perf_counter() - t0

            if use_param_ensemble:
                _run_portfolio_replay_phase(
                    today,
                    "cleanup_ensemble_extended_signals",
                    _cleanup_ensemble_extended_signals_for_day,
                    ensemble_members=day_ensemble_members,
                    ensemble_contexts=day_ensemble_contexts,
                    active_extended_signals_by_member=active_extended_signals_by_member,
                    portfolio=portfolio,
                    today=today,
                    current_equity_money=current_equity_money,
                    initial_capital=initial_capital,
                    verbose=bool(verbose and (not is_training)),
                )
            else:
                _run_portfolio_replay_phase(
                    today,
                    "cleanup_extended_signals",
                    cleanup_extended_signals_for_day,
                    active_extended_signals=active_extended_signals,
                    portfolio=portfolio,
                    all_dfs_fast=day_all_dfs_fast,
                    today=today,
                    params=day_params,
                    sizing_capital=sizing_equity,
                )
        elif replay_counts is not None:
            before_trade_rows = len(trade_history)
        else:
            before_trade_rows = -1

        t0 = time.perf_counter() if profile_timing_enabled else None
        if portfolio:
            today_equity = calc_mark_to_market_equity(cash, portfolio, day_all_dfs_fast, today, day_params)
        else:
            today_equity = cash
        today_equity_money = milli_to_money(today_equity)
        cash_money = milli_to_money(cash)
        if profile_timing_enabled:
            equity_mark_sec += time.perf_counter() - t0

        if active_level_rows is not None:
            _append_portfolio_active_level_rows(active_level_rows, portfolio, today)
            if use_param_ensemble:
                _append_portfolio_extended_shadow_level_rows(active_level_rows, _flatten_ensemble_extended_signals(active_extended_signals_by_member), portfolio, today)
            else:
                _append_portfolio_extended_shadow_level_rows(active_level_rows, active_extended_signals, portfolio, today)

        if capacity_rows is not None:
            pre_market_free_slots = max(0, int(max_positions) - int(pre_market_position_count))
            orderable_candidate_count = int(len(orderable_candidates_today))
            post_execution_position_count = int(len(portfolio))
            capacity_rows.append({
                'Date': today.strftime('%Y-%m-%d') if hasattr(today, 'strftime') else str(today),
                'Max_Positions': int(max_positions),
                'Pre_Market_Positions': int(pre_market_position_count),
                'Pre_Market_Free_Slots': int(pre_market_free_slots),
                'Candidate_Source_Active': bool(candidate_sources_today),
                'Orderable_Candidates': orderable_candidate_count,
                'Candidate_Supply_Gap': max(0, pre_market_free_slots - orderable_candidate_count),
                'Post_Execution_Positions': post_execution_position_count,
                'End_Position_Gap': max(0, int(max_positions) - post_execution_position_count),
                'Filled_Buys_Today': max(0, int(portfolio_entry_stats.get('filled_buy_count', 0) or 0) - daily_filled_buy_count_before),
                'Missed_Buys_Today': max(0, int(total_missed_buys) - daily_missed_buy_count_before),
                'Resource_Aware_DL_Enabled': bool(resource_selection_diag.get('enabled', False)),
                'Resource_Aware_Mode': str(resource_selection_diag.get('mode') or 'inactive'),
                'Resource_Aware_Changed': bool(resource_selection_diag.get('changed', False)),
                'Resource_Aware_Baseline_Selected': int(resource_selection_diag.get('baseline_selected_count', 0) or 0),
                'Resource_Aware_Baseline_PASS': int(resource_selection_diag.get('baseline_pass_count', 0) or 0),
                'Resource_Aware_Selected': int(resource_selection_diag.get('selected_count', 0) or 0),
                'Resource_Aware_Selected_PASS': int(resource_selection_diag.get('selected_pass_count', 0) or 0),
                'Resource_Aware_Promoted_PASS': int(resource_selection_diag.get('promoted_pass_count', 0) or 0),
                'Resource_Aware_Baseline_Reserved_Milli': int(resource_selection_diag.get('baseline_reserved_cost_milli', 0) or 0),
                'Resource_Aware_Reserved_Milli': int(resource_selection_diag.get('reserved_cost_milli', 0) or 0),
                'Resource_Aware_Baseline_PASS_Reserved_Milli': int(resource_selection_diag.get('baseline_pass_reserved_cost_milli', 0) or 0),
                'Resource_Aware_PASS_Reserved_Milli': int(resource_selection_diag.get('pass_reserved_cost_milli', 0) or 0),
                'Resource_Aware_Baseline_Scored_Selected': int(resource_selection_diag.get('baseline_scored_selected_count', 0) or 0),
                'Resource_Aware_Selected_Scored': int(resource_selection_diag.get('selected_scored_count', 0) or 0),
                'Resource_Aware_Baseline_Score_Sum': float(resource_selection_diag.get('baseline_selected_score_sum', 0.0) or 0.0),
                'Resource_Aware_Score_Sum': float(resource_selection_diag.get('selected_score_sum', 0.0) or 0.0),
                'Resource_Aware_Baseline_Score_Mean': resource_selection_diag.get('baseline_selected_score_mean'),
                'Resource_Aware_Score_Mean': resource_selection_diag.get('selected_score_mean'),
                'Resource_Aware_Promoted_Score_Orders': int(resource_selection_diag.get('promoted_score_orders', 0) or 0),
                'Resource_Aware_Direct_Score_Order_Feasible': bool(resource_selection_diag.get('direct_score_order_feasible', False)),
                'Resource_Aware_Preservation_Required': bool(resource_selection_diag.get('resource_preservation_required', False)),
                'Resource_Aware_Selected_Count_Preserved': bool(resource_selection_diag.get('selected_count_preserved', True)),
                'Resource_Aware_Reserved_Capital_Preserved': bool(resource_selection_diag.get('reserved_capital_preserved', True)),
                'Resource_Aware_Pre_Market_Order_Limit': resource_selection_diag.get('pre_market_order_limit'),
                'Resource_Aware_Max_DL_Eligible': bool(resource_selection_diag.get('max_dl_eligible', False)),
                'Resource_Aware_Max_DL_Repair_Steps': int(resource_selection_diag.get('max_dl_repair_steps', 0) or 0),
                'Resource_Aware_Max_DL_Repair_Evaluations': int(resource_selection_diag.get('max_dl_repair_evaluations', 0) or 0),
                'Resource_Aware_Max_DL_Fallback': bool(resource_selection_diag.get('max_dl_fallback_to_baseline', False)),
                'Resource_Aware_Max_DL_Seed_Fallback': bool(resource_selection_diag.get('max_dl_seed_fallback', False)),
                'Resource_Aware_Max_DL_Feasible_Ascent_Steps': int(resource_selection_diag.get('max_dl_feasible_ascent_steps', 0) or 0),
                'Resource_Aware_Max_DL_Feasible_Ascent_Evaluations': int(resource_selection_diag.get('max_dl_feasible_ascent_evaluations', 0) or 0),
                'Resource_Aware_Max_DL_Feasible_Ascent_Local_Optimum': bool(resource_selection_diag.get('max_dl_feasible_ascent_local_optimum', False)),
                'Resource_Aware_Stale_Score_Guard_Enabled': bool(resource_selection_diag.get('stale_score_membership_guard_enabled', False)),
                'Resource_Aware_Stale_Score_Guard_Max_Age_Days': resource_selection_diag.get('stale_score_membership_guard_max_age_days'),
                'Resource_Aware_Stale_Score_Candidate_Count': int(resource_selection_diag.get('stale_score_candidate_count', 0) or 0),
                'Resource_Aware_Stale_Score_Guard_Triggered': bool(resource_selection_diag.get('stale_score_guard_triggered', False)),
                'Resource_Aware_Stale_Score_Guard_Seed_Blocked': bool(resource_selection_diag.get('stale_score_guard_seed_blocked', False)),
                'Resource_Aware_Stale_Score_Guard_Blocked_Swaps': int(resource_selection_diag.get('stale_score_guard_blocked_swaps', 0) or 0),
                'Resource_Aware_Selector_Elapsed_Ns': int(resource_selection_diag.get('selector_elapsed_ns', 0) or 0),
            })

        current_equity = today_equity
        current_equity_money = today_equity_money
        invested_capital = current_equity_money - cash_money
        exposure_pct = (invested_capital / current_equity_money) * 100 if current_equity_money > 0 else 0
        total_exposure += exposure_pct
        if exposure_pct > max_exp:
            max_exp = exposure_pct

        strategy_ret_pct = (current_equity_money - initial_capital) / initial_capital * 100

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

        if replay_counts is not None and before_trade_rows >= 0:
            for row in trade_history[before_trade_rows:]:
                ticker = str(row.get("Ticker", "")).strip()
                bucket = replay_counts.setdefault(
                    ticker,
                    {
                        "candidate_dates": [],
                        "orderable_dates": [],
                        "candidate_rows": [],
                        "orderable_rows": [],
                        "trade_rows": [],
                    },
                )
                bucket.setdefault("trade_rows", []).append(row)

        if profile_timing_enabled:
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
    )

    if profile_timing_enabled:
        closeout_sec = time.perf_counter() - t0

    final_cash = today_equity
    if last_date is not None and last_date.year in year_end_equity:
        year_end_equity[last_date.year] = milli_to_money(today_equity)
        last_month_key = (int(last_date.year), int(last_date.month))
        if last_month_key in month_end_equity:
            month_end_equity[last_month_key] = milli_to_money(today_equity)
        last_quarter_key = (int(last_date.year), int((last_date.month - 1) // 3 + 1))
        if last_quarter_key in quarter_end_equity:
            quarter_end_equity[last_quarter_key] = milli_to_money(today_equity)

    # # (AI註: 期末強制結算後補做一次 peak / drawdown 更新，避免 final closeout 對 MDD 漏算)
    if today_equity > peak_equity:
        peak_equity = today_equity
    drawdown = (peak_equity - today_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    if drawdown > max_drawdown:
        max_drawdown = drawdown

    if len(sorted_dates) > start_idx:
        monthly_equities.append(milli_to_money(today_equity))
        if benchmark_loop_enabled and current_bm_px is not None:
            bm_monthly_equities.append(current_bm_px)

    final_equity_money = milli_to_money(today_equity)
    total_return = (final_equity_money - initial_capital) / initial_capital * 100

    if ((not is_training) or capture_equity_curve) and len(equity_curve) > 0:
        equity_curve[-1]['Equity'] = final_equity_money
        equity_curve[-1]['Strategy_Return_Pct'] = total_return
        equity_curve[-1]['Invested_Amount'] = 0.0
        equity_curve[-1]['Exposure_Pct'] = 0.0

    t0 = time.perf_counter() if profile_timing_enabled else None
    r_squared, monthly_win_rate = calc_curve_stats(monthly_equities)
    if benchmark_period_stats is not None:
        bm_r_squared = float(benchmark_period_stats.get('bm_r_squared', 0.0))
        bm_monthly_win_rate = float(benchmark_period_stats.get('bm_monthly_win_rate', 0.0))
    else:
        bm_r_squared, bm_monthly_win_rate = calc_curve_stats(bm_monthly_equities)
    curve_stats_sec = (time.perf_counter() - t0) if profile_timing_enabled else 0.0

    trade_count = len(closed_trades_stats)
    portfolio_r_stats = summarize_closed_trade_r_stats(closed_trades_stats)
    entry_type_counts = summarize_closed_trade_entry_type_counts(closed_trades_stats)
    normal_trade_count = int(entry_type_counts.get('normal_trades', 0))
    extended_trade_count = int(entry_type_counts.get('extended_trades', 0))
    breakout_trade_count = int(entry_type_counts.get('breakout_trades', normal_trade_count))
    reentry_trade_count = int(entry_type_counts.get('reentry_trades', 0))
    score_single_stock_trade_stats = {}
    if profile_stats is not None and active_context_resolver is None and active_context_ensemble_resolver is None:
        score_single_stock_trade_stats = summarize_single_stock_trade_stats_from_pit_index(
            pit_stats_index,
            sorted_dates[start_idx:],
        )
    if trade_count > 0:
        wins = [t for t in closed_trades_stats if t['pnl'] > 0]
        losses = [t for t in closed_trades_stats if t['pnl'] <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / trade_count) * 100
        avg_win_amount = sum(t['pnl'] for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss_amount = abs(sum(t['pnl'] for t in losses) / loss_count) if loss_count > 0 else 0.0
        pf_payoff = (avg_win_amount / avg_loss_amount) if avg_loss_amount > 0 else (99.9 if avg_win_amount > 0 else 0.0)

        avg_win_r = sum(t['r_mult'] for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss_r = abs(sum(t['r_mult'] for t in losses) / loss_count) if loss_count > 0 else 0.0
        if get_ev_calc_method() == 'B':
            payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
            pf_ev = (win_rate / 100.0 * payoff_for_ev) - (1 - win_rate / 100.0)
        else:
            pf_ev = sum(t['r_mult'] for t in closed_trades_stats) / trade_count
    else:
        win_rate, pf_ev, pf_payoff = 0.0, 0.0, 0.0

    avg_exp = total_exposure / sim_days if sim_days > 0 else 0.0
    sim_years = calc_sim_years(sorted_dates, start_idx)
    annual_trades = (trade_count / sim_years) if sim_years > 0 else 0.0
    filled_buy_count = int(portfolio_entry_stats.get('filled_buy_count', 0) or 0)
    reserved_buy_fill_rate = (filled_buy_count / (filled_buy_count + total_missed_buys) * 100.0) if (filled_buy_count + total_missed_buys) > 0 else 0.0
    annual_return_pct = calc_annual_return_pct(initial_capital, final_equity_money, sim_years)

    bm_start_value = float(benchmark_start_price) if benchmark_start_price is not None else 0.0
    bm_end_value = bm_start_value * (1.0 + bm_ret_pct / 100.0) if bm_start_value > 0 else 0.0
    if benchmark_period_stats is not None:
        bm_annual_return_pct = float(benchmark_period_stats.get('bm_annual_return_pct', 0.0))
    else:
        bm_annual_return_pct = calc_annual_return_pct(bm_start_value, bm_end_value, sim_years)

    yearly_stats = build_full_year_return_stats(
        sorted_dates=sorted_dates,
        year_start_equity=year_start_equity,
        year_end_equity=year_end_equity,
        year_first_sim_date=year_first_sim_date,
        year_last_sim_date=year_last_sim_date,
    )
    monthly_stats = build_full_month_return_stats(
        sorted_dates=sorted_dates,
        month_start_equity=month_start_equity,
        month_end_equity=month_end_equity,
        month_first_sim_date=month_first_sim_date,
        month_last_sim_date=month_last_sim_date,
    )
    quarterly_stats = build_full_quarter_return_stats(
        sorted_dates=sorted_dates,
        quarter_start_equity=quarter_start_equity,
        quarter_end_equity=quarter_end_equity,
        quarter_first_sim_date=quarter_first_sim_date,
        quarter_last_sim_date=quarter_last_sim_date,
    )
    if benchmark_period_stats is not None:
        full_years = {int(row['year']) for row in yearly_stats['yearly_return_rows'] if row.get('is_full_year')}
        bm_yearly_rows = [
            row for row in benchmark_period_stats.get('bm_yearly_return_rows', [])
            if int(row.get('year', -1)) in full_years
        ]
        bm_yearly_stats = {
            'bm_full_year_count': int(len(bm_yearly_rows)),
            'bm_min_full_year_return_pct': float(min((row['year_return_pct'] for row in bm_yearly_rows), default=0.0)),
            'bm_yearly_return_rows': bm_yearly_rows,
        }
        full_months = {(int(row['year']), int(row['month'])) for row in monthly_stats['monthly_return_rows'] if row.get('is_full_month')}
        bm_monthly_rows = [
            row for row in benchmark_period_stats.get('bm_monthly_return_rows', [])
            if (int(row.get('year', -1)), int(row.get('month', -1))) in full_months
        ]
        bm_monthly_stats = {
            'bm_full_month_count': int(len(bm_monthly_rows)),
            'bm_min_month_return_pct': float(min((row['month_return_pct'] for row in bm_monthly_rows), default=0.0)),
            'bm_monthly_return_rows': bm_monthly_rows,
        }
        full_quarters = {(int(row['year']), int(row['quarter'])) for row in quarterly_stats['quarterly_return_rows'] if row.get('is_full_quarter')}
        bm_quarterly_rows = [
            row for row in benchmark_period_stats.get('bm_quarterly_return_rows', [])
            if (int(row.get('year', -1)), int(row.get('quarter', -1))) in full_quarters
        ]
        bm_quarterly_stats = {
            'bm_full_quarter_count': int(len(bm_quarterly_rows)),
            'bm_min_quarter_return_pct': float(min((row['quarter_return_pct'] for row in bm_quarterly_rows), default=0.0)),
            'bm_quarterly_return_rows': bm_quarterly_rows,
        }
    else:
        bm_yearly_stats = build_benchmark_full_year_return_stats(
            sorted_dates=sorted_dates,
            benchmark_data=benchmark_data,
            yearly_return_rows=yearly_stats['yearly_return_rows'],
        )
        bm_monthly_stats = build_benchmark_full_month_return_stats(
            sorted_dates=sorted_dates,
            benchmark_data=benchmark_data,
            monthly_return_rows=monthly_stats['monthly_return_rows'],
        )
        bm_quarterly_stats = build_benchmark_full_quarter_return_stats(
            sorted_dates=sorted_dates,
            benchmark_data=benchmark_data,
            quarterly_return_rows=quarterly_stats['quarterly_return_rows'],
        )

    if profile_stats is not None:
        profile_stats['portfolio_wall_sec'] = (time.perf_counter() - t_portfolio_start) if profile_timing_enabled else 0.0
        profile_stats['portfolio_ticker_dates_sec'] = ticker_dates_sec
        profile_stats['portfolio_build_trade_index_sec'] = build_trade_index_sec
        profile_stats['portfolio_day_loop_sec'] = day_loop_sec
        profile_stats['portfolio_candidate_scan_sec'] = candidate_scan_sec
        profile_stats['portfolio_rotation_sec'] = rotation_sec
        profile_stats['portfolio_settle_sec'] = settle_sec
        profile_stats['portfolio_buy_sec'] = buy_sec
        profile_stats['portfolio_equity_mark_sec'] = equity_mark_sec
        profile_stats['portfolio_closeout_sec'] = closeout_sec
        profile_stats['curve_stats_sec'] = curve_stats_sec
        profile_stats['idle_fast_path_days'] = int(idle_fast_path_days)
        profile_stats['dominant_year_dependency_diagnostics'] = build_dominant_year_dependency_diagnostics(closed_trades_stats)
        profile_stats['sim_years'] = sim_years
        profile_stats['annual_return_pct'] = annual_return_pct
        profile_stats['bm_annual_return_pct'] = bm_annual_return_pct
        profile_stats['reserved_buy_fill_rate'] = reserved_buy_fill_rate
        profile_stats['portfolio_total_r'] = float(portfolio_r_stats.get('total_r', 0.0))
        profile_stats['portfolio_median_r'] = float(portfolio_r_stats.get('median_r', 0.0))
        profile_stats['portfolio_avg_r'] = float(portfolio_r_stats.get('avg_r', 0.0))
        if not is_training:
            # # (AI註: 一筆一列的扣費後 round-trip 稽核資料；只供報表歸因，不改交易或統計口徑。)
            profile_stats['closed_trade_rows'] = [dict(row) for row in closed_trades_stats]
        profile_stats['breakout_trades'] = int(breakout_trade_count)
        profile_stats['reentry_trades'] = int(reentry_trade_count)
        profile_stats['normal_trades'] = int(normal_trade_count)
        profile_stats['extended_trades'] = int(extended_trade_count)
        if score_single_stock_trade_stats:
            profile_stats.update(build_score_single_stock_profile_fields(score_single_stock_trade_stats))
        profile_stats['filled_buy_count'] = filled_buy_count
        if active_level_rows is not None:
            profile_stats['portfolio_active_level_rows'] = list(active_level_rows)
        if capacity_rows is not None:
            profile_stats['portfolio_capacity_rows'] = list(capacity_rows)
        if capture_equity_curve:
            profile_stats['equity_curve'] = list(equity_curve)
        profile_stats.update(yearly_stats)
        profile_stats.update(monthly_stats)
        profile_stats.update(quarterly_stats)
        profile_stats.update(bm_yearly_stats)
        profile_stats.update(bm_monthly_stats)
        profile_stats.update(bm_quarterly_stats)

    if is_training:
        return total_return, max_drawdown, trade_count, final_equity_money, avg_exp, max_exp, bm_ret_pct, bm_max_drawdown, win_rate, pf_ev, pf_payoff, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct

    df_equity = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trade_history)
    final_bm_return = df_equity.iloc[-1][f"Benchmark_{benchmark_ticker}_Pct"] if not df_equity.empty else 0.0
    return df_equity, df_trades, total_return, max_drawdown, trade_count, win_rate, pf_ev, pf_payoff, final_equity_money, avg_exp, max_exp, final_bm_return, bm_max_drawdown, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct

