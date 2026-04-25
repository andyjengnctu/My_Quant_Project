import pandas as pd
import time
from core.exact_accounting import milli_to_money, money_to_milli
from core.capital_policy import resolve_portfolio_sizing_equity
from core.config import get_ev_calc_method
from core.portfolio_fast_data import (
    build_normal_setup_index,
    build_trade_stats_index,
    calc_mark_to_market_equity,
    get_fast_close,
    get_fast_close_on_or_before,
    has_fast_date,
    is_extended_entry_type,
)
from core.portfolio_stats import (
    build_benchmark_full_year_return_stats,
    build_full_year_return_stats,
    calc_annual_return_pct,
    calc_curve_stats,
    calc_sim_years,
    find_sim_start_idx,
)
from core.portfolio_stats import calc_portfolio_score
from core.portfolio_candidates import build_daily_candidates, track_normal_setup_signals_for_day
from core.portfolio_ops import (
    cleanup_extended_signals_for_day,
    closeout_open_positions,
    execute_reserved_entries_for_day,
    settle_portfolio_positions,
    try_rotate_weakest_position,
)

def run_portfolio_timeline(all_dfs_fast, all_standalone_logs, sorted_dates, start_year, params, max_positions, enable_rotation, benchmark_ticker="0050", benchmark_data=None, is_training=True, profile_stats=None, verbose=True, replay_counts=None, pit_stats_index=None):
    profile_timing_enabled = bool(profile_stats.get("_timing_enabled", True)) if profile_stats is not None else False
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

    t0 = time.perf_counter() if profile_timing_enabled else None
    start_idx = find_sim_start_idx(sorted_dates, start_year)
    if profile_timing_enabled:
        ticker_dates_sec = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_timing_enabled else None
    if pit_stats_index is None:
        pit_stats_index = {t: build_trade_stats_index(logs) for t, logs in all_standalone_logs.items()}
    else:
        pit_stats_index = dict(pit_stats_index)
        if all_standalone_logs:
            for ticker, logs in all_standalone_logs.items():
                if pit_stats_index.get(ticker) is None:
                    pit_stats_index[ticker] = build_trade_stats_index(logs)
    normal_setup_index = build_normal_setup_index(all_dfs_fast)
    if profile_timing_enabled:
        build_trade_index_sec = time.perf_counter() - t0

    initial_capital = params.initial_capital
    initial_capital_milli = money_to_milli(initial_capital)
    cash = initial_capital_milli
    portfolio = {}
    active_extended_signals = {}
    trade_history, equity_curve, closed_trades_stats = [], [], []
    normal_trade_count, extended_trade_count = 0, 0
    peak_equity, max_drawdown, current_equity = initial_capital_milli, 0.0, initial_capital_milli
    total_exposure, sim_days, total_missed_buys, total_missed_sells, max_exp = 0.0, 0, 0, 0, 0.0
    monthly_equities, bm_monthly_equities = [initial_capital], []
    yesterday_equity = initial_capital
    year_start_equity, year_end_equity = {}, {}
    year_first_sim_date, year_last_sim_date = {}, {}

    current_month = sorted_dates[start_idx].month if start_idx < len(sorted_dates) else 1

    current_bm_px = None
    benchmark_start_price = None
    bm_peak_price = None
    if benchmark_data and start_idx < len(sorted_dates):
        benchmark_start_price, benchmark_anchor_date = get_fast_close_on_or_before(benchmark_data, sorted_dates[start_idx])
        if benchmark_start_price is not None:
            bm_peak_price = benchmark_start_price
            bm_monthly_equities.append(benchmark_start_price)
            if benchmark_anchor_date == sorted_dates[start_idx]:
                current_bm_px = benchmark_start_price

    yesterday_bm_px, bm_max_drawdown, bm_ret_pct = current_bm_px, 0.0, 0.0

    if replay_counts is not None:
        for ticker, bucket in replay_counts.items():
            if not isinstance(bucket, dict):
                replay_counts[ticker] = {}
                bucket = replay_counts[ticker]
            bucket.setdefault("candidate_dates", [])
            bucket.setdefault("orderable_dates", [])
            bucket.setdefault("trade_rows", [])

    for i in range(start_idx, len(sorted_dates)):
        t_day_start = time.perf_counter() if profile_timing_enabled else None
        sim_days += 1
        today = sorted_dates[i]

        if today.year not in year_start_equity:
            year_start_equity[today.year] = milli_to_money(current_equity)
            year_first_sim_date[today.year] = pd.Timestamp(today)

        current_equity_money = milli_to_money(current_equity)
        cash_money = milli_to_money(cash)

        sold_today = set()
        normal_setup_entries_today = normal_setup_index.get(today, [])
        has_portfolio_work_today = bool(portfolio) or bool(active_extended_signals) or bool(normal_setup_entries_today)

        if verbose and (not is_training) and i % 20 == 0:
            exp = ((current_equity_money - cash_money) / current_equity_money) * 100 if current_equity_money > 0 else 0
            print(f"\033[90m⏳ 推進中: {today.strftime('%Y-%m')} | 資產: {current_equity_money:,.0f} | 水位: {exp:>5.1f}%...\033[0m", end="\r", flush=True)

        if has_portfolio_work_today:
            available_cash = cash
            sizing_equity = resolve_portfolio_sizing_equity(current_equity_money, initial_capital, params)
            pre_market_occupied = len(portfolio) + len(sold_today)
            no_entry_capacity_today = (
                bool(is_training)
                and replay_counts is None
                and (not bool(enable_rotation))
                and pre_market_occupied >= int(max_positions)
            )

            if no_entry_capacity_today:
                t0 = time.perf_counter() if profile_timing_enabled else None
                track_normal_setup_signals_for_day(
                    normal_setup_entries=normal_setup_entries_today,
                    portfolio=portfolio,
                    sold_today=sold_today,
                    all_dfs_fast=all_dfs_fast,
                    active_extended_signals=active_extended_signals,
                    pit_stats_index=pit_stats_index,
                    today=today,
                    params=params,
                )
                if profile_timing_enabled:
                    candidate_scan_sec += time.perf_counter() - t0
                before_trade_rows = -1
            else:
                t0 = time.perf_counter() if profile_timing_enabled else None
                candidates_today, orderable_candidates_today, normal_setup_tickers_today = build_daily_candidates(
                    normal_setup_index=normal_setup_index,
                    active_extended_signals=active_extended_signals,
                    portfolio=portfolio,
                    sold_today=sold_today,
                    all_dfs_fast=all_dfs_fast,
                    pit_stats_index=pit_stats_index,
                    today=today,
                    sizing_equity=sizing_equity,
                    params=params,
                    collect_all_candidates=replay_counts is not None,
                )
                if profile_timing_enabled:
                    candidate_scan_sec += time.perf_counter() - t0

                if replay_counts is not None:
                    for candidate in candidates_today:
                        ticker = str(candidate.get("ticker", ""))
                        bucket = replay_counts.get(ticker)
                        if bucket is not None:
                            bucket["candidate_dates"].append(today)
                    for candidate in orderable_candidates_today:
                        ticker = str(candidate.get("ticker", ""))
                        bucket = replay_counts.get(ticker)
                        if bucket is not None:
                            bucket["orderable_dates"].append(today)
                    before_trade_rows = len(trade_history)
                else:
                    before_trade_rows = -1

                t0 = time.perf_counter() if profile_timing_enabled else None
                cash, normal_trade_count, extended_trade_count = try_rotate_weakest_position(
                    portfolio=portfolio,
                    orderable_candidates_today=orderable_candidates_today,
                    max_positions=max_positions,
                    enable_rotation=enable_rotation,
                    sold_today=sold_today,
                    all_dfs_fast=all_dfs_fast,
                    today=today,
                    pit_stats_index=pit_stats_index,
                    params=params,
                    cash=cash,
                    closed_trades_stats=closed_trades_stats,
                    trade_history=trade_history,
                    is_training=is_training,
                    normal_trade_count=normal_trade_count,
                    extended_trade_count=extended_trade_count,
                )
                if profile_timing_enabled:
                    rotation_sec += time.perf_counter() - t0

            t0 = time.perf_counter() if profile_timing_enabled else None
            cash, total_missed_sells, normal_trade_count, extended_trade_count = settle_portfolio_positions(
                portfolio=portfolio,
                sold_today=sold_today,
                all_dfs_fast=all_dfs_fast,
                today=today,
                params=params,
                cash=cash,
                closed_trades_stats=closed_trades_stats,
                trade_history=trade_history,
                is_training=is_training,
                total_missed_sells=total_missed_sells,
                normal_trade_count=normal_trade_count,
                extended_trade_count=extended_trade_count,
            )
            if profile_timing_enabled:
                settle_sec += time.perf_counter() - t0

            if not no_entry_capacity_today:
                t0 = time.perf_counter() if profile_timing_enabled else None
                cash, total_missed_buys = execute_reserved_entries_for_day(
                    portfolio=portfolio,
                    active_extended_signals=active_extended_signals,
                    orderable_candidates_today=orderable_candidates_today,
                    sold_today=sold_today,
                    all_dfs_fast=all_dfs_fast,
                    today=today,
                    params=params,
                    cash=cash,
                    available_cash=available_cash,
                    max_positions=max_positions,
                    trade_history=trade_history,
                    is_training=is_training,
                    total_missed_buys=total_missed_buys,
                )
                if profile_timing_enabled:
                    buy_sec += time.perf_counter() - t0

            cleanup_extended_signals_for_day(
                active_extended_signals=active_extended_signals,
                portfolio=portfolio,
                all_dfs_fast=all_dfs_fast,
                today=today,
                params=params,
                sizing_capital=sizing_equity,
            )
        elif replay_counts is not None:
            before_trade_rows = len(trade_history)
        else:
            before_trade_rows = -1

        t0 = time.perf_counter() if profile_timing_enabled else None
        if portfolio:
            today_equity = calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params)
        else:
            today_equity = cash
        today_equity_money = milli_to_money(today_equity)
        cash_money = milli_to_money(cash)
        if profile_timing_enabled:
            equity_mark_sec += time.perf_counter() - t0

        current_equity = today_equity
        current_equity_money = today_equity_money
        invested_capital = current_equity_money - cash_money
        exposure_pct = (invested_capital / current_equity_money) * 100 if current_equity_money > 0 else 0
        total_exposure += exposure_pct
        if exposure_pct > max_exp:
            max_exp = exposure_pct

        strategy_ret_pct = (current_equity_money - initial_capital) / initial_capital * 100

        if benchmark_data and benchmark_start_price is not None and has_fast_date(benchmark_data, today):
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
            if yesterday_bm_px is not None:
                bm_monthly_equities.append(yesterday_bm_px)
            current_month = today.month

        yesterday_equity = current_equity_money
        yesterday_bm_px = current_bm_px
        year_end_equity[today.year] = current_equity_money
        year_last_sim_date[today.year] = pd.Timestamp(today)

        if not is_training:
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
                bucket = replay_counts.get(ticker)
                if bucket is not None:
                    bucket["trade_rows"].append(row)

        if profile_timing_enabled:
            day_loop_sec += time.perf_counter() - t_day_start

    # # (AI註: 月底權益應以期末強制結算後的真實 final equity 為準，故移到 closeout 後再 append)

    t0 = time.perf_counter() if profile_timing_enabled else None
    last_date = sorted_dates[-1] if len(sorted_dates) > 0 else None
    today_equity, normal_trade_count, extended_trade_count = closeout_open_positions(
        portfolio=portfolio,
        cash=cash,
        params=params,
        trade_history=trade_history,
        is_training=is_training,
        closed_trades_stats=closed_trades_stats,
        normal_trade_count=normal_trade_count,
        extended_trade_count=extended_trade_count,
        last_date=last_date,
    )

    if profile_timing_enabled:
        closeout_sec = time.perf_counter() - t0

    final_cash = today_equity
    if last_date is not None and last_date.year in year_end_equity:
        year_end_equity[last_date.year] = milli_to_money(today_equity)

    # # (AI註: 期末強制結算後補做一次 peak / drawdown 更新，避免 final closeout 對 MDD 漏算)
    if today_equity > peak_equity:
        peak_equity = today_equity
    drawdown = (peak_equity - today_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    if drawdown > max_drawdown:
        max_drawdown = drawdown

    if len(sorted_dates) > start_idx:
        monthly_equities.append(milli_to_money(today_equity))
        if current_bm_px is not None:
            bm_monthly_equities.append(current_bm_px)

    final_equity_money = milli_to_money(today_equity)
    total_return = (final_equity_money - initial_capital) / initial_capital * 100

    if not is_training and len(equity_curve) > 0:
        equity_curve[-1]['Equity'] = final_equity_money
        equity_curve[-1]['Strategy_Return_Pct'] = total_return
        equity_curve[-1]['Invested_Amount'] = 0.0
        equity_curve[-1]['Exposure_Pct'] = 0.0

    t0 = time.perf_counter() if profile_timing_enabled else None
    r_squared, monthly_win_rate = calc_curve_stats(monthly_equities)
    bm_r_squared, bm_monthly_win_rate = calc_curve_stats(bm_monthly_equities)
    curve_stats_sec = (time.perf_counter() - t0) if profile_timing_enabled else 0.0

    trade_count = len(closed_trades_stats)
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
    total_reserved_entries = normal_trade_count + extended_trade_count
    reserved_buy_fill_rate = (total_reserved_entries / (total_reserved_entries + total_missed_buys) * 100.0) if (total_reserved_entries + total_missed_buys) > 0 else 0.0
    annual_return_pct = calc_annual_return_pct(initial_capital, final_equity_money, sim_years)

    bm_start_value = float(benchmark_start_price) if benchmark_start_price is not None else 0.0
    bm_end_value = bm_start_value * (1.0 + bm_ret_pct / 100.0) if bm_start_value > 0 else 0.0
    bm_annual_return_pct = calc_annual_return_pct(bm_start_value, bm_end_value, sim_years)

    yearly_stats = build_full_year_return_stats(
        sorted_dates=sorted_dates,
        year_start_equity=year_start_equity,
        year_end_equity=year_end_equity,
        year_first_sim_date=year_first_sim_date,
        year_last_sim_date=year_last_sim_date,
    )
    bm_yearly_stats = build_benchmark_full_year_return_stats(
        sorted_dates=sorted_dates,
        benchmark_data=benchmark_data,
        yearly_return_rows=yearly_stats['yearly_return_rows'],
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
        profile_stats['sim_years'] = sim_years
        profile_stats['annual_return_pct'] = annual_return_pct
        profile_stats['bm_annual_return_pct'] = bm_annual_return_pct
        profile_stats['reserved_buy_fill_rate'] = reserved_buy_fill_rate
        profile_stats.update(yearly_stats)
        profile_stats.update(bm_yearly_stats)

    if is_training:
        return total_return, max_drawdown, trade_count, final_equity_money, avg_exp, max_exp, bm_ret_pct, bm_max_drawdown, win_rate, pf_ev, pf_payoff, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct

    df_equity = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trade_history)
    final_bm_return = df_equity.iloc[-1][f"Benchmark_{benchmark_ticker}_Pct"] if not df_equity.empty else 0.0
    return df_equity, df_trades, total_return, max_drawdown, trade_count, win_rate, pf_ev, pf_payoff, final_equity_money, avg_exp, max_exp, final_bm_return, bm_max_drawdown, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct
