from core.buy_sort import calc_buy_sort_value
from core.config import get_buy_sort_method
from core.exact_accounting import build_buy_ledger_from_price, milli_to_money
from core.trade_plans import (
    build_extended_candidate_plan_from_signal,
    build_normal_candidate_plan,
    create_signal_tracking_state,
    is_extended_signal_orderable_for_day,
)
from core.portfolio_fast_data import (
    get_fast_close,
    get_fast_pos,
    get_fast_security_profile,
    get_fast_value,
    get_pit_stats_from_index,
)


def _make_candidate_row(
    *,
    ticker,
    candidate_type,
    est_limit_px,
    ev,
    y_atr,
    t_pos,
    y_pos,
    est_qty,
    win_rate,
    trade_count,
    asset_growth_pct,
    est_init_sl,
    est_init_trail,
    est_target_price,
    entry_atr,
    is_orderable,
    params,
    signal_state=None,
    continuation_invalidation_barrier=None,
    continuation_completion_barrier=None,
    entry_ref_price=None,
    security_profile=None,
    trade_date=None,
    sizing_capital=None,
):
    if est_qty > 0:
        est_ledger = build_buy_ledger_from_price(est_limit_px, est_qty, params)
        est_cost_milli = int(est_ledger["net_buy_total_milli"])
        est_cost = milli_to_money(est_cost_milli)
    else:
        est_cost_milli = 0
        est_cost = 0.0
    sort_value = calc_buy_sort_value(get_buy_sort_method(), ev, est_cost, win_rate, trade_count, asset_growth_pct)
    row = {
        'ticker': ticker,
        'type': candidate_type,
        'limit_px': est_limit_px,
        'ev': ev,
        'y_atr': y_atr,
        'today_pos': t_pos,
        'yesterday_pos': y_pos,
        'qty': est_qty,
        'proj_cost': est_cost,
        'proj_cost_milli': est_cost_milli,
        'sort_value': sort_value,
        'hist_win_rate': win_rate,
        'hist_trade_count': trade_count,
        'asset_growth_pct': asset_growth_pct,
        'init_sl': est_init_sl,
        'init_trail': est_init_trail,
        'target_price': est_target_price,
        'entry_atr': entry_atr,
        'is_orderable': is_orderable,
        'continuation_invalidation_barrier': continuation_invalidation_barrier,
        'continuation_completion_barrier': continuation_completion_barrier,
        'entry_ref_price': entry_ref_price,
        'security_profile': security_profile,
        'trade_date': trade_date,
        'sizing_capital': sizing_capital,
    }
    if signal_state is not None:
        row['signal_state'] = signal_state
    return row


def _collect_normal_candidates(
    *,
    normal_setup_entries,
    portfolio,
    sold_today,
    all_dfs_fast,
    active_extended_signals,
    pit_stats_index,
    today,
    sizing_equity,
    params,
    collect_all_candidates,
):
    candidates_today = [] if collect_all_candidates else None
    orderable_candidates_today = []
    normal_setup_tickers_today = set()

    for ticker, y_pos, t_pos in sorted(normal_setup_entries, key=lambda x: x[0]):
        normal_setup_tickers_today.add(ticker)
        if ticker in portfolio or ticker in sold_today:
            continue

        # # (AI註: 新的 normal setup 必須先覆蓋舊延續訊號；若當日 history filter 不合格，禁止沿用舊 setup 續掛)
        active_extended_signals.pop(ticker, None)

        fast_df = all_dfs_fast[ticker]
        y_buy_limit = get_fast_value(fast_df, 'buy_limit', pos=y_pos)
        y_atr = get_fast_value(fast_df, 'ATR', pos=y_pos)
        security_profile = get_fast_security_profile(fast_df)

        is_candidate, ev, win_rate, trade_count, asset_growth_pct = get_pit_stats_from_index(
            pit_stats_index[ticker], today, params
        )
        if not is_candidate:
            continue

        candidate_plan = build_normal_candidate_plan(
            y_buy_limit,
            y_atr,
            sizing_equity,
            params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=today,
        )
        if candidate_plan is None:
            continue

        signal_state = create_signal_tracking_state(
            y_buy_limit,
            y_atr,
            params,
            ticker=ticker,
            security_profile=security_profile,
        )
        if signal_state is not None:
            active_extended_signals[ticker] = signal_state

        if (not collect_all_candidates) and (not bool(candidate_plan['is_orderable'])):
            continue

        candidate_row = _make_candidate_row(
            ticker=ticker,
            candidate_type='normal',
            est_limit_px=candidate_plan['limit_price'],
            ev=ev,
            y_atr=y_atr,
            t_pos=t_pos,
            y_pos=y_pos,
            est_qty=candidate_plan['qty'],
            win_rate=win_rate,
            trade_count=trade_count,
            asset_growth_pct=asset_growth_pct,
            est_init_sl=candidate_plan['init_sl'],
            est_init_trail=candidate_plan['init_trail'],
            est_target_price=candidate_plan['target_price'],
            entry_atr=candidate_plan['entry_atr'],
            is_orderable=candidate_plan['is_orderable'],
            params=params,
            security_profile=candidate_plan.get('security_profile'),
            trade_date=today,
            sizing_capital=candidate_plan.get('sizing_capital'),
        )
        if candidates_today is not None:
            candidates_today.append(candidate_row)
        if candidate_row['is_orderable']:
            orderable_candidates_today.append(candidate_row)

    return candidates_today, orderable_candidates_today, normal_setup_tickers_today


# # (AI註: 滿倉且關閉 rotation 時，portfolio 仍需承接新 setup 為延續狀態；但不需要建立可買候選、排序或預估投入成本)
def track_normal_setup_signals_for_day(
    *,
    normal_setup_entries,
    portfolio,
    sold_today,
    all_dfs_fast,
    active_extended_signals,
    pit_stats_index,
    today,
    params,
):
    normal_setup_tickers_today = set()
    for ticker, y_pos, _t_pos in sorted(normal_setup_entries, key=lambda x: x[0]):
        normal_setup_tickers_today.add(ticker)
        if ticker in portfolio or ticker in sold_today:
            continue

        # # (AI註: 與完整候選路徑一致：新的 normal setup 先覆蓋舊延續訊號；若 history filter 不合格則不續掛)
        active_extended_signals.pop(ticker, None)

        fast_df = all_dfs_fast[ticker]
        y_buy_limit = get_fast_value(fast_df, 'buy_limit', pos=y_pos)
        y_atr = get_fast_value(fast_df, 'ATR', pos=y_pos)
        security_profile = get_fast_security_profile(fast_df)

        is_candidate, _ev, _win_rate, _trade_count, _asset_growth_pct = get_pit_stats_from_index(
            pit_stats_index[ticker], today, params
        )
        if not is_candidate:
            continue

        signal_state = create_signal_tracking_state(
            y_buy_limit,
            y_atr,
            params,
            ticker=ticker,
            security_profile=security_profile,
        )
        if signal_state is not None:
            active_extended_signals[ticker] = signal_state
    return normal_setup_tickers_today

def _collect_extended_candidates(
    *,
    active_extended_signals,
    portfolio,
    sold_today,
    normal_setup_tickers_today,
    all_dfs_fast,
    pit_stats_index,
    today,
    sizing_equity,
    params,
    collect_all_candidates,
):
    candidates_today = [] if collect_all_candidates else None
    orderable_candidates_today = []

    for ticker in sorted(list(active_extended_signals.keys())):
        if ticker in portfolio or ticker in sold_today or ticker in normal_setup_tickers_today:
            continue

        fast_df = all_dfs_fast.get(ticker)
        if fast_df is None:
            continue

        t_pos = get_fast_pos(fast_df, today)
        if t_pos <= 0:
            continue
        y_pos = t_pos - 1

        is_candidate, ev, win_rate, trade_count, asset_growth_pct = get_pit_stats_from_index(
            pit_stats_index[ticker], today, params
        )
        if not is_candidate:
            continue

        y_close = get_fast_close(fast_df, pos=y_pos)
        security_profile = get_fast_security_profile(fast_df)
        candidate_plan = build_extended_candidate_plan_from_signal(
            active_extended_signals[ticker],
            sizing_equity,
            params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=today,
        )
        if candidate_plan is None:
            continue

        today_orderable = is_extended_signal_orderable_for_day(
            active_extended_signals[ticker],
            candidate_plan,
            y_close,
            ticker=ticker,
        )

        if (not collect_all_candidates) and (not bool(today_orderable)):
            continue

        candidate_row = _make_candidate_row(
            ticker=ticker,
            candidate_type='extended',
            est_limit_px=candidate_plan['limit_price'],
            ev=ev,
            y_atr=candidate_plan['orig_atr'],
            t_pos=t_pos,
            y_pos=y_pos,
            est_qty=candidate_plan['qty'],
            win_rate=win_rate,
            trade_count=trade_count,
            asset_growth_pct=asset_growth_pct,
            est_init_sl=candidate_plan['init_sl'],
            est_init_trail=candidate_plan['init_trail'],
            est_target_price=candidate_plan['target_price'],
            entry_atr=candidate_plan['entry_atr'],
            is_orderable=today_orderable,
            params=params,
            security_profile=candidate_plan.get('security_profile'),
            trade_date=today,
            sizing_capital=candidate_plan.get('sizing_capital'),
            signal_state=active_extended_signals[ticker],
            continuation_invalidation_barrier=candidate_plan.get('continuation_invalidation_barrier'),
            continuation_completion_barrier=candidate_plan.get('continuation_completion_barrier'),
            entry_ref_price=candidate_plan.get('entry_ref_price'),
        )
        if candidates_today is not None:
            candidates_today.append(candidate_row)
        if candidate_row['is_orderable']:
            orderable_candidates_today.append(candidate_row)

    return candidates_today, orderable_candidates_today


def build_daily_candidates(
    *,
    normal_setup_index,
    active_extended_signals,
    portfolio,
    sold_today,
    all_dfs_fast,
    pit_stats_index,
    today,
    sizing_equity,
    params,
    collect_all_candidates=True,
):
    normal_candidates, normal_orderable, normal_setup_tickers_today = _collect_normal_candidates(
        normal_setup_entries=normal_setup_index.get(today, []),
        portfolio=portfolio,
        sold_today=sold_today,
        all_dfs_fast=all_dfs_fast,
        active_extended_signals=active_extended_signals,
        pit_stats_index=pit_stats_index,
        today=today,
        sizing_equity=sizing_equity,
        params=params,
        collect_all_candidates=collect_all_candidates,
    )
    extended_candidates, extended_orderable = _collect_extended_candidates(
        active_extended_signals=active_extended_signals,
        portfolio=portfolio,
        sold_today=sold_today,
        normal_setup_tickers_today=normal_setup_tickers_today,
        all_dfs_fast=all_dfs_fast,
        pit_stats_index=pit_stats_index,
        today=today,
        sizing_equity=sizing_equity,
        params=params,
        collect_all_candidates=collect_all_candidates,
    )

    if collect_all_candidates:
        candidates_today = normal_candidates + extended_candidates
    else:
        candidates_today = []
    orderable_candidates_today = normal_orderable + extended_orderable
    if collect_all_candidates:
        candidates_today.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
    orderable_candidates_today.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
    return candidates_today, orderable_candidates_today, normal_setup_tickers_today
