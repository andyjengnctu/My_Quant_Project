from core.buy_sort import (
    calc_buy_sort_value,
    calc_projected_capital_metrics,
    sort_candidate_rows,
)
from core.config import get_buy_sort_method
from core.exact_accounting import build_buy_ledger_from_price, milli_to_money
from filters.breakout_quality.runtime import (
    get_breakout_quality_ranking_source_context,
    resolve_breakout_quality_candidate_rank,
)

from core.trade_plans import (
    build_extended_candidate_plan_from_signal,
    clone_shadow_position,
    build_normal_candidate_plan,
    create_signal_tracking_state,
    is_extended_signal_orderable_for_day,
    resolve_signal_tracking_params,
)
from core.extended_signals import (
    attach_breakout_quality_rank,
    normalize_breakout_quality_rank_payload,
    resolve_breakout_quality_rank,
)
from core.portfolio_fast_data import (
    get_fast_close,
    get_fast_dates,
    get_fast_pos,
    get_fast_security_profile,
    get_fast_value,
    get_pit_stats_from_index,
)


def _resolve_candidate_quality_ranking(*, params, ticker, signal_date, signal_state=None):
    enabled = bool(getattr(params, "use_breakout_quality_ranking", False))
    if not enabled:
        return None

    filter_id = str(getattr(params, "breakout_quality_filter_id"))
    inherited_rank = resolve_breakout_quality_rank(signal_state)
    if inherited_rank is not None:
        inherited_filter_id = str(inherited_rank.get("filter_id") or "").strip()
        if inherited_filter_id and inherited_filter_id != filter_id:
            raise ValueError(
                "continuation／re-entry breakout quality filter_id 與當前參數不一致: "
                f"ticker={ticker}, inherited={inherited_filter_id}, current={filter_id}"
            )
        active_source = get_breakout_quality_ranking_source_context().score_source
        inherited_source = str(inherited_rank.get("score_source") or "canonical_runtime").strip()
        if inherited_source != active_source:
            raise ValueError(
                "continuation／re-entry breakout quality score source 與當前 replay 不一致: "
                f"ticker={ticker}, inherited={inherited_source}, current={active_source}"
            )
        inherited_rank["filter_id"] = filter_id
        inherited_rank["score_source"] = active_source
        return inherited_rank

    if signal_date is None:
        raise ValueError(f"breakout quality ranking 候選缺少原始 signal_date: ticker={ticker}")
    return resolve_breakout_quality_candidate_rank(
        ticker=str(ticker),
        signal_date=signal_date,
        high_len=int(getattr(params, "high_len")),
        filter_id=filter_id,
    )


def _make_candidate_row(
    *,
    buy_sort_method,
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
    signal_date=None,
    sizing_capital=None,
    shadow_position_state=None,
    max_qty=None,
    prev_close=None,
    quality_rank=None,
):
    if est_qty > 0:
        est_ledger = build_buy_ledger_from_price(est_limit_px, est_qty, params)
        est_cost_milli = int(est_ledger["net_buy_total_milli"])
        est_cost = milli_to_money(est_cost_milli)
    else:
        est_cost_milli = 0
        est_cost = 0.0
    buy_limit_overage_pct = calc_buy_sort_value(
        'BUY_LIMIT_OVERAGE_THEN_PROJ_COST',
        ev,
        est_cost,
        win_rate,
        trade_count,
        asset_growth_pct,
        prev_close=prev_close,
        limit_price=est_limit_px,
    )
    sort_value = calc_buy_sort_value(
        buy_sort_method,
        ev,
        est_cost,
        win_rate,
        trade_count,
        asset_growth_pct,
        prev_close=prev_close,
        limit_price=est_limit_px,
    )
    normalized_quality_rank = (
        None
        if quality_rank is None
        else normalize_breakout_quality_rank_payload(quality_rank)
    )
    ranking_context = get_breakout_quality_ranking_source_context()
    max_position_cap_pct = float(getattr(params, "max_position_cap_pct"))
    projected_capital_fraction, projected_capital_deployment_rate = (
        calc_projected_capital_metrics(
            proj_cost=est_cost,
            sizing_capital=sizing_capital,
            max_position_cap_pct=max_position_cap_pct,
        )
    )
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
        'buy_limit_overage_pct': buy_limit_overage_pct,
        'prev_close': prev_close,
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
        'candidate_date': trade_date,
        'signal_date': signal_date,
        'sizing_capital': sizing_capital,
        'max_position_cap_pct': max_position_cap_pct,
        'projected_capital_fraction': projected_capital_fraction,
        'projected_capital_deployment_rate': projected_capital_deployment_rate,
        'params_obj': params,
        'max_qty': max_qty,
        'orig_limit': (signal_state or {}).get('orig_limit') if signal_state is not None else est_limit_px,
        'orig_atr': (signal_state or {}).get('orig_atr') if signal_state is not None else entry_atr,
        'entry_source': (signal_state or {}).get('source') if signal_state is not None else candidate_type,
        'use_breakout_quality_ranking': bool(normalized_quality_rank is not None),
        'breakout_quality_ranking_policy': str(ranking_context.ranking_policy),
        'breakout_quality_score': (
            None
            if normalized_quality_rank is None
            or not bool(normalized_quality_rank.get('available', False))
            else normalized_quality_rank['score']
        ),
        'breakout_quality_score_date': (
            '' if normalized_quality_rank is None
            else str(normalized_quality_rank.get('score_date') or '')
        ),
        'breakout_quality_score_source': (
            '' if normalized_quality_rank is None
            else str(normalized_quality_rank.get('score_source') or '')
        ),
        'breakout_quality_rank': (
            None if normalized_quality_rank is None else dict(normalized_quality_rank)
        ),
    }
    if signal_state is not None:
        row['signal_state'] = signal_state
    resolved_shadow_position = shadow_position_state
    if resolved_shadow_position is None and signal_state is not None:
        resolved_shadow_position = (signal_state or {}).get('shadow_position')
    if resolved_shadow_position is not None:
        row['shadow_position_state'] = clone_shadow_position(resolved_shadow_position)
    return row


def _collect_normal_candidates(
    *,
    normal_setup_entries,
    portfolio,
    sold_today,
    all_dfs_fast,
    active_extended_signals,
    pit_stats_index,
    pit_stats_cursor,
    buy_sort_method,
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

        # # (AI註: 新的 normal setup 必須先覆蓋舊延續訊號；即使當日因已持有/剛賣出而不能下單，
        # #        也不能讓舊 setup 的延續訊號跨日殘留。若 history filter 不合格，禁止沿用舊 setup 續掛)
        active_extended_signals.pop(ticker, None)

        if ticker in portfolio or ticker in sold_today:
            continue

        fast_df = all_dfs_fast[ticker]
        y_buy_limit = get_fast_value(fast_df, 'buy_limit', pos=y_pos)
        y_atr = get_fast_value(fast_df, 'ATR', pos=y_pos)
        y_close = get_fast_close(fast_df, pos=y_pos)
        signal_date = get_fast_dates(fast_df)[y_pos]
        security_profile = get_fast_security_profile(fast_df)

        is_candidate, ev, win_rate, trade_count, asset_growth_pct = get_pit_stats_from_index(
            pit_stats_index[ticker], today, params, cursor_state=pit_stats_cursor, ticker=ticker
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

        quality_rank = _resolve_candidate_quality_ranking(
            params=params, ticker=ticker, signal_date=signal_date
        )
        signal_state = create_signal_tracking_state(
            y_buy_limit,
            y_atr,
            params,
            ticker=ticker,
            security_profile=security_profile,
            signal_date=signal_date,
        )
        if signal_state is not None:
            attach_breakout_quality_rank(signal_state, quality_rank)
            active_extended_signals[ticker] = signal_state

        if (not collect_all_candidates) and (not bool(candidate_plan['is_orderable'])):
            continue

        candidate_row = _make_candidate_row(
            buy_sort_method=buy_sort_method,
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
            signal_date=signal_date,
            sizing_capital=candidate_plan.get('sizing_capital'),
            prev_close=y_close,
            quality_rank=quality_rank,
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
    pit_stats_cursor,
    today,
    params,
):
    normal_setup_tickers_today = set()
    for ticker, y_pos, _t_pos in sorted(normal_setup_entries, key=lambda x: x[0]):
        normal_setup_tickers_today.add(ticker)

        # # (AI註: 與完整候選路徑一致：新的 normal setup 先覆蓋舊延續訊號；即使當日因已持有/剛賣出而不能下單，
        # #        也不能讓舊 setup 的延續訊號跨日殘留。若 history filter 不合格則不續掛)
        active_extended_signals.pop(ticker, None)

        if ticker in portfolio or ticker in sold_today:
            continue

        fast_df = all_dfs_fast[ticker]
        y_buy_limit = get_fast_value(fast_df, 'buy_limit', pos=y_pos)
        y_atr = get_fast_value(fast_df, 'ATR', pos=y_pos)
        signal_date = get_fast_dates(fast_df)[y_pos]
        security_profile = get_fast_security_profile(fast_df)

        is_candidate, _ev, _win_rate, _trade_count, _asset_growth_pct = get_pit_stats_from_index(
            pit_stats_index[ticker], today, params, cursor_state=pit_stats_cursor, ticker=ticker
        )
        if not is_candidate:
            continue

        quality_rank = _resolve_candidate_quality_ranking(
            params=params, ticker=ticker, signal_date=signal_date
        )
        signal_state = create_signal_tracking_state(
            y_buy_limit,
            y_atr,
            params,
            ticker=ticker,
            security_profile=security_profile,
            signal_date=signal_date,
        )
        if signal_state is not None:
            attach_breakout_quality_rank(signal_state, quality_rank)
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
    pit_stats_cursor,
    buy_sort_method,
    today,
    sizing_equity,
    params,
    collect_all_candidates,
):
    candidates_today = [] if collect_all_candidates else None
    orderable_candidates_today = []

    for ticker in sorted(list(active_extended_signals.keys())):
        signal_state = active_extended_signals.get(ticker)
        candidate_params = resolve_signal_tracking_params(signal_state, params)

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
            pit_stats_index[ticker], today, candidate_params, cursor_state=pit_stats_cursor, ticker=ticker
        )
        if not is_candidate:
            continue

        y_close = get_fast_close(fast_df, pos=y_pos)
        security_profile = get_fast_security_profile(fast_df)
        candidate_plan = build_extended_candidate_plan_from_signal(
            signal_state,
            sizing_equity,
            candidate_params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=today,
        )
        if candidate_plan is None:
            continue

        quality_rank = _resolve_candidate_quality_ranking(
            params=candidate_params,
            ticker=ticker,
            signal_date=candidate_plan.get("signal_date"),
            signal_state=signal_state,
        )
        today_orderable = is_extended_signal_orderable_for_day(
            signal_state,
            candidate_plan,
            y_close,
            ticker=ticker,
        )

        if (not collect_all_candidates) and (not bool(today_orderable)):
            continue

        candidate_row = _make_candidate_row(
            buy_sort_method=buy_sort_method,
            ticker=ticker,
            candidate_type=str((signal_state or {}).get('source') or 'extended'),
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
            params=candidate_params,
            security_profile=candidate_plan.get('security_profile'),
            trade_date=today,
            signal_date=candidate_plan.get('signal_date'),
            sizing_capital=candidate_plan.get('sizing_capital'),
            signal_state=signal_state,
            continuation_invalidation_barrier=candidate_plan.get('continuation_invalidation_barrier'),
            continuation_completion_barrier=candidate_plan.get('continuation_completion_barrier'),
            entry_ref_price=candidate_plan.get('entry_ref_price'),
            shadow_position_state=candidate_plan.get('shadow_position_state'),
            max_qty=candidate_plan.get('max_qty'),
            prev_close=y_close,
            quality_rank=quality_rank,
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
    pit_stats_cursor=None,
    today,
    sizing_equity,
    params,
    collect_all_candidates=True,
):
    buy_sort_method = get_buy_sort_method()
    normal_candidates, normal_orderable, normal_setup_tickers_today = _collect_normal_candidates(
        normal_setup_entries=normal_setup_index.get(today, []),
        portfolio=portfolio,
        sold_today=sold_today,
        all_dfs_fast=all_dfs_fast,
        active_extended_signals=active_extended_signals,
        pit_stats_index=pit_stats_index,
        pit_stats_cursor=pit_stats_cursor,
        buy_sort_method=buy_sort_method,
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
        pit_stats_cursor=pit_stats_cursor,
        buy_sort_method=buy_sort_method,
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
        sort_candidate_rows(candidates_today, buy_sort_method)
    sort_candidate_rows(orderable_candidates_today, buy_sort_method)
    return candidates_today, orderable_candidates_today, normal_setup_tickers_today
