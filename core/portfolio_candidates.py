import time

from core.buy_sort import (
    BUY_LIMIT_OVERAGE_SORT_METHOD,
    ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD,
    calc_buy_limit_overage_pct,
    calc_buy_sort_value,
    calc_projected_capital_metrics,
    sort_candidate_rows,
)
from core.config import get_buy_sort_method
from core.exact_accounting import (
    calc_cash_buy_total_milli_from_milli,
    calc_cash_buy_total_milli_from_price,
    milli_to_money,
    price_to_milli,
)
from core.breakout_quality_ranking_context import (
    breakout_quality_ranking_uses_daily_information_date,
    get_breakout_quality_ranking_source_context,
)
from core.breakout_quality_runtime_port import resolve_breakout_quality_candidate_rank

from core.trade_plans import (
    build_extended_candidate_plan_from_signal,
    calc_candidate_qty_for_capital,
    clone_shadow_position,
    build_normal_candidate_plan,
    resolve_normal_candidate_plan_geometry,
    create_signal_tracking_state,
    has_extended_signal_orderable_context_for_day,
    has_extended_signal_orderable_context_for_limit,
    is_extended_signal_orderable_for_day,
    resize_candidate_plan_to_capital,
    resolve_extended_candidate_plan_geometry,
    resolve_extended_candidate_plan_geometry_training_fast,
)
from core.extended_signals import (
    attach_breakout_quality_rank,
    normalize_breakout_quality_rank_payload,
    resolve_breakout_quality_rank,
)
from core.order_lot_policy import TAIWAN_BOARD_LOT_SHARES
from core.price_utils import (
    _NATIVE_BROKER_FEE_PPM,
    _NATIVE_BROKER_MIN_FEE_MILLI,
    build_position_sizing_capital_context,
    build_position_sizing_fee_context,
)
from core.native_optimizer_hotpath import (
    resolve_live_extended_candidate_core as _native_resolve_live_extended_candidate_core,
    resolve_live_extended_candidate_batch_sized as _native_resolve_live_extended_candidate_batch_sized,
)
from core.training_performance import is_optimizer_native_hotpath_kernels_enabled

_NATIVE_EXTENDED_CANDIDATE_CORE_ENABLED = bool(
    _native_resolve_live_extended_candidate_core is not None
    and is_optimizer_native_hotpath_kernels_enabled()
)
_NATIVE_EXTENDED_CANDIDATE_BATCH_SIZED_ENABLED = bool(
    _native_resolve_live_extended_candidate_batch_sized is not None
    and is_optimizer_native_hotpath_kernels_enabled()
)
from core.portfolio_fast_data import (
    get_fast_extended_day_market_values,
    get_fast_security_profile,
    get_fast_normal_signal_values,
    get_pit_stats_from_index,
)


def _candidate_profile_add(profile_stats, key, started_at):
    profile_stats[key] = float(profile_stats.get(key, 0.0)) + max(
        0.0,
        time.perf_counter() - started_at,
    )


def _resolve_candidate_sizing_fee_context(
    cache,
    *,
    params,
    ticker,
    security_profile,
    trade_date,
):
    profile = security_profile if isinstance(security_profile, dict) else {}
    family = str(profile.get("family") or "")
    broad_type = str(profile.get("broad_type") or "")
    fallback_symbol = "" if profile else str(ticker or "")
    # Fee/tax semantics are date-independent for every supported family except
    # bond ETFs.  Keep the cache reusable across the whole replay for stocks and
    # normal funds, while preserving exact day-specific semantics for bond ETFs.
    fee_date_key = (
        str(trade_date)[:10]
        if family == "etf" and broad_type == "bond"
        else ""
    )
    key = (id(params), family, broad_type, fallback_symbol, fee_date_key)
    cached = cache.get(key)
    if cached is not None:
        cached_params, resolved = cached
        if cached_params is params:
            return resolved
    resolved = build_position_sizing_fee_context(
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
    )
    # Retain the params object alongside the value so an integer id key can never
    # be accidentally reused by a later short-lived runtime params object.
    cache[key] = (params, resolved)
    return resolved


def _resolve_candidate_qty_cached(
    cache,
    *,
    limit_price,
    init_sl,
    sizing_equity,
    params,
    ticker,
    security_profile,
    trade_date,
    max_qty,
    sizing_capital_context,
    sizing_fee_context,
    limit_price_milli,
    stop_price_milli,
):
    if cache is None:
        return calc_candidate_qty_for_capital(
            limit_price, init_sl, sizing_equity, params,
            ticker=ticker, security_profile=security_profile, trade_date=trade_date,
            max_qty=max_qty, sizing_capital_context=sizing_capital_context,
            sizing_fee_context=sizing_fee_context,
            limit_price_milli=limit_price_milli, stop_price_milli=stop_price_milli,
        )

    profile = security_profile if isinstance(security_profile, dict) else {}
    family = str(profile.get("family") or "")
    broad_type = str(profile.get("broad_type") or "")
    try:
        normalized_max_qty = None if max_qty is None else int(max_qty)
    except (TypeError, ValueError):
        normalized_max_qty = 0
    key = (
        str(ticker or ""),
        family,
        broad_type,
        int(limit_price_milli),
        int(stop_price_milli),
        tuple(sizing_capital_context),
        tuple(sizing_fee_context),
        normalized_max_qty,
    )
    cached_qty = cache.get(key)
    if cached_qty is not None:
        return int(cached_qty)
    qty = calc_candidate_qty_for_capital(
        limit_price, init_sl, sizing_equity, params,
        ticker=ticker, security_profile=security_profile, trade_date=trade_date,
        max_qty=max_qty, sizing_capital_context=sizing_capital_context,
        sizing_fee_context=sizing_fee_context,
        limit_price_milli=limit_price_milli, stop_price_milli=stop_price_milli,
    )
    cache[key] = int(qty)
    return int(qty)


def _resolve_candidate_quality_ranking(
    *,
    params,
    ticker,
    signal_date,
    information_date=None,
    signal_state=None,
):
    enabled = bool(getattr(params, "use_breakout_quality_ranking", False))
    if not enabled:
        return None

    filter_id = str(getattr(params, "breakout_quality_filter_id"))
    daily_refresh = breakout_quality_ranking_uses_daily_information_date()
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
        if not daily_refresh:
            inherited_rank["filter_id"] = filter_id
            inherited_rank["score_source"] = active_source
            return inherited_rank

    if signal_date is None:
        raise ValueError(f"breakout quality ranking 候選缺少原始 signal_date: ticker={ticker}")
    if daily_refresh and information_date is None:
        raise ValueError(
            f"daily breakout quality ranking 候選缺少最新已完成交易日: ticker={ticker}"
        )
    return resolve_breakout_quality_candidate_rank(
        ticker=str(ticker),
        signal_date=signal_date,
        information_date=information_date,
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
    shadow_position_state_is_snapshot=False,
    max_qty=None,
    prev_close=None,
    quality_rank=None,
    ranking_context=None,
    est_limit_px_milli=None,
    precomputed_proj_cost_milli=None,
    precomputed_sort_value=None,
    compact_training_row=False,
):
    if precomputed_proj_cost_milli is not None:
        est_cost_milli = int(precomputed_proj_cost_milli)
        est_cost = milli_to_money(est_cost_milli)
    elif est_qty > 0:
        if est_limit_px_milli is None:
            est_cost_milli = int(calc_cash_buy_total_milli_from_price(est_limit_px, est_qty))
        else:
            est_cost_milli = int(
                calc_cash_buy_total_milli_from_milli(est_limit_px_milli, est_qty)
            )
        est_cost = milli_to_money(est_cost_milli)
    else:
        est_cost_milli = 0
        est_cost = 0.0
    buy_limit_overage_pct = (
        float(precomputed_sort_value)
        if precomputed_sort_value is not None
        and buy_sort_method in {BUY_LIMIT_OVERAGE_SORT_METHOD, ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD}
        else calc_buy_limit_overage_pct(prev_close, est_limit_px)
    )
    if precomputed_sort_value is not None:
        sort_value = float(precomputed_sort_value)
    elif buy_sort_method in {BUY_LIMIT_OVERAGE_SORT_METHOD, ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD}:
        sort_value = buy_limit_overage_pct
    else:
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
    if compact_training_row:
        # Training replay only needs execution/sort state here.  Diagnostic and
        # ranking payloads are preserved by the full-row path below.
        row = {
            'ticker': ticker,
            'type': candidate_type,
            'limit_px': est_limit_px,
            'today_pos': t_pos,
            'yesterday_pos': y_pos,
            'qty': est_qty,
            'proj_cost': est_cost,
            'proj_cost_milli': est_cost_milli,
            'sort_value': sort_value,
            'init_sl': est_init_sl,
            'init_trail': est_init_trail,
            'target_price': est_target_price,
            'entry_atr': entry_atr,
            'is_orderable': is_orderable,
            'security_profile': security_profile,
            'trade_date': trade_date,
            'candidate_date': trade_date,
            'signal_date': signal_date,
            'sizing_capital': sizing_capital,
            'params_obj': params,
            'max_qty': max_qty,
            'orig_limit': (signal_state or {}).get('orig_limit') if signal_state is not None else est_limit_px,
            'orig_atr': (signal_state or {}).get('orig_atr') if signal_state is not None else entry_atr,
            'entry_source': (signal_state or {}).get('source') if signal_state is not None else candidate_type,
            'use_breakout_quality_ranking': False,
        }
    else:
        normalized_quality_rank = (
            None
            if quality_rank is None
            else normalize_breakout_quality_rank_payload(quality_rank)
        )
        resolved_ranking_context = (
            get_breakout_quality_ranking_source_context()
            if ranking_context is None
            else ranking_context
        )
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
            'breakout_quality_ranking_policy': str(resolved_ranking_context.ranking_policy),
            'breakout_quality_ranking_options': dict(resolved_ranking_context.ranking_options or {}),
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
            'breakout_quality_safety_score': (
                None
                if normalized_quality_rank is None
                or not bool(normalized_quality_rank.get('safety_available', False))
                else normalized_quality_rank.get('safety_score')
            ),
            'breakout_quality_safety_score_available': bool(
                normalized_quality_rank is not None
                and normalized_quality_rank.get('safety_available', False)
            ),
            'breakout_quality_safety_score_date': (
                '' if normalized_quality_rank is None
                else str(normalized_quality_rank.get('safety_score_date') or '')
            ),
            'breakout_quality_safety_score_source': (
                '' if normalized_quality_rank is None
                else str(normalized_quality_rank.get('safety_score_source') or '')
            ),
            'breakout_quality_rank': (
                None if normalized_quality_rank is None else dict(normalized_quality_rank)
            ),
        }
    if signal_state is not None:
        row['signal_state'] = signal_state
    resolved_shadow_position = shadow_position_state
    resolved_shadow_is_snapshot = bool(
        shadow_position_state is not None and shadow_position_state_is_snapshot
    )
    if resolved_shadow_position is None and signal_state is not None:
        resolved_shadow_position = (signal_state or {}).get('shadow_position')
    if resolved_shadow_position is not None:
        # AI註: extended candidate_plan already owns an isolated shadow snapshot.
        # Transfer that snapshot into the ephemeral row instead of cloning it twice.
        # A fallback read from signal_state is still cloned because it is live state.
        row['shadow_position_state'] = (
            resolved_shadow_position
            if compact_training_row or resolved_shadow_is_snapshot
            else clone_shadow_position(resolved_shadow_position)
        )
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
    sizing_capital_context,
    sizing_fee_context_cache,
    params,
    collect_all_candidates,
    ranking_context,
    compact_training_rows=False,
    profile_stats=None,
):
    profile_enabled = profile_stats is not None
    total_started = time.perf_counter() if profile_enabled else None
    candidates_today = [] if collect_all_candidates else None
    orderable_candidates_today = []
    normal_setup_tickers_today = set()

    for ticker, y_pos, t_pos in normal_setup_entries:
        normal_setup_tickers_today.add(ticker)

        # # (AI註: 新的 normal setup 必須先覆蓋舊延續訊號；即使當日因已持有/剛賣出而不能下單，
        # #        也不能讓舊 setup 的延續訊號跨日殘留。若 history filter 不合格，禁止沿用舊 setup 續掛)
        active_extended_signals.pop(ticker, None)

        if ticker in portfolio or ticker in sold_today:
            continue

        fast_df = all_dfs_fast[ticker]
        y_buy_limit, y_atr, y_close, signal_date = get_fast_normal_signal_values(
            fast_df, y_pos
        )
        security_profile = get_fast_security_profile(fast_df)

        pit_started = time.perf_counter() if profile_enabled else None
        is_candidate, ev, win_rate, trade_count, asset_growth_pct = get_pit_stats_from_index(
            pit_stats_index[ticker], today, params, cursor_state=pit_stats_cursor, ticker=ticker
        )
        if profile_enabled:
            _candidate_profile_add(profile_stats, "portfolio_candidate_pit_stats_sec", pit_started)
        if not is_candidate:
            continue

        sizing_fee_context = _resolve_candidate_sizing_fee_context(
            sizing_fee_context_cache, params=params, ticker=ticker,
            security_profile=security_profile, trade_date=today,
        )
        plan_started = time.perf_counter() if profile_enabled else None
        if collect_all_candidates:
            candidate_plan = build_normal_candidate_plan(
                y_buy_limit, y_atr, sizing_equity, params, ticker=ticker,
                security_profile=security_profile, trade_date=today,
                sizing_capital_context=sizing_capital_context,
                sizing_fee_context=sizing_fee_context,
            )
            if candidate_plan is None:
                if profile_enabled:
                    _candidate_profile_add(profile_stats, "portfolio_candidate_plan_sec", plan_started)
                continue
            limit_price = candidate_plan['limit_price']
            init_sl = candidate_plan['init_sl']
            init_trail = candidate_plan['init_trail']
            target_price = candidate_plan['target_price']
            entry_atr = candidate_plan['entry_atr']
            qty = int(candidate_plan.get('qty', 0) or 0)
            is_orderable = bool(candidate_plan.get('is_orderable'))
            sizing_capital = candidate_plan.get('sizing_capital')
            limit_price_milli = price_to_milli(limit_price)
        else:
            geometry = resolve_normal_candidate_plan_geometry(
                y_buy_limit, y_atr, params, ticker=ticker, security_profile=security_profile
            )
            if geometry is None:
                if profile_enabled:
                    _candidate_profile_add(profile_stats, "portfolio_candidate_plan_sec", plan_started)
                continue
            limit_price, init_sl, init_trail, target_price, entry_atr = geometry
            limit_price_milli = price_to_milli(limit_price)
            qty = calc_candidate_qty_for_capital(
                limit_price, init_sl, sizing_equity, params, ticker=ticker,
                security_profile=security_profile, trade_date=today,
                sizing_capital_context=sizing_capital_context,
                sizing_fee_context=sizing_fee_context,
                limit_price_milli=limit_price_milli,
                stop_price_milli=price_to_milli(init_sl),
            )
            is_orderable = qty > 0
            sizing_capital = sizing_equity
            candidate_plan = None
        if profile_enabled:
            _candidate_profile_add(profile_stats, "portfolio_candidate_plan_sec", plan_started)

        if bool(getattr(params, "use_breakout_quality_ranking", False)):
            quality_started = time.perf_counter() if profile_enabled else None
            quality_rank = _resolve_candidate_quality_ranking(
                params=params, ticker=ticker, signal_date=signal_date,
                information_date=signal_date,
            )
            if profile_enabled:
                _candidate_profile_add(profile_stats, "portfolio_candidate_quality_rank_sec", quality_started)
        else:
            quality_rank = None
        signal_state = create_signal_tracking_state(
            y_buy_limit, y_atr, params, ticker=ticker,
            security_profile=security_profile, signal_date=signal_date,
        )
        if signal_state is not None:
            attach_breakout_quality_rank(signal_state, quality_rank)
            active_extended_signals[ticker] = signal_state

        if (not collect_all_candidates) and (not is_orderable):
            continue

        row_started = time.perf_counter() if profile_enabled else None
        candidate_row = _make_candidate_row(
            buy_sort_method=buy_sort_method, ticker=ticker, candidate_type='normal',
            est_limit_px=limit_price, ev=ev, y_atr=y_atr, t_pos=t_pos, y_pos=y_pos,
            est_qty=qty, win_rate=win_rate, trade_count=trade_count,
            asset_growth_pct=asset_growth_pct, est_init_sl=init_sl,
            est_init_trail=init_trail, est_target_price=target_price, entry_atr=entry_atr,
            is_orderable=is_orderable, params=params, security_profile=security_profile,
            trade_date=today, signal_date=signal_date, sizing_capital=sizing_capital,
            prev_close=y_close, quality_rank=quality_rank, ranking_context=ranking_context,
            est_limit_px_milli=limit_price_milli,
            compact_training_row=bool(compact_training_rows),
        )
        if profile_enabled:
            _candidate_profile_add(profile_stats, "portfolio_candidate_row_sec", row_started)
        if candidates_today is not None:
            candidates_today.append(candidate_row)
        if candidate_row['is_orderable']:
            orderable_candidates_today.append(candidate_row)

    if profile_enabled:
        _candidate_profile_add(profile_stats, "portfolio_candidate_normal_sec", total_started)
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
    profile_stats=None,
):
    profile_enabled = profile_stats is not None
    total_started = time.perf_counter() if profile_enabled else None
    normal_setup_tickers_today = set()
    for ticker, y_pos, _t_pos in normal_setup_entries:
        normal_setup_tickers_today.add(ticker)

        # # (AI註: 與完整候選路徑一致：新的 normal setup 先覆蓋舊延續訊號；即使當日因已持有/剛賣出而不能下單，
        # #        也不能讓舊 setup 的延續訊號跨日殘留。若 history filter 不合格則不續掛)
        active_extended_signals.pop(ticker, None)

        if ticker in portfolio or ticker in sold_today:
            continue

        fast_df = all_dfs_fast[ticker]
        y_buy_limit, y_atr, _y_close, signal_date = get_fast_normal_signal_values(
            fast_df, y_pos
        )
        security_profile = get_fast_security_profile(fast_df)

        pit_started = time.perf_counter() if profile_enabled else None
        is_candidate, _ev, _win_rate, _trade_count, _asset_growth_pct = get_pit_stats_from_index(
            pit_stats_index[ticker], today, params, cursor_state=pit_stats_cursor, ticker=ticker
        )
        if profile_enabled:
            _candidate_profile_add(profile_stats, "portfolio_candidate_pit_stats_sec", pit_started)
        if not is_candidate:
            continue

        if bool(getattr(params, "use_breakout_quality_ranking", False)):
            quality_started = time.perf_counter() if profile_enabled else None
            quality_rank = _resolve_candidate_quality_ranking(
                params=params, ticker=ticker, signal_date=signal_date,
                information_date=signal_date,
            )
            if profile_enabled:
                _candidate_profile_add(profile_stats, "portfolio_candidate_quality_rank_sec", quality_started)
        else:
            quality_rank = None
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
    if profile_enabled:
        _candidate_profile_add(profile_stats, "portfolio_candidate_normal_sec", total_started)
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
    sizing_capital_context,
    sizing_capital_context_cache,
    sizing_fee_context_cache,
    sizing_qty_cache,
    params,
    collect_all_candidates,
    ranking_context,
    extended_day_market_cache=None,
    compact_training_rows=False,
    profile_stats=None,
):
    profile_enabled = profile_stats is not None
    total_started = time.perf_counter() if profile_enabled else None
    candidates_today = [] if collect_all_candidates else None
    orderable_candidates_today = []
    native_sized_batch_items = []
    native_sized_batch_meta = []

    sort_started = time.perf_counter() if profile_enabled else None
    # The final executable candidate list is sorted canonically below.  Training
    # without diagnostic rows has no observable dependency on the traversal order
    # of independent continuation states, so preserve the dict's deterministic
    # insertion order and avoid re-sorting the active set every simulated day.
    sorted_extended_tickers = (
        tuple(active_extended_signals)
        if compact_training_rows
        else tuple(sorted(active_extended_signals))
    )
    if extended_day_market_cache is not None:
        extended_day_market_cache[("__meta__", "sorted_extended_tickers")] = sorted_extended_tickers
    if profile_enabled:
        _candidate_profile_add(profile_stats, "portfolio_candidate_sort_sec", sort_started)
    for ticker in sorted_extended_tickers:
        signal_state = active_extended_signals[ticker]
        candidate_params = signal_state.get("_params_obj") or params

        if ticker in portfolio or ticker in sold_today or ticker in normal_setup_tickers_today:
            continue

        fast_df = all_dfs_fast.get(ticker)
        if fast_df is None:
            continue

        cached_day_market = (
            None if extended_day_market_cache is None
            else extended_day_market_cache.get(ticker)
        )
        if cached_day_market is None:
            cached_day_market = get_fast_extended_day_market_values(fast_df, today)
            if extended_day_market_cache is not None:
                extended_day_market_cache[ticker] = cached_day_market
        t_pos = int(cached_day_market[0])
        if t_pos <= 0:
            continue
        y_pos = t_pos - 1

        pit_started = time.perf_counter() if profile_enabled else None
        is_candidate, ev, win_rate, trade_count, asset_growth_pct = get_pit_stats_from_index(
            pit_stats_index[ticker], today, candidate_params, cursor_state=pit_stats_cursor, ticker=ticker
        )
        if profile_enabled:
            _candidate_profile_add(profile_stats, "portfolio_candidate_pit_stats_sec", pit_started)
        if not is_candidate:
            continue

        previous_values = cached_day_market[1]
        information_date = cached_day_market[2]
        security_profile = cached_day_market[3]
        y_close = previous_values[2]

        if (
            not collect_all_candidates
            and compact_training_rows
            and _NATIVE_EXTENDED_CANDIDATE_BATCH_SIZED_ENABLED
            and not bool(getattr(candidate_params, "use_breakout_quality_ranking", False))
        ):
            live_shadow = signal_state.get("shadow_position")
            tick_profile = (security_profile or {}).get("tick_profile") if isinstance(security_profile, dict) else None
            if (
                isinstance(live_shadow, dict)
                and int(live_shadow.get("qty", 0) or 0) > 0
                and tick_profile in {"stock_six_tier", "fund_two_tier"}
            ):
                plan_started = time.perf_counter() if profile_enabled else None
                candidate_capital_context = sizing_capital_context_cache.get(id(candidate_params))
                if candidate_capital_context is None:
                    candidate_capital_context = build_position_sizing_capital_context(
                        sizing_equity, candidate_params.fixed_risk, candidate_params,
                    )
                    sizing_capital_context_cache[id(candidate_params)] = candidate_capital_context
                sizing_fee_context = _resolve_candidate_sizing_fee_context(
                    sizing_fee_context_cache,
                    params=candidate_params,
                    ticker=ticker,
                    security_profile=security_profile,
                    trade_date=today,
                )
                if int(sizing_fee_context[4]) >= 0:
                    native_sized_batch_items.append((
                        ticker, signal_state, live_shadow, float(y_close), security_profile,
                        candidate_capital_context, sizing_fee_context,
                    ))
                    native_sized_batch_meta.append((
                        ticker, candidate_params, t_pos, y_pos, ev, win_rate, trade_count,
                        asset_growth_pct, y_close, security_profile, signal_state,
                    ))
                    if profile_enabled:
                        _candidate_profile_add(profile_stats, "portfolio_candidate_plan_sec", plan_started)
                    continue

        if not collect_all_candidates:
            plan_started = time.perf_counter() if profile_enabled else None
            ranking_enabled_for_candidate = bool(
                getattr(candidate_params, "use_breakout_quality_ranking", False)
            )
            native_core = None
            if (
                compact_training_rows
                and _NATIVE_EXTENDED_CANDIDATE_CORE_ENABLED
                and not ranking_enabled_for_candidate
            ):
                live_shadow = signal_state.get("shadow_position")
                if isinstance(live_shadow, dict) and int(live_shadow.get("qty", 0) or 0) > 0:
                    native_core = _native_resolve_live_extended_candidate_core(
                        signal_state,
                        live_shadow,
                        float(y_close),
                        ticker,
                        security_profile,
                    )

            if native_core is False:
                if profile_enabled:
                    _candidate_profile_add(
                        profile_stats, "portfolio_candidate_plan_sec", plan_started
                    )
                continue

            if native_core is not None:
                (
                    limit_price, init_sl, init_trail, target_price, entry_atr,
                    entry_ref_price, signal_date, max_qty, entry_source,
                    _source_entry_order_id, _shadow_position, limit_price_milli,
                    stop_price_milli, y_close_milli,
                ) = native_core
                orig_limit = limit_price
                quality_rank = None
                orderable_context = True
            else:
                geometry = (
                    resolve_extended_candidate_plan_geometry_training_fast(
                        signal_state,
                        candidate_params,
                        ticker=ticker,
                        security_profile=security_profile,
                    )
                    if compact_training_rows
                    else resolve_extended_candidate_plan_geometry(
                        signal_state,
                        candidate_params,
                        ticker=ticker,
                        security_profile=security_profile,
                    )
                )
                if geometry is None:
                    if profile_enabled:
                        _candidate_profile_add(
                            profile_stats, "portfolio_candidate_plan_sec", plan_started
                        )
                    continue
                (
                    limit_price, init_sl, init_trail, target_price, entry_atr,
                    orig_limit, entry_ref_price, _resolved_ticker, _resolved_profile,
                    signal_date, max_qty, entry_source, _source_entry_order_id,
                    _shadow_position,
                ) = geometry

                if ranking_enabled_for_candidate:
                    quality_started = time.perf_counter() if profile_enabled else None
                    quality_rank = _resolve_candidate_quality_ranking(
                        params=candidate_params, ticker=ticker, signal_date=signal_date,
                        information_date=information_date, signal_state=signal_state,
                    )
                    if profile_enabled:
                        _candidate_profile_add(
                            profile_stats,
                            "portfolio_candidate_quality_rank_sec",
                            quality_started,
                        )
                else:
                    quality_rank = None

                limit_price_milli = price_to_milli(limit_price)
                stop_price_milli = price_to_milli(init_sl)
                y_close_milli = price_to_milli(y_close)
                orderable_context = has_extended_signal_orderable_context_for_limit(
                    signal_state, limit_price, y_close, ticker=ticker,
                    security_profile=security_profile,
                    limit_price_milli=limit_price_milli, y_close_milli=y_close_milli,
                )
                if not bool(orderable_context):
                    if profile_enabled:
                        _candidate_profile_add(
                            profile_stats, "portfolio_candidate_plan_sec", plan_started
                        )
                    continue

            if profile_enabled:
                _candidate_profile_add(
                    profile_stats, "portfolio_candidate_plan_sec", plan_started
                )

            resize_started = time.perf_counter() if profile_enabled else None
            candidate_capital_context = sizing_capital_context_cache.get(id(candidate_params))
            if candidate_capital_context is None:
                candidate_capital_context = build_position_sizing_capital_context(
                    sizing_equity,
                    candidate_params.fixed_risk,
                    candidate_params,
                )
                sizing_capital_context_cache[id(candidate_params)] = candidate_capital_context
            sizing_fee_context = _resolve_candidate_sizing_fee_context(
                sizing_fee_context_cache,
                params=candidate_params,
                ticker=ticker,
                security_profile=security_profile,
                trade_date=today,
            )
            qty = _resolve_candidate_qty_cached(
                sizing_qty_cache,
                limit_price=limit_price, init_sl=init_sl, sizing_equity=sizing_equity,
                params=candidate_params, ticker=ticker, security_profile=security_profile,
                trade_date=today, max_qty=max_qty,
                sizing_capital_context=candidate_capital_context,
                sizing_fee_context=sizing_fee_context,
                limit_price_milli=limit_price_milli,
                stop_price_milli=stop_price_milli,
            )
            if profile_enabled:
                _candidate_profile_add(profile_stats, "portfolio_candidate_plan_sec", resize_started)
            if qty <= 0:
                continue
            today_orderable = True
            candidate_plan = None
        else:
            # Diagnostic/inspection mode preserves the complete plan payload.
            plan_started = time.perf_counter() if profile_enabled else None
            candidate_plan = build_extended_candidate_plan_from_signal(
                signal_state, sizing_equity, candidate_params, ticker=ticker,
                security_profile=security_profile, trade_date=today,
                resize_to_capital=False, copy_shadow_position=False,
            )
            if profile_enabled:
                _candidate_profile_add(profile_stats, "portfolio_candidate_plan_sec", plan_started)
            if candidate_plan is None:
                continue

            if bool(getattr(candidate_params, "use_breakout_quality_ranking", False)):
                quality_started = time.perf_counter() if profile_enabled else None
                quality_rank = _resolve_candidate_quality_ranking(
                    params=candidate_params, ticker=ticker,
                    signal_date=candidate_plan.get("signal_date"),
                    information_date=information_date, signal_state=signal_state,
                )
                if profile_enabled:
                    _candidate_profile_add(profile_stats, "portfolio_candidate_quality_rank_sec", quality_started)
            else:
                quality_rank = None

            limit_price_milli = price_to_milli(candidate_plan['limit_price'])
            stop_price_milli = price_to_milli(candidate_plan['init_sl'])
            y_close_milli = price_to_milli(y_close)
            orderable_context = has_extended_signal_orderable_context_for_day(
                signal_state, candidate_plan, y_close, ticker=ticker,
                security_profile=security_profile,
                limit_price_milli=limit_price_milli, y_close_milli=y_close_milli,
            )
            resize_started = time.perf_counter() if profile_enabled else None
            candidate_capital_context = sizing_capital_context_cache.get(id(candidate_params))
            if candidate_capital_context is None:
                candidate_capital_context = build_position_sizing_capital_context(
                    sizing_equity,
                    candidate_params.fixed_risk,
                    candidate_params,
                )
                sizing_capital_context_cache[id(candidate_params)] = candidate_capital_context
            sizing_fee_context = _resolve_candidate_sizing_fee_context(
                sizing_fee_context_cache,
                params=candidate_params,
                ticker=ticker,
                security_profile=security_profile,
                trade_date=today,
            )
            candidate_plan = resize_candidate_plan_to_capital(
                candidate_plan, sizing_equity, candidate_params,
                sizing_capital_context=candidate_capital_context,
                sizing_fee_context=sizing_fee_context,
                limit_price_milli=limit_price_milli,
                stop_price_milli=stop_price_milli, copy_plan=False,
            )
            if profile_enabled:
                _candidate_profile_add(profile_stats, "portfolio_candidate_plan_sec", resize_started)
            if candidate_plan is None:
                continue
            today_orderable = bool(orderable_context and candidate_plan.get("qty", 0) > 0)
            qty = candidate_plan.get("qty", 0)
            limit_price = candidate_plan['limit_price']
            init_sl = candidate_plan['init_sl']
            init_trail = candidate_plan['init_trail']
            target_price = candidate_plan['target_price']
            entry_atr = candidate_plan['entry_atr']
            orig_limit = candidate_plan.get('orig_limit')
            entry_ref_price = candidate_plan.get('entry_ref_price')
            signal_date = candidate_plan.get('signal_date')
            max_qty = candidate_plan.get('max_qty')
            entry_source = str((signal_state or {}).get('source') or 'extended')

        row_started = time.perf_counter() if profile_enabled else None
        candidate_row = _make_candidate_row(
            buy_sort_method=buy_sort_method,
            ticker=ticker,
            candidate_type=entry_source,
            est_limit_px=limit_price,
            ev=ev,
            y_atr=entry_atr,
            t_pos=t_pos,
            y_pos=y_pos,
            est_qty=qty,
            win_rate=win_rate,
            trade_count=trade_count,
            asset_growth_pct=asset_growth_pct,
            est_init_sl=init_sl,
            est_init_trail=init_trail,
            est_target_price=target_price,
            entry_atr=entry_atr,
            is_orderable=today_orderable,
            params=candidate_params,
            security_profile=security_profile,
            trade_date=today,
            signal_date=signal_date,
            sizing_capital=sizing_equity,
            signal_state=signal_state,
            continuation_invalidation_barrier=init_sl,
            continuation_completion_barrier=target_price,
            entry_ref_price=entry_ref_price,
            shadow_position_state=(None if candidate_plan is None else candidate_plan.get('shadow_position_state')),
            shadow_position_state_is_snapshot=(candidate_plan is not None and candidate_plan.get('shadow_position_state') is not None),
            max_qty=max_qty,
            prev_close=y_close,
            quality_rank=quality_rank,
            ranking_context=ranking_context,
            est_limit_px_milli=limit_price_milli,
            compact_training_row=bool(compact_training_rows),
        )
        if profile_enabled:
            _candidate_profile_add(profile_stats, "portfolio_candidate_row_sec", row_started)
        if candidates_today is not None:
            candidates_today.append(candidate_row)
        if candidate_row['is_orderable']:
            orderable_candidates_today.append(candidate_row)

    if native_sized_batch_items:
        plan_started = time.perf_counter() if profile_enabled else None
        native_results = _native_resolve_live_extended_candidate_batch_sized(
            native_sized_batch_items,
            int(_NATIVE_BROKER_FEE_PPM),
            int(_NATIVE_BROKER_MIN_FEE_MILLI),
        )
        if profile_enabled:
            _candidate_profile_add(profile_stats, "portfolio_candidate_plan_sec", plan_started)
        for meta, native_result in zip(native_sized_batch_meta, native_results):
            (
                ticker, candidate_params, t_pos, y_pos, ev, win_rate, trade_count,
                asset_growth_pct, y_close, security_profile, signal_state,
            ) = meta
            if native_result is False:
                # The native kernel treats NT$20 as the minimum at *every*
                # quantity.  False may therefore be an invalid odd-lot sizing
                # conclusion rather than a genuinely unorderable signal.
                # The Python branch confirms both the signal and exact fee.
                native_result = None
            if isinstance(native_result, tuple) and 0 < int(native_result[10]) < TAIWAN_BOARD_LOT_SHARES:
                # The compiled batch kernel uses a board-lot commission floor.
                # Its sub-board-lot qty and precomputed cash cost are not legal
                # for Yuanta odd-lot fees; recompute through the single Python
                # fee/quantity owner instead of trusting the stale native cost.
                native_result = None
            if native_result is None:
                # Unsupported edge cases remain owned by the canonical Python path.
                geometry = resolve_extended_candidate_plan_geometry_training_fast(
                    signal_state,
                    candidate_params,
                    ticker=ticker,
                    security_profile=security_profile,
                )
                if geometry is None:
                    continue
                (
                    limit_price, init_sl, init_trail, target_price, entry_atr,
                    _orig_limit, _entry_ref_price, _resolved_ticker, _resolved_profile,
                    signal_date, max_qty, entry_source, _source_entry_order_id,
                    _shadow_position,
                ) = geometry
                limit_price_milli = price_to_milli(limit_price)
                stop_price_milli = price_to_milli(init_sl)
                y_close_milli = price_to_milli(y_close)
                if not has_extended_signal_orderable_context_for_limit(
                    signal_state, limit_price, y_close, ticker=ticker,
                    security_profile=security_profile,
                    limit_price_milli=limit_price_milli, y_close_milli=y_close_milli,
                ):
                    continue
                candidate_capital_context = sizing_capital_context_cache.get(id(candidate_params))
                if candidate_capital_context is None:
                    candidate_capital_context = build_position_sizing_capital_context(
                        sizing_equity, candidate_params.fixed_risk, candidate_params,
                    )
                    sizing_capital_context_cache[id(candidate_params)] = candidate_capital_context
                sizing_fee_context = _resolve_candidate_sizing_fee_context(
                    sizing_fee_context_cache, params=candidate_params, ticker=ticker,
                    security_profile=security_profile, trade_date=today,
                )
                qty = _resolve_candidate_qty_cached(
                    sizing_qty_cache, limit_price=limit_price, init_sl=init_sl,
                    sizing_equity=sizing_equity, params=candidate_params, ticker=ticker,
                    security_profile=security_profile, trade_date=today, max_qty=max_qty,
                    sizing_capital_context=candidate_capital_context,
                    sizing_fee_context=sizing_fee_context,
                    limit_price_milli=limit_price_milli,
                    stop_price_milli=stop_price_milli,
                )
                if qty <= 0:
                    continue
                proj_cost_milli = None
                sort_value = None
            else:
                (
                    limit_price, init_sl, init_trail, target_price, entry_atr,
                    signal_date, max_qty, entry_source, limit_price_milli,
                    stop_price_milli, qty, proj_cost_milli, sort_value,
                ) = native_result

            row_started = time.perf_counter() if profile_enabled else None
            candidate_row = _make_candidate_row(
                buy_sort_method=buy_sort_method,
                ticker=ticker,
                candidate_type=entry_source,
                est_limit_px=limit_price,
                ev=ev,
                y_atr=entry_atr,
                t_pos=t_pos,
                y_pos=y_pos,
                est_qty=qty,
                win_rate=win_rate,
                trade_count=trade_count,
                asset_growth_pct=asset_growth_pct,
                est_init_sl=init_sl,
                est_init_trail=init_trail,
                est_target_price=target_price,
                entry_atr=entry_atr,
                is_orderable=True,
                params=candidate_params,
                security_profile=security_profile,
                trade_date=today,
                signal_date=signal_date,
                sizing_capital=sizing_equity,
                signal_state=signal_state,
                continuation_invalidation_barrier=init_sl,
                continuation_completion_barrier=target_price,
                max_qty=max_qty,
                prev_close=y_close,
                quality_rank=None,
                ranking_context=ranking_context,
                est_limit_px_milli=limit_price_milli,
                precomputed_proj_cost_milli=proj_cost_milli,
                precomputed_sort_value=sort_value,
                compact_training_row=True,
            )
            if profile_enabled:
                _candidate_profile_add(profile_stats, "portfolio_candidate_row_sec", row_started)
            orderable_candidates_today.append(candidate_row)

    if profile_enabled:
        _candidate_profile_add(profile_stats, "portfolio_candidate_extended_sec", total_started)
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
    extended_day_market_cache=None,
    compact_training_rows=False,
    persistent_sizing_fee_context_cache=None,
    persistent_sizing_qty_cache=None,
    profile_stats=None,
):
    buy_sort_method = get_buy_sort_method()
    ranking_enabled = bool(getattr(params, "use_breakout_quality_ranking", False))
    ranking_context = (
        None
        if compact_training_rows and not ranking_enabled
        else get_breakout_quality_ranking_source_context()
    )
    sizing_capital_context = build_position_sizing_capital_context(
        sizing_equity,
        params.fixed_risk,
        params,
    )
    sizing_capital_context_cache = {id(params): sizing_capital_context}
    sizing_fee_context_cache = (
        persistent_sizing_fee_context_cache
        if persistent_sizing_fee_context_cache is not None
        else {}
    )
    sizing_qty_cache = (
        persistent_sizing_qty_cache if compact_training_rows else None
    )
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
        sizing_capital_context=sizing_capital_context,
        sizing_fee_context_cache=sizing_fee_context_cache,
        params=params,
        collect_all_candidates=collect_all_candidates,
        ranking_context=ranking_context,
        compact_training_rows=compact_training_rows,
        profile_stats=profile_stats,
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
        sizing_capital_context=sizing_capital_context,
        sizing_capital_context_cache=sizing_capital_context_cache,
        sizing_fee_context_cache=sizing_fee_context_cache,
        sizing_qty_cache=sizing_qty_cache,
        params=params,
        collect_all_candidates=collect_all_candidates,
        ranking_context=ranking_context,
        extended_day_market_cache=extended_day_market_cache,
        compact_training_rows=compact_training_rows,
        profile_stats=profile_stats,
    )

    if collect_all_candidates:
        candidates_today = normal_candidates + extended_candidates
    else:
        candidates_today = []
    orderable_candidates_today = normal_orderable + extended_orderable
    profile_enabled = profile_stats is not None
    sort_started = time.perf_counter() if profile_enabled else None
    if collect_all_candidates:
        sort_candidate_rows(candidates_today, buy_sort_method)
    sort_candidate_rows(orderable_candidates_today, buy_sort_method)
    if extended_day_market_cache is not None:
        # Cleanup runs later in the same simulated day.  Reuse the candidate rows
        # that were already materialized instead of traversing the orderable list
        # again in the portfolio engine just to build a ticker lookup.
        extended_day_market_cache[("__meta__", "orderable_candidate_rows_by_ticker")] = {
            row.get("ticker"): row
            for row in orderable_candidates_today
            if row.get("ticker") is not None
        }
    if profile_enabled:
        _candidate_profile_add(profile_stats, "portfolio_candidate_sort_sec", sort_started)
    return candidates_today, orderable_candidates_today, normal_setup_tickers_today
