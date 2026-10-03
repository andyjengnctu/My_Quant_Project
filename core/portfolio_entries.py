from core.fee_rebate import accrue_fee_rebate
from core.exact_accounting import (
    coerce_money_like_to_milli,
    milli_to_money,
    restore_money_like_from_milli,
)
from core.portfolio_entry_plans import (
    build_candidate_plan_seed,  # B200 same-object compatibility alias; canonical owner remains portfolio_entry_plans.
    _build_candidate_full_entry_plan_if_affordable,
    _build_cash_capped_entry_plan_for_candidate,
    _resolve_training_entry_plan,
)
from core.portfolio_entry_selection import (
    _candidate_binary_pass,
    _candidate_continuous_score,
    _selected_continuous_score_metrics,
    _continuous_selected_quality_key,
    _simulate_reserved_candidate_order,
    _resource_aware_quality_policy,
    _resource_aware_default_diag,
    _resource_aware_diag_from_result,
    _reorder_resource_aware_binary_greedy,
    _resource_aware_trial_rank_key,
    _reorder_resource_aware_binary_basket,
    _reorder_resource_aware_continuous,
    _capital_preserving_continuous_quality_key,
    _preserves_min_roos_resource_geometry,
    _reorder_resource_aware_continuous_capital_preserving,
    _max_dl_score_order,
    _max_dl_execution_order,
    _max_dl_basket_quality_key,
    _max_dl_resource_deficit,
    _max_dl_basket_is_feasible,
    _reorder_resource_aware_continuous_max_dl,
    _candidate_date_value,
    _stale_score_guard_max_age_days,
    _candidate_has_stale_scored_signal,
    _reorder_resource_aware_continuous_max_dl_feasible_ascent,
    select_resource_aware_action_candidates,
    reorder_candidates_for_resource_aware_quality,
)
from core.portfolio_fast_data import (
    get_fast_dates,
    get_fast_entry_session_values,
    get_fast_live_shadow_cleanup_values,
    get_fast_management_session_values,
    get_fast_current_high_low_values,
    get_fast_current_session_values,
    get_fast_pos,
    get_fast_previous_management_values,
    get_fast_security_profile,
)
from core.trade_plans import (
    clone_shadow_position,
    execute_pre_market_entry_plan,
    execute_pre_market_entry_plan_training,
    resolve_signal_tracking_params,
    should_clear_extended_signal,
    resolve_extended_tbd_shadow_position,
    should_clear_live_extended_signal_without_accounting,
)


def _format_candidate_date(value):
    if value is None or value != value:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime('%Y-%m-%d')
    return str(value)


def _candidate_kind_label(candidate_type):
    normalized = str(candidate_type or '')
    if normalized == 'extended':
        return '延續候選'
    if normalized == 'reentry':
        return 'Re-entry'
    return '新訊號'


def execute_reserved_entries_for_day(
    portfolio,
    active_extended_signals,
    orderable_candidates_today,
    sold_today,
    all_dfs_fast,
    today,
    params,
    cash,
    available_cash,
    sizing_equity,
    max_positions,
    trade_history,
    is_training,
    total_missed_buys,
    entry_stats=None,
    replay_execution_rows=None,
    fee_rebate_state=None,
    extended_day_market_cache=None,
):
    pre_market_occupied = len(portfolio) + len(sold_today)
    cash_template = cash
    cash_milli = coerce_money_like_to_milli(cash)
    available_cash_milli = coerce_money_like_to_milli(available_cash)

    compact_training_execution = bool(is_training and replay_execution_rows is None)

    for cand in orderable_candidates_today:
        if pre_market_occupied >= max_positions:
            break
        candidate_params = cand.get('params_obj') or params
        # Portfolio entry budget is the currently available broker cash. Keep the
        # canonical milli-money value end-to-end; only the rare cash-cap fallback
        # needs a display-money conversion.
        effective_entry_budget = None
        effective_entry_budget_milli = int(available_cash_milli)

        if cand.get('is_orderable') is False:
            continue
        if compact_training_execution:
            # Attribution accepts Timestamp/date values and formats them only when
            # a closed-trade diagnostic is materialized.  Avoid two strftime calls
            # for every attempted training order, including misses.
            signal_date_value = cand.get('signal_date')
            candidate_date_value = cand.get('candidate_date') or cand.get('trade_date') or today
            candidate_kind_label = None
        else:
            candidate_kind_label = _candidate_kind_label(cand.get('type'))
            signal_date_value = _format_candidate_date(cand.get('signal_date'))
            candidate_date_value = _format_candidate_date(cand.get('candidate_date') or cand.get('trade_date') or today)

        if compact_training_execution:
            chosen_entry_plan = _resolve_training_entry_plan(
                cand, effective_entry_budget_milli, candidate_params, sizing_equity
            )
        else:
            chosen_entry_plan = _build_cash_capped_entry_plan_for_candidate(
                cand,
                effective_entry_budget,
                effective_entry_budget_milli,
                candidate_params,
                sizing_equity,
                copy_shadow_position=False,
            )
        if chosen_entry_plan is None:
            continue

        execution_snapshot = None
        if replay_execution_rows is not None:
            execution_snapshot = {
                "_event_type": "entry_execution",
                "execution_order": int(len(replay_execution_rows)),
                "ticker": cand.get("ticker"),
                "type": cand.get("type"),
                "entry_source": cand.get("entry_source"),
                "signal_date": signal_date_value,
                "candidate_date": candidate_date_value,
                "trade_date": _format_candidate_date(today),
                "limit_px": chosen_entry_plan.get("limit_price"),
                "init_sl": chosen_entry_plan.get("init_sl"),
                "candidate_qty": int(cand.get("qty", 0) or 0),
                "chosen_qty": int(chosen_entry_plan.get("qty", 0) or 0),
                "max_qty": cand.get("max_qty"),
                "candidate_sizing_capital": cand.get("sizing_capital"),
                "sizing_equity": sizing_equity,
                "effective_entry_budget": milli_to_money(effective_entry_budget_milli),
                "available_cash_before": milli_to_money(available_cash_milli),
                "reserved_cost": chosen_entry_plan.get("reserved_cost"),
                "params_obj": candidate_params,
                "security_profile": cand.get("security_profile"),
                "entry_filled": False,
                "filled_qty": 0,
                "actual_initial_risk_total_milli": 0,
            }
            replay_execution_rows.append(execution_snapshot)

        candidate_context = cand.get('_ensemble_context') if isinstance(cand.get('_ensemble_context'), dict) else {}
        candidate_all_dfs_fast = candidate_context.get('all_dfs_fast') or all_dfs_fast
        fast_df = candidate_all_dfs_fast[cand['ticker']]
        t_pos = cand['today_pos']
        y_pos = cand['yesterday_pos']
        t_open, t_high, t_low, t_close, t_volume, y_close = (
            get_fast_entry_session_values(fast_df, t_pos, y_pos)
        )
        if extended_day_market_cache is not None:
            cached_day_market = extended_day_market_cache.get(cand['ticker'])
            current_session_values = (t_open, t_high, t_low, t_close, t_volume)
            if (
                cached_day_market is not None
                and int(cached_day_market[0]) == int(t_pos)
            ):
                extended_day_market_cache[cand['ticker']] = (
                    cached_day_market[0],
                    cached_day_market[1],
                    cached_day_market[2],
                    cached_day_market[3],
                    current_session_values,
                )
            else:
                # Normal setups are intentionally skipped by the extended-candidate
                # scan on their signal day.  If such an order is attempted, attach
                # the completed-bar context here so EOD cleanup can reuse both the
                # previous bar and the session that execution already consumed.
                extended_day_market_cache[cand['ticker']] = (
                    int(t_pos),
                    get_fast_previous_management_values(fast_df, t_pos),
                    get_fast_dates(fast_df)[y_pos],
                    get_fast_security_profile(fast_df),
                    current_session_values,
                )

        reserved_cost_milli = int(chosen_entry_plan.get('reserved_cost_milli', chosen_entry_plan.get('proj_cost_milli', 0)) or 0)
        available_cash_milli -= reserved_cost_milli
        pre_market_occupied += 1

        if compact_training_execution:
            filled_position, count_as_missed_buy = execute_pre_market_entry_plan_training(
                entry_plan=chosen_entry_plan,
                t_open=t_open,
                t_high=t_high,
                t_low=t_low,
                t_close=t_close,
                t_volume=t_volume,
                y_close=y_close,
                params=candidate_params,
                entry_type=cand['type'],
                ticker=cand['ticker'],
                trade_date=today,
            )
            entry_filled = filled_position is not None
            entry_fill_price = (
                None if filled_position is None else filled_position.get("entry_fill_price")
            )
        else:
            entry_result = execute_pre_market_entry_plan(
                entry_plan=chosen_entry_plan,
                t_open=t_open,
                t_high=t_high,
                t_low=t_low,
                t_close=t_close,
                t_volume=t_volume,
                y_close=y_close,
                params=candidate_params,
                entry_type=cand['type'],
                ticker=cand['ticker'],
                trade_date=today,
            )
            entry_filled = bool(entry_result.get("filled", False))
            filled_position = entry_result.get("position")
            count_as_missed_buy = bool(entry_result.get("count_as_missed_buy", False))
            entry_fill_price = entry_result.get(
                "entry_fill_price", entry_result.get("buy_price")
            )

        if execution_snapshot is not None:
            execution_snapshot["entry_filled"] = bool(entry_filled)
            if entry_filled:
                execution_snapshot["filled_qty"] = int(filled_position.get("initial_qty", 0) or 0)
                execution_snapshot["actual_initial_risk_total_milli"] = int(
                    filled_position.get("initial_risk_total_milli", 0) or 0
                )
                execution_snapshot["entry_fill_price"] = entry_fill_price

        if entry_filled:
            if candidate_kind_label is None:
                candidate_kind_label = _candidate_kind_label(cand.get('type'))
            # The fill session already read today's close; preserve it for the
            # later end-of-day equity mark so the same bar is not fetched twice.
            filled_position['last_px'] = t_close
            actual_total_cost_milli = int(filled_position['cash_buy_total_milli'])
            cash_milli -= actual_total_cost_milli
            accrue_fee_rebate(
                fee_rebate_state,
                filled_position['buy_fee_rebate_receivable_milli'],
            )
            filled_position['_entry_params_obj'] = candidate_params
            params_signature = cand.get('params_signature')
            if params_signature:
                filled_position['_entry_params_signature'] = str(params_signature)
            ensemble_vote_count = cand.get('ensemble_vote_count')
            if ensemble_vote_count is not None:
                filled_position['_ensemble_vote_count'] = ensemble_vote_count
            ensemble_member_key = cand.get('ensemble_member_key')
            if ensemble_member_key is not None:
                filled_position['_ensemble_member_key'] = ensemble_member_key
            ensemble_min_agree = cand.get('ensemble_min_agree')
            if ensemble_min_agree is not None:
                filled_position['_ensemble_min_agree'] = ensemble_min_agree
            ensemble_member_keys = cand.get('ensemble_member_keys')
            if isinstance(ensemble_member_keys, (list, tuple, set)):
                filled_position['_ensemble_member_keys'] = sorted({str(key) for key in ensemble_member_keys if str(key).strip()})
            ensemble_member_params_by_key = cand.get('ensemble_member_params_by_key')
            if isinstance(ensemble_member_params_by_key, dict):
                # # (AI註: 持倉需承接本次共識的全部 member 參數，讓 STOP 後各 member 能獨立進入 reclaim watchlist。)
                filled_position['_ensemble_member_params_by_key'] = dict(ensemble_member_params_by_key)
            ensemble_member_quality_rank_by_key = cand.get('ensemble_member_quality_rank_by_key')
            if isinstance(ensemble_member_quality_rank_by_key, dict):
                filled_position['_ensemble_member_quality_rank_by_key'] = {
                    str(key): dict(value)
                    for key, value in ensemble_member_quality_rank_by_key.items()
                    if str(key).strip() and isinstance(value, dict)
                }
            if candidate_context:
                filled_position['_entry_context'] = candidate_context
            # # (AI註: 保存本次實際進場對應的訊號／候選日期，只供 round-trip 歸因與稽核，不介入交易決策。)
            filled_position['signal_date'] = signal_date_value
            filled_position['candidate_date'] = candidate_date_value
            filled_position['candidate_type'] = candidate_kind_label
            # (AI註: 保存進場時的 quality ranking 診斷欄位；只供事後歸因，不介入持倉管理。)
            if bool(cand.get('use_breakout_quality_ranking', False)):
                filled_position['breakout_quality_score'] = cand.get('breakout_quality_score')
                filled_position['breakout_quality_score_date'] = cand.get('breakout_quality_score_date')
                filled_position['use_breakout_quality_ranking'] = True
            quality_rank_payload = cand.get('breakout_quality_rank')
            if isinstance(quality_rank_payload, dict):
                filled_position['breakout_quality_rank'] = dict(quality_rank_payload)
            portfolio[cand['ticker']] = filled_position
            if entry_stats is not None:
                entry_stats['filled_buy_count'] = int(entry_stats.get('filled_buy_count', 0) or 0) + 1

            if cand['ticker'] in active_extended_signals:
                del active_extended_signals[cand['ticker']]
            if not is_training:
                buy_position = filled_position
                tp_half_price = buy_position.get('tp_half')
                if bool(buy_position.get('sold_half', False)):
                    tp_half_price = None
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': cand['ticker'],
                        'Type': f"買進 ({candidate_kind_label}, EV:{cand['ev']:.2f}R)",
                        '買訊日': signal_date_value,
                        '候選日': candidate_date_value,
                        '候選類型': candidate_kind_label,
                        '買入限價': chosen_entry_plan['limit_price'],
                        '成交價': entry_fill_price,
                        '成本均價': entry_result.get('cost_basis_price', entry_result['entry_price']),
                        '停損價': buy_position.get('sl'),
                        '半倉停利價': tp_half_price,
                        'Shadow買進價': buy_position.get('shadow_entry_fill_price'),
                        '股數': buy_position['initial_qty'],
                        '預留總金額': milli_to_money(reserved_cost_milli),
                        '投入總金額': milli_to_money(actual_total_cost_milli),
                        '進場類型': cand['type'],
                        '單筆損益': 0.0,
                        '該筆總損益': 0.0,
                        'R_Multiple': 0.0,
                        'Risk': candidate_params.fixed_risk,
                        'Quality Score': cand.get('breakout_quality_score'),
                        'Quality Score Date': cand.get('breakout_quality_score_date'),
                        'Quality Ranking': bool(cand.get('use_breakout_quality_ranking', False)),
                        'Ensemble Vote Count': cand.get('ensemble_vote_count'),
                    }
                )
        elif count_as_missed_buy:
            total_missed_buys += 1
            if not is_training:
                if candidate_kind_label is None:
                    candidate_kind_label = _candidate_kind_label(cand.get('type'))
                miss_buy_type = '錯失買進(Re-entry)' if cand['type'] == 'reentry' else ('錯失買進(延續候選)' if cand['type'] == 'extended' else '錯失買進(新訊號)')
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': cand['ticker'],
                        'Type': miss_buy_type,
                        '買訊日': signal_date_value,
                        '候選日': candidate_date_value,
                        '候選類型': candidate_kind_label,
                        '進場類型': cand['type'],
                        '單筆損益': 0.0,
                        '該筆總損益': 0.0,
                        'R_Multiple': 0.0,
                        'Risk': candidate_params.fixed_risk,
                        '買入限價': chosen_entry_plan['limit_price'],
                        '成交價': None,
                        '成本均價': None,
                        '股數': chosen_entry_plan['qty'],
                        '預留總金額': milli_to_money(reserved_cost_milli),
                        '投入總金額': 0.0,
                        'Quality Score': cand.get('breakout_quality_score'),
                        'Quality Score Date': cand.get('breakout_quality_score_date'),
                        'Quality Ranking': bool(cand.get('use_breakout_quality_ranking', False)),
                        'Ensemble Vote Count': cand.get('ensemble_vote_count'),
                        '備註': f"預掛限價 {chosen_entry_plan['limit_price']:.2f} 未成交",
                    }
                )

    return restore_money_like_from_milli(cash_milli, cash_template), total_missed_buys


def cleanup_extended_signals_for_day(
    active_extended_signals,
    portfolio,
    all_dfs_fast,
    today,
    params,
    sizing_capital,
    extended_day_market_cache=None,
    candidate_rows_by_ticker=None,
):
    sorted_tickers = (
        None if extended_day_market_cache is None
        else extended_day_market_cache.get(("__meta__", "sorted_extended_tickers"))
    )
    if sorted_tickers is None:
        sorted_tickers = sorted(active_extended_signals)
    for ticker in sorted_tickers:
        signal_state = active_extended_signals.get(ticker)
        if signal_state is None:
            continue
        signal_params = resolve_signal_tracking_params(signal_state, params)

        if ticker in portfolio:
            del active_extended_signals[ticker]
            continue

        fast_df = all_dfs_fast.get(ticker)
        if fast_df is None:
            continue

        cached_day_market = (
            None if extended_day_market_cache is None
            else extended_day_market_cache.get(ticker)
        )
        if cached_day_market is not None:
            t_pos = int(cached_day_market[0])
        else:
            t_pos = get_fast_pos(fast_df, today)
        if t_pos <= 0:
            continue
        cached_previous = (
            cached_day_market[1]
            if cached_day_market is not None and int(cached_day_market[0]) == int(t_pos)
            else None
        )
        cached_current = (
            cached_day_market[4]
            if cached_day_market is not None
            and len(cached_day_market) > 4
            and int(cached_day_market[0]) == int(t_pos)
            else None
        )

        live_shadow_position = resolve_extended_tbd_shadow_position(signal_state)
        if live_shadow_position is not None:
            if cached_previous is not None:
                y_atr, y_ind_sell, y_close, y_high = cached_previous
                if cached_current is not None:
                    t_open_cached, t_high, t_low, t_close_cached, t_volume_cached = cached_current
                else:
                    t_high, t_low = get_fast_current_high_low_values(fast_df, t_pos)
                    t_open_cached = t_close_cached = t_volume_cached = float('nan')
                if y_ind_sell:
                    if cached_current is not None:
                        t_open, t_close, t_volume = t_open_cached, t_close_cached, t_volume_cached
                    else:
                        t_open, t_high, t_low, t_close, t_volume = get_fast_current_session_values(
                            fast_df, t_pos
                        )
                else:
                    t_open = t_close = t_volume = y_close = float('nan')
            else:
                t_high, t_low, y_high, y_atr, y_ind_sell = (
                    get_fast_live_shadow_cleanup_values(fast_df, t_pos)
                )
                if y_ind_sell:
                    (
                        _y_atr, _y_ind_sell, y_close, _y_high, t_open,
                        _t_high, _t_low, t_close, t_volume,
                    ) = get_fast_management_session_values(fast_df, t_pos)
                else:
                    t_open = t_close = t_volume = y_close = float('nan')
        else:
            if cached_previous is not None:
                y_atr, y_ind_sell, y_close, y_high = cached_previous
                if cached_current is not None:
                    t_open, t_high, t_low, t_close, t_volume = cached_current
                else:
                    t_open, t_high, t_low, t_close, t_volume = get_fast_current_session_values(
                        fast_df, t_pos
                    )
            else:
                (
                    y_atr, y_ind_sell, y_close, y_high, t_open, t_high, t_low,
                    t_close, t_volume,
                ) = get_fast_management_session_values(fast_df, t_pos)
        if live_shadow_position is not None:
            should_clear = should_clear_live_extended_signal_without_accounting(
                signal_state, shadow_position=live_shadow_position,
                t_low=t_low, t_high=t_high, t_open=t_open, t_close=t_close,
                t_volume=t_volume, y_close=y_close, y_high=y_high, y_atr=y_atr,
                y_ind_sell=y_ind_sell, params=signal_params,
            )
        else:
            counterfactual_anchor_plan = None
            if candidate_rows_by_ticker is not None:
                candidate_row = candidate_rows_by_ticker.get(ticker)
                if candidate_row is not None:
                    # Candidate rows already contain the frozen geometry and exact
                    # quantity consumed by the counterfactual shadow constructor.
                    # The constructor accepts the compact ``limit_px`` alias, so no
                    # second plan dict is needed for optimizer cleanup.
                    counterfactual_anchor_plan = candidate_row
            should_clear = should_clear_extended_signal(
                signal_state, t_low, t_high, t_open=t_open, t_close=t_close,
                t_volume=t_volume, y_close=y_close, y_high=y_high, y_atr=y_atr,
                y_ind_sell=y_ind_sell, sizing_capital=sizing_capital,
                current_date=today, params=signal_params, copy_shadow_position=False,
                counterfactual_anchor_plan=counterfactual_anchor_plan,
            )
        if should_clear:
            del active_extended_signals[ticker]


