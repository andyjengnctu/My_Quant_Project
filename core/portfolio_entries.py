from core.capital_policy import resolve_portfolio_entry_budget
from core.exact_accounting import (
    coerce_money_like_to_milli,
    milli_to_money,
    restore_money_like_from_milli,
)
from core.trade_plans import (
    build_cash_capped_entry_plan,
    entry_notional_meets_minimum,
    execute_pre_market_entry_plan,
    should_clear_extended_signal,
)
from core.portfolio_fast_data import get_fast_close, get_fast_pos, get_fast_value


def _build_candidate_plan_seed(candidate_row, sizing_equity=None):
    sizing_capital = candidate_row.get('sizing_capital')
    if (sizing_capital is None or sizing_capital != sizing_capital) and sizing_equity is not None:
        sizing_capital = sizing_equity
    return {
        'limit_price': candidate_row['limit_px'],
        'init_sl': candidate_row['init_sl'],
        'init_trail': candidate_row['init_trail'],
        'target_price': candidate_row.get('target_price'),
        'entry_atr': candidate_row.get('entry_atr'),
        'ticker': candidate_row.get('ticker'),
        'security_profile': candidate_row.get('security_profile'),
        'trade_date': candidate_row.get('trade_date'),
        'sizing_capital': sizing_capital,
    }


def _build_candidate_full_entry_plan_if_affordable(candidate_row, available_cash_milli, params, sizing_equity=None):
    qty = int(candidate_row.get('qty', 0) or 0)
    reserved_cost_milli = int(candidate_row.get('proj_cost_milli', 0) or 0)
    if qty <= 0 or reserved_cost_milli <= 0 or reserved_cost_milli > int(available_cash_milli):
        return None
    if not entry_notional_meets_minimum(candidate_row.get('limit_px'), qty, params):
        return None

    entry_plan = _build_candidate_plan_seed(candidate_row, sizing_equity=sizing_equity)
    entry_plan['qty'] = qty
    entry_plan['is_orderable'] = True
    entry_plan['reserved_cost_milli'] = reserved_cost_milli
    entry_plan['reserved_cost'] = milli_to_money(reserved_cost_milli)
    return entry_plan


def _build_cash_capped_entry_plan_for_candidate(candidate_row, effective_entry_budget, effective_entry_budget_milli, params, sizing_equity):
    full_entry_plan = _build_candidate_full_entry_plan_if_affordable(
        candidate_row,
        effective_entry_budget_milli,
        params,
        sizing_equity=sizing_equity,
    )
    if full_entry_plan is not None:
        return full_entry_plan
    return build_cash_capped_entry_plan(
        _build_candidate_plan_seed(candidate_row, sizing_equity=sizing_equity),
        effective_entry_budget,
        params,
    )


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
):
    pre_market_occupied = len(portfolio) + len(sold_today)
    remaining_orderable_candidates = list(orderable_candidates_today)
    cash_template = cash
    cash_milli = coerce_money_like_to_milli(cash)
    available_cash_milli = coerce_money_like_to_milli(available_cash)

    while remaining_orderable_candidates and pre_market_occupied < max_positions:
        effective_entry_budget = resolve_portfolio_entry_budget(
            milli_to_money(available_cash_milli),
            params.initial_capital,
            params,
        )
        effective_entry_budget_milli = coerce_money_like_to_milli(effective_entry_budget)

        cand = remaining_orderable_candidates.pop(0)
        if cand.get('is_orderable') is False:
            continue

        chosen_entry_plan = _build_cash_capped_entry_plan_for_candidate(
            cand,
            effective_entry_budget,
            effective_entry_budget_milli,
            params,
            sizing_equity,
        )
        if chosen_entry_plan is None:
            continue

        fast_df = all_dfs_fast[cand['ticker']]
        t_pos = cand['today_pos']
        y_pos = cand['yesterday_pos']
        t_open = get_fast_value(fast_df, 'Open', pos=t_pos)
        t_high = get_fast_value(fast_df, 'High', pos=t_pos)
        t_low = get_fast_value(fast_df, 'Low', pos=t_pos)
        t_close = get_fast_close(fast_df, pos=t_pos)
        t_volume = get_fast_value(fast_df, 'Volume', pos=t_pos)
        y_close = get_fast_close(fast_df, pos=y_pos)

        reserved_cost_milli = chosen_entry_plan['reserved_cost_milli']
        available_cash_milli -= reserved_cost_milli
        pre_market_occupied += 1

        entry_result = execute_pre_market_entry_plan(
            entry_plan=chosen_entry_plan,
            t_open=t_open,
            t_high=t_high,
            t_low=t_low,
            t_close=t_close,
            t_volume=t_volume,
            y_close=y_close,
            params=params,
            entry_type=cand['type'],
            ticker=cand['ticker'],
            trade_date=today,
        )

        if entry_result['filled']:
            actual_total_cost_milli = entry_result['position']['net_buy_total_milli']
            cash_milli -= actual_total_cost_milli
            portfolio[cand['ticker']] = entry_result['position']

            if cand['ticker'] in active_extended_signals:
                del active_extended_signals[cand['ticker']]
            if not is_training:
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': cand['ticker'],
                        'Type': f"買進 (EV:{cand['ev']:.2f}R)",
                        '買入限價': chosen_entry_plan['limit_price'],
                        '成交價': entry_result.get('entry_fill_price', entry_result['buy_price']),
                        '成本均價': entry_result.get('cost_basis_price', entry_result['entry_price']),
                        '股數': entry_result['position']['initial_qty'],
                        '投入總金額': milli_to_money(actual_total_cost_milli),
                        '進場類型': cand['type'],
                        '單筆損益': 0.0,
                        '該筆總損益': 0.0,
                        'R_Multiple': 0.0,
                        'Risk': params.fixed_risk,
                    }
                )
        elif entry_result['count_as_missed_buy']:
            total_missed_buys += 1
            if not is_training:
                miss_buy_type = '錯失買進(延續候選)' if cand['type'] == 'extended' else '錯失買進(新訊號)'
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': cand['ticker'],
                        'Type': miss_buy_type,
                        '單筆損益': 0.0,
                        '該筆總損益': 0.0,
                        'R_Multiple': 0.0,
                        'Risk': params.fixed_risk,
                        '買入限價': chosen_entry_plan['limit_price'],
                        '備註': f"預掛限價 {chosen_entry_plan['limit_price']:.2f} 未成交",
                        '投入總金額': milli_to_money(reserved_cost_milli),
                    }
                )

    return restore_money_like_from_milli(cash_milli, cash_template), total_missed_buys


def cleanup_extended_signals_for_day(active_extended_signals, portfolio, all_dfs_fast, today, params, sizing_capital):
    for ticker in sorted(list(active_extended_signals.keys())):
        if ticker in portfolio:
            del active_extended_signals[ticker]
            continue

        fast_df = all_dfs_fast.get(ticker)
        if fast_df is None:
            continue

        t_pos = get_fast_pos(fast_df, today)
        if t_pos < 0:
            continue

        if t_pos <= 0:
            continue
        y_pos = t_pos - 1
        t_open = get_fast_value(fast_df, 'Open', pos=t_pos)
        t_low = get_fast_value(fast_df, 'Low', pos=t_pos)
        t_high = get_fast_value(fast_df, 'High', pos=t_pos)
        t_close = get_fast_close(fast_df, pos=t_pos)
        t_volume = get_fast_value(fast_df, 'Volume', pos=t_pos)
        y_close = get_fast_close(fast_df, pos=y_pos)
        y_high = get_fast_value(fast_df, 'High', pos=y_pos)
        y_atr = get_fast_value(fast_df, 'ATR', pos=y_pos)
        y_ind_sell = bool(get_fast_value(fast_df, 'ind_sell_signal', pos=y_pos))
        if should_clear_extended_signal(
            active_extended_signals[ticker],
            t_low,
            t_high,
            t_open=t_open,
            t_close=t_close,
            t_volume=t_volume,
            y_close=y_close,
            y_high=y_high,
            y_atr=y_atr,
            y_ind_sell=y_ind_sell,
            sizing_capital=sizing_capital,
            current_date=today,
            params=params,
        ):
            del active_extended_signals[ticker]
