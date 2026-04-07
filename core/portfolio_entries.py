from core.capital_policy import resolve_portfolio_entry_budget
from core.config import get_buy_sort_method
from core.trade_plans import (
    build_cash_capped_entry_plan,
    execute_pre_market_entry_plan,
    should_clear_extended_signal,
)
from core.portfolio_fast_data import get_fast_close, get_fast_pos, get_fast_value


def _build_candidate_plan_seed(candidate_row):
    return {
        'limit_price': candidate_row['limit_px'],
        'init_sl': candidate_row['init_sl'],
        'init_trail': candidate_row['init_trail'],
        'target_price': candidate_row.get('target_price'),
        'entry_atr': candidate_row.get('entry_atr'),
    }


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
    max_positions,
    trade_history,
    is_training,
    total_missed_buys,
):
    pre_market_occupied = len(portfolio) + len(sold_today)
    remaining_orderable_candidates = list(orderable_candidates_today)

    while remaining_orderable_candidates and pre_market_occupied < max_positions:
        chosen_idx = 0
        chosen_entry_plan = None

        effective_entry_budget = resolve_portfolio_entry_budget(
            available_cash,
            params.initial_capital,
            params,
        )

        if get_buy_sort_method() == 'PROJ_COST':
            chosen_key = None
            for cand_idx, probe_cand in enumerate(remaining_orderable_candidates):
                probe_entry_plan = build_cash_capped_entry_plan(
                    _build_candidate_plan_seed(probe_cand),
                    effective_entry_budget,
                    params,
                )
                if probe_entry_plan is None:
                    continue

                probe_key = (probe_entry_plan['reserved_cost'], probe_cand['ticker'])
                if chosen_key is None or probe_key > chosen_key:
                    chosen_key = probe_key
                    chosen_idx = cand_idx
                    chosen_entry_plan = probe_entry_plan

            if chosen_entry_plan is None:
                break

        cand = remaining_orderable_candidates.pop(chosen_idx)
        if cand.get('is_orderable') is False:
            continue
        if chosen_entry_plan is None:
            chosen_entry_plan = build_cash_capped_entry_plan(
                _build_candidate_plan_seed(cand),
                effective_entry_budget,
                params,
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

        reserved_cost = chosen_entry_plan['reserved_cost']
        available_cash -= reserved_cost
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
        )

        if entry_result['filled']:
            qty = chosen_entry_plan['qty']
            actual_total_cost = entry_result['entry_price'] * qty
            cash -= actual_total_cost
            portfolio[cand['ticker']] = entry_result['position']

            if cand['ticker'] in active_extended_signals:
                del active_extended_signals[cand['ticker']]
            if not is_training:
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': cand['ticker'],
                        'Type': f"買進 (EV:{cand['ev']:.2f}R)",
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
                        '備註': f"預掛限價 {chosen_entry_plan['limit_price']:.2f} 未成交",
                        '投入總金額': reserved_cost,
                    }
                )

    return cash, total_missed_buys

def cleanup_extended_signals_for_day(active_extended_signals, portfolio, all_dfs_fast, today):
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

        t_low = get_fast_value(fast_df, 'Low', pos=t_pos)
        t_high = get_fast_value(fast_df, 'High', pos=t_pos)
        if should_clear_extended_signal(active_extended_signals[ticker], t_low, t_high):
            del active_extended_signals[ticker]
