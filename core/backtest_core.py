import numpy as np

from core.backtest_finalize import build_backtest_stats, finalize_open_position_at_end
from core.capital_policy import resolve_single_backtest_sizing_capital
from core.exact_accounting import (
    allocate_cost_basis_milli,
    build_sell_ledger_from_price,
    calc_buy_net_total_milli_from_milli,
    calc_initial_risk_total_milli,
    calc_ratio_from_milli,
    calc_sell_net_total_milli_from_milli,
    milli_to_money,
    money_to_milli,
    price_to_milli,
    rate_to_ppm,
)
from core.strategy_params import V16StrategyParams
from core.position_step import execute_bar_step
from core.price_utils import (
    adjust_long_buy_fill_price,
    adjust_long_sell_fill_price,
    adjust_long_stop_price,
    calc_frozen_target_price,
    calc_half_take_profit_sell_qty,
    calc_initial_stop_from_reference,
    calc_initial_trailing_stop_from_reference,
    calc_position_size,
    get_exit_sell_block_reason,
    is_limit_buy_price_reachable_for_day,
    is_locked_limit_up_bar,
)
from core.signal_utils import generate_signals
from core.trade_plans import (
    build_extended_entry_plan_from_signal,
    build_normal_entry_plan,
    create_signal_tracking_state,
    execute_pre_market_entry_plan,
    should_clear_extended_signal,
)


def _empty_pit_stats_index():
    return {
        'exit_dates': [],
        'cum_trade_count': np.array([], dtype=np.int32),
        'cum_win_count': np.array([], dtype=np.int32),
        'cum_win_r_sum': np.array([], dtype=np.float64),
        'cum_loss_r_sum': np.array([], dtype=np.float64),
        'cum_total_r_sum': np.array([], dtype=np.float64),
        'cum_pnl_sum': np.array([], dtype=np.float64),
    }


def _create_pit_stats_builder():
    return {
        'exit_dates': [],
        'cum_trade_count': [],
        'cum_win_count': [],
        'cum_win_r_sum': [],
        'cum_loss_r_sum': [],
        'cum_total_r_sum': [],
        'cum_pnl_sum': [],
        '_trade_count': 0,
        '_win_count': 0,
        '_win_r_sum': 0.0,
        '_loss_r_sum': 0.0,
        '_total_r_sum': 0.0,
        '_pnl_sum': 0.0,
    }


def _append_pit_trade(builder, *, exit_date, pnl, r_mult):
    if builder is None:
        return

    builder['_trade_count'] += 1
    builder['_total_r_sum'] += float(r_mult)
    builder['_pnl_sum'] += float(pnl)
    if float(pnl) > 0.0:
        builder['_win_count'] += 1
        builder['_win_r_sum'] += float(r_mult)
    else:
        builder['_loss_r_sum'] += float(r_mult)

    builder['exit_dates'].append(exit_date)
    builder['cum_trade_count'].append(builder['_trade_count'])
    builder['cum_win_count'].append(builder['_win_count'])
    builder['cum_win_r_sum'].append(builder['_win_r_sum'])
    builder['cum_loss_r_sum'].append(builder['_loss_r_sum'])
    builder['cum_total_r_sum'].append(builder['_total_r_sum'])
    builder['cum_pnl_sum'].append(builder['_pnl_sum'])


def _finalize_pit_stats_index(builder):
    if builder is None:
        return None
    if not builder['exit_dates']:
        return _empty_pit_stats_index()
    return {
        'exit_dates': list(builder['exit_dates']),
        'cum_trade_count': np.array(builder['cum_trade_count'], dtype=np.int32),
        'cum_win_count': np.array(builder['cum_win_count'], dtype=np.int32),
        'cum_win_r_sum': np.array(builder['cum_win_r_sum'], dtype=np.float64),
        'cum_loss_r_sum': np.array(builder['cum_loss_r_sum'], dtype=np.float64),
        'cum_total_r_sum': np.array(builder['cum_total_r_sum'], dtype=np.float64),
        'cum_pnl_sum': np.array(builder['cum_pnl_sum'], dtype=np.float64),
    }


def _optimizer_limit_reachable_for_entry_day(t_low, t_open, t_volume, limit_price):
    """Cheap no-fill guard for optimizer PIT mode.

    The formal entry function remains the single source of truth for fills.
    This guard only skips building full entry plans on days that cannot fill
    because required bar fields are invalid or the day low never reaches the
    plan limit. It is used only when collect_stats=False, so scanner/display
    miss-buy accounting stays on the original full path.
    """
    if np.isnan(t_low) or np.isnan(t_open) or np.isnan(t_volume) or np.isnan(limit_price):
        return False
    if float(t_volume) <= 0.0:
        return False
    return price_to_milli(t_low) <= price_to_milli(limit_price)


def _optimizer_extended_entry_limit(active_extended_signal):
    if active_extended_signal is None:
        return np.nan
    shadow_position = active_extended_signal.get("shadow_position")
    if shadow_position is not None and int(shadow_position.get("qty", 0) or 0) > 0:
        if shadow_position.get("pending_exit_action") is not None:
            return np.nan
        return shadow_position.get("entry_fill_price", np.nan)
    return active_extended_signal.get("orig_limit", np.nan)



def _pit_to_price(price_milli):
    return milli_to_money(int(price_milli))


def _pit_make_position(
    *,
    buy_price,
    qty,
    params,
    entry_type,
    target_price=None,
    limit_price=None,
    entry_atr=None,
    init_sl=None,
    init_trail=None,
    ticker=None,
    security_profile=None,
    trade_date=None,
    inherited_shadow=None,
    t_high=np.nan,
    t_low=np.nan,
):
    if pd_isna(buy_price) or buy_price <= 0 or qty <= 0:
        return None

    if entry_atr is not None and not pd_isna(entry_atr):
        resolved_init_sl = calc_initial_stop_from_reference(
            buy_price,
            entry_atr,
            params,
            ticker=ticker,
            security_profile=security_profile,
        )
        resolved_init_trail = calc_initial_trailing_stop_from_reference(
            buy_price,
            entry_atr,
            params,
            ticker=ticker,
            security_profile=security_profile,
        )
        resolved_target_price = calc_frozen_target_price(
            buy_price,
            resolved_init_sl,
            ticker=ticker,
            security_profile=security_profile,
        )
    else:
        resolved_init_sl = init_sl
        resolved_init_trail = init_trail
        resolved_target_price = target_price
        if resolved_target_price is None or pd_isna(resolved_target_price):
            target_basis = buy_price if limit_price is None else limit_price
            resolved_target_price = calc_frozen_target_price(
                target_basis,
                resolved_init_sl,
                ticker=ticker,
                security_profile=security_profile,
            )

    if pd_isna(resolved_init_sl) or pd_isna(resolved_init_trail) or pd_isna(resolved_target_price):
        return None

    buy_price_milli = price_to_milli(buy_price)
    initial_stop_milli = price_to_milli(resolved_init_sl)
    trailing_stop_milli = price_to_milli(resolved_init_trail)
    target_price_milli = price_to_milli(resolved_target_price)
    net_buy_total_milli = calc_buy_net_total_milli_from_milli(buy_price_milli, qty, params)
    stop_net_total_milli = calc_sell_net_total_milli_from_milli(
        initial_stop_milli,
        qty,
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
    )
    initial_risk_total_milli = calc_initial_risk_total_milli(
        net_buy_total_milli,
        stop_net_total_milli,
        rate_to_ppm(params.fixed_risk),
    )

    position = {
        'qty': int(qty),
        'initial_qty': int(qty),
        'entry_fill_price_milli': buy_price_milli,
        'net_buy_total_milli': net_buy_total_milli,
        'remaining_cost_basis_milli': net_buy_total_milli,
        'sl_milli': max(initial_stop_milli, trailing_stop_milli),
        'initial_stop_milli': initial_stop_milli,
        'trailing_stop_milli': trailing_stop_milli,
        'tp_half_milli': target_price_milli,
        'sold_half': False,
        'realized_pnl_milli': 0,
        'initial_risk_total_milli': initial_risk_total_milli,
        'entry_type': entry_type,
        'ticker': ticker,
        'security_profile': security_profile,
        'entry_trade_date': trade_date,
        'pending_exit_action': None,
        'pending_exit_trigger_price_milli': 0,
        'highest_high_since_entry_milli': buy_price_milli,
    }

    if inherited_shadow is not None:
        for field in (
            'sl_milli',
            'initial_stop_milli',
            'trailing_stop_milli',
            'tp_half_milli',
            'sold_half',
            'highest_high_since_entry_milli',
            'pending_exit_action',
            'pending_exit_trigger_price_milli',
        ):
            if field in inherited_shadow:
                position[field] = inherited_shadow[field]
        if not pd_isna(t_high):
            position['highest_high_since_entry_milli'] = max(
                int(position.get('highest_high_since_entry_milli', position['entry_fill_price_milli'])),
                price_to_milli(t_high),
            )
        return position

    if not pd_isna(t_high):
        position['highest_high_since_entry_milli'] = max(position['highest_high_since_entry_milli'], price_to_milli(t_high))

    stop_hit = (not pd_isna(t_low)) and price_to_milli(t_low) <= int(position['sl_milli'])
    half_sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    tp_hit = (not pd_isna(t_high)) and price_to_milli(t_high) >= int(position['tp_half_milli'])
    if stop_hit and tp_hit:
        tp_hit = False
    if stop_hit:
        position['pending_exit_action'] = 'STOP'
        position['pending_exit_trigger_price_milli'] = int(position['sl_milli'])
    elif tp_hit and half_sell_qty > 0 and not position.get('sold_half', False):
        position['pending_exit_action'] = 'TP_HALF'
        position['pending_exit_trigger_price_milli'] = int(position['tp_half_milli'])
    return position


def pd_isna(value):
    return bool(np.isnan(value)) if isinstance(value, (float, np.floating)) else value is None


def _pit_build_normal_order(limit_price, atr, sizing_capital, params, *, ticker=None, security_profile=None, trade_date=None):
    if pd_isna(limit_price) or pd_isna(atr):
        return None
    init_sl = calc_initial_stop_from_reference(limit_price, atr, params, ticker=ticker, security_profile=security_profile)
    init_trail = calc_initial_trailing_stop_from_reference(limit_price, atr, params, ticker=ticker, security_profile=security_profile)
    target_price = calc_frozen_target_price(limit_price, init_sl, ticker=ticker, security_profile=security_profile)
    if pd_isna(init_sl) or pd_isna(init_trail) or pd_isna(target_price):
        return None
    qty = calc_position_size(
        limit_price,
        init_sl,
        sizing_capital,
        params.fixed_risk,
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
    )
    if qty <= 0:
        return None
    return {
        'limit_price': limit_price,
        'init_sl': init_sl,
        'init_trail': init_trail,
        'target_price': target_price,
        'entry_atr': atr,
        'qty': int(qty),
        'ticker': ticker,
        'security_profile': security_profile,
        'trade_date': trade_date,
    }


def _pit_signal_state(original_limit, atr, *, ticker=None, security_profile=None):
    if pd_isna(original_limit) or pd_isna(atr):
        return None
    return {
        'orig_limit': original_limit,
        'orig_atr': atr,
        'shadow_position': None,
        'ticker': ticker,
        'security_profile': security_profile,
    }


def _pit_shadow_alive(signal_state):
    if signal_state is None:
        return False
    shadow = signal_state.get('shadow_position')
    return shadow is not None and int(shadow.get('qty', 0) or 0) > 0


def _pit_shadow_touch_barrier(signal_state, *, day_low, day_high):
    if signal_state is None:
        return False
    shadow = signal_state.get('shadow_position')
    if shadow is not None and int(shadow.get('qty', 0) or 0) > 0:
        if shadow.get('pending_exit_action') in {'STOP', 'TP_HALF'}:
            return True
        if bool(shadow.get('sold_half', False)):
            return True
        invalidation_milli = shadow.get('sl_milli')
    else:
        invalidation_milli = signal_state.get('continuation_invalidation_barrier_milli')
    if invalidation_milli is None or pd_isna(day_low):
        return False
    return price_to_milli(day_low) <= int(invalidation_milli)


def _pit_build_extended_order(signal_state, sizing_capital, params, *, y_close=np.nan, ticker=None, security_profile=None, trade_date=None, require_orderable=True):
    if signal_state is None:
        return None
    resolved_ticker = ticker or signal_state.get('ticker')
    resolved_security_profile = security_profile or signal_state.get('security_profile')
    entry_atr = signal_state.get('orig_atr', np.nan)
    if pd_isna(entry_atr):
        return None
    shadow = signal_state.get('shadow_position')
    inherited_shadow = None
    if shadow is not None and int(shadow.get('qty', 0) or 0) > 0:
        if require_orderable and shadow.get('pending_exit_action') is not None:
            return None
        limit_price = _pit_to_price(shadow['entry_fill_price_milli'])
        sizing_stop_ref = _pit_to_price(shadow['sl_milli'])
        init_trail = _pit_to_price(shadow['trailing_stop_milli'])
        target_price = _pit_to_price(shadow['tp_half_milli'])
        inherited_shadow = shadow
    else:
        limit_price = signal_state.get('orig_limit', np.nan)
        sizing_stop_ref = calc_initial_stop_from_reference(limit_price, entry_atr, params, ticker=resolved_ticker, security_profile=resolved_security_profile)
        init_trail = calc_initial_trailing_stop_from_reference(limit_price, entry_atr, params, ticker=resolved_ticker, security_profile=resolved_security_profile)
        target_price = calc_frozen_target_price(limit_price, sizing_stop_ref, ticker=resolved_ticker, security_profile=resolved_security_profile)

    if pd_isna(limit_price) or pd_isna(sizing_stop_ref) or pd_isna(init_trail) or pd_isna(target_price):
        return None
    qty = calc_position_size(
        limit_price,
        sizing_stop_ref,
        sizing_capital,
        params.fixed_risk,
        params,
        ticker=resolved_ticker,
        security_profile=resolved_security_profile,
        trade_date=trade_date,
    )
    if qty <= 0:
        return None
    if require_orderable and not is_limit_buy_price_reachable_for_day(limit_price, y_close, ticker=resolved_ticker, security_profile=resolved_security_profile):
        return None
    return {
        'limit_price': limit_price,
        'init_sl': sizing_stop_ref,
        'init_trail': init_trail,
        'target_price': target_price,
        'entry_atr': entry_atr,
        'qty': int(qty),
        'shadow_position_state': inherited_shadow,
        'ticker': resolved_ticker,
        'security_profile': resolved_security_profile,
        'trade_date': trade_date,
    }


def _pit_try_entry(order, *, t_open, t_high, t_low, t_close, t_volume, y_close, params, entry_type, ticker=None, security_profile=None, trade_date=None):
    if order is None:
        return None
    if pd_isna(t_volume) or t_volume <= 0 or pd_isna(t_open) or pd_isna(t_low):
        return None
    resolved_ticker = ticker or order.get('ticker')
    resolved_security_profile = security_profile or order.get('security_profile')
    if is_locked_limit_up_bar(t_open, t_high, t_low, t_close, y_close, ticker=resolved_ticker, security_profile=resolved_security_profile):
        return None
    limit_price = order['limit_price']
    if price_to_milli(t_low) > price_to_milli(limit_price):
        return None
    buy_price = adjust_long_buy_fill_price(min(t_open, limit_price), ticker=resolved_ticker, security_profile=resolved_security_profile)
    return _pit_make_position(
        buy_price=buy_price,
        qty=order['qty'],
        params=params,
        entry_type=entry_type,
        init_sl=order.get('init_sl'),
        init_trail=order.get('init_trail'),
        target_price=order.get('target_price'),
        limit_price=limit_price,
        entry_atr=order.get('entry_atr'),
        ticker=resolved_ticker,
        security_profile=resolved_security_profile,
        trade_date=trade_date,
        inherited_shadow=order.get('shadow_position_state'),
        t_high=t_high,
        t_low=t_low,
    )


def _pit_sell_leg(position, *, exec_price, sell_qty, params, trade_date=None):
    sell_qty = int(sell_qty)
    if sell_qty <= 0 or int(position.get('qty', 0) or 0) <= 0:
        return 0, 0
    exec_price_milli = price_to_milli(exec_price)
    freed_cash_milli = calc_sell_net_total_milli_from_milli(
        exec_price_milli,
        sell_qty,
        params,
        ticker=position.get('ticker'),
        security_profile=position.get('security_profile'),
        trade_date=trade_date,
    )
    allocated_cost_milli = allocate_cost_basis_milli(position['remaining_cost_basis_milli'], position['qty'], sell_qty)
    pnl_milli = freed_cash_milli - allocated_cost_milli
    position['realized_pnl_milli'] += pnl_milli
    position['remaining_cost_basis_milli'] -= allocated_cost_milli
    position['qty'] -= sell_qty
    if position['qty'] <= 0:
        position['qty'] = 0
        position['remaining_cost_basis_milli'] = 0
    return freed_cash_milli, pnl_milli


def _pit_execute_bar_step(position, *, y_atr, y_ind_sell, y_close, y_high, t_open, t_high, t_low, t_close, t_volume, params, current_date=None):
    if position is None or int(position.get('qty', 0) or 0) <= 0:
        return 0, []
    freed_cash_milli = 0
    events = []

    pending_action = position.get('pending_exit_action')
    if pending_action is not None:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=position.get('ticker'))
        if sell_block_reason is not None:
            return 0, ['MISSED_SELL', sell_block_reason]
        exec_price = adjust_long_sell_fill_price(t_open, ticker=position.get('ticker'))
        if pending_action == 'STOP':
            leg_freed_cash_milli, _leg_pnl_milli = _pit_sell_leg(
                position,
                exec_price=exec_price,
                sell_qty=position['qty'],
                params=params,
                trade_date=current_date,
            )
            freed_cash_milli += leg_freed_cash_milli
            position['pending_exit_action'] = None
            position['pending_exit_trigger_price_milli'] = 0
            return freed_cash_milli, ['DEFERRED_STOP_ON_OPEN', 'STOP']
        if pending_action == 'TP_HALF':
            sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
            if sell_qty <= 0:
                position['pending_exit_action'] = None
                position['pending_exit_trigger_price_milli'] = 0
                return freed_cash_milli, []
            leg_freed_cash_milli, _leg_pnl_milli = _pit_sell_leg(
                position,
                exec_price=exec_price,
                sell_qty=sell_qty,
                params=params,
                trade_date=current_date,
            )
            freed_cash_milli += leg_freed_cash_milli
            position['sold_half'] = True
            position['pending_exit_action'] = None
            position['pending_exit_trigger_price_milli'] = 0
            return freed_cash_milli, ['DEFERRED_TP_HALF_ON_OPEN', 'TP_HALF']

    if not pd_isna(y_atr):
        highest_high_milli = int(position.get('highest_high_since_entry_milli', position['entry_fill_price_milli']))
        if not pd_isna(y_high):
            highest_high_milli = max(highest_high_milli, price_to_milli(y_high))
        position['highest_high_since_entry_milli'] = highest_high_milli
        candidate_trail = adjust_long_stop_price(
            _pit_to_price(highest_high_milli) - (y_atr * params.atr_times_trail),
            ticker=position.get('ticker'),
            security_profile=position.get('security_profile'),
        )
        candidate_trail_milli = price_to_milli(candidate_trail)
        position['trailing_stop_milli'] = max(position.get('trailing_stop_milli', 0), candidate_trail_milli)
        position['sl_milli'] = max(position['initial_stop_milli'], position['trailing_stop_milli'])

    if y_ind_sell:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=position.get('ticker'))
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(t_open, ticker=position.get('ticker'))
            leg_freed_cash_milli, _leg_pnl_milli = _pit_sell_leg(
                position,
                exec_price=exec_price,
                sell_qty=position['qty'],
                params=params,
                trade_date=current_date,
            )
            freed_cash_milli += leg_freed_cash_milli
            events.append('IND_SELL')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])
        return freed_cash_milli, events

    is_stop_hit = price_to_milli(t_low) <= int(position['sl_milli'])
    half_sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    is_tp_hit = price_to_milli(t_high) >= int(position['tp_half_milli']) and not position['sold_half'] and half_sell_qty > 0
    if is_stop_hit and is_tp_hit:
        is_tp_hit = False

    if is_tp_hit and not (pd_isna(t_volume) or t_volume <= 0):
        exec_price = adjust_long_sell_fill_price(max(_pit_to_price(position['tp_half_milli']), t_open), ticker=position.get('ticker'))
        leg_freed_cash_milli, _leg_pnl_milli = _pit_sell_leg(
            position,
            exec_price=exec_price,
            sell_qty=half_sell_qty,
            params=params,
            trade_date=current_date,
        )
        freed_cash_milli += leg_freed_cash_milli
        position['sold_half'] = True
        events.append('TP_HALF')

    if is_stop_hit and position['qty'] > 0:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=position.get('ticker'))
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(min(_pit_to_price(position['sl_milli']), t_open), ticker=position.get('ticker'))
            leg_freed_cash_milli, _leg_pnl_milli = _pit_sell_leg(
                position,
                exec_price=exec_price,
                sell_qty=position['qty'],
                params=params,
                trade_date=current_date,
            )
            freed_cash_milli += leg_freed_cash_milli
            events.append('STOP')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])
    return freed_cash_milli, events


def _pit_ensure_shadow_anchor(signal_state, *, t_open, t_high, t_low, sizing_capital, params, current_date=None):
    if signal_state is None or _pit_shadow_alive(signal_state):
        return signal_state
    order = _pit_build_extended_order(
        signal_state,
        sizing_capital,
        params,
        ticker=signal_state.get('ticker'),
        security_profile=signal_state.get('security_profile'),
        trade_date=current_date,
        require_orderable=False,
    )
    if order is None or pd_isna(t_open):
        return signal_state
    limit_price = order.get('limit_price', np.nan)
    if pd_isna(limit_price):
        return signal_state
    buy_price = adjust_long_buy_fill_price(min(t_open, limit_price), ticker=signal_state.get('ticker'), security_profile=signal_state.get('security_profile'))
    shadow = _pit_make_position(
        buy_price=buy_price,
        qty=order['qty'],
        params=params,
        entry_type='extended_shadow',
        init_sl=order.get('init_sl'),
        init_trail=order.get('init_trail'),
        target_price=order.get('target_price'),
        limit_price=limit_price,
        entry_atr=order.get('entry_atr'),
        ticker=signal_state.get('ticker'),
        security_profile=signal_state.get('security_profile'),
        trade_date=current_date,
        t_high=t_high,
        t_low=t_low,
    )
    signal_state['shadow_position'] = shadow
    return signal_state


def _pit_should_clear_extended_signal(signal_state, *, t_low, t_high, t_open, t_close, t_volume, y_close, y_high, y_atr, y_ind_sell, sizing_capital, current_date, params):
    if signal_state is None:
        return False
    if _pit_shadow_alive(signal_state):
        shadow = signal_state.get('shadow_position')
        _freed, _events = _pit_execute_bar_step(
            shadow,
            y_atr=y_atr,
            y_ind_sell=y_ind_sell,
            y_close=y_close,
            y_high=y_high,
            t_open=t_open,
            t_high=t_high,
            t_low=t_low,
            t_close=t_close,
            t_volume=t_volume,
            params=params,
            current_date=current_date,
        )
        if shadow is None or int(shadow.get('qty', 0) or 0) <= 0:
            signal_state['shadow_position'] = None
            return True
        return _pit_shadow_touch_barrier(signal_state, day_low=t_low, day_high=t_high)

    _pit_ensure_shadow_anchor(
        signal_state,
        t_open=t_open,
        t_high=t_high,
        t_low=t_low,
        sizing_capital=sizing_capital,
        params=params,
        current_date=current_date,
    )
    if not _pit_shadow_alive(signal_state):
        return False
    return _pit_shadow_touch_barrier(signal_state, day_low=t_low, day_high=t_high)


def _run_v16_pit_only_backtest(df, params, precomputed_signals, *, ticker=None):
    resolved_ticker = ticker or df.attrs.get('ticker')
    resolved_security_profile = df.attrs.get('security_profile')
    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    Dates = df.index
    ATR_main, buyCondition, sellCondition, buy_limits = precomputed_signals

    if len(C) == 0:
        return _empty_pit_stats_index()

    pit_stats_builder = _create_pit_stats_builder()
    position = {'qty': 0}
    active_extended_signal = None
    currentCapital_milli = money_to_milli(params.initial_capital)

    valid_setup_mask = np.asarray(buyCondition[:-1], dtype=bool) & ~np.isnan(np.asarray(ATR_main[:-1], dtype=np.float64))
    optimizer_setup_entry_positions = np.flatnonzero(valid_setup_mask) + 1

    j = 1
    while j < len(C):
        if np.isnan(ATR_main[j - 1]):
            j += 1
            continue

        pos_start_of_current_bar = int(position.get('qty', 0) or 0)
        if (
            pos_start_of_current_bar == 0
            and active_extended_signal is None
            and not bool(buyCondition[j - 1])
        ):
            next_setup_cursor = np.searchsorted(optimizer_setup_entry_positions, j + 1)
            if next_setup_cursor < len(optimizer_setup_entry_positions):
                j = int(optimizer_setup_entry_positions[next_setup_cursor])
                continue
            break

        if pos_start_of_current_bar > 0:
            freed_cash_milli, events = _pit_execute_bar_step(
                position,
                y_atr=ATR_main[j - 1],
                y_ind_sell=sellCondition[j - 1],
                y_close=C[j - 1],
                y_high=H[j - 1],
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                params=params,
                current_date=Dates[j],
            )
            currentCapital_milli += freed_cash_milli
            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0)
                total_pnl = milli_to_money(total_pnl_milli)
                trade_r_mult = calc_ratio_from_milli(total_pnl_milli, position.get('initial_risk_total_milli', 0))
                _append_pit_trade(pit_stats_builder, exit_date=Dates[j], pnl=total_pnl, r_mult=trade_r_mult)

        isSetup_prev = bool(buyCondition[j - 1]) and pos_start_of_current_bar == 0
        buyTriggered = False
        sizing_cap = None

        if isSetup_prev:
            sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            active_extended_signal = _pit_signal_state(
                buy_limits[j - 1],
                ATR_main[j - 1],
                ticker=resolved_ticker,
                security_profile=resolved_security_profile,
            )
            if _optimizer_limit_reachable_for_entry_day(L[j], O[j], V[j], buy_limits[j - 1]):
                order = _pit_build_normal_order(
                    buy_limits[j - 1],
                    ATR_main[j - 1],
                    sizing_cap,
                    params,
                    ticker=resolved_ticker,
                    security_profile=resolved_security_profile,
                    trade_date=Dates[j],
                )
                filled_position = _pit_try_entry(
                    order,
                    t_open=O[j],
                    t_high=H[j],
                    t_low=L[j],
                    t_close=C[j],
                    t_volume=V[j],
                    y_close=C[j - 1],
                    params=params,
                    entry_type='normal',
                    ticker=resolved_ticker,
                    security_profile=resolved_security_profile,
                    trade_date=Dates[j],
                )
                if filled_position is not None:
                    position = filled_position
                    currentCapital_milli -= position['net_buy_total_milli']
                    buyTriggered = True
                    active_extended_signal = None

        elif active_extended_signal is not None and pos_start_of_current_bar == 0:
            sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            extended_limit = _optimizer_extended_entry_limit(active_extended_signal)
            if _optimizer_limit_reachable_for_entry_day(L[j], O[j], V[j], extended_limit):
                order = _pit_build_extended_order(
                    active_extended_signal,
                    sizing_cap,
                    params,
                    y_close=C[j - 1],
                    ticker=resolved_ticker,
                    security_profile=active_extended_signal.get('security_profile'),
                    trade_date=Dates[j],
                    require_orderable=True,
                )
                filled_position = _pit_try_entry(
                    order,
                    t_open=O[j],
                    t_high=H[j],
                    t_low=L[j],
                    t_close=C[j],
                    t_volume=V[j],
                    y_close=C[j - 1],
                    params=params,
                    entry_type='extended',
                    ticker=resolved_ticker,
                    security_profile=active_extended_signal.get('security_profile'),
                    trade_date=Dates[j],
                )
                if filled_position is not None:
                    position = filled_position
                    currentCapital_milli -= position['net_buy_total_milli']
                    buyTriggered = True
                    active_extended_signal = None

        if not buyTriggered and int(position.get('qty', 0) or 0) == 0 and active_extended_signal is not None:
            if sizing_cap is None:
                sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            if _pit_should_clear_extended_signal(
                active_extended_signal,
                t_low=L[j],
                t_high=H[j],
                t_open=O[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                y_high=H[j - 1],
                y_atr=ATR_main[j - 1],
                y_ind_sell=sellCondition[j - 1],
                sizing_capital=sizing_cap,
                current_date=Dates[j],
                params=params,
            ):
                active_extended_signal = None

        j += 1

    if int(position.get('qty', 0) or 0) > 0:
        exec_price = adjust_long_sell_fill_price(C[-1], ticker=resolved_ticker)
        exec_price_milli = price_to_milli(exec_price)
        final_net_sell_milli = calc_sell_net_total_milli_from_milli(
            exec_price_milli,
            position['qty'],
            params,
            ticker=position.get('ticker', resolved_ticker),
            security_profile=position.get('security_profile'),
            trade_date=Dates[-1],
        )
        pnl_milli = final_net_sell_milli - position['remaining_cost_basis_milli']
        total_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0) + pnl_milli
        total_pnl = milli_to_money(total_pnl_milli)
        trade_r_mult = calc_ratio_from_milli(total_pnl_milli, position.get('initial_risk_total_milli', 0))
        _append_pit_trade(pit_stats_builder, exit_date=Dates[-1], pnl=total_pnl, r_mult=trade_r_mult)

    return _finalize_pit_stats_index(pit_stats_builder)


def run_v16_backtest(df, params=None, return_logs=False, precomputed_signals=None, ticker=None, collect_stats=True, return_pit_stats_index=False):
    if params is None:
        params = V16StrategyParams()

    resolved_ticker = ticker or df.attrs.get('ticker')
    resolved_security_profile = df.attrs.get('security_profile')

    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    Dates = df.index

    if precomputed_signals is None:
        ATR_main, buyCondition, sellCondition, buy_limits = generate_signals(df, params, ticker=resolved_ticker)
        precomputed_signals = (ATR_main, buyCondition, sellCondition, buy_limits)
    else:
        ATR_main, buyCondition, sellCondition, buy_limits = precomputed_signals

    if (not collect_stats) and return_pit_stats_index and (not return_logs):
        return None, _run_v16_pit_only_backtest(
            df,
            params,
            precomputed_signals,
            ticker=resolved_ticker,
        )

    position = {'qty': 0}
    active_extended_signal = None
    scanner_extended_signal = None
    pit_stats_builder = _create_pit_stats_builder() if return_pit_stats_index else None
    currentCapital_milli = money_to_milli(params.initial_capital)
    tradeCount, fullWins, missedBuyCount, missedSellCount = 0, 0, 0, 0
    totalProfit_milli, totalLoss_milli = 0, 0
    peakCapital_milli, maxDrawdownPct = currentCapital_milli, 0.0
    total_r_multiple, total_r_win, total_r_loss, total_bars_held = 0.0, 0.0, 0.0, 0
    trade_logs = []
    currentEquity_milli = currentCapital_milli
    collect_stats = bool(collect_stats)
    optimizer_setup_entry_positions = None
    if not collect_stats:
        valid_setup_mask = np.asarray(buyCondition[:-1], dtype=bool) & ~np.isnan(np.asarray(ATR_main[:-1], dtype=np.float64))
        optimizer_setup_entry_positions = np.flatnonzero(valid_setup_mask) + 1

    if len(C) == 0:
        stats_dict = build_backtest_stats(
            params=params,
            ticker=resolved_ticker,
            current_capital=milli_to_money(currentCapital_milli),
            current_equity=milli_to_money(currentEquity_milli),
            max_drawdown_pct=maxDrawdownPct,
            trade_count=tradeCount,
            full_wins=fullWins,
            total_profit=0.0,
            total_loss=0.0,
            total_r_multiple=total_r_multiple,
            total_r_win=total_r_win,
            total_r_loss=total_r_loss,
            missed_buy_count=missedBuyCount,
            missed_sell_count=missedSellCount,
            buy_condition_last=False,
            atr_last=float('nan'),
            close_last=float('nan'),
            low_last=float('nan'),
            had_open_position_at_end=False,
            active_extended_signal=None,
            end_position_qty=0,
            avg_bars_held=0,
            final_date=None,
            security_profile=resolved_security_profile,
        )
        stats_dict['is_candidate'] = False
        if return_pit_stats_index:
            pit_stats_index = _empty_pit_stats_index()
            if return_logs:
                return (stats_dict if collect_stats else None), trade_logs, pit_stats_index
            return (stats_dict if collect_stats else None), pit_stats_index
        if return_logs:
            return (stats_dict if collect_stats else None), trade_logs
        return stats_dict if collect_stats else None

    j = 1
    while j < len(C):
        if np.isnan(ATR_main[j - 1]):
            if collect_stats:
                currentEquity_milli = currentCapital_milli
                peakCapital_milli = max(peakCapital_milli, currentEquity_milli)
                currentDrawdownPct = ((peakCapital_milli - currentEquity_milli) / peakCapital_milli) * 100 if peakCapital_milli > 0 else 0.0
                maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)
            j += 1
            continue

        pos_start_of_current_bar = position['qty']
        if (
            not collect_stats
            and pos_start_of_current_bar == 0
            and active_extended_signal is None
            and not bool(buyCondition[j - 1])
        ):
            next_setup_cursor = np.searchsorted(optimizer_setup_entry_positions, j + 1) if optimizer_setup_entry_positions is not None else 0
            if optimizer_setup_entry_positions is not None and next_setup_cursor < len(optimizer_setup_entry_positions):
                j = int(optimizer_setup_entry_positions[next_setup_cursor])
                continue
            break

        if pos_start_of_current_bar > 0:
            if collect_stats:
                total_bars_held += 1
            position, freed_cash_milli, _pnl_realized_milli, events = execute_bar_step(
                position,
                ATR_main[j - 1],
                sellCondition[j - 1],
                C[j - 1],
                O[j],
                H[j],
                L[j],
                C[j],
                V[j],
                params,
                current_date=Dates[j],
                y_high=H[j - 1],
                return_milli=True,
                record_exec_contexts=False,
                sync_display_fields=collect_stats or return_logs,
            )
            currentCapital_milli += freed_cash_milli
            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0)
                total_pnl = position['realized_pnl'] if (collect_stats or return_logs) else milli_to_money(total_pnl_milli)
                trade_r_mult = calc_ratio_from_milli(total_pnl_milli, position.get('initial_risk_total_milli', 0))
                if collect_stats:
                    total_r_multiple += trade_r_mult
                    tradeCount += 1
                if return_logs:
                    trade_logs.append({'exit_date': Dates[j], 'pnl': total_pnl, 'r_mult': trade_r_mult})
                if pit_stats_builder is not None:
                    _append_pit_trade(pit_stats_builder, exit_date=Dates[j], pnl=total_pnl, r_mult=trade_r_mult)
                if collect_stats:
                    if position['realized_pnl_milli'] > 0:
                        fullWins += 1
                        totalProfit_milli += position['realized_pnl_milli']
                        total_r_win += trade_r_mult
                    else:
                        totalLoss_milli += abs(position['realized_pnl_milli'])
                        total_r_loss += abs(trade_r_mult)
            elif 'MISSED_SELL' in events and collect_stats:
                missedSellCount += 1

        isSetup_prev = bool(buyCondition[j - 1]) and (pos_start_of_current_bar == 0)
        buyTriggered = False
        sizing_cap = None

        if isSetup_prev:
            sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            signal_state = create_signal_tracking_state(
                buy_limits[j - 1],
                ATR_main[j - 1],
                params,
                ticker=resolved_ticker,
                security_profile=resolved_security_profile,
            )
            if signal_state is not None:
                active_extended_signal = signal_state
                if collect_stats:
                    scanner_extended_signal = create_signal_tracking_state(
                        buy_limits[j - 1],
                        ATR_main[j - 1],
                        params,
                        ticker=resolved_ticker,
                        security_profile=resolved_security_profile,
                    )

            should_try_normal_entry = collect_stats or _optimizer_limit_reachable_for_entry_day(
                L[j], O[j], V[j], buy_limits[j - 1]
            )
            if should_try_normal_entry:
                entry_plan = build_normal_entry_plan(
                    buy_limits[j - 1],
                    ATR_main[j - 1],
                    sizing_cap,
                    params,
                    ticker=resolved_ticker,
                    security_profile=resolved_security_profile,
                    trade_date=Dates[j],
                )
                entry_result = execute_pre_market_entry_plan(
                    entry_plan=entry_plan,
                    t_open=O[j],
                    t_high=H[j],
                    t_low=L[j],
                    t_close=C[j],
                    t_volume=V[j],
                    y_close=C[j - 1],
                    params=params,
                    entry_type='normal',
                    ticker=resolved_ticker,
                    trade_date=Dates[j],
                )
                entry_filled = bool(entry_result['filled'])
                entry_count_as_missed_buy = bool(entry_result['count_as_missed_buy'])
            else:
                entry_filled = False
                entry_count_as_missed_buy = False
            if entry_filled:
                filled_signal_state = active_extended_signal
                position = entry_result['position']
                currentCapital_milli -= position['net_buy_total_milli']
                buyTriggered = True
                active_extended_signal = None
            elif entry_count_as_missed_buy and collect_stats:
                missedBuyCount += 1

        elif active_extended_signal is not None and pos_start_of_current_bar == 0:
            sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            should_try_extended_entry = collect_stats or _optimizer_limit_reachable_for_entry_day(
                L[j], O[j], V[j], _optimizer_extended_entry_limit(active_extended_signal)
            )
            if should_try_extended_entry:
                entry_plan = build_extended_entry_plan_from_signal(
                    active_extended_signal,
                    sizing_cap,
                    params,
                    y_close=C[j - 1],
                    ticker=resolved_ticker,
                    security_profile=active_extended_signal.get('security_profile'),
                    trade_date=Dates[j],
                )
                entry_result = execute_pre_market_entry_plan(
                    entry_plan=entry_plan,
                    t_open=O[j],
                    t_high=H[j],
                    t_low=L[j],
                    t_close=C[j],
                    t_volume=V[j],
                    y_close=C[j - 1],
                    params=params,
                    entry_type='extended',
                    ticker=resolved_ticker,
                    trade_date=Dates[j],
                )
                entry_filled = bool(entry_result['filled'])
                entry_count_as_missed_buy = bool(entry_result['count_as_missed_buy'])
            else:
                entry_filled = False
                entry_count_as_missed_buy = False
            if entry_filled:
                filled_signal_state = active_extended_signal
                position = entry_result['position']
                currentCapital_milli -= position['net_buy_total_milli']
                buyTriggered = True
                active_extended_signal = None
            elif entry_count_as_missed_buy and collect_stats:
                missedBuyCount += 1

        if not buyTriggered and position['qty'] == 0 and active_extended_signal is not None:
            if sizing_cap is None:
                sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            if should_clear_extended_signal(
                active_extended_signal,
                L[j],
                H[j],
                t_open=O[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                y_high=H[j - 1],
                y_atr=ATR_main[j - 1],
                y_ind_sell=sellCondition[j - 1],
                sizing_capital=sizing_cap,
                current_date=Dates[j],
                params=params,
                copy_shadow_position=collect_stats,
            ):
                active_extended_signal = None

        if collect_stats and scanner_extended_signal is not None:
            if sizing_cap is None:
                sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            should_clear_scanner_extended = should_clear_extended_signal(
                scanner_extended_signal,
                L[j],
                H[j],
                t_open=O[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                y_high=H[j - 1],
                y_atr=ATR_main[j - 1],
                y_ind_sell=sellCondition[j - 1],
                sizing_capital=sizing_cap,
                current_date=Dates[j],
                params=params,
            )
            if should_clear_scanner_extended:
                scanner_extended_signal = None

        if collect_stats:
            currentEquity_milli = currentCapital_milli
            if position['qty'] > 0:
                floating_exec_price = adjust_long_sell_fill_price(C[j], ticker=resolved_ticker)
                floating_sell_ledger = build_sell_ledger_from_price(
                    floating_exec_price,
                    position['qty'],
                    params,
                    ticker=position.get('ticker', resolved_ticker),
                    security_profile=position.get('security_profile'),
                    trade_date=Dates[j],
                )
                currentEquity_milli = currentCapital_milli + floating_sell_ledger['net_sell_total_milli']

            peakCapital_milli = max(peakCapital_milli, currentEquity_milli)
            currentDrawdownPct = ((peakCapital_milli - currentEquity_milli) / peakCapital_milli) * 100 if peakCapital_milli > 0 else 0.0
            maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

        j += 1

    final_state = finalize_open_position_at_end(
        position=position,
        ticker=resolved_ticker,
        final_close=C[-1],
        final_date=Dates[-1],
        current_capital_milli=currentCapital_milli,
        current_equity_milli=currentEquity_milli,
        peak_capital_milli=peakCapital_milli,
        max_drawdown_pct=maxDrawdownPct,
        trade_count=tradeCount,
        full_wins=fullWins,
        total_profit_milli=totalProfit_milli,
        total_loss_milli=totalLoss_milli,
        total_r_multiple=total_r_multiple,
        total_r_win=total_r_win,
        total_r_loss=total_r_loss,
        trade_logs=trade_logs,
        return_logs=return_logs,
        params=params,
        collect_stats=collect_stats,
    )
    currentCapital_milli = final_state['current_capital_milli']
    currentEquity_milli = final_state['current_equity_milli']
    maxDrawdownPct = final_state['max_drawdown_pct']
    tradeCount = final_state['trade_count']
    fullWins = final_state['full_wins']
    totalProfit_milli = final_state['total_profit_milli']
    totalLoss_milli = final_state['total_loss_milli']
    total_r_multiple = final_state['total_r_multiple']
    total_r_win = final_state['total_r_win']
    total_r_loss = final_state['total_r_loss']
    had_open_position_at_end = final_state['had_open_position_at_end']
    end_position_qty = final_state['end_position_qty']
    trade_logs = final_state['trade_logs']
    if pit_stats_builder is not None and final_state.get('final_trade_exit_date') is not None:
        _append_pit_trade(
            pit_stats_builder,
            exit_date=final_state['final_trade_exit_date'],
            pnl=final_state['final_trade_pnl'],
            r_mult=final_state['final_trade_r_mult'],
        )
    pit_stats_index = _finalize_pit_stats_index(pit_stats_builder) if return_pit_stats_index else None

    if not collect_stats:
        if return_pit_stats_index:
            if return_logs:
                return None, trade_logs, pit_stats_index
            return None, pit_stats_index
        if return_logs:
            return None, trade_logs
        return None

    avg_bars_held = total_bars_held / tradeCount if tradeCount > 0 else 0
    stats_dict = build_backtest_stats(
        params=params,
        ticker=resolved_ticker,
        current_capital=milli_to_money(currentCapital_milli),
        current_equity=milli_to_money(currentEquity_milli),
        max_drawdown_pct=maxDrawdownPct,
        trade_count=tradeCount,
        full_wins=fullWins,
        total_profit=milli_to_money(totalProfit_milli),
        total_loss=milli_to_money(totalLoss_milli),
        total_r_multiple=total_r_multiple,
        total_r_win=total_r_win,
        total_r_loss=total_r_loss,
        missed_buy_count=missedBuyCount,
        missed_sell_count=missedSellCount,
        buy_condition_last=buyCondition[-1],
        atr_last=ATR_main[-1],
        close_last=C[-1],
        low_last=L[-1],
        had_open_position_at_end=had_open_position_at_end,
        active_extended_signal=active_extended_signal,
        active_extended_signal_tbd=scanner_extended_signal,
        end_position_qty=end_position_qty,
        avg_bars_held=avg_bars_held,
        final_date=Dates[-1],
        security_profile=resolved_security_profile,
    )

    if return_pit_stats_index:
        if return_logs:
            return stats_dict, trade_logs, pit_stats_index
        return stats_dict, pit_stats_index
    if return_logs:
        return stats_dict, trade_logs
    return stats_dict
