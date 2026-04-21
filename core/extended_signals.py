
import copy
import numpy as np
import pandas as pd

from core.entry_plans import resize_candidate_plan_to_capital
from core.price_utils import (
    adjust_long_buy_fill_price,
    calc_frozen_target_price,
    calc_initial_stop_from_reference,
    calc_initial_trailing_stop_from_reference,
    is_limit_buy_price_reachable_for_day,
)


def _has_counterfactual_anchor(signal_state):
    if signal_state is None:
        return False
    return not pd.isna(signal_state.get("entry_ref_price", np.nan))


# # (AI註: 單一真理來源 - 延續候選只在首個可成交反事實進場日固定 entry ref 與失效 / 達標 barrier，不得每日漂移)
def ensure_extended_signal_counterfactual_anchor(signal_state, *, t_open, t_low, params):
    if signal_state is None or params is None:
        return signal_state
    if _has_counterfactual_anchor(signal_state):
        return signal_state

    original_limit = signal_state.get("orig_limit", np.nan)
    original_atr = signal_state.get("orig_atr", np.nan)
    if pd.isna(original_limit) or pd.isna(original_atr) or pd.isna(t_open) or pd.isna(t_low):
        return signal_state
    if t_low > original_limit:
        return signal_state

    resolved_ticker = signal_state.get("ticker")
    resolved_security_profile = signal_state.get("security_profile")
    entry_ref_price = adjust_long_buy_fill_price(min(t_open, original_limit), ticker=resolved_ticker, security_profile=resolved_security_profile)
    invalidation_barrier = calc_initial_stop_from_reference(entry_ref_price, original_atr, params, ticker=resolved_ticker, security_profile=resolved_security_profile)
    completion_barrier = calc_frozen_target_price(entry_ref_price, invalidation_barrier, ticker=resolved_ticker, security_profile=resolved_security_profile)

    signal_state["entry_ref_price"] = entry_ref_price
    signal_state["continuation_invalidation_barrier"] = invalidation_barrier
    signal_state["continuation_completion_barrier"] = completion_barrier
    return signal_state


# # (AI註: 單一真理來源 - 延續候選何時失效統一由此判斷；僅在已建立固定反事實 barrier 後，才可用 Low/High inclusive hit 清除)
def should_clear_extended_signal(signal_state, t_low, t_high=None, *, t_open=np.nan, params=None):
    if signal_state is None:
        return False

    if params is not None:
        ensure_extended_signal_counterfactual_anchor(signal_state, t_open=t_open, t_low=t_low, params=params)

    invalidation_barrier = signal_state.get("continuation_invalidation_barrier", np.nan)
    completion_barrier = signal_state.get("continuation_completion_barrier", np.nan)
    stop_hit = (not pd.isna(t_low)) and (not pd.isna(invalidation_barrier)) and t_low <= invalidation_barrier
    target_hit = (not pd.isna(t_high)) and (not pd.isna(completion_barrier)) and t_high >= completion_barrier
    return bool(stop_hit or target_hit)


# # (AI註: 單一真理來源 - 延續候選原始訊號狀態統一由此建立；未成交前只保留 setup anchor / limit 與可選的固定反事實 barrier)
def create_signal_tracking_state(original_limit, atr, params, ticker=None, security_profile=None):
    if pd.isna(original_limit) or pd.isna(atr):
        return None

    return {
        "orig_limit": original_limit,
        "orig_atr": atr,
        "entry_ref_price": np.nan,
        "continuation_invalidation_barrier": np.nan,
        "continuation_completion_barrier": np.nan,
        "ticker": ticker,
        "security_profile": security_profile,
    }


def create_extended_tbd_tracking_state(signal_state, shadow_position):
    if signal_state is None or shadow_position is None:
        return None
    if int(shadow_position.get('qty', 0) or 0) <= 0:
        return None
    return {
        'signal_state': copy.deepcopy(signal_state),
        'shadow_position': copy.deepcopy(shadow_position),
        'ticker': signal_state.get('ticker') or shadow_position.get('ticker'),
        'security_profile': signal_state.get('security_profile') or shadow_position.get('security_profile'),
    }


def _resolve_tbd_signal_state(tbd_state):
    if tbd_state is None:
        return None
    return tbd_state.get('signal_state')


def _resolve_tbd_shadow_position(tbd_state):
    if tbd_state is None:
        return None
    return tbd_state.get('shadow_position')


def is_extended_tbd_shadow_alive(tbd_state):
    shadow_position = _resolve_tbd_shadow_position(tbd_state)
    if shadow_position is None:
        return False
    return int(shadow_position.get('qty', 0) or 0) > 0


def update_extended_tbd_shadow_trade_for_bar(
    tbd_state,
    *,
    y_atr,
    y_ind_sell,
    y_close,
    y_high,
    t_open,
    t_high,
    t_low,
    t_close,
    t_volume,
    params,
    current_date=None,
):
    if tbd_state is None or not is_extended_tbd_shadow_alive(tbd_state):
        return None

    from core.position_step import execute_bar_step

    shadow_position = copy.deepcopy(_resolve_tbd_shadow_position(tbd_state))
    shadow_position, _freed_cash, _pnl_realized, _events = execute_bar_step(
        shadow_position,
        y_atr,
        y_ind_sell,
        y_close,
        t_open,
        t_high,
        t_low,
        t_close,
        t_volume,
        params,
        current_date=current_date,
        y_high=y_high,
    )
    if int(shadow_position.get('qty', 0) or 0) <= 0:
        return None

    updated_state = dict(tbd_state)
    updated_state['shadow_position'] = shadow_position
    return updated_state


# # (AI註: 單一真理來源 - 延續候選掛單只延續 setup 的可掛單資格；若已建立反事實 entry ref，後續 limit 固定為該 entry ref，避免每日追價漂移)
def build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params, ticker=None, security_profile=None, trade_date=None):
    if signal_state is None:
        return None

    resolved_ticker = ticker or signal_state.get("ticker")
    resolved_security_profile = security_profile or signal_state.get("security_profile")
    limit_reference = signal_state.get("entry_ref_price", np.nan)
    if pd.isna(limit_reference):
        limit_reference = signal_state.get("orig_limit", np.nan)
    entry_atr = signal_state.get("orig_atr", np.nan)
    if pd.isna(limit_reference) or pd.isna(entry_atr):
        return None

    sizing_stop_ref = calc_initial_stop_from_reference(limit_reference, entry_atr, params, ticker=resolved_ticker, security_profile=resolved_security_profile)
    sizing_trail_ref = calc_initial_trailing_stop_from_reference(limit_reference, entry_atr, params, ticker=resolved_ticker, security_profile=resolved_security_profile)
    target_price = calc_frozen_target_price(limit_reference, sizing_stop_ref, ticker=resolved_ticker, security_profile=resolved_security_profile)
    base_plan = {
        "limit_price": limit_reference,
        "init_sl": sizing_stop_ref,
        "init_trail": sizing_trail_ref,
        "orig_limit": signal_state.get("orig_limit", np.nan),
        "orig_atr": entry_atr,
        "target_price": target_price,
        "entry_atr": entry_atr,
        "entry_ref_price": signal_state.get("entry_ref_price", np.nan),
        "continuation_invalidation_barrier": signal_state.get("continuation_invalidation_barrier", np.nan),
        "continuation_completion_barrier": signal_state.get("continuation_completion_barrier", np.nan),
        "ticker": resolved_ticker,
        "security_profile": resolved_security_profile,
        "trade_date": trade_date,
    }
    return resize_candidate_plan_to_capital(base_plan, sizing_capital, params)


def build_extended_tbd_candidate_plan_from_state(tbd_state, sizing_capital, params, ticker=None, security_profile=None, trade_date=None):
    if tbd_state is None or not is_extended_tbd_shadow_alive(tbd_state):
        return None
    signal_state = _resolve_tbd_signal_state(tbd_state)
    shadow_position = _resolve_tbd_shadow_position(tbd_state)
    if signal_state is None or shadow_position is None:
        return None

    candidate_plan = build_extended_candidate_plan_from_signal(
        signal_state,
        sizing_capital,
        params,
        ticker=ticker or tbd_state.get('ticker'),
        security_profile=security_profile or tbd_state.get('security_profile'),
        trade_date=trade_date,
    )
    if candidate_plan is None:
        return None

    enriched_plan = dict(candidate_plan)
    enriched_plan['shadow_sl'] = shadow_position.get('sl', np.nan)
    enriched_plan['shadow_initial_stop'] = shadow_position.get('initial_stop', np.nan)
    enriched_plan['shadow_trailing_stop'] = shadow_position.get('trailing_stop', np.nan)
    enriched_plan['shadow_tp_half'] = shadow_position.get('tp_half', np.nan)
    enriched_plan['shadow_sold_half'] = bool(shadow_position.get('sold_half', False))
    enriched_plan['shadow_entry_price'] = shadow_position.get('entry_fill_price', np.nan)
    return enriched_plan


# # (AI註: 單一真理來源 - 延續訊號今日能否掛單；signal_valid 與 today_orderable 分層，固定 limit 若低於今日跌停價則不得進 orderable list)
def is_extended_signal_orderable_for_day(signal_state, candidate_plan, y_close, ticker=None, security_profile=None):
    if signal_state is None or candidate_plan is None:
        return False
    if candidate_plan.get("qty", 0) <= 0:
        return False
    resolved_ticker = ticker or candidate_plan.get("ticker") or signal_state.get("ticker")
    resolved_security_profile = security_profile or candidate_plan.get("security_profile") or signal_state.get("security_profile")
    return is_limit_buy_price_reachable_for_day(candidate_plan["limit_price"], y_close, ticker=resolved_ticker, security_profile=resolved_security_profile)


def is_extended_tbd_orderable_for_day(tbd_state, candidate_plan, y_close, ticker=None, security_profile=None):
    if tbd_state is None or not is_extended_tbd_shadow_alive(tbd_state):
        return False
    signal_state = _resolve_tbd_signal_state(tbd_state)
    return is_extended_signal_orderable_for_day(
        signal_state,
        candidate_plan,
        y_close,
        ticker=ticker or tbd_state.get('ticker'),
        security_profile=security_profile or tbd_state.get('security_profile'),
    )


# # (AI註: 單一真理來源 - 延續單實際掛單規格；僅接受有效候選且今日價格帶仍可達)
def build_extended_entry_plan_from_signal(signal_state, sizing_capital, params, *, y_close, ticker=None, security_profile=None, trade_date=None):
    candidate_plan = build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params, ticker=ticker, security_profile=security_profile, trade_date=trade_date)
    if not is_extended_signal_orderable_for_day(signal_state, candidate_plan, y_close, ticker=ticker, security_profile=security_profile):
        return None
    return candidate_plan


# # (AI註: 延續候選資格評估)
def evaluate_extended_candidate_eligibility(original_limit, atr, sizing_capital, params, ticker=None, security_profile=None, trade_date=None):
    signal_state = create_signal_tracking_state(original_limit, atr, params, ticker=ticker, security_profile=security_profile)
    return build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params, ticker=ticker, security_profile=security_profile, trade_date=trade_date)
