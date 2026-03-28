import pandas as pd

from core.v16_entry_plans import resize_candidate_plan_to_capital
from core.v16_price_utils import adjust_long_buy_limit, adjust_long_stop_price


# # (AI註: 單一真理來源 - 延續訊號何時失效統一由此判斷)
def should_clear_extended_signal(signal_state, t_low):
    if signal_state is None or pd.isna(t_low):
        return False
    return t_low <= signal_state["init_sl"]


# # (AI註: 單一真理來源 - 延續候選原始訊號狀態統一由此建立)
def create_signal_tracking_state(original_limit, atr, params):
    if pd.isna(original_limit) or pd.isna(atr):
        return None

    init_sl = adjust_long_stop_price(original_limit - atr * params.atr_times_init)
    return {
        "orig_limit": original_limit,
        "orig_atr": atr,
        "init_sl": init_sl,
    }


# # (AI註: 單一真理來源 - 延續候選每日盤前資格規格統一由此產生；必須仍在原始買入區間內)
def build_extended_candidate_plan_from_signal(signal_state, reference_price, sizing_capital, params):
    if signal_state is None or pd.isna(reference_price):
        return None

    original_limit = signal_state["orig_limit"]
    atr = signal_state["orig_atr"]
    init_sl = signal_state["init_sl"]
    if not (init_sl < reference_price <= original_limit):
        return None

    limit_price = adjust_long_buy_limit(reference_price)
    if pd.isna(limit_price) or not (init_sl < limit_price <= original_limit):
        return None

    init_trail = adjust_long_stop_price(limit_price - atr * params.atr_times_trail)
    base_plan = {
        "limit_price": limit_price,
        "init_sl": init_sl,
        "init_trail": init_trail,
        "orig_limit": original_limit,
        "orig_atr": atr,
    }
    return resize_candidate_plan_to_capital(base_plan, sizing_capital, params)


# # (AI註: 單一真理來源 - 延續單實際掛單規格；僅接受可掛單候選)
def build_extended_entry_plan_from_signal(signal_state, reference_price, sizing_capital, params):
    candidate_plan = build_extended_candidate_plan_from_signal(signal_state, reference_price, sizing_capital, params)
    if candidate_plan is None or candidate_plan["qty"] <= 0:
        return None
    return candidate_plan


# # (AI註: 延續候選資格評估)
def evaluate_extended_candidate_eligibility(close_price, original_limit, atr, sizing_capital, params):
    signal_state = create_signal_tracking_state(original_limit, atr, params)
    return build_extended_candidate_plan_from_signal(signal_state, close_price, sizing_capital, params)
