import pandas as pd

from core.entry_plans import resize_candidate_plan_to_capital
from core.price_utils import (
    adjust_long_stop_price,
    adjust_long_target_price,
    is_limit_buy_price_reachable_for_day,
)


# # (AI註: 單一真理來源 - 延續訊號何時失效統一由此判斷；A2 規格採 frozen stop/target barrier，於當日收盤後確認、次日盤前清除)
def should_clear_extended_signal(signal_state, t_low, t_high=None):
    if signal_state is None:
        return False

    stop_hit = (not pd.isna(t_low)) and t_low <= signal_state["init_sl"]
    target_price = signal_state.get("target_price", pd.NA)
    target_hit = (not pd.isna(t_high)) and (not pd.isna(target_price)) and t_high >= target_price
    return bool(stop_hit or target_hit)


# # (AI註: 單一真理來源 - 延續候選原始訊號狀態統一由此建立；signal day 凍結 L/S/T 與 trail，不得每日重算)
def create_signal_tracking_state(original_limit, atr, params):
    if pd.isna(original_limit) or pd.isna(atr):
        return None

    init_sl = adjust_long_stop_price(original_limit - atr * params.atr_times_init)
    init_trail = adjust_long_stop_price(original_limit - atr * params.atr_times_trail)
    target_price = adjust_long_target_price(original_limit + (original_limit - init_sl))
    return {
        "orig_limit": original_limit,
        "orig_atr": atr,
        "init_sl": init_sl,
        "init_trail": init_trail,
        "target_price": target_price,
    }


# # (AI註: 單一真理來源 - 延續候選固定 frozen plan；僅依 sizing 重算 qty/is_orderable，不得再以 reference price 改寫 L/S/T)
def build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params):
    if signal_state is None:
        return None

    base_plan = {
        "limit_price": signal_state["orig_limit"],
        "init_sl": signal_state["init_sl"],
        "init_trail": signal_state["init_trail"],
        "orig_limit": signal_state["orig_limit"],
        "orig_atr": signal_state["orig_atr"],
        "target_price": signal_state["target_price"],
    }
    return resize_candidate_plan_to_capital(base_plan, sizing_capital, params)


# # (AI註: 單一真理來源 - 延續訊號今日能否掛單；signal_valid 與 today_orderable 分層，固定 L 若低於今日跌停價則不得進 orderable list)
def is_extended_signal_orderable_for_day(signal_state, candidate_plan, y_close):
    if signal_state is None or candidate_plan is None:
        return False
    if candidate_plan.get("qty", 0) <= 0:
        return False
    return is_limit_buy_price_reachable_for_day(candidate_plan["limit_price"], y_close)


# # (AI註: 單一真理來源 - 延續單實際掛單規格；僅接受 frozen 候選且今日價格帶仍可達)
def build_extended_entry_plan_from_signal(signal_state, sizing_capital, params, *, y_close):
    candidate_plan = build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params)
    if not is_extended_signal_orderable_for_day(signal_state, candidate_plan, y_close):
        return None
    return candidate_plan


# # (AI註: 延續候選資格評估)
def evaluate_extended_candidate_eligibility(original_limit, atr, sizing_capital, params):
    signal_state = create_signal_tracking_state(original_limit, atr, params)
    return build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params)
