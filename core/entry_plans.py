import numpy as np
import pandas as pd

from core.price_utils import (
    adjust_long_buy_fill_price,
    adjust_long_stop_price,
    calc_entry_price,
    calc_initial_risk_total,
    calc_net_sell_price,
    calc_position_size,
    calc_frozen_target_price,
    is_locked_limit_up_bar,
)


# # (AI註: 單一真理來源 - 候選單在不同資金上限下的 qty / is_orderable 重算統一由此處理)
def resize_candidate_plan_to_capital(candidate_plan, sizing_capital, params):
    if candidate_plan is None:
        return None

    limit_price = candidate_plan.get("limit_price")
    init_sl = candidate_plan.get("init_sl")
    if pd.isna(limit_price) or pd.isna(init_sl):
        return None

    resized_plan = dict(candidate_plan)
    qty = calc_position_size(limit_price, init_sl, sizing_capital, params.fixed_risk, params)
    resized_plan["qty"] = qty
    resized_plan["is_orderable"] = qty > 0
    return resized_plan


# # (AI註: 單一真理來源 - 盤前依當下可用現金重算有效掛單規格與保留資金)
def build_cash_capped_entry_plan(candidate_plan, available_cash, params):
    resized_plan = resize_candidate_plan_to_capital(candidate_plan, available_cash, params)
    if resized_plan is None or resized_plan["qty"] <= 0:
        return None

    reserved_cost = calc_entry_price(resized_plan["limit_price"], resized_plan["qty"], params) * resized_plan["qty"]
    if pd.isna(reserved_cost) or reserved_cost > available_cash:
        return None

    resized_plan["reserved_cost"] = reserved_cost
    return resized_plan


# # (AI註: 單一真理來源 - 正常候選的 limit/sl/trail/qty 規格統一由此產生；候選資格與可掛單分離)
def build_normal_candidate_plan(limit_price, atr, sizing_capital, params):
    if pd.isna(limit_price) or pd.isna(atr):
        return None

    init_sl = adjust_long_stop_price(limit_price - atr * params.atr_times_init)
    init_trail = adjust_long_stop_price(limit_price - atr * params.atr_times_trail)
    target_price = calc_frozen_target_price(limit_price, init_sl)
    base_plan = {
        "limit_price": limit_price,
        "init_sl": init_sl,
        "init_trail": init_trail,
        "target_price": target_price,
    }
    return resize_candidate_plan_to_capital(base_plan, sizing_capital, params)


# # (AI註: 單一真理來源 - 正常單實際掛單規格；僅接受可掛單候選)
def build_normal_entry_plan(limit_price, atr, sizing_capital, params):
    candidate_plan = build_normal_candidate_plan(limit_price, atr, sizing_capital, params)
    if candidate_plan is None or candidate_plan["qty"] <= 0:
        return None
    return candidate_plan


# # (AI註: miss buy 正式定義單一真理來源 - 必須先有有效限價買單，且不能是先達停損而放棄進場)
def should_count_miss_buy(order_qty, is_worse_than_initial_stop=False):
    if order_qty is None or order_qty <= 0:
        return False
    if is_worse_than_initial_stop:
        return False
    return True


def should_count_normal_miss_buy(order_qty, is_worse_than_initial_stop=False):
    return should_count_miss_buy(order_qty, is_worse_than_initial_stop=is_worse_than_initial_stop)


# # (AI註: 單一真理來源 - 成交後的部位欄位統一由此建立)
def build_position_from_entry_fill(buy_price, qty, init_sl, init_trail, params, entry_type="normal", *, target_price=None, limit_price=None):
    if pd.isna(buy_price) or buy_price <= 0 or qty <= 0:
        raise ValueError(f"build_position_from_entry_fill 需要有效 buy_price/qty，收到 buy_price={buy_price!r}, qty={qty!r}")

    entry_price = calc_entry_price(buy_price, qty, params)
    net_sl = calc_net_sell_price(init_sl, qty, params)
    frozen_target_price = target_price
    if frozen_target_price is None or pd.isna(frozen_target_price):
        target_basis_limit = limit_price if limit_price is not None else buy_price
        frozen_target_price = calc_frozen_target_price(target_basis_limit, init_sl)
    init_risk = calc_initial_risk_total(entry_price, net_sl, qty, params)

    return {
        "qty": qty,
        "entry": entry_price,
        "sl": init_sl,
        "initial_stop": init_sl,
        "trailing_stop": init_trail,
        "tp_half": frozen_target_price,
        "sold_half": False,
        "pure_buy_price": buy_price,
        "realized_pnl": 0.0,
        "initial_risk_total": init_risk,
        "entry_type": entry_type,
    }


# # (AI註: 單一真理來源 - 盤前有效買單的當日成交 / miss buy / 先達停損放棄進場邏輯統一由此判斷)
def execute_pre_market_entry_plan(entry_plan, t_open, t_high, t_low, t_close, t_volume, y_close, params, entry_type="normal"):
    result = {
        "filled": False,
        "count_as_missed_buy": False,
        "is_worse_than_initial_stop": False,
        "is_locked_limit_up": False,
        "buy_price": np.nan,
        "entry_price": np.nan,
        "tp_half": np.nan,
        "position": None,
        "entry_type": entry_type,
    }
    if entry_plan is None:
        return result

    qty = entry_plan.get("qty", 0)
    result["count_as_missed_buy"] = should_count_miss_buy(qty)
    result["is_locked_limit_up"] = is_locked_limit_up_bar(t_open, t_high, t_low, t_close, y_close)

    if qty <= 0:
        result["count_as_missed_buy"] = False
        return result

    if pd.isna(t_volume) or t_volume <= 0 or pd.isna(t_open) or pd.isna(t_low) or result["is_locked_limit_up"]:
        return result

    limit_price = entry_plan["limit_price"]
    init_sl = entry_plan["init_sl"]
    init_trail = entry_plan["init_trail"]

    if t_low > limit_price:
        return result

    buy_price = adjust_long_buy_fill_price(min(t_open, limit_price))
    result["buy_price"] = buy_price

    if buy_price <= init_sl:
        result["is_worse_than_initial_stop"] = True
        result["count_as_missed_buy"] = False
        return result

    position = build_position_from_entry_fill(
        buy_price=buy_price,
        qty=qty,
        init_sl=init_sl,
        init_trail=init_trail,
        params=params,
        entry_type=entry_type,
        target_price=entry_plan.get("target_price"),
        limit_price=limit_price,
    )
    result["filled"] = True
    result["count_as_missed_buy"] = False
    result["position"] = position
    result["entry_price"] = position["entry"]
    result["tp_half"] = position["tp_half"]
    return result
