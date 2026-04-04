import math

import numpy as np
import pandas as pd


def tv_round(number):
    return math.floor(number + 0.5)


def get_tick_size(price):
    if price < 1:
        return 0.001
    elif price < 10:
        return 0.01
    elif price < 50:
        return 0.05
    elif price < 100:
        return 0.1
    elif price < 500:
        return 0.5
    elif price < 1000:
        return 1.0
    else:
        return 5.0


# # (AI註: 第12點 - 跳價取整方向統一收斂到單一函式，避免買/賣/停損/停利各自手寫)
def round_to_tick(price, direction="nearest"):
    if pd.isna(price):
        return np.nan
    tick = get_tick_size(price)
    ratio = price / tick
    if direction == "up":
        return math.ceil(ratio - 1e-12) * tick
    if direction == "down":
        return math.floor(ratio + 1e-12) * tick
    return tv_round(ratio) * tick


def adjust_to_tick(price):
    return round_to_tick(price, direction="nearest")


def adjust_price_up_to_tick(price):
    return round_to_tick(price, direction="up")


def adjust_price_down_to_tick(price):
    return round_to_tick(price, direction="down")


# # (AI註: 長倉語義化封裝 - 買單/賣單/停損/停利各自固定方向，避免保守性漂移)
def adjust_long_buy_limit(price):
    return adjust_price_down_to_tick(price)


def adjust_long_stop_price(price):
    return adjust_price_up_to_tick(price)


def adjust_long_target_price(price):
    return adjust_price_up_to_tick(price)


def adjust_long_buy_fill_price(price):
    return adjust_price_up_to_tick(price)


def adjust_long_sell_fill_price(price):
    return adjust_price_down_to_tick(price)


def get_tick_size_array(prices):
    prices = np.asarray(prices, dtype=np.float64)
    ticks = np.full(prices.shape, 5.0, dtype=np.float64)
    ticks[prices < 1000] = 1.0
    ticks[prices < 500] = 0.5
    ticks[prices < 100] = 0.1
    ticks[prices < 50] = 0.05
    ticks[prices < 10] = 0.01
    ticks[prices < 1] = 0.001
    return ticks


def round_to_tick_array(prices, direction="nearest"):
    prices = np.asarray(prices, dtype=np.float64)
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(prices)
    if not np.any(valid):
        return out
    valid_prices = prices[valid]
    ticks = get_tick_size_array(valid_prices)
    ratios = valid_prices / ticks
    if direction == "up":
        out[valid] = np.ceil(ratios - 1e-12) * ticks
    elif direction == "down":
        out[valid] = np.floor(ratios + 1e-12) * ticks
    else:
        out[valid] = np.floor(ratios + 0.5) * ticks
    return out


def adjust_to_tick_array(prices):
    return round_to_tick_array(prices, direction="nearest")


def adjust_long_buy_limit_array(prices):
    return round_to_tick_array(prices, direction="down")


def calc_entry_price(bPrice, bQty, params):
    if pd.isna(bPrice) or pd.isna(bQty) or bPrice <= 0 or bQty <= 0:
        return np.nan
    fee = max(bPrice * bQty * params.buy_fee, params.min_fee)
    return bPrice + (fee / bQty)


def calc_net_sell_price(sPrice, sQty, params):
    if pd.isna(sPrice) or pd.isna(sQty) or sPrice <= 0 or sQty <= 0:
        return np.nan
    fee = max(sPrice * sQty * params.sell_fee, params.min_fee)
    tax = sPrice * sQty * params.tax_rate
    return sPrice - ((fee + tax) / sQty)


def calc_position_size(bPrice, stopPrice, cap, riskPct, params):
    if pd.isna(bPrice) or pd.isna(stopPrice) or bPrice <= 0 or stopPrice <= 0:
        return 0
    max_risk_amount = cap * riskPct
    estEntryCost_unit = bPrice * (1 + params.buy_fee)
    estExitNet_unit = stopPrice * (1 - params.sell_fee - params.tax_rate)
    riskPerUnit = estEntryCost_unit - estExitNet_unit

    if pd.isna(riskPerUnit) or riskPerUnit <= 0:
        return 0
    maxQty_by_cap = cap / estEntryCost_unit
    qty = int(math.floor(min(max_risk_amount / riskPerUnit, maxQty_by_cap)))

    while qty > 0:
        entry_fee = max(bPrice * qty * params.buy_fee, params.min_fee)
        exact_entry_cost = bPrice * qty + entry_fee
        sell_fee = max(stopPrice * qty * params.sell_fee, params.min_fee)
        tax = stopPrice * qty * params.tax_rate
        exact_exit_net = stopPrice * qty - sell_fee - tax
        actual_risk = exact_entry_cost - exact_exit_net

        if exact_entry_cost <= cap and actual_risk <= max_risk_amount:
            return qty
        qty -= 1
    return 0


# # (AI註: scanner 參考投入 / 掛單股數統一使用 scanner_live_capital，避免顯示與實務下單資金來源分叉)
def calc_reference_candidate_qty(bPrice, stopPrice, params):
    from core.config import resolve_scanner_live_capital

    return calc_position_size(bPrice, stopPrice, resolve_scanner_live_capital(params), params.fixed_risk, params)


# # (AI註: 單一真理來源 - 半倉停利可執行股數統一由此計算，避免核心/掃描/除錯工具各自判斷)
def calc_half_take_profit_sell_qty(position_qty, tp_percent):
    if pd.isna(position_qty) or pd.isna(tp_percent):
        return 0

    position_qty = int(position_qty)
    tp_percent = float(tp_percent)
    if position_qty <= 0 or tp_percent <= 0.0:
        return 0

    sell_qty = int(math.floor(position_qty * tp_percent))
    if sell_qty <= 0 or position_qty <= sell_qty:
        return 0
    return sell_qty


def can_execute_half_take_profit(position_qty, tp_percent):
    return calc_half_take_profit_sell_qty(position_qty, tp_percent) > 0


# # (AI註: 單一真理來源 - 統一 initial_risk_total 的計算與 fallback 口徑，禁止散落 magic number)
def calc_initial_risk_total(entry_price, net_stop_price, qty, params):
    if pd.isna(entry_price) or pd.isna(net_stop_price) or qty <= 0:
        return 0.0

    init_risk = (entry_price - net_stop_price) * qty
    if init_risk > 0:
        return init_risk

    actual_total_cost = entry_price * qty
    return max(actual_total_cost * params.fixed_risk, 0.0)


# # (AI註: 單一真理來源 - 漲跌停價與一字漲/跌停判斷統一由此處理，避免 core / portfolio / tool 各寫各的)
def calc_limit_up_price(reference_price):
    if pd.isna(reference_price) or reference_price <= 0:
        return np.nan
    return adjust_price_down_to_tick(reference_price * 1.10)


def calc_limit_down_price(reference_price):
    if pd.isna(reference_price) or reference_price <= 0:
        return np.nan
    return adjust_price_up_to_tick(reference_price * 0.90)


def is_single_price_bar(t_open, t_high, t_low, t_close):
    if pd.isna(t_open) or pd.isna(t_high) or pd.isna(t_low) or pd.isna(t_close):
        return False
    return (
        np.isclose(t_open, t_high)
        and np.isclose(t_high, t_low)
        and np.isclose(t_low, t_close)
    )


def is_locked_limit_up_bar(t_open, t_high, t_low, t_close, y_close):
    limit_up_price = calc_limit_up_price(y_close)
    if pd.isna(limit_up_price):
        return False
    return is_single_price_bar(t_open, t_high, t_low, t_close) and np.isclose(t_close, limit_up_price)


def is_locked_limit_down_bar(t_open, t_high, t_low, t_close, y_close):
    limit_down_price = calc_limit_down_price(y_close)
    if pd.isna(limit_down_price):
        return False
    return is_single_price_bar(t_open, t_high, t_low, t_close) and np.isclose(t_close, limit_down_price)


# # (AI註: 單一真理來源 - 賣出被阻塞的原因統一由此判斷，避免零量 / 一字跌停口徑分裂)
def get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close):
    if pd.isna(t_volume) or t_volume <= 0:
        return 'NO_VOLUME'
    if is_locked_limit_down_bar(t_open, t_high, t_low, t_close, y_close):
        return 'LOCKED_DOWN'
    return None
