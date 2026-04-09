import math

import numpy as np
import pandas as pd

from core.capital_policy import resolve_scanner_live_capital
from core.exact_accounting import (
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    calc_average_price_from_total_milli,
    calc_entry_total_cost,
    calc_exit_net_total,
    calc_initial_risk_total_milli,
    calc_limit_down_price_milli,
    calc_limit_up_price_milli,
    calc_risk_budget_milli,
    get_tick_milli,
    milli_to_money,
    milli_to_price,
    money_to_milli,
    price_to_milli,
    rate_to_ppm,
    round_price_milli_to_tick,
)


def tv_round(number):
    return math.floor(number + 0.5)


def get_tick_size(price):
    return milli_to_price(get_tick_milli(price_to_milli(price)))


# # (AI註: 第12點 - 跳價取整方向統一收斂到單一函式，避免買/賣/停損/停利各自手寫)
def round_to_tick(price, direction="nearest"):
    if pd.isna(price):
        return np.nan
    return milli_to_price(round_price_milli_to_tick(price_to_milli(price), direction=direction))


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


# # (AI註: 單一真理來源 - 首個可執行停損價格一律由入場基準價與 ATR 倍數計算)
def calc_initial_stop_from_reference(reference_price, atr, params):
    if pd.isna(reference_price) or pd.isna(atr):
        return np.nan
    return adjust_long_stop_price(reference_price - atr * params.atr_times_init)


# # (AI註: 單一真理來源 - 初始 trailing 基準價一律由入場基準價與 ATR 倍數計算)
def calc_initial_trailing_stop_from_reference(reference_price, atr, params):
    if pd.isna(reference_price) or pd.isna(atr):
        return np.nan
    return adjust_long_stop_price(reference_price - atr * params.atr_times_trail)


# # (AI註: 單一真理來源 - 目標價一律由入場基準價與停損價差推導；可用於盤前 sizing 基準或實際成交後首個可執行停利)
def calc_frozen_target_price(entry_reference_price, stop_price):
    if pd.isna(entry_reference_price) or pd.isna(stop_price):
        return np.nan
    return adjust_long_target_price(entry_reference_price + (entry_reference_price - stop_price))


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
    ledger = build_buy_ledger_from_price(bPrice, int(bQty), params)
    return calc_average_price_from_total_milli(ledger["net_buy_total_milli"], int(bQty))


def calc_net_sell_price(sPrice, sQty, params):
    if pd.isna(sPrice) or pd.isna(sQty) or sPrice <= 0 or sQty <= 0:
        return np.nan
    ledger = build_sell_ledger_from_price(sPrice, int(sQty), params)
    return calc_average_price_from_total_milli(ledger["net_sell_total_milli"], int(sQty))


def calc_position_size(bPrice, stopPrice, cap, riskPct, params):
    if pd.isna(bPrice) or pd.isna(stopPrice) or bPrice <= 0 or stopPrice <= 0:
        return 0

    buy_price_milli = price_to_milli(bPrice)
    stop_price_milli = price_to_milli(stopPrice)
    cap_milli = money_to_milli(cap)
    max_risk_milli = calc_risk_budget_milli(cap_milli, riskPct)

    buy_fee_ppm = rate_to_ppm(params.buy_fee)
    sell_fee_ppm = rate_to_ppm(params.sell_fee)
    tax_ppm = rate_to_ppm(params.tax_rate)

    est_buy_fee_per_share_milli = 0 if buy_fee_ppm <= 0 else (buy_price_milli * buy_fee_ppm + 999_999) // 1_000_000
    est_sell_cost_per_share_milli = 0 if (sell_fee_ppm + tax_ppm) <= 0 else ((stop_price_milli * (sell_fee_ppm + tax_ppm) + 999_999) // 1_000_000)
    est_entry_unit_milli = buy_price_milli + est_buy_fee_per_share_milli
    est_exit_unit_milli = stop_price_milli - est_sell_cost_per_share_milli
    risk_per_share_milli = est_entry_unit_milli - est_exit_unit_milli
    if risk_per_share_milli <= 0:
        return 0

    qty = int(min(cap_milli // max(est_entry_unit_milli, 1), max_risk_milli // max(risk_per_share_milli, 1)))

    while qty > 0:
        buy_ledger = build_buy_ledger_from_price(bPrice, qty, params)
        sell_ledger = build_sell_ledger_from_price(stopPrice, qty, params)
        exact_entry_cost_milli = buy_ledger["net_buy_total_milli"]
        exact_exit_net_milli = sell_ledger["net_sell_total_milli"]
        actual_risk_milli = exact_entry_cost_milli - exact_exit_net_milli

        if exact_entry_cost_milli <= cap_milli and actual_risk_milli <= max_risk_milli:
            return qty
        qty -= 1
    return 0


# # (AI註: scanner 參考投入 / 掛單股數統一使用 scanner_live_capital，避免顯示與實務下單資金來源分叉)
def calc_reference_candidate_qty(bPrice, stopPrice, params):
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

    entry_total_milli = money_to_milli(entry_price * qty)
    stop_net_total_milli = money_to_milli(net_stop_price * qty)
    init_risk_milli = calc_initial_risk_total_milli(entry_total_milli, stop_net_total_milli, rate_to_ppm(params.fixed_risk))
    return milli_to_money(init_risk_milli)


# # (AI註: 單一真理來源 - 漲跌停價與一字漲/跌停判斷統一由此處理，避免 core / portfolio / tool 各寫各的)
def calc_limit_up_price(reference_price):
    if pd.isna(reference_price) or reference_price <= 0:
        return np.nan
    return milli_to_price(calc_limit_up_price_milli(price_to_milli(reference_price)))


def calc_limit_down_price(reference_price):
    if pd.isna(reference_price) or reference_price <= 0:
        return np.nan
    return milli_to_price(calc_limit_down_price_milli(price_to_milli(reference_price)))


# # (AI註: 單一真理來源 - 盤前固定限價單今日是否仍有法定價格帶可達性統一由此判斷)
def is_limit_buy_price_reachable_for_day(limit_price, y_close):
    if pd.isna(limit_price) or limit_price <= 0 or pd.isna(y_close) or y_close <= 0:
        return False
    limit_down_price = calc_limit_down_price(y_close)
    if pd.isna(limit_down_price):
        return False
    return price_to_milli(limit_price) >= price_to_milli(limit_down_price)


def is_limit_up_bar(t_open, t_high, t_low, t_close, y_close):
    if any(pd.isna(v) for v in (t_open, t_high, t_low, t_close, y_close)):
        return False
    limit_up_price_milli = calc_limit_up_price_milli(price_to_milli(y_close))
    return all(price_to_milli(v) == limit_up_price_milli for v in (t_open, t_high, t_low, t_close))


def is_limit_down_bar(t_open, t_high, t_low, t_close, y_close):
    if any(pd.isna(v) for v in (t_open, t_high, t_low, t_close, y_close)):
        return False
    limit_down_price_milli = calc_limit_down_price_milli(price_to_milli(y_close))
    return all(price_to_milli(v) == limit_down_price_milli for v in (t_open, t_high, t_low, t_close))


def is_locked_limit_up_bar(t_open, t_high, t_low, t_close, y_close):
    return is_limit_up_bar(t_open, t_high, t_low, t_close, y_close)


def is_locked_limit_down_bar(t_open, t_high, t_low, t_close, y_close):
    return is_limit_down_bar(t_open, t_high, t_low, t_close, y_close)


def get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close):
    if pd.isna(t_volume) or t_volume <= 0:
        return 'NO_VOLUME'
    if is_locked_limit_down_bar(t_open, t_high, t_low, t_close, y_close):
        return 'LOCKED_DOWN'
    return None
