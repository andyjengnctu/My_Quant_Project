import math
from functools import lru_cache

import numpy as np
import pandas as pd

from config.broker_accounting import BROKER_FEE_RATE, BROKER_MIN_FEE_TWD
from core.training_performance import is_optimizer_native_hotpath_kernels_enabled

from core.native_optimizer_hotpath import calc_position_size_exact as _native_calc_position_size_exact

_NATIVE_HOTPATH_ENABLED = bool(
    _native_calc_position_size_exact is not None and is_optimizer_native_hotpath_kernels_enabled()
)

from core.capital_policy import resolve_scanner_live_capital
from core.order_lot_policy import (
    apply_board_lot_preferred_qty,
    apply_board_lot_preferred_qty_milli,
    get_min_entry_notional_milli,
)
from core.exact_accounting import (
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    calc_average_price_from_total_milli,
    calc_buy_net_total_milli_from_milli,
    calc_cash_buy_total_milli_from_milli,
    calc_entry_total_cost,
    calc_exit_net_total,
    calc_initial_risk_total_milli,
    calc_total_from_average_price_milli,
    calc_limit_down_price_milli,
    calc_limit_up_price_milli,
    calc_risk_budget_milli,
    calc_sell_net_total_milli_from_milli,
    get_tick_milli_from_price,
    infer_security_profile,
    normalize_security_symbol,
    milli_to_money,
    milli_to_price,
    money_to_milli,
    price_to_milli,
    rate_to_ppm,
    resolve_fee_schedule_tuple,
    round_price_to_tick_milli,
)

_NATIVE_BROKER_FEE_PPM = rate_to_ppm(BROKER_FEE_RATE)
_NATIVE_BROKER_MIN_FEE_MILLI = money_to_milli(BROKER_MIN_FEE_TWD)


def tv_round(number):
    return math.floor(number + 0.5)


@lru_cache(maxsize=2048)
def _cached_price_security_profile(normalized_ticker):
    # AI註: price helpers only read the inferred profile; cache that read-only
    # internal view so repeated tick/limit calls do not allocate a dict copy.
    return infer_security_profile(normalized_ticker)


def _resolve_price_security_profile(ticker, security_profile):
    if security_profile is not None:
        return security_profile
    return _cached_price_security_profile(normalize_security_symbol(ticker))


def get_tick_size(price, ticker=None, security_profile=None):
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    return milli_to_price(get_tick_milli_from_price(price, security_profile=resolved_profile))


# # (AI註: 第12點 - 跳價取整方向統一收斂到單一函式，避免買/賣/停損/停利各自手寫)
def round_to_tick(price, direction="nearest", ticker=None, security_profile=None):
    if pd.isna(price):
        return np.nan
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    return milli_to_price(round_price_to_tick_milli(price, direction=direction, security_profile=resolved_profile))


def adjust_to_tick(price, ticker=None, security_profile=None):
    return round_to_tick(price, direction="nearest", ticker=ticker, security_profile=security_profile)


def adjust_price_up_to_tick(price, ticker=None, security_profile=None):
    return round_to_tick(price, direction="up", ticker=ticker, security_profile=security_profile)


def adjust_price_down_to_tick(price, ticker=None, security_profile=None):
    return round_to_tick(price, direction="down", ticker=ticker, security_profile=security_profile)


# # (AI註: 長倉語義化封裝 - 買單/賣單/停損/停利各自固定方向，避免保守性漂移)
def adjust_long_buy_limit(price, ticker=None, security_profile=None):
    return adjust_price_down_to_tick(price, ticker=ticker, security_profile=security_profile)


def adjust_long_stop_price(price, ticker=None, security_profile=None):
    return adjust_price_up_to_tick(price, ticker=ticker, security_profile=security_profile)


def adjust_long_target_price(price, ticker=None, security_profile=None):
    return adjust_price_up_to_tick(price, ticker=ticker, security_profile=security_profile)


# # (AI註: 單一真理來源 - 首個可執行停損價格一律由入場基準價與 ATR 倍數計算)
def calc_initial_stop_from_reference(reference_price, atr, params, ticker=None, security_profile=None):
    if pd.isna(reference_price) or pd.isna(atr):
        return np.nan
    return adjust_long_stop_price(reference_price - atr * params.atr_times_init, ticker=ticker, security_profile=security_profile)


# # (AI註: 單一真理來源 - 初始 trailing 基準價一律由入場基準價與 ATR 倍數計算)
def calc_initial_trailing_stop_from_reference(reference_price, atr, params, ticker=None, security_profile=None):
    if pd.isna(reference_price) or pd.isna(atr):
        return np.nan
    return adjust_long_stop_price(reference_price - atr * params.atr_times_trail, ticker=ticker, security_profile=security_profile)


# # (AI註: 單一真理來源 - 目標價一律由入場基準價與停損價差推導；可用於盤前 sizing 基準或實際成交後首個可執行停利)
def calc_frozen_target_price(entry_reference_price, stop_price, ticker=None, security_profile=None):
    if pd.isna(entry_reference_price) or pd.isna(stop_price):
        return np.nan
    return adjust_long_target_price(entry_reference_price + (entry_reference_price - stop_price), ticker=ticker, security_profile=security_profile)


def adjust_long_buy_fill_price(price, ticker=None, security_profile=None):
    return adjust_price_up_to_tick(price, ticker=ticker, security_profile=security_profile)


def adjust_long_sell_fill_price(price, ticker=None, security_profile=None):
    return adjust_price_down_to_tick(price, ticker=ticker, security_profile=security_profile)


def get_tick_size_array(prices, ticker=None, security_profile=None):
    prices = np.asarray(prices, dtype=np.float64)
    ticks = np.full(prices.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(prices)
    if not np.any(valid):
        return ticks
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    ticks[valid] = np.fromiter(
        (milli_to_price(get_tick_milli_from_price(price, security_profile=resolved_profile)) for price in prices[valid]),
        dtype=np.float64,
        count=int(np.count_nonzero(valid)),
    )
    return ticks


def round_to_tick_array(prices, direction="nearest", ticker=None, security_profile=None):
    prices = np.asarray(prices, dtype=np.float64)
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(prices)
    if not np.any(valid):
        return out
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    out[valid] = np.fromiter(
        (
            milli_to_price(round_price_to_tick_milli(price, direction=direction, security_profile=resolved_profile))
            for price in prices[valid]
        ),
        dtype=np.float64,
        count=int(np.count_nonzero(valid)),
    )
    return out


def adjust_to_tick_array(prices, ticker=None, security_profile=None):
    return round_to_tick_array(prices, direction="nearest", ticker=ticker, security_profile=security_profile)


def adjust_long_buy_limit_array(prices, ticker=None, security_profile=None):
    return round_to_tick_array(prices, direction="down", ticker=ticker, security_profile=security_profile)


def calc_entry_price(bPrice, bQty, params):
    if pd.isna(bPrice) or pd.isna(bQty) or bPrice <= 0 or bQty <= 0:
        return np.nan
    ledger = build_buy_ledger_from_price(bPrice, int(bQty), params)
    return calc_average_price_from_total_milli(ledger["net_buy_total_milli"], int(bQty))


def calc_net_sell_price(sPrice, sQty, params, ticker=None, security_profile=None, trade_date=None):
    if pd.isna(sPrice) or pd.isna(sQty) or sPrice <= 0 or sQty <= 0:
        return np.nan
    ledger = build_sell_ledger_from_price(
        sPrice,
        int(sQty),
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
    )
    return calc_average_price_from_total_milli(ledger["net_sell_total_milli"], int(sQty))


def build_position_sizing_capital_context(cap, riskPct, params):
    """Precompute capital-only exact sizing bounds for repeated same-capital candidate sizing."""
    cap_milli = money_to_milli(cap)
    max_risk_milli = calc_risk_budget_milli(cap_milli, riskPct)
    max_position_cap_pct = float(getattr(params, "max_position_cap_pct", 1.0))
    max_position_cap_milli = min(
        cap_milli,
        calc_risk_budget_milli(cap_milli, max_position_cap_pct),
    )
    return int(cap_milli), int(max_risk_milli), int(max_position_cap_milli)


def build_position_sizing_fee_context(params, *, ticker=None, security_profile=None, trade_date=None):
    """Pre-resolve fee/tax/rounding inputs reused by repeated exact sizing calls."""
    buy_fee_ppm, sell_fee_ppm, tax_ppm, min_fee_milli, _fixed_risk_ppm = resolve_fee_schedule_tuple(
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
    )
    charge_rounding_mode = str(
        getattr(params, "charge_rounding_mode", "nearest_milli") or "nearest_milli"
    )
    rounding_mode_code = (
        1 if charge_rounding_mode == "floor_twd"
        else 0 if charge_rounding_mode == "nearest_milli"
        else -1
    )
    return (
        int(buy_fee_ppm),
        int(sell_fee_ppm),
        int(tax_ppm),
        int(min_fee_milli),
        int(rounding_mode_code),
        int(get_min_entry_notional_milli(params)),
    )


def calc_position_size_from_milli_context(
    buy_price_milli,
    stop_price_milli,
    capital_context,
    fee_context,
    params,
    *,
    ticker=None,
    security_profile=None,
    trade_date=None,
):
    """Exact sizing primitive for hot paths that already own milli/context inputs.

    The public float adapter below remains the generic API.  Candidate replay can
    consume this primitive directly after its canonical price/context resolution,
    avoiding duplicate NaN/type/context work without changing any sizing rule.
    """
    buy_price_milli = int(buy_price_milli)
    stop_price_milli = int(stop_price_milli)
    if buy_price_milli <= 0 or stop_price_milli <= 0:
        return 0

    cap_milli, max_risk_milli, max_position_cap_milli = capital_context
    buy_fee_ppm, sell_fee_ppm, tax_ppm, min_fee_milli, rounding_mode_code, _min_entry_notional_milli = fee_context

    if _NATIVE_HOTPATH_ENABLED and int(rounding_mode_code) >= 0:
        return int(_native_calc_position_size_exact(
            int(buy_price_milli),
            int(stop_price_milli),
            int(cap_milli),
            int(max_risk_milli),
            int(max_position_cap_milli),
            int(buy_fee_ppm),
            int(sell_fee_ppm),
            int(tax_ppm),
            int(min_fee_milli),
            int(rounding_mode_code) == 1,
            int(_NATIVE_BROKER_FEE_PPM),
            int(_NATIVE_BROKER_MIN_FEE_MILLI),
        ))

    est_buy_fee_per_share_milli = 0 if buy_fee_ppm <= 0 else (buy_price_milli * buy_fee_ppm + 999_999) // 1_000_000
    est_sell_cost_per_share_milli = 0 if (sell_fee_ppm + tax_ppm) <= 0 else ((stop_price_milli * (sell_fee_ppm + tax_ppm) + 999_999) // 1_000_000)
    est_entry_unit_milli = buy_price_milli + est_buy_fee_per_share_milli
    est_exit_unit_milli = stop_price_milli - est_sell_cost_per_share_milli
    risk_per_share_milli = est_entry_unit_milli - est_exit_unit_milli
    if risk_per_share_milli <= 0 or max_position_cap_milli <= 0:
        return 0

    qty = int(min(
        cap_milli // max(est_entry_unit_milli, 1),
        max_risk_milli // max(risk_per_share_milli, 1),
        max_position_cap_milli // max(est_entry_unit_milli, 1),
    ))

    def _is_valid_qty(test_qty):
        exact_entry_cost_milli = calc_buy_net_total_milli_from_milli(
            buy_price_milli, test_qty, params
        )
        exact_entry_cash_milli = calc_cash_buy_total_milli_from_milli(
            buy_price_milli, test_qty
        )
        exact_exit_net_milli = calc_sell_net_total_milli_from_milli(
            stop_price_milli,
            test_qty,
            params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=trade_date,
        )
        actual_risk_milli = exact_entry_cost_milli - exact_exit_net_milli
        return (
            exact_entry_cash_milli <= cap_milli
            and exact_entry_cash_milli <= max_position_cap_milli
            and actual_risk_milli <= max_risk_milli
        )

    if qty <= 0:
        return 0
    if _is_valid_qty(qty):
        return qty

    probe_qty = qty
    for _ in range(8):
        probe_qty -= 1
        if probe_qty <= 0:
            return 0
        if _is_valid_qty(probe_qty):
            return probe_qty

    lo, hi = 0, probe_qty - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _is_valid_qty(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def calc_position_size(
    bPrice,
    stopPrice,
    cap,
    riskPct,
    params,
    ticker=None,
    security_profile=None,
    trade_date=None,
    *,
    _capital_context=None,
    _fee_context=None,
    _buy_price_milli=None,
    _stop_price_milli=None,
):
    if pd.isna(bPrice) or pd.isna(stopPrice) or bPrice <= 0 or stopPrice <= 0:
        return 0

    buy_price_milli = price_to_milli(bPrice) if _buy_price_milli is None else int(_buy_price_milli)
    stop_price_milli = price_to_milli(stopPrice) if _stop_price_milli is None else int(_stop_price_milli)
    capital_context = (
        build_position_sizing_capital_context(cap, riskPct, params)
        if _capital_context is None
        else _capital_context
    )
    fee_context = (
        build_position_sizing_fee_context(
            params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=trade_date,
        )
        if _fee_context is None
        else _fee_context
    )
    return calc_position_size_from_milli_context(
        buy_price_milli,
        stop_price_milli,
        capital_context,
        fee_context,
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
    )


# # (AI註: scanner 參考投入 / 掛單股數統一使用 scanner_live_capital，避免顯示與實務下單資金來源分叉)
def calc_reference_candidate_qty(bPrice, stopPrice, params, ticker=None, security_profile=None, trade_date=None):
    qty = calc_position_size(
        bPrice,
        stopPrice,
        resolve_scanner_live_capital(params),
        params.fixed_risk,
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
    )
    return apply_board_lot_preferred_qty(bPrice, qty, params)


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

    entry_total_milli = calc_total_from_average_price_milli(entry_price, qty)
    stop_net_total_milli = calc_total_from_average_price_milli(net_stop_price, qty)
    init_risk_milli = calc_initial_risk_total_milli(entry_total_milli, stop_net_total_milli, rate_to_ppm(params.fixed_risk))
    return milli_to_money(init_risk_milli)


# # (AI註: 單一真理來源 - 漲跌停價與一字漲/跌停判斷統一由此處理，避免 core / portfolio / tool 各寫各的)
def calc_limit_up_price(reference_price, ticker=None, security_profile=None):
    if pd.isna(reference_price) or reference_price <= 0:
        return np.nan
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    return milli_to_price(calc_limit_up_price_milli(price_to_milli(reference_price), security_profile=resolved_profile))


def calc_limit_down_price(reference_price, ticker=None, security_profile=None):
    if pd.isna(reference_price) or reference_price <= 0:
        return np.nan
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    return milli_to_price(calc_limit_down_price_milli(price_to_milli(reference_price), security_profile=resolved_profile))


# # (AI註: 單一真理來源 - 盤前固定限價單今日是否仍有法定價格帶可達性統一由此判斷)
def is_limit_buy_price_milli_reachable_for_day(
    limit_price_milli,
    y_close_milli,
    ticker=None,
    security_profile=None,
):
    try:
        limit_milli = int(limit_price_milli)
        close_milli = int(y_close_milli)
    except (TypeError, ValueError, OverflowError):
        return False
    if limit_milli <= 0 or close_milli <= 0:
        return False
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    limit_down_milli = calc_limit_down_price_milli(
        close_milli,
        security_profile=resolved_profile,
    )
    return limit_milli >= int(limit_down_milli)


def is_limit_buy_price_reachable_for_day(limit_price, y_close, ticker=None, security_profile=None):
    if pd.isna(limit_price) or limit_price <= 0 or pd.isna(y_close) or y_close <= 0:
        return False
    return is_limit_buy_price_milli_reachable_for_day(
        price_to_milli(limit_price),
        price_to_milli(y_close),
        ticker=ticker,
        security_profile=security_profile,
    )


def is_limit_up_bar(t_open, t_high, t_low, t_close, y_close, ticker=None, security_profile=None):
    if any(pd.isna(v) for v in (t_open, t_high, t_low, t_close, y_close)):
        return False
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    limit_up_price_milli = calc_limit_up_price_milli(price_to_milli(y_close), security_profile=resolved_profile)
    return all(price_to_milli(v) == limit_up_price_milli for v in (t_open, t_high, t_low, t_close))


def is_limit_down_bar(t_open, t_high, t_low, t_close, y_close, ticker=None, security_profile=None):
    if any(pd.isna(v) for v in (t_open, t_high, t_low, t_close, y_close)):
        return False
    resolved_profile = _resolve_price_security_profile(ticker, security_profile)
    limit_down_price_milli = calc_limit_down_price_milli(price_to_milli(y_close), security_profile=resolved_profile)
    return all(price_to_milli(v) == limit_down_price_milli for v in (t_open, t_high, t_low, t_close))


def is_locked_limit_up_bar(t_open, t_high, t_low, t_close, y_close, ticker=None, security_profile=None):
    return is_limit_up_bar(t_open, t_high, t_low, t_close, y_close, ticker=ticker, security_profile=security_profile)


def is_locked_limit_down_bar(t_open, t_high, t_low, t_close, y_close, ticker=None, security_profile=None):
    return is_limit_down_bar(t_open, t_high, t_low, t_close, y_close, ticker=ticker, security_profile=security_profile)


def get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=None, security_profile=None):
    if pd.isna(t_volume) or t_volume <= 0:
        return 'NO_VOLUME'
    if is_locked_limit_down_bar(t_open, t_high, t_low, t_close, y_close, ticker=ticker, security_profile=security_profile):
        return 'LOCKED_DOWN'
    return None
