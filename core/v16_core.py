import pandas as pd
import numpy as np
import math
from core.v16_config import V16StrategyParams, EV_CALC_METHOD

def tv_round(number): return math.floor(number + 0.5)

def get_tick_size(price):
    if price < 1: return 0.001
    elif price < 10: return 0.01
    elif price < 50: return 0.05
    elif price < 100: return 0.1
    elif price < 500: return 0.5
    elif price < 1000: return 1.0
    else: return 5.0


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
    if pd.isna(bPrice) or pd.isna(stopPrice) or bPrice <= 0 or stopPrice <= 0: return 0
    max_risk_amount = cap * riskPct
    estEntryCost_unit = bPrice * (1 + params.buy_fee)
    estExitNet_unit = stopPrice * (1 - params.sell_fee - params.tax_rate)
    riskPerUnit = estEntryCost_unit - estExitNet_unit
    
    if pd.isna(riskPerUnit) or riskPerUnit <= 0: return 0
    maxQty_by_cap = cap / estEntryCost_unit
    qty = int(math.floor(min(max_risk_amount / riskPerUnit, maxQty_by_cap)))
    
    while qty > 0:
        entry_fee = max(bPrice * qty * params.buy_fee, params.min_fee)
        exact_entry_cost = bPrice * qty + entry_fee
        sell_fee = max(stopPrice * qty * params.sell_fee, params.min_fee)
        tax = stopPrice * qty * params.tax_rate
        exact_exit_net = stopPrice * qty - sell_fee - tax
        actual_risk = exact_entry_cost - exact_exit_net
        
        if exact_entry_cost <= cap and actual_risk <= max_risk_amount: return qty
        qty -= 1
    return 0


# # (AI註: scanner / consistency tool 的候選顯示統一用 initial_capital 估算，避免被單股回測複利資本污染)
def calc_reference_candidate_qty(bPrice, stopPrice, params):
    return calc_position_size(bPrice, stopPrice, params.initial_capital, params.fixed_risk, params)


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

def evaluate_history_candidate_metrics(trade_count, win_count, total_r_sum, win_r_sum, loss_r_sum, params):
    min_trades_req = getattr(params, 'min_history_trades', 0)
    min_ev_req = getattr(params, 'min_history_ev', 0.0)
    min_win_rate_req = getattr(params, 'min_history_win_rate', 0.30)

    allow_zero_history = (
        (min_trades_req == 0) and
        (min_ev_req <= 0) and
        (min_win_rate_req <= 0)
    )

    if trade_count < min_trades_req:
        return False, 0.0, 0.0, trade_count

    if trade_count == 0:
        return allow_zero_history, 0.0, 0.0, trade_count

    win_rate = win_count / trade_count

    if EV_CALC_METHOD == 'B':
        avg_win_r = (win_r_sum / win_count) if win_count > 0 else 0.0
        loss_count = trade_count - win_count
        avg_loss_r = abs(loss_r_sum / loss_count) if loss_count > 0 else 0.0

        if avg_loss_r > 0:
            payoff_for_ev = min(10.0, avg_win_r / avg_loss_r)
        elif avg_win_r > 0:
            payoff_for_ev = 99.9
        else:
            payoff_for_ev = 0.0

        expected_value = (win_rate * payoff_for_ev) - (1 - win_rate)
    else:
        expected_value = total_r_sum / trade_count

    is_candidate = (
        (win_rate >= min_win_rate_req)
        and
        (expected_value >= min_ev_req)
    )
    return is_candidate, expected_value, win_rate, trade_count

# # (AI註: 單一真理來源 - 候選單在不同資金上限下的 qty / is_orderable 重算統一由此處理)
def resize_candidate_plan_to_capital(candidate_plan, sizing_capital, params):
    if candidate_plan is None:
        return None

    limit_price = candidate_plan.get('limit_price')
    init_sl = candidate_plan.get('init_sl')
    if pd.isna(limit_price) or pd.isna(init_sl):
        return None

    resized_plan = dict(candidate_plan)
    qty = calc_position_size(limit_price, init_sl, sizing_capital, params.fixed_risk, params)
    resized_plan['qty'] = qty
    resized_plan['is_orderable'] = qty > 0
    return resized_plan


# # (AI註: 單一真理來源 - 盤前依當下可用現金重算有效掛單規格與保留資金)
def build_cash_capped_entry_plan(candidate_plan, available_cash, params):
    resized_plan = resize_candidate_plan_to_capital(candidate_plan, available_cash, params)
    if resized_plan is None or resized_plan['qty'] <= 0:
        return None

    reserved_cost = calc_entry_price(resized_plan['limit_price'], resized_plan['qty'], params) * resized_plan['qty']
    if pd.isna(reserved_cost) or reserved_cost > available_cash:
        return None

    resized_plan['reserved_cost'] = reserved_cost
    return resized_plan


# # (AI註: 單一真理來源 - 正常候選的 limit/sl/trail/qty 規格統一由此產生；候選資格與可掛單分離)
def build_normal_candidate_plan(limit_price, atr, sizing_capital, params):
    if pd.isna(limit_price) or pd.isna(atr):
        return None

    init_sl = adjust_long_stop_price(limit_price - atr * params.atr_times_init)
    init_trail = adjust_long_stop_price(limit_price - atr * params.atr_times_trail)
    base_plan = {
        'limit_price': limit_price,
        'init_sl': init_sl,
        'init_trail': init_trail,
    }
    return resize_candidate_plan_to_capital(base_plan, sizing_capital, params)


# # (AI註: 單一真理來源 - 正常單實際掛單規格；僅接受可掛單候選)
def build_normal_entry_plan(limit_price, atr, sizing_capital, params):
    candidate_plan = build_normal_candidate_plan(limit_price, atr, sizing_capital, params)
    if candidate_plan is None or candidate_plan['qty'] <= 0:
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
        np.isclose(t_open, t_high) and
        np.isclose(t_high, t_low) and
        np.isclose(t_low, t_close)
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


# # (AI註: 單一真理來源 - 成交後的部位欄位統一由此建立)
def build_position_from_entry_fill(buy_price, qty, init_sl, init_trail, params, entry_type='normal'):
    if pd.isna(buy_price) or buy_price <= 0 or qty <= 0:
        raise ValueError(f"build_position_from_entry_fill 需要有效 buy_price/qty，收到 buy_price={buy_price!r}, qty={qty!r}")

    entry_price = calc_entry_price(buy_price, qty, params)
    net_sl = calc_net_sell_price(init_sl, qty, params)
    tp_half = adjust_long_target_price(buy_price + (entry_price - net_sl))
    init_risk = calc_initial_risk_total(entry_price, net_sl, qty, params)

    return {
        'qty': qty,
        'entry': entry_price,
        'sl': max(init_sl, init_trail),
        'initial_stop': init_sl,
        'trailing_stop': init_trail,
        'tp_half': tp_half,
        'sold_half': False,
        'pure_buy_price': buy_price,
        'realized_pnl': 0.0,
        'initial_risk_total': init_risk,
        'entry_type': entry_type,
    }


# # (AI註: 單一真理來源 - 盤前有效買單的當日成交 / miss buy / 先達停損放棄進場邏輯統一由此判斷)
def execute_pre_market_entry_plan(entry_plan, t_open, t_high, t_low, t_close, t_volume, y_close, params, entry_type='normal'):
    result = {
        'filled': False,
        'count_as_missed_buy': False,
        'is_worse_than_initial_stop': False,
        'is_locked_limit_up': False,
        'buy_price': np.nan,
        'entry_price': np.nan,
        'tp_half': np.nan,
        'position': None,
        'entry_type': entry_type,
    }
    if entry_plan is None:
        return result

    qty = entry_plan.get('qty', 0)
    result['count_as_missed_buy'] = should_count_miss_buy(qty)
    result['is_locked_limit_up'] = is_locked_limit_up_bar(t_open, t_high, t_low, t_close, y_close)

    if qty <= 0:
        result['count_as_missed_buy'] = False
        return result

    if pd.isna(t_volume) or t_volume <= 0 or pd.isna(t_open) or pd.isna(t_low) or result['is_locked_limit_up']:
        return result

    limit_price = entry_plan['limit_price']
    init_sl = entry_plan['init_sl']
    init_trail = entry_plan['init_trail']

    if t_low > limit_price:
        return result

    buy_price = adjust_long_buy_fill_price(min(t_open, limit_price))
    result['buy_price'] = buy_price

    if buy_price <= init_sl:
        result['is_worse_than_initial_stop'] = True
        result['count_as_missed_buy'] = False
        return result

    position = build_position_from_entry_fill(
        buy_price=buy_price,
        qty=qty,
        init_sl=init_sl,
        init_trail=init_trail,
        params=params,
        entry_type=entry_type,
    )
    result['filled'] = True
    result['count_as_missed_buy'] = False
    result['position'] = position
    result['entry_price'] = position['entry']
    result['tp_half'] = position['tp_half']
    return result


# # (AI註: 單一真理來源 - 延續訊號何時失效統一由此判斷)
def should_clear_extended_signal(signal_state, t_low):
    if signal_state is None or pd.isna(t_low):
        return False
    return t_low <= signal_state['init_sl']


# # (AI註: 單一真理來源 - 延續候選原始訊號狀態統一由此建立)
def create_signal_tracking_state(original_limit, atr, params):
    if pd.isna(original_limit) or pd.isna(atr):
        return None

    init_sl = adjust_long_stop_price(original_limit - atr * params.atr_times_init)
    return {
        'orig_limit': original_limit,
        'orig_atr': atr,
        'init_sl': init_sl,
    }


# # (AI註: 單一真理來源 - 延續候選每日盤前資格規格統一由此產生；必須仍在原始買入區間內)
def build_extended_candidate_plan_from_signal(signal_state, reference_price, sizing_capital, params):
    if signal_state is None or pd.isna(reference_price):
        return None

    original_limit = signal_state['orig_limit']
    atr = signal_state['orig_atr']
    init_sl = signal_state['init_sl']
    if not (init_sl < reference_price <= original_limit):
        return None

    limit_price = adjust_long_buy_limit(reference_price)
    if pd.isna(limit_price) or not (init_sl < limit_price <= original_limit):
        return None

    init_trail = adjust_long_stop_price(limit_price - atr * params.atr_times_trail)
    base_plan = {
        'limit_price': limit_price,
        'init_sl': init_sl,
        'init_trail': init_trail,
        'orig_limit': original_limit,
        'orig_atr': atr,
    }
    return resize_candidate_plan_to_capital(base_plan, sizing_capital, params)


# # (AI註: 單一真理來源 - 延續單實際掛單規格；僅接受可掛單候選)
def build_extended_entry_plan_from_signal(signal_state, reference_price, sizing_capital, params):
    candidate_plan = build_extended_candidate_plan_from_signal(signal_state, reference_price, sizing_capital, params)
    if candidate_plan is None or candidate_plan['qty'] <= 0:
        return None
    return candidate_plan


# # (AI註: 延續候選資格評估)
def evaluate_extended_candidate_eligibility(close_price, original_limit, atr, sizing_capital, params):
    signal_state = create_signal_tracking_state(original_limit, atr, params)
    return build_extended_candidate_plan_from_signal(signal_state, close_price, sizing_capital, params)


# # (AI註: 單一真理來源 - K棒推進與結算，徹底消滅 Portfolio 與 Backtest 分歧)
def execute_bar_step(position, y_atr, y_ind_sell, y_close, t_open, t_high, t_low, t_close, t_volume, params):
    freed_cash, pnl_realized = 0.0, 0.0
    events = []

    if position['qty'] <= 0:
        return position, freed_cash, pnl_realized, events

    if y_close > position['pure_buy_price'] + (y_atr * params.atr_times_trail):
        new_trail = adjust_long_stop_price(y_close - (y_atr * params.atr_times_trail))
        position['trailing_stop'] = max(position.get('trailing_stop', 0.0), new_trail)
        position['sl'] = max(position['initial_stop'], position['trailing_stop'])

    # # (AI註: y_ind_sell 是 T-1 收盤後已知、T 日盤前即可決定的賣出指令，
    # # (AI註: 必須優先於 T 日盤中的 TP_HALF / STOP 判斷，避免出現不可能的事件序列)
    if y_ind_sell:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close)
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(t_open)
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
            events.append('IND_SELL')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

        return position, freed_cash, pnl_realized, events

    is_stop_hit = t_low <= position['sl']
    half_sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    is_tp_hit = t_high >= position['tp_half'] and not position['sold_half'] and half_sell_qty > 0

    # # (AI註: 同棒同時碰停損與半倉停利時，維持最壞情境，優先視為停損)
    if is_stop_hit and is_tp_hit:
        is_tp_hit = False

    if is_tp_hit and not (pd.isna(t_volume) or t_volume <= 0):
        exec_price = adjust_long_sell_fill_price(max(position['tp_half'], t_open))
        sell_qty = half_sell_qty
        net_price = calc_net_sell_price(exec_price, sell_qty, params)
        freed_cash = net_price * sell_qty
        pnl = (net_price - position['entry']) * sell_qty
        pnl_realized += pnl
        position['realized_pnl'] += pnl
        position['qty'] -= sell_qty
        position['sold_half'] = True
        events.append('TP_HALF')

    if is_stop_hit:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close)
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(min(position['sl'], t_open))
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
            events.append('STOP')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

    return position, freed_cash, pnl_realized, events

def tv_rma(source, length):
    source = np.asarray(source, dtype=np.float64)
    rma = np.full(source.shape, np.nan, dtype=np.float64)
    valid_idx = np.where(~np.isnan(source))[0]
    if len(valid_idx) < length: return rma
    first_valid = valid_idx[length - 1]
    rma[first_valid] = np.mean(source[valid_idx[0]:first_valid + 1])
    alpha = 1.0 / length
    for i in range(first_valid + 1, len(source)):
        if not np.isnan(source[i]): rma[i] = alpha * source[i] + (1 - alpha) * rma[i - 1]
        else: rma[i] = rma[i - 1]
    return rma

def tv_atr(high, low, close, length):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0], tr3[0] = np.nan, np.nan 
    tr = np.nanmax([tr1, tr2, tr3], axis=0)
    return tv_rma(tr, length)

def tv_ema(source, length):
    source = np.asarray(source, dtype=np.float64)
    ema = np.full(source.shape, np.nan, dtype=np.float64)
    valid_idx = np.where(~np.isnan(source))[0]
    if len(valid_idx) == 0: return ema
    first_valid = valid_idx[0]
    ema[first_valid] = source[first_valid]
    alpha = 2.0 / (length + 1)
    for i in range(first_valid + 1, len(source)):
        if not np.isnan(source[i]):
            if np.isnan(ema[i-1]): ema[i] = source[i]
            else: ema[i] = alpha * source[i] + (1 - alpha) * ema[i - 1]
        else: ema[i] = ema[i - 1]
    return ema

def tv_supertrend(high, low, close, atr, multiplier):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub = np.full(close.shape, np.nan, dtype=np.float64)
    final_lb = np.full(close.shape, np.nan, dtype=np.float64)
    direction = np.full(close.shape, 1, dtype=np.int8) 
    first_valid = np.where(~np.isnan(atr))[0]
    if len(first_valid) == 0: return direction
    first = first_valid[0]
    final_ub[first], final_lb[first] = basic_ub[first], basic_lb[first]
    for i in range(first + 1, len(close)):
        if close[i-1] > final_lb[i-1]: final_lb[i] = max(basic_lb[i], final_lb[i-1])
        else: final_lb[i] = basic_lb[i]
        if close[i-1] < final_ub[i-1]: final_ub[i] = min(basic_ub[i], final_ub[i-1])
        else: final_ub[i] = basic_ub[i]
        if direction[i-1] == -1 and close[i] < final_lb[i-1]: direction[i] = 1
        elif direction[i-1] == 1 and close[i] > final_ub[i-1]: direction[i] = -1
        else: direction[i] = direction[i-1]
    return direction

def generate_signals(df, params):
    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    is_tradable_bar = V > 0
    ATR_main = tv_atr(H, L, C, params.atr_len)

    close_series = pd.Series(C)
    high_series = pd.Series(H)
    volume_series = pd.Series(V)
    HighN = high_series.shift(1).rolling(params.high_len, min_periods=params.high_len).max().values
    SuperTrend_Dir = tv_supertrend(H, L, C, ATR_main, params.atr_times_trail)

    prev_supertrend = np.empty_like(SuperTrend_Dir)
    prev_supertrend[0] = SuperTrend_Dir[0]
    prev_supertrend[1:] = SuperTrend_Dir[:-1]
    isSupertrend_Bearish_Flip = (SuperTrend_Dir == 1) & (prev_supertrend == -1)
    isSupertrend_Bearish_Flip[0] = False

    prev_close = np.empty_like(C)
    prev_close[0] = C[0]
    prev_close[1:] = C[:-1]
    prev_highn = np.empty_like(HighN)
    prev_highn[0] = HighN[0]
    prev_highn[1:] = HighN[:-1]
    isPriceCrossover = (C > HighN) & (prev_close <= prev_highn)
    isPriceCrossover[0] = False

    if getattr(params, 'use_bb', True):
        BB_Mid = close_series.rolling(params.bb_len).mean().values
        BB_Upper = BB_Mid + params.bb_mult * close_series.rolling(params.bb_len).std(ddof=0).values
        bbCondition = (C > BB_Upper)
    else:
        bbCondition = np.ones_like(C, dtype=bool)

    if getattr(params, 'use_vol', True):
        VolS = volume_series.rolling(params.vol_short_len).mean().values
        VolL = volume_series.rolling(params.vol_long_len).mean().values
        volCondition = np.isnan(V) | (VolS > VolL)
    else:
        volCondition = np.ones_like(C, dtype=bool)

    if getattr(params, 'use_kc', True):
        ATR_kc = tv_atr(H, L, C, params.kc_len)
        KC_Mid = tv_ema(C, params.kc_len)
        KC_Lower = KC_Mid - ATR_kc * params.kc_mult
        prev_kc_lower = np.empty_like(KC_Lower)
        prev_kc_lower[0] = KC_Lower[0]
        prev_kc_lower[1:] = KC_Lower[:-1]
        isKcCrossunder = (C < KC_Lower) & (prev_close >= prev_kc_lower)
        isKcCrossunder[0] = False
        kcSellCondition = isKcCrossunder & (C < O)
    else:
        kcSellCondition = np.zeros_like(C, dtype=bool)

    buyCondition = is_tradable_bar & (C > O) & isPriceCrossover & bbCondition & volCondition
    sellCondition = is_tradable_bar & ((isSupertrend_Bearish_Flip & (C <= O)) | kcSellCondition)
    raw_buy_limits = C + ATR_main * params.atr_buy_tol
    buy_limits = np.full_like(C, np.nan)
    valid_buy_mask = buyCondition & ~np.isnan(raw_buy_limits)
    if np.any(valid_buy_mask):
        buy_limits[valid_buy_mask] = adjust_long_buy_limit_array(raw_buy_limits[valid_buy_mask])
    return ATR_main, buyCondition, sellCondition, buy_limits

def run_v16_backtest(df, params=None, return_logs=False, precomputed_signals=None):
    if params is None:
        params = V16StrategyParams()
    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    Dates = df.index
    if precomputed_signals is None:
        ATR_main, buyCondition, sellCondition, buy_limits = generate_signals(df, params)
    else:
        ATR_main, buyCondition, sellCondition, buy_limits = precomputed_signals

    position = {'qty': 0}
    active_extended_signal = None
    currentCapital = params.initial_capital
    tradeCount, fullWins, missedBuyCount, missedSellCount = 0, 0, 0, 0
    totalProfit, totalLoss = 0.0, 0.0
    peakCapital, maxDrawdownPct = currentCapital, 0.0
    total_r_multiple, total_r_win, total_r_loss, total_bars_held = 0.0, 0.0, 0.0, 0
    trade_logs = []
    currentEquity = currentCapital

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j-1]):
            continue
        pos_start_of_current_bar = position['qty']

        if pos_start_of_current_bar > 0:
            total_bars_held += 1
            position, freed_cash, pnl_realized, events = execute_bar_step(
                position, ATR_main[j-1], sellCondition[j-1], C[j-1],
                O[j], H[j], L[j], C[j], V[j], params
            )
            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl = position['realized_pnl']
                trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0
                total_r_multiple += trade_r_mult
                tradeCount += 1
                if return_logs:
                    trade_logs.append({'exit_date': Dates[j], 'pnl': total_pnl, 'r_mult': trade_r_mult})
                if total_pnl > 0:
                    fullWins += 1
                    totalProfit += total_pnl
                    total_r_win += trade_r_mult
                else:
                    totalLoss += abs(total_pnl)
                    total_r_loss += abs(trade_r_mult)
            elif 'MISSED_SELL' in events:
                missedSellCount += 1
            currentCapital += pnl_realized

        isSetup_prev = buyCondition[j-1] and (pos_start_of_current_bar == 0)
        buyTriggered = False
        sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital

        if isSetup_prev:
            signal_state = create_signal_tracking_state(buy_limits[j-1], ATR_main[j-1], params)
            if signal_state is not None:
                active_extended_signal = signal_state

            entry_plan = build_normal_entry_plan(buy_limits[j-1], ATR_main[j-1], sizing_cap, params)
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j-1],
                params=params,
                entry_type='normal',
            )
            if entry_result['filled']:
                position = entry_result['position']
                buyTriggered = True
                active_extended_signal = None
            elif entry_result['count_as_missed_buy']:
                missedBuyCount += 1

        elif active_extended_signal is not None and pos_start_of_current_bar == 0:
            entry_plan = build_extended_entry_plan_from_signal(active_extended_signal, C[j-1], sizing_cap, params)
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j-1],
                params=params,
                entry_type='extended',
            )
            if entry_result['filled']:
                position = entry_result['position']
                buyTriggered = True
                active_extended_signal = None
            elif entry_result['count_as_missed_buy']:
                missedBuyCount += 1

        if not buyTriggered and position['qty'] == 0 and should_clear_extended_signal(active_extended_signal, L[j]):
            active_extended_signal = None

        currentEquity = currentCapital
        if position['qty'] > 0:
            floating_exec_price = adjust_long_sell_fill_price(C[j])
            floatingSellNet = calc_net_sell_price(floating_exec_price, position['qty'], params)
            floatingPnL = (floatingSellNet - position['entry']) * position['qty']
            currentEquity = currentCapital + floatingPnL

        peakCapital = max(peakCapital, currentEquity)
        currentDrawdownPct = ((peakCapital - currentEquity) / peakCapital) * 100 if peakCapital > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

    end_position_qty = position['qty']
    had_open_position_at_end = end_position_qty > 0

    if had_open_position_at_end:
        exec_price = adjust_long_sell_fill_price(C[-1])
        net_price = calc_net_sell_price(exec_price, position['qty'], params)
        pnl = (net_price - position['entry']) * position['qty']
        total_pnl = position['realized_pnl'] + pnl
        trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0.0

        total_r_multiple += trade_r_mult
        tradeCount += 1

        if return_logs:
            trade_logs.append({
                'exit_date': Dates[-1],
                'pnl': total_pnl,
                'r_mult': trade_r_mult
            })

        if total_pnl > 0:
            fullWins += 1
            totalProfit += total_pnl
            total_r_win += trade_r_mult
        else:
            totalLoss += abs(total_pnl)
            total_r_loss += abs(trade_r_mult)

        currentCapital += pnl
        currentEquity = currentCapital

        peakCapital = max(peakCapital, currentEquity)
        currentDrawdownPct = ((peakCapital - currentEquity) / peakCapital) * 100 if peakCapital > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

        position['qty'] = 0

    winRate = (fullWins / tradeCount * 100) if tradeCount > 0 else 0
    avgWin = totalProfit / fullWins if fullWins > 0 else 0
    lossCount = tradeCount - fullWins
    avgLoss = totalLoss / lossCount if lossCount > 0 else 0
    payoffRatio = (avgWin / avgLoss) if avgLoss > 0 else (99.9 if avgWin > 0 else 0.0)

    if EV_CALC_METHOD == 'B':
        win_rate_dec = fullWins / tradeCount if tradeCount > 0 else 0
        avg_win_r = total_r_win / fullWins if fullWins > 0 else 0
        avg_loss_r = total_r_loss / lossCount if lossCount > 0 else 0
        payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
        expectedValue = (win_rate_dec * payoff_for_ev) - (1 - win_rate_dec)
    else:
        expectedValue = (total_r_multiple / tradeCount) if tradeCount > 0 else 0.0

    finalEquity = currentEquity
    totalNetProfitPct = ((finalEquity - params.initial_capital) / params.initial_capital) * 100
    score = totalNetProfitPct / tradeCount if tradeCount > 0 else 0

    isSetup_today = buyCondition[-1] and (not had_open_position_at_end)
    buyLimit_today = adjust_long_buy_limit(C[-1] + ATR_main[-1] * params.atr_buy_tol) if isSetup_today else np.nan
    stopLoss_today = adjust_long_stop_price(buyLimit_today - ATR_main[-1] * params.atr_times_init) if isSetup_today else np.nan

    isCandidate, _, _, _ = evaluate_history_candidate_metrics(
        tradeCount,
        fullWins,
        total_r_multiple,
        total_r_win,
        total_r_loss,
        params,
    )

    extended_candidate_today = None
    if (not had_open_position_at_end) and active_extended_signal is not None:
        sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
        extended_candidate_today = build_extended_candidate_plan_from_signal(active_extended_signal, C[-1], sizing_cap, params)

    avg_bars_held = total_bars_held / tradeCount if tradeCount > 0 else 0

    stats_dict = {
        "asset_growth": totalNetProfitPct,
        "trade_count": tradeCount,
        "missed_buys": missedBuyCount,
        "missed_sells": missedSellCount,
        "score": score,
        "win_rate": winRate,
        "avg_win": avgWin,
        "avg_loss": avgLoss,
        "payoff_ratio": payoffRatio,
        "expected_value": expectedValue,
        "max_drawdown": maxDrawdownPct,
        "is_candidate": isCandidate,
        "is_setup_today": isSetup_today,
        "buy_limit": buyLimit_today,
        "stop_loss": stopLoss_today,
        "extended_candidate_today": extended_candidate_today,
        "current_position": end_position_qty,
        "avg_bars_held": avg_bars_held
    }

    if return_logs: return stats_dict, trade_logs
    return stats_dict