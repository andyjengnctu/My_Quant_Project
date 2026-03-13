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
    fee = max(bPrice * bQty * params.buy_fee, params.min_fee)
    return bPrice + (fee / bQty)

def calc_net_sell_price(sPrice, sQty, params):
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

# # (AI註: 單一真理來源 - 統一 initial_risk_total 的計算與 fallback 口徑，禁止散落 magic number)
def calc_initial_risk_total(entry_price, net_stop_price, qty, params):
    if pd.isna(entry_price) or pd.isna(net_stop_price) or qty <= 0:
        return 0.0

    init_risk = (entry_price - net_stop_price) * qty
    if init_risk > 0:
        return init_risk

    actual_total_cost = entry_price * qty
    return max(actual_total_cost * params.fixed_risk, 0.0)

# # (AI註: 單一真理來源 - 正常單的 limit/sl/trail/qty 規格統一由此產生)
def build_normal_entry_plan(limit_price, atr, sizing_capital, params):
    if pd.isna(limit_price) or pd.isna(atr):
        return None

    init_sl = adjust_long_stop_price(limit_price - atr * params.atr_times_init)
    init_trail = adjust_long_stop_price(limit_price - atr * params.atr_times_trail)
    qty = calc_position_size(limit_price, init_sl, sizing_capital, params.fixed_risk, params)
    if qty <= 0:
        return None

    return {
        'limit_price': limit_price,
        'init_sl': init_sl,
        'init_trail': init_trail,
        'qty': qty,
    }

# # (AI註: 單一真理來源 - 實現「絕對精準 1R」，考量縮倉與雙邊手續費計算 RR)
def evaluate_chase_condition(close_price, original_limit, atr, sizing_capital, params):
    if pd.isna(close_price) or pd.isna(original_limit) or pd.isna(atr): return None
    
    planned_init_sl = adjust_long_stop_price(original_limit - atr * params.atr_times_init)
    
    # 1. 精準還原原始策略 1R 目標價
    orig_qty = calc_position_size(original_limit, planned_init_sl, sizing_capital, params.fixed_risk, params)
    if orig_qty <= 0: return None
    orig_entry_price = calc_entry_price(original_limit, orig_qty, params)
    orig_net_sl = calc_net_sell_price(planned_init_sl, orig_qty, params)
    orig_risk_per_share = orig_entry_price - orig_net_sl
    original_tp = adjust_long_target_price(original_limit + orig_risk_per_share)
    
    if planned_init_sl < close_price < original_tp:
        # 2. 確切計算追車縮減股數與實際成本盈虧比
        chase_qty = calc_position_size(close_price, planned_init_sl, sizing_capital, params.fixed_risk, params)
        if chase_qty <= 0: return None
        
        chase_entry_price = calc_entry_price(close_price, chase_qty, params)
        chase_net_sl = calc_net_sell_price(planned_init_sl, chase_qty, params)
        chase_risk_total = (chase_entry_price - chase_net_sl) * chase_qty
        if chase_risk_total <= 0: return None
        
        chase_net_tp = calc_net_sell_price(original_tp, chase_qty, params)
        chase_reward_total = (chase_net_tp - chase_entry_price) * chase_qty
        
        rr_threshold = getattr(params, 'min_chase_rr', 0.5) 
        if chase_reward_total > 0 and (chase_reward_total / chase_risk_total) >= rr_threshold:
            return {
                'chase_price': close_price, 'sl': planned_init_sl, 'tp': original_tp, 
                'rr': chase_reward_total / chase_risk_total, 'qty': chase_qty,
                'orig_limit': original_limit, 'orig_atr': atr 
            }
    return None

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

    # # (AI註: 零成交量日保留在時間序列中，但當日不可成交；
    # # (AI註: 僅允許先用前一日已知資訊更新停損線，再禁止當日任何賣出事件)
    if pd.isna(t_volume) or t_volume <= 0:
        return position, freed_cash, pnl_realized, events

    # # (AI註: y_ind_sell 是 T-1 收盤後已知、T 日盤前即可決定的賣出指令，
    # # (AI註: 必須優先於 T 日盤中的 TP_HALF / STOP 判斷，避免出現不可能的事件序列)
    if y_ind_sell:
        is_locked_down = (
            (t_open == t_high) and
            (t_high == t_low) and
            (t_low == t_close) and
            (t_close < y_close)
        )

        if not is_locked_down:
            exec_price = adjust_long_sell_fill_price(t_open)
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
            events.append('IND_SELL')
        else:
            events.append('LOCKED_DOWN')

        return position, freed_cash, pnl_realized, events

    is_stop_hit = t_low <= position['sl']
    is_tp_hit = t_high >= position['tp_half'] and not position['sold_half']

    # # (AI註: 同棒同時碰停損與半倉停利時，維持最壞情境，優先視為停損)
    if is_stop_hit and is_tp_hit:
        is_tp_hit = False

    if is_tp_hit:
        exec_price = adjust_long_sell_fill_price(max(position['tp_half'], t_open))
        sell_qty = int(math.floor(position['qty'] * params.tp_percent))
        if sell_qty > 0 and position['qty'] > sell_qty:
            net_price = calc_net_sell_price(exec_price, sell_qty, params)
            freed_cash = net_price * sell_qty
            pnl = (net_price - position['entry']) * sell_qty
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] -= sell_qty
            position['sold_half'] = True
            events.append('TP_HALF')

    if is_stop_hit:
        is_locked_down = (
            (t_open == t_high) and
            (t_high == t_low) and
            (t_low == t_close) and
            (t_close < y_close)
        )

        if not is_locked_down:
            exec_price = adjust_long_sell_fill_price(min(position['sl'], t_open))
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
            events.append('STOP')
        else:
            events.append('LOCKED_DOWN')

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

def run_v16_backtest(df, params: V16StrategyParams = V16StrategyParams(), return_logs=False, precomputed_signals=None):
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
    pending_chase = None
    currentCapital = params.initial_capital
    tradeCount, fullWins, missedBuyCount, missedSellCount = 0, 0, 0, 0
    totalProfit, totalLoss = 0.0, 0.0
    peakCapital, maxDrawdownPct = currentCapital, 0.0
    total_r_multiple, total_r_win, total_r_loss, total_bars_held = 0.0, 0.0, 0.0, 0
    trade_logs = []
    currentEquity = currentCapital

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j-1]): continue
        pos_start_of_current_bar = position['qty']
        
        # 1. 執行 T 日的持倉結算
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
            elif 'LOCKED_DOWN' in events:
                missedSellCount += 1
            currentCapital += pnl_realized
            
        # 2. 處理 T 日進場
        isSetup_prev = buyCondition[j-1] and (pos_start_of_current_bar == 0)
        is_locked_limit_up = (O[j] == H[j]) and (H[j] == L[j]) and (L[j] == C[j]) and (C[j] > C[j-1])
        buyTriggered = False
        
        if isSetup_prev:
            # # (AI註: 問題1修復 - 嚴格盤前定錨，qty與sl鎖死在 T-1 buyLimitPrice)
            buyLimitPrice = buy_limits[j-1]
            planned_init_sl = adjust_long_stop_price(buyLimitPrice - ATR_main[j-1] * params.atr_times_init)
            planned_init_trail = adjust_long_stop_price(buyLimitPrice - ATR_main[j-1] * params.atr_times_trail)
            sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
            buyQty = calc_position_size(buyLimitPrice, planned_init_sl, sizing_cap, params.fixed_risk, params)
            
            if V[j] > 0 and L[j] <= buyLimitPrice and not is_locked_limit_up and buyQty > 0:
                buyPrice = adjust_long_buy_fill_price(min(O[j], buyLimitPrice))
                # 確保跳空暴跌沒擊穿盤前停損死線
                if buyPrice > planned_init_sl:
                    entryPrice = calc_entry_price(buyPrice, buyQty, params)
                    net_sl = calc_net_sell_price(planned_init_sl, buyQty, params)
                    tp_half = adjust_long_target_price(buyPrice + (entryPrice - net_sl))
                    init_risk = calc_initial_risk_total(entryPrice, net_sl, buyQty, params)
                    
                    position = {
                        'qty': buyQty, 'entry': entryPrice, 'sl': max(planned_init_sl, planned_init_trail),
                        'initial_stop': planned_init_sl, 'trailing_stop': planned_init_trail,
                        'tp_half': tp_half, 'sold_half': False, 'pure_buy_price': buyPrice,
                        'realized_pnl': 0.0, 'initial_risk_total': init_risk
                    }
                    buyTriggered = True
                    pending_chase = None
                else:
                    missedBuyCount += 1 # 放棄交易
            else:
                missedBuyCount += 1
                chase_res = evaluate_chase_condition(C[j], buyLimitPrice, ATR_main[j-1], sizing_cap, params)
                pending_chase = chase_res if chase_res else None

        elif pending_chase is not None and pos_start_of_current_bar == 0:
            chase_limit = pending_chase['chase_price']
            planned_init_sl = pending_chase['sl']
            sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
            buyQty = pending_chase['qty']
            
            if V[j] > 0 and L[j] <= chase_limit and not is_locked_limit_up and buyQty > 0:
                buyPrice = adjust_long_buy_fill_price(min(O[j], chase_limit))
                if buyPrice > planned_init_sl:
                    entryPrice = calc_entry_price(buyPrice, buyQty, params)
                    net_sl = calc_net_sell_price(planned_init_sl, buyQty, params)
                    init_risk = calc_initial_risk_total(entryPrice, net_sl, buyQty, params)
                    
                    position = {
                        'qty': buyQty, 'entry': entryPrice, 'sl': planned_init_sl,
                        'initial_stop': planned_init_sl, 'trailing_stop': planned_init_sl,
                        'tp_half': pending_chase['tp'], 'sold_half': False, 'pure_buy_price': buyPrice,
                        'realized_pnl': 0.0, 'initial_risk_total': init_risk
                    }
                    buyTriggered = True
                    pending_chase = None
            
            # # (AI註: 統一續追邏輯)
            if not buyTriggered:
                if pending_chase['sl'] < C[j] < pending_chase['tp']:
                    chase_res = evaluate_chase_condition(C[j], pending_chase['orig_limit'], pending_chase['orig_atr'], sizing_cap, params)
                    pending_chase = chase_res if chase_res else None
                else: pending_chase = None

        currentEquity = currentCapital
        if position['qty'] > 0:
            # # (AI註: 浮動權益也統一用保守可賣出價口徑，避免權益曲線 / MDD 比最終結算樂觀)
            floating_exec_price = adjust_long_sell_fill_price(C[j])
            floatingSellNet = calc_net_sell_price(floating_exec_price, position['qty'], params)
            floatingPnL = (floatingSellNet - position['entry']) * position['qty']
            currentEquity = currentCapital + floatingPnL

        peakCapital = max(peakCapital, currentEquity)
        currentDrawdownPct = ((peakCapital - currentEquity) / peakCapital) * 100 if peakCapital > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

    # # (AI註: 保留「原始回測終點持倉狀態」給 scanner / validate 使用；
    # # (AI註: 統計則另外做虛擬期末強制結算，避免 trade_count / EV / win_rate 少一筆)
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

        # # (AI註: currentCapital 平常只累加已實現損益，
        # # (AI註: 這裡只補上剩餘部位尚未實現的 pnl，不可加整筆賣出金額)
        currentCapital += pnl
        currentEquity = currentCapital

        # # (AI註: 期末強制結算後補做一次 peak / drawdown 更新，避免 final closeout 對 MDD 漏算)
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
    else: expectedValue = (total_r_multiple / tradeCount) if tradeCount > 0 else 0.0

    finalEquity = currentEquity
    totalNetProfitPct = ((finalEquity - params.initial_capital) / params.initial_capital) * 100
    score = totalNetProfitPct / tradeCount if tradeCount > 0 else 0
    
    isSetup_today = buyCondition[-1] and (not had_open_position_at_end)
    buyLimit_today = adjust_long_buy_limit(C[-1] + ATR_main[-1] * params.atr_buy_tol) if isSetup_today else np.nan
    stopLoss_today = adjust_long_stop_price(buyLimit_today - ATR_main[-1] * params.atr_times_init) if isSetup_today else np.nan

    min_trades = getattr(params, 'min_history_trades', 0)
    min_win_rate = getattr(params, 'min_history_win_rate', 0.30) * 100
    min_ev = getattr(params, 'min_history_ev', 0.0)

    # # (AI註: 零樣本不可繞過 EV / 勝率門檻；
    # # (AI註: 只有在使用者明確把三個門檻都放到完全寬鬆時，才允許零歷史直接通過)
    allow_zero_history = (
        (min_trades == 0) and
        (min_win_rate <= 0) and
        (min_ev <= 0)
    )

    if tradeCount < min_trades:
        isCandidate = False
    elif tradeCount == 0:
        isCandidate = allow_zero_history
    else:
        isCandidate = (winRate >= min_win_rate) and (expectedValue > min_ev)

    # # (AI註: 解除 Scanner 分歧 - 回測內部 pending_chase 活著，就直接回傳給 Scanner)
    chase_today = pending_chase if not had_open_position_at_end else None
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
        "chase_today": chase_today,
        "current_position": end_position_qty,
        "avg_bars_held": avg_bars_held
    }
    
    if return_logs: return stats_dict, trade_logs
    return stats_dict