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

def adjust_to_tick(price):
    if pd.isna(price): return np.nan
    tick = get_tick_size(price)
    return tv_round(price / tick) * tick

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

# # (AI註: 單一真理來源 - 精準 1R 數學空間推算)
def evaluate_chase_condition(close_price, original_limit, atr, params):
    if pd.isna(close_price) or pd.isna(original_limit) or pd.isna(atr): return None
    original_sl = adjust_to_tick(original_limit - atr * params.atr_times_init)
    
    risk_price_dist = original_limit - original_sl
    if risk_price_dist <= 0: return None
    original_tp = adjust_to_tick(original_limit + risk_price_dist)
    
    if original_sl < close_price < original_tp:
        current_risk = close_price - original_sl
        current_reward = original_tp - close_price
        rr_threshold = getattr(params, 'min_chase_rr', 0.5) 
        if current_risk > 0 and (current_reward / current_risk) >= rr_threshold:
            return {'chase_price': close_price, 'sl': original_sl, 'tp': original_tp, 'rr': current_reward/current_risk}
    return None

# # (AI註: 單一真理來源 - K棒推進與結算，杜絕 Portfolio 與 Backtest 分歧)
def execute_bar_step(position, y_atr, y_ind_sell, y_close, t_open, t_high, t_low, t_close, params):
    freed_cash, pnl_realized = 0.0, 0.0
    events = []
    if position['qty'] <= 0: return position, freed_cash, pnl_realized, events
        
    if y_close > position['pure_buy_price'] + (y_atr * params.atr_times_trail):
        new_trail = adjust_to_tick(y_close - (y_atr * params.atr_times_trail))
        position['trailing_stop'] = max(position.get('trailing_stop', 0.0), new_trail)
        position['sl'] = max(position['initial_stop'], position['trailing_stop'])

    is_stop_hit = t_low <= position['sl']
    is_tp_hit = t_high >= position['tp_half'] and not position['sold_half']
    if is_stop_hit and is_tp_hit: is_tp_hit = False

    if is_tp_hit:
        exec_price = adjust_to_tick(max(position['tp_half'], t_open))
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
            
    if is_stop_hit or y_ind_sell:
        is_locked_down = (t_open == t_high) and (t_high == t_low) and (t_low == t_close) and (t_close < y_close)
        if not is_locked_down:
            exec_price = adjust_to_tick(min(position['sl'], t_open) if is_stop_hit else t_open)
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
            events.append('STOP' if is_stop_hit else 'IND_SELL')
        else:
            events.append('LOCKED_DOWN')
            
    return position, freed_cash, pnl_realized, events

def tv_rma(source, length):
    rma = np.full_like(source, np.nan)
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
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0], tr3[0] = np.nan, np.nan 
    tr = np.nanmax([tr1, tr2, tr3], axis=0)
    return tv_rma(tr, length)

def tv_ema(source, length):
    ema = np.full_like(source, np.nan)
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
    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub, final_lb, direction = np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, 1) 
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
    H, L, C = df['High'].values, df['Low'].values, df['Close'].values
    O, V = df['Open'].values, df['Volume'].values
    ATR_main = tv_atr(H, L, C, params.atr_len)
    HighN = pd.Series(H).shift(1).rolling(params.high_len, min_periods=1).max().values
    SuperTrend_Dir = tv_supertrend(H, L, C, ATR_main, params.atr_times_trail)
    isSupertrend_Bearish_Flip = (SuperTrend_Dir == 1) & (np.roll(SuperTrend_Dir, 1) == -1)
    isSupertrend_Bearish_Flip[0] = False
    isPriceCrossover = (C > HighN) & (np.roll(C, 1) <= np.roll(HighN, 1))
    isPriceCrossover[0] = False
    
    if getattr(params, 'use_bb', True):
        BB_Mid = pd.Series(C).rolling(params.bb_len).mean().values
        BB_Upper = BB_Mid + params.bb_mult * pd.Series(C).rolling(params.bb_len).std(ddof=0).values
        bbCondition = (C > BB_Upper)
    else: bbCondition = np.ones_like(C, dtype=bool) 
    if getattr(params, 'use_vol', True):
        VolS = pd.Series(V).rolling(params.vol_short_len).mean().values
        VolL = pd.Series(V).rolling(params.vol_long_len).mean().values
        volCondition = np.isnan(V) | (VolS > VolL)
    else: volCondition = np.ones_like(C, dtype=bool)
    if getattr(params, 'use_kc', True):
        ATR_kc = tv_atr(H, L, C, params.kc_len)
        KC_Mid = tv_ema(C, params.kc_len)
        KC_Lower = KC_Mid - ATR_kc * params.kc_mult
        isKcCrossunder = (C < KC_Lower) & (np.roll(C, 1) >= np.roll(KC_Lower, 1))
        isKcCrossunder[0] = False
        kcSellCondition = (isKcCrossunder & (C < O))
    else: kcSellCondition = np.zeros_like(C, dtype=bool) 
    
    buyCondition = (C > O) & isPriceCrossover & bbCondition & volCondition
    sellCondition = (isSupertrend_Bearish_Flip & (C <= O)) | kcSellCondition
    buy_limits = np.full_like(C, np.nan)
    for i in range(len(C)):
        if buyCondition[i]: buy_limits[i] = adjust_to_tick(C[i] + ATR_main[i] * params.atr_buy_tol)
    return ATR_main, buyCondition, sellCondition, buy_limits

def run_v16_backtest(df, params: V16StrategyParams = V16StrategyParams()):
    H, L, C = df['High'].values, df['Low'].values, df['Close'].values
    O, V = df['Open'].values, df['Volume'].values
    ATR_main, buyCondition, sellCondition, buy_limits = generate_signals(df, params)

    position = {'qty': 0}
    pending_chase = None
    currentCapital = params.initial_capital
    tradeCount, fullWins, missedBuyCount, missedSellCount = 0, 0, 0, 0
    totalProfit, totalLoss = 0.0, 0.0
    peakCapital, maxDrawdownPct = currentCapital, 0.0
    total_r_multiple, total_r_win, total_r_loss, total_bars_held = 0.0, 0.0, 0.0, 0

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j-1]): continue
        pos_start_of_current_bar = position['qty']
        
        # 1. 執行 T 日的持倉結算
        if pos_start_of_current_bar > 0:
            total_bars_held += 1
            position, freed_cash, pnl_realized, events = execute_bar_step(
                position, ATR_main[j-1], sellCondition[j-1], C[j-1], 
                O[j], H[j], L[j], C[j], params
            )
            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl = position['realized_pnl']
                trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0
                total_r_multiple += trade_r_mult
                tradeCount += 1
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
            # # (AI註: 嚴守盤前資金定錨，用 T-1 算死 Stop Loss 與 Qty)
            buyLimitPrice = adjust_to_tick(C[j-1] + ATR_main[j-1] * params.atr_buy_tol)
            planned_init_sl = adjust_to_tick(buyLimitPrice - ATR_main[j-1] * params.atr_times_init)
            planned_init_trail = adjust_to_tick(buyLimitPrice - ATR_main[j-1] * params.atr_times_trail)
            
            sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
            buyQty = calc_position_size(buyLimitPrice, planned_init_sl, sizing_cap, params.fixed_risk, params)
            
            if L[j] <= buyLimitPrice and not is_locked_limit_up and buyQty > 0:
                buyPrice = adjust_to_tick(min(O[j], buyLimitPrice))
                # 確保跳空暴跌沒有擊穿盤前設定的停損防線才進場
                if buyPrice > planned_init_sl:
                    entryPrice = calc_entry_price(buyPrice, buyQty, params)
                    net_sl = calc_net_sell_price(planned_init_sl, buyQty, params)
                    tp_half = adjust_to_tick(buyPrice + (entryPrice - net_sl))
                    init_risk = (entryPrice - net_sl) * buyQty
                    if init_risk <= 0: init_risk = sizing_cap * 0.01
                    
                    position = {
                        'qty': buyQty, 'entry': entryPrice, 'sl': max(planned_init_sl, planned_init_trail),
                        'initial_stop': planned_init_sl, 'trailing_stop': planned_init_trail,
                        'tp_half': tp_half, 'sold_half': False, 'pure_buy_price': buyPrice,
                        'realized_pnl': 0.0, 'initial_risk_total': init_risk
                    }
                    buyTriggered = True
                    pending_chase = None
                else:
                    missedBuyCount += 1
            else:
                missedBuyCount += 1
                chase_res = evaluate_chase_condition(C[j], buyLimitPrice, ATR_main[j-1], params)
                pending_chase = chase_res if chase_res else None

        elif pending_chase is not None and pos_start_of_current_bar == 0:
            chase_limit = pending_chase['chase_price']
            planned_init_sl = pending_chase['sl']
            sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
            buyQty = calc_position_size(chase_limit, planned_init_sl, sizing_cap, params.fixed_risk, params)
            
            if L[j] <= chase_limit and not is_locked_limit_up and buyQty > 0:
                buyPrice = adjust_to_tick(min(O[j], chase_limit))
                if buyPrice > planned_init_sl:
                    entryPrice = calc_entry_price(buyPrice, buyQty, params)
                    net_sl = calc_net_sell_price(planned_init_sl, buyQty, params)
                    init_risk = (entryPrice - net_sl) * buyQty
                    if init_risk <= 0: init_risk = sizing_cap * 0.01
                    
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
                    risk = C[j] - pending_chase['sl']
                    reward = pending_chase['tp'] - C[j]
                    rr_threshold = getattr(params, 'min_chase_rr', 0.5)
                    if risk > 0 and (reward / risk) >= rr_threshold:
                        pending_chase['chase_price'] = C[j]
                    else: pending_chase = None
                else: pending_chase = None

        currentEquity = currentCapital
        if position['qty'] > 0:
            floatingSellNet = calc_net_sell_price(C[j], position['qty'], params)
            floatingPnL = position['realized_pnl'] + (floatingSellNet - position['entry']) * position['qty']
            currentEquity = currentCapital + floatingPnL

        peakCapital = max(peakCapital, currentEquity)
        currentDrawdownPct = ((peakCapital - currentEquity) / peakCapital) * 100 if peakCapital > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

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

    totalNetProfitPct = ((currentCapital - params.initial_capital) / params.initial_capital) * 100
    score = totalNetProfitPct / tradeCount if tradeCount > 0 else 0
    
    isSetup_today = buyCondition[-1] and (position['qty'] == 0)
    buyLimit_today = adjust_to_tick(C[-1] + ATR_main[-1] * params.atr_buy_tol) if isSetup_today else np.nan
    stopLoss_today = adjust_to_tick(buyLimit_today - ATR_main[-1] * params.atr_times_init) if isSetup_today else np.nan

    min_trades = getattr(params, 'min_history_trades', 0)
    min_win_rate = getattr(params, 'min_history_win_rate', 0.30) * 100
    min_ev = getattr(params, 'min_history_ev', 0.0)

    if tradeCount < min_trades: isCandidate = False
    elif tradeCount == 0 and min_trades == 0: isCandidate = True
    else: isCandidate = (winRate >= min_win_rate) and (expectedValue > min_ev)

    chase_today = None
    last_setup_idx = np.where(buyCondition)[0]
    if len(last_setup_idx) > 0 and position['qty'] == 0:
        last_idx = last_setup_idx[-1]
        days_since_setup = (len(C) - 1) - last_idx
        if 0 < days_since_setup <= 3: 
            last_limit = adjust_to_tick(C[last_idx] + ATR_main[last_idx] * params.atr_buy_tol)
            chase_today = evaluate_chase_condition(C[-1], last_limit, ATR_main[last_idx], params)
            if chase_today: chase_today['days_since_setup'] = days_since_setup 

    avg_bars_held = total_bars_held / tradeCount if tradeCount > 0 else 0

    return {
        "asset_growth": totalNetProfitPct, "trade_count": tradeCount, "missed_buys": missedBuyCount,
        "missed_sells": missedSellCount, "score": score, "win_rate": winRate, "payoff_ratio": payoffRatio, 
        "expected_value": expectedValue, "max_drawdown": maxDrawdownPct, "is_candidate": isCandidate, 
        "is_setup_today": isSetup_today, "buy_limit": buyLimit_today, "stop_loss": stopLoss_today, 
        "chase_today": chase_today, "current_position": position['qty'], "avg_bars_held": avg_bars_held  
    }