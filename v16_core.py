import pandas as pd
import numpy as np
import math
from v16_config import V16StrategyParams

# ==========================================
# 0. 輔助函數
# ==========================================
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
    maxQty = cap / (bPrice * (1 + params.buy_fee))
    estEntryCost = bPrice * (1 + params.buy_fee)
    estExitNet = stopPrice * (1 - params.sell_fee - params.tax_rate)
    riskPerUnit = estEntryCost - estExitNet
    if riskPerUnit > 0:
        return int(math.floor(min(cap * riskPct / riskPerUnit, maxQty)))
    return 0

# ==========================================
# 1. 100% 完美復刻 TradingView 核心數學
# ==========================================
def tv_rma(source, length):
    rma = np.full_like(source, np.nan)
    valid_idx = np.where(~np.isnan(source))[0]
    if len(valid_idx) < length: return rma
    first_valid = valid_idx[length - 1]
    rma[first_valid] = np.mean(source[valid_idx[0]:first_valid + 1])
    alpha = 1.0 / length
    for i in range(first_valid + 1, len(source)):
        if not np.isnan(source[i]):
            rma[i] = alpha * source[i] + (1 - alpha) * rma[i - 1]
        else:
            rma[i] = rma[i - 1]
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
            if np.isnan(ema[i-1]):
                ema[i] = source[i]
            else:
                ema[i] = alpha * source[i] + (1 - alpha) * ema[i - 1]
        else:
            ema[i] = ema[i - 1]
    return ema

def tv_supertrend(high, low, close, atr, multiplier):
    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub = np.full_like(close, np.nan)
    final_lb = np.full_like(close, np.nan)
    direction = np.full_like(close, 1) 
    
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

# ==========================================
# 2. 回測核心引擎
# ==========================================
def run_v16_backtest(df, params: V16StrategyParams = V16StrategyParams()):
    H, L, C = df['High'].values, df['Low'].values, df['Close'].values
    O, V = df['Open'].values, df['Volume'].values

    ATR_main = tv_atr(H, L, C, params.atr_len)
    HighN = df['High'].shift(1).rolling(params.high_len, min_periods=1).max().values
    
    SuperTrend_Dir = tv_supertrend(H, L, C, ATR_main, params.atr_times_trail)
    isSupertrend_Bearish_Flip = (SuperTrend_Dir == 1) & (np.roll(SuperTrend_Dir, 1) == -1)
    isSupertrend_Bearish_Flip[0] = False
    
    isPriceCrossover = (C > HighN) & (np.roll(C, 1) <= np.roll(HighN, 1))
    isPriceCrossover[0] = False
    
    # 🌟 AI 動態開關：布林通道
    if params.use_bb:
        BB_Mid = df['Close'].rolling(params.bb_len).mean().values
        BB_Upper = BB_Mid + params.bb_mult * df['Close'].rolling(params.bb_len).std(ddof=0).values
        bbCondition = (C > BB_Upper)
    else:
        bbCondition = np.ones_like(C, dtype=bool) # 全數通過
        BB_Upper = np.ones_like(C) # 防呆

    # 🌟 AI 動態開關：成交量
    if params.use_vol:
        VolS = df['Volume'].rolling(params.vol_short_len).mean().values
        VolL = df['Volume'].rolling(params.vol_long_len).mean().values
        volCondition = np.isnan(V) | (VolS > VolL)
    else:
        volCondition = np.ones_like(C, dtype=bool)

    # 🌟 AI 動態開關：阿肯那通道 (KC)
    if params.use_kc:
        ATR_kc = tv_atr(H, L, C, params.kc_len)
        KC_Mid = tv_ema(C, params.kc_len)
        KC_Lower = KC_Mid - ATR_kc * params.kc_mult
        isKcCrossunder = (C < KC_Lower) & (np.roll(C, 1) >= np.roll(KC_Lower, 1))
        isKcCrossunder[0] = False
        kcSellCondition = (isKcCrossunder & (C < O))
    else:
        kcSellCondition = np.zeros_like(C, dtype=bool) # 永遠不觸發 KC 賣出

    # 最終買賣訊號組裝
    buyCondition = (C > O) & isPriceCrossover & bbCondition & volCondition
    sellCondition = (isSupertrend_Bearish_Flip & (C <= O)) | kcSellCondition

    positionSize = 0
    buyPrice, entryPrice = np.nan, np.nan
    initialStopLossPrice, trailingStopPrice = np.nan, np.nan
    sellPriceHalf = np.nan
    soldHalf = False
    cumulativeProfit = 0.0

    currentCapital = params.initial_capital
    
    tradeCount, fullWins = 0, 0
    totalProfit, totalLoss = 0.0, 0.0
    missedBuyCount = 0
    peakCapital, maxDrawdownPct = currentCapital, 0.0

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j-1]): continue
        
        pos_start_of_current_bar = positionSize
            
        if positionSize > 0 and C[j] > buyPrice + (ATR_main[j] * params.atr_times_trail):
            new_trail = C[j] - (ATR_main[j] * params.atr_times_trail)
            trailingStopPrice = adjust_to_tick(max(trailingStopPrice, new_trail))
            
        isSetup_prev = buyCondition[j-1] and (pos_start_of_current_bar == 0)
        buyLimitPrice = adjust_to_tick(C[j-1] + ATR_main[j-1] * params.atr_buy_tol) if isSetup_prev else np.nan
        buyTriggered = isSetup_prev and L[j] <= buyLimitPrice
        
        if isSetup_prev and not buyTriggered: missedBuyCount += 1
            
        if buyTriggered:
            buyPrice = adjust_to_tick(min(O[j], buyLimitPrice))
            initialStopLossPrice = adjust_to_tick(buyPrice - ATR_main[j-1] * params.atr_times_init)
            trailingStopPrice = adjust_to_tick(buyPrice - ATR_main[j-1] * params.atr_times_trail)
            soldHalf, cumulativeProfit = False, 0.0
            
            buyQty = calc_position_size(buyPrice, initialStopLossPrice, currentCapital, params.fixed_risk, params)
            
            if buyQty > 0:
                entryPrice = calc_entry_price(buyPrice, buyQty, params)
                sellPriceHalf = adjust_to_tick(buyPrice + (entryPrice - calc_net_sell_price(initialStopLossPrice, buyQty, params)))
                positionSize = buyQty

        isHoldingFromYesterday = (pos_start_of_current_bar > 0) and (not buyTriggered)
        
        if isHoldingFromYesterday:
            activeStopPrice = max(initialStopLossPrice, trailingStopPrice)
            isStopHit = L[j] <= activeStopPrice
            isTakeProfitHit = H[j] >= sellPriceHalf and not soldHalf
            isIndicatorSell = sellCondition[j-1]
            
            if isStopHit and isTakeProfitHit: isTakeProfitHit = False
                
            if isTakeProfitHit:
                execSellPriceHalf = adjust_to_tick(max(sellPriceHalf, O[j]))
                sellQtyHalf = int(math.floor(positionSize * params.tp_percent))
                if sellQtyHalf > 0 and positionSize > sellQtyHalf:
                    sellNetPriceHalf = calc_net_sell_price(execSellPriceHalf, sellQtyHalf, params)
                    cumulativeProfit += (sellNetPriceHalf - entryPrice) * sellQtyHalf
                    positionSize -= sellQtyHalf
                    soldHalf = True
                else:
                    isTakeProfitHit = False
                
            if isStopHit or isIndicatorSell:
                sellPrice = adjust_to_tick(min(activeStopPrice, O[j]) if isStopHit else O[j])
                sellQty = positionSize
                sellNetPrice = calc_net_sell_price(sellPrice, sellQty, params)
                profitValue = cumulativeProfit + (sellNetPrice - entryPrice) * sellQty
                
                if profitValue > 0:
                    fullWins += 1
                    totalProfit += profitValue
                else:
                    totalLoss += abs(profitValue)
                    
                currentCapital += profitValue
                positionSize = 0
                soldHalf = False
                tradeCount += 1 
                
        currentEquity = currentCapital
        if positionSize > 0:
            floatingSellNet = calc_net_sell_price(C[j], positionSize, params)
            floatingPnL = cumulativeProfit + (floatingSellNet - entryPrice) * positionSize
            currentEquity = currentCapital + floatingPnL

        peakCapital = max(peakCapital, currentEquity)
        currentDrawdownPct = ((peakCapital - currentEquity) / peakCapital) * 100 if peakCapital > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

    winRate = (fullWins / tradeCount * 100) if tradeCount > 0 else 0
    avgWin = totalProfit / fullWins if fullWins > 0 else 0
    lossCount = tradeCount - fullWins
    avgLoss = totalLoss / lossCount if lossCount > 0 else 0
    payoffRatio = (avgWin / avgLoss) if avgLoss > 0 else (99.9 if avgWin > 0 else 0)
    expectedValue = (winRate / 100 * payoffRatio) - (1 - winRate / 100)
    totalNetProfitPct = ((currentCapital - params.initial_capital) / params.initial_capital) * 100
    score = totalNetProfitPct / tradeCount if tradeCount > 0 else 0
    
    isSetup_today = buyCondition[-1] and (positionSize == 0)
    buyLimit_today = adjust_to_tick(C[-1] + ATR_main[-1] * params.atr_buy_tol) if isSetup_today else np.nan
    stopLoss_today = adjust_to_tick(buyLimit_today - ATR_main[-1] * params.atr_times_init) if isSetup_today else np.nan
    isCandidate = (tradeCount >= 5) and (winRate >= 50) and (expectedValue > 0)

    active_stop_today = max(initialStopLossPrice, trailingStopPrice) if positionSize > 0 else np.nan
    is_in_buy_zone = (positionSize > 0) and (not soldHalf) and (C[-1] > active_stop_today) and (C[-1] < sellPriceHalf)

    return {
        "asset_growth": totalNetProfitPct, "trade_count": tradeCount, "missed_buys": missedBuyCount,
        "score": score, "win_rate": winRate, "payoff_ratio": payoffRatio, "expected_value": expectedValue,
        "max_drawdown": maxDrawdownPct, "is_candidate": isCandidate, "is_setup_today": isSetup_today,
        "buy_limit": buyLimit_today, "stop_loss": stopLoss_today, "is_in_buy_zone": is_in_buy_zone,
        "active_stop": active_stop_today, "target_half": sellPriceHalf, "current_position": positionSize
    }