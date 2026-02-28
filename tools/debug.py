import pandas as pd
import numpy as np
import pandas_ta as ta
import math
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 輔助函數 (維持 v16 嚴格跳動點與成本)
# ==========================================
def tv_round(number):
    return math.floor(number + 0.5)

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

def calc_entry_price(bPrice, bQty):
    fee = max(bPrice * bQty * 0.000855, 20)
    return bPrice + (fee / bQty)

def calc_net_sell_price(sPrice, sQty):
    fee = max(sPrice * sQty * 0.000855, 20)
    tax = sPrice * sQty * 0.003
    return sPrice - ((fee + tax) / sQty)

def calc_position_size(bPrice, stopPrice, cap, riskPct):
    maxQty = cap / (bPrice * (1 + 0.000855))
    estEntryCost = bPrice * (1 + 0.000855)
    estExitNet = stopPrice * (1 - 0.000855 - 0.003)
    riskPerUnit = estEntryCost - estExitNet
    if riskPerUnit > 0:
        return int(math.floor(min(cap * riskPct / riskPerUnit, maxQty)))
    return 0

# ==========================================
# 1. 核心驗證引擎
# ==========================================
def run_tv_comparison(csv_file_path):
    print(f"\n🔍 正在讀取 TV 歷史資料: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        print(f"❌ 找不到檔案！請確認 {csv_file_path} 是否存在。")
        return
        
    # 讀取 TV 下載的 CSV
    df = pd.read_csv(csv_file_path)
    
    # 統一把欄位轉成首字母大寫
    df.columns = [c.capitalize() for c in df.columns]
    
    # 如果 TV 輸出的時間欄位叫 'Time'，我們將其設為 Index
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)

    # --- 計算指標 ---
    df['ATR12'] = ta.atr(df['High'], df['Low'], df['Close'], length=12, mamode='rma')
    df['ATR20'] = ta.atr(df['High'], df['Low'], df['Close'], length=20, mamode='rma')
    df['High50'] = df['High'].shift(1).rolling(50).max()
    df['KC_Mid'] = ta.ema(df['Close'], length=20)
    df['KC_Lower'] = df['KC_Mid'] - df['ATR20'] * 2.0
    
    df.ta.supertrend(length=12, multiplier=4.0, append=True)
    st_cols = [c for c in df.columns if "SUPERTd" in c]
    st_dir_col = st_cols[0] 
    
    df['isSupertrend_Bearish_Flip'] = (df[st_dir_col] < df[st_dir_col].shift(1))
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(20).std(ddof=0)
    df['VolS'] = df['Volume'].rolling(5).mean()
    df['VolL'] = df['Volume'].rolling(19).mean()
    
    # --- 買賣條件 ---
    df['isPriceCrossover'] = (df['Close'] > df['High50']) & (df['Close'].shift(1) <= df['High50'].shift(1))
    df['isKcCrossunder'] = (df['Close'] < df['KC_Lower']) & (df['Close'].shift(1) >= df['KC_Lower'].shift(1))
    
    df['buyCondition'] = (df['Close'] > df['Open']) & (df['Close'] > df['BB_Upper']) & \
                         df['isPriceCrossover'] & (df['VolS'] > df['VolL'])
    df['sellCondition'] = (df['isSupertrend_Bearish_Flip'] & (df['Close'] <= df['Open'])) | \
                          (df['isKcCrossunder'] & (df['Close'] < df['Open']))

    # --- 狀態變數初始化 (對齊 TV) ---
    positionSize = 0
    buyPrice, entryPrice = np.nan, np.nan
    initialStopLossPrice, trailingStopPrice = np.nan, np.nan
    sellPriceHalf = np.nan
    soldHalf = False
    cumulativeProfit = 0.0

    i_capital = 4000000.0
    currentCapital = i_capital  
    dynamicMaxLoss = 0.01       
    i_tpPercent = 0.55          

    # TV 面板統計所需變數
    tradeCount, fullWins = 0, 0
    totalProfit, totalLoss = 0.0, 0.0
    missedBuyCount = 0
    peakCapital = i_capital
    maxDrawdownPct = 0.0
    
    prev_positionSize_start = 0 

    # --- 逐K線撮合回測引擎 ---
    for j in range(1, len(df)):
        curr = df.iloc[j]
        prev = df.iloc[j-1]
        
        if pd.isna(prev['ATR12']) or pd.isna(prev['BB_Upper']): 
            continue
        
        pos_start_of_current_bar = positionSize
            
        # 6. 買入執行與移動停損更新
        if positionSize > 0 and curr['Close'] > buyPrice + (curr['ATR12'] * 4.0):
            new_trail = curr['Close'] - (curr['ATR12'] * 4.0)
            trailingStopPrice = adjust_to_tick(max(trailingStopPrice, new_trail))
            
        isSetup_prev = prev['buyCondition'] and (prev_positionSize_start == 0)
        buyLimitPrice = adjust_to_tick(prev['Close'] + prev['ATR12'] * 0.5) if isSetup_prev else np.nan
        buyTriggered = isSetup_prev and curr['Low'] <= buyLimitPrice
        
        # 錯失買點紀錄 (TV 邏輯：有 Setup 但沒 Trigger)
        if isSetup_prev and not buyTriggered:
            missedBuyCount += 1
            
        if buyTriggered:
            buyPrice = adjust_to_tick(min(curr['Open'], buyLimitPrice))
            initialStopLossPrice = adjust_to_tick(buyPrice - prev['ATR12'] * 2.5)
            trailingStopPrice = adjust_to_tick(buyPrice - prev['ATR12'] * 4.0)
            
            soldHalf = False
            cumulativeProfit = 0.0
            buyQty = calc_position_size(buyPrice, initialStopLossPrice, currentCapital, dynamicMaxLoss)
            
            if buyQty > 0:
                entryPrice = calc_entry_price(buyPrice, buyQty)
                sellPriceHalf = adjust_to_tick(buyPrice + (entryPrice - calc_net_sell_price(initialStopLossPrice, buyQty)))
                positionSize = buyQty

        # 7. 盤中賣出與停利邏輯整合
        isHoldingFromYesterday = (pos_start_of_current_bar > 0) and (not buyTriggered)
        
        if isHoldingFromYesterday:
            activeStopPrice = max(initialStopLossPrice, trailingStopPrice)
            isStopHit = curr['Low'] <= activeStopPrice
            isTakeProfitHit = curr['High'] >= sellPriceHalf and not soldHalf
            isIndicatorSell = prev['sellCondition']
            
            if isStopHit and isTakeProfitHit: 
                isTakeProfitHit = False
                
            if isTakeProfitHit:
                execSellPriceHalf = adjust_to_tick(max(sellPriceHalf, curr['Open']))
                sellQtyHalf = int(math.floor(positionSize * i_tpPercent))
                if sellQtyHalf > 0 and positionSize > sellQtyHalf:
                    sellNetPriceHalf = calc_net_sell_price(execSellPriceHalf, sellQtyHalf)
                    cumulativeProfit += (sellNetPriceHalf - entryPrice) * sellQtyHalf
                    positionSize -= sellQtyHalf
                    soldHalf = True
                else:
                    isTakeProfitHit = False
                
            if isStopHit or isIndicatorSell:
                sellPrice = adjust_to_tick(min(activeStopPrice, curr['Open']) if isStopHit else curr['Open'])
                sellQty = positionSize
                sellNetPrice = calc_net_sell_price(sellPrice, sellQty)
                profitValue = cumulativeProfit + (sellNetPrice - entryPrice) * sellQty
                
                if profitValue > 0:
                    fullWins += 1
                    totalProfit += profitValue
                    dynamicMaxLoss = min(dynamicMaxLoss * 1.05, 0.02)
                else:
                    totalLoss += abs(profitValue)
                    dynamicMaxLoss = max(dynamicMaxLoss * 0.9, 0.005)
                    
                currentCapital += profitValue
                positionSize = 0
                soldHalf = False
                tradeCount += 1 
                
        # 8. 浮動權益與最大回撤 (MDD) - 完全對齊 TV 的計算方式
        currentEquity = currentCapital
        if positionSize > 0:
            floatingSellNet = calc_net_sell_price(curr['Close'], positionSize)
            floatingPnL = cumulativeProfit + (floatingSellNet - entryPrice) * positionSize
            currentEquity = currentCapital + floatingPnL

        peakCapital = max(peakCapital, currentEquity)
        currentDrawdownPct = ((peakCapital - currentEquity) / peakCapital) * 100 if peakCapital > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)
        
        prev_positionSize_start = pos_start_of_current_bar

    # --- 計算最終績效輸出 ---
    winRate = (fullWins / tradeCount * 100) if tradeCount > 0 else 0
    avgWin = totalProfit / fullWins if fullWins > 0 else 0
    lossCount = tradeCount - fullWins
    avgLoss = totalLoss / lossCount if lossCount > 0 else 0
    payoffRatio = (avgWin / avgLoss) if avgLoss > 0 else (99.9 if avgWin > 0 else 0)
    expectedValue = (winRate / 100 * payoffRatio) - (1 - winRate / 100)
    totalNetProfitPct = ((currentCapital - i_capital) / i_capital) * 100
    score = totalNetProfitPct / tradeCount if tradeCount > 0 else 0

    # --- 印出與 TV 相同的面板 ---
    print("\n[Python 核心回測結果與 TV 交叉比對]")
    print("┌───────────────────────────┐")
    print(f"│ 資產成長: {totalNetProfitPct:>6.2f}%       │")
    print(f"│   交易次數: {tradeCount:>3}           │")
    print(f"│   錯失買點: {missedBuyCount:>3}次         │")
    print(f"│   單筆報酬: {score:>6.2f}%       │")
    print("│                           │")
    print(f"│     勝率: {winRate:>6.2f}%         │")
    print(f"│   風報比: {payoffRatio:>6.2f}          │")
    print(f"│   期望值: {expectedValue:>6.2f} R        │")
    print(f"│ 最大回撤: {maxDrawdownPct:>6.2f}%       │")
    print("└───────────────────────────┘")
    print("對比完成！請對照您的 TV 截圖。\n")

# ==========================================
# 2. 啟動互動區
# ==========================================
if __name__ == "__main__":
    while True:
        # 讓使用者輸入代碼
        ticker_input = input("請輸入要測試的股票代碼 (例如 2330，輸入 'q' 離開): ").strip()
        
        if ticker_input.lower() == 'q':
            print("離開測試程式。")
            break
            
        if not ticker_input:
            print("代碼不能為空，請重新輸入。")
            continue
            
        # 自動拼湊檔名
        target_csv_path = f"testing_csv/TV_Data_Full_{ticker_input}.csv"
        
        # 執行比對
        run_tv_comparison(target_csv_path)
        #test-stage2