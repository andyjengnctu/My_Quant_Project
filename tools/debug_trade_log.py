import os
import json
import pandas as pd
import numpy as np
import math
import warnings
from datetime import datetime

# 引入您原本的設定與輔助數學函數 (不改動核心)
from v16_config import V16StrategyParams
from v16_core import (
    tv_atr, tv_ema, tv_supertrend, adjust_to_tick,
    calc_entry_price, calc_net_sell_price, calc_position_size
)

warnings.filterwarnings('ignore')

C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RED = '\031[91m'
C_RESET = '\033[0m'

DATA_DIR = "tw_stock_data_vip"

def load_params(json_file="v16_best_params.json"):
    params = V16StrategyParams()
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(params, k):
                    setattr(params, k, v)
            print(f"{C_GREEN}✅ 成功載入參數大腦: {json_file}{C_RESET}")
        except Exception:
            pass
    return params

def run_debug_backtest(df, ticker, params):
    """擁有 100% 核心邏輯，但加入了詳細明細追蹤的除錯引擎"""
    H, L, C = df['High'].values, df['Low'].values, df['Close'].values
    O, V = df['Open'].values, df['Volume'].values
    Dates = df.index # 抓取日期供除錯

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
    else:
        bbCondition = np.ones_like(C, dtype=bool) 

    if getattr(params, 'use_vol', True):
        VolS = pd.Series(V).rolling(params.vol_short_len).mean().values
        VolL = pd.Series(V).rolling(params.vol_long_len).mean().values
        volCondition = np.isnan(V) | (VolS > VolL)
    else:
        volCondition = np.ones_like(C, dtype=bool)

    if getattr(params, 'use_kc', True):
        ATR_kc = tv_atr(H, L, C, params.kc_len)
        KC_Mid = tv_ema(C, params.kc_len)
        KC_Lower = KC_Mid - ATR_kc * params.kc_mult
        isKcCrossunder = (C < KC_Lower) & (np.roll(C, 1) >= np.roll(KC_Lower, 1))
        isKcCrossunder[0] = False
        kcSellCondition = (isKcCrossunder & (C < O))
    else:
        kcSellCondition = np.zeros_like(C, dtype=bool) 

    buyCondition = (C > O) & isPriceCrossover & bbCondition & volCondition
    sellCondition = (isSupertrend_Bearish_Flip & (C <= O)) | kcSellCondition

    positionSize = 0
    buyPrice, entryPrice = np.nan, np.nan
    initialStopLossPrice, trailingStopPrice = np.nan, np.nan
    sellPriceHalf = np.nan
    soldHalf = False
    cumulativeProfit = 0.0
    currentCapital = params.initial_capital

    # 🌟 交易明細清單
    trade_logs = []

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j-1]): continue
        pos_start_of_current_bar = positionSize
            
        if positionSize > 0 and C[j] > buyPrice + (ATR_main[j] * params.atr_times_trail):
            new_trail = C[j] - (ATR_main[j] * params.atr_times_trail)
            trailingStopPrice = adjust_to_tick(max(trailingStopPrice, new_trail))
            
        isSetup_prev = buyCondition[j-1] and (pos_start_of_current_bar == 0)
        buyLimitPrice = adjust_to_tick(C[j-1] + ATR_main[j-1] * params.atr_buy_tol) if isSetup_prev else np.nan
        buyTriggered = isSetup_prev and L[j] <= buyLimitPrice
            
        if buyTriggered:
            buyPrice = adjust_to_tick(min(O[j], buyLimitPrice))
            initialStopLossPrice = adjust_to_tick(buyPrice - ATR_main[j-1] * params.atr_times_init)
            trailingStopPrice = adjust_to_tick(buyPrice - ATR_main[j-1] * params.atr_times_trail)
            soldHalf, cumulativeProfit = False, 0.0
            
            sizing_capital = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
            buyQty = calc_position_size(buyPrice, initialStopLossPrice, sizing_capital, params.fixed_risk, params)
            
            if buyQty > 0:
                entryPrice = calc_entry_price(buyPrice, buyQty, params)
                sellPriceHalf = adjust_to_tick(buyPrice + (entryPrice - calc_net_sell_price(initialStopLossPrice, buyQty, params)))
                positionSize = buyQty
                
                # 📝 寫入買進明細
                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "買進",
                    "成交價": buyPrice,
                    "含息成本價": entryPrice,
                    "股數": buyQty,
                    "投入總金額": entryPrice * buyQty,
                    "設定停損價": initialStopLossPrice,
                    "半倉停利價": sellPriceHalf,
                    "ATR(前日)": ATR_main[j-1],
                    "單筆實質損益": 0.0
                })

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
                    pnl_half = (sellNetPriceHalf - entryPrice) * sellQtyHalf
                    cumulativeProfit += pnl_half
                    positionSize -= sellQtyHalf
                    soldHalf = True
                    
                    # 📝 寫入半倉停利明細
                    trade_logs.append({
                        "日期": Dates[j].strftime('%Y-%m-%d'),
                        "動作": "半倉停利",
                        "成交價": execSellPriceHalf,
                        "含息成本價": sellNetPriceHalf,
                        "股數": sellQtyHalf,
                        "投入總金額": sellNetPriceHalf * sellQtyHalf,
                        "設定停損價": activeStopPrice,
                        "半倉停利價": np.nan,
                        "ATR(前日)": ATR_main[j-1],
                        "單筆實質損益": pnl_half
                    })
                else:
                    isTakeProfitHit = False
                
            if isStopHit or isIndicatorSell:
                sellPrice = adjust_to_tick(min(activeStopPrice, O[j]) if isStopHit else O[j])
                sellQty = positionSize
                sellNetPrice = calc_net_sell_price(sellPrice, sellQty, params)
                pnl_final = (sellNetPrice - entryPrice) * sellQty
                profitValue = cumulativeProfit + pnl_final
                
                action_str = "停損殺出" if isStopHit else "指標賣出"
                
                # 📝 寫入清倉明細
                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": action_str,
                    "成交價": sellPrice,
                    "含息成本價": sellNetPrice,
                    "股數": sellQty,
                    "投入總金額": sellNetPrice * sellQty,
                    "設定停損價": activeStopPrice,
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j-1],
                    "單筆實質損益": pnl_final
                })
                
                currentCapital += profitValue
                positionSize = 0
                soldHalf = False

    if not trade_logs:
        print(f"{C_YELLOW}⚠️ 這檔股票沒有任何交易紀錄。{C_RESET}")
        return None

    # 輸出成 Excel
    df_logs = pd.DataFrame(trade_logs)
    
    # 增加一個輔助欄位，把同一筆交易串起來看 (根據 PnL 是否大於 0 來累加)
    df_logs['單筆實質損益'] = df_logs['單筆實質損益'].round(2)
    df_logs['投入總金額'] = df_logs['投入總金額'].round(0)
    
    output_filename = f"Debug_TradeLog_{ticker}.xlsx"
    df_logs.to_excel(output_filename, index=False)
    print(f"{C_GREEN}📁 交易明細已成功匯出至：{output_filename}{C_RESET}")
    
    # 針對除錯，特別印出前五大虧損
    losses = df_logs[df_logs['單筆實質損益'] < 0]
    if not losses.empty:
        print(f"\n{C_CYAN}🚨 [抓漏分析] 前 3 大嚴重虧損明細：{C_RESET}")
        worst_losses = losses.sort_values(by='單筆實質損益', ascending=True).head(3)
        for _, row in worst_losses.iterrows():
            print(f"日期: {row['日期']} | 動作: {row['動作']:<4} | 股價: {row['成交價']:>6.2f} | 股數: {row['股數']:>6}股 | 總投入金: {row['投入總金額']:>9,.0f} | 💸 虧損: {row['單筆實質損益']:>9,.0f}")
            print(f"   ➤ 當下 ATR 僅有 {row['ATR(前日)']:.2f}，導致您在 {row['設定停損價']:.2f} 的停損防線被跳空/滑價擊穿。")

def main():
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"🛠️ {C_YELLOW}V16 放大鏡：單檔股票交易明細除錯工具{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    ticker = input("\n👉 請輸入要除錯的股票代號 (例如: 00972): ").strip()
    if not ticker: return
    
    # 支援 TV_Data_Full_ 前綴或是直接數字
    file_path = os.path.join(DATA_DIR, f"TV_Data_Full_{ticker}.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        
    if not os.path.exists(file_path):
        # 允許使用者手動上傳的檔案
        if os.path.exists(f"{ticker}.csv"):
            file_path = f"{ticker}.csv"
        else:
            print(f"❌ 找不到 {ticker} 的歷史資料 CSV。")
            return
            
    print(f"📥 讀取 {file_path}...")
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() for c in df.columns]
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
    params = load_params()
    
    print("⏳ 正在產生完整交易明細...")
    run_debug_backtest(df, ticker, params)

if __name__ == "__main__":
    main()