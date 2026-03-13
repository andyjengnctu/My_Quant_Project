import sys
import os
import json
import pandas as pd
import numpy as np
import math
import warnings
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.v16_config import V16StrategyParams
from core.v16_core import (
    generate_signals,
    execute_bar_step,
    adjust_long_buy_limit,
    adjust_long_stop_price,
    adjust_long_target_price,
    adjust_long_buy_fill_price,
    adjust_long_sell_fill_price,
    calc_entry_price,
    calc_net_sell_price,
    calc_position_size,
    calc_initial_risk_total,
    evaluate_chase_condition
)
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows
from core.v16_log_utils import format_exception_summary

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)

C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RED = '\033[91m'
C_RESET = '\033[0m'

DATA_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def load_params(json_file=os.path.join(BASE_DIR, "models", "v16_best_params.json")):
    params = V16StrategyParams()
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"找不到參數檔: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(params, k):
                setattr(params, k, v)
        print(f"{C_GREEN}✅ 成功載入參數大腦: {json_file}{C_RESET}")
    except Exception as e:
        raise RuntimeError(f"載入 {json_file} 失敗: {format_exception_summary(e)}") from e
    return params

def run_debug_backtest(df, ticker, params, export_excel=True, verbose=True):
    """以正式核心邏輯為準，輸出可讀交易明細的除錯工具"""
    H, L, C = df['High'].values, df['Low'].values, df['Close'].values
    O = df['Open'].values
    Dates = df.index

    ATR_main, buyCondition, sellCondition, _ = generate_signals(df, params)

    position = {'qty': 0}
    pending_chase = None
    currentCapital = params.initial_capital
    currentEquity = currentCapital
    trade_logs = []

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j - 1]):
            continue

        pos_start_of_current_bar = position['qty']

        # 1. 執行 T 日持倉結算（與正式核心共用 execute_bar_step）
        if pos_start_of_current_bar > 0:
            prev_qty = position['qty']
            prev_realized = position.get('realized_pnl', 0.0)
            prev_tp_half = position.get('tp_half', np.nan)

            position, freed_cash, pnl_realized, events = execute_bar_step(
                position,
                ATR_main[j - 1],
                sellCondition[j - 1],
                C[j - 1],
                O[j],
                H[j],
                L[j],
                C[j],
                params
            )

            realized_delta = position.get('realized_pnl', 0.0) - prev_realized
            active_stop_after_update = position.get('sl', np.nan)

            if 'TP_HALF' in events and realized_delta != 0:
                sold_qty = prev_qty - position['qty']
                exec_sell_price_half = adjust_long_sell_fill_price(max(prev_tp_half, O[j]))
                sell_net_price_half = calc_net_sell_price(exec_sell_price_half, sold_qty, params)

                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "半倉停利",
                    "成交價": exec_sell_price_half,
                    "含息成本價": sell_net_price_half,
                    "股數": sold_qty,
                    "投入總金額": sell_net_price_half * sold_qty,
                    "設定停損價": active_stop_after_update,
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": realized_delta
                })

            if 'STOP' in events or 'IND_SELL' in events:
                half_qty = prev_qty - position['qty'] if 'TP_HALF' in events else 0
                final_exit_qty = prev_qty - half_qty
                action_str = "停損殺出" if 'STOP' in events else "指標賣出"

                if 'STOP' in events:
                    sell_price = adjust_long_sell_fill_price(min(active_stop_after_update, O[j]))
                else:
                    sell_price = adjust_long_sell_fill_price(O[j])

                sell_net_price = calc_net_sell_price(sell_price, final_exit_qty, params)
                final_leg_pnl = pnl_realized - realized_delta if 'TP_HALF' in events else pnl_realized

                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": action_str,
                    "成交價": sell_price,
                    "含息成本價": sell_net_price,
                    "股數": final_exit_qty,
                    "投入總金額": sell_net_price * final_exit_qty,
                    "設定停損價": active_stop_after_update,
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": final_leg_pnl
                })

            currentCapital += pnl_realized

        # 2. 處理 T 日進場（完全比照正式核心）
        isSetup_prev = buyCondition[j - 1] and (pos_start_of_current_bar == 0)
        is_locked_limit_up = (
            (O[j] == H[j]) and
            (H[j] == L[j]) and
            (L[j] == C[j]) and
            (C[j] > C[j - 1])
        )
        buyTriggered = False

        if isSetup_prev:
            buyLimitPrice = adjust_long_buy_limit(C[j - 1] + ATR_main[j - 1] * params.atr_buy_tol)
            planned_init_sl = adjust_long_stop_price(buyLimitPrice - ATR_main[j - 1] * params.atr_times_init)
            planned_init_trail = adjust_long_stop_price(buyLimitPrice - ATR_main[j - 1] * params.atr_times_trail)
            sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
            buyQty = calc_position_size(buyLimitPrice, planned_init_sl, sizing_cap, params.fixed_risk, params)

            if L[j] <= buyLimitPrice and not is_locked_limit_up and buyQty > 0:
                buyPrice = adjust_long_buy_fill_price(min(O[j], buyLimitPrice))

                # # (AI註: 與正式核心一致 - 若開盤已跌破盤前停損死線，直接放棄，不可硬買)
                if buyPrice > planned_init_sl:
                    entryPrice = calc_entry_price(buyPrice, buyQty, params)
                    net_sl = calc_net_sell_price(planned_init_sl, buyQty, params)
                    tp_half = adjust_long_target_price(buyPrice + (entryPrice - net_sl))
                    init_risk = calc_initial_risk_total(entryPrice, net_sl, buyQty, params)

                    position = {
                        'qty': buyQty,
                        'entry': entryPrice,
                        'sl': max(planned_init_sl, planned_init_trail),
                        'initial_stop': planned_init_sl,
                        'trailing_stop': planned_init_trail,
                        'tp_half': tp_half,
                        'sold_half': False,
                        'pure_buy_price': buyPrice,
                        'realized_pnl': 0.0,
                        'initial_risk_total': init_risk
                    }
                    buyTriggered = True
                    pending_chase = None

                    trade_logs.append({
                        "日期": Dates[j].strftime('%Y-%m-%d'),
                        "動作": "買進",
                        "成交價": buyPrice,
                        "含息成本價": entryPrice,
                        "股數": buyQty,
                        "投入總金額": entryPrice * buyQty,
                        "設定停損價": planned_init_sl,
                        "半倉停利價": tp_half,
                        "ATR(前日)": ATR_main[j - 1],
                        "單筆實質損益": 0.0
                    })
                else:
                    pending_chase = None
            else:
                chase_res = evaluate_chase_condition(
                    C[j],
                    buyLimitPrice,
                    ATR_main[j - 1],
                    sizing_cap,
                    params
                )
                pending_chase = chase_res if chase_res else None

        elif pending_chase is not None and pos_start_of_current_bar == 0:
            chase_limit = pending_chase['chase_price']
            planned_init_sl = pending_chase['sl']
            sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
            buyQty = pending_chase['qty']

            if L[j] <= chase_limit and not is_locked_limit_up and buyQty > 0:
                buyPrice = adjust_long_buy_fill_price(min(O[j], chase_limit))

                if buyPrice > planned_init_sl:
                    entryPrice = calc_entry_price(buyPrice, buyQty, params)
                    net_sl = calc_net_sell_price(planned_init_sl, buyQty, params)
                    init_risk = calc_initial_risk_total(entryPrice, net_sl, buyQty, params)

                    position = {
                        'qty': buyQty,
                        'entry': entryPrice,
                        'sl': planned_init_sl,
                        'initial_stop': planned_init_sl,
                        'trailing_stop': planned_init_sl,
                        'tp_half': pending_chase['tp'],
                        'sold_half': False,
                        'pure_buy_price': buyPrice,
                        'realized_pnl': 0.0,
                        'initial_risk_total': init_risk
                    }
                    buyTriggered = True
                    pending_chase = None

                    trade_logs.append({
                        "日期": Dates[j].strftime('%Y-%m-%d'),
                        "動作": "買進",
                        "成交價": buyPrice,
                        "含息成本價": entryPrice,
                        "股數": buyQty,
                        "投入總金額": entryPrice * buyQty,
                        "設定停損價": planned_init_sl,
                        "半倉停利價": position['tp_half'],
                        "ATR(前日)": ATR_main[j - 1],
                        "單筆實質損益": 0.0
                    })

            # # (AI註: 與正式核心一致 - 未成交時，依原始 limit / ATR 續追或放棄)
            if not buyTriggered:
                if pending_chase['sl'] < C[j] < pending_chase['tp']:
                    chase_res = evaluate_chase_condition(
                        C[j],
                        pending_chase['orig_limit'],
                        pending_chase['orig_atr'],
                        sizing_cap,
                        params
                    )
                    pending_chase = chase_res if chase_res else None
                else:
                    pending_chase = None

        currentEquity = currentCapital
        if position['qty'] > 0:
            floatingSellNet = calc_net_sell_price(C[j], position['qty'], params)
            floatingPnL = (floatingSellNet - position['entry']) * position['qty']
            currentEquity = currentCapital + floatingPnL

    if not trade_logs:
        if verbose:
            print(f"{C_YELLOW}⚠️ 這檔股票沒有任何交易紀錄。{C_RESET}")
        return None

    df_logs = pd.DataFrame(trade_logs)
    df_logs['單筆實質損益'] = df_logs['單筆實質損益'].round(2)
    df_logs['投入總金額'] = df_logs['投入總金額'].round(0)

    if export_excel:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_filename = os.path.join(OUTPUT_DIR, f"Debug_TradeLog_{ticker}.xlsx")
        df_logs.to_excel(output_filename, index=False)
        if verbose:
            print(f"{C_GREEN}📁 交易明細已成功匯出至：{output_filename}{C_RESET}")

    if verbose:
        losses = df_logs[df_logs['單筆實質損益'] < 0]
        if not losses.empty:
            print(f"\n{C_CYAN}🚨 [抓漏分析] 前 3 大嚴重虧損明細：{C_RESET}")
            worst_losses = losses.sort_values(by='單筆實質損益', ascending=True).head(3)
            for _, row in worst_losses.iterrows():
                print(
                    f"日期: {row['日期']} | 動作: {row['動作']:<4} | 股價: {row['成交價']:>6.2f} | "
                    f"股數: {int(row['股數']):>6}股 | 總投入金: {row['投入總金額']:>9,.0f} | "
                    f"💸 虧損: {row['單筆實質損益']:>9,.0f}"
                )
                print(
                    f"   ➤ 當下 ATR 為 {row['ATR(前日)']:.2f}，"
                    f"停損/賣出參考價為 {row['設定停損價']:.2f}。"
                )

    return df_logs

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
    raw_df = pd.read_csv(file_path)

    params = load_params()
    min_rows_needed = get_required_min_rows(params)
    df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

    dropped_row_count = sanitize_stats['dropped_row_count']
    invalid_row_count = sanitize_stats['invalid_row_count']
    duplicate_date_count = sanitize_stats['duplicate_date_count']

    if dropped_row_count > 0:
        print(
            f"{C_YELLOW}⚠️ {ticker} 清洗移除 {dropped_row_count} 列 "
            f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count}){C_RESET}"
        )

    print("⏳ 正在產生完整交易明細...")
    run_debug_backtest(df, ticker, params)
    
if __name__ == "__main__":
    main()