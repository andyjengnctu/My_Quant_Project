import os
import json
import pandas as pd
import numpy as np 
import warnings
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from core.v16_config import V16StrategyParams, BUY_SORT_METHOD
from core.v16_core import run_v16_backtest, calc_position_size, calc_entry_price
from core.v16_display import print_scanner_header, C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET

warnings.filterwarnings('ignore')

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_dynamic_params(json_file):
    params = V16StrategyParams()
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(params, key): setattr(params, key, value)
            return params, True
        except Exception as e: 
            print(f"{C_YELLOW}⚠️ 讀取參數檔 {json_file} 失敗: {e}{C_RESET}")
    return params, False

def process_single_stock(file_path, ticker, params):
    try:
        df = pd.read_csv(file_path)
        if len(df) < params.high_len + 10: return None
            
        df.columns = [c.capitalize() for c in df.columns]
        
        # (AI註: 修復 5: 執行 OHLC ffill)
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan).ffill()
        
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        latest_close = df['Close'].iloc[-1]
        stats = run_v16_backtest(df, params)
        
        if not stats or not stats['is_candidate']: return None
            
        stat_str = f"勝率:{stats['win_rate']:>5.1f}% | 期望值:{stats['expected_value']:>5.2f}R | 交易:{stats['trade_count']:>3}次 | MDD:{stats['max_drawdown']:>5.1f}%"
        
        # (AI註: 修復 5: 直接使用 params.initial_capital)
        DUMMY_CAPITAL = params.initial_capital
        
        if stats['is_setup_today']:
            proj_qty = calc_position_size(stats['buy_limit'], stats['stop_loss'], DUMMY_CAPITAL, params.fixed_risk, params)
            proj_cost = calc_entry_price(stats['buy_limit'], proj_qty, params) * proj_qty if proj_qty > 0 else 0.0
            
            est_target = stats['buy_limit'] + (stats['buy_limit'] - stats['stop_loss'])
            buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 停利(預估):{est_target:>6.2f} | 投入資金:{proj_cost:>7,.0f}"
            msg = f"[🚨 最新買訊] {ticker:<6} | {stat_str} | {buy_str}"
            
            return ('buy', proj_cost, stats['expected_value'], msg)
            
        elif stats['is_in_buy_zone']:
            proj_qty = calc_position_size(latest_close, stats['active_stop'], DUMMY_CAPITAL, params.fixed_risk, params)
            proj_cost = calc_entry_price(latest_close, proj_qty, params) * proj_qty if proj_qty > 0 else 0.0
            
            zone_str = f"最新收盤:{latest_close:>6.2f} | 停損:{stats['active_stop']:>6.2f} | 停利(半倉):{stats['target_half']:>6.2f} | 投入資金:{proj_cost:>7,.0f}"
            msg = f"[⚠️ 買進區間] {ticker:<6} | {stat_str} | {zone_str}"
            
            return ('zone', proj_cost, stats['expected_value'], msg)
            
        return ('candidate', None, None, None)
        
    except Exception as e:
        return ('error', None, None, f"⚠️ 股票 {ticker} 資料處理異常: {e}")

def run_daily_scanner(data_dir):
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"{C_CYAN}🚀 啟初【v16 尊爵版】極速平行掃描儀 | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    if not os.path.exists(data_dir):
        print(f"❌ 找不到資料夾 {data_dir}。")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    total_files = len(csv_files)
    if total_files == 0: return

    params, is_loaded = load_dynamic_params("models/v16_best_params.json")
    if is_loaded: print(f"{C_GREEN}✅ 成功載入 AI 聖杯參數大腦！{C_RESET}")
    else: print(f"{C_YELLOW}⚠️ 找不到最佳參數，使用系統內建預設值。{C_RESET}")
        
    print_scanner_header(params)
    print(f"{C_CYAN}--------------------------------------------------------------------------------{C_RESET}")

    count_scanned, count_candidates = 0, 0
    buy_list, in_zone_list = [], []
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_stock, os.path.join(data_dir, f), f.replace('.csv', '').replace('TV_Data_Full_', ''), params): f for f in csv_files}

        for future in as_completed(futures):
            count_scanned += 1
            print(f"{C_GRAY}⏳ 極速運算中: [{count_scanned}/{total_files}]{C_RESET}", end="\r", flush=True)
            
            result = future.result()
            if result and len(result) == 4:
                status, proj_cost, ev, msg = result
                # (AI註: 修復 5: 正確計入所有通過歷史門檻的 candidate)
                if status in ['buy', 'zone', 'candidate']:
                    count_candidates += 1
                    
                if status == 'buy':
                    print(" " * 120, end="\r")
                    print(f"{C_RED}{msg}{C_RESET}")
                    buy_list.append({'proj_cost': proj_cost, 'ev': ev, 'text': msg})
                elif status == 'zone':
                    print(" " * 120, end="\r")
                    print(f"{C_YELLOW}{msg}{C_RESET}")
                    in_zone_list.append({'proj_cost': proj_cost, 'ev': ev, 'text': msg})
                elif status == 'error':
                    print(" " * 120, end="\r")
                    print(f"{C_YELLOW}{msg}{C_RESET}")

    elapsed_time = time.time() - start_time
    
    if BUY_SORT_METHOD == 'EV':
        buy_list.sort(key=lambda x: x['ev'], reverse=True)
        in_zone_list.sort(key=lambda x: x['ev'], reverse=True)
        sort_title = "按期望值 (EV) 由大到小排序"
    else:
        buy_list.sort(key=lambda x: x['proj_cost'], reverse=True)
        in_zone_list.sort(key=lambda x: x['proj_cost'], reverse=True)
        sort_title = "按預估投入資金由大到小排序"

    print(" " * 120, end="\r") 
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚡ 掃描完畢！共掃描 {count_scanned} 檔標的，耗時 {elapsed_time:.2f} 秒。歷史及格: {count_candidates} 檔")
    
    if buy_list or in_zone_list:
        print(f"\n{C_RED}🔥 【第一優先：明日掛單清單 (最新買訊)】 {sort_title} 🔥{C_RESET}")
        if buy_list:
            for item in buy_list: print(f"   {C_RED}➤ {item['text']}{C_RESET}")
        else: print(f"   {C_RED}無最新買訊。{C_RESET}")

        print(f"\n{C_YELLOW}⚠️ 【第二優先：仍在買進區間 (錯過可補上車)】 {sort_title} ⚠️{C_RESET}")
        if in_zone_list:
            for item in in_zone_list: print(f"   {C_YELLOW}➤ {item['text']}{C_RESET}")
        else: print(f"   {C_YELLOW}無買進區間標的。{C_RESET}")
    else:
        print(f"\n{C_GREEN}💤 今日無符合實戰買點的標的，保留現金，明日再戰！{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")

if __name__ == "__main__":
    run_daily_scanner("tw_stock_data_vip")