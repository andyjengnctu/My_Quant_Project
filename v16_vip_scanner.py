import os
import json
import pandas as pd
import numpy as np 
import warnings
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from core.v16_config import V16StrategyParams, BUY_SORT_METHOD
from core.v16_core import run_v16_backtest, calc_position_size, calc_entry_price, adjust_to_tick, calc_net_sell_price
from core.v16_display import print_scanner_header, C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET

# # (AI註: 收窄 warning 範圍；預設保留 warning，可疑資料與數值問題不要被全域吃掉)
warnings.simplefilter("default")

# # (AI註: 相同 RuntimeWarning 只顯示一次；保留可見性，但避免掃描輸出被重複洗版)
warnings.filterwarnings("once", category=RuntimeWarning)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

LOAD_DATA_REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
SCANNER_PROGRESS_EVERY = 1


def sanitize_ohlcv_dataframe(df, ticker):
    working = df.copy()
    working.columns = [c.capitalize() for c in working.columns]

    date_col = 'Time' if 'Time' in working.columns else 'Date'
    if date_col not in working.columns:
        raise KeyError("缺少 Time / Date 欄位")

    missing_cols = [c for c in LOAD_DATA_REQUIRED_COLS if c not in working.columns]
    if missing_cols:
        raise KeyError(f"缺少必要欄位: {missing_cols}")

    for col in LOAD_DATA_REQUIRED_COLS:
        working[col] = pd.to_numeric(working[col], errors='coerce')

    working[date_col] = pd.to_datetime(working[date_col], errors='coerce')

    invalid_mask = (
        working[date_col].isna() |
        working['Open'].isna() |
        working['High'].isna() |
        working['Low'].isna() |
        working['Close'].isna() |
        working['Volume'].isna() |
        (working['Open'] <= 0) |
        (working['High'] <= 0) |
        (working['Low'] <= 0) |
        (working['Close'] <= 0) |
        (working['Volume'] < 0)
    )

    bad_count = int(invalid_mask.sum())
    if bad_count > 0:
        working = working.loc[~invalid_mask].copy()

    if working.empty:
        raise ValueError(f"{ticker} 清洗後無有效資料")

    working.set_index(date_col, inplace=True)
    working.sort_index(inplace=True)

    if bad_count > 0:
        print(f"{C_YELLOW}[警告] {ticker} 剔除 {bad_count} 筆無效 OHLCV 資料{C_RESET}")

    return working

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
        df = sanitize_ohlcv_dataframe(df, ticker)
        if len(df) < params.high_len + 10:
            return None

        stats = run_v16_backtest(df, params)
        
        if not stats or not stats['is_candidate']: return None
            
        stat_str = f"勝率:{stats['win_rate']:>5.1f}% | 期望值:{stats['expected_value']:>5.2f}R | 交易:{stats['trade_count']:>3}次 | MDD:{stats['max_drawdown']:>5.1f}%"
        reference_capital = params.initial_capital
        
        if stats['is_setup_today']:
            proj_qty = calc_position_size(stats['buy_limit'], stats['stop_loss'], reference_capital, params.fixed_risk, params)
            if proj_qty == 0: return ('candidate', None, None, None, ticker)
                
            proj_cost = calc_entry_price(stats['buy_limit'], proj_qty, params) * proj_qty
            
            actual_cost_per_share = calc_entry_price(stats['buy_limit'], proj_qty, params)
            net_sl_per_share = calc_net_sell_price(stats['stop_loss'], proj_qty, params)
            est_target = adjust_to_tick(stats['buy_limit'] + (actual_cost_per_share - net_sl_per_share))
            
            buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 停利(預估):{est_target:>6.2f} | 參考投入:{proj_cost:>7,.0f}"
            msg = f"[🚨 最新買訊] {ticker:<6} | {stat_str} | {buy_str}"
            
            return ('buy', proj_cost, stats['expected_value'], msg, ticker)
            
        elif stats.get('chase_today') is not None:
            chase = stats['chase_today']
            proj_qty = chase['qty'] 
            if proj_qty == 0: return ('candidate', None, None, None, ticker)
                
            proj_cost = calc_entry_price(chase['chase_price'], proj_qty, params) * proj_qty
            
            zone_str = f"追買限價:{chase['chase_price']:>6.2f} | 停損:{chase['sl']:>6.2f} | 盈虧比:{chase['rr']:>4.2f} | 參考投入:{proj_cost:>7,.0f}"
            msg = f"[⚠️ 遲到補車 (精準1R縮倉)] {ticker:<5} | {stat_str} | {zone_str}"
            
            return ('zone', proj_cost, stats['expected_value'], msg, ticker)
            
        return ('candidate', None, None, None, ticker)
        
    except Exception as e:
        return ('error', None, None, f"⚠️ 股票 {ticker} 資料處理異常: {e}", ticker)

def run_daily_scanner(data_dir):
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"{C_CYAN}🚀 啟動【v16 尊爵版】極速平行掃描儀 | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    if not os.path.exists(data_dir):
        print(f"❌ 找不到資料夾 {data_dir}。")
        return

    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    total_files = len(csv_files)
    if total_files == 0: return

    params, is_loaded = load_dynamic_params("models/v16_best_params.json")
    if is_loaded: print(f"{C_GREEN}✅ 成功載入 AI 聖杯參數大腦！{C_RESET}")
    else: print(f"{C_YELLOW}⚠️ 找不到最佳參數，使用系統內建預設值。{C_RESET}")
        
    print_scanner_header(params)
    print(f"{C_YELLOW}ℹ️ 本掃描器的投入金額僅以 initial_capital 作為參考估算，非帳戶級真實可下單金額。{C_RESET}")
    print(f"{C_CYAN}--------------------------------------------------------------------------------{C_RESET}")

    count_scanned, count_candidates = 0, 0
    buy_list, in_zone_list = [], []
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_stock, os.path.join(data_dir, f), f.replace('.csv', '').replace('TV_Data_Full_', ''), params): f for f in csv_files}

        for future in as_completed(futures):
            count_scanned += 1
            if count_scanned % SCANNER_PROGRESS_EVERY == 0:
                print(f"{C_GRAY}⏳ 極速運算中: [{count_scanned}/{total_files}]{C_RESET}", end="\r", flush=True)
            
            result = future.result()
            if result and len(result) == 5:
                status, proj_cost, ev, msg, ticker = result
                if status in ['buy', 'zone', 'candidate']: count_candidates += 1
                    
                if status == 'buy':
                    print(" " * 120, end="\r")
                    print(f"{C_RED}{msg}{C_RESET}")
                    buy_list.append({'proj_cost': proj_cost, 'ev': ev, 'text': msg, 'ticker': ticker})
                elif status == 'zone':
                    print(" " * 120, end="\r")
                    print(f"{C_YELLOW}{msg}{C_RESET}")
                    in_zone_list.append({'proj_cost': proj_cost, 'ev': ev, 'text': msg, 'ticker': ticker})
                # # (AI註: 修復低嚴重度問題 - 將異常檔案造成的錯誤印出，避免安靜地被吞掉)
                elif status == 'error':
                    print(" " * 120, end="\r")
                    print(f"{C_YELLOW}{msg}{C_RESET}")

    elapsed_time = time.time() - start_time
    
    if BUY_SORT_METHOD == 'EV':
        buy_list.sort(key=lambda x: (x['ev'], x['ticker']), reverse=True)
        in_zone_list.sort(key=lambda x: (x['ev'], x['ticker']), reverse=True)
        sort_title = "按期望值 (EV) 由大到小排序"
    else:
        buy_list.sort(key=lambda x: (x['proj_cost'], x['ticker']), reverse=True)
        in_zone_list.sort(key=lambda x: (x['proj_cost'], x['ticker']), reverse=True)
        sort_title = "按預估投入資金由大到小排序"

    print(" " * 120, end="\r") 
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚡ 掃描完畢！共掃描 {count_scanned} 檔標的，耗時 {elapsed_time:.2f} 秒。歷史及格: {count_candidates} 檔")
    
    if buy_list or in_zone_list:
        print(f"\n{C_RED}🔥 【第一優先：明日掛單清單 (最新買訊)】 {sort_title} 🔥{C_RESET}")
        if buy_list:
            for item in buy_list: print(f"   {C_RED}➤ {item['text']}{C_RESET}")
        else: print(f"   {C_RED}無最新買訊。{C_RESET}")

        print(f"\n{C_YELLOW}⚠️ 【第二優先：遲到安全追車清單 (已精準還原1R且縮倉)】 {sort_title} ⚠️{C_RESET}")
        if in_zone_list:
            for item in in_zone_list: print(f"   {C_YELLOW}➤ {item['text']}{C_RESET}")
        else: print(f"   {C_YELLOW}無符合安全追買條件的標的。{C_RESET}")
    else:
        print(f"\n{C_GREEN}💤 今日無符合實戰買點的標的，保留現金，明日再戰！{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")

if __name__ == "__main__":
    run_daily_scanner("tw_stock_data_vip")