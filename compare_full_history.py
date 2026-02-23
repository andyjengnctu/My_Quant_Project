import pandas as pd
import os
import json #test new computer 
import warnings

from v16_config import V16StrategyParams
from v16_core import run_v16_backtest

warnings.filterwarnings('ignore')

C_RED = '\033[91m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'

def load_params_from_json(json_file):
    params = V16StrategyParams()
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        # 動態將 JSON 的值覆寫到 params 物件
        for key, value in data.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params, True
    return params, False

def compare_with_tv(csv_file_path, params, param_source):
    print(f"\n🔍 正在讀取資料: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        print(f"❌ 找不到檔案！請確認 {csv_file_path} 是否存在。")
        return
        
    df = pd.read_csv(csv_file_path)
    df.columns = [c.capitalize() for c in df.columns]
    
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
    stats = run_v16_backtest(df, params)
    
    if not stats:
        print("❌ 運算失敗：無法產生指標訊號。")
        return

    print(f"\n{C_CYAN}[Python 核心運算結果 vs TradingView 面板]{C_RESET}")
    print(f"⚙️ 載入參數來源: {param_source}")
    print(f"   (新高:{params.high_len}, ATR:{params.atr_len}, 追買:{params.atr_buy_tol}倍, "
          f"停損:{params.atr_times_init}倍, 追蹤:{params.atr_times_trail}倍, 半倉:{params.tp_percent})")
    print("┌───────────────────────────┐")
    print(f"│ 資產成長: {stats['asset_growth']:>6.2f}%       │")
    print(f"│   交易次數: {stats['trade_count']:>3}           │")
    print(f"│   錯失買點: {stats['missed_buys']:>3}次         │")
    print(f"│   單筆報酬: {stats['score']:>6.2f}%       │")
    print("│                           │")
    print(f"│     勝率: {stats['win_rate']:>6.2f}%         │")
    print(f"│   風報比: {stats['payoff_ratio']:>6.2f}          │")
    print(f"│   期望值: {stats['expected_value']:>6.2f} R        │")
    print(f"│ 最大回撤: {stats['max_drawdown']:>6.2f}%       │")
    print("└───────────────────────────┘")
    print(f"{C_GREEN}對比完成！請打開 TradingView，確保套用【相同參數】進行對照。{C_RESET}\n")

if __name__ == "__main__":
    print(f"{C_CYAN}🚀 v16 模組化獨立核心驗證工具啟動...{C_RESET}")
    print("請選擇要掛載的參數大腦：")
    print("1. v16_original_params.json (原始手動參數)")
    print("2. v16_best_params.json (AI 訓練最佳參數)")
    print("3. 使用 v16_config.py 系統預設值")
    
    choice = input("請輸入 1, 2, 或 3 (預設為 1): ").strip()
    
    if choice == '2':
        params, success = load_params_from_json("v16_best_params.json")
        param_source = "v16_best_params.json" if success else "找不到檔案，使用預設值"
    elif choice == '3':
        params = V16StrategyParams()
        param_source = "v16_config.py 系統預設"
    else:
        params, success = load_params_from_json("v16_original_params.json")
        param_source = "v16_original_params.json" if success else "找不到檔案，使用預設值"

    while True:
        ticker_input = input(f"請輸入要測試的股票代碼 (例如 2330，輸入 'q' 離開): ").strip()
        
        if ticker_input.lower() == 'q': break
        if not ticker_input: continue
            
        # 🌟 直接強制指定去 testing_csv 找檔案
        # 假設您從 TV 下載的檔名是 TV_Data_Full_2330.csv
        target_csv_path = f"testing_csv/TV_Data_Full_{ticker_input}.csv"
        
        # 如果您的檔名沒有 TV_Data_Full_ 前綴，只有 2330.csv，請把上一行改成下面這行：
        # target_csv_path = f"testing_csv/{ticker_input}.csv"
             
        compare_with_tv(target_csv_path, params, param_source)