import pandas as pd
import sys
import os
import json
import warnings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
    
from core.v16_config import V16StrategyParams
from core.v16_core import run_v16_backtest
from core.v16_log_utils import format_exception_summary
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows, normalize_ticker_from_csv_filename, resolve_unique_csv_path

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)

C_RED = '\033[91m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'

def load_params_from_json(json_file):
    params = V16StrategyParams()
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"找不到參數檔: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 動態將 JSON 的值覆寫到 params 物件
        for key, value in data.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params, True
    except Exception as e:
        raise RuntimeError(f"讀取參數檔失敗: {format_exception_summary(e)}") from e

def compare_with_tv(csv_file_path, params, param_source):
    print(f"\n🔍 正在讀取資料: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"找不到檔案！請確認 {csv_file_path} 是否存在。")
        
    raw_df = pd.read_csv(csv_file_path)
    min_rows_needed = get_required_min_rows(params)
    ticker = normalize_ticker_from_csv_filename(os.path.basename(csv_file_path))
    df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
        
    stats = run_v16_backtest(df, params)
    
    dropped_row_count = sanitize_stats['dropped_row_count']
    invalid_row_count = sanitize_stats['invalid_row_count']
    duplicate_date_count = sanitize_stats['duplicate_date_count']

    if dropped_row_count > 0:
        print(
            f"⚠️ {ticker} 清洗移除 {dropped_row_count} 列 "
            f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
        )
    
    if not stats:
        raise RuntimeError("運算失敗：無法產生指標訊號。")

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
    print("2. models/v16_best_params.json (AI 訓練最佳參數)")
    print("3. 使用 v16_config.py 系統預設值")
    
    choice = input("請輸入 1, 2, 或 3 (預設為 1): ").strip()
    original_params_path = os.path.join(BASE_DIR, "v16_original_params.json")
    
    if choice == '2':
        params, success = load_params_from_json(os.path.join(BASE_DIR, "models", "v16_best_params.json"))
        param_source = "models/v16_best_params.json"
    elif choice == '3':
        params = V16StrategyParams()
        param_source = "v16_config.py 系統預設"
    else:
        if os.path.exists(original_params_path):
            params, success = load_params_from_json(original_params_path)
            param_source = "v16_original_params.json"
        else:
            print(f"{C_CYAN}⚠️ 找不到 {original_params_path}，改用 v16_config.py 系統預設值。{C_RESET}")
            params = V16StrategyParams()
            param_source = "v16_config.py 系統預設 (fallback: 缺少 v16_original_params.json)"

    while True:
        ticker_input = input(f"請輸入要測試的股票代碼 (例如 2330，輸入 'q' 離開): ").strip()
        
        if ticker_input.lower() == 'q': break
        if not ticker_input: continue
            
        testing_csv_dir = os.path.join(BASE_DIR, "testing_csv")
        target_csv_path, _duplicate_file_issue_lines = resolve_unique_csv_path(testing_csv_dir, ticker_input)
        compare_with_tv(target_csv_path, params, param_source)