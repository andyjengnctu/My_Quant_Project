import sys
import os
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 🌟 依賴您的模組化架構：直接引入原本的參數檔與核心引擎
from core.v16_config import V16StrategyParams
from core.v16_core import run_v16_backtest
from core.v16_data_utils import sanitize_ohlcv_dataframe, LOAD_DATA_MIN_ROWS

warnings.filterwarnings('ignore')

C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")
OUTPUT_FILE = "outputs/V16_All_Stocks_Stats_Report.xlsx"
DEFAULT_EXPORT_MAX_WORKERS = 14

def load_params(json_file=os.path.join(BASE_DIR, "models", "v16_best_params.json")):
    params = V16StrategyParams()
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(params, k):
                    setattr(params, k, v)
            print(f"{C_GREEN}✅ 成功載入參數大腦: {json_file}{C_RESET}")
        except Exception as e:
            print(f"{C_YELLOW}⚠️ 載入 {json_file} 失敗，使用系統預設值。({type(e).__name__}: {e}){C_RESET}")
    else:
        print(f"{C_YELLOW}⚠️ 找不到 {json_file}，使用系統預設值。{C_RESET}")
    return params

# # (AI註: 第16點 - 平行度可控，避免 ProcessPoolExecutor 預設開太滿造成上下文切換成本上升)
def resolve_export_max_workers(params):
    configured = getattr(params, 'export_max_workers', DEFAULT_EXPORT_MAX_WORKERS)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_EXPORT_MAX_WORKERS
    return max(1, configured)

def process_single_stock_for_export(file_name, params):
    """處理單檔股票並回傳所需數據"""
    ticker = file_name.replace('.csv', '').replace('TV_Data_Full_', '')
    file_path = os.path.join(DATA_DIR, file_name)

    try:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = max(LOAD_DATA_MIN_ROWS, params.high_len + 10)
        if len(raw_df) < min_rows_needed:
            return None

        df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

        # 🌟 呼叫您原汁原味的 v16_core 引擎
        stats = run_v16_backtest(df, params)

        # 過濾掉沒有交易的股票
        if not stats or stats['trade_count'] == 0:
            return None

        invalid_row_count = sanitize_stats['invalid_row_count']
        duplicate_date_count = sanitize_stats['duplicate_date_count']
        dropped_row_count = sanitize_stats['dropped_row_count']

        # 將核心引擎算出來的數字打包
        return {
            "股票代號": ticker,
            "交易次數": stats.get("trade_count", 0),
            "勝率 (Win Rate %)": stats.get("win_rate", 0.0),
            "平均獲利金額 (avgWin)": stats.get("avg_win", 0.0),
            "平均虧損金額 (avgLoss)": stats.get("avg_loss", 0.0),
            "盈虧比 (payoffRatio)": stats.get("payoff_ratio", 0.0),
            "期望值 (expectedValue)": stats.get("expected_value", 0.0),
            "平均持倉天數": stats.get("avg_bars_held", 0.0),
            "總資產報酬率 (%)": stats.get("asset_growth", 0.0),
            "最大回撤 MDD (%)": stats.get("max_drawdown", 0.0),
            "清洗移除筆數": dropped_row_count,
            "異常OHLCV筆數": invalid_row_count,
            "重複日期筆數": duplicate_date_count
        }
    except Exception as e:
        # # (AI註: 補捉具體錯誤，避免運算失敗被無聲忽略)
        print(f"{C_YELLOW}⚠️ 處理 {ticker} 時發生錯誤: {type(e).__name__}: {e}{C_RESET}")
        return None
    
def main():
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"📊 {C_YELLOW}V16 全市場數據匯出工具啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(DATA_DIR):
        print("❌ 找不到資料夾，請確認路徑。")
        return
        
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("📂 資料夾內沒有 CSV 檔案。")
        return
        
    params = load_params()
    results = []
    max_workers = resolve_export_max_workers(params)

    print(f"\n🚀 開始平行掃描 {len(csv_files)} 檔股票... (max_workers={max_workers})")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_stock_for_export, f, params) for f in csv_files]

        completed = 0
        for future in as_completed(futures):
            completed += 1
            print(f"\r⏳ 進度: [{completed}/{len(csv_files)}] 掃描中...", end="", flush=True)
            
            res = future.result()
            if res:
                results.append(res)
                
    print(f"\n\n{C_CYAN}📈 運算完成！正在整理並匯出 Excel...{C_RESET}")
    
    if results:
        df_out = pd.DataFrame(results)
        
        # 預設依期望值由大到小排序
        df_out.sort_values(by="期望值 (expectedValue)", ascending=False, inplace=True)
        
        # 美化小數點位數
        df_out = df_out.round({
            "勝率 (Win Rate %)": 2,
            "平均獲利金額 (avgWin)": 0,
            "平均虧損金額 (avgLoss)": 0,
            "盈虧比 (payoffRatio)": 2,
            "期望值 (expectedValue)": 2,
            "平均持倉天數": 1,
            "總資產報酬率 (%)": 2,
            "最大回撤 MDD (%)": 2,
            "清洗移除筆數": 0,
            "異常OHLCV筆數": 0,
            "重複日期筆數": 0
        })
        
        df_out.to_excel(OUTPUT_FILE, index=False)
        print(f"{C_GREEN}🎉 匯出成功！請在資料夾中打開 '{OUTPUT_FILE}' 查看結果！{C_RESET}")
        print(f"{C_CYAN}================================================================================{C_RESET}")
    else:
        print(f"{C_YELLOW}⚠️ 沒有任何股票產生有效交易紀錄。{C_RESET}")

if __name__ == "__main__":
    main()