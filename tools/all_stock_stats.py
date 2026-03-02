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

warnings.filterwarnings('ignore')

C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")
OUTPUT_FILE = "outputs/V16_All_Stocks_Stats_Report.xlsx"

def load_params(json_file=os.path.join(BASE_DIR, "models", "v16_best_params.json")):
    params = V16StrategyParams()
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(params, k):
                    setattr(params, k, v)
            print(f"{C_GREEN}✅ 成功載入參數大腦: {json_file}{C_RESET}")
        except Exception as e:
            print(f"{C_YELLOW}⚠️ 載入 {json_file} 失敗，使用系統預設值。({e}){C_RESET}")
    else:
        print(f"{C_YELLOW}⚠️ 找不到 {json_file}，使用系統預設值。{C_RESET}")
    return params

def process_single_stock_for_export(file_name, params):
    """處理單檔股票並回傳所需數據"""
    ticker = file_name.replace('.csv', '').replace('TV_Data_Full_', '')
    file_path = os.path.join(DATA_DIR, file_name)
    
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: 
            return None
            
        df.columns = [c.capitalize() for c in df.columns]
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # 🌟 呼叫您原汁原味的 v16_core 引擎
        stats = run_v16_backtest(df, params)
        
        # 過濾掉沒有交易的股票
        if not stats or stats['trade_count'] == 0:
            return None
            
        # 將核心引擎算出來的數字打包
        return {
            "股票代號": ticker,
            "交易次數": stats.get("trade_count", 0),
            "勝率 (Win Rate %)": stats.get("win_rate", 0.0),
            "平均獲利金額 (avgWin)": stats.get("avg_win", 0.0),      # 從 core 抓出
            "平均虧損金額 (avgLoss)": stats.get("avg_loss", 0.0),   # 從 core 抓出
            "盈虧比 (payoffRatio)": stats.get("payoff_ratio", 0.0),
            "期望值 (expectedValue)": stats.get("expected_value", 0.0),
            "平均持倉天數": stats.get("avg_bars_held", 0.0),       # 從 core 抓出
            "總資產報酬率 (%)": stats.get("asset_growth", 0.0),
            "最大回撤 MDD (%)": stats.get("max_drawdown", 0.0)
        }
    except Exception:
        return None

def main():
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"📊 {C_YELLOW}V16 全市場數據匯出工具啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    if not os.path.exists(DATA_DIR):
        print("❌ 找不到資料夾，請確認路徑。")
        return
        
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("📂 資料夾內沒有 CSV 檔案。")
        return
        
    params = load_params()
    results = []
    
    print(f"\n🚀 開始平行掃描 {len(csv_files)} 檔股票...")
    
    with ProcessPoolExecutor() as executor:
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
            "最大回撤 MDD (%)": 2
        })
        
        df_out.to_excel(OUTPUT_FILE, index=False)
        print(f"{C_GREEN}🎉 匯出成功！請在資料夾中打開 '{OUTPUT_FILE}' 查看結果！{C_RESET}")
        print(f"{C_CYAN}================================================================================{C_RESET}")
    else:
        print(f"{C_YELLOW}⚠️ 沒有任何股票產生有效交易紀錄。{C_RESET}")

if __name__ == "__main__":
    main()