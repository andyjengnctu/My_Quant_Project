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
from core.v16_config import V16StrategyParams, BUY_SORT_METHOD
from core.v16_buy_sort import calc_buy_sort_value, get_buy_sort_title
from core.v16_core import run_v16_backtest
from core.v16_log_utils import format_exception_summary, write_issue_log
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows, discover_unique_csv_inputs

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)

C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")
OUTPUT_FILE = "outputs/V16_All_Stocks_Stats_Report.xlsx"
DEFAULT_EXPORT_MAX_WORKERS = 14

def resolve_report_sort_method():
    if BUY_SORT_METHOD == 'PROJ_COST':
        return 'EV'
    return BUY_SORT_METHOD

def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))

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

# # (AI註: 第16點 - 平行度可控，避免 ProcessPoolExecutor 預設開太滿造成上下文切換成本上升)
def resolve_export_max_workers(params):
    configured = getattr(params, 'export_max_workers', DEFAULT_EXPORT_MAX_WORKERS)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_EXPORT_MAX_WORKERS
    return max(1, configured)

def process_single_stock_for_export(ticker, file_path, params):
    """處理單檔股票並回傳所需數據"""

    try:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        if len(raw_df) < min_rows_needed:
            return {
                "__status__": "skip_insufficient",
                "__ticker__": ticker,
                "__reason__": f"原始資料列數不足 ({len(raw_df)})，至少需要 {min_rows_needed} 列"
            }

        df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

        # 🌟 呼叫您原汁原味的 v16_core 引擎
        stats = run_v16_backtest(df, params)

        # 過濾掉沒有交易的股票
        if not stats or stats['trade_count'] == 0:
            return None

        invalid_row_count = sanitize_stats['invalid_row_count']
        duplicate_date_count = sanitize_stats['duplicate_date_count']
        dropped_row_count = sanitize_stats['dropped_row_count']
        active_sort_method = resolve_report_sort_method()
        sort_value = calc_buy_sort_value(
            active_sort_method,
            stats.get("expected_value", 0.0),
            0.0,
            stats.get("win_rate", 0.0) / 100.0,
            stats.get("trade_count", 0)
        )

        # 將核心引擎算出來的數字打包
        return {
            "股票代號": ticker,
            "交易次數": stats.get("trade_count", 0),
            "勝率 (Win Rate %)": stats.get("win_rate", 0.0),
            "平均獲利金額 (avgWin)": stats.get("avg_win", 0.0),
            "平均虧損金額 (avgLoss)": stats.get("avg_loss", 0.0),
            "盈虧比 (payoffRatio)": stats.get("payoff_ratio", 0.0),
            "期望值 (expectedValue)": stats.get("expected_value", 0.0),
            "排序值 (sortValue)": sort_value,
            "平均持倉天數": stats.get("avg_bars_held", 0.0),
            "總資產報酬率 (%)": stats.get("asset_growth", 0.0),
            "最大回撤 MDD (%)": stats.get("max_drawdown", 0.0),
            "清洗移除筆數": dropped_row_count,
            "異常OHLCV筆數": invalid_row_count,
            "重複日期筆數": duplicate_date_count
        }
    except Exception as e:
        if is_insufficient_data_error(e):
            return {
                "__status__": "skip_insufficient",
                "__ticker__": ticker,
                "__reason__": f"{type(e).__name__}: {e}"
            }
        raise RuntimeError(
            f"全市場匯出失敗: ticker={ticker} | {format_exception_summary(e)}"
        ) from e
    
def main():
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"📊 {C_YELLOW}V16 全市場數據匯出工具啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"找不到資料夾: {DATA_DIR}")
        
    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(DATA_DIR)
    if not csv_inputs:
        raise FileNotFoundError(f"資料夾內沒有 CSV 檔案: {DATA_DIR}")
        
    params = load_params()
    results = []
    skipped_insufficient_count = 0
    export_issue_lines = list(duplicate_file_issue_lines)
    max_workers = resolve_export_max_workers(params)

    print(f"\n🚀 開始平行掃描 {len(csv_inputs)} 檔股票... (max_workers={max_workers})")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_stock_for_export, ticker, file_path, params)
            for ticker, file_path in csv_inputs
        ]

        completed = 0
        for future in as_completed(futures):
            completed += 1
            print(f"\r⏳ 進度: [{completed}/{len(csv_inputs)}] 掃描中...", end="", flush=True)
            
            res = future.result()
            if isinstance(res, dict) and res.get("__status__") == "skip_insufficient":
                skipped_insufficient_count += 1
                export_issue_lines.append(f"[資料不足] {res['__ticker__']}: {res['__reason__']}")
                continue

            if res:
                results.append(res)
                
    print(f"\n\n{C_CYAN}📈 運算完成！正在整理並匯出 Excel...{C_RESET}")

    export_issue_log_path = write_issue_log("all_stock_stats_skips", export_issue_lines) if export_issue_lines else None
    if export_issue_log_path:
        print(f"{C_YELLOW}⚠️ 資料不足略過 {skipped_insufficient_count} 檔，摘要已寫入: {export_issue_log_path}{C_RESET}")
    elif skipped_insufficient_count > 0:
        print(f"{C_YELLOW}⚠️ 資料不足略過 {skipped_insufficient_count} 檔。{C_RESET}")
    
    if results:
        df_out = pd.DataFrame(results)
        active_sort_method = resolve_report_sort_method()

        if BUY_SORT_METHOD == 'PROJ_COST':
            print(f"{C_YELLOW}⚠️ all_stock_stats 缺少盤前 sizing / qty 上下文，無法真實反映 PROJ_COST，已自動改用 EV 排序。{C_RESET}")

        print(f"   ➤ 報表排序方式: {get_buy_sort_title(active_sort_method)}")
        df_out.sort_values(by=["排序值 (sortValue)", "股票代號"], ascending=[False, False], inplace=True)

        # 美化小數點位數
        df_out = df_out.round({
            "勝率 (Win Rate %)": 2,
            "平均獲利金額 (avgWin)": 0,
            "平均虧損金額 (avgLoss)": 0,
            "盈虧比 (payoffRatio)": 2,
            "期望值 (expectedValue)": 2,
            "排序值 (sortValue)": 2,
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