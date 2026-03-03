import os
import sys
import json
import pandas as pd
import numpy as np
import warnings
import time

from core.v16_config import V16StrategyParams
from core.v16_portfolio_engine import prep_stock_data_and_trades, run_portfolio_timeline

warnings.filterwarnings('ignore')
C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET = '\033[91m', '\033[93m', '\033[96m', '\033[92m', '\033[90m', '\033[0m'

def load_dynamic_params(json_file):
    params = V16StrategyParams()
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(params, k): setattr(params, k, v)
            return params, True
        except: pass
    return params, False

def run_portfolio_simulation(data_dir, params, max_positions=5, enable_rotation=False, start_year=2015, benchmark_ticker="0050"):
    print(f"{C_CYAN}📦 正在預載入歷史軌跡，構建真實時間軸...{C_RESET}")
    all_dfs, all_trade_logs, master_dates = {}, {}, set()
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for count, file in enumerate(csv_files):
        ticker = file.replace('.csv', '').replace('TV_Data_Full_', '')
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            if len(df) < 150: continue
            df.columns = [c.capitalize() for c in df.columns]
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan).ffill()
            date_col = 'Time' if 'Time' in df.columns else 'Date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            df, logs = prep_stock_data_and_trades(df, params)
            all_dfs[ticker], all_trade_logs[ticker] = df, logs
            master_dates.update(df.index)
        except Exception as e:
            print(f"{C_RED}載入 {ticker} 發生錯誤: {e}{C_RESET}")
            continue
            
        if count % 50 == 0: print(f"{C_GRAY}   進度: 已處理 {count} 檔股票...{C_RESET}", end="\r")

    if not all_dfs:
        print(f"\n{C_RED}❌ 嚴重錯誤：未能成功載入任何股票資料！{C_RESET}")
        sys.exit()

    sorted_dates = sorted(list(master_dates))
    all_dfs_fast = {t: d.to_dict('index') for t, d in all_dfs.items()}
    print(f"\n{C_GREEN}✅ 預處理完成！自 {start_year} 年開始啟動真實時間軸回測...{C_RESET}\n")

    benchmark_data = all_dfs_fast.get(benchmark_ticker, None)

    print(" " * 120, end="\r") 
    df_eq, df_tr, tot_ret, mdd, win_rate, pf_ev, pf_payoff, final_eq, bm_ret, bm_mdd = run_portfolio_timeline(
        all_dfs_fast, all_trade_logs, sorted_dates, start_year, params, max_positions, enable_rotation, 
        benchmark_ticker=benchmark_ticker, 
        benchmark_data=benchmark_data, 
        is_training=False
    )
    return df_eq, df_tr, tot_ret, mdd, win_rate, pf_ev, pf_payoff, final_eq, bm_ret, bm_mdd

if __name__ == "__main__":
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 投資組合模擬器：機構級實戰期望值 (終極模組化對齊版){C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    USER_ROTATION = input(f"👉 1. 啟用「汰弱換股」？ (Y/N, 預設 N): ").strip().upper() == 'Y'
    USER_MAX_POS = int(input(f"👉 2. 最大持倉數量 (預設 10): ").strip() or 10)
    USER_START_YEAR = int(input(f"👉 3. 開始回測年份 (預設 2015): ").strip() or 2015)
    USER_BENCHMARK = input(f"👉 4. 大盤比較標的 (預設 0050): ").strip() or "0050"

    params, is_loaded = load_dynamic_params("models/v16_best_params.json")
    if is_loaded: print(f"\n{C_GREEN}✅ 成功載入 AI 訓練大腦！{C_RESET}")

    start_time = time.time()
    df_eq, df_tr, tot_ret, mdd, win_rate, pf_ev, pf_payoff, final_eq, bm_ret, bm_mdd = run_portfolio_simulation(
        "tw_stock_data_vip", params, USER_MAX_POS, USER_ROTATION, USER_START_YEAR, USER_BENCHMARK
    )
    end_time = time.time()
    
    avg_exposure = df_eq['Exposure_Pct'].mean() if not df_eq.empty else 0.0
    max_exposure = df_eq['Exposure_Pct'].max() if not df_eq.empty else 0.0
    mode_display = "開啟 (強勢輪動)" if USER_ROTATION else "關閉 (穩定鎖倉)"
    alpha = tot_ret - bm_ret
    mdd_diff = bm_mdd - mdd
    
    sys_ret_str = f"+{tot_ret:.2f}%" if tot_ret > 0 else f"{tot_ret:.2f}%"
    bm_ret_str  = f"+{bm_ret:.2f}%" if bm_ret > 0 else f"{bm_ret:.2f}%"
    alpha_str   = f"+{alpha:.2f}%" if alpha > 0 else f"{alpha:.2f}%"
    
    sys_mdd_str = f"-{mdd:.2f}%"
    bm_mdd_str  = f"-{bm_mdd:.2f}%"
    mdd_diff_str = f"少跌 {mdd_diff:.2f}%" if mdd_diff > 0 else f"多跌 {abs(mdd_diff):.2f}%"
    
    alpha_color = C_GREEN if alpha > 0 else C_RED
    sys_ret_color = C_GREEN if tot_ret > 0 else C_RED
    mdd_diff_color = C_GREEN if mdd_diff > 0 else C_RED
    
    sys_romd = (tot_ret / abs(mdd)) if mdd != 0 else 0.0
    bm_romd = (bm_ret / abs(bm_mdd)) if bm_mdd != 0 else 0.0
    romd_diff = sys_romd - bm_romd

    sys_romd_str = f"{sys_romd:.2f}"
    bm_romd_str = f"{bm_romd:.2f}"
    romd_diff_str = f"+{romd_diff:.2f}" if romd_diff > 0 else f"{romd_diff:.2f}"
    romd_diff_color = C_GREEN if romd_diff > 0 else C_RED
    
    closed_trades_count = len(df_tr[~df_tr['Type'].str.contains('買進|換入', regex=True, na=False)]) if not df_tr.empty else 0

    print(f"\n{C_CYAN}================================================================================{C_RESET}")
    print(f"📊 【投資組合實戰模擬報告 (自 {USER_START_YEAR} 年起算)】")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ 基礎設定")
    print(f"--------------------------------------------------------------------------------")
    print(f"模式: {mode_display} | 最大持股: {USER_MAX_POS} 檔 | 回測總耗時: {end_time - start_time:.2f} 秒")
    print(f"總交易紀錄: {closed_trades_count} 筆 (明細流水帳 {len(df_tr)} 筆) | 最終資產: {final_eq:,.0f} 元")
    print(f"平均資金水位: {avg_exposure:>.2f} % (最高 {max_exposure:>.2f} %)")
    print(f"--------------------------------------------------------------------------------")
    print(f"🏆 績效與風險對比表")
    print(f"--------------------------------------------------------------------------------")
    print(f"| 指標項目         | V16 尊爵系統   | 同期大盤 ({USER_BENCHMARK:<4}) | 差異 (Alpha)   |")
    print(f"|------------------|----------------|-----------------|----------------|")
    print(f"| 總資產報酬率     | {sys_ret_color}{sys_ret_str:<14}{C_RESET} | {bm_ret_str:<15} | {alpha_color}{alpha_str:<14}{C_RESET} |")
    print(f"| 最大回撤 (MDD)   | {C_YELLOW}{sys_mdd_str:<14}{C_RESET} | {bm_mdd_str:<15} | {mdd_diff_color}{mdd_diff_str:<14}{C_RESET} |")
    print(f"| 報酬回撤比(RoMD) | {C_CYAN}{sys_romd_str:<14}{C_RESET} | {bm_romd_str:<15} | {romd_diff_color}{romd_diff_str:<14}{C_RESET} |")
    print(f"| 系統實戰勝率     | {win_rate:>6.2f} %       | -               | -              |")
    print(f"| 盈虧風報比       | {pf_payoff:>6.2f}         | -               | -              |")
    print(f"| 實戰期望值(EV)   | {pf_ev:>6.2f} R       | -               | -              |")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    with pd.ExcelWriter("outputs/V16_Portfolio_Report.xlsx") as writer:
        df_eq.to_excel(writer, sheet_name="Equity Curve", index=False)
        df_tr.to_excel(writer, sheet_name="Trade History", index=False)
    print(f"{C_GREEN}📁 完整資產曲線與交易明細已匯出至: V16_Portfolio_Report.xlsx{C_RESET}")

    try:
        import plotly.graph_objects as go
        import webbrowser
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq['Strategy_Return_Pct'], mode='lines', name='V16 尊爵系統報酬 (%)', line=dict(color='#ff3333', width=3)))
        
        bm_col = f"Benchmark_{USER_BENCHMARK}_Pct"
        if bm_col in df_eq.columns:
            fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq[bm_col], mode='lines', name=f'同期大盤 {USER_BENCHMARK} (%)', line=dict(color='#4dabf5', width=2), opacity=0.8))
            
        fig.update_layout(title=f'<b>V16 投資組合實戰淨值 vs {USER_BENCHMARK} 大盤</b> ({USER_START_YEAR} 至今)', xaxis_title='日期', yaxis_title='累積報酬率 (%)', template='plotly_dark', hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"), margin=dict(l=40, r=40, t=60, b=40))
        
        html_filename = "outputs/V16_Portfolio_Dashboard.html"
        fig.write_html(html_filename)
        print(f"{C_GREEN}📊 互動式網頁已生成: {html_filename}{C_RESET}")
        webbrowser.open('file://' + os.path.realpath(html_filename))
        
    except ImportError: print(f"{C_RED}⚠️ 無法產生動態網頁。請先執行: pip install plotly{C_RESET}")
    except Exception as e: print(f"{C_RED}⚠️ 圖表產生失敗: {e}{C_RESET}")