import os
import sys
import json
import pandas as pd
import numpy as np
import warnings
import time

from core.v16_config import V16StrategyParams
from core.v16_portfolio_engine import prep_stock_data_and_trades, run_portfolio_timeline
from core.v16_display import print_strategy_dashboard, C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET

warnings.filterwarnings('ignore')

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_dynamic_params(json_file):
    params = V16StrategyParams()
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(params, k): setattr(params, k, v)
            return params, True
        except Exception as e: 
            print(f"{C_YELLOW}⚠️ 讀取參數 {json_file} 失敗: {e}{C_RESET}")
    return params, False

def run_portfolio_simulation(data_dir, params, max_positions=5, enable_rotation=False, start_year=2015, benchmark_ticker="0050"):
    if not os.path.exists(data_dir):
        print(f"\n{C_RED}❌ 嚴重錯誤：找不到資料夾 {data_dir}，請確認路徑或先下載資料！{C_RESET}")
        sys.exit(1)
    print(f"{C_CYAN}📦 正在預載入歷史軌跡，構建真實時間軸...{C_RESET}")
    all_dfs, all_trade_logs, master_dates = {}, {}, set()
    
    # 確保作業系統層級的檔案讀取順序絕對一致
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    for count, file in enumerate(csv_files):
        ticker = file.replace('.csv', '').replace('TV_Data_Full_', '')
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            if len(df) < params.high_len + 10: continue
            
            df.columns = [c.capitalize() for c in df.columns]
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan).ffill()
            date_col = 'Time' if 'Time' in df.columns else 'Date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            df, logs = prep_stock_data_and_trades(df, params)
            all_dfs[ticker], all_trade_logs[ticker] = df, logs
            master_dates.update(df.index)
        except Exception as e:
            print(f"{C_YELLOW}\n[警告] 股票 {ticker} 處理失敗: {e}{C_RESET}")
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
    
    # 精準 19 個變數解包，支援 max_exp
    df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate = run_portfolio_timeline(
        all_dfs_fast, all_trade_logs, sorted_dates, start_year, params, max_positions, enable_rotation, 
        benchmark_ticker=benchmark_ticker, benchmark_data=benchmark_data, is_training=False
    )
    return df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate

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
    df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate = run_portfolio_simulation(
        "tw_stock_data_vip", params, USER_MAX_POS, USER_ROTATION, USER_START_YEAR, USER_BENCHMARK
    )
    end_time = time.time()
    
    mode_display = "開啟 (強勢輪動)" if USER_ROTATION else "關閉 (穩定鎖倉)"
    
    print(f"\n{C_CYAN}================================================================================{C_RESET}")
    print(f"📊 【投資組合實戰模擬報告 (自 {USER_START_YEAR} 年起算)】")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"回測總耗時: {end_time - start_time:.2f} 秒")
    
    print_strategy_dashboard(
        params=params, title="績效與風險對比表", mode_display=mode_display, max_pos=USER_MAX_POS,
        trades=trade_count, missed_b=total_missed, missed_s=total_missed_sells,
        final_eq=final_eq, avg_exp=avg_exp, sys_ret=tot_ret, bm_ret=bm_ret,
        sys_mdd=mdd, bm_mdd=bm_mdd, win_rate=win_rate, payoff=pf_payoff, ev=pf_ev,
        benchmark_ticker=USER_BENCHMARK, max_exp=max_exp,
        r_sq=r_sq, m_win_rate=m_win_rate, bm_r_sq=bm_r_sq, bm_m_win_rate=bm_m_win_rate
    )
    
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
    except Exception: pass