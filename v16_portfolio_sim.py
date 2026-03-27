import os
import time
import warnings
import webbrowser

import numpy as np
import pandas as pd

from core.v16_params_io import load_params_from_json
from core.v16_portfolio_engine import prep_stock_data_and_trades, pack_prepared_stock_data, run_portfolio_timeline
from core.v16_display import print_strategy_dashboard, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows, discover_unique_csv_inputs
from core.v16_log_utils import write_issue_log, format_exception_summary

# # (AI註: 收窄 warning 範圍；不要把資料品質與數值異常全部全域吃掉)
warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "tw_stock_data_vip")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "v16_best_params.json")
REPORT_XLSX_PATH = os.path.join(OUTPUT_DIR, "V16_Portfolio_Report.xlsx")
DASHBOARD_HTML_PATH = os.path.join(OUTPUT_DIR, "V16_Portfolio_Dashboard.html")


# # (AI註: 將目錄建立延後到實際執行期，避免被 import 時污染呼叫端工作目錄)
def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


LOAD_PROGRESS_EVERY = 50


def load_strict_params(json_file):
    return load_params_from_json(json_file)


# # (AI註: 將「清洗後有效資料不足」與真正異常分流，避免 portfolio_sim 被新上市/短歷史標的洗板)
def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))


# # (AI註: 只顯示年度報酬，不再在 portfolio_sim 內維持股票集中度相關報表)
def print_yearly_return_report(yearly_return_rows):
    print(f"\n{C_CYAN}================================================================================{C_RESET}")
    print("📅 【各年度報酬率】")
    print(f"{C_CYAN}================================================================================{C_RESET}")

    if not yearly_return_rows:
        print(f"{C_YELLOW}⚠️ 無年度報酬率資料。{C_RESET}")
        return pd.DataFrame(columns=["year", "year_return_pct", "is_full_year", "start_date", "end_date"])

    df_yearly = pd.DataFrame(yearly_return_rows).copy()
    if df_yearly.empty:
        print(f"{C_YELLOW}⚠️ 無年度報酬率資料。{C_RESET}")
        return df_yearly

    df_yearly["year_label"] = df_yearly["year"].astype(str)
    df_yearly["year_type"] = np.where(df_yearly["is_full_year"], "完整", "非完整")

    print(
        df_yearly[["year_label", "year_return_pct", "year_type", "start_date", "end_date"]].to_string(
            index=False,
            formatters={"year_return_pct": "{:.2f}%".format}
        )
    )
    return df_yearly


def run_portfolio_simulation(data_dir, params, max_positions=5, enable_rotation=False, start_year=2015, benchmark_ticker="0050", verbose=True):
    ensure_runtime_dirs()
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到資料夾 {data_dir}，請確認路徑或先下載資料！")

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"{C_CYAN}📦 正在預載入歷史軌跡，構建真實時間軸...{C_RESET}")
    all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
    load_issue_lines = []
    total_invalid_rows = 0
    total_duplicate_dates = 0
    total_dropped_rows = 0
    total_skipped_insufficient = 0
    total_sanitize_issue_tickers = 0

    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    load_issue_lines.extend(duplicate_file_issue_lines)
    total_files = len(csv_inputs)

    for count, (ticker, file_path) in enumerate(csv_inputs, start=1):
        try:
            raw_df = pd.read_csv(file_path)
            min_rows_needed = get_required_min_rows(params)

            if len(raw_df) < min_rows_needed:
                total_skipped_insufficient += 1
                load_issue_lines.append(
                    f"[資料不足] {ticker}: 原始資料列數不足 ({len(raw_df)})，至少需要 {min_rows_needed} 筆"
                )
                continue

            df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

            invalid_row_count = sanitize_stats['invalid_row_count']
            duplicate_date_count = sanitize_stats['duplicate_date_count']
            dropped_row_count = sanitize_stats['dropped_row_count']

            total_invalid_rows += invalid_row_count
            total_duplicate_dates += duplicate_date_count
            total_dropped_rows += dropped_row_count

            if dropped_row_count > 0:
                total_sanitize_issue_tickers += 1
                load_issue_lines.append(
                    f"[清洗] {ticker}: 清洗移除 {dropped_row_count} 列 "
                    f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
                )

            df, logs = prep_stock_data_and_trades(df, params)
            master_dates.update(df.index)
            all_dfs_fast[ticker] = pack_prepared_stock_data(df)
            all_trade_logs[ticker] = logs

        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
            if is_insufficient_data_error(e):
                total_skipped_insufficient += 1
                load_issue_lines.append(f"[資料不足] {ticker}: {type(e).__name__}: {e}")
                continue
            raise RuntimeError(
                f"預載入失敗: ticker={ticker} | {format_exception_summary(e)}"
            ) from e

        if count % LOAD_PROGRESS_EVERY == 0 or count == total_files:
            vprint(
                f"{C_GRAY}   預載入進度: [{count}/{total_files}] "
                f"成功:{len(all_dfs_fast)} | 資料不足:{total_skipped_insufficient}{C_RESET}",
                end="\r",
                flush=True
            )

    load_log_path = write_issue_log("portfolio_sim_load_issues", load_issue_lines, log_dir=OUTPUT_DIR) if load_issue_lines else None

    vprint(" " * 160, end="\r")

    if load_log_path:
        vprint(f"{C_YELLOW}⚠️ 預載入摘要已寫入: {load_log_path}{C_RESET}")

    if not all_dfs_fast:
        raise RuntimeError("未能成功載入任何股票資料！")

    sorted_dates = sorted(list(master_dates))

    vprint(
        f"\n{C_GREEN}✅ 預處理完成！共載入 {len(all_dfs_fast)} 檔標的，"
        f"移除 {total_dropped_rows} 列資料 "
        f"(異常OHLCV={total_invalid_rows}, 重複日期={total_duplicate_dates})，"
        f"候選清洗 {total_sanitize_issue_tickers} 檔，"
        f"資料不足跳過 {total_skipped_insufficient} 檔。"
        f"自 {start_year} 年開始啟動真實時間軸回測...{C_RESET}\n"
    )

    benchmark_data = all_dfs_fast.get(benchmark_ticker, None)

    vprint(" " * 120, end="\r")

    pf_profile = {}
    df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct = run_portfolio_timeline(
        all_dfs_fast, all_trade_logs, sorted_dates, start_year, params, max_positions, enable_rotation,
        benchmark_ticker=benchmark_ticker, benchmark_data=benchmark_data, is_training=False, profile_stats=pf_profile, verbose=verbose
    )
    return df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, pf_profile


def _safe_prompt(prompt_text, default_value):
    try:
        raw = input(prompt_text).strip()
    except EOFError:
        return default_value
    return raw if raw != "" else default_value


if __name__ == "__main__":
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 投資組合模擬器：機構級實戰期望值 (終極模組化對齊版){C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")

    USER_ROTATION = _safe_prompt("👉 1. 啟用「汰弱換股」？ (Y/N, 預設 N): ", "N").upper() == 'Y'
    USER_MAX_POS = int(_safe_prompt("👉 2. 最大持倉數量 (預設 10): ", "10"))
    USER_START_YEAR = int(_safe_prompt("👉 3. 開始回測年份 (預設 2015): ", "2015"))
    USER_BENCHMARK = _safe_prompt("👉 4. 大盤比較標的 (預設 0050): ", "0050")

    ensure_runtime_dirs()
    params = load_strict_params(BEST_PARAMS_PATH)
    print(f"\n{C_GREEN}✅ 成功載入 AI 訓練大腦！{C_RESET}")

    start_time = time.time()
    df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, pf_profile = run_portfolio_simulation(
        DEFAULT_DATA_DIR, params, USER_MAX_POS, USER_ROTATION, USER_START_YEAR, USER_BENCHMARK
    )
    end_time = time.time()

    mode_display = "開啟 (強勢輪動)" if USER_ROTATION else "關閉 (穩定鎖倉)"

    min_full_year_return_pct = pf_profile.get("min_full_year_return_pct", 0.0)
    bm_min_full_year_return_pct = pf_profile.get("bm_min_full_year_return_pct", 0.0)

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
        r_sq=r_sq, m_win_rate=m_win_rate, bm_r_sq=bm_r_sq, bm_m_win_rate=bm_m_win_rate,
        normal_trades=normal_trade_count, extended_trades=extended_trade_count,
        annual_trades=annual_trades, reserved_buy_fill_rate=reserved_buy_fill_rate,
        annual_return_pct=annual_return_pct, bm_annual_return_pct=bm_annual_return_pct,
        min_full_year_return_pct=min_full_year_return_pct, bm_min_full_year_return_pct=bm_min_full_year_return_pct
    )

    df_yearly = print_yearly_return_report(pf_profile.get("yearly_return_rows", []))
    if pf_profile.get("full_year_count", 0) > 0:
        print(
            f"{C_GRAY}完整年度數: {pf_profile.get('full_year_count', 0)} | "
            f"最差完整年度報酬: {pf_profile.get('min_full_year_return_pct', 0.0):.2f}% | "
            f"大盤最差完整年度報酬: {pf_profile.get('bm_min_full_year_return_pct', 0.0):.2f}% | "
            f"年化報酬率: {annual_return_pct:.2f}%{C_RESET}"
        )

    with pd.ExcelWriter(REPORT_XLSX_PATH) as writer:
        df_eq.to_excel(writer, sheet_name="Equity Curve", index=False)
        df_tr.to_excel(writer, sheet_name="Trade History", index=False)
        df_yearly.to_excel(writer, sheet_name="Yearly Returns", index=False)
    print(f"{C_GREEN}📁 完整資產曲線、交易明細與各年度報酬率已匯出至: {REPORT_XLSX_PATH}{C_RESET}")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq['Strategy_Return_Pct'], mode='lines', name='V16 尊爵系統報酬 (%)', line=dict(color='#ff3333', width=3)))
        bm_col = f"Benchmark_{USER_BENCHMARK}_Pct"
        if bm_col in df_eq.columns:
            fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq[bm_col], mode='lines', name=f'同期大盤 {USER_BENCHMARK} (%)', line=dict(color='#4dabf5', width=2), opacity=0.8))
        fig.update_layout(title=f'<b>V16 投資組合實戰淨值 vs {USER_BENCHMARK} 大盤</b> ({USER_START_YEAR} 至今)', xaxis_title='日期', yaxis_title='累積報酬率 (%)', template='plotly_dark', hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"), margin=dict(l=40, r=40, t=60, b=40))
        html_filename = DASHBOARD_HTML_PATH
        fig.write_html(html_filename)
        print(f"{C_GREEN}📊 互動式網頁已生成: {html_filename}{C_RESET}")
        webbrowser.open('file://' + os.path.realpath(html_filename))
    except (ImportError, OSError, ValueError, RuntimeError, webbrowser.Error) as e:
        print(f"{C_YELLOW}⚠️ Plotly 圖表輸出或開啟失敗: {format_exception_summary(e)}{C_RESET}")