import os
import webbrowser

import numpy as np
import pandas as pd

from core.display import C_CYAN, C_GREEN, C_RESET, C_YELLOW
from core.log_utils import format_exception_summary
from core.output_paths import build_output_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "portfolio_sim")
REPORT_XLSX_PATH = os.path.join(OUTPUT_DIR, "V16_Portfolio_Report.xlsx")
DASHBOARD_HTML_PATH = os.path.join(OUTPUT_DIR, "V16_Portfolio_Dashboard.html")


def print_yearly_return_report(yearly_return_rows, benchmark_yearly_return_rows=None, benchmark_ticker="0050"):
    print(f"\n{C_CYAN}================================================================================{C_RESET}")
    print(f"📅 【各年度報酬率 vs {benchmark_ticker} 大盤】")
    print(f"{C_CYAN}================================================================================{C_RESET}")

    base_columns = ["year", "year_return_pct", "is_full_year", "start_date", "end_date"]
    if not yearly_return_rows:
        print(f"{C_YELLOW}⚠️ 無年度報酬率資料。{C_RESET}")
        return pd.DataFrame(columns=base_columns)

    df_yearly = pd.DataFrame(yearly_return_rows).copy()
    if df_yearly.empty:
        print(f"{C_YELLOW}⚠️ 無年度報酬率資料。{C_RESET}")
        return pd.DataFrame(columns=base_columns)

    df_yearly["year_label"] = df_yearly["year"].astype(str)
    df_yearly["year_type"] = np.where(df_yearly["is_full_year"], "完整", "非完整")

    df_display = df_yearly.copy()
    bm_rows = list(benchmark_yearly_return_rows or [])
    if bm_rows:
        df_bm = pd.DataFrame(bm_rows).copy().rename(columns={"year_return_pct": "benchmark_year_return_pct"})
        merge_keys = ["year", "is_full_year", "start_date", "end_date"]
        available_merge_keys = [col for col in merge_keys if col in df_bm.columns and col in df_display.columns]
        df_display = df_display.merge(
            df_bm[available_merge_keys + ["benchmark_year_return_pct"]],
            on=available_merge_keys,
            how="left",
        )
    else:
        df_display["benchmark_year_return_pct"] = np.nan

    df_display["alpha_pct"] = df_display["year_return_pct"] - df_display["benchmark_year_return_pct"]

    print(
        df_display[[
            "year_label", "year_return_pct", "benchmark_year_return_pct", "alpha_pct",
            "year_type", "start_date", "end_date"
        ]].to_string(
            index=False,
            formatters={
                "year_return_pct": "{:.2f}%".format,
                "benchmark_year_return_pct": lambda x: "-" if pd.isna(x) else f"{x:.2f}%",
                "alpha_pct": lambda x: "-" if pd.isna(x) else f"{x:.2f}%",
            }
        )
    )
    return df_yearly



def export_portfolio_reports(df_eq, df_tr, df_yearly, benchmark_ticker, start_year, end_year=None):
    with pd.ExcelWriter(REPORT_XLSX_PATH) as writer:
        df_eq.to_excel(writer, sheet_name="Equity Curve", index=False)
        df_tr.to_excel(writer, sheet_name="Trade History", index=False)
        df_yearly.to_excel(writer, sheet_name="Yearly Returns", index=False)
    print(f"{C_GREEN}📁 完整資產曲線、交易明細與各年度報酬率已匯出至: {REPORT_XLSX_PATH}{C_RESET}")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq['Strategy_Return_Pct'], mode='lines', name='V16 尊爵系統報酬 (%)', line=dict(color='#ff3333', width=3)))
        bm_col = f"Benchmark_{benchmark_ticker}_Pct"
        if bm_col in df_eq.columns:
            fig.add_trace(go.Scatter(x=df_eq['Date'], y=df_eq[bm_col], mode='lines', name=f'同期大盤 {benchmark_ticker} (%)', line=dict(color='#4dabf5', width=2), opacity=0.8))
        end_label = "至今" if end_year is None else f"至 {int(end_year)}"
        fig.update_layout(title=f'<b>V16 投資組合實戰淨值 vs {benchmark_ticker} 大盤</b> ({start_year} {end_label})', xaxis_title='日期', yaxis_title='累積報酬率 (%)', template='plotly_dark', hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"), margin=dict(l=40, r=40, t=60, b=40))
        fig.write_html(DASHBOARD_HTML_PATH)
        print(f"{C_GREEN}📊 互動式網頁已生成: {DASHBOARD_HTML_PATH}{C_RESET}")
    except (ImportError, OSError, ValueError, RuntimeError, webbrowser.Error) as e:
        print(f"{C_YELLOW}⚠️ Plotly 圖表輸出或開啟失敗: {format_exception_summary(e)}{C_RESET}")
