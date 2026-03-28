import os
import webbrowser

import numpy as np
import pandas as pd

from core.display import C_CYAN, C_GREEN, C_GRAY, C_RESET, C_YELLOW
from core.log_utils import format_exception_summary
from core.runtime_utils import should_auto_open_browser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORT_XLSX_PATH = os.path.join(OUTPUT_DIR, "V16_Portfolio_Report.xlsx")
DASHBOARD_HTML_PATH = os.path.join(OUTPUT_DIR, "V16_Portfolio_Dashboard.html")


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


def export_portfolio_reports(df_eq, df_tr, df_yearly, benchmark_ticker, start_year):
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
        fig.update_layout(title=f'<b>V16 投資組合實戰淨值 vs {benchmark_ticker} 大盤</b> ({start_year} 至今)', xaxis_title='日期', yaxis_title='累積報酬率 (%)', template='plotly_dark', hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"), margin=dict(l=40, r=40, t=60, b=40))
        fig.write_html(DASHBOARD_HTML_PATH)
        print(f"{C_GREEN}📊 互動式網頁已生成: {DASHBOARD_HTML_PATH}{C_RESET}")
        if should_auto_open_browser(os.environ):
            webbrowser.open('file://' + os.path.realpath(DASHBOARD_HTML_PATH))
        else:
            print(f"{C_GRAY}ℹ️ 目前為非互動或無圖形環境，略過自動開啟瀏覽器。{C_RESET}")
    except (ImportError, OSError, ValueError, RuntimeError, webbrowser.Error) as e:
        print(f"{C_YELLOW}⚠️ Plotly 圖表輸出或開啟失敗: {format_exception_summary(e)}{C_RESET}")
