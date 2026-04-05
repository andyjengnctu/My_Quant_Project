import os

import pandas as pd

from tools.debug.charting import export_debug_chart_html


def _emit_loss_summary(df_logs, colors):
    losses = df_logs[df_logs['單筆實質損益'] < 0]
    if losses.empty:
        return

    print(f"\n{colors['cyan']}🚨 [抓漏分析] 前 3 大嚴重虧損明細：{colors['reset']}")
    worst_losses = losses.sort_values(by='單筆實質損益', ascending=True).head(3)
    for _, row in worst_losses.iterrows():
        print(
            f"日期: {row['日期']} | 動作: {row['動作']:<4} | 股價: {row['成交價']:>6.2f} | "
            f"股數: {int(row['股數']):>6}股 | 總投入金: {row['投入總金額']:>9,.0f} | "
            f"💸 虧損: {row['單筆實質損益']:>9,.0f}"
        )
        print(
            f"   ➤ 當下 ATR 為 {row['ATR(前日)']:.2f}，"
            f"停損/賣出參考價為 {row['設定停損價']:.2f}。"
        )


def finalize_debug_analysis(
    *,
    trade_logs,
    ticker,
    output_dir,
    colors,
    export_excel=True,
    export_chart=True,
    verbose=True,
    price_df=None,
    chart_context=None,
):
    if not trade_logs:
        if verbose:
            print(f"{colors['yellow']}⚠️ 這檔股票沒有任何交易紀錄。{colors['reset']}")
        return {
            "trade_logs_df": None,
            "excel_path": None,
            "chart_path": None,
        }

    df_logs = pd.DataFrame(trade_logs)
    df_logs['投入總金額'] = df_logs['投入總金額'].round(0)

    excel_path = None
    chart_path = None
    if export_excel or export_chart:
        os.makedirs(output_dir, exist_ok=True)

    if export_excel:
        excel_path = os.path.join(output_dir, f"Debug_TradeLog_{ticker}.xlsx")
        df_logs.to_excel(excel_path, index=False)
        if verbose:
            print(f"{colors['green']}📁 交易明細已成功匯出至：{excel_path}{colors['reset']}")

    if export_chart:
        if price_df is None or chart_context is None:
            raise ValueError("export_chart=True 時，必須提供 price_df 與 chart_context。")
        chart_path = export_debug_chart_html(
            price_df,
            ticker=ticker,
            output_dir=output_dir,
            chart_context=chart_context,
        )
        if verbose:
            print(f"{colors['green']}📈 K 線交易檢視已成功匯出至：{chart_path}{colors['reset']}")

    if verbose:
        _emit_loss_summary(df_logs, colors)

    return {
        "trade_logs_df": df_logs,
        "excel_path": excel_path,
        "chart_path": chart_path,
    }


def finalize_debug_trade_logs(*, trade_logs, ticker, output_dir, colors, export_excel=True, verbose=True):
    result = finalize_debug_analysis(
        trade_logs=trade_logs,
        ticker=ticker,
        output_dir=output_dir,
        colors=colors,
        export_excel=export_excel,
        export_chart=False,
        verbose=verbose,
    )
    return result["trade_logs_df"]
