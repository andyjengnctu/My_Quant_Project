import os

import pandas as pd

from tools.debug.charting import build_debug_chart_payload, create_debug_chart_context, export_debug_chart_html


def _build_placeholder_price_df(chart_context=None):
    context = chart_context or {}
    context_dates = pd.DatetimeIndex(pd.to_datetime(context.get("dates", [])))
    placeholder_ts = context_dates[-1] if len(context_dates) > 0 else pd.Timestamp("1970-01-01")
    return pd.DataFrame(
        {
            "Open": [0.0],
            "High": [0.0],
            "Low": [0.0],
            "Close": [0.0],
            "Volume": [0.0],
        },
        index=pd.DatetimeIndex([placeholder_ts]),
    )


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


def _chart_payload_has_bars(chart_payload):
    if chart_payload is None:
        return False
    x_values = chart_payload.get("x") if isinstance(chart_payload, dict) else None
    return x_values is not None and len(x_values) > 0


def finalize_debug_analysis(
    *,
    trade_logs,
    ticker,
    output_dir,
    colors,
    export_excel=True,
    export_chart=True,
    return_chart_payload=False,
    verbose=True,
    price_df=None,
    chart_context=None,
):
    has_trade_logs = bool(trade_logs)
    if has_trade_logs:
        df_logs = pd.DataFrame(trade_logs)
        if '投入總金額' in df_logs.columns:
            df_logs['投入總金額'] = df_logs['投入總金額'].round(0)
    else:
        df_logs = None
        if verbose:
            print(f"{colors['yellow']}⚠️ 這檔股票沒有任何交易紀錄。{colors['reset']}")

    excel_path = None
    chart_path = None
    chart_payload = None
    if export_excel or export_chart or return_chart_payload:
        os.makedirs(output_dir, exist_ok=True)

    if export_excel and df_logs is not None:
        excel_path = os.path.join(output_dir, f"Debug_TradeLog_{ticker}.xlsx")
        df_logs.to_excel(excel_path, index=False)
        if verbose:
            print(f"{colors['green']}📁 交易明細已成功匯出至：{excel_path}{colors['reset']}")

    if export_chart or return_chart_payload:
        if price_df is None:
            raise ValueError("export_chart=True 或 return_chart_payload=True 時，必須提供 price_df。")
        effective_price_df = price_df if len(price_df) > 0 else _build_placeholder_price_df(chart_context)
        effective_chart_context = chart_context if chart_context is not None and len(effective_price_df) == len(price_df) else create_debug_chart_context(effective_price_df)
        chart_payload = build_debug_chart_payload(effective_price_df, effective_chart_context)
        if not _chart_payload_has_bars(chart_payload):
            chart_payload = build_debug_chart_payload(effective_price_df, create_debug_chart_context(effective_price_df))
        if not _chart_payload_has_bars(chart_payload):
            if len(price_df) == 0:
                raise ValueError("chart_payload 建立失敗：price_df 為空且 placeholder payload 無任何 bar。")
            raise ValueError("chart_payload 建立失敗：price_df 非空但 payload 無任何 bar。")

    if export_chart:
        chart_path = export_debug_chart_html(
            price_df,
            ticker=ticker,
            output_dir=output_dir,
            chart_context=chart_context,
            chart_payload=chart_payload,
        )
        if verbose:
            print(f"{colors['green']}📈 K 線交易檢視已成功匯出至：{chart_path}{colors['reset']}")

    if verbose and df_logs is not None and not df_logs.empty:
        _emit_loss_summary(df_logs, colors)

    return {
        "trade_logs_df": df_logs,
        "excel_path": excel_path,
        "chart_path": chart_path,
        "chart_payload": chart_payload,
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
