import os
import pandas as pd


def finalize_debug_trade_logs(*, trade_logs, ticker, output_dir, colors, export_excel=True, verbose=True):
    if not trade_logs:
        if verbose:
            print(f"{colors['yellow']}⚠️ 這檔股票沒有任何交易紀錄。{colors['reset']}")
        return None

    df_logs = pd.DataFrame(trade_logs)
    df_logs['投入總金額'] = df_logs['投入總金額'].round(0)

    if export_excel:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"Debug_TradeLog_{ticker}.xlsx")
        df_logs.to_excel(output_filename, index=False)
        if verbose:
            print(f"{colors['green']}📁 交易明細已成功匯出至：{output_filename}{colors['reset']}")

    if verbose:
        losses = df_logs[df_logs['單筆實質損益'] < 0]
        if not losses.empty:
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

    return df_logs
