import sys
import os
import pandas as pd
import numpy as np
import warnings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.v16_params_io import load_params_from_json
from core.v16_core import (
    generate_signals,
    execute_bar_step,
    adjust_long_sell_fill_price,
    calc_entry_price,
    calc_net_sell_price,
    create_signal_tracking_state,
    build_extended_entry_plan_from_signal,
    build_normal_entry_plan,
    execute_pre_market_entry_plan,
    should_clear_extended_signal,
    can_execute_half_take_profit,
)
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows, resolve_unique_csv_path

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)

C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RED = '\033[91m'
C_RESET = '\033[0m'

DATA_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def load_params(json_file=os.path.join(BASE_DIR, "models", "v16_best_params.json")):
    params = load_params_from_json(json_file)
    print(f"{C_GREEN}✅ 成功載入參數大腦: {json_file}{C_RESET}")
    return params


def get_debug_tp_half_price(tp_half, qty, params):
    return tp_half if can_execute_half_take_profit(qty, params.tp_percent) else np.nan


def run_debug_backtest(df, ticker, params, export_excel=True, verbose=True):
    """以正式核心邏輯為準，輸出可讀交易明細的除錯工具"""
    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    Dates = df.index

    ATR_main, buyCondition, sellCondition, buy_limits = generate_signals(df, params)

    position = {'qty': 0}
    active_extended_signal = None
    currentCapital = params.initial_capital
    trade_logs = []

    tradeCount, fullWins = 0, 0
    total_r_multiple, total_r_win, total_r_loss = 0.0, 0.0, 0.0

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j - 1]):
            continue

        pos_start_of_current_bar = position['qty']

        if pos_start_of_current_bar > 0:
            prev_qty = position['qty']
            prev_realized = position.get('realized_pnl', 0.0)
            prev_tp_half = position.get('tp_half', np.nan)

            position, freed_cash, pnl_realized, events = execute_bar_step(
                position,
                ATR_main[j - 1],
                sellCondition[j - 1],
                C[j - 1],
                O[j],
                H[j],
                L[j],
                C[j],
                V[j],
                params
            )

            realized_delta = position.get('realized_pnl', 0.0) - prev_realized
            active_stop_after_update = position.get('sl', np.nan)

            if 'TP_HALF' in events and realized_delta != 0:
                sold_qty = prev_qty - position['qty']
                exec_sell_price_half = adjust_long_sell_fill_price(max(prev_tp_half, O[j]))
                sell_net_price_half = calc_net_sell_price(exec_sell_price_half, sold_qty, params)

                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "半倉停利",
                    "成交價": exec_sell_price_half,
                    "含息成本價": sell_net_price_half,
                    "股數": sold_qty,
                    "投入總金額": sell_net_price_half * sold_qty,
                    "設定停損價": active_stop_after_update,
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": realized_delta
                })

            if 'STOP' in events or 'IND_SELL' in events:
                half_qty = prev_qty - position['qty'] if 'TP_HALF' in events else 0
                final_exit_qty = prev_qty - half_qty
                action_str = "停損殺出" if 'STOP' in events else "指標賣出"

                if 'STOP' in events:
                    sell_price = adjust_long_sell_fill_price(min(active_stop_after_update, O[j]))
                else:
                    sell_price = adjust_long_sell_fill_price(O[j])

                sell_net_price = calc_net_sell_price(sell_price, final_exit_qty, params)
                final_leg_pnl = pnl_realized - realized_delta if 'TP_HALF' in events else pnl_realized

                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": action_str,
                    "成交價": sell_price,
                    "含息成本價": sell_net_price,
                    "股數": final_exit_qty,
                    "投入總金額": sell_net_price * final_exit_qty,
                    "設定停損價": active_stop_after_update,
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": final_leg_pnl
                })
            elif 'MISSED_SELL' in events:
                sell_block_reason = next((event for event in events if event in {'NO_VOLUME', 'LOCKED_DOWN'}), None)
                reason_note = {
                    'NO_VOLUME': '零量，當日無法賣出',
                    'LOCKED_DOWN': '一字跌停鎖死，當日無法賣出',
                }.get(sell_block_reason, '賣出受阻，當日無法賣出')

                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "錯失賣出",
                    "成交價": np.nan,
                    "含息成本價": np.nan,
                    "股數": prev_qty,
                    "投入總金額": np.nan,
                    "設定停損價": active_stop_after_update,
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": 0.0,
                    "備註": reason_note,
                })

            currentCapital += pnl_realized

            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl = position['realized_pnl']
                trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0.0
                tradeCount += 1
                total_r_multiple += trade_r_mult
                if total_pnl > 0:
                    fullWins += 1
                    total_r_win += trade_r_mult
                else:
                    total_r_loss += abs(trade_r_mult)

        isSetup_prev = buyCondition[j - 1] and (pos_start_of_current_bar == 0)
        buyTriggered = False
        sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital

        if isSetup_prev:
            signal_state = create_signal_tracking_state(buy_limits[j - 1], ATR_main[j - 1], params)
            if signal_state is not None:
                active_extended_signal = signal_state

            entry_plan = build_normal_entry_plan(buy_limits[j - 1], ATR_main[j - 1], sizing_cap, params)
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                params=params,
                entry_type='normal',
            )
            if entry_result['filled']:
                position = entry_result['position']
                buyTriggered = True
                active_extended_signal = None

                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "買進",
                    "成交價": entry_result['buy_price'],
                    "含息成本價": entry_result['entry_price'],
                    "股數": entry_plan['qty'],
                    "投入總金額": entry_result['entry_price'] * entry_plan['qty'],
                    "設定停損價": entry_plan['init_sl'],
                    "半倉停利價": get_debug_tp_half_price(entry_result['tp_half'], entry_plan['qty'], params),
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": 0.0,
                    "備註": ""
                })
            elif entry_result['count_as_missed_buy']:
                reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "錯失買進(新訊號)",
                    "成交價": np.nan,
                    "含息成本價": np.nan,
                    "股數": entry_plan['qty'],
                    "投入總金額": reserved_cost,
                    "設定停損價": entry_plan['init_sl'],
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": 0.0,
                    "備註": f"預掛限價 {entry_plan['limit_price']:.2f} 未成交"
                })
            elif entry_result['is_worse_than_initial_stop']:
                reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "放棄進場(先達停損)",
                    "成交價": entry_result['buy_price'],
                    "含息成本價": np.nan,
                    "股數": entry_plan['qty'],
                    "投入總金額": reserved_cost,
                    "設定停損價": entry_plan['init_sl'],
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": 0.0,
                    "備註": "不計 miss buy"
                })

        elif active_extended_signal is not None and pos_start_of_current_bar == 0:
            entry_plan = build_extended_entry_plan_from_signal(active_extended_signal, C[j - 1], sizing_cap, params)
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                params=params,
                entry_type='extended',
            )
            if entry_result['filled']:
                position = entry_result['position']
                buyTriggered = True
                active_extended_signal = None

                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "買進(延續候選)",
                    "成交價": entry_result['buy_price'],
                    "含息成本價": entry_result['entry_price'],
                    "股數": entry_plan['qty'],
                    "投入總金額": entry_result['entry_price'] * entry_plan['qty'],
                    "設定停損價": entry_plan['init_sl'],
                    "半倉停利價": get_debug_tp_half_price(entry_result['tp_half'], entry_plan['qty'], params),
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": 0.0,
                    "備註": ""
                })
            elif entry_result['count_as_missed_buy']:
                reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "錯失買進(延續候選)",
                    "成交價": np.nan,
                    "含息成本價": np.nan,
                    "股數": entry_plan['qty'],
                    "投入總金額": reserved_cost,
                    "設定停損價": entry_plan['init_sl'],
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": 0.0,
                    "備註": f"預掛限價 {entry_plan['limit_price']:.2f} 未成交"
                })
            elif entry_result['is_worse_than_initial_stop']:
                reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
                trade_logs.append({
                    "日期": Dates[j].strftime('%Y-%m-%d'),
                    "動作": "放棄進場(延續先達停損)",
                    "成交價": entry_result['buy_price'],
                    "含息成本價": np.nan,
                    "股數": entry_plan['qty'],
                    "投入總金額": reserved_cost,
                    "設定停損價": entry_plan['init_sl'],
                    "半倉停利價": np.nan,
                    "ATR(前日)": ATR_main[j - 1],
                    "單筆實質損益": 0.0,
                    "備註": "不計 miss buy"
                })

        if not buyTriggered and position['qty'] == 0 and should_clear_extended_signal(active_extended_signal, L[j]):
            active_extended_signal = None

    
    if position['qty'] > 0:
        exec_sell_price = adjust_long_sell_fill_price(C[-1])
        sell_net_price = calc_net_sell_price(exec_sell_price, position['qty'], params)
        final_leg_pnl = (sell_net_price - position['entry']) * position['qty']

        trade_logs.append({
            "日期": Dates[-1].strftime('%Y-%m-%d'),
            "動作": "期末強制結算",
            "成交價": exec_sell_price,
            "含息成本價": sell_net_price,
            "股數": position['qty'],
            "投入總金額": sell_net_price * position['qty'],
            "設定停損價": position.get('sl', np.nan),
            "半倉停利價": np.nan,
            "ATR(前日)": ATR_main[-1] if len(ATR_main) > 0 else np.nan,
            "單筆實質損益": final_leg_pnl
        })

    if not trade_logs:
        if verbose:
            print(f"{C_YELLOW}⚠️ 這檔股票沒有任何交易紀錄。{C_RESET}")
        return None

    df_logs = pd.DataFrame(trade_logs)
    df_logs['投入總金額'] = df_logs['投入總金額'].round(0)

    if export_excel:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_filename = os.path.join(OUTPUT_DIR, f"Debug_TradeLog_{ticker}.xlsx")
        df_logs.to_excel(output_filename, index=False)
        if verbose:
            print(f"{C_GREEN}📁 交易明細已成功匯出至：{output_filename}{C_RESET}")

    if verbose:
        losses = df_logs[df_logs['單筆實質損益'] < 0]
        if not losses.empty:
            print(f"\n{C_CYAN}🚨 [抓漏分析] 前 3 大嚴重虧損明細：{C_RESET}")
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

def main():
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"🛠️ {C_YELLOW}V16 放大鏡：單檔股票交易明細除錯工具{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    ticker = input("\n👉 請輸入要除錯的股票代號 (例如: 00972): ").strip()
    if not ticker: return
    
    try:
        file_path, _duplicate_file_issue_lines = resolve_unique_csv_path(DATA_DIR, ticker)
    except FileNotFoundError:
        # 允許使用者手動上傳的檔案
        if os.path.exists(f"{ticker}.csv"):
            file_path = f"{ticker}.csv"
        else:
            raise FileNotFoundError(f"找不到 {ticker} 的歷史資料 CSV。")
            
    print(f"📥 讀取 {file_path}...")
    raw_df = pd.read_csv(file_path)

    params = load_params()
    min_rows_needed = get_required_min_rows(params)
    df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

    dropped_row_count = sanitize_stats['dropped_row_count']
    invalid_row_count = sanitize_stats['invalid_row_count']
    duplicate_date_count = sanitize_stats['duplicate_date_count']

    if dropped_row_count > 0:
        print(
            f"{C_YELLOW}⚠️ {ticker} 清洗移除 {dropped_row_count} 列 "
            f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count}){C_RESET}"
        )

    print("⏳ 正在產生完整交易明細...")
    run_debug_backtest(df, ticker, params)
    
if __name__ == "__main__":
    main()