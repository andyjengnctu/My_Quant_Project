# core/v16_display.py
import os
from core.v16_config import EV_CALC_METHOD, BUY_SORT_METHOD, SCORE_CALC_METHOD, MIN_FULL_YEAR_RETURN_PCT, MIN_ANNUAL_TRADES, MIN_BUY_FILL_RATE
from core.v16_portfolio_engine import calc_portfolio_score

C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_GRAY = '\033[90m'
C_RESET = '\033[0m'

def get_p(params, key, default=None):
    if isinstance(params, dict): return params.get(key, default)
    return getattr(params, key, default)

def print_scanner_header(params):
    print(f"   ➤ 全域戰略: 買入排序 [{C_YELLOW}{BUY_SORT_METHOD}{C_RESET}] | EV算法 [{C_YELLOW}{EV_CALC_METHOD}{C_RESET}] | 評分模型 [{C_YELLOW}{SCORE_CALC_METHOD}{C_RESET}]")
    print(f"   ➤ 核心風控: 創高 {get_p(params, 'high_len', 201)}日 | ATR {get_p(params, 'atr_len', 14)}日 | 掛單 +{get_p(params, 'atr_buy_tol', 1.5):.1f}倍")
    print(f"   ➤ 停損停利: 初始 -{get_p(params, 'atr_times_init', 2.0):.1f}倍 | 追蹤 -{get_p(params, 'atr_times_trail', 3.5):.1f}倍 | 半倉 {get_p(params, 'tp_percent', 0.5)*100:.0f}%")
    # 🚀 FIX: 對齊 0 預設值
    print(f"   ➤ 歷史濾網: 交易 >= {get_p(params, 'min_history_trades', 0)} 次 | 勝率 >= {get_p(params, 'min_history_win_rate', 0.30)*100:.0f}% | 期望值 >= {get_p(params, 'min_history_ev', 0.0):.2f}R")

def print_strategy_dashboard(params, title, mode_display, max_pos, trades, missed_b, missed_s, final_eq, avg_exp, sys_ret, bm_ret, sys_mdd, bm_mdd, win_rate, payoff, ev, benchmark_ticker="0050", max_exp=None, r_sq=0.0, m_win_rate=0.0, bm_r_sq=0.0, bm_m_win_rate=0.0, normal_trades=None, chase_trades=None, annual_trades=0.0, buy_fill_rate=0.0, annual_return_pct=0.0, bm_annual_return_pct=0.0, min_full_year_return_pct=0.0, bm_min_full_year_return_pct=0.0):
    
    alpha = sys_ret - bm_ret
    annual_alpha = annual_return_pct - bm_annual_return_pct
    mdd_diff = bm_mdd - sys_mdd
    
    sys_ret_str = f"+{sys_ret:.2f}%" if sys_ret > 0 else f"{sys_ret:.2f}%"
    bm_ret_str  = f"+{bm_ret:.2f}%" if bm_ret > 0 else f"{bm_ret:.2f}%"
    alpha_str   = f"+{alpha:.2f}%" if alpha > 0 else f"{alpha:.2f}%"
    sys_ann_ret_str = f"+{annual_return_pct:.2f}%" if annual_return_pct > 0 else f"{annual_return_pct:.2f}%"
    bm_ann_ret_str = f"+{bm_annual_return_pct:.2f}%" if bm_annual_return_pct > 0 else f"{bm_annual_return_pct:.2f}%"
    annual_alpha_str = f"+{annual_alpha:.2f}%" if annual_alpha > 0 else f"{annual_alpha:.2f}%"
    worst_year_alpha = min_full_year_return_pct - bm_min_full_year_return_pct
    sys_worst_year_str = f"+{min_full_year_return_pct:.2f}%" if min_full_year_return_pct > 0 else f"{min_full_year_return_pct:.2f}%"
    bm_worst_year_str = f"+{bm_min_full_year_return_pct:.2f}%" if bm_min_full_year_return_pct > 0 else f"{bm_min_full_year_return_pct:.2f}%"
    worst_year_alpha_str = f"+{worst_year_alpha:.2f}%" if worst_year_alpha > 0 else f"{worst_year_alpha:.2f}%"
    sys_mdd_str = f"-{abs(sys_mdd):.2f}%"
    bm_mdd_str  = f"-{abs(bm_mdd):.2f}%"
    mdd_diff_str = f"少跌 {abs(mdd_diff):.2f}%" if mdd_diff > 0 else f"多跌 {abs(mdd_diff):.2f}%"
    
    sys_romd = (sys_ret / (abs(sys_mdd) + 0.0001)) if sys_mdd != 0 else 0.0
    bm_romd = (bm_ret / (abs(bm_mdd) + 0.0001)) if bm_mdd != 0 else 0.0
    romd_diff = sys_romd - bm_romd
    sys_romd_str = f"{sys_romd:.2f}"
    bm_romd_str = f"{bm_romd:.2f}"
    romd_diff_str = f"+{romd_diff:.2f}" if romd_diff > 0 else f"{romd_diff:.2f}"
    
    final_score = calc_portfolio_score(sys_ret, sys_mdd, m_win_rate, r_sq, annual_return_pct=annual_return_pct)

    rsq_diff = r_sq - bm_r_sq
    mwin_diff = m_win_rate - bm_m_win_rate
    rsq_diff_str = f"+{rsq_diff:.2f}" if rsq_diff > 0 else f"{rsq_diff:.2f}"
    mwin_diff_str = f"+{mwin_diff:.2f}%" if mwin_diff > 0 else f"{mwin_diff:.2f}%"
    sys_rsq_str, bm_rsq_str = f"{r_sq:.2f}", f"{bm_r_sq:.2f}"
    sys_mwin_str, bm_mwin_str = f"{m_win_rate:.2f} %", f"{bm_m_win_rate:.2f} %"

    alpha_color = C_GREEN if alpha > 0 else C_RED
    annual_alpha_color = C_GREEN if annual_alpha > 0 else C_RED
    sys_ret_color = C_GREEN if sys_ret > 0 else C_RED
    mdd_diff_color = C_GREEN if mdd_diff > 0 else C_RED
    romd_diff_color = C_GREEN if romd_diff > 0 else C_RED
    rsq_color = C_GREEN if rsq_diff > 0 else C_RED
    mwin_color = C_GREEN if mwin_diff > 0 else C_RED
    sys_worst_year_color = C_GREEN if min_full_year_return_pct > 0 else C_RED
    worst_year_alpha_color = C_GREEN if worst_year_alpha > 0 else C_RED

    bb_str = f"啟用 (長{get_p(params, 'bb_len', 20)}, 寬{get_p(params, 'bb_mult', 2.0):.1f}x)" if get_p(params, 'use_bb', False) else "關閉"
    kc_str = f"啟用 (長{get_p(params, 'kc_len', 20)}, 寬{get_p(params, 'kc_mult', 2.0):.1f}x)" if get_p(params, 'use_kc', False) else "關閉"
    vol_str = f"啟用 (短{get_p(params, 'vol_short_len', 5)}>長{get_p(params, 'vol_long_len', 19)})" if get_p(params, 'use_vol', False) else "關閉"

    exp_str = f" (最高 {max_exp:>.2f} %)" if max_exp is not None else ""
    normal_trades = trades if normal_trades is None else normal_trades
    chase_trades = 0 if chase_trades is None else chase_trades
    trade_split_str = f"{trades} 筆 (正常:{normal_trades} | 追價:{chase_trades})"

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print(f"🎯 全域戰略: 買入排序 [{C_YELLOW}{BUY_SORT_METHOD}{C_RESET}] | EV算法 [{C_YELLOW}{EV_CALC_METHOD}{C_RESET}] | 評分模型 [{C_YELLOW}{SCORE_CALC_METHOD}{C_RESET}] | 系統得分: {C_CYAN}{final_score:.2f}{C_RESET}")
    print(f"模式: {mode_display} | 最大持股: {max_pos} 檔")
    print(f"總交易次數: {trade_split_str} | 年化交易次數: {annual_trades:.2f} 次/年")
    print(f"錯失次數: 買 {missed_b} | 賣 {missed_s} | 買進成交率: {buy_fill_rate:.2f}% | 最終資產: {final_eq:,.0f} 元")
    print(f"平均資金水位: {avg_exp:.2f} %{exp_str}")
    print(f"--------------------------------------------------------------------------------")
    print(f"| 指標項目         | V16 尊爵系統   | 同期大盤 ({benchmark_ticker:<4}) | 差異 (Alpha)   |")
    print(f"| 總資產報酬率     | {sys_ret_color}{sys_ret_str:<14}{C_RESET} | {bm_ret_str:<15} | {alpha_color}{alpha_str:<14}{C_RESET} |")
    print(f"| 年化報酬率       | {sys_ret_color}{sys_ann_ret_str:<14}{C_RESET} | {bm_ann_ret_str:<15} | {annual_alpha_color}{annual_alpha_str:<14}{C_RESET} |")
    print(f"| 年度最差報酬     | {sys_worst_year_color}{sys_worst_year_str:<14}{C_RESET} | {bm_worst_year_str:<15} | {worst_year_alpha_color}{worst_year_alpha_str:<14}{C_RESET} |")
    print(f"| 最大回撤 (MDD)   | {C_YELLOW}{sys_mdd_str:<14}{C_RESET} | {bm_mdd_str:<15} | {mdd_diff_color}{mdd_diff_str:<12}{C_RESET} |")
    print(f"| 報酬回撤比(RoMD) | {C_CYAN}{sys_romd_str:<14}{C_RESET} | {bm_romd_str:<15} | {romd_diff_color}{romd_diff_str:<14}{C_RESET} |")
    print(f"| 平滑度(Log R²)   | {sys_rsq_str:<14} | {bm_rsq_str:<15} | {rsq_color}{rsq_diff_str:<14}{C_RESET} |")
    print(f"| 月度獲利勝率     | {sys_mwin_str:<14} | {bm_mwin_str:<15} | {mwin_color}{mwin_diff_str:<14}{C_RESET} |")
    print(f"| 系統實戰勝率     | {win_rate:>6.2f} %       | -               | -              |")
    print(f"| 盈虧風報比       | {payoff:>6.2f}         | -               | -              |")
    print(f"| 實戰期望值(EV)   | {ev:>6.2f} R       | -               | -              |")
    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print(f"核心: 突破 {get_p(params, 'high_len', 201):>3} 日新高 | ATR {get_p(params, 'atr_len', 14):>2} 日 | 半倉停利 {get_p(params, 'tp_percent', 0.5)*100:>2.0f}%")
    print(f"風控: 掛單 +{get_p(params, 'atr_buy_tol', 1.5):.1f} ATR | 停損 -{get_p(params, 'atr_times_init', 2.0):.1f} ATR | 追蹤 -{get_p(params, 'atr_times_trail', 3.5):.1f} ATR")
    print(f"濾網: 布林(BB) {bb_str} | 阿肯那(KC) {kc_str} | 均量 {vol_str}")
    # 🚀 FIX: 對齊 0 預設值
    print(f"歷史: 交易 >= {get_p(params, 'min_history_trades', 0)} 次 | 勝率 >= {get_p(params, 'min_history_win_rate', 0.3)*100:.0f}% | EV >= {get_p(params, 'min_history_ev', 0.0):.2f} R")
    print(f"門檻: 完整年度最差報酬 > {MIN_FULL_YEAR_RETURN_PCT:.2f}% | 年化交易次數 >= {MIN_ANNUAL_TRADES:.2f} 次/年 | 買進成交率 >= {MIN_BUY_FILL_RATE:.2f}%")
    print(f"{C_CYAN}================================================================================{C_RESET}\n")