# core/v16_display.py
import os
# 🌟 守則 8 對接：向 config 請求全域戰略開關
from core.v16_config import EV_CALC_METHOD, BUY_SORT_METHOD

# 全域顏色定義
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_GRAY = '\033[90m'
C_RESET = '\033[0m'

def get_p(params, key, default=None):
    """安全獲取參數的輔助函數 (相容 dict 與 Object)"""
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)

def print_scanner_header(params):
    """Scanner 專用：印出目前套用的 AI 參數標頭"""
    print(f"   ➤ 全域戰略: 買入排序 [{C_YELLOW}{BUY_SORT_METHOD}{C_RESET}] | EV算法 [{C_YELLOW}{EV_CALC_METHOD}{C_RESET}]")
    print(f"   ➤ 核心風控: 創高 {get_p(params, 'high_len', 201)}日 | ATR {get_p(params, 'atr_len', 14)}日 | 掛單 +{get_p(params, 'atr_buy_tol', 1.5):.1f}倍")
    print(f"   ➤ 停損停利: 初始 -{get_p(params, 'atr_times_init', 2.0):.1f}倍 | 追蹤 -{get_p(params, 'atr_times_trail', 3.5):.1f}倍 | 半倉 {get_p(params, 'tp_percent', 0.5)*100:.0f}%")
    print(f"   ➤ 歷史濾網: 交易 >= {get_p(params, 'min_history_trades', 1)} 次 | 勝率 >= {get_p(params, 'min_history_win_rate', 0.30)*100:.0f}% | 期望值 >= {get_p(params, 'min_history_ev', 0.0):.2f}R")

def print_strategy_dashboard(params, title, mode_display, max_pos, trades, missed_b, missed_s, final_eq, avg_exp, sys_ret, bm_ret, sys_mdd, bm_mdd, win_rate, payoff, ev, benchmark_ticker="0050", max_exp=None):
    """Sim 與 Optimizer 共用：印出統一的豪華戰情面板"""
    alpha = sys_ret - bm_ret
    mdd_diff = bm_mdd - sys_mdd
    
    sys_ret_str = f"+{sys_ret:.2f}%" if sys_ret > 0 else f"{sys_ret:.2f}%"
    bm_ret_str  = f"+{bm_ret:.2f}%" if bm_ret > 0 else f"{bm_ret:.2f}%"
    alpha_str   = f"+{alpha:.2f}%" if alpha > 0 else f"{alpha:.2f}%"
    
    # 使用 abs() 確保負號不會因為數學運算變成 --33% 的奇怪顯示
    sys_mdd_str = f"-{abs(sys_mdd):.2f}%"
    bm_mdd_str  = f"-{abs(bm_mdd):.2f}%"
    mdd_diff_str = f"少跌 {abs(mdd_diff):.2f}%" if mdd_diff > 0 else f"多跌 {abs(mdd_diff):.2f}%"
    
    alpha_color = C_GREEN if alpha > 0 else C_RED
    sys_ret_color = C_GREEN if sys_ret > 0 else C_RED
    mdd_diff_color = C_GREEN if mdd_diff > 0 else C_RED
    
    # 避免除以零，並計算系統最終評分 (RoMD)
    sys_romd = (sys_ret / (abs(sys_mdd) + 0.0001)) if sys_mdd != 0 else 0.0
    bm_romd = (bm_ret / (abs(bm_mdd) + 0.0001)) if bm_mdd != 0 else 0.0
    romd_diff = sys_romd - bm_romd

    sys_romd_str = f"{sys_romd:.2f}"
    bm_romd_str = f"{bm_romd:.2f}"
    romd_diff_str = f"+{romd_diff:.2f}" if romd_diff > 0 else f"{romd_diff:.2f}"
    romd_diff_color = C_GREEN if romd_diff > 0 else C_RED

    bb_str = f"啟用 (長{get_p(params, 'bb_len', 20)}, 寬{get_p(params, 'bb_mult', 2.0):.1f}x)" if get_p(params, 'use_bb', False) else "關閉"
    kc_str = f"啟用 (長{get_p(params, 'kc_len', 20)}, 寬{get_p(params, 'kc_mult', 2.0):.1f}x)" if get_p(params, 'use_kc', False) else "關閉"
    vol_str = f"啟用 (短{get_p(params, 'vol_short_len', 5)}>長{get_p(params, 'vol_long_len', 19)})" if get_p(params, 'use_vol', False) else "關閉"

    exp_str = f" (最高 {max_exp:>.2f} %)" if max_exp is not None else ""

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print(f"🎯 全域戰略: 買入排序 [{C_YELLOW}{BUY_SORT_METHOD}{C_RESET}] | EV算法 [{C_YELLOW}{EV_CALC_METHOD}{C_RESET}] | 系統評分(RoMD): {C_CYAN}{sys_romd:.2f}{C_RESET}")
    print(f"模式: {mode_display} | 最大持股: {max_pos} 檔")
    print(f"總交易紀錄: {trades} 筆 (錯失: 買 {missed_b} | 賣 {missed_s}) | 最終資產: {final_eq:,.0f} 元")
    print(f"平均資金水位: {avg_exp:.2f} %{exp_str}")
    
    # 🌟 像素級強迫症專區：嚴格對齊所有框線與內容 (中文字元視覺寬度扣除)
    print(f"--------------------------------------------------------------------------------")
    print(f"| 指標項目         | V16 尊爵系統   | 同期大盤 ({benchmark_ticker:<4}) | 差異 (Alpha)   |")
    print(f"|------------------|----------------|-----------------|----------------|")
    print(f"| 總資產報酬率     | {sys_ret_color}{sys_ret_str:<14}{C_RESET} | {bm_ret_str:<15} | {alpha_color}{alpha_str:<14}{C_RESET} |")
    # 注意這裡的 mdd_diff_str 用了 <12，因為「少跌」或「多跌」兩個字佔了額外 2 格視覺寬度
    print(f"| 最大回撤 (MDD)   | {C_YELLOW}{sys_mdd_str:<14}{C_RESET} | {bm_mdd_str:<15} | {mdd_diff_color}{mdd_diff_str:<12}{C_RESET} |")
    print(f"| 報酬回撤比(RoMD) | {C_CYAN}{sys_romd_str:<14}{C_RESET} | {bm_romd_str:<15} | {romd_diff_color}{romd_diff_str:<14}{C_RESET} |")
    print(f"| 系統實戰勝率     | {win_rate:>6.2f} %       | -               | -              |")
    print(f"| 盈虧風報比       | {payoff:>6.2f}         | -               | -              |")
    print(f"| 實戰期望值(EV)   | {ev:>6.2f} R       | -               | -              |")
    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    
    print(f"核心: 突破 {get_p(params, 'high_len', 201):>3} 日新高 | ATR {get_p(params, 'atr_len', 14):>2} 日 | 半倉停利 {get_p(params, 'tp_percent', 0.5)*100:>2.0f}%")
    print(f"風控: 掛單 +{get_p(params, 'atr_buy_tol', 1.5):.1f} ATR | 停損 -{get_p(params, 'atr_times_init', 2.0):.1f} ATR | 追蹤 -{get_p(params, 'atr_times_trail', 3.5):.1f} ATR")
    print(f"濾網: 布林(BB) {bb_str}")
    print(f"      阿肯那(KC) {kc_str}")
    print(f"      均量 {vol_str}")
    print(f"歷史: 交易 >= {get_p(params, 'min_history_trades', 1)} 次 | 勝率 >= {get_p(params, 'min_history_win_rate', 0.3)*100:.0f}% | EV >= {get_p(params, 'min_history_ev', 0.0):.2f} R")
    print(f"{C_CYAN}================================================================================{C_RESET}\n")