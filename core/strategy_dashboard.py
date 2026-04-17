from core.buy_sort import get_buy_sort_title
from core.config import (
    BUY_SORT_METHOD,
    EV_CALC_METHOD,
    MAX_PORTFOLIO_MDD_PCT,
    MIN_ANNUAL_TRADES,
    MIN_BUY_FILL_RATE,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_FULL_YEAR_RETURN_PCT,
    MIN_MONTHLY_WIN_RATE,
    MIN_TRADE_WIN_RATE,
    SCORE_CALC_METHOD,
    SCORE_NUMERATOR_METHOD,
    SYSTEM_SCORE_DISPLAY_MULTIPLIER,
)
from core.display_common import (
    C_CYAN,
    C_GRAY,
    C_GREEN,
    C_RED,
    C_RESET,
    C_YELLOW,
    _pad_display,
    _table_row,
    get_p,
)
from core.portfolio_stats import calc_portfolio_score


def print_strategy_dashboard(
    params,
    title,
    mode_display,
    max_pos,
    trades,
    missed_b,
    missed_s,
    final_eq,
    avg_exp,
    sys_ret,
    bm_ret,
    sys_mdd,
    bm_mdd,
    win_rate,
    payoff,
    ev,
    benchmark_ticker="0050",
    max_exp=None,
    r_sq=0.0,
    m_win_rate=0.0,
    bm_r_sq=0.0,
    bm_m_win_rate=0.0,
    normal_trades=None,
    extended_trades=None,
    annual_trades=0.0,
    reserved_buy_fill_rate=0.0,
    annual_return_pct=0.0,
    bm_annual_return_pct=0.0,
    min_full_year_return_pct=0.0,
    bm_min_full_year_return_pct=0.0,
):
    alpha = sys_ret - bm_ret
    annual_alpha = annual_return_pct - bm_annual_return_pct
    mdd_diff = bm_mdd - sys_mdd

    sys_ret_str = f"+{sys_ret:.2f}%" if sys_ret > 0 else f"{sys_ret:.2f}%"
    bm_ret_str = f"+{bm_ret:.2f}%" if bm_ret > 0 else f"{bm_ret:.2f}%"
    alpha_str = f"+{alpha:.2f}%" if alpha > 0 else f"{alpha:.2f}%"

    sys_ann_ret_str = f"+{annual_return_pct:.2f}%" if annual_return_pct > 0 else f"{annual_return_pct:.2f}%"
    bm_ann_ret_str = f"+{bm_annual_return_pct:.2f}%" if bm_annual_return_pct > 0 else f"{bm_annual_return_pct:.2f}%"
    annual_alpha_str = f"+{annual_alpha:.2f}%" if annual_alpha > 0 else f"{annual_alpha:.2f}%"

    worst_year_alpha = min_full_year_return_pct - bm_min_full_year_return_pct
    sys_worst_year_str = f"+{min_full_year_return_pct:.2f}%" if min_full_year_return_pct > 0 else f"{min_full_year_return_pct:.2f}%"
    bm_worst_year_str = f"+{bm_min_full_year_return_pct:.2f}%" if bm_min_full_year_return_pct > 0 else f"{bm_min_full_year_return_pct:.2f}%"
    worst_year_alpha_str = f"+{worst_year_alpha:.2f}%" if worst_year_alpha > 0 else f"{worst_year_alpha:.2f}%"

    sys_mdd_str = f"-{abs(sys_mdd):.2f}%"
    bm_mdd_str = f"-{abs(bm_mdd):.2f}%"
    mdd_diff_str = f"少跌 {abs(mdd_diff):.2f}%" if mdd_diff > 0 else f"多跌 {abs(mdd_diff):.2f}%"

    sys_romd = (sys_ret / (abs(sys_mdd) + 0.0001)) if sys_mdd != 0 else 0.0
    bm_romd = (bm_ret / (abs(bm_mdd) + 0.0001)) if bm_mdd != 0 else 0.0
    romd_diff = sys_romd - bm_romd
    sys_romd_str = f"{sys_romd:.2f}"
    bm_romd_str = f"{bm_romd:.2f}"
    romd_diff_str = f"+{romd_diff:.2f}" if romd_diff > 0 else f"{romd_diff:.2f}"

    final_score = calc_portfolio_score(
        sys_ret,
        sys_mdd,
        m_win_rate,
        r_sq,
        annual_return_pct=annual_return_pct,
    )

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

    exp_str = f" (最高 {max_exp:.2f} %)" if max_exp is not None else ""
    normal_trades = trades if normal_trades is None else normal_trades
    extended_trades = 0 if extended_trades is None else extended_trades
    trade_split_str = f"{trades} 筆 (正常:{normal_trades} | 延續:{extended_trades})"

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    if title:
        print(f"{C_CYAN}{title}{C_RESET}")
    print(
        f"🎯 全域戰略: 買入排序 [{C_YELLOW}{get_buy_sort_title(BUY_SORT_METHOD)}{C_RESET}] | "
        f"EV算法 [{C_YELLOW}{EV_CALC_METHOD}{C_RESET}] | "
        f"評分模型 [{C_YELLOW}{SCORE_CALC_METHOD}{C_RESET}] | "
        f"評分分子 [{C_YELLOW}{SCORE_NUMERATOR_METHOD}{C_RESET}] | "
        f"系統得分: {C_CYAN}{final_score * SYSTEM_SCORE_DISPLAY_MULTIPLIER:.2f}{C_RESET}"
    )
    print(f"模式: {mode_display} | 最大持股: {max_pos} 檔")
    print(f"總交易次數: {trade_split_str} | 年化交易次數: {annual_trades:.2f} 次/年")
    print(f"錯失次數: 買 {missed_b} | 賣 {missed_s} | 保留後買進成交率: {reserved_buy_fill_rate:.2f}% | 最終資產: {final_eq:,.0f} 元")
    print(f"平均資金水位: {avg_exp:.2f} %{exp_str}")

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print(_table_row("指標項目", "V16 尊爵系統", f"同期大盤 ({benchmark_ticker})", "差異 (Alpha)"))
    print(_table_row("總資產報酬率", f"{sys_ret_color}{sys_ret_str}{C_RESET}", bm_ret_str, f"{alpha_color}{alpha_str}{C_RESET}"))
    print(_table_row("年化報酬率", f"{sys_ret_color}{sys_ann_ret_str}{C_RESET}", bm_ann_ret_str, f"{annual_alpha_color}{annual_alpha_str}{C_RESET}"))
    print(_table_row("年度最差報酬", f"{sys_worst_year_color}{sys_worst_year_str}{C_RESET}", bm_worst_year_str, f"{worst_year_alpha_color}{worst_year_alpha_str}{C_RESET}"))
    print(_table_row("最大回撤 (MDD)", f"{C_YELLOW}{sys_mdd_str}{C_RESET}", bm_mdd_str, f"{mdd_diff_color}{mdd_diff_str}{C_RESET}"))
    print(_table_row("報酬回撤比(RoMD)", f"{C_CYAN}{sys_romd_str}{C_RESET}", bm_romd_str, f"{romd_diff_color}{romd_diff_str}{C_RESET}"))
    print(_table_row("平滑度(Log R²)", sys_rsq_str, bm_rsq_str, f"{rsq_color}{rsq_diff_str}{C_RESET}"))
    print(_table_row("月度獲利勝率", sys_mwin_str, bm_mwin_str, f"{mwin_color}{mwin_diff_str}{C_RESET}"))
    print(_table_row("系統實戰勝率", f"{win_rate:.2f} %", "-", "-"))
    print(_table_row("盈虧風報比", f"{payoff:.2f}", "-", "-"))
    print(_table_row("實戰期望值(EV)", f"{ev:.2f} R", "-", "-"))

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print("【訓練參數】")
    print(
        f"核心參數 : "
        f"突破 {get_p(params, 'high_len', 201):>3} 日新高 | "
        f"ATR {get_p(params, 'atr_len', 14):>2} 日 | "
        f"半倉停利 {get_p(params, 'tp_percent', 0.5) * 100:>5.1f}%"
    )
    print(
        f"風控參數 : "
        f"掛單 +{get_p(params, 'atr_buy_tol', 1.5):>4.1f} ATR | "
        f"停損 -{get_p(params, 'atr_times_init', 2.0):>4.1f} ATR | "
        f"追蹤 -{get_p(params, 'atr_times_trail', 3.5):>4.1f} ATR"
    )
    print(
        f"濾網參數 : "
        f"布林(BB) {bb_str} | "
        f"阿肯那(KC) {kc_str} | "
        f"均量 {vol_str}"
    )
    print(
        f"歷史門檻 : "
        f"交易 >= {get_p(params, 'min_history_trades', 0):>3} 次 | "
        f"勝率 >= {get_p(params, 'min_history_win_rate', 0.3) * 100:>5.1f}% | "
        f"EV >= {get_p(params, 'min_history_ev', 0.0):>5.2f} R"
    )

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print("【共用硬門檻】")
    print(
        f"交易頻率 : "
        f"年化交易次數 >= {MIN_ANNUAL_TRADES:>5.2f} 次/年 | "
        f"保留後買進成交率 >= {MIN_BUY_FILL_RATE:>5.2f}% | "
        f"完整交易勝率 >= {MIN_TRADE_WIN_RATE:>5.2f}%"
    )
    print(
        f"績效風險 : "
        f"完整年度最差報酬 >= {MIN_FULL_YEAR_RETURN_PCT:>6.2f}% | "
        f"最大回撤(MDD) <= {MAX_PORTFOLIO_MDD_PCT:>5.2f}%"
    )
    print(
        f"穩定度   : "
        f"月度獲利勝率 >= {MIN_MONTHLY_WIN_RATE:>5.2f}% | "
        f"權益曲線 R² >= {MIN_EQUITY_CURVE_R_SQUARED:>4.2f}"
    )
    print(f"{C_CYAN}================================================================================{C_RESET}\n")


def _format_pct_plain(value: float) -> str:
    value = float(value)
    return f"+{value:.2f}%" if value > 0 else f"{value:.2f}%"


def _format_pct_diff(value: float) -> str:
    value = float(value)
    return f"(+{value:.2f}%)" if value > 0 else f"({value:.2f}%)"


def _format_float_diff(value: float, digits: int = 2, unit: str = "") -> str:
    value = float(value)
    if unit:
        return f"(+{value:.{digits}f}{unit})" if value > 0 else f"({value:.{digits}f}{unit})"
    return f"(+{value:.{digits}f})" if value > 0 else f"({value:.{digits}f})"


def _format_money(value: float) -> str:
    return f"{float(value):,.0f}"


def _format_money_diff(value: float) -> str:
    value = float(value)
    sign = "+" if value > 0 else ""
    return f"({sign}{value:,.0f})"


def _format_mdd_plain(value: float) -> str:
    return f"-{abs(float(value)):.2f}%"


def _format_mdd_diff(candidate: float, baseline: float) -> str:
    diff = float(baseline) - float(candidate)
    if diff > 0:
        return f"(少跌 {abs(diff):.2f}%)"
    if diff < 0:
        return f"(多跌 {abs(diff):.2f}%)"
    return "(0.00%)"


def _format_value_with_delta(value: str, delta: str) -> str:
    if delta in ("", "-", None):
        return str(value)
    return f"{value} {delta}"


def _table_row5(c1, c2, c3, c4, c5, w1=20, w2=19, w3=24, w4=18, w5=6):
    return (
        f"| {_pad_display(c1, w1)} "
        f"| {_pad_display(c2, w2)} "
        f"| {_pad_display(c3, w3)} "
        f"| {_pad_display(c4, w4)} "
        f"| {_pad_display(c5, w5)} |"
    )


def _table_row4_compact(c1, c2, c3, c4, w1=20, w2=19, w3=24, w4=24):
    return (
        f"| {_pad_display(c1, w1)} "
        f"| {_pad_display(c2, w2)} "
        f"| {_pad_display(c3, w3)} "
        f"| {_pad_display(c4, w4)} |"
    )


def print_optimizer_trial_console_dashboard(*,
    title: str,
    milestone_title: str,
    global_strategy_text: str,
    mode_display: str,
    max_pos: int,
    model_mode: str,
    search_train_range_text: str,
    wf_range_text: str,
    data_end_text: str,
    objective_mode: str,
    score_calc_method: str,
    score_numerator_method: str,
    base_score: float,
    system_score_display: str,
    first_zone_rows: list[dict],
    upgrade_rows: list[dict] | None,
    compare_rows: list[dict] | None,
    params_lines: list[str],
    hard_gate_lines: list[str],
):
    print(f"{C_GRAY}------------------------------------------------------------------------------------------------------------------------{C_RESET}")
    print(f"{C_CYAN}{title}{C_RESET}")
    print(f"{C_RED}{milestone_title}{C_RESET}")
    print(f"全域戰略：{global_strategy_text} | 模式：{mode_display} | 最大持股：{max_pos} 檔 | model_mode：{model_mode.upper()}")
    print(f"【區間設定】 主搜尋訓練：{search_train_range_text} | OOS / WF驗證：{wf_range_text} | 本輪資料終點：{data_end_text}")
    print(f"【評分模式】 objective_mode：{objective_mode} | 評分模型：[{score_calc_method}] | 評分分子：[{score_numerator_method}] | base_score：{float(base_score):.2f} | 系統得分：{system_score_display}")
    print("------------------------------------------------------------------------------------------------------------------------")
    print(_table_row4_compact("指標項目", "本輪候選", "Champion (差異)", "同期大盤 (差異)"))
    for row in first_zone_rows:
        print(_table_row4_compact(row["name"], row["candidate"], row["champion"], row["benchmark"]))
    if upgrade_rows:
        print("------------------------------------------------------------------------------------------------------------------------")
        print(_table_row4_compact("升版判斷項目", "本輪候選", "門檻 / 基準", "狀態", w1=20, w2=19, w3=24, w4=8))
        for row in upgrade_rows:
            print(_table_row4_compact(row["name"], row["candidate"], row["threshold"], row["status"], w1=20, w2=19, w3=24, w4=8))
    if compare_rows:
        print("------------------------------------------------------------------------------------------------------------------------")
        print(_table_row5("接班判斷項目", "本輪候選", "Champion (差異)", "門檻 / 基準", "狀態"))
        for row in compare_rows:
            print(_table_row5(row["name"], row["candidate"], row["champion"], row["threshold"], row["status"]))
    print("------------------------------------------------------------------------------------------------------------------------")
    for idx, line in enumerate(params_lines):
        prefix = "【訓練參數】 " if idx == 0 else "　　　　     "
        print(f"{prefix}{line}")
    for idx, line in enumerate(hard_gate_lines):
        prefix = "【共用硬門檻】 " if idx == 0 else "　　　　　     "
        print(f"{prefix}{line}")
    print(f"{C_CYAN}========================================================================================================================{C_RESET}\n")
