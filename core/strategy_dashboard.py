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
    _table_row,
    get_p,
)
from core.portfolio_stats import calc_portfolio_score


def _format_signed_pct(value):
    value = float(value)
    return f"+{value:.2f}%" if value > 0 else f"{value:.2f}%"


def _format_signed_value(value):
    value = float(value)
    return f"+{value:.2f}" if value > 0 else f"{value:.2f}"


def build_strategy_dashboard_sections(
    params,
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
    alpha = float(sys_ret) - float(bm_ret)
    annual_alpha = float(annual_return_pct) - float(bm_annual_return_pct)
    mdd_diff = float(bm_mdd) - float(sys_mdd)
    worst_year_alpha = float(min_full_year_return_pct) - float(bm_min_full_year_return_pct)

    sys_ret_str = _format_signed_pct(sys_ret)
    bm_ret_str = _format_signed_pct(bm_ret)
    alpha_str = _format_signed_pct(alpha)
    sys_ann_ret_str = _format_signed_pct(annual_return_pct)
    bm_ann_ret_str = _format_signed_pct(bm_annual_return_pct)
    annual_alpha_str = _format_signed_pct(annual_alpha)
    sys_worst_year_str = _format_signed_pct(min_full_year_return_pct)
    bm_worst_year_str = _format_signed_pct(bm_min_full_year_return_pct)
    worst_year_alpha_str = _format_signed_pct(worst_year_alpha)
    sys_mdd_str = f"-{abs(float(sys_mdd)):.2f}%"
    bm_mdd_str = f"-{abs(float(bm_mdd)):.2f}%"
    mdd_diff_str = f"少跌 {abs(mdd_diff):.2f}%" if mdd_diff > 0 else f"多跌 {abs(mdd_diff):.2f}%"

    sys_romd = (float(sys_ret) / (abs(float(sys_mdd)) + 0.0001)) if float(sys_mdd) != 0 else 0.0
    bm_romd = (float(bm_ret) / (abs(float(bm_mdd)) + 0.0001)) if float(bm_mdd) != 0 else 0.0
    romd_diff = sys_romd - bm_romd
    sys_romd_str = f"{sys_romd:.2f}"
    bm_romd_str = f"{bm_romd:.2f}"
    romd_diff_str = _format_signed_value(romd_diff)

    final_score = calc_portfolio_score(
        sys_ret,
        sys_mdd,
        m_win_rate,
        r_sq,
        annual_return_pct=annual_return_pct,
    )

    rsq_diff = float(r_sq) - float(bm_r_sq)
    mwin_diff = float(m_win_rate) - float(bm_m_win_rate)
    rsq_diff_str = _format_signed_value(rsq_diff)
    mwin_diff_str = _format_signed_pct(mwin_diff)
    sys_rsq_str, bm_rsq_str = f"{float(r_sq):.2f}", f"{float(bm_r_sq):.2f}"
    sys_mwin_str, bm_mwin_str = f"{float(m_win_rate):.2f} %", f"{float(bm_m_win_rate):.2f} %"

    bb_str = f"啟用 (長{get_p(params, 'bb_len', 20)}, 寬{get_p(params, 'bb_mult', 2.0):.1f}x)" if get_p(params, 'use_bb', False) else "關閉"
    kc_str = f"啟用 (長{get_p(params, 'kc_len', 20)}, 寬{get_p(params, 'kc_mult', 2.0):.1f}x)" if get_p(params, 'use_kc', False) else "關閉"
    vol_str = f"啟用 (短{get_p(params, 'vol_short_len', 5)}>長{get_p(params, 'vol_long_len', 19)})" if get_p(params, 'use_vol', False) else "關閉"

    normal_trades = trades if normal_trades is None else normal_trades
    extended_trades = 0 if extended_trades is None else extended_trades
    trade_split_str = f"{trades} 筆 (正常:{normal_trades} | 延續:{extended_trades})"
    exp_str = f" (最高 {float(max_exp):.2f} %)" if max_exp is not None else ""

    return {
        "global_strategy_line": (
            f"🎯 全域戰略: 買入排序 [{get_buy_sort_title(BUY_SORT_METHOD)}] | "
            f"EV算法 [{EV_CALC_METHOD}] | "
            f"評分模型 [{SCORE_CALC_METHOD}] | "
            f"評分分子 [{SCORE_NUMERATOR_METHOD}] | "
            f"系統得分: {final_score * SYSTEM_SCORE_DISPLAY_MULTIPLIER:.2f}"
        ),
        "overview_lines": [
            f"模式: {mode_display} | 最大持股: {max_pos} 檔",
            f"總交易次數: {trade_split_str} | 年化交易次數: {float(annual_trades):.2f} 次/年",
            f"錯失次數: 買 {missed_b} | 賣 {missed_s} | 保留後買進成交率: {float(reserved_buy_fill_rate):.2f}% | 最終資產: {float(final_eq):,.0f} 元",
            f"平均資金水位: {float(avg_exp):.2f} %{exp_str}",
        ],
        "metric_rows": [
            {"item": "總資產報酬率", "system": sys_ret_str, "benchmark": bm_ret_str, "alpha": alpha_str},
            {"item": "年化報酬率", "system": sys_ann_ret_str, "benchmark": bm_ann_ret_str, "alpha": annual_alpha_str},
            {"item": "年度最差報酬", "system": sys_worst_year_str, "benchmark": bm_worst_year_str, "alpha": worst_year_alpha_str},
            {"item": "最大回撤 (MDD)", "system": sys_mdd_str, "benchmark": bm_mdd_str, "alpha": mdd_diff_str},
            {"item": "報酬回撤比(RoMD)", "system": sys_romd_str, "benchmark": bm_romd_str, "alpha": romd_diff_str},
            {"item": "平滑度(Log R²)", "system": sys_rsq_str, "benchmark": bm_rsq_str, "alpha": rsq_diff_str},
            {"item": "月度獲利勝率", "system": sys_mwin_str, "benchmark": bm_mwin_str, "alpha": mwin_diff_str},
            {"item": "系統實戰勝率", "system": f"{float(win_rate):.2f} %", "benchmark": "-", "alpha": "-"},
            {"item": "盈虧風報比", "system": f"{float(payoff):.2f}", "benchmark": "-", "alpha": "-"},
            {"item": "實戰期望值(EV)", "system": f"{float(ev):.2f} R", "benchmark": "-", "alpha": "-"},
        ],
        "training_rows": [
            {"item": "核心參數", "value": f"突破 {get_p(params, 'high_len', 201):>3} 日新高 | ATR {get_p(params, 'atr_len', 14):>2} 日 | 半倉停利 {get_p(params, 'tp_percent', 0.5) * 100:>5.1f}%"},
            {"item": "風控參數", "value": f"掛單 +{get_p(params, 'atr_buy_tol', 1.5):>4.1f} ATR | 停損 -{get_p(params, 'atr_times_init', 2.0):>4.1f} ATR | 追蹤 -{get_p(params, 'atr_times_trail', 3.5):>4.1f} ATR"},
            {"item": "濾網參數", "value": f"布林(BB) {bb_str} | 阿肯那(KC) {kc_str} | 均量 {vol_str}"},
            {"item": "歷史門檻", "value": f"交易 >= {get_p(params, 'min_history_trades', 0):>3} 次 | 勝率 >= {get_p(params, 'min_history_win_rate', 0.3) * 100:>5.1f}% | EV >= {get_p(params, 'min_history_ev', 0.0):>5.2f} R"},
        ],
        "threshold_rows": [
            {"item": "交易頻率", "value": f"年化交易次數 >= {MIN_ANNUAL_TRADES:>5.2f} 次/年 | 保留後買進成交率 >= {MIN_BUY_FILL_RATE:>5.2f}% | 完整交易勝率 >= {MIN_TRADE_WIN_RATE:>5.2f}%"},
            {"item": "績效風險", "value": f"完整年度最差報酬 >= {MIN_FULL_YEAR_RETURN_PCT:>6.2f}% | 最大回撤(MDD) <= {MAX_PORTFOLIO_MDD_PCT:>5.2f}%"},
            {"item": "穩定度", "value": f"月度獲利勝率 >= {MIN_MONTHLY_WIN_RATE:>5.2f}% | 權益曲線 R² >= {MIN_EQUITY_CURVE_R_SQUARED:>4.2f}"},
        ],
        "benchmark_ticker": str(benchmark_ticker),
    }



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
    sections = build_strategy_dashboard_sections(
        params=params,
        mode_display=mode_display,
        max_pos=max_pos,
        trades=trades,
        missed_b=missed_b,
        missed_s=missed_s,
        final_eq=final_eq,
        avg_exp=avg_exp,
        sys_ret=sys_ret,
        bm_ret=bm_ret,
        sys_mdd=sys_mdd,
        bm_mdd=bm_mdd,
        win_rate=win_rate,
        payoff=payoff,
        ev=ev,
        benchmark_ticker=benchmark_ticker,
        max_exp=max_exp,
        r_sq=r_sq,
        m_win_rate=m_win_rate,
        bm_r_sq=bm_r_sq,
        bm_m_win_rate=bm_m_win_rate,
        normal_trades=normal_trades,
        extended_trades=extended_trades,
        annual_trades=annual_trades,
        reserved_buy_fill_rate=reserved_buy_fill_rate,
        annual_return_pct=annual_return_pct,
        bm_annual_return_pct=bm_annual_return_pct,
        min_full_year_return_pct=min_full_year_return_pct,
        bm_min_full_year_return_pct=bm_min_full_year_return_pct,
    )

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    if title:
        print(f"{C_CYAN}{title}{C_RESET}")
    print(
        sections["global_strategy_line"]
        .replace(f"買入排序 [{get_buy_sort_title(BUY_SORT_METHOD)}]", f"買入排序 [{C_YELLOW}{get_buy_sort_title(BUY_SORT_METHOD)}{C_RESET}]")
        .replace(f"EV算法 [{EV_CALC_METHOD}]", f"EV算法 [{C_YELLOW}{EV_CALC_METHOD}{C_RESET}]")
        .replace(f"評分模型 [{SCORE_CALC_METHOD}]", f"評分模型 [{C_YELLOW}{SCORE_CALC_METHOD}{C_RESET}]")
        .replace(f"評分分子 [{SCORE_NUMERATOR_METHOD}]", f"評分分子 [{C_YELLOW}{SCORE_NUMERATOR_METHOD}{C_RESET}]")
        .replace("系統得分: ", f"系統得分: {C_CYAN}")
        + C_RESET
    )
    for line in sections["overview_lines"]:
        print(line)

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print(_table_row("指標項目", "V16 尊爵系統", f"同期大盤 ({sections['benchmark_ticker']})", "差異 (Alpha)"))
    for row in sections["metric_rows"]:
        print(_table_row(row["item"], row["system"], row["benchmark"], row["alpha"]))

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print("【訓練參數】")
    for row in sections["training_rows"]:
        print(f"{row['item']} : {row['value']}")

    print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
    print("【共用硬門檻】")
    for row in sections["threshold_rows"]:
        print(f"{row['item']} : {row['value']}")
    print(f"{C_CYAN}================================================================================{C_RESET}\n")
