from core.buy_sort import (
    format_buy_sort_metric_value,
    get_buy_sort_metric_label,
    get_buy_sort_title,
)
from core.capital_policy import resolve_scanner_live_capital
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
)
from core.display_common import C_RESET, C_YELLOW, get_p


def _normalize_history_win_rate_pct(win_rate):
    numeric_value = float(win_rate)
    if numeric_value <= 1.0:
        return numeric_value * 100.0
    return numeric_value


def build_scanner_sort_probe_text(
    *,
    ev,
    win_rate,
    trade_count,
    asset_growth_pct,
    sort_value,
    method=BUY_SORT_METHOD,
):
    label = get_buy_sort_metric_label(method)
    sort_metric_text = format_buy_sort_metric_value(sort_value, method)
    return (
        f'EV {float(ev):.2f}R | '
        f'勝率 {_normalize_history_win_rate_pct(win_rate):.2f}% | '
        f'交易 {int(trade_count)} | '
        f'資產成長 {float(asset_growth_pct):.2f}% | '
        f'{label} {sort_metric_text}'
    )


def print_scanner_header(params):
    bb_str = f"啟用 (長{get_p(params, 'bb_len', 20)}, 寬{get_p(params, 'bb_mult', 2.0):.1f}x)" if get_p(params, 'use_bb', False) else "關閉"
    kc_str = f"啟用 (長{get_p(params, 'kc_len', 20)}, 寬{get_p(params, 'kc_mult', 2.0):.1f}x)" if get_p(params, 'use_kc', False) else "關閉"
    vol_str = f"啟用 (短{get_p(params, 'vol_short_len', 5)}>長{get_p(params, 'vol_long_len', 19)})" if get_p(params, 'use_vol', False) else "關閉"

    print(
        f"   ➤ 全域戰略: 買入排序 [{C_YELLOW}{get_buy_sort_title(BUY_SORT_METHOD)}{C_RESET}] | "
        f"EV算法 [{C_YELLOW}{EV_CALC_METHOD}{C_RESET}] | "
        f"評分模型 [{C_YELLOW}{SCORE_CALC_METHOD}{C_RESET}] | "
        f"評分分子 [{C_YELLOW}{SCORE_NUMERATOR_METHOD}{C_RESET}]"
    )
    print(
        f"   ➤ 排序探針: "
        f"逐項輸出應顯示 EV / 勝率 / 交易次數 / 資產成長 / {get_buy_sort_metric_label(BUY_SORT_METHOD)}；"
        f"目前排序欄位 = {get_buy_sort_metric_label(BUY_SORT_METHOD)}"
    )
    print(
        f"   ➤ 訓練參數: "
        f"突破 {get_p(params, 'high_len', 201)}日 | "
        f"ATR {get_p(params, 'atr_len', 14)}日 | "
        f"掛單 +{get_p(params, 'atr_buy_tol', 1.5):.1f}倍 | "
        f"初始 -{get_p(params, 'atr_times_init', 2.0):.1f}倍 | "
        f"追蹤 -{get_p(params, 'atr_times_trail', 3.5):.1f}倍 | "
        f"半倉 {get_p(params, 'tp_percent', 0.5) * 100:.0f}%"
    )
    print(
        f"   ➤ 濾網參數: "
        f"布林(BB) {bb_str} | "
        f"阿肯那(KC) {kc_str} | "
        f"均量 {vol_str}"
    )
    print(
        f"   ➤ 歷史門檻: "
        f"交易 >= {get_p(params, 'min_history_trades', 0)} 次 | "
        f"勝率 >= {get_p(params, 'min_history_win_rate', 0.30) * 100:.0f}% | "
        f"期望值 >= {get_p(params, 'min_history_ev', 0.0):.2f}R"
    )
    print(
        f"   ➤ Scanner資金: "
        f"live capital = {resolve_scanner_live_capital(params):,.0f}"
    )
    print(
        f"   ➤ 共用硬門檻: "
        f"年化交易次數 >= {MIN_ANNUAL_TRADES:.2f} 次/年 | "
        f"保留後買進成交率 >= {MIN_BUY_FILL_RATE:.2f}% | "
        f"完整交易勝率 >= {MIN_TRADE_WIN_RATE:.2f}%"
    )
    print(
        f"   ➤            "
        f"完整年度最差報酬 >= {MIN_FULL_YEAR_RETURN_PCT:.2f}% | "
        f"最大回撤(MDD) <= {MAX_PORTFOLIO_MDD_PCT:.2f}% | "
        f"月度獲利勝率 >= {MIN_MONTHLY_WIN_RATE:.2f}% | "
        f"權益曲線 R² >= {MIN_EQUITY_CURVE_R_SQUARED:.2f}"
    )
