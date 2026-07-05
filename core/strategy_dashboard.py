from config.breakout_policy import BREAKOUT_DEFAULT_HIGH_LEN
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
)
from core.buy_sort import get_buy_sort_title
from core.config import (
    BUY_SORT_METHOD,
    MAX_PORTFOLIO_MDD_PCT,
    MIN_ANNUAL_TRADES,
    MIN_BUY_FILL_RATE,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_FULL_YEAR_RETURN_PCT,
    MIN_MONTHLY_WIN_RATE,
    MIN_TRADE_WIN_RATE,
    SCORE_CALC_METHOD,
    SCORE_NUMERATOR_METHOD,
    format_system_score_for_display,
)
from core.display_common import (
    C_CYAN,
    C_GRAY,
    C_GREEN,
    C_RED,
    C_RESET,
    C_YELLOW,
    _display_width,
    _pad_display,
    _table_row,
    get_p,
)
from core.portfolio_stats import calc_plain_romd, calc_portfolio_score
from core.history_filters import history_threshold_is_enabled


def _format_history_threshold_text(params):
    if not history_threshold_is_enabled(params):
        return "歷史門檻：關閉"
    return (
        f"歷史門檻：交易 >= {get_p(params, 'min_history_trades', 0)} 次｜"
        f"勝率 >= {get_p(params, 'min_history_win_rate', 0.3) * 100:.1f}%｜"
        f"EV >= {get_p(params, 'min_history_ev', 0.0):.2f} R"
    )


def _format_filter_param_text(params):
    bb_str = f"布林(BB) 啟用（長{get_p(params, 'bb_len', 20)}, 寬{get_p(params, 'bb_mult', 2.0):.1f}x）" if get_p(params, 'use_bb', False) else "布林(BB) 關閉"
    kc_str = f"阿肯那(KC) 啟用（長{get_p(params, 'kc_len', 20)}, 寬{get_p(params, 'kc_mult', 2.0):.1f}x）" if get_p(params, 'use_kc', False) else "阿肯那(KC) 關閉"
    vol_str = f"均量 啟用（突破日量 > 前{get_p(params, 'vol_long_len', 20)}日均量 × {get_p(params, 'vol_breakout_mult', 1.5):.1f}）" if get_p(params, 'use_vol', False) else "均量 關閉"
    return_filter_str = f"漲幅 啟用（突破日漲幅 > {get_p(params, 'breakout_return_min', 0.0) * 100:.1f}%）" if get_p(params, 'use_breakout_return_filter', False) else "漲幅 關閉"
    false_filter_str = (
        f"假突破 啟用（ATR%≤{get_p(params, 'breakout_false_filter_atr_pct_min', 0.045) * 100:.1f}%）"
        if get_p(params, 'use_breakout_false_filter', False)
        else "假突破 關閉"
    )
    ema_filter_str = f"EMA濾網 啟用（Close > EMA{get_p(params, 'breakout_ema_len', 240)}）" if get_p(params, 'use_breakout_ema_filter', True) else "EMA濾網 關閉"
    quality_filter_str = (
        f"品質模型 啟用（{get_p(params, 'breakout_quality_filter_id', BREAKOUT_QUALITY_DEFAULT_FILTER_ID)}｜分數≥{get_p(params, 'breakout_quality_score_threshold', BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD):.2f}）"
        if get_p(params, 'use_breakout_quality_filter', False)
        else "品質模型 關閉"
    )
    return bb_str, kc_str, vol_str, return_filter_str, false_filter_str, ema_filter_str, quality_filter_str


def format_training_param_lines(params, entry_trade_counts=None):
    bb_str, kc_str, vol_str, return_filter_str, false_filter_str, ema_filter_str, quality_filter_str = _format_filter_param_text(params)
    breakout_str = (
        f"突破買進 啟用 (突破 {get_p(params, 'high_len', BREAKOUT_DEFAULT_HIGH_LEN)} 日新高)"
        if get_p(params, 'use_breakout_buy', True)
        else "突破買進 關閉"
    )
    reentry_str = (
        f"Re-entry 啟用（{get_p(params, 'breakout_reclaim_window_bars', 20)}日內站回 STOP+{get_p(params, 'breakout_reclaim_confirm_atr', 0.8):.1f}ATR）"
        if get_p(params, 'use_breakout_reclaim_reentry', False)
        else "Re-entry 關閉"
    )
    return [
        f"進場：{breakout_str}｜{reentry_str}",
        f"風控：ATR {get_p(params, 'atr_len', 14)} 日| 掛單 +{get_p(params, 'atr_buy_tol', 1.5):.1f} ATR｜停損 -{get_p(params, 'atr_times_init', 2.0):.1f} ATR｜追蹤 -{get_p(params, 'atr_times_trail', 3.5):.1f} ATR｜半倉停利 {get_p(params, 'tp_percent', 0.5) * 100:.1f}%",
        f"基礎濾網：{bb_str}｜{kc_str}｜{vol_str}",
        f"進階濾網：{return_filter_str}｜{false_filter_str}｜{ema_filter_str}｜{quality_filter_str}",
        _format_history_threshold_text(params),
    ]


def _format_training_param_lines(params):
    return format_training_param_lines(params)


def _record_get(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _format_schedule_row_label(record, *, row_index=None, row_count=None):
    year = _record_get(record, "year") or _record_get(record, "oos_year")
    effective_date = _record_get(record, "effective_date_text") or _record_get(record, "effective_date") or "-"
    effective_end = _record_get(record, "effective_end_date_text") or _record_get(record, "effective_end_date") or ""
    mode = str(_record_get(record, "mode", "") or "").strip().lower()
    row_prefix = ""
    if row_index is not None and row_count is not None and row_count > 1:
        row_prefix = f"fold {int(row_index)}/{int(row_count)} | "
    elif row_index is not None and row_count is not None and mode == "rolling":
        row_prefix = f"fold {int(row_index)}/{int(row_count)} | "
    year_text = str(year).strip() if year is not None else "-"
    if mode == "static":
        if str(effective_date) in {"", "-", "1900-01-01"} and str(effective_end) in {"", "-", "9999-12-31"}:
            return f"{row_prefix}static ensemble"
    range_text = f"{effective_date}~{effective_end}" if effective_end else str(effective_date)
    return f"{row_prefix}OOS {year_text}（effective={range_text}）"


def _resolve_schedule_row_params(record):
    return _record_get(record, "params_obj") or _record_get(record, "params") or record


def _resolve_schedule_row_members(record):
    members = _record_get(record, "members")
    return list(members or [])


def _format_schedule_member_label(member, *, member_index=None, member_count=None):
    seed = _record_get(member, "seed") or _record_get(member, "optimizer_seed")
    member_key = _record_get(member, "member_key") or _record_get(member, "member_index")
    trial = _record_get(member, "selected_trial") or _record_get(member, "trial_number") or _record_get(member, "trial")
    parts = []
    if member_index is not None and member_count is not None:
        parts.append(f"seed {int(member_index)}/{int(member_count)}")
    elif member_key is not None:
        parts.append(f"member {member_key}")
    if seed is not None and str(seed).strip():
        parts.append(f"seed={seed}")
    if trial is not None and str(trial).strip():
        parts.append(f"trial=#{trial}")
    return " | ".join(parts) if parts else "member"


def _resolve_schedule_member_params(member):
    return _record_get(member, "params_obj") or _record_get(member, "params") or member


def _print_prefixed_training_param_lines(lines, *, params_section_title="訓練參數"):
    for idx, line in enumerate(list(lines or [])):
        prefix = f"{C_CYAN}【{params_section_title}】{C_RESET} " if idx == 0 else "　　　　     "
        print(f"{prefix}{line}")


def _print_training_params_section(params, params_schedule_rows=None, *, params_section_title="訓練參數"):
    schedule_rows = list(params_schedule_rows or [])
    if schedule_rows:
        print(f"{C_CYAN}【{params_section_title}】{C_RESET}")
        row_count = len(schedule_rows)
        for row_idx, record in enumerate(schedule_rows, start=1):
            print(f"{C_CYAN}{_format_schedule_row_label(record, row_index=row_idx, row_count=row_count)}{C_RESET}")
            members = _resolve_schedule_row_members(record)
            if members:
                member_count = len(members)
                for member_idx, member in enumerate(members, start=1):
                    row_params = _resolve_schedule_member_params(member)
                    print(f"  {C_YELLOW}{_format_schedule_member_label(member, member_index=member_idx, member_count=member_count)}{C_RESET}")
                    for line in _format_training_param_lines(row_params):
                        print(f"    {line}")
                continue
            row_params = _resolve_schedule_row_params(record)
            for line in _format_training_param_lines(row_params):
                print(f"  {line}")
        return

    _print_prefixed_training_param_lines(
        _format_training_param_lines(params),
        params_section_title=params_section_title,
    )



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
    reentry_trades=None,
    annual_trades=0.0,
    reserved_buy_fill_rate=0.0,
    annual_return_pct=0.0,
    bm_annual_return_pct=0.0,
    min_full_year_return_pct=0.0,
    bm_min_full_year_return_pct=0.0,
    min_month_return_pct=0.0,
    bm_min_month_return_pct=0.0,
    min_quarter_return_pct=0.0,
    bm_min_quarter_return_pct=0.0,
    portfolio_total_r=0.0,
    portfolio_median_r=0.0,
    score_total_r=None,
    score_median_r=None,
    params_section_title="訓練參數",
    params_note_lines=None,
    params_schedule_rows=None,
    comparison_period_text=None,
):
    normal_trades = trades if normal_trades is None else int(normal_trades)
    extended_trades = 0 if extended_trades is None else int(extended_trades)
    reentry_trades = 0 if reentry_trades is None else int(reentry_trades)
    if int(normal_trades) + int(extended_trades) + int(reentry_trades) != int(trades):
        extended_trades = max(int(trades) - int(normal_trades) - int(reentry_trades), 0)

    final_score = calc_portfolio_score(
        sys_ret,
        sys_mdd,
        m_win_rate,
        r_sq,
        annual_return_pct=annual_return_pct,
        trade_win_rate_pct=win_rate,
        min_full_year_return_pct=min_full_year_return_pct,
        min_month_return_pct=min_month_return_pct,
        min_quarter_return_pct=min_quarter_return_pct,
        total_r=portfolio_total_r if score_total_r is None else score_total_r,
        median_r=portfolio_median_r if score_median_r is None else score_median_r,
    )
    initial_capital = float(get_p(params, "initial_capital", 1_000_000.0))
    candidate_metrics = {
        "pf_return": float(sys_ret),
        "annual_return_pct": float(annual_return_pct),
        "min_full_year_return_pct": float(min_full_year_return_pct),
        "min_quarter_return_pct": float(min_quarter_return_pct),
        "min_month_return_pct": float(min_month_return_pct),
        "pf_mdd": float(sys_mdd),
        "r_squared": float(r_sq),
        "m_win_rate": float(m_win_rate),
        "win_rate": float(win_rate),
        "pf_payoff": float(payoff),
        "pf_ev": float(ev),
        "pf_trades": int(trades),
        "normal_trades": int(normal_trades),
        "extended_trades": int(extended_trades),
        "reentry_trades": int(reentry_trades),
        "missed_total": int(missed_b) + int(missed_s),
        "missed_buys": int(missed_b),
        "missed_sells": int(missed_s),
        "annual_trades": float(annual_trades),
        "reserved_buy_fill_rate": float(reserved_buy_fill_rate),
        "avg_exposure": float(avg_exp),
        "final_equity": float(final_eq),
    }
    benchmark_metrics = {
        "pf_return": float(bm_ret),
        "annual_return_pct": float(bm_annual_return_pct),
        "min_full_year_return_pct": float(bm_min_full_year_return_pct),
        "min_quarter_return_pct": float(bm_min_quarter_return_pct),
        "min_month_return_pct": float(bm_min_month_return_pct),
        "pf_mdd": float(bm_mdd),
        "r_squared": float(bm_r_sq),
        "m_win_rate": float(bm_m_win_rate),
        "final_equity": initial_capital * (1.0 + float(bm_ret) / 100.0),
    }
    rows = build_optimizer_dashboard_metric_rows(
        candidate_metrics=candidate_metrics,
        reference_metrics=None,
        benchmark_metrics=benchmark_metrics,
    )

    print(f"{C_GRAY}------------------------------------------------------------------------------------------------------------------------{C_RESET}")
    if title:
        print(f"{C_CYAN}{title}{C_RESET}")
    print(
        f"{C_CYAN}【全域戰略】{C_RESET} {C_YELLOW}{format_global_strategy_text()}{C_RESET} | "
        f"模式：{mode_display} | 最大持股：{max_pos} 檔"
    )
    print(
        f"{C_CYAN}【評分模式】{C_RESET} 評分模型：[{C_YELLOW}{SCORE_CALC_METHOD}{C_RESET}] | "
        f"評分分子：[{C_YELLOW}{SCORE_NUMERATOR_METHOD}{C_RESET}] | "
        f"系統得分：{C_CYAN}{format_system_score_for_display(final_score, decimals=2)}{C_RESET}"
    )
    print(f"{C_GRAY}------------------------------------------------------------------------------------------------------------------------{C_RESET}")
    if comparison_period_text:
        print(f"【回測期間績效對比｜{_normalize_comparison_period_text(comparison_period_text)}】")
        print(f"{C_GRAY}------------------------------------------------------------------------------------------------------------------------{C_RESET}")
    print_optimizer_dashboard_metric_table(rows, benchmark_ticker=benchmark_ticker)

    print(f"{C_GRAY}------------------------------------------------------------------------------------------------------------------------{C_RESET}")
    for note_line in (params_note_lines or []):
        print(f"{C_GRAY}{note_line}{C_RESET}")
    _print_training_params_section(
        params,
        params_schedule_rows=params_schedule_rows,
        params_section_title=params_section_title,
    )
    print(f"{C_GRAY}------------------------------------------------------------------------------------------------------------------------{C_RESET}")
    for idx, line in enumerate(format_hard_gate_lines()):
        prefix = f"{C_CYAN}【共用硬門檻】{C_RESET} " if idx == 0 else "　　　　　     "
        print(f"{prefix}{line}")
    print(f"{C_CYAN}========================================================================================================================{C_RESET}\n")

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


def _normalize_comparison_period_text(value) -> str:
    text = str(value or "-").strip()
    if "~" not in text:
        return text
    left, right = text.split("~", 1)
    return f"{left.strip()} ~ {right.strip()}"


def format_global_strategy_text() -> str:
    return f"買入排序 [{get_buy_sort_title(BUY_SORT_METHOD)}]"


def format_hard_gate_lines() -> list[str]:
    return [
        f"交易頻率：年化交易次數 >= {MIN_ANNUAL_TRADES:.2f} 次/年｜保留後買進成交率 >= {MIN_BUY_FILL_RATE:.2f}%｜完整交易勝率 >= {MIN_TRADE_WIN_RATE:.2f}%",
        f"績效風險：完整年度最差報酬 >= {MIN_FULL_YEAR_RETURN_PCT:.2f}%｜最大回撤(MDD) <= {MAX_PORTFOLIO_MDD_PCT:.2f}%  穩定度：月度獲利勝率 >= {MIN_MONTHLY_WIN_RATE:.2f}%｜權益曲線 R² >= {MIN_EQUITY_CURVE_R_SQUARED:.2f}",
    ]


def _colorize(text: str, color: str) -> str:
    if not color:
        return str(text)
    return f"{color}{text}{C_RESET}"


def _delta_color(value: float) -> str:
    value = float(value)
    if value > 0:
        return C_GREEN
    if value < 0:
        return C_RED
    return ""


def _format_metric_pair(left_value: float, right_value: float, *, left_digits: int = 2, right_digits: int = 3, right_unit: str = "R") -> str:
    return f"{float(left_value):.{left_digits}f}: {float(right_value):.{right_digits}f}{right_unit}"


def _format_metric_pair_diff(left_value: float, right_value: float, *, left_digits: int = 2, right_digits: int = 3, right_unit: str = "R") -> str:
    return f"({float(left_value):+.{left_digits}f}: {float(right_value):+.{right_digits}f}{right_unit})"


def _format_split_bucket(total: int, left_label: str, left_value: int, right_label: str, right_value: int, *, third_label: str | None = None, third_value: int | None = None, separator: str = "｜") -> str:
    total = int(total)
    left_value = int(left_value)
    right_value = int(right_value)
    if third_label is not None and third_value is not None:
        third_value = int(third_value)
        if left_value + right_value + third_value != total:
            right_value = max(total - left_value - third_value, 0)
    text = f"{total} ({left_label}: {left_value}{separator}{right_label}: {right_value}"
    if third_label is not None and third_value is not None:
        text += f"{separator}{third_label}: {third_value}"
    return text + ")"


def _first_zone_base_color(metric_name: str, numeric_value: float) -> str:
    if metric_name in {"總資產報酬率", "年化報酬率", "年度最差報酬", "季度最差報酬", "月度最差報酬"}:
        return C_GREEN if float(numeric_value) > 0 else C_RED
    if metric_name == "最大回撤 (MDD)":
        return C_YELLOW if abs(float(numeric_value)) <= float(MAX_PORTFOLIO_MDD_PCT) else C_RED
    return ""


def _compose_first_zone_cell(metric_name: str, base_text: str, numeric_value: float, *, delta_text: str = "", delta_value: float | None = None, use_blue: bool = False, base_color_override: str | None = None) -> str:
    if base_color_override:
        rendered = _colorize(base_text, base_color_override)
    elif use_blue:
        rendered = _colorize(base_text, C_CYAN)
    else:
        rendered = _colorize(base_text, _first_zone_base_color(metric_name, numeric_value))
    if delta_text in {"", "-", None} or delta_value is None:
        return rendered
    return f"{rendered} {_colorize(delta_text, _delta_color(delta_value))}"


def _optimizer_romd_metric_label() -> str:
    return "報酬回撤比 (RoMD)"


def build_optimizer_dashboard_metric_rows(*, candidate_metrics: dict, reference_metrics: dict | None, benchmark_metrics: dict | None = None) -> list[dict]:
    candidate_metrics = dict(candidate_metrics or {})
    reference_metrics = dict(reference_metrics or {})
    benchmark_metrics = dict(benchmark_metrics or {})

    comparable_romd_key = "display_romd"
    candidate_metrics[comparable_romd_key] = calc_plain_romd(candidate_metrics.get("pf_return", 0.0), candidate_metrics.get("pf_mdd", 0.0))
    if reference_metrics:
        reference_metrics[comparable_romd_key] = calc_plain_romd(reference_metrics.get("pf_return", 0.0), reference_metrics.get("pf_mdd", 0.0))
    if benchmark_metrics:
        benchmark_metrics[comparable_romd_key] = calc_plain_romd(benchmark_metrics.get("pf_return", 0.0), benchmark_metrics.get("pf_mdd", 0.0))

    def champ_value(key, default=None):
        return reference_metrics.get(key, default)

    def bench_value(key, default=None):
        return benchmark_metrics.get(key, default)

    rows = []

    def _append_row(name, candidate_text, candidate_numeric, *, reference_text="-", reference_numeric=None, reference_delta_text="", reference_delta_value=None, benchmark_text="-", benchmark_numeric=None, benchmark_delta_text="", benchmark_delta_value=None, use_blue=False, candidate_color_override=None, base_color_override=None):
        if benchmark_numeric is not None:
            benchmark_cell = _compose_first_zone_cell(
                name,
                benchmark_text,
                float(benchmark_numeric),
                use_blue=use_blue,
                base_color_override=base_color_override,
            )
            benchmark_delta_cell = (
                _colorize(benchmark_delta_text, _delta_color(benchmark_delta_value))
                if benchmark_delta_text not in {"", "-", None} and benchmark_delta_value is not None
                else "-"
            )
        else:
            benchmark_cell = str(benchmark_text)
            benchmark_delta_cell = "-"
        row = {
            "name": name,
            "candidate": _compose_first_zone_cell(name, candidate_text, float(candidate_numeric), use_blue=use_blue, base_color_override=base_color_override) if candidate_numeric is not None else str(candidate_text),
            "candidate_precolored": candidate_numeric is not None,
            "reference": _compose_first_zone_cell(name, reference_text, float(reference_numeric), delta_text=reference_delta_text, delta_value=reference_delta_value, use_blue=use_blue, base_color_override=base_color_override) if reference_numeric is not None else str(reference_text),
            "reference_precolored": reference_numeric is not None,
            "benchmark": benchmark_cell,
            "benchmark_precolored": benchmark_numeric is not None,
            "benchmark_delta": benchmark_delta_cell,
            "benchmark_delta_precolored": benchmark_delta_cell != "-",
        }
        if candidate_color_override is not None:
            row["candidate"] = _colorize(str(candidate_text), candidate_color_override)
            row["candidate_precolored"] = True
        rows.append(row)

    def add_row(name, key, *, kind="pct", candidate_unit=""):
        candidate_value = candidate_metrics.get(key, 0.0)
        reference_value_raw = champ_value(key)
        benchmark_value_raw = bench_value(key)
        if benchmark_value_raw is None and key.startswith("bm_"):
            benchmark_value_raw = candidate_metrics.get(key)
        if kind == "pct":
            cand_plain = _format_pct_plain(candidate_value)
            bench_plain = _format_pct_plain(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            reference_plain = _format_pct_plain(reference_value_raw) if reference_value_raw is not None else "-"
            bench_delta_value = float(candidate_value) - float(benchmark_value_raw) if benchmark_value_raw is not None else None
            reference_delta_value = float(candidate_value) - float(reference_value_raw) if reference_value_raw is not None else None
            bench_delta_text = _format_pct_diff(bench_delta_value) if bench_delta_value is not None else ""
            reference_delta_text = _format_pct_diff(reference_delta_value) if reference_delta_value is not None else ""
        elif kind == "mdd":
            cand_plain = _format_mdd_plain(candidate_value)
            bench_plain = _format_mdd_plain(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            reference_plain = _format_mdd_plain(reference_value_raw) if reference_value_raw is not None else "-"
            bench_delta_value = float(benchmark_value_raw) - float(candidate_value) if benchmark_value_raw is not None else None
            reference_delta_value = float(reference_value_raw) - float(candidate_value) if reference_value_raw is not None else None
            bench_delta_text = _format_mdd_diff(candidate_value, benchmark_value_raw) if benchmark_value_raw is not None else ""
            reference_delta_text = _format_mdd_diff(candidate_value, reference_value_raw) if reference_value_raw is not None else ""
        elif kind in {"float2", "float3"}:
            digits = 2 if kind == "float2" else 3
            cand_plain = f"{float(candidate_value):.{digits}f}{candidate_unit}"
            bench_plain = f"{float(benchmark_value_raw):.{digits}f}{candidate_unit}" if benchmark_value_raw is not None else "-"
            reference_plain = f"{float(reference_value_raw):.{digits}f}{candidate_unit}" if reference_value_raw is not None else "-"
            bench_delta_value = float(candidate_value) - float(benchmark_value_raw) if benchmark_value_raw is not None else None
            reference_delta_value = float(candidate_value) - float(reference_value_raw) if reference_value_raw is not None else None
            bench_delta_text = _format_float_diff(bench_delta_value, digits, candidate_unit) if bench_delta_value is not None else ""
            reference_delta_text = _format_float_diff(reference_delta_value, digits, candidate_unit) if reference_delta_value is not None else ""
        elif kind == "count_split":
            cand_plain = _format_split_bucket(
                candidate_metrics.get("pf_trades", 0),
                "正常",
                candidate_metrics.get("normal_trades", 0),
                "延續",
                candidate_metrics.get("extended_trades", 0),
                third_label="重進",
                third_value=candidate_metrics.get("reentry_trades", 0),
            )
            reference_plain = _format_split_bucket(
                reference_metrics.get("pf_trades", 0),
                "正常",
                reference_metrics.get("normal_trades", 0),
                "延續",
                reference_metrics.get("extended_trades", 0),
                third_label="重進",
                third_value=reference_metrics.get("reentry_trades", 0),
            ) if reference_metrics else "-"
            _append_row(name, cand_plain, None, reference_text=reference_plain, reference_numeric=None, benchmark_text="-", benchmark_numeric=None)
            return
        elif kind == "missed_split":
            cand_plain = _format_split_bucket(
                candidate_metrics.get("missed_total", 0),
                "買",
                candidate_metrics.get("missed_buys", 0),
                "賣",
                candidate_metrics.get("missed_sells", 0),
            )
            reference_plain = _format_split_bucket(
                reference_metrics.get("missed_total", 0),
                "買",
                reference_metrics.get("missed_buys", 0),
                "賣",
                reference_metrics.get("missed_sells", 0),
            ) if reference_metrics else "-"
            _append_row(name, cand_plain, None, reference_text=reference_plain, reference_numeric=None, benchmark_text="-", benchmark_numeric=None)
            return
        elif kind == "float2_nodiff":
            cand_plain = f"{float(candidate_value):.2f}{candidate_unit}"
            reference_plain = f"{float(reference_value_raw):.2f}{candidate_unit}" if reference_value_raw is not None else "-"
            _append_row(name, cand_plain, None, reference_text=reference_plain, reference_numeric=None, benchmark_text="-", benchmark_numeric=None)
            return
        elif kind == "money":
            cand_plain = _format_money(candidate_value)
            bench_plain = _format_money(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            reference_plain = _format_money(reference_value_raw) if reference_value_raw is not None else "-"
            bench_delta_value = float(candidate_value) - float(benchmark_value_raw) if benchmark_value_raw is not None else None
            reference_delta_value = float(candidate_value) - float(reference_value_raw) if reference_value_raw is not None else None
            bench_delta_text = _format_money_diff(bench_delta_value) if bench_delta_value is not None else ""
            reference_delta_text = _format_money_diff(reference_delta_value) if reference_delta_value is not None else ""
        else:
            cand_plain = str(candidate_value)
            bench_plain = str(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            reference_plain = str(reference_value_raw) if reference_value_raw is not None else "-"
            bench_delta_value = None
            reference_delta_value = None
            bench_delta_text = ""
            reference_delta_text = ""

        use_blue = name == _optimizer_romd_metric_label()
        base_color_override = C_CYAN if name == "系統實戰勝率" else None
        _append_row(
            name,
            cand_plain,
            candidate_value,
            reference_text=reference_plain,
            reference_numeric=reference_value_raw,
            reference_delta_text=reference_delta_text,
            reference_delta_value=reference_delta_value,
            benchmark_text=bench_plain,
            benchmark_numeric=benchmark_value_raw,
            benchmark_delta_text=bench_delta_text,
            benchmark_delta_value=bench_delta_value,
            use_blue=use_blue,
            base_color_override=base_color_override,
        )

    add_row("總資產報酬率", "pf_return", kind="pct")
    add_row("年化報酬率", "annual_return_pct", kind="pct")
    add_row("年度最差報酬", "min_full_year_return_pct", kind="pct")
    add_row("季度最差報酬", "min_quarter_return_pct", kind="pct")
    add_row("月度最差報酬", "min_month_return_pct", kind="pct")
    add_row(_optimizer_romd_metric_label(), comparable_romd_key, kind="float2")
    add_row("最大回撤 (MDD)", "pf_mdd", kind="mdd")
    add_row("月度獲利勝率", "m_win_rate", kind="pct")
    add_row("系統實戰勝率", "win_rate", kind="pct")

    candidate_payoff = float(candidate_metrics.get("pf_payoff", 0.0))
    candidate_ev = float(candidate_metrics.get("pf_ev", 0.0))
    reference_payoff = champ_value("pf_payoff")
    reference_ev = champ_value("pf_ev")
    benchmark_payoff = bench_value("pf_payoff")
    benchmark_ev = bench_value("pf_ev")

    candidate_combo = _format_metric_pair(candidate_payoff, candidate_ev)
    if reference_payoff is not None and reference_ev is not None:
        reference_payoff = float(reference_payoff)
        reference_ev = float(reference_ev)
        payoff_diff = candidate_payoff - reference_payoff
        ev_diff = candidate_ev - reference_ev
        diff_text = _format_metric_pair_diff(payoff_diff, ev_diff)
        reference_combo = f"{_format_metric_pair(reference_payoff, reference_ev)} {_colorize(diff_text, _delta_color(ev_diff))}"
    else:
        reference_combo = "-"
    if benchmark_payoff is not None and benchmark_ev is not None:
        benchmark_payoff = float(benchmark_payoff)
        benchmark_ev = float(benchmark_ev)
        payoff_diff = candidate_payoff - benchmark_payoff
        ev_diff = candidate_ev - benchmark_ev
        diff_text = _format_metric_pair_diff(payoff_diff, ev_diff)
        benchmark_combo = f"{_format_metric_pair(benchmark_payoff, benchmark_ev)} {_colorize(diff_text, _delta_color(ev_diff))}"
    else:
        benchmark_combo = "-"
    _append_row("風報比: 期望值", candidate_combo, None, reference_text=reference_combo, reference_numeric=None, benchmark_text=benchmark_combo, benchmark_numeric=None)

    add_row("總交易次數", "pf_trades", kind="count_split")
    add_row("錯失交易次數", "missed_total", kind="missed_split")
    add_row("年化交易次數", "annual_trades", kind="float2_nodiff")
    add_row("保留後買進成交率", "reserved_buy_fill_rate", kind="pct")
    add_row("平均資金水位", "avg_exposure", kind="pct")
    add_row("最終資產", "final_equity", kind="money")
    return rows


def print_optimizer_dashboard_metric_table(rows: list[dict], *, benchmark_ticker: str = "0050") -> None:
    header = ("指標項目", "本輪候選", f"同期大盤{benchmark_ticker}", "差異")
    rendered_rows = [header]
    for row in rows:
        rendered_rows.append(
            (
                row["name"],
                _render_optimizer_dashboard_cell(row, "candidate", row["name"]),
                _render_optimizer_dashboard_cell(row, "benchmark", row["name"]),
                _render_optimizer_dashboard_cell(row, "benchmark_delta", row["name"]),
            )
        )
    widths = _build_table4_compact_widths(rendered_rows, min_widths=(20, 36, 14, 14))
    print(_table_row4_compact(*header, *widths))
    for rendered_row in rendered_rows[1:]:
        print(_table_row4_compact(*rendered_row, *widths))


def _table_row5(c1, c2, c3, c4, c5, w1=20, w2=19, w3=24, w4=18, w5=6):
    return (
        f"| {_pad_display(c1, w1)} "
        f"| {_pad_display(c2, w2)} "
        f"| {_pad_display(c3, w3)} "
        f"| {_pad_display(c4, w4)} "
        f"| {_pad_display(c5, w5)} |"
    )


def _table_row4_compact(c1, c2, c3, c4, w1=20, w2=24, w3=14, w4=14):
    return (
        f"| {_pad_display(c1, w1)} "
        f"| {_pad_display(c2, w2)} "
        f"| {_pad_display(c3, w3)} "
        f"| {_pad_display(c4, w4)} |"
    )


def _table_row_cells(cells, widths):
    rendered = [f" {_pad_display(cell, width)} " for cell, width in zip(cells, widths)]
    return "|" + "|".join(rendered) + "|"


def _build_table_widths(rows, *, min_widths):
    widths = [int(v) for v in min_widths]
    for row in rows:
        for idx, cell in enumerate(row):
            if idx >= len(widths):
                widths.append(0)
            widths[idx] = max(widths[idx], _display_width(cell))
    return tuple(widths)


def _build_table4_compact_widths(rows: list[tuple[str, str, str, str]], *, min_widths: tuple[int, int, int, int] = (20, 24, 14, 14)) -> tuple[int, int, int, int]:
    widths = [int(v) for v in min_widths]
    for row in rows:
        for idx, cell in enumerate(row[:4]):
            widths[idx] = max(widths[idx], _display_width(cell))
    return tuple(widths)


def _build_table5_widths(rows: list[tuple[str, str, str, str, str]], *, min_widths: tuple[int, int, int, int, int] = (20, 19, 24, 18, 6)) -> tuple[int, int, int, int, int]:
    widths = [int(v) for v in min_widths]
    for row in rows:
        for idx, cell in enumerate(row[:5]):
            widths[idx] = max(widths[idx], _display_width(cell))
    return tuple(widths)


def _optimizer_dashboard_metric_color(metric_name: str, value: str) -> str:
    metric_name = str(metric_name or "")
    value_text = str(value or "").strip()
    if value_text in {"-", ""}:
        return ""
    if any(token in metric_name for token in ("報酬回撤比", "RoMD")):
        return C_CYAN
    if metric_name in {"總資產報酬率", "年化報酬率", "年度最差報酬", "季度最差報酬", "月度最差報酬"}:
        if value_text.startswith("+"):
            return C_GREEN
        return C_RED
    if "平滑度" in metric_name:
        try:
            numeric = float(value_text.replace('%', '').replace('R', '').replace(',', '').strip())
        except ValueError:
            return ""
        return C_GREEN if numeric >= float(MIN_EQUITY_CURVE_R_SQUARED) else C_RED
    if metric_name == "月度獲利勝率":
        try:
            numeric = float(value_text.replace('%', '').replace('R', '').replace(',', '').strip())
        except ValueError:
            return ""
        return C_GREEN if numeric >= float(MIN_MONTHLY_WIN_RATE) else C_RED
    if "最大回撤" in metric_name or "最大視窗 MDD" in metric_name:
        try:
            numeric = abs(float(value_text.replace('少跌', '').replace('多跌', '').replace('%', '').replace('(', '').replace(')', '').replace('-', '').replace(',', '').strip()))
        except ValueError:
            return C_YELLOW
        return C_YELLOW if numeric <= float(MAX_PORTFOLIO_MDD_PCT) else C_RED
    return ""


def _optimizer_dashboard_status_color(status_text: str) -> str:
    normalized = str(status_text or "").strip().lower()
    if normalized in {"pass", "ok", "true", "升版", "接班", "通過"}:
        return C_GREEN
    if normalized in {"watch", "warn", "warning", "觀察"}:
        return C_YELLOW
    if normalized in {"fail", "false", "不升版", "不接班", "淘汰", "未通過"}:
        return C_RED
    return ""


def _wrap_optimizer_dashboard_cell(text: str, color: str) -> str:
    if not color:
        return str(text)
    return f"{color}{text}{C_RESET}"


def _render_optimizer_dashboard_cell(row: dict, key: str, metric_name: str) -> str:
    text = row.get(key, "-")
    if row.get(f"{key}_precolored"):
        return str(text)
    explicit_color = row.get(f"{key}_color")
    if explicit_color is not None:
        return _wrap_optimizer_dashboard_cell(text, explicit_color)
    return _wrap_optimizer_dashboard_cell(text, _optimizer_dashboard_metric_color(metric_name, text))


# 新版 optimizer console 版型；實際欄位配色規則由 callbacks 依區域語意與 training_policy 門檻預先決定。
# 這裡僅保留 fallback 顏色推導與最外層版面渲染。


def print_optimizer_trial_console_dashboard(*,
    title: str,
    milestone_title: str,
    global_strategy_text: str,
    mode_display: str,
    max_pos: int,
    model_mode: str,
    objective_mode: str,
    score_calc_method: str,
    score_numerator_method: str,
    system_score_display: str,
    training_title: str,
    training_rows: list[dict],
    testing_title: str | None,
    testing_rows: list[dict] | None,
    upgrade_rows: list[dict] | None,
    compare_rows: list[dict] | None,
    params_lines: list[str],
    hard_gate_lines: list[str],
    study_full_breakout_stats: dict | None = None,
    study_full_breakout_stats_title: str | None = None,
):
    training_header = ("指標項目", "本輪候選", "同期大盤0050", "差異")
    training_table_rows = [training_header]
    for row in training_rows:
        training_table_rows.append(
            (
                row["name"],
                _render_optimizer_dashboard_cell(row, "candidate", row["name"]),
                _render_optimizer_dashboard_cell(row, "benchmark", row["name"]),
                _render_optimizer_dashboard_cell(row, "benchmark_delta", row["name"]),
            )
        )
    if testing_title and testing_rows:
        for row in testing_rows:
            training_table_rows.append(
                (
                    row["name"],
                    _render_optimizer_dashboard_cell(row, "candidate", row["name"]),
                    _render_optimizer_dashboard_cell(row, "benchmark", row["name"]),
                    _render_optimizer_dashboard_cell(row, "benchmark_delta", row["name"]),
                )
            )
    training_widths = _build_table4_compact_widths(training_table_rows, min_widths=(20, 36, 14, 14))
    training_header_line = _table_row4_compact(*training_header, *training_widths)

    upgrade_header = ("升版判斷項目", "本輪候選", "門檻 / 基準", "狀態")
    upgrade_widths = None
    upgrade_header_line = ""
    upgrade_render_rows = []
    if upgrade_rows:
        upgrade_render_rows = [upgrade_header]
        for row in upgrade_rows:
            upgrade_render_rows.append(
                (
                    row["name"],
                    _render_optimizer_dashboard_cell(row, "candidate", row["name"]),
                    row["threshold"],
                    _wrap_optimizer_dashboard_cell(row["status"], _optimizer_dashboard_status_color(row["status"])),
                )
            )
        upgrade_widths = _build_table4_compact_widths(upgrade_render_rows, min_widths=(20, 19, 24, 8))
        upgrade_header_line = _table_row4_compact(*upgrade_header, *upgrade_widths)

    compare_header = ("比較判斷項目", "本輪候選", "run_best (差異)", "門檻 / 基準", "狀態")
    compare_widths = None
    compare_header_line = ""
    compare_render_rows = []
    if compare_rows:
        compare_render_rows = [compare_header]
        for row in compare_rows:
            compare_render_rows.append(
                (
                    row["name"],
                    _render_optimizer_dashboard_cell(row, "candidate", row["name"]),
                    _render_optimizer_dashboard_cell(row, "reference", row["name"]),
                    row["threshold"],
                    _wrap_optimizer_dashboard_cell(row["status"], _optimizer_dashboard_status_color(row["status"])),
                )
            )
        compare_widths = _build_table5_widths(compare_render_rows)
        compare_header_line = _table_row5(*compare_header, *compare_widths)

    study_full_stats_header = ("交易數", "勝率", "風報比", "平均 R", "R 中位數", "總 R")
    study_full_stats_rows = []
    study_full_stats_header_line = ""
    study_full_stats_widths = None
    if study_full_breakout_stats:
        study_full_stats_rows = [
            study_full_stats_header,
            (
                str(study_full_breakout_stats.get("trade_count", "-")),
                f'{C_CYAN}{study_full_breakout_stats.get("win_rate", "-")}{C_RESET}',
                str(study_full_breakout_stats.get("payoff", "-")),
                str(study_full_breakout_stats.get("avg_r", "-")),
                str(study_full_breakout_stats.get("median_r", "-")),
                str(study_full_breakout_stats.get("total_r", "-")),
            ),
        ]
        study_full_stats_widths = _build_table_widths(study_full_stats_rows, min_widths=(10, 10, 10, 12, 12, 14))
        study_full_stats_header_line = _table_row_cells(study_full_stats_header, study_full_stats_widths)


    separator_width = max(
        120,
        _display_width(training_header_line),
        _display_width(upgrade_header_line) if upgrade_header_line else 0,
        _display_width(compare_header_line) if compare_header_line else 0,
        _display_width(study_full_stats_header_line) if study_full_stats_header_line else 0,
    )
    separator = "-" * separator_width
    print(f"{C_GRAY}{separator}{C_RESET}")
    print(f"{C_RED}{milestone_title}{C_RESET}")
    print(
        f"{C_CYAN}【全域戰略】{C_RESET} {C_YELLOW}{global_strategy_text}{C_RESET} | "
        f"模式：{mode_display} | 最大持股：{max_pos} 檔 | model_mode：{C_YELLOW}{model_mode.upper()}{C_RESET}"
    )
    score_numerator_display = score_numerator_method
    print(
        f"{C_CYAN}【評分模式】{C_RESET} 評分模型：[{C_YELLOW}{score_calc_method}{C_RESET}] | "
        f"評分分子：[{C_YELLOW}{score_numerator_display}{C_RESET}] | "
        f"系統得分：{C_CYAN}{system_score_display}{C_RESET}"
    )
    if study_full_stats_rows and study_full_stats_widths is not None:
        print(separator)
        stats_title = str(study_full_breakout_stats_title or "【Study-Full 單股突破統計】")
        print(f"{C_CYAN}{stats_title}{C_RESET}")
        print(study_full_stats_header_line)
        print(_table_row_cells(study_full_stats_rows[1], study_full_stats_widths))
    print(separator)
    print(training_title)
    print(separator)
    print(training_header_line)
    for rendered_row in training_table_rows[1:len(training_rows) + 1]:
        print(_table_row4_compact(*rendered_row, *training_widths))
    if testing_title and testing_rows:
        print(separator)
        print(testing_title)
        print(separator)
        print(training_header_line)
        testing_render_rows = training_table_rows[len(training_rows) + 1:]
        for rendered_row in testing_render_rows:
            print(_table_row4_compact(*rendered_row, *training_widths))
    if upgrade_rows and upgrade_widths is not None:
        print(separator)
        print(upgrade_header_line)
        for rendered_row in upgrade_render_rows[1:]:
            print(_table_row4_compact(*rendered_row, *upgrade_widths))
    if compare_rows and compare_widths is not None:
        print(separator)
        print(compare_header_line)
        for rendered_row in compare_render_rows[1:]:
            print(_table_row5(*rendered_row, *compare_widths))
    print(separator)
    for idx, line in enumerate(params_lines):
        prefix = f"{C_CYAN}【訓練參數】{C_RESET} " if idx == 0 else "　　　　     "
        print(f"{prefix}{line}")
    for idx, line in enumerate(hard_gate_lines):
        prefix = f"{C_CYAN}【共用硬門檻】{C_RESET} " if idx == 0 else "　　　　　     "
        print(f"{prefix}{line}")
    print(f"{C_CYAN}========================================================================================================================{C_RESET}\n")
