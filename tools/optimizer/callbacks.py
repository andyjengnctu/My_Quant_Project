import contextlib
import io
import os
import time

import pandas as pd

from core.buy_sort import get_buy_sort_title
from core.config import (
    BUY_SORT_METHOD,
    MAX_PORTFOLIO_MDD_PCT,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_MONTHLY_WIN_RATE,
    MIN_TRADE_WIN_RATE,
    SCORE_CALC_METHOD,
    SCORE_NUMERATOR_METHOD,
    format_system_score_for_display,
)
from core.display_common import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, get_p
from core.walk_forward_policy import filter_search_train_dates
from core.model_paths import resolve_run_best_params_path
from core.params_io import build_params_from_mapping, load_params_from_json, params_to_json_dict
from core.portfolio_param_runtime import load_portfolio_param_source_from_json
from core.active_param_ensemble import get_active_param_ensemble_policy, load_json_file
from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_stats import calc_plain_romd, calc_portfolio_score
from core.runtime_utils import stdout_supports_inline_progress, write_inline_progress
from core.strategy_params import V16StrategyParams, build_runtime_param_raw_value
from core.strategy_dashboard import (
    _format_float_diff,
    _format_mdd_diff,
    _format_mdd_plain,
    _format_money,
    _format_money_diff,
    _format_pct_diff,
    _format_pct_plain,
    _format_value_with_delta,
    format_training_param_lines,
    print_optimizer_trial_console_dashboard,
)
from tools.optimizer.prep import prepare_trial_inputs
from tools.optimizer.study_utils import (
    OBJECTIVE_MODE_SPLIT_TRAIN_ROMD,
    is_qualified_trial_value,
    normalize_objective_mode,
)
from tools.optimizer.walk_forward import build_test_holdout_period, build_test_period_metrics, evaluate_walk_forward


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_model_mode(objective_mode: str) -> str:
    mode = normalize_objective_mode(objective_mode)
    if mode == OBJECTIVE_MODE_SPLIT_TRAIN_ROMD:
        return "oos"
    return "legacy"


def _resolve_session_model_mode(session) -> str:
    policy = dict(getattr(session, "walk_forward_policy", {}) or {})
    mode = str(policy.get("model_mode") or "").strip().lower()
    scope = str(policy.get("evaluation_scope") or "").strip().lower()
    if scope.startswith("study_full"):
        return "trade"
    if mode == "split":
        return "oos"
    if mode == "full":
        return "trade"
    if mode == "study":
        return "oos"
    if mode in {"trade", "oos"}:
        return mode
    if scope.startswith("trade"):
        return "trade"
    if scope.startswith("oos") or scope.startswith("study"):
        return "oos"
    return _resolve_model_mode(getattr(session, "objective_mode", ""))


def _is_study_full_session(session) -> bool:
    policy = dict(getattr(session, "walk_forward_policy", {}) or {})
    return str(policy.get("evaluation_scope") or "").strip().lower().startswith("study_full")



def _format_signed_r(value, *, digits: int = 3, large: bool = False) -> str:
    numeric = _safe_float(value, 0.0)
    if large:
        return f"{numeric:+,.1f} R"
    return f"{numeric:+.{int(digits)}f} R"


def _colorize_signed_number(text: str, value) -> str:
    return f"{C_GREEN if _safe_float(value, 0.0) >= 0.0 else C_RED}{text}{C_RESET}"


def _build_study_full_breakout_stats(attrs: dict) -> dict:
    trade_count = _safe_int(attrs.get("single_stock_trade_count", 0))
    win_rate = _safe_float(attrs.get("single_stock_win_rate", 0.0))
    payoff = _safe_float(attrs.get("single_stock_payoff_r", 0.0))
    avg_r = _safe_float(attrs.get("single_stock_avg_r", 0.0))
    median_r = _safe_float(attrs.get("single_stock_median_r", 0.0))
    total_r = _safe_float(attrs.get("single_stock_total_r", 0.0))
    win_rate_text = f"{win_rate:.2f}%"
    return {
        "trade_count": f"{trade_count:,}",
        "win_rate": f"{C_GREEN if win_rate >= MIN_TRADE_WIN_RATE else C_RED}{win_rate_text}{C_RESET}",
        "payoff": f"{C_CYAN}{payoff:.2f}{C_RESET}",
        "avg_r": _colorize_signed_number(_format_signed_r(avg_r), avg_r),
        "median_r": _colorize_signed_number(_format_signed_r(median_r), median_r),
        "total_r": _colorize_signed_number(_format_signed_r(total_r, large=True), total_r),
    }


def _policy_date(session, key: str) -> str | None:
    policy = dict(getattr(session, "walk_forward_policy", {}) or {})
    text = str(policy.get(key) or "").strip()
    if not text:
        return None
    return pd.Timestamp(text).normalize().strftime("%Y-%m-%d")

def _range_text_from_dates(dates) -> str:
    if not dates:
        return "-"
    start = str(dates[0].date()) if hasattr(dates[0], 'date') else str(dates[0])[:10]
    end = str(dates[-1].date()) if hasattr(dates[-1], 'date') else str(dates[-1])[:10]
    return f"{start} ~ {end}"


def _latest_data_end_text(session) -> str:
    if not session.sorted_master_dates:
        return "-"
    last_date = session.sorted_master_dates[-1]
    return str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]

def _build_trial_params_object(param_mapping: dict) -> V16StrategyParams:
    base_payload = params_to_json_dict(V16StrategyParams())
    base_payload.update(dict(param_mapping))
    return build_params_from_mapping(base_payload)




def _build_oos_metrics_from_report(*, report: dict | None, initial_capital: float) -> tuple[dict, dict, str]:
    total = build_test_period_metrics(report)
    period = dict((report or {}).get("period") or {})
    missed_buys = int(period.get("missed_buys", 0) or 0)
    missed_sells = int(period.get("missed_sells", 0) or 0)
    candidate_metrics = {
        "pf_return": float(total.get("total_return_pct", 0.0)),
        "annual_return_pct": float(total.get("annualized_return_pct", 0.0)),
        "min_full_year_return_pct": float(total.get("min_full_year_return_pct", 0.0)),
        "pf_mdd": float(total.get("max_drawdown_pct", 0.0)),
        "pf_romd": float(total.get("test_score_romd", 0.0)),
        "r_squared": float(total.get("r_squared", 0.0)),
        "m_win_rate": float(total.get("monthly_win_rate", 0.0)),
        "win_rate": float(total.get("win_rate", 0.0)),
        "pf_payoff": float(total.get("payoff", 0.0)),
        "pf_ev": float(total.get("ev", 0.0)),
        "pf_trades": int(total.get("trade_count", 0)),
        "normal_trades": int(total.get("normal_trades", 0)),
        "extended_trades": int(total.get("extended_trades", 0)),
        "missed_buys": missed_buys,
        "missed_sells": missed_sells,
        "missed_total": missed_buys + missed_sells,
        "annual_trades": float(total.get("annual_trades", 0.0)),
        "reserved_buy_fill_rate": float(total.get("fill_rate", 0.0)),
        "avg_exposure": float(total.get("avg_exposure", 0.0)),
        "final_equity": float(total.get("final_equity", 0.0)),
    }
    benchmark_metrics = {
        "pf_return": float(total.get("benchmark_total_return_pct", 0.0)),
        "annual_return_pct": float(total.get("benchmark_annualized_return_pct", 0.0)),
        "min_full_year_return_pct": float(total.get("benchmark_min_full_year_return_pct", 0.0)),
        "pf_mdd": float(total.get("benchmark_max_drawdown_pct", 0.0)),
        "r_squared": float(total.get("benchmark_r_squared", 0.0)),
        "m_win_rate": float(total.get("benchmark_monthly_win_rate", 0.0)),
        "pf_romd": float(total.get("benchmark_score_romd", 0.0)),
        "final_equity": _benchmark_final_equity(float(initial_capital), float(total.get("benchmark_total_return_pct", 0.0))),
    }
    range_start = str(total.get("oos_start") or "")
    range_end = str(total.get("oos_end") or "")
    range_text = f"{range_start} ~ {range_end}" if range_start and range_end else "-"
    return candidate_metrics, benchmark_metrics, range_text


def _build_search_train_dates_for_session(session):
    sorted_dates = list(session.sorted_master_dates or [])
    if normalize_objective_mode(session.objective_mode) != OBJECTIVE_MODE_SPLIT_TRAIN_ROMD:
        return sorted_dates
    return filter_search_train_dates(
        sorted_dates=sorted_dates,
        train_start_year=int(session.train_start_year),
        search_train_end_year=int(session.search_train_end_year),
        train_start_date=_policy_date(session, "train_start_date"),
        search_train_end_date=_policy_date(session, "search_train_end_date"),
    )


def _build_global_strategy_text() -> str:
    return f"買入排序 [{get_buy_sort_title(BUY_SORT_METHOD)}]"


def _calc_romd(ret_pct: float, mdd_pct: float) -> float:
    mdd_pct = abs(float(mdd_pct))
    if mdd_pct <= 0:
        return 0.0
    return float(ret_pct) / (mdd_pct + 0.0001)


def _benchmark_final_equity(initial_capital: float, bm_return_pct: float) -> float:
    return float(initial_capital) * (1.0 + float(bm_return_pct) / 100.0)


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


def _format_split_bucket(total: int, left_label: str, left_value: int, right_label: str, right_value: int, *, separator: str = "｜") -> str:
    return f"{int(total)} ({left_label}: {int(left_value)}{separator}{right_label}: {int(right_value)})"


def _pass_color(passed: bool, *, pass_color: str = C_GREEN, fail_color: str = C_RED) -> str:
    return pass_color if bool(passed) else fail_color


def _pass_with_positive_color(passed: bool, numeric_value: float) -> str:
    if not bool(passed):
        return C_RED
    return C_GREEN if float(numeric_value) > 0 else C_YELLOW


def _first_zone_base_color(metric_name: str, numeric_value: float) -> str:
    if metric_name in {"總資產報酬率", "年化報酬率", "年度最差報酬"}:
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


def _optimizer_train_score_display_label() -> str:
    method = str(SCORE_CALC_METHOD or "").strip() or "Score"
    numerator = str(SCORE_NUMERATOR_METHOD or "").strip() or "TOTAL_RETURN"
    return f"Train Score[{method}/{numerator}]"



def _build_first_zone_rows(*, candidate_metrics: dict, reference_metrics: dict | None, benchmark_metrics: dict | None = None):
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
        row = {
            "name": name,
            "candidate": _compose_first_zone_cell(name, candidate_text, float(candidate_numeric), use_blue=use_blue, base_color_override=base_color_override) if candidate_numeric is not None else str(candidate_text),
            "candidate_precolored": candidate_numeric is not None,
            "reference": _compose_first_zone_cell(name, reference_text, float(reference_numeric), delta_text=reference_delta_text, delta_value=reference_delta_value, use_blue=use_blue, base_color_override=base_color_override) if reference_numeric is not None else str(reference_text),
            "reference_precolored": reference_numeric is not None,
            "benchmark": _compose_first_zone_cell(name, benchmark_text, float(benchmark_numeric), delta_text=benchmark_delta_text, delta_value=benchmark_delta_value, use_blue=use_blue, base_color_override=base_color_override) if benchmark_numeric is not None else str(benchmark_text),
            "benchmark_precolored": benchmark_numeric is not None,
        }
        if candidate_color_override is not None:
            row["candidate"] = _colorize(str(candidate_text), candidate_color_override)
            row["candidate_precolored"] = True
        rows.append(row)

    def add_row(name, key, *, kind="pct", candidate_unit="", digits=2):
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
                candidate_metrics.get('pf_trades', 0),
                '正常',
                candidate_metrics.get('normal_trades', 0),
                '延續',
                candidate_metrics.get('extended_trades', 0),
            )
            reference_plain = _format_split_bucket(
                reference_metrics.get('pf_trades', 0),
                '正常',
                reference_metrics.get('normal_trades', 0),
                '延續',
                reference_metrics.get('extended_trades', 0),
            ) if reference_metrics else "-"
            _append_row(name, cand_plain, None, reference_text=reference_plain, reference_numeric=None, benchmark_text="-", benchmark_numeric=None)
            return
        elif kind == "missed_split":
            cand_plain = _format_split_bucket(
                candidate_metrics.get('missed_total', 0),
                '買',
                candidate_metrics.get('missed_buys', 0),
                '賣',
                candidate_metrics.get('missed_sells', 0),
            )
            reference_plain = _format_split_bucket(
                reference_metrics.get('missed_total', 0),
                '買',
                reference_metrics.get('missed_buys', 0),
                '賣',
                reference_metrics.get('missed_sells', 0),
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


def _build_training_param_lines(params):
    return format_training_param_lines(params)


def _build_hard_gate_lines():
    from core.config import (
        MAX_PORTFOLIO_MDD_PCT,
        MIN_ANNUAL_TRADES,
        MIN_BUY_FILL_RATE,
        MIN_EQUITY_CURVE_R_SQUARED,
        MIN_FULL_YEAR_RETURN_PCT,
        MIN_MONTHLY_WIN_RATE,
        MIN_TRADE_WIN_RATE,
    )
    return [
        f"交易頻率：年化交易次數 >= {MIN_ANNUAL_TRADES:.2f} 次/年｜保留後買進成交率 >= {MIN_BUY_FILL_RATE:.2f}%｜完整交易勝率 >= {MIN_TRADE_WIN_RATE:.2f}%",
        f"績效風險：完整年度最差報酬 >= {MIN_FULL_YEAR_RETURN_PCT:.2f}%｜最大回撤(MDD) <= {MAX_PORTFOLIO_MDD_PCT:.2f}%  穩定度：月度獲利勝率 >= {MIN_MONTHLY_WIN_RATE:.2f}%｜權益曲線 R² >= {MIN_EQUITY_CURVE_R_SQUARED:.2f}",
    ]




def _portfolio_replay_metrics_from_result(result, *, initial_capital: float) -> tuple[dict, dict, str]:
    if len(result) < 26:
        raise ValueError(f"portfolio replay result 欄位數不足：{len(result)}")
    total_return = _safe_float(result[2])
    max_drawdown = _safe_float(result[3])
    trade_count = _safe_int(result[4])
    win_rate = _safe_float(result[5])
    pf_ev = _safe_float(result[6])
    pf_payoff = _safe_float(result[7])
    final_equity = _safe_float(result[8])
    avg_exp = _safe_float(result[9])
    benchmark_return = _safe_float(result[11])
    benchmark_mdd = _safe_float(result[12])
    missed_buys = _safe_int(result[13])
    missed_sells = _safe_int(result[14])
    r_squared = _safe_float(result[15])
    monthly_win_rate = _safe_float(result[16])
    bm_r_squared = _safe_float(result[17])
    bm_monthly_win_rate = _safe_float(result[18])
    normal_trade_count = _safe_int(result[19])
    extended_trade_count = _safe_int(result[20])
    annual_trades = _safe_float(result[21])
    reserved_buy_fill_rate = _safe_float(result[22])
    annual_return_pct = _safe_float(result[23])
    bm_annual_return_pct = _safe_float(result[24])
    profile = dict(result[25] or {})
    portfolio_total_r = _safe_float(profile.get("portfolio_total_r", 0.0))
    portfolio_median_r = _safe_float(profile.get("portfolio_median_r", 0.0))
    candidate_score = calc_portfolio_score(
        total_return,
        max_drawdown,
        monthly_win_rate,
        r_squared,
        annual_return_pct=annual_return_pct,
        trade_win_rate_pct=win_rate,
        min_full_year_return_pct=_safe_float(profile.get("min_full_year_return_pct", 0.0)),
        total_r=portfolio_total_r,
        median_r=portfolio_median_r,
    )
    benchmark_score = calc_portfolio_score(
        benchmark_return,
        benchmark_mdd,
        bm_monthly_win_rate,
        bm_r_squared,
        annual_return_pct=bm_annual_return_pct,
        min_full_year_return_pct=_safe_float(profile.get("bm_min_full_year_return_pct", 0.0)),
    )
    candidate_metrics = {
        "pf_return": total_return,
        "annual_return_pct": annual_return_pct,
        "min_full_year_return_pct": _safe_float(profile.get("min_full_year_return_pct", 0.0)),
        "pf_mdd": max_drawdown,
        "pf_romd": float(candidate_score),
        "r_squared": r_squared,
        "m_win_rate": monthly_win_rate,
        "win_rate": win_rate,
        "pf_payoff": pf_payoff,
        "pf_ev": pf_ev,
        "pf_total_r": portfolio_total_r,
        "pf_median_r": portfolio_median_r,
        "pf_trades": trade_count,
        "normal_trades": normal_trade_count,
        "extended_trades": extended_trade_count,
        "missed_buys": missed_buys,
        "missed_sells": missed_sells,
        "missed_total": missed_buys + missed_sells,
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "avg_exposure": avg_exp,
        "final_equity": final_equity,
    }
    benchmark_metrics = {
        "pf_return": benchmark_return,
        "annual_return_pct": bm_annual_return_pct,
        "min_full_year_return_pct": _safe_float(profile.get("bm_min_full_year_return_pct", 0.0)),
        "pf_mdd": benchmark_mdd,
        "pf_romd": float(benchmark_score),
        "r_squared": bm_r_squared,
        "m_win_rate": bm_monthly_win_rate,
        "final_equity": _benchmark_final_equity(float(initial_capital), benchmark_return),
    }
    range_start = str(profile.get("active_replay_start_date") or "").strip()
    range_end = str(profile.get("active_replay_end_date") or "").strip()
    if not range_start and len(result) > 0 and hasattr(result[0], "empty") and not result[0].empty:
        range_start = str(result[0].iloc[0].get("Date", ""))[:10]
        range_end = str(result[0].iloc[-1].get("Date", ""))[:10]
    range_text = f"{range_start} ~ {range_end}" if range_start and range_end else "-"
    return candidate_metrics, benchmark_metrics, range_text


def _run_static_ensemble_dashboard_replay(session, ensemble_payload: dict, *, start_date, end_date, initial_capital: float):
    from tools.portfolio_sim.simulation_runner import run_portfolio_simulation_with_param_ensemble
    data_dir = getattr(session, "raw_data_cache_data_dir", None)
    if not data_dir:
        raise ValueError("session 尚未載入 data_dir，無法建立 ensemble console dashboard")
    result = run_portfolio_simulation_with_param_ensemble(
        data_dir,
        ensemble_payload,
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
        start_year=int(pd.Timestamp(start_date).year) if start_date is not None else None,
        benchmark_ticker="0050",
        verbose=False,
        start_date=start_date,
        end_date=end_date,
        use_prepared_cache=False,
        write_prepared_cache=False,
    )
    return _portfolio_replay_metrics_from_result(result, initial_capital=float(initial_capital))

def _compute_reference_console_cache(session):
    params_path = resolve_run_best_params_path(PROJECT_ROOT)
    if not os.path.exists(params_path):
        return None
    try:
        source = load_portfolio_param_source_from_json(params_path)
        initial_capital = _safe_float(get_p(source["primary_params"], "initial_capital", 0.0))
        search_train_dates = _build_search_train_dates_for_session(session)
        reference_oos_metrics = None
        reference_oos_benchmark_metrics = None
        if source.get("source_type") == "active_param_ensemble":
            train_start_date = search_train_dates[0] if search_train_dates else _policy_date(session, "train_start_date")
            train_end_date = search_train_dates[-1] if search_train_dates else _policy_date(session, "search_train_end_date")
            cache, _benchmark_metrics, _train_range = _run_static_ensemble_dashboard_replay(
                session,
                source["payload"],
                start_date=train_start_date,
                end_date=train_end_date,
                initial_capital=initial_capital,
            )
            cache["source_path"] = params_path
            cache["wf_report"] = None
            if _resolve_session_model_mode(session) == "oos":
                oos_start_date = _policy_date(session, "oos_start_date")
                if oos_start_date is None and session.walk_forward_policy.get("oos_start_year") is not None:
                    oos_start_date = f"{int(session.walk_forward_policy['oos_start_year'])}-01-01"
                if oos_start_date:
                    reference_oos_metrics, reference_oos_benchmark_metrics, _unused_range = _run_static_ensemble_dashboard_replay(
                        session,
                        source["payload"],
                        start_date=oos_start_date,
                        end_date=_policy_date(session, "oos_end_date"),
                        initial_capital=initial_capital,
                    )
                    cache["oos_metrics"] = reference_oos_metrics
                    cache["oos_benchmark_metrics"] = reference_oos_benchmark_metrics
            return cache

        params = source["primary_params"]
        prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(params, "optimizer_max_workers"))
        prep_result = prepare_trial_inputs(
            raw_data_cache=session.raw_data_cache,
            params=params,
            default_max_workers=session.default_max_workers,
            executor_bundle=prep_executor_bundle,
            static_fast_cache=session.static_fast_cache,
            static_master_dates=session.master_dates,
            include_trade_logs=False,
            include_pit_stats_index=True,
        )
        benchmark_data = prep_result["all_dfs_fast"].get("0050", None)
        pf_profile = {}
        (
            ret_pct,
            mdd,
            trade_count,
            final_eq,
            avg_exp,
            _max_exp,
            bm_ret,
            bm_mdd,
            win_rate,
            pf_ev,
            pf_payoff,
            total_missed,
            total_missed_sells,
            r_sq,
            m_win_rate,
            bm_r_sq,
            bm_m_win_rate,
            normal_trade_count,
            extended_trade_count,
            annual_trades,
            reserved_buy_fill_rate,
            annual_return_pct,
            bm_annual_return_pct,
        ) = run_portfolio_timeline(
            prep_result["all_dfs_fast"],
            prep_result["all_trade_logs"],
            search_train_dates,
            session.train_start_year,
            params,
            session.train_max_positions,
            session.train_enable_rotation,
            benchmark_ticker="0050",
            benchmark_data=benchmark_data,
            is_training=True,
            profile_stats=pf_profile,
            verbose=False,
            pit_stats_index=prep_result.get("all_pit_stats_index"),
        )
        cache = {
            "pf_return": float(ret_pct),
            "annual_return_pct": float(annual_return_pct),
            "min_full_year_return_pct": float(pf_profile.get("min_full_year_return_pct", 0.0)),
            "pf_total_r": float(pf_profile.get("portfolio_total_r", 0.0)),
            "pf_median_r": float(pf_profile.get("portfolio_median_r", 0.0)),
            "pf_mdd": float(mdd),
            "r_squared": float(r_sq),
            "m_win_rate": float(m_win_rate),
            "win_rate": float(win_rate),
            "pf_payoff": float(pf_payoff),
            "pf_ev": float(pf_ev),
            "pf_trades": int(trade_count),
            "normal_trades": int(normal_trade_count),
            "extended_trades": int(extended_trade_count),
            "missed_buys": int(total_missed),
            "missed_sells": int(total_missed_sells),
            "missed_total": int(total_missed) + int(total_missed_sells),
            "annual_trades": float(annual_trades),
            "reserved_buy_fill_rate": float(reserved_buy_fill_rate),
            "avg_exposure": float(avg_exp),
            "final_equity": float(final_eq),
            "pf_romd": float(calc_portfolio_score(
                float(ret_pct),
                float(mdd),
                float(m_win_rate),
                float(r_sq),
                annual_return_pct=float(annual_return_pct),
                trade_win_rate_pct=float(win_rate),
                min_full_year_return_pct=float(pf_profile.get("min_full_year_return_pct", 0.0)),
                total_r=float(pf_profile.get("portfolio_total_r", 0.0)),
                median_r=float(pf_profile.get("portfolio_median_r", 0.0)),
            )),
            "source_path": params_path,
            "wf_report": None,
        }
        if _resolve_session_model_mode(session) == "oos":
            cache["wf_report"] = evaluate_walk_forward(
                all_dfs_fast=prep_result["all_dfs_fast"],
                all_trade_logs=prep_result["all_trade_logs"],
                sorted_dates=sorted(prep_result["master_dates"]),
                holdout_period=_get_cached_walk_forward_holdout_period(session, sorted(prep_result["master_dates"])),
                params=params,
                max_positions=session.train_max_positions,
                enable_rotation=session.train_enable_rotation,
                benchmark_ticker="0050",
                train_start_year=int(session.walk_forward_policy["train_start_year"]),
                min_train_years=int(session.walk_forward_policy["min_train_years"]),
                oos_start_year=session.walk_forward_policy.get("oos_start_year"),
                train_start_date=_policy_date(session, "train_start_date"),
                oos_start_date=_policy_date(session, "oos_start_date"),
                oos_end_date=_policy_date(session, "oos_end_date"),
                pit_stats_index=prep_result.get("all_pit_stats_index"),
            )
        return cache
    except Exception as exc:
        setattr(session, "_optimizer_console_reference_cache_error", repr(exc))
        return None

def _get_reference_console_cache(session):
    cache = getattr(session, "_optimizer_console_reference_cache", None)
    if cache is None:
        cache = _compute_reference_console_cache(session)
        setattr(session, "_optimizer_console_reference_cache", cache)
    return cache



def _get_cached_walk_forward_holdout_period(session, sorted_dates):
    sorted_dates = list(sorted_dates or [])
    if not sorted_dates:
        return None
    key = (
        len(sorted_dates),
        sorted_dates[0],
        sorted_dates[-1],
        int(session.walk_forward_policy["train_start_year"]),
        int(session.walk_forward_policy["min_train_years"]),
        session.walk_forward_policy.get("oos_start_year"),
        _policy_date(session, "train_start_date"),
        _policy_date(session, "oos_start_date"),
        _policy_date(session, "oos_end_date"),
    )
    cache = getattr(session, "_optimizer_wf_holdout_period_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(session, "_optimizer_wf_holdout_period_cache", cache)
    if key not in cache:
        cache[key] = build_test_holdout_period(
            sorted_dates,
            train_start_year=key[3],
            min_train_years=key[4],
            oos_start_year=key[5],
            train_start_date=key[6],
            oos_start_date=key[7],
            oos_end_date=key[8],
        )
    return cache.get(key)



def _build_optimizer_trial_dashboard_payload(session, trial, *, timing_breakdown=None):
    attrs = trial.user_attrs
    params_mapping = session.build_optimizer_trial_params(trial.params, attrs, fixed_tp_percent=session.optimizer_fixed_tp_percent)
    params = _build_trial_params_object(params_mapping)
    mode_display = "關閉明牌（穩定鎖倉）" if not session.train_enable_rotation else "啟用 (汰弱換強)"
    model_mode = _resolve_session_model_mode(session)
    search_train_dates = _build_search_train_dates_for_session(session)
    latest_data_end = _latest_data_end_text(session)
    if model_mode == "oos":
        system_score_display = f"{format_system_score_for_display(attrs.get('base_score', 0.0), decimals=3)}（{_optimizer_train_score_display_label()}／僅供選參）"
    else:
        system_score_display = f"{format_system_score_for_display(attrs.get('base_score', 0.0), decimals=2)}（base_score）"
    initial_capital = _safe_float(get_p(params, "initial_capital", 0.0))
    candidate_train_metrics = {
        "pf_return": _safe_float(attrs.get("pf_return", 0.0)),
        "annual_return_pct": _safe_float(attrs.get("annual_return_pct", 0.0)),
        "min_full_year_return_pct": _safe_float(attrs.get("min_full_year_return_pct", 0.0)),
        "pf_total_r": _safe_float(attrs.get("pf_total_r", 0.0)),
        "pf_median_r": _safe_float(attrs.get("pf_median_r", 0.0)),
        "pf_mdd": _safe_float(attrs.get("pf_mdd", 0.0)),
        "r_squared": _safe_float(attrs.get("r_squared", 0.0)),
        "m_win_rate": _safe_float(attrs.get("m_win_rate", 0.0)),
        "win_rate": _safe_float(attrs.get("win_rate", 0.0)),
        "pf_payoff": _safe_float(attrs.get("pf_payoff", 0.0)),
        "pf_ev": _safe_float(attrs.get("pf_ev", 0.0)),
        "pf_trades": _safe_int(attrs.get("pf_trades", 0)),
        "normal_trades": _safe_int(attrs.get("normal_trades", attrs.get("pf_trades", 0))),
        "extended_trades": _safe_int(attrs.get("extended_trades", 0)),
        "missed_buys": _safe_int(attrs.get("missed_buys", 0)),
        "missed_sells": _safe_int(attrs.get("missed_sells", 0)),
        "missed_total": _safe_int(attrs.get("missed_buys", 0)) + _safe_int(attrs.get("missed_sells", 0)),
        "annual_trades": _safe_float(attrs.get("annual_trades", 0.0)),
        "reserved_buy_fill_rate": _safe_float(attrs.get("reserved_buy_fill_rate", 0.0)),
        "avg_exposure": _safe_float(attrs.get("avg_exposure", 0.0)),
        "final_equity": _safe_float(attrs.get("final_equity", 0.0)),
        "pf_romd": float(calc_portfolio_score(
            _safe_float(attrs.get("pf_return", 0.0)),
            _safe_float(attrs.get("pf_mdd", 0.0)),
            _safe_float(attrs.get("m_win_rate", 0.0)),
            _safe_float(attrs.get("r_squared", 0.0)),
            annual_return_pct=_safe_float(attrs.get("annual_return_pct", 0.0)),
            trade_win_rate_pct=_safe_float(attrs.get("win_rate", 0.0)),
            min_full_year_return_pct=_safe_float(attrs.get("min_full_year_return_pct", 0.0)),
            total_r=_safe_float(attrs.get("pf_total_r", 0.0)),
            median_r=_safe_float(attrs.get("pf_median_r", 0.0)),
        )),
    }
    benchmark_train_metrics = {
        "pf_return": _safe_float(attrs.get("bm_return", 0.0)),
        "annual_return_pct": _safe_float(attrs.get("bm_annual_return_pct", 0.0)),
        "min_full_year_return_pct": _safe_float(attrs.get("bm_min_full_year_return_pct", 0.0)),
        "pf_mdd": _safe_float(attrs.get("bm_mdd", 0.0)),
        "r_squared": _safe_float(attrs.get("bm_r_squared", 0.0)),
        "m_win_rate": _safe_float(attrs.get("bm_m_win_rate", 0.0)),
        "pf_romd": float(calc_portfolio_score(
            _safe_float(attrs.get("bm_return", 0.0)),
            _safe_float(attrs.get("bm_mdd", 0.0)),
            _safe_float(attrs.get("bm_m_win_rate", 0.0)),
            _safe_float(attrs.get("bm_r_squared", 0.0)),
            annual_return_pct=_safe_float(attrs.get("bm_annual_return_pct", 0.0)),
            min_full_year_return_pct=_safe_float(attrs.get("bm_min_full_year_return_pct", 0.0)),
        )),
        "final_equity": _benchmark_final_equity(initial_capital, _safe_float(attrs.get("bm_return", 0.0))),
    }
    reference_cache = _get_reference_console_cache(session)
    training_title = f"【訓練期間績效對比｜{_range_text_from_dates(search_train_dates)}】"
    train_rows = _build_first_zone_rows(
        candidate_metrics=candidate_train_metrics,
        reference_metrics=reference_cache,
        benchmark_metrics=benchmark_train_metrics,
    )

    test_title = None
    test_rows = None
    if model_mode == "oos":
        prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(params, "optimizer_max_workers"))
        consume_trial_milestone_inputs = getattr(session, "consume_trial_milestone_inputs", None)
        cached_trial_inputs = consume_trial_milestone_inputs(trial.number) if callable(consume_trial_milestone_inputs) else None
        candidate_wf_started_at = time.perf_counter()
        if cached_trial_inputs is not None:
            candidate_wf_report = evaluate_walk_forward(
                all_dfs_fast=(cached_trial_inputs.get("all_dfs_fast") or session.static_fast_cache),
                all_trade_logs={},
                sorted_dates=list(cached_trial_inputs.get("sorted_master_dates") or []),
                holdout_period=_get_cached_walk_forward_holdout_period(session, cached_trial_inputs.get("sorted_master_dates") or []),
                params=params,
                max_positions=session.train_max_positions,
                enable_rotation=session.train_enable_rotation,
                benchmark_ticker="0050",
                train_start_year=int(session.walk_forward_policy["train_start_year"]),
                min_train_years=int(session.walk_forward_policy["min_train_years"]),
                oos_start_year=session.walk_forward_policy.get("oos_start_year"),
                train_start_date=_policy_date(session, "train_start_date"),
                oos_start_date=_policy_date(session, "oos_start_date"),
                oos_end_date=_policy_date(session, "oos_end_date"),
                pit_stats_index=cached_trial_inputs.get("all_pit_stats_index"),
            )
        else:
            prep_result = prepare_trial_inputs(
                raw_data_cache=session.raw_data_cache,
                params=params,
                default_max_workers=session.default_max_workers,
                executor_bundle=prep_executor_bundle,
                static_fast_cache=session.static_fast_cache,
                static_master_dates=session.master_dates,
                include_trade_logs=False,
                include_pit_stats_index=True,
            )
            candidate_wf_report = evaluate_walk_forward(
                all_dfs_fast=prep_result["all_dfs_fast"],
                all_trade_logs=prep_result["all_trade_logs"],
                sorted_dates=sorted(prep_result["master_dates"]),
                holdout_period=_get_cached_walk_forward_holdout_period(session, sorted(prep_result["master_dates"])),
                params=params,
                max_positions=session.train_max_positions,
                enable_rotation=session.train_enable_rotation,
                benchmark_ticker="0050",
                train_start_year=int(session.walk_forward_policy["train_start_year"]),
                min_train_years=int(session.walk_forward_policy["min_train_years"]),
                oos_start_year=session.walk_forward_policy.get("oos_start_year"),
                train_start_date=_policy_date(session, "train_start_date"),
                oos_start_date=_policy_date(session, "oos_start_date"),
                oos_end_date=_policy_date(session, "oos_end_date"),
                pit_stats_index=prep_result.get("all_pit_stats_index"),
            )
        candidate_wf_elapsed = max(0.0, time.perf_counter() - candidate_wf_started_at)
        if isinstance(timing_breakdown, dict):
            timing_breakdown["candidate_wf_sec"] = candidate_wf_elapsed
        candidate_test_metrics, benchmark_test_metrics, oos_range_text = _build_oos_metrics_from_report(
            report=candidate_wf_report,
            initial_capital=initial_capital,
        )
        reference_test_metrics = None
        if reference_cache and reference_cache.get("wf_report"):
            reference_test_metrics, _unused_bm, _unused_range = _build_oos_metrics_from_report(
                report=reference_cache.get("wf_report"),
                initial_capital=initial_capital,
            )
        elif reference_cache and isinstance(reference_cache.get("oos_metrics"), dict):
            reference_test_metrics = dict(reference_cache.get("oos_metrics") or {})
        test_title = f"【OOS 驗證績效摘要｜{oos_range_text}｜資料終點：{latest_data_end}】"
        test_rows = _build_first_zone_rows(
            candidate_metrics=candidate_test_metrics,
            reference_metrics=reference_test_metrics,
            benchmark_metrics=benchmark_test_metrics,
        )

    return {
        "mode_display": mode_display,
        "model_mode": model_mode,
        "system_score_display": system_score_display,
        "params": params,
        "training_title": training_title,
        "train_rows": train_rows,
        "test_title": test_title,
        "test_rows": test_rows,
        "upgrade_rows": None,
        "compare_rows": None,
        "study_full_breakout_stats": _build_study_full_breakout_stats(attrs) if _is_study_full_session(session) else None,
        "base_score": _safe_float(attrs.get("base_score", 0.0)),
    }


def print_optimizer_trial_milestone_dashboard(session, trial, *, milestone_title: str, title: str = "績效與風險對比表"):
    timing_breakdown = {}
    payload_started_at = time.perf_counter()
    payload = _build_optimizer_trial_dashboard_payload(session, trial, timing_breakdown=timing_breakdown)
    payload_elapsed = max(0.0, time.perf_counter() - payload_started_at)
    render_started_at = time.perf_counter()
    print_optimizer_trial_console_dashboard(
        title=title,
        milestone_title=milestone_title,
        global_strategy_text=_build_global_strategy_text(),
        mode_display=payload["mode_display"],
        max_pos=session.train_max_positions,
        model_mode=payload["model_mode"],
        objective_mode=str(session.objective_mode),
        score_calc_method=SCORE_CALC_METHOD,
        score_numerator_method=SCORE_NUMERATOR_METHOD,
        system_score_display=payload["system_score_display"],
        training_title=payload["training_title"],
        training_rows=payload["train_rows"],
        testing_title=payload["test_title"],
        testing_rows=payload["test_rows"],
        upgrade_rows=payload["upgrade_rows"],
        compare_rows=payload["compare_rows"],
        params_lines=_build_training_param_lines(payload["params"]),
        hard_gate_lines=_build_hard_gate_lines(),
        study_full_breakout_stats=payload.get("study_full_breakout_stats"),
    )
    render_elapsed = max(0.0, time.perf_counter() - render_started_at)
    return {
        "payload_sec": float(payload_elapsed),
        "candidate_wf_sec": float(timing_breakdown.get("candidate_wf_sec", 0.0)),
        "render_sec": float(render_elapsed),
    }





def _static_replay_progress_message(message: str) -> str:
    return f"{C_GRAY}⏳ {str(message).strip()}{C_RESET}"


def _emit_static_replay_progress(progress_state: dict | None, message: str, *, finish: bool = False) -> None:
    if progress_state is None:
        return
    text = _static_replay_progress_message(message)
    if stdout_supports_inline_progress():
        previous_width = int(progress_state.get("width", 0) or 0)
        progress_state["width"] = write_inline_progress(text, previous_width=previous_width)
        if finish:
            print(flush=True)
            progress_state["width"] = 0
    else:
        last = str(progress_state.get("last", ""))
        if text != last or finish:
            print(text, flush=True)
            progress_state["last"] = text


def _finish_static_replay_progress(progress_state: dict | None) -> None:
    if progress_state is None:
        return
    if stdout_supports_inline_progress() and int(progress_state.get("width", 0) or 0) > 0:
        print(flush=True)
    progress_state["width"] = 0


def _build_static_policy_oos_row_from_metrics(candidate_metrics: dict, benchmark_metrics: dict) -> dict:
    candidate_score = _safe_float(candidate_metrics.get("pf_romd", 0.0))
    benchmark_score = _safe_float(benchmark_metrics.get("pf_romd", 0.0))
    return {
        "available": True,
        "rank_1_trial": None,
        "rank_1_oos": float(candidate_score),
        "rank_1_return_pct": _safe_float(candidate_metrics.get("pf_return", 0.0)),
        "rank_1_mdd_pct": _safe_float(candidate_metrics.get("pf_mdd", 0.0)),
        "rank_1_trades": _safe_int(candidate_metrics.get("pf_trades", 0)),
        "benchmark_0050_gap": float(candidate_score) - float(benchmark_score),
    }


def _build_static_policy_oos_row_from_active_metrics(metrics: dict, *, benchmark_score: float) -> dict:
    candidate_score = _safe_float(metrics.get("score", 0.0))
    return {
        "available": True,
        "rank_1_trial": None,
        "rank_1_oos": float(candidate_score),
        "rank_1_return_pct": _safe_float(metrics.get("return_pct", 0.0)),
        "rank_1_mdd_pct": _safe_float(metrics.get("mdd_pct", 0.0)),
        "rank_1_trades": _safe_int(metrics.get("trade_count", 0)),
        "benchmark_0050_gap": float(candidate_score) - float(benchmark_score),
        "benchmark_return_pct": _safe_float(metrics.get("benchmark_return_pct", 0.0)),
        "benchmark_mdd_pct": _safe_float(metrics.get("benchmark_mdd_pct", 0.0)),
        "benchmark_oos_score": _safe_float(metrics.get("benchmark_oos_score", 0.0)),
    }


def _load_static_policy_paramset_payloads(policy_paramsets: dict | None) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for policy_name, source in dict(policy_paramsets or {}).items():
        if isinstance(source, dict):
            payloads[str(policy_name)] = dict(source)
            continue
        path = str(source or "").strip()
        if not path:
            continue
        try:
            payload = load_json_file(path)
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            payloads[str(policy_name)] = payload
    return payloads


def _build_static_policy_rows_from_paramsets(
    session,
    *,
    policy_paramsets: dict | None,
    start_date,
    end_date,
    initial_capital: float,
    progress_state: dict | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    progress_emit=None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    _ = (initial_capital, progress_offset, progress_total)
    payloads = _load_static_policy_paramset_payloads(policy_paramsets)
    if not payloads:
        return {}, {}
    data_dir = getattr(session, "raw_data_cache_data_dir", None)
    if not data_dir:
        raise ValueError("session 尚未載入 data_dir，無法建立 ensemble policy replay")
    from tools.optimizer.outer_rolling_oos import (
        _build_policy_replay_context,
        _run_policy_replay_tasks,
        _stable_policy_replay_signature,
        is_optimizer_policy_replay_dedup_by_signature_enabled_default,
    )

    replay_context = _build_policy_replay_context(
        selected_data_dir=str(data_dir),
        oos_year=int(pd.Timestamp(start_date).year) if start_date is not None else int(session.train_start_year),
        oos_start_date=str(start_date)[:10],
        oos_end_date=str(end_date)[:10] if end_date is not None else str(_latest_data_end_text(session))[:10],
        max_positions=int(session.train_max_positions),
        enable_rotation=bool(session.train_enable_rotation),
    )
    dedup_enabled = is_optimizer_policy_replay_dedup_by_signature_enabled_default()
    unique_jobs_by_signature: dict[str, dict] = {}
    policy_jobs_by_name: dict[str, dict] = {}
    for policy_name, payload in payloads.items():
        signature = _stable_policy_replay_signature(payload) if dedup_enabled else f"{policy_name}:{_stable_policy_replay_signature(payload)}"
        job = policy_jobs_by_name[str(policy_name)] = {
            "signature": signature,
            "policy_name": str(policy_name),
            "policy_names": [str(policy_name)],
            "payload": dict(payload),
            "replay_context": replay_context,
        }
        if signature in unique_jobs_by_signature:
            unique_jobs_by_signature[signature].setdefault("policy_names", []).append(str(policy_name))
            continue
        unique_jobs_by_signature[signature] = dict(job)

    metrics_by_signature = _run_policy_replay_tasks(
        list(unique_jobs_by_signature.values()),
        progress_emit=progress_emit,
    )
    if not callable(progress_emit):
        _emit_static_replay_progress(progress_state, "seed ensemble policy replay | done", finish=True)
    raw_metrics_by_policy: dict[str, dict] = {
        name: dict(metrics_by_signature.get(str(job.get("signature"))) or {})
        for name, job in policy_jobs_by_name.items()
    }
    benchmark_source = next((metrics for metrics in raw_metrics_by_policy.values() if metrics.get("benchmark_oos_score") is not None), {})
    benchmark_score = _safe_float(benchmark_source.get("benchmark_oos_score", 0.0))
    rows: dict[str, dict] = {}
    for policy_name, metrics in raw_metrics_by_policy.items():
        if not metrics:
            rows[str(policy_name)] = {
                "available": False,
                "rank_1_oos": 0.0,
                "unavailable_reason": "policy_paramset_replay_failed",
            }
            continue
        rows[str(policy_name)] = _build_static_policy_oos_row_from_active_metrics(metrics, benchmark_score=benchmark_score)
    return rows, raw_metrics_by_policy


def build_optimizer_static_ensemble_single_fold_oos_row(session, *, ensemble_payload: dict, elapsed_sec: float | None = None, policy_paramsets: dict | None = None, progress_callback=None) -> dict:
    """Build the single-fold row used by the shared rolling-OOS table renderer."""
    policy = get_active_param_ensemble_policy(ensemble_payload)
    schedule = ensemble_payload.get("params_ensemble") or []
    first_member = schedule[0] if schedule else {}
    primary_params = _build_trial_params_object(first_member.get("params") or {})
    initial_capital = _safe_float(get_p(primary_params, "initial_capital", 0.0))
    model_mode = _resolve_session_model_mode(session)
    search_train_dates = _build_search_train_dates_for_session(session)
    selection_start = search_train_dates[0] if search_train_dates else _policy_date(session, "train_start_date")
    selection_end = search_train_dates[-1] if search_train_dates else _policy_date(session, "search_train_end_date")
    oos_start_date = None
    oos_end_date = None
    if model_mode == "oos":
        oos_start_date = _policy_date(session, "oos_start_date")
        if oos_start_date is None and session.walk_forward_policy.get("oos_start_year") is not None:
            oos_start_date = f"{int(session.walk_forward_policy['oos_start_year'])}-01-01"
        oos_end_date = _policy_date(session, "oos_end_date")
    replay_start_date = oos_start_date if model_mode == "oos" and oos_start_date else selection_start
    replay_end_date = oos_end_date if model_mode == "oos" and oos_start_date else selection_end
    replay_policy_paramsets = policy_paramsets or ensemble_payload.get("policy_paramsets")
    replay_payloads = _load_static_policy_paramset_payloads(replay_policy_paramsets)
    progress_state: dict | None = None if callable(progress_callback) else {}
    replay_started_perf = time.perf_counter()
    replay_started_ts = time.time()
    replay_last_done_ts: float | None = None

    def _emit_replay(policy_name: str, *, done: int, total: int | None = None, status: str = "RUN") -> None:
        nonlocal replay_last_done_ts
        done_value = int(done or 0)
        total_value = int(total if total is not None else len(replay_payloads))
        if done_value > 0:
            replay_last_done_ts = time.time()
        if callable(progress_callback):
            progress_callback({
                "stage": "ENSEMBLE_REPLAY",
                "status": str(status or "RUN"),
                "policy": str(policy_name),
                "replay_done": int(done_value),
                "replay_total": int(total_value),
                "replay_started_ts": float(replay_started_ts),
                "replay_last_done_ts": float(replay_last_done_ts) if replay_last_done_ts is not None else None,
                "elapsed_sec": max(0.0, time.perf_counter() - replay_started_perf),
            })
        else:
            step = max(1, min(max(1, total_value), done_value if done_value > 0 else 1))
            _emit_static_replay_progress(progress_state, f"seed ensemble policy replay | {step}/{max(1, total_value)} | {policy_name}")

    policy_rows, raw_policy_metrics = _build_static_policy_rows_from_paramsets(
        session,
        policy_paramsets=replay_payloads,
        start_date=replay_start_date,
        end_date=replay_end_date,
        initial_capital=initial_capital,
        progress_state=progress_state,
        progress_emit=_emit_replay if callable(progress_callback) else None,
    )
    available_policy_rows = [dict(row) for row in policy_rows.values() if bool(dict(row).get("available", False))]
    candidate_score = max((_safe_float(row.get("rank_1_oos", 0.0)) for row in available_policy_rows), default=0.0)
    benchmark_score = 0.0
    benchmark_return_pct = 0.0
    benchmark_mdd_pct = 0.0
    for metrics in raw_policy_metrics.values():
        if metrics:
            benchmark_score = _safe_float(metrics.get("benchmark_oos_score", 0.0))
            benchmark_return_pct = _safe_float(metrics.get("benchmark_return_pct", 0.0))
            benchmark_mdd_pct = _safe_float(metrics.get("benchmark_mdd_pct", 0.0))
            break
    range_start = str(oos_start_date or "")[:10] if model_mode == "oos" else ""
    range_end = str(oos_end_date or "")[:10] if model_mode == "oos" and oos_end_date else ""
    from tools.optimizer.outer_rolling_oos import (
        BASE_RETENTION_COMPARISON_POLICY_NAMES,
        REPORT_POLICY_NAMES,
        build_optimizer_seed_ensemble_fold_context,
        normalize_optimizer_seed_ensemble_fold_row,
    )

    fold_context = build_optimizer_seed_ensemble_fold_context(
        fold_idx=1,
        fold_count=1,
        selection_start_date=str(selection_start or "")[:10],
        selection_end_date=str(selection_end or "")[:10],
        oos_start_date=range_start,
        oos_end_date=range_end or ("latest" if model_mode == "oos" else ""),
        oos_period="" if model_mode != "oos" else None,
    )
    row = dict(fold_context)
    row.update({
        "best_finalist_oos_score": float(candidate_score),
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(benchmark_return_pct),
        "benchmark_mdd_pct": float(benchmark_mdd_pct),
        "elapsed_sec": elapsed_sec,
        "random_seed_ensemble": dict(policy),
    })

    unavailable_policy = {
        "available": False,
        "rank_1_oos": 0.0,
        "unavailable_reason": "nonrolling_policy_paramset_not_available",
    }
    for policy_name in tuple(REPORT_POLICY_NAMES) + tuple(BASE_RETENTION_COMPARISON_POLICY_NAMES):
        if policy_name in policy_rows:
            row[str(policy_name)] = dict(policy_rows[policy_name])
        else:
            row[str(policy_name)] = dict(unavailable_policy)
    return normalize_optimizer_seed_ensemble_fold_row(row)

def print_optimizer_static_ensemble_rolling_oos_table(session, *, ensemble_payload: dict, elapsed_sec: float | None = None, policy_paramsets: dict | None = None, progress_callback=None) -> dict | None:
    """Print non-rolling ensemble summary through the exact rolling-OOS table renderer."""
    from tools.optimizer.outer_rolling_oos import optimizer_seed_ensemble_table_titles, render_optimizer_results_tables

    row = build_optimizer_static_ensemble_single_fold_oos_row(
        session,
        ensemble_payload=ensemble_payload,
        elapsed_sec=elapsed_sec,
        policy_paramsets=policy_paramsets,
        progress_callback=progress_callback,
    )
    main_title, retention_title = optimizer_seed_ensemble_table_titles()
    table_text = render_optimizer_results_tables(
        [row],
        color=True,
        include_chain=False,
        include_oos_avg=False,
        main_table_title=main_title,
        retention_table_title=retention_title,
    )
    if table_text:
        print("\n" + table_text)


def print_optimizer_static_ensemble_console_dashboard(
    session,
    *,
    ensemble_payload: dict,
    seeds: list[int],
    milestone_title: str = "🏆 ENSEMBLE 訓練結果",
    title: str = "ENSEMBLE 績效與風險對比表",
    force: bool = False,
):
    if not bool(force):
        return {"payload_sec": 0.0, "render_sec": 0.0}
    payload_started_at = time.perf_counter()
    progress_state: dict = {}
    _emit_static_replay_progress(progress_state, "seed ensemble dashboard replay | train")
    policy = get_active_param_ensemble_policy(ensemble_payload)
    schedule = ensemble_payload.get("params_ensemble") or []
    first_member = schedule[0] if schedule else {}
    primary_params = _build_trial_params_object(first_member.get("params") or {})
    initial_capital = _safe_float(get_p(primary_params, "initial_capital", 0.0))
    model_mode = _resolve_session_model_mode(session)
    search_train_dates = _build_search_train_dates_for_session(session)
    train_start_date = search_train_dates[0] if search_train_dates else _policy_date(session, "train_start_date")
    train_end_date = search_train_dates[-1] if search_train_dates else _policy_date(session, "search_train_end_date")
    candidate_train_metrics, benchmark_train_metrics, train_range_text = _run_static_ensemble_dashboard_replay(
        session,
        ensemble_payload,
        start_date=train_start_date,
        end_date=train_end_date,
        initial_capital=initial_capital,
    )
    reference_cache = _get_reference_console_cache(session)
    train_rows = _build_first_zone_rows(
        candidate_metrics=candidate_train_metrics,
        reference_metrics=reference_cache,
        benchmark_metrics=benchmark_train_metrics,
    )
    test_title = None
    test_rows = None
    latest_data_end = _latest_data_end_text(session)
    if model_mode == "oos":
        oos_start_date = _policy_date(session, "oos_start_date")
        if oos_start_date is None and session.walk_forward_policy.get("oos_start_year") is not None:
            oos_start_date = f"{int(session.walk_forward_policy['oos_start_year'])}-01-01"
        if oos_start_date:
            _emit_static_replay_progress(progress_state, "seed ensemble dashboard replay | OOS")
            candidate_test_metrics, benchmark_test_metrics, oos_range_text = _run_static_ensemble_dashboard_replay(
                session,
                ensemble_payload,
                start_date=oos_start_date,
                end_date=_policy_date(session, "oos_end_date"),
                initial_capital=initial_capital,
            )
            reference_test_metrics = None
            if reference_cache and reference_cache.get("wf_report"):
                reference_test_metrics, _unused_bm, _unused_range = _build_oos_metrics_from_report(
                    report=reference_cache.get("wf_report"),
                    initial_capital=initial_capital,
                )
            elif reference_cache and isinstance(reference_cache.get("oos_metrics"), dict):
                reference_test_metrics = dict(reference_cache.get("oos_metrics") or {})
            test_title = f"【OOS 驗證績效摘要｜{oos_range_text}｜資料終點：{latest_data_end}】"
            test_rows = _build_first_zone_rows(
                candidate_metrics=candidate_test_metrics,
                reference_metrics=reference_test_metrics,
                benchmark_metrics=benchmark_test_metrics,
            )
    seed_text = ",".join(str(int(seed)) for seed in seeds)
    params_lines = [
        f"ENSEMBLE：N={int(policy['seed_count'])}｜min_agree={int(policy['min_agree'])}｜seeds={seed_text}",
        *[f"代表 member#1｜{line}" if idx == 0 else line for idx, line in enumerate(_build_training_param_lines(primary_params))],
    ]
    payload_elapsed = max(0.0, time.perf_counter() - payload_started_at)
    _emit_static_replay_progress(progress_state, "seed ensemble dashboard replay | render")
    render_started_at = time.perf_counter()
    print_optimizer_trial_console_dashboard(
        title=title,
        milestone_title=milestone_title,
        global_strategy_text=_build_global_strategy_text(),
        mode_display="關閉明牌（穩定鎖倉）" if not session.train_enable_rotation else "啟用 (汰弱換強)",
        max_pos=session.train_max_positions,
        model_mode=model_mode,
        objective_mode=str(session.objective_mode),
        score_calc_method=SCORE_CALC_METHOD,
        score_numerator_method=SCORE_NUMERATOR_METHOD,
        system_score_display=f"{format_system_score_for_display(candidate_train_metrics.get('pf_romd', 0.0), decimals=3)}（{_optimizer_train_score_display_label()}／ENSEMBLE）",
        training_title=f"【訓練期間績效對比｜{train_range_text}】",
        training_rows=train_rows,
        testing_title=test_title,
        testing_rows=test_rows,
        upgrade_rows=None,
        compare_rows=None,
        params_lines=params_lines,
        hard_gate_lines=_build_hard_gate_lines(),
    )
    _finish_static_replay_progress(progress_state)
    return {
        "payload_sec": float(payload_elapsed),
        "render_sec": float(max(0.0, time.perf_counter() - render_started_at)),
    }

def run_optimizer_monitoring_callback(session, study, trial):
    callback_started_at = time.perf_counter()
    callback_best_lookup_sec = 0.0
    callback_status_line_sec = 0.0
    callback_milestone_dashboard_sec = 0.0
    callback_milestone_payload_sec = 0.0
    callback_milestone_candidate_wf_sec = 0.0
    callback_milestone_render_sec = 0.0

    session.current_session_trial += 1
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    profile_row = trial.user_attrs.get("profile_row") or {}
    try:
        objective_wall_sec = float(profile_row.get("objective_wall_sec", 0.0) or 0.0)
    except (TypeError, ValueError):
        objective_wall_sec = 0.0
    prep_mode = trial.user_attrs.get("prep_mode", "parallel")
    mode_suffix = " [fallback]" if prep_mode == "sequential_fallback" else ""

    if trial.value is None:
        state_name = getattr(trial.state, "name", str(trial.state))
        status_text, score_text = f"{session.colors['yellow']}{state_name}{mode_suffix}{session.colors['reset']}", "N/A"
    elif not is_qualified_trial_value(trial.value):
        fail_msg = trial.user_attrs.get("fail_reason", "策略無效")
        status_text, score_text = f"{session.colors['yellow']}淘汰 [{fail_msg}]{mode_suffix}{session.colors['reset']}", "N/A"
    else:
        status_text = f"{session.colors['green']}進化中{mode_suffix}{session.colors['reset']}"
        score_text = format_system_score_for_display(trial.value, decimals=3)

    total_trials_display = str(session.n_trials) if isinstance(session.n_trials, int) and session.n_trials > 0 else "?"

    def _print_status_line(display_total_wall_sec: float) -> float:
        if bool(getattr(session, "disable_optimizer_status_line", False)):
            return 0.0
        status_started_at = time.perf_counter()
        line = (
            f"{session.colors['gray']}⏳ [累積 {trial.number + 1:>4} | 本輪 {session.current_session_trial:>3}/{total_trials_display}] "
            f"耗時: {float(display_total_wall_sec):>5.1f}s | 系統得分: {score_text:>7} | 狀態: {status_text}{session.colors['reset']}"
        )
        if bool(getattr(session, "timing_mode", False)):
            print(line, flush=True)
        elif stdout_supports_inline_progress():
            previous_width = int(getattr(session, "optimizer_inline_progress_width", 0) or 0)
            session.optimizer_inline_progress_width = write_inline_progress(line, previous_width=previous_width)
            session.optimizer_inline_progress_rendered = True
        return max(0.0, time.perf_counter() - status_started_at)

    best_lookup_started_at = time.perf_counter()
    best_completed_trial = session.get_best_completed_trial_or_none(study)
    callback_best_lookup_sec = max(0.0, time.perf_counter() - best_lookup_started_at)
    disable_milestone_dashboard = getattr(session, "disable_milestone_dashboard", None)
    if disable_milestone_dashboard is None:
        disable_milestone_dashboard = True
    should_render_milestone_dashboard = not bool(disable_milestone_dashboard)
    is_new_best = (
        should_render_milestone_dashboard
        and best_completed_trial is not None
        and best_completed_trial.number == trial.number
        and is_qualified_trial_value(trial.value)
    )

    if is_new_best:
        milestone_started_at = time.perf_counter()
        dashboard_buffer = io.StringIO()
        with contextlib.redirect_stdout(dashboard_buffer):
            milestone_stats = print_optimizer_trial_milestone_dashboard(
                session,
                trial,
                title="績效與風險對比表",
                milestone_title=f"🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試)",
            )
        dashboard_text = dashboard_buffer.getvalue()
        callback_milestone_dashboard_sec = max(0.0, time.perf_counter() - milestone_started_at)
        callback_milestone_payload_sec = float((milestone_stats or {}).get("payload_sec", 0.0))
        callback_milestone_candidate_wf_sec = float((milestone_stats or {}).get("candidate_wf_sec", 0.0))
        callback_milestone_render_sec = float((milestone_stats or {}).get("render_sec", 0.0))

        # 先完成本 trial 的破紀錄報表計算，再印狀態列。
        # 狀態列耗時 = 該 trial objective + 該 trial callback 已完成工作；不是 optimizer 累積 wall time。
        pre_status_elapsed = max(0.0, time.perf_counter() - callback_started_at)
        callback_status_line_sec += _print_status_line(float(duration) + float(pre_status_elapsed))
        if (
            not bool(getattr(session, "timing_mode", False))
            and bool(getattr(session, "optimizer_inline_progress_rendered", False))
        ):
            print()
            session.optimizer_inline_progress_rendered = False
            session.optimizer_inline_progress_width = 0
        if dashboard_text:
            print(dashboard_text, end="", flush=True)

    else:
        discard_trial_milestone_inputs = getattr(session, "discard_trial_milestone_inputs", None)
        if callable(discard_trial_milestone_inputs):
            discard_trial_milestone_inputs(trial.number)

        pre_status_elapsed = max(0.0, time.perf_counter() - callback_started_at)
        callback_status_line_sec += _print_status_line(float(duration) + float(pre_status_elapsed))

    callback_wall_sec = max(0.0, time.perf_counter() - callback_started_at)
    trial_total_wall_sec = float(duration) + float(callback_wall_sec)
    session.profile_recorder.patch_row(
        trial.number,
        {
            "trial_total_wall_sec": float(trial_total_wall_sec),
            "outer_nonobjective_sec": max(0.0, float(trial_total_wall_sec) - float(objective_wall_sec)),
            "callback_wall_sec": float(callback_wall_sec),
            "callback_best_lookup_sec": float(callback_best_lookup_sec),
            "callback_status_line_sec": float(callback_status_line_sec),
            "callback_milestone_dashboard_sec": float(callback_milestone_dashboard_sec),
            "callback_milestone_payload_sec": float(callback_milestone_payload_sec),
            "callback_milestone_candidate_wf_sec": float(callback_milestone_candidate_wf_sec),
            "callback_milestone_render_sec": float(callback_milestone_render_sec),
        },
    )
    session.profile_recorder.mark_trial_completed(trial.number)
