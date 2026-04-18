import os

import pandas as pd

from core.buy_sort import get_buy_sort_title
from core.config import (
    BUY_SORT_METHOD,
    MAX_PORTFOLIO_MDD_PCT,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_MONTHLY_WIN_RATE,
    SCORE_CALC_METHOD,
    SCORE_NUMERATOR_METHOD,
)
from core.display_common import C_CYAN, C_GREEN, C_RED, C_RESET, C_YELLOW, get_p
from core.model_paths import resolve_champion_params_path
from core.walk_forward_policy import filter_search_train_dates
from core.params_io import build_params_from_mapping, load_params_from_json, params_to_json_dict
from core.portfolio_engine import run_portfolio_timeline
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
    print_optimizer_trial_console_dashboard,
)
from tools.optimizer.prep import prepare_trial_inputs
from tools.optimizer.study_utils import (
    OBJECTIVE_MODE_WF_GATE_MEDIAN,
    OBJECTIVE_MODE_SPLIT_TEST_ROMD,
    is_qualified_trial_value,
)
from tools.optimizer.walk_forward import build_compare_assessment, build_oos_total_performance, evaluate_walk_forward


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAMPION_PARAMS_PATH = resolve_champion_params_path(PROJECT_ROOT)


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
    mode = str(objective_mode)
    if mode == OBJECTIVE_MODE_WF_GATE_MEDIAN:
        return "wf"
    if mode == OBJECTIVE_MODE_SPLIT_TEST_ROMD:
        return "split"
    return "legacy"


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




def _weighted_average_from_rows(rows, *, value_key: str, weight_key: str) -> float:
    rows = list(rows or [])
    if not rows:
        return 0.0
    weighted_total = 0.0
    total_weight = 0.0
    fallback_values = []
    for row in rows:
        value = _safe_float(row.get(value_key, 0.0))
        weight = max(0.0, _safe_float(row.get(weight_key, 0.0)))
        fallback_values.append(value)
        if weight > 0.0:
            weighted_total += value * weight
            total_weight += weight
    if total_weight > 0.0:
        return weighted_total / total_weight
    return sum(fallback_values) / len(fallback_values) if fallback_values else 0.0


def _resolve_oos_window_year(row) -> int | None:
    start_raw = row.get("oos_start")
    if start_raw not in {None, ""}:
        try:
            return int(pd.Timestamp(str(start_raw)).year)
        except (TypeError, ValueError):
            start_raw = None
    label = str(row.get("label") or "").strip()
    if len(label) >= 4 and label[:4].isdigit():
        return int(label[:4])
    return None


def _worst_full_year_return_from_oos_windows(rows, *, ret_key: str) -> float:
    yearly_growth = {}
    for row in list(rows or []):
        year = _resolve_oos_window_year(row)
        if year is None:
            continue
        yearly_growth.setdefault(int(year), 1.0)
        yearly_growth[int(year)] *= 1.0 + (_safe_float(row.get(ret_key, 0.0)) / 100.0)
    if not yearly_growth:
        return 0.0
    yearly_returns = [((growth - 1.0) * 100.0) for growth in yearly_growth.values()]
    return min(yearly_returns) if yearly_returns else 0.0


def _build_oos_metrics_from_report(*, report: dict | None, initial_capital: float) -> tuple[dict, dict, str]:
    windows = list((report or {}).get("windows") or [])
    total = build_oos_total_performance(windows, ret_key="ret_pct")
    benchmark_total = build_oos_total_performance(windows, ret_key="benchmark_return_pct")
    candidate_metrics = {
        "pf_return": float(total.get("linked_total_return_pct", 0.0)),
        "annual_return_pct": float(total.get("annualized_return_pct", 0.0)),
        "min_full_year_return_pct": float(_worst_full_year_return_from_oos_windows(windows, ret_key="ret_pct")),
        "pf_mdd": float(total.get("max_drawdown_pct", 0.0)),
        "pf_romd": _calc_romd(total.get("linked_total_return_pct", 0.0), total.get("max_drawdown_pct", 0.0)),
        "r_squared": float(_weighted_average_from_rows(windows, value_key="r_squared", weight_key="trade_count")),
        "m_win_rate": float(_weighted_average_from_rows(windows, value_key="monthly_win_rate", weight_key="trade_count")),
        "win_rate": float(_weighted_average_from_rows(windows, value_key="win_rate", weight_key="trade_count")),
        "pf_payoff": float(_weighted_average_from_rows(windows, value_key="pf_payoff", weight_key="trade_count")),
        "pf_ev": float(_weighted_average_from_rows(windows, value_key="pf_ev", weight_key="trade_count")),
        "pf_trades": int(sum(_safe_int(row.get("trade_count", 0)) for row in windows)),
        "normal_trades": int(sum(_safe_int(row.get("normal_trades", 0)) for row in windows)),
        "extended_trades": int(sum(_safe_int(row.get("extended_trades", 0)) for row in windows)),
        "missed_buys": int(sum(_safe_int(row.get("missed_buys", 0)) for row in windows)),
        "missed_sells": int(sum(_safe_int(row.get("missed_sells", 0)) for row in windows)),
        "missed_total": int(sum(_safe_int(row.get("missed_buys", 0)) + _safe_int(row.get("missed_sells", 0)) for row in windows)),
        "annual_trades": float(_weighted_average_from_rows(windows, value_key="annual_trades", weight_key="trade_count")),
        "reserved_buy_fill_rate": float(_weighted_average_from_rows(windows, value_key="reserved_buy_fill_rate", weight_key="trade_count")),
        "avg_exposure": float(_weighted_average_from_rows(windows, value_key="avg_exposure", weight_key="trade_count")),
        "final_equity": float(initial_capital) * float(total.get("ending_equity_factor", 1.0)),
    }
    benchmark_metrics = {
        "pf_return": float(benchmark_total.get("linked_total_return_pct", 0.0)),
        "annual_return_pct": float(benchmark_total.get("annualized_return_pct", 0.0)),
        "min_full_year_return_pct": float(_worst_full_year_return_from_oos_windows(windows, ret_key="benchmark_return_pct")),
        "pf_mdd": float(benchmark_total.get("max_drawdown_pct", 0.0)),
        "r_squared": float(_weighted_average_from_rows(windows, value_key="benchmark_r_squared", weight_key="trade_count")),
        "m_win_rate": float(_weighted_average_from_rows(windows, value_key="benchmark_monthly_win_rate", weight_key="trade_count")),
        "pf_romd": _calc_romd(benchmark_total.get("linked_total_return_pct", 0.0), benchmark_total.get("max_drawdown_pct", 0.0)),
        "final_equity": float(initial_capital) * float(benchmark_total.get("ending_equity_factor", 1.0)),
    }
    range_start = str(total.get("oos_start") or "")
    range_end = str(total.get("oos_end") or "")
    if range_start and range_end:
        range_text = f"{range_start} ~ {range_end}"
    else:
        range_text = "-"
    return candidate_metrics, benchmark_metrics, range_text


def _build_search_train_dates_for_session(session):
    sorted_dates = list(session.sorted_master_dates or [])
    if _resolve_model_mode(session.objective_mode) == "legacy":
        return sorted_dates
    return filter_search_train_dates(
        sorted_dates=sorted_dates,
        train_start_year=int(session.train_start_year),
        search_train_end_year=int(session.search_train_end_year),
    )


def _build_minimal_report_from_trial(trial):
    quality_status = str(trial.user_attrs.get("wf_quality_gate_status", "fail")).lower()
    coverage_status = str(trial.user_attrs.get("wf_coverage_gate_status", "watch")).lower()
    upgrade_status = str(trial.user_attrs.get("wf_upgrade_status", "fail")).lower()
    return {
        "summary": {
            "median_window_score": _safe_float(trial.user_attrs.get("wf_median_window_score", 0.0)),
            "worst_ret_pct": _safe_float(trial.user_attrs.get("wf_worst_ret_pct", 0.0)),
            "max_mdd": _safe_float(trial.user_attrs.get("wf_max_mdd", 0.0)),
        },
        "regime_summary": {
            "flat": {"median_score": _safe_float(trial.user_attrs.get("wf_flat_median_score", 0.0))},
            "down": {"window_count": _safe_int(trial.user_attrs.get("wf_down_window_count", 0))},
        },
        "upgrade_gate": {
            "status": upgrade_status,
            "quality_gate": {"status": quality_status},
            "coverage_gate": {"status": coverage_status},
        },
    }


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


def _compose_first_zone_cell(metric_name: str, base_text: str, numeric_value: float, *, delta_text: str = "", delta_value: float | None = None, use_blue: bool = False) -> str:
    if use_blue:
        rendered = _colorize(base_text, C_CYAN)
    else:
        rendered = _colorize(base_text, _first_zone_base_color(metric_name, numeric_value))
    if delta_text in {"", "-", None} or delta_value is None:
        return rendered
    return f"{rendered} {_colorize(delta_text, _delta_color(delta_value))}"


def _metric_candidate_color_for_gate(metric_name: str, candidate_value: float, passed: bool) -> str:
    if metric_name in {"最大視窗 MDD", "最大回撤 (MDD)"}:
        return _pass_color(passed, pass_color=C_YELLOW, fail_color=C_RED)
    if metric_name in {"視窗分數中位數", "flat 視窗中位分數"}:
        return _delta_color(candidate_value)
    return _pass_with_positive_color(passed, candidate_value)


def _status_candidate_color(status_text: str) -> str:
    normalized = str(status_text or "").strip().lower()
    if normalized == "pass":
        return C_GREEN
    if normalized == "watch":
        return C_YELLOW
    return C_RED




def _build_first_zone_rows(*, candidate_metrics: dict, champion_metrics: dict | None, benchmark_metrics: dict | None = None):
    champion_metrics = dict(champion_metrics or {})
    benchmark_metrics = dict(benchmark_metrics or {})

    def champ_value(key, default=None):
        return champion_metrics.get(key, default)

    def bench_value(key, default=None):
        return benchmark_metrics.get(key, default)

    rows = []

    def _append_row(name, candidate_text, candidate_numeric, *, champion_text="-", champion_numeric=None, champion_delta_text="", champion_delta_value=None, benchmark_text="-", benchmark_numeric=None, benchmark_delta_text="", benchmark_delta_value=None, use_blue=False, candidate_color_override=None):
        row = {
            "name": name,
            "candidate": _compose_first_zone_cell(name, candidate_text, float(candidate_numeric), use_blue=use_blue) if candidate_numeric is not None else str(candidate_text),
            "candidate_precolored": candidate_numeric is not None,
            "champion": _compose_first_zone_cell(name, champion_text, float(champion_numeric), delta_text=champion_delta_text, delta_value=champion_delta_value, use_blue=use_blue) if champion_numeric is not None else str(champion_text),
            "champion_precolored": champion_numeric is not None,
            "benchmark": _compose_first_zone_cell(name, benchmark_text, float(benchmark_numeric), delta_text=benchmark_delta_text, delta_value=benchmark_delta_value, use_blue=use_blue) if benchmark_numeric is not None else str(benchmark_text),
            "benchmark_precolored": benchmark_numeric is not None,
        }
        if candidate_color_override is not None:
            row["candidate"] = _colorize(str(candidate_text), candidate_color_override)
            row["candidate_precolored"] = True
        rows.append(row)

    def add_row(name, key, *, kind="pct", candidate_unit="", digits=2):
        candidate_value = candidate_metrics.get(key, 0.0)
        champion_value_raw = champ_value(key)
        benchmark_value_raw = bench_value(key)
        if benchmark_value_raw is None and key.startswith("bm_"):
            benchmark_value_raw = candidate_metrics.get(key)
        if kind == "pct":
            cand_plain = _format_pct_plain(candidate_value)
            bench_plain = _format_pct_plain(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            champ_plain = _format_pct_plain(champion_value_raw) if champion_value_raw is not None else "-"
            bench_delta_value = float(candidate_value) - float(benchmark_value_raw) if benchmark_value_raw is not None else None
            champ_delta_value = float(candidate_value) - float(champion_value_raw) if champion_value_raw is not None else None
            bench_delta_text = _format_pct_diff(bench_delta_value) if bench_delta_value is not None else ""
            champ_delta_text = _format_pct_diff(champ_delta_value) if champ_delta_value is not None else ""
        elif kind == "mdd":
            cand_plain = _format_mdd_plain(candidate_value)
            bench_plain = _format_mdd_plain(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            champ_plain = _format_mdd_plain(champion_value_raw) if champion_value_raw is not None else "-"
            bench_delta_value = float(benchmark_value_raw) - float(candidate_value) if benchmark_value_raw is not None else None
            champ_delta_value = float(champion_value_raw) - float(candidate_value) if champion_value_raw is not None else None
            bench_delta_text = _format_mdd_diff(candidate_value, benchmark_value_raw) if benchmark_value_raw is not None else ""
            champ_delta_text = _format_mdd_diff(candidate_value, champion_value_raw) if champion_value_raw is not None else ""
        elif kind in {"float2", "float3"}:
            digits = 2 if kind == "float2" else 3
            cand_plain = f"{float(candidate_value):.{digits}f}{candidate_unit}"
            bench_plain = f"{float(benchmark_value_raw):.{digits}f}{candidate_unit}" if benchmark_value_raw is not None else "-"
            champ_plain = f"{float(champion_value_raw):.{digits}f}{candidate_unit}" if champion_value_raw is not None else "-"
            bench_delta_value = float(candidate_value) - float(benchmark_value_raw) if benchmark_value_raw is not None else None
            champ_delta_value = float(candidate_value) - float(champion_value_raw) if champion_value_raw is not None else None
            bench_delta_text = _format_float_diff(bench_delta_value, digits, candidate_unit) if bench_delta_value is not None else ""
            champ_delta_text = _format_float_diff(champ_delta_value, digits, candidate_unit) if champ_delta_value is not None else ""
        elif kind == "count_split":
            cand_plain = f"{int(candidate_metrics.get('pf_trades', 0))} (正常:{int(candidate_metrics.get('normal_trades', 0))} | 延續:{int(candidate_metrics.get('extended_trades', 0))})"
            champ_plain = f"{int(champion_metrics.get('pf_trades', 0))} (正常:{int(champion_metrics.get('normal_trades', 0))} | 延續:{int(champion_metrics.get('extended_trades', 0))})" if champion_metrics else "-"
            _append_row(name, cand_plain, None, champion_text=champ_plain, champion_numeric=None, benchmark_text="-", benchmark_numeric=None)
            return
        elif kind == "missed_split":
            cand_plain = f"{int(candidate_metrics.get('missed_total', 0))} (買:{int(candidate_metrics.get('missed_buys', 0))} | 賣:{int(candidate_metrics.get('missed_sells', 0))})"
            champ_plain = f"{int(champion_metrics.get('missed_total', 0))} (買:{int(champion_metrics.get('missed_buys', 0))} | 賣:{int(champion_metrics.get('missed_sells', 0))})" if champion_metrics else "-"
            _append_row(name, cand_plain, None, champion_text=champ_plain, champion_numeric=None, benchmark_text="-", benchmark_numeric=None)
            return
        elif kind == "float2_nodiff":
            cand_plain = f"{float(candidate_value):.2f}{candidate_unit}"
            champ_plain = f"{float(champion_value_raw):.2f}{candidate_unit}" if champion_value_raw is not None else "-"
            _append_row(name, cand_plain, None, champion_text=champ_plain, champion_numeric=None, benchmark_text="-", benchmark_numeric=None)
            return
        elif kind == "money":
            cand_plain = _format_money(candidate_value)
            bench_plain = _format_money(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            champ_plain = _format_money(champion_value_raw) if champion_value_raw is not None else "-"
            bench_delta_value = float(candidate_value) - float(benchmark_value_raw) if benchmark_value_raw is not None else None
            champ_delta_value = float(candidate_value) - float(champion_value_raw) if champion_value_raw is not None else None
            bench_delta_text = _format_money_diff(bench_delta_value) if bench_delta_value is not None else ""
            champ_delta_text = _format_money_diff(champ_delta_value) if champ_delta_value is not None else ""
        else:
            cand_plain = str(candidate_value)
            bench_plain = str(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            champ_plain = str(champion_value_raw) if champion_value_raw is not None else "-"
            bench_delta_value = None
            champ_delta_value = None
            bench_delta_text = ""
            champ_delta_text = ""

        use_blue = name == "報酬回撤比 (RoMD)"
        _append_row(
            name,
            cand_plain,
            candidate_value,
            champion_text=champ_plain,
            champion_numeric=champion_value_raw,
            champion_delta_text=champ_delta_text,
            champion_delta_value=champ_delta_value,
            benchmark_text=bench_plain,
            benchmark_numeric=benchmark_value_raw,
            benchmark_delta_text=bench_delta_text,
            benchmark_delta_value=bench_delta_value,
            use_blue=use_blue,
        )

    add_row("總資產報酬率", "pf_return", kind="pct")
    add_row("年化報酬率", "annual_return_pct", kind="pct")
    add_row("年度最差報酬", "min_full_year_return_pct", kind="pct")
    add_row("報酬回撤比 (RoMD)", "pf_romd", kind="float2")
    add_row("最大回撤 (MDD)", "pf_mdd", kind="mdd")
    add_row("月度獲利勝率", "m_win_rate", kind="pct")
    add_row("系統實戰勝率", "win_rate", kind="pct")

    candidate_payoff = float(candidate_metrics.get("pf_payoff", 0.0))
    candidate_ev = float(candidate_metrics.get("pf_ev", 0.0))
    champion_payoff = champ_value("pf_payoff")
    champion_ev = champ_value("pf_ev")
    benchmark_payoff = bench_value("pf_payoff")
    benchmark_ev = bench_value("pf_ev")

    candidate_combo = f"{candidate_payoff:.2f} : {candidate_ev:.3f} R"
    if champion_payoff is not None and champion_ev is not None:
        champion_payoff = float(champion_payoff)
        champion_ev = float(champion_ev)
        payoff_diff = candidate_payoff - champion_payoff
        ev_diff = candidate_ev - champion_ev
        diff_text = f"({payoff_diff:+.2f} : {ev_diff:+.3f} R)"
        champion_combo = f"{champion_payoff:.2f} : {champion_ev:.3f} R {_colorize(diff_text, _delta_color(ev_diff))}"
    else:
        champion_combo = "-"
    if benchmark_payoff is not None and benchmark_ev is not None:
        benchmark_payoff = float(benchmark_payoff)
        benchmark_ev = float(benchmark_ev)
        payoff_diff = candidate_payoff - benchmark_payoff
        ev_diff = candidate_ev - benchmark_ev
        diff_text = f"({payoff_diff:+.2f} : {ev_diff:+.3f} R)"
        benchmark_combo = f"{benchmark_payoff:.2f} : {benchmark_ev:.3f} R {_colorize(diff_text, _delta_color(ev_diff))}"
    else:
        benchmark_combo = "-"
    _append_row("風報比 : 期望值", candidate_combo, None, champion_text=champion_combo, champion_numeric=None, benchmark_text=benchmark_combo, benchmark_numeric=None)

    add_row("總交易次數", "pf_trades", kind="count_split")
    add_row("錯失交易次數", "missed_total", kind="missed_split")
    add_row("年化交易次數", "annual_trades", kind="float2_nodiff")
    add_row("保留後買進成交率", "reserved_buy_fill_rate", kind="pct")
    add_row("平均資金水位", "avg_exposure", kind="pct")
    add_row("最終資產", "final_equity", kind="money")
    return rows


def _compose_compare_champion_cell(metric_name: str, champion_text: str, diff_text: str, diff_value: float | None) -> str:
    rendered = str(champion_text)
    if diff_text in {"", "-", None} or diff_value is None:
        return rendered
    return f"{rendered} {_colorize(diff_text, _delta_color(diff_value))}"


def _build_upgrade_rows(*, trial, policy):
    median_score = _safe_float(trial.user_attrs.get("wf_median_window_score", 0.0))
    worst_ret = _safe_float(trial.user_attrs.get("wf_worst_ret_pct", 0.0))
    flat_score = _safe_float(trial.user_attrs.get("wf_flat_median_score", 0.0))
    down_count = _safe_int(trial.user_attrs.get("wf_down_window_count", 0))
    quality_status = str(trial.user_attrs.get("wf_quality_gate_status", "fail")).upper()
    coverage_status = str(trial.user_attrs.get("wf_coverage_gate_status", "watch")).upper()
    gate_median = _safe_float(policy.get("gate_min_median_score", 0.0))
    gate_worst = _safe_float(policy.get("gate_min_worst_ret_pct", 0.0))
    gate_flat = _safe_float(policy.get("gate_min_flat_median_score", 0.0))
    rows = [
        {"name": "視窗分數中位數", "candidate": f"{median_score:.3f}", "threshold": f">= {gate_median:.3f}", "status": "PASS" if median_score >= gate_median else "FAIL", "candidate_numeric": median_score},
        {"name": "最差視窗報酬", "candidate": f"{worst_ret:.2f}%", "threshold": f">= {gate_worst:.2f}%", "status": "PASS" if worst_ret >= gate_worst else "FAIL", "candidate_numeric": worst_ret},
        {"name": "flat 視窗中位分數", "candidate": f"{flat_score:.3f}", "threshold": f">= {gate_flat:.3f}", "status": "PASS" if flat_score >= gate_flat else "FAIL", "candidate_numeric": flat_score},
        {"name": "down_regime_coverage", "candidate": str(down_count), "threshold": ">= 1", "status": "PASS" if down_count >= 1 else "FAIL", "candidate_numeric": float(down_count)},
        {"name": "quality_gate", "candidate": quality_status, "threshold": "必須 PASS", "status": quality_status, "candidate_numeric": None},
        {"name": "coverage_gate", "candidate": coverage_status, "threshold": "PASS / WATCH 可接受", "status": coverage_status, "candidate_numeric": None},
    ]
    for row in rows:
        status_norm = str(row["status"]).strip().upper()
        if row["name"] in {"quality_gate", "coverage_gate"}:
            row["candidate"] = _colorize(row["candidate"], _status_candidate_color(status_norm))
        else:
            row["candidate"] = _colorize(row["candidate"], _metric_candidate_color_for_gate(row["name"], row["candidate_numeric"], status_norm in {"PASS", "WATCH"}))
        row["candidate_precolored"] = True
    return rows


def _build_compare_rows(*, trial, champion_cache, policy):
    if not champion_cache or not champion_cache.get("wf_report"):
        return None
    champion_report = champion_cache["wf_report"]
    challenger_report = _build_minimal_report_from_trial(trial)
    assessment = build_compare_assessment(
        champion_report=champion_report,
        challenger_report=challenger_report,
        compare_worst_ret_tolerance_pct=_safe_float(policy.get("compare_worst_ret_tolerance_pct", 1.0)),
        compare_max_mdd_tolerance_pct=_safe_float(policy.get("compare_max_mdd_tolerance_pct", 2.0)),
    )
    champion_summary = dict((champion_report or {}).get("summary") or {})
    champion_regime = dict((champion_report or {}).get("regime_summary") or {})
    champion_flat = _safe_float((dict(champion_regime.get("flat") or {})).get("median_score", 0.0))
    candidate_median = _safe_float(trial.user_attrs.get("wf_median_window_score", 0.0))
    champion_median = _safe_float(champion_summary.get("median_window_score", 0.0))
    candidate_worst = _safe_float(trial.user_attrs.get("wf_worst_ret_pct", 0.0))
    champion_worst = _safe_float(champion_summary.get("worst_ret_pct", 0.0))
    candidate_flat = _safe_float(trial.user_attrs.get("wf_flat_median_score", 0.0))
    candidate_mdd = _safe_float(trial.user_attrs.get("wf_max_mdd", 0.0))
    champion_mdd = _safe_float(champion_summary.get("max_mdd", 0.0))
    worst_tol = _safe_float(policy.get("compare_worst_ret_tolerance_pct", 1.0))
    mdd_tol = _safe_float(policy.get("compare_max_mdd_tolerance_pct", 2.0))

    checks = {str(item.get("name")): str("PASS" if item.get("passed") else "FAIL") for item in assessment.get("checks") or []}
    median_diff = candidate_median - champion_median
    worst_diff = candidate_worst - champion_worst
    flat_diff = candidate_flat - champion_flat
    mdd_diff = champion_mdd - candidate_mdd
    rows = [
        {"name": "視窗分數中位數", "candidate": f"{candidate_median:.3f}", "champion": _compose_compare_champion_cell("視窗分數中位數", f"{champion_median:.3f}", _format_float_diff(median_diff, 3), median_diff), "threshold": "差異 > 0", "status": checks.get("median_window_score_vs_champion", "FAIL"), "candidate_numeric": candidate_median},
        {"name": "最差視窗報酬", "candidate": f"{candidate_worst:.2f}%", "champion": _compose_compare_champion_cell("最差視窗報酬", f"{champion_worst:.2f}%", _format_pct_diff(worst_diff), worst_diff), "threshold": f"差異 >= -{worst_tol:.1f}%", "status": checks.get("worst_ret_pct_vs_champion", "FAIL"), "candidate_numeric": candidate_worst},
        {"name": "flat 視窗中位分數", "candidate": f"{candidate_flat:.3f}", "champion": _compose_compare_champion_cell("flat 視窗中位分數", f"{champion_flat:.3f}", _format_float_diff(flat_diff, 3), flat_diff), "threshold": "差異 >= 0", "status": checks.get("flat_median_score_vs_champion", "FAIL"), "candidate_numeric": candidate_flat},
        {"name": "最大視窗 MDD", "candidate": f"-{abs(candidate_mdd):.2f}%", "champion": _compose_compare_champion_cell("最大視窗 MDD", f"-{abs(champion_mdd):.2f}%", _format_mdd_diff(candidate_mdd, champion_mdd), mdd_diff), "champion_precolored": True, "threshold": f"差異 >= -{mdd_tol:.1f}%", "status": checks.get("max_mdd_vs_champion", "FAIL"), "candidate_numeric": candidate_mdd},
        {"name": "compare_gate", "candidate": str(assessment.get("status", "fail")).upper(), "champion": "-", "threshold": "必須 PASS", "status": str(assessment.get("status", "fail")).upper(), "candidate_numeric": None},
    ]
    for row in rows:
        status_norm = str(row["status"]).strip().upper()
        if row["name"] == "compare_gate":
            row["candidate"] = _colorize(row["candidate"], _status_candidate_color(status_norm))
        else:
            row["candidate"] = _colorize(row["candidate"], _metric_candidate_color_for_gate(row["name"], row["candidate_numeric"], status_norm == "PASS"))
        row["candidate_precolored"] = True
        row["champion_precolored"] = True
    return rows


def _build_training_param_lines(params):
    bb_str = f"布林(BB) 啟用（長{get_p(params, 'bb_len', 20)}, 寬{get_p(params, 'bb_mult', 2.0):.1f}x）" if get_p(params, 'use_bb', False) else "布林(BB) 關閉"
    kc_str = f"阿肯那(KC) 啟用（長{get_p(params, 'kc_len', 20)}, 寬{get_p(params, 'kc_mult', 2.0):.1f}x）" if get_p(params, 'use_kc', False) else "阿肯那(KC) 關閉"
    vol_str = f"均量 啟用（短{get_p(params, 'vol_short_len', 5)} > 長{get_p(params, 'vol_long_len', 19)}）" if get_p(params, 'use_vol', False) else "均量 關閉"
    return [
        f"核心：突破 {get_p(params, 'high_len', 201)} 日新高｜ATR {get_p(params, 'atr_len', 14)} 日｜半倉停利 {get_p(params, 'tp_percent', 0.0) * 100:.1f}%",
        f"風控：掛單 +{get_p(params, 'atr_buy_tol', 1.5):.1f} ATR｜停損 -{get_p(params, 'atr_times_init', 2.0):.1f} ATR｜追蹤 -{get_p(params, 'atr_times_trail', 3.5):.1f} ATR",
        f"濾網：{bb_str}｜{kc_str}｜{vol_str}",
        f"歷史門檻：交易 >= {get_p(params, 'min_history_trades', 0)} 次｜勝率 >= {get_p(params, 'min_history_win_rate', 0.3) * 100:.1f}%｜EV >= {get_p(params, 'min_history_ev', 0.0):.2f} R",
    ]


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


def _compute_champion_console_cache(session):
    if not os.path.exists(CHAMPION_PARAMS_PATH):
        return None
    champion_params = load_params_from_json(CHAMPION_PARAMS_PATH)
    prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(champion_params, "optimizer_max_workers"))
    prep_result = prepare_trial_inputs(
        raw_data_cache=session.raw_data_cache,
        params=champion_params,
        default_max_workers=session.default_max_workers,
        executor_bundle=prep_executor_bundle,
        static_fast_cache=session.static_fast_cache,
        static_master_dates=session.master_dates,
    )
    sorted_dates = sorted(prep_result["master_dates"])
    search_train_dates = _build_search_train_dates_for_session(session)
    benchmark_data = prep_result["all_dfs_fast"].get("0050")
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
        missed_buys,
        missed_sells,
        r_sq,
        m_win_rate,
        bm_r_sq,
        bm_m_win_rate,
        normal_trades,
        extended_trades,
        annual_trades,
        reserved_buy_fill_rate,
        annual_return_pct,
        bm_annual_return_pct,
    ) = run_portfolio_timeline(
        prep_result["all_dfs_fast"],
        prep_result["all_trade_logs"],
        search_train_dates,
        session.train_start_year,
        champion_params,
        session.train_max_positions,
        session.train_enable_rotation,
        benchmark_ticker="0050",
        benchmark_data=benchmark_data,
        is_training=True,
        profile_stats=pf_profile,
        verbose=False,
    )
    cache = {
        "pf_return": ret_pct,
        "pf_mdd": mdd,
        "pf_trades": trade_count,
        "final_equity": final_eq,
        "avg_exposure": avg_exp,
        "bm_return": bm_ret,
        "bm_mdd": bm_mdd,
        "win_rate": win_rate,
        "pf_ev": pf_ev,
        "pf_payoff": pf_payoff,
        "missed_buys": int(missed_buys),
        "missed_sells": int(missed_sells),
        "missed_total": int(missed_buys) + int(missed_sells),
        "normal_trades": int(normal_trades),
        "extended_trades": int(extended_trades),
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "avg_exposure": avg_exp,
        "annual_return_pct": annual_return_pct,
        "bm_annual_return_pct": bm_annual_return_pct,
        "min_full_year_return_pct": float(pf_profile.get("min_full_year_return_pct", 0.0)),
        "bm_min_full_year_return_pct": float(pf_profile.get("bm_min_full_year_return_pct", 0.0)),
        "r_squared": r_sq,
        "m_win_rate": m_win_rate,
        "bm_r_squared": bm_r_sq,
        "bm_m_win_rate": bm_m_win_rate,
        "pf_romd": _calc_romd(ret_pct, mdd),
    }
    if _resolve_model_mode(session.objective_mode) in {"wf", "split"}:
        wf_policy = dict(session.walk_forward_policy)
        wf_report = evaluate_walk_forward(
            all_dfs_fast=prep_result["all_dfs_fast"],
            all_trade_logs=prep_result["all_trade_logs"],
            sorted_dates=sorted_dates,
            params=champion_params,
            max_positions=session.train_max_positions,
            enable_rotation=session.train_enable_rotation,
            benchmark_ticker="0050",
            train_start_year=int(wf_policy["train_start_year"]),
            min_train_years=int(wf_policy["min_train_years"]),
            test_window_months=int(wf_policy["test_window_months"]),
            regime_up_threshold_pct=float(wf_policy["regime_up_threshold_pct"]),
            regime_down_threshold_pct=float(wf_policy["regime_down_threshold_pct"]),
            min_window_bars=int(wf_policy["min_window_bars"]),
            gate_min_median_score=float(wf_policy["gate_min_median_score"]),
            gate_min_worst_ret_pct=float(wf_policy["gate_min_worst_ret_pct"]),
            gate_min_flat_median_score=float(wf_policy["gate_min_flat_median_score"]),
        )
        cache["wf_report"] = wf_report
    else:
        cache["wf_report"] = None
    return cache


def _get_champion_console_cache(session):
    cache = getattr(session, "_optimizer_console_champion_cache", None)
    if cache is None:
        cache = _compute_champion_console_cache(session)
        setattr(session, "_optimizer_console_champion_cache", cache)
    return cache




def _build_optimizer_trial_dashboard_payload(session, trial):
    attrs = trial.user_attrs
    params_mapping = session.build_optimizer_trial_params(trial.params, attrs, fixed_tp_percent=session.optimizer_fixed_tp_percent)
    params = _build_trial_params_object(params_mapping)
    mode_display = "關閉明牌（穩定鎖倉）" if not session.train_enable_rotation else "啟用 (汰弱換強)"
    model_mode = _resolve_model_mode(session.objective_mode)
    search_train_dates = _build_search_train_dates_for_session(session)
    latest_data_end = _latest_data_end_text(session)
    if model_mode == "wf":
        system_score_display = f"{_safe_float(attrs.get('wf_median_window_score', 0.0)):.3f}（WF中位分數）"
    elif model_mode == "split":
        system_score_display = f"{_safe_float(attrs.get('base_score', 0.0)):.3f}（Train RoMD）"
    else:
        system_score_display = f"{_safe_float(attrs.get('base_score', 0.0)):.2f}（base_score）"
    initial_capital = _safe_float(get_p(params, "initial_capital", 0.0))
    candidate_train_metrics = {
        "pf_return": _safe_float(attrs.get("pf_return", 0.0)),
        "annual_return_pct": _safe_float(attrs.get("annual_return_pct", 0.0)),
        "min_full_year_return_pct": _safe_float(attrs.get("min_full_year_return_pct", 0.0)),
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
        "pf_romd": _calc_romd(_safe_float(attrs.get("pf_return", 0.0)), _safe_float(attrs.get("pf_mdd", 0.0))),
    }
    benchmark_train_metrics = {
        "pf_return": _safe_float(attrs.get("bm_return", 0.0)),
        "annual_return_pct": _safe_float(attrs.get("bm_annual_return_pct", 0.0)),
        "min_full_year_return_pct": _safe_float(attrs.get("bm_min_full_year_return_pct", 0.0)),
        "pf_mdd": _safe_float(attrs.get("bm_mdd", 0.0)),
        "r_squared": _safe_float(attrs.get("bm_r_squared", 0.0)),
        "m_win_rate": _safe_float(attrs.get("bm_m_win_rate", 0.0)),
        "pf_romd": _calc_romd(_safe_float(attrs.get("bm_return", 0.0)), _safe_float(attrs.get("bm_mdd", 0.0))),
        "final_equity": _benchmark_final_equity(initial_capital, _safe_float(attrs.get("bm_return", 0.0))),
    }
    champion_cache = _get_champion_console_cache(session)
    training_title = f"【訓練期間績效對比｜{_range_text_from_dates(search_train_dates)}】"
    train_rows = _build_first_zone_rows(
        candidate_metrics=candidate_train_metrics,
        champion_metrics=champion_cache,
        benchmark_metrics=benchmark_train_metrics,
    )

    test_title = None
    test_rows = None
    if model_mode in {"wf", "split"}:
        prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(params, "optimizer_max_workers"))
        prep_result = prepare_trial_inputs(
            raw_data_cache=session.raw_data_cache,
            params=params,
            default_max_workers=session.default_max_workers,
            executor_bundle=prep_executor_bundle,
            static_fast_cache=session.static_fast_cache,
            static_master_dates=session.master_dates,
        )
        candidate_wf_report = evaluate_walk_forward(
            all_dfs_fast=prep_result["all_dfs_fast"],
            all_trade_logs=prep_result["all_trade_logs"],
            sorted_dates=sorted(prep_result["master_dates"]),
            params=params,
            max_positions=session.train_max_positions,
            enable_rotation=session.train_enable_rotation,
            benchmark_ticker="0050",
            train_start_year=int(session.walk_forward_policy["train_start_year"]),
            min_train_years=int(session.walk_forward_policy["min_train_years"]),
            test_window_months=int(session.walk_forward_policy["test_window_months"]),
            regime_up_threshold_pct=float(session.walk_forward_policy["regime_up_threshold_pct"]),
            regime_down_threshold_pct=float(session.walk_forward_policy["regime_down_threshold_pct"]),
            min_window_bars=int(session.walk_forward_policy["min_window_bars"]),
            gate_min_median_score=float(session.walk_forward_policy["gate_min_median_score"]),
            gate_min_worst_ret_pct=float(session.walk_forward_policy["gate_min_worst_ret_pct"]),
            gate_min_flat_median_score=float(session.walk_forward_policy["gate_min_flat_median_score"]),
        )
        candidate_test_metrics, benchmark_test_metrics, oos_range_text = _build_oos_metrics_from_report(
            report=candidate_wf_report,
            initial_capital=initial_capital,
        )
        champion_test_metrics = None
        if champion_cache and champion_cache.get("wf_report"):
            champion_test_metrics, _unused_bm, _unused_range = _build_oos_metrics_from_report(
                report=champion_cache.get("wf_report"),
                initial_capital=initial_capital,
            )
        test_title = f"【測試期間績效對比｜{oos_range_text}｜資料終點：{latest_data_end}】"
        test_rows = _build_first_zone_rows(
            candidate_metrics=candidate_test_metrics,
            champion_metrics=champion_test_metrics,
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
        "upgrade_rows": _build_upgrade_rows(trial=trial, policy=session.walk_forward_policy) if model_mode == "wf" else None,
        "compare_rows": _build_compare_rows(trial=trial, champion_cache=champion_cache, policy=session.walk_forward_policy) if model_mode == "wf" else None,
        "base_score": _safe_float(attrs.get("base_score", 0.0)),
    }


def print_optimizer_trial_milestone_dashboard(session, trial, *, milestone_title: str, title: str = "績效與風險對比表"):
    payload = _build_optimizer_trial_dashboard_payload(session, trial)
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
    )


def run_optimizer_monitoring_callback(session, study, trial):
    session.current_session_trial += 1
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    prep_mode = trial.user_attrs.get("prep_mode", "parallel")
    mode_suffix = " [fallback]" if prep_mode == "sequential_fallback" else ""

    if trial.value is None:
        state_name = getattr(trial.state, "name", str(trial.state))
        status_text, score_text = f"{session.colors['yellow']}{state_name}{mode_suffix}{session.colors['reset']}", "N/A"
    elif not is_qualified_trial_value(trial.value):
        fail_msg = trial.user_attrs.get("fail_reason", "策略無效")
        status_text, score_text = f"{session.colors['yellow']}淘汰 [{fail_msg}]{mode_suffix}{session.colors['reset']}", "N/A"
    else:
        status_text, score_text = f"{session.colors['green']}進化中{mode_suffix}{session.colors['reset']}", f"{trial.value:.3f}"

    total_trials_display = str(session.n_trials) if isinstance(session.n_trials, int) and session.n_trials > 0 else "?"
    print(
        f"\r{session.colors['gray']}⏳ [累積 {trial.number + 1:>4} | 本輪 {session.current_session_trial:>3}/{total_trials_display}] "
        f"耗時: {duration:>5.1f}s | 系統評分: {score_text:>7} | 狀態: {status_text}{session.colors['reset']}\033[K",
        end="",
        flush=True,
    )

    best_completed_trial = session.get_best_completed_trial_or_none(study)
    if (
        best_completed_trial is not None
        and best_completed_trial.number == trial.number
        and is_qualified_trial_value(trial.value)
    ):
        print()
        print_optimizer_trial_milestone_dashboard(
            session,
            trial,
            title="績效與風險對比表",
            milestone_title=f"🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試)",
        )
