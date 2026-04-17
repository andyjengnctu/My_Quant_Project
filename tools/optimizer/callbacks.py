import os

from core.buy_sort import get_buy_sort_title
from core.config import BUY_SORT_METHOD, SCORE_CALC_METHOD, SCORE_NUMERATOR_METHOD
from core.display_common import get_p
from core.model_paths import resolve_champion_params_path
from core.params_io import load_params_from_json
from core.portfolio_engine import run_portfolio_timeline
from core.strategy_params import build_runtime_param_raw_value
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
    is_qualified_trial_value,
)
from tools.optimizer.walk_forward import build_compare_assessment, evaluate_walk_forward


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


def _filter_search_train_dates(*, sorted_dates, train_start_year: int, search_train_end_year: int):
    filtered = []
    for raw_date in sorted_dates:
        year = int(getattr(raw_date, 'year', 0) or 0)
        if int(train_start_year) <= year <= int(search_train_end_year):
            filtered.append(raw_date)
    return filtered


def _resolve_model_mode(objective_mode: str) -> str:
    return "wf" if str(objective_mode) == OBJECTIVE_MODE_WF_GATE_MEDIAN else "legacy"


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


def _build_search_train_dates_for_session(session):
    sorted_dates = list(session.sorted_master_dates or [])
    if _resolve_model_mode(session.objective_mode) == "legacy":
        return sorted_dates
    return _filter_search_train_dates(
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


def _build_first_zone_rows(*, candidate_metrics: dict, champion_metrics: dict | None):
    benchmark_return = _safe_float(candidate_metrics.get("bm_return", 0.0))
    benchmark_annual_return = _safe_float(candidate_metrics.get("bm_annual_return_pct", 0.0))
    benchmark_worst_year = _safe_float(candidate_metrics.get("bm_min_full_year_return_pct", 0.0))
    benchmark_mdd = _safe_float(candidate_metrics.get("bm_mdd", 0.0))
    benchmark_romd = _calc_romd(benchmark_return, benchmark_mdd)
    benchmark_rsq = _safe_float(candidate_metrics.get("bm_r_squared", 0.0))
    benchmark_mwin = _safe_float(candidate_metrics.get("bm_m_win_rate", 0.0))
    initial_capital = _safe_float(candidate_metrics.get("initial_capital", 0.0))
    benchmark_final_eq = _benchmark_final_equity(initial_capital, benchmark_return)

    champion_metrics = dict(champion_metrics or {})

    def champ_value(key, default=None):
        return champion_metrics.get(key, default)

    rows = []

    def add_row(name, candidate_value, champion_value_raw=None, benchmark_value_raw=None, *, kind="pct", champion_better_high=True, candidate_unit=""):
        if kind == "pct":
            cand = _format_pct_plain(candidate_value)
            bench = _format_value_with_delta(_format_pct_plain(benchmark_value_raw), _format_pct_diff(float(candidate_value) - float(benchmark_value_raw))) if benchmark_value_raw is not None else "-"
            if champion_value_raw is None:
                champ = "-"
            else:
                champ = _format_value_with_delta(_format_pct_plain(champion_value_raw), _format_pct_diff(float(candidate_value) - float(champion_value_raw)))
        elif kind == "mdd":
            cand = _format_mdd_plain(candidate_value)
            bench = _format_value_with_delta(_format_mdd_plain(benchmark_value_raw), _format_mdd_diff(candidate_value, benchmark_value_raw)) if benchmark_value_raw is not None else "-"
            champ = _format_value_with_delta(_format_mdd_plain(champion_value_raw), _format_mdd_diff(candidate_value, champion_value_raw)) if champion_value_raw is not None else "-"
        elif kind == "float2":
            cand = f"{float(candidate_value):.2f}{candidate_unit}"
            bench = _format_value_with_delta(f"{float(benchmark_value_raw):.2f}{candidate_unit}", _format_float_diff(float(candidate_value) - float(benchmark_value_raw), 2, candidate_unit)) if benchmark_value_raw is not None else "-"
            champ = _format_value_with_delta(f"{float(champion_value_raw):.2f}{candidate_unit}", _format_float_diff(float(candidate_value) - float(champion_value_raw), 2, candidate_unit)) if champion_value_raw is not None else "-"
        elif kind == "float3":
            cand = f"{float(candidate_value):.3f}"
            bench = _format_value_with_delta(f"{float(benchmark_value_raw):.3f}", _format_float_diff(float(candidate_value) - float(benchmark_value_raw), 3)) if benchmark_value_raw is not None else "-"
            champ = _format_value_with_delta(f"{float(champion_value_raw):.3f}", _format_float_diff(float(candidate_value) - float(champion_value_raw), 3)) if champion_value_raw is not None else "-"
        elif kind == "count":
            cand = str(int(candidate_value))
            bench = "-"
            champ = _format_value_with_delta(str(int(champion_value_raw)), f"({int(candidate_value) - int(champion_value_raw):+d})") if champion_value_raw is not None else "-"
        elif kind == "money":
            cand = _format_money(candidate_value)
            bench = _format_value_with_delta(_format_money(benchmark_value_raw), _format_money_diff(float(candidate_value) - float(benchmark_value_raw))) if benchmark_value_raw is not None else "-"
            champ = _format_value_with_delta(_format_money(champion_value_raw), _format_money_diff(float(candidate_value) - float(champion_value_raw))) if champion_value_raw is not None else "-"
        else:
            cand = str(candidate_value)
            bench = str(benchmark_value_raw) if benchmark_value_raw is not None else "-"
            champ = str(champion_value_raw) if champion_value_raw is not None else "-"
        rows.append({"name": name, "candidate": cand, "champion": champ, "benchmark": bench})

    add_row("總資產報酬率", candidate_metrics["pf_return"], champ_value("pf_return"), benchmark_return, kind="pct")
    add_row("年化報酬率", candidate_metrics["annual_return_pct"], champ_value("annual_return_pct"), benchmark_annual_return, kind="pct")
    add_row("年度最差報酬", candidate_metrics["min_full_year_return_pct"], champ_value("min_full_year_return_pct"), benchmark_worst_year, kind="pct")
    add_row("最大回撤 (MDD)", candidate_metrics["pf_mdd"], champ_value("pf_mdd"), benchmark_mdd, kind="mdd")
    add_row("報酬回撤比 (RoMD)", candidate_metrics["pf_romd"], champ_value("pf_romd"), benchmark_romd, kind="float2")
    add_row("平滑度 (Log R²)", candidate_metrics["r_squared"], champ_value("r_squared"), benchmark_rsq, kind="float2")
    add_row("月度獲利勝率", candidate_metrics["m_win_rate"], champ_value("m_win_rate"), benchmark_mwin, kind="pct")
    add_row("系統實戰勝率", candidate_metrics["win_rate"], champ_value("win_rate"), None, kind="pct")
    add_row("盈虧因子", candidate_metrics["pf_payoff"], champ_value("pf_payoff"), None, kind="float2")
    add_row("實戰期望值 (EV)", candidate_metrics["pf_ev"], champ_value("pf_ev"), None, kind="float2", candidate_unit=" R")
    add_row("總交易次數", candidate_metrics["pf_trades"], champ_value("pf_trades"), None, kind="count")
    add_row("年化交易次數", candidate_metrics["annual_trades"], champ_value("annual_trades"), None, kind="float2")
    add_row("保留後買進成交率", candidate_metrics["reserved_buy_fill_rate"], champ_value("reserved_buy_fill_rate"), None, kind="pct")
    add_row("最終資產", candidate_metrics["final_equity"], champ_value("final_equity"), benchmark_final_eq, kind="money")
    add_row("平均資金水位", candidate_metrics["avg_exposure"], champ_value("avg_exposure"), None, kind="pct")
    return rows


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
    return [
        {"name": "視窗分數中位數", "candidate": f"{median_score:.3f}", "threshold": f">= {gate_median:.3f}", "status": "PASS" if median_score >= gate_median else "FAIL"},
        {"name": "最差視窗報酬", "candidate": f"{worst_ret:.2f}%", "threshold": f">= {gate_worst:.2f}%", "status": "PASS" if worst_ret >= gate_worst else "FAIL"},
        {"name": "flat 視窗中位分數", "candidate": f"{flat_score:.3f}", "threshold": f">= {gate_flat:.3f}", "status": "PASS" if flat_score >= gate_flat else "FAIL"},
        {"name": "down_regime_coverage", "candidate": str(down_count), "threshold": ">= 1", "status": "PASS" if down_count >= 1 else "FAIL"},
        {"name": "quality_gate", "candidate": quality_status, "threshold": "必須 PASS", "status": quality_status},
        {"name": "coverage_gate", "candidate": coverage_status, "threshold": "PASS / WATCH 可接受", "status": coverage_status},
    ]


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
    return [
        {"name": "視窗分數中位數", "candidate": f"{candidate_median:.3f}", "champion": _format_value_with_delta(f"{champion_median:.3f}", _format_float_diff(candidate_median - champion_median, 3)), "threshold": "差異 >= 0", "status": checks.get("median_window_score_vs_champion", "FAIL")},
        {"name": "最差視窗報酬", "candidate": f"{candidate_worst:.2f}%", "champion": _format_value_with_delta(f"{champion_worst:.2f}%", _format_pct_diff(candidate_worst - champion_worst)), "threshold": f"差異 >= -{worst_tol:.1f}%", "status": checks.get("worst_ret_pct_vs_champion", "FAIL")},
        {"name": "flat 視窗中位分數", "candidate": f"{candidate_flat:.3f}", "champion": _format_value_with_delta(f"{champion_flat:.3f}", _format_float_diff(candidate_flat - champion_flat, 3)), "threshold": "差異 >= 0", "status": checks.get("flat_median_score_vs_champion", "FAIL")},
        {"name": "最大視窗 MDD", "candidate": f"{candidate_mdd:.2f}%", "champion": _format_value_with_delta(f"{champion_mdd:.2f}%", _format_mdd_diff(candidate_mdd, champion_mdd)), "threshold": f"差異 >= -{mdd_tol:.1f}%", "status": checks.get("max_mdd_vs_champion", "FAIL")},
        {"name": "compare_gate", "candidate": str(assessment.get("status", "fail")).upper(), "champion": "-", "threshold": "必須 PASS", "status": str(assessment.get("status", "fail")).upper()},
    ]


def _build_training_param_lines(params):
    bb_str = f"布林(BB) 啟用（長{get_p(params, 'bb_len', 20)}, 寬{get_p(params, 'bb_mult', 2.0):.1f}x）" if get_p(params, 'use_bb', False) else "布林(BB) 關閉"
    kc_str = f"阿唐那(KC) 啟用（長{get_p(params, 'kc_len', 20)}, 寬{get_p(params, 'kc_mult', 2.0):.1f}x）" if get_p(params, 'use_kc', False) else "阿唐那(KC) 關閉"
    vol_str = "均量 啟用" if get_p(params, 'use_vol', False) else "均量 關閉"
    return [
        f"核心：突破 {get_p(params, 'high_len', 201)} 日新高｜ATR {get_p(params, 'atr_len', 14)} 日｜半倉停利 {get_p(params, 'tp_percent', 0.0) * 100:.1f}%  風控：掛單 +{get_p(params, 'atr_buy_tol', 1.5):.1f} ATR｜停損 -{get_p(params, 'atr_times_init', 2.0):.1f} ATR｜追蹤 -{get_p(params, 'atr_times_trail', 3.5):.1f} ATR",
        f"濾網：{bb_str}｜{kc_str}｜{vol_str}  歷史門檻：交易 >= {get_p(params, 'min_history_trades', 0)} 次｜勝率 >= {get_p(params, 'min_history_win_rate', 0.3) * 100:.1f}%｜EV >= {get_p(params, 'min_history_ev', 0.0):.2f} R",
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
        _missed_b,
        _missed_s,
        r_sq,
        m_win_rate,
        bm_r_sq,
        bm_m_win_rate,
        _normal_trades,
        _extended_trades,
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
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
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
    if _resolve_model_mode(session.objective_mode) == "wf":
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
        attrs = trial.user_attrs
        params = session.build_optimizer_trial_params(trial.params, attrs, fixed_tp_percent=session.optimizer_fixed_tp_percent)
        mode_display = "關閉明牌（穩定鎖倉)" if not session.train_enable_rotation else "啟用 (汰弱換強)"
        if not session.train_enable_rotation:
            mode_display = "關閉明牌（穩定鎖倉）"
        model_mode = _resolve_model_mode(session.objective_mode)
        search_train_dates = _build_search_train_dates_for_session(session)
        latest_data_end = _latest_data_end_text(session)
        if model_mode == "wf":
            wf_range_text = f"2020-01-01 ~ {latest_data_end}（6 個月 × {_safe_int(attrs.get('wf_window_count', 0))} 窗）"
            system_score_display = f"{_safe_float(attrs.get('wf_median_window_score', 0.0)):.3f}（WF中位分數）"
        else:
            wf_range_text = "未啟用"
            system_score_display = f"{_safe_float(attrs.get('base_score', 0.0)):.2f}（base_score）"
        candidate_metrics = {
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
            "annual_trades": _safe_float(attrs.get("annual_trades", 0.0)),
            "reserved_buy_fill_rate": _safe_float(attrs.get("reserved_buy_fill_rate", 0.0)),
            "final_equity": _safe_float(attrs.get("final_equity", 0.0)),
            "avg_exposure": _safe_float(attrs.get("avg_exposure", 0.0)),
            "bm_return": _safe_float(attrs.get("bm_return", 0.0)),
            "bm_annual_return_pct": _safe_float(attrs.get("bm_annual_return_pct", 0.0)),
            "bm_min_full_year_return_pct": _safe_float(attrs.get("bm_min_full_year_return_pct", 0.0)),
            "bm_mdd": _safe_float(attrs.get("bm_mdd", 0.0)),
            "bm_r_squared": _safe_float(attrs.get("bm_r_squared", 0.0)),
            "bm_m_win_rate": _safe_float(attrs.get("bm_m_win_rate", 0.0)),
            "initial_capital": _safe_float(get_p(params, "initial_capital", 0.0)),
        }
        candidate_metrics["pf_romd"] = _calc_romd(candidate_metrics["pf_return"], candidate_metrics["pf_mdd"])
        champion_cache = _get_champion_console_cache(session)
        first_zone_rows = _build_first_zone_rows(candidate_metrics=candidate_metrics, champion_metrics=champion_cache)
        upgrade_rows = _build_upgrade_rows(trial=trial, policy=session.walk_forward_policy) if model_mode == "wf" else None
        compare_rows = _build_compare_rows(trial=trial, champion_cache=champion_cache, policy=session.walk_forward_policy) if model_mode == "wf" else None
        print_optimizer_trial_console_dashboard(
            title="績效與風險對比表",
            milestone_title=f"🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試)",
            global_strategy_text=_build_global_strategy_text(),
            mode_display=mode_display,
            max_pos=session.train_max_positions,
            model_mode=model_mode,
            search_train_range_text=_range_text_from_dates(search_train_dates),
            wf_range_text=wf_range_text,
            data_end_text=latest_data_end,
            objective_mode=str(session.objective_mode),
            score_calc_method=SCORE_CALC_METHOD,
            score_numerator_method=SCORE_NUMERATOR_METHOD,
            base_score=_safe_float(attrs.get("base_score", 0.0)),
            system_score_display=system_score_display,
            first_zone_rows=first_zone_rows,
            upgrade_rows=upgrade_rows,
            compare_rows=compare_rows,
            params_lines=_build_training_param_lines(params),
            hard_gate_lines=_build_hard_gate_lines(),
        )
