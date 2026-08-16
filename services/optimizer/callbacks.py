import contextlib
import io
import os
import time

import pandas as pd

from core.config import (
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
from core.report_style import SIGNAL_NEGATIVE, SIGNAL_POSITIVE, signal_for_signed_value, terminal_signal
from core.runtime_utils import stdout_supports_inline_progress, write_inline_progress
from core.strategy_params import V16StrategyParams, build_runtime_param_raw_value
from core.strategy_dashboard import (
    build_optimizer_dashboard_metric_rows,
    format_global_strategy_text,
    format_hard_gate_lines,
    format_training_param_lines,
    print_optimizer_trial_console_dashboard,
)
from services.optimizer.prep import prepare_trial_inputs
from services.optimizer.study_utils import (
    OBJECTIVE_MODE_SPLIT_TRAIN_ROMD,
    is_qualified_trial_value,
    normalize_objective_mode,
)
from services.optimizer.walk_forward import build_test_holdout_period, build_test_period_metrics, evaluate_walk_forward


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


def _study_single_stock_breakout_stats_title(session) -> str | None:
    policy = dict(getattr(session, "walk_forward_policy", {}) or {})
    scope = str(policy.get("evaluation_scope") or "").strip().lower()
    if scope.startswith("study_full"):
        return "【Study-Full 單股突破統計】"
    if scope == "study_single_seed" or scope.startswith("study_oos") or scope.startswith("study-oos"):
        return "【Study-OOS 單股突破統計｜Train Period】"
    return None


def _format_signed_r(value, *, digits: int = 3, large: bool = False) -> str:
    numeric = _safe_float(value, 0.0)
    if large:
        return f"{numeric:+,.1f} R"
    return f"{numeric:+.{int(digits)}f} R"


def _colorize_signed_number(text: str, value) -> str:
    return terminal_signal(str(text), signal_for_signed_value(_safe_float(value, 0.0)), enabled=True)


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
        "win_rate": terminal_signal(
            win_rate_text,
            SIGNAL_POSITIVE if win_rate >= MIN_TRADE_WIN_RATE else SIGNAL_NEGATIVE,
            enabled=True,
        ),
        "payoff": f"{C_CYAN}{payoff:.2f}{C_RESET}",
        "avg_r": _colorize_signed_number(_format_signed_r(avg_r), avg_r),
        "median_r": _colorize_signed_number(_format_signed_r(median_r), median_r),
        "total_r": _colorize_signed_number(_format_signed_r(total_r, large=True), total_r),
    }


def _score_total_r_from_attrs(attrs: dict) -> float:
    return _safe_float(attrs.get("score_total_r", attrs.get("single_stock_total_r", 0.0)))


def _score_median_r_from_attrs(attrs: dict) -> float:
    return _safe_float(attrs.get("score_median_r", attrs.get("single_stock_median_r", 0.0)))


def _score_total_r_from_profile(profile: dict) -> float:
    return _safe_float(profile.get("score_total_r", profile.get("single_stock_total_r", 0.0)))


def _score_median_r_from_profile(profile: dict) -> float:
    return _safe_float(profile.get("score_median_r", profile.get("single_stock_median_r", 0.0)))


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
        "min_month_return_pct": float(total.get("min_month_return_pct", 0.0)),
        "min_quarter_return_pct": float(total.get("min_quarter_return_pct", 0.0)),
        "pf_mdd": float(total.get("max_drawdown_pct", 0.0)),
        "pf_romd": float(total.get("test_score_romd", 0.0)),
        "r_squared": float(total.get("r_squared", 0.0)),
        "m_win_rate": float(total.get("monthly_win_rate", 0.0)),
        "win_rate": float(total.get("win_rate", 0.0)),
        "pf_payoff": float(total.get("payoff", 0.0)),
        "pf_ev": float(total.get("ev", 0.0)),
        "score_total_r": float(total.get("score_total_r", total.get("single_stock_total_r", 0.0))),
        "score_median_r": float(total.get("score_median_r", total.get("single_stock_median_r", 0.0))),
        "score_r_source": str(total.get("score_r_source", "single_stock")),
        "pf_trades": int(total.get("trade_count", 0)),
        "normal_trades": int(total.get("normal_trades", 0)),
        "extended_trades": int(total.get("extended_trades", 0)),
        "breakout_trades": int(total.get("breakout_trades", total.get("normal_trades", 0))),
        "reentry_trades": int(total.get("reentry_trades", 0)),
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
        "min_month_return_pct": float(total.get("benchmark_min_month_return_pct", 0.0)),
        "min_quarter_return_pct": float(total.get("benchmark_min_quarter_return_pct", 0.0)),
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
    return format_global_strategy_text()


def _calc_romd(ret_pct: float, mdd_pct: float) -> float:
    mdd_pct = abs(float(mdd_pct))
    if mdd_pct <= 0:
        return 0.0
    return float(ret_pct) / (mdd_pct + 0.0001)


def _benchmark_final_equity(initial_capital: float, bm_return_pct: float) -> float:
    return float(initial_capital) * (1.0 + float(bm_return_pct) / 100.0)


def _optimizer_train_score_display_label() -> str:
    method = str(SCORE_CALC_METHOD or "").strip() or "Score"
    numerator = str(SCORE_NUMERATOR_METHOD or "").strip() or "TOTAL_RETURN"
    return f"Train Score[{method}/{numerator}]"



def _build_first_zone_rows(*, candidate_metrics: dict, reference_metrics: dict | None, benchmark_metrics: dict | None = None):
    return build_optimizer_dashboard_metric_rows(
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        benchmark_metrics=benchmark_metrics,
    )


def _build_training_param_lines(params, entry_trade_counts=None):
    return format_training_param_lines(params, entry_trade_counts=entry_trade_counts)


def _build_hard_gate_lines():
    return format_hard_gate_lines()




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
    score_total_r = _score_total_r_from_profile(profile)
    score_median_r = _score_median_r_from_profile(profile)
    candidate_score = calc_portfolio_score(
        total_return,
        max_drawdown,
        monthly_win_rate,
        r_squared,
        annual_return_pct=annual_return_pct,
        trade_win_rate_pct=win_rate,
        min_full_year_return_pct=_safe_float(profile.get("min_full_year_return_pct", 0.0)),
        min_month_return_pct=_safe_float(profile.get("min_month_return_pct", 0.0)),
        min_quarter_return_pct=_safe_float(profile.get("min_quarter_return_pct", 0.0)),
        total_r=score_total_r,
        median_r=score_median_r,
    )
    benchmark_score = calc_plain_romd(benchmark_return, benchmark_mdd)
    candidate_metrics = {
        "pf_return": total_return,
        "annual_return_pct": annual_return_pct,
        "min_full_year_return_pct": _safe_float(profile.get("min_full_year_return_pct", 0.0)),
        "min_month_return_pct": _safe_float(profile.get("min_month_return_pct", 0.0)),
        "min_quarter_return_pct": _safe_float(profile.get("min_quarter_return_pct", 0.0)),
        "pf_mdd": max_drawdown,
        "pf_romd": float(candidate_score),
        "r_squared": r_squared,
        "m_win_rate": monthly_win_rate,
        "win_rate": win_rate,
        "pf_payoff": pf_payoff,
        "pf_ev": pf_ev,
        "pf_total_r": portfolio_total_r,
        "pf_median_r": portfolio_median_r,
        "score_total_r": score_total_r,
        "score_median_r": score_median_r,
        "score_r_source": str(profile.get("score_r_source", "single_stock")),
        "pf_trades": trade_count,
        "normal_trades": _safe_int(profile.get("normal_trades", normal_trade_count)),
        "extended_trades": _safe_int(profile.get("extended_trades", extended_trade_count)),
        "breakout_trades": _safe_int(profile.get("breakout_trades", profile.get("normal_trades", normal_trade_count))),
        "reentry_trades": _safe_int(profile.get("reentry_trades", 0)),
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
        "min_month_return_pct": _safe_float(profile.get("bm_min_month_return_pct", 0.0)),
        "min_quarter_return_pct": _safe_float(profile.get("bm_min_quarter_return_pct", 0.0)),
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
    from services.portfolio_replay import run_portfolio_simulation_with_param_ensemble
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
            "min_month_return_pct": float(pf_profile.get("min_month_return_pct", 0.0)),
            "min_quarter_return_pct": float(pf_profile.get("min_quarter_return_pct", 0.0)),
            "pf_total_r": float(pf_profile.get("portfolio_total_r", 0.0)),
            "pf_median_r": float(pf_profile.get("portfolio_median_r", 0.0)),
            "score_total_r": _score_total_r_from_profile(pf_profile),
            "score_median_r": _score_median_r_from_profile(pf_profile),
            "score_r_source": str(pf_profile.get("score_r_source", "single_stock")),
            "pf_mdd": float(mdd),
            "r_squared": float(r_sq),
            "m_win_rate": float(m_win_rate),
            "win_rate": float(win_rate),
            "pf_payoff": float(pf_payoff),
            "pf_ev": float(pf_ev),
            "pf_trades": int(trade_count),
            "normal_trades": _safe_int(pf_profile.get("normal_trades", normal_trade_count)),
            "extended_trades": _safe_int(pf_profile.get("extended_trades", extended_trade_count)),
            "breakout_trades": _safe_int(pf_profile.get("breakout_trades", pf_profile.get("normal_trades", normal_trade_count))),
            "reentry_trades": _safe_int(pf_profile.get("reentry_trades", 0)),
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
                min_month_return_pct=float(pf_profile.get("min_month_return_pct", 0.0)),
                min_quarter_return_pct=float(pf_profile.get("min_quarter_return_pct", 0.0)),
                total_r=_score_total_r_from_profile(pf_profile),
                median_r=_score_median_r_from_profile(pf_profile),
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
        "min_month_return_pct": _safe_float(attrs.get("min_month_return_pct", 0.0)),
        "min_quarter_return_pct": _safe_float(attrs.get("min_quarter_return_pct", 0.0)),
        "pf_total_r": _safe_float(attrs.get("pf_total_r", 0.0)),
        "pf_median_r": _safe_float(attrs.get("pf_median_r", 0.0)),
        "score_total_r": _score_total_r_from_attrs(attrs),
        "score_median_r": _score_median_r_from_attrs(attrs),
        "score_r_source": str(attrs.get("score_r_source", "single_stock") or "single_stock"),
        "pf_mdd": _safe_float(attrs.get("pf_mdd", 0.0)),
        "r_squared": _safe_float(attrs.get("r_squared", 0.0)),
        "m_win_rate": _safe_float(attrs.get("m_win_rate", 0.0)),
        "win_rate": _safe_float(attrs.get("win_rate", 0.0)),
        "pf_payoff": _safe_float(attrs.get("pf_payoff", 0.0)),
        "pf_ev": _safe_float(attrs.get("pf_ev", 0.0)),
        "pf_trades": _safe_int(attrs.get("pf_trades", 0)),
        "normal_trades": _safe_int(attrs.get("normal_trades", attrs.get("pf_trades", 0))),
        "extended_trades": _safe_int(attrs.get("extended_trades", 0)),
        "breakout_trades": _safe_int(attrs.get("breakout_trades", attrs.get("normal_trades", attrs.get("pf_trades", 0)))),
        "reentry_trades": _safe_int(attrs.get("reentry_trades", 0)),
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
            min_month_return_pct=_safe_float(attrs.get("min_month_return_pct", 0.0)),
            min_quarter_return_pct=_safe_float(attrs.get("min_quarter_return_pct", 0.0)),
            total_r=_score_total_r_from_attrs(attrs),
            median_r=_score_median_r_from_attrs(attrs),
        )),
    }
    benchmark_train_metrics = {
        "pf_return": _safe_float(attrs.get("bm_return", 0.0)),
        "annual_return_pct": _safe_float(attrs.get("bm_annual_return_pct", 0.0)),
        "min_full_year_return_pct": _safe_float(attrs.get("bm_min_full_year_return_pct", 0.0)),
        "min_month_return_pct": _safe_float(attrs.get("bm_min_month_return_pct", 0.0)),
        "min_quarter_return_pct": _safe_float(attrs.get("bm_min_quarter_return_pct", 0.0)),
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
            min_month_return_pct=_safe_float(attrs.get("bm_min_month_return_pct", 0.0)),
            min_quarter_return_pct=_safe_float(attrs.get("bm_min_quarter_return_pct", 0.0)),
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

    study_breakout_stats_title = _study_single_stock_breakout_stats_title(session)
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
        "study_full_breakout_stats": _build_study_full_breakout_stats(attrs) if study_breakout_stats_title else None,
        "study_full_breakout_stats_title": study_breakout_stats_title,
        "base_score": _safe_float(attrs.get("base_score", 0.0)),
        "entry_trade_counts": {
            "breakout_trades": candidate_train_metrics.get("breakout_trades"),
            "extended_trades": candidate_train_metrics.get("extended_trades"),
            "reentry_trades": candidate_train_metrics.get("reentry_trades"),
        },
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
        params_lines=_build_training_param_lines(payload["params"], entry_trade_counts=payload.get("entry_trade_counts")),
        hard_gate_lines=_build_hard_gate_lines(),
        study_full_breakout_stats=payload.get("study_full_breakout_stats"),
        study_full_breakout_stats_title=payload.get("study_full_breakout_stats_title"),
    )
    render_elapsed = max(0.0, time.perf_counter() - render_started_at)
    return {
        "payload_sec": float(payload_elapsed),
        "candidate_wf_sec": float(timing_breakdown.get("candidate_wf_sec", 0.0)),
        "render_sec": float(render_elapsed),
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
