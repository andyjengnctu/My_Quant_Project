from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from core.buy_sort import calc_buy_sort_value
from core.config import SCORE_CALC_METHOD, V16StrategyParams
from core.params_io import params_to_json_dict
from core.portfolio_stats import calc_portfolio_score
from tools.optimizer.study_utils import build_best_params_payload_from_trial, build_optimizer_trial_params
from tools.scanner.stock_processor import process_single_stock
from tools.validate.scanner_expectations import normalize_scanner_result

from .checks import add_check


class _FakeTrial:
    def __init__(self, params, user_attrs=None):
        self.params = dict(params)
        self.user_attrs = dict(user_attrs or {})


def _build_dummy_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07"],
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [101.0, 102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.5, 101.5, 102.5, 103.5],
            "Volume": [1000, 1100, 1200, 1300],
        }
    )


def validate_model_io_schema_case(base_params):
    case_id = "MODEL_IO_SCHEMA"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    default_payload = params_to_json_dict(V16StrategyParams())
    trial_params = {
        "high_len": 55,
        "atr_len": 12,
        "atr_buy_tol": 0.8,
        "atr_times_init": 1.7,
        "atr_times_trail": 2.3,
        "bb_len": 18,
        "bb_mult": 2.1,
        "kc_len": 20,
        "kc_mult": 1.8,
        "vol_short_len": 4,
        "vol_long_len": 10,
        "fixed_risk": 0.02,
        "min_history_trades": 5,
        "min_history_ev": 0.2,
        "min_history_win_rate": 0.35,
    }
    fake_trial = _FakeTrial(trial_params, user_attrs={"fixed_tp_percent": 0.25})
    best_params_payload = build_best_params_payload_from_trial(fake_trial, fixed_tp_percent=None)

    add_check(
        results,
        "strategy_schema",
        case_id,
        "optimizer_best_params_payload_keys_match_strategy_schema",
        sorted(default_payload.keys()),
        sorted(best_params_payload.keys()),
    )
    add_check(
        results,
        "strategy_schema",
        case_id,
        "optimizer_best_params_payload_excludes_runtime_only_fields",
        True,
        all(key not in best_params_payload for key in ("optimizer_max_workers", "scanner_max_workers")),
    )

    for field_name, default_value in default_payload.items():
        actual_value = best_params_payload[field_name]
        if isinstance(default_value, bool):
            expected = True
            actual = isinstance(actual_value, bool)
        elif isinstance(default_value, int) and not isinstance(default_value, bool):
            expected = True
            actual = isinstance(actual_value, int) and not isinstance(actual_value, bool)
        elif isinstance(default_value, float):
            expected = True
            actual = isinstance(actual_value, (int, float)) and not isinstance(actual_value, bool)
        elif default_value is None:
            expected = True
            actual = actual_value is None or isinstance(actual_value, int)
        else:
            expected = True
            actual = isinstance(actual_value, type(default_value))
        add_check(results, "strategy_schema", case_id, f"best_params_type::{field_name}", expected, actual)

    restored_trial_params = build_optimizer_trial_params(fake_trial.params, fake_trial.user_attrs, fixed_tp_percent=None)
    add_check(
        results,
        "strategy_schema",
        case_id,
        "optimizer_trial_params_contains_fixed_tp_percent",
        True,
        math.isclose(restored_trial_params["tp_percent"], 0.25, rel_tol=0.0, abs_tol=1e-9),
    )

    with TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "2330.csv"
        _build_dummy_ohlcv_df().to_csv(file_path, index=False)
        sanitize_stats = {
            "invalid_row_count": 0,
            "duplicate_date_count": 0,
            "dropped_row_count": 0,
            "negative_volume_corrected_count": 0,
        }
        dummy_df = _build_dummy_ohlcv_df()

        buy_stats = {
            "is_candidate": True,
            "is_setup_today": True,
            "buy_limit": 105.0,
            "stop_loss": 99.0,
            "expected_value": 1.25,
            "win_rate": 55.0,
            "trade_count": 12,
            "max_drawdown": 9.5,
            "extended_candidate_today": None,
        }
        candidate_stats = {
            "is_candidate": True,
            "is_setup_today": False,
            "buy_limit": 105.0,
            "stop_loss": 99.0,
            "expected_value": 0.8,
            "win_rate": 52.0,
            "trade_count": 8,
            "max_drawdown": 8.0,
            "extended_candidate_today": None,
        }
        skip_exc = ValueError("有效資料不足: 2330")

        with patch("tools.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "tools.scanner.stock_processor.run_v16_backtest", return_value=buy_stats
        ):
            buy_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", base_params))

        expected_keys = ["expected_value", "message", "proj_cost", "sanitize_issue", "sort_value", "status", "ticker"]
        add_check(results, "strategy_schema", case_id, "scanner_buy_result_keys", expected_keys, sorted(buy_result.keys()))
        add_check(results, "strategy_schema", case_id, "scanner_buy_status_type", True, isinstance(buy_result["status"], str))
        add_check(results, "strategy_schema", case_id, "scanner_buy_proj_cost_finite", True, math.isfinite(float(buy_result["proj_cost"])))
        add_check(results, "strategy_schema", case_id, "scanner_buy_expected_value_finite", True, math.isfinite(float(buy_result["expected_value"])))
        add_check(results, "strategy_schema", case_id, "scanner_buy_sort_value_finite", True, math.isfinite(float(buy_result["sort_value"])))
        add_check(results, "strategy_schema", case_id, "scanner_buy_message_contains_ticker", True, "2330" in str(buy_result["message"]))
        add_check(results, "strategy_schema", case_id, "scanner_buy_sanitize_issue_nullable", True, buy_result["sanitize_issue"] is None or isinstance(buy_result["sanitize_issue"], str))

        with patch("tools.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "tools.scanner.stock_processor.run_v16_backtest", return_value=candidate_stats
        ):
            candidate_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", base_params))
        add_check(results, "strategy_schema", case_id, "scanner_candidate_status", "candidate", candidate_result["status"])
        add_check(results, "strategy_schema", case_id, "scanner_candidate_proj_cost_none", None, candidate_result["proj_cost"])
        add_check(results, "strategy_schema", case_id, "scanner_candidate_expected_value_none", None, candidate_result["expected_value"])
        add_check(results, "strategy_schema", case_id, "scanner_candidate_sort_value_none", None, candidate_result["sort_value"])

        with patch("tools.scanner.stock_processor.sanitize_ohlcv_dataframe", side_effect=skip_exc):
            skip_result = process_single_stock(str(file_path), "2330", base_params)
        normalized_skip = normalize_scanner_result(skip_result)
        add_check(results, "strategy_schema", case_id, "scanner_skip_status", "skip_insufficient", normalized_skip["status"])
        add_check(results, "strategy_schema", case_id, "scanner_skip_message_none", None, normalized_skip["message"])

    summary["best_params_key_count"] = len(best_params_payload)
    return results, summary


def validate_ranking_scoring_sanity_case(base_params):
    case_id = "RANKING_SCORING_SANITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    ev_low = calc_buy_sort_value("EV", 0.5, 10000, 0.4, 10)
    ev_high = calc_buy_sort_value("EV", 1.5, 10000, 0.4, 10)
    add_check(results, "strategy_score", case_id, "buy_sort_ev_monotonic", True, ev_high > ev_low)
    add_check(results, "strategy_score", case_id, "buy_sort_ev_type", True, isinstance(ev_high, float))
    add_check(results, "strategy_score", case_id, "buy_sort_ev_finite", True, math.isfinite(ev_high))

    cost_low = calc_buy_sort_value("PROJ_COST", 0.5, 5000, 0.4, 10)
    cost_high = calc_buy_sort_value("PROJ_COST", 0.5, 9000, 0.4, 10)
    add_check(results, "strategy_score", case_id, "buy_sort_proj_cost_monotonic", True, cost_high > cost_low)

    hist_low = calc_buy_sort_value("HIST_WIN_X_TRADES", 0.5, 10000, 0.35, 8)
    hist_high = calc_buy_sort_value("HIST_WIN_X_TRADES", 0.5, 10000, 0.45, 9)
    add_check(results, "strategy_score", case_id, "buy_sort_hist_win_x_trades_monotonic", True, hist_high > hist_low)

    with patch("core.config.SCORE_CALC_METHOD", "RoMD"):
        score_low_return = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=10.0)
        score_high_return = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=20.0)
        score_worse_mdd = calc_portfolio_score(sys_ret=10.0, sys_mdd=-30.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=20.0)
    add_check(results, "strategy_score", case_id, "portfolio_score_return_monotonic", True, score_high_return > score_low_return)
    add_check(results, "strategy_score", case_id, "portfolio_score_mdd_monotonic", True, score_high_return > score_worse_mdd)
    add_check(results, "strategy_score", case_id, "portfolio_score_romd_finite", True, math.isfinite(score_high_return))

    with patch("core.config.SCORE_CALC_METHOD", "LOG_R2"):
        score_low_quality = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=40.0, r_sq=0.4, annual_return_pct=20.0)
        score_high_quality = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=60.0, r_sq=0.9, annual_return_pct=20.0)
    add_check(results, "strategy_score", case_id, "portfolio_score_log_r2_quality_monotonic", True, score_high_quality > score_low_quality)
    add_check(results, "strategy_score", case_id, "portfolio_score_log_r2_finite", True, math.isfinite(score_high_quality))

    with TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "2330.csv"
        _build_dummy_ohlcv_df().to_csv(file_path, index=False)
        sanitize_stats = {
            "invalid_row_count": 0,
            "duplicate_date_count": 0,
            "dropped_row_count": 0,
            "negative_volume_corrected_count": 0,
        }
        dummy_df = _build_dummy_ohlcv_df()
        buy_stats = {
            "is_candidate": True,
            "is_setup_today": True,
            "buy_limit": 105.0,
            "stop_loss": 99.0,
            "expected_value": 1.25,
            "win_rate": 55.0,
            "trade_count": 12,
            "max_drawdown": 9.5,
            "extended_candidate_today": None,
        }
        with patch("tools.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "tools.scanner.stock_processor.run_v16_backtest", return_value=buy_stats
        ):
            buy_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", base_params))
        from tools.scanner.stock_processor import BUY_SORT_METHOD as ACTIVE_BUY_SORT_METHOD
        expected_sort = calc_buy_sort_value(
            ACTIVE_BUY_SORT_METHOD,
            buy_stats["expected_value"],
            buy_result["proj_cost"],
            buy_stats["win_rate"] / 100.0,
            buy_stats["trade_count"],
        )
        add_check(results, "strategy_score", case_id, "scanner_sort_value_matches_buy_sort_formula", expected_sort, buy_result["sort_value"])
        add_check(results, "strategy_score", case_id, "scanner_sort_value_comparable", True, isinstance(buy_result["sort_value"], float) and math.isfinite(float(buy_result["sort_value"])))

    summary["score_calc_method_default"] = SCORE_CALC_METHOD
    return results, summary
