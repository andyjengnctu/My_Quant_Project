from __future__ import annotations

import io
import json
import math
from pathlib import Path
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.buy_sort import calc_buy_sort_value
from core.strategy_dashboard import print_strategy_dashboard
from core.config import SCORE_CALC_METHOD, V16StrategyParams
from core.params_io import params_to_json_dict
from core.portfolio_stats import calc_portfolio_score
from tools.optimizer.objective_runner import run_optimizer_objective
from tools.portfolio_sim.reporting import print_yearly_return_report
from tools.optimizer.runtime import export_best_params_if_requested
from tools.optimizer.study_utils import INVALID_TRIAL_VALUE, build_best_params_payload_from_trial, build_optimizer_trial_params
from tools.scanner.reporting import print_scanner_summary
from tools.scanner.stock_processor import process_single_stock
from tools.validate.scanner_expectations import normalize_scanner_result

from .checks import add_check


class _FakeTrial:
    def __init__(self, params, user_attrs=None):
        self.params = dict(params)
        self.user_attrs = dict(user_attrs or {})


class _FakeOptunaTrial:
    def __init__(self, *, number=0, preset_values=None, user_attrs=None, value=None):
        self.number = number
        self._preset_values = dict(preset_values or {})
        self.params = {}
        self.user_attrs = dict(user_attrs or {})
        self.value = value

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def _resolve_value(self, name, default_value):
        value = self._preset_values.get(name, default_value)
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = self._resolve_value(name, choices[0])
        if value not in choices:
            raise ValueError(f"{name} 不在 choices 範圍內: {value}")
        return value

    def suggest_int(self, name, low, high, step=1):
        value = int(self._resolve_value(name, low))
        if value < low or value > high or ((value - low) % step) != 0:
            raise ValueError(f"{name} 超出合法整數範圍: {value}")
        return value

    def suggest_float(self, name, low, high, step=None):
        value = float(self._resolve_value(name, low))
        if value < low or value > high:
            raise ValueError(f"{name} 超出合法浮點範圍: {value}")
        if step is not None:
            scaled = round((value - low) / step)
            reconstructed = low + scaled * step
            if not math.isclose(value, reconstructed, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{name} 不符合 step: {value}")
        return value


class _FakeProfileRecorder:
    def __init__(self):
        self.rows = []
        self.enabled = False
        self.console_print = False
        self.print_every_n_trials = 999999

    def append_row(self, row):
        self.rows.append(dict(row))


class _FakeOptimizerSession:
    def __init__(self, *, fixed_tp_percent=0.25):
        self.raw_data_cache = {}
        self.default_max_workers = 1
        self.train_start_year = 2020
        self.train_max_positions = 3
        self.train_enable_rotation = False
        self.optimizer_high_len_min = 20
        self.optimizer_high_len_max = 250
        self.optimizer_high_len_step = 5
        self.optimizer_fixed_tp_percent = fixed_tp_percent
        self.profile_recorder = _FakeProfileRecorder()
        self.recorded_prep_failures = []

    def record_optimizer_prep_failures(self, failures):
        self.recorded_prep_failures.extend(list(failures))

    def resolve_optimizer_tp_percent(self, trial, fixed_tp_percent):
        if fixed_tp_percent is None:
            return trial.suggest_float("tp_percent", 0.0, 0.6, step=0.01)
        trial.set_user_attr("fixed_tp_percent", float(fixed_tp_percent))
        return float(fixed_tp_percent)


class _FakeStudy:
    def __init__(self, trials, *, best_trial=None, best_trial_error=False):
        self.trials = list(trials)
        self._best_trial = best_trial
        self._best_trial_error = best_trial_error

    @property
    def best_trial(self):
        if self._best_trial_error:
            raise ValueError("best_trial unavailable")
        if self._best_trial is None:
            raise ValueError("best_trial unavailable")
        return self._best_trial


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
            buy_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))

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
            candidate_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))
        add_check(results, "strategy_schema", case_id, "scanner_candidate_status", "candidate", candidate_result["status"])
        add_check(results, "strategy_schema", case_id, "scanner_candidate_proj_cost_none", None, candidate_result["proj_cost"])
        add_check(results, "strategy_schema", case_id, "scanner_candidate_expected_value_none", None, candidate_result["expected_value"])
        add_check(results, "strategy_schema", case_id, "scanner_candidate_sort_value_none", None, candidate_result["sort_value"])

        with patch("tools.scanner.stock_processor.sanitize_ohlcv_dataframe", side_effect=skip_exc):
            skip_result = process_single_stock(str(file_path), "2330", params)
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
            buy_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))
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


def validate_strategy_repeatability_case(base_params):
    params = base_params if base_params is not None else V16StrategyParams()
    case_id = "STRATEGY_REPEATABILITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

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
            scanner_result_1 = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))
            scanner_result_2 = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))

    add_check(results, "strategy_repeatability", case_id, "scanner_result_repeatable", scanner_result_1, scanner_result_2)

    def _run_objective_once():
        session = _FakeOptimizerSession(fixed_tp_percent=0.25)
        trial = _FakeOptunaTrial(
            number=2,
            preset_values={
                "use_bb": False,
                "use_kc": False,
                "use_vol": False,
                "atr_len": 11,
                "atr_times_init": 1.6,
                "atr_times_trail": 2.6,
                "atr_buy_tol": 0.9,
                "high_len": 60,
                "min_history_trades": 1,
                "min_history_ev": 0.0,
                "min_history_win_rate": 0.35,
            },
        )
        perf_counter_values = iter([0.00, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        with patch("tools.optimizer.objective_runner.prepare_trial_inputs", return_value=_build_fake_prepare_result(master_dates=["2026-01-02", "2026-01-03"])), patch(
            "tools.optimizer.objective_runner.run_portfolio_timeline",
            side_effect=_make_fake_portfolio_runner(
                ret_pct=26.0,
                mdd=-9.0,
                annual_return_pct=18.0,
                yearly_return_rows=[{"year": 2024, "return_pct": 12.5}, {"year": 2025, "return_pct": 9.3}],
            ),
        ), patch("tools.optimizer.objective_runner.apply_filter_rules", return_value=None), patch(
            "tools.optimizer.objective_runner.calc_portfolio_score", return_value=88.123
        ), patch("tools.optimizer.objective_runner.time.perf_counter", side_effect=lambda: next(perf_counter_values)):
            objective_value = run_optimizer_objective(session, trial)

        stable_user_attrs = {
            key: trial.user_attrs.get(key)
            for key in (
                "prep_mode",
                "prep_start_method",
                "pf_return",
                "annual_return_pct",
                "r_squared",
                "m_win_rate",
                "base_score",
                "profile_row",
            )
        }
        return {
            "value": objective_value,
            "trial_params": dict(trial.params),
            "user_attrs": stable_user_attrs,
            "profile_rows": list(session.profile_recorder.rows),
        }

    optimizer_run_1 = _run_objective_once()
    optimizer_run_2 = _run_objective_once()
    add_check(results, "strategy_repeatability", case_id, "optimizer_objective_repeatable", optimizer_run_1, optimizer_run_2)

    summary["scanner_status"] = scanner_result_1.get("status")
    summary["optimizer_value"] = optimizer_run_1["value"]
    return results, summary


def validate_strategy_minimum_viability_case(base_params):
    params = base_params if base_params is not None else V16StrategyParams()
    case_id = "STRATEGY_MINIMUM_VIABILITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

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
            scanner_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))
    add_check(results, "strategy_viability", case_id, "scanner_smoke_status", "buy", scanner_result["status"])

    session = _FakeOptimizerSession(fixed_tp_percent=0.25)
    trial = _FakeOptunaTrial(
        number=3,
        preset_values={
            "use_bb": False,
            "use_kc": False,
            "use_vol": False,
            "atr_len": 11,
            "atr_times_init": 1.6,
            "atr_times_trail": 2.6,
            "atr_buy_tol": 0.9,
            "high_len": 60,
            "min_history_trades": 1,
            "min_history_ev": 0.0,
            "min_history_win_rate": 0.35,
        },
    )
    perf_counter_values = iter([0.00, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    with patch("tools.optimizer.objective_runner.prepare_trial_inputs", return_value=_build_fake_prepare_result(master_dates=["2026-01-02", "2026-01-03"])), patch(
        "tools.optimizer.objective_runner.run_portfolio_timeline",
        side_effect=_make_fake_portfolio_runner(
            ret_pct=26.0,
            mdd=-9.0,
            annual_return_pct=18.0,
            yearly_return_rows=[{"year": 2024, "return_pct": 12.5}, {"year": 2025, "return_pct": 9.3}],
        ),
    ), patch("tools.optimizer.objective_runner.apply_filter_rules", return_value=None), patch(
        "tools.optimizer.objective_runner.calc_portfolio_score", return_value=88.123
    ), patch("tools.optimizer.objective_runner.time.perf_counter", side_effect=lambda: next(perf_counter_values)):
        optimizer_value = run_optimizer_objective(session, trial)
    add_check(results, "strategy_viability", case_id, "optimizer_smoke_returns_score", 88.123, optimizer_value)

    scanner_summary_buffer = io.StringIO()
    with redirect_stdout(scanner_summary_buffer):
        print_scanner_summary(
            count_scanned=1,
            elapsed_time=0.12,
            count_history_qualified=1,
            count_skipped_insufficient=0,
            count_sanitized_candidates=0,
            max_workers=1,
            pool_start_method="spawn",
            candidate_rows=[{"kind": "buy", "text": "2330 EV=1.25R", "sort_value": 1.25, "ticker": "2330"}],
            scanner_issue_log_path="outputs/scanner/issues.csv",
        )
    scanner_summary_text = scanner_summary_buffer.getvalue()
    add_check(results, "strategy_viability", case_id, "scanner_reporting_smoke_runs", True, "明日候選清單" in scanner_summary_text and "issues.csv" in scanner_summary_text)

    yearly_buffer = io.StringIO()
    yearly_rows = [
        {"year": 2024, "year_return_pct": 12.5, "is_full_year": True, "start_date": "2024-01-02", "end_date": "2024-12-31"},
        {"year": 2025, "year_return_pct": 9.3, "is_full_year": False, "start_date": "2025-01-02", "end_date": "2025-06-30"},
    ]
    with redirect_stdout(yearly_buffer):
        df_yearly = print_yearly_return_report(yearly_rows)
    add_check(results, "strategy_viability", case_id, "portfolio_reporting_smoke_runs", 2, len(df_yearly))

    dashboard_buffer = io.StringIO()
    with redirect_stdout(dashboard_buffer):
        print_strategy_dashboard(
            params=params,
            title="策略 smoke",
            mode_display="synthetic",
            max_pos=3,
            trades=7,
            missed_b=2,
            missed_s=1,
            final_eq=123456.0,
            avg_exp=42.0,
            sys_ret=26.0,
            bm_ret=8.0,
            sys_mdd=-9.0,
            bm_mdd=-12.0,
            win_rate=57.0,
            payoff=1.8,
            ev=1.3,
            r_sq=0.87,
            m_win_rate=61.0,
            bm_r_sq=0.72,
            bm_m_win_rate=55.0,
            normal_trades=6,
            extended_trades=1,
            annual_trades=12.5,
            reserved_buy_fill_rate=78.0,
            annual_return_pct=18.0,
            bm_annual_return_pct=6.1,
            min_full_year_return_pct=5.5,
            bm_min_full_year_return_pct=1.2,
        )
    add_check(results, "strategy_viability", case_id, "strategy_dashboard_smoke_runs", True, "【訓練參數】" in dashboard_buffer.getvalue() and "系統得分" in dashboard_buffer.getvalue())

    summary["scanner_status"] = scanner_result["status"]
    summary["optimizer_value"] = optimizer_value
    summary["yearly_rows"] = len(df_yearly)
    return results, summary


def validate_strategy_reporting_schema_compatibility_case(base_params):
    params = base_params if base_params is not None else V16StrategyParams()
    case_id = "STRATEGY_REPORTING_SCHEMA_COMPATIBILITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    default_payload = params_to_json_dict(V16StrategyParams())
    export_trial = SimpleNamespace(
        number=4,
        params={
            "high_len": 65,
            "atr_len": 13,
            "atr_buy_tol": 0.9,
            "atr_times_init": 1.8,
            "atr_times_trail": 2.8,
            "bb_len": 20,
            "bb_mult": 2.0,
            "kc_len": 20,
            "kc_mult": 2.0,
            "vol_short_len": 5,
            "vol_long_len": 19,
            "fixed_risk": 0.01,
            "min_history_trades": 1,
            "min_history_ev": 0.0,
            "min_history_win_rate": 0.3,
        },
        user_attrs={"fixed_tp_percent": 0.27},
        value=88.123,
    )
    export_colors = {"red": "", "green": "", "reset": ""}

    with TemporaryDirectory() as tmp_dir:
        export_path = Path(tmp_dir) / "best_params.json"
        study = _FakeStudy([export_trial], best_trial=export_trial)
        export_status = export_best_params_if_requested(
            study,
            best_params_path=str(export_path),
            fixed_tp_percent=None,
            colors=export_colors,
        )
        exported_payload = json.loads(export_path.read_text(encoding="utf-8"))

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
            scanner_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))

    add_check(results, "strategy_reporting", case_id, "best_params_export_status", 0, export_status)
    add_check(results, "strategy_reporting", case_id, "best_params_export_keys_match_strategy_schema", sorted(default_payload.keys()), sorted(exported_payload.keys()))

    expected_scanner_keys = ["expected_value", "message", "proj_cost", "sanitize_issue", "sort_value", "status", "ticker"]
    add_check(results, "strategy_reporting", case_id, "scanner_normalized_payload_keys_stable", expected_scanner_keys, sorted(scanner_result.keys()))

    yearly_rows = [
        {"year": 2024, "year_return_pct": 12.5, "is_full_year": True, "start_date": "2024-01-02", "end_date": "2024-12-31"},
        {"year": 2025, "year_return_pct": 9.3, "is_full_year": False, "start_date": "2025-01-02", "end_date": "2025-06-30"},
    ]
    with redirect_stdout(io.StringIO()):
        df_yearly = print_yearly_return_report(yearly_rows)
    add_check(
        results,
        "strategy_reporting",
        case_id,
        "yearly_report_columns_stable",
        ["year", "year_return_pct", "is_full_year", "start_date", "end_date", "year_label", "year_type"],
        list(df_yearly.columns),
    )
    add_check(results, "strategy_reporting", case_id, "yearly_report_year_type_values", ["完整", "非完整"], df_yearly["year_type"].tolist())

    summary["exported_key_count"] = len(exported_payload)
    summary["scanner_status"] = scanner_result["status"]
    return results, summary


def _build_fake_prepare_result(*, master_dates):
    return {
        "prep_mode": "parallel",
        "pool_start_method": "spawn",
        "pool_error_text": None,
        "prep_failures": [("2330", "有效資料不足: 2330"), ("0050", "unexpected error")],
        "prep_wall_sec": 0.12,
        "prep_profile": {
            "worker_total_sum_sec": 0.4,
            "copy_sum_sec": 0.01,
            "generate_signals_sum_sec": 0.09,
            "assign_sum_sec": 0.02,
            "run_backtest_sum_sec": 0.2,
            "to_dict_sum_sec": 0.03,
            "ok_count": 3,
            "fail_count": 1,
            "prep_total_sum_sec": 0.36,
        },
        "master_dates": list(master_dates),
        "all_dfs_fast": {"0050": object()},
        "all_trade_logs": {},
    }


def _make_fake_portfolio_runner(*, ret_pct, mdd, annual_return_pct, yearly_return_rows):
    def _runner(
        all_dfs_fast,
        all_trade_logs,
        sorted_dates,
        train_start_year,
        ai_params,
        train_max_positions,
        train_enable_rotation,
        benchmark_ticker=None,
        benchmark_data=None,
        is_training=False,
        profile_stats=None,
        verbose=False,
    ):
        if profile_stats is not None:
            profile_stats.update({
                "portfolio_wall_sec": 0.22,
                "portfolio_day_loop_sec": 0.08,
                "curve_stats_sec": 0.03,
                "full_year_count": 2,
                "min_full_year_return_pct": 5.5,
                "bm_min_full_year_return_pct": 1.2,
                "yearly_return_rows": list(yearly_return_rows),
            })
        return (
            ret_pct,
            mdd,
            7,
            123456.0,
            42.0,
            58.0,
            8.0,
            -12.0,
            57.0,
            1.3,
            1.8,
            2,
            1,
            0.87,
            61.0,
            0.72,
            55.0,
            6,
            1,
            12.5,
            78.0,
            annual_return_pct,
            6.1,
        )

    return _runner


def validate_optimizer_objective_export_contract_case(_base_params):
    case_id = "OPTIMIZER_OBJECTIVE_EXPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    explicit_tp_params = build_optimizer_trial_params(
        {"high_len": 55, "tp_percent": 0.11},
        user_attrs={"fixed_tp_percent": 0.22},
        fixed_tp_percent=0.33,
    )
    add_check(results, "strategy_contract", case_id, "tp_percent_prefers_trial_params", 0.11, explicit_tp_params["tp_percent"])

    attr_tp_params = build_optimizer_trial_params(
        {"high_len": 55},
        user_attrs={"fixed_tp_percent": 0.22},
        fixed_tp_percent=0.33,
    )
    add_check(results, "strategy_contract", case_id, "tp_percent_falls_back_to_user_attr", 0.22, attr_tp_params["tp_percent"])

    fixed_tp_params = build_optimizer_trial_params({"high_len": 55}, user_attrs={}, fixed_tp_percent=0.33)
    add_check(results, "strategy_contract", case_id, "tp_percent_falls_back_to_fixed_setting", 0.33, fixed_tp_params["tp_percent"])

    missing_tp_raises = False
    try:
        build_optimizer_trial_params({"high_len": 55}, user_attrs={}, fixed_tp_percent=None)
    except ValueError:
        missing_tp_raises = True
    add_check(results, "strategy_contract", case_id, "tp_percent_missing_everywhere_is_fail_fast", True, missing_tp_raises)

    filter_fail_session = _FakeOptimizerSession(fixed_tp_percent=0.25)
    filter_fail_trial = _FakeOptunaTrial(
        number=0,
        preset_values={
            "use_bb": False,
            "use_kc": False,
            "use_vol": False,
            "atr_len": 10,
            "atr_times_init": 1.5,
            "atr_times_trail": 2.5,
            "atr_buy_tol": 0.8,
            "high_len": 55,
            "min_history_trades": 1,
            "min_history_ev": 0.0,
            "min_history_win_rate": 0.3,
        },
    )
    with patch("tools.optimizer.objective_runner.prepare_trial_inputs", return_value=_build_fake_prepare_result(master_dates=["2026-01-02"])), patch(
        "tools.optimizer.objective_runner.run_portfolio_timeline",
        side_effect=_make_fake_portfolio_runner(
            ret_pct=18.0,
            mdd=-11.0,
            annual_return_pct=14.0,
            yearly_return_rows=[{"year": 2024, "return_pct": 8.5}],
        ),
    ), patch("tools.optimizer.objective_runner.apply_filter_rules", return_value="月勝率偏低 (30%)"):
        filter_fail_value = run_optimizer_objective(filter_fail_session, filter_fail_trial)

    filter_fail_profile = filter_fail_trial.user_attrs.get("profile_row", {})
    add_check(results, "strategy_contract", case_id, "objective_filter_fail_returns_invalid_trial_value", INVALID_TRIAL_VALUE, filter_fail_value)
    add_check(results, "strategy_contract", case_id, "objective_filter_fail_sets_fail_reason", "月勝率偏低 (30%)", filter_fail_trial.user_attrs.get("fail_reason"))
    add_check(results, "strategy_contract", case_id, "objective_filter_fail_profile_trial_value", INVALID_TRIAL_VALUE, filter_fail_profile.get("trial_value"))
    add_check(results, "strategy_contract", case_id, "objective_filter_fail_profile_reason", "月勝率偏低 (30%)", filter_fail_profile.get("fail_reason"))
    add_check(results, "strategy_contract", case_id, "objective_filter_fail_profile_row_recorded", 1, len(filter_fail_session.profile_recorder.rows))
    add_check(results, "strategy_contract", case_id, "objective_filter_fail_records_only_insufficient_prep_failures", [("2330", "有效資料不足: 2330")], filter_fail_session.recorded_prep_failures)
    add_check(results, "strategy_contract", case_id, "objective_filter_fail_does_not_set_pf_return", False, "pf_return" in filter_fail_trial.user_attrs)

    success_session = _FakeOptimizerSession(fixed_tp_percent=0.25)
    success_trial = _FakeOptunaTrial(
        number=1,
        preset_values={
            "use_bb": False,
            "use_kc": False,
            "use_vol": False,
            "atr_len": 11,
            "atr_times_init": 1.6,
            "atr_times_trail": 2.6,
            "atr_buy_tol": 0.9,
            "high_len": 60,
            "min_history_trades": 1,
            "min_history_ev": 0.0,
            "min_history_win_rate": 0.35,
        },
    )
    with patch("tools.optimizer.objective_runner.prepare_trial_inputs", return_value=_build_fake_prepare_result(master_dates=["2026-01-02", "2026-01-03"])), patch(
        "tools.optimizer.objective_runner.run_portfolio_timeline",
        side_effect=_make_fake_portfolio_runner(
            ret_pct=26.0,
            mdd=-9.0,
            annual_return_pct=18.0,
            yearly_return_rows=[{"year": 2024, "return_pct": 12.5}, {"year": 2025, "return_pct": 9.3}],
        ),
    ), patch("tools.optimizer.objective_runner.apply_filter_rules", return_value=None), patch(
        "tools.optimizer.objective_runner.calc_portfolio_score", return_value=88.123
    ):
        success_value = run_optimizer_objective(success_session, success_trial)

    success_profile = success_trial.user_attrs.get("profile_row", {})
    add_check(results, "strategy_contract", case_id, "objective_success_returns_score", 88.123, success_value)
    add_check(results, "strategy_contract", case_id, "objective_success_sets_base_score", 88.123, success_trial.user_attrs.get("base_score"))
    add_check(results, "strategy_contract", case_id, "objective_success_sets_pf_return", 26.0, success_trial.user_attrs.get("pf_return"))
    add_check(results, "strategy_contract", case_id, "objective_success_sets_yearly_rows", 2, len(success_trial.user_attrs.get("yearly_return_rows", [])))
    add_check(results, "strategy_contract", case_id, "objective_success_profile_trial_value", 88.123, success_profile.get("trial_value"))
    add_check(results, "strategy_contract", case_id, "objective_success_profile_reason_empty", "", success_profile.get("fail_reason"))
    add_check(results, "strategy_contract", case_id, "objective_success_profile_row_recorded", 1, len(success_session.profile_recorder.rows))

    export_colors = {"red": "", "green": "", "reset": ""}
    qualified_export_trial = SimpleNamespace(
        number=3,
        params={"high_len": 65, "atr_len": 13},
        user_attrs={"fixed_tp_percent": 0.27},
        value=88.123,
    )
    rejected_export_trial = SimpleNamespace(
        number=2,
        params={"high_len": 55, "atr_len": 10},
        user_attrs={"fixed_tp_percent": 0.19},
        value=INVALID_TRIAL_VALUE,
    )

    with TemporaryDirectory() as tmp_dir:
        success_export_path = Path(tmp_dir) / "best_params_success.json"
        success_study = _FakeStudy([rejected_export_trial, qualified_export_trial], best_trial=qualified_export_trial)
        success_status = export_best_params_if_requested(
            success_study,
            best_params_path=str(success_export_path),
            fixed_tp_percent=None,
            colors=export_colors,
        )
        exported_payload = json.loads(success_export_path.read_text(encoding="utf-8"))

        failure_export_path = Path(tmp_dir) / "best_params_failure.json"
        failure_study = _FakeStudy([rejected_export_trial], best_trial=rejected_export_trial)
        failure_status = export_best_params_if_requested(
            failure_study,
            best_params_path=str(failure_export_path),
            fixed_tp_percent=None,
            colors=export_colors,
        )

    add_check(results, "strategy_contract", case_id, "export_best_params_success_status", 0, success_status)
    add_check(results, "strategy_contract", case_id, "export_best_params_uses_best_trial_tp_percent", 0.27, exported_payload["tp_percent"])
    add_check(results, "strategy_contract", case_id, "export_best_params_preserves_best_trial_high_len", 65, exported_payload["high_len"])
    add_check(results, "strategy_contract", case_id, "export_best_params_failure_status_for_unqualified_best_trial", 1, failure_status)
    add_check(results, "strategy_contract", case_id, "export_best_params_failure_does_not_create_payload", False, failure_export_path.exists())

    summary["success_export_key_count"] = len(exported_payload)
    summary["registry_contract_checks"] = len(results)
    return results, summary
