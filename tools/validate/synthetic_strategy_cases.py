from __future__ import annotations

import importlib
import io
import json
import math
import re
import unicodedata
from decimal import Decimal
from pathlib import Path
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.buy_sort import calc_buy_sort_value
from core.strategy_dashboard import print_optimizer_trial_console_dashboard, print_strategy_dashboard
from core.walk_forward_policy import load_walk_forward_policy
from core.config import SCORE_CALC_METHOD, SCORE_NUMERATOR_METHOD, V16StrategyParams
from core.params_io import params_to_json_dict
from core.portfolio_stats import calc_portfolio_score
from tools.optimizer.objective_runner import run_optimizer_objective
from tools.optimizer.session import OptimizerSession
from tools.optimizer import callbacks as optimizer_callbacks
from tools.portfolio_sim.reporting import print_yearly_return_report
from tools.optimizer.runtime import export_best_params_if_requested
from tools.optimizer.study_utils import (
    DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE,
    INVALID_TRIAL_VALUE,
    OBJECTIVE_MODE_LEGACY_BASE_SCORE,
    OPTIMIZER_TP_PERCENT_SEARCH_SPEC,
    build_best_params_payload_from_trial,
    build_optimizer_trial_params,
)
from tools.scanner.reporting import print_scanner_summary
from tools.scanner.stock_processor import process_single_stock
from tools.validate.scanner_expectations import normalize_scanner_result
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE

from .checks import add_check


def _optimizer_export_canonical_decimal_places():
    decimal_places = {}
    for field_name, spec in BREAKOUT_OPTIMIZER_SEARCH_SPACE.items():
        if spec.get("kind") != "float" or spec.get("step") is None:
            continue
        decimal_places[field_name] = max(0, -Decimal(str(spec["step"])).normalize().as_tuple().exponent)
    decimal_places["tp_percent"] = max(0, -Decimal(str(OPTIMIZER_TP_PERCENT_SEARCH_SPEC["step"])).normalize().as_tuple().exponent)
    return decimal_places


def _canonicalize_optimizer_export_repr(field_name, raw_value):
    decimal_places = _optimizer_export_canonical_decimal_places().get(field_name)
    if decimal_places is None:
        return repr(raw_value)
    quantizer = Decimal("1").scaleb(-decimal_places)
    return repr(float(Decimal(str(float(raw_value))).quantize(quantizer)))


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

    def mark_trial_completed(self, trial_number=None):
        return None


class _FakeOptimizerSession:
    def __init__(self, *, fixed_tp_percent=0.25):
        self.raw_data_cache = {}
        self.raw_data_cache_data_dir = None
        self.default_max_workers = 1
        self.static_fast_cache = {}
        self.master_dates = set()
        self.sorted_master_dates = []
        self.train_start_year = 2020
        self.search_train_end_year = 2025
        self.objective_mode = OBJECTIVE_MODE_LEGACY_BASE_SCORE
        self.train_max_positions = 3
        self.train_enable_rotation = False
        self.optimizer_fixed_tp_percent = fixed_tp_percent
        self.profile_recorder = _FakeProfileRecorder()
        self.recorded_prep_failures = []

    def record_optimizer_prep_failures(self, failures):
        self.recorded_prep_failures.extend(list(failures))

    def get_trial_prep_executor_bundle(self, max_workers):
        return None

    def resolve_optimizer_tp_percent(self, trial, fixed_tp_percent):
        if fixed_tp_percent is None:
            return trial.suggest_float(
                "tp_percent",
                OPTIMIZER_TP_PERCENT_SEARCH_SPEC["low"],
                OPTIMIZER_TP_PERCENT_SEARCH_SPEC["high"],
                step=OPTIMIZER_TP_PERCENT_SEARCH_SPEC["step"],
            )
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
    params = base_params if base_params is not None else V16StrategyParams()
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
            actual = isinstance(actual_value, float)
        elif default_value is None:
            expected = True
            actual = actual_value is None or isinstance(actual_value, int)
        else:
            expected = True
            actual = isinstance(actual_value, type(default_value))
        add_check(results, "strategy_schema", case_id, f"best_params_type::{field_name}", expected, actual)

    shipped_best_params_paths = [Path("models/run_best_params.json"), Path("models/run_best_params.json")]
    shipped_payload_keys = {}
    shipped_payload_type_mismatches = {}
    for shipped_path in shipped_best_params_paths:
        shipped_payload = json.loads(shipped_path.read_text(encoding="utf-8"))
        shipped_payload_keys[shipped_path.name] = sorted(shipped_payload.keys())
        mismatch_fields = []
        for field_name, default_value in default_payload.items():
            actual_value = shipped_payload[field_name]
            if isinstance(default_value, bool):
                field_ok = isinstance(actual_value, bool)
            elif isinstance(default_value, int) and not isinstance(default_value, bool):
                field_ok = isinstance(actual_value, int) and not isinstance(actual_value, bool)
            elif isinstance(default_value, float):
                field_ok = isinstance(actual_value, float)
            elif default_value is None:
                field_ok = actual_value is None or isinstance(actual_value, int)
            else:
                field_ok = isinstance(actual_value, type(default_value))
            if not field_ok:
                mismatch_fields.append(f"{field_name}:{type(actual_value).__name__}")
        shipped_payload_type_mismatches[shipped_path.name] = mismatch_fields

    add_check(
        results,
        "strategy_schema",
        case_id,
        "repo_shipped_reference_payload_keys_match_strategy_schema",
        {path.name: sorted(default_payload.keys()) for path in shipped_best_params_paths},
        shipped_payload_keys,
    )
    add_check(
        results,
        "strategy_schema",
        case_id,
        "repo_shipped_reference_payload_types_match_strategy_schema",
        {path.name: [] for path in shipped_best_params_paths},
        shipped_payload_type_mismatches,
    )

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
            "asset_growth": 18.0,
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
    params = base_params if base_params is not None else V16StrategyParams()
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

    growth_low = calc_buy_sort_value("ASSET_GROWTH", 0.5, 10000, 0.35, 8, 12.0)
    growth_high = calc_buy_sort_value("ASSET_GROWTH", 0.5, 10000, 0.35, 8, 28.0)
    add_check(results, "strategy_score", case_id, "buy_sort_asset_growth_monotonic", True, growth_high > growth_low)
    add_check(results, "strategy_score", case_id, "buy_sort_asset_growth_type", True, isinstance(growth_high, float))
    add_check(results, "strategy_score", case_id, "buy_sort_asset_growth_finite", True, math.isfinite(growth_high))

    with patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"):
        score_low_return = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=10.0)
        score_high_return = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=20.0)
        score_worse_mdd = calc_portfolio_score(sys_ret=10.0, sys_mdd=-30.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=20.0)
    add_check(results, "strategy_score", case_id, "portfolio_score_return_monotonic", True, score_high_return > score_low_return)
    add_check(results, "strategy_score", case_id, "portfolio_score_mdd_monotonic", True, score_high_return > score_worse_mdd)
    add_check(results, "strategy_score", case_id, "portfolio_score_romd_finite", True, math.isfinite(score_high_return))

    with patch("config.training_policy.SCORE_CALC_METHOD", "LOG_R2"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"):
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
            "asset_growth": 18.0,
            "max_drawdown": 9.5,
            "extended_candidate_today": None,
        }
        with patch("tools.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "tools.scanner.stock_processor.run_v16_backtest", return_value=buy_stats
        ):
            buy_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))
        from core.config import get_buy_sort_method
        expected_sort = calc_buy_sort_value(
            get_buy_sort_method(),
            buy_stats["expected_value"],
            buy_result["proj_cost"],
            buy_stats["win_rate"] / 100.0,
            buy_stats["trade_count"],
            buy_stats["asset_growth"],
        )
        add_check(results, "strategy_score", case_id, "scanner_sort_value_matches_buy_sort_formula", expected_sort, buy_result["sort_value"])
        add_check(results, "strategy_score", case_id, "scanner_sort_value_comparable", True, isinstance(buy_result["sort_value"], float) and math.isfinite(float(buy_result["sort_value"])))

    summary["score_calc_method_default"] = SCORE_CALC_METHOD
    summary["score_numerator_method_default"] = SCORE_NUMERATOR_METHOD
    return results, summary


def validate_score_numerator_option_case(_base_params):
    case_id = "SCORE_NUMERATOR_OPTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"):
        annual_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0)
    expected_annual = 18.0 / (abs(-20.0) + 0.0001)
    add_check(results, "strategy_score", case_id, "annual_return_numerator_formula", expected_annual, annual_score)

    with patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_RETURN"):
        total_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0)
    expected_total = 12.0 / (abs(-20.0) + 0.0001)
    add_check(results, "strategy_score", case_id, "total_return_numerator_formula", expected_total, total_score)
    add_check(results, "strategy_score", case_id, "numerator_switch_changes_score_when_returns_differ", True, annual_score != total_score)

    with patch("config.training_policy.SCORE_CALC_METHOD", "LOG_R2"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_RETURN"):
        low_total_score = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=40.0, r_sq=0.4, annual_return_pct=40.0)
        high_total_score = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=60.0, r_sq=0.9, annual_return_pct=40.0)
    add_check(results, "strategy_score", case_id, "log_r2_total_return_quality_monotonic", True, high_total_score > low_total_score)

    annual_missing = None
    with patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"):
        annual_missing = calc_portfolio_score(sys_ret=9.0, sys_mdd=-15.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=None)
    expected_fallback = 9.0 / (abs(-15.0) + 0.0001)
    add_check(results, "strategy_score", case_id, "annual_return_numerator_falls_back_to_total_return_when_missing", expected_fallback, annual_missing)

    summary["score_calc_method_default"] = SCORE_CALC_METHOD
    summary["score_numerator_method_default"] = SCORE_NUMERATOR_METHOD
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

    scanner_issue_log_path = "outputs/vip_scanner/issues.csv"
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
            scanner_issue_log_path=scanner_issue_log_path,
        )
    scanner_summary_text = scanner_summary_buffer.getvalue()
    add_check(results, "strategy_viability", case_id, "scanner_reporting_smoke_runs", True, "明日候選清單" in scanner_summary_text and scanner_issue_log_path in scanner_summary_text)

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
        export_path = Path(tmp_dir) / "run_best_params.json"
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
        "all_pit_stats_index": {},
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
        replay_counts=None,
        pit_stats_index=None,
        **_unused_kwargs,
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
        params={
            "high_len": 65,
            "atr_len": 13,
            "atr_buy_tol": 1.3000000000000003,
            "min_history_ev": 0.10000000000000009,
            "tp_percent": 0.30000000000000004,
        },
        user_attrs={},
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
    add_check(results, "strategy_contract", case_id, "export_best_params_uses_best_trial_tp_percent", 0.3, exported_payload["tp_percent"])
    add_check(results, "strategy_contract", case_id, "export_best_params_canonicalizes_tp_percent_step_float", "0.3", repr(exported_payload["tp_percent"]))
    add_check(results, "strategy_contract", case_id, "export_best_params_preserves_best_trial_high_len", 65, exported_payload["high_len"])
    add_check(results, "strategy_contract", case_id, "export_best_params_canonicalizes_atr_buy_tol_step_float", "1.3", repr(exported_payload["atr_buy_tol"]))
    add_check(results, "strategy_contract", case_id, "export_best_params_canonicalizes_min_history_ev_step_float", "0.1", repr(exported_payload["min_history_ev"]))
    add_check(results, "strategy_contract", case_id, "export_best_params_keeps_default_buy_fee_canonical_decimal", "0.000399", repr(exported_payload["buy_fee"]))
    add_check(results, "strategy_contract", case_id, "export_best_params_keeps_default_sell_fee_canonical_decimal", "0.000399", repr(exported_payload["sell_fee"]))
    add_check(results, "strategy_contract", case_id, "export_best_params_failure_status_for_unqualified_best_trial", 1, failure_status)
    add_check(results, "strategy_contract", case_id, "export_best_params_failure_does_not_create_payload", False, failure_export_path.exists())

    canonical_model_files = [
        Path("models/candidate_best_params.json"),
        Path("models/run_best_params.json"),
    ]
    canonical_field_names = set(_optimizer_export_canonical_decimal_places()) | {"buy_fee", "sell_fee"}
    shipped_repr_map = {}
    expected_shipped_repr_map = {}
    for artifact_path in canonical_model_files:
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        for field_name in canonical_field_names:
            if field_name not in artifact_payload:
                continue
            map_key = f"{artifact_path.name}::{field_name}"
            shipped_repr_map[map_key] = repr(artifact_payload[field_name])
            if field_name in {"buy_fee", "sell_fee"}:
                expected_shipped_repr_map[map_key] = "0.000399"
            else:
                expected_shipped_repr_map[map_key] = _canonicalize_optimizer_export_repr(field_name, artifact_payload[field_name])

    add_check(
        results,
        "strategy_contract",
        case_id,
        "repo_shipped_reference_artifacts_use_canonical_optimizer_decimal_repr",
        expected_shipped_repr_map,
        shipped_repr_map,
    )

    optimizer_main = importlib.import_module("tools.optimizer.main")
    incompatible_candidate_summary = {
        "base_score": 10.0,
        "local_min_score": 7.0,
        "retention": 0.7,
        "objective_mode": "split_train_romd",
        "train_start_year": 2016,
        "search_train_end_year": 2020,
        "oos_start_year": 2021,
    }
    stale_run_best_summary = {
        "base_score": 9.0,
        "local_min_score": 8.0,
        "retention": 0.8,
        "objective_mode": "split_train_romd",
        "train_start_year": 2023,
        "search_train_end_year": 2024,
        "oos_start_year": 2025,
    }
    should_promote_stale_policy, stale_policy_reason = optimizer_main._should_promote_candidate(
        candidate_summary=incompatible_candidate_summary,
        run_best_summary=stale_run_best_summary,
    )
    add_check(results, "strategy_contract", case_id, "optimizer_promote_ignores_stale_run_best_policy_baseline", True, should_promote_stale_policy)
    add_check(results, "strategy_contract", case_id, "optimizer_promote_reports_stale_run_best_policy_baseline", True, "effective policy" in stale_policy_reason)

    robustness = importlib.import_module("tools.optimizer.robustness")
    no_neighbor_session = SimpleNamespace()
    no_neighbor_trial = SimpleNamespace(number=99, user_attrs={"base_score": 42.0}, value=42.0)
    with patch.object(robustness, "_build_neighbor_candidates", return_value=[]):
        no_neighbor_local_min = robustness.compute_local_min_score(no_neighbor_session, no_neighbor_trial)
    add_check(results, "strategy_contract", case_id, "local_min_score_no_legal_neighbor_fails_gate_conservatively", INVALID_TRIAL_VALUE, no_neighbor_local_min)

    training_policy = importlib.import_module("config.training_policy")
    add_check(
        results,
        "strategy_contract",
        case_id,
        "optimizer_default_interactive_trials_is_1000",
        1000,
        DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE,
    )
    add_check(
        results,
        "strategy_contract",
        case_id,
        "local_min_score_finalist_top_k_defaults_to_2pct_of_1000_trials",
        20,
        training_policy.resolve_optimizer_local_min_score_finalist_top_k(1000),
    )
    add_check(
        results,
        "strategy_contract",
        case_id,
        "local_min_score_finalist_top_k_uses_2pct_with_minimum_floor",
        5,
        training_policy.resolve_optimizer_local_min_score_finalist_top_k(100),
    )
    add_check(
        results,
        "strategy_contract",
        case_id,
        "local_min_score_finalist_top_k_uses_session_trial_count_by_default",
        20,
        robustness._resolve_local_min_score_finalist_top_k(SimpleNamespace(n_trials=1000)),
    )

    summary["success_export_key_count"] = len(exported_payload)
    summary["registry_contract_checks"] = len(results)
    return results, summary



def validate_optimizer_interrupt_export_contract_case(_base_params):
    from tools.optimizer.runtime import resolve_training_session_export_policy

    case_id = "OPTIMIZER_INTERRUPT_EXPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    export_only_allowed, export_only_reason = resolve_training_session_export_policy(
        requested_n_trials=0,
        completed_session_trials=0,
        interrupted=False,
    )
    add_check(results, "strategy_contract", case_id, "export_policy_allows_export_only_mode", True, export_only_allowed)
    add_check(results, "strategy_contract", case_id, "export_policy_export_only_reason", "export_only", export_only_reason)

    target_reached_allowed, target_reached_reason = resolve_training_session_export_policy(
        requested_n_trials=5,
        completed_session_trials=5,
        interrupted=True,
    )
    add_check(results, "strategy_contract", case_id, "export_policy_allows_export_when_target_reached", True, target_reached_allowed)
    add_check(results, "strategy_contract", case_id, "export_policy_target_reached_reason", "target_reached", target_reached_reason)

    interrupted_allowed, interrupted_reason = resolve_training_session_export_policy(
        requested_n_trials=5,
        completed_session_trials=2,
        interrupted=True,
    )
    add_check(results, "strategy_contract", case_id, "export_policy_blocks_interrupted_partial_session", False, interrupted_allowed)
    add_check(results, "strategy_contract", case_id, "export_policy_interrupted_reason", "interrupted_before_target", interrupted_reason)

    optimizer_main = importlib.import_module("tools.optimizer.main")

    class _FakeMainProfileRecorder:
        def __init__(self):
            self.enabled = False
            self.csv_path = ""

        def init_output_files(self):
            return None

        def mark_run_started(self):
            return None

        def mark_trial_completed(self, trial_number=None):
            return None

        def print_summary(self):
            return None

    class _FakeMainSession:
        def __init__(self):
            self.n_trials = 0
            self.current_session_trial = 0
            self.profile_recorder = _FakeMainProfileRecorder()
            self.get_best_completed_trial_or_none = lambda study: getattr(study, "best_trial", None)

        def load_raw_data(self, *args, **kwargs):
            return None

        def objective(self, trial):
            raise AssertionError("optimizer main interrupt contract 不應直接執行 objective")

        def monitoring_callback(self, study, trial):
            return None

        def print_optimizer_prep_summary(self):
            return None

        def close_trial_prep_executor(self):
            return None

    qualified_trial = SimpleNamespace(
        number=7,
        params={"high_len": 65, "atr_len": 13},
        user_attrs={"fixed_tp_percent": 0.27},
        value=88.123,
    )

    with TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        dataset_dir = tmp_root / "dataset"
        dataset_dir.mkdir()
        csv_path = dataset_dir / "2330.csv"
        csv_path.write_text("Date,Open,High,Low,Close,Volume\n2026-01-02,1,1,1,1,1\n", encoding="utf-8")
        db_file = tmp_root / "portfolio_ai.db"
        db_file.write_text("stub", encoding="utf-8")
        best_params_path = tmp_root / "run_best_params.json"
        best_params_path.write_text(json.dumps({"marker": "keep"}, ensure_ascii=False, indent=2), encoding="utf-8")

        fake_session = _FakeMainSession()
        fake_export_calls = []

        class _InterruptingStudy:
            def __init__(self, session):
                self.trials = [qualified_trial]
                self._session = session

            @property
            def best_trial(self):
                return qualified_trial

            def optimize(self, objective, n_trials, n_jobs=1, callbacks=None):
                self._session.current_session_trial = 2
                raise KeyboardInterrupt()

        def _fake_build_optimizer_session(*, walk_forward_policy=None):
            _ = walk_forward_policy
            return fake_session

        def _fake_resolve_trial_count_or_exit(session, *, environ, resolve_optimizer_trial_count, colors):
            session.n_trials = 5
            return None, "TEST"

        def _fake_create_optimizer_study(_db_name, *, seed=None, sampler_kind="tpe"):
            _ = sampler_kind
            return _InterruptingStudy(fake_session)

        def _fake_export_best_params_if_requested(study, *, best_params_path, fixed_tp_percent, colors):
            fake_export_calls.append(best_params_path)
            Path(best_params_path).write_text(json.dumps({"marker": "overwritten"}, ensure_ascii=False, indent=2), encoding="utf-8")
            return 0

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(optimizer_main, "OUTPUT_DIR", str(tmp_root / "outputs")))
            stack.enter_context(patch.object(optimizer_main, "MODELS_DIR", str(tmp_root / "models")))
            stack.enter_context(patch.object(optimizer_main, "RUN_BEST_PARAMS_PATH", str(best_params_path)))
            stack.enter_context(patch.object(optimizer_main, "ensure_runtime_dirs", return_value=None))
            stack.enter_context(patch.object(optimizer_main, "configure_optuna_logging", return_value=None))
            stack.enter_context(patch.object(optimizer_main, "build_optimizer_session", side_effect=_fake_build_optimizer_session))
            stack.enter_context(patch("core.dataset_profiles.resolve_dataset_profile_from_cli_env", return_value=("full", "TEST")))
            stack.enter_context(patch("core.dataset_profiles.get_dataset_dir", return_value=str(dataset_dir)))
            stack.enter_context(patch("core.dataset_profiles.get_dataset_profile_label", return_value="完整資料集"))
            stack.enter_context(patch("core.data_utils.discover_unique_csv_inputs", return_value=([str(csv_path)], None)))
            stack.enter_context(patch("tools.optimizer.study_utils.build_optimizer_db_file_path", return_value=str(db_file)))
            stack.enter_context(patch("tools.optimizer.study_utils.resolve_optimizer_seed", return_value=(None, "ENV:UNSET")))
            stack.enter_context(patch("tools.optimizer.runtime.resolve_trial_count_or_exit", side_effect=_fake_resolve_trial_count_or_exit))
            stack.enter_context(patch("tools.optimizer.runtime.ensure_optimizer_db_usable", return_value=None))
            stack.enter_context(patch("tools.optimizer.runtime.prompt_existing_db_policy", return_value=None))
            stack.enter_context(patch("tools.optimizer.runtime.create_optimizer_study", side_effect=_fake_create_optimizer_study))
            stack.enter_context(patch("tools.optimizer.runtime.maybe_print_history_best", return_value=None))
            stack.enter_context(patch("tools.optimizer.runtime.print_resolved_trial_count", return_value=None))
            stack.enter_context(patch("tools.optimizer.runtime.export_best_params_if_requested", side_effect=_fake_export_best_params_if_requested))
            stack.enter_context(redirect_stdout(stdout_buffer))
            stack.enter_context(redirect_stderr(stderr_buffer))
            rc = optimizer_main.main(["tools/optimizer/main.py", "--dataset", "full"], environ={})

        persisted_payload = json.loads(best_params_path.read_text(encoding="utf-8"))

    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    add_check(results, "strategy_contract", case_id, "optimizer_main_interrupt_returns_zero", 0, rc)
    add_check(results, "strategy_contract", case_id, "optimizer_main_interrupt_does_not_call_export", 0, len(fake_export_calls))
    add_check(results, "strategy_contract", case_id, "optimizer_main_interrupt_preserves_existing_run_best_params_json", "keep", persisted_payload.get("marker"))
    add_check(results, "strategy_contract", case_id, "optimizer_main_interrupt_reports_warning", True, "使用者中斷訓練流程" in stdout_text)
    add_check(results, "strategy_contract", case_id, "optimizer_main_interrupt_reports_skip_overwrite", True, "不自動覆寫" in stdout_text and "2/5" in stdout_text)
    add_check(results, "strategy_contract", case_id, "optimizer_main_interrupt_stderr_empty", "", stderr_text)

    summary["checks"] = len(results)
    return results, summary


def validate_optimizer_session_milestone_cache_case(_base_params):
    case_id = "OPTIMIZER_SESSION_MILESTONE_CACHE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    class _NoopRecorder:
        def __init__(self, **kwargs):
            self.enabled = False
            self.console_print = False
            self.print_every_n_trials = 999999
            self.rows = []

        def append_row(self, row):
            self.rows.append(dict(row))

        def patch_row(self, trial_number, patch):
            return None

        def mark_trial_completed(self, trial_number=None):
            return None

        def mark_run_started(self):
            return None

        def init_output_files(self):
            return None

        def print_summary(self):
            return None

    session = OptimizerSession(
        output_dir=".",
        session_ts="synthetic",
        profile_recorder_cls=_NoopRecorder,
        build_optimizer_trial_params=lambda trial, fixed_tp_percent=None: None,
        get_best_completed_trial_or_none=lambda study: None,
        objective_mode=OBJECTIVE_MODE_LEGACY_BASE_SCORE,
        search_train_end_year=2025,
        walk_forward_policy={"oos_start_year": 2021, "oos_end_year": 2025},
        resolve_optimizer_tp_percent=lambda trial, fixed_tp_percent: 0.25,
        print_strategy_dashboard=lambda *args, **kwargs: None,
        colors={"yellow": "", "reset": ""},
        optimizer_fixed_tp_percent=0.25,
        train_max_positions=3,
        train_start_year=2020,
        train_enable_rotation=False,
        default_max_workers=1,
        enable_optimizer_profiling=False,
        enable_profile_console_print=False,
        profile_print_every_n_trials=999999,
    )

    session.cache_trial_milestone_inputs(1)
    session.cache_trial_milestone_inputs(1, sorted_master_dates=[1, 2], all_pit_stats_index={"2330": {1: []}}, all_dfs_fast={"2330": {"is_setup": [False, True]}})
    session.cache_trial_milestone_inputs(2, all_pit_stats_index={"1101": {2: []}})
    session.cache_trial_milestone_inputs(3, all_pit_stats_index={"1216": {3: []}})
    session.cache_trial_milestone_inputs(4, all_pit_stats_index={"2603": {4: []}})
    evicted_payload = session.consume_trial_milestone_inputs(1)
    kept_payload = session.consume_trial_milestone_inputs(4)
    session.discard_trial_milestone_inputs(999)

    add_check(results, "strategy_contract", case_id, "optimizer_session_ignores_empty_milestone_payload", True, evicted_payload is None)
    add_check(results, "strategy_contract", case_id, "optimizer_session_keeps_recent_milestone_payload", True, isinstance(kept_payload, dict) and kept_payload.get("all_pit_stats_index") == {"2603": {4: []}})
    add_check(results, "strategy_contract", case_id, "optimizer_session_caps_milestone_cache_size", 2, len(session._optimizer_trial_milestone_inputs))

    summary["checks"] = len(results)
    return results, summary


def validate_optimizer_walk_forward_policy_contract_case(_base_params):
    case_id = "OPTIMIZER_WALK_FORWARD_POLICY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    project_root = Path(__file__).resolve().parents[2]
    default_policy = load_walk_forward_policy(str(project_root), environ={})
    default_policy_path = str(default_policy.get("policy_path", "")).replace("\\", "/")
    add_check(results, "strategy_contract", case_id, "default_walk_forward_policy_uses_training_policy", True, default_policy_path.endswith("config/training_policy.py"))
    default_oos_start_year = default_policy.get("oos_start_year")
    if default_oos_start_year is not None:
        expected_default_search_train_end_year = int(default_oos_start_year) - 1
    else:
        expected_default_search_train_end_year = int(default_policy.get("train_start_year", 0)) + int(default_policy.get("min_train_years", 0)) - 1
    add_check(
        results,
        "strategy_contract",
        case_id,
        "default_walk_forward_policy_auto_derives_search_train_end_year",
        expected_default_search_train_end_year,
        int(default_policy.get("search_train_end_year", 0)),
    )
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "wf_override.py"
        tmp_path.write_text(
            'TRAINING_SPLIT_POLICY = {"train_start_year": 2008, "min_train_years": 7}\n',
            encoding="utf-8",
        )
        override_policy = load_walk_forward_policy(
            str(project_root),
            environ={"V16_WALK_FORWARD_POLICY_PATH": str(tmp_path)},
        )
    add_check(results, "strategy_contract", case_id, "python_override_policy_auto_derives_search_train_end_year", 2014, int(override_policy.get("search_train_end_year", 0)))

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "wf_override.json"
        tmp_path.write_text(
            json.dumps({"train_start_year": 2005, "min_train_years": 6}, ensure_ascii=False),
            encoding="utf-8",
        )
        json_override_policy = load_walk_forward_policy(
            str(project_root),
            environ={"V16_WALK_FORWARD_POLICY_PATH": str(tmp_path)},
        )
    add_check(results, "strategy_contract", case_id, "json_override_policy_still_supported", 2010, int(json_override_policy.get("search_train_end_year", 0)))


    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "wf_invalid.py"
        tmp_path.write_text('WALK_FORWARD_POLICY = {"objective_mode": "bad_mode"}\n', encoding="utf-8")
        try:
            load_walk_forward_policy(str(project_root), environ={"V16_WALK_FORWARD_POLICY_PATH": str(tmp_path)})
            invalid_legacy_symbol_rejected = False
        except ValueError as exc:
            invalid_legacy_symbol_rejected = "TRAINING_SPLIT_POLICY" in str(exc)
    add_check(results, "strategy_contract", case_id, "legacy_walk_forward_policy_symbol_rejected", True, invalid_legacy_symbol_rejected)

    callbacks_source = Path(optimizer_callbacks.__file__).read_text(encoding="utf-8")
    objective_runner_source = (project_root / "tools" / "optimizer" / "objective_runner.py").read_text(encoding="utf-8")
    add_check(results, "strategy_contract", case_id, "optimizer_callbacks_imports_pandas_for_oos_year_parsing", True, "import pandas as pd" in callbacks_source)
    add_check(results, "strategy_contract", case_id, "search_train_date_filter_reuses_core_single_source_in_callbacks", True, "from core.walk_forward_policy import filter_search_train_dates" in callbacks_source and "def _filter_search_train_dates" not in callbacks_source)
    add_check(results, "strategy_contract", case_id, "search_train_date_filter_reuses_core_single_source_in_objective_runner", True, "from core.walk_forward_policy import filter_search_train_dates" in objective_runner_source and "def _filter_search_train_dates" not in objective_runner_source)

    params = V16StrategyParams()
    params.use_kc = True
    params.kc_len = 34
    params.kc_mult = 1.8
    training_lines = optimizer_callbacks._build_training_param_lines(params)
    rendered_training_text = "\n".join(training_lines)
    add_check(results, "strategy_contract", case_id, "optimizer_callbacks_kc_label_matches_dashboard_wording", True, "阿肯那(KC)" in rendered_training_text and "阿唐那(KC)" not in rendered_training_text)

    sample_rows = optimizer_callbacks._build_first_zone_rows(
        candidate_metrics={
            "pf_return": 12.34,
            "annual_return_pct": 10.0,
            "min_full_year_return_pct": -3.21,
            "pf_romd": 1.23,
            "pf_mdd": 8.76,
            "m_win_rate": 55.0,
            "win_rate": 60.0,
            "pf_payoff": 1.5,
            "pf_ev": 0.25,
            "pf_trades": 11,
            "normal_trades": 8,
            "extended_trades": 3,
            "missed_total": 2,
            "missed_buys": 1,
            "missed_sells": 1,
            "annual_trades": 4.5,
            "reserved_buy_fill_rate": 83.0,
            "avg_exposure": 62.0,
            "final_equity": 1234567.0,
        },
        reference_metrics={
            "pf_return": 10.0,
            "annual_return_pct": 9.0,
            "min_full_year_return_pct": -5.0,
            "pf_romd": 1.0,
            "pf_mdd": 10.0,
            "m_win_rate": 52.0,
            "win_rate": 58.0,
            "pf_payoff": 1.4,
            "pf_ev": 0.2,
            "pf_trades": 10,
            "normal_trades": 7,
            "extended_trades": 3,
            "missed_total": 3,
            "missed_buys": 2,
            "missed_sells": 1,
            "annual_trades": 4.0,
            "reserved_buy_fill_rate": 80.0,
            "avg_exposure": 58.0,
            "final_equity": 1200000.0,
        },
        benchmark_metrics={
            "pf_return": 8.0,
            "annual_return_pct": 7.5,
            "min_full_year_return_pct": -6.0,
            "pf_romd": 0.8,
            "pf_mdd": 12.0,
            "m_win_rate": 49.0,
            "final_equity": 1180000.0,
        },
    )
    row_names = [str(row.get("name", "")) for row in sample_rows]
    row_map = {str(row.get("name", "")): row for row in sample_rows}
    add_check(results, "strategy_contract", case_id, "optimizer_first_zone_keeps_final_equity_row", True, "最終資產" in row_names)
    add_check(results, "strategy_contract", case_id, "optimizer_first_zone_combines_payoff_and_ev_label_without_extra_spaces", True, "風報比: 期望值" in row_names and "風報比 : 期望值" not in row_names)
    add_check(results, "strategy_contract", case_id, "optimizer_first_zone_formats_payoff_ev_pair_with_consistent_spacing", True, row_map.get("風報比: 期望值", {}).get("candidate") == "1.50: 0.250R" and "1.40: 0.200R" in str(row_map.get("風報比: 期望值", {}).get("reference", "")) and "(+0.10: +0.050R)" in str(row_map.get("風報比: 期望值", {}).get("reference", "")))
    add_check(results, "strategy_contract", case_id, "optimizer_first_zone_formats_trade_split_with_consistent_spacing", True, row_map.get("總交易次數", {}).get("candidate") == "11 (正常: 8｜延續: 3)" and row_map.get("總交易次數", {}).get("reference") == "10 (正常: 7｜延續: 3)")
    add_check(results, "strategy_contract", case_id, "optimizer_first_zone_formats_missed_split_with_consistent_spacing", True, row_map.get("錯失交易次數", {}).get("candidate") == "2 (買: 1｜賣: 1)" and row_map.get("錯失交易次數", {}).get("reference") == "3 (買: 2｜賣: 1)")

    long_training_rows = [
        {
            "name": "風報比: 期望值",
            "candidate": "4.23: 0.607R",
            "candidate_precolored": False,
            "reference": "3.38: 0.505R (+0.85: +0.102R)",
            "reference_precolored": False,
            "benchmark": "-",
            "benchmark_precolored": False,
        },
        {
            "name": "總交易次數",
            "candidate": "649 (正常: 346｜延續: 303)",
            "candidate_precolored": False,
            "reference": "510 (正常: 425｜延續: 85)",
            "reference_precolored": False,
            "benchmark": "-",
            "benchmark_precolored": False,
        },
        {
            "name": "錯失交易次數",
            "candidate": "88 (買: 87｜賣: 1)",
            "candidate_precolored": False,
            "reference": "52 (買: 52｜賣: 0)",
            "reference_precolored": False,
            "benchmark": "-",
            "benchmark_precolored": False,
        },
    ]
    dashboard_buffer = io.StringIO()
    with redirect_stdout(dashboard_buffer):
        print_optimizer_trial_console_dashboard(
            title="",
            milestone_title="🏆 破紀錄！",
            global_strategy_text="買入排序 [按資產成長由大到小排序]",
            mode_display="關閉明牌（穩定鎖倉）",
            max_pos=10,
            model_mode="split",
            objective_mode="split_train_romd",
            score_calc_method=SCORE_CALC_METHOD,
            score_numerator_method=SCORE_NUMERATOR_METHOD,
            system_score_display="1.234",
            training_title="【訓練期間績效對比｜1995-01-05 ~ 2019-12-31】",
            training_rows=long_training_rows,
            testing_title="【測試期間績效對比｜2020-01-01 ~ 2026-03-02｜資料終點：2026-03-02】",
            testing_rows=long_training_rows,
            upgrade_rows=None,
            compare_rows=None,
            params_lines=["核心：測試"],
            hard_gate_lines=["交易頻率：測試"],
        )
    ansi_re = re.compile(r"\[[0-9;]*m")
    dashboard_lines = [ansi_re.sub("", line) for line in dashboard_buffer.getvalue().splitlines()]
    table_lines = [line for line in dashboard_lines if line.startswith("| ") and any(token in line for token in ("風報比: 期望值", "總交易次數", "錯失交易次數"))]

    def _pipe_display_positions(text: str) -> tuple[int, ...]:
        positions = []
        display_offset = 0
        for ch in text:
            if ch == "|":
                positions.append(display_offset)
            if unicodedata.combining(ch):
                continue
            display_offset += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        return tuple(positions)

    pipe_positions = [_pipe_display_positions(line) for line in table_lines]
    add_check(results, "strategy_contract", case_id, "optimizer_console_table_keeps_pipe_alignment_for_long_first_zone_rows", True, len(pipe_positions) == 6 and len(set(pipe_positions)) == 1)

    optimizer_main_source = (project_root / "tools" / "optimizer" / "main.py").read_text(encoding="utf-8")
    train_test_policy_lines = [line for line in optimizer_main_source.splitlines() if "Train/Test policy:" in line]
    add_check(results, "strategy_contract", case_id, "optimizer_start_banner_omits_raw_objective_mode_token", True, bool(train_test_policy_lines) and all("objective=" not in line for line in train_test_policy_lines))

    summary["checks"] = len(results)
    return results, summary
