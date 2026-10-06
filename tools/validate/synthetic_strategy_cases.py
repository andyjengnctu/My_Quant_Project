from __future__ import annotations

from .checks import run_bound_checks

import importlib
import io
import json
import math
import os
import re
import unicodedata
from decimal import Decimal
from pathlib import Path
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.buy_sort import calc_buy_sort_value, sort_candidate_rows
from core.strategy_dashboard import print_optimizer_trial_console_dashboard, print_strategy_dashboard
from core.walk_forward_policy import build_optimizer_runtime_policy, load_walk_forward_policy
from core.config import SCORE_CALC_METHOD, SCORE_NUMERATOR_METHOD, SYSTEM_SCORE_DISPLAY_MULTIPLIER, V16StrategyParams, format_system_score_for_display, get_score_mdd_denominator_epsilon, get_score_mdd_power
from config.training_policy import FULL_END_YEAR, OOS_EVALUATION_END_YEAR
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.strategy_param_artifacts import (
    POLICY_FILENAME_BY_NAME,
    STRATEGY_PARAM_EVALUATION_MODES,
    STRATEGY_PARAM_FAMILIES,
    STRATEGY_PARAM_SCHEDULE_MODES,
    STRATEGY_PARAM_STATE_FILENAME_BY_NAME,
    normalize_strategy_param_policy,
)
from core.portfolio_fast_data import build_score_single_stock_profile_fields
from core.portfolio_stats import calc_plain_romd, calc_portfolio_score, calc_score_median_r_multiplier, calc_score_min_full_year_return_multiplier, calc_score_min_month_return_multiplier, calc_score_min_quarter_return_multiplier, calc_score_portfolio_return_multiplier, calc_score_positive_return_multiplier, calc_score_win_rate_multiplier
from services.optimizer.objective_runner import run_optimizer_objective
from tools.optimizer.session import OptimizerSession, close_study_storage
from tools.optimizer import callbacks as optimizer_callbacks, study_utils
from services.portfolio_sim.reporting import print_yearly_return_report
from tools.optimizer.runtime import (
    OPTIMIZER_DETERMINISTIC_RESUME_STUDY_NAME,
    OPTIMIZER_LEGACY_STUDY_NAME,
    create_optimizer_study,
    export_best_params_if_requested,
    resolve_optimizer_single_fold_search_parallel_trials,
)
from tools.optimizer.score_display import format_optimizer_score_for_display
from services.optimizer.outer_rolling_formatting import _format_system_score_compare_plain
from tools.optimizer.study_utils import (
    DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE,
    INVALID_TRIAL_VALUE,
    OBJECTIVE_MODE_LEGACY_BASE_SCORE,
    OPTIMIZER_TP_PERCENT_SEARCH_SPEC,
    build_best_params_payload_from_trial,
    build_optimizer_trial_params,
)
from services.scanner.reporting import print_scanner_summary
from services.scanner.stock_processor import process_single_stock
from tools.validate.scanner_expectations import normalize_scanner_result
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE

from .checks import raises_expected, bind_synthetic_case, bind_checks, add_check


def _optimizer_search_space_spec(name):
    if name not in BREAKOUT_OPTIMIZER_SEARCH_SPACE:
        raise KeyError(f"optimizer search space 缺少必要參數: {name}")
    spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE[name]
    kind = str(spec.get("kind", "") or "")
    if kind not in {"categorical", "int", "float"}:
        raise ValueError(f"optimizer search space 參數 {name} 的 kind 不合法: {kind!r}")
    return spec


def _optimizer_search_space_choices(name):
    choices = _optimizer_search_space_spec(name).get("choices")
    if isinstance(choices, bool):
        normalized = [choices]
    else:
        normalized = list(choices or [])
    if not normalized:
        raise ValueError(f"optimizer search space 參數 {name} 的 choices 不可為空")
    return normalized


def _optimizer_search_space_sample_value(name, *, prefer=None, step_offset=0, add_float_tail=False):
    spec = _optimizer_search_space_spec(name)
    kind = str(spec["kind"])

    if kind == "categorical":
        choices = _optimizer_search_space_choices(name)
        if prefer in choices:
            return prefer
        default_value = spec.get("default")
        if default_value in choices:
            return default_value
        return choices[0]

    low = spec["low"]
    high = spec["high"]
    step = spec.get("step", 1 if kind == "int" else None)

    if kind == "int":
        low_value = int(low)
        high_value = int(high)
        step_value = int(step or 1)
        candidate = low_value + max(0, int(step_offset)) * step_value
        if candidate > high_value:
            candidate = low_value
        return int(candidate)

    low_decimal = Decimal(str(low))
    high_decimal = Decimal(str(high))
    candidate_decimal = low_decimal
    if step is not None and int(step_offset) > 0:
        stepped = low_decimal + Decimal(str(step)) * int(step_offset)
        if stepped <= high_decimal:
            candidate_decimal = stepped
    value = float(candidate_decimal)
    if add_float_tail and step is not None and Decimal(str(value)) < high_decimal:
        value = float(value + 3e-16)
    return value


def _optimizer_synthetic_trial_values(*, prefer_enabled=True):
    values = {}
    for field_name, spec in BREAKOUT_OPTIMIZER_SEARCH_SPACE.items():
        preferred = True if prefer_enabled and spec.get("kind") == "categorical" else None
        values[field_name] = _optimizer_search_space_sample_value(field_name, prefer=preferred)
    return values


def _optimizer_synthetic_param_payload(*, prefer_enabled=True):
    payload = _optimizer_synthetic_trial_values(prefer_enabled=prefer_enabled)
    defaults = V16StrategyParams()
    payload.setdefault("vol_short_len", defaults.vol_short_len)
    payload.setdefault("fixed_risk", defaults.fixed_risk)
    return payload


def _extract_reference_param_payloads(payload):
    from core.active_param_ensemble import build_active_param_ensemble_schedule, is_active_param_ensemble_payload

    if not is_active_param_ensemble_payload(payload):
        return [payload]
    members = []
    for record in build_active_param_ensemble_schedule(payload):
        for member in record.get("members") or []:
            params = member.get("params")
            if isinstance(params, dict):
                members.append(params)
    return members


def _strategy_param_reference_filenames():
    """Return parameter artifact filenames derived from the canonical artifact SSOT."""

    names = {
        filename
        for filename in STRATEGY_PARAM_STATE_FILENAME_BY_NAME.values()
        if filename.endswith("_params.json")
    }
    for family in STRATEGY_PARAM_FAMILIES:
        for evaluation_mode in STRATEGY_PARAM_EVALUATION_MODES:
            for policy_filename in POLICY_FILENAME_BY_NAME.values():
                if evaluation_mode in STRATEGY_PARAM_SCHEDULE_MODES:
                    names.add(f"{family}_{policy_filename}")
                else:
                    names.add(f"{family}_{evaluation_mode}_{policy_filename}")
    # Historical ROOS references remain schema-readable compatibility artifacts.
    names.update(f"roos_{filename}" for filename in POLICY_FILENAME_BY_NAME.values())
    return frozenset(names)


def _is_strategy_param_reference_path(path):
    name = Path(path).name
    if name in _strategy_param_reference_filenames():
        return True
    family_pattern = "|".join(re.escape(value) for value in STRATEGY_PARAM_FAMILIES)
    policy_stem_pattern = "|".join(
        re.escape(Path(filename).stem)
        for filename in POLICY_FILENAME_BY_NAME.values()
    )
    return re.fullmatch(
        rf"(?:{family_pattern})_(?:{policy_stem_pattern})_seed_[1-9][0-9]*\.json",
        name,
    ) is not None


def _existing_shipped_reference_param_paths():
    root = Path("models") / "strategy_params"
    if not root.is_dir():
        return []
    return [
        path
        for path in sorted(root.rglob("*.json"))
        if "backups" not in path.parts and _is_strategy_param_reference_path(path)
    ]


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


def _history_threshold_payload_is_neutral(payload):
    try:
        min_trades = int(payload.get("min_history_trades", 0))
        min_ev = float(payload.get("min_history_ev", -1.0))
        min_win_rate = float(payload.get("min_history_win_rate", 0.0))
    except (TypeError, ValueError):
        return False
    return min_trades == 0 and min_ev <= 0.0 and min_win_rate <= 0.0


def _optimizer_dependent_field_is_active(field_name, param_payload):
    spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE.get(field_name, {})
    switch_field = spec.get("enabled_by")
    if not switch_field:
        return True
    if switch_field == "use_history_threshold" and switch_field not in param_payload:
        return not _history_threshold_payload_is_neutral(param_payload)
    return bool(param_payload.get(switch_field, False))


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
    def __init__(
        self,
        *,
        output_dir=None,
        session_ts=None,
        enabled=False,
        console_print=False,
        print_every_n_trials=999999,
    ):
        self.rows = []
        self.output_dir = output_dir
        self.session_ts = session_ts
        self.enabled = bool(enabled)
        self.console_print = bool(console_print)
        self.print_every_n_trials = int(print_every_n_trials)

    def append_row(self, row):
        self.rows.append(dict(row))

    def mark_trial_completed(self, trial_number=None):
        return None


def _resolve_fixture_optimizer_tp_percent(trial, fixed_tp_percent):
    if fixed_tp_percent is None:
        return trial.suggest_float(
            "tp_percent",
            OPTIMIZER_TP_PERCENT_SEARCH_SPEC["low"],
            OPTIMIZER_TP_PERCENT_SEARCH_SPEC["high"],
            step=OPTIMIZER_TP_PERCENT_SEARCH_SPEC["step"],
        )
    trial.set_user_attr("fixed_tp_percent", float(fixed_tp_percent))
    return float(fixed_tp_percent)


class _FakeOptimizerSession(OptimizerSession):
    """Canonical OptimizerSession-backed fixture with only test tracing added.

    The fixture intentionally inherits the production session contract instead of
    hand-copying its public methods/attributes.  This keeps synthetic tests aligned
    when the canonical session gains new runtime hooks.
    """

    def __init__(
        self,
        *,
        fixed_tp_percent=0.25,
        fixed_strategy_param_overrides=None,
        runtime_cache_identity=None,
    ):
        super().__init__(
            output_dir="",
            session_ts="synthetic",
            profile_recorder_cls=_FakeProfileRecorder,
            build_optimizer_trial_params=build_optimizer_trial_params,
            get_best_completed_trial_or_none=lambda study: getattr(study, "best_trial", None),
            objective_mode=OBJECTIVE_MODE_LEGACY_BASE_SCORE,
            search_train_end_year=2025,
            walk_forward_policy={},
            resolve_optimizer_tp_percent=_resolve_fixture_optimizer_tp_percent,
            print_strategy_dashboard=print_strategy_dashboard,
            colors={"yellow": "", "reset": ""},
            optimizer_fixed_tp_percent=fixed_tp_percent,
            train_max_positions=3,
            train_start_year=2020,
            train_enable_rotation=False,
            default_max_workers=1,
            enable_optimizer_profiling=False,
            enable_profile_console_print=False,
            profile_print_every_n_trials=999999,
            fixed_strategy_param_overrides=fixed_strategy_param_overrides,
            runtime_context_factory=None,
            runtime_cache_identity=runtime_cache_identity,
        )
        self.recorded_prep_failures = []

    def record_optimizer_prep_failures(self, failures):
        self.recorded_prep_failures.extend(list(failures))
        super().record_optimizer_prep_failures(failures)


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
    results, summary, check, check_true = bind_synthetic_case(case_id, 'strategy_schema')

    default_payload = params_to_json_dict(V16StrategyParams())
    trial_params = _optimizer_synthetic_param_payload(prefer_enabled=True)
    fake_trial = _FakeTrial(trial_params, user_attrs={"fixed_tp_percent": 0.25})
    best_params_payload = build_best_params_payload_from_trial(fake_trial, fixed_tp_percent=None)

    run_bound_checks(
        check,
        (
            ('optimizer_best_params_payload_keys_match_strategy_schema', sorted(default_payload.keys()), sorted(best_params_payload.keys()),),
            ('optimizer_best_params_payload_excludes_runtime_only_fields', True, all((key not in best_params_payload for key in ('optimizer_max_workers', 'scanner_max_workers'))),),
        ),
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
        check(f"best_params_type::{field_name}", expected, actual)

    shipped_best_params_paths = _existing_shipped_reference_param_paths()
    shipped_payload_keys = {}
    shipped_payload_type_mismatches = {}
    for shipped_path in shipped_best_params_paths:
        shipped_payload = json.loads(shipped_path.read_text(encoding="utf-8"))
        reference_param_payloads = _extract_reference_param_payloads(shipped_payload)
        canonical_param_payloads = []
        load_errors = []
        for member_idx, param_payload in enumerate(reference_param_payloads, start=1):
            try:
                canonical_param_payloads.append(params_to_json_dict(build_params_from_mapping(param_payload)))
            except (TypeError, ValueError) as exc:
                load_errors.append(f"member#{member_idx}:{type(exc).__name__}:{exc}")
        representative_payload = canonical_param_payloads[0] if canonical_param_payloads else {}
        shipped_payload_keys[shipped_path.name] = sorted(representative_payload.keys())
        mismatch_fields = []
        if not reference_param_payloads:
            mismatch_fields.append("params:MISSING")
        mismatch_fields.extend(load_errors)
        for member_idx, param_payload in enumerate(canonical_param_payloads, start=1):
            member_prefix = f"member#{member_idx}:" if len(canonical_param_payloads) > 1 else ""
            for field_name, default_value in default_payload.items():
                actual_value = param_payload[field_name]
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
                    mismatch_fields.append(f"{member_prefix}{field_name}:{type(actual_value).__name__}")
        shipped_payload_type_mismatches[shipped_path.name] = mismatch_fields

    run_bound_checks(
        check,
        (
            ('repo_shipped_reference_payload_keys_match_strategy_schema', {path.name: sorted(default_payload.keys()) for path in shipped_best_params_paths}, shipped_payload_keys,),
            ('repo_shipped_reference_payload_types_match_strategy_schema', {path.name: [] for path in shipped_best_params_paths}, shipped_payload_type_mismatches,),
        ),
    )

    restored_trial_params = build_optimizer_trial_params(fake_trial.params, fake_trial.user_attrs, fixed_tp_percent=None)
    check("optimizer_trial_params_contains_fixed_tp_percent",
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
            "close_last": 103.0,
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
            "close_last": 103.0,
            "stop_loss": 99.0,
            "expected_value": 0.8,
            "win_rate": 52.0,
            "trade_count": 8,
            "max_drawdown": 8.0,
            "extended_candidate_today": None,
        }
        skip_exc = ValueError("有效資料不足: 2330")

        with patch("services.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "services.scanner.stock_processor.run_v16_backtest", return_value=buy_stats
        ):
            buy_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))

        expected_keys = ["expected_value", "message", "proj_cost", "sanitize_issue", "sort_value", "status", "ticker"]
        check("scanner_buy_result_keys", expected_keys, sorted(buy_result.keys()))
        check("scanner_buy_status_type", True, isinstance(buy_result["status"], str))
        check("scanner_buy_proj_cost_finite", True, math.isfinite(float(buy_result["proj_cost"])))
        check("scanner_buy_expected_value_finite", True, math.isfinite(float(buy_result["expected_value"])))
        check("scanner_buy_sort_value_finite", True, math.isfinite(float(buy_result["sort_value"])))
        check("scanner_buy_message_contains_ticker", True, "2330" in str(buy_result["message"]))
        check("scanner_buy_sanitize_issue_nullable", True, buy_result["sanitize_issue"] is None or isinstance(buy_result["sanitize_issue"], str))

        with patch("services.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "services.scanner.stock_processor.run_v16_backtest", return_value=candidate_stats
        ):
            candidate_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))
        check("scanner_candidate_status", "candidate", candidate_result["status"])
        check("scanner_candidate_proj_cost_none", None, candidate_result["proj_cost"])
        check("scanner_candidate_expected_value_none", None, candidate_result["expected_value"])
        check("scanner_candidate_sort_value_none", None, candidate_result["sort_value"])

        with patch("services.scanner.stock_processor.sanitize_ohlcv_dataframe", side_effect=skip_exc):
            skip_result = process_single_stock(str(file_path), "2330", params)
        normalized_skip = normalize_scanner_result(skip_result)
        check("scanner_skip_status", "skip_insufficient", normalized_skip["status"])
        check("scanner_skip_message_none", None, normalized_skip["message"])

    summary["best_params_key_count"] = len(best_params_payload)
    return results, summary


def validate_ranking_scoring_sanity_case(base_params):
    params = base_params if base_params is not None else V16StrategyParams()
    case_id = "RANKING_SCORING_SANITY"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'strategy_score')

    ev_low = calc_buy_sort_value("EV", 0.5, 10000, 0.4, 10)
    ev_high = calc_buy_sort_value("EV", 1.5, 10000, 0.4, 10)
    check("buy_sort_ev_monotonic", True, ev_high > ev_low)
    check("buy_sort_ev_type", True, isinstance(ev_high, float))
    check("buy_sort_ev_finite", True, math.isfinite(ev_high))

    cost_low = calc_buy_sort_value("PROJ_COST", 0.5, 5000, 0.4, 10)
    cost_high = calc_buy_sort_value("PROJ_COST", 0.5, 9000, 0.4, 10)
    check("buy_sort_proj_cost_monotonic", True, cost_high > cost_low)

    hist_low = calc_buy_sort_value("HIST_WIN_X_TRADES", 0.5, 10000, 0.35, 8)
    hist_high = calc_buy_sort_value("HIST_WIN_X_TRADES", 0.5, 10000, 0.45, 9)
    check("buy_sort_hist_win_x_trades_monotonic", True, hist_high > hist_low)

    growth_low = calc_buy_sort_value("ASSET_GROWTH", 0.5, 10000, 0.35, 8, 12.0)
    growth_high = calc_buy_sort_value("ASSET_GROWTH", 0.5, 10000, 0.35, 8, 28.0)
    check("buy_sort_asset_growth_monotonic", True, growth_high > growth_low)
    check("buy_sort_asset_growth_type", True, isinstance(growth_high, float))
    check("buy_sort_asset_growth_finite", True, math.isfinite(growth_high))

    overage_below_limit = calc_buy_sort_value(
        "BUY_LIMIT_OVERAGE_THEN_PROJ_COST",
        0.5,
        10000,
        0.4,
        10,
        prev_close=98.0,
        limit_price=100.0,
    )
    overage_above_limit = calc_buy_sort_value(
        "BUY_LIMIT_OVERAGE_THEN_PROJ_COST",
        0.5,
        10000,
        0.4,
        10,
        prev_close=101.0,
        limit_price=100.0,
    )
    overage_far_above_limit = calc_buy_sort_value(
        "BUY_LIMIT_OVERAGE_THEN_PROJ_COST",
        0.5,
        10000,
        0.4,
        10,
        prev_close=103.0,
        limit_price=100.0,
    )
    check("buy_sort_limit_overage_below_limit_zero", 0.0, overage_below_limit)
    check("buy_sort_limit_overage_monotonic", True, overage_below_limit < overage_above_limit < overage_far_above_limit)
    overage_rows = [
        {"ticker": "A", "sort_value": overage_above_limit, "proj_cost": 9000.0},
        {"ticker": "B", "sort_value": overage_below_limit, "proj_cost": 5000.0},
        {"ticker": "C", "sort_value": overage_below_limit, "proj_cost": 8000.0},
    ]
    sort_candidate_rows(overage_rows, "BUY_LIMIT_OVERAGE_THEN_PROJ_COST")
    check("buy_sort_limit_overage_order", ["C", "B", "A"], [row["ticker"] for row in overage_rows])

    entry_type_rows = [
        {"ticker": "EXT_LOW_OVER", "type": "extended", "sort_value": 0.0, "proj_cost": 12000.0, "buy_limit_overage_pct": 0.0},
        {"ticker": "NEW_HIGH_OVER", "type": "normal", "sort_value": 3.0, "proj_cost": 12000.0, "buy_limit_overage_pct": 3.0},
        {"ticker": "RE_LOW_OVER", "type": "reentry", "sort_value": 0.0, "proj_cost": 5000.0, "buy_limit_overage_pct": 0.0},
        {"ticker": "NEW_LOW_OVER", "type": "normal", "sort_value": 1.0, "proj_cost": 8000.0, "buy_limit_overage_pct": 1.0},
    ]
    sort_candidate_rows(entry_type_rows, "ENTRY_TYPE_THEN_PROJ_COST")
    check("buy_sort_entry_type_then_buy_limit_overage_order",
        ["RE_LOW_OVER", "NEW_LOW_OVER", "NEW_HIGH_OVER", "EXT_LOW_OVER"],
        [row["ticker"] for row in entry_type_rows],
    )
    entry_type_sort_value = calc_buy_sort_value("ENTRY_TYPE_THEN_PROJ_COST", 0.5, 7000, 0.4, 10, prev_close=103.0, limit_price=100.0)
    check("buy_sort_entry_type_uses_buy_limit_overage_value", 3.0, entry_type_sort_value)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"):
        score_low_return = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=10.0)
        score_high_return = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=20.0)
        score_worse_mdd = calc_portfolio_score(sys_ret=10.0, sys_mdd=-30.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=20.0)
    check("portfolio_score_return_monotonic", True, score_high_return > score_low_return)
    check("portfolio_score_mdd_monotonic", True, score_high_return > score_worse_mdd)
    check("portfolio_score_romd_finite", True, math.isfinite(score_high_return))

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "LOG_R2"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"):
        score_low_quality = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=40.0, r_sq=0.4, annual_return_pct=20.0)
        score_high_quality = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=60.0, r_sq=0.9, annual_return_pct=20.0)
    check("portfolio_score_log_r2_quality_monotonic", True, score_high_quality > score_low_quality)
    check("portfolio_score_log_r2_finite", True, math.isfinite(score_high_quality))

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
            "close_last": 103.0,
            "stop_loss": 99.0,
            "expected_value": 1.25,
            "win_rate": 55.0,
            "trade_count": 12,
            "asset_growth": 18.0,
            "max_drawdown": 9.5,
            "extended_candidate_today": None,
        }
        with patch("services.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "services.scanner.stock_processor.run_v16_backtest", return_value=buy_stats
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
            prev_close=buy_stats.get("close_last"),
            limit_price=buy_stats.get("buy_limit"),
        )
        check("scanner_sort_value_matches_buy_sort_formula", expected_sort, buy_result["sort_value"])
        check("scanner_sort_value_comparable", True, isinstance(buy_result["sort_value"], float) and math.isfinite(float(buy_result["sort_value"])))

    summary["score_calc_method_default"] = SCORE_CALC_METHOD
    summary["score_numerator_method_default"] = SCORE_NUMERATOR_METHOD
    summary["score_mdd_power_default"] = get_score_mdd_power()
    return results, summary


def validate_score_numerator_option_case(_base_params):
    case_id = "SCORE_NUMERATOR_OPTION"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'strategy_score')

    base_mdd_power = 1.0
    base_mdd_epsilon = 0.0001
    stronger_mdd_power = 2.0

    def _romd_expected(numerator, sys_mdd, *, power=base_mdd_power, epsilon=base_mdd_epsilon):
        return float(numerator) / ((abs(float(sys_mdd)) ** float(power)) + float(epsilon))

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon):
        annual_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0)
    expected_annual = _romd_expected(18.0, -20.0)
    check("annual_return_numerator_formula", expected_annual, annual_score)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon):
        total_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0)
    expected_total = _romd_expected(12.0, -20.0)
    check("total_return_numerator_formula", expected_total, total_score)
    check("numerator_switch_changes_score_when_returns_differ", True, annual_score != total_score)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", stronger_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon):
        powered_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0)
    expected_powered = _romd_expected(12.0, -20.0, power=stronger_mdd_power)
    check("mdd_power_denominator_formula", expected_powered, powered_score)
    check("mdd_power_stronger_than_base_penalty", True, powered_score < total_score)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "LOG_R2"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_RETURN"):
        low_total_score = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=40.0, r_sq=0.4, annual_return_pct=40.0)
        high_total_score = calc_portfolio_score(sys_ret=10.0, sys_mdd=-20.0, m_win_rate=60.0, r_sq=0.9, annual_return_pct=40.0)
    check("log_r2_total_return_quality_monotonic", True, high_total_score > low_total_score)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_R"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", False):
        total_r_low = calc_portfolio_score(sys_ret=500.0, sys_mdd=-5.0, m_win_rate=90.0, r_sq=0.99, annual_return_pct=80.0, total_r=12.5)
        total_r_high = calc_portfolio_score(sys_ret=-20.0, sys_mdd=-5.0, m_win_rate=10.0, r_sq=0.10, annual_return_pct=-5.0, total_r=24.0)
    check("total_r_numerator_formula", _romd_expected(12.5, -5.0), total_r_low)
    check("total_r_numerator_monotonic_when_mdd_equal", True, total_r_high > total_r_low)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_R_X_PORTFOLIO_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", False):
        total_r_x_return_positive = calc_portfolio_score(sys_ret=150.0, sys_mdd=-5.0, m_win_rate=90.0, r_sq=0.99, annual_return_pct=80.0, total_r=12.5)
        total_r_x_return_negative = calc_portfolio_score(sys_ret=-20.0, sys_mdd=-5.0, m_win_rate=90.0, r_sq=0.99, annual_return_pct=80.0, total_r=12.5)
    check("portfolio_return_multiplier_pct_to_multiple", 1.5, calc_score_portfolio_return_multiplier(150.0))
    check("portfolio_return_multiplier_clamps_negative", 0.0, calc_score_portfolio_return_multiplier(-20.0))
    check("total_r_x_portfolio_return_numerator_formula", _romd_expected(12.5 * 1.5, -5.0), total_r_x_return_positive)
    check("total_r_x_portfolio_return_zero_when_portfolio_loss", 0.0, total_r_x_return_negative)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_R_X_ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", False):
        total_r_x_annual_positive = calc_portfolio_score(sys_ret=150.0, sys_mdd=-5.0, m_win_rate=90.0, r_sq=0.99, annual_return_pct=80.0, total_r=12.5)
        total_r_x_annual_negative = calc_portfolio_score(sys_ret=150.0, sys_mdd=-5.0, m_win_rate=90.0, r_sq=0.99, annual_return_pct=-20.0, total_r=12.5)
        total_r_x_annual_missing = calc_portfolio_score(sys_ret=150.0, sys_mdd=-5.0, m_win_rate=90.0, r_sq=0.99, annual_return_pct=None, total_r=12.5)
    check("positive_return_multiplier_pct_to_multiple", 0.8, calc_score_positive_return_multiplier(80.0))
    check("positive_return_multiplier_clamps_negative", 0.0, calc_score_positive_return_multiplier(-20.0))
    check("total_r_x_annual_return_numerator_formula", _romd_expected(12.5 * 0.8, -5.0), total_r_x_annual_positive)
    check("total_r_x_annual_return_zero_when_annual_loss", 0.0, total_r_x_annual_negative)
    check("total_r_x_annual_return_missing_uses_total_return_fallback", _romd_expected(12.5 * 1.5, -5.0), total_r_x_annual_missing)

    replay_result = (
        None,
        None,
        141.48,
        -17.56,
        10,
        40.0,
        0.4,
        2.0,
        2414813.0,
        79.32,
        None,
        203.67,
        -33.96,
        0,
        0,
        0.7,
        58.73,
        0.6,
        61.90,
        10,
        0,
        10.0,
        95.0,
        18.64,
        24.03,
        {
            "score_total_r": 728.1,
            "score_median_r": -0.117,
            "single_stock_total_r": 728.1,
            "single_stock_median_r": -0.117,
            "min_full_year_return_pct": -5.89,
            "bm_min_full_year_return_pct": -21.84,
        },
    )
    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_R_X_ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED", False):
        _candidate_metrics, benchmark_metrics, _range_text = optimizer_callbacks._portfolio_replay_metrics_from_result(replay_result, initial_capital=1000000.0)
    check("benchmark_oos_display_uses_plain_romd_under_total_r_numerator", calc_plain_romd(203.67, -33.96), benchmark_metrics["pf_romd"])

    from tools.optimizer.outer_rolling_oos import _render_results_table
    optimizer_table = _render_results_table(
        [
            {
                "fold": "1/1",
                "selection_period": "2011-01-01~2020-12-31",
                "oos_period": "2021-01-01~latest",
                "best_finalist_oos_score": 1.94633,
                "benchmark_oos_score": calc_plain_romd(203.67, -33.96),
                "base_finalists_agree": {
                    "available": True,
                    "rank_1_oos": 1.94633,
                    "rank_1_return_pct": 148.26,
                    "rank_1_mdd_pct": -18.60,
                },
            }
        ],
        color=False,
        include_chain=False,
        policy_names=("base_finalists_agree",),
        table_title="FINALISTS AGREE RESULTS",
    )
    check("optimizer_oos_table_hides_system_score_column", True, "score" not in optimizer_table and "1946.33" not in optimizer_table)
    check("optimizer_oos_table_keeps_plain_romd_and_0050_compare", True, "RoMD" in optimizer_table and "7.97" in optimizer_table and "6.00 (+1.97)" in optimizer_table)

    single_stock_score_fields = build_score_single_stock_profile_fields({
        "trade_count": 2854,
        "win_rate": 46.39,
        "payoff_r": 3.04,
        "avg_r": 0.601,
        "median_r": -0.109,
        "total_r": 1715.0,
    })
    check("total_r_score_source_is_single_stock", 1715.0, single_stock_score_fields["score_total_r"])
    check("median_r_score_source_is_single_stock", -0.109, single_stock_score_fields["score_median_r"])
    check("score_r_source_label_is_single_stock", "single_stock", single_stock_score_fields["score_r_source"])

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_R"), patch("config.training_policy.SCORE_MDD_POWER", stronger_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", False):
        total_r_powered_score = calc_portfolio_score(sys_ret=500.0, sys_mdd=-5.0, m_win_rate=90.0, r_sq=0.99, annual_return_pct=80.0, total_r=12.5)
    check("total_r_numerator_still_uses_mdd_denominator", True, total_r_powered_score < total_r_low)

    old_linear_low = 40.0 / 50.0
    old_linear_high = 50.0 / 50.0
    new_amp_low = calc_score_win_rate_multiplier(40.0, 50.0)
    new_amp_mid = calc_score_win_rate_multiplier(50.0, 50.0)
    new_amp_high = calc_score_win_rate_multiplier(60.0, 50.0)
    check("score_win_rate_amp_hits_target_at_one", 1.0, new_amp_mid)
    check("score_win_rate_amp_stronger_than_old_linear_below_target", True, new_amp_low < old_linear_low)
    check("score_win_rate_amp_boosts_above_target", True, new_amp_high > new_amp_mid)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", True), patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_TARGET", 50.0), patch("config.training_policy.SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", False):
        low_monthly_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=40.0, r_sq=0.8, annual_return_pct=18.0)
        target_monthly_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0)
        high_monthly_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=60.0, r_sq=0.8, annual_return_pct=18.0)
    check("portfolio_score_monthly_win_rate_amp_monotonic", True, low_monthly_score < target_monthly_score < high_monthly_score)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_WIN_RATE_AMP_ENABLED", True), patch("config.training_policy.SCORE_WIN_RATE_TARGET", 50.0):
        low_win_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, trade_win_rate_pct=40.0)
        target_win_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, trade_win_rate_pct=50.0)
        high_win_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, trade_win_rate_pct=60.0)
    check("portfolio_score_win_rate_amp_monotonic", True, low_win_score < target_win_score < high_win_score)
    check("portfolio_score_win_rate_amp_beats_old_linear_gap", True, (target_win_score / low_win_score) > (old_linear_high / old_linear_low))

    min_year_low_amp = calc_score_min_full_year_return_multiplier(-20.0, -35.0, 0.0)
    min_year_target_amp = calc_score_min_full_year_return_multiplier(0.0, -35.0, 0.0)
    min_year_high_amp = calc_score_min_full_year_return_multiplier(10.0, -35.0, 0.0)
    check("score_min_full_year_return_amp_hits_target_at_one", 1.0, min_year_target_amp)
    check("score_min_full_year_return_amp_monotonic", True, min_year_low_amp < min_year_target_amp < min_year_high_amp)

    min_month_low_amp = calc_score_min_month_return_multiplier(-2.0, -5.0, 5.0)
    min_month_target_amp = calc_score_min_month_return_multiplier(5.0, -5.0, 5.0)
    min_month_high_amp = calc_score_min_month_return_multiplier(8.0, -5.0, 5.0)
    check("score_min_month_return_amp_hits_target_at_one", 1.0, min_month_target_amp)
    check("score_min_month_return_amp_monotonic", True, min_month_low_amp < min_month_target_amp < min_month_high_amp)

    min_quarter_low_amp = calc_score_min_quarter_return_multiplier(-5.0, -10.0, 10.0)
    min_quarter_target_amp = calc_score_min_quarter_return_multiplier(10.0, -10.0, 10.0)
    min_quarter_high_amp = calc_score_min_quarter_return_multiplier(15.0, -10.0, 10.0)
    check("score_min_quarter_return_amp_hits_target_at_one", 1.0, min_quarter_target_amp)
    check("score_min_quarter_return_amp_monotonic", True, min_quarter_low_amp < min_quarter_target_amp < min_quarter_high_amp)

    median_r_low_amp = calc_score_median_r_multiplier(-0.5, -1.0, 0.0)
    median_r_target_amp = calc_score_median_r_multiplier(0.0, -1.0, 0.0)
    median_r_high_amp = calc_score_median_r_multiplier(0.5, -1.0, 0.0)
    check("score_median_r_amp_hits_target_at_one", 1.0, median_r_target_amp)
    check("score_median_r_amp_monotonic", True, median_r_low_amp < median_r_target_amp < median_r_high_amp)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED", True), patch("config.training_policy.MIN_FULL_YEAR_RETURN_PCT", -35.0), patch("config.training_policy.SCORE_MIN_FULL_YEAR_RETURN_TARGET", 0.0):
        low_min_year_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_full_year_return_pct=-20.0)
        target_min_year_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_full_year_return_pct=0.0)
        high_min_year_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_full_year_return_pct=10.0)
    check("portfolio_score_min_full_year_return_amp_monotonic", True, low_min_year_score < target_min_year_score < high_min_year_score)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_MONTH_RETURN_AMP_ENABLED", True), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_MONTH_RETURN_TARGET", 5.0), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", False):
        low_min_month_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_month_return_pct=-2.0)
        target_min_month_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_month_return_pct=5.0)
        high_min_month_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_month_return_pct=8.0)
    check("portfolio_score_min_month_return_amp_monotonic", True, low_min_month_score < target_min_month_score < high_min_month_score)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_MONTH_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon), patch("config.training_policy.SCORE_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", True), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_TARGET", 10.0), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", False):
        low_min_quarter_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_quarter_return_pct=-5.0)
        target_min_quarter_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_quarter_return_pct=10.0)
        high_min_quarter_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=18.0, min_quarter_return_pct=15.0)
    check("portfolio_score_min_quarter_return_amp_monotonic", True, low_min_quarter_score < target_min_quarter_score < high_min_quarter_score)

    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "TOTAL_R"), patch("config.training_policy.SCORE_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_MEDIAN_R_AMP_ENABLED", True), patch("config.training_policy.SCORE_MEDIAN_R_FLOOR", -1.0), patch("config.training_policy.SCORE_MEDIAN_R_TARGET", 0.0):
        low_median_r_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, total_r=10.0, median_r=-0.5)
        target_median_r_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, total_r=10.0, median_r=0.0)
        high_median_r_score = calc_portfolio_score(sys_ret=12.0, sys_mdd=-20.0, m_win_rate=50.0, r_sq=0.8, total_r=10.0, median_r=0.5)
    check("portfolio_score_median_r_amp_monotonic", True, low_median_r_score < target_median_r_score < high_median_r_score)

    annual_missing = None
    with patch("config.training_policy.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False), patch("config.training_policy.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False), patch("config.training_policy.SCORE_CALC_METHOD", "RoMD"), patch("config.training_policy.SCORE_NUMERATOR_METHOD", "ANNUAL_RETURN"), patch("config.training_policy.SCORE_MDD_POWER", base_mdd_power), patch("config.training_policy.SCORE_MDD_DENOMINATOR_EPSILON", base_mdd_epsilon):
        annual_missing = calc_portfolio_score(sys_ret=9.0, sys_mdd=-15.0, m_win_rate=50.0, r_sq=0.8, annual_return_pct=None)
    expected_fallback = _romd_expected(9.0, -15.0)
    check("annual_return_numerator_falls_back_to_total_return_when_missing", expected_fallback, annual_missing)

    current_mdd_power = get_score_mdd_power()
    current_mdd_epsilon = get_score_mdd_denominator_epsilon()
    check("score_mdd_power_setting_accepts_manual_training_policy_value", True, math.isfinite(current_mdd_power) and current_mdd_power >= 0.0)
    check("score_mdd_epsilon_setting_accepts_manual_training_policy_value", True, math.isfinite(current_mdd_epsilon) and current_mdd_epsilon > 0.0)

    summary["score_calc_method_default"] = SCORE_CALC_METHOD
    summary["score_numerator_method_default"] = SCORE_NUMERATOR_METHOD
    summary["score_mdd_power_default"] = current_mdd_power
    summary["score_mdd_denominator_epsilon_default"] = current_mdd_epsilon
    summary["score_monthly_win_rate_amp_enabled_default"] = bool(__import__("config.training_policy", fromlist=["SCORE_MONTHLY_WIN_RATE_AMP_ENABLED"]).SCORE_MONTHLY_WIN_RATE_AMP_ENABLED)
    summary["score_monthly_win_rate_target_default"] = float(__import__("config.training_policy", fromlist=["SCORE_MONTHLY_WIN_RATE_TARGET"]).SCORE_MONTHLY_WIN_RATE_TARGET)
    summary["score_min_full_year_return_amp_enabled_default"] = bool(__import__("config.training_policy", fromlist=["SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED"]).SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED)
    summary["score_min_full_year_return_target_default"] = float(__import__("config.training_policy", fromlist=["SCORE_MIN_FULL_YEAR_RETURN_TARGET"]).SCORE_MIN_FULL_YEAR_RETURN_TARGET)
    summary["score_min_month_return_amp_enabled_default"] = bool(__import__("config.training_policy", fromlist=["SCORE_MIN_MONTH_RETURN_AMP_ENABLED"]).SCORE_MIN_MONTH_RETURN_AMP_ENABLED)
    summary["score_min_month_return_target_default"] = float(__import__("config.training_policy", fromlist=["SCORE_MIN_MONTH_RETURN_TARGET"]).SCORE_MIN_MONTH_RETURN_TARGET)
    summary["score_min_quarter_return_amp_enabled_default"] = bool(__import__("config.training_policy", fromlist=["SCORE_MIN_QUARTER_RETURN_AMP_ENABLED"]).SCORE_MIN_QUARTER_RETURN_AMP_ENABLED)
    summary["score_min_quarter_return_target_default"] = float(__import__("config.training_policy", fromlist=["SCORE_MIN_QUARTER_RETURN_TARGET"]).SCORE_MIN_QUARTER_RETURN_TARGET)
    summary["score_median_r_amp_enabled_default"] = bool(__import__("config.training_policy", fromlist=["SCORE_MEDIAN_R_AMP_ENABLED"]).SCORE_MEDIAN_R_AMP_ENABLED)
    summary["score_median_r_floor_default"] = float(__import__("config.training_policy", fromlist=["SCORE_MEDIAN_R_FLOOR"]).SCORE_MEDIAN_R_FLOOR)
    summary["score_median_r_target_default"] = float(__import__("config.training_policy", fromlist=["SCORE_MEDIAN_R_TARGET"]).SCORE_MEDIAN_R_TARGET)
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
            "close_last": 103.0,
            "stop_loss": 99.0,
            "expected_value": 1.25,
            "win_rate": 55.0,
            "trade_count": 12,
            "max_drawdown": 9.5,
            "extended_candidate_today": None,
        }
        with patch("services.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "services.scanner.stock_processor.run_v16_backtest", return_value=buy_stats
        ):
            scanner_result_1 = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))
            scanner_result_2 = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))

    add_check(results, "strategy_repeatability", case_id, "scanner_result_repeatable", scanner_result_1, scanner_result_2)

    def _run_objective_once():
        session = _FakeOptimizerSession(fixed_tp_percent=0.25)
        trial = _FakeOptunaTrial(
            number=2,
            preset_values=_optimizer_synthetic_trial_values(prefer_enabled=True),
        )
        perf_counter_values = iter([0.00, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        with patch("services.optimizer.objective_runner.prepare_trial_inputs", return_value=_build_fake_prepare_result(master_dates=["2026-01-02", "2026-01-03"])), patch(
            "services.optimizer.objective_runner.run_portfolio_timeline",
            side_effect=_make_fake_portfolio_runner(
                ret_pct=26.0,
                mdd=-9.0,
                annual_return_pct=18.0,
                yearly_return_rows=[{"year": 2024, "return_pct": 12.5}, {"year": 2025, "return_pct": 9.3}],
            ),
        ), patch("services.optimizer.objective_runner.apply_filter_rules", return_value=None), patch(
            "services.optimizer.objective_runner.calc_portfolio_score", return_value=88.123
        ), patch("services.optimizer.objective_runner.time.perf_counter", side_effect=lambda: next(perf_counter_values)):
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
    results, summary, check, check_true = bind_synthetic_case(case_id, 'strategy_viability')

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
            "close_last": 103.0,
            "stop_loss": 99.0,
            "expected_value": 1.25,
            "win_rate": 55.0,
            "trade_count": 12,
            "max_drawdown": 9.5,
            "extended_candidate_today": None,
        }
        with patch("services.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "services.scanner.stock_processor.run_v16_backtest", return_value=buy_stats
        ):
            scanner_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))
    check("scanner_smoke_status", "buy", scanner_result["status"])

    session = _FakeOptimizerSession(fixed_tp_percent=0.25)
    trial = _FakeOptunaTrial(
        number=3,
        preset_values=_optimizer_synthetic_trial_values(prefer_enabled=True),
    )
    perf_counter_values = iter([0.00, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    with patch("services.optimizer.objective_runner.prepare_trial_inputs", return_value=_build_fake_prepare_result(master_dates=["2026-01-02", "2026-01-03"])), patch(
        "services.optimizer.objective_runner.run_portfolio_timeline",
        side_effect=_make_fake_portfolio_runner(
            ret_pct=26.0,
            mdd=-9.0,
            annual_return_pct=18.0,
            yearly_return_rows=[{"year": 2024, "return_pct": 12.5}, {"year": 2025, "return_pct": 9.3}],
        ),
    ), patch("services.optimizer.objective_runner.apply_filter_rules", return_value=None), patch(
        "services.optimizer.objective_runner.calc_portfolio_score", return_value=88.123
    ), patch("services.optimizer.objective_runner.time.perf_counter", side_effect=lambda: next(perf_counter_values)):
        optimizer_value = run_optimizer_objective(session, trial)
    check("optimizer_smoke_returns_score", 88.123, optimizer_value)

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
    check("scanner_reporting_smoke_runs", True, "明日候選清單" in scanner_summary_text and scanner_issue_log_path in scanner_summary_text)

    yearly_buffer = io.StringIO()
    yearly_rows = [
        {"year": 2024, "year_return_pct": 12.5, "is_full_year": True, "start_date": "2024-01-02", "end_date": "2024-12-31"},
        {"year": 2025, "year_return_pct": 9.3, "is_full_year": False, "start_date": "2025-01-02", "end_date": "2025-06-30"},
    ]
    with redirect_stdout(yearly_buffer):
        df_yearly = print_yearly_return_report(yearly_rows)
    check("portfolio_reporting_smoke_runs", 2, len(df_yearly))

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
            extended_trades=0,
            reentry_trades=1,
            annual_trades=12.5,
            reserved_buy_fill_rate=78.0,
            annual_return_pct=18.0,
            bm_annual_return_pct=6.1,
            min_full_year_return_pct=5.5,
            bm_min_full_year_return_pct=1.2,
        )
    check("strategy_dashboard_smoke_runs", True, "【訓練參數】" in dashboard_buffer.getvalue() and "系統得分" in dashboard_buffer.getvalue())

    summary["scanner_status"] = scanner_result["status"]
    summary["optimizer_value"] = optimizer_value
    summary["yearly_rows"] = len(df_yearly)
    return results, summary


def validate_strategy_reporting_schema_compatibility_case(base_params):
    params = base_params if base_params is not None else V16StrategyParams()
    case_id = "STRATEGY_REPORTING_SCHEMA_COMPATIBILITY"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'strategy_reporting')

    default_payload = params_to_json_dict(V16StrategyParams())
    export_trial = SimpleNamespace(
        number=4,
        params=_optimizer_synthetic_param_payload(prefer_enabled=True),
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
            "close_last": 103.0,
            "stop_loss": 99.0,
            "expected_value": 1.25,
            "win_rate": 55.0,
            "trade_count": 12,
            "max_drawdown": 9.5,
            "extended_candidate_today": None,
        }
        with patch("services.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "services.scanner.stock_processor.run_v16_backtest", return_value=buy_stats
        ):
            scanner_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", params))

    check("best_params_export_status", 0, export_status)
    check("best_params_export_keys_match_strategy_schema", sorted(default_payload.keys()), sorted(exported_payload.keys()))

    expected_scanner_keys = ["expected_value", "message", "proj_cost", "sanitize_issue", "sort_value", "status", "ticker"]
    check("scanner_normalized_payload_keys_stable", expected_scanner_keys, sorted(scanner_result.keys()))

    yearly_rows = [
        {"year": 2024, "year_return_pct": 12.5, "is_full_year": True, "start_date": "2024-01-02", "end_date": "2024-12-31"},
        {"year": 2025, "year_return_pct": 9.3, "is_full_year": False, "start_date": "2025-01-02", "end_date": "2025-06-30"},
    ]
    with redirect_stdout(io.StringIO()):
        df_yearly = print_yearly_return_report(yearly_rows)
    check("yearly_report_columns_stable",
        ["year", "year_return_pct", "is_full_year", "start_date", "end_date", "year_label", "year_type"],
        list(df_yearly.columns),
    )
    check("yearly_report_year_type_values", ["完整", "非完整"], df_yearly["year_type"].tolist())

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
    results, summary, check, check_true = bind_synthetic_case(case_id, 'strategy_contract')

    runtime_session = _FakeOptimizerSession(
        fixed_strategy_param_overrides={"use_breakout_quality_ranking": True},
        runtime_cache_identity="synthetic-pit-identity",
    )
    check("optimizer_test_session_inherits_canonical_session_contract",
        True,
        isinstance(runtime_session, OptimizerSession),
    )
    overridden_params = runtime_session.apply_fixed_strategy_param_overrides(
        V16StrategyParams(use_breakout_quality_ranking=False)
    )
    with runtime_session.optimizer_runtime_context():
        runtime_context_entered = True
    check("optimizer_test_session_mirrors_fixed_override_and_runtime_identity_hooks",
        (True, "synthetic-pit-identity", True),
        (
            overridden_params.use_breakout_quality_ranking,
            runtime_session.runtime_cache_identity,
            runtime_context_entered,
        ),
    )

    sample_high_len = _optimizer_search_space_sample_value("high_len")
    explicit_tp_params = build_optimizer_trial_params(
        {"high_len": sample_high_len, "tp_percent": 0.11},
        user_attrs={"fixed_tp_percent": 0.22},
        fixed_tp_percent=0.33,
    )
    check("tp_percent_prefers_trial_params", 0.11, explicit_tp_params["tp_percent"])

    attr_tp_params = build_optimizer_trial_params(
        {"high_len": sample_high_len},
        user_attrs={"fixed_tp_percent": 0.22},
        fixed_tp_percent=0.33,
    )
    check("tp_percent_falls_back_to_user_attr", 0.22, attr_tp_params["tp_percent"])

    fixed_tp_params = build_optimizer_trial_params({"high_len": sample_high_len}, user_attrs={}, fixed_tp_percent=0.33)
    check("tp_percent_falls_back_to_fixed_setting", 0.33, fixed_tp_params["tp_percent"])

    missing_tp_raises = raises_expected(ValueError, lambda: build_optimizer_trial_params({"high_len": sample_high_len}, user_attrs={}, fixed_tp_percent=None))
    check("tp_percent_missing_everywhere_is_fail_fast", True, missing_tp_raises)

    filter_fail_session = _FakeOptimizerSession(fixed_tp_percent=0.25)
    filter_fail_trial = _FakeOptunaTrial(
        number=0,
        preset_values=_optimizer_synthetic_trial_values(prefer_enabled=True),
    )
    with patch("services.optimizer.objective_runner.prepare_trial_inputs", return_value=_build_fake_prepare_result(master_dates=["2026-01-02"])), patch(
        "services.optimizer.objective_runner.run_portfolio_timeline",
        side_effect=_make_fake_portfolio_runner(
            ret_pct=18.0,
            mdd=-11.0,
            annual_return_pct=14.0,
            yearly_return_rows=[{"year": 2024, "return_pct": 8.5}],
        ),
    ), patch("services.optimizer.objective_runner.apply_filter_rules", return_value="月勝率偏低 (30%)"):
        filter_fail_value = run_optimizer_objective(filter_fail_session, filter_fail_trial)

    filter_fail_profile = filter_fail_trial.user_attrs.get("profile_row", {})
    check("objective_filter_fail_returns_invalid_trial_value", INVALID_TRIAL_VALUE, filter_fail_value)
    check("objective_filter_fail_sets_fail_reason", "月勝率偏低 (30%)", filter_fail_trial.user_attrs.get("fail_reason"))
    check("objective_filter_fail_profile_trial_value", INVALID_TRIAL_VALUE, filter_fail_profile.get("trial_value"))
    check("objective_filter_fail_profile_reason", "月勝率偏低 (30%)", filter_fail_profile.get("fail_reason"))
    check("objective_filter_fail_profile_row_recorded", 1, len(filter_fail_session.profile_recorder.rows))
    check("objective_filter_fail_records_only_insufficient_prep_failures", [("2330", "有效資料不足: 2330")], filter_fail_session.recorded_prep_failures)
    check("objective_filter_fail_does_not_set_pf_return", False, "pf_return" in filter_fail_trial.user_attrs)

    success_session = _FakeOptimizerSession(fixed_tp_percent=0.25)
    success_trial = _FakeOptunaTrial(
        number=1,
        preset_values=_optimizer_synthetic_trial_values(prefer_enabled=True),
    )
    with patch("services.optimizer.objective_runner.prepare_trial_inputs", return_value=_build_fake_prepare_result(master_dates=["2026-01-02", "2026-01-03"])), patch(
        "services.optimizer.objective_runner.run_portfolio_timeline",
        side_effect=_make_fake_portfolio_runner(
            ret_pct=26.0,
            mdd=-9.0,
            annual_return_pct=18.0,
            yearly_return_rows=[{"year": 2024, "return_pct": 12.5}, {"year": 2025, "return_pct": 9.3}],
        ),
    ), patch("services.optimizer.objective_runner.apply_filter_rules", return_value=None), patch(
        "services.optimizer.objective_runner.calc_portfolio_score", return_value=88.123
    ):
        success_value = run_optimizer_objective(success_session, success_trial)

    success_profile = success_trial.user_attrs.get("profile_row", {})
    check("objective_success_returns_score", 88.123, success_value)
    check("objective_success_sets_base_score", 88.123, success_trial.user_attrs.get("base_score"))
    check("objective_success_sets_pf_return", 26.0, success_trial.user_attrs.get("pf_return"))
    check("objective_success_sets_yearly_rows", 2, len(success_trial.user_attrs.get("yearly_return_rows", [])))
    check("objective_success_profile_trial_value", 88.123, success_profile.get("trial_value"))
    check("objective_success_profile_reason_empty", "", success_profile.get("fail_reason"))
    check("objective_success_profile_row_recorded", 1, len(success_session.profile_recorder.rows))

    export_colors = {"red": "", "green": "", "reset": ""}
    qualified_high_len = _optimizer_search_space_sample_value("high_len", step_offset=3)
    qualified_atr_len = _optimizer_search_space_sample_value("atr_len", step_offset=3)
    qualified_atr_buy_tol = _optimizer_search_space_sample_value("atr_buy_tol", step_offset=3, add_float_tail=True)
    qualified_min_history_ev = _optimizer_search_space_sample_value("min_history_ev", step_offset=1, add_float_tail=True)
    qualified_export_trial = SimpleNamespace(
        number=3,
        params={
            "high_len": qualified_high_len,
            "atr_len": qualified_atr_len,
            "atr_buy_tol": qualified_atr_buy_tol,
            "min_history_ev": qualified_min_history_ev,
            "tp_percent": 0.30000000000000004,
        },
        user_attrs={},
        value=88.123,
    )
    rejected_export_trial = SimpleNamespace(
        number=2,
        params={
            "high_len": _optimizer_search_space_sample_value("high_len"),
            "atr_len": _optimizer_search_space_sample_value("atr_len"),
        },
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

    run_bound_checks(
        check,
        (
            ('export_best_params_success_status', 0, success_status,),
            ('export_best_params_uses_best_trial_tp_percent', 0.3, exported_payload['tp_percent'],),
            ('export_best_params_canonicalizes_tp_percent_step_float', '0.3', repr(exported_payload['tp_percent']),),
            ('export_best_params_preserves_best_trial_high_len', qualified_high_len, exported_payload['high_len'],),
            ('export_best_params_canonicalizes_atr_buy_tol_step_float', _canonicalize_optimizer_export_repr('atr_buy_tol', qualified_atr_buy_tol), repr(exported_payload['atr_buy_tol']),),
            ('export_best_params_canonicalizes_min_history_ev_step_float', _canonicalize_optimizer_export_repr('min_history_ev', qualified_min_history_ev), repr(exported_payload['min_history_ev']),),
            ('export_best_params_keeps_default_buy_fee_canonical_decimal', '0.000399', repr(exported_payload['buy_fee']),),
            ('export_best_params_keeps_default_sell_fee_canonical_decimal', '0.000399', repr(exported_payload['sell_fee']),),
            ('export_best_params_failure_status_for_unqualified_best_trial', 1, failure_status,),
            ('export_best_params_failure_does_not_create_payload', False, failure_export_path.exists(),),
        ),
    )

    canonical_model_files = _existing_shipped_reference_param_paths()
    canonical_field_names = set(_optimizer_export_canonical_decimal_places()) | {"buy_fee", "sell_fee"}
    shipped_repr_map = {}
    expected_shipped_repr_map = {}
    for artifact_path in canonical_model_files:
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        for member_idx, param_payload in enumerate(_extract_reference_param_payloads(artifact_payload), start=1):
            for field_name in canonical_field_names:
                if field_name not in param_payload:
                    continue
                if not _optimizer_dependent_field_is_active(field_name, param_payload):
                    continue
                map_key = f"{artifact_path.name}::member#{member_idx}::{field_name}"
                shipped_repr_map[map_key] = repr(param_payload[field_name])
                if field_name in {"buy_fee", "sell_fee"}:
                    expected_shipped_repr_map[map_key] = "0.000399"
                else:
                    expected_shipped_repr_map[map_key] = _canonicalize_optimizer_export_repr(field_name, param_payload[field_name])

    check("repo_shipped_reference_artifacts_use_canonical_optimizer_decimal_repr",
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
    check("optimizer_promote_rejects_stale_run_best_policy_baseline", False, should_promote_stale_policy)
    check("optimizer_promote_reports_stale_run_best_policy_baseline", True, "effective policy" in stale_policy_reason)

    robustness = importlib.import_module("tools.optimizer.robustness")
    training_policy = importlib.import_module("config.training_policy")
    training_policy_runtime = importlib.import_module("core.training_policy")
    no_neighbor_session = SimpleNamespace()
    no_neighbor_trial = SimpleNamespace(number=99, user_attrs={"base_score": 42.0}, value=42.0)
    with patch.object(robustness, "_build_neighbor_candidates", return_value=[]):
        no_neighbor_local_min = robustness.compute_local_min_score(no_neighbor_session, no_neighbor_trial)
    expected_no_neighbor_local_min = INVALID_TRIAL_VALUE if training_policy_runtime.is_optimizer_local_min_review_enabled() else 42.0
    check("local_min_score_no_legal_neighbor_respects_training_policy", expected_no_neighbor_local_min, no_neighbor_local_min)

    previous_runtime_mode = training_policy_runtime.resolve_optimizer_runtime_model_mode()
    try:
        training_policy_runtime.set_optimizer_runtime_model_mode("full")
        full_mode_local_enabled = training_policy_runtime.is_optimizer_local_min_review_enabled()
        full_mode_local_policy_names = training_policy_runtime.resolve_optimizer_enabled_policy_indicators((
            "base_finalist_best",
            "local_finalist_best",
            "retention_finalist_best",
            "base_finalists_agree",
            "local_finalists_agree",
            "retention_finalists_agree",
            "base",
            "local",
            "retention",
        ))
        training_policy_runtime.set_optimizer_runtime_model_mode("oos")
        oos_mode_local_enabled = training_policy_runtime.is_optimizer_local_min_review_enabled()
    finally:
        training_policy_runtime.set_optimizer_runtime_model_mode(previous_runtime_mode)
    check("optimizer_full_mode_local_min_review_default_disabled", False, full_mode_local_enabled)
    check("optimizer_full_mode_local_retention_policy_indicators_disabled_by_default", ("base_finalist_best", "base_finalists_agree", "base"), full_mode_local_policy_names)
    check("optimizer_oos_mode_preserves_global_local_min_review_default", bool(training_policy.OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED), oos_mode_local_enabled)

    outer_rolling_oos = importlib.import_module("tools.optimizer.outer_rolling_oos")
    check("optimizer_full_seed_ensemble_base_filename_has_ensemble_dimension", "full_ensemble_base.json", outer_rolling_oos.get_optimizer_nonrolling_policy_paramset_filename("base", mode="full"))
    check("optimizer_oos_seed_ensemble_base_filename_has_ensemble_dimension", "oos_ensemble_base.json", outer_rolling_oos.get_optimizer_nonrolling_policy_paramset_filename("base", mode="oos"))
    check("optimizer_roos_seed_ensemble_base_filename_has_ensemble_dimension", "roos_ensemble_base.json", outer_rolling_oos.get_optimizer_policy_paramset_filename("base"))
    check("optimizer_study_legacy_base_filename_remains_stable", "base.json", outer_rolling_oos.get_optimizer_nonrolling_policy_paramset_filename("base", mode="study"))

    check("optimizer_default_interactive_trials_is_1000",
        1000,
        DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE,
    )
    sample_trial_count = 1000
    small_trial_count = 100
    expected_sample_top_k = max(
        int(training_policy.OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN),
        int(math.ceil(sample_trial_count * float(training_policy.OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE))),
    )
    expected_small_top_k = max(
        int(training_policy.OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN),
        int(math.ceil(small_trial_count * float(training_policy.OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE))),
    )
    run_bound_checks(
        check,
        (
            ('local_min_score_finalist_top_k_respects_training_policy_rate', expected_sample_top_k, training_policy_runtime.resolve_optimizer_local_min_score_finalist_top_k(sample_trial_count),),
            ('local_min_score_finalist_top_k_respects_training_policy_floor', expected_small_top_k, training_policy_runtime.resolve_optimizer_local_min_score_finalist_top_k(small_trial_count),),
            ('local_min_score_finalist_top_k_uses_session_trial_count_by_default', expected_sample_top_k, robustness._resolve_local_min_score_finalist_top_k(SimpleNamespace(n_trials=sample_trial_count)),),
        ),
    )

    summary["success_export_key_count"] = len(exported_payload)
    summary["registry_contract_checks"] = len(results)
    return results, summary



def validate_optimizer_interrupt_export_contract_case(_base_params):
    from tools.optimizer.runtime import resolve_training_session_export_policy

    case_id = "OPTIMIZER_INTERRUPT_EXPORT_CONTRACT"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'strategy_contract')

    export_only_allowed, export_only_reason = resolve_training_session_export_policy(
        requested_n_trials=0,
        completed_session_trials=0,
        interrupted=False,
    )
    check("export_policy_allows_export_only_mode", True, export_only_allowed)
    check("export_policy_export_only_reason", "export_only", export_only_reason)

    target_reached_allowed, target_reached_reason = resolve_training_session_export_policy(
        requested_n_trials=5,
        completed_session_trials=5,
        interrupted=True,
    )
    check("export_policy_allows_export_when_target_reached", True, target_reached_allowed)
    check("export_policy_target_reached_reason", "target_reached", target_reached_reason)

    interrupted_allowed, interrupted_reason = resolve_training_session_export_policy(
        requested_n_trials=5,
        completed_session_trials=2,
        interrupted=True,
    )
    check("export_policy_blocks_interrupted_partial_session", False, interrupted_allowed)
    check("export_policy_interrupted_reason", "interrupted_before_target", interrupted_reason)

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
            self.session_ts = "synthetic"
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
        params={
            "high_len": _optimizer_search_space_sample_value("high_len", step_offset=3),
            "atr_len": _optimizer_search_space_sample_value("atr_len", step_offset=3),
        },
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
            stack.enter_context(patch.object(optimizer_main, "OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED", False))
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
    check("optimizer_main_interrupt_returns_zero", 0, rc)
    check("optimizer_main_interrupt_does_not_call_export", 0, len(fake_export_calls))
    check("optimizer_main_interrupt_preserves_existing_run_best_params_json", "keep", persisted_payload.get("marker"))
    check("optimizer_main_interrupt_reports_warning", True, "使用者中斷訓練流程" in stdout_text)
    check("optimizer_main_interrupt_reports_skip_overwrite", True, "不自動覆寫" in stdout_text and "2/5" in stdout_text)
    check("optimizer_main_interrupt_stderr_empty", "", stderr_text)

    summary["checks"] = len(results)
    return results, summary



def validate_optimizer_sampler_resume_determinism_case(_base_params):
    case_id = "OPTIMIZER_SAMPLER_RESUME_DETERMINISM"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    import optuna

    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial):
        alpha = trial.suggest_int("alpha", 1, 12)
        beta = trial.suggest_float("beta", 0.0, 1.0)
        mode = trial.suggest_categorical("mode", ["a", "b", "c"])
        gamma = trial.suggest_float("gamma", 1e-3, 1.0, log=True)
        return float(alpha) + float(beta) + (0.0 if mode == "a" else 1.0) + float(gamma)

    def _signature(study):
        return [
            (
                int(trial.number),
                tuple((str(key), value) for key, value in dict(trial.params or {}).items()),
                None if trial.value is None else float(trial.value),
                str(trial.state.name),
            )
            for trial in list(study.trials or [])
        ]

    try:
        with TemporaryDirectory() as temp_dir:
            opened_studies = []

            def _track_study(study):
                opened_studies.append(study)
                return study

            try:
                temp_root = Path(temp_dir)
                # Keep one real SQLite resume path to cover durable reopen/cleanup semantics.
                # The remaining independent sampler scenarios use Optuna's in-memory storage:
                # they exercise the same Study storage contract without paying repeated
                # SQLAlchemy schema/bootstrap cost inside the synthetic suite.
                uninterrupted_storage = optuna.storages.InMemoryStorage()
                resumed_storage = f"sqlite:///{temp_root / 'resumed.db'}"
                legacy_storage = optuna.storages.InMemoryStorage()

                plain_tpe_sampler = optuna.samplers.TPESampler(seed=42)
                tpe_startup_trials = int(getattr(plain_tpe_sampler, "_n_startup_trials", 10) or 10)
                tpe_probe_trials = max(3, tpe_startup_trials + 2)
                tpe_resume_prefix_trials = tpe_probe_trials - 1
                persisted_seed_probe_trials = 3
                random_probe_trials = 4
                restart_recovery_probe_trials = 3

                uninterrupted = _track_study(create_optimizer_study(
                    uninterrupted_storage, seed=42, sampler_kind="tpe"
                ))
                uninterrupted.optimize(_objective, n_trials=tpe_probe_trials)

                plain_sampler_reference = _track_study(optuna.create_study(
                    direction="maximize", sampler=plain_tpe_sampler
                ))
                plain_sampler_reference.optimize(_objective, n_trials=tpe_probe_trials)
                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "resume_checkpointing_does_not_change_fresh_tpe_trial_sequence",
                    _signature(plain_sampler_reference),
                    _signature(uninterrupted),
                )

                resumed = _track_study(create_optimizer_study(resumed_storage, seed=42, sampler_kind="tpe"))
                resumed.optimize(_objective, n_trials=tpe_resume_prefix_trials)
                resumed = _track_study(create_optimizer_study(resumed_storage, seed=42, sampler_kind="tpe"))
                check_before = len(list(resumed.trials or []))
                resumed.optimize(_objective, n_trials=1)

                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "seeded_tpe_resume_preserves_existing_trial_count",
                    tpe_resume_prefix_trials,
                    check_before,
                )
                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "seeded_tpe_uninterrupted_and_resume_trial_sequence_exact_match",
                    _signature(uninterrupted),
                    _signature(resumed),
                )

                system_attrs = dict(
                    resumed._storage.get_study_system_attrs(resumed._study_id) or {}
                )
                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "seeded_tpe_resume_persists_pre_and_post_rng_checkpoints",
                    True,
                    "optimizer_sampler_rng_pre_v1" in system_attrs
                    and "optimizer_sampler_rng_post_v1" in system_attrs,
                )

                persisted_seed_storage = optuna.storages.InMemoryStorage()
                persisted_seed_reference = _track_study(create_optimizer_study(
                    None, seed=123, sampler_kind="tpe"
                ))
                persisted_seed_reference.optimize(_objective, n_trials=persisted_seed_probe_trials)
                persisted_seed_study = _track_study(create_optimizer_study(
                    persisted_seed_storage, seed=123, sampler_kind="tpe"
                ))
                persisted_seed_study.optimize(_objective, n_trials=persisted_seed_probe_trials - 1)
                persisted_seed_study = _track_study(create_optimizer_study(
                    persisted_seed_storage, seed=0, sampler_kind="tpe"
                ))
                persisted_seed_study.optimize(_objective, n_trials=1)
                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "persisted_study_seed_overrides_resume_call_seed",
                    _signature(persisted_seed_reference),
                    _signature(persisted_seed_study),
                )

                legacy = _track_study(optuna.create_study(
                    study_name=OPTIMIZER_LEGACY_STUDY_NAME,
                    storage=legacy_storage,
                    load_if_exists=True,
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                ))
                legacy.optimize(_objective, n_trials=1)
                migrated = _track_study(create_optimizer_study(legacy_storage, seed=42, sampler_kind="tpe"))
                preserved_legacy = _track_study(optuna.load_study(
                    study_name=OPTIMIZER_LEGACY_STUDY_NAME, storage=legacy_storage
                ))
                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "legacy_non_checkpointed_study_is_preserved_but_not_resumed",
                    (True, 0, 1),
                    (
                        str(migrated.study_name).startswith(OPTIMIZER_DETERMINISTIC_RESUME_STUDY_NAME),
                        len(list(migrated.trials or [])),
                        len(list(preserved_legacy.trials or [])),
                    ),
                )

                random_uninterrupted_storage = optuna.storages.InMemoryStorage()
                random_resumed_storage = optuna.storages.InMemoryStorage()
                random_uninterrupted = _track_study(create_optimizer_study(
                    random_uninterrupted_storage, seed=42, sampler_kind="random"
                ))
                random_uninterrupted.optimize(_objective, n_trials=random_probe_trials)
                random_resumed = _track_study(create_optimizer_study(
                    random_resumed_storage, seed=42, sampler_kind="random"
                ))
                random_resumed.optimize(_objective, n_trials=random_probe_trials // 2)
                random_resumed = _track_study(create_optimizer_study(
                    random_resumed_storage, seed=42, sampler_kind="random"
                ))
                random_resumed.optimize(_objective, n_trials=random_probe_trials // 2)
                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "seeded_random_uninterrupted_and_resume_trial_sequence_exact_match",
                    _signature(random_uninterrupted),
                    _signature(random_resumed),
                )

                running_reference_storage = optuna.storages.InMemoryStorage()
                running_recovery_storage = optuna.storages.InMemoryStorage()

                def _suggest_crash_prefix(trial):
                    trial.suggest_int("alpha", 1, 12)
                    trial.suggest_float("beta", 0.0, 1.0)
                    trial.suggest_categorical("mode", ["a", "b", "c"])

                running_reference = _track_study(create_optimizer_study(
                    running_reference_storage, seed=42, sampler_kind="tpe"
                ))
                reference_trial = running_reference.ask()
                _suggest_crash_prefix(reference_trial)
                running_reference.tell(reference_trial, state=optuna.trial.TrialState.FAIL)
                running_reference.optimize(_objective, n_trials=restart_recovery_probe_trials)

                running_recovery = _track_study(create_optimizer_study(
                    running_recovery_storage, seed=42, sampler_kind="tpe"
                ))
                abandoned_trial = running_recovery.ask()
                _suggest_crash_prefix(abandoned_trial)
                running_recovery = _track_study(create_optimizer_study(
                    running_recovery_storage, seed=42, sampler_kind="tpe"
                ))
                running_recovery.optimize(_objective, n_trials=restart_recovery_probe_trials)
                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "seeded_tpe_hard_restart_recovers_abandoned_running_trial_deterministically",
                    _signature(running_reference),
                    _signature(running_recovery),
                )

                missing_post_reference_storage = optuna.storages.InMemoryStorage()
                missing_post_recovery_storage = optuna.storages.InMemoryStorage()
                missing_post_reference = _track_study(create_optimizer_study(
                    missing_post_reference_storage, seed=42, sampler_kind="tpe"
                ))
                missing_post_reference.optimize(_objective, n_trials=tpe_probe_trials)
                missing_post_recovery = _track_study(create_optimizer_study(
                    missing_post_recovery_storage, seed=42, sampler_kind="tpe"
                ))
                missing_post_recovery.optimize(_objective, n_trials=tpe_resume_prefix_trials)
                missing_post_recovery._storage.set_study_system_attr(
                    missing_post_recovery._study_id, "optimizer_sampler_rng_post_v1", {}
                )
                missing_post_recovery = _track_study(create_optimizer_study(
                    missing_post_recovery_storage, seed=42, sampler_kind="tpe"
                ))
                missing_post_recovery.optimize(_objective, n_trials=1)
                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "seeded_tpe_resume_recovers_finished_trial_when_post_checkpoint_is_missing",
                    _signature(missing_post_reference),
                    _signature(missing_post_recovery),
                )

                add_check(
                    results,
                    "strategy_contract",
                    case_id,
                    "tpe_parallel_override_cannot_break_deterministic_trial_sequence",
                    1,
                    resolve_optimizer_single_fold_search_parallel_trials(
                        {
                            "OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS": "8",
                            "OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH": "1",
                        },
                        sampler_kind="tpe",
                    ),
                )
            finally:
                for study in reversed(opened_studies):
                    close_study_storage(study)
    finally:
        optuna.logging.set_verbosity(previous_verbosity)

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
    results, summary, check, check_true = bind_synthetic_case(case_id, 'strategy_contract')

    project_root = Path(__file__).resolve().parents[2]
    default_policy = load_walk_forward_policy(str(project_root), environ={})
    default_policy_path = str(default_policy.get("policy_path", "")).replace("\\", "/")
    check("default_walk_forward_policy_uses_training_policy", True, default_policy_path.endswith("config/training_policy.py"))
    default_oos_start_year = default_policy.get("oos_start_year")
    if default_oos_start_year is not None:
        expected_default_search_train_end_year = int(default_oos_start_year) - 1
    else:
        expected_default_search_train_end_year = int(default_policy.get("train_start_year", 0)) + int(default_policy.get("min_train_years", 0)) - 1
    check("default_walk_forward_policy_auto_derives_search_train_end_year",
        expected_default_search_train_end_year,
        int(default_policy.get("search_train_end_year", 0)),
    )
    expected_full_end_year = None if FULL_END_YEAR is None else int(FULL_END_YEAR)
    expected_full_end_date = None if expected_full_end_year is None else f"{expected_full_end_year:04d}-12-31"
    expected_oos_end_year = None if OOS_EVALUATION_END_YEAR is None else int(OOS_EVALUATION_END_YEAR)
    expected_oos_end_date = None if expected_oos_end_year is None else f"{expected_oos_end_year:04d}-12-31"
    check("default_walk_forward_policy_has_full_end_year", expected_full_end_year, default_policy.get("full_end_year"))
    check("default_walk_forward_policy_has_oos_end_year", expected_oos_end_year, default_policy.get("oos_end_year"))
    check("default_walk_forward_policy_derives_oos_end_date", expected_oos_end_date, default_policy.get("oos_end_date"))

    latest_data_date = "2026-03-02"
    study_full_policy = build_optimizer_runtime_policy(default_policy, "study", latest_data_date=latest_data_date, study_scope="full")
    expected_full_runtime_end_date = min(value for value in (latest_data_date, expected_full_end_date) if value is not None)
    check("study_full_runtime_respects_configured_full_end_year", expected_full_runtime_end_date, study_full_policy.get("search_train_end_date"))
    check("study_full_runtime_has_no_oos_end_date", None, study_full_policy.get("oos_end_date"))

    study_full_latest_clip_policy = build_optimizer_runtime_policy(
        {**default_policy, "full_end_year": 2028},
        "study",
        latest_data_date="2026-03-02",
        study_scope="full",
    )
    check("full_runtime_clips_future_end_year_to_latest_data", "2026-03-02", str(study_full_latest_clip_policy.get("search_train_end_date") or ""))

    oos_policy = build_optimizer_runtime_policy(default_policy, "oos")
    study_oos_policy = build_optimizer_runtime_policy(default_policy, "study", study_scope="oos")
    full_policy = build_optimizer_runtime_policy(default_policy, "full", latest_data_date="2026-03-02")
    trade_policy = build_optimizer_runtime_policy(default_policy, "trade", latest_data_date="2026-03-02")
    check("full_runtime_uses_configured_full_window_and_has_no_oos", True, full_policy.get("evaluation_scope") == "full_seed_ensemble" and full_policy.get("search_train_end_date") == expected_full_runtime_end_date and full_policy.get("oos_end_date") is None)
    check("oos_runtime_respects_configured_end_year", expected_oos_end_date, oos_policy.get("oos_end_date"))
    check("study_oos_runtime_respects_configured_end_year", expected_oos_end_date, study_oos_policy.get("oos_end_date"))
    check("trade_runtime_drops_oos_end_year", None, trade_policy.get("oos_end_year"))

    # Regression: a fixed outer-rolling fold may be built first as a Forward
    # schedule prefix and later resumed while Rolling extends the same physical
    # schedule.  Parent schedule horizon must not make that same fold incompatible.
    from services.optimizer.session_factory import ensure_study_effective_policy_compatible
    from core.walk_forward_policy import build_optimizer_effective_policy_fingerprint

    class _SyntheticStudy:
        def __init__(self, attrs, trials):
            self.user_attrs = dict(attrs)
            self.trials = list(trials)

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    legacy_fold_policy = {
        **oos_policy,
        "selection_start_year": 2011,
        "train_start_year": 2011,
        "search_train_end_year": 2020,
        "oos_start_year": 2021,
        "selection_start_date": "2011-01-01",
        "train_start_date": "2011-01-01",
        "search_train_end_date": "2020-12-31",
        "oos_start_date": "2021-01-01",
        "oos_end_date": "2021-12-31",
        "latest_data_date": "2021-01-31",
        "evaluation_scope": "oos_single_fold",
        "model_mode": "oos",
    }
    current_fold_policy = {**legacy_fold_policy, "latest_data_date": "2020-12-31"}
    legacy_contract = build_optimizer_effective_policy_fingerprint(legacy_fold_policy)
    current_contract = build_optimizer_effective_policy_fingerprint(current_fold_policy)
    fold_resume_study = _SyntheticStudy(
        {
            "optimizer_effective_policy_fingerprint_sha256": legacy_contract["fingerprint_sha256"],
            "optimizer_effective_policy_snapshot": legacy_contract["snapshot"],
        },
        trials=[object()],
    )
    ensure_study_effective_policy_compatible(
        study=fold_resume_study,
        walk_forward_policy=current_fold_policy,
    )
    check(
        "outer_fold_resume_migrates_parent_horizon_only_legacy_identity",
        current_contract["fingerprint_sha256"],
        fold_resume_study.user_attrs.get("optimizer_effective_policy_fingerprint_sha256"),
    )
    cutoff_bug_policy = {**current_fold_policy, "oos_end_date": "2021-01-01"}
    corrected_oos_policy = {**current_fold_policy, "oos_end_date": "2021-12-31"}
    cutoff_bug_contract = build_optimizer_effective_policy_fingerprint(cutoff_bug_policy)
    corrected_oos_contract = build_optimizer_effective_policy_fingerprint(corrected_oos_policy)
    cutoff_resume_study = _SyntheticStudy(
        {
            "optimizer_effective_policy_fingerprint_sha256": cutoff_bug_contract["fingerprint_sha256"],
            "optimizer_effective_policy_snapshot": cutoff_bug_contract["snapshot"],
        },
        trials=[object()],
    )
    ensure_study_effective_policy_compatible(
        study=cutoff_resume_study,
        walk_forward_policy=corrected_oos_policy,
    )
    check(
        "outer_fold_resume_migrates_diagnostic_oos_end_only_bug",
        corrected_oos_contract["fingerprint_sha256"],
        cutoff_resume_study.user_attrs.get("optimizer_effective_policy_fingerprint_sha256"),
    )
    drifted_fold_policy = {**current_fold_policy, "search_train_end_date": "2020-11-30"}
    rejected_real_drift = False
    try:
        ensure_study_effective_policy_compatible(
            study=_SyntheticStudy(
                {
                    "optimizer_effective_policy_fingerprint_sha256": legacy_contract["fingerprint_sha256"],
                    "optimizer_effective_policy_snapshot": legacy_contract["snapshot"],
                },
                trials=[object()],
            ),
            walk_forward_policy=drifted_fold_policy,
        )
    except RuntimeError:
        rejected_real_drift = True
    check(
        "outer_fold_resume_still_rejects_real_scientific_drift",
        True,
        rejected_real_drift,
    )

    from tools.optimizer.outer_rolling_oos import _resolve_config as resolve_outer_rolling_config

    rolling_config = resolve_outer_rolling_config(
        ["optimizer"],
        {},
        base_policy=default_policy,
        latest_year=2026,
        latest_date="2026-03-02",
        default_trials=1,
        timing_mode=True,
    )
    latest_oos_month = pd.Timestamp(latest_data_date).normalize().replace(day=1)
    if expected_oos_end_date is not None:
        latest_oos_month = min(latest_oos_month, pd.Timestamp(expected_oos_end_date).normalize().replace(day=1))
    check("rolling_oos_default_last_date_respects_configured_end_year", latest_oos_month.strftime("%Y-%m-%d"), str(rolling_config.last_oos_date))
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
    check("python_override_policy_auto_derives_search_train_end_year", 2014, int(override_policy.get("search_train_end_year", 0)))

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
    check("json_override_policy_still_supported", 2010, int(json_override_policy.get("search_train_end_year", 0)))
    check("external_override_without_oos_scope_drops_default_oos_end_date", None, json_override_policy.get("oos_end_date"))

    env_oos_end_policy = load_walk_forward_policy(str(project_root), environ={"V16_WF_OOS_END_YEAR": "2024"})
    check("inline_oos_end_year_override_is_supported", "2024-12-31", str(env_oos_end_policy.get("oos_end_date") or ""))
    env_oos_end_date_policy = load_walk_forward_policy(str(project_root), environ={"V16_WF_OOS_END_DATE": "2024-06-30"})
    check("inline_oos_end_date_override_takes_precedence_over_default_year", "2024-06-30", str(env_oos_end_date_policy.get("oos_end_date") or ""))

    try:
        load_walk_forward_policy(str(project_root), environ={"V16_WF_OOS_END_YEAR": "2024", "V16_WF_OOS_END_DATE": "2025-06-30"})
        conflicting_oos_end_override_rejected = False
    except ValueError as exc:
        conflicting_oos_end_override_rejected = "oos_end_year" in str(exc)
    check("conflicting_explicit_oos_end_year_and_date_rejected", True, conflicting_oos_end_override_rejected)

    try:
        build_optimizer_runtime_policy({**default_policy, "full_start_year": 2023, "full_end_year": 2022}, "study", latest_data_date="2026-03-02", study_scope="full")
        invalid_study_full_range_rejected = False
    except ValueError as exc:
        invalid_study_full_range_rejected = "full_end_year" in str(exc)
    check("full_reverse_year_range_rejected", True, invalid_study_full_range_rejected)


    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "wf_invalid.py"
        tmp_path.write_text('WALK_FORWARD_POLICY = {"objective_mode": "bad_mode"}\n', encoding="utf-8")
        try:
            load_walk_forward_policy(str(project_root), environ={"V16_WALK_FORWARD_POLICY_PATH": str(tmp_path)})
            invalid_legacy_symbol_rejected = False
        except ValueError as exc:
            invalid_legacy_symbol_rejected = "TRAINING_SPLIT_POLICY" in str(exc)
    check("legacy_walk_forward_policy_symbol_rejected", True, invalid_legacy_symbol_rejected)

    callbacks_source = Path(optimizer_callbacks.__file__).read_text(encoding="utf-8")
    static_ensemble_dashboard_source = (project_root / "services" / "optimizer" / "static_ensemble_dashboard.py").read_text(encoding="utf-8")
    optimizer_main_source = (project_root / "services" / "optimizer" / "application.py").read_text(encoding="utf-8")
    optimizer_robustness_source = (project_root / "services" / "optimizer" / "robustness.py").read_text(encoding="utf-8")
    optimizer_outer_rolling_source = (project_root / "services" / "optimizer" / "outer_rolling_oos.py").read_text(encoding="utf-8")
    optimizer_live_boards_source = (project_root / "services" / "optimizer" / "outer_rolling_live_boards.py").read_text(encoding="utf-8")
    optimizer_walk_forward_source = (project_root / "services" / "optimizer" / "walk_forward.py").read_text(encoding="utf-8")
    objective_runner_source = (project_root / "services" / "optimizer" / "objective_runner.py").read_text(encoding="utf-8")
    check("optimizer_callbacks_imports_pandas_for_oos_year_parsing", True, "import pandas as pd" in callbacks_source)
    check("search_train_date_filter_reuses_core_single_source_in_callbacks", True, "from core.walk_forward_policy import filter_search_train_dates" in callbacks_source and "def _filter_search_train_dates" not in callbacks_source)
    check("search_train_date_filter_reuses_core_single_source_in_objective_runner", True, "from core.walk_forward_policy import filter_search_train_dates" in objective_runner_source and "def _filter_search_train_dates" not in objective_runner_source)

    params = V16StrategyParams()
    params.use_kc = True
    params.kc_len = 34
    params.kc_mult = 1.8
    training_lines = optimizer_callbacks._build_training_param_lines(params)
    rendered_training_text = "\n".join(training_lines)
    check("optimizer_callbacks_kc_label_matches_dashboard_wording", True, "阿肯那(KC)" in rendered_training_text and "阿唐那(KC)" not in rendered_training_text)
    params.use_breakout_reclaim_reentry = True
    counted_training_text = "\n".join(optimizer_callbacks._build_training_param_lines(params, entry_trade_counts={"breakout_trades": 12, "extended_trades": 7, "reentry_trades": 3}))
    check("optimizer_training_params_hide_entry_trade_counts",
        True,
        "突破買進 啟用 (突破 201 日新高)" in counted_training_text
        and "Re-entry 啟用（20日內站回 STOP+0.8ATR）" in counted_training_text
        and "交易次數" not in counted_training_text,
    )

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
            "extended_trades": 2,
            "reentry_trades": 1,
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
            "extended_trades": 2,
            "reentry_trades": 1,
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
    check("optimizer_first_zone_keeps_final_equity_row", True, "最終資產" in row_names)
    check("optimizer_first_zone_combines_payoff_and_ev_label_without_extra_spaces", True, "風報比: 期望值" in row_names and "風報比 : 期望值" not in row_names)
    romd_row = row_map.get("報酬回撤比 (RoMD)", {})
    check("optimizer_first_zone_romd_row_uses_plain_return_mdd_not_system_score", True, "1.41" in str(romd_row.get("candidate", "")) and "0.67" in str(romd_row.get("benchmark", "")))
    check("optimizer_first_zone_formats_payoff_ev_pair_with_consistent_spacing", True, row_map.get("風報比: 期望值", {}).get("candidate") == "1.50: 0.250R" and "1.40: 0.200R" in str(row_map.get("風報比: 期望值", {}).get("reference", "")) and "(+0.10: +0.050R)" in str(row_map.get("風報比: 期望值", {}).get("reference", "")))
    check("optimizer_first_zone_formats_trade_split_with_consistent_spacing", True, row_map.get("總交易次數", {}).get("candidate") == "11 (正常: 8｜延續: 2｜重進: 1)" and row_map.get("總交易次數", {}).get("reference") == "10 (正常: 7｜延續: 2｜重進: 1)")
    check("optimizer_first_zone_formats_missed_split_with_consistent_spacing", True, row_map.get("錯失交易次數", {}).get("candidate") == "2 (買: 1｜賣: 1)" and row_map.get("錯失交易次數", {}).get("reference") == "3 (買: 2｜賣: 1)")

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
            "candidate": "649 (正常: 346｜延續: 300｜重進: 3)",
            "candidate_precolored": False,
            "reference": "510 (正常: 425｜延續: 84｜重進: 1)",
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
    dashboard_text_plain = "\n".join(dashboard_lines)
    check("optimizer_console_training_table_uses_benchmark_and_delta_columns", True, "| 指標項目" in dashboard_text_plain and "| 本輪候選" in dashboard_text_plain and "| 同期大盤0050" in dashboard_text_plain and "| 差異" in dashboard_text_plain and "run_best (差異)" not in dashboard_text_plain)
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
    check("optimizer_console_table_keeps_pipe_alignment_for_long_first_zone_rows", True, len(pipe_positions) == 6 and len(set(pipe_positions)) == 1)

    optimizer_study_utils_source = (project_root / "services" / "optimizer" / "study_utils.py").read_text(encoding="utf-8")
    train_test_policy_lines = [line for line in optimizer_main_source.splitlines() if "Train/Test policy:" in line]
    check("optimizer_start_banner_omits_raw_objective_mode_token", True, bool(train_test_policy_lines) and all("objective=" not in line for line in train_test_policy_lines))
    check("system_score_display_formatter_applies_multiplier", f"{1.23 * SYSTEM_SCORE_DISPLAY_MULTIPLIER:.2f}", format_system_score_for_display(1.23, decimals=2))
    check("optimizer_system_score_display_helper_applies_multiplier", f"{1.23 * SYSTEM_SCORE_DISPLAY_MULTIPLIER:.3f}", format_optimizer_score_for_display(1.23, decimals=3))
    check("optimizer_system_score_display_helper_preserves_invalid_sentinel", f"{INVALID_TRIAL_VALUE:.3f}", format_optimizer_score_for_display(INVALID_TRIAL_VALUE, decimals=3))
    check("optimizer_dashboard_system_score_display_uses_multiplier_formatter",
        True,
        "format_system_score_for_display(attrs.get('base_score'" in callbacks_source
        and "format_system_score_for_display(candidate_train_metrics.get('pf_romd'" in static_ensemble_dashboard_source
        and "format_system_score_for_display(trial.value" in callbacks_source
        and "format_optimizer_score_for_display" in optimizer_main_source
        and "format_optimizer_score_for_display" in optimizer_robustness_source
        and "_format_system_score_compare_plain" in optimizer_outer_rolling_source
        and _format_system_score_compare_plain(1.0, 2.0)
            == f"{1.0 * SYSTEM_SCORE_DISPLAY_MULTIPLIER:.2f} ({1.0 * SYSTEM_SCORE_DISPLAY_MULTIPLIER:+.2f})"
        and "OOS 系統得分" in optimizer_walk_forward_source
        and "OOS RoMD" in optimizer_walk_forward_source,
    )
    check("study_memory_prompt_defaults_to_resume_with_restart_on_1", True, "👉 Study 記憶庫：[Enter] 接續訓練  [1] 重頭開始 : " in optimizer_study_utils_source and "👉 Study 記憶庫：[Enter] 接續訓練  [1] 重頭開始 : " in optimizer_main_source)
    check("interactive_optimizer_menu_defaults_to_study_and_excludes_trade", True, "[Enter] Study Mode [1] Full Mode [2] OOS Mode [3] Rolling OOS Mode" in optimizer_study_utils_source and "[4] Trade Mode" not in optimizer_study_utils_source)
    try:
        study_utils._parse_optimizer_mode_request_raw("4", source_label="UI/MENU")
        trade_menu_rejected = False
    except ValueError:
        trade_menu_rejected = True
    check("interactive_optimizer_menu_rejects_trade_workbench_owned_mode", True, trade_menu_rejected)
    trade_compat_request = study_utils._parse_optimizer_run_request_raw("TRADE", source_label="ENV:compat")
    check("optimizer_trade_remains_noninteractive_compatibility_only", True, trade_compat_request.get("model_mode") == "trade" and trade_compat_request.get("action") == study_utils.OPTIMIZER_MENU_ACTION_TRAIN)
    with patch("builtins.input", side_effect=["1", "0"]):
        interactive_full_zero_request = study_utils._resolve_interactive_optimizer_run_request()
    check("interactive_one_routes_to_full_mode", True, interactive_full_zero_request.get("model_mode") == "full" and interactive_full_zero_request.get("n_trials") == 0 and interactive_full_zero_request.get("action") == study_utils.OPTIMIZER_MENU_ACTION_EXPORT_CANDIDATE)
    check("interactive_study_scope_menu_uses_one_for_oos", True, "[Enter] Study-Full [1] Study-OOS" in optimizer_study_utils_source and "[2] Study-OOS" not in optimizer_study_utils_source)
    check("study_oos_dashboard_shows_train_period_single_stock_breakout_stats", True, "def _study_single_stock_breakout_stats_title" in callbacks_source and "Study-OOS 單股突破統計｜Train Period" in callbacks_source and "study_full_breakout_stats_title" in callbacks_source)
    check("interactive_trial_prompt_exposes_zero_export", True, "[0] 輸出參數" in optimizer_study_utils_source and "min_value=0" in optimizer_study_utils_source)
    with patch("builtins.input", side_effect=["S", "", "0"]), patch("tools.optimizer.study_utils.safe_prompt_choice", return_value=""):
        interactive_zero_request = study_utils._resolve_interactive_optimizer_run_request()
    check("interactive_zero_trials_routes_to_export_candidate", True, interactive_zero_request.get("n_trials") == 0 and interactive_zero_request.get("action") == study_utils.OPTIMIZER_MENU_ACTION_EXPORT_CANDIDATE and interactive_zero_request.get("model_mode") == "study" and interactive_zero_request.get("study_db_action") == "resume")
    with patch("builtins.input", side_effect=["", "", "0"]), patch("tools.optimizer.study_utils.safe_prompt_choice", return_value=""):
        interactive_default_zero_request = study_utils._resolve_interactive_optimizer_run_request()
    check("interactive_enter_mode_defaults_to_study_base_export", True, interactive_default_zero_request.get("model_mode") == "study" and interactive_default_zero_request.get("n_trials") == 0 and interactive_default_zero_request.get("action") == study_utils.OPTIMIZER_MENU_ACTION_EXPORT_CANDIDATE)

    strategy_param_config_source = (project_root / "config" / "strategy_params.py").read_text(encoding="utf-8")
    strategy_param_app_source = (project_root / "services" / "research" / "strategy_param_application.py").read_text(encoding="utf-8")
    research_entry_source = (project_root / "apps" / "research.py").read_text(encoding="utf-8")
    from core.strategy_param_policy import (
        get_active_strategy_param_training_profile,
        get_strategy_param_fitting_seed_protocol,
        get_strategy_param_training_profile,
    )
    from services.optimizer.validation_budget_refit import resolve_validation_budget_spec
    active_param_profile = get_active_strategy_param_training_profile()
    validation_param_profile = get_strategy_param_training_profile("validation_budget_current")
    plateau_validation_param_profile = get_strategy_param_training_profile(
        "validation_plateau_current"
    )
    plateau_best_validation_param_profile = get_strategy_param_training_profile(
        "validation_plateau_best_current"
    )
    plateau_best_patience100_param_profile = get_strategy_param_training_profile(
        "validation_plateau_best_patience100_current"
    )
    plateau_best_quarter_off_param_profile = get_strategy_param_training_profile(
        "validation_plateau_best_quarter_off_current"
    )
    production_multiseed_param_profile = get_strategy_param_training_profile(
        "production_multiseed_current"
    )
    trading_aligned_8x1000_param_profile = get_strategy_param_training_profile(
        "trading_aligned_8x1000_current"
    )
    production_multiseed_fitting_seed_protocol = get_strategy_param_fitting_seed_protocol(
        production_multiseed_param_profile.fitting_seed_protocol
    )
    trading_aligned_8x1000_fitting_seed_protocol = get_strategy_param_fitting_seed_protocol(
        trading_aligned_8x1000_param_profile.fitting_seed_protocol
    )
    validation_fitting_seed_protocol = get_strategy_param_fitting_seed_protocol(
        validation_param_profile.fitting_seed_protocol
    )
    validation_spec = resolve_validation_budget_spec(300)
    plateau_validation_spec = resolve_validation_budget_spec(
        300, protocol_id="validation_plateau_refit_v3"
    )
    plateau_best_validation_spec = resolve_validation_budget_spec(
        300, protocol_id="validation_plateau_best_refit_v1"
    )
    plateau_best_patience100_spec = resolve_validation_budget_spec(
        300, protocol_id="validation_plateau_best_patience100_refit_v1"
    )
    check("strategy_param_training_target_is_config_driven", True, "ACTIVE_STRATEGY_PARAM_PROFILE" in strategy_param_config_source and "get_active_strategy_param_training_profile" in strategy_param_app_source)
    check("strategy_param_menu_contains_only_stable_work_types", True, all(label in strategy_param_app_source for label in ("Forward OOS 參數訓練", "Rolling OOS 參數訓練", "Forward OOS Robustness 參數訓練", "Rolling OOS Robustness 參數訓練", "進階／單次 Optimizer")) and "PARAM-P5" not in research_entry_source)
    check("strategy_param_validation_is_profile_config_not_optimizer_menu_switch", True, validation_param_profile.validation_enabled and validation_param_profile.validation_protocol == "validation_budget_refit_v1" and "STRATEGY_PARAM_VALIDATION_PROTOCOLS" in strategy_param_config_source)
    check("strategy_param_validation_config_preserves_phase2_contract", True, validation_spec.as_dict() == {"protocol": "validation_budget_refit_v1", "holdout_months": 12, "checkpoints": [50, 100, 150, 200, 250, 300], "policy_name": "base_finalists_agree", "selection_metric": "canonical_optimizer_portfolio_score", "tie_break": "smaller_budget", "oos_feedback_used": False, "full_refit_required": True})
    check(
        "strategy_param_round6_plateau_validation_is_new_versioned_contract",
        True,
        plateau_validation_param_profile.parameter_set_identity == "PARAM-P9"
        and plateau_validation_param_profile.validation_protocol == "validation_plateau_refit_v3"
        and plateau_validation_param_profile.fitting_seed_protocol == "single_seed_42"
        and plateau_validation_spec.as_dict() == {
            "protocol": "validation_plateau_refit_v3",
            "holdout_months": 12,
            "patience_trials": 50,
            "observation_interval": 1,
            "min_delta": 0.0,
            "reuse_same_policy_signature": True,
            "observation_start_trial": 1,
            "min_trials_before_stop": 50,
            "policy_name": "base_finalists_agree",
            "selection_metric": "canonical_optimizer_portfolio_score",
            "tie_break": "smaller_budget",
            "oos_feedback_used": False,
            "full_refit_required": True,
        },
    )
    check(
        "strategy_param_round7_plateau_best_validation_is_new_identity",
        True,
        plateau_best_validation_param_profile.parameter_set_identity == "PARAM-P10"
        and plateau_best_validation_param_profile.parameter_policy == "base-finalist-best"
        and plateau_best_validation_param_profile.validation_protocol == "validation_plateau_best_refit_v1"
        and plateau_best_validation_param_profile.fitting_seed_protocol == "single_seed_42"
        and plateau_best_validation_spec.as_dict() == {
            "protocol": "validation_plateau_best_refit_v1",
            "holdout_months": 12,
            "patience_trials": 50,
            "observation_interval": 1,
            "min_delta": 0.0,
            "reuse_same_policy_signature": True,
            "observation_start_trial": 1,
            "min_trials_before_stop": 50,
            "policy_name": "base_finalist_best",
            "selection_metric": "canonical_optimizer_portfolio_score",
            "tie_break": "smaller_budget",
            "oos_feedback_used": False,
            "full_refit_required": True,
        },
    )
    check(
        "strategy_param_round8_patience100_is_single_variable_identity",
        True,
        plateau_best_patience100_param_profile.parameter_set_identity == "PARAM-P11"
        and plateau_best_patience100_param_profile.parameter_policy == "base-finalist-best"
        and plateau_best_patience100_param_profile.validation_protocol
            == "validation_plateau_best_patience100_refit_v1"
        and plateau_best_patience100_param_profile.fitting_seed_protocol == "single_seed_42"
        and plateau_best_patience100_spec.as_dict() == {
            "protocol": "validation_plateau_best_patience100_refit_v1",
            "holdout_months": 12,
            "patience_trials": 100,
            "observation_interval": 1,
            "min_delta": 0.0,
            "reuse_same_policy_signature": True,
            "observation_start_trial": 1,
            "min_trials_before_stop": 50,
            "policy_name": "base_finalist_best",
            "selection_metric": "canonical_optimizer_portfolio_score",
            "tie_break": "smaller_budget",
            "oos_feedback_used": False,
            "full_refit_required": True,
        }
        and plateau_best_patience100_spec.as_dict()["patience_trials"]
            != plateau_best_validation_spec.as_dict()["patience_trials"],
    )
    from core.strategy_param_scientific_contract import build_strategy_param_scientific_contract
    p10_scientific_contract = build_strategy_param_scientific_contract(
        project_root, plateau_best_validation_param_profile.profile_id
    )["scientific_contract"]
    p12_scientific_contract = build_strategy_param_scientific_contract(
        project_root, plateau_best_quarter_off_param_profile.profile_id
    )["scientific_contract"]
    p10_score = dict(p10_scientific_contract["score"])
    p12_score = dict(p12_scientific_contract["score"])
    p10_non_score = {key: value for key, value in p10_scientific_contract.items() if key != "score"}
    p12_non_score = {key: value for key, value in p12_scientific_contract.items() if key != "score"}
    score_diffs = {
        key: (p10_score.get(key), p12_score.get(key))
        for key in set(p10_score) | set(p12_score)
        if p10_score.get(key) != p12_score.get(key)
    }
    check(
        "strategy_param_round9_quarter_off_is_single_score_variable_identity",
        True,
        plateau_best_quarter_off_param_profile.parameter_set_identity == "PARAM-P12"
        and plateau_best_quarter_off_param_profile.parameter_policy == "base-finalist-best"
        and plateau_best_quarter_off_param_profile.validation_protocol
            == plateau_best_validation_param_profile.validation_protocol
        and plateau_best_quarter_off_param_profile.fitting_seed_protocol == "single_seed_42"
        and dict(plateau_best_quarter_off_param_profile.score_policy_overrides)
            == {"SCORE_MIN_QUARTER_RETURN_AMP_ENABLED": False}
        and p10_non_score == p12_non_score
        and score_diffs
            == {"SCORE_MIN_QUARTER_RETURN_AMP_ENABLED": (True, False)},
    )

    from config.trading import (
        TRADING_OPTIMIZER_SEED_COUNT,
        TRADING_OPTIMIZER_SEED_MIN_AGREE,
        TRADING_OPTIMIZER_TRAIN_WINDOW_MONTHS,
        TRADING_OPTIMIZER_TRIALS_PER_SEED,
        TRADING_PARAM_FAMILY,
        TRADING_PARAM_SELECTOR,
    )
    from core.seed_ensemble_policy import resolve_seed_ensemble_min_agree
    p13_scientific = build_strategy_param_scientific_contract(
        project_root, trading_aligned_8x1000_param_profile.profile_id
    )
    p13_fitting = dict(p13_scientific["scientific_contract"]["fitting"])
    p13_schedule = dict(p13_scientific["scientific_contract"]["schedule"])
    check(
        "strategy_param_round10_trading_aligned_budget_matches_trading_contract",
        True,
        trading_aligned_8x1000_param_profile.parameter_set_identity == "PARAM-P13"
        and trading_aligned_8x1000_param_profile.filter_profile == str(TRADING_PARAM_FAMILY)
        and trading_aligned_8x1000_param_profile.parameter_policy
            == str(TRADING_PARAM_SELECTOR).replace("_", "-")
        and not trading_aligned_8x1000_param_profile.validation_enabled
        and trading_aligned_8x1000_fitting_seed_protocol.seed_mode == "fixed_generated_bundle"
        and p13_fitting["seed_count"] == int(TRADING_OPTIMIZER_SEED_COUNT)
        and p13_fitting["trials_per_seed"] == int(TRADING_OPTIMIZER_TRIALS_PER_SEED)
        and p13_fitting["min_agree"]
            == resolve_seed_ensemble_min_agree(
                TRADING_OPTIMIZER_SEED_COUNT, TRADING_OPTIMIZER_SEED_MIN_AGREE
            )
        and p13_schedule["train_window_months"]
            == int(TRADING_OPTIMIZER_TRAIN_WINDOW_MONTHS),
    )

    check(
        "strategy_param_p10_requires_resume_safe_execution_evidence_without_scientific_change",
        True,
        plateau_best_validation_param_profile.requires_resume_safe_execution_evidence
        and not plateau_best_patience100_param_profile.requires_resume_safe_execution_evidence,
    )

    from config.execution_policy import (
        DEFAULT_FIXED_RISK as _DEFAULT_FIXED_RISK,
        DEFAULT_MAX_POSITION_CAP_PCT as _DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS as _DEFAULT_PORTFOLIO_MAX_POSITIONS,
        DEFAULT_PORTFOLIO_ROTATION as _DEFAULT_PORTFOLIO_ROTATION,
    )
    from core.strategy_param_artifacts import compute_strategy_param_file_sha256 as _param_file_sha
    from services.optimizer.strategy_param_service import (
        _validation_budget_build_contract as _validation_build_contract,
        ensure_validation_budget_strategy_parameter_artifact as _ensure_validation_artifact,
        inspect_validation_budget_strategy_parameter_artifact as _inspect_validation_artifact,
        validation_budget_strategy_param_artifact_path as _validation_artifact_path,
        validation_budget_strategy_param_manifest_path as _validation_manifest_path,
    )

    with TemporaryDirectory() as _resume_safe_temp:
        _resume_safe_root = Path(_resume_safe_temp)
        _p10_policy = "base-finalist-best"
        _p10_set = "PARAM-P10"
        _p10_protocol = "validation_plateau_best_refit_v1"
        _p10_target = _validation_artifact_path(
            _resume_safe_root, policy=_p10_policy,
            parameter_set_identity=_p10_set, validation_protocol_id=_p10_protocol,
        )
        _p10_manifest = _validation_manifest_path(
            _resume_safe_root, parameter_set_identity=_p10_set,
            validation_protocol_id=_p10_protocol,
        )
        _p10_target.parent.mkdir(parents=True, exist_ok=True)
        _p10_payload = {
            "schema_type": "optimizer_active_param_ensemble",
            "schema_version": 1,
            "mode": "rolling",
            "type": "active_param_ensemble",
            "selector": "base_finalist_best",
            "params_ensemble_by_effective_date": {
                "2021-01-01": [{"seed": 42, "params": {"high_len": 200}}]
            },
            "folds": [{
                "oos_start_date": "2021-01-01",
                "oos_end_date": "2021-12-31",
            }],
            "breakout_quality_param_adaptation": {
                "parameter_set": _p10_set,
                "training_dl_enabled": False,
                "optimizer_search_protocol": _p10_protocol,
            },
        }
        _p10_target.write_text(json.dumps(_p10_payload), encoding="utf-8")
        _legacy_p10_sha = _param_file_sha(_p10_target)
        _p10_contract = _validation_build_contract(
            family="full", dataset="full",
            max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(_DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(_DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=_p10_set, validation_protocol_id=_p10_protocol,
        )
        _p10_manifest.write_text(json.dumps({
            "schema_type": "strategy_parameter_research_identity",
            "schema_version": 1,
            "producer": "optimizer",
            "training_dl_enabled": False,
            "parameter_set": _p10_set,
            "family": "full",
            "policy": _p10_policy,
            "optimizer_search_protocol": _p10_protocol,
            "build_contract": _p10_contract,
            "artifact_path": _p10_target.relative_to(_resume_safe_root).as_posix(),
            "artifact_sha256": _legacy_p10_sha,
            "coverage_end": "2021-12-31",
            "production_promotion": False,
        }), encoding="utf-8")
        _legacy_p10_state = _inspect_validation_artifact(
            _resume_safe_root, policy=_p10_policy, comparison_end_date="2021-12-31",
            dataset="full", max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(_DEFAULT_PORTFOLIO_ROTATION), fixed_risk=float(_DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=_p10_set, validation_protocol_id=_p10_protocol,
        )
        _failed_repair_call = {}

        def _fake_invalid_p10_repair(**kwargs):
            _failed_repair_call.update(kwargs)
            _repair_root = Path(kwargs["output_relative_dir"])
            _repair_source = _repair_root / "active_params" / "roos_base_best.json"
            _repair_source.parent.mkdir(parents=True, exist_ok=True)
            _invalid_payload = json.loads(json.dumps(_p10_payload))
            _invalid_payload["folds"][0]["oos_end_date"] = "2021-01-01"
            _repair_source.write_text(json.dumps(_invalid_payload), encoding="utf-8")
            return {"params_path": _repair_source}

        _failed_repair_raised = False
        with patch(
            "services.optimizer.strategy_param_training.prepare_selection_historical_full_roos_params",
            side_effect=_fake_invalid_p10_repair,
        ):
            try:
                _ensure_validation_artifact(
                    _resume_safe_root, policy=_p10_policy, comparison_end_date="2021-12-31",
                    dataset="full", max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
                    rotation=str(_DEFAULT_PORTFOLIO_ROTATION),
                    fixed_risk=float(_DEFAULT_FIXED_RISK),
                    max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
                    parameter_set_identity=_p10_set, validation_protocol_id=_p10_protocol,
                )
            except RuntimeError as exc:
                _failed_repair_raised = "schedule coverage不足" in str(exc)
        _canonical_after_failed_repair = json.loads(_p10_target.read_text(encoding="utf-8"))
        _manifest_after_failed_repair = json.loads(_p10_manifest.read_text(encoding="utf-8"))
        check(
            "p10_failed_execution_evidence_repair_does_not_publish_invalid_canonical",
            True,
            _failed_repair_raised
            and _canonical_after_failed_repair["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]["high_len"] == 200
            and _manifest_after_failed_repair.get("artifact_sha256") == _legacy_p10_sha
            and "optimizer_execution_evidence_contract" not in _manifest_after_failed_repair,
        )

        _repair_call = {}

        def _fake_p10_repair(**kwargs):
            _repair_call.update(kwargs)
            _repair_root = Path(kwargs["output_relative_dir"])
            _repair_source = _repair_root / "active_params" / "roos_base_best.json"
            _repair_source.parent.mkdir(parents=True, exist_ok=True)
            _rebuilt_payload = json.loads(json.dumps(_p10_payload))
            _rebuilt_payload["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]["high_len"] = 260
            _repair_source.write_text(json.dumps(_rebuilt_payload), encoding="utf-8")
            return {"params_path": _repair_source}

        with patch(
            "services.optimizer.strategy_param_training.prepare_selection_historical_full_roos_params",
            side_effect=_fake_p10_repair,
        ):
            _rebuilt_p10 = _ensure_validation_artifact(
                _resume_safe_root, policy=_p10_policy, comparison_end_date="2021-12-31",
                dataset="full", max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
                rotation=str(_DEFAULT_PORTFOLIO_ROTATION),
                fixed_risk=float(_DEFAULT_FIXED_RISK),
                max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
                parameter_set_identity=_p10_set, validation_protocol_id=_p10_protocol,
            )
        _rebuilt_p10_state = _inspect_validation_artifact(
            _resume_safe_root, policy=_p10_policy, comparison_end_date="2021-12-31",
            dataset="full", max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(_DEFAULT_PORTFOLIO_ROTATION), fixed_risk=float(_DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=_p10_set, validation_protocol_id=_p10_protocol,
        )
        _repair_relative = Path(_repair_call.get("output_relative_dir") or ".").resolve().relative_to(
            _resume_safe_root.resolve()
        ).as_posix()
        _archive_root = (
            _resume_safe_root
            / "outputs/optimizer/strategy_param_schedule/experiments/param_p10_validation_plateau_best_refit"
            / "execution_evidence_archive"
            / _legacy_p10_sha
        )
        _archived_payload = json.loads((_archive_root / _p10_target.name).read_text(encoding="utf-8"))
        _archived_manifest = json.loads((_archive_root / "identity_manifest.json").read_text(encoding="utf-8"))
        check(
            "p10_legacy_resume_evidence_is_archived_before_canonical_republish",
            True,
            _archive_root.is_dir()
            and (_archive_root / "archive_metadata.json").is_file()
            and _archived_payload["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]["high_len"] == 200
            and _archived_manifest.get("artifact_sha256") == _legacy_p10_sha
            and "optimizer_execution_evidence_contract" not in _archived_manifest
            and json.loads(_p10_target.read_text(encoding="utf-8"))["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]["high_len"] == 260,
        )
        check(
            "p10_legacy_resume_evidence_auto_rebuilds_in_stable_resume_safe_workspace",
            True,
            _legacy_p10_state["status"] == "OPTIMIZER_EXECUTION_EVIDENCE_STALE"
            and _legacy_p10_state["action"] == "REBUILD"
            and _rebuilt_p10.get("execution_evidence_rebuild") is True
            and _repair_call.get("resume_parameter_training") is True
            and _repair_relative.endswith(
                "param_p10_validation_plateau_best_refit/execution_evidence_rebuilds/optimizer_resume_determinism_v1"
            )
            and _rebuilt_p10_state["status"] == "READY"
            and _rebuilt_p10_state["action"] == "REUSE"
            and _rebuilt_p10_state["actual_optimizer_execution_evidence_contract"]
                == "optimizer_resume_determinism_v1",
        )

    # Regression: P12's original published artifact can contain validation scores
    # replayed under the global Quarter-ON default even though its scientific score
    # override is Quarter-OFF. The repaired replay implementation must make that
    # published artifact stale without changing the inner Optuna training-study cache.
    from services.optimizer.strategy_param_training import (
        _build_selection_full_roos_schedule_contract as _build_validation_schedule,
        _selection_full_roos_optimizer_runtime_cache_identity as _fold_cache_identity,
        _selection_full_roos_runtime_contract as _validation_runtime_contract,
    )
    from services.optimizer.validation_budget_refit import (
        VALIDATION_REPLAY_IMPLEMENTATION_CONTRACT as _validation_replay_contract,
    )

    _p12_runtime_args = SimpleNamespace(
        dataset="full", trials_per_fold=300, fixed_risk=float(_DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
        max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(_DEFAULT_PORTFOLIO_ROTATION),
        score_policy_overrides={"SCORE_MIN_QUARTER_RETURN_AMP_ENABLED": False},
        seed_ensemble_policy_override={},
    )
    _p12_schedule = _build_validation_schedule(
        first_oos_date="2021-01-01", last_oos_date="2021-01-01",
        train_window_months=120, oos_months=12,
    )
    _p12_runtime_contract = _validation_runtime_contract(
        root=project_root, args=_p12_runtime_args, schedule_contract=_p12_schedule,
        dataset_identity_override={"synthetic": True, "sha256": "b" * 64},
    )
    from copy import deepcopy as _p12_deepcopy
    _p12_legacy_runtime_contract = _p12_deepcopy(_p12_runtime_contract)
    _p12_legacy_runtime_contract.pop("validation_replay_implementation_contract", None)
    check(
        "p12_validation_replay_fix_refreshes_work_identity_but_preserves_training_study_identity",
        True,
        _p12_runtime_contract.get("validation_replay_implementation_contract")
            == _validation_replay_contract
        and _fold_cache_identity(
            _p12_runtime_contract, optimizer_search_protocol="validation_plateau_best_refit_v1"
        ) == _fold_cache_identity(
            _p12_legacy_runtime_contract, optimizer_search_protocol="validation_plateau_best_refit_v1"
        ),
    )

    with TemporaryDirectory() as _p12_replay_temp:
        _p12_root = Path(_p12_replay_temp)
        _p12_policy = "base-finalist-best"
        _p12_set = "PARAM-P12"
        _p12_protocol = "validation_plateau_best_refit_v1"
        _p12_target = _validation_artifact_path(
            _p12_root, policy=_p12_policy, parameter_set_identity=_p12_set,
            validation_protocol_id=_p12_protocol,
        )
        _p12_manifest = _validation_manifest_path(
            _p12_root, parameter_set_identity=_p12_set, validation_protocol_id=_p12_protocol,
        )
        _p12_target.parent.mkdir(parents=True, exist_ok=True)
        _p12_payload = {
            "schema_type": "optimizer_active_param_ensemble",
            "schema_version": 1,
            "mode": "rolling",
            "type": "active_param_ensemble",
            "selector": "base_finalist_best",
            "params_ensemble_by_effective_date": {
                "2021-01-01": [{"seed": 42, "params": {"high_len": 200}}]
            },
            "folds": [{
                "oos_start_date": "2021-01-01",
                "oos_end_date": "2021-12-31",
            }],
            "breakout_quality_param_adaptation": {
                "parameter_set": _p12_set,
                "training_dl_enabled": False,
                "optimizer_search_protocol": _p12_protocol,
            },
        }
        _p12_target.write_text(json.dumps(_p12_payload), encoding="utf-8")
        _p12_contract = _validation_build_contract(
            family="full", dataset="full",
            max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(_DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(_DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=_p12_set, validation_protocol_id=_p12_protocol,
        )
        _p12_manifest_payload = {
            "schema_type": "strategy_parameter_research_identity",
            "schema_version": 1,
            "producer": "optimizer",
            "training_dl_enabled": False,
            "parameter_set": _p12_set,
            "family": "full",
            "policy": _p12_policy,
            "optimizer_search_protocol": _p12_protocol,
            "build_contract": _p12_contract,
            "artifact_path": _p12_target.relative_to(_p12_root).as_posix(),
            "artifact_sha256": _param_file_sha(_p12_target),
            "coverage_end": "2021-12-31",
            "optimizer_execution_evidence_contract": "optimizer_resume_determinism_v1",
            "production_promotion": False,
        }
        _p12_manifest.write_text(json.dumps(_p12_manifest_payload), encoding="utf-8")
        _stale_p12_state = _inspect_validation_artifact(
            _p12_root, policy=_p12_policy, comparison_end_date="2021-12-31",
            dataset="full", max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(_DEFAULT_PORTFOLIO_ROTATION), fixed_risk=float(_DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=_p12_set, validation_protocol_id=_p12_protocol,
        )
        _p12_rebuild_call = {}

        def _fake_p12_replay_rebuild(**kwargs):
            _p12_rebuild_call.update(kwargs)
            _work_root = Path(kwargs["output_relative_dir"])
            _source = _work_root / "active_params" / _p12_target.name
            _source.parent.mkdir(parents=True, exist_ok=True)
            _source.write_text(json.dumps(_p12_payload), encoding="utf-8")
            return {"params_path": _source}

        with patch(
            "services.optimizer.strategy_param_training.prepare_selection_historical_full_roos_params",
            side_effect=_fake_p12_replay_rebuild,
        ):
            _rebuilt_p12 = _ensure_validation_artifact(
                _p12_root, policy=_p12_policy, comparison_end_date="2021-12-31",
                dataset="full", max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
                rotation=str(_DEFAULT_PORTFOLIO_ROTATION), fixed_risk=float(_DEFAULT_FIXED_RISK),
                max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
                parameter_set_identity=_p12_set, validation_protocol_id=_p12_protocol,
            )
        _current_p12_state = _inspect_validation_artifact(
            _p12_root, policy=_p12_policy, comparison_end_date="2021-12-31",
            dataset="full", max_positions=int(_DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(_DEFAULT_PORTFOLIO_ROTATION), fixed_risk=float(_DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(_DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=_p12_set, validation_protocol_id=_p12_protocol,
        )
        check(
            "p12_prefixed_validation_artifact_auto_rebuilds_replay_without_retraining_study",
            True,
            _stale_p12_state["status"] == "VALIDATION_REPLAY_IMPLEMENTATION_STALE"
            and _stale_p12_state["action"] == "REBUILD"
            and _rebuilt_p12.get("validation_replay_rebuild") is True
            and _p12_rebuild_call.get("resume_parameter_training") is True
            and _current_p12_state["status"] == "READY"
            and _current_p12_state["action"] == "REUSE"
            and _current_p12_state.get("actual_validation_replay_implementation_contract")
                == _validation_replay_contract,
        )

    from filters.breakout_quality.strategy_optimizer_policy import build_outer_rolling_argv
    from services.optimizer.outer_rolling_plan import OuterRollingConfig, _build_rolling_folds, _resolve_config

    forward_contract = {
        "meta": {
            "first_oos_date": "2021-01-01",
            "last_oos_date": "2021-01-01",
            "train_window_months": 120,
            "oos_horizon_months": 12,
        }
    }
    forward_argv = build_outer_rolling_argv(
        args=SimpleNamespace(trials_per_fold=300),
        baseline_contract=forward_contract,
    )
    forward_config = _resolve_config(
        forward_argv,
        {},
        base_policy={"oos_start_year": 2021},
        latest_year=2026,
        latest_date="2026-03-02",
        default_trials=300,
        timing_mode=False,
    )
    forward_folds = _build_rolling_folds(forward_config)
    check(
        "outer_rolling_forward_single_member_keeps_full_12m_oos",
        ("", "2021-12-31"),
        (forward_config.oos_end_cap_date, forward_folds[-1]["oos_end_date"]),
    )

    rolling_contract = {
        "meta": {
            "first_oos_date": "2021-01-01",
            "last_oos_date": "2026-03-02",
            "train_window_months": 120,
            "oos_horizon_months": 12,
        }
    }
    rolling_argv = build_outer_rolling_argv(
        args=SimpleNamespace(trials_per_fold=300),
        baseline_contract=rolling_contract,
        oos_end_cap_date="2026-03-02",
    )
    rolling_config = _resolve_config(
        rolling_argv,
        {},
        base_policy={"oos_start_year": 2021},
        latest_year=2026,
        latest_date="2026-03-02",
        default_trials=300,
        timing_mode=False,
    )
    cutoff_folds = _build_rolling_folds(rolling_config)
    check(
        "outer_rolling_last_fold_clips_exact_research_cutoff",
        "2026-03-02",
        cutoff_folds[-1]["oos_end_date"],
    )
    from services.optimizer.outer_rolling_oos import _build_validation_progress_period_context
    progress_period_context = _build_validation_progress_period_context(
        selection_start="2011-01-01",
        selection_end="2020-12-31",
        validation_budget_spec=plateau_best_patience100_spec,
    )
    check(
        "outer_rolling_progress_period_context_uses_canonical_validation_split",
        {
            "training_start_date": "2011-01-01",
            "training_end_date": "2019-12-31",
            "validation_start_date": "2020-01-01",
            "validation_end_date": "2020-12-31",
        },
        progress_period_context,
    )
    from services.optimizer.outer_rolling_progress import _format_parallel_fold_progress_line
    timing_progress_line = _format_parallel_fold_progress_line(
        {
            "fold_idx": 6,
            "fold_count": 6,
            "oos_year": 202601,
            "oos_period": "2026-01-01~2026-03-02",
            "oos_start_date": "2026-01-01",
            "oos_end_date": "2026-03-02",
            "selection_period": "2016-01-01~2025-12-31",
            "training_start_date": "2016-01-01",
            "training_end_date": "2024-12-31",
            "validation_start_date": "2025-01-01",
            "validation_end_date": "2025-12-31",
            "show_oos": True,
            "progress_display_contract": {"show_local_min": False},
        },
        {
            "stage": "DONE",
            "status": "done",
            "selection_start": "2016-01-01",
            "selection_end": "2025-12-31",
            "best_base_score": 0.168898,
            "selected_n": 87,
            "early_stop_trial": 157,
            "training_wall_sec": 123.0,
            "validation_wall_sec": 45.0,
            "oos_wall_sec": 20.0,
            "elapsed_sec": 190.0,
        },
    )
    check(
        "outer_rolling_done_progress_shows_training_validation_oos_periods",
        True,
        "Training=16-01-01~24-12-31" in timing_progress_line
        and "Validation=25-01-01~25-12-31" in timing_progress_line
        and "OOS=26-01-01~26-03-02" in timing_progress_line
        and "N*=87" in timing_progress_line
        and "EarlyStop=T157" in timing_progress_line
        and "Train=02:03" not in timing_progress_line
        and "Val=00:45" not in timing_progress_line,
    )
    validation_stop_progress_line = _format_parallel_fold_progress_line(
        {
            "fold_idx": 1,
            "fold_count": 6,
            "oos_year": 202101,
            "oos_period": "2021-01-01~2021-12-31",
            "oos_start_date": "2021-01-01",
            "oos_end_date": "2021-12-31",
            "selection_period": "2011-01-01~2020-12-31",
            "training_start_date": "2011-01-01",
            "training_end_date": "2019-12-31",
            "validation_start_date": "2020-01-01",
            "validation_end_date": "2020-12-31",
            "show_oos": True,
            "progress_display_contract": {"show_local_min": False},
        },
        {
            "stage": "VALIDATION_SEARCH",
            "selection_start": "2011-01-01",
            "selection_end": "2020-12-31",
            "completed": 157,
            "total": 300,
            "selected_n": 57,
            "best_validation_score": 0.004399,
            "validation_wall_sec": 45.0,
            "no_improve": 100,
            "patience": 100,
            "stop_reason": "validation_plateau",
            "early_stop_trial": 157,
        },
    )
    check(
        "outer_rolling_validation_progress_shows_early_stop_trial",
        True,
        "Training=11-01-01~19-12-31" in validation_stop_progress_line
        and "Validation=20-01-01~20-12-31" in validation_stop_progress_line
        and "OOS=21-01-01~21-12-31" in validation_stop_progress_line
        and "validation 157/300" in validation_stop_progress_line
        and "N*=57" in validation_stop_progress_line
        and "EarlyStop=T157" in validation_stop_progress_line
        and "Val=00:45" not in validation_stop_progress_line,
    )
    from services.optimizer import validation_budget_refit as validation_budget_refit_module
    validation_replay_session = SimpleNamespace(
        fixed_strategy_param_overrides=None,
        optimizer_runtime_context=lambda: ExitStack(),
    )
    validation_replay_trial = SimpleNamespace(number=0)
    with patch.object(
        validation_budget_refit_module,
        "build_effective_trial_params_payload",
        return_value={"atr_len": 23, "stop_mult": 2.7},
    ):
        single_policy_signature = validation_budget_refit_module._policy_replay_signature(
            session=validation_replay_session,
            item={"trial": validation_replay_trial},
            spec=plateau_best_validation_spec,
        )
    check(
        "validation_replay_signature_supports_single_finalist_policy",
        True,
        isinstance(single_policy_signature, str) and len(single_policy_signature) == 64,
    )
    with patch.object(
        validation_budget_refit_module,
        "_evaluate_period_oos",
        return_value={"period": {}},
    ) as single_validation_replay, patch.object(
        validation_budget_refit_module,
        "_extract_period_metrics",
        return_value={
            "oos_score": 0.123456,
            "oos_plain_romd_score": 1.75,
            "ret_pct": 14.0,
            "mdd_pct": 8.0,
        },
    ):
        single_validation_record = validation_budget_refit_module._evaluate_policy_item_on_validation(
            session=validation_replay_session,
            item={"trial": validation_replay_trial},
            spec=plateau_best_validation_spec,
            validation_start_date="2020-01-01",
            validation_end_date="2020-12-31",
        )
    check(
        "validation_replay_executes_single_finalist_policy_without_ensemble",
        True,
        single_validation_replay.call_count == 1
        and single_validation_record == {
            "available": True,
            "validation_score": 0.123456,
            "plain_romd_score": 1.75,
            "return_pct": 14.0,
            "mdd_pct": 8.0,
            "member_count": 1,
        },
    )

    class _LegacyValidationStateStudy:
        def __init__(self, key, protocol):
            self.user_attrs = {
                key: {
                    "schema_version": 1,
                    "protocol": protocol,
                    "last_observed_trials": 90,
                    "best_trials": 42,
                    "best_validation_score": 0.000852,
                    "last_improvement_trial": 42,
                    "stop_trial": None,
                    "stop_reason": None,
                    "signature_cache": {"legacy": {"validation_score": 0.0}},
                    "observation_results": [{"trials": 90, "validation_score": 0.0}],
                }
            }

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    validation_state_controller = validation_budget_refit_module.AdaptiveValidationBudgetController(
        session=validation_replay_session,
        objective_mode="split_train_romd",
        spec=plateau_best_validation_spec,
        validation_start_date="2020-01-01",
        validation_end_date="2020-12-31",
    )
    legacy_state_study = _LegacyValidationStateStudy(
        validation_state_controller.state_key,
        plateau_best_validation_spec.protocol,
    )
    rebuilt_validation_state = validation_state_controller._load_state(legacy_state_study)
    check(
        "adaptive_validation_old_replay_state_is_rebuilt_without_discarding_study",
        True,
        rebuilt_validation_state["schema_version"]
            == validation_state_controller._STATE_SCHEMA_VERSION
        and rebuilt_validation_state["last_observed_trials"] == 0
        and rebuilt_validation_state["best_trials"] is None
        and rebuilt_validation_state["signature_cache"] == {}
        and rebuilt_validation_state["observation_results"] == [],
    )
    from config.strategy_params import ACTIVE_STRATEGY_PARAM_PROFILE
    active_fitting_seed_protocol = get_strategy_param_fitting_seed_protocol(
        active_param_profile.fitting_seed_protocol
    )
    check(
        "strategy_param_active_profile_and_seed_protocol_are_config_driven",
        True,
        active_param_profile.profile_id == str(ACTIVE_STRATEGY_PARAM_PROFILE)
        and active_param_profile.parameter_set_identity.strip() != ""
        and active_fitting_seed_protocol.trials_per_seed > 0
        and active_fitting_seed_protocol.seed_count
            == len(active_fitting_seed_protocol.resolved_seeds)
        and len(set(active_fitting_seed_protocol.resolved_seeds))
            == len(active_fitting_seed_protocol.resolved_seeds)
        and (
            not active_fitting_seed_protocol.enabled
            or active_fitting_seed_protocol.seed_count >= 2
        )
        and validation_fitting_seed_protocol.seed_mode == "fixed_single_seed"
        and len(validation_fitting_seed_protocol.resolved_seeds) == 1,
    )
    from core.strategy_param_policy import build_replicated_fitting_seed_ensemble_policy
    from core.training_policy import get_robustness_benchmark_policy_snapshot
    benchmark_snapshot = get_robustness_benchmark_policy_snapshot()
    replication_seed = int(benchmark_snapshot["resolved_seeds"][0])
    replicated_p6_policy = build_replicated_fitting_seed_ensemble_policy(
        production_multiseed_param_profile.fitting_seed_protocol,
        replication_seed=replication_seed,
    )
    check(
        "strategy_param_replicated_fitting_bundle_preserves_registered_seed_count_and_orthogonality",
        True,
        replicated_p6_policy["seed_count"]
            == production_multiseed_fitting_seed_protocol.seed_count
        and replicated_p6_policy["generator_seed"] == replication_seed
        and len(replicated_p6_policy["resolved_seeds"])
            == production_multiseed_fitting_seed_protocol.seed_count
        and len(set(replicated_p6_policy["resolved_seeds"]))
            == production_multiseed_fitting_seed_protocol.seed_count
        and tuple(replicated_p6_policy["resolved_seeds"])
            != tuple(production_multiseed_fitting_seed_protocol.resolved_seeds),
    )
    from config.strategy_compare import (
        STRATEGY_COMPARE_CURRENT_OOS_PROFILE,
        STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE,
        STRATEGY_COMPARE_CURRENT_OOS_ROBUSTNESS,
        STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS,
    )
    from core.strategy_compare_policy import (
        get_strategy_comparison_settings,
        get_strategy_multi_seed_robustness_settings,
    )
    current_oos_settings = get_strategy_comparison_settings(
        STRATEGY_COMPARE_CURRENT_OOS_PROFILE
    )
    current_rolling_settings = get_strategy_comparison_settings(
        STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
    )
    current_oos_robustness = get_strategy_multi_seed_robustness_settings(
        STRATEGY_COMPARE_CURRENT_OOS_ROBUSTNESS
    )
    current_rolling_robustness = get_strategy_multi_seed_robustness_settings(
        STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS
    )
    check(
        "strategy_param_current_compare_and_robustness_share_one_canonical_suite",
        True,
        current_oos_settings.suite_id == current_rolling_settings.suite_id
        and current_oos_robustness.suite_id == current_rolling_robustness.suite_id
        and current_oos_robustness.suite_id == current_oos_settings.suite_id,
    )
    from services.research.strategy_multi_seed_robustness import _benchmark_param_binding
    raw_robustness_binding = _benchmark_param_binding(
        settings=current_oos_settings,
        robustness=current_oos_robustness,
        arm=current_oos_settings.arms["C93"],
        seed=int(current_oos_robustness.resolved_seeds[0]),
    )
    check(
        "raw_price_compare_arm_robustness_is_derived_from_same_suite_and_profile",
        True,
        int(current_oos_robustness.seed_count) == 4
        and len(tuple(current_oos_robustness.resolved_seeds)) == 4
        and str(raw_robustness_binding.get("producer_variant")) == "raw_price_isolated_v1"
        and str(raw_robustness_binding.get("training_profile_id"))
            == active_param_profile.profile_id
        and "robustness/end_to_end_v1/seed_"
            in Path(raw_robustness_binding["path"]).as_posix(),
    )
    from services.optimizer.strategy_param_service import (
        _benchmark_schedule_build_contract,
        _canonical_schedule_build_contract,
        _fixed_multiseed_build_contract,
        ensure_fixed_multiseed_robustness_strategy_parameter_artifact,
        fixed_multiseed_robustness_strategy_param_artifact_path,
        fixed_multiseed_robustness_strategy_param_manifest_path,
        fixed_multiseed_strategy_param_artifact_path,
        inspect_fixed_multiseed_strategy_parameter_artifact,
        inspect_fixed_multiseed_robustness_strategy_parameter_artifact,
        inspect_robustness_benchmark_strategy_parameter_artifact,
    )
    from config.execution_policy import (
        DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS, DEFAULT_PORTFOLIO_ROTATION,
    )
    import config.training_policy as training_policy_settings
    from core.strategy_param_policy import resolve_canonical_strategy_param_training_profile
    full_best_profile_for_fit = resolve_canonical_strategy_param_training_profile(
        "full", "base-finalist-best"
    )
    full_best_fit_protocol = get_strategy_param_fitting_seed_protocol(
        full_best_profile_for_fit.fitting_seed_protocol
    )
    canonical_full_contract = _canonical_schedule_build_contract(family="full")
    check(
        "strategy_param_registered_fitting_protocol_owns_canonical_seed_and_trials",
        True,
        int(canonical_full_contract["optimizer_seed"])
            == int(full_best_fit_protocol.generator_seed)
        and int(canonical_full_contract["trials_per_fold"])
            == int(full_best_fit_protocol.trials_per_seed),
    )
    with TemporaryDirectory() as temp_dir:
        legacy_root = Path(temp_dir)
        legacy_contract = _fixed_multiseed_build_contract(
            training_profile_id=production_multiseed_param_profile.profile_id,
            dataset=production_multiseed_param_profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
        legacy_path = fixed_multiseed_strategy_param_artifact_path(
            legacy_root, policy=production_multiseed_param_profile.parameter_policy
        )
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_seed = int(legacy_contract["seed_ensemble_policy"]["resolved_seeds"][0])
        legacy_path.write_text(
            json.dumps({
                "type": "active_param_ensemble",
                "mode": "rolling",
                "selector": normalize_strategy_param_policy(
                    production_multiseed_param_profile.parameter_policy
                ),
                "params_ensemble_by_effective_date": {
                    "2021-01-01": [{
                        "seed": legacy_seed,
                        "params": {"high_len": 260, "atr_len": 23},
                    }]
                },
                "breakout_quality_param_adaptation": {
                    "parameter_set": legacy_contract["parameter_set"],
                    "training_dl_enabled": False,
                    "optimizer_search_protocol": legacy_contract["optimizer_search_protocol"],
                    "seed_ensemble_policy": legacy_contract["seed_ensemble_policy"],
                },
            }),
            encoding="utf-8",
        )
        legacy_p6_state = inspect_fixed_multiseed_strategy_parameter_artifact(
            legacy_root,
            training_profile_id=production_multiseed_param_profile.profile_id,
            policy=production_multiseed_param_profile.parameter_policy,
            comparison_end_date="2021-12-31",
            dataset=production_multiseed_param_profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
        from services.research.strategy_compare_preparation_status import (
            _collect_parameter_artifact_status,
        )
        legacy_source_id = next(
            source_id
            for source_id, source in current_oos_settings.parameter_sources.items()
            if source.builder is not None
            and source.builder.builder_type == "optimizer_fixed_multiseed_strategy_params"
            and str(dict(source.builder.options).get("training_profile_id") or "")
                == "production_multiseed_current"
        )
        legacy_actions = []
        legacy_rows, _legacy_paths, _legacy_arm_paths = _collect_parameter_artifact_status(
            root=legacy_root,
            settings=current_oos_settings,
            required_param_sources={legacy_source_id},
            comparison_start="2021-01-01",
            comparison_end="2021-12-31",
            dl_model_ready={},
            comparison_period_dependencies=(),
            artifact_identities={},
            actions=legacy_actions,
        )
        legacy_plan_action = next(
            row for row in legacy_actions if row.artifact_key == f"param:{legacy_source_id}"
        )
        with patch.object(
            training_policy_settings,
            "MIN_ANNUAL_TRADES",
            float(training_policy_settings.MIN_ANNUAL_TRADES) + 1.0,
        ):
            drifted_p6_state = inspect_fixed_multiseed_strategy_parameter_artifact(
                legacy_root,
                training_profile_id=production_multiseed_param_profile.profile_id,
                policy=production_multiseed_param_profile.parameter_policy,
                comparison_end_date="2021-12-31",
                dataset=production_multiseed_param_profile.dataset,
                max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
                rotation=str(DEFAULT_PORTFOLIO_ROTATION),
                fixed_risk=float(DEFAULT_FIXED_RISK),
                max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            )
    check(
        "strategy_param_legacy_unbound_p6_routes_to_canonical_rebuild_when_source_registration_matches",
        ("LEGACY_SCIENTIFIC_IDENTITY_UNBOUND", "REBUILD", True),
        (
            legacy_p6_state["status"],
            legacy_p6_state["action"],
            bool(legacy_p6_state["scientific_identity"]["registration_matches"]),
        ),
    )
    check(
        "strategy_compare_legacy_unbound_p6_dependency_plan_remains_buildable_not_blocked",
        True,
        legacy_rows[legacy_source_id]["action"] in {"BUILD", "REBUILD"}
        and legacy_plan_action.builder_type == "optimizer_fixed_multiseed_strategy_params",
    )
    check(
        "strategy_param_true_registered_p6_scientific_drift_remains_blocked",
        ("REGISTERED_SCIENTIFIC_CONTRACT_DRIFT", "BLOCKED"),
        (drifted_p6_state["status"], drifted_p6_state["action"]),
    )

    benchmark_full_contract = _benchmark_schedule_build_contract(
        benchmark_id=str(benchmark_snapshot["benchmark_id"]),
        seed=replication_seed,
        family="full",
        policy="base_finalist_best",
        dataset="full",
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
    )
    check(
        "strategy_param_robustness_trials_are_owned_by_registered_fitting_protocol",
        int(full_best_fit_protocol.trials_per_seed),
        int(benchmark_full_contract["trials_per_fold"]),
    )
    with patch.object(
        training_policy_settings,
        "OPTIMIZER_RANDOM_SEED_DEFAULT",
        int(training_policy_settings.OPTIMIZER_RANDOM_SEED_DEFAULT) + 1000,
    ), patch.object(
        training_policy_settings,
        "OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT",
        int(training_policy_settings.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT) + 111,
    ):
        mutated_canonical_contract = _canonical_schedule_build_contract(family="full")
        mutated_benchmark_contract = _benchmark_schedule_build_contract(
            benchmark_id=str(benchmark_snapshot["benchmark_id"]),
            seed=replication_seed,
            family="full",
            policy="base_finalist_best",
            dataset="full",
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
    check(
        "strategy_param_generic_optimizer_seed_trial_defaults_do_not_override_registered_fit_protocol",
        True,
        int(mutated_canonical_contract["optimizer_seed"])
            == int(canonical_full_contract["optimizer_seed"])
        and int(mutated_canonical_contract["trials_per_fold"])
            == int(canonical_full_contract["trials_per_fold"])
        and int(mutated_benchmark_contract["trials_per_fold"])
            == int(benchmark_full_contract["trials_per_fold"]),
    )
    from core.strategy_param_scientific_contract import build_strategy_param_scientific_contract
    full_best_fp = build_strategy_param_scientific_contract(
        project_root, full_best_profile_for_fit.profile_id
    )["scientific_fingerprint_sha256"]
    with patch.object(
        training_policy_settings,
        "OPTIMIZER_RANDOM_SEED_DEFAULT",
        int(training_policy_settings.OPTIMIZER_RANDOM_SEED_DEFAULT) + 2000,
    ), patch.object(
        training_policy_settings,
        "OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT",
        int(training_policy_settings.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT) + 222,
    ):
        mutated_full_best_fp = build_strategy_param_scientific_contract(
            project_root, full_best_profile_for_fit.profile_id
        )["scientific_fingerprint_sha256"]
    check(
        "strategy_param_registered_fit_fingerprint_ignores_generic_optimizer_seed_trial_defaults",
        full_best_fp,
        mutated_full_best_fp,
    )
    with TemporaryDirectory() as temp_dir:
        single_seed_robustness_state = inspect_robustness_benchmark_strategy_parameter_artifact(
            temp_dir,
            benchmark_id=str(benchmark_snapshot["benchmark_id"]),
            seed=replication_seed,
            family="full",
            evaluation_mode="rolling",
            policy="base-finalist-best",
            comparison_end_date="2021-01-01",
            dataset="full",
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
    single_seed_identity = dict(single_seed_robustness_state["scientific_identity"])
    single_seed_fitting = dict(single_seed_identity["scientific_contract"]["fitting"])
    check(
        "strategy_param_single_seed_robustness_derives_identity_from_registered_source",
        True,
        single_seed_robustness_state["action"] == "BUILD"
        and bool(single_seed_identity["source_registration_matches"])
        and int(single_seed_fitting["generator_seed"]) == replication_seed
        and list(single_seed_fitting["resolved_seeds"]) == [replication_seed]
        and int(single_seed_fitting["trials_per_seed"])
            == int(full_best_fit_protocol.trials_per_seed),
    )
    with patch.object(
        training_policy_settings,
        "MIN_ANNUAL_TRADES",
        float(training_policy_settings.MIN_ANNUAL_TRADES) + 1.0,
    ):
        with TemporaryDirectory() as temp_dir:
            drifted_single_seed_robustness_state = inspect_robustness_benchmark_strategy_parameter_artifact(
                temp_dir,
                benchmark_id=str(benchmark_snapshot["benchmark_id"]),
                seed=replication_seed,
                family="full",
                evaluation_mode="rolling",
                policy="base-finalist-best",
                comparison_end_date="2021-01-01",
                dataset="full",
                max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
                rotation=str(DEFAULT_PORTFOLIO_ROTATION),
                fixed_risk=float(DEFAULT_FIXED_RISK),
                max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            )
    check(
        "strategy_param_single_seed_robustness_blocks_registered_scientific_drift",
        ("BLOCKED", "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT"),
        (
            drifted_single_seed_robustness_state["action"],
            drifted_single_seed_robustness_state["status"],
        ),
    )
    with TemporaryDirectory() as temp_dir:
        p6_robustness_state = inspect_fixed_multiseed_robustness_strategy_parameter_artifact(
            temp_dir,
            training_profile_id="production_multiseed_current",
            benchmark_id=str(benchmark_snapshot["benchmark_id"]),
            replication_seed=replication_seed,
            policy="base-finalists-agree",
            comparison_end_date="2021-01-01",
            dataset="full",
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
    p6_robustness_contract = dict(p6_robustness_state["build_contract"])
    p6_robustness_seed_policy = dict(p6_robustness_contract["seed_ensemble_policy"])
    check(
        "strategy_param_round3_p6_robustness_artifact_identity_keeps_outer_replication_and_inner_bundle_separate",
        True,
        p6_robustness_state["action"] == "BUILD"
        and p6_robustness_contract["robustness_replication_seed"] == replication_seed
        and p6_robustness_contract["robustness_seed_semantics"]
            == "replication_seed_generates_fitting_seed_bundle"
        and p6_robustness_seed_policy["seed_count"]
            == production_multiseed_fitting_seed_protocol.seed_count
        and p6_robustness_seed_policy["generator_seed"] == replication_seed
        and replication_seed not in set(p6_robustness_seed_policy["resolved_seeds"]),
    )
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        captured_p6_robustness_build = {}

        def _fake_p6_robustness_build(**kwargs):
            captured_p6_robustness_build.update(kwargs)
            work_root = Path(kwargs["output_relative_dir"])
            source = work_root / "active_params" / "roos_base_finalists_agree.json"
            source.parent.mkdir(parents=True, exist_ok=True)
            seed_policy = dict(kwargs["seed_ensemble_policy_override"])
            members = [
                {
                    "seed": int(seed),
                    "params": {"high_len": 260 + idx, "atr_len": 23},
                }
                for idx, seed in enumerate(seed_policy["resolved_seeds"])
            ]
            source.write_text(
                json.dumps({
                    "type": "active_param_ensemble",
                    "mode": "rolling",
                    "selector": "base_finalists_agree",
                    "params_ensemble_by_effective_date": {"2021-01-01": members},
                    "random_seed_ensemble": seed_policy,
                    "breakout_quality_param_adaptation": {
                        "parameter_set": kwargs["parameter_set_identity"],
                        "optimizer_search_protocol": kwargs["optimizer_search_protocol"],
                        "seed_ensemble_policy": seed_policy,
                    },
                }),
                encoding="utf-8",
            )
            return {"params_path": source}

        with patch(
            "services.optimizer.strategy_param_training.prepare_selection_historical_full_roos_params",
            side_effect=_fake_p6_robustness_build,
        ):
            built_p6_robustness = ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
                temp_root,
                training_profile_id="production_multiseed_current",
                benchmark_id=str(benchmark_snapshot["benchmark_id"]),
                replication_seed=replication_seed,
                policy="base-finalists-agree",
                comparison_end_date="2021-01-01",
                dataset="full",
                max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
                rotation=str(DEFAULT_PORTFOLIO_ROTATION),
                fixed_risk=float(DEFAULT_FIXED_RISK),
                max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
                quiet=True,
            )
        rebuilt_p6_robustness_state = inspect_fixed_multiseed_robustness_strategy_parameter_artifact(
            temp_root,
            training_profile_id="production_multiseed_current",
            benchmark_id=str(benchmark_snapshot["benchmark_id"]),
            replication_seed=replication_seed,
            policy="base-finalists-agree",
            comparison_end_date="2021-01-01",
            dataset="full",
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
        p6_manifest_path = fixed_multiseed_robustness_strategy_param_manifest_path(
            temp_root,
            training_profile_id="production_multiseed_current",
            benchmark_id=str(benchmark_snapshot["benchmark_id"]),
            replication_seed=replication_seed,
        )
        p6_manifest = json.loads(p6_manifest_path.read_text(encoding="utf-8"))
        p6_manifest["build_contract"]["publication_schema"] = "legacy_workspace_v1"
        p6_manifest_path.write_text(json.dumps(p6_manifest), encoding="utf-8")
        p6_publication_metadata_drift_state = inspect_fixed_multiseed_robustness_strategy_parameter_artifact(
            temp_root,
            training_profile_id="production_multiseed_current",
            benchmark_id=str(benchmark_snapshot["benchmark_id"]),
            replication_seed=replication_seed,
            policy="base-finalists-agree",
            comparison_end_date="2021-01-01",
            dataset="full",
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
    check(
        "strategy_param_round3_p6_robustness_builder_passes_inner_bundle_and_publishes_ready_artifact",
        True,
        built_p6_robustness["action"] == "BUILD"
        and rebuilt_p6_robustness_state["ready"]
        and p6_publication_metadata_drift_state["action"] == "REUSE"
        and bool(
            rebuilt_p6_robustness_state["scientific_identity"][
                "source_registration_matches"
            ]
        )
        and int(captured_p6_robustness_build["optimizer_seed"]) == replication_seed
        and tuple(captured_p6_robustness_build["seed_ensemble_policy_override"]["resolved_seeds"])
            == tuple(p6_robustness_seed_policy["resolved_seeds"])
        and len(captured_p6_robustness_build["seed_ensemble_policy_override"]["resolved_seeds"])
            == production_multiseed_fitting_seed_protocol.seed_count
        and replication_seed
            not in set(captured_p6_robustness_build["seed_ensemble_policy_override"]["resolved_seeds"]),
    )
    from services.optimizer.outer_rolling_parallel_progress import (
        PARALLEL_FOLD_PROGRESS_PREFIX,
        _ParallelFoldProgressLogFilter,
    )
    progress_log_handle = io.StringIO()

    class _SyntheticTTYMirror(io.StringIO):
        def isatty(self):
            return True

        def fileno(self):
            return 123

    progress_live_handle = _SyntheticTTYMirror()
    progress_filter = _ParallelFoldProgressLogFilter(
        progress_log_handle, mirror_handle=progress_live_handle
    )
    progress_filter.write(PARALLEL_FOLD_PROGRESS_PREFIX + '{"stage":"SEARCH"}\n')
    progress_filter.write("[1/1] PHASE 1/3 | INNER TRAIN SEARCH | trials=0/300\n")
    progress_filter.flush()
    check(
        "validation_budget_sequential_fold_keeps_machine_progress_in_log_and_mirrors_human_progress_to_console",
        True,
        PARALLEL_FOLD_PROGRESS_PREFIX in progress_log_handle.getvalue()
        and "INNER TRAIN SEARCH" not in progress_log_handle.getvalue()
        and PARALLEL_FOLD_PROGRESS_PREFIX not in progress_live_handle.getvalue()
        and "INNER TRAIN SEARCH" in progress_live_handle.getvalue()
        and progress_filter.isatty()
        and progress_filter.fileno() == 123,
    )
    from services.optimizer.outer_rolling_live_boards import OptimizerSeedEnsembleProgressBoard
    tty_log_handle = io.StringIO()
    tty_live_handle = _SyntheticTTYMirror()
    tty_progress_filter = _ParallelFoldProgressLogFilter(
        tty_log_handle, mirror_handle=tty_live_handle
    )
    tty_contexts = [
        {
            "fold_idx": 1,
            "fold_count": 1,
            "seed_index": seed_index,
            "seed_count": 4,
            "selection_start": "2011-01-01",
            "selection_end": "2020-12-31",
            "oos_period": "2021-01-01~2021-12-31",
        }
        for seed_index in range(1, 5)
    ]
    with patch.dict(os.environ, {"TERM": "xterm"}, clear=False), redirect_stdout(tty_progress_filter):
        tty_board = OptimizerSeedEnsembleProgressBoard(tty_contexts, header="synthetic")
        tty_board.render(force=True)
        tty_board.update_many(
            {(1, seed_index): {"stage": "START"} for seed_index in range(1, 5)},
            force=True,
        )
        tty_board.close()
    check(
        "fold_log_wrapper_preserves_existing_seed_live_board_inline_refresh",
        True,
        tty_board.inline
        and "\x1b[4A" in tty_live_handle.getvalue()
        and "synthetic" in tty_live_handle.getvalue()
        and tty_log_handle.getvalue() == "",
    )
    check(
        "validation_budget_sequential_fold_reuses_shared_raw_cache_and_exposes_three_live_phases",
        True,
        '"mirror_console": True' in optimizer_outer_rolling_source
        and '"shared_raw_context": shared_raw_context' in optimizer_outer_rolling_source
        and 'shared raw cache installed' in optimizer_outer_rolling_source
        and 'PHASE 1/3 | INNER TRAIN SEARCH' in optimizer_outer_rolling_source
        and 'PHASE 2/3 | VALIDATION BUDGET SELECTION' in optimizer_outer_rolling_source
        and 'PHASE 3/3 | FULL REFIT' in optimizer_outer_rolling_source,
    )
    check(
        "fixed_multiseed_single_fold_has_live_seed_board_and_avoids_unused_parent_raw_cache",
        True,
        "seed_ensemble_process_parallel" in optimizer_outer_rolling_source
        and "progress_board = OptimizerSeedEnsembleProgressBoard" in optimizer_outer_rolling_source
        and "_read_latest_parallel_fold_seed_progresses_for_task(task)" in optimizer_outer_rolling_source
        and "last_progress_render_at" not in optimizer_outer_rolling_source
        and "wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)" in optimizer_outer_rolling_source
        and "略過父程序 shared raw cache" in optimizer_outer_rolling_source
        and '"mirror_console": True' in optimizer_outer_rolling_source,
    )
    check(
        "optimizer_seed_and_parallel_fold_live_boards_reuse_fixed_progress_block_ssot",
        True,
        "from core.display_common import FixedProgressBlock" in optimizer_live_boards_source
        and optimizer_live_boards_source.count("FixedProgressBlock()") == 2
        and "sys.stdout.write" not in optimizer_live_boards_source
        and "_noninline_seed_progress_key" not in optimizer_live_boards_source
        and "_build_noninline_render_key" not in optimizer_live_boards_source,
    )
    compact_contexts = [
        {
            "fold_idx": 1,
            "fold_count": 1,
            "seed_index": seed_index,
            "seed_count": 4,
            "selection_start": "2011-01-01",
            "selection_end": "2020-12-31",
            "oos_period": "2021-01-01~2021-12-31",
        }
        for seed_index in range(1, 5)
    ]
    compact_output = io.StringIO()
    with redirect_stdout(compact_output):
        compact_board = OptimizerSeedEnsembleProgressBoard(compact_contexts, header="synthetic")
        compact_board.render(force=True)
        compact_board.update_many({(1, seed_index): {"stage": "START"} for seed_index in range(1, 5)})
        compact_board.update_many({
            (1, 1): {"stage": "OPTIMIZER_SEARCH", "completed": 10, "total": 300},
            (1, 2): {"stage": "OPTIMIZER_SEARCH", "completed": 20, "total": 300},
            (1, 3): {"stage": "OPTIMIZER_SEARCH", "completed": 25, "total": 300},
            (1, 4): {"stage": "OPTIMIZER_SEARCH", "completed": 15, "total": 300},
        })
        compact_board.close()
    check(
        "fixed_multiseed_redirected_progress_reuses_fixed_progress_block_quiet_contract",
        True,
        not compact_board.inline
        and compact_output.getvalue() == "",
    )

    # Resume progress is display-only but must rehydrate persisted study truth:
    # completed trials must not restart at zero elapsed, and best_base must not
    # disappear merely because a new Python process opened the same study.
    from datetime import datetime, timedelta, timezone
    from services.optimizer.outer_rolling_progress import (
        PARALLEL_FOLD_PROGRESS_PREFIX as _RESUME_PROGRESS_PREFIX,
        _FoldLogSearchProgress as _ResumeFoldLogSearchProgress,
        _collect_seed_progress_phase_metrics as _collect_resume_phase_metrics,
        _effective_resume_total_elapsed as _resume_total_elapsed,
        _safe_progress_json_loads as _resume_progress_json_loads,
        build_optimizer_study_progress_snapshot as _build_resume_progress_snapshot,
    )

    resume_base = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
    resume_trials = [
        SimpleNamespace(
            value=1.25,
            datetime_start=resume_base,
            datetime_complete=resume_base + timedelta(seconds=3.0),
        ),
        SimpleNamespace(
            value=7.093,
            datetime_start=resume_base + timedelta(seconds=3.2),
            datetime_complete=resume_base + timedelta(seconds=6.2),
        ),
        SimpleNamespace(
            value=0.0,
            datetime_start=resume_base + timedelta(minutes=10),
            datetime_complete=resume_base + timedelta(minutes=10, seconds=3.0),
        ),
    ]
    resume_study = SimpleNamespace(trials=resume_trials)
    resume_snapshot = _build_resume_progress_snapshot(
        resume_study, best_trial_resolver=lambda study: study.trials[1]
    )
    check(
        "optimizer_resume_progress_rehydrates_best_base_and_active_search_elapsed",
        True,
        resume_snapshot.get("completed_trials") == 3
        and float(resume_snapshot.get("best_base_score")) == 7.093
        and len(list(resume_snapshot.get("search_active_intervals") or [])) == 2
        and float(resume_snapshot.get("search_active_elapsed_sec", 0.0)) > 9.0,
    )

    resume_progress = _ResumeFoldLogSearchProgress(
        fold_idx=1,
        fold_count=6,
        oos_year=2021,
        selection_start="2011-01-01",
        selection_end="2020-12-31",
        total_trials=1000,
        seed_context={
            "seed_ensemble_member_index": 1,
            "seed_ensemble_member_count": 4,
            "seed": 42,
        },
        initial_best_score=resume_snapshot.get("best_base_score"),
        historical_search_intervals=resume_snapshot.get("search_active_intervals"),
    )
    resume_emit_output = io.StringIO()
    with redirect_stdout(resume_emit_output):
        resume_progress.emit(1000, force=True)
    resume_event_line = next(
        (
            line
            for line in resume_emit_output.getvalue().splitlines()
            if line.startswith(_RESUME_PROGRESS_PREFIX)
        ),
        "",
    )
    resume_event = _resume_progress_json_loads(
        resume_event_line.split("\t", 1)[1] if "\t" in resume_event_line else "{}"
    )
    check(
        "optimizer_resume_progress_first_event_keeps_best_base_and_elapsed_instead_of_na_zero",
        True,
        int(resume_event.get("completed", 0) or 0) == 1000
        and float(resume_event.get("best_base_score")) == 7.093
        and float(resume_event.get("elapsed_sec", 0.0)) > 9.0
        and len(list(resume_event.get("search_active_intervals") or [])) == 2,
    )

    resume_phase_metrics = _collect_resume_phase_metrics(
        [
            {
                "fold_idx": 1,
                "seed_ensemble_member_index": 1,
                "completed": 1000,
                "search_active_intervals": resume_snapshot.get("search_active_intervals"),
            }
        ]
    )
    check(
        "optimizer_resume_header_uses_persisted_active_search_time_instead_of_new_process_clock",
        True,
        float(resume_phase_metrics.get("search_wall_elapsed_sec", 0.0)) > 9.0
        and _resume_total_elapsed(
            2.0, search_wall_elapsed_sec=resume_phase_metrics.get("search_wall_elapsed_sec")
        ) > 9.0,
    )

    import core.strategy_param_policy as strategy_param_policy
    disabled_profiles = {
        "synthetic_no_validation": {
            "label": "Synthetic",
            "strategy_id": "synthetic",
            "filter_profile": "synthetic_filter",
            "parameter_policy": "synthetic-policy",
            "parameter_set_identity": "SYNTHETIC",
            "dataset": "full",
            "schedule_start_date": "2021-01-01",
            "coverage_end_date": "2021-12-31",
            "validation": {"enabled": False},
            "fitting_seed_protocol": "single_seed_42",
        }
    }
    with patch.object(strategy_param_policy, "STRATEGY_PARAM_PROFILES", disabled_profiles):
        disabled_profile = strategy_param_policy.get_strategy_param_training_profile("synthetic_no_validation")
    check("strategy_param_profile_can_disable_validation_without_menu_change", True, not disabled_profile.validation_enabled and disabled_profile.validation_protocol is None and disabled_profile.filter_profile == "synthetic_filter")
    import services.research.strategy_param_application as strategy_param_application
    producer_call = {}
    def _capture_validation_producer(*args, **kwargs):
        producer_call.update(kwargs)
        return {"action": "REUSE"}
    with patch.object(strategy_param_application, "ensure_fixed_multiseed_strategy_parameter_artifact", side_effect=_capture_validation_producer):
        strategy_param_application._ensure_single(
            production_multiseed_param_profile, required_end="2021-01-01"
        )
    check(
        "strategy_param_menu_delegates_config_identity_to_optimizer_owner",
        True,
        producer_call.get("training_profile_id") == production_multiseed_param_profile.profile_id
        and producer_call.get("comparison_end_date") == "2021-01-01",
    )
    force_call = {}
    def _capture_force_producer(*args, **kwargs):
        force_call.update(kwargs)
        return {"action": "FORCE_REBUILD"}
    with patch.object(strategy_param_application, "ensure_fixed_multiseed_strategy_parameter_artifact", side_effect=_capture_force_producer):
        strategy_param_application._ensure_single(
            production_multiseed_param_profile,
            required_end="2021-01-01",
            force_rebuild=True,
        )
    check(
        "strategy_param_force_rebuild_delegates_fresh_execution_to_optimizer_owner",
        True,
        force_call.get("force_rebuild") is True
        and force_call.get("resume_parameter_training") is False
        and force_call.get("comparison_end_date") == "2021-01-01",
    )
    producer_execution_source = (project_root / "services" / "research" / "producer_execution.py").read_text(encoding="utf-8")
    model_app_source = (project_root / "services" / "research" / "breakout_quality_application.py").read_text(encoding="utf-8")
    check(
        "research_training_producers_share_one_execution_action_contract",
        True,
        "prompt_producer_execution_action" in strategy_param_app_source
        and "prompt_producer_execution_action" in model_app_source
        and "強制重新訓練" in producer_execution_source
        and "查看既有訓練報表" in producer_execution_source,
    )
    from services.research.producer_execution import (
        PRODUCER_ACTION_FORCE_REBUILD, PRODUCER_ACTION_VIEW_REPORT,
        prompt_producer_execution_action,
    )
    with patch("builtins.input", return_value=""), redirect_stdout(io.StringIO()):
        ready_default = prompt_producer_execution_action(ready=True, can_view_report=True)
    with patch("builtins.input", return_value="2"), redirect_stdout(io.StringIO()):
        ready_force = prompt_producer_execution_action(ready=True, can_view_report=True)
    check(
        "ready_producer_defaults_to_view_report_not_retrain",
        PRODUCER_ACTION_VIEW_REPORT, ready_default.action,
    )
    check(
        "ready_producer_force_rebuild_is_explicit_secondary_action",
        PRODUCER_ACTION_FORCE_REBUILD, ready_force.action,
    )
    check(
        "force_rebuild_selection_is_the_only_confirmation_step",
        True,
        "confirm_force_rebuild" not in producer_execution_source
        and "輸入 REBUILD" not in producer_execution_source
        and "confirm_force_rebuild" not in strategy_param_app_source
        and "confirm_force_rebuild" not in model_app_source,
    )
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        artifact = temp_root / "params.json"
        artifact.write_text(json.dumps({
            "mode": "rolling",
            "type": "active_param_ensemble",
            "params_ensemble_by_effective_date": {
                "2021-01-01": [{"seed": 42, "params": {"high_len": 260}}]
            },
            "breakout_quality_param_adaptation": {
                "parameter_set": validation_param_profile.parameter_set_identity,
                "optimizer_search_protocol": validation_param_profile.validation_protocol,
            },
            "folds": [{
                "oos_start_date": "2021-01-01",
                "oos_end_date": "2021-12-31",
                "training_protocol": {
                    "protocol": "validation_budget_refit_v1",
                    "validation_start_date": "2020-01-01",
                    "validation_end_date": "2020-12-31",
                    "selected_trials": 100,
                    "selected_validation_score": 1.2345,
                    "oos_feedback_used": False,
                    "checkpoint_results": [{
                        "trials": 100, "available": True,
                        "validation_score": 1.2345, "plain_romd_score": 9.87,
                        "return_pct": 12.3, "mdd_pct": 4.5, "member_count": 3,
                    }],
                },
            }],
        }), encoding="utf-8")
        report_plan = {
            "work": strategy_param_application.WORK_TYPES[0],
            "profile": validation_param_profile,
            "rows": [{"item": "Canonical seed 42", "path": "params.json"}],
        }
        with patch.object(strategy_param_application, "PROJECT_ROOT", temp_root):
            rendered_training_report = strategy_param_application.render_strategy_param_training_report(report_plan)
    check(
        "strategy_param_existing_training_report_reads_persisted_validation_evidence_without_retraining",
        True,
        "Selected N*  ：100" in rendered_training_report
        and "1.2345" in rendered_training_report
        and "2020-01-01" in rendered_training_report
        and "100" in rendered_training_report,
    )
    from services.research import breakout_quality_application as model_application
    with TemporaryDirectory() as temp_dir:
        publication = Path(temp_dir) / "canonical_model"
        publication.mkdir(parents=True, exist_ok=True)
        sentinel = publication / "sentinel.txt"
        sentinel.write_text("old-publication", encoding="utf-8")
        rolled_back = False
        try:
            with model_application._force_rebuild_publication_guard(publication):
                publication.mkdir(parents=True, exist_ok=True)
                (publication / "partial.txt").write_text("new-partial", encoding="utf-8")
                raise RuntimeError("synthetic fresh producer failure")
        except RuntimeError:
            rolled_back = True
        check(
            "force_rebuild_failure_restores_previous_canonical_publication",
            True,
            rolled_back
            and sentinel.is_file()
            and sentinel.read_text(encoding="utf-8") == "old-publication"
            and not (publication / "partial.txt").exists()
            and not list(publication.parent.glob("canonical_model.pre_force_rebuild_*")),
        )
    with patch("builtins.input", side_effect=["S", "", "1"]), patch("tools.optimizer.study_utils.safe_prompt_choice", return_value="1"):
        interactive_restart_request = study_utils._resolve_interactive_optimizer_run_request()
    check("interactive_study_memory_one_routes_to_restart", True, interactive_restart_request.get("study_db_action") == "restart" and interactive_restart_request.get("n_trials") == 1 and interactive_restart_request.get("action") == study_utils.OPTIMIZER_MENU_ACTION_TRAIN)

    # Canonical Strategy Parameter scientific-contract boundary.  Round 1 is
    # observation-only: it must detect result-changing semantics while ignoring
    # display/performance knobs, without changing legacy runtime fingerprints.
    import config.display_policy as display_policy_settings
    import config.execution_policy as execution_policy_settings
    import config.training_performance_policy as training_performance_settings
    import config.training_policy as training_policy_settings
    import strategies.breakout.schema as breakout_schema
    from core.strategy_param_policy import (
        get_active_strategy_param_training_profile,
        get_strategy_param_fitting_seed_protocol,
        get_strategy_param_validation_protocol,
    )
    from core.strategy_param_scientific_contract import (
        build_registered_strategy_param_scientific_identity,
        build_strategy_param_scientific_contract,
    )
    from core.training_policy import get_strategy_parameter_training_policy_snapshot

    active_param_profile = get_active_strategy_param_training_profile()
    active_fitting_protocol = get_strategy_param_fitting_seed_protocol(
        active_param_profile.fitting_seed_protocol
    )
    active_training_policy = get_strategy_parameter_training_policy_snapshot(
        evaluation_mode="rolling"
    )
    active_profile_id = active_param_profile.profile_id
    if active_param_profile.validation_enabled:
        active_validation_protocol = get_strategy_param_validation_protocol(
            str(active_param_profile.validation_protocol)
        )
        expected_validation_contract = {
            "enabled": True,
            **{
                key: value
                for key, value in active_validation_protocol.as_dict().items()
                if key != "protocol"
            },
        }
    else:
        expected_validation_contract = {"enabled": False}
    base_scientific = build_strategy_param_scientific_contract(
        project_root, active_profile_id
    )
    base_scientific_fp = base_scientific["scientific_fingerprint_sha256"]
    check(
        "strategy_param_scientific_contract_matches_active_profile_owners",
        True,
        base_scientific["references"]["profile_id"] == active_profile_id
        and base_scientific["references"]["parameter_set_identity"]
        == active_param_profile.parameter_set_identity
        and base_scientific["scientific_contract"]["schedule"]["train_window_months"]
        == int(active_training_policy["train_window_months"])
        and base_scientific["scientific_contract"]["schedule"]["oos_horizon_months"]
        == int(active_training_policy["oos_horizon_months"])
        and base_scientific["scientific_contract"]["fitting"]["seed_count"]
        == int(active_fitting_protocol.seed_count)
        and base_scientific["scientific_contract"]["fitting"]["trials_per_seed"]
        == int(active_fitting_protocol.trials_per_seed)
        and base_scientific["scientific_contract"]["validation"]
        == expected_validation_contract,
    )

    with patch.object(training_policy_settings, "MIN_ANNUAL_TRADES", float(training_policy_settings.MIN_ANNUAL_TRADES) + 1.0):
        gate_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_tracks_hard_gate_change", True, gate_fp != base_scientific_fp)

    with patch.object(training_policy_settings, "OUTER_ROLLING_TRAIN_WINDOW_MONTHS", int(training_policy_settings.OUTER_ROLLING_TRAIN_WINDOW_MONTHS) + 12):
        schedule_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_tracks_schedule_change", True, schedule_fp != base_scientific_fp)

    with patch.object(training_policy_settings, "SCORE_MDD_POWER", float(training_policy_settings.SCORE_MDD_POWER) + 0.25):
        score_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_tracks_score_change", True, score_fp != base_scientific_fp)

    with patch.object(training_policy_settings, "OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE", 1):
        selector_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_tracks_selector_change", True, selector_fp != base_scientific_fp)

    with patch.object(training_policy_settings, "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED", not bool(training_policy_settings.OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED)):
        local_min_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_tracks_local_min_change", True, local_min_fp != base_scientific_fp)

    min_fee_spec = execution_policy_settings.EXECUTION_POLICY_PARAM_SPECS["min_fee"]
    with patch.dict(min_fee_spec, {"default": float(min_fee_spec["default"]) + 1.0}):
        execution_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_tracks_execution_economics_change", True, execution_fp != base_scientific_fp)

    vol_long_spec = breakout_schema.BREAKOUT_PARAM_SPECS["vol_long_len"]
    with patch.dict(vol_long_spec, {"default": int(vol_long_spec["default"]) + 1}):
        schema_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_tracks_parameter_schema_default_change", True, schema_fp != base_scientific_fp)

    with patch.object(display_policy_settings, "SYSTEM_SCORE_DISPLAY_MULTIPLIER", float(display_policy_settings.SYSTEM_SCORE_DISPLAY_MULTIPLIER) * 2.0):
        display_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_ignores_display_only_change", base_scientific_fp, display_fp)

    from copy import deepcopy
    import core.strategy_param_policy as strategy_param_policy_module
    renamed_profiles = deepcopy(strategy_param_policy_module.STRATEGY_PARAM_PROFILES)
    renamed_profiles[active_profile_id]["label"] = "Renamed Display Label"
    renamed_profiles[active_profile_id]["parameter_set_identity"] = "PARAM-RENAMED-ONLY"
    with patch.object(strategy_param_policy_module, "STRATEGY_PARAM_PROFILES", renamed_profiles):
        renamed_contract = build_strategy_param_scientific_contract(project_root, active_profile_id)
    check("strategy_param_scientific_fingerprint_excludes_human_and_artifact_aliases", base_scientific_fp, renamed_contract["scientific_fingerprint_sha256"])
    check("strategy_param_scientific_contract_keeps_aliases_as_references", "PARAM-RENAMED-ONLY", renamed_contract["references"]["parameter_set_identity"])

    with patch.object(training_performance_settings, "OPTIMIZER_RESOURCE_SAMPLE_INTERVAL_SEC", float(training_performance_settings.OPTIMIZER_RESOURCE_SAMPLE_INTERVAL_SEC) + 1.0):
        performance_fp = build_strategy_param_scientific_contract(project_root, active_profile_id)["scientific_fingerprint_sha256"]
    check("strategy_param_scientific_fingerprint_ignores_performance_only_change", base_scientific_fp, performance_fp)

    registered_profile_ids = tuple(
        profile_id
        for profile_id, raw_profile in strategy_param_policy_module.STRATEGY_PARAM_PROFILES.items()
        if str(dict(raw_profile).get("registered_scientific_fingerprint_sha256") or "").strip()
    )
    registered_identity_rows = [
        build_registered_strategy_param_scientific_identity(project_root, profile_id)
        for profile_id in registered_profile_ids
    ]
    check(
        "registered_strategy_param_profiles_match_resolved_scientific_contracts",
        True,
        bool(registered_identity_rows)
        and all(bool(row.get("registration_matches", False)) for row in registered_identity_rows),
    )
    with patch.object(
        training_policy_settings,
        "MIN_ANNUAL_TRADES",
        float(training_policy_settings.MIN_ANNUAL_TRADES) + 1.0,
    ):
        registered_drift_rows = [
            build_registered_strategy_param_scientific_identity(project_root, profile_id)
            for profile_id in registered_profile_ids
        ]
    check(
        "registered_strategy_param_identity_detects_common_scientific_drift",
        True,
        bool(registered_drift_rows)
        and all(not bool(row.get("registration_matches", True)) for row in registered_drift_rows),
    )

    # Round 4 convergence: current non-validation Strategy Compare arms resolve
    # their parameter semantics through registered profiles, while C84/P5 stays
    # on the historical validation path and is intentionally not migrated here.
    from core.strategy_compare_policy import get_strategy_comparison_settings
    from core.strategy_param_policy import resolve_canonical_strategy_param_training_profile
    from services.optimizer.strategy_param_service import inspect_strategy_parameter_artifact

    current_compare_settings = get_strategy_comparison_settings()
    canonical_profile_rows = []
    fixed_multiseed_profile_rows = []
    for arm in current_compare_settings.enabled_arms:
        source = current_compare_settings.parameter_sources[arm.param_source]
        builder_type = "" if source.builder is None else str(source.builder.builder_type)
        if builder_type == "canonical_optimizer_strategy_params":
            profile = resolve_canonical_strategy_param_training_profile(
                str(source.canonical_family or ""), str(arm.param_policy or "")
            )
            canonical_profile_rows.append((arm, source, profile))
        elif builder_type == "optimizer_fixed_multiseed_strategy_params":
            profile_id = str(dict(source.builder.options).get("training_profile_id") or "")
            fixed_multiseed_profile_rows.append(
                (arm, get_strategy_param_training_profile(profile_id))
            )
    check(
        "strategy_compare_non_validation_canonical_arms_resolve_registered_profiles",
        True,
        bool(canonical_profile_rows)
        and all(
            profile is not None
            and not profile.validation_enabled
            and profile.registered_scientific_fingerprint_sha256 is not None
            and profile.filter_profile == str(source.canonical_family)
            and profile.parameter_policy == str(arm.param_policy)
            for arm, source, profile in canonical_profile_rows
        ),
    )
    registered_fixed_multiseed_profiles = tuple(
        get_strategy_param_training_profile(profile_id)
        for profile_id in (
            "production_multiseed_current",
            "production_multiseed_8x300_compare",
            "production_multiseed_4x1000_compare",
            "trading_aligned_8x1000_current",
        )
    )
    check(
        "strategy_compare_registered_fixed_multiseed_profiles_remain_bound_after_round6_switch",
        True,
        all(
            not profile.validation_enabled
            and profile.registered_scientific_fingerprint_sha256 is not None
            and profile.parameter_policy == "base-finalists-agree"
            for profile in registered_fixed_multiseed_profiles
        ),
    )
    fixed_profile_ids = tuple(profile.profile_id for profile in registered_fixed_multiseed_profiles)
    fixed_profile_artifact_paths = tuple(
        fixed_multiseed_strategy_param_artifact_path(
            project_root,
            training_profile_id=profile_id,
            policy="base-finalists-agree",
        )
        for profile_id in fixed_profile_ids
    )
    fixed_profile_robustness_paths = tuple(
        fixed_multiseed_robustness_strategy_param_artifact_path(
            project_root,
            training_profile_id=profile_id,
            benchmark_id=str(benchmark_snapshot["benchmark_id"]),
            replication_seed=replication_seed,
            policy="base-finalists-agree",
        )
        for profile_id in fixed_profile_ids
    )
    check(
        "strategy_compare_fixed_multiseed_profiles_have_isolated_artifact_namespaces",
        True,
        len(fixed_profile_ids) == 4
        and len(set(fixed_profile_artifact_paths)) == len(fixed_profile_ids)
        and len(set(fixed_profile_robustness_paths)) == len(fixed_profile_ids)
        and all(
            get_strategy_param_training_profile(profile_id).artifact_slug
            for profile_id in fixed_profile_ids
        ),
    )
    current_builder_types = {
        str(current_compare_settings.parameter_sources[arm.param_source].builder.builder_type)
        for arm in current_compare_settings.enabled_arms
        if current_compare_settings.parameter_sources[arm.param_source].builder is not None
    }
    current_validation_parameter_sets = {
        str(dict(current_compare_settings.parameter_sources[arm.param_source].builder.options).get("parameter_set") or "")
        for arm in current_compare_settings.enabled_arms
        if current_compare_settings.parameter_sources[arm.param_source].builder is not None
        and current_compare_settings.parameter_sources[arm.param_source].builder.builder_type
            == "optimizer_validation_budget_refit_strategy_params"
    }
    current_fixed_multiseed_profiles = {
        str(dict(current_compare_settings.parameter_sources[arm.param_source].builder.options).get("training_profile_id") or "")
        for arm in current_compare_settings.enabled_arms
        if current_compare_settings.parameter_sources[arm.param_source].builder is not None
        and current_compare_settings.parameter_sources[arm.param_source].builder.builder_type
            == "optimizer_fixed_multiseed_strategy_params"
    }
    current_raw_profile_rows = []
    for arm in current_compare_settings.enabled_arms:
        source = current_compare_settings.parameter_sources[arm.param_source]
        builder = source.builder
        if builder is None or builder.builder_type != "optimizer_raw_price_strategy_params":
            continue
        profile_id = str(dict(builder.options).get("training_profile_id") or "")
        profile = get_strategy_param_training_profile(profile_id)
        protocol = get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)
        current_raw_profile_rows.append((arm, source, profile, protocol))
    current_raw_profiles = {profile.profile_id for _arm, _source, profile, _protocol in current_raw_profile_rows}
    current_arm_ids = tuple(arm.arm_id for arm in current_compare_settings.enabled_arms)
    current_contrast_ids = tuple(contrast.contrast_id for contrast in current_compare_settings.enabled_contrasts)
    required_reference_ids = {"C58", "C61", "C85"}
    check(
        "strategy_compare_current_suite_preserves_required_references_and_derives_raw_profiles",
        True,
        required_reference_ids.issubset(set(current_arm_ids))
        and all(
            contrast.left in set(current_arm_ids) and contrast.right in set(current_arm_ids)
            for contrast in current_compare_settings.enabled_contrasts
        )
        and len(current_contrast_ids) == len(set(current_contrast_ids))
        and "optimizer_validation_budget_refit_strategy_params" not in current_builder_types
        and "optimizer_fixed_multiseed_strategy_params" in current_builder_types
        and "optimizer_raw_price_strategy_params" in current_builder_types
        and not current_validation_parameter_sets
        and current_fixed_multiseed_profiles == {"production_multiseed_current"}
        and active_param_profile.profile_id in current_raw_profiles
        and len(current_raw_profiles) == len(current_raw_profile_rows)
        and len({str(source.path_template or "") for _arm, source, _profile, _protocol in current_raw_profile_rows})
            == len(current_raw_profile_rows)
        and all(
            arm.data_source == "raw_unadjusted_controlled"
            and profile.price_basis == "raw_unadjusted"
            and profile.filter_profile == "full"
            and not profile.validation_enabled
            and profile.parameter_policy == arm.param_policy
            and bool(profile.artifact_slug)
            and str(profile.artifact_slug) in str(source.path_template or "")
            for arm, source, profile, _protocol in current_raw_profile_rows
        )
        and any(not protocol.enabled for _arm, _source, _profile, protocol in current_raw_profile_rows)
        and any(protocol.enabled for _arm, _source, _profile, protocol in current_raw_profile_rows)
        and current_compare_settings.arms["C58"].data_source == "canonical"
        and current_compare_settings.arms["C61"].data_source == "canonical"
        and current_compare_settings.arms["C85"].data_source == "canonical"
        and active_param_profile.price_basis == "raw_unadjusted"
        and active_param_profile.filter_profile == "full"
        and not active_param_profile.validation_enabled
        # Historical P13 remains registered and reproducible; it is not current.
        and trading_aligned_8x1000_param_profile.profile_id
            == "trading_aligned_8x1000_current"
        and trading_aligned_8x1000_param_profile.parameter_set_identity == "PARAM-P13"
        and not trading_aligned_8x1000_param_profile.validation_enabled,
    )
    raw_work_plans = [
        strategy_param_application._build_plan(active_param_profile, work)
        for work in strategy_param_application.WORK_TYPES
    ]
    check(
        "active_strategy_param_profile_opens_all_four_work_types_from_one_profile",
        True,
        len(raw_work_plans) == 4
        and all(plan.get("status") in {"READY", "PREPARABLE"} for plan in raw_work_plans)
        and len(raw_work_plans[2].get("rows") or ()) == 4
        and len(raw_work_plans[3].get("rows") or ()) == 4
        and "Robustness下一輪再開" not in strategy_param_app_source
        and "RAW_PRICE_BASELINE_FIRST_ROBUSTNESS_NOT_ENABLED" not in strategy_param_app_source,
    )
    from services.research.strategy_comparison import _display_ordered_strategy_arms
    legacy_unsorted_settings = get_strategy_comparison_settings(
        "non_dl_production_multiseed_rolling"
    )
    check(
        "strategy_compare_human_tables_sort_c_ids_numeric_ascending",
        ("C58", "C61", "C62", "C85", "C86", "C87"),
        tuple(arm.arm_id for arm in _display_ordered_strategy_arms(legacy_unsorted_settings)),
    )
    min_profile = resolve_canonical_strategy_param_training_profile(
        "min", "base-finalist-best"
    )
    full_best_profile = resolve_canonical_strategy_param_training_profile(
        "full", "base-finalist-best"
    )
    full_agree_profile = resolve_canonical_strategy_param_training_profile(
        "full", "base-finalists-agree"
    )
    check(
        "canonical_single_seed_policy_outputs_have_distinct_registered_scientific_identities",
        True,
        all(profile is not None for profile in (min_profile, full_best_profile, full_agree_profile))
        and len({
            min_profile.registered_scientific_fingerprint_sha256,
            full_best_profile.registered_scientific_fingerprint_sha256,
            full_agree_profile.registered_scientific_fingerprint_sha256,
        }) == 3,
    )
    with patch.object(
        training_policy_settings,
        "MIN_ANNUAL_TRADES",
        float(training_policy_settings.MIN_ANNUAL_TRADES) + 1.0,
    ):
        drift_state = inspect_strategy_parameter_artifact(
            project_root,
            family="min",
            evaluation_mode="rolling",
            policy="base-finalist-best",
            comparison_end_date="2021-01-01",
            dataset=min_profile.dataset,
        )
    check(
        "canonical_strategy_param_registered_drift_blocks_before_artifact_reuse",
        ("BLOCKED", "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT"),
        (drift_state.get("action"), drift_state.get("status")),
    )

    # Round 2 runtime injection: once a producer starts a fitting invocation,
    # inner consumers must keep using that resolved contract even if mutable
    # common config is changed afterwards.  Generic sessions without a contract
    # retain the legacy config-backed fallback.
    from core.scientific_runtime_context import scientific_runtime_context
    import core.config as core_config_module
    import core.portfolio_attribution as portfolio_attribution_module
    import strategies.breakout.search_space as breakout_search_module
    from services.optimizer.outer_rolling_policy import (
        BASE_FINALISTS_AGREE_POLICY_NAME,
        _finalists_agree_policy_config,
    )
    from services.optimizer.scientific_runtime import (
        build_optimizer_progress_display_contract,
        get_session_scientific_section,
        resolve_session_fixed_tp_percent,
        resolve_session_local_min_review_enabled,
    )

    from services.optimizer.outer_rolling_progress import (
        format_optimizer_seed_ensemble_progress_header,
        render_optimizer_seed_progress_line,
    )

    progress_display_contract = build_optimizer_progress_display_contract(base_scientific)
    expected_selector = str(
        base_scientific["scientific_contract"]["strategy"].get("parameter_policy") or ""
    )
    expected_show_seed_min_agree = not expected_selector.replace("_", "-").lower().endswith(
        "-finalists-agree"
    )
    local_min_contract = dict(
        base_scientific["scientific_contract"]["selection_and_anti_overfit"].get(
            "local_min_review"
        )
        or {}
    )
    filter_profile = str(
        base_scientific["scientific_contract"]["strategy"].get("filter_profile") or ""
    ).lower()
    expected_show_local_min = bool(
        local_min_contract.get(
            "full_mode_enabled" if filter_profile == "full" else "default_enabled",
            local_min_contract.get("default_enabled", False),
        )
    )
    check(
        "optimizer_progress_display_contract_is_derived_from_scientific_contract",
        True,
        progress_display_contract.get("selector") == expected_selector
        and bool(progress_display_contract.get("show_seed_min_agree"))
        == expected_show_seed_min_agree
        and bool(progress_display_contract.get("show_local_min")) == expected_show_local_min,
    )
    progress_header = format_optimizer_seed_ensemble_progress_header(
        folds=1,
        seeds=max(1, int(active_fitting_protocol.seed_count)),
        min_agree=(
            int(active_fitting_protocol.min_agree)
            if progress_display_contract.get("show_seed_min_agree")
            else None
        ),
        parallel_workers=1,
        selector=progress_display_contract.get("selector"),
        backend="synthetic",
    )
    check(
        "optimizer_progress_header_hides_inapplicable_seed_agreement",
        expected_show_seed_min_agree,
        "seed_min_agree=" in progress_header,
    )
    progress_line = render_optimizer_seed_progress_line(
        context={
            "fold_idx": 1,
            "fold_count": 1,
            "seed_index": 1,
            "seed_count": max(1, int(active_fitting_protocol.seed_count)),
            "selection_start": "2020-01-01",
            "selection_end": "2020-12-31",
            "oos_period": "2021-01-01~2021-12-31",
            "progress_display_contract": progress_display_contract,
        },
        progress={
            "stage": "DONE",
            "best_base_score": 1.0,
            "best_local_min_score": 2.0,
        },
    )
    check(
        "optimizer_progress_line_hides_inactive_local_min_metric",
        expected_show_local_min,
        "best_local_min=" in progress_line,
    )

    injected_session = SimpleNamespace(strategy_param_scientific_contract=deepcopy(base_scientific))
    injected_score = get_session_scientific_section(injected_session, "score") or {}
    injected_selection = get_session_scientific_section(injected_session, "selection_and_anti_overfit") or {}
    injected_parameter_generation = get_session_scientific_section(injected_session, "parameter_generation") or {}

    frozen_buy_sort = injected_score.get("BUY_SORT_METHOD")
    frozen_ev_method = injected_score.get("EV_CALC_METHOD")
    with patch.object(training_policy_settings, "BUY_SORT_METHOD", "__MUTATED_BUY_SORT__"), patch.object(
        training_policy_settings, "EV_CALC_METHOD", "__MUTATED_EV__"
    ):
        with scientific_runtime_context(base_scientific):
            runtime_buy_sort = core_config_module.get_buy_sort_method()
            runtime_ev_method = core_config_module.get_ev_calc_method()
    check("strategy_param_runtime_context_freezes_buy_sort_method", frozen_buy_sort, runtime_buy_sort)
    check("strategy_param_runtime_context_freezes_ev_calc_method", frozen_ev_method, runtime_ev_method)

    frozen_score_value = calc_portfolio_score(
        35.0,
        -12.0,
        70.0,
        0.8,
        annual_return_pct=18.0,
        trade_win_rate_pct=55.0,
        min_full_year_return_pct=5.0,
        min_month_return_pct=-2.0,
        min_quarter_return_pct=-3.0,
        total_r=12.0,
        median_r=0.6,
        score_policy=injected_score,
        threshold_policy=get_session_scientific_section(injected_session, "hard_gates"),
    )
    with patch.object(training_policy_settings, "SCORE_MDD_POWER", float(training_policy_settings.SCORE_MDD_POWER) + 0.75), patch.object(
        training_policy_settings, "SCORE_WIN_RATE_TARGET", float(training_policy_settings.SCORE_WIN_RATE_TARGET) + 11.0
    ):
        runtime_score_value = calc_portfolio_score(
            35.0,
            -12.0,
            70.0,
            0.8,
            annual_return_pct=18.0,
            trade_win_rate_pct=55.0,
            min_full_year_return_pct=5.0,
            min_month_return_pct=-2.0,
            min_quarter_return_pct=-3.0,
            total_r=12.0,
            median_r=0.6,
            score_policy=injected_score,
            threshold_policy=get_session_scientific_section(injected_session, "hard_gates"),
        )
    check("strategy_param_session_freezes_score_policy", frozen_score_value, runtime_score_value)

    # Deep validation/replay callers intentionally omit explicit score_policy.
    # They must still consume the bound scientific score contract rather than
    # falling back to mutable global config.  Quarter-off is a useful regression
    # because the global default remains ON and can hard-zero the same sample.
    quarter_off_scientific = deepcopy(base_scientific)
    quarter_off_score = dict(quarter_off_scientific["scientific_contract"]["score"])
    quarter_off_score["SCORE_MIN_QUARTER_RETURN_AMP_ENABLED"] = False
    quarter_off_scientific["scientific_contract"]["score"] = quarter_off_score
    quarter_floor = float(quarter_off_score["SCORE_MIN_QUARTER_RETURN_FLOOR"])
    score_kwargs = {
        "sys_ret": 30.0,
        "sys_mdd": -10.0,
        "m_win_rate": 95.0,
        "r_sq": 0.9,
        "annual_return_pct": 20.0,
        "trade_win_rate_pct": 70.0,
        "min_full_year_return_pct": 10.0,
        "min_month_return_pct": 10.0,
        "min_quarter_return_pct": quarter_floor,
        "total_r": 20.0,
        "median_r": 0.5,
    }
    explicit_quarter_off_score = calc_portfolio_score(
        **score_kwargs,
        score_policy=quarter_off_score,
        threshold_policy=quarter_off_scientific["scientific_contract"]["hard_gates"],
    )
    with scientific_runtime_context(quarter_off_scientific):
        implicit_quarter_off_score = calc_portfolio_score(**score_kwargs)
    check(
        "strategy_param_runtime_context_applies_score_policy_to_deep_replay",
        True,
        explicit_quarter_off_score > 0.0
        and abs(implicit_quarter_off_score - explicit_quarter_off_score) < 1e-12,
    )

    frozen_local_enabled = bool(
        (injected_selection.get("local_min_review") or {}).get("default_enabled", False)
    )
    with patch.object(
        training_policy_settings,
        "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED",
        not bool(training_policy_settings.OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED),
    ):
        runtime_local_enabled = resolve_session_local_min_review_enabled(injected_session)
    check("strategy_param_session_freezes_local_min_enablement", frozen_local_enabled, runtime_local_enabled)

    frozen_fixed_tp = float(injected_selection.get("fixed_tp_percent", 0.0) or 0.0)
    with patch.object(
        training_policy_settings,
        "OPTIMIZER_FIXED_TP_PERCENT",
        frozen_fixed_tp + 0.123,
    ):
        runtime_fixed_tp = resolve_session_fixed_tp_percent(
            injected_session, training_policy_settings.OPTIMIZER_FIXED_TP_PERCENT
        )
    check("strategy_param_session_freezes_fixed_tp_percent", frozen_fixed_tp, runtime_fixed_tp)

    frozen_base_agree = (injected_selection.get("finalists_agree") or {}).get("base_min_agree")
    with patch.object(training_policy_settings, "OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE", 1):
        runtime_base_agree = _finalists_agree_policy_config(
            BASE_FINALISTS_AGREE_POLICY_NAME, session=injected_session
        ).get("min_agree_requested")
    check("strategy_param_session_freezes_finalists_agree_request", frozen_base_agree, runtime_base_agree)

    frozen_search_space = dict(injected_parameter_generation.get("search_space") or {})
    original_atr_low = float(frozen_search_space["atr_len"]["low"])
    with patch.dict(breakout_search_module.BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"], {"low": original_atr_low + 7}):
        _step, _kind, runtime_atr_low, _high = breakout_search_module.resolve_breakout_neighbor_spec(
            "atr_len", session=injected_session
        )
    check("strategy_param_session_freezes_local_min_neighbor_search_space", original_atr_low, float(runtime_atr_low))

    class _ContractTrial:
        def suggest_categorical(self, _name, choices):
            return list(choices)[0]
        def suggest_int(self, _name, low, _high, step=1):
            return int(low)
        def suggest_float(self, _name, low, _high, step=None):
            return float(low)
        def set_user_attr(self, *_args, **_kwargs):
            return None

    class _ContractSession(SimpleNamespace):
        optimizer_fixed_tp_percent = frozen_fixed_tp
        def has_fixed_strategy_param(self, _field_name):
            return False
        def get_fixed_strategy_param(self, _field_name, default=None):
            return default
        def resolve_optimizer_tp_percent(self, _trial, fixed_tp_percent):
            return float(fixed_tp_percent)

    from core.strategy_params import strategy_params_to_dict
    contract_session = _ContractSession(strategy_param_scientific_contract=deepcopy(base_scientific))
    with patch.dict(
        breakout_search_module.BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"],
        {"low": int(original_atr_low) + 9},
    ):
        contract_trial_params = strategy_params_to_dict(
            breakout_search_module.build_trial_params(contract_session, _ContractTrial())
        )
    check(
        "strategy_param_session_freezes_trial_parameter_generation",
        int(original_atr_low),
        int(contract_trial_params["atr_len"]),
    )

    frozen_dependency = dict(injected_selection.get("dominant_year_dependency") or {})
    with scientific_runtime_context(base_scientific):
        runtime_dependency_thresholds = portfolio_attribution_module._threshold_snapshot()
    check(
        "strategy_param_runtime_context_freezes_dominant_year_thresholds",
        {
            key: frozen_dependency[key]
            for key in (
                "high_positive_pnl_share",
                "narrow_positive_trade_count",
                "narrow_positive_symbol_count",
                "top_trade_outlier_pnl_share",
            )
        },
        runtime_dependency_thresholds,
    )

    summary["checks"] = len(results)
    return results, summary
