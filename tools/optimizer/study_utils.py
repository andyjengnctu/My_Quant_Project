import os
from decimal import Decimal
from typing import Callable

from core.config import V16StrategyParams
from core.dataset_profiles import (
    DATASET_PROFILE_FULL,
    DEFAULT_DATASET_PROFILE,
    normalize_dataset_profile_key,
)
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.runtime_utils import is_interactive_stdin, parse_int_strict
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE

OPTIMIZER_TRIALS_ENV_VAR = "V16_OPTIMIZER_TRIALS"
OPTIMIZER_SEED_ENV_VAR = "V16_OPTIMIZER_SEED"
DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE = 50000
DEFAULT_OPTIMIZER_TRIALS_NON_INTERACTIVE = 0
INVALID_TRIAL_VALUE = -9999.0
MIN_QUALIFIED_TRIAL_VALUE = -9000.0
OPTIMIZER_TP_PERCENT_SEARCH_SPEC = {"low": 0.0, "high": 0.6, "step": 0.01}
OBJECTIVE_MODE_LEGACY_BASE_SCORE = "legacy_base_score"
OBJECTIVE_MODE_SPLIT_TEST_ROMD = "split_test_romd"



OPTIMIZER_MENU_ACTION_TRAIN = "train"
OPTIMIZER_MENU_ACTION_EXPORT_BEST = "export_best"
OPTIMIZER_MENU_ACTION_PROMOTE_CHAMPION = "promote_champion"


def _parse_optimizer_run_request_raw(raw_value: str, *, source_label: str):
    normalized = str(raw_value or "").strip()
    if normalized == "":
        return {
            "n_trials": int(DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE),
            "action": OPTIMIZER_MENU_ACTION_TRAIN,
            "source": source_label,
        }
    if normalized.lower() == "p":
        return {
            "n_trials": 0,
            "action": OPTIMIZER_MENU_ACTION_PROMOTE_CHAMPION,
            "source": source_label,
        }
    trial_count = parse_int_strict(normalized, "訓練次數", min_value=0)
    return {
        "n_trials": int(trial_count),
        "action": OPTIMIZER_MENU_ACTION_EXPORT_BEST if int(trial_count) == 0 else OPTIMIZER_MENU_ACTION_TRAIN,
        "source": source_label,
    }


def resolve_optimizer_run_request(environ):
    env_value = str(environ.get(OPTIMIZER_TRIALS_ENV_VAR, "")).strip()
    if env_value != "":
        return _parse_optimizer_run_request_raw(env_value, source_label=f"ENV:{OPTIMIZER_TRIALS_ENV_VAR}")

    if is_interactive_stdin():
        prompt = (
            f"👉 Optimizer 動作（預設 {DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE}）："
            f"Enter=訓練 {DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE} 次，"
            "數字=訓練指定次數，0=匯出 run_best，P=挑戰 Champion: "
        )
        raw_input = input(prompt)
        return _parse_optimizer_run_request_raw(raw_input, source_label="UI/MENU")

    return {
        "n_trials": int(DEFAULT_OPTIMIZER_TRIALS_NON_INTERACTIVE),
        "action": OPTIMIZER_MENU_ACTION_EXPORT_BEST,
        "source": "DEFAULT/NON_INTERACTIVE",
    }


def _resolve_decimal_places_from_step(step_value):
    normalized_step = Decimal(str(step_value)).normalize()
    exponent = normalized_step.as_tuple().exponent
    return 0 if exponent >= 0 else -exponent


_OPTIMIZER_STEP_DECIMAL_PLACES = {
    field_name: _resolve_decimal_places_from_step(spec["step"])
    for field_name, spec in {
        **BREAKOUT_OPTIMIZER_SEARCH_SPACE,
        "tp_percent": {"kind": "float", **OPTIMIZER_TP_PERCENT_SEARCH_SPEC},
    }.items()
    if spec.get("kind") == "float" and spec.get("step") is not None
}


def _canonicalize_step_float_for_export(field_name, raw_value):
    decimal_places = _OPTIMIZER_STEP_DECIMAL_PLACES.get(field_name)
    if decimal_places is None:
        return float(raw_value)

    quantizer = Decimal("1").scaleb(-decimal_places)
    return float(Decimal(str(float(raw_value))).quantize(quantizer))


def _canonicalize_best_params_for_export(resolved_params):
    canonicalized = dict(resolved_params)
    for field_name, field_value in canonicalized.items():
        if isinstance(field_value, bool) or not isinstance(field_value, float):
            continue
        canonicalized[field_name] = _canonicalize_step_float_for_export(field_name, field_value)
    return canonicalized


def is_qualified_trial_value(value):
    return value is not None and float(value) > MIN_QUALIFIED_TRIAL_VALUE


def resolve_optimizer_trial_count(environ):
    request = resolve_optimizer_run_request(environ)
    return int(request["n_trials"]), str(request["source"])


def resolve_optimizer_seed(environ):
    env_value = str(environ.get(OPTIMIZER_SEED_ENV_VAR, "")).strip()
    if env_value == "":
        return None, "ENV:UNSET"
    seed = parse_int_strict(env_value, f"環境變數 {OPTIMIZER_SEED_ENV_VAR}", min_value=0)
    return seed, f"ENV:{OPTIMIZER_SEED_ENV_VAR}"


def build_optimizer_db_file_path(dataset_profile_key, models_dir):
    normalized_key = normalize_dataset_profile_key(dataset_profile_key, default=DEFAULT_DATASET_PROFILE)
    suffix = "" if normalized_key == DATASET_PROFILE_FULL else f"_{normalized_key}"
    return os.path.join(models_dir, f"portfolio_ai_10pos_overnight{suffix}.db")


def validate_optimizer_param_overrides(param_mapping):
    if not isinstance(param_mapping, dict):
        raise TypeError(f"optimizer override 必須是 dict，收到 {type(param_mapping).__name__}")

    base_payload = params_to_json_dict(V16StrategyParams())
    merged_payload = dict(base_payload)
    merged_payload.update(param_mapping)
    validated = build_params_from_mapping(merged_payload)
    return {key: getattr(validated, key) for key in param_mapping}


def resolve_optimizer_tp_percent(trial, fixed_tp_percent):
    if fixed_tp_percent is None:
        return trial.suggest_float(
            "tp_percent",
            OPTIMIZER_TP_PERCENT_SEARCH_SPEC["low"],
            OPTIMIZER_TP_PERCENT_SEARCH_SPEC["high"],
            step=OPTIMIZER_TP_PERCENT_SEARCH_SPEC["step"],
        )

    validated_params = validate_optimizer_param_overrides({"tp_percent": float(fixed_tp_percent)})
    resolved_tp = float(validated_params["tp_percent"])
    trial.set_user_attr("fixed_tp_percent", resolved_tp)
    return resolved_tp


def build_optimizer_trial_params(param_mapping, user_attrs=None, fixed_tp_percent=None):
    resolved_params = dict(param_mapping)
    if "tp_percent" in resolved_params:
        resolved_params["tp_percent"] = float(resolved_params["tp_percent"])
    else:
        attr_tp = None if user_attrs is None else user_attrs.get("fixed_tp_percent")
        chosen_tp = attr_tp if attr_tp is not None else fixed_tp_percent
        if chosen_tp is None:
            raise ValueError("最佳 trial 缺少 tp_percent，且目前未設定 OPTIMIZER_FIXED_TP_PERCENT，無法還原完整參數。")
        resolved_params["tp_percent"] = float(chosen_tp)

    validated_params = validate_optimizer_param_overrides(resolved_params)
    return {key: validated_params[key] for key in resolved_params}


def build_best_params_payload_from_trial(best_trial, fixed_tp_percent=None):
    resolved_params = build_optimizer_trial_params(best_trial.params, best_trial.user_attrs, fixed_tp_percent=fixed_tp_percent)
    canonicalized_params = _canonicalize_best_params_for_export(resolved_params)
    base_payload = params_to_json_dict(V16StrategyParams())
    base_payload.update(canonicalized_params)
    return params_to_json_dict(build_params_from_mapping(base_payload))


def list_completed_study_trials(study):
    return [trial for trial in study.trials if trial.value is not None]




def _trial_matches_objective_mode(trial, objective_mode: str) -> bool:
    expected = str(objective_mode or "").strip()
    actual = str(trial.user_attrs.get("objective_mode", "")).strip()
    if actual:
        return actual == expected
    return expected == OBJECTIVE_MODE_LEGACY_BASE_SCORE

def _wf_attr_float(trial, key: str, default: float = float("-inf")) -> float:
    value = trial.user_attrs.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _resolve_legacy_best_trial_or_none(study, *, objective_mode=OBJECTIVE_MODE_LEGACY_BASE_SCORE):
    completed_trials = [trial for trial in list_completed_study_trials(study) if is_qualified_trial_value(trial.value) and _trial_matches_objective_mode(trial, objective_mode)]
    if not completed_trials:
        return None
    return max(completed_trials, key=lambda trial: (float(trial.value), -int(trial.number)))


def resolve_best_completed_trial_or_none(study, *, objective_mode=OBJECTIVE_MODE_LEGACY_BASE_SCORE):
    mode = str(objective_mode or OBJECTIVE_MODE_LEGACY_BASE_SCORE).strip()
    return _resolve_legacy_best_trial_or_none(study, objective_mode=mode)


def build_best_completed_trial_resolver(objective_mode: str) -> Callable:
    def _resolver(study):
        return resolve_best_completed_trial_or_none(study, objective_mode=objective_mode)

    return _resolver


def get_best_completed_trial_or_none(study):
    return resolve_best_completed_trial_or_none(study, objective_mode=OBJECTIVE_MODE_LEGACY_BASE_SCORE)
