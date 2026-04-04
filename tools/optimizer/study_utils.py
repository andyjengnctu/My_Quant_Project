import os

from core.config import V16StrategyParams
from core.dataset_profiles import (
    DATASET_PROFILE_FULL,
    DEFAULT_DATASET_PROFILE,
    normalize_dataset_profile_key,
)
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.runtime_utils import is_interactive_stdin, parse_int_strict, safe_prompt_int

OPTIMIZER_TRIALS_ENV_VAR = "V16_OPTIMIZER_TRIALS"
OPTIMIZER_SEED_ENV_VAR = "V16_OPTIMIZER_SEED"
DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE = 50000
DEFAULT_OPTIMIZER_TRIALS_NON_INTERACTIVE = 0
INVALID_TRIAL_VALUE = -9999.0
MIN_QUALIFIED_TRIAL_VALUE = -9000.0


def is_qualified_trial_value(value):
    return value is not None and float(value) > MIN_QUALIFIED_TRIAL_VALUE


def resolve_optimizer_trial_count(environ):
    env_value = str(environ.get(OPTIMIZER_TRIALS_ENV_VAR, "")).strip()
    if env_value != "":
        return parse_int_strict(env_value, f"環境變數 {OPTIMIZER_TRIALS_ENV_VAR}", min_value=0), f"ENV:{OPTIMIZER_TRIALS_ENV_VAR}"

    default_trials = DEFAULT_OPTIMIZER_TRIALS_INTERACTIVE if is_interactive_stdin() else DEFAULT_OPTIMIZER_TRIALS_NON_INTERACTIVE
    prompt_default = str(default_trials)
    user_input = safe_prompt_int(
        f"👉 請輸入訓練次數 (預設 {prompt_default}，輸入 0 則直接提取匯出參數): ",
        prompt_default,
        "訓練次數",
        min_value=0,
    )
    return user_input, "UI/DEFAULT"


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
        return trial.suggest_float("tp_percent", 0.0, 0.6, step=0.01)

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
    base_payload = params_to_json_dict(V16StrategyParams())
    base_payload.update(resolved_params)
    return params_to_json_dict(build_params_from_mapping(base_payload))


def list_completed_study_trials(study):
    return [trial for trial in study.trials if trial.value is not None]


def get_best_completed_trial_or_none(study):
    completed_trials = list_completed_study_trials(study)
    if not completed_trials:
        return None
    try:
        return study.best_trial
    except ValueError:
        return None
