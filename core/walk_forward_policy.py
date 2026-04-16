import json
import os
from typing import Mapping, Optional

WALK_FORWARD_POLICY_PATH_ENV_VAR = "V16_WALK_FORWARD_POLICY_PATH"
DEFAULT_WALK_FORWARD_POLICY = {
    "train_start_year": 2012,
    "min_train_years": 8,
    "test_window_months": 6,
    "regime_up_threshold_pct": 8.0,
    "regime_down_threshold_pct": -8.0,
}


def _resolve_policy_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(WALK_FORWARD_POLICY_PATH_ENV_VAR, "")).strip()
    if override:
        if os.path.isabs(override):
            return os.path.abspath(override)
        return os.path.abspath(os.path.join(project_root, override))
    return os.path.abspath(os.path.join(project_root, "config", "walk_forward_policy.json"))


def _coerce_int(data: dict, key: str, minimum: int) -> int:
    value = int(data.get(key, DEFAULT_WALK_FORWARD_POLICY[key]))
    if value < minimum:
        raise ValueError(f"walk-forward policy: {key} 必須 >= {minimum}，收到: {value}")
    return value


def _coerce_float(data: dict, key: str) -> float:
    return float(data.get(key, DEFAULT_WALK_FORWARD_POLICY[key]))


def load_walk_forward_policy(project_root: str, environ: Optional[Mapping[str, str]] = None) -> dict:
    path = _resolve_policy_path(project_root, environ=environ)
    payload = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("walk-forward policy 檔案必須是 JSON object")
        payload = dict(loaded)
    merged = dict(DEFAULT_WALK_FORWARD_POLICY)
    merged.update(payload)
    merged['train_start_year'] = _coerce_int(merged, 'train_start_year', 1900)
    merged['min_train_years'] = _coerce_int(merged, 'min_train_years', 1)
    merged['test_window_months'] = _coerce_int(merged, 'test_window_months', 1)
    merged['regime_up_threshold_pct'] = _coerce_float(merged, 'regime_up_threshold_pct')
    merged['regime_down_threshold_pct'] = _coerce_float(merged, 'regime_down_threshold_pct')
    if merged['regime_down_threshold_pct'] > merged['regime_up_threshold_pct']:
        raise ValueError("walk-forward policy: regime_down_threshold_pct 不可大於 regime_up_threshold_pct")
    merged['policy_path'] = path
    return merged
