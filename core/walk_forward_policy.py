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
    "min_window_bars": 20,
    "gate_min_median_score": 0.0,
    "gate_min_worst_ret_pct": -8.0,
    "gate_min_flat_median_score": 0.0,
    "compare_worst_ret_tolerance_pct": 1.0,
    "compare_max_mdd_tolerance_pct": 2.0,
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
    merged['min_window_bars'] = _coerce_int(merged, 'min_window_bars', 1)
    merged['gate_min_median_score'] = _coerce_float(merged, 'gate_min_median_score')
    merged['gate_min_worst_ret_pct'] = _coerce_float(merged, 'gate_min_worst_ret_pct')
    merged['gate_min_flat_median_score'] = _coerce_float(merged, 'gate_min_flat_median_score')
    merged['compare_worst_ret_tolerance_pct'] = _coerce_float(merged, 'compare_worst_ret_tolerance_pct')
    merged['compare_max_mdd_tolerance_pct'] = _coerce_float(merged, 'compare_max_mdd_tolerance_pct')
    if merged['regime_down_threshold_pct'] > merged['regime_up_threshold_pct']:
        raise ValueError("walk-forward policy: regime_down_threshold_pct 不可大於 regime_up_threshold_pct")
    if merged['compare_worst_ret_tolerance_pct'] < 0.0:
        raise ValueError("walk-forward policy: compare_worst_ret_tolerance_pct 不可 < 0")
    if merged['compare_max_mdd_tolerance_pct'] < 0.0:
        raise ValueError("walk-forward policy: compare_max_mdd_tolerance_pct 不可 < 0")
    merged['policy_path'] = path
    return merged
