import importlib.util
import json
import os
from typing import Mapping, Optional

WALK_FORWARD_POLICY_PATH_ENV_VAR = "V16_WALK_FORWARD_POLICY_PATH"
VALID_OBJECTIVE_MODES = {"legacy_base_score", "wf_gate_median"}

DEFAULT_WALK_FORWARD_POLICY = {
    "train_start_year": 2012,
    "min_train_years": 8,
    "test_window_months": 6,
    "search_train_end_year": None,
    "objective_mode": "legacy_base_score",
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
    return os.path.abspath(os.path.join(project_root, "config", "walk_forward_policy.py"))


def _coerce_int(data: dict, key: str, minimum: int) -> int:
    value = int(data.get(key, DEFAULT_WALK_FORWARD_POLICY[key]))
    if value < minimum:
        raise ValueError(f"walk-forward policy: {key} 必須 >= {minimum}，收到: {value}")
    return value


def _coerce_float(data: dict, key: str) -> float:
    return float(data.get(key, DEFAULT_WALK_FORWARD_POLICY[key]))


def _coerce_optional_int(data: dict, key: str, minimum: int) -> int | None:
    value = data.get(key, DEFAULT_WALK_FORWARD_POLICY[key])
    if value is None:
        return None
    value = int(value)
    if value < minimum:
        raise ValueError(f"walk-forward policy: {key} 必須 >= {minimum}，收到: {value}")
    return value


def _load_policy_payload(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("walk-forward policy JSON 檔案必須是 object")
        return dict(loaded)
    if ext == ".py":
        module_name = f"_v16_walk_forward_policy_{abs(hash(os.path.abspath(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"walk-forward policy 無法載入 Python 設定檔: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded = getattr(module, "WALK_FORWARD_POLICY", None)
        if not isinstance(loaded, dict):
            raise ValueError("walk-forward policy Python 設定檔必須定義 WALK_FORWARD_POLICY dict")
        return dict(loaded)
    raise ValueError(f"walk-forward policy 只支援 .py 或 .json，收到: {path}")


def load_walk_forward_policy(project_root: str, environ: Optional[Mapping[str, str]] = None) -> dict:
    path = _resolve_policy_path(project_root, environ=environ)
    payload = _load_policy_payload(path)
    merged = dict(DEFAULT_WALK_FORWARD_POLICY)
    merged.update(payload)
    merged['train_start_year'] = _coerce_int(merged, 'train_start_year', 1900)
    merged['min_train_years'] = _coerce_int(merged, 'min_train_years', 1)
    merged['test_window_months'] = _coerce_int(merged, 'test_window_months', 1)
    merged['search_train_end_year'] = _coerce_optional_int(merged, 'search_train_end_year', 1900)
    merged['regime_up_threshold_pct'] = _coerce_float(merged, 'regime_up_threshold_pct')
    merged['regime_down_threshold_pct'] = _coerce_float(merged, 'regime_down_threshold_pct')
    merged['min_window_bars'] = _coerce_int(merged, 'min_window_bars', 1)
    merged['gate_min_median_score'] = _coerce_float(merged, 'gate_min_median_score')
    merged['gate_min_worst_ret_pct'] = _coerce_float(merged, 'gate_min_worst_ret_pct')
    merged['gate_min_flat_median_score'] = _coerce_float(merged, 'gate_min_flat_median_score')
    merged['compare_worst_ret_tolerance_pct'] = _coerce_float(merged, 'compare_worst_ret_tolerance_pct')
    merged['compare_max_mdd_tolerance_pct'] = _coerce_float(merged, 'compare_max_mdd_tolerance_pct')
    objective_mode = str(merged.get('objective_mode', DEFAULT_WALK_FORWARD_POLICY['objective_mode'])).strip()
    if objective_mode not in VALID_OBJECTIVE_MODES:
        raise ValueError(f"walk-forward policy: objective_mode 不合法，收到: {objective_mode}")
    merged['objective_mode'] = objective_mode
    if merged['search_train_end_year'] is None:
        merged['search_train_end_year'] = int(merged['train_start_year']) + int(merged['min_train_years']) - 1
    if merged['search_train_end_year'] < merged['train_start_year']:
        raise ValueError('walk-forward policy: search_train_end_year 不可小於 train_start_year')
    if merged['regime_down_threshold_pct'] > merged['regime_up_threshold_pct']:
        raise ValueError("walk-forward policy: regime_down_threshold_pct 不可大於 regime_up_threshold_pct")
    if merged['compare_worst_ret_tolerance_pct'] < 0.0:
        raise ValueError("walk-forward policy: compare_worst_ret_tolerance_pct 不可 < 0")
    if merged['compare_max_mdd_tolerance_pct'] < 0.0:
        raise ValueError("walk-forward policy: compare_max_mdd_tolerance_pct 不可 < 0")
    merged['policy_path'] = path
    return merged

def filter_search_train_dates(*, sorted_dates, train_start_year: int, search_train_end_year: int):
    filtered = []
    start_year = int(train_start_year)
    end_year = int(search_train_end_year)
    for raw_date in list(sorted_dates or []):
        year = int(getattr(raw_date, "year", 0) or 0)
        if year == 0:
            raw_text = str(raw_date or "").strip()
            try:
                year = int(raw_text[:4])
            except (TypeError, ValueError):
                continue
        if start_year <= year <= end_year:
            filtered.append(raw_date)
    return filtered

