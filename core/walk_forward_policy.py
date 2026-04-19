import importlib.util
import json
import os
from typing import Mapping, Optional

from config.training_policy import TRAINING_SPLIT_POLICY

WALK_FORWARD_POLICY_PATH_ENV_VAR = "V16_WALK_FORWARD_POLICY_PATH"

DEFAULT_TRAINING_SPLIT_POLICY = {
    "architecture_mode": str(TRAINING_SPLIT_POLICY.get("architecture_mode", "fixed_predeploy_oos")),
    "disable_legacy_model": bool(TRAINING_SPLIT_POLICY.get("disable_legacy_model", True)),
    "disable_promotion_flow": bool(TRAINING_SPLIT_POLICY.get("disable_promotion_flow", True)),
    "train_start_year": int(TRAINING_SPLIT_POLICY["train_start_year"]),
    "min_train_years": int(TRAINING_SPLIT_POLICY["min_train_years"]),
    "search_train_end_year": None,
    "oos_start_year": TRAINING_SPLIT_POLICY.get("oos_start_year", None),
}


def _resolve_policy_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(WALK_FORWARD_POLICY_PATH_ENV_VAR, "")).strip()
    if override:
        if os.path.isabs(override):
            return os.path.abspath(override)
        return os.path.abspath(os.path.join(project_root, override))
    return os.path.abspath(os.path.join(project_root, "config", "training_policy.py"))


def _coerce_int(data: dict, key: str, minimum: int) -> int:
    value = int(data.get(key, DEFAULT_TRAINING_SPLIT_POLICY[key]))
    if value < minimum:
        raise ValueError(f"training split policy: {key} 必須 >= {minimum}，收到: {value}")
    return value


def _coerce_optional_int(data: dict, key: str, minimum: int) -> int | None:
    value = data.get(key, DEFAULT_TRAINING_SPLIT_POLICY[key])
    if value is None:
        return None
    value = int(value)
    if value < minimum:
        raise ValueError(f"training split policy: {key} 必須 >= {minimum}，收到: {value}")
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
            raise ValueError("training split policy JSON 檔案必須是 object")
        return dict(loaded)
    if ext == ".py":
        module_name = f"_v16_training_policy_{abs(hash(os.path.abspath(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"training split policy 無法載入 Python 設定檔: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded = getattr(module, "TRAINING_SPLIT_POLICY", None)
        if loaded is None and os.path.basename(path) == 'training_policy.py':
            train_start_year = getattr(module, 'OPTIMIZER_TRAIN_START_YEAR', None)
            min_train_years = getattr(module, 'OPTIMIZER_MIN_TRAIN_YEARS', None)
            if train_start_year is not None and min_train_years is not None:
                loaded = {
                    'train_start_year': int(train_start_year),
                    'min_train_years': int(min_train_years),
                }
        if not isinstance(loaded, dict):
            raise ValueError("training split policy Python 設定檔必須定義 TRAINING_SPLIT_POLICY dict")
        return dict(loaded)
    raise ValueError(f"training split policy 只支援 .py 或 .json，收到: {path}")


def load_walk_forward_policy(project_root: str, environ: Optional[Mapping[str, str]] = None) -> dict:
    path = _resolve_policy_path(project_root, environ=environ)
    payload = _load_policy_payload(path)
    merged = dict(DEFAULT_TRAINING_SPLIT_POLICY)
    merged.update(payload)
    merged['architecture_mode'] = str(merged.get('architecture_mode', DEFAULT_TRAINING_SPLIT_POLICY['architecture_mode'])).strip() or str(DEFAULT_TRAINING_SPLIT_POLICY['architecture_mode'])
    merged['disable_legacy_model'] = bool(merged.get('disable_legacy_model', DEFAULT_TRAINING_SPLIT_POLICY['disable_legacy_model']))
    merged['disable_promotion_flow'] = bool(merged.get('disable_promotion_flow', DEFAULT_TRAINING_SPLIT_POLICY['disable_promotion_flow']))
    merged['train_start_year'] = _coerce_int(merged, 'train_start_year', 1900)
    merged['min_train_years'] = _coerce_int(merged, 'min_train_years', 1)
    merged['search_train_end_year'] = _coerce_optional_int(merged, 'search_train_end_year', 1900)
    merged['oos_start_year'] = _coerce_optional_int(merged, 'oos_start_year', 1900)
    if merged['oos_start_year'] is not None:
        merged['search_train_end_year'] = int(merged['oos_start_year']) - 1
    if merged['search_train_end_year'] is None:
        merged['search_train_end_year'] = int(merged['train_start_year']) + int(merged['min_train_years']) - 1
    if merged['search_train_end_year'] < merged['train_start_year']:
        raise ValueError('training split policy: search_train_end_year 不可小於 train_start_year')
    if merged['oos_start_year'] is not None and int(merged['oos_start_year']) <= int(merged['search_train_end_year']):
        pass
    if merged['oos_start_year'] is not None and int(merged['oos_start_year']) <= int(merged['train_start_year']):
        raise ValueError('training split policy: oos_start_year 必須大於 train_start_year')
    merged['policy_path'] = path
    return merged


def filter_search_train_dates(*, sorted_dates, train_start_year: int, search_train_end_year: int):
    filtered = []
    start_year = int(train_start_year)
    end_year = int(search_train_end_year)
    for raw_date in list(sorted_dates or []):
        year = int(getattr(raw_date, 'year', 0) or 0)
        if year == 0:
            raw_text = str(raw_date or '').strip()
            try:
                year = int(raw_text[:4])
            except (TypeError, ValueError):
                continue
        if start_year <= year <= end_year:
            filtered.append(raw_date)
    return filtered
