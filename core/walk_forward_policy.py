import hashlib
import importlib.util
import json
import os
from typing import Mapping, Optional

from config.training_policy import TRAINING_SPLIT_POLICY, build_training_score_policy_snapshot

WALK_FORWARD_POLICY_PATH_ENV_VAR = "V16_WALK_FORWARD_POLICY_PATH"
WALK_FORWARD_SELECTION_START_YEAR_ENV_VAR = "V16_WF_SELECTION_START_YEAR"
WALK_FORWARD_TRAIN_START_YEAR_ENV_VAR = "V16_WF_TRAIN_START_YEAR"
WALK_FORWARD_MIN_TRAIN_YEARS_ENV_VAR = "V16_WF_MIN_TRAIN_YEARS"
WALK_FORWARD_SEARCH_TRAIN_END_YEAR_ENV_VAR = "V16_WF_SEARCH_TRAIN_END_YEAR"
WALK_FORWARD_OOS_START_YEAR_ENV_VAR = "V16_WF_OOS_START_YEAR"

def _normalize_objective_mode(objective_mode: str) -> str:
    mode = str(objective_mode or "").strip()
    if mode == "":
        return str(DEFAULT_TRAINING_SPLIT_POLICY["objective_mode"])
    return mode


DEFAULT_TRAINING_SPLIT_POLICY = {
    "selection_start_year": int(TRAINING_SPLIT_POLICY.get("selection_start_year", TRAINING_SPLIT_POLICY["train_start_year"])),
    "train_start_year": int(TRAINING_SPLIT_POLICY["train_start_year"]),
    "min_train_years": int(TRAINING_SPLIT_POLICY["min_train_years"]),
    "search_train_end_year": None,
    "oos_start_year": TRAINING_SPLIT_POLICY.get("oos_start_year", None),
    "objective_mode": str(TRAINING_SPLIT_POLICY.get("objective_mode", "split_train_romd")),
}


def _extract_inline_policy_overrides(environ: Optional[Mapping[str, str]] = None) -> dict:
    env = os.environ if environ is None else environ
    raw_mapping = {
        "selection_start_year": str(env.get(WALK_FORWARD_SELECTION_START_YEAR_ENV_VAR, "")).strip(),
        "train_start_year": str(env.get(WALK_FORWARD_TRAIN_START_YEAR_ENV_VAR, "")).strip(),
        "min_train_years": str(env.get(WALK_FORWARD_MIN_TRAIN_YEARS_ENV_VAR, "")).strip(),
        "search_train_end_year": str(env.get(WALK_FORWARD_SEARCH_TRAIN_END_YEAR_ENV_VAR, "")).strip(),
        "oos_start_year": str(env.get(WALK_FORWARD_OOS_START_YEAR_ENV_VAR, "")).strip(),
    }
    overrides = {}
    for key, raw_value in raw_mapping.items():
        if raw_value == "":
            continue
        try:
            overrides[key] = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"training split policy env override: {key} 必須是整數，收到: {raw_value}") from exc
    if "selection_start_year" in overrides and "train_start_year" not in overrides:
        overrides["train_start_year"] = int(overrides["selection_start_year"])
    if "train_start_year" in overrides and "selection_start_year" not in overrides:
        overrides["selection_start_year"] = int(overrides["train_start_year"])
    return overrides


def build_walk_forward_policy_effective_snapshot(policy: Mapping[str, object]) -> dict:
    return {
        "selection_start_year": int(policy["selection_start_year"]),
        "train_start_year": int(policy["train_start_year"]),
        "min_train_years": int(policy["min_train_years"]),
        "search_train_end_year": int(policy["search_train_end_year"]),
        "oos_start_year": None if policy.get("oos_start_year") is None else int(policy["oos_start_year"]),
        "objective_mode": str(policy.get("objective_mode", DEFAULT_TRAINING_SPLIT_POLICY["objective_mode"])),
        "model_mode": str(policy.get("model_mode", "split")),
    }


def build_optimizer_effective_policy_snapshot(policy: Mapping[str, object]) -> dict:
    return {
        "walk_forward_policy": build_walk_forward_policy_effective_snapshot(policy),
        "training_score_policy": build_training_score_policy_snapshot(),
        "policy_schema_version": 1,
    }


def build_optimizer_effective_policy_fingerprint(policy: Mapping[str, object]) -> dict:
    snapshot = build_optimizer_effective_policy_snapshot(policy)
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "snapshot": snapshot,
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
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
    inline_overrides = _extract_inline_policy_overrides(environ=environ)
    merged = dict(DEFAULT_TRAINING_SPLIT_POLICY)
    merged.update(payload)
    merged.update(inline_overrides)
    default_policy_path = os.path.abspath(os.path.join(project_root, "config", "training_policy.py"))
    is_external_override = os.path.abspath(path) != default_policy_path
    has_explicit_oos_or_end = any(key in payload or key in inline_overrides for key in ("oos_start_year", "search_train_end_year"))
    if is_external_override and not has_explicit_oos_or_end:
        merged["oos_start_year"] = None
    merged['selection_start_year'] = _coerce_int(merged, 'selection_start_year', 1900)
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
    merged['objective_mode'] = _normalize_objective_mode(merged.get('objective_mode', DEFAULT_TRAINING_SPLIT_POLICY['objective_mode']))
    merged['policy_path'] = path
    merged['inline_override_fields'] = sorted(inline_overrides.keys())
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


def build_optimizer_runtime_policy(base_policy: dict, model_mode: str) -> dict:
    normalized = str(model_mode or '').strip().lower() or 'split'
    if normalized not in {'split', 'full'}:
        raise ValueError(f"optimizer model_mode 只接受 split 或 full，收到: {model_mode}")
    runtime_policy = dict(base_policy or {})
    runtime_policy['model_mode'] = normalized
    if normalized == 'split':
        runtime_policy['objective_mode'] = 'split_train_romd'
        if runtime_policy.get('oos_start_year') is not None:
            runtime_policy['search_train_end_year'] = int(runtime_policy['oos_start_year']) - 1
    else:
        runtime_policy['objective_mode'] = 'legacy_base_score'
        runtime_policy['oos_start_year'] = None
    return runtime_policy
