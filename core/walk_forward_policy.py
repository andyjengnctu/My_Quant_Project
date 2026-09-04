import hashlib
import importlib.util
import json
import os
from typing import Mapping, Optional

from core.display_policy import build_display_policy_snapshot
from config.training_policy import OUTER_ROLLING_TRAIN_WINDOW_MONTHS, build_training_score_policy_snapshot

WALK_FORWARD_POLICY_PATH_ENV_VAR = "V16_WALK_FORWARD_POLICY_PATH"
WALK_FORWARD_SELECTION_START_YEAR_ENV_VAR = "V16_WF_SELECTION_START_YEAR"
WALK_FORWARD_TRAIN_START_YEAR_ENV_VAR = "V16_WF_TRAIN_START_YEAR"
WALK_FORWARD_MIN_TRAIN_YEARS_ENV_VAR = "V16_WF_MIN_TRAIN_YEARS"
WALK_FORWARD_SEARCH_TRAIN_END_YEAR_ENV_VAR = "V16_WF_SEARCH_TRAIN_END_YEAR"
WALK_FORWARD_OOS_START_YEAR_ENV_VAR = "V16_WF_OOS_START_YEAR"
WALK_FORWARD_OOS_END_YEAR_ENV_VAR = "V16_WF_OOS_END_YEAR"
WALK_FORWARD_SELECTION_START_DATE_ENV_VAR = "V16_WF_SELECTION_START_DATE"
WALK_FORWARD_TRAIN_START_DATE_ENV_VAR = "V16_WF_TRAIN_START_DATE"
WALK_FORWARD_SEARCH_TRAIN_END_DATE_ENV_VAR = "V16_WF_SEARCH_TRAIN_END_DATE"
WALK_FORWARD_OOS_START_DATE_ENV_VAR = "V16_WF_OOS_START_DATE"
WALK_FORWARD_OOS_END_DATE_ENV_VAR = "V16_WF_OOS_END_DATE"

def _normalize_objective_mode(objective_mode: str, default_objective_mode: str) -> str:
    mode = str(objective_mode or "").strip()
    if mode == "":
        return str(default_objective_mode)
    return mode


def _normalize_date_text(value) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"training split policy: 日期必須是 YYYY-MM-DD，收到: {value!r}") from exc


def _year_from_date_text(value) -> int:
    parsed = _normalize_date_text(value)
    if not parsed:
        return 0
    return int(parsed[:4])


def _end_of_year_date_text(year: int | None) -> str | None:
    if year is None:
        return None
    return f"{int(year):04d}-12-31"


def _earliest_date_text(*values) -> str | None:
    normalized = [_normalize_date_text(value) for value in values if value is not None]
    normalized = [value for value in normalized if value]
    return min(normalized) if normalized else None


def _build_default_training_split_policy(project_root: str) -> dict:
    default_policy_path = os.path.abspath(os.path.join(project_root, "config", "training_policy.py"))
    payload = _load_policy_payload(default_policy_path)
    if not payload:
        raise ValueError(f"training split policy 預設設定檔不存在或無法載入: {default_policy_path}")
    train_start_year = int(payload["train_start_year"])
    min_train_years = int(payload["min_train_years"])
    selection_start_year = int(payload.get("selection_start_year", train_start_year))
    oos_start_year = payload.get("oos_start_year", None)
    oos_end_year = payload.get("oos_end_year", None)
    return {
        "selection_start_year": selection_start_year,
        "train_start_year": train_start_year,
        "min_train_years": min_train_years,
        "search_train_end_year": None,
        "oos_start_year": None if oos_start_year is None else int(oos_start_year),
        "oos_end_year": None if oos_end_year is None else int(oos_end_year),
        "full_start_year": payload.get("full_start_year"),
        "full_end_year": payload.get("full_end_year"),
        "objective_mode": str(payload.get("objective_mode", "split_train_romd")),
    }


def _extract_inline_policy_overrides(environ: Optional[Mapping[str, str]] = None) -> dict:
    env = os.environ if environ is None else environ
    raw_mapping = {
        "selection_start_year": str(env.get(WALK_FORWARD_SELECTION_START_YEAR_ENV_VAR, "")).strip(),
        "train_start_year": str(env.get(WALK_FORWARD_TRAIN_START_YEAR_ENV_VAR, "")).strip(),
        "min_train_years": str(env.get(WALK_FORWARD_MIN_TRAIN_YEARS_ENV_VAR, "")).strip(),
        "search_train_end_year": str(env.get(WALK_FORWARD_SEARCH_TRAIN_END_YEAR_ENV_VAR, "")).strip(),
        "oos_start_year": str(env.get(WALK_FORWARD_OOS_START_YEAR_ENV_VAR, "")).strip(),
        "oos_end_year": str(env.get(WALK_FORWARD_OOS_END_YEAR_ENV_VAR, "")).strip(),
        "selection_start_date": str(env.get(WALK_FORWARD_SELECTION_START_DATE_ENV_VAR, "")).strip(),
        "train_start_date": str(env.get(WALK_FORWARD_TRAIN_START_DATE_ENV_VAR, "")).strip(),
        "search_train_end_date": str(env.get(WALK_FORWARD_SEARCH_TRAIN_END_DATE_ENV_VAR, "")).strip(),
        "oos_start_date": str(env.get(WALK_FORWARD_OOS_START_DATE_ENV_VAR, "")).strip(),
        "oos_end_date": str(env.get(WALK_FORWARD_OOS_END_DATE_ENV_VAR, "")).strip(),
    }
    overrides = {}
    for key, raw_value in raw_mapping.items():
        if raw_value == "":
            continue
        if key.endswith("_date"):
            overrides[key] = _normalize_date_text(raw_value)
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
        "selection_start_date": policy.get("selection_start_date"),
        "train_start_date": policy.get("train_start_date"),
        "search_train_end_date": policy.get("search_train_end_date"),
        "oos_start_date": policy.get("oos_start_date"),
        "oos_end_date": policy.get("oos_end_date"),
        "latest_data_date": policy.get("latest_data_date"),
        "trade_train_window_months": policy.get("trade_train_window_months"),
        "train_window_months": policy.get("train_window_months"),
        "evaluation_scope": str(policy.get("evaluation_scope", "")),
        "study_scope": str(policy.get("study_scope", "")),
        "objective_mode": str(policy.get("objective_mode", "split_train_romd")),
        "model_mode": str(policy.get("model_mode", "oos")),
        "full_start_year": policy.get("full_start_year"),
        "full_end_year": policy.get("full_end_year"),
    }


def build_optimizer_effective_policy_snapshot(policy: Mapping[str, object]) -> dict:
    return {
        "walk_forward_policy": build_walk_forward_policy_effective_snapshot(policy),
        "training_score_policy": build_training_score_policy_snapshot(),
        "display_policy": build_display_policy_snapshot(),
        "policy_schema_version": 2,
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


def _coerce_int(data: dict, key: str, minimum: int, default_policy: Mapping[str, object]) -> int:
    value = int(data.get(key, default_policy[key]))
    if value < minimum:
        raise ValueError(f"training split policy: {key} 必須 >= {minimum}，收到: {value}")
    return value


def _coerce_optional_int(data: dict, key: str, minimum: int, default_policy: Mapping[str, object]) -> int | None:
    value = data.get(key, default_policy[key])
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
    default_policy = _build_default_training_split_policy(project_root)
    path = _resolve_policy_path(project_root, environ=environ)
    payload = _load_policy_payload(path)
    inline_overrides = _extract_inline_policy_overrides(environ=environ)
    merged = dict(default_policy)
    merged.update(payload)
    merged.update(inline_overrides)
    default_policy_path = os.path.abspath(os.path.join(project_root, "config", "training_policy.py"))
    is_external_override = os.path.abspath(path) != default_policy_path
    payload_oos_end_year_is_explicit = "oos_end_year" in payload
    payload_oos_end_date_is_explicit = "oos_end_date" in payload
    inline_oos_end_year_is_explicit = "oos_end_year" in inline_overrides
    inline_oos_end_date_is_explicit = "oos_end_date" in inline_overrides
    has_explicit_oos_or_end = any(
        key in payload or key in inline_overrides
        for key in ("oos_start_year", "oos_end_year", "search_train_end_year", "oos_start_date", "oos_end_date", "search_train_end_date")
    )
    has_explicit_oos_end_pair = (
        inline_oos_end_year_is_explicit and inline_oos_end_date_is_explicit
    ) or (
        not inline_oos_end_year_is_explicit
        and not inline_oos_end_date_is_explicit
        and payload_oos_end_year_is_explicit
        and payload_oos_end_date_is_explicit
    )
    if is_external_override and not has_explicit_oos_or_end:
        merged["oos_start_year"] = None
        merged["oos_end_year"] = None
        merged["oos_end_date"] = None
    elif inline_oos_end_year_is_explicit and not inline_oos_end_date_is_explicit:
        merged["oos_end_date"] = None
    elif is_external_override and payload_oos_end_year_is_explicit and not payload_oos_end_date_is_explicit:
        merged["oos_end_date"] = None
    for date_key in ("selection_start_date", "train_start_date", "search_train_end_date", "oos_start_date", "oos_end_date"):
        if date_key in merged:
            merged[date_key] = _normalize_date_text(merged.get(date_key))
    if merged.get("train_start_date") and "train_start_year" not in inline_overrides:
        merged["train_start_year"] = _year_from_date_text(merged["train_start_date"])
    if merged.get("selection_start_date") and "selection_start_year" not in inline_overrides:
        merged["selection_start_year"] = _year_from_date_text(merged["selection_start_date"])
    if merged.get("search_train_end_date") and "search_train_end_year" not in inline_overrides:
        merged["search_train_end_year"] = _year_from_date_text(merged["search_train_end_date"])
    if merged.get("oos_start_date") and "oos_start_year" not in inline_overrides:
        merged["oos_start_year"] = _year_from_date_text(merged["oos_start_date"])
    merged['selection_start_year'] = _coerce_int(merged, 'selection_start_year', 1900, default_policy)
    merged['train_start_year'] = _coerce_int(merged, 'train_start_year', 1900, default_policy)
    merged['min_train_years'] = _coerce_int(merged, 'min_train_years', 1, default_policy)
    merged['search_train_end_year'] = _coerce_optional_int(merged, 'search_train_end_year', 1900, default_policy)
    merged['oos_start_year'] = _coerce_optional_int(merged, 'oos_start_year', 1900, default_policy)
    merged['oos_end_year'] = _coerce_optional_int(merged, 'oos_end_year', 1900, default_policy)
    merged['full_start_year'] = _coerce_optional_int(merged, 'full_start_year', 1900, default_policy)
    merged['full_end_year'] = _coerce_optional_int(merged, 'full_end_year', 1900, default_policy)
    if merged.get('oos_end_date'):
        oos_end_date_year = _year_from_date_text(merged['oos_end_date'])
        if has_explicit_oos_end_pair and merged['oos_end_year'] is not None and int(merged['oos_end_year']) != int(oos_end_date_year):
            raise ValueError('training split policy: oos_end_year 必須與 oos_end_date 的年份一致')
        merged['oos_end_year'] = int(oos_end_date_year)
    elif merged['oos_end_year'] is not None:
        merged['oos_end_date'] = _end_of_year_date_text(merged['oos_end_year'])
    if merged['oos_start_year'] is not None and not merged.get("search_train_end_date"):
        merged['search_train_end_year'] = int(merged['oos_start_year']) - 1
    if merged['search_train_end_year'] is None:
        merged['search_train_end_year'] = int(merged['train_start_year']) + int(merged['min_train_years']) - 1
    if merged['search_train_end_year'] < merged['train_start_year']:
        raise ValueError('training split policy: search_train_end_year 不可小於 train_start_year')
    if merged['oos_start_year'] is not None and int(merged['oos_start_year']) <= int(merged['train_start_year']):
        raise ValueError('training split policy: oos_start_year 必須大於 train_start_year')
    if merged['oos_start_year'] is not None and merged['oos_end_year'] is not None and int(merged['oos_end_year']) < int(merged['oos_start_year']):
        raise ValueError('training split policy: oos_end_year 不可早於 oos_start_year')
    if merged.get('oos_start_date') and merged.get('oos_end_date') and str(merged['oos_end_date']) < str(merged['oos_start_date']):
        raise ValueError('training split policy: oos_end_date 不可早於 oos_start_date')
    if merged['full_start_year'] is not None and merged['full_end_year'] is not None and int(merged['full_end_year']) < int(merged['full_start_year']):
        raise ValueError('training split policy: full_end_year 不可早於 full_start_year')
    merged['objective_mode'] = _normalize_objective_mode(merged.get('objective_mode', default_policy['objective_mode']), default_policy['objective_mode'])
    merged['policy_path'] = path
    merged['inline_override_fields'] = sorted(inline_overrides.keys())
    return merged


def filter_search_train_dates(*, sorted_dates, train_start_year: int, search_train_end_year: int, train_start_date: str | None = None, search_train_end_date: str | None = None):
    filtered = []
    start_date_text = _normalize_date_text(train_start_date)
    end_date_text = _normalize_date_text(search_train_end_date)
    if start_date_text or end_date_text:
        import pandas as pd
        start_date = pd.Timestamp(start_date_text or f"{int(train_start_year)}-01-01").normalize()
        end_date = pd.Timestamp(end_date_text or f"{int(search_train_end_year)}-12-31").normalize()
        for raw_date in list([] if sorted_dates is None else sorted_dates):
            try:
                ts = pd.Timestamp(raw_date).normalize()
            except (TypeError, ValueError):
                continue
            if start_date <= ts <= end_date:
                filtered.append(raw_date)
        return filtered
    start_year = int(train_start_year)
    end_year = int(search_train_end_year)
    for raw_date in list([] if sorted_dates is None else sorted_dates):
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


def normalize_optimizer_model_mode(model_mode: str) -> str:
    normalized = str(model_mode or '').strip().lower() or 'oos'
    aliases = {
        'split': 'oos',
        'oos': 'oos',
        'study': 'study',
        'full': 'full',
        'trade': 'trade',
    }
    if normalized not in aliases:
        raise ValueError(f"optimizer model_mode 只接受 full、trade、oos 或 study，收到: {model_mode}")
    return aliases[normalized]


def normalize_optimizer_study_scope(study_scope: str | None) -> str:
    normalized = str(study_scope or '').strip().lower().replace('_', '-').replace(' ', '-')
    aliases = {
        '': 'full',
        'f': 'full',
        'full': 'full',
        'study-full': 'full',
        '1': 'oos',
        '2': 'oos',
        'o': 'oos',
        'oos': 'oos',
        'study-oos': 'oos',
    }
    if normalized not in aliases:
        raise ValueError(f"study mode 只接受 Enter/Study-Full 或 1/Study-OOS，收到: {study_scope}")
    return aliases[normalized]


def _resolve_trade_window_from_latest_date(latest_data_date, train_window_months: int) -> dict:
    import pandas as pd

    latest_ts = pd.Timestamp(_normalize_date_text(latest_data_date)).normalize()
    train_months = max(1, int(train_window_months or OUTER_ROLLING_TRAIN_WINDOW_MONTHS))
    selection_start = latest_ts - pd.DateOffset(months=train_months)
    selection_end = latest_ts
    return {
        'selection_start_year': int(selection_start.year),
        'train_start_year': int(selection_start.year),
        'search_train_end_year': int(selection_end.year),
        'selection_start_date': selection_start.strftime('%Y-%m-%d'),
        'train_start_date': selection_start.strftime('%Y-%m-%d'),
        'search_train_end_date': selection_end.strftime('%Y-%m-%d'),
        'latest_data_date': selection_end.strftime('%Y-%m-%d'),
        'trade_train_window_months': int(train_months),
    }


def _apply_full_window_runtime_policy(runtime_policy: dict, *, latest_data_date=None, evaluation_scope: str) -> dict:
    runtime_policy['evaluation_scope'] = str(evaluation_scope)
    runtime_policy['oos_start_year'] = None
    runtime_policy['oos_end_year'] = None
    runtime_policy['oos_start_date'] = None
    runtime_policy['oos_end_date'] = None
    runtime_policy['oos_horizon_months'] = 0
    full_start_year = int(
        runtime_policy.get('full_start_year')
        or runtime_policy.get('train_start_year')
        or runtime_policy.get('selection_start_year')
        or 0
    )
    if full_start_year <= 0:
        raise ValueError('training split policy: full_start_year 必須是有效西元年')
    runtime_policy['full_start_year'] = full_start_year
    runtime_policy['selection_start_year'] = full_start_year
    runtime_policy['train_start_year'] = full_start_year
    runtime_policy['selection_start_date'] = f"{full_start_year:04d}-01-01"
    runtime_policy['train_start_date'] = f"{full_start_year:04d}-01-01"
    full_end_year = runtime_policy.get('full_end_year')
    if full_end_year is not None:
        full_end_year = int(full_end_year)
        if full_end_year < full_start_year:
            raise ValueError('training split policy: full_end_year 不可早於 full_start_year')
        runtime_policy['full_end_year'] = full_end_year
    latest_text = _normalize_date_text(latest_data_date) if latest_data_date is not None else None
    effective_end_text = _earliest_date_text(
        latest_text,
        _end_of_year_date_text(full_end_year),
    )
    if effective_end_text is None:
        effective_end_text = _normalize_date_text(runtime_policy.get('search_train_end_date'))
    if effective_end_text is not None:
        runtime_policy['search_train_end_date'] = effective_end_text
        runtime_policy['search_train_end_year'] = _year_from_date_text(effective_end_text)
    if latest_text is not None:
        runtime_policy['latest_data_date'] = latest_text
    if int(runtime_policy['search_train_end_year']) < int(runtime_policy['train_start_year']):
        raise ValueError('training split policy: full_start_year 不可晚於 search_train_end_year')
    runtime_policy['min_train_years'] = max(
        1,
        int(runtime_policy['search_train_end_year']) - int(runtime_policy['train_start_year']) + 1,
    )
    return runtime_policy


def build_optimizer_runtime_policy(base_policy: dict, model_mode: str, *, latest_data_date=None, study_scope: str | None = None) -> dict:
    normalized = normalize_optimizer_model_mode(model_mode)
    runtime_policy = dict(base_policy or {})
    runtime_policy['model_mode'] = normalized
    runtime_policy['objective_mode'] = 'split_train_romd'
    runtime_policy['study_scope'] = ''
    if normalized == 'oos':
        runtime_policy['evaluation_scope'] = 'oos_single_fold'
        if runtime_policy.get('oos_start_year') is not None and not runtime_policy.get('search_train_end_date'):
            runtime_policy['search_train_end_year'] = int(runtime_policy['oos_start_year']) - 1
        return runtime_policy

    if normalized == 'full':
        return _apply_full_window_runtime_policy(
            runtime_policy,
            latest_data_date=latest_data_date,
            evaluation_scope='full_seed_ensemble',
        )

    if normalized == 'study':
        resolved_study_scope = normalize_optimizer_study_scope(study_scope)
        runtime_policy['study_scope'] = resolved_study_scope
        if resolved_study_scope == 'full':
            return _apply_full_window_runtime_policy(
                runtime_policy,
                latest_data_date=latest_data_date,
                evaluation_scope='study_full_single_seed',
            )
        runtime_policy['evaluation_scope'] = 'study_single_seed'
        if runtime_policy.get('oos_start_year') is not None and not runtime_policy.get('search_train_end_date'):
            runtime_policy['search_train_end_year'] = int(runtime_policy['oos_start_year']) - 1
        return runtime_policy

    runtime_policy['evaluation_scope'] = 'trade_train_only'
    runtime_policy['oos_start_year'] = None
    runtime_policy['oos_end_year'] = None
    runtime_policy['oos_start_date'] = None
    runtime_policy['oos_end_date'] = None
    runtime_policy['oos_horizon_months'] = 0
    runtime_policy['train_window_months'] = int(runtime_policy.get('trade_train_window_months') or OUTER_ROLLING_TRAIN_WINDOW_MONTHS)
    if latest_data_date is not None:
        runtime_policy.update(_resolve_trade_window_from_latest_date(
            latest_data_date,
            int(runtime_policy.get('train_window_months') or OUTER_ROLLING_TRAIN_WINDOW_MONTHS),
        ))
    return runtime_policy
