"""Strategy Compare parameter/source resolution and controlled-pair construction."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
    get_active_param_ensemble_date_range,
    get_active_param_ensemble_policy,
    is_active_param_ensemble_payload,
    resolve_active_param_ensemble_mode,
)
from core.buy_sort import BREAKOUT_QUALITY_RANKING_POLICY_SCORE
from core.model_paths import resolve_default_primary_param_source_record
from core.file_integrity import canonical_json_sha256
from core.strategy_param_artifacts import (
    freeze_strategy_param_payload_for_period,
    resolve_strategy_param_artifact_path,
)
from core.params_io import build_params_from_mapping, load_params_from_json, params_to_json_dict
from core.rolling_oos_params import build_active_param_schedule, is_rolling_oos_param_set_payload
from core.seed_ensemble_policy import normalize_seed_ensemble_members
from filters.breakout_quality.artifacts import compute_file_sha256 as _sha256_file
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CANONICAL_RUNTIME,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
)
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_HARD_FILTER,
    COMPARISON_MODE_SCORE_RANKING,
    COMPARISON_MODES,
    comparison_labels as _comparison_labels,
    comparison_switch_spec as _comparison_switch_spec,
)

def read_json_object_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_project_relative_path(root: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"設定路徑必須是專案root相對路徑: {value}")
    return (root / path).resolve()


PARAM_POLICY_AUTO = "auto"
PARAM_POLICY_BASE_FINALIST_BEST = "base-finalist-best"
PARAM_POLICY_BASE_FINALISTS_AGREE = "base-finalists-agree"
PARAM_POLICY_SPECS = {
    PARAM_POLICY_BASE_FINALIST_BEST: {
        "selector": "base_finalist_best",
        "filename": "roos_base_best.json",
        "output_suffix": "base_finalist_best",
        "expected_member_count": 1,
        "expected_min_agree": 1,
    },
    PARAM_POLICY_BASE_FINALISTS_AGREE: {
        "selector": "base_finalists_agree",
        "filename": "roos_base_finalists_agree.json",
        "output_suffix": "base_finalists_agree",
        "expected_member_count": None,
        "expected_min_agree": None,
    },
}
PARAM_POLICIES = (PARAM_POLICY_AUTO, *PARAM_POLICY_SPECS.keys())

OPTIONAL_ENTRY_FILTER_POLICY_CURRENT = "current"
OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF = "all-off"
SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES = (
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
)
OPTIONAL_ENTRY_FILTER_FIELDS = (
    "use_breakout_ema_filter",
    "use_bb",
    "use_vol",
    "use_breakout_return_filter",
    "use_breakout_false_filter",
)


def _assert_controlled_param_pair(no_filter_params, quality_params, *, comparison_mode=COMPARISON_MODE_HARD_FILTER) -> None:
    left = params_to_json_dict(no_filter_params)
    right = params_to_json_dict(quality_params)
    switch_field, left_expected, right_expected = _comparison_switch_spec(comparison_mode)
    differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
    if differing != [switch_field]:
        raise ValueError(f"策略對照只允許 {switch_field} 不同，實際差異={differing}")
    if left[switch_field] is not left_expected or right[switch_field] is not right_expected:
        raise ValueError(f"策略對照的 {switch_field} 開關方向不正確")
    if bool(left.get("use_breakout_quality_filter")) and bool(left.get("use_breakout_quality_ranking")):
        raise ValueError("baseline 不可同時啟用 hard filter 與 score ranking")
    if bool(right.get("use_breakout_quality_filter")) and bool(right.get("use_breakout_quality_ranking")):
        raise ValueError("active scenario 不可同時啟用 hard filter 與 score ranking")

def _collect_payload_differences(left: Any, right: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                out.append((path + (key,), left.get(key), right.get(key)))
            else:
                out.extend(_collect_payload_differences(left[key], right[key], path + (key,)))
        return out
    if isinstance(left, list) and isinstance(right, list):
        out = []
        if len(left) != len(right):
            return [(path + ("length",), len(left), len(right))]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            out.extend(_collect_payload_differences(left_item, right_item, path + (index,)))
        return out
    return [] if left == right else [(path, left, right)]

def _assert_controlled_payload_pair(no_filter_payload: dict, quality_payload: dict, *, comparison_mode=COMPARISON_MODE_HARD_FILTER) -> None:
    differences = _collect_payload_differences(no_filter_payload, quality_payload)
    switch_field, left_expected, right_expected = _comparison_switch_spec(comparison_mode)
    if not differences:
        raise ValueError(f"策略對照 payload 沒有切換 {switch_field}")
    invalid = [
        ("/".join(map(str, path)), left, right)
        for path, left, right in differences
        if not path or path[-1] != switch_field or left is not left_expected or right is not right_expected
    ]
    if invalid:
        raise ValueError(f"策略對照只允許 {switch_field} 由 {left_expected} 切為 {right_expected}，實際額外差異={invalid}")

def _resolve_param_selector(source: dict[str, Any]) -> str:
    payload = source.get("payload")
    if isinstance(payload, dict):
        selector = str(payload.get("selector") or "").strip()
        if selector:
            return selector
        policy = get_active_param_ensemble_policy(payload)
        selector = str((policy or {}).get("policy_name") or "").strip()
        if selector:
            return selector
    return "single_param" if source.get("kind") == "single_param" else "unknown"

def _rolling_member_counts(source: dict[str, Any]) -> list[int]:
    payload = source.get("payload")
    if not isinstance(payload, dict):
        return []
    mapping = payload.get("params_ensemble_by_effective_date")
    if not isinstance(mapping, dict):
        return []
    counts = []
    for effective_date, raw_members in mapping.items():
        members = normalize_seed_ensemble_members(raw_members)
        if not members or len(members) != len(list(raw_members or [])):
            raise ValueError(f"生效日 {effective_date} 的 active-param members 無效")
        counts.append(len(members))
    return counts

def _validate_requested_param_policy(source: dict[str, Any], requested_policy: str) -> dict[str, Any]:
    selector = _resolve_param_selector(source)
    member_counts = _rolling_member_counts(source)
    policy = get_active_param_ensemble_policy(source.get("payload") or {}) if isinstance(source.get("payload"), dict) else None
    actual_min_agree = int((policy or {}).get("min_agree", 0) or 0) if policy else None
    if requested_policy == PARAM_POLICY_AUTO:
        return {
            "requested_policy": requested_policy,
            "selector": selector,
            "member_count_min": min(member_counts) if member_counts else None,
            "member_count_max": max(member_counts) if member_counts else None,
            "min_agree": actual_min_agree,
        }

    spec = PARAM_POLICY_SPECS[requested_policy]
    if source.get("kind") != "rolling_active_param_ensemble":
        raise ValueError(
            f"--param-policy {requested_policy} 只接受 rolling active-param ensemble JSON，"
            f"實際={source.get('kind')}"
        )
    if selector != spec["selector"]:
        raise ValueError(
            f"--param-policy {requested_policy} 與參數檔 selector 不一致: "
            f"expected={spec['selector']}, actual={selector}"
        )
    expected_member_count = spec.get("expected_member_count")
    if expected_member_count is not None and (not member_counts or any(count != expected_member_count for count in member_counts)):
        raise ValueError(
            f"{spec['selector']} 必須每個生效日恰有 {expected_member_count} 個 runtime member，"
            f"實際範圍={min(member_counts) if member_counts else 0}~{max(member_counts) if member_counts else 0}"
        )
    expected_min_agree = spec.get("expected_min_agree")
    if expected_min_agree is not None and actual_min_agree != expected_min_agree:
        raise ValueError(
            f"{spec['selector']} 的 min_agree 必須為 {expected_min_agree}，實際={actual_min_agree}"
        )
    return {
        "requested_policy": requested_policy,
        "selector": selector,
        "member_count_min": min(member_counts) if member_counts else None,
        "member_count_max": max(member_counts) if member_counts else None,
        "min_agree": actual_min_agree,
    }

def _resolve_params_path(
    *, root: Path, params_path: str | None, param_policy: str,
    allow_static_diagnostic: bool, score_source: str = SCORE_SOURCE_CANONICAL_RUNTIME,
) -> Path:
    if params_path:
        requested = Path(params_path)
        return requested.resolve() if requested.is_absolute() else (root / requested).resolve()
    if param_policy != PARAM_POLICY_AUTO:
        return resolve_strategy_param_artifact_path(
            root,
            family="full",
            evaluation_mode="rolling",
            policy=PARAM_POLICY_SPECS[param_policy]["selector"],
        ).resolve()
    if allow_static_diagnostic:
        return Path(resolve_default_primary_param_source_record(str(root))["path"]).resolve()
    raise ValueError(
        "正式策略對照必須以 --params 指定明確工件，或用 "
        "--param-policy base-finalist-best / base-finalists-agree 解析 canonical Strategy Parameter SSOT；"
        "只有非 OOS 敏感度診斷才可加 --allow-static-diagnostic 使用 canonical Trade active state。"
    )

def _comparison_output_dir_name(
    comparison_mode: str,
    labels: dict[str, str],
    *,
    param_policy: str,
    ranking_policy: str = BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
) -> str:
    if param_policy != PARAM_POLICY_AUTO:
        name = f"{labels['output_dir']}_{PARAM_POLICY_SPECS[param_policy]['output_suffix']}"
    else:
        name = labels["output_dir"]
    if (
        comparison_mode == COMPARISON_MODE_SCORE_RANKING
        and ranking_policy != BREAKOUT_QUALITY_RANKING_POLICY_SCORE
    ):
        name += "_" + str(ranking_policy).replace("-", "_")
    if optional_entry_filter_policy == OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF:
        name += "_optional_entry_filters_all_off"
    return name

def canonical_strategy_compare_output_dir_names(
    comparison_mode: str | None = None,
) -> tuple[str, ...]:
    """Return active strategy-compare directory names in semantic priority order."""

    if comparison_mode is not None and comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"不支援的 comparison mode: {comparison_mode!r}")
    hard_filter_labels = _comparison_labels(COMPARISON_MODE_HARD_FILTER)
    score_ranking_labels = _comparison_labels(COMPARISON_MODE_SCORE_RANKING)
    hard_filter_names = (
        _comparison_output_dir_name(
            COMPARISON_MODE_HARD_FILTER,
            hard_filter_labels,
            param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        ),
        _comparison_output_dir_name(
            COMPARISON_MODE_HARD_FILTER,
            hard_filter_labels,
            param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
        ),
        _comparison_output_dir_name(
            COMPARISON_MODE_HARD_FILTER,
            hard_filter_labels,
            param_policy=PARAM_POLICY_AUTO,
        ),
    )
    score_ranking_names = (
        _comparison_output_dir_name(
            COMPARISON_MODE_SCORE_RANKING,
            score_ranking_labels,
            param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
        ),
        _comparison_output_dir_name(
            COMPARISON_MODE_SCORE_RANKING,
            score_ranking_labels,
            param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        ),
        _comparison_output_dir_name(
            COMPARISON_MODE_SCORE_RANKING,
            score_ranking_labels,
            param_policy=PARAM_POLICY_AUTO,
        ),
    )
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        return hard_filter_names
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        return score_ranking_names
    return (*hard_filter_names, *score_ranking_names)

def _first_existing_comparison_dir(
    root: Path,
    *,
    comparison_mode: str,
) -> Path:
    candidates = [
        root / name
        for name in canonical_strategy_compare_output_dir_names(comparison_mode)
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "找不到既有strategy compare輸出目錄；已檢查: "
        + ", ".join(str(path) for path in candidates)
    )

def apply_strategy_param_evaluation_view(
    source: dict[str, Any], *, evaluation_mode: str | None, start_date: str, end_date: str
) -> dict[str, Any]:
    """Apply OOS/Rolling consumption semantics without creating another file.

    Canonical and benchmark strategy parameter JSONs contain one complete PIT-safe
    effective-date schedule.  Rolling consumes that schedule as-is; OOS freezes the
    latest member legal at the comparison start entirely in memory.
    """
    mode = str(evaluation_mode or "rolling").strip().lower()
    if mode in {"roos"}:
        mode = "rolling"
    if mode in {"split"}:
        mode = "oos"
    if mode not in {"oos", "rolling"}:
        return source
    if source.get("kind") != "rolling_active_param_ensemble":
        return source
    if mode == "rolling":
        return source
    payload = freeze_strategy_param_payload_for_period(
        source["payload"], start_date=str(start_date), end_date=str(end_date)
    )
    return {**source, "payload": payload, "evaluation_view": "frozen_oos"}


def strategy_param_source_identity_sha256(source: dict[str, Any], *, source_path: Path) -> str:
    """Hash the parameter content actually consumed by this evaluation.

    Rolling hashes the full schedule; OOS hashes its in-memory frozen view.  This
    keeps OOS result reuse stable when later Rolling-only members are appended to
    the same physical JSON.
    """
    payload = source.get("payload")
    if isinstance(payload, dict):
        return canonical_json_sha256(payload)
    return _sha256_file(Path(source_path))


def _load_param_source(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取參數來源失敗: {path}｜{type(exc).__name__}: {exc}") from exc

    if is_rolling_oos_param_set_payload(payload):
        if resolve_active_param_ensemble_mode(payload) == "rolling":
            get_active_param_ensemble_date_range(payload)
            return {"kind": "rolling_active_param_ensemble", "payload": payload}
        build_active_param_schedule(payload)
        return {"kind": "rolling_oos_param_schedule", "payload": payload}

    if is_active_param_ensemble_payload(payload):
        mode = resolve_active_param_ensemble_mode(payload)
        if mode != ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
            raise ValueError(f"不支援的 active-param ensemble mode={mode or 'unknown'}")
        members = normalize_seed_ensemble_members(payload.get("params_ensemble"))
        if not members or len(members) != len(list(payload.get("params_ensemble") or [])):
            raise ValueError("static active-param ensemble 內含無效或缺失的 member params")
        return {"kind": "static_active_param_ensemble", "payload": payload}
    return {"kind": "single_param", "params": load_params_from_json(str(path))}

def _apply_scenario_overrides(
    params,
    *,
    active: bool,
    comparison_mode: str,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    max_position_cap_pct: float | None = None,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    shared_param_overrides: dict[str, Any] | None = None,
):
    _comparison_switch_spec(comparison_mode)
    overrides = {
        "use_breakout_quality_filter": bool(active and comparison_mode == COMPARISON_MODE_HARD_FILTER),
        "use_breakout_quality_ranking": bool(active and comparison_mode == COMPARISON_MODE_SCORE_RANKING),
        "breakout_quality_filter_id": str(filter_id),
        "breakout_quality_score_threshold": float(threshold),
    }
    optional_entry_filter_policy = str(optional_entry_filter_policy).strip()
    if optional_entry_filter_policy not in SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES:
        raise ValueError(
            f"不支援的 optional entry filter policy: {optional_entry_filter_policy!r}"
        )
    if optional_entry_filter_policy == OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF:
        overrides.update({field: False for field in OPTIONAL_ENTRY_FILTER_FIELDS})
    if shared_param_overrides:
        reserved_fields = {
            "use_breakout_quality_filter",
            "use_breakout_quality_ranking",
            "breakout_quality_filter_id",
            "breakout_quality_score_threshold",
            *OPTIONAL_ENTRY_FILTER_FIELDS,
        }
        invalid_reserved = sorted(set(shared_param_overrides) & reserved_fields)
        if invalid_reserved:
            raise ValueError(
                "shared_param_overrides不可覆寫比較開關或optional filter policy: "
                f"{invalid_reserved}"
            )
        unknown_fields = sorted(
            set(shared_param_overrides) - set(params_to_json_dict(params))
        )
        if unknown_fields:
            raise ValueError(f"shared_param_overrides含未知策略參數: {unknown_fields}")
        overrides.update(dict(shared_param_overrides))
    if fixed_risk is not None:
        overrides["fixed_risk"] = float(fixed_risk)
    if max_position_cap_pct is not None:
        overrides["max_position_cap_pct"] = float(max_position_cap_pct)
    return replace(params, **overrides)

def _assert_controlled_ensemble_pair(no_filter_payload: dict, quality_payload: dict, *, comparison_mode=COMPARISON_MODE_HARD_FILTER) -> None:
    if resolve_active_param_ensemble_mode(no_filter_payload) != resolve_active_param_ensemble_mode(quality_payload):
        raise ValueError("策略對照的 ensemble mode 不一致")
    _assert_controlled_payload_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
    left_mode = resolve_active_param_ensemble_mode(no_filter_payload)
    if left_mode == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
        left_members = normalize_seed_ensemble_members(no_filter_payload.get("params_ensemble"))
        right_members = normalize_seed_ensemble_members(quality_payload.get("params_ensemble"))
        if len(left_members) != len(right_members) or not left_members:
            raise ValueError("策略對照的 static ensemble member 數量不一致或為空")
    else:
        get_active_param_ensemble_date_range(no_filter_payload)
        get_active_param_ensemble_date_range(quality_payload)

def _rewrite_param_mapping(
    mapping: dict,
    *,
    active: bool,
    comparison_mode: str,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    max_position_cap_pct: float | None = None,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    shared_param_overrides: dict[str, Any] | None = None,
) -> dict:
    rewritten = {}
    for key, raw_params in dict(mapping or {}).items():
        base_params = build_params_from_mapping(raw_params)
        rewritten[str(key)] = params_to_json_dict(
            _apply_scenario_overrides(
                base_params,
                active=active,
                comparison_mode=comparison_mode,
                filter_id=filter_id,
                threshold=threshold,
                fixed_risk=fixed_risk,
                max_position_cap_pct=max_position_cap_pct,
                optional_entry_filter_policy=optional_entry_filter_policy,
                shared_param_overrides=shared_param_overrides,
            )
        )
    return rewritten

def _rewrite_ensemble_mapping(
    mapping: dict,
    *,
    active: bool,
    comparison_mode: str,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    max_position_cap_pct: float | None = None,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    shared_param_overrides: dict[str, Any] | None = None,
) -> dict:
    rewritten = {}
    for key, raw_members in dict(mapping or {}).items():
        members = normalize_seed_ensemble_members(raw_members)
        if not members or len(members) != len(list(raw_members or [])):
            raise ValueError(f"生效日 {key} 的 ensemble members 無效")
        output_members = []
        for member in members:
            base_params = build_params_from_mapping(member["params"])
            output_member = copy.deepcopy(member)
            output_member["params"] = params_to_json_dict(
                _apply_scenario_overrides(
                    base_params,
                    active=active,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                    shared_param_overrides=shared_param_overrides,
                )
            )
            output_members.append(output_member)
        rewritten[str(key)] = output_members
    return rewritten

def _build_controlled_param_source_pair(
    source: dict[str, Any],
    *,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    max_position_cap_pct: float | None = None,
    comparison_mode: str = COMPARISON_MODE_HARD_FILTER,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    shared_param_overrides: dict[str, Any] | None = None,
) -> tuple[str, Any, Any, Any, Any, dict[str, Any] | None]:
    kind = str(source["kind"])
    if kind == "single_param":
        base_params = source["params"]
        no_filter_params = _apply_scenario_overrides(
            base_params,
            active=False,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
            shared_param_overrides=shared_param_overrides,
        )
        quality_params = _apply_scenario_overrides(
            base_params,
            active=True,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
            shared_param_overrides=shared_param_overrides,
        )
        _assert_controlled_param_pair(no_filter_params, quality_params, comparison_mode=comparison_mode)
        return (
            kind,
            no_filter_params,
            quality_params,
            params_to_json_dict(no_filter_params),
            params_to_json_dict(quality_params),
            None,
        )

    if kind not in {
        "static_active_param_ensemble",
        "rolling_oos_param_schedule",
        "rolling_active_param_ensemble",
    }:
        raise ValueError(f"不支援的參數來源類型: {kind}")

    base_payload = copy.deepcopy(source["payload"])
    no_filter_payload = copy.deepcopy(base_payload)
    quality_payload = copy.deepcopy(base_payload)

    if kind == "static_active_param_ensemble":
        base_mapping = {"static": base_payload.get("params_ensemble")}
        no_filter_payload["params_ensemble"] = _rewrite_ensemble_mapping(
            base_mapping,
            active=False,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
            shared_param_overrides=shared_param_overrides,
        )["static"]
        quality_payload["params_ensemble"] = _rewrite_ensemble_mapping(
            base_mapping,
            active=True,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
            shared_param_overrides=shared_param_overrides,
        )["static"]
        _assert_controlled_ensemble_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
        policy = get_active_param_ensemble_policy(base_payload)
    elif kind == "rolling_active_param_ensemble":
        for field in ("params_by_effective_date", "params_by_oos_year"):
            raw_mapping = base_payload.get(field)
            if isinstance(raw_mapping, dict) and raw_mapping:
                no_filter_payload[field] = _rewrite_param_mapping(
                    raw_mapping,
                    active=False,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                    shared_param_overrides=shared_param_overrides,
                )
                quality_payload[field] = _rewrite_param_mapping(
                    raw_mapping,
                    active=True,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                    shared_param_overrides=shared_param_overrides,
                )
        ensemble_mapping = base_payload.get("params_ensemble_by_effective_date")
        if not isinstance(ensemble_mapping, dict) or not ensemble_mapping:
            raise ValueError("rolling active-param ensemble 缺少 params_ensemble_by_effective_date")
        no_filter_payload["params_ensemble_by_effective_date"] = _rewrite_ensemble_mapping(
            ensemble_mapping,
            active=False,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
            shared_param_overrides=shared_param_overrides,
        )
        quality_payload["params_ensemble_by_effective_date"] = _rewrite_ensemble_mapping(
            ensemble_mapping,
            active=True,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
            shared_param_overrides=shared_param_overrides,
        )
        _assert_controlled_ensemble_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
        policy = get_active_param_ensemble_policy(base_payload)
    else:
        updated_any = False
        for field in ("params_by_effective_date", "params_by_oos_year"):
            raw_mapping = base_payload.get(field)
            if isinstance(raw_mapping, dict) and raw_mapping:
                updated_any = True
                no_filter_payload[field] = _rewrite_param_mapping(
                    raw_mapping,
                    active=False,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                    shared_param_overrides=shared_param_overrides,
                )
                quality_payload[field] = _rewrite_param_mapping(
                    raw_mapping,
                    active=True,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                    shared_param_overrides=shared_param_overrides,
                )
        if not updated_any:
            raise ValueError("rolling OOS param schedule 缺少 params_by_effective_date / params_by_oos_year")
        build_active_param_schedule(no_filter_payload)
        build_active_param_schedule(quality_payload)
        _assert_controlled_payload_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
        policy = None

    return (
        kind,
        no_filter_payload,
        quality_payload,
        no_filter_payload,
        quality_payload,
        policy,
    )

# Stable public aliases for read-only consumers.
first_existing_comparison_dir = _first_existing_comparison_dir
load_param_source = _load_param_source
build_controlled_param_source_pair = _build_controlled_param_source_pair
