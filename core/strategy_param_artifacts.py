"""Canonical strategy-parameter artifact paths and identities.

Strategy parameters are stored by strategy identity, not by evaluation mode.
OOS and Rolling are consumption semantics over the same PIT-safe effective-date
schedule.  The optimizer domain remains the only producer of current artifacts;
Research/Strategy Compare resolves paths through this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
    build_active_param_ensemble_schedule,
    get_active_param_ensemble_date_range,
    get_active_param_ensemble_policy,
    is_active_param_ensemble_payload,
    resolve_active_param_ensemble_mode,
)
from core.file_integrity import canonical_json_sha256, load_json_strict
from core.raw_universe_contract import resolve_raw_universe_required_min_rows
from core.rolling_oos_params import build_active_param_schedule, is_rolling_oos_param_set_payload
from core.seed_ensemble_policy import normalize_seed_ensemble_members
from core.serialization_utils import json_native_value

STRATEGY_PARAM_ARTIFACT_SCHEMA_VERSION = 4
STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA = "strategy_param_runtime_v1"
STRATEGY_PARAM_ROOT_RELATIVE = Path("models") / "strategy_params"
STRATEGY_PARAM_CANONICAL_DIRNAME = "canonical"
STRATEGY_PARAM_BENCHMARK_DIRNAME = "benchmark"
STRATEGY_PARAM_WORK_ROOT_RELATIVE = Path("outputs") / "optimizer" / "strategy_param_schedule"
STRATEGY_PARAM_WORK_SCOPE_CANONICAL = "canonical"
STRATEGY_PARAM_WORK_SCOPE_BENCHMARK = "benchmark"

STRATEGY_PARAM_FAMILIES = ("full", "min")
STRATEGY_PARAM_EVALUATION_MODES = ("study", "full", "oos", "rolling", "trade")
STRATEGY_PARAM_SCHEDULE_MODES = ("oos", "rolling")

# Named current runtime state lives beside canonical strategy files; Optimizer
# resume/intermediate workspace never does. ``active`` is kept only as a compatibility
# API key; its physical filename is the explicit run_best artifact, not a second truth.
STRATEGY_PARAM_STATE_FILENAME_BY_NAME = {
    "active": "run_best_params.json",
    "active_summary": "run_best_summary.json",
    "candidate_best": "candidate_best_params.json",
    "candidate_best_summary": "candidate_best_summary.json",
    "candidate_retention_best": "candidate_retention_best_params.json",
    "candidate_retention_best_summary": "candidate_retention_best_summary.json",
    "candidate_val_score_best": "candidate_val_score_best_params.json",
    "candidate_val_score_best_summary": "candidate_val_score_best_summary.json",
}

POLICY_FILENAME_BY_NAME = {
    "base_finalist_best": "base_best.json",
    "local_finalist_best": "local_best.json",
    "retention_finalist_best": "retention_best.json",
    "base_finalists_agree": "base_finalists_agree.json",
    "local_finalists_agree": "local_finalists_agree.json",
    "retention_finalists_agree": "retention_finalists_agree.json",
    "base": "ensemble_base.json",
    "local": "ensemble_local.json",
    "retention": "ensemble_retention.json",
}

COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY = {
    "base-finalist-best": "base_finalist_best",
    "base-finalists-agree": "base_finalists_agree",
}


def _assert_finite_strategy_runtime_params(value: Any, *, path: str) -> None:
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"strategy runtime params無法轉換scalar: path={path}, type={type(value).__name__}"
            ) from exc
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"strategy runtime params含non-finite數值: path={path}, value={value}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_strategy_runtime_params(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _assert_finite_strategy_runtime_params(item, path=f"{path}[{idx}]")


def normalize_strategy_param_payload_for_persistence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return strict-JSON strategy-param payload without leaking ranking sentinels.

    Optimizer diagnostics may use +/-inf internally as ordering sentinels. Those values
    are not valid persisted JSON semantics and become ``null`` at the artifact boundary.
    Actual runtime parameter maps are stricter: a non-finite parameter is corruption and
    must fail instead of being silently normalized.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("strategy parameter payload必須是mapping")
    source = dict(payload)

    direct_params = source.get("params")
    if isinstance(direct_params, Mapping):
        _assert_finite_strategy_runtime_params(dict(direct_params), path="params")

    for container_name in ("params_by_effective_date", "params_by_oos_year"):
        container = source.get(container_name)
        if isinstance(container, Mapping):
            for identity, params in container.items():
                if isinstance(params, Mapping):
                    _assert_finite_strategy_runtime_params(
                        dict(params), path=f"{container_name}.{identity}"
                    )

    ensemble = source.get("params_ensemble_by_effective_date")
    if isinstance(ensemble, Mapping):
        for effective_date, members in ensemble.items():
            if not isinstance(members, list):
                continue
            for idx, member in enumerate(members):
                if not isinstance(member, Mapping):
                    continue
                params = member.get("params")
                if isinstance(params, Mapping):
                    _assert_finite_strategy_runtime_params(
                        dict(params),
                        path=f"params_ensemble_by_effective_date.{effective_date}[{idx}].params",
                    )

    direct_ensemble = source.get("params_ensemble")
    if isinstance(direct_ensemble, list):
        for idx, member in enumerate(direct_ensemble):
            if not isinstance(member, Mapping):
                continue
            params = member.get("params")
            if isinstance(params, Mapping):
                _assert_finite_strategy_runtime_params(
                    dict(params), path=f"params_ensemble[{idx}].params"
                )

    folds = source.get("folds")
    if isinstance(folds, list):
        for idx, fold in enumerate(folds):
            if not isinstance(fold, Mapping):
                continue
            params = fold.get("params")
            if isinstance(params, Mapping):
                _assert_finite_strategy_runtime_params(
                    dict(params), path=f"folds[{idx}].params"
                )

    normalized = json_native_value(source)
    if not isinstance(normalized, dict):
        raise ValueError("strategy parameter payload正規化後必須是JSON object")
    return normalized


def normalize_strategy_param_family(value: str) -> str:
    family = str(value or "").strip().lower()
    if family not in STRATEGY_PARAM_FAMILIES:
        raise ValueError(f"不支援的策略參數family: {value!r}")
    return family


def normalize_strategy_param_evaluation_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode == "roos":
        mode = "rolling"
    if mode == "split":
        mode = "oos"
    if mode not in STRATEGY_PARAM_EVALUATION_MODES:
        raise ValueError(f"不支援的策略參數evaluation mode: {value!r}")
    return mode


def normalize_strategy_param_policy(value: str) -> str:
    policy = str(value or "").strip()
    policy = COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.get(policy, policy)
    if policy not in POLICY_FILENAME_BY_NAME:
        raise ValueError(f"不支援的策略參數policy: {value!r}")
    return policy


def _canonical_dir(
    project_root: str | Path, *, strategy_params_root: str | Path | None = None
) -> Path:
    root = (
        Path(strategy_params_root).resolve()
        if strategy_params_root is not None
        else Path(project_root).resolve() / STRATEGY_PARAM_ROOT_RELATIVE
    )
    return root / STRATEGY_PARAM_CANONICAL_DIRNAME


def _canonical_policy_filename(*, family: str, evaluation_mode: str, policy: str) -> str:
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    policy = normalize_strategy_param_policy(policy)
    base = POLICY_FILENAME_BY_NAME[policy]
    # Current OOS/Rolling share one complete cross-time schedule per strategy.
    if mode in STRATEGY_PARAM_SCHEDULE_MODES:
        return f"{family}_{base}"
    # Optimizer Study/Full/Trade products remain named, but no longer create mode folders.
    return f"{family}_{mode}_{base}"


def resolve_strategy_param_dir(project_root: str | Path, *, family: str, evaluation_mode: str, strategy_params_root: str | Path | None = None) -> Path:
    # Compatibility API: family/mode validation is retained, physical storage is flat.
    normalize_strategy_param_family(family)
    normalize_strategy_param_evaluation_mode(evaluation_mode)
    return _canonical_dir(project_root, strategy_params_root=strategy_params_root)


def resolve_strategy_param_artifact_path(
    project_root: str | Path,
    *,
    family: str,
    evaluation_mode: str,
    policy: str,
    strategy_params_root: str | Path | None = None,
) -> Path:
    return _canonical_dir(project_root, strategy_params_root=strategy_params_root) / _canonical_policy_filename(
        family=family, evaluation_mode=evaluation_mode, policy=policy
    )


def resolve_strategy_param_state_dir(
    project_root: str | Path,
    *,
    family: str = "full",
    evaluation_mode: str = "trade",
    strategy_params_root: str | Path | None = None,
) -> Path:
    normalize_strategy_param_family(family)
    normalize_strategy_param_evaluation_mode(evaluation_mode)
    return _canonical_dir(project_root, strategy_params_root=strategy_params_root)


def resolve_strategy_param_state_path(
    project_root: str | Path,
    *,
    artifact: str,
    family: str = "full",
    evaluation_mode: str = "trade",
    strategy_params_root: str | Path | None = None,
) -> Path:
    key = str(artifact or "").strip()
    if key not in STRATEGY_PARAM_STATE_FILENAME_BY_NAME:
        raise ValueError(f"不支援的策略參數state artifact: {artifact!r}")
    return resolve_strategy_param_state_dir(
        project_root, family=family, evaluation_mode=evaluation_mode,
        strategy_params_root=strategy_params_root,
    ) / STRATEGY_PARAM_STATE_FILENAME_BY_NAME[key]


def resolve_strategy_param_manifest_path(project_root: str | Path, *, family: str, evaluation_mode: str, strategy_params_root: str | Path | None = None) -> Path:
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    if mode in STRATEGY_PARAM_SCHEDULE_MODES:
        filename = f"{family}_manifest.json"
    else:
        filename = f"{family}_{mode}_manifest.json"
    return _canonical_dir(project_root, strategy_params_root=strategy_params_root) / filename


def normalize_strategy_param_benchmark_id(value: str) -> str:
    benchmark_id = str(value or "").strip()
    if not benchmark_id:
        raise ValueError("策略參數benchmark_id不可空白")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in benchmark_id):
        raise ValueError(f"策略參數benchmark_id含不支援字元: {value!r}")
    return benchmark_id


def normalize_strategy_param_benchmark_seed(value: int) -> int:
    seed = int(value)
    if seed <= 0:
        raise ValueError("策略參數benchmark seed必須>0")
    return seed


def resolve_strategy_param_optimizer_work_dir(
    project_root: str | Path,
    *,
    scope: str,
    family: str,
    benchmark_id: str | None = None,
    seed: int | None = None,
) -> Path:
    """Resolve the only current Optimizer workspace namespace for strategy schedules.

    Persistent strategy truth always lives under ``models/strategy_params``. Optimizer
    resume/intermediate state always lives under ``outputs/optimizer`` so workspace
    trees can never be mistaken for canonical or benchmark truth.
    """
    family = normalize_strategy_param_family(family)
    normalized_scope = str(scope or "").strip().lower()
    root = Path(project_root).resolve() / STRATEGY_PARAM_WORK_ROOT_RELATIVE
    if normalized_scope == STRATEGY_PARAM_WORK_SCOPE_CANONICAL:
        if benchmark_id not in (None, "") or seed is not None:
            raise ValueError("canonical strategy-param workspace不可帶benchmark identity")
        return root / STRATEGY_PARAM_WORK_SCOPE_CANONICAL / family
    if normalized_scope == STRATEGY_PARAM_WORK_SCOPE_BENCHMARK:
        if benchmark_id in (None, "") or seed is None:
            raise ValueError("benchmark strategy-param workspace需要benchmark_id與seed")
        return (
            root
            / STRATEGY_PARAM_WORK_SCOPE_BENCHMARK
            / normalize_strategy_param_benchmark_id(str(benchmark_id))
            / f"seed_{normalize_strategy_param_benchmark_seed(int(seed))}"
            / family
        )
    raise ValueError(f"不支援的strategy-param workspace scope: {scope!r}")


def resolve_strategy_param_benchmark_dir(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    family: str,
    evaluation_mode: str,
) -> Path:
    normalize_strategy_param_benchmark_seed(seed)
    normalize_strategy_param_family(family)
    normalize_strategy_param_evaluation_mode(evaluation_mode)
    return (
        Path(project_root).resolve()
        / STRATEGY_PARAM_ROOT_RELATIVE
        / STRATEGY_PARAM_BENCHMARK_DIRNAME
        / normalize_strategy_param_benchmark_id(benchmark_id)
    )


def resolve_strategy_param_benchmark_artifact_path(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    family: str,
    evaluation_mode: str,
    policy: str,
) -> Path:
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    if mode not in STRATEGY_PARAM_SCHEDULE_MODES:
        raise ValueError("robustness benchmark strategy params只支援oos/rolling")
    policy = normalize_strategy_param_policy(policy)
    stem = Path(POLICY_FILENAME_BY_NAME[policy]).stem
    seed_value = normalize_strategy_param_benchmark_seed(seed)
    return resolve_strategy_param_benchmark_dir(
        project_root,
        benchmark_id=benchmark_id,
        seed=seed_value,
        family=family,
        evaluation_mode=mode,
    ) / f"{family}_{stem}_seed_{seed_value}.json"


def resolve_strategy_param_benchmark_manifest_path(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    family: str,
    evaluation_mode: str,
) -> Path:
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    if mode not in STRATEGY_PARAM_SCHEDULE_MODES:
        raise ValueError("robustness benchmark strategy params只支援oos/rolling")
    seed_value = normalize_strategy_param_benchmark_seed(seed)
    return resolve_strategy_param_benchmark_dir(
        project_root,
        benchmark_id=benchmark_id,
        seed=seed_value,
        family=family,
        evaluation_mode=mode,
    ) / f"{family}_seed_{seed_value}_manifest.json"


def compute_strategy_param_file_sha256(path: str | Path) -> str:
    """Return byte-level artifact integrity SHA.

    This hash intentionally changes when publication/provenance metadata changes.
    It must not be used as the scientific identity of a replay parameter schedule.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_strategy_param_runtime_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a strategy-param artifact onto replay-affecting semantics only.

    Optimizer exports carry publication/provenance fields such as ``created_at`` and
    diagnostic ``meta``.  Re-publishing the same effective-date parameter schedule
    must not invalidate Strategy Compare or multi-seed observations merely because
    those fields changed.  The scientific identity is therefore the canonical
    runtime schedule consumed by portfolio replay, plus selector/ensemble policy and
    the raw-universe contract that can change the replay universe.
    """

    normalized = normalize_strategy_param_payload_for_persistence(payload)
    selector = str(
        normalized.get("selector")
        or dict(normalized.get("meta") or {}).get("selector")
        or ""
    ).strip()
    universe_min_rows = resolve_raw_universe_required_min_rows(normalized)

    if is_active_param_ensemble_payload(normalized):
        schedule = build_active_param_ensemble_schedule(normalized)
        return {
            "identity_schema": STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA,
            "kind": "active_param_ensemble",
            "mode": resolve_active_param_ensemble_mode(normalized),
            "selector": selector,
            "raw_universe_required_min_rows": universe_min_rows,
            "ensemble_policy": get_active_param_ensemble_policy(normalized),
            "schedule": [
                {
                    "effective_start": str(record.get("effective_date_text") or ""),
                    "effective_end": str(record.get("effective_end_date_text") or ""),
                    # Replay consumes the ordered member parameter sets plus the
                    # member key.  Optimizer provenance such as seed, selected
                    # trial, scores/ranks and diagnostics must not turn a pure
                    # publication refresh into a new scientific schedule.
                    "members": [
                        {
                            "member_index": int(member.get("member_index") or idx),
                            "params": json_native_value(dict(member.get("params") or {})),
                        }
                        for idx, member in enumerate(
                            list(record.get("members") or []), start=1
                        )
                    ],
                }
                for record in schedule
            ],
        }

    if is_rolling_oos_param_set_payload(normalized):
        schedule = build_active_param_schedule(normalized)
        return {
            "identity_schema": STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA,
            "kind": "rolling_param_schedule",
            "mode": "rolling",
            "selector": selector,
            "raw_universe_required_min_rows": universe_min_rows,
            "schedule": [
                {
                    "effective_start": str(record.get("effective_date_text") or ""),
                    "effective_end": str(record.get("effective_end_date_text") or ""),
                    "params": json_native_value(dict(record.get("params") or {})),
                }
                for record in schedule
            ],
        }

    direct_params = normalized.get("params")
    if isinstance(direct_params, Mapping):
        return {
            "identity_schema": STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA,
            "kind": "static_params",
            "selector": selector,
            "raw_universe_required_min_rows": universe_min_rows,
            "params": json_native_value(dict(direct_params)),
        }

    raise ValueError("無法建立strategy parameter runtime scientific identity")


def compute_strategy_param_scientific_sha256(path: str | Path) -> str:
    """Return stable replay-scientific identity, excluding publication metadata."""

    payload = load_json_strict(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"strategy parameter JSON根層必須是object: {path}")
    return canonical_json_sha256(build_strategy_param_runtime_identity_payload(payload))


def load_strategy_param_manifest(project_root: str | Path, *, family: str, evaluation_mode: str) -> dict[str, Any] | None:
    path = resolve_strategy_param_manifest_path(project_root, family=family, evaluation_mode=evaluation_mode)
    if not path.is_file():
        return None
    try:
        payload = load_json_strict(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def freeze_strategy_param_payload_for_period(
    payload: Mapping[str, Any], *, start_date: str, end_date: str
) -> dict[str, Any]:
    """Return an in-memory frozen view of one rolling strategy schedule.

    No file is written.  The latest member legally effective on ``start_date`` is
    used for the entire requested period.  This makes OOS a consumption mode over
    the same canonical schedule used by Rolling.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("strategy parameter payload必須是mapping")
    source = copy.deepcopy(dict(payload))
    start = str(start_date)[:10]
    end = str(end_date)[:10]
    if end < start:
        raise ValueError("frozen strategy parameter period不合法")

    if not is_active_param_ensemble_payload(source) or resolve_active_param_ensemble_mode(source) != ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING:
        raise ValueError("OOS frozen view只接受rolling active-param ensemble")
    mapping = dict(source.get("params_ensemble_by_effective_date") or {})
    eligible = sorted(date for date in mapping if str(date)[:10] <= start)
    if not eligible:
        raise ValueError(f"canonical strategy schedule在{start}前沒有合法effective member")
    effective = eligible[-1]
    members = normalize_seed_ensemble_members(mapping[effective])
    if not members:
        raise ValueError(f"canonical strategy schedule effective member無效: {effective}")

    frozen = copy.deepcopy(source)
    frozen["params_ensemble_by_effective_date"] = {start: members}
    # Keep any single-member compatibility mapping consistent when present.
    if isinstance(frozen.get("params_by_effective_date"), Mapping):
        raw = dict(frozen.get("params_by_effective_date") or {})
        if effective in raw:
            frozen["params_by_effective_date"] = {start: copy.deepcopy(raw[effective])}
    if isinstance(frozen.get("params_by_oos_year"), Mapping):
        raw_years = dict(frozen.get("params_by_oos_year") or {})
        effective_year = str(effective)[:4]
        if effective_year in raw_years:
            frozen["params_by_oos_year"] = {str(start)[:4]: copy.deepcopy(raw_years[effective_year])}
    frozen["folds"] = [{
        "effective_start": start,
        "effective_end": end,
        "oos_start_date": start,
        "oos_end_date": end,
    }]
    frozen.setdefault("meta", {}).update({
        "evaluation_view": "frozen_oos",
        "source_effective_date": str(effective),
        "freeze_start_date": start,
        "freeze_end_date": end,
    })
    frozen.setdefault("summary", {}).update({
        "folds": 1,
        "oos_period": f"{start}~{end}",
        "oos_start_date": start,
        "oos_end_date": end,
    })
    # Assert the generated view actually covers the requested period.
    actual_start, actual_end = get_active_param_ensemble_date_range(frozen)
    if str(actual_start) != start or str(actual_end) != end:
        raise ValueError(
            "frozen strategy parameter coverage不一致: "
            f"expected={start}~{end}, actual={actual_start}~{actual_end}"
        )
    return frozen


__all__ = [
    "STRATEGY_PARAM_ARTIFACT_SCHEMA_VERSION",
    "STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA",
    "STRATEGY_PARAM_ROOT_RELATIVE",
    "STRATEGY_PARAM_CANONICAL_DIRNAME",
    "STRATEGY_PARAM_BENCHMARK_DIRNAME",
    "STRATEGY_PARAM_FAMILIES",
    "STRATEGY_PARAM_EVALUATION_MODES",
    "STRATEGY_PARAM_SCHEDULE_MODES",
    "STRATEGY_PARAM_STATE_FILENAME_BY_NAME",
    "POLICY_FILENAME_BY_NAME",
    "COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY",
    "normalize_strategy_param_family",
    "normalize_strategy_param_evaluation_mode",
    "normalize_strategy_param_policy",
    "normalize_strategy_param_payload_for_persistence",
    "normalize_strategy_param_benchmark_id",
    "STRATEGY_PARAM_WORK_ROOT_RELATIVE",
    "STRATEGY_PARAM_WORK_SCOPE_CANONICAL",
    "STRATEGY_PARAM_WORK_SCOPE_BENCHMARK",
    "normalize_strategy_param_benchmark_seed",
    "resolve_strategy_param_dir",
    "resolve_strategy_param_artifact_path",
    "resolve_strategy_param_state_dir",
    "resolve_strategy_param_state_path",
    "resolve_strategy_param_manifest_path",
    "resolve_strategy_param_optimizer_work_dir",
    "resolve_strategy_param_benchmark_dir",
    "resolve_strategy_param_benchmark_artifact_path",
    "resolve_strategy_param_benchmark_manifest_path",
    "compute_strategy_param_file_sha256",
    "compute_strategy_param_scientific_sha256",
    "build_strategy_param_runtime_identity_payload",
    "load_strategy_param_manifest",
    "freeze_strategy_param_payload_for_period",
]
