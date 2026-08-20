"""Canonical optimizer-owned strategy-parameter artifact service.

Current consumers request a parameter identity (family/mode/policy).  This service
owns migration/materialization into ``models/strategy_params`` and records the
optimizer training-policy identity.  Research code must not own seed/trial/path
settings for current parameter artifacts.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config.training_policy import get_strategy_parameter_training_policy_snapshot
from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
    ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
    build_active_param_ensemble_schedule,
    get_active_param_ensemble_date_range,
    is_active_param_ensemble_payload,
    resolve_active_param_ensemble_mode,
)
from core.seed_ensemble_policy import normalize_seed_ensemble_members
from core.strategy_param_artifacts import (
    POLICY_FILENAME_BY_NAME,
    compute_strategy_param_file_sha256,
    normalize_strategy_param_evaluation_mode,
    normalize_strategy_param_family,
    normalize_strategy_param_policy,
    resolve_strategy_param_artifact_path,
    resolve_strategy_param_dir,
    resolve_strategy_param_state_path,
    STRATEGY_PARAM_STATE_FILENAME_BY_NAME,
)
from services.optimizer.strategy_param_repository import refresh_strategy_parameter_manifest


def _project_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _legacy_policy_filename(policy: str, *, evaluation_mode: str) -> str:
    normalized = normalize_strategy_param_policy(policy)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    base = POLICY_FILENAME_BY_NAME[normalized]
    if mode == "study":
        if normalized in {"base", "local", "retention"}:
            return f"{normalized}.json"
        return base
    if mode in {"full", "trade"}:
        if normalized in {"base", "local", "retention"}:
            return f"{mode}_ensemble_{normalized}.json"
        return f"{mode}_{base}"
    prefix = "oos" if mode == "oos" else "roos"
    if normalized in {"base", "local", "retention"}:
        return f"{prefix}_ensemble_{normalized}.json"
    return f"{prefix}_{base}"


def _legacy_candidates(root: Path, *, family: str, evaluation_mode: str, policy: str) -> tuple[Path, ...]:
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    policy = normalize_strategy_param_policy(policy)
    legacy_filename = _legacy_policy_filename(policy, evaluation_mode=mode)
    candidates: list[Path] = []
    if family == "full":
        candidates.append(root / "models" / legacy_filename)
        if mode == "rolling":
            candidates.append(root / "models" / "research" / "breakout_quality" / "strategy_compare" / "extending_full_roos" / "active_params" / legacy_filename)
        elif mode == "oos":
            candidates.append(root / "models" / "research" / "breakout_quality" / "strategy_compare" / "oos_full_roos" / "active_params" / _legacy_policy_filename(policy, evaluation_mode="rolling"))
    else:
        if mode == "rolling":
            candidates.extend([
                root / "models" / "research" / "breakout_quality" / "strategy_compare" / "extending_min_roos" / "active_params" / _legacy_policy_filename(policy, evaluation_mode="rolling"),
                root / "models" / "research" / "breakout_quality" / "binary_dl_filter_param_adaptation" / "risk_only_rolling" / "p2_dl_off_trained" / "active_params" / _legacy_policy_filename(policy, evaluation_mode="rolling"),
            ])
        elif mode == "oos":
            candidates.append(root / "models" / "research" / "breakout_quality" / "strategy_compare" / "oos_min_roos" / "active_params" / _legacy_policy_filename(policy, evaluation_mode="rolling"))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return tuple(unique)



def _normalize_legacy_period_date(value: Any, *, is_end: bool = False) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"latest", "auto", "none", "null"}:
        return ""
    if len(text) == 4 and text.isdigit():
        return f"{text}-12-31" if is_end else f"{text}-01-01"
    return text[:10]


def _legacy_static_oos_period(payload: dict[str, Any]) -> tuple[str, str]:
    meta = dict(payload.get("meta") or {})
    workflow = dict(meta.get("walk_forward_policy") or {})
    summary = dict(payload.get("summary") or {})
    period = str(summary.get("oos_period") or payload.get("oos_period") or "").strip()
    period_start = period_end = ""
    if "~" in period:
        period_start, period_end = [part.strip() for part in period.split("~", 1)]
    start_candidates = (
        meta.get("oos_start_date"),
        payload.get("oos_start_date"),
        summary.get("oos_start_date"),
        workflow.get("oos_start_date"),
        period_start,
        workflow.get("oos_start_year"),
    )
    end_candidates = (
        meta.get("oos_end_date"),
        meta.get("latest_data_date"),
        payload.get("oos_end_date"),
        summary.get("oos_end_date"),
        workflow.get("oos_end_date"),
        period_end,
        workflow.get("oos_end_year"),
    )
    start = next(
        (resolved for raw in start_candidates if (resolved := _normalize_legacy_period_date(raw))),
        "2021-01-01",
    )
    end = next(
        (resolved for raw in end_candidates if (resolved := _normalize_legacy_period_date(raw, is_end=True))),
        "",
    )
    return start, end


def _resolve_rolling_policy_end_date(root: Path, *, family: str, policy: str) -> str:
    candidates = [
        resolve_strategy_param_artifact_path(
            root, family=family, evaluation_mode="rolling", policy=policy
        )
    ]
    candidates.extend(
        _legacy_candidates(root, family=family, evaluation_mode="rolling", policy=policy)
    )
    for source in candidates:
        if not source.is_file():
            continue
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not is_active_param_ensemble_payload(payload):
            continue
        if resolve_active_param_ensemble_mode(payload) != ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING:
            continue
        try:
            _start, end = get_active_param_ensemble_date_range(payload)
        except (TypeError, ValueError, KeyError):
            continue
        if end:
            return str(end)
    return ""


def _materialize_legacy_candidate(
    *,
    root: Path,
    family: str,
    policy: str,
    candidate: Path,
    target: Path,
    evaluation_mode: str,
) -> bool:
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    if mode != "oos":
        shutil.copy2(candidate, target)
        return True
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if is_active_param_ensemble_payload(payload) and resolve_active_param_ensemble_mode(payload) == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
        members = normalize_seed_ensemble_members(payload.get("params_ensemble"))
        if not members:
            return False
        start, end = _legacy_static_oos_period(payload)
        if not end:
            end = _resolve_rolling_policy_end_date(root, family=family, policy=policy)
        if not end:
            return False
        rolling = dict(payload)
        rolling["mode"] = ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING
        rolling["type"] = "canonical_oos_frozen_param_set"
        rolling.pop("params_ensemble", None)
        rolling["params_ensemble_by_effective_date"] = {start: members}
        rolling["folds"] = [{
            "effective_start": start,
            "effective_end": end,
            "oos_start_date": start,
            "oos_end_date": end,
        }]
        rolling.setdefault("meta", {}).update({
            "canonical_oos_freeze": True,
            "source_static_optimizer_artifact": candidate.name,
            "oos_start_date": start,
            "oos_end_date": end,
        })
        rolling.setdefault("summary", {}).update({
            "folds": 1,
            "oos_period": f"{start}~{end}",
            "oos_start_date": start,
            "oos_end_date": end,
        })
        _write_json(target, rolling)
        return True
    shutil.copy2(candidate, target)
    return True



def migrate_legacy_trade_state_artifacts(project_root: str | Path) -> dict[str, Any]:
    """Copy legacy root Trade runtime state into the canonical repository once."""
    root = Path(project_root).resolve()
    legacy_by_artifact = {
        "active": "run_best_params.json",
        "active_summary": "run_best_summary.json",
        "candidate_best": "candidate_best_params.json",
        "candidate_best_summary": "candidate_best_summary.json",
        "candidate_retention_best": "candidate_retention_best_params.json",
        "candidate_retention_best_summary": "candidate_retention_best_summary.json",
        "candidate_val_score_best": "candidate_val_score_best_params.json",
        "candidate_val_score_best_summary": "candidate_val_score_best_summary.json",
    }
    migrated: dict[str, str] = {}
    reused: list[str] = []
    for artifact in STRATEGY_PARAM_STATE_FILENAME_BY_NAME:
        target = resolve_strategy_param_state_path(root, artifact=artifact)
        if target.is_file():
            reused.append(artifact)
            continue
        legacy_name = legacy_by_artifact[artifact]
        source = root / "models" / legacy_name
        if not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        migrated[artifact] = _project_relative(root, source)
    manifest = refresh_strategy_parameter_manifest(root, family="full", evaluation_mode="trade")
    return {
        "family": "full",
        "evaluation_mode": "trade",
        "migrated": migrated,
        "reused": reused,
        "manifest_path": _project_relative(root, manifest),
    }

def migrate_legacy_strategy_parameter_artifacts(project_root: str | Path, *, family: str, evaluation_mode: str) -> dict[str, Any]:
    """Migrate existing optimizer/research legacy files into the canonical repository.

    Migration is content-preserving: no optimizer search, refit, selection, seed choice,
    or policy recomputation is performed.
    """
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    target_dir = resolve_strategy_param_dir(root, family=family, evaluation_mode=mode)
    target_dir.mkdir(parents=True, exist_ok=True)
    migrated: dict[str, str] = {}
    reused: list[str] = []
    sources: dict[str, dict[str, Any]] = {}
    for policy in POLICY_FILENAME_BY_NAME:
        target = resolve_strategy_param_artifact_path(root, family=family, evaluation_mode=mode, policy=policy)
        if target.is_file():
            reused.append(policy)
            continue
        for candidate in _legacy_candidates(root, family=family, evaluation_mode=mode, policy=policy):
            if not candidate.is_file():
                continue
            if not _materialize_legacy_candidate(
                root=root,
                family=family,
                policy=policy,
                candidate=candidate,
                target=target,
                evaluation_mode=mode,
            ):
                continue
            sources[policy] = {
                "migration_only": True,
                "path": _project_relative(root, candidate),
                "sha256": compute_strategy_param_file_sha256(candidate),
            }
            migrated[policy] = _project_relative(root, candidate)
            break
    state_migration = None
    if family == "full" and mode == "trade":
        state_migration = migrate_legacy_trade_state_artifacts(root)
    manifest_path = refresh_strategy_parameter_manifest(root, family=family, evaluation_mode=mode, source_records=sources)
    return {
        "family": family,
        "evaluation_mode": mode,
        "migrated": migrated,
        "reused": reused,
        "manifest_path": _project_relative(root, manifest_path),
        "state_migration": state_migration,
    }





def collect_legacy_root_strategy_parameter_cleanup_plan(project_root: str | Path) -> dict[str, Any]:
    """Return legacy root strategy JSONs that are safe to delete after migration."""
    root = Path(project_root).resolve()
    models_root = root / "models"
    mappings: dict[Path, Path] = {}
    for mode in ("study", "full", "rolling", "oos", "trade"):
        for policy in POLICY_FILENAME_BY_NAME:
            source = models_root / _legacy_policy_filename(policy, evaluation_mode=mode)
            target = resolve_strategy_param_artifact_path(
                root, family="full", evaluation_mode=mode, policy=policy
            )
            mappings[source] = target
    state_legacy = {
        "active": "run_best_params.json",
        "active_summary": "run_best_summary.json",
        "candidate_best": "candidate_best_params.json",
        "candidate_best_summary": "candidate_best_summary.json",
        "candidate_retention_best": "candidate_retention_best_params.json",
        "candidate_retention_best_summary": "candidate_retention_best_summary.json",
        "candidate_val_score_best": "candidate_val_score_best_params.json",
        "candidate_val_score_best_summary": "candidate_val_score_best_summary.json",
    }
    for artifact, legacy_name in state_legacy.items():
        mappings[models_root / legacy_name] = resolve_strategy_param_state_path(
            root, artifact=artifact
        )
    removable: list[dict[str, str]] = []
    blockers: list[dict[str, str]] = []
    for source, target in sorted(mappings.items(), key=lambda item: item[0].name.lower()):
        if not source.is_file():
            continue
        record = {
            "source": _project_relative(root, source),
            "target": _project_relative(root, target),
        }
        if target.is_file():
            removable.append(record)
        else:
            blockers.append(record)
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "removable": removable,
        "blockers": blockers,
    }

def migrate_all_legacy_strategy_parameter_artifacts(project_root: str | Path) -> dict[str, Any]:
    """One-shot content-preserving migration of every legacy current parameter family."""
    root = Path(project_root).resolve()
    results: list[dict[str, Any]] = []
    for mode in ("study", "full", "oos", "rolling", "trade"):
        results.append(
            migrate_legacy_strategy_parameter_artifacts(
                root, family="full", evaluation_mode=mode
            )
        )
    for mode in ("oos", "rolling"):
        results.append(
            migrate_legacy_strategy_parameter_artifacts(
                root, family="min", evaluation_mode=mode
            )
        )
    return {
        "status": "PASS",
        "project_root": str(root),
        "results": results,
    }

def _freeze_rolling_policy_to_oos(
    root: Path,
    *,
    family: str,
    policy: str,
    source_path: Path,
    target_path: Path,
    freeze_effective_date: str = "2021-01-01",
    freeze_cutoff_date: str = "2020-12-31",
) -> dict[str, Any]:
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"canonical rolling策略參數無法讀取: {source_path}") from exc
    if not isinstance(payload, dict) or not is_active_param_ensemble_payload(payload):
        raise ValueError(f"canonical rolling策略參數格式不合法: {source_path}")
    if resolve_active_param_ensemble_mode(payload) != ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING:
        raise ValueError(f"OOS freeze來源必須是rolling active-param ensemble: {source_path}")
    schedule = build_active_param_ensemble_schedule(payload)
    selected = [
        item for item in schedule
        if str(item.get("effective_date") or "") == str(freeze_effective_date)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"OOS freeze要求唯一{freeze_effective_date} effective member: "
            f"family={family}, policy={policy}, found={len(selected)}"
        )
    mapping = dict(payload.get("params_ensemble_by_effective_date") or {})
    members = mapping.get(str(freeze_effective_date))
    if not isinstance(members, list) or not members:
        raise ValueError(f"OOS freeze member缺失: {freeze_effective_date}")
    _source_start, source_end = get_active_param_ensemble_date_range(payload)
    frozen = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    frozen["type"] = "canonical_oos_frozen_param_set"
    frozen["mode"] = ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING
    frozen["params_ensemble_by_effective_date"] = {str(freeze_effective_date): members}
    frozen.pop("params_by_effective_date", None)
    frozen.pop("params_by_oos_year", None)
    frozen["folds"] = [{
        "effective_start": str(freeze_effective_date),
        "effective_end": str(source_end),
        "oos_start_date": str(freeze_effective_date),
        "oos_end_date": str(source_end),
        "source_effective_date": str(freeze_effective_date),
        "freeze_cutoff_date": str(freeze_cutoff_date),
        "oos_frozen_parameter_reuse": True,
    }]
    frozen.setdefault("meta", {}).update({
        "first_oos_date": str(freeze_effective_date),
        "last_oos_date": str(source_end),
        "oos_parameter_mode": "fixed_information_cutoff",
        "freeze_effective_date": str(freeze_effective_date),
        "freeze_cutoff_date": str(freeze_cutoff_date),
        "source_effective_date": str(freeze_effective_date),
        "canonical_optimizer_derivation": True,
    })
    frozen.setdefault("summary", {}).update({
        "folds": 1,
        "oos_period": f"{freeze_effective_date}~{source_end}",
        "oos_start_date": str(freeze_effective_date),
        "oos_end_date": str(source_end),
        "oos_parameter_mode": "fixed_information_cutoff",
        "freeze_cutoff_date": str(freeze_cutoff_date),
        "source_effective_date": str(freeze_effective_date),
    })
    _write_json(target_path, frozen)
    return {
        "producer": "optimizer",
        "derivation": "freeze_rolling_effective_member",
        "path": _project_relative(root, source_path),
        "sha256": compute_strategy_param_file_sha256(source_path),
        "freeze_effective_date": str(freeze_effective_date),
        "freeze_cutoff_date": str(freeze_cutoff_date),
    }


def _build_canonical_min_rolling(root: Path) -> dict[str, Any]:
    """Run the optimizer-owned P2 Min search into the canonical repository."""
    # Min is a constrained search derived from the same PIT-safe Full rolling schedule.
    # Resolve/migrate that canonical Full truth first; do not fall back to a Research path.
    ensure_strategy_parameter_artifact(
        root, family="full", evaluation_mode="rolling", policy="base_finalist_best"
    )
    from config.breakout_quality import get_breakout_quality_workflow_settings
    from config.execution_policy import (
        DEFAULT_FIXED_RISK,
        DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS,
        DEFAULT_PORTFOLIO_ROTATION,
    )
    from core.dataset_profiles import DEFAULT_DATASET_PROFILE
    from services.optimizer.strategy_param_training import prepare_strategy_parameter_source

    workflow = get_breakout_quality_workflow_settings()
    training_policy = get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling")
    return prepare_strategy_parameter_source(
        project_root=root,
        dataset=str(DEFAULT_DATASET_PROFILE),
        filter_id=str(workflow.filter_id),
        model_architecture=str(workflow.model_architecture),
        experiment_profile=str(workflow.experiment_profile),
        param_policy="base-finalist-best",
        parameter_set="p2",
        trials_per_fold=int(training_policy["trials_per_fold"]),
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        build_binary_pit=False,
        binary_pit_resume=True,
        resume_parameter_training=True,
        quiet=False,
        canonical_strategy_param_family="min",
    )

def ensure_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    family: str,
    evaluation_mode: str,
    policy: str,
) -> dict[str, Any]:
    """Resolve a canonical artifact, migrating an existing legacy truth if necessary.

    Missing artifacts are not recomputed here by Research.  The caller can report the
    canonical Optimizer work type as the required producer.  This prevents a second
    Research-owned optimizer policy from appearing.
    """
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    policy = normalize_strategy_param_policy(policy)
    target = resolve_strategy_param_artifact_path(root, family=family, evaluation_mode=mode, policy=policy)
    action = "REUSE"
    migration = None
    if not target.is_file():
        migration = migrate_legacy_strategy_parameter_artifacts(root, family=family, evaluation_mode=mode)
        action = "MIGRATE" if target.is_file() else "MISSING"
    derived_source = None
    if not target.is_file() and mode == "rolling" and family == "min":
        _build_canonical_min_rolling(root)
        action = "BUILD" if target.is_file() else "MISSING"
    if not target.is_file() and mode == "oos":
        rolling = ensure_strategy_parameter_artifact(
            root, family=family, evaluation_mode="rolling", policy=policy
        )
        source_path = Path(rolling["path"]).resolve()
        derived_source = _freeze_rolling_policy_to_oos(
            root,
            family=family,
            policy=policy,
            source_path=source_path,
            target_path=target,
        )
        action = "DERIVE" if target.is_file() else "MISSING"
    if not target.is_file():
        raise FileNotFoundError(
            "缺少canonical策略參數工件；Optimizer service無法由既有truth建立："
            f"family={family}, mode={mode}, policy={policy}, target={_project_relative(root, target)}"
        )
    source_records = None if derived_source is None else {policy: derived_source}
    manifest = refresh_strategy_parameter_manifest(
        root, family=family, evaluation_mode=mode, source_records=source_records
    )
    return {
        "action": action,
        "family": family,
        "evaluation_mode": mode,
        "policy": policy,
        "path": target,
        "manifest_path": manifest,
        "sha256": compute_strategy_param_file_sha256(target),
        "migration": migration,
    }


__all__ = [
    "ensure_strategy_parameter_artifact",
    "migrate_legacy_strategy_parameter_artifacts",
    "migrate_legacy_trade_state_artifacts",
    "migrate_all_legacy_strategy_parameter_artifacts",
    "collect_legacy_root_strategy_parameter_cleanup_plan",
    "refresh_strategy_parameter_manifest",
]
