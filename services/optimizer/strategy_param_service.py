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

from config.training_policy import (
    get_strategy_parameter_training_policy_snapshot,
    get_robustness_benchmark_policy_snapshot,
)
from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
    ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
    build_active_param_ensemble_schedule,
    get_active_param_ensemble_date_range,
    is_active_param_ensemble_payload,
    resolve_active_param_ensemble_mode,
)
from core.seed_ensemble_policy import normalize_seed_ensemble_members
from core.file_integrity import canonical_json_sha256
from core.strategy_param_artifacts import (
    POLICY_FILENAME_BY_NAME,
    compute_strategy_param_file_sha256,
    normalize_strategy_param_evaluation_mode,
    normalize_strategy_param_family,
    normalize_strategy_param_policy,
    resolve_strategy_param_artifact_path,
    resolve_strategy_param_benchmark_artifact_path,
    resolve_strategy_param_benchmark_dir,
    resolve_strategy_param_benchmark_manifest_path,
    resolve_strategy_param_dir,
    resolve_strategy_param_manifest_path,
    resolve_strategy_param_state_path,
    STRATEGY_PARAM_STATE_FILENAME_BY_NAME,
)
from services.optimizer.strategy_param_repository import (
    refresh_strategy_parameter_benchmark_manifest,
    refresh_strategy_parameter_manifest,
)


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





def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def collect_legacy_root_strategy_parameter_cleanup_plan(project_root: str | Path) -> dict[str, Any]:
    """Return only root legacy strategy JSONs proven safe to delete.

    A legacy file is removable only when its canonical target is pinned by the
    Optimizer-owned manifest and either (a) the bytes still match or (b) the
    manifest explicitly preserves the exact migration source path + SHA.  Mere
    target existence is intentionally insufficient.
    """
    root = Path(project_root).resolve()
    models_root = root / "models"
    mappings: list[dict[str, Any]] = []
    for mode in ("study", "full", "rolling", "oos", "trade"):
        for policy in POLICY_FILENAME_BY_NAME:
            mappings.append({
                "kind": "policy",
                "family": "full",
                "evaluation_mode": mode,
                "policy": policy,
                "source": models_root / _legacy_policy_filename(policy, evaluation_mode=mode),
                "target": resolve_strategy_param_artifact_path(
                    root, family="full", evaluation_mode=mode, policy=policy
                ),
            })
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
        mappings.append({
            "kind": "state",
            "family": "full",
            "evaluation_mode": "trade",
            "state_artifact": artifact,
            "source": models_root / legacy_name,
            "target": resolve_strategy_param_state_path(root, artifact=artifact),
        })

    removable: list[dict[str, str]] = []
    blockers: list[dict[str, str]] = []
    for item in sorted(mappings, key=lambda row: str(row["source"]).lower()):
        source = Path(item["source"])
        target = Path(item["target"])
        if not source.is_file():
            continue
        source_rel = _project_relative(root, source)
        target_rel = _project_relative(root, target)
        base_record = {"source": source_rel, "target": target_rel}
        if not target.is_file():
            blockers.append({**base_record, "reason": "canonical_target_missing"})
            continue

        family = str(item["family"])
        mode = str(item["evaluation_mode"])
        manifest_path = resolve_strategy_param_manifest_path(
            root, family=family, evaluation_mode=mode
        )
        manifest = _load_json_object(manifest_path)
        if manifest is None:
            blockers.append({**base_record, "reason": "canonical_manifest_missing_or_invalid"})
            continue
        if (
            str(manifest.get("producer") or "") != "optimizer"
            or str(manifest.get("family") or "") != family
            or str(manifest.get("evaluation_mode") or "") != mode
        ):
            blockers.append({**base_record, "reason": "canonical_manifest_identity_mismatch"})
            continue

        target_sha = compute_strategy_param_file_sha256(target)
        source_sha = compute_strategy_param_file_sha256(source)
        if item["kind"] == "policy":
            policy = str(item["policy"])
            manifest_record = dict(dict(manifest.get("artifacts") or {}).get(policy) or {})
        else:
            state_name = str(item["state_artifact"])
            manifest_record = dict(dict(manifest.get("state_artifacts") or {}).get(state_name) or {})
        if (
            str(manifest_record.get("path") or "") != target_rel
            or str(manifest_record.get("sha256") or "") != target_sha
        ):
            blockers.append({**base_record, "reason": "canonical_manifest_target_not_pinned"})
            continue

        evidence = "content_sha_match" if source_sha == target_sha else ""
        if not evidence and item["kind"] == "policy":
            source_record = dict(manifest_record.get("source") or {})
            if (
                bool(source_record.get("migration_only"))
                and str(source_record.get("path") or "") == source_rel
                and str(source_record.get("sha256") or "") == source_sha
            ):
                evidence = "manifest_migration_lineage"
        if not evidence:
            blockers.append({**base_record, "reason": "legacy_source_not_proven_migrated"})
            continue

        removable.append({**base_record, "evidence": evidence})

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

def finalize_legacy_strategy_parameter_migration(project_root: str | Path) -> dict[str, Any]:
    """Run the explicit one-time migration and return a READY-gated cleanup plan.

    This function never deletes user files.  Deletion remains an explicit local
    operation after the returned cleanup plan is READY.
    """
    migration = migrate_all_legacy_strategy_parameter_artifacts(project_root)
    cleanup = collect_legacy_root_strategy_parameter_cleanup_plan(project_root)
    return {
        "status": "READY_FOR_CLEANUP" if cleanup["status"] == "READY" else "BLOCKED",
        "migration": migration,
        "cleanup": cleanup,
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
    coverage_end_date: str | None = None,
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
    source_end = str(coverage_end_date or source_end)
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
    """Resolve/build only the canonical current strategy-parameter truth.

    Legacy discovery is deliberately excluded from this current execution path.
    One-time migration must be performed explicitly with
    ``migrate_all_legacy_strategy_parameter_artifacts`` before legacy root cleanup.
    """
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    policy = normalize_strategy_param_policy(policy)
    target = resolve_strategy_param_artifact_path(root, family=family, evaluation_mode=mode, policy=policy)
    action = "REUSE"
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
            "缺少canonical策略參數工件；current流程不掃描legacy root。"
            "請先由Optimizer正式流程建立，或執行一次性Strategy Parameter SSOT migration："
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
        "migration": None,
    }


def _benchmark_optimizer_work_root(
    root: Path, *, benchmark_id: str, seed: int, family: str
) -> Path:
    rolling_dir = resolve_strategy_param_benchmark_dir(
        root,
        benchmark_id=benchmark_id,
        seed=int(seed),
        family=family,
        evaluation_mode="rolling",
    )
    # .../seed_<seed>/<family>/rolling -> .../seed_<seed>/_optimizer_work/<family>
    seed_root = rolling_dir.parents[1]
    return seed_root / "_optimizer_work" / str(family)


def _copy_json_payload(source: Path, target: Path) -> dict[str, Any]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"benchmark策略參數source無法讀取: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark策略參數source必須是JSON object: {source}")
    _write_json(target, payload)
    return payload


def _stitch_benchmark_rolling_payload(
    *, initial_path: Path, tail_path: Path | None, target_path: Path, comparison_end_date: str
) -> None:
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    if not isinstance(initial, dict) or not is_active_param_ensemble_payload(initial):
        raise ValueError("benchmark 2021初始策略參數格式不合法")
    merged = json.loads(json.dumps(initial, ensure_ascii=False, default=str))
    initial_mapping = dict(initial.get("params_ensemble_by_effective_date") or {})
    if "2021-01-01" not in initial_mapping:
        raise ValueError("benchmark初始策略參數缺少2021-01-01 effective member")
    combined = {"2021-01-01": initial_mapping["2021-01-01"]}
    tail = None
    if tail_path is not None:
        tail = json.loads(tail_path.read_text(encoding="utf-8"))
        if not isinstance(tail, dict) or not is_active_param_ensemble_payload(tail):
            raise ValueError("benchmark Rolling tail策略參數格式不合法")
        for key, value in dict(tail.get("params_ensemble_by_effective_date") or {}).items():
            if str(key) >= "2022-01-01":
                combined[str(key)] = value
    merged["params_ensemble_by_effective_date"] = {
        key: combined[key] for key in sorted(combined)
    }
    for optional_field in ("params_by_effective_date", "params_by_oos_year"):
        out = {}
        for source in (initial, tail or {}):
            for key, value in dict(source.get(optional_field) or {}).items():
                text = str(key)
                if text.startswith("2021") or text >= "2022":
                    out[text] = value
        if out:
            merged[optional_field] = {key: out[key] for key in sorted(out)}
        else:
            merged.pop(optional_field, None)
    folds = []
    for source in (initial, tail or {}):
        for item in list(source.get("folds") or []):
            if not isinstance(item, dict):
                continue
            start = str(item.get("effective_start") or item.get("oos_start_date") or "")
            if start and start >= "2021-01-01":
                folds.append(json.loads(json.dumps(item, ensure_ascii=False, default=str)))
    folds.sort(key=lambda item: str(item.get("effective_start") or item.get("oos_start_date") or ""))
    if folds:
        merged["folds"] = folds
    merged.setdefault("meta", {}).update({
        "first_oos_date": "2021-01-01",
        "last_oos_date": str(comparison_end_date),
        "robustness_benchmark_same_seed_initial": True,
    })
    merged.setdefault("summary", {}).update({
        "folds": len(combined),
        "oos_start_date": "2021-01-01",
        "oos_end_date": str(comparison_end_date),
        "oos_period": f"2021-01-01~{comparison_end_date}",
        "robustness_benchmark_same_seed_initial": True,
    })
    _write_json(target_path, merged)



def _benchmark_effective_member_sha256(path: Path, effective_date: str = "2021-01-01") -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = dict(payload.get("params_ensemble_by_effective_date") or {})
    members = mapping.get(str(effective_date))
    if not isinstance(members, list) or not members:
        raise ValueError(f"benchmark策略參數缺少{effective_date} member: {path}")
    return canonical_json_sha256(members)


def _validate_benchmark_oos_rolling_initial_alignment(
    root: Path, *, benchmark_id: str, seed: int, family: str, policy: str
) -> str | None:
    oos = resolve_strategy_param_benchmark_artifact_path(
        root, benchmark_id=benchmark_id, seed=seed, family=family,
        evaluation_mode="oos", policy=policy,
    )
    rolling = resolve_strategy_param_benchmark_artifact_path(
        root, benchmark_id=benchmark_id, seed=seed, family=family,
        evaluation_mode="rolling", policy=policy,
    )
    if not oos.is_file() or not rolling.is_file():
        return None
    left = _benchmark_effective_member_sha256(oos)
    right = _benchmark_effective_member_sha256(rolling)
    if left != right:
        raise RuntimeError(
            "robustness benchmark OOS/Rolling 2021起始param不一致: "
            f"family={family}, seed={seed}, oos={left}, rolling={right}"
        )
    return left



def _benchmark_parameter_coverage_end(path: Path) -> str | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    meta = dict(payload.get("meta") or {})
    summary = dict(payload.get("summary") or {})
    value = meta.get("last_oos_date") or summary.get("oos_end_date")
    return None if value in (None, "") else str(value)


def _benchmark_manifest_matches_current_policy(
    manifest_path: Path, *, benchmark: dict[str, Any], benchmark_id: str, seed: int,
    family: str, evaluation_mode: str, policy: str, target_path: Path,
) -> bool:
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if (
        str(payload.get("benchmark_id") or "") != str(benchmark_id)
        or int(payload.get("benchmark_seed") or -1) != int(seed)
        or str(payload.get("family") or "") != str(family)
        or str(payload.get("evaluation_mode") or "") != str(evaluation_mode)
    ):
        return False
    pinned = dict(payload.get("benchmark") or {})
    if (
        str(pinned.get("benchmark_id") or "") != str(benchmark.get("benchmark_id") or "")
        or tuple(int(v) for v in tuple(pinned.get("resolved_seeds") or ()))
            != tuple(int(v) for v in tuple(benchmark.get("resolved_seeds") or ()))
        or int(pinned.get("strategy_trials_per_fold") or -1)
            != int(benchmark.get("strategy_trials_per_fold") or -2)
    ):
        return False
    training = dict(payload.get("training_policy") or {})
    if (
        int(training.get("optimizer_seed") or -1) != int(seed)
        or int(training.get("trials_per_fold") or -1)
            != int(benchmark.get("strategy_trials_per_fold") or -2)
        or training.get("benchmark_override") is not True
    ):
        return False
    artifact = dict(dict(payload.get("artifacts") or {}).get(policy) or {})
    expected_sha = str(artifact.get("sha256") or "")
    return bool(expected_sha) and expected_sha == compute_strategy_param_file_sha256(target_path)

def ensure_robustness_benchmark_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    family: str,
    evaluation_mode: str,
    policy: str,
    comparison_end_date: str,
    dataset: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    max_positions: int,
    rotation: str,
    fixed_risk: float,
    max_position_cap_pct: float,
    resume_parameter_training: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Build/reuse one optimizer-owned end-to-end robustness benchmark param artifact.

    OOS trains only the <=2020 -> 2021 initial fold and freezes that exact member.
    Rolling reuses the same cached 2021 member and optimizes only 2022+ folds before
    stitching the schedule, so OOS and Rolling cannot silently diverge at 2021.
    """
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    policy = normalize_strategy_param_policy(policy)
    if mode not in {"oos", "rolling"}:
        raise ValueError("robustness benchmark策略參數只支援oos/rolling")
    if policy != "base_finalist_best":
        raise ValueError("robustness benchmark per-seed策略參數固定使用base-finalist-best")
    benchmark = get_robustness_benchmark_policy_snapshot()
    if str(benchmark_id) != str(benchmark["benchmark_id"]):
        raise ValueError(f"benchmark_id與current SSOT不一致: {benchmark_id}")
    if int(seed) not in {int(value) for value in benchmark["resolved_seeds"]}:
        raise ValueError(f"seed不屬於current robustness benchmark題庫: {seed}")
    trials = int(benchmark["strategy_trials_per_fold"])
    target = resolve_strategy_param_benchmark_artifact_path(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        family=family,
        evaluation_mode=mode,
        policy=policy,
    )
    manifest_path = resolve_strategy_param_benchmark_manifest_path(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        family=family,
        evaluation_mode=mode,
    )
    reusable = (
        target.is_file()
        and manifest_path.is_file()
        and _benchmark_parameter_coverage_end(target) == str(comparison_end_date)
        and _benchmark_manifest_matches_current_policy(
            manifest_path,
            benchmark=benchmark,
            benchmark_id=str(benchmark_id),
            seed=int(seed),
            family=family,
            evaluation_mode=mode,
            policy=policy,
            target_path=target,
        )
    )
    if reusable:
        initial_sha = _benchmark_effective_member_sha256(target)
        aligned_sha = _validate_benchmark_oos_rolling_initial_alignment(
            root, benchmark_id=str(benchmark_id), seed=int(seed), family=family, policy=policy
        )
        return {
            "action": "REUSE", "path": target, "manifest_path": manifest_path,
            "sha256": compute_strategy_param_file_sha256(target),
            "initial_2021_member_sha256": str(aligned_sha or initial_sha),
        }

    work_root = _benchmark_optimizer_work_root(
        root, benchmark_id=str(benchmark_id), seed=int(seed), family=family
    )
    work_root.mkdir(parents=True, exist_ok=True)
    initial_cache = work_root / "initial_2021_base_best.json"
    from services.optimizer.strategy_param_training import (
        prepare_selection_historical_full_roos_params,
        prepare_selection_historical_p2_params,
    )

    if not initial_cache.is_file():
        initial_output = work_root / "initial_2021"
        common = dict(
            project_root=root,
            dataset=str(dataset),
            param_policy="base-finalist-best",
            trials_per_fold=trials,
            max_positions=int(max_positions),
            rotation=str(rotation),
            fixed_risk=float(fixed_risk),
            max_position_cap_pct=float(max_position_cap_pct),
            optimizer_seed=int(seed),
            resume_parameter_training=bool(resume_parameter_training),
            quiet=bool(quiet),
            first_oos_date="2021-01-01",
            last_oos_date="2021-12-31",
            train_window_months=120,
            oos_months=12,
            output_relative_dir=initial_output,
        )
        if family == "full":
            result = prepare_selection_historical_full_roos_params(**common)
        else:
            result = prepare_selection_historical_p2_params(
                **common,
                comparison_output_root=str(work_root / "no_recovery"),
                comparison_output_roots=(str(work_root / "no_recovery"),),
            )
        _copy_json_payload(Path(result["params_path"]), initial_cache)

    if mode == "oos":
        _freeze_rolling_policy_to_oos(
            root,
            family=family,
            policy=policy,
            source_path=initial_cache,
            target_path=target,
            coverage_end_date=str(comparison_end_date),
        )
    else:
        tail_path = None
        if str(comparison_end_date) >= "2022-01-01":
            tail_output = work_root / "rolling_tail_2022_plus"
            common_tail = dict(
                project_root=root,
                dataset=str(dataset),
                param_policy="base-finalist-best",
                trials_per_fold=trials,
                max_positions=int(max_positions),
                rotation=str(rotation),
                fixed_risk=float(fixed_risk),
                max_position_cap_pct=float(max_position_cap_pct),
                optimizer_seed=int(seed),
                resume_parameter_training=bool(resume_parameter_training),
                quiet=bool(quiet),
                first_oos_date="2022-01-01",
                last_oos_date=str(comparison_end_date),
                train_window_months=120,
                oos_months=12,
                output_relative_dir=tail_output,
            )
            if family == "full":
                tail_result = prepare_selection_historical_full_roos_params(**common_tail)
            else:
                tail_result = prepare_selection_historical_p2_params(
                    **common_tail,
                    comparison_output_root=str(work_root / "no_recovery_tail"),
                    comparison_output_roots=(str(work_root / "no_recovery_tail"),),
                )
            tail_path = Path(tail_result["params_path"])
        _stitch_benchmark_rolling_payload(
            initial_path=initial_cache,
            tail_path=tail_path,
            target_path=target,
            comparison_end_date=str(comparison_end_date),
        )

    initial_member_sha = _benchmark_effective_member_sha256(target)
    aligned_member_sha = _validate_benchmark_oos_rolling_initial_alignment(
        root, benchmark_id=str(benchmark_id), seed=int(seed), family=family, policy=policy
    )
    manifest = refresh_strategy_parameter_benchmark_manifest(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        family=family,
        evaluation_mode=mode,
        source_records={
            policy: {
                "benchmark_initial_2021_member_sha256": str(aligned_member_sha or initial_member_sha),
                "same_seed_oos_rolling_initial_required": True,
            }
        },
    )
    return {
        "action": "BUILD",
        "path": target,
        "manifest_path": manifest,
        "sha256": compute_strategy_param_file_sha256(target),
        "initial_2021_member_sha256": str(aligned_member_sha or initial_member_sha),
    }


__all__ = [
    "ensure_robustness_benchmark_strategy_parameter_artifact",
    "ensure_strategy_parameter_artifact",
    "migrate_legacy_strategy_parameter_artifacts",
    "migrate_legacy_trade_state_artifacts",
    "migrate_all_legacy_strategy_parameter_artifacts",
    "collect_legacy_root_strategy_parameter_cleanup_plan",
    "finalize_legacy_strategy_parameter_migration",
    "refresh_strategy_parameter_manifest",
]
