"""Canonical optimizer-owned strategy-parameter artifact service.

Current consumers request a parameter identity (family/mode/policy).  This service
owns migration/materialization into ``models/strategy_params`` and records the
optimizer training-policy identity.  Research code must not own seed/trial/path
settings for current parameter artifacts.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.training_policy import (
    get_strategy_parameter_training_policy_snapshot,
    get_robustness_benchmark_policy_snapshot,
)
from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
    ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
    build_active_param_ensemble_schedule,
    get_active_param_ensemble_date_range,
    get_active_param_ensemble_policy,
    is_active_param_ensemble_payload,
    resolve_active_param_ensemble_mode,
)
from core.seed_ensemble_policy import (
    normalize_seed_ensemble_members,
    resolve_seed_ensemble_min_agree,
    resolve_seed_ensemble_seeds,
)
from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.strategy_param_scientific_contract import (
    build_registered_strategy_param_scientific_identity,
    derive_strategy_param_scientific_identity_from_registered_profile,
)
from core.strategy_param_policy import (
    get_strategy_param_fitting_seed_protocol,
    resolve_canonical_strategy_param_training_profile,
)
from core.strategy_param_artifacts import (
    POLICY_FILENAME_BY_NAME,
    COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY,
    compute_strategy_param_file_sha256,
    compute_strategy_param_scientific_sha256,
    normalize_strategy_param_evaluation_mode,
    normalize_strategy_param_family,
    normalize_strategy_param_policy,
    normalize_strategy_param_payload_for_persistence,
    normalize_strategy_param_benchmark_id,
    normalize_strategy_param_benchmark_seed,
    resolve_strategy_param_artifact_path,
    resolve_strategy_param_benchmark_artifact_path,
    resolve_strategy_param_benchmark_dir,
    resolve_strategy_param_benchmark_manifest_path,
    resolve_strategy_param_dir,
    resolve_strategy_param_optimizer_work_dir,
    STRATEGY_PARAM_WORK_ROOT_RELATIVE,
    STRATEGY_PARAM_WORK_SCOPE_CANONICAL,
    STRATEGY_PARAM_WORK_SCOPE_BENCHMARK,
    resolve_strategy_param_manifest_path,
    resolve_strategy_param_state_path,
    STRATEGY_PARAM_STATE_FILENAME_BY_NAME,
)
from services.optimizer.strategy_param_repository import (
    refresh_strategy_parameter_benchmark_manifest,
    refresh_strategy_parameter_manifest,
)
from services.optimizer.strategy_param_artifact_validation import (
    first_manifest_identity_mismatch,
    manifest_artifact_sha_matches,
    resolve_artifact_reuse_action,
)


def _force_rebuild_attempt_root(base: Path) -> Path:
    """Return a fresh execution-only namespace; never part of scientific identity."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = Path(base) / "force_rebuild_attempts" / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _current_optimizer_execution_evidence_contract() -> str:
    from services.optimizer.runtime import OPTIMIZER_EXECUTION_EVIDENCE_CONTRACT

    return str(OPTIMIZER_EXECUTION_EVIDENCE_CONTRACT)


def _resume_safe_execution_rebuild_root(base: Path, *, contract: str) -> Path:
    """Stable repair workspace for stale pre-deterministic execution evidence.

    The path is deliberately stable for one execution contract so an interrupted
    automatic repair resumes the same deterministic study rather than starting a
    new force-rebuild attempt on every application restart. Historical workspace
    remains untouched for audit.
    """

    token = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(contract or "")
    ).strip("_")
    if not token:
        raise ValueError("optimizer execution evidence contract不得為空")
    path = Path(base) / "execution_evidence_rebuilds" / token
    path.mkdir(parents=True, exist_ok=True)
    return path


def _archive_stale_execution_evidence_once(
    *, root: Path, target: Path, manifest_path: Path, work_root: Path, reason: str
) -> Path | None:
    """Preserve stale published bytes before an automatic execution-evidence repair."""

    target = Path(target)
    if not target.is_file():
        return None
    artifact_sha = compute_strategy_param_file_sha256(target)
    archive_root = Path(work_root) / "execution_evidence_archive" / artifact_sha
    archive_root.mkdir(parents=True, exist_ok=True)
    archived_target = archive_root / target.name
    if not archived_target.exists():
        shutil.copy2(target, archived_target)
    archived_manifest = archive_root / "identity_manifest.json"
    if Path(manifest_path).is_file() and not archived_manifest.exists():
        shutil.copy2(manifest_path, archived_manifest)
    metadata_path = archive_root / "archive_metadata.json"
    if not metadata_path.exists():
        _write_json(
            metadata_path,
            {
                "schema_type": "strategy_parameter_execution_evidence_archive",
                "schema_version": 1,
                "reason": str(reason),
                "artifact_sha256": artifact_sha,
                "artifact_path": _project_relative(root, target),
                "manifest_path": _project_relative(root, manifest_path),
            },
        )
    return archive_root


def _project_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _prune_empty_parents(start: Path, *, stop: Path) -> None:
    current = Path(start)
    stop = Path(stop).resolve()
    while True:
        try:
            resolved = current.resolve()
        except OSError:
            return
        if resolved == stop or stop not in resolved.parents:
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _adopt_legacy_optimizer_work_root(
    *, current_root: Path, legacy_root: Path, prune_stop: Path | None = None
) -> Path:
    """Move a legacy optimizer workspace into the current outputs-only namespace once.

    Truth resolvers never read the legacy workspace.  This migration only preserves
    resumable producer state when the new workspace has not been created yet.
    """
    current_root = Path(current_root).resolve()
    legacy_root = Path(legacy_root).resolve()
    if not legacy_root.is_dir():
        return current_root
    current_root.parent.mkdir(parents=True, exist_ok=True)
    if not current_root.exists():
        shutil.move(str(legacy_root), str(current_root))
    else:
        archive_dir = current_root.parent / "_legacy_workspace_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_target = archive_dir / current_root.name
        suffix = 1
        while archive_target.exists():
            archive_target = archive_dir / f"{current_root.name}_{suffix}"
            suffix += 1
        shutil.move(str(legacy_root), str(archive_target))
    if prune_stop is not None:
        _prune_empty_parents(legacy_root.parent, stop=Path(prune_stop))
    return current_root


def _assert_optimizer_work_source(*, source_path: Path, work_root: Path) -> None:
    source = Path(source_path).resolve()
    work = Path(work_root).resolve()
    if work != source and work not in source.parents:
        raise RuntimeError(
            "Optimizer current producer回傳的params_path不在正式workspace內: "
            f"source={source.as_posix()}, workspace={work.as_posix()}"
        )


def _optimizer_compare_policy_outputs(source_path: Path) -> dict[str, Path]:
    """Return all first-class compare-policy files emitted by one rolling search."""
    source_dir = Path(source_path).resolve().parent
    outputs: dict[str, Path] = {}
    for policy in COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.values():
        normalized = normalize_strategy_param_policy(policy)
        candidate = source_dir / f"roos_{POLICY_FILENAME_BY_NAME[normalized]}"
        if not candidate.is_file():
            continue
        payload = _load_json_object_permissive(candidate)
        if payload is None:
            raise ValueError(
                "Optimizer策略參數工作工件不是合法JSON: "
                + candidate.as_posix()
            )
        if str(payload.get("selector") or "") != normalized:
            raise ValueError(
                "Optimizer策略參數policy selector不一致: "
                f"policy={normalized}, selector={payload.get('selector')}, path={candidate.as_posix()}"
            )
        outputs[normalized] = candidate
    return outputs


def _print_published_strategy_param_paths(
    root: Path, *, title: str, paths: dict[str, Path], manifest_path: Path
) -> None:
    print(str(title))
    for policy in COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.values():
        path = paths.get(policy)
        if path is not None:
            print(f"  {policy}: {_project_relative(root, path)}")
    print(f"  manifest: {_project_relative(root, manifest_path)}")


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
    # OOS and Rolling now share one complete schedule.  Only a legacy Rolling
    # schedule is eligible to seed that truth; legacy frozen-OOS files are not.
    source_mode = "rolling" if mode in {"oos", "rolling"} else mode
    legacy_filename = _legacy_policy_filename(policy, evaluation_mode=source_mode)
    canonical_old_filename = POLICY_FILENAME_BY_NAME[policy]
    candidates: list[Path] = [
        root / "models" / "strategy_params" / family / source_mode / canonical_old_filename,
    ]
    if family == "full":
        candidates.append(root / "models" / legacy_filename)
        if source_mode == "rolling":
            candidates.append(
                root / "models" / "research" / "breakout_quality" / "strategy_compare"
                / "extending_full_roos" / "active_params" / legacy_filename
            )
    elif source_mode == "rolling":
        candidates.extend([
            root / "models" / "research" / "breakout_quality" / "strategy_compare"
            / "extending_min_roos" / "active_params" / legacy_filename,
            root / "models" / "research" / "breakout_quality" / "binary_dl_filter_param_adaptation"
            / "risk_only_rolling" / "p2_dl_off_trained" / "active_params" / legacy_filename,
        ])
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return tuple(unique)









def _materialize_legacy_candidate(
    *, root: Path, family: str, policy: str, candidate: Path, target: Path,
    evaluation_mode: str,
) -> bool:
    # The new repository stores one schedule per strategy.  Legacy OOS frozen
    # artifacts are intentionally excluded by _legacy_candidates.
    del root, family, policy, evaluation_mode
    target.parent.mkdir(parents=True, exist_ok=True)
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
        nested_name_by_artifact = {
            "active": "active.json",
            "active_summary": "active_summary.json",
            "candidate_best": "candidate_best.json",
            "candidate_best_summary": "candidate_best_summary.json",
            "candidate_retention_best": "candidate_retention_best.json",
            "candidate_retention_best_summary": "candidate_retention_best_summary.json",
            "candidate_val_score_best": "candidate_val_score_best.json",
            "candidate_val_score_best_summary": "candidate_val_score_best_summary.json",
        }
        candidates = (
            root / "models" / "strategy_params" / "full" / "trade" / "state"
            / nested_name_by_artifact[artifact],
            root / "models" / legacy_name,
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
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
        payload = load_json_strict(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_object_permissive(path: Path) -> dict[str, Any] | None:
    """Load legacy/workspace JSON that may contain Python's Infinity constants."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


_LEGACY_OOS_ARCHIVE_RELATIVE = (
    Path("models") / "research" / "breakout_quality" / "strategy_param_legacy_oos"
)


def _legacy_oos_archive_manifest_path(root: Path) -> Path:
    return root / _LEGACY_OOS_ARCHIVE_RELATIVE / "archive_manifest.json"


def _legacy_oos_archive_path(root: Path, *, family: str, source_sha: str, source: Path) -> Path:
    stem = source.stem.replace(" ", "_")
    return (
        root
        / _LEGACY_OOS_ARCHIVE_RELATIVE
        / f"{family}_oos_{stem}_{str(source_sha)[:16]}.json"
    )


def _load_legacy_oos_archive_records(root: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json_object(_legacy_oos_archive_manifest_path(root)) or {}
    out: dict[str, dict[str, Any]] = {}
    for record in list(payload.get("artifacts") or []):
        if not isinstance(record, dict):
            continue
        source_path = str(record.get("source_path") or "")
        source_sha = str(record.get("source_sha256") or "")
        archive_path = str(record.get("archive_path") or "")
        archive_sha = str(record.get("archive_sha256") or "")
        if source_path and source_sha and archive_path and archive_sha:
            out[f"{source_path}|{source_sha}"] = dict(record)
    return out


def _archive_legacy_frozen_oos_artifacts(project_root: str | Path) -> dict[str, Any]:
    """Preserve divergent legacy OOS artifacts before flat-SSOT cleanup.

    Old OOS parameters were historically allowed to be trained independently from
    Rolling.  After OOS became a frozen view of the single canonical schedule, such
    payloads are no longer current truth, but they still remain historical evidence.
    This explicit migration copies every existing legacy OOS source byte-for-byte to
    a historical research archive and records both source/archive SHA256 values.
    """
    root = Path(project_root).resolve()
    models_root = root / "models"
    sources: list[tuple[str, str, Path]] = []
    for family in ("full", "min"):
        for policy in POLICY_FILENAME_BY_NAME:
            nested = (
                models_root / "strategy_params" / family / "oos"
                / POLICY_FILENAME_BY_NAME[policy]
            )
            sources.append((family, policy, nested))
            if family == "full":
                sources.append((
                    family,
                    policy,
                    models_root / _legacy_policy_filename(policy, evaluation_mode="oos"),
                ))

    records = _load_legacy_oos_archive_records(root)
    archived: list[dict[str, str]] = []
    seen: set[str] = set()
    for family, policy, source in sorted(sources, key=lambda item: str(item[2]).lower()):
        if not source.is_file():
            continue
        target = resolve_strategy_param_artifact_path(
            root, family=family, evaluation_mode="rolling", policy=policy
        )
        if target.is_file() and _legacy_frozen_oos_matches_schedule_initial(source, target):
            # The canonical schedule already preserves this frozen OOS runtime view.
            continue
        source_rel = _project_relative(root, source)
        source_sha = compute_strategy_param_file_sha256(source)
        key = f"{source_rel}|{source_sha}"
        if key in seen:
            continue
        seen.add(key)
        archive = _legacy_oos_archive_path(
            root, family=family, source_sha=source_sha, source=source
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file() or compute_strategy_param_file_sha256(archive) != source_sha:
            shutil.copy2(source, archive)
        archive_sha = compute_strategy_param_file_sha256(archive)
        if archive_sha != source_sha:
            raise RuntimeError(
                "legacy OOS historical archive SHA mismatch: "
                f"{source_rel} -> {_project_relative(root, archive)}"
            )
        record = {
            "family": family,
            "source_path": source_rel,
            "source_sha256": source_sha,
            "archive_path": _project_relative(root, archive),
            "archive_sha256": archive_sha,
        }
        records[key] = record
        archived.append({
            "source": source_rel,
            "archive": record["archive_path"],
            "sha256": source_sha,
        })

    manifest_path = _legacy_oos_archive_manifest_path(root)
    if records:
        manifest_payload = {
            "schema_type": "historical_strategy_parameter_oos_archive",
            "schema_version": 1,
            "producer": "optimizer_migration",
            "usage": "historical_evidence_only",
            "artifacts": sorted(
                records.values(),
                key=lambda record: (str(record["source_path"]), str(record["source_sha256"])),
            ),
        }
        _write_json(manifest_path, manifest_payload)
    return {
        "status": "PASS",
        "archive_manifest_path": (
            _project_relative(root, manifest_path) if records else ""
        ),
        "archived": archived,
    }


def _runtime_param_member_signatures(members: Any) -> list[str]:
    """Return replay-semantic hashes for ensemble members.

    Legacy OOS and Rolling artifacts may carry different optimizer bookkeeping
    (seed/member_index/trial metadata) even when the actual strategy parameter
    mappings consumed by replay are identical. Cleanup proof therefore compares
    only each member's ``params`` mapping, not producer-only member metadata.
    """
    return sorted(
        canonical_json_sha256(dict(member["params"]))
        for member in normalize_seed_ensemble_members(members)
    )


def _legacy_frozen_oos_matches_schedule_initial(source: Path, target: Path) -> bool:
    legacy = _load_json_object_permissive(source)
    schedule = _load_json_object(target)
    if (
        legacy is None
        or schedule is None
        or not is_active_param_ensemble_payload(legacy)
        or not is_active_param_ensemble_payload(schedule)
        or resolve_active_param_ensemble_mode(schedule) != ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING
    ):
        return False

    try:
        legacy_schedule = build_active_param_ensemble_schedule(legacy)
        canonical_schedule = build_active_param_ensemble_schedule(schedule)
        legacy_policy = get_active_param_ensemble_policy(legacy)
        canonical_policy = get_active_param_ensemble_policy(schedule)
    except (KeyError, TypeError, ValueError):
        return False
    if not legacy_schedule or not canonical_schedule:
        return False

    legacy_members = list(legacy_schedule[0].get("members") or [])
    canonical_members = list(canonical_schedule[0].get("members") or [])
    legacy_signatures = _runtime_param_member_signatures(legacy_members)
    canonical_signatures = _runtime_param_member_signatures(canonical_members)
    if not legacy_signatures or legacy_signatures != canonical_signatures:
        return False

    # Ensemble voting is replay-semantic. A different min_agree can change which
    # candidates are tradable even when every member's params are identical.
    return int(legacy_policy.get("min_agree") or 0) == int(canonical_policy.get("min_agree") or 0)


def collect_legacy_root_strategy_parameter_cleanup_plan(project_root: str | Path) -> dict[str, Any]:
    """Return legacy strategy files proven safe to delete after flat-SSOT migration.

    Both the pre-Round-3 root files and the Round-3 nested repository are covered.
    Existence alone is never enough: target path/SHA must be manifest-pinned and the
    old source must either byte-match or be recorded as the exact migration lineage.
    """
    root = Path(project_root).resolve()
    models_root = root / "models"
    mappings: list[dict[str, Any]] = []
    for mode in ("study", "full", "rolling", "trade"):
        for policy in POLICY_FILENAME_BY_NAME:
            target = resolve_strategy_param_artifact_path(
                root, family="full", evaluation_mode=mode, policy=policy
            )
            nested = models_root / "strategy_params" / "full" / mode / POLICY_FILENAME_BY_NAME[policy]
            root_legacy = models_root / _legacy_policy_filename(policy, evaluation_mode=mode)
            for source in (nested, root_legacy):
                mappings.append({
                    "kind": "policy", "family": "full", "evaluation_mode": mode,
                    "policy": policy, "source": source, "target": target,
                })
    for family in ("full", "min"):
        for policy in POLICY_FILENAME_BY_NAME:
            target = resolve_strategy_param_artifact_path(
                root, family=family, evaluation_mode="rolling", policy=policy
            )
            for source in (
                models_root / "strategy_params" / family / "oos" / POLICY_FILENAME_BY_NAME[policy],
                models_root / _legacy_policy_filename(policy, evaluation_mode="oos")
                if family == "full" else Path("__missing_min_root_oos__"),
            ):
                mappings.append({
                    "kind": "frozen_oos", "family": family, "evaluation_mode": "rolling",
                    "policy": policy, "source": source, "target": target,
                })

    for policy in POLICY_FILENAME_BY_NAME:
        target = resolve_strategy_param_artifact_path(
            root, family="min", evaluation_mode="rolling", policy=policy
        )
        nested = models_root / "strategy_params" / "min" / "rolling" / POLICY_FILENAME_BY_NAME[policy]
        mappings.append({
            "kind": "policy", "family": "min", "evaluation_mode": "rolling",
            "policy": policy, "source": nested, "target": target,
        })

    nested_state_names = {
        "active": "active.json",
        "active_summary": "active_summary.json",
        "candidate_best": "candidate_best.json",
        "candidate_best_summary": "candidate_best_summary.json",
        "candidate_retention_best": "candidate_retention_best.json",
        "candidate_retention_best_summary": "candidate_retention_best_summary.json",
        "candidate_val_score_best": "candidate_val_score_best.json",
        "candidate_val_score_best_summary": "candidate_val_score_best_summary.json",
    }
    root_state_names = {
        "active": "run_best_params.json",
        "active_summary": "run_best_summary.json",
        "candidate_best": "candidate_best_params.json",
        "candidate_best_summary": "candidate_best_summary.json",
        "candidate_retention_best": "candidate_retention_best_params.json",
        "candidate_retention_best_summary": "candidate_retention_best_summary.json",
        "candidate_val_score_best": "candidate_val_score_best_params.json",
        "candidate_val_score_best_summary": "candidate_val_score_best_summary.json",
    }
    for artifact in STRATEGY_PARAM_STATE_FILENAME_BY_NAME:
        target = resolve_strategy_param_state_path(root, artifact=artifact)
        for source in (
            models_root / "strategy_params" / "full" / "trade" / "state" / nested_state_names[artifact],
            models_root / root_state_names[artifact],
        ):
            mappings.append({
                "kind": "state", "family": "full", "evaluation_mode": "trade",
                "state_artifact": artifact, "source": source, "target": target,
            })

    removable: list[dict[str, str]] = []
    blockers: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    legacy_oos_archive_records = _load_legacy_oos_archive_records(root)
    for item in sorted(mappings, key=lambda row: str(row["source"]).lower()):
        source = Path(item["source"]); target = Path(item["target"])
        source_key = str(source.resolve())
        if source_key in seen_sources or not source.is_file():
            continue
        seen_sources.add(source_key)
        source_rel = _project_relative(root, source); target_rel = _project_relative(root, target)
        base_record = {"source": source_rel, "target": target_rel}
        if not target.is_file():
            blockers.append({**base_record, "reason": "canonical_target_missing"}); continue

        family = str(item["family"]); mode = str(item["evaluation_mode"])
        manifest_path = resolve_strategy_param_manifest_path(root, family=family, evaluation_mode=mode)
        manifest = _load_json_object(manifest_path)
        if manifest is None:
            blockers.append({**base_record, "reason": "canonical_manifest_missing_or_invalid"}); continue
        expected_storage_mode = "schedule" if mode in {"oos", "rolling"} else mode
        if (
            str(manifest.get("producer") or "") != "optimizer"
            or str(manifest.get("family") or "") != family
            or str(manifest.get("evaluation_mode") or "") != expected_storage_mode
        ):
            blockers.append({**base_record, "reason": "canonical_manifest_identity_mismatch"}); continue

        target_sha = compute_strategy_param_file_sha256(target)
        source_sha = compute_strategy_param_file_sha256(source)
        archive_records = legacy_oos_archive_records if item["kind"] == "frozen_oos" else {}
        if item["kind"] in {"policy", "frozen_oos"}:
            policy = str(item["policy"]); manifest_record = dict(dict(manifest.get("artifacts") or {}).get(policy) or {})
        else:
            state_name = str(item["state_artifact"]); manifest_record = dict(dict(manifest.get("state_artifacts") or {}).get(state_name) or {})
        if (
            str(manifest_record.get("path") or "") != target_rel
            or str(manifest_record.get("sha256") or "") != target_sha
        ):
            blockers.append({**base_record, "reason": "canonical_manifest_target_not_pinned"}); continue

        evidence = "content_sha_match" if source_sha == target_sha else ""
        if not evidence and item["kind"] == "frozen_oos":
            if _legacy_frozen_oos_matches_schedule_initial(source, target):
                evidence = "frozen_view_matches_schedule_initial"
            else:
                archive_record = dict(
                    archive_records.get(f"{source_rel}|{source_sha}") or {}
                )
                archive_rel = str(archive_record.get("archive_path") or "")
                archive_path = root / archive_rel if archive_rel else Path("__missing_archive__")
                if (
                    archive_rel
                    and str(archive_record.get("source_sha256") or "") == source_sha
                    and archive_path.is_file()
                    and compute_strategy_param_file_sha256(archive_path) == source_sha
                    and str(archive_record.get("archive_sha256") or "") == source_sha
                ):
                    evidence = "historical_oos_archive_sha_match"
        if not evidence and item["kind"] == "policy":
            source_record = dict(manifest_record.get("source") or {})
            if (
                bool(source_record.get("migration_only"))
                and str(source_record.get("path") or "") == source_rel
                and str(source_record.get("sha256") or "") == source_sha
            ):
                evidence = "manifest_migration_lineage"
        if not evidence:
            blockers.append({**base_record, "reason": "legacy_source_not_proven_migrated"}); continue
        removable.append({**base_record, "evidence": evidence})

    return {"status": "READY" if not blockers else "BLOCKED", "removable": removable, "blockers": blockers}


def migrate_all_legacy_strategy_parameter_artifacts(project_root: str | Path) -> dict[str, Any]:
    """One-shot content-preserving migration of every legacy current parameter family."""
    root = Path(project_root).resolve()
    results: list[dict[str, Any]] = []
    for mode in ("study", "full", "rolling", "trade"):
        results.append(
            migrate_legacy_strategy_parameter_artifacts(
                root, family="full", evaluation_mode=mode
            )
        )
    results.append(
        migrate_legacy_strategy_parameter_artifacts(
            root, family="min", evaluation_mode="rolling"
        )
    )
    legacy_oos_archive = _archive_legacy_frozen_oos_artifacts(root)
    return {
        "status": "PASS",
        "project_root": str(root),
        "results": results,
        "legacy_oos_archive": legacy_oos_archive,
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


def _canonical_family_fitting_seed_protocol(family: str):
    """Return the one fitting protocol shared by a canonical physical schedule.

    A Min/Full physical Optimizer search may emit multiple policy outputs, but all
    outputs from that search must share the same seed/trial fitting semantics.
    Those semantics are owned by the registered Strategy Parameter profiles, not
    by mutable generic Optimizer defaults.
    """

    normalized_family = normalize_strategy_param_family(family)
    profiles = []
    for compare_policy in COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY:
        profile = resolve_canonical_strategy_param_training_profile(
            normalized_family, compare_policy
        )
        if profile is not None:
            profiles.append(profile)
    if not profiles:
        raise ValueError(
            "canonical strategy parameter family缺少registered fitting profile: "
            f"family={normalized_family}"
        )
    protocol_ids = {str(profile.fitting_seed_protocol) for profile in profiles}
    if len(protocol_ids) != 1:
        raise ValueError(
            "同一canonical physical schedule的policy outputs不得使用不同fitting protocol: "
            f"family={normalized_family}, protocols={sorted(protocol_ids)}"
        )
    protocol = get_strategy_param_fitting_seed_protocol(next(iter(protocol_ids)))
    if protocol.enabled or protocol.seed_count != 1:
        raise ValueError(
            "canonical single-seed physical schedule必須使用single-seed fitting protocol: "
            f"family={normalized_family}, protocol={protocol.protocol_id}"
        )
    return protocol


def _canonical_schedule_training_policy(family: str) -> dict[str, Any]:
    policy = dict(get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling"))
    fitting_protocol = _canonical_family_fitting_seed_protocol(family)
    policy["optimizer_seed"] = int(fitting_protocol.generator_seed)
    policy["trials_per_fold"] = int(fitting_protocol.trials_per_seed)
    policy["evaluation_mode"] = "schedule"
    policy["consumption_modes"] = ["oos", "rolling"]
    return policy


def _schedule_training_policy_reuse_projection(policy: dict[str, Any]) -> dict[str, Any]:
    """Return only physical-schedule fields owned by the Strategy Parameter fit.

    Selector/anti-overfit semantics are bound by each policy output's scientific
    identity. Generic random-ensemble defaults are not used by these registered
    single-seed schedules and must not invalidate their reuse.
    """

    keys = (
        "owner",
        "evaluation_mode",
        "optimizer_seed",
        "trials_per_fold",
        "train_window_months",
        "oos_horizon_months",
        "consumption_modes",
    )
    return {key: policy.get(key) for key in keys}


def _canonical_schedule_build_contract(
    *,
    family: str,
    dataset: str | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
) -> dict[str, Any]:
    from config.execution_policy import (
        DEFAULT_FIXED_RISK,
        DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS,
        DEFAULT_PORTFOLIO_ROTATION,
    )
    from core.dataset_profiles import DEFAULT_DATASET_PROFILE
    from services.optimizer.strategy_param_training import (
        FULL_ROOS_SEARCH_FIELDS,
        MIN_ROOS_SEARCH_FIELDS,
    )
    from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE

    family = normalize_strategy_param_family(family)
    policy = dict(get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling"))
    fitting_protocol = _canonical_family_fitting_seed_protocol(family)
    return {
        "family": family,
        "dataset": str(DEFAULT_DATASET_PROFILE if dataset in (None, "") else dataset),
        "search_space_sha256": canonical_json_sha256(BREAKOUT_OPTIMIZER_SEARCH_SPACE),
        "optimizer_seed": int(fitting_protocol.generator_seed),
        "trials_per_fold": int(fitting_protocol.trials_per_seed),
        "train_window_months": int(policy["train_window_months"]),
        "oos_horizon_months": int(policy["oos_horizon_months"]),
        "max_positions": int(DEFAULT_PORTFOLIO_MAX_POSITIONS if max_positions is None else max_positions),
        "rotation": str(DEFAULT_PORTFOLIO_ROTATION if rotation in (None, "") else rotation),
        "fixed_risk": float(DEFAULT_FIXED_RISK if fixed_risk is None else fixed_risk),
        "max_position_cap_pct": float(
            DEFAULT_MAX_POSITION_CAP_PCT if max_position_cap_pct is None else max_position_cap_pct
        ),
        "search_fields": list(FULL_ROOS_SEARCH_FIELDS if family == "full" else MIN_ROOS_SEARCH_FIELDS),
        "schedule_start": "2021-01-01",
    }


def _strategy_param_execution_overrides(
    *,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
) -> dict[str, Any]:
    from config.execution_policy import (
        DEFAULT_FIXED_RISK,
        DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS,
        DEFAULT_PORTFOLIO_ROTATION,
    )

    return {
        "max_positions": int(
            DEFAULT_PORTFOLIO_MAX_POSITIONS if max_positions is None else max_positions
        ),
        "rotation": str(
            DEFAULT_PORTFOLIO_ROTATION if rotation in (None, "") else rotation
        ),
        "fixed_risk": float(DEFAULT_FIXED_RISK if fixed_risk is None else fixed_risk),
        "max_position_cap_pct": float(
            DEFAULT_MAX_POSITION_CAP_PCT
            if max_position_cap_pct is None
            else max_position_cap_pct
        ),
    }


def _canonical_schedule_scientific_identity(
    root: Path,
    *,
    family: str,
    policy: str,
    dataset: str | None,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
) -> dict[str, Any] | None:
    profile = resolve_canonical_strategy_param_training_profile(family, policy)
    if profile is None:
        return None
    requested_dataset = str(profile.dataset if dataset in (None, "") else dataset)
    if requested_dataset != str(profile.dataset):
        raise ValueError(
            "registered canonical strategy parameter identity不得覆寫dataset: "
            f"profile={profile.profile_id}, registered={profile.dataset!r}, "
            f"requested={requested_dataset!r}"
        )
    return build_registered_strategy_param_scientific_identity(
        root,
        profile.profile_id,
        execution_overrides=_strategy_param_execution_overrides(
            max_positions=max_positions,
            rotation=rotation,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
        ),
    )


def _scientific_identity_record(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(identity["schema_version"]),
        "profile_id": str(identity["profile_id"]),
        "registered_scientific_fingerprint_sha256": str(
            identity["registered_scientific_fingerprint_sha256"]
        ),
        "scientific_fingerprint_sha256": str(
            identity["scientific_fingerprint_sha256"]
        ),
        "scientific_contract": dict(identity["scientific_contract"]),
    }


def _derived_scientific_identity_record(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(identity["schema_version"]),
        "source_profile_id": str(identity["source_profile_id"]),
        "source_registered_scientific_fingerprint_sha256": str(
            identity.get("source_registered_scientific_fingerprint_sha256") or ""
        ),
        "source_scientific_fingerprint_sha256": str(
            identity.get("source_scientific_fingerprint_sha256") or ""
        ),
        "scientific_fingerprint_sha256": str(
            identity["scientific_fingerprint_sha256"]
        ),
        "scientific_contract": dict(identity["scientific_contract"]),
    }


def _canonical_family_scientific_identities(
    root: Path,
    *,
    family: str,
    dataset: str | None,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for compare_policy in COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY:
        normalized_policy = normalize_strategy_param_policy(compare_policy)
        identity = _canonical_schedule_scientific_identity(
            root,
            family=family,
            policy=normalized_policy,
            dataset=dataset,
            max_positions=max_positions,
            rotation=rotation,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
        )
        if identity is None:
            continue
        if not bool(identity.get("registered", False)):
            raise RuntimeError(
                "canonical strategy parameter profile尚未註冊scientific fingerprint: "
                f"profile={identity.get('profile_id')}"
            )
        if not bool(identity.get("registration_matches", False)):
            raise RuntimeError(
                "canonical strategy parameter registered scientific contract drift: "
                f"profile={identity.get('profile_id')}, "
                f"registered={identity.get('registered_scientific_fingerprint_sha256')}, "
                f"current={identity.get('scientific_fingerprint_sha256')}"
            )
        identities[normalized_policy] = identity
    return identities


def _canonical_manifest_matches_current_schedule(
    root: Path, *, family: str, policy: str, target: Path,
    evaluation_mode: str = "rolling", comparison_end_date: str | None = None,
    dataset: str | None = None, max_positions: int | None = None,
    rotation: str | None = None, fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
) -> tuple[bool, str]:
    scientific_identity = _canonical_schedule_scientific_identity(
        root,
        family=family,
        policy=policy,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    if scientific_identity is not None:
        if not bool(scientific_identity.get("registered", False)):
            return False, "SCIENTIFIC_IDENTITY_UNREGISTERED"
        if not bool(scientific_identity.get("registration_matches", False)):
            return False, "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT"

    manifest_path = resolve_strategy_param_manifest_path(
        root, family=family, evaluation_mode="rolling"
    )
    manifest = _load_json_object(manifest_path)
    if manifest is None:
        return False, "MANIFEST_MISSING_OR_INVALID"
    manifest_issue = first_manifest_identity_mismatch(
        manifest,
        (
            ("producer", "string", "optimizer", "PRODUCER_MISMATCH"),
            ("family", "string", family, "FAMILY_MISMATCH"),
            ("evaluation_mode", "string", "schedule", "STORAGE_MODE_MISMATCH"),
        ),
    )
    if manifest_issue is not None:
        return False, manifest_issue
    expected_policy = _canonical_schedule_training_policy(family)
    actual_policy = dict(manifest.get("training_policy") or {})
    if _schedule_training_policy_reuse_projection(actual_policy) != (
        _schedule_training_policy_reuse_projection(expected_policy)
    ):
        return False, "TRAINING_POLICY_MISMATCH"
    record = dict(dict(manifest.get("artifacts") or {}).get(policy) or {})
    if not record:
        return False, "ARTIFACT_NOT_PINNED"
    if str(record.get("path") or "") != _project_relative(root, target):
        return False, "ARTIFACT_PATH_MISMATCH"
    if not target.is_file():
        return False, "ARTIFACT_MISSING"
    if not manifest_artifact_sha_matches(record, target, sha_field="sha256"):
        return False, "ARTIFACT_SHA_MISMATCH"
    artifact_payload = _load_json_object(target)
    if artifact_payload is None:
        return False, "ARTIFACT_PAYLOAD_INVALID"
    if str(artifact_payload.get("selector") or "") != normalize_strategy_param_policy(policy):
        return False, "POLICY_SELECTOR_MISMATCH"
    source = dict(record.get("source") or {})
    expected_build_contract = _canonical_schedule_build_contract(
        family=family,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    if dict(source.get("build_contract") or {}) != expected_build_contract:
        return False, "BUILD_CONTRACT_MISMATCH"
    schedule_kind = str(source.get("schedule_kind") or "")
    if schedule_kind != "annual_refit":
        return False, "SCHEDULE_KIND_MISSING_OR_INVALID"

    if scientific_identity is not None:
        persisted_identity = dict(source.get("scientific_identity") or {})
        if not persisted_identity:
            return False, "LEGACY_SCIENTIFIC_IDENTITY_UNBOUND"
        expected_record = _scientific_identity_record(scientific_identity)
        if persisted_identity != expected_record:
            return False, "SCIENTIFIC_IDENTITY_MISMATCH"

    if comparison_end_date not in (None, ""):
        coverage_end = _canonical_schedule_coverage_end(target)
        if coverage_end is None or str(coverage_end) < str(comparison_end_date):
            return False, "SCHEDULE_COVERAGE_INCOMPLETE"
    return True, "READY"


def validate_strategy_parameter_artifact_identity(
    project_root: str | Path, *, family: str, policy: str,
    evaluation_mode: str = "rolling", comparison_end_date: str | None = None,
    dataset: str | None = None, max_positions: int | None = None,
    rotation: str | None = None, fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
) -> tuple[bool, str, Path]:
    """Validate one current canonical schedule against the live Optimizer contract."""
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    policy = normalize_strategy_param_policy(policy)
    target = resolve_strategy_param_artifact_path(
        root, family=family, evaluation_mode="rolling", policy=policy
    )
    ready, status = _canonical_manifest_matches_current_schedule(
        root, family=family, policy=policy, target=target,
        evaluation_mode=evaluation_mode, comparison_end_date=comparison_end_date,
        dataset=dataset, max_positions=max_positions, rotation=rotation,
        fixed_risk=fixed_risk, max_position_cap_pct=max_position_cap_pct,
    )
    return ready, status, target


def _canonical_schedule_coverage_end(path: Path) -> str | None:
    if not path.is_file():
        return None
    payload = _load_json_object(path)
    if payload is None or not is_active_param_ensemble_payload(payload):
        return None
    if resolve_active_param_ensemble_mode(payload) != ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING:
        return None
    try:
        _start, end = get_active_param_ensemble_date_range(payload)
    except (TypeError, ValueError, KeyError):
        return None
    return str(end) if end else None


def _compare_policy_name(policy: str) -> str:
    normalized = normalize_strategy_param_policy(policy)
    reverse = {
        optimizer_policy: compare_policy
        for compare_policy, optimizer_policy in COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.items()
    }
    if normalized not in reverse:
        raise ValueError(
            "current canonical auto-build只支援Strategy Compare正式policy: "
            f"{normalized}"
        )
    return reverse[normalized]


def _build_canonical_schedule(
    root: Path, *, family: str, policy: str, evaluation_mode: str, comparison_end_date: str,
    dataset: str | None = None, max_positions: int | None = None,
    rotation: str | None = None, fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None, force_rebuild: bool = False,
    resume_parameter_training: bool = True,
) -> dict[str, Any]:
    from services.optimizer.strategy_param_training import (
        prepare_selection_historical_full_roos_params,
        prepare_selection_historical_p2_params,
    )

    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    if mode not in {"oos", "rolling"}:
        raise ValueError("canonical schedule auto-build只支援oos/rolling")
    contract = _canonical_schedule_build_contract(
        family=family,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    compare_policy = _compare_policy_name(policy)
    scientific_identities = _canonical_family_scientific_identities(
        root,
        family=family,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    # One optimizer-owned 2021->latest rolling schedule is the only physical truth.
    # OOS is an in-memory consumption view applied by Strategy Compare, never a
    # second 2021-only training job or frozen parameter JSON.
    canonical_work_root = resolve_strategy_param_optimizer_work_dir(
        root, scope=STRATEGY_PARAM_WORK_SCOPE_CANONICAL, family=family
    )
    if force_rebuild:
        work_root = _force_rebuild_attempt_root(Path(canonical_work_root))
    else:
        legacy_work_root = (
            root / STRATEGY_PARAM_WORK_ROOT_RELATIVE / str(family) / "shared_policy_search"
        )
        work_root = _adopt_legacy_optimizer_work_root(
            current_root=canonical_work_root,
            legacy_root=legacy_work_root,
            prune_stop=root / STRATEGY_PARAM_WORK_ROOT_RELATIVE,
        )
    work_identity = {
        "schema": "canonical_strategy_schedule_work_v4",
        "family": family,
        "build_contract": contract,
        "policy_outputs": list(COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.values()),
    }
    work_contract_path = work_root / "work_contract.json"
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(work_contract_path, work_identity)
    schedule_work = work_root / "schedule_2021_forward"
    legacy_initial_work = work_root / "initial_2021"
    if legacy_initial_work.is_dir() and not schedule_work.exists():
        # Preserve/resume compatible 2021 study state from the retired split producer.
        shutil.move(str(legacy_initial_work), str(schedule_work))

    common = dict(
        project_root=root,
        dataset=str(contract["dataset"]),
        param_policy=compare_policy,
        trials_per_fold=int(contract["trials_per_fold"]),
        max_positions=int(contract["max_positions"]),
        rotation=str(contract["rotation"]),
        fixed_risk=float(contract["fixed_risk"]),
        max_position_cap_pct=float(contract["max_position_cap_pct"]),
        optimizer_seed=int(contract["optimizer_seed"]),
        resume_parameter_training=(False if force_rebuild else bool(resume_parameter_training)),
        quiet=False,
        first_oos_date=str(contract["schedule_start"]),
        last_oos_date=str(comparison_end_date),
        train_window_months=int(contract["train_window_months"]),
        oos_months=int(contract["oos_horizon_months"]),
        output_relative_dir=schedule_work,
        build_context_label="Canonical",
        optimizer_output_title="Optimizer 工作工件（非正式策略參數）",
    )
    if family == "full":
        result = prepare_selection_historical_full_roos_params(**common)
    else:
        result = prepare_selection_historical_p2_params(
            **common, recover_completed_strategy_compare=False
        )
    source_path = Path(result["params_path"]).resolve()
    _assert_optimizer_work_source(source_path=source_path, work_root=work_root)
    if not source_path.is_file():
        raise FileNotFoundError(
            "Optimizer完成但找不到策略參數輸出: " + _project_relative(root, source_path)
        )

    emitted = _optimizer_compare_policy_outputs(source_path)
    if policy not in emitted:
        raise FileNotFoundError(
            "Optimizer完成但requested policy工作工件缺少: "
            f"policy={policy}, source_dir={_project_relative(root, source_path.parent)}"
        )
    published: dict[str, Path] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for emitted_policy, emitted_path in emitted.items():
        target_path = resolve_strategy_param_artifact_path(
            root, family=family, evaluation_mode="rolling", policy=emitted_policy
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_json_payload(emitted_path, target_path)
        published[emitted_policy] = target_path
        source_records[emitted_policy] = {
            "optimizer_build": True,
            "path": _project_relative(root, emitted_path),
            "build_contract": contract,
            "schedule_kind": "annual_refit",
            "same_json_oos_rolling": True,
        }
        scientific_identity = scientific_identities.get(emitted_policy)
        if scientific_identity is not None:
            source_records[emitted_policy]["scientific_identity"] = (
                _scientific_identity_record(scientific_identity)
            )
    manifest = refresh_strategy_parameter_manifest(
        root,
        family=family,
        evaluation_mode="rolling",
        source_records=source_records,
        training_policy_override=_canonical_schedule_training_policy(family),
    )
    target = published[policy]
    _print_published_strategy_param_paths(
        root, title="Canonical 正式策略參數", paths=published, manifest_path=manifest
    )
    return {
        "path": target,
        "manifest_path": manifest,
        "sha256": compute_strategy_param_file_sha256(target),
        "schedule_kind": "annual_refit",
        "published_paths": published,
    }

def build_isolated_strategy_parameter_schedule(
    project_root: str | Path,
    *,
    family: str,
    policy: str,
    comparison_end_date: str,
    dataset: str,
    selected_data_dir: str | Path,
    raw_data_loader,
    strategy_params_root: str | Path,
    models_dir: str | Path,
    work_root: str | Path,
    runtime_domain: str,
    data_identity: dict[str, Any],
    optimizer_seed_override: int | None = None,
    trials_per_fold_override: int | None = None,
    seed_ensemble_policy_override: dict[str, Any] | None = None,
    optimizer_search_protocol: str | None = None,
    parameter_set_identity: str | None = None,
    score_policy_overrides: dict[str, Any] | None = None,
    resume_parameter_training: bool = True,
    environ=None,
) -> dict[str, Any]:
    """Build one PIT schedule with the canonical Optimizer into an isolated domain.

    The search/trial/seed/window semantics come exclusively from the same rolling
    Strategy Parameter contract used by Research.  Callers may isolate only storage
    and data access; they may not redefine the scientific training policy.
    """
    from services.optimizer.strategy_param_training import (
        prepare_selection_historical_full_roos_params,
    )

    root = Path(project_root).resolve()
    normalized_family = normalize_strategy_param_family(family)
    normalized_policy = normalize_strategy_param_policy(policy)
    if normalized_family != "full":
        raise ValueError("isolated PIT schedule目前只支援Trading使用的full family")
    end_date = str(comparison_end_date or "").strip()
    if not end_date:
        raise ValueError("isolated PIT schedule缺少comparison_end_date")
    domain = str(runtime_domain or "").strip()
    if not domain:
        raise ValueError("isolated PIT schedule缺少runtime_domain")
    if not isinstance(data_identity, dict) or not data_identity:
        raise ValueError("isolated PIT schedule缺少data_identity")

    contract = _canonical_schedule_build_contract(
        family=normalized_family,
        dataset=str(dataset),
    )
    if optimizer_seed_override is not None:
        override_seed = int(optimizer_seed_override)
        if override_seed <= 0:
            raise ValueError("isolated PIT schedule optimizer_seed_override必須>0")
        contract = {**dict(contract), "optimizer_seed": override_seed}
    if trials_per_fold_override is not None:
        override_trials = int(trials_per_fold_override)
        if override_trials <= 0:
            raise ValueError("isolated PIT schedule trials_per_fold_override必須>0")
        contract = {**dict(contract), "trials_per_fold": override_trials}
    if seed_ensemble_policy_override is not None:
        if not isinstance(seed_ensemble_policy_override, dict) or not seed_ensemble_policy_override:
            raise ValueError("isolated PIT schedule seed_ensemble_policy_override必須為非空mapping")
        contract = {**dict(contract), "seed_ensemble_policy": dict(seed_ensemble_policy_override)}
    normalized_search_protocol = str(optimizer_search_protocol or "").strip()
    normalized_parameter_set = str(parameter_set_identity or "").strip()
    if normalized_search_protocol:
        contract = {**dict(contract), "optimizer_search_protocol": normalized_search_protocol}
    if normalized_parameter_set:
        contract = {**dict(contract), "parameter_set": normalized_parameter_set}
    compare_policy = _compare_policy_name(normalized_policy)
    isolated_work_root = Path(work_root).resolve()
    schedule_work = isolated_work_root / "schedule_2021_forward"
    isolated_work_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        isolated_work_root / "work_contract.json",
        {
            "schema": "isolated_strategy_schedule_work_v1",
            "runtime_domain": domain,
            "family": normalized_family,
            "build_contract": contract,
            "data_identity": dict(data_identity),
            "policy_outputs": list(COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.values()),
        },
    )

    result = prepare_selection_historical_full_roos_params(
        project_root=root,
        dataset=str(contract["dataset"]),
        param_policy=compare_policy,
        trials_per_fold=int(contract["trials_per_fold"]),
        max_positions=int(contract["max_positions"]),
        rotation=str(contract["rotation"]),
        fixed_risk=float(contract["fixed_risk"]),
        max_position_cap_pct=float(contract["max_position_cap_pct"]),
        optimizer_seed=int(contract["optimizer_seed"]),
        resume_parameter_training=bool(resume_parameter_training),
        quiet=False,
        first_oos_date=str(contract["schedule_start"]),
        last_oos_date=end_date,
        train_window_months=int(contract["train_window_months"]),
        oos_months=int(contract["oos_horizon_months"]),
        output_relative_dir=schedule_work,
        build_context_label=f"{domain.capitalize()} PIT",
        optimizer_output_title="Optimizer 工作工件（非正式策略參數）",
        selected_data_dir_override=selected_data_dir,
        raw_data_loader=raw_data_loader,
        latest_data_date_override=end_date,
        models_dir_override=models_dir,
        dataset_identity_override=dict(data_identity),
        environ_override=environ,
        optimizer_search_protocol=(normalized_search_protocol or None),
        parameter_set_identity=(normalized_parameter_set or "P4_HISTORY"),
        seed_ensemble_policy_override=(
            None if seed_ensemble_policy_override is None else dict(seed_ensemble_policy_override)
        ),
        score_policy_overrides=(
            None if score_policy_overrides is None else dict(score_policy_overrides)
        ),
    )
    source_path = Path(result["params_path"]).resolve()
    _assert_optimizer_work_source(source_path=source_path, work_root=isolated_work_root)
    emitted = _optimizer_compare_policy_outputs(source_path)
    if normalized_policy not in emitted:
        raise FileNotFoundError(
            "Optimizer完成但requested policy工作工件缺少: "
            f"policy={normalized_policy}, source_dir={source_path.parent.as_posix()}"
        )

    runtime_strategy_params_root = Path(strategy_params_root).resolve()
    published: dict[str, Path] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for emitted_policy, emitted_path in emitted.items():
        target_path = resolve_strategy_param_artifact_path(
            root,
            family=normalized_family,
            evaluation_mode="rolling",
            policy=emitted_policy,
            strategy_params_root=runtime_strategy_params_root,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_json_payload(emitted_path, target_path)
        published[emitted_policy] = target_path
        source_records[emitted_policy] = {
            "optimizer_build": True,
            "path": _project_relative(root, emitted_path),
            "build_contract": contract,
            "schedule_kind": "annual_refit",
            "same_json_oos_rolling": True,
            "runtime_domain": domain,
            "data_identity_sha256": canonical_json_sha256(data_identity),
        }

    training_policy = _canonical_schedule_training_policy(normalized_family)
    training_policy["optimizer_seed"] = int(contract["optimizer_seed"])
    training_policy["trials_per_fold"] = int(contract["trials_per_fold"])
    training_policy["runtime_domain"] = domain
    training_policy["data_source"] = "trading_market_data_v2_historical_latest_view" if domain == "trading" else domain
    manifest = refresh_strategy_parameter_manifest(
        root,
        family=normalized_family,
        evaluation_mode="rolling",
        source_records=source_records,
        strategy_params_root=runtime_strategy_params_root,
        training_policy_override=training_policy,
    )
    target = published[normalized_policy]
    coverage_end = _canonical_schedule_coverage_end(target)
    if coverage_end is None or str(coverage_end) < end_date:
        raise RuntimeError(
            "isolated PIT schedule coverage不足: "
            f"required={end_date}, actual={coverage_end or '-'}"
        )
    return {
        "status": "READY",
        "path": target,
        "manifest_path": manifest,
        "sha256": compute_strategy_param_file_sha256(target),
        "schedule_kind": "annual_refit",
        "published_paths": published,
        "build_contract": contract,
        "coverage_end": coverage_end,
    }


_CANONICAL_SCIENTIFIC_IDENTITY_BLOCKED_STATUSES = {
    "SCIENTIFIC_IDENTITY_UNREGISTERED",
    "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT",
}


def inspect_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    family: str,
    evaluation_mode: str,
    policy: str,
    comparison_end_date: str | None = None,
    dataset: str | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
) -> dict[str, Any]:
    """Inspect one canonical Optimizer-owned schedule without creating it."""

    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    policy = normalize_strategy_param_policy(policy)
    target = resolve_strategy_param_artifact_path(
        root, family=family, evaluation_mode=mode, policy=policy
    )
    coverage_end = _canonical_schedule_coverage_end(target)
    if mode not in {"oos", "rolling"}:
        ready = bool(target.is_file())
        return {
            "ready": ready,
            "action": "REUSE" if ready else "BUILD",
            "status": "READY" if ready else "ARTIFACT_MISSING",
            "path": target,
            "coverage_end": coverage_end,
        }

    identity_ready, identity_status = _canonical_manifest_matches_current_schedule(
        root, family=family, policy=policy, target=target,
        evaluation_mode=mode, comparison_end_date=comparison_end_date,
        dataset=dataset, max_positions=max_positions, rotation=rotation,
        fixed_risk=fixed_risk, max_position_cap_pct=max_position_cap_pct,
    )
    coverage_ready = bool(
        comparison_end_date in (None, "")
        or (coverage_end is not None and str(coverage_end) >= str(comparison_end_date))
    )
    ready = bool(target.is_file() and identity_ready and coverage_ready)
    if ready:
        status = "READY"
    elif (
        not identity_ready
        and str(identity_status) in _CANONICAL_SCIENTIFIC_IDENTITY_BLOCKED_STATUSES
    ):
        status = str(identity_status)
    elif not target.is_file():
        status = "ARTIFACT_MISSING"
    elif not identity_ready:
        status = str(identity_status)
    else:
        status = "SCHEDULE_COVERAGE_INCOMPLETE"
    return {
        "ready": ready,
        "action": (
            "REUSE"
            if ready
            else (
                "BLOCKED"
                if status in _CANONICAL_SCIENTIFIC_IDENTITY_BLOCKED_STATUSES
                else ("REBUILD" if target.is_file() else "BUILD")
            )
        ),
        "status": status,
        "path": target,
        "coverage_end": coverage_end,
        "identity_status": identity_status,
    }


def ensure_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    family: str,
    evaluation_mode: str,
    policy: str,
    comparison_end_date: str | None = None,
    dataset: str | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
    force_rebuild: bool = False,
    resume_parameter_training: bool = True,
) -> dict[str, Any]:
    """Resolve/reuse/build one canonical strategy schedule.

    For current Strategy Compare, OOS and Rolling always resolve to the same cross-time
    JSON.  Staleness is determined from Optimizer training-policy + build contract +
    payload SHA, never from mere file existence.  Missing/stale schedules are delegated
    to the canonical Optimizer builder and may resume existing optimizer work.
    """
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    policy = normalize_strategy_param_policy(policy)
    target = resolve_strategy_param_artifact_path(
        root, family=family, evaluation_mode=mode, policy=policy
    )

    if mode not in {"oos", "rolling"}:
        if not target.is_file():
            raise FileNotFoundError(
                "缺少canonical策略參數工件；非OOS/Rolling模式請由Optimizer正式流程建立: "
                f"family={family}, mode={mode}, policy={policy}, target={_project_relative(root, target)}"
            )
        manifest = refresh_strategy_parameter_manifest(
            root, family=family, evaluation_mode=mode
        )
        return {
            "action": "REUSE", "family": family, "evaluation_mode": mode,
            "policy": policy, "path": target, "manifest_path": manifest,
            "sha256": compute_strategy_param_file_sha256(target), "migration": None,
        }

    identity_ready, identity_status = _canonical_manifest_matches_current_schedule(
        root, family=family, policy=policy, target=target,
        evaluation_mode=mode, comparison_end_date=comparison_end_date,
        dataset=dataset, max_positions=max_positions, rotation=rotation,
        fixed_risk=fixed_risk, max_position_cap_pct=max_position_cap_pct,
    )
    coverage_end = _canonical_schedule_coverage_end(target)
    if (
        not identity_ready
        and str(identity_status) in _CANONICAL_SCIENTIFIC_IDENTITY_BLOCKED_STATUSES
    ):
        raise RuntimeError(
            "canonical strategy parameter scientific identity blocked: "
            f"status={identity_status}, family={family}, policy={policy}"
        )
    coverage_ready = bool(
        comparison_end_date in (None, "")
        or (coverage_end is not None and str(coverage_end) >= str(comparison_end_date))
    )
    if target.is_file() and identity_ready and coverage_ready and not force_rebuild:
        return {
            "action": "REUSE", "family": family, "evaluation_mode": mode,
            "policy": policy, "path": target,
            "manifest_path": resolve_strategy_param_manifest_path(
                root, family=family, evaluation_mode="rolling"
            ),
            "sha256": compute_strategy_param_file_sha256(target),
            "identity_status": identity_status,
            "coverage_end": coverage_end,
            "migration": None,
        }

    if comparison_end_date in (None, ""):
        reason = identity_status if not identity_ready else "COVERAGE_END_REQUIRED"
        raise FileNotFoundError(
            "canonical策略參數缺少／過期且無法判定需建立到哪個比較終點；"
            f"status={reason}, family={family}, policy={policy}, target={_project_relative(root, target)}"
        )

    existed = target.is_file()
    build = _build_canonical_schedule(
        root, family=family, policy=policy, evaluation_mode=mode,
        comparison_end_date=str(comparison_end_date),
        dataset=dataset, max_positions=max_positions, rotation=rotation,
        fixed_risk=fixed_risk, max_position_cap_pct=max_position_cap_pct,
        force_rebuild=bool(force_rebuild),
        resume_parameter_training=bool(resume_parameter_training),
    )
    final_ready, final_status = _canonical_manifest_matches_current_schedule(
        root, family=family, policy=policy, target=target,
        evaluation_mode=mode, comparison_end_date=str(comparison_end_date),
        dataset=dataset, max_positions=max_positions, rotation=rotation,
        fixed_risk=fixed_risk, max_position_cap_pct=max_position_cap_pct,
    )
    final_coverage = _canonical_schedule_coverage_end(target)
    if not final_ready or final_coverage is None or str(final_coverage) < str(comparison_end_date):
        raise RuntimeError(
            "Optimizer建立後canonical策略參數仍不符合current contract: "
            f"identity={final_status}, coverage={final_coverage}, required={comparison_end_date}"
        )
    return {
        "action": "FORCE_REBUILD" if force_rebuild else ("REBUILD" if existed else "BUILD"),
        "family": family,
        "evaluation_mode": mode,
        "policy": policy,
        "path": target,
        "manifest_path": Path(build["manifest_path"]),
        "sha256": compute_strategy_param_file_sha256(target),
        "identity_status": final_status,
        "coverage_end": final_coverage,
        "migration": None,
    }


def _benchmark_optimizer_work_root(
    root: Path, *, benchmark_id: str, seed: int, family: str
) -> Path:
    current_root = resolve_strategy_param_optimizer_work_dir(
        root,
        scope=STRATEGY_PARAM_WORK_SCOPE_BENCHMARK,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        family=family,
    )
    benchmark_dir = resolve_strategy_param_benchmark_dir(
        root,
        benchmark_id=benchmark_id,
        seed=int(seed),
        family=family,
        evaluation_mode="rolling",
    )
    legacy_root = benchmark_dir / "_optimizer_work" / f"seed_{int(seed)}" / str(family)
    return _adopt_legacy_optimizer_work_root(
        current_root=current_root, legacy_root=legacy_root, prune_stop=benchmark_dir
    )


def _copy_json_payload(source: Path, target: Path) -> dict[str, Any]:
    payload = _load_json_object_permissive(source)
    if payload is None:
        raise ValueError(f"策略參數source無法讀取或不是JSON object: {source}")
    normalized = normalize_strategy_param_payload_for_persistence(payload)
    if normalized != payload:
        # One-time repair for work artifacts emitted before strict-JSON publication
        # was enforced. Runtime params are validated before any normalization.
        atomic_write_json(source, normalized)
    _write_json(target, normalized)
    return normalized


def _benchmark_effective_member_sha256(path: Path, effective_date: str = "2021-01-01") -> str:
    payload = load_json_strict(path)
    mapping = dict(payload.get("params_ensemble_by_effective_date") or {})
    members = mapping.get(str(effective_date))
    if not isinstance(members, list) or not members:
        raise ValueError(f"benchmark策略參數缺少{effective_date} member: {path}")
    return canonical_json_sha256(members)


def _benchmark_parameter_coverage_end(path: Path) -> str | None:
    try:
        payload = load_json_strict(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    meta = dict(payload.get("meta") or {})
    summary = dict(payload.get("summary") or {})
    value = meta.get("last_oos_date") or summary.get("oos_end_date")
    return None if value in (None, "") else str(value)


def _benchmark_training_policy_snapshot(*, family: str, seed: int) -> dict[str, Any]:
    policy = _canonical_schedule_training_policy(family)
    policy["optimizer_seed"] = int(seed)
    policy["benchmark_seed_override"] = True
    return policy


def _benchmark_schedule_build_contract(
    *, benchmark_id: str, seed: int, family: str, policy: str,
    dataset: str, max_positions: int, rotation: str, fixed_risk: float,
    max_position_cap_pct: float,
) -> dict[str, Any]:
    from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE

    optimizer_policy = _benchmark_training_policy_snapshot(family=family, seed=int(seed))
    fitting_protocol = _canonical_family_fitting_seed_protocol(family)
    return {
        "schema": "robustness_strategy_schedule_work_v3",
        "benchmark_id": str(benchmark_id),
        "seed": int(seed),
        "family": normalize_strategy_param_family(family),
        "policy": normalize_strategy_param_policy(policy),
        "trials_per_fold": int(fitting_protocol.trials_per_seed),
        "dataset": str(dataset),
        "max_positions": int(max_positions),
        "rotation": str(rotation),
        "fixed_risk": float(fixed_risk),
        "max_position_cap_pct": float(max_position_cap_pct),
        "train_window_months": int(optimizer_policy["train_window_months"]),
        "oos_months": int(optimizer_policy["oos_horizon_months"]),
        "search_space_sha256": canonical_json_sha256(BREAKOUT_OPTIMIZER_SEARCH_SPACE),
    }


def _benchmark_build_contract_scientific_payload(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the fitted-parameter inputs, excluding workspace schema metadata.

    Workspace/manifest schema revisions are publication engineering, not a reason to
    rerun an already-completed optimizer when the fitted payload and every optimizer
    input are unchanged.
    """

    return {
        str(key): value
        for key, value in dict(contract or {}).items()
        if str(key) not in {"schema", "policy"}
    }


def _benchmark_scientific_identity(
    root: Path,
    *,
    seed: int,
    family: str,
    policy: str,
    dataset: str,
    max_positions: int,
    rotation: str,
    fixed_risk: float,
    max_position_cap_pct: float,
) -> dict[str, Any]:
    profile = resolve_canonical_strategy_param_training_profile(family, policy)
    if profile is None:
        raise ValueError(
            "robustness benchmark缺少registered canonical strategy-param profile: "
            f"family={family}, policy={policy}"
        )
    if str(dataset) != str(profile.dataset):
        raise ValueError(
            "registered robustness strategy parameter identity不得覆寫dataset: "
            f"profile={profile.profile_id}, registered={profile.dataset!r}, requested={dataset!r}"
        )
    protocol = get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)
    if protocol.enabled or protocol.seed_count != 1:
        raise ValueError(
            "single-seed robustness baseline必須源自single-seed fitting profile: "
            f"profile={profile.profile_id}, protocol={protocol.protocol_id}"
        )
    fitting = dict(protocol.as_seed_ensemble_policy())
    fitting["generator_seed"] = int(seed)
    fitting["resolved_seeds"] = [int(seed)]
    fitting["trials_per_seed"] = int(protocol.trials_per_seed)
    return derive_strategy_param_scientific_identity_from_registered_profile(
        root,
        profile.profile_id,
        fitting_override=fitting,
        execution_overrides=_strategy_param_execution_overrides(
            max_positions=int(max_positions),
            rotation=str(rotation),
            fixed_risk=float(fixed_risk),
            max_position_cap_pct=float(max_position_cap_pct),
        ),
        reference_overrides={
            "robustness_fitting_seed": int(seed),
            "robustness_seed_semantics": "benchmark_seed_overrides_single_fitting_seed",
        },
    )


def _benchmark_workspace_source_path(
    work_root: Path, *, family: str, policy: str
) -> Path:
    schedule_root = Path(work_root) / "schedule_2021_forward"
    if normalize_strategy_param_family(family) == "min":
        schedule_root = schedule_root / "p2_dl_off_trained"
    filename = f"roos_{POLICY_FILENAME_BY_NAME[normalize_strategy_param_policy(policy)]}"
    return schedule_root / "active_params" / filename


def _benchmark_optimizer_work_roots_for_inspection(
    root: Path, *, benchmark_id: str, seed: int, family: str
) -> tuple[Path, ...]:
    """Return current + legacy Optimizer work roots without mutating either tree."""

    current_root = resolve_strategy_param_optimizer_work_dir(
        root,
        scope=STRATEGY_PARAM_WORK_SCOPE_BENCHMARK,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        family=normalize_strategy_param_family(family),
    )
    benchmark_dir = resolve_strategy_param_benchmark_dir(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        family=normalize_strategy_param_family(family),
        evaluation_mode="rolling",
    )
    legacy_root = (
        benchmark_dir
        / "_optimizer_work"
        / f"seed_{int(seed)}"
        / normalize_strategy_param_family(family)
    )
    roots: list[Path] = []
    for candidate in (Path(current_root), Path(legacy_root)):
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def _benchmark_fast_republish_source(
    *, root: Path, benchmark_id: str, seed: int, family: str, policy: str,
    target_path: Path, manifest_path: Path, build_contract: dict[str, Any],
    scientific_identity: dict[str, Any], comparison_end_date: str,
) -> Path | None:
    """Return proof that publication metadata can be repaired without training.

    A bound manifest whose scientific hash/identity/contract still match is already
    sufficient proof when only raw publication bytes drifted.  Otherwise a retained
    current or legacy Optimizer workspace may prove the same scientific payload.
    The helper is read-only so rendering a plan never migrates historical workspace.
    """

    if not target_path.is_file():
        return None
    target_payload = _load_json_object(target_path)
    if target_payload is None:
        return None
    if str(target_payload.get("selector") or "") != normalize_strategy_param_policy(policy):
        return None
    coverage_end = _benchmark_parameter_coverage_end(target_path)
    if coverage_end is None or str(coverage_end) < str(comparison_end_date):
        return None
    manifest = _load_json_object(manifest_path)
    artifact = dict(dict((manifest or {}).get("artifacts") or {}).get(policy) or {})
    source = dict(artifact.get("source") or {})
    expected_identity = _derived_scientific_identity_record(scientific_identity)
    if dict(source.get("scientific_identity") or {}) != expected_identity:
        return None
    target_scientific_sha = compute_strategy_param_scientific_sha256(target_path)
    manifest_scientific_sha = str(artifact.get("scientific_sha256") or "").strip().lower()
    manifest_contract_matches = (
        _benchmark_build_contract_scientific_payload(
            dict(source.get("build_contract") or {})
        )
        == _benchmark_build_contract_scientific_payload(build_contract)
    )
    if (
        manifest_contract_matches
        and manifest_scientific_sha
        and manifest_scientific_sha == target_scientific_sha
    ):
        # The manifest is already bound to the exact registered scientific identity
        # and fitted-input contract.  A raw SHA mismatch is publication-only drift.
        return target_path.resolve()

    for work_root in _benchmark_optimizer_work_roots_for_inspection(
        root, benchmark_id=str(benchmark_id), seed=int(seed), family=family
    ):
        work_contract_path = Path(work_root) / "work_contract.json"
        source_path = _benchmark_workspace_source_path(
            Path(work_root), family=family, policy=policy
        )
        if not work_contract_path.is_file() or not source_path.is_file():
            continue
        try:
            retained_contract = load_json_strict(work_contract_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(retained_contract, dict):
            continue
        if _benchmark_build_contract_scientific_payload(retained_contract) != (
            _benchmark_build_contract_scientific_payload(build_contract)
        ):
            continue
        if _load_json_object_permissive(source_path) is None:
            continue
        if compute_strategy_param_scientific_sha256(source_path) != target_scientific_sha:
            continue
        return source_path.resolve()
    return None


def _benchmark_manifest_matches_current_policy(
    manifest_path: Path, *, benchmark: dict[str, Any], benchmark_id: str, seed: int,
    family: str, evaluation_mode: str, policy: str, target_path: Path,
    comparison_end_date: str, build_contract: dict[str, Any],
    scientific_identity: dict[str, Any],
) -> bool:
    try:
        payload = load_json_strict(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    if first_manifest_identity_mismatch(
        payload,
        (
            ("benchmark_id", "string", benchmark_id, "BENCHMARK_ID_MISMATCH"),
            ("benchmark_seed", "integer", seed, "BENCHMARK_SEED_MISMATCH"),
            ("family", "string", family, "FAMILY_MISMATCH"),
            ("evaluation_mode", "string", "schedule", "STORAGE_MODE_MISMATCH"),
        ),
    ) is not None:
        return False
    pinned = dict(payload.get("benchmark") or {})
    # A per-seed Optimizer artifact is identified by its concrete seed plus the
    # Optimizer build/training contract below.  ``seed_count`` / ``resolved_seeds``
    # describe the current benchmark *membership*, not the scientific identity of
    # an already-fitted seed.  Expanding N=2 -> N=4 must therefore leave the first
    # two per-seed artifacts reusable when every fitting input is unchanged.
    if str(pinned.get("benchmark_id") or "") != str(benchmark.get("benchmark_id") or ""):
        return False
    training = dict(payload.get("training_policy") or {})
    if (
        int(training.get("optimizer_seed") or -1) != int(seed)
        or int(training.get("trials_per_fold") or -1)
            != int(build_contract["trials_per_fold"])
        or training.get("benchmark_seed_override") is not True
        or str(training.get("evaluation_mode") or "") != "schedule"
    ):
        return False
    artifact = dict(dict(payload.get("artifacts") or {}).get(policy) or {})
    expected_sha = str(artifact.get("sha256") or "")
    if not expected_sha or not target_path.is_file():
        return False
    if not manifest_artifact_sha_matches(artifact, target_path, sha_field="sha256"):
        return False
    expected_scientific_sha = str(artifact.get("scientific_sha256") or "").strip().lower()
    actual_scientific_sha = compute_strategy_param_scientific_sha256(target_path)
    if expected_scientific_sha and expected_scientific_sha != actual_scientific_sha:
        return False
    artifact_payload = _load_json_object(target_path)
    if artifact_payload is None:
        return False
    if str(artifact_payload.get("selector") or "") != normalize_strategy_param_policy(policy):
        return False
    coverage_end = _benchmark_parameter_coverage_end(target_path)
    if coverage_end is None or str(coverage_end) < str(comparison_end_date):
        return False
    source = dict(artifact.get("source") or {})
    if _benchmark_build_contract_scientific_payload(
        dict(source.get("build_contract") or {})
    ) != _benchmark_build_contract_scientific_payload(build_contract):
        return False
    if dict(source.get("scientific_identity") or {}) != _derived_scientific_identity_record(
        scientific_identity
    ):
        return False
    schedule_kind = str(source.get("schedule_kind") or "")
    return schedule_kind == "annual_refit"


def _resolve_robustness_benchmark_strategy_parameter_state(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    family: str,
    evaluation_mode: str,
    policy: str,
    comparison_end_date: str,
    dataset: str,
    max_positions: int,
    rotation: str,
    fixed_risk: float,
    max_position_cap_pct: float,
) -> dict[str, Any]:
    """Resolve one per-seed benchmark artifact without performing build work."""

    root = Path(project_root).resolve()
    normalized_family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    normalized_policy = normalize_strategy_param_policy(policy)
    if mode not in {"oos", "rolling"}:
        raise ValueError("robustness benchmark策略參數只支援oos/rolling")
    supported_policies = set(COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.values())
    if normalized_policy not in supported_policies:
        raise ValueError(
            "robustness benchmark per-seed策略參數只支援Strategy Compare first-class policies: "
            + ",".join(sorted(supported_policies))
        )
    benchmark = get_robustness_benchmark_policy_snapshot()
    if str(benchmark_id) != str(benchmark["benchmark_id"]):
        raise ValueError(f"benchmark_id與current SSOT不一致: {benchmark_id}")
    if int(seed) not in {int(value) for value in benchmark["resolved_seeds"]}:
        raise ValueError(f"seed不屬於current robustness benchmark題庫: {seed}")

    optimizer_policy = _benchmark_training_policy_snapshot(
        family=normalized_family, seed=int(seed)
    )
    build_contract = _benchmark_schedule_build_contract(
        benchmark_id=str(benchmark_id), seed=int(seed), family=normalized_family,
        policy=normalized_policy, dataset=str(dataset), max_positions=int(max_positions),
        rotation=str(rotation), fixed_risk=float(fixed_risk),
        max_position_cap_pct=float(max_position_cap_pct),
    )
    scientific_identity = _benchmark_scientific_identity(
        root,
        seed=int(seed),
        family=normalized_family,
        policy=normalized_policy,
        dataset=str(dataset),
        max_positions=int(max_positions),
        rotation=str(rotation),
        fixed_risk=float(fixed_risk),
        max_position_cap_pct=float(max_position_cap_pct),
    )
    target = resolve_strategy_param_benchmark_artifact_path(
        root, benchmark_id=str(benchmark_id), seed=int(seed), family=normalized_family,
        evaluation_mode="rolling", policy=normalized_policy,
    )
    manifest_path = resolve_strategy_param_benchmark_manifest_path(
        root, benchmark_id=str(benchmark_id), seed=int(seed), family=normalized_family,
        evaluation_mode="rolling",
    )
    reusable = (
        target.is_file()
        and manifest_path.is_file()
        and (_benchmark_parameter_coverage_end(target) or "") >= str(comparison_end_date)
        and _benchmark_manifest_matches_current_policy(
            manifest_path, benchmark=benchmark, benchmark_id=str(benchmark_id),
            seed=int(seed), family=normalized_family, evaluation_mode=mode,
            policy=normalized_policy, target_path=target,
            comparison_end_date=str(comparison_end_date), build_contract=build_contract,
            scientific_identity=scientific_identity,
        )
    )
    repair_source = None
    if not reusable:
        repair_source = _benchmark_fast_republish_source(
            root=root, benchmark_id=str(benchmark_id), seed=int(seed),
            family=normalized_family, policy=normalized_policy, target_path=target,
            manifest_path=manifest_path, build_contract=build_contract,
            scientific_identity=scientific_identity,
            comparison_end_date=str(comparison_end_date),
        )
    if not bool(scientific_identity.get("source_registered", False)):
        status = "SCIENTIFIC_IDENTITY_UNREGISTERED"
        action = "BLOCKED"
    elif not bool(scientific_identity.get("source_registration_matches", False)):
        status = "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT"
        action = "BLOCKED"
    elif reusable:
        status = "READY"
        action = "REUSE"
    elif repair_source is not None:
        status = "PUBLICATION_METADATA_STALE"
        action = "REPAIR"
    elif target.exists() or manifest_path.exists():
        status = "ARTIFACT_STALE_OR_IDENTITY_UNBOUND"
        action = "REBUILD"
    else:
        status = "ARTIFACT_MISSING"
        action = "BUILD"
    return {
        "root": root,
        "family": normalized_family,
        "evaluation_mode": mode,
        "policy": normalized_policy,
        "benchmark": benchmark,
        "optimizer_policy": optimizer_policy,
        "build_contract": build_contract,
        "scientific_identity": scientific_identity,
        "status": status,
        "target": target,
        "manifest_path": manifest_path,
        "repair_source_path": repair_source,
        "action": action,
    }


def inspect_robustness_benchmark_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    family: str,
    evaluation_mode: str,
    policy: str,
    comparison_end_date: str,
    dataset: str,
    max_positions: int,
    rotation: str,
    fixed_risk: float,
    max_position_cap_pct: float,
) -> dict[str, Any]:
    """Read-only BUILD/REBUILD/REUSE decision for the robustness planner."""

    state = _resolve_robustness_benchmark_strategy_parameter_state(
        project_root,
        benchmark_id=benchmark_id,
        seed=seed,
        family=family,
        evaluation_mode=evaluation_mode,
        policy=policy,
        comparison_end_date=comparison_end_date,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    return {
        "action": str(state["action"]),
        "status": str(state["status"]),
        "path": Path(state["target"]),
        "manifest_path": Path(state["manifest_path"]),
        "scientific_identity": dict(state["scientific_identity"]),
    }


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
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Build/reuse one optimizer-owned full rolling schedule for a benchmark seed.

    The benchmark controls only the seed identity. Trials/window/cadence/parallel
    execution stay owned by the canonical Optimizer. OOS and Rolling resolve to the
    same physical 2021->latest JSON; OOS freeze is applied only in memory by replay.
    """
    state = _resolve_robustness_benchmark_strategy_parameter_state(
        project_root,
        benchmark_id=benchmark_id,
        seed=seed,
        family=family,
        evaluation_mode=evaluation_mode,
        policy=policy,
        comparison_end_date=comparison_end_date,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    root = Path(state["root"])
    family = str(state["family"])
    mode = str(state["evaluation_mode"])
    policy = str(state["policy"])
    benchmark = dict(state["benchmark"])
    optimizer_policy = dict(state["optimizer_policy"])
    trials = int(optimizer_policy["trials_per_fold"])
    train_window_months = int(optimizer_policy["train_window_months"])
    oos_months = int(optimizer_policy["oos_horizon_months"])
    build_contract = dict(state["build_contract"])
    scientific_identity = dict(state["scientific_identity"])
    target = Path(state["target"])
    manifest_path = Path(state["manifest_path"])
    if str(state["action"]) == "BLOCKED":
        raise RuntimeError(
            "robustness strategy parameter scientific identity blocked: "
            f"status={state['status']}, family={family}, policy={policy}, seed={int(seed)}"
        )
    reusable = str(state["action"]) == "REUSE" and not force_rebuild
    if reusable:
        initial_sha = _benchmark_effective_member_sha256(target)
        return {
            "action": "REUSE", "path": target, "manifest_path": manifest_path,
            "sha256": compute_strategy_param_file_sha256(target),
            "scientific_sha256": compute_strategy_param_scientific_sha256(target),
            "initial_2021_member_sha256": str(initial_sha),
        }

    if str(state["action"]) == "REPAIR" and not force_rebuild:
        repair_proof = Path(state["repair_source_path"]).resolve()
        source_path = repair_proof
        if repair_proof != target.resolve():
            # Actual execution may migrate a compatible legacy workspace into the
            # current outputs/optimizer namespace.  Plan inspection above remains
            # read-only; migration happens only when the user executes the repair.
            work_root = _benchmark_optimizer_work_root(
                root, benchmark_id=str(benchmark_id), seed=int(seed), family=family
            )
            source_path = _benchmark_workspace_source_path(
                Path(work_root), family=family, policy=policy
            ).resolve()
            _assert_optimizer_work_source(source_path=source_path, work_root=Path(work_root))
            if not source_path.is_file():
                raise FileNotFoundError(
                    "benchmark publication repair source disappeared after workspace migration: "
                    + _project_relative(root, source_path)
                )
            if compute_strategy_param_scientific_sha256(source_path) != (
                compute_strategy_param_scientific_sha256(target)
            ):
                raise RuntimeError(
                    "benchmark publication repair source scientific SHA drifted after workspace migration"
                )
            _write_json(Path(work_root) / "work_contract.json", dict(build_contract))
        else:
            old_manifest = _load_json_object(manifest_path) or {}
            old_artifact = dict(dict(old_manifest.get("artifacts") or {}).get(policy) or {})
            old_source = dict(old_artifact.get("source") or {})
            old_source_path = str(old_source.get("path") or "").strip()
            if old_source_path:
                candidate = (root / old_source_path).resolve()
                if candidate.is_file():
                    source_path = candidate

        initial_member_sha = _benchmark_effective_member_sha256(target)
        manifest = refresh_strategy_parameter_benchmark_manifest(
            root, benchmark_id=str(benchmark_id), seed=int(seed), family=family,
            evaluation_mode="rolling",
            source_records={
                policy: {
                    "benchmark_initial_2021_member_sha256": str(initial_member_sha),
                    "same_seed_oos_rolling_initial_required": True,
                    "schedule_kind": "annual_refit",
                    "same_json_oos_rolling": True,
                    "build_contract": build_contract,
                    "scientific_identity": _derived_scientific_identity_record(
                        scientific_identity
                    ),
                    "path": _project_relative(root, source_path),
                }
            },
            training_policy_override=dict(state["optimizer_policy"]),
        )
        if not bool(quiet):
            _print_published_strategy_param_paths(
                root,
                title=f"Benchmark 正式策略參數 | {benchmark_id} | seed={int(seed)} | publication repair",
                paths={policy: target},
                manifest_path=manifest,
            )
        return {
            "action": "REPAIR", "path": target, "manifest_path": manifest,
            "sha256": compute_strategy_param_file_sha256(target),
            "scientific_sha256": compute_strategy_param_scientific_sha256(target),
            "initial_2021_member_sha256": str(initial_member_sha),
        }

    canonical_work_root = _benchmark_optimizer_work_root(
        root, benchmark_id=str(benchmark_id), seed=int(seed), family=family
    )
    work_root = (
        _force_rebuild_attempt_root(Path(canonical_work_root))
        if force_rebuild
        else Path(canonical_work_root)
    )
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(work_root / "work_contract.json", dict(build_contract))
    schedule_work = work_root / "schedule_2021_forward"
    legacy_initial_work = work_root / "initial_2021"
    if legacy_initial_work.is_dir() and not schedule_work.exists():
        # Resume the already-running/completed 2021 study instead of discarding it.
        shutil.move(str(legacy_initial_work), str(schedule_work))

    from services.optimizer.strategy_param_training import (
        prepare_selection_historical_full_roos_params,
        prepare_selection_historical_p2_params,
    )
    common = dict(
        project_root=root,
        dataset=str(dataset),
        # One optimizer search produces both first-class selector outputs.  Use the
        # canonical best selector as the search anchor, then publish every available
        # Strategy Compare policy from the same workspace below.
        param_policy="base-finalist-best",
        trials_per_fold=trials,
        max_positions=int(max_positions),
        rotation=str(rotation),
        fixed_risk=float(fixed_risk),
        max_position_cap_pct=float(max_position_cap_pct),
        optimizer_seed=int(seed),
        resume_parameter_training=(False if force_rebuild else bool(resume_parameter_training)),
        quiet=bool(quiet),
        first_oos_date="2021-01-01",
        last_oos_date=str(comparison_end_date),
        train_window_months=train_window_months,
        oos_months=oos_months,
        output_relative_dir=schedule_work,
        build_context_label=f"Benchmark {benchmark_id} seed={int(seed)}",
        optimizer_output_title="Optimizer 工作工件（非正式策略參數）",
    )
    if family == "full":
        result = prepare_selection_historical_full_roos_params(**common)
    else:
        result = prepare_selection_historical_p2_params(
            **common, recover_completed_strategy_compare=False
        )
    requested_source = Path(result["params_path"]).resolve()
    _assert_optimizer_work_source(source_path=requested_source, work_root=work_root)
    if not requested_source.is_file():
        raise FileNotFoundError(
            "Benchmark Optimizer完成但找不到策略參數輸出: "
            + _project_relative(root, requested_source)
        )

    published_paths: dict[str, Path] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for compare_policy, optimizer_policy in COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.items():
        # Optimizer workspaces may emit convenience selectors that are not registered
        # canonical products for this family (for example Min + finalists-agree).
        # Publication must be driven by the canonical strategy-param registry, not by
        # whatever files happen to exist in the workspace.
        if resolve_canonical_strategy_param_training_profile(
            family, optimizer_policy
        ) is None:
            continue
        source_path = _benchmark_workspace_source_path(
            work_root, family=family, policy=optimizer_policy
        ).resolve()
        if not source_path.is_file():
            continue
        _assert_optimizer_work_source(source_path=source_path, work_root=work_root)
        policy_target = resolve_strategy_param_benchmark_artifact_path(
            root,
            benchmark_id=str(benchmark_id),
            seed=int(seed),
            family=family,
            evaluation_mode="rolling",
            policy=optimizer_policy,
        )
        policy_target.parent.mkdir(parents=True, exist_ok=True)
        _copy_json_payload(source_path, policy_target)
        initial_member_sha = _benchmark_effective_member_sha256(policy_target)
        policy_contract = _benchmark_schedule_build_contract(
            benchmark_id=str(benchmark_id),
            seed=int(seed),
            family=family,
            policy=optimizer_policy,
            dataset=str(dataset),
            max_positions=int(max_positions),
            rotation=str(rotation),
            fixed_risk=float(fixed_risk),
            max_position_cap_pct=float(max_position_cap_pct),
        )
        policy_scientific_identity = _benchmark_scientific_identity(
            root,
            seed=int(seed),
            family=family,
            policy=optimizer_policy,
            dataset=str(dataset),
            max_positions=int(max_positions),
            rotation=str(rotation),
            fixed_risk=float(fixed_risk),
            max_position_cap_pct=float(max_position_cap_pct),
        )
        source_records[optimizer_policy] = {
            "benchmark_initial_2021_member_sha256": str(initial_member_sha),
            "same_seed_oos_rolling_initial_required": True,
            "schedule_kind": "annual_refit",
            "same_json_oos_rolling": True,
            "build_contract": policy_contract,
            "scientific_identity": _derived_scientific_identity_record(
                policy_scientific_identity
            ),
            "path": _project_relative(root, source_path),
        }
        published_paths[compare_policy] = policy_target

    if policy not in source_records or not target.is_file():
        raise FileNotFoundError(
            "Benchmark Optimizer完成但requested selector輸出不存在: "
            f"family={family}, policy={policy}"
        )
    manifest = refresh_strategy_parameter_benchmark_manifest(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        family=family,
        evaluation_mode="rolling",
        source_records=source_records,
        training_policy_override=dict(state["optimizer_policy"]),
    )
    if not bool(quiet):
        _print_published_strategy_param_paths(
            root,
            title=f"Benchmark 正式策略參數 | {benchmark_id} | seed={int(seed)}",
            paths=published_paths,
            manifest_path=manifest,
        )
    requested_initial_member_sha = _benchmark_effective_member_sha256(target)
    return {
        "action": "FORCE_REBUILD" if force_rebuild else ("REBUILD" if str(state["action"]) == "REBUILD" else "BUILD"),
        "path": target,
        "manifest_path": manifest,
        "sha256": compute_strategy_param_file_sha256(target),
        "scientific_sha256": compute_strategy_param_scientific_sha256(target),
        "initial_2021_member_sha256": str(requested_initial_member_sha),
    }


_VALIDATION_PARAMSET_ROOT_RELATIVE = (
    Path("models") / "research" / "breakout_quality" / "strategy_params"
)
_VALIDATION_WORK_ROOT_RELATIVE = (
    Path("outputs") / "optimizer" / "strategy_param_schedule" / "experiments"
)


def _validation_artifact_slug(
    *, parameter_set_identity: str, validation_protocol_id: str
) -> str:
    from core.strategy_param_policy import resolve_validation_strategy_param_training_profile

    profile = resolve_validation_strategy_param_training_profile(
        parameter_set_identity, validation_protocol_id
    )
    slug = str(profile.artifact_slug or "").strip()
    if not slug or Path(slug).name != slug or slug in {".", ".."}:
        raise ValueError(
            f"validation strategy parameter artifact_slug不合法: {profile.profile_id}"
        )
    return slug


def _validation_paramset_relative(
    *, parameter_set_identity: str, validation_protocol_id: str
) -> Path:
    return _VALIDATION_PARAMSET_ROOT_RELATIVE / _validation_artifact_slug(
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )


def _validation_work_relative(
    *, parameter_set_identity: str, validation_protocol_id: str
) -> Path:
    return _VALIDATION_WORK_ROOT_RELATIVE / _validation_artifact_slug(
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )

_FIXED_MULTI_SEED_PARAMSET_ROOT_RELATIVE = (
    Path("models") / "research" / "breakout_quality" / "strategy_params"
)
_FIXED_MULTI_SEED_WORK_ROOT_RELATIVE = (
    Path("outputs") / "optimizer" / "strategy_param_schedule" / "experiments"
)
_STRATEGY_PARAM_TIMING_ROOT_RELATIVE = (
    Path("outputs") / "ml_optimizer" / "strategy_param_timing"
)
_FIXED_MULTI_SEED_SEARCH_PROTOCOL = "fixed_multiseed_v1"


def _fixed_multiseed_artifact_slug(training_profile_id: str) -> str:
    from core.strategy_param_policy import get_strategy_param_training_profile

    profile = get_strategy_param_training_profile(training_profile_id)
    slug = str(profile.artifact_slug or "").strip()
    if not slug:
        raise ValueError(
            "fixed multi-seed strategy parameter profile缺少artifact_slug: "
            f"{profile.profile_id}"
        )
    return slug


def _fixed_multiseed_paramset_relative(training_profile_id: str) -> Path:
    return _FIXED_MULTI_SEED_PARAMSET_ROOT_RELATIVE / _fixed_multiseed_artifact_slug(
        training_profile_id
    )


def _fixed_multiseed_work_relative(training_profile_id: str) -> Path:
    return _FIXED_MULTI_SEED_WORK_ROOT_RELATIVE / _fixed_multiseed_artifact_slug(
        training_profile_id
    )


def _fixed_multiseed_execution_overrides(
    *,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
) -> dict[str, Any]:
    return _strategy_param_execution_overrides(
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )


def _fixed_multiseed_scientific_identity(
    project_root: str | Path,
    *,
    training_profile_id: str,
    dataset: str | None,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
) -> dict[str, Any]:
    from core.strategy_param_policy import get_strategy_param_training_profile

    profile = get_strategy_param_training_profile(training_profile_id)
    requested_dataset = str(profile.dataset if dataset in (None, "") else dataset)
    if requested_dataset != str(profile.dataset):
        raise ValueError(
            "registered fixed multi-seed identity不得覆寫dataset: "
            f"profile={profile.dataset!r}, requested={requested_dataset!r}"
        )
    return build_registered_strategy_param_scientific_identity(
        project_root,
        training_profile_id,
        execution_overrides=_fixed_multiseed_execution_overrides(
            max_positions=max_positions,
            rotation=rotation,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
        ),
    )


def fixed_multiseed_strategy_param_artifact_path(
    project_root: str | Path,
    *,
    training_profile_id: str = "production_multiseed_current",
    policy: str = "base-finalists-agree",
) -> Path:
    from filters.breakout_quality.strategy_compare_sources import PARAM_POLICY_SPECS

    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    spec = PARAM_POLICY_SPECS.get(compare_policy)
    if spec is None:
        raise ValueError(f"fixed multi-seed不支援param policy: {compare_policy!r}")
    return root / _fixed_multiseed_paramset_relative(training_profile_id) / str(spec["filename"])


def fixed_multiseed_strategy_param_manifest_path(
    project_root: str | Path,
    *,
    training_profile_id: str = "production_multiseed_current",
) -> Path:
    return (
        Path(project_root).resolve()
        / _fixed_multiseed_paramset_relative(training_profile_id)
        / "identity_manifest.json"
    )


def _fixed_multiseed_build_contract(
    *,
    training_profile_id: str,
    dataset: str | None,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
) -> dict[str, Any]:
    from core.strategy_param_policy import (
        get_strategy_param_fitting_seed_protocol,
        get_strategy_param_training_profile,
    )

    profile = get_strategy_param_training_profile(training_profile_id)
    seed_protocol = get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)
    if profile.filter_profile not in {"full", "min"}:
        raise ValueError("registered strategy parameter不支援filter profile")
    if profile.price_basis != "adjusted" or profile.validation_enabled:
        raise ValueError("registered strategy parameter必須adjusted且validation關閉")
    if seed_protocol.enabled:
        if profile.filter_profile != "full" or profile.parameter_policy != "base-finalists-agree":
            raise ValueError("fixed multi-seed僅支援Full/base-finalists-agree")
    elif seed_protocol.seed_count != 1 or profile.parameter_policy != "base-finalist-best":
        raise ValueError("fixed single-seed僅支援Min/Full base-finalist-best")
    base = _canonical_schedule_build_contract(
        family=profile.filter_profile,
        dataset=(profile.dataset if dataset in (None, "") else dataset),
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    seed_policy = seed_protocol.as_seed_ensemble_policy()
    return {
        **base,
        "optimizer_seed": int(seed_protocol.generator_seed),
        "trials_per_fold": int(seed_protocol.trials_per_seed),
        "parameter_set": str(profile.parameter_set_identity),
        "optimizer_search_protocol": (
            _FIXED_MULTI_SEED_SEARCH_PROTOCOL if seed_protocol.enabled else "fixed_single_seed_v1"
        ),
        "training_profile_id": str(profile.profile_id),
        "fitting_seed_protocol": str(seed_protocol.protocol_id),
        "seed_ensemble_policy": seed_policy,
    }


def _strategy_param_timing_seed_policy(
    source_policy: dict[str, Any], *, benchmark_seed_count: int
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Return a deterministic timing-only prefix of the production seed bundle."""

    source = dict(source_policy or {})
    source_seeds = tuple(int(value) for value in resolve_seed_ensemble_seeds(source))
    requested = int(benchmark_seed_count)
    if requested < 1:
        raise ValueError("Strategy Parameter timing seed count必須>=1")
    if requested > len(source_seeds):
        raise ValueError(
            "Strategy Parameter timing seed count不得超過production fitting seed bundle: "
            f"requested={requested}, production={len(source_seeds)}"
        )
    selected = tuple(source_seeds[:requested])
    min_agree_requested = source.get("min_agree_requested", "auto")
    policy = {
        "enabled": bool(requested > 1),
        "seed_count": int(requested),
        "min_agree": int(resolve_seed_ensemble_min_agree(requested, min_agree_requested)),
        "min_agree_requested": min_agree_requested,
        "min_agree_max": int(requested),
        "min_agree_rule": "clamp_1_to_seed_count",
        "seed_mode": "fixed_generated_bundle" if requested > 1 else "fixed_single_seed",
        "params_storage": str(source.get("params_storage") or "same_json"),
        "generator_seed": int(source.get("generator_seed", selected[0]) or selected[0]),
        "resolved_seeds": [int(value) for value in selected],
    }
    return policy, selected


def _strategy_param_timing_last_fold_start(
    *, schedule_start: str, coverage_end: str, oos_horizon_months: int
) -> str:
    first = datetime.strptime(str(schedule_start), "%Y-%m-%d").date()
    coverage = datetime.strptime(str(coverage_end), "%Y-%m-%d").date()
    step_months = int(oos_horizon_months)
    if step_months < 1:
        raise ValueError("Strategy Parameter timing OOS horizon必須>=1")
    month_delta = (coverage.year - first.year) * 12 + (coverage.month - first.month)
    if month_delta < 0:
        raise ValueError("Strategy Parameter timing coverage早於schedule start")
    step_count = month_delta // step_months
    month_index = first.year * 12 + (first.month - 1) + step_count * step_months
    year, month_zero = divmod(month_index, 12)
    return f"{int(year):04d}-{int(month_zero + 1):02d}-01"


def run_fixed_multiseed_strategy_parameter_timing_benchmark(
    project_root: str | Path,
    *,
    training_profile_id: str,
    benchmark_trials_per_seed: int,
    benchmark_seed_count: int = 1,
    dataset: str | None = None,
    fold_workers: int | None = None,
    requested_first_oos_date: str | None = None,
    requested_last_oos_date: str | None = None,
    requested_window_mode: str | None = None,
    requested_train_window_months: int | None = None,
    requested_oos_months: int | None = None,
    requested_first_oos_year: int | None = None,
    requested_last_oos_year: int | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
) -> dict[str, Any]:
    """Run a timing overlay through the registered Research parameter producer.

    The registered profile remains the owner of sampler, seed bundle, selector,
    rolling schedule, SQLite storage and execution policy.  Timing may only reduce
    the per-seed trial budget and add telemetry; the benchmark is isolated under
    ``outputs/`` and never publishes or reuses canonical strategy-parameter truth.
    """

    from core.strategy_param_policy import get_strategy_param_training_profile
    from services.optimizer.strategy_param_training import (
        prepare_selection_historical_full_roos_params,
    )

    root = Path(project_root).resolve()
    profile = get_strategy_param_training_profile(training_profile_id)
    contract = _fixed_multiseed_build_contract(
        training_profile_id=profile.profile_id,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    scientific_identity = _fixed_multiseed_scientific_identity(
        root,
        training_profile_id=profile.profile_id,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    if not bool(scientific_identity.get("registered", False)):
        raise RuntimeError(
            f"Strategy Parameter timing source未註冊scientific identity: {profile.profile_id}"
        )
    if not bool(scientific_identity.get("registration_matches", False)):
        raise RuntimeError(
            f"Strategy Parameter timing source scientific contract drift: {profile.profile_id}"
        )

    trials = int(benchmark_trials_per_seed)
    source_trials = int(contract["trials_per_fold"])
    source_seed_policy = dict(contract["seed_ensemble_policy"])
    benchmark_seed_policy, benchmark_seeds = _strategy_param_timing_seed_policy(
        source_seed_policy, benchmark_seed_count=int(benchmark_seed_count)
    )
    source_seed_count = int(
        source_seed_policy.get("seed_count", len(benchmark_seeds)) or len(benchmark_seeds)
    )
    if trials < 1:
        raise ValueError("Strategy Parameter timing trials/seed必須>=1")
    if trials > source_trials:
        raise ValueError(
            "Strategy Parameter timing只允許縮小registered fitting budget: "
            f"requested={trials}, source={source_trials}"
        )
    if fold_workers is not None and int(fold_workers) < 1:
        raise ValueError("Strategy Parameter timing fold workers必須>=1")

    expected_first = str(profile.schedule_start_date)
    expected_coverage_end = str(profile.coverage_end_date)
    expected_train_months = int(contract["train_window_months"])
    expected_oos_months = int(contract["oos_horizon_months"])
    expected_last_fold_start = _strategy_param_timing_last_fold_start(
        schedule_start=expected_first,
        coverage_end=expected_coverage_end,
        oos_horizon_months=expected_oos_months,
    )

    if requested_first_oos_date not in (None, "") and str(requested_first_oos_date) != expected_first:
        raise ValueError(
            "Timing schedule由active Strategy Parameter profile持有；"
            f"first_oos expected={expected_first}, requested={requested_first_oos_date}"
        )
    if requested_last_oos_date not in (None, "") and str(requested_last_oos_date) not in {
        expected_last_fold_start,
        expected_coverage_end,
    }:
        raise ValueError(
            "Timing schedule由active Strategy Parameter profile持有；"
            f"last_oos_fold_start expected={expected_last_fold_start}, "
            f"coverage_end={expected_coverage_end}, requested={requested_last_oos_date}"
        )
    if requested_window_mode not in (None, "") and str(requested_window_mode).strip().lower() != "fixed":
        raise ValueError("目前active Strategy Parameter producer的rolling window mode固定為fixed")
    if requested_train_window_months is not None and int(requested_train_window_months) != expected_train_months:
        raise ValueError(
            "Timing train window必須沿用active Strategy Parameter producer: "
            f"expected={expected_train_months}, requested={requested_train_window_months}"
        )
    if requested_oos_months is not None and int(requested_oos_months) != expected_oos_months:
        raise ValueError(
            "Timing OOS horizon必須沿用active Strategy Parameter producer: "
            f"expected={expected_oos_months}, requested={requested_oos_months}"
        )
    if requested_first_oos_year is not None and int(requested_first_oos_year) != int(expected_first[:4]):
        raise ValueError(
            "Timing first OOS year必須沿用active Strategy Parameter producer: "
            f"expected={expected_first[:4]}, requested={requested_first_oos_year}"
        )
    if requested_last_oos_year is not None and int(requested_last_oos_year) != int(expected_last_fold_start[:4]):
        raise ValueError(
            "Timing last OOS year必須沿用active Strategy Parameter producer: "
            f"expected={expected_last_fold_start[:4]}, requested={requested_last_oos_year}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    benchmark_root_relative = (
        _STRATEGY_PARAM_TIMING_ROOT_RELATIVE
        / _fixed_multiseed_artifact_slug(profile.profile_id)
        / stamp
    )
    benchmark_root = root / benchmark_root_relative
    benchmark_root.mkdir(parents=True, exist_ok=False)

    timing_environ = dict(os.environ)
    if fold_workers is not None:
        timing_environ["OPTIMIZER_ROLLING_FOLD_WORKERS"] = str(int(fold_workers))

    print(
        "Strategy Parameter timing overlay | "
        f"profile={profile.profile_id} | parameter_set={contract['parameter_set']} | "
        f"seeds={len(benchmark_seeds)} timing/{source_seed_count} production | "
        f"trials/seed={trials} (production={source_trials}) | "
        "storage=sqlite | resume=on | publish=off"
    )
    result = prepare_selection_historical_full_roos_params(
        project_root=root,
        dataset=str(contract["dataset"]),
        param_policy=str(profile.parameter_policy),
        trials_per_fold=trials,
        max_positions=int(contract["max_positions"]),
        rotation=str(contract["rotation"]),
        fixed_risk=float(contract["fixed_risk"]),
        max_position_cap_pct=float(contract["max_position_cap_pct"]),
        optimizer_seed=(
            int(benchmark_seeds[0])
            if len(benchmark_seeds) == 1
            else int(contract["optimizer_seed"])
        ),
        resume_parameter_training=True,
        quiet=False,
        first_oos_date=expected_first,
        last_oos_date=expected_coverage_end,
        train_window_months=expected_train_months,
        oos_months=expected_oos_months,
        output_relative_dir=benchmark_root_relative,
        build_context_label=f"{contract['parameter_set']} Timing",
        optimizer_output_title="Strategy Parameter timing artifacts",
        environ_override=timing_environ,
        optimizer_search_protocol=str(contract["optimizer_search_protocol"]),
        parameter_set_identity=str(contract["parameter_set"]),
        seed_ensemble_policy_override=dict(benchmark_seed_policy),
        score_policy_overrides=(
            dict(profile.score_policy_overrides)
            if profile.score_policy_overrides
            else None
        ),
        timing_mode=True,
    )
    timing = dict(result.get("timing") or {})
    timing_payload = dict(timing.get("payload") or {})
    timing_summary = dict(timing_payload.get("summary") or {})
    benchmark_manifest = {
        "schema_type": "strategy_parameter_timing_benchmark",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": "optimizer_strategy_parameter",
        "training_profile_id": str(profile.profile_id),
        "parameter_set": str(contract["parameter_set"]),
        "registered_scientific_fingerprint_sha256": str(
            scientific_identity.get("registered_scientific_fingerprint_sha256") or ""
        ),
        "source_scientific_fingerprint_sha256": str(
            scientific_identity.get("scientific_fingerprint_sha256") or ""
        ),
        "benchmark_scope": {
            "publish_parameter_truth": False,
            "source_trials_per_seed": source_trials,
            "benchmark_trials_per_seed": trials,
            "source_seed_count": int(source_seed_count),
            "benchmark_seed_count": int(len(benchmark_seeds)),
            "source_seed_ensemble_policy": dict(source_seed_policy),
            "seed_ensemble_policy": dict(benchmark_seed_policy),
            "benchmark_resolved_seeds": [int(value) for value in benchmark_seeds],
            "schedule_start": expected_first,
            "coverage_end": expected_coverage_end,
            "last_fold_start": expected_last_fold_start,
            "train_window_months": expected_train_months,
            "oos_horizon_months": expected_oos_months,
            "study_storage": str(timing_summary.get("study_storage") or "sqlite"),
            "sampler_kind": str(timing_payload.get("sampler_kind") or ""),
            "fold_workers": int(timing_summary.get("rolling_fold_workers_max", 0) or 0),
        },
        "timing_outputs": {
            "json": _project_relative(root, Path(timing["json_path"])) if timing.get("json_path") else None,
            "csv": _project_relative(root, Path(timing["csv_path"])) if timing.get("csv_path") else None,
            "resource_csv": _project_relative(root, Path(timing["resource_csv_path"])) if timing.get("resource_csv_path") else None,
        },
    }
    manifest_path = benchmark_root / "timing_benchmark_manifest.json"
    atomic_write_json(manifest_path, benchmark_manifest)
    return {
        **result,
        "benchmark_root": benchmark_root,
        "benchmark_manifest_path": manifest_path,
        "benchmark_manifest": benchmark_manifest,
        "source_build_contract": contract,
        "source_scientific_identity": scientific_identity,
    }


def inspect_fixed_multiseed_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    training_profile_id: str = "production_multiseed_current",
    policy: str = "base-finalists-agree",
    comparison_end_date: str,
    dataset: str | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
) -> dict[str, Any]:
    from core.strategy_param_policy import get_strategy_param_training_profile

    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    if compare_policy != get_strategy_param_training_profile(training_profile_id).parameter_policy:
        raise ValueError("registered strategy parameter source selector與profile不一致")
    normalized_policy = normalize_strategy_param_policy(compare_policy)
    end_date = str(comparison_end_date or "").strip()
    if not end_date:
        raise ValueError("fixed multi-seed缺少comparison_end_date")
    target = fixed_multiseed_strategy_param_artifact_path(
        root, training_profile_id=training_profile_id, policy=compare_policy
    )
    manifest_path = fixed_multiseed_strategy_param_manifest_path(
        root, training_profile_id=training_profile_id
    )
    contract = _fixed_multiseed_build_contract(
        training_profile_id=training_profile_id,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    scientific_identity = _fixed_multiseed_scientific_identity(
        root,
        training_profile_id=training_profile_id,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    manifest = _load_json_object(manifest_path)
    payload = _load_json_object(target)
    coverage_end = _canonical_schedule_coverage_end(target)
    adaptation = dict((payload or {}).get("breakout_quality_param_adaptation") or {})
    expected_seed_policy = dict(contract["seed_ensemble_policy"])
    actual_seed_policy = dict(adaptation.get("seed_ensemble_policy") or {})
    manifest_scientific_identity = dict((manifest or {}).get("scientific_identity") or {})

    status = "READY"
    if not bool(scientific_identity.get("registered", False)):
        status = "SCIENTIFIC_IDENTITY_UNREGISTERED"
    elif not bool(scientific_identity.get("registration_matches", False)):
        status = "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT"
    elif payload is None:
        status = "ARTIFACT_MISSING_OR_INVALID"
    elif str(payload.get("selector") or "") != normalized_policy:
        status = "POLICY_SELECTOR_MISMATCH"
    elif str(adaptation.get("parameter_set") or "") != str(contract["parameter_set"]):
        status = "PARAMETER_SET_MISMATCH"
    elif bool(adaptation.get("training_dl_enabled", True)):
        status = "TRAINING_DL_ENABLED_MISMATCH"
    elif str(adaptation.get("optimizer_search_protocol") or "") != str(contract["optimizer_search_protocol"]):
        status = "OPTIMIZER_SEARCH_PROTOCOL_MISMATCH"
    elif actual_seed_policy != expected_seed_policy:
        status = "FITTING_SEED_POLICY_MISMATCH"
    elif coverage_end is None or str(coverage_end) < end_date:
        status = "SCHEDULE_COVERAGE_INCOMPLETE"
    elif manifest is None:
        status = "LEGACY_SCIENTIFIC_IDENTITY_UNBOUND"
    elif not manifest_scientific_identity:
        status = "LEGACY_SCIENTIFIC_IDENTITY_UNBOUND"
    elif manifest_issue := first_manifest_identity_mismatch(
        manifest,
        (
            ("producer", "string", "optimizer", "PRODUCER_MISMATCH"),
            ("parameter_set", "string", contract["parameter_set"], "MANIFEST_PARAMETER_SET_MISMATCH"),
            ("policy", "string", compare_policy, "MANIFEST_POLICY_MISMATCH"),
            ("optimizer_search_protocol", "string", contract["optimizer_search_protocol"], "MANIFEST_PROTOCOL_MISMATCH"),
            ("seed_ensemble_policy", "mapping", expected_seed_policy, "MANIFEST_FITTING_SEED_POLICY_MISMATCH"),
            ("build_contract", "mapping", contract, "BUILD_CONTRACT_MISMATCH"),
        ),
    ):
        status = manifest_issue
    elif int(manifest_scientific_identity.get("schema_version", 0) or 0) != int(
        scientific_identity["schema_version"]
    ):
        status = "SCIENTIFIC_IDENTITY_SCHEMA_MISMATCH"
    elif str(
        manifest_scientific_identity.get("registered_scientific_fingerprint_sha256") or ""
    ) != str(scientific_identity.get("registered_scientific_fingerprint_sha256") or ""):
        status = "SCIENTIFIC_REGISTRATION_MISMATCH"
    elif str(manifest_scientific_identity.get("scientific_fingerprint_sha256") or "") != str(
        scientific_identity.get("scientific_fingerprint_sha256") or ""
    ):
        status = "SCIENTIFIC_CONTRACT_DRIFT"
    elif dict(manifest_scientific_identity.get("scientific_contract") or {}) != dict(
        scientific_identity.get("scientific_contract") or {}
    ):
        status = "SCIENTIFIC_CONTRACT_DRIFT"
    elif not manifest_artifact_sha_matches(manifest, target):
        status = "ARTIFACT_SHA_MISMATCH"

    ready = status == "READY"
    existed = bool(target.exists() or manifest_path.exists())
    # A legacy P6 artifact that predates scientific-identity binding must not be
    # REUSED, but it also must not dead-end the canonical Research workflow.
    # When the registered source contract still matches current semantics, the
    # Optimizer sole producer can deterministically REBUILD (and resume any
    # compatible workspace) to publish a newly bound manifest.  True source or
    # bound-manifest scientific drift remains BLOCKED and can never be bypassed
    # by rebuild/force-rebuild under the same PARAM identity.
    blocked_statuses = {
        "SCIENTIFIC_IDENTITY_UNREGISTERED",
        "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT",
        "SCIENTIFIC_IDENTITY_SCHEMA_MISMATCH",
        "SCIENTIFIC_REGISTRATION_MISMATCH",
        "SCIENTIFIC_CONTRACT_DRIFT",
    }
    return {
        "ready": ready,
        "status": status,
        "action": resolve_artifact_reuse_action(
            status, existed=existed, blocked_statuses=blocked_statuses,
        ),
        "path": target,
        "manifest_path": manifest_path,
        "coverage_end": coverage_end,
        "build_contract": contract,
        "scientific_identity": scientific_identity,
    }


def ensure_fixed_multiseed_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    training_profile_id: str = "production_multiseed_current",
    policy: str = "base-finalists-agree",
    comparison_end_date: str,
    dataset: str | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
    force_rebuild: bool = False,
    resume_parameter_training: bool = True,
) -> dict[str, Any]:
    from services.optimizer.strategy_param_training import (
        prepare_selection_historical_full_roos_params,
        prepare_selection_historical_p2_params,
    )

    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    state = inspect_fixed_multiseed_strategy_parameter_artifact(
        root,
        training_profile_id=training_profile_id,
        policy=compare_policy,
        comparison_end_date=comparison_end_date,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    target = Path(state["path"])
    manifest_path = Path(state["manifest_path"])
    contract = dict(state["build_contract"])
    scientific_identity = dict(state["scientific_identity"])
    if str(state.get("action") or "") == "BLOCKED":
        status = str(state.get("status") or "")
        raise RuntimeError(
            f"{contract['parameter_set']} scientific identity blocked: "
            f"status={status}. Existing registered identity must not be rebuilt "
            "under changed scientific semantics."
        )
    existed_before = str(state["action"]) == "REBUILD"
    if bool(state["ready"]) and not force_rebuild:
        return {
            "action": "REUSE",
            "path": target,
            "manifest_path": manifest_path,
            "sha256": compute_strategy_param_file_sha256(target),
            "coverage_end": state.get("coverage_end"),
            "build_contract": contract,
            "scientific_identity": scientific_identity,
        }

    canonical_work_root = root / _fixed_multiseed_work_relative(training_profile_id)
    work_root = _force_rebuild_attempt_root(canonical_work_root) if force_rebuild else canonical_work_root
    end_date = str(comparison_end_date or "").strip()
    common_train = dict(
        project_root=root,
        dataset=str(contract["dataset"]),
        param_policy=compare_policy,
        trials_per_fold=int(contract["trials_per_fold"]),
        max_positions=int(contract["max_positions"]),
        rotation=str(contract["rotation"]),
        fixed_risk=float(contract["fixed_risk"]),
        max_position_cap_pct=float(contract["max_position_cap_pct"]),
        optimizer_seed=int(contract["optimizer_seed"]),
        resume_parameter_training=(False if force_rebuild else bool(resume_parameter_training)),
        quiet=False,
        first_oos_date=str(contract["schedule_start"]),
        last_oos_date=end_date,
        train_window_months=int(contract["train_window_months"]),
        oos_months=int(contract["oos_horizon_months"]),
        output_relative_dir=work_root,
        build_context_label=f"{contract['parameter_set']} Registered Optimizer",
        optimizer_output_title=f"Optimizer {contract['parameter_set']} 工作工件（非production）",
    )
    if str(contract["family"]) == "min":
        result = prepare_selection_historical_p2_params(
            **common_train,
            recover_completed_strategy_compare=False,
            parameter_set_identity=str(contract["parameter_set"]),
            seed_ensemble_policy_override=dict(contract["seed_ensemble_policy"]),
        )
    else:
        result = prepare_selection_historical_full_roos_params(
            **common_train,
            optimizer_search_protocol=(
                str(contract["optimizer_search_protocol"])
                if bool(dict(contract["seed_ensemble_policy"]).get("enabled"))
                else None
            ),
            parameter_set_identity=str(contract["parameter_set"]),
            seed_ensemble_policy_override=dict(contract["seed_ensemble_policy"]),
        )
    source = Path(result["params_path"]).resolve()
    _assert_optimizer_work_source(source_path=source, work_root=work_root)
    if not source.is_file():
        raise FileNotFoundError(
            f"{contract['parameter_set']} Optimizer完成但找不到策略參數工件: "
            + _project_relative(root, source)
        )
    _copy_json_payload(source, target)
    if not bool(dict(contract["seed_ensemble_policy"]).get("enabled")):
        single_payload = _load_json_object(target)
        if single_payload is None:
            raise RuntimeError("registered single-seed optimizer output JSON invalid")
        adaptation = dict(single_payload.get("breakout_quality_param_adaptation") or {})
        adaptation.update({
            "parameter_set": str(contract["parameter_set"]),
            "training_dl_enabled": False,
            "optimizer_search_protocol": str(contract["optimizer_search_protocol"]),
            "seed_ensemble_policy": dict(contract["seed_ensemble_policy"]),
        })
        single_payload["breakout_quality_param_adaptation"] = adaptation
        _write_json(target, single_payload)
    coverage_end = _canonical_schedule_coverage_end(target)
    if coverage_end is None or str(coverage_end) < end_date:
        raise RuntimeError(
            f"{contract['parameter_set']} schedule coverage不足: "
            f"required={end_date}, actual={coverage_end or '-'}"
        )
    payload = _load_json_object(target) or {}
    adaptation = dict(payload.get("breakout_quality_param_adaptation") or {})
    if str(adaptation.get("parameter_set") or "") != str(contract["parameter_set"]):
        raise RuntimeError(f"{contract['parameter_set']} artifact缺少parameter_set identity")
    if dict(adaptation.get("seed_ensemble_policy") or {}) != dict(contract["seed_ensemble_policy"]):
        raise RuntimeError(f"{contract['parameter_set']} artifact缺少fixed fitting-seed identity")

    target_sha = compute_strategy_param_file_sha256(target)
    _write_json(
        manifest_path,
        {
            "schema_type": "strategy_parameter_research_identity",
            "schema_version": 1,
            "producer": "optimizer",
            "training_dl_enabled": False,
            "parameter_set": str(contract["parameter_set"]),
            "family": str(contract["family"]),
            "policy": compare_policy,
            "optimizer_search_protocol": str(contract["optimizer_search_protocol"]),
            "training_profile_id": str(contract["training_profile_id"]),
            "fitting_seed_protocol": str(contract["fitting_seed_protocol"]),
            "seed_ensemble_policy": dict(contract["seed_ensemble_policy"]),
            "build_contract": contract,
            "scientific_identity": {
                "schema_version": int(scientific_identity["schema_version"]),
                "registered_scientific_fingerprint_sha256": str(
                    scientific_identity["registered_scientific_fingerprint_sha256"]
                ),
                "scientific_fingerprint_sha256": str(
                    scientific_identity["scientific_fingerprint_sha256"]
                ),
                "scientific_contract": dict(scientific_identity["scientific_contract"]),
            },
            "artifact_path": _project_relative(root, target),
            "artifact_sha256": target_sha,
            "coverage_end": coverage_end,
            "production_promotion": False,
        },
    )
    return {
        "action": "FORCE_REBUILD" if force_rebuild else ("REBUILD" if existed_before else "BUILD"),
        "path": target,
        "manifest_path": manifest_path,
        "sha256": target_sha,
        "coverage_end": coverage_end,
        "build_contract": contract,
        "scientific_identity": scientific_identity,
    }



def fixed_multiseed_robustness_strategy_param_artifact_path(
    project_root: str | Path,
    *,
    training_profile_id: str = "production_multiseed_current",
    benchmark_id: str,
    replication_seed: int,
    policy: str = "base-finalists-agree",
) -> Path:
    root = Path(project_root).resolve()
    benchmark = normalize_strategy_param_benchmark_id(str(benchmark_id))
    seed = normalize_strategy_param_benchmark_seed(int(replication_seed))
    filename = fixed_multiseed_strategy_param_artifact_path(
        root, training_profile_id=training_profile_id, policy=policy
    ).name
    return (
        root
        / _fixed_multiseed_paramset_relative(training_profile_id)
        / "benchmark"
        / benchmark
        / f"replicate_{seed}"
        / filename
    )


def fixed_multiseed_robustness_strategy_param_manifest_path(
    project_root: str | Path,
    *,
    training_profile_id: str = "production_multiseed_current",
    benchmark_id: str,
    replication_seed: int,
) -> Path:
    root = Path(project_root).resolve()
    benchmark = normalize_strategy_param_benchmark_id(str(benchmark_id))
    seed = normalize_strategy_param_benchmark_seed(int(replication_seed))
    return (
        root
        / _fixed_multiseed_paramset_relative(training_profile_id)
        / "benchmark"
        / benchmark
        / f"replicate_{seed}"
        / "identity_manifest.json"
    )


def _fixed_multiseed_robustness_build_contract(
    *,
    training_profile_id: str,
    benchmark_id: str,
    replication_seed: int,
    dataset: str | None,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
) -> dict[str, Any]:
    from core.strategy_param_policy import (
        build_replicated_fitting_seed_ensemble_policy,
        get_strategy_param_fitting_seed_protocol,
        get_strategy_param_training_profile,
    )

    benchmark = dict(get_robustness_benchmark_policy_snapshot())
    benchmark_name = normalize_strategy_param_benchmark_id(str(benchmark_id))
    replicate = normalize_strategy_param_benchmark_seed(int(replication_seed))
    if benchmark_name != str(benchmark["benchmark_id"]):
        raise ValueError(f"benchmark_id與current SSOT不一致: {benchmark_id}")
    if replicate not in {int(value) for value in benchmark["resolved_seeds"]}:
        raise ValueError(f"replication seed不屬於current robustness benchmark題庫: {replicate}")
    profile = get_strategy_param_training_profile(training_profile_id)
    base = _fixed_multiseed_build_contract(
        training_profile_id=training_profile_id,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    seed_protocol = get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)
    if seed_protocol.enabled:
        replicated_policy = build_replicated_fitting_seed_ensemble_policy(
            profile.fitting_seed_protocol, replication_seed=replicate
        )
    else:
        # The outer robustness seed is the one fitting seed, not an additional
        # seed bundle.  The base profile remains frozen at Seed42.
        replicated_policy = seed_protocol.as_seed_ensemble_policy()
        replicated_policy.update(
            generator_seed=int(replicate), resolved_seeds=[int(replicate)]
        )
    return {
        **base,
        "optimizer_seed": int(replicate),
        "seed_ensemble_policy": replicated_policy,
        "benchmark_id": benchmark_name,
        "robustness_replication_seed": int(replicate),
        "robustness_seed_semantics": (
            "replication_seed_generates_fitting_seed_bundle" if seed_protocol.enabled
            else "replication_seed_is_single_fitting_seed"
        ),
    }


def _fixed_multiseed_robustness_scientific_identity(
    project_root: str | Path,
    *,
    training_profile_id: str,
    build_contract: dict[str, Any],
    dataset: str | None,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
) -> dict[str, Any]:
    from core.strategy_param_policy import get_strategy_param_training_profile

    profile = get_strategy_param_training_profile(training_profile_id)
    requested_dataset = str(profile.dataset if dataset in (None, "") else dataset)
    if requested_dataset != str(profile.dataset):
        raise ValueError(
            "registered fixed multi-seed robustness identity不得覆寫dataset: "
            f"profile={profile.profile_id}, registered={profile.dataset!r}, requested={requested_dataset!r}"
        )
    fitting = dict(build_contract["seed_ensemble_policy"])
    fitting["trials_per_seed"] = int(build_contract["trials_per_fold"])
    return derive_strategy_param_scientific_identity_from_registered_profile(
        project_root,
        training_profile_id,
        fitting_override=fitting,
        execution_overrides=_fixed_multiseed_execution_overrides(
            max_positions=max_positions,
            rotation=rotation,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
        ),
        reference_overrides={
            "robustness_replication_seed": int(
                build_contract["robustness_replication_seed"]
            ),
            "robustness_seed_semantics": str(
                build_contract["robustness_seed_semantics"]
            ),
        },
    )


def _fixed_multiseed_robustness_repair_source(
    *, root: Path, training_profile_id: str, benchmark_id: str, replication_seed: int,
    policy: str, target: Path, manifest: dict[str, Any] | None,
) -> Path | None:
    """Return proof of publication-only drift for an already-trained replicate.

    Bound manifest scientific SHA is sufficient when available.  Older manifests
    can still be repaired without retraining when the retained Optimizer workspace
    resolves to the exact same runtime-scientific parameter payload.
    """

    if not target.is_file() or manifest is None:
        return None
    target_scientific_sha = compute_strategy_param_scientific_sha256(target)
    manifest_scientific_sha = str(
        manifest.get("artifact_scientific_sha256") or ""
    ).strip().lower()
    if manifest_scientific_sha and manifest_scientific_sha == target_scientific_sha:
        return target.resolve()

    work_root = (
        root
        / _fixed_multiseed_work_relative(training_profile_id)
        / "benchmark"
        / normalize_strategy_param_benchmark_id(str(benchmark_id))
        / f"replicate_{normalize_strategy_param_benchmark_seed(int(replication_seed))}"
    )
    filename = fixed_multiseed_strategy_param_artifact_path(
        root, training_profile_id=training_profile_id, policy=policy
    ).name
    source = Path(work_root) / "active_params" / filename
    if not source.is_file():
        return None
    if compute_strategy_param_scientific_sha256(source) != target_scientific_sha:
        return None
    return source.resolve()


def inspect_fixed_multiseed_robustness_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    training_profile_id: str = "production_multiseed_current",
    benchmark_id: str,
    replication_seed: int,
    policy: str = "base-finalists-agree",
    comparison_end_date: str,
    dataset: str | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    end_date = str(comparison_end_date or "").strip()
    if not end_date:
        raise ValueError("fixed multi-seed robustness缺少comparison_end_date")
    target = fixed_multiseed_robustness_strategy_param_artifact_path(
        root,
        training_profile_id=training_profile_id,
        benchmark_id=benchmark_id,
        replication_seed=replication_seed,
        policy=compare_policy,
    )
    manifest_path = fixed_multiseed_robustness_strategy_param_manifest_path(
        root,
        training_profile_id=training_profile_id,
        benchmark_id=benchmark_id,
        replication_seed=replication_seed,
    )
    contract = _fixed_multiseed_robustness_build_contract(
        training_profile_id=training_profile_id,
        benchmark_id=benchmark_id,
        replication_seed=replication_seed,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    scientific_identity = _fixed_multiseed_robustness_scientific_identity(
        root,
        training_profile_id=training_profile_id,
        build_contract=contract,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    payload = _load_json_object(target)
    manifest = _load_json_object(manifest_path)
    coverage_end = _canonical_schedule_coverage_end(target)
    adaptation = dict((payload or {}).get("breakout_quality_param_adaptation") or {})
    expected_seed_policy = dict(contract["seed_ensemble_policy"])
    actual_seed_policy = dict(adaptation.get("seed_ensemble_policy") or {})
    manifest_scientific_identity = dict((manifest or {}).get("scientific_identity") or {})
    status = "READY"
    if not bool(scientific_identity.get("source_registered", False)):
        status = "SCIENTIFIC_IDENTITY_UNREGISTERED"
    elif not bool(scientific_identity.get("source_registration_matches", False)):
        status = "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT"
    elif payload is None:
        status = "ARTIFACT_MISSING_OR_INVALID"
    elif str(payload.get("selector") or "") != normalize_strategy_param_policy(compare_policy):
        status = "POLICY_SELECTOR_MISMATCH"
    elif str(adaptation.get("parameter_set") or "") != str(contract["parameter_set"]):
        status = "PARAMETER_SET_MISMATCH"
    elif str(adaptation.get("optimizer_search_protocol") or "") != str(contract["optimizer_search_protocol"]):
        status = "OPTIMIZER_SEARCH_PROTOCOL_MISMATCH"
    elif actual_seed_policy != expected_seed_policy:
        status = "FITTING_SEED_POLICY_MISMATCH"
    elif coverage_end is None or str(coverage_end) < end_date:
        status = "SCHEDULE_COVERAGE_INCOMPLETE"
    elif manifest is None:
        status = "MANIFEST_MISSING_OR_INVALID"
    elif manifest_issue := first_manifest_identity_mismatch(
        manifest,
        (
            ("producer", "string", "optimizer", "PRODUCER_MISMATCH"),
            ("benchmark_id", "string", contract["benchmark_id"], "BENCHMARK_ID_MISMATCH"),
            ("robustness_replication_seed", "integer", contract["robustness_replication_seed"], "REPLICATION_SEED_MISMATCH"),
            ("parameter_set", "string", contract["parameter_set"], "MANIFEST_PARAMETER_SET_MISMATCH"),
            ("policy", "string", compare_policy, "MANIFEST_POLICY_MISMATCH"),
            ("seed_ensemble_policy", "mapping", expected_seed_policy, "MANIFEST_FITTING_SEED_POLICY_MISMATCH"),
            ("build_contract", "mapping_subset", contract, "BUILD_CONTRACT_MISMATCH"),
        ),
    ):
        status = manifest_issue
    elif not manifest_scientific_identity:
        status = "LEGACY_SCIENTIFIC_IDENTITY_UNBOUND"
    elif manifest_scientific_identity != _derived_scientific_identity_record(
        scientific_identity
    ):
        status = "SCIENTIFIC_IDENTITY_MISMATCH"
    elif not manifest_artifact_sha_matches(manifest, target):
        status = "ARTIFACT_SHA_MISMATCH"
    ready = status == "READY"
    existed = bool(target.exists() or manifest_path.exists())
    blocked_statuses = {
        "SCIENTIFIC_IDENTITY_UNREGISTERED",
        "REGISTERED_SCIENTIFIC_CONTRACT_DRIFT",
    }
    repair_source = None
    if status == "ARTIFACT_SHA_MISMATCH":
        repair_source = _fixed_multiseed_robustness_repair_source(
            root=root,
            training_profile_id=training_profile_id,
            benchmark_id=str(contract["benchmark_id"]),
            replication_seed=int(contract["robustness_replication_seed"]),
            policy=compare_policy,
            target=target,
            manifest=manifest,
        )
    action = resolve_artifact_reuse_action(
        status, existed=existed, blocked_statuses=blocked_statuses,
        repair_available=repair_source is not None,
    )
    return {
        "ready": ready,
        "status": ("PUBLICATION_METADATA_STALE" if action == "REPAIR" else status),
        "action": action,
        "repair_source_path": repair_source,
        "path": target,
        "manifest_path": manifest_path,
        "coverage_end": coverage_end,
        "build_contract": contract,
        "scientific_identity": scientific_identity,
    }


def ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    training_profile_id: str = "production_multiseed_current",
    benchmark_id: str,
    replication_seed: int,
    policy: str = "base-finalists-agree",
    comparison_end_date: str,
    dataset: str | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
    force_rebuild: bool = False,
    resume_parameter_training: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    from config.execution_policy import (
        DEFAULT_FIXED_RISK,
        DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS,
        DEFAULT_PORTFOLIO_ROTATION,
    )
    from core.dataset_profiles import DEFAULT_DATASET_PROFILE
    from services.optimizer.strategy_param_training import (
        prepare_selection_historical_full_roos_params,
        prepare_selection_historical_p2_params,
    )

    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    state = inspect_fixed_multiseed_robustness_strategy_parameter_artifact(
        root,
        training_profile_id=training_profile_id,
        benchmark_id=benchmark_id,
        replication_seed=replication_seed,
        policy=compare_policy,
        comparison_end_date=comparison_end_date,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    target = Path(state["path"])
    manifest_path = Path(state["manifest_path"])
    contract = dict(state["build_contract"])
    scientific_identity = dict(state["scientific_identity"])
    if str(state.get("action") or "") == "BLOCKED":
        raise RuntimeError(
            f"{contract['parameter_set']} robustness scientific identity blocked: "
            f"status={state.get('status')}"
        )
    if bool(state["ready"]) and not force_rebuild:
        return {
            "action": "REUSE",
            "path": target,
            "manifest_path": manifest_path,
            "sha256": compute_strategy_param_file_sha256(target),
            "scientific_sha256": compute_strategy_param_scientific_sha256(target),
            "coverage_end": state.get("coverage_end"),
            "build_contract": contract,
            "scientific_identity": scientific_identity,
        }
    replicate = int(contract["robustness_replication_seed"])
    benchmark = str(contract["benchmark_id"])
    if str(state.get("action") or "") == "REPAIR" and not force_rebuild:
        repair_source = Path(state["repair_source_path"]).resolve()
        if repair_source != target.resolve():
            _assert_optimizer_work_source(
                source_path=repair_source,
                work_root=(
                    root
                    / _fixed_multiseed_work_relative(training_profile_id)
                    / "benchmark"
                    / benchmark
                    / f"replicate_{replicate}"
                ),
            )
            if compute_strategy_param_scientific_sha256(repair_source) != (
                compute_strategy_param_scientific_sha256(target)
            ):
                raise RuntimeError(
                    f"{contract['parameter_set']} robustness publication repair scientific SHA drift"
                )
        coverage_end = _canonical_schedule_coverage_end(target)
        target_sha = compute_strategy_param_file_sha256(target)
        target_scientific_sha = compute_strategy_param_scientific_sha256(target)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            manifest_path,
            {
                "schema_type": "strategy_parameter_research_robustness_identity",
                "schema_version": 1,
                "producer": "optimizer",
                "training_dl_enabled": False,
                "parameter_set": str(contract["parameter_set"]),
                "family": str(contract["family"]),
                "policy": compare_policy,
                "optimizer_search_protocol": str(contract["optimizer_search_protocol"]),
                "training_profile_id": str(contract["training_profile_id"]),
                "fitting_seed_protocol": str(contract["fitting_seed_protocol"]),
                "benchmark_id": benchmark,
                "robustness_replication_seed": replicate,
                "seed_ensemble_policy": dict(contract["seed_ensemble_policy"]),
                "build_contract": contract,
                "scientific_identity": _derived_scientific_identity_record(
                    scientific_identity
                ),
                "artifact_path": _project_relative(root, target),
                "artifact_sha256": target_sha,
                "artifact_scientific_sha256": target_scientific_sha,
                "coverage_end": coverage_end,
                "production_promotion": False,
            },
        )
        return {
            "action": "REPAIR",
            "path": target,
            "manifest_path": manifest_path,
            "sha256": target_sha,
            "scientific_sha256": target_scientific_sha,
            "coverage_end": coverage_end,
            "build_contract": contract,
            "scientific_identity": scientific_identity,
        }
    canonical_work_root = (
        root
        / _fixed_multiseed_work_relative(training_profile_id)
        / "benchmark"
        / benchmark
        / f"replicate_{replicate}"
    )
    work_root = _force_rebuild_attempt_root(canonical_work_root) if force_rebuild else canonical_work_root
    common_train = dict(
        project_root=root,
        dataset=str(DEFAULT_DATASET_PROFILE if dataset in (None, "") else dataset),
        param_policy=compare_policy,
        trials_per_fold=int(contract["trials_per_fold"]),
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS if max_positions is None else max_positions),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION if rotation in (None, "") else rotation),
        fixed_risk=float(DEFAULT_FIXED_RISK if fixed_risk is None else fixed_risk),
        max_position_cap_pct=float(
            DEFAULT_MAX_POSITION_CAP_PCT if max_position_cap_pct is None else max_position_cap_pct
        ),
        optimizer_seed=int(contract["optimizer_seed"]),
        resume_parameter_training=(False if force_rebuild else bool(resume_parameter_training)),
        quiet=bool(quiet),
        first_oos_date=str(contract["schedule_start"]),
        last_oos_date=str(comparison_end_date),
        train_window_months=int(contract["train_window_months"]),
        oos_months=int(contract["oos_horizon_months"]),
        output_relative_dir=work_root,
        build_context_label=(
            f"{contract['parameter_set']} Robustness {benchmark} replicate={replicate}"
        ),
        optimizer_output_title=(
            f"Optimizer {contract['parameter_set']} robustness工作工件（非production）"
        ),
    )
    if str(contract["family"]) == "min":
        result = prepare_selection_historical_p2_params(
            **common_train,
            recover_completed_strategy_compare=False,
            parameter_set_identity=str(contract["parameter_set"]),
            seed_ensemble_policy_override=dict(contract["seed_ensemble_policy"]),
        )
    else:
        result = prepare_selection_historical_full_roos_params(
            **common_train,
            optimizer_search_protocol=(
                str(contract["optimizer_search_protocol"])
                if bool(dict(contract["seed_ensemble_policy"]).get("enabled"))
                else None
            ),
            parameter_set_identity=str(contract["parameter_set"]),
            seed_ensemble_policy_override=dict(contract["seed_ensemble_policy"]),
        )
    source = Path(result["params_path"]).resolve()
    _assert_optimizer_work_source(source_path=source, work_root=work_root)
    if not source.is_file():
        raise FileNotFoundError(
            f"{contract['parameter_set']} robustness Optimizer完成但找不到策略參數工件: "
            + _project_relative(root, source)
        )
    _copy_json_payload(source, target)
    if not bool(dict(contract["seed_ensemble_policy"]).get("enabled")):
        payload = _load_json_object(target)
        if payload is None:
            raise RuntimeError("registered single-seed robustness output JSON invalid")
        adaptation = dict(payload.get("breakout_quality_param_adaptation") or {})
        adaptation.update({
            "parameter_set": str(contract["parameter_set"]),
            "training_dl_enabled": False,
            "optimizer_search_protocol": str(contract["optimizer_search_protocol"]),
            "seed_ensemble_policy": dict(contract["seed_ensemble_policy"]),
        })
        payload["breakout_quality_param_adaptation"] = adaptation
        _write_json(target, payload)
    coverage_end = _canonical_schedule_coverage_end(target)
    if coverage_end is None or str(coverage_end) < str(comparison_end_date):
        raise RuntimeError(
            f"{contract['parameter_set']} robustness schedule coverage不足: "
            f"required={comparison_end_date}, actual={coverage_end or '-'}"
        )
    payload = _load_json_object(target) or {}
    adaptation = dict(payload.get("breakout_quality_param_adaptation") or {})
    if str(adaptation.get("parameter_set") or "") != str(contract["parameter_set"]):
        raise RuntimeError(
            f"{contract['parameter_set']} robustness artifact缺少parameter_set identity"
        )
    if dict(adaptation.get("seed_ensemble_policy") or {}) != dict(contract["seed_ensemble_policy"]):
        raise RuntimeError(
            f"{contract['parameter_set']} robustness artifact缺少replicated fitting-seed identity"
        )
    target_sha = compute_strategy_param_file_sha256(target)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        manifest_path,
        {
            "schema_type": "strategy_parameter_research_robustness_identity",
            "schema_version": 1,
            "producer": "optimizer",
            "training_dl_enabled": False,
            "parameter_set": str(contract["parameter_set"]),
            "family": str(contract["family"]),
            "policy": compare_policy,
            "optimizer_search_protocol": str(contract["optimizer_search_protocol"]),
            "training_profile_id": str(contract["training_profile_id"]),
            "fitting_seed_protocol": str(contract["fitting_seed_protocol"]),
            "benchmark_id": benchmark,
            "robustness_replication_seed": replicate,
            "seed_ensemble_policy": dict(contract["seed_ensemble_policy"]),
            "build_contract": contract,
            "scientific_identity": _derived_scientific_identity_record(
                scientific_identity
            ),
            "artifact_path": _project_relative(root, target),
            "artifact_sha256": target_sha,
            "artifact_scientific_sha256": compute_strategy_param_scientific_sha256(target),
            "coverage_end": coverage_end,
            "production_promotion": False,
        },
    )
    return {
        "action": "FORCE_REBUILD" if force_rebuild else ("REBUILD" if str(state["action"]) == "REBUILD" else "BUILD"),
        "path": target,
        "manifest_path": manifest_path,
        "sha256": target_sha,
        "scientific_sha256": compute_strategy_param_scientific_sha256(target),
        "coverage_end": coverage_end,
        "build_contract": contract,
    }

def validation_budget_strategy_param_artifact_path(
    project_root: str | Path,
    *,
    policy: str = "base-finalists-agree",
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> Path:
    from filters.breakout_quality.strategy_compare_sources import PARAM_POLICY_SPECS

    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    spec = PARAM_POLICY_SPECS.get(compare_policy)
    if spec is None:
        raise ValueError(f"validation-budget不支援param policy: {compare_policy!r}")
    relative = _validation_paramset_relative(
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    return root / relative / str(spec["filename"])


def validation_budget_strategy_param_manifest_path(
    project_root: str | Path,
    *,
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> Path:
    relative = _validation_paramset_relative(
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    return Path(project_root).resolve() / relative / "identity_manifest.json"


def _resolve_validation_budget_publication_identity(
    *, parameter_set_identity: str, validation_protocol_id: str
) -> tuple[str, str, str]:
    from core.strategy_param_policy import (
        get_strategy_param_validation_protocol,
        resolve_validation_strategy_param_training_profile,
    )

    parameter_set = str(parameter_set_identity or "").strip()
    protocol = str(validation_protocol_id or "").strip()
    profile = resolve_validation_strategy_param_training_profile(parameter_set, protocol)
    if profile.filter_profile != "full":
        raise ValueError(
            "validation durable publication目前要求Full filter: "
            f"profile={profile.profile_id}"
        )
    validation_spec = get_strategy_param_validation_protocol(protocol)
    optimizer_policy = normalize_strategy_param_policy(profile.parameter_policy)
    if str(validation_spec.selection_policy) != str(optimizer_policy):
        raise ValueError(
            "validation profile與protocol selector不一致: "
            f"profile={profile.profile_id}, profile_policy={profile.parameter_policy}, "
            f"protocol_policy={validation_spec.selection_policy}"
        )
    return parameter_set, protocol, str(profile.parameter_policy)


def _validation_budget_build_contract(
    *,
    family: str,
    dataset: str | None,
    max_positions: int | None,
    rotation: str | None,
    fixed_risk: float | None,
    max_position_cap_pct: float | None,
    optimizer_seed: int | None = None,
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> dict[str, Any]:
    from services.optimizer.validation_budget_refit import resolve_validation_budget_spec

    base = _canonical_schedule_build_contract(
        family=family,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
    )
    if optimizer_seed is not None:
        base["optimizer_seed"] = int(optimizer_seed)
    parameter_set, protocol_id, _compare_policy = _resolve_validation_budget_publication_identity(
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    from core.strategy_param_policy import resolve_validation_strategy_param_training_profile

    profile = resolve_validation_strategy_param_training_profile(parameter_set, protocol_id)
    spec = resolve_validation_budget_spec(
        int(base["trials_per_fold"]), protocol_id=protocol_id
    )
    contract = {
        **base,
        "parameter_set": parameter_set,
        "optimizer_search_protocol": spec.protocol,
        "validation_budget_refit": spec.as_dict(),
    }
    if profile.score_policy_overrides:
        contract["score_policy_overrides"] = dict(profile.score_policy_overrides)
    return contract



def _validation_budget_robustness_param_root(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> Path:
    root = Path(project_root).resolve()
    benchmark = normalize_strategy_param_benchmark_id(str(benchmark_id))
    seed_value = normalize_strategy_param_benchmark_seed(int(seed))
    relative = _validation_paramset_relative(
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    return root / relative / "benchmark" / benchmark / f"seed_{seed_value}"


def validation_budget_robustness_strategy_param_artifact_path(
    project_root: str | Path, *, benchmark_id: str, seed: int,
    policy: str = "base-finalists-agree",
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> Path:
    from filters.breakout_quality.strategy_compare_sources import PARAM_POLICY_SPECS

    compare_policy = str(policy).strip()
    spec = PARAM_POLICY_SPECS.get(compare_policy)
    if spec is None:
        raise ValueError(f"validation-budget robustness不支援param policy: {compare_policy!r}")
    return _validation_budget_robustness_param_root(
        project_root,
        benchmark_id=benchmark_id,
        seed=seed,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    ) / str(spec["filename"])


def validation_budget_robustness_strategy_param_manifest_path(
    project_root: str | Path, *, benchmark_id: str, seed: int,
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> Path:
    return _validation_budget_robustness_param_root(
        project_root,
        benchmark_id=benchmark_id,
        seed=seed,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    ) / "identity_manifest.json"


def _validation_budget_robustness_contract(
    *, benchmark_id: str, seed: int, dataset: str, max_positions: int,
    rotation: str, fixed_risk: float, max_position_cap_pct: float,
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> dict[str, Any]:
    benchmark = dict(get_robustness_benchmark_policy_snapshot())
    benchmark_name = normalize_strategy_param_benchmark_id(str(benchmark_id))
    seed_value = normalize_strategy_param_benchmark_seed(int(seed))
    if benchmark_name != str(benchmark["benchmark_id"]):
        raise ValueError(f"benchmark_id與current SSOT不一致: {benchmark_id}")
    if seed_value not in {int(value) for value in benchmark["resolved_seeds"]}:
        raise ValueError(f"seed不屬於current robustness benchmark題庫: {seed_value}")
    contract = _validation_budget_build_contract(
        family="full", dataset=str(dataset), max_positions=int(max_positions),
        rotation=str(rotation), fixed_risk=float(fixed_risk),
        max_position_cap_pct=float(max_position_cap_pct), optimizer_seed=seed_value,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    return {
        **contract,
        "benchmark_id": benchmark_name,
        "benchmark_seed": seed_value,
        "same_seed_oos_rolling": True,
    }


def inspect_validation_budget_robustness_strategy_parameter_artifact(
    project_root: str | Path, *, benchmark_id: str, seed: int, policy: str,
    comparison_end_date: str, dataset: str, max_positions: int, rotation: str,
    fixed_risk: float, max_position_cap_pct: float,
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    target = validation_budget_robustness_strategy_param_artifact_path(
        root,
        benchmark_id=benchmark_id,
        seed=seed,
        policy=policy,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    manifest_path = validation_budget_robustness_strategy_param_manifest_path(
        root,
        benchmark_id=benchmark_id,
        seed=seed,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    contract = _validation_budget_robustness_contract(
        benchmark_id=benchmark_id, seed=seed, dataset=dataset,
        max_positions=max_positions, rotation=rotation, fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    manifest = _load_json_object(manifest_path)
    coverage_end = _canonical_schedule_coverage_end(target)
    ready = bool(
        target.is_file()
        and manifest is not None
        and first_manifest_identity_mismatch(
            manifest,
            (
                ("producer", "string", "optimizer", "PRODUCER_MISMATCH"),
                ("parameter_set", "string", contract["parameter_set"], "MANIFEST_PARAMETER_SET_MISMATCH"),
                ("policy", "string", str(policy).strip(), "MANIFEST_POLICY_MISMATCH"),
                ("build_contract", "mapping", contract, "BUILD_CONTRACT_MISMATCH"),
            ),
        ) is None
        and manifest_artifact_sha_matches(manifest, target)
        and coverage_end is not None
        and str(coverage_end) >= str(comparison_end_date)
    )
    action = resolve_artifact_reuse_action(
        "READY" if ready else "ARTIFACT_UNAVAILABLE_OR_STALE",
        existed=bool(target.exists() or manifest_path.exists()),
    )
    return {"action": action, "path": target, "manifest_path": manifest_path}


def ensure_validation_budget_robustness_strategy_parameter_artifact(
    project_root: str | Path, *, benchmark_id: str, seed: int, policy: str,
    comparison_end_date: str, dataset: str, max_positions: int, rotation: str,
    fixed_risk: float, max_position_cap_pct: float, resume_parameter_training: bool = True,
    quiet: bool = False, parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
    force_rebuild: bool = False,
    oos_end_cap_date: str | None = None,
) -> dict[str, Any]:
    from services.optimizer.strategy_param_training import prepare_selection_historical_full_roos_params
    from services.optimizer.validation_budget_refit import resolve_validation_budget_spec

    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    _parameter_set, _protocol_id, expected_compare_policy = (
        _resolve_validation_budget_publication_identity(
            parameter_set_identity=parameter_set_identity,
            validation_protocol_id=validation_protocol_id,
        )
    )
    if compare_policy != expected_compare_policy:
        raise ValueError(
            "validation robustness policy與registered profile不一致: "
            f"expected={expected_compare_policy!r}, actual={compare_policy!r}"
        )
    state = inspect_validation_budget_robustness_strategy_parameter_artifact(
        root, benchmark_id=benchmark_id, seed=seed, policy=compare_policy,
        comparison_end_date=comparison_end_date, dataset=dataset,
        max_positions=max_positions, rotation=rotation, fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    target = Path(state["path"])
    manifest_path = Path(state["manifest_path"])
    contract = _validation_budget_robustness_contract(
        benchmark_id=benchmark_id, seed=seed, dataset=dataset,
        max_positions=max_positions, rotation=rotation, fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    if str(state["action"]) == "REUSE" and not force_rebuild:
        return {
            "action": "REUSE", "path": target, "manifest_path": manifest_path,
            "sha256": compute_strategy_param_file_sha256(target),
            "scientific_sha256": compute_strategy_param_scientific_sha256(target),
        }

    benchmark = str(contract["benchmark_id"])
    seed_value = int(contract["benchmark_seed"])
    canonical_work_root = (
        root
        / _validation_work_relative(
            parameter_set_identity=parameter_set_identity,
            validation_protocol_id=validation_protocol_id,
        )
        / "benchmark"
        / benchmark
        / f"seed_{seed_value}"
    )
    work_root = (
        _force_rebuild_attempt_root(canonical_work_root)
        if force_rebuild
        else canonical_work_root
    )
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(work_root / "work_contract.json", dict(contract))
    schedule_work = work_root / "schedule_2021_forward"
    spec = resolve_validation_budget_spec(int(contract["trials_per_fold"]), protocol_id=validation_protocol_id)
    result = prepare_selection_historical_full_roos_params(
        project_root=root, dataset=str(dataset), param_policy=compare_policy,
        trials_per_fold=int(contract["trials_per_fold"]),
        max_positions=int(max_positions), rotation=str(rotation),
        fixed_risk=float(fixed_risk), max_position_cap_pct=float(max_position_cap_pct),
        optimizer_seed=seed_value, resume_parameter_training=(False if force_rebuild else bool(resume_parameter_training)),
        quiet=bool(quiet), first_oos_date=str(contract["schedule_start"]),
        last_oos_date=str(comparison_end_date),
        train_window_months=int(contract["train_window_months"]),
        oos_months=int(contract["oos_horizon_months"]),
        output_relative_dir=schedule_work,
        build_context_label=(
            f"{contract['parameter_set']} Robustness {benchmark} seed={seed_value}"
        ),
        optimizer_output_title=(
            f"Optimizer {contract['parameter_set']} robustness工作工件（非production）"
        ),
        optimizer_search_protocol=spec.protocol, parameter_set_identity=str(contract["parameter_set"]),
        score_policy_overrides=dict(contract.get("score_policy_overrides") or {}),
        oos_end_cap_date=oos_end_cap_date,
    )
    source = Path(result["params_path"]).resolve()
    _assert_optimizer_work_source(source_path=source, work_root=work_root)
    if not source.is_file():
        raise FileNotFoundError(
            f"{contract['parameter_set']} robustness Optimizer完成但找不到策略參數工件: "
            + _project_relative(root, source)
        )
    payload = _copy_json_payload(source, target)
    coverage_end = _canonical_schedule_coverage_end(target)
    if coverage_end is None or str(coverage_end) < str(comparison_end_date):
        raise RuntimeError(
            f"{contract['parameter_set']} robustness schedule coverage不足: "
            f"required={comparison_end_date}, actual={coverage_end or '-'}"
        )
    adaptation = dict(payload.get("breakout_quality_param_adaptation") or {})
    if str(adaptation.get("parameter_set") or "") != str(contract["parameter_set"]):
        raise RuntimeError("PARAM-P5 robustness artifact缺少parameter_set identity")
    if str(adaptation.get("optimizer_search_protocol") or "") != spec.protocol:
        raise RuntimeError("PARAM-P5 robustness artifact缺少validation-budget protocol identity")
    artifact_sha = compute_strategy_param_file_sha256(target)
    _write_json(
        manifest_path,
        {
            "schema_type": "strategy_parameter_research_robustness_identity",
            "schema_version": 1,
            "producer": "optimizer",
            "training_dl_enabled": False,
            "parameter_set": str(contract["parameter_set"]),
            "policy": compare_policy,
            "benchmark_id": benchmark,
            "benchmark_seed": seed_value,
            "optimizer_search_protocol": spec.protocol,
            "build_contract": contract,
            "artifact_path": _project_relative(root, target),
            "artifact_sha256": artifact_sha,
            "coverage_end": coverage_end,
            "production_promotion": False,
        },
    )
    return {
        "action": "FORCE_REBUILD" if force_rebuild else ("REBUILD" if str(state["action"]) == "REBUILD" else "BUILD"),
        "path": target, "manifest_path": manifest_path,
        "sha256": artifact_sha,
        "scientific_sha256": compute_strategy_param_scientific_sha256(target),
    }


def inspect_validation_budget_strategy_parameter_artifact(
    project_root: str | Path, *, policy: str = "base-finalists-agree",
    comparison_end_date: str, dataset: str | None = None,
    max_positions: int | None = None, rotation: str | None = None,
    fixed_risk: float | None = None, max_position_cap_pct: float | None = None,
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
) -> dict[str, Any]:
    """Inspect one PARAM-P5 artifact against the exact producer/reuse contract.

    Reuse must be at least as strict as Strategy Compare consumption readiness.
    In particular, a manifest/hash match is insufficient when the payload predates
    the PARAM-P5 adaptation/protocol stamp. Returning REBUILD here prevents the
    orchestrator from repeatedly seeing a stale consumer contract after a no-op REUSE.
    """

    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    normalized_policy = normalize_strategy_param_policy(compare_policy)
    end_date = str(comparison_end_date or "").strip()
    if not end_date:
        raise ValueError("validation-budget缺少comparison_end_date")
    target = validation_budget_strategy_param_artifact_path(
        root,
        policy=compare_policy,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    manifest_path = validation_budget_strategy_param_manifest_path(
        root,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    contract = _validation_budget_build_contract(
        family="full", dataset=dataset, max_positions=max_positions, rotation=rotation,
        fixed_risk=fixed_risk, max_position_cap_pct=max_position_cap_pct,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    manifest = _load_json_object(manifest_path)
    payload = _load_json_object(target)
    coverage_end = _canonical_schedule_coverage_end(target)
    spec_protocol = str(dict(contract.get("validation_budget_refit") or {}).get("protocol") or "")
    adaptation = dict((payload or {}).get("breakout_quality_param_adaptation") or {})
    from core.strategy_param_policy import resolve_validation_strategy_param_training_profile

    profile = resolve_validation_strategy_param_training_profile(
        parameter_set_identity, validation_protocol_id
    )
    required_execution_contract = (
        _current_optimizer_execution_evidence_contract()
        if profile.requires_resume_safe_execution_evidence
        else None
    )
    actual_execution_contract = str(
        (manifest or {}).get("optimizer_execution_evidence_contract") or ""
    ).strip() or None
    required_validation_replay_contract = None
    if dict(profile.score_policy_overrides):
        from services.optimizer.validation_budget_refit import (
            VALIDATION_REPLAY_IMPLEMENTATION_CONTRACT,
        )

        required_validation_replay_contract = VALIDATION_REPLAY_IMPLEMENTATION_CONTRACT
    actual_validation_replay_contract = str(
        (manifest or {}).get("validation_replay_implementation_contract") or ""
    ).strip() or None

    status = "READY"
    if payload is None:
        status = "ARTIFACT_MISSING_OR_INVALID"
    elif str(payload.get("selector") or "") != normalized_policy:
        status = "POLICY_SELECTOR_MISMATCH"
    elif str(adaptation.get("parameter_set") or "") != str(contract["parameter_set"]):
        status = "PARAMETER_SET_MISMATCH"
    elif bool(adaptation.get("training_dl_enabled", True)):
        status = "TRAINING_DL_ENABLED_MISMATCH"
    elif str(adaptation.get("optimizer_search_protocol") or "") != spec_protocol:
        status = "OPTIMIZER_SEARCH_PROTOCOL_MISMATCH"
    elif coverage_end is None or str(coverage_end) < end_date:
        status = "SCHEDULE_COVERAGE_INCOMPLETE"
    elif manifest is None:
        status = "MANIFEST_MISSING_OR_INVALID"
    elif manifest_issue := first_manifest_identity_mismatch(
        manifest,
        (
            ("producer", "string", "optimizer", "PRODUCER_MISMATCH"),
            ("training_dl_enabled", "boolean", False, "MANIFEST_TRAINING_DL_MISMATCH"),
            ("parameter_set", "string", contract["parameter_set"], "MANIFEST_PARAMETER_SET_MISMATCH"),
            ("policy", "string", compare_policy, "MANIFEST_POLICY_MISMATCH"),
            ("optimizer_search_protocol", "string", spec_protocol, "MANIFEST_PROTOCOL_MISMATCH"),
            ("build_contract", "mapping", contract, "BUILD_CONTRACT_MISMATCH"),
        ),
    ):
        status = manifest_issue
    elif not manifest_artifact_sha_matches(manifest, target):
        status = "ARTIFACT_SHA_MISMATCH"
    elif (
        required_validation_replay_contract is not None
        and actual_validation_replay_contract != required_validation_replay_contract
    ):
        status = "VALIDATION_REPLAY_IMPLEMENTATION_STALE"
    elif (
        required_execution_contract is not None
        and actual_execution_contract != required_execution_contract
    ):
        status = "OPTIMIZER_EXECUTION_EVIDENCE_STALE"

    ready = status == "READY"
    existed = bool(target.exists() or manifest_path.exists())
    return {
        "ready": ready,
        "status": status,
        "action": resolve_artifact_reuse_action(status, existed=existed),
        "path": target,
        "manifest_path": manifest_path,
        "coverage_end": coverage_end,
        "build_contract": contract,
        "required_optimizer_execution_evidence_contract": required_execution_contract,
        "actual_optimizer_execution_evidence_contract": actual_execution_contract,
        "required_validation_replay_implementation_contract": required_validation_replay_contract,
        "actual_validation_replay_implementation_contract": actual_validation_replay_contract,
    }

def ensure_validation_budget_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    policy: str = "base-finalists-agree",
    comparison_end_date: str,
    dataset: str | None = None,
    max_positions: int | None = None,
    rotation: str | None = None,
    fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
    parameter_set_identity: str = "PARAM-P5",
    validation_protocol_id: str = "validation_budget_refit_v1",
    force_rebuild: bool = False,
    resume_parameter_training: bool = True,
    oos_end_cap_date: str | None = None,
) -> dict[str, Any]:
    """Build/reuse the PARAM-P5 validation-selected-budget Full PIT schedule.

    This is an optimizer-owned research artifact.  It deliberately does not replace
    the canonical Full schedule used by Production-PIT/C61/C62.
    """
    from config.execution_policy import (
        DEFAULT_FIXED_RISK,
        DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS,
        DEFAULT_PORTFOLIO_ROTATION,
    )
    from core.dataset_profiles import DEFAULT_DATASET_PROFILE
    from services.optimizer.strategy_param_training import (
        prepare_selection_historical_full_roos_params,
    )
    from services.optimizer.validation_budget_refit import resolve_validation_budget_spec

    root = Path(project_root).resolve()
    compare_policy = str(policy).strip()
    _parameter_set, _protocol_id, expected_compare_policy = (
        _resolve_validation_budget_publication_identity(
            parameter_set_identity=parameter_set_identity,
            validation_protocol_id=validation_protocol_id,
        )
    )
    if compare_policy != expected_compare_policy:
        raise ValueError(
            "validation policy與registered profile不一致: "
            f"expected={expected_compare_policy!r}, actual={compare_policy!r}"
        )
    end_date = str(comparison_end_date or "").strip()
    if not end_date:
        raise ValueError("validation-budget缺少comparison_end_date")

    state = inspect_validation_budget_strategy_parameter_artifact(
        root,
        policy=compare_policy,
        comparison_end_date=end_date,
        dataset=dataset,
        max_positions=max_positions,
        rotation=rotation,
        fixed_risk=fixed_risk,
        max_position_cap_pct=max_position_cap_pct,
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    target = Path(state["path"])
    manifest_path = Path(state["manifest_path"])
    contract = dict(state["build_contract"])
    existed_before = str(state["action"]) == "REBUILD"
    if bool(state["ready"]) and not force_rebuild:
        return {
            "action": "REUSE",
            "path": target,
            "manifest_path": manifest_path,
            "sha256": compute_strategy_param_file_sha256(target),
            "coverage_end": state.get("coverage_end"),
            "build_contract": contract,
        }

    canonical_work_root = root / _validation_work_relative(
        parameter_set_identity=parameter_set_identity,
        validation_protocol_id=validation_protocol_id,
    )
    required_execution_contract = state.get(
        "required_optimizer_execution_evidence_contract"
    )
    stale_execution_evidence = (
        str(state.get("status") or "") == "OPTIMIZER_EXECUTION_EVIDENCE_STALE"
    )
    stale_validation_replay = (
        str(state.get("status") or "") == "VALIDATION_REPLAY_IMPLEMENTATION_STALE"
    )
    if stale_execution_evidence:
        if not required_execution_contract:
            raise RuntimeError(
                "execution evidence stale卻缺少required optimizer execution contract"
            )
        _archive_stale_execution_evidence_once(
            root=root,
            target=target,
            manifest_path=manifest_path,
            work_root=canonical_work_root,
            reason=str(state.get("status") or "OPTIMIZER_EXECUTION_EVIDENCE_STALE"),
        )
        work_root = _resume_safe_execution_rebuild_root(
            canonical_work_root, contract=str(required_execution_contract)
        )
    elif force_rebuild:
        work_root = _force_rebuild_attempt_root(canonical_work_root)
    else:
        work_root = canonical_work_root
    spec = resolve_validation_budget_spec(int(contract["trials_per_fold"]), protocol_id=validation_protocol_id)
    result = prepare_selection_historical_full_roos_params(
        project_root=root,
        dataset=str(DEFAULT_DATASET_PROFILE if dataset in (None, "") else dataset),
        param_policy=compare_policy,
        trials_per_fold=int(contract["trials_per_fold"]),
        max_positions=int(
            DEFAULT_PORTFOLIO_MAX_POSITIONS if max_positions is None else max_positions
        ),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION if rotation in (None, "") else rotation),
        fixed_risk=float(DEFAULT_FIXED_RISK if fixed_risk is None else fixed_risk),
        max_position_cap_pct=float(
            DEFAULT_MAX_POSITION_CAP_PCT
            if max_position_cap_pct is None
            else max_position_cap_pct
        ),
        optimizer_seed=int(contract["optimizer_seed"]),
        resume_parameter_training=(
            bool(resume_parameter_training)
            if stale_execution_evidence
            else (False if force_rebuild else bool(resume_parameter_training))
        ),
        quiet=False,
        first_oos_date=str(contract["schedule_start"]),
        last_oos_date=end_date,
        train_window_months=int(contract["train_window_months"]),
        oos_months=int(contract["oos_horizon_months"]),
        output_relative_dir=work_root,
        build_context_label=f"{contract['parameter_set']} Validation-Budget",
        optimizer_output_title=(
            f"Optimizer {contract['parameter_set']} 工作工件（非production）"
        ),
        optimizer_search_protocol=spec.protocol,
        parameter_set_identity=str(contract["parameter_set"]),
        score_policy_overrides=dict(contract.get("score_policy_overrides") or {}),
        oos_end_cap_date=oos_end_cap_date,
    )
    source = Path(result["params_path"]).resolve()
    _assert_optimizer_work_source(source_path=source, work_root=work_root)
    if not source.is_file():
        raise FileNotFoundError(
            f"{contract['parameter_set']} Optimizer完成但找不到策略參數工件: "
            + _project_relative(root, source)
        )
    source_payload = _load_json_object_permissive(source)
    if source_payload is None:
        raise ValueError(
            f"{contract['parameter_set']} Optimizer source無法讀取或不是JSON object: "
            + _project_relative(root, source)
        )
    payload = normalize_strategy_param_payload_for_persistence(source_payload)
    if payload != source_payload:
        # Keep the optimizer work artifact strict-JSON without publishing anything yet.
        atomic_write_json(source, payload)
    if not is_active_param_ensemble_payload(payload):
        coverage_end = None
    elif resolve_active_param_ensemble_mode(payload) != ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING:
        coverage_end = None
    else:
        try:
            _coverage_start, _coverage_end = get_active_param_ensemble_date_range(payload)
            coverage_end = str(_coverage_end) if _coverage_end else None
        except (TypeError, ValueError, KeyError):
            coverage_end = None
    if coverage_end is None or str(coverage_end) < end_date:
        raise RuntimeError(
            f"{contract['parameter_set']} schedule coverage不足: "
            f"required={end_date}, actual={coverage_end or '-'}"
        )
    if str(dict(payload.get("breakout_quality_param_adaptation") or {}).get("parameter_set") or "") != str(contract["parameter_set"]):
        raise RuntimeError(f"{contract['parameter_set']} artifact缺少parameter_set identity")
    if str(dict(payload.get("breakout_quality_param_adaptation") or {}).get("optimizer_search_protocol") or "") != spec.protocol:
        raise RuntimeError(
            f"{contract['parameter_set']} artifact缺少validation-budget protocol identity"
        )

    current_execution_contract = _current_optimizer_execution_evidence_contract()
    required_validation_replay_contract = state.get(
        "required_validation_replay_implementation_contract"
    )
    if (
        required_execution_contract is not None
        and str(required_execution_contract) != current_execution_contract
    ):
        raise RuntimeError(
            "目前Optimizer runtime無法滿足artifact要求的execution evidence contract: "
            f"required={required_execution_contract}, current={current_execution_contract}"
        )
    # Publish only after the complete rebuilt payload and execution contract are validated.
    _write_json(target, payload)
    target_sha = compute_strategy_param_file_sha256(target)
    manifest_payload = {
        "schema_type": "strategy_parameter_research_identity",
        "schema_version": 1,
        "producer": "optimizer",
        "training_dl_enabled": False,
        "parameter_set": str(contract["parameter_set"]),
        "family": "full",
        "policy": compare_policy,
        "optimizer_search_protocol": spec.protocol,
        "build_contract": contract,
        "artifact_path": _project_relative(root, target),
        "artifact_sha256": target_sha,
        "coverage_end": coverage_end,
        "optimizer_execution_evidence_contract": current_execution_contract,
        "production_promotion": False,
    }
    if required_validation_replay_contract is not None:
        manifest_payload["validation_replay_implementation_contract"] = str(
            required_validation_replay_contract
        )
    _write_json(manifest_path, manifest_payload)
    return {
        "action": "FORCE_REBUILD" if force_rebuild else ("REBUILD" if existed_before else "BUILD"),
        "path": target,
        "manifest_path": manifest_path,
        "sha256": target_sha,
        "coverage_end": coverage_end,
        "build_contract": contract,
        "optimizer_execution_evidence_contract": current_execution_contract,
        "validation_replay_implementation_contract": required_validation_replay_contract,
        "execution_evidence_rebuild": bool(stale_execution_evidence),
        "validation_replay_rebuild": bool(stale_validation_replay),
    }


__all__ = [
    "build_isolated_strategy_parameter_schedule",
    "ensure_robustness_benchmark_strategy_parameter_artifact",
    "inspect_robustness_benchmark_strategy_parameter_artifact",
    "ensure_strategy_parameter_artifact",
    "inspect_strategy_parameter_artifact",
    "ensure_validation_budget_strategy_parameter_artifact",
    "inspect_validation_budget_strategy_parameter_artifact",
    "ensure_validation_budget_robustness_strategy_parameter_artifact",
    "inspect_validation_budget_robustness_strategy_parameter_artifact",
    "ensure_fixed_multiseed_strategy_parameter_artifact",
    "run_fixed_multiseed_strategy_parameter_timing_benchmark",
    "ensure_fixed_multiseed_robustness_strategy_parameter_artifact",
    "inspect_fixed_multiseed_robustness_strategy_parameter_artifact",
    "fixed_multiseed_robustness_strategy_param_manifest_path",
    "fixed_multiseed_robustness_strategy_param_artifact_path",
    "inspect_fixed_multiseed_strategy_parameter_artifact",
    "fixed_multiseed_strategy_param_artifact_path",
    "fixed_multiseed_strategy_param_manifest_path",
    "validation_budget_strategy_param_artifact_path",
    "validation_budget_strategy_param_manifest_path",
    "validation_budget_robustness_strategy_param_artifact_path",
    "validation_budget_robustness_strategy_param_manifest_path",
    "validate_strategy_parameter_artifact_identity",
    "migrate_legacy_strategy_parameter_artifacts",
    "migrate_legacy_trade_state_artifacts",
    "migrate_all_legacy_strategy_parameter_artifacts",
    "collect_legacy_root_strategy_parameter_cleanup_plan",
    "finalize_legacy_strategy_parameter_migration",
    "refresh_strategy_parameter_manifest",
]
