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
    get_active_param_ensemble_policy,
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
    legacy = _load_json_object(source)
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
    from filters.breakout_quality.strategy_param_training import (
        FULL_ROOS_SEARCH_FIELDS,
        MIN_ROOS_SEARCH_FIELDS,
    )
    from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE

    family = normalize_strategy_param_family(family)
    policy = dict(get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling"))
    return {
        "family": family,
        "dataset": str(DEFAULT_DATASET_PROFILE if dataset in (None, "") else dataset),
        "search_space_sha256": canonical_json_sha256(BREAKOUT_OPTIMIZER_SEARCH_SPACE),
        "optimizer_seed": int(policy["optimizer_seed"]),
        "trials_per_fold": int(policy["trials_per_fold"]),
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


def _canonical_manifest_matches_current_schedule(
    root: Path, *, family: str, policy: str, target: Path,
    evaluation_mode: str = "rolling", comparison_end_date: str | None = None,
    dataset: str | None = None, max_positions: int | None = None,
    rotation: str | None = None, fixed_risk: float | None = None,
    max_position_cap_pct: float | None = None,
) -> tuple[bool, str]:
    manifest_path = resolve_strategy_param_manifest_path(
        root, family=family, evaluation_mode="rolling"
    )
    manifest = _load_json_object(manifest_path)
    if manifest is None:
        return False, "MANIFEST_MISSING_OR_INVALID"
    if str(manifest.get("producer") or "") != "optimizer":
        return False, "PRODUCER_MISMATCH"
    if str(manifest.get("family") or "") != str(family):
        return False, "FAMILY_MISMATCH"
    if str(manifest.get("evaluation_mode") or "") != "schedule":
        return False, "STORAGE_MODE_MISMATCH"
    expected_policy = dict(get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling"))
    expected_policy["evaluation_mode"] = "schedule"
    expected_policy["consumption_modes"] = ["oos", "rolling"]
    if dict(manifest.get("training_policy") or {}) != expected_policy:
        return False, "TRAINING_POLICY_MISMATCH"
    record = dict(dict(manifest.get("artifacts") or {}).get(policy) or {})
    if not record:
        return False, "ARTIFACT_NOT_PINNED"
    if str(record.get("path") or "") != _project_relative(root, target):
        return False, "ARTIFACT_PATH_MISMATCH"
    if not target.is_file():
        return False, "ARTIFACT_MISSING"
    if str(record.get("sha256") or "") != compute_strategy_param_file_sha256(target):
        return False, "ARTIFACT_SHA_MISMATCH"
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
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    if schedule_kind not in {"frozen_initial", "annual_refit"}:
        return False, "SCHEDULE_KIND_MISSING_OR_INVALID"
    if mode == "rolling" and comparison_end_date not in (None, "") and str(comparison_end_date) >= "2022-01-01":
        if schedule_kind != "annual_refit":
            return False, "ROLLING_SCHEDULE_INCOMPLETE"
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
    reverse = {value: key for key, value in {
        "base-finalist-best": "base_finalist_best",
        "base-finalists-agree": "base_finalists_agree",
    }.items()}
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
    max_position_cap_pct: float | None = None,
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
    work_root = (
        root / "outputs" / "optimizer" / "strategy_param_schedule"
        / str(family) / str(normalize_strategy_param_policy(policy))
    )
    work_identity = {
        "schema": "canonical_strategy_schedule_work_v1",
        "family": family,
        "policy": normalize_strategy_param_policy(policy),
        "build_contract": contract,
    }
    work_contract_path = work_root / "work_contract.json"
    prior_work_identity = _load_json_object(work_contract_path) if work_contract_path.is_file() else None
    if prior_work_identity != work_identity and work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(work_contract_path, work_identity)
    initial_cache = work_root / "initial_2021.json"

    def _train(first_date: str, last_date: str, output_dir: Path, *, suffix: str) -> Path:
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
            resume_parameter_training=True,
            quiet=False,
            first_oos_date=str(first_date),
            last_oos_date=str(last_date),
            train_window_months=int(contract["train_window_months"]),
            oos_months=int(contract["oos_horizon_months"]),
            output_relative_dir=output_dir,
        )
        if family == "full":
            result = prepare_selection_historical_full_roos_params(**common)
        else:
            result = prepare_selection_historical_p2_params(
                **common,
                comparison_output_root=str(work_root / f"no_recovery_{suffix}"),
                comparison_output_roots=(str(work_root / f"no_recovery_{suffix}"),),
            )
        path = Path(result["params_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Optimizer完成但找不到策略參數輸出: " + _project_relative(root, path)
            )
        return path

    if not initial_cache.is_file():
        initial_source = _train(
            "2021-01-01", "2021-12-31", work_root / "initial_2021", suffix="initial"
        )
        _copy_json_payload(initial_source, initial_cache)

    target = resolve_strategy_param_artifact_path(
        root, family=family, evaluation_mode="rolling", policy=policy
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "oos":
        source_record = _freeze_rolling_policy_to_oos(
            root,
            family=family,
            policy=policy,
            source_path=initial_cache,
            target_path=target,
            coverage_end_date=str(comparison_end_date),
        )
        schedule_kind = "frozen_initial"
    else:
        tail_path: Path | None = None
        if str(comparison_end_date) >= "2022-01-01":
            tail_path = _train(
                "2022-01-01", str(comparison_end_date),
                work_root / "rolling_tail_2022_plus", suffix="tail",
            )
        _stitch_benchmark_rolling_payload(
            initial_path=initial_cache,
            tail_path=tail_path,
            target_path=target,
            comparison_end_date=str(comparison_end_date),
        )
        source_record = {
            "optimizer_build": True,
            "path": _project_relative(root, work_root),
            "initial_path": _project_relative(root, initial_cache),
        }
        schedule_kind = "annual_refit"
    source_record.update({
        "build_contract": contract,
        "schedule_kind": schedule_kind,
        "same_json_oos_rolling": True,
    })
    manifest = refresh_strategy_parameter_manifest(
        root,
        family=family,
        evaluation_mode="rolling",
        source_records={policy: source_record},
    )
    return {
        "path": target,
        "manifest_path": manifest,
        "source_path": initial_cache,
        "build_contract": contract,
        "schedule_kind": schedule_kind,
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
    coverage_ready = bool(
        comparison_end_date in (None, "")
        or (coverage_end is not None and str(coverage_end) >= str(comparison_end_date))
    )
    if target.is_file() and identity_ready and coverage_ready:
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
        "action": "REBUILD" if existed else "BUILD",
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
    benchmark_dir = resolve_strategy_param_benchmark_dir(
        root,
        benchmark_id=benchmark_id,
        seed=int(seed),
        family=family,
        evaluation_mode="rolling",
    )
    return benchmark_dir / "_optimizer_work" / f"seed_{int(seed)}" / str(family)


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
    # OOS and Rolling are two consumption views of the same physical benchmark JSON.
    path = resolve_strategy_param_benchmark_artifact_path(
        root, benchmark_id=benchmark_id, seed=seed, family=family,
        evaluation_mode="rolling", policy=policy,
    )
    if not path.is_file():
        return None
    return _benchmark_effective_member_sha256(path)



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


def _benchmark_schedule_build_contract(
    *, benchmark_id: str, seed: int, family: str, policy: str, trials_per_fold: int,
    dataset: str, max_positions: int, rotation: str, fixed_risk: float,
    max_position_cap_pct: float,
) -> dict[str, Any]:
    from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE

    return {
        "schema": "robustness_strategy_schedule_work_v1",
        "benchmark_id": str(benchmark_id),
        "seed": int(seed),
        "family": normalize_strategy_param_family(family),
        "policy": normalize_strategy_param_policy(policy),
        "trials_per_fold": int(trials_per_fold),
        "dataset": str(dataset),
        "max_positions": int(max_positions),
        "rotation": str(rotation),
        "fixed_risk": float(fixed_risk),
        "max_position_cap_pct": float(max_position_cap_pct),
        "train_window_months": 120,
        "oos_months": 12,
        "search_space_sha256": canonical_json_sha256(BREAKOUT_OPTIMIZER_SEARCH_SPACE),
    }


def _benchmark_manifest_matches_current_policy(
    manifest_path: Path, *, benchmark: dict[str, Any], benchmark_id: str, seed: int,
    family: str, evaluation_mode: str, policy: str, target_path: Path,
    comparison_end_date: str, build_contract: dict[str, Any],
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
        or str(payload.get("evaluation_mode") or "") != "schedule"
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
        or str(training.get("evaluation_mode") or "") != "schedule"
    ):
        return False
    artifact = dict(dict(payload.get("artifacts") or {}).get(policy) or {})
    expected_sha = str(artifact.get("sha256") or "")
    if not expected_sha or not target_path.is_file():
        return False
    if expected_sha != compute_strategy_param_file_sha256(target_path):
        return False
    coverage_end = _benchmark_parameter_coverage_end(target_path)
    if coverage_end is None or str(coverage_end) < str(comparison_end_date):
        return False
    source = dict(artifact.get("source") or {})
    if dict(source.get("build_contract") or {}) != dict(build_contract):
        return False
    schedule_kind = str(source.get("schedule_kind") or "")
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    if mode == "rolling" and str(comparison_end_date) >= "2022-01-01":
        return schedule_kind == "annual_refit"
    return schedule_kind in {"frozen_initial", "annual_refit"}


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
    build_contract = _benchmark_schedule_build_contract(
        benchmark_id=str(benchmark_id), seed=int(seed), family=family, policy=policy,
        trials_per_fold=trials, dataset=str(dataset), max_positions=int(max_positions),
        rotation=str(rotation), fixed_risk=float(fixed_risk),
        max_position_cap_pct=float(max_position_cap_pct),
    )
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
        and (_benchmark_parameter_coverage_end(target) or "") >= str(comparison_end_date)
        and _benchmark_manifest_matches_current_policy(
            manifest_path,
            benchmark=benchmark,
            benchmark_id=str(benchmark_id),
            seed=int(seed),
            family=family,
            evaluation_mode=mode,
            policy=policy,
            target_path=target,
            comparison_end_date=str(comparison_end_date),
            build_contract=build_contract,
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
    benchmark_work_identity = dict(build_contract)
    benchmark_work_contract = work_root / "work_contract.json"
    prior_benchmark_work = (
        _load_json_object(benchmark_work_contract) if benchmark_work_contract.is_file() else None
    )
    if prior_benchmark_work != benchmark_work_identity and work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(benchmark_work_contract, benchmark_work_identity)
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
                "schedule_kind": "frozen_initial" if mode == "oos" else "annual_refit",
                "build_contract": build_contract,
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
    "validate_strategy_parameter_artifact_identity",
    "migrate_legacy_strategy_parameter_artifacts",
    "migrate_legacy_trade_state_artifacts",
    "migrate_all_legacy_strategy_parameter_artifacts",
    "collect_legacy_root_strategy_parameter_cleanup_plan",
    "finalize_legacy_strategy_parameter_migration",
    "refresh_strategy_parameter_manifest",
]
