"""Canonical publication and reconciliation of already-fitted DL checkpoints.

The same fitted model can be serialized into distinct Forward/Native PIT envelopes.
This service owns immutable FIT cache entries, cross-mode admission, and their
single per-FitKey OS lock. Evaluation/score producers must not own FIT truth.

Existing cache roots and historical FitKey bytes are preserved: moving this
service does not migrate any checkpoint or change scientific identities.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import pickle
import zipfile
import shutil
import time
from pathlib import Path
from typing import Any

from core.file_integrity import exclusive_file_lock, sibling_publication_lock_path
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME
from filters.breakout_quality.pit_fold_provenance import FOLD_MANIFEST_FILENAME
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from services.breakout_quality.fitting_data_dependencies import build_fitting_context_origin
from services.breakout_quality.fitted_model_artifacts import (
    _forward_checkpoint_import_issues,
    _fold_training_contract_is_compatible,
    fold_training_identity,
    fold_contract_with_original_resources,
    resource_equivalent_fitting_cache_entry,
    load_fitted_model_training_evidence,
    certified_fitted_checkpoint_equivalence,
    scientific_torch_execution,
)

def _checkpoint_cache_entry_dir(cache_root: Path, fold_contract: dict[str, Any]) -> Path:
    return Path(cache_root).resolve() / fold_training_identity(fold_contract)



def _import_forward_checkpoint_to_fitting_cache_unlocked(
    *,
    source_model_dir: Path,
    source_report_path: Path,
    cache_root: Path,
    fold_contract: dict[str, Any],
    bundle,
    torch_module,
    execution_plan,
) -> dict[str, Any] | None:
    """Publish an already-fitted Forward checkpoint into the canonical fitting cache.

    Returns ``None`` when the Forward artifact is simply not the fit required by this fold.
    Corruption or a conflicting checkpoint for the same fitting identity fails fast.
    """

    source_model_dir = Path(source_model_dir).resolve()
    source_report_path = Path(source_report_path).resolve()
    source_manifest_path = source_model_dir / FOLD_MANIFEST_FILENAME
    source_model_path = source_model_dir / DEFAULT_MODEL_FILENAME
    if not (source_manifest_path.is_file() and source_model_path.is_file() and source_report_path.is_file()):
        return None
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        if not isinstance(source_manifest, dict) or not isinstance(source_report, dict):
            return None
        checkpoint_manifest = build_file_manifest(source_model_path)
        checkpoint = torch_module.load(source_model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, RuntimeError):
        return None

    source_fitting_evidence, _fitting_evidence_reason = load_fitted_model_training_evidence(
        model_dir=source_model_dir,
        report_path=source_report_path,
    )
    if source_fitting_evidence is None:
        return None

    issues = _forward_checkpoint_import_issues(
        source_manifest=source_manifest,
        source_report=source_report,
        source_fitting_evidence=source_fitting_evidence,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        fold_contract=fold_contract,
        bundle=bundle,
        execution_plan=execution_plan,
        project_root=PROJECT_ROOT,
        source_model_dir=source_model_dir,
    )
    if issues:
        # For the canonical first Rolling fold, Forward and Rolling share the
        # same score start and therefore are expected to share the exact fit.
        # Never silently refit that fold after advertising a validated Forward
        # checkpoint candidate; surface the violated identity field instead.
        source_oos_start = str(dict(source_report.get("split_report") or {}).get("oos_start_date") or "")
        fold_score_start = str(dict(fold_contract.get("planned_periods") or {}).get("score_start") or "")
        if source_oos_start and source_oos_start == fold_score_start:
            raise RuntimeError(
                "Forward OOS checkpoint與第一個Rolling fold應為同一fitting identity但驗證失敗: "
                + "; ".join(issues[:12])
            )
        return None

    # Source publication, execution and checkpoint bytes were all verified above.
    # Capture its original weight-affecting context before returning an existing
    # cache entry, too: prior importers left otherwise valid caches unwitnessed.
    forward_training = dict(source_report.get("training") or {})
    source_origin = build_fitting_context_origin(
        profile=str(fold_contract["experiment_profile"]),
        target=dict(source_manifest.get("source_continuous_target") or {}),
        checkpoint=checkpoint_manifest, kind="forward_import",
        forward_training=forward_training,
        forward_evidence={
            "training_settings": dict(source_fitting_evidence.training_settings),
            "torch_execution": dict(source_fitting_evidence.torch_execution),
            "selected_epoch": int(source_fitting_evidence.selected_epoch),
        },
    )
    forward_cache_metadata = {
        "selected_epoch": int(source_manifest["selected_epoch"]),
        "epoch_selection": dict(forward_training.get("epoch_selection") or {}),
        "final_refit_history": list(forward_training.get("final_refit_history") or []),
        "torch_execution": dict(source_fitting_evidence.torch_execution),
    }

    identity = fold_training_identity(fold_contract)
    entry_dir = _checkpoint_cache_entry_dir(Path(cache_root), fold_contract)
    cached_manifest_path = entry_dir / FOLD_MANIFEST_FILENAME
    cached_model_path = entry_dir / DEFAULT_MODEL_FILENAME
    if entry_dir.exists():
        if entry_dir.is_symlink() or not entry_dir.is_dir():
            raise RuntimeError(f"fitting identity cache entry is not a plain directory: identity={identity}")
        if not (cached_manifest_path.is_file() and cached_model_path.is_file()):
            raise RuntimeError(f"fitting identity cache entry不完整: identity={identity}")
        try:
            cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8"))
            cached_checkpoint = build_file_manifest(cached_model_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError(f"fitting identity cache entry無法驗證: identity={identity}") from exc
        if not isinstance(cached_manifest, dict) or not _fold_training_contract_is_compatible(
            cached_manifest, expected_contract=fold_contract
        ):
            raise RuntimeError(f"fitting identity cache contract衝突: identity={identity}")
        if cached_checkpoint != dict(dict(cached_manifest.get("artifacts") or {}).get("checkpoint") or {}):
            raise RuntimeError(f"fitting identity cache checkpoint manifest mismatch: identity={identity}")
        same_fit_different_envelope = False
        if cached_checkpoint != checkpoint_manifest:
            # Raw hashes authenticate each publication, but are not the FIT
            # equivalence key. A legacy native PIT envelope is permitted only
            # when its embedded original fold proof and exact tensors agree
            # with the independently certified Forward fitted-model sidecar.
            # An already-imported Forward or unknown migration cannot be
            # rebranded: original source provenance must remain immutable.
            origin = cached_manifest.get("fitting_origin")
            migration = cached_manifest.get("migration")
            native_cache = (
                (origin is None or (isinstance(origin, dict)
                    and origin.get("kind") == "rolling_native"
                    and origin.get("checkpoint") == cached_checkpoint))
                and (migration is None or (isinstance(migration, dict)
                    and migration.get("kind") == "rolling_native"))
            )
            proof_reason = "cache is not an original native Rolling FIT publication"
            if native_cache:
                try:
                    native_payload = torch_module.load(
                        cached_model_path, map_location="cpu", weights_only=True,
                    )
                    same_fit_different_envelope, proof_reason = certified_fitted_checkpoint_equivalence(
                        existing_checkpoint=native_payload,
                        existing_manifest=cached_manifest,
                        existing_kind="native",
                        incoming_checkpoint=checkpoint,
                        incoming_manifest=source_manifest,
                        incoming_kind="forward",
                        expected_fold_contract=fold_contract,
                        expected_execution=scientific_torch_execution(execution_plan),
                        torch_module=torch_module,
                    )
                except (OSError, RuntimeError, TypeError, ValueError, KeyError,
                        EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as exc:
                    proof_reason = f"native FIT payload validation failed ({type(exc).__name__})"
            if not same_fit_different_envelope:
                raise RuntimeError(
                    "同一fitting identity已有不同checkpoint；"
                    f"identity={identity}, cached={cached_checkpoint.get('sha256')}, "
                    f"forward={checkpoint_manifest.get('sha256')}; "
                    f"canonical FIT proof rejected: {proof_reason}"
                )
        if source_origin is not None:
            existing_origin = cached_manifest.get("fitting_origin")
            migration = cached_manifest.get("migration")
            imported_cache = (
                isinstance(migration, dict)
                and migration.get("kind") == "forward_model_fitting_identity_import"
            )
            if imported_cache and existing_origin is not None and existing_origin != source_origin:
                raise RuntimeError(
                    "same FIT/cache checkpoint has conflicting Forward context origin: "
                    f"identity={identity}"
                )
            if imported_cache and existing_origin is None:
                # Upgrade only an *existing Forward-import cache* whose bytes
                # match the fully verified original Forward checkpoint. This is
                # not a historical inference from the cache alone. Keep native
                # Rolling caches, original weights and the FitKey unchanged.
                for key, value in forward_cache_metadata.items():
                    if key in cached_manifest and cached_manifest[key] is not None and cached_manifest[key] != value:
                        raise RuntimeError(
                            "same FIT/cache has conflicting original Forward evidence: "
                            f"identity={identity}, field={key}"
                        )
                upgraded = {**cached_manifest, **forward_cache_metadata, "fitting_origin": source_origin}
                temporary = cached_manifest_path.with_name(
                    f".{cached_manifest_path.name}.verified_origin_{os.getpid()}_{time.time_ns()}"
                )
                try:
                    write_json(temporary, upgraded)
                    temporary.replace(cached_manifest_path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
        return {
            "training_identity": identity,
            "cache_entry": str(entry_dir),
            "reused_existing_cache": True,
            "same_fit_different_checkpoint_envelope": same_fit_different_envelope,
        }

    cache_manifest = {
        **fold_contract,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **forward_cache_metadata,
        **({"fitting_origin": source_origin} if source_origin is not None else {}),
        "migration": {
            "kind": "forward_model_fitting_identity_import",
            "training_contract_unchanged": True,
            "source_score_reused": False,
            "source_checkpoint_sha_preserved": True,
            "source_forward_checkpoint": checkpoint_manifest,
        },
        "artifacts": {"checkpoint": checkpoint_manifest},
    }
    entry_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = entry_dir.parent / f".{entry_dir.name}.forward_import_{os.getpid()}_{time.time_ns()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source_model_path, staging / DEFAULT_MODEL_FILENAME)
        write_json(staging / FOLD_MANIFEST_FILENAME, cache_manifest)
        if build_file_manifest(staging / DEFAULT_MODEL_FILENAME) != checkpoint_manifest:
            raise RuntimeError(f"Forward cache staged checkpoint changed: identity={identity}")
        write_json(
            staging / "cache_manifest.json",
            {
                "schema_version": 1,
                "training_identity": identity,
                "source_kind": "forward_model_fitting_identity_import",
                "checkpoint": checkpoint_manifest,
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if entry_dir.exists():
            raise RuntimeError(f"fitting identity cache entry appeared during Forward import: identity={identity}")
        staging.replace(entry_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "training_identity": identity,
        "cache_entry": str(entry_dir),
        "reused_existing_cache": False,
    }


def _import_forward_checkpoint_to_fitting_cache(
    *, source_model_dir: Path, source_report_path: Path, cache_root: Path,
    fold_contract: dict[str, Any], bundle, torch_module, execution_plan,
) -> dict[str, Any] | None:
    """One FitKey writer at a time; source/evidence checks stay in the sole FIT owner."""
    entry = _checkpoint_cache_entry_dir(Path(cache_root), fold_contract)
    with exclusive_file_lock(
        sibling_publication_lock_path(entry),
        conflict_message="Another shared FIT checkpoint publication is active",
    ):
        return _import_forward_checkpoint_to_fitting_cache_unlocked(
            source_model_dir=source_model_dir,
            source_report_path=source_report_path,
            cache_root=cache_root,
            fold_contract=fold_contract,
            bundle=bundle, torch_module=torch_module,
            execution_plan=execution_plan,
        )


_SCORE_ONLY_CHECKPOINT_REUSE_MIGRATION_KINDS = frozenset({
    "expanded_daily_score_universe_checkpoint_reuse",
    "checkpoint_score_regeneration",
    "fitting_identity_checkpoint_reuse",
})


def _is_score_only_checkpoint_reuse_manifest(manifest: dict[str, Any]) -> bool:
    migration = manifest.get("migration")
    return (
        isinstance(migration, dict)
        and migration.get("training_contract_unchanged") is True
        and str(migration.get("kind") or "") in _SCORE_ONLY_CHECKPOINT_REUSE_MIGRATION_KINDS
    )


def _publish_fold_checkpoint_to_cache_unlocked(
    *, fold_dir: Path, cache_root: Path | None, fold_contract: dict[str, Any]
) -> dict[str, Any] | None:
    if cache_root is None:
        return None
    source_fold_dir = Path(fold_dir).resolve()
    manifest_path = source_fold_dir / FOLD_MANIFEST_FILENAME
    model_path = source_fold_dir / DEFAULT_MODEL_FILENAME
    if not (manifest_path.is_file() and model_path.is_file()):
        raise RuntimeError(f"PIT fold缺少可發布checkpoint: {source_fold_dir.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not _fold_training_contract_is_compatible(
        manifest, expected_contract=fold_contract
    ):
        raise RuntimeError(f"PIT fold fitting identity與目前contract不一致: {source_fold_dir.name}")
    checkpoint_manifest = build_file_manifest(model_path)
    if checkpoint_manifest != dict(manifest.get("artifacts") or {}).get("checkpoint"):
        raise RuntimeError(f"PIT fold checkpoint hash/size與manifest不一致: {source_fold_dir.name}")

    identity = fold_training_identity(fold_contract)
    entry_dir = _checkpoint_cache_entry_dir(Path(cache_root), fold_contract)
    cached_manifest_path = entry_dir / FOLD_MANIFEST_FILENAME
    cached_model_path = entry_dir / DEFAULT_MODEL_FILENAME
    if entry_dir.exists():
        if entry_dir.is_symlink() or not entry_dir.is_dir():
            raise RuntimeError(f"fitting identity cache entry is not a plain directory: identity={identity}")
        if not (cached_manifest_path.is_file() and cached_model_path.is_file()):
            raise RuntimeError(f"fitting identity cache entry不完整: identity={identity}")
        try:
            cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8"))
            cached_checkpoint = build_file_manifest(cached_model_path)
            cached_identity = fold_training_identity(cached_manifest)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError(f"FIT cache manifest cannot be certified: identity={identity}") from exc
        if cached_identity == identity and cached_checkpoint != dict(dict(cached_manifest.get("artifacts") or {}).get("checkpoint") or {}):
            raise RuntimeError(f"fitting identity cache checkpoint manifest mismatch: identity={identity}")
        if cached_identity == identity and cached_checkpoint == checkpoint_manifest:
            return {
                "training_identity": identity,
                "cache_entry": str(entry_dir),
                "reused_existing_cache": True,
            }
        if cached_identity == identity and cached_checkpoint != checkpoint_manifest:
            # Both are native PIT Fold publications. A certified duplicate
            # cannot create a second physical truth under the same FitKey.
            # Keep original checkpoint and manifest bytes in both locations;
            # only score-only legacy migration has explicit restore authority.
            if not _is_score_only_checkpoint_reuse_manifest(manifest):
                try:
                    import torch
                    original_model = torch.load(
                        cached_model_path, map_location="cpu", weights_only=True,
                    )
                    candidate_model = torch.load(
                        model_path, map_location="cpu", weights_only=True,
                    )
                    original_epoch = int(cached_manifest.get("selected_epoch") or 0)
                    candidate_epoch = int(manifest.get("selected_epoch") or 0)
                    execution = dict(cached_manifest.get("torch_execution") or {})
                    execution.pop("requested_device", None)
                    same_fit, _reason = certified_fitted_checkpoint_equivalence(
                        existing_checkpoint=original_model,
                        existing_manifest=cached_manifest,
                        existing_kind="native",
                        incoming_checkpoint=candidate_model,
                        incoming_manifest=manifest,
                        incoming_kind="native",
                        expected_fold_contract=fold_contract,
                        expected_execution=execution,
                        torch_module=torch,
                    ) if original_epoch and candidate_epoch else (False, "missing epoch")
                    if same_fit:
                        return {
                            "training_identity": identity,
                            "cache_entry": str(entry_dir),
                            "reused_existing_cache": True,
                            "same_fit_different_checkpoint_envelope": True,
                        }
                except (OSError, RuntimeError, ValueError, TypeError, KeyError,
                        EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as exc:
                    # Unverifiable evidence cannot be treated as an equivalent
                    # FIT. Preserve the exact failure and both original artifacts;
                    # the publisher must neither overwrite nor silently REUSE.
                    raise RuntimeError(
                        "同一fitting identity產生不同checkpoint；"
                        f"identity={identity}, cached={cached_checkpoint.get('sha256')}, "
                        f"current={checkpoint_manifest.get('sha256')}; "
                        f"FIT equivalence verification failed: {type(exc).__name__}: {exc}"
                    ) from exc
            if _is_score_only_checkpoint_reuse_manifest(manifest):
                # A score-only rescore must never create a second fitted checkpoint.
                # Older code rewrote model.pt only to refresh its embedded fold_contract,
                # which changed the file SHA despite identical model weights.  Canonicalize
                # such resumable folds back to the existing fitting-identity cache bytes.
                restore_path = source_fold_dir / (
                    f".{DEFAULT_MODEL_FILENAME}.cache_restore_{os.getpid()}_{time.time_ns()}"
                )
                try:
                    shutil.copy2(cached_model_path, restore_path)
                    restore_path.replace(model_path)
                finally:
                    if restore_path.exists():
                        restore_path.unlink()
                restored_checkpoint = build_file_manifest(model_path)
                if restored_checkpoint != cached_checkpoint:
                    raise RuntimeError(
                        "fitting identity cache checkpoint還原後hash不一致；"
                        f"identity={identity}, cached={cached_checkpoint.get('sha256')}, "
                        f"restored={restored_checkpoint.get('sha256')}"
                    )
                migration = dict(manifest.get("migration") or {})
                manifest = {
                    **manifest,
                    "migration": {
                        **migration,
                        "canonical_checkpoint_cache_restore": True,
                        "source_checkpoint_sha_preserved": True,
                        "superseded_local_checkpoint": checkpoint_manifest,
                    },
                    "artifacts": {
                        **dict(manifest.get("artifacts") or {}),
                        "checkpoint": restored_checkpoint,
                    },
                }
                write_json(manifest_path, manifest)
                return {
                    "training_identity": identity,
                    "cache_entry": str(entry_dir),
                    "reused_existing_cache": True,
                    "restored_source_fold_checkpoint": True,
                }
            raise RuntimeError(
                "同一fitting identity產生不同checkpoint；"
                f"identity={identity}, cached={cached_checkpoint.get('sha256')}, "
                f"current={checkpoint_manifest.get('sha256')}"
            )

    entry_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = entry_dir.parent / f".{entry_dir.name}.publish_{os.getpid()}_{time.time_ns()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(model_path, staging / DEFAULT_MODEL_FILENAME)
        shutil.copy2(manifest_path, staging / FOLD_MANIFEST_FILENAME)
        if build_file_manifest(staging / DEFAULT_MODEL_FILENAME) != checkpoint_manifest:
            raise RuntimeError(f"PIT cache staged checkpoint changed: identity={identity}")
        staged_manifest = json.loads((staging / FOLD_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        if (not isinstance(staged_manifest, dict)
            or not _fold_training_contract_is_compatible(staged_manifest, expected_contract=fold_contract)
            or dict(dict(staged_manifest.get("artifacts") or {}).get("checkpoint") or {}) != checkpoint_manifest):
            raise RuntimeError(f"PIT cache staged manifest/weights changed: identity={identity}")
        write_json(
            staging / "cache_manifest.json",
            {
                "schema_version": 1,
                "training_identity": identity,
                "source_fold_id": str(manifest.get("fold_id") or source_fold_dir.name),
                "checkpoint": checkpoint_manifest,
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        # Do not destroy an existing FIT identity, including incomplete or
        # concurrently published evidence. The per-FitKey OS lock serializes
        # cooperating producers; rename is the first visible publication.
        if entry_dir.exists():
            raise RuntimeError(f"fitting identity cache entry appeared during publication: identity={identity}")
        staging.replace(entry_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "training_identity": identity,
        "cache_entry": str(entry_dir),
        "reused_existing_cache": False,
    }


def _publish_fold_checkpoint_to_cache(
    *, fold_dir: Path, cache_root: Path | None, fold_contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Single FIT cache publisher for native fold checkpoints and score-only reuse."""
    if cache_root is None:
        return None
    candidate = resource_equivalent_fitting_cache_entry(Path(cache_root), fold_contract)
    original_manifest_path = (
        candidate / FOLD_MANIFEST_FILENAME if candidate.exists()
        else Path(fold_dir) / FOLD_MANIFEST_FILENAME
    )
    if original_manifest_path.is_file():
        original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise RuntimeError("FIT cache publication requires an object manifest")
        if candidate.exists() and candidate.name != fold_training_identity(original):
            raise RuntimeError("FIT cache physical key differs from its manifest; publication blocked")
        fold_contract = fold_contract_with_original_resources(fold_contract, original)
    entry = _checkpoint_cache_entry_dir(Path(cache_root), fold_contract)
    with exclusive_file_lock(
        sibling_publication_lock_path(entry),
        conflict_message="Another shared FIT checkpoint publication is active",
    ):
        return _publish_fold_checkpoint_to_cache_unlocked(
            fold_dir=fold_dir, cache_root=cache_root, fold_contract=fold_contract,
        )


def publish_verified_native_first_fold_cache(
    *, rolling_dir: Path, score_start: str, cache_root: Path,
    expected_checkpoint_sha256: str,
) -> str:
    """PIT producer publication after an isolated canonical FIT rebuild.

    The old first-fold shared cache must already be transactionally isolated.
    Normal force-rebuild uses a temporary cache to prevent accidental mutation;
    after complete Rolling/Forward identity verification the repairing caller
    restores this one new native FitKey entry through the canonical publisher.
    """
    from services.breakout_quality.fitted_model_artifacts import _fit_repair_first_fold

    path, manifest = _fit_repair_first_fold(rolling_dir=rolling_dir, score_start=score_start)
    fold_dir = path.parent
    actual = build_file_manifest(fold_dir / DEFAULT_MODEL_FILENAME)
    origin = manifest.get("fitting_origin")
    if (not isinstance(origin, dict) or origin.get("kind") != "rolling_native"
        or dict(origin.get("checkpoint") or {}) != actual
        or dict(dict(manifest.get("artifacts") or {}).get("checkpoint") or {}) != actual
        or actual.get("sha256") != str(expected_checkpoint_sha256)):
        raise RuntimeError("Rebuilt native first-fold FIT has no verified checkpoint origin")
    identity = fold_training_identity(manifest)
    entry = Path(cache_root).resolve() / identity
    # The repair-specific "must be absent" rule and canonical publication
    # must be one lock transaction; a late concurrent writer is not a REUSE.
    with exclusive_file_lock(
        sibling_publication_lock_path(entry),
        conflict_message="Another shared FIT checkpoint publication is active",
    ):
        if entry.exists():
            raise RuntimeError("New FIT cache identity already occupied during transaction")
        # All score, target, and sample semantics are from the actual PIT Fold
        # contract; no Research-side cache writer or synthetic FitKey is introduced.
        _publish_fold_checkpoint_to_cache_unlocked(
            fold_dir=fold_dir, cache_root=Path(cache_root), fold_contract=manifest,
        )
        cached = entry / FOLD_MANIFEST_FILENAME
        cached_model = entry / DEFAULT_MODEL_FILENAME
        if (not cached.is_file() or not cached_model.is_file()
            or build_file_manifest(cached_model) != actual
            or fold_training_identity(json.loads(cached.read_text(encoding="utf-8"))) != identity):
            raise RuntimeError("Rebuilt native shared FIT cache publication failed verification")
    return identity


