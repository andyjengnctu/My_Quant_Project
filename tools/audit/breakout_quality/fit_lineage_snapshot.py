"""Read-only, all-profile FIT lineage snapshot for a single provenance diagnosis.

Only checkpoint metadata and file SHA are exported. It does not copy weights,
stock rows, raw dataset contents or tokens, and never changes fitted artifacts.
This is an engineering audit, not a source of FIT reuse authorization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.breakout_quality import BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT
from core.breakout_quality_policy import (
    get_breakout_quality_model_test_settings,
    get_breakout_quality_rolling_test_mode,
    get_breakout_quality_workflow_settings,
)
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
)
from services.breakout_quality.fitted_model_artifacts import (
    fold_training_identity,
    resolve_rolling_fitted_folds_root,
)
from services.breakout_quality.fitting_data_dependencies import context_source_for_profile
from filters.breakout_quality.predicted_context_artifact import maybe_predicted_context_artifact_spec


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "[outside project root]"


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "MISSING"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        return None, f"INVALID_JSON:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "INVALID_SCHEMA"
    return payload, None


def _file_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            size += len(block)
    return {"filename": path.name, "sha256": sha.hexdigest(), "size_bytes": size}


def _record_match(record: Any, actual: Any) -> bool:
    return (
        isinstance(record, Mapping)
        and isinstance(actual, Mapping)
        and record.get("sha256") == actual.get("sha256")
        and record.get("size_bytes") == actual.get("size_bytes")
    )


def _simple_fingerprint(payload: Any) -> str | None:
    if payload is None:
        return None
    try:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _context_summary(source: Any, key: str | None) -> dict[str, Any] | None:
    if not isinstance(source, Mapping) or key is None:
        return None
    context = source.get(key)
    if not isinstance(context, Mapping):
        return None
    # These are published identity records only, not price rows or secret values.
    return {
        "schema_version": context.get("schema_version"),
        "artifact": context.get("artifact"),
        "source_stage1": context.get("source_stage1"),
        "context_identity_sha256": _simple_fingerprint(context),
    }


def _checkpoint_metadata(path: Path, *, include_torch: bool) -> dict[str, Any]:
    if not include_torch or not path.is_file():
        return {"inspected": False}
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"inspected": False, "error": "torch unavailable: " + type(exc).__name__}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return {"inspected": False, "error": "checkpoint metadata unreadable: " + type(exc).__name__}
    if not isinstance(payload, dict):
        return {"inspected": False, "error": "checkpoint is not a mapping"}
    embedded = payload.get("fold_contract")
    result = {
        "inspected": True,
        "keys": sorted(str(k) for k in payload.keys()),
        "has_embedded_fold_contract": isinstance(embedded, dict),
        "embedded_fold_contract_sha256": _simple_fingerprint(embedded),
        "embedded_source_contract_sha256": _simple_fingerprint(
            embedded.get("source_contract") if isinstance(embedded, dict) else None
        ),
        "selected_epoch": payload.get("selected_epoch"),
        "experiment_profile": payload.get("experiment_profile"),
        "training_sample_scope": payload.get("training_sample_scope"),
    }
    del payload
    return result


def _inspect_profile(root: Path, model_id: str, profile: str, *, include_torch: bool) -> dict[str, Any]:
    settings = get_breakout_quality_workflow_settings(experiment_profile=profile)
    paths = resolve_filter_artifact_paths(root, settings.filter_id, settings.model_architecture, profile)
    output_dir = resolve_filter_model_output_dir(root, settings.filter_id, settings.model_architecture, profile)
    folds_root = resolve_rolling_fitted_folds_root(
        project_root=root, filter_id=settings.filter_id,
        model_architecture=settings.model_architecture, experiment_profile=profile,
    )
    expected_start = str(get_breakout_quality_rolling_test_mode("rolling").score_start_date)
    aggregate_path = folds_root.parent / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
    aggregate, aggregate_error = _read_json(aggregate_path)
    forward_manifest, forward_error = _read_json(paths.manifest_path)
    forward_model_record = _file_record(paths.model_path)
    forward_report_path = output_dir / "continuous_ranker_report.json"
    forward_report, report_error = _read_json(forward_report_path)
    forward_fit, forward_fit_error = _read_json(paths.model_dir / "fitted_model_manifest.json")
    context_source = context_source_for_profile(profile)
    spec = maybe_predicted_context_artifact_spec(context_source)
    context_key = spec.manifest_key if spec is not None else None
    external_context = spec is not None

    matches: list[tuple[Path, dict[str, Any]]] = []
    fold_inventory: list[dict[str, Any]] = []
    if folds_root.is_dir():
        for fold_path in sorted(folds_root.glob("*/manifest.json")):
            fold, error = _read_json(fold_path)
            start = str(dict(fold.get("planned_periods") or {}).get("score_start") or "") if fold else ""
            fold_inventory.append({
                "fold_id": fold_path.parent.name,
                "score_start": start or None,
                "manifest_error": error,
                "has_checkpoint": (fold_path.parent / "model.pt").is_file(),
            })
            if fold and start == expected_start:
                matches.append((fold_path, fold))
    selected: list[dict[str, Any]] = []
    for fold_path, fold in matches:
        checkpoint_path = fold_path.parent / "model.pt"
        checkpoint_record = _file_record(checkpoint_path)
        declared_record = dict(fold.get("artifacts") or {}).get("checkpoint")
        migration = fold.get("migration")
        aggregate_folds = aggregate.get("folds") if aggregate else None
        aggregate_matches = (
            [row for row in aggregate_folds
             if isinstance(row, Mapping) and row.get("fold_id") == fold.get("fold_id")]
            if isinstance(aggregate_folds, list) else []
        )
        binding = len(aggregate_matches) == 1 and _record_match(
            aggregate_matches[0].get("fold_manifest"), _file_record(fold_path)
        )
        origin_sha = checkpoint_record.get("sha256") if checkpoint_record else None
        archive = paths.model_dir / "historical_forward_fits" / str(origin_sha)
        archived_model = _file_record(archive / "model" / "model.pt") if origin_sha else None
        archived_proof, archive_error = _read_json(archive / "archive_manifest.json")
        archived_origin, archive_origin_error = _read_json(archive / "model" / "manifest.json")
        cache_location = None
        cache_record = None
        cached_manifest = None
        cached_manifest_error = "UNKNOWN_FIT_IDENTITY"
        try:
            identity = fold_training_identity(fold)
            cache_location = root / BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT / identity
            cache_record = _file_record(cache_location / "model.pt")
            cached_manifest, cached_manifest_error = _read_json(cache_location / "manifest.json")
        except (KeyError, TypeError, ValueError):
            identity = None
        provenance_sources: list[str] = []
        if _record_match(checkpoint_record, forward_model_record):
            provenance_sources.append("current_forward_same_checkpoint_sha")
        if _record_match(checkpoint_record, archived_model):
            provenance_sources.append("historical_forward_same_checkpoint_sha")
        if _record_match(checkpoint_record, cache_record):
            provenance_sources.append("fit_cache_same_checkpoint_sha")
        chosen = {
            "fold_id": fold_path.parent.name,
            "path": _rel(root, fold_path),
            "fold_manifest": _file_record(fold_path),
            "checkpoint": checkpoint_record,
            "declared_checkpoint_matches_bytes": _record_match(declared_record, checkpoint_record),
            "aggregate_fold_manifest_matches": binding,
            "migration": migration if isinstance(migration, dict) else None,
            "fitting_context_provenance": fold.get("fitting_context_provenance"),
            "fitting_origin": fold.get("fitting_origin"),
            "origin_candidate_locations": provenance_sources,
            "fit_cache_identity": identity,
            "cache_checkpoint": cache_record,
            "cache_evidence": {
                "manifest_error": cached_manifest_error,
                "checkpoint_matches_cache_manifest": _record_match(
                    dict((cached_manifest or {}).get("artifacts") or {}).get("checkpoint"), cache_record
                ),
                "origin": (cached_manifest or {}).get("fitting_origin"),
                "migration": (cached_manifest or {}).get("migration"),
                "selected_epoch": (cached_manifest or {}).get("selected_epoch"),
                "source_contract_sha256": _simple_fingerprint(
                    (cached_manifest or {}).get("source_contract")
                ),
            },
            "archive_checkpoint": archived_model,
            "archive_evidence": {
                "manifest_error": archive_error,
                "origin_manifest_error": archive_origin_error,
                "recorded_original_checkpoint_matches": _record_match(
                    (archived_proof or {}).get("original_model"), archived_model
                ),
                "original_target_contract_sha256": _simple_fingerprint(
                    (archived_origin or {}).get("source_continuous_target", {}).get("target_contract")
                    if isinstance((archived_origin or {}).get("source_continuous_target"), Mapping)
                    else None
                ),
                "original_context": _context_summary(
                    (archived_origin or {}).get("source_continuous_target"), context_key
                ),
            },
            "selected_epoch": fold.get("selected_epoch"),
            "has_epoch_selection": isinstance(fold.get("epoch_selection"), Mapping),
            "fold_context": _context_summary((aggregate or {}).get("source_continuous_target"), context_key),
            "checkpoint_metadata": _checkpoint_metadata(checkpoint_path, include_torch=include_torch),
            "source_contract_identity_sha256": _simple_fingerprint(fold.get("source_contract")),
            "fold_fitting_field_sha256": {
                key: _simple_fingerprint(fold.get(key))
                for key in ("training_settings", "model_spec", "experiment_settings",
                            "planned_periods", "observed_periods", "group_counts", "lookahead_contract")
            },
        }
        selected.append(chosen)

    blockers: list[str] = []
    if not matches:
        blockers.append("FIRST_FOLD_MANIFEST_MISSING")
    if aggregate_error:
        blockers.append("ROLLING_AGGREGATE_" + aggregate_error)
    if external_context and _context_summary((aggregate or {}).get("source_continuous_target"), context_key) is None:
        blockers.append("PUBLISHED_EXTERNAL_CONTEXT_MISSING")
    for item in selected:
        if not item["declared_checkpoint_matches_bytes"]:
            blockers.append("FOLD_CHECKPOINT_SHA_MISMATCH")
        if not item["aggregate_fold_manifest_matches"]:
            blockers.append("AGGREGATE_FOLD_SHA_MISMATCH")
        migration = item["migration"]
        if external_context and migration is not None and not item["fitting_context_provenance"] and not item["fitting_origin"]:
            if not any(source.endswith("forward_same_checkpoint_sha") for source in item["origin_candidate_locations"]):
                blockers.append("MIGRATED_EXTERNAL_CONTEXT_ORIGINAL_FORWARD_MISSING")
    return {
        "model_id": model_id, "profile": profile, "external_context_source": context_source,
        "forward": {
            "model": forward_model_record,
            "manifest_error": forward_error,
            "manifest_file": _file_record(paths.manifest_path),
            "report_file": _file_record(forward_report_path),
            "fitted_sidecar_file": _file_record(paths.model_dir / "fitted_model_manifest.json"),
            "published_source_dataset_sha256": _simple_fingerprint(
                (forward_manifest or {}).get("source_dataset")
            ),
            "published_target_contract_sha256": _simple_fingerprint(
                ((forward_manifest or {}).get("source_continuous_target") or {}).get("target_contract")
                if isinstance((forward_manifest or {}).get("source_continuous_target"), Mapping)
                else None
            ),
            "training_evidence": {
                "selected_epoch": ((forward_report or {}).get("training") or {}).get("selected_epoch"),
                "epoch_selection": ((forward_report or {}).get("training") or {}).get("epoch_selection"),
                "torch_execution": (forward_report or {}).get("torch_execution"),
            },
            "manifest_model_record_matches": _record_match((forward_manifest or {}).get("model"), forward_model_record),
            "report_error": report_error,
            "report_checkpoint_matches": _record_match(dict((forward_report or {}).get("artifacts") or {}).get("model"), forward_model_record),
            "fitted_sidecar_error": forward_fit_error,
            "fitted_sidecar": {
                "selected_epoch": (forward_fit or {}).get("selected_epoch"),
                "torch_execution": (forward_fit or {}).get("torch_execution"),
                "training_settings": (forward_fit or {}).get("training_settings"),
            },
            "source_context": _context_summary((forward_manifest or {}).get("source_continuous_target"), context_key),
        },
        "rolling_aggregate": {
            "path": _rel(root, aggregate_path), "error": aggregate_error,
            "source_context": _context_summary((aggregate or {}).get("source_continuous_target"), context_key),
            "fold_count": len(fold_inventory),
        },
        "fold_inventory": fold_inventory,
        "first_folds": selected,
        "blockers": sorted(set(blockers)),
        "classification": "SOURCE_EVIDENCE_GAP" if blockers else "CANDIDATES_PRESENT_NOT_FULLY_VERIFIED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only evidence bundle for all configured FIT comparison profiles")
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument("--no-torch", action="store_true", help="skip embedded checkpoint metadata inspection")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    models = []
    for model_id, profile in get_breakout_quality_model_test_settings().model_profiles:
        models.append(_inspect_profile(root, model_id, profile, include_torch=not args.no_torch))
    report = {
        "audit_schema": "fit_lineage_snapshot_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "note": "This snapshot does not certify FIT reuse; model/context integrity requires the canonical validator.",
        "model_count": len(models),
        "models": models,
    }
    output_dir = root / "outputs" / "filters" / "breakout_quality" / "breakout_quality_v1" / "audit" / "fit_lineage_snapshot"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fit_lineage_snapshot.json"
    zip_path = output_dir / "fit_lineage_snapshot.zip"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(json_path, arcname=json_path.name)
    print("\nAll-profile FIT provenance snapshot | read-only inspection")
    for model in models:
        blocker_list = ", ".join(model["blockers"]) or "no structural evidence gap detected"
        print(f"{model['model_id']}: {model['classification']} | {blocker_list}")
    print("Output: " + _rel(root, zip_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
