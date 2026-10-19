"""One owner for Continuous DL fitting *data* dependencies across evaluation modes.

The canonical dependency snapshot is projected into the historical Forward and
Rolling publication schemas without changing their serialized identity/FitKey.
Comparison is based on data actually consumed by fitting, not unrelated event
Dataset cache bytes. External context is checked through its registered artifact
capability and a published Fold -> context provenance chain (fail closed).

Do not put MR identities, evaluation score horizons or report fields here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.breakout_quality_policy import get_continuous_ranker_execution_recipe
from core.breakout_quality_registry import TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.dataset_store import resolve_dataset_paths
from filters.breakout_quality.model_source_provenance import (
    canonical_source_dataset_artifacts, source_inventory_matches_current,
)
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME, resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
)
from filters.breakout_quality.predicted_context_artifact import maybe_predicted_context_artifact_spec


@dataclass(frozen=True)
class ForwardOriginEvidence:
    """Previously published original FIT, addressed by checkpoint bytes."""

    model_dir: Path
    report_path: Path
    report: Mapping[str, Any]


@dataclass(frozen=True)
class FitDataDependencies:
    """Single resolved snapshot; projections preserve archived manifest field names."""

    summary: Mapping[str, Any]
    target: Mapping[str, Any]

    @classmethod
    def from_bundle(cls, bundle: Any) -> "FitDataDependencies":
        return cls(summary=dict(bundle.summary or {}), target=dict(bundle.target_manifest or {}))

    def forward_source(self) -> dict[str, Any]:
        """Exact V1 Forward source schema; never introduce a second fitting identity."""
        summary, target = self.summary, self.target
        return {
            "dataset": summary.get("dataset"),
            "dataset_policy": summary.get("policy"),
            "training_universe_start_date": summary.get("training_universe_start_date"),
            "source_data_inventory": summary.get("source_data_inventory"),
            **({"input_price_source": summary["input_price_source"]} if "input_price_source" in summary else {}),
            **({"input_trade_count_source": summary["input_trade_count_source"]} if "input_trade_count_source" in summary else {}),
            "source_dataset_artifacts": canonical_source_dataset_artifacts(summary),
            "source_target_artifacts": target.get("artifacts"),
            "target_id": target.get("target_id"),
            "target_contract": target.get("target_contract"),
            "context_features": target.get("context_features"),
            "pair_weight_context_features": target.get("pair_weight_context_features"),
            "pair_weight_context_contract": target.get("pair_weight_context_contract"),
            "risk_param_coverage_start": target.get("risk_param_coverage_start"),
            "predicted_upside_context_manifest": target.get("predicted_upside_context_manifest"),
            "predicted_safety_context_manifest": target.get("predicted_safety_context_manifest"),
        }

    def rolling_source(
        self, *, daily_training_universe_start_date: str | None = None,
    ) -> dict[str, Any]:
        """Exact V2 PIT Fold source schema; preserve historical FitKey bytes."""
        summary, target = self.summary, self.target
        result = {
            "dataset_policy": summary.get("policy"),
            "dataset_storage_schema_version": summary.get("dataset_storage_schema_version"),
            "source_data_inventory": summary.get("source_data_inventory"),
            **({"input_price_source": summary["input_price_source"]} if "input_price_source" in summary else {}),
            **({"input_trade_count_source": summary["input_trade_count_source"]} if "input_trade_count_source" in summary else {}),
            "dataset_artifacts": canonical_source_dataset_artifacts(summary),
            "target_schema_version": target.get("schema_version"),
            "target_contract": target.get("target_contract"),
            "target_artifacts": target.get("artifacts"),
        }
        if daily_training_universe_start_date is not None:
            result["training_universe_start_date"] = str(daily_training_universe_start_date)
        return result


def context_source_for_profile(profile: str) -> str:
    """Resolve declared fitting context solely through canonical runtime recipe."""
    return get_continuous_ranker_execution_recipe(str(profile)).context_policy.source


def _event_dataset_record_blocker(published: Any, current: Any) -> str | None:
    """Validate evidence shape; missing *historical* event-only cache is legal."""
    files = resolve_dataset_paths(Path(".")).artifact_paths()
    expected_keys = set(files)
    if published is not None and not isinstance(published, Mapping):
        return "historical event Dataset records have an unsupported representation"
    for label, records in (("historical", published), ("current", current)):
        if label == "historical" and records is None:
            continue
        if not isinstance(records, Mapping):
            return f"{label} event Dataset records absent or not a mapping"
        if set(records) != expected_keys:
            missing = sorted(expected_keys - set(records))
            return (
                f"{label} event Dataset key set differs: "
                f"missing={','.join(missing) or '-'}, "
                f"unexpected_count={len(set(records) - expected_keys)}, "
                f"record_count={len(records)}"
            )
        for name, path in files.items():
            record = records.get(name)
            if not isinstance(record, Mapping):
                return f"{label} {name} record is not a mapping"
            if record.get("filename") != path.name:
                return f"{label} {name} filename differs from canonical Dataset contract"
            sha = record.get("sha256")
            if (
                not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha.lower())
            ):
                return f"{label} {name} SHA256 metadata invalid"
            size = record.get("size_bytes")
            if type(size) is not int or size < 0:
                return f"{label} {name} file-size metadata invalid"
    return None


def _non_consumed_event_cache_drift_blocker(
    candidate: Mapping[str, Any], expected: Mapping[str, Any],
    old_source: Mapping[str, Any], new_source: Mapping[str, Any],
    *, project_root: Path | None, inventory_verified: bool,
    external_context_verified: bool,
) -> str | None:
    """A capability policy, not a profile/model-specific exception."""
    if project_root is None:
        return "canonical project root missing"
    if not inventory_verified:
        return "source CSV inventory equivalence not verified"
    if (
        candidate.get("training_sample_scope") != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        or expected.get("training_sample_scope") != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        return "training sample scope is not daily-universal"
    if candidate.get("experiment_profile") != expected.get("experiment_profile"):
        return "experiment profile differs"
    try:
        recipe = get_continuous_ranker_execution_recipe(str(expected.get("experiment_profile") or ""))
    except (KeyError, TypeError, ValueError):
        return "experiment profile has no verified execution recipe"
    if recipe.target_policy.materialization_mode != CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT:
        return "target materialization is not daily-component"
    if recipe.context_policy.source != CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE and not external_context_verified:
        return "external context requires published context/fit binding proof"
    for label, source in (("historical", old_source), ("current", new_source)):
        if not isinstance(source.get("source_data_inventory"), Mapping) or not source["source_data_inventory"]:
            return f"{label} source CSV inventory evidence absent"
    old_target = old_source.get("target_contract")
    if not isinstance(old_target, Mapping) or not old_target:
        return "historical target contract missing or empty"
    if old_target != new_source.get("target_contract"):
        return "target contract differs"
    for key in ("dataset_artifacts", "target_artifacts"):
        if key not in old_source or key not in new_source:
            return f"{key} field absent"
    target_artifacts = old_source["target_artifacts"]
    if target_artifacts is not None and (not isinstance(target_artifacts, Mapping) or not target_artifacts):
        return "historical target artifact record invalid"
    if target_artifacts != new_source["target_artifacts"]:
        return "target artifact lineage differs"
    return _event_dataset_record_blocker(
        old_source["dataset_artifacts"], new_source["dataset_artifacts"],
    )


def fold_source_dependency_issues(
    candidate: Mapping[str, Any], expected: Mapping[str, Any],
    *, project_root: Path | None,
    inventory_matcher: Callable[..., bool],
    external_context_verified: bool = False,
) -> tuple[str, ...]:
    """Compare all Fold fitting sources through one strict dependency resolver."""
    old_source = candidate.get("source_contract")
    new_source = expected.get("source_contract")
    if not isinstance(old_source, Mapping) or not isinstance(new_source, Mapping):
        return ("source_contract missing",)
    inventory_verified = (
        "source_data_inventory" in old_source
        and "source_data_inventory" in new_source
        and inventory_matcher(
            published=old_source["source_data_inventory"],
            current=new_source["source_data_inventory"], project_root=project_root,
        )
    )
    issues = []
    for key in sorted(set(old_source) | set(new_source)):
        if key not in old_source or key not in new_source:
            issues.append(f"source_contract.{key} missing")
        elif key == "source_data_inventory":
            if not inventory_verified:
                issues.append("source_contract.source_data_inventory mismatch")
        elif old_source[key] != new_source[key]:
            if key == "dataset_artifacts":
                blocker = _non_consumed_event_cache_drift_blocker(
                    candidate, expected, old_source, new_source,
                    project_root=project_root, inventory_verified=inventory_verified,
                    external_context_verified=external_context_verified,
                )
                if blocker is not None:
                    issues.append("source_contract.dataset_artifacts mismatch")
                    issues.append(f"dataset_artifacts proof blocked: {blocker}")
            else:
                issues.append(f"source_contract.{key} mismatch")
    return tuple(issues)


def _valid_file_record(record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    sha = record.get("sha256")
    size = record.get("size_bytes")
    return (
        isinstance(sha, str) and len(sha) == 64
        and all(c in "0123456789abcdef" for c in sha.lower())
        and type(size) is int and size >= 0
    )


def _context_manifest_mismatch(published: Any, current: Any) -> str | None:
    """One registered-context content/Stage-1 proof for both FIT directions."""
    if not isinstance(published, Mapping) or not isinstance(current, Mapping):
        return "published or current predicted context manifest missing"
    mandatory = (
        "schema_version", "filter_id", "model_architecture", "experiment_profile",
        "contract", "dataset_policy", "periods", "rows", "artifact",
    )
    if any(published.get(key) is None or published.get(key) != current.get(key)
           for key in mandatory):
        return "published predicted context bytes/contract differ"
    if not _valid_file_record(published.get("artifact")):
        return "published context content hash missing or invalid"
    old_stage = published.get("source_stage1")
    new_stage = current.get("source_stage1")
    if not isinstance(old_stage, Mapping) or not isinstance(new_stage, Mapping):
        return "published Stage-1 lineage missing"
    if any(old_stage.get(key) is None or old_stage.get(key) != new_stage.get(key)
           for key in ("research_id", "profile", "architecture", "seed")):
        return "published Stage-1 scientific identity differs"
    for key in ("selection_score", "selection_manifest", "forward_score", "forward_manifest"):
        if not _valid_file_record(old_stage.get(key)) or not _valid_file_record(new_stage.get(key)):
            return "published Stage-1 provenance incomplete"
        if old_stage[key] != new_stage[key]:
            return f"published Stage-1 {key} artifact lineage differs"
    return None


def forward_publication_fit_dependency_issues(
    *, source_dataset: Mapping[str, Any], source_target: Mapping[str, Any],
    expected_fold_source: Mapping[str, Any], current_target: Mapping[str, Any],
    experiment_profile: str, project_root: Path,
    inventory_matcher: Callable[..., bool],
) -> tuple[str, ...]:
    """Forward -> Rolling import uses the same persisted-source projection owner.

    The old Forward publication is evidence, not a second scientific source
    identity. Fold manifest/identity/checkpoint decisions remain in FIT owner.
    """
    source = FitDataDependencies(source_dataset, source_target).rolling_source(
        daily_training_universe_start_date=(
            source_dataset.get("training_universe_start_date")
            if "training_universe_start_date" in expected_fold_source else None
        ),
    )
    issues = []
    for key in expected_fold_source:
        if key == "source_data_inventory":
            matches = inventory_matcher(
                published=source.get(key), current=expected_fold_source[key],
                project_root=project_root,
            )
        else:
            matches = source.get(key) == expected_fold_source[key]
        if not matches:
            issues.append(f"source_contract {key} mismatch")
    try:
        context_source = context_source_for_profile(experiment_profile)
    except (KeyError, ValueError, TypeError):
        return tuple(issues + ["unregistered fitting context recipe"])
    if context_source != CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE:
        spec = maybe_predicted_context_artifact_spec(context_source)
        if spec is None:
            issues.append("registered external fitting context capability unavailable")
        else:
            for key in (
                "schema_version", "target_contract", "artifacts", "context_features",
                "pair_weight_context_features", "pair_weight_context_contract", "risk_param_coverage_start",
            ):
                if source_target.get(key) != current_target.get(key):
                    issues.append(f"source_target {key} mismatch")
            reason = _context_manifest_mismatch(
                source_target.get(spec.manifest_key), current_target.get(spec.manifest_key),
            )
            if reason is not None:
                issues.append(f"source_context proof blocked: {reason}")
    return tuple(dict.fromkeys(issues))


def _same_file_bytes(first: Any, second: Any) -> bool:
    """A copied checkpoint may have a different filename but identical bytes."""
    return (
        _valid_file_record(first) and _valid_file_record(second)
        and first["sha256"] == second["sha256"]
        and first["size_bytes"] == second["size_bytes"]
    )


def _read_published_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _migrated_fold_forward_context_proof(
    *, fold_manifest: Mapping[str, Any], original_target: Mapping[str, Any],
    current_target: Mapping[str, Any], context_key: str | None, project_root: Path | None,
) -> tuple[ForwardOriginEvidence | None, str | None]:
    """Recover a migrated FIT's *original* context from a byte-bound Forward publication.

    A rescore/reuse migration may overwrite the native Fold's origin marker; its
    new aggregate is not evidence of which context created the old weights.
    Only a matching original Forward checkpoint publication (canonical or
    immutable historical archive) can establish that missing source binding.
    No synthetic lineage is written back into historical artifacts.
    """
    migration = fold_manifest.get("migration")
    valid_kinds = frozenset({
        "forward_model_fitting_identity_import",
        "fitting_identity_checkpoint_reuse",
        "expanded_daily_score_universe_checkpoint_reuse",
        "checkpoint_score_regeneration",
    })
    if (not isinstance(migration, Mapping)
        or migration.get("kind") not in valid_kinds
        or migration.get("training_contract_unchanged") is not True
        or migration.get("source_checkpoint_sha_preserved") is not True):
        return None, "migrated Fold has no approved unchanged-checkpoint provenance"
    artifacts = fold_manifest.get("artifacts")
    checkpoint_record = artifacts.get("checkpoint") if isinstance(artifacts, Mapping) else None
    if not _valid_file_record(checkpoint_record):
        return None, "migrated Fold checkpoint identity absent"
    for field in ("source_checkpoint", "source_forward_checkpoint"):
        if field in migration and not _same_file_bytes(migration[field], checkpoint_record):
            return None, f"migrated Fold {field} SHA differs from checkpoint"
    if migration.get("kind") == "forward_model_fitting_identity_import" and (
        not _same_file_bytes(migration.get("source_forward_checkpoint"), checkpoint_record)
    ):
        return None, "Forward import lacks source checkpoint SHA binding"
    if project_root is None:
        return None, "original Forward provenance project root unavailable"
    root = Path(project_root).resolve()
    try:
        paths = resolve_filter_artifact_paths(
            root, str(fold_manifest["filter_id"]),
            str(fold_manifest["model_architecture"]),
            str(fold_manifest["experiment_profile"]),
        )
        output_dir = resolve_filter_model_output_dir(
            root, str(fold_manifest["filter_id"]),
            str(fold_manifest["model_architecture"]),
            str(fold_manifest["experiment_profile"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, "migrated Fold has no resolvable original Forward namespace"

    sources = [(paths.model_dir, output_dir, None)]
    # The historical archive is hash-addressed. Never scan unrelated profiles
    # or rely on the mutable current research-model selection.
    archive = paths.model_dir / "historical_forward_fits" / checkpoint_record["sha256"]
    if archive.is_dir():
        sources.append((archive / "model", archive / "report", archive))

    saw_checkpoint = False
    for model_dir, report_dir, archive_dir in sources:
        original_model = model_dir / "model.pt"
        if (not original_model.is_file()
            or not _same_file_bytes(build_file_manifest(original_model), checkpoint_record)):
            continue
        saw_checkpoint = True
        original_manifest_path = model_dir / "manifest.json"
        original_report_path = report_dir / "continuous_ranker_report.json"
        original_manifest = _read_published_json(original_manifest_path)
        original_report = _read_published_json(original_report_path)
        if not isinstance(original_manifest, Mapping) or not isinstance(original_report, Mapping):
            continue
        if archive_dir is not None:
            archive_manifest = _read_published_json(archive_dir / "archive_manifest.json")
            if (not isinstance(archive_manifest, Mapping)
                or archive_manifest.get("archive_version") != 1
                or not _same_file_bytes(archive_manifest.get("original_model"), checkpoint_record)
                or not _valid_file_record(archive_manifest.get("replacement_rolling_model"))):
                continue
            preserved = archive_manifest.get("preserved_files")
            if not isinstance(preserved, list):
                continue
            records = {
                row.get("path"): row.get("manifest") for row in preserved
                if isinstance(row, Mapping) and isinstance(row.get("path"), str)
            }
            original_fit_sidecar = model_dir / "fitted_model_manifest.json"
            evidence_files = [
                ("model/model.pt", original_model),
                ("model/manifest.json", original_manifest_path),
                ("report/continuous_ranker_report.json", original_report_path),
            ]
            if original_fit_sidecar.is_file():
                evidence_files.append(("model/fitted_model_manifest.json", original_fit_sidecar))
            if any(records.get(relative) != build_file_manifest(path)
                   for relative, path in evidence_files):
                continue
        report_artifacts = original_report.get("artifacts")
        forward_outputs = original_manifest.get("research_outputs")
        if not isinstance(report_artifacts, Mapping) or not isinstance(forward_outputs, Mapping):
            continue
        if not _same_file_bytes(original_manifest.get("model"), checkpoint_record):
            continue
        if not _same_file_bytes(report_artifacts.get("model"), checkpoint_record):
            continue
        if forward_outputs.get("report_json") != build_file_manifest(original_report_path):
            continue
        if any(original_manifest.get(key) != fold_manifest.get(key) for key in (
            "filter_id", "model_architecture", "experiment_profile",
            "continuous_target_id", "training_label_scope", "model_spec",
            "experiment_settings", "model_information_cutoff", "selected_epoch",
        )):
            continue
        published_settings = original_manifest.get("experiment_settings")
        if not isinstance(published_settings, Mapping):
            continue
        declared_scope = (original_manifest.get("training_sample_scope")
                          or published_settings.get("training_sample_scope"))
        if declared_scope != fold_manifest.get("training_sample_scope"):
            continue
        training = original_report.get("training")
        split = original_report.get("split_report")
        if not isinstance(training, Mapping) or not isinstance(split, Mapping):
            continue
        sop = original_report.get("standard_model_sop")
        planned = fold_manifest.get("planned_periods")
        counts = fold_manifest.get("group_counts")
        split_counts = split.get("counts")
        if not all(isinstance(item, Mapping) for item in (sop, planned, counts, split_counts)):
            continue
        if (training.get("seed") != fold_manifest.get("seed")
            or training.get("selected_epoch") != fold_manifest.get("selected_epoch")
            or sop.get("evaluation_mode") != "forward_oos"):
            continue
        if any(str(split.get(key)) != str(planned.get(value)) for key, value in (
            ("inner_validation_start_date", "validation_start"),
            ("selection_end_date", "validation_end"),
            ("oos_start_date", "score_start"),
        )) or any(split_counts.get(key) != counts.get(value) for key, value in (
            ("inner_train", "inner_train"),
            ("validation", "validation"),
            ("selection", "final_refit"),
        )):
            continue
        source_dataset = original_manifest.get("source_dataset")
        fold_source = fold_manifest.get("source_contract")
        if not isinstance(source_dataset, Mapping) or not isinstance(fold_source, Mapping):
            continue
        if source_dataset.get("policy") != fold_source.get("dataset_policy"):
            continue
        published_inventory = source_dataset.get("source_data_inventory")
        fold_inventory = fold_source.get("source_data_inventory")
        if (not isinstance(published_inventory, Mapping) or not published_inventory
            or not isinstance(fold_inventory, Mapping) or not fold_inventory
            or not source_inventory_matches_current(
                published=published_inventory,
                current=fold_inventory, project_root=root,
            )):
            continue
        if fold_source.get("training_universe_start_date") is not None and (
            source_dataset.get("training_universe_start_date") != fold_source.get("training_universe_start_date")
        ):
            continue
        original_source_target = original_manifest.get("source_continuous_target")
        if not isinstance(original_source_target, Mapping):
            continue
        if (not isinstance(original_source_target.get("target_contract"), Mapping)
            or not original_source_target["target_contract"]):
            continue
        if any(original_source_target.get(key) != current_target.get(key) for key in (
            "schema_version", "target_contract", "artifacts", "context_features",
            "pair_weight_context_features", "pair_weight_context_contract",
            "risk_param_coverage_start",
        )):
            continue
        if context_key is not None:
            if _context_manifest_mismatch(
                original_source_target.get(context_key), current_target.get(context_key),
            ) is not None:
                continue
            if _context_manifest_mismatch(
                original_source_target.get(context_key), original_target.get(context_key),
            ) is not None:
                continue
        return ForwardOriginEvidence(
            model_dir=model_dir, report_path=original_report_path, report=original_report,
        ), None
    if saw_checkpoint:
        return None, "original Forward checkpoint found but publication/context identity cannot be proven"
    return None, "original Forward checkpoint SHA publication unavailable (including historical archive)"



def migrated_forward_origin_training_evidence(
    *, fold_manifest: Mapping[str, Any], bundle: Any,
    project_root: Path | None,
) -> tuple[ForwardOriginEvidence | None, str | None]:
    """Resolve legacy Forward-origin epoch selection without fabricating Fold metadata.

    The main context proof additionally verifies published Rolling aggregate and
    Fold SHA first. This helper only recovers the *same* original Forward report
    bound to the unchanged checkpoint bytes, after the FIT owner has validated
    the expected cohort/execution and checkpoint integrity.
    """
    target = getattr(bundle, "target_manifest", None)
    if not isinstance(target, Mapping):
        return None, "current Target manifest missing"
    try:
        source = context_source_for_profile(str(fold_manifest.get("experiment_profile") or ""))
    except (KeyError, ValueError, TypeError):
        return None, "execution profile not registered"
    if source == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE:
        key = None
    else:
        spec = maybe_predicted_context_artifact_spec(source)
        if spec is None:
            return None, "predicted-context artifact capability not registered"
        key = spec.manifest_key
    return _migrated_fold_forward_context_proof(
        fold_manifest=fold_manifest, original_target=target,
        current_target=target, context_key=key, project_root=project_root,
    )


def build_fitting_context_origin(
    *, profile: str, target: Mapping[str, Any], checkpoint: Mapping[str, Any],
    kind: str, forward_training: Mapping[str, Any] | None = None,
    forward_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Capture original weight-affecting context once, at the canonical producer.

    This witness belongs to the *fitted checkpoint*, not a score refresh. It
    intentionally does not change historical FitKey serialization. No witness
    is manufactured for legacy models after their original publication is lost.
    """
    context_source = context_source_for_profile(profile)
    if context_source == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE:
        return None
    spec = maybe_predicted_context_artifact_spec(context_source)
    if spec is None:
        raise ValueError("unknown registered fitting-context artifact capability")
    context = target.get(spec.manifest_key)
    if not isinstance(context, Mapping) or not _valid_file_record(context.get("artifact")):
        raise ValueError("original fitting-context content witness missing")
    if not _valid_file_record(checkpoint):
        raise ValueError("original fitted checkpoint SHA witness missing")
    witness: dict[str, Any] = {
        "schema_version": 1,
        "kind": str(kind),
        "context_source": context_source,
        "checkpoint": dict(checkpoint),
        "target_schema_version": target.get("schema_version"),
        "target_contract": target.get("target_contract"),
        "context_manifest": dict(context),
    }
    if kind == "forward_import":
        if not isinstance(forward_training, Mapping) or not isinstance(forward_evidence, Mapping):
            raise ValueError("original Forward fitting evidence missing")
        if not isinstance(forward_training.get("epoch_selection"), Mapping) or not forward_training["epoch_selection"]:
            raise ValueError("original Forward epoch-selection witness missing")
        mandatory = ("training_settings", "torch_execution", "selected_epoch")
        if any(forward_evidence.get(key) is None for key in mandatory):
            raise ValueError("original Forward execution witness incomplete")
        witness["epoch_selection"] = dict(forward_training["epoch_selection"])
        witness["final_refit_history"] = list(forward_training.get("final_refit_history") or [])
        witness["training_evidence"] = {
            key: forward_evidence[key] for key in mandatory
        }
    elif kind != "rolling_native":
        raise ValueError("unsupported fitted-context origin kind")
    return witness


def fitting_context_origin_issue(
    *, origin: Any, profile: str, target: Mapping[str, Any],
    published_target: Mapping[str, Any], checkpoint: Any,
) -> str | None:
    """Verify a producer-captured checkpoint/context witness, without legacy guesses."""
    if not isinstance(origin, Mapping) or origin.get("schema_version") != 1:
        return "original fitting context witness absent or unsupported"
    source = context_source_for_profile(profile)
    spec = maybe_predicted_context_artifact_spec(source)
    if spec is None:
        return "external fitting context capability unavailable"
    if origin.get("context_source") != source:
        return "original fitting context source differs"
    if not _same_file_bytes(origin.get("checkpoint"), checkpoint):
        return "original fitting checkpoint SHA binding differs"
    if origin.get("target_schema_version") != target.get("schema_version"):
        return "original fitting target schema differs"
    if origin.get("target_contract") != target.get("target_contract"):
        return "original fitting target contract differs"
    for label, witness in (("current", target), ("published", published_target)):
        reason = _context_manifest_mismatch(origin.get("context_manifest"), witness.get(spec.manifest_key))
        if reason is not None:
            return f"original fitting context vs {label}: {reason}"
    if origin.get("kind") not in {"forward_import", "rolling_native"}:
        return "original fitting context kind unsupported"
    if origin.get("kind") == "forward_import":
        evidence = origin.get("training_evidence")
        if not isinstance(evidence, Mapping) or not isinstance(origin.get("epoch_selection"), Mapping):
            return "original imported fitting execution/epoch witness missing"
        if not origin["epoch_selection"] or any(
            evidence.get(key) is None for key in ("training_settings", "torch_execution", "selected_epoch")
        ):
            return "original imported fitting execution/epoch witness incomplete"
    return None


def preserve_fitting_migration_chain(
    source_manifest: Mapping[str, Any], next_migration: Mapping[str, Any],
    *, checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Carry the oldest migration descriptor through score-only refreshes.

    The descriptor is *diagnostic*, not an authorization to reuse weights. An
    original fitted-context witness, when present, lives separately in the
    immutable ``fitting_origin`` field and is kept by manifest projections.
    """
    current = dict(next_migration)
    previous = source_manifest.get("migration")
    if not isinstance(previous, Mapping):
        return current
    first = previous.get("initial_fitting_migration")
    if first is None:
        first = dict(previous)
        first.pop("initial_fitting_migration", None)
    if not isinstance(first, Mapping):
        raise RuntimeError("existing FIT migration lineage is invalid")
    if first.get("source_checkpoint_sha_preserved") is True:
        for field in ("source_forward_checkpoint", "source_checkpoint"):
            if field in first and not _same_file_bytes(first[field], checkpoint):
                raise RuntimeError("FIT migration origin checkpoint SHA differs")
    current["initial_fitting_migration"] = dict(first)
    return current


def verified_published_context_fit_proof(
    *, manifest_path: Path, fold_manifest: Mapping[str, Any], bundle: Any,
    project_root: Path | None = None,
) -> tuple[bool, str]:
    """Bind any declared external fitting context to the original published Fold.

    The native Fold V2 contract did not embed context SHA. The published PIT
    aggregate binds the exact Fold manifest bytes to the old target/context
    manifest. A merely READY current Context must never prove historical FIT.
    """
    try:
        source = context_source_for_profile(str(fold_manifest.get("experiment_profile") or ""))
    except (KeyError, ValueError, TypeError):
        return False, "unknown execution recipe"
    if source == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE:
        return True, "no external fitting context"
    spec = maybe_predicted_context_artifact_spec(source)
    if spec is None:
        return False, "no registered external-context artifact proof"
    target = getattr(bundle, "target_manifest", None)
    if not isinstance(target, Mapping):
        return False, "validated current target/context manifest unavailable"
    current_context = target.get(spec.manifest_key)
    if not isinstance(current_context, Mapping):
        return False, "validated current predicted context manifest missing"
    aggregate_path = manifest_path.parent.parent.parent / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
    if not aggregate_path.is_file():
        return False, "published Rolling aggregate manifest unavailable"
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False, "published Rolling aggregate manifest unreadable"
    if not isinstance(aggregate, Mapping):
        return False, "published Rolling aggregate is not a mapping"
    for key in (
        "filter_id", "model_architecture", "experiment_profile",
        "continuous_target_id", "training_sample_scope", "training_label_scope", "seed",
    ):
        if key not in aggregate or aggregate[key] != fold_manifest.get(key):
            return False, f"published Rolling aggregate {key} identity differs"
    folds = aggregate.get("folds")
    if not isinstance(folds, list):
        return False, "published Rolling fold index missing"
    records = [
        row for row in folds if isinstance(row, Mapping)
        and row.get("fold_id") == fold_manifest.get("fold_id")
    ]
    if len(records) != 1:
        return False, "published Rolling fold binding absent or ambiguous"
    record = records[0]
    if (
        record.get("fold_manifest") != build_file_manifest(manifest_path)
        or not isinstance(record.get("artifacts"), Mapping)
        or record["artifacts"].get("checkpoint")
        != dict(fold_manifest.get("artifacts") or {}).get("checkpoint")
    ):
        return False, "published Rolling fold/checkpoint SHA binding differs"
    original_target = aggregate.get("source_continuous_target")
    if not isinstance(original_target, Mapping):
        return False, "published Rolling target/context source missing"
    for key in (
        "schema_version", "target_contract", "artifacts", "context_features",
        "pair_weight_context_features", "pair_weight_context_contract", "risk_param_coverage_start",
    ):
        if original_target.get(key) != target.get(key):
            return False, f"published Rolling target {key} lineage differs"
    old_context = original_target.get(spec.manifest_key)
    if not isinstance(old_context, Mapping):
        return False, "published Rolling predicted context manifest missing"
    context_error = _context_manifest_mismatch(old_context, current_context)
    if context_error is not None:
        return False, context_error
    origin = fold_manifest.get("fitting_origin")
    if origin is not None:
        issue = fitting_context_origin_issue(
            origin=origin,
            profile=str(fold_manifest.get("experiment_profile") or ""),
            target=target, published_target=original_target,
            checkpoint=dict(fold_manifest.get("artifacts") or {}).get("checkpoint"),
        )
        if issue is not None:
            return False, "original FIT context witness invalid: " + issue
    elif fold_manifest.get("migration") is not None:
        _origin_report, origin_error = _migrated_fold_forward_context_proof(
            fold_manifest=fold_manifest, original_target=original_target,
            current_target=target, context_key=spec.manifest_key,
            project_root=project_root,
        )
        if origin_error is not None:
            return False, "migrated Rolling checkpoint lacks original context binding: " + origin_error
    return True, "verified published Rolling context bytes, original FIT witness and PIT fold SHA"


__all__ = [
    "FitDataDependencies", "ForwardOriginEvidence", "context_source_for_profile", "fold_source_dependency_issues",
    "verified_published_context_fit_proof", "migrated_forward_origin_training_evidence",
    "build_fitting_context_origin", "fitting_context_origin_issue", "preserve_fitting_migration_chain",
    "forward_publication_fit_dependency_issues",
]
