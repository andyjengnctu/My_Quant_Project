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
from filters.breakout_quality.model_source_provenance import canonical_source_dataset_artifacts
from filters.breakout_quality.paths import SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
from filters.breakout_quality.predicted_context_artifact import maybe_predicted_context_artifact_spec


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


def verified_published_context_fit_proof(
    *, manifest_path: Path, fold_manifest: Mapping[str, Any], bundle: Any,
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
    if fold_manifest.get("migration") is not None:
        return False, "migrated Rolling checkpoint lacks original context binding"
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
    return True, "verified published Rolling context bytes and PIT fold SHA"


__all__ = [
    "FitDataDependencies", "context_source_for_profile", "fold_source_dependency_issues",
    "verified_published_context_fit_proof",
    "forward_publication_fit_dependency_issues",
]
