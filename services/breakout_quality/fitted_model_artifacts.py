"""Canonical fitted-model lifecycle for Continuous DL research.

A fitted checkpoint and its fitting identity are a different artifact layer from
Forward/Rolling scores and reports.  Report/schema refreshes must never imply a
new fit when this contract is still compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from core.breakout_quality_registry import TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
)
from core.breakout_quality_policy import get_breakout_quality_rolling_test_mode
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
    resolve_filter_point_in_time_dir,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.model_source_provenance import (
    canonical_source_dataset_artifacts,
    forward_published_source_provenance_issues,
    source_inventory_matches_current,
)
from filters.breakout_quality.models.factory import build_model, count_trainable_parameters


FITTED_MODEL_MANIFEST_FILENAME = "fitted_model_manifest.json"
FITTED_MODEL_CONTRACT_VERSION = 1

# Dataset / target lineage that can change fitted weights, not Forward evaluation
# horizon or report layout.  Historical sidecars without these fields are checked
# against BOTH original Forward publication records before a one-time upgrade.
FITTING_SOURCE_LINEAGE_KEYS = (
    "source_data_inventory",
    "source_dataset_artifacts",
    "source_target_artifacts",
)

# Immutable compatibility contract for Daily Universal artifacts produced before
# fitted_model_manifest.json existed.  These values describe historical bytes;
# they must never follow current config changes.
LEGACY_DAILY_FORWARD_FITTING_SETTINGS_V1 = {
    "max_epochs": 200,
    "batch_size": 128,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "use_inner_validation": True,
    "inner_validation_months": 24,
    "early_stopping_patience": 1,
    "early_stopping_min_delta": 0.0,
}

FITTING_SETTING_KEYS = tuple(LEGACY_DAILY_FORWARD_FITTING_SETTINGS_V1)
PIT_PERSISTED_FITTING_SETTING_KEYS = (
    "max_epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "early_stopping_patience",
    "early_stopping_min_delta",
)


class FittedModelConflictError(RuntimeError):
    """Raised when the same declared fitting identity points at incompatible bytes."""


@dataclass(frozen=True)
class FittedModelContract:
    manifest_path: Path
    model_path: Path
    manifest: dict[str, Any]
    fitting_identity: dict[str, Any]
    selected_epoch: int
    epoch_selection: dict[str, Any]
    final_refit_history: list[Any]
    source: str


@dataclass(frozen=True)
class FittedModelTrainingEvidence:
    """Persisted weight-affecting evidence for cross-evaluation checkpoint reuse."""

    training_settings: dict[str, Any]
    torch_execution: dict[str, Any]
    selected_epoch: int
    model_record: dict[str, Any] | None
    source: str




def _canonical_fold_payload_fingerprint(payload: Mapping[str, Any]) -> str:
    """Exact pre-refactor cache serialization: do not change historical FitKey bytes."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fold_training_contract_is_compatible(
    candidate: dict[str, Any], *, expected_contract: dict[str, Any]
) -> bool:
    """Check model-fitting identity while intentionally ignoring score-only horizon differences."""

    exact_fields = (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "continuous_target_id",
        "training_label_scope",
        "training_sample_scope",
        "seed",
        "model_information_cutoff",
        "model_spec",
        "experiment_settings",
        "training_settings",
        "source_contract",
        "lookahead_contract",
    )
    if not all(candidate.get(field) == expected_contract.get(field) for field in exact_fields):
        return False

    candidate_planned = candidate.get("planned_periods")
    expected_planned = expected_contract.get("planned_periods")
    if not isinstance(candidate_planned, dict) or not isinstance(expected_planned, dict):
        return False
    # score_end only controls how far an already-fitted model is evaluated.  Any OOS / Rolling
    # folds therefore share one model whenever all fitting periods plus the common
    # score_start/information cutoff are identical; no calendar year is special-cased.
    for field in ("validation_start", "validation_end", "score_start", "history_start"):
        if candidate_planned.get(field) != expected_planned.get(field):
            return False

    for field in ("observed_periods", "group_counts", "event_row_counts"):
        candidate_values = candidate.get(field)
        expected_values = expected_contract.get(field)
        if not isinstance(candidate_values, dict) or not isinstance(expected_values, dict):
            return False
        for phase in ("inner_train", "validation", "final_refit"):
            if candidate_values.get(phase) != expected_values.get(phase):
                return False
    return True


def _fold_training_identity_payload(contract: dict[str, Any]) -> dict[str, Any]:
    """Canonical model-fitting identity; score-only horizon fields are excluded."""

    planned = dict(contract.get("planned_periods") or {})
    observed = dict(contract.get("observed_periods") or {})
    group_counts = dict(contract.get("group_counts") or {})
    event_row_counts = dict(contract.get("event_row_counts") or {})
    payload = {
        key: contract.get(key)
        for key in (
            "filter_id",
            "model_architecture",
            "experiment_profile",
            "continuous_target_id",
            "training_label_scope",
            "training_sample_scope",
            "seed",
            "model_information_cutoff",
            "model_spec",
            "experiment_settings",
            "training_settings",
            "source_contract",
            "lookahead_contract",
        )
    }
    payload["planned_periods"] = {
        key: planned.get(key)
        for key in ("validation_start", "validation_end", "score_start", "history_start")
    }
    payload["observed_periods"] = {
        phase: observed.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    payload["group_counts"] = {
        phase: group_counts.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    payload["event_row_counts"] = {
        phase: event_row_counts.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    return payload


def fold_training_identity(contract: dict[str, Any]) -> str:
    return _canonical_fold_payload_fingerprint(_fold_training_identity_payload(contract))


# These values control routing / throughput, not optimizer samples, gradients,
# loss, step order, or selected epochs.  The *resolved* torch execution is
# independently required to match before any weight reuse is approved.
_FIRST_FOLD_NON_FITTING_SETTINGS = frozenset({
    "evaluation_batch_size", "train_prefetch_batches", "train_prefetch_workers",
    "device", "mixed_precision", "mixed_precision_dtype",
    "deterministic_algorithms", "allow_tf32",
})


def _first_fold_fit_identity_issues(
    candidate: dict[str, Any], expected: dict[str, Any],
) -> tuple[str, ...]:
    """Explain a cross-mode fitting mismatch without weakening PIT's native reuse.

    Forward and Rolling can use different score throughput settings.  Only those
    enumerated options are ignored here; all other candidate fields, including
    optimizer settings and actual training cohorts, remain exact requirements.
    """
    fields = (
        "filter_id", "model_architecture", "experiment_profile",
        "continuous_target_id", "training_label_scope", "training_sample_scope",
        "seed", "model_information_cutoff", "model_spec", "experiment_settings",
        "source_contract", "lookahead_contract",
    )
    issues = [f"{field} mismatch" for field in fields
              if candidate.get(field) != expected.get(field)]
    left_settings = candidate.get("training_settings")
    right_settings = expected.get("training_settings")
    if not isinstance(left_settings, dict) or not isinstance(right_settings, dict):
        issues.append("training_settings missing")
    else:
        for field in sorted(set(left_settings) | set(right_settings)):
            if field not in _FIRST_FOLD_NON_FITTING_SETTINGS and (
                field not in left_settings or field not in right_settings
                or left_settings[field] != right_settings[field]
            ):
                issues.append(f"training_settings.{field} mismatch")
    left_planned = candidate.get("planned_periods")
    right_planned = expected.get("planned_periods")
    if not isinstance(left_planned, dict) or not isinstance(right_planned, dict):
        issues.append("planned_periods missing")
    else:
        for field in ("validation_start", "validation_end", "score_start", "history_start"):
            if left_planned.get(field) != right_planned.get(field):
                issues.append(f"planned_periods.{field} mismatch")
    for family in ("observed_periods", "group_counts", "event_row_counts"):
        left = candidate.get(family)
        right = expected.get(family)
        if not isinstance(left, dict) or not isinstance(right, dict):
            issues.append(f"{family} missing")
        else:
            for stage in ("inner_train", "validation", "final_refit"):
                if left.get(stage) != right.get(stage):
                    issues.append(f"{family}.{stage} mismatch")
    return tuple(issues)






def validate_rolling_fold_fitted_checkpoint(
    *, manifest_path: Path, manifest: Mapping[str, Any],
    expected: Mapping[str, Any], expected_execution: Mapping[str, Any],
    bundle, torch_module,
) -> tuple[FittedModelContract | None, str | None]:
    """Only FIT owner may approve Rolling checkpoint bytes for another evaluation mode.

    The PIT producer supplies sample membership and its native fold contract; this
    function owns fitting semantics, resolved execution and checkpoint integrity.
    A mismatched fit is ineligible; a corrupt declared fit fails closed.
    """
    manifest_path = Path(manifest_path).resolve()
    fold_id = str(manifest.get("fold_id") or manifest_path.parent.name)
    issues = _first_fold_fit_identity_issues(dict(manifest), dict(expected))
    if issues:
        return None, "; ".join(issues[:7])
    actual_execution = dict(manifest.get("torch_execution") or {})
    actual_execution.pop("requested_device", None)
    if actual_execution != dict(expected_execution):
        return None, "torch execution mismatch"
    checkpoint_path = manifest_path.parent / "model.pt"
    expected_checkpoint = dict(dict(manifest.get("artifacts") or {}).get("checkpoint") or {})
    if not expected_checkpoint or not checkpoint_path.is_file():
        raise RuntimeError(f"Rolling fitted checkpoint missing: {fold_id}")
    if build_file_manifest(checkpoint_path) != expected_checkpoint:
        raise RuntimeError(f"Rolling fitted checkpoint hash mismatch: {fold_id}")
    selected_epoch = int(manifest.get("selected_epoch", 0) or 0)
    if selected_epoch < 1 or not isinstance(manifest.get("epoch_selection"), dict):
        raise RuntimeError(f"Rolling fitted checkpoint has invalid epoch evidence: {fold_id}")
    checkpoint = torch_module.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Rolling fitted checkpoint has invalid content: {fold_id}")
    if (
        checkpoint.get("model_spec") != bundle.model_spec.as_manifest_payload()
        or checkpoint.get("experiment_settings") != bundle.profile.as_manifest_payload()
        or int(checkpoint.get("selected_epoch", 0) or 0) != selected_epoch
    ):
        raise RuntimeError(f"Rolling fitted checkpoint embedded identity mismatch: {fold_id}")
    embedded = checkpoint.get("fold_contract")
    migration = dict(manifest.get("migration") or {})
    if isinstance(embedded, dict):
        embedded_issues = _first_fold_fit_identity_issues(embedded, dict(expected))
        if embedded_issues:
            raise RuntimeError(
                f"Rolling checkpoint embedded fitting contract mismatch: {fold_id}: "
                + "; ".join(embedded_issues[:7])
            )
    elif not (
        migration.get("kind") == "forward_model_fitting_identity_import"
        and migration.get("training_contract_unchanged") is True
        and migration.get("source_checkpoint_sha_preserved") is True
    ):
        raise RuntimeError(f"Rolling checkpoint missing native fitting contract: {fold_id}")
    return FittedModelContract(
        manifest_path=manifest_path, model_path=checkpoint_path,
        manifest=dict(manifest), fitting_identity=dict(expected),
        selected_epoch=selected_epoch,
        epoch_selection=dict(manifest["epoch_selection"]),
        final_refit_history=list(manifest.get("final_refit_history") or []),
        source="rolling_first_fold_verified",
    ), None


def select_unique_fitted_contract(
    verified: list[FittedModelContract],
) -> FittedModelContract:
    """Reject conflicting checkpoint bytes/epochs claimed under the same fit."""
    if not verified:
        raise ValueError("verified fitted-model candidate list is empty")
    hashes = {build_file_manifest(item.model_path)["sha256"] for item in verified}
    if len(hashes) != 1 or len({item.selected_epoch for item in verified}) != 1:
        raise RuntimeError("one Forward fitting identity has conflicting Rolling checkpoints")
    return verified[0]



@dataclass(frozen=True)
class FittedCandidateProbe:
    """Status-only lookup; never authorizes REUSE without full FIT validation."""
    forward_fit_candidate: bool
    rolling_first_fold_candidate: bool


def resolve_rolling_fitted_folds_root(
    *, project_root: Path, filter_id: str, model_architecture: str,
    experiment_profile: str,
) -> Path:
    """Single canonical locator for Rolling fitted folds across all consumers."""
    mode = get_breakout_quality_rolling_test_mode("rolling")
    if mode.point_in_time_dirname not in (None, ""):
        pit_root = resolve_filter_model_output_dir(
            project_root, filter_id, model_architecture, experiment_profile,
        ) / str(mode.point_in_time_dirname)
    else:
        pit_root = resolve_filter_point_in_time_dir(
            project_root, filter_id, model_architecture, experiment_profile,
        )
    return pit_root / "folds"


def probe_fitted_candidates(
    *, project_root: Path, filter_id: str, model_architecture: str,
    experiment_profile: str, report_filename: str,
    expected_settings: Mapping[str, Any] | None = None,
) -> FittedCandidateProbe:
    """Locate provisional FIT inputs, not full scientific readiness.

    The expensive member-by-member proof lives in the canonical FIT validator;
    a status screen may only say VERIFY FIT, never REUSE, from this probe.
    """
    paths = resolve_filter_artifact_paths(
        project_root, filter_id, model_architecture, experiment_profile,
    )
    report_path = resolve_filter_model_output_dir(
        project_root, filter_id, model_architecture, experiment_profile,
    ) / report_filename
    forward_candidate = bool(
        paths.model_path.is_file()
        and not fitted_model_settings_issues(
            model_dir=paths.model_dir, report_path=report_path,
            expected=expected_settings,
        )
    )
    mode = get_breakout_quality_rolling_test_mode("rolling")
    start = str(mode.score_start_date or "")
    rolling_candidate = False
    if len(start) == 10 and start[4] == "-" and start[7] == "-":
        folds = resolve_rolling_fitted_folds_root(
            project_root=project_root, filter_id=filter_id,
            model_architecture=model_architecture, experiment_profile=experiment_profile,
        )
        pattern = "fold_" + start.replace("-", "") + "_*"
        rolling_candidate = folds.is_dir() and any(
            (candidate / "model.pt").is_file() for candidate in folds.glob(pattern)
        )
    return FittedCandidateProbe(forward_candidate, rolling_candidate)

def _forward_checkpoint_import_issues(
    *,
    source_manifest: dict[str, Any],
    source_report: dict[str, Any],
    source_fitting_evidence,
    checkpoint: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    fold_contract: dict[str, Any],
    bundle,
    execution_plan,
    project_root: Path | None = None,
) -> tuple[str, ...]:
    """Validate that a canonical Forward model is the exact fit required by this PIT fold.

    Forward and Rolling remain separate evaluation identities.  This bridge only imports the
    fitted model bytes; score rows are always regenerated for the Rolling fold.  The current
    bridge is intentionally limited to daily-universal stock-day training, whose Forward split
    report fully pins the deterministic group membership used by the first Rolling fold.
    """

    issues: list[str] = []
    if str(fold_contract.get("training_sample_scope") or "") != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        issues.append("checkpoint import目前只支援daily-universal stock-day fitting identity")

    for field in (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "continuous_target_id",
        "training_label_scope",
        "model_information_cutoff",
        "model_spec",
        "experiment_settings",
    ):
        if source_manifest.get(field) != fold_contract.get(field):
            issues.append(f"manifest {field} mismatch")

    source_profile_settings = dict(source_manifest.get("experiment_settings") or {})
    manifest_sample_scope = str(
        source_manifest.get("training_sample_scope")
        or source_profile_settings.get("training_sample_scope")
        or ""
    )
    if manifest_sample_scope != str(fold_contract.get("training_sample_scope") or ""):
        issues.append("manifest training_sample_scope mismatch")

    training = dict(source_report.get("training") or {})
    if int(training.get("seed", -1)) != int(fold_contract.get("seed", -2)):
        issues.append("report seed mismatch")

    # Weight-affecting optimizer/epoch-selection settings belong to the fitted-model
    # lifecycle contract, not to the evaluation report schema.  Forward reports may
    # legitimately omit these fields, so cross-evaluation checkpoint reuse must read
    # the canonical fitted-model sidecar (or immutable legacy defaults) instead.
    training_settings = dict(fold_contract.get("training_settings") or {})
    persisted_settings = dict(getattr(source_fitting_evidence, "training_settings", {}) or {})
    fitting_setting_pairs = (
        ("max_epochs", "epochs_max", int),
        ("batch_size", "batch_size", int),
        ("learning_rate", "learning_rate", float),
        ("weight_decay", "weight_decay", float),
        ("gradient_clip_norm", "gradient_clip_norm", float),
        ("early_stopping_patience", "early_stopping_patience", int),
        ("early_stopping_min_delta", "early_stopping_min_delta", float),
    )
    for persisted_field, fold_field, caster in fitting_setting_pairs:
        try:
            actual = caster(persisted_settings.get(persisted_field))
            expected = caster(training_settings.get(fold_field))
        except (TypeError, ValueError):
            issues.append(f"fitted-model training {persisted_field} missing")
            continue
        if actual != expected:
            issues.append(f"fitted-model training {persisted_field} mismatch")
    if persisted_settings.get("use_inner_validation") is not True:
        issues.append("fitted-model use_inner_validation mismatch")

    expected_execution = scientific_torch_execution(execution_plan)
    persisted_execution = dict(getattr(source_fitting_evidence, "torch_execution", {}) or {})
    if persisted_execution != expected_execution:
        issues.append("fitted-model torch_execution mismatch")
    # ``fitted_model_manifest.json`` is the fitting-execution SSOT.  The final
    # Forward ``manifest.json`` is an evaluation/publication manifest and the
    # canonical producer does not persist ``torch_execution`` there.  Requiring
    # that non-canonical field made every real Forward -> first Rolling import
    # fail even though the fitted sidecar held the exact resolved execution.
    report_execution = dict(source_report.get("torch_execution") or {})
    report_execution.pop("requested_device", None)
    if report_execution and report_execution != expected_execution:
        issues.append("report torch_execution mismatch")

    manifest_selected_epoch = int(source_manifest.get("selected_epoch", 0) or 0)
    if manifest_selected_epoch < 1:
        issues.append("manifest selected_epoch missing")
    if int(training.get("selected_epoch", 0) or 0) != manifest_selected_epoch:
        issues.append("report/manifest selected_epoch mismatch")
    if int(getattr(source_fitting_evidence, "selected_epoch", 0) or 0) != manifest_selected_epoch:
        issues.append("fitted-model/manifest selected_epoch mismatch")
    persisted_model_record = getattr(source_fitting_evidence, "model_record", None)
    if persisted_model_record is not None and dict(persisted_model_record) != checkpoint_manifest:
        issues.append("fitted-model checkpoint hash/size mismatch")

    standard = dict(source_report.get("standard_model_sop") or {})
    if str(standard.get("evaluation_mode") or "") != "forward_oos":
        issues.append("source report is not canonical forward_oos evidence")

    planned = dict(fold_contract.get("planned_periods") or {})
    split_report = dict(source_report.get("split_report") or {})
    expected_split_values = {
        "selection_start_date": dict(fold_contract.get("source_contract") or {}).get("training_universe_start_date"),
        "inner_validation_start_date": planned.get("validation_start"),
        "selection_end_date": planned.get("validation_end"),
        "oos_start_date": planned.get("score_start"),
    }
    for field, expected in expected_split_values.items():
        if expected is not None and str(split_report.get(field) or "") != str(expected):
            issues.append(f"split_report {field} mismatch")

    counts = dict(split_report.get("counts") or {})
    expected_counts = dict(fold_contract.get("group_counts") or {})
    for source_key, fold_key in (
        ("inner_train", "inner_train"),
        ("validation", "validation"),
        ("selection", "final_refit"),
    ):
        if int(counts.get(source_key, -1)) != int(expected_counts.get(fold_key, -2)):
            issues.append(f"split_report {source_key} count mismatch")

    source_dataset = dict(source_manifest.get("source_dataset") or {})
    source_target = dict(source_manifest.get("source_continuous_target") or {})
    expected_source = dict(fold_contract.get("source_contract") or {})
    source_pairs = (
        ("dataset_policy", source_dataset.get("policy")),
        ("dataset_storage_schema_version", source_dataset.get("dataset_storage_schema_version")),
        ("source_data_inventory", source_dataset.get("source_data_inventory")),
        ("input_price_source", source_dataset.get("input_price_source")),
        ("input_trade_count_source", source_dataset.get("input_trade_count_source")),
        ("dataset_artifacts", source_dataset.get("source_dataset_artifacts", source_dataset.get("dataset_artifacts"))),
        ("target_schema_version", source_target.get("schema_version")),
        ("target_contract", source_target.get("target_contract")),
        ("target_artifacts", source_target.get("artifacts")),
        ("training_universe_start_date", source_dataset.get("training_universe_start_date")),
    )
    for field, actual in source_pairs:
        if field == "source_data_inventory" and field in expected_source:
            matched = source_inventory_matches_current(
                published=actual, current=expected_source.get(field),
                project_root=(Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]),
            )
        else:
            matched = actual == expected_source.get(field)
        if field in expected_source and not matched:
            issues.append(f"source_contract {field} mismatch")

    if checkpoint.get("model_spec") != fold_contract.get("model_spec"):
        issues.append("checkpoint model_spec mismatch")
    if checkpoint.get("experiment_settings") != fold_contract.get("experiment_settings"):
        issues.append("checkpoint experiment_settings mismatch")
    if str(checkpoint.get("experiment_profile") or "") != str(fold_contract.get("experiment_profile") or ""):
        issues.append("checkpoint experiment_profile mismatch")
    checkpoint_profile_settings = dict(checkpoint.get("experiment_settings") or {})
    checkpoint_sample_scope = str(
        checkpoint.get("training_sample_scope")
        or checkpoint_profile_settings.get("training_sample_scope")
        or ""
    )
    if checkpoint_sample_scope != str(fold_contract.get("training_sample_scope") or ""):
        issues.append("checkpoint training_sample_scope mismatch")

    expected_shape = (
        int(bundle.feature_bank.shape[1]),
        int(bundle.feature_bank.shape[2]),
        int(bundle.group_context.shape[1]),
    )
    checkpoint_shape = tuple(
        int(checkpoint.get(field)) if checkpoint.get(field) is not None else -1
        for field in ("sequence_length", "feature_count", "context_count")
    )
    if checkpoint_shape != expected_shape:
        issues.append("checkpoint input shape mismatch")

    source_model_manifest = source_manifest.get("model")
    report_model_manifest = dict(source_report.get("artifacts") or {}).get("model")
    if source_model_manifest != checkpoint_manifest:
        issues.append("source manifest checkpoint hash/size mismatch")
    if report_model_manifest is not None and report_model_manifest != checkpoint_manifest:
        issues.append("source report checkpoint hash/size mismatch")
    return tuple(dict.fromkeys(issues))




def _checkpoint_matches_rescore_contract(
    *,
    checkpoint: dict[str, Any],
    manifest: dict[str, Any],
    fold_contract: dict[str, Any],
    bundle,
) -> bool:
    """Validate PIT-native or imported-Forward checkpoint metadata before score regeneration."""

    migration = dict(manifest.get("migration") or {})
    imported_forward = (
        str(migration.get("kind") or "") == "forward_model_fitting_identity_import"
        and migration.get("training_contract_unchanged") is True
        and migration.get("source_checkpoint_sha_preserved") is True
    )
    checkpoint_contract = checkpoint.get("fold_contract")
    if isinstance(checkpoint_contract, dict):
        if not _fold_training_contract_is_compatible(
            checkpoint_contract, expected_contract=fold_contract
        ):
            return False
    elif not imported_forward:
        return False

    manifest_epoch = int(manifest.get("selected_epoch") or 0)
    if manifest_epoch < 1:
        return False
    if not imported_forward:
        checkpoint_epoch = int(checkpoint.get("selected_epoch") or 0)
        if checkpoint_epoch != manifest_epoch:
            return False

    if checkpoint.get("model_spec") != fold_contract.get("model_spec"):
        return False
    if imported_forward:
        if checkpoint.get("experiment_settings") != fold_contract.get("experiment_settings"):
            return False
        if str(checkpoint.get("experiment_profile") or "") != str(fold_contract.get("experiment_profile") or ""):
            return False
        checkpoint_profile_settings = dict(checkpoint.get("experiment_settings") or {})
        checkpoint_sample_scope = str(
            checkpoint.get("training_sample_scope")
            or checkpoint_profile_settings.get("training_sample_scope")
            or ""
        )
        if checkpoint_sample_scope != str(fold_contract.get("training_sample_scope") or ""):
            return False

    expected_shape = (
        int(bundle.feature_bank.shape[1]),
        int(bundle.feature_bank.shape[2]),
        int(bundle.group_context.shape[1]),
    )
    checkpoint_shape = tuple(
        int(checkpoint.get(field)) if checkpoint.get(field) is not None else -1
        for field in ("sequence_length", "feature_count", "context_count")
    )
    return checkpoint_shape == expected_shape


def _fitting_settings_payload(
    *,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    use_inner_validation: bool,
    inner_validation_months: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
) -> dict[str, Any]:
    return {
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip_norm": float(gradient_clip_norm),
        "use_inner_validation": bool(use_inner_validation),
        "inner_validation_months": int(inner_validation_months),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
    }


def canonical_fitting_settings(args) -> dict[str, Any]:
    """Return only settings that can change the fitted weights/epoch selection."""

    return _fitting_settings_payload(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        use_inner_validation=args.use_inner_validation,
        inner_validation_months=args.inner_validation_months,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


def current_default_fitting_settings() -> dict[str, Any]:
    """Current formal-menu fitting settings from config, through one canonical owner."""

    return _fitting_settings_payload(
        max_epochs=BREAKOUT_QUALITY_DEFAULT_EPOCHS,
        batch_size=BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
        learning_rate=BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
        weight_decay=BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
        gradient_clip_norm=BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
        use_inner_validation=BREAKOUT_QUALITY_USE_INNER_VALIDATION,
        inner_validation_months=BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
        early_stopping_patience=BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    )


def legacy_default_fitting_settings() -> dict[str, Any]:
    """Immutable historical settings for pre-sidecar Daily Universal artifacts."""

    return dict(LEGACY_DAILY_FORWARD_FITTING_SETTINGS_V1)


def _canonicalize_persisted_fitting_settings(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(payload or {})
    if not raw:
        return {}
    normalized = {
        "max_epochs": raw.get("max_epochs", raw.get("epochs_max")),
        "batch_size": raw.get("batch_size"),
        "learning_rate": raw.get("learning_rate", raw.get("lr")),
        "weight_decay": raw.get("weight_decay"),
        "gradient_clip_norm": raw.get("gradient_clip_norm"),
        "use_inner_validation": raw.get("use_inner_validation"),
        "inner_validation_months": raw.get("inner_validation_months"),
        "early_stopping_patience": raw.get("early_stopping_patience"),
        "early_stopping_min_delta": raw.get("early_stopping_min_delta"),
    }
    return {key: value for key, value in normalized.items() if value is not None}


def fitting_settings_issues(
    persisted: Mapping[str, Any] | None,
    *,
    expected: Mapping[str, Any] | None = None,
    require_all: bool = True,
) -> tuple[str, ...]:
    """Compare persisted fitting knobs to current settings without re-deriving training logic."""

    actual = _canonicalize_persisted_fitting_settings(persisted)
    target = dict(current_default_fitting_settings() if expected is None else expected)
    issues: list[str] = []
    for key in FITTING_SETTING_KEYS:
        if key not in actual:
            if require_all:
                issues.append(f"{key} missing")
            continue
        if actual[key] != target.get(key):
            issues.append(f"{key} mismatch: artifact={actual[key]!r}, current={target.get(key)!r}")
    return tuple(issues)


def load_fitted_model_training_evidence(
    *,
    model_dir: Path,
    report_path: Path | None = None,
) -> tuple[FittedModelTrainingEvidence | None, str | None]:
    """Load canonical fitting evidence without requiring evaluation/report completeness.

    New artifacts read the fitted-model sidecar.  Pre-sidecar Daily Universal artifacts
    may use the immutable historical defaults, but only after the caller independently
    validates the legacy manifest/report scientific identity.
    """

    model_dir = Path(model_dir).resolve()
    sidecar_path = model_dir / FITTED_MODEL_MANIFEST_FILENAME
    sidecar = _read_json(sidecar_path)
    if sidecar_path.exists():
        if sidecar is None:
            return None, "fitted-model sidecar invalid"
        if int(sidecar.get("contract_version", -1)) != FITTED_MODEL_CONTRACT_VERSION:
            return None, "fitted-model contract version mismatch"
        identity = dict(sidecar.get("fitting_identity") or {})
        settings = _canonicalize_persisted_fitting_settings(identity.get("training_settings"))
        if not settings:
            return None, "fitted-model sidecar training_settings missing"
        result = dict(sidecar.get("fitting_result") or {})
        selected_epoch = int(result.get("selected_epoch", 0) or 0)
        if selected_epoch < 1:
            return None, "fitted-model sidecar selected_epoch missing"
        execution = dict(identity.get("torch_execution") or {})
        execution.pop("requested_device", None)
        return FittedModelTrainingEvidence(
            training_settings=settings,
            torch_execution=execution,
            selected_epoch=selected_epoch,
            model_record=(
                dict(sidecar.get("model") or {})
                if isinstance(sidecar.get("model"), Mapping)
                else None
            ),
            source="fitted_model_manifest",
        ), None

    report = _read_json(Path(report_path)) if report_path is not None else None
    if report is None:
        return None, "legacy forward report missing or invalid"
    training = dict(report.get("training") or {})
    persisted = training.get("fitting_settings")
    settings = (
        _canonicalize_persisted_fitting_settings(persisted)
        if persisted is not None
        else legacy_default_fitting_settings()
    )
    selected_epoch = int(training.get("selected_epoch", 0) or 0)
    if selected_epoch < 1:
        return None, "legacy forward report selected_epoch missing"
    execution = dict(report.get("torch_execution") or {})
    execution.pop("requested_device", None)
    return FittedModelTrainingEvidence(
        training_settings=settings,
        torch_execution=execution,
        selected_epoch=selected_epoch,
        model_record=None,
        source=(
            "legacy_report_fitting_settings"
            if persisted is not None
            else "legacy_daily_forward_defaults"
        ),
    ), None


def fitted_model_settings_issues(
    *,
    model_dir: Path,
    report_path: Path | None = None,
    expected: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Cheap FIT-readiness probe used before accepting a complete Forward report.

    Full fitting identity remains owned by load_reusable_fitted_model_contract().
    This preflight only prevents a complete evaluation/report artifact from hiding
    a changed weight-affecting setting and bypassing the trainer entirely.
    """

    model_dir = Path(model_dir).resolve()
    model_path = model_dir / "model.pt"
    if not model_path.is_file():
        return ("model.pt missing",)
    target = dict(current_default_fitting_settings() if expected is None else expected)
    sidecar = _read_json(model_dir / FITTED_MODEL_MANIFEST_FILENAME)
    if sidecar is not None:
        persisted = dict(sidecar.get("fitting_identity") or {}).get("training_settings")
        return fitting_settings_issues(persisted, expected=target, require_all=True)

    report = _read_json(Path(report_path)) if report_path is not None else None
    persisted = dict((report or {}).get("training") or {}).get("fitting_settings")
    if persisted is not None:
        return fitting_settings_issues(persisted, expected=target, require_all=True)
    if target != legacy_default_fitting_settings():
        return (
            "legacy artifact lacks fitting settings and current settings differ from immutable historical defaults",
        )
    return ()


def pit_fold_fitting_settings_issues(
    *,
    point_in_time_dir: Path,
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate persisted Rolling fold fitting knobs against current config.

    PIT fold manifests predate the Forward sidecar and use epochs_max.  Their
    split/time identity is still validated by the PIT producer; this probe exists
    so [2]/[4]/[6] cannot report REUSE when a weight-affecting config knob changed.
    """

    target = dict(current_default_fitting_settings() if expected is None else expected)
    issues: list[str] = []
    top_level_months = manifest.get("inner_validation_months")
    if top_level_months is not None and int(top_level_months) != int(target["inner_validation_months"]):
        issues.append(
            f"inner_validation_months mismatch: artifact={int(top_level_months)!r}, "
            f"current={int(target['inner_validation_months'])!r}"
        )
    for fold in manifest.get("folds", []):
        if not isinstance(fold, Mapping):
            issues.append("fold record invalid")
            continue
        fold_id = str(fold.get("fold_id") or "").strip()
        if not fold_id:
            issues.append("fold_id missing")
            continue
        fold_manifest = _read_json(Path(point_in_time_dir) / "folds" / fold_id / "manifest.json")
        if fold_manifest is None:
            issues.append(f"{fold_id} manifest missing or invalid")
            continue
        persisted_settings = _canonicalize_persisted_fitting_settings(
            fold_manifest.get("training_settings")
        )
        missing = [
            key for key in PIT_PERSISTED_FITTING_SETTING_KEYS
            if key not in persisted_settings
        ]
        issues.extend(f"{fold_id} {key} missing" for key in missing)
        fold_issues = fitting_settings_issues(
            persisted_settings,
            expected=target,
            require_all=False,
        )
        issues.extend(f"{fold_id} {issue}" for issue in fold_issues)
    return tuple(dict.fromkeys(issues))


def scientific_torch_execution(plan) -> dict[str, Any]:
    payload = dict(plan.as_manifest_payload())
    # requested_device is routing metadata.  The resolved execution is what can
    # affect deterministic fitted weights.
    payload.pop("requested_device", None)
    return payload


def fitting_split_identity(split_report: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(split_report or {})
    counts = dict(report.get("counts") or {})
    return {
        "sample_scope": report.get("sample_scope"),
        "selection_start_date": report.get("selection_start_date"),
        "selection_end_date": report.get("selection_end_date"),
        "inner_validation_start_date": report.get("inner_validation_start_date"),
        "oos_start_date": report.get("oos_start_date"),
        "counts": {
            "inner_train": counts.get("inner_train"),
            "validation": counts.get("validation"),
            "selection": counts.get("selection"),
        },
        "no_lookahead": report.get("no_lookahead"),
        "selection_label_end_before_oos": report.get("selection_label_end_before_oos"),
        "inner_train_label_end_before_validation": report.get("inner_train_label_end_before_validation"),
        "target_valid_filter_applied": report.get("target_valid_filter_applied"),
    }


def source_training_identity(bundle) -> dict[str, Any]:
    summary = dict(bundle.summary or {})
    target = dict(bundle.target_manifest or {})
    # Do not bind the fit to Forward score horizon/tail metadata. Only source
    # and target bytes/semantics that can affect Selection fitting belong here.
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


def build_daily_forward_fitting_identity(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    seed: int,
    bundle,
    split_report: Mapping[str, Any],
    training_semantics: Mapping[str, Any],
    args,
    plan,
) -> dict[str, Any]:
    return {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
        "experiment_settings": bundle.profile.as_manifest_payload(),
        "model_spec": bundle.model_spec.as_manifest_payload(),
        "training_objective": str(bundle.profile.training_objective),
        "training_label_scope": str(bundle.profile.training_label_scope),
        "training_sample_scope": str(bundle.profile.training_sample_scope),
        "training_semantics": dict(training_semantics or {}),
        "continuous_target_id": str(bundle.profile.continuous_target_id or ""),
        "seed": int(seed),
        "training_settings": canonical_fitting_settings(args),
        "torch_execution": scientific_torch_execution(plan),
        "split": fitting_split_identity(split_report),
        "source": source_training_identity(bundle),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _file_record_matches(record: Mapping[str, Any] | None, path: Path) -> bool:
    if not isinstance(record, Mapping) or not Path(path).is_file():
        return False
    return dict(record) == build_file_manifest(Path(path))


def forward_fitted_source_provenance_issues(
    *,
    model_dir: Path,
    model_manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    target_manifest: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Check current Dataset/Target lineage against BOTH Forward publications.

    This is a fast manifest-identity check, not a Dataset readiness replacement:
    callers MUST first use canonical upstream readiness (including current CSV
    inventory / artifact metadata and Target hash validation). The canonical
    training loader performs full Dataset SHA verification before checkpoint reuse.
    """

    expected_inventory = dataset_summary.get("source_data_inventory")
    expected_artifacts = dataset_summary.get("dataset_artifacts")
    if not isinstance(expected_inventory, Mapping) or not isinstance(expected_artifacts, Mapping):
        return ("current Dataset lacks canonical source inventory/artifact hashes",)

    issues: list[str] = list(forward_published_source_provenance_issues(
        model_manifest=model_manifest,
        report=report,
        dataset_summary=dataset_summary,
        target_manifest=target_manifest,
        project_root=project_root,
    ))
    expected_target_records = (
        target_manifest.get("artifacts") if isinstance(target_manifest, Mapping) else None
    )
    sidecar_path = Path(model_dir) / FITTED_MODEL_MANIFEST_FILENAME
    if sidecar_path.exists():
        sidecar = _read_json(sidecar_path)
        if sidecar is None or int(sidecar.get("contract_version", -1)) != FITTED_MODEL_CONTRACT_VERSION:
            issues.append("fitted-model sidecar missing valid contract")
        elif not _file_record_matches(sidecar.get("model"), Path(model_dir) / "model.pt"):
            issues.append("fitted-model sidecar checkpoint hash mismatch")
        else:
            fitted_source = dict(dict(sidecar.get("fitting_identity") or {}).get("source") or {})
            # Older sidecars did not persist these keys; their original manifest
            # AND report above must still prove current source bytes. Present
            # keys cannot be ignored when inconsistent.
            for key, expected in (
                ("source_data_inventory", expected_inventory),
                ("source_dataset_artifacts", expected_artifacts),
                ("source_target_artifacts", expected_target_records),
            ):
                if key == "source_data_inventory" and key in fitted_source:
                    matches = source_inventory_matches_current(
                        published=fitted_source[key], current=expected,
                        project_root=project_root,
                    )
                else:
                    matches = fitted_source.get(key) == expected
                if key in fitted_source and not matches:
                    issues.append(f"fitted-model {key} mismatch")
            for source_key in ("input_price_source", "input_trade_count_source"):
                if dataset_summary.get(source_key) is not None and (
                    fitted_source.get(source_key) != dataset_summary[source_key]
                ):
                    issues.append(f"fitted-model {source_key} mismatch")
    if any(dataset_summary.get(key) is not None for key in ("input_price_source", "input_trade_count_source")) and not sidecar_path.exists():
        issues.append("fitted-model pinned supplemental input sidecar missing")
    return tuple(dict.fromkeys(issues))


def _historical_fitted_source_compatible(
    *,
    persisted_identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    model_dir: Path,
    final_manifest_path: Path,
    report_path: Path,
    project_root: str | Path | None = None,
) -> bool:
    """Allow old sidecar bytes only when original publications prove new lineage.

    This does not infer source hashes from a model's shape or row counts.  The
    newly added source keys are the ONLY allowable fitting-identity difference.
    """

    persisted = dict(persisted_identity)
    expected = dict(expected_identity)
    old_source = persisted.get("source")
    new_source = expected.get("source")
    if not isinstance(old_source, Mapping) or not isinstance(new_source, Mapping):
        return False
    if any(not isinstance(new_source.get(k), Mapping) for k in FITTING_SOURCE_LINEAGE_KEYS[:2]):
        return False
    # Older sidecars can omit lineage fields or publish the archived V1 CSV
    # inventory. No other source/fitting difference may be grandfathered.
    missing_lineage = any(k not in old_source for k in FITTING_SOURCE_LINEAGE_KEYS)
    changed_inventory = old_source.get("source_data_inventory") != new_source.get("source_data_inventory")
    if not (missing_lineage or changed_inventory):
        return False
    if {k: v for k, v in persisted.items() if k != "source"} != {
        k: v for k, v in expected.items() if k != "source"
    }:
        return False
    if {k: v for k, v in old_source.items() if k not in FITTING_SOURCE_LINEAGE_KEYS} != {
        k: v for k, v in new_source.items() if k not in FITTING_SOURCE_LINEAGE_KEYS
    }:
        return False
    for key in FITTING_SOURCE_LINEAGE_KEYS:
        if key not in old_source:
            continue
        if key == "source_data_inventory":
            if not source_inventory_matches_current(
                published=old_source[key], current=new_source.get(key),
                project_root=project_root,
            ):
                return False
        elif old_source[key] != new_source.get(key):
            return False
    manifest = _read_json(final_manifest_path)
    report = _read_json(report_path)
    if manifest is None or report is None:
        return False
    current_summary = {
        "dataset": new_source.get("dataset"),
        "policy": new_source.get("dataset_policy"),
        "source_data_inventory": new_source.get("source_data_inventory"),
        "dataset_artifacts": new_source.get("source_dataset_artifacts"),
    }
    expected_target = (
        {
            "artifacts": new_source.get("source_target_artifacts"),
            "target_contract": new_source.get("target_contract"),
        }
        if new_source.get("source_target_artifacts") is not None else None
    )
    return not forward_fitted_source_provenance_issues(
        model_dir=model_dir,
        model_manifest=manifest,
        report=report,
        dataset_summary=current_summary,
        target_manifest=expected_target,
        project_root=project_root,
    )


def _legacy_identity_issues(
    *,
    expected_identity: Mapping[str, Any],
    model_path: Path,
    final_manifest_path: Path,
    report_path: Path,
    project_root: str | Path | None = None,
) -> tuple[str, ...]:
    manifest = _read_json(final_manifest_path)
    report = _read_json(report_path)
    if manifest is None or report is None:
        return ("legacy manifest/report missing or invalid",)

    issues: list[str] = []
    expected = dict(expected_identity)
    for key in (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "experiment_settings",
        "model_spec",
        "training_objective",
        "training_label_scope",
        "training_sample_scope",
        "continuous_target_id",
    ):
        actual = manifest.get(key)
        if key == "training_sample_scope" and actual in (None, ""):
            actual = dict(manifest.get("experiment_settings") or {}).get(key)
        if actual != expected.get(key):
            issues.append(f"manifest {key} mismatch")

    # The canonical semantics payload already lives in the expected identity;
    # compare directly because a proxy cannot safely reconstruct every profile field.
    if dict(manifest.get("training_semantics") or {}) != dict(expected.get("training_semantics") or {}):
        issues.append("manifest training_semantics mismatch")

    training = dict(report.get("training") or {})
    if int(training.get("seed", -1)) != int(expected.get("seed", -2)):
        issues.append("report seed mismatch")
    if int(manifest.get("selected_epoch", 0) or 0) < 1:
        issues.append("manifest selected_epoch missing")
    if int(training.get("selected_epoch", 0) or 0) != int(manifest.get("selected_epoch", 0) or 0):
        issues.append("report/manifest selected_epoch mismatch")

    persisted_settings = training.get("fitting_settings")
    if persisted_settings is None:
        if dict(expected.get("training_settings") or {}) != legacy_default_fitting_settings():
            issues.append("legacy artifact lacks fitting_settings and current settings are not historical defaults")
    elif dict(persisted_settings or {}) != dict(expected.get("training_settings") or {}):
        issues.append("report fitting_settings mismatch")

    manifest_execution = dict(manifest.get("torch_execution") or {})
    manifest_execution.pop("requested_device", None)
    report_execution = dict(report.get("torch_execution") or {})
    report_execution.pop("requested_device", None)
    if manifest_execution != dict(expected.get("torch_execution") or {}):
        issues.append("manifest torch_execution mismatch")
    if report_execution and report_execution != dict(expected.get("torch_execution") or {}):
        issues.append("report torch_execution mismatch")

    if fitting_split_identity(dict(report.get("split_report") or {})) != dict(expected.get("split") or {}):
        issues.append("report fitting split mismatch")

    source_manifest = dict(manifest.get("source_continuous_target") or {})
    expected_source = dict(expected.get("source") or {})
    if source_manifest.get("target_contract") != expected_source.get("target_contract"):
        issues.append("target contract mismatch")
    source_dataset = dict(manifest.get("source_dataset") or {})
    if source_dataset.get("policy") != expected_source.get("dataset_policy"):
        issues.append("dataset policy mismatch")
    if source_dataset.get("training_universe_start_date") != expected_source.get("training_universe_start_date"):
        issues.append("training universe start mismatch")
    if any(key in expected_source for key in FITTING_SOURCE_LINEAGE_KEYS):
        source_records = forward_fitted_source_provenance_issues(
            model_dir=model_path.parent,
            model_manifest=manifest,
            report=report,
            dataset_summary={
                "dataset": expected_source.get("dataset"),
                "policy": expected_source.get("dataset_policy"),
                "source_data_inventory": expected_source.get("source_data_inventory"),
                "dataset_artifacts": expected_source.get("source_dataset_artifacts"),
            },
            target_manifest=(
                {"artifacts": expected_source.get("source_target_artifacts"),
                 "target_contract": expected_source.get("target_contract")}
                if expected_source.get("source_target_artifacts") is not None else None
            ),
            project_root=project_root,
        )
        issues.extend(source_records)

    if not _file_record_matches(manifest.get("model"), model_path):
        issues.append("manifest model file identity mismatch")
    report_model = dict(report.get("artifacts") or {}).get("model")
    if report_model is not None and not _file_record_matches(report_model, model_path):
        issues.append("report model file identity mismatch")
    return tuple(dict.fromkeys(issues))


def load_reusable_fitted_model_contract(
    *,
    model_dir: Path,
    expected_identity: Mapping[str, Any],
    final_manifest_path: Path,
    report_path: Path,
    project_root: str | Path | None = None,
) -> tuple[FittedModelContract | None, str | None]:
    model_dir = Path(model_dir).resolve()
    model_path = model_dir / "model.pt"
    sidecar_path = model_dir / FITTED_MODEL_MANIFEST_FILENAME
    if not model_path.is_file():
        return None, "model.pt missing"

    sidecar = _read_json(sidecar_path)
    if sidecar_path.exists():
        if sidecar is None:
            return None, "fitted-model sidecar corrupt or invalid"
        if int(sidecar.get("contract_version", -1)) != FITTED_MODEL_CONTRACT_VERSION:
            return None, "fitted-model contract version mismatch"
        persisted_identity = dict(sidecar.get("fitting_identity") or {})
        if persisted_identity != dict(expected_identity) and not _historical_fitted_source_compatible(
            persisted_identity=persisted_identity,
            expected_identity=expected_identity,
            model_dir=model_dir,
            final_manifest_path=Path(final_manifest_path),
            report_path=Path(report_path),
            project_root=project_root,
        ):
            return None, "fitted-model fitting identity mismatch (source data/target lineage unproven)"
        if not _file_record_matches(sidecar.get("model"), model_path):
            raise FittedModelConflictError(
                "same fitted-model fitting identity points at incompatible checkpoint bytes"
            )
        result = dict(sidecar.get("fitting_result") or {})
        selected_epoch = int(result.get("selected_epoch", 0) or 0)
        if selected_epoch < 1:
            raise FittedModelConflictError("fitted-model contract is corrupt: selected_epoch missing")
        return FittedModelContract(
            manifest_path=sidecar_path,
            model_path=model_path,
            manifest=sidecar,
            fitting_identity=persisted_identity,
            selected_epoch=selected_epoch,
            epoch_selection=dict(result.get("epoch_selection") or {}),
            final_refit_history=list(result.get("final_refit_history") or []),
            source="fitted_model_manifest",
        ), None

    issues = _legacy_identity_issues(
        expected_identity=expected_identity,
        model_path=model_path,
        final_manifest_path=Path(final_manifest_path),
        report_path=Path(report_path),
        project_root=project_root,
    )
    if issues:
        return None, "; ".join(issues[:8])
    manifest = _read_json(final_manifest_path) or {}
    report = _read_json(report_path) or {}
    training = dict(report.get("training") or {})
    return FittedModelContract(
        manifest_path=Path(final_manifest_path),
        model_path=model_path,
        manifest=manifest,
        fitting_identity=dict(expected_identity),
        selected_epoch=int(manifest.get("selected_epoch", 0) or 0),
        epoch_selection=dict(training.get("epoch_selection") or {}),
        final_refit_history=list(training.get("final_refit_history") or []),
        source="legacy_manifest_report",
    ), None


def write_fitted_model_contract(
    *,
    model_dir: Path,
    fitting_identity: Mapping[str, Any],
    selected_epoch: int,
    epoch_selection: Mapping[str, Any],
    final_refit_history: list[Any],
    trainable_parameter_count: int,
    total_parameter_count: int,
) -> Path:
    model_dir = Path(model_dir).resolve()
    model_path = model_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"cannot publish fitted-model contract without checkpoint: {model_path}")
    payload = {
        "contract_version": FITTED_MODEL_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fitting_identity": dict(fitting_identity),
        "fitting_result": {
            "selected_epoch": int(selected_epoch),
            "epoch_selection": dict(epoch_selection or {}),
            "final_refit_history": list(final_refit_history or []),
            "trainable_parameter_count": int(trainable_parameter_count),
            "total_parameter_count": int(total_parameter_count),
        },
        "model": build_file_manifest(model_path),
    }
    path = model_dir / FITTED_MODEL_MANIFEST_FILENAME
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return path



def import_verified_rolling_forward_checkpoint(
    *, model_dir: Path, output_dir: Path, rolling_contract: FittedModelContract,
) -> tuple[Path | None, bool]:
    """Publish an independently verified Rolling FIT under the Forward location.

    If the existing Forward model differs, preserve the complete *published* local
    evidence byte-for-byte in a deterministic hash-addressed historical snapshot
    before replacing any bytes. Never rewrite an existing historical snapshot.
    This routine must be called only after the PIT fitting-identity verifier and
    model loader have both accepted the candidate.
    """
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    destination = model_dir / "model.pt"
    source = Path(rolling_contract.model_path).resolve()
    verified_record = build_file_manifest(source)
    original_record = dict(dict(rolling_contract.manifest.get("artifacts") or {}).get("checkpoint") or {})
    if not original_record or verified_record != original_record:
        raise FittedModelConflictError(
            "Rolling checkpoint changed after PIT fitting-identity verification"
        )
    def identical_bytes(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        # Manifest.filename denotes the storage location, not tensor content.
        # Staged files intentionally have a different basename.
        return (
            first.get("sha256") == second.get("sha256")
            and first.get("size_bytes") == second.get("size_bytes")
        )

    historical_dir = None
    if destination.is_file():
        current_record = build_file_manifest(destination)
        if identical_bytes(current_record, verified_record):
            return None, False
        legacy_sha = str(current_record["sha256"])
        archive = model_dir / "historical_forward_fits" / legacy_sha
        if archive.exists():
            evidence = _read_json(archive / "archive_manifest.json")
            if (evidence is None or evidence.get("original_model") != current_record
                    or evidence.get("replacement_rolling_model") != verified_record
                    or not (archive / "model" / "model.pt").is_file()
                    or build_file_manifest(archive / "model" / "model.pt") != current_record):
                raise FittedModelConflictError("historical Forward archive conflict: " + legacy_sha)
        else:
            archive.parent.mkdir(parents=True, exist_ok=True)
            stage = archive.parent / ("." + legacy_sha + f".stage_{time.time_ns()}")
            stage.mkdir(parents=True, exist_ok=False)
            try:
                preserved = []
                # Restrict to already published files; never traverse another
                # profile's directory, include a private dataset, or copy secrets.
                for bucket, origin in (("model", model_dir), ("report", output_dir)):
                    if not origin.is_dir():
                        continue
                    for candidate in sorted(origin.iterdir()):
                        if not candidate.is_file() or candidate.name.startswith("."):
                            continue
                        relative = Path(bucket) / candidate.name
                        target = stage / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        original = build_file_manifest(candidate)
                        shutil.copy2(candidate, target)
                        if build_file_manifest(target) != original:
                            raise RuntimeError("historical Forward archive copy verification failed")
                        preserved.append({"path": relative.as_posix(), "manifest": original})
                if not (stage / "model" / "model.pt").is_file():
                    raise RuntimeError("historical Forward checkpoint missing during archive")
                (stage / "archive_manifest.json").write_text(
                    json.dumps({
                        "archive_version": 1,
                        "original_model": current_record,
                        "replacement_rolling_model": verified_record,
                        "preserved_files": preserved,
                    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                stage.replace(archive)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        historical_dir = archive
    if destination.is_file() and identical_bytes(build_file_manifest(destination), verified_record):
        return historical_dir, False
    staged = destination.with_name("." + destination.name + f".rolling_import_{time.time_ns()}")
    try:
        shutil.copy2(source, staged)
        if not identical_bytes(build_file_manifest(staged), verified_record):
            raise RuntimeError("Rolling checkpoint changed during verified Forward import")
        staged.replace(destination)
    finally:
        if staged.exists():
            staged.unlink()
    if not identical_bytes(build_file_manifest(destination), verified_record):
        raise RuntimeError("Forward checkpoint import verification failed")
    return historical_dir, True





@dataclass(frozen=True)
class ForwardFittedResolution:
    """Single decision about fitted weights; independent of SCORE/EVIDENCE/REPORT."""
    contract: FittedModelContract | None
    preloaded_model: Any | None
    source: str
    reason: str | None
    archived_forward: Path | None = None


def resolve_forward_fitted_checkpoint(
    *, reuse_fitted: bool, model_dir: Path, output_dir: Path,
    final_manifest_path: Path, report_path: Path,
    expected_identity: Mapping[str, Any], project_root: Path,
    bundle, forward_split, forward_args, forward_plan, torch_module,
) -> ForwardFittedResolution:
    """FIT owner resolves all legitimate sources before any Forward trainer can build.

    The PIT producer is used only to reconstruct a first-fold *sample/cohort proof*;
    identity validation, checkpoint integrity, candidate conflicts and import are
    owned by this module. Forward never reuses the Rolling scores themselves.
    """
    if not reuse_fitted:
        return ForwardFittedResolution(None, None, "FIT_REQUIRED", "explicit new fit")
    local, local_reason = load_reusable_fitted_model_contract(
        model_dir=model_dir, expected_identity=expected_identity,
        final_manifest_path=final_manifest_path, report_path=report_path,
        project_root=project_root,
    )
    if local is not None:
        return ForwardFittedResolution(local, None, "FORWARD_FIT", None)

    # Late import prevents a circular dependency. PIT is the cohort producer,
    # not the source of fitted equivalence, which is validated above in FIT owner.
    from services.breakout_quality.point_in_time_scores import (
        load_forward_compatible_rolling_checkpoint,
    )
    rolling, rolling_reason = load_forward_compatible_rolling_checkpoint(
        bundle=bundle, forward_split=forward_split,
        forward_args=forward_args, forward_plan=forward_plan,
        torch_module=torch_module, project_root=project_root,
    )
    if rolling is None:
        reason = (
            f"Forward identity: {local_reason or 'unverified'}; "
            f"Rolling identity: {rolling_reason or 'unverified'}"
        )
        if (Path(model_dir) / "model.pt").exists():
            raise RuntimeError(
                "Existing Forward checkpoint is not reusable and no verified Rolling FIT "
                "can replace it. The legacy model has NOT been overwritten; "
                "automatic retraining during report refresh is blocked. " + reason
            )
        return ForwardFittedResolution(None, None, "FIT_REQUIRED", reason)
    loaded = load_model_from_fitted_checkpoint(
        torch_module, contract=rolling, bundle=bundle, plan=forward_plan,
    )
    archive, _copied = import_verified_rolling_forward_checkpoint(
        model_dir=model_dir, output_dir=output_dir, rolling_contract=rolling,
    )
    return ForwardFittedResolution(
        rolling, loaded, "ROLLING_FIT", None, archived_forward=archive,
    )

def load_model_from_fitted_checkpoint(torch, *, contract: FittedModelContract, bundle, plan):
    checkpoint = torch.load(contract.model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("fitted checkpoint root must be a mapping")
    if checkpoint.get("model_spec") != bundle.model_spec.as_manifest_payload():
        raise ValueError("fitted checkpoint model_spec mismatch")
    if checkpoint.get("experiment_settings") != bundle.profile.as_manifest_payload():
        raise ValueError("fitted checkpoint experiment_settings mismatch")
    if str(checkpoint.get("experiment_profile") or "") != str(bundle.profile.name):
        raise ValueError("fitted checkpoint experiment_profile mismatch")
    expected_shape = (
        int(bundle.feature_bank.shape[1]),
        int(bundle.feature_bank.shape[2]),
        int(bundle.group_context.shape[1]),
    )
    actual_shape = tuple(
        int(checkpoint[field]) if field in checkpoint and checkpoint[field] is not None else -1
        for field in ("sequence_length", "feature_count", "context_count")
    )
    if actual_shape != expected_shape:
        raise ValueError(f"fitted checkpoint input shape mismatch: expected={expected_shape}, actual={actual_shape}")
    if int(checkpoint.get("selected_epoch", 0) or 0) != int(contract.selected_epoch):
        raise ValueError("fitted checkpoint selected_epoch mismatch")

    model = build_model(
        feature_count=int(bundle.feature_bank.shape[2]),
        context_count=int(bundle.group_context.shape[1]),
        architecture=str(bundle.model_spec.architecture),
        model_spec=checkpoint.get("model_spec"),
    ).to(plan.device)
    model.load_state_dict(dict(checkpoint.get("model_state_dict") or {}), strict=True)
    model.eval()
    if count_trainable_parameters(model) != int(
        dict(contract.manifest.get("fitting_result") or {}).get("trainable_parameter_count", count_trainable_parameters(model))
    ) and contract.source == "fitted_model_manifest":
        raise ValueError("fitted checkpoint trainable parameter count mismatch")
    return model


__all__ = [
    "resolve_forward_fitted_checkpoint",
    "probe_fitted_candidates",
    "resolve_rolling_fitted_folds_root",
    "select_unique_fitted_contract",
    "validate_rolling_fold_fitted_checkpoint",
    "fold_training_identity",
    "FittedCandidateProbe",
    "ForwardFittedResolution",
    "FITTED_MODEL_CONTRACT_VERSION",
    "FITTED_MODEL_MANIFEST_FILENAME",
    "FittedModelConflictError",
    "FittedModelTrainingEvidence",
    "FittedModelContract",
    "build_daily_forward_fitting_identity",
    "canonical_fitting_settings",
    "current_default_fitting_settings",
    "fitted_model_settings_issues",
    "import_verified_rolling_forward_checkpoint",
    "forward_fitted_source_provenance_issues",
    "fitting_settings_issues",
    "legacy_default_fitting_settings",
    "load_fitted_model_training_evidence",
    "pit_fold_fitting_settings_issues",
    "load_model_from_fitted_checkpoint",
    "load_reusable_fitted_model_contract",
    "scientific_torch_execution",
    "write_fitted_model_contract",
]
