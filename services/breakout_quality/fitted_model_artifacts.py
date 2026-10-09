"""Canonical fitted-model lifecycle for Continuous DL research.

A fitted checkpoint and its fitting identity are a different artifact layer from
Forward/Rolling scores and reports.  Report/schema refreshes must never imply a
new fit when this contract is still compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

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
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.model_source_provenance import (
    canonical_source_dataset_artifacts,
    forward_published_source_provenance_issues,
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
                if key in fitted_source and fitted_source[key] != expected:
                    issues.append(f"fitted-model {key} mismatch")
    return tuple(dict.fromkeys(issues))


def _historical_fitted_source_compatible(
    *,
    persisted_identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    model_dir: Path,
    final_manifest_path: Path,
    report_path: Path,
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
    if not any(k not in old_source for k in FITTING_SOURCE_LINEAGE_KEYS):
        return False
    if {k: v for k, v in persisted.items() if k != "source"} != {
        k: v for k, v in expected.items() if k != "source"
    }:
        return False
    if dict(old_source) != {k: v for k, v in new_source.items() if k not in FITTING_SOURCE_LINEAGE_KEYS}:
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
    )


def _legacy_identity_issues(
    *,
    expected_identity: Mapping[str, Any],
    model_path: Path,
    final_manifest_path: Path,
    report_path: Path,
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
    "FITTED_MODEL_CONTRACT_VERSION",
    "FITTED_MODEL_MANIFEST_FILENAME",
    "FittedModelConflictError",
    "FittedModelTrainingEvidence",
    "FittedModelContract",
    "build_daily_forward_fitting_identity",
    "canonical_fitting_settings",
    "current_default_fitting_settings",
    "fitted_model_settings_issues",
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
