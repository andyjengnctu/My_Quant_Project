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
from filters.breakout_quality.models.factory import build_model, count_trainable_parameters


FITTED_MODEL_MANIFEST_FILENAME = "fitted_model_manifest.json"
FITTED_MODEL_CONTRACT_VERSION = 1


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


def canonical_fitting_settings(args) -> dict[str, Any]:
    """Return only settings that can change the fitted weights/epoch selection."""

    return {
        "max_epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "use_inner_validation": bool(args.use_inner_validation),
        "inner_validation_months": int(args.inner_validation_months),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
    }


def legacy_default_fitting_settings() -> dict[str, Any]:
    """Historical formal-menu fitting settings before an explicit sidecar existed.

    Legacy Daily Universal reports did not persist these fields.  Reuse is only
    permitted when the current request still equals the historical formal-menu
    defaults; any explicit/current setting change forces a new fit.
    """

    return {
        "max_epochs": int(BREAKOUT_QUALITY_DEFAULT_EPOCHS),
        "batch_size": int(BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE),
        "learning_rate": float(BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE),
        "weight_decay": float(BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY),
        "gradient_clip_norm": float(BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM),
        "use_inner_validation": bool(BREAKOUT_QUALITY_USE_INNER_VALIDATION),
        "inner_validation_months": int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS),
        "early_stopping_patience": int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE),
        "early_stopping_min_delta": float(BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA),
    }


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
    # Do not bind the fit to Forward score horizon/tail metadata.  Only source
    # and target semantics that can affect Selection fitting belong here.
    return {
        "dataset": summary.get("dataset"),
        "dataset_policy": summary.get("policy"),
        "training_universe_start_date": summary.get("training_universe_start_date"),
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
    if sidecar is not None:
        if int(sidecar.get("contract_version", -1)) != FITTED_MODEL_CONTRACT_VERSION:
            return None, "fitted-model contract version mismatch"
        if dict(sidecar.get("fitting_identity") or {}) != dict(expected_identity):
            return None, "fitted-model fitting identity mismatch"
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
            fitting_identity=dict(expected_identity),
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
    "FittedModelContract",
    "build_daily_forward_fitting_identity",
    "canonical_fitting_settings",
    "legacy_default_fitting_settings",
    "load_model_from_fitted_checkpoint",
    "load_reusable_fitted_model_contract",
    "scientific_torch_execution",
    "write_fitted_model_contract",
]
