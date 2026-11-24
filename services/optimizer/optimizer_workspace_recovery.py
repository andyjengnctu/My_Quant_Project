"""Byte-preserving recovery of a doubly nested canonical Optimizer arm.

The Optimizer owns its workspace. A completed or resumable signed arm may be
relocated to its intended physical root, but it must never be silently adopted
from a different fitting contract or merged into an existing arm.
"""
from __future__ import annotations

from pathlib import Path

from core.file_integrity import canonical_json_sha256, load_json_strict
from core.artifact_inspection import ArtifactInspection, InspectionStatus


_PREFLIGHT_VOLATILE_FIELDS = frozenset({
    "created_at", "status", "runtime_identity_sha256",
    "binary_pit_optimizer_coverage_path",
})


def _nested_relocation_paths(canonical_arm_dir: Path) -> tuple[Path, Path]:
    arm = Path(canonical_arm_dir).resolve()
    return arm / arm.name, arm.parent / f".{arm.name}.nested_relocation"


def nested_optimizer_workspace_candidate(canonical_arm_dir: Path) -> Path | None:
    """Locate only the known double-nesting accident or interrupted relocation."""
    nested, staged = _nested_relocation_paths(canonical_arm_dir)
    if nested.exists() and staged.exists():
        raise ValueError("Conflicting nested and staged Optimizer workspaces")
    return staged if staged.exists() else (nested if nested.exists() else None)


def verify_optimizer_runtime_preflight(
    path: Path, *, expected_runtime_identity_sha256: str,
) -> dict[str, object]:
    """Verify a preflight against its current canonical scientific identity."""
    preflight = load_json_strict(Path(path))
    if not isinstance(preflight, dict):
        raise ValueError("Optimizer signed preflight must be an object")
    identity = str(preflight.get("runtime_identity_sha256") or "")
    if (not identity or identity != str(expected_runtime_identity_sha256)
            or canonical_json_sha256({
                k: v for k, v in preflight.items() if k not in _PREFLIGHT_VOLATILE_FIELDS
            }) != identity):
        raise ValueError("Optimizer signed preflight identity mismatch")
    return preflight


def inspect_full_roos_preflight(path: Path, *, project_root: Path) -> ArtifactInspection:
    """Return the producer's verified preflight and legal direct/fold study IDs."""
    from types import SimpleNamespace
    from services.optimizer.strategy_param_training import (
        SCHEMA_VERSION, build_selection_full_roos_schedule_contract,
        build_selection_full_roos_runtime_contract,
        selection_full_roos_optimizer_runtime_cache_identity,
    )

    try:
        raw = load_json_strict(path)
        if not isinstance(raw, dict):
            raise ValueError("Optimizer signed preflight must be an object")
        preflight = verify_optimizer_runtime_preflight(
            path, expected_runtime_identity_sha256=raw.get("runtime_identity_sha256", ""),
        )
        if (preflight.get("schema_version") != SCHEMA_VERSION
                or preflight.get("mode") != "selection_full_roos_parameter_training"
                or preflight.get("status") != "P4_HISTORY_PREFLIGHT_PASS"
                or preflight.get("training_dl_enabled") is not False):
            return ArtifactInspection(InspectionStatus.INCOMPATIBLE, "unsupported Full ROOS preflight contract")
        contract = {key: value for key, value in preflight.items() if key not in {"created_at", "status"}}
        rolling = preflight["rolling_policy"]
        schedule = build_selection_full_roos_schedule_contract(
            first_oos_date=rolling["first_oos_date"], last_oos_date=rolling["last_oos_date"],
            train_window_months=rolling["train_window_months"], oos_months=rolling["oos_horizon_months"],
        )
        args = SimpleNamespace(**{key: preflight[key] for key in (
            "dataset", "trials_per_fold", "fixed_risk", "max_position_cap_pct", "max_positions", "rotation",
        )}, seed_ensemble_policy_override=preflight.get("seed_ensemble_policy"),
            score_policy_overrides=preflight.get("score_policy_overrides"))
        if contract != build_selection_full_roos_runtime_contract(
            root=project_root, args=args, schedule_contract=schedule,
            dataset_identity_override=preflight["dataset_identity"],
        ):
            return ArtifactInspection(InspectionStatus.INCOMPATIBLE, "Full ROOS preflight differs from producer contract")
        identities = frozenset(
            selection_full_roos_optimizer_runtime_cache_identity(contract, optimizer_search_protocol=protocol)
            for protocol in (None, "fixed_multiseed_v1")
        )
        return ArtifactInspection(InspectionStatus.READY, "", (preflight, identities))
    except FileNotFoundError as exc:
        return ArtifactInspection(InspectionStatus.MISSING, str(exc))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        return ArtifactInspection(InspectionStatus.INVALID, f"{type(exc).__name__}: {exc}")


def inspect_nested_optimizer_workspace(
    *, canonical_arm_dir: Path, expected_runtime_identity_sha256: str,
) -> Path | None:
    """Read-only detection. Verify the producer's signed preflight before adoption.

    Also detects a relocation interrupted between its two same-volume renames.
    A distinct canonical workspace or two recovery candidates is a hard conflict.
    """
    arm = Path(canonical_arm_dir).resolve()
    nested, staged = _nested_relocation_paths(arm)
    source = nested_optimizer_workspace_candidate(arm)
    if source is None:
        return None
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Nested Optimizer workspace must be a real directory")
    if arm.exists():
        if not arm.is_dir():
            raise ValueError("Canonical Optimizer arm is not a directory")
        if any(item != nested for item in arm.iterdir()):
            raise ValueError("Canonical Optimizer arm has conflicting work artifacts")
    verify_optimizer_runtime_preflight(
        source / "rolling_preflight.json",
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
    )
    return source


def adopt_nested_optimizer_workspace(
    *, canonical_arm_dir: Path, expected_runtime_identity_sha256: str,
) -> bool:
    """Move the verified workspace via same-volume renames, without rewriting bytes.

    The expected arm is initially a container for its accidental child. Rename
    the child to a sibling, remove the empty container, then rename back to the
    canonical arm location. A failed/interrupted step remains inspectable and
    resumable; no preexisting canonical artifact is overwritten.
    """
    arm = Path(canonical_arm_dir).resolve()
    source = inspect_nested_optimizer_workspace(
        canonical_arm_dir=arm,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
    )
    if source is None:
        return False
    nested, staged = _nested_relocation_paths(arm)
    if source == nested:
        nested.rename(staged)
    if arm.exists():
        arm.rmdir()
    staged.rename(arm)
    return True
