"""Byte-preserving recovery of a doubly nested canonical Optimizer arm.

The Optimizer owns its workspace. A completed or resumable signed arm may be
relocated to its intended physical root, but it must never be silently adopted
from a different fitting contract or merged into an existing arm.
"""
from __future__ import annotations

from pathlib import Path

from core.file_integrity import canonical_json_sha256, load_json_strict


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
