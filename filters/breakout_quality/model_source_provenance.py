"""Read-only current-source provenance for published Forward and Rolling DL.

CSV inventory, physical Dataset/Target readiness and hash validation are owned by
artifact_dependency_registry and their canonical producer/loader. This module
only compares the source records published in scientific model artifacts.
"""

from __future__ import annotations

from typing import Any, Mapping


def canonical_source_dataset_artifacts(summary: Mapping[str, Any]) -> Any:
    """Normalize daily-universal vs event-group Dataset source SHA location."""
    return summary.get("source_dataset_artifacts", summary.get("dataset_artifacts"))


def _published_source_provenance_issues(
    *, label: str, source_dataset: Any, source_target: Any,
    dataset_summary: Mapping[str, Any],
    target_manifest: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    expected_inventory = dataset_summary.get("source_data_inventory")
    expected_artifacts = dataset_summary.get("dataset_artifacts")
    if not isinstance(expected_inventory, Mapping) or not isinstance(expected_artifacts, Mapping):
        return ("current Dataset lacks canonical source inventory/artifact hashes",)
    if not isinstance(source_dataset, Mapping):
        return (f"{label} source_dataset missing",)
    issues: list[str] = []
    if source_dataset.get("source_data_inventory") != expected_inventory:
        issues.append(f"{label} source_data_inventory mismatch")
    if canonical_source_dataset_artifacts(source_dataset) != expected_artifacts:
        issues.append(f"{label} source_dataset_artifacts mismatch")
    if source_dataset.get("dataset") != dataset_summary.get("dataset"):
        issues.append(f"{label} source dataset profile mismatch")
    if source_dataset.get("policy") != dataset_summary.get("policy"):
        issues.append(f"{label} source dataset policy mismatch")
    if target_manifest is not None:
        if not isinstance(source_target, Mapping):
            issues.append(f"{label} source target manifest missing")
        else:
            if source_target.get("artifacts") != target_manifest.get("artifacts"):
                issues.append(f"{label} source target artifacts mismatch")
            if source_target.get("target_contract") != target_manifest.get("target_contract"):
                issues.append(f"{label} source target contract mismatch")
    return tuple(issues)


def forward_published_source_provenance_issues(
    *, model_manifest: Mapping[str, Any], report: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    target_manifest: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Require independent matching Dataset/Target evidence in both publications."""
    issues: list[str] = []
    for label, publication, target_field in (
        ("manifest", model_manifest, "source_continuous_target"),
        ("report", report, "target_manifest"),
    ):
        issues.extend(_published_source_provenance_issues(
            label=label,
            source_dataset=publication.get("source_dataset"),
            source_target=publication.get(target_field),
            dataset_summary=dataset_summary,
            target_manifest=target_manifest,
        ))
    return tuple(dict.fromkeys(issues))


def pit_published_source_provenance_issues(
    *, pit_manifest: Mapping[str, Any], dataset_summary: Mapping[str, Any],
    target_manifest: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Require PIT top-level source proof against present canonical Dataset truth.

    Fold-level score bytes are separately bound by the PIT score/audit source
    records and the PIT producer owns per-fold fitting cache contracts.
    """
    return _published_source_provenance_issues(
        label="PIT manifest",
        source_dataset=pit_manifest.get("source_dataset"),
        source_target=pit_manifest.get("source_continuous_target"),
        dataset_summary=dataset_summary,
        target_manifest=target_manifest,
    )


__all__ = [
    "canonical_source_dataset_artifacts",
    "forward_published_source_provenance_issues",
    "pit_published_source_provenance_issues",
]
