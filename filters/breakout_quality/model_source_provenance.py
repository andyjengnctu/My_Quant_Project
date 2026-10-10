"""Read-only current-source provenance for published Forward and Rolling DL.

CSV inventory, physical Dataset/Target readiness and hash validation are owned by
artifact_dependency_registry and their canonical producer/loader. This module
only compares the source records published in scientific model artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.file_integrity import canonical_json_sha256, compute_file_sha256, load_json_strict

from filters.breakout_quality.source_inventory import (
    legacy_source_inventory_semantically_equivalent,
)


def source_inventory_matches_current(
    *, published: Any, current: Any, project_root: str | Path | None = None,
) -> bool:
    """Strict provenance match, with only the verified 2026-09-12 bridge.

    A caller without a canonical project root never guesses that differently
    versioned inventories are equivalent. Dataset/Target hashes remain separate
    mandatory checks; this function never weakens those comparisons.
    """
    if published == current:
        return True
    if (
        project_root is None
        or not isinstance(published, Mapping)
        or not isinstance(current, Mapping)
    ):
        return False
    return legacy_source_inventory_semantically_equivalent(
        project_root, published=published, current=current,
    )


def canonical_source_dataset_artifacts(summary: Mapping[str, Any]) -> Any:
    """Normalize daily-universal vs event-group Dataset source SHA location."""
    return summary.get("source_dataset_artifacts", summary.get("dataset_artifacts"))


def recorded_input_price_source_issues(
    *, published: Any, project_root: str | Path | None,
) -> tuple[str, ...]:
    """Validate the exact pinned immutable raw feature bytes, without creating data.

    Current adjusted Dataset and Target remain separately governed by the
    canonical upstream-readiness checks. A score never silently falls back to
    adjusted inputs if its raw source cannot be proven.
    """
    if project_root is None or not isinstance(published, Mapping):
        return ("pinned raw input source is missing",)
    identity = dict(published)
    relative = identity.get("manifest_relative_path")
    root = Path(project_root).resolve()
    if not isinstance(relative, str) or not relative.strip():
        return ("pinned raw input manifest path missing",)
    path = (root / relative).resolve()
    expected_folder = (root / "outputs" / "research" / "raw_price_strategy_dataset").resolve()
    if not path.is_relative_to(expected_folder) or path.name != "manifest.json":
        return ("pinned raw input manifest path escapes research source namespace",)
    try:
        manifest = load_json_strict(path)
        if not isinstance(manifest, dict):
            return ("pinned raw input manifest invalid",)
        source = dict(manifest.get("source_identity") or {})
        source_fp = source.pop("source_identity_fingerprint", None)
        if canonical_json_sha256(source) != source_fp or (
            manifest.get("source_identity_fingerprint") != source_fp
            or identity.get("source_identity_fingerprint") != source_fp
        ):
            return ("pinned raw input provider lineage mismatch",)
        data = dict(manifest.get("data_identity") or {})
        rows = manifest.get("files")
        if not isinstance(rows, list) or not rows:
            return ("pinned raw input file evidence missing",)
        inventory_fp = canonical_json_sha256(rows)
        materialization_fp = canonical_json_sha256({
            "source_identity_fingerprint": source_fp,
            "dataset_inventory_sha256": inventory_fp,
        })
        # A quote-aware model must not silently reuse strict/numeric raw input
        # evidence or claim a mask absent from the immutable producer contract.
        observed_policy = source.get("quote_policy")
        if observed_policy is None:
            if identity.get("quote_policy") is not None or identity.get("input_representation") is not None:
                return ("pinned raw input quote representation mismatch",)
        elif (
            observed_policy != "observed_with_missing_quote_mask_v1"
            or identity.get("quote_policy") != observed_policy
            or identity.get("input_representation") != "raw_observed_quote_mask_300x12_v1"
        ):
            return ("pinned raw input quote representation mismatch",)
        if (
            identity.get("price_basis") != "raw_unadjusted_ohlc_canonical_volume"
            or identity.get("target_price_basis") != "canonical_adjusted"
            or identity.get("source_contract_id") != source.get("contract_id")
            or identity.get("dataset_inventory_sha256") != inventory_fp
            or manifest.get("dataset_inventory_sha256") != inventory_fp
            or data.get("materialization_fingerprint") != materialization_fp
            or identity.get("materialization_fingerprint") != materialization_fp
        ):
            return ("pinned raw input scientific/materialization identity mismatch",)
        dataset_dir = path.parent / "tw_stock_data_vip"
        expected_paths: set[str] = set()
        for row in rows:
            rel = str(row.get("relative_path") or "")
            file_path = (dataset_dir / rel).resolve()
            if (not rel or file_path.parent != dataset_dir.resolve()
                    or rel in expected_paths):
                return ("pinned raw input path/duplicate invalid",)
            expected_paths.add(rel)
            if (not file_path.is_file()
                or int(file_path.stat().st_size) != int(row.get("size_bytes") or -1)
                or compute_file_sha256(file_path) != row.get("content_sha256")):
                return ("pinned raw input CSV SHA256 mismatch: " + rel,)
        actual_paths = {file.name for file in dataset_dir.glob("*.csv")}
        if expected_paths != actual_paths:
            return ("pinned raw input CSV set mismatch",)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return (f"pinned raw input unavailable: {type(exc).__name__}: {exc}",)
    return ()


def _published_source_provenance_issues(
    *, label: str, source_dataset: Any, source_target: Any,
    dataset_summary: Mapping[str, Any],
    target_manifest: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> tuple[str, ...]:
    expected_inventory = dataset_summary.get("source_data_inventory")
    expected_artifacts = dataset_summary.get("dataset_artifacts")
    if not isinstance(expected_inventory, Mapping) or not isinstance(expected_artifacts, Mapping):
        return ("current Dataset lacks canonical source inventory/artifact hashes",)
    if not isinstance(source_dataset, Mapping):
        return (f"{label} source_dataset missing",)
    issues: list[str] = []
    if not source_inventory_matches_current(
        published=source_dataset.get("source_data_inventory"),
        current=expected_inventory, project_root=project_root,
    ):
        issues.append(f"{label} source_data_inventory mismatch")
    if canonical_source_dataset_artifacts(source_dataset) != expected_artifacts:
        issues.append(f"{label} source_dataset_artifacts mismatch")
    if source_dataset.get("dataset") != dataset_summary.get("dataset"):
        issues.append(f"{label} source dataset profile mismatch")
    if source_dataset.get("policy") != dataset_summary.get("policy"):
        issues.append(f"{label} source dataset policy mismatch")
    raw_source = source_dataset.get("input_price_source")
    if dataset_summary.get("requires_input_price_source") and not isinstance(raw_source, Mapping):
        issues.append(f"{label} required raw input source missing")
    if raw_source is not None:
        if dataset_summary.get("input_price_source") is not None and (
            raw_source != dataset_summary["input_price_source"]
        ):
            issues.append(f"{label} input_price_source mismatch")
        issues.extend(recorded_input_price_source_issues(published=raw_source, project_root=project_root))
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
    project_root: str | Path | None = None,
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
            project_root=project_root,
        ))
    return tuple(dict.fromkeys(issues))


def pit_published_source_provenance_issues(
    *, pit_manifest: Mapping[str, Any], dataset_summary: Mapping[str, Any],
    target_manifest: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
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
        project_root=project_root,
    )


__all__ = [
    "canonical_source_dataset_artifacts",
    "source_inventory_matches_current",
    "forward_published_source_provenance_issues",
    "pit_published_source_provenance_issues",
]
