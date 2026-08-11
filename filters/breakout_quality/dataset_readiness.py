"""Canonical metadata-only readiness for the breakout-quality Dataset artifact.

The model-research workflow and every downstream artifact planner must agree on
whether the indexed Dataset can be reused.  This module owns that decision; it
never builds or relabels the Dataset itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from filters.breakout_quality.contract import (
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    FEATURE_COLUMNS,
)
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    dataset_artifact_metadata_reasons,
    market_set_artifact_metadata_reasons,
    resolve_dataset_paths,
)
from filters.breakout_quality.market_set import market_set_contract_payload
from filters.breakout_quality.models.spec import get_model_spec
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.source_inventory import build_source_data_inventory


@dataclass(frozen=True)
class DatasetReadiness:
    ready: bool
    refresh_mode: str
    reasons: tuple[str, ...]
    summary_path: Path
    summary: dict[str, Any] | None


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def collect_dataset_readiness(
    project_root: str | Path,
    *,
    filter_id: str,
    dataset: str,
    max_tickers: int,
    model_architecture: str,
) -> DatasetReadiness:
    """Return the single canonical Dataset reuse/rebuild/relabel decision.

    This is the metadata-only equivalent of the strict runtime loader.  It keeps
    status/preparation cheap while still checking storage schema, artifact
    metadata, source-inventory freshness, policy and architecture-specific
    sidecars.  Builders remain outside this module.
    """

    root = Path(project_root).resolve()
    output_dir = resolve_filter_output_dir(root, filter_id=str(filter_id))
    paths = resolve_dataset_paths(output_dir)
    full_rebuild_reasons: list[str] = []
    relabel_reasons: list[str] = []

    required_paths = {**paths.artifact_paths(), "summary": paths.summary}
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        full_rebuild_reasons.append(f"dataset 工件缺少: {', '.join(missing)}")

    summary = _read_json_object(paths.summary)
    if summary is None:
        full_rebuild_reasons.append("dataset_summary.json 缺少、損壞或不是 JSON object")
        return DatasetReadiness(
            ready=False,
            refresh_mode="rebuild",
            reasons=tuple(full_rebuild_reasons),
            summary_path=paths.summary,
            summary=None,
        )

    if str(summary.get("filter_id") or "").strip() != str(filter_id).strip():
        full_rebuild_reasons.append("dataset filter_id 與目前設定不一致")
    if int(summary.get("dataset_storage_schema_version", -1)) != DATASET_STORAGE_SCHEMA_VERSION:
        full_rebuild_reasons.append("dataset storage schema 已變更")
    if str(summary.get("dataset_storage_format") or "") != DATASET_STORAGE_FORMAT:
        full_rebuild_reasons.append("dataset storage format 已變更")

    full_rebuild_reasons.extend(
        dataset_artifact_metadata_reasons(paths, summary.get("dataset_artifacts"))
    )

    model_spec = get_model_spec(str(model_architecture))
    if bool(model_spec.requires_market_set):
        if summary.get("market_set_contract") != market_set_contract_payload():
            full_rebuild_reasons.append("market-set input contract 已變更或缺少")
        full_rebuild_reasons.extend(
            market_set_artifact_metadata_reasons(
                paths,
                summary.get("market_set_artifacts"),
            )
        )

    requested_profile = str(dataset).strip().lower()
    stored_profile = str(summary.get("dataset") or "").strip().lower()
    if requested_profile not in {"reduced", "full"}:
        raise ValueError(f"不支援的dataset profile: {dataset!r}")
    if stored_profile != requested_profile:
        full_rebuild_reasons.append(
            f"dataset profile 不符: existing={stored_profile or 'missing'}, requested={requested_profile}"
        )

    source_selection = summary.get("source_selection")
    stored_max_tickers = None
    if isinstance(source_selection, dict):
        try:
            stored_max_tickers = int(source_selection.get("requested_max_tickers"))
        except (TypeError, ValueError):
            stored_max_tickers = None
    requested_max_tickers = max(0, int(max_tickers))
    if stored_max_tickers != requested_max_tickers:
        full_rebuild_reasons.append(
            "dataset ticker coverage 不符: "
            f"existing_max_tickers={stored_max_tickers}, requested_max_tickers={requested_max_tickers}"
        )

    if summary.get("feature_cache_policy") != DEFAULT_LABEL_POLICY.feature_cache_manifest_payload():
        full_rebuild_reasons.append("feature／high_len／benchmark／path-cache policy 已變更")
    if list(summary.get("feature_columns") or []) != list(FEATURE_COLUMNS):
        full_rebuild_reasons.append("feature contract 已變更")
    if list(summary.get("context_columns") or []) != list(CONTEXT_COLUMNS):
        full_rebuild_reasons.append("context contract 已變更")

    stored_inventory = summary.get("source_data_inventory")
    if not isinstance(stored_inventory, dict):
        full_rebuild_reasons.append("dataset 缺少 source_data_inventory；需完整重建一次")
    elif stored_profile == requested_profile:
        current_inventory = build_source_data_inventory(root, requested_profile)
        if stored_inventory != current_inventory:
            full_rebuild_reasons.append(
                "來源 CSV inventory 已更新: "
                f"existing={stored_inventory.get('csv_inventory_sha256')}, "
                f"current={current_inventory.get('csv_inventory_sha256')}"
            )

    if full_rebuild_reasons:
        return DatasetReadiness(
            ready=False,
            refresh_mode="rebuild",
            reasons=tuple(full_rebuild_reasons),
            summary_path=paths.summary,
            summary=summary,
        )

    if summary.get("label_policy") != DEFAULT_LABEL_POLICY.label_manifest_payload():
        relabel_reasons.append("label horizon／MFE／MAE／reward-risk policy 已變更")
    if summary.get("policy") != DEFAULT_LABEL_POLICY.as_manifest_payload():
        if not relabel_reasons:
            full_rebuild_reasons.append("dataset policy metadata 與目前設定不一致")

    if full_rebuild_reasons:
        return DatasetReadiness(
            ready=False,
            refresh_mode="rebuild",
            reasons=tuple(full_rebuild_reasons),
            summary_path=paths.summary,
            summary=summary,
        )
    if relabel_reasons:
        return DatasetReadiness(
            ready=False,
            refresh_mode="relabel",
            reasons=tuple(relabel_reasons),
            summary_path=paths.summary,
            summary=summary,
        )
    return DatasetReadiness(
        ready=True,
        refresh_mode="none",
        reasons=(),
        summary_path=paths.summary,
        summary=summary,
    )


__all__ = ["DatasetReadiness", "collect_dataset_readiness"]
