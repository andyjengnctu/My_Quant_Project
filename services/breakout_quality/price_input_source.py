"""Research-only alternate feature source. AO adjusted targets remain canonical.

Provider and raw materialization ownership stays in the existing Research service;
DL consumers receive one verified declarative source record. Non-raw models do not
observe any changed metadata, feature windows or identity fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from filters.breakout_quality.models.architectures import get_architecture_descriptor
from filters.breakout_quality.daily_ranker_data import load_daily_universal_ranker_data
from filters.breakout_quality.workflow_io import PROJECT_ROOT
from services.research.raw_price_strategy_dataset import (
    ensure_raw_price_strategy_dataset,
    inspect_raw_price_strategy_dataset,
)

RAW_PRICE_INPUT_CAPABILITY = "raw_unadjusted_price_input"


def uses_alternate_price_input(model_architecture: str) -> bool:
    return get_architecture_descriptor(model_architecture).has_capability(RAW_PRICE_INPUT_CAPABILITY)


def resolve_feature_price_source(
    *, model_architecture: str, project_root: str | Path = PROJECT_ROOT,
    create: bool = False,
) -> dict[str, Any] | None:
    """Verify immutable provider/control lineage and all materialized raw CSV SHA.

    For REUSE reads `create=False` cannot build or rewrite research artifacts.
    """
    if not uses_alternate_price_input(model_architecture):
        return None
    root = Path(project_root).resolve()
    result = (
        ensure_raw_price_strategy_dataset(root)
        if create else inspect_raw_price_strategy_dataset(root, deep=True)
    )
    if not bool(result.get("ready")):
        raise ValueError(
            "DL raw input source未READY；需先由canonical Research producer建立："
            + str(result.get("status"))
        )
    data_identity = result.get("data_identity")
    if not isinstance(data_identity, dict) or not data_identity.get("materialization_fingerprint"):
        raise ValueError("DL raw input materialization identity缺失")
    source_identity = result.get("source_identity")
    if not isinstance(source_identity, dict):
        raise ValueError("DL raw input provider source identity缺失")
    return {
        "dataset_dir": Path(result["dataset_dir"]),
        "identity": {
            "price_basis": "raw_unadjusted_ohlc_canonical_volume",
            "target_price_basis": "canonical_adjusted",
            "source_contract_id": str(source_identity["contract_id"]),
            "manifest_relative_path": Path(result["manifest_path"]).relative_to(root).as_posix(),
            "source_identity_fingerprint": str(result["source_identity_fingerprint"]),
            "materialization_fingerprint": str(data_identity["materialization_fingerprint"]),
            "dataset_inventory_sha256": str(data_identity["dataset_inventory_sha256"]),
        },
    }


def load_daily_universal_ranker_data_with_price_source(**kwargs):
    """Drop-in domain adapter for every direct daily-universal producer."""
    root = kwargs.get("project_root", PROJECT_ROOT)
    alternate = resolve_feature_price_source(
        model_architecture=kwargs["model_architecture"],
        project_root=root,
        create=True,
    )
    return load_daily_universal_ranker_data(**kwargs, feature_price_source=alternate)


def verified_current_price_source_identity(
    *, model_architecture: str, project_root: str | Path,
) -> dict[str, Any] | None:
    record = resolve_feature_price_source(
        model_architecture=model_architecture, project_root=project_root, create=False,
    )
    return None if record is None else dict(record["identity"])
