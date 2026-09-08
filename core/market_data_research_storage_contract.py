"""Filesystem contract for frozen-candidate Research Market Data V2 artifacts."""
from __future__ import annotations

from pathlib import Path
import re

RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT = Path("data") / "research" / "market_data_v2"
RESEARCH_MARKET_DATA_V2_CANDIDATES_DIRNAME = "candidates"
RESEARCH_MARKET_DATA_V2_FREEZE_CANDIDATES_DIRNAME = "freeze_candidates"
RESEARCH_MARKET_DATA_V2_MATERIALIZATIONS_DIRNAME = "materializations"
RESEARCH_MARKET_DATA_V2_PROMOTIONS_DIRNAME = "promotions"
RESEARCH_V2_CANDIDATE_MANIFEST_FILENAME = "research_v2_candidate_manifest.json"
RESEARCH_V2_FREEZE_CANDIDATE_MANIFEST_FILENAME = "research_v2_freeze_candidate_manifest.json"
RESEARCH_V2_MATERIALIZATION_MANIFEST_FILENAME = "research_v2_materialization_manifest.json"
RESEARCH_V2_PROMOTION_MANIFEST_FILENAME = "research_v2_promotion_manifest.json"
RESEARCH_ACTIVE_GENERATION_FILENAME = "active_research_generation.json"
RESEARCH_V2_COMPATIBILITY_DATASET_DIRNAME = "tw_stock_data_vip"
RESEARCH_V2_DAILY_UNIVERSE_FILENAME = "daily_universe.sqlite3"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_fingerprint(value: str) -> str:
    text = str(value or "").strip().lower()
    if not _HEX64_RE.fullmatch(text):
        raise ValueError("Research V2 artifact fingerprint 不合法")
    return text


def resolve_research_v2_candidate_dir(project_root, provider_snapshot_fingerprint: str) -> Path:
    return (
        Path(project_root).resolve()
        / RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT
        / RESEARCH_MARKET_DATA_V2_CANDIDATES_DIRNAME
        / _require_fingerprint(provider_snapshot_fingerprint)
    )


def resolve_research_v2_candidate_manifest_path(project_root, provider_snapshot_fingerprint: str) -> Path:
    return resolve_research_v2_candidate_dir(project_root, provider_snapshot_fingerprint) / RESEARCH_V2_CANDIDATE_MANIFEST_FILENAME


def resolve_research_v2_daily_universe_path(project_root, provider_snapshot_fingerprint: str) -> Path:
    return resolve_research_v2_candidate_dir(project_root, provider_snapshot_fingerprint) / RESEARCH_V2_DAILY_UNIVERSE_FILENAME


def resolve_research_v2_freeze_candidate_dir(project_root, freeze_candidate_fingerprint: str) -> Path:
    return (
        Path(project_root).resolve()
        / RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT
        / RESEARCH_MARKET_DATA_V2_FREEZE_CANDIDATES_DIRNAME
        / _require_fingerprint(freeze_candidate_fingerprint)
    )


def resolve_research_v2_freeze_candidate_manifest_path(project_root, freeze_candidate_fingerprint: str) -> Path:
    return (
        resolve_research_v2_freeze_candidate_dir(project_root, freeze_candidate_fingerprint)
        / RESEARCH_V2_FREEZE_CANDIDATE_MANIFEST_FILENAME
    )


def resolve_research_v2_materialization_dir(project_root, materialization_fingerprint: str) -> Path:
    return (
        Path(project_root).resolve()
        / RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT
        / RESEARCH_MARKET_DATA_V2_MATERIALIZATIONS_DIRNAME
        / _require_fingerprint(materialization_fingerprint)
    )


def resolve_research_v2_materialization_manifest_path(project_root, materialization_fingerprint: str) -> Path:
    return (
        resolve_research_v2_materialization_dir(project_root, materialization_fingerprint)
        / RESEARCH_V2_MATERIALIZATION_MANIFEST_FILENAME
    )


def resolve_research_v2_compatibility_dataset_dir(project_root, materialization_fingerprint: str) -> Path:
    return (
        resolve_research_v2_materialization_dir(project_root, materialization_fingerprint)
        / RESEARCH_V2_COMPATIBILITY_DATASET_DIRNAME
    )


def resolve_research_v2_promotion_dir(project_root, promotion_fingerprint: str) -> Path:
    return (
        Path(project_root).resolve()
        / RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT
        / RESEARCH_MARKET_DATA_V2_PROMOTIONS_DIRNAME
        / _require_fingerprint(promotion_fingerprint)
    )


def resolve_research_v2_promotion_manifest_path(project_root, promotion_fingerprint: str) -> Path:
    return (
        resolve_research_v2_promotion_dir(project_root, promotion_fingerprint)
        / RESEARCH_V2_PROMOTION_MANIFEST_FILENAME
    )


def resolve_active_research_generation_path(project_root) -> Path:
    return Path(project_root).resolve() / RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT / RESEARCH_ACTIVE_GENERATION_FILENAME


__all__ = [
    "RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT",
    "RESEARCH_MARKET_DATA_V2_CANDIDATES_DIRNAME",
    "RESEARCH_MARKET_DATA_V2_FREEZE_CANDIDATES_DIRNAME",
    "RESEARCH_MARKET_DATA_V2_MATERIALIZATIONS_DIRNAME",
    "RESEARCH_MARKET_DATA_V2_PROMOTIONS_DIRNAME",
    "RESEARCH_V2_CANDIDATE_MANIFEST_FILENAME",
    "RESEARCH_V2_FREEZE_CANDIDATE_MANIFEST_FILENAME",
    "RESEARCH_V2_MATERIALIZATION_MANIFEST_FILENAME",
    "RESEARCH_V2_PROMOTION_MANIFEST_FILENAME",
    "RESEARCH_ACTIVE_GENERATION_FILENAME",
    "RESEARCH_V2_COMPATIBILITY_DATASET_DIRNAME",
    "RESEARCH_V2_DAILY_UNIVERSE_FILENAME",
    "resolve_research_v2_candidate_dir",
    "resolve_research_v2_candidate_manifest_path",
    "resolve_research_v2_daily_universe_path",
    "resolve_research_v2_freeze_candidate_dir",
    "resolve_research_v2_freeze_candidate_manifest_path",
    "resolve_research_v2_materialization_dir",
    "resolve_research_v2_materialization_manifest_path",
    "resolve_research_v2_compatibility_dataset_dir",
    "resolve_research_v2_promotion_dir",
    "resolve_research_v2_promotion_manifest_path",
    "resolve_active_research_generation_path",
]
