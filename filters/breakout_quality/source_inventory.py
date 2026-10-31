"""Fast source-data freshness fingerprint for breakout quality datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from core.data_utils import OHLCV_SANITIZATION_CONTRACT_VERSION, discover_unique_csv_inputs
from core.file_integrity import canonical_json_sha256, is_sha256_hex
from core.dataset_profiles import DATASET_PROFILE_FULL, get_dataset_dir, normalize_dataset_profile_key

SOURCE_DATA_INVENTORY_SCHEMA_VERSION = 1
SOURCE_DATA_INVENTORY_ALGORITHM = "sha256(ticker,relative_path,size_bytes,mtime_ns)"
# This additional local observation is intentionally NOT part of the historic
# CSV inventory identity, FIT fingerprint, or serialized research manifests.
SOURCE_CONTENT_WITNESS_SCHEMA_VERSION = 1
PRICE_CONTENT_MANIFEST_SCHEMA_VERSION = 1
# AI: This one-way compatibility proof is specific to the 2026-09-12 change.
PRE_FINITE_STABLE_OHLCV_CONTRACT_VERSION = 2


def _published_source_data_dir(project_root: str | Path, profile: str) -> tuple[Path, object | None]:
    """Resolve a *published* V2 generation from small signed manifests only.

    Ordinary consumers continue using get_dataset_dir, including its strict
    content verification. This narrow read-only query is only for the source
    publication manifest, which is itself bound to a previously verified
    content SHA, source metadata and V2 promotion fingerprint.
    """
    from core.dataset_profiles import get_unpromoted_dataset_dir
    if profile != DATASET_PROFILE_FULL:
        return Path(get_unpromoted_dataset_dir(project_root, profile)).resolve(), None
    from core.market_data_research_promotion import load_active_research_v2_promotion
    promoted = load_active_research_v2_promotion(
        project_root, required=False, verify_materialization_file_content=False,
    )
    if promoted is None:
        from config.market_data import RESEARCH_DATA_GENERATION_V2
        from core.market_data_contract import get_active_research_data_generation
        if get_active_research_data_generation().generation_id == RESEARCH_DATA_GENERATION_V2:
            raise ValueError("Active Research V2 generation requires validated publication")
        return Path(get_unpromoted_dataset_dir(project_root, profile)).resolve(), None
    return Path(promoted.full_dataset_dir).resolve(), promoted


def _build_source_data_inventory(
    project_root: str | Path, dataset: str, *, legacy_csv_only: bool = False,
    published_manifest_check: bool = False,
) -> dict[str, Any]:
    """One metadata algorithm owner for current and archived CSV-only identity."""

    profile = normalize_dataset_profile_key(dataset)
    promoted = None
    if published_manifest_check:
        data_dir, promoted = _published_source_data_dir(project_root, profile)
    else:
        data_dir = Path(get_dataset_dir(str(project_root), profile)).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"找不到 breakout quality source data directory: {data_dir}")

    csv_inputs, duplicate_lines = discover_unique_csv_inputs(str(data_dir))
    if not csv_inputs:
        raise FileNotFoundError(f"breakout quality source data directory 沒有可用 CSV: {data_dir}")

    digest = hashlib.sha256()
    total_size_bytes = 0
    latest_mtime_ns = 0
    for ticker, raw_path in csv_inputs:
        path = Path(raw_path).resolve()
        stat = path.stat()
        size_bytes = int(stat.st_size)
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
        relative_path = path.relative_to(data_dir).as_posix()

        total_size_bytes += size_bytes
        latest_mtime_ns = max(latest_mtime_ns, mtime_ns)
        for token in (str(ticker), relative_path, str(size_bytes), str(mtime_ns)):
            digest.update(token.encode("utf-8"))
            digest.update(b"\0")

    generation_lineage = None
    fingerprint_algorithm = SOURCE_DATA_INVENTORY_ALGORITHM
    if not legacy_csv_only:
        # AI: Invalidate derived artifacts when canonical OHLCV input semantics change.
        digest.update(f"ohlcv_sanitization_contract={OHLCV_SANITIZATION_CONTRACT_VERSION}\0".encode("utf-8"))
        fingerprint_algorithm += "+ohlcv_sanitization_contract"

    if profile == DATASET_PROFILE_FULL and not legacy_csv_only:
        from config.market_data import RESEARCH_DATA_GENERATION_V2
        from core.market_data_research_promotion import (
            get_effective_research_data_generation,
            load_active_research_v2_promotion,
        )

        if published_manifest_check:
            active = None
        else:
            active = get_effective_research_data_generation(project_root)
        if ((promoted is not None and published_manifest_check) or
                (active is not None and active.generation_id == RESEARCH_DATA_GENERATION_V2)):
            if promoted is None:
                promoted = load_active_research_v2_promotion(project_root, required=True)
            generation_lineage = {
                "generation_id": RESEARCH_DATA_GENERATION_V2,
                "frozen_cutoff": str(promoted.frozen_cutoff),
                "promotion_fingerprint": promoted.promotion_fingerprint,
                "materialization_fingerprint": promoted.materialization_fingerprint,
            }
            for token in (
                generation_lineage["generation_id"],
                generation_lineage["frozen_cutoff"],
                generation_lineage["promotion_fingerprint"],
                generation_lineage["materialization_fingerprint"],
            ):
                digest.update(str(token).encode("utf-8"))
                digest.update(b"\0")
            fingerprint_algorithm += "+research_v2_generation_lineage"

    payload = {
        "schema_version": SOURCE_DATA_INVENTORY_SCHEMA_VERSION,
        "dataset_profile": profile,
        "csv_file_count": int(len(csv_inputs)),
        "csv_total_bytes": int(total_size_bytes),
        "csv_latest_mtime_ns": int(latest_mtime_ns),
        "csv_inventory_sha256": digest.hexdigest(),
        "fingerprint_algorithm": fingerprint_algorithm,
        "ignored_duplicate_count": int(len(duplicate_lines or [])),
    }
    if not legacy_csv_only:
        payload["ohlcv_sanitization_contract_version"] = OHLCV_SANITIZATION_CONTRACT_VERSION
    if generation_lineage is not None:
        payload["research_market_data_lineage"] = generation_lineage
    return payload


def _source_content_witness(
    project_root: str | Path, dataset: str, inventory: Mapping[str, Any],
) -> str:
    """Detect a same-metadata source rewrite without changing legacy identity.

    A content witness is a local integrity observation, not a new scientific
    input or an authorization to treat previously unproven history as verified.
    The *first* observation is trust-on-first-use; subsequent calls (including
    in new processes) fail closed if the same metadata inventory has new bytes.
    Changing legitimate mtime/size creates a different inventory identity and
    continues to be handled by the canonical dataset/fit readiness planner.
    """
    root = Path(project_root).resolve()
    profile = normalize_dataset_profile_key(dataset)
    data_dir = Path(get_dataset_dir(str(root), profile)).resolve()
    csv_inputs, _ = discover_unique_csv_inputs(str(data_dir))
    content = hashlib.sha256()
    price_digest = hashlib.sha256()
    for ticker, raw_path in sorted(csv_inputs, key=lambda row: (str(row[0]), str(row[1]))):
        path = Path(raw_path).resolve()
        before = path.stat()
        file_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                file_hash.update(chunk)
        after = path.stat()
        if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns,
             before.st_dev, before.st_ino)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns,
                    after.st_dev, after.st_ino)):
            raise ValueError("來源 CSV 在內容驗證期間已更新，拒絕判定同一 source inventory")
        for token in (str(ticker), path.relative_to(data_dir).as_posix(), file_hash.hexdigest()):
            content.update(token.encode("utf-8"))
            content.update(b"\0")
        # Exactly the historic Oracle Replay dataset SHA algorithm. This
        # runs during the canonical source-content check, not on menu entry.
        price_digest.update(str(ticker).encode("utf-8") + b"\0")
        price_digest.update(file_hash.hexdigest().encode("ascii") + b"\n")
    # Detect additions/deletions and regular metadata changes during the scan.
    if dict(inventory) != _build_source_data_inventory(root, profile):
        raise ValueError("來源 CSV inventory 在內容驗證期間已變更")

    digest = str(inventory["csv_inventory_sha256"])
    witness_dir = root / "outputs" / "filters" / "breakout_quality" / "source_integrity" / profile
    witness = witness_dir / (digest + ".json")
    payload = {
        "schema_version": SOURCE_CONTENT_WITNESS_SCHEMA_VERSION,
        "dataset_profile": profile,
        "csv_inventory_sha256": digest,
        "csv_content_sha256": content.hexdigest(),
        "csv_file_count": len(csv_inputs),
    }
    witness_dir.mkdir(parents=True, exist_ok=True)
    if not witness.exists():
        # An immutable, atomic first observation. Never overwrite a previous
        # observation even when another process races to publish it. A staged
        # hard-link is atomic on the same volume on Windows and POSIX.
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".source_content_", suffix=".tmp", dir=str(witness_dir),
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, witness)
            except FileExistsError as exc:
                # A concurrent writer won the atomic publish race. Validate its
                # immutable witness below; keep the collision traceable at DEBUG.
                logging.getLogger(__name__).debug(
                    "來源 CSV 內容見證已由其他程序建立；改驗證現有檔案 (%s): %s",
                    witness.name, exc,
                )
        finally:
            temporary.unlink(missing_ok=True)
    try:
        observed = json.loads(witness.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("來源 CSV 內容見證無法讀取，不允許沿用資料身分") from exc
    if observed != payload:
        raise ValueError(
            "來源 CSV 內容已變更，但檔案大小與修改時間等既有 identity 未變；"
            "拒絕沿用來源資料、FIT 或 Score，須調查來源及重新取得合法工件"
        )
    return price_digest.hexdigest()


def _price_manifest_path(root: Path, profile: str,
                         inventory_sha: str, revision_sha: str) -> Path:
    # A source may be replaced by identical bytes while preserving size/mtime.
    # The certified content witness then stays the same, but its physical
    # revision must be separately attested, never overwrite an older proof.
    if not is_sha256_hex(inventory_sha) or not is_sha256_hex(revision_sha):
        raise ValueError("Price manifest revision identity must be SHA256")
    return (root / "outputs" / "filters" / "breakout_quality" / "source_integrity"
            / profile / "price_manifests" / (inventory_sha + "." + revision_sha + ".json"))


def _price_file_revision(root: Path, profile: str, *, fast_read: bool = False) -> str:
    """Cheap untrusted-mutable-source change detector; never reads CSV bytes.

    The publication signature also includes inode/creation-or-change time so
    rewrites which restore the old mtime/size cannot silently pass on POSIX.
    Windows still requires a producer-managed generation for strong guarantees:
    external writers with privileges to spoof all metadata are outside this
    witness's trust boundary and must be caught by a new content publication.
    """
    data_root = (_published_source_data_dir(root, profile)[0]
                 if fast_read else Path(get_dataset_dir(str(root), profile)).resolve())
    files, duplicates = discover_unique_csv_inputs(str(data_root))
    if not files or duplicates:
        raise ValueError("Price publication requires a unique nonempty CSV inventory")
    digest = hashlib.sha256()
    for ticker, filename in sorted(files, key=lambda row: (str(row[0]), str(row[1]))):
        path = Path(filename).resolve()
        stat = path.stat()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise ValueError("Price publication source escapes canonical Research dataset")
        fields = (str(ticker), path.relative_to(data_root).as_posix(),
                  str(stat.st_size), str(stat.st_mtime_ns), str(stat.st_ctime_ns),
                  str(stat.st_dev), str(stat.st_ino))
        for field in fields:
            digest.update(field.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _publish_price_manifest(project_root: str | Path, dataset: str,
                            inventory: Mapping[str, Any], price_sha: str,
                            *, scanned_revision_sha256: str) -> None:
    """Called only by the canonical source owner after content verification.

    The old source-content witness remains immutable. A separately signed,
    content-addressed projection preserves the *original* Oracle price SHA
    without rebuilding prices or changing any scientific/Replay identity.
    """
    root = Path(project_root).resolve()
    profile = normalize_dataset_profile_key(dataset)
    if not is_sha256_hex(price_sha) or inventory != _build_source_data_inventory(root, profile):
        raise ValueError("Price manifest publication raced a source update")
    revision = _price_file_revision(root, profile)
    if revision != scanned_revision_sha256:
        raise ValueError("Price source physical revision changed during content certification")
    if inventory != _build_source_data_inventory(root, profile):
        raise ValueError("Price manifest publication observed a changed source inventory")
    witness = (root / "outputs" / "filters" / "breakout_quality" / "source_integrity"
               / profile / (str(inventory["csv_inventory_sha256"]) + ".json"))
    try:
        proof = json.loads(witness.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("Canonical price source content witness missing") from exc
    if (not isinstance(proof, dict) or proof.get("schema_version") != SOURCE_CONTENT_WITNESS_SCHEMA_VERSION
            or proof.get("dataset_profile") != profile
            or proof.get("csv_inventory_sha256") != inventory["csv_inventory_sha256"]
            or proof.get("csv_file_count") != inventory["csv_file_count"]
            or not is_sha256_hex(proof.get("csv_content_sha256"))):
        raise ValueError("Canonical price source content witness is incompatible")
    identity = {
        "schema_version": PRICE_CONTENT_MANIFEST_SCHEMA_VERSION,
        "dataset_profile": profile,
        "source_inventory": dict(inventory),
        "price_inventory_sha256": price_sha,
        "file_revision_sha256": revision,
        "content_witness_sha256": canonical_json_sha256(proof),
    }
    payload = {**identity, "manifest_sha256": canonical_json_sha256(identity)}
    path = _price_manifest_path(root, profile, str(inventory["csv_inventory_sha256"]), revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".price_manifest_", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("Published price manifest corrupt") from exc
    if recorded != payload:
        raise ValueError("Published price manifest conflicts with certified Research source")


def read_published_price_inventory(project_root: str | Path, dataset: str) -> str | None:
    """Read only the small published proof and file metadata, NEVER CSV content.

    None means legacy Research data was not yet published with this capability;
    callers must continue using the original strict full-content verifier.
    Malformed/stale published evidence never authorizes a fast REUSE.
    """
    root = Path(project_root).resolve()
    profile = normalize_dataset_profile_key(dataset)
    # Legacy or isolated price-only fixtures have no published generation.
    # Keep the original strict content-SHA path without requiring an indexed
    # breakout-quality Dataset, and do not create a manifest in REUSE preflight.
    directory = (root / "outputs" / "filters" / "breakout_quality"
                 / "source_integrity" / profile / "price_manifests")
    if not directory.is_dir():
        return None
    inventory = _build_source_data_inventory(root, profile, published_manifest_check=True)
    revision = _price_file_revision(root, profile, fast_read=True)
    path = _price_manifest_path(root, profile, str(inventory["csv_inventory_sha256"]), revision)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = {k: v for k, v in payload.items() if k != "manifest_sha256"}
        if (set(identity) != {"schema_version", "dataset_profile", "source_inventory",
                              "price_inventory_sha256", "file_revision_sha256", "content_witness_sha256"}
                or identity["schema_version"] != PRICE_CONTENT_MANIFEST_SCHEMA_VERSION
                or identity["dataset_profile"] != profile
                or identity["source_inventory"] != inventory
                or not is_sha256_hex(identity["price_inventory_sha256"])
                or not is_sha256_hex(identity["file_revision_sha256"])
                or not is_sha256_hex(identity["content_witness_sha256"])
                or canonical_json_sha256(identity) != payload.get("manifest_sha256")
                or _price_file_revision(root, profile, fast_read=True) != identity["file_revision_sha256"]):
            raise ValueError("Published price manifest changed or does not match source revision")
        witness = (root / "outputs" / "filters" / "breakout_quality" / "source_integrity"
                   / profile / (str(inventory["csv_inventory_sha256"]) + ".json"))
        proof = json.loads(witness.read_text(encoding="utf-8"))
        if (not isinstance(proof, dict)
                or proof.get("csv_inventory_sha256") != inventory["csv_inventory_sha256"]
                or proof.get("dataset_profile") != profile
                or proof.get("csv_file_count") != inventory["csv_file_count"]
                or not is_sha256_hex(proof.get("csv_content_sha256"))
                or canonical_json_sha256(proof) != identity["content_witness_sha256"]):
            raise ValueError("Published price content witness has changed")
        # A rewrite between the metadata checks cannot pass unnoticed.
        if (_build_source_data_inventory(root, profile, published_manifest_check=True) != inventory
                or _price_file_revision(root, profile, fast_read=True) != revision):
            raise ValueError("Source inventory or physical revision changed while verifying price manifest")
    except (OSError, ValueError, UnicodeError, KeyError, TypeError) as exc:
        raise ValueError("Published price manifest invalid: fail closed") from exc
    return str(identity["price_inventory_sha256"])


def certify_existing_dataset_price_generation(project_root: str | Path,
                                               dataset: str) -> str | None:
    """One-time content attestation for an already published canonical Dataset.

    Called by the Oracle reader only when the fast proof does not yet exist.
    It cannot grant new scientific truth: the pre-existing Dataset summary must
    match the actual canonical source inventory, then the *original* full-byte
    source witness is verified before the source owner atomically publishes a
    price-only evidence projection. Replay receipts and Dataset summary bytes
    are never modified; repeat previews only inspect the small manifest.

    Price-only/isolated research fixtures without a published Dataset do not
    meet this condition and retain the exact historical strict SHA fallback.
    """
    root = Path(project_root).resolve()
    profile = normalize_dataset_profile_key(dataset)
    from filters.breakout_quality.contract import DEFAULT_FILTER_ID
    path = (root / "outputs" / "filters" / "breakout_quality"
            / DEFAULT_FILTER_ID / "dataset_summary.json")
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Canonical Dataset summary is unreadable; no automatic price certification") from exc
    if not isinstance(summary, dict) or summary.get("dataset") != profile:
        # This Dataset belongs to another profile, not to this source.
        return None
    inventory = _build_source_data_inventory(root, profile)
    if summary.get("source_data_inventory") != inventory:
        # A changed or stale Dataset cannot certify a new price generation.
        # The existing slow content-hash fallback will detect the new bytes.
        return None
    build_source_data_inventory(root, profile, publish_price_manifest=True)
    published = read_published_price_inventory(root, profile)
    if published is None:
        raise ValueError("One-time source certification failed to publish price evidence")
    return published


def build_source_data_inventory(project_root: str | Path, dataset: str, *,
                                publish_price_manifest: bool = False) -> dict[str, Any]:
    """Preserve the legacy source identity; optionally publish its verified price SHA.

    The default remains the byte-verifying call used by ordinary consumers.
    This source owner is used by the Dataset Producer and by a one-time
    migration of an already published, identity-matched Dataset.
    """
    inventory = _build_source_data_inventory(project_root, dataset)
    # A full-content proof cannot be signed for a different physical source
    # revision, even when an external writer restores the old size/mtime.
    before_revision = (_price_file_revision(Path(project_root).resolve(),
                                            normalize_dataset_profile_key(dataset))
                       if publish_price_manifest else None)
    price_sha = _source_content_witness(project_root, dataset, inventory)
    if publish_price_manifest:
        _publish_price_manifest(project_root, dataset, inventory, price_sha,
                                scanned_revision_sha256=before_revision)
    return inventory


@lru_cache(maxsize=8)
def _legacy_sanitizer_semantics_unchanged(
    root: str, dataset: str, current_inventory_sha256: str
) -> bool:
    """Fail-closed content scan for the archived sanitizer-v1 -> v2 delta.

    Prior sanitization could differ from v2 for non-finite OHLCV or unstable
    sorting of duplicate trading dates. Reject both patterns across ALL sources,
    not just the tickers that were selected for a particular fitted model.
    """
    import numpy as np
    import pandas as pd

    data_dir = Path(get_dataset_dir(root, dataset)).resolve()
    csv_inputs, _ = discover_unique_csv_inputs(str(data_dir))
    if not csv_inputs:
        return False
    required = ("Open", "High", "Low", "Close", "Volume")
    try:
        for _, raw_path in csv_inputs:
            frame = pd.read_csv(
                raw_path,
                usecols=lambda name: str(name).capitalize() in (*required, "Time", "Date"),
            )
            frame.columns = [str(col).capitalize() for col in frame.columns]
            date_key = "Time" if "Time" in frame.columns else "Date"
            if date_key not in frame.columns or any(col not in frame.columns for col in required):
                return False
            numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
            if not np.isfinite(numeric).all():
                return False
            dates = pd.to_datetime(frame[date_key], errors="coerce")
            if dates.isna().any() or dates.duplicated().any():
                return False
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, pd.errors.ParserError):
        return False
    # A source changed while the scan was running: never cache a passing proof.
    return (
        build_source_data_inventory(root, dataset).get("csv_inventory_sha256")
        == current_inventory_sha256
    )


def legacy_source_inventory_semantically_equivalent(
    project_root: str | Path, *, published: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Check narrowly allowed source-equivalence WITHOUT rewriting old evidence.

    Require exact historical CSV-only digest, exact current digest, and a full
    scan showing the two modified sanitizer behaviors cannot change any row.
    A version mismatch alone is never sufficient to approve reuse.
    """
    if dict(published) == dict(current):
        return True
    if (
        current.get("ohlcv_sanitization_contract_version") != PRE_FINITE_STABLE_OHLCV_CONTRACT_VERSION
        or current.get("fingerprint_algorithm") != SOURCE_DATA_INVENTORY_ALGORITHM + "+ohlcv_sanitization_contract"
        or published.get("fingerprint_algorithm") != SOURCE_DATA_INVENTORY_ALGORITHM
        or "ohlcv_sanitization_contract_version" in published
        or "research_market_data_lineage" in current
        or "research_market_data_lineage" in published
    ):
        return False
    dataset = current.get("dataset_profile")
    if dataset not in ("full", "reduced") or published.get("dataset_profile") != dataset:
        return False
    root = str(Path(project_root).resolve())
    try:
        if dict(current) != build_source_data_inventory(root, dataset):
            return False
        if dict(published) != _build_source_data_inventory(root, dataset, legacy_csv_only=True):
            return False
        return _legacy_sanitizer_semantics_unchanged(
            root, dataset, str(current.get("csv_inventory_sha256"))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


__all__ = [
    "SOURCE_DATA_INVENTORY_ALGORITHM",
    "SOURCE_DATA_INVENTORY_SCHEMA_VERSION",
    "build_source_data_inventory",
    "legacy_source_inventory_semantically_equivalent",
]
