"""Disposable, independently verified real-path *post-replay diagnostic* cache.

The canonical model sample provider is the only writer.  Readers must resolve
and verify the same seed-specific Min teacher BEFORE trusting this projection.
No model FIT, score producer, portfolio selection or Replay consumes these
future outcomes.  A miss falls back to the ordinary canonical sample provider.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from zipfile import BadZipFile
import zlib

import numpy as np
import pandas as pd

from core.file_integrity import compute_file_sha256, file_change_signature
from filters.breakout_quality.dataset_store import resolve_dataset_paths
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs
from services.breakout_quality.formal_mintrade_outcome_cache import (
    _execution_dependency_files, _implementation_digest,
)

_SCHEMA = 1
_CACHE_SUBDIR = Path("outputs/filters/breakout_quality/real_path_diagnostic_cache")
_LOG = logging.getLogger(__name__)


def _digest_json(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _slot(root: Path, filter_id: str, architecture: str, profile: str, seed: int) -> Path:
    name = _digest_json((filter_id, architecture, profile, int(seed)))
    return root / _CACHE_SUBDIR / name[:2] / (name + ".npz")


def real_path_diagnostic_candidate_exists(
    *, root: Path, filter_id: str, architecture: str, profile: str, seed: int,
) -> bool:
    """Cheap no-I/O-preparation probe; never authorizes a cache hit."""
    return _slot(Path(root).resolve(), filter_id, architecture, profile, seed).is_file()


@lru_cache(maxsize=1)
def _data_implementation_sha256() -> str:
    """Track the actual Python dependencies of the canonical daily provider.

    Reorganizing unrelated report code does not change this execution-only
    fingerprint unless the provider itself imports the changed code.
    """
    code_root = Path(__file__).resolve().parents[2]
    sources = _execution_dependency_files(
        code_root, entrypoints=(
            "filters/breakout_quality/daily_ranker_data.py",
            "filters/breakout_quality/profile_ranker_data.py",
        ),
    )
    digest = sha256()
    for path in sources:
        digest.update(path.relative_to(code_root).as_posix().encode("utf-8"))
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _verified_current_inputs(root: Path, filter_id: str, *, origin_bundle=None) -> dict:
    """Read source CONTENT, not just mtime, before a disposable cache hit.

    A rewrite preserving file length and mtime must not resurrect old Target
    labels.  The official Dataset manifest and each declared backing artifact
    are also checked byte-for-byte, without rebuilding or republishing them.
    """
    from filters.breakout_quality.paths import resolve_filter_output_dir
    paths = resolve_dataset_paths(resolve_filter_output_dir(root, filter_id))
    before = file_change_signature(paths.summary)
    summary_bytes = paths.summary.read_bytes()
    if before != file_change_signature(paths.summary):
        raise RuntimeError("Dataset summary changed while verifying diagnostic inputs")
    summary = json.loads(summary_bytes.decode("utf-8-sig"))
    if not isinstance(summary, dict) or str(summary.get("filter_id")) != str(filter_id):
        raise ValueError("Real-path diagnostic requires the canonical Dataset summary")
    # A producer result must belong to the SAME Dataset generation that is
    # being attested now. Otherwise a source replacement between building the
    # bundle and writing the cache could re-sign obsolete future outcomes.
    if origin_bundle is not None:
        origin = origin_bundle.summary
        recorded = (
            ("filter_id", "filter_id"),
            ("dataset", "dataset"),
            ("source_data_inventory", "source_data_inventory"),
            ("source_dataset_artifacts", "dataset_artifacts"),
            ("source_data_date_range", "source_data_date_range"),
            ("policy", "policy"),
        )
        if any(origin.get(bundle_key) != summary.get(dataset_key)
               for bundle_key, dataset_key in recorded):
            raise ValueError("Canonical real-path bundle Dataset source changed before publication")
    dataset = str(summary.get("dataset") or "")
    inventory = build_source_data_inventory(root, dataset)
    if inventory != summary.get("source_data_inventory"):
        raise ValueError("Real-path diagnostic Dataset source inventory has changed")
    artifacts = summary.get("dataset_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Real-path diagnostic Dataset artifacts missing")
    for name, path in paths.artifact_paths().items():
        record = artifacts.get(name)
        if (not isinstance(record, dict) or record.get("filename") != path.name or
            record.get("sha256") != compute_file_sha256(path) or
            int(record.get("size_bytes", -1)) != path.stat().st_size):
            raise ValueError(f"Real-path diagnostic Dataset artifact changed: {name}")
    # Some canonical Dataset generations also publish market-set context.
    # Its bytes are inputs to the same stock-day eligibility/feature contract.
    market_records = summary.get("market_set_artifacts")
    if summary.get("market_set_contract") is not None and not isinstance(market_records, dict):
        raise ValueError("Real-path diagnostic market-set artifact manifest missing")
    if isinstance(market_records, dict):
        for name, path in paths.market_set_artifact_paths().items():
            record = market_records.get(name)
            if (not isinstance(record, dict) or record.get("filename") != path.name or
                record.get("sha256") != compute_file_sha256(path) or
                int(record.get("size_bytes", -1)) != path.stat().st_size):
                raise ValueError(f"Real-path diagnostic market-set artifact changed: {name}")
    csv_inputs, duplicates = discover_dataset_csv_inputs(root, dataset)
    if duplicates or not csv_inputs:
        raise ValueError("Real-path diagnostic requires unique nonempty source CSVs")
    digest = sha256()
    for ticker, path in sorted(csv_inputs, key=lambda row: (str(row[0]), str(row[1]))):
        before = file_change_signature(path)
        content_hash = compute_file_sha256(path)
        if before != file_change_signature(path):
            raise RuntimeError("Real-path diagnostic CSV changed during content hash")
        digest.update(str(ticker).encode("utf-8") + b"\0")
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(content_hash.encode("ascii") + b"\0")
    if inventory != build_source_data_inventory(root, dataset):
        raise RuntimeError("Real-path diagnostic CSV inventory changed during verification")
    return {
        "dataset_summary_sha256": sha256(summary_bytes).hexdigest(),
        "csv_content_sha256": digest.hexdigest(),
        "dataset": dataset,
        "source_data_date_range": summary.get("source_data_date_range"),
    }


def _identity(*, root: Path, filter_id: str, architecture: str,
              profile: str, seed: int, target_lineage: dict, origin_bundle=None) -> str:
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy

    source = _verified_current_inputs(root, filter_id, origin_bundle=origin_bundle)
    date_range = source.get("source_data_date_range") or {}
    if not date_range.get("end"):
        raise ValueError("Real-path diagnostic missing Dataset date boundary")
    outer_policy = resolve_breakout_quality_outer_policy(
        root, source_data_end_date=str(date_range["end"]))
    if origin_bundle is not None and origin_bundle.outer_policy != outer_policy:
        raise ValueError("Canonical real-path bundle PIT policy changed before publication")
    return _digest_json({
        "schema": _SCHEMA,
        "source": source,
        "outer_policy": outer_policy,
        "filter_id": filter_id, "architecture": architecture, "profile": profile,
        "seed": seed,
        "teacher": target_lineage,
        "execution_sha256": _implementation_digest(),
        "data_provider_sha256": _data_implementation_sha256(),
        "runtime": get_continuous_ranker_execution_recipe(profile),
        "model_spec": get_model_spec(architecture),
        "pandas": pd.__version__, "numpy": np.__version__,
    })


def _payload_digest(ticker, dates, labels, raw_r, valid) -> str:
    digest = sha256()
    for values in (ticker, dates, labels, raw_r, valid):
        arr = np.ascontiguousarray(values)
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def publish_real_path_diagnostic(
    *, root: Path, filter_id: str, architecture: str, profile: str,
    seed: int, target_lineage: dict, bundle,
) -> None:
    """Publish a replaceable projection only after canonical Target is built.

    This is a performance hint, not scientific evidence; optional I/O errors
    cannot affect a successful model FIT.  Source mismatches refuse to publish.
    """
    root = Path(root).resolve()
    try:
        identity = _identity(root=root, filter_id=filter_id, architecture=architecture,
                             profile=profile, seed=seed, target_lineage=target_lineage,
                             origin_bundle=bundle)
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        recipe = get_continuous_ranker_execution_recipe(profile)
        if str(bundle.profile.continuous_target_id) != str(recipe.continuous_target_id):
            raise ValueError("Real-path diagnostic bundle does not match canonical Target identity")
        if str(bundle.summary.get("filter_id")) != str(filter_id):
            raise ValueError("Real-path diagnostic bundle does not match canonical filter")
        groups = bundle.group_table
        count = len(groups)
        indices = groups["group_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(indices, np.arange(count, dtype=np.int64)):
            raise ValueError("Real-path diagnostic requires canonical ordered group indices")
        ticker = groups["ticker"].astype(str).to_numpy(dtype=str)
        dates = pd.to_datetime(groups["date"], errors="raise").to_numpy(dtype="datetime64[ns]")
        labels = groups["label"].to_numpy(dtype=np.int64)
        raw_r = np.asarray(
            groups["target_realized_net_r_for_diagnostics"]
            if "target_realized_net_r_for_diagnostics" in groups else bundle.raw_target,
            dtype=np.float64,
        )
        valid = np.asarray(bundle.target_valid, dtype=bool)
        if not (len(ticker) == len(dates) == len(labels) == len(raw_r) == len(valid) == count):
            raise ValueError("Real-path diagnostic invalid Target array lengths")
        path = _slot(root, filter_id, architecture, profile, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.parent / f".{path.stem}.{os.getpid()}.{uuid4().hex}.npz"
        try:
            np.savez_compressed(
                staging, identity=np.asarray(identity), ticker=ticker,
                dates=dates, labels=labels, raw_r=raw_r, valid=valid,
                payload_sha256=np.asarray(_payload_digest(ticker, dates, labels, raw_r, valid)),
            )
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        _LOG.debug("Disposable real-path diagnostic cache not published: %s: %s",
                     type(exc).__name__, exc)


def load_verified_real_path_diagnostic(
    *, root: Path, filter_id: str, architecture: str, profile: str, seed: int,
    target_lineage: dict,
):
    """Return diagnostic-only bundle or None; NEVER permit FIT/replay reuse."""
    root = Path(root).resolve()
    path = _slot(root, filter_id, architecture, profile, seed)
    if not path.is_file():
        return None
    try:
        identity = _identity(root=root, filter_id=filter_id, architecture=architecture,
                             profile=profile, seed=seed, target_lineage=target_lineage)
        with np.load(path, allow_pickle=False) as store:
            if set(store.files) != {"identity", "ticker", "dates", "labels", "raw_r", "valid", "payload_sha256"}:
                return None
            if str(store["identity"].item()) != identity:
                return None
            ticker = np.asarray(store["ticker"], dtype=str)
            dates = np.asarray(store["dates"], dtype="datetime64[ns]")
            labels = np.asarray(store["labels"], dtype=np.int64)
            raw_r = np.asarray(store["raw_r"], dtype=np.float64)
            valid = np.asarray(store["valid"], dtype=bool)
            integrity = str(store["payload_sha256"].item())
        count = len(ticker)
        if (any(x.ndim != 1 or len(x) != count for x in (ticker, dates, labels, raw_r, valid))
                or count == 0 or np.isnat(dates).any()
                or not np.isfinite(raw_r[valid]).all()
                or integrity != _payload_digest(ticker, dates, labels, raw_r, valid)):
            return None
        group_table = pd.DataFrame({
            "group_index": np.arange(count, dtype=np.int64),
            "ticker": ticker, "date": pd.to_datetime(dates), "label": labels,
        })
        # Only these three attributes are consumed by the canonical post-replay
        # join, never by a model/score producer or the portfolio Replay engine.
        return SimpleNamespace(group_table=group_table, raw_target=raw_r, target_valid=valid)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, BadZipFile, zlib.error, EOFError) as exc:
        _LOG.warning("Disposable real-path diagnostic cache rejected: %s: %s",
                     type(exc).__name__, exc)
        return None
