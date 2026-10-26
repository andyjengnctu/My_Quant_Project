"""Durable, content-certified perfect-price Target score panels.

This score producer cache is deliberately independent of portfolio Replay,
selected DailyUniversal horizons and report rendering. Only source-equivalent
score tables can REUSE. Cache absence authorizes a score BUILD, never a
Portfolio Replay; a corrupt cache always BLOCKS rather than rebuilding over it.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Callable

import numpy as np

from core.file_integrity import (
    atomic_replace_with_retry, canonical_json_sha256,
    compute_file_sha256, exclusive_file_lock, is_sha256_hex,
)
from filters.breakout_quality.dataset_store import resolve_dataset_paths
from filters.breakout_quality.paths import resolve_filter_output_dir

_PRICE_SCORE_CACHE_SCHEMA = 1
# Includes pure-price and official daily-DL eligibility capabilities, but not
# portfolio execution, menu, rendering or experiment report source code.
_PRICE_SCORE_SOURCE_MODULES = (
    "filters/breakout_quality/daily_ranker_data.py",
    "filters/breakout_quality/continuous_ranker_data.py",
    "filters/breakout_quality/continuous_target.py",
    "filters/breakout_quality/mfe_safety_geometry.py",
    "filters/breakout_quality/profile_ranker_data.py",
    "filters/breakout_quality/workflow_io.py",
    "filters/breakout_quality/splits.py",
    "filters/breakout_quality/dataset_store.py",
    "filters/breakout_quality/dataset_readiness.py",
    "filters/breakout_quality/contract.py",
    "core/breakout_quality_registry.py",
    "core/breakout_quality_runtime.py",
    "core/breakout_quality_runtime_resolver.py",
    "core/dataset_profiles.py",
    "config/breakout_quality.py",
    "config/research.py",
)


def price_score_source_identity(
    *, root: Path, dataset: str, filter_id: str, price_sha256: str,
    providers: Mapping[str, str], recipes: Mapping[str, tuple[str, str]],
    score_start: str, cutoff: str, score_algorithm_sha256: str,
    label_policy: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Verify source file bytes and form a cache identity without scoring rows.

    ``None`` is allowed only if the formal source dataset summary is absent:
    the canonical producer must then validate or fail on its own. This keeps
    isolated synthetic fixtures independent of a real historical Dataset.
    Existing but incompatible evidence is always an error, not a cache miss.
    """
    if not is_sha256_hex(price_sha256) or not is_sha256_hex(score_algorithm_sha256):
        raise ValueError("Oracle price-score identity requires verified source SHA")
    root = Path(root).resolve()
    paths = resolve_dataset_paths(resolve_filter_output_dir(root, filter_id))
    if not paths.summary.is_file():
        return None
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("dataset") != dataset or summary.get("filter_id") != filter_id:
        raise ValueError("Oracle price-score source dataset summary does not match canonical scope")
    # Certify every declared binary input, not merely summary metadata.
    # Some officially registered daily feature providers also consume market
    # set artifacts; an unchanged event-bank SHA alone is not proof of reuse.
    for field, paths_by_name in (
        ("dataset_artifacts", paths.artifact_paths()),
        ("market_set_artifacts", paths.market_set_artifact_paths()),
    ):
        records = summary.get(field)
        if records is None and field == "market_set_artifacts":
            continue
        if not isinstance(records, dict) or set(records) != set(paths_by_name):
            raise ValueError(f"Oracle price-score source has incomplete {field}")
        for name, path in paths_by_name.items():
            stored = records[name]
            if (not isinstance(stored, dict)
                    or not is_sha256_hex(stored.get("sha256"))
                    or not path.is_file()
                    or compute_file_sha256(path) != stored["sha256"]):
                raise ValueError(f"Oracle price-score upstream {field} SHA differs: {name}")
    source_digest = sha256()
    for relative in _PRICE_SCORE_SOURCE_MODULES:
        source_file = root / relative
        if not source_file.is_file():
            raise ValueError(f"Oracle price-score source implementation unavailable: {relative}")
        source_digest.update(relative.encode("utf-8") + b"\0")
        source_digest.update(compute_file_sha256(source_file).encode("ascii") + b"\n")
    return {
        "schema": _PRICE_SCORE_CACHE_SCHEMA,
        "dataset": dataset, "filter_id": filter_id,
        "price_sha256": price_sha256,
        "dataset_summary_sha256": compute_file_sha256(paths.summary),
        "source_code_sha256": source_digest.hexdigest(),
        "score_algorithm_sha256": score_algorithm_sha256,
        "providers": dict(sorted(providers.items())),
        "recipes": {name: list(spec) for name, spec in sorted(recipes.items())},
        "score_start": score_start, "cutoff": cutoff,
        "label_policy": dict(label_policy),
    }


class OraclePriceScoreStore:
    """One immutable score-panel result per certified provider/algorithm input."""

    def __init__(self, cache_root: Path):
        self.cache_root = Path(cache_root)

    def _path(self, identity: Mapping[str, Any]) -> Path:
        if identity.get("schema") != _PRICE_SCORE_CACHE_SCHEMA:
            raise ValueError("Oracle price-score cache schema invalid")
        return self.cache_root / f"price_scores_{canonical_json_sha256(identity)}.npz"

    def read(
        self, identity: Mapping[str, Any],
        *, score_sha: Callable[[Mapping[tuple[str, str], float]], str],
    ) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, Any]] | None:
        path = self._path(identity)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as obj:
                if set(obj.files) != {"metadata", "tickers", "days", "matrix"}:
                    raise ValueError("Oracle score cache unexpected array fields")
                meta = json.loads(str(obj["metadata"].item()))
                tickers, days, matrix = obj["tickers"], obj["days"], obj["matrix"]
            if (meta.get("schema") != _PRICE_SCORE_CACHE_SCHEMA
                    or meta.get("identity") != dict(identity)):
                raise ValueError("Oracle price-score cache identity mismatch")
            receipt = {field: meta[field] for field in
                       ("schema", "identity", "targets", "cohort", "score_sha256")}
            if canonical_json_sha256(receipt) != meta.get("receipt_sha256"):
                raise ValueError("Oracle price-score evidence receipt hash mismatch")
            targets = meta["targets"]
            if (not isinstance(targets, list) or not targets
                    or not all(isinstance(target, str) for target in targets)
                    or len(set(targets)) != len(targets)
                    or tickers.ndim != 1 or days.ndim != 1 or matrix.ndim != 2
                    or len(tickers) == 0 or len(days) != len(tickers)
                    or matrix.shape != (len(tickers), len(targets))
                    or tickers.dtype.kind != "U" or days.dtype.kind != "U"
                    or matrix.dtype.kind != "f" or not np.isfinite(matrix).all()):
                raise ValueError("Oracle price-score data shape/type/value invalid")
            keys = list(zip(tickers.tolist(), days.tolist()))
            if len(set(keys)) != len(keys):
                raise ValueError("Oracle price-score duplicate stock-day")
            scores = {target: dict(zip(keys, matrix[:, col].tolist()))
                      for col, target in enumerate(targets)}
            if set(scores) != set(meta["score_sha256"]):
                raise ValueError("Oracle price-score target inventory differs")
            if any(score_sha(scores[target]) != meta["score_sha256"][target]
                   for target in targets):
                raise ValueError("Oracle price-score matrix SHA mismatch")
            cohort = meta["cohort"]
            if (cohort.get("cohort_rows") != len(keys)
                    or cohort.get("first_common_day") != min(days)
                    or cohort.get("last_common_day") != max(days)):
                raise ValueError("Oracle price-score cohort evidence mismatch")
            return scores, cohort
        except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError,
                zipfile.BadZipFile) as exc:
            raise ValueError(f"Oracle price-score cache is corrupt; BLOCKED: {path.name}: {exc}") from exc

    def write(
        self, identity: Mapping[str, Any],
        scores: Mapping[str, Mapping[tuple[str, str], float]],
        cohort: Mapping[str, Any],
        *, score_sha: Callable[[Mapping[tuple[str, str], float]], str],
    ) -> None:
        targets = sorted(scores)
        if not targets:
            raise ValueError("Oracle price-score cache requires targets")
        keys = sorted(scores[targets[0]])
        if not keys or any(set(scores[target]) != set(keys) for target in targets):
            raise ValueError("Oracle price-score cache requires one common cohort")
        receipt = {
            "schema": _PRICE_SCORE_CACHE_SCHEMA, "identity": dict(identity),
            "targets": targets, "cohort": dict(cohort),
            "score_sha256": {name: score_sha(scores[name]) for name in targets},
        }
        metadata = json.dumps({**receipt, "receipt_sha256": canonical_json_sha256(receipt)},
                              ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        matrix = np.empty((len(keys), len(targets)), dtype=np.float64)
        for col, target in enumerate(targets):
            matrix[:, col] = np.fromiter(
                (float(scores[target][key]) for key in keys),
                dtype=np.float64, count=len(keys),
            )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        path = self._path(identity)
        fd, temp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=self.cache_root)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, metadata=np.asarray(metadata),
                                    tickers=np.asarray([key[0] for key in keys]),
                                    days=np.asarray([key[1] for key in keys]),
                                    matrix=matrix)
                stream.flush()
                os.fsync(stream.fileno())
            with exclusive_file_lock(path.parent / ".price_scores.publish.lock"):
                existing = self.read(identity, score_sha=score_sha)
                if existing is not None:
                    if existing[0] != {name: dict(values) for name, values in scores.items()} or existing[1] != dict(cohort):
                        raise ValueError("Oracle same-identity price-score source has conflicting outcomes")
                else:
                    atomic_replace_with_retry(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
