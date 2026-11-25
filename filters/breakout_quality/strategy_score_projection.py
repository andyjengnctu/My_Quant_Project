"""Runtime-consumed score projection identities for Strategy Compare reuse.

Physical score artifacts may gain auxiliary columns without changing an existing arm's
replay inputs.  This module hashes only the score columns a configured arm can consume,
while keeping ticker/date membership in every column projection.
"""

from __future__ import annotations

from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from itertools import islice
import json
import logging
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.file_integrity import (
    atomic_write_json, canonical_json_sha256, compute_observed_file_sha256,
    file_change_signature, is_sha256_hex, require_sha256_hex,
)

from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_selection_point_in_time_score_table_from_path,
    register_selection_pit_cache_invalidator,
)

SCORE_PROJECTION_SCHEMA_VERSION = 1
SELECTION_PIT_PRIMARY_SCORE_COLUMN = "breakout_quality_score"
CONTINUOUS_PRIMARY_SCORE_COLUMN = "model_score"
# AI: Bound temporary text to a small block while preserving the v1 byte stream.
_PROJECTION_HASH_BLOCK_ROWS = 8192
_READ_SCOPE = ContextVar("score_projection_read_scope", default=None)


def _projection_implementation_sha256() -> str:
    """Bind derived digests to all local runtime code and parser versions.

    This cache is not a scientific identity. A broad content key deliberately
    invalidates it on any local implementation/config change, without maintaining
    a second, hand-written list of transitive numerical dependencies.
    """
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for package in sorted({"core", "filters", "services", "strategies", "config"}):
        for path in sorted((root / package).rglob("*.py")):
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return canonical_json_sha256({
        "sources":sources, "python":sys.version, "numpy":np.__version__, "pandas":pd.__version__,
    })


@contextmanager
def score_projection_read_scope():
    """Share implementation observation only within one read-only plan."""
    if _READ_SCOPE.get() is not None:
        yield
        return
    state = {}
    token = _READ_SCOPE.set(state)
    try:
        yield
    finally:
        _READ_SCOPE.reset(token)
    if state and _projection_implementation_sha256() != state["implementation"]:
        raise RuntimeError("Score projection implementation changed during inspection")


def _read_projection_cache(path: Path, identity: dict) -> dict[str, str] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Ignoring unreadable derived score cache: %s", type(exc).__name__)
        return None
    if not isinstance(payload, dict):
        return None
    body = {key:value for key,value in payload.items() if key != "sha256"}
    values = body.get("projections")
    if (set(body) != {"identity", "projections"} or body["identity"] != identity
            or payload.get("sha256") != canonical_json_sha256(body)
            or not isinstance(values, dict) or not set(values).issubset(identity["columns"])
            or any(not isinstance(value, str) or value != value.strip().lower()
                   or not is_sha256_hex(value) for value in values.values())):
        return None
    return values


def primary_score_column_for_source(score_source: str) -> str:
    source = str(score_source or "").strip()
    if source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        return SELECTION_PIT_PRIMARY_SCORE_COLUMN
    if source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
        return CONTINUOUS_PRIMARY_SCORE_COLUMN
    return SELECTION_PIT_PRIMARY_SCORE_COLUMN


def runtime_score_projection_requirements(
    *,
    settings_payload: dict[str, Any],
    on_arm_payload: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return score columns consumed by each DL source for one replay arm."""

    dl_sources = dict(settings_payload.get("dl_sources") or {})
    requirements: dict[str, set[str]] = {}
    options = dict(on_arm_payload.get("dl_runtime_options") or {})
    dl_id = str(on_arm_payload.get("dl_id") or "").strip()
    if dl_id:
        source = dict(dl_sources.get(dl_id) or {})
        requirements.setdefault(dl_id, set()).add(
            str(options.get("primary_score_column") or "").strip()
            or primary_score_column_for_source(str(source.get("score_source") or ""))
        )

    safety_dl_id = str(options.get("safety_dl_id") or "").strip()
    from core.rank_score_composition import rank_composition_source_ids
    for source_id in rank_composition_source_ids(options):
        if source_id not in dl_sources:
            raise ValueError("Rank composition score projection lacks a registered source")
        requirements.setdefault(source_id, set()).add(primary_score_column_for_source(
            str(dl_sources[source_id].get("score_source") or "")))
    safety_score_column = str(options.get("safety_score_column") or "").strip()
    if safety_dl_id and safety_score_column:
        requirements.setdefault(safety_dl_id, set()).add(safety_score_column)

    from core.positive_score_admission import requires_positive_score, uses_primary_positive_classifier
    if requires_positive_score(options) and not uses_primary_positive_classifier(options):
        positive_dl_id = str(options["positive_dl_id"])
        if not dl_id or positive_dl_id == dl_id or positive_dl_id not in dl_sources:
            raise ValueError("Positive score column requires a distinct registered classifier DL")
        requirements.setdefault(positive_dl_id, set()).add(str(options["positive_score_column"]))

    return {
        source_id: tuple(sorted(columns))
        for source_id, columns in sorted(requirements.items())
    }


def required_score_columns_for_dl(
    *,
    settings_payload: dict[str, Any],
    arm_payloads: Iterable[dict[str, Any]],
    dl_id: str,
) -> tuple[str, ...]:
    columns: set[str] = set()
    target = str(dl_id or "").strip()
    for arm_payload in arm_payloads:
        for source_id, required in runtime_score_projection_requirements(
            settings_payload=settings_payload,
            on_arm_payload=dict(arm_payload or {}),
        ).items():
            if source_id == target:
                columns.update(required)
    return tuple(sorted(columns))


def _score_file_signature(path: Path, *, expected_sha256: str | None) -> tuple[int, int, str]:
    """Bind projection-cache identity to physical *content*, not file metadata.

    Strategy Compare's artifact planner already computes the physical SHA,
    which is reused as the cache key. A newly loaded table independently
    checks that SHA once; candidate-by-candidate lookups never rehash the CSV.
    """
    stat = path.stat()
    digest = compute_observed_file_sha256(path)
    if expected_sha256 and digest != require_sha256_hex(expected_sha256, field="Selection PIT score sha256"):
        raise ValueError("Selection PIT score在讀取時來源 SHA256 已變更")
    return int(stat.st_mtime_ns), int(stat.st_size), digest


def _projection_digest(*, table, score_column: str) -> str:
    if score_column not in table.columns:
        raise KeyError(score_column)
    digest = hashlib.sha256()
    digest.update(
        f"strategy_score_projection_v{SCORE_PROJECTION_SCHEMA_VERSION}\0{score_column}\n".encode(
            "utf-8"
        )
    )
    values = table[score_column].to_numpy(dtype=np.float64, copy=False)
    # AI: Dates were ASCII in v1. Validate distinct used values before encoding
    # a whole UTF-8 block, so batching cannot silently accept non-ASCII dates.
    if len(values):
        if table.index.nlevels == 2:
            for code, date_text in enumerate(table.index.levels[1]):
                if not str(date_text).isascii() and np.any(table.index.codes[1] == code):
                    str(date_text).encode("ascii")
        else:
            for _ticker, date_text in table.index:
                str(date_text).encode("ascii")
    rows = iter(table.index)
    for start in range(0, len(values), _PROJECTION_HASH_BLOCK_ROWS):
        stop = start + _PROJECTION_HASH_BLOCK_ROWS
        # AI: tolist gives the same Python float as float(np.float64(value));
        # hex retains signed zero, subnormals, NaN/inf and every finite bit.
        block = "".join(
            f"{str(ticker)}\0{str(date_text)}\0{value.hex()}\n"
            for (ticker, date_text), value in zip(
                islice(rows, _PROJECTION_HASH_BLOCK_ROWS), values[start:stop].tolist(), strict=True,
            )
        )
        digest.update(block.encode("utf-8"))
    return digest.hexdigest().lower()


# AI: Cache only tiny digest tuples, not score tables. Keep enough revisions
# for multi-source/seed planning so one pass does not evict the next pass.
@lru_cache(maxsize=256)
def _selection_pit_projection_cached(
    score_path: str,
    signature: tuple[int, int, str],
    columns: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    table = load_selection_point_in_time_score_table_from_path(
        score_path, expected_score_sha256=signature[2],
    )
    output: list[tuple[str, str]] = []
    for column in columns:
        if column not in table.columns:
            continue
        output.append((column, _projection_digest(table=table, score_column=column)))
    return tuple(output)


def compute_score_projection_sha256s(
    *,
    score_path: str | Path,
    score_source: str,
    columns: Iterable[str],
    score_file_sha256: str | None = None,
    projection_cache_dir: str | Path | None = None,
) -> dict[str, str]:
    """Hash available requested score columns without requiring unrelated auxiliaries."""

    if projection_cache_dir is not None and _READ_SCOPE.get() is None:
        with score_projection_read_scope():
            return compute_score_projection_sha256s(
                score_path=score_path, score_source=score_source, columns=columns,
                score_file_sha256=score_file_sha256, projection_cache_dir=projection_cache_dir,
            )
    path = Path(score_path).resolve()
    requested = tuple(sorted({str(column).strip() for column in columns if str(column).strip()}))
    if not path.is_file() or not requested:
        return {}
    if str(score_source or "").strip() != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        # Keep non-PIT sources on their existing whole-artifact identity until a
        # dedicated migration is scientifically required.
        return {}
    before = file_change_signature(path)
    signature = _score_file_signature(path, expected_sha256=score_file_sha256)
    cache_path = None
    if projection_cache_dir is not None:
        scope = _READ_SCOPE.get()
        if scope is None:
            implementation = _projection_implementation_sha256()
        else:
            if "implementation" not in scope:
                scope["implementation"] = _projection_implementation_sha256()
            implementation = scope["implementation"]
        identity = {"schema_version":SCORE_PROJECTION_SCHEMA_VERSION, "score_sha256":signature[2],
                    "implementation":implementation, "columns":list(requested)}
        cache_path = Path(projection_cache_dir) / (canonical_json_sha256(identity) + ".json")
        cached = _read_projection_cache(cache_path, identity)
        if cached is not None:
            if file_change_signature(path) != before:
                raise RuntimeError("Selection PIT score changed during projection inspection")
            return cached
    result = dict(_selection_pit_projection_cached(str(path), signature, requested))
    if file_change_signature(path) != before:
        raise RuntimeError("Selection PIT score changed during projection inspection")
    if cache_path is not None:
        body = {"identity":identity, "projections":result}
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(cache_path, {**body, "sha256":canonical_json_sha256(body)})
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not save derived score cache: %s", type(exc).__name__)
    return result


def clear_selection_point_in_time_projection_cache() -> None:
    """Drop the read-only projection cache when the PIT score owner invalidates."""
    _selection_pit_projection_cached.cache_clear()


# One-way dependency: the consumer registers a cache invalidator with its
# upstream score owner; the owner never imports Strategy Compare.
register_selection_pit_cache_invalidator(clear_selection_point_in_time_projection_cache)


__all__ = [
    "SCORE_PROJECTION_SCHEMA_VERSION",
    "compute_score_projection_sha256s",
    "clear_selection_point_in_time_projection_cache",
    "primary_score_column_for_source",
    "required_score_columns_for_dl",
    "runtime_score_projection_requirements",
]
