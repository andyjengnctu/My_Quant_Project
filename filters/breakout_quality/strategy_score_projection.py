"""Runtime-consumed score projection identities for Strategy Compare reuse.

Physical score artifacts may gain auxiliary columns without changing an existing arm's
replay inputs.  This module hashes only the score columns a configured arm can consume,
while keeping ticker/date membership in every column projection.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_selection_point_in_time_score_table_from_path,
)

SCORE_PROJECTION_SCHEMA_VERSION = 1
SELECTION_PIT_PRIMARY_SCORE_COLUMN = "breakout_quality_score"
CONTINUOUS_PRIMARY_SCORE_COLUMN = "model_score"


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
    safety_score_column = str(options.get("safety_score_column") or "").strip()
    if safety_dl_id and safety_score_column:
        requirements.setdefault(safety_dl_id, set()).add(safety_score_column)

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


def _score_file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


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
    for (ticker, date_text), value in zip(table.index, values, strict=True):
        digest.update(str(ticker).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(date_text).encode("ascii"))
        digest.update(b"\0")
        digest.update(float(value).hex().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().lower()


@lru_cache(maxsize=12)
def _selection_pit_projection_cached(
    score_path: str,
    signature: tuple[int, int],
    columns: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    del signature
    table = load_selection_point_in_time_score_table_from_path(score_path)
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
) -> dict[str, str]:
    """Hash available requested score columns without requiring unrelated auxiliaries."""

    path = Path(score_path).resolve()
    requested = tuple(sorted({str(column).strip() for column in columns if str(column).strip()}))
    if not path.is_file() or not requested:
        return {}
    if str(score_source or "").strip() != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        # Keep non-PIT sources on their existing whole-artifact identity until a
        # dedicated migration is scientifically required.
        return {}
    return dict(
        _selection_pit_projection_cached(
            str(path),
            _score_file_signature(path),
            requested,
        )
    )


__all__ = [
    "SCORE_PROJECTION_SCHEMA_VERSION",
    "compute_score_projection_sha256s",
    "primary_score_column_for_source",
    "required_score_columns_for_dl",
    "runtime_score_projection_requirements",
]
