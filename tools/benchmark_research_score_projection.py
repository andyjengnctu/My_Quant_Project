"""Read-only benchmark of the existing v1 score projection byte contract.

Run as ``python -m tools.benchmark_research_score_projection --score-path PATH``.
The canonical loader validates the real score table before timing either digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from core.file_integrity import compute_file_sha256
from filters.breakout_quality.ranking_score_store import load_selection_point_in_time_score_table_from_path
from filters.breakout_quality.strategy_score_projection import (
    SCORE_PROJECTION_SCHEMA_VERSION, _projection_digest,
)


def reference_projection_digest(*, table, score_column):
    """Frozen pre-optimization byte stream; independent scalar equivalence oracle."""
    if score_column not in table.columns:
        raise KeyError(score_column)
    digest = hashlib.sha256()
    digest.update(
        f"strategy_score_projection_v{SCORE_PROJECTION_SCHEMA_VERSION}\0{score_column}\n".encode("utf-8")
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


def measure(score_path, *, score_column, repeats):
    path = Path(score_path).resolve()
    source_sha = compute_file_sha256(path)
    table = load_selection_point_in_time_score_table_from_path(
        str(path), expected_score_sha256=source_sha,
    )
    timings = {"reference": [], "optimized": []}
    digests = set()
    for repeat in range(repeats):
        functions = [("reference", reference_projection_digest), ("optimized", _projection_digest)]
        if repeat % 2:
            functions.reverse()
        for name, function in functions:
            started = time.perf_counter()
            digest = function(table=table, score_column=score_column)
            timings[name].append(time.perf_counter() - started)
            digests.add(digest)
    if len(digests) != 1 or compute_file_sha256(path) != source_sha:
        raise AssertionError("Projection changed or source changed during benchmark")
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "source": path.as_posix(), "source_sha256": source_sha,
        "rows": len(table), "score_column": score_column,
        "repeats": repeats, "seconds": timings, "median_seconds": medians,
        "speedup": medians["reference"] / medians["optimized"],
        "projection_sha256": next(iter(digests)), "exact_digest_match": True,
        "scope": "Validated score table digest only; excludes CSV loading and fitting",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-path", required=True, nargs="+")
    parser.add_argument("--score-column", default="breakout_quality_score")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps([
        measure(path, score_column=args.score_column, repeats=args.repeats)
        for path in args.score_path
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
