"""Measure date-coherent batch planning against its original implementation.

Run from the project root with ``python -m tools.benchmark_research_training_batches``.
Only synthetic inputs are used; no model or research artifacts are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import numpy as np
import pandas as pd

from services.breakout_quality.train_continuous_ranker import _date_coherent_batches


def reference_date_coherent_batches(
    group_ids, group_dates, *, batch_size, seed, single_date_per_batch=False,
):
    """Frozen scalar planner for batch-order and RNG equivalence checks."""
    ids = np.asarray(group_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("date-coherent ranker group_ids必須是一維")
    dates = pd.to_datetime(pd.Series(group_dates), errors="raise").dt.normalize()
    if len(dates) <= int(ids.max(initial=-1)):
        raise ValueError("date-coherent ranker group_dates長度不足")
    by_date = {}
    for group_id in ids:
        by_date.setdefault(pd.Timestamp(dates.iloc[int(group_id)]), []).append(int(group_id))
    date_keys = np.asarray(sorted(by_date), dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(date_keys)
    batches = []
    if bool(single_date_per_batch):
        return [np.asarray(by_date[pd.Timestamp(key)], dtype=np.int64) for key in date_keys]
    current = []
    for date_key in date_keys:
        day_ids = by_date[pd.Timestamp(date_key)]
        if current and len(current) + len(day_ids) > int(batch_size):
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
        if len(day_ids) > int(batch_size):
            if current:
                batches.append(np.asarray(current, dtype=np.int64))
                current = []
            batches.append(np.asarray(day_ids, dtype=np.int64))
        else:
            current.extend(day_ids)
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def batch_digest(batches):
    """Include every boundary as well as row order in the equivalence hash."""
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(int(len(batch)).to_bytes(8, "little"))
        digest.update(np.asarray(batch, dtype="<i8").tobytes())
    return digest.hexdigest()


def measure(*, rows, dates, repeats, seed, batch_size):
    rng = np.random.default_rng(seed)
    date_values = pd.bdate_range("2000-01-03", periods=dates).to_numpy()
    group_dates = date_values[np.arange(rows) % dates]
    ids = rng.permutation(rows)
    timings = {"reference": [], "optimized": []}
    digests = set()
    for repeat in range(repeats):
        functions = [
            ("reference", reference_date_coherent_batches),
            ("optimized", _date_coherent_batches),
        ]
        if repeat % 2:
            functions.reverse()
        for name, function in functions:
            started = time.perf_counter()
            batches = function(ids, group_dates, batch_size=batch_size, seed=seed)
            timings[name].append(time.perf_counter() - started)
            digests.add(batch_digest(batches))
    if len(digests) != 1:
        raise AssertionError("batch boundaries or row order changed")
    baseline = statistics.median(timings["reference"])
    candidate = statistics.median(timings["optimized"])
    return {
        "rows": rows, "dates": dates, "batch_size": batch_size, "seed": seed,
        "repeats": repeats, "seconds": timings,
        "reference_median_sec": baseline, "optimized_median_sec": candidate,
        "speedup": baseline / candidate,
        "batch_count": len(batches), "batch_sha256": digests.pop(),
        "batch_boundaries_and_order_exact": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 500_000, 1_200_000])
    parser.add_argument("--dates", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if min(*args.rows, args.dates, args.repeats, args.batch_size) < 1:
        parser.error("rows, dates, repeats and batch-size must be positive")
    print(json.dumps({
        "benchmark": "research_date_coherent_batch_planning",
        "scope": "batch planning only; not end-to-end training",
        "cases": [measure(rows=rows, dates=args.dates, repeats=args.repeats,
                          seed=args.seed, batch_size=args.batch_size)
                  for rows in args.rows],
    }, indent=2))


if __name__ == "__main__":
    main()
