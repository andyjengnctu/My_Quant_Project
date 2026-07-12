"""Strict-result CPU inference helpers for breakout quality models."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from filters.breakout_quality.dataset_store import IndexedFeatureBank


def materialize_indexed_feature_inputs(
    features: IndexedFeatureBank,
    context: np.ndarray,
    *,
    enabled: bool,
) -> tuple[IndexedFeatureBank, np.ndarray]:
    """Optionally copy mmap-backed inference inputs to C-contiguous RAM arrays."""

    if not enabled:
        return features, context
    feature_bank_memory = np.array(
        features.feature_bank,
        dtype=np.float32,
        copy=True,
        order="C",
    )
    group_index_memory = np.array(
        features.event_group_index,
        dtype=np.int64,
        copy=True,
        order="C",
    )
    context_memory = np.array(context, dtype=np.float32, copy=True, order="C")
    return IndexedFeatureBank(feature_bank_memory, group_index_memory), context_memory


def strict_parallel_batched_logits(
    torch: Any,
    model: Any,
    features: Any,
    context: np.ndarray,
    *,
    indices: np.ndarray | None,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    """Run fixed-boundary CPU inference while preserving row and reduction order.

    Each batch is identical to the serial path. Independent batches may run on
    read-only model replicas, but their logits are written back to the original
    row positions before downstream metrics or score conversion are performed.
    """

    normalized_batch_size = int(batch_size)
    normalized_workers = int(workers)
    if normalized_batch_size < 1:
        raise ValueError("batch_size 必須 >=1")
    if normalized_workers < 1:
        raise ValueError("workers 必須 >=1")

    if indices is None:
        row_count = int(len(features))
        idx = None
    else:
        idx = np.asarray(indices, dtype=np.int64)
        row_count = int(idx.size)
    if row_count == 0:
        return np.empty((0, 2), dtype=np.float32)

    ranges = [
        (start, min(start + normalized_batch_size, row_count))
        for start in range(0, row_count, normalized_batch_size)
    ]
    worker_count = min(normalized_workers, len(ranges))
    logits_np = np.empty((row_count, 2), dtype=np.float32)
    model.eval()

    def _batch_inputs(start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        if idx is None:
            return (
                features[start:stop],
                np.array(context[start:stop], dtype=np.float32, copy=True),
            )
        batch_idx = idx[start:stop]
        return features[batch_idx], context[batch_idx]

    if worker_count == 1:
        with torch.no_grad():
            for start, stop in ranges:
                batch_features, batch_context = _batch_inputs(start, stop)
                logits_np[start:stop] = model(
                    torch.from_numpy(batch_features),
                    torch.from_numpy(batch_context),
                ).cpu().numpy()
        return logits_np

    assignments: list[list[tuple[int, int]]] = [
        [] for _ in range(worker_count)
    ]
    for job_index, item in enumerate(ranges):
        assignments[job_index % worker_count].append(item)
    replicas = [copy.deepcopy(model).eval() for _ in range(worker_count)]

    def _worker(
        replica: Any,
        assigned_ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int, np.ndarray]]:
        outputs: list[tuple[int, int, np.ndarray]] = []
        with torch.no_grad():
            for start, stop in assigned_ranges:
                batch_features, batch_context = _batch_inputs(start, stop)
                batch_logits = replica(
                    torch.from_numpy(batch_features),
                    torch.from_numpy(batch_context),
                ).cpu().numpy().copy()
                outputs.append((start, stop, batch_logits))
        return outputs

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="breakout-quality-inference",
    ) as executor:
        futures = [
            executor.submit(_worker, replicas[i], assignments[i])
            for i in range(worker_count)
        ]
        for future in futures:
            for start, stop, batch_logits in future.result():
                logits_np[start:stop] = batch_logits
    return logits_np


__all__ = [
    "materialize_indexed_feature_inputs",
    "strict_parallel_batched_logits",
]
