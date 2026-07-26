"""Strict-result CPU inference helpers for breakout quality models."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from filters.breakout_quality.dataset_store import IndexedFeatureBank
from filters.breakout_quality.torch_runtime import TorchExecutionPlan, autocast_context


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
    execution_plan: TorchExecutionPlan | None = None,
) -> np.ndarray:
    """Run fixed-boundary inference while preserving row and reduction order.

    CUDA uses one model on one device; CPU may use read-only model replicas for
    independent batches. Logits are always written back to the original row
    positions before downstream metrics or score conversion are performed.
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
    device_type = "cpu" if execution_plan is None else execution_plan.device_type
    worker_count = 1 if device_type == "cuda" else min(normalized_workers, len(ranges))
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
        device = torch.device("cpu") if execution_plan is None else execution_plan.device
        with torch.inference_mode():
            for start, stop in ranges:
                batch_features, batch_context = _batch_inputs(start, stop)
                feature_tensor = torch.from_numpy(batch_features).to(device)
                context_tensor = torch.from_numpy(batch_context).to(device)
                context_manager = (
                    autocast_context(torch, execution_plan)
                    if execution_plan is not None
                    else torch.autocast(device_type="cpu", enabled=False)
                )
                with context_manager:
                    batch_logits = model(feature_tensor, context_tensor)
                logits_np[start:stop] = batch_logits.float().cpu().numpy()
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


def strict_unique_group_batched_logits(
    torch: Any,
    model: Any,
    features: IndexedFeatureBank,
    context: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    execution_plan: TorchExecutionPlan | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer each shared feature group once and return event-row broadcast indices.

    This path is only valid for model specs with ``use_dataset_context=False``.
    The returned logits contain one row per used feature group; ``event_to_group``
    maps every original event row to that unique-logit row. Converting logits to
    probabilities before applying this mapping guarantees bit-identical scores
    for every event that shares the same ticker/date feature group.
    """

    if not isinstance(features, IndexedFeatureBank):
        raise TypeError("unique-group inference 需要 IndexedFeatureBank")
    event_group_index = np.asarray(features.event_group_index, dtype=np.int64)
    if event_group_index.ndim != 1:
        raise ValueError("event_group_index 必須是 1D")
    if int(context.shape[0]) != int(event_group_index.size):
        raise ValueError(
            "unique-group inference 的 context/event row_count 不一致: "
            f"context={context.shape[0]}, events={event_group_index.size}"
        )
    if event_group_index.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)

    unique_group_indices, first_event_positions, event_to_group = np.unique(
        event_group_index,
        return_index=True,
        return_inverse=True,
    )
    feature_group_count = int(features.feature_bank.shape[0])
    if int(unique_group_indices[0]) < 0 or int(unique_group_indices[-1]) >= feature_group_count:
        raise ValueError("event_group_index 超出 feature bank 範圍")

    group_features = np.asarray(
        features.feature_bank[unique_group_indices],
        dtype=np.float32,
    )
    # # (AI註: Active sequence-only model 會忽略 context；每個 group 沿用一筆真實代表列，
    # #        只維持 tensor shape 契約，不建立虛構資料。)
    group_context = np.asarray(
        context[first_event_positions],
        dtype=np.float32,
    )
    group_logits = strict_parallel_batched_logits(
        torch,
        model,
        group_features,
        group_context,
        indices=None,
        batch_size=batch_size,
        workers=workers,
        execution_plan=execution_plan,
    )
    return group_logits, np.asarray(event_to_group, dtype=np.int64)


__all__ = [
    "materialize_indexed_feature_inputs",
    "strict_parallel_batched_logits",
    "strict_unique_group_batched_logits",
]
