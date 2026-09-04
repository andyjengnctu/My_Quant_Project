"""Strict-result CPU inference helpers for breakout quality models."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from core.breakout_quality_runtime import BREAKOUT_QUALITY_OUTPUT_SCHEMA

from filters.breakout_quality.dataset_store import IndexedFeatureBank
from filters.breakout_quality.market_set import IndexedMarketSetBank, MarketSetBatch
from filters.breakout_quality.torch_runtime import TorchExecutionPlan, autocast_context


def _resolve_output_width(output_head: str | None) -> int:
    return BREAKOUT_QUALITY_OUTPUT_SCHEMA.width_for(output_head)


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


def market_batch_to_torch(torch: Any, batch: MarketSetBatch, device) -> tuple[Any, Any, Any, Any]:
    return (
        torch.from_numpy(batch.sequences).to(device),
        torch.from_numpy(batch.history_mask).to(device),
        torch.from_numpy(batch.valid_stock_mask).to(device),
        torch.from_numpy(batch.event_to_market).to(device),
    )


def forward_breakout_quality_model(
    model: Any,
    feature_tensor: Any,
    context_tensor: Any,
    market_inputs=None,
    *,
    output_head: str | None = None,
):
    requires_market = bool(getattr(model, "requires_market_set", False))
    if requires_market:
        if market_inputs is None:
            raise ValueError("requires_market_set model 缺少 market inputs")
        return model(feature_tensor, context_tensor, market_inputs)
    if market_inputs is not None:
        raise ValueError("非 market-set model 不得收到 market inputs")
    if output_head is not None:
        forward_head = getattr(model, "forward_output_head", None)
        if forward_head is None:
            raise ValueError("requested output head但model未提供forward_output_head")
        return forward_head(feature_tensor, context_tensor, str(output_head))
    return model(feature_tensor, context_tensor)


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
    market_set_bank: IndexedMarketSetBank | None = None,
    market_group_indices: np.ndarray | None = None,
    output_head: str | None = None,
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
    output_width = _resolve_output_width(output_head)
    if row_count == 0:
        return np.empty((0, output_width), dtype=np.float32)
    explicit_market_groups = (
        None if market_group_indices is None else np.asarray(market_group_indices, dtype=np.int64)
    )
    if explicit_market_groups is not None and explicit_market_groups.shape != (row_count,):
        raise ValueError(
            "market_group_indices 長度必須等於 inference row_count: "
            f"market={explicit_market_groups.shape}, rows={row_count}"
        )
    if market_set_bank is not None and explicit_market_groups is None and not isinstance(
        features, IndexedFeatureBank
    ):
        raise TypeError("market-set inference 需要 IndexedFeatureBank 或 explicit market_group_indices")
    if market_set_bank is None and bool(getattr(model, "requires_market_set", False)):
        raise ValueError("market-set model inference 缺少 IndexedMarketSetBank")

    local_positions = np.arange(row_count, dtype=np.int64)
    if market_set_bank is None:
        jobs = [
            local_positions[start:start + normalized_batch_size]
            for start in range(0, row_count, normalized_batch_size)
        ]
    else:
        if explicit_market_groups is not None:
            all_group_indices = explicit_market_groups
        else:
            assert isinstance(features, IndexedFeatureBank)
            source_rows = local_positions if idx is None else idx
            all_group_indices = np.asarray(
                features.event_group_index[source_rows], dtype=np.int64
            )
        market_dates = market_set_bank.market_date_indices_for_group_indices(
            all_group_indices
        )
        unique_dates = np.unique(market_dates)
        date_to_block = {
            int(date_index): int(position // market_set_bank.max_dates_per_batch)
            for position, date_index in enumerate(unique_dates.tolist())
        }
        positions_by_block: dict[int, list[int]] = {}
        for position, date_index in enumerate(market_dates.tolist()):
            positions_by_block.setdefault(date_to_block[int(date_index)], []).append(position)
        jobs = []
        for block in sorted(positions_by_block):
            positions = np.asarray(positions_by_block[block], dtype=np.int64)
            for start in range(0, len(positions), normalized_batch_size):
                jobs.append(positions[start:start + normalized_batch_size])

    device_type = "cpu" if execution_plan is None else execution_plan.device_type
    worker_count = 1 if device_type == "cuda" else min(normalized_workers, len(jobs))
    logits_np = np.empty((row_count, output_width), dtype=np.float32)
    model.eval()

    def _batch_inputs(positions: np.ndarray):
        row_indices = positions if idx is None else idx[positions]
        batch_features = features[row_indices]
        batch_context = np.asarray(context[row_indices], dtype=np.float32)
        market_batch = None
        if market_set_bank is not None:
            if explicit_market_groups is not None:
                group_indices = explicit_market_groups[positions]
            else:
                assert isinstance(features, IndexedFeatureBank)
                group_indices = np.asarray(
                    features.event_group_index[row_indices], dtype=np.int64
                )
            market_batch = market_set_bank.materialize_for_group_indices(group_indices)
        return batch_features, batch_context, market_batch

    if worker_count == 1:
        device = torch.device("cpu") if execution_plan is None else execution_plan.device
        with torch.inference_mode():
            for positions in jobs:
                batch_features, batch_context, market_batch = _batch_inputs(positions)
                feature_tensor = torch.from_numpy(batch_features).to(device)
                context_tensor = torch.from_numpy(batch_context).to(device)
                market_inputs = (
                    None
                    if market_batch is None
                    else market_batch_to_torch(torch, market_batch, device)
                )
                context_manager = (
                    autocast_context(torch, execution_plan)
                    if execution_plan is not None
                    else torch.autocast(device_type="cpu", enabled=False)
                )
                with context_manager:
                    batch_logits = forward_breakout_quality_model(
                        model, feature_tensor, context_tensor, market_inputs,
                        output_head=output_head,
                    )
                logits_np[positions] = batch_logits.float().cpu().numpy()
        return logits_np

    assignments: list[list[np.ndarray]] = [[] for _ in range(worker_count)]
    for job_index, item in enumerate(jobs):
        assignments[job_index % worker_count].append(item)
    replicas = [copy.deepcopy(model).eval() for _ in range(worker_count)]

    def _worker(
        replica: Any,
        assigned_jobs: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        outputs: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.no_grad():
            for positions in assigned_jobs:
                batch_features, batch_context, market_batch = _batch_inputs(positions)
                market_inputs = (
                    None
                    if market_batch is None
                    else market_batch_to_torch(torch, market_batch, torch.device("cpu"))
                )
                batch_logits = forward_breakout_quality_model(
                    replica,
                    torch.from_numpy(batch_features),
                    torch.from_numpy(batch_context),
                    market_inputs,
                    output_head=output_head,
                ).cpu().numpy().copy()
                outputs.append((positions, batch_logits))
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
            for positions, batch_logits in future.result():
                logits_np[positions] = batch_logits
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
    market_set_bank: IndexedMarketSetBank | None = None,
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
        market_set_bank=market_set_bank,
        market_group_indices=np.asarray(unique_group_indices, dtype=np.int64),
    )
    return group_logits, np.asarray(event_to_group, dtype=np.int64)


__all__ = [
    "forward_breakout_quality_model",
    "market_batch_to_torch",
    "materialize_indexed_feature_inputs",
    "strict_parallel_batched_logits",
    "strict_unique_group_batched_logits",
]
