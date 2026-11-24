"""Share fold workers across independent canonical Optimizer producers.

Producer coordinators remain separate processes: their environment scopes and
publication paths retain the ordinary single-seed lifetime. Only complete fold
tasks enter the shared pool, so a study still has exactly one sequential search.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import multiprocessing
from multiprocessing.managers import BaseManager
from pathlib import Path
from threading import Condition
import time

from core.training_performance import resolve_optimizer_independent_seed_fold_global_workers
from services.optimizer.execution_settings import (
    OptimizerExecutionSettings,
    optimizer_process_environment,
    temporary_process_environment,
)


_COORDINATOR_BROKER = None
_COORDINATOR_WORKERS = None
_COORDINATOR_FINALIZATION_HELD = False
_FOLD_TASK_ACTIVE = False


def _execute_isolated_fold(task: dict) -> dict:
    global _FOLD_TASK_ACTIVE
    settings = task.get("execution_settings") or OptimizerExecutionSettings.from_environ()
    # ProcessPool workers are reused across seeds. The explicit snapshot must
    # replace the preceding job's Optimizer environment, including removed keys.
    _FOLD_TASK_ACTIVE = True
    try:
        with temporary_process_environment({**settings.as_environ(), "V16_PORTFOLIO_MAX_WORKERS": "1"}), optimizer_process_environment(settings):
            from services.optimizer.outer_rolling_oos import _run_outer_rolling_oos_fold_task
            return _run_outer_rolling_oos_fold_task(task)
    finally:
        _FOLD_TASK_ACTIVE = False


class _FoldBroker:
    """The manager process is the sole owner of the bounded heavy-worker pool."""

    def __init__(self, max_workers: int):
        self._condition = Condition()
        self._closed = False
        self._max_workers = int(max_workers)
        self._active = 0
        self._tasks = []
        self._pool = ProcessPoolExecutor(
            max_workers=int(max_workers), mp_context=multiprocessing.get_context("spawn")
        )

    def run_fold(self, task: dict) -> dict:
        fields = ("project_root", "output_dir", "log_path", "fold_idx", "fold_count",
                  "oos_year", "oos_period", "selection_period", "selection_start_date",
                  "selection_end_date", "optimizer_seed", "progress_display_contract")
        observed = {key: task.get(key) for key in fields}
        observed["queue_status"] = "QUEUED"
        with self._condition:
            self._tasks.append(observed)
        self.acquire_finalization()
        try:
            with self._condition:
                if self._closed:
                    raise RuntimeError("Independent Optimizer fold queue is closed")
                observed["queue_status"] = "RUNNING"
                future = self._pool.submit(_execute_isolated_fold, task)
            result = future.result()
            with self._condition:
                observed["queue_status"] = "DONE"
            return result
        except BaseException:
            with self._condition:
                observed["queue_status"] = "FAILED"
            raise
        finally:
            self.release_finalization()

    def acquire_finalization(self) -> None:
        with self._condition:
            while self._active >= self._max_workers and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("Independent Optimizer fold queue is closed")
            self._active += 1

    def release_finalization(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def progress(self) -> list[dict]:
        with self._condition:
            return [dict(task) for task in self._tasks]

    def shutdown(self, abort: bool = False) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        if abort:
            # Only this broker's processes are stopped; never inspect or signal
            # unrelated Research processes running elsewhere on the machine.
            from services.optimizer.outer_rolling_parallel_runner import _cancel_parallel_fold_executor_after_failure
            _cancel_parallel_fold_executor_after_failure(self._pool, set())
        else:
            self._pool.shutdown(wait=True, cancel_futures=True)


class _FoldBrokerManager(BaseManager):
    pass


_FoldBrokerManager.register("FoldBroker", _FoldBroker)


class _SharedFoldExecutor:
    """Local Future adapter; its threads perform IPC and never evaluate trials."""

    def __init__(self, broker, max_workers: int):
        self._broker = broker
        self._calls = ThreadPoolExecutor(max_workers=int(max_workers))

    def submit(self, function, task):
        from services.optimizer.outer_rolling_oos import _run_outer_rolling_oos_fold_task
        if function is not _run_outer_rolling_oos_fold_task:
            raise ValueError("The shared Optimizer queue accepts canonical fold tasks only")
        return self._calls.submit(self._broker.run_fold, dict(task))

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._calls.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._broker.shutdown(abort=True)
        self.shutdown(wait=True, cancel_futures=exc_type is not None)


def independent_fold_queue_workers() -> int | None:
    return _COORDINATOR_WORKERS


def independent_fold_preparation_bundle() -> dict | None:
    """Keep post-fold preparation inside its already-reserved heavy slot."""
    if _COORDINATOR_BROKER is not None or _FOLD_TASK_ACTIVE:
        return {"executor_kind": "inline", "max_workers": 1, "pool_start_method": "inline"}
    return None


def acquire_independent_finalization_slot() -> None:
    global _COORDINATOR_FINALIZATION_HELD
    if _COORDINATOR_BROKER is not None and not _COORDINATOR_FINALIZATION_HELD:
        _COORDINATOR_BROKER.acquire_finalization()
        _COORDINATOR_FINALIZATION_HELD = True


def release_independent_finalization_slot() -> None:
    global _COORDINATOR_FINALIZATION_HELD
    if _COORDINATOR_FINALIZATION_HELD:
        _COORDINATOR_BROKER.release_finalization()
        _COORDINATOR_FINALIZATION_HELD = False


def independent_fold_executor(*, fold_count: int):
    if _COORDINATOR_BROKER is None:
        raise RuntimeError("No independent Optimizer fold queue is active")
    return _SharedFoldExecutor(_COORDINATOR_BROKER, max(1, int(fold_count)))


def _run_producer(producer, request: dict, broker, max_workers: int, environment: dict, log_path: str):
    global _COORDINATOR_BROKER, _COORDINATOR_WORKERS
    _COORDINATOR_BROKER, _COORDINATOR_WORKERS = broker, int(max_workers)
    try:
        # Each coordinator gets the caller's exact process environment before
        # entering the unmodified canonical producer's scoped overrides.
        with open(log_path, "w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log), temporary_process_environment({**environment, "V16_PORTFOLIO_MAX_WORKERS": "1"}):
                return producer(**request)
    finally:
        release_independent_finalization_slot()
        _COORDINATOR_BROKER, _COORDINATOR_WORKERS = None, None


def _progress_lines(tasks: list[dict], *, seeds: int, workers: int, started: float) -> list[str]:
    from services.optimizer.outer_rolling_parallel_progress import _read_latest_parallel_fold_progress
    from services.optimizer.outer_rolling_progress import (
        _collect_optimizer_progress_phase_metrics, _effective_resume_total_elapsed,
        _format_parallel_fold_progress_line, format_optimizer_progress_performance_header,
    )
    tasks = sorted(tasks, key=lambda task: (int(task.get("optimizer_seed") or 0), int(task.get("fold_idx") or 0)))
    progresses = [_read_latest_parallel_fold_progress(str(task.get("log_path") or "")) for task in tasks]
    # Single-seed logs all use seed_index=0. Give the display-only observations
    # unique keys so equal fold numbers across producers cannot overwrite one
    # another in the canonical telemetry collector.
    metrics = _collect_optimizer_progress_phase_metrics(
        {**item, "fold_idx": index + 1, "seed_index": 0, "seed_ensemble_member_index": 0}
        for index, item in enumerate(progresses)
    )
    elapsed = _effective_resume_total_elapsed(
        time.perf_counter() - started,
        search_wall_elapsed_sec=metrics.get("search_wall_elapsed_sec"),
        local_min_wall_elapsed_sec=metrics.get("local_min_wall_elapsed_sec"),
        replay_wall_elapsed_sec=None,
    )
    header = format_optimizer_progress_performance_header(
        folds=max((int(task.get("fold_count") or 0) for task in tasks), default=0),
        seeds=seeds, min_agree=None, parallel_workers=workers, backend="process",
        total_elapsed_sec=elapsed,
        completed_trials=metrics.get("completed_trials"),
        search_wall_elapsed_sec=metrics.get("search_wall_elapsed_sec"),
        completed_local_min_trials=metrics.get("completed_local_min_trials"),
        local_min_wall_elapsed_sec=metrics.get("local_min_wall_elapsed_sec"),
    )
    header += f" | completed_jobs={sum(task['queue_status'] == 'DONE' for task in tasks)}/{len(tasks)}"
    lines = [f"Independent seed × fold shared queue | workers={workers}", header]
    for task, progress in zip(tasks, progresses):
        progress = progress or {"stage": task["queue_status"], "status": task["queue_status"]}
        lines.append(f"seed={task.get('optimizer_seed')} " + _format_parallel_fold_progress_line(task, progress))
    return lines


def run_independent_optimizer_producers_shared_fold_queue(
    producer, requests: list[dict], *, max_workers: int | None = None
) -> list:
    """Run independent producers and return results in declared request order.

    ``producer`` must be a top-level pickle-safe callable accepting each request
    as keyword arguments. Callers validate dependency authorization/identity
    before dispatch. This scheduler neither chooses seeds nor publishes truth.
    """
    import os

    requests = [dict(request) for request in requests]
    if not requests:
        return []
    if independent_fold_queue_workers() is not None:
        raise RuntimeError("Nested independent Optimizer producer queues are not supported")
    workers = (
        resolve_optimizer_independent_seed_fold_global_workers()
        if max_workers is None else int(max_workers)
    )
    if workers < 1:
        raise ValueError("Independent Optimizer fold workers must be positive")
    from core.display_common import FixedProgressBlock
    context = multiprocessing.get_context("spawn")
    manager = None
    broker = None
    coordinators = None
    futures = {}
    completed = False
    progress = FixedProgressBlock()
    started = time.perf_counter()
    try:
        manager = _FoldBrokerManager(ctx=context)
        manager.start()
        broker = manager.FoldBroker(workers)
        coordinators = ProcessPoolExecutor(max_workers=len(requests), mp_context=context)
        environment = dict(os.environ)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for index, request in enumerate(requests):
            root = Path(request.get("project_root") or os.getcwd())
            log_path = root / "outputs" / "optimizer" / "shared_seed_fold_queue" / run_id / f"producer_{index + 1}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            future = coordinators.submit(_run_producer, producer, request, broker, workers, environment, str(log_path))
            futures[future] = index
        results = [None] * len(requests)
        pending = set(futures)
        last_render = 0.0
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=futures.get):
                results[futures[future]] = future.result()
            now = time.perf_counter()
            if done or now - last_render >= 5.0:
                progress.update(_progress_lines(broker.progress(), seeds=len(requests), workers=workers, started=started))
                last_render = now
        completed = True
        return results
    finally:
        try:
            if broker is not None:
                broker.shutdown(abort=not completed)
        finally:
            if coordinators is not None and not completed:
                from services.optimizer.outer_rolling_parallel_runner import _cancel_parallel_fold_executor_after_failure
                _cancel_parallel_fold_executor_after_failure(coordinators, set(futures))
            if coordinators is not None:
                coordinators.shutdown(wait=True, cancel_futures=True)
            shutdown_manager = getattr(manager, "shutdown", None)
            if callable(shutdown_manager):
                shutdown_manager()
            progress.finish()
