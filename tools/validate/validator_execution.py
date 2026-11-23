"""Bounded, spawn-safe execution of the canonical synthetic registry.

Only execution is partitioned: every validator runs once, and receipts are
assembled in registry order. No previous result or coverage file is reused.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback


def worker_count(limit: int, case_count: int, *, cpu_count: int | None = None, threads: int = 1) -> int:
    if type(limit) is not int or limit < 1:
        raise ValueError("synthetic_workers must be a positive integer")
    cpus = (os.cpu_count() or 1) if cpu_count is None else max(1, cpu_count)
    # Leave capacity for the other formal producers; a small host stays serial.
    if type(threads) is not int or threads < 1:
        raise ValueError("test_cpu_threads must be a positive integer")
    available = max(1, cpus // threads - 3)
    return max(1, min(limit, case_count, available))


def describe_entries(entries):
    specs = [dict(name=entry.name, module=entry.validator.__module__,
                  layer=entry.layer, cost_class=entry.cost_class) for entry in entries]
    names = [spec["name"] for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate synthetic validator identity")
    return specs


def plan_batches(specs, workers):
    # Keep each owner together so static indexes and expensive fixture imports
    # are not rebuilt by every worker. Assign larger owners first, then restore
    # registry order inside each batch. Scheduling never uses previous PASSes.
    owners = {}
    for index, spec in enumerate(specs):
        owners.setdefault(spec["module"], []).append((index, spec))
    batches = [[] for _ in range(min(workers, len(owners)))]
    for group in sorted(owners.values(), key=lambda items: (-len(items), items[0][0])):
        min(batches, key=len).extend(group)
    return [sorted(batch, key=lambda item: item[0]) for batch in batches]


def run_entries(base_params, entries):
    """The same per-case execution/diagnostics for serial and spawned runs."""
    results, summaries = [], []
    for entry in entries:
        started = time.perf_counter()
        try:
            rows, summary = entry.validator(base_params)
        except Exception as exc:
            raise RuntimeError(
                f"synthetic validator crashed: {entry.name}: {type(exc).__name__}: {exc}"
            ) from exc
        duration_sec = round(time.perf_counter() - started, 6)
        results.extend(rows)
        summaries.append({**dict(summary), "validator_name": entry.name,
                          "layer": entry.layer, "cost_class": entry.cost_class,
                          "duration_sec": duration_sec})
    return results, summaries


@contextmanager
def _worker_thread_budget(count):
    # Set native budgets before importing validator dependencies. This process
    # is private to this batch; the parent/Research environment is untouched.
    from tools.local_regression.common import test_cpu_thread_environment
    budget = test_cpu_thread_environment({"test_cpu_threads": count})
    original_env = {name: os.environ.get(name) for name in budget}
    os.environ.update(budget)
    torch = sys.modules.get("torch")
    previous = torch.get_num_threads() if torch is not None else None
    try:
        if torch is not None:
            torch.set_num_threads(count)
        yield
    finally:
        if previous is not None:
            torch.set_num_threads(previous)
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_batch(base_params, batch, coverage_options, data_file, threads):
    from types import SimpleNamespace

    with _worker_thread_budget(threads):
        # Match the existing runner: validator imports precede measurement.
        entries = [SimpleNamespace(**spec, validator=getattr(
            importlib.import_module(spec["module"]), spec["name"])) for _, spec in batch]
        cov = None
        if coverage_options is not None:
            import coverage
            cov = coverage.Coverage(data_file=data_file, **coverage_options)
        if cov is not None:
            cov.start()
        try:
            receipts = []
            for (index, _spec), entry in zip(batch, entries):
                rows, summaries = run_entries(base_params, [entry])
                receipts.append((index, rows, summaries[0]))
            return receipts
        finally:
            if cov is not None:
                cov.stop()
                cov.save()


def _worker(send, base_params, batch, coverage_options, data_file, threads):
    try:
        receipts = _run_batch(base_params, batch, coverage_options, data_file, threads)
        send.send({"receipts": receipts})
    except BaseException as exc:
        # SystemExit/KeyboardInterrupt in a validator must not look like an
        # empty successful batch. The parent retains the original traceback.
        send.send({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        send.close()


def assemble_results(specs, receipts):
    by_index = {}
    for index, rows, summary in receipts:
        if type(index) is not int or not 0 <= index < len(specs) or index in by_index:
            raise RuntimeError(f"invalid/duplicate synthetic receipt index: {index!r}")
        if summary.get("validator_name") != specs[index]["name"]:
            raise RuntimeError(f"synthetic receipt identity mismatch: {index}")
        by_index[index] = (rows, summary)
    if len(by_index) != len(specs):
        raise RuntimeError("incomplete synthetic receipts")
    results, summaries = [], []
    for index in range(len(specs)):
        rows, summary = by_index[index]
        results.extend(rows)
        summaries.append(summary)
    return results, summaries


def merge_coverage(cov, data_files):
    """Read each current batch explicitly: missing/corrupt coverage is fatal."""
    if not data_files:
        return
    from coverage import CoverageData

    for path in data_files:
        if not Path(path).is_file():
            raise RuntimeError(f"synthetic worker coverage is missing: {Path(path).name}")
        data = CoverageData(basename=path)
        data.read()
        if not data.has_arcs():
            raise RuntimeError("synthetic worker branch coverage is missing")
        cov.get_data().update(data)


def run_parallel(base_params, specs, *, workers, timeout, threads,
                 coverage_options=None, coverage_dir=None):
    if type(workers) is not int or workers < 1 or timeout <= 0 or threads < 1:
        raise ValueError("invalid synthetic execution budget")
    if not specs:
        return [], [], []
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("duplicate synthetic validator identity")
    if coverage_options is not None and coverage_dir is None:
        raise ValueError("coverage requires an isolated batch directory")
    context = multiprocessing.get_context("spawn")
    batches = plan_batches(specs, workers)
    processes, readers, data_files, receipts = [], {}, [], []
    deadline = time.monotonic() + timeout
    try:
        for index, batch in enumerate(batches):
            data_file = None
            if coverage_options is not None:
                data_file = str(Path(coverage_dir) / f".coverage.synthetic.{index}")
                if Path(data_file).exists():
                    raise RuntimeError(f"coverage batch must be fresh: {index}")
                data_files.append(data_file)
            receive, send = context.Pipe(duplex=False)
            process = context.Process(target=_worker,
                args=(send, base_params, batch, coverage_options, data_file, threads))
            try:
                process.start()
            except BaseException:
                receive.close()
                raise
            finally:
                send.close()
            processes.append(process)
            readers[receive] = process
        while readers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"synthetic workers exceeded {timeout}s")
            for receive in wait(list(readers), timeout=min(remaining, 0.1)):
                try:
                    message = receive.recv()
                except EOFError as exc:
                    raise RuntimeError("synthetic worker exited without a receipt") from exc
                receive.close()
                del readers[receive]
                if "error" in message:
                    raise RuntimeError(f"synthetic worker failed: {message['error']}\n{message['traceback']}")
                receipts.extend(message["receipts"])
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
            if process.is_alive():
                raise TimeoutError("synthetic worker did not shut down")
            if process.exitcode != 0:
                raise RuntimeError(f"synthetic worker exit code: {process.exitcode}")
        if any(not Path(path).is_file() for path in data_files):
            raise RuntimeError("synthetic worker coverage is missing")
        results, summaries = assemble_results(specs, receipts)
        return results, summaries, data_files
    finally:
        # Bound cleanup too: a stuck child must not outlive a failed formal step.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        for receive in readers:
            receive.close()
