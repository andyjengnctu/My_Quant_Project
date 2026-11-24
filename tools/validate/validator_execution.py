"""Bounded, spawn-safe execution of the canonical synthetic registry.

Only execution is partitioned: every validator runs once, and receipts are
assembled in registry order. No previous result or coverage file is reused.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import multiprocessing
from multiprocessing.connection import wait
from numbers import Integral, Real
import os
from pathlib import Path
import sys
import time
import traceback


class SyntheticExecutionError(RuntimeError):
    """Failure with completed diagnostics; never a successful partial suite."""

    def __init__(self, message, *, results=(), summaries=()):
        super().__init__(message)
        self.completed_results = list(results)
        self.completed_summaries = list(summaries)


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
    # are not rebuilt by every worker. Balance declared work rather than case
    # counts: a model-fit contract costs much more than a scalar assertion.
    # These relative hints affect scheduling only, never execution or verdicts.
    # Restore registry order inside each batch; no previous result is reused.
    weights = {"fast": 1, "medium": 5, "heavy": 25}
    owners = {}
    for index, spec in enumerate(specs):
        owners.setdefault(spec["module"], []).append((index, spec))
    batches = [[] for _ in range(min(workers, len(owners)))]
    loads = [0] * len(batches)
    def cost(group):
        return sum(weights.get(spec.get("cost_class", "fast"), 1) for _, spec in group)
    for group in sorted(owners.values(), key=lambda items: (-cost(items), items[0][0])):
        target = min(range(len(batches)), key=lambda i: (loads[i], len(batches[i]), i))
        batches[target].extend(group)
        loads[target] += cost(group)
    return [sorted(batch, key=lambda item: item[0]) for batch in batches]


def _snapshot_report_value(value):
    """Detach evaluated diagnostics from process-local/live Python objects.

    Assertion verdicts have already been computed by the validator. Keep plain
    values and container shapes; render other report cells just as CSV/Excel
    would. In particular, never pickle callables retained across module reloads.
    """
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, dict):
        return {_snapshot_report_value(key): _snapshot_report_value(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        container = next(kind for kind in (list, tuple, set, frozenset) if isinstance(value, kind))
        return container(_snapshot_report_value(item) for item in value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    # NumPy booleans are not numbers.Integral. Do not import NumPy merely to
    # transport reports; any validator returning its scalars already loaded it.
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, numpy.bool_):
        return bool(value)
    return str(value)


def run_entries(base_params, entries):
    """The same per-case execution/diagnostics for serial and spawned runs."""
    results, summaries = [], []
    for entry in entries:
        started = time.perf_counter()
        try:
            rows, summary = entry.validator(base_params)
            # Snapshot before the next validator can rebind imports or mutate
            # shared diagnostics. Serial and spawn use the same report values.
            rows = _snapshot_report_value(rows)
            summary = _snapshot_report_value(dict(summary))
        except BaseException as exc:
            if isinstance(exc, Exception):
                raise SyntheticExecutionError(
                    f"synthetic validator crashed: {entry.name}: {type(exc).__name__}: {exc}",
                    results=results, summaries=summaries,
                ) from exc
            # Preserve cancellation/exit semantics while retaining earlier cases.
            exc.completed_results = list(results)
            exc.completed_summaries = list(summaries)
            raise
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


def _run_batch(base_params, batch, coverage_options, data_file, threads, on_receipt=None):
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
                receipt = (index, rows, summaries[0])
                receipts.append(receipt)
                if on_receipt is not None:
                    on_receipt(receipt)
            return receipts
        finally:
            if cov is not None:
                cov.stop()
                cov.save()


def _worker(send, base_params, batch, coverage_options, data_file, threads):
    try:
        _run_batch(base_params, batch, coverage_options, data_file, threads,
                   on_receipt=lambda receipt: send.send({"receipt": receipt}))
        send.send({"completed": True})
    except BaseException as exc:
        # SystemExit/KeyboardInterrupt in a validator must not look like an
        # empty successful batch. The parent retains the original traceback.
        send.send({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        send.close()


def assemble_results(specs, receipts, *, require_complete=True):
    by_index = {}
    for index, rows, summary in receipts:
        if type(index) is not int or not 0 <= index < len(specs) or index in by_index:
            raise RuntimeError(f"invalid/duplicate synthetic receipt index: {index!r}")
        if summary.get("validator_name") != specs[index]["name"]:
            raise RuntimeError(f"synthetic receipt identity mismatch: {index}")
        by_index[index] = (rows, summary)
    if require_complete and len(by_index) != len(specs):
        raise RuntimeError("incomplete synthetic receipts")
    results, summaries = [], []
    for index in sorted(by_index):
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
                    raise RuntimeError("synthetic worker exited before batch completion") from exc
                if "receipt" in message:
                    candidate = [*receipts, message["receipt"]]
                    assemble_results(specs, candidate, require_complete=False)
                    receipts = candidate
                    continue
                receive.close()
                del readers[receive]
                if "error" in message:
                    raise RuntimeError(f"synthetic worker failed: {message['error']}\n{message['traceback']}")
                if message.get("completed") is not True:
                    raise RuntimeError("invalid synthetic worker completion")
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
    except BaseException as exc:
        # Preserve the original failure type (including timeout/cancellation).
        # Only already validated, streamed receipts are diagnostic evidence.
        exc.completed_results, exc.completed_summaries = assemble_results(
            specs, receipts, require_complete=False,
        )
        raise
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
