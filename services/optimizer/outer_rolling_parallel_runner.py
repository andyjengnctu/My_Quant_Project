from __future__ import annotations

from core.display import C_CYAN, C_RESET
from services.optimizer.outer_rolling_formatting import _display_month_period
from services.optimizer.outer_rolling_parallel_progress import _build_parallel_fold_failure_message
from services.optimizer.outer_rolling_results import _render_optimizer_results_tables
from services.optimizer.outer_rolling_runtime import _format_exception_summary

def _print_parallel_fold_result(row: dict) -> None:
    if not row:
        return
    print(f"{C_CYAN}📌 Parallel fold result | fold={row.get('fold')} | selection={_display_month_period(row.get('selection_period'))} | OOS={_display_month_period(row.get('oos_period') or row.get('oos_year'))}{C_RESET}", flush=True)
    rendered = _render_optimizer_results_tables([row], color=True, include_chain=False)
    if rendered:
        print(rendered, flush=True)


def _consume_parallel_fold_result(*, result: dict, rows: list[dict], fold_timing_rows: list[dict], chain_state: dict) -> None:
    result = dict(result or {})
    row = result.get("row")
    if row is not None:
        rows.append(row)
    if result.get("timing_row") is not None:
        fold_timing_rows.append(result["timing_row"])
    if result.get("chain_max_positions") is not None:
        chain_state["chain_max_positions"] = int(result.get("chain_max_positions"))
    if result.get("chain_enable_rotation") is not None:
        chain_state["chain_enable_rotation"] = bool(result.get("chain_enable_rotation"))


def _consume_parallel_fold_future(*, future, task: dict, rows: list[dict], fold_timing_rows: list[dict], chain_state: dict) -> None:
    try:
        result = future.result()
    except Exception as exc:
        raise RuntimeError(_build_parallel_fold_failure_message(task=task, exc=exc)) from exc
    _consume_parallel_fold_result(result=result, rows=rows, fold_timing_rows=fold_timing_rows, chain_state=chain_state)


def _cancel_parallel_fold_executor_after_failure(executor, pending: set) -> None:
    for future in list(pending or []):
        try:
            future.cancel()
        except Exception as exc:
            _format_exception_summary(exc)
    processes = getattr(executor, "_processes", None)
    live_processes = []
    if isinstance(processes, dict):
        for process in list(processes.values()):
            try:
                if process is not None and process.is_alive():
                    live_processes.append(process)
                    process.terminate()
            except Exception as exc:
                _format_exception_summary(exc)
        for process in list(live_processes):
            try:
                process.join(timeout=5.0)
            except Exception as exc:
                _format_exception_summary(exc)
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(wait=False, cancel_futures=True)
        except TypeError:
            try:
                shutdown(wait=False)
            except Exception as exc:
                _format_exception_summary(exc)
        except Exception as exc:
            _format_exception_summary(exc)
