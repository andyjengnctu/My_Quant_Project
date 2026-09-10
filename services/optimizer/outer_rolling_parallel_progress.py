"""Outer-rolling parallel-fold progress log IO and diagnostics.

This module owns log filtering, compact failure diagnostics, progress/result
event reads, replay-phase timing extraction, and live-result row merging. It
does not own fold execution, selection, replay semantics, or artifact identity.
"""

from __future__ import annotations

import os

from services.optimizer.outer_rolling_fold_context import normalize_optimizer_seed_ensemble_fold_row
from services.optimizer.outer_rolling_progress import (
    PARALLEL_FOLD_PROGRESS_PREFIX,
    _safe_progress_float,
    _safe_progress_json_loads,
    _safe_progress_ts,
)
from services.optimizer.outer_rolling_runtime import _format_exception_summary, _tail_text_file


PARALLEL_FOLD_LOG_STATUS_MAX_CHARS = 140


class _ParallelFoldProgressLogFilter:
    def __init__(self, handle):
        self.handle = handle
        self._buffer = ""

    def write(self, text):
        if not text:
            return 0
        raw = str(text)
        self._buffer += raw
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
                self.handle.write(line + "\n")
                self.handle.flush()
        return len(raw)

    def flush(self):
        if self._buffer.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
            self.handle.write(self._buffer)
            self._buffer = ""
        self.handle.flush()


def _parallel_fold_log_path_for_fallback(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return raw
    root, ext = os.path.splitext(raw)
    return f"{root}_fallback{ext or '.log'}"


def _build_parallel_fold_failure_message(*, task: dict, exc: BaseException, label: str = "parallel fold failed") -> str:
    fold_idx = int((task or {}).get("fold_idx", 0) or 0)
    fold_count = int((task or {}).get("fold_count", 0) or 0)
    oos_year = int((task or {}).get("oos_year", 0) or 0)
    log_path = str((task or {}).get("log_path") or "")
    tail = _tail_text_file(log_path, max_lines=24)
    detail = f"\n最後 fold log：\n{tail}" if tail else ""
    return f"{label}: fold={fold_idx}/{fold_count} OOS={oos_year} error={_format_exception_summary(exc)} log={log_path}{detail}"


def _latest_parallel_fold_log_status(path: str, *, max_chars: int = PARALLEL_FOLD_LOG_STATUS_MAX_CHARS) -> str:
    tail = _tail_text_file(path, max_lines=12)
    if not tail:
        return ""
    for line in reversed(tail.splitlines()):
        text = str(line).strip()
        if not text:
            continue
        text = " ".join(text.split())
        if len(text) > int(max_chars):
            text = text[: max(0, int(max_chars) - 3)] + "..."
        return text
    return ""


def _read_latest_parallel_fold_progress(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return {}
    for line in reversed(lines[-240:]):
        raw = str(line).strip()
        if not raw.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
            continue
        payload = _safe_progress_json_loads(raw.split("\t", 1)[1])
        if payload:
            return payload
    return {}


def _read_parallel_fold_result_row(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    for line in reversed(lines[-400:]):
        raw = str(line).strip()
        if not raw.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
            continue
        payload = _safe_progress_json_loads(raw.split("\t", 1)[1])
        if str(payload.get("stage") or "").upper() != "FOLD_RESULT":
            continue
        result_row = payload.get("result_row")
        if isinstance(result_row, dict) and result_row:
            return dict(result_row)
    return None


def _read_parallel_fold_replay_phase_metrics(path: str) -> dict:
    """Read fold-level policy replay throughput metrics from a fold log.

    The replay display metric uses the same wall-clock throughput definition as
    search/local-min: first replay event timestamp to last completed replay event
    timestamp, divided by completed replay units.  The completed count is the
    latest replay_done value for this fold, not the number of log events.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return {}
    started: list[float] = []
    done_ts: list[float] = []
    completed = 0
    for line in lines[-800:]:
        raw = str(line).strip()
        if not raw.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
            continue
        payload = _safe_progress_json_loads(raw.split("\t", 1)[1])
        if str(payload.get("stage") or "").upper() != "ENSEMBLE_REPLAY":
            continue
        ts = _safe_progress_ts(payload)
        replay_started = _safe_progress_float(payload, "replay_started_ts")
        if replay_started is not None:
            started.append(float(replay_started))
        elif ts is not None:
            started.append(float(ts))
        try:
            replay_done = int(payload.get("replay_done", 0) or 0)
        except (TypeError, ValueError):
            replay_done = 0
        if replay_done > completed:
            completed = int(replay_done)
        replay_done_ts = _safe_progress_float(payload, "replay_last_done_ts")
        if replay_done_ts is not None:
            done_ts.append(float(replay_done_ts))
        elif replay_done > 0 and ts is not None:
            done_ts.append(float(ts))
    result = {"completed_replays": int(max(0, completed))}
    if started:
        result["replay_started_ts"] = min(started)
    if done_ts:
        result["replay_last_done_ts"] = max(done_ts)
    if started and done_ts:
        result["replay_wall_elapsed_sec"] = max(0.0, max(done_ts) - min(started))
    return result


def _collect_parallel_fold_replay_phase_metrics(tasks: list[dict]) -> dict:
    completed_replays = 0
    replay_started: list[float] = []
    replay_done: list[float] = []
    for task in list(tasks or []):
        metrics = _read_parallel_fold_replay_phase_metrics(str((task or {}).get("log_path") or ""))
        completed_replays += int(metrics.get("completed_replays", 0) or 0)
        if metrics.get("replay_started_ts") is not None:
            replay_started.append(float(metrics["replay_started_ts"]))
        if metrics.get("replay_last_done_ts") is not None:
            replay_done.append(float(metrics["replay_last_done_ts"]))
    replay_span = None
    if replay_started and replay_done:
        replay_span = max(0.0, max(replay_done) - min(replay_started))
    return {
        "completed_replays": int(completed_replays),
        "replay_wall_elapsed_sec": replay_span,
    }


def _merge_live_result_rows(completed_rows: list[dict], tasks: list[dict]) -> list[dict]:
    rows_by_oos: dict[int, dict] = {}
    for row in list(completed_rows or []):
        if not row:
            continue
        normalized = normalize_optimizer_seed_ensemble_fold_row(row)
        oos_key = int(normalized.get("oos_year", 0) or 0)
        if oos_key > 0:
            rows_by_oos[oos_key] = normalized
    for task in list(tasks or []):
        row = _read_parallel_fold_result_row(str((task or {}).get("log_path") or ""))
        if not row:
            continue
        normalized = normalize_optimizer_seed_ensemble_fold_row(row, context=task)
        oos_key = int(normalized.get("oos_year", 0) or 0)
        if oos_key > 0 and oos_key not in rows_by_oos:
            rows_by_oos[oos_key] = normalized
    return [rows_by_oos[key] for key in sorted(rows_by_oos)]
