from __future__ import annotations

import sys
import time

from core.display import C_CYAN, C_GRAY, C_RESET
from core.runtime_utils import stdout_supports_inline_progress
from core.training_performance import (
    resolve_optimizer_random_seed_ensemble_parallel_backend_default,
    resolve_optimizer_random_seed_ensemble_parallel_workers_default,
)
from services.optimizer.outer_rolling_artifacts import _build_seed_ensemble_policy_payload
from services.optimizer.outer_rolling_fold_context import (
    normalize_optimizer_seed_ensemble_fold_row,
    optimizer_seed_ensemble_row_sort_key,
)
from services.optimizer.outer_rolling_formatting import _fmt_duration
from services.optimizer.outer_rolling_parallel_progress import (
    _collect_parallel_fold_replay_phase_metrics,
    _latest_parallel_fold_log_status,
    _merge_live_result_rows,
    _read_latest_parallel_fold_progress,
)
from services.optimizer.outer_rolling_policy import (
    _active_optimizer_table_titles,
    _is_rolling_random_seed_ensemble_enabled,
)
from services.optimizer.outer_rolling_progress import (
    _collect_seed_progress_phase_metrics,
    _count_local_min_completed_from_progress,
    _count_local_min_completed_neighbors_from_progress,
    _format_parallel_fold_progress_line,
    _read_latest_parallel_fold_seed_progresses_for_task,
    _safe_progress_float,
    _safe_progress_ts,
    _seed_progress_context_key,
    build_optimizer_seed_ensemble_live_lines,
    format_optimizer_seed_ensemble_progress_header,
)
from services.optimizer.outer_rolling_results import _render_optimizer_results_tables

_build_rolling_seed_ensemble_policy_payload = _build_seed_ensemble_policy_payload

class OptimizerSeedEnsembleProgressBoard:
    """Render fold × seed progress lines from one source for rolling and non-rolling."""

    def __init__(self, contexts: list[dict], *, header: str = "", header_factory=None):
        self.contexts = sorted(
            [dict(item) for item in list(contexts or [])],
            key=lambda item: (int(item.get("fold_idx", 0) or 0), int(item.get("seed_index", 0) or 0)),
        )
        self.header = str(header or "")
        self.header_factory = header_factory
        self.inline = stdout_supports_inline_progress()
        self.rendered_lines = 0
        self.progress_by_key: dict[tuple[int, int], dict] = {}
        self.completed_trials_by_key: dict[tuple[int, int], int] = {}
        self.completed_local_min_trials_by_key: dict[tuple[int, int], int] = {}
        self.completed_local_min_neighbors_by_key: dict[tuple[int, int], int] = {}
        self.search_started_ts_by_key: dict[tuple[int, int], float] = {}
        self.search_last_done_ts_by_key: dict[tuple[int, int], float] = {}
        self.local_min_started_ts_by_key: dict[tuple[int, int], float] = {}
        self.local_min_last_done_ts_by_key: dict[tuple[int, int], float] = {}
        self.fold_progress_by_idx: dict[int, dict] = {}
        self.result_rows_by_fold: dict[int, dict] = {}
        self.completed_replays_by_fold: dict[int, int] = {}
        self.replay_started_ts_by_fold: dict[int, float] = {}
        self.replay_last_done_ts_by_fold: dict[int, float] = {}
        self.last_lines: list[str] = []

    def _context_key(self, context: dict) -> tuple[int, int]:
        return (int(context.get("fold_idx", 0) or 0), int(context.get("seed_index", 0) or 0))

    def _record_progress_metrics(self, key: tuple[int, int], progress: dict) -> None:
        data = dict(progress or {})
        if "ts" not in data:
            data["ts"] = time.time()
        try:
            completed = int(data.get("completed", 0) or 0)
        except (TypeError, ValueError):
            completed = 0
        if completed > int(self.completed_trials_by_key.get(key, 0) or 0):
            self.completed_trials_by_key[key] = int(completed)
        local_completed = _count_local_min_completed_from_progress(data)
        if local_completed > int(self.completed_local_min_trials_by_key.get(key, 0) or 0):
            self.completed_local_min_trials_by_key[key] = int(local_completed)
        local_neighbor_completed = _count_local_min_completed_neighbors_from_progress(data)
        if local_neighbor_completed > int(self.completed_local_min_neighbors_by_key.get(key, 0) or 0):
            self.completed_local_min_neighbors_by_key[key] = int(local_neighbor_completed)
        explicit_search_start = _safe_progress_float(data, "search_started_ts")
        if explicit_search_start is None and str(data.get("stage") or "").upper() == "OPTIMIZER_SEARCH" and completed <= 0:
            explicit_search_start = _safe_progress_ts(data)
        if explicit_search_start is not None:
            previous = self.search_started_ts_by_key.get(key)
            self.search_started_ts_by_key[key] = explicit_search_start if previous is None else min(previous, explicit_search_start)
        explicit_search_done = _safe_progress_float(data, "search_last_done_ts")
        if explicit_search_done is None and str(data.get("stage") or "").upper() == "OPTIMIZER_SEARCH" and completed > 0:
            explicit_search_done = _safe_progress_ts(data)
        if explicit_search_done is not None:
            self.search_last_done_ts_by_key[key] = max(float(self.search_last_done_ts_by_key.get(key, 0.0) or 0.0), explicit_search_done)
        explicit_local_start = _safe_progress_float(data, "local_min_started_ts")
        if explicit_local_start is not None:
            previous = self.local_min_started_ts_by_key.get(key)
            self.local_min_started_ts_by_key[key] = explicit_local_start if previous is None else min(previous, explicit_local_start)
        explicit_local_done = _safe_progress_float(data, "local_min_last_neighbor_done_ts")
        if explicit_local_done is None:
            explicit_local_done = _safe_progress_float(data, "local_min_last_done_ts")
        if explicit_local_done is None and local_neighbor_completed > 0:
            explicit_local_done = _safe_progress_ts(data)
        if explicit_local_done is not None:
            self.local_min_last_done_ts_by_key[key] = max(float(self.local_min_last_done_ts_by_key.get(key, 0.0) or 0.0), explicit_local_done)

    def get_completed_trial_count(self) -> int:
        return sum(int(value or 0) for value in self.completed_trials_by_key.values())

    def get_completed_local_min_trial_count(self) -> int:
        # avg_local is per local-min neighbor, not per finalist.  Keep the method
        # name for caller compatibility while returning the finalized neighbor units.
        return sum(int(value or 0) for value in self.completed_local_min_neighbors_by_key.values())

    def get_completed_local_min_finalist_count(self) -> int:
        return sum(int(value or 0) for value in self.completed_local_min_trials_by_key.values())

    def get_search_wall_elapsed_sec(self) -> float | None:
        if not self.search_started_ts_by_key or not self.search_last_done_ts_by_key:
            return None
        return max(0.0, max(self.search_last_done_ts_by_key.values()) - min(self.search_started_ts_by_key.values()))

    def get_local_min_wall_elapsed_sec(self) -> float | None:
        if not self.local_min_started_ts_by_key or not self.local_min_last_done_ts_by_key:
            return None
        return max(0.0, max(self.local_min_last_done_ts_by_key.values()) - min(self.local_min_started_ts_by_key.values()))

    def get_completed_fold_count(self) -> int:
        return len(self.result_rows_by_fold)

    def _record_replay_metrics(self, fold_idx: int, progress: dict) -> None:
        data = dict(progress or {})
        replay_started = _safe_progress_float(data, "replay_started_ts")
        if replay_started is not None:
            previous = self.replay_started_ts_by_fold.get(fold_idx)
            self.replay_started_ts_by_fold[fold_idx] = replay_started if previous is None else min(previous, replay_started)
        replay_done_ts = _safe_progress_float(data, "replay_last_done_ts")
        ts = _safe_progress_ts(data)
        try:
            replay_done = int(data.get("replay_done", 0) or 0)
        except (TypeError, ValueError):
            replay_done = 0
        if replay_done > int(self.completed_replays_by_fold.get(fold_idx, 0) or 0):
            self.completed_replays_by_fold[fold_idx] = int(replay_done)
        if replay_done_ts is None and replay_done > 0:
            replay_done_ts = ts
        if replay_done_ts is not None:
            self.replay_last_done_ts_by_fold[fold_idx] = max(float(self.replay_last_done_ts_by_fold.get(fold_idx, 0.0) or 0.0), replay_done_ts)

    def get_completed_replay_count(self) -> int:
        return sum(int(value or 0) for value in self.completed_replays_by_fold.values())

    def get_replay_wall_elapsed_sec(self) -> float | None:
        if not self.replay_started_ts_by_fold or not self.replay_last_done_ts_by_fold:
            return None
        return max(0.0, max(self.replay_last_done_ts_by_fold.values()) - min(self.replay_started_ts_by_fold.values()))

    def update_fold_progress(self, *, fold_idx: int, progress: dict, force: bool = False) -> None:
        idx = int(fold_idx)
        payload = dict(progress or {})
        payload.setdefault("fold_idx", idx)
        self.fold_progress_by_idx[idx] = payload
        self._record_replay_metrics(idx, payload)
        self.render(force=force)

    def update_result_row(self, row: dict, *, force: bool = False) -> None:
        payload = dict(row or {})
        try:
            fold_idx = int(payload.get("fold_idx", 0) or 0)
        except (TypeError, ValueError):
            fold_idx = 0
        if fold_idx <= 0:
            fold_idx = 1
        self.result_rows_by_fold[fold_idx] = payload
        self.fold_progress_by_idx[fold_idx] = {
            "stage": "FOLD_RESULT",
            "status": "fold result ready",
            "fold_idx": fold_idx,
            "fold_count": max((int(ctx.get("fold_count", 0) or 0) for ctx in self.contexts), default=fold_idx),
            "oos_year": int(payload.get("oos_year", 0) or 0),
            "elapsed_sec": payload.get("elapsed_sec"),
        }
        self.render(force=force)

    def update(self, *, fold_idx: int, seed_index: int, progress: dict, force: bool = False) -> None:
        key = (int(fold_idx), int(seed_index))
        self.progress_by_key[key] = dict(progress or {})
        self._record_progress_metrics(key, dict(progress or {}))
        self.render(force=force)

    def update_many(self, progress_map: dict[tuple[int, int], dict], *, force: bool = False) -> None:
        for key, progress in dict(progress_map or {}).items():
            try:
                normalized_key = (int(key[0]), int(key[1]))
            except (TypeError, ValueError, IndexError):
                continue
            self.progress_by_key[normalized_key] = dict(progress or {})
            self._record_progress_metrics(normalized_key, dict(progress or {}))
        self.render(force=force)

    def _build_lines(self) -> list[str]:
        header_text = str(self.header_factory(self) if callable(self.header_factory) else self.header)
        progress_by_key = {
            self._context_key(context): dict(self.progress_by_key.get(self._context_key(context)) or {"stage": "QUEUED", "status": "queued"})
            for context in self.contexts
        }
        fold_progress_lines: list[str] = []
        context_by_fold = {int(ctx.get("fold_idx", 0) or 0): dict(ctx) for ctx in self.contexts}
        for fold_idx, progress in sorted(self.fold_progress_by_idx.items()):
            stage = str((progress or {}).get("stage") or "").upper()
            if stage not in {"ENSEMBLE_REPLAY", "FOLD_RESULT"}:
                continue
            context = dict(context_by_fold.get(int(fold_idx)) or {})
            task = {
                "fold_idx": int(fold_idx),
                "fold_count": int(context.get("fold_count", progress.get("fold_count", 0)) or 0),
                "oos_year": int(context.get("oos_year", progress.get("oos_year", 0)) or 0),
                "oos_period": str(context.get("oos_period") or progress.get("oos_period") or ""),
                "selection_period": str(context.get("selection_period") or context.get("selection_start") or ""),
                "selection_start_date": str(context.get("selection_start") or ""),
                "selection_end_date": str(context.get("selection_end") or ""),
                "show_oos": bool(context.get("show_oos", True)),
            }
            fold_progress_lines.append(_format_parallel_fold_progress_line(task, dict(progress or {})))
        table_text = ""
        if self.result_rows_by_fold:
            main_title, retention_title = _active_optimizer_table_titles()
            show_oos_avg = any(bool(ctx.get("show_oos", True)) for ctx in self.contexts)
            table_text = _render_optimizer_results_tables(
                sorted((normalize_optimizer_seed_ensemble_fold_row(row) for row in self.result_rows_by_fold.values()), key=optimizer_seed_ensemble_row_sort_key),
                color=True,
                include_chain=False,
                include_oos_avg=show_oos_avg,
                main_table_title=main_title,
                retention_table_title=retention_title,
            )
        return build_optimizer_seed_ensemble_live_lines(
            header_text=header_text,
            seed_contexts=self.contexts,
            seed_progress_by_key=progress_by_key,
            fold_progress_lines=fold_progress_lines,
            table_text=table_text,
            color=True,
        )

    def render(self, *, force: bool = False) -> None:
        lines = self._build_lines()
        if not force and lines == self.last_lines:
            return
        self.last_lines = list(lines)
        if self.inline:
            if self.rendered_lines > 0:
                sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            for line in lines:
                sys.stdout.write("\r" + line + "\x1b[K\n")
            if self.rendered_lines > len(lines):
                for _ in range(self.rendered_lines - len(lines)):
                    sys.stdout.write("\r\x1b[K\n")
            sys.stdout.flush()
            self.rendered_lines = len(lines)
        else:
            print("\n".join(lines), flush=True)
            self.rendered_lines = 0

    def close(self) -> None:
        if self.inline and self.rendered_lines > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.rendered_lines = 0








class _ParallelFoldLiveBoard:
    """Render parallel-fold progress and completed results as one refresh block."""

    def __init__(self, tasks: list[dict], *, overall_start: float | None = None, raw_data_load_sec: float = 0.0):
        self.tasks = sorted(list(tasks or []), key=lambda item: int(item.get("fold_idx", 0) or 0))
        self.inline = stdout_supports_inline_progress()
        self.started_at = time.perf_counter()
        self.overall_start = float(overall_start) if overall_start is not None else self.started_at
        self.raw_data_load_sec = max(0.0, float(raw_data_load_sec or 0.0))
        self.rendered_lines = 0
        self.last_lines: list[str] = []
        self.last_render_key: list[str] = []

    def _build_lines(self, *, pending: set, future_map: dict, completed_rows: list[dict], fold_timing_rows: list[dict] | None = None) -> list[str]:
        completed_rows_sorted = sorted((normalize_optimizer_seed_ensemble_fold_row(item) for item in list(completed_rows or [])), key=optimizer_seed_ensemble_row_sort_key)
        completed_oos = {int(row.get("oos_year", 0) or 0) for row in completed_rows_sorted}
        fold_wall_elapsed = max(0.0, time.perf_counter() - self.started_at)
        total_elapsed = max(0.0, time.perf_counter() - self.overall_start)
        setup_elapsed = max(0.0, self.started_at - self.overall_start)
        # parallel fold 模式下，raw data 由各 fold worker 自行載入，
        # worker raw 時間已包含在 fold_time wall 裡；不要再以 raw/other 顯示，避免與總時間對帳時重複或缺項。
        total_folds = max((int(task.get("fold_count", 0) or 0) for task in self.tasks), default=len(self.tasks)) or len(self.tasks)
        seed_ensemble_display = _is_rolling_random_seed_ensemble_enabled()
        live_result_rows = _merge_live_result_rows(completed_rows_sorted, self.tasks)
        table_text = ""
        if live_result_rows:
            main_title, retention_title = _active_optimizer_table_titles()
            table_text = _render_optimizer_results_tables(
                live_result_rows,
                color=True,
                include_chain=False,
                include_oos_avg=True,
                main_table_title=main_title,
                retention_table_title=retention_title,
            )
        if seed_ensemble_display:
            seed_policy = _build_rolling_seed_ensemble_policy_payload()
            seed_count = int(seed_policy.get("seed_count", 1) or 1)
            seed_progress_maps = {id(task): _read_latest_parallel_fold_seed_progresses_for_task(task) for task in self.tasks}
            phase_metrics = _collect_seed_progress_phase_metrics(
                progress
                for progress_map in seed_progress_maps.values()
                for progress in dict(progress_map or {}).values()
            )
            completed_trials = int(phase_metrics.get("completed_trials", 0) or 0)
            replay_metrics = _collect_parallel_fold_replay_phase_metrics(self.tasks)
            header = format_optimizer_seed_ensemble_progress_header(
                folds=total_folds,
                seeds=seed_count,
                min_agree=int(seed_policy.get("min_agree", seed_count) or seed_count),
                parallel_workers=resolve_optimizer_random_seed_ensemble_parallel_workers_default(seed_count),
                backend=resolve_optimizer_random_seed_ensemble_parallel_backend_default(),
                completed_folds=len(completed_rows_sorted),
                pending_folds=len(pending),
                total_elapsed_sec=total_elapsed,
                fold_elapsed_sec=fold_wall_elapsed,
                setup_elapsed_sec=setup_elapsed,
                completed_trials=completed_trials,
                search_wall_elapsed_sec=phase_metrics.get("search_wall_elapsed_sec"),
                completed_local_min_trials=int(phase_metrics.get("completed_local_min_trials", 0) or 0),
                local_min_wall_elapsed_sec=phase_metrics.get("local_min_wall_elapsed_sec"),
                completed_replays=int(replay_metrics.get("completed_replays", 0) or 0),
                replay_wall_elapsed_sec=replay_metrics.get("replay_wall_elapsed_sec"),
            )
            seed_contexts: list[dict] = []
            seed_progress_by_key: dict[tuple[int, int], dict] = {}
            fold_progress_lines: list[str] = []
            for task in self.tasks:
                seed_progresses = dict(seed_progress_maps.get(id(task)) or {})
                fold_completed = int(task.get("oos_year", 0) or 0) in completed_oos
                for seed_index in range(1, seed_count + 1):
                    progress = dict(seed_progresses.get(seed_index) or {})
                    if not progress:
                        progress = {"stage": "DONE" if fold_completed else "QUEUED", "status": "done" if fold_completed else "queued"}
                    context = {
                        "fold_idx": int(task.get("fold_idx", 0) or 0),
                        "fold_count": int(task.get("fold_count", 0) or 0),
                        "seed_index": int(seed_index),
                        "seed_count": int(seed_count),
                        "seed": progress.get("seed"),
                        "oos_year": int(task.get("oos_year", 0) or 0),
                        "oos_period": str(task.get("oos_period") or ""),
                        "selection_start": str(task.get("selection_start_date") or task.get("selection_period") or ""),
                        "selection_end": str(task.get("selection_end_date") or ""),
                        "selection_period": str(task.get("selection_period") or ""),
                    }
                    seed_contexts.append(context)
                    seed_progress_by_key[_seed_progress_context_key(context)] = progress
                fold_progress = _read_latest_parallel_fold_progress(str(task.get("log_path") or ""))
                fold_stage = str(fold_progress.get("stage") or "").upper()
                if fold_progress and fold_stage in {"ENSEMBLE_REPLAY", "FOLD_RESULT"}:
                    fold_progress_lines.append(_format_parallel_fold_progress_line(task, fold_progress))
            lines = build_optimizer_seed_ensemble_live_lines(
                header_text=header,
                seed_contexts=seed_contexts,
                seed_progress_by_key=seed_progress_by_key,
                fold_progress_lines=fold_progress_lines,
                table_text=table_text,
                color=True,
            )
        else:
            _ = (pending, fold_wall_elapsed, setup_elapsed)
            header = (
                f"⏱️ Rolling fold parallel | completed={len(completed_rows_sorted)}/{total_folds} | "
                f"total_time={_fmt_duration(total_elapsed)}"
            )
            lines = [f"{C_CYAN}{header}{C_RESET}"]
            for task in self.tasks:
                log_path = str(task.get("log_path") or "")
                progress = _read_latest_parallel_fold_progress(log_path)
                if int(task.get("oos_year", 0) or 0) in completed_oos and not progress:
                    progress = {
                        "stage": "DONE",
                        "status": "done",
                        "fold_idx": int(task.get("fold_idx", 0) or 0),
                        "fold_count": int(task.get("fold_count", 0) or 0),
                        "oos_year": int(task.get("oos_year", 0) or 0),
                    }
                log_status = _latest_parallel_fold_log_status(log_path)
                lines.append(f"{C_GRAY}  {_format_parallel_fold_progress_line(task, progress, log_status=log_status)}{C_RESET}")
        if (not seed_ensemble_display) and table_text:
            lines.append("")
            lines.extend(table_text.splitlines())
        return lines

    @staticmethod
    def _stable_render_key(lines: list[str]) -> list[str]:
        # Ignore elapsed-only heartbeat changes; refresh when fold progress or completed results change.
        key: list[str] = []
        for line in list(lines or []):
            text = str(line)
            if "⏱️ 耗時摘要" in text:
                text = "⏱️ 耗時摘要"
            elif "seed ensemble |" in text or "⏱️ Rolling fold parallel" in text or text.lstrip().startswith("folds="):
                for marker in (" | total_time=", " | total=", " | elapsed="):
                    if marker in text:
                        text = text.split(marker, 1)[0]
                        break
            elif " | elapsed=" in text:
                text = text.split(" | elapsed=", 1)[0]
            key.append(text)
        return key

    def render(self, *, pending: set, future_map: dict, completed_rows: list[dict], fold_timing_rows: list[dict] | None = None, force: bool = False) -> None:
        lines = self._build_lines(pending=pending, future_map=future_map, completed_rows=completed_rows, fold_timing_rows=fold_timing_rows)
        render_key = self._stable_render_key(lines)
        if not force and render_key == self.last_render_key:
            return
        self.last_lines = list(lines)
        self.last_render_key = list(render_key)
        if self.inline:
            if self.rendered_lines > 0:
                sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            for line in lines:
                sys.stdout.write("\r" + line + "\x1b[K\n")
            if self.rendered_lines > len(lines):
                for _ in range(self.rendered_lines - len(lines)):
                    sys.stdout.write("\r\x1b[K\n")
            sys.stdout.flush()
            self.rendered_lines = len(lines)
        else:
            # Non-interactive outputs cannot refresh safely; print only on meaningful changes.
            print("\n".join(lines), flush=True)
            self.rendered_lines = 0

    def close(self) -> None:
        if self.inline and self.rendered_lines > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.rendered_lines = 0
