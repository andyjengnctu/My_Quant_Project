"""Outer-rolling single-fold optimizer search progress rendering.

This module owns the inline single-fold search progress state only. It does not
own fold construction, selection, replay, seed semantics, or artifact identity.
"""

from __future__ import annotations

import time
from threading import Lock

from core.runtime_utils import (
    choose_inline_progress_message,
    stdout_supports_inline_progress,
    write_inline_progress,
)
from services.optimizer.outer_rolling_formatting import (
    _display_month_period,
    _display_month_value,
    _fmt_duration_compact,
)
from services.optimizer.score_display import format_optimizer_score_for_display
from services.optimizer.study_utils import is_qualified_trial_value


class _SearchProgress:
    def __init__(self, *, fold_idx: int, fold_count: int, oos_year: int, selection_start, selection_end, total_trials: int, completed_results: list[dict], overall_start: float):
        self.fold_idx = int(fold_idx)
        self.fold_count = int(fold_count)
        self.oos_year = int(oos_year)
        self.selection_start = str(selection_start)
        self.selection_end = str(selection_end)
        self.total_trials = int(total_trials)
        self.completed_results = completed_results
        self.overall_start = float(overall_start)
        self.stage_start = time.perf_counter()
        self.stage_start_ts = time.time()
        self.best_score = float("-inf")
        self.last_render = 0.0
        self._lock = Lock()
        self.inline_progress_enabled = stdout_supports_inline_progress()
        self.inline_progress_width = 0

    def _eta_stage(self, completed: int) -> float | None:
        if completed <= 0:
            return None
        elapsed = max(0.0, time.perf_counter() - self.stage_start)
        avg = elapsed / float(completed)
        return avg * max(0, self.total_trials - completed)

    def render(self, completed: int, *, force: bool = False):
        if not self.inline_progress_enabled:
            return
        now = time.perf_counter()
        if not force and now - self.last_render < 0.5 and completed < self.total_trials:
            return
        self.last_render = now
        pct = 100.0 * float(completed) / max(1, self.total_trials)
        eta_stage = self._eta_stage(completed)
        elapsed_total = now - self.overall_start
        eta_total = None
        done_folds = len(self.completed_results)
        if done_folds > 0:
            avg_done = sum(float(row.get("elapsed_sec", 0.0)) for row in self.completed_results) / float(done_folds)
            current_remaining = eta_stage or 0.0
            eta_total = current_remaining + avg_done * max(0, self.fold_count - self.fold_idx)
        best_score = self.best_score if self.best_score != float("-inf") else 0.0
        selection_text = _display_month_period(self.selection_start, self.selection_end)
        oos_text = _display_month_value(self.oos_year)
        elapsed_text = _fmt_duration_compact(now - self.stage_start)
        eta_stage_text = _fmt_duration_compact(eta_stage)
        eta_total_text = _fmt_duration_compact(eta_total)
        line = choose_inline_progress_message((
            (
                f"[{self.fold_idx}/{self.fold_count}] selection={selection_text} | OOS={oos_text} | "
                f"OPTIMIZER_SEARCH | 進度={completed}/{self.total_trials} ({pct:5.1f}%) | "
                f"best_base_score={format_optimizer_score_for_display(best_score, decimals=3)} | elapsed={elapsed_text} | eta={eta_stage_text}/{eta_total_text}"
            ),
            (
                f"[{self.fold_idx}/{self.fold_count}] selection={selection_text} | OOS={oos_text} | "
                f"search | 進度={completed}/{self.total_trials} ({pct:5.1f}%) | "
                f"best_base={format_optimizer_score_for_display(best_score, decimals=3)} | elapsed={elapsed_text} | eta={eta_stage_text}/{eta_total_text}"
            ),
            (
                f"[{self.fold_idx}/{self.fold_count}] {selection_text}>OOS{oos_text} | "
                f"search {completed}/{self.total_trials} | best_base={format_optimizer_score_for_display(best_score, decimals=3)} | eta={eta_stage_text}/{eta_total_text}"
            ),
        ))
        self.inline_progress_width = write_inline_progress(line, previous_width=self.inline_progress_width)

    def callback(self, session):
        def _callback(study, trial):
            with self._lock:
                session.current_session_trial += 1
                if trial.value is not None and is_qualified_trial_value(trial.value):
                    self.best_score = max(self.best_score, float(trial.value))
                completed = int(session.current_session_trial)
            self.render(completed)
        return _callback

    def done(self, completed: int):
        if self.inline_progress_enabled:
            self.render(completed, force=True)
            print()
