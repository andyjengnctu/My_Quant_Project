"""Shared read-only parsing of canonical trainer progress logs."""

from __future__ import annotations

import os
from pathlib import Path
import re

_EPOCH_PROGRESS_MARKER_RE = re.compile(
    r"__BQ_EPOCH_PROGRESS__\s+phase=(select|refit)\s+epoch=(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
_EPOCH_LOG_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_EPOCH_PHASE_RE = re.compile(
    r"(Epoch\s*選擇|Fold歷史資料重訓|完整 Selection(?: 重訓| 訓練)?)"
)
_PIT_TRAINING_FOLD_RE = re.compile(r"fold_\d{8}_\d{8}.*訓練並評分")
_PIT_PLAN_RE = re.compile(r"\[PIT plan\]\s*建立\s*(\d+)\s*個fold", re.IGNORECASE)
_PIT_ACTIVE_FOLD_RE = re.compile(r"PIT fold\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_PIT_COMPLETE_RE = re.compile(r"PIT Scores 完成\s*\|\s*folds=(\d+)", re.IGNORECASE)
_PIT_FOLD_ID_RE = re.compile(r"fold_\d{8}_\d{8}", re.IGNORECASE)
_PIT_VERBOSE_ACTIVE_FOLD_RE = re.compile(
    r"(fold_\d{8}_\d{8})[^\n]*訓練並評分", re.IGNORECASE
)


def _read_trainer_log_tail(log_path: str | Path, *, max_bytes: int = 262144) -> str | None:
    """Read a bounded tail from one canonical trainer log."""

    path = Path(log_path)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - int(max_bytes)), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def _read_trainer_log_head(log_path: str | Path, *, max_bytes: int = 131072) -> str | None:
    """Read a bounded head containing the canonical PIT plan."""

    path = Path(log_path)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            return handle.read(int(max_bytes)).decode("utf-8", errors="replace")
    except OSError:
        return None


def read_trainer_pit_progress(log_path: str | Path) -> tuple[int, int] | None:
    """Return canonical PIT ``(completed_or_prior, total)`` progress from trainer output.

    The PIT builder owns the fold schedule.  Consumers read that builder's plan and
    active-fold output instead of recomputing the schedule or inspecting a second
    artifact convention.  During an active fold ``i/N``, the first value is ``i-1``
    so the shared renderer reports ``active fold i/N``.
    """

    head = _read_trainer_log_head(log_path) or ""
    tail = _read_trainer_log_tail(log_path) or ""
    if not head and not tail:
        return None

    completed = list(_PIT_COMPLETE_RE.finditer(tail))
    if completed:
        total = int(completed[-1].group(1))
        return total, total

    active = list(_PIT_ACTIVE_FOLD_RE.finditer(tail))
    if active:
        current = int(active[-1].group(1))
        total = int(active[-1].group(2))
        if total > 0:
            current = min(max(1, current), total)
            return current - 1, total

    plan_matches = list(_PIT_PLAN_RE.finditer(head)) or list(_PIT_PLAN_RE.finditer(tail))
    total = int(plan_matches[-1].group(1)) if plan_matches else 0
    if total <= 0:
        return None

    # Strategy Compare intentionally runs the detailed PIT console in its hidden
    # trainer log.  Map the current verbose fold id back to the plan table so the
    # same live fold progress is available to single- and multi-seed callers.
    plan_section = head
    section_start = plan_section.find("Selection point-in-time fold plan")
    if section_start >= 0:
        plan_section = plan_section[section_start:]
    section_end = plan_section.find("執行環境")
    if section_end >= 0:
        plan_section = plan_section[:section_end]
    planned_fold_ids: list[str] = []
    for match in _PIT_FOLD_ID_RE.finditer(plan_section):
        fold_id = str(match.group(0))
        if fold_id not in planned_fold_ids:
            planned_fold_ids.append(fold_id)
    verbose_active = list(_PIT_VERBOSE_ACTIVE_FOLD_RE.finditer(tail))
    if verbose_active and planned_fold_ids:
        active_id = str(verbose_active[-1].group(1))
        try:
            current = planned_fold_ids.index(active_id) + 1
        except ValueError:
            current = 0
        if current > 0:
            bounded_total = max(total, len(planned_fold_ids))
            return current - 1, bounded_total

    return 0, total


def read_trainer_epoch_progress(log_path: str | Path) -> tuple[str, int, int] | None:
    """Return latest ``(phase, epoch, total)`` from one canonical trainer log."""

    text = _read_trainer_log_tail(log_path)
    if not text:
        return None

    fold_matches = list(_PIT_TRAINING_FOLD_RE.finditer(text))
    if fold_matches:
        text = text[fold_matches[-1].start():]
    marker_matches = list(_EPOCH_PROGRESS_MARKER_RE.finditer(text))
    if marker_matches:
        marker = marker_matches[-1]
        return (
            str(marker.group(1)).lower(),
            int(marker.group(2)),
            int(marker.group(3)),
        )
    phase_matches = list(_EPOCH_PHASE_RE.finditer(text))
    if not phase_matches:
        return None
    phase_match = phase_matches[-1]
    phase_text = str(phase_match.group(1))
    phase = "select" if "選擇" in phase_text else "refit"
    epoch_matches = list(_EPOCH_LOG_RE.finditer(text, phase_match.end()))
    if epoch_matches:
        epoch_match = epoch_matches[-1]
        return phase, int(epoch_match.group(1)), int(epoch_match.group(2))
    return phase, 0, 0


def render_training_unit_progress(
    *,
    unit_id: str,
    elapsed_seconds: float,
    source_index: int | None = None,
    source_count: int | None = None,
    pit_progress: tuple[int, int] | None = None,
    epoch_progress: tuple[str, int, int] | None = None,
) -> str:
    """Render one canonical Strategy Compare trainer unit status."""

    from core.display_common import render_elapsed

    identity = str(unit_id)
    if source_index is not None and source_count is not None:
        identity += f" {int(source_index)}/{int(source_count)}"
    parts = [identity, render_elapsed(float(elapsed_seconds))]

    if pit_progress is not None:
        saved, expected = (int(pit_progress[0]), int(pit_progress[1]))
        parts.extend([f"PIT {saved}/{expected}", f"remain {max(0, expected - saved)}"])
        if saved < expected:
            parts.append(f"active fold {saved + 1}/{expected}")

    if epoch_progress is None:
        parts.append("epoch pending")
    else:
        phase, epoch, epoch_count = epoch_progress
        if int(epoch_count) > 0:
            parts.append(f"epoch {phase} {int(epoch)}/{int(epoch_count)}")
        else:
            parts.append(f"epoch {phase} -")
    return " | ".join(parts)


__all__ = [
    "read_trainer_epoch_progress",
    "read_trainer_pit_progress",
    "render_training_unit_progress",
]
