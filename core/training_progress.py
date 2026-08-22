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


def read_trainer_epoch_progress(log_path: str | Path) -> tuple[str, int, int] | None:
    """Return latest ``(phase, epoch, total)`` from one canonical trainer log."""

    path = Path(log_path)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 131072), os.SEEK_SET)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
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


__all__ = ["read_trainer_epoch_progress", "render_training_unit_progress"]
