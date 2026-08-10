"""Shared project-wide console-report formatting."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

from core.display import (
    C_CYAN,
    C_GRAY,
    C_GREEN,
    C_RED,
    C_RESET,
    C_YELLOW,
    _display_width,
    _strip_ansi,
)

DEFAULT_REPORT_WIDTH = 100
COMPACT_CONSOLE_ENV = "BREAKOUT_QUALITY_COMPACT_CONSOLE"


def compact_console_enabled() -> bool:
    """Return whether interactive workflow output should hide internal artifacts."""

    value = os.environ.get(COMPACT_CONSOLE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def console_color_enabled(stream=None) -> bool:
    """Return whether ANSI color is appropriate for the active console."""

    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    target = stream if stream is not None else getattr(sys, "stdout", None)
    return bool(target is not None and hasattr(target, "isatty") and target.isatty())


def paint(text: object, tone: str, *, enabled: bool, bold: bool = False) -> str:
    raw = str(text)
    if not enabled:
        return raw
    colors = {
        "cyan": C_CYAN,
        "green": C_GREEN,
        "yellow": C_YELLOW,
        "red": C_RED,
        "gray": C_GRAY,
    }
    color = colors.get(str(tone), "")
    bold_code = "\033[1m" if bold else ""
    return f"{bold_code}{color}{raw}{C_RESET}"


def project_relative_display_path(
    path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> str:
    """Render project-scoped paths from repository root using forward slashes."""

    raw = os.fspath(path)
    candidate = Path(raw)
    root = Path(project_root)
    try:
        relative = candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        if candidate.is_absolute():
            return raw.replace("\\", "/")
        relative = candidate
    text = relative.as_posix()
    return text if text not in {"", "."} else "."


def _pad(text: object, width: int, *, align: str) -> str:
    raw = str(text)
    padding = max(0, int(width) - _display_width(raw))
    if align == "right":
        return " " * padding + raw
    if align == "center":
        left = padding // 2
        return " " * left + raw + " " * (padding - left)
    return raw + " " * padding


def render_title(title: str, *, width: int = DEFAULT_REPORT_WIDTH) -> str:
    line = "=" * int(width)
    return f"{line}\n {title}\n{line}"


def render_menu_item(index: int, label: object, *, default: bool = False) -> str:
    """Render one interactive menu row using the project-wide Enter convention."""

    number = int(index)
    if number < 0:
        raise ValueError("menu index不得小於0")
    text = str(label)
    if default:
        return f"[{number} ] {text}  (Enter)"
    return f"[{number}]  {text}"


def render_section(title: str, *, number: int | None = None) -> str:
    label = f"{int(number)}. {title}" if number is not None else str(title)
    return f"\n{label}\n{'-' * _display_width(label)}"


def render_key_values(
    rows: Iterable[tuple[object, object]],
    *,
    separator: str = "：",
) -> str:
    normalized = [(str(label), str(value)) for label, value in rows]
    if not normalized:
        return ""
    label_width = max(_display_width(label) for label, _value in normalized)
    return "\n".join(
        f"{_pad(label, label_width, align='left')}{separator}{value}"
        for label, value in normalized
    )


def render_table(
    headers: Sequence[object],
    rows: Iterable[Sequence[object]],
    *,
    alignments: Sequence[str] | None = None,
) -> str:
    header_text = [str(value) for value in headers]
    body = [[str(value) for value in row] for row in rows]
    column_count = len(header_text)
    if any(len(row) != column_count for row in body):
        raise ValueError("console table row欄數必須與header一致")
    aligns = list(alignments or ("left",) * column_count)
    if len(aligns) != column_count:
        raise ValueError("console table alignments欄數必須與header一致")
    widths = []
    for index in range(column_count):
        widths.append(
            max(
                [_display_width(header_text[index])]
                + [_display_width(row[index]) for row in body]
            )
        )
    header_line = "  ".join(
        _pad(value, widths[index], align="left")
        for index, value in enumerate(header_text)
    )
    separator_line = "  ".join("-" * width for width in widths)
    body_lines = [
        "  ".join(
            _pad(value, widths[index], align=aligns[index])
            for index, value in enumerate(row)
        )
        for row in body
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def render_artifact_paths(
    artifacts: Iterable[tuple[str, str | os.PathLike[str]]],
    *,
    project_root: str | os.PathLike[str],
    title: str = "工件輸出",
) -> str:
    rows = [
        (str(label), project_relative_display_path(path, project_root=project_root))
        for label, path in artifacts
    ]
    if not rows:
        return ""
    return f"{render_section(title)}\n{render_key_values(rows)}"


def print_artifact_paths(
    artifacts: Iterable[tuple[str, str | os.PathLike[str]]],
    *,
    project_root: str | os.PathLike[str],
    title: str = "工件輸出",
) -> None:
    if compact_console_enabled():
        return
    rendered = render_artifact_paths(
        artifacts,
        project_root=project_root,
        title=title,
    )
    if rendered:
        print(rendered)


def render_status_paths(
    artifacts: Iterable[tuple[str, str | os.PathLike[str], bool]],
    *,
    project_root: str | os.PathLike[str],
    color: bool = False,
) -> str:
    rows = []
    for label, path, exists in artifacts:
        status = "存在" if bool(exists) else "缺少"
        status_tone = "green" if bool(exists) else "red"
        rows.append(
            (
                paint(f"[{status}]", status_tone, enabled=color),
                str(label),
                project_relative_display_path(path, project_root=project_root),
            )
        )
    return render_table(
        ("狀態", "工件", "路徑"),
        rows,
        alignments=("left", "left", "left"),
    )


def strip_ansi(text: object) -> str:
    return _strip_ansi(str(text))


__all__ = [
    "DEFAULT_REPORT_WIDTH",
    "COMPACT_CONSOLE_ENV",
    "compact_console_enabled",
    "console_color_enabled",
    "paint",
    "project_relative_display_path",
    "render_title",
    "render_menu_item",
    "render_section",
    "render_key_values",
    "render_table",
    "render_artifact_paths",
    "print_artifact_paths",
    "render_status_paths",
    "strip_ansi",
]
