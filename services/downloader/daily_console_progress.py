"""Shared console renderer for canonical Market Data V2 Daily Update progress events."""
from __future__ import annotations

import sys

from core.display import C_CYAN, C_GREEN, C_GRAY, C_RED, C_RESET, C_YELLOW, _strip_ansi
from core.display_common import console_color_enabled


_COLOR_ENABLED = console_color_enabled()


def _paint(value: object, color: str) -> str:
    text = str(value)
    return f"{color}{text}{C_RESET}" if _COLOR_ENABLED else text


def _status_color(status: object) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"PASS", "READY", "UPDATED", "DONE", "AVAILABLE", "YES", "SYNCED", "COMPLETE", "REUSE"}:
        return C_GREEN
    if normalized in {"DEFERRED", "WAIT_PUBLISH", "WAIT_QUOTA", "UNVERIFIED", "ATTENTION", "PARTIAL", "INCOMPLETE", "RUN"}:
        return C_YELLOW
    if normalized in {"NO", "NO_DUE", "NOT_IN_BATCH"}:
        return C_GRAY
    if normalized in {"TARGET_ADVANCED"}:
        return C_CYAN
    if normalized in {"FAIL", "BLOCKED", "ERROR", "UNAVAILABLE", "STALE"}:
        return C_RED
    return C_CYAN


class MarketDataDailyConsoleProgress:
    """Render the exact progress event stream shared by Downloader and Workbench."""

    def __init__(self) -> None:
        self._line_open = False
        self._line_width = 0

    def close(self) -> None:
        if self._line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._line_open = False
            self._line_width = 0

    def _write_progress_line(self, text: str) -> None:
        rendered = str(text)
        visible_width = len(_strip_ansi(rendered))
        padding = " " * max(0, self._line_width - visible_width)
        sys.stdout.write("\r" + rendered + padding)
        sys.stdout.flush()
        self._line_open = True
        self._line_width = max(self._line_width, visible_width)

    def progress(self, event: dict[str, object]) -> None:
        kind = str(event.get("kind") or "")
        if kind == "QUOTA_OBSERVATION":
            self.close()
            status = str(event.get("status") or "UNKNOWN")
            if status == "CURRENT":
                print(
                    f"{_paint('[Quota]', C_CYAN)} {_paint(status, _status_color(status))}"
                    f" | used={event.get('quota_user_count')} / {event.get('quota_limit')}"
                    f" | remaining={event.get('quota_remaining')}"
                )
            else:
                print(
                    f"{_paint('[Quota]', C_CYAN)} {_paint(status, _status_color(status))}"
                    f" | {event.get('error') or '-'}"
                )
            return
        if kind == "PLAN":
            self.close()
            mode = "FORCE REFRESH" if event.get("force_refresh") else "DUE ONLY"
            print(
                _paint("[Daily]", C_CYAN)
                + f" {mode} | target={event.get('target_date')}"
                f" | datasets={event.get('dataset_count')} | requests={event.get('total')}"
            )
            return
        if kind != "REQUEST_PROGRESS":
            return
        done = int(event.get("done") or 0)
        total = int(event.get("total") or 0)
        pct = (100.0 * done / total) if total else 0.0
        dataset = str(event.get("dataset") or "-")
        data_id = event.get("data_id")
        target = dataset + (f"/{data_id}" if data_id else "")
        start_date = event.get("start_date")
        end_date = event.get("end_date")
        if start_date and end_date:
            request_scope = f"date={start_date}" if start_date == end_date else f"date={start_date}~{end_date}"
        else:
            request_scope = "date=STATIC"
        phase = str(event.get("phase") or "RUN")
        if bool(event.get("recovered")):
            phase = "REUSE"
        data_used = int(event.get("process_data_requests") or 0)
        usage_used = int(event.get("process_usage_requests") or 0)
        q_used = event.get("quota_user_count")
        q_limit = event.get("quota_limit")
        quota = "quota=--" if q_used is None or q_limit is None else f"quota≈{int(q_used)}/{int(q_limit)}"
        rendered_phase = _paint(phase, _status_color(phase))
        self._write_progress_line(
            f"{_paint('[Daily]', C_CYAN)} {done}/{total} ({pct:5.1f}%) | {target} | {request_scope} | {rendered_phase}"
            f" | data={data_used} usage={usage_used} | {quota}"
        )

    def quota_wait(self, event: dict[str, object]) -> None:
        self.close()
        print(
            f"{_paint('[WAIT]', C_YELLOW)} {event.get('done')}/{event.get('total')}"
            f" | quota={event.get('quota_user_count') or '-'} / {event.get('quota_limit') or '-'}"
            f" | reason={event.get('reason') or 'quota'}"
        )

    def result(self, result: dict[str, object], *, label: str = "Daily Update") -> None:
        """Render a compact final line from the canonical updater result."""

        self.close()
        status = str(result.get("status") or "UNKNOWN")
        print(
            f"{_paint('[Daily]', C_CYAN)} {label} | status={_paint(status, _status_color(status))}"
            f" | target={result.get('target_date') or '-'}"
            f" | due={result.get('due_dataset_count', 0)}"
            f" | data={result.get('data_requests', 0)} usage={result.get('usage_requests', 0)}"
            f" | next={result.get('next_check_at') or '-'}"
        )


__all__ = ["MarketDataDailyConsoleProgress"]
