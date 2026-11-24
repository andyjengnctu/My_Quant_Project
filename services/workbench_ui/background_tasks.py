"""Tk-independent background jobs with main-loop delivery and stale-result guards."""
from dataclasses import dataclass
import queue
import threading


@dataclass(frozen=True)
class TaskResult:
    token: int
    value: object = None
    error: Exception | None = None


class BackgroundTaskRunner:
    """One logical job slot; superseded/closed jobs may finish but never render.

    schedule/cancel must be supplied by the UI thread. Workers only enqueue
    immutable result envelopes and never call Tk, even during destruction.
    """

    def __init__(self, schedule, cancel, *, poll_ms=25):
        self._schedule = schedule
        self._cancel = cancel
        self._poll_ms = int(poll_ms)
        self._results = queue.Queue()
        self._token = 0
        self._thread = None
        self._after_id = None
        self._callback = None
        self._closed = False

    @property
    def busy(self):
        return self._callback is not None

    def submit(self, worker, callback, *, name="workbench-task", replace=False):
        if self._closed or (self.busy and not replace):
            return None
        self._token += 1
        token = self._token
        self._callback = callback
        def run():
            try:
                result = TaskResult(token, value=worker())
            except Exception as exc:
                result = TaskResult(token, error=exc)
            self._results.put(result)
        thread = threading.Thread(target=run, name=name, daemon=True)
        self._thread = thread
        thread.start()
        self._ensure_poll()
        return thread

    def _ensure_poll(self):
        if self._after_id is None and not self._closed:
            self._after_id = self._schedule(self._poll_ms, self._drain)

    def _drain(self):
        self._after_id = None
        if self._closed:
            return
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            if result.token != self._token:
                continue
            callback, self._callback = self._callback, None
            self._thread = None
            if callback is not None:
                callback(result)
        if self.busy:
            self._ensure_poll()

    def cancel(self):
        """Invalidate delivery; a running domain command is never interrupted."""
        self._token += 1
        self._callback = None
        if self._after_id is not None:
            self._cancel(self._after_id)
            self._after_id = None

    def close(self):
        self.cancel()
        self._closed = True
