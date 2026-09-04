"""Shared subprocess lifecycle for isolated Research model-training units."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from threading import Lock
from typing import Mapping, MutableMapping, Sequence

from config.runtime import RESEARCH_TRAINER_TERMINATION_GRACE_SECONDS


def terminate_training_process(
    proc: subprocess.Popen,
    *,
    grace_seconds: float = RESEARCH_TRAINER_TERMINATION_GRACE_SECONDS,
) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=float(grace_seconds))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def terminate_registered_training_processes(
    registry: Mapping[str, subprocess.Popen],
    registry_lock: Lock,
    *,
    grace_seconds: float = RESEARCH_TRAINER_TERMINATION_GRACE_SECONDS,
) -> None:
    with registry_lock:
        processes = list(registry.values())
    for proc in processes:
        terminate_training_process(proc, grace_seconds=grace_seconds)


def run_logged_training_process(
    *,
    command: Sequence[str],
    log_path: str | Path,
    cwd: str | Path,
    registry: MutableMapping[str, subprocess.Popen],
    registry_lock: Lock,
    registry_key: str,
    env_overrides: Mapping[str, str] | None = None,
    failure_prefix: str,
    log_tail_chars: int = 5000,
    grace_seconds: float = RESEARCH_TRAINER_TERMINATION_GRACE_SECONDS,
) -> dict[str, object]:
    """Run one trainer with shared logging, cancellation and failure-tail semantics."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in dict(env_overrides or {}).items()})
    started = time.perf_counter()
    with path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            list(command),
            cwd=str(Path(cwd).resolve()),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        with registry_lock:
            registry[str(registry_key)] = proc
        try:
            returncode = proc.wait()
        except BaseException:
            terminate_training_process(proc, grace_seconds=grace_seconds)
            raise
        finally:
            with registry_lock:
                registry.pop(str(registry_key), None)
    elapsed = time.perf_counter() - started
    if int(returncode or 0) != 0:
        tail = ""
        if path.is_file():
            tail = path.read_text(encoding="utf-8", errors="replace")[-int(log_tail_chars):]
        raise RuntimeError(
            f"{failure_prefix}: returncode={returncode}; log_tail={tail}"
        )
    return {"elapsed_sec": elapsed, "returncode": int(returncode or 0)}


__all__ = [
    "run_logged_training_process",
    "terminate_registered_training_processes",
    "terminate_training_process",
]
