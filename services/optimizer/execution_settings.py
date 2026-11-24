"""Execution-only settings and the scoped Windows-spawn environment adapter."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from inspect import signature
import os
from threading import RLock
from typing import Mapping


_ENVIRONMENT_LOCK = RLock()


@dataclass(frozen=True)
class OptimizerExecutionSettings:
    """An immutable, pickle-safe snapshot; never included in scientific hashes."""

    environment: tuple[tuple[str, str], ...]

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None):
        values = os.environ if environ is None else environ
        return cls(tuple(sorted((str(key), str(value)) for key, value in values.items())))

    def as_environ(self) -> dict[str, str]:
        return dict(self.environment)


def _optimizer_key(key: str) -> bool:
    return key == "V16_MODELS_DIR" or key.startswith("OPTIMIZER_")


@contextmanager
def optimizer_process_environment(settings: OptimizerExecutionSettings):
    """Expose only one run's settings to spawn, then restore even on cancellation."""
    with _ENVIRONMENT_LOCK:
        before = {key: value for key, value in os.environ.items() if _optimizer_key(key)}
        selected = {key: value for key, value in settings.environment if _optimizer_key(key)}
        try:
            for key in before.keys() - selected.keys():
                os.environ.pop(key, None)
            os.environ.update(selected)
            yield
        finally:
            for key in tuple(os.environ):
                if _optimizer_key(key) and key not in before:
                    os.environ.pop(key, None)
            os.environ.update(before)


@contextmanager
def temporary_process_environment(values: Mapping[str, str]):
    """Compatibility adapter for score-source variables used by older workers."""
    with _ENVIRONMENT_LOCK:
        before = {key: os.environ.get(key) for key in values}
        try:
            os.environ.update(values)
            yield
        finally:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def isolate_optimizer_process_environment(func):
    """Resolve once at the application boundary without mutating the caller map."""
    call_signature = signature(func)

    @wraps(func)
    def wrapped(*args, **kwargs):
        # Ambient resolution must wait for the previous scoped run to restore.
        with _ENVIRONMENT_LOCK:
            bound = call_signature.bind_partial(*args, **kwargs)
            settings = OptimizerExecutionSettings.from_environ(bound.arguments.get("environ"))
            bound.arguments["environ"] = settings.as_environ()
            with optimizer_process_environment(settings):
                return func(*bound.args, **bound.kwargs)

    return wrapped
