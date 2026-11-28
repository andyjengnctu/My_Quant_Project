"""Execution-only settings and the scoped Windows-spawn environment adapter."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import signature
import os
from threading import RLock
from typing import Mapping


_ENVIRONMENT_LOCK = RLock()
OPTIMIZER_OUTPUT_DIR_ENV_VAR = "OPTIMIZER_OUTPUT_DIR"


def resolve_optimizer_output_dir(project_root, environ: Mapping[str, str] | None = None) -> str:
    """AI: Isolate study databases and diagnostics alongside an execution sandbox."""
    from core.output_paths import build_output_dir

    env = os.environ if environ is None else environ
    override = str(env.get(OPTIMIZER_OUTPUT_DIR_ENV_VAR, "")).strip()
    return os.path.abspath(os.path.join(project_root, override)) if override else build_output_dir(project_root, "ml_optimizer")


@dataclass(frozen=True)
class OptimizerStoragePaths:
    """Execution storage, isolated from scientific identity and other callers."""

    output_dir: str
    models_dir: str
    run_best_params_path: str
    candidate_best_params_path: str
    candidate_best_summary_path: str
    candidate_retention_best_params_path: str
    candidate_retention_best_summary_path: str
    candidate_val_score_best_params_path: str
    candidate_val_score_best_summary_path: str
    run_best_summary_path: str

    @classmethod
    def from_environ(cls, project_root, environ: Mapping[str, str] | None = None):
        from pathlib import Path
        from core.model_paths import (
            resolve_models_dir, resolve_run_best_params_path,
            resolve_candidate_best_params_path, resolve_candidate_retention_best_params_path,
            resolve_candidate_val_score_best_params_path,
        )
        from core.strategy_param_artifacts import resolve_strategy_param_state_path, STRATEGY_PARAM_ROOT_RELATIVE

        env = os.environ if environ is None else environ
        models = resolve_models_dir(str(project_root), environ=env)
        storage = Path(models) / STRATEGY_PARAM_ROOT_RELATIVE.name
        def state_path(artifact):
            return str(resolve_strategy_param_state_path(project_root, artifact=artifact, strategy_params_root=storage))
        return cls(
            output_dir=resolve_optimizer_output_dir(project_root, env), models_dir=models,
            run_best_params_path=resolve_run_best_params_path(str(project_root), env),
            candidate_best_params_path=resolve_candidate_best_params_path(str(project_root), env),
            candidate_best_summary_path=state_path("candidate_best_summary"),
            candidate_retention_best_params_path=resolve_candidate_retention_best_params_path(str(project_root), env),
            candidate_retention_best_summary_path=state_path("candidate_retention_best_summary"),
            candidate_val_score_best_params_path=resolve_candidate_val_score_best_params_path(str(project_root), env),
            candidate_val_score_best_summary_path=state_path("candidate_val_score_best_summary"),
            run_best_summary_path=state_path("active_summary"),
        )


_OPTIMIZER_STORAGE_PATHS = ContextVar("optimizer_storage_paths", default=None)


def current_optimizer_storage_paths() -> OptimizerStoragePaths | None:
    return _OPTIMIZER_STORAGE_PATHS.get()


@contextmanager
def optimizer_storage_scope(paths: OptimizerStoragePaths):
    token = _OPTIMIZER_STORAGE_PATHS.set(paths)
    try:
        yield paths
    finally:
        _OPTIMIZER_STORAGE_PATHS.reset(token)


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
