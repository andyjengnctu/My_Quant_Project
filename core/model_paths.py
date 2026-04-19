import os
from typing import Mapping, Optional

MODELS_DIR_ENV_VAR = "V16_MODELS_DIR"
CHAMPION_PARAMS_PATH_ENV_VAR = "V16_CHAMPION_PARAMS_PATH"
DEFAULT_PARAM_SOURCE = "run_best"
VALID_PARAM_SOURCES = ("run_best", "champion")

def _resolve_override_path(project_root: str, raw_value: str) -> str:
    resolved = str(raw_value).strip()
    if resolved == "":
        raise ValueError("models 路徑覆寫不可為空白")
    if os.path.isabs(resolved):
        return os.path.abspath(resolved)
    return os.path.abspath(os.path.join(project_root, resolved))


def resolve_models_dir(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(MODELS_DIR_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return os.path.abspath(os.path.join(project_root, "models"))


def resolve_champion_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(CHAMPION_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return os.path.join(resolve_models_dir(project_root, environ=env), "champion_params.json")


def resolve_run_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    return os.path.join(resolve_models_dir(project_root, environ=env), "run_best_params.json")


def resolve_active_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    return resolve_run_best_params_path(project_root, environ=environ)


def normalize_param_source(param_source: str | None, *, default: str = DEFAULT_PARAM_SOURCE) -> str:
    normalized = str(default if param_source is None else param_source).strip().lower()
    if normalized == "":
        normalized = default
    if normalized not in VALID_PARAM_SOURCES:
        allowed = ", ".join(VALID_PARAM_SOURCES)
        raise ValueError(f"參數來源不合法，收到: {param_source}；允許值: {allowed}")
    return normalized


def resolve_named_params_path(project_root: str, param_source: str | None = None, *, environ: Optional[Mapping[str, str]] = None) -> str:
    normalized = normalize_param_source(param_source)
    if normalized == "champion":
        return resolve_champion_params_path(project_root, environ=environ)
    return resolve_run_best_params_path(project_root, environ=environ)
