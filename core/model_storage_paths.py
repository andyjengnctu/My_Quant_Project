"""AI: Model storage root shared by discovery and canonical artifact owners."""
import os
from typing import Mapping, Optional

MODELS_DIR_ENV_VAR = "V16_MODELS_DIR"


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


