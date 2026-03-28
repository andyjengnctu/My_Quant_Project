import os
import sys
from typing import Dict, Iterable, Optional, Sequence, Tuple

DATASET_PROFILE_REDUCED = "reduced"
DATASET_PROFILE_FULL = "full"
DEFAULT_DATASET_PROFILE = DATASET_PROFILE_FULL
DEFAULT_VALIDATE_DATASET_PROFILE = DATASET_PROFILE_REDUCED
DEFAULT_DATASET_ENV_VAR = "V16_DATASET_PROFILE"
VALIDATE_DATASET_ENV_VAR = "V16_VALIDATE_DATASET"

DATASET_PROFILE_SPECS: Dict[str, Dict[str, str]] = {
    DATASET_PROFILE_REDUCED: {
        "label": "縮減",
        "dir_name": "tw_stock_data_vip_reduced",
        "menu_key": "1",
    },
    DATASET_PROFILE_FULL: {
        "label": "完整",
        "dir_name": "tw_stock_data_vip",
        "menu_key": "2",
    },
}

_DATASET_PROFILE_ALIASES = {
    "1": DATASET_PROFILE_REDUCED,
    "r": DATASET_PROFILE_REDUCED,
    "reduced": DATASET_PROFILE_REDUCED,
    "mini": DATASET_PROFILE_REDUCED,
    "small": DATASET_PROFILE_REDUCED,
    "縮減": DATASET_PROFILE_REDUCED,
    "精簡": DATASET_PROFILE_REDUCED,
    "2": DATASET_PROFILE_FULL,
    "f": DATASET_PROFILE_FULL,
    "full": DATASET_PROFILE_FULL,
    "完整": DATASET_PROFILE_FULL,
}


def normalize_dataset_profile_key(value, default=DEFAULT_DATASET_PROFILE):
    if value is None:
        return default

    text = str(value).strip()
    if text == "":
        return default

    normalized = _DATASET_PROFILE_ALIASES.get(text.lower())
    if normalized is None:
        normalized = _DATASET_PROFILE_ALIASES.get(text)
    if normalized is None:
        valid_choices = ", ".join(sorted(DATASET_PROFILE_SPECS.keys()))
        raise ValueError(f"不支援的資料集模式: {value}，可用值: {valid_choices}")
    return normalized


def get_dataset_dir(project_root, profile_key):
    normalized_key = normalize_dataset_profile_key(profile_key)
    return os.path.join(project_root, DATASET_PROFILE_SPECS[normalized_key]["dir_name"])


def get_dataset_profile_label(profile_key):
    normalized_key = normalize_dataset_profile_key(profile_key)
    return DATASET_PROFILE_SPECS[normalized_key]["label"]


def build_validate_dataset_prompt(default=DEFAULT_VALIDATE_DATASET_PROFILE):
    default_key = normalize_dataset_profile_key(default)
    default_menu_key = DATASET_PROFILE_SPECS[default_key]["menu_key"]
    return (
        "👉 0. 驗證資料集 "
        f"(1=縮減 [{DATASET_PROFILE_SPECS[DATASET_PROFILE_REDUCED]['dir_name']}], "
        f"2=完整 [{DATASET_PROFILE_SPECS[DATASET_PROFILE_FULL]['dir_name']}], "
        f"預設 {default_menu_key}): "
    )


def _safe_prompt(prompt_text, default_value):
    if not sys.stdin or not sys.stdin.isatty():
        return default_value

    try:
        raw = input(prompt_text).strip()
    except EOFError:
        return default_value
    return raw if raw != "" else default_value


def extract_dataset_cli_value(argv: Optional[Sequence[str]]):
    if not argv:
        return None

    argv = list(argv)
    for idx in range(1, len(argv)):
        arg = argv[idx]
        if arg.startswith("--dataset="):
            return arg.split("=", 1)[1]
        if arg == "--dataset" and (idx + 1) < len(argv):
            return argv[idx + 1]
    return None


def resolve_dataset_profile_key(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[dict] = None,
    *,
    default=DEFAULT_DATASET_PROFILE,
    env_var_names: Optional[Iterable[str]] = None,
    allow_ui_prompt=False,
    prompt_text: Optional[str] = None,
) -> Tuple[str, str]:
    cli_value = extract_dataset_cli_value(sys.argv if argv is None else argv)
    if cli_value is not None:
        return normalize_dataset_profile_key(cli_value, default=default), "CLI"

    env = os.environ if environ is None else environ
    for env_var in tuple(env_var_names or (DEFAULT_DATASET_ENV_VAR,)):
        env_value = env.get(env_var)
        if env_value not in (None, ""):
            return normalize_dataset_profile_key(env_value, default=default), f"ENV:{env_var}"

    if allow_ui_prompt and prompt_text is not None:
        selected_value = _safe_prompt(prompt_text, default)
        return normalize_dataset_profile_key(selected_value, default=default), "UI"

    return normalize_dataset_profile_key(default, default=default), "DEFAULT"
