import os
from typing import Dict, Iterable, Optional, Tuple

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


def extract_dataset_cli_value(argv: Optional[Iterable[str]]):
    if argv is None:
        return None

    args = list(argv)
    for idx, arg in enumerate(args[1:], start=1):
        if arg.startswith("--dataset="):
            return arg.split("=", 1)[1].strip()
        if arg == "--dataset" and idx + 1 < len(args):
            return args[idx + 1].strip()
    return None


def resolve_dataset_profile_from_cli_env(argv=None, environ=None, *, default=DEFAULT_DATASET_PROFILE, env_var=DEFAULT_DATASET_ENV_VAR) -> Tuple[str, str]:
    cli_value = extract_dataset_cli_value(argv)
    if cli_value:
        return normalize_dataset_profile_key(cli_value, default=default), "CLI"

    env = {} if environ is None else environ
    env_value = env.get(env_var)
    if env_value is not None and str(env_value).strip() != "":
        return normalize_dataset_profile_key(env_value, default=default), "ENV"

    return normalize_dataset_profile_key(default, default=default), "DEFAULT"


def build_validate_dataset_prompt(default=DEFAULT_VALIDATE_DATASET_PROFILE):
    default_key = normalize_dataset_profile_key(default, default=DEFAULT_VALIDATE_DATASET_PROFILE)
    default_menu_key = DATASET_PROFILE_SPECS[default_key]["menu_key"]
    return (
        "👉 0. 驗證資料集 "
        f"(1=縮減 [{DATASET_PROFILE_SPECS[DATASET_PROFILE_REDUCED]['dir_name']}], "
        f"2=完整 [{DATASET_PROFILE_SPECS[DATASET_PROFILE_FULL]['dir_name']}], "
        f"預設 {default_menu_key}): "
    )
