import os
from typing import Dict, Iterable, Optional, Tuple

DATASET_PROFILE_REDUCED = "reduced"
DATASET_PROFILE_FULL = "full"
DEFAULT_DATASET_PROFILE = DATASET_PROFILE_FULL
DEFAULT_VALIDATE_DATASET_PROFILE = DATASET_PROFILE_REDUCED
DEFAULT_DATASET_ENV_VAR = "V16_DATASET_PROFILE"
VALIDATE_DATASET_ENV_VAR = "V16_VALIDATE_DATASET"
UNIX_DATASET_ROOT_DIR = "/data"
PROJECT_DATA_DIRNAME = "data"

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


def get_dataset_root_dir(project_root):
    del project_root
    return os.path.normpath(UNIX_DATASET_ROOT_DIR)


def build_missing_dataset_dir_message(profile_key, data_dir):
    normalized_key = normalize_dataset_profile_key(profile_key)
    if normalized_key == DATASET_PROFILE_REDUCED:
        return f"找不到資料夾 {data_dir}，請確認本輪測試資料 ZIP 已解到該路徑。"
    return f"找不到資料夾 {data_dir}，請先執行 apps/smart_downloader.py 下載完整資料集。"


def build_empty_dataset_dir_message(profile_key, data_dir):
    normalized_key = normalize_dataset_profile_key(profile_key)
    if normalized_key == DATASET_PROFILE_REDUCED:
        return f"資料夾 {data_dir} 內沒有任何 CSV 檔案，請確認本輪測試資料 ZIP 內容是否完整。"
    return f"資料夾 {data_dir} 內沒有任何 CSV 檔案。"


def infer_dataset_profile_key_from_dir(data_dir, default=DEFAULT_DATASET_PROFILE):
    normalized_default = normalize_dataset_profile_key(default, default=default)
    if not data_dir:
        return normalized_default
    base_name = os.path.basename(os.path.normpath(str(data_dir)))
    for key, spec in DATASET_PROFILE_SPECS.items():
        if base_name == spec["dir_name"]:
            return key
    return normalized_default


def get_dataset_dir(project_root, profile_key):
    normalized_key = normalize_dataset_profile_key(profile_key)
    return os.path.join(get_dataset_root_dir(project_root), DATASET_PROFILE_SPECS[normalized_key]["dir_name"])


def get_dataset_profile_label(profile_key):
    normalized_key = normalize_dataset_profile_key(profile_key)
    return DATASET_PROFILE_SPECS[normalized_key]["label"]


def extract_dataset_cli_value(argv: Optional[Iterable[str]]):
    if argv is None:
        return None

    args = list(argv)
    for idx, arg in enumerate(args[1:], start=1):
        if arg.startswith("--dataset="):
            value = arg.split("=", 1)[1].strip()
            if value == "":
                raise ValueError("--dataset= 不能為空，請使用 reduced 或 full")
            return value
        if arg == "--dataset":
            if idx + 1 >= len(args):
                raise ValueError("--dataset 缺少值，請使用 reduced 或 full")
            value = str(args[idx + 1]).strip()
            if value == "":
                raise ValueError("--dataset 缺少值，請使用 reduced 或 full")
            return value
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
