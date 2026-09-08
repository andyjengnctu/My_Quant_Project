import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

DATASET_PROFILE_REDUCED = "reduced"
DATASET_PROFILE_FULL = "full"
DEFAULT_DATASET_PROFILE = DATASET_PROFILE_FULL
DEFAULT_VALIDATE_DATASET_PROFILE = DATASET_PROFILE_REDUCED
DEFAULT_DATASET_ENV_VAR = "V16_DATASET_PROFILE"
VALIDATE_DATASET_ENV_VAR = "V16_VALIDATE_DATASET"
UNIX_DATASET_ROOT_DIR = "/data"
PROJECT_DATA_DIRNAME = "data"
DATASET_GENERATION_IDENTITY_SCHEMA_VERSION = 1

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


def get_dataset_root_dir(project_root, profile_key=DEFAULT_DATASET_PROFILE):
    normalized_key = normalize_dataset_profile_key(profile_key)
    project_root_abs = os.path.abspath(project_root)
    project_data_root = os.path.join(project_root_abs, PROJECT_DATA_DIRNAME)

    if normalized_key == DATASET_PROFILE_REDUCED:
        return project_data_root

    unix_root = os.path.normpath(UNIX_DATASET_ROOT_DIR)
    if os.path.isdir(unix_root):
        return unix_root

    return project_data_root


def get_unpromoted_dataset_dir(project_root, profile_key):
    """Resolve the physical legacy dataset path without active-generation routing.

    Lifecycle/proof code uses this only when it must reference the persisted
    Research V1 bytes explicitly.  Normal consumers must continue using
    ``get_dataset_dir`` so active Research routing remains centralized.
    """
    normalized_key = normalize_dataset_profile_key(profile_key)
    return os.path.join(
        get_dataset_root_dir(project_root, normalized_key),
        DATASET_PROFILE_SPECS[normalized_key]["dir_name"],
    )


def get_dataset_dir(project_root, profile_key):
    normalized_key = normalize_dataset_profile_key(profile_key)
    if normalized_key == DATASET_PROFILE_FULL:
        # A validated Research V2 promotion atomically redirects only the formal
        # full Research profile.  Reduced remains the stable local test fixture.
        from core.market_data_research_promotion import resolve_promoted_research_full_dataset_dir

        promoted = resolve_promoted_research_full_dataset_dir(project_root)
        if promoted is not None:
            return str(promoted)
    return get_unpromoted_dataset_dir(project_root, normalized_key)



def build_dataset_generation_identity(project_root, profile_key=DEFAULT_DATASET_PROFILE):
    """Return the canonical dataset-generation identity for downstream Research artifacts.

    Legacy V1/reduced profiles intentionally keep their historical physical namespaces.
    A promoted Research V2 full profile carries immutable promotion/materialization lineage
    so downstream caches and artifacts can no longer treat every ``full`` dataset as the
    same scientific source.
    """

    from core.file_integrity import canonical_json_sha256

    profile = normalize_dataset_profile_key(profile_key)
    if profile != DATASET_PROFILE_FULL:
        identity = {
            "schema_version": DATASET_GENERATION_IDENTITY_SCHEMA_VERSION,
            "dataset_profile": profile,
            "generation_id": f"profile:{profile}",
            "frozen_cutoff": None,
        }
    else:
        from config.market_data import RESEARCH_DATA_GENERATION_V2
        from core.market_data_contract import get_active_research_data_generation
        from core.market_data_research_promotion import load_active_research_v2_promotion

        # Artifact namespace/fingerprint resolution needs validated publication lineage,
        # but not a repeated full CSV re-hash.  Normal data routing still performs the
        # Repair-3 content-integrity validation before any materialized bytes are read.
        promoted = load_active_research_v2_promotion(
            project_root,
            required=False,
            verify_materialization_file_content=False,
        )
        if promoted is None:
            active = get_active_research_data_generation()
            identity = {
                "schema_version": DATASET_GENERATION_IDENTITY_SCHEMA_VERSION,
                "dataset_profile": profile,
                "generation_id": str(active.generation_id),
                "frozen_cutoff": None if active.cutoff is None else str(active.cutoff),
            }
        else:
            payload = dict(promoted.payload or {})
            identity = {
                "schema_version": DATASET_GENERATION_IDENTITY_SCHEMA_VERSION,
                "dataset_profile": profile,
                "generation_id": RESEARCH_DATA_GENERATION_V2,
                "frozen_cutoff": str(promoted.frozen_cutoff),
                "promotion_fingerprint": str(promoted.promotion_fingerprint),
                "materialization_fingerprint": str(promoted.materialization_fingerprint),
                "required_source_projection_fingerprint": str(
                    payload.get("required_source_projection_fingerprint") or ""
                ),
                "adjusted_price_revision_proof_fingerprint": str(
                    payload.get("adjusted_price_revision_proof_fingerprint") or ""
                ),
            }

    fingerprint_payload = dict(identity)
    identity["identity_fingerprint"] = canonical_json_sha256(fingerprint_payload)
    return identity


def get_dataset_generation_namespace(project_root, profile_key=DEFAULT_DATASET_PROFILE):
    """Return a physical namespace only when the current dataset truth needs isolation.

    V1 and reduced keep legacy paths for historical compatibility.  Promoted V2 uses
    the complete canonical generation-identity fingerprint so different V2 truths can
    never overwrite each other.
    """

    from config.market_data import RESEARCH_DATA_GENERATION_V2

    identity = build_dataset_generation_identity(project_root, profile_key)
    if (
        identity.get("dataset_profile") == DATASET_PROFILE_FULL
        and identity.get("generation_id") == RESEARCH_DATA_GENERATION_V2
    ):
        return f"research_v2_{identity['identity_fingerprint']}"
    return None


def get_dataset_profile_label(profile_key):
    normalized_key = normalize_dataset_profile_key(profile_key)
    return DATASET_PROFILE_SPECS[normalized_key]["label"]


def infer_dataset_profile_key_from_data_dir(data_dir, default=DEFAULT_DATASET_PROFILE):
    normalized_default = normalize_dataset_profile_key(default, default=DEFAULT_DATASET_PROFILE)
    if data_dir is None:
        return normalized_default

    text = str(data_dir).strip()
    if text == "":
        return normalized_default

    dir_name = os.path.basename(os.path.normpath(text))
    if dir_name == DATASET_PROFILE_SPECS[DATASET_PROFILE_REDUCED]["dir_name"]:
        return DATASET_PROFILE_REDUCED
    if dir_name == DATASET_PROFILE_SPECS[DATASET_PROFILE_FULL]["dir_name"]:
        return DATASET_PROFILE_FULL
    return normalized_default


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



def _display_dataset_path(data_dir):
    from core.path_utils import project_relative_display_path

    project_root = Path(__file__).resolve().parents[1]
    return project_relative_display_path(data_dir, project_root=project_root)


def build_missing_dataset_dir_message(profile_key, data_dir):
    normalized_key = normalize_dataset_profile_key(profile_key)
    display_path = _display_dataset_path(data_dir)
    if normalized_key == DATASET_PROFILE_REDUCED:
        return f"找不到資料夾 {display_path}，請先將 tw_stock_data_vip_reduced 放到 <repo>/data/。"
    return f"找不到資料夾 {display_path}，請先執行 apps/smart_downloader.py！"


def build_empty_dataset_dir_message(profile_key, data_dir):
    normalized_key = normalize_dataset_profile_key(profile_key)
    display_path = _display_dataset_path(data_dir)
    if normalized_key == DATASET_PROFILE_REDUCED:
        return f"資料夾 {display_path} 內沒有任何 CSV 檔案；請先將 tw_stock_data_vip_reduced 放到 <repo>/data/。"
    return f"資料夾 {display_path} 內沒有任何 CSV 檔案。"
