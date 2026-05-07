import os
from typing import Dict, List, Mapping, Optional

from core.output_paths import build_output_dir
from core.rolling_oos_params import build_rolling_oos_display_label, is_rolling_oos_param_set_file, load_json_file

MODELS_DIR_ENV_VAR = "V16_MODELS_DIR"
RUN_BEST_PARAMS_PATH_ENV_VAR = "V16_RUN_BEST_PARAMS_PATH"
CANDIDATE_BEST_PARAMS_PATH_ENV_VAR = "V16_CANDIDATE_BEST_PARAMS_PATH"
CANDIDATE_RETENTION_BEST_PARAMS_PATH_ENV_VAR = "V16_CANDIDATE_RETENTION_BEST_PARAMS_PATH"
CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH_ENV_VAR = "V16_CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH"

PARAMS_FILENAME_SUFFIX = "_params.json"
CANONICAL_PARAM_FILENAME_LABELS = {
    "run_best_params.json": "run_best | 目前參數",
    "candidate_best_params.json": "candidate_best | 候選參數",
    "candidate_retention_best_params.json": "candidate_retention_best | retention 最大候選",
    "candidate_val_score_best_params.json": "candidate_val_score_best | val_score 第一候選",
}
CANONICAL_PARAM_FILENAME_ORDER = tuple(CANONICAL_PARAM_FILENAME_LABELS.keys())


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


def resolve_run_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(RUN_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return os.path.join(resolve_models_dir(project_root, environ=env), "run_best_params.json")




def resolve_candidate_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(CANDIDATE_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return os.path.join(resolve_models_dir(project_root, environ=env), "candidate_best_params.json")


def resolve_candidate_retention_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(CANDIDATE_RETENTION_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return os.path.join(resolve_models_dir(project_root, environ=env), "candidate_retention_best_params.json")


def resolve_candidate_val_score_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return os.path.join(resolve_models_dir(project_root, environ=env), "candidate_val_score_best_params.json")


def _param_source_key_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    if stem.endswith("_params"):
        return stem[:-len("_params")]
    return stem


def _format_param_source_label(filename: str) -> str:
    basename = os.path.basename(str(filename))
    canonical_label = CANONICAL_PARAM_FILENAME_LABELS.get(basename)
    if canonical_label:
        return canonical_label
    return f"{_param_source_key_from_filename(basename)} | {basename}"


def _canonical_param_source_sort_rank(filename: str) -> int:
    try:
        return CANONICAL_PARAM_FILENAME_ORDER.index(os.path.basename(str(filename)))
    except ValueError:
        return len(CANONICAL_PARAM_FILENAME_ORDER)


def _discover_rolling_oos_param_sets(project_root: str) -> List[Dict[str, str]]:
    search_dirs = [
        os.path.join(build_output_dir(project_root, "optimizer"), "outer_rolling_oos"),
        resolve_models_dir(project_root),
    ]
    records: List[Dict[str, str]] = []
    seen_paths = set()
    for folder in search_dirs:
        try:
            filenames = os.listdir(folder)
        except FileNotFoundError:
            continue
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            path = os.path.abspath(os.path.join(folder, filename))
            if path in seen_paths or not os.path.isfile(path):
                continue
            if not is_rolling_oos_param_set_file(path):
                continue
            try:
                payload = load_json_file(path)
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            seen_paths.add(path)
            selector = str(payload.get("selector") or (payload.get("meta") or {}).get("selector") or _param_source_key_from_filename(filename)).strip()
            records.append({
                "key": f"rolling_oos:{selector}:{os.path.splitext(filename)[0]}",
                "label": build_rolling_oos_display_label(path, payload),
                "path": path,
                "filename": filename,
                "kind": "rolling_oos_param_set",
            })

    records.sort(key=lambda item: (
        -int(os.path.getmtime(item["path"])) if os.path.exists(item["path"]) else 0,
        str(item["label"]).lower(),
    ))
    return records


def discover_model_param_sources(project_root: str, environ: Optional[Mapping[str, str]] = None, *, include_rolling_oos: bool = False) -> List[Dict[str, str]]:
    """Return selectable parameter files that currently exist under models/.

    By default only ``*_params.json`` single-param files are exposed, so optimizer
    summary files do not pollute trading/scanner dropdowns.  Rolling OOS
    validation param sets are opt-in because they are not live-trading params.
    """
    env = os.environ if environ is None else environ
    models_dir = resolve_models_dir(project_root, environ=env)
    records: List[Dict[str, str]] = []
    try:
        filenames = os.listdir(models_dir)
    except FileNotFoundError:
        filenames = []

    for filename in filenames:
        if not filename.endswith(PARAMS_FILENAME_SUFFIX):
            continue
        path = os.path.abspath(os.path.join(models_dir, filename))
        if not os.path.isfile(path):
            continue
        records.append({
            "key": _param_source_key_from_filename(filename),
            "label": _format_param_source_label(filename),
            "path": path,
            "filename": filename,
            "kind": "single_param",
        })

    if include_rolling_oos:
        records.extend(_discover_rolling_oos_param_sets(project_root))

    records.sort(key=lambda item: (
        0 if str(item.get("kind", "single_param")) == "single_param" else 1,
        _canonical_param_source_sort_rank(item["filename"]) if str(item.get("kind", "single_param")) == "single_param" else 0,
        -int(os.path.getmtime(item["path"])) if str(item.get("kind")) == "rolling_oos_param_set" and os.path.exists(item["path"]) else 0,
        str(item["label"]).lower(),
    ))
    return records


def resolve_active_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    return resolve_run_best_params_path(project_root, environ=environ)
