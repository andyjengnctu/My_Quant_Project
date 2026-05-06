import os
from typing import Dict, List, Mapping, Optional

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


def discover_model_param_sources(project_root: str, environ: Optional[Mapping[str, str]] = None) -> List[Dict[str, str]]:
    """Return selectable parameter files that currently exist under models/.

    Only ``*_params.json`` files are exposed, so optimizer summary files such as
    ``*_summary.json`` do not pollute the Workbench parameter dropdown.
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
        })

    records.sort(key=lambda item: (
        _canonical_param_source_sort_rank(item["filename"]),
        str(item["label"]).lower(),
    ))
    return records


def resolve_active_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    return resolve_run_best_params_path(project_root, environ=environ)
