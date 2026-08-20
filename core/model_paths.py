import os
from typing import Dict, List, Mapping, Optional

from core.active_param_ensemble import is_active_param_ensemble_file, load_json_file as load_ensemble_json_file
from core.output_paths import build_output_dir
from core.rolling_oos_params import is_rolling_oos_param_set_file, load_json_file

MODELS_DIR_ENV_VAR = "V16_MODELS_DIR"
RUN_BEST_PARAMS_PATH_ENV_VAR = "V16_RUN_BEST_PARAMS_PATH"
CANDIDATE_BEST_PARAMS_PATH_ENV_VAR = "V16_CANDIDATE_BEST_PARAMS_PATH"
CANDIDATE_RETENTION_BEST_PARAMS_PATH_ENV_VAR = "V16_CANDIDATE_RETENTION_BEST_PARAMS_PATH"
CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH_ENV_VAR = "V16_CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH"

PARAMS_FILENAME_SUFFIX = "_params.json"
CANONICAL_PARAM_FILENAME_LABELS = {
    "run_best_params.json": "run_best_params.json",
    "candidate_best_params.json": "candidate_best_params.json",
    "candidate_retention_best_params.json": "candidate_retention_best_params.json",
    "candidate_val_score_best_params.json": "candidate_val_score_best_params.json",
}
CANONICAL_PARAM_FILENAME_ORDER = tuple(CANONICAL_PARAM_FILENAME_LABELS.keys())

PREFERRED_PRIMARY_PARAM_SOURCE_FILENAMES = (
    "run_best_params.json",
    "candidate_best_params.json",
    "trade_base_best.json",
    "trade_local_best.json",
    "trade_retention_best.json",
    "trade_base_finalists_agree.json",
    "trade_local_finalists_agree.json",
    "trade_retention_finalists_agree.json",
    "trade_ensemble_base.json",
    "trade_ensemble_local.json",
    "trade_ensemble_retention.json",
    "full_base_best.json",
    "full_local_best.json",
    "full_retention_best.json",
    "full_base_finalists_agree.json",
    "full_local_finalists_agree.json",
    "full_retention_finalists_agree.json",
    "full_ensemble_base.json",
    "full_ensemble_local.json",
    "full_ensemble_retention.json",
    "base_best.json",
    "local_best.json",
    "retention_best.json",
    "base_finalists_agree.json",
    "local_finalists_agree.json",
    "retention_finalists_agree.json",
    "base.json",
    "local.json",
    "retention.json",
    "candidate_retention_best_params.json",
    "oos_base_best.json",
    "oos_local_best.json",
    "oos_retention_best.json",
    "oos_base_finalists_agree.json",
    "oos_local_finalists_agree.json",
    "oos_retention_finalists_agree.json",
    "oos_ensemble_base.json",
    "oos_ensemble_local.json",
    "oos_ensemble_retention.json",
    "roos_base_best.json",
    "roos_local_best.json",
    "roos_retention_best.json",
    "roos_base_finalists_agree.json",
    "roos_local_finalists_agree.json",
    "roos_retention_finalists_agree.json",
    "roos_ensemble_base.json",
    "roos_ensemble_local.json",
    "roos_ensemble_retention.json",
)


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
    return CANONICAL_PARAM_FILENAME_LABELS.get(basename, basename)


def _canonical_param_source_sort_rank(filename: str) -> int:
    try:
        return CANONICAL_PARAM_FILENAME_ORDER.index(os.path.basename(str(filename)))
    except ValueError:
        return len(CANONICAL_PARAM_FILENAME_ORDER)


def _strategy_param_repository_dirs(project_root: str, environ: Optional[Mapping[str, str]] = None) -> List[str]:
    env = os.environ if environ is None else environ
    base = os.path.join(resolve_models_dir(project_root, environ=env), "strategy_params")
    if not os.path.isdir(base):
        return []
    folders: List[str] = []
    for folder, _dirs, _files in os.walk(base):
        folders.append(folder)
    return folders


def _discover_active_param_ensemble_sets(project_root: str, environ: Optional[Mapping[str, str]] = None) -> List[Dict[str, str]]:
    env = os.environ if environ is None else environ
    search_dirs = [
        *_strategy_param_repository_dirs(project_root, environ=env),
        resolve_models_dir(project_root, environ=env),
        os.path.join(build_output_dir(project_root, "optimizer"), "outer_rolling_oos"),
    ]
    records: List[Dict[str, str]] = []
    seen_paths = set()
    seen_labels = set()
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
            if not is_active_param_ensemble_file(path):
                continue
            try:
                payload = load_ensemble_json_file(path)
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            models_dir = resolve_models_dir(project_root, environ=env)
            try:
                relative_label = os.path.relpath(path, models_dir).replace(os.sep, "/")
            except ValueError:
                relative_label = os.path.basename(filename)
            label = relative_label if relative_label.startswith("strategy_params/") else os.path.basename(filename)
            if label in seen_labels:
                continue
            seen_paths.add(path)
            seen_labels.add(label)
            selector = str(payload.get("selector") or (payload.get("meta") or {}).get("selector") or _param_source_key_from_filename(filename)).strip()
            mode = str(payload.get("mode") or "ensemble").strip() or "ensemble"
            records.append({
                "key": f"active_param_ensemble:{mode}:{selector}:{os.path.splitext(filename)[0]}",
                "label": label,
                "path": path,
                "filename": filename,
                "kind": "active_param_ensemble",
            })

    records.sort(key=lambda item: (
        -int(os.path.getmtime(item["path"])) if os.path.exists(item["path"]) else 0,
        str(item["label"]).lower(),
    ))
    return records


def _discover_rolling_oos_param_sets(project_root: str, environ: Optional[Mapping[str, str]] = None) -> List[Dict[str, str]]:
    # Workbench 下拉選單以「實際檔名」作為唯一顯示名稱；models/ 為主來源，
    # outputs/ 只補充尚未複製到 models/ 的 rolling OOS 參數組。
    env = os.environ if environ is None else environ
    search_dirs = [
        *_strategy_param_repository_dirs(project_root, environ=env),
        resolve_models_dir(project_root, environ=env),
        os.path.join(build_output_dir(project_root, "optimizer"), "outer_rolling_oos"),
    ]
    records: List[Dict[str, str]] = []
    seen_paths = set()
    seen_labels = set()
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
            models_dir = resolve_models_dir(project_root, environ=env)
            try:
                relative_label = os.path.relpath(path, models_dir).replace(os.sep, "/")
            except ValueError:
                relative_label = os.path.basename(filename)
            label = relative_label if relative_label.startswith("strategy_params/") else os.path.basename(filename)
            if label in seen_labels:
                continue
            seen_paths.add(path)
            seen_labels.add(label)
            selector = str(payload.get("selector") or (payload.get("meta") or {}).get("selector") or _param_source_key_from_filename(filename)).strip()
            records.append({
                "key": f"rolling_oos:{selector}:{os.path.splitext(filename)[0]}",
                "label": label,
                "path": path,
                "filename": filename,
                "kind": "rolling_oos_param_set",
            })

    records.sort(key=lambda item: (
        -int(os.path.getmtime(item["path"])) if os.path.exists(item["path"]) else 0,
        str(item["label"]).lower(),
    ))
    return records


def discover_model_param_sources(project_root: str, environ: Optional[Mapping[str, str]] = None, *, include_rolling_oos: bool = False, include_active_param_ensemble: bool = False) -> List[Dict[str, str]]:
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
        if is_active_param_ensemble_file(path) or is_rolling_oos_param_set_file(path):
            continue
        records.append({
            "key": _param_source_key_from_filename(filename),
            "label": _format_param_source_label(filename),
            "path": path,
            "filename": filename,
            "kind": "single_param",
        })

    if include_active_param_ensemble:
        records.extend(_discover_active_param_ensemble_sets(project_root, environ=env))

    if include_rolling_oos:
        records.extend(_discover_rolling_oos_param_sets(project_root, environ=env))

    kind_rank = {"single_param": 0, "active_param_ensemble": 1, "rolling_oos_param_set": 2}
    records.sort(key=lambda item: (
        kind_rank.get(str(item.get("kind", "single_param")), 99),
        _canonical_param_source_sort_rank(item["filename"]) if str(item.get("kind", "single_param")) == "single_param" else 0,
        -int(os.path.getmtime(item["path"])) if str(item.get("kind")) in {"active_param_ensemble", "rolling_oos_param_set"} and os.path.exists(item["path"]) else 0,
        str(item["label"]).lower(),
    ))
    return records



def _model_record_for_path(path: str, *, key: str, label: Optional[str] = None) -> Dict[str, str]:
    filename = os.path.basename(str(path))
    kind = "single_param"
    try:
        if is_active_param_ensemble_file(path):
            kind = "active_param_ensemble"
        elif is_rolling_oos_param_set_file(path):
            kind = "rolling_oos_param_set"
    except (OSError, UnicodeDecodeError, ValueError):
        kind = "single_param"
    return {
        "key": key,
        "label": str(label or _format_param_source_label(filename)),
        "path": os.path.abspath(str(path)),
        "filename": filename,
        "kind": kind,
    }


def resolve_default_primary_param_source_record(project_root: str, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Return the canonical primary parameter source record.

    ``V16_RUN_BEST_PARAMS_PATH`` is the only supported runtime override.  Without
    that override, the canonical source always resolves to
    ``models/run_best_params.json`` even when the optional artifact has not been
    created yet.  Discovery of other existing parameter artifacts belongs to
    ``discover_model_param_sources()`` and must not silently change the default
    runtime source.
    """
    env = os.environ if environ is None else environ
    override = str(env.get(RUN_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        path = _resolve_override_path(project_root, override)
        return _model_record_for_path(path, key="run_best_override", label=os.path.basename(path))

    return _model_record_for_path(
        resolve_run_best_params_path(project_root, environ=env),
        key="run_best",
        label="run_best_params.json",
    )


def resolve_default_primary_param_source_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    return str(resolve_default_primary_param_source_record(project_root, environ=environ)["path"])


def resolve_active_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    return resolve_default_primary_param_source_path(project_root, environ=environ)
