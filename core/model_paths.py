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
STRATEGY_PARAM_REPOSITORY_DIRNAME = "strategy_params"
STRATEGY_PARAM_TRADE_STATE_RELATIVE_DIR = os.path.join("strategy_params", "full", "trade", "state")
CANONICAL_PARAM_FILENAME_LABELS = {
    "run_best_params.json": "run_best_params.json",
    "candidate_best_params.json": "candidate_best_params.json",
    "candidate_retention_best_params.json": "candidate_retention_best_params.json",
    "candidate_val_score_best_params.json": "candidate_val_score_best_params.json",
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


def _resolve_trade_state_path(project_root: str, filename: str, *, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    return os.path.join(
        resolve_models_dir(project_root, environ=env),
        STRATEGY_PARAM_TRADE_STATE_RELATIVE_DIR,
        str(filename),
    )


def resolve_run_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(RUN_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return _resolve_trade_state_path(project_root, "active.json", environ=env)




def resolve_candidate_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(CANDIDATE_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return _resolve_trade_state_path(project_root, "candidate_best.json", environ=env)


def resolve_candidate_retention_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(CANDIDATE_RETENTION_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return _resolve_trade_state_path(project_root, "candidate_retention_best.json", environ=env)


def resolve_candidate_val_score_best_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    override = str(env.get(CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        return _resolve_override_path(project_root, override)
    return _resolve_trade_state_path(project_root, "candidate_val_score_best.json", environ=env)


def _param_source_key_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    if stem.endswith("_params"):
        return stem[:-len("_params")]
    return stem


def _format_param_source_label(filename: str) -> str:
    basename = os.path.basename(str(filename))
    return CANONICAL_PARAM_FILENAME_LABELS.get(basename, basename)


def _strategy_param_repository_dirs(project_root: str, environ: Optional[Mapping[str, str]] = None) -> List[str]:
    env = os.environ if environ is None else environ
    base = os.path.join(resolve_models_dir(project_root, environ=env), STRATEGY_PARAM_REPOSITORY_DIRNAME)
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
    """Return selectable canonical strategy-parameter artifacts.

    Current discovery never scans legacy JSON files from ``models/`` root.  The
    default live parameter state and all first-class policy artifacts live below
    ``models/strategy_params/``; optimizer ``outputs/`` are validation-only extras.
    """
    env = os.environ if environ is None else environ
    records: List[Dict[str, str]] = []
    active_path = resolve_run_best_params_path(project_root, environ=env)
    if os.path.isfile(active_path):
        records.append(_model_record_for_path(active_path, key="run_best", label="strategy_params/full/trade/state/active.json"))

    if include_active_param_ensemble:
        records.extend(_discover_active_param_ensemble_sets(project_root, environ=env))

    if include_rolling_oos:
        records.extend(_discover_rolling_oos_param_sets(project_root, environ=env))

    seen: set[str] = set()
    deduped: List[Dict[str, str]] = []
    for record in records:
        path = os.path.abspath(str(record.get("path") or ""))
        if not path or path in seen:
            continue
        seen.add(path)
        deduped.append(record)
    kind_rank = {"single_param": 0, "active_param_ensemble": 1, "rolling_oos_param_set": 2}
    deduped.sort(key=lambda item: (
        0 if str(item.get("key")) == "run_best" else 1,
        kind_rank.get(str(item.get("kind", "single_param")), 99),
        -int(os.path.getmtime(item["path"])) if os.path.exists(item["path"]) else 0,
        str(item["label"]).lower(),
    ))
    return deduped



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

    ``V16_RUN_BEST_PARAMS_PATH`` remains a test/compatibility override.  Without
    that override, the current source always resolves to the Optimizer-owned
    ``models/strategy_params/full/trade/state/active.json`` artifact.
    """
    env = os.environ if environ is None else environ
    override = str(env.get(RUN_BEST_PARAMS_PATH_ENV_VAR, "")).strip()
    if override != "":
        path = _resolve_override_path(project_root, override)
        return _model_record_for_path(path, key="run_best_override", label=os.path.basename(path))

    return _model_record_for_path(
        resolve_run_best_params_path(project_root, environ=env),
        key="run_best",
        label="strategy_params/full/trade/state/active.json",
    )


def resolve_default_primary_param_source_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    return str(resolve_default_primary_param_source_record(project_root, environ=environ)["path"])


def resolve_active_params_path(project_root: str, environ: Optional[Mapping[str, str]] = None) -> str:
    return resolve_default_primary_param_source_path(project_root, environ=environ)
