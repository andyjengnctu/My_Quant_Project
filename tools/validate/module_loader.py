import importlib.util
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_path_separators(path):
    return str(path or '').strip().replace("\\", "/")


def get_project_root_path():
    root = PROJECT_ROOT
    return root if isinstance(root, Path) else Path(str(root))


def build_project_absolute_path(*parts):
    return get_project_root_path().joinpath(*parts)


def normalize_project_relative_path(path):
    raw_path = str(path or '').strip()
    if not raw_path:
        return ''

    normalized = _normalize_path_separators(raw_path)
    project_root_norm = _normalize_path_separators(get_project_root_path())

    if normalized.startswith(project_root_norm + '/'):
        return normalized[len(project_root_norm) + 1 :]

    absolute_norm = _normalize_path_separators(os.path.abspath(raw_path))
    if absolute_norm.startswith(project_root_norm + '/'):
        return absolute_norm[len(project_root_norm) + 1 :]

    return normalized


VALIDATION_RECOVERABLE_EXCEPTIONS = (
    AssertionError,
    ArithmeticError,
    AttributeError,
    ImportError,
    LookupError,
    NameError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
)

MODULE_LOAD_RECOVERABLE_EXCEPTIONS = VALIDATION_RECOVERABLE_EXCEPTIONS + (
    SyntaxError,
)

MODULE_CACHE = {}


def load_module_from_candidates(cache_key, candidate_files, required_attrs):
    if cache_key in MODULE_CACHE:
        return MODULE_CACHE[cache_key]

    checked_paths = []
    rejected_paths = []

    for file_name in candidate_files:
        module_path = build_project_absolute_path(file_name)
        display_module_path = normalize_project_relative_path(module_path)
        checked_paths.append(display_module_path)

        if not os.path.exists(module_path):
            continue

        spec = importlib.util.spec_from_file_location(cache_key, module_path)
        if spec is None or spec.loader is None:
            rejected_paths.append(f"{display_module_path} -> 無法建立 spec/loader")
            continue

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except MODULE_LOAD_RECOVERABLE_EXCEPTIONS as e:
            rejected_paths.append(f"{display_module_path} -> 載入失敗: {type(e).__name__}: {e}")
            continue

        missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
        if missing_attrs:
            rejected_paths.append(f"{display_module_path} -> 缺少必要屬性: {missing_attrs}")
            continue

        MODULE_CACHE[cache_key] = (module, display_module_path)
        return module, display_module_path

    detail_msg = "；".join(rejected_paths) if rejected_paths else "沒有任何可用候選檔"
    raise FileNotFoundError(
        f"找不到符合條件的模組。檢查路徑: {checked_paths}。原因: {detail_msg}"
    )
