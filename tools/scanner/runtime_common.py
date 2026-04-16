import os

from core.model_paths import resolve_champion_params_path, resolve_models_dir
from core.params_io import load_params_from_json
from core.output_paths import build_output_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "vip_scanner")
MODELS_DIR = resolve_models_dir(PROJECT_ROOT)
CHAMPION_PARAMS_PATH = resolve_champion_params_path(PROJECT_ROOT)
SCANNER_PROGRESS_EVERY = 25
DEFAULT_SCANNER_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1) // 2))


def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def resolve_scanner_max_workers(params):
    configured = getattr(params, 'scanner_max_workers', DEFAULT_SCANNER_MAX_WORKERS)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_SCANNER_MAX_WORKERS
    return max(1, configured)


def load_strict_params(json_file):
    return load_params_from_json(json_file)


def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))
