import os

from core.v16_params_io import load_params_from_json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "v16_best_params.json")
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
