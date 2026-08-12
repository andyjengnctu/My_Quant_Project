from core.runtime_utils import is_insufficient_data_error
import os

from core.model_paths import resolve_active_params_path, resolve_models_dir
from core.params_io import load_params_from_json as load_strict_params
from core.output_paths import build_output_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "portfolio_sim")
MODELS_DIR = resolve_models_dir(PROJECT_ROOT)
ACTIVE_PARAMS_PATH = resolve_active_params_path(PROJECT_ROOT)
LOAD_PROGRESS_EVERY = 50


def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


