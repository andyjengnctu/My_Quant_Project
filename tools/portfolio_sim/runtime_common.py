import os

from core.params_io import load_params_from_json
from core.output_paths import build_output_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "portfolio_sim")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params.json")
LOAD_PROGRESS_EVERY = 50


def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_strict_params(json_file):
    return load_params_from_json(json_file)


def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))
