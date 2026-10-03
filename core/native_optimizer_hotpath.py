"""Optional native optimizer hot-path loader.

The compiled extension is deliberately stored under ``outputs/ml_optimizer/native``
instead of the source tree.  Source-only bundles therefore stay portable, while
machines with a compatible C compiler can build the acceleration locally.

All callers must tolerate this module reporting the native functions as ``None``;
that is the canonical Python fallback path and has identical scientific semantics.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_DIR = _ROOT / "outputs" / "ml_optimizer" / "native"
_MODULE_NAME = "_native_optimizer_hotpath"
_SOURCE = _ROOT / "core" / "native_optimizer_hotpath.c"
_MANIFEST = _NATIVE_DIR / "build_manifest.json"
_NATIVE_MODULE: ModuleType | None = None
_NATIVE_LOAD_ATTEMPTED = False
_NATIVE_LOAD_ERROR: Exception | None = None



def _source_sha256() -> str | None:
    try:
        return hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
    except OSError:
        return None


def _native_build_is_current() -> bool:
    try:
        payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        payload.get("source_sha256") == _source_sha256()
        and payload.get("python_cache_tag") == getattr(sys.implementation, "cache_tag", None)
    )

def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = _NATIVE_DIR / f"{_MODULE_NAME}{suffix}"
        if path.is_file():
            candidates.append(path)
    if not candidates and _NATIVE_DIR.is_dir():
        candidates.extend(sorted(_NATIVE_DIR.glob(f"{_MODULE_NAME}*.pyd")))
        candidates.extend(sorted(_NATIVE_DIR.glob(f"{_MODULE_NAME}*.so")))
    return candidates


def _load_native_module() -> ModuleType | None:
    global _NATIVE_MODULE, _NATIVE_LOAD_ATTEMPTED, _NATIVE_LOAD_ERROR
    if _NATIVE_LOAD_ATTEMPTED:
        return _NATIVE_MODULE
    _NATIVE_LOAD_ATTEMPTED = True
    if not _native_build_is_current():
        return None
    for path in _candidate_paths():
        try:
            spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _NATIVE_MODULE = module
            _NATIVE_LOAD_ERROR = None
            return module
        except Exception as exc:  # pragma: no cover - platform/compiler dependent
            _NATIVE_LOAD_ERROR = exc
    return None


def is_native_optimizer_hotpath_available() -> bool:
    return _load_native_module() is not None


def get_native_optimizer_hotpath_load_error() -> Exception | None:
    _load_native_module()
    return _NATIVE_LOAD_ERROR


_native = _load_native_module()
supertrend_direction_into = None if _native is None else getattr(_native, "supertrend_direction_into", None)
rolling_max_shift1_into = None if _native is None else getattr(_native, "rolling_max_shift1_into", None)
rolling_mean_shift1_into = None if _native is None else getattr(_native, "rolling_mean_shift1_into", None)
rolling_mean_std0_into = None if _native is None else getattr(_native, "rolling_mean_std0_into", None)
scan_position_span = None if _native is None else getattr(_native, "scan_position_span", None)
scan_reentry_watch_span = None if _native is None else getattr(_native, "scan_reentry_watch_span", None)
calc_position_size_exact = None if _native is None else getattr(_native, "calc_position_size_exact", None)
cleanup_live_shadow_inplace = None if _native is None else getattr(_native, "cleanup_live_shadow_inplace", None)
cleanup_live_shadow_batch = None if _native is None else getattr(_native, "cleanup_live_shadow_batch", None)
cleanup_live_shadow_cached_batch = None if _native is None else getattr(_native, "cleanup_live_shadow_cached_batch", None)
resolve_live_extended_candidate_core = None if _native is None else getattr(_native, "resolve_live_extended_candidate_core", None)
resolve_live_extended_candidate_batch_sized = None if _native is None else getattr(_native, "resolve_live_extended_candidate_batch_sized", None)
