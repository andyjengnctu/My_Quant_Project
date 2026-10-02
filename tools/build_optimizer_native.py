"""Build the optional optimizer native C extension under the output domain.

The Research runtime always has a Python fallback.  This builder only creates
the optional optimizer native extension under ``outputs/ml_optimizer/native`` for machines with a supported C compiler.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "core" / "native_optimizer_hotpath.c"


def _compile_args() -> list[str]:
    if os.name == "nt":
        return ["/O2", "/fp:strict"]
    return ["-O3", "-fno-fast-math", "-ffp-contract=off"]


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    native_dir = ROOT / "outputs" / "ml_optimizer" / "native"
    build_temp = ROOT / "outputs" / "ml_optimizer" / "native_build_temp"
    native_dir.mkdir(parents=True, exist_ok=True)
    build_temp.mkdir(parents=True, exist_ok=True)
    for stale_path in list(native_dir.glob("_native_optimizer_hotpath*.pyd")) + list(native_dir.glob("_native_optimizer_hotpath*.so")):
        stale_path.unlink(missing_ok=True)
    ext = Extension(
        "_native_optimizer_hotpath",
        [str(SOURCE)],
        extra_compile_args=_compile_args(),
    )
    original_argv = list(sys.argv)
    try:
        sys.argv[:] = [
            str(Path(__file__).name),
            "build_ext",
            "--build-lib",
            str(native_dir),
            "--build-temp",
            str(build_temp),
        ]
        setup(
            name="optimizer-native-hotpath",
            version="0.1.0",
            ext_modules=[ext],
            script_args=sys.argv[1:],
        )
    finally:
        sys.argv[:] = original_argv
    manifest = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "python_cache_tag": getattr(sys.implementation, "cache_tag", None),
        "platform": sys.platform,
        "machine": platform.machine(),
    }
    (native_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"optimizer native extension build: DONE | output={native_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
