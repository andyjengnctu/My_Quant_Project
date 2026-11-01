"""Build the optional optimizer native C extension under the output domain.

The Research runtime always has a Python fallback.  This builder only creates
the optional optimizer native extension under ``outputs/ml_optimizer/native`` for machines with a supported C compiler.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "core" / "native_optimizer_hotpath.c"


def _compile_args() -> list[str]:
    if os.name == "nt":
        return ["/O2", "/fp:strict"]
    return ["-O3", "-fno-fast-math", "-ffp-contract=off"]


def _validate_compiled_binary(binary: Path) -> None:
    """Import the actual staged ABI in an isolated interpreter before publish.

    A build can return successfully while producing an unloadable extension;
    that must never replace the last working binary. Keep the smoke API list
    minimal: numerical parity remains owned by the optimizer regressions.
    """
    probe = r"""
import importlib.util
import sys
from pathlib import Path
path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_native_optimizer_hotpath", path)
if spec is None or spec.loader is None:
    raise RuntimeError("No extension loader for staged native binary")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("rolling_max_shift1_into", "calc_position_size_exact", "scan_position_span"):
    if not callable(getattr(module, name, None)):
        raise RuntimeError(f"Missing optimizer native ABI method: {name}")
"""
    subprocess.run(
        [sys.executable, "-B", "-c", probe, str(binary)],
        check=True, capture_output=True, text=True, timeout=45,
    )


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    native_dir = ROOT / "outputs" / "ml_optimizer" / "native"
    build_parent = ROOT / "outputs" / "ml_optimizer"
    build_parent.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    ext = Extension(
        "_native_optimizer_hotpath",
        [str(SOURCE)],
        extra_compile_args=_compile_args(),
    )
    # Never delete a working .pyd/.so before a new build is compiled, imported,
    # checked and ready. Windows may refuse replacement while the DLL is loaded;
    # that refusal leaves the installed binary and manifest untouched.
    with tempfile.TemporaryDirectory(prefix=".native_build_", dir=str(build_parent)) as tmp:
        staging = Path(tmp)
        stage_lib, stage_temp = staging / "lib", staging / "object"
        stage_lib.mkdir()
        stage_temp.mkdir()
        original_argv = list(sys.argv)
        try:
            sys.argv[:] = [
                str(Path(__file__).name), "build_ext",
                "--build-lib", str(stage_lib), "--build-temp", str(stage_temp),
            ]
            setup(
                name="optimizer-native-hotpath", version="0.1.0",
                ext_modules=[ext], script_args=sys.argv[1:],
            )
        finally:
            sys.argv[:] = original_argv
        binaries = sorted(stage_lib.glob("_native_optimizer_hotpath*.pyd")) + sorted(
            stage_lib.glob("_native_optimizer_hotpath*.so"))
        if len(binaries) != 1:
            raise RuntimeError("Optimizer native build did not produce exactly one extension")
        compiled = binaries[0]
        _validate_compiled_binary(compiled)
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError("Optimizer native source changed during staged compilation")
        manifest = {
            "schema_version": 1,
            "source_sha256": source_sha,
            "python_cache_tag": getattr(sys.implementation, "cache_tag", None),
            "platform": sys.platform,
            "machine": platform.machine(),
        }
        # Preserve both previous files for rollback if manifest publication
        # fails after the binary was replaced. Never remove other ABI builds.
        destination = native_dir / compiled.name
        manifest_path = native_dir / "build_manifest.json"
        old_binary = staging / "old_binary"
        old_manifest = staging / "old_manifest"
        if destination.is_file():
            shutil.copy2(destination, old_binary)
        if manifest_path.is_file():
            shutil.copy2(manifest_path, old_manifest)
        staged_manifest = staging / "build_manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        published = False
        try:
            os.replace(compiled, destination)
            published = True
            os.replace(staged_manifest, manifest_path)
        except OSError:
            if published:
                if old_binary.is_file():
                    os.replace(old_binary, destination)
                else:
                    destination.unlink(missing_ok=True)
                if old_manifest.is_file():
                    os.replace(old_manifest, manifest_path)
                else:
                    manifest_path.unlink(missing_ok=True)
            raise
    print(f"optimizer native extension build: DONE | output={native_dir}")
    print("Restart Python processes to load the freshly compiled native extension.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
