from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements" / "requirements.txt"

MODULE_NAME_MAP = {
    "FinMind": "FinMind",
}


def _read_requirements(requirements_path: Path) -> List[str]:
    packages: List[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ";" in line:
            line = line.split(";", 1)[0].strip()
        for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if marker in line:
                line = line.split(marker, 1)[0].strip()
                break
        if "[" in line:
            line = line.split("[", 1)[0].strip()
        if line:
            packages.append(line)
    return packages


def _module_name(package_name: str) -> str:
    return MODULE_NAME_MAP.get(package_name, package_name.replace("-", "_"))


def _check_module_import(package_name: str) -> Dict[str, Any]:
    module_name = _module_name(package_name)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "package": package_name,
            "module": module_name,
            "status": "FAIL",
            "detail": f"import 失敗: {exc.__class__.__name__}: {exc}",
        }
    version = getattr(module, "__version__", None)
    detail = f"import 成功: {module_name}"
    if version:
        detail += f" {version}"
    return {
        "package": package_name,
        "module": module_name,
        "status": "PASS",
        "detail": detail,
    }


def _check_dir_writable(path: Path, *, label: str) -> Dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    probe_file: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path, prefix=".preflight_", suffix=".tmp", delete=False) as handle:
            handle.write("ok")
            probe_file = handle.name
    except Exception as exc:
        return {
            "name": label,
            "path": str(path),
            "status": "FAIL",
            "detail": f"不可寫: {exc.__class__.__name__}: {exc}",
        }
    finally:
        if probe_file:
            try:
                Path(probe_file).unlink(missing_ok=True)
            except OSError:
                pass
    return {
        "name": label,
        "path": str(path),
        "status": "PASS",
        "detail": "可寫",
    }


def _check_file_exists(path: Path, *, label: str) -> Dict[str, Any]:
    return {
        "name": label,
        "path": str(path),
        "status": "PASS" if path.exists() else "FAIL",
        "detail": "存在" if path.exists() else "不存在",
    }


def run_preflight_checks(
    *,
    project_root: Optional[Path] = None,
    requirements_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    req_path = DEFAULT_REQUIREMENTS_PATH if requirements_path is None else Path(requirements_path).resolve()
    outputs_root = root / "outputs"
    staging_root = outputs_root / "local_regression" / "_staging"

    requirements = _read_requirements(req_path) if req_path.exists() else []
    imports = [_check_module_import(name) for name in requirements]
    import_failures = [item for item in imports if item["status"] != "PASS"]

    path_checks = [
        _check_file_exists(root / "models" / "best_params.json", label="best_params_json"),
        _check_file_exists(req_path, label="requirements_txt"),
        _check_dir_writable(root, label="project_root_writable"),
        _check_dir_writable(outputs_root, label="outputs_root_writable"),
        _check_dir_writable(staging_root, label="local_regression_staging_writable"),
    ]
    path_failures = [item for item in path_checks if item["status"] != "PASS"]

    overall_ok = not import_failures and not path_failures
    status = "PASS" if overall_ok else "FAIL"
    return {
        "status": status,
        "project_root": str(root),
        "requirements_path": str(req_path),
        "requirement_count": len(requirements),
        "imports": imports,
        "paths": path_checks,
        "failed_imports": [item["package"] for item in import_failures],
        "failed_paths": [item["name"] for item in path_failures],
        "detail": "環境可用" if overall_ok else "缺套件或寫入權限不足",
        "guidance": (
            "請先安裝 requirements 並確認 repo / outputs 可寫，再執行動態驗證。"
            if not overall_ok
            else "環境前置檢查通過，可執行 local regression。"
        ),
    }


def write_preflight_summary(path: Path, summary: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="檢查 local regression / test suite 所需環境是否就緒。")
    parser.add_argument("--json-out", dest="json_out", default="", help="輸出 summary JSON 路徑")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_preflight_checks()
    if args.json_out:
        write_preflight_summary(Path(args.json_out), summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


__all__ = ["run_preflight_checks", "write_preflight_summary", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
