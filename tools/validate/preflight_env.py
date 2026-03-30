from __future__ import annotations

import importlib
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import has_help_flag

REQUIREMENTS_PATH = PROJECT_ROOT / "requirements" / "requirements.txt"
_IMPORT_NAME_OVERRIDES = {
    "python-dateutil": "dateutil",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
}


def _strip_requirement_name(line: str) -> str:
    text = line.split("#", 1)[0].strip()
    if not text or text.startswith(("-r", "--")):
        return ""
    text = text.split(";", 1)[0].strip()
    text = re.split(r"[<>=!~]", text, maxsplit=1)[0].strip()
    text = text.split("[", 1)[0].strip()
    return text


def load_requirement_names(requirements_path: Path = REQUIREMENTS_PATH) -> List[str]:
    if not requirements_path.is_file():
        raise FileNotFoundError(f"requirements 檔不存在: {requirements_path}")

    names: List[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        package_name = _strip_requirement_name(raw_line)
        if package_name:
            names.append(package_name)
    return names


def _resolve_import_name(distribution_name: str) -> str:
    key = distribution_name.strip().lower()
    return _IMPORT_NAME_OVERRIDES.get(key, distribution_name.strip().replace("-", "_"))


def _check_distribution(distribution_name: str) -> Tuple[bool, str]:
    try:
        return True, metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return False, ""


def _check_import(import_name: str) -> Tuple[bool, str]:
    try:
        module = importlib.import_module(import_name)
        module_path = getattr(module, "__file__", "") or ""
        return True, str(module_path)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_preflight(requirements_path: Path = REQUIREMENTS_PATH) -> Dict[str, Any]:
    requirement_names = load_requirement_names(requirements_path)
    checks: List[Dict[str, Any]] = []
    failed_packages: List[str] = []

    for distribution_name in requirement_names:
        import_name = _resolve_import_name(distribution_name)
        installed, version_text = _check_distribution(distribution_name)
        import_ok, import_detail = _check_import(import_name)
        status = "PASS" if installed and import_ok else "FAIL"
        if status != "PASS":
            failed_packages.append(distribution_name)
        checks.append({
            "distribution": distribution_name,
            "import_name": import_name,
            "status": status,
            "installed": installed,
            "version": version_text,
            "import_ok": import_ok,
            "detail": import_detail,
        })

    return {
        "status": "PASS" if not failed_packages else "FAIL",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "requirements_path": str(requirements_path),
        "checked_packages": requirement_names,
        "failed_packages": failed_packages,
        "checks": checks,
    }


def format_preflight_summary(payload: Dict[str, Any]) -> str:
    lines = [
        f"status          : {payload['status']}",
        f"python          : {payload['python_executable']}",
        f"python_version  : {payload['python_version']}",
        f"requirements    : {payload['requirements_path']}",
    ]
    failed_packages = payload.get("failed_packages", [])
    if failed_packages:
        lines.append(f"failed_packages : {', '.join(failed_packages)}")
    else:
        lines.append("failed_packages : (none)")
    return "\n".join(lines)


def main() -> int:
    if has_help_flag(sys.argv):
        print("用法: python tools/validate/preflight_env.py")
        print("說明: 只檢查目前 Python 環境是否已具備 requirements 所需套件；不自動安裝。")
        return 0

    payload = run_preflight()
    print(format_preflight_summary(payload))
    if payload["status"] != "PASS":
        print("\n[詳細結果]")
        for item in payload["checks"]:
            if item["status"] == "PASS":
                continue
            print(
                f"- {item['distribution']} ({item['import_name']}) | "
                f"installed={item['installed']} | import_ok={item['import_ok']} | {item['detail']}"
            )
    return 0 if payload["status"] == "PASS" else 1


__all__ = [
    "REQUIREMENTS_PATH",
    "format_preflight_summary",
    "load_requirement_names",
    "run_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
