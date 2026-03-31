from __future__ import annotations

import importlib
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import run_cli_entrypoint, has_help_flag, resolve_cli_program_name, validate_cli_args

REQUIREMENTS_PATH = PROJECT_ROOT / "requirements" / "requirements.txt"
_IMPORT_NAME_OVERRIDES = {
    "python-dateutil": "dateutil",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "sqlalchemy": "sqlalchemy",
}
_LOCAL_REGRESSION_STEP_ORDER = ("quick_gate", "consistency", "chain_checks", "ml_smoke")
_LOCAL_REGRESSION_STEP_REQUIREMENTS = {
    "quick_gate": {"numpy", "pandas", "openpyxl"},
    "consistency": {"numpy", "pandas", "openpyxl"},
    "chain_checks": {"numpy", "pandas", "openpyxl"},
    "ml_smoke": {"numpy", "pandas", "openpyxl", "optuna", "SQLAlchemy"},
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




def _normalize_local_regression_steps(selected_steps: Optional[Iterable[str]]) -> List[str]:
    if selected_steps is None:
        return []

    normalized: List[str] = []
    seen = set()
    invalid: List[str] = []
    for raw_name in selected_steps:
        name = str(raw_name).strip()
        if not name:
            continue
        if name not in _LOCAL_REGRESSION_STEP_ORDER:
            invalid.append(name)
            continue
        if name in seen:
            continue
        normalized.append(name)
        seen.add(name)

    if invalid:
        valid_text = ", ".join(_LOCAL_REGRESSION_STEP_ORDER)
        raise ValueError(f"--steps 只接受 {valid_text}，收到: {', '.join(invalid)}")
    return normalized


def _resolve_checked_requirement_names(requirement_names: List[str], selected_steps: Optional[Iterable[str]]) -> Tuple[List[str], List[str], str]:
    normalized_steps = _normalize_local_regression_steps(selected_steps)
    if not normalized_steps:
        return list(requirement_names), [], "full_requirements"

    required_packages = set()
    for step_name in normalized_steps:
        required_packages.update(_LOCAL_REGRESSION_STEP_REQUIREMENTS[step_name])

    selected_requirements = [name for name in requirement_names if name in required_packages]
    return selected_requirements, normalized_steps, "local_regression_steps"


def _parse_cli_steps(argv) -> Optional[List[str]]:
    args = list([] if argv is None else argv[1:])
    for idx, arg in enumerate(args):
        text = str(arg).strip()
        if text.startswith("--steps="):
            raw_value = text.split("=", 1)[1].strip()
            return [item.strip() for item in raw_value.split(",")]
        if text == "--steps":
            raw_value = str(args[idx + 1]).strip()
            return [item.strip() for item in raw_value.split(",")]
    return None


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


def run_preflight(requirements_path: Path = REQUIREMENTS_PATH, *, selected_steps: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    requirement_names = load_requirement_names(requirements_path)
    checked_packages, normalized_steps, mode = _resolve_checked_requirement_names(requirement_names, selected_steps)
    checks: List[Dict[str, Any]] = []
    failed_packages: List[str] = []

    for distribution_name in checked_packages:
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
        "mode": mode,
        "selected_steps": normalized_steps,
        "checked_packages": checked_packages,
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
    selected_steps = payload.get("selected_steps", [])
    if selected_steps:
        lines.append(f"mode            : {payload.get('mode', 'local_regression_steps')}")
        lines.append(f"selected_steps  : {', '.join(str(item) for item in selected_steps)}")
    else:
        lines.append(f"mode            : {payload.get('mode', 'full_requirements')}")
    failed_packages = payload.get("failed_packages", [])
    if failed_packages:
        lines.append(f"failed_packages : {', '.join(failed_packages)}")
    else:
        lines.append("failed_packages : (none)")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv, value_options=("--steps",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/validate/preflight_env.py")
        print(f"用法: python {program_name} [--steps quick_gate,consistency,chain_checks,ml_smoke]")
        print("說明: 預設檢查 requirements 全部套件；若指定 --steps，則只檢查 local regression 所選步驟所需套件；不自動安裝。")
        return 0

    payload = run_preflight(selected_steps=_parse_cli_steps(argv))
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
    run_cli_entrypoint(main)
