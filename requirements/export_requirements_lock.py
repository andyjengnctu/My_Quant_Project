import os
import re
from importlib.metadata import PackageNotFoundError, version


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")
LOCK_FILE = os.path.join(BASE_DIR, "requirements-lock.txt")


def normalize_requirement_name(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    pkg = re.split(r'[<>=!~;\[]', line, maxsplit=1)[0].strip()
    return pkg or None


def read_requirement_names(requirements_file: str) -> list[str]:
    if not os.path.exists(requirements_file):
        raise FileNotFoundError(f"找不到 requirements.txt: {requirements_file}")

    names: list[str] = []
    with open(requirements_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            pkg = normalize_requirement_name(raw_line)
            if pkg is not None:
                names.append(pkg)

    if not names:
        raise RuntimeError(f"{requirements_file} 沒有可用的套件名稱")

    return names


def resolve_locked_requirements(package_names: list[str]) -> tuple[list[str], list[str]]:
    locked_lines: list[str] = []
    missing: list[str] = []

    for pkg in package_names:
        try:
            ver = version(pkg)
            locked_lines.append(f"{pkg}=={ver}")
        except PackageNotFoundError:
            missing.append(pkg)

    return locked_lines, missing


def write_lock_file(lock_file: str, locked_lines: list[str]) -> None:
    with open(lock_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(locked_lines) + '\n')


def main() -> None:
    package_names = read_requirement_names(REQUIREMENTS_FILE)
    locked_lines, missing = resolve_locked_requirements(package_names)

    if missing:
        missing_str = ', '.join(missing)
        raise RuntimeError(
            "以下套件尚未安裝，無法輸出 requirements-lock.txt: "
            f"{missing_str}"
        )

    write_lock_file(LOCK_FILE, locked_lines)
    print(f"✅ 已輸出鎖版本檔：{LOCK_FILE}")


if __name__ == "__main__":
    main()