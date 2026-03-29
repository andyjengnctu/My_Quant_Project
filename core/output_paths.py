from __future__ import annotations

import os
from pathlib import Path


def build_output_dir(project_root: str | os.PathLike[str], category: str) -> str:
    return os.path.join(str(project_root), "outputs", category)


def ensure_output_dir(project_root: str | os.PathLike[str], category: str) -> str:
    output_dir = build_output_dir(project_root, category)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def output_dir_path(project_root: str | os.PathLike[str], category: str) -> Path:
    return Path(build_output_dir(project_root, category))
