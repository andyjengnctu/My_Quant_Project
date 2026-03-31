from __future__ import annotations

import os
from pathlib import Path


def normalize_output_category(category: str) -> str:
    if not isinstance(category, str):
        raise TypeError(f"category 需要 str，收到 {type(category).__name__}")

    normalized = category.strip()
    if not normalized:
        raise ValueError("category 必填，避免直接寫入 outputs/ 根目錄")

    category_path = Path(normalized)
    if category_path.is_absolute():
        raise ValueError("category 不可為絕對路徑")

    parts = category_path.parts
    if len(parts) != 1 or parts[0] in {"", ".", ".."}:
        raise ValueError("category 只能是 outputs/ 下的單一工具分類資料夾名稱")

    return parts[0]


def build_output_dir(project_root: str | os.PathLike[str], category: str) -> str:
    normalized_category = normalize_output_category(category)
    return os.path.join(str(project_root), "outputs", normalized_category)


def ensure_output_dir(project_root: str | os.PathLike[str], category: str) -> str:
    output_dir = build_output_dir(project_root, category)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def output_dir_path(project_root: str | os.PathLike[str], category: str) -> Path:
    return Path(build_output_dir(project_root, category))
