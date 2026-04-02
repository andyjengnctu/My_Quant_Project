from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

PROJECT_SETTINGS_SINGLE_ENTRY_TEXT = "`apps/test_suite.py` 必須作為所有已實作測試的單一正式入口"
CMD_SINGLE_ENTRY_TEXT = "正式對外入口為 `apps/test_suite.py`"
ARCHITECTURE_SINGLE_ENTRY_TEXT = "`apps/test_suite.py` 是日常唯一建議使用的一鍵測試入口"
LEGACY_APP_ENTRY_PATHS = ("apps/local_regression.py", "apps/validate_consistency.py")
SUSPICIOUS_APP_ENTRY_PATTERN = re.compile(r"(?:test|validate|regression|consistency)", re.IGNORECASE)


def summarize_single_formal_test_entry_contract(project_root: Path) -> Dict[str, Any]:
    apps_dir = project_root / "apps"
    app_py_files = sorted(path.name for path in apps_dir.glob("*.py") if path.is_file())
    suspicious_app_entries = [
        name
        for name in app_py_files
        if name != "test_suite.py" and SUSPICIOUS_APP_ENTRY_PATTERN.search(Path(name).stem)
    ]
    legacy_entry_paths = [path for path in LEGACY_APP_ENTRY_PATHS if (project_root / path).exists()]

    project_settings_text = (project_root / "doc" / "PROJECT_SETTINGS.md").read_text(encoding="utf-8")
    cmd_text = (project_root / "doc" / "CMD.md").read_text(encoding="utf-8")
    architecture_text = (project_root / "doc" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    return {
        "test_suite_exists": (project_root / "apps" / "test_suite.py").exists(),
        "app_py_files": app_py_files,
        "suspicious_app_entries": suspicious_app_entries,
        "legacy_entry_paths": legacy_entry_paths,
        "project_settings_declares_single_entry": PROJECT_SETTINGS_SINGLE_ENTRY_TEXT in project_settings_text,
        "cmd_declares_single_entry": CMD_SINGLE_ENTRY_TEXT in cmd_text,
        "architecture_declares_single_entry": ARCHITECTURE_SINGLE_ENTRY_TEXT in architecture_text,
    }
