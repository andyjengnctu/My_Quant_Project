"""Read-only access to retired Runtime Promotion Gate status and reports."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from config.breakout_quality import BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE
from core.strategy_compare_policy import get_strategy_runtime_integration_settings
from core.console_report import render_table, render_title
from core.file_integrity import canonical_json_sha256, exclusive_file_lock, sibling_publication_lock_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATE_SCHEMA_VERSION = 2
RESULT_FILENAME = "runtime_integration.json"
REPORT_FILENAME = "runtime_integration.md"

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON根節點必須是object: {path}")
    return payload


def collect_runtime_integration_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Report retirement without loading or executing historical research catalogs."""
    cfg = get_strategy_runtime_integration_settings()
    return {
        "schema_version": GATE_SCHEMA_VERSION, "status": "BLOCKED",
        "generated_at": datetime.now().astimezone().isoformat(),
        "settings": cfg.as_dict(), "candidate_contract": {},
        "current_workflow_experiment_profile": str(BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE),
        "checks": [{"check_id": "gate_retired", "category": "config", "status": "BLOCKED",
                    "detail": "Runtime Promotion Gate 已退役；只保留 status/latest 歷史唯讀檢視。"}],
        "fingerprint": canonical_json_sha256({"settings": cfg.as_dict(), "status": "RETIRED"}),
    }



def _print_console_summary(payload: dict[str, Any]) -> None:
    print("\n" + render_title("Runtime Promotion Gate"))
    runtime = dict(dict(payload.get("candidate_contract") or {}).get("runtime") or {})
    print(f"Decision          ：{payload.get('status')}")
    print(f"Experiment Profile：{runtime.get('experiment_profile')}")
    print(f"Runtime Mode      ：{runtime.get('dl_runtime_mode')}")
    print(f"Current Workflow  ：{payload.get('current_workflow_experiment_profile')}")
    rows = [
        [row.get("category"), row.get("check_id"), row.get("status"), row.get("detail")]
        for row in payload.get("checks") or ()
    ]
    print(render_table(["類別", "檢查", "狀態", "說明"], rows))


def show_runtime_integration_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payload = collect_runtime_integration_status(project_root=project_root)
    _print_console_summary(payload)
    return payload


def show_latest_runtime_integration_report(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    cfg = get_strategy_runtime_integration_settings()
    result_path = root / cfg.output_root / "latest" / RESULT_FILENAME
    report_path = root / cfg.output_root / "latest" / REPORT_FILENAME
    if not result_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("尚無Runtime integration Gate最新結果")
    with exclusive_file_lock(
        sibling_publication_lock_path(result_path.parent),
        conflict_message="Runtime integration最新結果正在發布；請稍後重試",
    ):
        payload = _read_json(result_path)
        report = report_path.read_text(encoding="utf-8")
    print("\n" + report)
    return payload
