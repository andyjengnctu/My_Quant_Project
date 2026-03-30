from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import parse_no_arg_cli, run_cli_entrypoint
from tools.local_regression.common import ensure_reduced_dataset, load_manifest, resolve_run_dir, run_command, write_json, write_text

REQUIRED_PARAM_KEYS = [
    "atr_len",
    "atr_times_init",
    "atr_times_trail",
    "atr_buy_tol",
    "high_len",
    "tp_percent",
    "min_history_trades",
    "min_history_ev",
    "min_history_win_rate",
]


@contextmanager
def preserve_existing_file(target_path: Path, backup_suffix: str):
    backup_path = target_path.with_name(target_path.name + backup_suffix)
    if backup_path.exists():
        raise RuntimeError(f"暫存備份已存在，疑似上次中斷殘留: {backup_path}")

    existed = target_path.exists()
    if existed:
        shutil.move(target_path, backup_path)
    try:
        yield {"existed": existed, "backup_path": backup_path}
    finally:
        if target_path.exists():
            target_path.unlink()
        if existed and backup_path.exists():
            shutil.move(backup_path, target_path)


def main(argv=None) -> int:
    parsed = parse_no_arg_cli(argv, "tools/local_regression/run_ml_smoke.py", description="執行 reduced optimizer smoke test；不接受額外參數。")
    if parsed["help"]:
        return 0

    manifest = load_manifest()
    run_dir = resolve_run_dir("ml_smoke")
    dataset_info = ensure_reduced_dataset()

    db_path = PROJECT_ROOT / "models" / "portfolio_ai_10pos_overnight_reduced.db"
    params_path = PROJECT_ROOT / "models" / "best_params.json"
    db_read_error = ""
    params_read_error = ""
    param_payload: Dict[str, Any] = {}
    failures = []
    trial_count = 0

    try:
        with preserve_existing_file(db_path, ".bak_ml_smoke_local_regression"), preserve_existing_file(params_path, ".bak_ml_smoke_local_regression"):
            outcome = run_command(
                [sys.executable, "apps/ml_optimizer.py", "--dataset", "reduced"],
                timeout=int(manifest["ml_smoke_timeout_sec"]),
                env={"V16_OPTIMIZER_TRIALS": str(int(manifest["ml_smoke_trials"]))},
            )
            write_text(run_dir / "ml_smoke_console.log", f"$ {outcome['cmd']}\nreturncode={outcome['returncode']}\n\n[stdout]\n{outcome['stdout']}\n\n[stderr]\n{outcome['stderr']}")

            if outcome["returncode"] != 0:
                failures.append("optimizer_exit_nonzero")

            if not db_path.exists():
                failures.append("missing_optimizer_db")
            else:
                shutil.copy2(db_path, run_dir / db_path.name)
                try:
                    conn = sqlite3.connect(db_path)
                    try:
                        cursor = conn.execute("SELECT COUNT(*) FROM trials")
                        trial_count = int(cursor.fetchone()[0])
                    finally:
                        conn.close()
                except (sqlite3.Error, OSError, ValueError) as exc:
                    db_read_error = f"{type(exc).__name__}: {exc}"
                    failures.append("optimizer_db_unreadable")
                else:
                    if trial_count < 1:
                        failures.append("no_trials_recorded")

            if not params_path.exists():
                failures.append("missing_best_params")
            else:
                shutil.copy2(params_path, run_dir / params_path.name)
                try:
                    param_payload = json.loads(params_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError, ValueError) as exc:
                    params_read_error = f"{type(exc).__name__}: {exc}"
                    failures.append("best_params_invalid_json")
                else:
                    missing_keys = [key for key in REQUIRED_PARAM_KEYS if key not in param_payload]
                    if missing_keys:
                        failures.append(f"best_params_missing_keys:{','.join(missing_keys)}")
    except RuntimeError as exc:
        failures.append("ml_smoke_preserve_failed")
        write_text(run_dir / "ml_smoke_console.log", f"preserve_failed={type(exc).__name__}: {exc}\n")

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "dataset": manifest["dataset"],
        "dataset_info": dataset_info,
        "db_path": str(db_path),
        "db_trial_count": trial_count,
        "db_read_error": db_read_error,
        "best_params_path": str(params_path),
        "best_params_keys": sorted(param_payload.keys()) if param_payload else [],
        "best_params_read_error": params_read_error,
        "failures": failures,
    }
    write_json(run_dir / "ml_smoke_summary.json", summary)
    print(json.dumps({"status": summary["status"], "db_trial_count": trial_count}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)
