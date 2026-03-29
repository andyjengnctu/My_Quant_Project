from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def main() -> int:
    manifest = load_manifest()
    run_dir = resolve_run_dir("ml_smoke")
    dataset_info = ensure_reduced_dataset()

    db_path = PROJECT_ROOT / "models" / "portfolio_ai_10pos_overnight_reduced.db"
    if db_path.exists():
        db_path.unlink()

    outcome = run_command(
        [sys.executable, "apps/ml_optimizer.py", "--dataset", "reduced"],
        timeout=int(manifest["ml_smoke_timeout_sec"]),
        env={"V16_OPTIMIZER_TRIALS": str(int(manifest["ml_smoke_trials"]))},
    )
    write_text(run_dir / "ml_smoke_console.log", f"$ {outcome['cmd']}\nreturncode={outcome['returncode']}\n\n[stdout]\n{outcome['stdout']}\n\n[stderr]\n{outcome['stderr']}")

    failures = []
    if outcome["returncode"] != 0:
        failures.append("optimizer_exit_nonzero")

    trial_count = 0
    if not db_path.exists():
        failures.append("missing_optimizer_db")
    else:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM trials")
            trial_count = int(cursor.fetchone()[0])
        finally:
            conn.close()
        if trial_count < 1:
            failures.append("no_trials_recorded")

    params_path = PROJECT_ROOT / "models" / "best_params.json"
    param_payload: Dict[str, Any] = {}
    if not params_path.exists():
        failures.append("missing_best_params")
    else:
        param_payload = json.loads(params_path.read_text(encoding="utf-8"))
        missing_keys = [key for key in REQUIRED_PARAM_KEYS if key not in param_payload]
        if missing_keys:
            failures.append(f"best_params_missing_keys:{','.join(missing_keys)}")

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "dataset": manifest["dataset"],
        "dataset_info": dataset_info,
        "db_path": str(db_path),
        "db_trial_count": trial_count,
        "best_params_path": str(params_path),
        "best_params_keys": sorted(param_payload.keys()) if param_payload else [],
        "failures": failures,
    }
    write_json(run_dir / "ml_smoke_summary.json", summary)
    print(json.dumps({"status": summary["status"], "db_trial_count": trial_count}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
