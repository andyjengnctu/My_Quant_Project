from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ML_OPTIMIZER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ml_optimizer"

from core.model_paths import BEST_PARAMS_PATH_ENV_VAR, MODELS_DIR_ENV_VAR
from core.runtime_utils import parse_no_arg_cli, run_cli_entrypoint
from tools.optimizer.study_utils import MIN_QUALIFIED_TRIAL_VALUE, OPTIMIZER_SEED_ENV_VAR
from tools.local_regression.common import ensure_dir, ensure_reduced_dataset, load_manifest, resolve_run_dir, run_command, write_json, write_text

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
ML_SMOKE_REPRO_RUN_COUNT = 2
ML_SMOKE_REPRO_SEED = 20260401


def _read_db_metrics(db_path: Path) -> Dict[str, Any]:
    db_read_error = ""
    trial_count = 0
    qualified_trial_count = 0
    best_trial_value = None
    try:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM trials")
            trial_count = int(cursor.fetchone()[0])
            cursor = conn.execute(
                """
                SELECT COUNT(*), MAX(tv.value)
                FROM trials t
                JOIN trial_values tv ON tv.trial_id = t.trial_id AND tv.objective = 0
                WHERE t.state = 'COMPLETE' AND tv.value > ?
                """,
                (MIN_QUALIFIED_TRIAL_VALUE,),
            )
            qualified_trial_count, best_trial_value = cursor.fetchone()
            qualified_trial_count = int(qualified_trial_count or 0)
        finally:
            conn.close()
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        db_read_error = f"{type(exc).__name__}: {exc}"
    return {
        "trial_count": trial_count,
        "qualified_trial_count": qualified_trial_count,
        "best_trial_value": best_trial_value,
        "db_read_error": db_read_error,
    }


def _load_params_payload(params_path: Path) -> Dict[str, Any]:
    params_read_error = ""
    payload: Dict[str, Any] = {}
    if not params_path.exists():
        return {
            "payload": payload,
            "params_read_error": params_read_error,
            "missing_keys": [],
        }
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        params_read_error = f"{type(exc).__name__}: {exc}"
        payload = {}
    missing_keys = [key for key in REQUIRED_PARAM_KEYS if key not in payload]
    return {
        "payload": payload,
        "params_read_error": params_read_error,
        "missing_keys": missing_keys,
    }


def _canonical_payload_digest(payload: Dict[str, Any]) -> str:
    canonical_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _collect_optimizer_profile_summaries() -> Dict[str, Path]:
    if not ML_OPTIMIZER_OUTPUT_DIR.exists():
        return {}
    return {
        path.name: path
        for path in ML_OPTIMIZER_OUTPUT_DIR.glob("optimizer_profile_summary_*.json")
        if path.is_file()
    }


def _read_latest_profile_metrics(new_paths: Dict[str, Path]) -> Dict[str, Any]:
    if not new_paths:
        return {
            "optimizer_profile_summary_path": "",
            "optimizer_profile_trial_count": 0,
            "optimizer_profile_avg_objective_wall_sec": None,
            "optimizer_profile_read_error": "",
        }
    latest_path = max(new_paths.values(), key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        avg_payload = payload.get("avg", {}) if isinstance(payload, dict) else {}
        return {
            "optimizer_profile_summary_path": str(latest_path),
            "optimizer_profile_trial_count": int(payload.get("trial_count", 0) or 0),
            "optimizer_profile_avg_objective_wall_sec": (
                None if avg_payload.get("objective_wall_sec") in (None, "") else float(avg_payload.get("objective_wall_sec"))
            ),
            "optimizer_profile_read_error": "",
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "optimizer_profile_summary_path": str(latest_path),
            "optimizer_profile_trial_count": 0,
            "optimizer_profile_avg_objective_wall_sec": None,
            "optimizer_profile_read_error": f"{type(exc).__name__}: {exc}",
        }


def _run_single_optimizer_smoke(*, label: str, parent_run_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    label_dir = ensure_dir(parent_run_dir / label)
    models_dir = ensure_dir(label_dir / "models")
    params_path = models_dir / "best_params.json"
    db_path = models_dir / "portfolio_ai_10pos_overnight_reduced.db"
    failures = []

    before_profile_summaries = _collect_optimizer_profile_summaries()

    outcome = run_command(
        [sys.executable, "apps/ml_optimizer.py", "--dataset", "reduced"],
        timeout=int(manifest["ml_smoke_timeout_sec"]),
        env={
            "V16_OPTIMIZER_TRIALS": str(int(manifest["ml_smoke_trials"])),
            OPTIMIZER_SEED_ENV_VAR: str(ML_SMOKE_REPRO_SEED),
            MODELS_DIR_ENV_VAR: str(models_dir),
            BEST_PARAMS_PATH_ENV_VAR: str(params_path),
        },
    )
    write_text(
        label_dir / "ml_smoke_console.log",
        "\n".join(
            [
                f"$ {outcome['cmd']}",
                f"returncode={outcome['returncode']}",
                f"timed_out={outcome.get('timed_out', False)}",
                f"error_type={outcome.get('error_type', '')}",
                f"error_message={outcome.get('error_message', '')}",
                "",
                "[stdout]",
                outcome["stdout"],
                "",
                "[stderr]",
                outcome["stderr"],
            ]
        ),
    )

    after_profile_summaries = _collect_optimizer_profile_summaries()
    new_profile_summaries = {
        name: path for name, path in after_profile_summaries.items() if name not in before_profile_summaries
    }
    profile_metrics = _read_latest_profile_metrics(new_profile_summaries)

    if outcome.get("timed_out"):
        failures.append("optimizer_timed_out")
    elif outcome["returncode"] != 0:
        failures.append("optimizer_exit_nonzero")

    if not db_path.exists():
        failures.append("missing_optimizer_db")
        db_metrics = {
            "trial_count": 0,
            "qualified_trial_count": 0,
            "best_trial_value": None,
            "db_read_error": "",
        }
    else:
        db_metrics = _read_db_metrics(db_path)
        if db_metrics["db_read_error"]:
            failures.append("optimizer_db_unreadable")
        elif db_metrics["trial_count"] < 1:
            failures.append("no_trials_recorded")

    params_info = _load_params_payload(params_path)
    best_params_required = db_metrics["qualified_trial_count"] > 0
    if not params_path.exists():
        if best_params_required:
            failures.append("missing_best_params_for_qualified_trial")
    elif params_info["params_read_error"]:
        failures.append("best_params_invalid_json")
    elif params_info["missing_keys"]:
        failures.append(f"best_params_missing_keys:{','.join(params_info['missing_keys'])}")

    payload_digest = _canonical_payload_digest(params_info["payload"]) if params_info["payload"] else ""
    result = {
        "label": label,
        "status": "PASS" if not failures else "FAIL",
        "db_path": str(db_path),
        "best_params_path": str(params_path),
        "db_trial_count": db_metrics["trial_count"],
        "qualified_trial_count": db_metrics["qualified_trial_count"],
        "best_trial_value": db_metrics["best_trial_value"],
        "best_params_required": best_params_required,
        "best_params_keys": sorted(params_info["payload"].keys()) if params_info["payload"] else [],
        "best_params_payload": params_info["payload"],
        "best_params_digest": payload_digest,
        "db_read_error": db_metrics["db_read_error"],
        "best_params_read_error": params_info["params_read_error"],
        "optimizer_profile_summary_path": profile_metrics["optimizer_profile_summary_path"],
        "optimizer_profile_trial_count": profile_metrics["optimizer_profile_trial_count"],
        "optimizer_profile_avg_objective_wall_sec": profile_metrics["optimizer_profile_avg_objective_wall_sec"],
        "optimizer_profile_read_error": profile_metrics["optimizer_profile_read_error"],
        "failures": failures,
    }
    write_json(label_dir / "ml_smoke_summary.json", result)
    return result


def _build_repro_summary(first_run: Dict[str, Any], second_run: Dict[str, Any]) -> Dict[str, Any]:
    comparisons = {
        "trial_count_match": first_run["db_trial_count"] == second_run["db_trial_count"],
        "qualified_trial_count_match": first_run["qualified_trial_count"] == second_run["qualified_trial_count"],
        "best_trial_value_match": first_run["best_trial_value"] == second_run["best_trial_value"],
        "best_params_digest_match": first_run["best_params_digest"] == second_run["best_params_digest"],
        "optimizer_profile_trial_count_match": first_run["optimizer_profile_trial_count"] == second_run["optimizer_profile_trial_count"],
    }
    all_match = all(comparisons.values()) and first_run["status"] == second_run["status"] == "PASS"
    return {
        "enabled": True,
        "run_count": ML_SMOKE_REPRO_RUN_COUNT,
        "seed": ML_SMOKE_REPRO_SEED,
        "all_match": all_match,
        "comparisons": comparisons,
        "runs": [
            {
                "label": first_run["label"],
                "status": first_run["status"],
                "db_trial_count": first_run["db_trial_count"],
                "qualified_trial_count": first_run["qualified_trial_count"],
                "best_trial_value": first_run["best_trial_value"],
                "best_params_digest": first_run["best_params_digest"],
                "optimizer_profile_trial_count": first_run["optimizer_profile_trial_count"],
                "optimizer_profile_avg_objective_wall_sec": first_run["optimizer_profile_avg_objective_wall_sec"],
                "failures": first_run["failures"],
            },
            {
                "label": second_run["label"],
                "status": second_run["status"],
                "db_trial_count": second_run["db_trial_count"],
                "qualified_trial_count": second_run["qualified_trial_count"],
                "best_trial_value": second_run["best_trial_value"],
                "best_params_digest": second_run["best_params_digest"],
                "optimizer_profile_trial_count": second_run["optimizer_profile_trial_count"],
                "optimizer_profile_avg_objective_wall_sec": second_run["optimizer_profile_avg_objective_wall_sec"],
                "failures": second_run["failures"],
            },
        ],
    }


def main(argv=None) -> int:
    parsed = parse_no_arg_cli(argv, "tools/local_regression/run_ml_smoke.py", description="執行 reduced optimizer smoke test；不接受額外參數。")
    if parsed["help"]:
        return 0

    manifest = load_manifest()
    run_dir = resolve_run_dir("ml_smoke")
    dataset_info = ensure_reduced_dataset()
    smoke_dir = ensure_dir(run_dir / "optimizer_repro")
    failures = []

    first_run = _run_single_optimizer_smoke(label="run_1", parent_run_dir=smoke_dir, manifest=manifest)
    second_run = _run_single_optimizer_smoke(label="run_2", parent_run_dir=smoke_dir, manifest=manifest)
    for single_run in (first_run, second_run):
        failures.extend(f"{single_run['label']}::{failure}" for failure in single_run["failures"])

    repro_summary = _build_repro_summary(first_run, second_run)
    if not repro_summary["all_match"]:
        failures.append("optimizer_repro_mismatch")

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "dataset": manifest["dataset"],
        "dataset_info": dataset_info,
        "db_path": first_run["db_path"],
        "db_trial_count": first_run["db_trial_count"],
        "db_read_error": first_run["db_read_error"],
        "best_params_path": first_run["best_params_path"],
        "best_params_required": first_run["best_params_required"],
        "qualified_trial_count": first_run["qualified_trial_count"],
        "best_trial_value": first_run["best_trial_value"],
        "best_params_keys": first_run["best_params_keys"],
        "best_params_read_error": first_run["best_params_read_error"],
        "optimizer_profile_summary_path": first_run["optimizer_profile_summary_path"],
        "optimizer_profile_trial_count": first_run["optimizer_profile_trial_count"],
        "optimizer_profile_avg_objective_wall_sec": first_run["optimizer_profile_avg_objective_wall_sec"],
        "optimizer_profile_read_error": first_run["optimizer_profile_read_error"],
        "runtime_error": "",
        "optimizer_repro": repro_summary,
        "failures": failures,
    }
    write_json(run_dir / "ml_smoke_summary.json", summary)
    print(json.dumps({"status": summary["status"], "db_trial_count": summary["db_trial_count"], "optimizer_repro_all_match": repro_summary["all_match"]}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)
