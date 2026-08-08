from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ML_OPTIMIZER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ml_optimizer"

from core.active_param_ensemble import is_active_param_ensemble_payload, build_active_param_ensemble_schedule
from core.model_paths import MODELS_DIR_ENV_VAR, RUN_BEST_PARAMS_PATH_ENV_VAR
from core.runtime_utils import PeakTracedMemoryTracker, parse_no_arg_cli, run_cli_entrypoint
from tools.optimizer.study_utils import MIN_QUALIFIED_TRIAL_VALUE, OPTIMIZER_SEED_ENV_VAR
from tools.local_regression.common import ensure_dir, ensure_reduced_dataset, load_manifest, resolve_run_dir, run_command, write_json, write_text

REQUIRED_PARAM_KEYS = [
    "atr_len",
    "atr_times_init",
    "atr_times_trail",
    "atr_buy_tol",
    "high_len",
    "use_breakout_ema_filter",
    "breakout_ema_len",
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


def _read_db_metrics_many(db_paths: list[Path]) -> Dict[str, Any]:
    if not db_paths:
        return {
            "trial_count": 0,
            "qualified_trial_count": 0,
            "best_trial_value": None,
            "db_read_error": "",
            "db_count": 0,
        }
    total_trials = 0
    total_qualified = 0
    best_values = []
    errors = []
    for db_path in db_paths:
        metrics = _read_db_metrics(db_path)
        total_trials += int(metrics.get("trial_count", 0) or 0)
        total_qualified += int(metrics.get("qualified_trial_count", 0) or 0)
        if metrics.get("best_trial_value") is not None:
            best_values.append(float(metrics["best_trial_value"]))
        if metrics.get("db_read_error"):
            errors.append(f"{db_path.name}: {metrics['db_read_error']}")
    return {
        "trial_count": int(total_trials),
        "qualified_trial_count": int(total_qualified),
        "best_trial_value": max(best_values) if best_values else None,
        "db_read_error": "; ".join(errors),
        "db_count": len(db_paths),
    }




def _load_json_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}




def _parse_seed_ensemble_progress_from_stdout(stdout: str) -> Dict[str, Any]:
    text = stdout or ""
    header_match = re.search(r"seed ensemble \|[^\n]*seeds=(\d+)", text)
    trial_matches = re.findall(r"trial\s+(\d+)\s*/\s*(\d+)", text)
    seed_count = 0
    if header_match:
        try:
            seed_count = int(header_match.group(1))
        except (TypeError, ValueError):
            seed_count = 0
    max_trial_total = 0
    max_trial_index = -1
    for current_text, total_text in trial_matches:
        try:
            max_trial_index = max(max_trial_index, int(current_text))
            max_trial_total = max(max_trial_total, int(total_text))
        except (TypeError, ValueError):
            continue
    return {
        "detected": bool(header_match or trial_matches),
        "seed_count": int(seed_count),
        "max_trial_index": int(max_trial_index),
        "trial_total_per_seed": int(max_trial_total),
    }

def _derive_trial_count_from_ensemble_summary(summary_payload: Dict[str, Any], member_count: int) -> int:
    if not isinstance(summary_payload, dict):
        return 0
    try:
        trials_per_seed = int(summary_payload.get("trials_per_seed", 0) or 0)
    except (TypeError, ValueError):
        trials_per_seed = 0
    try:
        seed_count = int(summary_payload.get("seed_count", member_count) or member_count or 0)
    except (TypeError, ValueError):
        seed_count = int(member_count or 0)
    if trials_per_seed <= 0 or seed_count <= 0:
        return 0
    return int(trials_per_seed * seed_count)

def _discover_optimizer_db_paths(models_dir: Path, legacy_db_path: Path) -> list[Path]:
    paths = []
    if legacy_db_path.exists():
        paths.append(legacy_db_path)
    ensemble_dir = models_dir / "seed_ensemble"
    if ensemble_dir.exists():
        paths.extend(sorted(path for path in ensemble_dir.glob("*.db") if path.is_file()))
    return paths


def _load_params_payload(params_path: Path) -> Dict[str, Any]:
    params_read_error = ""
    payload: Dict[str, Any] = {}
    is_ensemble = False
    member_count = 0
    if not params_path.exists():
        return {
            "payload": payload,
            "params_read_error": params_read_error,
            "missing_keys": [],
            "is_active_param_ensemble": False,
            "ensemble_member_count": 0,
        }
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
        is_ensemble = is_active_param_ensemble_payload(payload)
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
        params_read_error = f"{type(exc).__name__}: {exc}"
        payload = {}
    missing_keys = []
    if payload and not params_read_error:
        if is_ensemble:
            try:
                schedule = build_active_param_ensemble_schedule(payload)
                members = []
                for record in schedule:
                    members.extend(list(record.get("members") or []))
                member_count = len(members)
                for idx, member in enumerate(members, start=1):
                    params = member.get("params") if isinstance(member, dict) else {}
                    for key in REQUIRED_PARAM_KEYS:
                        if key not in params:
                            missing_keys.append(f"member{idx}.{key}")
            except (ValueError, TypeError, KeyError) as exc:
                params_read_error = f"{type(exc).__name__}: {exc}"
        else:
            missing_keys = [key for key in REQUIRED_PARAM_KEYS if key not in payload]
    return {
        "payload": payload,
        "params_read_error": params_read_error,
        "missing_keys": missing_keys,
        "is_active_param_ensemble": bool(is_ensemble),
        "ensemble_member_count": int(member_count),
    }


VOLATILE_REPRO_DIGEST_KEYS = {"created_at", "promoted_at"}


def _stable_payload_for_repro_digest(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_payload_for_repro_digest(item)
            for key, item in value.items()
            if key not in VOLATILE_REPRO_DIGEST_KEYS
        }
    if isinstance(value, list):
        return [_stable_payload_for_repro_digest(item) for item in value]
    return value


def _canonical_payload_digest(payload: Dict[str, Any]) -> str:
    stable_payload = _stable_payload_for_repro_digest(payload)
    canonical_json = json.dumps(stable_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
    with PeakTracedMemoryTracker() as tracker:
        started = time.perf_counter()
        label_dir = ensure_dir(parent_run_dir / label)
        models_dir = ensure_dir(label_dir / "models")
        params_path = models_dir / "run_best_params.json"
        candidate_params_path = models_dir / "candidate_best_params.json"
        candidate_summary_path = models_dir / "candidate_best_summary.json"
        candidate_retention_params_path = models_dir / "candidate_retention_best_params.json"
        candidate_retention_summary_path = models_dir / "candidate_retention_best_summary.json"
        db_path = models_dir / "portfolio_ai_10pos_overnight_reduced.db"
        failures = []

        before_profile_summaries = _collect_optimizer_profile_summaries()

        outcome = run_command(
            [sys.executable, "apps/research.py", "optimizer", "--dataset", "reduced"],
            timeout=int(manifest["ml_smoke_timeout_sec"]),
            env={
                "V16_OPTIMIZER_TRIALS": str(int(manifest["ml_smoke_trials"])),
                OPTIMIZER_SEED_ENV_VAR: str(ML_SMOKE_REPRO_SEED),
                MODELS_DIR_ENV_VAR: str(models_dir),
                RUN_BEST_PARAMS_PATH_ENV_VAR: str(params_path),
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

        db_paths = _discover_optimizer_db_paths(models_dir, db_path)
        missing_optimizer_db = False
        if not db_paths:
            missing_optimizer_db = True
            db_metrics = {
                "trial_count": 0,
                "qualified_trial_count": 0,
                "best_trial_value": None,
                "db_read_error": "",
                "db_count": 0,
            }
        else:
            db_metrics = _read_db_metrics_many(db_paths)
            if db_metrics["db_read_error"]:
                failures.append("optimizer_db_unreadable")
            elif db_metrics["trial_count"] < 1:
                failures.append("no_trials_recorded")

        candidate_params_info = _load_params_payload(candidate_params_path)
        embedded_candidate_summary = {}
        if isinstance(candidate_params_info.get("payload"), dict):
            embedded = candidate_params_info["payload"].get("summary")
            if isinstance(embedded, dict):
                embedded_candidate_summary = dict(embedded)
        candidate_summary_payload = embedded_candidate_summary or _load_json_payload(candidate_summary_path)
        candidate_summary_sidecar_exists = candidate_summary_path.exists()
        candidate_summary_exists = bool(embedded_candidate_summary) or candidate_summary_sidecar_exists
        candidate_retention_params_exists = candidate_retention_params_path.exists()
        candidate_retention_summary_exists = candidate_retention_summary_path.exists()
        candidate_best_available = candidate_params_path.exists() or candidate_summary_exists
        params_info = _load_params_payload(params_path)
        run_best_params_required = bool(candidate_best_available)
        if candidate_params_path.exists():
            if candidate_params_info["params_read_error"]:
                failures.append("candidate_best_params_invalid_json")
            elif candidate_params_info["missing_keys"]:
                failures.append(f"candidate_best_params_missing_keys:{','.join(candidate_params_info['missing_keys'])}")
        if candidate_params_path.exists() and not candidate_summary_exists:
            failures.append("missing_candidate_best_embedded_summary")
        if candidate_summary_exists and not candidate_params_path.exists():
            failures.append("missing_candidate_best_params")

        seed_ensemble_progress = _parse_seed_ensemble_progress_from_stdout(outcome.get("stdout", ""))
        if missing_optimizer_db:
            derived_trial_count = _derive_trial_count_from_ensemble_summary(
                candidate_summary_payload,
                int(candidate_params_info.get("ensemble_member_count", 0) or 0),
            )
            if bool(candidate_params_info.get("is_active_param_ensemble", False)) and derived_trial_count > 0:
                db_metrics["trial_count"] = int(derived_trial_count)
                db_metrics["qualified_trial_count"] = int(candidate_params_info.get("ensemble_member_count", 0) or 0)
                db_metrics["best_trial_value"] = None
                db_metrics["trial_count_source"] = "active_param_ensemble_summary"
            elif bool(seed_ensemble_progress.get("detected", False)) and outcome["returncode"] == 0:
                seed_count = int(seed_ensemble_progress.get("seed_count", 0) or 0)
                trial_total_per_seed = int(seed_ensemble_progress.get("trial_total_per_seed", 0) or 0)
                if seed_count <= 0:
                    seed_count = 1
                if trial_total_per_seed <= 0:
                    trial_total_per_seed = int(manifest.get("ml_smoke_trials", 0) or 0)
                db_metrics["trial_count"] = int(max(0, seed_count) * max(0, trial_total_per_seed))
                db_metrics["qualified_trial_count"] = 0
                db_metrics["best_trial_value"] = None
                db_metrics["trial_count_source"] = "seed_ensemble_in_memory_progress"
                db_metrics["db_count"] = 0
            elif (
                outcome["returncode"] == 0
                and not profile_metrics.get("optimizer_profile_read_error")
                and int(profile_metrics.get("optimizer_profile_trial_count", 0) or 0) > 0
            ):
                db_metrics["trial_count"] = int(profile_metrics["optimizer_profile_trial_count"])
                db_metrics["qualified_trial_count"] = 1 if candidate_best_available else 0
                db_metrics["best_trial_value"] = None
                db_metrics["trial_count_source"] = "optimizer_profile_summary"
                db_metrics["db_count"] = 0
            else:
                failures.append("missing_optimizer_db")

        if not params_path.exists():
            if run_best_params_required:
                failures.append("missing_run_best_params_for_exported_candidate")
        elif params_info["params_read_error"]:
            failures.append("run_best_params_invalid_json")
        elif params_info["missing_keys"]:
            failures.append(f"run_best_params_missing_keys:{','.join(params_info['missing_keys'])}")

        payload_digest = _canonical_payload_digest(params_info["payload"]) if params_info["payload"] else ""
        candidate_payload_digest = _canonical_payload_digest(candidate_params_info["payload"]) if candidate_params_info["payload"] else ""
        result = {
            "label": label,
            "status": "PASS" if not failures else "FAIL",
            "db_path": str(db_path),
            "db_paths": [str(path) for path in db_paths],
            "db_count": int(db_metrics.get("db_count", len(db_paths))),
            "run_best_params_path": str(params_path),
            "candidate_best_params_path": str(candidate_params_path),
            "candidate_best_summary_path": str(candidate_summary_path),
            "candidate_retention_best_params_path": str(candidate_retention_params_path),
            "candidate_retention_best_summary_path": str(candidate_retention_summary_path),
            "db_trial_count": db_metrics["trial_count"],
            "qualified_trial_count": db_metrics["qualified_trial_count"],
            "best_trial_value": db_metrics["best_trial_value"],
            "run_best_params_required": run_best_params_required,
            "candidate_best_available": bool(candidate_best_available),
            "candidate_best_params_exists": candidate_params_path.exists(),
            "candidate_best_summary_exists": candidate_summary_exists,
            "candidate_best_summary_embedded": bool(embedded_candidate_summary),
            "candidate_best_summary_sidecar_exists": bool(candidate_summary_sidecar_exists),
            "candidate_best_params_keys": sorted(candidate_params_info["payload"].keys()) if candidate_params_info["payload"] else [],
            "candidate_best_params_digest": candidate_payload_digest,
            "candidate_best_is_active_param_ensemble": bool(candidate_params_info.get("is_active_param_ensemble", False)),
            "candidate_best_ensemble_member_count": int(candidate_params_info.get("ensemble_member_count", 0) or 0),
            "candidate_best_params_read_error": candidate_params_info["params_read_error"],
            "candidate_retention_best_params_exists": candidate_retention_params_exists,
            "candidate_retention_best_summary_exists": candidate_retention_summary_exists,
            "run_best_params_keys": sorted(params_info["payload"].keys()) if params_info["payload"] else [],
            "run_best_is_active_param_ensemble": bool(params_info.get("is_active_param_ensemble", False)),
            "run_best_ensemble_member_count": int(params_info.get("ensemble_member_count", 0) or 0),
            "run_best_params_payload": params_info["payload"],
            "run_best_params_digest": payload_digest,
            "db_read_error": db_metrics["db_read_error"],
            "db_trial_count_source": str(db_metrics.get("trial_count_source", "optimizer_db" if db_paths else "")),
            "random_seed_ensemble_progress_detected": bool(seed_ensemble_progress.get("detected", False)),
            "random_seed_ensemble_progress_seed_count": int(seed_ensemble_progress.get("seed_count", 0) or 0),
            "random_seed_ensemble_progress_trial_total_per_seed": int(seed_ensemble_progress.get("trial_total_per_seed", 0) or 0),
            "run_best_params_read_error": params_info["params_read_error"],
            "optimizer_profile_summary_path": profile_metrics["optimizer_profile_summary_path"],
            "optimizer_profile_trial_count": profile_metrics["optimizer_profile_trial_count"],
            "optimizer_profile_avg_objective_wall_sec": profile_metrics["optimizer_profile_avg_objective_wall_sec"],
            "optimizer_profile_read_error": profile_metrics["optimizer_profile_read_error"],
            "failures": failures,
            "duration_sec": round(time.perf_counter() - started, 3),
            "peak_traced_memory_mb": tracker.snapshot_peak_mb(),
        }
        write_json(label_dir / "ml_smoke_summary.json", result)
        return result

def _extract_run_best_params_digest(run_summary: Dict[str, Any]) -> str:
    digest = run_summary.get("run_best_params_digest")
    if isinstance(digest, str) and digest != "":
        return digest
    legacy_digest = run_summary.get("best_params_digest")
    if isinstance(legacy_digest, str):
        return legacy_digest
    return ""


def _build_repro_summary(first_run: Dict[str, Any], second_run: Dict[str, Any]) -> Dict[str, Any]:
    first_digest = _extract_run_best_params_digest(first_run)
    second_digest = _extract_run_best_params_digest(second_run)
    random_seed_ensemble_mode = (
        bool(first_run.get("candidate_best_is_active_param_ensemble", False))
        or bool(second_run.get("candidate_best_is_active_param_ensemble", False))
        or bool(first_run.get("random_seed_ensemble_progress_detected", False))
        or bool(second_run.get("random_seed_ensemble_progress_detected", False))
        or int(first_run.get("db_count", 0) or 0) > 1
        or int(second_run.get("db_count", 0) or 0) > 1
    )
    comparisons = {
        "trial_count_match": first_run["db_trial_count"] == second_run["db_trial_count"],
        "candidate_best_available_match": first_run.get("candidate_best_available", False) == second_run.get("candidate_best_available", False),
        "optimizer_profile_trial_count_match": first_run["optimizer_profile_trial_count"] == second_run["optimizer_profile_trial_count"],
    }
    if random_seed_ensemble_mode:
        comparisons.update({
            "random_seed_ensemble_mode": True,
            "both_runs_pass": first_run["status"] == second_run["status"] == "PASS",
            "both_runs_record_trials": first_run["db_trial_count"] >= 1 and second_run["db_trial_count"] >= 1,
        })
        all_match = (
            comparisons["both_runs_pass"]
            and comparisons["both_runs_record_trials"]
            and comparisons["trial_count_match"]
            and comparisons["candidate_best_available_match"]
        )
    else:
        comparisons.update({
            "qualified_trial_count_match": first_run["qualified_trial_count"] == second_run["qualified_trial_count"],
            "best_trial_value_match": first_run["best_trial_value"] == second_run["best_trial_value"],
            "candidate_best_params_digest_match": first_run.get("candidate_best_params_digest", "") == second_run.get("candidate_best_params_digest", ""),
            "run_best_params_digest_match": first_digest == second_digest,
        })
        all_match = all(comparisons.values()) and first_run["status"] == second_run["status"] == "PASS"
    return {
        "enabled": True,
        "run_count": ML_SMOKE_REPRO_RUN_COUNT,
        "seed": ML_SMOKE_REPRO_SEED,
        "mode": "random_seed_ensemble_contract" if random_seed_ensemble_mode else "deterministic_single_seed_repro",
        "all_match": all_match,
        "comparisons": comparisons,
        "runs": [
            {
                "label": first_run["label"],
                "status": first_run["status"],
                "db_count": first_run.get("db_count", 0),
                "db_trial_count": first_run["db_trial_count"],
                "qualified_trial_count": first_run["qualified_trial_count"],
                "best_trial_value": first_run["best_trial_value"],
                "run_best_params_digest": first_digest,
                "candidate_best_available": first_run.get("candidate_best_available", False),
                "candidate_best_is_active_param_ensemble": first_run.get("candidate_best_is_active_param_ensemble", False),
                "candidate_best_ensemble_member_count": first_run.get("candidate_best_ensemble_member_count", 0),
                "candidate_best_params_digest": first_run.get("candidate_best_params_digest", ""),
                "optimizer_profile_trial_count": first_run["optimizer_profile_trial_count"],
                "optimizer_profile_avg_objective_wall_sec": first_run["optimizer_profile_avg_objective_wall_sec"],
                "failures": first_run["failures"],
                "duration_sec": first_run.get("duration_sec", 0.0),
            },
            {
                "label": second_run["label"],
                "status": second_run["status"],
                "db_count": second_run.get("db_count", 0),
                "db_trial_count": second_run["db_trial_count"],
                "qualified_trial_count": second_run["qualified_trial_count"],
                "best_trial_value": second_run["best_trial_value"],
                "run_best_params_digest": second_digest,
                "candidate_best_available": second_run.get("candidate_best_available", False),
                "candidate_best_is_active_param_ensemble": second_run.get("candidate_best_is_active_param_ensemble", False),
                "candidate_best_ensemble_member_count": second_run.get("candidate_best_ensemble_member_count", 0),
                "candidate_best_params_digest": second_run.get("candidate_best_params_digest", ""),
                "optimizer_profile_trial_count": second_run["optimizer_profile_trial_count"],
                "optimizer_profile_avg_objective_wall_sec": second_run["optimizer_profile_avg_objective_wall_sec"],
                "failures": second_run["failures"],
                "duration_sec": second_run.get("duration_sec", 0.0),
            },
        ],
    }


def main(argv=None) -> int:
    parsed = parse_no_arg_cli(argv, "tools/local_regression/run_ml_smoke.py", description="執行 reduced optimizer smoke test；不接受額外參數。")
    if parsed["help"]:
        return 0

    with PeakTracedMemoryTracker() as tracker:
        started = time.perf_counter()
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
            "db_paths": first_run.get("db_paths", []),
            "db_count": first_run.get("db_count", 0),
            "db_trial_count": first_run["db_trial_count"],
            "db_read_error": first_run["db_read_error"],
            "db_trial_count_source": first_run.get("db_trial_count_source", ""),
            "random_seed_ensemble_progress_detected": first_run.get("random_seed_ensemble_progress_detected", False),
            "random_seed_ensemble_progress_seed_count": first_run.get("random_seed_ensemble_progress_seed_count", 0),
            "random_seed_ensemble_progress_trial_total_per_seed": first_run.get("random_seed_ensemble_progress_trial_total_per_seed", 0),
            "run_best_params_path": first_run["run_best_params_path"],
            "run_best_params_required": first_run["run_best_params_required"],
            "candidate_best_available": first_run.get("candidate_best_available", False),
            "candidate_best_params_path": first_run.get("candidate_best_params_path", ""),
            "candidate_best_summary_path": first_run.get("candidate_best_summary_path", ""),
            "candidate_best_params_keys": first_run.get("candidate_best_params_keys", []),
            "candidate_best_params_digest": first_run.get("candidate_best_params_digest", ""),
            "candidate_best_is_active_param_ensemble": first_run.get("candidate_best_is_active_param_ensemble", False),
            "candidate_best_ensemble_member_count": first_run.get("candidate_best_ensemble_member_count", 0),
            "candidate_best_params_read_error": first_run.get("candidate_best_params_read_error", ""),
            "candidate_best_summary_embedded": first_run.get("candidate_best_summary_embedded", False),
            "candidate_best_summary_sidecar_exists": first_run.get("candidate_best_summary_sidecar_exists", False),
            "candidate_retention_best_params_path": first_run.get("candidate_retention_best_params_path", ""),
            "candidate_retention_best_summary_path": first_run.get("candidate_retention_best_summary_path", ""),
            "candidate_retention_best_params_exists": first_run.get("candidate_retention_best_params_exists", False),
            "candidate_retention_best_summary_exists": first_run.get("candidate_retention_best_summary_exists", False),
            "qualified_trial_count": first_run["qualified_trial_count"],
            "best_trial_value": first_run["best_trial_value"],
            "run_best_params_keys": first_run["run_best_params_keys"],
            "run_best_is_active_param_ensemble": first_run.get("run_best_is_active_param_ensemble", False),
            "run_best_ensemble_member_count": first_run.get("run_best_ensemble_member_count", 0),
            "run_best_params_read_error": first_run["run_best_params_read_error"],
            "optimizer_profile_summary_path": first_run["optimizer_profile_summary_path"],
            "optimizer_profile_trial_count": first_run["optimizer_profile_trial_count"],
            "optimizer_profile_avg_objective_wall_sec": first_run["optimizer_profile_avg_objective_wall_sec"],
            "optimizer_profile_read_error": first_run["optimizer_profile_read_error"],
            "runtime_error": "",
            "optimizer_repro": repro_summary,
            "failures": failures,
            "duration_sec": round(time.perf_counter() - started, 3),
            "peak_traced_memory_mb": tracker.snapshot_peak_mb(),
        }
        write_json(run_dir / "ml_smoke_summary.json", summary)
        print(json.dumps({"status": summary["status"], "db_trial_count": summary["db_trial_count"], "optimizer_repro_all_match": repro_summary["all_match"]}, ensure_ascii=False))
        return 0 if not failures else 1

if __name__ == "__main__":
    run_cli_entrypoint(main)
