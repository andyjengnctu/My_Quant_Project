import json
import os
from typing import Any

OPTIMIZER_TIMING_MODE_DEFAULT_TRIALS = 3


def build_timing_db_file_path(*, output_dir: str, dataset_profile_key: str, session_ts: str) -> str:
    safe_dataset = str(dataset_profile_key or "full").strip().lower() or "full"
    return os.path.join(output_dir, f"optimizer_timing_{safe_dataset}_{session_ts}.db")


def _row_get_float(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return float(default)
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_timing_summary(*, session, dataset_label: str, dataset_profile_key: str, selected_model_mode: str, db_file: str, raw_data_load_sec: float, optimize_wall_sec: float, total_wall_sec: float, optimizer_seed: int | None = None, timing_sampler_kind: str | None = None) -> dict[str, Any]:
    profile_summary = session.profile_recorder.build_summary_payload()
    first_row = session.profile_recorder.rows[0] if session.profile_recorder.rows else None
    completed_trials = int(getattr(session, "current_session_trial", 0))
    avg = dict(profile_summary.get("avg", {}))
    profile_objective_sum_sec = float(avg.get("objective_wall_sec", 0.0)) * float(completed_trials)
    profile_trial_total_sum_sec = float(avg.get("trial_total_wall_sec", 0.0)) * float(completed_trials)
    return {
        "mode": "timing",
        "session_ts": session.session_ts,
        "dataset_profile": str(dataset_profile_key),
        "dataset_label": str(dataset_label),
        "model_mode": str(selected_model_mode),
        "db_file": str(db_file),
        "timing_sampler_kind": str(timing_sampler_kind or ""),
        "optimizer_seed": (None if optimizer_seed is None else int(optimizer_seed)),
        "requested_trials": int(getattr(session, "n_trials", 0)),
        "completed_trials": completed_trials,
        "raw_data_load_sec": float(raw_data_load_sec),
        "optimize_wall_sec": float(optimize_wall_sec),
        "total_wall_sec": float(total_wall_sec),
        "first_trial_completed_wall_sec": profile_summary.get("first_trial_completed_wall_sec"),
        "profile_csv_path": session.profile_recorder.csv_path,
        "profile_summary_path": session.profile_recorder.summary_path,
        "profile_objective_sum_sec": float(profile_objective_sum_sec),
        "profile_trial_total_sum_sec": float(profile_trial_total_sum_sec),
        "profile_trial_plus_raw_sum_sec": float(profile_trial_total_sum_sec) + float(raw_data_load_sec),
        "optimize_minus_profile_objective_sec": float(optimize_wall_sec) - float(profile_objective_sum_sec),
        "optimize_minus_profile_trial_total_sec": float(optimize_wall_sec) - float(profile_trial_total_sum_sec),
        "total_minus_profile_trial_plus_raw_sec": float(total_wall_sec) - (float(profile_trial_total_sum_sec) + float(raw_data_load_sec)),
        "first_trial": {
            "objective_wall_sec": _row_get_float(first_row, "objective_wall_sec"),
            "prep_wall_sec": _row_get_float(first_row, "prep_wall_sec"),
            "portfolio_wall_sec": _row_get_float(first_row, "portfolio_wall_sec"),
            "prep_worker_generate_signals_sum_sec": _row_get_float(first_row, "prep_worker_generate_signals_sum_sec"),
            "prep_worker_run_backtest_sum_sec": _row_get_float(first_row, "prep_worker_run_backtest_sum_sec"),
            "prep_worker_to_dict_sum_sec": _row_get_float(first_row, "prep_worker_to_dict_sum_sec"),
            "portfolio_build_trade_index_sec": _row_get_float(first_row, "portfolio_build_trade_index_sec"),
            "portfolio_day_loop_sec": _row_get_float(first_row, "portfolio_day_loop_sec"),
        },
        "avg": avg,
    }


def write_timing_summary(*, output_dir: str, session_ts: str, payload: dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"optimizer_timing_summary_{session_ts}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def print_timing_summary(*, payload: dict[str, Any]):
    completed_trials = int(payload.get("completed_trials") or 0)
    total_wall_sec = float(payload.get("total_wall_sec", 0.0))
    raw_data_load_sec = float(payload.get("raw_data_load_sec", 0.0))
    optimize_wall_sec = float(payload.get("optimize_wall_sec", 0.0))
    avg_training_sec = (optimize_wall_sec / completed_trials) if completed_trials > 0 else 0.0
    print("📏 Optimizer 測時摘要")
    print(
        "   "
        f"總時間={total_wall_sec:.3f}s "
        f"（前處理={raw_data_load_sec:.3f}s + 訓練={optimize_wall_sec:.3f}s）"
    )
    print(f"   每 trial 平均訓練時間={avg_training_sec:.3f}s（{completed_trials} trials）")
