import inspect
import json
import os
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dataset_profiles import (
    DEFAULT_DATASET_PROFILE,
    get_dataset_dir,
    get_dataset_profile_label,
    resolve_dataset_profile_from_cli_env,
    build_missing_dataset_dir_message,
    build_empty_dataset_dir_message,
)
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.model_paths import resolve_models_dir, resolve_run_best_params_path
from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, get_process_pool_executor_kwargs, get_taipei_now, has_help_flag, resolve_cli_program_name, validate_cli_args, is_interactive_console, safe_prompt_choice, stdout_supports_inline_progress, write_inline_progress
from core.output_paths import build_output_dir
from core.walk_forward_policy import (
    build_optimizer_effective_policy_fingerprint,
    build_optimizer_runtime_policy,
    load_walk_forward_policy,
)
from core.active_param_ensemble import build_static_active_param_ensemble_payload, is_active_param_ensemble_payload
from core.seed_ensemble_policy import build_seed_ensemble_policy_snapshot, generate_random_seed_ensemble
from config.training_policy import (
    DEFAULT_OPTIMIZER_MODEL_MODE,
    OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_FIXED_TP_PERCENT,
    OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE,
    OPTIMIZER_INNER_VALIDATE_MIN_SCORE,
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    is_optimizer_local_min_review_enabled,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
    is_optimizer_nonrolling_train_result_table_enabled,
)

from config.training_performance_policy import resolve_optimizer_random_seed_ensemble_parallel_backend_default, resolve_optimizer_random_seed_ensemble_parallel_workers_default

warnings.simplefilter("default")
warnings.filterwarnings("once", category=FutureWarning, module=r"optuna(\..*)?$")
warnings.filterwarnings("once", category=RuntimeWarning)



def _print_profile_summary_compatible(profile_recorder, *, emit_console: bool):
    print_summary = getattr(profile_recorder, "print_summary", None)
    if print_summary is None:
        return
    try:
        signature = inspect.signature(print_summary)
    except (TypeError, ValueError):
        print_summary()
        return

    params = signature.parameters.values()
    supports_emit_console = (
        "emit_console" in signature.parameters
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    )
    if supports_emit_console:
        print_summary(emit_console=emit_console)
        return
    print_summary()

def configure_optuna_logging():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)


OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "ml_optimizer")
MODELS_DIR = resolve_models_dir(PROJECT_ROOT)
RUN_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "run_best_params.json")
CANDIDATE_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "candidate_best_params.json")
CANDIDATE_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "candidate_best_summary.json")
CANDIDATE_RETENTION_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "candidate_retention_best_params.json")
CANDIDATE_RETENTION_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "candidate_retention_best_summary.json")
CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "candidate_val_score_best_params.json")
CANDIDATE_VAL_SCORE_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "candidate_val_score_best_summary.json")
RUN_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "run_best_summary.json")
DEFAULT_WALK_FORWARD_POLICY = load_walk_forward_policy(PROJECT_ROOT)
TRAIN_MAX_POSITIONS = 10
TRAIN_ENABLE_ROTATION = False
DEFAULT_OPTIMIZER_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1))) if os.name == "nt" else min(6, max(1, (os.cpu_count() or 1) // 2))
ENABLE_OPTIMIZER_PROFILING = True
ENABLE_PROFILE_CONSOLE_PRINT = False
PROFILE_PRINT_EVERY_N_TRIALS = 1
START_BANNER_POLICY_LABEL = "Train/Test policy:"

COLORS = {
    "cyan": C_CYAN,
    "gray": C_GRAY,
    "green": C_GREEN,
    "red": C_RED,
    "reset": C_RESET,
    "yellow": C_YELLOW,
}


def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CANDIDATE_BEST_PARAMS_PATH), exist_ok=True)


def _write_json_file(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def _load_json_file_or_none(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_study_effective_policy_compatible(*, study, walk_forward_policy: dict):
    if not hasattr(study, "user_attrs") or not hasattr(study, "set_user_attr"):
        return
    contract = build_optimizer_effective_policy_fingerprint(walk_forward_policy)
    fingerprint_key = "optimizer_effective_policy_fingerprint_sha256"
    snapshot_key = "optimizer_effective_policy_snapshot"
    existing_fingerprint = getattr(study, "user_attrs", {}).get(fingerprint_key)
    existing_trials = list(getattr(study, "trials", []) or [])
    if not existing_fingerprint:
        study.set_user_attr(fingerprint_key, contract["fingerprint_sha256"])
        study.set_user_attr(snapshot_key, contract["snapshot"])
        return
    if str(existing_fingerprint) == str(contract["fingerprint_sha256"]):
        if getattr(study, "user_attrs", {}).get(snapshot_key) is None:
            study.set_user_attr(snapshot_key, contract["snapshot"])
        return
    if len(existing_trials) == 0:
        study.set_user_attr(fingerprint_key, contract["fingerprint_sha256"])
        study.set_user_attr(snapshot_key, contract["snapshot"])
        return
    raise RuntimeError(
        "Optimizer 記憶庫的 effective policy 與目前設定不一致，禁止接續同一個 study。"
        f"\n目前 policy: {contract['snapshot']}"
        "\n請改用新記憶庫，或先刪除舊記憶庫再重來。"
    )


def _find_finalist_entry(finalists, winner_trial):
    if winner_trial is None:
        return None
    for item in finalists:
        trial = item.get("trial")
        if trial is not None and int(trial.number) == int(winner_trial.number):
            return item
    return None


def _build_best_summary_payload(*, winner_trial, finalist_entry, objective_mode: str, walk_forward_policy: dict, action_label: str, selection_rule: str, compare_only: bool = False):
    if winner_trial is None or finalist_entry is None:
        raise ValueError("缺少 winner_trial 或 finalist_entry，無法建立 summary")
    effective_search_train_end_year = int(
        winner_trial.user_attrs.get("search_train_end_year", walk_forward_policy.get("search_train_end_year", 0))
    )
    payload = {
        "trial_number": int(winner_trial.number) + 1,
        "base_score": float(finalist_entry["base_score"]),
        "local_min_score": float(finalist_entry["local_min_score"]),
        "retention": float(finalist_entry["local_retention"]),
        "local_gate": bool(finalist_entry["gate_pass"]),
        "objective_mode": str(objective_mode),
        "train_start_year": int(walk_forward_policy.get("train_start_year", 0)),
        "search_train_end_year": int(effective_search_train_end_year),
        "selection_end_year": int(walk_forward_policy.get("search_train_end_year", effective_search_train_end_year)),
        "oos_start_year": walk_forward_policy.get("oos_start_year"),
        "action": str(action_label),
        "selection_rule": str(selection_rule),
        "compare_only": bool(compare_only),
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        "local_min_review_mode": str(finalist_entry.get("local_min_review_mode", "exact")),
        "local_min_exact": bool(finalist_entry.get("local_min_exact", bool(is_optimizer_local_min_review_enabled()))),
        "inner_validate_anti_overfit_enabled": bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED),
        "inner_validate_min_score": float(OPTIMIZER_INNER_VALIDATE_MIN_SCORE),
        "inner_validate_max_rank_percentile": float(OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE),
        "dominant_year_dependency_anti_overfit_enabled": bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED),
        "created_at": get_taipei_now().isoformat(),
    }
    inner_validate_diagnostics = finalist_entry.get("inner_validate_diagnostics")
    if isinstance(inner_validate_diagnostics, dict):
        payload["inner_validate_diagnostics"] = inner_validate_diagnostics
        payload["inner_validate_score"] = float(inner_validate_diagnostics.get("inner_validate_score", 0.0))
        payload["inner_validate_gate"] = bool(inner_validate_diagnostics.get("inner_validate_gate", False))
        if inner_validate_diagnostics.get("inner_validate_rank") is not None:
            payload["inner_validate_rank"] = int(inner_validate_diagnostics["inner_validate_rank"])
        if inner_validate_diagnostics.get("inner_validate_rank_cutoff") is not None:
            payload["inner_validate_rank_cutoff"] = int(inner_validate_diagnostics["inner_validate_rank_cutoff"])
        if inner_validate_diagnostics.get("inner_validate_rank_total") is not None:
            payload["inner_validate_rank_total"] = int(inner_validate_diagnostics["inner_validate_rank_total"])
        if inner_validate_diagnostics.get("validate_year") is not None:
            payload["inner_validate_year"] = int(inner_validate_diagnostics["validate_year"])
        if inner_validate_diagnostics.get("inner_train_end_year") is not None:
            payload["inner_train_end_year"] = int(inner_validate_diagnostics["inner_train_end_year"])
    diagnostics = finalist_entry.get("dominant_year_dependency_diagnostics")
    if isinstance(diagnostics, dict):
        payload["dominant_year_dependency_diagnostics"] = diagnostics
        payload["dependency_warning"] = bool(diagnostics.get("dependency_warning", False))
    return payload


def _resolve_candidate_best_selection_rule():
    parts = ["max_local_min_score" if is_optimizer_local_min_review_enabled() else "max_base_score_local_min_disabled"]
    if bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED):
        parts.append("inner_validate_score_gt_0_and_rank_top_half")
    if bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED):
        parts.append("intuitive_dominant_year_dependency_veto")
    return "_with_".join(parts)


def _summary_policy_signature(summary: dict | None):
    if not isinstance(summary, dict):
        return None
    return (
        str(summary.get("objective_mode", "")),
        summary.get("train_start_year"),
        summary.get("search_train_end_year"),
        summary.get("oos_start_year"),
    )


def _format_summary_policy_signature(summary: dict | None) -> str:
    signature = _summary_policy_signature(summary)
    if signature is None:
        return "N/A"
    objective_mode, train_start_year, search_train_end_year, oos_start_year = signature
    return (
        f"objective={objective_mode or 'N/A'}"
        f", train_start={train_start_year if train_start_year is not None else 'N/A'}"
        f", search_end={search_train_end_year if search_train_end_year is not None else 'N/A'}"
        f", oos_start={oos_start_year if oos_start_year is not None else 'N/A'}"
    )


def _summaries_have_compatible_policy(*, candidate_summary: dict, run_best_summary: dict | None) -> bool:
    if run_best_summary is None:
        return False
    return _summary_policy_signature(candidate_summary) == _summary_policy_signature(run_best_summary)


def _print_candidate_vs_run_best_summary(*, candidate_summary: dict, run_best_summary: dict | None):
    print(f"{C_GRAY}{'-' * 96}{C_RESET}")
    print(
        f"      candidate | base={float(candidate_summary.get('base_score', 0.0)):.3f} "
        f"| local_min={float(candidate_summary.get('local_min_score', 0.0)):.3f} "
        f"| retention={float(candidate_summary.get('retention', 0.0)):.3f} "
        f"| policy=({_format_summary_policy_signature(candidate_summary)})"
    )
    if run_best_summary is None:
        print(f"      run_best  | 尚無 summary，視同未建立 promote 基線")
    else:
        print(
            f"      run_best  | base={float(run_best_summary.get('base_score', 0.0)):.3f} "
            f"| local_min={float(run_best_summary.get('local_min_score', 0.0)):.3f} "
            f"| retention={float(run_best_summary.get('retention', 0.0)):.3f} "
            f"| policy=({_format_summary_policy_signature(run_best_summary)})"
        )
    print(f"{C_GRAY}{'-' * 96}{C_RESET}")


def _should_promote_candidate(*, candidate_summary: dict, run_best_summary: dict | None):
    candidate_local_min = float(candidate_summary.get("local_min_score", 0.0))
    candidate_base_score = float(candidate_summary.get("base_score", 0.0))
    candidate_retention = float(candidate_summary.get("retention", float("-inf")))
    if candidate_local_min <= 0.0:
        return False, "candidate.local_min_score <= 0"
    if candidate_base_score <= 0.0:
        return False, "candidate.base_score <= 0"
    if bool(candidate_summary.get("inner_validate_anti_overfit_enabled", False)):
        if not bool(candidate_summary.get("inner_validate_gate", False)):
            return False, "candidate.inner_validate_gate = FAIL"
    if run_best_summary is None:
        return True, "run_best summary 缺失，視同首次 promote"
    if not _summaries_have_compatible_policy(candidate_summary=candidate_summary, run_best_summary=run_best_summary):
        return True, "run_best summary effective policy 與 candidate 不一致，視同舊基準重新 promote"
    run_best_local_min = float(run_best_summary.get("local_min_score", float("-inf")))
    run_best_retention = float(run_best_summary.get("retention", float("-inf")))
    local_min_not_worse = candidate_local_min >= run_best_local_min
    retention_not_worse = candidate_retention >= run_best_retention
    strictly_better = candidate_local_min > run_best_local_min or candidate_retention > run_best_retention
    if local_min_not_worse and retention_not_worse and strictly_better:
        return True, "candidate local_min_score 與 retention 對 run_best 形成 Pareto 不劣，且至少一項勝出"
    if candidate_local_min > run_best_local_min and candidate_retention < run_best_retention:
        return False, "trade-off：candidate.local_min_score 較高，但 retention 較低，不自動 promote"
    if candidate_local_min < run_best_local_min and candidate_retention > run_best_retention:
        return False, "trade-off：candidate.retention 較高，但 local_min_score 較低，不自動 promote"
    return False, "candidate 未通過 Pareto promote 比較規則"


def _load_candidate_params_payload_for_promote():
    candidate_params = _load_json_file_or_none(CANDIDATE_BEST_PARAMS_PATH)
    if candidate_params is None:
        print(f"{C_RED}❌ 找不到 candidate_best 參數檔: {CANDIDATE_BEST_PARAMS_PATH}{C_RESET}", file=sys.stderr)
        return None
    if is_active_param_ensemble_payload(candidate_params):
        return candidate_params
    from core.params_io import load_params_from_json
    return load_params_from_json(CANDIDATE_BEST_PARAMS_PATH)


def _promote_candidate_to_run_best():
    candidate_params = _load_candidate_params_payload_for_promote()
    if candidate_params is None:
        return 1
    candidate_summary = _load_json_file_or_none(CANDIDATE_BEST_SUMMARY_PATH)
    if candidate_summary is None:
        print(f"{C_RED}❌ 找不到 candidate_best summary: {CANDIDATE_BEST_SUMMARY_PATH}{C_RESET}", file=sys.stderr)
        return 1
    run_best_summary = _load_json_file_or_none(RUN_BEST_SUMMARY_PATH)
    _print_candidate_vs_run_best_summary(candidate_summary=candidate_summary, run_best_summary=run_best_summary)
    should_promote, reason = _should_promote_candidate(candidate_summary=candidate_summary, run_best_summary=run_best_summary)
    if not should_promote:
        print(f"{C_YELLOW}ℹ️ run_best 未進版 ：{reason}{C_RESET}")
        return 0
    promoted_summary = dict(candidate_summary)
    promoted_summary["promoted_at"] = get_taipei_now().isoformat()
    _write_json_file(RUN_BEST_PARAMS_PATH, candidate_params)
    _write_json_file(RUN_BEST_SUMMARY_PATH, promoted_summary)
    print(f"{C_GREEN}✅ run_best 已進版：{RUN_BEST_PARAMS_PATH}{C_RESET}")
    return 0


def build_optimizer_session(*, walk_forward_policy: dict):
    from tools.optimizer.profile import OptimizerProfileRecorder
    from tools.optimizer.session import OptimizerSession
    from tools.optimizer.study_utils import (
        build_best_completed_trial_resolver,
        build_optimizer_trial_params,
        resolve_optimizer_tp_percent,
    )

    session_ts = get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    objective_mode = str(walk_forward_policy.get("objective_mode", "split_train_romd"))
    return OptimizerSession(
        output_dir=OUTPUT_DIR,
        session_ts=session_ts,
        profile_recorder_cls=OptimizerProfileRecorder,
        build_optimizer_trial_params=build_optimizer_trial_params,
        get_best_completed_trial_or_none=build_best_completed_trial_resolver(objective_mode),
        objective_mode=objective_mode,
        search_train_end_year=int(walk_forward_policy["search_train_end_year"]),
        walk_forward_policy=walk_forward_policy,
        resolve_optimizer_tp_percent=resolve_optimizer_tp_percent,
        print_strategy_dashboard=print_strategy_dashboard,
        colors=COLORS,
        optimizer_fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
        train_max_positions=TRAIN_MAX_POSITIONS,
        train_start_year=int(walk_forward_policy["train_start_year"]),
        train_enable_rotation=TRAIN_ENABLE_ROTATION,
        default_max_workers=DEFAULT_OPTIMIZER_MAX_WORKERS,
        enable_optimizer_profiling=ENABLE_OPTIMIZER_PROFILING,
        enable_profile_console_print=ENABLE_PROFILE_CONSOLE_PRINT,
        profile_print_every_n_trials=PROFILE_PRINT_EVERY_N_TRIALS,
    )


def generate_walk_forward_report_from_payload(*, session, params_payload, dataset_label, db_file, best_trial_number=None, walk_forward_policy: dict):
    from core.params_io import build_params_from_mapping
    from core.strategy_params import build_runtime_param_raw_value
    from tools.optimizer.prep import prepare_trial_inputs
    from tools.optimizer.walk_forward import evaluate_walk_forward, write_walk_forward_report

    ai_params = build_params_from_mapping(params_payload)
    prep_executor_bundle = session.get_trial_prep_executor_bundle(
        build_runtime_param_raw_value(ai_params, "optimizer_max_workers")
    )
    prep_result = prepare_trial_inputs(
        raw_data_cache=session.raw_data_cache,
        params=ai_params,
        default_max_workers=session.default_max_workers,
        executor_bundle=prep_executor_bundle,
        static_fast_cache=session.static_fast_cache,
        static_master_dates=session.master_dates,
        include_trade_logs=False,
        include_pit_stats_index=True,
    )
    report = evaluate_walk_forward(
        all_dfs_fast=prep_result["all_dfs_fast"],
        all_trade_logs=prep_result["all_trade_logs"],
        sorted_dates=sorted(prep_result["master_dates"]),
        params=ai_params,
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
        benchmark_ticker="0050",
        train_start_year=int(walk_forward_policy["train_start_year"]),
        min_train_years=int(walk_forward_policy["min_train_years"]),
        oos_start_year=walk_forward_policy.get("oos_start_year"),
        train_start_date=walk_forward_policy.get("train_start_date"),
        oos_start_date=walk_forward_policy.get("oos_start_date"),
        oos_end_date=walk_forward_policy.get("oos_end_date"),
        pit_stats_index=prep_result.get("all_pit_stats_index"),
    )
    report_paths = write_walk_forward_report(
        output_dir=session.output_dir,
        params_payload=params_payload,
        dataset_label=dataset_label,
        report=report,
        best_trial_number=best_trial_number,
        source_db_path=db_file,
        session_ts=session.session_ts,
    )
    return report, report_paths


def generate_best_trial_walk_forward_report(*, session, best_trial, dataset_label, db_file, walk_forward_policy: dict):
    from tools.optimizer.study_utils import build_best_params_payload_from_trial

    params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=session.optimizer_fixed_tp_percent)
    report, report_paths = generate_walk_forward_report_from_payload(
        session=session,
        params_payload=params_payload,
        dataset_label=dataset_label,
        db_file=db_file,
        best_trial_number=int(best_trial.number) + 1,
        walk_forward_policy=walk_forward_policy,
    )
    return report, report_paths


def print_walk_forward_outputs(*, report, report_paths):
    return None


def finalize_best_trial_outputs(*, session, study, best_trial_resolver, dataset_label: str, db_file: str, walk_forward_policy: dict):
    from tools.optimizer.study_utils import is_qualified_trial_value

    best_trial = best_trial_resolver(study)
    if best_trial is None:
        print(f"{C_YELLOW}ℹ️ 目前尚無通過 local_min_score gate 的 winner，略過 OOS 報表輸出。{C_RESET}")
        return 0
    if not is_qualified_trial_value(best_trial.value):
        return 0

    report, report_paths = generate_best_trial_walk_forward_report(
        session=session,
        best_trial=best_trial,
        dataset_label=dataset_label,
        db_file=db_file,
        walk_forward_policy=walk_forward_policy,
    )
    print_walk_forward_outputs(report=report, report_paths=report_paths)
    return 0


def _export_selected_candidate_artifacts(
    *,
    study,
    finalists,
    selected_trial,
    params_path: str,
    summary_path: str,
    fixed_tp_percent: float,
    colors: dict,
    objective_mode: str,
    walk_forward_policy: dict,
    action_label: str,
    selection_rule: str,
    compare_only: bool,
    artifact_label: str,
    export_best_params_if_requested,
    is_qualified_trial_value,
):
    if selected_trial is None or not is_qualified_trial_value(selected_trial.value):
        print(f"{C_YELLOW}ℹ️ {artifact_label} 無可匯出 trial。{C_RESET}")
        return False
    export_status = export_best_params_if_requested(
        study,
        best_params_path=params_path,
        fixed_tp_percent=fixed_tp_percent,
        colors=colors,
        best_trial_resolver=(lambda _study, _selected_trial=selected_trial: _selected_trial),
        suppress_success_message=True,
    )
    if export_status != 0:
        return False
    finalist_entry = _find_finalist_entry(finalists, selected_trial)
    if finalist_entry is None:
        raise ValueError(f"找不到 {artifact_label} 的 finalist entry，無法建立 summary")
    candidate_summary = _build_best_summary_payload(
        winner_trial=selected_trial,
        finalist_entry=finalist_entry,
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        action_label=action_label,
        selection_rule=selection_rule,
        compare_only=compare_only,
    )
    _write_json_file(summary_path, candidate_summary)
    compare_note = "（比較用，不參與 promote）" if compare_only else ""
    print(f"{C_GREEN}💾 {artifact_label}{compare_note} 已寫入：{params_path}{C_RESET}")
    return True


def _resolve_nonrolling_seed_ensemble_policy():
    return build_seed_ensemble_policy_snapshot(
        enabled=OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
        seed_count=OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
        min_agree=OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
    )


def _resolve_optimizer_session_ts(session, *, fallback_label: str = "") -> str:
    raw_session_ts = str(getattr(session, "session_ts", "") or "").strip()
    if raw_session_ts:
        return raw_session_ts
    suffix = str(fallback_label or "").strip()
    generated = get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    return generated if not suffix else f"{generated}_{suffix}"


def _build_seed_ensemble_member(*, member_index: int, seed: int, best_trial, finalist_entry: dict, params_payload: dict) -> dict:
    member = {
        "member_index": int(member_index),
        "seed": int(seed),
        "selected_trial": int(best_trial.number) + 1,
        "params": dict(params_payload),
    }
    if isinstance(finalist_entry, dict):
        member["score"] = float(finalist_entry.get("base_score", 0.0))
        member["base_score"] = float(finalist_entry.get("base_score", 0.0))
        member["local_min_score"] = float(finalist_entry.get("local_min_score", 0.0))
        member["retention"] = float(finalist_entry.get("local_retention", 0.0))
        member["local_gate"] = bool(finalist_entry.get("gate_pass", False))
        member["local_min_review_enabled"] = bool(finalist_entry.get("local_min_review_enabled", is_optimizer_local_min_review_enabled()))
        member["local_min_review_mode"] = str(finalist_entry.get("local_min_review_mode", "exact"))
        member["local_min_exact"] = bool(finalist_entry.get("local_min_exact", bool(is_optimizer_local_min_review_enabled())))
    return member



def _format_nonrolling_result_number(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _build_nonrolling_single_fold_period_context(walk_forward_policy: dict) -> dict:
    selection_start_year = int(walk_forward_policy.get("selection_start_year", walk_forward_policy.get("train_start_year", 0)) or 0)
    selection_end_year = int(walk_forward_policy.get("search_train_end_year", selection_start_year) or selection_start_year)
    oos_start_year = int(walk_forward_policy.get("oos_start_year", selection_end_year + 1) or (selection_end_year + 1))
    return {
        "fold_idx": 1,
        "fold_count": 1,
        "selection_start": f"{selection_start_year:04d}-01-01" if selection_start_year > 0 else "",
        "selection_end": f"{selection_end_year:04d}-12-31" if selection_end_year > 0 else "",
        "selection_period": f"{selection_start_year:04d}-01~{selection_end_year % 100:02d}-12" if selection_start_year > 0 and selection_end_year > 0 else "",
        "oos_year": oos_start_year,
        "oos_period": f"{oos_start_year:04d}~latest" if oos_start_year > 0 else "",
    }


def _render_nonrolling_single_fold_progress_line(walk_forward_policy: dict, *, stage: str, status: str = "", completed: int = 0, total: int = 0, best_score=None, best_base_score=None, best_local_min_score=None, elapsed_sec=None) -> str:
    from tools.optimizer.outer_rolling_oos import render_optimizer_fold_progress_line

    context = _build_nonrolling_single_fold_period_context(walk_forward_policy)
    return render_optimizer_fold_progress_line(
        **context,
        stage=stage,
        status=status,
        completed=completed,
        total=total,
        best_score=best_score,
        best_base_score=best_base_score,
        best_local_min_score=best_local_min_score,
        elapsed_sec=elapsed_sec,
    )


def _finalize_nonrolling_single_fold_inline_progress(progress_owner) -> None:
    if not bool(getattr(progress_owner, "nonrolling_fold_inline_progress_rendered", False)):
        return
    print(flush=True)
    progress_owner.nonrolling_fold_inline_progress_rendered = False
    progress_owner.nonrolling_fold_inline_progress_width = 0


def _print_nonrolling_single_fold_progress(walk_forward_policy: dict, *, stage: str, status: str = "", completed: int = 0, total: int = 0, best_score=None, best_base_score=None, best_local_min_score=None, elapsed_sec=None, inline: bool = False, progress_owner=None) -> None:
    line = _render_nonrolling_single_fold_progress_line(
        walk_forward_policy,
        stage=stage,
        status=status,
        completed=completed,
        total=total,
        best_score=best_score,
        best_base_score=best_base_score,
        best_local_min_score=best_local_min_score,
        elapsed_sec=elapsed_sec,
    )
    owner = progress_owner
    if inline and owner is not None and stdout_supports_inline_progress():
        previous_width = int(getattr(owner, "nonrolling_fold_inline_progress_width", 0) or 0)
        owner.nonrolling_fold_inline_progress_width = write_inline_progress(f"{C_GRAY}{line}{C_RESET}", previous_width=previous_width)
        owner.nonrolling_fold_inline_progress_rendered = True
        return
    if owner is not None:
        _finalize_nonrolling_single_fold_inline_progress(owner)
    print(f"{C_GRAY}{line}{C_RESET}", flush=True)


def _make_nonrolling_seed_trial_progress_callback(
    *,
    member_session,
    walk_forward_policy: dict,
    requested_trials: int,
    started_at: float,
    status: str = "",
    inline: bool = True,
    emit_lock=None,
    throttle_trials: int = 1,
    throttle_seconds: float = 0.0,
    progress_board=None,
    seed_index: int = 0,
    seed: int = 0,
    seed_count: int = 0,
):
    state = {"last_completed": 0, "last_emitted_at": 0.0}

    def _should_emit(completed: int, now: float) -> bool:
        total = max(1, int(requested_trials))
        if completed <= 0:
            return False
        if completed >= total:
            return True
        if int(throttle_trials) <= 1:
            return True
        if completed - int(state.get("last_completed", 0) or 0) >= int(throttle_trials):
            return True
        if float(throttle_seconds) > 0 and now - float(state.get("last_emitted_at", 0.0) or 0.0) >= float(throttle_seconds):
            return True
        return False

    def _emit_progress(**kwargs) -> None:
        if progress_board is not None and int(seed_index or 0) > 0:
            progress = {
                "stage": kwargs.get("stage"),
                "status": kwargs.get("status"),
                "seed_ensemble_member_index": int(seed_index),
                "seed_ensemble_member_count": int(seed_count or 0),
                "seed": int(seed or 0),
                "completed": kwargs.get("completed"),
                "total": kwargs.get("total"),
                "best_score": kwargs.get("best_score"),
                "elapsed_sec": kwargs.get("elapsed_sec"),
            }
            if emit_lock is None:
                progress_board.update(fold_idx=1, seed_index=int(seed_index), progress=progress)
                return
            with emit_lock:
                progress_board.update(fold_idx=1, seed_index=int(seed_index), progress=progress)
            return
        if emit_lock is None:
            _print_nonrolling_single_fold_progress(**kwargs)
            return
        with emit_lock:
            _print_nonrolling_single_fold_progress(**kwargs)

    def _callback(study, trial):
        completed = int(getattr(trial, "number", -1)) + 1
        now = time.perf_counter()
        if not _should_emit(completed, now):
            return
        best_trial = member_session.get_best_completed_trial_or_none(study)
        best_score = None if best_trial is None else best_trial.value
        _emit_progress(
            walk_forward_policy=walk_forward_policy,
            stage="OPTIMIZER_SEARCH",
            status=status,
            completed=completed,
            total=int(requested_trials),
            best_score=best_score,
            elapsed_sec=max(0.0, now - float(started_at)),
            inline=bool(inline),
            progress_owner=member_session if inline else None,
        )
        state["last_completed"] = completed
        state["last_emitted_at"] = now

    return _callback




def _emit_nonrolling_seed_process_progress_event(task: dict, *, stage: str, status: str = "", completed: int = 0, total: int = 0, best_score=None, best_base_score=None, best_local_min_score=None, elapsed_sec=None) -> None:
    from tools.optimizer.outer_rolling_oos import write_optimizer_seed_progress_event

    context = dict((task or {}).get("progress_context") or {})
    member_index = int((task or {}).get("member_index", 0) or 0)
    member_count = int((task or {}).get("member_count", 0) or 0)
    seed = int((task or {}).get("seed", 0) or 0)
    payload = {
        "seed_ensemble_member_index": int(member_index),
        "seed_ensemble_member_count": int(member_count),
        "seed": int(seed),
        "status": str(status or ""),
        "elapsed_sec": elapsed_sec,
    }
    if completed is not None:
        payload["completed"] = int(completed or 0)
    if total is not None:
        payload["total"] = int(total or 0)
    if best_score is not None:
        payload["best_score"] = float(best_score)
    if best_base_score is not None:
        payload["best_base_score"] = float(best_base_score)
    if best_local_min_score is not None:
        payload["best_local_min_score"] = float(best_local_min_score)
    write_optimizer_seed_progress_event(
        stage=str(stage),
        fold_idx=int(context.get("fold_idx", 1) or 1),
        fold_count=int(context.get("fold_count", 1) or 1),
        oos_year=int(context.get("oos_year", 0) or 0),
        selection_start=str(context.get("selection_start") or ""),
        selection_end=str(context.get("selection_end") or ""),
        **payload,
    )


def _make_nonrolling_seed_process_trial_progress_callback(*, task: dict, member_session, requested_trials: int, started_at: float):
    state = {"last_completed": 0}

    def _callback(study, trial):
        completed = int(getattr(trial, "number", -1)) + 1
        if completed <= 0 or completed == int(state.get("last_completed", 0) or 0):
            return
        best_trial = member_session.get_best_completed_trial_or_none(study)
        best_score = None if best_trial is None else best_trial.value
        member_index = int((task or {}).get("member_index", 0) or 0)
        member_count = int((task or {}).get("member_count", 0) or 0)
        _emit_nonrolling_seed_process_progress_event(
            task,
            stage="OPTIMIZER_SEARCH",
            status=f"seed {member_index}/{member_count}",
            completed=int(completed),
            total=int(requested_trials),
            best_score=best_score,
            elapsed_sec=max(0.0, time.perf_counter() - float(started_at)),
        )
        state["last_completed"] = int(completed)

    return _callback


def _run_nonrolling_seed_ensemble_member_process_task(task: dict) -> dict | None:
    log_path = str((task or {}).get("log_path") or "")

    def _execute() -> dict | None:
        from tools.optimizer.prep import load_all_raw_data as load_all_raw_data_func
        from tools.optimizer.runtime import create_optimizer_study, resolve_optimizer_single_fold_search_parallel_trials
        from tools.optimizer.robustness import print_local_min_score_finalist_review, print_local_min_score_winner_summary
        from tools.optimizer.session import close_study_storage
        from tools.optimizer.study_utils import build_best_params_payload_from_trial, is_qualified_trial_value

        configure_optuna_logging()
        walk_forward_policy = dict(task["walk_forward_policy"])
        selected_data_dir = str(task["selected_data_dir"])
        optimizer_required_min_rows = int(task["optimizer_required_min_rows"])
        requested_trials = int(task["requested_trials"])
        seed = int(task["seed"])
        member_index = int(task["member_index"])
        member_count = int(task["member_count"])
        started_at = time.perf_counter()
        study = None
        member_session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
        member_session.n_trials = int(requested_trials)
        member_session.run_action = "train"
        member_session.disable_milestone_dashboard = True
        member_session.disable_optimizer_status_line = True
        try:
            _emit_nonrolling_seed_process_progress_event(
                task,
                stage="START",
                status=f"seed {member_index}/{member_count} seed={seed}",
                elapsed_sec=0.0,
            )
            task_db_name = task.get("db_name")
            if isinstance(task_db_name, str) and task_db_name.strip().lower() in {"", "none", "null"}:
                task_db_name = None
            study = create_optimizer_study(task_db_name, seed=int(seed), sampler_kind="tpe")
            _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
            _emit_nonrolling_seed_process_progress_event(
                task,
                stage="RAW_DATA",
                status=f"seed {member_index}/{member_count} raw data",
                elapsed_sec=max(0.0, time.perf_counter() - started_at),
            )
            member_session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data_func,
                required_min_rows=optimizer_required_min_rows,
                verbose=False,
            )
            member_session.profile_recorder.init_output_files()
            member_session.profile_recorder.mark_run_started()
            _emit_nonrolling_seed_process_progress_event(
                task,
                stage="OPTIMIZER_SEARCH",
                status=f"seed {member_index}/{member_count}",
                completed=0,
                total=int(requested_trials),
                elapsed_sec=max(0.0, time.perf_counter() - started_at),
            )
            search_parallel_trials = resolve_optimizer_single_fold_search_parallel_trials(task.get("environ") or os.environ, sampler_kind="tpe")
            study.optimize(
                member_session.objective,
                n_trials=int(requested_trials),
                n_jobs=int(search_parallel_trials),
                callbacks=[
                    member_session.monitoring_callback,
                    _make_nonrolling_seed_process_trial_progress_callback(
                        task=task,
                        member_session=member_session,
                        requested_trials=int(requested_trials),
                        started_at=started_at,
                    ),
                ],
            )
            finalists, best_trial = print_local_min_score_finalist_review(
                study,
                session=member_session,
                objective_mode=str(task.get("objective_mode") or "split_train_romd"),
                colors=COLORS,
                winner_trial=None,
                emit_table=False,
                show_progress=False,
            )
            if best_trial is None or not is_qualified_trial_value(best_trial.value):
                _emit_nonrolling_seed_process_progress_event(
                    task,
                    stage="DONE",
                    status=f"seed {member_index}/{member_count} skipped",
                    elapsed_sec=max(0.0, time.perf_counter() - started_at),
                )
                return None
            print_local_min_score_winner_summary(winner_trial=best_trial, session=member_session, colors=COLORS)
            finalist_entry = _find_finalist_entry(finalists, best_trial)
            if finalist_entry is None:
                raise ValueError(f"seed={seed} 找不到 finalist entry，無法建立 ensemble member")
            params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
            member_payload = _build_seed_ensemble_member(
                member_index=member_index,
                seed=seed,
                best_trial=best_trial,
                finalist_entry=finalist_entry,
                params_payload=params_payload,
            )
            from tools.optimizer.outer_rolling_oos import build_optimizer_policy_members_from_finalists
            policy_members = build_optimizer_policy_members_from_finalists(
                finalists,
                objective_mode=str(task.get("objective_mode") or "split_train_romd"),
                member_index=int(member_index),
                seed=int(seed),
            )
            _emit_nonrolling_seed_process_progress_event(
                task,
                stage="DONE",
                status=f"seed {member_index}/{member_count} done",
                best_base_score=member_payload.get("base_score"),
                best_local_min_score=member_payload.get("local_min_score"),
                elapsed_sec=max(0.0, time.perf_counter() - started_at),
            )
            return {"member": member_payload, "policy_members": policy_members}
        finally:
            member_session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    try:
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as log_handle:
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    return _execute()
        return _execute()
    except Exception as exc:
        if log_path:
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
                traceback.print_exc(file=log_handle)
        raise RuntimeError(f"nonrolling seed member failed: seed={int((task or {}).get('seed', 0) or 0)} error={type(exc).__name__}: {exc}") from exc


def _print_static_seed_ensemble_result_table(*, members: list[dict], policy: dict, colors: dict) -> None:
    if not bool(is_optimizer_nonrolling_train_result_table_enabled()):
        return
    gray = colors.get("gray", "")
    green = colors.get("green", "")
    red = colors.get("red", "")
    yellow = colors.get("yellow", "")
    reset = colors.get("reset", "")
    member_count = len(members)
    min_agree = int(policy.get("min_agree", member_count or 1))
    title = "NON-ROLLING RANDOM SEED ENSEMBLE RESULTS"
    header = (
        f"{'member':<8} | {'seed':>10} | {'trial':>8} | "
        f"{'base':>10} | {'local_min':>10} | {'retention':>10} | {'gate':>8} | {'result':>10}"
    )
    separator_width = max(len(title), len(header))
    print(f"{gray}{'-' * separator_width}{reset}")
    print(title)
    print(f"{gray}{'-' * separator_width}{reset}")
    print(header)
    print(f"{gray}{'-' * separator_width}{reset}")
    pass_count = 0
    for idx, item in enumerate(members, start=1):
        gate_pass = bool(item.get("local_gate", False))
        if gate_pass:
            pass_count += 1
        gate_text = "PASS" if gate_pass else "FAIL"
        gate_color = green if gate_pass else red
        result_text = "member"
        result_color = green if gate_pass else red
        print(
            f"#{int(item.get('member_index', idx)):<7} | "
            f"{int(item.get('seed', 0)):>10} | "
            f"#{int(item.get('selected_trial', 0)):>7} | "
            f"{_format_nonrolling_result_number(item.get('base_score')):>10} | "
            f"{_format_nonrolling_result_number(item.get('local_min_score')):>10} | "
            f"{_format_nonrolling_result_number(item.get('retention')):>10} | "
            f"{gate_color}{gate_text:>8}{reset} | "
            f"{result_color}{result_text:>10}{reset}"
        )
    base_scores = [float(item.get("base_score", 0.0)) for item in members]
    local_scores = [float(item.get("local_min_score", 0.0)) for item in members]
    retentions = [float(item.get("retention", 0.0)) for item in members]
    ensemble_pass = bool(member_count > 0 and pass_count >= min_agree)
    ensemble_color = green if ensemble_pass else red
    ensemble_result = "PASS" if ensemble_pass else "FAIL"
    print(f"{gray}{'=' * separator_width}{reset}")
    print(
        f"ENSEMBLE | N={member_count} | min_agree={min_agree} | "
        f"gate_pass={pass_count}/{member_count} | "
        f"base_min={_format_nonrolling_result_number(min(base_scores) if base_scores else None)} | "
        f"local_min_min={_format_nonrolling_result_number(min(local_scores) if local_scores else None)} | "
        f"retention_min={_format_nonrolling_result_number(min(retentions) if retentions else None)} | "
        f"result={ensemble_color}{ensemble_result}{reset}"
    )
    print(f"{yellow}正式輸出：candidate_best / run_best 使用同一個 static ensemble JSON。{reset}")


def _build_static_seed_ensemble_summary(*, members: list[dict], seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int) -> dict:
    local_scores = [float(item.get("local_min_score", 0.0)) for item in members]
    base_scores = [float(item.get("base_score", 0.0)) for item in members]
    retentions = [float(item.get("retention", 0.0)) for item in members]
    return {
        "schema_type": "optimizer_active_param_ensemble_summary",
        "schema_version": 1,
        "action": "train",
        "selection_rule": "random_seed_ensemble_static_consensus",
        "objective_mode": str(objective_mode),
        "train_start_year": int(walk_forward_policy.get("train_start_year", 0)),
        "search_train_end_year": int(walk_forward_policy.get("search_train_end_year", 0)),
        "selection_end_year": int(walk_forward_policy.get("search_train_end_year", 0)),
        "oos_start_year": walk_forward_policy.get("oos_start_year"),
        "dataset_label": str(dataset_label),
        "selected_model_mode": str(selected_model_mode),
        "trials_per_seed": int(trials_per_seed),
        "seed_count": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "base_score": min(base_scores) if base_scores else 0.0,
        "local_min_score": min(local_scores) if local_scores else 0.0,
        "retention": min(retentions) if retentions else 0.0,
        "local_gate": all(bool(item.get("local_gate", False)) for item in members),
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        "member_metrics": [
            {
                "member_index": int(item.get("member_index", idx + 1)),
                "seed": int(item.get("seed", 0)),
                "selected_trial": int(item.get("selected_trial", 0)),
                "base_score": float(item.get("base_score", 0.0)),
                "local_min_score": float(item.get("local_min_score", 0.0)),
                "retention": float(item.get("retention", 0.0)),
                "local_gate": bool(item.get("local_gate", False)),
                "local_min_review_enabled": bool(item.get("local_min_review_enabled", is_optimizer_local_min_review_enabled())),
                "local_min_review_mode": str(item.get("local_min_review_mode", "exact")),
            }
            for idx, item in enumerate(members)
        ],
        "random_seed_ensemble": _resolve_nonrolling_seed_ensemble_policy(),
        "nonrolling_train_result_table_enabled": bool(is_optimizer_nonrolling_train_result_table_enabled()),
        "created_at": get_taipei_now().isoformat(),
    }


def _build_static_seed_ensemble_meta(*, seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int, source: str, policy_name: str = "") -> dict:
    meta = {
        "source": str(source),
        "dataset_label": str(dataset_label),
        "selected_model_mode": str(selected_model_mode),
        "objective_mode": str(objective_mode),
        "trials_per_seed": int(trials_per_seed),
        "seeds": [int(seed) for seed in seeds],
        "walk_forward_policy": dict(walk_forward_policy),
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
    }
    if policy_name:
        meta["policy"] = str(policy_name)
    return meta


def _write_static_seed_ensemble_policy_paramsets(*, policy_members_by_policy: dict[str, list[dict]], seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int) -> dict[str, str]:
    from tools.optimizer.outer_rolling_oos import (
        get_optimizer_paramset_policy_names,
        get_optimizer_policy_paramset_filename,
    )

    paths: dict[str, str] = {}
    requested_policy = _resolve_nonrolling_seed_ensemble_policy()
    for policy_name in get_optimizer_paramset_policy_names():
        members = sorted(
            list((policy_members_by_policy or {}).get(str(policy_name)) or []),
            key=lambda item: int(dict(item).get("member_index", 0) or 0),
        )
        if not members:
            continue
        payload = build_static_active_param_ensemble_payload(
            members=members,
            random_seed_ensemble=requested_policy,
            selector=str(policy_name),
            created_at=get_taipei_now().isoformat(),
            meta=_build_static_seed_ensemble_meta(
                seeds=seeds,
                objective_mode=objective_mode,
                walk_forward_policy=walk_forward_policy,
                dataset_label=dataset_label,
                selected_model_mode=selected_model_mode,
                trials_per_seed=trials_per_seed,
                source="nonrolling_random_seed_ensemble_policy_paramset",
                policy_name=str(policy_name),
            ),
        )
        payload["summary"] = {
            "folds": 1,
            "mode": "static",
            "selector": str(policy_name),
            "selection_period": f"{int(walk_forward_policy.get('train_start_year', 0) or 0):04d}~{int(walk_forward_policy.get('search_train_end_year', 0) or 0):04d}",
            "oos_period": f"{int(walk_forward_policy.get('oos_start_year', 0) or 0):04d}~latest" if int(walk_forward_policy.get('oos_start_year', 0) or 0) > 0 else "",
            "member_count": int(len(members)),
            "requested_seed_count": int(len(seeds)),
            "trials_per_seed": int(trials_per_seed),
            "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        }
        filename = get_optimizer_policy_paramset_filename(str(policy_name))
        path = os.path.join(MODELS_DIR, filename)
        _write_json_file(path, payload)
        paths[str(policy_name)] = path
    return paths


def _write_static_seed_ensemble_candidate(*, members: list[dict], seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int, policy_members_by_policy: dict[str, list[dict]] | None = None) -> None:
    policy = _resolve_nonrolling_seed_ensemble_policy()
    payload = build_static_active_param_ensemble_payload(
        members=members,
        random_seed_ensemble=policy,
        selector="candidate_best",
        created_at=get_taipei_now().isoformat(),
        meta=_build_static_seed_ensemble_meta(
            seeds=seeds,
            objective_mode=objective_mode,
            walk_forward_policy=walk_forward_policy,
            dataset_label=dataset_label,
            selected_model_mode=selected_model_mode,
            trials_per_seed=trials_per_seed,
            source="nonrolling_random_seed_ensemble",
        ),
    )
    policy_paramset_paths = _write_static_seed_ensemble_policy_paramsets(
        policy_members_by_policy=dict(policy_members_by_policy or {}),
        seeds=seeds,
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        dataset_label=dataset_label,
        selected_model_mode=selected_model_mode,
        trials_per_seed=trials_per_seed,
    )
    summary = _build_static_seed_ensemble_summary(
        members=members,
        seeds=seeds,
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        dataset_label=dataset_label,
        selected_model_mode=selected_model_mode,
        trials_per_seed=trials_per_seed,
    )
    summary["policy_paramsets"] = dict(policy_paramset_paths)
    _write_json_file(CANDIDATE_BEST_PARAMS_PATH, payload)
    _write_json_file(CANDIDATE_BEST_SUMMARY_PATH, summary)
    print(f"{C_GREEN}💾 candidate_best seed ensemble 已寫入：{CANDIDATE_BEST_PARAMS_PATH}{C_RESET}")
    if policy_paramset_paths:
        joined = " | ".join(f"{name}={path}" for name, path in policy_paramset_paths.items())
        print(f"{C_GREEN}💾 nonrolling policy paramsets 已寫入：{joined}{C_RESET}")
    return payload, summary




def _tail_nonrolling_seed_log(path: str, *, max_lines: int = 24) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-int(max_lines):]).strip()
    except OSError:
        return ""


def _run_nonrolling_random_seed_ensemble_training(
    *,
    environ,
    selected_data_dir: str,
    dataset_profile_key: str,
    dataset_label: str,
    selected_model_mode: str,
    load_all_raw_data: bool,
    optimizer_required_min_rows: int,
    objective_mode: str,
    walk_forward_policy: dict,
    requested_trials: int,
    build_optimizer_session,
    create_optimizer_study,
    ensure_study_effective_policy_compatible,
    configure_optuna_logging,
    print_local_min_score_finalist_review,
    print_local_min_score_winner_summary,
    close_study_storage,
    is_qualified_trial_value,
    resolve_optimizer_single_fold_search_parallel_trials,
    build_best_params_payload_from_trial,
):
    if int(requested_trials) <= 0:
        return None
    if not os.path.isdir(selected_data_dir):
        raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
    policy = _resolve_nonrolling_seed_ensemble_policy()
    if not bool(policy.get("enabled", False)) or int(policy.get("seed_count", 1)) <= 1:
        return None

    seeds = generate_random_seed_ensemble(int(policy["seed_count"]))
    compact_display = not bool(is_optimizer_nonrolling_train_result_table_enabled())

    members: list[dict] = []
    policy_members_by_policy: dict[str, list[dict]] = {}
    dashboard_session = None
    ensemble_started_at = time.perf_counter()
    parallel_workers = resolve_optimizer_random_seed_ensemble_parallel_workers_default(len(seeds))
    parallel_backend = resolve_optimizer_random_seed_ensemble_parallel_backend_default()
    process_parallel_enabled = bool(int(parallel_workers) > 1 and parallel_backend == "process")
    process_log_dir = os.path.join(OUTPUT_DIR, "seed_ensemble_logs", get_taipei_now().strftime("%Y%m%d_%H%M%S_%f"))
    process_log_paths: dict[int, str] = {}
    progress_lock = threading.Lock()
    progress_board = None
    def _collect_policy_members(policy_members: dict | None) -> None:
        for policy_name, member in dict(policy_members or {}).items():
            if not isinstance(member, dict):
                continue
            policy_members_by_policy.setdefault(str(policy_name), []).append(dict(member))

    if compact_display:
        from tools.optimizer.outer_rolling_oos import OptimizerSeedEnsembleProgressBoard

        period_context = _build_nonrolling_single_fold_period_context(walk_forward_policy)
        contexts = []
        for member_index, seed in enumerate(seeds, start=1):
            context = dict(period_context)
            context.update({
                "seed_index": int(member_index),
                "seed_count": int(len(seeds)),
                "seed": int(seed),
                "log_path": os.path.join(process_log_dir, f"seed_{int(member_index):02d}_{int(seed)}.log"),
                "selection_start": str(period_context.get("selection_start") or ""),
                "selection_end": str(period_context.get("selection_end") or ""),
                "oos_period": str(period_context.get("oos_period") or ""),
            })
            contexts.append(context)
        progress_board = OptimizerSeedEnsembleProgressBoard(
            contexts,
            header=(
                f"seed ensemble | folds=1 | seeds={len(seeds)} | min_agree={policy['min_agree']} | "
                f"parallel_workers={int(parallel_workers)} | backend={parallel_backend}"
            ),
        )
        progress_board.render(force=True)
    else:
        print(f"{C_GRAY}🎲 Random seed ensemble｜N={len(seeds)}｜min_agree={policy['min_agree']}｜backend={parallel_backend}｜seeds={','.join(str(seed) for seed in seeds)}{C_RESET}")
    configure_optuna_logging()

    def _emit_seed_progress_for_member(member_index: int, seed: int, **kwargs) -> None:
        if compact_display and progress_board is not None:
            progress = dict(kwargs)
            progress.setdefault("seed_ensemble_member_index", int(member_index))
            progress.setdefault("seed_ensemble_member_count", int(len(seeds)))
            progress.setdefault("seed", int(seed))
            with progress_lock:
                progress_board.update(fold_idx=1, seed_index=int(member_index), progress=progress)
            return
        _print_nonrolling_single_fold_progress(walk_forward_policy, **kwargs)

    def _run_one_seed_member(member_index: int, seed: int) -> dict | None:
        member_session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
        member_session.n_trials = int(requested_trials)
        member_session.run_action = "train"
        member_session.disable_milestone_dashboard = compact_display
        member_session.disable_optimizer_status_line = compact_display
        study = None
        try:
            if compact_display:
                _emit_seed_progress_for_member(member_index, int(seed),
                    stage="START",
                    status=f"seed {member_index}/{len(seeds)} seed={int(seed)}",
                    elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                )
            else:
                print(f"{C_CYAN}[seed {member_index}/{len(seeds)}] seed={int(seed)} | trials={int(requested_trials)}{C_RESET}")
            study = create_optimizer_study(None, seed=int(seed), sampler_kind="tpe")
            ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
            if compact_display:
                _emit_seed_progress_for_member(member_index, int(seed),
                    stage="RAW_DATA",
                    status=f"seed {member_index}/{len(seeds)} raw data",
                    elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                )
            member_session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data,
                required_min_rows=optimizer_required_min_rows,
                verbose=bool(is_optimizer_nonrolling_train_result_table_enabled()),
            )
            member_session.profile_recorder.init_output_files()
            member_session.profile_recorder.mark_run_started()
            search_parallel_trials = resolve_optimizer_single_fold_search_parallel_trials(environ, sampler_kind="tpe")
            search_callbacks = [member_session.monitoring_callback]
            # 多 seed 並行時避免多個 inline trial progress 同時爭用 stdout；仍保留 seed-level rolling 同源進度。
            emit_trial_progress = compact_display
            if emit_trial_progress:
                progress_status = f"seed {member_index}/{len(seeds)}" if int(parallel_workers) > 1 else ""
                _emit_seed_progress_for_member(
                    member_index,
                    int(seed),
                    stage="OPTIMIZER_SEARCH",
                    status=progress_status,
                    completed=0,
                    total=int(requested_trials),
                    elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                )
                search_callbacks.append(
                    _make_nonrolling_seed_trial_progress_callback(
                        member_session=member_session,
                        walk_forward_policy=walk_forward_policy,
                        requested_trials=int(requested_trials),
                        started_at=ensemble_started_at,
                        status=progress_status,
                        inline=False if progress_board is not None else bool(int(parallel_workers) <= 1),
                        emit_lock=progress_lock,
                        throttle_trials=1,
                        throttle_seconds=0.0,
                        progress_board=progress_board,
                        seed_index=int(member_index),
                        seed=int(seed),
                        seed_count=int(len(seeds)),
                    )
                )
            study.optimize(
                member_session.objective,
                n_trials=int(requested_trials),
                n_jobs=int(search_parallel_trials),
                callbacks=search_callbacks,
            )
            if compact_display:
                _finalize_nonrolling_single_fold_inline_progress(member_session)
            member_session.outer_rolling_local_progress_context = _build_nonrolling_single_fold_period_context(walk_forward_policy) if compact_display and int(parallel_workers) <= 1 else None
            finalists, best_trial = print_local_min_score_finalist_review(
                study,
                session=member_session,
                objective_mode=objective_mode,
                colors=COLORS,
                winner_trial=None,
                emit_table=bool(is_optimizer_nonrolling_train_result_table_enabled()),
                show_progress=True if compact_display and int(parallel_workers) <= 1 else bool(is_optimizer_nonrolling_train_result_table_enabled()),
            )
            if compact_display:
                _finalize_nonrolling_single_fold_inline_progress(member_session)
            if best_trial is None or not is_qualified_trial_value(best_trial.value):
                print(f"{C_YELLOW}ℹ️ seed={int(seed)} 目前尚無通過 local_min_score gate 的 winner；本次不建立完整 N-seed ensemble member。{C_RESET}")
                return None
            print_local_min_score_winner_summary(
                winner_trial=best_trial,
                session=member_session,
                colors=COLORS,
            )
            finalist_entry = _find_finalist_entry(finalists, best_trial)
            if finalist_entry is None:
                raise ValueError(f"seed={int(seed)} 找不到 finalist entry，無法建立 ensemble member")
            params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
            member_payload = _build_seed_ensemble_member(
                member_index=member_index,
                seed=int(seed),
                best_trial=best_trial,
                finalist_entry=finalist_entry,
                params_payload=params_payload,
            )
            from tools.optimizer.outer_rolling_oos import build_optimizer_policy_members_from_finalists
            policy_members = build_optimizer_policy_members_from_finalists(
                finalists,
                objective_mode=objective_mode,
                member_index=int(member_index),
                seed=int(seed),
            )
            if compact_display:
                _emit_seed_progress_for_member(member_index, int(seed),
                    stage="DONE",
                    status=f"seed {member_index}/{len(seeds)} done",
                    best_base_score=member_payload.get("base_score"),
                    best_local_min_score=member_payload.get("local_min_score"),
                    elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                )
            return {"member": member_payload, "session": member_session, "policy_members": policy_members}
        finally:
            member_session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    def _refresh_process_progress_board(*, force: bool = False) -> None:
        if not compact_display or progress_board is None or not process_log_paths:
            return
        from tools.optimizer.outer_rolling_oos import read_optimizer_seed_progresses_from_log_paths

        latest = read_optimizer_seed_progresses_from_log_paths(process_log_paths.values())
        if not latest:
            return
        progress_map = {(1, int(member_index)): dict(progress) for member_index, progress in latest.items()}
        with progress_lock:
            progress_board.update_many(progress_map, force=force)

    if int(parallel_workers) <= 1:
        for member_index, seed in enumerate(seeds, start=1):
            result = _run_one_seed_member(member_index, int(seed))
            if result and result.get("member"):
                members.append(dict(result["member"]))
                _collect_policy_members(result.get("policy_members"))
                dashboard_session = result.get("session") or dashboard_session
    elif process_parallel_enabled:
        os.makedirs(process_log_dir, exist_ok=True)
        process_tasks = {}
        process_context = _build_nonrolling_single_fold_period_context(walk_forward_policy)
        for member_index, seed in enumerate(seeds, start=1):
            log_path = os.path.join(process_log_dir, f"seed_{int(member_index):02d}_{int(seed)}.log")
            process_log_paths[int(member_index)] = log_path
            process_tasks[int(member_index)] = {
                "member_index": int(member_index),
                "member_count": int(len(seeds)),
                "seed": int(seed),
                "walk_forward_policy": dict(walk_forward_policy),
                "selected_data_dir": str(selected_data_dir),
                "optimizer_required_min_rows": int(optimizer_required_min_rows),
                "objective_mode": str(objective_mode),
                "requested_trials": int(requested_trials),
                "db_name": None,
                "environ": dict(environ or {}),
                "log_path": log_path,
                "progress_context": {
                    "fold_idx": 1,
                    "fold_count": 1,
                    "oos_year": int(process_context.get("oos_year", 0) or 0),
                    "oos_period": str(process_context.get("oos_period") or ""),
                    "selection_start": str(process_context.get("selection_start") or ""),
                    "selection_end": str(process_context.get("selection_end") or ""),
                    "selection_period": str(process_context.get("selection_period") or ""),
                },
            }
        executor_kwargs, _start_method = get_process_pool_executor_kwargs()
        with ProcessPoolExecutor(max_workers=int(parallel_workers), **executor_kwargs) as executor:
            future_map = {
                executor.submit(_run_nonrolling_seed_ensemble_member_process_task, task): int(member_index)
                for member_index, task in process_tasks.items()
            }
            pending = set(future_map)
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                _refresh_process_progress_board(force=bool(done))
                for future in done:
                    member_index = int(future_map[future])
                    try:
                        result = future.result()
                    except Exception as exc:
                        tail = _tail_nonrolling_seed_log(process_log_paths.get(member_index, ""), max_lines=30)
                        detail = f"\n最後 seed log：\n{tail}" if tail else ""
                        raise RuntimeError(f"nonrolling seed process failed: member={member_index}/{len(seeds)} error={type(exc).__name__}: {exc}{detail}") from exc
                    if result and result.get("member"):
                        members.append(dict(result["member"]))
                        _collect_policy_members(result.get("policy_members"))
                    elif compact_display:
                        _emit_seed_progress_for_member(member_index, int(seeds[member_index - 1]),
                            stage="DONE",
                            status=f"seed {member_index}/{len(seeds)} seed={int(seeds[member_index - 1])} skipped",
                            elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                        )
            _refresh_process_progress_board(force=True)
        members.sort(key=lambda item: int(item.get("member_index", 0) or 0))
    else:
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
            future_map = {
                executor.submit(_run_one_seed_member, int(member_index), int(seed)): (int(member_index), int(seed))
                for member_index, seed in enumerate(seeds, start=1)
            }
            pending = set(future_map)
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    member_index, seed = future_map[future]
                    result = future.result()
                    if result and result.get("member"):
                        members.append(dict(result["member"]))
                        _collect_policy_members(result.get("policy_members"))
                        dashboard_session = result.get("session") or dashboard_session
                    elif compact_display:
                        _emit_seed_progress_for_member(member_index, int(seed),
                            stage="DONE",
                            status=f"seed {member_index}/{len(seeds)} seed={int(seed)} skipped",
                            elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                        )
        members.sort(key=lambda item: int(item.get("member_index", 0) or 0))

    if len(members) != len(seeds):
        if compact_display and progress_board is not None:
            progress_board.close()
        print(
            f"{C_YELLOW}ℹ️ 非 rolling random seed ensemble 未建立 candidate："
            f"可用 members={len(members)}/{len(seeds)}，不輸出不完整 ensemble。{C_RESET}"
        )
        return 0

    ensemble_payload, _ensemble_summary = _write_static_seed_ensemble_candidate(
        members=members,
        seeds=seeds,
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        dataset_label=dataset_label,
        selected_model_mode=selected_model_mode,
        trials_per_seed=int(requested_trials),
        policy_members_by_policy=policy_members_by_policy,
    )
    from tools.optimizer.callbacks import (
        print_optimizer_static_ensemble_console_dashboard,
        print_optimizer_static_ensemble_rolling_oos_table,
    )
    if compact_display and progress_board is not None:
        progress_board.close()
    if dashboard_session is None:
        dashboard_session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
        dashboard_session.load_raw_data(
            selected_data_dir,
            load_all_raw_data=load_all_raw_data,
            required_min_rows=optimizer_required_min_rows,
            verbose=False,
        )
    print_optimizer_static_ensemble_rolling_oos_table(
        dashboard_session,
        ensemble_payload=ensemble_payload,
    )
    print_optimizer_static_ensemble_console_dashboard(
        dashboard_session,
        ensemble_payload=ensemble_payload,
        seeds=seeds,
        milestone_title="🏆 ENSEMBLE 訓練結果",
        title="ENSEMBLE 績效與風險對比表",
        force=True,
    )
    promote_status = _promote_candidate_to_run_best()
    if promote_status != 0:
        return int(promote_status)
    print(f"{C_GREEN}✅ 非 rolling random seed ensemble 訓練完成｜members={len(members)}｜params={CANDIDATE_BEST_PARAMS_PATH}{C_RESET}")
    return 0


def _extract_cli_value(argv, option_name: str):
    args = [] if argv is None else list(argv)
    for idx in range(1, len(args)):
        raw_arg = str(args[idx]).strip()
        if raw_arg == option_name and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        if raw_arg.startswith(option_name + '='):
            return raw_arg.split('=', 1)[1].strip()
    return ''


def _has_cli_flag(argv, option_name: str) -> bool:
    args = [] if argv is None else list(argv)
    return any(str(arg).strip() == option_name for arg in args[1:])


def _resolve_cli_run_request(argv):
    from core.runtime_utils import parse_int_strict
    from tools.optimizer.benchmark import OPTIMIZER_TIMING_MODE_DEFAULT_TRIALS

    timing_mode = _has_cli_flag(argv, '--timing')
    outer_oos_mode = _has_cli_flag(argv, '--outer-oos')
    cli_trials_raw = _extract_cli_value(argv, '--trials')
    if outer_oos_mode:
        return {
            'timing_mode': timing_mode,
            'n_trials': int(cli_trials_raw) if str(cli_trials_raw or '').strip() else (OPTIMIZER_TIMING_MODE_DEFAULT_TRIALS if timing_mode else 0),
            'action': 'outer_rolling_oos',
            'source': 'CLI:--outer-oos+--timing' if timing_mode else 'CLI:--outer-oos',
        }
    if cli_trials_raw:
        cli_trials = parse_int_strict(cli_trials_raw, 'CLI 參數 --trials', min_value=0)
        if timing_mode and cli_trials <= 0:
            raise ValueError('--timing 模式要求 --trials >= 1。')
        return {
            'timing_mode': timing_mode,
            'n_trials': int(cli_trials),
            'action': 'train' if int(cli_trials) > 0 else 'export_candidate',
            'source': 'CLI:--timing+--trials' if timing_mode else 'CLI:--trials',
        }
    if timing_mode:
        return {
            'timing_mode': True,
            'n_trials': 3,
            'action': 'train',
            'source': 'CLI:--timing',
        }
    return None


def _prompt_optimizer_model_mode(default_model: str):
    default_normalized = str(default_model).strip().lower() or 'split'
    default_choice = '1' if default_normalized == 'split' else '2'
    print(f"{C_GRAY}ℹ️ 模式說明：split=固定 pre-deploy train 選參 + OOS 獨立驗證；full=全資料選參。{C_RESET}")
    choice = safe_prompt_choice(
        "👉 訓練模式：[1] split (預設)  [2] full : ",
        default_choice,
        ('1', '2'),
        'optimizer 模式',
    )
    return ('split', 'UI/MENU') if choice == '1' else ('full', 'UI/MENU')


def resolve_optimizer_model_mode(argv, environ, *, default_model: str = DEFAULT_OPTIMIZER_MODEL_MODE):
    cli_value = _extract_cli_value(argv, '--model')
    if cli_value:
        normalized = cli_value.strip().lower()
        source = 'CLI:--model'
    elif _has_cli_flag(argv, '--timing'):
        normalized = 'split'
        source = 'TIMING_DEFAULT:split'
    else:
        env_value = str((environ or {}).get('V16_OPTIMIZER_MODEL', '')).strip()
        if env_value:
            normalized = env_value.lower()
            source = 'ENV:V16_OPTIMIZER_MODEL'
        elif is_interactive_console():
            return _prompt_optimizer_model_mode(default_model)
        else:
            normalized = str(default_model).strip().lower() or 'split'
            source = 'DEFAULT'
    if normalized not in {'split', 'full'}:
        raise ValueError(f"optimizer 模式只接受 split 或 full，收到: {normalized}")
    return normalized, source


def main(argv=None, environ=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    validate_cli_args(argv, value_options=("--dataset", "--model", "--trials", "--outer-train-start", "--outer-first-oos", "--outer-last-oos", "--outer-first-oos-date", "--outer-last-oos-date", "--outer-window-mode", "--outer-train-window-years", "--outer-train-window-months", "--outer-oos-months"), flag_options=("--timing", "--outer-oos", "--yes"))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/optimizer/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full] [--model split|full] [--trials N] [--timing] [--outer-oos] [--outer-window-mode fixed|expanding] [--outer-train-window-months N] [--outer-oos-months N]")
        print("說明: split=固定 pre-deploy train 選參 + OOS 獨立驗證；full=全資料選參。可用 --trials N 直接指定訓練次數；可用 --timing 啟用 CLI 測時模式，預設跑 3 個 trials，亦可搭配 --trials N。未使用 --trials 時，仍維持既有互動選單 / ENV 行為。輸入 0 匯出 candidate_best，並同步輸出 retention 最大的 candidate_retention_best 與 val_score 最大的 candidate_val_score_best 作比較；輸入 P promote candidate；輸入 R 或 --outer-oos 執行 outer rolling monthly OOS test；outer rolling 可搭配 --timing 輸出分段耗時。正常完成訓練後會自動寫入 candidate_best、candidate_retention_best 與 candidate_val_score_best，並由 candidate_best 自動挑戰進版 run_best；若使用者中斷則不做。")
        return 0

    from core.data_utils import discover_unique_csv_inputs
    from strategies.breakout.search_space import get_breakout_optimizer_required_min_rows
    from tools.optimizer.prep import load_all_raw_data
    from tools.optimizer.runtime import (
        create_optimizer_study,
        ensure_export_only_db_not_empty,
        ensure_optimizer_db_usable,
        export_best_params_if_requested,
        maybe_print_history_best,
        print_resolved_trial_count,
        prompt_existing_db_policy,
        resolve_optimizer_single_fold_search_parallel_trials,
        resolve_training_session_export_policy,
        resolve_trial_count_or_exit,
    )
    from tools.optimizer.robustness import (
        build_local_min_score_best_trial_resolver,
        print_local_min_score_finalist_review,
        print_local_min_score_winner_summary,
        select_best_finalist_by_inner_validate_score,
        select_best_finalist_by_local_retention,
    )
    from tools.optimizer.session import close_study_storage
    from tools.optimizer.callbacks import print_optimizer_static_ensemble_console_dashboard
    from tools.optimizer.study_utils import (
        build_best_params_payload_from_trial,
        build_optimizer_db_file_path,
        is_qualified_trial_value,
        resolve_optimizer_seed,
        resolve_optimizer_trial_count,
    )
    from tools.optimizer.benchmark import (
        build_timing_db_file_path,
        build_timing_summary,
        print_timing_summary,
        write_timing_summary,
    )

    loaded_policy = load_walk_forward_policy(PROJECT_ROOT)
    try:
        selected_model_mode, model_mode_source = resolve_optimizer_model_mode(argv, environ)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    walk_forward_policy = build_optimizer_runtime_policy(loaded_policy, selected_model_mode)
    optimizer_required_min_rows = get_breakout_optimizer_required_min_rows()
    objective_mode = str(walk_forward_policy.get('objective_mode', 'split_train_romd'))
    session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
    best_trial_resolver = build_local_min_score_best_trial_resolver(session=session, objective_mode=objective_mode)

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(argv, environ, default=DEFAULT_DATASET_PROFILE)
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        dataset_label = get_dataset_profile_label(dataset_profile_key)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    try:
        cli_run_request = _resolve_cli_run_request(argv)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    timing_mode = bool(cli_run_request and cli_run_request.get("timing_mode"))
    db_file = build_timing_db_file_path(output_dir=OUTPUT_DIR, dataset_profile_key=dataset_profile_key, session_ts=session.session_ts) if timing_mode else build_optimizer_db_file_path(dataset_profile_key, MODELS_DIR)
    db_name = f"sqlite:///{db_file}"
    ensure_runtime_dirs()

    if cli_run_request is not None:
        session.n_trials = int(cli_run_request["n_trials"])
        session.run_action = str(cli_run_request.get("action", "train"))
        trial_source = str(cli_run_request.get("source", "CLI"))
        trial_count_exit = None
    else:
        trial_count_exit, trial_source = resolve_trial_count_or_exit(
            session,
            environ=environ,
            resolve_optimizer_trial_count=resolve_optimizer_trial_count,
            colors=COLORS,
        )
    if trial_count_exit is not None:
        return trial_count_exit
    if str(getattr(session, "run_action", "train")) == "outer_rolling_oos":
        from tools.optimizer.outer_rolling_oos import run_outer_rolling_oos
        try:
            optimizer_seed, seed_source = resolve_optimizer_seed(environ)
        except ValueError as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
        outer_timing_mode = bool(cli_run_request and cli_run_request.get("timing_mode"))
        if outer_timing_mode and optimizer_seed is None:
            optimizer_seed, seed_source = 42, 'TIMING_DEFAULT:42'
        return run_outer_rolling_oos(
            argv=argv,
            environ=environ,
            project_root=PROJECT_ROOT,
            output_dir=OUTPUT_DIR,
            base_policy=loaded_policy,
            selected_data_dir=selected_data_dir,
            dataset_label=dataset_label,
            load_all_raw_data=load_all_raw_data,
            optimizer_required_min_rows=optimizer_required_min_rows,
            build_optimizer_session=build_optimizer_session,
            create_optimizer_study=create_optimizer_study,
            ensure_study_effective_policy_compatible=_ensure_study_effective_policy_compatible,
            configure_optuna_logging=configure_optuna_logging,
            optimizer_seed=optimizer_seed,
            default_trials=int(getattr(session, "n_trials", 0) or OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT),
            timing_mode=outer_timing_mode,
        )
    if str(getattr(session, "run_action", "train")) == "promote_candidate":
        ensure_runtime_dirs()
        return _promote_candidate_to_run_best()

    try:
        optimizer_seed, seed_source = resolve_optimizer_seed(environ)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    if timing_mode and optimizer_seed is None:
        optimizer_seed, seed_source = 42, 'TIMING_DEFAULT:42'

    if not timing_mode and str(getattr(session, "run_action", "train")) == "train":
        ensemble_result = _run_nonrolling_random_seed_ensemble_training(
            environ=environ,
            selected_data_dir=selected_data_dir,
            dataset_profile_key=dataset_profile_key,
            dataset_label=dataset_label,
            selected_model_mode=selected_model_mode,
            load_all_raw_data=load_all_raw_data,
            optimizer_required_min_rows=optimizer_required_min_rows,
            objective_mode=objective_mode,
            walk_forward_policy=walk_forward_policy,
            requested_trials=int(session.n_trials),
            build_optimizer_session=build_optimizer_session,
            create_optimizer_study=create_optimizer_study,
            ensure_study_effective_policy_compatible=_ensure_study_effective_policy_compatible,
            configure_optuna_logging=configure_optuna_logging,
            print_local_min_score_finalist_review=print_local_min_score_finalist_review,
            print_local_min_score_winner_summary=print_local_min_score_winner_summary,
            close_study_storage=close_study_storage,
            is_qualified_trial_value=is_qualified_trial_value,
            resolve_optimizer_single_fold_search_parallel_trials=resolve_optimizer_single_fold_search_parallel_trials,
            build_best_params_payload_from_trial=build_best_params_payload_from_trial,
        )
        if ensemble_result is not None:
            return int(ensemble_result)

    if session.n_trials == 0:
        if not os.path.exists(db_file):
            print(f"{C_RED}❌ 記憶庫不存在，無法匯出: {db_file}；非互動模式預設 trial 數為 0，若要在乾淨 repo 建立新記憶庫，請先設定 V16_OPTIMIZER_TRIALS>0、使用 --trials N，或先完成一次訓練。{C_RESET}", file=sys.stderr)
            return 1
        try:
            ensure_optimizer_db_usable(db_file)
            ensure_export_only_db_not_empty(db_file)
            study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=("random" if timing_mode else "tpe"))
            _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
        except RuntimeError as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
        try:
            if not os.path.isdir(selected_data_dir):
                raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
            csv_inputs, _ = discover_unique_csv_inputs(selected_data_dir)
            if not csv_inputs:
                raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir))
            session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data,
                required_min_rows=optimizer_required_min_rows,
                verbose=bool(is_optimizer_nonrolling_train_result_table_enabled()),
            )
            finalists, best_trial = print_local_min_score_finalist_review(
                study,
                session=session,
                objective_mode=objective_mode,
                colors=COLORS,
                winner_trial=None,
                emit_table=bool(is_optimizer_nonrolling_train_result_table_enabled()),
            )
            if best_trial is None or not is_qualified_trial_value(best_trial.value):
                print(f"{C_YELLOW}ℹ️ 匯出模式完成，但目前尚無通過 local_min_score gate 的 winner。{C_RESET}")
                return 0

            print_local_min_score_winner_summary(
                winner_trial=best_trial,
                session=session,
                colors=COLORS,
            )
            exported = _export_selected_candidate_artifacts(
                study=study,
                finalists=finalists,
                selected_trial=best_trial,
                params_path=CANDIDATE_BEST_PARAMS_PATH,
                summary_path=CANDIDATE_BEST_SUMMARY_PATH,
                fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                colors=COLORS,
                objective_mode=objective_mode,
                walk_forward_policy=walk_forward_policy,
                action_label="export_candidate",
                selection_rule=_resolve_candidate_best_selection_rule(),
                compare_only=False,
                artifact_label="candidate_best",
                export_best_params_if_requested=export_best_params_if_requested,
                is_qualified_trial_value=is_qualified_trial_value,
            )
            if not exported:
                return 1
            retention_best_finalist = select_best_finalist_by_local_retention(finalists)
            retention_best_trial = None if retention_best_finalist is None else retention_best_finalist["trial"]
            _export_selected_candidate_artifacts(
                study=study,
                finalists=finalists,
                selected_trial=retention_best_trial,
                params_path=CANDIDATE_RETENTION_BEST_PARAMS_PATH,
                summary_path=CANDIDATE_RETENTION_BEST_SUMMARY_PATH,
                fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                colors=COLORS,
                objective_mode=objective_mode,
                walk_forward_policy=walk_forward_policy,
                action_label="export_candidate",
                selection_rule="max_retention_compare",
                compare_only=True,
                artifact_label="candidate_retention_best",
                export_best_params_if_requested=export_best_params_if_requested,
                is_qualified_trial_value=is_qualified_trial_value,
            )
            if bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED):
                val_score_best_finalist = select_best_finalist_by_inner_validate_score(finalists)
                val_score_best_trial = None if val_score_best_finalist is None else val_score_best_finalist["trial"]
                _export_selected_candidate_artifacts(
                    study=study,
                    finalists=finalists,
                    selected_trial=val_score_best_trial,
                    params_path=CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH,
                    summary_path=CANDIDATE_VAL_SCORE_BEST_SUMMARY_PATH,
                    fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                    colors=COLORS,
                    objective_mode=objective_mode,
                    walk_forward_policy=walk_forward_policy,
                    action_label="export_candidate",
                    selection_rule="max_inner_validate_score_compare",
                    compare_only=True,
                    artifact_label="candidate_val_score_best",
                    export_best_params_if_requested=export_best_params_if_requested,
                    is_qualified_trial_value=is_qualified_trial_value,
                )
            if selected_model_mode == 'split':
                return finalize_best_trial_outputs(
                    session=session,
                    study=study,
                    best_trial_resolver=(lambda _study, _best_trial=best_trial: _best_trial),
                    dataset_label=dataset_label,
                    db_file=db_file,
                    walk_forward_policy=walk_forward_policy,
                )
            return 0
        finally:
            session.close_trial_prep_executor()
            close_study_storage(study)

    try:
        if not os.path.isdir(selected_data_dir):
            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
    except FileNotFoundError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    csv_inputs, _ = discover_unique_csv_inputs(selected_data_dir)
    if not csv_inputs:
        print(f"{C_RED}❌ {build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir)}{C_RESET}", file=sys.stderr)
        return 1

    try:
        if os.path.exists(db_file):
            ensure_optimizer_db_usable(db_file)
    except RuntimeError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    configure_optuna_logging()
    if not timing_mode:
        print_resolved_trial_count(session, trial_source=trial_source, colors=COLORS)
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端投資組合 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    selection_start_year = int(walk_forward_policy.get('selection_start_year', walk_forward_policy['train_start_year']))
    search_train_end_year = int(walk_forward_policy['search_train_end_year'])
    oos_start_year = walk_forward_policy.get('oos_start_year')
    if selected_model_mode == 'split':
        inner_scope_text = ""
        if bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED):
            inner_validate_year = int(search_train_end_year)
            inner_train_end_year = int(inner_validate_year) - 1
            inner_scope_text = f" | inner_train={selection_start_year}~{inner_train_end_year} | inner_val={inner_validate_year}"
        scope_text = (
            f"selection={selection_start_year}~{search_train_end_year}{inner_scope_text} | "
            f"oos={oos_start_year if oos_start_year is not None else search_train_end_year + 1}~latest"
        )
    else:
        scope_text = 'selection=all_data | oos=disabled'
    inline_override_fields = list(walk_forward_policy.get("inline_override_fields", []) or [])
    override_text = "" if not inline_override_fields else f" | override={','.join(inline_override_fields)}"
    print(
        f"{C_GRAY}📌 設定｜資料集={dataset_label}｜模式={selected_model_mode}{override_text}｜"
        f"{scope_text}｜trials={session.n_trials}{C_RESET}"
    )
    seed_text = str(optimizer_seed) if optimizer_seed is not None else "未設定"
    print(f"{C_GRAY}🎲 Optimizer seed: {seed_text} | 來源: {seed_source}{C_RESET}")

    try:
        if not timing_mode:
            prompt_existing_db_policy(db_file, COLORS)
        if os.path.exists(db_file):
            ensure_optimizer_db_usable(db_file)
        sampler_kind = "random" if timing_mode else "tpe"
        study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=sampler_kind)
        _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
    except (ValueError, RuntimeError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    session.timing_mode = timing_mode
    session.disable_milestone_dashboard = not bool(is_optimizer_nonrolling_train_result_table_enabled())

    overall_started_at = time.perf_counter()
    raw_data_load_sec = 0.0
    startup_total_sec = 0.0
    optimize_wall_sec = 0.0
    try:
        raw_data_load_started_at = time.perf_counter()
        session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data,
                required_min_rows=optimizer_required_min_rows,
                verbose=bool(is_optimizer_nonrolling_train_result_table_enabled()),
            )
        raw_data_load_sec = max(0.0, time.perf_counter() - raw_data_load_started_at)

        maybe_print_history_best(
            study,
            fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
            train_enable_rotation=TRAIN_ENABLE_ROTATION,
            train_max_positions=TRAIN_MAX_POSITIONS,
            colors=COLORS,
            best_trial_resolver=session.get_best_completed_trial_or_none,
            session=session,
        )

        session.profile_recorder.init_output_files()

        startup_total_sec = max(0.0, time.perf_counter() - overall_started_at)
        print(f"{C_CYAN}⏱️ 前置完成：raw_data_load={raw_data_load_sec:.3f}s | startup_total={startup_total_sec:.3f}s | 即將進入第 1 輪{C_RESET}")
        print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
        session.profile_recorder.mark_run_started()
        optimize_started_at = time.perf_counter()
        training_interrupted = False
        try:
            search_parallel_trials = resolve_optimizer_single_fold_search_parallel_trials(
                environ,
                sampler_kind=sampler_kind,
            )
            study.optimize(
                session.objective,
                n_trials=session.n_trials,
                n_jobs=int(search_parallel_trials),
                callbacks=[session.monitoring_callback],
            )
        except KeyboardInterrupt:
            training_interrupted = True
            print(f"\n{C_YELLOW}⚠️ 使用者中斷訓練流程。{C_RESET}")

        optimize_wall_sec = max(0.0, time.perf_counter() - optimize_started_at)
        print()
        _print_profile_summary_compatible(session.profile_recorder, emit_console=not timing_mode)
        session.print_optimizer_prep_summary()
        should_export, export_policy = resolve_training_session_export_policy(
            requested_n_trials=session.n_trials,
            completed_session_trials=session.current_session_trial,
            interrupted=training_interrupted,
        )
        if timing_mode:
            timing_payload = build_timing_summary(
                session=session,
                dataset_label=dataset_label,
                dataset_profile_key=dataset_profile_key,
                selected_model_mode=selected_model_mode,
                db_file=db_file,
                raw_data_load_sec=raw_data_load_sec,
                startup_total_sec=startup_total_sec,
                optimize_wall_sec=optimize_wall_sec,
                total_wall_sec=max(0.0, time.perf_counter() - overall_started_at),
                optimizer_seed=optimizer_seed,
                timing_sampler_kind=sampler_kind,
            )
            timing_summary_path = write_timing_summary(
                output_dir=OUTPUT_DIR,
                session_ts=session.session_ts,
                payload=timing_payload,
            )
            print_timing_summary(payload=timing_payload)
        elif should_export:
            finalists, best_trial = print_local_min_score_finalist_review(
                study,
                session=session,
                objective_mode=objective_mode,
                colors=COLORS,
                winner_trial=None,
                emit_table=bool(is_optimizer_nonrolling_train_result_table_enabled()),
            )
            if best_trial is not None and is_qualified_trial_value(best_trial.value):
                print_local_min_score_winner_summary(
                    winner_trial=best_trial,
                    session=session,
                    colors=COLORS,
                )
                export_status = export_best_params_if_requested(
                    study,
                    best_params_path=CANDIDATE_BEST_PARAMS_PATH,
                    fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                    colors=COLORS,
                    best_trial_resolver=(lambda _study, _best_trial=best_trial: _best_trial),
                    suppress_success_message=True,
                )
                if export_status != 0:
                    return 1
                finalist_entry = _find_finalist_entry(finalists, best_trial)
                candidate_summary = _build_best_summary_payload(
                    winner_trial=best_trial,
                    finalist_entry=finalist_entry,
                    objective_mode=objective_mode,
                    walk_forward_policy=walk_forward_policy,
                    action_label="train",
                    selection_rule=_resolve_candidate_best_selection_rule(),
                    compare_only=False,
                )
                _write_json_file(CANDIDATE_BEST_SUMMARY_PATH, candidate_summary)
                print(f"{C_GREEN}💾 candidate_best 已寫入：{CANDIDATE_BEST_PARAMS_PATH}{C_RESET}")
                retention_best_finalist = select_best_finalist_by_local_retention(finalists)
                retention_best_trial = None if retention_best_finalist is None else retention_best_finalist["trial"]
                _export_selected_candidate_artifacts(
                    study=study,
                    finalists=finalists,
                    selected_trial=retention_best_trial,
                    params_path=CANDIDATE_RETENTION_BEST_PARAMS_PATH,
                    summary_path=CANDIDATE_RETENTION_BEST_SUMMARY_PATH,
                    fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                    colors=COLORS,
                    objective_mode=objective_mode,
                    walk_forward_policy=walk_forward_policy,
                    action_label="train",
                    selection_rule="max_retention_compare",
                    compare_only=True,
                    artifact_label="candidate_retention_best",
                    export_best_params_if_requested=export_best_params_if_requested,
                    is_qualified_trial_value=is_qualified_trial_value,
                )
                if bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED):
                    val_score_best_finalist = select_best_finalist_by_inner_validate_score(finalists)
                    val_score_best_trial = None if val_score_best_finalist is None else val_score_best_finalist["trial"]
                    _export_selected_candidate_artifacts(
                        study=study,
                        finalists=finalists,
                        selected_trial=val_score_best_trial,
                        params_path=CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH,
                        summary_path=CANDIDATE_VAL_SCORE_BEST_SUMMARY_PATH,
                        fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                        colors=COLORS,
                        objective_mode=objective_mode,
                        walk_forward_policy=walk_forward_policy,
                        action_label="train",
                        selection_rule="max_inner_validate_score_compare",
                        compare_only=True,
                        artifact_label="candidate_val_score_best",
                        export_best_params_if_requested=export_best_params_if_requested,
                        is_qualified_trial_value=is_qualified_trial_value,
                    )
                _promote_candidate_to_run_best()
                if selected_model_mode == 'split':
                    finalize_best_trial_outputs(
                        session=session,
                        study=study,
                        best_trial_resolver=(lambda _study, _best_trial=best_trial: _best_trial),
                        dataset_label=dataset_label,
                        db_file=db_file,
                        walk_forward_policy=walk_forward_policy,
                    )
            else:
                print(f"{C_YELLOW}ℹ️ 訓練完成，但目前尚無通過 local_min_score gate 的 winner。{C_RESET}")
        elif export_policy == "interrupted_before_target":
            print(
                f"{C_YELLOW}ℹ️ 本輪由使用者中斷，已完成 {session.current_session_trial}/{session.n_trials}；"
                f"不自動覆寫 candidate_best 或 run_best。{C_RESET}"
            )
        print(f"\n{C_YELLOW}🛑 訓練階段結束或已中斷。{C_RESET}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    finally:
        session.close_trial_prep_executor()
        close_study_storage(study)


if __name__ == "__main__":
    run_cli_entrypoint(main)
