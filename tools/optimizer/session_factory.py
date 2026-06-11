import os

from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.output_paths import build_output_dir
from core.runtime_utils import get_taipei_now
from core.walk_forward_policy import build_optimizer_effective_policy_fingerprint
from config.training_policy import OPTIMIZER_FIXED_TP_PERCENT


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "ml_optimizer")
TRAIN_MAX_POSITIONS = 10
TRAIN_ENABLE_ROTATION = False
DEFAULT_OPTIMIZER_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1))) if os.name == "nt" else min(6, max(1, (os.cpu_count() or 1) // 2))
ENABLE_OPTIMIZER_PROFILING = True
ENABLE_PROFILE_CONSOLE_PRINT = False
PROFILE_PRINT_EVERY_N_TRIALS = 1

COLORS = {
    "cyan": C_CYAN,
    "gray": C_GRAY,
    "green": C_GREEN,
    "red": C_RED,
    "reset": C_RESET,
    "yellow": C_YELLOW,
}


def configure_optuna_logging():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def ensure_study_effective_policy_compatible(*, study, walk_forward_policy: dict):
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
