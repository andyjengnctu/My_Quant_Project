import os
import sys
import warnings

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
from core.model_paths import resolve_best_params_path, resolve_models_dir
from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, get_taipei_now, has_help_flag, resolve_cli_program_name, validate_cli_args
from core.output_paths import build_output_dir
from config.training_policy import OPTIMIZER_FIXED_TP_PERCENT

warnings.simplefilter("default")
warnings.filterwarnings("once", category=FutureWarning, module=r"optuna(\..*)?$")
warnings.filterwarnings("once", category=RuntimeWarning)


def configure_optuna_logging():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)


OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "ml_optimizer")
MODELS_DIR = resolve_models_dir(PROJECT_ROOT)
BEST_PARAMS_PATH = resolve_best_params_path(PROJECT_ROOT)
TRAIN_MAX_POSITIONS = 10
TRAIN_START_YEAR = 2015
TRAIN_ENABLE_ROTATION = False
DEFAULT_OPTIMIZER_MAX_WORKERS = min(6, max(1, (os.cpu_count() or 1) // 2))
OPTIMIZER_HIGH_LEN_MIN = 40
OPTIMIZER_HIGH_LEN_MAX = 250
OPTIMIZER_HIGH_LEN_STEP = 5
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


def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def build_optimizer_session():
    session_ts = get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    from tools.optimizer.profile import OptimizerProfileRecorder
    from tools.optimizer.session import OptimizerSession
    from tools.optimizer.study_utils import (
        build_optimizer_trial_params,
        get_best_completed_trial_or_none,
        resolve_optimizer_tp_percent,
    )

    return OptimizerSession(
        output_dir=OUTPUT_DIR,
        session_ts=session_ts,
        profile_recorder_cls=OptimizerProfileRecorder,
        build_optimizer_trial_params=build_optimizer_trial_params,
        get_best_completed_trial_or_none=get_best_completed_trial_or_none,
        resolve_optimizer_tp_percent=resolve_optimizer_tp_percent,
        print_strategy_dashboard=print_strategy_dashboard,
        colors=COLORS,
        optimizer_fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
        train_max_positions=TRAIN_MAX_POSITIONS,
        train_start_year=TRAIN_START_YEAR,
        train_enable_rotation=TRAIN_ENABLE_ROTATION,
        optimizer_high_len_min=OPTIMIZER_HIGH_LEN_MIN,
        optimizer_high_len_max=OPTIMIZER_HIGH_LEN_MAX,
        optimizer_high_len_step=OPTIMIZER_HIGH_LEN_STEP,
        default_max_workers=DEFAULT_OPTIMIZER_MAX_WORKERS,
        enable_optimizer_profiling=ENABLE_OPTIMIZER_PROFILING,
        enable_profile_console_print=ENABLE_PROFILE_CONSOLE_PRINT,
        profile_print_every_n_trials=PROFILE_PRINT_EVERY_N_TRIALS,
    )


def main(argv=None, environ=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/optimizer/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 預設資料集為完整；非互動模式預設訓練次數為 0；可用環境變數 V16_OPTIMIZER_TRIALS 指定 trial 數、V16_OPTIMIZER_SEED 指定固定 seed；只有完成指定訓練次數或輸入 0 走匯出模式時，才會更新 best_params.json。")
        return 0
    from core.data_utils import discover_unique_csv_inputs, get_required_min_rows_from_high_len
    from tools.optimizer.prep import load_all_raw_data
    from tools.optimizer.runtime import (
        create_optimizer_study,
        ensure_export_only_db_not_empty,
        ensure_optimizer_db_usable,
        export_best_params_if_requested,
        maybe_print_history_best,
        print_resolved_trial_count,
        resolve_training_session_export_policy,
        prompt_existing_db_policy,
        resolve_trial_count_or_exit,
    )
    from tools.optimizer.session import close_study_storage
    from tools.optimizer.study_utils import build_optimizer_db_file_path, get_best_completed_trial_or_none, is_qualified_trial_value, resolve_optimizer_seed, resolve_optimizer_trial_count

    optimizer_required_min_rows = get_required_min_rows_from_high_len(OPTIMIZER_HIGH_LEN_MAX)
    session = build_optimizer_session()

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            environ,
            default=DEFAULT_DATASET_PROFILE,
        )
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        dataset_label = get_dataset_profile_label(dataset_profile_key)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    db_file = build_optimizer_db_file_path(dataset_profile_key, MODELS_DIR)
    db_name = f"sqlite:///{db_file}"
    ensure_runtime_dirs()

    trial_count_exit, trial_source = resolve_trial_count_or_exit(
        session,
        environ=environ,
        resolve_optimizer_trial_count=resolve_optimizer_trial_count,
        colors=COLORS,
    )
    if trial_count_exit is not None:
        return trial_count_exit

    try:
        optimizer_seed, seed_source = resolve_optimizer_seed(environ)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    if session.n_trials == 0:
        if not os.path.exists(db_file):
            print(f"{C_RED}❌ 記憶庫不存在，無法匯出: {db_file}{C_RESET}", file=sys.stderr)
            return 1
        try:
            ensure_optimizer_db_usable(db_file)
            ensure_export_only_db_not_empty(db_file)
            study = create_optimizer_study(db_name, seed=optimizer_seed)
        except RuntimeError as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
        try:
            return export_best_params_if_requested(
                study,
                best_params_path=BEST_PARAMS_PATH,
                fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                colors=COLORS,
            )
        finally:
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
    print_resolved_trial_count(session, trial_source=trial_source, colors=COLORS)
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端 (End-to-End) 投資組合極速 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"{C_GRAY}📁 使用資料集: {dataset_label} | "
        f"來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}"
    )
    print(f"{C_GRAY}🗃️ Optimizer 記憶庫: {db_file}{C_RESET}")
    print(f"{C_GRAY}🎲 Optimizer seed: {optimizer_seed if optimizer_seed is not None else '未設定'} | 來源: {seed_source}{C_RESET}")

    try:
        prompt_existing_db_policy(db_file, COLORS)
        if os.path.exists(db_file):
            ensure_optimizer_db_usable(db_file)
        study = create_optimizer_study(db_name, seed=optimizer_seed)
    except (ValueError, RuntimeError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    try:
        maybe_print_history_best(
            study,
            fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
            train_enable_rotation=TRAIN_ENABLE_ROTATION,
            train_max_positions=TRAIN_MAX_POSITIONS,
            colors=COLORS,
        )

        session.load_raw_data(
            selected_data_dir,
            load_all_raw_data=load_all_raw_data,
            required_min_rows=optimizer_required_min_rows,
        )

        session.profile_recorder.init_output_files()
        if session.profile_recorder.enabled:
            print(f"{C_GRAY}🧪 Profiling 已啟用，trial 明細將寫入: {session.profile_recorder.csv_path}{C_RESET}")

        print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
        training_interrupted = False
        try:
            study.optimize(session.objective, n_trials=session.n_trials, n_jobs=1, callbacks=[session.monitoring_callback])
        except KeyboardInterrupt:
            training_interrupted = True
            print(f"\n{C_YELLOW}⚠️ 使用者中斷訓練流程。{C_RESET}")

        print()
        session.profile_recorder.print_summary()
        session.print_optimizer_prep_summary()
        should_export, export_policy = resolve_training_session_export_policy(
            requested_n_trials=session.n_trials,
            completed_session_trials=session.current_session_trial,
            interrupted=training_interrupted,
        )
        if should_export:
            best_trial = get_best_completed_trial_or_none(study)
            if best_trial is not None and is_qualified_trial_value(best_trial.value):
                export_status = export_best_params_if_requested(
                    study,
                    best_params_path=BEST_PARAMS_PATH,
                    fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                    colors=COLORS,
                )
                if export_status != 0:
                    return 1
        elif export_policy == "interrupted_before_target":
            print(
                f"{C_YELLOW}ℹ️ 本輪僅完成 {session.current_session_trial}/{session.n_trials} 次，"
                f"依規則不自動覆寫 {BEST_PARAMS_PATH}。{C_RESET}"
            )
        print(f"\n{C_YELLOW}🛑 訓練階段結束或已中斷。{C_RESET}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    finally:
        close_study_storage(study)



if __name__ == "__main__":
    run_cli_entrypoint(main)
