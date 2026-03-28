import os
import sys
import time
import warnings

import optuna

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data_utils import get_required_min_rows_from_high_len
from core.dataset_profiles import (
    DEFAULT_DATASET_PROFILE,
    get_dataset_dir,
    get_dataset_profile_label,
    resolve_dataset_profile_from_cli_env,
)
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.runtime_utils import has_help_flag
from tools.optimizer.prep import load_all_raw_data
from tools.optimizer.profile import OptimizerProfileRecorder
from tools.optimizer.runtime import (
    create_optimizer_study,
    export_best_params_if_requested,
    maybe_print_history_best,
    print_trial_count_or_exit,
    prompt_existing_db_policy,
)
from tools.optimizer.session import OptimizerSession, close_study_storage
from tools.optimizer.study_utils import (
    build_optimizer_db_file_path,
    build_optimizer_trial_params,
    get_best_completed_trial_or_none,
    resolve_optimizer_tp_percent,
    resolve_optimizer_trial_count,
)

warnings.simplefilter("default")
warnings.filterwarnings("once", category=FutureWarning, module=r"optuna(\..*)?$")
warnings.filterwarnings("once", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params.json")
TRAIN_MAX_POSITIONS = 10
TRAIN_START_YEAR = 2015
TRAIN_ENABLE_ROTATION = False
DEFAULT_OPTIMIZER_MAX_WORKERS = min(6, max(1, (os.cpu_count() or 1) // 2))
OPTIMIZER_HIGH_LEN_MIN = 40
OPTIMIZER_HIGH_LEN_MAX = 250
OPTIMIZER_HIGH_LEN_STEP = 5
OPTIMIZER_REQUIRED_MIN_ROWS = get_required_min_rows_from_high_len(OPTIMIZER_HIGH_LEN_MAX)
OPTIMIZER_FIXED_TP_PERCENT = None
OPTIMIZER_SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
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
    return OptimizerSession(
        output_dir=OUTPUT_DIR,
        session_ts=OPTIMIZER_SESSION_TS,
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
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    if has_help_flag(argv):
        print("用法: python apps/ml_optimizer.py [--dataset reduced|full]")
        print("說明: 非互動模式預設訓練次數為 0；可用環境變數 V16_OPTIMIZER_TRIALS 指定 trial 數。")
        return 0
    session = build_optimizer_session()

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            environ,
            default=DEFAULT_DATASET_PROFILE,
        )
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        dataset_label = get_dataset_profile_label(dataset_profile_key)
        if not os.path.isdir(selected_data_dir):
            raise FileNotFoundError(f"找不到資料夾 {selected_data_dir}，請先執行 apps/smart_downloader.py！")
    except (ValueError, FileNotFoundError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端 (End-to-End) 投資組合極速 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")

    db_file = build_optimizer_db_file_path(dataset_profile_key, MODELS_DIR)
    db_name = f"sqlite:///{db_file}"

    ensure_runtime_dirs()
    try:
        session.load_raw_data(
            selected_data_dir,
            load_all_raw_data=load_all_raw_data,
            required_min_rows=OPTIMIZER_REQUIRED_MIN_ROWS,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    print(
        f"{C_GRAY}📁 使用資料集: {dataset_label} | "
        f"來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}"
    )
    print(f"{C_GRAY}🗃️ Optimizer 記憶庫: {db_file}{C_RESET}")
    session.profile_recorder.init_output_files()
    if session.profile_recorder.enabled:
        print(f"{C_GRAY}🧪 Profiling 已啟用，trial 明細將寫入: {session.profile_recorder.csv_path}{C_RESET}")

    try:
        prompt_existing_db_policy(db_file, COLORS)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    trial_count_exit = print_trial_count_or_exit(
        session,
        environ=environ,
        resolve_optimizer_trial_count=resolve_optimizer_trial_count,
        colors=COLORS,
    )
    if trial_count_exit is not None:
        return trial_count_exit

    study = create_optimizer_study(db_name)
    try:
        maybe_print_history_best(
            study,
            fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
            train_enable_rotation=TRAIN_ENABLE_ROTATION,
            train_max_positions=TRAIN_MAX_POSITIONS,
            colors=COLORS,
        )

        if session.n_trials == 0:
            return export_best_params_if_requested(
                study,
                best_params_path=BEST_PARAMS_PATH,
                fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                colors=COLORS,
            )

        print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
        try:
            study.optimize(session.objective, n_trials=session.n_trials, n_jobs=1, callbacks=[session.monitoring_callback])
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}⚠️ 使用者中斷訓練流程。{C_RESET}")

        print()
        session.profile_recorder.print_summary()
        session.print_optimizer_prep_summary()
        print(f"\n{C_YELLOW}🛑 訓練階段結束或已中斷。{C_RESET}")
        return 0
    finally:
        close_study_storage(study)


if __name__ == "__main__":
    raise SystemExit(main())
