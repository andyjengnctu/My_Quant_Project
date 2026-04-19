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
from core.model_paths import resolve_models_dir, resolve_run_best_params_path
from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, get_taipei_now, has_help_flag, resolve_cli_program_name, validate_cli_args, is_interactive_console, safe_prompt_choice
from core.output_paths import build_output_dir
from core.walk_forward_policy import build_optimizer_runtime_policy, load_walk_forward_policy
from config.training_policy import DEFAULT_OPTIMIZER_MODEL_MODE, OPTIMIZER_FIXED_TP_PERCENT

warnings.simplefilter("default")
warnings.filterwarnings("once", category=FutureWarning, module=r"optuna(\..*)?$")
warnings.filterwarnings("once", category=RuntimeWarning)


def configure_optuna_logging():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)


OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "ml_optimizer")
MODELS_DIR = resolve_models_dir(PROJECT_ROOT)
RUN_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "run_best_params.json")
DEFAULT_WALK_FORWARD_POLICY = load_walk_forward_policy(PROJECT_ROOT)
TRAIN_MAX_POSITIONS = 10
TRAIN_ENABLE_ROTATION = False
DEFAULT_OPTIMIZER_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1))) if os.name == "nt" else min(6, max(1, (os.cpu_count() or 1) // 2))
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
        optimizer_high_len_min=OPTIMIZER_HIGH_LEN_MIN,
        optimizer_high_len_max=OPTIMIZER_HIGH_LEN_MAX,
        optimizer_high_len_step=OPTIMIZER_HIGH_LEN_STEP,
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
    from tools.optimizer.walk_forward import build_test_period_metrics

    print(f"{C_GREEN}🧭 已輸出 OOS 期間報表：{report_paths['md_path']}{C_RESET}")
    total_metrics = build_test_period_metrics(report)
    print(
        f"{C_GRAY}   OOS 區間: {total_metrics.get('oos_start', '')} ~ {total_metrics.get('oos_end', '')} | "
        f"RoMD: {float(total_metrics.get('test_score_romd', 0.0)):.3f} | "
        f"最大回撤: {float(total_metrics.get('max_drawdown_pct', 0.0)):.2f}%{C_RESET}"
    )
    upgrade_gate = dict(report.get("upgrade_gate") or {})
    gate_status = str(upgrade_gate.get("status", "fail")).upper()
    gate_color = C_GREEN if gate_status == "PASS" else C_RED
    print(f"{gate_color}   OOS 可用性檢查: {gate_status} | {upgrade_gate.get('recommendation', 'N/A')}{C_RESET}")


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


def _extract_cli_value(argv, option_name: str):
    args = [] if argv is None else list(argv)
    for idx in range(1, len(args)):
        raw_arg = str(args[idx]).strip()
        if raw_arg == option_name and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        if raw_arg.startswith(option_name + '='):
            return raw_arg.split('=', 1)[1].strip()
    return ''


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
    validate_cli_args(argv, value_options=("--dataset", "--model"))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/optimizer/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full] [--model split|full]")
        print("說明: split=固定 pre-deploy train 選參 + OOS 獨立驗證；full=全資料選參。兩者都只輸出 run_best。")
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
        prompt_existing_db_policy,
        resolve_training_session_export_policy,
        resolve_trial_count_or_exit,
    )
    from tools.optimizer.robustness import (
        build_local_min_score_best_trial_resolver,
        print_local_min_score_finalist_review,
        print_local_min_score_winner_summary,
    )
    from tools.optimizer.session import close_study_storage
    from tools.optimizer.study_utils import (
        build_optimizer_db_file_path,
        is_qualified_trial_value,
        resolve_optimizer_seed,
        resolve_optimizer_trial_count,
    )

    loaded_policy = load_walk_forward_policy(PROJECT_ROOT)
    try:
        selected_model_mode, model_mode_source = resolve_optimizer_model_mode(argv, environ)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    walk_forward_policy = build_optimizer_runtime_policy(loaded_policy, selected_model_mode)
    optimizer_required_min_rows = get_required_min_rows_from_high_len(OPTIMIZER_HIGH_LEN_MAX)
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
            print(f"{C_RED}❌ 記憶庫不存在，無法匯出: {db_file}；非互動模式預設 trial 數為 0，若要在乾淨 repo 建立新記憶庫，請先設定 V16_OPTIMIZER_TRIALS>0 或先完成一次訓練。{C_RESET}", file=sys.stderr)
            return 1
        try:
            ensure_optimizer_db_usable(db_file)
            ensure_export_only_db_not_empty(db_file)
            study = create_optimizer_study(db_name, seed=optimizer_seed)
        except RuntimeError as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
        try:
            if not os.path.isdir(selected_data_dir):
                raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
            csv_inputs, _ = discover_unique_csv_inputs(selected_data_dir)
            if not csv_inputs:
                raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir))
            session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
            export_status = export_best_params_if_requested(
                study,
                best_params_path=RUN_BEST_PARAMS_PATH,
                fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                colors=COLORS,
                best_trial_resolver=best_trial_resolver,
            )
            if export_status != 0:
                return export_status
            if selected_model_mode == 'split':
                return finalize_best_trial_outputs(
                    session=session,
                    study=study,
                    best_trial_resolver=best_trial_resolver,
                    dataset_label=dataset_label,
                    db_file=db_file,
                    walk_forward_policy=walk_forward_policy,
                )
            print(f"{C_GREEN}🧭 全資料模式僅匯出 run_best，不額外產出 OOS 報表。{C_RESET}")
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
    print_resolved_trial_count(session, trial_source=trial_source, colors=COLORS)
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端投資組合 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"{C_GRAY}📁 使用資料集: {dataset_label} | 來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}")
    print(f"{C_GRAY}🗃️ Optimizer 記憶庫: {db_file}{C_RESET}")
    print(f"{C_GRAY}🎲 Optimizer seed: {optimizer_seed if optimizer_seed is not None else '未設定'} | 來源: {seed_source}{C_RESET}")
    selection_start_year = int(walk_forward_policy.get('selection_start_year', walk_forward_policy['train_start_year']))
    search_train_end_year = int(walk_forward_policy['search_train_end_year'])
    oos_start_year = walk_forward_policy.get('oos_start_year')
    if selected_model_mode == 'split':
        scope_text = (
            f"selection={selection_start_year}~{search_train_end_year} | "
            f"oos={oos_start_year if oos_start_year is not None else search_train_end_year + 1}~latest"
        )
    else:
        scope_text = 'selection=all_data | oos=disabled'
    print(
        f"{C_GRAY}🧭 訓練模式: {selected_model_mode} | 來源: {model_mode_source} | "
        f"設定: {walk_forward_policy.get('policy_path', 'config/training_policy.py')} | {scope_text}{C_RESET}"
    )
    print(f"{C_GRAY}Train/Test policy: {scope_text}{C_RESET}")

    try:
        prompt_existing_db_policy(db_file, COLORS)
        if os.path.exists(db_file):
            ensure_optimizer_db_usable(db_file)
        study = create_optimizer_study(db_name, seed=optimizer_seed)
    except (ValueError, RuntimeError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    try:
        session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)

        maybe_print_history_best(
            study,
            fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
            train_enable_rotation=TRAIN_ENABLE_ROTATION,
            train_max_positions=TRAIN_MAX_POSITIONS,
            colors=COLORS,
            best_trial_resolver=best_trial_resolver,
            session=session,
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
            _, best_trial = print_local_min_score_finalist_review(
                study,
                session=session,
                objective_mode=objective_mode,
                colors=COLORS,
                winner_trial=None,
            )
            if best_trial is not None and is_qualified_trial_value(best_trial.value):
                print_local_min_score_winner_summary(
                    winner_trial=best_trial,
                    session=session,
                    colors=COLORS,
                )
                export_status = export_best_params_if_requested(
                    study,
                    best_params_path=RUN_BEST_PARAMS_PATH,
                    fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                    colors=COLORS,
                    best_trial_resolver=best_trial_resolver,
                )
                if export_status != 0:
                    return 1
                if selected_model_mode == 'split':
                    finalize_best_trial_outputs(
                        session=session,
                        study=study,
                        best_trial_resolver=best_trial_resolver,
                        dataset_label=dataset_label,
                        db_file=db_file,
                        walk_forward_policy=walk_forward_policy,
                    )
                else:
                    print(f"{C_GREEN}🧭 全資料模式僅匯出 run_best，不額外產出 OOS 報表。{C_RESET}")
            else:
                print(f"{C_YELLOW}ℹ️ 訓練完成，但目前尚無通過 local_min_score gate 的 winner。{C_RESET}")
        elif export_policy == "interrupted_before_target":
            print(f"{C_YELLOW}ℹ️ 本輪僅完成 {session.current_session_trial}/{session.n_trials} 次，依規則不自動覆寫 {RUN_BEST_PARAMS_PATH}。{C_RESET}")
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
