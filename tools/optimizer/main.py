import json
import os
import sys
import time
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
from core.walk_forward_policy import (
    build_optimizer_effective_policy_fingerprint,
    build_optimizer_runtime_policy,
    load_walk_forward_policy,
)
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
CANDIDATE_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "candidate_best_params.json")
CANDIDATE_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "candidate_best_summary.json")
RUN_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "run_best_summary.json")
DEFAULT_WALK_FORWARD_POLICY = load_walk_forward_policy(PROJECT_ROOT)
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


def _build_best_summary_payload(*, winner_trial, finalist_entry, objective_mode: str, walk_forward_policy: dict, action_label: str):
    if winner_trial is None or finalist_entry is None:
        raise ValueError("缺少 winner_trial 或 finalist_entry，無法建立 summary")
    return {
        "trial_number": int(winner_trial.number) + 1,
        "base_score": float(finalist_entry["base_score"]),
        "local_min_score": float(finalist_entry["local_min_score"]),
        "retention": float(finalist_entry["local_retention"]),
        "local_gate": bool(finalist_entry["gate_pass"]),
        "objective_mode": str(objective_mode),
        "train_start_year": int(walk_forward_policy.get("train_start_year", 0)),
        "search_train_end_year": int(walk_forward_policy.get("search_train_end_year", 0)),
        "oos_start_year": walk_forward_policy.get("oos_start_year"),
        "action": str(action_label),
        "created_at": get_taipei_now().isoformat(),
    }


def _print_candidate_vs_run_best_summary(*, candidate_summary: dict, run_best_summary: dict | None):
    print(f"{C_GRAY}{'-' * 96}{C_RESET}")
    print(f"      candidate | base={float(candidate_summary.get('base_score', 0.0)):.3f} | local_min={float(candidate_summary.get('local_min_score', 0.0)):.3f} | retention={float(candidate_summary.get('retention', 0.0)):.3f}")
    if run_best_summary is None:
        print(f"      run_best  | 尚無 summary，視同未建立 promote 基線")
    else:
        print(f"      run_best  | base={float(run_best_summary.get('base_score', 0.0)):.3f} | local_min={float(run_best_summary.get('local_min_score', 0.0)):.3f} | retention={float(run_best_summary.get('retention', 0.0)):.3f}")
    print(f"{C_GRAY}{'-' * 96}{C_RESET}")


def _should_promote_candidate(*, candidate_summary: dict, run_best_summary: dict | None):
    candidate_local_min = float(candidate_summary.get("local_min_score", 0.0))
    candidate_base_score = float(candidate_summary.get("base_score", 0.0))
    candidate_retention = float(candidate_summary.get("retention", float("-inf")))
    if candidate_local_min <= 0.0:
        return False, "candidate.local_min_score <= 0"
    if candidate_base_score <= 0.0:
        return False, "candidate.base_score <= 0"
    if run_best_summary is None:
        return True, "run_best summary 缺失，視同首次 promote"
    run_best_retention = float(run_best_summary.get("retention", float("-inf")))
    if candidate_retention > run_best_retention:
        return True, "candidate.retention > run_best.retention"
    if candidate_retention == run_best_retention:
        run_best_local_min = float(run_best_summary.get("local_min_score", float("-inf")))
        if candidate_local_min > run_best_local_min:
            return True, "candidate.retention == run_best.retention，且 candidate.local_min_score > run_best.local_min_score"
    return False, "candidate 未通過 promote 比較規則"


def _promote_candidate_to_run_best():
    candidate_params = _load_json_file_or_none(CANDIDATE_BEST_PARAMS_PATH)
    if candidate_params is None:
        print(f"{C_RED}❌ 找不到 candidate_best 參數檔: {CANDIDATE_BEST_PARAMS_PATH}{C_RESET}", file=sys.stderr)
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
    cli_trials_raw = _extract_cli_value(argv, '--trials')
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
    validate_cli_args(argv, value_options=("--dataset", "--model", "--trials"), flag_options=("--timing",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/optimizer/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full] [--model split|full] [--trials N] [--timing]")
        print("說明: split=固定 pre-deploy train 選參 + OOS 獨立驗證；full=全資料選參。可用 --trials N 直接指定訓練次數；可用 --timing 啟用 CLI 測時模式，預設跑 3 個 trials，亦可搭配 --trials N。未使用 --trials 時，仍維持既有互動選單 / ENV 行為。輸入 0 匯出 candidate_best；輸入 P promote candidate。正常完成訓練後會自動寫入 candidate_best 並自動挑戰進版 run_best；若使用者中斷則不做。")
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
            session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
            finalists, best_trial = print_local_min_score_finalist_review(
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
                best_params_path=CANDIDATE_BEST_PARAMS_PATH,
                fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                colors=COLORS,
                best_trial_resolver=(lambda _study, _best_trial=best_trial: _best_trial),
                suppress_success_message=True,
            )
            if export_status != 0:
                return export_status
            finalist_entry = _find_finalist_entry(finalists, best_trial)
            candidate_summary = _build_best_summary_payload(
                winner_trial=best_trial,
                finalist_entry=finalist_entry,
                objective_mode=objective_mode,
                walk_forward_policy=walk_forward_policy,
                action_label="export_candidate",
            )
            _write_json_file(CANDIDATE_BEST_SUMMARY_PATH, candidate_summary)
            print(f"{C_GREEN}💾 candidate_best 已寫入：{CANDIDATE_BEST_PARAMS_PATH}{C_RESET}")
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
    print_resolved_trial_count(session, trial_source=trial_source, colors=COLORS)
    if timing_mode:
        print(f"{C_YELLOW}⏱️ CLI 測時模式已啟用：不覆寫 candidate_best / run_best，記憶庫改寫到 outputs/ml_optimizer，並以固定 seed 的 RandomSampler 重播同一組 trial 組合，且關閉 milestone dashboard 顯示以避免 UI 開銷污染測時。{C_RESET}")
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
    inline_override_fields = list(walk_forward_policy.get("inline_override_fields", []) or [])
    override_text = "" if not inline_override_fields else f" | override={','.join(inline_override_fields)}"
    print(
        f"{C_GRAY}🧭 訓練模式: {selected_model_mode} | 來源: {model_mode_source} | "
        f"設定: {walk_forward_policy.get('policy_path', 'config/training_policy.py')}{override_text} | {scope_text}{C_RESET}"
    )
    print(f"{C_GRAY}Train/Test policy: {scope_text}{C_RESET}")

    try:
        if not timing_mode:
            prompt_existing_db_policy(db_file, COLORS)
        if os.path.exists(db_file):
            ensure_optimizer_db_usable(db_file)
        study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=("random" if timing_mode else "tpe"))
        _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
    except (ValueError, RuntimeError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    session.timing_mode = timing_mode
    session.disable_milestone_dashboard = bool(timing_mode)

    overall_started_at = time.perf_counter()
    raw_data_load_sec = 0.0
    optimize_wall_sec = 0.0
    try:
        raw_data_load_started_at = time.perf_counter()
        session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
        raw_data_load_sec = max(0.0, time.perf_counter() - raw_data_load_started_at)

        maybe_print_history_best(
            study,
            fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
            train_enable_rotation=TRAIN_ENABLE_ROTATION,
            train_max_positions=TRAIN_MAX_POSITIONS,
            colors=COLORS,
            best_trial_resolver=session.get_best_completed_trial_or_none,
            session=None,
        )

        session.profile_recorder.init_output_files()

        print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
        session.profile_recorder.mark_run_started()
        optimize_started_at = time.perf_counter()
        training_interrupted = False
        try:
            study.optimize(session.objective, n_trials=session.n_trials, n_jobs=1, callbacks=[session.monitoring_callback])
        except KeyboardInterrupt:
            training_interrupted = True
            print(f"\n{C_YELLOW}⚠️ 使用者中斷訓練流程。{C_RESET}")

        optimize_wall_sec = max(0.0, time.perf_counter() - optimize_started_at)
        print()
        session.profile_recorder.print_summary()
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
                optimize_wall_sec=optimize_wall_sec,
                total_wall_sec=max(0.0, time.perf_counter() - overall_started_at),
                optimizer_seed=optimizer_seed,
                timing_sampler_kind=("random" if timing_mode else "tpe"),
            )
            timing_summary_path = write_timing_summary(
                output_dir=OUTPUT_DIR,
                session_ts=session.session_ts,
                payload=timing_payload,
            )
            print_timing_summary(payload=timing_payload)
            print(f"{C_GRAY}   測時摘要: {timing_summary_path}{C_RESET}")
        elif should_export:
            finalists, best_trial = print_local_min_score_finalist_review(
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
                )
                _write_json_file(CANDIDATE_BEST_SUMMARY_PATH, candidate_summary)
                print(f"{C_GREEN}💾 candidate_best 已寫入：{CANDIDATE_BEST_PARAMS_PATH}{C_RESET}")
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
