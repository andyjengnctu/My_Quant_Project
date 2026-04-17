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
from core.model_paths import resolve_champion_params_path, resolve_models_dir
from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, get_taipei_now, has_help_flag, resolve_cli_program_name, validate_cli_args
from core.output_paths import build_output_dir
from core.walk_forward_policy import load_walk_forward_policy
from config.training_policy import OPTIMIZER_FIXED_TP_PERCENT

warnings.simplefilter("default")
warnings.filterwarnings("once", category=FutureWarning, module=r"optuna(\..*)?$")
warnings.filterwarnings("once", category=RuntimeWarning)


def configure_optuna_logging():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)


OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "ml_optimizer")
MODELS_DIR = resolve_models_dir(PROJECT_ROOT)
RUN_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "run_best_params.json")
CHAMPION_PARAMS_PATH = resolve_champion_params_path(PROJECT_ROOT)
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
        test_window_months=int(walk_forward_policy["test_window_months"]),
        regime_up_threshold_pct=float(walk_forward_policy["regime_up_threshold_pct"]),
        regime_down_threshold_pct=float(walk_forward_policy["regime_down_threshold_pct"]),
        min_window_bars=int(walk_forward_policy["min_window_bars"]),
        gate_min_median_score=float(walk_forward_policy["gate_min_median_score"]),
        gate_min_worst_ret_pct=float(walk_forward_policy["gate_min_worst_ret_pct"]),
        gate_min_flat_median_score=float(walk_forward_policy["gate_min_flat_median_score"]),
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

    params_payload = build_best_params_payload_from_trial(
        best_trial,
        fixed_tp_percent=session.optimizer_fixed_tp_percent,
    )
    report, report_paths = generate_walk_forward_report_from_payload(
        session=session,
        params_payload=params_payload,
        dataset_label=dataset_label,
        db_file=db_file,
        best_trial_number=int(best_trial.number) + 1,
        walk_forward_policy=walk_forward_policy,
    )
    return report, report_paths, params_payload



def ensure_champion_params_bootstrap(*, champion_params_path: str, run_best_params_path: str):
    import shutil

    if os.path.exists(champion_params_path):
        return None
    if os.path.exists(run_best_params_path):
        shutil.copy2(run_best_params_path, champion_params_path)
        return "run_best"
    return None



def _prompt_promote_choice(default: bool = False) -> tuple[bool, str]:
    default_text = "N" if not default else "Y"
    raw = input(f"👉 完成後若 Compare PASS，是否自動升版 Champion？ (y/N，預設 {default_text}): ").strip().lower()
    if raw == "":
        return default, "prompt_default"
    if raw in {"y", "yes", "1", "true", "on"}:
        return True, "prompt_yes"
    if raw in {"n", "no", "0", "false", "off"}:
        return False, "prompt_no"
    print(f"{C_YELLOW}⚠️ 無法辨識的輸入，沿用預設值：{'啟用' if default else '停用'} promote{C_RESET}")
    return default, "prompt_invalid_default"


def resolve_promote_request(argv, environ, *, requested_n_trials: int) -> tuple[bool, str]:
    args = [] if argv is None else list(argv)
    if int(requested_n_trials) == 0:
        return False, "trial_zero_locked"
    if any(str(arg).strip() == "--promote" for arg in args[1:]):
        return True, "cli_flag"
    env_map = os.environ if environ is None else environ
    raw_env = str(env_map.get("V16_OPTIMIZER_AUTO_PROMOTE", "")).strip().lower()
    if raw_env in {"1", "true", "yes", "on"}:
        return True, "env_var"
    if raw_env in {"0", "false", "no", "off"}:
        return False, "env_var_off"
    try:
        interactive_bare_run = (
            len(args) <= 1
            and sys.stdin is not None and sys.stdin.isatty()
            and sys.stdout is not None and sys.stdout.isatty()
        )
        if interactive_bare_run:
            return _prompt_promote_choice(default=False)
    except Exception as exc:
        _ = exc
        pass
    return False, "default_off"


def _json_file_equal(path_a: str, path_b: str) -> bool:
    import json
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return False
    with open(path_a, "r", encoding="utf-8") as handle:
        left = json.load(handle)
    with open(path_b, "r", encoding="utf-8") as handle:
        right = json.load(handle)
    return left == right


def promote_run_best_to_champion(*, session, compare_result, compare_paths, auto_promote_enabled: bool, promote_source: str) -> dict:
    import json
    import shutil

    assessment = dict(((compare_result or {}).get("compare_payload") or {}).get("compare_assessment") or {})
    status = str(assessment.get("status", "fail"))
    recommended = bool(assessment.get("recommended_for_promotion", False))
    result = {
        "attempted": False,
        "performed": False,
        "status": status,
        "reason": "disabled" if not auto_promote_enabled else "not_recommended",
        "promote_source": str(promote_source),
        "archive_path": None,
        "record_md_path": None,
        "record_json_path": None,
    }
    if not auto_promote_enabled:
        return result
    result["attempted"] = True
    if not recommended:
        return result
    if not os.path.exists(RUN_BEST_PARAMS_PATH):
        result["reason"] = "missing_run_best"
        return result
    if os.path.exists(CHAMPION_PARAMS_PATH) and _json_file_equal(CHAMPION_PARAMS_PATH, RUN_BEST_PARAMS_PATH):
        result["reason"] = "same_as_champion"
        return result

    archive_dir = os.path.join(MODELS_DIR, "champion_archive")
    records_dir = os.path.join(session.output_dir, "upgrade_records")
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(records_dir, exist_ok=True)
    archive_path = None
    if os.path.exists(CHAMPION_PARAMS_PATH):
        archive_path = os.path.join(archive_dir, f"champion_{session.session_ts}.json")
        shutil.copy2(CHAMPION_PARAMS_PATH, archive_path)
    shutil.copy2(RUN_BEST_PARAMS_PATH, CHAMPION_PARAMS_PATH)

    compare_payload = dict((compare_result or {}).get("compare_payload") or {})
    summary_compare = dict(compare_payload.get("summary_compare") or {})
    key_metrics = {}
    for key in ("median_window_score", "flat_median_score", "worst_ret_pct", "max_mdd", "median_annual_trades"):
        metric = dict(summary_compare.get(key) or {})
        key_metrics[key] = {
            "champion": metric.get("champion"),
            "challenger": metric.get("challenger"),
            "delta": metric.get("delta"),
        }
    record_payload = {
        "session_ts": str(session.session_ts),
        "status": status,
        "promote_source": str(promote_source),
        "archive_path": archive_path,
        "champion_params_path": CHAMPION_PARAMS_PATH,
        "run_best_params_path": RUN_BEST_PARAMS_PATH,
        "compare_md_path": None if compare_paths is None else compare_paths.get("md_path"),
        "compare_json_path": None if compare_paths is None else compare_paths.get("json_path"),
        "key_metrics": key_metrics,
        "recommendation": assessment.get("recommendation", "N/A"),
    }
    record_json_path = os.path.join(records_dir, f"upgrade_record_{session.session_ts}.json")
    record_md_path = os.path.join(records_dir, f"upgrade_record_{session.session_ts}.md")
    with open(record_json_path, "w", encoding="utf-8") as handle:
        json.dump(record_payload, handle, indent=2, ensure_ascii=False)
    lines = [
        "# 升版紀錄",
        "",
        f"- 時間：{session.session_ts}",
        f"- 狀態：`{status}`",
        f"- promote 來源：{promote_source}",
        f"- compare 報表：`{record_payload['compare_md_path'] or ''}`",
        f"- 舊 Champion 封存：`{archive_path or ''}`",
        "",
        "## 核心比較",
        "",
        "| 指標 | Champion | Challenger | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key, metric in key_metrics.items():
        lines.append(f"| {key} | {metric['champion']} | {metric['challenger']} | {metric['delta']} |")
    with open(record_md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    result.update({
        "performed": True,
        "reason": "promoted",
        "archive_path": archive_path,
        "record_md_path": record_md_path,
        "record_json_path": record_json_path,
    })
    return result


def generate_champion_challenger_compare_report(*, session, dataset_label, db_file, challenger_payload, challenger_report, walk_forward_policy: dict):
    from core.params_io import load_params_from_json, params_to_json_dict
    from tools.optimizer.walk_forward import build_walk_forward_compare_payload, write_walk_forward_compare_report

    if not os.path.exists(CHAMPION_PARAMS_PATH):
        return None, None

    champion_params = load_params_from_json(CHAMPION_PARAMS_PATH)
    champion_payload = params_to_json_dict(champion_params)
    champion_report, champion_report_paths = generate_walk_forward_report_from_payload(
        session=session,
        params_payload=champion_payload,
        dataset_label=f"{dataset_label}-champion",
        db_file=db_file,
        best_trial_number=None,
        walk_forward_policy=walk_forward_policy,
    )
    compare_payload = build_walk_forward_compare_payload(
        champion_payload=champion_payload,
        champion_report=champion_report,
        challenger_payload=challenger_payload,
        challenger_report=challenger_report,
        dataset_label=dataset_label,
        source_db_path=db_file,
        session_ts=session.session_ts,
        compare_worst_ret_tolerance_pct=float(walk_forward_policy["compare_worst_ret_tolerance_pct"]),
        compare_max_mdd_tolerance_pct=float(walk_forward_policy["compare_max_mdd_tolerance_pct"]),
    )
    compare_paths = write_walk_forward_compare_report(
        output_dir=session.output_dir,
        compare_payload=compare_payload,
    )
    return {
        "champion_report": champion_report,
        "champion_report_paths": champion_report_paths,
        "compare_payload": compare_payload,
    }, compare_paths


def print_walk_forward_outputs(*, report, report_paths):
    print(f"{C_GREEN}🧭 已輸出 Walk-Forward 驗證報表：{report_paths['md_path']}{C_RESET}")
    print(
        f"{C_GRAY}   視窗數: {report['summary'].get('window_count', 0)} | "
        f"分數中位數: {float(report['summary'].get('median_window_score', 0.0)):.3f} | "
        f"最差視窗報酬: {float(report['summary'].get('worst_ret_pct', 0.0)):.2f}%{C_RESET}"
    )
    upgrade_gate = dict(report.get("upgrade_gate") or {})
    gate_status = str(upgrade_gate.get("status", "fail")).upper()
    gate_color = C_GREEN if gate_status == "PASS" else (C_YELLOW if gate_status == "WATCH" else C_RED)
    print(f"{gate_color}   升版門檻(MVP): {gate_status} | {upgrade_gate.get('recommendation', 'N/A')}{C_RESET}")



def _format_pct_simple(value) -> str:
    return f"{float(value):.2f}%"



def print_compare_outputs(*, compare_result, compare_paths, bootstrap_source: str | None, promote_result: dict | None = None, auto_promote_enabled: bool = False, promote_source: str = "default_off"):
    if compare_result is not None and compare_paths is not None:
        compare_payload = dict((compare_result.get("compare_payload") or {}))
        compare_assessment = dict(compare_payload.get("compare_assessment") or {})
        quality_gate = dict(compare_assessment.get("quality_gate") or {})
        coverage_gate = dict(compare_assessment.get("coverage_gate") or {})
        compare_status = str(compare_assessment.get("status", "fail")).upper()
        compare_color = C_GREEN if compare_status == "PASS" else (C_YELLOW if compare_status == "WATCH" else C_RED)
        summary_compare = dict(compare_payload.get("summary_compare") or {})
        total_compare = dict(compare_payload.get("oos_total_compare") or {})
        total_metrics = dict(total_compare.get("metrics") or {})
        total_ret = dict(total_metrics.get("linked_total_return_pct") or {})
        mws = dict(summary_compare.get("median_window_score") or {})
        flat_metric = dict(summary_compare.get("flat_median_score") or {})
        print(f"{C_GREEN}🆚 已輸出 Champion/Challenger 比較報表（正式升版主入口）：{compare_paths['md_path']}{C_RESET}")
        print(
            f"{C_GRAY}   median_window_score: {float(mws.get('champion', 0.0)):.3f} -> {float(mws.get('challenger', 0.0)):.3f} "
            f"({float(mws.get('delta', 0.0)):+.3f}) | flat_median_score: {float(flat_metric.get('champion', 0.0)):.3f} -> {float(flat_metric.get('challenger', 0.0)):.3f} "
            f"({float(flat_metric.get('delta', 0.0)):+.3f}){C_RESET}"
        )
        print(
            f"{C_GRAY}   OOS總報酬(串接): Champion {_format_pct_simple(total_ret.get('champion', 0.0))} | "
            f"Challenger {_format_pct_simple(total_ret.get('challenger', 0.0))} | 0050 {_format_pct_simple(total_ret.get('benchmark', 0.0))}{C_RESET}"
        )
        print(
            f"{C_GRAY}   品質 gate: {str(quality_gate.get('status', 'fail')).upper()} | 覆蓋 gate: {str(coverage_gate.get('status', 'watch')).upper()} | "
            f"promote: {'ON' if auto_promote_enabled else 'OFF'} ({promote_source}){C_RESET}"
        )
        print(f"{compare_color}   升版比較(MVP): {compare_status} | {compare_assessment.get('recommendation', 'N/A')}{C_RESET}")
        challenger_upgrade_gate = dict(compare_assessment.get("challenger_upgrade_gate") or {})
        if challenger_upgrade_gate:
            print(
                f"{C_GRAY}   候選版自身 upgrade gate: {str(challenger_upgrade_gate.get('status', 'fail')).upper()} | "
                f"{challenger_upgrade_gate.get('recommendation', 'N/A')}{C_RESET}"
            )
        if promote_result is not None:
            if bool(promote_result.get("performed")):
                print(f"{C_GREEN}   已升級 Champion | 封存: {promote_result.get('archive_path') or 'N/A'}{C_RESET}")
                print(f"{C_GREEN}   升版紀錄: {promote_result.get('record_md_path') or 'N/A'}{C_RESET}")
            elif auto_promote_enabled:
                print(f"{C_YELLOW}   未執行 promote：{promote_result.get('reason', 'N/A')}{C_RESET}")
    elif bootstrap_source == "run_best":
        print(f"{C_YELLOW}ℹ️ 已由本輪最佳 run_best_params.json 建立初始現役版：{CHAMPION_PARAMS_PATH}{C_RESET}")

def main(argv=None, environ=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    validate_cli_args(argv, value_options=("--dataset",), flag_options=("--promote",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/optimizer/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full] [--promote]")
        print("說明: 預設資料集為完整；非互動模式預設訓練次數為 0；walk-forward 設定來自 config/walk_forward_policy.json；完成指定訓練次數或輸入 0 匯出時，會更新本輪最佳 run_best_params.json；現役正式版固定使用 champion_params.json；只有指定 --promote 或 V16_OPTIMIZER_AUTO_PROMOTE=1，且非 trial=0，且候選版自身 upgrade gate 與 compare gate 同時通過，才會實際升級 Champion。")
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
    from tools.optimizer.study_utils import (
        build_optimizer_db_file_path,
        get_best_completed_trial_or_none,
        is_qualified_trial_value,
        resolve_optimizer_seed,
        resolve_optimizer_trial_count,
    )

    walk_forward_policy = load_walk_forward_policy(PROJECT_ROOT, environ=environ)
    optimizer_required_min_rows = get_required_min_rows_from_high_len(OPTIMIZER_HIGH_LEN_MAX)
    session = build_optimizer_session(walk_forward_policy=walk_forward_policy)

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
    auto_promote_enabled, promote_source = resolve_promote_request(argv, environ, requested_n_trials=session.n_trials)

    try:
        optimizer_seed, seed_source = resolve_optimizer_seed(environ)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    if session.n_trials == 0:
        if not os.path.exists(db_file):
            print(
                f"{C_RED}❌ 記憶庫不存在，無法匯出: {db_file}；非互動模式預設 trial 數為 0，若要在乾淨 repo 建立新記憶庫，請先設定 V16_OPTIMIZER_TRIALS>0 或先完成一次訓練。{C_RESET}",
                file=sys.stderr,
            )
            return 1
        try:
            ensure_optimizer_db_usable(db_file)
            ensure_export_only_db_not_empty(db_file)
            study = create_optimizer_study(db_name, seed=optimizer_seed)
        except RuntimeError as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
        try:
            _bootstrap_source_pre = ensure_champion_params_bootstrap(
                champion_params_path=CHAMPION_PARAMS_PATH,
                run_best_params_path=RUN_BEST_PARAMS_PATH,
            )
            export_status = export_best_params_if_requested(
                study,
                best_params_path=RUN_BEST_PARAMS_PATH,
                fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                colors=COLORS,
            )
            if export_status != 0:
                return export_status
            if not os.path.isdir(selected_data_dir):
                raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
            csv_inputs, _ = discover_unique_csv_inputs(selected_data_dir)
            if not csv_inputs:
                raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir))
            best_trial = get_best_completed_trial_or_none(study)
            if best_trial is not None and is_qualified_trial_value(best_trial.value):
                session.load_raw_data(
                    selected_data_dir,
                    load_all_raw_data=load_all_raw_data,
                    required_min_rows=optimizer_required_min_rows,
                )
                report, report_paths, challenger_payload = generate_best_trial_walk_forward_report(
                    session=session,
                    best_trial=best_trial,
                    dataset_label=dataset_label,
                    db_file=db_file,
                    walk_forward_policy=walk_forward_policy,
                )
                print_walk_forward_outputs(report=report, report_paths=report_paths)
                bootstrap_source = ensure_champion_params_bootstrap(
                    champion_params_path=CHAMPION_PARAMS_PATH,
                    run_best_params_path=RUN_BEST_PARAMS_PATH,
                )
                compare_result, compare_paths = generate_champion_challenger_compare_report(
                    session=session,
                    dataset_label=dataset_label,
                    db_file=db_file,
                    challenger_payload=challenger_payload,
                    challenger_report=report,
                    walk_forward_policy=walk_forward_policy,
                )
                print_compare_outputs(
                    compare_result=compare_result,
                    compare_paths=compare_paths,
                    bootstrap_source=bootstrap_source,
                    promote_result=None,
                    auto_promote_enabled=False,
                    promote_source="trial_zero_locked",
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
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端 (End-to-End) 投資組合極速 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"{C_GRAY}📁 使用資料集: {dataset_label} | "
        f"來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}"
    )
    print(f"{C_GRAY}🗃️ Optimizer 記憶庫: {db_file}{C_RESET}")
    print(f"{C_GRAY}🎲 Optimizer seed: {optimizer_seed if optimizer_seed is not None else '未設定'} | 來源: {seed_source}{C_RESET}")
    print(
        f"{C_GRAY}🧭 Walk-forward policy: {walk_forward_policy.get('policy_path', 'config/walk_forward_policy.json')} | "
        f"start={walk_forward_policy['train_start_year']} | min_years={walk_forward_policy['min_train_years']} | "
        f"test_months={walk_forward_policy['test_window_months']} | min_bars={walk_forward_policy['min_window_bars']} | "
        f"gate(score>={float(walk_forward_policy['gate_min_median_score']):.3f}, worst>={float(walk_forward_policy['gate_min_worst_ret_pct']):.2f}%, "
        f"flat>={float(walk_forward_policy['gate_min_flat_median_score']):.3f}) | "
        f"compare_tol(ret={float(walk_forward_policy['compare_worst_ret_tolerance_pct']):.2f}%, mdd={float(walk_forward_policy['compare_max_mdd_tolerance_pct']):.2f}%)"
        f"{C_RESET}"
    )
    print(f"{C_GRAY}⬆️ Auto promote: {'ON' if auto_promote_enabled else 'OFF'} | 來源: {promote_source}{C_RESET}")

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
                _bootstrap_source_pre = ensure_champion_params_bootstrap(
                    champion_params_path=CHAMPION_PARAMS_PATH,
                    run_best_params_path=RUN_BEST_PARAMS_PATH,
                )
                export_status = export_best_params_if_requested(
                    study,
                    best_params_path=RUN_BEST_PARAMS_PATH,
                    fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                    colors=COLORS,
                )
                if export_status != 0:
                    return 1
                report, report_paths, challenger_payload = generate_best_trial_walk_forward_report(
                    session=session,
                    best_trial=best_trial,
                    dataset_label=dataset_label,
                    db_file=db_file,
                    walk_forward_policy=walk_forward_policy,
                )
                print_walk_forward_outputs(report=report, report_paths=report_paths)
                bootstrap_source = ensure_champion_params_bootstrap(
                    champion_params_path=CHAMPION_PARAMS_PATH,
                    run_best_params_path=RUN_BEST_PARAMS_PATH,
                )
                compare_result, compare_paths = generate_champion_challenger_compare_report(
                    session=session,
                    dataset_label=dataset_label,
                    db_file=db_file,
                    challenger_payload=challenger_payload,
                    challenger_report=report,
                    walk_forward_policy=walk_forward_policy,
                )
                promote_result = promote_run_best_to_champion(
                    session=session,
                    compare_result=compare_result,
                    compare_paths=compare_paths,
                    auto_promote_enabled=auto_promote_enabled,
                    promote_source=promote_source,
                )
                print_compare_outputs(
                    compare_result=compare_result,
                    compare_paths=compare_paths,
                    bootstrap_source=bootstrap_source,
                    promote_result=promote_result,
                    auto_promote_enabled=auto_promote_enabled,
                    promote_source=promote_source,
                )
        elif export_policy == "interrupted_before_target":
            print(
                f"{C_YELLOW}ℹ️ 本輪僅完成 {session.current_session_trial}/{session.n_trials} 次，"
                f"依規則不自動覆寫 {RUN_BEST_PARAMS_PATH}。{C_RESET}"
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
