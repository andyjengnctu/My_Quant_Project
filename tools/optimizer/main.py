import json
import os
import shutil
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
CHAMPION_PARAMS_PATH = os.path.join(MODELS_DIR, "champion_params.json")
CHAMPION_ARCHIVE_DIR = os.path.join(MODELS_DIR, "champion_archive")
UPGRADE_RECORD_DIR = os.path.join(OUTPUT_DIR, "upgrade_records")
TRAIN_MAX_POSITIONS = 10
TRAIN_START_YEAR = 2012
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
    os.makedirs(CHAMPION_ARCHIVE_DIR, exist_ok=True)
    os.makedirs(UPGRADE_RECORD_DIR, exist_ok=True)


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


def generate_walk_forward_report_from_payload(*, session, params_payload, dataset_label, db_file, best_trial_number=None):
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
        train_start_year=session.train_start_year,
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


def generate_best_trial_walk_forward_report(*, session, best_trial, dataset_label, db_file):
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
    )
    return report, report_paths, params_payload


def ensure_champion_params_bootstrap(*, champion_params_path: str, best_params_path: str):
    import shutil

    if os.path.exists(champion_params_path):
        return False
    if not os.path.exists(best_params_path):
        return False
    shutil.copy2(best_params_path, champion_params_path)
    return True


def generate_champion_challenger_compare_report(*, session, dataset_label, db_file, challenger_payload, challenger_report):
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
    )
    compare_payload = build_walk_forward_compare_payload(
        champion_payload=champion_payload,
        champion_report=champion_report,
        challenger_payload=challenger_payload,
        challenger_report=challenger_report,
        dataset_label=dataset_label,
        source_db_path=db_file,
        session_ts=session.session_ts,
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


def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_payload_equal(left: dict, right: dict) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(right, sort_keys=True, ensure_ascii=False)


def _write_upgrade_record(*, session_ts: str, compare_result: dict, compare_paths: dict, archived_champion_path: str | None, champion_params_path: str, challenger_best_params_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    compare_payload = dict((compare_result or {}).get("compare_payload") or {})
    assessment = dict(compare_payload.get("compare_assessment") or {})
    summary_compare = dict(compare_payload.get("summary_compare") or {})
    record = {
        "meta": {
            "session_ts": str(session_ts),
            "champion_params_path": str(champion_params_path),
            "challenger_best_params_path": str(challenger_best_params_path),
            "archived_champion_path": str(archived_champion_path or ""),
            "compare_md_path": str((compare_paths or {}).get("md_path", "")),
            "compare_json_path": str((compare_paths or {}).get("json_path", "")),
        },
        "assessment": assessment,
        "summary_compare": summary_compare,
    }
    base_name = f"upgrade_record_{session_ts}"
    json_path = os.path.join(output_dir, f"{base_name}.json")
    md_path = os.path.join(output_dir, f"{base_name}.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)

    ordered_keys = [
        "median_window_score",
        "median_ret_pct",
        "worst_ret_pct",
        "max_mdd",
        "median_annual_trades",
        "median_fill_rate",
        "flat_median_score",
        "down_window_count",
    ]
    def _fmt_metric(metric_key: str, value: float | int) -> str:
        if metric_key in ("median_window_score", "flat_median_score"):
            return f"{float(value):.3f}"
        if metric_key in ("median_annual_trades",):
            return f"{float(value):.2f}"
        if metric_key in ("down_window_count",):
            return str(int(value))
        return f"{float(value):.2f}%"

    lines = [
        "# Champion 升版紀錄",
        "",
        f"- 升版時間：{session_ts}",
        f"- 結果：`{assessment.get('status', 'fail')}`",
        f"- 建議：{assessment.get('recommendation', 'N/A')}",
        f"- 新 Champion：`{champion_params_path}`",
        f"- 舊 Champion 封存：`{archived_champion_path or 'N/A'}`",
        f"- Compare 報表：`{(compare_paths or {}).get('md_path', '')}`",
        "",
        "## 核心差異",
        "",
        "| 指標 | Champion(舊) | Challenger(新) | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric_key in ordered_keys:
        metric = dict(summary_compare.get(metric_key) or {})
        lines.append(
            f"| {metric_key} | {_fmt_metric(metric_key, metric.get('champion', 0.0))} | {_fmt_metric(metric_key, metric.get('challenger', 0.0))} | {_fmt_metric(metric_key, metric.get('delta', 0.0)) if metric_key not in ('down_window_count',) else f'{int(metric.get("delta", 0) or 0):+d}'} |"
        )

    lines.extend([
        "",
        "## 升版檢查項目",
        "",
        "| 檢查項目 | 實際值 | 門檻 | 結果 | 說明 |",
        "|---|---:|---:|---|---|",
    ])
    for check in list(assessment.get("checks") or []):
        actual = check.get("actual")
        name = str(check.get("name", ""))
        if "score" in name:
            actual_str = f"{float(actual):.3f}"
        elif "coverage" in name:
            actual_str = str(int(actual))
        else:
            actual_str = f"{float(actual):.2f}%"
        lines.append(
            f"| {name} | {actual_str} | {check.get('threshold', '')} | {'PASS' if check.get('passed') else 'FAIL'} | {check.get('note', '')} |"
        )

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"json_path": json_path, "md_path": md_path}


def promote_challenger_to_champion(*, compare_result: dict | None, compare_paths: dict | None, challenger_best_params_path: str, champion_params_path: str, archive_dir: str, upgrade_record_dir: str, session_ts: str):
    if compare_result is None or compare_paths is None:
        return None
    compare_payload = dict((compare_result.get("compare_payload") or {}))
    assessment = dict(compare_payload.get("compare_assessment") or {})
    if not bool(assessment.get("recommended_for_promotion", False)):
        return None
    if not os.path.exists(challenger_best_params_path) or not os.path.exists(champion_params_path):
        return None

    challenger_payload = _load_json_file(challenger_best_params_path)
    champion_payload = _load_json_file(champion_params_path)
    if _json_payload_equal(champion_payload, challenger_payload):
        return {
            "status": "noop",
            "archived_champion_path": None,
            "champion_params_path": champion_params_path,
            "upgrade_record_paths": None,
        }

    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(archive_dir, f"champion_{session_ts}.json")
    shutil.copy2(champion_params_path, archive_path)
    shutil.copy2(challenger_best_params_path, champion_params_path)
    upgrade_record_paths = _write_upgrade_record(
        session_ts=session_ts,
        compare_result=compare_result,
        compare_paths=compare_paths,
        archived_champion_path=archive_path,
        champion_params_path=champion_params_path,
        challenger_best_params_path=challenger_best_params_path,
        output_dir=upgrade_record_dir,
    )
    return {
        "status": "promoted",
        "archived_champion_path": archive_path,
        "champion_params_path": champion_params_path,
        "upgrade_record_paths": upgrade_record_paths,
    }


def print_promotion_outputs(*, promotion_result):
    if promotion_result is None:
        return
    status = str(promotion_result.get("status", ""))
    if status == "promoted":
        record_paths = dict(promotion_result.get("upgrade_record_paths") or {})
        print(f"{C_GREEN}⬆️ 已將 Challenger 升級為新的 Champion：{promotion_result.get('champion_params_path', '')}{C_RESET}")
        print(f"{C_GRAY}   舊 Champion 已封存：{promotion_result.get('archived_champion_path', '')}{C_RESET}")
        print(f"{C_GREEN}📝 已輸出升版紀錄：{record_paths.get('md_path', '')}{C_RESET}")
    elif status == "noop":
        print(f"{C_YELLOW}ℹ️ Challenger 與現役 Champion 相同，略過升版與封存。{C_RESET}")


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


def print_compare_outputs(*, compare_result, compare_paths, bootstrap_created: bool):
    if compare_result is not None and compare_paths is not None:
        compare_assessment = dict((compare_result.get("compare_payload") or {}).get("compare_assessment") or {})
        compare_status = str(compare_assessment.get("status", "fail")).upper()
        compare_color = C_GREEN if compare_status == "PASS" else (C_YELLOW if compare_status == "WATCH" else C_RED)
        summary_compare = dict((compare_result.get("compare_payload") or {}).get("summary_compare") or {})
        mws = dict(summary_compare.get("median_window_score") or {})
        flat_metric = dict(summary_compare.get("flat_median_score") or {})
        print(f"{C_GREEN}🆚 已輸出 Champion/Challenger 比較報表：{compare_paths['md_path']}{C_RESET}")
        print(
            f"{C_GRAY}   median_window_score: {float(mws.get('champion', 0.0)):.3f} -> {float(mws.get('challenger', 0.0)):.3f} "
            f"({float(mws.get('delta', 0.0)):+.3f}) | flat_median_score: {float(flat_metric.get('champion', 0.0)):.3f} -> {float(flat_metric.get('challenger', 0.0)):.3f} "
            f"({float(flat_metric.get('delta', 0.0)):+.3f}){C_RESET}"
        )
        print(f"{compare_color}   升版比較(MVP): {compare_status} | {compare_assessment.get('recommendation', 'N/A')}{C_RESET}")
    elif bootstrap_created:
        print(f"{C_YELLOW}ℹ️ 已由既有 best_params.json 自動建立初始現役版：{CHAMPION_PARAMS_PATH}{C_RESET}")


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
    from tools.optimizer.study_utils import (
        build_optimizer_db_file_path,
        get_best_completed_trial_or_none,
        is_qualified_trial_value,
        resolve_optimizer_seed,
        resolve_optimizer_trial_count,
    )

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
            bootstrap_created = ensure_champion_params_bootstrap(
                champion_params_path=CHAMPION_PARAMS_PATH,
                best_params_path=BEST_PARAMS_PATH,
            )
            export_status = export_best_params_if_requested(
                study,
                best_params_path=BEST_PARAMS_PATH,
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
                )
                print_walk_forward_outputs(report=report, report_paths=report_paths)
                compare_result, compare_paths = generate_champion_challenger_compare_report(
                    session=session,
                    dataset_label=dataset_label,
                    db_file=db_file,
                    challenger_payload=challenger_payload,
                    challenger_report=report,
                )
                print_compare_outputs(
                    compare_result=compare_result,
                    compare_paths=compare_paths,
                    bootstrap_created=bootstrap_created,
                )
                promotion_result = promote_challenger_to_champion(
                    compare_result=compare_result,
                    compare_paths=compare_paths,
                    challenger_best_params_path=BEST_PARAMS_PATH,
                    champion_params_path=CHAMPION_PARAMS_PATH,
                    archive_dir=CHAMPION_ARCHIVE_DIR,
                    upgrade_record_dir=UPGRADE_RECORD_DIR,
                    session_ts=session.session_ts,
                )
                print_promotion_outputs(promotion_result=promotion_result)
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
                bootstrap_created = ensure_champion_params_bootstrap(
                    champion_params_path=CHAMPION_PARAMS_PATH,
                    best_params_path=BEST_PARAMS_PATH,
                )
                export_status = export_best_params_if_requested(
                    study,
                    best_params_path=BEST_PARAMS_PATH,
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
                )
                print_walk_forward_outputs(report=report, report_paths=report_paths)
                compare_result, compare_paths = generate_champion_challenger_compare_report(
                    session=session,
                    dataset_label=dataset_label,
                    db_file=db_file,
                    challenger_payload=challenger_payload,
                    challenger_report=report,
                )
                print_compare_outputs(
                    compare_result=compare_result,
                    compare_paths=compare_paths,
                    bootstrap_created=bootstrap_created,
                )
                promotion_result = promote_challenger_to_champion(
                    compare_result=compare_result,
                    compare_paths=compare_paths,
                    challenger_best_params_path=BEST_PARAMS_PATH,
                    champion_params_path=CHAMPION_PARAMS_PATH,
                    archive_dir=CHAMPION_ARCHIVE_DIR,
                    upgrade_record_dir=UPGRADE_RECORD_DIR,
                    session_ts=session.session_ts,
                )
                print_promotion_outputs(promotion_result=promotion_result)
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
        session.close_trial_prep_executor()
        close_study_storage(study)


if __name__ == "__main__":
    run_cli_entrypoint(main)
