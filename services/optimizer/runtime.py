import json
import os
import sqlite3
import sys

from core.console_report import render_menu_item
from core.log_utils import format_exception_summary

from config.training_performance_policy import (
    is_optimizer_single_fold_tpe_parallel_search_allowed_default,
    resolve_optimizer_single_fold_search_parallel_trials_default,
)

from core.display import print_strategy_dashboard
from services.optimizer.callbacks import print_optimizer_trial_milestone_dashboard
from core.runtime_utils import is_interactive_console, safe_prompt_choice
from services.optimizer.study_utils import (
    build_best_params_payload_from_trial,
    build_optimizer_trial_params,
    get_best_completed_trial_or_none,
    is_qualified_trial_value,
    list_completed_study_trials,
)


def prompt_existing_db_policy(db_file, colors):
    if not db_file or not os.path.exists(db_file):
        return
    if not is_interactive_console():
        return
    choice = safe_prompt_choice(
        "\n👉 Portfolio 記憶庫："
        + render_menu_item(1, "接續訓練", default=True)
        + "  "
        + render_menu_item(2, "刪除重來")
        + " : ",
        "1",
        ("1", "2"),
        "記憶庫操作選項",
    )
    if choice == "2":
        os.remove(db_file)
        print(f"{colors['red']}🗑️ 已刪除舊記憶。{colors['reset']}")


def create_optimizer_study(db_name, *, seed=None, sampler_kind="tpe"):
    try:
        import optuna
        from sqlalchemy.exc import SQLAlchemyError

        create_kwargs = {
            "study_name": "portfolio_optimization_overnight",
            "direction": "maximize",
        }
        if db_name is not None:
            create_kwargs["storage"] = db_name
            create_kwargs["load_if_exists"] = True
        sampler_key = str(sampler_kind or "tpe").strip().lower()
        if sampler_key == "random":
            create_kwargs["sampler"] = optuna.samplers.RandomSampler(seed=(None if seed is None else int(seed)))
        elif seed is not None:
            create_kwargs["sampler"] = optuna.samplers.TPESampler(seed=int(seed))

        return optuna.create_study(**create_kwargs)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Optimizer 記憶庫開啟失敗: {format_exception_summary(exc, include_traceback=False)}") from exc
    except (sqlite3.Error, SQLAlchemyError, RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Optimizer 記憶庫開啟失敗: {format_exception_summary(exc, include_traceback=False)}") from exc


def _env_value_for_optimizer_runtime(environ, name: str, default: str) -> str:
    if isinstance(environ, dict) and environ.get(name) is not None:
        value = environ.get(name)
    else:
        value = os.environ.get(name, default)
    if value is None or str(value).strip() == "":
        return str(default)
    return str(value).strip()


def _env_flag_for_optimizer_runtime(environ, name: str, default: bool) -> bool:
    value = None
    if isinstance(environ, dict):
        value = environ.get(name)
    if value is None:
        value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", "n"}


def resolve_optimizer_single_fold_search_parallel_trials(environ=None, *, sampler_kind: str) -> int:
    """解析 rolling / 非 rolling 共用的單一 search unit Optuna trial 併發數。"""
    default_trials = resolve_optimizer_single_fold_search_parallel_trials_default()
    raw_value = _env_value_for_optimizer_runtime(
        environ,
        "OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS",
        str(default_trials),
    )
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError):
        resolved = int(default_trials)
    resolved = max(1, min(16, resolved))
    sampler_text = str(sampler_kind or "").strip().lower()
    if sampler_text == "tpe" and resolved > 1:
        allow_tpe_parallel = _env_flag_for_optimizer_runtime(
            environ,
            "OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH",
            is_optimizer_single_fold_tpe_parallel_search_allowed_default(),
        )
        if not allow_tpe_parallel:
            return 1
    return int(resolved)

def ensure_optimizer_db_usable(db_file):
    if not os.path.exists(db_file):
        return

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA schema_version;").fetchone()
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise RuntimeError(f"Optimizer 記憶庫檔案損壞或不可讀: {format_exception_summary(exc, include_traceback=False)}") from exc
    finally:
        if conn is not None:
            conn.close()


def ensure_export_only_db_not_empty(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        has_any_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
        ).fetchone()
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise RuntimeError(f"Optimizer 記憶庫檔案損壞或不可讀: {format_exception_summary(exc, include_traceback=False)}") from exc
    finally:
        if conn is not None:
            conn.close()

    if not has_any_table:
        raise RuntimeError("記憶庫為空，無法匯出")


def print_best_trial_dashboard(trial, *, fixed_tp_percent, train_enable_rotation, train_max_positions, colors):
    attrs = trial.user_attrs
    params = build_optimizer_trial_params(trial.params, attrs, fixed_tp_percent=fixed_tp_percent)
    mode_display = "啟用 (汰弱換強)" if train_enable_rotation else "關閉 (穩定鎖倉)"
    print_strategy_dashboard(
        params=params,
        title="績效與風險對比表",
        mode_display=mode_display,
        max_pos=train_max_positions,
        trades=attrs["pf_trades"],
        missed_b=attrs.get("missed_buys", 0),
        missed_s=attrs.get("missed_sells", 0),
        final_eq=attrs["final_equity"],
        avg_exp=attrs["avg_exposure"],
        max_exp=attrs.get("max_exposure", None),
        sys_ret=attrs["pf_return"],
        bm_ret=attrs["bm_return"],
        sys_mdd=attrs["pf_mdd"],
        bm_mdd=attrs["bm_mdd"],
        win_rate=attrs["win_rate"],
        payoff=attrs["pf_payoff"],
        ev=attrs["pf_ev"],
        r_sq=attrs.get("r_squared", 0.0),
        m_win_rate=attrs.get("m_win_rate", 0.0),
        bm_r_sq=attrs.get("bm_r_squared", 0.0),
        bm_m_win_rate=attrs.get("bm_m_win_rate", 0.0),
        normal_trades=attrs.get("normal_trades", attrs["pf_trades"]),
        extended_trades=attrs.get("extended_trades", 0),
        reentry_trades=attrs.get("reentry_trades", 0),
        annual_trades=attrs.get("annual_trades", 0.0),
        reserved_buy_fill_rate=attrs.get("reserved_buy_fill_rate", 0.0),
        annual_return_pct=attrs.get("annual_return_pct", 0.0),
        bm_annual_return_pct=attrs.get("bm_annual_return_pct", 0.0),
        min_full_year_return_pct=attrs.get("min_full_year_return_pct", 0.0),
        min_month_return_pct=attrs.get("min_month_return_pct", 0.0),
        min_quarter_return_pct=attrs.get("min_quarter_return_pct", 0.0),
        portfolio_total_r=attrs.get("pf_total_r", 0.0),
        portfolio_median_r=attrs.get("pf_median_r", 0.0),
        score_total_r=attrs.get("score_total_r", attrs.get("single_stock_total_r", 0.0)),
        score_median_r=attrs.get("score_median_r", attrs.get("single_stock_median_r", 0.0)),
        bm_min_full_year_return_pct=attrs.get("bm_min_full_year_return_pct", 0.0),
        bm_min_month_return_pct=attrs.get("bm_min_month_return_pct", 0.0),
        bm_min_quarter_return_pct=attrs.get("bm_min_quarter_return_pct", 0.0),
        params_section_title="目前 trial 參數（非 final artifact ensemble）",
    )
    print(
        f"{colors['gray']}   年化報酬率: {attrs.get('annual_return_pct', 0.0):.2f}% | "
        f"年化交易次數: {attrs.get('annual_trades', 0.0):.1f} 次/年 | "
        f"保留後買進成交率: {attrs.get('reserved_buy_fill_rate', 0.0):.1f}% | "
        f"完整年度數: {attrs.get('full_year_count', 0)} | "
        f"最差完整年度: {attrs.get('min_full_year_return_pct', 0.0):.2f}% | "
        f"完整季度數: {attrs.get('full_quarter_count', 0)} | "
        f"最差完整季度: {attrs.get('min_quarter_return_pct', 0.0):.2f}% | "
        f"完整月度數: {attrs.get('full_month_count', 0)} | "
        f"最差完整月度: {attrs.get('min_month_return_pct', 0.0):.2f}%{colors['reset']}"
    )


def maybe_print_history_best(study, *, fixed_tp_percent, train_enable_rotation, train_max_positions, colors, best_trial_resolver=None, session=None):
    if len(study.trials) <= 0:
        return
    print(f"\n{colors['green']}✅ 已累積 {len(study.trials)} 次經驗。{colors['reset']}")
    resolver = get_best_completed_trial_or_none if best_trial_resolver is None else best_trial_resolver
    best_trial = resolver(study)
    if best_trial is None:
        print(f"{colors['gray']}ℹ️ 記憶庫目前尚無已完成 trial，略過歷史最佳儀表板還原。{colors['reset']}")
        return
    if not is_qualified_trial_value(best_trial.value):
        return
    if session is not None and bool(getattr(session, "disable_milestone_dashboard", False)):
        print(f"{colors['gray']}ℹ️ 非 rolling 訓練結果表格已關閉，略過歷史最佳儀表板還原。{colors['reset']}")
        return
    print(f"\n{colors['cyan']}📜 【歷史突破紀錄還原】{colors['reset']}")
    if session is not None:
        print_optimizer_trial_milestone_dashboard(
            session,
            best_trial,
            title="績效與風險對比表",
            milestone_title=f"🏆 目前記憶庫的最強參數！ (來自累積第 {best_trial.number + 1} 次測試)",
        )
        return
    print(f"{colors['red']}🏆 目前記憶庫的最強參數！ (來自累積第 {best_trial.number + 1} 次測試){colors['reset']}")
    print_best_trial_dashboard(
        best_trial,
        fixed_tp_percent=fixed_tp_percent,
        train_enable_rotation=train_enable_rotation,
        train_max_positions=train_max_positions,
        colors=colors,
    )


def export_best_params_if_requested(study, *, best_params_path, fixed_tp_percent, colors, best_trial_resolver=None, artifact_label=None, suppress_success_message: bool = False):
    if len(study.trials) == 0:
        print(f"{colors['red']}❌ 記憶庫為空，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1
    completed_trials = list_completed_study_trials(study)
    if not completed_trials:
        print(f"{colors['red']}❌ 目前記憶庫中尚無已完成紀錄，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1

    resolver = get_best_completed_trial_or_none if best_trial_resolver is None else best_trial_resolver
    best_trial = resolver(study)
    if best_trial is None:
        print(f"{colors['red']}❌ 目前記憶庫中尚無通過 local_min_score gate 的最佳 completed trial，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1
    if is_qualified_trial_value(best_trial.value):
        best_params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=fixed_tp_percent)
        with open(best_params_path, "w", encoding="utf-8") as handle:
            json.dump(best_params_payload, handle, indent=4, ensure_ascii=False)
        if not suppress_success_message:
            print(f"\n{colors['green']}💾 匯出成功！已從記憶庫提取最強參數！{colors['reset']}\n")
        return 0

    print(f"{colors['red']}❌ 目前記憶庫中尚無及格的紀錄，無法匯出。{colors['reset']}", file=sys.stderr)
    return 1


def resolve_training_session_export_policy(*, requested_n_trials, completed_session_trials, interrupted):
    requested_trials = int(requested_n_trials)
    completed_trials = int(completed_session_trials)
    if requested_trials <= 0:
        return True, "export_only"
    if completed_trials >= requested_trials:
        return True, "target_reached"
    if interrupted:
        return False, "interrupted_before_target"
    return False, "target_not_reached"


def resolve_run_request_or_exit(*, environ, resolve_optimizer_run_request, colors):
    try:
        return None, resolve_optimizer_run_request(environ)
    except ValueError as exc:
        print(f"{colors['red']}❌ {exc}{colors['reset']}", file=sys.stderr)
        return 1, None


def resolve_trial_count_or_exit(session, *, environ, resolve_optimizer_trial_count=None, colors, resolve_optimizer_run_request=None):
    if resolve_optimizer_run_request is None:
        from services.optimizer.study_utils import resolve_optimizer_run_request as _resolve_optimizer_run_request
        resolve_optimizer_run_request = _resolve_optimizer_run_request
    exit_code, request = resolve_run_request_or_exit(
        environ=environ,
        resolve_optimizer_run_request=resolve_optimizer_run_request,
        colors=colors,
    )
    if exit_code is not None:
        return exit_code, None
    session.n_trials = int(request["n_trials"])
    session.run_action = str(request.get("action", "train"))
    requested_model_mode = str(request.get("model_mode", "") or "").strip().lower()
    if requested_model_mode:
        session.requested_model_mode = requested_model_mode
    requested_study_scope = str(request.get("study_scope", "") or "").strip().lower()
    if requested_study_scope:
        session.requested_study_scope = requested_study_scope
    requested_study_db_action = str(request.get("study_db_action", "") or "").strip().lower()
    if requested_study_db_action:
        session.requested_study_db_action = requested_study_db_action
    return None, str(request.get("source", "unknown"))


def print_resolved_run_request(*, n_trials, action, source, colors):
    action_labels = {
        "train": f"訓練 {int(n_trials)} 次",
        "export_candidate": "輸出參數",
        "promote_candidate": "promote candidate",
        "outer_rolling_oos": "outer rolling monthly OOS test",
    }
    print(f"{colors['gray']}🎯 Optimizer 動作: {action_labels.get(str(action), str(action))} | 來源: {source}{colors['reset']}")


def print_resolved_trial_count(session, *, trial_source, colors):
    action = str(getattr(session, "run_action", "train"))
    print_resolved_run_request(
        n_trials=int(getattr(session, "n_trials", 0)),
        action=action,
        source=trial_source,
        colors=colors,
    )
