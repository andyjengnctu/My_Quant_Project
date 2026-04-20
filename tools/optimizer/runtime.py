import json
import os
import sqlite3
import sys

from core.log_utils import format_exception_summary

from core.display import print_strategy_dashboard
from tools.optimizer.callbacks import print_optimizer_trial_milestone_dashboard
from core.runtime_utils import is_interactive_console, safe_prompt_choice
from tools.optimizer.study_utils import (
    build_best_params_payload_from_trial,
    build_optimizer_trial_params,
    get_best_completed_trial_or_none,
    is_qualified_trial_value,
    list_completed_study_trials,
)


def prompt_existing_db_policy(db_file, colors):
    if not os.path.exists(db_file):
        return
    if not is_interactive_console():
        return
    choice = safe_prompt_choice(
        "\n👉 Portfolio 記憶庫：[1] 接續訓練 (預設)  [2] 刪除重來 : ",
        "1",
        ("1", "2"),
        "記憶庫操作選項",
    )
    if choice == "2":
        os.remove(db_file)
        print(f"{colors['red']}🗑️ 已刪除舊記憶。{colors['reset']}")


def create_optimizer_study(db_name, *, seed=None):
    try:
        import optuna
        from sqlalchemy.exc import SQLAlchemyError

        create_kwargs = {
            "study_name": "portfolio_optimization_overnight",
            "storage": db_name,
            "load_if_exists": True,
            "direction": "maximize",
        }
        if seed is not None:
            create_kwargs["sampler"] = optuna.samplers.TPESampler(seed=int(seed))

        return optuna.create_study(**create_kwargs)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Optimizer 記憶庫開啟失敗: {format_exception_summary(exc, include_traceback=False)}") from exc
    except (sqlite3.Error, SQLAlchemyError, RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Optimizer 記憶庫開啟失敗: {format_exception_summary(exc, include_traceback=False)}") from exc


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
        annual_trades=attrs.get("annual_trades", 0.0),
        reserved_buy_fill_rate=attrs.get("reserved_buy_fill_rate", 0.0),
        annual_return_pct=attrs.get("annual_return_pct", 0.0),
        bm_annual_return_pct=attrs.get("bm_annual_return_pct", 0.0),
        min_full_year_return_pct=attrs.get("min_full_year_return_pct", 0.0),
        bm_min_full_year_return_pct=attrs.get("bm_min_full_year_return_pct", 0.0),
    )
    print(
        f"{colors['gray']}   年化報酬率: {attrs.get('annual_return_pct', 0.0):.2f}% | "
        f"年化交易次數: {attrs.get('annual_trades', 0.0):.1f} 次/年 | "
        f"保留後買進成交率: {attrs.get('reserved_buy_fill_rate', 0.0):.1f}% | "
        f"完整年度數: {attrs.get('full_year_count', 0)} | "
        f"最差完整年度: {attrs.get('min_full_year_return_pct', 0.0):.2f}%{colors['reset']}"
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
        from tools.optimizer.study_utils import resolve_optimizer_run_request as _resolve_optimizer_run_request
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
    return None, str(request.get("source", "unknown"))


def print_resolved_run_request(*, n_trials, action, source, colors):
    action_labels = {
        "train": f"訓練 {int(n_trials)} 次",
        "export_candidate": "匯出 candidate_best",
        "promote_candidate": "promote candidate",
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
