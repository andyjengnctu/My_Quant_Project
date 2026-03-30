import json
import os
import sqlite3
import sys

import optuna
from sqlalchemy.exc import SQLAlchemyError

from core.log_utils import format_exception_summary

from core.display import print_strategy_dashboard
from core.runtime_utils import safe_prompt_choice
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
    choice = safe_prompt_choice(
        "\n👉 發現舊有 Portfolio 記憶庫！ [1] 接續訓練  [2] 刪除重來 (預設 1): ",
        "1",
        ("1", "2"),
        "記憶庫操作選項",
    )
    if choice == "2":
        os.remove(db_file)
        print(f"{colors['red']}🗑️ 已刪除舊記憶。{colors['reset']}")


def create_optimizer_study(db_name):
    try:
        return optuna.create_study(
            study_name="portfolio_optimization_overnight",
            storage=db_name,
            load_if_exists=True,
            direction="maximize",
        )
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


def maybe_print_history_best(study, *, fixed_tp_percent, train_enable_rotation, train_max_positions, colors):
    if len(study.trials) <= 0:
        return
    print(f"\n{colors['green']}✅ 已累積 {len(study.trials)} 次經驗。{colors['reset']}")
    best_trial = get_best_completed_trial_or_none(study)
    if best_trial is None:
        print(f"{colors['gray']}ℹ️ 記憶庫目前尚無已完成 trial，略過歷史最佳儀表板還原。{colors['reset']}")
        return
    if not is_qualified_trial_value(best_trial.value):
        return
    print(f"\n{colors['cyan']}📜 【歷史突破紀錄還原】{colors['reset']}")
    print(f"{colors['red']}🏆 目前記憶庫的最強參數！ (來自累積第 {best_trial.number + 1} 次測試){colors['reset']}")
    print_best_trial_dashboard(
        best_trial,
        fixed_tp_percent=fixed_tp_percent,
        train_enable_rotation=train_enable_rotation,
        train_max_positions=train_max_positions,
        colors=colors,
    )


def export_best_params_if_requested(study, *, best_params_path, fixed_tp_percent, colors):
    if len(study.trials) == 0:
        print(f"{colors['red']}❌ 記憶庫為空，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1
    completed_trials = list_completed_study_trials(study)
    if not completed_trials:
        print(f"{colors['red']}❌ 目前記憶庫中尚無已完成紀錄，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1

    best_trial = get_best_completed_trial_or_none(study)
    if best_trial is None:
        print(f"{colors['red']}❌ 目前記憶庫中尚無可提取的最佳 completed trial，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1
    if is_qualified_trial_value(best_trial.value):
        best_params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=fixed_tp_percent)
        with open(best_params_path, "w", encoding="utf-8") as handle:
            json.dump(best_params_payload, handle, indent=4, ensure_ascii=False)
        print(f"\n{colors['green']}💾 匯出成功！已從記憶庫提取最強參數！{colors['reset']}\n")
        return 0

    print(f"{colors['red']}❌ 目前記憶庫中尚無及格的紀錄，無法匯出。{colors['reset']}", file=sys.stderr)
    return 1


def resolve_trial_count_or_exit(session, *, environ, resolve_optimizer_trial_count, colors):
    try:
        session.n_trials, trial_source = resolve_optimizer_trial_count(environ)
    except ValueError as exc:
        print(f"{colors['red']}❌ {exc}{colors['reset']}", file=sys.stderr)
        return 1, None
    return None, trial_source


def print_resolved_trial_count(session, *, trial_source, colors):
    print(f"{colors['gray']}🎯 訓練次數: {session.n_trials} | 來源: {trial_source}{colors['reset']}")
