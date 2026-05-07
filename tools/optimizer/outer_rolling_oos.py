from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from config.training_policy import (
    OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_FIXED_TP_PERCENT,
    OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
)
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW
from core.params_io import build_params_from_mapping
from core.runtime_utils import get_taipei_now, is_interactive_console, safe_prompt_choice
from core.strategy_params import build_runtime_param_raw_value
from core.walk_forward_policy import build_optimizer_runtime_policy
from tools.optimizer.prep import prepare_trial_inputs
from tools.optimizer.robustness import (
    _has_dependency_warning,
    _has_inner_validate_pass,
    is_dominant_year_dependency_anti_overfit_enabled,
    is_inner_validate_anti_overfit_enabled,
    list_local_min_score_finalists,
)
from tools.optimizer.study_utils import (
    INVALID_TRIAL_VALUE,
    build_best_params_payload_from_trial,
    is_qualified_trial_value,
)
from tools.optimizer.walk_forward import evaluate_walk_forward


@dataclass
class OuterRollingConfig:
    training_start_year: int
    first_oos_year: int
    last_oos_year: int
    trials_per_fold: int
    window_mode: str = "fixed"
    train_window_years: int = 5
    confirm: bool = True


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "N/A"
    total = max(0, int(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _prompt_int(label: str, default: int, *, minimum: int | None = None) -> int:
    if not is_interactive_console():
        return int(default)
    raw = input(f"{label:<28} [{int(default)}] : ").strip()
    if raw == "":
        value = int(default)
    else:
        value = int(raw)
    if minimum is not None and value < int(minimum):
        raise ValueError(f"{label} 必須 >= {minimum}，收到: {value}")
    return int(value)


def _prompt_str(label: str, default: str, *, allowed: tuple[str, ...] | None = None) -> str:
    if not is_interactive_console():
        return str(default)
    raw = input(f"{label:<28} [{default}] : ").strip()
    value = str(default if raw == "" else raw).strip().lower()
    if allowed is not None and value not in allowed:
        raise ValueError(f"{label} 必須是 {allowed}，收到: {value}")
    return value


def _extract_cli_value(argv, option_name: str) -> str:
    args = list(argv or [])
    for idx in range(1, len(args)):
        raw = str(args[idx]).strip()
        if raw == option_name and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        if raw.startswith(option_name + "="):
            return raw.split("=", 1)[1].strip()
    return ""


def _has_cli_flag(argv, option_name: str) -> bool:
    return any(str(arg).strip() == option_name for arg in list(argv or [])[1:])


def _resolve_latest_year_from_dates(dates) -> int | None:
    years = []
    for raw_date in list(dates or []):
        year = int(getattr(raw_date, "year", 0) or 0)
        if year:
            years.append(year)
    return max(years) if years else None


def _resolve_config(argv, environ, *, base_policy: dict, latest_year: int | None, default_trials: int) -> OuterRollingConfig:
    env = os.environ if environ is None else environ
    train_start_default = int(base_policy.get("train_start_year", 2016) or 2016)
    first_oos_default = int(base_policy.get("oos_start_year") or base_policy.get("search_train_end_year", train_start_default + 4) + 1)
    last_oos_default = int(latest_year or first_oos_default)
    trials_default = int(default_trials if int(default_trials or 0) > 0 else int(env.get("V16_OUTER_ROLLING_OOS_TRIALS", "500") or 500))
    window_mode_default = str(env.get("V16_OUTER_ROLLING_WINDOW_MODE", "fixed") or "fixed").strip().lower()
    if window_mode_default not in ("fixed", "expanding"):
        window_mode_default = "fixed"
    train_window_default = max(1, int(env.get("V16_OUTER_ROLLING_TRAIN_WINDOW_YEARS", "5") or 5))

    cli_train_start = _extract_cli_value(argv, "--outer-train-start")
    cli_first = _extract_cli_value(argv, "--outer-first-oos")
    cli_last = _extract_cli_value(argv, "--outer-last-oos")
    cli_trials = _extract_cli_value(argv, "--trials")
    cli_window_mode = _extract_cli_value(argv, "--outer-window-mode")
    cli_train_window_years = _extract_cli_value(argv, "--outer-train-window-years")

    if cli_window_mode:
        window_mode = str(cli_window_mode).strip().lower()
        if window_mode not in ("fixed", "expanding"):
            raise ValueError("--outer-window-mode 必須是 fixed 或 expanding")
    else:
        window_mode = _prompt_str("window mode", window_mode_default, allowed=("fixed", "expanding"))

    if cli_train_window_years:
        train_window_years = int(cli_train_window_years)
    else:
        train_window_years = _prompt_int("train window years", train_window_default, minimum=1)

    if cli_train_start:
        train_start = int(cli_train_start)
    elif str(env.get("V16_OUTER_ROLLING_TRAIN_START", "")).strip():
        train_start = int(str(env["V16_OUTER_ROLLING_TRAIN_START"]).strip())
    else:
        train_start = _prompt_int("training start year", train_start_default, minimum=1900)

    if cli_first:
        first_oos = int(cli_first)
    elif str(env.get("V16_OUTER_ROLLING_FIRST_OOS", "")).strip():
        first_oos = int(str(env["V16_OUTER_ROLLING_FIRST_OOS"]).strip())
    else:
        first_oos = _prompt_int("first OOS year", first_oos_default, minimum=train_start + 1)

    if cli_last:
        last_oos = int(cli_last)
    elif str(env.get("V16_OUTER_ROLLING_LAST_OOS", "")).strip():
        last_oos = int(str(env["V16_OUTER_ROLLING_LAST_OOS"]).strip())
    else:
        last_oos = _prompt_int("last OOS year", last_oos_default, minimum=first_oos)

    if cli_trials:
        trials = int(cli_trials)
    else:
        trials = _prompt_int("optimizer trials per fold", trials_default, minimum=1)

    if first_oos > last_oos:
        raise ValueError("first OOS year 不可大於 last OOS year")
    if train_start >= first_oos:
        raise ValueError("training start year 必須小於 first OOS year")
    if train_window_years <= 0:
        raise ValueError("train window years 必須大於 0")
    if window_mode == "fixed" and first_oos - train_window_years < train_start:
        raise ValueError("fixed window 下 first OOS year - train_window_years 不可早於 training start year")
    if trials <= 0:
        raise ValueError("optimizer trials per fold 必須大於 0")
    return OuterRollingConfig(
        train_start,
        first_oos,
        last_oos,
        trials,
        window_mode=window_mode,
        train_window_years=train_window_years,
        confirm=not _has_cli_flag(argv, "--yes"),
    )


def _selection_start_for_oos(config: OuterRollingConfig, oos_year: int) -> int:
    if str(config.window_mode).lower() == "fixed":
        return int(oos_year) - int(config.train_window_years)
    return int(config.training_start_year)


def _print_plan(config: OuterRollingConfig):
    years = list(range(config.first_oos_year, config.last_oos_year + 1))
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print("OUTER ROLLING OOS TEST | VERY NEXT 1 YEAR")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print(f"window mode      : {config.window_mode}")
    if str(config.window_mode).lower() == "fixed":
        print(f"train window     : {config.train_window_years} years")
    else:
        print(f"training start   : {config.training_start_year}")
    print("oos feedback     : False")
    print("promotion        : disabled")
    print("oos horizon      : next 1 year only")
    print(f"optimizer trials : {config.trials_per_fold} per fold")
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    print(f"{'fold':<6} | {'selection period':<18} | {'OOS test period':<15}")
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    for idx, oos_year in enumerate(years, start=1):
        selection_start = _selection_start_for_oos(config, oos_year)
        print(f"{idx}/{len(years):<4} | {selection_start}~{oos_year - 1:<13} | {oos_year}")
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    print(f"LOCAL_MIN_SCORE              : True")
    print(f"INNER_VALIDATE_RANK          : {bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED)}")
    print(f"DOMINANT_YEAR_DEPENDENCY     : {bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED)}")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")


def _confirm_plan(config: OuterRollingConfig) -> bool:
    if not config.confirm or not is_interactive_console():
        return True
    choice = safe_prompt_choice("開始執行 outer rolling OOS？[Y/n] : ", "Y", ("Y", "N"), "outer rolling OOS 確認")
    return choice.upper() == "Y"


class _SearchProgress:
    def __init__(self, *, fold_idx: int, fold_count: int, oos_year: int, selection_start: int, selection_end: int, total_trials: int, completed_results: list[dict], overall_start: float):
        self.fold_idx = int(fold_idx)
        self.fold_count = int(fold_count)
        self.oos_year = int(oos_year)
        self.selection_start = int(selection_start)
        self.selection_end = int(selection_end)
        self.total_trials = int(total_trials)
        self.completed_results = completed_results
        self.overall_start = float(overall_start)
        self.stage_start = time.perf_counter()
        self.best_score = float("-inf")
        self.last_render = 0.0

    def _eta_stage(self, completed: int) -> float | None:
        if completed <= 0:
            return None
        elapsed = max(0.0, time.perf_counter() - self.stage_start)
        avg = elapsed / float(completed)
        return avg * max(0, self.total_trials - completed)

    def render(self, completed: int, *, force: bool = False):
        now = time.perf_counter()
        if not force and now - self.last_render < 0.5 and completed < self.total_trials:
            return
        self.last_render = now
        pct = 100.0 * float(completed) / max(1, self.total_trials)
        eta_stage = self._eta_stage(completed)
        elapsed_total = now - self.overall_start
        eta_total = None
        done_folds = len(self.completed_results)
        if done_folds > 0:
            avg_done = sum(float(row.get("elapsed_sec", 0.0)) for row in self.completed_results) / float(done_folds)
            current_remaining = eta_stage or 0.0
            eta_total = current_remaining + avg_done * max(0, self.fold_count - self.fold_idx)
        line = (
            f"\r[{self.fold_idx}/{self.fold_count}] OOS {self.oos_year} | selection={self.selection_start}~{self.selection_end} | "
            f"OPTIMIZER_SEARCH trial {completed}/{self.total_trials} ({pct:5.1f}%) | "
            f"best_score={self.best_score if self.best_score != float('-inf') else 0.0:.3f} | "
            f"elapsed={_fmt_duration(now - self.stage_start)} | eta_stage={_fmt_duration(eta_stage)} | eta_total={_fmt_duration(eta_total)}"
        )
        print(line + "\033[K", end="", flush=True)

    def callback(self, session):
        def _callback(study, trial):
            session.current_session_trial += 1
            if trial.value is not None and is_qualified_trial_value(trial.value):
                self.best_score = max(self.best_score, float(trial.value))
            self.render(int(session.current_session_trial))
        return _callback

    def done(self, completed: int):
        self.render(completed, force=True)
        print()


def _select_winner(finalists: list[dict], *, objective_mode: str):
    eligible = [item for item in finalists if bool(item.get("gate_pass", False))]
    if is_inner_validate_anti_overfit_enabled(objective_mode):
        eligible = [item for item in eligible if _has_inner_validate_pass(item)]
    if is_dominant_year_dependency_anti_overfit_enabled():
        safe = [item for item in eligible if not _has_dependency_warning(item)]
        if safe:
            eligible = safe
    if not eligible:
        return None
    eligible = sorted(
        eligible,
        key=lambda item: (
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("local_retention", float("-inf"))),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return eligible[0]


def _evaluate_next_1y_oos(*, session, trial, oos_year: int):
    payload = build_best_params_payload_from_trial(trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
    params = build_params_from_mapping(payload)
    prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(params, "optimizer_max_workers"))
    prep_result = prepare_trial_inputs(
        raw_data_cache=session.raw_data_cache,
        params=params,
        default_max_workers=session.default_max_workers,
        executor_bundle=prep_executor_bundle,
        static_fast_cache=session.static_fast_cache,
        static_master_dates=session.master_dates,
        include_trade_logs=True,
        include_pit_stats_index=True,
    )
    all_dates = sorted(prep_result["master_dates"])
    test_dates = [dt for dt in all_dates if int(getattr(dt, "year", 0) or 0) == int(oos_year)]
    if not test_dates:
        raise RuntimeError(f"OOS {oos_year} 無有效交易日期")
    holdout_period = {
        "label": f"OOS-{int(oos_year)}",
        "train_start": f"{int(session.train_start_year)}-01-01",
        "train_end": f"{int(oos_year) - 1}-12-31",
        "oos_start": pd.Timestamp(test_dates[0]).strftime("%Y-%m-%d"),
        "oos_end": pd.Timestamp(test_dates[-1]).strftime("%Y-%m-%d"),
        "test_dates": test_dates,
    }
    return evaluate_walk_forward(
        all_dfs_fast=prep_result["all_dfs_fast"],
        all_trade_logs=prep_result["all_trade_logs"],
        sorted_dates=all_dates,
        params=params,
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
        benchmark_ticker="0050",
        train_start_year=int(session.train_start_year),
        min_train_years=int(getattr(session, "walk_forward_policy", {}).get("min_train_years", 1) or 1),
        oos_start_year=int(oos_year),
        pit_stats_index=prep_result.get("all_pit_stats_index"),
        holdout_period=holdout_period,
    )


def _extract_period_metrics(report: dict) -> dict:
    period = dict(report.get("period") or {})
    return {
        "oos_score": float(period.get("test_score_romd", 0.0)),
        "ret_pct": float(period.get("ret_pct", 0.0)),
        "mdd_pct": float(period.get("mdd", 0.0)),
        "trades": int(period.get("trade_count", 0) or 0),
        "benchmark_oos_score": float(period.get("benchmark_score_romd", 0.0)),
    }


def _format_oos_delta(reference_score, selected_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    selected = _safe_float(selected_score, 0.0)
    return f"{ref:.3f} ({selected - ref:+.3f})"


def _evaluate_finalist_oos_diagnostics(*, session, finalists: list[dict], selected_trial_number: int, selected_report: dict, oos_year: int) -> dict:
    best_score = float("-inf")
    best_trial_number = None
    selected_metrics = _extract_period_metrics(selected_report)
    report_cache: dict[int, dict] = {int(selected_trial_number): selected_report}
    for item in list(finalists or []):
        trial = item.get("trial")
        if trial is None:
            continue
        trial_number = int(trial.number)
        report = report_cache.get(trial_number)
        if report is None:
            report = _evaluate_next_1y_oos(session=session, trial=trial, oos_year=int(oos_year))
            report_cache[trial_number] = report
        metrics = _extract_period_metrics(report)
        score = float(metrics["oos_score"])
        if score > best_score:
            best_score = score
            best_trial_number = trial_number
    if best_score == float("-inf"):
        best_score = float(selected_metrics["oos_score"])
        best_trial_number = int(selected_trial_number)
    return {
        "selected_metrics": selected_metrics,
        "best_finalist_oos_score": float(best_score),
        "best_finalist_trial": int(best_trial_number) + 1 if best_trial_number is not None else None,
        "benchmark_oos_score": float(selected_metrics.get("benchmark_oos_score", 0.0)),
    }


def _print_completed_results(rows: list[dict]):
    if not rows:
        return
    print("\nROLLING NEXT-1Y OOS RESULTS")
    print("-" * 132)
    print(
        f"{'fold':<5} | {'oos_year':<8} | {'selection':<11} | {'selected':<8} | "
        f"{'base_rank':<9} | {'local_rank':<10} | {'selected_oos':>12} | "
        f"{'best_finalist_oos':>22} | {'0050_oos':>18} | {'elapsed':>8}"
    )
    print("-" * 132)
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        selected_oos = float(row["oos_score"])
        best_text = _format_oos_delta(row.get("best_finalist_oos_score", selected_oos), selected_oos)
        benchmark_text = _format_oos_delta(row.get("benchmark_oos_score", 0.0), selected_oos)
        print(
            f"{idx}/{total:<3} | {int(row['oos_year']):<8} | {row['selection_period']:<11} | #{int(row['selected_trial']):<7} | "
            f"#{int(row['base_rank']):<8} | #{int(row['local_rank']):<9} | {selected_oos:>12.3f} | "
            f"{best_text:>22} | {benchmark_text:>18} | {_fmt_duration(row.get('elapsed_sec', 0.0)):>8}"
        )
    scores = [float(row["oos_score"]) for row in rows]
    benchmark_wins = sum(1 for row in rows if float(row["oos_score"]) > float(row.get("benchmark_oos_score", 0.0)))
    print("-" * 132)
    print(
        f"running median_oos={statistics.median(scores):.3f} | "
        f"worst_oos={min(scores):.3f} | positive_years={sum(1 for s in scores if s > 0)}/{len(scores)} | "
        f"win_vs_0050={benchmark_wins}/{len(rows)}"
    )


def _write_reports(*, output_dir: str, session_ts: str, rows: list[dict], config: OuterRollingConfig) -> dict:
    report_dir = os.path.join(output_dir, "outer_rolling_oos")
    os.makedirs(report_dir, exist_ok=True)
    base = os.path.join(report_dir, f"outer_rolling_oos_next1y_{session_ts}")
    json_path = base + ".json"
    csv_path = base + ".csv"
    txt_path = base + ".txt"
    payload = {
        "created_at": get_taipei_now().isoformat(),
        "mode": "outer_rolling_oos_next1y",
        "config": config.__dict__,
        "rows": rows,
        "summary": _build_summary(rows),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_format_final_report(rows, payload["summary"]))
    return {"json": json_path, "csv": csv_path, "txt": txt_path}


def _build_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"folds": 0}
    scores = [float(row["oos_score"]) for row in rows]
    best_gaps = [float(row["oos_score"]) - float(row.get("best_finalist_oos_score", row["oos_score"])) for row in rows]
    benchmark_gaps = [float(row["oos_score"]) - float(row.get("benchmark_oos_score", 0.0)) for row in rows]
    return {
        "folds": len(rows),
        "positive_years": sum(1 for score in scores if score > 0),
        "win_vs_0050": sum(1 for gap in benchmark_gaps if gap > 0),
        "avg_oos_score": sum(scores) / len(scores),
        "median_oos_score": float(statistics.median(scores)),
        "worst_oos_score": min(scores),
        "avg_gap_to_best_finalist": sum(best_gaps) / len(best_gaps),
        "median_gap_to_best_finalist": float(statistics.median(best_gaps)),
        "avg_gap_to_0050": sum(benchmark_gaps) / len(benchmark_gaps),
        "median_gap_to_0050": float(statistics.median(benchmark_gaps)),
    }


def _format_final_report(rows: list[dict], summary: dict) -> str:
    lines = []
    lines.append("=" * 132)
    lines.append("OUTER ROLLING OOS TEST | VERY NEXT 1 YEAR")
    lines.append("=" * 132)
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 132)
    if rows:
        lines.append(f"folds                    : {summary['folds']}")
        lines.append(f"positive_years           : {summary['positive_years']}/{summary['folds']}")
        lines.append(f"win_vs_0050              : {summary['win_vs_0050']}/{summary['folds']}")
        lines.append(f"avg_selected_oos         : {summary['avg_oos_score']:.3f}")
        lines.append(f"median_selected_oos*     : {summary['median_oos_score']:.3f}")
        lines.append(f"worst_selected_oos       : {summary['worst_oos_score']:.3f}")
        lines.append(f"avg_gap_to_best_finalist : {summary['avg_gap_to_best_finalist']:.3f}")
        lines.append(f"median_gap_to_0050       : {summary['median_gap_to_0050']:.3f}")
    lines.append("-" * 132)
    lines.append("ROLLING NEXT-1Y OOS RESULTS")
    lines.append("-" * 132)
    lines.append(
        f"{'fold':<5} | {'oos_year':<8} | {'selection':<11} | {'selected':<8} | "
        f"{'base_rank':<9} | {'local_rank':<10} | {'selected_oos':>12} | "
        f"{'best_finalist_oos':>22} | {'0050_oos':>18} | {'elapsed':>8}"
    )
    lines.append("-" * 132)
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        selected_oos = float(row["oos_score"])
        best_text = _format_oos_delta(row.get("best_finalist_oos_score", selected_oos), selected_oos)
        benchmark_text = _format_oos_delta(row.get("benchmark_oos_score", 0.0), selected_oos)
        lines.append(
            f"{idx}/{total:<3} | {int(row['oos_year']):<8} | {row['selection_period']:<11} | #{int(row['selected_trial']):<7} | "
            f"#{int(row['base_rank']):<8} | #{int(row['local_rank']):<9} | {selected_oos:>12.3f} | "
            f"{best_text:>22} | {benchmark_text:>18} | {_fmt_duration(row.get('elapsed_sec', 0.0)):>8}"
        )
    lines.append("-" * 132)
    lines.append("best_finalist_oos / 0050_oos 括號內 = selected_oos - 對照分數；diagnostic only，不參與 selection。")
    lines.append("oos_feedback_used : False")
    return "\n".join(lines) + "\n"


def run_outer_rolling_oos(
    *,
    argv,
    environ,
    project_root: str,
    output_dir: str,
    base_policy: dict,
    selected_data_dir: str,
    dataset_label: str,
    load_all_raw_data,
    optimizer_required_min_rows: int,
    build_optimizer_session,
    create_optimizer_study,
    ensure_study_effective_policy_compatible,
    configure_optuna_logging,
    optimizer_seed=None,
    default_trials: int = 500,
) -> int:
    from tools.optimizer.session import close_study_storage

    session_ts = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(output_dir, "outer_rolling_oos"), exist_ok=True)

    probe_policy = build_optimizer_runtime_policy(dict(base_policy), "split")
    probe_session = build_optimizer_session(walk_forward_policy=probe_policy)
    probe_session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
    latest_year = _resolve_latest_year_from_dates(probe_session.sorted_master_dates)
    probe_session.close_trial_prep_executor()

    config = _resolve_config(argv, environ, base_policy=base_policy, latest_year=latest_year, default_trials=default_trials)
    _print_plan(config)
    if not _confirm_plan(config):
        print(f"{C_YELLOW}已取消 outer rolling OOS。{C_RESET}")
        return 0

    configure_optuna_logging()
    rows: list[dict] = []
    years = list(range(config.first_oos_year, config.last_oos_year + 1))
    overall_start = time.perf_counter()
    print(f"{C_CYAN}開始 outer rolling OOS：資料集={dataset_label} | folds={len(years)} | trials/fold={config.trials_per_fold}{C_RESET}")

    for fold_idx, oos_year in enumerate(years, start=1):
        fold_start = time.perf_counter()
        selection_end = int(oos_year) - 1
        selection_start = _selection_start_for_oos(config, int(oos_year))
        fold_policy = dict(base_policy)
        fold_policy["selection_start_year"] = int(selection_start)
        fold_policy["train_start_year"] = int(selection_start)
        fold_policy["search_train_end_year"] = int(selection_end)
        fold_policy["oos_start_year"] = int(oos_year)
        fold_policy = build_optimizer_runtime_policy(fold_policy, "split")
        objective_mode = str(fold_policy.get("objective_mode", "split_train_romd"))
        session = build_optimizer_session(walk_forward_policy=fold_policy)
        session.n_trials = int(config.trials_per_fold)
        session.disable_milestone_dashboard = True
        session.timing_mode = False
        db_dir = os.path.join(output_dir, "outer_rolling_oos", "db")
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, f"outer_oos_{session_ts}_{int(oos_year)}.db")
        db_name = f"sqlite:///{db_file}"
        study = None
        try:
            print(f"\n{C_CYAN}[{fold_idx}/{len(years)}] OOS {oos_year} | selection={config.training_start_year}~{selection_end}{C_RESET}")
            session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
            session.profile_recorder.init_output_files()
            session.profile_recorder.mark_run_started()
            study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind="tpe")
            ensure_study_effective_policy_compatible(study=study, walk_forward_policy=fold_policy)
            progress = _SearchProgress(
                fold_idx=fold_idx,
                fold_count=len(years),
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                total_trials=config.trials_per_fold,
                completed_results=rows,
                overall_start=overall_start,
            )
            study.optimize(session.objective, n_trials=int(config.trials_per_fold), n_jobs=1, callbacks=[progress.callback(session)])
            progress.done(int(session.current_session_trial))

            local_started = time.perf_counter()
            session.outer_rolling_local_progress_context = {
                "fold_idx": int(fold_idx),
                "fold_count": int(len(years)),
                "oos_year": int(oos_year),
                "selection_start": int(selection_start),
                "selection_end": int(selection_end),
                "completed_results": rows,
                "overall_start": overall_start,
            }
            finalists = list_local_min_score_finalists(
                study,
                session=session,
                objective_mode=objective_mode,
                include_trial=None,
                show_progress=True,
                include_oos_diagnostics=False,
            )
            if hasattr(session, "outer_rolling_local_progress_context"):
                delattr(session, "outer_rolling_local_progress_context")
            local_elapsed = time.perf_counter() - local_started
            print(
                f"[{fold_idx}/{len(years)}] OOS {oos_year} | LOCAL_MIN_REVIEW DONE | "
                f"finalists={len(finalists)} | best_local={float(finalists[0].get('local_min_score', 0.0)) if finalists else 0.0:.3f} "
                f"#{int(finalists[0]['trial'].number) + 1 if finalists else 0} | elapsed={_fmt_duration(local_elapsed)}"
            )
            winner = _select_winner(finalists, objective_mode=objective_mode)
            if winner is None:
                print(f"{C_YELLOW}[{fold_idx}/{len(years)}] OOS {oos_year} | 無通過 gate 的候選，略過。{C_RESET}")
                continue
            local_rank_map = {int(item["trial"].number): rank for rank, item in enumerate(finalists, start=1)}
            trial = winner["trial"]
            report = _evaluate_next_1y_oos(session=session, trial=trial, oos_year=int(oos_year))
            diagnostics = _evaluate_finalist_oos_diagnostics(
                session=session,
                finalists=finalists,
                selected_trial_number=int(trial.number),
                selected_report=report,
                oos_year=int(oos_year),
            )
            metrics = dict(diagnostics["selected_metrics"])
            fold_elapsed = time.perf_counter() - fold_start
            row = {
                "oos_year": int(oos_year),
                "selection_period": f"{selection_start}~{selection_end}",
                "selected_trial": int(trial.number) + 1,
                "base_score": float(winner.get("base_score", INVALID_TRIAL_VALUE)),
                "base_rank": int(winner.get("base_rank", 0)),
                "local_min": float(winner.get("local_min_score", INVALID_TRIAL_VALUE)),
                "local_rank": int(local_rank_map.get(int(trial.number), 0)),
                "retention": float(winner.get("local_retention", 0.0)),
                "local_gate": bool(winner.get("gate_pass", False)),
                "oos_score": float(metrics.get("oos_score", 0.0)),
                "best_finalist_oos_score": float(diagnostics.get("best_finalist_oos_score", metrics.get("oos_score", 0.0))),
                "best_finalist_trial": diagnostics.get("best_finalist_trial"),
                "benchmark_oos_score": float(diagnostics.get("benchmark_oos_score", metrics.get("benchmark_oos_score", 0.0))),
                "ret_pct": float(metrics.get("ret_pct", 0.0)),
                "mdd_pct": float(metrics.get("mdd_pct", 0.0)),
                "trades": int(metrics.get("trades", 0) or 0),
                "elapsed_sec": float(fold_elapsed),
                "optimizer_search_sec": float(max(0.0, fold_elapsed - local_elapsed)),
                "local_min_review_sec": float(local_elapsed),
            }
            if is_dominant_year_dependency_anti_overfit_enabled():
                row["dep_gate"] = not _has_dependency_warning(winner)
            rows.append(row)
            print(f"{C_GREEN}[{fold_idx}/{len(years)}] OOS {oos_year} | DONE | selected=#{row['selected_trial']} | selected_oos={row['oos_score']:.3f} | best_finalist_oos={_format_oos_delta(row['best_finalist_oos_score'], row['oos_score'])} | 0050_oos={_format_oos_delta(row['benchmark_oos_score'], row['oos_score'])} | elapsed={_fmt_duration(fold_elapsed)}{C_RESET}")
            _print_completed_results(rows)
        finally:
            session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    paths = _write_reports(output_dir=output_dir, session_ts=session_ts, rows=rows, config=config)
    print(f"\n{C_CYAN}{'=' * 100}{C_RESET}")
    print("FINAL REPORT")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print(_format_final_report(rows, _build_summary(rows)))
    print(f"{C_GREEN}已輸出：{paths['txt']}{C_RESET}")
    print(f"{C_GREEN}已輸出：{paths['json']}{C_RESET}")
    print(f"{C_GREEN}已輸出：{paths['csv']}{C_RESET}")
    return 0
