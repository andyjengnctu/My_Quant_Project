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
from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import build_normal_setup_index
from core.portfolio_stats import calc_portfolio_score
from core.runtime_utils import choose_inline_progress_message, get_taipei_now, is_interactive_console, safe_prompt_choice, stdout_supports_inline_progress, write_inline_progress
from core.rolling_oos_params import ROLLING_OOS_PARAM_SET_SCHEMA_TYPE, ROLLING_OOS_USAGE
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


POLICY_NAMES = ("base", "local", "retention")
CHAIN_SCHEDULE_NAMES = ("best", "base", "local", "retention")


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


def _fmt_duration_compact(seconds: float | int | None) -> str:
    text = _fmt_duration(seconds)
    if text.startswith("00:"):
        return text[3:]
    return text


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", str(text))


def _visible_len(text: str) -> int:
    return len(_strip_ansi(str(text)))


def _pad_ansi(text: str, width: int, *, align: str = "<") -> str:
    raw = str(text)
    pad = max(0, int(width) - _visible_len(raw))
    if align == ">":
        return " " * pad + raw
    return raw + " " * pad


def _color_numeric_text(text: str, value: float | int | None) -> str:
    if value is None:
        return str(text)
    v = _safe_float(value, 0.0)
    if v > 0.0:
        return f"{C_GREEN}{text}{C_RESET}"
    if v < 0.0:
        return f"{C_RED}{text}{C_RESET}"
    return str(text)


def _format_score(value) -> str:
    v = _safe_float(value, 0.0)
    return _color_numeric_text(f"{v:.3f}", v)


def _format_compare(reference_score, rank_1_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    rank_1 = _safe_float(rank_1_score, 0.0)
    gap = rank_1 - ref
    return f"{_color_numeric_text(f'{ref:.3f}', ref)} ({_color_numeric_text(f'{gap:+.3f}', gap)})"


def _format_compare_plain(reference_score, rank_1_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    rank_1 = _safe_float(rank_1_score, 0.0)
    return f"{ref:.3f} ({rank_1 - ref:+.3f})"


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


def _resolve_latest_year_from_csv_data_dir(data_dir: str) -> int | None:
    # AI註: outer rolling OOS 的互動設定只需要 last OOS 預設值；
    # 不應為此先觸發 optimizer 完整資料清洗、快取摘要與 issue log。
    if not os.path.isdir(str(data_dir)):
        return None
    try:
        from core.data_utils import discover_unique_csv_inputs
        csv_inputs, _duplicate_file_issue_lines = discover_unique_csv_inputs(str(data_dir))
    except (OSError, ValueError, TypeError):
        return None

    latest_year = None
    date_column_names = {"date", "datetime", "time", "timestamp", "日期"}
    for _ticker, file_path in list(csv_inputs or []):
        try:
            columns = list(pd.read_csv(file_path, nrows=0).columns)
            date_col = next((col for col in columns if str(col).strip().lower() in date_column_names), None)
            if date_col is None:
                continue
            date_values = pd.read_csv(file_path, usecols=[date_col])[date_col]
            if date_values.empty:
                continue
            parsed_dates = pd.to_datetime(date_values, errors="coerce")
            if parsed_dates.isna().all():
                continue
            file_year = int(parsed_dates.dt.year.max())
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError):
            continue
        if latest_year is None or file_year > latest_year:
            latest_year = file_year
    return latest_year


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
        self.inline_progress_enabled = stdout_supports_inline_progress()
        self.inline_progress_width = 0

    def _eta_stage(self, completed: int) -> float | None:
        if completed <= 0:
            return None
        elapsed = max(0.0, time.perf_counter() - self.stage_start)
        avg = elapsed / float(completed)
        return avg * max(0, self.total_trials - completed)

    def render(self, completed: int, *, force: bool = False):
        if not self.inline_progress_enabled:
            return
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
        best_score = self.best_score if self.best_score != float("-inf") else 0.0
        elapsed_text = _fmt_duration_compact(now - self.stage_start)
        eta_stage_text = _fmt_duration_compact(eta_stage)
        eta_total_text = _fmt_duration_compact(eta_total)
        line = choose_inline_progress_message((
            (
                f"[{self.fold_idx}/{self.fold_count}] selection={self.selection_start}~{self.selection_end} | OOS={self.oos_year} | "
                f"OPTIMIZER_SEARCH | 進度={completed}/{self.total_trials} ({pct:5.1f}%) | "
                f"best_score={best_score:.3f} | elapsed={elapsed_text} | eta={eta_stage_text}/{eta_total_text}"
            ),
            (
                f"[{self.fold_idx}/{self.fold_count}] selection={self.selection_start % 100:02d}~{self.selection_end % 100:02d} | OOS={self.oos_year % 100:02d} | "
                f"search | 進度={completed}/{self.total_trials} ({pct:5.1f}%) | "
                f"best={best_score:.3f} | elapsed={elapsed_text} | eta={eta_stage_text}/{eta_total_text}"
            ),
            (
                f"[{self.fold_idx}/{self.fold_count}] {self.selection_start % 100:02d}~{self.selection_end % 100:02d}>OOS{self.oos_year % 100:02d} | "
                f"search {completed}/{self.total_trials} | best={best_score:.3f} | eta={eta_stage_text}/{eta_total_text}"
            ),
        ))
        self.inline_progress_width = write_inline_progress(line, previous_width=self.inline_progress_width)

    def callback(self, session):
        def _callback(study, trial):
            session.current_session_trial += 1
            if trial.value is not None and is_qualified_trial_value(trial.value):
                self.best_score = max(self.best_score, float(trial.value))
            self.render(int(session.current_session_trial))
        return _callback

    def done(self, completed: int):
        if self.inline_progress_enabled:
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


def _build_local_rank_map(finalists: list[dict]) -> dict[int, int]:
    return {int(item["trial"].number): rank for rank, item in enumerate(finalists or [], start=1) if item.get("trial") is not None}


def _build_retention_rank_map(finalists: list[dict]) -> dict[int, int]:
    ranked = sorted(
        [item for item in list(finalists or []) if item.get("trial") is not None],
        key=lambda item: (
            float(item.get("local_retention", float("-inf"))),
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return {int(item["trial"].number): rank for rank, item in enumerate(ranked, start=1)}


def _select_base_rank1_item(finalists: list[dict]):
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    if not items:
        return None
    return min(items, key=lambda item: (int(item.get("base_rank", 10**9) or 10**9), -float(item.get("base_score", INVALID_TRIAL_VALUE)), int(item["trial"].number)))


def _select_local_rank1_item(finalists: list[dict], *, objective_mode: str):
    winner = _select_winner(finalists, objective_mode=objective_mode)
    if winner is not None:
        return winner
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    return items[0] if items else None


def _select_retention_rank1_item(finalists: list[dict]):
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    if not items:
        return None
    return max(
        items,
        key=lambda item: (
            float(item.get("local_retention", float("-inf"))),
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
    )


def _build_policy_items(finalists: list[dict], *, objective_mode: str) -> dict[str, dict | None]:
    return {
        "base": _select_base_rank1_item(finalists),
        "local": _select_local_rank1_item(finalists, objective_mode=objective_mode),
        "retention": _select_retention_rank1_item(finalists),
    }


def _policy_description(policy_name: str) -> str:
    if policy_name == "base":
        return "Use base_rank #1 params for each OOS year."
    if policy_name == "local":
        return "Use local_rank #1 params for each OOS year."
    if policy_name == "retention":
        return "Use retention_rank #1 params for each OOS year."
    return f"Use {policy_name} params for each OOS year."


def _build_policy_schedule_entry(*, item: dict, policy_name: str, oos_year: int, selection_period: str, local_rank_map: dict[int, int], retention_rank_map: dict[int, int]) -> dict:
    trial = item["trial"]
    trial_number = int(trial.number)
    return {
        "effective_start": f"{int(oos_year)}-01-01",
        "effective_end": f"{int(oos_year)}-12-31",
        "selection": str(selection_period),
        "oos_year": int(oos_year),
        "policy": str(policy_name),
        "selected_trial": trial_number + 1,
        "base_score": float(item.get("base_score", INVALID_TRIAL_VALUE)),
        "base_rank": int(item.get("base_rank", 0) or 0),
        "local_min": float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
        "local_rank": int(local_rank_map.get(trial_number, 0)),
        "retention": float(item.get("local_retention", 0.0)),
        "retention_rank": int(retention_rank_map.get(trial_number, 0)),
        "params": build_best_params_payload_from_trial(trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT),
    }


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
        "benchmark_return_pct": float(period.get("benchmark_return_pct", 0.0)),
        "benchmark_mdd_pct": float(period.get("benchmark_mdd", 0.0)),
    }


def _format_oos_delta(reference_score, selected_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    selected = _safe_float(selected_score, 0.0)
    return f"{ref:.3f} ({selected - ref:+.3f})"


def _evaluate_finalist_oos_diagnostics(*, session, finalists: list[dict], policy_items: dict[str, dict | None], oos_year: int) -> dict:
    best_score = float("-inf")
    best_trial_number = None
    best_item = None
    best_metrics: dict = {}
    report_cache: dict[int, dict] = {}
    metrics_by_trial: dict[int, dict] = {}
    benchmark_score = 0.0
    benchmark_return_pct = 0.0
    benchmark_mdd_pct = 0.0
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
        metrics_by_trial[trial_number] = metrics
        benchmark_score = float(metrics.get("benchmark_oos_score", benchmark_score))
        benchmark_return_pct = float(metrics.get("benchmark_return_pct", benchmark_return_pct))
        benchmark_mdd_pct = float(metrics.get("benchmark_mdd_pct", benchmark_mdd_pct))
        score = float(metrics["oos_score"])
        if score > best_score:
            best_score = score
            best_trial_number = trial_number
            best_item = item
            best_metrics = dict(metrics)
    if best_score == float("-inf"):
        best_score = 0.0
        best_metrics = {}
    policies: dict[str, dict] = {}
    for policy_name, item in dict(policy_items or {}).items():
        if item is None or item.get("trial") is None:
            policies[policy_name] = {
                "rank_1_trial": None,
                "rank_1_oos": 0.0,
                "rank_1_return_pct": 0.0,
                "rank_1_mdd_pct": 0.0,
                "rank_1_trades": 0,
                "best_gap": 0.0 - float(best_score),
                "benchmark_0050_gap": 0.0 - float(benchmark_score),
            }
            continue
        trial_number = int(item["trial"].number)
        if trial_number not in metrics_by_trial:
            report = _evaluate_next_1y_oos(session=session, trial=item["trial"], oos_year=int(oos_year))
            metrics_by_trial[trial_number] = _extract_period_metrics(report)
        metrics = dict(metrics_by_trial[trial_number])
        rank_1_oos = float(metrics.get("oos_score", 0.0))
        benchmark_score = float(metrics.get("benchmark_oos_score", benchmark_score))
        benchmark_return_pct = float(metrics.get("benchmark_return_pct", benchmark_return_pct))
        benchmark_mdd_pct = float(metrics.get("benchmark_mdd_pct", benchmark_mdd_pct))
        policies[policy_name] = {
            "rank_1_trial": trial_number + 1,
            "rank_1_oos": rank_1_oos,
            "rank_1_return_pct": float(metrics.get("ret_pct", 0.0)),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_trades": int(metrics.get("trades", 0) or 0),
            "best_gap": rank_1_oos - float(best_score),
            "benchmark_0050_gap": rank_1_oos - float(benchmark_score),
        }
    return {
        "best_finalist_oos_score": float(best_score),
        "best_finalist_trial": int(best_trial_number) + 1 if best_trial_number is not None else None,
        "_best_finalist_item": best_item,
        "best_finalist_return_pct": float(best_metrics.get("ret_pct", 0.0)),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "best_finalist_trades": int(best_metrics.get("trades", 0) or 0),
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(benchmark_return_pct),
        "benchmark_mdd_pct": float(benchmark_mdd_pct),
        "policies": policies,
    }


def _compound_return_pct(return_pcts: list[float]) -> float:
    equity = 1.0
    for value in list(return_pcts or []):
        equity *= 1.0 + float(value) / 100.0
    return (equity - 1.0) * 100.0


def _build_chained_oos_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    first_year = min(int(row["oos_year"]) for row in rows)
    last_year = max(int(row["oos_year"]) for row in rows)
    selection_start = min(int(row.get("selection_start_year", str(row.get("selection_period", "0~0")).split("~", 1)[0])) for row in rows)
    selection_end = max(int(row.get("selection_end_year", str(row.get("selection_period", "0~0")).split("~", 1)[-1])) for row in rows)
    best_return = _compound_return_pct([float(row.get("best_finalist_return_pct", 0.0)) for row in rows])
    benchmark_return = _compound_return_pct([float(row.get("benchmark_return_pct", 0.0)) for row in rows])
    summary = {
        "method": "compound_yearly_oos_returns",
        "note": "由各年度 next-1Y OOS 報酬按時間順序複利串接；不是年度 score 平均。完整實盤口徑請用 rolling paramset 在 portfolio/workbench 進行 active-param replay。",
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_period": f"{first_year}~{last_year}",
        "best_finalist_return_pct": float(best_return),
        "benchmark_return_pct": float(benchmark_return),
        "benchmark_alpha_pct": float(best_return - benchmark_return),
    }
    for policy_name in ("base", "local", "retention"):
        returns = [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows]
        chain_return = _compound_return_pct(returns)
        summary[policy_name] = {
            "rank_1_return_pct": float(chain_return),
            "best_gap_pct": float(chain_return - best_return),
            "benchmark_0050_gap_pct": float(chain_return - benchmark_return),
            "yearly_return_pct": returns,
        }
    return summary


def _calc_romd_score(ret_pct: float, mdd_pct: float) -> float:
    return float(ret_pct) / (abs(float(mdd_pct)) + 0.0001)


def _build_chained_oos_row(chained: dict | None) -> dict | None:
    if not chained:
        return None
    row = {
        "fold": "OOS_CHAIN",
        "selection_period": chained.get("selection_period", ""),
        "oos_year": chained.get("oos_period", ""),
        "best_finalist_oos_score": float(chained.get("best_finalist_oos_score", 0.0)),
        "best_finalist_return_pct": float(chained.get("best_finalist_return_pct", 0.0)),
        "best_finalist_mdd_pct": float(chained.get("best_finalist_mdd_pct", 0.0)),
        "best_finalist_trades": int(chained.get("best_finalist_trades", 0) or 0),
        "benchmark_oos_score": float(chained.get("benchmark_oos_score", 0.0)),
        "benchmark_return_pct": float(chained.get("benchmark_return_pct", 0.0)),
        "benchmark_mdd_pct": float(chained.get("benchmark_mdd_pct", 0.0)),
        "elapsed_sec": float(chained.get("elapsed_sec", 0.0)),
        "aggregation_method": chained.get("method"),
    }
    best_score = float(row["best_finalist_oos_score"])
    benchmark_score = float(row["benchmark_oos_score"])
    for policy_name in POLICY_NAMES:
        item = dict(chained.get(policy_name) or {})
        rank_score = float(item.get("rank_1_oos", 0.0))
        row[policy_name] = {
            "rank_1_trial": None,
            "rank_1_oos": rank_score,
            "rank_1_return_pct": float(item.get("rank_1_return_pct", 0.0)),
            "rank_1_mdd_pct": float(item.get("rank_1_mdd_pct", 0.0)),
            "rank_1_trades": int(item.get("rank_1_trades", 0) or 0),
            "best_gap": rank_score - best_score,
            "benchmark_0050_gap": rank_score - benchmark_score,
        }
    return row


def _coerce_replay_date(value: Any):
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "date"):
        return value.date()
    return pd.Timestamp(value).date()


def _params_payload_signature(payload: dict) -> str:
    return json.dumps(dict(payload or {}), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _build_chain_schedule_entries(rows: list[dict], schedule_name: str) -> list[dict]:
    entries: list[dict] = []
    for row in sorted(list(rows or []), key=lambda item: int(item.get("oos_year", 0) or 0)):
        schedule = dict((row.get("policy_schedules") or {}).get(schedule_name) or {})
        if not schedule:
            continue
        entries.append(schedule)
    return entries


def _prepare_active_replay_context(*, session, params, context_cache: dict[str, dict]) -> dict:
    params_payload = build_runtime_param_raw_value(params, "optimizer_max_workers")
    max_workers = params_payload
    signature = _params_payload_signature(build_best_params_payload_from_params(params))
    cached = context_cache.get(signature)
    if cached is not None:
        return cached
    prep_executor_bundle = session.get_trial_prep_executor_bundle(max_workers)
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
    context = {
        "all_dfs_fast": prep_result["all_dfs_fast"],
        "all_trade_logs": prep_result["all_trade_logs"],
        "all_pit_stats_index": prep_result.get("all_pit_stats_index") or {},
        "normal_setup_index": build_normal_setup_index(prep_result["all_dfs_fast"]),
        "master_dates": set(prep_result.get("master_dates") or []),
        "prep_wall_sec": float(prep_result.get("prep_wall_sec", 0.0)),
        "prep_mode": prep_result.get("prep_mode"),
    }
    context_cache[signature] = context
    return context


def _build_best_params_payload_from_schedule(schedule: dict) -> dict:
    return dict(schedule.get("params") or {})


def build_best_params_payload_from_params(params) -> dict:
    from core.params_io import params_to_json_dict

    payload = params_to_json_dict(params)
    payload["fixed_tp_percent"] = float(OPTIMIZER_FIXED_TP_PERCENT)
    return payload


def _build_active_replay_records(*, session, schedule_entries: list[dict], context_cache: dict[str, dict]) -> list[dict]:
    records: list[dict] = []
    for schedule in list(schedule_entries or []):
        params_payload = _build_best_params_payload_from_schedule(schedule)
        params = build_params_from_mapping(params_payload)
        context = _prepare_active_replay_context(session=session, params=params, context_cache=context_cache)
        effective_start = str(schedule.get("effective_start") or f"{int(schedule.get('oos_year'))}-01-01")
        effective_date = _coerce_replay_date(effective_start)
        records.append({
            "effective_date": effective_date,
            "effective_date_text": effective_date.isoformat(),
            "year": int(schedule.get("oos_year") or effective_date.year),
            "params_obj": params,
            "context": context,
            "schedule": dict(schedule),
        })
    records.sort(key=lambda item: item["effective_date"])
    return records


def _resolve_active_replay_record(records: list[dict], trade_date) -> dict:
    current_date = _coerce_replay_date(trade_date)
    selected = None
    for record in records:
        if record["effective_date"] <= current_date:
            selected = record
        else:
            break
    if selected is None:
        first_date = records[0]["effective_date_text"] if records else "N/A"
        raise RuntimeError(f"{current_date.isoformat()} is before first active replay param date {first_date}")
    return selected


def _evaluate_schedule_active_replay(*, session, schedule_entries: list[dict], first_oos_year: int, last_oos_year: int, context_cache: dict[str, dict]) -> dict:
    records = _build_active_replay_records(session=session, schedule_entries=schedule_entries, context_cache=context_cache)
    if not records:
        return {}
    merged_dates = set()
    for record in records:
        merged_dates.update(record["context"].get("master_dates") or [])
    test_dates = [
        dt for dt in sorted(merged_dates)
        if int(first_oos_year) <= int(getattr(dt, "year", 0) or 0) <= int(last_oos_year)
    ]
    if not test_dates:
        raise RuntimeError(f"active-param replay has no market dates for OOS {first_oos_year}~{last_oos_year}")

    first_record = _resolve_active_replay_record(records, test_dates[0])
    base_context = first_record["context"]
    base_params = first_record["params_obj"]
    benchmark_ticker = "0050"
    benchmark_data = (base_context.get("all_dfs_fast") or {}).get(benchmark_ticker)

    def active_params_resolver(trade_date):
        return _resolve_active_replay_record(records, trade_date)["params_obj"]

    def active_context_resolver(trade_date):
        return _resolve_active_replay_record(records, trade_date)["context"]

    pf_profile = {"_timing_enabled": False}
    (
        ret_pct,
        mdd,
        trade_count,
        final_eq,
        avg_exp,
        max_exp,
        bm_ret,
        bm_mdd,
        win_rate,
        pf_ev,
        pf_payoff,
        total_missed,
        total_missed_sells,
        r_sq,
        m_win_rate,
        bm_r_sq,
        bm_m_win_rate,
        normal_trade_count,
        extended_trade_count,
        annual_trades,
        reserved_buy_fill_rate,
        annual_return_pct,
        bm_annual_return_pct,
    ) = run_portfolio_timeline(
        base_context.get("all_dfs_fast") or {},
        base_context.get("all_trade_logs") or {},
        test_dates,
        int(first_oos_year),
        base_params,
        session.train_max_positions,
        session.train_enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=benchmark_data,
        is_training=True,
        profile_stats=pf_profile,
        verbose=False,
        pit_stats_index=base_context.get("all_pit_stats_index"),
        active_params_resolver=active_params_resolver,
        active_context_resolver=active_context_resolver,
    )
    oos_score = calc_portfolio_score(ret_pct, mdd, m_win_rate, r_sq, annual_return_pct=annual_return_pct)
    return {
        "oos_score": float(oos_score),
        "return_pct": float(ret_pct),
        "annual_return_pct": float(annual_return_pct),
        "mdd_pct": float(mdd),
        "trades": int(trade_count),
        "normal_trades": int(normal_trade_count),
        "extended_trades": int(extended_trade_count),
        "annual_trades": float(annual_trades),
        "reserved_buy_fill_rate": float(reserved_buy_fill_rate),
        "win_rate": float(win_rate),
        "pf_payoff": float(pf_payoff),
        "pf_ev": float(pf_ev),
        "r_squared": float(r_sq),
        "monthly_win_rate": float(m_win_rate),
        "final_equity": float(final_eq),
        "avg_exposure": float(avg_exp),
        "max_exposure": float(max_exp),
        "missed_buys": int(total_missed),
        "missed_sells": int(total_missed_sells),
        "benchmark_oos_score": float(_calc_romd_score(bm_ret, bm_mdd)),
        "benchmark_return_pct": float(bm_ret),
        "benchmark_annual_return_pct": float(bm_annual_return_pct),
        "benchmark_mdd_pct": float(bm_mdd),
        "benchmark_r_squared": float(bm_r_sq),
        "benchmark_monthly_win_rate": float(bm_m_win_rate),
        "yearly_return_rows": list(pf_profile.get("yearly_return_rows") or []),
        "benchmark_yearly_return_rows": list(pf_profile.get("bm_yearly_return_rows") or []),
        "active_param_schedule": [
            {
                "effective_date": record["effective_date_text"],
                "oos_year": int(record["year"]),
                "selected_trial": record["schedule"].get("selected_trial"),
                "selection": record["schedule"].get("selection"),
            }
            for record in records
        ],
    }


def _evaluate_active_param_chain_replays(
    *,
    rows: list[dict],
    config: OuterRollingConfig,
    base_policy: dict,
    selected_data_dir: str,
    load_all_raw_data,
    optimizer_required_min_rows: int,
    build_optimizer_session,
) -> dict:
    if not rows:
        return {}
    first_year = min(int(row["oos_year"]) for row in rows)
    last_year = max(int(row["oos_year"]) for row in rows)
    selection_start = min(int(row.get("selection_start_year", 0) or 0) for row in rows)
    selection_end = max(int(row.get("selection_end_year", 0) or 0) for row in rows)
    replay_policy = build_optimizer_runtime_policy(dict(base_policy), "split")
    session = build_optimizer_session(walk_forward_policy=replay_policy)
    started = time.perf_counter()
    try:
        session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
        context_cache: dict[str, dict] = {}
        replay_by_name: dict[str, dict] = {}
        for schedule_name in CHAIN_SCHEDULE_NAMES:
            entries = _build_chain_schedule_entries(rows, schedule_name)
            if not entries:
                continue
            replay_by_name[schedule_name] = _evaluate_schedule_active_replay(
                session=session,
                schedule_entries=entries,
                first_oos_year=first_year,
                last_oos_year=last_year,
                context_cache=context_cache,
            )
    finally:
        session.close_trial_prep_executor()

    best_metrics = dict(replay_by_name.get("best") or {})
    if not best_metrics:
        policy_metrics = [dict(replay_by_name.get(policy_name) or {}) for policy_name in POLICY_NAMES]
        best_metrics = max(policy_metrics, key=lambda item: float(item.get("oos_score", float("-inf"))), default={})
    benchmark_score = float(best_metrics.get("benchmark_oos_score", 0.0))
    benchmark_return = float(best_metrics.get("benchmark_return_pct", 0.0))
    best_score = float(best_metrics.get("oos_score", 0.0))
    best_return = float(best_metrics.get("return_pct", 0.0))
    replay_elapsed = time.perf_counter() - started
    total_elapsed = sum(float(row.get("elapsed_sec", 0.0)) for row in rows) + replay_elapsed
    summary = {
        "method": "active_param_replay_oos_score",
        "note": "Continuous portfolio replay over completed OOS years using each year's selected active params; OOS_CHAIN values are OOS scores, not compounded returns.",
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_period": f"{first_year}~{last_year}",
        "best_finalist_oos_score": best_score,
        "best_finalist_return_pct": best_return,
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "best_finalist_trades": int(best_metrics.get("trades", 0) or 0),
        "benchmark_oos_score": benchmark_score,
        "benchmark_return_pct": benchmark_return,
        "benchmark_mdd_pct": float(best_metrics.get("benchmark_mdd_pct", 0.0)),
        "elapsed_sec": total_elapsed,
        "replay_elapsed_sec": replay_elapsed,
        "window_mode": str(config.window_mode),
        "train_window_years": int(config.train_window_years),
    }
    for policy_name in POLICY_NAMES:
        metrics = dict(replay_by_name.get(policy_name) or {})
        policy_score = float(metrics.get("oos_score", 0.0))
        policy_return = float(metrics.get("return_pct", 0.0))
        summary[policy_name] = {
            "rank_1_oos": policy_score,
            "rank_1_return_pct": policy_return,
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_trades": int(metrics.get("trades", 0) or 0),
            "best_gap": policy_score - best_score,
            "benchmark_0050_gap": policy_score - benchmark_score,
            "return_gap_vs_best_pct": policy_return - best_return,
            "return_gap_vs_0050_pct": policy_return - benchmark_return,
            "annual_return_pct": float(metrics.get("annual_return_pct", 0.0)),
            "annual_trades": float(metrics.get("annual_trades", 0.0)),
            "reserved_buy_fill_rate": float(metrics.get("reserved_buy_fill_rate", 0.0)),
            "yearly_return_rows": list(metrics.get("yearly_return_rows") or []),
            "active_param_schedule": list(metrics.get("active_param_schedule") or []),
        }
    return summary


def _policy_cell_text(policy_row: dict, *, best_score: float, benchmark_score: float, color: bool = True) -> tuple[str, str, str]:
    rank_1 = float(policy_row.get("rank_1_oos", 0.0))
    if color:
        return (
            _format_score(rank_1),
            _format_compare(best_score, rank_1),
            _format_compare(benchmark_score, rank_1),
        )
    return (
        f"{rank_1:.3f}",
        _format_compare_plain(best_score, rank_1),
        _format_compare_plain(benchmark_score, rank_1),
    )


def _table_separator(width: int = 218) -> str:
    return "-" * int(width)


def _render_results_table(rows: list[dict], *, color: bool = True, include_chain: bool = True, chained_oos: dict | None = None) -> str:
    display_rows = list(rows or [])
    if include_chain:
        chain_row = _build_chained_oos_row(chained_oos)
        if chain_row is not None:
            display_rows = display_rows + [chain_row]
    if not display_rows:
        return ""
    widths = {
        "fold": 9,
        "selection": 11,
        "oos_year": 9,
        "rank": 8,
        "best": 17,
        "bench": 17,
        "elapsed": 8,
    }
    lines: list[str] = []
    lines.append("ROLLING NEXT-1Y OOS RESULTS")
    lines.append(_table_separator())
    header1 = (
        f"{_pad_ansi('fold', widths['fold'])} | {_pad_ansi('selection', widths['selection'])} | {_pad_ansi('oos_year', widths['oos_year'])} | "
        f"{_pad_ansi('base', widths['rank'] + widths['best'] + widths['bench'] + 6)} | "
        f"{_pad_ansi('local*', widths['rank'] + widths['best'] + widths['bench'] + 6)} | "
        f"{_pad_ansi('retention', widths['rank'] + widths['best'] + widths['bench'] + 6)} | {_pad_ansi('elapsed', widths['elapsed'], align='>')}"
    )
    header2 = (
        f"{_pad_ansi('', widths['fold'])} | {_pad_ansi('', widths['selection'])} | {_pad_ansi('', widths['oos_year'])} | "
        f"{_pad_ansi('rank_1', widths['rank'], align='>')} | {_pad_ansi('best', widths['best'], align='>')} | {_pad_ansi('0050', widths['bench'], align='>')} | "
        f"{_pad_ansi('rank_1', widths['rank'], align='>')} | {_pad_ansi('best', widths['best'], align='>')} | {_pad_ansi('0050', widths['bench'], align='>')} | "
        f"{_pad_ansi('rank_1', widths['rank'], align='>')} | {_pad_ansi('best', widths['best'], align='>')} | {_pad_ansi('0050', widths['bench'], align='>')} | {_pad_ansi('', widths['elapsed'])}"
    )
    lines.append(header1)
    lines.append(header2)
    lines.append(_table_separator())
    total = len(rows or [])
    for idx, row in enumerate(display_rows, start=1):
        is_chain = str(row.get("fold", "")).upper() == "OOS_CHAIN"
        fold_text = "OOS_CHAIN" if is_chain else f"{idx}/{total}"
        best_score = float(row.get("best_finalist_oos_score", 0.0))
        benchmark_score = float(row.get("benchmark_oos_score", 0.0))
        base_rank, base_best, base_bench = _policy_cell_text(row.get("base") or {}, best_score=best_score, benchmark_score=benchmark_score, color=color)
        local_rank, local_best, local_bench = _policy_cell_text(row.get("local") or {}, best_score=best_score, benchmark_score=benchmark_score, color=color)
        retention_rank, retention_best, retention_bench = _policy_cell_text(row.get("retention") or {}, best_score=best_score, benchmark_score=benchmark_score, color=color)
        line = (
            f"{_pad_ansi(fold_text, widths['fold'])} | {_pad_ansi(str(row.get('selection_period', '')), widths['selection'])} | {_pad_ansi(str(row.get('oos_year', '')), widths['oos_year'])} | "
            f"{_pad_ansi(base_rank, widths['rank'], align='>')} | {_pad_ansi(base_best, widths['best'], align='>')} | {_pad_ansi(base_bench, widths['bench'], align='>')} | "
            f"{_pad_ansi(local_rank, widths['rank'], align='>')} | {_pad_ansi(local_best, widths['best'], align='>')} | {_pad_ansi(local_bench, widths['bench'], align='>')} | "
            f"{_pad_ansi(retention_rank, widths['rank'], align='>')} | {_pad_ansi(retention_best, widths['best'], align='>')} | {_pad_ansi(retention_bench, widths['bench'], align='>')} | "
            f"{_pad_ansi(_fmt_duration(row.get('elapsed_sec', 0.0)), widths['elapsed'], align='>')}"
        )
        lines.append(line)
    lines.append(_table_separator())
    lines.append("best / 0050 括號內 = rank_1_oos - 對照分數；best_finalist_oos / 0050_oos 均為 diagnostic only，不參與 selection。")
    if include_chain and chained_oos:
        lines.append("OOS_CHAIN uses continuous active-param replay over the completed OOS years; values are OOS scores, not compounded returns.")
    return "\n".join(lines)


def _print_completed_results(rows: list[dict]):
    if not rows:
        return
    print("\n" + _render_results_table(rows, color=True, include_chain=False))


def _flatten_policy_for_csv(row: dict, policy_name: str) -> dict:
    policy = dict(row.get(policy_name) or {})
    best = float(row.get("best_finalist_oos_score", 0.0))
    bench = float(row.get("benchmark_oos_score", 0.0))
    rank_1 = float(policy.get("rank_1_oos", 0.0))
    return {
        f"{policy_name}_rank_1_trial": policy.get("rank_1_trial"),
        f"{policy_name}_rank_1_oos": rank_1,
        f"{policy_name}_rank_1_return_pct": float(policy.get("rank_1_return_pct", 0.0)),
        f"{policy_name}_rank_1_mdd_pct": float(policy.get("rank_1_mdd_pct", 0.0)),
        f"{policy_name}_rank_1_trades": int(policy.get("rank_1_trades", 0) or 0),
        f"{policy_name}_best_oos": best,
        f"{policy_name}_best_gap": rank_1 - best,
        f"{policy_name}_0050_oos": bench,
        f"{policy_name}_0050_gap": rank_1 - bench,
    }


def _flatten_row_for_csv(row: dict) -> dict:
    flat = {
        "fold": row.get("fold"),
        "selection": row.get("selection_period"),
        "oos_year": row.get("oos_year"),
        "best_finalist_return_pct": float(row.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": float(row.get("benchmark_return_pct", 0.0)),
    }
    for policy_name in POLICY_NAMES:
        flat.update(_flatten_policy_for_csv(row, policy_name))
    flat["elapsed"] = _fmt_duration(row.get("elapsed_sec", 0.0))
    return flat


def _build_policies_schedule(rows: list[dict]) -> dict:
    policies: dict[str, dict] = {}
    for policy_name in POLICY_NAMES:
        policies[policy_name] = {
            "description": _policy_description(policy_name),
            "schedule": [dict(row.get("policy_schedules", {}).get(policy_name) or {}) for row in rows if row.get("policy_schedules", {}).get(policy_name)],
        }
    return policies


def _build_policy_paramset_payload(*, policy_name: str, rows: list[dict], config: OuterRollingConfig, summary: dict) -> dict:
    params_by_oos_year = {}
    params_by_effective_date = {}
    fold_entries = []
    for row in rows:
        schedule = dict((row.get("policy_schedules") or {}).get(policy_name) or {})
        if not schedule:
            continue
        oos_year = str(int(schedule.get("oos_year") or row.get("oos_year")))
        params_payload = dict(schedule.get("params") or {})
        params_by_oos_year[oos_year] = params_payload
        params_by_effective_date[str(schedule.get("effective_start") or f"{oos_year}-01-01")] = params_payload
        policy_metrics = dict(row.get(policy_name) or {})
        fold_entries.append({
            "fold": row.get("fold"),
            "selection_period": row.get("selection_period"),
            "oos_year": int(row.get("oos_year")),
            "effective_start": schedule.get("effective_start"),
            "effective_end": schedule.get("effective_end"),
            "selected_trial": schedule.get("selected_trial"),
            "base_score": schedule.get("base_score"),
            "base_rank": schedule.get("base_rank"),
            "local_min": schedule.get("local_min"),
            "local_rank": schedule.get("local_rank"),
            "retention": schedule.get("retention"),
            "retention_rank": schedule.get("retention_rank"),
            "oos_score": policy_metrics.get("rank_1_oos"),
            "return_pct": policy_metrics.get("rank_1_return_pct"),
            "mdd_pct": policy_metrics.get("rank_1_mdd_pct"),
            "trades": policy_metrics.get("rank_1_trades"),
            "benchmark_return_pct": row.get("benchmark_return_pct"),
            "benchmark_oos_score": row.get("benchmark_oos_score"),
            "best_finalist_return_pct": row.get("best_finalist_return_pct"),
            "best_finalist_oos_score": row.get("best_finalist_oos_score"),
        })
    chain_all = dict(summary.get("chained_oos") or {})
    chain_policy = dict(chain_all.get(policy_name) or {})
    chained_oos = {
        "method": chain_all.get("method"),
        "note": chain_all.get("note"),
        "selection_period": chain_all.get("selection_period"),
        "oos_period": chain_all.get("oos_period"),
        "rank_1_oos_score": float(chain_policy.get("rank_1_oos", 0.0)),
        "rank_1_return_pct": float(chain_policy.get("rank_1_return_pct", 0.0)),
        "rank_1_mdd_pct": float(chain_policy.get("rank_1_mdd_pct", 0.0)),
        "rank_1_trades": int(chain_policy.get("rank_1_trades", 0) or 0),
        "best_finalist_oos_score": float(chain_all.get("best_finalist_oos_score", 0.0)),
        "best_finalist_return_pct": float(chain_all.get("best_finalist_return_pct", 0.0)),
        "best_finalist_mdd_pct": float(chain_all.get("best_finalist_mdd_pct", 0.0)),
        "benchmark_oos_score": float(chain_all.get("benchmark_oos_score", 0.0)),
        "benchmark_return_pct": float(chain_all.get("benchmark_return_pct", 0.0)),
        "benchmark_mdd_pct": float(chain_all.get("benchmark_mdd_pct", 0.0)),
        "alpha_oos_score": float(chain_policy.get("benchmark_0050_gap", 0.0)),
        "alpha_return_pct": float(chain_policy.get("return_gap_vs_0050_pct", 0.0)),
        "best_gap_oos_score": float(chain_policy.get("best_gap", 0.0)),
        "best_gap_return_pct": float(chain_policy.get("return_gap_vs_best_pct", 0.0)),
    }
    policy_summary = dict(summary.get(policy_name) or {})
    return {
        "schema_type": ROLLING_OOS_PARAM_SET_SCHEMA_TYPE,
        "schema_version": 1,
        "usage": ROLLING_OOS_USAGE,
        "type": "outer_rolling_oos_param_set",
        "created_at": get_taipei_now().isoformat(),
        "selector": str(policy_name),
        "meta": {
            "window_mode": str(config.window_mode),
            "train_window_years": int(config.train_window_years),
            "training_start_year": int(config.training_start_year),
            "first_oos_year": int(config.first_oos_year),
            "last_oos_year": int(config.last_oos_year),
            "oos_horizon": "next_1y",
            "oos_feedback_used": False,
            "promotion_enabled": False,
            "trials_per_fold": int(config.trials_per_fold),
            "live_trading_param": False,
            "active_param_policy": "daily_active_param",
            "active_param_policy_note": "驗證 replay 時，每個交易日所有決策都使用該日期已生效的 active param；實盤同理使用當下正式 promote 的最新 param.json。",
        },
        "summary": {
            "folds": int(summary.get("folds", 0)),
            "selection_period": summary.get("selection_period"),
            "oos_period": summary.get("oos_period"),
            "aggregation_method": summary.get("aggregation_method"),
            "selector": str(policy_name),
            "chain_oos_score": float(policy_summary.get("chain_oos_score", 0.0)),
            "chain_gap_vs_best_oos": float(policy_summary.get("chain_gap_vs_best_oos", 0.0)),
            "chain_gap_vs_0050_oos": float(policy_summary.get("chain_gap_vs_0050_oos", 0.0)),
            "chained_return_pct": float(policy_summary.get("chained_return_pct", 0.0)),
            "chained_gap_vs_0050_pct": float(policy_summary.get("chained_gap_vs_0050_pct", 0.0)),
            "avg_oos_score": float(policy_summary.get("avg_oos_score", 0.0)),
            "median_oos_score": float(policy_summary.get("median_oos_score", 0.0)),
            "worst_oos_score": float(policy_summary.get("worst_oos_score", 0.0)),
            "positive_years": int(policy_summary.get("positive_years", 0)),
            "total_years": int(policy_summary.get("total_years", 0)),
        },
        "active_param_policy": "daily_active_param",
        "active_param_policy_note": "每日決策使用該日 active param；rolling 只是用歷史 effective date replay，不代表實盤使用年度參數組。",
        "chained_oos": chained_oos,
        "params_by_effective_date": params_by_effective_date,
        "params_by_oos_year": params_by_oos_year,
        "folds": fold_entries,
    }


def _write_policy_paramset_files(*, report_dir: str, session_ts: str, rows: list[dict], config: OuterRollingConfig, summary: dict) -> dict:
    paths = {}
    for policy_name in POLICY_NAMES:
        payload = _build_policy_paramset_payload(policy_name=policy_name, rows=rows, config=config, summary=summary)
        path = os.path.join(report_dir, f"outer_rolling_oos_{policy_name}_paramset_{session_ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        paths[policy_name] = path
    return paths


def _write_reports(*, output_dir: str, session_ts: str, rows: list[dict], config: OuterRollingConfig, chained_oos: dict | None = None) -> dict:
    report_dir = os.path.join(output_dir, "outer_rolling_oos")
    os.makedirs(report_dir, exist_ok=True)
    base = os.path.join(report_dir, f"outer_rolling_oos_next1y_{session_ts}")
    json_path = base + ".json"
    csv_path = base + ".csv"
    txt_path = base + ".txt"
    summary = _build_summary(rows, config=config, chained_oos=chained_oos)
    paramset_paths = _write_policy_paramset_files(report_dir=report_dir, session_ts=session_ts, rows=rows, config=config, summary=summary)
    yearly_results = []
    for row in rows:
        light_row = dict(row)
        light_row.pop("policy_schedules", None)
        yearly_results.append(light_row)
    payload = {
        "type": "outer_rolling_oos_next1y",
        "version": 1,
        "created_at": get_taipei_now().isoformat(),
        "meta": {
            "window_mode": str(config.window_mode),
            "train_window_years": int(config.train_window_years),
            "training_start_year": int(config.training_start_year),
            "first_oos_year": int(config.first_oos_year),
            "last_oos_year": int(config.last_oos_year),
            "oos_horizon": "next_1y",
            "oos_feedback_used": False,
            "promotion_enabled": False,
            "trials_per_fold": int(config.trials_per_fold),
        },
        "gate_config": {
            "LOCAL_MIN_SCORE": True,
            "INNER_VALIDATE_RANK": bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED),
            "DOMINANT_YEAR_DEPENDENCY": bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED),
        },
        "summary": summary,
        "yearly_results": yearly_results,
        "policies": _build_policies_schedule(rows),
        "rolling_paramsets": paramset_paths,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    if rows:
        flat_rows = [_flatten_row_for_csv(row) for row in rows]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_format_final_report(rows, summary))
    return {"json": json_path, "csv": csv_path, "txt": txt_path, "paramsets": paramset_paths}


def _build_summary(rows: list[dict], *, config: OuterRollingConfig | None = None, chained_oos: dict | None = None) -> dict:
    if not rows:
        return {"folds": 0}
    selection_start = min(int(row.get("selection_start_year", 0) or 0) for row in rows)
    selection_end = max(int(row.get("selection_end_year", 0) or 0) for row in rows)
    first_oos = min(int(row["oos_year"]) for row in rows)
    last_oos = max(int(row["oos_year"]) for row in rows)
    chained = dict(chained_oos or {})
    summary = {
        "folds": len(rows),
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_period": f"{first_oos}~{last_oos}",
        "aggregation_method": chained.get("method") or "yearly_oos_score_diagnostics",
        "note": chained.get("note") or "OOS_CHAIN is unavailable until active-param replay is computed.",
        "chained_oos": chained,
    }
    for policy_name in POLICY_NAMES:
        scores = [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows]
        returns = [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows]
        benchmark_gaps = [float((row.get(policy_name) or {}).get("benchmark_0050_gap", 0.0)) for row in rows]
        chain_policy = dict(chained.get(policy_name) or {})
        summary[policy_name] = {
            "chain_oos_score": float(chain_policy.get("rank_1_oos", 0.0)),
            "chain_gap_vs_best_oos": float(chain_policy.get("best_gap", 0.0)),
            "chain_gap_vs_0050_oos": float(chain_policy.get("benchmark_0050_gap", 0.0)),
            "chained_return_pct": float(chain_policy.get("rank_1_return_pct", 0.0)),
            "chained_gap_vs_best_pct": float(chain_policy.get("return_gap_vs_best_pct", 0.0)),
            "chained_gap_vs_0050_pct": float(chain_policy.get("return_gap_vs_0050_pct", 0.0)),
            "avg_oos_score": sum(scores) / float(len(scores)),
            "median_oos_score": float(statistics.median(scores)),
            "worst_oos_score": min(scores),
            "positive_years": sum(1 for value in returns if value > 0.0),
            "win_vs_0050_score": sum(1 for gap in benchmark_gaps if gap > 0.0),
            "total_years": len(scores),
            "yearly_return_pct": returns,
            "yearly_oos_score": scores,
        }
    benchmark_returns = [float(row.get("benchmark_return_pct", 0.0)) for row in rows]
    benchmark_scores = [float(row.get("benchmark_oos_score", 0.0)) for row in rows]
    summary["benchmark_0050"] = {
        "chain_oos_score": float(chained.get("benchmark_oos_score", 0.0)),
        "chained_return_pct": float(chained.get("benchmark_return_pct", 0.0)),
        "avg_oos_score": sum(benchmark_scores) / float(len(benchmark_scores)),
        "positive_years": sum(1 for value in benchmark_returns if value > 0.0),
        "total_years": len(benchmark_returns),
        "yearly_return_pct": benchmark_returns,
        "yearly_oos_score": benchmark_scores,
    }
    if config is not None:
        summary["window_mode"] = str(config.window_mode)
        summary["train_window_years"] = int(config.train_window_years)
    return summary

def _format_final_report(rows: list[dict], summary: dict) -> str:
    lines = []
    lines.append("=" * 218)
    lines.append("OUTER ROLLING OOS TEST | VERY NEXT 1 YEAR")
    lines.append("=" * 218)
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 218)
    if rows:
        lines.append(f"folds                    : {summary['folds']}")
        lines.append(f"selection_period         : {summary['selection_period']}")
        lines.append(f"oos_period               : {summary['oos_period']}")
        lines.append(f"aggregation_method       : {summary.get('aggregation_method', 'N/A')}")
        lines.append("note                     : OOS_CHAIN is continuous active-param replay using the same OOS score units as yearly rows.")
        for policy_name in POLICY_NAMES:
            item = summary.get(policy_name) or {}
            lines.append(
                f"{policy_name:<24}: chain_score={float(item.get('chain_oos_score', 0.0)):.3f} | "
                f"score_gap_vs_0050={float(item.get('chain_gap_vs_0050_oos', 0.0)):+.3f} | "
                f"chain_return={float(item.get('chained_return_pct', 0.0)):.2f}% | "
                f"avg_score={float(item.get('avg_oos_score', 0.0)):.3f} | "
                f"median_score={float(item.get('median_oos_score', 0.0)):.3f} | "
                f"worst_score={float(item.get('worst_oos_score', 0.0)):.3f} | "
                f"positive_years={int(item.get('positive_years', 0))}/{int(item.get('total_years', 0))}"
            )
        bench = summary.get("benchmark_0050") or {}
        lines.append(
            f"benchmark_0050           : chain_score={float(bench.get('chain_oos_score', 0.0)):.3f} | "
            f"chain_return={float(bench.get('chained_return_pct', 0.0)):.2f}% | "
            f"avg_score={float(bench.get('avg_oos_score', 0.0)):.3f} | "
            f"positive_years={int(bench.get('positive_years', 0))}/{int(bench.get('total_years', 0))}"
        )
    lines.append("-" * 218)
    rendered = _render_results_table(rows, color=False, include_chain=True, chained_oos=summary.get("chained_oos"))
    if rendered:
        lines.append(rendered)
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

    latest_year = _resolve_latest_year_from_csv_data_dir(selected_data_dir)

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
            print(f"\n{C_CYAN}[{fold_idx}/{len(years)}] selection={selection_start}~{selection_end} | OOS {oos_year}{C_RESET}")
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
                f"[{fold_idx}/{len(years)}] selection={selection_start}~{selection_end} | OOS {oos_year} | LOCAL_MIN_REVIEW DONE | "
                f"finalists={len(finalists)} | best_local={float(finalists[0].get('local_min_score', 0.0)) if finalists else 0.0:.3f} "
                f"#{int(finalists[0]['trial'].number) + 1 if finalists else 0} | elapsed={_fmt_duration(local_elapsed)}"
            )
            policy_items = _build_policy_items(finalists, objective_mode=objective_mode)
            if not any(item is not None for item in policy_items.values()):
                print(f"{C_YELLOW}[{fold_idx}/{len(years)}] selection={selection_start}~{selection_end} | OOS {oos_year} | 無可用 finalist，略過。{C_RESET}")
                continue
            local_rank_map = _build_local_rank_map(finalists)
            retention_rank_map = _build_retention_rank_map(finalists)
            diagnostics = _evaluate_finalist_oos_diagnostics(
                session=session,
                finalists=finalists,
                policy_items=policy_items,
                oos_year=int(oos_year),
            )
            fold_elapsed = time.perf_counter() - fold_start
            selection_period = f"{selection_start}~{selection_end}"
            policy_schedules = {
                name: _build_policy_schedule_entry(
                    item=item,
                    policy_name=name,
                    oos_year=int(oos_year),
                    selection_period=selection_period,
                    local_rank_map=local_rank_map,
                    retention_rank_map=retention_rank_map,
                )
                for name, item in policy_items.items()
                if item is not None
            }
            best_finalist_item = diagnostics.get("_best_finalist_item")
            if best_finalist_item is not None:
                policy_schedules["best"] = _build_policy_schedule_entry(
                    item=best_finalist_item,
                    policy_name="best",
                    oos_year=int(oos_year),
                    selection_period=selection_period,
                    local_rank_map=local_rank_map,
                    retention_rank_map=retention_rank_map,
                )
            row = {
                "fold": f"{fold_idx}/{len(years)}",
                "oos_year": int(oos_year),
                "selection_period": selection_period,
                "selection_start_year": int(selection_start),
                "selection_end_year": int(selection_end),
                "best_finalist_oos_score": float(diagnostics.get("best_finalist_oos_score", 0.0)),
                "best_finalist_trial": diagnostics.get("best_finalist_trial"),
                "best_finalist_return_pct": float(diagnostics.get("best_finalist_return_pct", 0.0)),
                "best_finalist_mdd_pct": float(diagnostics.get("best_finalist_mdd_pct", 0.0)),
                "best_finalist_trades": int(diagnostics.get("best_finalist_trades", 0) or 0),
                "benchmark_oos_score": float(diagnostics.get("benchmark_oos_score", 0.0)),
                "benchmark_return_pct": float(diagnostics.get("benchmark_return_pct", 0.0)),
                "benchmark_mdd_pct": float(diagnostics.get("benchmark_mdd_pct", 0.0)),
                "base": diagnostics.get("policies", {}).get("base", {}),
                "local": diagnostics.get("policies", {}).get("local", {}),
                "retention": diagnostics.get("policies", {}).get("retention", {}),
                "policy_schedules": policy_schedules,
                "elapsed_sec": float(fold_elapsed),
                "optimizer_search_sec": float(max(0.0, fold_elapsed - local_elapsed)),
                "local_min_review_sec": float(local_elapsed),
            }
            rows.append(row)
            local_oos = float((row.get("local") or {}).get("rank_1_oos", 0.0))
            print(
                f"{C_GREEN}[{fold_idx}/{len(years)}] selection={selection_start}~{selection_end} | OOS {oos_year} | DONE | "
                f"local_rank_1_oos={local_oos:.3f} | best={_format_compare_plain(row['best_finalist_oos_score'], local_oos)} | "
                f"0050={_format_compare_plain(row['benchmark_oos_score'], local_oos)} | elapsed={_fmt_duration(fold_elapsed)}{C_RESET}"
            )
            _print_completed_results(rows)
        finally:
            session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    chained_oos = {}
    if rows:
        print(f"\n{C_CYAN}ACTIVE-PARAM OOS_CHAIN REPLAY | OOS {rows[0]['oos_year']}~{rows[-1]['oos_year']}{C_RESET}")
        chained_oos = _evaluate_active_param_chain_replays(
            rows=rows,
            config=config,
            base_policy=base_policy,
            selected_data_dir=selected_data_dir,
            load_all_raw_data=load_all_raw_data,
            optimizer_required_min_rows=optimizer_required_min_rows,
            build_optimizer_session=build_optimizer_session,
        )
    paths = _write_reports(output_dir=output_dir, session_ts=session_ts, rows=rows, config=config, chained_oos=chained_oos)
    print(f"\n{C_CYAN}{'=' * 100}{C_RESET}")
    print("FINAL REPORT")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print(_format_final_report(rows, _build_summary(rows, config=config, chained_oos=chained_oos)))
    print(f"{C_GREEN}已輸出：{paths['txt']}{C_RESET}")
    print(f"{C_GREEN}已輸出：{paths['json']}{C_RESET}")
    print(f"{C_GREEN}已輸出：{paths['csv']}{C_RESET}")
    for policy_name, paramset_path in dict(paths.get("paramsets") or {}).items():
        print(f"{C_GREEN}已輸出 rolling {policy_name} 年度參數組：{paramset_path}{C_RESET}")
    return 0
