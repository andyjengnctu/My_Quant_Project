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
from core.runtime_utils import choose_inline_progress_message, get_taipei_now, is_interactive_console, safe_prompt_choice, stdout_supports_inline_progress, write_inline_progress
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
    }


def _format_oos_delta(reference_score, selected_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    selected = _safe_float(selected_score, 0.0)
    return f"{ref:.3f} ({selected - ref:+.3f})"


def _evaluate_finalist_oos_diagnostics(*, session, finalists: list[dict], policy_items: dict[str, dict | None], oos_year: int) -> dict:
    best_score = float("-inf")
    best_trial_number = None
    report_cache: dict[int, dict] = {}
    metrics_by_trial: dict[int, dict] = {}
    benchmark_score = 0.0
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
        score = float(metrics["oos_score"])
        if score > best_score:
            best_score = score
            best_trial_number = trial_number
    if best_score == float("-inf"):
        best_score = 0.0
    policies: dict[str, dict] = {}
    for policy_name, item in dict(policy_items or {}).items():
        if item is None or item.get("trial") is None:
            policies[policy_name] = {
                "rank_1_trial": None,
                "rank_1_oos": 0.0,
                "best_gap": 0.0 - float(best_score),
                "benchmark_0050_gap": 0.0 - float(benchmark_score),
            }
            continue
        trial_number = int(item["trial"].number)
        if trial_number not in metrics_by_trial:
            report = _evaluate_next_1y_oos(session=session, trial=item["trial"], oos_year=int(oos_year))
            metrics_by_trial[trial_number] = _extract_period_metrics(report)
        rank_1_oos = float(metrics_by_trial[trial_number].get("oos_score", 0.0))
        benchmark_score = float(metrics_by_trial[trial_number].get("benchmark_oos_score", benchmark_score))
        policies[policy_name] = {
            "rank_1_trial": trial_number + 1,
            "rank_1_oos": rank_1_oos,
            "best_gap": rank_1_oos - float(best_score),
            "benchmark_0050_gap": rank_1_oos - float(benchmark_score),
        }
    return {
        "best_finalist_oos_score": float(best_score),
        "best_finalist_trial": int(best_trial_number) + 1 if best_trial_number is not None else None,
        "benchmark_oos_score": float(benchmark_score),
        "policies": policies,
    }


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


def _build_average_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    first_year = min(int(row["oos_year"]) for row in rows)
    last_year = max(int(row["oos_year"]) for row in rows)
    selection_start = min(int(row.get("selection_start_year", str(row.get("selection_period", "0~0")).split("~", 1)[0])) for row in rows)
    selection_end = max(int(row.get("selection_end_year", str(row.get("selection_period", "0~0")).split("~", 1)[-1])) for row in rows)
    best_avg = sum(float(row.get("best_finalist_oos_score", 0.0)) for row in rows) / float(len(rows))
    benchmark_avg = sum(float(row.get("benchmark_oos_score", 0.0)) for row in rows) / float(len(rows))
    avg_row = {
        "fold": "AVG",
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_year": f"{first_year}~{last_year}",
        "best_finalist_oos_score": best_avg,
        "benchmark_oos_score": benchmark_avg,
        "elapsed_sec": sum(float(row.get("elapsed_sec", 0.0)) for row in rows),
        "aggregation_method": "avg_yearly_oos",
    }
    for policy_name in ("base", "local", "retention"):
        rank_avg = sum(float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows) / float(len(rows))
        avg_row[policy_name] = {
            "rank_1_trial": None,
            "rank_1_oos": rank_avg,
            "best_gap": rank_avg - best_avg,
            "benchmark_0050_gap": rank_avg - benchmark_avg,
        }
    return avg_row


def _table_separator(width: int = 218) -> str:
    return "-" * int(width)


def _render_results_table(rows: list[dict], *, color: bool = True, include_average: bool = True) -> str:
    display_rows = list(rows or [])
    if include_average:
        avg_row = _build_average_row(display_rows)
        if avg_row is not None:
            display_rows = display_rows + [avg_row]
    if not display_rows:
        return ""
    widths = {
        "fold": 6,
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
        is_avg = str(row.get("fold", "")).upper() == "AVG"
        fold_text = "AVG" if is_avg else f"{idx}/{total}"
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
    if include_average:
        lines.append("AVG 列為年度 OOS score 平均；尚未冒充為 rolling-param 全期間 portfolio sim 重算結果。")
    return "\n".join(lines)


def _print_completed_results(rows: list[dict]):
    if not rows:
        return
    print("\n" + _render_results_table(rows, color=True, include_average=True))


def _flatten_policy_for_csv(row: dict, policy_name: str) -> dict:
    policy = dict(row.get(policy_name) or {})
    best = float(row.get("best_finalist_oos_score", 0.0))
    bench = float(row.get("benchmark_oos_score", 0.0))
    rank_1 = float(policy.get("rank_1_oos", 0.0))
    return {
        f"{policy_name}_rank_1_trial": policy.get("rank_1_trial"),
        f"{policy_name}_rank_1_oos": rank_1,
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
    }
    for policy_name in ("base", "local", "retention"):
        flat.update(_flatten_policy_for_csv(row, policy_name))
    flat["elapsed"] = _fmt_duration(row.get("elapsed_sec", 0.0))
    return flat


def _build_policies_schedule(rows: list[dict]) -> dict:
    policies: dict[str, dict] = {}
    for policy_name in ("base", "local", "retention"):
        policies[policy_name] = {
            "description": _policy_description(policy_name),
            "schedule": [dict(row.get("policy_schedules", {}).get(policy_name) or {}) for row in rows if row.get("policy_schedules", {}).get(policy_name)],
        }
    return policies


def _write_reports(*, output_dir: str, session_ts: str, rows: list[dict], config: OuterRollingConfig) -> dict:
    report_dir = os.path.join(output_dir, "outer_rolling_oos")
    os.makedirs(report_dir, exist_ok=True)
    base = os.path.join(report_dir, f"outer_rolling_oos_next1y_{session_ts}")
    json_path = base + ".json"
    csv_path = base + ".csv"
    txt_path = base + ".txt"
    summary = _build_summary(rows, config=config)
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
    return {"json": json_path, "csv": csv_path, "txt": txt_path}


def _build_summary(rows: list[dict], *, config: OuterRollingConfig | None = None) -> dict:
    if not rows:
        return {"folds": 0}
    selection_start = min(int(row.get("selection_start_year", 0) or 0) for row in rows)
    selection_end = max(int(row.get("selection_end_year", 0) or 0) for row in rows)
    first_oos = min(int(row["oos_year"]) for row in rows)
    last_oos = max(int(row["oos_year"]) for row in rows)
    summary = {
        "folds": len(rows),
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_period": f"{first_oos}~{last_oos}",
        "aggregation_method": "avg_yearly_oos",
        "note": "overall_oos is yearly OOS-score average unless portfolio sim later recomputes rolling-param full-period result.",
    }
    for policy_name in ("base", "local", "retention"):
        scores = [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows]
        benchmark_gaps = [float((row.get(policy_name) or {}).get("benchmark_0050_gap", 0.0)) for row in rows]
        summary[policy_name] = {
            "overall_oos": sum(scores) / float(len(scores)),
            "positive_years": sum(1 for score in scores if score > 0.0),
            "win_vs_0050": sum(1 for gap in benchmark_gaps if gap > 0.0),
            "total_years": len(scores),
            "worst_oos": min(scores),
            "median_oos": float(statistics.median(scores)),
        }
    benchmark_scores = [float(row.get("benchmark_oos_score", 0.0)) for row in rows]
    summary["benchmark_0050"] = {
        "overall_oos": sum(benchmark_scores) / float(len(benchmark_scores)),
        "positive_years": sum(1 for score in benchmark_scores if score > 0.0),
        "total_years": len(benchmark_scores),
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
        for policy_name in ("base", "local", "retention"):
            item = summary.get(policy_name) or {}
            lines.append(
                f"{policy_name:<24}: avg_oos={float(item.get('overall_oos', 0.0)):.3f} | "
                f"median={float(item.get('median_oos', 0.0)):.3f} | worst={float(item.get('worst_oos', 0.0)):.3f} | "
                f"positive={int(item.get('positive_years', 0))}/{int(item.get('total_years', 0))} | "
                f"win_vs_0050={int(item.get('win_vs_0050', 0))}/{int(item.get('total_years', 0))}"
            )
        bench = summary.get("benchmark_0050") or {}
        lines.append(f"benchmark_0050           : avg_oos={float(bench.get('overall_oos', 0.0)):.3f} | positive={int(bench.get('positive_years', 0))}/{int(bench.get('total_years', 0))}")
    lines.append("-" * 218)
    rendered = _render_results_table(rows, color=False, include_average=True)
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
            row = {
                "fold": f"{fold_idx}/{len(years)}",
                "oos_year": int(oos_year),
                "selection_period": selection_period,
                "selection_start_year": int(selection_start),
                "selection_end_year": int(selection_end),
                "best_finalist_oos_score": float(diagnostics.get("best_finalist_oos_score", 0.0)),
                "best_finalist_trial": diagnostics.get("best_finalist_trial"),
                "benchmark_oos_score": float(diagnostics.get("benchmark_oos_score", 0.0)),
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

    paths = _write_reports(output_dir=output_dir, session_ts=session_ts, rows=rows, config=config)
    print(f"\n{C_CYAN}{'=' * 100}{C_RESET}")
    print("FINAL REPORT")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print(_format_final_report(rows, _build_summary(rows, config=config)))
    print(f"{C_GREEN}已輸出：{paths['txt']}{C_RESET}")
    print(f"{C_GREEN}已輸出：{paths['json']}{C_RESET}")
    print(f"{C_GREEN}已輸出：{paths['csv']}{C_RESET}")
    return 0
