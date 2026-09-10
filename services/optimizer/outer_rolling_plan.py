from __future__ import annotations

import math
import os
from dataclasses import dataclass

import pandas as pd

from config.training_policy import (
    OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    OUTER_ROLLING_OOS_HORIZON_MONTHS,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)
from core.display import C_CYAN, C_GRAY, C_RESET
from core.training_policy import is_optimizer_local_min_review_enabled
from services.optimizer.outer_rolling_fold_context import (
    _fold_label_display,
    _fold_selection_label_display,
    build_optimizer_seed_ensemble_fold_context,
)
from services.optimizer.outer_rolling_formatting import (
    _extract_cli_value,
    _month_start,
    _parse_oos_boundary,
    _prompt_int,
)


@dataclass
class OuterRollingConfig:
    training_start_year: int
    first_oos_year: int
    last_oos_year: int
    trials_per_fold: int
    window_mode: str = "fixed"
    train_window_years: int = 5
    confirm: bool = True
    first_oos_date: str = ""
    last_oos_date: str = ""
    train_window_months: int = OUTER_ROLLING_TRAIN_WINDOW_MONTHS
    oos_horizon_months: int = OUTER_ROLLING_OOS_HORIZON_MONTHS
    raw_universe_required_min_rows: int | None = None

def _build_rolling_folds(config: OuterRollingConfig) -> list[dict]:
    first_oos = _parse_oos_boundary(config.first_oos_date, default=pd.Timestamp(year=int(config.first_oos_year), month=1, day=1))
    last_oos = _parse_oos_boundary(config.last_oos_date, default=pd.Timestamp(year=int(config.last_oos_year), month=12, day=1), year_boundary="end")
    horizon_months = max(1, int(config.oos_horizon_months or OUTER_ROLLING_OOS_HORIZON_MONTHS))
    train_months = max(1, int(config.train_window_months or OUTER_ROLLING_TRAIN_WINDOW_MONTHS))
    training_start = pd.Timestamp(year=int(config.training_start_year), month=1, day=1)
    if first_oos > last_oos:
        raise ValueError("first OOS date 不可晚於 last OOS date")
    folds: list[dict] = []
    current = first_oos
    while current <= last_oos:
        oos_start = _month_start(current)
        oos_end = oos_start + pd.DateOffset(months=horizon_months) - pd.Timedelta(days=1)
        selection_end = oos_start - pd.Timedelta(days=1)
        if str(config.window_mode).lower() == "fixed":
            selection_start = oos_start - pd.DateOffset(months=train_months)
        else:
            selection_start = training_start
        if selection_start < training_start:
            raise ValueError("fixed window 下 first OOS date - train window months 不可早於 training start date")
        context = build_optimizer_seed_ensemble_fold_context(
            fold_idx=len(folds) + 1,
            fold_count=0,
            selection_start_date=selection_start.strftime("%Y-%m-%d"),
            selection_end_date=selection_end.strftime("%Y-%m-%d"),
            oos_start_date=oos_start.strftime("%Y-%m-%d"),
            oos_end_date=oos_end.strftime("%Y-%m-%d"),
        )
        context["fold_key"] = int(context["oos_year"])
        context["selection_start_year"] = int(selection_start.year)
        context["selection_end_year"] = int(selection_end.year)
        folds.append(context)
        current = oos_start + pd.DateOffset(months=horizon_months)
    total_folds = len(folds)
    for idx, fold in enumerate(folds, start=1):
        fold["fold_idx"] = int(idx)
        fold["fold_count"] = int(total_folds)
        fold["fold"] = f"{idx}/{total_folds}"
    return folds




def _resolve_latest_date_from_csv_data_dir(data_dir: str) -> pd.Timestamp | None:
    # AI註: outer rolling OOS 的互動設定只需要 last OOS 預設值；
    # 不應為此先觸發 optimizer 完整資料清洗、快取摘要與 issue log。
    if not os.path.isdir(str(data_dir)):
        return None
    try:
        from core.data_utils import discover_unique_csv_inputs
        csv_inputs, _duplicate_file_issue_lines = discover_unique_csv_inputs(str(data_dir))
    except (OSError, ValueError, TypeError):
        return None

    latest_date = None
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
            parsed_dates = pd.to_datetime(date_values, errors="coerce").dropna()
            if parsed_dates.empty:
                continue
            file_date = pd.Timestamp(parsed_dates.max()).normalize()
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError):
            continue
        if latest_date is None or file_date > latest_date:
            latest_date = file_date
    return latest_date




def _resolve_config(argv, environ, *, base_policy: dict, latest_year: int | None, latest_date=None, default_trials: int, timing_mode: bool = False) -> OuterRollingConfig:
    env = os.environ if environ is None else environ
    first_oos_year_default = int(base_policy.get("oos_start_year") or base_policy.get("search_train_end_year", 0) + 1 or 2023)
    first_oos_date_default = pd.Timestamp(year=first_oos_year_default, month=1, day=1)
    latest_ts = pd.Timestamp(latest_date).normalize() if latest_date is not None else pd.Timestamp(year=int(latest_year or first_oos_year_default), month=1, day=1)
    last_oos_date_default = _month_start(latest_ts)
    configured_oos_end_date = str(base_policy.get("oos_end_date") or "").strip()
    if configured_oos_end_date:
        last_oos_date_default = min(last_oos_date_default, _month_start(pd.Timestamp(configured_oos_end_date).normalize()))
    trials_default = int(
        default_trials
        if int(default_trials or 0) > 0
        else int(
            env.get(
                "V16_OUTER_ROLLING_OOS_TRIALS",
                str(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT),
            )
            or OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
        )
    )

    cli_first_date = _extract_cli_value(argv, "--outer-first-oos-date")
    cli_last_date = _extract_cli_value(argv, "--outer-last-oos-date")
    cli_first = _extract_cli_value(argv, "--outer-first-oos")
    cli_last = _extract_cli_value(argv, "--outer-last-oos")
    cli_trials = _extract_cli_value(argv, "--trials")
    cli_window_mode = _extract_cli_value(argv, "--outer-window-mode")
    cli_train_start = _extract_cli_value(argv, "--outer-train-start")
    cli_train_window_months = _extract_cli_value(argv, "--outer-train-window-months")
    cli_train_window_years = _extract_cli_value(argv, "--outer-train-window-years")
    cli_oos_months = _extract_cli_value(argv, "--outer-oos-months")

    requested_window_mode = str(cli_window_mode or env.get("V16_OUTER_ROLLING_WINDOW_MODE", "fixed") or "fixed").strip().lower()
    if requested_window_mode != "fixed":
        raise ValueError("目前固定採 fixed-window 架構，--outer-window-mode / V16_OUTER_ROLLING_WINDOW_MODE 只接受 fixed。")
    window_mode = "fixed"

    train_window_month_default = max(1, int(env.get("V16_OUTER_ROLLING_TRAIN_WINDOW_MONTHS", str(OUTER_ROLLING_TRAIN_WINDOW_MONTHS)) or OUTER_ROLLING_TRAIN_WINDOW_MONTHS))
    if not str(env.get("V16_OUTER_ROLLING_TRAIN_WINDOW_MONTHS", "")).strip() and str(env.get("V16_OUTER_ROLLING_TRAIN_WINDOW_YEARS", "")).strip():
        train_window_month_default = max(1, int(env.get("V16_OUTER_ROLLING_TRAIN_WINDOW_YEARS")) * 12)
    if cli_train_window_months:
        train_window_months = int(cli_train_window_months)
    elif cli_train_window_years:
        train_window_months = int(cli_train_window_years) * 12
    else:
        train_window_months = train_window_month_default

    oos_horizon_month_default = max(1, int(env.get("V16_OUTER_ROLLING_OOS_MONTHS", str(OUTER_ROLLING_OOS_HORIZON_MONTHS)) or OUTER_ROLLING_OOS_HORIZON_MONTHS))
    if cli_oos_months:
        oos_horizon_months = int(cli_oos_months)
    else:
        oos_horizon_months = oos_horizon_month_default

    first_source = cli_first_date or cli_first or str(env.get("V16_OUTER_ROLLING_FIRST_OOS_DATE", "")).strip() or str(env.get("V16_OUTER_ROLLING_FIRST_OOS", "")).strip()
    last_source = cli_last_date or cli_last or str(env.get("V16_OUTER_ROLLING_LAST_OOS_DATE", "")).strip() or str(env.get("V16_OUTER_ROLLING_LAST_OOS", "")).strip()
    first_oos_date = _parse_oos_boundary(first_source, default=first_oos_date_default) if first_source else first_oos_date_default
    last_oos_date = _parse_oos_boundary(last_source, default=last_oos_date_default, year_boundary="end") if last_source else last_oos_date_default

    if cli_trials:
        trials = int(cli_trials)
    elif bool(timing_mode) or int(default_trials or 0) > 0:
        trials = trials_default
    else:
        trials = _prompt_int("optimizer trials per fold", trials_default, minimum=1)

    if train_window_months <= 0:
        raise ValueError("train window months 必須大於 0")
    if oos_horizon_months <= 0:
        raise ValueError("OOS horizon months 必須大於 0")
    if first_oos_date > last_oos_date:
        raise ValueError("first OOS date 不可晚於 last OOS date")
    if trials <= 0:
        raise ValueError("optimizer trials per fold 必須大於 0")

    derived_train_start_date = first_oos_date - pd.DateOffset(months=int(train_window_months))
    derived_train_start_year = int(derived_train_start_date.year)
    explicit_train_start = cli_train_start or str(env.get("V16_OUTER_ROLLING_TRAIN_START", "")).strip()
    if explicit_train_start:
        explicit_year = int(str(explicit_train_start).strip())
        if explicit_year != derived_train_start_year:
            raise ValueError(
                "fixed-window 架構下 training start year 由 first OOS date - train window months 推導，"
                f"不可獨立設定為 {explicit_year}；目前推導值為 {derived_train_start_year}。"
            )

    return OuterRollingConfig(
        int(derived_train_start_year),
        int(first_oos_date.year),
        int(last_oos_date.year),
        int(trials),
        window_mode=window_mode,
        train_window_years=max(1, int(math.ceil(int(train_window_months) / 12.0))),
        confirm=False,
        first_oos_date=first_oos_date.strftime("%Y-%m-%d"),
        last_oos_date=last_oos_date.strftime("%Y-%m-%d"),
        train_window_months=int(train_window_months),
        oos_horizon_months=int(oos_horizon_months),
    )



def _print_plan(config: OuterRollingConfig, *, parallel_settings_line: str | None = None):
    folds = _build_rolling_folds(config)
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print(f"OUTER ROLLING OOS TEST | NEXT {int(config.oos_horizon_months)} MONTHS")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print(f"window mode      : {config.window_mode}")
    if str(config.window_mode).lower() == "fixed":
        print(f"train window     : {int(config.train_window_months)} months")
    else:
        print(f"training start   : {config.training_start_year}-01-01")
    print("oos feedback     : False")
    print("promotion        : disabled")
    print(f"oos horizon      : next {int(config.oos_horizon_months)} months")
    print(f"optimizer trials : {config.trials_per_fold} per fold")
    if parallel_settings_line:
        print(str(parallel_settings_line))
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    print(f"{'fold':<6} | {'selection period':<15} | {'OOS test period':<15}")
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    for idx, fold in enumerate(folds, start=1):
        print(f"{idx}/{len(folds):<4} | {_fold_selection_label_display(fold):<15} | {_fold_label_display(fold):<15}")
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    print(f"LOCAL_MIN_SCORE              : {bool(is_optimizer_local_min_review_enabled())}")
    print(f"INNER_VALIDATE_RANK          : {bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED)}")
    print(f"DOMINANT_YEAR_DEPENDENCY     : {bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED)}")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")


def _confirm_plan(config: OuterRollingConfig) -> bool:
    return True
