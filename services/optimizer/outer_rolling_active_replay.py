from __future__ import annotations

import pandas as pd

from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
    ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
    get_active_param_ensemble_policy,
    is_active_param_ensemble_payload,
)
from core.portfolio_param_runtime import (
    build_active_param_ensemble_objects_from_payload,
    build_active_param_objects_from_payload,
)
from core.portfolio_stats import calc_plain_romd, calc_portfolio_score
from core.raw_universe_contract import (
    RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD,
    build_raw_universe_contract_fields as _raw_universe_contract_fields,
    resolve_raw_universe_required_min_rows,
)
from core.rolling_oos_params import ROLLING_OOS_PARAM_SET_SCHEMA_TYPE, ROLLING_OOS_USAGE
from core.seed_ensemble_policy import renumber_seed_ensemble_members
from core.training_policy import is_optimizer_local_min_review_enabled
from services.optimizer.outer_rolling_artifacts import (
    _build_effective_seed_ensemble_policy_payload,
    _build_params_ensemble_members_for_schedule,
)
from services.optimizer.outer_rolling_fold_context import (
    normalize_optimizer_seed_ensemble_fold_row,
    optimizer_seed_ensemble_row_sort_key,
)

def _build_active_param_replay_payload_from_rows(rows: list[dict], *, policy_name: str | None = None, best_finalist: bool = False, raw_universe_required_min_rows=None) -> dict:
    params_by_oos_year: dict[str, dict] = {}
    params_by_effective_date: dict[str, dict] = {}
    params_ensemble_by_effective_date: dict[str, list[dict]] = {}
    fold_entries: list[dict] = []
    has_ensemble_members = False
    for row in sorted((normalize_optimizer_seed_ensemble_fold_row(item) for item in list(rows or [])), key=optimizer_seed_ensemble_row_sort_key):
        try:
            oos_key = int(row.get("oos_year"))
        except (TypeError, ValueError):
            continue
        effective_start = str(row.get("oos_start_date") or f"{str(oos_key)[:4]}-01-01")
        effective_end = str(row.get("oos_end_date") or f"{str(oos_key)[:4]}-12-31")
        if best_finalist:
            params_payload = dict(row.get("best_finalist_params") or {})
            members = renumber_seed_ensemble_members(row.get("best_finalist_params_ensemble"))
        else:
            schedule = dict((row.get("policy_schedules") or {}).get(str(policy_name)) or {})
            params_payload = dict(schedule.get("params") or {})
            effective_start = str(schedule.get("effective_start") or effective_start)
            effective_end = str(schedule.get("effective_end") or effective_end)
            members = renumber_seed_ensemble_members(_build_params_ensemble_members_for_schedule(schedule))
        if not params_payload and members:
            params_payload = dict(members[0].get("params") or {})
        if not params_payload:
            continue
        params_by_oos_year[str(oos_key)] = params_payload
        params_by_effective_date[effective_start] = params_payload
        if members:
            params_ensemble_by_effective_date[effective_start] = members
            if len(members) > 1:
                has_ensemble_members = True
        fold_entries.append({
            "oos_year": int(oos_key),
            "effective_start": effective_start,
            "effective_end": effective_end,
            "oos_start_date": str(row.get("oos_start_date") or effective_start),
            "oos_end_date": str(row.get("oos_end_date") or effective_end),
        })
    if has_ensemble_members:
        return {
            **_raw_universe_contract_fields(raw_universe_required_min_rows),
            "schema_type": ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
            "schema_version": 1,
            "mode": ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
            "type": "outer_rolling_oos_param_set",
            "active_param_policy": "daily_active_param_ensemble",
            "random_seed_ensemble": _build_effective_seed_ensemble_policy_payload(params_ensemble_by_effective_date, policy_name=policy_name),
            "params_ensemble_by_effective_date": params_ensemble_by_effective_date,
            "params_by_oos_year": params_by_oos_year,
            "params_by_effective_date": params_by_effective_date,
            "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
            "folds": fold_entries,
        }
    return {
        **_raw_universe_contract_fields(raw_universe_required_min_rows),
        "schema_type": ROLLING_OOS_PARAM_SET_SCHEMA_TYPE,
        "schema_version": 1,
        "usage": ROLLING_OOS_USAGE,
        "type": "outer_rolling_oos_param_set",
        "active_param_policy": "daily_active_param",
        "params_by_oos_year": params_by_oos_year,
        "params_by_effective_date": params_by_effective_date,
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        "folds": fold_entries,
    }

def _active_replay_payload_has_params(payload: dict) -> bool:
    if is_active_param_ensemble_payload(payload):
        return bool(dict(payload.get("params_ensemble_by_effective_date") or {}) or list(payload.get("params_ensemble") or []))
    return bool(dict(payload.get("params_by_effective_date") or {}) or dict(payload.get("params_by_oos_year") or {}))

def _extract_active_replay_metrics(result) -> dict:
    if not isinstance(result, tuple) or len(result) < 25:
        raise RuntimeError("active-param replay 回傳格式不完整，無法建立 OOS_CHAIN")
    ret_pct = float(result[2])
    mdd_pct = float(result[3])
    trade_count = int(result[4] or 0)
    win_rate = float(result[5] or 0.0)
    bm_ret_pct = float(result[11])
    bm_mdd_pct = float(result[12])
    r_squared = float(result[15])
    monthly_win_rate = float(result[16])
    bm_r_squared = float(result[17])
    bm_monthly_win_rate = float(result[18])
    annual_return_pct = float(result[23])
    bm_annual_return_pct = float(result[24])
    profile = dict(result[-1]) if isinstance(result[-1], dict) else {}
    equity_curve_points = len(profile.get("equity_curve") or [])
    full_year_count = int(profile.get("full_year_count", 0) or 0)
    min_full_year_return_pct = float(profile.get("min_full_year_return_pct", 0.0) or 0.0)
    yearly_return_rows = list(profile.get("yearly_return_rows") or [])
    full_month_count = int(profile.get("full_month_count", 0) or 0)
    min_month_return_pct = float(profile.get("min_month_return_pct", 0.0) or 0.0)
    monthly_return_rows = list(profile.get("monthly_return_rows") or [])
    full_quarter_count = int(profile.get("full_quarter_count", 0) or 0)
    min_quarter_return_pct = float(profile.get("min_quarter_return_pct", 0.0) or 0.0)
    quarterly_return_rows = list(profile.get("quarterly_return_rows") or [])
    benchmark_full_year_count = int(profile.get("bm_full_year_count", 0) or 0)
    benchmark_min_full_year_return_pct = float(profile.get("bm_min_full_year_return_pct", 0.0) or 0.0)
    benchmark_yearly_return_rows = list(profile.get("bm_yearly_return_rows") or [])
    benchmark_full_month_count = int(profile.get("bm_full_month_count", 0) or 0)
    benchmark_min_month_return_pct = float(profile.get("bm_min_month_return_pct", 0.0) or 0.0)
    benchmark_monthly_return_rows = list(profile.get("bm_monthly_return_rows") or [])
    benchmark_full_quarter_count = int(profile.get("bm_full_quarter_count", 0) or 0)
    benchmark_min_quarter_return_pct = float(profile.get("bm_min_quarter_return_pct", 0.0) or 0.0)
    benchmark_quarterly_return_rows = list(profile.get("bm_quarterly_return_rows") or [])
    portfolio_total_r = float(profile.get("portfolio_total_r", 0.0) or 0.0)
    portfolio_median_r = float(profile.get("portfolio_median_r", 0.0) or 0.0)
    score_total_r = float(profile.get("score_total_r", profile.get("single_stock_total_r", 0.0)) or 0.0)
    score_median_r = float(profile.get("score_median_r", profile.get("single_stock_median_r", 0.0)) or 0.0)
    score = calc_portfolio_score(
        ret_pct,
        mdd_pct,
        monthly_win_rate,
        r_squared,
        annual_return_pct=annual_return_pct,
        trade_win_rate_pct=win_rate,
        min_full_year_return_pct=min_full_year_return_pct,
        min_month_return_pct=min_month_return_pct,
        min_quarter_return_pct=min_quarter_return_pct,
        total_r=score_total_r,
        median_r=score_median_r,
    )
    benchmark_score = calc_plain_romd(bm_ret_pct, bm_mdd_pct)
    plain_romd_score = calc_plain_romd(ret_pct, mdd_pct)
    return {
        "score": float(score),
        "plain_romd_score": float(plain_romd_score),
        "return_pct": float(ret_pct),
        "mdd_pct": float(mdd_pct),
        "annual_return_pct": float(annual_return_pct),
        "full_year_count": int(full_year_count),
        "min_full_year_return_pct": float(min_full_year_return_pct),
        "yearly_return_rows": yearly_return_rows,
        "full_month_count": int(full_month_count),
        "min_month_return_pct": float(min_month_return_pct),
        "monthly_return_rows": monthly_return_rows,
        "full_quarter_count": int(full_quarter_count),
        "min_quarter_return_pct": float(min_quarter_return_pct),
        "quarterly_return_rows": quarterly_return_rows,
        "r_squared": float(r_squared),
        "monthly_win_rate": float(monthly_win_rate),
        "trade_count": int(trade_count),
        "win_rate": float(win_rate),
        "portfolio_total_r": float(portfolio_total_r),
        "portfolio_median_r": float(portfolio_median_r),
        "score_total_r": float(score_total_r),
        "score_median_r": float(score_median_r),
        "score_r_source": str(profile.get("score_r_source", "single_stock")),
        "single_stock_total_r": float(profile.get("single_stock_total_r", score_total_r) or 0.0),
        "single_stock_median_r": float(profile.get("single_stock_median_r", score_median_r) or 0.0),
        "curve_points": int(equity_curve_points),
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(bm_ret_pct),
        "benchmark_mdd_pct": float(bm_mdd_pct),
        "benchmark_annual_return_pct": float(bm_annual_return_pct),
        "benchmark_full_year_count": int(benchmark_full_year_count),
        "benchmark_min_full_year_return_pct": float(benchmark_min_full_year_return_pct),
        "benchmark_yearly_return_rows": benchmark_yearly_return_rows,
        "benchmark_full_month_count": int(benchmark_full_month_count),
        "benchmark_min_month_return_pct": float(benchmark_min_month_return_pct),
        "benchmark_monthly_return_rows": benchmark_monthly_return_rows,
        "benchmark_full_quarter_count": int(benchmark_full_quarter_count),
        "benchmark_min_quarter_return_pct": float(benchmark_min_quarter_return_pct),
        "benchmark_quarterly_return_rows": benchmark_quarterly_return_rows,
        "benchmark_r_squared": float(bm_r_squared),
        "benchmark_monthly_win_rate": float(bm_monthly_win_rate),
    }

def _build_active_replay_schedule_records(payload: dict) -> list[dict]:
    if not _active_replay_payload_has_params(payload):
        return []
    raw_universe_required_min_rows = resolve_raw_universe_required_min_rows(payload)
    if is_active_param_ensemble_payload(payload):
        policy = get_active_param_ensemble_policy(payload)
        records = list(build_active_param_ensemble_objects_from_payload(payload, fixed_risk=None))
        for record in records:
            record[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = raw_universe_required_min_rows
            record["ensemble_min_agree"] = int(policy.get("min_agree", 1) or 1)
            record["ensemble_seed_count"] = int(policy.get("seed_count", len(record.get("members") or []) or 1) or 1)
        return records
    records = list(build_active_param_objects_from_payload(payload, fixed_risk=None))
    for record in records:
        record[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = raw_universe_required_min_rows
    return records

def _row_oos_start_date(row: dict) -> str:
    try:
        return str(row.get("oos_start_date") or f"{str(int(row.get('oos_year')) )[:4]}-01-01")
    except (TypeError, ValueError):
        return str(row.get("oos_start_date") or "")

def _policy_has_complete_active_schedule(rows: list[dict], policy_name: str) -> bool:
    source_rows = sorted(
        (normalize_optimizer_seed_ensemble_fold_row(item) for item in list(rows or [])),
        key=optimizer_seed_ensemble_row_sort_key,
    )
    if not source_rows:
        return False
    first_oos_start = None
    active_starts = []
    for row in source_rows:
        oos_start = _row_oos_start_date(row)
        if oos_start:
            try:
                oos_start_ts = pd.Timestamp(oos_start).normalize()
            except (TypeError, ValueError):
                return False
            if first_oos_start is None or oos_start_ts < first_oos_start:
                first_oos_start = oos_start_ts
        schedule = dict((row.get("policy_schedules") or {}).get(str(policy_name)) or {})
        if not dict(schedule.get("params") or {}):
            continue
        effective_start = str(schedule.get("effective_start") or oos_start or "")
        try:
            effective_start_ts = pd.Timestamp(effective_start).normalize()
        except (TypeError, ValueError):
            return False
        if oos_start and effective_start_ts > oos_start_ts:
            return False
        active_starts.append(effective_start_ts)
    if first_oos_start is None or not active_starts:
        return False
    return min(active_starts) <= first_oos_start

def _empty_unavailable_chain_metrics() -> dict:
    return {
        "available": False,
        "score": 0.0,
        "return_pct": 0.0,
        "mdd_pct": 0.0,
        "annual_return_pct": 0.0,
        "full_month_count": 0,
        "min_month_return_pct": 0.0,
        "full_quarter_count": 0,
        "min_quarter_return_pct": 0.0,
        "r_squared": 0.0,
        "monthly_win_rate": 0.0,
        "trade_count": 0,
        "curve_points": 0,
        "benchmark_oos_score": 0.0,
        "benchmark_return_pct": 0.0,
        "benchmark_mdd_pct": 0.0,
        "benchmark_annual_return_pct": 0.0,
        "benchmark_full_month_count": 0,
        "benchmark_min_month_return_pct": 0.0,
        "benchmark_full_quarter_count": 0,
        "benchmark_min_quarter_return_pct": 0.0,
        "benchmark_r_squared": 0.0,
        "benchmark_monthly_win_rate": 0.0,
    }

def _iter_active_replay_context_records(*, policy_name: str, group_records: list[dict]):
    for record in list(group_records or []):
        members = list(record.get("members") or [])
        if members:
            for member in members:
                item = dict(record)
                item.pop("members", None)
                item.update({
                    "params_obj": member.get("params_obj"),
                    "params_signature": member.get("params_signature"),
                    "member_index": member.get("member_index"),
                    "seed": member.get("seed"),
                    "selected_trial": member.get("selected_trial"),
                })
                item["_ensemble_parent_effective_date"] = record.get("effective_date_text")
                yield item
        else:
            yield dict(record)

def _merge_active_replay_market_dates(contexts_by_signature: dict[str, dict]) -> list:
    market_dates = set()
    for context in dict(contexts_by_signature or {}).values():
        market_dates.update(context.get("sorted_dates") or [])
    return sorted(market_dates)

def _filter_market_dates_by_date_range(market_dates, *, start_date: str | None, end_date: str | None):
    start_ts = pd.Timestamp(start_date).normalize() if start_date else None
    end_ts = pd.Timestamp(end_date).normalize() if end_date else None
    resolved = []
    for raw_date in list(market_dates or []):
        try:
            ts = pd.Timestamp(raw_date).normalize()
        except (TypeError, ValueError):
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        resolved.append(raw_date)
    return sorted(resolved)

