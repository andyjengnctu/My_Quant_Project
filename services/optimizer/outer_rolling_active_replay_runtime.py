from __future__ import annotations

import time

import pandas as pd

from core.display import C_CYAN, C_RESET
from core.portfolio_stats import calc_plain_romd
from core.raw_universe_contract import RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD
from core.runtime_utils import stdout_supports_inline_progress, write_inline_progress
from core.training_performance import (
    is_optimizer_active_replay_include_pit_stats_index_enabled,
    is_optimizer_active_replay_include_trade_logs_enabled,
    is_optimizer_active_replay_use_prepared_cache_enabled,
    is_optimizer_active_replay_write_prepared_cache_enabled,
    resolve_optimizer_active_replay_prep_workers,
)
from services.optimizer.outer_rolling_active_replay import (
    _build_active_param_replay_payload_from_rows,
    _build_active_replay_schedule_records,
    _empty_unavailable_chain_metrics,
    _extract_active_replay_metrics,
    _filter_market_dates_by_date_range,
    _iter_active_replay_context_records,
    _merge_active_replay_market_dates,
    _policy_has_complete_active_schedule,
)
from services.optimizer.outer_rolling_formatting import _fmt_duration, _period_label
from services.optimizer.outer_rolling_plan import OuterRollingConfig
from services.optimizer.outer_rolling_policy import CHAIN_POLICY_NAMES
from services.optimizer.prep import prepare_trial_inputs

def _load_active_replay_contexts_by_signature(
    *,
    data_dir: str,
    schedule_groups: dict[str, list[dict]],
    output_dir: str,
    first_year: int | None = None,
    last_year: int | None = None,
    overall_start: float | None = None,
) -> dict[str, dict]:
    from core.data_utils import get_required_min_rows
    from core.portfolio_fast_data import build_normal_setup_index, build_trade_stats_index, pack_static_market_data
    from services.optimizer.raw_cache import load_all_raw_data
    from services.portfolio_replay import load_portfolio_market_context

    contexts_by_signature: dict[str, dict] = {}
    records: list[dict] = []
    by_signature: dict[str, dict] = {}
    for policy_name, group_records in dict(schedule_groups or {}).items():
        for record in _iter_active_replay_context_records(policy_name=str(policy_name), group_records=list(group_records or [])):
            signature = str(record.get("params_signature") or "")
            if not signature:
                continue
            existing = by_signature.get(signature)
            if existing is None:
                existing = dict(record)
                existing["_policies"] = []
                by_signature[signature] = existing
                records.append(existing)
            policies = existing.setdefault("_policies", [])
            if str(policy_name) not in policies:
                policies.append(str(policy_name))

    def _record_year(item: dict) -> int:
        try:
            return int(item.get("year", 0) or 0)
        except (TypeError, ValueError):
            text = str(item.get("effective_date_text") or "")
            try:
                return int(text[:4])
            except (TypeError, ValueError):
                return 0

    records.sort(key=lambda item: (_record_year(item), str(item.get("effective_date_text") or ""), str(item.get("params_signature") or "")))
    total = len(records)
    previous_width = 0
    supports_inline = stdout_supports_inline_progress()
    policy_names = "/".join(sorted({policy for record in records for policy in record.get("_policies", [])})) or "N/A"
    replay_context_start = time.perf_counter()
    years_label = ""
    if first_year and last_year:
        years_label = f" | years={int(first_year)}~{int(last_year)}"

    include_trade_logs = is_optimizer_active_replay_include_trade_logs_enabled()
    include_pit_stats_index = is_optimizer_active_replay_include_pit_stats_index_enabled()
    use_prepared_cache = is_optimizer_active_replay_use_prepared_cache_enabled()
    write_prepared_cache = is_optimizer_active_replay_write_prepared_cache_enabled()

    raw_data_cache = None
    static_fast_cache = None
    static_master_dates = None
    active_prep_workers = None
    if total and not (use_prepared_cache or write_prepared_cache):
        params_required_min_rows = max(get_required_min_rows(record["params_obj"]) for record in records)
        contract_min_rows = max((int(record.get(RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD) or 0) for record in records), default=0)
        required_min_rows = max(int(params_required_min_rows), int(contract_min_rows))
        raw_data_cache = load_all_raw_data(data_dir, required_min_rows, output_dir, verbose=False)
        static_fast_cache = {ticker: pack_static_market_data(df) for ticker, df in raw_data_cache.items()}
        static_master_dates = set()
        for df in raw_data_cache.values():
            static_master_dates.update(df.index)
        active_prep_workers = resolve_optimizer_active_replay_prep_workers(len(raw_data_cache))

    def _covers_text(index: int, item: dict) -> str:
        year = _record_year(item)
        if year <= 0:
            return "N/A"
        next_year = None
        for follow in records[index:]:
            candidate = _record_year(follow)
            if candidate > year:
                next_year = candidate
                break
        end_year = int(last_year) if last_year else year
        if next_year is not None:
            end_year = min(end_year, int(next_year) - 1)
        if end_year <= year:
            return str(year)
        return f"{year}~{end_year}"

    for idx, record in enumerate(records, start=1):
        signature = str(record["params_signature"])
        policies_text = ",".join(record.get("_policies", [])) or "N/A"
        chain_elapsed = max(0.0, time.perf_counter() - replay_context_start)
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        message = (
            f"{C_CYAN}⏱️ OOS_CHAIN active replay context [{idx}/{total}] | "
            f"covers={_covers_text(idx, record)} | policies={policies_text} | effective={record.get('effective_date_text')} | "
            f"signature={signature[:8]}{years_label} | chain={_fmt_duration(chain_elapsed)}{total_elapsed_text}{C_RESET}"
        )
        if supports_inline:
            previous_width = write_inline_progress(message, previous_width=previous_width)
        else:
            print(message)

        if raw_data_cache is not None:
            prep_result = prepare_trial_inputs(
                raw_data_cache=raw_data_cache,
                params=record["params_obj"],
                default_max_workers=int(active_prep_workers or 1),
                static_fast_cache=static_fast_cache,
                static_master_dates=static_master_dates,
                include_trade_logs=bool(include_trade_logs),
                include_pit_stats_index=bool(include_pit_stats_index),
                profile_enabled=False,
            )
            context = {
                "all_dfs_fast": prep_result.get("all_dfs_fast") or {},
                "all_trade_logs": prep_result.get("all_trade_logs") or {},
                "all_pit_stats_index": prep_result.get("all_pit_stats_index") or {},
                "sorted_dates": sorted(prep_result.get("master_dates") or []),
                "prep_wall_sec": float(prep_result.get("prep_wall_sec", 0.0) or 0.0),
                "prep_mode": f"active_replay_shared_raw_{prep_result.get('prep_mode')}",
            }
        else:
            context = load_portfolio_market_context(
                data_dir,
                record["params_obj"],
                verbose=False,
                use_prepared_cache=use_prepared_cache,
                write_prepared_cache=write_prepared_cache,
                raw_universe_required_min_rows=record.get(RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD),
            )
            context = dict(context)

        if not context.get("all_pit_stats_index"):
            if not context.get("all_trade_logs"):
                raise RuntimeError("active replay context 缺少 PIT stats index；請保留 OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX=1")
            context["all_pit_stats_index"] = {
                ticker: build_trade_stats_index(logs)
                for ticker, logs in (context.get("all_trade_logs") or {}).items()
            }
        context["normal_setup_index"] = build_normal_setup_index(context.get("all_dfs_fast") or {})
        contexts_by_signature[signature] = context

    if total:
        chain_elapsed = max(0.0, time.perf_counter() - replay_context_start)
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        summary = (
            f"{C_CYAN}⏱️ OOS_CHAIN active replay context 完成 | "
            f"contexts={total}/{total}{years_label} | policies={policy_names} | "
            f"chain={_fmt_duration(chain_elapsed)}{total_elapsed_text}{C_RESET}"
        )
        if supports_inline:
            write_inline_progress(summary, previous_width=previous_width)
            print()
        else:
            print(summary)
    return contexts_by_signature

def _run_active_replay_metrics_from_schedule_records(
    *,
    schedule_records: list[dict],
    contexts_by_signature: dict[str, dict],
    start_year: int,
    end_year: int,
    max_positions: int,
    enable_rotation: bool,
    benchmark_ticker: str = "0050",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    if not schedule_records:
        return {}
    from core.portfolio_engine import run_portfolio_timeline
    from core.portfolio_stats import find_sim_start_idx
    from services.portfolio_replay import (
        _apply_active_single_stock_score_stats,
        _filter_market_dates_by_end_year,
        _resolve_active_schedule_record,
    )

    all_market_dates = _merge_active_replay_market_dates(contexts_by_signature)
    if start_date or end_date:
        resolved_sorted_dates = _filter_market_dates_by_date_range(
            all_market_dates,
            start_date=start_date or f"{int(start_year)}-01-01",
            end_date=end_date or f"{int(end_year)}-12-31",
        )
        resolved_start_year = int(pd.Timestamp(resolved_sorted_dates[0]).year) if resolved_sorted_dates else int(start_year)
    else:
        resolved_sorted_dates = _filter_market_dates_by_end_year(
            all_market_dates,
            start_year=int(start_year),
            end_year=int(end_year),
        )
        resolved_start_year = int(start_year)
    if not resolved_sorted_dates:
        raise ValueError("active-param replay 沒有可回測日期")
    first_sim_idx = find_sim_start_idx(resolved_sorted_dates, int(resolved_start_year))
    if first_sim_idx >= len(resolved_sorted_dates):
        raise ValueError("active-param replay 起始日期沒有可回測日期")
    first_record = _resolve_active_schedule_record(schedule_records, resolved_sorted_dates[first_sim_idx])
    use_param_ensemble = bool(first_record.get("members"))
    if use_param_ensemble:
        first_members = list(first_record.get("members") or [])
        if not first_members:
            raise ValueError("active-param ensemble schedule 缺少 members")
        base_contexts = [contexts_by_signature[str(member["params_signature"])] for member in first_members]
        base_context = base_contexts[0]
        base_params = first_members[0]["params_obj"]
        benchmark_data = (base_context.get("all_dfs_fast") or {}).get(benchmark_ticker)
        ensemble_min_agree = int(first_record.get("ensemble_min_agree", len(first_members)) or len(first_members))

        def active_param_ensemble_resolver(trade_date):
            return _resolve_active_schedule_record(schedule_records, trade_date)["members"]

        def active_context_ensemble_resolver(trade_date):
            record = _resolve_active_schedule_record(schedule_records, trade_date)
            return [contexts_by_signature[str(member["params_signature"])] for member in list(record.get("members") or [])]

        pf_profile = {
            "param_policy": "active_param_ensemble_replay",
            "capture_equity_curve": True,
            "active_param_ensemble": {
                "seed_count": int(first_record.get("ensemble_seed_count", len(first_members)) or len(first_members)),
                "min_agree": int(ensemble_min_agree),
            },
            "active_param_ensemble_schedule": [
                {
                    "effective_date": str(record.get("effective_date_text")),
                    "year": int(record.get("year", 0) or 0),
                    "member_count": int(len(record.get("members") or [])),
                }
                for record in schedule_records
            ],
        }
        contexts_by_effective_date = {
            str(record.get("effective_date_text")): [
                contexts_by_signature[str(member["params_signature"])]
                for member in list(record.get("members") or [])
            ]
            for record in schedule_records
        }
        _apply_active_single_stock_score_stats(
            pf_profile,
            schedule_records,
            contexts_by_effective_date,
            resolved_sorted_dates,
            ensemble=True,
        )
        result = run_portfolio_timeline(
            base_context.get("all_dfs_fast") or {},
            base_context.get("all_trade_logs") or {},
            resolved_sorted_dates,
            int(resolved_start_year),
            base_params,
            int(max_positions),
            bool(enable_rotation),
            benchmark_ticker=str(benchmark_ticker),
            benchmark_data=benchmark_data,
            is_training=False,
            profile_stats=pf_profile,
            verbose=False,
            pit_stats_index=base_context.get("all_pit_stats_index"),
            active_param_ensemble_resolver=active_param_ensemble_resolver,
            active_context_ensemble_resolver=active_context_ensemble_resolver,
            ensemble_min_agree=int(ensemble_min_agree),
        )
        return _extract_active_replay_metrics((*result, pf_profile))

    base_context = contexts_by_signature[str(first_record["params_signature"])]
    benchmark_data = (base_context.get("all_dfs_fast") or {}).get(benchmark_ticker)

    def active_params_resolver(trade_date):
        return _resolve_active_schedule_record(schedule_records, trade_date)["params_obj"]

    def active_context_resolver(trade_date):
        record = _resolve_active_schedule_record(schedule_records, trade_date)
        return contexts_by_signature[str(record["params_signature"])]

    pf_profile = {
        "param_policy": "active_param_replay",
        "capture_equity_curve": True,
        "active_param_schedule": [
            {
                "effective_date": str(record.get("effective_date_text")),
                "year": int(record.get("year", 0) or 0),
                "params_signature": str(record.get("params_signature")),
            }
            for record in schedule_records
        ],
    }
    contexts_by_effective_date = {
        str(record.get("effective_date_text")): contexts_by_signature[str(record["params_signature"])]
        for record in schedule_records
    }
    _apply_active_single_stock_score_stats(
        pf_profile,
        schedule_records,
        contexts_by_effective_date,
        resolved_sorted_dates,
        ensemble=False,
    )
    result = run_portfolio_timeline(
        base_context.get("all_dfs_fast") or {},
        base_context.get("all_trade_logs") or {},
        resolved_sorted_dates,
        int(resolved_start_year),
        first_record["params_obj"],
        int(max_positions),
        bool(enable_rotation),
        benchmark_ticker=str(benchmark_ticker),
        benchmark_data=benchmark_data,
        is_training=False,
        profile_stats=pf_profile,
        verbose=False,
        pit_stats_index=base_context.get("all_pit_stats_index"),
        active_params_resolver=active_params_resolver,
        active_context_resolver=active_context_resolver,
    )
    return _extract_active_replay_metrics((*result, pf_profile))

def _build_active_replay_chained_oos_summary(
    *,
    rows: list[dict],
    config: OuterRollingConfig,
    selected_data_dir: str,
    output_dir: str,
    max_positions: int,
    enable_rotation: bool,
    overall_start: float | None = None,
) -> dict:
    if not rows:
        return {}
    first_oos_start = min(pd.Timestamp(row.get("oos_start_date") or f"{str(row.get('oos_year'))[:4]}-01-01").normalize() for row in rows)
    last_oos_end = max(pd.Timestamp(row.get("oos_end_date") or f"{str(row.get('oos_year'))[:4]}-12-31").normalize() for row in rows)
    first_year = int(first_oos_start.year)
    last_year = int(last_oos_end.year)
    selection_start_ts = min(pd.Timestamp(row.get("selection_start_date") or f"{int(row.get('selection_start_year', first_year))}-01-01").normalize() for row in rows)
    selection_end_ts = max(pd.Timestamp(row.get("selection_end_date") or f"{int(row.get('selection_end_year', last_year))}-12-31").normalize() for row in rows)
    selection_period_label = _period_label(selection_start_ts, selection_end_ts)
    oos_period_label = _period_label(first_oos_start, last_oos_end)

    payloads = {
        "best": _build_active_param_replay_payload_from_rows(
            rows,
            best_finalist=True,
            raw_universe_required_min_rows=config.raw_universe_required_min_rows,
        )
    }
    skipped_chain_policies: dict[str, str] = {}
    for policy_name in CHAIN_POLICY_NAMES:
        if _policy_has_complete_active_schedule(rows, policy_name):
            payloads[policy_name] = _build_active_param_replay_payload_from_rows(
                rows,
                policy_name=policy_name,
                raw_universe_required_min_rows=config.raw_universe_required_min_rows,
            )
        else:
            skipped_chain_policies[policy_name] = "incomplete_oos_schedule"
    total_elapsed_text = ""
    if overall_start is not None:
        total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
    print(
        f"{C_CYAN}⏳ OOS_CHAIN active replay | schedule policies={len(payloads)} | "
        f"period={oos_period_label}{total_elapsed_text}{C_RESET}",
        flush=True,
    )
    schedule_groups = {name: _build_active_replay_schedule_records(payload) for name, payload in payloads.items()}
    contexts_by_signature = _load_active_replay_contexts_by_signature(
        data_dir=selected_data_dir,
        schedule_groups=schedule_groups,
        output_dir=output_dir,
        first_year=first_year,
        last_year=last_year,
        overall_start=overall_start,
    )

    replay_metrics: dict[str, dict] = {}
    replay_started = time.perf_counter()
    replay_total = len(schedule_groups)
    replay_previous_width = 0
    replay_inline = stdout_supports_inline_progress()
    for replay_idx, (name, records) in enumerate(schedule_groups.items(), start=1):
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        replay_message = (
            f"{C_CYAN}⏳ OOS_CHAIN active replay [{replay_idx}/{replay_total}] | "
            f"policy={name} | schedule={len(records)} | "
            f"chain={_fmt_duration(time.perf_counter() - replay_started)}{total_elapsed_text}{C_RESET}"
        )
        if replay_inline:
            replay_previous_width = write_inline_progress(replay_message, previous_width=replay_previous_width)
        else:
            print(replay_message, flush=True)
        replay_metrics[name] = _run_active_replay_metrics_from_schedule_records(
            schedule_records=records,
            contexts_by_signature=contexts_by_signature,
            start_year=first_year,
            end_year=last_year,
            start_date=first_oos_start.strftime("%Y-%m-%d"),
            end_date=last_oos_end.strftime("%Y-%m-%d"),
            max_positions=max_positions,
            enable_rotation=enable_rotation,
        )
    if replay_total:
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        replay_done_message = (
            f"{C_CYAN}⏳ OOS_CHAIN active replay 完成 | "
            f"policies={replay_total}/{replay_total} | chain={_fmt_duration(time.perf_counter() - replay_started)}"
            f"{total_elapsed_text}{C_RESET}"
        )
        if replay_inline:
            write_inline_progress(replay_done_message, previous_width=replay_previous_width)
            print()
        else:
            print(replay_done_message, flush=True)
    for policy_name, reason in skipped_chain_policies.items():
        metrics = _empty_unavailable_chain_metrics()
        metrics["unavailable_reason"] = str(reason)
        replay_metrics[policy_name] = metrics

    best_metrics = dict(replay_metrics.get("best") or {})
    benchmark_source = dict(best_metrics)
    for policy_name in CHAIN_POLICY_NAMES:
        if not benchmark_source and replay_metrics.get(policy_name):
            benchmark_source = dict(replay_metrics[policy_name])
            break

    best_score = float(best_metrics.get("score", 0.0))
    best_return = float(best_metrics.get("return_pct", 0.0))
    benchmark_score = float(benchmark_source.get("benchmark_oos_score", 0.0))
    benchmark_return = float(benchmark_source.get("benchmark_return_pct", 0.0))
    summary = {
        "method": "continuous_active_param_replay",
        "score_aggregation_method": "continuous_active_param_replay_recomputed_score",
        "return_aggregation_method": "continuous_active_param_replay_total_return",
        "note": "OOS_CHAIN 由 portfolio_sim active-param replay 連續重跑後重算 score；持股、現金與 benchmark 跨 fold 延續，不使用獨立 OOS closeout stitch 或區間 score mean。",
        "selection_period": selection_period_label,
        "oos_period": oos_period_label,
        "max_positions": int(max_positions),
        "enable_rotation": bool(enable_rotation),
        "best_finalist_oos_score": float(best_score),
        "benchmark_oos_score": float(benchmark_score),
        "best_finalist_return_pct": float(best_return),
        "benchmark_return_pct": float(benchmark_return),
        "benchmark_alpha_pct": float(best_return - benchmark_return),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "benchmark_mdd_pct": float(benchmark_source.get("benchmark_mdd_pct", 0.0)),
        "best_finalist_annual_return_pct": float(best_metrics.get("annual_return_pct", 0.0)),
        "benchmark_annual_return_pct": float(benchmark_source.get("benchmark_annual_return_pct", 0.0)),
        "best_finalist_curve_points": int(best_metrics.get("curve_points", 0)),
        "benchmark_curve_points": int(benchmark_source.get("curve_points", 0)),
        "yearly_best_finalist_oos_score": [float(row.get("best_finalist_oos_score", 0.0)) for row in rows],
        "yearly_benchmark_oos_score": [float(row.get("benchmark_oos_score", 0.0)) for row in rows],
        "skipped_chain_policies": dict(skipped_chain_policies),
    }
    for policy_name in CHAIN_POLICY_NAMES:
        metrics = dict(replay_metrics.get(policy_name) or {})
        chain_score = float(metrics.get("score", 0.0))
        chain_plain_romd = float(metrics.get("plain_romd_score", calc_plain_romd(metrics.get("return_pct", 0.0), metrics.get("mdd_pct", 0.0))))
        chain_return = float(metrics.get("return_pct", 0.0))
        summary[policy_name] = {
            "available": bool(metrics.get("available", int(metrics.get("curve_points", 0) or 0) > 0)),
            "rank_1_oos": float(chain_score),
            "rank_1_plain_romd": float(chain_plain_romd),
            "best_gap": float(chain_score - best_score),
            "benchmark_0050_gap": float(chain_score - benchmark_score),
            "benchmark_0050_plain_romd_gap": float(chain_plain_romd - benchmark_score),
            "rank_1_return_pct": float(chain_return),
            "best_gap_pct": float(chain_return - best_return),
            "benchmark_0050_gap_pct": float(chain_return - benchmark_return),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_annual_return_pct": float(metrics.get("annual_return_pct", 0.0)),
            "rank_1_curve_points": int(metrics.get("curve_points", 0)),
            "rank_1_trades": int(metrics.get("trade_count", 0)),
            "unavailable_reason": str(metrics.get("unavailable_reason") or skipped_chain_policies.get(policy_name, "")),
            "yearly_oos_score": [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows],
            "yearly_return_pct": [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows],
        }
    return summary

