import math
import time

from core.breakout_reentry import activate_breakout_reentry_signals_for_day
from core.buy_sort import (
    BUY_LIMIT_OVERAGE_SORT_METHOD,
    ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD,
    build_breakout_quality_ranking_prefixes,
    calc_buy_limit_overage_pct_from_row,
    calc_entry_type_priority_from_row,
    resolve_breakout_quality_ranking_policy,
)
from core.capital_policy import resolve_portfolio_sizing_equity
from core.config import get_buy_sort_method
from core.portfolio_candidates import build_daily_candidates, track_normal_setup_signals_for_day
from core.portfolio_ops import cleanup_extended_signals_for_day
from core.portfolio_replay_support import _print_slow_ensemble_phase_heartbeat


def _iter_ensemble_member_pairs(ensemble_members, ensemble_contexts):
    members = list(ensemble_members or [])
    contexts = list(ensemble_contexts or [])
    for idx, member in enumerate(members):
        context = contexts[idx] if idx < len(contexts) and isinstance(contexts[idx], dict) else {}
        yield member, context


def _resolve_ensemble_member_key(member, idx):
    for key in ("member_key", "member_index", "seed"):
        value = member.get(key) if isinstance(member, dict) else None
        if value is not None and str(value).strip() != "":
            return str(value)
    return str(idx)


def _annotate_ensemble_candidate(candidate, *, member, params_obj, member_key, context):
    row = dict(candidate)
    row["params_obj"] = params_obj
    row["_ensemble_context"] = context
    row["ensemble_member_key"] = member_key
    row["ensemble_member_index"] = member.get("member_index") if isinstance(member, dict) else None
    row["ensemble_seed"] = member.get("seed") if isinstance(member, dict) else None
    row["params_signature"] = member.get("params_signature") if isinstance(member, dict) else ""
    return row


def _median_candidate_row(rows):
    ordered = sorted(rows, key=lambda item: (float(item.get("sort_value", 0.0) or 0.0), str(item.get("ticker") or "")))
    return dict(ordered[(len(ordered) - 1) // 2])


def _aggregate_ensemble_candidate_rows(rows, *, min_agree):
    grouped = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        grouped.setdefault(ticker, []).append(row)

    aggregated = []
    for ticker, group_rows in grouped.items():
        member_keys = {str(row.get("ensemble_member_key") or "") for row in group_rows}
        member_keys.discard("")
        vote_count = len(member_keys)
        if vote_count < int(min_agree):
            continue
        representative = _median_candidate_row(group_rows)
        sort_values = sorted(float(row.get("sort_value", 0.0) or 0.0) for row in group_rows)
        median_sort = sort_values[(len(sort_values) - 1) // 2]
        member_params_by_key = {
            str(row.get("ensemble_member_key") or "").strip(): row.get("params_obj")
            for row in group_rows
            if str(row.get("ensemble_member_key") or "").strip() and row.get("params_obj") is not None
        }
        representative["ensemble_vote_count"] = int(vote_count)
        representative["ensemble_min_agree"] = int(min_agree)
        representative["ensemble_member_count"] = int(len(member_keys))
        representative["ensemble_median_sort_value"] = float(median_sort)
        representative["sort_value"] = float(median_sort)
        representative["ensemble_member_keys"] = sorted(member_keys)
        ranking_policies = {
            str(row.get("breakout_quality_ranking_policy") or "score").strip()
            for row in group_rows
        }
        if len(ranking_policies) > 1:
            raise ValueError(
                f"同一 ticker 的 ensemble members ranking policy不一致: ticker={ticker}"
            )
        representative["breakout_quality_ranking_policy"] = next(iter(ranking_policies), "score")
        ranking_options = {
            repr(sorted(dict(row.get("breakout_quality_ranking_options") or {}).items()))
            for row in group_rows
        }
        if len(ranking_options) > 1:
            raise ValueError(
                f"同一 ticker 的 ensemble members ranking options不一致: ticker={ticker}"
            )
        representative["breakout_quality_ranking_options"] = dict(
            group_rows[0].get("breakout_quality_ranking_options") or {}
        )
        for metric_name in (
            "projected_capital_fraction",
            "projected_capital_deployment_rate",
            "max_position_cap_pct",
        ):
            metric_values = []
            for row in group_rows:
                try:
                    metric_value = float(row.get(metric_name))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(metric_value):
                    metric_values.append(metric_value)
            if metric_values:
                metric_values.sort()
                representative[metric_name] = float(
                    metric_values[(len(metric_values) - 1) // 2]
                )
        ranking_flags = {bool(row.get("use_breakout_quality_ranking", False)) for row in group_rows}
        if len(ranking_flags) > 1:
            raise ValueError(f"同一 ticker 的 ensemble members quality ranking 設定不一致: ticker={ticker}")
        quality_ranking = bool(ranking_flags and True in ranking_flags)
        representative["use_breakout_quality_ranking"] = quality_ranking
        if quality_ranking:
            quality_rows = []
            member_quality_rank_by_key = {}
            availability_flags = set()
            score_sources = set()
            for row in group_rows:
                member_key = str(row.get("ensemble_member_key") or "").strip()
                rank_payload = row.get("breakout_quality_rank")
                if not isinstance(rank_payload, dict):
                    raw_score = row.get("breakout_quality_score")
                    try:
                        parsed_score = float(raw_score)
                    except (TypeError, ValueError):
                        parsed_score = float("nan")
                    rank_payload = {
                        "score": parsed_score if math.isfinite(parsed_score) else None,
                        "available": bool(math.isfinite(parsed_score)),
                        "unavailable_reason": "" if math.isfinite(parsed_score) else "missing_score",
                        "score_date": row.get("breakout_quality_score_date"),
                        "score_source": row.get("breakout_quality_score_source") or "canonical_runtime",
                        "shared_group_score": True,
                        "filter_id": str(getattr(row.get("params_obj"), "breakout_quality_filter_id", "") or ""),
                    }
                available = bool(rank_payload.get("available", False))
                score_date = str(rank_payload.get("score_date") or "").strip()
                score_source = str(rank_payload.get("score_source") or "canonical_runtime").strip()
                availability_flags.add(available)
                score_sources.add(score_source)
                normalized_rank = dict(rank_payload)
                normalized_rank.update({
                    "available": available,
                    "score_date": score_date,
                    "score_source": score_source,
                })
                if available:
                    try:
                        score = float(rank_payload.get("score"))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"啟用 quality ranking 的 ensemble 候選分數不是有限數值: ticker={ticker}"
                        ) from exc
                    if not math.isfinite(score) or not score_date:
                        raise ValueError(
                            f"啟用 quality ranking 的 ensemble 候選缺少有效原始事件分數: ticker={ticker}"
                        )
                    normalized_rank["score"] = score
                    quality_rows.append((score, score_date, normalized_rank))
                else:
                    normalized_rank["score"] = None
                if member_key:
                    member_quality_rank_by_key[member_key] = dict(normalized_rank)

            if len(score_sources) > 1:
                raise ValueError(f"同一 ticker 的 ensemble members Score source不一致: ticker={ticker}")

            availability_by_score_date = {}
            for member_key, rank_payload in member_quality_rank_by_key.items():
                score_date = str(rank_payload.get("score_date") or "").strip()
                if not score_date:
                    continue
                availability_by_score_date.setdefault(score_date, set()).add(
                    bool(rank_payload.get("available", False))
                )
            inconsistent_dates = sorted(
                score_date
                for score_date, flags in availability_by_score_date.items()
                if len(flags) > 1
            )
            if inconsistent_dates:
                raise ValueError(
                    "同一 ticker／score_date 的 ensemble members Score availability不一致: "
                    f"ticker={ticker}, score_dates={inconsistent_dates}"
                )

            available_member_keys = sorted(
                member_key
                for member_key, rank_payload in member_quality_rank_by_key.items()
                if bool(rank_payload.get("available", False))
            )
            unavailable_member_keys = sorted(
                member_key
                for member_key, rank_payload in member_quality_rank_by_key.items()
                if not bool(rank_payload.get("available", False))
            )
            representative["ensemble_quality_score_available_member_count"] = len(available_member_keys)
            representative["ensemble_quality_score_unavailable_member_count"] = len(unavailable_member_keys)
            representative["ensemble_quality_score_available_member_keys"] = available_member_keys
            representative["ensemble_quality_score_unavailable_member_keys"] = unavailable_member_keys

            if availability_flags == {True}:
                quality_rows.sort(key=lambda item: (item[0], item[1]))
                median_score, median_score_date, median_rank = quality_rows[(len(quality_rows) - 1) // 2]
                representative["ensemble_quality_score_availability"] = "all_available"
                representative["ensemble_median_quality_score"] = float(median_score)
                representative["ensemble_median_quality_score_date"] = median_score_date
                representative["ensemble_quality_score_dates"] = sorted({item[1] for item in quality_rows if item[1]})
                representative["breakout_quality_score"] = float(median_score)
                representative["breakout_quality_score_date"] = median_score_date
                representative["breakout_quality_score_source"] = str(median_rank.get("score_source") or "")
                representative["breakout_quality_rank"] = dict(median_rank)
            else:
                fallback_member_key = (unavailable_member_keys or sorted(member_quality_rank_by_key))[0]
                fallback_rank = dict(member_quality_rank_by_key.get(fallback_member_key) or {})
                partial_availability = bool(available_member_keys and unavailable_member_keys)
                if partial_availability:
                    fallback_rank.update({
                        "score": None,
                        "available": False,
                        "unavailable_reason": "partial_ensemble_member_score_availability",
                        "score_date": "",
                        "shared_group_score": False,
                    })
                    representative["ensemble_quality_score_availability"] = "partial_available_fallback"
                else:
                    representative["ensemble_quality_score_availability"] = "none_available"
                representative["ensemble_median_quality_score"] = None
                representative["ensemble_median_quality_score_date"] = str(fallback_rank.get("score_date") or "")
                representative["ensemble_quality_score_dates"] = sorted(
                    {str(item.get("score_date") or "") for item in member_quality_rank_by_key.values()} - {""}
                )
                representative["breakout_quality_score"] = None
                representative["breakout_quality_score_date"] = str(fallback_rank.get("score_date") or "")
                representative["breakout_quality_score_source"] = str(fallback_rank.get("score_source") or "")
                representative["breakout_quality_rank"] = fallback_rank
            representative["ensemble_member_quality_rank_by_key"] = member_quality_rank_by_key
        # # (AI註: STOP 後 Re-entry 必須保留原始共識 member 的各自參數與原始 breakout score；
        # #        只保存代表 member 會讓 min_agree>1 的 re-entry 共識或 score 語意分叉。)
        representative["ensemble_member_params_by_key"] = member_params_by_key
        aggregated.append(representative)
    active_sort_method = get_buy_sort_method()
    quality_ranking = bool(aggregated and aggregated[0].get("use_breakout_quality_ranking", False))
    if any(bool(item.get("use_breakout_quality_ranking", False)) != quality_ranking for item in aggregated):
        raise ValueError("同日 aggregated candidates 的 quality ranking 設定不一致")
    if quality_ranking:
        resolve_breakout_quality_ranking_policy(aggregated)
        quality_prefixes = build_breakout_quality_ranking_prefixes(aggregated)
    else:
        quality_prefixes = {id(item): () for item in aggregated}

    def _quality_prefix(item):
        return quality_prefixes.get(id(item), ())

    if active_sort_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        aggregated.sort(
            key=lambda item: (
                -int(item.get("ensemble_vote_count", 0) or 0),
                *_quality_prefix(item),
                float(item.get("ensemble_median_sort_value", item.get("sort_value", 0.0)) or 0.0),
                -float(item.get("proj_cost", 0.0) or 0.0),
                str(item.get("ticker") or ""),
            )
        )
    elif active_sort_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        aggregated.sort(
            key=lambda item: (
                -int(item.get("ensemble_vote_count", 0) or 0),
                *_quality_prefix(item),
                calc_entry_type_priority_from_row(item),
                calc_buy_limit_overage_pct_from_row(item),
                str(item.get("ticker") or ""),
            )
        )
    else:
        aggregated.sort(
            key=lambda item: (
                -int(item.get("ensemble_vote_count", 0) or 0),
                *_quality_prefix(item),
                -float(item.get("ensemble_median_sort_value", item.get("sort_value", 0.0)) or 0.0),
                str(item.get("ticker") or ""),
            )
        )
    return aggregated


def _flatten_ensemble_extended_signals(active_extended_signals_by_member):
    flattened = {}
    for member_signals in (active_extended_signals_by_member or {}).values():
        for ticker, signal_state in (member_signals or {}).items():
            flattened.setdefault(ticker, signal_state)
    return flattened


def _build_daily_ensemble_candidates(
    *,
    ensemble_members,
    ensemble_contexts,
    active_extended_signals_by_member,
    portfolio,
    sold_today,
    today,
    current_equity_money,
    initial_capital,
    collect_all_candidates,
    min_agree,
    verbose=False,
):
    all_candidate_rows = []
    all_orderable_rows = []
    member_pairs = list(_iter_ensemble_member_pairs(ensemble_members, ensemble_contexts))
    phase_start = time.perf_counter()
    for idx, (member, context) in enumerate(member_pairs, start=1):
        params_obj = member.get("params_obj") if isinstance(member, dict) else None
        if params_obj is None:
            continue
        member_key = _resolve_ensemble_member_key(member, idx)
        member_signals = active_extended_signals_by_member.setdefault(member_key, {})
        member_pit_cursor = member.setdefault("pit_stats_cursor", {}) if isinstance(member, dict) else {}
        member_all_dfs_fast = context.get("all_dfs_fast") or {}
        member_pit_stats_index = context.get("all_pit_stats_index") or {}
        member_normal_setup_index = context.get("normal_setup_index") or {}
        sizing_equity = resolve_portfolio_sizing_equity(current_equity_money, initial_capital, params_obj)
        candidates_today, orderable_candidates_today, _normal_setup_tickers_today = build_daily_candidates(
            normal_setup_index=member_normal_setup_index,
            active_extended_signals=member_signals,
            portfolio=portfolio,
            sold_today=sold_today,
            all_dfs_fast=member_all_dfs_fast,
            pit_stats_index=member_pit_stats_index,
            pit_stats_cursor=member_pit_cursor,
            today=today,
            sizing_equity=sizing_equity,
            params=params_obj,
            collect_all_candidates=collect_all_candidates,
        )
        for row in candidates_today:
            all_candidate_rows.append(_annotate_ensemble_candidate(row, member=member, params_obj=params_obj, member_key=member_key, context=context))
        for row in orderable_candidates_today:
            all_orderable_rows.append(_annotate_ensemble_candidate(row, member=member, params_obj=params_obj, member_key=member_key, context=context))
        _print_slow_ensemble_phase_heartbeat(
            verbose=verbose,
            phase="candidate_scan",
            today=today,
            completed=idx,
            total=len(member_pairs),
            started_at=phase_start,
        )

    candidates = _aggregate_ensemble_candidate_rows(all_candidate_rows, min_agree=min_agree) if collect_all_candidates else []
    orderable = _aggregate_ensemble_candidate_rows(all_orderable_rows, min_agree=min_agree)
    normal_setup_tickers = {str(row.get("ticker") or "") for row in candidates if str(row.get("type") or "") == "normal"}
    normal_setup_tickers.discard("")
    return candidates, orderable, normal_setup_tickers


def _track_ensemble_normal_setup_signals_for_day(
    *,
    ensemble_members,
    ensemble_contexts,
    active_extended_signals_by_member,
    portfolio,
    sold_today,
    today,
):
    for idx, (member, context) in enumerate(_iter_ensemble_member_pairs(ensemble_members, ensemble_contexts), start=1):
        params_obj = member.get("params_obj") if isinstance(member, dict) else None
        if params_obj is None:
            continue
        member_key = _resolve_ensemble_member_key(member, idx)
        member_signals = active_extended_signals_by_member.setdefault(member_key, {})
        member_pit_cursor = member.setdefault("pit_stats_cursor", {}) if isinstance(member, dict) else {}
        track_normal_setup_signals_for_day(
            normal_setup_entries=(context.get("normal_setup_index") or {}).get(today, []),
            portfolio=portfolio,
            sold_today=sold_today,
            all_dfs_fast=context.get("all_dfs_fast") or {},
            active_extended_signals=member_signals,
            pit_stats_index=context.get("all_pit_stats_index") or {},
            pit_stats_cursor=member_pit_cursor,
            today=today,
            params=params_obj,
        )


def _activate_ensemble_reentry_signals_for_day(
    *,
    ensemble_members,
    ensemble_contexts,
    active_reentry_watchlists_by_member,
    active_extended_signals_by_member,
    portfolio,
    sold_today,
    today,
):
    for idx, (member, context) in enumerate(_iter_ensemble_member_pairs(ensemble_members, ensemble_contexts), start=1):
        params_obj = member.get("params_obj") if isinstance(member, dict) else None
        if params_obj is None:
            continue
        member_key = _resolve_ensemble_member_key(member, idx)
        member_watchlist = active_reentry_watchlists_by_member.setdefault(member_key, {})
        member_signals = active_extended_signals_by_member.setdefault(member_key, {})
        activate_breakout_reentry_signals_for_day(
            active_reentry_watchlist=member_watchlist,
            active_extended_signals=member_signals,
            portfolio=portfolio,
            sold_today=sold_today,
            all_dfs_fast=context.get("all_dfs_fast") or {},
            today=today,
            params=params_obj,
        )


def _cleanup_ensemble_extended_signals_for_day(
    *,
    ensemble_members,
    ensemble_contexts,
    active_extended_signals_by_member,
    portfolio,
    today,
    current_equity_money,
    initial_capital,
    verbose=False,
):
    member_pairs = list(_iter_ensemble_member_pairs(ensemble_members, ensemble_contexts))
    phase_start = time.perf_counter()
    for idx, (member, context) in enumerate(member_pairs, start=1):
        params_obj = member.get("params_obj") if isinstance(member, dict) else None
        if params_obj is None:
            continue
        member_key = _resolve_ensemble_member_key(member, idx)
        member_signals = active_extended_signals_by_member.setdefault(member_key, {})
        sizing_equity = resolve_portfolio_sizing_equity(current_equity_money, initial_capital, params_obj)
        cleanup_extended_signals_for_day(
            active_extended_signals=member_signals,
            portfolio=portfolio,
            all_dfs_fast=context.get("all_dfs_fast") or {},
            today=today,
            params=params_obj,
            sizing_capital=sizing_equity,
        )
        _print_slow_ensemble_phase_heartbeat(
            verbose=verbose,
            phase="extended_signal_cleanup",
            today=today,
            completed=idx,
            total=len(member_pairs),
            started_at=phase_start,
        )


