"""Resolve per-session replay sources without owning the trading algorithm.

This is the only Portfolio Engine owner of daily single-member versus
multi-member replay selection and PIT context/cursor fallback. No scoring,
param fitting, or candidate membership is performed here.
"""

from typing import Any, NamedTuple

from core.seed_ensemble_policy import resolve_seed_ensemble_min_agree


class PortfolioDayContext(NamedTuple):
    ensemble_members: list
    ensemble_contexts: list
    ensemble_min_agree: int | None
    use_param_ensemble: bool
    params: Any
    all_dfs_fast: Any
    pit_stats_index: Any
    normal_setup_index: Any
    pit_stats_cursor: dict


def resolve_portfolio_day_context(
    today,
    *,
    params,
    all_dfs_fast,
    pit_stats_index,
    normal_setup_index,
    pit_stats_cursor,
    active_params_resolver,
    active_context_resolver,
    active_param_ensemble_resolver,
    active_context_ensemble_resolver,
    ensemble_min_agree,
) -> PortfolioDayContext:
    """Resolve sources once per day in the original callback order.

    The single-member optimization deliberately reuses the ordinary candidate
    path; the first ensemble member nevertheless supplies parameters/context.
    The PIT cursor is isolated by the resolved stats-index object's identity.
    """
    day_ensemble_members = list(active_param_ensemble_resolver(today) or []) if active_param_ensemble_resolver is not None else []
    day_ensemble_contexts = list(active_context_ensemble_resolver(today) or []) if active_context_ensemble_resolver is not None else []
    day_ensemble_min_agree = resolve_seed_ensemble_min_agree(
        len(day_ensemble_members),
        ensemble_min_agree if ensemble_min_agree is not None else "auto",
    ) if day_ensemble_members else None
    single_member_identity_replay = bool(
        len(day_ensemble_members) == 1
        and int(day_ensemble_min_agree or 1) <= 1
    )
    use_param_ensemble = bool(day_ensemble_members) and not single_member_identity_replay
    if day_ensemble_members:
        day_params = day_ensemble_members[0].get("params_obj") or params
        day_context = day_ensemble_contexts[0] if day_ensemble_contexts else None
    else:
        day_params = active_params_resolver(today) if active_params_resolver is not None else params
        day_context = active_context_resolver(today) if active_context_resolver is not None else None
    if day_context is None:
        day_all_dfs_fast = all_dfs_fast
        day_pit_stats_index = pit_stats_index
        day_normal_setup_index = normal_setup_index
        day_pit_stats_cursor = pit_stats_cursor
    else:
        day_all_dfs_fast = day_context.get('all_dfs_fast') or all_dfs_fast
        day_pit_stats_index = day_context.get('all_pit_stats_index') or pit_stats_index
        day_normal_setup_index = day_context.get('normal_setup_index') or normal_setup_index
        day_cursor_key = id(day_pit_stats_index)
        day_pit_stats_cursor = pit_stats_cursor.setdefault(day_cursor_key, {})
    return PortfolioDayContext(
        day_ensemble_members,
        day_ensemble_contexts,
        day_ensemble_min_agree,
        use_param_ensemble,
        day_params,
        day_all_dfs_fast,
        day_pit_stats_index,
        day_normal_setup_index,
        day_pit_stats_cursor,
    )


def portfolio_has_day_work(
    today,
    *,
    day_context: PortfolioDayContext,
    portfolio,
    active_extended_signals,
    active_extended_signals_by_member,
    active_reentry_watchlist,
    active_reentry_watchlists_by_member,
    normal_setup_entries_today,
) -> bool:
    """Preserve the pre-market idle fast-path predicate across replay modes."""
    if day_context.use_param_ensemble:
        has_ensemble_normal_setup = any(bool((ctx.get("normal_setup_index") or {}).get(today, [])) for ctx in day_context.ensemble_contexts)
        has_ensemble_extended = any(bool(member_signals) for member_signals in active_extended_signals_by_member.values())
        has_ensemble_reentry_watch = any(bool(member_watchlist) for member_watchlist in active_reentry_watchlists_by_member.values())
        return bool(portfolio) or bool(has_ensemble_extended) or bool(has_ensemble_normal_setup) or bool(has_ensemble_reentry_watch)
    return bool(portfolio) or bool(active_extended_signals) or bool(active_reentry_watchlist) or bool(normal_setup_entries_today)
