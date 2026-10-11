"""Portfolio Replay-only capture: trace, candidate snapshots, and trade rows.

Must not change ranking, order reservation, cash or scientific metric logic.
"""

def append_portfolio_selector_trace(
    *, today, replay_selector_trace_rows, resource_selection_input_rows,
    resource_selection_diag, available_cash, sizing_equity, params,
    repair_mechanism_builder, candidate_snapshot,
):
    """Emit original selector-repair and basket trace schema; no ranking logic."""
    if replay_selector_trace_rows is None:
        return
    repair_mechanism = repair_mechanism_builder(
        resource_selection_input_rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        params=params,
        resource_selection_diag=resource_selection_diag,
    )
    if repair_mechanism is not None:
        summary = {
            'trace_kind': 'repair_summary',
            'stage': '',
            'stage_rank': 0,
            'ticker': '',
            'trade_date': today.strftime('%Y-%m-%d') if hasattr(today, 'strftime') else str(today),
            'candidate_date': '',
            'signal_date': '',
            'candidate_type': '',
            'entry_source': '',
            'breakout_quality_score': None,
            'breakout_quality_score_date': '',
            'breakout_quality_score_available': False,
            'pre_market_order_limit': resource_selection_diag.get('pre_market_order_limit'),
            'direct_score_order_feasible': bool(resource_selection_diag.get('direct_score_order_feasible', False)),
            'repair_steps': int(resource_selection_diag.get('max_dl_repair_steps', 0) or 0),
            'ascent_steps': int(resource_selection_diag.get('max_dl_feasible_ascent_steps', 0) or 0),
            **{
                key: value
                for key, value in dict(repair_mechanism).items()
                if key != 'repair_steps_trace'
            },
        }
        replay_selector_trace_rows.append(summary)

        for step in list(repair_mechanism.get('repair_steps_trace') or []):
            for role, candidate_key in (('out', 'out_row'), ('in', 'in_row')):
                candidate = step.get(candidate_key)
                if candidate is None:
                    continue
                snapshot = candidate_snapshot(
                    candidate,
                    fallback_trade_date=today,
                    is_orderable=True,
                )
                snapshot.update({
                    'trace_kind': 'repair_swap',
                    'repair_role': role,
                    'repair_step': int(step.get('step', 0) or 0),
                    'stage': '',
                    'stage_rank': 0,
                    'pre_market_order_limit': resource_selection_diag.get('pre_market_order_limit'),
                    'direct_score_order_feasible': bool(resource_selection_diag.get('direct_score_order_feasible', False)),
                    'repair_steps': int(resource_selection_diag.get('max_dl_repair_steps', 0) or 0),
                    'ascent_steps': int(resource_selection_diag.get('max_dl_feasible_ascent_steps', 0) or 0),
                    'before_selected_count': step.get('before_selected_count'),
                    'after_selected_count': step.get('after_selected_count'),
                    'before_reserved_cost_milli': step.get('before_reserved_cost_milli'),
                    'after_reserved_cost_milli': step.get('after_reserved_cost_milli'),
                    'before_count_deficit': step.get('before_count_deficit'),
                    'after_count_deficit': step.get('after_count_deficit'),
                    'before_reserve_deficit_milli': step.get('before_reserve_deficit_milli'),
                    'after_reserve_deficit_milli': step.get('after_reserve_deficit_milli'),
                    'after_feasible': bool(step.get('after_feasible', False)),
                    'evaluated_swap_count': step.get('evaluated_swap_count'),
                    'progress_swap_count': step.get('progress_swap_count'),
                    'feasible_swap_count': step.get('feasible_swap_count'),
                })
                replay_selector_trace_rows.append(snapshot)

    trace_baskets = dict(resource_selection_diag.get('_selector_trace_baskets') or {})
    for stage_name in ('raw_top_n', 'minimum_repair_seed', 'feasible_ascent_final'):
        stage_rows = list(trace_baskets.get(stage_name) or [])
        for stage_rank, candidate in enumerate(stage_rows, start=1):
            snapshot = candidate_snapshot(
                candidate,
                fallback_trade_date=today,
                is_orderable=True,
            )
            snapshot.update({
                'trace_kind': 'basket_stage',
                'stage': stage_name,
                'stage_rank': int(stage_rank),
                'pre_market_order_limit': resource_selection_diag.get('pre_market_order_limit'),
                'direct_score_order_feasible': bool(resource_selection_diag.get('direct_score_order_feasible', False)),
                'repair_steps': int(resource_selection_diag.get('max_dl_repair_steps', 0) or 0),
                'ascent_steps': int(resource_selection_diag.get('max_dl_feasible_ascent_steps', 0) or 0),
            })
            replay_selector_trace_rows.append(snapshot)




def record_portfolio_candidate_replay(
    *, today, replay_counts, candidates_today, orderable_candidates_today,
    replay_execution_rows, trade_history, all_dfs_fast, sizing_equity,
    candidate_snapshot, execution_snapshot,
):
    """Record candidate/orderable/execution rows in the original phase order."""
    if replay_counts is not None:
        for candidate in candidates_today:
            snapshot = candidate_snapshot(
                candidate, fallback_trade_date=today, is_orderable=False,
            )
            ticker = str(snapshot.get("ticker", ""))
            bucket = replay_counts.setdefault(
                ticker,
                {
                    "candidate_dates": [], "orderable_dates": [],
                    "candidate_rows": [], "orderable_rows": [], "trade_rows": [],
                },
            )
            bucket.setdefault("candidate_dates", []).append(today)
            bucket.setdefault("candidate_rows", []).append(snapshot)
        for candidate in orderable_candidates_today:
            snapshot = candidate_snapshot(
                candidate, fallback_trade_date=today, is_orderable=True,
            )
            ticker = str(snapshot.get("ticker", ""))
            bucket = replay_counts.setdefault(
                ticker,
                {
                    "candidate_dates": [], "orderable_dates": [],
                    "candidate_rows": [], "orderable_rows": [], "trade_rows": [],
                },
            )
            bucket.setdefault("orderable_dates", []).append(today)
            bucket.setdefault("orderable_rows", []).append(snapshot)
        before_trade_rows = len(trade_history)
    else:
        before_trade_rows = -1
    if replay_execution_rows is not None:
        for candidate in orderable_candidates_today:
            replay_execution_rows.append(
                execution_snapshot(
                    candidate, fallback_trade_date=today,
                    all_dfs_fast=all_dfs_fast, sizing_equity=sizing_equity,
                )
            )
    return before_trade_rows


def initialize_portfolio_replay_counts(replay_counts):
    """Normalize existing replay buckets once; retain their dictionary identity."""
    if replay_counts is not None:
        for ticker, bucket in replay_counts.items():
            if not isinstance(bucket, dict):
                replay_counts[ticker] = {}
                bucket = replay_counts[ticker]
            bucket.setdefault("candidate_dates", [])
            bucket.setdefault("orderable_dates", [])
            bucket.setdefault("candidate_rows", [])
            bucket.setdefault("orderable_rows", [])
            bucket.setdefault("trade_rows", [])


def append_portfolio_trade_replay(*, replay_counts, before_trade_rows, trade_history):
    """Append post-execution trade rows after the equity/level phases."""
    if replay_counts is not None and before_trade_rows >= 0:
        for row in trade_history[before_trade_rows:]:
            ticker = str(row.get("Ticker", "")).strip()
            bucket = replay_counts.setdefault(
                ticker,
                {
                    "candidate_dates": [], "orderable_dates": [],
                    "candidate_rows": [], "orderable_rows": [], "trade_rows": [],
                },
            )
            bucket.setdefault("trade_rows", []).append(row)
