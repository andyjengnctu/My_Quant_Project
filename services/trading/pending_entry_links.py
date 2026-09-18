"""Read-only identity links between confirmed pending intents and account BUYs.

AI: A correction is new account evidence, not a new pending intent. Resolve its
original audit event before joining so constraints and order dates survive edits.
No ticker/date coincidence alone can establish a lineage.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

from core.exact_accounting import price_to_milli
from core.trading_account_state import TRADE_MUTATION_MANUAL_MANAGED_BUY, TRADE_MUTATION_STRATEGY_BUY


def original_buy_event(event: Mapping[str, Any], account_events: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_revision = {int(row['revision']): row for row in account_events if row.get('revision') is not None}
    current = event
    visited: set[int] = set()
    while True:
        details = dict(current.get('details') or {})
        prior = details.get('replaces_event_revision')
        if prior is None:
            prior = details.get('replacement_for_revision')
        if prior is None:
            return current
        revision = int(prior)
        if revision in visited:
            raise RuntimeError('Trading correction event chain contains a cycle')
        visited.add(revision)
        if revision not in by_revision:
            raise RuntimeError('Trading correction is missing its original account event')
        current = by_revision[revision]


def pending_entry_matches_buy(entry: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    if str(entry.get('status') or '') != 'FILLED':
        return False
    fields = {TRADE_MUTATION_STRATEGY_BUY: 'strategy_lineage', TRADE_MUTATION_MANUAL_MANAGED_BUY: 'management_lineage'}
    field = fields.get(str(event.get('mutation_type') or ''))
    if field is None:
        return False
    details = dict(event.get('details') or {})
    if str(entry.get('ticker') or '').strip().upper() != str(details.get('ticker') or '').strip().upper():
        return False
    lineage = details.get(field)
    if not isinstance(lineage, Mapping):
        lineage = (details.get('position_after') or {}).get(field)
    expected = str((entry.get('management_lineage') or {}).get('lineage_id') or '')
    if not expected or not isinstance(lineage, Mapping) or str(lineage.get('lineage_id') or '') != expected:
        return False
    fill = dict(entry.get('fill') or {})
    if str(fill.get('trade_date') or '') != str(details.get('trade_date') or ''):
        return False
    if int(fill.get('qty') or 0) != int(details.get('qty') or details.get('fill_qty') or 0):
        return False
    price_milli = details.get('entry_fill_price_milli', details.get('fill_price_milli'))
    if fill.get('price') is not None and price_milli is not None:
        return price_to_milli(fill['price']) == int(price_milli)
    return True


def resolve_pending_entry_for_buy_event(
    event: Mapping[str, Any],
    pending_entries: Iterable[Mapping[str, Any]],
    *,
    account_events: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    original = original_buy_event(event, account_events)
    matches = [dict(row) for row in pending_entries if pending_entry_matches_buy(row, original)]
    if not matches:
        return None

    def key(entry):
        distance = float('inf')
        try:
            fill_time = (entry.get('fill') or {}).get('confirmed_at') or entry.get('closed_at')
            distance = abs((datetime.fromisoformat(str(fill_time)) - datetime.fromisoformat(str(original.get('timestamp')))).total_seconds())
        except (TypeError, ValueError):
            # AI: Legacy absent timestamps sort deterministically after precise links.
            distance = float('inf')
        return distance, str(entry.get('pending_entry_id') or '')

    return min(matches, key=key)


__all__ = ['original_buy_event', 'pending_entry_matches_buy', 'resolve_pending_entry_for_buy_event']
