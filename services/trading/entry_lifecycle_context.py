"""One as-of acquisition gateway for confirmed Trading entry adapters.

AI: Accepted order choices stay immutable. Only strategy management is replayed
from frozen lineage to the instant before the reported fill. No fill inference.
"""
from copy import deepcopy
import pandas as pd
from core.data_utils import get_required_min_rows
from core.trading_lifecycle_plans import resolve_confirmed_entry_plan_from_frame
from services.trading.lifecycle_context import resolve_trading_lifecycle_context
from services.trading.market_data_consumer import load_trading_v2_sanitized_ohlcv_frame


def resolve_trading_confirmed_entry_seed(project_root, *, ticker, lineage, params,
                                         fill_date, information_date=None, entry=None):
    fill = pd.Timestamp(fill_date).normalize()
    info = information_date or lineage.get("candidate_trade_date")
    info = pd.Timestamp(info).normalize() if info else None
    seed = deepcopy(dict(lineage.get("execution_plan_seed") or {}))
    signal = lineage.get("signal_date") or (entry or {}).get("signal_date") or info
    if signal is not None and fill <= pd.Timestamp(signal):
        raise ValueError("Confirmed strategy fill precedes its original signal/order")
    # A verified information-day snapshot is sufficient only when there is no
    # intervening calendar day at all. Do not guess exchange calendars/holidays.
    if entry is None and info is not None and fill == info + pd.Timedelta(days=1):
        seed["management_information_date"] = info.strftime("%Y-%m-%d")
        return seed
    context = resolve_trading_lifecycle_context(project_root)
    if fill.strftime("%Y-%m-%d") > context.finalized_date:
        raise ValueError("Confirmed fill exceeds the pinned finalized boundary")
    frame = load_trading_v2_sanitized_ohlcv_frame(
        context.view, ticker=ticker, through_date=fill.strftime("%Y-%m-%d"),
        min_rows=get_required_min_rows(params),
    )
    intent = deepcopy(entry) if entry is not None else {
        "ticker": ticker, "origin": "scanner_strategy", "management_lineage": lineage,
        "information_date": str(info.date()) if info is not None else signal,
        "signal_date": signal, "execution_plan_seed": seed,
    }
    return resolve_confirmed_entry_plan_from_frame(
        entry=intent, frame=frame, params=params, fill_date=fill,
    )
