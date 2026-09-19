"""Materialize historical pre-fill geometry from its frozen policy and PIT bars.

AI: A serialization/checkpoint date is not a signal date.  An immutable seed
already known at the origin remains authoritative.  When only a later snapshot
was saved, this producer rebuilds the earlier normal plan from the SAME frozen
Params and origin-bar inputs, never from that snapshot's Stop/Target/ATR.  A
confirmed manual limit is an order fact; a strategy limit requires the original
signal from the canonical signal producer.  No selection, fill or ledger write
is performed here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.capital_policy import resolve_scanner_live_capital
from core.entry_plans import build_normal_candidate_plan
from core.file_integrity import canonical_json_sha256
from core.params_io import params_to_json_dict
from core.signal_utils import generate_signals, unpack_precomputed_signals


def materialize_prefill_origin(*, frame: pd.DataFrame, plan: Mapping[str, Any], params) -> dict[str, Any]:
    """Bind a late saved plan to a causal historical prefix, when verifiable.

    Missing bars/policy cannot be substituted with present-day inputs.  Preserve
    any dated checkpoint in that case and expose the missing prerequisite.  A
    corrupted policy identity is an error, not a reason to try current Params.
    """
    result = deepcopy(dict(plan))
    request = result.pop("origin_reconstruction", None)
    if not isinstance(request, Mapping) or params is None or frame.empty:
        return result
    original = str(request["origin_date"])
    recorded = str(result.get("plan_as_of_date") or result["signal_date"])
    if original >= recorded:
        return result
    # Never legitimize an adapter which merely re-dated the later geometry.
    if str(result.get("signal_date")) < recorded:
        raise ValueError("PIT: lifecycle origin precedes frozen plan availability")
    expected = str(request.get("frozen_params_sha256") or "")
    if not expected or canonical_json_sha256(params_to_json_dict(params)) != expected:
        raise ValueError("Historical pre-fill reconstruction requires its exact frozen Params")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Historical pre-fill reconstruction requires ordered unique market bars")
    cutoff = pd.Timestamp(original)
    prefix = frame.loc[frame.index <= cutoff]
    if prefix.empty or pd.Timestamp(prefix.index[-1]).normalize() != cutoff:
        result["origin_resolution_error"] = "Original completed information bar is unavailable: " + original
        return result
    # A fresh prefix prevents suffix, stale dataframe features or display-array
    # inputs from deciding the origin.  Only canonical OHLCV is supplied.
    prefix = prefix.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    prefix.attrs.clear()
    atr, _buy, _sell, limits = unpack_precomputed_signals(generate_signals(prefix, params, ticker=plan.get("ticker")))
    entry_atr = float(atr[-1])
    if not np.isfinite(entry_atr) or entry_atr <= 0:
        result["origin_resolution_error"] = "Original frozen-policy ATR lacks sufficient history: " + original
        return result
    if request.get("kind") == "manual_order":
        limit = request.get("accepted_limit_price")
    else:
        # The saved signal identity alone is not permission to synthesize a
        # different signal using a later price or the user's current Params.
        limit = limits[-1]
    if limit is None or not np.isfinite(float(limit)) or float(limit) <= 0:
        result["origin_resolution_error"] = "Original frozen-policy signal/limit cannot be verified: " + original
        return result
    seed = build_normal_candidate_plan(
        float(limit), entry_atr, float(resolve_scanner_live_capital(params)), params,
        ticker=plan.get("ticker"), security_profile=plan.get("security_profile"), trade_date=original,
    )
    if seed is None:
        raise ValueError("Cannot materialize the original canonical entry plan")
    checkpoint = None
    if isinstance(result.get("shadow_position_state"), Mapping):
        # A mature saved checkpoint is evidence at ITS date, not before it.
        # Reconstructed history is a prefix; forward replay verifies management
        # compatibility before resuming the exact checkpoint. A conflicting
        # derived checkpoint cannot splice a different origin into this prefix.
        checkpoint = deepcopy(result)
        checkpoint.pop("order_intent", None)
    result.update(
        signal_date=original, plan_as_of_date=original,
        limit_price=seed["limit_price"], stop_price=seed["init_sl"], init_trail=seed["init_trail"],
        tp_price=seed["target_price"], entry_atr=seed["entry_atr"],
        shadow_position_state=None, shadow_as_of_date=None,
        origin_resolution="frozen_policy_historical_reconstruction",
        recorded_plan_as_of_date=recorded,
        origin_params_sha256=expected,
    )
    if checkpoint is not None:
        result["management_checkpoint"] = checkpoint
    return result
