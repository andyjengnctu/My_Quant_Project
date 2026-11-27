"""Optional, completed-bar position decisions; execution remains canonical.

The provider receives no current-session prices. A requested early exit persists
through blocked sessions and uses the existing indicator/open sale transition.
Protection obligations retain their original priority and accounting.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math

_PROVIDER = ContextVar("position_exit_provider", default=None)


def position_exit_spec(options):
    raw = dict(options or {}).get("position_exit_model")
    if raw is None:
        return None
    if raw != {"model_id": "HV1", "contract_version": 1}:
        raise ValueError("Unknown position exit model contract")
    return dict(raw)


@contextmanager
def bind_position_exit_provider(provider):
    token = _PROVIDER.set(provider)
    try:
        yield
    finally:
        _PROVIDER.reset(token)


def request_position_early_exit(*, ticker, position, information_date,
                                execution_date, completed_atr, completed_high,
                                params, natural_exit=False):
    provider = _PROVIDER.get()
    if provider is None or not bool(getattr(params, "use_breakout_quality_ranking", False)):
        return False
    if information_date >= execution_date:
        raise ValueError("Holding decision must precede its execution session")
    if int(position.get("qty", 0)) <= 0:
        return False
    if position.get("_model_exit_pending"):
        return True
    # An existing protective/open exit needs no extra discretionary prediction.
    if natural_exit or position.get("pending_exit_action") == "STOP":
        return False
    prediction = provider(
        ticker=str(ticker), position=dict(position), information_date=information_date,
        execution_date=execution_date, completed_atr=completed_atr,
        completed_high=completed_high, params=params,
    )
    if prediction is None:  # explicitly unscorable input: original rules continue
        return False
    prediction = float(prediction)
    if not math.isfinite(prediction):
        raise ValueError("Nonfinite continuation prediction")
    if prediction < 0.0:
        position["_model_exit_pending"] = True
        position["_model_exit_information_date"] = str(information_date)
        position["_model_exit_delta_r"] = prediction
        return True
    return False
