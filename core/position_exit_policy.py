"""Optional, completed-bar position decisions; execution remains canonical.

The provider receives no current-session prices. A requested early exit persists
through blocked sessions and uses the existing indicator/open sale transition.
Protection obligations retain their original priority and accounting.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math

_PROVIDER = ContextVar("position_exit_provider", default=None)


@dataclass(frozen=True)
class PositionExitDecision:
    requested: bool
    value: float
    value_name: str


def position_exit_spec(options):
    raw = dict(options or {}).get("position_exit_model")
    if raw is None:
        return None
    from core.position_model_registry import position_model_descriptor
    descriptor = position_model_descriptor(raw.get("model_id"))
    if raw != {"model_id": raw["model_id"], "contract_version": descriptor["version"]}:
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
    if provider is None or not (bool(position.get("use_breakout_quality_ranking"))
                               or bool(getattr(params, "use_breakout_quality_ranking", False))):
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
    decision = (prediction if isinstance(prediction, PositionExitDecision) else
                PositionExitDecision(float(prediction) < 0.0, float(prediction), "delta_r"))
    if not math.isfinite(decision.value):
        raise ValueError("Nonfinite position prediction")
    if decision.value_name not in ("delta_r", "exit_probability") or type(decision.requested) is not bool:
        raise ValueError("Invalid position decision output")
    if decision.value_name == "exit_probability" and not 0 <= decision.value <= 1:
        raise ValueError("Invalid position exit probability")
    if decision.requested:
        position["_model_exit_pending"] = True
        position["_model_exit_information_date"] = str(information_date)
        position["_model_exit_" + decision.value_name] = decision.value
        return True
    return False
