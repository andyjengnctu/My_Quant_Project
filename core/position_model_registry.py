"""Single registry for independent position-model producers and contracts."""
from importlib import import_module


POSITION_MODELS = {
    "HV1": {"version": 1, "producer": "services.breakout_quality.holding_value_model"},
    "HV2": {"version": 1, "producer": "services.breakout_quality.holding_policy_model"},
}


def position_model_descriptor(model_id):
    try:
        return dict(POSITION_MODELS[str(model_id)])
    except KeyError as exc:
        raise ValueError(f"Unknown position model: {model_id}") from exc


def position_model_producer(model_id):
    return import_module(position_model_descriptor(model_id)["producer"])
