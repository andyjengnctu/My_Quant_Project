"""Generic service facade; consumers never dispatch on experiment identities."""
from core.position_exit_policy import position_exit_spec
from core.position_model_registry import position_model_producer


def configuration(settings, arm):
    spec = position_exit_spec(arm.dl_runtime_options)
    if spec is None:
        raise ValueError("Arm has no position model")
    producer = position_model_producer(spec["model_id"])
    return {**producer.configuration(settings, arm), "model_id": spec["model_id"]}


def _dispatch(config, name, root, **kwargs):
    producer = position_model_producer(config["model_id"])
    fitting_config = {key: value for key, value in config.items() if key != "model_id"}
    return getattr(producer, name)(root, config=fitting_config, **kwargs)


def inspect_holding_model(root, *, seed, config):
    return _dispatch(config, "inspect_holding_model", root, seed=seed)


def ensure_holding_model(root, *, seed, config):
    return _dispatch(config, "ensure_holding_model", root, seed=seed)


def holding_model_runtime(root, *, seed, config, evidence_path=None):
    return _dispatch(config, "holding_model_runtime", root, seed=seed, evidence_path=evidence_path)
