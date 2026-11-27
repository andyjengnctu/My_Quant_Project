"""Research adapter for independently fitted position-decision dependencies."""
from pathlib import Path
from core.research_orchestration import ResearchArtifactAction, resolve_research_artifact_action
from filters.breakout_quality.holding_value import MODEL_ID
from services.breakout_quality.holding_value_model import (
    configuration, position_exit_spec, inspect_holding_model, ensure_holding_model,
)


def position_model_bindings(settings):
    result = {}
    for arm in settings.enabled_arms:
        if position_exit_spec(arm.dl_runtime_options) is not None:
            config = configuration(settings, arm)
            if MODEL_ID in result and result[MODEL_ID] != config:
                raise ValueError("One position model cannot have conflicting active time contracts")
            result[MODEL_ID] = config
    return result


def append_position_model_actions(*, root, settings, actions, identities, seeds):
    rows = {}
    for model_id, config in position_model_bindings(settings).items():
        for seed in seeds:
            status = inspect_holding_model(root, seed=int(seed), config=config)
            policy = settings.preparation
            action = resolve_research_artifact_action(
                ready=status["ready"], artifact_exists=Path(status["path"]).exists(),
                has_builder=True, auto_prepare=policy.auto_prepare,
                reuse_ready_artifacts=policy.reuse_ready_artifacts,
                rebuild_stale_artifacts=policy.rebuild_stale_artifacts,
                resume_partial_artifacts=True, resumable=not status["ready"],
            )
            key = f"position_model:{model_id}:{seed}"
            actions.append(ResearchArtifactAction(
                action_id=key, artifact_key=key, action=action,
                builder_type="position_value_training" if action not in {"REUSE", "BLOCKED"} else None,
                description=f"{model_id} seed={seed}: {status['reason']}", path=status["path"],
                producer_work_type="position_model_training", execution_priority=80,
                dependencies=tuple(a.artifact_key for a in actions if a.artifact_key.startswith("param:")),
            ))
            identities[key] = {"path": status["path"], "sha256": status.get("sha256"),
                               "status": "READY" if status["ready"] else "MISSING"}
            rows[key] = {**status, "action": action, "seed": seed, "model_id": model_id}
    return rows


def execute_position_model_action(*, root, settings, action):
    _, model_id, seed = action.artifact_key.split(":")
    return ensure_holding_model(root, seed=int(seed), config=position_model_bindings(settings)[model_id])


def ensure_position_models(*, root, settings, seeds):
    for config in position_model_bindings(settings).values():
        for seed in seeds:
            ensure_holding_model(root, seed=int(seed), config=config)


def position_model_identity(*, root, settings, arm, seed):
    if position_exit_spec(arm.dl_runtime_options) is None:
        return None
    result = inspect_holding_model(root, seed=seed, config=configuration(settings, arm))
    if not result["ready"]:
        return {"status": "NOT_READY", "reason": result["reason"]}
    return {"model_id": MODEL_ID, "seed": int(seed), "manifest_sha256": result["sha256"]}
