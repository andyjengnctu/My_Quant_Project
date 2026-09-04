from __future__ import annotations

from .checks import bind_checks

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from core.strategy_compare_policy import get_strategy_comparison_settings
from core.strategy_comparison import StrategyPreparationAction, StrategyPreparationPlan
from services.research import strategy_comparison as strategy_comparison_module
from filters.breakout_quality.strategy_compare_plan import ResolvedComparisonPlan
from .synthetic_breakout_quality_support import add_check, replace


def _blocked_continuous_source_status(*, dl_id: str) -> dict:
    actions = [
        StrategyPreparationAction(
            action_id=f"dl:{dl_id}:{artifact_name}",
            artifact_key=f"dl:{dl_id}:{artifact_name}",
            action="BLOCKED",
            builder_type=None,
            description="synthetic unresolved continuous source",
            path=(
                "current/canonical/missing_scores.csv"
                if artifact_name == "forward_scores"
                else f"current/canonical/{artifact_name}"
            ),
            dependencies=(),
            producer_work_type="model_training",
        )
        for artifact_name in ("model", "manifest", "report", "forward_scores")
    ]
    return {
        "comparison_period": None,
        "comparison_period_source": "runtime_pending",
        "dl_sources": {
            dl_id: {
                "ready": False,
                "status": "MISSING",
                "files": {
                    "model": {"path": "current/canonical/model"},
                    "manifest": {"path": "current/canonical/manifest"},
                    "report": {"path": "current/canonical/report"},
                    "forward_scores": {
                        "path": "current/canonical/missing_scores.csv"
                    },
                },
            }
        },
        "artifact_identities": {},
        "preparation_plan": StrategyPreparationPlan.from_actions(actions),
        "overall_status": "BLOCKED",
        "comparison_ready": False,
    }


def validate_strategy_compare_resolved_plan_transition_contract_case(_base_params):
    """Profile membership changes must not alter proven artifact/period bindings."""

    case_id = "STRATEGY_COMPARE_RESOLVED_PLAN_TRANSITION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    base = get_strategy_comparison_settings("forward_oos")
    current_dl_arms = tuple(arm for arm in base.enabled_arms if arm.dl_enabled)
    current_arm = None
    retired_same_source = None
    for candidate in current_dl_arms:
        retired = next(
            (
                arm
                for arm in base.arms.values()
                if not arm.enabled
                and arm.dl_enabled
                and arm.dl_id == candidate.dl_id
            ),
            None,
        )
        if retired is not None:
            current_arm = candidate
            retired_same_source = retired
            break
    check_true(
        "fixture_has_current_dl_arm_with_retired_same_source_arm",
        current_arm is not None and retired_same_source is not None,
    )
    if current_arm is None or retired_same_source is None:
        return results, summary

    dl_id = str(current_arm.dl_id or "")
    isolated_arms = {
        arm_id: (
            replace(arm, enabled=(arm_id == current_arm.arm_id))
            if arm.dl_enabled
            else arm
        )
        for arm_id, arm in base.arms.items()
    }
    base = replace(base, arms=isolated_arms)
    current_arm = base.arms[current_arm.arm_id]
    retired_same_source = base.arms[retired_same_source.arm_id]

    expanded_arms = dict(base.arms)
    expanded_arms[retired_same_source.arm_id] = replace(
        retired_same_source,
        enabled=True,
    )
    expanded = replace(base, arms=expanded_arms)

    with tempfile.TemporaryDirectory() as raw_temp:
        root = Path(raw_temp).resolve()
        score_path = root / "archive" / "continuous_scores.csv"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"ticker": "2330", "date": "2021-01-04", "group_index": 0, "model_score": 0.8},
                {"ticker": "2317", "date": "2021-01-05", "group_index": 0, "model_score": 0.7},
            ]
        ).to_csv(score_path, index=False)
        score_sha = strategy_comparison_module._file_sha256(score_path)

        runs_root = (
            root
            / str(base.output_root)
            / "runs"
        )
        run_dir = runs_root / "archived_provenance"
        run_dir.mkdir(parents=True, exist_ok=True)
        group_id = strategy_comparison_module._pair_group_id(
            param_source=retired_same_source.param_source,
            rule_policy=retired_same_source.rule_policy,
            dl_id=dl_id,
            dl_runtime_mode=str(retired_same_source.dl_runtime_mode or ""),
        )
        archived_payload = {
            "status": "COMPLETED",
            "settings": expanded.as_dict(),
            "artifact_identities": {
                f"dl:{dl_id}:forward_scores": {
                    "path": "archive/continuous_scores.csv",
                    "sha256": score_sha,
                }
            },
            "comparison_period": {"start": "2021-01-01", "end": "2021-01-05"},
            "pairs": {
                group_id: {
                    "metadata": {
                        "comparison_period": {
                            "start": "2021-01-01",
                            "end": "2021-01-05",
                        },
                        "score_signal_coverage": {
                            "required_start": "2021-01-01",
                            "first_scored_event": "2021-01-04",
                            "available_through": "2021-01-05",
                        },
                    }
                }
            },
        }
        (run_dir / "strategy_comparison.json").write_text(
            json.dumps(archived_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        replay_cache = {"pairs": {}, "baseline_groups": {}}
        expanded_status = strategy_comparison_module._apply_completed_pair_frozen_score_reuse(
            root=root,
            settings=expanded,
            status=_blocked_continuous_source_status(dl_id=dl_id),
            replay_cache=replay_cache,
        )
        reduced_status = strategy_comparison_module._apply_completed_pair_frozen_score_reuse(
            root=root,
            settings=base,
            status=_blocked_continuous_source_status(dl_id=dl_id),
            replay_cache=replay_cache,
        )

        expanded_binding = dict(
            (expanded_status.get("continuous_score_overrides") or {}).get(dl_id) or {}
        )
        reduced_binding = dict(
            (reduced_status.get("continuous_score_overrides") or {}).get(dl_id) or {}
        )
        check_true(
            "retiring_unrelated_same_source_arm_preserves_artifact_and_period_binding",
            bool(expanded_binding)
                        and bool(reduced_binding)
                        and expanded_binding.get("sha256") == reduced_binding.get("sha256") == score_sha
                        and expanded_binding.get("score_path") == reduced_binding.get("score_path")
                        and expanded_status.get("comparison_period")
                        == reduced_status.get("comparison_period")
                        == {"start": "2021-01-01", "end": "2021-01-05"},
        )

        ready_status = deepcopy(reduced_status)
        ready_status["config_fingerprint"] = "synthetic-plan"
        ready_status["replay_cache"] = replay_cache
        ready_status["resolved_parameter_paths"] = {
            source_id: root / f"params/{source_id}.json"
            for source_id in {arm.param_source for arm in base.enabled_arms}
        }
        ready_status["resolved_arm_parameter_paths"] = {
            arm.arm_id: root / (
                f"params/{arm.param_source}__"
                f"{str(arm.param_policy or base.param_policy)}.json"
            )
            for arm in base.enabled_arms
        }
        ready_status["resolved_arm_parameter_identities"] = {
            arm.arm_id: {"sha256": f"eval-{arm.arm_id}"}
            for arm in base.enabled_arms
        }
        ready_status["replay_cache"] = {
            **dict(ready_status.get("replay_cache") or {}),
            "arm_states": {
                arm.arm_id: {
                    "action": "RUN",
                    "scientific_result_ready": False,
                    "context_required": False,
                    "context_ready": False,
                    "source_dir": None,
                    "reason": "synthetic cache miss",
                    "evidence": None,
                }
                for arm in base.enabled_arms
            },
        }
        resolved = ResolvedComparisonPlan.from_status(
            settings=base,
            project_root=root,
            status=ready_status,
        )
        detached = resolved.status_dict()
        detached["comparison_period"]["start"] = "2099-01-01"
        check_true(
            "resolved_plan_is_immutable_and_ready_contract_is_complete",
            resolved.overall_status == "READY"
                        and resolved.comparison_period.get("start") == "2021-01-01"
                        and resolved.comparison_period.get("end") == "2021-01-05",
        )

        broken = deepcopy(ready_status)
        broken["comparison_period"] = None
        try:
            ResolvedComparisonPlan.from_status(
                settings=base,
                project_root=root,
                status=broken,
            )
        except RuntimeError as exc:
            missing_period_rejected = "comparison period" in str(exc)
        else:
            missing_period_rejected = False
        check_true("ready_plan_rejects_missing_comparison_period_before_execution", missing_period_rejected)

    from apps import research as research_app
    from services.research import strategy_compare_application as strategy_compare_app_service

    with (
        patch.object(
            strategy_compare_app_service,
            "get_strategy_comparison_settings",
            return_value=base,
        ),
        patch.object(
            strategy_comparison_module,
            "resolve_comparison_plan",
            return_value=resolved,
        ) as resolve_mock,
        patch.object(
            strategy_comparison_module,
            "render_execution_plan",
            return_value="synthetic resolved plan",
        ) as render_mock,
    ):
        service_settings, service_plan, rendered_plan = (
            strategy_compare_app_service.resolve_strategy_comparison_execution(base.profile_id)
        )

    with (
        patch.object(
            research_app,
            "resolve_strategy_comparison_execution",
            return_value=(base, resolved, "synthetic resolved plan"),
        ),
        patch.object(
            research_app,
            "execute_strategy_comparison",
            return_value={"status": "COMPLETED"},
        ) as execute_mock,
        patch("builtins.print"),
    ):
        app_result = research_app._run_current_comparison(
            profile_id=base.profile_id,
            confirm=False,
        )

    resolve_call = resolve_mock.call_args
    render_call = render_mock.call_args
    execute_call = execute_mock.call_args
    check_true(
        "interactive_status_and_run_share_same_resolved_plan",
        service_settings is base
                and service_plan is resolved
                and rendered_plan == "synthetic resolved plan"
                and resolve_call is not None
                and resolve_call.kwargs.get("settings") is base
                and render_call is not None
                and render_call.kwargs.get("settings") is base
                and render_call.kwargs.get("status") == resolved.status_dict()
                and execute_call is not None
                and execute_call.kwargs.get("settings") is base
                and execute_call.kwargs.get("resolved_plan") is resolved
                and app_result == {"status": "COMPLETED"},
    )

    summary["profile_id"] = base.profile_id
    summary["current_dl_arm"] = current_arm.arm_id
    summary["retired_same_source_arm"] = retired_same_source.arm_id
    return results, summary


__all__ = ["validate_strategy_compare_resolved_plan_transition_contract_case"]
