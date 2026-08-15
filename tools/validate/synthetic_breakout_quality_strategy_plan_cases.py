from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pandas as pd

from config.strategy_compare import get_strategy_comparison_settings
from core.strategy_comparison import StrategyPreparationAction, StrategyPreparationPlan
from filters.breakout_quality import strategy_comparison as strategy_comparison_module
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

    base = get_strategy_comparison_settings("forward_oos")
    current_dl_arms = tuple(arm for arm in base.enabled_arms if arm.dl_enabled)
    current_arm = current_dl_arms[0] if len(current_dl_arms) == 1 else None
    retired_same_source = (
        next(
            (
                arm
                for arm in base.arms.values()
                if current_arm is not None
                and not arm.enabled
                and arm.dl_enabled
                and arm.dl_id == current_arm.dl_id
            ),
            None,
        )
        if current_arm is not None
        else None
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "fixture_has_one_current_dl_arm_and_one_retired_same_source_arm",
        True,
        current_arm is not None and retired_same_source is not None,
    )
    if current_arm is None or retired_same_source is None:
        return results, summary

    dl_id = str(current_arm.dl_id or "")
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
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "retiring_unrelated_same_source_arm_preserves_artifact_and_period_binding",
            True,
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
        resolved = ResolvedComparisonPlan.from_status(
            settings=base,
            project_root=root,
            status=ready_status,
        )
        detached = resolved.status_dict()
        detached["comparison_period"]["start"] = "2099-01-01"
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "resolved_plan_is_immutable_and_ready_contract_is_complete",
            True,
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
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "ready_plan_rejects_missing_comparison_period_before_execution",
            True,
            missing_period_rejected,
        )

    app_source = (Path(__file__).resolve().parents[2] / "apps" / "research.py").read_text(
        encoding="utf-8"
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "interactive_status_and_run_share_same_resolved_plan",
        True,
        "resolved_plan = resolve_comparison_plan(settings=settings)" in app_source
        and "resolved_plan=resolved_plan" in app_source,
    )

    summary["profile_id"] = base.profile_id
    summary["current_dl_arm"] = current_arm.arm_id
    summary["retired_same_source_arm"] = retired_same_source.arm_id
    return results, summary


__all__ = ["validate_strategy_compare_resolved_plan_transition_contract_case"]
