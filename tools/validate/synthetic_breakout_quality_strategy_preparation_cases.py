"""Generic Strategy Compare preparation contract checks.

This module owns preparation-plan/builder boundary checks that were previously
embedded in the config-driven application validator.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from core.strategy_comparison import StrategyPreparationAction, StrategyPreparationPlan
from .synthetic_breakout_quality_support import add_check


def append_strategy_compare_preparation_contract_checks(
    *,
    results: list[dict[str, Any]],
    case_id: str,
    project_root: Path,
    settings: Any,
    preparation_source: str,
    orchestration_source: str,
) -> None:
    min_roos_source = settings.parameter_sources["min_roos"]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "min_roos_uses_forward_p2_artifact_and_has_auto_builder",
        True,
        "binary_dl_filter_param_adaptation/risk_only_rolling/p2_dl_off_trained"
        in str(min_roos_source.path_template)
        and "trade_path_label/a2_teacher_params" not in str(min_roos_source.path_template)
        and min_roos_source.identity_manifest_path is not None
        and min_roos_source.builder is not None
        and str(min_roos_source.builder.options.get("parameter_set")) == "p2",
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "parameter_coverage_is_preflighted_before_any_pair_replay",
        True,
        "PARAM_PERIOD_MISMATCH" in preparation_source
        and "comparison_period" in preparation_source
        and "comparison_start_date=comparison_start" in orchestration_source
        and "comparison_end_date=comparison_end" in orchestration_source,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_supports_dependency_waves_after_score_period_becomes_known",
        True,
        "max_waves" in preparation_source
        and "重新規劃後沒有可執行且依賴已就緒的動作" in preparation_source,
    )

    from filters.breakout_quality import strategy_compare_preparation as preparation_module
    score_action = StrategyPreparationAction(
        action_id="dl:TP1:forward_scores",
        artifact_key="dl:TP1:forward_scores",
        action="BUILD",
        builder_type="forward_oos_scores",
        description="build scores",
        path="models/scores.csv",
    )
    p2_action = StrategyPreparationAction(
        action_id="param:min_roos",
        artifact_key="param:min_roos",
        action="BUILD",
        builder_type="binary_dl_min_roos_rolling",
        description="build p2",
        path="models/p2.json",
    )
    initial_wave_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="PREPARABLE", actions=(score_action,)
        ),
    }
    second_wave_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="PREPARABLE", actions=(p2_action,)
        ),
    }
    final_wave_status = {
        "comparison_ready": True,
        "overall_status": "READY",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="READY", actions=tuple()
        ),
    }
    with patch.object(
        preparation_module,
        "collect_artifact_status",
        side_effect=(second_wave_status, final_wave_status),
    ), patch.object(
        preparation_module,
        "_execute_preparation_action",
    ) as mocked_prepare_action, redirect_stdout(io.StringIO()):
        multi_wave_result = preparation_module.prepare_strategy_comparison_artifacts(
            project_root=project_root,
            settings=settings,
            status=initial_wave_status,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_executes_newly_revealed_dependencies_in_later_wave",
        True,
        multi_wave_result["comparison_ready"]
        and mocked_prepare_action.call_count == 2
        and [
            call.kwargs["action"].artifact_key
            for call in mocked_prepare_action.call_args_list
        ] == ["dl:TP1:forward_scores", "param:min_roos"],
    )

    dependency_reuse = StrategyPreparationAction(
        action_id="dataset:truth", artifact_key="dataset:truth",
        action="REUSE", builder_type=None, description="reuse truth", path="outputs/dataset.json",
        producer_work_type="existing_artifact",
    )
    dependency_build = StrategyPreparationAction(
        action_id="score:forward", artifact_key="score:forward",
        action="BUILD", builder_type="score_builder", description="build score", path="outputs/score.csv",
        dependencies=("dataset:truth",), producer_work_type="strategy_compare_deterministic_rebuild",
    )
    dependency_plan = StrategyPreparationPlan.from_actions(
        (dependency_build, dependency_reuse)
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_declares_dependency_and_producer_contract",
        True,
        dependency_plan.overall_status == "PREPARABLE"
        and dependency_plan.next_runnable_action() is dependency_build
        and dependency_build.as_dict()["dependencies"] == ["dataset:truth"]
        and dependency_build.as_dict()["producer_work_type"]
        == "strategy_compare_deterministic_rebuild",
    )

    dependency_cycle_rejected = False
    try:
        StrategyPreparationPlan.from_actions((
            StrategyPreparationAction(
                action_id="a", artifact_key="a", action="BUILD", builder_type="x",
                description="a", path="a", dependencies=("b",),
            ),
            StrategyPreparationAction(
                action_id="b", artifact_key="b", action="BUILD", builder_type="x",
                description="b", path="b", dependencies=("a",),
            ),
        ))
    except ValueError:
        dependency_cycle_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_rejects_dependency_cycles",
        True,
        dependency_cycle_rejected,
    )

    unknown_dependency_rejected = False
    try:
        StrategyPreparationPlan.from_actions((
            StrategyPreparationAction(
                action_id="score", artifact_key="score", action="BUILD", builder_type="x",
                description="score", path="score", dependencies=("missing:truth",),
            ),
        ))
    except ValueError:
        unknown_dependency_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_rejects_unknown_dependencies",
        True,
        unknown_dependency_rejected,
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "parameter_preflight_identity_tracks_current_min_roos_contract_not_removed_full_baseline",
        True,
        all(token in preparation_source for token in (
            "TRAINING_CONFIG_MISMATCH", "BINARY_PIT_IDENTITY_MISSING",
            "MIN_ROOS_SEARCH_FIELDS_MISMATCH", "trials_per_fold",
            "max_position_cap_pct",
        ))
        and "BASELINE_PARAMS_IDENTITY_MISMATCH" not in preparation_source,
    )




__all__ = ["append_strategy_compare_preparation_contract_checks"]
