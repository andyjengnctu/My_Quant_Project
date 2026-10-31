"""Shared per-arm Strategy Compare execution contract.

Normal Strategy Compare and multi-seed robustness must replay the same arm through the
same argument assembly.  Robustness may only vary seed-bound artifact overrides,
parameter namespace and scheduling; it must not reconstruct scientific replay options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.breakout_quality_policy import (
    get_breakout_quality_workflow_settings,
)
from core.strategy_comparison import (
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
    resolve_strategy_comparison_arm_param_policy,
)
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
)
from filters.breakout_quality.strategy_compare_pit_contract import (
    resolve_strategy_compare_selection_pit_bundle_dir,
    resolve_strategy_compare_selection_pit_contract_override,
)
from services.research.strategy_compare_engine import run_comparison
from services.breakout_quality.continuous_ranker_pipeline import load_continuous_ranker_data
from services.research.strategy_compare_replay import run_standalone_baseline
from filters.breakout_quality.strategy_compare_runtime import _arm_runtime_spec
from filters.breakout_quality.strategy_compare_sources import (
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
)
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES

PROJECT_ROOT = Path(__file__).resolve().parents[2]




def resolve_strategy_compare_arm_data_dir(
    arm: Any,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path | None:
    """Resolve an explicit per-arm dataset only for scientifically isolated arms."""

    source = str(getattr(arm, "data_source", "canonical") or "canonical").strip()
    if source == "canonical":
        return None
    if source == "raw_unadjusted_controlled":
        from services.research.raw_price_strategy_dataset import ensure_raw_price_strategy_dataset

        state = ensure_raw_price_strategy_dataset(Path(project_root).resolve())
        return Path(state["dataset_dir"]).resolve()
    raise ValueError(f"未知Strategy Compare arm data_source: {source!r}")


def selection_pit_mode_paths(
    source: Any,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Resolve the exact PIT score/manifest bundle used by Strategy Compare replay."""

    base = resolve_strategy_compare_selection_pit_bundle_dir(
        root=project_root,
        source=source,
    )
    return {
        "score": base / SELECTION_POINT_IN_TIME_SCORE_FILENAME,
        "manifest": base / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    }


def resolve_strategy_compare_ranking_options(
    settings: Any,
    arm: Any,
    *,
    score_overrides: dict[str, dict[str, Any]] | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Resolve one arm's ranking options with explicit isolated overrides first.

    The precedence is deliberate: a caller-supplied benchmark seed override is the
    scientific identity of a robustness replay.  Canonical mode-specific PIT paths are
    only a fallback for the normal single-seed pipeline.  This prevents a multi-source or multi-head
    arm from silently mixing benchmark primary scores with production safety scores.
    """

    options = dict(arm.dl_runtime_options or {})
    if arm.dl_runtime_mode not in {
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
    }:
        return options
    safety_dl_id = str(options.get("safety_dl_id") or "").strip()
    if not safety_dl_id or safety_dl_id not in settings.dl_sources:
        raise ValueError(f"safety-constrained arm缺少合法safety_dl_id: {arm.arm_id}")
    source = settings.dl_sources[safety_dl_id]
    safety_override = dict((score_overrides or {}).get(safety_dl_id) or {})
    if not safety_override and str(source.score_source) == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        mode_paths = selection_pit_mode_paths(source, project_root=project_root)
        if mode_paths is not None:
            safety_override = {
                "score_path": mode_paths["score"],
                "manifest_path": mode_paths["manifest"],
            }
    options.update(
        {
            "safety_filter_id": str(source.filter_id),
            "safety_score_source": str(source.score_source),
            "safety_model_architecture": str(source.model_architecture),
            "safety_experiment_profile": str(source.experiment_profile),
            "safety_score_path_override": (
                None
                if not safety_override.get("score_path")
                else str(safety_override["score_path"])
            ),
            "safety_score_manifest_path_override": (
                None
                if not safety_override.get("manifest_path")
                else str(safety_override["manifest_path"])
            ),
        }
    )
    return options


def run_strategy_compare_baseline_arm(
    *,
    settings: Any,
    arm: Any,
    project_root: str | Path,
    params_path: str | Path,
    output_dir: str | Path,
    comparison_start: str,
    comparison_end: str,
    param_evaluation_mode: str,
    quiet: bool,
    baseline_reuse_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one DL-off arm through the canonical shared baseline replay."""

    all_off = str(arm.rule_policy) == "all_off"
    return run_standalone_baseline(
        project_root=Path(project_root).resolve(),
        dataset=settings.dataset,
        params_path=str(params_path),
        param_policy=resolve_strategy_comparison_arm_param_policy(settings, arm),
        max_positions=settings.max_positions,
        enable_rotation=settings.rotation == "on",
        optional_entry_filter_policy=(
            OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
            if all_off
            else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
        ),
        output_dir_override=Path(output_dir),
        comparison_start_date=str(comparison_start),
        comparison_end_date=str(comparison_end),
        quiet=bool(quiet),
        shared_param_overrides=(ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None),
        baseline_reuse_dir=baseline_reuse_dir,
        param_evaluation_mode=str(param_evaluation_mode),
        data_dir_override=resolve_strategy_compare_arm_data_dir(
            arm, project_root=Path(project_root).resolve()
        ),
    )


def _load_verified_real_path_diagnostic_projection(
    *, root: Path, filter_id: str, architecture: str, profile: str, seed: int,
):
    """Read a disposable diagnostic using the canonical, read-only teacher.

    The execution owner injects this optional fast path into the comparison
    engine; no optimizer, FIT or Target producer is executed on a cache hit.
    A cache miss delegates to the engine's original canonical bundle loader.
    """
    from services.breakout_quality.real_path_diagnostic_cache import (
        real_path_diagnostic_candidate_exists,
        load_verified_real_path_diagnostic,
    )

    root = Path(root).resolve()
    if not real_path_diagnostic_candidate_exists(
        root=root, filter_id=filter_id, architecture=architecture,
        profile=profile, seed=seed,
    ):
        return None
    from services.breakout_quality.trade_path_target_source import loader_kwargs_for_trade_path

    teacher = loader_kwargs_for_trade_path(
        experiment_profile=profile, project_root=root, seed=seed,
        authorized_build=False,
    )
    return load_verified_real_path_diagnostic(
        root=root, filter_id=filter_id, architecture=architecture,
        profile=profile, seed=seed, target_lineage=teacher["target_lineage"],
    )


def run_strategy_compare_active_arm(
    *,
    settings: Any,
    arm: Any,
    project_root: str | Path,
    params_path: str | Path,
    output_dir: str | Path,
    comparison_start: str,
    comparison_end: str,
    param_evaluation_mode: str,
    score_overrides: dict[str, dict[str, Any]] | None = None,
    ranking_options_extra: dict[str, Any] | None = None,
    expected_seed_override: int | None = None,
    baseline_reuse_dir: str | Path | None = None,
    quiet: bool = False,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    capture_execution_diagnostics: bool = False,
    capture_selection_target_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run one DL-on arm through the canonical shared replay argument assembly."""

    root = Path(project_root).resolve()
    dl = settings.dl_sources[str(arm.dl_id)]
    runtime_spec = _arm_runtime_spec(arm)
    all_off = str(arm.rule_policy) == "all_off"
    overrides = {
        str(key): dict(value or {})
        for key, value in dict(score_overrides or {}).items()
    }
    primary_override = dict(overrides.get(str(arm.dl_id)) or {})

    selection_paths = None
    selection_expected_seed = None
    if str(dl.score_source) == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        has_score = bool(primary_override.get("score_path"))
        has_manifest = bool(primary_override.get("manifest_path"))
        if has_score != has_manifest:
            raise ValueError("Selection PIT isolated score/manifest override必須成對提供")
        if expected_seed_override is not None and not has_score:
            raise ValueError("Selection PIT robustness seed必須綁定顯式隔離score/manifest工件")
        if has_score:
            selection_paths = {
                "score": Path(str(primary_override["score_path"])),
                "manifest": Path(str(primary_override["manifest_path"])),
            }
            selection_expected_seed = expected_seed_override
        else:
            # The canonical Rolling store splits PIT score/manifest (models/)
            # from Audit (outputs/). Passing its score directory as an override
            # would reclassify it as an isolated, colocated bundle during replay.
            # Reuse the same override decision as the preparation READY gate.
            pit_dir_override = resolve_strategy_compare_selection_pit_contract_override(
                root=root, source=dl,
            )
            if pit_dir_override is not None:
                selection_paths = selection_pit_mode_paths(dl, project_root=root)
                selection_expected_seed = (
                    int(expected_seed_override)
                    if expected_seed_override is not None
                    else int(
                        get_breakout_quality_workflow_settings(
                            experiment_profile=str(dl.experiment_profile)
                        ).seed
                    )
                )

    ranking_options = resolve_strategy_compare_ranking_options(
        settings,
        arm,
        score_overrides=overrides,
        project_root=root,
    )
    ranking_options.update(dict(ranking_options_extra or {}))

    return run_comparison(
        project_root=root,
        dataset=settings.dataset,
        params_path=str(params_path),
        param_policy=resolve_strategy_comparison_arm_param_policy(settings, arm),
        max_positions=settings.max_positions,
        enable_rotation=settings.rotation == "on",
        fixed_risk=None,
        max_position_cap_pct=None,
        comparison_mode=runtime_spec["comparison_mode"],
        ranking_policy=runtime_spec["ranking_policy"],
        ranking_options=ranking_options,
        optional_entry_filter_policy=(
            OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
            if all_off
            else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
        ),
        filter_id=dl.filter_id,
        score_source=dl.score_source,
        model_architecture=dl.model_architecture,
        experiment_profile=dl.experiment_profile,
        threshold=dl.threshold,
        output_dir_override=Path(output_dir),
        comparison_start_date=str(comparison_start),
        comparison_end_date=str(comparison_end),
        quiet=bool(quiet),
        progress_callback=progress_callback,
        shared_param_overrides=(ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None),
        baseline_reuse_dir=baseline_reuse_dir,
        param_evaluation_mode=str(param_evaluation_mode),
        continuous_score_path_override=(
            str(primary_override["score_path"])
            if (
                str(dl.score_source) == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
                and primary_override.get("score_path")
            )
            else None
        ),
        continuous_score_execution_start_override=(
            str(primary_override["execution_start"])
            if (
                str(dl.score_source) == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
                and primary_override.get("execution_start")
            )
            else None
        ),
        selection_pit_score_path_override=(
            None if selection_paths is None else str(selection_paths["score"])
        ),
        selection_pit_manifest_path_override=(
            None if selection_paths is None else str(selection_paths["manifest"])
        ),
        selection_pit_expected_seed_override=selection_expected_seed,
        capture_execution_diagnostics=bool(capture_execution_diagnostics),
        capture_selection_target_diagnostics=bool(capture_selection_target_diagnostics),
        verified_real_path_bundle_loader=load_continuous_ranker_data,
        verified_real_path_diagnostic_loader=_load_verified_real_path_diagnostic_projection,
    )


__all__ = [
    "resolve_strategy_compare_ranking_options",
    "run_strategy_compare_active_arm",
    "run_strategy_compare_baseline_arm",
    "selection_pit_mode_paths",
]
