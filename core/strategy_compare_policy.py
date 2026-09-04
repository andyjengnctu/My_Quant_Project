"""Strategy Compare resolver, validation, and derived runtime policy.

All functions in this module consume declarative config plus the canonical comparison
registry; they do not own user-adjustable values or duplicate scientific catalogs.
"""

from __future__ import annotations

from dataclasses import replace

from config.breakout_quality import (
    get_breakout_quality_model_test_settings,
    is_breakout_quality_model_test_profile,
    get_breakout_quality_workflow_settings,
)
from config.compatibility.strategy_compare_history import (
    HISTORICAL_STRATEGY_COMPARE_ARMS,
    HISTORICAL_STRATEGY_COMPARE_CONTRASTS,
    HISTORICAL_STRATEGY_DL_SOURCES,
    HISTORICAL_STRATEGY_PARAM_SOURCES,
)
from config.strategy_compare import (
    STRATEGY_COMPARE_DEFAULT_PROFILE,
    STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE,
    STRATEGY_COMPARE_GPU_TRAIN_WORKERS,
    STRATEGY_COMPARE_MENU_PROFILE_IDS,
    STRATEGY_COMPARE_REUSE_COMPLETED_RESULTS,
    STRATEGY_COMPARE_REUSE_SHARED_BASELINE,
    STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS,
    STRATEGY_RUNTIME_INTEGRATION,
)
from config.training_policy import (
    ROBUSTNESS_BENCHMARK_ID,
    ROBUSTNESS_BENCHMARK_SEED_COUNT,
    ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED,
)
from core.research_policy import get_research_artifact_preparation_policy
from core.strategy_compare_registry import (
    STRATEGY_COMPARE_ARMS,
    STRATEGY_COMPARE_CONTRASTS,
    STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT,
    STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES,
    STRATEGY_COMPARE_PROFILES,
    STRATEGY_COMPARE_ROLLING_TEST_MODES,
    STRATEGY_COMPARE_SCHEMA_VERSION,
    STRATEGY_COMPARE_SUITES,
    STRATEGY_DL_SOURCES,
    STRATEGY_PARAM_SOURCES,
)
from core.training_policy import (
    get_strategy_parameter_training_policy_snapshot,
    resolve_robustness_benchmark_seeds,
)
from core.strategy_comparison import (
    StrategyArtifactBuilder,
    StrategyComparisonArm,
    StrategyComparisonContrast,
    StrategyComparisonSettings,
    StrategyMultiSeedRobustnessSettings,
    StrategyRuntimeIntegrationSettings,
    StrategyDLSource,
    StrategyParameterSource,
    StrategyPreparationPolicy,
    resolve_strategy_comparison_arm_param_policy,
    validate_strategy_comparison_settings,
    validate_strategy_compare_gpu_train_workers,
    validate_strategy_multi_seed_robustness_settings,
    validate_strategy_runtime_integration_settings,
)


def _get_strategy_compare_preparation() -> dict[str, bool]:
    research = get_research_artifact_preparation_policy()
    return {
        "auto_prepare": bool(research.auto_prepare),
        "reuse_ready_artifacts": bool(research.reuse_ready_artifacts),
        "rebuild_stale_artifacts": bool(research.rebuild_stale_artifacts),
        "resume_parameter_training": bool(research.resume_partial_artifacts),
        "require_confirmation": bool(research.require_single_confirmation),
        "reuse_completed_results": bool(STRATEGY_COMPARE_REUSE_COMPLETED_RESULTS),
        "reuse_shared_baseline": bool(STRATEGY_COMPARE_REUSE_SHARED_BASELINE),
    }

def _merge_compatibility_catalog(
    active: dict[str, dict],
    historical: dict[str, dict],
    *,
    catalog_name: str,
) -> dict[str, dict]:
    overlap = sorted(set(active).intersection(historical))
    if overlap:
        raise ValueError(
            f"Strategy Compare active/historical {catalog_name} ID重複: {overlap}"
        )
    return {**active, **historical}


def _compatibility_catalogs() -> tuple[
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
]:
    """Return runtime catalogs including read-only historical compatibility entries."""

    return (
        _merge_compatibility_catalog(
            STRATEGY_PARAM_SOURCES,
            HISTORICAL_STRATEGY_PARAM_SOURCES,
            catalog_name="parameter source",
        ),
        _merge_compatibility_catalog(
            STRATEGY_DL_SOURCES,
            HISTORICAL_STRATEGY_DL_SOURCES,
            catalog_name="DL source",
        ),
        _merge_compatibility_catalog(
            STRATEGY_COMPARE_ARMS,
            HISTORICAL_STRATEGY_COMPARE_ARMS,
            catalog_name="arm",
        ),
        _merge_compatibility_catalog(
            STRATEGY_COMPARE_CONTRASTS,
            HISTORICAL_STRATEGY_COMPARE_CONTRASTS,
            catalog_name="contrast",
        ),
    )


def _builder(raw) -> StrategyArtifactBuilder | None:
    if raw in (None, {}):
        return None
    return StrategyArtifactBuilder(
        enabled=bool(raw.get("enabled")),
        builder_type=str(raw.get("builder_type") or "").strip(),
        options=dict(raw.get("options") or {}),
    )


def _resolved_suite_id(profile: dict) -> str | None:
    value = str(profile.get("suite_id") or "").strip()
    return value or None


def _resolved_profile_matrix(profile: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    suite_id = _resolved_suite_id(profile)
    if suite_id is None:
        return (
            tuple(str(value) for value in profile.get("arm_ids", ())),
            tuple(str(value) for value in profile.get("contrast_ids", ())),
        )
    if suite_id not in STRATEGY_COMPARE_SUITES:
        raise ValueError(f"Strategy Compare profile引用不存在Compare Suite: {suite_id}")
    suite = get_strategy_compare_suite(suite_id)
    if profile.get("arm_ids") not in (None, (), []):
        raise ValueError(f"current suite profile不得另寫arm_ids: suite={suite_id}")
    if profile.get("contrast_ids") not in (None, (), []):
        raise ValueError(f"current suite profile不得另寫contrast_ids: suite={suite_id}")
    return (
        tuple(str(value) for value in suite.get("arm_ids", ())),
        tuple(str(value) for value in suite.get("contrast_ids", ())),
    )


def _arm_has_seed_sensitive_model_dependency(arm: StrategyComparisonArm) -> bool:
    return bool(arm.dl_enabled and str(arm.dl_id or "").strip())


def _derived_robustness_membership(
    profile_settings: StrategyComparisonSettings,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Derive current robustness roles from the shared Compare Suite.

    Current robustness membership is derived from the same shared Model Compare/Test
    List that owns [3]～[6].  A historical single-seed-only source restriction remains
    fail-closed only when that profile is not selected by the current shared list; once
    selected, the shared-list work item is the current robustness authorization.
    The first tuple is intentionally empty and exists only for the legacy fixed-arm
    compatibility shape used by disabled historical robustness profiles.
    """
    robustness_eligible = tuple(
        arm
        for arm in profile_settings.enabled_arms
        if not (
            arm.dl_enabled
            and str(arm.dl_id or "").strip()
            and bool(
                profile_settings.dl_sources[str(arm.dl_id)].single_seed_strategy_conversion_authorized
            )
            and not is_breakout_quality_model_test_profile(
                profile_settings.dl_sources[str(arm.dl_id)].experiment_profile
            )
        )
    )
    benchmark = tuple(arm.arm_id for arm in robustness_eligible)
    model_sensitive = tuple(
        arm.arm_id
        for arm in robustness_eligible
        if _arm_has_seed_sensitive_model_dependency(arm)
    )
    if not benchmark or not model_sensitive:
        raise ValueError(
            "Compare Suite end-to-end robustness角色不完整: "
            f"suite={profile_settings.suite_id}, benchmark={list(benchmark)}, "
            f"model_sensitive={list(model_sensitive)}"
        )
    return (), benchmark, model_sensitive


def _derived_stochastic_contrasts(
    profile_settings: StrategyComparisonSettings, stochastic_arm_ids: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    stochastic = set(stochastic_arm_ids)
    rows: list[dict[str, str]] = []
    for contrast in profile_settings.enabled_contrasts:
        if contrast.left in stochastic and contrast.right in stochastic:
            rows.append({
                "contrast_id": contrast.contrast_id,
                "left": contrast.left,
                "right": contrast.right,
                "description": contrast.description,
            })
    return tuple(rows)


def get_strategy_runtime_integration_settings() -> StrategyRuntimeIntegrationSettings:
    raw = dict(STRATEGY_RUNTIME_INTEGRATION)
    settings = StrategyRuntimeIntegrationSettings(
        label=str(raw.get("label") or "").strip(),
        enabled=bool(raw.get("enabled", True)),
        selection_profile_id=str(raw.get("selection_profile_id") or "").strip(),
        forward_profile_id=str(raw.get("forward_profile_id") or "").strip(),
        selection_candidate_arm_id=str(raw.get("selection_candidate_arm_id") or "").strip(),
        forward_candidate_arm_id=str(raw.get("forward_candidate_arm_id") or "").strip(),
        selection_robustness_id=str(raw.get("selection_robustness_id") or "").strip(),
        forward_robustness_id=str(raw.get("forward_robustness_id") or "").strip(),
        output_root=str(raw.get("output_root") or "").strip(),
        comparison_anchor_experiment_profile=str(
            raw.get("comparison_anchor_experiment_profile") or ""
        ).strip(),
        require_strict_romd_majority=bool(raw.get("require_strict_romd_majority", True)),
        max_selector_latency_ms=float(raw.get("max_selector_latency_ms", 10000.0)),
    )
    validate_strategy_runtime_integration_settings(settings)
    profile_ids = set(STRATEGY_COMPARE_PROFILES)
    robustness_ids = set(STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES)
    if settings.selection_profile_id not in profile_ids or settings.forward_profile_id not in profile_ids:
        raise ValueError("runtime integration引用不存在的Strategy Compare profile")
    candidate_specs = (
        (settings.selection_profile_id, settings.selection_candidate_arm_id, "Selection"),
        (settings.forward_profile_id, settings.forward_candidate_arm_id, "Forward"),
    )
    for profile_id, arm_id, stage_label in candidate_specs:
        active_ids = set(_resolved_profile_matrix(dict(STRATEGY_COMPARE_PROFILES[profile_id]))[0])
        if arm_id not in active_ids:
            raise ValueError(
                f"runtime integration {stage_label} candidate不在active profile: {arm_id}"
            )
    if settings.selection_robustness_id not in robustness_ids or settings.forward_robustness_id not in robustness_ids:
        raise ValueError("runtime integration引用不存在的robustness profile")
    return settings


def get_strategy_multi_seed_robustness_profiles() -> tuple[dict[str, str], ...]:
    return tuple(
        {"robustness_id": str(robustness_id), "label": str(raw.get("label") or robustness_id)}
        for robustness_id, raw in STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES.items()
        if bool(raw.get("enabled", True))
    )


def get_strategy_multi_seed_robustness_settings(
    robustness_id: str | None = None,
) -> StrategyMultiSeedRobustnessSettings:
    selected_id = str(robustness_id or STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE).strip()
    if selected_id not in STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES:
        raise ValueError(f"不存在的multi-seed robustness profile: {selected_id}")
    raw = dict(STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES[selected_id])
    settings = StrategyMultiSeedRobustnessSettings(
        robustness_id=selected_id,
        label=str(raw.get("label") or selected_id).strip(),
        enabled=bool(raw.get("enabled", True)),
        profile_id=str(raw.get("profile_id") or "").strip(),
        suite_id=(None if raw.get("suite_id") in (None, "") else str(raw.get("suite_id")).strip()),
        benchmark_id=(None if raw.get("benchmark_id") in (None, "") else str(raw.get("benchmark_id")).strip()),
        seed_count=int(raw.get("seed_count", ROBUSTNESS_BENCHMARK_SEED_COUNT) or 0),
        seed_generator_seed=int(raw.get("seed_generator_seed", ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED) or 0),
        resolved_seeds=tuple(
            int(value) for value in tuple(
                raw.get("resolved_seeds")
                or (resolve_robustness_benchmark_seeds() if raw.get("benchmark_id") not in (None, "") else ())
                or ()
            )
        ),
        strategy_trials_per_fold=(
            int(get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling")["trials_per_fold"])
            if raw.get("benchmark_id") not in (None, "")
            else (
                None if raw.get("strategy_trials_per_fold") in (None, "")
                else int(raw.get("strategy_trials_per_fold"))
            )
        ),
        gpu_train_workers=int(raw.get("gpu_train_workers", STRATEGY_COMPARE_GPU_TRAIN_WORKERS)),
        cpu_replay_workers=int(raw.get("cpu_replay_workers", 0) or 0),
        reuse_completed=bool(raw.get("reuse_completed", True)),
        console_mode=str(raw.get("console_mode") or "compact").strip(),
        progress_interval_seconds=float(raw.get("progress_interval_seconds", 60.0) or 60.0),
        yearly_report=bool(raw.get("yearly_report", True)),
        keep_checkpoints=bool(raw.get("keep_checkpoints", False)),
        keep_scores=bool(raw.get("keep_scores", False)),
        keep_replay_details=bool(raw.get("keep_replay_details", False)),
        keep_attribution_source=bool(raw.get("keep_attribution_source", True)),
        romd_reference_baselines={
            str(key).strip(): {
                "param_source": str(dict(value or {}).get("param_source") or "").strip(),
                "param_policy": str(dict(value or {}).get("param_policy") or "").strip(),
                "rule_policy": str(dict(value or {}).get("rule_policy") or "").strip(),
            }
            for key, value in dict(raw.get("romd_reference_baselines") or {}).items()
        },
        fixed_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("fixed_arm_ids") or ()) if str(value).strip()),
        stochastic_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("stochastic_arm_ids") or ()) if str(value).strip()),
        benchmark_strategy_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("benchmark_strategy_arm_ids") or raw.get("stochastic_arm_ids") or ()) if str(value).strip()),
        model_seed_sensitive_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("model_seed_sensitive_arm_ids") or raw.get("stochastic_arm_ids") or ()) if str(value).strip()),
        consensus_reference_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("consensus_reference_arm_ids") or raw.get("fixed_arm_ids") or ()) if str(value).strip()),
        paired_contrasts=tuple(
            {
                "contrast_id": str(dict(item or {}).get("contrast_id") or "").strip(),
                "left": str(dict(item or {}).get("left") or "").strip(),
                "right": str(dict(item or {}).get("right") or "").strip(),
                "description": str(dict(item or {}).get("description") or "").strip(),
            }
            for item in tuple(raw.get("paired_contrasts") or ())
        ),
        output_root=str(raw.get("output_root") or "").strip(),
        model_work_root=str(raw.get("model_work_root") or "").strip(),
        checkpoint_cache_root=(
            None
            if raw.get("checkpoint_cache_root") in (None, "")
            else str(raw.get("checkpoint_cache_root")).strip()
        ),
    )
    if settings.suite_id is None:
        if not settings.resolved_seeds:
            settings = replace(
                settings,
                resolved_seeds=resolve_robustness_benchmark_seeds(
                    seed_count=settings.seed_count, generator_seed=settings.seed_generator_seed
                ),
            )
        validate_strategy_multi_seed_robustness_settings(settings)
    if settings.profile_id not in STRATEGY_COMPARE_PROFILES:
        raise ValueError(
            f"multi-seed robustness引用不存在的Strategy Compare profile: {settings.profile_id}"
        )
    profile_settings = get_strategy_comparison_settings(settings.profile_id)
    if settings.suite_id is not None:
        if profile_settings.suite_id != settings.suite_id:
            raise ValueError(
                "multi-seed robustness suite與Strategy Compare profile不一致: "
                f"robustness={settings.suite_id}, profile={profile_settings.suite_id}"
            )
        fixed_ids, benchmark_ids, model_ids = _derived_robustness_membership(profile_settings)
        settings = replace(
            settings,
            fixed_arm_ids=fixed_ids,
            stochastic_arm_ids=benchmark_ids,
            benchmark_strategy_arm_ids=benchmark_ids,
            model_seed_sensitive_arm_ids=model_ids,
            consensus_reference_arm_ids=(),
            paired_contrasts=_derived_stochastic_contrasts(profile_settings, benchmark_ids),
        )
        validate_strategy_multi_seed_robustness_settings(settings)
    enabled_by_id = {arm.arm_id: arm for arm in profile_settings.enabled_arms}
    missing_fixed = [arm_id for arm_id in settings.fixed_arm_ids if arm_id not in enabled_by_id]
    missing_stochastic = [arm_id for arm_id in settings.stochastic_arm_ids if arm_id not in enabled_by_id]
    if missing_fixed or missing_stochastic:
        raise ValueError(
            "multi-seed robustness引用未啟用或不存在的arm: "
            f"fixed={missing_fixed}, stochastic={missing_stochastic}"
        )
    fixed = [enabled_by_id[arm_id] for arm_id in settings.fixed_arm_ids]
    stochastic = [enabled_by_id[arm_id] for arm_id in settings.stochastic_arm_ids]
    stochastic_ids = {arm.arm_id for arm in stochastic}
    for spec in settings.paired_contrasts:
        left = str(dict(spec).get("left") or "")
        right = str(dict(spec).get("right") or "")
        if left not in stochastic_ids or right not in stochastic_ids:
            raise ValueError(
                "multi-seed paired contrast只能引用目前stochastic arms: "
                f"contrast={dict(spec).get('contrast_id')}, left={left}, right={right}, "
                f"stochastic={sorted(stochastic_ids)}"
            )
    score_sources = {
        profile_settings.dl_sources[str(arm.dl_id)].score_source
        for arm in stochastic if arm.dl_id
    }
    if len(score_sources) > 1:
        raise ValueError(
            "multi-seed stochastic DL arms必須使用同一score source: "
            f"actual={sorted(score_sources)}"
        )
    reference_pool = stochastic if settings.benchmark_id is not None else fixed
    reference_role = "same-seed benchmark baseline" if settings.benchmark_id is not None else "fixed baseline"
    for reference_key, spec in settings.romd_reference_baselines.items():
        expected_param_policy = str(
            spec.get("param_policy") or profile_settings.param_policy
        ).strip()
        matches = [
            arm for arm in reference_pool
            if arm.param_source == spec["param_source"]
            and resolve_strategy_comparison_arm_param_policy(profile_settings, arm)
            == expected_param_policy
            and arm.rule_policy == spec["rule_policy"]
            and not arm.dl_enabled
        ]
        if len(matches) != 1:
            raise ValueError(
                f"multi-seed RoMD reference必須唯一對應一個{reference_role}: "
                f"reference={reference_key}, param_source={spec['param_source']}, "
                f"param_policy={expected_param_policy}, rule_policy={spec['rule_policy']}, "
                f"matches={len(matches)}"
            )
    return settings


def get_strategy_rolling_test_modes() -> tuple[dict[str, object], ...]:
    """Return OOS/Rolling mode bindings for the nested Strategy Compare menus."""

    rows: list[dict[str, object]] = []
    seen_profiles: set[str] = set()
    seen_robustness: set[str] = set()
    for raw in STRATEGY_COMPARE_ROLLING_TEST_MODES:
        item = dict(raw)
        profile_id = str(item.get("profile_id") or "").strip()
        robustness_id = str(item.get("robustness_id") or "").strip()
        if profile_id not in STRATEGY_COMPARE_PROFILES:
            raise ValueError(f"Rolling Test mode引用不存在Strategy Compare profile: {profile_id!r}")
        if robustness_id not in STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES:
            raise ValueError(f"Rolling Test mode引用不存在robustness profile: {robustness_id!r}")
        if profile_id in seen_profiles or robustness_id in seen_robustness:
            raise ValueError("Rolling Test mode profile/robustness不可重複")
        rows.append(item)
        seen_profiles.add(profile_id)
        seen_robustness.add(robustness_id)
    if not rows:
        raise ValueError("Strategy Compare至少需要一個Rolling Test mode")
    current_suite_ids = {
        _resolved_suite_id(dict(STRATEGY_COMPARE_PROFILES[str(item["profile_id"])]))
        for item in rows
    }
    if None in current_suite_ids or len(current_suite_ids) != 1:
        raise ValueError(
            "current OOS／Rolling modes必須引用同一Compare Suite: "
            f"suite_ids={sorted(str(value) for value in current_suite_ids)}"
        )
    for item in rows:
        raw_robustness = dict(
            STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES[str(item["robustness_id"])]
        )
        if str(raw_robustness.get("suite_id") or "").strip() not in current_suite_ids:
            raise ValueError(
                "current robustness mode必須引用與single-seed相同Compare Suite: "
                f"robustness={item['robustness_id']}"
            )
    return tuple(rows)


def get_strategy_compare_suite(suite_id: str) -> dict[str, object]:
    selected = str(suite_id).strip()
    if selected not in STRATEGY_COMPARE_SUITES:
        raise ValueError(f"不存在的Strategy Compare suite: {selected}")
    raw = dict(STRATEGY_COMPARE_SUITES[selected])
    arm_ids = tuple(str(value) for value in raw.get("arm_ids", ()))
    contrast_ids = tuple(str(value) for value in raw.get("contrast_ids", ()))
    display_name_bases = {
        str(key): str(value).strip()
        for key, value in dict(raw.get("display_name_bases") or {}).items()
    }
    if not arm_ids or len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"Strategy Compare suite arm_ids不可空白／重複: {selected}")
    if len(set(contrast_ids)) != len(contrast_ids):
        raise ValueError(f"Strategy Compare suite contrast_ids不得重複: {selected}")
    if set(display_name_bases) != set(arm_ids):
        raise ValueError(
            f"Strategy Compare suite display_name_bases必須完整覆蓋arms: suite={selected}"
        )
    missing_arms = [arm_id for arm_id in arm_ids if arm_id not in STRATEGY_COMPARE_ARMS]
    missing_contrasts = [cid for cid in contrast_ids if cid not in STRATEGY_COMPARE_CONTRASTS]
    if missing_arms or missing_contrasts:
        raise ValueError(
            f"Strategy Compare suite引用不存在設定: arms={missing_arms}, contrasts={missing_contrasts}"
        )
    for cid in contrast_ids:
        spec = dict(STRATEGY_COMPARE_CONTRASTS[cid])
        if str(spec.get("left") or "") not in arm_ids or str(spec.get("right") or "") not in arm_ids:
            raise ValueError(f"Strategy Compare suite contrast端點不在suite: {selected}/{cid}")

    # Current Extending matrix consumes the same model-level SSOT as [1][3]～[1][6].
    # Baselines remain enabled. DL arms are enabled only when their source profile is
    # selected in BREAKOUT_QUALITY_MODEL_TEST_PROFILES and a formal arm already exists.
    # A selected model without an authorized strategy arm remains MODEL-ONLY; no binding
    # is synthesized here. Historical/non-current suites keep their stored membership.
    if selected == "extending_current":
        selected_profiles = {
            str(profile)
            for _model_id, profile in get_breakout_quality_model_test_settings().model_profiles
        }
        active_arm_ids = []
        for arm_id in arm_ids:
            arm = dict(STRATEGY_COMPARE_ARMS[arm_id])
            if not bool(arm.get("dl_enabled")):
                active_arm_ids.append(arm_id)
                continue
            dl_id = str(arm.get("dl_id") or "").strip()
            source = dict(STRATEGY_DL_SOURCES.get(dl_id) or {})
            if str(source.get("experiment_profile") or "") in selected_profiles:
                active_arm_ids.append(arm_id)
        active = tuple(active_arm_ids)
        active_set = set(active)
        contrast_ids = tuple(
            cid for cid in contrast_ids
            if str(STRATEGY_COMPARE_CONTRASTS[cid].get("left") or "") in active_set
            and str(STRATEGY_COMPARE_CONTRASTS[cid].get("right") or "") in active_set
        )
        arm_ids = active
        display_name_bases = {key: value for key, value in display_name_bases.items() if key in active_set}

    return {
        "suite_id": selected,
        "arm_ids": arm_ids,
        "contrast_ids": contrast_ids,
        "display_name_bases": display_name_bases,
    }


def get_strategy_compare_model_bindings() -> tuple[dict[str, object], ...]:
    """Resolve shared model-list bindings without authorizing new strategy conversions."""

    rows = []
    for model_id, profile in get_breakout_quality_model_test_settings().model_profiles:
        bound = []
        for arm_id, arm_raw in STRATEGY_COMPARE_ARMS.items():
            arm = dict(arm_raw)
            if not bool(arm.get("dl_enabled")):
                continue
            source = dict(STRATEGY_DL_SOURCES.get(str(arm.get("dl_id") or "")) or {})
            if str(source.get("experiment_profile") or "") == str(profile):
                bound.append(str(arm_id))
        rows.append({
            "model_id": str(model_id),
            "experiment_profile": str(profile),
            "strategy_arm_ids": tuple(bound),
            "status": "BOUND" if bound else "MODEL-ONLY / NO STRATEGY BINDING",
        })
    return tuple(rows)


def get_strategy_comparison_profiles() -> tuple[dict[str, str], ...]:
    """Return the full configured profile catalog, including historical/replay profiles."""
    return tuple(
        {
            "profile_id": str(profile_id),
            "label": str(raw["label"]),
            "description": str(raw.get("description") or ""),
        }
        for profile_id, raw in STRATEGY_COMPARE_PROFILES.items()
    )


def get_strategy_comparison_menu_profiles() -> tuple[dict[str, str], ...]:
    """Return only the config-selected generic work stages exposed by the main menu."""
    missing = [
        profile_id
        for profile_id in STRATEGY_COMPARE_MENU_PROFILE_IDS
        if profile_id not in STRATEGY_COMPARE_PROFILES
    ]
    if missing:
        raise ValueError(f"Strategy Compare主選單引用不存在profile: {missing}")
    return tuple(
        {
            "profile_id": str(profile_id),
            "label": str(STRATEGY_COMPARE_PROFILES[profile_id]["label"]),
            "description": str(STRATEGY_COMPARE_PROFILES[profile_id].get("description") or ""),
        }
        for profile_id in STRATEGY_COMPARE_MENU_PROFILE_IDS
    )


def get_strategy_comparison_settings(profile_id: str | None = None) -> StrategyComparisonSettings:
    selected_profile_id = str(profile_id or STRATEGY_COMPARE_DEFAULT_PROFILE).strip()
    if selected_profile_id not in STRATEGY_COMPARE_PROFILES:
        raise ValueError(f"未知Strategy Compare profile: {selected_profile_id}")
    profile = dict(STRATEGY_COMPARE_PROFILES[selected_profile_id])
    profile_arm_ids, profile_contrast_ids = _resolved_profile_matrix(profile)
    profile_suite_id = _resolved_suite_id(profile)
    profile_suite_display_bases = (
        {} if profile_suite_id is None
        else {
            str(key): str(value).strip()
            for key, value in dict(get_strategy_compare_suite(profile_suite_id).get("display_name_bases") or {}).items()
        }
    )
    profile_display_suffix = (
        None if profile.get("display_suffix") in (None, "")
        else str(profile.get("display_suffix")).strip()
    )
    (
        parameter_catalog,
        dl_catalog,
        arm_catalog,
        contrast_catalog,
    ) = _compatibility_catalogs()
    missing_arms = [arm_id for arm_id in profile_arm_ids if arm_id not in arm_catalog]
    missing_contrasts = [
        contrast_id
        for contrast_id in profile_contrast_ids
        if contrast_id not in contrast_catalog
    ]
    if missing_arms or missing_contrasts:
        raise ValueError(
            "Strategy Compare profile引用不存在的設定: "
            f"arms={missing_arms or '-'}, contrasts={missing_contrasts or '-'}"
        )

    preparation_raw = _get_strategy_compare_preparation()
    preparation = StrategyPreparationPolicy(
        auto_prepare=bool(preparation_raw.get("auto_prepare")),
        reuse_ready_artifacts=bool(
            preparation_raw.get("reuse_ready_artifacts")
        ),
        rebuild_stale_artifacts=bool(
            preparation_raw.get("rebuild_stale_artifacts")
        ),
        resume_parameter_training=bool(
            preparation_raw.get("resume_parameter_training")
        ),
        require_confirmation=bool(
            preparation_raw.get("require_confirmation")
        ),
        reuse_completed_results=bool(
            preparation_raw.get("reuse_completed_results", True)
        ),
        reuse_shared_baseline=bool(
            preparation_raw.get("reuse_shared_baseline", True)
        ),
    )
    parameter_sources = {
        source_id: StrategyParameterSource(
            source_id=source_id,
            path_template=raw.get("path_template"),
            description=str(raw.get("description") or "").strip(),
            identity_manifest_path=raw.get("identity_manifest_path"),
            trained_with_dl_id=raw.get("trained_with_dl_id"),
            artifact_contract=(
                None
                if raw.get("artifact_contract") in (None, {})
                else dict(raw.get("artifact_contract") or {})
            ),
            builder=_builder(raw.get("builder")),
            canonical_family=(None if raw.get("canonical_family") in (None, "") else str(raw.get("canonical_family"))),
            canonical_evaluation_mode=(None if raw.get("canonical_evaluation_mode") in (None, "") else str(raw.get("canonical_evaluation_mode"))),
        )
        for source_id, raw in parameter_catalog.items()
    }
    profile_pit_score_start_date = (
        None
        if profile.get("point_in_time_score_start_date") in (None, "")
        else str(profile.get("point_in_time_score_start_date")).strip()
    )
    profile_pit_score_end_date = (
        None
        if profile.get("point_in_time_score_end_date") in (None, "")
        else str(profile.get("point_in_time_score_end_date")).strip()
    )
    profile_pit_single_score_block = bool(profile.get("point_in_time_single_score_block", False))
    profile_pit_fold_months = (
        None
        if profile.get("point_in_time_fold_months") in (None, "")
        else int(profile.get("point_in_time_fold_months"))
    )
    profile_pit_anchor_date = (
        None
        if profile.get("point_in_time_fold_anchor_date") in (None, "")
        else str(profile.get("point_in_time_fold_anchor_date")).strip()
    )
    profile_pit_dirname = (
        None
        if profile.get("point_in_time_dirname") in (None, "")
        else str(profile.get("point_in_time_dirname")).strip()
    )
    dl_sources = {
        dl_id: StrategyDLSource(
            dl_id=dl_id,
            filter_id=str(raw.get("filter_id") or "").strip(),
            model_architecture=str(raw.get("model_architecture") or "").strip(),
            experiment_profile=str(raw.get("experiment_profile") or "").strip(),
            threshold=(None if raw.get("threshold") in (None, "") else float(raw.get("threshold"))),
            description=str(raw.get("description") or "").strip(),
            score_source=str(raw.get("score_source") or "canonical_runtime").strip(),
            single_seed_strategy_conversion_authorized=bool(
                raw.get("single_seed_strategy_conversion_authorized", False)
            ),
            forward_scores_builder=_builder(raw.get("forward_scores_builder")),
            point_in_time_score_start_date=(
                profile_pit_score_start_date
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
            point_in_time_score_end_date=(
                profile_pit_score_end_date
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
            point_in_time_fold_months=(
                profile_pit_fold_months
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
            point_in_time_fold_anchor_date=(
                profile_pit_anchor_date
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
            point_in_time_single_score_block=(
                profile_pit_single_score_block
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else False
            ),
            point_in_time_dirname=(
                profile_pit_dirname
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
        )
        for dl_id, raw in dl_catalog.items()
    }
    profile_arm_param_source_overrides = {
        str(arm_id): str(source_id)
        for arm_id, source_id in dict(profile.get("arm_param_source_overrides") or {}).items()
    }
    unknown_param_override_arms = sorted(
        set(profile_arm_param_source_overrides) - set(arm_catalog)
    )
    unknown_param_override_sources = sorted(
        set(profile_arm_param_source_overrides.values()) - set(parameter_sources)
    )
    if unknown_param_override_arms or unknown_param_override_sources:
        raise ValueError(
            "Strategy Compare profile param-source override無效: "
            f"arms={unknown_param_override_arms or '-'}, "
            f"params={unknown_param_override_sources or '-'}"
        )
    ordered_arm_ids = tuple(dict.fromkeys((*profile_arm_ids, *arm_catalog.keys())))
    arms = {
        arm_id: StrategyComparisonArm(
            arm_id=arm_id,
            enabled=arm_id in profile_arm_ids,
            name=(
                f"{profile_suite_display_bases.get(arm_id, str(raw.get('name') or '').strip())} {profile_display_suffix}"
                if arm_id in profile_suite_display_bases and profile_display_suffix
                else str(raw.get("name") or "").strip()
            ),
            description=str(raw.get("description") or "").strip(),
            param_source=str(
                profile_arm_param_source_overrides.get(
                    arm_id, raw.get("param_source") or ""
                )
            ).strip(),
            param_policy=(
                None
                if raw.get("param_policy") in (None, "")
                else str(raw.get("param_policy")).strip()
            ),
            rule_policy=str(raw.get("rule_policy") or "").strip(),
            dl_enabled=bool(raw.get("dl_enabled")),
            dl_id=(None if raw.get("dl_id") in (None, "") else str(raw.get("dl_id"))),
            dl_runtime_mode=(
                None
                if raw.get("dl_runtime_mode") in (None, "")
                else str(raw.get("dl_runtime_mode")).strip()
            ),
            dl_runtime_options=(
                None
                if raw.get("dl_runtime_options") in (None, {})
                else dict(raw.get("dl_runtime_options") or {})
            ),
            robustness_role=str(raw.get("robustness_role") or "off").strip(),
        )
        for arm_id in ordered_arm_ids
        for raw in (arm_catalog[arm_id],)
    }
    ordered_contrast_ids = tuple(dict.fromkeys((*profile_contrast_ids, *contrast_catalog.keys())))
    contrasts = {
        contrast_id: StrategyComparisonContrast(
            contrast_id=contrast_id,
            enabled=contrast_id in profile_contrast_ids,
            left=str(raw.get("left") or "").strip(),
            right=str(raw.get("right") or "").strip(),
            description=str(raw.get("description") or "").strip().format(
                left=arms[str(raw.get("left") or "").strip()].name,
                right=arms[str(raw.get("right") or "").strip()].name,
            ),
        )
        for contrast_id in ordered_contrast_ids
        for raw in (contrast_catalog[contrast_id],)
    }
    workflow_settings = get_breakout_quality_workflow_settings()
    settings = StrategyComparisonSettings(
        schema_version=int(STRATEGY_COMPARE_SCHEMA_VERSION),
        profile_id=selected_profile_id,
        profile_label=str(profile["label"]),
        suite_id=profile_suite_id,
        display_suffix=profile_display_suffix,
        dataset=str(workflow_settings.strategy_dataset).strip(),
        start_date=(
            None
            if profile.get("start_date") in (None, "")
            else str(profile.get("start_date")).strip()
        ),
        end_date=(
            None
            if profile.get("end_date") in (None, "")
            else str(profile.get("end_date")).strip()
        ),
        param_policy=str(workflow_settings.strategy_param_policy).strip(),
        max_positions=int(workflow_settings.strategy_max_positions),
        rotation=str(workflow_settings.strategy_rotation).strip(),
        output_root=str(profile["output_root"]).strip(),
        reuse_output_roots=tuple(
            str(value).strip()
            for value in tuple(profile.get("reuse_output_roots") or ())
            if str(value).strip()
        ),
        preparation=preparation,
        parameter_sources=parameter_sources,
        dl_sources=dl_sources,
        arms=arms,
        contrasts=contrasts,
    )
    validate_strategy_comparison_settings(settings)
    return settings


def validate_single_seed_strategy_conversion_authorization(
    *,
    strategy_profile_id: str,
    dl_id: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    seed: int,
) -> StrategyDLSource:
    """Validate the narrow Strategy Compare exception for one canonical single seed.

    This is deliberately separate from the model research authorization.  It cannot be
    used by generic PIT/Rolling/Fixed workflows or multi-seed robustness.
    """

    profile_id = str(strategy_profile_id or "").strip()
    if profile_id not in STRATEGY_COMPARE_MENU_PROFILE_IDS:
        raise ValueError(
            "single-seed strategy conversion只允許current OOS/Rolling Strategy Compare profile"
        )
    settings = get_strategy_comparison_settings(profile_id)
    source_id = str(dl_id or "").strip()
    source = settings.dl_sources.get(source_id)
    if source is None:
        raise ValueError(f"Strategy Compare conversion source不存在: {source_id!r}")
    if not bool(source.single_seed_strategy_conversion_authorized):
        raise ValueError(
            f"Strategy Compare source未授權single-seed conversion: {source_id}"
        )
    if not any(
        arm.enabled and arm.dl_enabled and str(arm.dl_id or "") == source_id
        for arm in settings.arms.values()
    ):
        raise ValueError(
            f"Strategy Compare source未被目前single-seed suite引用: {source_id}"
        )
    actual_identity = (
        str(source.filter_id),
        str(source.model_architecture),
        str(source.experiment_profile),
    )
    expected_identity = (
        str(filter_id),
        str(model_architecture),
        str(experiment_profile),
    )
    if actual_identity != expected_identity:
        raise ValueError(
            "Strategy Compare conversion source identity不一致: "
            f"source={actual_identity}, requested={expected_identity}"
        )
    workflow = get_breakout_quality_workflow_settings(
        experiment_profile=str(source.experiment_profile)
    )
    if int(seed) != int(workflow.seed):
        raise ValueError(
            "single-seed strategy conversion只允許canonical workflow seed: "
            f"expected={int(workflow.seed)}, actual={int(seed)}"
        )
    return source


def validate_single_seed_strategy_conversion_pit_scope(
    *,
    strategy_profile_id: str,
    dl_id: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    seed: int,
    score_start_date: str,
    score_end_date: str | None,
    fold_months: int,
    fold_anchor_date: str | None,
    single_score_block: bool,
    inner_validation_months: int,
    train_window_months: int | None,
) -> StrategyDLSource:
    """Validate the exact PIT build scope of a single-seed strategy exception.

    The source-scoped exception authorizes only the canonical OOS/Rolling PIT plan
    owned by the selected Strategy Compare profile.  It must not be reusable as a
    hidden escape hatch for Fixed-Window or arbitrary date/fold PIT builds.
    """

    source = validate_single_seed_strategy_conversion_authorization(
        strategy_profile_id=strategy_profile_id,
        dl_id=dl_id,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        seed=seed,
    )
    if train_window_months is not None:
        raise ValueError(
            "single-seed Strategy Compare conversion只授權canonical OOS/Rolling；"
            "Fixed-Window PIT不得使用此例外"
        )
    workflow = get_breakout_quality_workflow_settings(
        experiment_profile=str(source.experiment_profile)
    )
    expected = {
        "score_start_date": str(source.point_in_time_score_start_date or "").strip(),
        "score_end_date": (
            None
            if source.point_in_time_score_end_date in (None, "")
            else str(source.point_in_time_score_end_date).strip()
        ),
        "fold_months": int(source.point_in_time_fold_months or workflow.point_in_time_fold_months),
        "fold_anchor_date": (
            None
            if source.point_in_time_fold_anchor_date in (None, "")
            else str(source.point_in_time_fold_anchor_date).strip()
        ),
        "single_score_block": bool(source.point_in_time_single_score_block),
        "inner_validation_months": int(workflow.point_in_time_inner_validation_months),
    }
    actual = {
        "score_start_date": str(score_start_date or "").strip(),
        "score_end_date": (
            None if score_end_date in (None, "") else str(score_end_date).strip()
        ),
        "fold_months": int(fold_months),
        "fold_anchor_date": (
            None
            if fold_anchor_date in (None, "")
            else str(fold_anchor_date).strip()
        ),
        "single_score_block": bool(single_score_block),
        "inner_validation_months": int(inner_validation_months),
    }
    if actual != expected:
        raise ValueError(
            "single-seed Strategy Compare conversion PIT scope不一致: "
            f"expected={expected}, actual={actual}"
        )
    return source


__all__ = [
    "get_strategy_compare_suite",
    "get_strategy_rolling_test_modes",
    "get_strategy_comparison_profiles",
    "get_strategy_comparison_menu_profiles",
    "get_strategy_multi_seed_robustness_profiles",
    "get_strategy_multi_seed_robustness_settings",
    "get_strategy_runtime_integration_settings",
    "get_strategy_comparison_settings",
    "get_strategy_compare_model_bindings",
    "validate_single_seed_strategy_conversion_authorization",
    "validate_single_seed_strategy_conversion_pit_scope",
]
