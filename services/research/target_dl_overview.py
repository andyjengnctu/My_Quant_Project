"""Read-only Target x DL evidence overview from certified local research results.

No training, replay, target materialization, output writing or result backfill.
Registries define membership/identities; certified Oracle receipts and approved
model/strategy reports supply metrics through their canonical result owners.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from statistics import fmean
from pathlib import Path
from typing import Any, Mapping

from core.research_summary import (get_research_summary_rows, research_summary_entry_label,
                                   research_summary_selector_label)
from core.audit_policy import get_active_audit_module_id
from core.audit_registry import (resolve_current_oracle_comparisons, oracle_comparison_key,
                                 oracle_parameter_family, oracle_baseline_arm_ids, oracle_baseline_display_name,
                                 oracle_work_inventory, oracle_work_display_dimensions,
                                 oracle_display_strategy,
                                 validate_oracle_compare_family_coverage,
                                 oracle_supervised_target_id, oracle_target_semantics)
from core.breakout_quality_registry import (
    SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
    get_breakout_quality_experiment_profile,
    get_continuous_ranker_research_spec,
)
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.breakout_quality_policy import (
    get_breakout_quality_model_test_settings, get_breakout_quality_workflow_settings,
)
from core.breakout_quality_fit_contract_version import (
    current_fit_contract_artifact_identity, published_fit_contract_version_state,
)
from core.report_contract_versions import load_report_contract_versions
from core.research_report_contract import (
    format_contract_value,
    require_approved_report_contract,
    table_contract,
    target_dl_summary_sort_columns,
)
from core.report_metrics import weighted_rolling_robustness_romd
from core.console_report import render_table, render_title
from core.report_style import best_worst_signals, full_column_extrema_only, styled_signal
from core.strategy_compare_policy import (
    get_strategy_comparison_settings,
    get_strategy_compare_suite,
    get_strategy_multi_seed_robustness_settings,
)
from core.strategy_compare_registry import (
    STRATEGY_COMPARE_ARMS, STRATEGY_COMPARE_PROFILES, STRATEGY_DL_SOURCES,
    STRATEGY_EXPERIMENTAL_PARAM_SOURCES,
)
from core.positive_score_admission import (
    POSITIVE_P_CLASS_COLUMN, requires_positive_score, uses_primary_positive_classifier,
)
from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_arm_runtime_dl_source_ids
from filters.breakout_quality.strategy_compare_pit_contract import (
    resolve_strategy_compare_selection_pit_bundle_dir,
    resolve_strategy_compare_selection_pit_contract_override,
)
from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
from filters.breakout_quality.paths import (
    resolve_filter_point_in_time_dir,
    resolve_filter_model_dir,
    resolve_filter_model_output_dir,
    resolve_filter_output_dir,
)
from services.research.target_oracle_application import get_formal_oracle_definition
from core.training_policy import get_robustness_benchmark_policy_snapshot
from config.robustness_benchmark import (
    TARGET_ORACLE_ROBUSTNESS_BENCHMARK_ID,
    MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID,
)
from services.research.oracle_artifact_contract import reference_arm_execution_matches
from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE


REPORT_ID = "research.target_dl_validation_overview"
DEFAULT_SORT_COLUMN = "oracle_romd"
_EXACT_MODE = "resource-aware-continuous-score-constrained-optimal"
_DIRECT_MODE = "resource-aware-continuous-score-no-k-no-r0"


def _summary_entry_label(arm: Mapping[str, Any], *, family: str) -> tuple[str, bool]:
    """Use the same Optimizer-owned parameter label as formal Oracle output."""
    return research_summary_entry_label(
        arm, family=family, parameter_sources=STRATEGY_EXPERIMENTAL_PARAM_SOURCES,
    )


def _summary_model_target_labels(groups: tuple[Mapping[str, Any], ...]) -> dict[str, set[str]]:
    """Recover a model's scientific Target from approved Oracle bindings only."""
    labels: dict[str, set[str]] = {}
    for group in groups:
        if "oracle" not in group:
            continue
        scientific_target, _, _ = oracle_target_semantics(group["oracle"])
        for binding in group.get("bindings", ()):
            pair = binding.get("rank_model")
            if pair:
                labels.setdefault(str(pair[1]), set()).add(scientific_target)
    return labels


def _summary_scientific_target(
    group: Mapping[str, Any], binding: Mapping[str, Any],
    model_targets: Mapping[str, set[str]],
) -> str:
    """Target is a supervised outcome, not strategy, parameter, selector or DL ID."""
    explicit = str(group.get("display_target") or "").strip()
    if binding.get("rank_models"):
        arm = STRATEGY_COMPARE_ARMS[binding["arm_id"]]
        from core.rank_score_composition import rank_score_composition
        composition = rank_score_composition(arm.get("dl_runtime_options"))
        targets = []
        for pair, member in zip(binding["rank_models"], composition["members"], strict=True):
            if member["weight"] > 0:
                labels = model_targets.get(pair[1], set())
                if len(labels) != 1:
                    return ""
                targets.extend(sorted(labels))
        return " + ".join(dict.fromkeys(targets))
    pair = binding.get("rank_model")
    candidates = model_targets.get(str(pair[1]), set()) if pair else set()
    if explicit:
        if not pair:
            raise ValueError(f"Summary Target requires a supervised model binding: {explicit}")
        if candidates and explicit not in candidates:
            raise ValueError(f"Summary Target conflicts with registered model Target: {explicit} / {candidates}")
        return explicit
    # A model may supervise several different scientific outcomes; don't guess
    # from its name, arm ID, or a strategy description if no unique Target exists.
    return next(iter(candidates)) if len(candidates) == 1 else ""


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _number(value: Any) -> float | None:
    from core.report_style import finite_number
    return finite_number(value)


def _count(value: Any) -> int | None:
    # A corrupt persisted count must never crash the read-only overview.
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 and result == value else None


def _approved(payload: Mapping[str, Any], report_id: str, versions: Mapping[str, tuple]) -> bool:
    # Reusable Audit uses an audit-local report_id, plus the approved contract_id;
    # Model and Strategy reports use the approved report_id/version directly.
    return any(
        (
            (payload.get("report_id") == report_id
             and payload.get("report_version") == record["version"])
            or payload.get("report_contract_id") == record["contract_id"]
        )
        and payload.get("report_contract_fingerprint") == record["fingerprint"]
        for record in versions.get(report_id, ())
    )


def _oracle_rows(root: Path, definition, versions: Mapping[str, tuple]):
    # The current canonical comparison inventory is the sole Target membership SSOT.
    comparisons = resolve_current_oracle_comparisons(definition.source)
    baseline = str(definition.source["baseline_arm_id"])
    # Comparison identity is Target x Entry Strategy; the scientific Target
    # remains unchanged and is displayed separately.  Old Breakout report
    # rows retain their original Target key, preserving historical REUSE.
    baselines = oracle_baseline_arm_ids(definition.source)
    targets = oracle_work_inventory(definition.source)
    if len(set(targets)) != len(targets):
        raise ValueError("Oracle Target x Strategy identities are not unique")
    by_comparison = {oracle_comparison_key(item): item for item in comparisons}
    current_definitions = {oracle_comparison_key(item): str(item["description"])
                           for item in comparisons}
    current_strategies = {oracle_comparison_key(item): str(item.get("strategy", "Breakout"))
                          for item in comparisons}
    # The Audit registry is the owner of retired historical identities; a
    # previously completed signed report may include them after [1] retires
    # those rows. Do not use them as current Oracle or DL results.
    retired = (set(definition.source.get("retired_oracle_targets") or ())
               | set(definition.source.get("historical_only_oracle_targets") or ()))
    if retired.intersection(oracle_comparison_key(item) for item in comparisons):
        raise ValueError("Current and retired Oracle treatment identities overlap")
    score_targets = {}
    for item in comparisons:
        raw = item
        if item.get("score_source"):
            raw = next((other for other in definition.source["oracle_source_catalog"]
                        if other["target"] == item["score_source"]), {})
        label = oracle_supervised_target_id(raw)
        # A hindsight-only eligibility intervention is not a deployable DL
        # model or ordinary Direct strategy, even when its score source is
        # identical. In particular Positive_Direct must NOT claim C124.
        score_targets[oracle_comparison_key(item)] = None if item.get("positive_gate") else label
    data_path = (root / "outputs" / "audit" / Path(definition.output_subdir)
                 / "latest" / "audit.json")
    doc_path = data_path.with_name("audit.md")
    payload = _json(data_path)
    current_oracle = get_robustness_benchmark_policy_snapshot(
        TARGET_ORACLE_ROBUSTNESS_BENCHMARK_ID
    )
    oracle_info: dict[str, Any] = {"path": None, "benchmark": "", "seeds": (), "period": ""}
    metrics: dict[str, dict[str, float | None]] = {}
    # Signed per-seed results are the canonical truth shared with [1][1].
    # Rebuilding a human-read publication or changing its model references
    # must not make certified Oracle metrics disappear from [1][2].
    # Each parameter family has its own certified replay cache. A new Full
    # family must not suppress certified Min receipts or vice versa.
    receipt_roots = tuple(
        root / "outputs" / "audit" / Path(definition.output_subdir) /
        ("replay_cache" if family == "min" else f"parameter_family_{family}/replay_cache")
        for family in baselines
    )
    if any(any(folder.glob("*/*.json")) for folder in receipt_roots):
        from services.audit.target_oracle_romd import collect_completed_oracle_results
        completed = collect_completed_oracle_results(definition, project_root=root)
        metrics = completed.pop("metrics")
        return targets, comparisons, score_targets, metrics, {
            "path": receipt_roots[0].relative_to(root).as_posix(), **completed,
        }
    if (payload and doc_path.is_file() and payload.get("status") == "COMPLETED"
            and payload.get("read_only") is True and payload.get("hindsight_non_pit") is True
            and _approved(payload, "audit.target_oracle_romd", versions)
            and isinstance(payload.get("oracles"), list)
            and isinstance(payload.get("seeds"), list)
            and payload.get("benchmark_id") == current_oracle["benchmark_id"]
            and tuple(payload["seeds"]) == tuple(current_oracle["resolved_seeds"])
            and payload.get("start_date") and payload.get("end_date")
            and payload.get("strategy_profile_id") == definition.source["strategy_profile_id"]):
        seen: set[str] = set()
        seen_scientific_targets: set[str] = set()
        published_definitions = payload.get("definitions")
        published_references = payload.get("comparison_reference_arms")
        if published_references is not None and not isinstance(published_references, dict):
            return targets, comparisons, score_targets, {}, oracle_info
        # Reference labels are provenance. The shared Oracle contract owner
        # verifies the actual score consumer, including historical receipts.
        settings = get_strategy_comparison_settings(str(definition.source["strategy_profile_id"]))
        original = [dict(payload.get("baseline") or {}),
                    *(dict(item) for item in payload.get("control_baselines", ())
                      if isinstance(item, dict)), *payload["oracles"]]
        control_arm_ids = set(baselines.values())
        for item in original:
            if not isinstance(item, dict):
                metrics.clear()
                break
            target = str(item.get("target") or "")
            strategy = str(item.get("strategy", "Breakout"))
            if target in control_arm_ids:
                # A no-DL control is the registered baseline arm, not a
                # Breakout-specific Target. Daily controls are equally valid.
                key = target
                expected_family = next(f for f, arm_id in baselines.items() if arm_id == target)
                registered_entry = str(STRATEGY_COMPARE_ARMS[target].get("entry_strategy", "Breakout"))
                if (strategy != registered_entry or
                        item.get("parameter_family") not in (None, expected_family)):
                    metrics.clear()
                    break
            else:
                try:
                    key = oracle_comparison_key({
                        "target": target, "strategy": strategy,
                        "parameter_family": item.get("parameter_family", "min"),
                    })
                except ValueError:
                    metrics.clear()
                    break
            if ((key not in targets and key not in retired) or key in seen or
                    _count(item.get("seeds")) != len(payload["seeds"]) or
                    (item.get("comparison_key") not in (None, key))):
                metrics.clear()
                break
            seen.add(key)
            if target not in control_arm_ids:
                seen_scientific_targets.add(target)
            romd, winrate = _number(item.get("romd")), _number(item.get("win_rate_pct"))
            if romd is None or winrate is None or not 0.0 <= winrate <= 100.0:
                metrics.clear()
                break
            current_comparison = by_comparison.get(key)
            consumer_matches = True
            if current_comparison is not None and current_comparison.get("strategy_arm_id"):
                if published_references is None:
                    # Historical v11/v12 results have no proof of which Daily
                    # strategy arm generated them.  Never present one as a
                    # current C133/C134 Oracle score-consumption result.
                    consumer_matches = False  # Explicit current consumer requires signed evidence.
                else:
                    consumer_matches = reference_arm_execution_matches(
                        published_references.get(key), current_comparison, settings)
            if (consumer_matches and (key in control_arm_ids or
                    (key in current_definitions
                     and isinstance(published_definitions, dict)
                     and published_definitions.get(target) == current_definitions[key]
                     and strategy == current_strategies[key]))):
                metrics[key] = {"oracle_romd": romd, "win_rate_pct": winrate}
        # A completed historical report is validated against its OWN published
        # inventory. Project only unchanged current rows; new/redefined Targets
        # must await their own completed Replay, never inherit a stale metric.
        if (baseline in seen and isinstance(published_definitions, dict)
                and set(published_definitions) == seen_scientific_targets
                and all(isinstance(desc, str) and desc for desc in published_definitions.values())
                and (payload.get("comparison_inventory") is None or (
                    isinstance(payload["comparison_inventory"], list)
                    and len(payload["comparison_inventory"]) == len(set(payload["comparison_inventory"]))
                    and set(payload["comparison_inventory"]) == seen
                    and all(key in targets for key in payload["comparison_inventory"])))
                and metrics):
            oracle_info = {
                "path": str(data_path.relative_to(root).as_posix()),
                "benchmark": str(payload.get("benchmark_id") or ""),
                "seeds": tuple(payload["seeds"]),
                "period": f'{payload["start_date"]} ~ {payload["end_date"]}',
            }
        else:
            metrics.clear()
    return targets, comparisons, score_targets, metrics, oracle_info


def _shares_oracle_scalar_truth(profile: str, target_id: str) -> bool:
    # Different DL tasks can share the SAME mature NetPnL truth.  This check
    # proves the input truth only; it does not identify the consumer as a ranker.
    from services.breakout_quality.trade_path_target_source import registered_trade_path_source
    registered = get_breakout_quality_experiment_profile(profile)
    recipe = get_continuous_ranker_execution_recipe(profile)
    policy = recipe.target_policy
    objective = recipe.objective_policy
    return bool(
        registered.continuous_target_id == target_id
        and (
            policy.component_target_id == target_id
            or (policy.materialization_mode == CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH
                and registered_trade_path_source(profile) is not None)
        )
        and policy.postprocess == CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE
        and policy.context_transform == CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE
        and objective.secondary_pair_scope == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL
        and objective.pair_target_schema in {"scalar", "scalar_with_context_weight"}
    )


def _is_exact_supervision(profile: str, target_id: str) -> bool:
    # Rolling rho is a *ranking* metric. A NetPnL>0 classifier must not be
    # presented as a continuous NetPnL ranker solely because it shares truth.
    return bool(
        _shares_oracle_scalar_truth(profile, target_id)
        and get_continuous_ranker_execution_recipe(profile).training_policy.categorical_class_count is None
    )


def _is_positive_classification(profile: str, target_id: str) -> bool:
    if not _shares_oracle_scalar_truth(profile, target_id):
        return False
    policy = get_continuous_ranker_execution_recipe(profile).training_policy
    return bool(
        policy.target_builder == "netpnl_positive_binary"
        and policy.loss_handler == "categorical_cross_entropy"
        and not policy.uses_pairwise_loss
        and policy.categorical_class_count == 2
        and policy.categorical_primary_class == 1
        and POSITIVE_P_CLASS_COLUMN in policy.categorical_score_columns
    )


def _positive_classifier_for_arm(arm: Mapping[str, Any], target_id: str) -> str | None:
    options = dict(arm.get("dl_runtime_options") or {})
    if not requires_positive_score(options):
        return None
    positive_id = arm.get("dl_id") if uses_primary_positive_classifier(options) else options["positive_dl_id"]
    positive_source = STRATEGY_DL_SOURCES.get(str(positive_id)) or {}
    profile = str(positive_source.get("experiment_profile") or "")
    if (profile not in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
            or not _is_positive_classification(profile, target_id)):
        return None
    return profile


def _model_paths(root: Path, *, filter_id: str, profile: str,
                 evaluation_mode: str = "rolling_oos") -> tuple[Path, Path]:
    registered = get_breakout_quality_experiment_profile(profile)
    from core.breakout_quality_policy import BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE
    architecture = str(registered.model_architecture or BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE)
    # AI: Standard SOP reports are EVIDENCE under outputs/, not FIT manifests
    # under models/. The latter are consulted separately for FIT lineage.
    directory = resolve_filter_model_output_dir(root, filter_id, architecture, profile)
    if evaluation_mode == "rolling_oos":
        directory = directory / "point_in_time"
    elif evaluation_mode != "forward_oos":
        raise ValueError(f"Unsupported Standard Model SOP evaluation_mode: {evaluation_mode}")
    return directory / "standard_model_report.json", directory / "standard_model_report.md"


def _model_fit_publication_dir(root: Path, *, filter_id: str, profile: str,
                               evaluation_mode: str) -> Path:
    registered = get_breakout_quality_experiment_profile(profile)
    from core.breakout_quality_policy import BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE
    architecture = str(registered.model_architecture or BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE)
    if evaluation_mode == "rolling_oos":
        return resolve_filter_point_in_time_dir(root, filter_id, architecture, profile)
    if evaluation_mode == "forward_oos":
        return resolve_filter_model_dir(root, filter_id, architecture, profile)
    raise ValueError(f"Unsupported Standard Model SOP evaluation_mode: {evaluation_mode}")


def _published_model_report(path: Path, markdown: Path, *, profile: str,
                            filter_id: str, evaluation_mode: str,
                            versions: Mapping[str, tuple],
                            fit_publication_dir: Path | None = None) -> dict[str, Any] | None:
    payload = _json(path)
    if not payload or not markdown.is_file() or not _approved(payload, "model.standard_sop", versions):
        return None
    if (payload.get("profile") != profile or payload.get("filter_id") != filter_id
            or payload.get("model_research_id") != get_continuous_ranker_research_spec(profile).model_research_id
            or payload.get("evaluation_mode") != evaluation_mode):
        return None
    try:
        # Match the canonical Model SOP publisher/readiness owner: digest the
        # decoded text, not raw bytes (Windows CRLF must not invalidate it).
        digest = hashlib.sha256(markdown.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    except (OSError, UnicodeError):
        return None
    if digest != payload.get("markdown_sha256"):
        return None
    lineage = current_fit_contract_artifact_identity(profile)
    if lineage is not None:
        # A v1 report cannot be relabeled as a v2 result, even after the v2
        # Fold has been rebuilt. Publishing v2 EVIDENCE creates a fresh report.
        if payload.get("fit_contract_version") != lineage["version"]:
            return None
        # AI: A report in outputs/ cannot prove a versioned FIT in models/.
        # Reject an unlocated FIT rather than accepting report metadata alone.
        if fit_publication_dir is None:
            return None
        state, _ = published_fit_contract_version_state(
            experiment_profile=profile, publication_dir=fit_publication_dir,
            rolling=evaluation_mode == "rolling_oos",
        )
        if state != "CURRENT":
            return None
    sop = payload.get("standard_model_sop") or {}
    if not isinstance(sop, dict) or sop.get("evaluation_mode") != evaluation_mode:
        return None
    return payload


def _single_model_rho(path: Path, markdown: Path, *, profile: str, filter_id: str,
                      versions: Mapping[str, tuple],
                      fit_publication_dir: Path | None = None) -> float | None:
    payload = _published_model_report(
        path, markdown, profile=profile, filter_id=filter_id,
        evaluation_mode="rolling_oos", versions=versions,
        fit_publication_dir=fit_publication_dir,
    )
    if payload is None:
        return None
    metrics = payload["standard_model_sop"].get("split_metrics")
    oos = metrics.get("oos") if isinstance(metrics, dict) else None
    return _number(oos.get("mean_daily_spearman")) if isinstance(oos, dict) else None


def _rolling_fit_version_current(root: Path, *, filter_id: str, profile: str) -> bool:
    """A v2+ model's published metrics need a current canonical FIT owner."""
    if current_fit_contract_artifact_identity(profile) is None:
        return True  # Preserve existing v1 publication and historical comparisons.
    state, _ = published_fit_contract_version_state(
        experiment_profile=profile,
        publication_dir=_model_fit_publication_dir(
            root, filter_id=filter_id, profile=profile, evaluation_mode="rolling_oos",
        ),
        rolling=True,
    )
    return state == "CURRENT"


def _robustness_fit_versions_current(
    root: Path, *, model: Mapping[str, Any], profile: str, seeds: tuple[int, ...],
) -> bool:
    """Check each isolated Robustness FIT, without requiring the Rolling main FIT.

    Model Robustness can legally finish before the main Rolling evaluation. Its
    Standard Comparison pins one audited model-report path per ordered seed.
    """
    paths = model.get("source_reports") or []
    if not isinstance(paths, list) or len(paths) != len(seeds):
        return False
    boundary = root.resolve()
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            return False
        report = (boundary / raw).resolve()
        if not report.is_relative_to(boundary) or not report.is_file():
            return False
        state, _ = published_fit_contract_version_state(
            experiment_profile=profile, publication_dir=report.parent, rolling=True,
        )
        if state != "CURRENT":
            return False
    return True


def _model_comparison_rhos(root: Path, *, filter_id: str, versions: Mapping[str, tuple],
                           robustness: bool):
    # Published Model Comparison is a first-class Standard SOP source for
    # both single-seed Rolling and multi-seed Robustness. In particular,
    # [3]~[6] can publish a comparison without publishing individual SOP files.
    # Never construct a multi-seed rho from a single-seed report.
    scope = "rolling_oos_robustness" if robustness else "rolling_oos"
    current_robustness_seeds = (tuple(get_robustness_benchmark_policy_snapshot(
        MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID)["resolved_seeds"])
        if robustness else ())
    path = (resolve_filter_output_dir(root, filter_id=filter_id) /
            "standard_model_comparison" / scope / "standard_model_comparison.json")
    payload = _json(path)
    result: dict[str, tuple[float, tuple[int, ...]]] = {}
    if (not payload or not path.with_suffix(".md").is_file()
            or not _approved(payload, "model.standard_comparison", versions)
            or payload.get("evaluation_mode") != "rolling_oos"
            or payload.get("robustness") is not robustness):
        payload = {}
    ambiguous: set[str] = set()
    for model in payload.get("models") or []:
        if not isinstance(model, dict) or model.get("filter_id") != filter_id:
            continue
        if robustness:
            seeds = tuple(model.get("benchmark_seeds") or ())
            if (seeds != current_robustness_seeds
                    or any(type(seed) is not int for seed in seeds)
                    or model.get("seed") is not None):
                continue
        else:
            seed = model.get("seed")
            if (type(seed) is not int or seed < 0
                    or model.get("benchmark_seeds") not in ([], None)):
                continue
            seeds = (seed,)
        view = model.get("sop_view") or {}
        if not isinstance(view, dict):
            continue
        rho = next((_number(item.get("mean_daily_spearman"))
                    for item in view.get("learnability") or []
                    if isinstance(item, dict) and item.get("split") == "oos"), None)
        if rho is None or view.get("evaluation_mode") != "rolling_oos":
            continue
        profile = str(model.get("profile") or "")
        if profile in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
            lineage = current_fit_contract_artifact_identity(profile)
            if lineage is not None:
                if model.get("fit_contract_version") != lineage["version"]:
                    continue
                if robustness:
                    if not _robustness_fit_versions_current(
                        root, model=model, profile=profile, seeds=seeds,
                    ):
                        continue
                elif not _rolling_fit_version_current(root, filter_id=filter_id, profile=profile):
                    continue
        identity = get_continuous_ranker_research_spec(profile) if profile in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES else None
        if identity and model.get("model_id") == identity.model_research_id and profile not in ambiguous:
            evidence = (rho, seeds)
            if profile in result and result[profile] != evidence:
                result.pop(profile, None)
                ambiguous.add(profile)
            else:
                result[profile] = evidence
    if robustness:
        from services.research.breakout_quality_application import load_completed_rolling_model_robustness

        rank_profiles = {
            binding["rank_model"][1]
            for group in get_research_summary_rows()
            for binding in group.get("bindings", ()) if binding.get("rank_model")
        }
        for _model_id, profile in get_breakout_quality_model_test_settings().model_profiles:
            if profile in result or profile in ambiguous or profile not in rank_profiles:
                continue
            settings = get_breakout_quality_workflow_settings(experiment_profile=profile)
            if settings.filter_id != filter_id:
                continue
            # AI: A completed model must not depend on publication of every
            # other model's comparison. Reuse the verified per-seed SOP owner,
            # including its complete-cohort aggregation, without any Producer.
            sop, _reason = load_completed_rolling_model_robustness(
                settings, project_root=root, seeds=current_robustness_seeds,
            )
            oos = ((sop or {}).get("split_metrics") or {}).get("oos") or {}
            rho = _number(oos.get("mean_daily_spearman"))
            if rho is not None:
                result[profile] = (rho, current_robustness_seeds)
    return result


def _strategy_bindings(*, target_names: tuple[str, ...],
                       score_targets: Mapping[str, str | None], definition) -> dict[tuple[str, str], str]:
    baseline = str(definition.source["baseline_arm_id"])
    base_arm = STRATEGY_COMPARE_ARMS[baseline]
    baselines = oracle_baseline_arm_ids(definition.source)
    bindings = {(control_id, ""): control_id for control_id in baselines.values()}
    comparisons = {oracle_comparison_key(item): item
                   for item in resolve_current_oracle_comparisons(definition.source)}
    positive_targets = {key for key, item in comparisons.items()
                        if item.get("positive_output")}
    # Only active Compare Suite arms can be bound to current Target x DL rows;
    # historical strategy definitions are still available for old artifacts.
    suite_id = str(STRATEGY_COMPARE_PROFILES[STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE]["suite_id"])
    active_arms = set(get_strategy_compare_suite(suite_id)["arm_ids"])
    from core.research_summary import unresolved_oracle_strategy_contracts
    unresolved = unresolved_oracle_strategy_contracts(arm_registry=STRATEGY_COMPARE_ARMS)
    candidates: dict[tuple[str, str], set[str]] = {}
    for target in target_names:
        if target == baseline or not score_targets.get(target):
            continue
        comparison = comparisons.get(target)
        if comparison is None:
            continue
        entry_strategy = str(comparison.get("strategy", "Breakout"))
        selector = str(comparison["selector"])
        # Historical Default/Positive-only is not in the current Summary.
        if selector not in {"exact", "direct"}:
            raise ValueError(f"Unsupported current Oracle selector: {selector}")
        desired_mode = _DIRECT_MODE if selector == "direct" else _EXACT_MODE
        family = oracle_parameter_family(comparison)
        reference_baseline = STRATEGY_COMPARE_ARMS[baselines[family]]
        declared_arm_id = str(comparison.get("strategy_arm_id") or "")
        # A specified Oracle arm anchors perfect-score replay, not a monopoly
        # on which DLs may learn the SAME Target. Iterate all compatible arms
        # for read-only Target x DL display, with strict downstream binding.
        if declared_arm_id and declared_arm_id not in STRATEGY_COMPARE_ARMS:
            raise ValueError(f"Oracle reference arm not registered: {declared_arm_id}")
        arm_choices = ((arm_id, STRATEGY_COMPARE_ARMS[arm_id]) for arm_id in active_arms)
        for arm_id, arm in arm_choices:
            if arm_id in unresolved:
                continue
            if (not arm.get("dl_enabled") or arm.get("dl_runtime_mode") != desired_mode
                    or str(arm.get("entry_strategy", "Breakout")) != entry_strategy
                    or arm.get("param_source") != reference_baseline["param_source"]
                    or arm.get("param_policy") != reference_baseline["param_policy"]
                    or arm.get("rule_policy") != reference_baseline["rule_policy"]):
                continue
            options = dict(arm.get("dl_runtime_options") or {})
            if requires_positive_score(options) != (target in positive_targets):
                continue
            if target in positive_targets and _positive_classifier_for_arm(
                    arm, str(score_targets[target])) is None:
                continue
            dl = STRATEGY_DL_SOURCES.get(arm.get("dl_id")) or {}
            primary_profile = str(dl.get("experiment_profile") or "")
            # An explicit Oracle reference arm defines the selector/entry
            # semantics, not proof that its fitted DL learns this Target.
            # Both ranking and Positive supervision must match the actual
            # scientific Target before [1][2] may publish model/strategy SOP
            # metrics.  In particular CO/CP event NetPnL cannot impersonate
            # a TrueDaily independently-entered NetPnL model.
            if uses_primary_positive_classifier(options):
                if (primary_profile not in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
                        or not _is_positive_classification(primary_profile, str(score_targets[target]))):
                    continue
                candidates.setdefault((target, ""), set()).add(str(arm_id))
                continue
            if (primary_profile not in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
                    or not _is_exact_supervision(primary_profile, str(score_targets[target]))):
                continue
            reporting_source = dl
            profile = str(reporting_source.get("experiment_profile") or "")
            if (profile not in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
                    or not _is_exact_supervision(profile, str(score_targets[target]))):
                continue
            candidates.setdefault((target, profile), set()).add(str(arm_id))
    # AI: Validate the Summary's explicit bindings against existing scientific
    # semantics. A compatible historical arm cannot silently join current work.
    for group in get_research_summary_rows():
        if "oracle" not in group:
            continue
        target = oracle_comparison_key(group["oracle"])
        if target not in comparisons:
            continue
        for binding in group.get("bindings", ()):
            if not binding.get("arm_id"):
                continue
            profile = str(binding["rank_model"][1])
            arm_id = str(binding["arm_id"])
            if candidates.get((target, profile)) != {arm_id}:
                raise ValueError(f"Summary Oracle/model/strategy contract differs: {target}/{arm_id}")
            bindings[(target, profile)] = arm_id

    return bindings


def _arm_fit_lineage_current(
    *, root: Path, settings: Any, arm_id: str,
    artifact_identities: Mapping[str, Any],
) -> bool:
    """Require the same FIT lineage that the replay-cache owner persists."""
    arm = settings.arms.get(str(arm_id))
    if arm is None:
        return False
    for source_id in resolve_arm_runtime_dl_source_ids(settings, arm):
        source = settings.dl_sources[source_id]
        if source.score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            continue
        lineage = current_fit_contract_artifact_identity(str(source.experiment_profile))
        if lineage is None:
            continue
        if artifact_identities.get(f"dl:{source_id}:fit_contract") != lineage:
            return False
        directory = (
            resolve_strategy_compare_selection_pit_contract_override(root=root, source=source)
            or resolve_strategy_compare_selection_pit_bundle_dir(root=root, source=source)
        )
        state, _ = published_fit_contract_version_state(
            experiment_profile=str(source.experiment_profile),
            publication_dir=directory, rolling=True,
        )
        if state != "CURRENT":
            return False
    return True


def _strategy_rolling_metrics(root: Path, versions: Mapping[str, tuple], *, required_arms=None):
    # The rolling profile is from the official user-facing Strategy Compare work type.
    settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
    base = root / settings.output_root
    files = [base / "latest" / "strategy_comparison.json"]
    files.extend(sorted((base / "runs").glob("*/strategy_comparison.json"), reverse=True))
    selected: dict[str, float] = {}
    for path in files:
        if required_arms is not None and set(required_arms).issubset(selected):
            break
        payload = _json(path)
        manifest = _json(path.with_name("manifest.json"))
        if (not payload or not manifest or not path.with_suffix(".md").is_file()
                or payload.get("status") != "COMPLETED"
                or not _approved(payload, "strategy.standard_sop", versions)
                or not _approved(manifest, "strategy.standard_sop", versions)
                or payload.get("config_fingerprint") != manifest.get("config_fingerprint")
                or payload.get("report_contract_fingerprint") != manifest.get("report_contract_fingerprint")
                or (payload.get("settings") or {}).get("profile_id") != settings.profile_id):
            continue
        expected = {str(arm.get("arm_id")) for arm in manifest.get("enabled_arms") or []
                    if isinstance(arm, dict)}
        scenarios = payload.get("scenarios") or {}
        if not isinstance(scenarios, dict):
            continue
        artifact_identities = dict(payload.get("artifact_identities") or {})
        stored_arms = dict((payload.get("settings") or {}).get("arms") or {})
        for arm_id, row in scenarios.items():
            if (arm_id in selected or arm_id not in expected or arm_id not in STRATEGY_COMPARE_ARMS
                    or (required_arms is not None and arm_id not in required_arms)):
                continue
            if not _replay_policy_current(STRATEGY_COMPARE_ARMS[arm_id], stored_arms.get(arm_id)):
                continue
            if not _arm_fit_lineage_current(
                root=root, settings=settings, arm_id=str(arm_id),
                artifact_identities=artifact_identities,
            ):
                continue
            value = _number(row.get("return_over_max_drawdown")) if isinstance(row, dict) else None
            if value is not None:
                selected[arm_id] = value
    return selected


def _replay_policy_current(current: Mapping[str, Any], published: Any) -> bool:
    if not isinstance(published, Mapping):
        return False
    try:
        current_version = int(current.get("replay_contract_version", 1))
        published_version = int(published.get("replay_contract_version", 1))
    except (TypeError, ValueError):
        return False
    return bool(
        str(current.get("entry_strategy") or "Breakout") ==
        str(published.get("entry_strategy") or "Breakout")
        and current_version == published_version
        and current_version >= 1
    )


def _robust_arm_fit_lineage_current(settings: Any, arm_id: str, pinned: Mapping[str, Any]) -> bool:
    """Robustness pins its own per-Seed FIT and Replay contract."""
    arm = settings.arms.get(arm_id)
    if arm is None or not _replay_policy_current(arm.as_dict(), pinned):
        return False
    published_sources = {
        str(source.get("dl_id") or ""): source
        for source in pinned.get("runtime_dl_sources") or []
        if isinstance(source, dict)
    }
    for source_id in resolve_arm_runtime_dl_source_ids(settings, arm):
        source = settings.dl_sources[source_id]
        lineage = current_fit_contract_artifact_identity(source.experiment_profile)
        if lineage is None:
            continue
        published = published_sources.get(source_id) or {}
        if (published.get("experiment_profile") != source.experiment_profile
                or published.get("experiment_profile_manifest") != lineage["experiment_settings"]):
            return False
    return True


def _certified_no_dl_oracle_8s(
    *, root: Path, run: Path, manifest: Mapping[str, Any], contract: Mapping[str, Any],
    oracle_info: Mapping[str, Any], arm_id: str, settings: Any,
    four_seeds: tuple[int, ...], published_four_mean: float,
) -> tuple[float, tuple[int, ...], str] | None:
    """Read-only 8S reuse only after cross-workflow, per-Seed economic parity.

    Oracle control receipts contain no hindsight scores, but are NOT granted
    automatic robustness authority. The certified first-four Strategy results,
    physical parameter SHA, return/MDD/RoMD, benchmark, period and replay arm
    must match before the additional four Oracle control seeds are consumed.
    No simulation, training, artifact copying or 4S report mutation is allowed.
    """
    from core.training_policy import benchmark_seed_execution_equivalent
    from services.research.strategy_multi_seed_robustness import (
        _validate_durable_result_artifacts,
    )

    oracle_seeds = tuple(oracle_info.get("seeds") or ())
    observations = tuple((oracle_info.get("baseline_evidence") or {}).get(arm_id) or ())
    old_benchmark = str(contract.get("benchmark_id") or "")
    new_benchmark = str(oracle_info.get("benchmark") or "")
    if (len(four_seeds) < 2 or len(oracle_seeds) <= len(four_seeds)
            or oracle_seeds[:len(four_seeds)] != four_seeds
            or len(set(oracle_seeds)) != len(oracle_seeds)
            or len(observations) != len(oracle_seeds)):
        return None
    try:
        benchmark_compatible = all(benchmark_seed_execution_equivalent(
            recorded_benchmark=old_benchmark, active_benchmark=new_benchmark,
            seed=int(seed),
        ) for seed in four_seeds)
    except ValueError:  # An unregistered historical benchmark is not reuseable.
        return None
    if not benchmark_compatible:
        return None
    period = dict(contract.get("comparison_period") or {})
    if (str(period.get("start") or "") != str(oracle_info.get("start_date") or "")
            or str(period.get("end") or "") != str(oracle_info.get("end_date") or "")
            or str(contract.get("dataset") or "") != str(settings.dataset)
            or int(contract.get("max_positions") or 0) != int(settings.max_positions)
            or str(contract.get("rotation") or "") != str(settings.rotation)):
        return None

    arm = settings.arms.get(arm_id)
    pinned_arms = [dict(item) for item in contract.get("stochastic_arms") or ()
                   if isinstance(item, Mapping) and item.get("arm_id") == arm_id]
    if arm is None or arm.dl_enabled or len(pinned_arms) != 1:
        return None
    pinned = pinned_arms[0]
    if (pinned.get("param_source") != arm.param_source
            or pinned.get("param_policy") != arm.param_policy
            or pinned.get("rule_policy") != arm.rule_policy
            or str(pinned.get("entry_strategy") or "Breakout") != str(arm.entry_strategy)
            or int(pinned.get("replay_contract_version") or 1) != int(arm.replay_contract_version)
            or bool(pinned.get("model_seed_sensitive"))):
        return None
    if any(
        (int(obs.get("seed") or -1) != int(seed)
         or str(obs.get("dataset") or "") != str(settings.dataset)
         or obs.get("max_positions") != settings.max_positions
         or str(obs.get("rotation") or "") != str(settings.rotation))
        for seed, obs in zip(oracle_seeds, observations)
    ):
        return None

    scientific = dict(manifest.get("durable_artifacts") or {})
    if not _validate_durable_result_artifacts(
        run, scientific, keys=("seed_results", "seed_yearly_results")
    ):
        return None
    with (run / "seed_results.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("arm_id") == arm_id]
    if len(rows) != len(four_seeds):
        return None
    by_seed: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            return None
        if seed in by_seed or seed not in four_seeds:
            return None
        by_seed[seed] = row
    if set(by_seed) != set(four_seeds):
        return None
    param_identities = dict(contract.get("benchmark_parameter_artifact_identities") or {})
    for obs in observations:
        values = (obs.get("romd"), obs.get("total_return_pct"), obs.get("mdd_pct"))
        try:
            if not all(math.isfinite(float(value)) for value in values):
                return None
        except (TypeError, ValueError, OverflowError):
            return None
    for seed, obs in zip(four_seeds, observations):
        row = by_seed[seed]
        registered = dict(param_identities.get(f"{arm_id}:seed={seed}") or {})
        if (not registered.get("sha256")
                or str(row.get("strategy_param_sha256") or "").lower() !=
                    str(registered["sha256"]).lower()
                or str(obs.get("param_scientific_sha256") or "").lower() !=
                    str(registered["sha256"]).lower()
                or not str(obs.get("param_sha256") or "")):
            return None
        for oracle_key, strategy_key in (
            ("romd", "return_over_max_drawdown"),
            ("total_return_pct", "total_return_pct"),
            ("mdd_pct", "max_drawdown_pct"),
        ):
            try:
                actual = float(row[strategy_key])
                expected = float(obs[oracle_key])
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if (not math.isfinite(actual)
                    or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)):
                return None
    if not math.isclose(
        fmean(float(by_seed[seed]["return_over_max_drawdown"]) for seed in four_seeds),
        published_four_mean, rel_tol=0.0, abs_tol=1e-6,
    ):
        return None
    return (fmean(float(obs["romd"]) for obs in observations),
            oracle_seeds, new_benchmark)


def _strategy_robustness_metrics(
    root: Path, versions: Mapping[str, tuple], *,
    oracle_info: Mapping[str, Any] | None = None,
    oracle_source: Mapping[str, Any] | None = None,
):
    from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS
    cfg = get_strategy_multi_seed_robustness_settings(STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS)
    current_settings = get_strategy_comparison_settings(cfg.profile_id)
    current_benchmark = get_robustness_benchmark_policy_snapshot(cfg.benchmark_id)
    active_seeds = tuple(current_benchmark["resolved_seeds"])
    base = root / cfg.output_root
    pointer = _json(base / "latest.json") or {}
    names = [str(pointer.get("fingerprint") or "")]
    names.extend(sorted((x.name for x in base.iterdir() if x.is_dir()), reverse=True) if base.is_dir() else ())
    results: dict[str, tuple[float, tuple[int, ...], str]] = {}
    for name in dict.fromkeys(names):
        run = base / name
        if not name or not run.is_dir() or run.parent != base or run.name != name:
            continue
        manifest = _json(run / "manifest.json")
        summary = _json(run / "robustness_summary.json")
        if (not manifest or not summary or not (run / "robustness_report.md").is_file()
                or manifest.get("status") != "COMPLETED"
                or summary.get("status") != "RESULT_AVAILABLE"
                or not _approved(summary, "strategy.standard_sop", versions)):
            continue
        contract = summary.get("contract") or {}
        expected = manifest.get("contract") or {}
        if (contract.get("fingerprint") != name or expected.get("fingerprint") != name
                or contract.get("profile_id") != cfg.profile_id
                or expected.get("profile_id") != cfg.profile_id
                or expected.get("benchmark_id") != contract.get("benchmark_id")):
            continue
        seeds = tuple(contract.get("resolved_seeds") or ())
        # Historical eight-seed run aggregates remain immutable evidence,
        # never a current four-seed robustness metric or a partial substitute.
        if (contract.get("benchmark_id") != current_benchmark["benchmark_id"]
                or seeds != active_seeds
                or _count(contract.get("seed_count")) != len(active_seeds)):
            continue
        # Benchmark results already pin each Seed's scientific model profile
        # manifest in its published stochastic-arm contract. Reject a pre-v2
        # report after its consumed model FIT contract was explicitly upgraded.
        pinned_arms = {
            str(row.get("arm_id") or ""): dict(row)
            for row in contract.get("stochastic_arms") or []
            if isinstance(row, dict)
        }
        for item in summary.get("romd_statistics") or []:
            if not isinstance(item, dict):
                continue
            arm_id = str(item.get("arm_id") or "")
            if not _robust_arm_fit_lineage_current(
                current_settings, arm_id, pinned_arms.get(arm_id) or {},
            ):
                continue
            if arm_id in results or arm_id not in STRATEGY_COMPARE_ARMS:
                continue
            value = _number(item.get("mean"))
            if value is not None and _count(item.get("n")) == len(seeds):
                # Aggregates are cohort-specific. Never relabel a historical
                # eight-seed Oracle statistic as current four-seed Robustness.
                resolved = (value, seeds, str(contract["benchmark_id"]))
                results[arm_id] = resolved
    return results


def sort_target_dl_overview_rows(
    rows: list[Mapping[str, Any]], *, column_key: str = DEFAULT_SORT_COLUMN,
) -> list[Mapping[str, Any]]:
    """Read-only, stable sort using the approved report column keys.

    Strategy IDs use ascending numeric order. Selector prioritizes Direct before
    Exact; other text sorts use descending lexical order. Missing values go last.
    """

    # Programmatic report sorting accepts every visible column.
    columns = (*table_contract(REPORT_ID, "comparison", "target_dl").columns,
               *target_dl_summary_sort_columns())
    column = next((item for item in columns if item.key == column_key), None)
    if column is None:
        raise ValueError(f"Summary 不支援的排序欄位: {column_key!r}")
    if column_key == "strategy":
        def sort_value(row: Mapping[str, Any]) -> tuple[bool, float, str]:
            value = str(row.get(column_key) or "").strip().casefold()
            suffix = value[1:] if value.startswith("c") else ""
            number = int(suffix) if suffix.isdecimal() else math.inf
            return (not bool(value), number, value)
    elif column_key == "selector":
        def sort_value(row: Mapping[str, Any]) -> tuple[int, str]:
            value = str(row.get(column_key) or "").strip().casefold()
            # Selector is a categorical choice, not an alphabetical metric.
            # Direct precedes Exact regardless of their Mean RoMD values.
            priority = {"direct": 2, "exact": 1}
            return (priority.get(value, 0 if value else -1), value)
    elif column.format_kind == "text":
        def sort_value(row: Mapping[str, Any]) -> tuple[bool, str]:
            value = str(row.get(column_key) or "").strip()
            return (bool(value), value.casefold())
    else:
        def sort_value(row: Mapping[str, Any]) -> tuple[bool, float]:
            value = _number(row.get(column_key))
            return (value is not None, value if value is not None else -math.inf)
    # Stable priorities: selected column, Mean RoMD DESC, Target ASC.
    # A missing primary or secondary value always follows a certified value;
    # neither missing metric nor a negative result is converted to zero.
    by_target = sorted(rows, key=lambda row: str(row.get("target") or ""))
    if column_key != "rolling_robustness_mean_romd":
        def mean_sort_value(row: Mapping[str, Any]) -> tuple[bool, float]:
            value = _number(row.get("rolling_robustness_mean_romd"))
            return (value is not None, value if value is not None else -math.inf)
        by_target = sorted(by_target, key=mean_sort_value, reverse=True)
    return sorted(by_target, key=sort_value, reverse=column_key != "strategy")


def collect_target_dl_overview(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    require_approved_report_contract(REPORT_ID)
    # Validate the presentation Target binding for both Oracle and non-Oracle
    # groups, including No-DL controls skipped by metric-collection loops.
    # A wrong display override must fail closed before reading result artifacts.
    groups = get_research_summary_rows()
    summary_model_targets = _summary_model_target_labels(groups)
    for group in groups:
        if "display_target" not in group:
            continue
        bindings = tuple(group.get("bindings") or ())
        for binding in bindings or ({"arm_id": group.get("arm_id")},):
            label = _summary_scientific_target(group, binding, summary_model_targets)
            if "oracle" in group and label != oracle_target_semantics(group["oracle"])[0]:
                raise ValueError(f"Summary display Target differs from Oracle supervised outcome: {label}")
    versions = load_report_contract_versions(root)
    definition = get_formal_oracle_definition(get_active_audit_module_id())
    suite_id = STRATEGY_COMPARE_PROFILES[STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE]["suite_id"]
    validate_oracle_compare_family_coverage(
        definition.source, active_arm_ids=get_strategy_compare_suite(suite_id)["arm_ids"],
        arm_registry=STRATEGY_COMPARE_ARMS,
    )
    targets, comparisons, score_targets, oracle, oracle_info = _oracle_rows(root, definition, versions)
    filter_id = str(definition.source["filter_id"])
    comparison_rolling = _model_comparison_rhos(
        root, filter_id=filter_id, versions=versions, robustness=False,
    )
    robustness_rho = _model_comparison_rhos(
        root, filter_id=filter_id, versions=versions, robustness=True,
    )
    bindings = _strategy_bindings(target_names=targets, score_targets=score_targets, definition=definition)
    positive_targets = {
        oracle_comparison_key(item) for item in comparisons if item.get("positive_output")
    }
    required_arms = set(bindings.values())
    for group in groups:
        if group.get("arm_id"):
            required_arms.add(str(group["arm_id"]))
        required_arms.update(str(item["arm_id"]) for item in group.get("bindings", ()) if item.get("arm_id"))
    required_arms.discard("")
    rolling_romd = _strategy_rolling_metrics(root, versions, required_arms=required_arms)
    robust_romd = _strategy_robustness_metrics(
        root, versions, oracle_info=oracle_info, oracle_source=definition.source,
    )
    model_map: dict[str, list[dict[str, Any]]] = {}
    model_evidence: dict[str, dict[str, Any]] = {}
    declared_models = {
        (oracle_comparison_key(group["oracle"]), binding["rank_model"][1])
        for group in groups if "oracle" in group
        for binding in group.get("bindings", ()) if binding.get("rank_model")
    }
    for _model_id, profile in get_breakout_quality_model_test_settings().model_profiles:
        possible_targets = [name for name, label_id in score_targets.items()
                            if (name, profile) in declared_models and label_id
                            and _is_exact_supervision(profile, label_id)]
        model = get_continuous_ranker_research_spec(profile)
        json_path, md_path = _model_paths(root, filter_id=filter_id, profile=profile)
        rolling_rho = _single_model_rho(
            json_path, md_path, profile=profile, filter_id=filter_id, versions=versions,
            fit_publication_dir=_model_fit_publication_dir(
                root, filter_id=filter_id, profile=profile, evaluation_mode="rolling_oos",
            ),
        )
        if rolling_rho is None:
            comparison = comparison_rolling.get(profile)
            rolling_rho = comparison[0] if comparison else None
        robust_rho, robust_seeds = robustness_rho.get(profile, (None, ()))
        if not _rolling_fit_version_current(root, filter_id=filter_id, profile=profile):
            # Robustness has independent per-seed FIT evidence, validated above.
            rolling_rho = None
        model_evidence[profile] = {
            "model": model.model_research_id,
            "profile": profile,
            "rolling_rho": rolling_rho,
            "robustness_rho": robust_rho,
            "rho_seeds": robust_seeds,
        }
        for target in possible_targets:
            model_map.setdefault(target, []).append(model_evidence[profile])
    # One presentation projection shared verbatim with formal Replay [2].
    # No display field participates in FIT, signed receipts or REUSE identity.
    display = oracle_work_display_dimensions(definition.source)
    rows = []

    def append_row(*, target: str, entry: str, selector: str, model: Mapping[str, Any],
                   arm: str, metrics: Mapping[str, Any], classifier: str | None = None) -> None:
        robust = robust_romd.get(arm) if arm else None
        rolling = rolling_romd.get(arm)
        robust_mean = robust[0] if robust else None
        robust_seeds = robust[1] if robust else ()
        rows.append({
            "target": target,
            "entry_strategy": entry,
            "selector": selector,
            "oracle_romd": metrics.get("oracle_romd"),
            "win_rate_pct": metrics.get("win_rate_pct"),
            "model": model.get("model") or "",
            "positive_model": (get_continuous_ranker_research_spec(classifier).model_research_id
                               if classifier else ""),
            "rolling_rho": model.get("rolling_rho"),
            "robustness_rho": model.get("robustness_rho"),
            "rho_seeds": model.get("rho_seeds") or (),
            "strategy": arm,
            "rolling_romd": rolling,
            "robustness_romd": robust_mean,
            "romd_seeds": robust_seeds,
            "romd_benchmark": robust[2] if robust else "",
            "rolling_robustness_mean_romd": weighted_rolling_robustness_romd(
                rolling, robust_mean, len(robust_seeds),
            ),
        })

    for target in targets:
        for model in model_map.get(target) or [{}]:
            profile = str(model.get("profile") or "")
            arm = bindings.get((target, profile), "")
            classifier = (
                _positive_classifier_for_arm(STRATEGY_COMPARE_ARMS[arm], str(score_targets[target]))
                if arm and target in positive_targets else None
            )
            # Unscored controls do not have hypothetical Oracle performance.
            is_baseline = target in oracle_baseline_arm_ids(definition.source).values()
            append_row(
                target=display[target][0],
                entry=display[target][1],
                selector=display[target][2], model=model, arm=arm,
                metrics={} if is_baseline else (oracle.get(target) or {}), classifier=classifier,
            )

    # AI: All configured research belongs in Summary. Some real model/strategy
    # contracts have no approved matching Oracle; never borrow another Target's
    # perfect-score result to fill these rows.
    # Oracle only emits a no-DL control while its family has an active
    # hindsight treatment.  Summary membership is independent: after a model
    # is retired, its still-enabled control remains visible as a normal
    # (unscored) strategy row.  Skip only controls already emitted above.
    emitted_oracle_controls = set(targets).intersection(
        oracle_baseline_arm_ids(definition.source).values()
    )
    for group in groups:
        if "oracle" in group:
            continue
        items = ([{"arm_id": group["arm_id"]}] if group.get("arm_id") else [])
        items.extend(group.get("bindings", ()))
        for binding in items:
            arm_id = str(binding.get("arm_id") or "")
            if arm_id in emitted_oracle_controls:
                continue
            arm = STRATEGY_COMPARE_ARMS.get(arm_id, {})
            pair = binding.get("rank_model")
            model = model_evidence.get(pair[1], {}) if pair else {}
            if binding.get("rank_models"):
                from core.rank_score_composition import rank_score_composition
                composition = rank_score_composition(arm.get("dl_runtime_options"))
                model = {"model": " + ".join(
                    f"{item[0]} pct {member['weight']:.0%}"
                    for item, member in zip(binding["rank_models"], composition["members"], strict=True)
                    if member["weight"] > 0)}
                # A deterministic runtime composition is not a newly fitted model.
                # Do not copy a constituent's rho or manufacture an averaged rho.
            # AI: A model-only Summary binding has no Strategy/Optimizer lineage.
            # Keep both strategy presentation fields blank; actual strategy arms
            # must continue to pass the canonical parameter-source verification.
            if arm_id:
                entry_label, _daily_fitted = _summary_entry_label(
                    arm, family=str(group.get("parameter_family", "min")),
                )
                selector = research_summary_selector_label(arm)
            else:
                entry_label, selector = "", ""
            # No-DL controls never have a supervised Target. Arm IDs stay in
            # the Strategy ID column and in signed Replay identities only.
            target = _summary_scientific_target(group, binding, summary_model_targets)
            if arm_id and target:
                from core.research_summary import (
                    research_summary_parameter_dimensions, research_summary_target_label,
                )
                _filter, parameter_version, _daily = research_summary_parameter_dimensions(
                    arm, parameter_sources=STRATEGY_EXPERIMENTAL_PARAM_SOURCES,
                )
                target = research_summary_target_label(target, parameter_version=parameter_version)
            equivalent_metrics = {}
            reference_id = group.get("oracle_equivalent_arm_id")
            if reference_id is not None:
                from core.positive_score_admission import perfect_positive_arms_equivalent
                references = [candidate for candidate in groups if "oracle" in candidate
                              and any(item.get("arm_id") == reference_id
                                      for item in candidate.get("bindings", ()))]
                if len(references) != 1:
                    raise ValueError("Summary perfect-output equivalence requires one registered Oracle reference")
                key = oracle_comparison_key(references[0]["oracle"])
                if (not references[0]["oracle"].get("positive_output")
                        or not perfect_positive_arms_equivalent(
                            arm, STRATEGY_COMPARE_ARMS[reference_id], sources=STRATEGY_DL_SOURCES,
                            target_id=str(score_targets.get(key) or ""),
                            resolve_recipe=get_continuous_ranker_execution_recipe)):
                    raise ValueError(f"Summary perfect-output execution differs: {arm_id}/{reference_id}")
                equivalent_metrics = oracle.get(key) or {}
            append_row(
                target=target, entry=entry_label,
                selector=selector, model=model, arm=arm_id, metrics=equivalent_metrics,
                classifier=(binding.get("positive_model") or (None, None))[1],
            )
    return {"rows": sort_target_dl_overview_rows(rows), "oracle": oracle_info,
            "source": "local_completed_reports_only"}


def render_target_dl_overview(
    report: Mapping[str, Any], *, sort_by: str = DEFAULT_SORT_COLUMN,
) -> str:
    contract = require_approved_report_contract(REPORT_ID)
    table = table_contract(REPORT_ID, "comparison", "target_dl")
    source = report["oracle"]
    header = [render_title(contract.sections[0].title)]
    header.append("Oracle: " + (
        f'{source["benchmark"]} | N={len(source["seeds"])} | {source["period"]}'
        if source["path"] else "no completed matching local report; metrics blank"
    ))
    rows = sort_target_dl_overview_rows(list(report["rows"]), column_key=sort_by)
    sort_column = next(column for column in (*table.columns, *target_dl_summary_sort_columns())
                       if column.key == sort_by)
    sort_description = (
        "策略編號 ↑ → Mean RoMD ↓（缺值最後）" if sort_by == "strategy"
        else "Mean RoMD ↓（缺值最後）" if sort_by == "rolling_robustness_mean_romd"
        else "Selector：Direct → Exact → Mean RoMD ↓（缺值最後）" if sort_by == "selector"
        else f"{sort_column.label} ↓ → Mean RoMD ↓（缺值最後）"
    )
    header.append(f"排序：{sort_description}")
    # Use the canonical per-column metric preference and best/worst signal owner.
    # Robustness is comparable only within the same actual seed set/benchmark;
    # repeated Exact/Direct rows may receive the same signal for their shared ρ.
    column_signals: dict[str, dict[str, str]] = {}
    for column in table.columns:
        if column.preference not in {"higher", "lower"}:
            continue
        cohorts: dict[tuple[Any, ...], dict[str, Any]] = {}
        for i, row in enumerate(rows):
            value = row.get(column.key)
            if value is None:
                continue
            if column.key == "robustness_rho":
                seeds = tuple(row.get("rho_seeds") or ())
                if len(seeds) < 2:
                    continue
                cohort = ("model", seeds)
            elif column.key == "robustness_romd":
                seeds = tuple(row.get("romd_seeds") or ())
                benchmark = str(row.get("romd_benchmark") or "")
                if len(seeds) < 2 or not benchmark:
                    continue
                cohort = ("strategy", benchmark, seeds)
            elif column.key == "rolling_robustness_mean_romd":
                # Do not compare a 4-seed weighted value against a rolling-only
                # value or values built from a different certified benchmark.
                if row.get("robustness_romd") is not None:
                    seeds = tuple(row.get("romd_seeds") or ())
                    benchmark = str(row.get("romd_benchmark") or "")
                    if len(seeds) < 2 or not benchmark:
                        continue
                    cohort = ("weighted", benchmark, seeds)
                else:
                    cohort = ("rolling_only",)
            else:
                cohort = ("same_column",)
            cohorts.setdefault(cohort, {})[str(i)] = value
        # Each valid evidence cohort may have different extrema.  A Summary
        # column, however, must never color intermediate visible values simply
        # because they are best/worst *within* a smaller cohort.  Preserve the
        # cohort comparability checks while suppressing extra colored pairs.
        cohort_signals = {
            row_id: signal
            for cohort_values in cohorts.values()
            for row_id, signal in best_worst_signals(
                cohort_values, preference=column.preference,
            ).items()
        }
        column_signals[column.key] = full_column_extrema_only(
            cohort_signals,
            {str(i): row.get(column.key) for i, row in enumerate(rows)},
            preference=column.preference,
        )
    formatted: list[list[str]] = []
    for i, row in enumerate(rows):
        cells = []
        for column in table.columns:
            value = row.get(column.key)
            # A missing official artifact is a visually EMPTY cell, never zero.
            cell = "" if value is None else format_contract_value(column, value)
            if value is not None:
                signal = column_signals.get(column.key, {}).get(str(i))
                if signal:
                    cell = styled_signal(cell, signal, target="console")
            if column.key in {"robustness_rho", "robustness_romd"} and value is not None:
                seeds = row["rho_seeds"] if column.key == "robustness_rho" else row["romd_seeds"]
                cell += f" ({len(seeds)}S)"
            cells.append(cell)
        formatted.append(cells)
    header.append(render_table(table.headers, formatted, alignments=table.alignments))
    return "\n".join(header)


__all__ = ["collect_target_dl_overview", "render_target_dl_overview", "sort_target_dl_overview_rows",
           "DEFAULT_SORT_COLUMN"]
