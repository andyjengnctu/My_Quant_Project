"""Single Forward and seed-isolated Forward Robustness share one source validator.

Canonical score-store validates scientific model/manifest/report identity,
training semantics, coverage and the checkpoint SHA. This service only layers
existing FIT-settings, current provenance and Standard Model SOP evidence gates.
No fitting, scientific selection or artifact publication happens here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.file_integrity import load_json_object_or_none
from filters.breakout_quality.artifact_dependency_registry import (
    ARTIFACT_DATASET_CORE,
    ARTIFACT_CONTINUOUS_TARGET,
    collect_model_upstream_readiness,
)
from filters.breakout_quality.paths import (
    build_filter_artifact_paths_from_dir,
    resolve_filter_artifact_paths,
)
from filters.breakout_quality.ranking_score_store import (
    load_continuous_ranker_oos_contract,
    load_continuous_ranker_oos_contract_at_paths,
)
from services.breakout_quality.fitted_model_artifacts import (
    current_default_fitting_settings,
    fitted_model_settings_issues,
    forward_fitted_source_provenance_issues,
)
from services.breakout_quality.standard_model_sop import (
    standard_model_sop_completeness_issues,
)


def _forward_source_provenance_issues(
    settings, *, project_root: Path, dataset: str, max_tickers: int,
    model_dir: Path, manifest, report
) -> tuple[str, ...]:
    """Canonical current-upstream + Forward FIT/SCORE source readiness gate.

    Called for full Forward REUSE and per-seed robustness REUSE. It uses the
    upstream owner rather than re-implementing Dataset / Target freshness policy.
    Missing or unverifiable history fails closed; no implicit data rebuild here.
    """

    filter_id = str(settings.filter_id)
    model_architecture = str(settings.model_architecture)
    experiment_profile = str(settings.experiment_profile)
    try:
        upstream = collect_model_upstream_readiness(
            project_root,
            filter_id=str(filter_id),
            model_architecture=str(model_architecture),
            experiment_profile=str(experiment_profile),
            dataset=str(dataset),
            max_tickers=int(max_tickers),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return (f"upstream readiness unavailable: {type(exc).__name__}: {exc}",)
    stale = tuple(
        f"{item.artifact_type}: {item.description}"
        for item in upstream if not item.ready
    )
    if stale:
        return stale
    dataset_nodes = [item for item in upstream if item.artifact_type == ARTIFACT_DATASET_CORE]
    if len(dataset_nodes) != 1:
        return ("canonical Dataset readiness record missing or duplicate",)
    dataset_summary = load_json_object_or_none(Path(dataset_nodes[0].path))
    if not isinstance(dataset_summary, dict):
        return ("canonical Dataset summary missing or invalid",)
    target_nodes = [item for item in upstream if item.artifact_type == ARTIFACT_CONTINUOUS_TARGET]
    if len(target_nodes) > 1:
        return ("canonical Continuous Target readiness record duplicate",)
    target_manifest = None
    if target_nodes:
        target_manifest = load_json_object_or_none(Path(target_nodes[0].path))
        if not isinstance(target_manifest, dict):
            return ("canonical Continuous Target manifest missing or invalid",)
    from services.breakout_quality.price_input_source import verified_current_price_source_identity
    try:
        input_identity = verified_current_price_source_identity(
            model_architecture=str(model_architecture), project_root=project_root,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return (f"current DL raw price input unavailable: {type(exc).__name__}: {exc}",)
    if input_identity is not None:
        dataset_summary["input_price_source"] = input_identity
        dataset_summary["requires_input_price_source"] = True
    return forward_fitted_source_provenance_issues(
        model_dir=Path(model_dir),
        model_manifest=manifest,
        report=report,
        dataset_summary=dataset_summary,
        target_manifest=target_manifest,
        project_root=project_root,
    )


def load_forward_model_source(
    *, project_root: str | Path, filter_id: str, model_architecture: str,
    experiment_profile: str, expected_seed: int, dataset: str, max_tickers: int,
    model_dir_override: Path | None = None,
    research_output_dir_override: Path | None = None,
    settings=None, canonical_loader=None, canonical_path_resolver=None,
    fitting_validator=None, source_validator=None,
):
    """Return (verified canonical source, reason); isolate both paths together.

    Missing evidence cannot be inferred from a valid checkpoint. Each robust
    seed must supply a separate physical model/report pair and the exact seed.
    """
    root = Path(project_root).resolve()
    if settings is None:
        settings = SimpleNamespace(
            filter_id=filter_id, model_architecture=model_architecture,
            experiment_profile=experiment_profile, seed=expected_seed,
        )
    isolated = model_dir_override is not None or research_output_dir_override is not None
    if isolated and (model_dir_override is None or research_output_dir_override is None):
        return None, "Forward isolated model/report directory pair incomplete"
    try:
        if isolated:
            contract = load_continuous_ranker_oos_contract_at_paths(
                str(root), str(filter_id), str(model_architecture),
                str(experiment_profile),
                model_dir=Path(model_dir_override),
                research_output_dir=Path(research_output_dir_override),
            )
            model_paths = build_filter_artifact_paths_from_dir(
                filter_id=str(filter_id), model_architecture=str(model_architecture),
                experiment_profile=str(experiment_profile), model_dir=model_dir_override,
            )
        else:
            # Optional callable ports preserve historical synthetic fault
            # injection. In production they are the same canonical functions,
            # never consumer-local validators.
            if canonical_loader is None:
                contract = load_continuous_ranker_oos_contract(
                    str(root), str(filter_id), str(model_architecture),
                    str(experiment_profile), require_scores=False,
                )
            else:
                contract = canonical_loader(
                    str(root), str(filter_id), str(model_architecture),
                    str(experiment_profile), require_scores=False,
                )
            if canonical_path_resolver is None:
                model_paths = resolve_filter_artifact_paths(
                    root, str(filter_id), str(model_architecture), str(experiment_profile),
                )
            else:
                model_paths = canonical_path_resolver(
                    root, str(filter_id), str(model_architecture), str(experiment_profile),
                )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if int(contract.seed) != int(expected_seed):
        return None, (
            "training seed不一致: "
            f"artifact={int(contract.seed)}, configured={int(expected_seed)}"
        )
    fitted_args = dict(
        model_dir=model_paths.model_dir,
        report_path=Path(contract.report_path),
        expected=current_default_fitting_settings(),
    )
    fit_issues = (
        fitted_model_settings_issues(**fitted_args)
        if fitting_validator is None else fitting_validator(**fitted_args)
    )
    if fit_issues:
        return None, "Fitted model training settings stale: " + "; ".join(fit_issues[:8])
    source_args = dict(
        model_dir=model_paths.model_dir,
        manifest=contract.manifest, report=contract.report,
    )
    provenance_issues = (
        _forward_source_provenance_issues(
            settings, project_root=root, dataset=dataset, max_tickers=max_tickers,
            **source_args,
        ) if source_validator is None else source_validator(settings, **source_args)
    )
    if provenance_issues:
        return None, "Forward fitted source lineage stale: " + "; ".join(provenance_issues[:8])
    report_payload = dict(contract.report or {})
    standard = dict(report_payload.get("standard_model_sop") or {})
    if str(standard.get("evaluation_mode") or "") != "forward_oos":
        return None, "Standard SOP evaluation_mode mismatch: expected='forward_oos'"
    completeness_issues = standard_model_sop_completeness_issues(report_payload)
    if completeness_issues:
        return None, (
            "Standard SOP共通evidence不完整，需由canonical producer補建: "
            + "; ".join(completeness_issues[:8])
        )
    return contract, None
