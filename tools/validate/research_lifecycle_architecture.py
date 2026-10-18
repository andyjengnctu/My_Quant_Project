"""Architecture gate for Research FIT ownership and cross-mode artifact decisions.

This is a structural gate, not an alternate runtime policy. It inspects producer
boundaries and rejects new non-owner FIT implementations before formal publication.
Its behavioral counterpart exercises the canonical owner with mutation cases.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping

FIT_OWNER = "services/breakout_quality/fitted_model_artifacts.py"
PIT_PRODUCER = "services/breakout_quality/point_in_time_scores.py"
FORWARD_PRODUCER = "services/breakout_quality/train_daily_ranker.py"
RESEARCH_APPLICATION = "services/research/breakout_quality_application.py"

# These are canonical behaviors, not scientific identities/profile names.
OWNED_FIT_BEHAVIORS = frozenset({
    "_fold_training_identity_payload",
    "_fold_training_contract_is_compatible",
    "fold_training_identity",
    "_first_fold_fit_identity_issues",
    "_forward_checkpoint_import_issues",
    "_checkpoint_matches_rescore_contract",
    "validate_rolling_fold_fitted_checkpoint",
    "select_unique_fitted_contract",
    "resolve_forward_fitted_checkpoint",
    "resolve_compatible_rolling_first_fold",
    "probe_fitted_candidates",
    "resolve_rolling_fitted_folds_root",
})



BENCHMARK_OWNER = "services/research/strategy_compare_benchmark.py"
STRATEGY_SINGLE = "services/research/strategy_comparison.py"
STRATEGY_MULTI = "services/research/strategy_multi_seed_robustness.py"


def check_strategy_benchmark_architecture_sources(
    sources: Mapping[str, str],
) -> tuple[str, ...]:
    """Disallow parallel Single/Multi 0050 decisions outside the canonical owner."""
    trees = {name: ast.parse(sources[name], filename=name)
             for name in (BENCHMARK_OWNER, STRATEGY_SINGLE, STRATEGY_MULTI)}
    issues: list[str] = []
    expected = {
        "resolve_strategy_compare_benchmark_yearly",
        "resolve_strategy_compare_robustness_benchmark_yearly",
        "resolve_strategy_robustness_report_benchmark_yearly",
        "collect_consistent_benchmark_yearly",
        "collect_strategy_pair_benchmark_yearly",
    }
    present = set(_defs(trees[BENCHMARK_OWNER]))
    for name in sorted(expected - present):
        issues.append(f"canonical benchmark owner missing: {name}")
    for module, required in (
        (STRATEGY_SINGLE, "collect_strategy_pair_benchmark_yearly"),
        (STRATEGY_MULTI, "resolve_strategy_robustness_report_benchmark_yearly"),
    ):
        imports = {
            alias.name for node in ast.walk(trees[module])
            if isinstance(node, ast.ImportFrom) and node.module == (
                "services.research.strategy_compare_benchmark"
            ) for alias in node.names
        }
        if required not in imports:
            issues.append(f"{module} bypasses canonical 0050 evidence owner: {required}")
        if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == required for node in ast.walk(trees[module])):
            issues.append(f"{module} never consumes canonical 0050 evidence owner: {required}")
        for name, fn in _defs(trees[module]).items():
            if name in expected or name in {
                "_common_fixed_benchmark_yearly", "_robustness_benchmark_yearly",
                "_strategy_benchmark_yearly",
            }:
                issues.append(f"duplicate non-owner 0050 evidence decision: {module}::{name}")
            # The canonical portfolio benchmark producer is callable only from
            # the benchmark adapter, never from either strategy orchestration.
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and
                    (isinstance(node.func, ast.Name) and node.func.id == "_build_benchmark_period_stats"
                    or isinstance(node.func, ast.Attribute) and node.func.attr == "_build_benchmark_period_stats")):
                    issues.append(f"{module} computes 0050 independently: {name}")
    return tuple(dict.fromkeys(issues))


def _defs(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in getattr(tree, "body", ())
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _owner_imports(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == (
            "services.breakout_quality.fitted_model_artifacts"
        ):
            result.update(alias.name for alias in node.names)
    return result


def check_research_fit_architecture_sources(
    sources: Mapping[str, str],
) -> tuple[str, ...]:
    """Return invariant failures for an in-memory source snapshot.

    Accepting injected in-memory source text allows independent fault injection
    without editing the checked code or its validator at runtime.
    """
    trees = {path: ast.parse(sources[path], filename=path)
             for path in (FIT_OWNER, PIT_PRODUCER, FORWARD_PRODUCER, RESEARCH_APPLICATION)}
    issues: list[str] = []
    owner_defs = set(_defs(trees[FIT_OWNER]))
    # Owner -> PIT is always a forbidden reverse dependency, including lazy
    # imports in nested functions. The dependency must be inverted via the
    # PIT-provided cohort proof port; do not exempt it from the cycle guard.
    for node in ast.walk(trees[FIT_OWNER]):
        if isinstance(node, ast.ImportFrom) and node.module == (
            "services.breakout_quality.point_in_time_scores"
        ):
            issues.append("FIT owner imports PIT producer (reverse dependency)")
        if isinstance(node, ast.Import):
            if any(alias.name == "services.breakout_quality.point_in_time_scores"
                   for alias in node.names):
                issues.append("FIT owner imports PIT producer (reverse dependency)")
    if "build_rolling_first_fold_fit_proof" not in {
        alias.name for node in ast.walk(trees[FORWARD_PRODUCER])
        if isinstance(node, ast.ImportFrom) and node.module == (
            "services.breakout_quality.point_in_time_scores"
        ) for alias in node.names
    }:
        issues.append("Forward producer does not bind PIT-native fold evidence port")
    for name in OWNED_FIT_BEHAVIORS:
        if name not in owner_defs:
            issues.append(f"FIT owner missing canonical capability: {name}")
    for path in (PIT_PRODUCER, FORWARD_PRODUCER, RESEARCH_APPLICATION):
        duplicate = set(_defs(trees[path])) & OWNED_FIT_BEHAVIORS
        for name in sorted(duplicate):
            issues.append(f"non-owner defines a second FIT policy: {path}::{name}")

    pit_imports = _owner_imports(trees[PIT_PRODUCER])
    for name in (
        "_fold_training_contract_is_compatible", "fold_training_identity",
        "_forward_checkpoint_import_issues", "_checkpoint_matches_rescore_contract",
        "validate_rolling_fold_fitted_checkpoint", "select_unique_fitted_contract",
        "resolve_rolling_fitted_folds_root",
    ):
        if name not in pit_imports:
            issues.append(f"PIT producer bypasses canonical FIT owner import: {name}")
    forward_defs = _defs(trees[FORWARD_PRODUCER])
    if "resolve_forward_fitted_checkpoint" not in _owner_imports(trees[FORWARD_PRODUCER]):
        issues.append("Forward trainer does not import canonical FIT resolution")
    if "run" not in forward_defs or not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_forward_fitted_checkpoint"
        for node in ast.walk(forward_defs.get("run", ast.Pass()))
    ):
        issues.append("Forward trainer does not invoke canonical FIT resolution")
    if not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_forward_fitted_checkpoint"
        and any(keyword.arg == "rolling_fold_proof_factory"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "build_rolling_first_fold_fit_proof"
                for keyword in node.keywords)
        for node in ast.walk(forward_defs.get("run", ast.Pass()))
    ):
        issues.append("Forward trainer bypasses native PIT fold proof binding")
    for node in ast.walk(trees[FORWARD_PRODUCER]):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "services.breakout_quality.fitted_model_artifacts",
            "services.breakout_quality.point_in_time_scores",
        }:
            for alias in node.names:
                if alias.name in {
                    "load_reusable_fitted_model_contract",
                    "load_forward_compatible_rolling_checkpoint",
                    "import_verified_rolling_forward_checkpoint",
                }:
                    issues.append(
                        f"Forward trainer imports a forbidden FIT decision API: {alias.name}"
                    )
    for node in ast.walk(trees[FORWARD_PRODUCER]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "load_reusable_fitted_model_contract", "load_forward_compatible_rolling_checkpoint",
            "import_verified_rolling_forward_checkpoint",
        }:
            issues.append("Forward trainer independently selects/verifies FIT checkpoint")

    app_defs = _defs(trees[RESEARCH_APPLICATION])
    for name in ("_has_continuous_forward_fitted_candidate", "_has_rolling_first_fold_fit_candidate"):
        if name in app_defs:
            issues.append(f"retired application FIT compatibility entry was recreated: {name}")
    if "load_forward_compatible_rolling_checkpoint" in _defs(trees[PIT_PRODUCER]):
        issues.append("retired PIT rolling FIT resolver was recreated")
    owner_resolution = _defs(trees[FIT_OWNER]).get("resolve_forward_fitted_checkpoint")
    if owner_resolution is not None and not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_compatible_rolling_first_fold"
        for node in ast.walk(owner_resolution)
    ):
        issues.append("FIT owner bypasses canonical cross-mode resolution")
    if "_probe_continuous_fitted_candidates" not in app_defs:
        issues.append("Application has no canonical FIT status boundary")
    elif not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "probe_fitted_candidates"
        for node in ast.walk(app_defs["_probe_continuous_fitted_candidates"])
    ):
        issues.append("Application FIT status does not call canonical FIT locator")

    # Semantic gate, independent of exact function names: no new application
    # helper may decide fitted readiness from a model file existence check.
    for fn_name, fn in app_defs.items():
        if fn_name in {"_load_forward_robustness_seed", "_probe_continuous_fitted_candidates"}:
            continue  # Legacy robustness report-file completeness, not FIT suitability.
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        has_path_probe = any(
            isinstance(n.func, ast.Attribute) and n.func.attr in {"is_file", "exists", "glob"}
            for n in calls
        )
        checks_model = any(
            isinstance(n, ast.Constant) and n.value == "model.pt"
            for n in ast.walk(fn)
        ) or any(
            isinstance(n, ast.Attribute) and n.attr == "model_path"
            for n in ast.walk(fn)
        )
        emits_decision = any(
            isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn)
        )
        if has_path_probe and checks_model and emits_decision:
            issues.append(f"Application duplicate file-based FIT readiness: {fn_name}")
    return tuple(dict.fromkeys(issues))


MODEL_SCORE_OWNER = "filters/breakout_quality/ranking_score_store.py"
ROLLING_MODEL_SOURCE_OWNER = "services/breakout_quality/rolling_model_source.py"
FORWARD_MODEL_SOURCE_OWNER = "services/breakout_quality/forward_model_source.py"
MODEL_EVIDENCE_OWNER = "services/breakout_quality/standard_model_sop.py"
MODEL_STAGE_OWNER = "services/breakout_quality/model_artifact_readiness.py"


def check_model_stage_readiness_architecture_sources(
    sources: Mapping[str, str],
) -> tuple[str, ...]:
    """Enforce one SOP evidence validator and one cross-mode stage policy.

    The verifier is structural, not a competing metric or readiness engine.
    Both required call sites and forbidden duplicate-policy patterns are checked.
    """
    paths = (MODEL_EVIDENCE_OWNER, MODEL_STAGE_OWNER, RESEARCH_APPLICATION,
             FORWARD_PRODUCER, PIT_PRODUCER)
    trees = {path: ast.parse(sources[path], filename=path) for path in paths}
    issues: list[str] = []
    evidence_defs = _defs(trees[MODEL_EVIDENCE_OWNER])
    stage_defs = _defs(trees[MODEL_STAGE_OWNER])
    app_defs = _defs(trees[RESEARCH_APPLICATION])
    if "standard_model_sop_completeness_issues" not in evidence_defs:
        issues.append("canonical Model SOP evidence owner is missing")
    for name in ("resolve_model_artifact_readiness", "standard_model_report_matches_source"):
        if name not in stage_defs:
            issues.append(f"canonical Model stage owner is missing: {name}")
    if "_resolve_continuous_model_readiness" not in app_defs:
        issues.append("Application does not delegate Model stage readiness")

    # Explicit ownership and direct, traceable imports: avoiding a second
    # consumer-side implementation is more important than the helper name.
    for path in (RESEARCH_APPLICATION, FORWARD_PRODUCER, PIT_PRODUCER):
        for fn_name in _defs(trees[path]):
            if fn_name in {
                "standard_model_sop_completeness_issues",
                "_standard_model_sop_completeness_issues",
                "resolve_model_artifact_readiness",
                "standard_model_report_matches_source",
            }:
                issues.append(f"non-owner duplicates Model readiness logic: {path}::{fn_name}")
    imported_eval = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "services.breakout_quality.standard_model_sop"
        and any(alias.name == "standard_model_sop_completeness_issues" for alias in n.names)
        for n in ast.walk(trees[RESEARCH_APPLICATION])
    )
    if not imported_eval:
        issues.append("Application bypasses Model SOP evidence owner")
    imported_stage = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "services.breakout_quality.model_artifact_readiness"
        and any(alias.name == "resolve_model_artifact_readiness" for alias in n.names)
        for n in ast.walk(trees[RESEARCH_APPLICATION])
    )
    if not imported_stage:
        issues.append("Application bypasses canonical Model stage owner")
    stage_owner_calls = [
        n for n in ast.walk(trees[MODEL_STAGE_OWNER])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    if not any(n.func.id == "standard_model_sop_completeness_issues" for n in stage_owner_calls):
        issues.append("Model stage owner does not consume canonical SOP evidence")
    if not any(n.func.id == "ResearchArtifactPlan" or n.func.id == "ResearchArtifactAction"
               for n in stage_owner_calls):
        issues.append("Model stage owner bypasses generic Research DAG")
    resolver = app_defs.get("_resolve_continuous_model_readiness")
    if resolver is not None and not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "resolve_model_artifact_readiness"
        for n in ast.walk(resolver)
    ):
        issues.append("Application Model status bypasses canonical stage resolver")
    for entry in ("_run_continuous_forward_model_gate",
                  "_run_continuous_rolling_mode_direct", "_run_configured_model_comparison"):
        fn = app_defs.get(entry)
        if fn is None or not any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_resolve_continuous_model_readiness"
            for n in ast.walk(fn)
        ):
            issues.append(f"{entry} does not consume shared Model stage plan")

    # The canonical stage owner plans every benchmark seed and its aggregate
    # REPORT. A future Application-local tally or fake SCORE proof can silently
    # diverge between Forward and Rolling robustness.
    robustness = app_defs.get("_run_configured_model_robustness")
    if "resolve_model_robustness_readiness" not in stage_defs:
        issues.append("canonical Model Robustness stage resolver is missing")
    if robustness is None:
        issues.append("Application Model Robustness entry is missing")
    else:
        robust_calls = {
            node.func.id for node in ast.walk(robustness)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if "resolve_model_robustness_readiness" not in robust_calls:
            issues.append("Model Robustness bypasses canonical multi-seed stage resolver")
        if "resolve_model_artifact_readiness" in robust_calls or "object" in robust_calls:
            issues.append("Model Robustness recreates local stage or placeholder SCORE source")
    # Semantic duplicate: another application helper evaluating truth-schema
    # identity and M4 numeric completeness must never become a new owner.
    for name, fn in app_defs.items():
        tokens = {
            n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        if "joint_quality_evaluation_id" in tokens and "mfe_pct_mean" in tokens:
            issues.append(f"Application independently validates M4 evidence: {name}")
        if "standard_model_report.json" in tokens and "report_contract_fingerprint" in tokens:
            # Publication is allowed; separate publication *readiness* is not.
            if name != "_write_standard_model_sop_report":
                issues.append(f"Application independently decides report readiness: {name}")
    return tuple(dict.fromkeys(issues))


def check_model_score_owner_architecture_sources(
    sources: Mapping[str, str],
) -> tuple[str, ...]:
    """Prevent SCORE/Audit coupling and another app-level score validator.

    This is a structural dependency gate; only the canonical SCORE owner
    computes hash/lineage/schema validity. Audit may consume SCORE, but
    Score readiness must never require an Audit JSON.
    """
    trees = {path: ast.parse(sources[path], filename=path)
             for path in (MODEL_SCORE_OWNER, ROLLING_MODEL_SOURCE_OWNER,
                          RESEARCH_APPLICATION)}
    owner_defs = _defs(trees[MODEL_SCORE_OWNER])
    selector_defs = _defs(trees[ROLLING_MODEL_SOURCE_OWNER])
    app_defs = _defs(trees[RESEARCH_APPLICATION])
    issues: list[str] = []
    source_name = "load_selection_point_in_time_score_source_contract"
    complete_name = "load_selection_point_in_time_ranking_contract"
    for name in (source_name, complete_name):
        if name not in owner_defs:
            issues.append(f"canonical PIT SCORE owner missing: {name}")
    if complete_name in owner_defs and not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == source_name
        for n in ast.walk(owner_defs[complete_name])
    ):
        issues.append("full PIT Audit contract bypasses canonical SCORE source")
    if source_name in owner_defs:
        score_owner = owner_defs[source_name]
        if any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == complete_name
            for n in ast.walk(score_owner)
        ):
            issues.append("canonical SCORE owner depends on downstream Audit")
        tokens = {n.value for n in ast.walk(score_owner)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if "Selection PIT audit" in tokens:
            issues.append("SCORE source directly validates Audit")
    # The producer validates both scientific sources.  The *single* shared
    # selector owns complete-Audit versus verified-SCORE fallback.  Standard
    # and Robustness must not independently build their own fallback policy.
    selector = selector_defs.get("load_rolling_model_source")
    if selector is None:
        issues.append("canonical Rolling SCORE/Audit source selector missing")
    else:
        selector_calls = {n.func.id for n in ast.walk(selector)
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for callee in (complete_name, source_name):
            if callee not in selector_calls:
                issues.append(f"shared Rolling source selector bypasses canonical owner: {callee}")
        for node in ast.walk(selector):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"build_file_manifest", "compute_file_sha256"}):
                issues.append("shared Rolling source selector duplicates SCORE hash validation")
    required_calls = {
        "_load_reusable_rolling_standard_report": ("load_rolling_model_source",),
        "_rolling_robustness_score_core_contract": (source_name,),
        "_rolling_robustness_score_core_ready": ("_rolling_robustness_score_core_contract",),
        "_load_rolling_robustness_seed": ("load_rolling_model_source",),
    }
    for fn_name, called in required_calls.items():
        fn = app_defs.get(fn_name)
        if fn is None:
            issues.append(f"Rolling owner consumer missing: {fn_name}")
            continue
        observed = {n.func.id for n in ast.walk(fn)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for callee in called:
            if callee not in observed:
                issues.append(f"{fn_name} bypasses canonical SCORE lifecycle: {callee}")
        if fn_name in {"_load_reusable_rolling_standard_report", "_load_rolling_robustness_seed"}:
            if observed & {source_name, complete_name}:
                issues.append(f"{fn_name} reintroduces a consumer-owned SCORE/Audit fallback")
        if any(n in observed for n in ("build_file_manifest", "compute_file_sha256")):
            issues.append(f"{fn_name} implements duplicate score checksum policy")
    # Semantic rejection independent of helper name. A newly added source
    # checker may not infer Score READY by checking score file presence/hash.
    for fn_name, fn in app_defs.items():
        calls = {n.func.id for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        tokens = {n.value for n in ast.walk(fn)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if (calls & {"build_file_manifest", "compute_file_sha256"}
            and ("score_columns" in tokens or "scores" in tokens)):
            issues.append(f"Application independently validates PIT SCORE: {fn_name}")
    return tuple(dict.fromkeys(issues))



def check_model_forward_source_architecture_sources(
    sources: Mapping[str, str],
) -> tuple[str, ...]:
    """Single and robustness Forward must share one full contract validator."""
    paths = (MODEL_SCORE_OWNER, FORWARD_MODEL_SOURCE_OWNER, RESEARCH_APPLICATION)
    trees = {path: ast.parse(sources[path], filename=path) for path in paths}
    owner_defs = _defs(trees[FORWARD_MODEL_SOURCE_OWNER])
    store_defs = _defs(trees[MODEL_SCORE_OWNER])
    app_defs = _defs(trees[RESEARCH_APPLICATION])
    issues: list[str] = []
    canonical = "load_continuous_ranker_oos_contract"
    isolated = "load_continuous_ranker_oos_contract_at_paths"
    shared = "load_forward_model_source"
    if not all(name in store_defs for name in (canonical, isolated)):
        issues.append("canonical Forward SCORE owner missing source/path capability")
    elif "_load_continuous_ranker_oos_contract_uncached" not in {
        node.func.id for node in ast.walk(store_defs[isolated])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }:
        issues.append("isolated Forward loader bypasses canonical validation")
    forward_source = owner_defs.get(shared)
    if forward_source is None:
        issues.append("canonical Forward model source owner missing")
    else:
        calls = {node.func.id for node in ast.walk(forward_source)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        for required in (
            canonical, isolated, "fitted_model_settings_issues",
            "_forward_source_provenance_issues", "standard_model_sop_completeness_issues",
        ):
            if required not in calls:
                issues.append(f"Forward source owner skips required gate: {required}")
    for fn_name in ("_load_reusable_continuous_forward_contract", "_load_forward_robustness_seed"):
        fn = app_defs.get(fn_name)
        if fn is None:
            issues.append(f"Forward Application source adapter missing: {fn_name}")
            continue
        calls = {node.func.id for node in ast.walk(fn)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        if shared not in calls:
            issues.append(f"Forward Application bypasses canonical owner: {fn_name}")
        if calls & {
            canonical, isolated, "fitted_model_settings_issues",
            "forward_fitted_source_provenance_issues", "_forward_source_provenance_issues",
            "load_json_object_or_none", "compute_file_sha256",
        }:
            issues.append(f"Forward Application independently validates source: {fn_name}")
        if any(isinstance(node, ast.Attribute) and node.attr in {"is_file", "exists"}
               for node in ast.walk(fn)):
            issues.append(f"Forward Application reintroduces file-presence REUSE: {fn_name}")
    if "_forward_source_provenance_issues" in app_defs:
        issues.append("Application recreates canonical Forward lineage owner")
    return tuple(dict.fromkeys(issues))


def check_model_standard_comparison_audit_boundary_sources(
    sources: Mapping[str, str],
) -> tuple[str, ...]:
    """Standard M1-M4 must not execute hypothesis-specific Audit producers.

    The report contract and one-off Audit implementations are independent.
    This structural gate checks the canonical Research Application entry, not
    the historical Audit libraries or their separately maintained tests.
    """
    tree = ast.parse(sources[RESEARCH_APPLICATION], filename=RESEARCH_APPLICATION)
    issues: list[str] = []
    entry = _defs(tree).get("_run_configured_model_comparison")
    if entry is None:
        return ("canonical Standard Model Comparison entry missing",)
    # No model-specific comparison library should be imported into the main
    # Research Application merely to enrich a Standard SOP. Explicit Audit
    # entrypoints remain free to import these same libraries independently.
    def is_one_off_comparison_module(module: str | None) -> bool:
        # Recognize a class of producer modules, not a fixed roster of models
        # or study identities. Future one-off hypotheses get the same boundary.
        return bool(module and module.startswith("services.breakout_quality.")
                    and module.endswith("_comparison"))

    imported_symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and is_one_off_comparison_module(node.module):
            imported_symbols.update(alias.asname or alias.name for alias in node.names)
            issues.append("Standard Research Application imports a one-off Model Audit producer")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if is_one_off_comparison_module(alias.name):
                    imported_symbols.add(alias.asname or alias.name)
                    issues.append("Standard Research Application imports a one-off Model Audit module")
    for call in (node for node in ast.walk(entry) if isinstance(node, ast.Call)):
        name = call.func.id if isinstance(call.func, ast.Name) else (
            call.func.attr if isinstance(call.func, ast.Attribute) else ""
        )
        if name in imported_symbols or (name.startswith("attach_") and name.endswith("_extension")):
            issues.append(f"Standard Model Comparison invokes one-off Audit: {name}")
    return tuple(dict.fromkeys(issues))

def check_research_fit_architecture(project_root: Path) -> tuple[str, ...]:
    root = Path(project_root)
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (FIT_OWNER, PIT_PRODUCER, FORWARD_PRODUCER, RESEARCH_APPLICATION)
    }
    strategy_sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (BENCHMARK_OWNER, STRATEGY_SINGLE, STRATEGY_MULTI)
    }
    model_sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (MODEL_EVIDENCE_OWNER, MODEL_STAGE_OWNER, MODEL_SCORE_OWNER,
                     ROLLING_MODEL_SOURCE_OWNER, FORWARD_MODEL_SOURCE_OWNER)
    }
    return (
        *check_research_fit_architecture_sources(sources),
        *check_strategy_benchmark_architecture_sources(strategy_sources),
        *check_model_stage_readiness_architecture_sources({**sources, **model_sources}),
        *check_model_score_owner_architecture_sources({**sources, **model_sources}),
        *check_model_forward_source_architecture_sources({**sources, **model_sources}),
        *check_model_standard_comparison_audit_boundary_sources(sources),
    )
