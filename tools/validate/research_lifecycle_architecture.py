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
        fn = app_defs.get(name)
        if fn is None:
            continue  # Removing the deprecated compatibility names is permitted.
        if not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_probe_continuous_fitted_candidates"
            for node in ast.walk(fn)
        ):
            issues.append(f"Application compatibility method bypasses FIT probe: {name}")
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"exists", "is_file", "glob", "is_dir"}
            for node in ast.walk(fn)
        ):
            issues.append(f"Application contains a second artifact locator: {name}")
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
    return (
        *check_research_fit_architecture_sources(sources),
        *check_strategy_benchmark_architecture_sources(strategy_sources),
    )
