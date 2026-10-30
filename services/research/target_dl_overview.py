"""Read-only Target x DL evidence overview from published local research reports.

No training, replay, target materialization, output writing or result backfill.
Registries define membership/identities; approved reports alone supply metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.audit_policy import get_active_audit_module_id
from core.breakout_quality_registry import (
    SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
    get_predicted_upside_context_contract,
    get_breakout_quality_experiment_profile,
    get_continuous_ranker_research_spec,
)
from core.breakout_quality_runtime import (
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    DAILY_TERMINAL_R40_TARGET_ID,
    DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID,
    DAILY_FULL_HORIZON_MAE_GATED_MFE_T40_TARGET_ID,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.breakout_quality_policy import get_breakout_quality_model_test_settings
from core.report_contract_versions import load_report_contract_versions
from core.research_report_contract import (
    format_contract_value,
    require_approved_report_contract,
    table_contract,
)
from core.console_report import render_table, render_title
from core.report_style import best_worst_signals, styled_signal
from core.strategy_compare_policy import (
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_settings,
)
from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS, STRATEGY_DL_SOURCES
from filters.breakout_quality.paths import (
    resolve_filter_point_in_time_dir,
    resolve_filter_model_output_dir,
    resolve_filter_output_dir,
)
from services.research.target_oracle_application import get_formal_oracle_definition


REPORT_ID = "research.target_dl_validation_overview"
_ORACLE_SCORE_TARGETS = {
    "mfe": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    "terminal_r": DAILY_TERMINAL_R40_TARGET_ID,
    "terminal_mae_gate": DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID,
    "mfe_mae_gate": DAILY_FULL_HORIZON_MAE_GATED_MFE_T40_TARGET_ID,
}
_EXACT_MODE = "resource-aware-continuous-score-constrained-optimal"
_DIRECT_MODE = "resource-aware-continuous-score-no-k-no-r0"


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
    comparisons = tuple(
        item for item in definition.source["oracle_comparisons"]
        if item.get("enabled", True)
    )
    baseline = str(definition.source["baseline_arm_id"])
    targets = (baseline, *(str(item["target"]) for item in comparisons))
    if len(set(targets)) != len(targets):
        raise ValueError("Oracle target identities are not unique")
    score_targets = {}
    for item in comparisons:
        raw = item
        if item.get("score_source"):
            raw = next((other for other in comparisons if other["target"] == item["score_source"]), {})
        score_targets[str(item["target"])] = _ORACLE_SCORE_TARGETS.get(raw.get("score"))
    data_path = (root / "outputs" / "audit" / Path(definition.output_subdir)
                 / "latest" / "audit.json")
    doc_path = data_path.with_name("audit.md")
    payload = _json(data_path)
    oracle_info: dict[str, Any] = {"path": None, "benchmark": "", "seeds": (), "period": ""}
    metrics: dict[str, dict[str, float | None]] = {}
    if (payload and doc_path.is_file() and payload.get("status") == "COMPLETED"
            and payload.get("read_only") is True and payload.get("hindsight_non_pit") is True
            and _approved(payload, "audit.target_oracle_romd", versions)
            and isinstance(payload.get("oracles"), list)
            and isinstance(payload.get("seeds"), list)
            and len(payload["seeds"]) >= 2
            and len(set(payload["seeds"])) == len(payload["seeds"])
            and payload.get("start_date") and payload.get("end_date")
            and payload.get("strategy_profile_id") == definition.source["strategy_profile_id"]):
        seen: set[str] = set()
        original = [dict(payload.get("baseline") or {}), *payload["oracles"]]
        for item in original:
            if not isinstance(item, dict):
                metrics.clear()
                break
            target = str(item.get("target") or "")
            if target not in targets or target in seen or _count(item.get("seeds")) != len(payload["seeds"]):
                metrics.clear()
                break
            seen.add(target)
            romd, winrate = _number(item.get("romd")), _number(item.get("win_rate_pct"))
            if romd is None or winrate is None or not 0.0 <= winrate <= 100.0:
                metrics.clear()
                break
            metrics[target] = {"oracle_romd": romd, "win_rate_pct": winrate}
        if seen == set(targets) and len(metrics) == len(targets):
            oracle_info = {
                "path": str(data_path.relative_to(root).as_posix()),
                "benchmark": str(payload.get("benchmark_id") or ""),
                "seeds": tuple(payload["seeds"]),
                "period": f'{payload["start_date"]} ~ {payload["end_date"]}',
            }
        else:
            metrics.clear()
    return targets, comparisons, score_targets, metrics, oracle_info


def _is_exact_supervision(profile: str, target_id: str) -> bool:
    # The primary supervised label must be precisely the Oracle scalar label;
    # a composite/conditional score with the same raw component is NOT equal.
    registered = get_breakout_quality_experiment_profile(profile)
    recipe = get_continuous_ranker_execution_recipe(profile)
    policy = recipe.target_policy
    objective = recipe.objective_policy
    return bool(
        registered.continuous_target_id == target_id
        and policy.component_target_id == target_id
        and policy.postprocess == CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE
        and policy.context_transform == CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE
        and objective.secondary_pair_scope == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL
        and objective.pair_target_schema in {"scalar", "scalar_with_context_weight"}
    )


def _model_paths(root: Path, *, filter_id: str, profile: str,
                 evaluation_mode: str = "rolling_oos") -> tuple[Path, Path]:
    registered = get_breakout_quality_experiment_profile(profile)
    from core.breakout_quality_policy import BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE
    architecture = str(registered.model_architecture or BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE)
    if evaluation_mode == "rolling_oos":
        directory = resolve_filter_point_in_time_dir(root, filter_id, architecture, profile)
    elif evaluation_mode == "forward_oos":
        directory = resolve_filter_model_output_dir(root, filter_id, architecture, profile)
    else:
        raise ValueError(f"Unsupported Standard Model SOP evaluation_mode: {evaluation_mode}")
    return directory / "standard_model_report.json", directory / "standard_model_report.md"


def _published_model_report(path: Path, markdown: Path, *, profile: str,
                            filter_id: str, evaluation_mode: str,
                            versions: Mapping[str, tuple]) -> dict[str, Any] | None:
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
    sop = payload.get("standard_model_sop") or {}
    if not isinstance(sop, dict) or sop.get("evaluation_mode") != evaluation_mode:
        return None
    return payload


def _single_model_rho(path: Path, markdown: Path, *, profile: str, filter_id: str,
                      versions: Mapping[str, tuple]) -> float | None:
    payload = _published_model_report(
        path, markdown, profile=profile, filter_id=filter_id,
        evaluation_mode="rolling_oos", versions=versions,
    )
    if payload is None:
        return None
    metrics = payload["standard_model_sop"].get("split_metrics")
    oos = metrics.get("oos") if isinstance(metrics, dict) else None
    return _number(oos.get("mean_daily_spearman")) if isinstance(oos, dict) else None


def _model_comparison_rhos(root: Path, *, filter_id: str, versions: Mapping[str, tuple],
                           robustness: bool):
    # Published Model Comparison is a first-class Standard SOP source for
    # both single-seed Rolling and multi-seed Robustness. In particular,
    # [3]~[6] can publish a comparison without publishing individual SOP files.
    # Never construct a multi-seed rho from a single-seed report.
    scope = "rolling_oos_robustness" if robustness else "rolling_oos"
    path = (resolve_filter_output_dir(root, filter_id=filter_id) /
            "standard_model_comparison" / scope / "standard_model_comparison.json")
    payload = _json(path)
    result: dict[str, tuple[float, tuple[int, ...]]] = {}
    if (not payload or not path.with_suffix(".md").is_file()
            or not _approved(payload, "model.standard_comparison", versions)
            or payload.get("evaluation_mode") != "rolling_oos"
            or payload.get("robustness") is not robustness):
        return result
    ambiguous: set[str] = set()
    for model in payload.get("models") or []:
        if not isinstance(model, dict) or model.get("filter_id") != filter_id:
            continue
        if robustness:
            seeds = tuple(model.get("benchmark_seeds") or ())
            if (len(seeds) < 2 or any(type(seed) is not int for seed in seeds)
                    or len(set(seeds)) != len(seeds) or model.get("seed") is not None):
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
        identity = get_continuous_ranker_research_spec(profile) if profile in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES else None
        if identity and model.get("model_id") == identity.model_research_id and profile not in ambiguous:
            evidence = (rho, seeds)
            if profile in result and result[profile] != evidence:
                result.pop(profile, None)
                ambiguous.add(profile)
            else:
                result[profile] = evidence
    return result


def _strategy_bindings(*, target_names: tuple[str, ...],
                       score_targets: Mapping[str, str | None], definition) -> dict[tuple[str, str], str]:
    baseline = str(definition.source["baseline_arm_id"])
    base_arm = STRATEGY_COMPARE_ARMS[baseline]
    bindings = {(baseline, ""): baseline}
    candidates: dict[tuple[str, str], set[str]] = {}
    for target in target_names:
        if target == baseline or not score_targets.get(target):
            continue
        selector = "direct" if target.endswith("_Direct") else "exact"
        desired_mode = _DIRECT_MODE if selector == "direct" else _EXACT_MODE
        for arm_id, arm in STRATEGY_COMPARE_ARMS.items():
            if (not arm.get("dl_enabled") or arm.get("dl_runtime_mode") != desired_mode
                    or arm.get("param_source") != base_arm["param_source"]
                    or arm.get("param_policy") != base_arm["param_policy"]
                    or arm.get("rule_policy") != base_arm["rule_policy"]):
                continue
            dl = STRATEGY_DL_SOURCES.get(arm.get("dl_id")) or {}
            profile = str(dl.get("experiment_profile") or "")
            if profile not in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
                continue
            if _is_exact_supervision(profile, str(score_targets[target])):
                candidates.setdefault((target, profile), set()).add(str(arm_id))
    # Ambiguous historical strategies must never be selected by ID order.
    bindings.update({key: next(iter(arm_ids)) for key, arm_ids in candidates.items()
                     if len(arm_ids) == 1})
    return bindings


def _strategy_rolling_metrics(root: Path, versions: Mapping[str, tuple]):
    # The rolling profile is from the official user-facing Strategy Compare work type.
    from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
    settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
    base = root / settings.output_root
    files = [base / "latest" / "strategy_comparison.json"]
    files.extend(sorted((base / "runs").glob("*/strategy_comparison.json"), reverse=True))
    selected: dict[str, float] = {}
    for path in files:
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
        for arm_id, row in scenarios.items():
            if arm_id in selected or arm_id not in expected or arm_id not in STRATEGY_COMPARE_ARMS:
                continue
            value = _number(row.get("return_over_max_drawdown")) if isinstance(row, dict) else None
            if value is not None:
                selected[arm_id] = value
    return selected


def _strategy_robustness_metrics(root: Path, versions: Mapping[str, tuple]):
    from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS
    cfg = get_strategy_multi_seed_robustness_settings(STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS)
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
        if (len(seeds) < 2 or len(seeds) != _count(contract.get("seed_count"))
                or len(set(seeds)) != len(seeds)):
            continue
        for item in summary.get("romd_statistics") or []:
            if not isinstance(item, dict):
                continue
            arm_id = str(item.get("arm_id") or "")
            if arm_id in results or arm_id not in STRATEGY_COMPARE_ARMS:
                continue
            value = _number(item.get("mean"))
            if value is not None and _count(item.get("n")) == len(seeds):
                results[arm_id] = value, seeds, str(contract.get("benchmark_id") or "")
    return results


def collect_target_dl_overview(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    require_approved_report_contract(REPORT_ID)
    versions = load_report_contract_versions(root)
    definition = get_formal_oracle_definition(get_active_audit_module_id())
    targets, comparisons, score_targets, oracle, oracle_info = _oracle_rows(root, definition, versions)
    filter_id = str(definition.source["filter_id"])
    active = {profile for _model_id, profile in get_breakout_quality_model_test_settings().model_profiles}
    comparison_rolling = _model_comparison_rhos(
        root, filter_id=filter_id, versions=versions, robustness=False,
    )
    robustness_rho = _model_comparison_rhos(
        root, filter_id=filter_id, versions=versions, robustness=True,
    )
    # The historical Strategy Registry is an existing model-identity owner.
    # It can establish a model's place in the overview, but never supply ρ or RoMD.
    historical_strategy_profiles = {
        str(source.get("experiment_profile") or "")
        for source in STRATEGY_DL_SOURCES.values()
    }
    # Historical model-composition dependencies are registered scientific
    # references too. They may appear with blank Rolling metrics when only
    # Forward evidence was published, without importing Forward rho as Rolling.
    historical_model_dependencies = {
        str(get_predicted_upside_context_contract()["stage1_profile"]),
    }
    bindings = _strategy_bindings(target_names=targets, score_targets=score_targets, definition=definition)
    rolling_romd = _strategy_rolling_metrics(root, versions)
    robust_romd = _strategy_robustness_metrics(root, versions)
    model_map: dict[str, list[dict[str, Any]]] = {}
    for profile in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
        possible_targets = [name for name, label_id in score_targets.items()
                            if label_id and _is_exact_supervision(profile, label_id)]
        if not possible_targets:
            continue
        model = get_continuous_ranker_research_spec(profile)
        json_path, md_path = _model_paths(root, filter_id=filter_id, profile=profile)
        rolling_rho = _single_model_rho(json_path, md_path, profile=profile,
                                        filter_id=filter_id, versions=versions)
        if rolling_rho is None:
            comparison = comparison_rolling.get(profile)
            rolling_rho = comparison[0] if comparison else None
        robust_rho, robust_seeds = robustness_rho.get(profile, (None, ()))
        forward_json, forward_md = _model_paths(
            root, filter_id=filter_id, profile=profile, evaluation_mode="forward_oos",
        )
        published_forward = _published_model_report(
            forward_json, forward_md, profile=profile, filter_id=filter_id,
            evaluation_mode="forward_oos", versions=versions,
        ) is not None
        if (profile not in active and rolling_rho is None and robust_rho is None
                and not published_forward and profile not in historical_strategy_profiles
                and profile not in historical_model_dependencies
                and not any((name, profile) in bindings for name in possible_targets)):
            continue
        for target in possible_targets:
            model_map.setdefault(target, []).append({
                "model": model.model_research_id,
                "profile": profile,
                "rolling_rho": rolling_rho,
                "robustness_rho": robust_rho,
                "rho_seeds": robust_seeds,
            })
    rows = []
    for target in targets:
        # The canonical scientific profile registry already provides a stable
        # chronology; keep historical model groups (e.g. K / AH / AK) together.
        models = model_map.get(target) or [{}]
        for model in models:
            profile = str(model.get("profile") or "")
            arm = bindings.get((target, profile), "")
            metrics = oracle.get(target) or {}
            robust = robust_romd.get(arm) if arm else None
            rows.append({
                "target": target,
                "oracle_romd": metrics.get("oracle_romd"),
                "win_rate_pct": metrics.get("win_rate_pct"),
                "model": model.get("model") or "",
                "rolling_rho": model.get("rolling_rho"),
                "robustness_rho": model.get("robustness_rho"),
                "rho_seeds": model.get("rho_seeds") or (),
                "strategy": arm,
                "rolling_romd": rolling_romd.get(arm),
                "robustness_romd": robust[0] if robust else None,
                "romd_seeds": robust[1] if robust else (),
                "romd_benchmark": robust[2] if robust else "",
            })
    rows.sort(key=lambda row: (
        row["oracle_romd"] is None,
        -(row["oracle_romd"] or 0.0),
        row["target"],
    ))
    return {"rows": rows, "oracle": oracle_info, "source": "local_completed_reports_only"}


def render_target_dl_overview(report: Mapping[str, Any]) -> str:
    contract = require_approved_report_contract(REPORT_ID)
    table = table_contract(REPORT_ID, "comparison", "target_dl")
    source = report["oracle"]
    header = [render_title(contract.sections[0].title)]
    header.append("Oracle: " + (
        f'{source["benchmark"]} | N={len(source["seeds"])} | {source["period"]}'
        if source["path"] else "no completed matching local report; metrics blank"
    ))
    rows = list(report["rows"])
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
            else:
                cohort = ("same_column",)
            cohorts.setdefault(cohort, {})[str(i)] = value
        column_signals[column.key] = {
            row_id: signal
            for cohort_values in cohorts.values()
            for row_id, signal in best_worst_signals(
                cohort_values, preference=column.preference,
            ).items()
        }
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
    model_seeds = sorted({len(row["rho_seeds"]) for row in rows if row["robustness_rho"] is not None})
    strategy_evidence = sorted({(row["romd_benchmark"], len(row["romd_seeds"]))
                                for row in rows if row["robustness_romd"] is not None})
    header.append("Model Robustness: " + (
        ", ".join(f"N={n}" for n in model_seeds) if model_seeds else "no validated report"
    ))
    header.append("Strategy Robustness: " + (
        ", ".join(f"{benchmark or 'unknown benchmark'} N={n}" for benchmark, n in strategy_evidence)
        if strategy_evidence else "no validated report"
    ))
    header.extend((
        "Source: Oracle audit.json / Model Standard SOP / Strategy Standard SOP (read-only)",
        "Missing/incompatible evidence = blank; Forward values never substitute Rolling.",
        "Exact/Direct share a target label but never share a strategy RoMD.",
        "Robustness values show actual seed count; separate benchmarks are not pooled.",
    ))
    return "\n".join(header)


__all__ = ["collect_target_dl_overview", "render_target_dl_overview"]
