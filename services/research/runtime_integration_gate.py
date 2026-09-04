"""Read-only strategy runtime promotion gate.

The gate promotes no configuration by itself.  It verifies that the currently
configured Selection/Forward research candidate is the same exact strategy
semantics, that canonical single-run and robustness evidence still supports it,
and that current artifacts are operationally READY.  The result is persisted as
GO / NO_GO / BLOCKED evidence for a separate runtime-default change.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from config.breakout_quality import (
    BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from core.strategy_compare_policy import (
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_settings,
    get_strategy_runtime_integration_settings,
)
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
)
from core.console_report import project_relative_display_path, render_table, render_title
from core.file_integrity import canonical_json_sha256
from core.strategy_comparison import (
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    StrategyComparisonArm,
    StrategyComparisonSettings,
)
from services.research.strategy_comparison import collect_artifact_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATE_SCHEMA_VERSION = 2
RESULT_FILENAME = "runtime_integration.json"
REPORT_FILENAME = "runtime_integration.md"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON根節點必須是object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _add_check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    category: str,
    status: str,
    detail: str,
    evidence: Any = None,
) -> None:
    if status not in {"PASS", "FAIL", "BLOCKED"}:
        raise ValueError(f"未知runtime integration check status: {status}")
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "status": status,
            "detail": detail,
            "evidence": evidence,
        }
    )


def _resolve_candidate_and_baseline(
    settings: StrategyComparisonSettings,
    *,
    candidate_arm_id: str,
) -> tuple[StrategyComparisonArm, StrategyComparisonArm]:
    candidates = [
        arm
        for arm in settings.enabled_arms
        if arm.arm_id == str(candidate_arm_id)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{settings.profile_id} runtime integration candidate必須由config唯一解析："
            f"arm_id={candidate_arm_id!r}, actual={len(candidates)}"
        )
    candidate = candidates[0]
    if not candidate.dl_enabled:
        raise ValueError(
            f"{settings.profile_id} runtime integration candidate必須是DL-enabled arm: {candidate.arm_id}"
        )
    baselines = [
        arm
        for arm in settings.enabled_arms
        if not arm.dl_enabled
        and arm.param_source == candidate.param_source
        and arm.rule_policy == candidate.rule_policy
    ]
    if len(baselines) != 1:
        raise ValueError(
            f"{settings.profile_id} candidate必須恰有一個同param/rule DL-off baseline，actual={len(baselines)}"
        )
    return candidate, baselines[0]


def resolve_runtime_candidate_contract() -> dict[str, Any]:
    cfg = get_strategy_runtime_integration_settings()
    selection = get_strategy_comparison_settings(cfg.selection_profile_id)
    forward = get_strategy_comparison_settings(cfg.forward_profile_id)
    selection_candidate, selection_baseline = _resolve_candidate_and_baseline(
        selection, candidate_arm_id=cfg.selection_candidate_arm_id
    )
    forward_candidate, forward_baseline = _resolve_candidate_and_baseline(
        forward, candidate_arm_id=cfg.forward_candidate_arm_id
    )
    selection_source = selection.dl_sources[str(selection_candidate.dl_id)]
    forward_source = forward.dl_sources[str(forward_candidate.dl_id)]
    return {
        "selection": {
            "profile_id": selection.profile_id,
            "candidate_arm_id": selection_candidate.arm_id,
            "baseline_arm_id": selection_baseline.arm_id,
            "param_source": selection_candidate.param_source,
            "rule_policy": selection_candidate.rule_policy,
            "dl_source": selection_source.as_dict(),
            "dl_runtime_mode": selection_candidate.dl_runtime_mode,
            "dl_runtime_options": dict(selection_candidate.dl_runtime_options or {}),
        },
        "forward": {
            "profile_id": forward.profile_id,
            "candidate_arm_id": forward_candidate.arm_id,
            "baseline_arm_id": forward_baseline.arm_id,
            "param_source": forward_candidate.param_source,
            "rule_policy": forward_candidate.rule_policy,
            "dl_source": forward_source.as_dict(),
            "dl_runtime_mode": forward_candidate.dl_runtime_mode,
            "dl_runtime_options": dict(forward_candidate.dl_runtime_options or {}),
        },
        "runtime": {
            "filter_id": forward_source.filter_id,
            "model_architecture": forward_source.model_architecture,
            "experiment_profile": forward_source.experiment_profile,
            "score_source": forward_source.score_source,
            "param_source": forward_candidate.param_source,
            "rule_policy": forward_candidate.rule_policy,
            "dl_runtime_mode": forward_candidate.dl_runtime_mode,
            "ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
            "ranking_options": dict(forward_candidate.dl_runtime_options or {}),
        },
    }


def _semantic_checks(checks: list[dict[str, Any]], contract: dict[str, Any]) -> None:
    selection = dict(contract["selection"])
    forward = dict(contract["forward"])
    selection_source = dict(selection["dl_source"])
    forward_source = dict(forward["dl_source"])
    identity_keys = ("filter_id", "model_architecture", "experiment_profile")
    same_model = all(selection_source.get(key) == forward_source.get(key) for key in identity_keys)
    _add_check(
        checks,
        check_id="same_model_identity_across_stages",
        category="semantics",
        status="PASS" if same_model else "FAIL",
        detail="Selection與Forward必須使用同一frozen model identity。",
        evidence={key: [selection_source.get(key), forward_source.get(key)] for key in identity_keys},
    )

    expected_mode = STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL
    same_exact_mode = selection.get("dl_runtime_mode") == forward.get("dl_runtime_mode") == expected_mode
    _add_check(
        checks,
        check_id="same_exact_runtime_mode",
        category="semantics",
        status="PASS" if same_exact_mode else "FAIL",
        detail="兩階段都必須是raw-score constrained exact runtime mode。",
        evidence=[selection.get("dl_runtime_mode"), forward.get("dl_runtime_mode")],
    )

    selection_options = dict(selection.get("dl_runtime_options") or {})
    forward_options = dict(forward.get("dl_runtime_options") or {})
    expected_common = {
        "preserve_k_r0": True,
        "constrained_solver": "exact_branch_and_bound_v1",
    }
    common_ok = all(selection_options.get(k) == v and forward_options.get(k) == v for k, v in expected_common.items())
    stage_ok = selection_options.get("selection_only") is True and forward_options.get("selection_only") is False
    _add_check(
        checks,
        check_id="exact_k_r0_solver_contract",
        category="semantics",
        status="PASS" if common_ok and stage_ok else "FAIL",
        detail="必須保留K/R0並使用同一exact branch-and-bound；只有selection_only可隨研究階段切換。",
        evidence={"selection": selection_options, "forward": forward_options},
    )

    source_ok = (
        selection_source.get("score_source") == "selection_point_in_time"
        and forward_source.get("score_source") == "continuous_ranker_oos"
    )
    _add_check(
        checks,
        check_id="pit_and_forward_score_sources",
        category="anti_lookahead",
        status="PASS" if source_ok else "FAIL",
        detail="Selection只能讀PIT score；Forward只能讀frozen Forward-OOS score。",
        evidence=[selection_source.get("score_source"), forward_source.get("score_source")],
    )

    profile_name = str(forward_source.get("experiment_profile") or "")
    profile = get_breakout_quality_experiment_profile(profile_name)
    daily_ok = profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    _add_check(
        checks,
        check_id="daily_information_date_contract",
        category="anti_lookahead",
        status="PASS" if daily_ok else "FAIL",
        detail="Runtime candidate必須是daily eligible stock-day profile，盤前以最新已完成交易日作information date。",
        evidence={"experiment_profile": profile_name, "training_sample_scope": profile.training_sample_scope},
    )

    baseline_ok = selection.get("rule_policy") == forward.get("rule_policy") == "all_off"
    _add_check(
        checks,
        check_id="min_roos_all_off_strategy_contract",
        category="semantics",
        status="PASS" if baseline_ok else "FAIL",
        detail="Promotion candidate必須維持Min ROOS/all-off策略體系，不得混入Full ROOS rule policy。",
        evidence={"selection": selection.get("rule_policy"), "forward": forward.get("rule_policy")},
    )

    policy_ok = (
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL
        in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES
    )
    _add_check(
        checks,
        check_id="core_exact_ranking_policy_supported",
        category="runtime",
        status="PASS" if policy_ok else "FAIL",
        detail="Exact constrained ranking policy必須由core/buy_sort.py正式支援。",
        evidence=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    )


def _strategy_result_matches_current_pair(
    payload: dict[str, Any],
    *,
    settings: StrategyComparisonSettings,
    candidate: StrategyComparisonArm,
    baseline: StrategyComparisonArm,
) -> bool:
    if payload.get("status") != "COMPLETED":
        return False
    stored_settings = dict(payload.get("settings") or {})
    stored_arms = dict(stored_settings.get("arms") or {})
    stored_sources = dict(stored_settings.get("dl_sources") or {})
    stored_candidate = dict(stored_arms.get(candidate.arm_id) or {})
    stored_baseline = dict(stored_arms.get(baseline.arm_id) or {})
    if not stored_candidate or not stored_baseline:
        return False
    candidate_fields = (
        "param_source",
        "rule_policy",
        "dl_enabled",
        "dl_id",
        "dl_runtime_mode",
        "dl_runtime_options",
    )
    baseline_fields = ("param_source", "rule_policy", "dl_enabled", "dl_id", "dl_runtime_mode")
    current_candidate = candidate.as_dict()
    current_baseline = baseline.as_dict()
    if any(stored_candidate.get(key) != current_candidate.get(key) for key in candidate_fields):
        return False
    if any(stored_baseline.get(key) != current_baseline.get(key) for key in baseline_fields):
        return False
    dl_id = str(candidate.dl_id or "")
    current_source = settings.dl_sources[dl_id].as_dict()
    stored_source = dict(stored_sources.get(dl_id) or {})
    source_fields = (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "threshold",
        "score_source",
    )
    return all(stored_source.get(key) == current_source.get(key) for key in source_fields)


def _find_strategy_result_evidence(
    root: Path,
    settings: StrategyComparisonSettings,
    *,
    candidate: StrategyComparisonArm,
    baseline: StrategyComparisonArm,
) -> tuple[Path, dict[str, Any]]:
    search_roots = [root / settings.output_root]
    search_roots.extend(root / raw for raw in settings.reuse_output_roots)
    paths: list[Path] = []
    for output_root in search_roots:
        latest = output_root / "latest" / "strategy_comparison.json"
        if latest.is_file():
            paths.append(latest)
        if output_root.is_dir():
            paths.extend(
                sorted(
                    output_root.glob("runs/*/strategy_comparison.json"),
                    key=lambda path: path.stat().st_mtime_ns,
                    reverse=True,
                )
            )
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            payload = _read_json(path)
        except (OSError, ValueError):
            continue
        if _strategy_result_matches_current_pair(
            payload,
            settings=settings,
            candidate=candidate,
            baseline=baseline,
        ):
            return path, payload
    searched = [
        project_relative_display_path(path, project_root=root)
        for path in search_roots
    ]
    raise FileNotFoundError(
        "找不到與目前candidate/baseline pair語意一致的canonical Strategy Compare結果；"
        f"searched_roots={searched}"
    )

def _exact_certificate_status(candidate_row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate exact certificates against the days on which the exact policy was active.

    ``resource_aware_max_dl_eligible_days`` is intentionally *not* the denominator.
    It only marks the non-trivial candidate-competition subset.  The exact owner also
    certifies trivial optima (for example candidate_count <= K), which are reported as
    ``capital-utilization`` days.  Therefore certified days may legitimately exceed
    max-DL eligible days.
    """

    required_fields = (
        "resource_aware_dl_selection_days",
        "resource_aware_capital_utilization_days",
        "resource_aware_constrained_optimality_certified_days",
        "resource_aware_max_dl_order_count_violation_days",
        "resource_aware_preservation_violation_days",
    )
    missing = [key for key in required_fields if key not in candidate_row]
    if missing:
        return "BLOCKED", {"missing_fields": missing}

    try:
        dl_selection_days = int(candidate_row["resource_aware_dl_selection_days"])
        capital_utilization_days = int(candidate_row["resource_aware_capital_utilization_days"])
        certified_days = int(candidate_row["resource_aware_constrained_optimality_certified_days"])
        order_violations = int(candidate_row["resource_aware_max_dl_order_count_violation_days"])
        preservation_violations = int(candidate_row["resource_aware_preservation_violation_days"])
        max_dl_eligible_days = int(candidate_row.get("resource_aware_max_dl_eligible_days") or 0)
    except (TypeError, ValueError):
        return "BLOCKED", {
            "invalid_fields": {key: candidate_row.get(key) for key in required_fields},
        }

    certificate_required_days = dl_selection_days + capital_utilization_days
    evidence = {
        "certificate_required_days": certificate_required_days,
        "dl_selection_days": dl_selection_days,
        "capital_utilization_days": capital_utilization_days,
        "max_dl_eligible_days_diagnostic_only": max_dl_eligible_days,
        "certified_days": certified_days,
        "order_count_violation_days": order_violations,
        "preservation_violation_days": preservation_violations,
    }
    if certificate_required_days <= 0:
        return "BLOCKED", evidence
    ok = (
        certified_days == certificate_required_days
        and order_violations == 0
        and preservation_violations == 0
    )
    return ("PASS" if ok else "FAIL"), evidence


def _single_run_checks(
    checks: list[dict[str, Any]],
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    stage_label: str,
    max_selector_latency_ms: float,
) -> None:
    try:
        candidate, baseline = _resolve_candidate_and_baseline(settings)
        result_path, payload = _find_strategy_result_evidence(
            root, settings, candidate=candidate, baseline=baseline
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        _add_check(
            checks,
            check_id=f"{settings.profile_id}_single_run_evidence",
            category="evidence",
            status="BLOCKED",
            detail=f"{stage_label}缺少可驗證的canonical latest Strategy Compare結果：{exc}",
        )
        return

    _add_check(
        checks,
        check_id=f"{settings.profile_id}_result_identity",
        category="evidence",
        status="PASS",
        detail=f"{stage_label}已找到與目前candidate/baseline pair語意一致的COMPLETED canonical結果；允許active matrix瘦身後重用historical run。",
        evidence={
            "result_path": project_relative_display_path(result_path, project_root=root),
            "stored_config_fingerprint": payload.get("config_fingerprint"),
        },
    )

    scenarios = dict(payload.get("scenarios") or {})
    candidate_row = dict(scenarios.get(candidate.arm_id) or {})
    baseline_row = dict(scenarios.get(baseline.arm_id) or {})
    try:
        candidate_romd = float(candidate_row["return_over_max_drawdown"])
        baseline_romd = float(baseline_row["return_over_max_drawdown"])
    except (KeyError, TypeError, ValueError):
        _add_check(
            checks,
            check_id=f"{settings.profile_id}_romd_improves_min",
            category="evidence",
            status="BLOCKED",
            detail=f"{stage_label} latest結果缺少candidate/baseline RoMD。",
        )
    else:
        _add_check(
            checks,
            check_id=f"{settings.profile_id}_romd_improves_min",
            category="evidence",
            status="PASS" if candidate_romd > baseline_romd else "FAIL",
            detail=f"{stage_label} candidate RoMD必須高於同param/rule Min ROOS baseline。",
            evidence={"candidate": candidate_romd, "baseline": baseline_romd, "delta": candidate_romd - baseline_romd},
        )

    certificate_status, certificate_evidence = _exact_certificate_status(candidate_row)
    _add_check(
        checks,
        check_id=f"{settings.profile_id}_exact_optimality_certificate",
        category="runtime",
        status=certificate_status,
        detail=(
            f"{stage_label}每個實際進入exact selector的active day（DL-selection或capital-utilization）"
            "都必須有optimality certificate，且K/R0 preservation/order count不可違規。"
        ),
        evidence=certificate_evidence,
    )

    raw_latency = candidate_row.get("resource_aware_selector_timing_max_ms")
    try:
        max_latency = float(raw_latency)
    except (TypeError, ValueError):
        latency_status = "BLOCKED"
        latency_evidence = {"max_ms": raw_latency, "limit_ms": max_selector_latency_ms}
    else:
        latency_status = "PASS" if max_latency <= float(max_selector_latency_ms) else "FAIL"
        latency_evidence = {"max_ms": max_latency, "limit_ms": float(max_selector_latency_ms)}
    _add_check(
        checks,
        check_id=f"{settings.profile_id}_selector_latency",
        category="runtime",
        status=latency_status,
        detail=f"{stage_label} exact selector單次最慢時間不得超過config的runtime上限。",
        evidence=latency_evidence,
    )


def _robustness_checks(
    checks: list[dict[str, Any]],
    *,
    root: Path,
    robustness_id: str,
    stage_label: str,
    anchor_experiment_profile: str,
    require_strict_majority: bool,
) -> None:
    """Validate same-generated-seed strategy evidence against the current anchor.

    Current robustness profiles intentionally omit retired MR-12B controls.  Runtime
    promotion must therefore reuse the archived canonical 8-seed result that contains
    both the exact candidate and the still-active workflow anchor, rather than forcing
    an unnecessary model retrain merely to recreate a control that already exists.
    """

    from core.training_policy import resolve_robustness_benchmark_seeds
    from services.research.strategy_multi_seed_robustness import SUMMARY_FILENAME

    cfg = get_strategy_multi_seed_robustness_settings(robustness_id)
    settings = get_strategy_comparison_settings(cfg.profile_id)
    candidate, _baseline = _resolve_candidate_and_baseline(settings)
    candidate_source = settings.dl_sources[str(candidate.dl_id)]
    expected_seeds = tuple(
        resolve_robustness_benchmark_seeds(
            seed_count=cfg.seed_count,
            generator_seed=cfg.seed_generator_seed,
        )
    )
    output_root = root / cfg.output_root
    summary_paths = sorted(
        output_root.rglob(SUMMARY_FILENAME) if output_root.is_dir() else (),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )

    matched: tuple[Path, dict[str, Any], dict[str, Any], str, str] | None = None
    for summary_path in summary_paths:
        try:
            summary = _read_json(summary_path)
        except (OSError, ValueError):
            continue
        contract = dict(summary.get("contract") or {})
        if summary.get("status") != "RESULT_AVAILABLE":
            continue
        if str(contract.get("profile_id") or "") != settings.profile_id:
            continue
        if int(contract.get("seed_count") or 0) != int(cfg.seed_count):
            continue
        if int(contract.get("seed_generator_seed") or -1) != int(cfg.seed_generator_seed):
            continue
        resolved_seeds = tuple(int(value) for value in tuple(contract.get("resolved_seeds") or ()))
        if resolved_seeds != expected_seeds:
            continue

        candidate_contract = None
        anchor_contract = None
        for raw in tuple(contract.get("stochastic_arms") or ()):
            arm = dict(raw or {})
            source = dict(arm.get("dl_source") or {})
            same_strategy_universe = (
                str(arm.get("param_source") or "") == candidate.param_source
                and str(arm.get("rule_policy") or "") == candidate.rule_policy
            )
            if not same_strategy_universe:
                continue
            if (
                str(source.get("experiment_profile") or "") == candidate_source.experiment_profile
                and str(arm.get("dl_runtime_mode") or "") == str(candidate.dl_runtime_mode or "")
                and dict(arm.get("dl_runtime_options") or {}) == dict(candidate.dl_runtime_options or {})
            ):
                candidate_contract = arm
            if str(source.get("experiment_profile") or "") == str(anchor_experiment_profile):
                anchor_contract = arm
        if candidate_contract is None or anchor_contract is None:
            continue
        candidate_arm_id = str(candidate_contract.get("arm_id") or "")
        anchor_arm_id = str(anchor_contract.get("arm_id") or "")
        if not candidate_arm_id or not anchor_arm_id or candidate_arm_id == anchor_arm_id:
            continue

        paired_rows = list(summary.get("paired_comparisons") or ())
        if not paired_rows and isinstance(summary.get("romd_same_seed_comparison"), dict):
            paired_rows = [{"romd_same_seed": summary.get("romd_same_seed_comparison")}]
        same_seed = None
        for raw_pair in paired_rows:
            pair = dict(raw_pair or {})
            raw_same = pair.get("romd_same_seed")
            if not isinstance(raw_same, dict):
                continue
            candidate_pair = dict(raw_same)
            pair_ids = {
                str(candidate_pair.get("left_arm_id") or ""),
                str(candidate_pair.get("right_arm_id") or ""),
            }
            if pair_ids == {candidate_arm_id, anchor_arm_id}:
                same_seed = candidate_pair
                break
        if same_seed is None:
            continue
        matched = (summary_path, same_seed, contract, candidate_arm_id, anchor_arm_id)
        break

    if matched is None:
        _add_check(
            checks,
            check_id=f"{robustness_id}_anchor_robustness_evidence",
            category="robustness",
            status="BLOCKED",
            detail=(
                f"{stage_label}找不到同generated seeds、同Min策略universe，且同時包含目前exact candidate與"
                f"pre-promotion anchor profile={anchor_experiment_profile}的canonical robustness結果。"
            ),
            evidence=project_relative_display_path(output_root, project_root=root),
        )
        return

    summary_path, same_seed, contract, candidate_arm_id, anchor_arm_id = matched
    n = int(same_seed.get("n") or 0)
    left_id = str(same_seed.get("left_arm_id") or "")
    right_id = str(same_seed.get("right_arm_id") or "")
    raw_delta = float(same_seed.get("right_minus_left_mean") or 0.0)
    if right_id == candidate_arm_id and left_id == anchor_arm_id:
        wins = int(same_seed.get("right_gt_left_count") or 0)
        losses = int(same_seed.get("left_gt_right_count") or 0)
        delta_mean = raw_delta
    elif left_id == candidate_arm_id and right_id == anchor_arm_id:
        wins = int(same_seed.get("left_gt_right_count") or 0)
        losses = int(same_seed.get("right_gt_left_count") or 0)
        delta_mean = -raw_delta
    else:
        _add_check(
            checks,
            check_id=f"{robustness_id}_anchor_romd_majority",
            category="robustness",
            status="BLOCKED",
            detail=f"{stage_label} robustness paired comparison方向無法解析。",
        )
        return

    strict_majority = wins > (n / 2.0)
    pass_gate = (
        n == int(cfg.seed_count)
        and delta_mean > 0.0
        and (strict_majority if require_strict_majority else wins >= n / 2.0)
    )
    _add_check(
        checks,
        check_id=f"{robustness_id}_anchor_romd_majority",
        category="robustness",
        status="PASS" if pass_gate else "FAIL",
        detail=(
            f"{stage_label} exact candidate對pre-promotion anchor的same-seed RoMD必須平均為正，且candidate勝出seed數嚴格過半。"
        ),
        evidence={
            "summary_path": project_relative_display_path(summary_path, project_root=root),
            "candidate_arm_id": candidate_arm_id,
            "anchor_arm_id": anchor_arm_id,
            "workflow_anchor_profile": anchor_experiment_profile,
            "wins": wins,
            "losses": losses,
            "n": n,
            "candidate_minus_anchor_mean_romd": delta_mean,
            "seed_generator_seed": contract.get("seed_generator_seed"),
        },
    )

def _operational_artifact_checks(
    checks: list[dict[str, Any]],
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    stage_label: str,
) -> None:
    try:
        status = collect_artifact_status(project_root=root, settings=settings)
        candidate, _baseline = _resolve_candidate_and_baseline(settings)
        parameter_status = dict(
            dict(status.get("parameters") or {}).get(candidate.param_source) or {}
        )
        dl_status = dict(
            dict(status.get("dl_sources") or {}).get(str(candidate.dl_id or "")) or {}
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        _add_check(
            checks,
            check_id=f"{settings.profile_id}_current_artifacts_ready",
            category="operational",
            status="BLOCKED",
            detail=f"{stage_label}目前工件狀態無法解析：{type(exc).__name__}: {exc}",
        )
        return

    parameter_ready = bool(parameter_status.get("ready"))
    dl_ready = bool(dl_status.get("ready"))
    relevant_ready = parameter_ready and dl_ready
    _add_check(
        checks,
        check_id=f"{settings.profile_id}_current_artifacts_ready",
        category="operational",
        status="PASS" if relevant_ready else "BLOCKED",
        detail=(
            f"{stage_label}只要求promotion candidate實際使用的param與DL score/model工件READY；"
            "不得因同profile內無關的Full benchmark artifact缺失而阻擋runtime promotion。Gate不自動訓練或重建。"
        ),
        evidence={
            "candidate_arm_id": candidate.arm_id,
            "param_source": candidate.param_source,
            "parameter_ready": parameter_ready,
            "parameter_status": parameter_status.get("status"),
            "dl_id": candidate.dl_id,
            "dl_ready": dl_ready,
            "dl_status": dl_status.get("status"),
        },
    )


def _overall_decision(checks: list[dict[str, Any]]) -> str:
    if any(row["status"] == "FAIL" for row in checks):
        return "NO_GO"
    if any(row["status"] == "BLOCKED" for row in checks):
        return "BLOCKED"
    return "GO"


def collect_runtime_integration_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    cfg = get_strategy_runtime_integration_settings()
    contract = resolve_runtime_candidate_contract()
    checks: list[dict[str, Any]] = []

    if not cfg.enabled:
        _add_check(
            checks,
            check_id="gate_enabled",
            category="config",
            status="BLOCKED",
            detail="Runtime integration Gate目前在config停用。",
        )
    else:
        _add_check(
            checks,
            check_id="gate_enabled",
            category="config",
            status="PASS",
            detail="Runtime integration Gate已啟用。",
        )

    _semantic_checks(checks, contract)
    selection = get_strategy_comparison_settings(cfg.selection_profile_id)
    forward = get_strategy_comparison_settings(cfg.forward_profile_id)
    _single_run_checks(
        checks,
        root=root,
        settings=selection,
        stage_label="Selection PIT",
        max_selector_latency_ms=cfg.max_selector_latency_ms,
    )
    _single_run_checks(
        checks,
        root=root,
        settings=forward,
        stage_label="Forward-OOS",
        max_selector_latency_ms=cfg.max_selector_latency_ms,
    )
    _robustness_checks(
        checks,
        root=root,
        robustness_id=cfg.selection_robustness_id,
        stage_label="Selection PIT",
        anchor_experiment_profile=cfg.comparison_anchor_experiment_profile,
        require_strict_majority=cfg.require_strict_romd_majority,
    )
    _robustness_checks(
        checks,
        root=root,
        robustness_id=cfg.forward_robustness_id,
        stage_label="Forward-OOS",
        anchor_experiment_profile=cfg.comparison_anchor_experiment_profile,
        require_strict_majority=cfg.require_strict_romd_majority,
    )
    _operational_artifact_checks(checks, root=root, settings=selection, stage_label="Selection PIT")
    _operational_artifact_checks(checks, root=root, settings=forward, stage_label="Forward-OOS")

    decision = _overall_decision(checks)
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": decision,
        "generated_at": datetime.now().astimezone().isoformat(),
        "settings": cfg.as_dict(),
        "candidate_contract": contract,
        "current_workflow_experiment_profile": str(BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE),
        "checks": checks,
    }
    payload["fingerprint"] = canonical_json_sha256(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "settings": payload["settings"],
            "candidate_contract": contract,
            "checks": [
                {k: row.get(k) for k in ("check_id", "category", "status", "evidence")}
                for row in checks
            ],
        }
    )
    return payload


def render_runtime_integration_report(payload: dict[str, Any]) -> str:
    checks = list(payload.get("checks") or ())
    counts = {status: sum(row.get("status") == status for row in checks) for status in ("PASS", "FAIL", "BLOCKED")}
    runtime = dict(dict(payload.get("candidate_contract") or {}).get("runtime") or {})
    lines = [
        "# Runtime Promotion Gate",
        "",
        f"- Decision: **{payload.get('status')}**",
        f"- Candidate: `{runtime.get('experiment_profile')}`",
        f"- Strategy: `{runtime.get('param_source')}` / `{runtime.get('rule_policy')}` / `{runtime.get('dl_runtime_mode')}`",
        f"- Score source: `{runtime.get('score_source')}`",
        f"- Checks: PASS={counts['PASS']} / FAIL={counts['FAIL']} / BLOCKED={counts['BLOCKED']}",
        f"- Current workflow profile: `{payload.get('current_workflow_experiment_profile')}`",
        f"- Fingerprint: `{payload.get('fingerprint')}`",
        "",
        "## Checks",
        "",
        "| Category | Check | Status | Detail |",
        "|---|---|---|---|",
    ]
    for row in checks:
        detail = str(row.get("detail") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row.get('category')} | `{row.get('check_id')}` | **{row.get('status')}** | {detail} |")
    lines.extend(
        [
            "",
            "## Decision semantics",
            "",
            "- `GO`: research semantics、single-run evidence、8-seed robustness、exact certificate與目前工件全部通過；可進入獨立runtime-default變更。",
            "- `NO_GO`: 至少一個研究／策略／exact runtime必要條件失敗；不得promotion。",
            "- `BLOCKED`: 缺少或stale canonical evidence／目前工件；不得用重新解讀文件數字取代正式工件。",
            "- 本Gate不訓練模型、不重跑策略、不自動修改runtime default。",
            "",
        ]
    )
    return "\n".join(lines)


def _print_console_summary(payload: dict[str, Any]) -> None:
    print("\n" + render_title("Runtime Promotion Gate"))
    runtime = dict(dict(payload.get("candidate_contract") or {}).get("runtime") or {})
    print(f"Decision          ：{payload.get('status')}")
    print(f"Experiment Profile：{runtime.get('experiment_profile')}")
    print(f"Runtime Mode      ：{runtime.get('dl_runtime_mode')}")
    print(f"Current Workflow  ：{payload.get('current_workflow_experiment_profile')}")
    rows = [
        [row.get("category"), row.get("check_id"), row.get("status"), row.get("detail")]
        for row in payload.get("checks") or ()
    ]
    print(render_table(["類別", "檢查", "狀態", "說明"], rows))


def show_runtime_integration_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payload = collect_runtime_integration_status(project_root=project_root)
    _print_console_summary(payload)
    return payload


def run_runtime_integration_gate(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    payload = collect_runtime_integration_status(project_root=root)
    cfg = get_strategy_runtime_integration_settings()
    output_root = root / cfg.output_root
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / f"{timestamp}_{payload['fingerprint'][:12]}"
    latest_dir = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=False)
    report = render_runtime_integration_report(payload)
    _write_json(run_dir / RESULT_FILENAME, payload)
    (run_dir / REPORT_FILENAME).write_text(report, encoding="utf-8")
    latest_dir.mkdir(parents=True, exist_ok=True)
    _write_json(latest_dir / RESULT_FILENAME, payload)
    (latest_dir / REPORT_FILENAME).write_text(report, encoding="utf-8")
    _print_console_summary(payload)
    print("永久輸出：")
    print(f"- {project_relative_display_path(run_dir / RESULT_FILENAME, project_root=root)}")
    print(f"- {project_relative_display_path(run_dir / REPORT_FILENAME, project_root=root)}")
    print(f"- {project_relative_display_path(latest_dir, project_root=root)}")
    return payload


def show_latest_runtime_integration_report(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    cfg = get_strategy_runtime_integration_settings()
    result_path = root / cfg.output_root / "latest" / RESULT_FILENAME
    report_path = root / cfg.output_root / "latest" / REPORT_FILENAME
    if not result_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("尚無Runtime integration Gate最新結果")
    payload = _read_json(result_path)
    print("\n" + report_path.read_text(encoding="utf-8"))
    return payload


__all__ = [
    "collect_runtime_integration_status",
    "render_runtime_integration_report",
    "resolve_runtime_candidate_contract",
    "run_runtime_integration_gate",
    "show_latest_runtime_integration_report",
    "show_runtime_integration_status",
]
