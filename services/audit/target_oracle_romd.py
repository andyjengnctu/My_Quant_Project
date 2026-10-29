"""Reusable hindsight target economics Audit; NEVER a legal prediction source.

This audit compares future-outcome stock-day rankings by running the same frozen
Broker-Cash P16 portfolio replay (Exact K/R0 or canonical No-K/No-R0). All truths are *ex post*.
No model training, PIT score publication, optimizer, or trading state is touched.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping
import math
import hashlib
import struct
import inspect
from contextlib import ExitStack
import json

import numpy as np
import pandas as pd

from core.breakout_quality_registry import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
    DAILY_TERMINAL_R40_TARGET_ID,
    DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID,
    DAILY_FULL_HORIZON_MAE_GATED_MFE_T40_TARGET_ID,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.console_report import render_table, render_title
from core.report_style import styled_workflow_status
from core.dataset_profiles import get_dataset_dir
from core.path_utils import project_relative_display_path
from core.strategy_compare_policy import (
    get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
)
from core.training_policy import benchmark_seed_execution_equivalent
from filters.breakout_quality.daily_trade40_target import HORIZON_SESSIONS
from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from filters.breakout_quality.continuous_target import (
    StrategyAlignedContinuousTargetSpec,
    full_horizon_mae_safety_r_from_minimum_low_return,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.runtime import audit_oracle_truth_score_context
from filters.breakout_quality.strategy_compare_contracts import COMPARISON_MODE_SCORE_RANKING
from filters.breakout_quality.strategy_compare_reporting import _scenario_summary, _yearly_frame, _to_json_native
from filters.breakout_quality.strategy_compare_sources import (
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    _build_controlled_param_source_pair,
    _validate_requested_param_policy,
    load_strategy_param_evaluation_view,
)
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES
from services.audit.reusable_report import persist_reusable_report
from services.audit.oracle_score_artifacts import (
    OraclePriceScoreStore, price_score_source_identity, price_score_replay_lineage,
    oracle_dataset_summary_sha256,
)
from config.audit import AUDIT_OUTPUT_ROOT
from services.audit.oracle_replay_artifacts import (
    OracleReplayStore, price_inventory_sha256, replay_identity,
)
from services.optimizer.strategy_param_service import (
    resolve_fixed_multiseed_robustness_strategy_parameter_artifact,
)
from config.execution_policy import (
    DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT, EXECUTION_POLICY_PARAM_SPECS,
)
from config.broker_accounting import (
    BROKER_ACCOUNTING_POLICY_ID, BROKER_FEE_RATE, BROKER_MIN_FEE_TWD,
    BROKER_ODD_LOT_MIN_FEE_TWD, BROKER_BOARD_LOT_SHARES,
    BROKER_STOCK_TAX_RATE, BROKER_CHARGE_ROUNDING_MODE,
)
from config.breakout_quality import BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.rolling_oos_params import get_active_param_date_range
from services.research.strategy_compare_replay import _run_scenario

from core.research_report_contract import (
    report_contract, require_approved_report_contract,
    persistent_report_contract_fingerprint,
)
from services.audit.reusable_report import format_comparison_rows
from services.audit.oracle_execution_policy import (
    execution_gate_policies, OracleExecutionPolicyError,
)

SUPPORTED_AUDIT_TYPE = "target_ground_truth_strategy_romd"
ORACLE_REPORT_SCHEMA_VERSION = 8
# Engineering bug-fix marker: older price-universe / pre-PIT-start replays are invalid.
# This is not a new scientific Target, Strategy arm, Report version or research choice.
ORACLE_PARITY_IMPLEMENTATION = "original_oracle_contract_parity_bugfix_20261027_full_research_cutoff_v4"
PERIOD_STATS_IMPLEMENTATION = "full_year_calendar_verified_20261026"
# Resolve comparison membership from the Audit definition on every run.
from core.audit_registry import AUDIT_REUSABLE_REPORTS

_ORACLE_CATALOG_SOURCE = AUDIT_REUSABLE_REPORTS["breakout_quality"]["target_oracle_romd"]["source"]
_ORACLE_DEFAULT_COMPARISONS = _ORACLE_CATALOG_SOURCE["oracle_comparisons"]
_ORACLE_RETIRED_TARGETS = frozenset(_ORACLE_CATALOG_SOURCE["retired_oracle_targets"])

# Recipe implementation is stable; WHICH recipes and providers are needed is
# resolved from the selected comparisons, never from a second target allowlist.
_ORACLE_SCORE_RECIPES = frozenset(("equal_rank_mfe_safety", "mfe", "terminal_r", "terminal_mae_gate", "mfe_mae_gate"))
_ORACLE_DAILY_SCORE_RECIPES = {"daily_trade_path_net_r": "net_r", "daily_trade_path_net_pnl": "net_pnl"}
_PROVIDER_COHORT_IDS = {
    "pure_mfe40_provider": "40d", "mfe20_d20_provider": "20d",
    "first_breach_provider": "first_breach",
}


# PRICE SOURCE IDENTITY BOUNDARY: this source-only resolver is fingerprinted
# by _price_score_algorithm_sha256(). Never add target execution/admission or
# reporting logic here. Execution-only policies belong to the separate
# services.audit.oracle_execution_policy owner, which receives this plan.

def _oracle_comparison_plan(definition) -> dict[str, Any]:
    """Canonical compare inventory -> score requirements -> provider graph."""
    # The registered daily-truth percentile is the pre-existing hypothetical
    # perfect-score contract, not a claim about raw pairwise logits.
    if (str(definition.dimensions.get("oracle_score_method"))
            != "canonical_daily_truth_percentile_v2"
            or str(definition.dimensions.get("percentile_method"))
            != "average_zero_based"):
        raise OracleBlockedError(
            "CONTRACT_UNRESOLVED: Oracle perfect Target-to-Score mapping must "
            "follow the original canonical daily-percentile contract"
        )
    declared = tuple(definition.source.get("oracle_comparisons") or ())
    selected = tuple(item for item in declared
                     if not isinstance(item, Mapping) or item.get("enabled", True))
    if not selected:
        raise OracleBlockedError("Oracle comparison清單為空；請由正式Registry設定比較對象")
    definitions: list[tuple[str, str]] = []
    recipes: dict[str, tuple[str, str]] = {}
    daily_recipes: dict[str, tuple[str, str, int | None]] = {}
    aliases: dict[str, str] = {}
    selectors: dict[str, str] = {}
    provider_fields: list[str] = []
    for entry in selected:
        if not isinstance(entry, Mapping):
            raise OracleBlockedError("Oracle comparison設定必須是mapping")
        name, desc = str(entry.get("target") or ""), str(entry.get("description") or "")
        selector = str(entry.get("selector") or "")
        if not name or not desc or name in selectors or name == str(definition.source["baseline_arm_id"]):
            raise OracleBlockedError(f"Oracle comparison name缺少、重複或非法：{name}")
        # Historical identities are owned by the canonical Audit Registry.
        # Removing a row does not free its name for a different recipe.
        if name in _ORACLE_RETIRED_TARGETS or str(entry.get("score_source") or "") in _ORACLE_RETIRED_TARGETS:
            raise OracleBlockedError(f"Oracle retired Target identity不得重新啟用或改作其他score：{name}")
        if selector not in {"exact", "direct"}:
            raise OracleBlockedError(f"Oracle comparison selector未支援：{name} {selector}")
        # Scientific target names are immutable. A previous name cannot be
        # reused for a different recipe or score source by changing the list.
        historical = next((item for item in _ORACLE_DEFAULT_COMPARISONS
                           if item["target"] == name), None)
        if historical is not None and (
                any(str(entry.get(key) or "") != str(historical.get(key) or "")
                    for key in ("selector", "score", "provider", "score_source"))
                or entry.get("horizon_sessions", HORIZON_SESSIONS) !=
                   historical.get("horizon_sessions", HORIZON_SESSIONS)
                or entry.get("horizon_mode", "fixed_sessions") !=
                   historical.get("horizon_mode", "fixed_sessions")) :
            raise OracleBlockedError(
                f"Oracle {name} 的score/selector身分已變更；須建立新Target科學ID，禁止沿用舊快取"
            )
        if entry.get("score_source"):
            source = str(entry["score_source"])
            if selector != "direct" or source in _ORACLE_RETIRED_TARGETS:
                raise OracleBlockedError(f"Oracle Direct score來源非法：{name} -> {source}")
            # A Direct arm consumes an existing approved score, but need not
            # display or replay the corresponding Exact arm. Resolve any
            # unselected source after processing the complete comparison list.
            aliases[name] = source
        else:
            recipe, provider = str(entry.get("score") or ""), str(entry.get("provider") or "")
            if not definition.source.get(provider):
                raise OracleBlockedError(f"Oracle {name} score/provider缺少正式定義：{recipe}/{provider}")
            if selector != "exact":
                raise OracleBlockedError(f"Oracle {name} 的新分數必須使用Exact；Direct僅能重用score來源")
            if recipe in _ORACLE_DAILY_SCORE_RECIPES:
                mode = entry.get("horizon_mode", "fixed_sessions")
                horizon = (None if mode == "full_trade_cycle" else
                           entry.get("horizon_sessions", HORIZON_SESSIONS))
                if mode not in {"fixed_sessions", "full_trade_cycle"} or (
                    mode == "full_trade_cycle" and "horizon_sessions" in entry
                ) or (horizon is not None and (
                    not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
                )):
                    raise OracleBlockedError(f"Oracle {name} invalid session horizon/mode")
                if provider != "daily_trade40_profile":
                    raise OracleBlockedError(f"Oracle {name} Daily-universal provider is not canonical: {provider}")
                daily_recipes[name] = (recipe, provider, horizon)
            elif recipe in _ORACLE_SCORE_RECIPES:
                recipes[name] = (recipe, provider)
                if provider not in provider_fields:
                    provider_fields.append(provider)
            else:
                raise OracleBlockedError(f"Oracle未授權的score recipe：{recipe}")
        definitions.append((name, desc))
        selectors[name] = selector
    # Direct-only and Direct-before-Exact compare lists must not require a
    # hidden Exact *Replay*. Their input truth is still derived from the same
    # immutable Exact score recipe registered by the canonical Audit owner.
    for direct_name, source_name in aliases.items():
        if source_name in recipes or source_name in daily_recipes:
            continue
        source_entry = next(
            (item for item in _ORACLE_DEFAULT_COMPARISONS
             if item["target"] == source_name), None,
        )
        if (source_entry is None or source_name in _ORACLE_RETIRED_TARGETS
                or source_entry.get("selector") != "exact"
                or source_entry.get("score_source")):
            raise OracleBlockedError(
                f"Oracle Direct score缺少canonical來源：{direct_name} -> {source_name}"
            )
        recipe = str(source_entry.get("score") or "")
        provider = str(source_entry.get("provider") or "")
        if not definition.source.get(provider):
            raise OracleBlockedError(f"Oracle Direct score來源缺少provider：{source_name}")
        if recipe in _ORACLE_DAILY_SCORE_RECIPES:
            if provider != "daily_trade40_profile":
                raise OracleBlockedError(f"Oracle Direct daily來源非法：{source_name}")
            horizon = (None if source_entry.get("horizon_mode") == "full_trade_cycle" else
                       source_entry.get("horizon_sessions", HORIZON_SESSIONS))
            if horizon is not None and (
                not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
            ):
                raise OracleBlockedError(f"Oracle Direct {source_name} horizon invalid")
            daily_recipes[source_name] = (recipe, provider, horizon)
        elif recipe in _ORACLE_SCORE_RECIPES:
            recipes[source_name] = (recipe, provider)
            if provider not in provider_fields:
                provider_fields.append(provider)
        else:
            raise OracleBlockedError(f"Oracle Direct使用未授權的score recipe：{recipe}")
    daily_sources = {name: name for name in daily_recipes if name in selectors}
    daily_sources.update({name: source for name, source in aliases.items()
                          if source in daily_recipes})
    if not recipes and not daily_recipes:
        raise OracleBlockedError("Oracle comparison缺少獨立真值score")
    return {"targets": tuple(definitions), "recipes": recipes,
            "daily_recipes": daily_recipes, "daily_sources": daily_sources,
            "aliases": aliases,
            "selectors": selectors, "providers": tuple(provider_fields)}


def _validated_execution_gates(definition, plan: Mapping[str, Any]) -> dict[str, str]:
    """Resolve execution-only rules outside price-score source identity."""
    try:
        return execution_gate_policies(definition, plan)
    except OracleExecutionPolicyError as exc:
        raise OracleBlockedError(f"NON-PIT Oracle execution policy BLOCKED: {exc}") from exc


# Display labels are intentionally separate from the immutable replay-cache keys.
ORACLE_DISPLAY_NAMES = {"MFE40_Control": "MFE40"}


def _canonical_score_map_sha256(scores: Mapping[tuple[str, str], float]) -> str:
    """Hash actual scored stock-day keys AND numerical values before any cache lookup.

    The independently verified Oracle source is the same object injected into
    the canonical score lookup. Display names, row counts, and source-provider
    summaries are NOT a substitute for these bytes.
    """
    if not scores:
        raise OracleBlockedError("Oracle perfect-score map is empty")
    digest = hashlib.sha256()
    for (ticker, day), value in sorted(scores.items()):
        val = float(value)
        if not math.isfinite(val) or val < 0.0 or val > 1.0:
            raise OracleBlockedError(
                "Oracle perfect-score map violates the approved finite [0,1] percentile scale"
            )
        digest.update(str(ticker).encode("utf-8") + b"\x00")
        digest.update(str(day).encode("ascii") + b"\x00")
        digest.update(struct.pack("!d", val))
    return digest.hexdigest()


def _oracle_requires_direct(definition) -> bool:
    """Resolve selector dependencies from the current canonical comparison list."""
    return "direct" in _oracle_comparison_plan(definition)["selectors"].values()


def _oracle_display_name(name: str) -> str:
    return ORACLE_DISPLAY_NAMES.get(str(name), str(name))




class OracleBlockedError(RuntimeError):
    """Insufficient or incompatible historical truth; do not publish a result."""


class OracleTruthBuildRequired(RuntimeError):
    """A read-only source-cache probe has no certified truth to return."""


def _contract(definition, *, settings=None) -> tuple[Any, Any, Any]:
    """Resolve the currently enabled canonical Research selector, not a retired C100.

    The historical Audit settings still name C100/C101 for provenance; they
    cannot authorize selecting an inactive DL model as the current reference.
    """
    source = definition.source
    if definition.audit_type != SUPPORTED_AUDIT_TYPE:
        raise ValueError("Oracle Audit identity不一致")
    if settings is None:
        settings = get_strategy_comparison_settings(str(source["strategy_profile_id"]))
    elif str(settings.profile_id) != str(source["strategy_profile_id"]):
        raise OracleBlockedError("CONTRACT_UNRESOLVED: Research selector profile identity mismatch")
    base = settings.arms[str(source["baseline_arm_id"])]
    selected_modes = set(_oracle_comparison_plan(definition)["selectors"].values())
    requires_exact = "exact" in selected_modes
    exact_candidates = [arm for arm in settings.enabled_arms if (
        arm.dl_enabled and arm.param_source == base.param_source
        and arm.param_policy == base.param_policy
        and arm.rule_policy == "all_off"
        and arm.dl_runtime_mode == "resource-aware-continuous-score-constrained-optimal"
        and (arm.dl_runtime_options or {}).get("preserve_k_r0") is True
        and (arm.dl_runtime_options or {}).get("constrained_solver") == "exact_branch_and_bound_v1"
        and (arm.dl_runtime_options or {}).get("selection_only") is True
    )]
    if (base.dl_enabled or base.param_policy != "base-finalist-best"
            or base.rule_policy != "all_off"
            or not settings.start_date or not settings.end_date
            or settings.dataset != "full" or settings.rotation != "off"):
        raise OracleBlockedError("CONTRACT_UNRESOLVED: Research Min P16 baseline contract mismatch")
    if requires_exact and not exact_candidates:
        raise OracleBlockedError("CONTRACT_UNRESOLVED: Research DL active Exact selector unavailable")
    canonical = exact_candidates[0] if requires_exact else None
    if requires_exact and any((arm.param_source, arm.param_policy, arm.rule_policy,
            arm.dl_runtime_mode, dict(arm.dl_runtime_options or {})) !=
           (canonical.param_source, canonical.param_policy, canonical.rule_policy,
            canonical.dl_runtime_mode, dict(canonical.dl_runtime_options or {}))
           for arm in exact_candidates[1:]):
        raise OracleBlockedError("CONTRACT_UNRESOLVED: 啟用的Research Exact arms不是同一選股契約")
    if "direct" in selected_modes:
        _direct_arm(definition, settings, base, canonical)
    # A Direct-only plan has no Exact reference. Never relabel a Direct arm
    # as Exact merely to provide a baseline cache ranking policy.
    return settings, base, canonical


def _direct_arm(definition, settings, base, exact):
    """Read a registered DL Direct arm; never synthesize a new selector."""
    compatible = [arm for arm in settings.enabled_arms if (
        arm.dl_enabled and arm.param_source == base.param_source
        and arm.param_policy == base.param_policy and arm.rule_policy == "all_off"
        and arm.dl_runtime_mode == "resource-aware-continuous-score-no-k-no-r0"
        and (arm.dl_runtime_options or {}).get("preserve_k") is False
        and (arm.dl_runtime_options or {}).get("preserve_r0") is False
        and (arm.dl_runtime_options or {}).get("selection_order") == "model_score_desc_then_canonical_tie_v1"
        and (arm.dl_runtime_options or {}).get("selection_only") is True
    )]
    if not compatible:
        # Direct is a formally declared controlled selector in this Audit,
        # even when all Direct Strategy Compare rows are currently disabled.
        compatible = [settings.arms[str(definition.source["direct_arm_id"])]]
    direct = compatible[0]
    options = dict(direct.dl_runtime_options or {})
    if (not direct.dl_enabled or direct.param_source != base.param_source
            or direct.param_policy != base.param_policy or direct.rule_policy != "all_off"
            or direct.dl_runtime_mode != "resource-aware-continuous-score-no-k-no-r0"
            or options.get("preserve_k") is not False
            or options.get("preserve_r0") is not False
            or options.get("selection_order") != "model_score_desc_then_canonical_tie_v1"
            or options.get("selection_only") is not True
            or any((arm.param_source, arm.param_policy, arm.rule_policy,
                    arm.dl_runtime_mode, dict(arm.dl_runtime_options or {})) !=
                   (direct.param_source, direct.param_policy, direct.rule_policy,
                    direct.dl_runtime_mode, options)
                   for arm in compatible[1:])):
        raise OracleBlockedError("CONTRACT_UNRESOLVED: Research DL Direct selector contract不唯一或漂移")
    return direct


def _resolve_oracle_target_score_contracts(definition, settings, plan) -> dict[str, dict[str, Any]]:
    """Resolve the *already approved* perfect Target-score counterfactual.

    A hypothetical perfect predictor need not have been FIT. The research
    comparison arm defines the score-consumption/selection interface, while
    the authorized Audit catalogue defines the one injected numerical score:
    within-day average-zero-based Target percentile. Pairwise raw logits do
    not possess a unique perfect magnitude; this study does NOT claim their
    unconstrained mathematical maximum. It makes one defined intervention at
    the formal score lookup boundary, with all portfolio rules unchanged.
    """
    _oracle_comparison_plan(definition)  # Assert immutable declared score method.
    _settings, _base, exact = _contract(definition, settings=settings)
    direct = (_direct_arm(definition, settings, _base, exact)
              if "direct" in plan["selectors"].values() else None)
    start = _formal_dl_score_start(definition, settings)
    result: dict[str, dict[str, Any]] = {}
    for name, _description in plan["targets"]:
        origin = plan["aliases"].get(name, name)
        if origin in plan["recipes"]:
            recipe, provider = plan["recipes"][origin]
            profile_id = str(definition.source[provider])
            profile = get_breakout_quality_experiment_profile(profile_id)
            source_target = str(profile.continuous_target_id)
            if not source_target:
                raise OracleBlockedError(f"{name}: formal feature/target provider缺少Target ID")
            target_identity = origin  # Existing Oracle target name, never a model/SC ID.
        else:
            recipe, provider, horizon = plan["daily_recipes"][origin]
            profile_id = str(definition.source[provider])
            profile = get_breakout_quality_experiment_profile(profile_id)
            from services.audit.daily_trade40_oracle import target_identity_for_horizon
            source_target = str(profile.continuous_target_id)
            target_identity, _label_id = target_identity_for_horizon(
                horizon, outcome_metric=_ORACLE_DAILY_SCORE_RECIPES[recipe],
            )
        if profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
            raise OracleBlockedError(f"{name}: 來源不是正式Daily DL可評分母體")
        arm = direct if plan["selectors"][name] == "direct" else exact
        arm_source = settings.dl_sources[str(arm.dl_id)]
        if str(arm_source.filter_id) != str(definition.source["filter_id"]):
            raise OracleBlockedError(
                f"CONTRACT_UNRESOLVED: {name} Research DL filter_id與Oracle候選來源不同"
            )
        if str(arm_source.score_source) != "selection_point_in_time":
            raise OracleBlockedError(f"{name}: Research selector不是正式PIT score消費路徑")
        if str(arm_source.point_in_time_score_start_date or "") != start:
            raise OracleBlockedError(f"{name}: Research DL score起始日不一致")
        extra_gate = _validated_execution_gates(definition, plan).get(name)
        result[name] = {
            **({"hindsight_extra_gate": extra_gate} if extra_gate is not None else {}),
            "reference_arm_id": str(arm.arm_id),
            "reference_score_source_id": str(arm.dl_id),
            "reference_profile_id": str(arm_source.experiment_profile),
            "source_profile_id": profile_id,
            "source_target_id": source_target,
            "target_id": target_identity,
            "score_source": str(arm_source.score_source),
            "score_start_date": start,
            "perfect_score_mapping": str(definition.dimensions["oracle_score_method"]),
            "percentile_method": str(definition.dimensions["percentile_method"]),
            "score_semantics": "declared_hypothetical_perfect_target_percentile_not_unique_raw_pairwise_logits",
            "selector": str(plan["selectors"][name]),
            "model_fit_required": False,
        }
    return result


def _oracle_parameter_reuse_plan(definition, root: Path) -> dict[str, Any]:
    """Read-only, one-contract P16 inspection for Oracle and its Research caller.

    Each seed is independently resolved by the canonical Optimizer owner.
    A missing P16 is not an Oracle Replay RUN; the Audit itself stays read-only.
    """
    settings, baseline, _exact = _contract(definition)
    robustness = get_strategy_multi_seed_robustness_settings(settings.profile_id)
    source = settings.parameter_sources[str(baseline.param_source)]
    builder = source.builder
    if (
        robustness.profile_id != settings.profile_id
        or not robustness.enabled or not robustness.benchmark_id
        or not robustness.resolved_seeds
        or len(robustness.resolved_seeds) != robustness.seed_count
        or builder is None or not builder.enabled
        or builder.builder_type != "optimizer_fixed_multiseed_strategy_params"
    ):
        raise OracleBlockedError("Oracle Benchmark無法對齊canonical P16訓練合約")
    training_profile = str(dict(builder.options).get("training_profile_id") or "")
    if not training_profile:
        raise OracleBlockedError("P16缺少canonical training profile")
    producer_kwargs = {
        "training_profile_id": training_profile,
        "benchmark_id": str(robustness.benchmark_id),
        "policy": str(baseline.param_policy),
        "comparison_end_date": str(settings.end_date),
        "dataset": settings.dataset,
        "max_positions": settings.max_positions,
        "rotation": settings.rotation,
        "fixed_risk": float(DEFAULT_FIXED_RISK),
        "max_position_cap_pct": float(DEFAULT_MAX_POSITION_CAP_PCT),
    }
    rows = []
    for seed in robustness.resolved_seeds:
        inspection = resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
            root, replication_seed=int(seed), **producer_kwargs,
        )
        rows.append({
            "seed": int(seed), "ready": bool(inspection["ready"]),
            "action": str(inspection.get("action") or ""),
            "status": str(inspection.get("status") or ""),
            "path": Path(inspection["path"]),
            "manifest_path": Path(inspection["manifest_path"]),
        })
    return {
        "robustness": robustness, "producer_kwargs": producer_kwargs,
        "rows": tuple(rows),
    }


def inspect_oracle_parameter_prerequisites(
    definition, *, project_root: Path,
) -> dict[str, Any]:
    """Validate Oracle source legality before offering any Optimizer work.

    Called by the separate Research entry orchestrator; this function cannot
    FIT parameters or write Replay/truth artifacts.
    """
    root = Path(project_root).resolve()
    settings, _base, _exact = _contract(definition)
    _resolve_oracle_target_score_contracts(
        definition, settings, _oracle_comparison_plan(definition),
    )
    data_dir = Path(get_dataset_dir(str(root), settings.dataset)).resolve()
    if not data_dir.is_dir() or not next(data_dir.rglob("*.csv"), None):
        raise OracleBlockedError("缺少完整研究價格資料（full dataset）")
    from filters.breakout_quality.paths import resolve_filter_output_dir
    truth_root = resolve_filter_output_dir(
        root, filter_id=str(definition.source["filter_id"]),
    )
    if not truth_root.is_dir():
        raise OracleBlockedError(
            "缺少validated breakout-quality Dataset bundle；Audit不得代替Producer建置"
        )
    return _oracle_parameter_reuse_plan(definition, root)


def _robustness_artifacts(definition, root: Path):
    plan = _oracle_parameter_reuse_plan(definition, root)
    for row in plan["rows"]:
        if not row["ready"]:
            raise OracleBlockedError(
                f"Benchmark P16 seed={row['seed']}非READY: {row['status']}"
                "（只讀Audit不自動重訓）"
            )
    return plan["robustness"], [
        (row["seed"], row["path"]) for row in plan["rows"]
    ]


def collect_status(definition, *, project_root: Path) -> dict[str, Any]:
    """Read-only current-benchmark P16 and daily Target prerequisite preflight."""
    root = Path(project_root).resolve()
    display = "Full Research Dataset + canonical P16 + daily target truth"
    try:
        plan = inspect_oracle_parameter_prerequisites(definition, project_root=root)
        incomplete = [row for row in plan["rows"] if not row["ready"]]
        if incomplete:
            first = incomplete[0]
            raise OracleBlockedError(
                f"Benchmark P16 seed={first['seed']}非READY: {first['status']}"
                "（只讀Audit不自動重訓）"
            )
        return {
            "status": "READY", "reason": "", "source": {
                "display": display, "preparable": False,
                "benchmark": plan["robustness"].benchmark_id,
                "seeds": [row["seed"] for row in plan["rows"]],
            },
        }
    except (ValueError, OSError, OracleBlockedError) as exc:
        return {
            "status": "BLOCKED", "reason": str(exc),
            "source": {"display": display, "preparable": False},
        }



def _formal_dl_score_start(definition, settings) -> str:
    """Resolve the start of the Exact/Direct Oracle reference sources only.

    Other enabled DL research arms are not inputs to this controlled Oracle.
    Their score periods, profile eligibility and architecture must not make
    an unrelated comparison change the selected Exact/Direct score contract.
    """
    _settings, base, exact = _contract(definition, settings=settings)
    direct = (_direct_arm(definition, settings, base, exact)
              if _oracle_requires_direct(definition) else None)
    # Validate only selectors consumed by the chosen comparisons. Direct-only
    # is a registered formal selector, not an implicit request to validate or
    # execute an unselected Exact arm. Compatible enabled references must
    # agree on score eligibility/period and filter identity.
    plan = _oracle_comparison_plan(definition)
    selected_references = []
    if "exact" in plan["selectors"].values():
        selected_references.append(exact)
    if direct is not None:
        selected_references.append(direct)
    reference_arms = []
    for reference in selected_references:
        signature = (reference.param_source, reference.param_policy,
                     reference.rule_policy, reference.dl_runtime_mode,
                     dict(reference.dl_runtime_options or {}))
        compatible = [arm for arm in settings.enabled_arms if (
            arm.dl_enabled and
            (arm.param_source, arm.param_policy, arm.rule_policy,
             arm.dl_runtime_mode, dict(arm.dl_runtime_options or {})) == signature
        )]
        if not compatible:
            compatible = [reference]  # Registered, disabled canonical selector.
        for arm in compatible:
            if all(arm.dl_id != prior.dl_id for prior in reference_arms):
                reference_arms.append(arm)
    starts = set()
    for arm in reference_arms:
        source = settings.dl_sources[str(arm.dl_id)]
        if str(source.filter_id) != str(definition.source["filter_id"]):
            raise OracleBlockedError(
                "CONTRACT_UNRESOLVED: Research Exact/Direct reference filter_id mismatch"
            )
        if str(source.score_source) != "selection_point_in_time":
            raise OracleBlockedError("Oracle只支援正式Research DL PIT score來源")
        profile = get_breakout_quality_experiment_profile(str(source.experiment_profile))
        if (profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                or (profile.model_architecture is not None
                    and str(profile.model_architecture) != str(source.model_architecture))):
            raise OracleBlockedError(
                "CONTRACT_UNRESOLVED: Research DL reference的feature eligibility/architecture"
                "不是可共用的Daily-universal分數契約"
            )
        start = str(source.point_in_time_score_start_date or "").strip()
        if not start:
            raise OracleBlockedError("CONTRACT_UNRESOLVED: Research DL缺少正式score start")
        starts.add(pd.Timestamp(start).strftime("%Y-%m-%d"))
    if len(starts) != 1:
        raise OracleBlockedError("CONTRACT_UNRESOLVED: Research DL score start不唯一")
    return starts.pop()


def _market_truth_for_oracle(
    definition, root: Path, *, cutoff: str,
    progress: Callable[[str], None] | None = None,
    provider_fields: tuple[str, ...] | None = None,
    eligible_truth: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Verify official DL-eligible target rows against Research price paths.

    Only the canonical model's eligible/valid stock-days may receive scores.
    The trading universe is NOT the score universe; no 0050 or incomplete
    feature history can be added by a future-price calculation.
    """
    if eligible_truth is None:
        raise OracleBlockedError("Oracle缺少正式DL可評分樣本母體")
    settings, _baseline, _exact = _contract(definition)
    plan = _oracle_comparison_plan(definition)
    fields = (provider_fields if provider_fields is not None else plan["providers"])
    # Scalar Oracle targets consume the canonical label output itself.  Rebuilding
    # T from float32 MFE/MAE components, or ranking an unrounded terminal return,
    # can split exact float32 label ties and change the selected portfolio.
    scalar_target_ids = {
        "terminal_r": DAILY_TERMINAL_R40_TARGET_ID,
        "terminal_mae_gate": DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID,
        "mfe_mae_gate": DAILY_FULL_HORIZON_MAE_GATED_MFE_T40_TARGET_ID,
    }
    scalar_recipes = {
        field: {recipe: scalar_target_ids[recipe]
                for recipe, provider in plan["recipes"].values()
                if provider == field and recipe in scalar_target_ids}
        for field in fields
    }
    specs = {}
    for field in fields:
        profile_id = str(definition.source[field])
        profile = get_breakout_quality_experiment_profile(profile_id)
        recipe = get_continuous_ranker_execution_recipe(profile_id)
        policy = recipe.target_policy
        if (
            policy.materialization_mode != CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT
            or policy.postprocess != CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE
            or not policy.component_target_id
            or recipe.continuous_target_id != profile.continuous_target_id
        ):
            raise OracleBlockedError(f"{field}不能以canonical daily target calculator計算")
        horizon = int(policy.horizon_bars or DEFAULT_LABEL_POLICY.label_horizon_bars)
        maturity = int(policy.sample_universe_horizon_bars or horizon)
        if horizon <= 0 or maturity < horizon:
            raise OracleBlockedError(f"{field}的horizon／maturity contract不合法")
        specs[field] = (
            StrategyAlignedContinuousTargetSpec.from_label_policy(
                replace(DEFAULT_LABEL_POLICY, label_horizon_bars=horizon)
            ),
            maturity,
            str(policy.component_target_id),
        )
        if any(recipe == "mfe" and provider == field
               for recipe, provider in plan["recipes"].values()):
            # Preserve the provider's approved path selection as well as H.
            # First-breach MFE must not become a full-horizon MFE merely to
            # obtain the canonical float32 label output.
            if policy.component_target_id not in {
                DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
            }:
                raise OracleBlockedError(f"{field}: scalar MFE requires a canonical pure-MFE provider")
            scalar_recipes[field]["mfe"] = str(policy.component_target_id)

    csv_inputs, duplicates = discover_dataset_csv_inputs(root, settings.dataset)
    if duplicates:
        raise OracleBlockedError(f"Oracle Research Dataset含重複ticker來源：{duplicates[:3]}")
    paths = {}
    for ticker, path in csv_inputs:
        key = normalize_ticker(ticker)
        if key in paths:
            raise OracleBlockedError(f"Oracle Research Dataset正規化ticker重複：{key}")
        paths[key] = Path(path)
    benchmark = normalize_ticker(DEFAULT_LABEL_POLICY.benchmark_ticker)
    if benchmark not in paths:
        raise OracleBlockedError(f"Oracle缺少可交易Benchmark價格：{benchmark}")

    # A score missing from the formal PIT coverage is unavailable, including
    # the pre-start information bar for the first OOS order.
    bench_frame = load_dataset_frame(paths[benchmark], benchmark, min_rows=1)
    score_start = pd.Timestamp(_formal_dl_score_start(definition, settings))
    eligible_keys = {}
    for field in fields:
        rows = eligible_truth.get(field)
        if rows is None or not {"ticker", "date"}.issubset(rows):
            raise OracleBlockedError(f"{field}缺少正式DL eligibility rows")
        eligible_keys[field] = {
            ticker: set(pd.to_datetime(group["date"]).dt.strftime("%Y-%m-%d"))
            for ticker, group in rows.loc[rows["date"] >= str(score_start.date())].groupby("ticker")
        }
    limit = pd.Timestamp(cutoff)
    chunks = {field: [] for field in fields}
    skipped = {}
    valid_counts = {field: 0 for field in fields}
    if progress is not None:
        progress(f"成熟真值檢查：0/{len(paths)} 檔價格路徑")
    for index, (ticker, path) in enumerate(paths.items(), start=1):
        if progress is not None and (index % 250 == 0 or index == len(paths)):
            progress(f"成熟真值檢查：{index}/{len(paths)} 檔價格路徑")
        try:
            frame = (bench_frame if ticker == benchmark else load_dataset_frame(path, ticker, min_rows=1))
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            # Keep diagnostics traceable without storing host-absolute data paths.
            reason = str(exc).replace(str(path), path.name).replace(str(root), ".")
            skipped[ticker] = f"{type(exc).__name__}: {reason}"
            continue
        dates = pd.DatetimeIndex(frame.index).normalize()
        closes = frame["Close"].to_numpy(dtype=np.float64, copy=False)
        for field, (spec, maturity, target_id) in specs.items():
            requested = eligible_keys[field].get(ticker, set())
            if not requested:
                continue
            positions = np.flatnonzero((dates >= score_start) & (dates <= limit)).astype(np.int64)
            positions = positions[np.isin(dates.take(positions).strftime("%Y-%m-%d"), list(requested))]
            if len(positions) != len(requested):
                raise OracleBlockedError(f"{field}正式DL stock-day找不到Research價格：{ticker}")
            # A target's H is a closeout bound, NOT an eligibility/maturity
            # screen. Historical training labels still require a full H;
            # hindsight scores instead settle at min(H, research cutoff).
            eligible = positions
            horizon = int(spec.horizon_bars)
            cutoff_index = int(dates.searchsorted(limit, side="right")) - 1
            remaining = np.minimum(horizon, cutoff_index - eligible)
            if (remaining < 0).any():
                raise OracleBlockedError(f"{field}: information date after Research cutoff")
            values = np.full((len(eligible), 4), np.nan, dtype=np.float64)
            scalar_raw = {recipe: np.full(len(eligible), np.nan, dtype=np.float32)
                          for recipe in scalar_recipes[field]}
            valid = np.zeros(len(eligible), dtype=bool)
            risk = float(spec.risk_budget_return)
            for bars in np.unique(remaining):
                selected = np.flatnonzero(remaining == bars)
                anchor = closes[eligible[selected]]
                if bars == 0:
                    # At the final information close, a zero-length future
                    # path has zero excursion and zero terminal return. The
                    # last-day score cannot create a next-day Research trade.
                    valid[selected] = np.isfinite(anchor) & (anchor > 0.0)
                    values[selected, :] = 0.0
                    for target_values in scalar_raw.values():
                        target_values[selected] = 0.0
                    continue
                # The canonical batch routine supports a one-bar *observed*
                # path; StrategyAligned spec validation applies only to the
                # original full-horizon training spec, not this Oracle clip.
                clipped = compute_daily_opportunity_target_batch(
                    frame, eligible[selected],
                    spec=replace(spec, horizon_bars=int(bars)), target_id=target_id,
                )
                valid[selected] = np.asarray(clipped.valid_mask, dtype=bool)
                values[selected, :] = np.column_stack((
                    np.asarray(clipped.favorable_return / risk, dtype=float),
                    np.asarray(clipped.adverse_return_to_peak / risk, dtype=float),
                    np.asarray(clipped.minimum_low_return, dtype=float),
                    np.asarray(clipped.target_raw_r, dtype=float),
                ))
                for recipe, scalar_target_id in scalar_recipes[field].items():
                    scalar_batch = clipped if scalar_target_id == target_id else (
                        compute_daily_opportunity_target_batch(
                            frame, eligible[selected],
                            spec=replace(spec, horizon_bars=int(bars)),
                            target_id=scalar_target_id,
                        )
                    )
                    valid[selected] &= np.asarray(scalar_batch.valid_mask, dtype=bool)
                    scalar_raw[recipe][selected] = scalar_batch.target_raw_r
            valid &= np.isfinite(values).all(axis=1)
            for target_values in scalar_raw.values():
                valid &= np.isfinite(target_values)
            if not valid.all():
                missing_day = str(dates[eligible[np.flatnonzero(~valid)[0]]].date())
                raise OracleBlockedError(
                    f"{field}: feature-eligible Oracle price path invalid at {ticker} {missing_day}; "
                    "cannot silently remove scored candidates"
                )
            valid_counts[field] += len(eligible)
            columns = {
                "ticker": ticker,
                "date": dates.take(eligible).strftime("%Y-%m-%d"),
                "mfe": values[:, 0],
                "peak_adverse": values[:, 1],
                "min_low": values[:, 2],
                "raw": values[:, 3],
            }
            columns.update({f"scalar_raw_{recipe}": target_values
                            for recipe, target_values in scalar_raw.items()})
            chunks[field].append(pd.DataFrame(columns))
    truth = {}
    for field in chunks:
        if not chunks[field]:
            raise OracleBlockedError(f"{field}沒有可用的已成熟Research價格真值")
        table = pd.concat(chunks[field], ignore_index=True)
        if table.duplicated(["ticker", "date"]).any():
            raise OracleBlockedError(f"{field}的Research價格真值ticker/date重複")
        truth[field] = table
    return truth, {
        "truth_source": "official feature-eligible model cohort verified via canonical Research OHLCV",
        "score_start": str(score_start.date()),
        "price_ticker_count": len(paths),
        "unreadable_tickers": skipped,
        "mature_rows_by_provider": valid_counts,
    }


def _reconcile_oracle_truth(
    source: pd.DataFrame, market: pd.DataFrame, *, field: str, score_start: str,
) -> tuple[pd.DataFrame, int]:
    """Match the whole legal score cohort; check already-mature DL truth.

    Only Oracle tail rows get cutoff-clipped economic truth. Completed
    supervised labels remain byte-for-byte the source of historical values,
    with market OHLCV independently verifying the same economics.
    """
    old = source.loc[source["date"] >= score_start]
    check = old.merge(market, on=["ticker", "date"], how="left",
                      suffixes=("_provider", "_market"), validate="one_to_one", indicator=True)
    missing = check["_merge"] == "left_only"
    if missing.any():
        row = check.loc[missing, ["ticker", "date"]].iloc[0]
        raise OracleBlockedError(
            f"{field}: canonical score-eligible stock-day lacks Research price path: "
            f"{row['ticker']} {row['date']}"
        )
    if len(market) != len(old):
        raise OracleBlockedError(f"{field}: Oracle prices must equal the complete eligible DL cohort")
    # The canonical loader supplies an explicit maturity witness. Historical
    # synthetic/evidence tables may instead contain only finite completed
    # labels; preserve their original numeric reconciliation contract.
    if "_oracle_provider_mature" in check:
        completed = check["_oracle_provider_mature"].to_numpy(dtype=bool)
    else:
        completed = np.isfinite(check[[f"{name}_provider" for name in
                                       ("mfe", "peak_adverse", "min_low", "raw")]].to_numpy(dtype=float)).all(axis=1)
    columns = ("mfe", "peak_adverse", "min_low", "raw")
    for name in columns:
        if not np.allclose(
            check.loc[completed, f"{name}_provider"].to_numpy(dtype=float),
            check.loc[completed, f"{name}_market"].to_numpy(dtype=float),
            rtol=2e-5, atol=2e-5,
        ):
            raise OracleBlockedError(f"{field}: mature DL target disagrees with OHLCV: {name}")
    matched = check[["ticker", "date"]].copy()
    for name in columns:
        matched[name] = np.where(
            completed, check[f"{name}_provider"], check[f"{name}_market"]
        )
    for name in market.columns:
        if name.startswith("scalar_raw_"):
            matched[name] = check[name]
    return matched, int((~completed).sum())


def _load_truth(definition, root: Path, source_field: str, *, cutoff: str) -> pd.DataFrame:
    """Read the official *feature* cohort, not the supervised maturity mask.

    Historical completed DL labels are an independent numerical parity witness.
    Inference-only rows at the Research tail remain legally score eligible for
    the hindsight Oracle; their clipped outcome is computed from OHLCV below.
    Nothing here alters PIT training-label maturity or model fitting.
    """
    profile_id = str(definition.source[source_field])
    profile = get_breakout_quality_experiment_profile(profile_id)
    architecture = str(profile.model_architecture or definition.source["model_architecture"])
    bundle = load_profile_continuous_ranker_data(
        filter_id=str(definition.source["filter_id"]),
        model_architecture=architecture,
        experiment_profile=profile_id,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    manifest_target = str(dict((bundle.target_manifest or {}).get("target_contract") or {}).get("target_id") or "")
    if manifest_target != str(profile.continuous_target_id):
        raise OracleBlockedError(f"canonical target identity drift: {source_field}")
    group = pd.DataFrame(bundle.group_table).reset_index(drop=True)
    required = {"ticker", "date", "label_eval_end_date", "target_favorable_r",
                "target_adverse_r", "target_minimum_low_return"}
    if not required.issubset(group) or len(group) != len(bundle.raw_target) or len(group) != len(bundle.target_valid):
        raise OracleBlockedError(f"{source_field}: incomplete canonical feature/label bundle")
    observed_end = pd.to_datetime(group["label_eval_end_date"], errors="coerce")
    completed = (np.asarray(bundle.target_valid, dtype=bool)
                 & (observed_end <= pd.Timestamp(cutoff)).to_numpy())
    frame = pd.DataFrame({
        "ticker": group["ticker"].map(normalize_ticker),
        "date": pd.to_datetime(group["date"], errors="raise").dt.strftime("%Y-%m-%d"),
        "mfe": pd.to_numeric(group["target_favorable_r"], errors="coerce"),
        "peak_adverse": pd.to_numeric(group["target_adverse_r"], errors="coerce"),
        "min_low": pd.to_numeric(group["target_minimum_low_return"], errors="coerce"),
        "raw": np.asarray(bundle.raw_target, dtype=float),
    })
    completed &= np.isfinite(frame[["mfe", "peak_adverse", "min_low", "raw"]].to_numpy(dtype=float)).all(axis=1)
    frame["_oracle_provider_mature"] = completed
    frame = frame.loc[frame["date"] <= str(cutoff)].copy()
    if frame.empty or frame.duplicated(["ticker", "date"]).any():
        raise OracleBlockedError(f"{source_field}: empty/duplicated canonical eligible cohort")
    return frame


def _build_oracle_score_panels(
    definition, *, project_root: Path, cutoff: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, Any]]:
    """Build exactly the selected score dependencies, never a stale provider set.

    All selected targets share the same mature common stock-day cohort, so a
    comparison cannot silently use a larger denominator for one score.  The
    historical cohort remains immutable in replay caches, and reuse is checked
    against actual price/parameter provenance by the cache owner.
    """
    plan = _oracle_comparison_plan(definition)
    fields = plan["providers"]
    if not fields:
        raise OracleBlockedError("Daily-only Target分數依賴正式Min per-seed參數，請由Research [1]執行")
    root = Path(project_root).resolve()
    official = {}
    for field in fields:
        if progress is not None:
            progress(f"核對正式DL target provider：{field}")
        official[field] = _load_truth(definition, root, field, cutoff=cutoff)
    market, market_evidence = _market_truth_for_oracle(
        definition, root, cutoff=cutoff, progress=progress,
        provider_fields=fields, eligible_truth=official,
    )
    complete: dict[str, pd.DataFrame] = {}
    extra_rows: dict[str, int] = {}
    for field in fields:
        complete[field], extra_rows[field] = _reconcile_oracle_truth(
            official[field], market[field], field=field,
            score_start=market_evidence["score_start"],
        )
    # A hidden provider intersection would remove otherwise PIT-score-eligible
    # candidates, creating an Oracle-only universe. The currently selected
    # 40D providers must resolve to one common formal cohort; if they do not,
    # do not guess which model's stock-day eligibility should win.
    provider_cohorts = {
        field: set(zip(frame["ticker"].astype(str), frame["date"].astype(str)))
        for field, frame in complete.items()
    }
    reference_cohort = next(iter(provider_cohorts.values()))
    if any(keys != reference_cohort for keys in provider_cohorts.values()):
        raise OracleBlockedError(
            "CONTRACT_UNRESOLVED: Oracle Target providers有不同正式DL可評分股票日，"
            "禁止使用交集縮小候選母體"
        )
    joint: pd.DataFrame | None = None
    for field in fields:
        framed = complete[field].rename(columns={
            col: f"{field}__{col}" for col in complete[field]
            if col not in ("ticker", "date")
        })
        joint = framed if joint is None else joint.merge(
            framed, on=["ticker", "date"], how="inner", validate="one_to_one",
        )
    if joint is None or joint.empty or joint.duplicated(["ticker", "date"]).any():
        raise OracleBlockedError("Oracle selected providers沒有共同成熟且唯一的stock-day")
    risk = float(StrategyAlignedContinuousTargetSpec.from_label_policy(
        DEFAULT_LABEL_POLICY).risk_budget_return)
    method = str(definition.dimensions["percentile_method"])
    if method != "average_zero_based":
        raise OracleBlockedError("Oracle percentile必須採canonical average_zero_based")

    def percentile(values: pd.Series) -> pd.Series:
        # Reuse the supervised DL label's exact daily percentile calculation
        # including float32 target precision, rather than a second rounding
        # convention that may change Exact K/R0 basket score sums.
        ranking = build_same_date_percentile_targets(
            np.asarray(values, dtype=np.float64),
            np.ones(len(joint), dtype=bool), joint["date"],
        )
        return pd.Series(ranking, index=joint.index)

    raw: dict[str, pd.Series] = {}
    safety_by_provider: dict[str, pd.Series] = {}
    for name, (recipe, provider) in plan["recipes"].items():
        def col(key: str) -> pd.Series:
            label = f"{provider}__{key}"
            if label not in joint:
                raise OracleBlockedError(f"Oracle {name}缺少canonical欄位：{label}")
            return pd.to_numeric(joint[label], errors="raise")

        if recipe == "equal_rank_mfe_safety":
            safety = safety_by_provider.get(provider)
            if safety is None:
                safety = full_horizon_mae_safety_r_from_minimum_low_return(
                    col("min_low").to_numpy(dtype=float), risk_budget_return=risk,
                )
                safety = pd.Series(safety, index=joint.index)
                if not np.isfinite(safety.to_numpy(dtype=float)).all():
                    raise OracleBlockedError(f"Oracle {provider} MAE safety包含不合法值")
                safety_by_provider[provider] = safety
            # CG is the approved geometry proxy of the formal MFE/Safety
            # components, not the scalar pure-MFE label or a learned head.
            value = (percentile(col("mfe")) + percentile(safety)) / 2
        else:
            value = col(f"scalar_raw_{recipe}")
        raw[name] = pd.Series(value, index=joint.index)

    keys = list(zip(joint["ticker"].astype(str), joint["date"].astype(str)))
    scores: dict[str, dict[tuple[str, str], float]] = {}
    for name, value in raw.items():
        ranking = percentile(value)
        if not np.isfinite(ranking.to_numpy(dtype=float)).all():
            raise OracleBlockedError(f"Oracle {name}存在非有限rank score")
        scores[name] = dict(zip(keys, ranking.to_numpy(dtype=float)))
    for direct, source in plan["aliases"].items():
        if source in scores:
            scores[direct] = scores[source]
    # Hidden Exact truth providers are dependencies, not selected treatments.
    # No unselected Exact arm may be replayed or published by a Direct-only run.
    selected_names = {name for name, _description in plan["targets"]}
    scores = {name: panel for name, panel in scores.items() if name in selected_names}
    if set(scores) | (set(plan["daily_sources"]) & selected_names) != selected_names:
        raise OracleBlockedError("Oracle比較清單／分數工件依賴不一致")

    # Backward-compatible evidence keys retain their original scientific
    # definitions; unused provider counters and joins are never computed.
    reference = safety_by_provider.get("pure_mfe40_provider")
    mfe_ref = joint.get("pure_mfe40_provider__mfe")
    reference = reference if reference is not None else pd.Series(dtype=float)
    evidence = {
        "cohort_rows": len(keys), "oracle_parity_implementation": ORACLE_PARITY_IMPLEMENTATION,
        "first_common_day": min(joint["date"]),
        "last_common_day": max(joint["date"]), "research_cutoff": cutoff,
        "risk_budget_return": risk,
        "providers": {_PROVIDER_COHORT_IDS.get(field, field): definition.source[field]
                      for field in fields},
        "scoring": "Perfect supervised daily-percentile label output, raw PIT DL score scale; source universe is feature-eligible only",
        "oracle_only": True,
        "ratio_zero_denominator_policy": "D40=0,MFE40>0 -> +infinity (highest rank); 0/0 -> 0; final score = finite daily percentile",
        "mfe40_over_d40_zero_downside_positive_upside_count": (
            int(((reference == 0.0) & (mfe_ref > 0.0)).sum()) if mfe_ref is not None and len(reference) else 0
        ),
        "mfe40_over_d40_zero_zero_count": (
            int(((reference == 0.0) & (mfe_ref == 0.0)).sum()) if mfe_ref is not None and len(reference) else 0
        ),
        "market_truth": market_evidence,
        "rows_outside_dl_training_provider_by_source": extra_rows,
    }
    return scores, evidence


def _price_score_algorithm_sha256() -> str:
    """Hash ONLY canonical price-truth implementation, not menu/report code."""
    functions = (
        _oracle_comparison_plan, _formal_dl_score_start,
        _build_oracle_score_panels, _load_truth, _market_truth_for_oracle,
        _reconcile_oracle_truth,
        _canonical_score_map_sha256,
    )
    digest = hashlib.sha256()
    for func in functions:
        digest.update(func.__qualname__.encode("utf-8") + b"\0")
        digest.update(inspect.getsource(func).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _price_score_replay_lineage(definition, project_root: Path | None = None) -> dict[str, Any]:
    return price_score_replay_lineage(
        root=Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2],
        filter_id=str(definition.source["filter_id"]),
        score_algorithm_sha256=_price_score_algorithm_sha256(),
        label_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
    )


def _price_score_cache_views(
    plan: Mapping[str, Any],
    panels: Mapping[str, dict[tuple[str, str], float]],
) -> tuple[dict[str, dict[tuple[str, str], float]],
           dict[str, dict[tuple[str, str], float]]]:
    """Keep source-neutral cache data separate from selected score recipients.

    Older certified panels may contain only an Exact recipient, only its
    registered Direct alias, or both. Resolve those names to the immutable
    source recipe without rewriting historical bytes or recomputing truth.
    Two saved recipients of the same source must contain identical scores.
    """
    aliases = {
        item["target"]: item["score_source"]
        for item in _ORACLE_DEFAULT_COMPARISONS if item.get("score_source")
    }
    aliases.update(plan["aliases"])
    sources: dict[str, dict[tuple[str, str], float]] = {}
    required_sources = set(plan["recipes"])
    for name, scores in panels.items():
        source = aliases.get(name, name)
        if source not in required_sources:
            raise OracleBlockedError("Oracle price-score cache has an unrecognized source")
        if source in sources and sources[source] != scores:
            raise OracleBlockedError("Oracle Exact/Direct price-score cache source conflict")
        sources.setdefault(source, scores)
    if set(sources) != required_sources:
        raise OracleBlockedError("Oracle price-score cache is missing a required source")
    selected = {
        name: sources[plan["aliases"].get(name, name)]
        for name, _description in plan["targets"]
        if plan["aliases"].get(name, name) in required_sources
    }
    return sources, selected


def _load_or_build_oracle_price_scores(
    definition, *, project_root: Path, cutoff: str, price_sha: str,
    progress: Callable[[str], None] | None = None,
    source_status_callback: Callable[[str], None] | None = None,
    allow_build: bool = True,
) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, Any]]:
    """One source-verified cache for all selected price-only Oracle scores.

    The daily trade horizon changes neither its score inputs nor its source
    identity. Existing 40D perfect-price ranks therefore survive adding a
    60D or full-trade-cycle comparison, and are cropped later by whole days.
    """
    plan = _oracle_comparison_plan(definition)
    if not plan["providers"]:
        raise OracleBlockedError("Price-score producer cannot run for daily-only comparisons")
    root = Path(project_root).resolve()
    settings, _base, _exact = _contract(definition)
    providers = {field: str(definition.source[field]) for field in plan["providers"]}
    from filters.breakout_quality.dataset_store import resolve_dataset_paths
    from filters.breakout_quality.paths import resolve_filter_output_dir
    manifest = resolve_dataset_paths(
        resolve_filter_output_dir(root, str(definition.source["filter_id"]))
    ).summary
    if not manifest.is_file():
        # Without official source evidence there can be no cache reuse; the
        # canonical producer alone decides whether data is available.
        if not allow_build:
            raise OracleTruthBuildRequired("Price Target needs certified truth BUILD")
        if source_status_callback is not None:
            source_status_callback("BUILD")
        panels, cohort = _build_oracle_score_panels(
            definition, project_root=root, cutoff=cutoff, progress=progress,
        )
        _sources, selected = _price_score_cache_views(plan, panels)
        return selected, cohort
    try:
        identity = price_score_source_identity(
            root=root, dataset=str(settings.dataset),
            filter_id=str(definition.source["filter_id"]), price_sha256=price_sha,
            providers=providers, recipes=plan["recipes"],
            score_start=_formal_dl_score_start(definition, settings), cutoff=cutoff,
            score_algorithm_sha256=_price_score_algorithm_sha256(),
            label_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise OracleBlockedError(f"Oracle price Target provenance BLOCKED: {exc}") from exc
    if identity is None:
        # The ordinary canonical producer diagnoses the missing Dataset.
        # Never claim score REUSE without an independently verifiable source.
        if not allow_build:
            raise OracleTruthBuildRequired("Price Target needs certified truth BUILD")
        if source_status_callback is not None:
            source_status_callback("BUILD")
        panels, cohort = _build_oracle_score_panels(
            definition, project_root=root, cutoff=cutoff, progress=progress,
        )
        _sources, selected = _price_score_cache_views(plan, panels)
        return selected, cohort
    store = OraclePriceScoreStore(root / AUDIT_OUTPUT_ROOT /
                                  definition.output_subdir / "truth_cache")
    try:
        cached = store.read(identity, score_sha=_canonical_score_map_sha256)
        if cached is not None:
            panels, cohort = cached
            _sources, selected = _price_score_cache_views(plan, panels)
            if source_status_callback is not None:
                source_status_callback("REUSE")
            if progress:
                progress("Price Target scores REUSE（完整來源／分數身分已驗證）")
            return selected, cohort
        if not allow_build:
            raise OracleTruthBuildRequired("Price Target score panel needs BUILD")
        if source_status_callback is not None:
            source_status_callback("BUILD")
        if progress:
            progress("Price Target scores BUILD（缺少相同身分真值工件）")
        panels, cohort = _build_oracle_score_panels(
            definition, project_root=root, cutoff=cutoff, progress=progress,
        )
        sources, selected = _price_score_cache_views(plan, panels)
        store.write(identity, sources, cohort, score_sha=_canonical_score_map_sha256)
        return selected, cohort
    except ValueError as exc:
        raise OracleBlockedError(f"Oracle price Target artifact BLOCKED: {exc}") from exc






def _metrics(replay: dict[str, Any], *, target: str) -> dict[str, Any]:
    summary = _scenario_summary(replay)
    required = ("total_return_pct", "max_drawdown_pct", "return_over_max_drawdown", "expected_value_r", "trade_count", "avg_exposure_pct", "win_rate_pct")
    if any(not math.isfinite(float(summary[name])) for name in (*required, "annual_return_pct")):
        raise OracleBlockedError(f"Oracle {target}缺少有效portfolio performance metrics")
    if not 0.0 <= float(summary["win_rate_pct"]) <= 100.0:
        raise OracleBlockedError(f"Oracle {target} has invalid completed-trade win rate")
    yearly = _yearly_frame(replay["profile"], "strategy")
    return {
        "target": target,
        "total_return_pct": float(summary["total_return_pct"]),
        "annualized_return_pct": float(summary["annual_return_pct"]),
        "mdd_pct": float(summary["max_drawdown_pct"]),
        "romd": float(summary["return_over_max_drawdown"]),
        "ev_a_r": float(summary["expected_value_r"]),
        "win_rate_pct": float(summary["win_rate_pct"]),
        "trades": int(summary["trade_count"]),
        "exposure_pct": float(summary["avg_exposure_pct"]),
        "yearly": _to_json_native(yearly.to_dict("records")),
    }


def _aggregate(
    records: list[dict[str, Any]], *, target: str,
    period_start: str | None = None, period_end: str | None = None,
) -> dict[str, Any]:
    if not records:
        raise OracleBlockedError("Oracle robustness missing seed observations")
    seeds = [int(row["seed"]) for row in records]
    if len(set(seeds)) != len(seeds):
        raise OracleBlockedError(f"Oracle {target} robustness has duplicate benchmark seeds")
    expected_years: tuple[str, ...] | None = None
    if (period_start is None) != (period_end is None):
        raise OracleBlockedError(f"Oracle {target} annual period boundaries are incomplete")
    if period_start is not None:
        first = pd.Timestamp(period_start)
        last = pd.Timestamp(period_end)
        if last < first:
            raise OracleBlockedError(f"Oracle {target} annual period boundaries are reversed")
        expected_years = tuple(str(year) for year in range(first.year, last.year + 1))
    # Every observed year must exist for *every* same-window seed. Otherwise
    # a one-seed yearly result could appear as an N-seed Oracle comparison.
    annual_rows: list[dict[str, dict[str, Any]]] = []
    common_years: tuple[str, ...] | None = None
    common_full_year: dict[str, bool] | None = None
    for record in records:
        yearly = record.get("yearly")
        if not isinstance(yearly, list) or not yearly:
            raise OracleBlockedError(f"Oracle {target} seed={record['seed']} missing annual evidence")
        by_year = {str(row["year"]): row for row in yearly}
        if len(by_year) != len(yearly):
            raise OracleBlockedError(f"Oracle {target} seed={record['seed']} duplicated annual evidence")
        years = tuple(sorted(by_year))
        # All seeds agreeing with one another is not enough: all of them may
        # have silently lost the SAME year. The Replay period is authoritative.
        if expected_years is not None and years != expected_years:
            raise OracleBlockedError(
                f"Oracle {target} seed={record['seed']} annual coverage does not "
                f"match replay period {period_start}~{period_end}: "
                f"expected={expected_years}, found={years}"
            )
        complete = {}
        for year in years:
            year_flag = by_year[year].get("is_full_year")
            if not isinstance(year_flag, (bool, np.bool_)):
                raise OracleBlockedError(
                    f"Oracle {target} seed={record['seed']} year={year} "
                    "missing/invalid full-year evidence"
                )
            complete[year] = bool(year_flag)
        if common_years is None:
            common_years, common_full_year = years, complete
        elif years != common_years or complete != common_full_year:
            raise OracleBlockedError(
                f"Oracle {target} annual seed coverage/full-year metadata differs: "
                f"seed={record['seed']}; expected years={common_years}, found={years}"
            )
        annual_rows.append(by_year)
    summary = {"target": target, "seeds": len(records)}
    fields = (
        "total_return_pct", "annualized_return_pct", "mdd_pct", "romd", "ev_a_r", "win_rate_pct", "trades", "exposure_pct",
    )
    for field in fields:
        values = np.asarray([float(x[field]) for x in records], dtype=float)
        summary[field] = float(np.mean(values))
        summary[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    summary["yearly"] = [
        {
            "year": year,
            "strategy_return_pct": float(np.mean([
                float(rows[year]["strategy_return_pct"]) for rows in annual_rows
            ])),
            "n_seeds": len(records),
            "is_full_year": bool(common_full_year[year]),
        }
        for year in (common_years or ())
    ]
    summary["seed_observations"] = records
    return summary


def render_replay_preflight(
    *,
    label: str,
    seeds: list[int],
    targets: list[str],
    decisions: Mapping[tuple[int, str], str],
) -> str:
    """Render the one complete identity-backed Replay authorization plan.

    Display is bounded by target count, not seeds x targets. Source data,
    training and result identities are never inferred from color/status.
    """
    counts = {action: sum(action == value for value in decisions.values())
              for action in ("REUSE", "RUN", "BLOCKED")}
    headers = ("Target", *(f"S{i + 1}" for i in range(len(seeds))))
    rows = [
        (target, *(styled_workflow_status(decisions[(seed, target)]) for seed in seeds))
        for target in targets
    ]
    return "\n".join((
        f"[Target 驗證 {label} | Portfolio Replay 正式計畫]",
        "Seed：" + "  ".join(f"S{i + 1}={seed}" for i, seed in enumerate(seeds)),
        render_table(headers, rows),
        f"Portfolio REUSE {counts['REUSE']} / RUN（需回放）{counts['RUN']} / "
        f"BLOCKED {counts['BLOCKED']}；REUSE=零Portfolio交易重算，"
        "RUN=舊執行身分過時或合法結果缺件，由系統只計算該項；"
        "BLOCKED=資料損毀或同一執行身分證據矛盾，須停止以避免覆寫；尚未開始回放",
    ))


def _prepare_oracle_parameters(
    definition, *, root: Path, parameter_paths, start: str, end: str, base_arm,
):
    """Single canonical parameter-view resolution for fast REUSE and Replay."""
    prepared = []
    param_evidence = []
    for seed, path in parameter_paths:
        source, param_sha = load_strategy_param_evaluation_view(
            path, evaluation_mode=str(definition.dimensions["evaluation_mode"]),
            start_date=start, end_date=end,
        )
        _validate_requested_param_policy(source, str(base_arm.param_policy))
        kind = str(source["kind"])
        if kind == "rolling_active_param_ensemble":
            schedule_start, schedule_end = get_active_param_ensemble_date_range(source["payload"])
        elif kind == "rolling_oos_param_schedule":
            schedule_start, schedule_end = get_active_param_date_range(source["payload"])
        else:
            raise OracleBlockedError("Oracle基準必須重用正式Rolling P16日程，禁止靜態前視參數")
        if str(schedule_start) > start or str(schedule_end) < end:
            raise OracleBlockedError(
                f"Oracle P16日程未完整覆蓋成熟期間: {schedule_start}~{schedule_end}"
            )
        kind, base_params, oracle_params, *_ = _build_controlled_param_source_pair(
            source, filter_id=str(definition.source["filter_id"]),
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD), fixed_risk=None,
            comparison_mode=COMPARISON_MODE_SCORE_RANKING,
            optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
            shared_param_overrides=ALL_RULE_FILTERS_OFF_OVERRIDES,
        )
        prepared.append((seed, param_sha, kind, base_params, oracle_params))
        param_evidence.append({
            "seed": seed,
            "param_path": project_relative_display_path(path, project_root=root),
            "param_sha256": param_sha,
        })
    return prepared, param_evidence


def _canonical_oracle_replay_policy(definition, *, base_arm, exact_arm=None) -> dict[str, Any]:
    """Shared economic identity for full Replay and fast verified REUSE."""
    replay_policy = {
        "baseline_arm": str(definition.source["baseline_arm_id"]),
        "exact_arm": str(exact_arm.arm_id) if exact_arm is not None else str(definition.source["exact_arm_id"]),
        "filter_id": str(definition.source["filter_id"]),
        "param_policy": str(base_arm.param_policy),
        "comparison_mode": COMPARISON_MODE_SCORE_RANKING,
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "all_rule_filters_off": dict(ALL_RULE_FILTERS_OFF_OVERRIDES),
        "score_threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        "fixed_risk": float(DEFAULT_FIXED_RISK),
        "max_position_cap_pct": float(DEFAULT_MAX_POSITION_CAP_PCT),
        "broker_accounting": {
            "policy_id": BROKER_ACCOUNTING_POLICY_ID,
            "fee_rate": str(BROKER_FEE_RATE),
            "min_fee": str(BROKER_MIN_FEE_TWD),
            "odd_lot_min_fee": str(BROKER_ODD_LOT_MIN_FEE_TWD),
            "board_lot_shares": int(BROKER_BOARD_LOT_SHARES),
            "stock_tax_rate": str(BROKER_STOCK_TAX_RATE),
            "rounding": BROKER_CHARGE_ROUNDING_MODE,
        },
        "execution_defaults": {
            key: str(spec["default"]) if spec["default"] is not None else None
            for key, spec in sorted(EXECUTION_POLICY_PARAM_SPECS.items())
        },
    }
    return replay_policy



def _arm_replay_policy(definition, *, plan: Mapping[str, Any], target: str,
                       replay_policy: Mapping[str, Any],
                       daily_evidence: Mapping[str, Any] | None = None,
                       score_sha256: str | None = None,
                       score_contract: Mapping[str, Any] | None = None,
                       project_root: Path | None = None) -> dict[str, Any]:
    """One target-scoped economic/score identity for fast and normal REUSE."""
    policy = dict(replay_policy)
    policy["period_stats_implementation"] = PERIOD_STATS_IMPLEMENTATION
    if target != str(definition.source["baseline_arm_id"]):
        policy["oracle_parity_implementation"] = ORACLE_PARITY_IMPLEMENTATION
    if plan["selectors"].get(target) == "direct":
        policy["direct_arm"] = str(definition.source["direct_arm_id"])
    # The unscored C96 baseline is deliberately built with a minimal plan;
    # neither it nor any non-selected recipient has an execution-only gate.
    execution_gate = (_validated_execution_gates(definition, plan).get(target)
                      if target in plan.get("selectors", {}) else None)
    if execution_gate is not None:
        from core.file_integrity import compute_file_sha256
        code_root = Path(__file__).resolve().parents[2]
        policy["hindsight_extra_gate"] = {
            "policy": execution_gate,
            "raw_truth_source": "canonical_fullcycle_min_broker_ledger_net_pnl",
            "null_and_nonpositive": "reject_all_orders",
            "execution_policy_code_sha256": compute_file_sha256(
                code_root / "services/audit/oracle_execution_policy.py"),
            "gate_code_sha256": compute_file_sha256(code_root / "core/oracle_positive_gate.py"),
            "raw_witness_code_sha256": compute_file_sha256(code_root / "services/audit/fullcycle_positive_truth.py"),
            "selector_code_sha256": compute_file_sha256(code_root / "core/portfolio_entry_selection_max_dl.py"),
            "admission_code_sha256": compute_file_sha256(code_root / "core/portfolio_candidate_admission.py"),
        }
    if target != str(definition.source["baseline_arm_id"]):
        if not score_sha256 or len(score_sha256) != 64 or not score_contract:
            raise OracleBlockedError("Oracle per-target score SHA and canonical DL score contract required")
        policy["perfect_score_sha256"] = str(score_sha256)
        policy["canonical_dl_score_contract"] = dict(score_contract)
    if target in plan["daily_sources"]:
        # New target identity is validated against the actual label values and
        # rank table, independently of legacy price-only Oracle cohort/cache.
        # The same evidence goes into both Exact and Direct replay identities.
        if daily_evidence is None:
            raise OracleBlockedError("Daily-universal Oracle missing verified target lineage")
        from services.audit.daily_trade40_oracle import (
            NET_PNL_EVIDENCE, target_identity_for_horizon, truth_producer_fingerprint,
        )
        source = plan["daily_sources"][target]
        recipe, provider, horizon = plan["daily_recipes"][source]
        if daily_evidence.get("truth_source_sha256") != truth_producer_fingerprint(
                _ORACLE_DAILY_SCORE_RECIPES[recipe]):
            raise OracleBlockedError("Oracle Target truth has no verified executing producer lineage")
        source_summary_sha = oracle_dataset_summary_sha256(
            root=Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2],
            filter_id=str(definition.source["filter_id"]),
        )
        if ("dataset_summary_sha256" not in daily_evidence or
                daily_evidence["dataset_summary_sha256"] != source_summary_sha):
            raise OracleBlockedError("Oracle Target truth has no matching dataset-source evidence")
        target_id, label_id = target_identity_for_horizon(
            horizon, outcome_metric=_ORACLE_DAILY_SCORE_RECIPES[recipe],
        )
        if (daily_evidence.get("profile_id") != str(definition.source[provider])
                or daily_evidence.get("target_id") != target_id
                or daily_evidence.get("label_id") != label_id):
            raise OracleBlockedError("Daily-universal Target/profile does not match the requested formal Min source")
        if recipe == "daily_trade_path_net_pnl" and any(
                daily_evidence.get(key) != expected for key, expected in NET_PNL_EVIDENCE.items()):
            raise OracleBlockedError("NetPnL Target lacks the canonical reference-capital/broker-ledger contract")
        policy["hindsight_score_recipe"] = recipe
        policy["hindsight_score_key_mode"] = "information_date"
        policy["hindsight_target_contract"] = {
            "label_id": daily_evidence["label_id"],
            "target_id": daily_evidence["target_id"],
            "profile_id": str(definition.source[provider]),
            **({"horizon_mode": "full_trade_cycle"} if horizon is None else
               {"horizon_sessions": horizon} if horizon != HORIZON_SESSIONS else {}),
            "rank_policy": daily_evidence["rank_policy"],
            **({key: daily_evidence[key] for key in NET_PNL_EVIDENCE}
               if recipe == "daily_trade_path_net_pnl" else {}),
            "rank_sha256": daily_evidence["rank_sha256"],
            "raw_label_sha256": daily_evidence["raw_label_sha256"],
            "truth_source_sha256": daily_evidence["truth_source_sha256"],
            "dataset_summary_sha256": daily_evidence["dataset_summary_sha256"],
            "calendar_sha256": daily_evidence["calendar_sha256"],
            "teacher": daily_evidence["teacher"],
            "mature_eligible_rows": daily_evidence["mature_eligible_rows"],
            "research_cutoff_closeout_rows": daily_evidence.get("research_cutoff_closeout_rows", 0),
            "cutoff_truncated_rows": daily_evidence.get("cutoff_truncated_rows", 0),
            "score_start_date": daily_evidence["score_start_date"],
            ("settled_net_pnl_rows" if recipe == "daily_trade_path_net_pnl" else
             "settled_net_r_rows"): daily_evidence[
                 "settled_net_pnl_rows" if recipe == "daily_trade_path_net_pnl" else "settled_net_r_rows"
             ],
            "censored_rank_last_rows": daily_evidence["censored_rank_last_rows"],
            # Matches the canonical DL lookup: a candidate outside the
            # feature-eligible score universe has no score, never 0R.
            # Distinct replay identity prevents accidental historical reuse.
            "missing_key_policy": "unavailable_outside_mature_feature_universe",
            "research_cutoff": daily_evidence["research_cutoff"],
            **({"oracle_evaluation_start_date": daily_evidence["oracle_evaluation_start_date"]}
               if daily_evidence.get("oracle_evaluation_start_date") else {}),
        }
    elif target != str(definition.source["baseline_arm_id"]):
        source = plan["aliases"].get(target, target)
        recipe, provider = plan["recipes"][source]
        policy["hindsight_target_contract"] = {
            **_price_score_replay_lineage(definition, project_root),
            "target_id": source,
            "profile_id": str(definition.source[provider]),
            "score_recipe": recipe,
        }
    return policy


def _certify_unscored_baseline_before_truth(
    definition, *, project_root: Path, settings, base_arm, exact_arm,
    robustness, prepared_parameters, price_sha: str, evaluation_end: str,
    certified_identity_out: dict[int, dict[str, Any]] | None = None,
    certified_cohort_keys: tuple[str, int, str, str] | None = None,
    allow_historical_execution_input_drift: bool = False,
) -> dict[int, str]:
    """Certify C96 directly from its actual economic inputs, not Target scores.

    C96 does NOT consume any Target score. Its historically saved cohort is
    report-only metadata; reusing that cohort to reconstruct an expected
    identity cannot authorize reuse of a scored Oracle arm. Price, P16,
    selector, dates, fees, execution and report-period policy are all checked
    by the SAME OracleReplayStore.plan_execution owner as a normal replay.
    """
    identity_arm = exact_arm or settings.arms[str(definition.source["exact_arm_id"])]
    if (identity_arm.dl_runtime_mode != "resource-aware-continuous-score-constrained-optimal"
            or (identity_arm.dl_runtime_options or {}).get("preserve_k_r0") is not True
            or (identity_arm.dl_runtime_options or {}).get("constrained_solver") !=
                "exact_branch_and_bound_v1"
            or (identity_arm.dl_runtime_options or {}).get("selection_only") is not True):
        raise OracleBlockedError("CONTRACT_UNRESOLVED: C96 baseline selector changed")
    root = Path(project_root).resolve()
    store = OracleReplayStore(root, definition.output_subdir)
    common_policy = _canonical_oracle_replay_policy(
        definition, base_arm=base_arm, exact_arm=identity_arm,
    )
    baseline_policy = _arm_replay_policy(
        definition, plan={"selectors": {}, "daily_sources": {}}, target=str(definition.source["baseline_arm_id"]),
        replay_policy=common_policy,
    )
    baseline = str(definition.source["baseline_arm_id"])
    result: dict[int, str] = {}
    for seed, param_sha, _kind, _base_params, _oracle_params in prepared_parameters:
        try:
            candidates = store.existing_completed_identities(seed=int(seed), target=baseline)
            if not candidates:
                result[int(seed)] = "RUN"
                continue
            decisions = []
            reusable_identities = []
            for candidate in candidates:
                cohort = candidate.get("cohort")
                if not isinstance(cohort, Mapping):
                    raise ValueError("C96 saved cohort must be a mapping")
                if certified_cohort_keys is not None:
                    # Saved report boundaries cannot certify a changed DL
                    # scored universe. Use previously signed *current* score
                    # membership; never compute labels just to plan a Replay.
                    _digest, rows, first, last = certified_cohort_keys
                    if (cohort.get("cohort_rows") != rows or
                            cohort.get("first_common_day") != first or
                            cohort.get("last_common_day") != last):
                        continue
                identity = replay_identity(
                    dataset=settings.dataset, dataset_sha256=price_sha,
                    target=baseline, definition={baseline: "Canonical same-window Min P16 Base-Finalist-Best"},
                    start=str(settings.start_date), end=evaluation_end,
                    benchmark=str(robustness.benchmark_id), param_sha256=param_sha,
                    seed=int(seed), cohort=cohort,
                    ranking_policy=str(identity_arm.dl_runtime_mode),
                    ranking_options=dict(identity_arm.dl_runtime_options),
                    evaluation_mode=str(definition.dimensions["evaluation_mode"]),
                    max_positions=settings.max_positions, replay_policy=baseline_policy,
                )
                action = store.plan_execution(
                    {(int(seed), baseline): identity},
                    allow_historical_execution_input_drift=allow_historical_execution_input_drift,
                ).action(seed, baseline)
                decisions.append(action)
                if action == "REUSE":
                    reusable_identities.append(identity)
            # Multiple incompatible candidates are never silently collapsed
            # to the first match. Conflicts fail closed before any Replay.
            result[int(seed)] = (
                "BLOCKED" if "BLOCKED" in decisions else
                "REUSE" if len(reusable_identities) == 1 else
                "BLOCKED" if len(reusable_identities) > 1 else "RUN"
            )
            if result[int(seed)] == "REUSE" and certified_identity_out is not None:
                certified_identity_out[int(seed)] = reusable_identities[0]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result[int(seed)] = "BLOCKED"
    return result




def _oracle_limitations(*, seed_count: int) -> list[str]:
    """One science statement across canonical RUN and certified REUSE reports."""
    return [
            "Ex-post future targets are NOT predictions, not PIT-legal, and never a deployable strategy.",
            "CG EqualRank is a fixed 50/50 geometry proxy, not a learned model-head fusion.",
            "Exact and Direct pairs share identical future truth; only canonical K/R0 allocation changes.",
            "Direct means canonical No-K/No-R0, NOT direct raw-R regression or executable prediction.",
            "C96 is the same-seed Min/Broker-Cash control for both selectors; paired deltas use that baseline.",
            f"N={seed_count} paired end-to-end parameter seeds; Oracle scores have no trainable seed.",
            "Win Rate Mean is the arithmetic mean of canonical per-seed completed Round-Trip win rates; Win/Seed instead counts RoMD wins vs C96.",
            "No best-target fitting from iterative OOS; multi-target search is exploratory.",
            "R40 is the future 40th session Close return, not Portfolio Replay exit return.",
            "T40 uses full-path nonnegative MAE and clamps to -1R at 1R adverse movement.",
            "T40-MFE replaces only terminal R40 by full-path MFE40_R; the identical MAE gate remains.",
            "1R is the fixed research Label risk budget, NOT the actual executed ATR stop-loss.",
        ]


def _oracle_report_envelope(*, settings, robustness, seeds, base: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ORACLE_REPORT_SCHEMA_VERSION,
        "status": "COMPLETED", "read_only": True,
        "hindsight_non_pit": True, "iterative_oos_evidence": True,
        "strategy_profile_id": settings.profile_id,
        "benchmark_id": robustness.benchmark_id,
        "seeds": list(seeds), "dataset": settings.dataset,
        **base, "limitations": _oracle_limitations(seed_count=len(seeds)),
    }


def _read_only_score_keys(
    definition, *, project_root: Path, contracts: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any], evaluation_end: str, cutoff: str,
) -> dict[str, tuple[str, int, str, str]]:
    """Materialize eligible input keys for an authorized cold-start RUN.

    The canonical DL provider also computes ordinary supervised 40D labels
    while loading its universe. This is NOT a lightweight REUSE preflight;
    warm preflight must use _replay_attested_score_keys instead. Only a RUN
    without a certified historical membership witness may call this owner.
    """
    cache: dict[tuple[str, bool], tuple[str, int, str, str]] = {}
    result: dict[str, tuple[str, int, str, str]] = {}
    for target, contract in contracts.items():
        is_daily = target in plan["daily_sources"]
        profile_id = str(contract["source_profile_id"])
        key = (profile_id, is_daily)
        if key not in cache:
            profile = get_breakout_quality_experiment_profile(profile_id)
            bundle = load_profile_continuous_ranker_data(
                filter_id=str(definition.source["filter_id"]),
                model_architecture=str(profile.model_architecture or definition.source["model_architecture"]),
                experiment_profile=profile_id, preload_feature_bank=False,
                allow_stale_source=False, project_root=project_root,
            )
            summary = dict(bundle.summary or {})
            group = pd.DataFrame(bundle.group_table).reset_index(drop=True)
            if (summary.get("filter_id") != str(definition.source["filter_id"])
                    or not {"ticker", "date"}.issubset(group)):
                raise OracleBlockedError(f"{target}: canonical feature cohort source mismatch")
            date = pd.to_datetime(group["date"], errors="raise")
            valid = ((date >= pd.Timestamp(contract["score_start_date"])) &
                     (date <= pd.Timestamp(evaluation_end))).to_numpy()
            if not is_daily and (len(bundle.target_valid) != len(group)
                                    or len(bundle.raw_target) != len(group)):
                raise OracleBlockedError(f"{target}: price feature source evidence incomplete")
            keys = sorted(zip(group.loc[valid, "ticker"].map(normalize_ticker),
                              date.loc[valid].dt.strftime("%Y-%m-%d")))
            if not keys or len(set(keys)) != len(keys):
                raise OracleBlockedError(f"{target}: duplicate/empty source eligibility")
            digest = hashlib.sha256()
            for ticker, day in keys:
                digest.update(f"{ticker}|{day}\n".encode("utf-8"))
            cache[key] = (digest.hexdigest(), len(keys), min(d for _t, d in keys),
                          max(d for _t, d in keys))
        result[target] = cache[key]
    # The same stock days, not only a common count, are mandatory for E13.
    if len({part[0] for part in result.values()}) != 1:
        raise OracleBlockedError("CONTRACT_UNRESOLVED: different certified stock-day populations")
    return result


def _replay_attested_score_keys(
    *, definition, store, contracts: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any], prepared, price_sha: str,
    settings, robustness, evaluation_end: str,
) -> dict[str, tuple[str, int, str, str]]:
    """Recover certified score *membership* from completed Replay receipts.

    This is a pure signed-artifact and current-source-identity probe. It does
    NOT open the Daily DL sample producer (which calculates historical 40D
    labels), nor does it run a MinTrade path or a price Target producer.
    A receipt is usable as a membership witness only if its economic/feature
    inputs, parity revision, profile, parameter SHA, and date bounds still
    match. The later per-arm full-identity verification is still mandatory.
    """
    from core.file_integrity import is_sha256_hex
    from services.audit.daily_trade40_oracle import truth_producer_fingerprint
    # Source code hashes are small, read-only provenance checks: NO outcomes,
    # DL labels, 558-ticker simulation or Portfolio are materialized here.
    # Net-R and the newly added NetPnL have separate producer identities.
    producer_sha_by_recipe = {
        recipe: truth_producer_fingerprint(_ORACLE_DAILY_SCORE_RECIPES[recipe])
        for recipe, _provider, _horizon in plan["daily_recipes"].values()
    }
    price_producer_sha = (_price_score_replay_lineage(definition, store.project_root)["truth_source_sha256"]
                          if plan["recipes"] else None)
    dataset_summary_sha = oracle_dataset_summary_sha256(
        root=store.project_root, filter_id=str(definition.source["filter_id"]))
    proofs: dict[tuple[str, bool], tuple[str, int, str, str]] = {}
    start = str(settings.start_date)
    for seed, param_sha, *_unused in prepared:
        for target, expected in contracts.items():
            group = (str(expected["source_profile_id"]), target in plan["daily_sources"])
            for saved in store.existing_completed_identities(seed=seed, target=target):
                if (saved.get("start_date") != start or
                        saved.get("end_date") != evaluation_end or
                        saved.get("dataset") != settings.dataset or
                        saved.get("dataset_sha256") != price_sha or
                        saved.get("param_sha256") != param_sha or
                        not benchmark_seed_execution_equivalent(
                            recorded_benchmark=str(saved.get("benchmark") or ""),
                            active_benchmark=str(robustness.benchmark_id),
                            seed=int(seed),
                        )):
                    continue
                policy = saved.get("replay_policy") or {}
                # A historical perfect-score SHA proves the *old* score bytes,
                # not that the current Target producer has the same semantics.
                # Both price and trade-path scores require current producer
                # evidence; a warm Replay must not bypass its truth-cache SHA.
                lineage = policy.get("hindsight_target_contract") or {}
                if target in plan["daily_sources"]:
                    source = plan["daily_sources"][target]
                    recipe = plan["daily_recipes"][source][0]
                    current_producer_sha = producer_sha_by_recipe[recipe]
                else:
                    current_producer_sha = price_producer_sha
                if (not isinstance(lineage, Mapping) or
                        lineage.get("truth_source_sha256") != current_producer_sha):
                    continue
                if target in plan["daily_sources"] and (
                        "dataset_summary_sha256" not in lineage or
                        lineage["dataset_summary_sha256"] != dataset_summary_sha):
                    continue
                cohort = saved.get("cohort") or {}
                source_contract = policy.get("canonical_dl_score_contract") or {}
                if (not isinstance(policy, Mapping) or
                        not isinstance(cohort, Mapping) or
                        not isinstance(source_contract, Mapping) or
                        policy.get("oracle_parity_implementation") != ORACLE_PARITY_IMPLEMENTATION or
                        policy.get("period_stats_implementation") != PERIOD_STATS_IMPLEMENTATION or
                        cohort.get("research_cutoff") != str(settings.end_date) or
                        not is_sha256_hex(policy.get("perfect_score_sha256"))):
                    continue
                for field, value in expected.items():
                    if field in {"reference_arm_id", "reference_score_source_id", "reference_profile_id"}:
                        continue
                    if source_contract.get(field) != value:
                        break
                else:
                    if source_contract.get("source_evidence") != (
                            "validated_generic_daily_feature_universe_and_research_cutoff_closeout"):
                        continue
                    digest = source_contract.get("canonical_feature_eligible_keys_sha256")
                    number = source_contract.get("oracle_scored_rows")
                    first = cohort.get("first_common_day")
                    last = cohort.get("last_common_day")
                    if (not is_sha256_hex(digest) or type(number) is not int or number <= 0 or
                            number != cohort.get("cohort_rows") or
                            not isinstance(first, str) or not isinstance(last, str) or
                            not (start <= first <= last <= evaluation_end)):
                        continue
                    proof = (digest, number, first, last)
                    prior = proofs.get(group)
                    if prior is not None and prior != proof:
                        raise OracleBlockedError(
                            "Oracle same-source signed Replay receipts disagree on stock-day membership; "
                            "BLOCKED without any Target calculation"
                        )
                    proofs[group] = proof
    return {target: proofs[(str(contract["source_profile_id"]), target in plan["daily_sources"])]
            for target, contract in contracts.items()
            if (str(contract["source_profile_id"]), target in plan["daily_sources"]) in proofs}


def _certified_saved_oracle_identity(
    definition, *, project_root: Path, store, saved: Mapping[str, Any],
    plan: Mapping[str, Any], contract: Mapping[str, Any],
    keys: tuple[str, int, str, str], param_sha: str,
    price_sha: str, seed: int, target: str, settings, robustness,
    base_policy: Mapping[str, Any], arm, rolling_periods: list[dict[str, Any]] | None,
    evaluation_end: str,
) -> dict[str, Any]:
    """Attest a completed replay from signed source/score lineage; no Producer.

    This is NOT file-existence reuse. The historical score SHA is accepted only
    when its producing revision, complete source bytes, P16 lineage, current
    feature keys and immutable score consumption contract are all certified.
    A mismatched or incomplete historical receipt must BLOCK, not silently RUN.
    """
    from services.audit.oracle_replay_artifacts import replay_identity
    from services.audit.daily_trade40_oracle import (
        NET_PNL_EVIDENCE, NET_PNL_RANK_POLICY,
        target_identity_for_horizon, truth_producer_fingerprint,
    )
    policy = saved.get("replay_policy")
    cohort = saved.get("cohort")
    if not isinstance(policy, Mapping) or not isinstance(cohort, Mapping):
        raise ValueError("completed Replay has no full source/replay evidence")
    if policy.get("oracle_parity_implementation") != ORACLE_PARITY_IMPLEMENTATION:
        raise ValueError("past replay uses an unapproved Oracle implementation")
    if (policy.get("period_stats_implementation") != PERIOD_STATS_IMPLEMENTATION
            or saved.get("dataset_sha256") != price_sha
            or saved.get("param_sha256") != param_sha
            or saved.get("start_date") != str(settings.start_date)
            or saved.get("end_date") != evaluation_end
            or saved.get("dataset") != settings.dataset
            or not benchmark_seed_execution_equivalent(
                            recorded_benchmark=str(saved.get("benchmark") or ""),
                            active_benchmark=str(robustness.benchmark_id),
                            seed=int(seed),
                        )):
        raise ValueError("current price/P16/evaluation contract differs from historical replay")
    key_sha, n, first, last = keys
    if (cohort.get("cohort_rows") != n
            or cohort.get("first_common_day") != first
            or cohort.get("last_common_day") != last
            or cohort.get("research_cutoff") != str(settings.end_date)):
        raise ValueError("current stock-day population differs from historical replay")
    source_contract = policy.get("canonical_dl_score_contract")
    if not isinstance(source_contract, Mapping):
        raise ValueError("completed replay missing certified hypothetical score interface")
    expected_score = dict(contract)
    expected_score.update(
        canonical_feature_eligible_keys_sha256=key_sha, oracle_scored_rows=n,
        source_evidence="validated_generic_daily_feature_universe_and_research_cutoff_closeout",
    )
    # Arm/model labels are evidence of the reference selector, not its executable
    # score. The actual runtime ranking policy/options remain strictly checked.
    for field, value in expected_score.items():
        if field in {"reference_arm_id", "reference_score_source_id", "reference_profile_id"}:
            continue
        if source_contract.get(field) != value:
            raise ValueError(f"historical perfect-score contract mismatch: {field}")
    if set(source_contract) - set(expected_score):
        raise ValueError("historical score contract contains unknown executable fields")
    score_sha = policy.get("perfect_score_sha256")
    from core.file_integrity import is_sha256_hex
    if not is_sha256_hex(score_sha):
        raise ValueError("historical perfect-score SHA evidence missing")
    source = plan["daily_sources"].get(target)
    daily_evidence = None
    if source:
        recipe, provider, horizon = plan["daily_recipes"][source]
        recorded = policy.get("hindsight_target_contract")
        if not isinstance(recorded, Mapping) or not rolling_periods:
            raise ValueError("historical daily-truth lineage missing")
        if recorded.get("truth_source_sha256") != truth_producer_fingerprint(
                _ORACLE_DAILY_SCORE_RECIPES[recipe]):
            raise ValueError("historical Target outcome producer semantics not certified")
        if ("dataset_summary_sha256" not in recorded or
                recorded["dataset_summary_sha256"] != oracle_dataset_summary_sha256(
                    root=project_root, filter_id=str(definition.source["filter_id"]))):
            raise ValueError("historical daily Target dataset-source evidence differs")
        actual_id, label_id = target_identity_for_horizon(
            horizon, outcome_metric=_ORACLE_DAILY_SCORE_RECIPES[recipe],
        )
        if recipe == "daily_trade_path_net_pnl" and any(
                recorded.get(key) != expected for key, expected in NET_PNL_EVIDENCE.items()):
            raise ValueError("NetPnL Oracle source has no approved reference-capital/ledger contract")
        teacher = recorded.get("teacher")
        if (recorded.get("target_id") != actual_id
                or recorded.get("label_id") != label_id
                or recorded.get("rank_policy") != (
                    NET_PNL_RANK_POLICY
                    if recipe == "daily_trade_path_net_pnl" else
                    "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation"
                )
                or recorded.get("research_cutoff") != str(settings.end_date)
                or recorded.get("score_start_date") != contract["score_start_date"]
                or not isinstance(teacher, Mapping)
                or teacher.get("seed") != seed
                or teacher.get("param_identity_sha256") != param_sha
                or teacher.get("periods") != rolling_periods):
            raise ValueError("current MinTrade Target/teacher policy differs")
        if any(not is_sha256_hex(recorded.get(field)) for field in
               ("rank_sha256", "raw_label_sha256", "calendar_sha256")):
            raise ValueError("historical daily outcome or ranking SHA missing")
        if int(recorded.get("mature_eligible_rows", -1)) != n:
            raise ValueError("historical MinTrade score population differs")
        valid_row_count_field = (
            "settled_net_pnl_rows" if recipe == "daily_trade_path_net_pnl" else "settled_net_r_rows"
        )
        if int(recorded.get(valid_row_count_field, -1)) + int(
                recorded.get("censored_rank_last_rows", -1)) != n:
            raise ValueError("historical MinTrade settled/unfilled accounting differs")
        # Canonical policy builder validates all explicit policy and selector
        # fields. Cutoff closeout *counts* are excluded from replay identity.
        daily_evidence = dict(recorded)
    reconstructed = _arm_replay_policy(
        definition, plan=plan, target=target, replay_policy=base_policy,
        daily_evidence=daily_evidence, score_sha256=score_sha,
        score_contract=expected_score, project_root=project_root,
    )
    if not source and policy.get("hindsight_target_contract") != reconstructed.get("hindsight_target_contract"):
        raise ValueError("historical price Target producer/source evidence differs")
    expected = replay_identity(
        dataset=settings.dataset, dataset_sha256=price_sha,
        target=target, definition={target: target},
        start=str(settings.start_date), end=evaluation_end,
        benchmark=str(robustness.benchmark_id), param_sha256=param_sha,
        seed=seed, cohort=cohort,
        ranking_policy=str(arm.dl_runtime_mode),
        ranking_options=dict(arm.dl_runtime_options),
        evaluation_mode=str(definition.dimensions["evaluation_mode"]),
        max_positions=settings.max_positions, replay_policy=reconstructed,
    )
    return expected


def _execute_single_stage_missing(
    definition, *, project_root: Path, quiet: bool,
    plan: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]],
    current_keys: Mapping[str, tuple[str, int, str, str]], settings, robustness,
    price_sha: str, prepared, decisions: Mapping[tuple[int, str], str],
    preflight_cached: Mapping[tuple[int, str], Mapping[str, Any]],
    start: str, end: str, base_arm, exact_arm, direct_arm,
    base_policy: Mapping[str, Any], cache,
    allow_historical_execution_input_drift: bool = False,
    metric_backfill: Mapping[tuple[int, str], tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Run ONLY the missing Seed x Target cells in the certified one-shot plan.

    Target truth is materialized only for sources consumed by a RUN cell.
    Exact/Direct consume one identical source, but a cached sibling never
    causes a second producer invocation. The canonical Broker-Cash Replay
    engine remains the sole economic execution owner.
    """
    from services.audit.oracle_replay_artifacts import replay_identity, price_inventory_sha256
    from services.audit.daily_trade40_oracle import (
        build_daily_universal_trade40_oracle_scores,
        DailyTrade40OracleBlockedError,
    )
    from filters.breakout_quality.daily_trade40_target import (
        build_formal_min_execution_schedule, resolve_formal_min_param_records,
    )
    # Do not consume preflight decisions if any material input was edited
    # during the interactive confirmation prompt.
    if price_inventory_sha256(project_root, settings.dataset) != price_sha:
        raise OracleBlockedError("Oracle price inputs changed after approved preflight")
    current_params, _ = _prepare_oracle_parameters(
        definition, root=project_root,
        parameter_paths=[(seed, path) for seed, path in _robustness_artifacts(
            definition, project_root)[1]],
        start=start, end=end, base_arm=base_arm,
    )
    if [(p[0], p[1]) for p in current_params] != [(p[0], p[1]) for p in prepared]:
        raise OracleBlockedError("Oracle P16 inputs changed after approved preflight")
    out: dict[tuple[int, str], dict[str, Any]] = {}
    required = {key for key, action in decisions.items() if action == "RUN"}
    metric_backfill = dict(metric_backfill or {})
    if not set(metric_backfill).issubset(required):
        raise OracleBlockedError("Win-rate supplement escaped the authorized RUN plan")
    if not required:
        return out
    all_cohorts = set(current_keys.values())
    if len(all_cohorts) > 1:
        raise OracleBlockedError("Oracle RUN has incompatible stock-day cohort")
    # A completely cold source has no historical score-key witness. Its
    # one-time data/eligibility materialization is a RUN dependency, after
    # explicit approval; it is never a REUSE verification dependency.
    if not all_cohorts:
        current_keys = _read_only_score_keys(
            definition, project_root=project_root, contracts=contracts,
            plan=plan, evaluation_end=end, cutoff=str(settings.end_date),
        )
        all_cohorts = set(current_keys.values())
        if len(all_cohorts) != 1:
            raise OracleBlockedError("Oracle first-run score sources have unequal eligibility")
    cohort_keys = next(iter(all_cohorts))
    cohort = {
        "cohort_rows": cohort_keys[1], "first_common_day": cohort_keys[2],
        "last_common_day": cohort_keys[3],
        "oracle_parity_implementation": ORACLE_PARITY_IMPLEMENTATION,
        "research_cutoff": str(settings.end_date),
        "risk_budget_return": float(StrategyAlignedContinuousTargetSpec.from_label_policy(
            DEFAULT_LABEL_POLICY).risk_budget_return), "oracle_only": True,
    }
    # Source truth is independent of the recipient selector. Build only
    # necessary price recipes; do not materialize unused price providers.
    required_price = {target for (_seed, target) in required
                      if target != "C96" and target not in plan["daily_sources"]}
    price_scores: dict[str, dict[tuple[str, str], float]] = {}
    if required_price:
        source = {**definition.source,
                  "oracle_comparisons": tuple(entry for entry in definition.source["oracle_comparisons"]
                                              if entry["target"] in required_price)}
        price_def = replace(definition, source=source)
        price_scores, _evidence = _load_or_build_oracle_price_scores(
            price_def, project_root=project_root, cutoff=str(settings.end_date),
            price_sha=price_sha, progress=None if quiet else
            lambda msg: print(f"[Target RUN price] {msg}", flush=True),
        )
        price_scores = {
            name: {key: value for key, value in scores.items() if key[1] <= end}
            for name, scores in price_scores.items()
        }
    feature_cohort_cache: dict[tuple[str, ...], tuple[pd.DataFrame, str]] = {}
    truth_cache: dict[tuple[int, str], tuple[dict[tuple[str, str], float], dict[str, Any]]] = {}
    positive_raw_cache: dict[int, dict[tuple[str, str], float]] = {}
    execution_gates = _validated_execution_gates(definition, plan)
    data_dir = Path(get_dataset_dir(str(project_root), settings.dataset)).resolve()
    for seed, param_sha, kind, base_params, oracle_params in prepared:
        seed_run = [target for target in ("C96", *(name for name, _ in plan["targets"]))
                    if (seed, target) in required]
        if not seed_run:
            continue
        schedule = None
        preparation = {}
        for target in seed_run:
            key = (seed, target)
            if target == "C96":
                arm = exact_arm or settings.arms[str(definition.source["exact_arm_id"])]
                score_map = None
                evidence = None
                policy = _arm_replay_policy(
                    definition, plan=plan, target="C96", replay_policy=base_policy,
                )
            else:
                arm = direct_arm if plan["selectors"][target] == "direct" else exact_arm
                if target in plan["daily_sources"]:
                    source_name = plan["daily_sources"][target]
                    source_key = (seed, source_name)
                    if source_key not in truth_cache:
                        if schedule is None:
                            schedule, _ = build_formal_min_execution_schedule(
                                resolve_formal_min_param_records(kind, base_params),
                                seed=seed, param_sha256=param_sha,
                            )
                        _recipe, provider, horizon = plan["daily_recipes"][source_name]
                        try:
                            truth_cache[source_key] = build_daily_universal_trade40_oracle_scores(
                                project_root=project_root, dataset=str(settings.dataset),
                                filter_id=str(definition.source["filter_id"]),
                                profile_id=str(definition.source[provider]),
                                cutoff=str(settings.end_date), evaluation_start=start,
                                evaluation_end=end,
                                score_start=_formal_dl_score_start(definition, settings),
                                seed=seed, param_sha256=param_sha,
                                canonical_schedule=schedule, horizon_sessions=horizon,
                                outcome_metric=_ORACLE_DAILY_SCORE_RECIPES[_recipe],
                                price_sha256=price_sha,
                                cache_root=project_root / AUDIT_OUTPUT_ROOT /
                                           definition.output_subdir / "truth_cache",
                                feature_cohort_cache=feature_cohort_cache,
                                progress_callback=None if quiet else
                                    lambda msg, name=source_name: print(
                                        f"[Target RUN seed={seed} {name}] {msg}", flush=True),
                            )
                        except (DailyTrade40OracleBlockedError, ValueError) as exc:
                            raise OracleBlockedError(
                                f"Oracle missing truth {source_name} seed={seed} BLOCKED: {exc}"
                            ) from exc
                    score_map, evidence = truth_cache[source_key]
                    teacher = evidence.get("teacher") or {}
                    if (teacher.get("seed") != seed
                            or teacher.get("param_identity_sha256") != param_sha):
                        raise OracleBlockedError("Oracle RUN Target and P16 teacher differ")
                else:
                    score_map = price_scores.get(target)
                    evidence = None
                if score_map is None or not score_map:
                    raise OracleBlockedError(f"Oracle missing RUN score: {target} seed={seed}")
                proof_sha, n, _first, _last = current_keys.get(target, cohort_keys)
                digest = hashlib.sha256()
                for ticker, day in sorted(score_map):
                    digest.update(f"{ticker}|{day}\n".encode("utf-8"))
                if len(score_map) != n or digest.hexdigest() != proof_sha:
                    raise OracleBlockedError(f"Oracle {target} producer score eligibility differs from preflight")
                full_contract = dict(contracts[target])
                full_contract.update(
                    canonical_feature_eligible_keys_sha256=proof_sha,
                    oracle_scored_rows=n,
                    source_evidence="validated_generic_daily_feature_universe_and_research_cutoff_closeout",
                )
                policy = _arm_replay_policy(
                    definition, plan=plan, target=target, replay_policy=base_policy,
                    daily_evidence=evidence,
                    score_sha256=_canonical_score_map_sha256(score_map),
                    score_contract=full_contract, project_root=project_root,
                )
            identity = replay_identity(
                dataset=settings.dataset, dataset_sha256=price_sha,
                target=target, definition={target: target},
                start=start, end=end, benchmark=str(robustness.benchmark_id),
                param_sha256=param_sha, seed=seed, cohort=cohort,
                ranking_policy=str(arm.dl_runtime_mode),
                ranking_options=dict(arm.dl_runtime_options),
                evaluation_mode=str(definition.dimensions["evaluation_mode"]),
                max_positions=settings.max_positions, replay_policy=policy,
            )
            action = cache.plan_execution(
                {key: identity},
                allow_historical_execution_input_drift=allow_historical_execution_input_drift,
            )
            try:
                action.require_ready()
            except ValueError as exc:
                raise OracleBlockedError(str(exc)) from exc
            is_metric_backfill = key in metric_backfill
            expected_action = "REUSE" if is_metric_backfill else "RUN"
            if action.action(seed, target) != expected_action:
                raise OracleBlockedError(
                    f"Oracle {target} seed={seed} changed after preflight; refusing silent replanning"
                )
            if is_metric_backfill:
                original_identity, original_observation = metric_backfill[key]
                current_saved, _ = cache.resolve_completed(
                    identity,
                    allow_historical_execution_input_drift=allow_historical_execution_input_drift,
                )
                if identity != original_identity or current_saved != original_observation:
                    raise OracleBlockedError(
                        f"Oracle {target} seed={seed} changed before win-rate evidence Replay"
                    )
            positive_raw = None
            if target in execution_gates:
                if source_name != "DailyUniversal_MinTrade_NetPnL":
                    raise OracleBlockedError("Positive gate must consume the full-cycle NetPnL source")
                if seed not in positive_raw_cache:
                    from services.audit.fullcycle_positive_truth import load_or_build_positive_truth
                    positive_raw_cache[seed], _ = load_or_build_positive_truth(
                        project_root=project_root, output_subdir=definition.output_subdir,
                        dataset=str(settings.dataset), filter_id=str(definition.source["filter_id"]),
                        profile_id=str(definition.source["daily_trade40_profile"]),
                        seed=seed, price_sha256=price_sha, param_sha256=param_sha,
                        canonical_schedule=schedule, research_cutoff=str(settings.end_date),
                        score_start=_formal_dl_score_start(definition, settings), evaluation_end=end,
                        canonical_scores=score_map, canonical_evidence=evidence,
                        feature_cohort_cache=feature_cohort_cache,
                        progress=None if quiet else lambda msg: print(f"[Positive Target] {msg}", flush=True),
                    )
                positive_raw = positive_raw_cache[seed]
            replay_execution_rows = [] if positive_raw is not None else None
            replay_kwargs = dict(
                replay_execution_rows=replay_execution_rows,
                data_dir=data_dir, start_date=start, end_date=end,
                max_positions=settings.max_positions, enable_rotation=False,
                quiet=quiet, replay_preparation_cache=preparation,
            )
            def produce():
                if target == "C96":
                    played = _run_scenario(
                        name=f"C96 Fixed seed={seed} same-window",
                        params=base_params, param_source_kind=kind, **replay_kwargs,
                    )
                else:
                    with ExitStack() as stack:
                        stack.enter_context(audit_oracle_truth_score_context(
                            score_map,
                            ranking_policy=str(arm.dl_runtime_mode),
                            ranking_options=arm.dl_runtime_options,
                            key_mode="information_date",
                            missing_key_policy="unavailable_outside_mature_feature_universe",
                            score_start=_formal_dl_score_start(definition, settings),
                        ))
                        if positive_raw is not None:
                            from core.oracle_positive_gate import positive_oracle_entry_gate
                            stack.enter_context(positive_oracle_entry_gate(positive_raw))
                        played = _run_scenario(
                            name=f"Oracle {target} seed={seed}",
                            params=oracle_params, param_source_kind=kind, **replay_kwargs,
                        )
                result = _metrics(played, target=target)
                if positive_raw is not None:
                    from services.audit.positive_gate_evidence import save_positive_replay_evidence
                    save_positive_replay_evidence(
                        project_root=project_root, output_subdir=definition.output_subdir,
                        seed=seed, replay_identity=identity, portfolio=played,
                        execution_rows=replay_execution_rows,
                        raw_net_pnl=positive_raw, summary=result,
                    )
                result["seed"] = seed
                if target != "C96":
                    baseline = out.get((seed, "C96")) or preflight_cached.get((seed, "C96"))
                    if baseline is None:
                        raise OracleBlockedError("Oracle RUN missing paired C96 benchmark")
                    result["delta_romd_vs_c96"] = result["romd"] - baseline["romd"]
                return result
            if is_metric_backfill:
                result = cache.write_win_rate_evidence(
                    identity, original_observation, produce(),
                    allow_historical_execution_input_drift=allow_historical_execution_input_drift,
                )
                if not quiet:
                    print(f"[RUN] seed={seed} {target} | Win Rate evidence completed; "
                          "original Replay preserved", flush=True)
            else:
                result, origin = action.execute(seed, target, produce)
                if origin != "newly completed":
                    raise OracleBlockedError("Oracle RUN unexpectedly reused a different receipt")
                if not quiet:
                    print(f"[RUN] seed={seed} {target} | completed", flush=True)
            out[key] = result
    if set(out) != required:
        raise OracleBlockedError("Oracle missing-only execution did not complete every RUN cell")
    return out


def _run_single_stage_oracle(
    definition, *, project_root: Path, quiet: bool,
    preflight_approval: Callable[[str], bool] | None,
    preflight_display: str = "full",
    allow_historical_execution_input_drift: bool = False,
    preview_only: bool = False,
    approved_replay_decisions: Mapping[tuple[int, str], str] | None = None,
) -> dict[str, Any]:
    """Read-only matrix planning (including missing P16), then certified Replay.

    A provenance-backed historical replay is a completed scientific result;
    it does not require replaying the label producer merely to rediscover an
    already-attested perfect-score SHA. No archived report authorizes reuse.
    """
    from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
    from services.audit.daily_trade40_oracle import truth_producer_fingerprint
    from filters.breakout_quality.daily_trade40_target import (
        build_formal_min_execution_schedule, resolve_formal_min_param_records,
    )
    root = Path(project_root).resolve()
    plan = _oracle_comparison_plan(definition)
    settings, base_arm, exact_arm = _contract(definition)
    contracts = _resolve_oracle_target_score_contracts(definition, settings, plan)
    pending_p16: tuple[int, ...] = ()
    if preview_only:
        # The Research entry must show the *Target* plan before building P16.
        # The Audit stays read-only: existing seeds undergo the same complete
        # Replay attestation below; missing P16 can only be provisional RUN.
        parameter_plan = inspect_oracle_parameter_prerequisites(definition, project_root=root)
        unsafe = [row for row in parameter_plan["rows"]
                  if not row["ready"] and row["action"] not in {"BUILD", "REPAIR"}]
        if unsafe:
            raise OracleBlockedError(
                "P16 既有工件身分不合法，禁止覆寫：" +
                ", ".join(f"seed={row['seed']}:{row['status']}" for row in unsafe)
            )
        robustness = parameter_plan["robustness"]
        paths = [(row["seed"], row["path"]) for row in parameter_plan["rows"]
                 if row["ready"]]
        pending_p16 = tuple(int(row["seed"]) for row in parameter_plan["rows"]
                            if not row["ready"])
    else:
        status = collect_status(definition, project_root=root)
        if status["status"] != "READY":
            raise OracleBlockedError(status["reason"])
        robustness, paths = _robustness_artifacts(definition, root)
    end = str(definition.dimensions.get("oracle_evaluation_end_date") or "")
    start = str(settings.start_date)
    if not (start < end <= str(settings.end_date)):
        raise OracleBlockedError("Oracle fixed evaluation window invalid")
    try:
        price_sha = price_inventory_sha256(root, settings.dataset)
    except ValueError as exc:
        raise OracleBlockedError(f"Oracle price provenance BLOCKED: {exc}") from exc
    prepared, param_evidence = _prepare_oracle_parameters(
        definition, root=root, parameter_paths=paths,
        start=start, end=end, base_arm=base_arm,
    )
    cache = OracleReplayStore(root, definition.output_subdir)
    baseline_arm = exact_arm or settings.arms[str(definition.source["exact_arm_id"])]
    direct_arm = (_direct_arm(definition, settings, base_arm, exact_arm)
                  if "direct" in plan["selectors"].values() else None)
    baseline_policy = _canonical_oracle_replay_policy(
        definition, base_arm=base_arm, exact_arm=baseline_arm,
    )
    # Artifact-only score-universe proof. A full DL data-provider call here
    # silently recomputed hundreds of 40D price labels even on 100% REUSE.
    keys = _replay_attested_score_keys(
        definition=definition, store=cache, contracts=contracts, plan=plan, prepared=prepared,
        price_sha=price_sha, settings=settings, robustness=robustness,
        evaluation_end=end,
    )
    seed_set = ([int(row["seed"]) for row in parameter_plan["rows"]]
                if preview_only else [seed for seed, _ in paths])
    targets = ("C96", *(name for name, _ in plan["targets"]))
    decisions: dict[tuple[int, str], str] = {}
    cached: dict[tuple[int, str], dict[str, Any]] = {}
    identities: dict[tuple[int, str], dict[str, Any]] = {}
    metric_backfill: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    blocked: dict[tuple[int, str], str] = {}
    # Reconstruct canonical Min rolling schedule metadata once per seed, not
    # a single stock's historical trade path or future Target outcome.
    schedules: dict[int, list[dict[str, Any]]] = {}
    if plan["daily_recipes"]:
        for seed, sha, kind, base_params, _ in prepared:
            schedule, _ = build_formal_min_execution_schedule(
                resolve_formal_min_param_records(kind, base_params),
                seed=seed, param_sha256=sha,
            )
            schedules[seed] = [{"start": str(p.start.date()),
                                "end": None if p.end is None else str(p.end.date()),
                                "params_signature": p.params_signature}
                               for p in schedule]
    baseline_certified: dict[int, dict[str, Any]] = {}
    # Source signatures may certify the shared cohort without materializing
    # any Target scores. If sources disagree, do not infer a C96 cohort.
    if len(set(keys.values())) > 1:
        raise OracleBlockedError("Oracle signed Replay witnesses disagree on common score cohort")
    signed_cohort = next(iter(keys.values())) if keys else None
    baseline_status = _certify_unscored_baseline_before_truth(
        definition, project_root=root, settings=settings, base_arm=base_arm,
        exact_arm=baseline_arm, robustness=robustness,
        prepared_parameters=prepared, price_sha=price_sha,
        evaluation_end=end, certified_identity_out=baseline_certified,
        certified_cohort_keys=signed_cohort,
        allow_historical_execution_input_drift=allow_historical_execution_input_drift,
    )
    # Compare signed provenance before considering a cached perfect-score
    # SHA. Legacy receipts without the producer closure fingerprint cannot
    # claim unchanged outcome semantics after the canonical cutoff bug fix.
    current_daily_producer_sha_by_recipe = {
        recipe: truth_producer_fingerprint(_ORACLE_DAILY_SCORE_RECIPES[recipe])
        for recipe, _provider, _horizon in plan["daily_recipes"].values()
    }
    current_price_producer_sha = (_price_score_replay_lineage(definition, root)["truth_source_sha256"]
                                  if plan["recipes"] else None)
    current_dataset_summary_sha = oracle_dataset_summary_sha256(
        root=root, filter_id=str(definition.source["filter_id"]))
    # Read the current canonical input manifest once; every stored score
    # receipt is checked against that same source and fixed evaluation scope.
    # No existing scored witness means every scored cell is a genuine RUN.
    # Expensive feature/price label construction is deferred until approved
    # RUN execution, NEVER performed merely to display the matrix.
    if len(set(keys.values())) > 1:
        raise OracleBlockedError("Oracle signed Replay witnesses disagree on common score cohort")
    for seed, param_sha, _kind, _base_params, _oracle_params in prepared:
        for target in targets:
            key = (seed, target)
            if target == "C96":
                c96 = baseline_certified.get(seed)
                if c96 is not None:
                    identity = c96
                    obs, _ = cache.resolve_completed(
                        identity,
                        allow_historical_execution_input_drift=allow_historical_execution_input_drift,
                    )
                    if obs is None:
                        raise OracleBlockedError("C96 evidence disappeared during preflight")
                    complete = cache.with_win_rate(identity, obs)
                    if complete is None:
                        decisions[key] = "RUN"
                        metric_backfill[key] = (identity, obs)
                    else:
                        decisions[key], cached[key], identities[key] = "REUSE", complete, identity
                else:
                    # Historical C96 inputs/revisions are not scientific
                    # conflicts. A stale result remains intact and the
                    # corrected current identity becomes its own RUN. Only
                    # corrupt receipts or genuinely contradictory evidence
                    # remain blocked; neither can be repaired by guessing.
                    state = baseline_status.get(seed, "BLOCKED")
                    if state == "BLOCKED":
                        blocked[key] = "C96 工件完整性或相同身分證據衝突；禁止自動覆寫"
                        decisions[key] = "BLOCKED"
                    else:
                        decisions[key] = "RUN"
                continue
            if target in plan["daily_sources"]:
                source = plan["daily_sources"][target]
                recipe = plan["daily_recipes"][source][0]
                producer_sha = current_daily_producer_sha_by_recipe[recipe]
            else:
                producer_sha = current_price_producer_sha
            known_lineage = {"truth_source_sha256": producer_sha}
            if target in plan["daily_sources"]:
                known_lineage["dataset_summary_sha256"] = current_dataset_summary_sha
            # Only already verified inputs enter this source probe. It never
            # guesses new scores or membership, and cannot authorize REUSE.
            # The cache owner also owns strict-versus-authorized input drift.
            current_candidates, source_conflict = cache.inspect_current_source(
                seed=seed, target=target,
                expected_inputs={
                    "start_date": start, "end_date": end,
                    "dataset": settings.dataset, "dataset_sha256": price_sha,
                    "param_sha256": param_sha, "benchmark": robustness.benchmark_id,
                    "replay_policy": {
                        "oracle_parity_implementation": ORACLE_PARITY_IMPLEMENTATION,
                        "period_stats_implementation": PERIOD_STATS_IMPLEMENTATION,
                        "hindsight_target_contract": known_lineage,
                    },
                },
                allow_historical_execution_input_drift=allow_historical_execution_input_drift,
            )
            if source_conflict is not None:
                decisions[key], blocked[key] = "BLOCKED", source_conflict
                continue
            if not current_candidates:
                # Old outcomes are preserved as historical evidence. Only
                # this missing *current* producer identity needs new truth.
                decisions[key] = "RUN"
                continue
            if len(current_candidates) > 1:
                decisions[key] = "BLOCKED"
                blocked[key] = "multiple same-window scored replays require source conflict resolution"
                continue
            arm = direct_arm if plan["selectors"][target] == "direct" else exact_arm
            try:
                if target not in keys:
                    raise ValueError("historical score key receipt cannot be certified from current inputs")
                expected = _certified_saved_oracle_identity(
                    definition, project_root=root, store=cache,
                    saved=current_candidates[0], plan=plan, contract=contracts[target],
                    keys=keys[target], param_sha=param_sha, price_sha=price_sha,
                    seed=seed, target=target, settings=settings, robustness=robustness,
                    base_policy=baseline_policy, arm=arm,
                    rolling_periods=schedules.get(seed), evaluation_end=end,
                )
                approved_plan = cache.plan_execution(
                    {key: expected},
                    allow_historical_execution_input_drift=allow_historical_execution_input_drift,
                )
                approved_plan.require_ready()
                if approved_plan.action(seed, target) != "REUSE":
                    raise ValueError("stored score receipt not reproducible from current source identity")
                obs, _ = cache.resolve_completed(
                    expected,
                    allow_historical_execution_input_drift=allow_historical_execution_input_drift,
                )
                if obs is None:
                    raise ValueError("certified historical replay disappeared")
                complete = cache.with_win_rate(expected, obs)
                if complete is None:
                    decisions[key] = "RUN"
                    metric_backfill[key] = (expected, obs)
                else:
                    decisions[key], cached[key], identities[key] = "REUSE", complete, expected
            except (OSError, ValueError, KeyError, TypeError, OracleBlockedError) as exc:
                decisions[key] = "BLOCKED"
                blocked[key] = str(exc)
    if preview_only:
        for seed in pending_p16:
            for target in targets:
                # Without canonical P16 parameters no full Replay identity
                # exists to certify; do NOT claim REUSE from a file inventory.
                decisions[(seed, target)] = "RUN"
    preflight = render_replay_preflight(
        label="Fixed", seeds=seed_set, targets=list(targets), decisions=decisions,
    )
    if not quiet:
        if blocked or preflight_display == "full":
            print("\n" + preflight, flush=True)
        else:
            reuse_count = sum(value == "REUSE" for value in decisions.values())
            run_count = sum(value == "RUN" for value in decisions.values())
            print(f"[Target 驗證｜身分核對完成] REUSE {reuse_count} / "
                  f"RUN（需計算）{run_count} / BLOCKED 0 | "
                  f"本次共同回放期間 {start} ～ {end}; "
                  "僅相同期間及完整身分可 REUSE", flush=True)
    if blocked:
        problems = [f"{target} seed={seed}: {reason}"
                    for (seed, target), reason in blocked.items()]
        raise OracleBlockedError("Oracle full-identity BLOCKED; no Target producer or Portfolio "
                                 "ran: " + " | ".join(problems[:3]))
    if preview_only:
        return {
            "status": "PREVIEW", "preflight": decisions, "text": preflight,
            "missing_p16": pending_p16, "seed_set": tuple(seed_set),
            "targets": tuple(targets),
            "producer_kwargs": dict(parameter_plan["producer_kwargs"]),
        }
    if approved_replay_decisions is not None:
        # The Research user authorized the *typed* (seed, Target) plan before
        # missing P16 was fitted. A previously REUSE cell cannot become RUN,
        # nor may a new cell appear. RUN -> REUSE is safe (less work).
        if preflight_approval is None or set(approved_replay_decisions) != set(decisions):
            raise OracleBlockedError("Target 授權後 Seed／Target 集合已改變；停止 Replay")
        unexpected = [
            (seed, target, approved, decisions[(seed, target)])
            for (seed, target), approved in approved_replay_decisions.items()
            if (approved not in {"REUSE", "RUN"} or
                decisions[(seed, target)] not in
                ({"REUSE"} if approved == "REUSE" else {"REUSE", "RUN"}))
        ]
        if unexpected:
            seed, target, approved, current = unexpected[0]
            raise OracleBlockedError(
                "Target 授權後 REUSE／RUN 身分或計畫已改變；停止 Replay："
                f"seed={seed} Target={target} 原={approved} 現={current}"
            )
    if preflight_approval is not None and not preflight_approval(preflight):
        return {"status": "CANCELLED", "read_only": True, "preflight": decisions}
    if oracle_dataset_summary_sha256(root=root, filter_id=str(definition.source["filter_id"])) != current_dataset_summary_sha:
        raise OracleBlockedError("Oracle dataset-source metadata changed after approved preflight")
    # Cold-start eligibility is needed only to produce genuinely missing arms.
    # Performing this DL provider load before the user's approval was the
    # original hidden 617-price-path preflight rerun.
    if not keys and "RUN" in decisions.values():
        keys = _read_only_score_keys(
            definition, project_root=root, contracts=contracts, plan=plan,
            evaluation_end=end, cutoff=str(settings.end_date),
        )
        if len(set(keys.values())) != 1:
            raise OracleBlockedError("Oracle first-run score sources have unequal eligibility")
    common_key = next(iter(keys.values())) if keys else None
    if common_key is None:
        raise OracleBlockedError("Oracle completed Replay missing score-universe evidence")
    # The same immutable matrix governs both the Target producer and Portfolio:
    # a missing arm never authorizes rebuilding any other (seed, target) cell.
    newly_completed = _execute_single_stage_missing(
        definition, project_root=root, quiet=quiet, plan=plan,
        contracts=contracts, current_keys=keys, settings=settings,
        robustness=robustness, price_sha=price_sha, prepared=prepared,
        decisions=decisions, preflight_cached=cached, start=start, end=end,
        base_arm=base_arm, exact_arm=exact_arm, direct_arm=direct_arm,
        base_policy=baseline_policy, cache=cache,
        allow_historical_execution_input_drift=allow_historical_execution_input_drift,
        metric_backfill=metric_backfill,
    )
    for key, record in newly_completed.items():
        if decisions[key] != "RUN" or key in cached:
            raise OracleBlockedError("Oracle Producer escaped the certified RUN cells")
        cached[key] = record
    observations: dict[str, list[dict[str, Any]]] = {target: [] for target in targets}
    for seed in seed_set:
        for target in targets:
            if (seed, target) not in cached:
                raise OracleBlockedError(f"Oracle planned {target} seed={seed} has no completed observation")
            observations[target].append(dict(cached[(seed, target)]))
    baseline_by_seed = {int(row["seed"]): float(row["romd"])
                        for row in observations["C96"]}
    for name in targets[1:]:
        for row in observations[name]:
            seed = int(row["seed"])
            if not math.isclose(float(row["delta_romd_vs_c96"]),
                                float(row["romd"]) - baseline_by_seed[seed],
                                rel_tol=1e-9, abs_tol=1e-9):
                raise OracleBlockedError(f"Oracle {name} seed={seed} C96 mismatch")
    ref: dict[str, Any] = {}
    cohort = {
            "cohort_rows": common_key[1], "first_common_day": common_key[2],
            "last_common_day": common_key[3],
            "research_cutoff": str(settings.end_date),
            "risk_budget_return": float(StrategyAlignedContinuousTargetSpec.from_label_policy(
                DEFAULT_LABEL_POLICY).risk_budget_return),
            "oracle_parity_implementation": ORACLE_PARITY_IMPLEMENTATION,
            "oracle_only": True,
            "evaluation_window_policy": "full_research_oos_to_research_cutoff",
            "target_truth_closeout_policy": "min_natural_exit_horizon_research_cutoff",
        }
    baseline = _aggregate(observations["C96"], target="C96", period_start=start, period_end=end)
    oracles = []
    for name in targets[1:]:
        item = _aggregate(observations[name], target=name, period_start=start, period_end=end)
        diffs = np.array([row["delta_romd_vs_c96"] for row in observations[name]], dtype=float)
        item.update(delta_romd_vs_c96=float(np.mean(diffs)),
                    delta_romd_std=float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                    positive_seed_count=int((diffs > 0).sum()))
        oracles.append(item)
    result = {**ref, "start_date": start, "end_date": end,
              "common_cohort": cohort, "baseline": baseline, "oracles": oracles,
              "definitions": dict(plan["targets"]), "param_artifacts": param_evidence,
              "seed_target_evidence": {}}
    return _oracle_report_envelope(settings=settings, robustness=robustness,
                                   seeds=seed_set, base=result)


def preview_oracle_replay_plan(
    definition, *, project_root: Path,
) -> dict[str, Any]:
    """One read-only Target x Seed plan; pending P16 never hides the Targets.

    A missing P16 is only a provisional RUN. All available P16 and Replay
    receipts are checked by the same canonical code as the execution plan.
    No Target truth, Optimizer or Portfolio Replay may execute here.
    """
    return _run_single_stage_oracle(
        definition, project_root=project_root, quiet=True,
        preflight_approval=None, preview_only=True,
        allow_historical_execution_input_drift=True,
    )


def build_report(
    definition, *, project_root: Path, quiet: bool = False,
    preflight_display: str = "full",
    preflight_approval: Callable[[str], bool] | None = None,
    approved_replay_decisions: Mapping[tuple[int, str], str] | None = None,
) -> dict[str, Any]:
    """One Research [1] work plan, one approval, zero science before approval.

    UI and explicit noninteractive run commands share this execution owner.
    Only the UI approval grants the existing historical-input rebuild policy;
    noninteractive execution retains the cache owner's strict input checks.
    """
    if preflight_display not in {"full", "summary"}:
        raise ValueError("未知 Target Replay preflight 顯示方式")
    return _run_single_stage_oracle(
        definition, project_root=Path(project_root), quiet=quiet,
        preflight_approval=preflight_approval,
        preflight_display=preflight_display,
        allow_historical_execution_input_drift=preflight_approval is not None,
        approved_replay_decisions=approved_replay_decisions,
    )


def render_result(report: Mapping[str, Any], *, target: str) -> str:
    # Persistent report columns/order/precision are owned by its approved contract.
    contract = report_contract("audit.target_oracle_romd")
    sections = contract.sections
    perf_table = sections[0].tables[0]
    yearly_table = sections[2].tables[0]
    paired_table = sections[3].tables[0]
    baseline = dict(report["baseline"])
    items = [dict(baseline, target="C96 Same-Window", wins="-"),
             *[dict(item, target=_oracle_display_name(item["target"]), wins=f'{item["positive_seed_count"]}/{item["seeds"]}')
               for item in report["oracles"]]]
    # Presentation-only sort; preserve the canonical comparison/Replay order.
    items.sort(key=lambda row: (float(row["romd"]), str(row["target"])))
    rows = format_comparison_rows(perf_table, items, target=target)
    intro = [
        "Target Ground Truth -> Strategy RoMD (HINDSIGHT ORACLE / NOT PIT)",
        f'Period: {report["start_date"]} ~ {report["end_date"]} '
        '(fixed evaluation window; MinTrade research-cutoff closeout)',
        f'Common stock-day cohort: {report["common_cohort"]["cohort_rows"]:,}',
        f'Benchmark: {report["benchmark_id"]}, same-seed pairs N={len(report["seeds"])}',
        f'Report schema: {report["schema_version"]}, Oracle treatments: '
        f'{len(report["oracles"])} (Exact + Direct); not Standard SOP',
        'Ranking: Exact K/R0 or Direct No-K/No-R0; same Broker-Cash, seed/P16 and future scores.',
        'Presentation: ascending RoMD Mean (lowest to highest; Target name breaks ties).',
        'T40: (1 - min(MAE40, 1R)) * R40 - min(MAE40, 1R); MAE40 >= 1R -> T40 = -1R.',
        'T40-MFE: (1 - min(MAE40, 1R)) * MFE40_R - min(MAE40, 1R); MAE40 >= 1R -> -1R.',
        'CG = equal-rank 50/50 geometry PROXY, not a neural-network model head.',
    ]
    if target == "console":
        return "\n\n".join((
            render_title(intro[0]), "\n".join(intro[1:]),
            render_table(perf_table.headers, rows),
            "NOTICE: Future labels are hindsight only; not legal for PIT/Trading.",
        ))
    lines = [f"# {intro[0]}", *[f"- {line}" for line in intro[1:]], "",
             f"## {sections[0].display_title}",
             "| " + " | ".join(perf_table.headers) + " |",
             "|" + "|".join("---" for _ in perf_table.headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(["", f"## {sections[1].display_title}",
                  *[f"- **{_oracle_display_name(key)}** - {desc.replace('D40', 'MAE40')}"
                    for key, desc in report["definitions"].items()],
                  "", f"## {sections[2].display_title}"])
    all_years = [year for item in items for year in item["yearly"]]
    year_cells = iter(format_comparison_rows(
        yearly_table, all_years, target=target,
        groups=[str(year.get("year")) for year in all_years],
    ))
    for item in items:
        lines.append(f'### {_oracle_display_name(item["target"])}')
        lines.extend(("| " + " | ".join(yearly_table.headers) + " |",
                      "|" + "|".join("---" for _ in yearly_table.headers) + "|"))
        for _year in item["yearly"]:
            lines.append("| " + " | ".join(next(year_cells)) + " |")
    lines.extend(["", f"## {sections[3].display_title}",
                  "| " + " | ".join(paired_table.headers) + " |",
                  "|" + "|".join("---" for _ in paired_table.headers) + "|"])
    baselines = {int(x["seed"]): float(x["romd"]) for x in baseline["seed_observations"]}
    paired_rows = [
        dict(obs, target=_oracle_display_name(item["target"]), c96_romd=baselines[int(obs["seed"])])
        for item in items if item["target"] != "C96 Same-Window"
        for obs in item["seed_observations"]
    ]
    for paired_cells in format_comparison_rows(
        paired_table, paired_rows, target=target,
        groups=[int(row["seed"]) for row in paired_rows],
    ):
        lines.append("| " + " | ".join(paired_cells) + " |")
    lines.extend(["", f"## {sections[4].display_title}",
                  *[f"- {limitation}" for limitation in report["limitations"]]])
    return "\n".join(lines)


def run_formal_audit(
    definition, *, project_root: Path, quiet: bool = False,
    preflight_display: str = "full",
    preflight_approval: Callable[[str], bool] | None = None,
    approved_replay_decisions: Mapping[tuple[int, str], str] | None = None,
) -> dict[str, Any]:
    # Authorization, frozen schema, and historical snapshot must be checked
    # before initiating the costly hindsight-only portfolio replays.
    approved = require_approved_report_contract("audit.target_oracle_romd")
    payload = dict(build_report(
        definition, project_root=project_root, quiet=quiet,
        preflight_display=preflight_display,
        preflight_approval=preflight_approval,
        approved_replay_decisions=approved_replay_decisions,
    ))
    if payload.get("status") == "CANCELLED":
        return payload
    payload["report_contract_id"] = f"{approved.report_id}@v{approved.version}"
    payload["report_contract_fingerprint"] = persistent_report_contract_fingerprint(approved.report_id)
    console = render_result(payload, target="console")
    markdown = render_result(payload, target="markdown")
    adapter = SimpleNamespace(
        report_id=definition.audit_id, report_type=definition.audit_type,
        output_subdir=definition.output_subdir,
    )
    saved = persist_reusable_report(
        definition=adapter, project_root=project_root,
        payload=payload, console_text=console, markdown_text=markdown,
    )
    if not quiet:
        print(console)
    return saved


__all__ = [
    "SUPPORTED_AUDIT_TYPE", "OracleBlockedError",
    "collect_status", "render_replay_preflight",
    "build_report", "render_result", "run_formal_audit",
]
