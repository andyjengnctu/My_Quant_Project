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

import numpy as np
import pandas as pd

from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.console_report import render_table, render_title
from core.report_style import styled_workflow_status
from core.dataset_profiles import get_dataset_dir
from core.path_utils import project_relative_display_path
from core.strategy_compare_policy import (
    get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
)
from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from filters.breakout_quality.continuous_target import (
    StrategyAlignedContinuousTargetSpec,
    full_horizon_mae_safety_r_from_minimum_low_return,
    full_horizon_mae_gated_reward_r,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.runtime import audit_oracle_truth_score_context
from filters.breakout_quality.ranking_score_store import (
    load_selection_point_in_time_ranking_contract,
)
from core.file_integrity import compute_file_sha256
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
from services.audit.oracle_replay_artifacts import (
    HistoricalOracleIndex, OracleReplayStore, price_inventory_sha256, replay_identity,
)
from services.optimizer.strategy_param_service import (
    inspect_fixed_multiseed_robustness_strategy_parameter_artifact,
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

SUPPORTED_AUDIT_TYPE = "target_ground_truth_strategy_romd"
ORACLE_REPORT_SCHEMA_VERSION = 8
# Engineering bug-fix marker: older price-universe / pre-PIT-start replays are invalid.
# This is not a new scientific Target, Strategy arm, Report version or research choice.
ORACLE_PARITY_IMPLEMENTATION = "original_oracle_contract_parity_bugfix_20261026_scores_sha_pit_v2"
PERIOD_STATS_IMPLEMENTATION = "full_year_calendar_verified_20261026"
# Current report's Oracle strategies. Direct is the existing canonical No-K/No-R0
# selector applied to the SAME score truth; not a new Target formula.
# For backward-compatible external call sites only. The runtime reads the
# actual comparison inventory from the resolved Audit definition on each run.
from core.audit_registry import AUDIT_REUSABLE_REPORTS

_ORACLE_DEFAULT_COMPARISONS = AUDIT_REUSABLE_REPORTS["breakout_quality"]["target_oracle_romd"]["source"]["oracle_comparisons"]
ORACLE_TARGETS = tuple((item["target"], item["description"]) for item in _ORACLE_DEFAULT_COMPARISONS)

# Recipe implementation is stable; WHICH recipes and providers are needed is
# resolved from the selected comparisons, never from a second target allowlist.
_ORACLE_SCORE_RECIPES = frozenset(("equal_rank_mfe_safety", "mfe", "terminal_r", "terminal_mae_gate", "mfe_mae_gate"))
_ORACLE_DAILY_SCORE_RECIPES = frozenset(("daily_trade_path_net_r",))
_PROVIDER_COHORT_IDS = {
    "pure_mfe40_provider": "40d", "mfe20_d20_provider": "20d",
    "first_breach_provider": "first_breach",
}


def _oracle_comparison_plan(definition) -> dict[str, Any]:
    """Canonical compare inventory -> score requirements -> provider graph."""
    # Preserve the existing Audit catalogue score representation, but this
    # catalogue alone does not prove a unique perfect DL model output.
    # _resolve_oracle_target_score_contracts certifies that separately.
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
    daily_recipes: dict[str, tuple[str, str]] = {}
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
        if selector not in {"exact", "direct"}:
            raise OracleBlockedError(f"Oracle comparison selector未支援：{name} {selector}")
        # Scientific target names are immutable. A previous name cannot be
        # reused for a different recipe or score source by changing the list.
        historical = next((item for item in _ORACLE_DEFAULT_COMPARISONS
                           if item["target"] == name), None)
        if historical is not None and any(
                str(entry.get(key) or "") != str(historical.get(key) or "")
                for key in ("selector", "score", "provider", "score_source")):
            raise OracleBlockedError(
                f"Oracle {name} 的score/selector身分已變更；須建立新Target科學ID，禁止沿用舊快取"
            )
        if entry.get("score_source"):
            source = str(entry["score_source"])
            if (source not in recipes and source not in daily_recipes) or selector != "direct":
                raise OracleBlockedError(f"Oracle Direct score來源必須先由Exact宣告：{name} -> {source}")
            aliases[name] = source
        else:
            recipe, provider = str(entry.get("score") or ""), str(entry.get("provider") or "")
            if not definition.source.get(provider):
                raise OracleBlockedError(f"Oracle {name} score/provider缺少正式定義：{recipe}/{provider}")
            if selector != "exact":
                raise OracleBlockedError(f"Oracle {name} 的新分數必須使用Exact；Direct僅能重用score來源")
            if recipe in _ORACLE_DAILY_SCORE_RECIPES:
                if provider != "daily_trade40_profile":
                    raise OracleBlockedError(f"Oracle {name} Daily-universal provider is not canonical: {provider}")
                daily_recipes[name] = (recipe, provider)
            elif recipe in _ORACLE_SCORE_RECIPES:
                recipes[name] = (recipe, provider)
                if provider not in provider_fields:
                    provider_fields.append(provider)
            else:
                raise OracleBlockedError(f"Oracle未授權的score recipe：{recipe}")
        definitions.append((name, desc))
        selectors[name] = selector
    daily_sources = {name: name for name in daily_recipes}
    daily_sources.update({name: source for name, source in aliases.items()
                          if source in daily_recipes})
    if not recipes:
        raise OracleBlockedError("Oracle comparison缺少獨立真值score")
    return {"targets": tuple(definitions), "recipes": recipes,
            "daily_recipes": daily_recipes, "daily_sources": daily_sources,
            "aliases": aliases,
            "selectors": selectors, "providers": tuple(provider_fields)}


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
        if not math.isfinite(val):
            raise OracleBlockedError("Oracle perfect-score map contains nonfinite values")
        digest.update(str(ticker).encode("utf-8") + b"\x00")
        digest.update(str(day).encode("ascii") + b"\x00")
        digest.update(struct.pack("!d", val))
    return digest.hexdigest()


def _score_map_fingerprints(
    plan: Mapping[str, Any],
    *, price_scores: Mapping[str, Mapping[tuple[str, str], float]],
    daily_scores: Mapping[int, Mapping[str, Mapping[tuple[str, str], float]]],
    seeds: list[int],
) -> dict[int, dict[str, str]]:
    """Hash once per unique source; aliases reuse the EXACT injected value."""
    for name, source in plan["aliases"].items():
        if name in price_scores and source in price_scores and price_scores[name] != price_scores[source]:
            raise OracleBlockedError("Oracle Direct score alias differs from canonical score source")
    price_shas = {name: _canonical_score_map_sha256(scores)
                  for name, scores in price_scores.items() if name not in plan["aliases"]}
    daily_shas = {
        seed: {
            name: _canonical_score_map_sha256(score_map)
            for name, score_map in table.items()
            if name not in plan["aliases"]
        }
        for seed, table in daily_scores.items()
    }
    return {
        seed: {
            name: (daily_shas[seed][plan["daily_sources"][name]]
                   if name in plan["daily_sources"]
                   else price_shas[plan["aliases"].get(name, name)])
            for name, _desc in plan["targets"]
        }
        for seed in seeds
    }


def _oracle_display_name(name: str) -> str:
    return ORACLE_DISPLAY_NAMES.get(str(name), str(name))


ORACLE_DIRECT_SCORE_SOURCES = {
    item["target"]: item["score_source"]
    for item in _ORACLE_DEFAULT_COMPARISONS if item.get("score_source")
}


class OracleBlockedError(RuntimeError):
    """Insufficient or incompatible historical truth; do not publish a result."""


def _contract(definition) -> tuple[Any, Any, Any]:
    """Resolve the currently enabled canonical Research selector, not a retired C100.

    The historical Audit settings still name C100/C101 for provenance; they
    cannot authorize selecting an inactive DL model as the current reference.
    """
    source = definition.source
    if definition.audit_type != SUPPORTED_AUDIT_TYPE:
        raise ValueError("Oracle Audit identity不一致")
    settings = get_strategy_comparison_settings(str(source["strategy_profile_id"]))
    base = settings.arms[str(source["baseline_arm_id"])]
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
            or base.rule_policy != "all_off" or not exact_candidates
            or not settings.start_date or not settings.end_date
            or settings.dataset != "full" or settings.rotation != "off"):
        raise OracleBlockedError("CONTRACT_UNRESOLVED: Research DL active Exact/Min P16 selector不唯一或缺失")
    canonical = exact_candidates[0]
    if any((arm.param_source, arm.param_policy, arm.rule_policy,
            arm.dl_runtime_mode, dict(arm.dl_runtime_options or {})) !=
           (canonical.param_source, canonical.param_policy, canonical.rule_policy,
            canonical.dl_runtime_mode, dict(canonical.dl_runtime_options or {}))
           for arm in exact_candidates[1:]):
        raise OracleBlockedError("CONTRACT_UNRESOLVED: 啟用的Research Exact arms不是同一選股契約")
    _direct_arm(definition, settings, base, canonical)
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


# The label identifiers below are existing Target identities, not new scientific
# contracts. An Oracle formula must match an actual Research model-output Target
# before one can assert that it is that model's perfect prediction.
_ORACLE_RECIPE_TARGET_IDS = {
    "mfe": "daily_full_horizon_pure_mfe_r_v1",
    "terminal_r": "daily_terminal_r40_v1",
    "terminal_mae_gate": "daily_terminal_mae_gated_t40_v1",
    "mfe_mae_gate": "daily_full_horizon_mae_gated_mfe_t40_v1",
    "daily_trade_path_net_r": "daily_universal_counterfactual_mintrade40_netr_v1",
}


def _resolve_oracle_target_score_contracts(definition, settings, plan) -> dict[str, dict[str, Any]]:
    """Prove a perfect-score numerical mapping from an enabled Research DL arm.

    Raw pairwise logits identify an ORDER, not a unique Exact score SUM.
    Never legitimize an arbitrary percentile as Exact's unique optimum. A
    direct selector is rank-only, but its model Target and head must still be
    identified; no target receives an invented PIT model identity.
    """
    eligible = []
    for arm in settings.enabled_arms:
        if not arm.dl_enabled:
            continue
        source = settings.dl_sources[str(arm.dl_id)]
        profile = get_breakout_quality_experiment_profile(source.experiment_profile)
        if str(source.score_source) != "selection_point_in_time":
            continue
        eligible.append((arm, source, profile))
    result = {}
    issues = []
    for name, _description in plan["targets"]:
        recipe_name = plan["aliases"].get(name, name)
        if recipe_name in plan["recipes"]:
            recipe = plan["recipes"][recipe_name][0]
        else:
            recipe = plan["daily_recipes"][recipe_name][0]
        target_id = _ORACLE_RECIPE_TARGET_IDS.get(recipe)
        if not target_id:
            issues.append(f"{name}: {recipe} 是幾何PROXY，沒有對應的DL perfect Head")
            continue
        # No multi-head fusion, loss or postprocessing is guessed from a
        # coincidentally identical label ID. Existing scalar head semantics only.
        matches = [(arm, source, profile) for arm, source, profile in eligible if (
            profile.continuous_target_id == target_id
            and profile.training_objective in {"daily_pairwise_ranking", "daily_percentile_regression"}
        )]
        if not matches:
            issues.append(f"{name}: Target {target_id} 沒有啟用且具唯一scalar score contract的Research DL arm")
            continue
        contracts = {(source.experiment_profile, profile.portfolio_score_scale,
                      profile.training_objective) for _arm, source, profile in matches}
        if len(contracts) != 1:
            issues.append(f"{name}: 多個啟用DL arm的score/head contract不一致")
            continue
        arm, source, profile = matches[0]
        selector = plan["selectors"][name]
        objective = str(profile.training_objective)
        scale = str(profile.portfolio_score_scale)
        if scale == "daily_percentile":
            mapping = "canonical_daily_target_percentile"
        elif objective == "daily_percentile_regression" and scale == "raw":
            mapping = "canonical_supervised_label_percentile"
        elif selector == "direct" and objective == "daily_pairwise_ranking" and scale == "raw":
            # In the canonical Direct selector, only relative rank matters.
            # Scores are represented as percentiles solely for deterministic
            # ordering, NOT as a claim about pairwise numerical logits.
            mapping = "canonical_rank_only_direct"
        else:
            issues.append(
                f"{name}: {arm.arm_id} 使用 {objective}/{scale}；"
                "raw pairwise logit沒有唯一完美數值，Exact不能假定為每日百分位"
            )
            continue
        result[name] = {
            "reference_arm_id": str(arm.arm_id),
            "dl_source_id": str(arm.dl_id),
            "profile_id": str(source.experiment_profile),
            "model_architecture": str(source.model_architecture),
            "target_id": target_id,
            "model_objective": objective, "portfolio_score_scale": scale,
            "perfect_score_mapping": mapping,
            "score_start_date": str(source.point_in_time_score_start_date or ""),
            "score_source": str(source.score_source),
        }
    if issues:
        raise OracleBlockedError(
            "CONTRACT_UNRESOLVED: Oracle/Research DL score parity:\n- " + "\n- ".join(issues)
        )
    return result



def _robustness_artifacts(definition, root: Path):
    settings, baseline, exact = _contract(definition)
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
        raise OracleBlockedError("Oracle Benchmark無法對齊canonical 4-seed P16訓練合約")
    training_profile = str(dict(builder.options).get("training_profile_id") or "")
    if not training_profile:
        raise OracleBlockedError("P16缺少canonical training profile")
    records = []
    for seed in robustness.resolved_seeds:
        inspection = inspect_fixed_multiseed_robustness_strategy_parameter_artifact(
            root,
            training_profile_id=training_profile,
            benchmark_id=str(robustness.benchmark_id),
            replication_seed=int(seed),
            policy=str(baseline.param_policy),
            comparison_end_date=str(settings.end_date),
            dataset=settings.dataset,
            max_positions=settings.max_positions,
            rotation=settings.rotation,
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
        if not bool(inspection["ready"]):
            raise OracleBlockedError(
                f"Benchmark P16 seed={seed}非READY: {inspection['status']}（只讀Audit不自動重訓）"
            )
        records.append((int(seed), Path(inspection["path"])))
    return robustness, records


def collect_status(definition, *, project_root: Path) -> dict[str, Any]:
    """Read-only 4-seed P16 and daily Target prerequisite preflight."""
    root = Path(project_root).resolve()
    display = "Full Research Dataset + 4-seed canonical P16 + daily target truth"
    try:
        settings, _base, _exact = _contract(definition)
        # Fail before checking large market/P16 artifacts: a scalar perfect DL
        # output not implied by the actual enabled model is not executable.
        _resolve_oracle_target_score_contracts(definition, settings, _oracle_comparison_plan(definition))
        data_dir = Path(get_dataset_dir(str(root), settings.dataset)).resolve()
        if not data_dir.is_dir() or not next(data_dir.rglob("*.csv"), None):
            raise OracleBlockedError("缺少完整研究價格資料（full dataset）")
        from filters.breakout_quality.paths import resolve_filter_output_dir
        truth_root = resolve_filter_output_dir(root, filter_id=str(definition.source["filter_id"]))
        if not truth_root.is_dir():
            raise OracleBlockedError("缺少validated breakout-quality Dataset bundle；Audit不得代替Producer建置")
        robustness, records = _robustness_artifacts(definition, root)
        # No separate DL teacher/optimizer is a precondition for Oracle truth.
        # The 4 verified per-seed Min artifacts above are the ONLY param source.
        return {
            "status": "READY", "reason": "", "source": {
                "display": display, "preparable": False,
                "benchmark": robustness.benchmark_id,
                "seeds": [seed for seed, _path in records],
            },
        }
    except (ValueError, OSError, OracleBlockedError) as exc:
        return {
            "status": "BLOCKED", "reason": str(exc),
            "source": {"display": display, "preparable": False},
        }



def _formal_dl_score_start(settings) -> str:
    """Read the score-coverage start from canonical Research DL sources.

    Never invent a pre-OOS score for the first order. If active Research DL
    sources disagree on availability, the Audit cannot infer a unique legal
    score boundary, and must block instead of choosing its own dates.
    """
    starts = set()
    for arm in settings.enabled_arms:
        if not arm.dl_enabled:
            continue
        source = settings.dl_sources[str(arm.dl_id)]
        if str(source.score_source) != "selection_point_in_time":
            raise OracleBlockedError("Oracle只支援正式Research DL PIT score來源")
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
    terminal_fields = {field for recipe, field in plan["recipes"].values()
                       if recipe in ("terminal_r", "terminal_mae_gate")}
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
    score_start = pd.Timestamp(_formal_dl_score_start(settings))
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
            eligible = positions[positions + maturity < len(frame)]
            if not eligible.size:
                continue
            eligible = eligible[dates.take(eligible + maturity) <= limit]
            if not eligible.size:
                continue
            batch = compute_daily_opportunity_target_batch(
                frame, eligible, spec=spec, target_id=target_id,
            )
            valid = np.asarray(batch.valid_mask, dtype=bool)
            if maturity != int(spec.horizon_bars):
                cohort = compute_daily_opportunity_target_batch(
                    frame, eligible, spec=replace(spec, horizon_bars=maturity),
                    target_id=target_id,
                )
                valid &= np.asarray(cohort.valid_mask, dtype=bool)
            risk = float(spec.risk_budget_return)
            values = np.column_stack((
                np.asarray(batch.favorable_return, dtype=float) / risk,
                np.asarray(batch.adverse_return_to_peak, dtype=float) / risk,
                np.asarray(batch.minimum_low_return, dtype=float),
                np.asarray(batch.target_raw_r, dtype=float),
            ))
            valid &= np.isfinite(values).all(axis=1)
            # R40 uses the *40th future trading bar Close*, not the peak High,
            # fixed-day MAE or realized Portfolio Replay exit. The canonical
            # price path and mature 40D positions are shared with MFE40.
            if field in terminal_fields:
                terminal_r = (closes[eligible + int(spec.horizon_bars)] / closes[eligible] - 1.0) / risk
                valid &= np.isfinite(terminal_r)
            if not valid.any():
                continue
            if field in valid_counts:
                valid_counts[field] += int(valid.sum())
            columns = {
                "ticker": ticker,
                "date": dates.take(eligible[valid]).strftime("%Y-%m-%d"),
                "mfe": values[valid, 0],
                "peak_adverse": values[valid, 1],
                "min_low": values[valid, 2],
                "raw": values[valid, 3],
            }
            if field in terminal_fields:
                columns["terminal_r"] = terminal_r[valid]
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
    """Require the price verifier to match exactly the official DL score cohort."""
    old = source.loc[source["date"] >= score_start]
    check = old.merge(market, on=["ticker", "date"], how="left",
                      suffixes=("_provider", "_market"), validate="one_to_one", indicator=True)
    missing = check["_merge"] == "left_only"
    if missing.any():
        row = check.loc[missing, ["ticker", "date"]].iloc[0]
        raise OracleBlockedError(
            f"{field}官方真值找不到相應Research價格成熟路徑：{row['ticker']} {row['date']}"
        )
    for name in ("mfe", "peak_adverse", "min_low", "raw"):
        if not np.allclose(check[f"{name}_provider"].to_numpy(dtype=float),
                           check[f"{name}_market"].to_numpy(dtype=float),
                           rtol=2e-5, atol=2e-5):
            raise OracleBlockedError(f"{field}的official target與Research價格重建真值不一致：{name}")
    if len(market) != len(old):
        raise OracleBlockedError(f"{field}價格校驗母體不得擴張正式DL母體")
    # Keep terminal close available for R40/T40 while preserving official model
    # sample identity and precision for all other fields.
    matched = old.copy()
    if "terminal_r" in market:
        matched = matched.merge(market[["ticker", "date", "terminal_r"]],
                                on=["ticker", "date"], validate="one_to_one")
    return matched, 0


def _load_truth(definition, root: Path, source_field: str, *, cutoff: str) -> pd.DataFrame:
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
        raise OracleBlockedError(f"canonical target identity漂移：{source_field}")
    group = pd.DataFrame(bundle.group_table).reset_index(drop=True)
    required = {"ticker", "date", "label_eval_end_date", "target_favorable_r", "target_adverse_r", "target_minimum_low_return"}
    if not required.issubset(group):
        raise OracleBlockedError(f"{source_field}缺少官方daily future-path欄位：{sorted(required - set(group))}")
    if len(group) != len(bundle.raw_target) or len(group) != len(bundle.target_valid):
        raise OracleBlockedError(f"{source_field} group/target shape不一致")
    mature = (
        np.asarray(bundle.target_valid, dtype=bool)
        & (pd.to_datetime(group["label_eval_end_date"], errors="raise") <= pd.Timestamp(cutoff)).to_numpy()
    )
    frame = pd.DataFrame({
        "ticker": group["ticker"].map(normalize_ticker),
        "date": pd.to_datetime(group["date"], errors="raise").dt.strftime("%Y-%m-%d"),
        "mfe": pd.to_numeric(group["target_favorable_r"], errors="coerce"),
        "peak_adverse": pd.to_numeric(group["target_adverse_r"], errors="coerce"),
        "min_low": pd.to_numeric(group["target_minimum_low_return"], errors="coerce"),
        "raw": np.asarray(bundle.raw_target, dtype=float),
    })
    numeric = frame[["mfe", "peak_adverse", "min_low", "raw"]].to_numpy(dtype=float)
    mature &= np.isfinite(numeric).all(axis=1)
    frame = frame.loc[mature].copy()
    if frame.empty:
        raise OracleBlockedError(f"{source_field}沒有研究截止日前成熟且完整的共同樣本")
    if frame.duplicated(["ticker", "date"]).any():
        raise OracleBlockedError(f"{source_field}存在ticker/date重複")
    return frame


def _terminal_mae_gate_r(r40: pd.Series, safety_mae40: pd.Series, *, horizon: int = 40) -> pd.Series:
    """T_h = (1-m)*R_h-m, m=clip(MAE_h_R,0,1); at 1R stop T_h=-1R.

    Oracle ranking label only: NOT a historical cash execution or ATR stop.
    Full-horizon MAE is nonnegative; Safety is its negative signed form.
    """
    terminal = np.asarray(r40, dtype=np.float64)
    safety = np.asarray(safety_mae40, dtype=np.float64)
    if (terminal.shape != safety.shape or not np.isfinite(terminal).all()
            or not np.isfinite(safety).all() or (safety > 0.0).any()):
        raise OracleBlockedError(
            f"T{horizon} requires finite R{horizon} and nonnegative MAE{horizon} in canonical R"
        )
    return pd.Series(full_horizon_mae_gated_reward_r(terminal, safety), index=r40.index, dtype=float)


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

        if recipe == "mfe":
            value = col("mfe")
        elif recipe == "terminal_r":
            value = col("terminal_r")
        else:
            safety = safety_by_provider.get(provider)
            if safety is None:
                safety = full_horizon_mae_safety_r_from_minimum_low_return(
                    col("min_low").to_numpy(dtype=float), risk_budget_return=risk,
                )
                safety = pd.Series(safety, index=joint.index)
                if not np.isfinite(safety.to_numpy(dtype=float)).all():
                    raise OracleBlockedError(f"Oracle {provider} MAE safety包含不合法值")
                safety_by_provider[provider] = safety
            if recipe == "equal_rank_mfe_safety":
                value = (percentile(col("mfe")) + percentile(safety)) / 2
            elif recipe == "terminal_mae_gate":
                value = _terminal_mae_gate_r(col("terminal_r"), safety)
            elif recipe == "mfe_mae_gate":
                value = _terminal_mae_gate_r(col("mfe"), safety)
            else:
                raise OracleBlockedError(f"Oracle未授權的score recipe：{recipe}")
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
    if set(scores) | set(plan["daily_sources"]) != {name for name, _desc in plan["targets"]}:
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


def build_oracle_score_tables(definition, *, project_root: Path, cutoff: str) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, Any]]:
    """Selected-score interface; provider selection is owned by the comparison plan."""
    return _build_oracle_score_panels(
        definition, project_root=project_root, cutoff=cutoff,
    )


def _certify_oracle_pit_score_coverage(
    definition, *, project_root: Path, contracts: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any], price_scores: Mapping[str, Mapping[tuple[str, str], float]],
    daily_scores: Mapping[int, Mapping[str, Mapping[tuple[str, str], float]]],
) -> dict[str, dict[str, Any]]:
    """Certify every injected key actually has an eligible formal PIT model row.

    Uses the existing PIT score manifest, SHA and FIT/fold provenance validators,
    not a second Oracle interpretation of model inference eligibility. Past
    model SCORE values are not used as perfect predictions. No FIT is triggered.
    """
    root = Path(project_root).resolve()
    validated: dict[tuple[str, str, str], tuple[set[tuple[str, str]], dict[str, str]]] = {}
    certified = {}
    for name, contract in contracts.items():
        identity = (str(definition.source["filter_id"]), str(contract["model_architecture"]),
                    str(contract["profile_id"]))
        if identity not in validated:
            try:
                # Same validated PIT ranking source the real DL selector uses,
                # including its model-validation PASS gate and FIT/fold lineage.
                pit = load_selection_point_in_time_ranking_contract(root, *identity)
                if str(pit.continuous_target_id) != str(contract["target_id"]):
                    raise ValueError("PIT model Target identity does not match Oracle Target")
                table = pd.read_csv(
                    pit.score_path, usecols=["ticker", "date", "fold_id", "model_information_cutoff"],
                    dtype={"ticker": str, "date": str, "fold_id": str,
                           "model_information_cutoff": str},
                )
                if table.empty or table.isna().any().any() or table.duplicated(["ticker", "date"]).any():
                    raise ValueError("PIT physical score coverage missing/duplicate")
                table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
                table["model_information_cutoff"] = pd.to_datetime(
                    table["model_information_cutoff"], errors="raise"
                ).dt.strftime("%Y-%m-%d")
                if (table["model_information_cutoff"] >= table["date"]).any():
                    raise ValueError("PIT score row has an illegal information cutoff")
                if ((table["date"] < str(pit.available_from)).any()
                        or (table["date"] > str(pit.available_through)).any()):
                    raise ValueError("PIT score date outside verified manifest")
                keys = set(zip(table["ticker"].astype(str), table["date"].astype(str)))
                digest = hashlib.sha256()
                for row in table.sort_values(["ticker", "date"], kind="stable").itertuples(index=False):
                    digest.update("|".join(str(value) for value in row).encode("utf-8") + b"\n")
                evidence = {
                    "pit_score_start": str(pit.available_from),
                    "pit_score_end": str(pit.available_through),
                    "pit_eligible_keys_sha256": hashlib.sha256(
                        "".join(f"{t}|{d}\n" for t, d in sorted(keys)).encode("utf-8")
                    ).hexdigest(),
                    "pit_fold_coverage_sha256": digest.hexdigest(),
                    "pit_score_manifest_sha256": compute_file_sha256(pit.manifest_path),
                    "pit_coverage_manifest_sha256": compute_file_sha256(pit.coverage_path),
                }
                validated[identity] = keys, evidence
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise OracleBlockedError(
                    f"PIT_SCORE_EVIDENCE_BLOCKED: {identity[2]}: {type(exc).__name__}: {exc}"
                ) from exc
        keys, evidence = validated[identity]
        if evidence["pit_score_start"] != str(contract["score_start_date"]):
            raise OracleBlockedError(f"{name}: PIT manifest score start differs from Research DL arm")
        if name in plan["daily_sources"]:
            source_name = plan["daily_sources"][name]
            candidates = {key for per_seed in daily_scores.values() for key in per_seed[source_name]}
        else:
            candidates = set(price_scores[name])
        absent = candidates - keys
        if absent:
            example = sorted(absent)[0]
            raise OracleBlockedError(
                f"{name}: 正式PIT score fold未覆蓋{len(absent):,}個Oracle可評分股票日；example={example}"
            )
        certified[name] = {**dict(contract), **evidence,
                           "oracle_scored_rows": len(candidates)}
    return certified


def _metrics(replay: dict[str, Any], *, target: str) -> dict[str, Any]:
    summary = _scenario_summary(replay)
    required = ("total_return_pct", "max_drawdown_pct", "return_over_max_drawdown", "expected_value_r", "trade_count", "avg_exposure_pct")
    if any(not math.isfinite(float(summary[name])) for name in (*required, "annual_return_pct")):
        raise OracleBlockedError(f"Oracle {target}缺少有效portfolio performance metrics")
    yearly = _yearly_frame(replay["profile"], "strategy")
    return {
        "target": target,
        "total_return_pct": float(summary["total_return_pct"]),
        "annualized_return_pct": float(summary["annual_return_pct"]),
        "mdd_pct": float(summary["max_drawdown_pct"]),
        "romd": float(summary["return_over_max_drawdown"]),
        "ev_a_r": float(summary["expected_value_r"]),
        "trades": int(summary["trade_count"]),
        "exposure_pct": float(summary["avg_exposure_pct"]),
        "yearly": _to_json_native(yearly.to_dict("records")),
    }


def _aggregate(records: list[dict[str, Any]], *, target: str) -> dict[str, Any]:
    if not records:
        raise OracleBlockedError("Oracle robustness missing seed observations")
    summary = {"target": target, "seeds": len(records)}
    fields = (
        "total_return_pct", "annualized_return_pct", "mdd_pct", "romd", "ev_a_r", "trades", "exposure_pct",
    )
    for field in fields:
        values = np.asarray([float(x[field]) for x in records], dtype=float)
        summary[field] = float(np.mean(values))
        summary[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    yearly = {}
    for row in records:
        for year_row in row["yearly"]:
            year = str(year_row["year"])
            yearly.setdefault(year, []).append(year_row)
    summary["yearly"] = [
        {
            "year": year,
            "strategy_return_pct": float(np.mean([float(z["strategy_return_pct"]) for z in values])),
            "n_seeds": len(values),
            "is_full_year": all(bool(z.get("is_full_year", False)) for z in values),
        }
        for year, values in sorted(yearly.items())
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
    """Read-only full-workload plan before any Broker-Cash replay starts.

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
        f"[Target 驗證 {label} | Replay preflight（執行前）]",
        "Seed：" + "  ".join(f"S{i + 1}={seed}" for i, seed in enumerate(seeds)),
        render_table(headers, rows),
        f"REUSE {counts['REUSE']} / RUN（需計算）{counts['RUN']} / "
        f"BLOCKED {counts['BLOCKED']}；本階段尚未執行任何 Replay",
    ))


def _existing_artifact_inventory(definition, *, project_root: Path) -> tuple[str, dict[tuple[int, str], str]]:
    """Show cached evidence before any market truth or P16 materialization.

    This is ONLY an existence/integrity inventory, NOT a replay dependency
    planner.  The full preflight in build_report() remains the sole authority
    for canonical REUSE/RUN/BLOCKED decisions and Replay authorization.
    """
    settings, _base, _exact = _contract(definition)
    _resolve_oracle_target_score_contracts(definition, settings, _oracle_comparison_plan(definition))
    benchmark = get_strategy_multi_seed_robustness_settings(settings.profile_id)
    seeds = tuple(int(seed) for seed in benchmark.resolved_seeds)
    if (not benchmark.enabled or not seeds or len(seeds) != benchmark.seed_count
            or len(set(seeds)) != len(seeds)):
        raise OracleBlockedError("Target benchmark Seed 設定不完整；無法列出候選工件")
    targets = (str(definition.source["baseline_arm_id"]),
               *(name for name, _description in _oracle_comparison_plan(definition)["targets"]))
    root = Path(project_root).resolve()
    try:
        scope = {
            "dataset": settings.dataset, "start_date": str(settings.start_date),
            "requested_end": str(settings.end_date), "benchmark": benchmark.benchmark_id,
        }
        per_arm = OracleReplayStore(root, definition.output_subdir).preview_candidates(
            seeds=seeds, targets=targets, **scope,
            required_oracle_revision=ORACLE_PARITY_IMPLEMENTATION,
        )
        historical = HistoricalOracleIndex(root).preview_candidates(
            seeds=seeds, targets=targets, **scope,
            required_oracle_revision=ORACLE_PARITY_IMPLEMENTATION,
            strategy_profile_id=settings.profile_id,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OracleBlockedError(f"Target 歷史工件盤點無法完成：{exc}") from exc
    actions: dict[tuple[int, str], str] = {}
    for key, presence in per_arm.items():
        if presence == "INVALID":
            actions[key] = "BLOCKED"
        elif presence in {"CACHE", "MULTIPLE"} or key in historical:
            actions[key] = "REUSE"
        else:
            actions[key] = "RUN"
    headers = ("Target", *(f"S{i + 1}" for i in range(len(seeds))))
    # AI: Both the fast inventory and the strict preflight use the shared
    # workflow palette (REUSE green, BLOCKED red, RUN neutral). No local ANSI
    # rules or duplicate status classification belong in the Oracle renderer.
    rows = [
        (target, *(styled_workflow_status(actions[(seed, target)]) for seed in seeds))
        for target in targets
    ]
    counts = {state: list(actions.values()).count(state)
              for state in ("REUSE", "RUN", "BLOCKED")}
    inventory = "\n".join((
        "[Target 驗證 | 既有 Replay 工件]",
        "Seed：" + "  ".join(f"S{i + 1}={seed}" for i, seed in enumerate(seeds)),
        render_table(headers, rows),
        f"REUSE {counts['REUSE']} / RUN（需計算）{counts['RUN']} / BLOCKED {counts['BLOCKED']}",
        "註：此表僅盤點現行 Oracle 實作版的候選工件；修復前結果不列入 REUSE。正式 REUSE 尚須核對資料、參數、期間及分數身分；Enter 後自動核對並執行必要 Replay。",
    ))
    return inventory, actions


def render_existing_artifact_inventory(definition, *, project_root: Path) -> str:
    """Existing-cache display and quick preliminary plan share one inventory owner."""
    inventory, _decisions = _existing_artifact_inventory(definition, project_root=project_root)
    return inventory


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
                       score_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One target-scoped economic/score identity for fast and normal REUSE."""
    policy = dict(replay_policy)
    policy["period_stats_implementation"] = PERIOD_STATS_IMPLEMENTATION
    if target != str(definition.source["baseline_arm_id"]):
        policy["oracle_parity_implementation"] = ORACLE_PARITY_IMPLEMENTATION
    if plan["selectors"].get(target) == "direct":
        policy["direct_arm"] = str(definition.source["direct_arm_id"])
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
        source = plan["daily_sources"][target]
        recipe, provider = plan["daily_recipes"][source]
        policy["hindsight_score_recipe"] = recipe
        policy["hindsight_score_key_mode"] = "information_date"
        policy["hindsight_target_contract"] = {
            "label_id": daily_evidence["label_id"],
            "target_id": daily_evidence["target_id"],
            "profile_id": str(definition.source[provider]),
            "rank_policy": daily_evidence["rank_policy"],
            "rank_sha256": daily_evidence["rank_sha256"],
            "raw_label_sha256": daily_evidence["raw_label_sha256"],
            "calendar_sha256": daily_evidence["calendar_sha256"],
            "teacher": daily_evidence["teacher"],
            "mature_eligible_rows": daily_evidence["mature_eligible_rows"],
            "score_start_date": daily_evidence["score_start_date"],
            "settled_net_r_rows": daily_evidence["settled_net_r_rows"],
            "censored_rank_last_rows": daily_evidence["censored_rank_last_rows"],
            # Matches the canonical DL lookup: a candidate outside the
            # feature-eligible mature score universe has no score, never 0R.
            # Distinct replay identity prevents accidental historical reuse.
            "missing_key_policy": "unavailable_outside_mature_feature_universe",
            "research_cutoff": daily_evidence["research_cutoff"],
            **({"oracle_evaluation_start_date": daily_evidence["oracle_evaluation_start_date"]}
               if daily_evidence.get("oracle_evaluation_start_date") else {}),
        }
    return policy


def build_report(
    definition, *, project_root: Path, quiet: bool = False,
    preflight_display: str = "full",
) -> dict[str, Any]:
    if preflight_display not in {"full", "summary"}:
        raise ValueError("未知 Target Replay preflight 顯示方式")
    root = Path(project_root).resolve()
    plan = _oracle_comparison_plan(definition)
    settings, base_arm, exact_arm = _contract(definition)
    # Fail before expensive 40D materialization and before any partial Replay.
    # A mapping that cannot be deduced is not a new science identity or a
    # license to emit another unverified "perfect DL" return number.
    resolved_score_contracts = _resolve_oracle_target_score_contracts(definition, settings, plan)
    status = collect_status(definition, project_root=root)
    if status["status"] != "READY":
        raise OracleBlockedError(status["reason"])
    robustness, parameter_paths = _robustness_artifacts(definition, root)
    # The old fast path could certify the report before recomputing the actual
    # perfect-score bytes or proving its formal PIT coverage. E13 forbids that
    # inference. Verified per-arm REUSE remains available after score materialization.
    def progress(message: str) -> None:
        print(f"[Target 驗證｜前置] {message}", flush=True)

    if not quiet:
        progress("建立完整 REUSE/RUN/BLOCKED 計畫（未執行 Replay）")
    # Validate the new trade-path Target BEFORE materializing the legacy price
    # providers or starting any portfolio Replay. Only the approved Audit
    # progress text changes; no report sections/columns or Trading path change.
    daily_seed_scores: dict[int, dict[str, dict[tuple[str, str], float]]] = {}
    daily_seed_evidence: dict[int, dict[str, Any]] = {}
    if plan["daily_recipes"]:
        if len(plan["daily_recipes"]) != 1:
            raise OracleBlockedError("Daily-universal Oracle must have a single Exact score source")
        from filters.breakout_quality.daily_trade40_target import (
            build_formal_min_execution_schedule, resolve_formal_min_param_records,
        )
        from services.audit.daily_trade40_oracle import (
            DailyTrade40OracleBlockedError, build_daily_universal_trade40_oracle_scores,
        )
        source_name, (_recipe, provider) = next(iter(plan["daily_recipes"].items()))
        # Use the identical resolved param view as C96/Min Portfolio.
        prepared_for_scores, _ = _prepare_oracle_parameters(
            definition, root=root, parameter_paths=parameter_paths,
            start=str(settings.start_date), end=str(settings.end_date), base_arm=base_arm,
        )
        for seed, param_sha, kind, base_params, _oracle_params in prepared_for_scores:
            try:
                canonical_records = resolve_formal_min_param_records(kind, base_params)
                schedule, _provenance = build_formal_min_execution_schedule(
                    canonical_records, seed=seed, param_sha256=param_sha,
                )
                daily_scores, evidence = build_daily_universal_trade40_oracle_scores(
                    project_root=root, dataset=str(settings.dataset),
                    filter_id=str(definition.source["filter_id"]),
                    profile_id=str(definition.source[provider]),
                    cutoff=str(settings.end_date), evaluation_start=str(settings.start_date),
                    score_start=_formal_dl_score_start(settings),
                    seed=seed, param_sha256=param_sha, canonical_schedule=schedule,
                    progress_callback=(lambda message: progress(f"seed={seed} {message}")) if not quiet else None,
                )
            except (DailyTrade40OracleBlockedError, ValueError) as exc:
                raise OracleBlockedError(f"Formal Min seed={seed} target blocked: {exc}") from exc
            teacher = dict(evidence.get("teacher") or {})
            if (teacher.get("seed") != seed
                    or teacher.get("param_identity_sha256") != param_sha
                    or evidence.get("profile_id") != str(definition.source[provider])):
                raise OracleBlockedError(f"Daily-universal seed={seed} not based on this formal Min parameter view")
            daily_seed_evidence[int(seed)] = evidence
            daily_seed_scores[int(seed)] = {source_name: daily_scores}
            for alias, original in plan["aliases"].items():
                if original == source_name:
                    daily_seed_scores[int(seed)][alias] = daily_scores
            if not quiet:
                progress(f"Daily-universal formal Min seed={seed}: "
                         f"{evidence['mature_eligible_rows']:,} mature, "
                         f"{evidence['settled_net_r_rows']:,} settled")
        expected_seeds = {seed for seed, _ in parameter_paths}
        if set(daily_seed_scores) != expected_seeds:
            raise OracleBlockedError("Daily-universal per-seed scores incomplete")
        for evidence in daily_seed_evidence.values():
            if (str(evidence["first_date"]) < _formal_dl_score_start(settings)
                    or str(evidence["first_replay_trade_date"]) < str(settings.start_date)):
                raise OracleBlockedError("Daily-universal PIT score coverage boundary mismatch")
    scores40, cohort40 = _build_oracle_score_panels(
        definition, project_root=root, cutoff=str(settings.end_date),
        progress=None if quiet else progress,
    )
    # A formal model's PIT eligibility and fold coverage are separate from the
    # daily feature-universe provider. Verify both BEFORE declaring cache REUSE.
    resolved_score_contracts = _certify_oracle_pit_score_coverage(
        definition, project_root=root, contracts=resolved_score_contracts,
        plan=plan, price_scores=scores40, daily_scores=daily_seed_scores,
    )
    if not quiet:
        progress("真值、PIT fold覆蓋與共同樣本完成；檢查價格 SHA、P16 與 Replay Cache")
    direct_arm = _direct_arm(definition, settings, base_arm, exact_arm)
    start = str(settings.start_date)  # Formal Min Replay start NEVER depends on a new Target.
    end40 = min(str(settings.end_date), str(cohort40["last_common_day"]))
    if end40 <= start:
        raise OracleBlockedError("40D成熟樣本期間不合法")
    data_dir = Path(get_dataset_dir(str(root), settings.dataset)).resolve()
    try:
        price_sha = price_inventory_sha256(root, settings.dataset)
        archived = HistoricalOracleIndex(root)
    except ValueError as exc:
        raise OracleBlockedError(str(exc)) from exc
    cache = OracleReplayStore(root, definition.output_subdir)

    # Replay cache identity is derived from the actual injectible score bytes.
    # The first computation intentionally invalidates historical unverified
    # price-score replays; subsequent runs with identical truth can REUSE.
    score_fingerprints = _score_map_fingerprints(
        plan, price_scores=scores40, daily_scores=daily_seed_scores,
        seeds=[seed for seed, _path in parameter_paths],
    )
    # Existing single mature 40D panel; Exact and Direct consume the same truth.
    # Every treatment carries its own canonical ranking/allocator policy identity.
    def run_panel(*, end: str, cohort: dict[str, Any], scores: dict[str, dict],
                  targets: tuple[tuple[str, str], ...], label: str) -> dict[str, Any]:
        if set(scores) | set(plan["daily_sources"]) != {name for name, _ in targets}:
            raise OracleBlockedError(f"Oracle {label}計算與報表Target清單不一致")
        replay_kwargs = {
            "data_dir": data_dir, "start_date": start, "end_date": end,
            "max_positions": settings.max_positions, "enable_rotation": False, "quiet": quiet,
        }
        prepared, param_evidence = _prepare_oracle_parameters(
            definition, root=root, parameter_paths=parameter_paths,
            start=start, end=end, base_arm=base_arm,
        )
        definitions = dict(targets)
        definitions_with_base = {
            str(definition.source["baseline_arm_id"]): "Canonical same-window Min P16 Base-Finalist-Best",
            **definitions,
        }
        # AI: Replay computation identity is owned by oracle_replay_artifacts.
        # AI: Strategy Compare arm enabled/name/description, comparison membership,
        # AI: source discovery paths and report inventory are NOT replay inputs.
        # AI: Per-seed P16 bytes, score policy, economic rules and price bytes are.
        legacy_context = {
            "benchmark": robustness.benchmark_id,
            "strategy_profile_id": settings.profile_id,
            "dataset": settings.dataset,
            "start_date": start,
            "end_date": end,
            "seeds": [seed for seed, _ in parameter_paths],
            "param_sha_by_seed": {int(item["seed"]): item["param_sha256"] for item in param_evidence},
            "cohort": cohort,
            "definitions": definitions,
        }
        replay_policy = _canonical_oracle_replay_policy(definition, base_arm=base_arm, exact_arm=exact_arm)
        observations: dict[str, list[dict[str, Any]]] = {"C96": []}
        observations.update({target: [] for target, _ in targets})

        # Resolve EVERY completed identity before starting any replay. A local
        # completed report with mismatched evidence is BLOCKED, not interpreted
        # as permission to silently spend hours computing the same arm again.
        # This plan is derived from on-disk artifacts, not a historic target list.
        completed_plan: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any] | None, str]] = {}
        blocked: dict[tuple[int, str], str] = {}
        for seed, param_sha, _kind, _base_params, _oracle_params in prepared:
            for target in ("C96", *(name for name, _ in targets)):
                is_direct = plan["selectors"].get(target) == "direct"
                arm = direct_arm if is_direct else exact_arm
                arm_replay_policy = _arm_replay_policy(
                    definition, plan=plan, target=target, replay_policy=replay_policy,
                    daily_evidence=daily_seed_evidence.get(seed),
                    score_sha256=(score_fingerprints[seed][target] if target != "C96" else None),
                    score_contract=(resolved_score_contracts[target] if target != "C96" else None),
                )
                identity = replay_identity(
                    dataset=settings.dataset, dataset_sha256=price_sha,
                    target=target, definition=definitions_with_base,
                    start=start, end=end,
                    benchmark=str(robustness.benchmark_id), param_sha256=param_sha,
                    seed=seed, cohort=cohort,
                    ranking_policy=str(arm.dl_runtime_mode),
                    ranking_options=dict(arm.dl_runtime_options),
                    evaluation_mode=str(definition.dimensions["evaluation_mode"]),
                    max_positions=settings.max_positions, replay_policy=arm_replay_policy,
                )
                try:
                    record, origin = cache.resolve_completed(identity)
                    completed_plan[(seed, target)] = (identity, record, origin)
                except ValueError as exc:
                    # Continue read-only inspection so the user sees the full
                    # REUSE/RUN/BLOCKED matrix; do not replay a partial cohort.
                    blocked[(seed, target)] = str(exc)
        decisions = {
            (seed, name): "BLOCKED" if (seed, name) in blocked else
            "REUSE" if completed_plan[(seed, name)][1] is not None else "RUN"
            for seed, _path in parameter_paths
            for name in ("C96", *(name for name, _desc in targets))
        }
        preview = render_replay_preflight(
            label=label,
            seeds=[seed for seed, _path in parameter_paths],
            targets=["C96", *(name for name, _desc in targets)],
            decisions=decisions,
        )
        if not quiet:
            if blocked or preflight_display == "full":
                print("\n" + preview, flush=True)
            else:
                reuse_count = sum(value == "REUSE" for value in decisions.values())
                run_count = sum(value == "RUN" for value in decisions.values())
                print(f"[Target 驗證｜身分核對完成] REUSE {reuse_count} / "
                      f"RUN（需計算）{run_count} / BLOCKED 0", flush=True)
        if blocked:
            details = [f"{target} seed={seed}: {reason}"
                       for (seed, target), reason in blocked.items()]
            raise OracleBlockedError(
                f"Oracle {label} REUSE preflight BLOCKED: "
                f"{len(blocked)} conflict(s); " + " | ".join(details[:3])
            )
        for seed, param_sha, kind, base_params, oracle_params in prepared:
            if not quiet and preflight_display == "full":
                print(f"[Oracle Audit {label}] seed={seed}: verify C96 + {len(targets)} Oracle results")

            def load_or_run(target: str, runner):
                identity, record, origin = completed_plan[(seed, target)]
                if record is not None:
                    try:
                        # AI: An older completed report is promoted to a strict
                        # AI: per-arm cache only after its provenance was checked.
                        if origin.startswith("verified legacy replay cache"):
                            # AI: Recheck the source immediately before promotion;
                            # AI: old completed bytes remain immutable and readable.
                            reread, reread_origin = cache.resolve_completed(identity)
                            if reread_origin != origin or reread != record:
                                raise ValueError("Oracle legacy cache changed during preflight; fail closed")
                            cache.write(identity, record, origin="verified_legacy_replay_cache")
                    except ValueError as exc:
                        raise OracleBlockedError(str(exc)) from exc
                    if not quiet and preflight_display == "full":
                        print(f"[REUSE] {label} seed={seed} {target} | {origin}")
                    return record
                if not quiet:
                    print(f"[RUN] {label} seed={seed} {target} | no completed local result")
                record = _metrics(runner(), target=target)
                record["seed"] = seed
                if target != "C96":
                    record["delta_romd_vs_c96"] = record["romd"] - observations["C96"][-1]["romd"]
                try:
                    cache.write(identity, record, origin="canonical_replay")
                except ValueError as exc:
                    raise OracleBlockedError(str(exc)) from exc
                return record

            baseline = load_or_run("C96", lambda: _run_scenario(
                name=f"C96 {label} seed={seed} same-window", params=base_params,
                param_source_kind=kind, **replay_kwargs,
            ))
            observations["C96"].append(baseline)
            for target, _description in targets:
                def run_oracle(target=target):
                    arm = direct_arm if plan["selectors"].get(target) == "direct" else exact_arm
                    score_map = (daily_seed_scores[seed][plan["daily_sources"][target]]
                                 if target in plan["daily_sources"] else scores[target])
                    if _canonical_score_map_sha256(score_map) != score_fingerprints[seed][target]:
                        raise OracleBlockedError("Oracle injected scores changed after cache preflight")
                    with audit_oracle_truth_score_context(
                        score_map,
                        ranking_policy=str(arm.dl_runtime_mode),
                        ranking_options=arm.dl_runtime_options,
                        key_mode="information_date",
                        missing_key_policy="unavailable_outside_mature_feature_universe",
                        score_start=_formal_dl_score_start(settings),
                    ):
                        return _run_scenario(
                            name=f"Oracle {target} seed={seed}", params=oracle_params,
                            param_source_kind=kind, **replay_kwargs,
                        )
                record = load_or_run(target, run_oracle)
                if not math.isclose(
                    float(record["delta_romd_vs_c96"]), float(record["romd"] - baseline["romd"]),
                    rel_tol=1e-9, abs_tol=1e-9,
                ):
                    raise OracleBlockedError(
                        f"Oracle {label} seed={seed} {target}歷史對照C96不一致，禁止重用"
                    )
                observations[target].append(record)

        baseline_summary = _aggregate(observations["C96"], target="C96")
        rows: list[dict[str, Any]] = []
        for target, _description in targets:
            item = _aggregate(observations[target], target=target)
            diffs = np.array([record["delta_romd_vs_c96"] for record in observations[target]], dtype=float)
            item["delta_romd_vs_c96"] = float(np.mean(diffs))
            item["delta_romd_std"] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            item["positive_seed_count"] = int((diffs > 0.0).sum())
            rows.append(item)
        return {
            "start_date": start, "end_date": end, "common_cohort": cohort,
            "baseline": baseline_summary, "oracles": rows,
            "definitions": definitions,
            "param_artifacts": param_evidence,
            "seed_target_evidence": {},
        }

    base = run_panel(
        end=end40, cohort=cohort40, scores=scores40,
        targets=plan["targets"], label="40D",
    )
    return {
        "schema_version": ORACLE_REPORT_SCHEMA_VERSION,
        "status": "COMPLETED", "read_only": True,
        "hindsight_non_pit": True, "iterative_oos_evidence": True,
        "strategy_profile_id": settings.profile_id,
        "benchmark_id": robustness.benchmark_id,
        "seeds": [seed for seed, _ in parameter_paths],
        "dataset": settings.dataset,
        **base,
        "limitations": [
            "Ex-post future targets are NOT predictions, not PIT-legal, and never a deployable strategy.",
            "CG EqualRank is a fixed 50/50 geometry proxy, not a learned model-head fusion.",
            "Exact and Direct pairs share identical future truth; only canonical K/R0 allocation changes.",
            "Direct means canonical No-K/No-R0, NOT direct raw-R regression or executable prediction.",
            "C96 is the same-seed Min/Broker-Cash control for both selectors; paired deltas use that baseline.",
            "N=4 paired end-to-end parameter seeds; Oracle scores have no trainable seed.",
            "No best-target fitting from iterative OOS; multi-target search is exploratory.",
            "R40 is the future 40th session Close return, not Portfolio Replay exit return.",
            "T40 uses full-path nonnegative MAE and clamps to -1R at 1R adverse movement.",
            "T40-MFE replaces only terminal R40 by full-path MFE40_R; the identical MAE gate remains.",
            "1R is the fixed research Label risk budget, NOT the actual executed ATR stop-loss.",
        ],
    }


def render_result(report: Mapping[str, Any], *, target: str) -> str:
    # Persistent report columns/order/precision are owned by its approved contract.
    contract = report_contract("audit.target_oracle_romd")
    sections = contract.sections
    perf_table = sections[0].tables[0]
    yearly_table = sections[2].tables[0]
    paired_table = sections[3].tables[0]
    baseline = dict(report["baseline"])
    items = [dict(baseline, target="C96 Same-Window", delta_romd_vs_c96=0.0, wins="-"),
             *[dict(item, target=_oracle_display_name(item["target"]), wins=f'{item["positive_seed_count"]}/{item["seeds"]}')
               for item in report["oracles"]]]
    rows = format_comparison_rows(perf_table, items, target=target)
    intro = [
        "Target Ground Truth -> Strategy RoMD (HINDSIGHT ORACLE / NOT PIT)",
        f'Period: {report["start_date"]} ~ {report["end_date"]} (40D-mature common window)',
        f'Common stock-day cohort: {report["common_cohort"]["cohort_rows"]:,}',
        f'Benchmark: {report["benchmark_id"]}, same-seed pairs N={len(report["seeds"])}',
        f'Report schema: {report["schema_version"]}, Oracle treatments: '
        f'{len(report["oracles"])} (40D, Exact + Direct); not Standard SOP',
        'Ranking: Exact K/R0 or Direct No-K/No-R0; same Broker-Cash, seed/P16 and future scores.',
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
                    for key, desc in report.get("definitions", dict(ORACLE_TARGETS)).items()],
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
        for item in report["oracles"]
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
) -> dict[str, Any]:
    # Authorization, frozen schema, and historical snapshot must be checked
    # before initiating the costly hindsight-only portfolio replays.
    approved = require_approved_report_contract("audit.target_oracle_romd")
    payload = dict(build_report(
        definition, project_root=project_root, quiet=quiet,
        preflight_display=preflight_display,
    ))
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
    "SUPPORTED_AUDIT_TYPE", "ORACLE_TARGETS", "ORACLE_DIRECT_SCORE_SOURCES", "OracleBlockedError",
    "collect_status", "build_oracle_score_tables", "render_replay_preflight",
    "render_existing_artifact_inventory",
    "build_report", "render_result", "run_formal_audit",
]
