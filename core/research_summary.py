"""Membership projections for the single current Research Summary roster.

This owner reads no artifacts and does not authorize or implement scientific
execution. Registry owners still validate every declared model/strategy binding.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


def get_research_summary_rows() -> tuple[dict[str, Any], ...]:
    from config import research_summary as cfg

    rows = cfg.RESEARCH_SUMMARY_ROWS
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("Research Summary roster must be a nonempty ordered list")
    # An Oracle binding may inherit an existing, approved perfect-score recipe
    # without replicating Target/selector/provider/description in another arm.
    # Resolve the full roster first: a reference may precede its template.
    oracle_templates: dict[str, Mapping[str, Any]] = {}
    for registered in rows:
        if not isinstance(registered, Mapping):
            raise ValueError("Research Summary row must be a mapping")
        if "oracle" in registered:
            for binding in registered.get("bindings", ()):
                arm_id = binding.get("arm_id")
                if arm_id:
                    if arm_id in oracle_templates:
                        raise ValueError(f"Duplicate Summary Oracle template arm: {arm_id}")
                    oracle_templates[arm_id] = registered
    selected = []
    for raw_row in rows:
        row = dict(raw_row)
        template_arm = row.get("oracle_template_arm_id")
        if template_arm is not None:
            template = oracle_templates.get(template_arm)
            bindings = tuple(row.get("bindings", ()))
            # AI: An inactive historical treatment can still supply its
            # approved immutable score recipe to a *different* parameter
            # family.  Retirement of the original comparison key is enforced
            # by the Oracle registry; this never reactivates the old arm.
            retired_template_same_family = bool(
                template is not None
                and not template.get("enabled", True)
                and str(row.get("parameter_family") or "min") ==
                    str(template["oracle"].get("parameter_family") or "min")
            )
            if (template is None or retired_template_same_family
                    or not template["oracle"].get("enabled", True)
                    or "oracle" in row or "arm_id" in row
                    or len(bindings) != 1 or not bindings[0].get("arm_id")
                    or not isinstance(row.get("parameter_family"), str)
                    or not row["parameter_family"].strip()):
                raise ValueError(f"Invalid Summary Oracle template binding: {template_arm}")
            template_bindings = tuple(b for b in template.get("bindings", ())
                                      if b.get("arm_id") == template_arm)
            if (len(template_bindings) != 1
                    or any(bindings[0].get(role) != template_bindings[0].get(role)
                           for role in ("rank_model", "positive_model"))):
                raise ValueError(f"Summary Oracle template changes supervised models: {template_arm}")
            row["oracle"] = {
                **template["oracle"],
                "strategy_arm_id": bindings[0]["arm_id"],
                "strategy": str(row.get("strategy") or
                                template["oracle"].get("strategy", "Breakout")),
                "parameter_family": row["parameter_family"],
            }
        if not isinstance(row, Mapping):
            raise ValueError("Research Summary row must be a mapping")
        if not any(key in row for key in ("arm_id", "oracle", "bindings")):
            raise ValueError("Research Summary row must declare a Target, model or strategy")
        if "display_target" in row:
            # AI: Summary presentation has orthogonal Target / Strategy /
            # Selector / Model columns. A target override may only be an
            # outcome name, never an arm's composite display name.
            target = row["display_target"]
            if (not isinstance(target, str) or not target.strip()
                    or target != target.strip() or any(token in target for token in (
                        "×", "Daily (", "Breakout (", "No DL", "Default Selector",
                        " Exact", " Direct"))):
                raise ValueError(f"Research Summary Target mixes report dimensions: {target!r}")
        if row.get("enabled", True) and row.get("oracle", {}).get("enabled", True):
            selected.append(dict(row))
    return tuple(selected)


def get_research_summary_oracle_comparisons() -> tuple[dict[str, Any], ...]:
    return tuple(dict(row["oracle"]) for row in get_research_summary_rows() if "oracle" in row)


def get_research_summary_oracle_source_catalog() -> tuple[dict[str, Any], ...]:
    """Immutable approved score producers, including sources with no active treatment.

    A retired Exact/Direct treatment may still supply the original canonical
    truth recipe for an independently active alias (e.g. Positive ranking).
    Only disabled Exact score-producer rows are exposed as source-only entries;
    they cannot become executable comparisons through this catalog.
    """
    from config import research_summary as cfg

    sources = list(get_research_summary_oracle_comparisons())
    needed_sources = {str(row["score_source"]) for row in sources if row.get("score_source")}
    active_sources = {str(row["target"]) for row in sources}
    for group in cfg.RESEARCH_SUMMARY_ROWS:
        if group.get("enabled", True) or "oracle" not in group:
            continue
        oracle = dict(group["oracle"])
        if (oracle.get("target") in needed_sources - active_sources
                and oracle.get("selector") == "exact" and oracle.get("score")
                and oracle.get("provider") and not oracle.get("score_source")):
            sources.append({**oracle, "source_only": True})
    targets = [row["target"] for row in sources]
    if len(targets) != len(set(targets)):
        # The same Target can serve both Daily and Breakout strategies; a
        # score-producer catalog needs each immutable Target only once.
        unique = {}
        for row in sources:
            name = row["target"]
            previous = unique.get(name)
            if previous is not None and {
                key: value for key, value in row.items()
                if key not in ("strategy", "strategy_arm_id", "parameter_family")
            } != {
                key: value for key, value in previous.items()
                if key not in ("strategy", "strategy_arm_id", "parameter_family")
            }:
                raise ValueError(f"Oracle producer catalog conflicting target: {name}")
            unique.setdefault(name, row)
        return tuple(unique.values())
    return tuple(sources)


def get_research_summary_baseline_arm_ids() -> dict[str, str]:
    result = {}
    for row in get_research_summary_rows():
        if "oracle_parameter_family" not in row:
            continue
        family, arm_id = str(row["oracle_parameter_family"]), str(row["arm_id"])
        if not family or not arm_id or family in result:
            raise ValueError("Research Summary Oracle baseline family is empty or duplicated")
        result[family] = arm_id
    return result


def get_research_summary_strategy_arm_ids() -> tuple[str, ...]:
    result = []
    for row in get_research_summary_rows():
        arms = ([row["arm_id"]] if "arm_id" in row else [])
        arms.extend(binding["arm_id"] for binding in row.get("bindings", ()) if binding.get("arm_id"))
        for arm_id in arms:
            if not isinstance(arm_id, str) or not arm_id or arm_id in result:
                raise ValueError(f"Research Summary strategy is empty or duplicated: {arm_id!r}")
            result.append(arm_id)
    return tuple(result)


def summary_binding_models(binding) -> tuple[tuple[str, str], ...]:
    """All supervised dependencies, including a replay-only rank composition."""
    return tuple(tuple(binding[role]) for role in ("rank_model", "positive_model") if role in binding) + tuple(
        tuple(pair) for pair in binding.get("rank_models", ()))


def get_research_summary_model_profiles() -> tuple[tuple[str, str], ...]:
    result = []
    for row in get_research_summary_rows():
        for binding in row.get("bindings", ()):
            for pair in summary_binding_models(binding):
                if len(pair) != 2 or any(not isinstance(value, str) or not value for value in pair):
                    raise ValueError(f"Invalid Summary model: {pair!r}")
                if pair not in result:
                    if any(pair[0] == old[0] or pair[1] == old[1] for old in result):
                        raise ValueError(f"Conflicting Summary model identity: {pair!r}")
                    result.append(pair)
    return tuple(result)



def research_summary_parameter_dimensions(
    arm: Mapping[str, Any], *, parameter_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, bool]:
    """Verified Optimizer parameter label for display only (e.g. Min, P16).

    Derive from the actual param_source's registered training profile and
    canonical parameter_set, not the strategy name or guessed current version.
    This must not participate in Replay/FIT/Optimizer fingerprint construction.
    """
    from config.strategy_params import STRATEGY_PARAM_PROFILES

    source_id = str(arm.get("param_source") or "")
    source = parameter_sources.get(source_id, {})
    profile_id = (source.get("builder") or {}).get("options", {}).get("training_profile_id")
    profile = STRATEGY_PARAM_PROFILES.get(profile_id, {}) if profile_id else {}
    actual_set = str(profile.get("parameter_set_identity") or "")
    registered_set = ((source.get("artifact_contract") or {}).get(
        "breakout_quality_param_adaptation") or {}).get("parameter_set")
    filter_profile = str(profile.get("filter_profile") or "").strip().lower()
    entry = str(arm.get("entry_strategy", "Breakout"))
    is_daily_fitted = profile.get("strategy_id") == "daily"
    if (not filter_profile or not actual_set.startswith("PARAM-")
            or not actual_set[6:] or actual_set != registered_set
            or entry not in {"Breakout", "Daily"}
            or (is_daily_fitted and entry != "Daily")):
        raise ValueError(f"Summary strategy/parameter lineage mismatch: {source_id}")
    return filter_profile.capitalize(), actual_set[6:], is_daily_fitted


def research_summary_target_label(name: str, *, parameter_version: str) -> str:
    """Format execution-path Targets identically with or without an Oracle row.

    The caller supplies the verified Optimizer identity; the scientific name
    and all saved Target/FIT/Replay identities remain unchanged.
    """
    if (any(part in name for part in ("MinTrade", "FullTrade"))
            and any(name.endswith(suffix) for suffix in ("_NetR", "_NetPnL", "_NetPnL_Positive"))):
        return f"{name} ({parameter_version})"
    return name


def research_summary_entry_label(
    arm: Mapping[str, Any], *, family: str = "min",
    parameter_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[str, bool]:
    """Full strategy label (Breakout (Min/P16), Daily (Full/P20), ...).

    The family argument is kept for callers' compatibility; the actual
    Optimizer parameter lineage is the only source of display identity.
    """
    filter_label, version, independently_daily_fitted = research_summary_parameter_dimensions(
        arm, parameter_sources=parameter_sources,
    )
    entry = str(arm.get("entry_strategy", "Breakout"))
    return f"{entry} ({filter_label}/{version})", independently_daily_fitted



def research_summary_selector_label(arm: Mapping[str, Any]) -> str:
    """Selector column from the execution contract, never an arm name.

    No-DL uses the canonical default selection. Unknown scored selectors are
    a contract error, not a silent '-' or display-only exception.
    """
    if not arm.get("dl_enabled"):
        if arm.get("dl_runtime_mode") is not None:
            raise ValueError("No-DL strategy unexpectedly declares a DL selector")
        return "Default"
    modes = {
        "resource-aware-continuous-score-constrained-optimal": "Exact",
        "resource-aware-continuous-score-no-k-no-r0": "Direct",
    }
    mode = arm.get("dl_runtime_mode")
    if mode not in modes:
        raise ValueError(f"Unsupported scored Strategy selector: {mode!r}")
    return modes[mode]


def validate_research_summary_strategy_bindings(
    *, arm_registry: Mapping[str, Mapping[str, Any]],
    dl_sources: Mapping[str, Mapping[str, Any]],
    model_spec_resolver: Callable[[str], Any],
) -> None:
    """Fail on declaration/registry conflicts rather than silently hiding work."""
    # AI: Registry consumers supply their catalogs. The shared membership
    # owner is also used by config and must never import those consumers back.
    from core.positive_score_admission import requires_positive_score, uses_primary_positive_classifier

    for arm_id in get_research_summary_strategy_arm_ids():
        if arm_id not in arm_registry:
            raise ValueError(f"Summary strategy is not registered: {arm_id}")
    for row in get_research_summary_rows():
        if "arm_id" in row and arm_registry[row["arm_id"]].get("dl_enabled"):
            raise ValueError("Summary DL strategies must declare their model bindings")
        for binding in row.get("bindings", ()):
            from core.rank_score_composition import rank_composition_source_ids
            if binding.get("rank_models"):
                if binding.get("rank_model") or binding.get("positive_model") or not binding.get("arm_id"):
                    raise ValueError("Summary composition requires one strategy and explicit constituent models")
                arm = arm_registry[binding["arm_id"]]
                source_ids = rank_composition_source_ids(arm.get("dl_runtime_options"))
                pairs = summary_binding_models(binding)
                if tuple(dl_sources[key]["experiment_profile"] for key in source_ids) != tuple(pair[1] for pair in pairs):
                    raise ValueError("Summary rank composition differs from its runtime sources")
                for model_id, profile in pairs:
                    if model_spec_resolver(profile).model_research_id != model_id:
                        raise ValueError("Summary composition model identity mismatch")
                if arm.get("dl_id") not in source_ids:
                    raise ValueError("Summary composition does not include its primary source")
                continue
            for role in ("rank_model", "positive_model"):
                if role in binding:
                    model_id, profile = binding[role]
                    if model_spec_resolver(profile).model_research_id != model_id:
                        raise ValueError(f"Summary {role} identity differs from the model registry")
            if not binding.get("arm_id"):
                continue
            arm = arm_registry[binding["arm_id"]]
            source = dl_sources.get(arm.get("dl_id"), {})
            if tuple(binding.get("rank_model", ()))[1:] != (source.get("experiment_profile"),):
                raise ValueError(f"Summary rank binding differs from strategy: {binding['arm_id']}")
            options = dict(arm.get("dl_runtime_options") or {})
            from core.candidate_score_permutation import permutation_spec
            repetition = permutation_spec(options)
            if repetition is not None:
                reference = arm_registry.get(repetition["control_arm_id"], {})
                reference_options = {key: value for key, value in options.items()
                                     if key != "candidate_score_permutation"}
                if ("oracle" in row or arm.get("dl_id") != reference.get("dl_id")
                        or arm.get("dl_runtime_mode") != reference.get("dl_runtime_mode")
                        or reference_options != reference.get("dl_runtime_options")):
                    raise ValueError("Permutation must preserve its real score/selector control without Oracle inheritance")
            positive = requires_positive_score(options) and not uses_primary_positive_classifier(options)
            if positive != bool(binding.get("positive_model")):
                raise ValueError(f"Summary Positive dependency differs from strategy: {binding['arm_id']}")
            if positive:
                classifier = dl_sources.get(options["positive_dl_id"], {})
                if binding["positive_model"][1] != classifier.get("experiment_profile"):
                    raise ValueError(f"Summary Positive binding differs from strategy: {binding['arm_id']}")


def unresolved_oracle_strategy_contracts(*, arm_registry):
    """Explicitly declare a new model-output composition awaiting Oracle parity."""
    from core.positive_score_admission import uses_model_positive_decision
    from core.rank_score_composition import rank_composition_source_ids
    from core.candidate_score_permutation import permutation_spec
    result = {}
    for group in get_research_summary_rows():
        if group.get("oracle_contract_status") is None:
            continue
        reason = str(group.get("oracle_contract_reason") or "").strip()
        if group["oracle_contract_status"] != "CONTRACT_UNRESOLVED" or not reason or "oracle" in group:
            raise ValueError("Invalid unresolved Oracle declaration")
        bindings = tuple(group.get("bindings", ()))
        if not bindings:
            raise ValueError("Unresolved Oracle declaration requires a strategy binding")
        for binding in bindings:
            arm_id = binding.get("arm_id")
            if (arm_id not in arm_registry or arm_id in result
                    or not (uses_model_positive_decision(arm_registry[arm_id].get("dl_runtime_options"))
                            or rank_composition_source_ids(arm_registry[arm_id].get("dl_runtime_options"))
                            or permutation_spec(arm_registry[arm_id].get("dl_runtime_options")))):
                raise ValueError("Unresolved Oracle requires a declared model-decision composition")
            result[arm_id] = reason
    return result
