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
    selected = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Research Summary row must be a mapping")
        if not any(key in row for key in ("arm_id", "oracle", "bindings")):
            raise ValueError("Research Summary row must declare a Target, model or strategy")
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


def get_research_summary_model_profiles() -> tuple[tuple[str, str], ...]:
    result = []
    for row in get_research_summary_rows():
        for binding in row.get("bindings", ()):
            for role in ("rank_model", "positive_model"):
                if role not in binding:
                    continue
                pair = tuple(binding[role])
                if len(pair) != 2 or any(not isinstance(value, str) or not value for value in pair):
                    raise ValueError(f"Invalid Summary {role}: {pair!r}")
                if pair not in result:
                    if any(pair[0] == old[0] or pair[1] == old[1] for old in result):
                        raise ValueError(f"Conflicting Summary model identity: {pair!r}")
                    result.append(pair)
    return tuple(result)


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
            positive = requires_positive_score(options) and not uses_primary_positive_classifier(options)
            if positive != bool(binding.get("positive_model")):
                raise ValueError(f"Summary Positive dependency differs from strategy: {binding['arm_id']}")
            if positive:
                classifier = dl_sources.get(options["positive_dl_id"], {})
                if binding["positive_model"][1] != classifier.get("experiment_profile"):
                    raise ValueError(f"Summary Positive binding differs from strategy: {binding['arm_id']}")
