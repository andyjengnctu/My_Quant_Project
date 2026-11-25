"""Declarative, unfitted same-day score composition for Research replay."""
from __future__ import annotations

import math
from collections.abc import Mapping


METHOD = "same_day_percentile_weighted_v1"


def rank_score_composition(options):
    """Validate the complete frozen transform, including zero-weight cohort inputs."""
    raw = dict(options or {}).get("rank_score_composition")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {"method", "universe", "tie_method", "members"}:
        raise ValueError("Rank composition requires a complete declared contract")
    if (raw["method"] != METHOD or raw["universe"] != "common_score_eligible_stock_days"
            or raw["tie_method"] != "average_zero_based"):
        raise ValueError("Unsupported rank composition semantics")
    members = tuple(raw["members"])
    if not members:
        raise ValueError("Rank composition requires score sources")
    seen = set()
    weights = []
    for item in members:
        if not isinstance(item, Mapping) or set(item) != {"dl_id", "weight"}:
            raise ValueError("Rank composition member requires DL identity and fixed weight")
        source = item["dl_id"]
        weight = item["weight"]
        if (not isinstance(source, str) or not source.strip() or source != source.strip()
                or source in seen or type(weight) not in (int, float)
                or not math.isfinite(weight) or weight < 0):
            raise ValueError("Invalid or duplicate rank composition member")
        seen.add(source)
        weights.append(weight)
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Rank composition weights must sum to one")
    if any(key in dict(options or {}) for key in ("positive_dl_id", "safety_dl_id", "primary_score_column")):
        raise ValueError("Rank composition cannot implicitly combine admission or alternate heads")
    return dict(raw)


def rank_composition_source_ids(options):
    contract = rank_score_composition(options)
    return tuple(item["dl_id"] for item in contract["members"]) if contract else ()
