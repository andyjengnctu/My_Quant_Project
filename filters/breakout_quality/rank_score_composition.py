"""Compose verified score tables before the unchanged candidate/selector boundary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.rank_score_composition import rank_score_composition
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets


def compose_rank_score_tables(tables, contract):
    """Intersect feature-eligible stock days, then rank each source on that cohort.

    No target, maturity, orderable-candidate or portfolio state enters this transform.
    Even zero-weight controls require every member, so all contrasts share a cohort.
    """
    contract = rank_score_composition({"rank_score_composition": contract})
    ids = tuple(item["dl_id"] for item in contract["members"])
    if set(tables) != set(ids):
        raise ValueError("Rank composition source set differs from its contract")
    for values in tables.values():
        if (not isinstance(values.index, pd.MultiIndex)
                or tuple(values.index.names) != ("ticker", "date") or not values.index.is_unique
                or not np.isfinite(values.to_numpy(dtype=float)).all()):
            raise ValueError("Rank composition requires unique finite stock-day scores")
    common = tables[ids[0]].index
    for source in ids[1:]:
        common = common.intersection(tables[source].index, sort=False)
    common = common.sort_values()
    if common.empty:
        raise ValueError("Rank composition has no common score-eligible stock days")
    composed = np.zeros(len(common), dtype=np.float64)
    dates = common.get_level_values("date")
    for member in contract["members"]:
        ranks = build_same_date_percentile_targets(
            tables[member["dl_id"]].reindex(common).to_numpy(dtype=float),
            np.ones(len(common), dtype=bool), dates,
        )
        composed += float(member["weight"]) * ranks.astype(np.float64)
    return pd.Series(composed, index=common, name="rank_composition_score")


def load_rank_composition_scores(*, project_root, options, score_bindings):
    """Admit every member through the existing PIT/Forward owner, once per replay."""
    from filters.breakout_quality.ranking_score_store import (
        SCORE_SOURCE_SELECTION_POINT_IN_TIME, SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
        load_selection_point_in_time_ranking_contract,
        load_selection_point_in_time_score_table_from_path,
        load_continuous_ranker_oos_score_table_from_path,
        VerifiedForwardScoreBinding,
    )
    contract = rank_score_composition(options)
    if contract is None:
        return None
    descriptors = options.get("rank_composition_sources", ())
    ids = tuple(item["dl_id"] for item in contract["members"])
    if tuple(item.get("dl_id") for item in descriptors) != ids:
        raise ValueError("Rank composition lacks complete resolved source identities")
    tables, starts, ends, seeds = {}, [], [], set()
    for source in descriptors:
        identity = {key: source[key] for key in ("filter_id", "model_architecture", "experiment_profile")}
        path = Path(source["score_path"]).resolve()
        seed = int(source["expected_seed"])
        if source["score_source"] == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            admitted = load_selection_point_in_time_ranking_contract(
                str(project_root), **identity, require_model_validation_pass=False,
                point_in_time_dir_override=source.get("point_in_time_dir_override"),
            )
            if (int(admitted.seed) != seed or admitted.score_path.resolve() != path
                    or admitted.manifest_path.resolve() != Path(source["manifest_path"]).resolve()):
                raise ValueError("Rank composition PIT seed/path differs from its admitted source")
            table = load_selection_point_in_time_score_table_from_path(
                str(path), manifest_path=str(admitted.manifest_path),
                experiment_profile=identity["experiment_profile"],
            )
            # The canonical table validates score values; keep cutoff legality explicit
            # at this multi-source boundary instead of silently dropping illegal rows.
            if (table["model_information_cutoff"].to_numpy() >= table.index.get_level_values("date")).any():
                raise ValueError("Rank composition contains a non-PIT information cutoff")
        elif source["score_source"] == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
            binding = score_bindings.get(str(path))
            if not isinstance(binding, VerifiedForwardScoreBinding):
                raise ValueError("Rank composition Forward source lacks a verified binding")
            admitted = binding.verify(score_path=path, **identity)
            if int(source["expected_seed"]) != int(admitted.seed):
                raise ValueError("Rank composition Forward seed mismatch")
            table = load_continuous_ranker_oos_score_table_from_path(str(path), identity["experiment_profile"])
        else:
            raise ValueError("Rank composition only consumes admitted Research score sources")
        seeds.add(seed)
        tables[source["dl_id"]] = table[source["score_column"]]
        starts.append(str(admitted.available_from))
        ends.append(str(admitted.available_through))
    if len(seeds) != 1:
        raise ValueError("Rank composition cannot mix benchmark seeds")
    values = compose_rank_score_tables(tables, contract)
    return {"scores": values.to_dict(), "available_from": max(starts), "available_through": min(ends)}
